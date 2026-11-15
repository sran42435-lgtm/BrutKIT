"""
storage.py
Lapisan Persistence & Transaction Control (Layer 15).

Fungsi: Menjamin integritas ACID pada SQLite. Mencegah "Half Memory" 
(Fragment tersimpan tapi Payload hilang, atau sebaliknya).
Memisahkan Metadata (Index) dan Payload (Binary) secara fisik di database 
agar pencarian temporal/asosiatif tidak perlu memuat blob besar ke RAM.

[HIERARCHICAL FRAGMENT EXPANSION]
FragmentDB sekarang menyimpan struktur hierarkis:
    parent_fragment_id, depth, fragment_key, fragment_helper_description,
    descendant_helper_evidence, aggregated_helper_description

Semua query hierarchy & trend BERADA DI SINI. memory_engine.py tidak boleh
melakukan SQL langsung.

CHILD COUNT adalah DERIVED value (COUNT(parent_fragment_id == X)),
bukan kolom yang disimpan — sesuai spesifikasi butir 1.3.
"""

import json
import logging
import os  # [FIX] Ditambahkan untuk manipulasi direktori
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Set, Tuple

from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, Integer, 
    LargeBinary, JSON, ForeignKey, Index, Boolean, func, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError

from config import CONFIG
import models # Pydantic models

logger = logging.getLogger(__name__)

# ==========================================
# 1. SQLALCHEMY ORM MODELS (Internal DB Schema)
# ==========================================
Base = declarative_base()

class SessionDB(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    last_active_at = Column(DateTime, nullable=False)


class SessionStateDB(Base):
    """
    [MEKANISME 3] Current Conversational State ("STATE NOW").

    Menyimpan topik & fragment yang sedang aktif dalam satu sesi, sehingga
    kelanjutan lokal tidak perlu selalu membandingkan ulang seluruh kandidat.

    BUKAN probabilitas permanen — jika evidence menunjukkan topik lain,
    engine tetap harus berpindah.
    """
    __tablename__ = "session_states"
    session_id = Column(String, ForeignKey("sessions.id"), primary_key=True)
    active_topic_key_id = Column(String, nullable=True)
    active_fragment_id = Column(String, nullable=True)
    active_depth = Column(Integer, default=0)
    last_updated = Column(DateTime, nullable=True)


class TopicKeyDB(Base):
    __tablename__ = "topic_keys"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    helper_description = Column(String, nullable=False)
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    representative_tokens = Column(JSON, default=list)
    unique_tokens = Column(JSON, default=list)
    token_profile = Column(JSON, default=dict)
    co_occurrence_profile = Column(JSON, default=dict)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)
    activation_count = Column(Integer, default=0)
    related_fragment_ids = Column(JSON, default=list)
    status = Column(String, default="ACTIVE")
    # [MEKANISME 3] Helper propagation dari root fragments.
    # helper_description = identitas OWN. aggregated = warisan descendant.
    aggregated_helper_description = Column(String, default="")
    helper_source_fragment_ids = Column(JSON, default=list)
    
    # Relationship (Lazy loading untuk menghemat memori)
    fragments = relationship("FragmentDB", back_populates="topic_key", lazy="dynamic")

class PayloadDB(Base):
    """Tabel terpisah untuk Binary Blob. Tidak akan ter-load saat query Fragment."""
    __tablename__ = "payloads"
    id = Column(String, primary_key=True)
    compressed_blob = Column(LargeBinary, nullable=False)
    checksum = Column(String, nullable=False)
    compression_algorithm = Column(String, nullable=False)
    original_size = Column(Integer, nullable=False)
    compressed_size = Column(Integer, nullable=False)
    # [NEW] Versi kompresi untuk tracking
    compression_version = Column(String, default="1.0")

class FragmentDB(Base):
    """
    [MEKANISME 3 + 5] Fragment sebagai NODE HIERARKIS.

    parent_fragment_id menunjuk fragment lain (self-referential).
    depth: 0 = root/opening, 1 = section, 2 = child spesifik.

    CATATAN: child_count TIDAK disimpan. Ia dihitung via
    COUNT(parent_fragment_id == X) dengan index idx_fragment_parent.
    """
    __tablename__ = "fragments"
    id = Column(String, primary_key=True)
    topic_key_id = Column(String, ForeignKey("topic_keys.id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    payload_id = Column(String, ForeignKey("payloads.id"), nullable=False)
    created_at = Column(DateTime, nullable=False)
    temporal_index = Column(DateTime, nullable=True)
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    content_hash = Column(String, nullable=True)
    sequence_index = Column(Integer, default=0)
    source_fragment_ids = Column(JSON, default=list)
    relationship_strength = Column(Float, default=0.0)
    # [CRITICAL FIX] True Fragment-Level Relevance Profiles
    token_profile = Column(JSON, default=dict)
    co_occurrence_profile = Column(JSON, default=dict)
    # [MEKANISME 3] Identitas fragment sebagai subtopik.
    fragment_key = Column(String, default="")
    fragment_helper_description = Column(String, default="")
    # [MEKANISME 5] Hierarchy structure.
    parent_fragment_id = Column(String, ForeignKey("fragments.id"), nullable=True)
    depth = Column(Integer, default=0)
    # [MEKANISME 3] Helper propagation (bounded + deduplicated).
    descendant_helper_evidence = Column(JSON, default=list)
    aggregated_helper_description = Column(String, default="")
    helper_contributor_ids = Column(JSON, default=list)
    # [MEKANISME 4] Cross-key composite.
    is_composite = Column(Boolean, default=False)
    source_topic_key_ids = Column(JSON, default=list)
    
    # Indexing untuk Temporal Filtering & Fast Retrieval (Spek Tinggi)
    __table_args__ = (
        Index('idx_fragment_topic', 'topic_key_id'),
        Index('idx_fragment_session', 'session_id'),
        Index('idx_fragment_temporal', 'temporal_index'),
        # [MEKANISME 5] Index hierarchy: query "semua child dari X" harus cepat.
        Index('idx_fragment_parent', 'parent_fragment_id'),
        # Trend retrieval sering butuh "fragment depth tertentu dalam topik".
        Index('idx_fragment_topic_depth', 'topic_key_id', 'depth'),
    )

    topic_key = relationship("TopicKeyDB", back_populates="fragments")
    payload = relationship("PayloadDB", lazy="joined") # Joined hanya jika diminta eksplisit

class GraphEdgeDB(Base):
    __tablename__ = "graph_edges"
    id = Column(String, primary_key=True)
    # [NEW] Renamed dari source_id/target_id agar eksplisit node-based
    source_node_id = Column(String, nullable=False)
    target_node_id = Column(String, nullable=False)
    relationship_type = Column(String, nullable=False)
    weight = Column(Float, default=1.0)
    # [NEW] Field diperkuat
    strength = Column(Float, default=1.0)
    evidence = Column(String, default="")
    created_at = Column(DateTime, nullable=False)

class PendingStateDB(Base):
    __tablename__ = "pending_states"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    original_input = Column(String, nullable=False)
    candidate_keys = Column(JSON, default=list)
    candidate_scores = Column(JSON, default=dict)
    assistant_question = Column(String, nullable=False)
    clarification_round = Column(Integer, default=1)
    created_at = Column(DateTime, nullable=False)
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    last_updated_at = Column(DateTime, nullable=True)
    status = Column(String, default="ACTIVE")
    temporal_reference = Column(String, nullable=True)
    origin_session_id = Column(String, nullable=True)
    # [MEKANISME 1] Recursive ambiguity accumulation.
    # SUMBER KEBENARAN = original_input + clarification_history.
    # combined_clarification_text hanya CACHE/derived.
    clarification_history = Column(JSON, default=list)
    combined_clarification_text = Column(String, default="")
    excluded_candidate_keys = Column(JSON, default=list)
    excluded_tokens = Column(JSON, default=list)
    candidate_history = Column(JSON, default=list)
    
    __table_args__ = (
        Index('idx_pending_status', 'status'),
    )


# ==========================================
# 2. REPOSITORY & TRANSACTION MANAGER
# ==========================================
class MemoryStorage:
    """
    Engine Persistence. Mengurung seluruh interaksi SQLite.
    Menggunakan Context Manager untuk menjamin aturan ACID (No Half-Memory).
    """
    
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_path = CONFIG.storage.SQLITE_DB_PATH
            
            # [FIX] Pastikan direktori induk (misal: 'data/') ada sebelum membuat file DB
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                
            db_url = f"sqlite:///{db_path}"
            
        self.engine = create_engine(
            db_url,
            connect_args={
                "check_same_thread": False, 
                "timeout": CONFIG.storage.DB_TIMEOUT_SECONDS
            },
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Inisialisasi Schema
        Base.metadata.create_all(bind=self.engine)

        # [MEKANISME 3+5] Migrasi aditif untuk DB yang SUDAH ADA.
        # create_all() melewati tabel existing → kolom baru tidak muncul.
        # Ini menutup celah itu tanpa pernah menghapus/mengubah data lama.
        self._ensure_additive_columns()

    # ==========================================
    # [MIGRASI] ADDITIVE-ONLY SCHEMA UPGRADE
    # ==========================================

    def _ensure_additive_columns(self):
        """
        Menambahkan kolom & index yang belum ada pada tabel EXISTING.

        ATURAN KEAMANAN:
            ✅ ALTER TABLE ... ADD COLUMN  (hanya jika kolom belum ada)
            ✅ CREATE INDEX IF NOT EXISTS
            ❌ TIDAK PERNAH DROP COLUMN / DROP TABLE / RENAME / ubah tipe

        Kenapa perlu:
            Base.metadata.create_all() hanya membuat tabel yang BELUM ada.
            Database lama (data/memory.db) tidak akan mendapat kolom hierarchy
            tanpa langkah ini, dan semua query baru akan gagal.

        Kolom JSON ditambahkan sebagai TEXT (SQLite menyimpan JSON sebagai TEXT),
        sehingga kompatibel dengan tipe SQLAlchemy JSON yang sudah dipakai.
        """
        expected_columns: Dict[str, Dict[str, str]] = {
            "fragments": {
                "fragment_key": "VARCHAR DEFAULT ''",
                "fragment_helper_description": "VARCHAR DEFAULT ''",
                "parent_fragment_id": "VARCHAR",
                "depth": "INTEGER DEFAULT 0",
                "descendant_helper_evidence": "TEXT",
                "aggregated_helper_description": "VARCHAR DEFAULT ''",
                "helper_contributor_ids": "TEXT",
                "is_composite": "BOOLEAN DEFAULT 0",
                "source_topic_key_ids": "TEXT",
            },
            "topic_keys": {
                "aggregated_helper_description": "VARCHAR DEFAULT ''",
                "helper_source_fragment_ids": "TEXT",
            },
            "pending_states": {
                "clarification_history": "TEXT",
                "combined_clarification_text": "VARCHAR DEFAULT ''",
                "excluded_candidate_keys": "TEXT",
                "excluded_tokens": "TEXT",
                "candidate_history": "TEXT",
            },
        }

        expected_indexes: Dict[str, Tuple[str, str]] = {
            "idx_fragment_parent": ("fragments", "parent_fragment_id"),
            "idx_fragment_topic_depth": ("fragments", "topic_key_id, depth"),
        }

        try:
            inspector = inspect(self.engine)
            with self.engine.begin() as conn:
                for table_name, columns in expected_columns.items():
                    if not inspector.has_table(table_name):
                        # Tabel baru sudah dibuat create_all dengan kolom lengkap.
                        continue
                    existing = {c["name"] for c in inspector.get_columns(table_name)}
                    for col_name, col_ddl in columns.items():
                        if col_name in existing:
                            continue
                        conn.execute(text(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_ddl}"
                        ))
                        logger.info(
                            f"SCHEMA MIGRATION: {table_name}.{col_name} ditambahkan (additive)."
                        )

                for index_name, (table_name, col_expr) in expected_indexes.items():
                    if not inspector.has_table(table_name):
                        continue
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col_expr})"
                    ))
        except SQLAlchemyError as e:
            # Migrasi gagal tidak boleh membuat engine mati total; tapi harus terlihat.
            logger.error(f"SCHEMA MIGRATION GAGAL (additive): {e}")
            raise

    @contextmanager
    def transaction(self):
        """
        SPEK TINGGI: Fail-Closed Transaction Gate.
        Menjamin jika satu entitas (misal: Payload) gagal disimpan, 
        maka entitas lain (Fragment, Graph) otomatis di-ROLLBACK.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.critical(f"TRANSACTION ROLLBACK: Kegagalan ACID terdeteksi. Error: {e}")
            raise RuntimeError(f"Penyimpanan memori dibatalkan (Rollback): {e}")
        except Exception as e:
            session.rollback()
            logger.critical(f"TRANSACTION ROLLBACK: Error tak terduga. Error: {e}")
            raise
        finally:
            session.close()

    # ==========================================
    # [MAPPER] DB ROW → PYDANTIC (dipakai method hierarchy/trend baru)
    # ==========================================

    def _to_fragment_model(self, f: "FragmentDB") -> models.FragmentMetadata:
        """
        Mapper tunggal untuk method hierarchy/trend baru.

        Method read lama tetap memakai mapping inline mereka (diperluas dengan
        field baru) agar tidak ada kode existing yang dihapus. Method baru
        memakai helper ini supaya tidak menambah belasan duplikasi 25-baris.
        """
        return models.FragmentMetadata(
            fragment_id=f.id,
            topic_key_id=f.topic_key_id,
            session_id=f.session_id,
            created_at=f.created_at,
            temporal_index=f.temporal_index,
            payload_reference=f.payload_id,
            content_hash=f.content_hash,
            sequence_index=f.sequence_index or 0,
            source_fragment_ids=f.source_fragment_ids or [],
            relationship_strength=f.relationship_strength or 0.0,
            token_profile=f.token_profile or {},
            co_occurrence_profile=f.co_occurrence_profile or {},
            fragment_key=f.fragment_key or "",
            fragment_helper_description=f.fragment_helper_description or "",
            parent_fragment_id=f.parent_fragment_id,
            depth=f.depth or 0,
            descendant_helper_evidence=f.descendant_helper_evidence or [],
            aggregated_helper_description=f.aggregated_helper_description or "",
            helper_contributor_ids=f.helper_contributor_ids or [],
            is_composite=bool(f.is_composite),
            source_topic_key_ids=f.source_topic_key_ids or [],
        )

    def _to_topic_model(self, k: "TopicKeyDB") -> models.TopicKey:
        """Mapper tunggal TopicKey untuk method trend baru."""
        return models.TopicKey(
            key_id=k.id,
            title=k.title,
            helper_description=k.helper_description,
            representative_tokens=k.representative_tokens or [],
            unique_tokens=k.unique_tokens or [],
            token_profile=k.token_profile or {},
            co_occurrence_profile=k.co_occurrence_profile or {},
            created_at=k.created_at,
            updated_at=k.updated_at,
            activation_count=k.activation_count or 0,
            related_fragment_ids=k.related_fragment_ids or [],
            status=k.status,
            aggregated_helper_description=k.aggregated_helper_description or "",
            helper_source_fragment_ids=k.helper_source_fragment_ids or [],
        )

    def _to_pending_model(self, p: "PendingStateDB") -> models.PendingState:
        """Mapper tunggal PendingState untuk method clarification baru."""
        return models.PendingState(
            pending_id=p.id,
            session_id=p.session_id,
            original_input=p.original_input,
            candidate_keys=p.candidate_keys or [],
            candidate_scores=p.candidate_scores or {},
            assistant_question=p.assistant_question,
            clarification_round=p.clarification_round or 1,
            created_at=p.created_at,
            last_updated_at=p.last_updated_at,
            status=models.PendingStatus(p.status),
            temporal_reference=p.temporal_reference,
            origin_session_id=p.origin_session_id,
            clarification_history=p.clarification_history or [],
            combined_clarification_text=p.combined_clarification_text or "",
            excluded_candidate_keys=p.excluded_candidate_keys or [],
            excluded_tokens=p.excluded_tokens or [],
            candidate_history=p.candidate_history or [],
        )

    # ==========================================
    # OPERASI PENYIMPANAN (WRITE)
    # ==========================================
    
    def ensure_session_exists(self, session_id: str, created_at: datetime):
        """[NEW] Menjamin entitas Session ada sebelum Fragment disimpan (Lifecycle)."""
        with self.SessionLocal() as session:
            existing = session.query(SessionDB).filter_by(id=session_id).first()
            if not existing:
                new_session = SessionDB(
                    id=session_id,
                    created_at=created_at,
                    last_active_at=created_at
                )
                session.add(new_session)
                session.commit()
            else:
                existing.last_active_at = created_at
                session.commit()

    def save_topic_key(self, session: Session, key: models.TopicKey):
        db_obj = TopicKeyDB(
            id=key.key_id, 
            title=key.title, 
            helper_description=key.helper_description,
            representative_tokens=key.representative_tokens,
            unique_tokens=key.unique_tokens,
            token_profile=key.token_profile, 
            co_occurrence_profile=key.co_occurrence_profile,
            created_at=key.created_at, 
            updated_at=key.updated_at,
            activation_count=key.activation_count,
            related_fragment_ids=key.related_fragment_ids,
            status=key.status,
            # [MEKANISME 3] Helper propagation.
            aggregated_helper_description=key.aggregated_helper_description,
            helper_source_fragment_ids=key.helper_source_fragment_ids,
        )
        session.merge(db_obj) # Merge mencegah error jika Key sudah ada (Update profile)

    def save_payload(self, session: Session, payload: models.CompressedPayload):
        db_obj = PayloadDB(
            id=payload.payload_id, 
            compressed_blob=payload.compressed_blob,
            checksum=payload.checksum, 
            compression_algorithm=payload.compression_algorithm,
            original_size=payload.original_size, 
            compressed_size=payload.compressed_size,
            compression_version="1.0"  # Default version
        )
        session.add(db_obj)

    def save_fragment(self, session: Session, fragment: models.FragmentMetadata):
        db_obj = FragmentDB(
            id=fragment.fragment_id, 
            topic_key_id=fragment.topic_key_id,
            session_id=fragment.session_id, 
            payload_id=fragment.payload_reference,
            created_at=fragment.created_at, 
            temporal_index=fragment.temporal_index,
            content_hash=fragment.content_hash,
            sequence_index=fragment.sequence_index,
            source_fragment_ids=fragment.source_fragment_ids,
            relationship_strength=fragment.relationship_strength,
            token_profile=fragment.token_profile,
            co_occurrence_profile=fragment.co_occurrence_profile,
            # [MEKANISME 3] Identitas subtopik.
            fragment_key=fragment.fragment_key,
            fragment_helper_description=fragment.fragment_helper_description,
            # [MEKANISME 5] Hierarchy.
            parent_fragment_id=fragment.parent_fragment_id,
            depth=fragment.depth,
            # [MEKANISME 3] Helper propagation.
            descendant_helper_evidence=fragment.descendant_helper_evidence,
            aggregated_helper_description=fragment.aggregated_helper_description,
            helper_contributor_ids=fragment.helper_contributor_ids,
            # [MEKANISME 4] Composite.
            is_composite=fragment.is_composite,
            source_topic_key_ids=fragment.source_topic_key_ids,
        )
        session.add(db_obj)

    def save_graph_edge(self, session: Session, edge: models.MemoryGraphEdge):
        db_obj = GraphEdgeDB(
            id=edge.edge_id, 
            source_node_id=edge.source_node_id,  # [NEW] Renamed field
            target_node_id=edge.target_node_id,  # [NEW] Renamed field
            relationship_type=edge.relationship_type.value, 
            weight=edge.weight,
            strength=edge.strength,
            evidence=edge.evidence,
            created_at=edge.created_at
        )
        session.add(db_obj)

    def save_pending_state(self, session: Session, pending: models.PendingState):
        # [CRITICAL FIX] Menggunakan explicit query update untuk menghindari quirk session.merge() 
        # pada objek detached lintas sesi (Cross-Session Pending Resolution).
        existing = session.query(PendingStateDB).filter_by(id=pending.pending_id).first()
        if existing:
            existing.status = pending.status.value
            existing.last_updated_at = pending.last_updated_at
            existing.original_input = pending.original_input
            existing.clarification_round = pending.clarification_round
            existing.candidate_keys = pending.candidate_keys
            existing.candidate_scores = pending.candidate_scores
            # [FIX — Batch 3] assistant_question sebelumnya TIDAK diperbarui di
            # branch update. Mekanisme 1 membutuhkan pertanyaan yang makin
            # spesifik per round, jadi field ini wajib ikut diperbarui.
            existing.assistant_question = pending.assistant_question
            # [MEKANISME 1] Akumulasi klarifikasi.
            existing.clarification_history = pending.clarification_history
            existing.combined_clarification_text = pending.combined_clarification_text
            existing.excluded_candidate_keys = pending.excluded_candidate_keys
            existing.excluded_tokens = pending.excluded_tokens
            existing.candidate_history = pending.candidate_history
        else:
            db_obj = PendingStateDB(
                id=pending.pending_id, 
                session_id=pending.session_id,
                original_input=pending.original_input, 
                candidate_keys=pending.candidate_keys,
                candidate_scores=pending.candidate_scores, 
                assistant_question=pending.assistant_question,
                clarification_round=pending.clarification_round, 
                created_at=pending.created_at,
                last_updated_at=pending.last_updated_at,
                status=pending.status.value, 
                temporal_reference=pending.temporal_reference,
                origin_session_id=pending.origin_session_id,
                # [MEKANISME 1] Akumulasi klarifikasi.
                clarification_history=pending.clarification_history,
                combined_clarification_text=pending.combined_clarification_text,
                excluded_candidate_keys=pending.excluded_candidate_keys,
                excluded_tokens=pending.excluded_tokens,
                candidate_history=pending.candidate_history,
            )
            session.add(db_obj)

    # ==========================================
    # [MEKANISME 3] HELPER PROPAGATION WRITE
    # ==========================================

    def update_fragment_helper(self, session: Session, fragment: models.FragmentMetadata):
        """
        [MEKANISME 3] Update HANYA field helper pada fragment existing.

        KENAPA METHOD TERPISAH:
            FragmentMetadata bersifat frozen. Propagasi membuat instance BARU
            via model_copy(update={...}). Method ini menulis HANYA kolom helper:
                fragment_helper_description
                descendant_helper_evidence
                aggregated_helper_description
                helper_contributor_ids

        YANG TIDAK PERNAH DISENTUH:
            payload_id, content_hash, created_at, temporal_index,
            relationship_strength, parent_fragment_id, depth

        Dengan begitu integritas exact memory & invariant Continuity Purity
        (relationship_strength == turn_continuity_score) tetap terjaga.
        """
        existing = session.query(FragmentDB).filter_by(id=fragment.fragment_id).first()
        if existing is None:
            logger.warning(
                f"HELPER PROPAGATION: fragment {fragment.fragment_id[:8]}... tidak ditemukan."
            )
            return
        existing.fragment_helper_description = fragment.fragment_helper_description
        existing.descendant_helper_evidence = fragment.descendant_helper_evidence
        existing.aggregated_helper_description = fragment.aggregated_helper_description
        existing.helper_contributor_ids = fragment.helper_contributor_ids

    def update_topic_helper(self, session: Session, topic_key: models.TopicKey):
        """
        [MEKANISME 3] Update HANYA helper TopicKey (titik akhir propagasi).

        helper_description boleh diperbarui HANYA jika engine sudah
        memverifikasi mutation gate. Storage tidak membuat keputusan itu.
        """
        existing = session.query(TopicKeyDB).filter_by(id=topic_key.key_id).first()
        if existing is None:
            return
        existing.helper_description = topic_key.helper_description
        existing.aggregated_helper_description = topic_key.aggregated_helper_description
        existing.helper_source_fragment_ids = topic_key.helper_source_fragment_ids
        existing.updated_at = topic_key.updated_at

    # ==========================================
    # [MEKANISME 3] SESSION ACTIVE STATE
    # ==========================================

    def save_session_state(self, state: models.SessionActiveState):
        """Simpan/perbarui active topic & fragment untuk satu sesi."""
        with self.SessionLocal() as session:
            existing = session.query(SessionStateDB).filter_by(
                session_id=state.session_id
            ).first()
            if existing:
                existing.active_topic_key_id = state.active_topic_key_id
                existing.active_fragment_id = state.active_fragment_id
                existing.active_depth = state.active_depth
                existing.last_updated = state.last_updated
            else:
                session.add(SessionStateDB(
                    session_id=state.session_id,
                    active_topic_key_id=state.active_topic_key_id,
                    active_fragment_id=state.active_fragment_id,
                    active_depth=state.active_depth,
                    last_updated=state.last_updated,
                ))
            session.commit()

    def get_session_state(self, session_id: str) -> Optional[models.SessionActiveState]:
        """Ambil active state sesi. None jika belum ada."""
        with self.SessionLocal() as session:
            s = session.query(SessionStateDB).filter_by(session_id=session_id).first()
            if not s:
                return None
            return models.SessionActiveState(
                session_id=s.session_id,
                active_topic_key_id=s.active_topic_key_id,
                active_fragment_id=s.active_fragment_id,
                active_depth=s.active_depth or 0,
                last_updated=s.last_updated,
            )

    # ==========================================
    # OPERASI PEMBACAAN (READ)
    # ==========================================

    def get_all_topic_keys(self) -> List[models.TopicKey]:
        """Mengambil seluruh index Topic Key (Tanpa memuat payload)."""
        with self.SessionLocal() as session:
            db_keys = session.query(TopicKeyDB).filter_by(status="ACTIVE").all()
            return [
                models.TopicKey(
                    key_id=k.id, 
                    title=k.title, 
                    helper_description=k.helper_description,
                    representative_tokens=k.representative_tokens or [],
                    unique_tokens=k.unique_tokens or [],
                    token_profile=k.token_profile or {}, 
                    co_occurrence_profile=k.co_occurrence_profile or {},
                    created_at=k.created_at, 
                    updated_at=k.updated_at,
                    activation_count=k.activation_count or 0,
                    related_fragment_ids=k.related_fragment_ids or [],
                    status=k.status,
                    aggregated_helper_description=k.aggregated_helper_description or "",
                    helper_source_fragment_ids=k.helper_source_fragment_ids or [],
                ) for k in db_keys
            ]

    def get_topic_key(self, key_id: str) -> Optional[models.TopicKey]:
        """[NEW] Mengambil satu Topic Key berdasarkan ID."""
        with self.SessionLocal() as session:
            k = session.query(TopicKeyDB).filter_by(id=key_id, status="ACTIVE").first()
            if not k:
                return None
            return models.TopicKey(
                key_id=k.id, 
                title=k.title, 
                helper_description=k.helper_description,
                representative_tokens=k.representative_tokens or [],
                unique_tokens=k.unique_tokens or [],
                token_profile=k.token_profile or {}, 
                co_occurrence_profile=k.co_occurrence_profile or {},
                created_at=k.created_at, 
                updated_at=k.updated_at,
                activation_count=k.activation_count or 0,
                related_fragment_ids=k.related_fragment_ids or [],
                status=k.status,
                aggregated_helper_description=k.aggregated_helper_description or "",
                helper_source_fragment_ids=k.helper_source_fragment_ids or [],
            )

    def get_active_pending_state(self, session_id: str) -> Optional[models.PendingState]:
        """Mencari Pending State yang masih ACTIVE untuk sesi tertentu."""
        with self.SessionLocal() as session:
            db_pending = session.query(PendingStateDB).filter(
                PendingStateDB.session_id == session_id,
                PendingStateDB.status == models.PendingStatus.ACTIVE.value
            ).order_by(PendingStateDB.created_at.desc()).first()
            
            if not db_pending:
                return None
                
            return models.PendingState(
                pending_id=db_pending.id, 
                session_id=db_pending.session_id,
                original_input=db_pending.original_input, 
                candidate_keys=db_pending.candidate_keys,
                candidate_scores=db_pending.candidate_scores, 
                assistant_question=db_pending.assistant_question,
                clarification_round=db_pending.clarification_round, 
                created_at=db_pending.created_at,
                last_updated_at=db_pending.last_updated_at,
                status=models.PendingStatus(db_pending.status), 
                temporal_reference=db_pending.temporal_reference,
                origin_session_id=db_pending.origin_session_id,
                clarification_history=db_pending.clarification_history or [],
                combined_clarification_text=db_pending.combined_clarification_text or "",
                excluded_candidate_keys=db_pending.excluded_candidate_keys or [],
                excluded_tokens=db_pending.excluded_tokens or [],
                candidate_history=db_pending.candidate_history or [],
            )

    def get_pending_by_reference(self, pending_id: str) -> Optional[models.PendingState]:
        """[NEW] Mengambil Pending State berdasarkan ID-nya."""
        with self.SessionLocal() as session:
            db_pending = session.query(PendingStateDB).filter_by(id=pending_id).first()
            if not db_pending:
                return None
            return models.PendingState(
                pending_id=db_pending.id, 
                session_id=db_pending.session_id,
                original_input=db_pending.original_input, 
                candidate_keys=db_pending.candidate_keys,
                candidate_scores=db_pending.candidate_scores, 
                assistant_question=db_pending.assistant_question,
                clarification_round=db_pending.clarification_round, 
                created_at=db_pending.created_at,
                last_updated_at=db_pending.last_updated_at,
                status=models.PendingStatus(db_pending.status), 
                temporal_reference=db_pending.temporal_reference,
                origin_session_id=db_pending.origin_session_id,
                clarification_history=db_pending.clarification_history or [],
                combined_clarification_text=db_pending.combined_clarification_text or "",
                excluded_candidate_keys=db_pending.excluded_candidate_keys or [],
                excluded_tokens=db_pending.excluded_tokens or [],
                candidate_history=db_pending.candidate_history or [],
            )

    def get_fragment_metadata_by_key(self, topic_key_id: str) -> List[models.FragmentMetadata]:
        """Mengambil metadata fragment berdasarkan Topic Key (Cepat, tanpa blob)."""
        with self.SessionLocal() as session:
            db_frags = session.query(FragmentDB).filter_by(topic_key_id=topic_key_id).all()
            return [
                models.FragmentMetadata(
                    fragment_id=f.id, 
                    topic_key_id=f.topic_key_id, 
                    session_id=f.session_id,
                    created_at=f.created_at, 
                    temporal_index=f.temporal_index, 
                    payload_reference=f.payload_id,
                    content_hash=f.content_hash,
                    sequence_index=f.sequence_index or 0,
                    source_fragment_ids=f.source_fragment_ids or [],
                    relationship_strength=f.relationship_strength or 0.0,
                    token_profile=f.token_profile or {},
                    co_occurrence_profile=f.co_occurrence_profile or {},
                    fragment_key=f.fragment_key or "",
                    fragment_helper_description=f.fragment_helper_description or "",
                    parent_fragment_id=f.parent_fragment_id,
                    depth=f.depth or 0,
                    descendant_helper_evidence=f.descendant_helper_evidence or [],
                    aggregated_helper_description=f.aggregated_helper_description or "",
                    helper_contributor_ids=f.helper_contributor_ids or [],
                    is_composite=bool(f.is_composite),
                    source_topic_key_ids=f.source_topic_key_ids or [],
                ) for f in db_frags
            ]

    def get_fragment(self, fragment_id: str) -> Optional[models.FragmentMetadata]:
        """[NEW] Mengambil satu Fragment berdasarkan ID."""
        with self.SessionLocal() as session:
            f = session.query(FragmentDB).filter_by(id=fragment_id).first()
            if not f:
                return None
            return models.FragmentMetadata(
                fragment_id=f.id, 
                topic_key_id=f.topic_key_id, 
                session_id=f.session_id,
                created_at=f.created_at, 
                temporal_index=f.temporal_index, 
                payload_reference=f.payload_id,
                content_hash=f.content_hash,
                sequence_index=f.sequence_index or 0,
                source_fragment_ids=f.source_fragment_ids or [],
                relationship_strength=f.relationship_strength or 0.0,
                token_profile=f.token_profile or {},
                co_occurrence_profile=f.co_occurrence_profile or {},
                fragment_key=f.fragment_key or "",
                fragment_helper_description=f.fragment_helper_description or "",
                parent_fragment_id=f.parent_fragment_id,
                depth=f.depth or 0,
                descendant_helper_evidence=f.descendant_helper_evidence or [],
                aggregated_helper_description=f.aggregated_helper_description or "",
                helper_contributor_ids=f.helper_contributor_ids or [],
                is_composite=bool(f.is_composite),
                source_topic_key_ids=f.source_topic_key_ids or [],
            )

    def get_recent_fragments(self, session_id: str, limit: int = 10) -> List[models.FragmentMetadata]:
        """[NEW] Mengambil fragment terbaru dari suatu sesi."""
        with self.SessionLocal() as session:
            db_frags = session.query(FragmentDB).filter_by(
                session_id=session_id
            ).order_by(FragmentDB.created_at.desc()).limit(limit).all()
            return [
                models.FragmentMetadata(
                    fragment_id=f.id, 
                    topic_key_id=f.topic_key_id, 
                    session_id=f.session_id,
                    created_at=f.created_at, 
                    temporal_index=f.temporal_index, 
                    payload_reference=f.payload_id,
                    content_hash=f.content_hash,
                    sequence_index=f.sequence_index or 0,
                    source_fragment_ids=f.source_fragment_ids or [],
                    relationship_strength=f.relationship_strength or 0.0,
                    token_profile=f.token_profile or {},
                    co_occurrence_profile=f.co_occurrence_profile or {},
                    fragment_key=f.fragment_key or "",
                    fragment_helper_description=f.fragment_helper_description or "",
                    parent_fragment_id=f.parent_fragment_id,
                    depth=f.depth or 0,
                    descendant_helper_evidence=f.descendant_helper_evidence or [],
                    aggregated_helper_description=f.aggregated_helper_description or "",
                    helper_contributor_ids=f.helper_contributor_ids or [],
                    is_composite=bool(f.is_composite),
                    source_topic_key_ids=f.source_topic_key_ids or [],
                ) for f in db_frags
            ]

    def get_exact_payload(self, payload_id: str) -> Optional[models.CompressedPayload]:
        """Mengambil Binary Blob. Hanya dipanggil jika fragment benar-benar terpilih."""
        with self.SessionLocal() as session:
            db_payload = session.query(PayloadDB).filter_by(id=payload_id).first()
            if not db_payload:
                return None
            return models.CompressedPayload(
                payload_id=db_payload.id, 
                compressed_blob=db_payload.compressed_blob,
                checksum=db_payload.checksum, 
                compression_algorithm=db_payload.compression_algorithm,
                original_size=db_payload.original_size, 
                compressed_size=db_payload.compressed_size
            )

    # ==========================================
    # GRAPH OPERATIONS (NEW)
    # ==========================================

    def get_related_fragments(self, fragment_id: str) -> List[models.FragmentMetadata]:
        """[NEW] Mengambil fragment yang terhubung via graph edges."""
        with self.SessionLocal() as session:
            # Cari edges di mana fragment ini adalah source atau target
            edges = session.query(GraphEdgeDB).filter(
                (GraphEdgeDB.source_node_id == fragment_id) | 
                (GraphEdgeDB.target_node_id == fragment_id)
            ).all()
            
            related_ids = set()
            for e in edges:
                if e.source_node_id != fragment_id:
                    related_ids.add(e.source_node_id)
                if e.target_node_id != fragment_id:
                    related_ids.add(e.target_node_id)
            
            if not related_ids:
                return []
            
            db_frags = session.query(FragmentDB).filter(
                FragmentDB.id.in_(related_ids)
            ).all()
            
            return [
                models.FragmentMetadata(
                    fragment_id=f.id, 
                    topic_key_id=f.topic_key_id, 
                    session_id=f.session_id,
                    created_at=f.created_at, 
                    temporal_index=f.temporal_index, 
                    payload_reference=f.payload_id,
                    content_hash=f.content_hash,
                    sequence_index=f.sequence_index or 0,
                    source_fragment_ids=f.source_fragment_ids or [],
                    relationship_strength=f.relationship_strength or 0.0,
                    token_profile=f.token_profile or {},
                    co_occurrence_profile=f.co_occurrence_profile or {},
                    fragment_key=f.fragment_key or "",
                    fragment_helper_description=f.fragment_helper_description or "",
                    parent_fragment_id=f.parent_fragment_id,
                    depth=f.depth or 0,
                    descendant_helper_evidence=f.descendant_helper_evidence or [],
                    aggregated_helper_description=f.aggregated_helper_description or "",
                    helper_contributor_ids=f.helper_contributor_ids or [],
                    is_composite=bool(f.is_composite),
                    source_topic_key_ids=f.source_topic_key_ids or [],
                ) for f in db_frags
            ]

    def get_related_topics(self, topic_key_id: str) -> List[models.TopicKey]:
        """[NEW] Mengambil Topic Key yang terhubung via graph edges."""
        with self.SessionLocal() as session:
            edges = session.query(GraphEdgeDB).filter(
                (GraphEdgeDB.source_node_id == topic_key_id) | 
                (GraphEdgeDB.target_node_id == topic_key_id)
            ).all()
            
            related_ids = set()
            for e in edges:
                if e.source_node_id != topic_key_id:
                    related_ids.add(e.source_node_id)
                if e.target_node_id != topic_key_id:
                    related_ids.add(e.target_node_id)
            
            if not related_ids:
                return []
            
            db_keys = session.query(TopicKeyDB).filter(
                TopicKeyDB.id.in_(related_ids),
                TopicKeyDB.status == "ACTIVE"
            ).all()
            
            return [
                models.TopicKey(
                    key_id=k.id, 
                    title=k.title, 
                    helper_description=k.helper_description,
                    representative_tokens=k.representative_tokens or [],
                    unique_tokens=k.unique_tokens or [],
                    token_profile=k.token_profile or {}, 
                    co_occurrence_profile=k.co_occurrence_profile or {},
                    created_at=k.created_at, 
                    updated_at=k.updated_at,
                    activation_count=k.activation_count or 0,
                    related_fragment_ids=k.related_fragment_ids or [],
                    status=k.status,
                    aggregated_helper_description=k.aggregated_helper_description or "",
                    helper_source_fragment_ids=k.helper_source_fragment_ids or [],
                ) for k in db_keys
            ]

    # ==========================================
    # GLOBAL OPERATIONS (Review Kedua)
    # ==========================================

    def get_global_active_pendings(self) -> List[models.PendingState]:
        """[NEW] Mengambil semua Pending State ACTIVE lintas sesi untuk Cross-Session Resolution."""
        with self.SessionLocal() as session:
            db_pendings = session.query(PendingStateDB).filter(
                PendingStateDB.status == models.PendingStatus.ACTIVE.value
            ).order_by(PendingStateDB.created_at.desc()).all()
            
            return [
                models.PendingState(
                    pending_id=p.id, session_id=p.session_id,
                    original_input=p.original_input, candidate_keys=p.candidate_keys,
                    candidate_scores=p.candidate_scores, assistant_question=p.assistant_question,
                    clarification_round=p.clarification_round, created_at=p.created_at,
                    last_updated_at=p.last_updated_at,
                    status=models.PendingStatus(p.status), temporal_reference=p.temporal_reference,
                    origin_session_id=p.origin_session_id,
                    clarification_history=p.clarification_history or [],
                    combined_clarification_text=p.combined_clarification_text or "",
                    excluded_candidate_keys=p.excluded_candidate_keys or [],
                    excluded_tokens=p.excluded_tokens or [],
                    candidate_history=p.candidate_history or [],
                ) for p in db_pendings
            ]

    def get_global_token_counts(self) -> dict:
        """
        [CRITICAL FIX] Menghitung Document Frequency (DF) global token.
        Menghitung di berapa banyak Topic Key berbeda sebuah token muncul, 
        BUKAN menjumlahkan total frekuensi kemunculannya. Ini adalah IDF yang benar.
        """
        with self.SessionLocal() as session:
            db_keys = session.query(TopicKeyDB).filter_by(status="ACTIVE").all()
            global_counts = {}
            for k in db_keys:
                if k.token_profile:
                    # Gunakan .keys() untuk mendapatkan token unik di setiap Topic Key
                    for token in k.token_profile.keys():
                        global_counts[token] = global_counts.get(token, 0) + 1
            return global_counts

    # ==========================================
    # [ASSOCIATIVE EVIDENCE LOCKING] GLOBAL STATISTICS
    # ==========================================

    def get_total_topic_key_count(self) -> int:
        """
        Total Topic Key aktif. Dipakai sebagai numerator pada formula rarity:
            rarity(t) = log((TotalKeys+1)/(DF(t)+1)) / log(TotalKeys+1)
        """
        with self.SessionLocal() as session:
            return session.query(TopicKeyDB).filter_by(status="ACTIVE").count()

    def count_topic_keys_containing_token(self, token: str) -> int:
        """
        Document Frequency satu token: di berapa Topic Key berbeda token ini muncul.
        Token yang jarang muncul memiliki nilai identitas lebih tinggi.
        """
        with self.SessionLocal() as session:
            db_keys = session.query(TopicKeyDB).filter_by(status="ACTIVE").all()
            count = 0
            for k in db_keys:
                if k.token_profile and token in k.token_profile:
                    count += 1
            return count

    def get_token_document_frequency(self, token: str) -> int:
        """Alias eksplisit sesuai dokumentasi Associative Evidence Locking."""
        return self.count_topic_keys_containing_token(token)

    def get_active_fragment_ids_for_key(self, topic_key_id: str, limit: int = 5) -> List[str]:
        """
        Fragment terbaru milik satu Topic Key. Dipakai sebagai titik referensi
        untuk menghitung graph evidence pada retrieval.
        """
        with self.SessionLocal() as session:
            db_frags = session.query(FragmentDB).filter_by(
                topic_key_id=topic_key_id
            ).order_by(FragmentDB.created_at.desc()).limit(limit).all()
            return [f.id for f in db_frags]

    # ==========================================
    # [REASONING PURITY — Prioritas #4] REAL FRAGMENT GRAPH EVIDENCE
    # ==========================================

    def _apply_relationship_weight(self, relationship_type: str) -> float:
        """
        [REASONING PURITY] Bobot per tipe hubungan graph.

        CROSS_SESSION_LINK > CONTINUES karena lintas sesi adalah bukti
        asosiatif yang lebih kuat (menembus batas session).

        RELATED_TO / REFERENCES diberi bobot baseline (1.0) — hubungan
        asosiatif lemah tanpa bukti lintas sesi.
        """
        gc = CONFIG.graph
        rt = models.RelationshipType
        if relationship_type == rt.CROSS_SESSION_LINK.value:
            return gc.DIRECT_CROSS_SESSION_WEIGHT
        if relationship_type == rt.CONTINUES.value:
            return gc.DIRECT_CONTINUATION_WEIGHT
        # RELATED_TO / REFERENCES / unknown → baseline
        return 1.0

    def _bfs_graph_evidence(
        self,
        session: Session,
        start_ids: Set[str],
        target_ids: Optional[Set[str]],
        max_depth: int,
    ) -> Tuple[Dict[str, float], Dict[str, dict]]:
        """
        [REASONING PURITY] BFS traversal untuk graph evidence dengan:
        - Depth limit (MAX_GRAPH_PATH_DEPTH)
        - Decay per hop (INDIRECT_GRAPH_DECAY ** hops)
        - Relationship-type weight (CROSS_SESSION_LINK > CONTINUES)
        - Cycle detection via visited set

        [CRITICAL FIX — Batch A]
        Sebelumnya `edges_cache` di-pre-populate dengan SEMUA node IDs:
            edges_cache = {nid: [] for nid in all_nodes}
        sehingga `nodes_to_load = [n for n in frontier if n not in edges_cache]`
        SELALU kosong → query DB tidak pernah dijalankan → semua skor 0.

        Sekarang tracking node yang sudah di-load dipisah ke `loaded_nodes`,
        sehingga query DB benar-benar dijalankan pada frontier pertama.

        Args:
            session: SQLAlchemy session aktif (caller harus manage lifecycle)
            start_ids: set fragment IDs untuk dihitung graph score-nya
            target_ids: set reference fragment IDs (None = pakai strength edge langsung)
            max_depth: maksimum hop traversal

        Returns:
            (scores, path_info)
            scores: Dict[fragment_id, float] — skor graph tertinggi per fragment (0..1)
            path_info: Dict[fragment_id, {best_depth, path_types, edge_count}] — untuk observability
        """
        gc = CONFIG.graph

        # [FIX] Cache dibangun LAZY dari DB, bukan pre-populated.
        edges_cache: Dict[str, List[Tuple[str, float, str]]] = {}
        loaded_nodes: Set[str] = set()

        # Iteratif load edges per level BFS (lazy expansion).
        frontier: Set[str] = set(start_ids)

        for depth in range(max_depth + 1):
            if not frontier:
                break

            # [FIX] Gunakan loaded_nodes, bukan edges_cache, untuk menentukan
            # node mana yang belum di-query dari DB.
            nodes_to_load = [n for n in frontier if n not in loaded_nodes]

            if nodes_to_load:
                db_edges = session.query(GraphEdgeDB).filter(
                    (GraphEdgeDB.source_node_id.in_(nodes_to_load)) |
                    (GraphEdgeDB.target_node_id.in_(nodes_to_load))
                ).all()

                # Tandai semua node yang di-query sebagai loaded,
                # termasuk yang tidak punya edge sama sekali.
                for n in nodes_to_load:
                    edges_cache.setdefault(n, [])
                    loaded_nodes.add(n)

                for e in db_edges:
                    strength = max(0.0, min(1.0, e.strength if e.strength is not None else e.weight))
                    rel_type = e.relationship_type
                    weighted_strength = strength * self._apply_relationship_weight(rel_type)

                    edges_cache.setdefault(e.source_node_id, []).append(
                        (e.target_node_id, weighted_strength, rel_type)
                    )
                    edges_cache.setdefault(e.target_node_id, []).append(
                        (e.source_node_id, weighted_strength, rel_type)
                    )

            # Frontier berikutnya: neighbor yang belum di-load dari DB.
            next_frontier: Set[str] = set()
            for node in frontier:
                for (other, _, _) in edges_cache.get(node, []):
                    if other not in loaded_nodes:
                        next_frontier.add(other)
            frontier = next_frontier

        # Sekarang hitung skor per fragment dengan BFS dari tiap start_id
        scores: Dict[str, float] = {fid: 0.0 for fid in start_ids}
        path_info: Dict[str, dict] = {
            fid: {"best_depth": -1, "path_types": [], "edge_count": 0}
            for fid in start_ids
        }

        for start in start_ids:
            # BFS dari start, cari path ke target_ids (jika ada) atau semua reachable
            queue: deque = deque()
            queue.append((start, 0, []))  # (current_node, depth, path_types_so_far)
            visited: Set[str] = {start}

            while queue:
                node, depth, path_types = queue.popleft()

                # Untuk setiap edge dari node ini
                for (neighbor, weighted_strength, rel_type) in edges_cache.get(node, []):
                    if depth + 1 > max_depth:
                        continue
                    if neighbor in visited:
                        continue

                    new_path_types = path_types + [rel_type]
                    new_depth = depth + 1

                    # Decay: score = weighted_strength * (INDIRECT_GRAPH_DECAY ** depth)
                    # depth = 0 untuk edge pertama (direct) → decay 1.0
                    decay_factor = gc.INDIRECT_GRAPH_DECAY ** depth
                    # [FIX] Clamp ke [0, 1] agar CROSS_SESSION weight 1.2 tidak
                    # menghasilkan skor > 1.0 (kontrak: graph_score dalam 0..1).
                    candidate_score = max(0.0, min(1.0, weighted_strength * decay_factor))

                    # Apakah neighbor ada di target?
                    is_target_hit = (
                        (target_ids is None)
                        or (neighbor in target_ids)
                    )

                    if is_target_hit:
                        # Update skor fragment ini jika lebih tinggi
                        if candidate_score > scores[start]:
                            scores[start] = candidate_score
                            path_info[start] = {
                                "best_depth": new_depth,
                                "path_types": list(set(new_path_types)),
                                "edge_count": len(new_path_types),
                            }
                        elif candidate_score == scores[start] and scores[start] > 0:
                            # Merge path types jika skor sama
                            merged = set(path_info[start]["path_types"]) | set(new_path_types)
                            path_info[start]["path_types"] = list(merged)

                    # Lanjutkan traversal jika belum max_depth
                    if new_depth < max_depth:
                        visited.add(neighbor)
                        queue.append((neighbor, new_depth, new_path_types))

        # Floor: skor di bawah FRAGMENT_GRAPH_MIN_SCORE dianggap 0
        for fid in scores:
            if scores[fid] < gc.FRAGMENT_GRAPH_MIN_SCORE:
                scores[fid] = 0.0
                path_info[fid] = {"best_depth": -1, "path_types": [], "edge_count": 0}

        return scores, path_info

    def get_fragment_graph_evidence(
        self,
        fragment_ids: List[str],
        reference_fragment_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        [REASONING PURITY — Prioritas #4] Graph sebagai EVIDENCE, bukan absolute truth.

        Mengembalikan skor graph (0..1) per fragment, dihitung melalui
        BFS traversal hingga MAX_GRAPH_PATH_DEPTH hop dengan decay per hop.

        Skor dihitung dari:
            edge_strength
            × relationship_type_weight (CROSS_SESSION_LINK > CONTINUES)
            × (INDIRECT_GRAPH_DECAY ** hops)

        Signature backward-compatible dengan versi 1-hop sebelumnya.

        Args:
            fragment_ids: fragment IDs yang akan diberi skor
            reference_fragment_ids: fragment referensi (target traversal).
                Jika None, skor = kekuatan edge terbaik milik fragment itu sendiri.

        Returns:
            Dict[fragment_id, float] — skor graph per fragment
        """
        if not fragment_ids:
            return {}

        gc = CONFIG.graph
        max_depth = gc.MAX_GRAPH_PATH_DEPTH

        with self.SessionLocal() as session:
            start_ids = set(fragment_ids)
            target_ids = set(reference_fragment_ids) if reference_fragment_ids else None

            # Jika reference adalah subset dari start_ids, kita tetap bisa traverse
            # asalkan tidak menghitung self-loop (start == target tanpa hop).
            # Logic di _bfs_graph_evidence sudah handle ini via visited set.

            scores, _ = self._bfs_graph_evidence(
                session, start_ids, target_ids, max_depth
            )
            return scores

    def get_fragment_graph_paths(
        self,
        fragment_ids: List[str],
        reference_fragment_ids: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """
        [REASONING PURITY — Observability]

        Mengembalikan info path graph per fragment untuk debugging / audit.

        Returns:
            Dict[fragment_id, {
                "best_depth": int,        # jumlah hop dari fragment ke reference
                "path_types": list[str],  # tipe edge yang dilalui
                "edge_count": int,        # jumlah edge di path terbaik
            }]

        fragment tanpa koneksi: {"best_depth": -1, "path_types": [], "edge_count": 0}
        """
        if not fragment_ids:
            return {}

        gc = CONFIG.graph
        max_depth = gc.MAX_GRAPH_PATH_DEPTH

        with self.SessionLocal() as session:
            start_ids = set(fragment_ids)
            target_ids = set(reference_fragment_ids) if reference_fragment_ids else None

            _, path_info = self._bfs_graph_evidence(
                session, start_ids, target_ids, max_depth
            )
            return path_info

    # ==========================================
    # [MEKANISME 5] HIERARCHY TRAVERSAL API
    # ==========================================

    def get_root_fragments_for_topic(self, topic_key_id: str) -> List[models.FragmentMetadata]:
        """
        Fragment root/opening dari satu TopicKey (parent_fragment_id IS NULL).

        Dipakai untuk:
            - membandingkan fragment baru dengan konteks pembuka
            - titik awal propagasi helper ke TopicKey
        """
        with self.SessionLocal() as session:
            rows = session.query(FragmentDB).filter(
                FragmentDB.topic_key_id == topic_key_id,
                FragmentDB.parent_fragment_id.is_(None),
            ).order_by(FragmentDB.created_at.asc()).all()
            return [self._to_fragment_model(f) for f in rows]

    def get_child_fragments(
        self,
        parent_fragment_id: str,
        limit: Optional[int] = None,
    ) -> List[models.FragmentMetadata]:
        """Semua child langsung dari satu fragment (terindex idx_fragment_parent)."""
        with self.SessionLocal() as session:
            q = session.query(FragmentDB).filter(
                FragmentDB.parent_fragment_id == parent_fragment_id
            ).order_by(FragmentDB.created_at.asc())
            if limit is not None:
                q = q.limit(limit)
            return [self._to_fragment_model(f) for f in q.all()]

    def get_fragment_children_count(self, parent_fragment_id: str) -> int:
        """
        [SPESIFIKASI 1.3] child_count adalah DERIVED, bukan kolom tersimpan.

            child_count = COUNT(parent_fragment_id == fragment_id)
        """
        with self.SessionLocal() as session:
            return session.query(func.count(FragmentDB.id)).filter(
                FragmentDB.parent_fragment_id == parent_fragment_id
            ).scalar() or 0

    def get_fragment_children_counts(self, parent_ids: List[str]) -> Dict[str, int]:
        """
        Batch child count untuk trend ranking (satu query, bukan N query).

        Node tanpa child tetap muncul dengan nilai 0 agar caller tidak
        perlu membedakan "tidak ada" vs "nol".
        """
        if not parent_ids:
            return {}
        with self.SessionLocal() as session:
            rows = session.query(
                FragmentDB.parent_fragment_id,
                func.count(FragmentDB.id),
            ).filter(
                FragmentDB.parent_fragment_id.in_(parent_ids)
            ).group_by(FragmentDB.parent_fragment_id).all()
            counts = {pid: 0 for pid in parent_ids}
            for pid, cnt in rows:
                counts[pid] = int(cnt or 0)
            return counts

    def get_fragment_ancestors(
        self,
        fragment_id: str,
        max_depth: Optional[int] = None,
    ) -> List[models.FragmentMetadata]:
        """
        Rantai parent ke ATAS, dimulai dari parent langsung.

        Cycle-safe via visited set. Dipakai Helper Propagation untuk
        meneruskan representasi ringkas sampai root.
        """
        limit = max_depth if max_depth is not None else CONFIG.hierarchy.MAX_ANCESTOR_TRAVERSAL
        ancestors: List[models.FragmentMetadata] = []
        visited: Set[str] = {fragment_id}

        with self.SessionLocal() as session:
            current = session.query(FragmentDB).filter_by(id=fragment_id).first()
            hops = 0
            while current is not None and current.parent_fragment_id and hops < limit:
                pid = current.parent_fragment_id
                if pid in visited:
                    break  # proteksi siklus
                visited.add(pid)
                parent = session.query(FragmentDB).filter_by(id=pid).first()
                if parent is None:
                    break
                ancestors.append(self._to_fragment_model(parent))
                current = parent
                hops += 1
        return ancestors

    def get_fragment_descendants(
        self,
        fragment_id: str,
        max_depth: Optional[int] = None,
    ) -> List[models.FragmentMetadata]:
        """
        Semua descendant (BFS ke BAWAH). Cycle-safe.

        Dipakai untuk audit struktur & agregasi bukti cabang.
        """
        limit = max_depth if max_depth is not None else CONFIG.hierarchy.MAX_FRAGMENT_DEPTH
        result: List[models.FragmentMetadata] = []
        visited: Set[str] = {fragment_id}

        with self.SessionLocal() as session:
            queue: deque = deque([(fragment_id, 0)])
            while queue:
                node_id, depth = queue.popleft()
                if depth >= limit:
                    continue
                children = session.query(FragmentDB).filter(
                    FragmentDB.parent_fragment_id == node_id
                ).all()
                for c in children:
                    if c.id in visited:
                        continue
                    visited.add(c.id)
                    result.append(self._to_fragment_model(c))
                    queue.append((c.id, depth + 1))
        return result

    def get_fragments_by_topic_and_depth(
        self,
        topic_key_id: str,
        depth: int,
    ) -> List[models.FragmentMetadata]:
        """
        Fragment pada depth tertentu dalam satu TopicKey.

        Terindex idx_fragment_topic_depth. Dipakai placement analysis
        (bandingkan dengan section depth 1) dan trend navigation.
        """
        with self.SessionLocal() as session:
            rows = session.query(FragmentDB).filter(
                FragmentDB.topic_key_id == topic_key_id,
                FragmentDB.depth == depth,
            ).order_by(FragmentDB.created_at.asc()).all()
            return [self._to_fragment_model(f) for f in rows]

    def get_fragment_depth(self, fragment_id: str) -> Optional[int]:
        """Depth satu fragment. None jika fragment tidak ada."""
        with self.SessionLocal() as session:
            row = session.query(FragmentDB.depth).filter_by(id=fragment_id).first()
            if row is None:
                return None
            return int(row[0] or 0)

    def count_fragments_for_topic(self, topic_key_id: str) -> int:
        """Total fragment dalam satu TopicKey (sinyal trend level topik)."""
        with self.SessionLocal() as session:
            return session.query(func.count(FragmentDB.id)).filter(
                FragmentDB.topic_key_id == topic_key_id
            ).scalar() or 0

    # ==========================================
    # [MEKANISME 5] TREND RETRIEVAL
    # ==========================================

    def _normalize_trend_count(self, count_signal: int) -> float:
        """Normalisasi count agar 500 child tidak mendominasi tak terbatas."""
        cap = max(1, CONFIG.trend.TREND_COUNT_NORMALIZATION_CAP)
        return min(count_signal, cap) / float(cap)

    def _compute_trend_score(
        self,
        count_signal: int,
        recency_score: float,
        temporal_score: float,
    ) -> float:
        """
        Skor trend. SINYAL UTAMA = count (anak/fragment).

        CHILD_COUNT_WEIGHT (0.60) > RECENCY + TEMPORAL (0.40),
        mengunci aturan "recency hanya bukti tambahan".
        """
        tc = CONFIG.trend
        return (
            tc.TREND_CHILD_COUNT_WEIGHT * self._normalize_trend_count(count_signal)
            + tc.TREND_RECENCY_WEIGHT * max(0.0, min(1.0, recency_score))
            + tc.TREND_TEMPORAL_WEIGHT * max(0.0, min(1.0, temporal_score))
        )

    def _compute_recency_score(
        self,
        latest: Optional[datetime],
        reference_time: Optional[datetime],
    ) -> float:
        """Recency 0..1 dengan decay per hari. Normalisasi timezone naive dari SQLite."""
        if latest is None or reference_time is None:
            return 0.0
        if latest.tzinfo is None and reference_time.tzinfo is not None:
            latest = latest.replace(tzinfo=reference_time.tzinfo)
        age_days = max(0.0, (reference_time - latest).total_seconds() / 86400.0)
        decay_days = max(1.0, CONFIG.trend.TREND_RECENCY_DECAY_DAYS)
        return max(0.0, 1.0 - (age_days / decay_days))

    def get_trending_topic_keys(
        self,
        limit: Optional[int] = None,
        reference_time: Optional[datetime] = None,
        temporal_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[models.TrendCandidate]:
        """
        [MEKANISME 5A] Top-N TopicKey untuk continuation tanpa key spesifik.

        Contoh: "Lanjutkan topik sebelumnya."

        Ranking:
            fragment_count + branch richness   (SINYAL UTAMA)
            + recency                          (tambahan)
            + temporal match jika mode temporal aktif (tambahan)

        Semua query agregat dijalankan di SQL, bukan loop Python per topik.
        """
        top_n = limit if limit is not None else CONFIG.trend.TREND_TOPIC_LIMIT

        with self.SessionLocal() as session:
            keys = session.query(TopicKeyDB).filter_by(status="ACTIVE").all()
            if not keys:
                return []

            # Agregat: jumlah fragment + aktivitas terakhir per topik.
            frag_rows = session.query(
                FragmentDB.topic_key_id,
                func.count(FragmentDB.id),
                func.max(FragmentDB.created_at),
            ).group_by(FragmentDB.topic_key_id).all()
            frag_count = {tid: int(cnt or 0) for tid, cnt, _ in frag_rows}
            last_activity = {tid: ts for tid, _, ts in frag_rows}

            # Branch richness: berapa fragment yang PUNYA parent (cabang hidup).
            branch_rows = session.query(
                FragmentDB.topic_key_id,
                func.count(FragmentDB.id),
            ).filter(
                FragmentDB.parent_fragment_id.isnot(None)
            ).group_by(FragmentDB.topic_key_id).all()
            branch_count = {tid: int(cnt or 0) for tid, cnt in branch_rows}

            # Temporal match per topik (hanya jika ada constraint waktu).
            temporal_hits: Dict[str, int] = {}
            if temporal_range is not None:
                start_t, end_t = temporal_range
                temp_rows = session.query(
                    FragmentDB.topic_key_id,
                    func.count(FragmentDB.id),
                ).filter(
                    FragmentDB.temporal_index >= start_t.replace(tzinfo=None),
                    FragmentDB.temporal_index <= end_t.replace(tzinfo=None),
                ).group_by(FragmentDB.topic_key_id).all()
                temporal_hits = {tid: int(cnt or 0) for tid, cnt in temp_rows}

            candidates: List[models.TrendCandidate] = []
            for k in keys:
                fc = frag_count.get(k.id, 0)
                bc = branch_count.get(k.id, 0)
                # Sinyal utama: total fragment + kekayaan cabang.
                count_signal = fc + bc
                recency = self._compute_recency_score(
                    last_activity.get(k.id) or k.updated_at or k.created_at,
                    reference_time,
                )
                temporal = 1.0 if temporal_hits.get(k.id, 0) > 0 else 0.0
                score = self._compute_trend_score(count_signal, recency, temporal)

                candidates.append(models.TrendCandidate(
                    node_id=k.id,
                    node_type=models.TrendNodeType.TOPIC_KEY,
                    label=k.title,
                    helper_description=(
                        k.aggregated_helper_description or k.helper_description or ""
                    ),
                    depth=0,
                    parent_fragment_id=None,
                    child_count=bc,
                    fragment_count=fc,
                    recency_score=recency,
                    temporal_score=temporal,
                    evidence_score=self._normalize_trend_count(count_signal),
                    trend_score=score,
                    # TopicKey selalu bisa di-expand ke fragment-nya.
                    expandable=True,
                ))

            candidates.sort(key=lambda c: c.trend_score, reverse=True)
            return candidates[:top_n]

    def get_trending_fragments(
        self,
        topic_key_id: str,
        depth: Optional[int] = None,
        limit: Optional[int] = None,
        reference_time: Optional[datetime] = None,
        temporal_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[models.TrendCandidate]:
        """
        [MEKANISME 5B] Top-N fragment pada depth tertentu dalam satu TopicKey.

        Contoh: "Lanjutkan bisnis kopi." → topik jelas, bagian belum.
        Default depth = NEW_BRANCH_DEPTH (1) = section utama.

        Ranking: jumlah child (SINYAL UTAMA) + recency + temporal.

        expandable = depth < MAX_FRAGMENT_DEPTH.
        Setelah user memilih node di depth maksimum, tidak ada expansion lagi.
        """
        target_depth = depth if depth is not None else CONFIG.hierarchy.NEW_BRANCH_DEPTH
        top_n = limit if limit is not None else CONFIG.trend.TREND_FRAGMENT_LIMIT

        with self.SessionLocal() as session:
            rows = session.query(FragmentDB).filter(
                FragmentDB.topic_key_id == topic_key_id,
                FragmentDB.depth == target_depth,
            ).all()
            if not rows:
                return []

            ids = [r.id for r in rows]
            child_counts = self._get_children_counts_in_session(session, ids)

            temporal_hits: Set[str] = set()
            if temporal_range is not None:
                start_t, end_t = temporal_range
                temp_rows = session.query(FragmentDB.id).filter(
                    FragmentDB.id.in_(ids),
                    FragmentDB.temporal_index >= start_t.replace(tzinfo=None),
                    FragmentDB.temporal_index <= end_t.replace(tzinfo=None),
                ).all()
                temporal_hits = {r[0] for r in temp_rows}

            candidates: List[models.TrendCandidate] = []
            for r in rows:
                cc = child_counts.get(r.id, 0)
                recency = self._compute_recency_score(r.created_at, reference_time)
                temporal = 1.0 if r.id in temporal_hits else 0.0
                score = self._compute_trend_score(cc, recency, temporal)
                candidates.append(models.TrendCandidate(
                    node_id=r.id,
                    node_type=models.TrendNodeType.FRAGMENT,
                    label=r.fragment_key or (r.fragment_helper_description or "")[:60],
                    helper_description=(
                        r.aggregated_helper_description
                        or r.fragment_helper_description
                        or ""
                    ),
                    depth=r.depth or 0,
                    parent_fragment_id=r.parent_fragment_id,
                    child_count=cc,
                    fragment_count=cc,
                    recency_score=recency,
                    temporal_score=temporal,
                    evidence_score=self._normalize_trend_count(cc),
                    trend_score=score,
                    expandable=(r.depth or 0) < CONFIG.hierarchy.MAX_FRAGMENT_DEPTH,
                ))

            candidates.sort(key=lambda c: c.trend_score, reverse=True)
            return candidates[:top_n]

    def get_trending_children(
        self,
        parent_fragment_id: str,
        limit: Optional[int] = None,
        reference_time: Optional[datetime] = None,
        temporal_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[models.TrendCandidate]:
        """
        [MEKANISME 5C] Trend expansion ke child dari node yang dipilih user.

        ATURAN DEPTH (DIKUNCI):
            Expansion HANYA dilakukan jika current_depth < MAX_FRAGMENT_DEPTH.
            Jika parent sudah di depth maksimum → return [] (tidak ada level berikut).
        """
        top_n = limit if limit is not None else CONFIG.trend.TREND_FRAGMENT_LIMIT

        with self.SessionLocal() as session:
            parent = session.query(FragmentDB).filter_by(id=parent_fragment_id).first()
            if parent is None:
                return []

            parent_depth = parent.depth or 0
            if CONFIG.trend.ENABLE_DEPTH_AWARE_EXPANSION:
                if parent_depth >= CONFIG.hierarchy.MAX_FRAGMENT_DEPTH:
                    # Depth limit tercapai — jangan cari trend level berikutnya.
                    return []

            rows = session.query(FragmentDB).filter(
                FragmentDB.parent_fragment_id == parent_fragment_id
            ).all()
            if not rows:
                return []

            ids = [r.id for r in rows]
            child_counts = self._get_children_counts_in_session(session, ids)

            temporal_hits: Set[str] = set()
            if temporal_range is not None:
                start_t, end_t = temporal_range
                temp_rows = session.query(FragmentDB.id).filter(
                    FragmentDB.id.in_(ids),
                    FragmentDB.temporal_index >= start_t.replace(tzinfo=None),
                    FragmentDB.temporal_index <= end_t.replace(tzinfo=None),
                ).all()
                temporal_hits = {r[0] for r in temp_rows}

            candidates: List[models.TrendCandidate] = []
            for r in rows:
                cc = child_counts.get(r.id, 0)
                recency = self._compute_recency_score(r.created_at, reference_time)
                temporal = 1.0 if r.id in temporal_hits else 0.0
                score = self._compute_trend_score(cc, recency, temporal)
                candidates.append(models.TrendCandidate(
                    node_id=r.id,
                    node_type=models.TrendNodeType.FRAGMENT,
                    label=r.fragment_key or (r.fragment_helper_description or "")[:60],
                    helper_description=(
                        r.aggregated_helper_description
                        or r.fragment_helper_description
                        or ""
                    ),
                    depth=r.depth or 0,
                    parent_fragment_id=r.parent_fragment_id,
                    child_count=cc,
                    fragment_count=cc,
                    recency_score=recency,
                    temporal_score=temporal,
                    evidence_score=self._normalize_trend_count(cc),
                    trend_score=score,
                    expandable=(r.depth or 0) < CONFIG.hierarchy.MAX_FRAGMENT_DEPTH,
                ))

            candidates.sort(key=lambda c: c.trend_score, reverse=True)
            return candidates[:top_n]

    def _get_children_counts_in_session(
        self,
        session: Session,
        parent_ids: List[str],
    ) -> Dict[str, int]:
        """Versi in-session dari batch child count (dipakai trend query)."""
        if not parent_ids:
            return {}
        rows = session.query(
            FragmentDB.parent_fragment_id,
            func.count(FragmentDB.id),
        ).filter(
            FragmentDB.parent_fragment_id.in_(parent_ids)
        ).group_by(FragmentDB.parent_fragment_id).all()
        counts = {pid: 0 for pid in parent_ids}
        for pid, cnt in rows:
            counts[pid] = int(cnt or 0)
        return counts
