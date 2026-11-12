"""
storage.py
Lapisan Persistence & Transaction Control (Layer 15).

Fungsi: Menjamin integritas ACID pada SQLite. Mencegah "Half Memory" 
(Fragment tersimpan tapi Payload hilang, atau sebaliknya).
Memisahkan Metadata (Index) dan Payload (Binary) secara fisik di database 
agar pencarian temporal/asosiatif tidak perlu memuat blob besar ke RAM.
"""

import json
import logging
import os  # [FIX] Ditambahkan untuk manipulasi direktori
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, Integer, 
    LargeBinary, JSON, ForeignKey, Index
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
    
    # Indexing untuk Temporal Filtering & Fast Retrieval (Spek Tinggi)
    __table_args__ = (
        Index('idx_fragment_topic', 'topic_key_id'),
        Index('idx_fragment_session', 'session_id'),
        Index('idx_fragment_temporal', 'temporal_index'),
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
            status=key.status
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
            relationship_strength=fragment.relationship_strength
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
            origin_session_id=pending.origin_session_id
        )
        session.merge(db_obj)

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
                    status=k.status
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
                status=k.status
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
                origin_session_id=db_pending.origin_session_id
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
                origin_session_id=db_pending.origin_session_id
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
                    relationship_strength=f.relationship_strength or 0.0
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
                relationship_strength=f.relationship_strength or 0.0
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
                    relationship_strength=f.relationship_strength or 0.0
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
                    relationship_strength=f.relationship_strength or 0.0
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
                    status=k.status
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
                    origin_session_id=p.origin_session_id
                ) for p in db_pendings
            ]

    def get_global_token_counts(self) -> dict:
        """[NEW] Menghitung frekuensi global token di seluruh Topic Key aktif (untuk Unique Token Evidence)."""
        with self.SessionLocal() as session:
            db_keys = session.query(TopicKeyDB).filter_by(status="ACTIVE").all()
            global_counts = {}
            for k in db_keys:
                if k.token_profile:
                    for token, count in k.token_profile.items():
                        global_counts[token] = global_counts.get(token, 0) + count
            return global_counts
