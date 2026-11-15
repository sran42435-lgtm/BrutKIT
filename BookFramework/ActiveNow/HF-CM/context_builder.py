"""
context_builder.py
Context Assembly & Budgeting (Layer 8).

Fungsi:
1. Menyusun Working Context yang aman dan terstruktur untuk Model API.
2. Menggabungkan System Prompt + Exact Retrieved Fragments + Current Prompt.
3. Menjalankan Token Budget Enforcer (mencegah context window explosion).
4. Memformat memori masa lalu dengan tag yang jelas (XML-style) agar Model 
   tidak mengalami halusinasi antara "ingatan" dan "instruksi baru".

PRINSIP:
- Strict Budgeting: Memori tidak boleh melanggar batas token.
- Semantic Isolation: Memori lama dan prompt baru harus terisolasi secara visual/struktural.
- Fail-Open: Jika tidak ada memori, percakapan tetap berjalan dengan prompt saat ini.

[ASSOCIATIVE EVIDENCE LOCKING]
Context != sekadar memori terbaru.
Context = memori paling terbukti relevan.

[REASONING PURITY]
Urutan konteks menggunakan evidence murni:
    1. Exact temporal match        (fragment tepat di waktu yang diminta)
    2. Fragment graph score         (dari edge nyata: CONTINUES/CROSS_SESSION_LINK)
    3. Temporal distance score      (decay per jam dari waktu yang diminta)
    4. Relevance score              (token + co-occurrence match)
    5. Recency                      (timestamp sebagai tie-breaker)

[HIERARCHICAL FRAGMENT EXPANSION — MEKANISME 3]
Context tidak lagi terlihat sebagai kumpulan teks acak. Setiap fragmen
dibingkai dengan posisi strukturalnya:

    TOPIC: Bisnis Kopi
    SECTION: Modal Bisnis Kopi
    DEPTH: 1
    [Waktu: ...] [EXACT_TIME | GRAPH(...)]
    EXACT MEMORY:
    <teks persis apa adanya>

ATURAN MUTLAK:
    Hierarchy framing adalah METADATA. Exact payload TIDAK PERNAH
    dipotong, diringkas, atau dimodifikasi. Framing hanya menambah
    baris di ATAS teks, tidak menyentuh teks itu sendiri.
"""

import logging
from typing import List, Dict, Optional, Union

from config import CONFIG
from models import ModelRequest, RetrievedMemory

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Mesin perakit konteks. Menerjemahkan data memori persis (Exact Memory) 
    dan input user menjadi satu kesatuan ModelRequest yang siap dikonsumsi LLM.
    """

    def __init__(self):
        self.max_context_tokens = CONFIG.context.MAX_CONTEXT_TOKENS
        self.max_memory_tokens = CONFIG.context.MAX_MEMORY_TOKENS
        self.max_fragments = CONFIG.context.MAX_FRAGMENTS_IN_CONTEXT
        
        # Heuristik sederhana untuk estimasi token PoC 
        # (Rata-rata 1 token = 4 karakter atau 0.75 kata)
        self.CHARS_PER_TOKEN = 4 

    def build_context(
        self,
        current_user_input: str,
        exact_memories: Union[List[Dict], List[RetrievedMemory]],
        active_topic_title: Optional[str] = None,
        system_instructions: Optional[str] = None
    ) -> ModelRequest:
        """
        Merakit konteks final untuk Model API.

        [EVIDENCE LOCKING] Menerima dict legacy MAUPUN RetrievedMemory terstruktur.
        """
        
        # 1. Siapkan System Context
        sys_prompt = system_instructions or self._get_default_system_prompt()
        
        # [OVERHAUL] Hitung token yang sudah terpakai oleh System dan User Input (Review Kedua)
        sys_tokens = self._estimate_tokens(sys_prompt)
        user_tokens = self._estimate_tokens(current_user_input)
        available_memory_tokens = max(0, self.max_context_tokens - sys_tokens - user_tokens)
        
        # Batasi juga dengan MAX_MEMORY_TOKENS absolut
        effective_memory_budget = min(self.max_memory_tokens, available_memory_tokens)
        
        # 2. Rakit dan Batasi Memori (Memory Budgeting)
        past_memory_text = self._assemble_and_budget_memories(exact_memories, active_topic_title, effective_memory_budget)
        
        # 4. Bungkus dalam ModelRequest
        return ModelRequest(
            system_context=sys_prompt,
            past_memory=past_memory_text,
            current_user_message=current_user_input
        )

    def _normalize_memories(
        self,
        memories: Union[List[Dict], List[RetrievedMemory]]
    ) -> List[Dict]:
        """
        [EVIDENCE LOCKING] Menyeragamkan input menjadi dict agar jalur legacy
        (dict tanpa metadata bukti) dan jalur baru (RetrievedMemory) dapat
        diproses oleh logika yang sama.

        [REASONING PURITY] Mengekstrak field baru untuk context ordering:
            - fragment_graph_score     (dari edge nyata, bukan metadata relationship_strength)
            - temporal_distance_score  (decay per jam, bukan binary 0.5/1.0)
            - temporal_distance_minutes, graph_edge_count, graph_path_types (observability)

        [MEKANISME 3] Mengekstrak field hierarchy untuk context framing:
            - topic_key_title, fragment_key, fragment_helper_description
            - parent_fragment_id, depth
            - source_fragment_ids, is_composite (M4)
        """
        normalized: List[Dict] = []
        for mem in memories:
            if isinstance(mem, RetrievedMemory):
                normalized.append({
                    'fragment_id': mem.fragment_id,
                    'text': mem.text,
                    'timestamp': mem.timestamp,
                    'topic_key_id': mem.topic_key_id,
                    # Legacy fields (backward compat)
                    'relevance_score': mem.relevance_score,
                    'temporal_score': mem.temporal_score,
                    'graph_score': mem.graph_score,
                    'evidence_strength': mem.evidence_strength,
                    'exact_temporal_match': mem.exact_temporal_match,
                    # [REASONING PURITY] Field baru untuk context ordering
                    'fragment_graph_score': mem.fragment_graph_score,
                    'temporal_distance_score': mem.temporal_distance_score,
                    'temporal_distance_minutes': mem.temporal_distance_minutes,
                    'graph_edge_count': mem.graph_edge_count,
                    'graph_path_types': mem.graph_path_types,
                    # [MEKANISME 3] Hierarchy framing metadata
                    'topic_key_title': mem.topic_key_title,
                    'fragment_key': mem.fragment_key,
                    'fragment_helper_description': mem.fragment_helper_description,
                    'parent_fragment_id': mem.parent_fragment_id,
                    'depth': mem.depth,
                    # [MEKANISME 4] Composite source metadata
                    'source_fragment_ids': list(mem.source_fragment_ids or []),
                    'is_composite': mem.is_composite,
                })
            elif isinstance(mem, dict):
                # Dict legacy: backward compat, default ke 0.0 untuk field baru
                normalized_dict = dict(mem)
                # Set default jika field baru belum ada
                normalized_dict.setdefault('fragment_graph_score', mem.get('graph_score', 0.0))
                normalized_dict.setdefault('temporal_distance_score', mem.get('temporal_score', 0.0))
                normalized_dict.setdefault('temporal_distance_minutes', None)
                normalized_dict.setdefault('graph_edge_count', 0)
                normalized_dict.setdefault('graph_path_types', [])
                # [MEKANISME 3] Default hierarchy untuk dict legacy.
                # Jika semua kosong, framing di-skip dan format lama dipertahankan.
                normalized_dict.setdefault('topic_key_title', "")
                normalized_dict.setdefault('fragment_key', "")
                normalized_dict.setdefault('fragment_helper_description', "")
                normalized_dict.setdefault('parent_fragment_id', None)
                normalized_dict.setdefault('depth', 0)
                normalized_dict.setdefault('source_fragment_ids', [])
                normalized_dict.setdefault('is_composite', False)
                normalized.append(normalized_dict)
        return normalized

    def _depth_label(self, depth: int) -> str:
        """
        [MEKANISME 3] Label struktural berdasarkan depth.

        Definisi depth DIKUNCI (sinkron dengan FragmentMetadata):
            depth 0 = root / opening
            depth 1 = section utama
            depth 2 = child spesifik
            depth 3+ = spesialisasi lebih dalam
        """
        if depth <= 0:
            return "OPENING"
        if depth == 1:
            return "SECTION"
        if depth == 2:
            return "SUBSECTION"
        return "DETAIL"

    def _build_hierarchy_frame(
        self,
        mem: Dict,
        fallback_topic_title: Optional[str],
    ) -> str:
        """
        [MEKANISME 3] Bangun header framing hierarkis untuk satu fragmen.

        Contoh output:
            TOPIC: Bisnis Kopi
            SECTION: Modal Bisnis Kopi
            DEPTH: 1

        Untuk composite (M4), ditambah baris SOURCES agar LLM tahu memori ini
        berasal dari beberapa bagian:
            SOURCES: 2 fragment (cross-topic)

        Mengembalikan string kosong jika tidak ada metadata hierarchy sama sekali
        (jalur dict legacy) — sehingga format lama tetap dipertahankan.

        PENTING: fungsi ini TIDAK menerima atau menyentuh exact text.
        Ia hanya menghasilkan baris metadata yang diletakkan DI ATAS teks.
        """
        topic_title = (mem.get('topic_key_title') or fallback_topic_title or "").strip()
        fragment_key = (mem.get('fragment_key') or "").strip()
        depth = mem.get('depth', 0) or 0
        parent_id = mem.get('parent_fragment_id')
        is_composite = bool(mem.get('is_composite'))
        source_ids = mem.get('source_fragment_ids') or []

        # Tidak ada metadata hierarchy → jangan paksa framing.
        if not topic_title and not fragment_key and not is_composite:
            return ""

        lines: List[str] = []

        if topic_title:
            lines.append(f"TOPIC: {topic_title}")

        if fragment_key:
            label = self._depth_label(depth)
            lines.append(f"{label}: {fragment_key}")
            lines.append(f"DEPTH: {depth}")
            if parent_id:
                lines.append(f"PARENT: {str(parent_id)[:8]}")
        elif topic_title:
            # Topik diketahui tapi bagian belum punya identitas.
            lines.append(f"DEPTH: {depth}")

        if is_composite and source_ids:
            lines.append(
                f"SOURCES: {len(source_ids)} fragment (cross-topic composite)"
            )

        return "\n".join(lines)

    def _assemble_and_budget_memories(
        self, 
        memories: Union[List[Dict], List[RetrievedMemory]], 
        topic_title: Optional[str],
        memory_budget: int
    ) -> str:
        """
        Menggabungkan fragmen memori dengan batas token dan jumlah fragmen.
        Menggunakan XML Tags untuk isolasi semantik.

        [MEKANISME 3] Setiap fragmen dibingkai dengan posisi hierarkisnya,
        lalu dipisahkan dari exact payload oleh label "EXACT MEMORY:".
        Exact text dipakai VERBATIM — tidak pernah dipotong atau diringkas.
        """
        if not memories:
            return ""

        memories = self._normalize_memories(memories)
        if not memories:
            return ""

        # [REASONING PURITY] Prioritas Context Ordering berbasis evidence murni:
        #   1. Exact temporal match    (fragment tepat di waktu yang diminta)
        #   2. Fragment graph score    (dari edge nyata: CONTINUES/CROSS_SESSION_LINK)
        #   3. Temporal distance score (decay per jam dari waktu yang diminta)
        #   4. Relevance score         (token + co-occurrence match)
        #   5. Recency                 (timestamp sebagai tie-breaker)
        #
        # Alasan urutan ini:
        #   - Graph evidence > relevance karena edge lintas sesi adalah bukti
        #     asosiatif paling independen dari query saat ini.
        #   - Temporal distance > legacy temporal_score (yang hanya binary).
        #   - Recency jadi tie-breaker terakhir, bukan faktor utama.
        def sort_key(mem):
            return (
                1 if mem.get('exact_temporal_match', False) else 0,
                mem.get('fragment_graph_score', 0.0),
                mem.get('temporal_distance_score', 0.0),
                mem.get('relevance_score', 0.0),
                mem.get('timestamp', ''),
            )
            
        memories = sorted(memories, key=sort_key, reverse=True)

        # Batasi jumlah fragmen berdasarkan config
        memories = memories[:self.max_fragments]
        
        assembled_parts = []
        current_memory_tokens = 0
        
        # Iterasi dari memori yang paling relevan
        for mem in memories:
            text = mem.get('text', '')
            timestamp = mem.get('timestamp', 'Unknown')
            
            # [REASONING PURITY] Sertakan metadata evidence dalam tag (untuk observability)
            # Tidak mengubah struktur XML, hanya menambah komentar untuk debugging.
            evidence_annotations = []
            if mem.get('exact_temporal_match'):
                evidence_annotations.append("EXACT_TIME")
            if mem.get('fragment_graph_score', 0.0) > 0.1:
                edge_count = mem.get('graph_edge_count', 0)
                path_types = mem.get('graph_path_types', [])
                evidence_annotations.append(
                    f"GRAPH(edges={edge_count},types={','.join(path_types) or 'none'})"
                )
            if mem.get('temporal_distance_minutes') is not None:
                evidence_annotations.append(
                    f"Δt={mem['temporal_distance_minutes']:.0f}min"
                )
            
            # Header waktu + anotasi evidence (format lama dipertahankan).
            if evidence_annotations:
                annotation_str = " | ".join(evidence_annotations)
                header_line = f"[Waktu: {timestamp}] [{annotation_str}]"
            else:
                header_line = f"[Waktu: {timestamp}]"

            # [MEKANISME 3] Hierarchy framing di ATAS exact text.
            hierarchy_frame = self._build_hierarchy_frame(mem, topic_title)

            if hierarchy_frame:
                formatted_mem = (
                    f"{hierarchy_frame}\n"
                    f"{header_line}\n"
                    f"EXACT MEMORY:\n"
                    f"{text}"
                )
            else:
                formatted_mem = f"{header_line}\n{text}"
            
            mem_tokens = self._estimate_tokens(formatted_mem)
            
            # [OVERHAUL] Degradasi Budget: Hentikan jika budget habis, jangan buang semua
            if current_memory_tokens + mem_tokens > memory_budget:
                # [MEKANISME 3] Graceful degradation: framing adalah metadata
                # opsional. Jika framing membuat fragmen tidak masuk budget,
                # coba ulang TANPA framing sebelum menyerah.
                # Exact memory tetap tidak boleh dipotong.
                if hierarchy_frame:
                    fallback_mem = f"{header_line}\n{text}"
                    fallback_tokens = self._estimate_tokens(fallback_mem)
                    if current_memory_tokens + fallback_tokens <= memory_budget:
                        logger.info(
                            "Hierarchy framing dilepas untuk menghemat budget; "
                            "exact memory tetap utuh."
                        )
                        assembled_parts.append(fallback_mem)
                        current_memory_tokens += fallback_tokens
                        continue

                # Jika fragment pertama saja sudah melebihi budget, kita tidak bisa memasukkannya
                # (Exact memory tidak boleh dipotong sembarangan / string slicing)
                if not assembled_parts:
                    logger.warning("Fragment tunggal melebihi sisa budget memori. Dilewati.")
                    break
                logger.info("Memory Token Budget tercapai. Fragmen sisa diabaikan.")
                break
                
            assembled_parts.append(formatted_mem)
            current_memory_tokens += mem_tokens

        if not assembled_parts:
            return ""

        # Bungkus dengan XML Tag agar LLM paham ini adalah data historis
        topic_context = f' topic="{topic_title}"' if topic_title else ""
        joined_memories = "\n\n---\n\n".join(assembled_parts)
        
        return (
            f"<past_memories{topic_context}>\n"
            f"{joined_memories}\n"
            f"</past_memories>"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimasi jumlah token berdasarkan panjang karakter."""
        if not text:
            return 0
        return len(text) // self.CHARS_PER_TOKEN + 1

    def _get_default_system_prompt(self) -> str:
        """System prompt default untuk menjaga perilaku Model API."""
        return (
            "Anda adalah asisten AI yang cerdas dan akurat. "
            "Anda memiliki akses ke <past_memories> yang berisi catatan percakapan masa lalu yang PERSIS (bukan tebakan). "
            "Gunakan <past_memories> HANYA jika relevan dengan <current_request>. "
            "Jika <past_memories> tidak menyediakan informasi yang diminta, katakan dengan jujur bahwa Anda tidak memiliki catatan tersebut, "
            "jangan pernah mengarang atau berhalusinasi tentang detail masa lalu. "
            "Jawablah <current_request> secara langsung dan natural."
        )
