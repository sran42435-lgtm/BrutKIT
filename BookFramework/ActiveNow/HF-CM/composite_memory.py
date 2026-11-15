"""
composite_memory.py
Cross-Key Composite Fragment Detection (Mekanisme 4).

Tanggung jawab:
- Mendeteksi respons AI yang benar-benar menggunakan evidence dari
  beberapa TopicKey berbeda.
- Menghasilkan CompositeSourceAnalysis dengan daftar source fragments.
- Memvalidasi evidence family requirement (minimal 2 family berbeda).
- Memberikan cross-topic contribution bonus.

Prinsip kunci:
    Ownership tetap SATU primary TopicKey (single foreign key).
    Multi-source direpresentasikan lewat:
        - source_fragment_ids (daftar fragment sumber)
        - source_topic_key_ids (daftar TopicKey sumber)
        - N graph edges (REFERENCES) dari setiap source ke fragment composite.

    BUKAN dual primary TopicKey — itu akan merusak foreign-key ownership.

Kondisi composite (SEMUA harus terpenuhi):
    1. Contribution score >= MIN_SOURCE_CONTRIBUTION_SCORE
    2. Minimal MIN_SOURCE_EVIDENCE_FAMILIES family bukti (bukan token saja)
    3. Minimal MIN_DISTINCT_SOURCES source fragments
    4. Source fragments dari >= 2 TopicKey BERBEDA (cross-key)

Kenapa cross-key disyaratkan:
    Continuation dalam satu topik sudah direpresentasikan oleh CONTINUES.
    Composite khusus untuk hubungan LINTAS topik, agar semua fragment
    bertema "kopi" tidak otomatis saling menjadi source.

Anti-false-composite (Prioritas 26):
    Token overlap lemah + satu family bukti != composite.
    MIN_SOURCE_EVIDENCE_FAMILIES=2 memastikan source didukung oleh
    minimal 2 family (mis. token + helper, atau token + graph).
"""

import logging
import re
from typing import List, Optional, Set

from config import CONFIG
from models import (
    RetrievedMemory, CompositeSourceAnalysis, FragmentSourceContribution,
    EvidenceFamily,
)

logger = logging.getLogger(__name__)


class CompositeMemoryAnalyzer:
    """
    Analyzer untuk deteksi multi-source composite fragments.

    Dipanggil oleh memory_engine.py setelah AI response dihasilkan
    dan fragment-fragment terkait direkonstruksi.

    Dependency:
        - storage: untuk fetch FragmentMetadata dari fragment_id

    Kontrak:
        - detect_composite_sources() adalah PURE analysis function.
        - Tidak melakukan write ke storage.
        - Graph edges dibuat oleh caller (memory_engine._persist_memory).
    """

    def __init__(self, storage):
        self.storage = storage

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def detect_composite_sources(
        self,
        response: str,
        retrieved_memories: List[RetrievedMemory],
        primary_topic_key_id: Optional[str],
    ) -> CompositeSourceAnalysis:
        """
        Analisis apakah response AI adalah composite dari beberapa sumber.

        Args:
            response: AI response text
            retrieved_memories: daftar memory yang direkonstruksi
            primary_topic_key_id: TopicKey utama (untuk cross-topic detection)

        Returns:
            CompositeSourceAnalysis dengan:
                - is_composite: True jika semua kondisi terpenuhi
                - sources: daftar FragmentSourceContribution
                - source_fragment_ids: daftar fragment ID sumber
                - distinct_topic_count: jumlah TopicKey unik
                - reason: penjelasan keputusan
        """
        cs = CONFIG.composite
        analysis = CompositeSourceAnalysis(
            primary_topic_key_id=primary_topic_key_id
        )

        # Guard: tidak ada memory atau response kosong
        if not retrieved_memories or not response:
            analysis.reason = "no retrieved memory or empty response"
            return analysis

        # Tokenize response
        resp_tokens = self._tokenize(response)
        if not resp_tokens:
            analysis.reason = "no meaningful response tokens"
            return analysis

        # Analyze setiap memory sebagai potential source
        sources: List[FragmentSourceContribution] = []
        for mem in retrieved_memories:
            contribution = self._analyze_single_source(
                mem=mem,
                resp_tokens=resp_tokens,
                primary_topic_key_id=primary_topic_key_id,
            )
            if contribution is not None:
                sources.append(contribution)

        # Sort by contribution score descending
        sources.sort(key=lambda s: s.contribution_score, reverse=True)

        # Limit to MAX_SOURCE_FRAGMENTS
        sources = sources[:cs.MAX_SOURCE_FRAGMENTS]

        # Count distinct topics
        distinct_topics = {s.topic_key_id for s in sources}

        # Populate analysis
        analysis.sources = sources
        analysis.source_fragment_ids = [s.fragment_id for s in sources]
        analysis.distinct_topic_count = len(distinct_topics)

        # Final check: composite conditions
        if (len(sources) >= cs.MIN_DISTINCT_SOURCES
                and any(s.is_cross_topic for s in sources)
                and len(distinct_topics) >= 2):
            analysis.is_composite = True
            analysis.reason = (
                f"{len(sources)} sources across {len(distinct_topics)} topics"
            )
        else:
            analysis.is_composite = False
            analysis.source_fragment_ids = []
            analysis.reason = "not cross-key composite"

        return analysis

    # ==========================================
    # SINGLE SOURCE ANALYSIS
    # ==========================================

    def _analyze_single_source(
        self,
        mem: RetrievedMemory,
        resp_tokens: Set[str],
        primary_topic_key_id: Optional[str],
    ) -> Optional[FragmentSourceContribution]:
        """
        Analisis satu RetrievedMemory sebagai potential source.

        Returns FragmentSourceContribution jika lolos threshold, else None.

        Steps:
            1. Fetch fragment dari storage.
            2. Hitung token overlap.
            3. Hitung helper overlap.
            4. Check graph score.
            5. Validate evidence families (minimal 2).
            6. Hitung contribution score (dengan cross-topic bonus).
            7. Check contribution >= MIN_SOURCE_CONTRIBUTION_SCORE.
        """
        cs = CONFIG.composite

        # Fetch fragment detail
        frag = self.storage.get_fragment(mem.fragment_id)
        if frag is None:
            return None

        # Token overlap
        frag_tokens = set(frag.token_profile.keys())
        if not frag_tokens:
            return None
        overlap = resp_tokens.intersection(frag_tokens)
        token_ratio = len(overlap) / len(frag_tokens)
        if token_ratio < cs.MIN_SOURCE_TOKEN_OVERLAP:
            return None

        # Evidence families tracking
        evidence_names: List[str] = ["token"]
        families: Set[EvidenceFamily] = {EvidenceFamily.LEXICAL}

        # Helper overlap
        helper_text = (
            frag.aggregated_helper_description
            or frag.fragment_helper_description
            or ""
        )
        helper_tokens = self._tokenize(helper_text)
        helper_ratio = 0.0
        if helper_tokens:
            helper_ratio = len(resp_tokens.intersection(helper_tokens)) / len(helper_tokens)
            if helper_ratio >= cs.MIN_SOURCE_TOKEN_OVERLAP:
                evidence_names.append("helper")
                families.add(EvidenceFamily.CONTEXTUAL)

        # Graph score
        if mem.fragment_graph_score >= CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD:
            evidence_names.append("graph")
            families.add(EvidenceFamily.RELATIONAL)

        # Validate evidence families (anti-false-composite)
        if len(families) < cs.MIN_SOURCE_EVIDENCE_FAMILIES:
            return None

        # Contribution score
        contribution = max(token_ratio, helper_ratio)
        is_cross = (
            primary_topic_key_id is not None
            and frag.topic_key_id != primary_topic_key_id
        )
        if is_cross:
            contribution = min(1.0, contribution + cs.CROSS_TOPIC_CONTRIBUTION_BONUS)

        # Final threshold
        if contribution < cs.MIN_SOURCE_CONTRIBUTION_SCORE:
            return None

        return FragmentSourceContribution(
            fragment_id=frag.fragment_id,
            topic_key_id=frag.topic_key_id,
            fragment_key=frag.fragment_key,
            contribution_score=contribution,
            supporting_evidence_names=evidence_names,
            is_cross_topic=is_cross,
        )

    # ==========================================
    # TOKENIZATION HELPER
    # ==========================================

    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text menjadi set of meaningful tokens.

        Filter:
            - Panjang >= MIN_TOKEN_LENGTH
            - Lowercase
            - Unique (set)
        """
        if not text:
            return set()
        min_len = CONFIG.signal.MIN_TOKEN_LENGTH
        raw = re.findall(r'\b\w+\b', text.lower())
        return {t for t in raw if len(t) >= min_len}
