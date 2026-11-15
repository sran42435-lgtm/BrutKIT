"""
fragment_hierarchy.py
Fragment Placement & Hierarchy Management (Mekanisme 5).

Tanggung jawab:
- Menentukan di mana fragment baru ditempatkan dalam hierarchy
- Scoring kandidat parent berdasarkan evidence (bukan recency)
- Menegakkan MAX_FRAGMENT_DEPTH sebagai hard boundary
- Menghasilkan FragmentPlacementResult yang bisa diaudit

Prinsip kunci:
  MAX_FRAGMENT_DEPTH = HARD STRUCTURAL BOUNDARY
  PARENT = berdasarkan EVIDENCE, bukan fragment terbaru
  TEMPORAL & RECENCY = DINOLKAN dalam scoring parent
                       (test_m5_e20 mengunci invariant ini)

Placement kinds:
  ROOT                 = fragment pertama di TopicKey baru, depth 0
  CHILD                = parent kuat ditemukan, depth = parent.depth + 1
  SIBLING              = tidak ada parent cukup kuat, branch baru
  DEPTH_CAPPED_ATTACH  = parent terbaik sudah di MAX_DEPTH,
                         attach sebagai sibling di depth yang sama
                         (jangan menciptakan depth+1)

Keterkaitan dengan Batch 2 (fragment_identity.py):
  FragmentIdentity hasil ekstraksi user+AI adalah PRIMARY signal
  untuk parent scoring. retrieval.signal_profile hanya dipakai
  sebagai FALLBACK ketika fragment_identity tidak tersedia
  (misalnya pada test yang memanggil langsung dengan RetrievalResult
  tanpa fragment_identity).
"""

import logging
import re
from typing import List, Optional, Dict, Set

from config import CONFIG
from models import (
    TopicKey, FragmentMetadata, FragmentPlacementResult,
    FragmentPlacementKind, FragmentCandidateScore,
)
from fragment_identity import FragmentIdentity
from retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class FragmentHierarchyManager:
    """
    Manajer penempatan fragment dalam hierarchy.

    Dipakai oleh memory_engine._determine_fragment_placement() ketika
    fragment_identity tersedia. Ini memungkinkan parent selection memakai
    identitas fragment hasil akhir (user + AI combined), bukan hanya
    retrieval signal dari user input.

    Dependency:
      - storage: untuk fetch fragment yang sudah ada di topic
      - retrieval: untuk fallback scoring (jika fragment_identity kosong)

    Kontrak:
      - Method ini TIDAK mengubah state (pure decision function).
      - Semua writes dilakukan oleh caller (memory_engine._persist_memory).
      - MAX_FRAGMENT_DEPTH adalah invariant yang DITEGAKKAN, bukan saran.
    """

    def __init__(self, storage):
        self.storage = storage

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def determine_placement(
        self,
        topic_key_id: str,
        retrieval: RetrievalResult,
        is_new_topic: bool,
        fragment_identity: Optional[FragmentIdentity] = None,
    ) -> FragmentPlacementResult:
        """
        Tentukan placement fragment baru.

        Alur:
            1. new topic? -> ROOT depth 0
            2. topic existing tapi belum ada fragment? -> ROOT depth 0
            3. Ada signal (fragment_identity atau retrieval)? -> score parents
            4. Parent terbaik >= threshold + margin? -> CHILD
               - tapi jika parent sudah di MAX_DEPTH? -> DEPTH_CAPPED_ATTACH
            5. Tidak ada parent cukup kuat? -> SIBLING

        Args:
            topic_key_id: ID TopicKey target
            retrieval: RetrievalResult (untuk fallback signal)
            is_new_topic: True jika classification == NEW_TOPIC
            fragment_identity: FragmentIdentity hasil Batch 2 (PRIMARY signal)

        Returns:
            FragmentPlacementResult dengan semua field terisi untuk audit
        """
        hc = CONFIG.hierarchy

        # Case 1: Topik baru -> selalu ROOT depth 0
        if is_new_topic:
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.ROOT,
                parent_fragment_id=None,
                depth=0,
                reason="new topic -> root fragment",
            )

        # Ambil fragment existing untuk topic ini
        existing = self.storage.get_fragment_metadata_by_key(topic_key_id)
        if not existing:
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.ROOT,
                parent_fragment_id=None,
                depth=0,
                reason="first fragment of topic -> root",
            )

        # Case 2: Tidak ada signal sama sekali -> SIBLING dari root
        signal_terms = self._extract_signal_terms(fragment_identity, retrieval)
        if not signal_terms:
            root = next((f for f in existing if f.parent_fragment_id is None), None)
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.SIBLING,
                parent_fragment_id=root.fragment_id if root else None,
                depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
                reason="no signal -> new sibling branch",
            )

        # Case 3: Scoring parent candidates
        eligible = [f for f in existing if f.depth < hc.MAX_FRAGMENT_DEPTH]
        scored = self.score_parent_candidates(
            signal_terms=signal_terms,
            candidates=eligible,
            fragment_identity=fragment_identity,
        )
        considered = [s.fragment_id for s in scored]

        if not scored:
            # Tidak ada eligible parent (semua sudah di MAX_DEPTH)
            root = next((f for f in existing if f.parent_fragment_id is None), None)
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.SIBLING,
                parent_fragment_id=root.fragment_id if root else None,
                depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
                considered_parent_ids=considered,
                reason="no eligible parent (all at MAX_DEPTH) -> sibling",
            )

        best = scored[0]
        runner = scored[1] if len(scored) > 1 else None
        runner_score = runner.final_score if runner else 0.0
        margin = best.final_score - runner_score

        strong_enough = best.final_score >= hc.MIN_PARENT_MATCH_SCORE
        clear_enough = (runner is None) or (margin >= hc.MIN_CHILD_MATCH_MARGIN)

        if strong_enough and clear_enough:
            parent_depth = best.depth
            if parent_depth + 1 <= hc.MAX_FRAGMENT_DEPTH:
                # Normal: CHILD di depth parent + 1
                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.CHILD,
                    parent_fragment_id=best.fragment_id,
                    depth=parent_depth + 1,
                    match_score=best.final_score,
                    runner_up_score=runner_score,
                    margin=margin,
                    considered_parent_ids=considered,
                    reason=(
                        f"parent '{best.fragment_key}' "
                        f"score={best.final_score:.3f} margin={margin:.3f}"
                    ),
                )
            else:
                # DEPTH_CAPPED_ATTACH: parent terbaik sudah di MAX_DEPTH
                # jangan ciptakan depth+1, attach di depth yang sama
                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.DEPTH_CAPPED_ATTACH,
                    parent_fragment_id=best.parent_fragment_id,
                    depth=parent_depth,
                    match_score=best.final_score,
                    runner_up_score=runner_score,
                    margin=margin,
                    depth_capped=True,
                    considered_parent_ids=considered,
                    reason=(
                        f"MAX_FRAGMENT_DEPTH={hc.MAX_FRAGMENT_DEPTH} reached; "
                        f"attached as sibling of '{best.fragment_key}'"
                    ),
                )

        # Tidak ada parent cukup kuat -> SIBLING baru
        root = next((f for f in existing if f.parent_fragment_id is None), None)
        return FragmentPlacementResult(
            kind=FragmentPlacementKind.SIBLING,
            parent_fragment_id=root.fragment_id if root else None,
            depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
            match_score=best.final_score,
            runner_up_score=runner_score,
            margin=margin,
            considered_parent_ids=considered,
            reason=(
                f"no parent above MIN_PARENT_MATCH_SCORE="
                f"{hc.MIN_PARENT_MATCH_SCORE:.2f} "
                f"(best={best.final_score:.3f}) -> new branch"
            ),
        )

    # ==========================================
    # SIGNAL EXTRACTION
    # ==========================================

    def _extract_signal_terms(
        self,
        fragment_identity: Optional[FragmentIdentity],
        retrieval: RetrievalResult,
    ) -> Set[str]:
        """
        Ekstrak signal terms dari fragment_identity (PRIMARY) atau
        retrieval.signal_profile (FALLBACK).

        Returns empty set jika keduanya tidak tersedia.
        """
        min_len = CONFIG.signal.MIN_TOKEN_LENGTH

        # PRIMARY: fragment_identity (Batch 2 combined user+AI)
        if fragment_identity is not None and fragment_identity.meaningful_tokens:
            return {
                t for t in fragment_identity.meaningful_tokens
                if len(t) >= min_len
            }

        # FALLBACK: retrieval.signal_profile (legacy path)
        if retrieval.signal_profile is not None and retrieval.signal_profile.unique_terms:
            return {
                t for t in retrieval.signal_profile.unique_terms
                if len(t) >= min_len
            }

        return set()

    # ==========================================
    # PARENT SCORING
    # ==========================================

    def score_parent_candidates(
        self,
        signal_terms: Set[str],
        candidates: List[FragmentMetadata],
        fragment_identity: Optional[FragmentIdentity] = None,
    ) -> List[FragmentCandidateScore]:
        """
        Score kandidat parent berdasarkan EVIDENCE.

        KRITIKAL: temporal_score dan recency_score di-hardcode ke 0.0.
        Ini mengunci invariant test_m5_e20:
            "parent dipilih by evidence, bukan fragment terbaru"

        Bobot diambil dari FragmentRankingConstants, lalu DINORMALISASI
        atas subset yang dipakai (identity, token, co-occurrence).
        Tidak ada angka ajaib, tidak ada pembobotan tambahan.

        Identity score = max(fragment_key_score, helper_score, aggregated_helper_score)
        """
        if not candidates or not signal_terms:
            return []

        # Normalisasi bobot dari FragmentRankingConstants
        w = CONFIG.fragment_ranking
        w_identity = w.FRAGMENT_TOKEN_WEIGHT
        w_token = w.FRAGMENT_COOC_WEIGHT
        w_cooc = w.FRAGMENT_COOC_WEIGHT
        w_sum = (w_identity + w_token + w_cooc) or 1.0

        ev_thr = CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD
        results: List[FragmentCandidateScore] = []

        for frag in candidates:
            # Token score: overlap antara signal dan fragment.token_profile
            token_score = self._compute_token_overlap(
                signal_terms, frag.token_profile or {}
            )

            # Identity scores: fragment_key, helper, aggregated_helper
            identity = self._compute_identity_score(frag, signal_terms)

            # Co-occurrence score (window pairs)
            cooc_score = self._compute_cooc_score(frag, fragment_identity, signal_terms)

            # Final score (tanpa temporal & recency)
            final = (
                identity * w_identity
                + token_score * w_token
                + cooc_score * w_cooc
            ) / w_sum

            # Catat evidence yang mendukung (untuk audit)
            supporting: List[str] = []
            if identity >= ev_thr:
                supporting.append("identity")
            if token_score >= ev_thr:
                supporting.append("token")
            if cooc_score >= ev_thr:
                supporting.append("co_occurrence")

            results.append(FragmentCandidateScore(
                fragment_id=frag.fragment_id,
                fragment_key=frag.fragment_key,
                topic_key_id=frag.topic_key_id,
                depth=frag.depth,
                parent_fragment_id=frag.parent_fragment_id,
                fragment_key_score=identity,
                helper_score=identity,
                aggregated_helper_score=identity,
                token_score=token_score,
                co_occurrence_score=cooc_score,
                # TEMPORAL & RECENCY DINOLKAN (invariant test_m5_e20)
                temporal_score=0.0,
                graph_score=0.0,
                recency_score=0.0,
                final_score=final,
                supporting_evidence_names=supporting,
                notes=["placement scoring: temporal & recency excluded by design"],
            ))

        results.sort(key=lambda c: c.final_score, reverse=True)
        return results

    def _compute_token_overlap(
        self,
        signal_terms: Set[str],
        token_profile: Dict[str, int],
    ) -> float:
        """
        Token overlap ratio: |signal ∩ profile| / |profile|.
        
        Returns 0.0 jika profile kosong.
        """
        if not token_profile:
            return 0.0
        profile_terms = set(token_profile.keys())
        if not profile_terms:
            return 0.0
        overlap = signal_terms.intersection(profile_terms)
        return len(overlap) / len(profile_terms)

    def _compute_identity_score(
        self,
        frag: FragmentMetadata,
        signal_terms: Set[str],
    ) -> float:
        """
        Identity score = MAX dari tiga sumber identitas fragment:
            - fragment_key (nama bagian)
            - fragment_helper_description (own helper)
            - aggregated_helper_description (warisan descendant)

        Formula: |signal ∩ pool| / |pool| untuk masing-masing pool,
                 lalu ambil maksimum.

        Returns 0.0 jika tidak ada identitas yang tersedia.
        """
        min_len = CONFIG.signal.MIN_TOKEN_LENGTH
        pools: List[Set[str]] = []
        for text in (
            frag.fragment_key,
            frag.fragment_helper_description,
            frag.aggregated_helper_description,
        ):
            if not text:
                continue
            toks = set(re.findall(r'\b\w+\b', text.lower()))
            toks = {t for t in toks if len(t) >= min_len}
            if toks:
                pools.append(toks)

        if not pools:
            return 0.0

        best = 0.0
        for pool in pools:
            overlap = signal_terms.intersection(pool)
            ratio = len(overlap) / len(pool)
            if ratio > best:
                best = ratio
        return min(1.0, best)

    def _compute_cooc_score(
        self,
        frag: FragmentMetadata,
        fragment_identity: Optional[FragmentIdentity],
        signal_terms: Set[str],
    ) -> float:
        """
        Co-occurrence score antara fragment dan signal.

        Jika fragment_identity punya co_occurrence data (window pairs),
        pakai itu. Jika tidak, fallback ke simple token overlap
        dengan co_occurrence_profile fragment.

        Returns 0.0 jika tidak ada co-occurrence evidence.
        """
        if not frag.co_occurrence_profile:
            return 0.0

        # Fallback: simple token-level co-occurrence
        # (window pairs dari fragment_identity belum di-expose ke sini,
        # jadi kita pakai token overlap sebagai proxy)
        profile = frag.co_occurrence_profile
        total_w = sum(profile.values())
        if total_w <= 0:
            return 0.0

        matched_w = 0.0
        for token in signal_terms:
            matched_w += profile.get(token, 0)
        return min(1.0, matched_w / total_w)

    # ==========================================
    # HELPER: DEPTH OPERATIONS
    # ==========================================

    def get_max_allowed_depth(self) -> int:
        """Accessor untuk MAX_FRAGMENT_DEPTH (memudahkan test mocking)."""
        return CONFIG.hierarchy.MAX_FRAGMENT_DEPTH

    def is_depth_valid(self, depth: int) -> bool:
        """Check apakah depth masih di bawah MAX_FRAGMENT_DEPTH."""
        return 0 <= depth <= CONFIG.hierarchy.MAX_FRAGMENT_DEPTH

    def would_exceed_depth(self, parent_depth: int) -> bool:
        """
        Check apakah menempatkan child di parent_depth + 1
        akan melanggar MAX_FRAGMENT_DEPTH.

        Returns True jika parent_depth + 1 > MAX_FRAGMENT_DEPTH.
        """
        return (parent_depth + 1) > CONFIG.hierarchy.MAX_FRAGMENT_DEPTH

    def get_depth_label(self, depth: int) -> str:
        """
        Label semantik untuk depth (untuk observability/context framing).

        Definisi DIKUNCI (sinkron dengan context_builder._depth_label):
            0 -> OPENING
            1 -> SECTION
            2 -> SUBSECTION
            3+ -> DETAIL
        """
        if depth <= 0:
            return "OPENING"
        if depth == 1:
            return "SECTION"
        if depth == 2:
            return "SUBSECTION"
        return "DETAIL"
