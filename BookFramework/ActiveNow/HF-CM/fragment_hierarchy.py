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

[BATCH 11] TOP-DOWN HIERARCHICAL TRAVERSAL
  Algoritma determine_placement diubah dari:
      ALL ELIGIBLE NODES → GLOBAL SCORE → BEST NODE
  menjadi:
      ROOT
          ↓
      DIRECT CHILDREN LEVEL 1 → BEST LEVEL-1 BRANCH
          ↓
      DIRECT CHILDREN OF SELECTED BRANCH → BEST LEVEL-2 BRANCH
          ↓
      ... (lanjut sampai MAX_DEPTH atau tidak ada match kuat)

  Ini mengunci perilaku bahwa branch depth 1 HARUS dipilih terlebih
  dahulu sebelum fragment depth 2 boleh dipertimbangkan. Descendant
  di branch lain TIDAK BOLEH mencuri fragment secara global.
"""

import logging
import re
from typing import List, Optional, Dict, Set, Tuple

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

    [BATCH 11] Algoritma placement sekarang top-down traversal.
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

        [BATCH 11] Alur baru (top-down traversal):
            1. new topic? -> ROOT depth 0
            2. topic existing tapi belum ada fragment? -> ROOT depth 0
            3. Tidak ada signal? -> SIBLING dari root
            4. Top-down traversal:
                a. Ambil ROOT (depth 0)
                b. Loop:
                   i.   current_depth >= MAX? -> DEPTH_CAPPED_ATTACH
                   ii.  ambil direct children current_parent
                   iii. jika tidak ada -> CHILD dari current_parent
                   iv.  score hanya antar sibling di level ini
                   v.   jika tidak ada match kuat -> SIBLING
                   vi.  jika ada match kuat -> descend ke branch terpilih

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
            root = self._get_root_fragment(existing)
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.SIBLING,
                parent_fragment_id=root.fragment_id if root else None,
                depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
                reason="no signal -> new sibling branch",
            )

        # Case 3: [BATCH 11] Top-down hierarchical traversal
        root = self._get_root_fragment(existing)

        if root is None:
            # Defensive: topik existing tapi tidak ada root (shouldn't happen).
            # Recovery: buat root baru.
            return FragmentPlacementResult(
                kind=FragmentPlacementKind.ROOT,
                parent_fragment_id=None,
                depth=0,
                reason="existing topic has no valid root -> root recovery",
            )

        # Audit trail: setiap fragment yang pernah di-score dicatat per-level.
        considered_parent_ids: List[str] = []
        current_parent = root

        # Safety: hard limit on iteration count (defensive against cycles).
        max_iterations = hc.MAX_FRAGMENT_DEPTH + 2
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            current_depth = current_parent.depth

            # --------------------------------------------------------------
            # A. MAX DEPTH sudah tercapai
            # --------------------------------------------------------------
            #
            # Fragment baru tidak boleh menjadi child dari current_parent
            # karena itu akan menciptakan depth = MAX + 1.
            #
            # Fallback: DEPTH_CAPPED_ATTACH sebagai sibling dari current_parent.
            #
            if current_depth >= hc.MAX_FRAGMENT_DEPTH:
                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.DEPTH_CAPPED_ATTACH,
                    parent_fragment_id=current_parent.parent_fragment_id,
                    depth=current_depth,
                    match_score=0.0,
                    runner_up_score=0.0,
                    margin=0.0,
                    depth_capped=True,
                    considered_parent_ids=considered_parent_ids,
                    reason=(
                        f"MAX_FRAGMENT_DEPTH={hc.MAX_FRAGMENT_DEPTH} reached "
                        f"at '{current_parent.fragment_key}' -> attach as sibling"
                    ),
                )

            # --------------------------------------------------------------
            # B. Ambil hanya anak langsung dari current_parent
            # --------------------------------------------------------------
            children = self._get_direct_children(
                existing, current_parent.fragment_id
            )

            # --------------------------------------------------------------
            # C. Belum ada child -> fragment baru jadi child pertama
            # --------------------------------------------------------------
            if not children:
                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.CHILD,
                    parent_fragment_id=current_parent.fragment_id,
                    depth=current_depth + 1,
                    considered_parent_ids=considered_parent_ids,
                    reason=(
                        f"no children under '{current_parent.fragment_key}' "
                        f"-> first child"
                    ),
                )

            # --------------------------------------------------------------
            # D. Score HANYA sibling pada level ini
            # --------------------------------------------------------------
            best, runner, scored, is_strong = self._select_best_at_level(
                signal_terms=signal_terms,
                candidates=children,
                fragment_identity=fragment_identity,
            )

            # Audit: catat semua kandidat yang dipertimbangkan di level ini.
            considered_parent_ids.extend(s.fragment_id for s in scored)

            # --------------------------------------------------------------
            # E. Tidak ada match kuat -> SIBLING dari current_parent
            # --------------------------------------------------------------
            if not is_strong or best is None:
                best_score = best.final_score if best else 0.0
                runner_score = (
                    runner.final_score if runner is not None else 0.0
                )
                margin = best_score - runner_score

                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.SIBLING,
                    parent_fragment_id=current_parent.fragment_id,
                    depth=current_depth + 1,
                    match_score=best_score,
                    runner_up_score=runner_score,
                    margin=margin,
                    considered_parent_ids=considered_parent_ids,
                    reason=(
                        f"no strong match among direct children of "
                        f"'{current_parent.fragment_key}' -> new sibling branch"
                    ),
                )

            # --------------------------------------------------------------
            # F. Match kuat -> descend ke branch terpilih
            # --------------------------------------------------------------
            next_parent = next(
                (f for f in children if f.fragment_id == best.fragment_id),
                None,
            )

            # Defensive: seharusnya tidak terjadi (best berasal dari children).
            if next_parent is None:
                return FragmentPlacementResult(
                    kind=FragmentPlacementKind.SIBLING,
                    parent_fragment_id=root.fragment_id,
                    depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
                    considered_parent_ids=considered_parent_ids,
                    reason=(
                        "selected hierarchy candidate not found "
                        "-> root sibling fallback"
                    ),
                )

            current_parent = next_parent

        # Safety net: kalau loop tidak pernah return (tidak mungkin dengan
        # max_iterations guard, tapi defensive).
        return FragmentPlacementResult(
            kind=FragmentPlacementKind.SIBLING,
            parent_fragment_id=root.fragment_id,
            depth=min(hc.NEW_BRANCH_DEPTH, hc.MAX_FRAGMENT_DEPTH),
            considered_parent_ids=considered_parent_ids,
            reason="iteration guard hit -> root sibling fallback",
        )

    # ==========================================
    # [BATCH 11] TRAVERSAL HELPERS
    # ==========================================

    def _get_root_fragment(
        self,
        fragments: List[FragmentMetadata],
    ) -> Optional[FragmentMetadata]:
        """
        Ambil root fragment dari list fragment.

        Root didefinisikan sebagai fragment dengan:
            parent_fragment_id is None
            depth == 0

        Fallback jika tidak ada dengan depth 0:
            fragment dengan parent_fragment_id is None (any depth).

        Returns None jika tidak ada fragment tanpa parent sama sekali.
        """
        for fragment in fragments:
            if (
                fragment.parent_fragment_id is None
                and (fragment.depth or 0) == 0
            ):
                return fragment

        # Fallback: fragment tanpa parent (meski depth mungkin != 0)
        for fragment in fragments:
            if fragment.parent_fragment_id is None:
                return fragment

        return None

    def _get_direct_children(
        self,
        fragments: List[FragmentMetadata],
        parent_fragment_id: str,
    ) -> List[FragmentMetadata]:
        """
        Ambil hanya anak langsung dari satu parent.

        TIDAK mengambil:
            - sibling dari parent
            - descendant lebih dalam
            - fragment dari branch lain

        Ini yang memastikan traversal benar-benar per-branch.
        """
        return [
            fragment
            for fragment in fragments
            if fragment.parent_fragment_id == parent_fragment_id
        ]

    def _select_best_at_level(
        self,
        signal_terms: Set[str],
        candidates: List[FragmentMetadata],
        fragment_identity: Optional[FragmentIdentity],
    ) -> Tuple[
        Optional[FragmentCandidateScore],
        Optional[FragmentCandidateScore],
        List[FragmentCandidateScore],
        bool,
    ]:
        """
        Score kandidat HANYA dalam satu level sibling.

        Returns tuple:
            (best, runner_up, scored_list, is_strong_match)

        is_strong_match = True jika:
            - best.final_score >= MIN_PARENT_MATCH_SCORE
            - (runner_up is None) OR
              (best.final_score - runner_up.final_score >= MIN_CHILD_MATCH_MARGIN)

        Margin dibandingkan dengan runner_up DI LEVEL YANG SAMA,
        bukan dengan fragment di branch lain. Ini yang membuat
        "Modal Murah vs Mahal" bisa menang meskipun "Lokasi" punya
        skor global lebih rendah tapi ada di branch berbeda.
        """
        if not candidates:
            return None, None, [], False

        scored = self.score_parent_candidates(
            signal_terms=signal_terms,
            candidates=candidates,
            fragment_identity=fragment_identity,
        )

        if not scored:
            return None, None, [], False

        best = scored[0]
        runner = scored[1] if len(scored) > 1 else None
        runner_score = runner.final_score if runner else 0.0
        margin = best.final_score - runner_score

        hc = CONFIG.hierarchy
        strong_enough = best.final_score >= hc.MIN_PARENT_MATCH_SCORE
        clear_enough = (runner is None) or (margin >= hc.MIN_CHILD_MATCH_MARGIN)
        is_strong = strong_enough and clear_enough

        return best, runner, scored, is_strong

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
