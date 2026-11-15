"""
helper_propagation.py
Helper Description Propagation (Mekanisme 3).

[REFACTOR BATCH 7]
Pemisahan dua gate:
  1. STRUCTURAL GATE (selalu lolos jika fragment berhasil disimpan)
     - Child helper boleh naik ke parent
     - Parent aggregate boleh bertambah
     - TopicKey aggregate boleh berkembang
  2. QUALITY GATE (cek substansi)
     - Helper tidak boleh kosong
     - Helper tidak boleh pure low-information noise
     - Helper tidak boleh duplikat persis

Ini berbeda dengan:
  PROBABILISTIC MUTATION GATE (memory_engine._compute_mutation_permissions)
     - token_profile tidak boleh tercemar
     - co_occurrence_profile tidak boleh tercemar
     - helper_description (own identity) tidak boleh tercemar

Prinsip kunci:
    FRAGMENT BERHASIL DISIMPAN
        ↓
    HELPER STRUKTURAL TETAP NAIK
        ↓
    ANTI-POLLUTION TETAP BEKERJA UNTUK PROFILE

Alur propagasi:
    Child fragment
        ↓ extract helper
    Parent fragment
        ↓ aggregate descendant evidence
    Ancestor fragments
        ↓ propagate upward
    Root fragment
        ↓
    TopicKey.aggregated_helper_description

DIPANGGIL SETELAH TX1 commit, bukan di dalam transaksi inti.
Alasannya: storage.get_topic_key() / get_fragment_ancestors() membuka
session sendiri. Jika dipanggil di dalam transaksi yang belum commit,
ia membaca nilai LAMA lalu menimpanya kembali (stale-read overwrite).
"""

import logging
import re
from typing import List, Dict, Optional

from config import CONFIG
from models import (
    FragmentMetadata, TopicKey, TurnAnalysisResult,
    HelperPropagationResult,
)

logger = logging.getLogger(__name__)


class HelperPropagationManager:
    """
    Manajer propagasi helper description dari child ke ancestor.

    Dependency:
        - storage: untuk fetch ancestors / root fragments / TopicKey
                   dan untuk update fragment/topic helper.

    Kontrak:
        - Method propagate() TIDAK membuka transaksi sendiri.
          Ia menerima session dari caller yang sudah membuka TX2.
        - propagate() mengembalikan HelperPropagationResult untuk observability.
        - [BATCH 7] Quality gate terpisah dari mutation permission.
          Fragment yang valid = helper strukturalnya naik,
          terlepas dari apakah mutation permission di-grant.
    """

    def __init__(self, storage):
        self.storage = storage

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def propagate(
        self,
        session,
        new_fragment: FragmentMetadata,
        topic_key_id: str,
        analysis: TurnAnalysisResult,
    ) -> HelperPropagationResult:
        """
        Propagasi helper dari fragment baru ke ancestor dan TopicKey.

        [BATCH 7] Alur:
            1. QUALITY GATE: cek helper fragment layak propagate.
            2. Extract contribution dari fragment (helper atau key).
            3. Fetch ancestors via storage.get_fragment_ancestors.
            4. Update setiap ancestor: dedup, bound, aggregate.
            5. Update TopicKey dari root fragments: dedup, bound, aggregate.
            6. Return HelperPropagationResult (observability).

        TIDAK ADA mutation permission check di sini.
        Itu sudah di-handle di TX1 (profile update).
        Structural propagation = hak setiap fragment yang tersimpan.
        """
        hp = CONFIG.helper_propagation
        result = HelperPropagationResult()

        # Extract kontribusi dari fragment baru.
        contribution = self._extract_contribution(new_fragment)

        # [BATCH 7] QUALITY GATE: cek substansi, bukan mutation permission.
        quality_check, quality_reason = self._quality_gate_check(contribution)
        if not quality_check:
            result.reason = f"quality gate rejected: {quality_reason}"
            return result

        # Ambil ancestors.
        ancestors = self.storage.get_fragment_ancestors(
            new_fragment.fragment_id, max_depth=hp.MAX_PROPAGATION_DEPTH
        )
        result.chain_depth = len(ancestors)

        # Propagasi ke setiap ancestor.
        ancestor_stats = self._propagate_to_ancestors(
            session=session,
            new_fragment=new_fragment,
            contribution=contribution,
            ancestors=ancestors,
        )

        # Propagasi ke TopicKey (opsional per config).
        topic_stats: Dict = {
            "added": 0,
            "deduplicated": 0,
            "truncated": False,
        }
        if hp.PROPAGATE_TO_TOPIC_KEY:
            topic_stats = self._propagate_to_topic_key(
                session=session,
                topic_key_id=topic_key_id,
                new_fragment=new_fragment,
            )

        # Aggregate stats
        added = ancestor_stats["added"] + topic_stats["added"]
        deduped = ancestor_stats["deduplicated"] + topic_stats["deduplicated"]
        truncated = ancestor_stats["truncated"] or topic_stats["truncated"]

        result.propagated = added > 0
        result.updated_fragment_ids = ancestor_stats["updated_ids"]
        result.contributions_added = added
        result.contributions_deduplicated = deduped
        result.truncated = truncated
        if topic_stats["added"] > 0:
            result.updated_topic_key_id = topic_key_id
        if added > 0:
            result.reason = f"propagated {added} contribution(s)"
        else:
            result.reason = "no new contribution (already present or duplicate)"

        return result

    # ==========================================
    # EXTRACTION
    # ==========================================

    def _extract_contribution(self, fragment: FragmentMetadata) -> str:
        """
        Ambil kontribusi helper dari fragment.

        Prioritas:
            1. fragment_helper_description (own helper)
            2. fragment_key (fallback)
            3. aggregated_helper_description (fallback terakhir)
            4. "" jika semuanya kosong

        Strip whitespace, return trimmed string.
        """
        text = (
            fragment.fragment_helper_description
            or fragment.fragment_key
            or fragment.aggregated_helper_description
            or ""
        )
        return text.strip()

    # ==========================================
    # [BATCH 7] QUALITY GATE
    # ==========================================

    def _quality_gate_check(self, contribution: str) -> tuple:
        """
        [BATCH 7] Quality gate untuk structural propagation.

        Returns: (passed: bool, reason: str)

        Checks:
            1. Contribution tidak boleh kosong
            2. Contribution harus punya minimal substance (token count)
            3. Contribution tidak boleh pure low-information noise

        BERBEDA dengan mutation permission di memory_engine:
            - Mutation permission: "bolehkah topic profile berubah?"
            - Quality gate: "apakah helper ini layak naik ke parent?"

        Fragment yang TIDAK lolos quality gate tetap disimpan,
        tapi helper-nya tidak propagate (misal: "Bagaimana itu?"
        → fragment tersimpan tapi tidak jadi bagian cerita parent).
        """
        hp = CONFIG.helper_propagation

        if not contribution:
            return False, "no helper contribution"

        # Extract meaningful tokens
        min_len = hp.MIN_HELPER_TOKEN_LENGTH
        raw_tokens = re.findall(r'\b\w+\b', contribution.lower())
        meaningful = [t for t in raw_tokens if len(t) >= min_len]

        if not meaningful:
            return False, "no meaningful tokens in contribution"

        # Check low-information ratio
        from config import CONFIG as cfg
        low_info_set = cfg.token_quality.LOW_INFORMATION_TOKENS
        low_info_count = sum(1 for t in meaningful if t in low_info_set)
        low_info_ratio = low_info_count / len(meaningful) if meaningful else 1.0

        # Jika > 80% low-information, tolak
        if low_info_ratio > 0.8:
            return False, f"too much low-information ({low_info_ratio:.2f})"

        return True, "quality gate passed"

    # ==========================================
    # ANCESTOR PROPAGATION
    # ==========================================

    def _propagate_to_ancestors(
        self,
        session,
        new_fragment: FragmentMetadata,
        contribution: str,
        ancestors: List[FragmentMetadata],
    ) -> Dict:
        """
        Propagasi kontribusi ke setiap ancestor fragment.

        Untuk setiap ancestor:
            - Cek apakah sudah berkontribusi (dedup by fragment_id).
            - Cek similaritas dengan evidence existing (Jaccard dedup).
            - Jika evidence penuh -> truncate (keep terbaru).
            - Append kontribusi + contributor.
            - Rebuild aggregated_helper_description.
            - Update fragment via storage.update_fragment_helper.

        Returns dict dengan:
            added: int (jumlah ancestor yang di-update)
            deduplicated: int (jumlah yang di-skip karena duplikat)
            truncated: bool (apakah ada ancestor yang evidence-nya di-truncate)
            updated_ids: List[str]
        """
        hp = CONFIG.helper_propagation
        added = 0
        deduped = 0
        truncated = False
        updated_ids: List[str] = []

        for ancestor in ancestors:
            # Cek apakah fragment ini sudah pernah berkontribusi ke ancestor ini.
            if new_fragment.fragment_id in (ancestor.helper_contributor_ids or []):
                deduped += 1
                continue

            evidence = list(ancestor.descendant_helper_evidence or [])
            contributors = list(ancestor.helper_contributor_ids or [])

            # Cek similaritas dengan evidence existing.
            duplicate = any(
                self._helper_similarity(contribution, existing)
                >= hp.DEDUP_SIMILARITY_THRESHOLD
                for existing in evidence
            )
            if duplicate:
                deduped += 1
                continue

            # Bounded evidence list.
            if len(evidence) >= hp.MAX_DESCENDANT_EVIDENCE:
                truncated = True
                # Keep yang terbaru (slice dari belakang).
                evidence = evidence[-(hp.MAX_DESCENDANT_EVIDENCE - 1):]

            # Append kontribusi baru.
            evidence.append(contribution)
            contributors.append(new_fragment.fragment_id)

            # Rebuild aggregated helper.
            new_aggregate = self._aggregate_helper(
                ancestor.fragment_helper_description, evidence
            )

            # Update fragment.
            updated = ancestor.model_copy(update={
                "descendant_helper_evidence": evidence,
                "helper_contributor_ids": contributors,
                "aggregated_helper_description": new_aggregate,
            })
            self.storage.update_fragment_helper(session, updated)
            updated_ids.append(ancestor.fragment_id)
            added += 1

        return {
            "added": added,
            "deduplicated": deduped,
            "truncated": truncated,
            "updated_ids": updated_ids,
        }

    # ==========================================
    # TOPIC KEY PROPAGATION
    # ==========================================

    def _propagate_to_topic_key(
        self,
        session,
        topic_key_id: str,
        new_fragment: FragmentMetadata,
    ) -> Dict:
        """
        Propagasi ke TopicKey.aggregated_helper_description dari root fragments.

        Alur:
            1. Fetch TopicKey.
            2. Cek apakah fragment sudah jadi sumber (dedup by fragment_id).
            3. Fetch root fragments.
            4. Ambil aggregated atau own helper dari setiap root.
            5. Dedup via Jaccard similarity.
            6. Bound jumlah evidence.
            7. Bangun aggregated_helper_description TopicKey.
            8. Update TopicKey via storage.update_topic_helper.

        Returns dict dengan:
            added: int (0 atau 1)
            deduplicated: int (0 atau 1)
            truncated: bool
        """
        hp = CONFIG.helper_propagation

        key_obj = self.storage.get_topic_key(topic_key_id)
        if key_obj is None:
            return {"added": 0, "deduplicated": 0, "truncated": False}

        # Dedup: cek apakah fragment ini sudah tercatat sebagai source.
        if new_fragment.fragment_id in (key_obj.helper_source_fragment_ids or []):
            return {"added": 0, "deduplicated": 1, "truncated": False}

        # Fetch root fragments.
        roots = self.storage.get_root_fragments_for_topic(topic_key_id)

        # Collect evidence dari setiap root.
        k_evidence: List[str] = []
        for r in roots:
            text = r.aggregated_helper_description or r.fragment_helper_description
            if text:
                k_evidence.append(text)

        # Dedup + bound.
        bounded: List[str] = []
        truncated = False
        for text in k_evidence:
            already = any(
                self._helper_similarity(text, b) >= hp.DEDUP_SIMILARITY_THRESHOLD
                for b in bounded
            )
            if already:
                continue
            bounded.append(text)
            if len(bounded) >= hp.MAX_DESCENDANT_EVIDENCE:
                truncated = True
                break

        # Append fragment baru sebagai source.
        sources = list(key_obj.helper_source_fragment_ids or [])
        sources.append(new_fragment.fragment_id)

        # Bangun aggregated helper untuk TopicKey.
        new_key_aggregate = self._aggregate_helper(key_obj.helper_description, bounded)

        # Update TopicKey.
        updated_key = key_obj.model_copy(update={
            "aggregated_helper_description": new_key_aggregate,
            "helper_source_fragment_ids": sources,
            "updated_at": new_fragment.created_at,
        })
        self.storage.update_topic_helper(session, updated_key)

        return {"added": 1, "deduplicated": 0, "truncated": truncated}

    # ==========================================
    # HELPER METHODS
    # ==========================================

    def _helper_similarity(self, a: str, b: str) -> float:
        """
        Jaccard similarity antara dua string berdasarkan token overlap.

        Formula: |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|

        Filter: hanya token dengan panjang >= MIN_HELPER_TOKEN_LENGTH.

        Returns 0.0 jika salah satu string kosong setelah filter.
        """
        min_len = CONFIG.helper_propagation.MIN_HELPER_TOKEN_LENGTH
        ta = {t for t in re.findall(r'\b\w+\b', a.lower()) if len(t) >= min_len}
        tb = {t for t in re.findall(r'\b\w+\b', b.lower()) if len(t) >= min_len}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _aggregate_helper(self, own: str, contributions: List[str]) -> str:
        """
        Bangun aggregated helper description: own + CHILD_SUMMARY_PREFIX + contributions.

        Format:
            "OWN. Mencakup: c1; c2; c3"
            atau jika own kosong:
            "Mencakup: c1; c2; c3"

        Bounded: output dipotong ke MAX_HELPER_LENGTH karakter jika terlalu panjang.

        Tidak mengubah exact payload. Hanya metadata.
        """
        hp = CONFIG.helper_propagation
        own = (own or "").strip()
        if not contributions:
            return own[:hp.MAX_HELPER_LENGTH]

        joined = hp.CONTRIBUTION_SEPARATOR.join(contributions)
        if own:
            aggregate = f"{own}. {hp.CHILD_SUMMARY_PREFIX}{joined}"
        else:
            aggregate = f"{hp.CHILD_SUMMARY_PREFIX}{joined}"
        if len(aggregate) > hp.MAX_HELPER_LENGTH:
            aggregate = aggregate[:hp.MAX_HELPER_LENGTH].rstrip()
        return aggregate
