"""
association_engine.py
Cross-Topic & Cross-Fragment Association Analysis (Batch E).

Tanggung jawab:
    Setelah fragment memiliki STRUCTURAL home (ownership + hierarchy),
    engine ini menentukan dengan APA SAJA fragment tersebut BERHUBUNGAN:
        - TopicKey lain (selain primary ownership)
        - Fragment spesifik lain (cross-topic atau cross-branch)
        - Sumber dominan (PRIMARY vs SECONDARY)

Prinsip kunci (dari dokumentasi):
    Ownership   = topic_key_id (single, structural)
    Hierarchy   = parent_fragment_id + depth (single path)
    Association = MANY targets (cross-topic & cross-fragment)
    Dominance   = PRIMARY / SECONDARY source ranking

Anti-false-association:
    1. MIN_ASSOCIATION_TOKEN_OVERLAP = 0.20 (floor token overlap)
    2. MIN_ASSOCIATION_EVIDENCE_FAMILIES = 2 (multi-family requirement)
       → token overlap SAJA tidak cukup, harus ada helper/graph/metadata
    3. Threshold berbeda per target type:
       - TOPIC_ASSOCIATION_THRESHOLD = 0.35 (fragment → TopicKey)
       - FRAGMENT_ASSOCIATION_THRESHOLD = 0.30 (fragment → fragment)
       - CROSS_TOPIC_ASSOCIATION_THRESHOLD = 0.30 (lintas topik)
    4. Clue processing (POSITIVE/NEGATIVE/EXCLUSION) adjust skor
    5. Dominance margin (DOMINANT_SOURCE_MARGIN = 0.15) prevents
       overconfident PRIMARY assignment when sources are close

Kontrak:
    - Stateless: tidak memegang storage, tidak mengubah state.
    - Caller (memory_engine) inject data: all TopicKeys, candidate fragments,
      fragment identity, clues, composite analysis.
    - analyze() returns AssociationAnalysis (immutable result).
    - Persistence (graph edges) adalah tanggung jawab caller, bukan engine ini.

Flow dalam pipeline (Batch G):
    CompositeMemoryAnalyzer.detect_composite_sources()
        ↓ (composite sources = subset of association sources)
    AssociationEngine.analyze()
        ↓ (broader: topic + fragment associations + dominance)
    memory_engine._persist_memory()
        ↓ (writes edges: REFERENCES from dominant sources)
    storage
"""

import logging
import re
from typing import Optional, List, Dict, Set, Tuple

from config import CONFIG
from models import (
    FragmentMetadata,
    TopicKey,
    ResponseMetadata,
    AssociationClue,
    AssociationTarget,
    AssociationAnalysis,
    DominantAssociation,
    CompositeSourceAnalysis,
    GraphNodeType,
    ClueType,
    AssociationRole,
    RelationshipType,
    EvidenceFamily,
)
from fragment_identity import FragmentIdentity


logger = logging.getLogger(__name__)


class AssociationEngine:
    """
    Analyzer untuk cross-topic & cross-fragment associations.

    Dependency:
        - Tidak ada (pure analysis function).
        - Caller injects data: TopicKeys, fragments, identity, clues.

    Kontrak:
        - Method analyze() adalah PURE function.
        - Tidak mengubah state, tidak menulis ke storage.
        - Returns AssociationAnalysis dengan semua field observability terisi.
    """

    def __init__(self):
        self._cfg = CONFIG.association
        self._signal_cfg = CONFIG.signal
        self._evidence_cfg = CONFIG.evidence

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def analyze(
        self,
        new_fragment_identity: Optional[FragmentIdentity],
        primary_topic_key_id: Optional[str],
        new_fragment_id: Optional[str],
        all_topic_keys: List[TopicKey],
        candidate_fragments: List[FragmentMetadata],
        clues: Optional[List[AssociationClue]] = None,
        composite_analysis: Optional[CompositeSourceAnalysis] = None,
        response_metadata: Optional[ResponseMetadata] = None,
    ) -> AssociationAnalysis:
        """
        Analisis asosiasi lengkap untuk fragment baru.

        Args:
            new_fragment_identity: identitas fragment (fragment_key, helper, tokens).
            primary_topic_key_id: structural owner (dikecualikan dari associations).
            new_fragment_id: ID fragment baru (dikecualikan untuk mencegah self-association).
            all_topic_keys: semua TopicKey aktif.
            candidate_fragments: fragment kandidat untuk cross-fragment association.
            clues: daftar AssociationClue (dari LLM metadata + signal profile).
            composite_analysis: output CompositeMemoryAnalyzer (Batch 5).
            response_metadata: metadata LLM tervalidasi (opsional, tambahan clue).

        Returns:
            AssociationAnalysis lengkap dengan topic/fragment associations,
            dominant sources, dan audit trail clue usage.
        """
        clues = clues or []
        result = AssociationAnalysis()

        # Guard: tidak ada identity → tidak bisa associate
        if new_fragment_identity is None or not new_fragment_identity.meaningful_tokens:
            result.reason = "no meaningful fragment identity"
            return result

        # Ekstrak token pool dari identity (primary evidence)
        signal_tokens = set(new_fragment_identity.meaningful_tokens)

        # Build exclusion set dari EXCLUSION clues (untuk short-circuit)
        exclusion_tokens: Set[str] = set()
        exclusion_target_ids: Set[str] = set()
        for clue in clues:
            if clue.clue_type == ClueType.EXCLUSION:
                exclusion_tokens.update(clue.tokens or [])
                exclusion_target_ids.update(clue.rejected_target_ids or [])

        # ==========================================
        # PHASE 1: TOPIC ASSOCIATIONS
        # ==========================================
        topic_targets: List[AssociationTarget] = []
        topic_positive_clues: List[AssociationClue] = []
        topic_negative_clues: List[AssociationClue] = []

        for key in all_topic_keys:
            # Skip primary topic (structural, bukan associative)
            if primary_topic_key_id and key.key_id == primary_topic_key_id:
                continue

            # Skip jika di-exclusion list
            if key.key_id in exclusion_target_ids:
                result.rejected_targets.append(key.key_id)
                continue

            target, pos_used, neg_used = self._score_topic_association(
                key=key,
                signal_tokens=signal_tokens,
                clues=clues,
                exclusion_tokens=exclusion_tokens,
                primary_topic_key_id=primary_topic_key_id,
            )

            if target is not None:
                topic_targets.append(target)
                topic_positive_clues.extend(pos_used)
                topic_negative_clues.extend(neg_used)

        # Sort & limit
        topic_targets.sort(key=lambda t: t.score, reverse=True)
        topic_targets = topic_targets[: self._cfg.MAX_TOPIC_ASSOCIATIONS]

        # ==========================================
        # PHASE 2: FRAGMENT ASSOCIATIONS
        # ==========================================
        fragment_targets: List[AssociationTarget] = []
        frag_positive_clues: List[AssociationClue] = []
        frag_negative_clues: List[AssociationClue] = []

        for frag in candidate_fragments:
            # Skip self-association
            if new_fragment_id and frag.fragment_id == new_fragment_id:
                continue

            # Skip jika di-exclusion list
            if frag.fragment_id in exclusion_target_ids:
                result.rejected_targets.append(frag.fragment_id)
                continue

            target, pos_used, neg_used = self._score_fragment_association(
                frag=frag,
                signal_tokens=signal_tokens,
                clues=clues,
                exclusion_tokens=exclusion_tokens,
                primary_topic_key_id=primary_topic_key_id,
            )

            if target is not None:
                fragment_targets.append(target)
                frag_positive_clues.extend(pos_used)
                frag_negative_clues.extend(neg_used)

        # Sort & limit
        fragment_targets.sort(key=lambda t: t.score, reverse=True)
        fragment_targets = fragment_targets[: self._cfg.MAX_FRAGMENT_ASSOCIATIONS]

        # ==========================================
        # PHASE 3: DOMINANCE RANKING
        # ==========================================
        dominant_sources = self._rank_dominant_sources(
            topic_targets=topic_targets,
            fragment_targets=fragment_targets,
            all_topic_keys=all_topic_keys,
            candidate_fragments=candidate_fragments,
        )

        # ==========================================
        # PHASE 4: AGGREGATE
        # ==========================================
        # Collect exclusion clues applied
        exclusions_applied: List[AssociationClue] = [
            c for c in clues if c.clue_type == ClueType.EXCLUSION
        ]

        # Deduplicate clue lists (by identity via token tuples)
        all_positive = self._dedupe_clues(topic_positive_clues + frag_positive_clues)
        all_negative = self._dedupe_clues(topic_negative_clues + frag_negative_clues)

        # Cross-context = any target is_cross_topic
        is_cross_context = (
            any(t.is_cross_topic for t in topic_targets)
            or any(t.is_cross_topic for t in fragment_targets)
        )

        # Include composite sources as additional cross-context signal
        if composite_analysis and composite_analysis.is_composite:
            is_cross_context = True

        result = AssociationAnalysis(
            topic_associations=topic_targets,
            fragment_associations=fragment_targets,
            dominant_sources=dominant_sources,
            is_cross_context=is_cross_context,
            positive_clues_used=all_positive,
            negative_clues_used=all_negative,
            exclusions_applied=exclusions_applied,
            rejected_targets=result.rejected_targets,
            reason=(
                f"topic_associations={len(topic_targets)}, "
                f"fragment_associations={len(fragment_targets)}, "
                f"dominant_sources={len(dominant_sources)}, "
                f"cross_context={is_cross_context}"
            ),
        )

        return result

    # ==========================================
    # TOPIC ASSOCIATION SCORING
    # ==========================================

    def _score_topic_association(
        self,
        key: TopicKey,
        signal_tokens: Set[str],
        clues: List[AssociationClue],
        exclusion_tokens: Set[str],
        primary_topic_key_id: Optional[str],
    ) -> Tuple[Optional[AssociationTarget], List[AssociationClue], List[AssociationClue]]:
        """
        Score satu TopicKey sebagai candidate association target.

        Returns:
            (target, positive_clues_used, negative_clues_used)
            target = None jika di bawah threshold / family check gagal.
        """
        # === Base evidence ===
        # Token overlap dengan TopicKey token_profile
        key_tokens = set(key.token_profile.keys()) if key.token_profile else set()
        if not key_tokens:
            return None, [], []

        overlap = signal_tokens.intersection(key_tokens)
        token_ratio = len(overlap) / len(key_tokens) if key_tokens else 0.0

        # Floor: MIN_ASSOCIATION_TOKEN_OVERLAP
        if token_ratio < self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP:
            return None, [], []

        # === Evidence families ===
        evidence_names: List[str] = ["token"]
        families: Set[EvidenceFamily] = {EvidenceFamily.LEXICAL}

        # Helper overlap (CONTEXTUAL family)
        helper_text = (
            (key.aggregated_helper_description or "")
            + " "
            + (key.helper_description or "")
        ).strip()
        helper_tokens = self._tokenize(helper_text)
        helper_ratio = 0.0
        if helper_tokens:
            helper_ratio = (
                len(signal_tokens.intersection(helper_tokens)) / len(helper_tokens)
            )
            if helper_ratio >= self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP:
                evidence_names.append("helper")
                families.add(EvidenceFamily.CONTEXTUAL)

        # Co-occurrence (LEXICAL family, same as token)
        cooc_profile = key.co_occurrence_profile or {}
        cooc_score = 0.0
        if cooc_profile:
            total_w = sum(cooc_profile.values())
            matched_w = sum(cooc_profile.get(t, 0) for t in signal_tokens)
            if total_w > 0:
                cooc_score = matched_w / total_w
                if cooc_score >= self._evidence_cfg.EVIDENCE_SUPPORT_THRESHOLD:
                    if "co_occurrence" not in evidence_names:
                        evidence_names.append("co_occurrence")

        # === Family gate ===
        if len(families) < self._cfg.MIN_ASSOCIATION_EVIDENCE_FAMILIES:
            return None, [], []

        # === Score composition ===
        base_score = max(token_ratio, helper_ratio, cooc_score)

        # Determine if cross-topic (selalu True di sini karena primary di-skip caller)
        is_cross = primary_topic_key_id is not None and key.key_id != primary_topic_key_id

        # Cross-topic bonus
        if is_cross:
            base_score = min(1.0, base_score + self._cfg.POSITIVE_CLUE_BONUS * 0.5)

        # Apply clues
        adjusted_score, pos_used, neg_used = self._apply_clues(
            base_score=base_score,
            target_id=key.key_id,
            target_tokens=key_tokens,
            target_helper_tokens=helper_tokens,
            clues=clues,
        )

        # Hard rejection via exclusion tokens
        if exclusion_tokens and exclusion_tokens.intersection(key_tokens):
            # Drop score significantly (EXCLUSION_PENALTY = 1.0 conceptually)
            adjusted_score = max(0.0, adjusted_score - self._cfg.EXCLUSION_PENALTY)

        # === Threshold check ===
        threshold = (
            self._cfg.CROSS_TOPIC_ASSOCIATION_THRESHOLD
            if is_cross
            else self._cfg.TOPIC_ASSOCIATION_THRESHOLD
        )

        if adjusted_score < threshold:
            return None, [], []

        # === Build target ===
        relationship_type = (
            RelationshipType.RELATED_TO
            if is_cross
            else RelationshipType.REFERENCES
        )

        target = AssociationTarget(
            target_id=key.key_id,
            target_type=GraphNodeType.TOPIC_KEY,
            score=adjusted_score,
            relationship_type=relationship_type,
            evidence=evidence_names,
            is_cross_topic=is_cross,
            primary_clue=pos_used[0] if pos_used else None,
        )

        return target, pos_used, neg_used

    # ==========================================
    # FRAGMENT ASSOCIATION SCORING
    # ==========================================

    def _score_fragment_association(
        self,
        frag: FragmentMetadata,
        signal_tokens: Set[str],
        clues: List[AssociationClue],
        exclusion_tokens: Set[str],
        primary_topic_key_id: Optional[str],
    ) -> Tuple[Optional[AssociationTarget], List[AssociationClue], List[AssociationClue]]:
        """
        Score satu FragmentMetadata sebagai candidate association target.

        Returns:
            (target, positive_clues_used, negative_clues_used)
            target = None jika di bawah threshold / family check gagal.
        """
        # === Base evidence ===
        frag_tokens = set(frag.token_profile.keys()) if frag.token_profile else set()

        # Token overlap
        token_ratio = 0.0
        if frag_tokens:
            overlap = signal_tokens.intersection(frag_tokens)
            token_ratio = len(overlap) / len(frag_tokens)

        # Fragment identity overlap (key + helper + aggregated)
        identity_text = " ".join(
            t for t in (
                frag.fragment_key,
                frag.fragment_helper_description,
                frag.aggregated_helper_description,
            ) if t
        )
        identity_tokens = self._tokenize(identity_text)
        identity_ratio = 0.0
        if identity_tokens:
            identity_ratio = (
                len(signal_tokens.intersection(identity_tokens)) / len(identity_tokens)
            )

        # Floor: at least one signal above MIN_ASSOCIATION_TOKEN_OVERLAP
        if (token_ratio < self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP
                and identity_ratio < self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP):
            return None, [], []

        # === Evidence families ===
        evidence_names: List[str] = []
        families: Set[EvidenceFamily] = set()

        if token_ratio >= self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP:
            evidence_names.append("token")
            families.add(EvidenceFamily.LEXICAL)

        if identity_ratio >= self._cfg.MIN_ASSOCIATION_TOKEN_OVERLAP:
            if "identity" not in evidence_names:
                evidence_names.append("identity")
            families.add(EvidenceFamily.CONTEXTUAL)

        # Co-occurrence
        cooc_profile = frag.co_occurrence_profile or {}
        cooc_score = 0.0
        if cooc_profile:
            total_w = sum(cooc_profile.values())
            matched_w = sum(cooc_profile.get(t, 0) for t in signal_tokens)
            if total_w > 0:
                cooc_score = matched_w / total_w
                if cooc_score >= self._evidence_cfg.EVIDENCE_SUPPORT_THRESHOLD:
                    evidence_names.append("co_occurrence")
                    families.add(EvidenceFamily.LEXICAL)

        # === Family gate ===
        if len(families) < self._cfg.MIN_ASSOCIATION_EVIDENCE_FAMILIES:
            return None, [], []

        # === Score composition ===
        base_score = max(token_ratio, identity_ratio, cooc_score)

        # Cross-topic check
        is_cross = (
            primary_topic_key_id is not None
            and frag.topic_key_id != primary_topic_key_id
        )

        if is_cross:
            base_score = min(1.0, base_score + self._cfg.POSITIVE_CLUE_BONUS * 0.5)

        # Apply clues
        # Combine frag_tokens + identity_tokens untuk clue matching
        all_frag_tokens = frag_tokens.union(identity_tokens)
        adjusted_score, pos_used, neg_used = self._apply_clues(
            base_score=base_score,
            target_id=frag.fragment_id,
            target_tokens=all_frag_tokens,
            target_helper_tokens=identity_tokens,
            clues=clues,
        )

        # Hard rejection via exclusion tokens
        if exclusion_tokens and exclusion_tokens.intersection(all_frag_tokens):
            adjusted_score = max(0.0, adjusted_score - self._cfg.EXCLUSION_PENALTY)

        # === Threshold check ===
        threshold = (
            self._cfg.CROSS_TOPIC_ASSOCIATION_THRESHOLD
            if is_cross
            else self._cfg.FRAGMENT_ASSOCIATION_THRESHOLD
        )

        if adjusted_score < threshold:
            return None, [], []

        # === Build target ===
        relationship_type = (
            RelationshipType.REFERENCES if is_cross else RelationshipType.RELATED_TO
        )

        target = AssociationTarget(
            target_id=frag.fragment_id,
            target_type=GraphNodeType.FRAGMENT,
            score=adjusted_score,
            relationship_type=relationship_type,
            evidence=evidence_names,
            is_cross_topic=is_cross,
            primary_clue=pos_used[0] if pos_used else None,
        )

        return target, pos_used, neg_used

    # ==========================================
    # CLUE APPLICATION
    # ==========================================

    def _apply_clues(
        self,
        base_score: float,
        target_id: str,
        target_tokens: Set[str],
        target_helper_tokens: Set[str],
        clues: List[AssociationClue],
    ) -> Tuple[float, List[AssociationClue], List[AssociationClue]]:
        """
        Apply POSITIVE/NEGATIVE clues ke base score.

        Returns:
            (adjusted_score, positive_clues_used, negative_clues_used)
        """
        score = base_score
        pos_used: List[AssociationClue] = []
        neg_used: List[AssociationClue] = []

        all_target_tokens = target_tokens.union(target_helper_tokens)
        if not all_target_tokens:
            return score, pos_used, neg_used

        for clue in clues:
            if clue.clue_type == ClueType.EXCLUSION:
                # EXCLUSION di-handle terpisah oleh caller
                continue

            # Cek apakah clue "menyentuh" target ini
            clue_tokens = set(clue.tokens or [])
            if not clue_tokens:
                continue
            overlap = clue_tokens.intersection(all_target_tokens)
            if not overlap:
                continue

            # Direct target_id match
            if clue.rejected_target_ids and target_id in clue.rejected_target_ids:
                # Treat as EXCLUSION for this specific target
                score = max(0.0, score - self._cfg.EXCLUSION_PENALTY)
                neg_used.append(clue)
                continue

            # Apply bonus/penalty
            strength = clue.strength if clue.strength > 0 else 0.5
            overlap_ratio = len(overlap) / max(len(clue_tokens), 1)

            if clue.clue_type == ClueType.POSITIVE:
                bonus = self._cfg.POSITIVE_CLUE_BONUS * strength * overlap_ratio
                score = min(1.0, score + bonus)
                pos_used.append(clue)

            elif clue.clue_type == ClueType.NEGATIVE:
                penalty = self._cfg.NEGATIVE_CLUE_PENALTY * strength * overlap_ratio
                score = max(0.0, score - penalty)
                neg_used.append(clue)

        return score, pos_used, neg_used

    # ==========================================
    # DOMINANCE RANKING
    # ==========================================

    def _rank_dominant_sources(
        self,
        topic_targets: List[AssociationTarget],
        fragment_targets: List[AssociationTarget],
        all_topic_keys: List[TopicKey],
        candidate_fragments: List[FragmentMetadata],
    ) -> List[DominantAssociation]:
        """
        Rank semua associations → tentukan PRIMARY/SECONDARY.

        Aturan:
            - Gabung topic + fragment targets
            - Sort by score descending
            - PRIMARY: score >= MIN_PRIMARY_SCORE AND
                      margin vs runner-up >= DOMINANT_SOURCE_MARGIN
            - SECONDARY: sisanya (di atas threshold tapi tidak PRIMARY)
            - Limit ke MAX_DOMINANT_SOURCES

        Returns list DominantAssociation terurut (PRIMARY dulu, lalu SECONDARY).
        """
        all_targets = list(topic_targets) + list(fragment_targets)
        if not all_targets:
            return []

        # Sort by score descending
        all_targets.sort(key=lambda t: t.score, reverse=True)

        # Build lookup untuk label extraction
        topic_map = {k.key_id: k for k in all_topic_keys}
        frag_map = {f.fragment_id: f for f in candidate_fragments}

        dominant: List[DominantAssociation] = []

        for idx, target in enumerate(all_targets):
            if len(dominant) >= self._cfg.MAX_DOMINANT_SOURCES:
                break

            # Determine role
            runner_up_score = (
                all_targets[idx + 1].score if idx + 1 < len(all_targets) else 0.0
            )
            margin = target.score - runner_up_score

            is_primary = (
                target.score >= self._cfg.MIN_PRIMARY_SCORE
                and margin >= self._cfg.DOMINANT_SOURCE_MARGIN
                and idx == 0  # hanya rank 1 yang boleh PRIMARY
            )

            role = AssociationRole.PRIMARY if is_primary else AssociationRole.SECONDARY

            label = self._extract_target_label(
                target_id=target.target_id,
                target_type=target.target_type,
                topic_map=topic_map,
                frag_map=frag_map,
            )

            dominant.append(DominantAssociation(
                source_id=target.target_id,
                source_type=target.target_type,
                contribution_score=target.score,
                role=role,
                label=label,
                margin_over_runner_up=margin,
            ))

        return dominant

    # ==========================================
    # HELPERS
    # ==========================================

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text dengan filter MIN_TOKEN_LENGTH."""
        if not text:
            return set()
        min_len = self._signal_cfg.MIN_TOKEN_LENGTH
        raw = re.findall(r'\b\w+\b', text.lower())
        return {t for t in raw if len(t) >= min_len}

    def _dedupe_clues(self, clues: List[AssociationClue]) -> List[AssociationClue]:
        """Deduplicate clues by (clue_type, source, tokens-tuple)."""
        seen: Set[Tuple] = set()
        result: List[AssociationClue] = []
        for c in clues:
            key = (c.clue_type.value, c.source, tuple(sorted(c.tokens or [])))
            if key in seen:
                continue
            seen.add(key)
            result.append(c)
        return result

    def _extract_target_label(
        self,
        target_id: str,
        target_type: GraphNodeType,
        topic_map: Dict[str, TopicKey],
        frag_map: Dict[str, FragmentMetadata],
    ) -> str:
        """
        Ekstrak label human-readable untuk observability.

        TOPIC_KEY → title
        FRAGMENT  → fragment_key atau helper (fallback ke ID prefix)
        """
        if target_type == GraphNodeType.TOPIC_KEY:
            key = topic_map.get(target_id)
            if key is not None:
                return key.title or target_id[:12]
            return target_id[:12]

        if target_type == GraphNodeType.FRAGMENT:
            frag = frag_map.get(target_id)
            if frag is not None:
                if frag.fragment_key:
                    return frag.fragment_key
                if frag.fragment_helper_description:
                    return frag.fragment_helper_description[:60]
            return target_id[:12]

        return ""
