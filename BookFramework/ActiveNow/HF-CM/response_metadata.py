"""
response_metadata.py
LLM Response Metadata — Parse, Validate, Resolve (Batch B).

Tanggung jawab:
    1. ResponseMetadataParser    : raw dict → ResponseMetadata
    2. ResponseMetadataValidator : validasi partial (field-per-field)
    3. ResponseMetadataResolver  : putuskan pakai metadata vs fallback heuristic

Prinsip kunci (dari dokumentasi):
    LLM menentukan identitas fragment SEBELUM respons disimpan,
    bukan sistem menebak identitas itu dari heuristic SETELAH
    jawaban selesai.

    TETAPI LLM bisa salah / hallucinate / provider bisa tidak mendukung
    structured output. Karena itu:
        - Parser toleran terhadap malformed input
        - Validator mendukung partial-valid (sebagian field valid)
        - Resolver punya fallback ke FragmentIdentityExtractor

Alur pipeline (Batch G akan memanggil):
    raw_metadata (dict / None)
        ↓
    ResponseMetadataParser.parse()
        ↓
    ResponseMetadata (or None if malformed)
        ↓
    ResponseMetadataValidator.validate()
        ↓
    MetadataValidationResult
        ↓
    ResponseMetadataResolver.resolve()
        ↓
    MetadataDecision (USE_METADATA / PARTIAL_METADATA / FALLBACK)

Tidak ada stateful dependency. File ini tidak menyentuh storage
kecuali Resolver yang menerima storage sebagai constructor argument
untuk validasi topic/fragment hints (opsional).
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Any, List

from config import CONFIG
from models import ResponseMetadata, MetadataValidationResult


logger = logging.getLogger(__name__)


# ==========================================
# DECISION ENUM
# ==========================================

class MetadataDecision:
    """Keputusan resolver: pakai metadata, pakai sebagian, atau fallback."""
    USE_METADATA = "USE_METADATA"           # Semua field valid, pakai penuh
    PARTIAL_METADATA = "PARTIAL_METADATA"   # Sebagian valid, pakai yang valid
    FALLBACK = "FALLBACK"                   # Tidak ada yang valid, pakai heuristic


# ==========================================
# 1. PARSER
# ==========================================

class ResponseMetadataParser:
    """
    Parse raw metadata (biasanya dict dari JSON LLM) → ResponseMetadata.

    Toleransi:
        - None → return None
        - dict kosong → return None
        - dict dengan field hilang → return ResponseMetadata dengan default
        - dict dengan field salah tipe → skip field tersebut
        - field string → truncate ke MAX_*_LENGTH
        - field list non-list → treat as empty list
    """

    def __init__(self):
        self._cfg = CONFIG.response_metadata

    def parse(self, raw: Any) -> Optional[ResponseMetadata]:
        """
        Parse raw metadata menjadi ResponseMetadata.

        Returns None jika input tidak bisa di-parse sama sekali
        (None, bukan dict, atau dict kosong tanpa field penting).
        """
        if raw is None:
            return None

        if not isinstance(raw, dict):
            logger.debug(
                f"ResponseMetadataParser: raw bukan dict, tipe={type(raw).__name__}"
            )
            return None

        if not raw:
            return None

        # Ekstrak field string dengan truncation
        fragment_key = self._safe_str(
            raw.get("fragment_key", ""),
            self._cfg.MAX_FRAGMENT_KEY_LENGTH,
        )
        helper_description = self._safe_str(
            raw.get("helper_description", ""),
            self._cfg.MAX_HELPER_DESCRIPTION_LENGTH,
        )
        primary_topic_hint = self._safe_str(
            raw.get("primary_topic_hint", ""),
            self._cfg.MAX_TOPIC_HINT_LENGTH,
        )

        # Ekstrak field list dengan bounds
        positive_clues = self._safe_str_list(
            raw.get("positive_clues", []),
            self._cfg.MAX_POSITIVE_CLUES,
        )
        negative_clues = self._safe_str_list(
            raw.get("negative_clues", []),
            self._cfg.MAX_NEGATIVE_CLUES,
        )
        related_topic_hints = self._safe_str_list(
            raw.get("related_topic_hints", []),
            self._cfg.MAX_RELATED_TOPIC_HINTS,
        )
        related_fragment_hints = self._safe_str_list(
            raw.get("related_fragment_hints", []),
            self._cfg.MAX_RELATED_FRAGMENT_HINTS,
        )

        # Confidence: clamp ke [0.0, 1.0]
        raw_confidence = raw.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        # Metadata version
        metadata_version = self._safe_str(
            raw.get("metadata_version", self._cfg.CURRENT_METADATA_VERSION),
            max_length=10,
        )

        return ResponseMetadata(
            fragment_key=fragment_key,
            helper_description=helper_description,
            primary_topic_hint=primary_topic_hint,
            positive_clues=positive_clues,
            negative_clues=negative_clues,
            related_topic_hints=related_topic_hints,
            related_fragment_hints=related_fragment_hints,
            confidence=confidence,
            metadata_version=metadata_version,
        )

    def _safe_str(self, value: Any, max_length: int) -> str:
        """Konversi ke string aman dengan truncation."""
        if value is None:
            return ""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                return ""
        value = value.strip()
        if len(value) > max_length:
            value = value[:max_length].rstrip()
        return value

    def _safe_str_list(self, value: Any, max_items: int) -> List[str]:
        """Konversi ke list string aman dengan bound."""
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if not isinstance(item, str):
                try:
                    item = str(item)
                except Exception:
                    continue
            item = item.strip()
            if item:
                result.append(item)
            if len(result) >= max_items:
                break
        return result


# ==========================================
# 2. VALIDATOR
# ==========================================

class ResponseMetadataValidator:
    """
    Validasi ResponseMetadata field-per-field.

    Mendukung partial-valid (ALLOW_PARTIAL_METADATA=True):
        - fragment_identity_valid : fragment_key + helper_description OK
        - topic_hints_valid       : related_topic_hints OK (minimal 1 atau none)
        - fragment_hints_valid    : related_fragment_hints OK

    is_valid = True jika MINIMAL fragment_identity_valid True
    (identity adalah field paling penting).

    rejection_reasons mencatat alasan setiap field yang ditolak,
    untuk observability.
    """

    def __init__(self):
        self._cfg = CONFIG.response_metadata

    def validate(
        self,
        metadata: Optional[ResponseMetadata],
        known_topic_keys: Optional[List[Any]] = None,
        known_fragments: Optional[List[Any]] = None,
    ) -> MetadataValidationResult:
        """
        Validate metadata.

        Args:
            metadata: ResponseMetadata hasil parser (bisa None).
            known_topic_keys: list TopicKey untuk validasi hints (opsional).
            known_fragments: list FragmentMetadata untuk validasi hints (opsional).

        Returns:
            MetadataValidationResult dengan field per-validity.
        """
        if metadata is None:
            return MetadataValidationResult(
                metadata=None,
                is_valid=False,
                fragment_identity_valid=False,
                topic_hints_valid=False,
                fragment_hints_valid=False,
                rejection_reasons=["metadata is None"],
                validated_confidence=0.0,
            )

        rejection_reasons: List[str] = []

        # 1. Version check
        version_ok = self._check_version(metadata.metadata_version)
        if not version_ok:
            rejection_reasons.append(
                f"metadata_version '{metadata.metadata_version}' != "
                f"expected '{self._cfg.CURRENT_METADATA_VERSION}'"
            )

        # 2. Fragment identity (fragment_key + helper)
        fragment_identity_valid = self._check_fragment_identity(
            metadata, rejection_reasons
        )

        # 3. Confidence threshold
        confidence_ok = metadata.confidence >= self._cfg.MIN_METADATA_CONFIDENCE
        if not confidence_ok:
            rejection_reasons.append(
                f"confidence {metadata.confidence:.2f} < "
                f"MIN_METADATA_CONFIDENCE {self._cfg.MIN_METADATA_CONFIDENCE:.2f}"
            )

        # 4. Topic hints validation
        topic_hints_valid = self._check_topic_hints(
            metadata, known_topic_keys, rejection_reasons
        )

        # 5. Fragment hints validation
        fragment_hints_valid = self._check_fragment_hints(
            metadata, known_fragments, rejection_reasons
        )

        # Overall validity:
        #   - confidence harus ok
        #   - fragment_identity harus valid (minimal ada fragment_key)
        #   - version warning tidak membuat invalid
        is_valid = (
            confidence_ok
            and fragment_identity_valid
        )

        # Partial-valid: jika ALLOW_PARTIAL_METADATA dan identity valid
        # tapi confidence rendah, tetap izinkan sebagian.
        if not is_valid and self._cfg.ALLOW_PARTIAL_METADATA:
            if fragment_identity_valid:
                is_valid = True

        # Validated confidence: bisa turun jika ada penolakan
        validated_confidence = metadata.confidence
        if rejection_reasons:
            penalty = 0.05 * len(rejection_reasons)
            validated_confidence = max(0.0, metadata.confidence - penalty)

        return MetadataValidationResult(
            metadata=metadata,
            is_valid=is_valid,
            fragment_identity_valid=fragment_identity_valid,
            topic_hints_valid=topic_hints_valid,
            fragment_hints_valid=fragment_hints_valid,
            rejection_reasons=rejection_reasons,
            validated_confidence=validated_confidence,
        )

    def _check_version(self, version: str) -> bool:
        """Check metadata_version matches current."""
        return version == self._cfg.CURRENT_METADATA_VERSION

    def _check_fragment_identity(
        self,
        metadata: ResponseMetadata,
        rejection_reasons: List[str],
    ) -> bool:
        """
        Validasi fragment identity (fragment_key + helper_description).

        Syarat minimal: fragment_key tidak kosong.
        Helper boleh kosong.
        """
        fragment_key = (metadata.fragment_key or "").strip()
        helper = (metadata.helper_description or "").strip()

        if not fragment_key:
            rejection_reasons.append("fragment_key is empty")
            return False

        # Cek karakter bermakna (minimal 1 alfanumerik atau CJK)
        if not re.search(r'\w', fragment_key, re.UNICODE):
            rejection_reasons.append(
                f"fragment_key '{fragment_key}' has no word characters"
            )
            return False

        # Helper description (optional): jika ada, cek substansi
        if helper:
            if not re.search(r'\w', helper, re.UNICODE):
                rejection_reasons.append(
                    f"helper_description '{helper}' has no word characters"
                )
                return False

        return True

    def _check_topic_hints(
        self,
        metadata: ResponseMetadata,
        known_topic_keys: Optional[List[Any]],
        rejection_reasons: List[str],
    ) -> bool:
        """
        Validasi related_topic_hints.

        Empty list = valid (LLM tidak punya hint).
        Non-empty list:
            - Jika REQUIRE_KNOWN_TARGET_FOR_HINT: cek semua match known topic
            - Jika tidak: terima apa adanya

        Returns True jika valid.
        """
        hints = metadata.related_topic_hints or []
        if not hints:
            return True

        if not self._cfg.REQUIRE_KNOWN_TARGET_FOR_HINT:
            # No strict matching required
            return True

        if known_topic_keys is None:
            # Cannot validate; accept as is
            return True

        # Build lookup: title → TopicKey (case-insensitive)
        known_titles = {
            (k.title or "").lower().strip()
            for k in known_topic_keys
            if hasattr(k, "title") and k.title
        }

        unmatched = []
        for hint in hints:
            hint_lower = hint.lower().strip()
            if hint_lower not in known_titles:
                unmatched.append(hint)

        if unmatched:
            rejection_reasons.append(
                f"unknown topic hints: {unmatched}"
            )
            # Partial rejection: jika ada yang unmatched, return False
            return False

        return True

    def _check_fragment_hints(
        self,
        metadata: ResponseMetadata,
        known_fragments: Optional[List[Any]],
        rejection_reasons: List[str],
    ) -> bool:
        """
        Validasi related_fragment_hints.

        Empty list = valid.
        Non-empty list:
            - Jika REQUIRE_KNOWN_TARGET_FOR_HINT: cek match
            - Jika tidak: accept
        """
        hints = metadata.related_fragment_hints or []
        if not hints:
            return True

        if not self._cfg.REQUIRE_KNOWN_TARGET_FOR_HINT:
            return True

        if known_fragments is None:
            return True

        # Build lookup dari fragment_id dan fragment_key
        known_ids = {
            f.fragment_id
            for f in known_fragments
            if hasattr(f, "fragment_id") and f.fragment_id
        }
        known_keys = {
            (f.fragment_key or "").lower().strip()
            for f in known_fragments
            if hasattr(f, "fragment_key") and f.fragment_key
        }

        unmatched = []
        for hint in hints:
            hint_stripped = hint.strip()
            hint_lower = hint_stripped.lower()
            # Match by ID or by fragment_key
            if hint_stripped in known_ids or hint_lower in known_keys:
                continue
            unmatched.append(hint)

        if unmatched:
            rejection_reasons.append(
                f"unknown fragment hints: {unmatched}"
            )
            return False

        return True


# ==========================================
# 3. RESOLVER
# ==========================================

@dataclass
class ResolverDecision:
    """Hasil keputusan resolver."""
    decision: str  # MetadataDecision constant
    metadata: Optional[ResponseMetadata]
    validation: Optional[MetadataValidationResult]
    reason: str = ""
    # Fields yang boleh dipakai (subset of metadata jika partial)
    usable_fragment_key: str = ""
    usable_helper_description: str = ""
    usable_positive_clues: List[str] = None
    usable_negative_clues: List[str] = None
    usable_topic_hints: List[str] = None
    usable_fragment_hints: List[str] = None

    def __post_init__(self):
        if self.usable_positive_clues is None:
            self.usable_positive_clues = []
        if self.usable_negative_clues is None:
            self.usable_negative_clues = []
        if self.usable_topic_hints is None:
            self.usable_topic_hints = []
        if self.usable_fragment_hints is None:
            self.usable_fragment_hints = []


class ResponseMetadataResolver:
    """
    Putuskan: pakai metadata LLM atau fallback ke heuristic.

    Decision flow:
        metadata None?              → FALLBACK
        validation.is_valid=False?  → FALLBACK (atau PARTIAL jika allow_partial)
        validation fully valid?     → USE_METADATA
        validation partial valid?   → PARTIAL_METADATA (jika ALLOW_PARTIAL_METADATA)

    Resolver juga mengekstrak field-field yang boleh dipakai ke
    ResolverDecision.usable_*.
    """

    def __init__(self):
        self._cfg = CONFIG.response_metadata
        self._parser = ResponseMetadataParser()
        self._validator = ResponseMetadataValidator()

    def resolve(
        self,
        raw_metadata: Any,
        known_topic_keys: Optional[List[Any]] = None,
        known_fragments: Optional[List[Any]] = None,
    ) -> ResolverDecision:
        """
        Resolve raw metadata menjadi keputusan.

        Args:
            raw_metadata: raw dict dari LLM (bisa None).
            known_topic_keys: list TopicKey untuk validasi hints.
            known_fragments: list FragmentMetadata untuk validasi hints.

        Returns:
            ResolverDecision dengan decision (USE_METADATA/PARTIAL/FALLBACK)
            dan field-field usable.
        """
        # 1. Parse
        metadata = self._parser.parse(raw_metadata)
        if metadata is None:
            return ResolverDecision(
                decision=MetadataDecision.FALLBACK,
                metadata=None,
                validation=None,
                reason="raw metadata is None or not parseable",
            )

        # 2. Validate
        validation = self._validator.validate(
            metadata,
            known_topic_keys=known_topic_keys,
            known_fragments=known_fragments,
        )

        # 3. Decide
        if not validation.is_valid:
            # Cek apakah fragment_identity valid meski validation invalid
            # (kasus: confidence rendah tapi identity ok)
            if (self._cfg.ALLOW_PARTIAL_METADATA
                    and validation.fragment_identity_valid):
                return self._build_partial_decision(metadata, validation)
            return self._build_fallback_decision(metadata, validation)

        # Check partial vs full
        all_valid = (
            validation.fragment_identity_valid
            and validation.topic_hints_valid
            and validation.fragment_hints_valid
        )
        if all_valid and not validation.rejection_reasons:
            return self._build_use_decision(metadata, validation)
        else:
            return self._build_partial_decision(metadata, validation)

    def _build_use_decision(
        self,
        metadata: ResponseMetadata,
        validation: MetadataValidationResult,
    ) -> ResolverDecision:
        """Full use: semua field valid."""
        return ResolverDecision(
            decision=MetadataDecision.USE_METADATA,
            metadata=metadata,
            validation=validation,
            reason="metadata fully valid",
            usable_fragment_key=metadata.fragment_key,
            usable_helper_description=metadata.helper_description,
            usable_positive_clues=list(metadata.positive_clues or []),
            usable_negative_clues=list(metadata.negative_clues or []),
            usable_topic_hints=list(metadata.related_topic_hints or []),
            usable_fragment_hints=list(metadata.related_fragment_hints or []),
        )

    def _build_partial_decision(
        self,
        metadata: ResponseMetadata,
        validation: MetadataValidationResult,
    ) -> ResolverDecision:
        """Partial use: hanya field yang valid."""
        usable_topic_hints = (
            list(metadata.related_topic_hints or [])
            if validation.topic_hints_valid else []
        )
        usable_fragment_hints = (
            list(metadata.related_fragment_hints or [])
            if validation.fragment_hints_valid else []
        )

        return ResolverDecision(
            decision=MetadataDecision.PARTIAL_METADATA,
            metadata=metadata,
            validation=validation,
            reason=(
                f"partial valid: identity={validation.fragment_identity_valid}, "
                f"topic={validation.topic_hints_valid}, "
                f"fragment={validation.fragment_hints_valid}; "
                f"rejections={validation.rejection_reasons}"
            ),
            usable_fragment_key=(
                metadata.fragment_key
                if validation.fragment_identity_valid else ""
            ),
            usable_helper_description=(
                metadata.helper_description
                if validation.fragment_identity_valid else ""
            ),
            usable_positive_clues=(
                list(metadata.positive_clues or [])
                if validation.fragment_identity_valid else []
            ),
            usable_negative_clues=list(metadata.negative_clues or []),
            usable_topic_hints=usable_topic_hints,
            usable_fragment_hints=usable_fragment_hints,
        )

    def _build_fallback_decision(
        self,
        metadata: ResponseMetadata,
        validation: MetadataValidationResult,
    ) -> ResolverDecision:
        """Fallback: metadata tidak valid, pakai heuristic."""
        return ResolverDecision(
            decision=MetadataDecision.FALLBACK,
            metadata=metadata,
            validation=validation,
            reason=(
                f"metadata invalid; rejections={validation.rejection_reasons}"
            ),
            usable_fragment_key="",
            usable_helper_description="",
            usable_positive_clues=[],
            usable_negative_clues=list(metadata.negative_clues or []),
            usable_topic_hints=[],
            usable_fragment_hints=[],
        )
