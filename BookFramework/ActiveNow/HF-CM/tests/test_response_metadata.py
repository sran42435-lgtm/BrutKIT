"""
tests/test_response_metadata.py

[ASSOCIATION EXPANSION - Batch I.1] VALIDATION SUITE — 13 KONDISI

Menguji 3 komponen pipeline metadata LLM:
    1. ResponseMetadataParser   : raw dict → ResponseMetadata
    2. ResponseMetadataValidator: validasi partial (field-per-field)
    3. ResponseMetadataResolver : putuskan USE_METADATA / PARTIAL / FALLBACK

Semua test adalah UNIT TEST (tidak menyentuh storage / engine).
Tujuan: memverifikasi kontrak pipeline metadata independen dari integrasi.

Konfigurasi yang dipakai (dari config.py Batch A2):
    MIN_METADATA_CONFIDENCE = 0.45
    MAX_FRAGMENT_KEY_LENGTH = 80
    MAX_HELPER_DESCRIPTION_LENGTH = 200
    ALLOW_PARTIAL_METADATA = True
    REQUIRE_KNOWN_TARGET_FOR_HINT = False (default)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from config import CONFIG
from models import ResponseMetadata, MetadataValidationResult
from response_metadata import (
    ResponseMetadataParser,
    ResponseMetadataValidator,
    ResponseMetadataResolver,
    MetadataDecision,
)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def parser():
    return ResponseMetadataParser()


@pytest.fixture
def validator():
    return ResponseMetadataValidator()


@pytest.fixture
def resolver():
    return ResponseMetadataResolver()


def _valid_metadata_dict() -> dict:
    """Raw metadata dict yang valid (untuk testing parser/resolver)."""
    return {
        "fragment_key": "Modal Bisnis Kopi",
        "helper_description": "Pembahasan modal awal usaha kopi.",
        "primary_topic_hint": "Bisnis Kopi",
        "positive_clues": ["modal", "kopi", "usaha"],
        "negative_clues": [],
        "related_topic_hints": [],
        "related_fragment_hints": [],
        "confidence": 0.85,
        "metadata_version": "1",
    }


# ==========================================
# 1. PARSER TESTS
# ==========================================

def test_parser_valid_dict(parser: ResponseMetadataParser):
    """[UNIT] Parser menerima dict valid dan menghasilkan ResponseMetadata."""
    raw = _valid_metadata_dict()
    result = parser.parse(raw)

    assert result is not None
    assert isinstance(result, ResponseMetadata)
    assert result.fragment_key == "Modal Bisnis Kopi"
    assert result.helper_description == "Pembahasan modal awal usaha kopi."
    assert result.confidence == 0.85
    assert result.metadata_version == "1"
    assert result.positive_clues == ["modal", "kopi", "usaha"]


def test_parser_none_input(parser: ResponseMetadataParser):
    """[UNIT] Parser mengembalikan None jika input None."""
    assert parser.parse(None) is None


def test_parser_empty_dict(parser: ResponseMetadataParser):
    """[UNIT] Parser mengembalikan None untuk dict kosong."""
    assert parser.parse({}) is None


def test_parser_truncates_long_fields(parser: ResponseMetadataParser):
    """[UNIT] Parser memotong field yang melebihi MAX_*_LENGTH."""
    cfg = CONFIG.response_metadata
    long_key = "x" * (cfg.MAX_FRAGMENT_KEY_LENGTH + 50)
    long_helper = "y" * (cfg.MAX_HELPER_DESCRIPTION_LENGTH + 50)

    raw = {
        "fragment_key": long_key,
        "helper_description": long_helper,
        "confidence": 0.9,
    }
    result = parser.parse(raw)

    assert result is not None
    assert len(result.fragment_key) <= cfg.MAX_FRAGMENT_KEY_LENGTH
    assert len(result.helper_description) <= cfg.MAX_HELPER_DESCRIPTION_LENGTH


def test_parser_invalid_confidence_clamps(parser: ResponseMetadataParser):
    """[UNIT] Confidence di-clamp ke [0.0, 1.0] jika out-of-range."""
    raw = {
        "fragment_key": "Test",
        "confidence": 2.5,  # over 1.0
    }
    result = parser.parse(raw)
    assert result is not None
    assert result.confidence == 1.0

    raw_low = {"fragment_key": "Test", "confidence": -0.5}
    result_low = parser.parse(raw_low)
    assert result_low.confidence == 0.0


def test_parser_non_dict_returns_none(parser: ResponseMetadataParser):
    """[UNIT] Parser mengembalikan None untuk input non-dict."""
    assert parser.parse("not a dict") is None
    assert parser.parse(42) is None
    assert parser.parse([1, 2, 3]) is None


# ==========================================
# 2. VALIDATOR TESTS
# ==========================================

def test_validator_valid_metadata(validator: ResponseMetadataValidator):
    """[UNIT] Metadata valid menghasilkan is_valid=True."""
    metadata = ResponseMetadata(
        fragment_key="Modal Bisnis Kopi",
        helper_description="Pembahasan modal awal.",
        confidence=0.85,
    )
    result = validator.validate(metadata)

    assert isinstance(result, MetadataValidationResult)
    assert result.is_valid is True
    assert result.fragment_identity_valid is True
    assert result.validated_confidence > 0


def test_validator_low_confidence_partial(validator: ResponseMetadataValidator):
    """
    [UNIT] Confidence rendah + identity valid + ALLOW_PARTIAL_METADATA
    → is_valid=True (partial use case).
    """
    assert CONFIG.response_metadata.ALLOW_PARTIAL_METADATA is True

    metadata = ResponseMetadata(
        fragment_key="Modal",
        helper_description="Pembahasan modal.",
        confidence=0.2,  # di bawah MIN_METADATA_CONFIDENCE (0.45)
    )
    result = validator.validate(metadata)

    # Identity valid, jadi ALLOW_PARTIAL_METADATA menyelamatkan
    assert result.fragment_identity_valid is True
    assert result.is_valid is True
    # Confidence divalidasi lebih rendah dari aslinya
    assert result.validated_confidence < metadata.confidence or result.rejection_reasons


def test_validator_empty_fragment_key_invalid(validator: ResponseMetadataValidator):
    """[UNIT] Metadata tanpa fragment_key → fragment_identity_valid=False."""
    metadata = ResponseMetadata(
        fragment_key="",
        helper_description="Pembahasan tanpa key.",
        confidence=0.9,
    )
    result = validator.validate(metadata)

    assert result.fragment_identity_valid is False
    assert any("fragment_key" in r.lower() for r in result.rejection_reasons)


def test_validator_none_metadata(validator: ResponseMetadataValidator):
    """[UNIT] None input → is_valid=False dengan rejection."""
    result = validator.validate(None)
    assert result.is_valid is False
    assert result.metadata is None
    assert len(result.rejection_reasons) > 0


def test_validator_no_word_characters(validator: ResponseMetadataValidator):
    """[UNIT] fragment_key tanpa karakter word → invalid."""
    metadata = ResponseMetadata(
        fragment_key="!@#$%",
        helper_description="",
        confidence=0.9,
    )
    result = validator.validate(metadata)
    assert result.fragment_identity_valid is False


# ==========================================
# 3. RESOLVER TESTS
# ==========================================

def test_resolver_use_metadata_fully_valid(
    resolver: ResponseMetadataResolver,
    validator: ResponseMetadataValidator,
):
    """[UNIT] Metadata fully valid → USE_METADATA decision."""
    raw = _valid_metadata_dict()
    result = resolver.resolve(raw_metadata=raw)

    assert result.decision == MetadataDecision.USE_METADATA
    assert result.metadata is not None
    assert result.validation is not None
    assert result.usable_fragment_key == "Modal Bisnis Kopi"
    assert result.usable_helper_description == "Pembahasan modal awal usaha kopi."
    assert result.usable_positive_clues == ["modal", "kopi", "usaha"]


def test_resolver_partial_metadata(
    resolver: ResponseMetadataResolver,
):
    """[UNIT] Identity valid tapi confidence rendah → PARTIAL_METADATA."""
    raw = {
        "fragment_key": "Modal",
        "helper_description": "Pembahasan modal.",
        "confidence": 0.2,  # di bawah MIN_METADATA_CONFIDENCE
    }
    result = resolver.resolve(raw_metadata=raw)

    # ALLOW_PARTIAL_METADATA=True + identity valid → PARTIAL
    assert result.decision == MetadataDecision.PARTIAL_METADATA
    assert result.usable_fragment_key == "Modal"


def test_resolver_fallback_on_invalid_identity(
    resolver: ResponseMetadataResolver,
):
    """[UNIT] Metadata tanpa fragment_key → FALLBACK."""
    raw = {
        "fragment_key": "",
        "confidence": 0.9,
    }
    result = resolver.resolve(raw_metadata=raw)

    assert result.decision == MetadataDecision.FALLBACK
    assert result.usable_fragment_key == ""
    assert result.usable_helper_description == ""


def test_resolver_fallback_on_malformed_input(
    resolver: ResponseMetadataResolver,
):
    """[UNIT] Malformed input (non-dict) → FALLBACK."""
    result = resolver.resolve(raw_metadata="bukan dict")

    assert result.decision == MetadataDecision.FALLBACK
    assert result.metadata is None
    assert result.validation is None


def test_resolver_fallback_on_none_input(
    resolver: ResponseMetadataResolver,
):
    """[UNIT] None input → FALLBACK (provider tidak mendukung metadata)."""
    result = resolver.resolve(raw_metadata=None)

    assert result.decision == MetadataDecision.FALLBACK
    assert result.metadata is None
    assert result.reason is not None and "none" in result.reason.lower() or "parseable" in result.reason.lower()
    assert result.usable_fragment_key == ""


# ==========================================
# [END OF RESPONSE METADATA TEST SUITE]
# ==========================================
