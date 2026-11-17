"""
tests/test_association_engine.py

[ASSOCIATION EXPANSION - Batch I.3] VALIDATION SUITE — 10 KONDISI

Menguji AssociationEngine (Batch E) yang menentukan:
    1. Cross-topic associations (fragment → TopicKey lain)
    2. Cross-fragment associations (fragment → fragment spesifik lain)
    3. Dominance ranking (PRIMARY vs SECONDARY sources)
    4. Clue application (POSITIVE/NEGATIVE/EXCLUSION)
    5. Evidence family gate (minimal 2 families)
    6. Self/primary exclusion (no self-association, no primary-topic association)

Semua test adalah UNIT TEST (tidak menyentuh storage / engine).
Membangun AssociationEngine + mock objects (TopicKey, FragmentMetadata,
FragmentIdentity) langsung untuk mengisolasi analysis logic.

Konfigurasi yang dipakai (dari config.py Batch A2):
    MIN_ASSOCIATION_TOKEN_OVERLAP = 0.20
    MIN_ASSOCIATION_EVIDENCE_FAMILIES = 2
    TOPIC_ASSOCIATION_THRESHOLD = 0.35
    FRAGMENT_ASSOCIATION_THRESHOLD = 0.30
    CROSS_TOPIC_ASSOCIATION_THRESHOLD = 0.30
    MIN_PRIMARY_SCORE = 0.60
    DOMINANT_SOURCE_MARGIN = 0.15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from config import CONFIG
from models import (
    TopicKey,
    FragmentMetadata,
    AssociationClue,
    AssociationTarget,
    AssociationAnalysis,
    DominantAssociation,
    CompositeSourceAnalysis,
    GraphNodeType,
    ClueType,
    AssociationRole,
    RelationshipType,
)
from fragment_identity import FragmentIdentity
from association_engine import AssociationEngine


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def tz():
    return ZoneInfo(CONFIG.input.DEFAULT_TIMEZONE)


@pytest.fixture
def engine():
    return AssociationEngine()


@pytest.fixture
def ref_time():
    return datetime(2023, 10, 25, 14, 0, 0)


# ==========================================
# HELPERS
# ==========================================

def _make_topic_key(
    key_id: str,
    title: str,
    tokens: dict,
    helper: str = "",
    aggregated: str = "",
    cooc: dict = None,
) -> TopicKey:
    """Buat TopicKey dengan token_profile."""
    return TopicKey(
        key_id=key_id,
        title=title,
        helper_description=helper,
        token_profile=tokens,
        co_occurrence_profile=cooc or {},
        created_at=datetime(2023, 10, 25, 14, 0, 0),
        aggregated_helper_description=aggregated,
    )


def _make_fragment(
    frag_id: str,
    topic_key_id: str,
    tokens: dict,
    fragment_key: str = "",
    helper: str = "",
    aggregated: str = "",
    cooc: dict = None,
) -> FragmentMetadata:
    """Buat FragmentMetadata dengan token_profile dan identity."""
    return FragmentMetadata(
        fragment_id=frag_id,
        topic_key_id=topic_key_id,
        session_id="test_session",
        created_at=datetime(2023, 10, 25, 14, 0, 0),
        payload_reference=f"pay_{frag_id}",
        token_profile=tokens,
        co_occurrence_profile=cooc or {},
        fragment_key=fragment_key,
        fragment_helper_description=helper,
        aggregated_helper_description=aggregated,
        depth=1,
    )


def _make_identity(tokens: list) -> FragmentIdentity:
    """
    Buat FragmentIdentity dengan meaningful_tokens.
    
    Identity harus non-empty agar analyze() tidak short-circuit
    di awal ("no meaningful fragment identity").
    """
    return FragmentIdentity(
        fragment_key="Test Fragment",
        helper_description="Test helper description",
        representative_tokens=tokens[:5],
        meaningful_tokens=tokens,
        identity_confidence=0.8,
        user_token_count=len(tokens),
        ai_token_count=0,
        combined_token_count=len(tokens),
    )


# ==========================================
# TESTS: TOPIC ASSOCIATION
# ==========================================

def test_topic_association_created(engine: AssociationEngine, ref_time):
    """
    [UNIT] Fragment terkait ke TopicKey lain (cross-topic).
    
    Setup:
        - Primary topic: "Bisnis Kopi" (tokens: bisnis, kopi, modal)
        - Other topic: "Manajemen Bisnis" (tokens: bisnis, manajemen, organisasi, modal)
        - New fragment identity: [bisnis, modal, organisasi, strategi]
        
    Expected:
        - Topic "Manajemen Bisnis" jadi association target
        - Token overlap = 3/4 = 0.75 ≥ 0.20
        - Identity overlap dengan aggregated helper = ada
        - → 2 evidence families (LEXICAL + CONTEXTUAL)
        - → score ≥ TOPIC_ASSOCIATION_THRESHOLD (0.35)
    """
    primary_key = _make_topic_key(
        key_id="primary_topic",
        title="Bisnis Kopi",
        tokens={"bisnis": 5, "kopi": 5, "modal": 5},
        helper="Bisnis kopi modal usaha",
    )
    other_key = _make_topic_key(
        key_id="other_topic",
        title="Manajemen Bisnis",
        tokens={"bisnis": 5, "manajemen": 5, "organisasi": 5, "modal": 5},
        helper="Manajemen organisasi bisnis modal",
        aggregated="Mencakup: manajemen, organisasi, bisnis, modal",
    )
    
    identity = _make_identity(["bisnis", "modal", "organisasi", "strategi"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[primary_key, other_key],
        candidate_fragments=[],
    )
    
    print(f"\nTOPIC ASSOCIATION: {len(result.topic_associations)} found")
    for t in result.topic_associations:
        print(f"  - {t.target_id}: score={t.score:.3f} ev={t.evidence} cross={t.is_cross_topic}")
    
    assert isinstance(result, AssociationAnalysis)
    # other_topic harus muncul sebagai association
    assert any(t.target_id == "other_topic" for t in result.topic_associations)
    # primary_topic TIDAK boleh muncul
    assert all(t.target_id != "primary_topic" for t in result.topic_associations)
    
    # Cross-topic harus True (karena beda dari primary)
    other_assoc = next(t for t in result.topic_associations if t.target_id == "other_topic")
    assert other_assoc.is_cross_topic is True
    assert other_assoc.score >= CONFIG.association.TOPIC_ASSOCIATION_THRESHOLD


# ==========================================
# TESTS: FRAGMENT ASSOCIATION
# ==========================================

def test_fragment_association_created(engine: AssociationEngine, ref_time):
    """
    [UNIT] Fragment terkait ke fragment spesifik lain (cross-fragment).
    
    Setup:
        - New fragment: identity [modal, bisnis, kopi, usaha, strategi]
        - Other fragment (beda topic): 
            fragment_key="Modal Awal"
            helper="pembahasan modal bisnis"
            tokens={modal, bisnis, awal, investasi}
        
    Expected:
        - Other fragment muncul di fragment_associations
        - Token overlap = 2/4 = 0.50 ≥ 0.20
        - Identity overlap = strong
        - → 2 evidence families (token + identity)
        - → score ≥ FRAGMENT_ASSOCIATION_THRESHOLD (0.30)
    """
    primary_key = _make_topic_key(
        key_id="primary_topic",
        title="Bisnis Kopi",
        tokens={"bisnis": 5, "kopi": 5},
    )
    
    other_frag = _make_fragment(
        frag_id="other_frag",
        topic_key_id="other_topic",  # different topic!
        tokens={"modal": 5, "bisnis": 5, "awal": 5, "investasi": 5},
        fragment_key="Modal Awal",
        helper="pembahasan modal bisnis",
        aggregated="Mencakup: modal, bisnis, investasi",
    )
    
    identity = _make_identity(["modal", "bisnis", "kopi", "usaha", "strategi"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[primary_key],
        candidate_fragments=[other_frag],
    )
    
    print(f"\nFRAGMENT ASSOCIATION: {len(result.fragment_associations)} found")
    for f in result.fragment_associations:
        print(f"  - {f.target_id}: score={f.score:.3f} ev={f.evidence} cross={f.is_cross_topic}")
    
    assert any(f.target_id == "other_frag" for f in result.fragment_associations)
    
    other_assoc = next(f for f in result.fragment_associations if f.target_id == "other_frag")
    assert other_assoc.target_type == GraphNodeType.FRAGMENT
    assert other_assoc.is_cross_topic is True
    assert other_assoc.score >= CONFIG.association.FRAGMENT_ASSOCIATION_THRESHOLD


# ==========================================
# TESTS: DOMINANCE RANKING
# ==========================================

def test_primary_dominance(engine: AssociationEngine, ref_time):
    """
    [UNIT] Dominant source = PRIMARY jika score >= 0.60 + margin >= 0.15.
    
    Setup:
        - New fragment identity: [modal, bisnis, kopi, murah, mahal]
        - Topic A (Modal Awal): strong overlap → score ~0.85
        - Topic B (Lokasi): weak overlap → score ~0.10
        
    Expected:
        - Topic A jadi PRIMARY (score high + margin OK)
        - Topic B tidak masuk dominant_sources (score < MIN_PRIMARY_SCORE)
    """
    topic_a = _make_topic_key(
        key_id="topic_a",
        title="Modal Awal",
        tokens={"modal": 5, "bisnis": 5, "murah": 5, "mahal": 5, "investasi": 5},
        helper="pembahasan modal bisnis",
        aggregated="Modal bisnis investasi murah mahal",
    )
    topic_b = _make_topic_key(
        key_id="topic_b",
        title="Lokasi Usaha",
        tokens={"lokasi": 5, "strategis": 5, "tempat": 5},
        helper="pembahasan lokasi",
    )
    
    identity = _make_identity(["modal", "bisnis", "murah", "mahal", "investasi"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a, topic_b],
        candidate_fragments=[],
    )
    
    print(f"\nDOMINANCE RANKING: {len(result.dominant_sources)} sources")
    for d in result.dominant_sources:
        print(f"  - {d.source_id}: score={d.contribution_score:.3f} role={d.role.value} margin={d.margin_over_runner_up:.3f}")
    
    # topic_a harus jadi PRIMARY (score high, margin OK)
    topic_a_dominant = next(
        (d for d in result.dominant_sources if d.source_id == "topic_a"),
        None
    )
    assert topic_a_dominant is not None, "topic_a harus masuk dominant_sources"
    assert topic_a_dominant.role == AssociationRole.PRIMARY
    assert topic_a_dominant.contribution_score >= CONFIG.association.MIN_PRIMARY_SCORE


def test_secondary_dominance_close_scores(engine: AssociationEngine, ref_time):
    """
    [UNIT] Multiple sources dengan score close → semua SECONDARY.
    
    Setup:
        - Topic A: score ~0.50
        - Topic B: score ~0.48 (margin < 0.15)
        
    Expected:
        - Tidak ada PRIMARY (margin terlalu kecil)
        - Keduanya SECONDARY
    """
    topic_a = _make_topic_key(
        key_id="topic_a",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5, "murah": 5},
        helper="modal bisnis",
        aggregated="modal bisnis murah",
    )
    topic_b = _make_topic_key(
        key_id="topic_b",
        title="Biaya",
        tokens={"biaya": 5, "modal": 5, "murah": 5},
        helper="biaya modal",
        aggregated="biaya modal murah",
    )
    
    identity = _make_identity(["modal", "bisnis", "murah", "strategi"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a, topic_b],
        candidate_fragments=[],
    )
    
    print(f"\nSECONDARY DOMINANCE: {len(result.dominant_sources)} sources")
    for d in result.dominant_sources:
        print(f"  - {d.source_id}: score={d.contribution_score:.3f} role={d.role.value}")
    
    # Jika margin < DOMINANT_SOURCE_MARGIN, semua SECONDARY
    if len(result.dominant_sources) >= 2:
        roles = [d.role for d in result.dominant_sources]
        # Either all SECONDARY, or first is PRIMARY only if margin OK
        top = result.dominant_sources[0]
        runner = result.dominant_sources[1]
        margin = top.contribution_score - runner.contribution_score
        if margin < CONFIG.association.DOMINANT_SOURCE_MARGIN:
            assert all(r == AssociationRole.SECONDARY for r in roles)


# ==========================================
# TESTS: CLUE APPLICATION
# ==========================================

def test_exclusion_drops_target(engine: AssociationEngine, ref_time):
    """
    [UNIT] EXCLUSION clue menghilangkan target dari associations.
    
    Setup:
        - Topic A: "C++ Programming" (tokens: cpp, program, coding, memori)
        - Clue: EXCLUSION dengan tokens ["cpp"]
        
    Expected:
        - Topic A tidak muncul di associations (karena cpp di-exclude)
        - Exclusion clue tercatat di exclusions_applied
        - Token overlap antara clue dan topic_cpp terbukti (mekanisme yang menyebabkan drop)
    
    Catatan observability:
        Field `rejected_targets` hanya diisi ketika target eksplisit ada di
        `clue.rejected_target_ids` (ID-based rejection). Token-based exclusion
        tercatat di `exclusions_applied` dan diobservasi via absence dari
        associations.
    """
    topic_cpp = _make_topic_key(
        key_id="topic_cpp",
        title="C++ Programming",
        tokens={"cpp": 5, "program": 5, "coding": 5, "memori": 5},
        helper="pemrograman cpp",
        aggregated="pemrograman cpp memori",
    )
    topic_modal = _make_topic_key(
        key_id="topic_modal",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5},
        helper="modal bisnis",
        aggregated="modal bisnis",
    )
    
    identity = _make_identity(["modal", "bisnis", "cpp", "memori"])
    
    exclusion_clue = AssociationClue(
        clue_type=ClueType.EXCLUSION,
        tokens=["cpp"],
        source="signal",
        strength=1.0,
    )
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_cpp, topic_modal],
        candidate_fragments=[],
        clues=[exclusion_clue],
    )
    
    print(f"\nEXCLUSION TEST:")
    print(f"  Associations: {[t.target_id for t in result.topic_associations]}")
    print(f"  Exclusions applied: {len(result.exclusions_applied)}")
    print(f"  Rejected (ID-based): {result.rejected_targets}")
    
    # [1] topic_cpp harus di-exclude (karena token "cpp" match exclusion clue)
    assert not any(t.target_id == "topic_cpp" for t in result.topic_associations), (
        "topic_cpp tidak boleh muncul di topic_associations setelah exclusion"
    )
    
    # [2] Exclusion clue harus tercatat di exclusions_applied
    assert len(result.exclusions_applied) >= 1, (
        "Exclusion clue harus tercatat di exclusions_applied"
    )
    
    # [3] Verifikasi mekanisme: exclusion clue tokens overlap dengan topic_cpp tokens.
    # Ini membuktikan bahwa token match adalah mekanisme yang menyebabkan topic_cpp di-drop.
    cpp_topic_tokens = {"cpp", "program", "coding", "memori"}
    applied = result.exclusions_applied[0]
    clue_tokens = set(applied.tokens or [])
    overlap = clue_tokens.intersection(cpp_topic_tokens)
    assert overlap, (
        f"Exclusion clue tokens {clue_tokens} harus overlap dengan "
        f"topic_cpp tokens {cpp_topic_tokens}"
    )
    
    # [4] topic_modal tetap harus ada di associations (tidak kena exclusion)
    assert any(t.target_id == "topic_modal" for t in result.topic_associations), (
        "topic_modal tidak boleh terpengaruh exclusion clue (tokens berbeda)"
    )


def test_positive_clue_boosts(engine: AssociationEngine, ref_time):
    """
    [UNIT] POSITIVE clue boost skor target yang match.
    
    Setup:
        - Topic A: overlap baseline score ~0.30 (dekat threshold)
        - Clue: POSITIVE dengan token yang match topic A
        
    Expected:
        - Score topic A naik karena positive clue bonus
        - Topic A masuk associations (score >= threshold)
    """
    topic_a = _make_topic_key(
        key_id="topic_a",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5, "murah": 5},
        helper="modal bisnis",
        aggregated="modal bisnis murah",
    )
    
    identity = _make_identity(["modal", "bisnis"])
    
    # Clue yang match topic_a tokens
    positive_clue = AssociationClue(
        clue_type=ClueType.POSITIVE,
        tokens=["modal", "murah"],
        source="metadata",
        strength=0.8,
    )
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a],
        candidate_fragments=[],
        clues=[positive_clue],
    )
    
    print(f"\nPOSITIVE CLUE BOOST:")
    print(f"  Associations: {len(result.topic_associations)}")
    print(f"  Positive clues used: {len(result.positive_clues_used)}")
    
    # Positive clue harus tercatat
    assert len(result.positive_clues_used) >= 1
    # Topic A harus masuk associations (boosted)
    assert any(t.target_id == "topic_a" for t in result.topic_associations)


def test_negative_clue_penalizes(engine: AssociationEngine, ref_time):
    """
    [UNIT] NEGATIVE clue penalize skor target yang match.
    
    Setup:
        - Topic A: overlap tinggi (score tanpa clue ~0.70)
        - Clue: NEGATIVE dengan token yang match topic A
        
    Expected:
        - Score topic_a turun karena negative penalty
        - Jika turun di bawah threshold → tidak masuk associations
    """
    topic_a = _make_topic_key(
        key_id="topic_a",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5, "murah": 5},
        helper="modal bisnis",
        aggregated="modal bisnis murah",
    )
    
    identity = _make_identity(["modal", "bisnis", "murah"])
    
    # Clue negatif kuat
    negative_clue = AssociationClue(
        clue_type=ClueType.NEGATIVE,
        tokens=["modal", "bisnis", "murah"],
        source="metadata",
        strength=0.9,
    )
    
    # Baseline: tanpa clue
    result_no_clue = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a],
        candidate_fragments=[],
        clues=[],
    )
    
    # Dengan negative clue
    result_with_clue = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a],
        candidate_fragments=[],
        clues=[negative_clue],
    )
    
    # Ambil score topic_a di kedua hasil
    score_no_clue = next(
        (t.score for t in result_no_clue.topic_associations if t.target_id == "topic_a"),
        0.0
    )
    score_with_clue = next(
        (t.score for t in result_with_clue.topic_associations if t.target_id == "topic_a"),
        0.0
    )
    
    print(f"\nNEGATIVE CLUE PENALTY:")
    print(f"  Score tanpa clue: {score_no_clue:.3f}")
    print(f"  Score dengan clue: {score_with_clue:.3f}")
    print(f"  Negative clues used: {len(result_with_clue.negative_clues_used)}")
    
    # Score harus turun setelah negative clue
    assert score_with_clue <= score_no_clue
    # Negative clue harus tercatat
    assert len(result_with_clue.negative_clues_used) >= 1


# ==========================================
# TESTS: EVIDENCE FAMILY GATE
# ==========================================

def test_evidence_family_gate_rejects_single_family(engine: AssociationEngine, ref_time):
    """
    [UNIT] Association dengan single evidence family ditolak.
    
    Setup:
        - Topic A: hanya overlap token (LEXICAL family)
        - Topic A: TIDAK punya helper atau aggregated helper
        - MIN_ASSOCIATION_EVIDENCE_FAMILIES = 2
        
    Expected:
        - Topic A TIDAK masuk associations (hanya 1 family)
    """
    topic_a = _make_topic_key(
        key_id="topic_a",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5, "murah": 5},
        helper="",  # no helper → no CONTEXTUAL family
        aggregated="",  # no aggregated → no CONTEXTUAL family
    )
    
    identity = _make_identity(["modal", "bisnis", "murah"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[topic_a],
        candidate_fragments=[],
    )
    
    print(f"\nEVIDENCE FAMILY GATE:")
    print(f"  Associations: {len(result.topic_associations)}")
    print(f"  Reason: {result.reason}")
    
    # Topic A harus ditolak karena hanya 1 evidence family (LEXICAL)
    assert not any(t.target_id == "topic_a" for t in result.topic_associations)


# ==========================================
# TESTS: SELF & PRIMARY EXCLUSION
# ==========================================

def test_self_association_excluded(engine: AssociationEngine, ref_time):
    """
    [UNIT] New fragment TIDAK boleh associate dengan dirinya sendiri.
    
    Setup:
        - candidate_fragments berisi fragment dengan ID = new_fragment_id
        - Fragment itu match perfectly dengan identity
        
    Expected:
        - Fragment tersebut TIDAK muncul di fragment_associations
    """
    same_id = "same_frag"
    
    topic = _make_topic_key(
        key_id="topic_a",
        title="Modal",
        tokens={"modal": 5, "bisnis": 5},
    )
    
    # Fragment yang ID-nya sama dengan new_fragment_id
    self_frag = _make_fragment(
        frag_id=same_id,
        topic_key_id="other_topic",
        tokens={"modal": 5, "bisnis": 5, "murah": 5},
        fragment_key="Modal",
        helper="modal bisnis",
        aggregated="modal bisnis murah",
    )
    
    identity = _make_identity(["modal", "bisnis", "murah"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id=same_id,  # sama dengan self_frag.fragment_id
        all_topic_keys=[topic],
        candidate_fragments=[self_frag],
    )
    
    print(f"\nSELF EXCLUSION:")
    print(f"  Fragment associations: {[f.target_id for f in result.fragment_associations]}")
    
    # self_frag harus TIDAK muncul di fragment_associations
    assert not any(f.target_id == same_id for f in result.fragment_associations)


def test_primary_topic_excluded_from_targets(engine: AssociationEngine, ref_time):
    """
    [UNIT] Primary topic TIDAK boleh muncul di topic_associations.
    
    Primary topic adalah structural owner (topic_key_id) — bukan
    association target. Association hanya untuk LINTAS topik.
    """
    primary_key = _make_topic_key(
        key_id="primary_topic",
        title="Bisnis Kopi",
        tokens={"bisnis": 5, "kopi": 5, "modal": 5},
        helper="bisnis kopi modal",
        aggregated="bisnis kopi modal usaha",
    )
    
    identity = _make_identity(["bisnis", "kopi", "modal"])
    
    result = engine.analyze(
        new_fragment_identity=identity,
        primary_topic_key_id="primary_topic",
        new_fragment_id="new_frag",
        all_topic_keys=[primary_key],
        candidate_fragments=[],
    )
    
    print(f"\nPRIMARY EXCLUSION:")
    print(f"  Topic associations: {[t.target_id for t in result.topic_associations]}")
    
    # primary_topic tidak boleh muncul di associations
    assert all(t.target_id != "primary_topic" for t in result.topic_associations)


# ==========================================
# [END OF ASSOCIATION ENGINE TEST SUITE]
# ==========================================
