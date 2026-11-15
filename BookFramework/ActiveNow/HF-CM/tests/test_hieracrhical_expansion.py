"""
tests/test_hieracrhical_expansion.py

[HIERARCHICAL FRAGMENT EXPANSION] VALIDATION SUITE — 31 KONDISI (A - I)

File ini TERPISAH dari test_integration.py agar kedua suite bisa
diverifikasi independen dan tidak ada risiko regresi pada 40 test lama.

M1  Recursive Ambiguity Context Accumulation      (kondisi 1-6)
M2  Continuation Intent Protection                (kondisi 7-10)
M3  Hierarchical Identity & Helper Propagation    (kondisi 11-15)
M4  Cross-Key Composite Fragment                  (kondisi 23-26)
M5  Tree + Depth + Trend                          (kondisi 16-22, 27-31)

LABEL:
  [INTEGRATION] = melalui engine.process_input / storage nyata
  [UNIT]        = memanggil method langsung agar mekanisme diuji terisolasi
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import uuid
import dataclasses
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import CONFIG
from models import (
    InputEnvelope, DecisionType, PendingStatus, MemoryClassification,
    TopicKey, FragmentMetadata, EngineResult, CandidateScore,
    RetrievedMemory, TokenQuality, EvidenceFamily, TurnAnalysisResult,
    FragmentPlacementKind, TrendNodeType, ClarificationAccumulation,
)
from storage import MemoryStorage
from retrieval import RetrievalEngine, SignalExtractor, RetrievalResult
from compression import PayloadCompressor
from memory_engine import MemoryEngine
from context_builder import ContextBuilder
from llm_client import FakeLLMClient
import memory_engine as memory_engine_module
import retrieval as retrieval_module


# ==========================================
# FIXTURES (self-contained)
# ==========================================

@pytest.fixture(scope="function")
def test_db_path():
    db_file = f"test_hier_{uuid.uuid4().hex[:8]}.db"
    yield f"sqlite:///{db_file}"
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture(scope="function")
def engine(test_db_path):
    storage = MemoryStorage(db_url=test_db_path)
    retrieval = RetrievalEngine()
    compressor = PayloadCompressor()
    return MemoryEngine(storage, retrieval, compressor, ContextBuilder(), FakeLLMClient())


@pytest.fixture(scope="function")
def tz():
    return ZoneInfo(CONFIG.input.DEFAULT_TIMEZONE)


def create_envelope(session_id: str, text: str, tz: ZoneInfo,
                    time_offset_hours: int = 0) -> InputEnvelope:
    base_time = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    return InputEnvelope(
        input_id=str(uuid.uuid4()),
        session_id=session_id,
        raw_text=text,
        normalized_text=text.lower(),
        timestamp=base_time + timedelta(hours=time_offset_hours),
        timezone=str(tz),
    )


def _mk_key(engine, tz, key_id, title, desc, tokens, cooc=None, agg=""):
    """Insert TopicKey langsung ke storage."""
    key = TopicKey(
        key_id=key_id, title=title, helper_description=desc,
        created_at=datetime.now(tz),
        token_profile=dict(tokens),
        co_occurrence_profile=dict(cooc or {}),
        aggregated_helper_description=agg,
    )
    with engine.storage.transaction() as tx:
        engine.storage.save_topic_key(tx, key)
    return key


def _mk_frag(engine, tz, fid, key_id, session_id, tokens,
             key_name="", helper="", parent=None, depth=0,
             when=None, cooc=None, strength=0.0):
    """Insert FragmentMetadata + payload langsung ke storage."""
    stamp = when or datetime.now(tz)
    engine.storage.ensure_session_exists(session_id, stamp)
    compressor = PayloadCompressor()
    payload = compressor.compress(f"Exact payload for {fid}")
    frag = FragmentMetadata(
        fragment_id=fid, topic_key_id=key_id, session_id=session_id,
        created_at=stamp, temporal_index=stamp,
        payload_reference=payload.payload_id, content_hash=payload.checksum,
        token_profile=dict(tokens), co_occurrence_profile=dict(cooc or {}),
        fragment_key=key_name, fragment_helper_description=helper,
        aggregated_helper_description=helper,
        parent_fragment_id=parent, depth=depth,
        relationship_strength=strength,
    )
    with engine.storage.transaction() as tx:
        engine.storage.save_payload(tx, payload)
        engine.storage.save_fragment(tx, frag)
    return frag


def _three_coffee_keys(engine, tz):
    """Tiga topik kopi yang saling mirip: bisnis / kedai / cpp kasir."""
    _mk_key(engine, tz, "m1_bisnis", "Bisnis Kopi",
            "Rencana bisnis kopi modal usaha",
            {"kopi": 10, "bisnis": 10, "modal": 10, "usaha": 10})
    _mk_key(engine, tz, "m1_kedai", "Kedai Kopi",
            "Rencana kedai kopi lokasi menu",
            {"kopi": 10, "kedai": 10, "lokasi": 10, "menu": 10})
    _mk_key(engine, tz, "m1_cpp", "C++ Kasir Kopi",
            "Program kasir kopi cpp",
            {"kopi": 10, "cpp": 10, "kasir": 10})


def _composite_setup(engine, tz):
    """Dua TopicKey berbeda, masing-masing satu fragment sumber."""
    _mk_key(engine, tz, "cmp_a", "Kopi Baru", "Resep kopi baru",
            {"resep": 10, "kopi": 10})
    _mk_key(engine, tz, "cmp_b", "Bisnis Kopi", "Bisnis kopi murah",
            {"resep": 10, "murah": 10, "bahan": 10})
    _mk_frag(engine, tz, "src_a", "cmp_a", "s_cmp",
             {"resep": 5, "kopi": 5}, key_name="Resep Kopi",
             helper="resep kopi", depth=0)
    _mk_frag(engine, tz, "src_b", "cmp_b", "s_cmp",
             {"resep": 5, "murah": 5, "bahan": 5}, key_name="Resep Murah",
             helper="resep murah bahan", depth=0)


# ==========================================
# A. CLARIFICATION ACCUMULATION (kondisi 1-6)
# ==========================================

def test_m1_a1_initial_ambiguous_creates_pending(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 1: input ambiguous -> beberapa kandidat + pending."""
    session_id = "m1_a1"
    _three_coffee_keys(engine, tz)

    env = create_envelope(session_id, "Bahas kopi.", tz)
    res = engine.process_input(env)

    assert res.decision.decision_type == DecisionType.AMBIGUOUS
    assert len(res.candidates) >= 2
    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None
    assert len(pending.candidate_keys) >= 2
    assert pending.clarification_history == []
    assert pending.combined_clarification_text == "Bahas kopi."


def test_m1_a2_history_accumulates_and_combines(engine: MemoryEngine, tz: ZoneInfo):
    """
    [UNIT] Kondisi 2: jawaban klarifikasi DIGABUNG dengan original,
    bukan menggantikan.
    """
    session_id = "m1_a2"
    _three_coffee_keys(engine, tz)
    engine.process_input(create_envelope(session_id, "Bahas kopi.", tz))
    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None

    env1 = create_envelope(session_id, "yang rencana", tz, time_offset_hours=1)
    ctx1 = engine._build_clarification_accumulation(env1, pending)
    print(f"\nCOMBINED R1: {ctx1.combined_text}")
    assert "Bahas kopi." in ctx1.combined_text
    assert "yang rencana" in ctx1.combined_text
    assert ctx1.clarification_history == ["yang rencana"]

    pending.clarification_history = ctx1.clarification_history
    pending.clarification_round = 2
    env2 = create_envelope(session_id, "yang untuk usaha", tz, time_offset_hours=2)
    ctx2 = engine._build_clarification_accumulation(env2, pending)
    print(f"COMBINED R2: {ctx2.combined_text}")
    assert len(ctx2.clarification_history) == 2
    assert "Bahas kopi." in ctx2.combined_text
    assert "yang rencana" in ctx2.combined_text
    assert "yang untuk usaha" in ctx2.combined_text


def test_m1_a3_weak_candidate_dropped(engine: MemoryEngine, tz: ZoneInfo):
    """
    [UNIT] Kondisi 3: A=0.81, B=0.79, C=0.20 -> C gugur.
    floor = max(0.35, 0.81*0.60=0.486) = 0.486; C=0.20 < 0.486.
    """
    cands = [
        CandidateScore(key_id="A", title="Bisnis Kopi", final_score=0.81),
        CandidateScore(key_id="B", title="Kedai Kopi", final_score=0.79),
        CandidateScore(key_id="C", title="C++ Kasir Kopi", final_score=0.20),
    ]
    ctx = ClarificationAccumulation(
        original_input="lanjut topik kopi kemarin",
        clarification_history=["yang rencana kopi"],
        combined_text="lanjut topik kopi kemarin yang rencana kopi",
        round_number=1,
    )
    kept = engine.retrieval._narrow_candidates(cands, ctx)
    kept_ids = [c.key_id for c in kept]
    print(f"\nNARROWED: {kept_ids}")
    assert "C" not in kept_ids
    assert "A" in kept_ids and "B" in kept_ids
    assert len(kept) == 2


def test_m1_a4_full_history_resolves(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 4: gabungan original + jawaban memilih kandidat tepat."""
    session_id = "m1_a4"
    _three_coffee_keys(engine, tz)
    engine.process_input(create_envelope(session_id, "Bahas kopi.", tz))
    assert engine.storage.get_active_pending_state(session_id) is not None

    env1 = create_envelope(session_id, "yang bisnis kopi modal usaha", tz,
                           time_offset_hours=1)
    res = engine.process_input(env1)
    print(f"\nDECISION: {res.decision.decision_type.value} "
          f"key={res.decision.selected_key_id}")
    assert res.decision.decision_type == DecisionType.CLEAR
    assert res.decision.selected_key_id == "m1_bisnis"
    assert len(res.clarification_history) >= 1
    after = engine.storage.get_active_pending_state(session_id)
    assert after is None or after.status != PendingStatus.ACTIVE


def test_m1_a5_negative_evidence_eliminates_cpp(engine: MemoryEngine, tz: ZoneInfo):
    """
    [UNIT + INTEGRATION] Kondisi 5: "bukan cpp" menjadi negative evidence.
    Scope-aware: "bukan cpp" ada target; "cpp bukan?" tidak ada target.
    """
    extractor = SignalExtractor()
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)

    sig_pos = extractor.extract("yang penting bukan cpp", ref)
    print(f"\nEXCLUDED (bukan cpp): {sig_pos.excluded_tokens}")
    assert "cpp" in sig_pos.excluded_tokens
    assert len(sig_pos.negation_scopes) >= 1

    sig_neg = extractor.extract("cpp bukan?", ref)
    print(f"EXCLUDED (cpp bukan?): {sig_neg.excluded_tokens}")
    assert "cpp" not in sig_neg.excluded_tokens

    session_id = "m1_a5"
    _three_coffee_keys(engine, tz)
    engine.process_input(create_envelope(session_id, "Bahas kopi.", tz))
    assert engine.storage.get_active_pending_state(session_id) is not None

    env1 = create_envelope(session_id, "yang bisnis bukan cpp", tz, time_offset_hours=1)
    res = engine.process_input(env1)
    print(f"EXCLUDED KEYS: {res.excluded_candidate_keys}")
    assert "m1_cpp" in res.excluded_candidate_keys
    assert "m1_cpp" not in [c.key_id for c in res.candidates]


def test_m1_a6_explicit_cancellation_pauses_pending(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 6: cancellation menghentikan loop klarifikasi."""
    session_id = "m1_a6"
    _three_coffee_keys(engine, tz)
    engine.process_input(create_envelope(session_id, "Bahas kopi.", tz))
    assert engine.storage.get_active_pending_state(session_id) is not None

    env1 = create_envelope(session_id, "ganti topik, bahas C++ saja", tz,
                           time_offset_hours=1)
    engine.process_input(env1)
    assert engine.storage.get_active_pending_state(session_id) is None


# ==========================================
# B. CONTINUATION INTENT SAFETY (kondisi 7-10)
# ==========================================

def test_m2_b7_continuation_no_candidate_no_new_topic(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 7: continuation tanpa kandidat != topik baru."""
    env = create_envelope("m2_b7", "lanjutkan topik ini", tz)
    res = engine.process_input(env)
    print(f"\nDECISION: {res.decision.decision_type.value}")
    assert res.continuation_intent is True
    assert res.decision.decision_type == DecisionType.UNRESOLVED_CONTINUATION
    assert len(engine.storage.get_all_topic_keys()) == 0


def test_m2_b8_continuation_one_strong_reactivates(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 8: satu kandidat kuat -> reactivation."""
    s1, s2 = "m2_b8_s1", "m2_b8_s2"
    res1 = engine.process_input(create_envelope(s1, "Bisnis kopi modal usaha.", tz))
    coffee_key = res1.active_topic_key_id
    assert coffee_key is not None

    res2 = engine.process_input(
        create_envelope(s2, "lanjutkan bisnis kopi", tz, time_offset_hours=24)
    )
    assert res2.continuation_intent is True
    assert res2.decision.decision_type == DecisionType.CLEAR
    assert res2.active_topic_key_id == coffee_key


def test_m2_b9_continuation_two_strong_ambiguous(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 9: dua kandidat kuat -> ambiguity."""
    _mk_key(engine, tz, "m2_bisnis", "Bisnis Kopi", "Kopi bisnis",
            {"kopi": 10, "bisnis": 10})
    _mk_key(engine, tz, "m2_kedai", "Kedai Kopi", "Kopi kedai",
            {"kopi": 10, "kedai": 10})
    res = engine.process_input(create_envelope("m2_b9", "lanjutkan kopi", tz))
    assert res.continuation_intent is True
    assert res.decision.decision_type == DecisionType.AMBIGUOUS
    assert len(res.candidates) >= 2


def test_m2_b10_unresolved_response_not_persisted(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 10: respons unresolved TIDAK dipersist."""
    session_id = "m2_b10"
    res = engine.process_input(create_envelope(session_id, "lanjutkan topik ini", tz))
    assert res.persistence_status == "BLOCKED_UNRESOLVED_CONTINUATION"
    assert res.created_fragment_id is None
    assert len(engine.storage.get_all_topic_keys()) == 0
    assert engine.storage.get_recent_fragments(session_id, limit=10) == []


# ==========================================
# C. ACTIVE TOPIC & FRAGMENT (kondisi 11-12)
# ==========================================

def test_m3_c11_active_topic_local_continuation(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 11: input lanjutan lokal tetap di TopicKey aktif."""
    session_id = "m3_c11"
    res1 = engine.process_input(create_envelope(session_id, "Bisnis kopi modal usaha.", tz))
    coffee_key = res1.active_topic_key_id

    res2 = engine.process_input(
        create_envelope(session_id, "Berapa modal usaha bisnis yang logis?",
                        tz, time_offset_hours=1)
    )
    assert res2.active_topic_key_id == coffee_key
    assert len(engine.storage.get_all_topic_keys()) == 1
    state = engine.storage.get_session_state(session_id)
    assert state is not None
    assert state.active_topic_key_id == coffee_key
    assert state.active_fragment_id is not None


def test_m3_c12_active_state_does_not_lock(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 12: active state bukan probabilitas permanen."""
    session_id = "m3_c12"
    res1 = engine.process_input(create_envelope(session_id, "Bisnis kopi modal usaha.", tz))
    coffee_key = res1.active_topic_key_id

    res2 = engine.process_input(
        create_envelope(session_id, "Sekarang bahas paket wisata travel umroh.",
                        tz, time_offset_hours=1)
    )
    print(f"\nKEYS: {len(engine.storage.get_all_topic_keys())}")
    assert res2.active_topic_key_id != coffee_key
    assert len(engine.storage.get_all_topic_keys()) == 2


# ==========================================
# D. FRAGMENT KEY & HELPER (kondisi 13-15)
# ==========================================

def test_m3_d13_fragment_key_and_helper_before_compress(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 13: identity diekstrak sebelum kompresi; payload lossless."""
    session_id = "m3_d13"
    text = "Modal bisnis kopi sekitar lima juta."
    res = engine.process_input(create_envelope(session_id, text, tz))
    assert res.created_fragment_id is not None

    frag = engine.storage.get_fragment(res.created_fragment_id)
    print(f"\nKEY   : {frag.fragment_key}")
    print(f"HELPER: {frag.fragment_helper_description}")
    assert frag.fragment_key
    assert frag.fragment_helper_description
    assert "modal" in frag.fragment_key.lower()

    payload = engine.storage.get_exact_payload(frag.payload_reference)
    exact = engine.compressor.decompress(payload).decode("utf-8")
    assert text in exact


def test_m3_d14_child_helper_propagates_to_parent(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 14: helper child naik ke parent."""
    session_id = "m3_d14"
    res1 = engine.process_input(create_envelope(session_id, "Bisnis kopi modal usaha.", tz))
    root_id = res1.created_fragment_id
    assert root_id is not None

    res2 = engine.process_input(
        create_envelope(session_id, "Modal kopi murah dan mahal.", tz, time_offset_hours=1)
    )
    child_id = res2.created_fragment_id
    assert child_id is not None

    child = engine.storage.get_fragment(child_id)
    root = engine.storage.get_fragment(root_id)
    print(f"\nCHILD depth={child.depth} parent={child.parent_fragment_id}")
    print(f"ROOT evidence={root.descendant_helper_evidence}")
    assert child.parent_fragment_id == root_id
    assert len(root.descendant_helper_evidence) >= 1
    assert child_id in root.helper_contributor_ids
    assert root.aggregated_helper_description


def test_m3_d15_parent_helper_propagates_to_topic(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 15: helper parent naik ke TopicKey."""
    session_id = "m3_d15"
    res1 = engine.process_input(create_envelope(session_id, "Bisnis kopi modal usaha.", tz))
    key_id = res1.active_topic_key_id

    res2 = engine.process_input(
        create_envelope(session_id, "Modal kopi murah dan mahal.", tz, time_offset_hours=1)
    )
    key_after = engine.storage.get_topic_key(key_id)
    print(f"\nTOPIC AGG: {key_after.aggregated_helper_description}")
    assert key_after.aggregated_helper_description
    assert len(key_after.helper_source_fragment_ids) >= 1
    assert res2.helper_propagation is not None
    assert res2.helper_propagation.updated_topic_key_id == key_id


# ==========================================
# E. FRAGMENT HIERARCHY (kondisi 16-20)
# ==========================================

def test_m5_e16_first_fragment_is_root(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 16: fragment pertama = root depth 0."""
    res = engine.process_input(create_envelope("m5_e16", "Bisnis kopi modal usaha.", tz))
    frag = engine.storage.get_fragment(res.created_fragment_id)
    assert frag.depth == 0
    assert frag.parent_fragment_id is None
    assert res.fragment_placement.kind == FragmentPlacementKind.ROOT


def test_m5_e17_section_placement(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 17: fragment bagian masuk level pertama yang tepat."""
    session_id = "m5_e17"
    engine.process_input(create_envelope(session_id, "Bisnis kopi modal usaha.", tz))
    res2 = engine.process_input(
        create_envelope(session_id, "Modal kopi murah dan mahal.", tz, time_offset_hours=1)
    )
    frag = engine.storage.get_fragment(res2.created_fragment_id)
    print(f"\nSECTION depth={frag.depth} kind={res2.fragment_placement.kind.value}")
    assert frag.depth == 1
    assert res2.fragment_placement.kind in (
        FragmentPlacementKind.CHILD, FragmentPlacementKind.SIBLING
    )
    assert frag.depth <= CONFIG.hierarchy.MAX_FRAGMENT_DEPTH


def test_m5_e18_child_under_correct_parent(engine: MemoryEngine, tz: ZoneInfo):
    """[UNIT] Kondisi 18: 'Modal Murah vs Mahal' memilih parent 'Modal', bukan 'Lokasi'."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    signal = SignalExtractor().extract("Modal kopi murah dan mahal", ref)

    parent_modal = FragmentMetadata(
        fragment_id="p_modal", topic_key_id="k", session_id="s",
        created_at=ref, payload_reference="pay1",
        token_profile={"modal": 5, "kopi": 5, "bisnis": 5},
        fragment_key="Modal Bisnis Kopi",
        fragment_helper_description="modal bisnis kopi", depth=1,
    )
    parent_lokasi = FragmentMetadata(
        fragment_id="p_lokasi", topic_key_id="k", session_id="s",
        created_at=ref, payload_reference="pay2",
        token_profile={"lokasi": 5, "usaha": 5, "tempat": 5},
        fragment_key="Lokasi Usaha",
        fragment_helper_description="lokasi usaha tempat", depth=1,
    )
    scored = engine.retrieval.score_parent_candidates(signal, [parent_lokasi, parent_modal])
    for s in scored:
        print(f"   {s.fragment_key:<24} score={s.final_score:.3f}")
    assert scored[0].fragment_id == "p_modal"
    assert scored[0].final_score > scored[1].final_score


def test_m5_e19_unrelated_becomes_sibling(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 19: fragment lokasi tidak dipaksa jadi child Modal."""
    session_id = "m5_e19"
    key_id = "m5_e19_key"
    _mk_key(engine, tz, key_id, "Bisnis Kopi", "Bisnis kopi",
            {"bisnis": 10, "kopi": 10})
    _mk_frag(engine, tz, "root19", key_id, session_id, {"bisnis": 5, "kopi": 5},
             key_name="Bisnis Kopi", helper="bisnis kopi", depth=0)
    _mk_frag(engine, tz, "modal19", key_id, session_id,
             {"modal": 5, "kopi": 5, "biaya": 5}, key_name="Modal Bisnis Kopi",
             helper="modal bisnis kopi biaya", parent="root19", depth=1)

    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    signal = SignalExtractor().extract("Lokasi strategis tempat usaha di kota", ref)
    retrieval = RetrievalResult(
        decision_type=DecisionType.CLEAR, selected_key_id=key_id, signal_profile=signal
    )
    placement = engine._determine_fragment_placement(key_id, retrieval, is_new_topic=False)
    print(f"\nPLACEMENT {placement.kind.value} depth={placement.depth} "
          f"parent={placement.parent_fragment_id}")
    assert placement.parent_fragment_id != "modal19"
    assert placement.kind in (
        FragmentPlacementKind.SIBLING, FragmentPlacementKind.DEPTH_CAPPED_ATTACH
    )


def test_m5_e20_parent_by_evidence_not_recency(engine: MemoryEngine, tz: ZoneInfo):
    """[UNIT] Kondisi 20: parent dipilih by evidence; temporal & recency dinolkan."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    old_time = datetime(2023, 10, 1, 10, 0, 0, tzinfo=tz)
    new_time = datetime(2023, 10, 25, 13, 0, 0, tzinfo=tz)
    signal = SignalExtractor().extract("Modal kopi murah", ref)

    old_match = FragmentMetadata(
        fragment_id="old_match", topic_key_id="k", session_id="s",
        created_at=old_time, payload_reference="p1",
        token_profile={"modal": 5, "kopi": 5},
        fragment_key="Modal Kopi", fragment_helper_description="modal kopi", depth=1,
    )
    new_mismatch = FragmentMetadata(
        fragment_id="new_mismatch", topic_key_id="k", session_id="s",
        created_at=new_time, payload_reference="p2",
        token_profile={"lokasi": 5, "tempat": 5},
        fragment_key="Lokasi Tempat", fragment_helper_description="lokasi tempat", depth=1,
    )
    scored = engine.retrieval.score_parent_candidates(signal, [new_mismatch, old_match])
    for s in scored:
        print(f"   {s.fragment_id:<16} score={s.final_score:.3f} "
              f"temporal={s.temporal_score} recency={s.recency_score}")
    assert scored[0].fragment_id == "old_match"
    assert all(s.temporal_score == 0.0 for s in scored)
    assert all(s.recency_score == 0.0 for s in scored)


# ==========================================
# F. DEPTH BOUNDARY (kondisi 21-22)
# ==========================================

def test_m5_f21_max_depth_2_no_depth_3(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 21: MAX=2 -> tidak boleh ada depth 3."""
    assert CONFIG.hierarchy.MAX_FRAGMENT_DEPTH == 2
    session_id = "m5_f21"
    key_id = "m5_f21_key"
    _mk_key(engine, tz, key_id, "Bisnis Kopi", "Bisnis kopi",
            {"bisnis": 10, "kopi": 10})
    _mk_frag(engine, tz, "r21", key_id, session_id, {"bisnis": 5, "kopi": 5},
             key_name="Bisnis Kopi", helper="bisnis kopi", depth=0)
    _mk_frag(engine, tz, "m21", key_id, session_id, {"modal": 5, "kopi": 5},
             key_name="Modal Kopi", helper="modal kopi", parent="r21", depth=1)
    _mk_frag(engine, tz, "d21", key_id, session_id,
             {"modal": 5, "murah": 5, "mahal": 5},
             key_name="Modal Murah Mahal", helper="modal murah mahal",
             parent="m21", depth=2)

    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    signal = SignalExtractor().extract("Modal murah mahal perbandingan", ref)
    retrieval = RetrievalResult(
        decision_type=DecisionType.CLEAR, selected_key_id=key_id, signal_profile=signal
    )
    placement = engine._determine_fragment_placement(key_id, retrieval, is_new_topic=False)
    print(f"\nPLACEMENT depth={placement.depth} kind={placement.kind.value}")
    assert placement.depth <= CONFIG.hierarchy.MAX_FRAGMENT_DEPTH
    assert all(f.depth <= 2 for f in engine.storage.get_fragment_metadata_by_key(key_id))


def test_m5_f22_max_depth_3_allows_depth_3(engine: MemoryEngine, tz: ZoneInfo, monkeypatch):
    """
    [INTEGRATION + CONFIG OVERRIDE] Kondisi 22: MAX=3 -> depth 3 diizinkan.
    CONFIG frozen, jadi dibangkitkan ulang via dataclasses.replace lalu
    di-bind ke module global. Monkeypatch memulihkan otomatis.
    """
    new_hierarchy = dataclasses.replace(CONFIG.hierarchy, MAX_FRAGMENT_DEPTH=3)
    new_cfg = dataclasses.replace(CONFIG, hierarchy=new_hierarchy)
    monkeypatch.setattr(memory_engine_module, "CONFIG", new_cfg)
    monkeypatch.setattr(retrieval_module, "CONFIG", new_cfg)

    session_id = "m5_f22"
    key_id = "m5_f22_key"
    _mk_key(engine, tz, key_id, "Bisnis Kopi", "Bisnis kopi",
            {"bisnis": 10, "kopi": 10})
    _mk_frag(engine, tz, "r22", key_id, session_id, {"bisnis": 5, "kopi": 5},
             key_name="Bisnis Kopi", helper="bisnis kopi", depth=0)
    _mk_frag(engine, tz, "m22", key_id, session_id, {"modal": 5, "kopi": 5},
             key_name="Modal Kopi", helper="modal kopi", parent="r22", depth=1)
    _mk_frag(engine, tz, "d22", key_id, session_id,
             {"modal": 5, "murah": 5, "mahal": 5},
             key_name="Modal Murah Mahal", helper="modal murah mahal",
             parent="m22", depth=2)

    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    signal = SignalExtractor().extract("Modal murah mahal perbandingan", ref)
    retrieval = RetrievalResult(
        decision_type=DecisionType.CLEAR, selected_key_id=key_id, signal_profile=signal
    )
    placement = engine._determine_fragment_placement(key_id, retrieval, is_new_topic=False)
    print(f"\nMAX=3 PLACEMENT depth={placement.depth} kind={placement.kind.value} "
          f"score={placement.match_score:.3f}")
    assert placement.depth == 3
    assert placement.kind == FragmentPlacementKind.CHILD
    assert placement.parent_fragment_id == "d22"


# ==========================================
# G. CROSS-KEY COMPOSITE (kondisi 23-26)
# ==========================================

def test_m4_g23_composite_detected_two_topics(engine: MemoryEngine, tz: ZoneInfo):
    """[UNIT] Kondisi 23: respons memakai evidence 2 TopicKey -> composite."""
    _composite_setup(engine, tz)
    mems = [
        RetrievedMemory(fragment_id="src_a", text="resep kopi", timestamp="t",
                        topic_key_id="cmp_a"),
        RetrievedMemory(fragment_id="src_b", text="resep murah bahan", timestamp="t",
                        topic_key_id="cmp_b"),
    ]
    response = "Gabungkan resep kopi dengan resep murah bahan yang lebih hemat"
    analysis = engine._detect_composite_sources(response, mems, "cmp_a")
    print(f"\nCOMPOSITE: {analysis.is_composite} | {analysis.reason}")
    for s in analysis.sources:
        print(f"   {s.fragment_id} topic={s.topic_key_id} "
              f"score={s.contribution_score:.3f} ev={s.supporting_evidence_names}")
    assert analysis.is_composite is True
    assert analysis.distinct_topic_count == 2
    assert len(analysis.sources) >= CONFIG.composite.MIN_DISTINCT_SOURCES


def test_m4_g24_source_ids_recorded(engine: MemoryEngine, tz: ZoneInfo):
    """[UNIT] Kondisi 24: kedua source fragment tercatat."""
    _composite_setup(engine, tz)
    mems = [
        RetrievedMemory(fragment_id="src_a", text="resep kopi", timestamp="t",
                        topic_key_id="cmp_a"),
        RetrievedMemory(fragment_id="src_b", text="resep murah bahan", timestamp="t",
                        topic_key_id="cmp_b"),
    ]
    response = "Gabungkan resep kopi dengan resep murah bahan yang lebih hemat"
    analysis = engine._detect_composite_sources(response, mems, "cmp_a")
    print(f"\nSOURCE IDS: {analysis.source_fragment_ids}")
    assert "src_a" in analysis.source_fragment_ids
    assert "src_b" in analysis.source_fragment_ids
    assert len(analysis.source_fragment_ids) == 2


def test_m4_g25_graph_edges_to_both_sources(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 25: graph edge terbentuk ke KEDUA sumber."""
    _composite_setup(engine, tz)
    session_id = "s_cmp_new"
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    response = "Gabungkan resep kopi dengan resep murah bahan yang lebih hemat"

    mems = [
        RetrievedMemory(fragment_id="src_a", text="resep kopi", timestamp="t",
                        topic_key_id="cmp_a"),
        RetrievedMemory(fragment_id="src_b", text="resep murah bahan", timestamp="t",
                        topic_key_id="cmp_b"),
    ]
    analysis = engine._detect_composite_sources(response, mems, "cmp_a")
    assert analysis.is_composite is True

    signal = SignalExtractor().extract(response, ref)
    retrieval = RetrievalResult(
        decision_type=DecisionType.CLEAR, selected_key_id="cmp_a", signal_profile=signal
    )
    turn = TurnAnalysisResult(
        classification=MemoryClassification.CONTINUE_TOPIC,
        active_topic_key_id="cmp_a", turn_continuity_score=0.5,
        mutation_allowed=True, description_update_allowed=True,
    )
    env = create_envelope(session_id, "Gabungkan resep kopi murah bahan", tz)
    created_id, placement, propagation = engine._persist_memory(
        env, response, turn, retrieval, "cmp_a", composite_analysis=analysis
    )
    assert created_id is not None

    created = engine.storage.get_fragment(created_id)
    print(f"\nCREATED composite={created.is_composite} "
          f"sources={created.source_fragment_ids}")
    assert created.is_composite is True
    assert set(created.source_fragment_ids) == {"src_a", "src_b"}

    related_ids = {f.fragment_id for f in engine.storage.get_related_fragments(created_id)}
    print(f"GRAPH NEIGHBORS: {related_ids}")
    assert "src_a" in related_ids
    assert "src_b" in related_ids


def test_m4_g26_weak_overlap_not_composite(engine: MemoryEngine, tz: ZoneInfo):
    """[UNIT] Kondisi 26: overlap lemah / satu family bukan composite."""
    _composite_setup(engine, tz)
    _mk_frag(engine, tz, "src_weak", "cmp_b", "s_cmp",
             {"resep": 5, "alpha": 5, "beta": 5, "gamma": 5, "delta": 5,
              "epsilon": 5, "zeta": 5, "eta": 5, "theta": 5, "iota": 5},
             key_name="", helper="", depth=0)

    mems = [
        RetrievedMemory(fragment_id="src_a", text="resep kopi", timestamp="t",
                        topic_key_id="cmp_a"),
        RetrievedMemory(fragment_id="src_weak", text="weak", timestamp="t",
                        topic_key_id="cmp_b", fragment_graph_score=0.0),
    ]
    analysis = engine._detect_composite_sources("resep kopi saja", mems, "cmp_a")
    print(f"\nWEAK COMPOSITE: {analysis.is_composite} | {analysis.reason}")
    assert analysis.is_composite is False
    assert analysis.source_fragment_ids == []


# ==========================================
# H. TREND TOPIC (kondisi 27-29)
# ==========================================

def test_m5_h27_trend_topic_limit_5(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 27: maksimal TOP 5 TopicKey."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    for i in range(7):
        kid = f"trend_key_{i}"
        _mk_key(engine, tz, kid, f"Topik {i}", f"topik nomor {i}",
                {"topik": 5, f"nomor{i}": 5})
        for j in range(i + 1):
            _mk_frag(engine, tz, f"f_{i}_{j}", kid, "s_trend",
                     {"topik": 5, f"nomor{i}": 5}, depth=0, when=ref)

    trends = engine.storage.get_trending_topic_keys(reference_time=ref)
    print(f"\nTREND TOPICS: {len(trends)}")
    for t in trends:
        print(f"   {t.label:<12} score={t.trend_score:.3f} frag={t.fragment_count}")
    assert len(trends) <= CONFIG.trend.TREND_TOPIC_LIMIT
    assert len(trends) == 5
    assert all(t.node_type == TrendNodeType.TOPIC_KEY for t in trends)


def test_m5_h28_richer_branch_ranks_higher(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 28: branch richness menaikkan prioritas."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    _mk_key(engine, tz, "rich", "Topik Kaya", "topik kaya", {"topik": 5, "kaya": 5})
    _mk_key(engine, tz, "poor", "Topik Miskin", "topik miskin", {"topik": 5, "miskin": 5})
    _mk_frag(engine, tz, "rich_root", "rich", "s_rich", {"topik": 5, "kaya": 5},
             depth=0, when=ref)
    for j in range(6):
        _mk_frag(engine, tz, f"rich_child_{j}", "rich", "s_rich",
                 {"topik": 5, "kaya": 5}, parent="rich_root", depth=1, when=ref)
    _mk_frag(engine, tz, "poor_root", "poor", "s_poor", {"topik": 5, "miskin": 5},
             depth=0, when=ref)

    trends = engine.storage.get_trending_topic_keys(reference_time=ref)
    for t in trends:
        print(f"   {t.node_id:<6} score={t.trend_score:.3f} "
              f"frag={t.fragment_count} branch={t.child_count}")
    assert trends[0].node_id == "rich"
    rich = next(t for t in trends if t.node_id == "rich")
    poor = next(t for t in trends if t.node_id == "poor")
    assert rich.trend_score > poor.trend_score
    assert rich.child_count > poor.child_count


def test_m5_h29_temporal_affects_trend(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 29: referensi temporal mempengaruhi ranking trend."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    yesterday = datetime(2023, 10, 24, 15, 0, 0, tzinfo=tz)
    long_ago = datetime(2023, 9, 1, 10, 0, 0, tzinfo=tz)

    _mk_key(engine, tz, "t_recent", "Topik Kemarin", "topik kemarin",
            {"topik": 5, "kemarin": 5})
    _mk_key(engine, tz, "t_old", "Topik Lama", "topik lama",
            {"topik": 5, "lama": 5})
    _mk_frag(engine, tz, "f_recent", "t_recent", "s_t", {"topik": 5},
             depth=0, when=yesterday)
    _mk_frag(engine, tz, "f_old", "t_old", "s_t", {"topik": 5},
             depth=0, when=long_ago)

    start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
    with_range = engine.storage.get_trending_topic_keys(
        reference_time=ref, temporal_range=(start, end)
    )
    for t in with_range:
        print(f"   {t.node_id:<9} score={t.trend_score:.3f} temporal={t.temporal_score}")
    recent = next(t for t in with_range if t.node_id == "t_recent")
    old = next(t for t in with_range if t.node_id == "t_old")
    assert recent.temporal_score == 1.0
    assert old.temporal_score == 0.0
    assert recent.trend_score > old.trend_score
    assert with_range[0].node_id == "t_recent"


# ==========================================
# I. TREND FRAGMENT (kondisi 30-31)
# ==========================================

def test_m5_i30_trend_fragments_depth1_limit5(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 30: maksimal 5 fragment depth 1, child terbanyak di atas."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    key_id = "trend_frag_key"
    _mk_key(engine, tz, key_id, "Bisnis Kopi", "bisnis kopi",
            {"bisnis": 5, "kopi": 5})
    _mk_frag(engine, tz, "tf_root", key_id, "s_tf", {"bisnis": 5, "kopi": 5},
             depth=0, when=ref)
    for i in range(7):
        sid = f"tf_sec_{i}"
        _mk_frag(engine, tz, sid, key_id, "s_tf", {"bisnis": 5, "kopi": 5},
                 key_name=f"Section {i}", helper=f"section {i}",
                 parent="tf_root", depth=1, when=ref)
        for j in range(i):
            _mk_frag(engine, tz, f"{sid}_c{j}", key_id, "s_tf",
                     {"bisnis": 5, "kopi": 5}, parent=sid, depth=2, when=ref)

    trends = engine.storage.get_trending_fragments(
        topic_key_id=key_id, depth=1, reference_time=ref
    )
    print(f"\nTREND FRAGMENTS: {len(trends)}")
    for t in trends:
        print(f"   {t.label:<14} depth={t.depth} children={t.child_count} "
              f"score={t.trend_score:.3f}")
    assert len(trends) <= CONFIG.trend.TREND_FRAGMENT_LIMIT
    assert len(trends) == 5
    assert all(t.depth == 1 for t in trends)
    assert all(t.node_type == TrendNodeType.FRAGMENT for t in trends)
    assert trends[0].child_count >= trends[-1].child_count
    assert trends[0].node_id == "tf_sec_6"


def test_m5_i31_no_expansion_at_max_depth(engine: MemoryEngine, tz: ZoneInfo):
    """[INTEGRATION] Kondisi 31: di depth maksimum tidak ada trend expansion."""
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    key_id = "max_depth_key"
    _mk_key(engine, tz, key_id, "Bisnis Kopi", "bisnis kopi",
            {"bisnis": 5, "kopi": 5})
    _mk_frag(engine, tz, "md_root", key_id, "s_md", {"bisnis": 5}, depth=0, when=ref)
    _mk_frag(engine, tz, "md_sec", key_id, "s_md", {"bisnis": 5},
             key_name="Modal", helper="modal", parent="md_root", depth=1, when=ref)
    _mk_frag(engine, tz, "md_leaf", key_id, "s_md", {"bisnis": 5},
             key_name="Modal Murah", helper="modal murah", parent="md_sec",
             depth=CONFIG.hierarchy.MAX_FRAGMENT_DEPTH, when=ref)
    _mk_frag(engine, tz, "md_leaf_child", key_id, "s_md", {"bisnis": 5},
             parent="md_leaf", depth=CONFIG.hierarchy.MAX_FRAGMENT_DEPTH, when=ref)

    expanded = engine.storage.get_trending_children("md_leaf", reference_time=ref)
    print(f"\nEXPANSION AT MAX: {len(expanded)} (harus 0)")
    assert expanded == []

    ok = engine.storage.get_trending_children("md_sec", reference_time=ref)
    print(f"EXPANSION BELOW MAX: {len(ok)}")
    assert len(ok) >= 1
    assert all(t.depth <= CONFIG.hierarchy.MAX_FRAGMENT_DEPTH for t in ok)


# ==========================================
# [END OF HIERARCHICAL FRAGMENT EXPANSION SUITE]
# ==========================================
