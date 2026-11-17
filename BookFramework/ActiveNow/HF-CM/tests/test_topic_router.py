"""
tests/test_topic_router.py

[ASSOCIATION EXPANSION - Batch I.2] VALIDATION SUITE — 9 KONDISI

Menguji TopicRouter (Batch D) yang menentukan apakah input baru
melanjutkan TopicKey aktif (LOCAL MATCH) atau mencari topik baru (GLOBAL).

Prinsip kunci:
    LOCAL-FIRST ROUTING:
        session active state exists?
            ├── tidak → GLOBAL ROUTING
            └── ya → LOCAL TOPIC EVALUATION
                        score >= ACTIVE_TOPIC_MATCH_THRESHOLD (0.45)?
                            ├── ya → ACTIVE_LOCAL_MATCH
                            └── tidak → margin >= LOCAL_VS_GLOBAL_MARGIN (0.03)?
                                            ├── ya → ACTIVE_LOCAL_MATCH
                                            └── tidak → GLOBAL SEARCH

Semua test adalah UNIT TEST (tidak menyentuh storage / engine).
Membangun TopicRouter + mock objects (RetrievalResult, TopicKey, dll)
langsung untuk mengisolasi routing logic.

Konfigurasi yang dipakai (dari config.py Batch A2):
    ACTIVE_TOPIC_MATCH_THRESHOLD = 0.45
    LOCAL_VS_GLOBAL_MARGIN = 0.03
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import CONFIG
from models import (
    InputEnvelope,
    SessionActiveState,
    TopicKey,
    TopicRouteResult,
    TopicRouteType,
    CandidateScore,
)
from retrieval import RetrievalResult, DecisionType, SignalProfile
from topic_router import TopicRouter


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def tz():
    return ZoneInfo(CONFIG.input.DEFAULT_TIMEZONE)


@pytest.fixture
def router():
    return TopicRouter()


# ==========================================
# HELPERS
# ==========================================

def _make_envelope(text: str, tz, session_id: str = "test_session") -> InputEnvelope:
    """Buat InputEnvelope minimal untuk testing."""
    base_time = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    return InputEnvelope(
        input_id="inp-1",
        session_id=session_id,
        raw_text=text,
        normalized_text=text.lower(),
        timestamp=base_time,
        timezone=str(tz),
    )


def _make_topic_key(
    key_id: str,
    title: str,
    tokens: dict,
    helper: str = "",
    aggregated: str = "",
    cooc: dict = None,
) -> TopicKey:
    """Buat TopicKey dengan token_profile untuk scoring."""
    return TopicKey(
        key_id=key_id,
        title=title,
        helper_description=helper,
        token_profile=tokens,
        co_occurrence_profile=cooc or {},
        created_at=datetime(2023, 10, 25, 14, 0, 0),
        aggregated_helper_description=aggregated,
    )


def _make_session_state(
    session_id: str,
    active_topic_key_id: str,
    active_depth: int = 1,
) -> SessionActiveState:
    """Buat SessionActiveState dengan active topic."""
    return SessionActiveState(
        session_id=session_id,
        active_topic_key_id=active_topic_key_id,
        active_fragment_id="frag-1",
        active_depth=active_depth,
        last_updated=datetime(2023, 10, 25, 14, 0, 0),
    )


def _make_retrieval(
    decision_type: DecisionType,
    selected_key_id=None,
    candidates=None,
    signal=None,
) -> RetrievalResult:
    """Buat RetrievalResult dengan decision type tertentu."""
    return RetrievalResult(
        decision_type=decision_type,
        candidates=candidates or [],
        selected_key_id=selected_key_id,
        selected_fragment_ids=[],
        signal_profile=signal,
    )


def _make_candidate(key_id: str, title: str, score: float) -> CandidateScore:
    """Buat CandidateScore untuk global retrieval."""
    return CandidateScore(
        key_id=key_id,
        title=title,
        final_score=score,
    )


# ==========================================
# TESTS: NO ACTIVE STATE
# ==========================================

def test_route_no_active_state_global_match(router: TopicRouter, tz):
    """
    [UNIT] Tidak ada active state + retrieval CLEAR → GLOBAL_MATCH.
    
    Ini adalah path paling sederhana: user bicara topik baru, retrieval
    menemukan kandidat CLEAR.
    """
    envelope = _make_envelope("Bahas bisnis kopi modal usaha", tz)
    retrieval = _make_retrieval(
        decision_type=DecisionType.CLEAR,
        selected_key_id="key_bisnis",
        candidates=[_make_candidate("key_bisnis", "Bisnis Kopi", 0.75)],
    )
    
    result = router.route(
        envelope=envelope,
        session_state=None,
        active_topic_key=None,
        retrieval_result=retrieval,
    )
    
    assert isinstance(result, TopicRouteResult)
    assert result.route_type == TopicRouteType.GLOBAL_MATCH
    assert result.selected_topic_key_id == "key_bisnis"
    assert result.local_attempted is False
    assert result.local_matched is False
    assert result.global_search_performed is True


# ==========================================
# TESTS: LOCAL MATCH
# ==========================================

def test_route_active_local_match_via_threshold(router: TopicRouter, tz):
    """
    [UNIT] Active topic dengan overlap token yang cukup
    (score >= ACTIVE_TOPIC_MATCH_THRESHOLD) → ACTIVE_LOCAL_MATCH.
    
    Input: "modal bisnis kopi usaha" (4 token bermakna)
    Active Topic: token_profile = {modal, bisnis, kopi, usaha, rencana} (5 token)
    Overlap: 4/5 = 0.80 ≥ 0.45 → LOCAL MATCH
    """
    session_id = "test_session"
    envelope = _make_envelope("modal bisnis kopi usaha", tz, session_id)
    
    active_key = _make_topic_key(
        key_id="key_bisnis",
        title="Bisnis Kopi",
        tokens={"modal": 5, "bisnis": 5, "kopi": 5, "usaha": 5, "rencana": 5},
    )
    session_state = _make_session_state(session_id, "key_bisnis")
    
    # Retrieval tidak penting karena local match menang sebelum global
    retrieval = _make_retrieval(decision_type=DecisionType.CLEAR)
    
    result = router.route(
        envelope=envelope,
        session_state=session_state,
        active_topic_key=active_key,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.ACTIVE_LOCAL_MATCH
    assert result.selected_topic_key_id == "key_bisnis"
    assert result.local_attempted is True
    assert result.local_matched is True
    assert result.active_topic_score >= CONFIG.association.ACTIVE_TOPIC_MATCH_THRESHOLD


def test_route_active_local_match_via_margin(router: TopicRouter, tz):
    """
    [UNIT] Active topic score < threshold TAPI margin vs global best OK.
    
    Input: "kopi" (1 token)
    Active Topic: {kopi, bisnis, modal} → overlap 1/3 = 0.33 < 0.45
    Global best: 0.25
    Margin: 0.33 - 0.25 = 0.08 >= 0.03 → LOCAL MATCH via margin
    
    Ini path sekunder: local score tidak absolut cukup tinggi,
    tapi lebih baik dari kandidat global mana pun.
    """
    session_id = "test_session"
    envelope = _make_envelope("kopi", tz, session_id)
    
    active_key = _make_topic_key(
        key_id="key_bisnis",
        title="Bisnis Kopi",
        tokens={"kopi": 5, "bisnis": 5, "modal": 5},
    )
    session_state = _make_session_state(session_id, "key_bisnis")
    
    # Global best score = 0.25 (lebih rendah dari local 0.33)
    retrieval = _make_retrieval(
        decision_type=DecisionType.CLEAR,
        selected_key_id="key_kedai",
        candidates=[_make_candidate("key_kedai", "Kedai Kopi", 0.25)],
    )
    
    result = router.route(
        envelope=envelope,
        session_state=session_state,
        active_topic_key=active_key,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.ACTIVE_LOCAL_MATCH
    assert result.selected_topic_key_id == "key_bisnis"
    assert result.local_attempted is True
    assert result.local_matched is True
    # Margin check: active score harus > global best
    assert result.active_topic_score >= 0.25


# ==========================================
# TESTS: GLOBAL BEATS LOCAL
# ==========================================

def test_route_global_beats_local(router: TopicRouter, tz):
    """
    [UNIT] Global score jauh lebih tinggi → GLOBAL_MATCH (bukan local).
    
    Input: "bisnis travel wisata umroh"
    Active Topic (kopi): {kopi, bisnis} → overlap 1/2 = 0.50
    Global best (travel): score 0.95
    
    Wait - active score 0.50 >= 0.45, jadi threshold terpenuhi dulu.
    
    Buat skenario lebih jelas:
    Input: "travel wisata umroh paket"
    Active Topic (kopi): {kopi, modal, bisnis} → overlap 0/3 = 0.00 < 0.45
    Global best (travel): score 0.95
    Margin: 0.00 - 0.95 = -0.95 < 0.03 → GLOBAL wins
    """
    session_id = "test_session"
    envelope = _make_envelope("travel wisata umroh paket", tz, session_id)
    
    active_key = _make_topic_key(
        key_id="key_kopi",
        title="Bisnis Kopi",
        tokens={"kopi": 5, "modal": 5, "bisnis": 5},
    )
    session_state = _make_session_state(session_id, "key_kopi")
    
    # Global kandidat dengan score tinggi (travel topic)
    retrieval = _make_retrieval(
        decision_type=DecisionType.CLEAR,
        selected_key_id="key_travel",
        candidates=[_make_candidate("key_travel", "Paket Wisata", 0.95)],
    )
    
    result = router.route(
        envelope=envelope,
        session_state=session_state,
        active_topic_key=active_key,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.GLOBAL_MATCH
    assert result.selected_topic_key_id == "key_travel"
    assert result.local_attempted is True  # local sudah dicoba
    assert result.local_matched is False   # tapi gagal
    assert result.global_search_performed is True


# ==========================================
# TESTS: DECISION PASSTHROUGH
# ==========================================

def test_route_new_topic_from_no_match(router: TopicRouter, tz):
    """
    [UNIT] Retrieval NO_MATCH → NEW_TOPIC.
    
    Tidak ada active state, tidak ada kandidat global yang cocok.
    """
    envelope = _make_envelope("xyz abc random topic", tz)
    retrieval = _make_retrieval(decision_type=DecisionType.NO_MATCH)
    
    result = router.route(
        envelope=envelope,
        session_state=None,
        active_topic_key=None,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.NEW_TOPIC
    assert result.selected_topic_key_id is None
    assert result.global_search_performed is True


def test_route_unresolved_from_ambiguous(router: TopicRouter, tz):
    """
    [UNIT] Retrieval AMBIGUOUS → UNRESOLVED.
    
    Router tidak memutuskan mana yang menang; itu tugas memory_engine
    via clarification loop.
    """
    envelope = _make_envelope("kopi", tz)
    retrieval = _make_retrieval(
        decision_type=DecisionType.AMBIGUOUS,
        candidates=[
            _make_candidate("key_bisnis", "Bisnis Kopi", 0.81),
            _make_candidate("key_kedai", "Kedai Kopi", 0.79),
        ],
    )
    
    result = router.route(
        envelope=envelope,
        session_state=None,
        active_topic_key=None,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.UNRESOLVED
    assert result.selected_topic_key_id is None
    # Global candidates harus ter-record untuk observability
    assert len(result.global_candidates) >= 1


def test_route_unresolved_from_insufficient(router: TopicRouter, tz):
    """
    [UNIT] Retrieval INSUFFICIENT_EVIDENCE → UNRESOLVED.
    
    Kandidat ada tapi evidence tidak cukup untuk CLEAR decision.
    """
    envelope = _make_envelope("kopi", tz)
    retrieval = _make_retrieval(
        decision_type=DecisionType.INSUFFICIENT_EVIDENCE,
        candidates=[_make_candidate("key_bisnis", "Bisnis Kopi", 0.30)],
    )
    
    result = router.route(
        envelope=envelope,
        session_state=None,
        active_topic_key=None,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.UNRESOLVED
    assert result.selected_topic_key_id is None


# ==========================================
# TESTS: CANCELLATION SHORT-CIRCUIT
# ==========================================

def test_route_cancellation_shortcircuit(router: TopicRouter, tz):
    """
    [UNIT] Cancellation intent → NEW_TOPIC tanpa local evaluation.
    
    User bilang "ganti topik, bahas C++ saja" → langsung NEW_TOPIC
    tanpa cek active state. Ini short-circuit penting agar cancellation
    tidak dipaksa menjadi continuation.
    """
    session_id = "test_session"
    envelope = _make_envelope("ganti topik, bahas C++ saja", tz, session_id)
    
    # Active state ADA, tapi harus diabaikan karena cancellation
    active_key = _make_topic_key(
        key_id="key_kopi",
        title="Bisnis Kopi",
        tokens={"kopi": 5, "modal": 5},
    )
    session_state = _make_session_state(session_id, "key_kopi")
    
    # Signal profile dengan cancellation_intent=True
    signal = SignalProfile(cancellation_intent=True)
    retrieval = _make_retrieval(
        decision_type=DecisionType.CLEAR,
        selected_key_id="key_cpp",
        signal=signal,
    )
    
    result = router.route(
        envelope=envelope,
        session_state=session_state,
        active_topic_key=active_key,
        retrieval_result=retrieval,
    )
    
    assert result.route_type == TopicRouteType.NEW_TOPIC
    assert result.local_attempted is False  # local tidak dijalankan
    assert result.local_matched is False
    # Reason harus mention cancellation
    assert "cancellation" in result.reason.lower()


# ==========================================
# TESTS: OBSERVABILITY
# ==========================================

def test_route_fields_populated(router: TopicRouter, tz):
    """
    [UNIT] Observability fields terisi dengan benar.
    
    Setiap TopicRouteResult harus punya:
        - route_type (enum)
        - selected_topic_key_id (bisa None)
        - active_topic_score (float)
        - local_attempted (bool)
        - local_matched (bool)
        - global_search_performed (bool)
        - reason (string, tidak kosong)
    """
    session_id = "test_session"
    envelope = _make_envelope("modal bisnis kopi", tz, session_id)
    
    active_key = _make_topic_key(
        key_id="key_bisnis",
        title="Bisnis Kopi",
        tokens={"modal": 5, "bisnis": 5, "kopi": 5},
    )
    session_state = _make_session_state(session_id, "key_bisnis")
    retrieval = _make_retrieval(decision_type=DecisionType.CLEAR)
    
    result = router.route(
        envelope=envelope,
        session_state=session_state,
        active_topic_key=active_key,
        retrieval_result=retrieval,
    )
    
    # Check semua field observability
    assert isinstance(result.route_type, TopicRouteType)
    assert isinstance(result.local_attempted, bool)
    assert isinstance(result.local_matched, bool)
    assert isinstance(result.global_search_performed, bool)
    assert isinstance(result.active_topic_score, float)
    assert result.reason  # tidak kosong
    
    # Print untuk observability debugging
    print(f"\nRoute: {result.route_type.value}")
    print(f"  selected: {result.selected_topic_key_id}")
    print(f"  active_score: {result.active_topic_score:.3f}")
    print(f"  local_attempted: {result.local_attempted}")
    print(f"  local_matched: {result.local_matched}")
    print(f"  global_search_performed: {result.global_search_performed}")
    print(f"  reason: {result.reason}")
    
    # Local harus dicoba karena ada active state
    assert result.local_attempted is True


# ==========================================
# [END OF TOPIC ROUTER TEST SUITE]
# ==========================================
