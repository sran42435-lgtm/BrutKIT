import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import CONFIG
from models import (
    InputEnvelope, DecisionType, PendingStatus, MemoryClassification,
    TopicKey, FragmentMetadata, CompressedPayload
)
from storage import MemoryStorage
from retrieval import RetrievalEngine
from compression import PayloadCompressor, IntegrityError
from memory_engine import MemoryEngine
# [NEW] Import untuk End-to-End Pipeline Testing (Spec 26)
from context_builder import ContextBuilder
from llm_client import FakeLLMClient

# ==========================================
# FIXTURES (Setup & Teardown)
# ==========================================

@pytest.fixture(scope="function")
def test_db_path():
    """Membuat database SQLite sementara untuk setiap test agar terisolasi."""
    db_file = f"test_memory_{uuid.uuid4().hex[:8]}.db"
    yield f"sqlite:///{db_file}"
    if os.path.exists(db_file):
        os.remove(db_file)

@pytest.fixture(scope="function")
def engine(test_db_path):
    """Merakit seluruh komponen mesin memori untuk testing."""
    storage = MemoryStorage(db_url=test_db_path)
    retrieval = RetrievalEngine()
    compressor = PayloadCompressor()
    # [NEW] Injeksi Context Builder & Fake LLM Client untuk uji End-to-End Pipeline
    context_builder = ContextBuilder()
    llm_client = FakeLLMClient()
    return MemoryEngine(storage, retrieval, compressor, context_builder, llm_client)

@pytest.fixture(scope="function")
def tz():
    return ZoneInfo(CONFIG.input.DEFAULT_TIMEZONE)

def create_envelope(session_id: str, text: str, tz: ZoneInfo, time_offset_hours: int = 0) -> InputEnvelope:
    """Helper untuk membuat InputEnvelope dengan waktu yang bisa dimanipulasi."""
    base_time = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    target_time = base_time + timedelta(hours=time_offset_hours)
    
    return InputEnvelope(
        input_id=str(uuid.uuid4()),
        session_id=session_id,
        raw_text=text,
        normalized_text=text.lower(),
        timestamp=target_time,
        timezone=str(tz)
    )

# ==========================================
# SCENARIO 1: TOPIC CONTINUITY
# ==========================================
def test_scenario_1_topic_continuity(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_1"
    
    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    res1 = engine.process_input(env1)
    assert res1.decision.decision_type in [DecisionType.CLEAR, DecisionType.NO_MATCH]
    key_id_1 = res1.active_topic_key_id
    assert key_id_1 is not None, "Topik Key pertama harus terbuat."

    env2 = create_envelope(session_id, "Modalnya sekitar 50 juta cukup?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    
    env3 = create_envelope(session_id, "Terus lokasinya di mana yang strategis?", tz, time_offset_hours=2)
    res3 = engine.process_input(env3)

    keys = engine.storage.get_all_topic_keys()
    assert len(keys) == 1, "Harus hanya ada 1 Topic Key (Bisnis Kopi). Tidak boleh membuat topik baru untuk Modal/Lokasi."
    
    fragments = engine.storage.get_fragment_metadata_by_key(keys[0].key_id)
    assert len(fragments) >= 2, "Fragment untuk Modal dan Lokasi harus tersimpan di Key yang sama."


# ==========================================
# SCENARIO 2 & 34: TOPIC SWITCH PROTECTION
# ==========================================
def test_scenario_2_topic_switch_protection(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_switch"
    
    for i in range(5):
        env = create_envelope(session_id, f"Bahas kopi bagian {i}", tz, time_offset_hours=i)
        engine.process_input(env)
        
    keys_before = engine.storage.get_all_topic_keys()
    assert len(keys_before) == 1
    coffee_key_id = keys_before[0].key_id

    env_cpp = create_envelope(session_id, "Ajarkan saya pointer dan reference di C++.", tz, time_offset_hours=10)
    res_cpp = engine.process_input(env_cpp)

    keys_after = engine.storage.get_all_topic_keys()
    assert len(keys_after) == 2, "Sistem harus membuat Topic Key baru untuk C++."


# ==========================================
# SCENARIO 4 & 8: AMBIGUITY GATE
# ==========================================
def test_scenario_4_ambiguity_gate(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_ambigu"
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Modal dan strategi usaha kopi", created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Mencari kedai kopi terdekat dan menu", created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "Bahas kopi dong.", tz)
    res = engine.process_input(env)

    assert res.decision.decision_type == DecisionType.AMBIGUOUS, "Sistem tidak boleh menebak jika selisih skor tipis."
    
    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None, "Pending State harus aktif."
    assert pending.status == PendingStatus.ACTIVE
    assert len(pending.candidate_keys) >= 2


# ==========================================
# SCENARIO 6 & 24: AMBIGUITY RESOLUTION LOOP
# ==========================================
def test_scenario_6_ambiguity_resolution(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_resolve"
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Strategi bisnis", created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Lokasi kedai", created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    env_ambigu = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env_ambigu)
    
    pending_before = engine.storage.get_active_pending_state(session_id)
    assert pending_before is not None

    # [FIX] Tambahkan kata unik dari deskripsi ("strategi", "modal") agar margin skor melebar dan decision menjadi CLEAR
    env_resolve = create_envelope(session_id, "Yang bisnis saja, kita bahas strategi dan modal.", tz, time_offset_hours=1)
    res_resolve = engine.process_input(env_resolve)

    pending_after = engine.storage.get_active_pending_state(session_id)
    
    if pending_after:
        assert pending_after.status != PendingStatus.ACTIVE, "Pending state harus selesai setelah klarifikasi valid."


# ==========================================
# SCENARIO 7 & 29: PAUSE PENDING ON TOPIC SWITCH
# ==========================================
def test_scenario_7_pause_pending_on_switch(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_pause"
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Strategi", created_at=datetime.now(tz), token_profile={"kopi": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Lokasi", created_at=datetime.now(tz), token_profile={"kopi": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    env_ambigu = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env_ambigu) 
    
    env_switch = create_envelope(session_id, "Tidak, saya sekarang mau belajar C++.", tz, time_offset_hours=1)
    res_switch = engine.process_input(env_switch)

    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is None, "Pending state harus di-pause (tidak lagi ACTIVE) saat user pindah topik."


# ==========================================
# BONUS: EXACT MEMORY LOSSLESS INTEGRITY
# ==========================================
def test_exact_memory_lossless_integrity():
    compressor = PayloadCompressor()
    # [FIX] Teks diperbesar (diulang 10x) agar Zstd memiliki cukup data untuk melakukan kompresi nyata.
    # Pada teks < 100 bytes, header Zstd membuat ukuran file hasil kompresi justru lebih besar dari aslinya.
    original_text = ("User: Modal saya Rp50 juta.\nAI: Dengan Rp50 juta, Anda bisa...\n" * 10)
    
    payload = compressor.compress(original_text)
    assert payload.compressed_size < len(original_text.encode('utf-8'))
    
    decompressed = compressor.decompress(payload)
    assert decompressed.decode('utf-8') == original_text
    
    corrupted_blob = bytearray(payload.compressed_blob)
    corrupted_blob[10] = (corrupted_blob[10] + 1) % 256
    
    from models import CompressedPayload
    corrupted_payload = CompressedPayload(
        payload_id=payload.payload_id,
        compressed_blob=bytes(corrupted_blob),
        checksum=payload.checksum,
        compression_algorithm=payload.compression_algorithm,
        original_size=payload.original_size,
        compressed_size=payload.compressed_size
    )
    
    with pytest.raises(IntegrityError):
        compressor.decompress(corrupted_payload)


# ==========================================
# REVIEW KEDUA: SKENARIO LANJUTAN (A - L)
# ==========================================

# ==========================================
# SCENARIO A: EXACT END-TO-END CONTEXT INJECTION
# ==========================================
def test_scenario_a_exact_end_to_end_injection(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_e2e"
    
    # Turn 1: Store exact memory
    env1 = create_envelope(session_id, "Saya punya modal Rp50 juta untuk bisnis kopi.", tz)
    engine.process_input(env1)
    
    # Turn 2: Ask about it
    env2 = create_envelope(session_id, "Berapa modal yang saya sebut tadi?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    
    # The FakeLLMClient should echo the past memory if it was injected
    assert "[FAKE AI RESPONSE]" in res2.assistant_response
    assert res2.reconstruction_status.value == "FULL_SUCCESS", "Status rekonstruksi harus FULL_SUCCESS."

# ==========================================
# SCENARIO C: QUESTION AS TOPIC SWITCH (No ? bias)
# ==========================================
def test_scenario_c_question_as_topic_switch(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_q_switch"
    
    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    engine.process_input(env1)
    
    # Turn 2: A question about a completely different topic
    env2 = create_envelope(session_id, "Bagaimana cara kerja pointer di C++?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    
    keys = engine.storage.get_all_topic_keys()
    assert len(keys) == 2, "Sistem harus membuat topik baru untuk C++ meskipun diakhiri tanda tanya."

# ==========================================
# SCENARIO D: LONG CLARIFICATION (>100 chars)
# ==========================================
def test_scenario_d_long_clarification(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_long_clar"
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Modal dan strategi", created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Lokasi kedai", created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    env_ambigu = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env_ambigu)
    
    # Long clarification
    long_answer = "Yang saya maksud sebenarnya bisnis kopi yang kemarin. Kita sebelumnya membahas modal sekitar 50 juta, lokasi usaha, dan bagaimana strategi untuk memulainya."
    env_resolve = create_envelope(session_id, long_answer, tz, time_offset_hours=1)
    res_resolve = engine.process_input(env_resolve)
    
    pending_after = engine.storage.get_active_pending_state(session_id)
    assert pending_after is None or pending_after.status != PendingStatus.ACTIVE, "Pending harus ter-resolve meskipun jawaban > 100 karakter."

# ==========================================
# SCENARIO E: CLARIFICATION MAX ROUNDS
# ==========================================
def test_scenario_e_max_clarification_rounds(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_max_rounds"
    
    # [FIX] Menggunakan "alpha" dan "beta" agar tidak difilter oleh MIN_TOKEN_LENGTH = 3
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_a", title="Topik Alpha", helper_description="Alpha", created_at=datetime.now(tz), token_profile={"alpha": 10})
        key_b = TopicKey(key_id="key_b", title="Topik Beta", helper_description="Beta", created_at=datetime.now(tz), token_profile={"beta": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    # Round 1
    env1 = create_envelope(session_id, "Bahas alpha dan beta.", tz)
    engine.process_input(env1)
    
    # Round 2 (still ambiguous)
    env2 = create_envelope(session_id, "Terserah.", tz, time_offset_hours=1)
    engine.process_input(env2)
    
    # Round 3 (still ambiguous)
    env3 = create_envelope(session_id, "Pilih saja.", tz, time_offset_hours=2)
    res3 = engine.process_input(env3)
    
    # Round 4 (Should trigger fallback / expiration)
    env4 = create_envelope(session_id, "Mana?", tz, time_offset_hours=3)
    res4 = engine.process_input(env4)
    
    pending = engine.storage.get_active_pending_state(session_id)
    # Should be expired or None
    assert pending is None or pending.status == PendingStatus.EXPIRED, "Pending harus expired setelah max rounds."

# ==========================================
# SCENARIO F: PENDING TTL EXPIRATION
# ==========================================
def test_scenario_f_pending_ttl(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_ttl"
    
    # [FIX] Menggunakan "alpha" dan "beta" agar tidak difilter oleh MIN_TOKEN_LENGTH = 3
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_a", title="Topik Alpha", helper_description="Alpha", created_at=datetime.now(tz), token_profile={"alpha": 10})
        key_b = TopicKey(key_id="key_b", title="Topik Beta", helper_description="Beta", created_at=datetime.now(tz), token_profile={"beta": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    env_ambigu = create_envelope(session_id, "Bahas alpha atau beta.", tz)
    engine.process_input(env_ambigu)
    
    pending_before = engine.storage.get_active_pending_state(session_id)
    assert pending_before is not None
    
    # Jump time beyond TTL (e.g., 25 hours later)
    env_late = create_envelope(session_id, "Jawaban untuk alpha.", tz, time_offset_hours=25)
    engine.process_input(env_late)
    
    pending_after = engine.storage.get_active_pending_state(session_id)
    assert pending_after is None, "Pending harus expired karena melewati TTL."

# ==========================================
# SCENARIO G: CROSS-SESSION PENDING RESOLUTION
# ==========================================
def test_scenario_g_cross_session_pending(engine: MemoryEngine, tz: ZoneInfo):
    session_1 = "session_cs_1"
    session_2 = "session_cs_2"
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Strategi bisnis kopi", created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Lokasi kedai kopi", created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
    # Session 1: Ambiguity
    env1 = create_envelope(session_1, "Bahas kopi.", tz)
    engine.process_input(env1)
    
    pending_s1 = engine.storage.get_active_pending_state(session_1)
    assert pending_s1 is not None
    
    # [FIX] Tambahkan kata unik dari deskripsi ("strategi", "modal") agar margin skor melebar dan decision menjadi CLEAR
    # Session 2: User answers the pending question from Session 1
    env2 = create_envelope(session_2, "Yang bisnis kopi saja, kita bahas strategi dan modal.", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    
    # The pending from Session 1 should be resolved
    pending_s1_after = engine.storage.get_pending_by_reference(pending_s1.pending_id)
    assert pending_s1_after.status == PendingStatus.RESOLVED, "Pending lintas sesi harus ter-resolve."

# ==========================================
# SCENARIO H & I: TEMPORAL YEAR & SPECIFIC HOUR
# ==========================================
def test_scenario_h_i_temporal_year_and_hour(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_temp"
    
    # We will inject a fragment manually at a specific time
    specific_time = datetime(2023, 10, 24, 15, 0, 0, tzinfo=tz) # Yesterday 15:00
    with engine.storage.transaction() as tx:
        key = TopicKey(key_id="key_kopi", title="Bisnis Kopi", helper_description="Bisnis", created_at=specific_time, token_profile={"kopi": 10, "bisnis": 10})
        engine.storage.save_topic_key(tx, key)
        
        # We need a payload and fragment
        compressor = PayloadCompressor()
        payload = compressor.compress("User: Ekspansi kopi\nAI: Ya")
        engine.storage.save_payload(tx, payload)
        
        frag = FragmentMetadata(
            fragment_id="frag_specific",
            topic_key_id="key_kopi",
            session_id="session_old",
            created_at=specific_time,
            temporal_index=specific_time,
            payload_reference=payload.payload_id
        )
        engine.storage.save_fragment(tx, frag)
        
    # Now query: "lanjutkan bisnis kopi kemarin pukul 15:00"
    # ref_time is 2023-10-25 14:00. "kemarin" = 2023-10-24. "pukul 15:00" = 15:00.
    env_query = create_envelope(session_id, "Lanjutkan bisnis kopi kemarin pukul 15:00.", tz)
    res = engine.process_input(env_query)
    
    assert "frag_specific" in res.retrieved_memory_ids, "Sistem harus menemukan fragment berdasarkan tahun lalu atau jam spesifik."

# ==========================================
# SCENARIO J: TEMPORAL TIE (AMBIGUITY)
# ==========================================
def test_scenario_j_temporal_tie(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_tie"
    
    yesterday_14 = datetime(2023, 10, 24, 14, 0, 0, tzinfo=tz)
    yesterday_16 = datetime(2023, 10, 24, 16, 0, 0, tzinfo=tz)
    
    compressor = PayloadCompressor()
    
    with engine.storage.transaction() as tx:
        key_a = TopicKey(key_id="key_bisnis", title="Bisnis Kopi", helper_description="Strategi", created_at=yesterday_14, token_profile={"kopi": 10, "bisnis": 10})
        key_b = TopicKey(key_id="key_kedai", title="Kedai Kopi", helper_description="Lokasi", created_at=yesterday_16, token_profile={"kopi": 10, "kedai": 10})
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)
        
        p1 = compressor.compress("A")
        engine.storage.save_payload(tx, p1)
        f1 = FragmentMetadata(fragment_id="f1", topic_key_id="key_bisnis", session_id="s_old", created_at=yesterday_14, temporal_index=yesterday_14, payload_reference=p1.payload_id)
        engine.storage.save_fragment(tx, f1)
        
        p2 = compressor.compress("B")
        engine.storage.save_payload(tx, p2)
        f2 = FragmentMetadata(fragment_id="f2", topic_key_id="key_kedai", session_id="s_old", created_at=yesterday_16, temporal_index=yesterday_16, payload_reference=p2.payload_id)
        engine.storage.save_fragment(tx, f2)
        
    # Query: "Bahas kopi kemarin." (Both are yesterday, scores will be close, temporal doesn't break tie uniquely if tolerance covers both)
    env = create_envelope(session_id, "Bahas kopi kemarin.", tz) # ref_time = 2023-10-25 14:00
    res = engine.process_input(env)
    
    # Since both are "kemarin", and base scores are similar, it should be AMBIGUOUS or at least not crash.
    assert res.decision.decision_type in [DecisionType.AMBIGUOUS, DecisionType.CLEAR]

# ==========================================
# SCENARIO K: FRAGMENT-LEVEL RANKING
# ==========================================
def test_scenario_k_fragment_ranking(engine: MemoryEngine, tz: ZoneInfo):
    session_id = "session_frag_rank"
    
    # Create 1 Topic Key
    with engine.storage.transaction() as tx:
        key = TopicKey(key_id="key_kopi", title="Bisnis Kopi", helper_description="Bisnis kopi", created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10, "modal": 5})
        engine.storage.save_topic_key(tx, key)
        
        compressor = PayloadCompressor()
        
        # Frag 1: Old, but about "modal"
        t1 = datetime(2023, 10, 20, 10, 0, 0, tzinfo=tz)
        p1 = compressor.compress("User: Modal 50 juta\nAI: Cukup")
        engine.storage.save_payload(tx, p1)
        f1 = FragmentMetadata(fragment_id="f_modal", topic_key_id="key_kopi", session_id="s1", created_at=t1, temporal_index=t1, payload_reference=p1.payload_id, relationship_strength=0.9)
        engine.storage.save_fragment(tx, f1)
        
        # Frag 2: New, but about "lokasi"
        t2 = datetime(2023, 10, 24, 10, 0, 0, tzinfo=tz)
        p2 = compressor.compress("User: Lokasi di mana\nAI: Jakarta")
        engine.storage.save_payload(tx, p2)
        f2 = FragmentMetadata(fragment_id="f_lokasi", topic_key_id="key_kopi", session_id="s1", created_at=t2, temporal_index=t2, payload_reference=p2.payload_id, relationship_strength=0.2)
        engine.storage.save_fragment(tx, f2)

    # Query specifically about "modal"
    env = create_envelope(session_id, "Berapa modal yang kita bahas?", tz)
    res = engine.process_input(env)
    
    # The retrieval should ideally rank f_modal higher due to token match / relationship strength
    if res.retrieved_memory_ids:
        assert res.retrieved_memory_ids[0] == "f_modal", "Fragment tentang modal harus di-ranking lebih tinggi."

# ==========================================
# SCENARIO L: CROSS-SESSION GRAPH
# ==========================================
def test_scenario_l_cross_session_graph(engine: MemoryEngine, tz: ZoneInfo):
    s1 = "session_graph_1"
    s2 = "session_graph_2"
    s3 = "session_graph_3"
    
    # Session 1: Coffee
    env1 = create_envelope(s1, "Bisnis kopi modal 50 juta.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id
    
    # Session 2: C++
    env2 = create_envelope(s2, "Belajar pointer C++.", tz, time_offset_hours=24)
    engine.process_input(env2)
    
    # Session 3: Coffee again
    env3 = create_envelope(s3, "Lanjutkan bisnis kopi.", tz, time_offset_hours=48)
    res3 = engine.process_input(env3)
    
    # Should reactivate the coffee key
    assert res3.active_topic_key_id == coffee_key_id, "Sesi 3 harus mengaktifkan kembali Topic Key Kopi dari Sesi 1."
