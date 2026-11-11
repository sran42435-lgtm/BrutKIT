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

    env_resolve = create_envelope(session_id, "Yang bisnis saja.", tz, time_offset_hours=1)
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
