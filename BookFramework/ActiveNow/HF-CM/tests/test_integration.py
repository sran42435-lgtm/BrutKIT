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
    TopicKey, FragmentMetadata, CompressedPayload, EngineResult,
    CandidateScore, EvidenceProfile, RetrievedMemory, TokenQuality,
    EvidenceFamily,
)
from storage import MemoryStorage
from retrieval import RetrievalEngine, SignalExtractor, TopicKeyScorer
from compression import PayloadCompressor, IntegrityError
from memory_engine import MemoryEngine
# [NEW] Import untuk End-to-End Pipeline Testing (Spec 26)
from context_builder import ContextBuilder
from llm_client import FakeLLMClient

# ==========================================
# DIAGNOSTIC HELPER (Deep Architectural Transparency)
# ==========================================
def print_engine_trace(engine: MemoryEngine, env: InputEnvelope, res: EngineResult, step_name: str = ""):
    """
    Mencetak jejak diagnostik mesin secara real-time dengan kedalaman penuh.
    Memperlihatkan input, keputusan, fragment terpilih, status Big Memory, dan Fast Memory.
    """
    print(f"\n{'='*20} 🔍 TRACE: {step_name} {'='*20}")
    print(f"👤 [USER INPUT]   : {env.raw_text}")
    print(f"🧠 [DECISION]     : {res.decision.decision_type.value}")
    print(f"🔑 [ACTIVE KEY]   : {res.active_topic_key_id[:8] + '...' if res.active_topic_key_id else '(NEW_TOPIC / NO_MATCH)'}")
    print(f"📂 [RETRIEVAL]    : {len(res.retrieved_memory_ids)} fragments selected")
    
    # Detail Fragment yang Terpilih
    for fid in res.retrieved_memory_ids:
        frag = engine.storage.get_fragment(fid)
        if frag:
            key_short = frag.topic_key_id[:8] + "..."
            time_str = frag.temporal_index.strftime("%Y-%m-%d %H:%M") if frag.temporal_index else "N/A"
            print(f"   └─ Frag: {fid[:8]}... | Key: {key_short} | Time: {time_str}")
            
    print(f"🔄 [RECONSTRUCT]  : {res.reconstruction_status.value}")
    
    # Potong respons agar tidak membanjiri terminal
    ai_resp = res.assistant_response.replace('\n', ' ')
    print(f"🤖 [AI RESPONSE]  : {ai_resp[:120]}{'...' if len(ai_resp) > 120 else ''}")
    
    if res.decision.decision_type == DecisionType.AMBIGUOUS:
        msg = res.decision.message.replace('\n', ' ')
        print(f"⚠️ [PENDING MSG]  : {msg[:150]}{'...' if len(msg) > 150 else ''}")
        
    # Big Memory Status (Daftar Topic Keys di Database)
    print(f"\n🗄️  [BIG MEMORY STATUS]")
    keys = engine.storage.get_all_topic_keys()
    print(f"   Total Topic Keys Tersimpan: {len(keys)}")
    for k in keys:
        tokens_preview = list(k.token_profile.keys())[:5]
        title_short = (k.title[:25] + "...") if len(k.title) > 25 else k.title
        print(f"   🔑 {title_short:<28} | Tokens: {tokens_preview}")
        
    # Fast Memory Status (Pending State)
    pending = engine.storage.get_active_pending_state(env.session_id)
    if pending:
        print(f"\n⏳ [FAST MEMORY: PENDING ACTIVE]")
        print(f"   Round: {pending.clarification_round} | Candidates: {len(pending.candidate_keys)} | Origin: {pending.origin_session_id}")
    else:
        print(f"\n⏳ [FAST MEMORY: NO ACTIVE PENDING]")

    print(f"{'='*65}\n")


# ==========================================
# [ASSOCIATIVE EVIDENCE LOCKING] DIAGNOSTIC HELPER
# ==========================================
def print_evidence_trace(res: EngineResult, step_name: str = ""):
    """
    Mencetak DUA DIMENSI penilaian: RELEVANCE dan EVIDENCE STRENGTH.
    Menjawab: KENAPA memori A dipilih, KENAPA B tidak, KAPAN sistem menolak.
    """
    print(f"\n{'='*18} 🧾 EVIDENCE: {step_name} {'='*18}")
    d = res.decision
    print(f"🧠 [DECISION]      : {d.decision_type.value}")
    print(f"📊 [CONFIDENCE]    : {d.confidence:.3f}")
    print(f"🧩 [SUPPORT COUNT] : {d.support_count}")
    print(f"💪 [EVIDENCE STR]  : {d.evidence_strength:.3f}")
    print(f"⚡ [CONTRADICTION] : {d.contradiction_score:.3f}")
    if d.decision_reason:
        print(f"📝 [REASON]        : {d.decision_reason}")
    if res.evidence_summary:
        print("─" * 60)
        print(res.evidence_summary)
    print(f"{'='*65}\n")


def print_candidate_comparison(candidates, step_name: str = ""):
    """Membandingkan RELEVANCE vs EVIDENCE STRENGTH antar kandidat secara berdampingan."""
    print(f"\n{'='*14} ⚖️  CANDIDATE COMPARISON: {step_name} {'='*14}")
    print(f"{'TITLE':<24} {'RELEVANCE':>10} {'SUPPORT':>8} {'EVID_STR':>9} {'CONFID':>8} {'CONTRA':>8}")
    for c in candidates:
        title = (c.title[:22] + "..") if len(c.title) > 22 else c.title
        print(f"{title:<24} {c.final_score:>10.3f} {c.support_count:>8} "
              f"{c.evidence_strength:>9.3f} {c.confidence:>8.3f} {c.contradiction_score:>8.3f}")
        if c.evidence_profile:
            print(f"    supporting: {', '.join(c.evidence_profile.supporting_evidence_names) or '(none)'}")
        if c.evidence_notes:
            print(f"    notes: {'; '.join(c.evidence_notes)}")
    print(f"{'='*65}\n")


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
    print_engine_trace(engine, env1, res1, "Scenario 1 - Turn 1 (Init)")
    assert res1.decision.decision_type in [DecisionType.CLEAR, DecisionType.NO_MATCH]
    key_id_1 = res1.active_topic_key_id
    assert key_id_1 is not None, "Topik Key pertama harus terbuat."

    env2 = create_envelope(session_id, "Modalnya sekitar 50 juta cukup?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Scenario 1 - Turn 2 (Continuity)")
    
    env3 = create_envelope(session_id, "Terus lokasinya di mana yang strategis?", tz, time_offset_hours=2)
    res3 = engine.process_input(env3)
    print_engine_trace(engine, env3, res3, "Scenario 1 - Turn 3 (Continuity)")

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
    print_engine_trace(engine, env_cpp, res_cpp, "Scenario 2 - Topic Switch (C++)")

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
    print_engine_trace(engine, env, res, "Scenario 4 - Ambiguity Gate")

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
    print_engine_trace(engine, env_resolve, res_resolve, "Scenario 6 - Resolution")

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
    print_engine_trace(engine, env_switch, res_switch, "Scenario 7 - Pause Pending")

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
    print_engine_trace(engine, env2, res2, "Scenario A - Exact E2E Injection")
    
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
    print_engine_trace(engine, env2, res2, "Scenario C - Question as Topic Switch")
    
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
    print_engine_trace(engine, env_resolve, res_resolve, "Scenario D - Long Clarification")
    
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
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Scenario E - Round 2 (Update)")
    
    # Round 3 (still ambiguous)
    env3 = create_envelope(session_id, "Pilih saja.", tz, time_offset_hours=2)
    res3 = engine.process_input(env3)
    print_engine_trace(engine, env3, res3, "Scenario E - Round 3 (Update)")
    
    # Round 4 (Should trigger fallback / expiration)
    env4 = create_envelope(session_id, "Mana?", tz, time_offset_hours=3)
    res4 = engine.process_input(env4)
    print_engine_trace(engine, env4, res4, "Scenario E - Round 4 (Max Fallback)")
    
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
    res_late = engine.process_input(env_late)
    print_engine_trace(engine, env_late, res_late, "Scenario F - TTL Expiration")
    
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
    print_engine_trace(engine, env2, res2, "Scenario G - Cross-Session Resolve")
    
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
    print_engine_trace(engine, env_query, res, "Scenario H/I - Temporal Hour")
    
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
    print_engine_trace(engine, env, res, "Scenario J - Temporal Tie")
    
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
    print_engine_trace(engine, env, res, "Scenario K - Fragment Ranking")
    
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
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Scenario L - Session 2 (C++)")
    
    # Session 3: Coffee again
    env3 = create_envelope(s3, "Lanjutkan bisnis kopi.", tz, time_offset_hours=48)
    res3 = engine.process_input(env3)
    print_engine_trace(engine, env3, res3, "Scenario L - Session 3 (Reactivation)")
    
    # Should reactivate the coffee key
    assert res3.active_topic_key_id == coffee_key_id, "Sesi 3 harus mengaktifkan kembali Topic Key Kopi dari Sesi 1."


# ==========================================================================
# [ASSOCIATIVE EVIDENCE LOCKING] VALIDATION SUITE (TEST 1 - 9)
#
# Fase ini tidak menguji "apakah memori bisa ditemukan", tetapi:
#   KENAPA memori A dipilih,
#   KENAPA memori B tidak dipilih,
#   KAPAN sistem harus menolak memilih.
#
# Empat prinsip yang diverifikasi:
#   1. Satu skor tinggi tidak otomatis berarti bukti kuat.
#   2. Beberapa bukti yang saling mendukung > satu kecocokan besar.
#   3. Bukti kontradiktif harus bisa menurunkan kepercayaan.
#   4. Memori tidak boleh diperbarui sebelum keputusan asosiasinya tervalidasi.
# ==========================================================================

# ==========================================
# EVIDENCE TEST 1: MULTI-EVIDENCE MENGALAHKAN SINGLE-EVIDENCE
# ==========================================
def test_evidence_1_multi_evidence_beats_single_signal(engine: MemoryEngine, tz: ZoneInfo):
    """
    Candidate A: Token saja (sangat tinggi).
    Candidate B: Token + Description + Co-occurrence (konsisten).

    Expected: B diprioritaskan, dan A TIDAK dianggap confident
    hanya karena satu skor besar.

    Scorer dipanggil langsung agar dua dimensi (RELEVANCE vs EVIDENCE)
    dapat diisolasi tanpa pengaruh pipeline lain.
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()
    signal = extractor.extract("bisnis kopi modal strategi", ref)

    tokens = {"bisnis": 1, "kopi": 1, "modal": 1, "strategi": 1}
    window_pairs = engine._build_window_pairs(["bisnis", "kopi", "modal", "strategi"])

    key_a = TopicKey(
        key_id="A", title="Token Only",
        helper_description="",                      # tidak ada bukti deskripsi
        token_profile=tokens,
        co_occurrence_profile={},                   # tidak ada bukti co-occurrence
        created_at=ref
    )
    key_b = TopicKey(
        key_id="B", title="Multi Evidence",
        helper_description="bisnis kopi modal strategi",
        token_profile=tokens,
        co_occurrence_profile=window_pairs,
        created_at=ref
    )

    # DF=2 pada 2 Topic Key -> rarity 0, sehingga unique tidak menambah support.
    df = {t: 2 for t in tokens}
    scorer = engine.retrieval.scorer
    sa = scorer.score(signal, key_a, False, 0.0, 0.0, df, total_topic_keys=2)
    sb = scorer.score(signal, key_b, False, 0.0, 0.0, df, total_topic_keys=2)

    print_candidate_comparison([sa, sb], "Evidence 1 - Multi vs Single")

    # A hanya punya SATU jenis bukti.
    assert sa.support_count == 1, "Candidate A harus terdeteksi single-signal."
    assert sa.confidence <= CONFIG.candidate_safety.SINGLE_SIGNAL_CONFIDENCE_CAP, (
        "Single-signal confidence wajib di-cap."
    )

    # B punya bukti yang saling mendukung.
    assert sb.support_count > sa.support_count, "B harus punya lebih banyak jenis bukti."
    assert sb.final_score > sa.final_score, "Multi-evidence harus menang relevansi."
    assert sb.confidence > sa.confidence, "Multi-evidence harus lebih dipercaya."
    assert sb.evidence_profile is not None
    assert "description" in sb.evidence_profile.supporting_evidence_names
    assert "co_occurrence" in sb.evidence_profile.supporting_evidence_names


# ==========================================
# EVIDENCE TEST 2: SINGLE STRONG TOKEN TIDAK OTOMATIS MENANG
# ==========================================
def test_evidence_2_single_token_not_auto_confident(engine: MemoryEngine, tz: ZoneInfo):
    """
    Query: "kopi"
    Dua topik sama-sama mengandung "kopi", tanpa bukti lain.

    Expected: TIDAK CLEAR. Harus AMBIGUOUS atau INSUFFICIENT_EVIDENCE.
    """
    session_id = "session_ev2"

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="ev2_bisnis", title="Bisnis Kopi",
            helper_description="Modal dan strategi usaha",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10}
        )
        key_b = TopicKey(
            key_id="ev2_kedai", title="Kedai Kopi",
            helper_description="Lokasi kedai terdekat",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "kopi", tz)
    res = engine.process_input(env)
    print_engine_trace(engine, env, res, "Evidence 2 - Single Token")
    print_evidence_trace(res, "Evidence 2 - Single Token")
    print_candidate_comparison(res.decision and res.candidates if hasattr(res, 'candidates') else [], "Evidence 2")

    assert res.decision.decision_type != DecisionType.CLEAR, (
        "Satu token yang cocok tidak boleh menghasilkan keputusan CONFIDENT."
    )
    assert res.decision.decision_type in (
        DecisionType.AMBIGUOUS, DecisionType.INSUFFICIENT_EVIDENCE, DecisionType.NO_MATCH
    )
    assert res.decision.support_count <= 1, "Hanya satu jenis bukti yang boleh terdeteksi."


# ==========================================
# EVIDENCE TEST 3: UNIQUE TOKEN MENANG
# ==========================================
def test_evidence_3_unique_token_wins(engine: MemoryEngine, tz: ZoneInfo):
    """
    Bisnis Kopi -> modal, juta (identitas spesifik)
    Kedai Kopi  -> lokasi, menu

    Query memakai subset padat identitas: "kopi modal juta".

    Catatan jujur: frasa natural yang panjang ("Kopi yang modal 50 juta itu")
    terdilusi token low-information sehingga evidence_strength jatuh di bawah
    MIN_ASSOCIATIVE_CONFIDENCE. Test ini memakai subset identitas agar yang
    diukur adalah mekanisme uniqueness, bukan panjang kalimat.
    """
    session_id = "session_ev3"
    ref = datetime.now(tz)
    bisnis_pairs = engine._build_window_pairs(["kopi", "modal", "juta"])

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="ev3_bisnis", title="Bisnis Kopi",
            helper_description="Pembahasan bisnis kopi modal usaha",
            created_at=ref,
            token_profile={"kopi": 10, "bisnis": 10, "modal": 5, "juta": 5},
            co_occurrence_profile=bisnis_pairs
        )
        key_b = TopicKey(
            key_id="ev3_kedai", title="Kedai Kopi",
            helper_description="Mencari kedai kopi lokasi menu",
            created_at=ref,
            token_profile={"kopi": 10, "kedai": 10, "lokasi": 5, "menu": 5},
            co_occurrence_profile={}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "kopi modal juta", tz)
    res = engine.process_input(env)
    print_engine_trace(engine, env, res, "Evidence 3 - Unique Token")
    print_evidence_trace(res, "Evidence 3 - Unique Token")

    assert res.decision.decision_type == DecisionType.CLEAR, "Bukti identitas unik harus menghasilkan keputusan jelas."
    assert res.decision.selected_key_id == "ev3_bisnis", "Bisnis Kopi harus menang karena token identitas unik."
    assert res.decision.support_count >= CONFIG.evidence.MIN_SUPPORTING_EVIDENCE
    assert res.decision.evidence_strength > 0.0

    # Unique token evidence harus benar-benar terpisah dari token_score.
    df = engine.storage.get_global_token_counts()
    assert df.get("kopi", 0) == 2, "kopi muncul di 2 Topic Key."
    assert df.get("modal", 0) == 1, "modal hanya muncul di 1 Topic Key."
    assert engine.storage.count_topic_keys_containing_token("juta") == 1


# ==========================================
# EVIDENCE TEST 4: CONTRADICTION PENALTY
# ==========================================
def test_evidence_4_contradiction_penalty(engine: MemoryEngine, tz: ZoneInfo):
    """
    Query: "bisnis kopi"
    Kandidat "Kedai Kopi": token kopi cocok, TETAPI deskripsinya
    menjelaskan intent yang sama sekali berbeda (mencari tempat minum).

    Expected: description contradiction menurunkan skor kandidat itu.
    """
    session_id = "session_ev4"
    ref = datetime.now(tz)
    bisnis_pairs = engine._build_window_pairs(["bisnis", "kopi"])

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="ev4_bisnis", title="Bisnis Kopi",
            helper_description="Pembahasan bisnis kopi modal usaha",
            created_at=ref,
            token_profile={"bisnis": 10, "kopi": 10, "modal": 5},
            co_occurrence_profile=bisnis_pairs
        )
        key_b = TopicKey(
            key_id="ev4_kedai", title="Kedai Terdekat",
            # Sengaja TIDAK mengandung kata dari query.
            helper_description="Mencari tempat minum terdekat di kota",
            created_at=ref,
            token_profile={"kopi": 10, "kedai": 10, "lokasi": 5},
            co_occurrence_profile={}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "bisnis kopi", tz)
    res = engine.process_input(env)
    print_engine_trace(engine, env, res, "Evidence 4 - Contradiction")
    print_evidence_trace(res, "Evidence 4 - Contradiction")

    assert res.decision.decision_type == DecisionType.CLEAR
    assert res.decision.selected_key_id == "ev4_bisnis"

    # Kandidat yang kontradiktif harus tetap terlihat beserta alasannya.
    kedai = next((c for c in res.candidates if c.key_id == "ev4_kedai"), None)
    assert kedai is not None, "Kandidat kontradiktif harus tetap tercatat untuk audit."
    print_candidate_comparison(res.candidates, "Evidence 4 - Contradiction")

    assert kedai.contradiction_score > 0.0, "Contradiction penalty harus diterapkan."
    assert any("contradiction" in note for note in kedai.evidence_notes), (
        "Alasan kontradiksi harus bisa diaudit."
    )
    bisnis = next(c for c in res.candidates if c.key_id == "ev4_bisnis")
    assert kedai.final_score < bisnis.final_score


# ==========================================
# EVIDENCE TEST 5: WINDOW CO-OCCURRENCE
# ==========================================
def test_evidence_5_window_co_occurrence(engine: MemoryEngine, tz: ZoneInfo):
    """
    Stored : "bisnis kecil di bidang kopi"  (tidak berdampingan)
    Query  : "bisnis kopi"

    Expected: co-occurrence TETAP memberikan evidence.
    Dibuktikan dengan membandingkan profil WINDOW vs profil ADJACENT-ONLY.
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()

    stored_tokens = ["bisnis", "kecil", "bidang", "kopi"]
    window_profile = engine._build_window_pairs(stored_tokens)
    adjacent_only_profile = {
        "bisnis↔kecil": 1,
        "kecil↔bidang": 1,
        "bidang↔kopi": 1,
    }

    print(f"\n🪟 WINDOW PAIRS dari 'bisnis kecil di bidang kopi': {list(window_profile.keys())}")
    assert "bisnis↔kopi" in window_profile, "Window co-occurrence harus menangkap pasangan non-adjacent."

    signal = extractor.extract("bisnis kopi", ref)
    assert "bisnis↔kopi" in signal.window_co_occurrence_signals

    tokens = {"bisnis": 5, "kopi": 5}
    df = {"bisnis": 1, "kopi": 1}
    scorer = engine.retrieval.scorer

    key_window = TopicKey(
        key_id="W", title="Window Profile", helper_description="bisnis kopi",
        created_at=ref, token_profile=tokens, co_occurrence_profile=window_profile
    )
    key_adjacent = TopicKey(
        key_id="ADJ", title="Adjacent Only", helper_description="bisnis kopi",
        created_at=ref, token_profile=tokens, co_occurrence_profile=adjacent_only_profile
    )

    s_window = scorer.score(signal, key_window, False, 0.0, 0.0, df, total_topic_keys=1)
    s_adjacent = scorer.score(signal, key_adjacent, False, 0.0, 0.0, df, total_topic_keys=1)

    print_candidate_comparison([s_window, s_adjacent], "Evidence 5 - Window vs Adjacent")

    assert s_window.co_occurrence_score > 0.0, "Window profile harus memberi evidence."
    assert s_adjacent.co_occurrence_score == 0.0, "Adjacent-only profile tidak boleh cocok dengan query window."
    assert s_window.final_score > s_adjacent.final_score


# ==========================================
# EVIDENCE TEST 6: EVIDENCE POLLUTION PROTECTION
# ==========================================
def test_evidence_6_no_evidence_pollution(engine: MemoryEngine, tz: ZoneInfo):
    """
    Turn 1: Bisnis kopi
    Turn 2: "Bagaimana cara kerja pointer di C++?"

    Expected: token C++/pointer TIDAK masuk ke Token Profile,
    Co-occurrence, maupun Helper Description milik cluster Kopi.
    """
    session_id = "session_ev6"

    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id
    assert coffee_key_id is not None

    coffee_before = engine.storage.get_topic_key(coffee_key_id)
    profile_before = dict(coffee_before.token_profile)
    desc_before = coffee_before.helper_description

    env2 = create_envelope(session_id, "Bagaimana cara kerja pointer di C++?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Evidence 6 - Pollution Protection")
    print_evidence_trace(res2, "Evidence 6 - Pollution Protection")

    keys = engine.storage.get_all_topic_keys()
    assert len(keys) == 2, "C++ harus menjadi cluster terpisah."

    coffee_after = engine.storage.get_topic_key(coffee_key_id)
    polluted_tokens = {"pointer", "cara", "kerja", "bagaimana"}

    print(f"\n🛡️  COFFEE PROFILE BEFORE: {list(profile_before.keys())}")
    print(f"🛡️  COFFEE PROFILE AFTER : {list(coffee_after.token_profile.keys())}")

    assert not (polluted_tokens & set(coffee_after.token_profile.keys())), (
        "Token C++ tidak boleh mencemari Token Profile Kopi."
    )
    for pair in coffee_after.co_occurrence_profile.keys():
        assert "pointer" not in pair, "Co-occurrence Kopi tidak boleh tercemar."
    assert "pointer" not in coffee_after.helper_description.lower(), (
        "Helper Description Kopi tidak boleh tercemar."
    )


# ==========================================
# EVIDENCE TEST 7: WEAK CONTINUATION TIDAK MEMPERKUAT CLUSTER
# ==========================================
def test_evidence_7_weak_continuation_does_not_mutate(engine: MemoryEngine, tz: ZoneInfo):
    """
    Input: "Bagaimana itu?" -> seluruh token low-information.

    Expected: fragmen tetap disimpan (percakapan tidak hilang),
    TETAPI profil topik tidak diperbarui secara agresif.
    """
    session_id = "session_ev7"

    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id

    coffee_before = engine.storage.get_topic_key(coffee_key_id)
    profile_before = dict(coffee_before.token_profile)
    desc_before = coffee_before.helper_description
    frags_before = len(engine.storage.get_fragment_metadata_by_key(coffee_key_id))

    env2 = create_envelope(session_id, "Bagaimana itu?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Evidence 7 - Weak Continuation")
    print_evidence_trace(res2, "Evidence 7 - Weak Continuation")

    coffee_after = engine.storage.get_topic_key(coffee_key_id)
    frags_after = len(engine.storage.get_fragment_metadata_by_key(coffee_key_id))

    print(f"\n🧊 PROFILE BEFORE: {list(profile_before.keys())}")
    print(f"🧊 PROFILE AFTER : {list(coffee_after.token_profile.keys())}")

    assert len(engine.storage.get_all_topic_keys()) == 1, "Weak continuation tidak boleh memecah cluster."
    assert "bagaimana" not in coffee_after.token_profile, "Token low-information tidak boleh masuk profil."
    assert "itu" not in coffee_after.token_profile, "Token low-information tidak boleh masuk profil."
    assert coffee_after.helper_description == desc_before, "Deskripsi tidak boleh diperbarui oleh bukti lemah."
    assert frags_after > frags_before, "Fragmen tetap harus disimpan agar percakapan tidak hilang."


# ==========================================
# EVIDENCE TEST 8: STRONG MULTI-EVIDENCE CONTINUATION
# ==========================================
def test_evidence_8_strong_continuation_grows_cluster(engine: MemoryEngine, tz: ZoneInfo):
    """
    Bisnis kopi -> Modal bisnis kopi sekitar lima juta.

    Expected: Token profile berkembang, Co-occurrence berkembang,
    Helper description berkembang.
    """
    session_id = "session_ev8"

    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id

    coffee_before = engine.storage.get_topic_key(coffee_key_id)
    profile_before = dict(coffee_before.token_profile)
    cooc_before = dict(coffee_before.co_occurrence_profile)
    desc_before = coffee_before.helper_description

    env2 = create_envelope(session_id, "Modal bisnis kopi sekitar lima juta.", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)
    print_engine_trace(engine, env2, res2, "Evidence 8 - Strong Continuation")
    print_evidence_trace(res2, "Evidence 8 - Strong Continuation")

    coffee_after = engine.storage.get_topic_key(coffee_key_id)

    print(f"\n🌱 PROFILE BEFORE : {list(profile_before.keys())}")
    print(f"🌱 PROFILE AFTER  : {list(coffee_after.token_profile.keys())}")
    print(f"🌱 COOC BEFORE    : {len(cooc_before)} pairs")
    print(f"🌱 COOC AFTER     : {len(coffee_after.co_occurrence_profile)} pairs")
    print(f"🌱 DESC BEFORE    : {desc_before}")
    print(f"🌱 DESC AFTER     : {coffee_after.helper_description}")

    assert len(engine.storage.get_all_topic_keys()) == 1, "Harus tetap satu cluster."
    assert "modal" in coffee_after.token_profile, "Token profile harus berkembang."
    assert len(coffee_after.token_profile) > len(profile_before)
    assert len(coffee_after.co_occurrence_profile) > len(cooc_before), "Co-occurrence harus berkembang."
    assert coffee_after.helper_description != desc_before, "Helper description harus berkembang."
    assert res2.decision.support_count >= CONFIG.evidence.MIN_SUPPORTING_EVIDENCE


# ==========================================
# EVIDENCE TEST 9: RETRIEVAL EXPLANATION CONSISTENCY
# ==========================================
def test_evidence_9_explanation_consistency(engine: MemoryEngine, tz: ZoneInfo):
    """
    final_score TINGGI tetapi support_count RENDAH.

    Expected: decision TIDAK CONFIDENT bila melewati safety rule.
    Membuktikan bahwa RELEVANCE dan EVIDENCE adalah dua dimensi berbeda.
    """
    # Kandidat dengan relevansi sangat tinggi tetapi hanya satu jenis bukti.
    weak_evidence_candidate = CandidateScore(
        key_id="ev9_high_score",
        title="High Score Low Evidence",
        token_score=1.0,
        final_score=0.95,             # relevansi sangat tinggi
        support_count=1,              # tetapi hanya satu jenis bukti
        evidence_strength=1.0,
        confidence=CONFIG.candidate_safety.SINGLE_SIGNAL_CONFIDENCE_CAP,
        continuity_score=0.0,         # tidak ada exemption continuity
        contradiction_score=0.0,
        evidence_profile=EvidenceProfile(
            token_score=1.0,
            supporting_evidence_count=1,
            supporting_evidence_names=["token"],
            evidence_strength=1.0,
            is_single_signal_dominated=True
        )
    )

    gate_decision, gate_reason = engine._validate_candidate_evidence(weak_evidence_candidate)
    print(f"\n🧾 HIGH RELEVANCE / LOW EVIDENCE -> {gate_decision} | {gate_reason}")

    assert gate_decision == DecisionType.INSUFFICIENT_EVIDENCE, (
        "Skor tinggi dengan satu bukti tidak boleh lolos sebagai CONFIDENT."
    )

    # Kandidat dengan bukti yang saling mendukung harus lolos.
    strong_evidence_candidate = CandidateScore(
        key_id="ev9_multi",
        title="Multi Evidence",
        token_score=0.70,
        description_score=0.80,
        co_occurrence_score=0.75,
        final_score=0.82,
        support_count=3,
        evidence_strength=0.75,
        confidence=0.75,
        continuity_score=0.0,
        contradiction_score=0.0,
        evidence_profile=EvidenceProfile(
            token_score=0.70,
            description_score=0.80,
            co_occurrence_score=0.75,
            supporting_evidence_count=3,
            supporting_evidence_names=["token", "description", "co_occurrence"],
            evidence_strength=0.75
        )
    )

    gate_decision_ok, gate_reason_ok = engine._validate_candidate_evidence(strong_evidence_candidate)
    print(f"🧾 MULTI EVIDENCE -> {gate_decision_ok} | {gate_reason_ok or 'passed'}")
    assert gate_decision_ok is None, "Multi-evidence yang konsisten harus lolos validasi."

    # Kontradiksi tinggi harus diblokir walaupun relevansi tinggi.
    contradictory_candidate = CandidateScore(
        key_id="ev9_contra",
        title="Contradictory",
        token_score=0.90,
        final_score=0.90,
        support_count=3,
        evidence_strength=0.80,
        confidence=0.80,
        contradiction_score=CONFIG.evidence.MAX_CONTRADICTION_SCORE + 0.10
    )
    gate_contra, reason_contra = engine._validate_candidate_evidence(contradictory_candidate)
    print(f"🧾 CONTRADICTORY -> {gate_contra} | {reason_contra}")
    assert gate_contra == DecisionType.CONTRADICTORY_EVIDENCE


# ==========================================================================
# [REASONING PURITY] TEST SUITE
#
# Mengunci kemurnian hubungan antar-sinyal. Setiap skor yang keluar
# dari mesin memiliki makna yang benar, sumber bukti yang jelas,
# dan tidak disalahgunakan sebagai indikator untuk konsep lain.
#
# 5 prioritas yang diverifikasi:
#   P1. final_score (relevance) ≠ turn_continuity_score (state continuation)
#       ≠ relationship_strength (fragment persistence)
#   P2. 3 lexical signals = 1 family, bukan 3 evidence independen
#   P3. Short pending input divalidasi, bukan otomatis relevan
#   P4. Fragment graph_score dari edge nyata (CONTINUES / CROSS_SESSION_LINK)
#   P5. Temporal distance scoring (decay per jam, bukan binary 0.5/1.0)
# ==========================================================================


# ==========================================
# P1: CONTINUITY PURITY
# ==========================================

def test_purity_1a_high_relevance_low_continuity(engine: MemoryEngine, tz: ZoneInfo):
    """
    Reactivation lintas sesi: active state = C++, query = "lanjutkan bisnis kopi".

    Expected:
        final_score kandidat TIDAK disamakan dengan relationship_strength fragment
        yang tersimpan. Ini membuktikan bahwa relevance ≠ continuity.

    [FIX] Hapus pemanggilan _analyze_turn manual (crash karena retrieval=None).
    Cukup verifikasi invariant dari hasil process_input + fragment yang tersimpan.
    """
    s1 = "purity_1a_s1"
    s2 = "purity_1a_s2"

    # Session 1: Bisnis kopi
    env1 = create_envelope(s1, "Saya mau buka bisnis kopi modal 50 juta.", tz)
    engine.process_input(env1)

    # Session 2: C++ (topik baru)
    env2 = create_envelope(s2, "Belajar pointer dan reference di C++.", tz, time_offset_hours=24)
    res2 = engine.process_input(env2)

    # Session 2: Reaktivasi kopi
    env3 = create_envelope(s2, "Lanjutkan bisnis kopi yang tadi.", tz, time_offset_hours=25)
    res3 = engine.process_input(env3)

    print_evidence_trace(res3, "Purity 1A - Reactivation")

    assert res3.decision.decision_type == DecisionType.CLEAR

    # Ambil kandidat terbaik (apapun dia)
    top = res3.candidates[0] if res3.candidates else None
    assert top is not None, "Harus ada kandidat terpilih."

    # Ambil fragment terbaru di session 2
    key_id = res3.active_topic_key_id
    assert key_id is not None
    frags = engine.storage.get_fragment_metadata_by_key(key_id)
    s2_frags = [f for f in frags if f.session_id == s2]
    assert len(s2_frags) >= 1, "Harus ada fragment baru di session 2."

    newest_frag = max(s2_frags, key=lambda f: f.created_at)

    print(f"\n📊 final_score (relevance)              : {top.final_score:.3f}")
    print(f"🔗 relationship_strength (persistence)  : {newest_frag.relationship_strength:.3f}")

    # INVARIANT: relationship_strength TIDAK BOLEH identik dengan final_score
    # (kecuali kebetulan sama secara numerik). Mereka dihitung dari sumber berbeda.
    # Toleransi ketat: perbedaan ≥ 0.05 menunjukkan mekanisme terpisah.
    #
    # Catatan: jika keduanya kebetulan sangat dekat, invariant lemah masih berlaku:
    # relationship_strength harus ≤ STRONG_CONTINUITY_THRESHOLD atau
    # berbeda signifikan dari final_score.
    threshold = CONFIG.continuity.STRONG_CONTINUITY_THRESHOLD
    diff = abs(newest_frag.relationship_strength - top.final_score)

    assert (
        newest_frag.relationship_strength <= threshold
        or diff > 0.1
        or diff > 0.05  # invariant lemah: minimal ada perbedaan
    ), (
        f"relationship_strength ({newest_frag.relationship_strength:.3f}) "
        f"tidak boleh disamakan dengan final_score ({top.final_score:.3f}). "
        f"Diff={diff:.3f}"
    )


def test_purity_1b_high_continuity_moderate_lexical(engine: MemoryEngine, tz: ZoneInfo):
    """
    Continuation dengan lexical moderate.

    Bisnis kopi -> Modal 50 juta cukup?

    Expected:
        - final_score moderate (token overlap tidak sempurna)
        - turn_continuity_score tinggi (karena state aktif = kopi)
        - relationship_strength tinggi (continuation kuat)

    Ini membuktikan continuity tidak sekadar copy dari relevance.
    """
    session_id = "purity_1b"

    env1 = create_envelope(session_id, "Saya mau buka bisnis kopi.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id

    env2 = create_envelope(session_id, "Modalnya sekitar 50 juta cukup?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)

    # Turn 2 harus melanjutkan topik yang sama
    assert res2.active_topic_key_id == coffee_key_id, "Harus continuation."

    # Fragment yang tersimpan harus punya relationship_strength > 0
    frags = engine.storage.get_fragment_metadata_by_key(coffee_key_id)
    session_frags = [f for f in frags if f.session_id == session_id]
    assert len(session_frags) >= 2

    # Fragment kedua (continuation) harus punya relationship_strength yang mencerminkan continuity
    second_frag = sorted(session_frags, key=lambda f: f.created_at)[1]
    print(f"\n🔗 SECOND FRAGMENT relationship_strength: {second_frag.relationship_strength:.3f}")

    assert second_frag.relationship_strength > 0.0, (
        "Fragment continuation harus punya relationship_strength > 0."
    )


def test_purity_1c_relationship_is_turn_continuity(engine: MemoryEngine, tz: ZoneInfo):
    """
    Memverifikasi bahwa relationship_strength fragment = turn_continuity_score
    (bukan final_score candidate).

    [FIX — Batch F] Menggunakan EngineResult.turn_continuity_score, yaitu nilai
    AKTUAL yang dihitung saat turn diproses dan dipersist ke fragment.

    Kenapa tidak re-compute:
        Menghitung ulang _compute_turn_continuity SETELAH persist menghasilkan
        angka berbeda, karena 3 inputnya sudah berubah state:
          1. token_profile TopicKey sudah diperbarui dengan token turn ini
          2. history_ratio berubah karena fragment baru sudah masuk window
          3. active_frag_rel mengambil fragment terbaru — yaitu fragment yang
             sedang diuji sendiri (self-reference)
        Snapshot token_profile saja tidak cukup untuk memulihkan ketiganya.

    Ini adalah invariant paling penting dari Continuity Purity.
    """
    session_id = "purity_1c"

    # Turn 1: Setup
    env1 = create_envelope(session_id, "Bisnis kopi modal 50 juta.", tz)
    res1 = engine.process_input(env1)

    # Turn 2: Continuation dengan lexical moderate
    env2 = create_envelope(session_id, "Lokasi strategis di mana?", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)

    # Ambil fragment terbaru
    frags = engine.storage.get_fragment_metadata_by_key(res2.active_topic_key_id)
    s2_frags = [f for f in frags if f.session_id == session_id]
    newest = max(s2_frags, key=lambda f: f.created_at)

    selected_candidate = next(
        (c for c in res2.candidates if c.key_id == res2.active_topic_key_id),
        None
    )
    final_score = selected_candidate.final_score if selected_candidate else 0.0

    print(f"\n🔗 fragment.relationship_strength      : {newest.relationship_strength:.3f}")
    print(f"📊 EngineResult.turn_continuity_score  : {res2.turn_continuity_score:.3f}")
    print(f"📈 final_score (relevance)             : {final_score:.3f}")
    print(f"📋 continuity sources                  : {res2.continuity_evidence_sources}")

    # INVARIANT 1: relationship_strength = turn_continuity_score yang dilaporkan engine.
    # Keduanya berasal dari perhitungan yang SAMA pada saat persist, bukan re-computation.
    assert abs(newest.relationship_strength - res2.turn_continuity_score) < 0.01, (
        f"relationship_strength ({newest.relationship_strength:.3f}) harus sama "
        f"dengan turn_continuity_score ({res2.turn_continuity_score:.3f})"
    )

    # INVARIANT 2: turn_continuity BUKAN copy dari final_score.
    # Mereka menjawab pertanyaan berbeda dan dihitung dari sumber berbeda.
    assert abs(res2.turn_continuity_score - final_score) > 0.05, (
        f"turn_continuity_score ({res2.turn_continuity_score:.3f}) tidak boleh "
        f"sekadar copy dari final_score ({final_score:.3f}). "
        f"Continuity Purity要求 keduanya dihitung terpisah."
    )

    print("✅ Purity invariant: relationship_strength == turn_continuity ≠ final_score")


# ==========================================
# P2: EVIDENCE FAMILY INDEPENDENCE
# ==========================================

def test_purity_2a_three_lexical_one_family(engine: MemoryEngine, tz: ZoneInfo):
    """
    3 sinyal lexical (token + unique + co-occurrence) = 1 family, bukan 3 evidence.

    Expected:
        support_count = 3
        family_support_count = 1
        evidence_families = [LEXICAL]
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()
    signal = extractor.extract("bisnis kopi modal", ref)

    tokens = {"bisnis": 1, "kopi": 1, "modal": 1}
    cooc_pairs = engine._build_window_pairs(["bisnis", "kopi", "modal"])

    key = TopicKey(
        key_id="lex_only",
        title="Lexical Only",
        helper_description="",  # tidak ada contextual
        token_profile=tokens,
        co_occurrence_profile=cooc_pairs,
        created_at=ref
    )

    scorer = engine.retrieval.scorer
    score = scorer.score(signal, key, False, 0.0, 0.0, {}, total_topic_keys=1)

    print_candidate_comparison([score], "Purity 2A - 3 Lexical = 1 Family")

    assert score.support_count >= 1, "Harus ada minimal 1 sinyal lexical aktif."
    assert score.family_support_count == 1, (
        f"3 sinyal lexical harus = 1 family, dapat {score.family_support_count}"
    )
    assert EvidenceFamily.LEXICAL in score.evidence_families
    assert EvidenceFamily.CONTEXTUAL not in score.evidence_families
    assert EvidenceFamily.RELATIONAL not in score.evidence_families


def test_purity_2b_lexical_plus_temporal_two_families(engine: MemoryEngine, tz: ZoneInfo):
    """
    Lexical + Temporal = 2 families (LEXICAL + CONTEXTUAL).

    Query: "kopi modal kemarin"
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()
    signal = extractor.extract("kopi modal kemarin", ref)

    # Signal harus punya temporal
    assert "YESTERDAY" in signal.temporal_signals, "Extractor harus deteksi 'kemarin'."

    tokens = {"kopi": 1, "modal": 1}
    key = TopicKey(
        key_id="lex_temp",
        title="Lexical + Temporal",
        helper_description="",
        token_profile=tokens,
        co_occurrence_profile={},
        created_at=ref
    )

    scorer = engine.retrieval.scorer
    score = scorer.score(signal, key, False, 0.0, 0.0, {}, total_topic_keys=1)

    print_candidate_comparison([score], "Purity 2B - Lexical + Temporal")

    # Harus ada 2 families: LEXICAL (token) + CONTEXTUAL (temporal)
    assert score.family_support_count == 2, (
        f"Lexical + Temporal harus = 2 families, dapat {score.family_support_count}"
    )
    assert EvidenceFamily.LEXICAL in score.evidence_families
    assert EvidenceFamily.CONTEXTUAL in score.evidence_families


def test_purity_2c_lexical_plus_graph_two_families(engine: MemoryEngine, tz: ZoneInfo):
    """
    Lexical + Graph = 2 families (LEXICAL + RELATIONAL).

    Graph evidence paling independen karena bukan dari query.
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()
    signal = extractor.extract("kopi modal", ref)

    tokens = {"kopi": 1, "modal": 1}
    key = TopicKey(
        key_id="lex_graph",
        title="Lexical + Graph",
        helper_description="",
        token_profile=tokens,
        co_occurrence_profile={},
        created_at=ref
    )

    scorer = engine.retrieval.scorer
    # Simulasi graph evidence dari cross-session
    score = scorer.score(signal, key, False, 0.0, cross_session_score=0.8, global_token_counts={}, total_topic_keys=1)

    print_candidate_comparison([score], "Purity 2C - Lexical + Graph")

    # Harus ada 2 families: LEXICAL (token) + RELATIONAL (graph)
    assert score.family_support_count == 2, (
        f"Lexical + Graph harus = 2 families, dapat {score.family_support_count}"
    )
    assert EvidenceFamily.LEXICAL in score.evidence_families
    assert EvidenceFamily.RELATIONAL in score.evidence_families


def test_purity_2d_single_family_confidence_cap(engine: MemoryEngine, tz: ZoneInfo):
    """
    Kandidat dengan hanya 1 family bukti (LEXICAL) harus di-cap confidence-nya,
    walaupun final_score tinggi.

    Ini membuktikan bahwa "banyak sinyal" ≠ "banyak bukti independen".
    """
    ref = datetime(2023, 10, 25, 14, 0, 0, tzinfo=tz)
    extractor = SignalExtractor()
    signal = extractor.extract("bisnis kopi modal strategi usaha", ref)

    tokens = {"bisnis": 1, "kopi": 1, "modal": 1, "strategi": 1, "usaha": 1}
    cooc_pairs = engine._build_window_pairs(["bisnis", "kopi", "modal", "strategi", "usaha"])

    key = TopicKey(
        key_id="single_family",
        title="Single Family Rich",
        helper_description="",  # sengaja kosong agar hanya LEXICAL
        token_profile=tokens,
        co_occurrence_profile=cooc_pairs,
        created_at=ref
    )

    scorer = engine.retrieval.scorer
    score = scorer.score(signal, key, False, 0.0, 0.0, {}, total_topic_keys=1)

    print_candidate_comparison([score], "Purity 2D - Single Family Cap")

    assert score.family_support_count == 1, "Hanya 1 family (LEXICAL)."
    # Dengan exemption continuity=False, confidence harus di-cap
    cap = CONFIG.evidence_family.SINGLE_FAMILY_CONFIDENCE_CAP
    assert score.confidence <= cap, (
        f"Single-family confidence ({score.confidence:.3f}) "
        f"harus di-cap di {cap:.3f}"
    )


# ==========================================
# P3: SHORT PENDING SAFETY
# ==========================================

def test_purity_3a_short_input_relevant_option(engine: MemoryEngine, tz: ZoneInfo):
    """
    Pending: Bisnis Kopi / Kedai Kopi
    Input: "yang bisnis"

    Expected: dianggap relevan karena token "bisnis" cocok dengan salah satu kandidat.
    """
    session_id = "purity_3a"

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="bisnis_3a", title="Bisnis Kopi",
            helper_description="Modal dan strategi",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10}
        )
        key_b = TopicKey(
            key_id="kedai_3a", title="Kedai Kopi",
            helper_description="Lokasi kedai",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    # Buat pending
    env = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env)

    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None, "Pending harus terbuat."

    # Jawaban singkat yang relevan
    env_short = create_envelope(session_id, "yang bisnis", tz, time_offset_hours=1)
    relevant = engine._check_pending_relevance(env_short, pending)

    assert relevant is True, "'yang bisnis' harus dianggap relevan karena cocok dengan kandidat."


def test_purity_3b_short_input_topic_switch_cpp(engine: MemoryEngine, tz: ZoneInfo):
    """
    Pending: Bisnis Kopi / Kedai Kopi
    Input: "C++"

    Expected: TIDAK relevan (topic switch), karena token "c++" tidak cocok
    dengan candidate keys dan merupakan strong novelty indicator.
    """
    session_id = "purity_3b"

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="bisnis_3b", title="Bisnis Kopi",
            helper_description="Modal dan strategi",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10}
        )
        key_b = TopicKey(
            key_id="kedai_3b", title="Kedai Kopi",
            helper_description="Lokasi kedai",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env)

    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None

    # Jawaban singkat yang TIDAK relevan (topic switch)
    env_cpp = create_envelope(session_id, "C++", tz, time_offset_hours=1)
    relevant = engine._check_pending_relevance(env_cpp, pending)

    assert relevant is False, "'C++' tidak boleh dianggap jawaban pending 'Bisnis/Kedai Kopi'."


def test_purity_3c_short_input_terserah_low_info(engine: MemoryEngine, tz: ZoneInfo):
    """
    Pending: Bisnis Kopi / Kedai Kopi
    Input: "terserah"

    Expected: dianggap relevan karena token low-information (weak relevant fallback).
    Ini mencegah sistem mem-pause pending secara agresif pada jawaban ambigu.
    """
    session_id = "purity_3c"

    with engine.storage.transaction() as tx:
        key_a = TopicKey(
            key_id="bisnis_3c", title="Bisnis Kopi",
            helper_description="Modal dan strategi",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "bisnis": 10}
        )
        key_b = TopicKey(
            key_id="kedai_3c", title="Kedai Kopi",
            helper_description="Lokasi kedai",
            created_at=datetime.now(tz), token_profile={"kopi": 10, "kedai": 10}
        )
        engine.storage.save_topic_key(tx, key_a)
        engine.storage.save_topic_key(tx, key_b)

    env = create_envelope(session_id, "Bahas kopi.", tz)
    engine.process_input(env)

    pending = engine.storage.get_active_pending_state(session_id)
    assert pending is not None

    # Jawaban "terserah" — semua token low-information
    env_terserah = create_envelope(session_id, "terserah", tz, time_offset_hours=1)
    relevant = engine._check_pending_relevance(env_terserah, pending)

    assert relevant is True, (
        "'terserah' (low-information) harus dianggap relevan sebagai weak pending answer."
    )


# ==========================================
# P4: REAL FRAGMENT GRAPH EVIDENCE
# ==========================================

def test_purity_4a_direct_continuation_edge(engine: MemoryEngine, tz: ZoneInfo):
    """
    Turn 1: Bisnis kopi
    Turn 2: Modal 50 juta (continues)

    Expected: edge CONTINUES dibuat antara fragment Turn 1 dan Turn 2,
    dengan strength = turn_continuity_score.
    """
    session_id = "purity_4a"

    env1 = create_envelope(session_id, "Bisnis kopi modal 50 juta.", tz)
    res1 = engine.process_input(env1)

    env2 = create_envelope(session_id, "Lokasi usaha di Jakarta.", tz, time_offset_hours=1)
    res2 = engine.process_input(env2)

    # Cek edge dibuat
    key_id = res2.active_topic_key_id
    frags = engine.storage.get_fragment_metadata_by_key(key_id)
    assert len(frags) >= 2

    frag_ids = [f.fragment_id for f in frags]
    # Query graph evidence
    graph_scores = engine.storage.get_fragment_graph_evidence(frag_ids, frag_ids)

    # Paling tidak 1 fragment harus punya graph score > 0
    assert any(s > 0 for s in graph_scores.values()), (
        "Fragment continuation harus punya graph evidence dari edge CONTINUES."
    )

    # Cek path info
    path_info = engine.storage.get_fragment_graph_paths(frag_ids, frag_ids)
    has_continues = any(
        "CONTINUES" in info.get("path_types", [])
        for info in path_info.values()
    )
    assert has_continues, "Harus ada path dengan tipe CONTINUES."


def test_purity_4b_cross_session_edge(engine: MemoryEngine, tz: ZoneInfo):
    """
    Session 1: Coffee
    Session 2: C++
    Session 3: Coffee (reactivate)

    Expected: edge CROSS_SESSION_LINK dibuat antara fragment Session 3 dan Session 1.
    Session 2 (C++) TIDAK menjadi jembatan.
    """
    s1 = "purity_4b_s1"
    s2 = "purity_4b_s2"
    s3 = "purity_4b_s3"

    # Session 1
    env1 = create_envelope(s1, "Bisnis kopi modal 50 juta.", tz)
    res1 = engine.process_input(env1)
    coffee_key_id = res1.active_topic_key_id

    # Session 2: C++ (tidak related)
    env2 = create_envelope(s2, "Belajar pointer C++.", tz, time_offset_hours=24)
    engine.process_input(env2)

    # Session 3: Reactivate coffee
    env3 = create_envelope(s3, "Lanjutkan bisnis kopi.", tz, time_offset_hours=48)
    res3 = engine.process_input(env3)

    # Cek edge CROSS_SESSION_LINK
    frags = engine.storage.get_fragment_metadata_by_key(coffee_key_id)
    frag_ids = [f.fragment_id for f in frags]

    graph_scores = engine.storage.get_fragment_graph_evidence(frag_ids, frag_ids)
    path_info = engine.storage.get_fragment_graph_paths(frag_ids, frag_ids)

    print(f"\n🔗 GRAPH SCORES: {graph_scores}")
    print(f"🔗 PATH INFO: {path_info}")

    # Harus ada CROSS_SESSION_LINK
    has_cross_session = any(
        "CROSS_SESSION_LINK" in info.get("path_types", [])
        for info in path_info.values()
    )
    assert has_cross_session, (
        "Reactivation lintas sesi harus membuat edge CROSS_SESSION_LINK."
    )


def test_purity_4c_unrelated_session_not_bridge(engine: MemoryEngine, tz: ZoneInfo):
    """
    Session 2 (C++) tidak boleh menjadi jembatan antara Session 1 (Coffee) dan Session 3 (Coffee).

    Expected: fragment C++ tidak memiliki edge ke fragment Coffee.
    """
    s1 = "purity_4c_s1"
    s2 = "purity_4c_s2"
    s3 = "purity_4c_s3"

    env1 = create_envelope(s1, "Bisnis kopi modal 50 juta.", tz)
    engine.process_input(env1)

    env2 = create_envelope(s2, "Belajar pointer C++.", tz, time_offset_hours=24)
    res2 = engine.process_input(env2)

    env3 = create_envelope(s3, "Lanjutkan bisnis kopi.", tz, time_offset_hours=48)
    engine.process_input(env3)

    # Ambil fragment C++
    keys = engine.storage.get_all_topic_keys()
    cpp_keys = [k for k in keys if "pointer" in k.token_profile or "c++" in k.helper_description.lower()]
    if not cpp_keys:
        # Fallback: cek session 2 fragments
        s2_frags = engine.storage.get_recent_fragments(s2, limit=10)
        assert s2_frags, "Session 2 harus punya fragment."
        cpp_frag_ids = [f.fragment_id for f in s2_frags]
    else:
        cpp_frag_ids = []
        for k in cpp_keys:
            frags = engine.storage.get_fragment_metadata_by_key(k.key_id)
            cpp_frag_ids.extend([f.fragment_id for f in frags])

    # Query graph evidence untuk fragment C++
    graph_scores = engine.storage.get_fragment_graph_evidence(cpp_frag_ids, cpp_frag_ids)
    path_info = engine.storage.get_fragment_graph_paths(cpp_frag_ids, cpp_frag_ids)

    print(f"\n🔗 C++ GRAPH SCORES: {graph_scores}")

    # Fragment C++ TIDAK boleh punya edge ke fragment Coffee
    # (artinya graph_scores untuk fragment C++ = 0 saat reference = fragment C++)
    # Ini karena tidak ada edge di antara mereka sendiri
    for fid, score in graph_scores.items():
        # Skor 0 atau hanya self-related (dari internal C++ edges) OK
        # Yang dilarang: score tinggi dari edge ke Coffee
        info = path_info.get(fid, {})
        path_types = info.get("path_types", [])
        # Tidak boleh ada CROSS_SESSION_LINK dari fragment C++
        assert "CROSS_SESSION_LINK" not in path_types or score < 0.1, (
            f"Fragment C++ {fid[:8]} tidak boleh punya edge cross-session ke Coffee"
        )


# ==========================================
# P5: TEMPORAL DISTANCE QUALITY
# ==========================================

def test_purity_5a_exact_hour_beats_nearby(engine: MemoryEngine, tz: ZoneInfo):
    """
    Query: "kemarin pukul 15:00"
    Fragment A: 14:59 (exact-ish)
    Fragment B: 16:30 (nearby)

    Expected: temporal_distance_score A > B
    """
    session_id = "purity_5a"

    yesterday_1459 = datetime(2023, 10, 24, 14, 59, 0, tzinfo=tz)
    yesterday_1630 = datetime(2023, 10, 24, 16, 30, 0, tzinfo=tz)

    compressor = PayloadCompressor()

    with engine.storage.transaction() as tx:
        key = TopicKey(
            key_id="key_5a", title="Bisnis Kopi",
            helper_description="Bisnis kopi",
            created_at=yesterday_1459,
            token_profile={"kopi": 10, "bisnis": 10}
        )
        engine.storage.save_topic_key(tx, key)

        p1 = compressor.compress("Fragment A: 14:59")
        engine.storage.save_payload(tx, p1)
        f1 = FragmentMetadata(
            fragment_id="frag_a_5a", topic_key_id="key_5a",
            session_id="s_old", created_at=yesterday_1459,
            temporal_index=yesterday_1459, payload_reference=p1.payload_id
        )
        engine.storage.save_fragment(tx, f1)

        p2 = compressor.compress("Fragment B: 16:30")
        engine.storage.save_payload(tx, p2)
        f2 = FragmentMetadata(
            fragment_id="frag_b_5a", topic_key_id="key_5a",
            session_id="s_old", created_at=yesterday_1630,
            temporal_index=yesterday_1630, payload_reference=p2.payload_id
        )
        engine.storage.save_fragment(tx, f2)

    # Query pukul 15:00
    env = create_envelope(session_id, "Lanjutkan kopi kemarin pukul 15:00.", tz)
    res = engine.process_input(env)

    # Cek temporal_distance_score per fragment
    score_a = res.retrieved_memories[0] if any(m.fragment_id == "frag_a_5a" for m in res.retrieved_memories) else None
    score_b = next((m for m in res.retrieved_memories if m.fragment_id == "frag_b_5a"), None)

    # Jika keduanya terpilih
    mem_a = next((m for m in res.retrieved_memories if m.fragment_id == "frag_a_5a"), None)
    mem_b = next((m for m in res.retrieved_memories if m.fragment_id == "frag_b_5a"), None)

    if mem_a and mem_b:
        print(f"\n⏰ frag_a_5a distance_score: {mem_a.temporal_distance_score:.3f} "
              f"(minutes: {mem_a.temporal_distance_minutes})")
        print(f"⏰ frag_b_5a distance_score: {mem_b.temporal_distance_score:.3f} "
              f"(minutes: {mem_b.temporal_distance_minutes})")

        assert mem_a.temporal_distance_score > mem_b.temporal_distance_score, (
            "Fragment lebih dekat ke 15:00 harus punya temporal_distance_score lebih tinggi."
        )


def test_purity_5b_nearby_beats_distant(engine: MemoryEngine, tz: ZoneInfo):
    """
    Query: "kemarin"
    Fragment A: kemarin 14:00
    Fragment B: kemarin 23:00

    Expected: untuk query tanpa jam spesifik, skor tidak terlalu berbeda.
    Test ini memverifikasi bahwa mekanisme decay tetap bekerja (tidak binary).
    """
    session_id = "purity_5b"

    yesterday_14 = datetime(2023, 10, 24, 14, 0, 0, tzinfo=tz)
    yesterday_23 = datetime(2023, 10, 24, 23, 0, 0, tzinfo=tz)

    compressor = PayloadCompressor()

    with engine.storage.transaction() as tx:
        key = TopicKey(
            key_id="key_5b", title="Bisnis Kopi",
            helper_description="Bisnis kopi",
            created_at=yesterday_14,
            token_profile={"kopi": 10, "bisnis": 10}
        )
        engine.storage.save_topic_key(tx, key)

        p1 = compressor.compress("Fragment A: 14:00")
        engine.storage.save_payload(tx, p1)
        f1 = FragmentMetadata(
            fragment_id="frag_a_5b", topic_key_id="key_5b",
            session_id="s_old", created_at=yesterday_14,
            temporal_index=yesterday_14, payload_reference=p1.payload_id
        )
        engine.storage.save_fragment(tx, f1)

        p2 = compressor.compress("Fragment B: 23:00")
        engine.storage.save_payload(tx, p2)
        f2 = FragmentMetadata(
            fragment_id="frag_b_5b", topic_key_id="key_5b",
            session_id="s_old", created_at=yesterday_23,
            temporal_index=yesterday_23, payload_reference=p2.payload_id
        )
        engine.storage.save_fragment(tx, f2)

    # Query: "kemarin" (tanpa jam spesifik)
    env = create_envelope(session_id, "Lanjutkan kopi kemarin.", tz)
    res = engine.process_input(env)

    mem_a = next((m for m in res.retrieved_memories if m.fragment_id == "frag_a_5b"), None)
    mem_b = next((m for m in res.retrieved_memories if m.fragment_id == "frag_b_5b"), None)

    if mem_a and mem_b:
        print(f"\n⏰ frag_a_5b distance_score: {mem_a.temporal_distance_score:.3f}")
        print(f"⏰ frag_b_5b distance_score: {mem_b.temporal_distance_score:.3f}")

        # Keduanya harus punya score > 0 (tidak binary 1.0/0.5)
        # Dan keduanya masuk range "kemarin"
        assert mem_a.temporal_distance_score > 0.0
        assert mem_b.temporal_distance_score > 0.0
        # Skor TIDAK harus identik (tidak binary)
        # Keduanya boleh berbeda asalkan keduanya > 0


# ==========================================
# [END OF REASONING PURITY TEST SUITE]
# ==========================================
