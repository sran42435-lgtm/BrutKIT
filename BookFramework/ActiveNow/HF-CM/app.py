"""
app.py
Application Entry Point, Session Manager & REPL (Layer 16 Interface).

Fungsi:
1. Menyediakan antarmuka CLI interaktif untuk menguji mesin HF-CM.
2. Mengelola siklus hidup Sesi (Session Lifecycle & Cross-Session).
3. Merakit InputEnvelope dan memanggil Global Orchestrator (MemoryEngine).
4. Menampilkan "Engine State" (Topik Aktif, Ambiguitas, Injeksi Memori) 
   untuk keperluan debugging dan pembuktian konsep PoC.
"""
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config import CONFIG
from models import InputEnvelope, EngineResult, DecisionType
from storage import MemoryStorage
from retrieval import RetrievalEngine
from compression import PayloadCompressor
from memory_engine import MemoryEngine
# [NEW] Import untuk Composition Root (Spec 11)
from context_builder import ContextBuilder
from llm_client import FakeLLMClient

# Setup logging untuk melihat apa yang terjadi di belakang layar (bisa disesuaikan ke WARNING jika terlalu berisik)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("app")

class HF_CMApp:
    """
    Aplikasi CLI untuk berinteraksi dengan Hybrid Fragmented Compressed Memory.
    Bertindak sebagai User Proxy yang menjembatani input mentah ke mesin memori.
    """
    def __init__(self):
        # 1. Inisialisasi komponen mesin (Dependency Injection)
        self.storage = MemoryStorage()
        self.retrieval = RetrievalEngine()
        self.compressor = PayloadCompressor()
        
        # [NEW] Inisialisasi Context Builder & LLM Client (Spec 11: Composition Root)
        self.context_builder = ContextBuilder()
        self.llm_client = FakeLLMClient() # Menggunakan Fake untuk PoC CLI
        
        # 2. Rakit Orchestrator Utama
        self.engine = MemoryEngine(
            storage=self.storage, 
            retrieval=self.retrieval, 
            compressor=self.compressor,
            context_builder=self.context_builder,
            llm_client=self.llm_client
        )
        
        # 3. State Aplikasi (Session Management)
        self.current_session_id = str(uuid.uuid4())[:8] # ID pendek agar mudah dibaca
        self.timezone = ZoneInfo(CONFIG.input.DEFAULT_TIMEZONE)
        
        self._print_welcome()

    def _print_welcome(self):
        print("="*65)
        print("  HF-CM PoC: Hybrid Fragmented Compressed Memory")
        print("  Mesin Memori Eksternal Cerdas (Antara User & Model API)")
        print("="*65)
        print(f"📌 Sesi saat ini: {self.current_session_id}")
        print("💡 Command khusus:")
        print("   /new       -> Mulai sesi baru (Uji Cross-Session)")
        print("   /switch ID -> Pindah ke sesi tertentu")
        print("   /state     -> Lihat snapshot Fast Memory & Topic Keys")
        print("   /quit      -> Keluar dari aplikasi")
        print("-"*65)

    def run(self):
        """Read-Eval-Print Loop (REPL) utama."""
        while True:
            try:
                user_input = input(f"\n[Session {self.current_session_id}] You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle Commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                    
                # Proses Input melalui Mesin Memori
                self._process_turn(user_input)
                
            except KeyboardInterrupt:
                print("\n\nSesi diakhiri oleh user. (Data tersimpan aman di SQLite).")
                break
            except Exception as e:
                logger.error(f"Terjadi error tak terduga: {e}", exc_info=True)
                print(f"[SYSTEM ERROR] {e}")

    def _handle_command(self, command: str):
        """Menangani command khusus CLI untuk manajemen sesi dan debugging."""
        cmd = command.lower().split()
        
        if cmd[0] in ["/quit", "/exit"]:
            print("Menutup aplikasi...")
            exit(0)
            
        elif cmd[0] == "/new":
            self.current_session_id = str(uuid.uuid4())[:8]
            print(f"\n--- 🔄 MEMULAI SESI BARU: {self.current_session_id} ---")
            print("(Context window Fast State dikosongkan, tetapi Big Memory tetap ada di database).")
            
        elif cmd[0] == "/switch" and len(cmd) > 1:
            self.current_session_id = cmd[1]
            print(f"\n--- 🔀 BERPINDAH KE SESI: {self.current_session_id} ---")
            
        elif cmd[0] == "/state":
            self._print_system_state()
            
        else:
            print("Command tidak dikenal. Tersedia: /new, /switch <id>, /state, /quit")

    def _process_turn(self, raw_text: str):
        """Memproses satu turn percakapan melalui Layer 1 hingga 16."""
        now = datetime.now(self.timezone)
        
        # 1. Bungkus dalam InputEnvelope (Layer 1: Input Gate)
        envelope = InputEnvelope(
            input_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            raw_text=raw_text,
            normalized_text=raw_text.lower(), # Simplifikasi normalisasi untuk PoC
            timestamp=now,
            timezone=str(self.timezone)
        )
        
        # 2. Panggil Orchestrator (Layer 16: Global Orchestrator)
        result: EngineResult = self.engine.process_input(envelope)
        
        # 3. Tampilkan Output ke User
        self._render_output(result)

    def _render_output(self, result: EngineResult):
        """
        Menampilkan respons AI dan metadata mesin.
        Ini adalah fitur pembuktian PoC: User bisa melihat keputusan mesin.
        """
        print("-" * 50)
        
        decision = result.decision
        
        # Tampilkan Status Mesin (Transparansi Arsitektur)
        if decision.decision_type == DecisionType.AMBIGUOUS:
            print("[SYSTEM] ⚠️  AMBIGUITAS TERDETEKSI (Ambiguity Gate Aktif)")
            print("[SYSTEM] Mesin menolak menebak. Memerlukan klarifikasi.")
            if decision.pending_state:
                print(f"[SYSTEM] Kandidat Terdeteksi: {decision.pending_state.candidate_keys}")
                
        elif decision.decision_type == DecisionType.CLEAR:
            topic_id = result.active_topic_key_id or 'Topik Baru'
            print(f"[SYSTEM] ✅ Topik Aktif: {topic_id}")
            if result.retrieved_memory_ids:
                print(f"[SYSTEM] 📂 Exact Memory Diinjeksikan: {len(result.retrieved_memory_ids)} fragment")
            else:
                print(f"[SYSTEM] 📂 Tidak ada memori persis yang cocok (Mungkin topik baru).")
                
        elif decision.decision_type == DecisionType.NO_MATCH:
             print(f"[SYSTEM] ❌ NO_MATCH: Sinyal terlalu lemah atau noise.")
             
        # Tampilkan Respons AI (Dari LLM / Simulasi)
        print(f"\nAI: {result.assistant_response}")
        print("-" * 50)

    def _print_system_state(self):
        """Menampilkan snapshot state Fast Memory & Big Memory untuk debugging."""
        print("\n" + "="*40)
        print("📊 SYSTEM STATE SNAPSHOT")
        print("="*40)
        
        # Cek Fast State (Pending Ambiguity)
        pending = self.storage.get_active_pending_state(self.current_session_id)
        if pending:
            print(f"⏳ Pending State: {pending.status.value}")
            print(f"   Pertanyaan Menggantung: '{pending.assistant_question}'")
            print(f"   Kandidat: {pending.candidate_keys}")
        else:
            print("⏳ Pending State: Kosong (Normal)")
            
        # Cek Big Memory (Topic Keys)
        keys = self.storage.get_all_topic_keys()
        print(f"\n🗄️  Total Topic Keys di Database: {len(keys)}")
        if not keys:
            print("   (Database memori masih kosong)")
        for k in keys:
            print(f"   🔑 {k.title} [ID: {k.key_id[:8]}...]")
            
        print("="*40)

if __name__ == "__main__":
    app = HF_CMApp()
    app.run()
