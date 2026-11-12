"""
context_builder.py
Context Assembly & Budgeting (Layer 8).

Fungsi:
1. Menyusun Working Context yang aman dan terstruktur untuk Model API.
2. Menggabungkan System Prompt + Exact Retrieved Fragments + Current Prompt.
3. Menjalankan Token Budget Enforcer (mencegah context window explosion).
4. Memformat memori masa lalu dengan tag yang jelas (XML-style) agar Model 
   tidak mengalami halusinasi antara "ingatan" dan "instruksi baru".

PRINSIP:
- Strict Budgeting: Memori tidak boleh melanggar batas token.
- Semantic Isolation: Memori lama dan prompt baru harus terisolasi secara visual/struktural.
- Fail-Open: Jika tidak ada memori, percakapan tetap berjalan dengan prompt saat ini.
"""

import logging
from typing import List, Dict, Optional

from config import CONFIG
from models import ModelRequest

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Mesin perakit konteks. Menerjemahkan data memori persis (Exact Memory) 
    dan input user menjadi satu kesatuan ModelRequest yang siap dikonsumsi LLM.
    """

    def __init__(self):
        self.max_context_tokens = CONFIG.context.MAX_CONTEXT_TOKENS
        self.max_memory_tokens = CONFIG.context.MAX_MEMORY_TOKENS
        self.max_fragments = CONFIG.context.MAX_FRAGMENTS_IN_CONTEXT
        
        # Heuristik sederhana untuk estimasi token PoC 
        # (Rata-rata 1 token = 4 karakter atau 0.75 kata)
        self.CHARS_PER_TOKEN = 4 

    def build_context(
        self,
        current_user_input: str,
        exact_memories: List[Dict],
        active_topic_title: Optional[str] = None,
        system_instructions: Optional[str] = None
    ) -> ModelRequest:
        """
        Merakit konteks final untuk Model API.
        """
        
        # 1. Siapkan System Context
        sys_prompt = system_instructions or self._get_default_system_prompt()
        
        # [OVERHAUL] Hitung token yang sudah terpakai oleh System dan User Input (Review Kedua)
        sys_tokens = self._estimate_tokens(sys_prompt)
        user_tokens = self._estimate_tokens(current_user_input)
        available_memory_tokens = max(0, self.max_context_tokens - sys_tokens - user_tokens)
        
        # Batasi juga dengan MAX_MEMORY_TOKENS absolut
        effective_memory_budget = min(self.max_memory_tokens, available_memory_tokens)
        
        # 2. Rakit dan Batasi Memori (Memory Budgeting)
        past_memory_text = self._assemble_and_budget_memories(exact_memories, active_topic_title, effective_memory_budget)
        
        # 4. Bungkus dalam ModelRequest
        return ModelRequest(
            system_context=sys_prompt,
            past_memory=past_memory_text,
            current_user_message=current_user_input
        )

    def _assemble_and_budget_memories(
        self, 
        memories: List[Dict], 
        topic_title: Optional[str],
        memory_budget: int
    ) -> str:
        """
        Menggabungkan fragmen memori dengan batas token dan jumlah fragmen.
        Menggunakan XML Tags untuk isolasi semantik.
        """
        if not memories:
            return ""

        # [OVERHAUL] Prioritas Fragment: Relevance Score -> Recency (Review Kedua)
        def sort_key(mem):
            # Tuple sorting: relevance_score (desc), then timestamp (desc)
            return (mem.get('relevance_score', 0.0), mem.get('timestamp', ''))
            
        memories = sorted(memories, key=sort_key, reverse=True)

        # Batasi jumlah fragmen berdasarkan config
        memories = memories[:self.max_fragments]
        
        assembled_parts = []
        current_memory_tokens = 0
        
        # Iterasi dari memori yang paling relevan
        for mem in memories:
            text = mem.get('text', '')
            timestamp = mem.get('timestamp', 'Unknown')
            
            formatted_mem = f"[Waktu: {timestamp}]\n{text}"
            mem_tokens = self._estimate_tokens(formatted_mem)
            
            # [OVERHAUL] Degradasi Budget: Hentikan jika budget habis, jangan buang semua
            if current_memory_tokens + mem_tokens > memory_budget:
                # Jika fragment pertama saja sudah melebihi budget, kita tidak bisa memasukkannya
                # (Exact memory tidak boleh dipotong sembarangan / string slicing)
                if not assembled_parts:
                    logger.warning("Fragment tunggal melebihi sisa budget memori. Dilewati.")
                    break
                logger.info("Memory Token Budget tercapai. Fragmen sisa diabaikan.")
                break
                
            assembled_parts.append(formatted_mem)
            current_memory_tokens += mem_tokens

        if not assembled_parts:
            return ""

        # Bungkus dengan XML Tag agar LLM paham ini adalah data historis
        topic_context = f' topic="{topic_title}"' if topic_title else ""
        joined_memories = "\n\n---\n\n".join(assembled_parts)
        
        return (
            f"<past_memories{topic_context}>\n"
            f"{joined_memories}\n"
            f"</past_memories>"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimasi jumlah token berdasarkan panjang karakter."""
        if not text:
            return 0
        return len(text) // self.CHARS_PER_TOKEN + 1

    def _get_default_system_prompt(self) -> str:
        """System prompt default untuk menjaga perilaku Model API."""
        return (
            "Anda adalah asisten AI yang cerdas dan akurat. "
            "Anda memiliki akses ke <past_memories> yang berisi catatan percakapan masa lalu yang PERSIS (bukan tebakan). "
            "Gunakan <past_memories> HANYA jika relevan dengan <current_request>. "
            "Jika <past_memories> tidak menyediakan informasi yang diminta, katakan dengan jujur bahwa Anda tidak memiliki catatan tersebut, "
            "jangan pernah mengarang atau berhalusinasi tentang detail masa lalu. "
            "Jawablah <current_request> secara langsung dan natural."
        )
