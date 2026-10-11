#!/usr/bin/env python3
"""
Script untuk inspeksi checkpoint dan state_dict model.
Menjawab pertanyaan: Apakah LM heads MoE tersimpan dengan benar?
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from config import CONFIG
from core.architecture import CustomTransformerLM
from pipeline.weight_manager import WeightManager

print("=" * 70)
print("DIAGNOSA CHECKPOINT")
print("=" * 70)

# Load checkpoint
ckpt_path = CONFIG.paths.checkpoint_path
print(f"\n1. Cek checkpoint path: {ckpt_path}")
print(f"   Exists: {ckpt_path.exists()}")

if not ckpt_path.exists():
    print("   ❌ Checkpoint tidak ditemukan!")
    sys.exit(1)

# Load data dari checkpoint
print(f"\n2. Load checkpoint data...")
ckpt = np.load(ckpt_path, allow_pickle=True)
print(f"   Keys in checkpoint: {list(ckpt.files)}")

# Cek state_dict
if "state_dict" in ckpt.files:
    state_dict_raw = ckpt["state_dict"].item() if ckpt["state_dict"].ndim == 0 else ckpt["state_dict"]
    
    if isinstance(state_dict_raw, dict):
        print(f"\n3. Tensors in state_dict: {len(state_dict_raw)}")
        
        # Pisahkan ke kategori
        lm_heads = []
        backbone = []
        semantic = []
        other = []
        
        for name in sorted(state_dict_raw.keys()):
            if name.startswith("lm_heads."):
                lm_heads.append(name)
            elif name.startswith("__semantic"):
                semantic.append(name)
            elif name.startswith(("embed_tokens", "layers", "norm", "pos_encoding")):
                backbone.append(name)
            else:
                other.append(name)
        
        print(f"\n   📦 Backbone tensors: {len(backbone)}")
        for name in backbone[:5]:
            print(f"      - {name}")
        if len(backbone) > 5:
            print(f"      ... dan {len(backbone) - 5} lainnya")
        
        print(f"\n   🎯 LM Heads: {len(lm_heads)}")
        for name in lm_heads:
            arr = state_dict_raw[name]
            print(f"      - {name}: shape {arr.shape}")
        
        print(f"\n   🧠 Semantic tensors: {len(semantic)}")
        for name in semantic:
            print(f"      - {name}")
        
        print(f"\n   ❓ Other tensors: {len(other)}")
        for name in other:
            print(f"      - {name}")
    else:
        print(f"   ⚠️ state_dict bukan dict: type={type(state_dict_raw)}")
else:
    print("\n   ❌ 'state_dict' TIDAK ADA di checkpoint!")
    print(f"   Yang ada: {list(ckpt.files)}")

# Cek model initialization
print("\n" + "=" * 70)
print("LOAD MODEL + STATE_DICT")
print("=" * 70)

model = CustomTransformerLM(CONFIG)
print(f"\n1. Model domains SEBELUM load: {model.list_domains()}")
print(f"   LM heads count: {len(model.lm_heads)}")

weight_manager = WeightManager(model, CONFIG)

# Coba load checkpoint
try:
    result = weight_manager.load_checkpoint()
    print(f"\n2. Load checkpoint result: {result}")
except Exception as e:
    print(f"\n2. Load checkpoint ERROR: {e}")

print(f"\n3. Model domains SETELAH load: {model.list_domains()}")
print(f"   LM heads count: {len(model.lm_heads)}")

# Cek state_dict model saat ini
current_state = model.state_dict()
lm_head_keys = [k for k in current_state.keys() if k.startswith("lm_heads.")]
print(f"\n4. LM heads di state_dict saat ini: {len(lm_head_keys)}")
for k in lm_head_keys:
    print(f"   - {k}: shape {current_state[k].shape}")

print("\n" + "=" * 70)
print("KESIMPULAN")
print("=" * 70)

if len(lm_heads) == 0:
    print("❌ LM heads TIDAK TERSIMPAN di checkpoint!")
    print("   Solusi: Fix WeightManager.save_checkpoint() untuk include LM heads")
elif len(lm_heads) < 8:
    print(f"⚠️  Hanya {len(lm_heads)} LM heads tersimpan (expected: 8)")
    print("   Kemungkinan: init_domains() dipanggil SETELAH save_checkpoint()")
else:
    if len(model.lm_heads) < 8:
        print(f"⚠️  LM heads ada di checkpoint ({len(lm_heads)}) tapi tidak ke-load ({len(model.lm_heads)})")
        print("   Solusi: Fix WeightManager.load_checkpoint() untuk load semua LM heads")
    else:
        print(f"✅ LM heads lengkap: {len(model.lm_heads)} domain")
