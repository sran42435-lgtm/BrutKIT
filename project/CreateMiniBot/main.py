#!/usr/bin/env python3
# main.py
#
# Entry point utama dan orkestrator pipeline untuk RAG Architecture + SOTA Training.
#
# Arsitektur: Single-Head + BM25 Retrieval + Cross-Attention + Constrained Generation
# - Context-conditioned training
# - Loss ↓ = Presisi ↑ (korelasi langsung, no topic mixing)
# - Retrieval-based context injection saat inference
#
# 🎯 SOTA Training Techniques:
# - Cosine Annealing LR + Warmup
# - Label Smoothing
# - Focal Loss
# - Token Importance Weighting
# - Adaptive Gradient Clipping (AGC)
# - Loss Spike Detection + Auto-Recovery
# - Curriculum Learning (easy → hard)
# - Self-Correction Loop
# - EMA Weights (Polyak Averaging)
#
# Mode operasi:
# 1. MODE PIPELINE (default):
#    Tokenizer → Load/Convert Dataset → SOTA Training → Eval → Test → Export
#
# 2. MODE INTERACTIVE (--interactive):
#    Tokenizer → Load Model → Playground (dengan RAG inference)
#    Training dilakukan via perintah 'training' di playground.
#
# Flag penting:
# --resume           : muat checkpoint dan lanjutkan dari epoch terakhir
# --interactive      : langsung masuk playground, training via perintah di dalam
# --skip-training    : lewati training (mode pipeline)
# --skip-eval        : lewati evaluasi
# --skip-test        : lewati test prompt
# --skip-export      : lewati export model
# --skip-convert     : lewati konversi dataset (jika dataset sudah ada)
# --force-convert    : paksa rebuild dataset meskipun sudah ada
# --debug-tokens     : tampilkan Token IDs dan Token Pieces saat inference
#
# Keterhubungan:
# - config.py              : konfigurasi global + SOTA parameters
# - core/tokenizer.py      : tokenizer BPE dengan RAG templates
# - core/architecture.py   : model transformer + cross-attention + constrained gen
# - core/retriever.py      : BM25 retrieval engine
# - core/model_exporter.py : export ke safetensors + RAG assets
# - pipeline/trainer.py    : SOTA context-conditioned training
# - pipeline/evaluator.py  : SOTA loss computation (focal, label smoothing, etc.)
# - pipeline/weight_manager.py : load/save checkpoint + EMA support
# - scripts/prepare_datasets.py : prepare SQuAD + custom datasets
# - scripts/convert_dataset.py  : convert custom dataset (fallback)
# - testing/playground.py  : RAG inference testing

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from config import (
    CONFIG,
    Config,
    ModelConfig,
    PathsConfig,
    RetrievalConfig,
    SpecialTokensConfig,
    TrainingConfig,
)

from core.tokenizer import Tokenizer
from core.architecture import CustomTransformerLM
from core.model_exporter import ModelExporter

from pipeline.evaluator import Evaluator
from pipeline.weight_manager import WeightManager
from pipeline.trainer import Trainer

from testing.playground import Playground


# ============================================================================
# PRETTY PRINT HELPERS
# ============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_config_summary(config: Config) -> None:
    """Print summary dari config, termasuk SOTA parameters."""
    print_section("KONFIGURASI PIPELINE (RAG + SOTA)")

    print(f"Datasets dir         : {config.paths.datasets_dir}")
    print(f"Output models dir    : {config.paths.output_models_dir}")
    print(f"Vocab path           : {config.paths.vocab_path}")
    print(f"Final model path     : {config.paths.final_model_path}")
    print(f"Checkpoint path      : {config.paths.checkpoint_path}")
    print(f"QA dataset path      : {config.paths.qa_dataset_path}")
    print(f"Merged train path    : {config.paths.merged_qa_train_path}")
    print(f"Retrieval index path : {config.paths.retrieval_index_path}")

    print()
    print("Model:")
    print(f"  vocab_size              : {config.model.vocab_size:,}")
    print(f"  embedding_dim           : {config.model.embedding_dim}")
    print(f"  num_attention_heads     : {config.model.num_attention_heads}")
    print(f"  num_layers              : {config.model.num_layers}")
    print(f"  ffn_hidden_dim          : {config.model.ffn_hidden_dim}")
    print(f"  max_position_embeddings : {config.model.max_position_embeddings}")
    print(f"  train_dtype             : {config.model.train_dtype}")
    print(f"  export_dtype            : {config.model.export_dtype}")
    print(f"  use_rope                : {config.model.use_rope}")
    print(f"  use_weight_tying        : {config.model.use_weight_tying}")

    print()
    print("Training:")
    print(f"  learning_rate                : {config.training.learning_rate}")
    print(f"  batch_size                   : {config.training.batch_size}")
    print(f"  epochs                       : {config.training.epochs}")
    print(f"  sequence_length              : {config.training.sequence_length}")
    print(f"  dropout_rate                 : {config.training.dropout_rate}")
    print(f"  weight_decay                 : {config.training.weight_decay}")
    print(f"  grad_clip_norm               : {config.training.grad_clip_norm}")
    print(f"  lr_warmup_steps              : {config.training.lr_warmup_steps}")
    print(f"  early_stopping_patience      : {config.training.early_stopping_patience}")

    print()
    print("🎯 SOTA Training Techniques:")
    use_cosine = getattr(config.training, 'use_cosine_schedule', True)
    use_focal = getattr(config.training, 'use_focal_loss', True)
    use_agc = getattr(config.training, 'use_agc', True)
    use_curriculum = getattr(config.training, 'use_curriculum', True)
    use_self_corr = getattr(config.training, 'use_self_correction', True)
    use_ema = getattr(config.training, 'use_ema', True)
    use_token_weight = getattr(config.training, 'use_token_weighting', True)
    use_spike_det = getattr(config.training, 'use_loss_spike_detection', True)
    
    label_smooth = getattr(config.training, 'label_smoothing', 0.1)
    focal_gamma = getattr(config.training, 'focal_gamma', 2.0)
    answer_weight = getattr(config.training, 'answer_token_weight', 3.0)
    ema_decay = getattr(config.training, 'ema_decay', 0.999)
    
    print(f"  Cosine LR + Warmup       : {use_cosine}")
    print(f"  Label Smoothing (ε={label_smooth})  : {label_smooth > 0}")
    print(f"  Focal Loss (γ={focal_gamma})        : {use_focal}")
    print(f"  Token Weighting (×{answer_weight})  : {use_token_weight}")
    print(f"  AGC (Adaptive Clip)      : {use_agc}")
    print(f"  Loss Spike Detection     : {use_spike_det}")
    print(f"  Curriculum Learning      : {use_curriculum}")
    print(f"  Self-Correction Loop     : {use_self_corr}")
    print(f"  EMA Weights (decay={ema_decay}) : {use_ema}")

    print()
    print("RAG Training:")
    print(f"  context_dropout_rate         : {config.training.context_dropout_rate:.1%}")
    print(f"  context_perturbation_rate    : {config.training.context_perturbation_rate:.1%}")
    print(f"  progressive_stages           : {config.training.progressive_stages}")
    print(f"  max_context_paragraphs       : {config.training.max_context_paragraphs}")
    print(f"  augmentation_factor          : {config.training.augmentation_factor}")

    print()
    print("Retrieval (BM25):")
    print(f"  top_k                        : {config.retrieval.top_k}")
    print(f"  bm25_k1                      : {config.retrieval.bm25_k1}")
    print(f"  bm25_b                       : {config.retrieval.bm25_b}")
    print(f"  dynamic_threshold_ratio      : {config.retrieval.dynamic_threshold_ratio}")
    print(f"  query_expansion_count        : {config.retrieval.query_expansion_count}")
    print(f"  max_paragraph_length         : {config.retrieval.max_paragraph_length}")


# ============================================================================
# DATASET LOADING HELPERS
# ============================================================================

def load_dataset_info(config: Config) -> dict:
    """
    Detect dataset yang tersedia dan return info.
    
    Priority:
    1. Merged SQuAD dataset (dari prepare_datasets.py)
    2. Custom QA dataset (dari convert_dataset.py)
    3. Raw .txt files (perlu convert)
    """
    info = {
        "has_merged": False,
        "has_custom_qa": False,
        "has_raw_txt": False,
        "merged_samples": 0,
        "custom_samples": 0,
        "raw_files": 0,
        "best_source": None,
    }
    
    # Check merged dataset (SQuAD)
    merged_path = Path(config.paths.merged_qa_train_path)
    if merged_path.exists():
        try:
            with open(merged_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["has_merged"] = True
            info["merged_samples"] = len(data.get("samples", []))
        except Exception:
            pass
    
    # Check custom QA dataset
    qa_path = Path(config.paths.qa_dataset_path)
    if qa_path.exists():
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["has_custom_qa"] = True
            info["custom_samples"] = len(data.get("samples", []))
        except Exception:
            pass
    
    # Check raw .txt files
    datasets_dir = Path(config.paths.datasets_dir)
    if datasets_dir.exists():
        skip_dirs = {"squad", "tydi", "indoqa", "merged"}
        txt_files = [
            f for f in datasets_dir.glob("*.txt")
            if f.parent.name not in skip_dirs
        ]
        info["has_raw_txt"] = len(txt_files) > 0
        info["raw_files"] = len(txt_files)
    
    # Determine best source
    if info["has_merged"] and info["merged_samples"] > 1000:
        info["best_source"] = "merged"
    elif info["has_custom_qa"] and info["custom_samples"] > 0:
        info["best_source"] = "custom"
    elif info["has_raw_txt"]:
        info["best_source"] = "raw"
    else:
        info["best_source"] = None
    
    return info


def print_dataset_info(info: dict) -> None:
    """Print dataset availability info."""
    print()
    print("📚 Dataset Status:")
    
    if info["has_merged"]:
        print(f"  ✅ Merged SQuAD      : {info['merged_samples']:,} samples")
    else:
        print(f"  ❌ Merged SQuAD      : not found")
    
    if info["has_custom_qa"]:
        print(f"  ✅ Custom QA         : {info['custom_samples']:,} samples")
    else:
        print(f"  ⚠️  Custom QA         : not found")
    
    if info["has_raw_txt"]:
        print(f"  📄 Raw .txt files    : {info['raw_files']} files")
    else:
        print(f"  ❌ Raw .txt files    : none")
    
    print(f"\n  🎯 Best source       : {info['best_source'] or 'NONE'}")


def prepare_dataset(
    config: Config,
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Prepare dataset: gunakan merged jika ada, atau convert raw files.
    
    Returns:
        True jika dataset siap, False jika gagal
    """
    info = load_dataset_info(config)
    
    # Jika merged dataset ada dan tidak force convert, pakai itu
    if info["has_merged"] and not force:
        if verbose:
            print(f"[main] ✅ Using merged SQuAD dataset: {info['merged_samples']:,} samples")
            print(f"[main]    Source: {config.paths.merged_qa_train_path}")
        
        # Copy merged train ke qa_dataset_path untuk kompatibilitas trainer
        merged_path = Path(config.paths.merged_qa_train_path)
        qa_path = Path(config.paths.qa_dataset_path)
        
        if merged_path != qa_path:
            import shutil
            try:
                shutil.copy2(merged_path, qa_path)
                if verbose:
                    print(f"[main]    Copied to: {qa_path}")
            except Exception as exc:
                if verbose:
                    print(f"[main] ⚠️ Failed to copy: {exc}")
        
        return True
    
    # Jika custom QA ada dan tidak force, pakai itu
    if info["has_custom_qa"] and not force:
        if verbose:
            print(f"[main] ✅ Using custom QA dataset: {info['custom_samples']:,} samples")
        return True
    
    # Coba jalankan prepare_datasets.py (untuk SQuAD)
    prepare_script = Path(__file__).resolve().parent / "scripts" / "prepare_datasets.py"
    if prepare_script.exists() and info["has_merged"] is False:
        # Check if raw SQuAD files exist
        squad_dir = Path(config.paths.datasets_dir) / "squad"
        if squad_dir.exists() and any(squad_dir.glob("*.json")):
            if verbose:
                print(f"[main] 📥 Running prepare_datasets.py untuk merge SQuAD...")
            
            try:
                result = subprocess.run(
                    [sys.executable, str(prepare_script)],
                    capture_output=not verbose,
                    text=True,
                    check=True,
                )
                if verbose and result.stdout:
                    print(result.stdout)
                
                # Reload info
                info = load_dataset_info(config)
                if info["has_merged"]:
                    if verbose:
                        print(f"[main] ✅ SQuAD prepared: {info['merged_samples']:,} samples")
                    return prepare_dataset(config, force=False, verbose=verbose)
            except Exception as exc:
                if verbose:
                    print(f"[main] ⚠️ prepare_datasets.py failed: {exc}")
    
    # Fallback: convert_dataset.py (untuk raw .txt files)
    if info["has_raw_txt"]:
        return convert_dataset(config, verbose=verbose)
    
    if verbose:
        print("[main] ❌ No dataset available.")
        print("[main]    Options:")
        print("[main]    1. Run: python scripts/download_datasets.py")
        print("[main]    2. Run: python scripts/prepare_datasets.py")
        print("[main]    3. Add .txt files ke datasets/ folder")
    
    return False


def convert_dataset(
    config: Config,
    augmentation_factor: Optional[int] = None,
    preview: int = 0,
    verbose: bool = True,
) -> bool:
    """
    Jalankan script konversi dataset mentah ke format QA.

    Memanggil scripts/convert_dataset.py sebagai subprocess.
    """
    script_path = Path(__file__).resolve().parent / "scripts" / "convert_dataset.py"

    if not script_path.exists():
        if verbose:
            print(f"[main] ⚠️ Script konversi tidak ditemukan: {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--datasets-dir", str(config.paths.datasets_dir),
        "--output", str(config.paths.qa_dataset_path),
    ]

    if augmentation_factor is not None:
        cmd.extend(["--augmentation-factor", str(augmentation_factor)])
    else:
        cmd.extend(["--augmentation-factor", str(config.training.augmentation_factor)])

    if preview > 0:
        cmd.extend(["--preview", str(preview)])

    if verbose:
        print(f"[main] Menjalankan konversi dataset...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )
        
        if verbose and result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        qa_path = Path(config.paths.qa_dataset_path)
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            n_samples = len(data.get("samples", []))
            if verbose:
                print(f"[main] ✅ Konversi sukses: {n_samples:,} QA samples")
            return True
        else:
            if verbose:
                print(f"[main] ❌ File output tidak dibuat: {qa_path}")
            return False
    
    except subprocess.CalledProcessError as exc:
        if verbose:
            print(f"[main] ❌ Konversi dataset gagal: {exc}")
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
        return False
    
    except Exception as exc:
        if verbose:
            print(f"[main] ❌ Error saat konversi dataset: {exc}")
        return False


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def limited_batches(trainer: Trainer, limit: Optional[int]):
    """Generator untuk membatasi jumlah batch evaluasi."""
    if limit is not None and limit <= 0:
        limit = None

    if not trainer.qa_samples:
        trainer.load_qa_dataset(verbose=False)

    for i, batch in enumerate(trainer.iter_batches(epoch=0, total_epochs=1, shuffle=False)):
        if limit is not None and i >= limit:
            break
        yield batch


def evaluate_model(
    model,
    trainer: Trainer,
    evaluator: Evaluator,
    max_batches: Optional[int],
) -> Optional[dict]:
    """Menjalankan audit evaluasi memakai evaluator.audit()."""
    batch_generator = limited_batches(trainer, max_batches)

    report = evaluator.audit(
        model=model,
        batches=batch_generator,
    )

    if report.get("num_valid_tokens", 0) == 0:
        return None

    return report


# ============================================================================
# EXPORT HELPER
# ============================================================================

def export_model(
    model,
    tokenizer,
    evaluator: Evaluator,
    config: Config,
    report: Optional[dict],
    args: argparse.Namespace,
) -> None:
    """
    Menentukan apakah model layak di-export, lalu menjalankannya.
    """
    print_section("EXPORT MODEL (RAG)")

    should_export = True
    reason = "Export dijalankan."

    has_threshold = (
        args.max_export_loss is not None
        or args.max_export_ppl is not None
        or args.min_export_accuracy is not None
    )

    if report is None and has_threshold:
        should_export = False
        reason = "Tidak ada report evaluasi untuk mengecek threshold export."

    if report is not None and has_threshold:
        should_export = evaluator.should_export(
            report=report,
            max_loss=args.max_export_loss,
            max_perplexity=args.max_export_ppl,
            min_accuracy=args.min_export_accuracy,
        )

        if not should_export:
            reason = "Model belum memenuhi threshold export."

    if args.force_export:
        should_export = True
        reason = "Export dipaksa dengan --force-export."

    if not should_export:
        print(f"[main] Export dibatalkan. {reason}")
        print("[main] Gunakan --force-export jika tetap ingin menyimpan model.")
        return

    exporter = ModelExporter(config)

    output_path = exporter.save(
        model=model,
        tokenizer=tokenizer,
        output_path=args.export_path,
        export_dtype=args.export_dtype,
        include_vocab=True,
        include_rag_assets=True,
        verbose=True,
    )

    file_size = Path(output_path).stat().st_size

    print()
    print(f"[main] ✅ {reason}")
    print(f"[main] Model berhasil di-export ke:")
    print(f"       {output_path}")
    print(f"[main] Ukuran file: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    info = exporter.inspect(output_path)
    print()
    print(f"[main] Model info:")
    print(f"  Architecture   : {info['architecture']}")
    print(f"  Tensor count   : {info['tensor_count']}")
    print(f"  Has tokenizer  : {info['has_tokenizer']}")
    print(f"  Has index      : {info['has_retrieval_index']}")
    if info['retrieval_index_stats']:
        stats = info['retrieval_index_stats']
        print(f"    - Docs       : {stats.get('total_docs', 0)}")
        print(f"    - Terms      : {stats.get('total_terms', 0)}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: Config, args: argparse.Namespace) -> None:
    """
    Menjalankan pipeline RAG dengan SOTA training.
    """
    config.ensure_dirs()
    print_config_summary(config)

    # --------------------------------------------------------------------
    # 0. DATASET DETECTION
    # --------------------------------------------------------------------
    info = load_dataset_info(config)
    print_dataset_info(info)

    # --------------------------------------------------------------------
    # 1. TOKENIZER
    # --------------------------------------------------------------------
    print_section("TAHAP 1: TOKENIZER")

    tokenizer = Tokenizer(config)

    if args.resume and config.paths.vocab_path.exists():
        tokenizer.load()
        print(f"[main] Tokenizer dimuat dari: {config.paths.vocab_path}")
        print(f"[main] Vocab size: {tokenizer.vocab_size:,}")
    else:
        tokenizer.train()
        print(f"[main] Tokenizer trained.")
        print(f"[main] Vocab disimpan di: {config.paths.vocab_path}")
        print(f"[main] Vocab size: {tokenizer.vocab_size:,}")

    # --------------------------------------------------------------------
    # 1.5 DATASET PREPARATION
    # --------------------------------------------------------------------
    if not args.skip_convert and not args.interactive:
        print_section("TAHAP 1.5: DATASET PREPARATION")
        
        force_convert = getattr(args, 'force_convert', False)
        
        if not prepare_dataset(config, force=force_convert, verbose=True):
            print("[main] ❌ Dataset tidak tersedia. Training tidak bisa dilanjutkan.")
            print("[main]    Jalankan: python scripts/download_datasets.py")
            print("[main]    Atau tambahkan file .txt ke datasets/")
            return

    # --------------------------------------------------------------------
    # 2. MODEL
    # --------------------------------------------------------------------
    print_section("TAHAP 2: ARSITEKTUR MODEL (RAG + Cross-Attention)")

    model = CustomTransformerLM(config)

    print(f"[main] Model dibuat.")
    print(f"[main] Parameter count: {model.parameter_count():,}")
    print(f"[main] Architecture   : Single LM head + Cross-Attention")
    print(f"[main] Weight tying   : {model.use_weight_tying}")
    print(f"[main] RoPE           : {model.use_rope}")
    
    # Print cross-attention info
    if hasattr(model, 'cross_attention_layer_indices'):
        n_cross = len(model.cross_attention_layer_indices)
        print(f"[main] Cross-Attention: {n_cross} layers ({sorted(model.cross_attention_layer_indices)})")
    
    if hasattr(model, 'use_constrained_generation'):
        print(f"[main] Constrained Gen: {model.use_constrained_generation}")

    # --------------------------------------------------------------------
    # 3. EVALUATOR & WEIGHT MANAGER
    # --------------------------------------------------------------------
    print_section("TAHAP 3: EVALUATOR & WEIGHT MANAGER (SOTA)")

    evaluator = Evaluator(config)
    weight_manager = WeightManager(model, config)

    print("[main] Evaluator siap (dengan SOTA loss functions).")
    print("[main] WeightManager siap.")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        weight_manager=weight_manager,
        config=config,
    )

    # --------------------------------------------------------------------
    # 4. RESUME CHECKPOINT
    # --------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    resumed = False

    if args.resume:
        try:
            ckpt_path = weight_manager.load_checkpoint()
            print(f"[main] Checkpoint dimuat dari: {ckpt_path}")
            print(f"[main] Optimizer step: {weight_manager.step_count}")

            last_epoch = getattr(weight_manager, "last_epoch", 0)
            global_step = getattr(weight_manager, "global_step", weight_manager.step_count)

            start_epoch = last_epoch
            resumed = True

            print(f"[main] Last epoch: {last_epoch}")
            print(f"[main] Global step: {global_step}")
            
            # Load EMA weights jika tersedia
            if hasattr(weight_manager, 'load_ema_weights'):
                try:
                    weight_manager.load_ema_weights()
                    print("[main] ✅ EMA weights loaded")
                except Exception:
                    pass
        except FileNotFoundError:
            print("[main] Checkpoint tidak ditemukan. Mulai dari awal.")
            resumed = False
        except Exception as exc:
            print(f"[main] Gagal memuat checkpoint: {exc}")
            print("[main] Mulai dari awal.")
            resumed = False

    if resumed:
        trainer.set_resume_state(start_epoch=start_epoch, global_step=global_step)

    # Variabel report untuk export
    report: Optional[dict] = None

    # ====================================================================
    # MODE INTERACTIVE vs MODE PIPELINE
    # ====================================================================

    if args.interactive:
        # ----------------------------------------------------------------
        # MODE INTERACTIVE
        # ----------------------------------------------------------------
        print_section("🎮 PLAYGROUND INTERACTIVE (RAG + SOTA)")

        if resumed:
            print(f"[main] Model dimuat dari checkpoint (epoch {start_epoch}, step {global_step}).")
        else:
            print("[main] Model baru dibuat (belum dilatih).")

        # Load QA dataset untuk trainer (untuk inline training)
        n_loaded = trainer.load_qa_dataset(verbose=True)
        if n_loaded > 0:
            print(f"[main] ✅ {n_loaded:,} QA samples siap untuk inline training.")
        else:
            print(f"[main] ⚠️ QA dataset kosong. Jalankan konversi dulu atau training via pipeline.")

        print()
        print("[main] 📚 Commands yang tersedia:")
        print("  /training --epochs N   : Latih model N epochs dengan SOTA techniques")
        print("  /training --steps N    : Latih model N steps")
        print("  /retrieve [query]      : Test retrieval (BM25)")
        print("  /context [on|off]      : Toggle context injection")
        print("  /status                : Lihat SOTA training stats")
        print("  /help                  : Bantuan lengkap")
        print()
        print("[main] Ketik prompt biasa untuk RAG inference.")

        playground = Playground(
            model=model,
            tokenizer=tokenizer,
            config=config,
            trainer=trainer,
        )

        playground.interactive(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            min_new_tokens=args.min_new_tokens,
            debug_tokens=args.debug_tokens,
            debug_retrieval=not args.no_debug_retrieval,
        )

        if not args.skip_export:
            export_model(
                model=model,
                tokenizer=tokenizer,
                evaluator=evaluator,
                config=config,
                report=report,
                args=args,
            )

    else:
        # ----------------------------------------------------------------
        # MODE PIPELINE
        # ----------------------------------------------------------------

        # TRAINING
        if not args.skip_training:
            print_section("🎯 TRAINING (RAG + SOTA Techniques)")

            if not trainer.qa_samples:
                n_loaded = trainer.load_qa_dataset(verbose=True)
                if n_loaded == 0:
                    print("[main] ❌ QA dataset tidak ditemukan.")
                    print("[main]    Jalankan: python scripts/download_datasets.py")
                    print("[main]    Lalu: python scripts/prepare_datasets.py")
                    return

            print(f"\n[main] Starting SOTA training dengan {len(trainer.qa_samples):,} samples...")
            print(f"[main] Model parameters: {model.parameter_count():,}")
            
            # Calculate training time estimate
            steps_per_epoch = len(trainer.qa_samples) // config.training.batch_size
            total_steps = steps_per_epoch * config.training.epochs
            est_time_hours = (total_steps * 0.3) / 3600  # ~300ms per step
            
            print(f"[main] Steps per epoch: {steps_per_epoch:,}")
            print(f"[main] Total steps    : {total_steps:,}")
            print(f"[main] Est. time      : {est_time_hours:.1f} hours")
            print()

            history = trainer.run_epochs(
                max_steps=args.max_steps,
                save_checkpoints=True,
                verbose=True,
                start_epoch=start_epoch,
                global_step=global_step,
            )

            if history:
                report = history[-1]

                print()
                print_section("📊 TRAINING RESULTS")
                print(f"[main] ✅ Training selesai.")
                print(f"  Epoch terakhir     : {report.get('epoch')}")
                print(f"  Stage              : {report.get('stage', 'N/A')}")
                print(f"  Loss               : {report.get('loss'):.6f}")
                print(f"  Perplexity         : {report.get('perplexity'):.4f}")
                print(f"  Accuracy           : {report.get('accuracy'):.4f}")
                print(f"  Top-5 Accuracy     : {report.get('top5_accuracy', 'N/A')}")
                print(f"  Valid tokens       : {report.get('num_valid_tokens'):,}")
                
                # SOTA stats
                if hasattr(trainer, 'get_sota_stats'):
                    stats = trainer.get_sota_stats()
                    print()
                    print("  🎯 SOTA Statistics:")
                    print(f"    Spikes detected     : {stats.get('spikes_detected', 0)}")
                    print(f"    Self-corrections    : {stats.get('corrections_applied', 0)}")
                    print(f"    Hard samples retrain: {stats.get('hard_samples_retrained', 0)}")
            else:
                print("[main] ⚠️ Training tidak menghasilkan history.")
        else:
            print_section("TRAINING")
            print("[main] Training dilewati karena --skip-training.")

        # EVALUATOR AUDIT
        if not args.skip_eval:
            print_section("📈 EVALUATOR AUDIT")

            eval_report = evaluate_model(
                model=model,
                trainer=trainer,
                evaluator=evaluator,
                max_batches=args.eval_batches,
            )

            if eval_report is not None:
                report = eval_report

                print("[main] ✅ Audit evaluasi selesai.")
                print(f"  Eval loss       : {eval_report['loss']:.6f}")
                print(f"  Eval perplexity : {eval_report['perplexity']:.4f}")
                print(f"  Eval accuracy   : {eval_report['accuracy']:.4f}")
                print(f"  Valid tokens    : {eval_report['num_valid_tokens']:,}")
                print(f"  Eval batches    : {eval_report['num_batches']}")
            else:
                print("[main] Audit evaluasi tidak menghasilkan token valid.")
        else:
            print_section("EVALUATOR AUDIT")
            print("[main] Evaluasi dilewati karena --skip-eval.")

        # PLAYGROUND TEST
        if not args.skip_test:
            print_section("🧪 PLAYGROUND TEST (RAG Inference)")

            playground = Playground(
                model=model,
                tokenizer=tokenizer,
                config=config,
                trainer=trainer,
            )

            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                min_new_tokens=args.min_new_tokens,
                verbose=True,
                debug_tokens=args.debug_tokens,
                debug_retrieval=not args.no_debug_retrieval,
            )
        else:
            print_section("PLAYGROUND TEST")
            print("[main] Testing dilewati karena --skip-test.")

        # EXPORT
        if not args.skip_export:
            export_model(
                model=model,
                tokenizer=tokenizer,
                evaluator=evaluator,
                config=config,
                report=report,
                args=args,
            )
        else:
            print_section("EXPORT MODEL")
            print("[main] Export dilewati karena --skip-export.")

    print_section("✅ PIPELINE SELESAI")


# ============================================================================
# DEMO PIPELINE
# ============================================================================

def run_demo(args: argparse.Namespace) -> None:
    """
    Menjalankan pipeline demo kecil dengan SOTA training.
    """
    print_section("MODE DEMO (RAG + SOTA)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        # Dataset demo
        mobil_file = datasets_dir / "mobil.txt"
        mobil_file.write_text(
            "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.\n\n"
            "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan dengan aman.\n\n"
            "Bahan bakar mobil bisa berupa bensin, diesel, atau listrik untuk kendaraan elektrik.\n\n"
            "Transmisi mobil mengatur perpindahan gigi untuk efisiensi tenaga mesin.\n\n"
            "Sistem kemudi mobil menggunakan setir yang terhubung ke roda depan.",
            encoding="utf-8",
        )

        sains_file = datasets_dir / "sains.txt"
        sains_file.write_text(
            "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.\n\n"
            "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.\n\n"
            "Matahari adalah bintang terdekat dengan Bumi yang menjadi sumber energi utama.\n\n"
            "Atom terdiri dari proton, neutron, dan elektron yang membentuk struktur materi.\n\n"
            "Gravitasi adalah gaya tarik menarik antara dua benda yang memiliki massa.",
            encoding="utf-8",
        )

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
            retrieval_index_path=output_models_dir / "retrieval_index.json",
            qa_dataset_path=output_models_dir / "qa_dataset.json",
        )

        demo_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=2,
                sequence_length=48,
                dropout_rate=0.1,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                # SOTA parameters (aktif semua)
                use_cosine_schedule=True,
                use_focal_loss=True,
                use_agc=True,
                use_curriculum=True,
                use_self_correction=True,
                use_ema=True,
                label_smoothing=0.1,
                focal_gamma=2.0,
                # RAG parameters
                context_dropout_rate=0.2,
                context_perturbation_rate=0.3,
                progressive_stages=2,
                max_context_paragraphs=2,
                augmentation_factor=2,
            ),
            model=ModelConfig(
                vocab_size=256,
                embedding_dim=64,
                num_attention_heads=4,
                num_layers=2,
                max_position_embeddings=128,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=128,
                train_dtype="float32",
                export_dtype="float32",
                use_rope=True,
                use_weight_tying=True,
            ),
            retrieval=RetrievalConfig(
                top_k=2,
                bm25_k1=1.5,
                bm25_b=0.75,
                max_paragraph_length=300,
            ),
        )

        demo_args = copy.copy(args)
        demo_args.resume = False
        demo_args.export_path = None
        demo_args.interactive = False
        demo_args.skip_convert = False
        demo_args.force_convert = True

        if demo_args.max_steps is None:
            demo_args.max_steps = None

        if demo_args.eval_batches != 0:
            demo_args.eval_batches = 2

        print("[main] Menjalankan demo pipeline RAG + SOTA:")
        print(f"[main] {tmp_path}")
        print()

        run_pipeline(demo_config, demo_args)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CreateMiniBot RAG+SOTA Pipeline",
        description=(
            "Orkestrator pipeline RAG (Retrieval-Augmented Generation) dengan SOTA training: "
            "tokenizer, dataset preparation, SOTA trainer, evaluator, playground, dan exporter."
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Jalankan demo pipeline kecil dengan dataset sementara.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Muat checkpoint optimizer/model dan lanjutkan dari epoch terakhir.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Masuk langsung ke playground interaktif dengan RAG inference. "
            "Training dilakukan via perintah 'training' di playground."
        ),
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Lewati tahap training (mode pipeline).",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Lewati tahap audit/evaluasi.",
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Lewati tahap playground testing.",
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Lewati tahap export model.",
    )

    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Lewati preparation dataset (asumsikan sudah ada).",
    )

    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Paksa rebuild dataset meskipun sudah ada.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Apa itu mobil?",
        help="Prompt untuk testing non-interaktif.",
    )

    # ---- PLAYGROUND PARAMETERS ----
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Jumlah maksimum token baru saat testing.",
    )

    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=4,
        help="Jumlah minimum token sebelum EOS diperbolehkan.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature sampling untuk playground.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling untuk playground.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-P sampling untuk playground.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Repetition penalty.",
    )

    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Tampilkan Token IDs dan Token Pieces secara detail.",
    )

    parser.add_argument(
        "--no-debug-retrieval",
        action="store_true",
        help="Jangan tampilkan info retrieval saat inference.",
    )

    # ---- TRAINING CONTROL ----
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Batasi jumlah step training global.",
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=4,
        help="Jumlah batch maksimum untuk audit evaluasi. 0 berarti semua.",
    )

    # ---- EXPORT THRESHOLDS ----
    parser.add_argument(
        "--max-export-loss",
        type=float,
        default=None,
        help="Threshold loss maksimum agar model boleh di-export.",
    )

    parser.add_argument(
        "--max-export-ppl",
        type=float,
        default=None,
        help="Threshold perplexity maksimum.",
    )

    parser.add_argument(
        "--min-export-accuracy",
        type=float,
        default=None,
        help="Threshold accuracy minimum.",
    )

    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Paksa export model tanpa memedulikan threshold.",
    )

    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="Path khusus untuk file model hasil export.",
    )

    parser.add_argument(
        "--export-dtype",
        type=str,
        default=None,
        help="Presisi export: float32, float16, bfloat16, int8, atau int4.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN ENTRY
# ============================================================================

def main() -> None:
    args = parse_args()

    try:
        if args.demo:
            run_demo(args)
        else:
            run_pipeline(CONFIG, args)

    except KeyboardInterrupt:
        print()
        print("[main] Pipeline dihentikan oleh pengguna.")
        sys.exit(1)

    except RuntimeError as exc:
        print()
        print("[main] ERROR RUNTIME:")
        print(str(exc))

        message = str(exc).lower()

        if "parameter" in message and "terlalu besar" in message:
            print()
            print("Saran:")
            print("1. Jalankan mode demo:      python main.py --demo")
            print("2. Kurangi ukuran model di  config.py")
            print("3. Atau set CustomTransformerLM.MAX_PARAMS = None")

        sys.exit(1)


if __name__ == "__main__":
    main()
