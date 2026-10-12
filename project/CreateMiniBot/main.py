#!/usr/bin/env python3
# main.py
#
# Entry point utama dan orkestrator pipeline untuk RAG Architecture.
#
# Arsitektur: Single-Head + BM25 Retrieval
# - Context-conditioned training
# - Loss ↓ = Presisi ↑ (korelasi langsung, no topic mixing)
# - Retrieval-based context injection saat inference
#
# Mode operasi:
# 1. MODE PIPELINE (default):
#    Tokenizer → Convert Dataset → Training → Eval → Test → Export
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
# --skip-convert     : lewati konversi dataset (jika qa_dataset.json sudah ada)
# --debug-tokens     : tampilkan Token IDs dan Token Pieces saat inference
#
# Keterhubungan:
# - config.py              : konfigurasi global
# - core/tokenizer.py      : tokenizer BPE dengan RAG templates
# - core/architecture.py   : model transformer (single LM head)
# - core/retriever.py      : BM25 retrieval engine
# - core/model_exporter.py : export ke safetensors + RAG assets
# - pipeline/trainer.py    : context-conditioned training
# - pipeline/evaluator.py  : evaluasi model
# - pipeline/weight_manager.py : load/save checkpoint
# - scripts/convert_dataset.py : konversi dataset ke QA format
# - testing/playground.py  : RAG inference testing

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

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
    """Print summary dari config, termasuk RAG parameters."""
    print_section("KONFIGURASI PIPELINE (RAG Architecture)")

    print(f"Datasets dir         : {config.paths.datasets_dir}")
    print(f"Output models dir    : {config.paths.output_models_dir}")
    print(f"Vocab path           : {config.paths.vocab_path}")
    print(f"Final model path     : {config.paths.final_model_path}")
    print(f"Checkpoint path      : {config.paths.checkpoint_path}")
    print(f"QA dataset path      : {config.paths.qa_dataset_path}")
    print(f"Retrieval index path : {config.paths.retrieval_index_path}")

    print()
    print("Model:")
    print(f"  vocab_size              : {config.model.vocab_size}")
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
    print(f"  inline_training_steps        : {config.training.inline_training_steps}")

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
# DATASET CONVERSION HELPER
# ============================================================================

def convert_dataset(
    config: Config,
    augmentation_factor: Optional[int] = None,
    preview: int = 0,
    verbose: bool = True,
) -> bool:
    """
    Jalankan script konversi dataset mentah ke format QA.

    Memanggil scripts/convert_dataset.py sebagai subprocess.

    Args:
        config: Config object
        augmentation_factor: override augmentation factor (default dari config)
        preview: jumlah samples untuk preview (0 = no preview)
        verbose: print progress

    Returns:
        True jika sukses, False jika gagal
    """
    script_path = Path(__file__).resolve().parent / "scripts" / "convert_dataset.py"

    if not script_path.exists():
        if verbose:
            print(f"[main] ⚠️ Script konversi tidak ditemukan: {script_path}")
            print(f"[main]    Buat script atau konversi manual ke format QA.")
        return False

    # Build command
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
        print(f"[main] Command: {' '.join(cmd)}")

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
        
        # Verifikasi output
        qa_path = Path(config.paths.qa_dataset_path)
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            n_samples = len(data.get("samples", []))
            if verbose:
                print(f"[main] ✅ Konversi sukses: {n_samples} QA samples")
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

def limited_batches(trainer: Trainer, limit: int | None):
    """
    Generator untuk membatasi jumlah batch evaluasi.
    """
    if limit is not None and limit <= 0:
        limit = None

    # Load QA dataset jika belum
    if not trainer.qa_samples:
        trainer.load_qa_dataset(verbose=False)

    # iter_batches butuh epoch dan total_epochs
    for i, batch in enumerate(trainer.iter_batches(epoch=0, total_epochs=1, shuffle=False)):
        if limit is not None and i >= limit:
            break
        yield batch


def evaluate_model(
    model,
    trainer: Trainer,
    evaluator: Evaluator,
    max_batches: int | None,
) -> dict | None:
    """
    Menjalankan audit evaluasi memakai evaluator.audit().
    """
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
    report: dict | None,
    args: argparse.Namespace,
) -> None:
    """
    Menentukan apakah model layak di-export, lalu menjalankannya.

    Untuk RAG: export akan copy retrieval_index.json + otak_engine.py
    sebagai file terpisah di output folder.
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
        include_rag_assets=True,  # Copy retrieval index + engine
        verbose=True,
    )

    file_size = Path(output_path).stat().st_size

    print()
    print(f"[main] ✅ {reason}")
    print(f"[main] Model berhasil di-export ke:")
    print(f"       {output_path}")
    print(f"[main] Ukuran file: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    # Inspect hasil export
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
    Menjalankan pipeline RAG.

    MODE INTERACTIVE (--interactive):
        Tokenizer → Load Model → Playground (dengan RAG inference)
        Training dilakukan via perintah 'training' di playground.

    MODE PIPELINE (default):
        Tokenizer → Convert Dataset → Training → Eval → Test → Export
    """
    config.ensure_dirs()
    print_config_summary(config)

    # --------------------------------------------------------------------
    # 1. TOKENIZER
    # --------------------------------------------------------------------
    print_section("TAHAP 1: TOKENIZER")

    tokenizer = Tokenizer(config)

    if args.resume and config.paths.vocab_path.exists():
        tokenizer.load()
        print(f"[main] Tokenizer dimuat dari: {config.paths.vocab_path}")
    else:
        tokenizer.train()
        print(f"[main] Tokenizer trained.")
        print(f"[main] Vocab disimpan di: {config.paths.vocab_path}")
        print(f"[main] Vocab size: {tokenizer.vocab_size}")

    # --------------------------------------------------------------------
    # 1.5 DATASET CONVERSION (opsional)
    # --------------------------------------------------------------------
    if not args.skip_convert and not args.interactive:
        print_section("TAHAP 1.5: DATASET CONVERSION")

        qa_path = Path(config.paths.qa_dataset_path)

        if qa_path.exists() and not args.force_convert:
            print(f"[main] QA dataset sudah ada: {qa_path}")
            print(f"[main] Lewati konversi. Gunakan --force-convert untuk rebuild.")
            
            # Load untuk validasi
            try:
                with open(qa_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_samples = len(data.get("samples", []))
                print(f"[main] ✅ Loaded {n_samples} existing QA samples")
            except Exception as exc:
                print(f"[main] ⚠️ Gagal load existing dataset: {exc}")
                print(f"[main] Akan coba generate baru...")
                convert_dataset(config, verbose=True)
        else:
            convert_dataset(config, preview=3, verbose=True)

    # --------------------------------------------------------------------
    # 2. MODEL
    # --------------------------------------------------------------------
    print_section("TAHAP 2: ARSITEKTUR MODEL (RAG Single-Head)")

    model = CustomTransformerLM(config)

    print(f"[main] Model dibuat.")
    print(f"[main] Parameter count: {model.parameter_count():,}")
    print(f"[main] Architecture   : Single LM head (no domain contamination)")
    print(f"[main] Weight tying   : {model.use_weight_tying}")
    print(f"[main] RoPE           : {model.use_rope}")

    # --------------------------------------------------------------------
    # 3. EVALUATOR & WEIGHT MANAGER
    # --------------------------------------------------------------------
    print_section("TAHAP 3: EVALUATOR & WEIGHT MANAGER")

    evaluator = Evaluator(config)
    weight_manager = WeightManager(model, config)

    print("[main] Evaluator siap.")
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
    report: dict | None = None

    # ====================================================================
    # MODE INTERACTIVE vs MODE PIPELINE
    # ====================================================================

    if args.interactive:
        # ----------------------------------------------------------------
        # MODE INTERACTIVE
        # Langsung masuk playground dengan RAG inference.
        # Training dilakukan via perintah 'training' di playground.
        # ----------------------------------------------------------------
        print_section("PLAYGROUND INTERACTIVE (RAG)")

        if resumed:
            print(f"[main] Model dimuat dari checkpoint (epoch {start_epoch}, step {global_step}).")
        else:
            print("[main] Model baru dibuat (belum dilatih).")

        # Load QA dataset untuk trainer (untuk inline training)
        qa_path = Path(config.paths.qa_dataset_path)
        if qa_path.exists():
            n_loaded = trainer.load_qa_dataset(verbose=True)
            if n_loaded > 0:
                print(f"[main] ✅ {n_loaded} QA samples siap untuk inline training.")
            else:
                print(f"[main] ⚠️ QA dataset kosong. Jalankan konversi dulu atau training via pipeline.")
        else:
            print(f"[main] ⚠️ QA dataset belum ada: {qa_path}")
            print(f"[main]    Jalankan: python scripts/convert_dataset.py")
            print(f"[main]    Atau jalankan tanpa --interactive untuk pipeline lengkap.")

        print()
        print("[main] Gunakan perintah '/training' untuk melatih lebih lanjut.")
        print("[main] Gunakan '/retrieve' untuk test retrieval.")
        print("[main] Ketik prompt biasa untuk RAG inference.")

        # Buat playground
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

        # Setelah keluar dari playground, export jika diminta
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
        # Training -> Eval -> Test -> Export
        # ----------------------------------------------------------------

        # TRAINING
        if not args.skip_training:
            print_section("TRAINING (RAG Context-Conditioned)")

            # Pastikan QA dataset ada
            if not trainer.qa_samples:
                n_loaded = trainer.load_qa_dataset(verbose=True)
                if n_loaded == 0:
                    print("[main] ❌ QA dataset tidak ditemukan.")
                    print("[main]    Jalankan dengan --skip-training=false tanpa --skip-convert")
                    print("[main]    Atau: python scripts/convert_dataset.py")
                    return

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
                print("[main] ✅ Training selesai.")
                print(f"[main] Epoch terakhir : {report.get('epoch')}")
                print(f"[main] Stage          : {report.get('stage', 'N/A')}")
                print(f"[main] Loss           : {report.get('loss'):.6f}")
                print(f"[main] Perplexity     : {report.get('perplexity'):.4f}")
                print(f"[main] Accuracy       : {report.get('accuracy'):.4f}")
                print(f"[main] Valid tokens   : {report.get('num_valid_tokens')}")
            else:
                print("[main] ⚠️ Training tidak menghasilkan history.")
        else:
            print_section("TRAINING")
            print("[main] Training dilewati karena --skip-training.")

        # EVALUATOR AUDIT
        if not args.skip_eval:
            print_section("EVALUATOR AUDIT")

            eval_report = evaluate_model(
                model=model,
                trainer=trainer,
                evaluator=evaluator,
                max_batches=args.eval_batches,
            )

            if eval_report is not None:
                report = eval_report

                print("[main] Audit evaluasi selesai.")
                print(f"[main] Eval loss       : {eval_report['loss']:.6f}")
                print(f"[main] Eval perplexity : {eval_report['perplexity']:.4f}")
                print(f"[main] Eval accuracy   : {eval_report['accuracy']:.4f}")
                print(f"[main] Valid tokens    : {eval_report['num_valid_tokens']}")
                print(f"[main] Eval batches    : {eval_report['num_batches']}")
            else:
                print("[main] Audit evaluasi tidak menghasilkan token valid.")
                print("[main] Pastikan QA dataset sudah di-generate.")
        else:
            print_section("EVALUATOR AUDIT")
            print("[main] Evaluasi dilewati karena --skip-eval.")

        # PLAYGROUND TEST (non-interactive dengan RAG)
        if not args.skip_test:
            print_section("PLAYGROUND TEST (RAG Inference)")

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

    print_section("PIPELINE SELESAI")


# ============================================================================
# DEMO PIPELINE
# ============================================================================

def run_demo(args: argparse.Namespace) -> None:
    """
    Menjalankan pipeline demo kecil memakai folder sementara.

    Demo ini menunjukkan flow RAG:
    - Corpus mentah (tanpa prefix domain)
    - Auto konversi ke QA format
    - Training dengan context conditioning
    - Inference dengan BM25 retrieval
    """
    print_section("MODE DEMO (RAG)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        # Dataset demo (tanpa domain prefix - RAG style)
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

        biologi_file = datasets_dir / "biologi.txt"
        biologi_file.write_text(
            "Sel adalah unit terkecil dari makhluk hidup yang memiliki fungsi spesifik.\n\n"
            "DNA adalah materi genetik yang menyimpan informasi keturunan dalam sel.\n\n"
            "Metamorfosis adalah proses perubahan bentuk tubuh hewan dari tahap muda ke dewasa.\n\n"
            "Ekosistem adalah hubungan timbal balik antara makhluk hidup dengan lingkungannya.",
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
            demo_args.max_steps = None  # Run full epochs

        if demo_args.eval_batches != 0:
            demo_args.eval_batches = 2

        print("[main] Menjalankan demo pipeline RAG dengan folder sementara:")
        print(f"[main] {tmp_path}")
        print()

        run_pipeline(demo_config, demo_args)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CreateMiniBot RAG Pipeline",
        description=(
            "Orkestrator pipeline RAG (Retrieval-Augmented Generation): "
            "tokenizer, dataset converter, trainer, evaluator, playground, dan exporter."
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
        help="Lewati konversi dataset (asumsikan qa_dataset.json sudah ada).",
    )

    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Paksa konversi dataset meskipun qa_dataset.json sudah ada.",
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
        help="Temperature sampling untuk playground (default: 0.3 untuk determinisme).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling untuk playground (default: 20 untuk fokus).",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-P sampling untuk playground (default: 0.85).",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Repetition penalty. Nilai > 1 mengurangi pengulangan (default: 1.3).",
    )

    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Tampilkan Token IDs dan Token Pieces secara detail saat inference.",
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
        help="Batasi jumlah step training global (berguna untuk testing).",
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
        help="Threshold perplexity maksimum agar model boleh di-export.",
    )

    parser.add_argument(
        "--min-export-accuracy",
        type=float,
        default=None,
        help="Threshold accuracy minimum agar model boleh di-export.",
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
        help=(
            "Presisi export: float32, float16, bfloat16, int8, atau int4. "
            "Jika tidak diisi, memakai config.model.export_dtype."
        ),
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
            print("   - vocab_size")
            print("   - embedding_dim")
            print("   - num_layers")
            print("   - ffn_hidden_dim")
            print("3. Atau nonaktifkan batas aman dengan:")
            print("   CustomTransformerLM.MAX_PARAMS = None")
            print("   (tidak disarankan untuk CPU / implementasi custom)")

        sys.exit(1)


if __name__ == "__main__":
    main()
