#!/usr/bin/env python3
# main.py
#
# Entry point utama dan orkestrator pipeline.
#
# Alur utama:
# 1. Tokenizer.train()
# 2. Trainer.run_epochs()
# 3. Evaluator.audit()
# 4. Playground.test_prompt() atau interactive
# 5. ModelExporter.save()
#
# Catatan:
# - Weight update sudah terjadi di dalam Trainer melalui Evaluator + WeightManager.
# - main.py tetap menyediakan audit terpisah setelah training.
# - Resume sekarang benar-benar melanjutkan dari epoch/global step terakhir.

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

from config import (
    CONFIG,
    Config,
    ModelConfig,
    PathsConfig,
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
    print_section("KONFIGURASI PIPELINE")

    print(f"Datasets dir       : {config.paths.datasets_dir}")
    print(f"Output models dir  : {config.paths.output_models_dir}")
    print(f"Vocab path         : {config.paths.vocab_path}")
    print(f"Final model path   : {config.paths.final_model_path}")
    print(f"Checkpoint path    : {config.paths.checkpoint_path}")

    print()
    print("Model:")
    print(f"  vocab_size              : {config.model.vocab_size}")
    print(f"  embedding_dim           : {config.model.embedding_dim}")
    print(f"  num_attention_heads     : {config.model.num_attention_heads}")
    print(f"  num_layers              : {config.model.num_layers}")
    print(f"  ffn_hidden_dim          : {config.model.ffn_hidden_dim}")
    print(f"  max_position_embeddings : {config.model.max_position_embeddings}")
    print(f"  dropout_rate (model)    : {config.model.dropout_rate}")
    print(f"  train_dtype             : {config.model.train_dtype}")
    print(f"  export_dtype            : {config.model.export_dtype}")

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
    print(f"  lr_decay_steps               : {config.training.lr_decay_steps}")
    print(f"  lr_decay_factor              : {config.training.lr_decay_factor}")
    print(f"  lr_min                       : {config.training.lr_min}")
    print(f"  gradient_accumulation_steps  : {config.training.gradient_accumulation_steps}")
    print(f"  early_stopping_patience      : {config.training.early_stopping_patience}")
    print(f"  inline_training_steps        : {config.training.inline_training_steps}")


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def limited_batches(trainer: Trainer, limit: int | None):
    """
    Generator untuk membatasi jumlah batch evaluasi.

    Jika limit None, seluruh batch dipakai.
    Jika limit 0, juga dianggap seluruh batch.
    """
    if limit is not None and limit <= 0:
        limit = None

    for i, batch in enumerate(trainer.iter_batches(shuffle=False)):
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
    """
    print_section("EXPORT MODEL")

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
    )

    file_size = Path(output_path).stat().st_size

    print(f"[main] {reason}")
    print(f"[main] Model berhasil di-export ke:")
    print(f"       {output_path}")
    print(f"[main] Ukuran file: {file_size:,} bytes")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: Config, args: argparse.Namespace) -> None:
    """
    Menjalankan pipeline penuh:
    Tokenizer -> Trainer -> Evaluator -> Playground -> Exporter
    """
    config.ensure_dirs()
    print_config_summary(config)

    # ------------------------------------------------------------------------
    # 1. TOKENIZER
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # 2. MODEL
    # ------------------------------------------------------------------------
    print_section("TAHAP 2: ARSITEKTUR MODEL")

    model = CustomTransformerLM(config)

    print(f"[main] Model dibuat.")
    print(f"[main] Parameter count: {model.parameter_count():,}")

    # ------------------------------------------------------------------------
    # 3. EVALUATOR & WEIGHT MANAGER
    # ------------------------------------------------------------------------
    print_section("TAHAP 3: EVALUATOR & WEIGHT MANAGER")

    evaluator = Evaluator(config)
    weight_manager = WeightManager(model, config)

    print("[main] Evaluator siap.")
    print("[main] WeightManager siap.")

    # ------------------------------------------------------------------------
    # 4. TRAINER
    # ------------------------------------------------------------------------
    print_section("TAHAP 4: TRAINER")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        weight_manager=weight_manager,
        config=config,
    )

    # ------------------------------------------------------------------------
    # 5. RESUME CHECKPOINT
    # ------------------------------------------------------------------------
    # Variabel untuk resume training dari epoch tertentu
    start_epoch = 0
    global_step = 0
    resumed = False

    if args.resume:
        try:
            ckpt_path = weight_manager.load_checkpoint()
            print(f"[main] Checkpoint dimuat dari: {ckpt_path}")
            print(f"[main] Optimizer step: {weight_manager.step_count}")

            # Coba baca metadata epoch dan global step dari weight_manager
            last_epoch = getattr(weight_manager, "last_epoch", 0)
            global_step = getattr(weight_manager, "global_step", weight_manager.step_count)

            start_epoch = last_epoch
            resumed = True

            print(f"[main] Last epoch: {last_epoch}")
            print(f"[main] Global step: {global_step}")
            print(f"[main] Training akan dilanjutkan dari epoch {last_epoch + 1}")
        except FileNotFoundError:
            print("[main] Checkpoint tidak ditemukan. Training dari awal.")
            resumed = False
        except Exception as exc:
            print(f"[main] Gagal memuat checkpoint: {exc}")
            print("[main] Training dari awal.")
            resumed = False

    # Informasikan trainer tentang state resume
    # Trainer nanti akan menggunakan start_epoch dan global_step ini
    if resumed:
        trainer.set_resume_state(start_epoch=start_epoch, global_step=global_step)

    # ------------------------------------------------------------------------
    # 6. TRAINING
    # ------------------------------------------------------------------------
    print_section("TAHAP 5: TRAINING")

    report: dict | None = None

    if not args.skip_training:
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
            print("[main] Training selesai.")
            print(f"[main] Epoch terakhir : {report.get('epoch')}")
            print(f"[main] Loss           : {report.get('loss'):.6f}")
            print(f"[main] Perplexity     : {report.get('perplexity'):.4f}")
            print(f"[main] Accuracy       : {report.get('accuracy'):.4f}")
            print(f"[main] Valid tokens   : {report.get('num_valid_tokens')}")
    else:
        print("[main] Training dilewati karena --skip-training.")

    # ------------------------------------------------------------------------
    # 7. EVALUATION / AUDIT
    # ------------------------------------------------------------------------
    print_section("TAHAP 6: EVALUATOR AUDIT")

    if not args.skip_eval:
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
            print("[main] Pastikan folder datasets/ berisi data teks.")
    else:
        print("[main] Evaluasi dilewati karena --skip-eval.")

    # ------------------------------------------------------------------------
    # 8. PLAYGROUND / TESTING
    # ------------------------------------------------------------------------
    print_section("TAHAP 7: PLAYGROUND TESTING")

    run_testing = (not args.skip_test) or args.interactive

    if run_testing:
        # Trainer diteruskan ke playground agar perintah 'training' bisa dipakai
        playground = Playground(
            model=model,
            tokenizer=tokenizer,
            config=config,
            trainer=trainer,
        )

        if args.interactive:
            playground.interactive(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                min_new_tokens=args.min_new_tokens,
            )
        else:
            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                min_new_tokens=args.min_new_tokens,
                verbose=True,
            )
    else:
        print("[main] Testing dilewati karena --skip-test.")

    # ------------------------------------------------------------------------
    # 9. EXPORT MODEL
    # ------------------------------------------------------------------------
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

    Tujuan:
    - memastikan seluruh pipeline bisa berjalan end-to-end
    - tidak mengubah dataset utama
    - tidak mengubah config utama
    """
    print_section("MODE DEMO")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "\n".join(
                [
                    "mobil memiliki mesin",
                    "mobil memakai bahan bakar",
                    "kecepatan mobil tergantung mesin",
                    "apa itu mobil",
                    "mesin mobil perlu dirawat",
                    "bahan bakar mempengaruhi performa mobil",
                    "performa mobil tergantung efisiensi mesin",
                ]
            ),
            encoding="utf-8",
        )

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
        )

        demo_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=1,
                sequence_length=8,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
            ),
            model=ModelConfig(
                vocab_size=96,
                embedding_dim=32,
                num_attention_heads=2,
                num_layers=1,
                dropout_rate=0.0,
                max_position_embeddings=64,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float16",
            ),
        )

        demo_args = copy.copy(args)

        # Demo tidak perlu resume dari checkpoint lama.
        demo_args.resume = False

        # Demo memakai path export sementara.
        demo_args.export_path = None

        # Batasi step jika user tidak menentukan.
        if demo_args.max_steps is None:
            demo_args.max_steps = 4

        # Batasi batch evaluasi agar demo cepat.
        if demo_args.eval_batches != 0:
            demo_args.eval_batches = 2

        print("[main] Menjalankan demo pipeline dengan folder sementara:")
        print(f"[main] {tmp_path}")
        print()

        run_pipeline(demo_config, demo_args)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CreateMiniBot Pipeline",
        description=(
            "Orkestrator pipeline training: tokenizer, trainer, evaluator, "
            "playground, dan model exporter."
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
        "--skip-training",
        action="store_true",
        help="Lewati tahap training.",
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
        "--interactive",
        action="store_true",
        help="Jalankan playground dalam mode CLI interaktif.",
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
        default=32,
        help="Jumlah maksimum token baru saat testing.",
    )

    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=8,
        help="Jumlah minimum token sebelum EOS diperbolehkan.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Temperature sampling untuk playground.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K sampling untuk playground.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.92,
        help="Top-P sampling untuk playground.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.25,
        help="Repetition penalty. Nilai > 1 mengurangi pengulangan.",
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
