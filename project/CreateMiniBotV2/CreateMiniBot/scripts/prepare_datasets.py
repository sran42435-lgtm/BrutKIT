#!/usr/bin/env python3
# scripts/prepare_datasets.py
#
# Merge & convert QA datasets ke unified format untuk training.
# 
# Supported datasets:
# - SQuAD 2.0 (English, wajib ada)
# - TyDi QA (Indonesian, optional)
# - IndoQA (Indonesian, optional)
# - Custom dataset user (Indonesian, optional)
#
# Output: datasets/merged/qa_dataset.json (unified format)

import json
import random
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MERGED_DIR = DATASETS_DIR / "merged"

# Unified format:
# {
#   "context": str,       # Paragraf sumber
#   "question": str,      # Pertanyaan
#   "answer": str,        # Jawaban (bisa kosong untuk impossible)
#   "source": str,        # squad/tydi/indoqa/custom
#   "language": str,      # en/id
#   "is_impossible": bool # True jika pertanyaan tidak bisa dijawab
# }


def clean_text(text: str) -> str:
    """Clean dan normalize text."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix common issues
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text.strip()


def process_squad(squad_dir: Path) -> List[Dict]:
    """
    Process SQuAD 2.0 ke unified format.
    
    SQuAD format:
    {
      "data": [
        {
          "title": "...",
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "question": "...",
                  "answers": [{"text": "...", "answer_start": 123}],
                  "is_impossible": false
                }
              ]
            }
          ]
        }
      ]
    }
    """
    print("\n[Processing] SQuAD 2.0")
    
    samples = []
    skipped_impossible = 0
    skipped_no_answer = 0
    
    train_file = squad_dir / "train-v2.0.json"
    dev_file = squad_dir / "dev-v2.0.json"
    
    files_to_process = []
    if train_file.exists():
        files_to_process.append(("train", train_file))
    if dev_file.exists():
        files_to_process.append(("dev", dev_file))
    
    if not files_to_process:
        print("  ⚠️ No SQuAD files found")
        return []
    
    for split, filepath in files_to_process:
        print(f"  Processing {split}: {filepath.name}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for article in data["data"]:
            title = article.get("title", "")
            
            for paragraph in article["paragraphs"]:
                context = clean_text(paragraph.get("context", ""))
                if not context:
                    continue
                
                for qa in paragraph["qas"]:
                    question = clean_text(qa.get("question", ""))
                    if not question:
                        continue
                    
                    # Check if impossible (SQuAD 2.0 feature)
                    is_impossible = qa.get("is_impossible", False)
                    
                    if is_impossible:
                        skipped_impossible += 1
                        # Kita tetap include sebagai "no answer" training sample
                        # Berguna untuk model belajar bilang "tidak tahu"
                        samples.append({
                            "context": context,
                            "question": question,
                            "answer": "",
                            "source": "squad",
                            "language": "en",
                            "title": title,
                            "is_impossible": True,
                            "split": split,
                        })
                        continue
                    
                    # Get answer (use first answer if multiple)
                    answers = qa.get("answers", [])
                    if not answers:
                        skipped_no_answer += 1
                        continue
                    
                    answer = clean_text(answers[0].get("text", ""))
                    if not answer:
                        skipped_no_answer += 1
                        continue
                    
                    samples.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "source": "squad",
                        "language": "en",
                        "title": title,
                        "is_impossible": False,
                        "split": split,
                    })
    
    print(f"  ✅ SQuAD: {len(samples)} samples")
    print(f"     - Answerable: {len([s for s in samples if not s['is_impossible']])}")
    print(f"     - Impossible (skipped from impossible): {skipped_impossible}")
    print(f"     - No answer (skipped): {skipped_no_answer}")
    
    return samples


def process_tydi(tydi_dir: Path) -> List[Dict]:
    """Process TyDi QA (Indonesian subset)."""
    print("\n[Processing] TyDi QA (Indonesian)")
    
    samples = []
    
    # Try Indonesian subset first
    id_file = tydi_dir / "indonesian_train.json"
    if not id_file.exists():
        # Fallback ke original
        id_file = tydi_dir / "tydiqa-goldp-v1.0-train.json"
    
    if not id_file.exists():
        print("  ⚠️ TyDi files not found (skipping)")
        return []
    
    with open(id_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = clean_text(paragraph.get("context", ""))
            if not context:
                continue
            
            # Detect language
            language = paragraph.get("language", 
                       article.get("language", "unknown"))
            
            # Skip non-Indonesian jika ada
            if language not in ["indonesian", "id"]:
                continue
            
            for qa in paragraph.get("qas", []):
                question = clean_text(qa.get("question", ""))
                if not question:
                    continue
                
                answers = qa.get("answers", [])
                if not answers:
                    continue
                
                answer = clean_text(answers[0].get("text", ""))
                if not answer:
                    continue
                
                samples.append({
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "source": "tydi",
                    "language": "id",
                    "is_impossible": False,
                })
    
    print(f"  ✅ TyDi (Indonesian): {len(samples)} samples")
    return samples


def process_indoqa(indoqa_dir: Path) -> List[Dict]:
    """Process IndoQA dataset."""
    print("\n[Processing] IndoQA")
    
    samples = []
    
    train_file = indoqa_dir / "indoqa_train.json"
    if not train_file.exists():
        print("  ⚠️ IndoQA files not found (skipping)")
        return []
    
    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # IndoQA format bisa bervariasi
    # Format 1: list of articles
    if isinstance(data, list):
        for article in data:
            passage = clean_text(article.get("passage", article.get("context", "")))
            if not passage:
                continue
            
            for qa in article.get("questions", []):
                question = clean_text(qa.get("question", ""))
                answer = clean_text(qa.get("answer", ""))
                
                if question and answer:
                    samples.append({
                        "context": passage,
                        "question": question,
                        "answer": answer,
                        "source": "indoqa",
                        "language": "id",
                        "is_impossible": False,
                    })
    
    # Format 2: SQuAD-like
    elif isinstance(data, dict) and "data" in data:
        for article in data["data"]:
            for paragraph in article.get("paragraphs", []):
                context = clean_text(paragraph.get("context", ""))
                if not context:
                    continue
                
                for qa in paragraph.get("qas", []):
                    question = clean_text(qa.get("question", ""))
                    answers = qa.get("answers", [])
                    
                    if question and answers:
                        answer = clean_text(answers[0].get("text", ""))
                        if answer:
                            samples.append({
                                "context": context,
                                "question": question,
                                "answer": answer,
                                "source": "indoqa",
                                "language": "id",
                                "is_impossible": False,
                            })
    
    print(f"  ✅ IndoQA: {len(samples)} samples")
    return samples


def process_custom(custom_dir: Path) -> List[Dict]:
    """Process custom dataset user (file .txt di datasets/)."""
    print("\n[Processing] Custom dataset (user)")
    
    samples = []
    
    # Skip subdirectories (squad, tydi, indoqa, merged)
    skip_dirs = {"squad", "tydi", "indoqa", "merged"}
    
    for txt_file in custom_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newline
        paragraphs = [clean_text(p) for p in content.split("\n\n")]
        paragraphs = [p for p in paragraphs if p]
        
        # Generate simple QA pairs untuk setiap paragraph
        for para in paragraphs:
            # Strip domain prefix jika ada
            clean_para = re.sub(r'^\[[A-Z_]+\]\s*', '', para)
            
            if len(clean_para) < 20:
                continue
            
            # Extract subject (first few words)
            words = clean_para.split()[:3]
            subject = " ".join(words)
            
            # Generate multiple question variants
            question_templates = [
                f"Apa itu {subject}?",
                f"Jelaskan tentang {subject}.",
                f"Apa yang dimaksud dengan {subject}?",
                f"Berikan informasi tentang {subject}.",
            ]
            
            for q in question_templates:
                samples.append({
                    "context": clean_para,
                    "question": q,
                    "answer": clean_para,
                    "source": "custom",
                    "language": "id",
                    "is_impossible": False,
                })
    
    print(f"  ✅ Custom: {len(samples)} samples (dari {len(list(custom_dir.glob('*.txt')))} files)")
    return samples


def remove_duplicates(samples: List[Dict]) -> List[Dict]:
    """Remove duplicate samples berdasarkan (question, answer)."""
    seen = set()
    unique = []
    
    for sample in samples:
        key = (
            sample["question"].lower().strip(),
            sample["answer"].lower().strip()[:100],  # first 100 chars
        )
        
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    
    removed = len(samples) - len(unique)
    print(f"\n[Deduplication] Removed {removed} duplicates")
    print(f"  Before: {len(samples)} samples")
    print(f"  After:  {len(unique)} samples")
    
    return unique


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 1337,
) -> Dict[str, List[Dict]]:
    """Split dataset ke train/val/test."""
    print(f"\n[Split] train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    
    random.seed(seed)
    samples = list(samples)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }
    
    for split_name, split_samples in splits.items():
        print(f"  {split_name:6s}: {len(split_samples):,} samples")
    
    return splits


def generate_statistics(samples: List[Dict]) -> Dict:
    """Generate statistics dari dataset."""
    stats = {
        "total": len(samples),
        "by_source": Counter(s["source"] for s in samples),
        "by_language": Counter(s["language"] for s in samples),
        "impossible": sum(1 for s in samples if s.get("is_impossible", False)),
        "avg_context_length": sum(len(s["context"].split()) for s in samples) / len(samples),
        "avg_question_length": sum(len(s["question"].split()) for s in samples) / len(samples),
        "avg_answer_length": sum(len(s["answer"].split()) for s in samples if s["answer"]) / max(1, len([s for s in samples if s["answer"]])),
    }
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics yang human-readable."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Total samples: {stats['total']:,}")
    
    print(f"\n📚 By source:")
    for source, count in sorted(stats["by_source"].items()):
        print(f"  - {source:15s}: {count:>8,} ({count/stats['total']:.1%})")
    
    print(f"\n🌐 By language:")
    for lang, count in sorted(stats["by_language"].items()):
        lang_name = {"en": "English", "id": "Indonesian"}.get(lang, lang)
        print(f"  - {lang_name:15s}: {count:>8,} ({count/stats['total']:.1%})")
    
    print(f"\n📏 Length statistics:")
    print(f"  - Avg context length:  {stats['avg_context_length']:.1f} words")
    print(f"  - Avg question length: {stats['avg_question_length']:.1f} words")
    print(f"  - Avg answer length:  {stats['avg_answer_length']:.1f} words")
    print(f"  - Impossible questions: {stats['impossible']:,}")
    
    print("=" * 70)


def main():
    print("=" * 70)
    print("Dataset Preparation - Merge & Convert ke Unified Format")
    print("=" * 70)
    
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all available datasets
    all_samples = []
    
    # SQuAD (wajib)
    squad_dir = DATASETS_DIR / "squad"
    if squad_dir.exists():
        all_samples.extend(process_squad(squad_dir))
    else:
        print("⚠️ SQuAD not found! Run download_datasets.py first.")
        return 1
    
    # TyDi (optional)
    tydi_dir = DATASETS_DIR / "tydi"
    if tydi_dir.exists():
        all_samples.extend(process_tydi(tydi_dir))
    
    # IndoQA (optional)
    indoqa_dir = DATASETS_DIR / "indoqa"
    if indoqa_dir.exists():
        all_samples.extend(process_indoqa(indoqa_dir))
    
    # Custom (optional)
    all_samples.extend(process_custom(DATASETS_DIR))
    
    if not all_samples:
        print("\n❌ No samples processed. Check dataset files.")
        return 1
    
    # Remove duplicates
    unique_samples = remove_duplicates(all_samples)
    
    # Split
    splits = split_dataset(unique_samples)
    
    # Generate statistics
    stats = generate_statistics(unique_samples)
    print_statistics(stats)
    
    # Save all splits
    print("\n[Save] Saving to disk...")
    
    for split_name, split_samples in splits.items():
        output_file = MERGED_DIR / f"qa_{split_name}.json"
        data = {
            "version": "1.0",
            "metadata": {
                "split": split_name,
                "total": len(split_samples),
                "stats": generate_statistics(split_samples),
            },
            "samples": split_samples,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Saved: {output_file.name} ({len(split_samples):,} samples)")
    
    # Save metadata summary
    summary_file = MERGED_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "total_samples": len(unique_samples),
            "splits": {k: len(v) for k, v in splits.items()},
            "statistics": stats,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset preparation selesai!")
    print(f"\n📁 Output directory: {MERGED_DIR}")
    print(f"\n🚀 Next step: Update config.py untuk model 20M params")
    print(f"   Lalu: Update core/architecture.py dengan constrained generation")
    print(f"   Lalu: Training!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
