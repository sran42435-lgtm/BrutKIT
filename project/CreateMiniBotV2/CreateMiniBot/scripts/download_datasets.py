#!/usr/bin/env python3
# scripts/download_datasets.py
#
# Download QA datasets untuk training model presisi:
# - SQuAD 2.0 (English, 150K samples)
# - TyDi QA Indonesian (20K samples)
# - IndoQA (Indonesian native, 10K samples)
#
# Output: datasets/{squad,tydi,indoqa}/

import json
import os
import sys
import urllib.request
from pathlib import Path
import hashlib

# Target directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Dataset URLs
DATASETS = {
    "squad": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    },
    "tydi": {
        "train": "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-goldp-v1.0-train.json",
        "dev": "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-goldp-v1.0-dev.json",
    },
    "indoqa": {
        "train": "https://raw.githubusercontent.com/Indonesian-NLP/IndoQA/main/data/indoqa_train.json",
        "dev": "https://raw.githubusercontent.com/Indonesian-NLP/IndoQA/main/data/indoqa_dev.json",
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download file dengan progress indicator."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return True
    
    print(f"  ↓ Downloading: {url}")
    try:
        # Progress callback
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
                print(f"\r  [{bar}] {percent:3d}% ({downloaded//1024}KB/{total_size//1024}KB)", end="")
        
        urllib.request.urlretrieve(url, dest, progress_hook)
        print()  # New line after progress
        print(f"  ✓ Downloaded: {dest.name}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return False


def download_squad() -> bool:
    """Download SQuAD 2.0 dataset."""
    print("\n[1/3] SQuAD 2.0 (English, ~150K samples)")
    squad_dir = DATASETS_DIR / "squad"
    squad_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    for split, url in DATASETS["squad"].items():
        filename = f"{split}-v2.0.json"
        dest = squad_dir / filename
        success = success and download_file(url, dest)
    
    if success:
        # Verify & count
        train_file = squad_dir / "train-v2.0.json"
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_qa = sum(len(article["paragraphs"]) for article in data["data"])
                total_q = sum(
                    len(p["qas"]) 
                    for article in data["data"] 
                    for p in article["paragraphs"]
                )
                print(f"  📊 Stats: {len(data['data'])} articles, {total_qa} paragraphs, {total_q} questions")
    
    return success


def download_tydi() -> bool:
    """Download TyDi QA (multilingual, termasuk Indonesian)."""
    print("\n[2/3] TyDi QA GoldP (Multilingual + Indonesian)")
    tydi_dir = DATASETS_DIR / "tydi"
    tydi_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    for split, url in DATASETS["tydi"].items():
        filename = f"tydiqa-goldp-v1.0-{split}.json"
        dest = tydi_dir / filename
        success = success and download_file(url, dest)
    
    if success:
        # Extract Indonesian subset
        train_file = tydi_dir / "tydiqa-goldp-v1.0-train.json"
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Filter Indonesian only
            indonesian_data = [
                article for article in data["data"]
                if article.get("language") == "indonesian" or 
                   any(p.get("language") == "indonesian" for p in article.get("paragraphs", []))
            ]
            
            # Save Indonesian subset
            id_subset_file = tydi_dir / "indonesian_train.json"
            id_data = {
                "version": data.get("version", "1.0"),
                "language": "indonesian",
                "data": indonesian_data,
            }
            with open(id_subset_file, "w", encoding="utf-8") as f:
                json.dump(id_data, f, ensure_ascii=False, indent=2)
            
            total_q = sum(
                len(p["qas"]) 
                for article in indonesian_data 
                for p in article.get("paragraphs", [])
            )
            print(f"  📊 Indonesian subset: {len(indonesian_data)} articles, {total_q} questions")
    
    return success


def download_indoqa() -> bool:
    """Download IndoQA (Indonesian native)."""
    print("\n[3/3] IndoQA (Indonesian Native)")
    indoqa_dir = DATASETS_DIR / "indoqa"
    indoqa_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    for split, url in DATASETS["indoqa"].items():
        filename = f"indoqa_{split}.json"
        dest = indoqa_dir / filename
        success = success and download_file(url, dest)
    
    if success:
        # Verify & count
        train_file = indoqa_dir / "indoqa_train.json"
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # IndoQA format bisa berbeda, detect structure
                if isinstance(data, list):
                    total_q = sum(len(item.get("questions", [])) for item in data)
                    print(f"  📊 Stats: {len(data)} articles, {total_q} questions")
                elif isinstance(data, dict) and "data" in data:
                    total_q = sum(
                        len(p.get("qas", []))
                        for article in data["data"]
                        for p in article.get("paragraphs", [])
                    )
                    print(f"  📊 Stats: {len(data['data'])} articles, {total_q} questions")
    
    return success


def main():
    print("=" * 70)
    print("Dataset Downloader - QA Datasets untuk Constrained Generation")
    print("=" * 70)
    print(f"\nTarget directory: {DATASETS_DIR}")
    print(f"Expected total download: ~200MB")
    print(f"Expected time: 5-15 menit (tergantung koneksi)")
    
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        "SQuAD 2.0": download_squad(),
        "TyDi QA": download_tydi(),
        "IndoQA": download_indoqa(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name:20s} {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✅ Semua dataset berhasil didownload!")
        print(f"\nNext step: Jalankan 'python scripts/prepare_datasets.py'")
        print("untuk merge dan convert ke unified format.")
    else:
        print("\n⚠️  Beberapa dataset gagal. Coba download manual atau retry.")
        print("Failed datasets:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
