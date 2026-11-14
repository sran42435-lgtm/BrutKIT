#!/usr/bin/env python3
# scripts/convert_dataset.py
#
# Dataset Converter untuk RAG (Retrieval-Augmented Generation).
#
# Tugas utama:
# 1. Load semua paragraf dari datasets/*.txt
# 2. Strip domain prefix ([MOBIL], [SAINS], dll) - tidak diperlukan di RAG
# 3. Generate QA pairs dari setiap paragraf menggunakan template
# 4. Augmentasi data (shuffle kalimat, truncate, paraphrase template)
# 5. Multi-context sampling (gabungkan beberapa paragraf)
# 6. Save ke file JSON (qa_dataset.json)
#
# Output format:
# {
#   "version": "1.0",
#   "total_samples": N,
#   "metadata": {...},
#   "samples": [
#     {
#       "context": "paragraf konteks",
#       "question": "pertanyaan",
#       "answer": "jawaban",
#       "source_file": "mobil.txt",
#       "template_type": "definition",
#       "augmentation": "original"
#     },
#     ...
#   ]
# }
#
# Cara pakai:
#   python scripts/convert_dataset.py
#   python scripts/convert_dataset.py --preview 5
#   python scripts/convert_dataset.py --output my_qa.json
#   python scripts/convert_dataset.py --augmentation-factor 5

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CONFIG, Config


# ============================================================================
# QA TEMPLATES
# ============================================================================
# Setiap template punya:
# - type: kategori pertanyaan (untuk tracking)
# - question_template: format string untuk pertanyaan
# - answer_template: format string untuk jawaban

QA_TEMPLATES = [
    # ============================================================
    # DEFINISI
    # ============================================================
    {
        "type": "definition",
        "question_template": "Apa itu {subject}?",
        "answer_template": "{paragraph}",
    },
    {
        "type": "definition",
        "question_template": "Apa yang dimaksud dengan {subject}?",
        "answer_template": "{paragraph}",
    },
    {
        "type": "definition",
        "question_template": "Jelaskan pengertian {subject}.",
        "answer_template": "{paragraph}",
    },
    {
        "type": "definition",
        "question_template": "Berikan definisi dari {subject}.",
        "answer_template": "{paragraph}",
    },
    
    # ============================================================
    # FUNGSI / KEGUNAAN
    # ============================================================
    {
        "type": "function",
        "question_template": "Apa fungsi {subject}?",
        "answer_template": "Berdasarkan informasi: {paragraph}",
    },
    {
        "type": "function",
        "question_template": "Untuk apa {subject} digunakan?",
        "answer_template": "{paragraph}",
    },
    {
        "type": "function",
        "question_template": "Apa kegunaan {subject}?",
        "answer_template": "{paragraph}",
    },
    
    # ============================================================
    # PENJELASAN UMUM
    # ============================================================
    {
        "type": "explanation",
        "question_template": "Jelaskan tentang {subject}.",
        "answer_template": "{paragraph}",
    },
    {
        "type": "explanation",
        "question_template": "Ceritakan mengenai {subject}.",
        "answer_template": "{paragraph}",
    },
    {
        "type": "explanation",
        "question_template": "Bagaimana dengan {subject}?",
        "answer_template": "{paragraph}",
    },
    {
        "type": "explanation",
        "question_template": "Apa yang kamu ketahui tentang {subject}?",
        "answer_template": "Berdasarkan informasi yang tersedia: {paragraph}",
    },
    
    # ============================================================
    # PROSES / CARA KERJA
    # ============================================================
    {
        "type": "process",
        "question_template": "Bagaimana cara kerja {subject}?",
        "answer_template": "{paragraph}",
    },
    {
        "type": "process",
        "question_template": "Bagaimana proses {subject}?",
        "answer_template": "{paragraph}",
    },
    
    # ============================================================
    # INFORMASI UMUM
    # ============================================================
    {
        "type": "information",
        "question_template": "Berikan informasi tentang {subject}.",
        "answer_template": "{paragraph}",
    },
    {
        "type": "information",
        "question_template": "Apa yang perlu diketahui tentang {subject}?",
        "answer_template": "{paragraph}",
    },
]


# ============================================================================
# FALLBACK ANSWER (untuk context dropout training)
# ============================================================================
NO_CONTEXT_ANSWER = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut."


# ============================================================================
# DOMAIN PREFIX PATTERN
# ============================================================================
DOMAIN_PREFIX_PATTERN = re.compile(r'^\[([A-Z_]+)\]\s*')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def strip_domain_prefix(text: str) -> str:
    """
    Hapus domain prefix dari teks.
    
    Contoh:
        "[MOBIL] Mobil adalah..." -> "Mobil adalah..."
        "Teks biasa" -> "Teks biasa"
    """
    return DOMAIN_PREFIX_PATTERN.sub('', text).strip()


def extract_subject(paragraph: str) -> str:
    """
    Ekstrak subjek dari paragraf.
    
    Strategi:
    1. Ambil 3-5 kata pertama (setelah strip prefix)
    2. Capitalize untuk membuat subject yang rapi
    
    Contoh:
        "Mobil adalah kendaraan beroda empat." -> "Mobil"
        "Fotosintesis adalah proses pembuatan makanan." -> "Fotosintesis"
        "Sistem rem mobil berfungsi untuk..." -> "Sistem rem mobil"
    """
    # Strip prefix dulu
    clean_text = strip_domain_prefix(paragraph)
    
    if not clean_text:
        return "topik ini"
    
    # Ambil kalimat pertama
    first_sentence = re.split(r'[.!?]', clean_text)[0].strip()
    
    # Ambil subjek: kata-kata sebelum kata kerja umum (adalah, merupakan, dll)
    verb_patterns = [
        r'^(.+?)\s+(adalah|merupakan|yaitu|ialah|berfungsi|digunakan|memiliki)\b',
        r'^(.+?)\s+(terdiri|berasal|berada|memiliki|memerlukan)\b',
    ]
    
    for pattern in verb_patterns:
        match = re.match(pattern, first_sentence, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            # Ambil maksimal 4 kata
            words = subject.split()[:4]
            return ' '.join(words)
    
    # Fallback: ambil 2-3 kata pertama
    words = first_sentence.split()[:3]
    if not words:
        return "topik ini"
    
    return ' '.join(words)


def split_sentences(text: str) -> List[str]:
    """
    Split teks menjadi kalimat-kalimat.
    
    Menggunakan pattern: titik/tanda tanya/tanda seru diikuti spasi atau akhir string.
    """
    # Pattern: [.!?] diikuti spasi + huruf kapital, atau akhir string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Filter empty sentences dan strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Jika tidak ada split, kembalikan teks asli sebagai 1 kalimat
    if not sentences:
        return [text.strip()]
    
    return sentences


def shuffle_sentences(text: str) -> str:
    """
    Shuffle kalimat dalam paragraf (untuk augmentasi).
    
    Digunakan untuk:
    - Mencegah model menghafal urutan kata
    - Memaksa model benar-benar membaca konteks
    """
    sentences = split_sentences(text)
    
    if len(sentences) <= 1:
        return text  # Tidak bisa shuffle jika hanya 1 kalimat
    
    random.shuffle(sentences)
    return ' '.join(sentences)


def truncate_paragraph(text: str, ratio: float = 0.5) -> str:
    """
    Truncate paragraf menjadi sebagian saja.
    
    Digunakan untuk:
    - Melatih model handle incomplete context
    - Mencegah model bergantung pada context yang terlalu panjang
    """
    sentences = split_sentences(text)
    
    if len(sentences) <= 1:
        # Jika hanya 1 kalimat, truncate karakter
        mid = len(text) // 2
        return text[:mid].strip()
    
    # Ambil sebagian kalimat
    n_keep = max(1, int(len(sentences) * ratio))
    kept = sentences[:n_keep]
    
    return ' '.join(kept)


def clean_paragraph(text: str) -> str:
    """
    Bersihkan paragraf:
    - Strip domain prefix
    - Normalisasi whitespace
    - Hapus newline
    """
    text = strip_domain_prefix(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================================
# DATASET CONVERTER
# ============================================================================

class DatasetConverter:
    """
    Konversi dataset mentah ke format QA untuk RAG training.
    """
    
    def __init__(
        self,
        config: Config = CONFIG,
        augmentation_factor: int = 3,
        include_multi_context: bool = True,
        include_no_context: bool = True,
        seed: int = 1337,
    ):
        self.config = config
        self.augmentation_factor = augmentation_factor
        self.include_multi_context = include_multi_context
        self.include_no_context = include_no_context
        self.rng = random.Random(seed)
        
        # State
        self.corpus: List[Dict] = []  # list of {text, source_file}
        self.qa_samples: List[Dict] = []
    
    # ========================================================================
    # CORPUS LOADING
    # ========================================================================
    
    def load_corpus(self, datasets_dir: Optional[Path] = None) -> None:
        """
        Load semua paragraf dari datasets/*.txt
        
        Args:
            datasets_dir: path ke folder datasets (default dari config)
        """
        if datasets_dir is None:
            datasets_dir = Path(self.config.paths.datasets_dir)
        else:
            datasets_dir = Path(datasets_dir)
        
        if not datasets_dir.exists():
            raise FileNotFoundError(f"Datasets directory tidak ditemukan: {datasets_dir}")
        
        self.corpus = []
        
        # Load semua file .txt
        txt_files = sorted(datasets_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"[Converter] ⚠️ Tidak ada file .txt di {datasets_dir}")
            return
        
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split by double newline (paragraph separator)
            raw_paragraphs = content.split("\n\n")
            
            for para in raw_paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                self.corpus.append({
                    "text": para,
                    "source_file": txt_file.name,
                })
        
        print(f"[Converter] Loaded {len(self.corpus)} paragraphs dari {len(txt_files)} file")
    
    # ========================================================================
    # QA GENERATION
    # ========================================================================
    
    def generate_qa_from_paragraph(self, paragraph: str, source_file: str) -> List[Dict]:
        """
        Generate multiple QA pairs dari satu paragraf.
        
        Args:
            paragraph: paragraf mentah (dengan/ tanpa prefix)
            source_file: nama file asal
        
        Returns:
            List of QA samples
        """
        # Clean paragraph
        clean_para = clean_paragraph(paragraph)
        if not clean_para:
            return []
        
        # Extract subject
        subject = extract_subject(paragraph)
        
        samples = []
        
        # Pilih subset templates (random) untuk variasi
        n_templates = min(3, len(QA_TEMPLATES))  # max 3 template per paragraph
        selected_templates = self.rng.sample(QA_TEMPLATES, n_templates)
        
        for template in selected_templates:
            try:
                question = template["question_template"].format(subject=subject)
                answer = template["answer_template"].format(
                    paragraph=clean_para,
                    subject=subject,
                )
                
                samples.append({
                    "context": clean_para,
                    "question": question,
                    "answer": answer,
                    "source_file": source_file,
                    "template_type": template["type"],
                    "augmentation": "original",
                })
            except KeyError as e:
                # Skip template yang tidak bisa di-format
                continue
        
        return samples
    
    def augment_sample(self, sample: Dict) -> List[Dict]:
        """
        Augmentasi satu QA sample dengan variasi:
        1. Shuffle kalimat konteks
        2. Truncate konteks
        3. Variasi template (paraphrase)
        
        Args:
            sample: QA sample original
        
        Returns:
            List of augmented samples
        """
        augmented = []
        context = sample["context"]
        
        # ============================================================
        # Augmentasi 1: Shuffle kalimat
        # ============================================================
        if len(split_sentences(context)) > 1:
            shuffled_context = shuffle_sentences(context)
            augmented.append({
                **sample,
                "context": shuffled_context,
                "augmentation": "shuffled",
            })
        
        # ============================================================
        # Augmentasi 2: Truncate konteks (50%)
        # ============================================================
        truncated_context = truncate_paragraph(context, ratio=0.5)
        if truncated_context != context:
            augmented.append({
                **sample,
                "context": truncated_context,
                "augmentation": "truncated",
            })
        
        # ============================================================
        # Augmentasi 3: Paraphrase template (random variasi lain)
        # ============================================================
        same_type_templates = [
            t for t in QA_TEMPLATES
            if t["type"] == sample["template_type"]
            and t["question_template"] != sample.get("_original_q_template")
        ]
        
        if same_type_templates:
            alt_template = self.rng.choice(same_type_templates)
            subject = extract_subject(sample["context"])
            try:
                alt_question = alt_template["question_template"].format(subject=subject)
                augmented.append({
                    **sample,
                    "question": alt_question,
                    "augmentation": "paraphrased",
                })
            except KeyError:
                pass
        
        return augmented
    
    def generate_multi_context_samples(self) -> List[Dict]:
        """
        Generate samples dengan multi-context (gabungan beberapa paragraf).
        
        Strategi:
        - Ambil 2-3 paragraf dari file yang sama
        - Gabungkan sebagai konteks
        - Gunakan salah satu paragraf sebagai basis Q&A
        
        Returns:
            List of multi-context QA samples
        """
        if not self.include_multi_context:
            return []
        
        # Kelompokkan paragraf per file
        by_file: Dict[str, List[Dict]] = defaultdict(list)
        for item in self.corpus:
            by_file[item["source_file"]].append(item)
        
        samples = []
        
        for file_name, paragraphs in by_file.items():
            if len(paragraphs) < 2:
                continue  # Butuh minimal 2 paragraf untuk multi-context
            
            # Generate beberapa multi-context samples per file
            n_samples = min(2, len(paragraphs) - 1)
            
            for _ in range(n_samples):
                # Ambil 2-3 paragraf random
                k = self.rng.choice([2, 3]) if len(paragraphs) >= 3 else 2
                selected = self.rng.sample(paragraphs, k)
                
                # Clean dan gabungkan
                clean_paragraphs = [clean_paragraph(p["text"]) for p in selected]
                combined_context = "\n\n".join(clean_paragraphs)
                
                # Pilih satu paragraf sebagai basis Q&A
                base_para = self.rng.choice(selected)
                base_clean = clean_paragraph(base_para["text"])
                subject = extract_subject(base_para["text"])
                
                # Pilih template random
                template = self.rng.choice(QA_TEMPLATES)
                
                try:
                    question = template["question_template"].format(subject=subject)
                    answer = template["answer_template"].format(
                        paragraph=base_clean,
                        subject=subject,
                    )
                    
                    samples.append({
                        "context": combined_context,
                        "question": question,
                        "answer": answer,
                        "source_file": file_name,
                        "template_type": f"{template['type']}_multi",
                        "augmentation": "multi_context",
                    })
                except KeyError:
                    continue
        
        return samples
    
    def generate_no_context_samples(self, n_samples: int = 5) -> List[Dict]:
        """
        Generate samples tanpa konteks (untuk context dropout training).
        
        Ini mengajarkan model untuk:
        - Berkata "tidak tahu" jika tidak ada informasi
        - Tidak hallucinate
        
        Args:
            n_samples: jumlah samples yang dibuat
        
        Returns:
            List of no-context samples
        """
        if not self.include_no_context or not self.corpus:
            return []
        
        samples = []
        selected = self.rng.sample(
            self.corpus,
            min(n_samples, len(self.corpus))
        )
        
        for item in selected:
            subject = extract_subject(item["text"])
            template = self.rng.choice(QA_TEMPLATES)
            
            try:
                question = template["question_template"].format(subject=subject)
                
                samples.append({
                    "context": "",  # Kosong
                    "question": question,
                    "answer": NO_CONTEXT_ANSWER,
                    "source_file": item["source_file"],
                    "template_type": "no_context",
                    "augmentation": "no_context",
                })
            except KeyError:
                continue
        
        return samples
    
    # ========================================================================
    # MAIN CONVERSION
    # ========================================================================
    
    def convert_all(self) -> List[Dict]:
        """
        Konversi seluruh corpus ke QA samples dengan augmentasi.
        
        Returns:
            List of all QA samples
        """
        if not self.corpus:
            print("[Converter] ⚠️ Corpus kosong. Panggil load_corpus() terlebih dahulu.")
            return []
        
        print(f"[Converter] Generating QA samples...")
        
        all_samples = []
        
        # ============================================================
        # 1. Single-context samples (original + augmented)
        # ============================================================
        for item in self.corpus:
            # Generate QA pairs original
            original_samples = self.generate_qa_from_paragraph(
                item["text"],
                item["source_file"],
            )
            
            all_samples.extend(original_samples)
            
            # Augmentasi
            if self.augmentation_factor > 1:
                for sample in original_samples:
                    augmented = self.augment_sample(sample)
                    # Ambil subset dari augmented (sesuai factor)
                    n_aug = min(self.augmentation_factor - 1, len(augmented))
                    if n_aug > 0:
                        all_samples.extend(self.rng.sample(augmented, n_aug))
        
        # ============================================================
        # 2. Multi-context samples
        # ============================================================
        multi_samples = self.generate_multi_context_samples()
        all_samples.extend(multi_samples)
        
        # ============================================================
        # 3. No-context samples (untuk context dropout)
        # ============================================================
        no_ctx_samples = self.generate_no_context_samples(n_samples=10)
        all_samples.extend(no_ctx_samples)
        
        # ============================================================
        # Shuffle final samples
        # ============================================================
        self.rng.shuffle(all_samples)
        
        self.qa_samples = all_samples
        
        # Print statistics
        self._print_statistics()
        
        return all_samples
    
    def _print_statistics(self) -> None:
        """Print statistik hasil konversi."""
        if not self.qa_samples:
            return
        
        print(f"\n[Converter] ✅ Total samples: {len(self.qa_samples)}")
        
        # Hitung per template type
        type_counts = defaultdict(int)
        aug_counts = defaultdict(int)
        
        for sample in self.qa_samples:
            type_counts[sample["template_type"]] += 1
            aug_counts[sample["augmentation"]] += 1
        
        print(f"\n  Per template type:")
        for t_type, count in sorted(type_counts.items()):
            print(f"    - {t_type:20s}: {count:4d} samples")
        
        print(f"\n  Per augmentation type:")
        for aug, count in sorted(aug_counts.items()):
            print(f"    - {aug:20s}: {count:4d} samples")
    
    # ========================================================================
    # SAVE / LOAD
    # ========================================================================
    
    def save_dataset(self, output_path: Optional[Path] = None) -> Path:
        """
        Save QA dataset ke file JSON.
        
        Args:
            output_path: path output (default dari config)
        
        Returns:
            Path ke file yang dibuat
        """
        if output_path is None:
            output_path = Path(self.config.paths.qa_dataset_path)
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Hitung stats
        type_counts = defaultdict(int)
        for sample in self.qa_samples:
            type_counts[sample["template_type"]] += 1
        
        data = {
            "version": "1.0",
            "architecture": "rag-single-head",
            "metadata": {
                "total_samples": len(self.qa_samples),
                "source_corpus_size": len(self.corpus),
                "augmentation_factor": self.augmentation_factor,
                "template_types": dict(type_counts),
            },
            "samples": self.qa_samples,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[Converter] 💾 Dataset disimpan ke: {output_path}")
        print(f"[Converter]    File size: {output_path.stat().st_size:,} bytes")
        
        return output_path
    
    def load_dataset(self, input_path: Optional[Path] = None) -> List[Dict]:
        """
        Load QA dataset dari file JSON.
        
        Args:
            input_path: path input (default dari config)
        
        Returns:
            List of QA samples
        """
        if input_path is None:
            input_path = Path(self.config.paths.qa_dataset_path)
        else:
            input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {input_path}")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.qa_samples = data.get("samples", [])
        
        print(f"[Converter] Loaded {len(self.qa_samples)} samples dari {input_path}")
        
        return self.qa_samples
    
    # ========================================================================
    # PREVIEW
    # ========================================================================
    
    def preview_samples(self, n: int = 5) -> None:
        """
        Print preview dari N samples pertama.
        
        Args:
            n: jumlah samples yang ditampilkan
        """
        if not self.qa_samples:
            print("[Converter] ⚠️ Tidak ada samples untuk di-preview.")
            return
        
        print(f"\n[Converter] Preview {min(n, len(self.qa_samples))} samples:")
        print("=" * 80)
        
        for i, sample in enumerate(self.qa_samples[:n], 1):
            print(f"\n--- Sample {i} ---")
            print(f"Source     : {sample['source_file']}")
            print(f"Type       : {sample['template_type']}")
            print(f"Augment    : {sample['augmentation']}")
            print(f"\nKonteks    : {sample['context'][:200]}{'...' if len(sample['context']) > 200 else ''}")
            print(f"\nPertanyaan : {sample['question']}")
            print(f"\nJawaban    : {sample['answer'][:200]}{'...' if len(sample['answer']) > 200 else ''}")
        
        print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Dataset Converter",
        description="Konversi dataset mentah ke format QA untuk RAG training.",
    )
    
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Path ke folder datasets (default: dari config)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path output file JSON (default: dari config)",
    )
    
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=3,
        help="Jumlah variasi augmentasi per sample (default: 3)",
    )
    
    parser.add_argument(
        "--no-multi-context",
        action="store_true",
        help="Disable multi-context sampling",
    )
    
    parser.add_argument(
        "--no-context-dropout",
        action="store_true",
        help="Disable no-context samples",
    )
    
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Tampilkan preview N samples (default: 0 = tidak preview)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (default: 1337)",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 80)
    print("Dataset Converter - RAG Training Data Generator")
    print("=" * 80)
    
    # Create converter
    converter = DatasetConverter(
        config=CONFIG,
        augmentation_factor=args.augmentation_factor,
        include_multi_context=not args.no_multi_context,
        include_no_context=not args.no_context_dropout,
        seed=args.seed,
    )
    
    # Load corpus
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else None
    converter.load_corpus(datasets_dir)
    
    if not converter.corpus:
        print("\n[Converter] ❌ Tidak ada data yang bisa dikonversi.")
        sys.exit(1)
    
    # Convert
    converter.convert_all()
    
    # Preview
    if args.preview > 0:
        converter.preview_samples(args.preview)
    
    # Save
    output_path = Path(args.output) if args.output else None
    converter.save_dataset(output_path)
    
    print("\n" + "=" * 80)
    print("✅ Konversi selesai!")
    print("=" * 80)
    print("\nLangkah selanjutnya:")
    print("  1. Review dataset: python scripts/convert_dataset.py --preview 10")
    print("  2. Train model: python main.py --interactive")
    print("     Lalu: /training --epochs 50")


if __name__ == "__main__":
    main()
