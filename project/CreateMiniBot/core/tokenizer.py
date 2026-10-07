# core/tokenizer.py
#
# Tokenizer custom berbasis Byte-Pair Encoding (BPE) sederhana.
# Modul ini tidak menggunakan pustaka tokenizer eksternal.
#
# Tugas utama:
# 1. Membaca dataset dari folder datasets/
# 2. Membangun vocabulary
# 3. Menyimpan vocab.json
# 4. Encode teks -> token ID
# 5. Decode token ID -> teks
#
# Keterhubungan:
# - config.py      : membaca path, vocab size, dan special tokens
# - trainer.py     : menyiapkan input berbasis token ID
# - playground.py  : encode prompt dan decode output
# - model_exporter : menyediakan vocab untuk dikemas ke file model

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import CONFIG, Config


class Tokenizer:
    """
    Tokenizer custom dengan algoritma BPE sederhana.

    Desain:
    - Teks dinormalisasi: whitespace dirapatkan menjadi satu spasi.
    - Teks dipecah menjadi kata berdasarkan spasi.
    - Kata pertama dalam satu baris/sequence tidak memakai marker spasi.
    - Kata berikutnya diawali marker spasi: "▁"
    - BPE dipelajari dari frekuensi pasangan token dasar.
    - Vocabulary dipaksa sesuai config.model.vocab_size dengan menambahkan
      token [UNUSED_x] jika jumlah token hasil training belum mencukupi.
    """

    SPACE_MARKER = "▁"

    def __init__(self, config: Config = CONFIG):
        self.config = config

        self.datasets_dir: Path = config.paths.datasets_dir
        self.vocab_path: Path = config.paths.vocab_path
        self.target_vocab_size: int = int(config.model.vocab_size)

        self.pad_token: str = config.special_tokens.pad
        self.unk_token: str = config.special_tokens.unk
        self.bos_token: str = config.special_tokens.bos
        self.eos_token: str = config.special_tokens.eos

        self.pad_id: int = config.special_tokens.pad_id
        self.unk_id: int = config.special_tokens.unk_id
        self.bos_id: int = config.special_tokens.bos_id
        self.eos_id: int = config.special_tokens.eos_id

        self.special_ordered: Tuple[str, ...] = config.special_tokens.ordered_tokens
        self._special_set = frozenset(self.special_ordered)

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        self._ready: bool = False

    # ======================================================================
    # PUBLIC PROPERTIES
    # ======================================================================

    @property
    def vocab_size(self) -> int:
        """
        Jumlah token yang benar-benar tersedia di tokenizer.
        """
        return len(self.token_to_id)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ======================================================================
    # PUBLIC METHODS
    # ======================================================================

    def train(self) -> "Tokenizer":
        """
        Membaca dataset, membangun vocabulary dengan BPE, lalu menyimpan
        hasil ke vocab.json.
        """
        self._validate_target_vocab()

        word_freq, char_freq = self._build_corpus_statistics()
        token_to_id = self._build_initial_vocab(char_freq)

        # Proyeksikan korpus ke vocab awal. Karakter langka menjadi [UNK].
        word_freq = self._project_word_freq_to_vocab(word_freq, token_to_id)

        merges: List[Tuple[str, str]] = []
        self._learn_bpe(
            word_freq=word_freq,
            token_to_id=token_to_id,
            merges=merges,
        )

        token_to_id = self._pad_vocab(token_to_id)

        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.merges = merges
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._ready = True

        self.save()
        return self

    def load(self) -> "Tokenizer":
        """
        Memuat vocab.json jika sudah ada.
        """
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"File vocab tidak ditemukan di: {self.vocab_path}. "
                "Jalankan tokenizer.train() terlebih dahulu."
            )

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "token_to_id" in payload:
            token_to_id_raw = payload["token_to_id"]
            merges_raw = payload.get("merges", [])
        else:
            # Dukungan fallback bila vocab.json hanya berisi mapping token->id.
            token_to_id_raw = payload
            merges_raw = []

        token_to_id: Dict[str, int] = {
            str(token): int(idx) for token, idx in token_to_id_raw.items()
        }

        # Jika vocab lama lebih kecil dari target, tambahkan token unused
        # agar sesuai dengan config.model.vocab_size.
        if len(token_to_id) < self.target_vocab_size:
            token_to_id = self._pad_vocab(token_to_id)

        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

        merges: List[Tuple[str, str]] = []
        for pair in merges_raw:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and isinstance(pair[0], str)
                and isinstance(pair[1], str)
            ):
                merges.append((pair[0], pair[1]))

        self.merges = merges
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._ready = True

        self._validate_loaded_vocab()
        return self

    def save(self) -> None:
        """
        Menyimpan vocabulary dan merge rules ke vocab.json.
        """
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "model": "custom-bpe",
            "target_vocab_size": self.target_vocab_size,
            "space_marker": self.SPACE_MARKER,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "bos": self.bos_token,
                "eos": self.eos_token,
            },
            "token_to_id": self.token_to_id,
            "merges": [[a, b] for a, b in self.merges],
        }

        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    def get_vocab(self) -> Dict[str, int]:
        """
        Mengembalikan salinan vocabulary token->id.
        Dipakai oleh model_exporter.py saat mengemas metadata vocab.
        """
        self._ensure_ready()
        return dict(self.token_to_id)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Mengubah teks menjadi daftar token ID.

        Parameter:
        - add_bos: tambahkan [BOS] di awal
        - add_eos: tambahkan [EOS] di akhir
        """
        self._ensure_ready()

        tokens = self.encode_to_tokens(text)
        ids: List[int] = [
            self.token_to_id.get(token, self.unk_id) for token in tokens
        ]

        if add_bos:
            ids = [self.bos_id] + ids

        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def encode_to_tokens(self, text: str) -> List[str]:
        """
        Mengubah teks menjadi daftar token string sebelum dikonversi ke ID.
        Berguna untuk debugging.
        """
        self._ensure_ready()

        normalized = self._normalize_text(str(text))
        if not normalized:
            return []

        words = normalized.split(" ")
        all_tokens: List[str] = []

        for idx, word in enumerate(words):
            base_tokens: List[str] = []

            # Kata selain pertama diberi marker spasi agar decode bisa
            # mengembalikan pemisahan kata secara alami.
            if idx > 0:
                base_tokens.append(self.SPACE_MARKER)

            for ch in word:
                if self._is_valid_char(ch):
                    base_tokens.append(ch)

            # Jika hanya berisi marker spasi tanpa karakter valid, lewati.
            if not base_tokens or base_tokens == [self.SPACE_MARKER]:
                continue

            # Token dasar yang tidak ada di vocab dipetakan ke [UNK].
            base_tokens = [
                token if token in self.token_to_id else self.unk_token
                for token in base_tokens
            ]

            all_tokens.extend(self._apply_bpe(base_tokens))

        return all_tokens

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Mengubah daftar token ID kembali menjadi teks.
        """
        self._ensure_ready()

        pieces: List[str] = []

        for token_id in token_ids:
            try:
                tid = int(token_id)
            except (TypeError, ValueError):
                continue

            token = self.id_to_token.get(tid)
            if token is None:
                continue

            if skip_special_tokens:
                if token in self._special_set:
                    continue
                if token.startswith("[UNUSED_"):
                    continue

            pieces.append(token)

        text = "".join(pieces)
        text = text.replace(self.SPACE_MARKER, " ")

        # Bersihkan spasi berlebih akibat proses generation.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ======================================================================
    # INTERNAL: VALIDATION
    # ======================================================================

    def _validate_target_vocab(self) -> None:
        minimum = len(self.special_ordered) + 2  # special + marker + 1 token
        if self.target_vocab_size < minimum:
            raise ValueError(
                "config.model.vocab_size terlalu kecil. "
                f"Minimal {minimum} untuk special tokens, space marker, "
                "dan satu token dasar."
            )

    def _validate_loaded_vocab(self) -> None:
        for token in self.special_ordered:
            if token not in self.token_to_id:
                raise ValueError(
                    f"Special token {token} tidak ditemukan di vocab.json."
                )

        if self.SPACE_MARKER not in self.token_to_id:
            # Jika space marker tidak ada, tambahkan sebagai token baru.
            self.token_to_id[self.SPACE_MARKER] = len(self.token_to_id)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        if self.vocab_path.exists():
            self.load()
            return

        raise RuntimeError(
            "Tokenizer belum siap. Jalankan tokenizer.train() atau "
            "sediakan vocab.json terlebih dahulu."
        )

    # ======================================================================
    # INTERNAL: CORPUS READING
    # ======================================================================

    def _iter_corpus_texts(self) -> Iterable[str]:
        """
        Membaca file dataset dari folder datasets/.

        Format yang didukung:
        - .txt
        - .json

        Untuk .json, seluruh string di dalam struktur JSON akan diambil.
        """
        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            return

        for path in sorted(self.datasets_dir.iterdir()):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            if suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield line

            elif suffix == ".json":
                # Coba baca sebagai JSON utuh. Jika gagal, coba sebagai
                # JSON Lines untuk fleksibilitas dataset mentah.
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)

                    for text in self._extract_json_strings(data):
                        yield text

                except json.JSONDecodeError:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            for text in self._extract_json_strings(obj):
                                yield text

    def _extract_json_strings(self, node) -> Iterable[str]:
        """
        Mengambil semua string dari struktur JSON secara rekursif.
        """
        if isinstance(node, str):
            yield node

        elif isinstance(node, dict):
            for value in node.values():
                yield from self._extract_json_strings(value)

        elif isinstance(node, list):
            for value in node:
                yield from self._extract_json_strings(value)

    # ======================================================================
    # INTERNAL: TEXT NORMALIZATION
    # ======================================================================

    def _normalize_text(self, text: str) -> str:
        """
        Normalisasi teks sebelum tokenisasi:
        - ganti newline/tab menjadi spasi
        - rapatkan whitespace berulang
        - hapus spasi di awal/akhir
        """
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_valid_char(self, ch: str) -> bool:
        """
        Karakter valid untuk token dasar:
        - bukan whitespace
        - printable
        """
        if ch in (" ", "\n", "\r", "\t"):
            return False
        return ch.isprintable()

    # ======================================================================
    # INTERNAL: CORPUS STATISTICS
    # ======================================================================

    def _build_corpus_statistics(
        self,
    ) -> Tuple[Dict[Tuple[str, ...], int], Counter]:
        """
        Membangun:
        - word_freq: frekuensi urutan token dasar per kata
        - char_freq: frekuensi karakter dasar untuk initial vocab
        """
        word_freq: Dict[Tuple[str, ...], int] = {}
        char_freq: Counter = Counter()

        for sequence in self._iter_corpus_texts():
            normalized = self._normalize_text(sequence)
            if not normalized:
                continue

            words = normalized.split(" ")

            for idx, word in enumerate(words):
                base_tokens: List[str] = []

                if idx > 0:
                    base_tokens.append(self.SPACE_MARKER)

                for ch in word:
                    if self._is_valid_char(ch):
                        base_tokens.append(ch)
                        char_freq[ch] += 1

                # Lewati kata yang tidak memiliki karakter valid.
                if not base_tokens or base_tokens == [self.SPACE_MARKER]:
                    continue

                token_tuple = tuple(base_tokens)
                word_freq[token_tuple] = word_freq.get(token_tuple, 0) + 1

        return word_freq, char_freq

    # ======================================================================
    # INTERNAL: INITIAL VOCAB
    # ======================================================================

    def _build_initial_vocab(self, char_freq: Counter) -> Dict[str, int]:
        """
        Membangun vocab awal:
        1. special tokens
        2. space marker
        3. karakter paling sering muncul sampai batas vocab_size
        """
        token_to_id: Dict[str, int] = {}

        # 1. Special tokens dengan ID tetap.
        for token in self.special_ordered:
            token_to_id[token] = len(token_to_id)

        # 2. Space marker.
        if len(token_to_id) >= self.target_vocab_size:
            raise ValueError("vocab_size terlalu kecil untuk space marker.")

        token_to_id[self.SPACE_MARKER] = len(token_to_id)

        # 3. Karakter dasar berdasarkan frekuensi.
        sorted_chars = sorted(char_freq.items(), key=lambda item: (-item[1], item[0]))

        for ch, _freq in sorted_chars:
            if len(token_to_id) >= self.target_vocab_size:
                break

            if ch == self.SPACE_MARKER:
                continue

            if ch in token_to_id:
                continue

            if not self._is_valid_char(ch):
                continue

            token_to_id[ch] = len(token_to_id)

        return token_to_id

    def _project_word_freq_to_vocab(
        self,
        word_freq: Dict[Tuple[str, ...], int],
        token_to_id: Dict[str, int],
    ) -> Dict[Tuple[str, ...], int]:
        """
        Mengubah token dasar pada word_freq agar sesuai vocab awal.
        Token yang tidak dikenal diganti [UNK].
        """
        projected: Dict[Tuple[str, ...], int] = {}

        for tokens, freq in word_freq.items():
            mapped: List[str] = []
            previous_unk = False

            for token in tokens:
                if token in token_to_id:
                    mapped.append(token)
                    previous_unk = False
                else:
                    if not previous_unk:
                        mapped.append(self.unk_token)
                        previous_unk = True

            token_tuple = tuple(mapped)
            if token_tuple:
                projected[token_tuple] = projected.get(token_tuple, 0) + freq

        return projected

    # ======================================================================
    # INTERNAL: BPE LEARNING
    # ======================================================================

    def _learn_bpe(
        self,
        word_freq: Dict[Tuple[str, ...], int],
        token_to_id: Dict[str, int],
        merges: List[Tuple[str, str]],
    ) -> None:
        """
        Mempelajari merge BPE secara iteratif.

        Setiap iterasi:
        1. Hitung frekuensi pasangan token.
        2. Pilih pasangan dengan frekuensi tertinggi.
        3. Gabungkan menjadi token baru.
        4. Perbarui representasi kata.
        """
        while len(token_to_id) < self.target_vocab_size:
            pair_counts: Dict[Tuple[str, str], int] = {}
            existing_tokens = set(token_to_id.keys())

            for tokens, freq in word_freq.items():
                if len(tokens) < 2:
                    continue

                for i in range(len(tokens) - 1):
                    a = tokens[i]
                    b = tokens[i + 1]

                    # Jangan libatkan special tokens dalam merge.
                    if a in self._special_set or b in self._special_set:
                        continue

                    # Hindari membuat token yang sudah ada, termasuk special.
                    candidate = a + b
                    if candidate in existing_tokens:
                        continue

                    pair = (a, b)
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            # Deterministik:
            # - frekuensi tertinggi
            # - jika sama, pilih pasangan lexicographically terkecil
            best_pair = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]

            new_token = best_pair[0] + best_pair[1]

            if len(token_to_id) >= self.target_vocab_size:
                break

            if new_token not in token_to_id:
                token_to_id[new_token] = len(token_to_id)

            merges.append(best_pair)

            # Update seluruh representasi kata dengan merge terbaik.
            a, b = best_pair
            updated_word_freq: Dict[Tuple[str, ...], int] = {}

            for tokens, freq in word_freq.items():
                new_tokens: List[str] = []
                i = 0

                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == a
                        and tokens[i + 1] == b
                    ):
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                token_tuple = tuple(new_tokens)
                updated_word_freq[token_tuple] = (
                    updated_word_freq.get(token_tuple, 0) + freq
                )

            word_freq = updated_word_freq

    # ======================================================================
    # INTERNAL: VOCAB PADDING
    # ======================================================================

    def _pad_vocab(self, token_to_id: Dict[str, int]) -> Dict[str, int]:
        """
        Menambahkan token [UNUSED_x] agar jumlah vocab sesuai
        config.model.vocab_size.
        """
        token_to_id = dict(token_to_id)

        while len(token_to_id) < self.target_vocab_size:
            base_index = len(token_to_id)
            candidate = f"[UNUSED_{base_index}]"
            suffix = 0

            while candidate in token_to_id:
                suffix += 1
                candidate = f"[UNUSED_{base_index}_{suffix}]"

            token_to_id[candidate] = len(token_to_id)

        return token_to_id

    # ======================================================================
    # INTERNAL: BPE APPLICATION
    # ======================================================================

    def _apply_bpe(self, tokens: List[str]) -> List[str]:
        """
        Menerapkan merge BPE pada daftar token dasar.
        """
        if len(tokens) <= 1:
            return tokens

        if not self.merge_ranks:
            return tokens

        current_tokens = list(tokens)

        while len(current_tokens) > 1:
            best_pair: Tuple[str, str] | None = None
            best_rank: int | None = None

            for i in range(len(current_tokens) - 1):
                a = current_tokens[i]
                b = current_tokens[i + 1]

                # Jangan merge special tokens.
                if a in self._special_set or b in self._special_set:
                    continue

                pair = (a, b)
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_token = best_pair[0] + best_pair[1]
            merged_tokens: List[str] = []
            i = 0

            while i < len(current_tokens):
                if (
                    i < len(current_tokens) - 1
                    and current_tokens[i] == best_pair[0]
                    and current_tokens[i + 1] == best_pair[1]
                ):
                    merged_tokens.append(new_token)
                    i += 2
                else:
                    merged_tokens.append(current_tokens[i])
                    i += 1

            current_tokens = merged_tokens

        return current_tokens


# ==========================================================================
# DIRECT EXECUTION
# ==========================================================================

if __name__ == "__main__":
    # Jika file ini dijalankan langsung, lakukan training tokenizer
    # lalu uji encode-decode sederhana.

    tokenizer = Tokenizer()
    tokenizer.train()

    sample_text = "Apa itu mobil?"

    token_ids = tokenizer.encode(
        sample_text,
        add_bos=True,
        add_eos=True,
    )

    tokens = tokenizer.encode_to_tokens(sample_text)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

    print("Tokenizer ready.")
    print(f"Vocab size        : {tokenizer.vocab_size}")
    print(f"Vocab path        : {tokenizer.vocab_path}")
    print(f"Sample text       : {sample_text}")
    print(f"Tokens            : {tokens}")
    print(f"Token IDs         : {token_ids}")
    print(f"Decoded           : {decoded}")
