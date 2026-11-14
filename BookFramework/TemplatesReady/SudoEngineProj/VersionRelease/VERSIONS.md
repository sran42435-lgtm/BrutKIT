# Sudoku Engine (C++) — Spesifikasi & Evolusi Versi

| Versi | Codename | Status | Karakteristik Utama |
| :--- | :--- | :--- | :--- |
| **v10.4** | *Hot Path Optimization* | `RAW PERFORMANCE SPEED` | Kecepatan propagasi murni tanpa *overhead* look-ahead, boros sumber daya saat *backtracking* dalam. |
| **v10.5** | *Contradiction Probe* | `HEURISTIC PRUNING` | Memangkas tebakan buntu lebih awal, namun melakukan *over-probing* tanpa filter kepadatan sel. |
| **v10.6** | *Precision Decision Cycle* | `OPTIMAL & RESOURCE-EFFICIENT` | Kondisi paling optimal; hemat CPU/memori dengan *probe* selektif dan memicu re-propagasi penuh. |

---

## v10.4 — Hot Path Optimization
* **Status:** `RAW PERFORMANCE SPEED`
* **Spesifikasi:** Bitmask kandidat (`uint16_t`), *UDP (Unit-Digit Position) Cache*, tabel `LOCAL_POS`, dan *bitmask worklist* (`dirtyUnits`).
* **Cara Kerja & Fitur:**
  * Mengevaluasi *Hidden Single*, *Locked Candidates*, dan *X-Wing* dalam waktu $O(1)$ berbasis operasi bitwise.
  * Masuk langsung ke *Backtracking (MRV + LCV)* setelah propagasi logika selesai tanpa fase *look-ahead*.

## v10.5 — Adaptive MRV Contradiction Probe
* **Status:** `HEURISTIC PRUNING`
* **Penambahan Fitur:**
  * **`probeContradiction()`:** Menguji kandidat secara sementara melalui 3 level pengecekan (eliminasi *peer*, rantai *Naked Single*, dan validasi 3 unit) pada sel MRV dengan 2 atau 3 kandidat.
  * Menghapus kandidat yang terbukti menghasilkan jalan buntu secara permanen sebelum melakukan percabangan tebakan.

## v10.6 — Precision Decision Cycle
* **Status:** `OPTIMAL & RESOURCE-EFFICIENT`
* **Penambahan Fitur:**
  * **Probe Eligibility Gate (`layakContradictionProbe`):** Memanfaatkan metadata `MRVResult` untuk membatasi *probe* hanya pada area padat (`2 kandidat` dengan `degree >= 8`, atau `3 kandidat` dengan `degree >= 12`).
  * **Decision Cycle Loop & Re-Propagation:** Menjalankan ulang `propagasiIncremental()` secara otomatis setelah *probe* berhasil menghapus kandidat, sehingga sering kali menyelesaikan papan tanpa *backtracking*.
  * **Queue Deduplication (`queued[81]`):** Mencegah sel yang sama masuk ke antrean berkali-kali saat *Naked Single cascade* di dalam *probe* untuk menghemat siklus CPU.
