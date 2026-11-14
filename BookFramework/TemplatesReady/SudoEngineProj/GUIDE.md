# Panduan Penggunaan Ekosistem Sudoku Engine

Sistem ini terdiri dari tiga modul utama yang bekerja secara berurutan maupun mandiri:
1. **Sudoku Image Extractor** (Ekstraksi gambar papan Sudoku menjadi data teks).
2. **Sudoku Engine** (Pemecah utama, kalkulasi koordinat, dan validasi solusi).
3. **Sudoku Engine Benchmark System** (Pengujian performa dan perbandingan antar versi engine).

---

## 1. Persiapan & Kompilasi

Pastikan sistem Anda memiliki compiler C++ (mendukung C++17 atau lebih baru), library **OpenCV**, dan **Tesseract OCR**.

### Kompilasi Ketiga Modul (Linux / macOS)
```bash
# 1. Kompilasi Image Extractor (membutuhkan OpenCV)
g++ -std=c++17 -O3 Extract_Image_ToData.cpp -o extractor $(pkg-config --cflags --libs opencv4)

# 2. Kompilasi Sudoku Engine (ganti nama file sumber sesuai versi yang digunakan)
g++ -std=c++17 -O3 Engine.cpp -o engine

# 3. Kompilasi Benchmark System
g++ -std=c++17 -O3 Benchmark_engine.cpp -o benchmark
```

---

## 2. Modul 1: Sudoku Image Extractor (`./extractor`)

Modul ini mengubah foto atau tangkapan layar papan Sudoku menjadi string 81 digit yang siap diproses oleh Engine.

### Langkah Penggunaan
1. Jalankan program:
   ```bash
   ./extractor
   ```
2. Masukkan lokasi file gambar saat diminta:
   ```text
   Input image path: soal_sudoku.jpg
   ```
3. Masukkan nama file output (ekstensi `.txt` akan ditambahkan otomatis jika tidak ditulis):
   ```text
   Output file name: puzzle1.txt
   ```

### Output Yang Dihasilkan
* `puzzle1.txt`: Berisi 81 deret angka (`0` untuk kotak kosong, `1-9` untuk angka terisi).
* `puzzle1_grid.png`: Gambar visualisasi deteksi garis kotak dan koreksi perspektif.
* `puzzle1_hasil.png`: Gambar hasil pembacaan OCR pada setiap sel untuk memudahkan pengecekan manual jika ada sel yang gagal terbaca (`Uncertain`).

### Error

Terkadang ada ketidaksesuaian data hasil dengan gambar periksalah, kemungkinan penyebab utama gambar papan terlalu kecil, harap memberikan gambar papan yang besar agar mesin extract dapat membacanya  

---

## 3. Modul 2: Sudoku Engine (`./engine`)

Modul interaktif untuk memuat puzzle, memetakan kandidat dalam sistem koordinat Kartesius, memecahkan puzzle, dan mengecek keunikan solusi.

### Alur Kerja Dasar
1. Jalankan engine:
   ```bash
   ./engine
   ```
2. Muat file data puzzle:
   ```text
   /load puzzle1.txt
   ```
   *Engine akan memvalidasi keabsahan petunjuk awal, membuat `Session ID`, dan mencatat memori persisten (`.mem`) untuk mendeteksi perubahan file di masa mendatang.*
3. Hitung database koordinat & kandidat awal:
   ```text
   /calculate
   ```
   *Langkah ini wajib dilakukan sebelum menjalankan `/solve` (jika belum ada, engine akan menawarkan pembuatan otomatis).*
4. Pecahkan puzzle:
   ```text
   /solve
   ```

### Daftar Perintah Lengkap

| Perintah | Fungsi | Keterangan |
| :--- | :--- | :--- |
| `/load <file>` | Memuat data puzzle | Membaca 81 digit pertama dari file teks. Otomatis membuat atau membaca `[id_session]`. |
| `/calculate` | Membangun cache koordinat | Menyimpan pemetaan titik Kartesius `(-4..4)` dan kandidat awal ke `data_koordinat_<id>.txt`. |
| `/solve` | Memecahkan Sudoku | Menjalankan propagasi logika & backtracking. Menyimpan hasil lengkap ke `data_solve_<id>.txt`. |
| `/count` | Validasi keunikan solusi | Menghitung seluruh kemungkinan solusi (maksimal 9999) untuk memastikan puzzle valid (1 solusi unik). |
| `/hitungxy` | Inspeksi sel spesifik | Meminta input koordinat `X` dan `Y` (`-4` hingga `4`), lalu menampilkan isi baris, kolom, dan kandidatnya. |
| `/seesolve` | Tampilkan papan solusi | Mengubah tampilan papan di layar menjadi hasil akhir setelah `/solve` berhasil. |
| `/back` | Kembali ke papan awal | Mengembalikan tampilan papan di layar ke kondisi soal asli. |
| `/quit` | Keluar dari program | Menutup sesi engine. |

---

## 4. Modul 3: Benchmark System (`./benchmark`)

Modul otomatis untuk mengukur kecepatan eksekusi (dalam milidetik & mikrodetik), beban kerja solver, serta dampak tiap teknik logika pada satu atau beberapa versi engine sekaligus.

### Langkah Penggunaan
1. Jalankan program benchmark:
   ```bash
   ./benchmark
   ```
2. Daftarkan executable engine yang ingin diuji (bisa lebih dari satu untuk perbandingan):
   ```text
   Masukkan file executable engine contoh ./main ./engine, atau ./main_ver #1: ./engine_A
   Tambahkan engine lain? (y/n): y
   Masukkan file executable engine contoh ./main ./engine, atau ./main_ver #2: ./engine_B
   Tambahkan engine lain? (y/n): n
   ```
3. Masukkan file data puzzle untuk masing-masing engine:
   ```text
   Masukkan data puzzle untuk Engine #1 (./engine_A): puzzle1.txt
   Masukkan data puzzle untuk Engine #2 (./engine_B): puzzle1.txt
   ```
4. Tentukan jumlah iterasi pemanasan (*warm-up*) dan pengujian utama (*runs*):
   ```text
   Jumlah warm-up: 5
   Jumlah benchmark runs: 100
   ```
5. Ketik `continue` untuk memulai proses pengujian otomatis:
   ```text
   Mulai benchmark? (ketik 'continue'): continue
   ```

### Mekanisme & Hasil Analisis
* **Pre-Flight Check:** Sistem otomatis memeriksa keberadaan `Session ID` dan file `data_koordinat_<id>.txt`. Jika belum tersedia, benchmark akan menjalankan perintah `/calculate` secara otomatis sebelum memulai pengukuran waktu.
* **Metrik Utama:** Menampilkan waktu *Minimum*, *Median* (indikator utama kestabilan), *Mean*, *Maximum*, dan *Standard Deviation*.
* **Core Workload & Technique Impact:** Menampilkan nilai median untuk jumlah propagasi, eliminasi kandidat, tebakan (*backtracking*), *deadend*, serta waktu eksekusi dan jumlah panggilan per teknik logika.
