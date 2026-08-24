# BrutKIT

BrutKIT adalah script Python interaktif untuk **pengujian keamanan aplikasi web yang berizin**. Script ini melakukan discovery parameter, membuat variasi payload, mengirim request HTTP, menganalisis respons, mempelajari feedback, dan menyimpan laporan.

> **Peringatan:** gunakan hanya pada sistem milik sendiri atau sistem yang secara tertulis mengizinkan pengujian. Jangan menguji layanan publik tanpa izin. Fitur proxy, throttling, dan variasi encoding tidak mengubah kewajiban untuk mematuhi hukum, kebijakan target, dan batas rate limit.

## Fitur

- Discovery parameter dari URL, form HTML, hidden field, JavaScript, link internal, robots/sitemap, path umum, dan endpoint tambahan.
- Generator payload berbasis kategori, validasi grammar, variasi encoding, polyglot, dan evolusi genetik.
- Analisis respons untuk mengelompokkan hasil sebagai `server_output`, `raw_html`, atau `blocked`.
- Pembelajaran feedback, fingerprint WAF berbasis respons, pemilihan mutasi adaptif, dan deteksi anomali.
- Rotasi proxy, fingerprint TLS, throttling adaptif, retry dengan backoff, dan dukungan HTTP async/sync.
- Penyimpanan target di SQLite dan laporan hasil dalam format TXT serta JSON.

## Prasyarat

- Python 3.9 atau lebih baru.
- Akses jaringan ke target yang telah memberi izin.
- `pip` yang dapat memasang dependency.
- Chromium jika ingin memakai fitur browser Playwright.

Dependency yang diperiksa script meliputi `numpy`, `scipy`, `scikit-learn`, `requests`, `beautifulsoup4`, `lxml`, `httpx`, `h2`, `aiohttp`, `selectolax`, `playwright`, `tldextract`, `fake-useragent`, `curl-cffi`, dan `tenacity`. `torch` hanya dipakai bila tersedia untuk generator LSTM.

## Instalasi

Clone repository lalu jalankan:

```bash
python3 brutkit.py
```

Saat mulai, script memeriksa dependency dan mencoba memasang package yang belum tersedia menggunakan `pip`. Karena pemeriksaan ini berjalan setiap kali script dijalankan, pastikan environment Python yang aktif adalah environment yang ingin digunakan.

Untuk memasang dependency secara manual:

```bash
python3 -m pip install numpy scipy scikit-learn requests beautifulsoup4 lxml httpx h2 aiohttp selectolax playwright tldextract fake-useragent curl-cffi tenacity
python3 -m playwright install chromium
```

## Cara Menggunakan

1. Jalankan `python3 brutkit.py`.
2. Masukkan URL, domain, atau IP target. Jika scheme tidak ditulis, script memakai `http://`.
3. Periksa daftar parameter yang ditemukan.
4. Jawab konfirmasi `Y` untuk melanjutkan pengujian, atau `N` untuk membatalkan.
5. Masukkan jumlah payload sebagai angka, atau `max` untuk mode 500 payload.
6. Setelah proses selesai, periksa ringkasan di terminal dan file laporan di `brut_results/`.

Contoh target:

```text
http://localhost:8080/test
```

## Perintah Lobby

| Perintah | Fungsi |
| --- | --- |
| `/save <url>` | Menyimpan target ke database SQLite. |
| `/viewdb` | Menampilkan target yang tersimpan. |
| `/load <id>` | Memuat target berdasarkan ID database. |
| `/proxy <file>` | Memakai daftar proxy dari file untuk proses berikutnya. |
| `/back` | Meminta konfirmasi untuk keluar dari lobby. |
| `/cancel` | Membatalkan target yang sedang dimuat atau proses yang dapat dibatalkan. |
| `/exit` atau `/quit` | Keluar dari script. |

File proxy dibaca dari baris per baris. Format yang umum digunakan adalah URL proxy, misalnya `http://127.0.0.1:8080`; baris kosong dan komentar dapat diabaikan oleh parser sesuai implementasi saat ini.

## Alur Internal

`BRUTPipeline` menjalankan proses dalam beberapa fase:

1. `phase1_discover()` menemukan parameter dan endpoint yang masih berada dalam scope domain target.
2. `phase2_generate(count)` membuat payload tervalidasi dan menginisialisasi populasi evolusi.
3. `phase3_inject()` menguji kombinasi payload dan parameter dengan throttling, proxy, retry, serta analisis respons.
4. `phase3_advanced_retry()` membuat varian tambahan dari hasil `blocked` atau `raw_html`.
5. `phase4_save()` menyimpan laporan dan `print_summary()` menampilkan ringkasan.

## File Output

- `brut_targets.db`: database SQLite untuk target yang disimpan melalui `/save`.
- `brut_results/`: direktori laporan per domain dan tahun.
- File `.txt`: laporan yang mudah dibaca manusia, termasuk evidence, status, proxy, dan statistik.
- File `.json`: data terstruktur untuk analisis lanjutan, termasuk hasil semua payload dan evolution log.

Hasil `server_output` adalah indikasi yang perlu diverifikasi secara manual. Status `raw_html` atau `blocked` bukan bukti kerentanan maupun bukti sistem aman. Gunakan mekanisme konfirmasi yang tersedia dan dokumentasikan izin serta ruang lingkup pengujian.

## Struktur Komponen

- `TargetDatabase`: penyimpanan target SQLite.
- `ParameterDiscovery`, `FastHTMLParser`, dan scanner endpoint: discovery.
- `MLPayloadGenerator`, `PolyglotGenerator`, dan `GeneticEvolver`: pembuatan serta evolusi payload.
- `Injector`, `ResponseAnalyzer`, dan `SecondOrderConfirmer`: request, klasifikasi, dan konfirmasi hasil.
- `ProxyPoolManager`, `AdaptiveThrottler`, `RetryEngine`, dan `TLSFingerprintEngine`: pengelolaan request.
- `ReportSaver` dan `DetailedLogger`: output laporan dan tampilan terminal.

## Troubleshooting

- **Dependency gagal dipasang:** aktifkan virtual environment, perbarui `pip`, lalu jalankan instalasi manual.
- **Tidak ada parameter ditemukan:** pastikan URL dapat diakses, endpoint memang memiliki input, dan target termasuk scope yang diizinkan.
- **Playwright tidak tersedia:** fitur browser akan dilewati; pasang package dan Chromium bila fitur tersebut dibutuhkan.
- **Banyak status 429:** hentikan pengujian atau kurangi beban sesuai kebijakan target; jangan mencoba mengakali batas layanan tanpa izin.
- **Dependency berat:** disarankan menggunakan lingkungan terminal yang suport seperti linux, tidak disarankan dilingkungan terminal termux, kecuali root.

## Lisensi

Lihat [LICENSE](LICENSE).
