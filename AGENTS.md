# AGENTS.md — ScanRider (WAJIB DIBACA SEBELUM MENGERJAKAN APA PUN)

## IDENTITAS
Scanner keamanan web modular, Python 3.11+, asyncio, stealth-first.
Lingkungan: GitHub Codespaces (Linux). Hanya untuk target berizin/lab legal
(example.com, badssl.com, testphp.vulnweb.com).

## STRUKTUR (fix, jangan tambah/kurangi)
```
├── main.py               #18 UI rich
├── orchestrator.py       #17 task manager asyncio
├── stealth_servant.py    #16 util stealth (DI ROOT)
├── modules/__init__.py   #13 registry
├── modules/*.py          #1-12 modul scanning
├── handlers/__init__.py  #15
├── handlers/report_generator.py #14 -> tulis ke storage/
└── storage/.gitkeep      #19
```
Import selalu top-level: `from stealth_servant import ...` (bukan relative).

## KONTRAK MODUL (hukum tertinggi)
```python
MODULE_NAME = "nama_modul"
async def scan(target: str, config: dict) -> dict:
    # return {module, target, findings[], summary, severity, duration, ...ekstra}
```
- severity: critical|high|medium|low|info|error
- scan() TIDAK BOLEH melempar exception keluar; error -> severity "error"
- tiap file: data hardcode penuh di atas, komentar 1 baris bahasa Indonesia,
  blok `__all__`, self-test `if __name__ == "__main__":`
- blocking call (socket/dns/ssl) dibungkus asyncio.to_thread
- payload read-only (tanpa DROP/DELETE; XSS pakai token assignment)
- secret di report wajib di-mask (4 awal + **** + 4 akhir)

## KONTRAK stealth_servant (FIX — 9 modul HTTP mengimport ini)
```python
from stealth_servant import get_stealth_headers, random_delay, build_session
```
- get_stealth_headers() -> dict (UA acak dari pool min 15 UA browser)
- random_delay(base: float) -> awaitable (tidur acak base*0.5..1.5)
- build_session(verify=False, timeout=10, proxy=None) -> async ctx manager
  mengembalikan aiohttp.ClientSession MENTAH (jangan di-wrap).
Session harus mendukung semua ini (dipakai modul):
  .get(url, headers=, allow_redirects=, timeout=, params=dict)
  .options(url, headers=, allow_redirects=False, timeout=)
  resp.status / resp.headers(.get/.items/.getall) / resp.url
  await resp.read() / await resp.text(errors="ignore") / resp.content.read(n)
  build_session(verify=True) untuk API resmi (shodan_scanner)

## DEPENDENCY (whitelist tertutup)
aiohttp, dnspython (hanya dns_recon), rich (hanya main.py) + stdlib.
DILARANG tambah library lain tanpa izin user.

## ROADMAP (centang [x] setelah file lulus verifikasi)
- [ ] #1  modules/port_scanner.py — async TCP connect, TOP_PORTS ~50, banner grab
- [ ] #2  modules/subdom_scanner.py — brute ~170 word, wildcard detect, http_check
- [ ] #3  modules/headers_scanner.py — 10 header wajib, scoring grade A+..F, cookie audit, CSP/HSTS analysis
- [ ] #4  modules/dir_buster.py — ~180 path, baseline soft-404 (3x random), klasifikasi status
- [ ] #5  modules/tech_stack.py — ~65 signature ala Wappalyzer + versi, body dibaca via resp.content.read(n)
- [ ] #6  modules/dns_recon.py — A/AAAA/MX/NS/TXT/SOA/CNAME + SPF/DMARC + AXFR (dnspython)
- [ ] #7  modules/ssl_scanner.py — expiry/SAN/self-signed + tes nyata TLS1.0/1.1 + cipher lemah (stdlib ssl)
- [ ] #8  modules/cors_checker.py — reflect/null origin/6 bypass vectors/preflight OPTIONS/vary
- [ ] #9  modules/sqli_scanner.py — error-based + boolean-blind + time-based 2 tahap (N lalu 2N)
- [ ] #10 modules/xss_scanner.py — canary token unik + konteks raw/attr/js + escaping analysis
- [ ] #11 modules/git_exposure.py — ~50 path + validasi signature konten (magic bytes dll) + git deep recon (logs/HEAD -> dev emails) + extract secrets masked
- [ ] #12 modules/shodan_scanner.py — pasif via REST api.shodan.io, key dari config/env SHODAN_API_KEY, handle 401/404
- [ ] #16 stealth_servant.py — ⚠️ kerjakan SETELAH #12, SEBELUM #13
- [ ] #13 modules/__init__.py — REGISTRY dict 12 modul {name, scan, description, category, requires}
- [ ] #14 handlers/report_generator.py — JSON/TXT/HTML (template string murni) -> storage/report_<domain>_<ts>.*
- [ ] #15 handlers/__init__.py
- [ ] #17 orchestrator.py — asyncio.gather + semaphore, pilih modul dinamis, hasil -> report_generator
- [ ] #18 main.py — rich UI: banner, input target, pilih modul, progress, jalankan orchestrator
- [ ] #19 storage/.gitkeep

## PROSEDUR PER FILE (wajib)
1. Baca AGENTS.md + file tetangga terkait sebelum menulis.
2. Tulis file LENGKAP (tanpa TODO/placeholder).
3. Verifikasi: `python -m py_compile <file>` + jalankan self-test.
4. Setelah selesai output: tabel __all__ + blok catatan:
   FILE: path | MODULE_NAME | CONFIG KEYS | DEPENDENCIES | IMPORT STEALTH |
   FINDING KEYS | EXTRA RESULT KEYS | CATATAN
5. Centang roadmap di file ini.
6. Jangan sentuh file lain.

## LARANGAN
- Jangan mengarang API library; jangan ubah signature yang sudah dipakai.
- Jangan menghapus/menulis ulang file selesai kecuali diminta.
- Wordlist/signature harus lengkap sesuai jumlah di roadmap.
- Menemukan bug di file lain -> LAPORKAN ke user, jangan diam-diam ubah.