#!/usr/bin/env python3
"""
CrawlX - Advanced Hybrid Offensive Reconnaissance Tool
Single-script, modular, stealth-first architecture.
FOR AUTHORIZED SECURITY TESTING ONLY.
"""

import os
import sys
import json
import time
import asyncio
import signal
import hashlib
import re
import socket
import struct
import random
import string
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Set
from urllib.parse import urlparse, urljoin, parse_qs, urlencode
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

# ============================================================
# 0. CONSTANTS & CONFIGURATION LOADER
# ============================================================
class Constants:
    """Centralized configuration. Reads config.env + wordlist paths."""
    
    BASE_DIR = Path(__file__).parent.resolve()
    WORDLISTS_DIR = BASE_DIR / "wordlists"
    STORAGE_DIR = BASE_DIR / "storage"
    CONFIG_FILE = BASE_DIR / "config.env"
    
    # Wordlist paths
    PORTS_COMMON = WORDLISTS_DIR / "ports" / "common_ports.txt"
    PORTS_UDP = WORDLISTS_DIR / "ports" / "udp_ports.txt"
    SUBS_SMALL = WORDLISTS_DIR / "subdomains" / "subs_small.txt"
    SUBS_DEEP = WORDLISTS_DIR / "subdomains" / "subs_deep.txt"
    DIRS_COMMON = WORDLISTS_DIR / "dirbust" / "dirs_common.txt"
    DIRS_API = WORDLISTS_DIR / "dirbust" / "dirs_api.txt"
    EXTENSIONS = WORDLISTS_DIR / "dirbust" / "extensions.txt"
    LEAK_PATHS = WORDLISTS_DIR / "sensitive" / "leak_paths.txt"
    SQLI_ERROR = WORDLISTS_DIR / "payloads" / "sqli_error_based.txt"
    SQLI_BLIND = WORDLISTS_DIR / "payloads" / "sqli_blind_bool.txt"
    XSS_REFLECT = WORDLISTS_DIR / "payloads" / "xss_reflect.txt"
    SERVER_HEADERS = WORDLISTS_DIR / "fingerprints" / "server_headers.json"
    TECH_HASHES = WORDLISTS_DIR / "fingerprints" / "tech_hashes.json"
    WAPPALYZER_RULES = WORDLISTS_DIR / "fingerprints" / "wappalyzer_rules.json"
    
    # Banner
    BANNER = r"""
     ___                 ___  __ 
  / __|_ _ __ ___ __ _| \ \/ / 
 | (__| '_/ _` \ V  V / |>  <  
  \___|_| \__,_|\_/\_/|_/_/\_\ 
                                                             
    CrawlX v1.0 | Authorized Testing Only
"""
    
    # ANSI Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    @classmethod
    def load_env(cls) -> Dict[str, str]:
        """Parse config.env manually (no dependency needed at boot)."""
        config = {}
        if cls.CONFIG_FILE.exists():
            with open(cls.CONFIG_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        config[key.strip()] = value.strip()
        return config
    
    @classmethod
    def get(cls, key: str, default: str = "") -> str:
        if not hasattr(cls, '_env'):
            cls._env = cls.load_env()
        return cls._env.get(key, default)
    
    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        try:
            return int(cls.get(key, str(default)))
        except ValueError:
            return default
    
    @classmethod
    def get_bool(cls, key: str, default: bool = False) -> bool:
        return cls.get(key, str(default)).lower() in ('true', '1', 'yes')
    
    @classmethod
    def load_wordlist(cls, path: Path) -> List[str]:
        if not path.exists():
            return []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    @classmethod
    def load_json(cls, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================
# 1. DEPENDENCY GUARD
# ============================================================
class DependencyGuard:
    """Check and auto-install required dependencies."""
    
    REQUIRED = {
        "curl_cffi": "curl_cffi",
        "dns": "dnspython",
        "tldextract": "tldextract",
        "cryptography": "cryptography",
        "OpenSSL": "pyOpenSSL",
        "mmh3": "mmh3",
        "rich": "rich",
        "aiofiles": "aiofiles",
        "bs4": "beautifulsoup4",
        "lxml": "lxml",
        "fake_useragent": "fake-useragent",
    }
    
    @classmethod
    def check_and_install(cls):
        missing = []
        for module, package in cls.REQUIRED.items():
            try:
                __import__(module)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"{Constants.YELLOW}[!] Missing dependencies: {', '.join(missing)}{Constants.RESET}")
            print(f"{Constants.CYAN}[*] Installing automatically...{Constants.RESET}")
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Clear screen after install
            os.system('cls' if os.name == 'nt' else 'clear')
        
        # Warning before banner
        print(f"{Constants.BOLD}{Constants.YELLOW}{'='*60}")
        print("  ⚠  IMPORTANT: Configure API keys & settings in config.env")
        print(f"     File: {Constants.CONFIG_FILE}")
        print(f"{'='*60}{Constants.RESET}")
        time.sleep(3)
        os.system('cls' if os.name == 'nt' else 'clear')


# ============================================================
# 2. STEALTH SERVANT
# ============================================================
class StealthServant:
    """Stealthy HTTP client with TLS impersonation, rotation, jitter."""
    
    _session = None
    _ua_list = []
    
    @classmethod
    def _init_session(cls):
        if cls._session is not None:
            return
        from curl_cffi.requests import Session
        from fake_useragent import UserAgent
        
        impersonate = Constants.get("CRAWLX_TLS_IMPERSONATE", "chrome124")
        cls._session = Session(impersonate=impersonate)
        
        try:
            ua = UserAgent(browsers=['chrome', 'firefox', 'edge'])
            cls._ua_list = [ua.chrome, ua.firefox, ua.edge] * 5
        except Exception:
            cls._ua_list = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
            ]
    
    @classmethod
    def _jitter(cls):
        delay_min = float(Constants.get("CRAWLX_DELAY_MIN", "0.3"))
        delay_max = float(Constants.get("CRAWLX_DELAY_MAX", "1.2"))
        time.sleep(random.uniform(delay_min, delay_max))
    
    @classmethod
    def request(cls, method: str, url: str, **kwargs) -> Optional[Any]:
        cls._init_session()
        cls._jitter()
        
        timeout = Constants.get_int("CRAWLX_TIMEOUT", 10)
        max_retries = Constants.get_int("CRAWLX_MAX_RETRIES", 3)
        
        headers = kwargs.pop('headers', {})
        if Constants.get("CRAWLX_USER_AGENT_MODE", "rotate") == "rotate" and cls._ua_list:
            headers['User-Agent'] = random.choice(cls._ua_list)
        
        for attempt in range(max_retries):
            try:
                resp = cls._session.request(
                    method, url, headers=headers, timeout=timeout,
                    allow_redirects=True, **kwargs
                )
                # WAF detection → slowdown
                if resp.status_code in (403, 429, 503):
                    server = resp.headers.get('server', '').lower()
                    if any(w in server for w in ['cloudflare', 'akamai', 'fastly', 'aws']):
                        time.sleep(random.uniform(2, 5))
                return resp
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.debug(f"Request failed {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None
    
    @classmethod
    def get(cls, url: str, **kwargs):
        return cls.request("GET", url, **kwargs)
    
    @classmethod
    def post(cls, url: str, **kwargs):
        return cls.request("POST", url, **kwargs)


# ============================================================
# 3. TARGET RESOLVER
# ============================================================
class TargetResolver:
    """Identify input type and normalize to base target."""
    
    IP_REGEX = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    URL_REGEX = re.compile(r'^https?://', re.IGNORECASE)
    DOMAIN_REGEX = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$')
    
    @classmethod
    def resolve(cls, raw_input: str) -> Optional[Dict[str, str]]:
        raw_input = raw_input.strip()
        if not raw_input:
            return None
        
        result = {"raw": raw_input, "type": "", "host": "", "base_url": ""}
        
        # IP address
        if cls.IP_REGEX.match(raw_input):
            result["type"] = "IP"
            result["host"] = raw_input
            try:
                hostname = socket.gethostbyaddr(raw_input)[0]
                result["resolved_name"] = hostname
                result["base_url"] = f"http://{hostname}"
            except socket.herror:
                result["resolved_name"] = raw_input
                result["base_url"] = f"http://{raw_input}"
            return result
        
        # Full URL/link
        if cls.URL_REGEX.match(raw_input):
            parsed = urlparse(raw_input)
            host = parsed.hostname or parsed.netloc.split(':')[0]
            result["type"] = "URL"
            result["host"] = host
            result["base_url"] = f"{parsed.scheme}://{parsed.netloc}"
            # Try reverse DNS if IP
            if cls.IP_REGEX.match(host):
                try:
                    result["resolved_name"] = socket.gethostbyaddr(host)[0]
                except socket.herror:
                    result["resolved_name"] = host
            else:
                result["resolved_name"] = host
            return result
        
        # Domain
        if cls.DOMAIN_REGEX.match(raw_input):
            result["type"] = "Domain"
            result["host"] = raw_input
            result["resolved_name"] = raw_input
            result["base_url"] = f"https://{raw_input}"
            return result
        
        return None


# ============================================================
# 4. SCAN ORCHESTRATOR
# ============================================================
class ScanOrchestrator:
    """Sequential module execution with Ctrl+C handling."""
    
    MODULES = [
        ("Port Scanner", "PortScanner"),
        ("Subdomain Enumeration", "SubdomScanner"),
        ("DNS Recon", "DNSRecon"),
        ("SSL/TLS Audit", "SSLScanner"),
        ("Shodan Intelligence", "ShodanScanner"),
        ("HTTP Headers Analysis", "HeadersScanner"),
        ("Directory Buster", "DirBuster"),
        ("Tech Stack Fingerprint", "TechStack"),
        ("CORS Misconfiguration", "CORSChecker"),
        ("Sensitive File Leak", "GitExposure"),
        ("SQL Injection Detection", "SQLiScanner"),
        ("Reflected XSS Detection", "XSSScanner"),
    ]
    
    def __init__(self, target_info: Dict[str, str]):
        self.target = target_info
        self.results: Dict[str, Any] = {}
        self._interrupted = False
    
    def run(self) -> Dict[str, Any]:
        print(f"\n{Constants.BOLD}{Constants.CYAN}{'='*60}")
        print(f"  Target: {self.target['resolved_name']} ({self.target['host']})")
        print(f"  Type:   {self.target['type']}")
        print(f"  Base:   {self.target['base_url']}")
        print(f"{'='*60}{Constants.RESET}\n")
        
        for display_name, class_name in self.MODULES:
            if self._interrupted:
                break
            
            print(f"{Constants.BOLD}{Constants.BLUE}[▶] Starting: {display_name}{Constants.RESET}")
            start = time.time()
            
            try:
                module_class = globals()[class_name]
                scanner = module_class(self.target)
                result = scanner.run()
                self.results[class_name] = result
                
                elapsed = time.time() - start
                findings = len(result.get("findings", []))
                print(f"{Constants.GREEN}[✔] {display_name} completed in {elapsed:.1f}s | Findings: {findings}{Constants.RESET}\n")
                
            except KeyboardInterrupt:
                self._interrupted = True
                print(f"\n{Constants.YELLOW}[⚠] Process stopped by user. Returning to lobby...{Constants.RESET}")
                time.sleep(2)
                break
            except Exception as e:
                self.results[class_name] = {"error": str(e), "findings": []}
                print(f"{Constants.RED}[✘] {display_name} failed: {e}{Constants.RESET}\n")
        
        return self.results


# ============================================================
# 5. PORT SCANNER
# ============================================================
class PortScanner:
    """Async TCP connect scan with banner grabbing."""
    
    def __init__(self, target: Dict[str, str]):
        self.host = target["host"]
        self.ports = Constants.load_wordlist(Constants.PORTS_COMMON)
        self.findings = []
    
    async def _scan_port(self, port: int) -> Optional[Dict]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, port), timeout=2
            )
            banner = ""
            with suppress(Exception):
                data = await asyncio.wait_for(reader.read(1024), timeout=2)
                banner = data.decode('utf-8', errors='ignore').strip()[:200]
            writer.close()
            await writer.wait_closed()
            return {"port": port, "state": "open", "banner": banner}
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return None
    
    async def _run_async(self):
        workers = Constants.get_int("CRAWLX_THREAD_WORKERS", 25)
        sem = asyncio.Semaphore(workers)
        
        async def bounded_scan(port):
            async with sem:
                return await self._scan_port(port)
        
        tasks = [bounded_scan(int(p)) for p in self.ports if p.isdigit()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, dict) and r:
                self.findings.append(r)
                print(f"    {Constants.GREEN}├─ Port {r['port']}/tcp OPEN{Constants.RESET}"
                      f"{' | ' + r['banner'][:60] if r['banner'] else ''}")
    
    def run(self) -> Dict:
        asyncio.run(self._run_async())
        return {"module": "PortScanner", "target": self.host, "findings": self.findings}


# ============================================================
# 6. SUBDOMAIN SCANNER
# ============================================================
class SubdomScanner:
    """DNS brute-force + passive sources."""
    
    def __init__(self, target: Dict[str, str]):
        self.domain = target["host"]
        mode = Constants.get("CRAWLX_SUBDOMAIN_MODE", "small")
        wl_path = Constants.SUBS_DEEP if mode == "deep" else Constants.SUBS_SMALL
        self.wordlist = Constants.load_wordlist(wl_path)
        self.findings = []
        self._seen: Set[str] = set()
    
    async def _resolve(self, subdomain: str) -> Optional[str]:
        import dns.asyncresolver
        resolver = dns.asyncresolver.Resolver()
        resolver.nameservers = [Constants.get("CRAWLX_DNS_RESOLVER", "1.1.1.1")]
        resolver.timeout = 2
        resolver.lifetime = 3
        fqdn = f"{subdomain}.{self.domain}"
        try:
            answers = await resolver.resolve(fqdn, 'A')
            ips = [str(rdata) for rdata in answers]
            return fqdn if ips else None
        except Exception:
            return None
    
    def _passive_sources(self):
        """Query crt.sh for certificate transparency logs."""
        url = f"https://crt.sh/?q=%.{self.domain}&output=json"
        resp = StealthServant.get(url)
        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                for entry in data:
                    name = entry.get("name_value", "").lower().strip()
                    if name.endswith(f".{self.domain}") and name not in self._seen:
                        self._seen.add(name)
                        self.findings.append({"subdomain": name, "source": "crt.sh"})
                        print(f"    {Constants.CYAN}├─ {name} (crt.sh){Constants.RESET}")
            except Exception:
                pass
    
    async def _brute_force(self):
        workers = Constants.get_int("CRAWLX_THREAD_WORKERS", 25)
        sem = asyncio.Semaphore(workers)
        
        async def bounded_resolve(sub):
            async with sem:
                result = await self._resolve(sub)
                if result and result not in self._seen:
                    self._seen.add(result)
                    self.findings.append({"subdomain": result, "source": "dns_brute"})
                    print(f"    {Constants.GREEN}├─ {result} (DNS){Constants.RESET}")
        
        tasks = [bounded_resolve(s) for s in self.wordlist]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def run(self) -> Dict:
        self._passive_sources()
        asyncio.run(self._brute_force())
        return {"module": "SubdomScanner", "target": self.domain, "findings": self.findings}


# ============================================================
# 7. HEADERS SCANNER
# ============================================================
class HeadersScanner:
    """HTTP Security Headers analysis with scoring."""
    
    SECURITY_HEADERS = {
        "Strict-Transport-Security": {"severity": "HIGH", "desc": "HSTS missing"},
        "Content-Security-Policy": {"severity": "HIGH", "desc": "CSP missing"},
        "X-Frame-Options": {"severity": "MEDIUM", "desc": "Clickjacking protection missing"},
        "X-Content-Type-Options": {"severity": "MEDIUM", "desc": "MIME sniffing protection missing"},
        "X-XSS-Protection": {"severity": "LOW", "desc": "XSS filter header missing"},
        "Referrer-Policy": {"severity": "LOW", "desc": "Referrer policy missing"},
        "Permissions-Policy": {"severity": "MEDIUM", "desc": "Feature policy missing"},
        "Cache-Control": {"severity": "INFO", "desc": "Cache control not set"},
        "Server": {"severity": "INFO", "desc": "Server version disclosed"},
        "X-Powered-By": {"severity": "MEDIUM", "desc": "Technology disclosure"},
    }
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.findings = []
    
    def run(self) -> Dict:
        resp = StealthServant.get(self.base_url)
        if not resp:
            return {"module": "HeadersScanner", "target": self.base_url, "findings": [], "error": "No response"}
        
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        
        for header, info in self.SECURITY_HEADERS.items():
            present = header.lower() in headers_lower
            finding = {
                "header": header,
                "present": present,
                "value": headers_lower.get(header.lower(), ""),
                "severity": info["severity"] if not present else "OK",
                "description": info["desc"] if not present else "Header present"
            }
            self.findings.append(finding)
            status = Constants.GREEN + "✔" if present else Constants.RED + "✘"
            print(f"    {status} {header}: {finding['value'][:80] or '(missing)'}{Constants.RESET}")
        
        # Cookie flags check
        cookies = resp.headers.get('set-cookie', '')
        if cookies:
            cookie_issues = []
            if 'httponly' not in cookies.lower():
                cookie_issues.append("HttpOnly missing")
            if 'secure' not in cookies.lower():
                cookie_issues.append("Secure flag missing")
            if 'samesite' not in cookies.lower():
                cookie_issues.append("SameSite missing")
            if cookie_issues:
                self.findings.append({
                    "header": "Set-Cookie",
                    "present": True,
                    "value": ", ".join(cookie_issues),
                    "severity": "MEDIUM",
                    "description": "Cookie security flags missing"
                })
                print(f"    {Constants.YELLOW}⚠ Cookies: {', '.join(cookie_issues)}{Constants.RESET}")
        
        return {"module": "HeadersScanner", "target": self.base_url, "findings": self.findings}


# ============================================================
# 8. DIRECTORY BUSTER
# ============================================================
class DirBuster:
    """Async directory/file brute-forcer with smart 404 calibration."""
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        wl_name = Constants.get("CRAWLX_DIRBUST_WORDLIST", "dirs_common.txt")
        wl_map = {
            "dirs_common.txt": Constants.DIRS_COMMON,
            "dirs_api.txt": Constants.DIRS_API,
        }
        self.paths = Constants.load_wordlist(wl_map.get(wl_name, Constants.DIRS_COMMON))
        self.extensions = Constants.load_wordlist(Constants.EXTENSIONS)[:20]  # Top 20 ext
        self.findings = []
        self._baseline_404_len = 0
        self._baseline_404_hash = ""
    
    def _calibrate_404(self):
        """Detect custom 404 page by length + hash."""
        random_path = ''.join(random.choices(string.ascii_lowercase, k=16))
        resp = StealthServant.get(f"{self.base_url}/{random_path}")
        if resp:
            body = resp.text
            self._baseline_404_len = len(body)
            self._baseline_404_hash = hashlib.md5(body.encode()).hexdigest()
    
    def _is_real_finding(self, resp) -> bool:
        if resp.status_code == 404:
            body = resp.text
            if abs(len(body) - self._baseline_404_len) > 100:
                return True
            if hashlib.md5(body.encode()).hexdigest() != self._baseline_404_hash:
                return True
            return False
        return resp.status_code in (200, 301, 302, 403, 405, 500)
    
    async def _check_path(self, path: str) -> Optional[Dict]:
        url = urljoin(self.base_url, path)
        resp = StealthServant.get(url)
        if resp and self._is_real_finding(resp):
            return {
                "path": path,
                "status": resp.status_code,
                "size": len(resp.content),
                "redirect": resp.headers.get('location', '')
            }
        return None
    
    async def _run_async(self):
        self._calibrate_404()
        workers = Constants.get_int("CRAWLX_THREAD_WORKERS", 25)
        sem = asyncio.Semaphore(workers)
        
        all_paths = list(self.paths)
        # Append extensions to paths without extension
        for p in list(self.paths):
            if '.' not in p.split('/')[-1]:
                for ext in self.extensions[:10]:
                    all_paths.append(f"{p}{ext}")
        
        async def bounded_check(path):
            async with sem:
                return await self._check_path(path)
        
        tasks = [bounded_check(p) for p in all_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, dict) and r:
                self.findings.append(r)
                color = Constants.GREEN if r['status'] == 200 else Constants.YELLOW
                redir = f" → {r['redirect']}" if r['redirect'] else ""
                print(f"    {color}├─ [{r['status']}] {r['path']} ({r['size']}B){redir}{Constants.RESET}")
    
    def run(self) -> Dict:
        asyncio.run(self._run_async())
        return {"module": "DirBuster", "target": self.base_url, "findings": self.findings}


# ============================================================
# 9. TECH STACK IDENTIFIER
# ============================================================
class TechStack:
    """3-layer fingerprinting: headers + hashes + regex rules."""
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.findings = []
        self._server_sigs = Constants.load_json(Constants.SERVER_HEADERS)
        self._tech_hashes = Constants.load_json(Constants.TECH_HASHES)
        self._wa_rules = Constants.load_json(Constants.WAPPALYZER_RULES)
    
    def _check_headers(self, headers: Dict):
        detected = set()
        headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
        for tech, signatures in self._server_sigs.items():
            for sig in signatures:
                for hdr_key, hdr_val in headers_lower.items():
                    if sig.lower() in hdr_val or sig.lower() in hdr_key:
                        detected.add(tech)
        return detected
    
    def _check_favicon(self):
        resp = StealthServant.get(f"{self.base_url}/favicon.ico")
        if resp and resp.status_code == 200 and len(resp.content) > 0:
            fav_hash = mmh3.hash(resp.content)
            hex_hash = format(fav_hash & 0xFFFFFFFF, 'x')
            for tech, expected in self._tech_hashes.items():
                if hex_hash == expected or str(fav_hash) == expected:
                    return tech
        return None
    
    def _check_wappalyzer(self, html: str, headers: Dict, cookies: str):
        detected = set()
        for tech, rules in self._wa_rules.items():
            match = False
            # HTML patterns
            for pattern in rules.get("html", []):
                if pattern.lower() in html.lower():
                    match = True; break
            # Meta generator
            if not match:
                meta_gen = rules.get("meta", {}).get("generator", "")
                if meta_gen and meta_gen.lower() in html.lower():
                    match = True
            # Cookies
            if not match:
                for ck in rules.get("cookies", {}):
                    if ck.lower() in cookies.lower():
                        match = True; break
            # Headers
            if not match:
                for hk, hv in rules.get("headers", {}).items():
                    actual = headers.get(hk, "")
                    if hv and re.search(hv, actual, re.IGNORECASE):
                        match = True; break
                    elif not hv and hk.lower() in {k.lower() for k in headers}:
                        match = True; break
            if match:
                detected.add(tech)
        return detected
    
    def run(self) -> Dict:
        resp = StealthServant.get(self.base_url)
        if not resp:
            return {"module": "TechStack", "target": self.base_url, "findings": [], "error": "No response"}
        
        technologies = set()
        
        # Layer 1: Headers
        header_techs = self._check_headers(dict(resp.headers))
        technologies.update(header_techs)
        
        # Layer 2: Favicon hash
        try:
            import mmh3
            fav_tech = self._check_favicon()
            if fav_tech:
                technologies.add(fav_tech)
        except ImportError:
            pass
        
        # Layer 3: Wappalyzer-style rules
        wa_techs = self._check_wappalyzer(resp.text, dict(resp.headers), resp.headers.get('set-cookie', ''))
        technologies.update(wa_techs)
        
        for tech in sorted(technologies):
            self.findings.append({"technology": tech, "confidence": "high"})
            print(f"    {Constants.CYAN}├─ Detected: {tech}{Constants.RESET}")
        
        return {"module": "TechStack", "target": self.base_url, "findings": self.findings}


# ============================================================
# 10. DNS RECON
# ============================================================
class DNSRecon:
    """DNS records enumeration + zone transfer check."""
    
    RECORD_TYPES = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA', 'CNAME', 'CAA']
    
    def __init__(self, target: Dict[str, str]):
        self.domain = target["host"]
        self.findings = []
    
    def _query_records(self):
        import dns.resolver
        resolver = dns.resolver.Resolver()
        resolver.nameservers = [Constants.get("CRAWLX_DNS_RESOLVER", "1.1.1.1")]
        resolver.timeout = 3
        resolver.lifetime = 5
        
        for rtype in self.RECORD_TYPES:
            try:
                answers = resolver.resolve(self.domain, rtype)
                records = [str(rdata) for rdata in answers]
                self.findings.append({"type": rtype, "records": records})
                print(f"    {Constants.GREEN}├─ {rtype}: {', '.join(records[:3])}"
                      f"{'...' if len(records) > 3 else ''}{Constants.RESET}")
            except Exception:
                pass
    
    def _zone_transfer(self):
        import dns.query
        import dns.zone
        import dns.resolver
        try:
            ns_answers = dns.resolver.resolve(self.domain, 'NS')
            for ns in ns_answers:
                ns_host = str(ns).rstrip('.')
                try:
                    zone = dns.zone.from_xfr(dns.query.xfr(ns_host, self.domain, timeout=5))
                    names = [str(n) for n in zone.nodes.keys()]
                    self.findings.append({"type": "AXFR", "nameserver": ns_host, "records": names})
                    print(f"    {Constants.RED}├─ ⚠ ZONE TRANSFER SUCCESSFUL via {ns_host}: {len(names)} records{Constants.RESET}")
                    return
                except Exception:
                    pass
        except Exception:
            pass
    
    def run(self) -> Dict:
        self._query_records()
        self._zone_transfer()
        return {"module": "DNSRecon", "target": self.domain, "findings": self.findings}


# ============================================================
# 11. SSL SCANNER
# ============================================================
class SSLScanner:
    """SSL/TLS certificate audit."""
    
    def __init__(self, target: Dict[str, str]):
        self.host = target["host"]
        self.port = 443
        self.findings = []
    
    def run(self) -> Dict:
        import ssl
        import OpenSSL.crypto as crypto
        
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((self.host, self.port), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=self.host) as ssock:
                    cert_bin = ssock.getpeercert(binary_form=True)
                    protocol = ssock.version()
                    cipher = ssock.cipher()
            
            x509 = crypto.load_certificate(crypto.FILETYPE_ASN1, cert_bin)
            
            cert_info = {
                "subject": dict(x509.get_subject().get_components()),
                "issuer": dict(x509.get_issuer().get_components()),
                "serial": str(x509.get_serial_number()),
                "not_before": x509.get_notBefore().decode(),
                "not_after": x509.get_notAfter().decode(),
                "san": [],
                "protocol": protocol,
                "cipher": {"name": cipher[0], "bits": cipher[2]} if cipher else None,
                "issues": []
            }
            
            # Extract SANs
            for i in range(x509.get_extension_count()):
                ext = x509.get_extension(i)
                if ext.get_short_name() == b'subjectAltName':
                    cert_info["san"] = str(ext).split(', ')
            
            # Check expiry
            from datetime import timezone
            not_after = datetime.strptime(cert_info["not_after"], "%Y%m%d%H%M%SZ").replace(tzinfo=timezone.utc)
            days_left = (not_after - datetime.now(timezone.utc)).days
            if days_left < 30:
                cert_info["issues"].append(f"Certificate expires in {days_left} days")
            elif days_left < 0:
                cert_info["issues"].append("Certificate EXPIRED")
            
            # Weak protocol check
            if protocol in ('TLSv1', 'TLSv1.1', 'SSLv3'):
                cert_info["issues"].append(f"Weak protocol: {protocol}")
            
            self.findings.append(cert_info)
            
            print(f"    {Constants.GREEN}├─ Issuer: {cert_info['issuer'].get(b'O', b'Unknown').decode()}{Constants.RESET}")
            print(f"    {Constants.GREEN}├─ Protocol: {protocol} | Cipher: {cipher[0] if cipher else 'N/A'}{Constants.RESET}")
            print(f"    {Constants.GREEN}├─ Expires: {cert_info['not_after']} ({days_left} days){Constants.RESET}")
            if cert_info["san"]:
                print(f"    {Constants.CYAN}├─ SANs: {', '.join(cert_info['san'][:5])}{Constants.RESET}")
            for issue in cert_info["issues"]:
                print(f"    {Constants.RED}├─ ⚠ {issue}{Constants.RESET}")
                
        except Exception as e:
            self.findings.append({"error": str(e)})
            print(f"    {Constants.RED}├─ SSL scan failed: {e}{Constants.RESET}")
        
        return {"module": "SSLScanner", "target": self.host, "findings": self.findings}


# ============================================================
# 12. CORS CHECKER
# ============================================================
class CORSChecker:
    """CORS misconfiguration detection."""
    
    TEST_ORIGINS = [
        "https://evil.com",
        "null",
        "https://{domain}.evil.com",
        "https://evil.{domain}",
    ]
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.domain = target["host"]
        self.findings = []
    
    def run(self) -> Dict:
        origins = [o.format(domain=self.domain) for o in self.TEST_ORIGINS]
        
        for origin in origins:
            resp = StealthServant.get(self.base_url, headers={"Origin": origin})
            if not resp:
                continue
            
            acao = resp.headers.get('Access-Control-Allow-Origin', '')
            acac = resp.headers.get('Access-Control-Allow-Credentials', '').lower() == 'true'
            
            vulnerable = False
            severity = "INFO"
            desc = ""
            
            if acao == '*':
                if acac:
                    vulnerable = True
                    severity = "CRITICAL"
                    desc = "Wildcard ACAO + credentials"
                else:
                    severity = "LOW"
                    desc = "Wildcard ACAO (no credentials)"
            elif acao == origin:
                vulnerable = True
                severity = "HIGH" if acac else "MEDIUM"
                desc = f"Origin reflected{' + credentials' if acac else ''}"
            elif acao == 'null':
                vulnerable = True
                severity = "HIGH"
                desc = "Null origin allowed"
            
            finding = {
                "origin_tested": origin,
                "acao": acao,
                "credentials": acac,
                "vulnerable": vulnerable,
                "severity": severity,
                "description": desc
            }
            self.findings.append(finding)
            
            if vulnerable:
                print(f"    {Constants.RED}├─ ⚠ [{severity}] Origin: {origin} → ACAO: {acao}{Constants.RESET}")
            else:
                print(f"    {Constants.GREEN}├─ OK Origin: {origin} → ACAO: {acao or '(none)'}{Constants.RESET}")
        
        return {"module": "CORSChecker", "target": self.base_url, "findings": self.findings}


# ============================================================
# 13. SQL INJECTION SCANNER (Detection Only)
# ============================================================
class SQLiScanner:
    """Non-destructive SQL injection detection via error signatures + boolean diff."""
    
    ERROR_SIGNATURES = {
        "MySQL": ["mysql_fetch", "you have an error in your sql syntax", "warning: mysql"],
        "PostgreSQL": ["pg_query", "unterminated quoted string", "postgresql error"],
        "MSSQL": ["microsoft sql server", "unclosed quotation mark", "sql server error"],
        "Oracle": ["ora-01756", "oracle error", "quoted string not properly terminated"],
        "SQLite": ["sqlite3.operationalerror", "near \"", "unrecognized token"],
        "Generic": ["sql syntax", "syntax error", "database error", "sql error"],
    }
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.findings = []
        self._enabled = Constants.get_bool("CRAWLX_SCAN_SQLI", True)
    
    def _discover_params(self) -> List[Tuple[str, str]]:
        """Find testable parameters from homepage forms + query strings."""
        params = []
        resp = StealthServant.get(self.base_url)
        if not resp:
            return params
        
        # Query string params
        parsed = urlparse(self.base_url)
        qs = parse_qs(parsed.query)
        for key in qs:
            params.append(("GET", f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{key}=TESTVAL"))
        
        # Form params (basic)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'lxml')
        for form in soup.find_all('form'):
            action = form.get('action', self.base_url)
            method = form.get('method', 'GET').upper()
            full_action = urljoin(self.base_url, action)
            inputs = form.find_all(['input', 'textarea', 'select'])
            for inp in inputs:
                name = inp.get('name')
                if name:
                    params.append((method, full_action, name))
        
        return params
    
    def _test_error_based(self, url: str, param_name: str = "") -> Optional[Dict]:
        payloads = Constants.load_wordlist(Constants.SQLI_ERROR)[:10]
        for payload in payloads:
            test_url = url.replace("TESTVAL", payload) if "TESTVAL" in url else f"{url}&{param_name}={payload}"
            resp = StealthServant.get(test_url)
            if resp and resp.status_code == 200:
                body_lower = resp.text.lower()
                for db_type, signatures in self.ERROR_SIGNATURES.items():
                    for sig in signatures:
                        if sig.lower() in body_lower:
                            return {
                                "type": "error_based",
                                "db_type": db_type,
                                "payload": payload,
                                "url": test_url,
                                "signature_matched": sig
                            }
        return None
    
    def _test_blind_boolean(self, url: str, param_name: str = "") -> Optional[Dict]:
        true_payload = Constants.load_wordlist(Constants.SQLI_BLIND)[0] if Constants.load_wordlist(Constants.SQLI_BLIND) else "' AND '1'='1"
        false_payload = Constants.load_wordlist(Constants.SQLI_BLIND)[1] if len(Constants.load_wordlist(Constants.SQLI_BLIND)) > 1 else "' AND '1'='2"
        
        true_url = url.replace("TESTVAL", true_payload) if "TESTVAL" in url else f"{url}&{param_name}={true_payload}"
        false_url = url.replace("TESTVAL", false_payload) if "TESTVAL" in url else f"{url}&{param_name}={false_payload}"
        
        resp_true = StealthServant.get(true_url)
        resp_false = StealthServant.get(false_url)
        
        if resp_true and resp_false:
            len_diff = abs(len(resp_true.text) - len(resp_false.text))
            if len_diff > 200:
                return {
                    "type": "blind_boolean",
                    "length_difference": len_diff,
                    "true_size": len(resp_true.text),
                    "false_size": len(resp_false.text),
                    "url": true_url
                }
        return None
    
    def run(self) -> Dict:
        if not self._enabled:
            print(f"    {Constants.YELLOW}├─ SQLi scanning disabled in config.env{Constants.RESET}")
            return {"module": "SQLiScanner", "target": self.base_url, "findings": [], "skipped": True}
        
        params = self._discover_params()
        tested = 0
        
        for param in params[:20]:  # Limit to prevent excessive requests
            if len(param) == 3:
                method, url, name = param
                if method == "GET":
                    result = self._test_error_based(url, name)
                    if not result:
                        result = self._test_blind_boolean(url, name)
                else:
                    continue  # POST handled similarly but simplified here
            elif len(param) == 2:
                method, url = param
                result = self._test_error_based(url)
                if not result:
                    result = self._test_blind_boolean(url)
            else:
                continue
            
            tested += 1
            if result:
                self.findings.append(result)
                print(f"    {Constants.RED}├─ ⚠ SQLi [{result['type']}] {result.get('db_type','')} @ {result['url'][:80]}{Constants.RESET}")
        
        print(f"    {Constants.BLUE}├─ Tested {tested} parameters{Constants.RESET}")
        return {"module": "SQLiScanner", "target": self.base_url, "findings": self.findings}


# ============================================================
# 14. XSS SCANNER (Detection Only)
# ============================================================
class XSSScanner:
    """Reflected XSS detection via canary reflection + contextual analysis."""
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.findings = []
        self._enabled = Constants.get_bool("CRAWLX_SCAN_XSS", True)
    
    def _generate_canary(self) -> str:
        return f"cr4wlx_{random.randint(100000,999999)}"
    
    def _discover_reflect_points(self) -> List[str]:
        """Find URLs where parameters are reflected."""
        points = []
        resp = StealthServant.get(self.base_url)
        if not resp:
            return points
        
        parsed = urlparse(self.base_url)
        qs = parse_qs(parsed.query)
        for key in qs:
            canary = self._generate_canary()
            test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{key}={canary}"
            check = StealthServant.get(test_url)
            if check and canary in check.text:
                points.append((key, test_url, canary))
        
        return points
    
    def _test_xss(self, param_name: str, url_template: str, canary: str) -> Optional[Dict]:
        payloads = Constants.load_wordlist(Constants.XSS_REFLECT)[:8]
        for payload in payloads:
            test_url = url_template.replace(canary, payload)
            resp = StealthServant.get(test_url)
            if resp and payload in resp.text:
                # Contextual check
                idx = resp.text.find(payload)
                context_start = max(0, idx - 50)
                context_end = min(len(resp.text), idx + len(payload) + 50)
                context = resp.text[context_start:context_end]
                
                return {
                    "type": "reflected_xss",
                    "parameter": param_name,
                    "payload": payload,
                    "url": test_url[:120],
                    "context_snippet": context[:100]
                }
        return None
    
    def run(self) -> Dict:
        if not self._enabled:
            print(f"    {Constants.YELLOW}├─ XSS scanning disabled in config.env{Constants.RESET}")
            return {"module": "XSSScanner", "target": self.base_url, "findings": [], "skipped": True}
        
        reflect_points = self._discover_reflect_points()
        
        for param_name, url_template, canary in reflect_points[:15]:
            result = self._test_xss(param_name, url_template, canary)
            if result:
                self.findings.append(result)
                print(f"    {Constants.RED}├─ ⚠ Reflected XSS: param='{param_name}' @ {result['url'][:80]}{Constants.RESET}")
        
        print(f"    {Constants.BLUE}├─ Checked {len(reflect_points)} reflection points{Constants.RESET}")
        return {"module": "XSSScanner", "target": self.base_url, "findings": self.findings}


# ============================================================
# 15. GIT EXPOSURE / SENSITIVE FILES
# ============================================================
class GitExposure:
    """Check for sensitive file leaks with content verification."""
    
    def __init__(self, target: Dict[str, str]):
        self.base_url = target["base_url"]
        self.paths = Constants.load_wordlist(Constants.LEAK_PATHS)
        self.findings = []
    
    def _verify_content(self, path: str, resp) -> bool:
        """Verify it's real content, not a generic 200 page."""
        body = resp.text.lower()
        content_checks = {
            ".git/config": ["[core]", "[remote", "repositoryformatversion"],
            ".git/head": ["ref:", "refs/heads"],
            ".env": ["=", "secret", "key", "password", "api", "db_", "database"],
            "wp-config.php": ["define(", "db_name", "db_password", "table_prefix"],
            "web.config": ["<configuration>", "<system.web>"],
            ".htpasswd": [":"],
            "elmah.axd": ["error log", "elmah"],
            "actuator/health": ["status", "up", "down"],
            "swagger.json": ["swagger", "paths", "info"],
            "openapi.json": ["openapi", "paths"],
            "phpinfo": ["php version", "phpinfo()"],
            "server-status": ["apache server status", "current time"],
        }
        
        for keyword, markers in content_checks.items():
            if keyword in path.lower():
                return any(m in body for m in markers)
        
        # Generic: reject if body looks like HTML landing page
        if '<html' in body and len(body) > 5000:
            title_match = re.search(r'<title>(.*?)</title>', body)
            if title_match and ('404' in title_match.group(1).lower() or 'not found' in title_match.group(1).lower()):
                return False
        
        return True
    
    async def _check_path(self, path: str) -> Optional[Dict]:
        url = urljoin(self.base_url, path)
        resp = StealthServant.get(url)
        if resp and resp.status_code == 200 and self._verify_content(path, resp):
            return {
                "path": path,
                "status": resp.status_code,
                "size": len(resp.content),
                "content_preview": resp.text[:200].replace('\n', ' ').strip()
            }
        return None
    
    async def _run_async(self):
        workers = Constants.get_int("CRAWLX_THREAD_WORKERS", 25)
        sem = asyncio.Semaphore(workers)
        
        async def bounded_check(path):
            async with sem:
                return await self._check_path(path)
        
        tasks = [bounded_check(p) for p in self.paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, dict) and r:
                self.findings.append(r)
                print(f"    {Constants.RED}├─ ⚠ LEAKED: {r['path']} ({r['size']}B){Constants.RESET}")
    
    def run(self) -> Dict:
        asyncio.run(self._run_async())
        return {"module": "GitExposure", "target": self.base_url, "findings": self.findings}


# ============================================================
# 16. SHODAN SCANNER
# ============================================================
class ShodanScanner:
    """Shodan API intelligence gathering."""
    
    def __init__(self, target: Dict[str, str]):
        self.host = target["host"]
        self.api_key = Constants.get("SHODAN_API_KEY", "")
        self.findings = []
    
    def run(self) -> Dict:
        if not self.api_key:
            print(f"    {Constants.YELLOW}├─ Shodan API key not configured. Skipping.{Constants.RESET}")
            return {"module": "ShodanScanner", "target": self.host, "findings": [], "skipped": True}
        
        url = f"https://api.shodan.io/shodan/host/search?key={self.api_key}&query=hostname:{self.host}"
        resp = StealthServant.get(url)
        
        if not resp or resp.status_code != 200:
            print(f"    {Constants.YELLOW}├─ Shodan query failed (status: {resp.status_code if resp else 'N/A'}){Constants.RESET}")
            return {"module": "ShodanScanner", "target": self.host, "findings": [], "error": "API error"}
        
        try:
            data = resp.json()
            matches = data.get("matches", [])
            
            for match in matches[:10]:
                finding = {
                    "ip": match.get("ip_str"),
                    "port": match.get("port"),
                    "product": match.get("product", ""),
                    "version": match.get("version", ""),
                    "vulns": match.get("vulns", []),
                    "isp": match.get("isp", ""),
                    "org": match.get("org", ""),
                    "location": f"{match.get('location', {}).get('city', '')}, {match.get('location', {}).get('country_name', '')}"
                }
                self.findings.append(finding)
                vuln_str = f" | CVEs: {len(finding['vulns'])}" if finding['vulns'] else ""
                print(f"    {Constants.CYAN}├─ {finding['ip']}:{finding['port']} {finding['product']} {finding['version']}{vuln_str}{Constants.RESET}")
            
            if not matches:
                print(f"    {Constants.BLUE}├─ No Shodan results for {self.host}{Constants.RESET}")
                
        except Exception as e:
            print(f"    {Constants.RED}├─ Shodan parse error: {e}{Constants.RESET}")
        
        return {"module": "ShodanScanner", "target": self.host, "findings": self.findings}


# ============================================================
# 17. REPORT GENERATOR
# ============================================================
class ReportGenerator:
    """Bundle all results into JSON + TXT reports."""
    
    @staticmethod
    def generate(target_info: Dict[str, str], results: Dict[str, Any]):
        Constants.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%d%m%Y")
        safe_name = re.sub(r'[^\w\-.]', '_', target_info['host'])
        base_filename = f"{safe_name}_{date_str}"
        
        json_path = Constants.STORAGE_DIR / f"{base_filename}.json"
        txt_path = Constants.STORAGE_DIR / f"{base_filename}.txt"
        
        # Build report structure
        report = {
            "tool": "CrawlX v1.0",
            "generated_at": datetime.now().isoformat(),
            "target": target_info,
            "summary": {
                "total_modules": len(results),
                "total_findings": sum(len(v.get("findings", [])) for v in results.values()),
                "critical": sum(1 for v in results.values() for f in v.get("findings", []) if f.get("severity") == "CRITICAL"),
                "high": sum(1 for v in results.values() for f in v.get("findings", []) if f.get("severity") == "HIGH"),
                "medium": sum(1 for v in results.values() for f in v.get("findings", []) if f.get("severity") == "MEDIUM"),
            },
            "modules": results
        }
        
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Save TXT (human-readable)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"  CrawlX Security Assessment Report\n")
            f.write(f"  Generated: {report['generated_at']}\n")
            f.write(f"  Target: {target_info['resolved_name']} ({target_info['host']})\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"SUMMARY\n")
            f.write(f"  Total Findings: {report['summary']['total_findings']}\n")
            f.write(f"  Critical: {report['summary']['critical']} | High: {report['summary']['high']} | Medium: {report['summary']['medium']}\n\n")
            
            for module_name, module_data in results.items():
                findings = module_data.get("findings", [])
                f.write(f"{'─'*70}\n")
                f.write(f"[{module_name}] — {len(findings)} finding(s)\n")
                f.write(f"{'─'*70}\n")
                
                if module_data.get("skipped"):
                    f.write("  ⊘ Skipped (not configured)\n\n")
                    continue
                if module_data.get("error"):
                    f.write(f"  ✘ Error: {module_data['error']}\n\n")
                    continue
                
                for finding in findings:
                    severity = finding.get("severity", "INFO")
                    if severity in ("CRITICAL", "HIGH"):
                        marker = "⚠"
                    elif severity == "MEDIUM":
                        marker = "△"
                    else:
                        marker = "•"
                    
                    # Format based on module type
                    if "port" in finding:
                        f.write(f"  {marker} Port {finding['port']}/tcp — {finding.get('banner', '')[:60]}\n")
                    elif "subdomain" in finding:
                        f.write(f"  {marker} {finding['subdomain']} ({finding.get('source', '')})\n")
                    elif "header" in finding:
                        f.write(f"  {marker} [{severity}] {finding['header']}: {finding.get('value', '')[:60]}\n")
                    elif "path" in finding:
                        f.write(f"  {marker} [{finding.get('status', '')}] {finding['path']} ({finding.get('size', 0)}B)\n")
                    elif "technology" in finding:
                        f.write(f"  {marker} {finding['technology']}\n")
                    elif "type" in finding and "record" in str(finding.get("type", "")).lower():
                        f.write(f"  {marker} {finding.get('type', '')}: {', '.join(str(r) for r in finding.get('records', [])[:3])}\n")
                    elif "origin_tested" in finding:
                        f.write(f"  {marker} [{severity}] CORS Origin: {finding['origin_tested']} → {finding.get('acao', '')}\n")
                    elif "db_type" in finding:
                        f.write(f"  {marker} SQLi [{finding['type']}] {finding['db_type']} @ {finding.get('url', '')[:80]}\n")
                    elif finding.get("type") == "reflected_xss":
                        f.write(f"  {marker} XSS param='{finding.get('parameter', '')}' @ {finding.get('url', '')[:80]}\n")
                    else:
                        f.write(f"  {marker} {json.dumps(finding, default=str)[:120]}\n")
                
                f.write("\n")
            
            f.write(f"{'='*70}\n")
            f.write(f"End of Report\n")
        
        print(f"\n{Constants.BOLD}{Constants.GREEN}{'='*60}")
        print(f"  📄 Reports saved:")
        print(f"     JSON: {json_path}")
        print(f"     TXT:  {txt_path}")
        print(f"{'='*60}{Constants.RESET}\n")


# ============================================================
# MAIN LOOP & LOBBY
# ============================================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_banner():
    print(Constants.BANNER)

def lobby():
    """Main input loop."""
    while True:
        clear_screen()
        show_banner()
        print(f"{Constants.BOLD}Enter target (IP / URL / Domain) or '/exit' to quit:{Constants.RESET}")
        
        try:
            user_input = input(f"\n{Constants.CYAN}crawlX> {Constants.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Constants.YELLOW}[!] Exiting CrawlX. Goodbye.{Constants.RESET}")
            sys.exit(0)
        
        if user_input.lower() in ('/exit', 'exit', 'quit'):
            print(f"{Constants.YELLOW}[!] Exiting CrawlX. Goodbye.{Constants.RESET}")
            sys.exit(0)
        
        if not user_input:
            print(f"{Constants.RED}[✘] Empty input. Refreshing in 3 seconds...{Constants.RESET}")
            time.sleep(3)
            continue
        
        # Resolve target
        target_info = TargetResolver.resolve(user_input)
        if not target_info:
            print(f"{Constants.RED}[✘] Invalid input: '{user_input}'. Must be IP, URL, or domain.{Constants.RESET}")
            print(f"{Constants.YELLOW}[*] Refreshing lobby in 3 seconds...{Constants.RESET}")
            time.sleep(3)
            continue
        
        # Display resolved target
        clear_screen()
        show_banner()
        print(f"{Constants.GREEN}[✔] Target identified:{Constants.RESET}")
        print(f"    Type:     {target_info['type']}")
        print(f"    Input:    {target_info['raw']}")
        print(f"    Host:     {target_info['host']}")
        print(f"    Resolved: {target_info.get('resolved_name', 'N/A')}")
        print(f"    Base URL: {target_info['base_url']}")
        print(f"\n{Constants.CYAN}[*] Starting scan modules...{Constants.RESET}")
        time.sleep(1)
        
        # Run orchestrator
        orchestrator = ScanOrchestrator(target_info)
        results = orchestrator.run()
        
        # Generate reports (only if not interrupted before any module completed)
        if results:
            ReportGenerator.generate(target_info, results)
        
        # Return to lobby
        input(f"\n{Constants.BOLD}Press Enter to return to lobby...{Constants.RESET}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Stage 1: Dependency check
    DependencyGuard.check_and_install()
    
    # Stage 2+: Lobby loop
    try:
        lobby()
    except KeyboardInterrupt:
        print(f"\n{Constants.YELLOW}[!] Exiting CrawlX. Goodbye.{Constants.RESET}")
        sys.exit(0)
