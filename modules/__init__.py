# modules/__init__.py — REGISTRY dict 12 modul {name, scan, description, category, requires}
from typing import Dict, Any, Callable, Awaitable

ScanFunc = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]

REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_module(
    name: str,
    scan_func: ScanFunc,
    description: str,
    category: str = "active",
    requires: list = None
) -> None:
    """Register modul scanner ke REGISTRY."""
    REGISTRY[name] = {
        "name": name,
        "scan": scan_func,
        "description": description,
        "category": category,
        "requires": requires or []
    }


# Import dan register semua modul
try:
    from modules.port_scanner import MODULE_NAME as PORT_NAME, scan as port_scan
    register_module(PORT_NAME, port_scan, "Async TCP port scanner dengan banner grab", "network", ["asyncio", "socket"])
except ImportError:
    pass

try:
    from modules.subdom_scanner import MODULE_NAME as SUBDOM_NAME, scan as subdom_scan
    register_module(SUBDOM_NAME, subdom_scan, "Subdomain brute force dengan wildcard detect & HTTP check", "recon", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.headers_scanner import MODULE_NAME as HEADERS_NAME, scan as headers_scan
    register_module(HEADERS_NAME, headers_scan, "Security headers audit + CSP/HSTS analysis + cookie flags", "web", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.dir_buster import MODULE_NAME as DIR_NAME, scan as dir_scan
    register_module(DIR_NAME, dir_scan, "Directory brute force dengan baseline soft-404 & status klasifikasi", "web", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.tech_stack import MODULE_NAME as TECH_NAME, scan as tech_scan
    register_module(TECH_NAME, tech_scan, "Tech stack fingerprinting ~65 signature + version extraction", "web", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.dns_recon import MODULE_NAME as DNS_NAME, scan as dns_scan
    register_module(DNS_NAME, dns_scan, "DNS recon: A/AAAA/MX/NS/TXT/SOA/CNAME + SPF/DMARC + AXFR", "network", ["dnspython"])
except ImportError:
    pass

try:
    from modules.ssl_scanner import MODULE_NAME as SSL_NAME, scan as ssl_scan
    register_module(SSL_NAME, ssl_scan, "SSL/TLS scanner: expiry, SAN, self-signed, legacy TLS, weak ciphers", "network", ["ssl", "socket"])
except ImportError:
    pass

try:
    from modules.cors_checker import MODULE_NAME as CORS_NAME, scan as cors_scan
    register_module(CORS_NAME, cors_scan, "CORS misconfig checker: reflect/null/bypass vectors + preflight + Vary", "web", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.sqli_scanner import MODULE_NAME as SQLI_NAME, scan as sqli_scan
    register_module(SQLI_NAME, sqli_scan, "SQLi scanner: error-based + boolean-blind + time-based (2 tahap)", "vuln", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.xss_scanner import MODULE_NAME as XSS_NAME, scan as xss_scan
    register_module(XSS_NAME, xss_scan, "XSS scanner: canary token + konteks raw/attr/js + escaping analysis", "vuln", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.git_exposure import MODULE_NAME as GIT_NAME, scan as git_scan
    register_module(GIT_NAME, git_scan, "Git exposure: ~50 path + signature validation + deep recon + secrets masked", "vuln", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

try:
    from modules.shodan_scanner import MODULE_NAME as SHODAN_NAME, scan as shodan_scan
    register_module(SHODAN_NAME, shodan_scan, "Shodan passive lookup via REST API", "recon", ["aiohttp", "stealth_servant"])
except ImportError:
    pass

__all__ = ["REGISTRY", "register_module", "ScanFunc"]


if __name__ == "__main__":
    import json
    print("Registered modules:", len(REGISTRY))
    for name, info in REGISTRY.items():
        print(f"  - {name}: {info['description']} [{info['category']}]")
    assert len(REGISTRY) == 12, f"Expected 12 modules, got {len(REGISTRY)}"
    print("All 12 modules registered successfully!")