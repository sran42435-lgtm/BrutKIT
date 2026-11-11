# modules/ssl_scanner.py — expiry/SAN/self-signed + tes nyata TLS1.0/1.1 + cipher lemah (stdlib ssl)
MODULE_NAME = "ssl_scanner"

import asyncio
import ssl
import socket
import time
from typing import Dict, Any, List
from datetime import datetime, timezone

__all__ = ["MODULE_NAME", "scan"]

WEAK_CIPHERS = {
    "RC4", "DES", "3DES", "MD5", "SHA1", "EXPORT", "NULL", "ANON", "ADH", "AECDH"
}

TLS_VERSIONS = {
    "TLSv1": ssl.PROTOCOL_TLSv1,
    "TLSv1.1": ssl.PROTOCOL_TLSv1_1,
    "TLSv1.2": ssl.PROTOCOL_TLSv1_2,
    "TLSv1.3": ssl.PROTOCOL_TLS
}


def _get_cert_info(host: str, port: int = 443) -> Dict[str, Any]:
    """Ambil info sertifikat SSL."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                cipher = ssock.cipher()
                version = ssock.version()
                return {
                    "subject": dict(x[0] for x in cert.get("subject", [])),
                    "issuer": dict(x[0] for x in cert.get("issuer", [])),
                    "version": cert.get("version"),
                    "serial_number": cert.get("serialNumber"),
                    "not_before": cert.get("notBefore"),
                    "not_after": cert.get("notAfter"),
                    "san": [x[1] for x in cert.get("subjectAltName", [])],
                    "signature_algorithm": cert.get("signatureAlgorithm"),
                    "public_key_algorithm": cert.get("subjectPublicKeyAlgorithm"),
                    "cipher": cipher[0] if cipher else "",
                    "cipher_version": cipher[1] if cipher else "",
                    "cipher_bits": cipher[2] if cipher else 0,
                    "tls_version": version
                }
    except Exception as e:
        return {"error": str(e)}


def _check_expiry(not_after: str) -> Dict[str, Any]:
    """Cek expiry sertifikat."""
    try:
        exp = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_left = (exp - now).days
        return {
            "expires": not_after,
            "days_left": days_left,
            "expired": days_left < 0,
            "expiring_soon": 0 <= days_left <= 30
        }
    except Exception:
        return {"expires": not_after, "days_left": -1, "expired": False, "expiring_soon": False}


def _check_self_signed(cert_info: Dict) -> bool:
    """Cek apakah self-signed."""
    subject = cert_info.get("subject", {})
    issuer = cert_info.get("issuer", {})
    return subject.get("commonName") == issuer.get("commonName")


def _check_weak_cipher(cipher: str) -> List[str]:
    """Deteksi cipher lemah."""
    weak = []
    cipher_upper = cipher.upper()
    for w in WEAK_CIPHERS:
        if w in cipher_upper:
            weak.append(w)
    return weak


async def _test_tls_version(host: str, port: int, version_name: str, protocol: int) -> Dict[str, Any]:
    """Test apakah versi TLS tertentu didukung."""
    try:
        ctx = ssl.SSLContext(protocol)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("ALL:@SECLEVEL=0")
        with socket.create_connection((host, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                ssock.do_handshake()
                return {"version": version_name, "supported": True, "cipher": ssock.cipher()[0] if ssock.cipher() else ""}
    except Exception:
        return {"version": version_name, "supported": False, "cipher": ""}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Scan SSL: expiry, SAN, self-signed, TLS versi, cipher lemah."""
    start = time.time()
    host = target.replace("https://", "").replace("http://", "").strip("/")
    port = config.get("port", 443)

    cert_info = await asyncio.to_thread(_get_cert_info, host, port)
    if "error" in cert_info:
        return {
            "module": MODULE_NAME,
            "target": target,
            "findings": [{"type": "error", "value": cert_info["error"], "severity": "error"}],
            "summary": f"Gagal ambil sertifikat: {cert_info['error']}",
            "severity": "error",
            "duration": round(time.time() - start, 2)
        }

    findings = []
    expiry = _check_expiry(cert_info["not_after"])
    if expiry["expired"]:
        findings.append({"type": "expiry", "value": f"Sertifikat expired {expiry['days_left']} hari lalu", "severity": "high"})
    elif expiry["expiring_soon"]:
        findings.append({"type": "expiry", "value": f"Sertifikat expires dalam {expiry['days_left']} hari", "severity": "medium"})
    else:
        findings.append({"type": "expiry", "value": f"Valid {expiry['days_left']} hari lagi", "severity": "info"})

    if _check_self_signed(cert_info):
        findings.append({"type": "self_signed", "value": "Self-signed certificate", "severity": "high"})

    san = cert_info.get("san", [])
    if not san:
        findings.append({"type": "san", "value": "Tidak ada SAN", "severity": "medium"})
    else:
        findings.append({"type": "san", "value": f"{len(san)} SAN: {', '.join(san[:5])}{'...' if len(san) > 5 else ''}", "severity": "info"})

    weak_ciphers = _check_weak_cipher(cert_info.get("cipher", ""))
    if weak_ciphers:
        findings.append({"type": "weak_cipher", "value": f"Cipher lemah: {', '.join(weak_ciphers)}", "severity": "high"})
    else:
        findings.append({"type": "cipher", "value": f"{cert_info.get('cipher')} ({cert_info.get('cipher_bits')} bits)", "severity": "info"})

    tls_tasks = [_test_tls_version(host, port, v, p) for v, p in TLS_VERSIONS.items()]
    tls_results = await asyncio.gather(*tls_tasks)
    for tr in tls_results:
        if tr["supported"]:
            if tr["version"] in ["TLSv1", "TLSv1.1"]:
                findings.append({"type": "legacy_tls", "value": f"{tr['version']} didukung (cipher: {tr['cipher']})", "severity": "high"})
            else:
                findings.append({"type": "tls_version", "value": f"{tr['version']} didukung", "severity": "info"})

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"Cert: {expiry['days_left']} hari, SAN: {len(san)}, TLS: {', '.join([r['version'] for r in tls_results if r['supported']])}",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else ("medium" if any(f["severity"] == "medium" for f in findings) else "info"),
        "duration": round(duration, 2),
        "certificate": cert_info,
        "expiry": expiry,
        "tls_support": tls_results
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "expired.badssl.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))