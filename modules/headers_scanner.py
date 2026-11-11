# modules/headers_scanner.py — 10 header wajib, scoring grade A+..F, cookie audit, CSP/HSTS analysis
MODULE_NAME = "headers_scanner"

import asyncio
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

REQUIRED_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
    "Cross-Origin-Opener-Policy",
    "Cross-Origin-Resource-Policy",
    "Cross-Origin-Embedder-Policy",
    "X-Permitted-Cross-Domain-Policies"
]

HEADER_WEIGHTS = {
    "Strict-Transport-Security": 15,
    "Content-Security-Policy": 20,
    "X-Frame-Options": 10,
    "X-Content-Type-Options": 10,
    "Referrer-Policy": 10,
    "Permissions-Policy": 10,
    "Cross-Origin-Opener-Policy": 5,
    "Cross-Origin-Resource-Policy": 5,
    "Cross-Origin-Embedder-Policy": 5,
    "X-Permitted-Cross-Domain-Policies": 5
}

GRADE_THRESHOLDS = {"A+": 90, "A": 80, "B": 70, "C": 60, "D": 50, "F": 0}


def _analyze_csp(csp: str) -> Dict[str, Any]:
    """Analisis CSP, return findings."""
    issues = []
    if not csp:
        return {"present": False, "issues": ["CSP tidak ada"]}
    if "unsafe-inline" in csp:
        issues.append("unsafe-inline di script/style-src")
    if "unsafe-eval" in csp:
        issues.append("unsafe-eval di script-src")
    if "data:" in csp and "script-src" in csp:
        issues.append("data: di script-src")
    if "http:" in csp and "https:" not in csp:
        issues.append("HTTP di CSP tanpa HTTPS")
    if "frame-ancestors" not in csp:
        issues.append("frame-ancestors tidak diset")
    return {"present": True, "policy": csp[:200], "issues": issues}


def _analyze_hsts(hsts: str) -> Dict[str, Any]:
    """Analisis HSTS, return findings."""
    issues = []
    if not hsts:
        return {"present": False, "issues": ["HSTS tidak ada"]}
    max_age = 0
    for part in hsts.split(";"):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                max_age = int(part.split("=")[1])
            except Exception:
                pass
    if max_age < 31536000:
        issues.append(f"max-age terlalu pendek: {max_age} (< 1 tahun)")
    if "includeSubDomains" not in hsts:
        issues.append("includeSubDomains tidak diset")
    if "preload" not in hsts:
        issues.append("preload tidak diset")
    return {"present": True, "max_age": max_age, "issues": issues}


def _analyze_cookies(cookie_header: str) -> List[Dict[str, Any]]:
    """Audit cookie flags."""
    findings = []
    if not cookie_header:
        return findings
    for cookie in cookie_header.split(","):
        cookie = cookie.strip()
        name = cookie.split("=")[0] if "=" in cookie else cookie
        issues = []
        if "Secure" not in cookie:
            issues.append("tidak Secure")
        if "HttpOnly" not in cookie:
            issues.append("tidak HttpOnly")
        if "SameSite" not in cookie:
            issues.append("SameSite tidak diset")
        elif "SameSite=None" in cookie and "Secure" not in cookie:
            issues.append("SameSite=None tanpa Secure")
        if issues:
            findings.append({"name": name, "issues": issues, "raw": cookie[:100]})
    return findings


def _calculate_grade(present_headers: List[str]) -> str:
    """Hitung grade berdasarkan header yang hadir."""
    score = sum(HEADER_WEIGHTS.get(h, 0) for h in present_headers)
    for grade, threshold in GRADE_THRESHOLDS.items():
        if score >= threshold:
            return grade
    return "F"


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Scan header keamanan, grading, cookie audit, CSP/HSTS analysis."""
    start = time.time()
    url = target if target.startswith("http") else f"https://{target}"

    async with build_session(verify=False, timeout=10) as session:
        await random_delay(0.2)
        try:
            async with session.get(url, headers=get_stealth_headers(), timeout=10, allow_redirects=True) as resp:
                headers = {k: v for k, v in resp.headers.items()}
                body = await resp.read()
        except Exception as e:
            return {
                "module": MODULE_NAME,
                "target": target,
                "findings": [],
                "summary": f"Gagal koneksi: {e}",
                "severity": "error",
                "duration": round(time.time() - start, 2)
            }

    present = [h for h in REQUIRED_HEADERS if h in headers]
    missing = [h for h in REQUIRED_HEADERS if h not in headers]
    grade = _calculate_grade(present)

    findings = []
    for h in missing:
        findings.append({"header": h, "status": "missing", "severity": "medium"})

    csp_analysis = _analyze_csp(headers.get("Content-Security-Policy", ""))
    if not csp_analysis["present"]:
        findings.append({"header": "Content-Security-Policy", "status": "missing", "severity": "high"})
    elif csp_analysis["issues"]:
        for issue in csp_analysis["issues"]:
            findings.append({"header": "Content-Security-Policy", "status": "weak", "detail": issue, "severity": "medium"})

    hsts_analysis = _analyze_hsts(headers.get("Strict-Transport-Security", ""))
    if not hsts_analysis["present"]:
        findings.append({"header": "Strict-Transport-Security", "status": "missing", "severity": "high"})
    elif hsts_analysis["issues"]:
        for issue in hsts_analysis["issues"]:
            findings.append({"header": "Strict-Transport-Security", "status": "weak", "detail": issue, "severity": "low"})

    cookie_findings = _analyze_cookies(headers.get("Set-Cookie", ""))
    for cf in cookie_findings:
        findings.append({"header": "Set-Cookie", "status": "weak", "cookie": cf["name"], "issues": cf["issues"], "severity": "medium"})

    info_findings = []
    for h in present:
        if h not in ["Content-Security-Policy", "Strict-Transport-Security"]:
            info_findings.append({"header": h, "value": headers[h][:200], "status": "present", "severity": "info"})

    all_findings = findings + info_findings
    max_sev = "error"
    for f in all_findings:
        if f["severity"] == "high":
            max_sev = "high"
        elif f["severity"] == "medium" and max_sev not in ("high", "error"):
            max_sev = "medium"
        elif f["severity"] == "low" and max_sev not in ("high", "medium", "error"):
            max_sev = "low"
        elif f["severity"] == "info" and max_sev not in ("high", "medium", "low", "error"):
            max_sev = "info"

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": all_findings,
        "summary": f"Grade {grade} — {len(present)}/{len(REQUIRED_HEADERS)} header hadir",
        "severity": max_sev if all_findings else "info",
        "duration": round(duration, 2),
        "grade": grade,
        "score": sum(HEADER_WEIGHTS.get(h, 0) for h in present),
        "headers_present": present,
        "headers_missing": missing,
        "csp_analysis": csp_analysis,
        "hsts_analysis": hsts_analysis,
        "cookie_audit": cookie_findings
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))