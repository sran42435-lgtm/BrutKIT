# modules/cors_checker.py — reflect/null origin/6 bypass vectors/preflight OPTIONS/vary
MODULE_NAME = "cors_checker"

import asyncio
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

BYPASS_VECTORS = [
    "https://evil.com",
    "https://sub.evil.com",
    "null",
    "https://example.com.evil.com",
    "https://evil.com.example.com",
    "https://example.com",
    ""
]

async def _test_origin(url: str, session, origin: str, method: str = "GET") -> Dict[str, Any]:
    """Test CORS dengan origin tertentu."""
    headers = get_stealth_headers()
    headers["Origin"] = origin
    if method == "OPTIONS":
        headers["Access-Control-Request-Method"] = "POST"
        headers["Access-Control-Request-Headers"] = "Content-Type"

    try:
        await random_delay(0.1)
        if method == "OPTIONS":
            async with session.options(url, headers=headers, timeout=8, allow_redirects=False) as resp:
                return _parse_cors_response(resp, origin)
        else:
            async with session.get(url, headers=headers, timeout=8, allow_redirects=False) as resp:
                return _parse_cors_response(resp, origin)
    except Exception as e:
        return {"origin": origin, "error": str(e)}


def _parse_cors_response(resp, origin: str) -> Dict[str, Any]:
    """Parse response CORS headers."""
    acao = resp.headers.get("Access-Control-Allow-Origin", "")
    acac = resp.headers.get("Access-Control-Allow-Credentials", "")
    acam = resp.headers.get("Access-Control-Allow-Methods", "")
    acah = resp.headers.get("Access-Control-Allow-Headers", "")
    acma = resp.headers.get("Access-Control-Max-Age", "")
    vary = resp.headers.get("Vary", "")

    reflected = acao == origin
    wildcard = acao == "*"
    null_origin = origin == "null" and acao == "null"

    return {
        "origin": origin,
        "acao": acao,
        "acac": acac,
        "acam": acam,
        "acah": acah,
        "acma": acma,
        "vary": vary,
        "reflected": reflected,
        "wildcard": wildcard,
        "null_origin": null_origin,
        "status": resp.status
    }


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """CORS checker: reflect/null/6 bypass vectors + preflight OPTIONS + Vary."""
    start = time.time()
    url = target if target.startswith("http") else f"https://{target}"
    test_origins = config.get("origins", BYPASS_VECTORS)

    async with build_session(verify=False, timeout=10) as session:
        tasks = [_test_origin(url, session, o) for o in test_origins]
        get_results = await asyncio.gather(*tasks)

        options_tasks = [_test_origin(url, session, o, "OPTIONS") for o in test_origins]
        options_results = await asyncio.gather(*options_tasks)

    findings = []
    for i, origin in enumerate(test_origins):
        gr = get_results[i]
        or_ = options_results[i]

        if "error" in gr:
            findings.append({"origin": origin, "type": "error", "detail": gr["error"], "severity": "error"})
            continue

        issues = []
        if gr["reflected"] and gr["acac"].lower() == "true":
            issues.append("Reflect origin + credentials=true (critical)")
        elif gr["reflected"] and origin not in ["", "null"]:
            issues.append(f"Reflect arbitrary origin: {origin}")
        if gr["wildcard"] and gr["acac"].lower() == "true":
            issues.append("Wildcard + credentials=true (invalid per spec)")
        if gr["null_origin"]:
            issues.append("Null origin accepted")
        if gr["vary"] and "origin" not in gr["vary"].lower():
            issues.append("Vary header missing 'Origin'")
        if not gr["vary"] and (gr["reflected"] or gr["wildcard"]):
            issues.append("Vary header missing entirely")

        for issue in issues:
            sev = "critical" if "critical" in issue.lower() or "credentials=true" in issue.lower() else "high"
            if "Vary" in issue:
                sev = "medium"
            findings.append({
                "origin": origin,
                "type": "cors_misconfig",
                "detail": issue,
                "acao": gr["acao"],
                "acac": gr["acac"],
                "vary": gr["vary"],
                "severity": sev
            })

        if not issues:
            findings.append({
                "origin": origin,
                "type": "cors_ok",
                "detail": "Tidak ada misconfig terdeteksi",
                "acao": gr["acao"],
                "severity": "info"
            })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len([f for f in findings if f['severity'] in ['critical','high']])} misconfig CORS dari {len(test_origins)} origin diuji",
        "severity": "critical" if any(f["severity"] == "critical" for f in findings) else ("high" if any(f["severity"] == "high" for f in findings) else ("medium" if any(f["severity"] == "medium" for f in findings) else "info")),
        "duration": round(duration, 2),
        "tested_origins": len(test_origins),
        "get_results": get_results,
        "options_results": options_results
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))