# modules/shodan_scanner.py — pasif via REST api.shodan.io, key dari config/env SHODAN_API_KEY, handle 401/404
MODULE_NAME = "shodan_scanner"

import asyncio
import os
import time
from typing import Dict, Any, List
from stealth_servant import build_session

__all__ = ["MODULE_NAME", "scan"]

SHODAN_API_URL = "https://api.shodan.io/shodan/host/"


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Shodan scanner pasif via REST API, key dari config/env SHODAN_API_KEY."""
    start = time.time()
    api_key = config.get("api_key") or os.getenv("SHODAN_API_KEY")

    if not api_key:
        return {
            "module": MODULE_NAME,
            "target": target,
            "findings": [{"type": "config_error", "detail": "SHODAN_API_KEY tidak diset (config/env)", "severity": "error"}],
            "summary": "API key tidak dikonfigurasi",
            "severity": "error",
            "duration": round(time.time() - start, 2)
        }

    domain = target.replace("https://", "").replace("http://", "").strip("/")

    import socket
    try:
        ips = await asyncio.to_thread(socket.gethostbyname_ex, domain)
        ip_list = ips[2]
    except Exception as e:
        return {
            "module": MODULE_NAME,
            "target": target,
            "findings": [{"type": "dns_error", "detail": f"Gagal resolve domain: {e}", "severity": "error"}],
            "summary": "Gagal resolve DNS",
            "severity": "error",
            "duration": round(time.time() - start, 2)
        }

    async with build_session(verify=True, timeout=15) as session:
        all_findings = []
        for ip in ip_list[:5]:
            url = f"{SHODAN_API_URL}{ip}?key={api_key}"
            try:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 401:
                        all_findings.append({"ip": ip, "type": "auth_error", "detail": "Invalid API key (401)", "severity": "error"})
                        break
                    elif resp.status == 404:
                        all_findings.append({"ip": ip, "type": "not_found", "detail": "IP tidak ditemukan di Shodan", "severity": "info"})
                        continue
                    elif resp.status != 200:
                        all_findings.append({"ip": ip, "type": "http_error", "detail": f"HTTP {resp.status}", "severity": "medium"})
                        continue

                    data = await resp.json()
                    ports = data.get("ports", [])
                    vulns = data.get("vulns", [])
                    hostnames = data.get("hostnames", [])
                    org = data.get("org", "")
                    isp = data.get("isp", "")
                    country = data.get("country_name", "")
                    city = data.get("city", "")
                    os_info = data.get("os", "")
                    tags = data.get("tags", [])

                    for port in ports:
                        port_data = data.get("data", [])
                        for pd in port_data:
                            if pd.get("port") == port:
                                all_findings.append({
                                    "ip": ip,
                                    "port": port,
                                    "protocol": pd.get("transport", "tcp"),
                                    "service": pd.get("product", ""),
                                    "version": pd.get("version", ""),
                                    "banner": pd.get("data", "")[:200],
                                    "severity": "info"
                                })

                    if vulns:
                        for vuln in vulns:
                            all_findings.append({
                                "ip": ip,
                                "type": "vulnerability",
                                "cve": vuln,
                                "severity": "high"
                            })

                    all_findings.append({
                        "ip": ip,
                        "type": "host_info",
                        "hostnames": hostnames,
                        "org": org,
                        "isp": isp,
                        "country": country,
                        "city": city,
                        "os": os_info,
                        "tags": tags,
                        "open_ports": ports,
                        "severity": "info"
                    })

            except Exception as e:
                all_findings.append({"ip": ip, "type": "error", "detail": str(e), "severity": "error"})

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": all_findings,
        "summary": f"{len(ip_list)} IP, {len([f for f in all_findings if f.get('type') == 'vulnerability'])} CVE, {len([f for f in all_findings if 'port' in f])} port terbuka",
        "severity": "high" if any(f["severity"] == "high" for f in all_findings) else ("error" if any(f["severity"] == "error" for f in all_findings) else "info"),
        "duration": round(duration, 2),
        "resolved_ips": ip_list,
        "api_key_used": _mask_key(api_key)
    }


def _mask_key(key: str) -> str:
    """Mask API key."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "****" + key[-4:]


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))