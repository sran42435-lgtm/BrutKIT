# modules/dns_recon.py — A/AAAA/MX/NS/TXT/SOA/CNAME + SPF/DMARC + AXFR (dnspython)
MODULE_NAME = "dns_recon"

import asyncio
import time
from typing import Dict, Any, List
import dns.resolver
import dns.zone
import dns.query
import dns.exception

__all__ = ["MODULE_NAME", "scan"]

RECORD_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]


def _resolve_blocking(domain: str, rtype: str) -> List[str]:
    """Blocking DNS resolve via dnspython."""
    try:
        answers = dns.resolver.resolve(domain, rtype)
        return [str(r) for r in answers]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers, dns.exception.DNSException):
        return []


def _check_axfr(domain: str, ns_servers: List[str]) -> List[Dict[str, Any]]:
    """Coba zone transfer AXFR pada setiap NS."""
    results = []
    for ns in ns_servers:
        try:
            zone = dns.zone.from_xfr(dns.query.xfr(ns, domain, timeout=10))
            records = []
            for name, node in zone.items():
                for rdataset in node:
                    records.append(f"{name} {rdataset}")
            results.append({"ns": ns, "success": True, "records": records[:100]})
        except Exception as e:
            results.append({"ns": ns, "success": False, "error": str(e)})
    return results


def _parse_spf(txt_records: List[str]) -> Dict[str, Any]:
    """Parse SPF dari TXT records."""
    for txt in txt_records:
        if txt.startswith("v=spf1"):
            mechanisms = txt.split()
            return {"present": True, "raw": txt, "mechanisms": mechanisms}
    return {"present": False, "raw": "", "mechanisms": []}


def _parse_dmarc(txt_records: List[str]) -> Dict[str, Any]:
    """Parse DMARC dari TXT records (_dmarc)."""
    for txt in txt_records:
        if txt.startswith("v=DMARC1"):
            tags = {}
            for part in txt.split(";"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    tags[k] = v
            return {"present": True, "raw": txt, "tags": tags}
    return {"present": False, "raw": "", "tags": {}}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """DNS recon lengkap: A/AAAA/MX/NS/TXT/SOA/CNAME + SPF/DMARC + AXFR."""
    start = time.time()
    domain = target.replace("https://", "").replace("http://", "").strip("/")

    findings = []
    all_records = {}

    for rtype in RECORD_TYPES:
        records = await asyncio.to_thread(_resolve_blocking, domain, rtype)
        if records:
            all_records[rtype] = records
            for r in records:
                sev = "info"
                if rtype in ["MX", "NS", "SOA"]:
                    sev = "low"
                findings.append({
                    "type": rtype,
                    "value": r,
                    "severity": sev
                })

    spf = _parse_spf(all_records.get("TXT", []))
    if not spf["present"]:
        findings.append({"type": "SPF", "value": "tidak ditemukan", "severity": "medium"})
    else:
        findings.append({"type": "SPF", "value": spf["raw"][:200], "severity": "info"})

    dmarc_domain = f"_dmarc.{domain}"
    dmarc_txt = await asyncio.to_thread(_resolve_blocking, dmarc_domain, "TXT")
    dmarc = _parse_dmarc(dmarc_txt)
    if not dmarc["present"]:
        findings.append({"type": "DMARC", "value": "tidak ditemukan", "severity": "medium"})
    else:
        findings.append({"type": "DMARC", "value": dmarc["raw"][:200], "severity": "info"})

    ns_servers = all_records.get("NS", [])
    axfr_results = []
    if ns_servers:
        axfr_results = await asyncio.to_thread(_check_axfr, domain, ns_servers)
        for axfr in axfr_results:
            if axfr["success"]:
                findings.append({"type": "AXFR", "value": f"Zone transfer berhasil di {axfr['ns']}", "severity": "high", "records_count": len(axfr["records"])})

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(all_records)} tipe record, SPF: {'ya' if spf['present'] else 'tidak'}, DMARC: {'ya' if dmarc['present'] else 'tidak'}, AXFR: {'ya' if any(a['success'] for a in axfr_results) else 'tidak'}",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else ("medium" if any(f["severity"] == "medium" for f in findings) else "info"),
        "duration": round(duration, 2),
        "records": all_records,
        "spf": spf,
        "dmarc": dmarc,
        "axfr": axfr_results
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))