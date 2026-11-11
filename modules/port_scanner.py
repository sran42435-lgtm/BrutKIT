# modules/port_scanner.py — async TCP connect, TOP_PORTS ~50, banner grab
MODULE_NAME = "port_scanner"

import asyncio
import socket
import time
from typing import Dict, Any, List

__all__ = ["MODULE_NAME", "scan"]

TOP_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 465, 587,
    993, 995, 1433, 1521, 1723, 2049, 2375, 2376, 3306, 3389, 5432, 5672,
    5900, 5984, 6379, 6667, 8000, 8080, 8081, 8443, 8888, 9000, 9090, 9200,
    9418, 10000, 11211, 27017, 27018, 27019, 50000, 50001
]

BANNER_PORTS = {21, 22, 25, 80, 110, 143, 443, 465, 587, 993, 995, 1433, 1521, 3306, 5432, 5900, 6379, 8080, 8443, 9200, 11211, 27017}


async def _connect_port(target: str, port: int, timeout: float = 3.0) -> Dict[str, Any]:
    """Mencoba koneksi TCP ke satu port, return dict hasil."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target, port),
            timeout=timeout
        )
        banner = b""
        if port in BANNER_PORTS:
            try:
                writer.write(b"\r\n")
                await writer.drain()
                banner = await asyncio.wait_for(reader.read(1024), timeout=2.0)
            except Exception:
                pass
        writer.close()
        await writer.wait_closed()
        return {"port": port, "open": True, "banner": banner.decode(errors="ignore").strip()[:200]}
    except Exception:
        return {"port": port, "open": False, "banner": ""}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Scan port TCP async pada target, return findings dan summary."""
    start = time.time()
    ports = config.get("ports", TOP_PORTS)
    timeout = config.get("timeout", 3.0)
    concurrency = config.get("concurrency", 100)

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(port: int):
        async with semaphore:
            return await _connect_port(target, port, timeout)

    tasks = [_bounded(p) for p in ports]
    results = await asyncio.gather(*tasks)

    open_ports = [r for r in results if r["open"]]
    findings = []
    for r in open_ports:
        sev = "high" if r["port"] in {21, 22, 23, 25, 110, 139, 445, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 27017} else "medium"
        findings.append({
            "port": r["port"],
            "service": socket.getservbyport(r["port"], "tcp") if r["port"] < 1024 else "unknown",
            "banner": r["banner"],
            "severity": sev
        })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(open_ports)}/{len(ports)} port terbuka",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else ("medium" if findings else "info"),
        "duration": round(duration, 2),
        "scanned_ports": len(ports),
        "open_count": len(open_ports)
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))