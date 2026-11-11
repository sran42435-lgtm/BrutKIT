# stealth_servant.py — util stealth (DI ROOT): get_stealth_headers, random_delay, build_session
import asyncio
import random
import ssl
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import ClientTimeout, TCPConnector

__all__ = ["get_stealth_headers", "random_delay", "build_session"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def get_stealth_headers() -> Dict[str, str]:
    """Return headers dengan User-Agent acak dari pool 18 UA browser."""
    ua = random.choice(USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


async def random_delay(base: float) -> None:
    """Tidur acak antara base*0.5 sampai base*1.5 detik."""
    delay = base * random.uniform(0.5, 1.5)
    await asyncio.sleep(delay)


@asynccontextmanager
async def build_session(
    verify: bool = False,
    timeout: int = 10,
    proxy: Optional[str] = None,
    **kwargs
):
    """Build aiohttp.ClientSession MENTAH dengan SSL verify option."""
    ssl_context = None
    if not verify:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    connector = TCPConnector(
        ssl=ssl_context,
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        enable_cleanup_closed=True
    )

    client_timeout = ClientTimeout(total=timeout, connect=5, sock_read=timeout, sock_connect=5)

    session = aiohttp.ClientSession(
        connector=connector,
        timeout=client_timeout,
        trust_env=True,
        **kwargs
    )

    try:
        yield session
    finally:
        await session.close()
        await connector.close()


if __name__ == "__main__":
    import json
    print("Headers:", json.dumps(get_stealth_headers(), indent=2))
    print("UA Pool size:", len(USER_AGENTS))

    async def _test():
        await random_delay(0.1)
        print("Delay OK")
        async with build_session(verify=False, timeout=5) as s:
            async with s.get("https://httpbin.org/get", headers=get_stealth_headers()) as resp:
                print("Session OK, status:", resp.status)

    asyncio.run(_test())