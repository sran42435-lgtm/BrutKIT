# modules/subdom_scanner.py — brute ~170 word, wildcard detect, http_check
MODULE_NAME = "subdom_scanner"

import asyncio
import time
from typing import Dict, Any, List, Set
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

SUBDOMAIN_WORDLIST = [
    "www", "mail", "ftp", "localhost", "webmail", "smtp", "pop", "ns1", "webdisk",
    "ns2", "cpanel", "whm", "autodiscover", "autoconfig", "m", "imap", "test",
    "ns", "blog", "pop3", "dev", "www2", "admin", "forum", "news", "vpn", "ns3",
    "mail2", "new", "mysql", "old", "lists", "support", "mobile", "mx", "static",
    "docs", "beta", "shop", "sql", "secure", "demo", "cp", "calendar", "wiki",
    "web", "media", "email", "images", "img", "www1", "intranet", "portal",
    "video", "sip", "dns2", "api", "cdn", "stage", "staging", "app", "chat",
    "help", "ssl", "sftp", "backup", "git", "svn", "hg", "cvs", "jenkins",
    "ci", "build", "deploy", "prod", "production", "qa", "test2", "test3",
    "dev2", "dev3", "uat", "preprod", "sandbox", "playground", "lab", "internal",
    "corp", "internal2", "office", "remote", "rdp", "ssh", "vpn2", "vpn3",
    "gateway", "proxy", "fw", "firewall", "ids", "ips", "waf", "lb", "loadbalancer",
    "haproxy", "nginx", "apache", "iis", "tomcat", "jboss", "weblogic",
    "websphere", "glassfish", "wildfly", "jetty", "node", "nodejs", "python",
    "django", "flask", "rails", "laravel", "symfony", "codeigniter", "yii",
    "zend", "cakephp", "magento", "wordpress", "wp", "drupal", "joomla",
    "phpmyadmin", "pma", "adminer", "db", "database", "redis", "memcached",
    "rabbitmq", "kafka", "zookeeper", "consul", "vault", "grafana", "prometheus",
    "kibana", "elasticsearch", "logstash", "jenkins2", "gitlab", "github",
    "bitbucket", "jira", "confluence", "bamboo", "crowd", "fisheye", "crucible",
    "nexus", "artifactory", "sonarqube", "registry", "harbor", "rancher",
    "kubernetes", "k8s", "openshift", "docker", "swarm", "mesos", "marathon",
    "nomad", "vault2", "consul2", "etcd", "zookeeper2", "kafka2"
]

COMMON_TLDS = ["com", "net", "org", "io", "co", "dev", "app", "cloud", "ai", "tech"]


async def _check_wildcard(domain: str, session) -> bool:
    """Deteksi wildcard DNS dengan query subdomain acak."""
    import random
    import string
    random_sub = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    test_domain = f"{random_sub}.{domain}"
    try:
        async with session.get(f"http://{test_domain}", timeout=5, allow_redirects=False) as resp:
            return resp.status < 400
    except Exception:
        return False


async def _http_check(subdomain: str, session, wildcard_detected: bool) -> Dict[str, Any]:
    """Cek HTTP/HTTPS pada subdomain, return info."""
    for scheme in ("https", "http"):
        url = f"{scheme}://{subdomain}"
        try:
            await random_delay(0.1)
            async with session.get(url, headers=get_stealth_headers(), timeout=8, allow_redirects=True) as resp:
                body = await resp.read()
                return {
                    "subdomain": subdomain,
                    "url": str(resp.url),
                    "status": resp.status,
                    "server": resp.headers.get("Server", ""),
                    "title": _extract_title(body),
                    "wildcard": wildcard_detected
                }
        except Exception:
            continue
    return {"subdomain": subdomain, "url": "", "status": 0, "server": "", "title": "", "wildcard": wildcard_detected}


def _extract_title(body: bytes) -> str:
    """Ekstrak title dari HTML body."""
    try:
        text = body.decode(errors="ignore")
        import re
        match = re.search(r"<title[^>]*>([^<]+)</title>", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    except Exception:
        return ""


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Brute force subdomain dengan wordlist ~170, wildcard detect, HTTP check."""
    start = time.time()
    domain = target.replace("https://", "").replace("http://", "").strip("/")
    wordlist = config.get("wordlist", SUBDOMAIN_WORDLIST)
    concurrency = config.get("concurrency", 50)

    async with build_session(verify=False, timeout=10) as session:
        wildcard = await _check_wildcard(domain, session)
        semaphore = asyncio.Semaphore(concurrency)

        async def _check_one(sub: str):
            async with semaphore:
                full = f"{sub}.{domain}"
                return await _http_check(full, session, wildcard)

        tasks = [_check_one(w) for w in wordlist]
        results = await asyncio.gather(*tasks)

    alive = [r for r in results if r["status"] > 0]
    findings = []
    for r in alive:
        sev = "medium" if r["wildcard"] else "info"
        findings.append({
            "subdomain": r["subdomain"],
            "url": r["url"],
            "status": r["status"],
            "server": r["server"],
            "title": r["title"],
            "severity": sev
        })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(alive)}/{len(wordlist)} subdomain hidup",
        "severity": "medium" if findings else "info",
        "duration": round(duration, 2),
        "tested": len(wordlist),
        "wildcard_detected": wildcard
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))