# modules/dir_buster.py — ~180 path, baseline soft-404 (3x random), klasifikasi status
MODULE_NAME = "dir_buster"

import asyncio
import random
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

COMMON_PATHS = [
    "admin", "administrator", "login", "wp-admin", "wp-login.php", "phpmyadmin", "pma",
    "adminer", "dashboard", "panel", "controlpanel", "cpanel", "manager", "management",
    "console", "backend", "portal", "cms", "api", "v1", "v2", "v3", "rest", "graphql",
    "swagger", "docs", "documentation", "readme", "README.md", "CHANGELOG", "CHANGELOG.md",
    "LICENSE", "LICENSE.txt", "composer.json", "package.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "setup.py", "pyproject.toml", "Dockerfile", "docker-compose.yml",
    ".env", ".env.example", ".env.local", ".env.production", ".env.development", ".env.staging",
    "config", "config.php", "config.json", "config.yml", "config.yaml", "settings.php",
    "settings.json", "settings.yml", "settings.yaml", "local.php", "local.json",
    "database.php", "db.php", "db.json", "db.yml", "database.yml", "database.yaml",
    "backup", "backups", "dump", "dump.sql", "backup.sql", "backup.zip", "backup.tar.gz",
    "site.zip", "site.tar.gz", "www.zip", "www.tar.gz", "public.zip", "public.tar.gz",
    "git", ".git", ".git/config", ".git/HEAD", ".git/logs/HEAD", ".git/index",
    "svn", ".svn", "hg", ".hg", "bzr", ".bzr", "cvs", "CVS",
    "robots.txt", "sitemap.xml", "sitemap_index.xml", "humans.txt", "security.txt",
    ".well-known", ".well-known/security.txt", ".well-known/change-password",
    "server-status", "server-info", "status", "health", "healthz", "ready", "live",
    "metrics", "prometheus", "actuator", "actuator/health", "actuator/info",
    "debug", "debug.php", "test.php", "info.php", "phpinfo.php", "i.php",
    "trace.axd", "elmah.axd", "web.config", "web.config.bak", "web.config.backup",
    "htaccess", ".htaccess", ".htaccess.bak", ".htpasswd", ".htpasswd.bak",
    "nginx.conf", "apache.conf", "httpd.conf", "my.cnf", "postgresql.conf",
    "redis.conf", "mongod.conf", "cassandra.yaml", "elasticsearch.yml",
    "kibana.yml", "logstash.yml", "filebeat.yml", "metricbeat.yml",
    "jenkins", "jenkins/", "gitlab", "gitlab/", "gitea", "gitea/", "drone", "drone/",
    "harbor", "harbor/", "registry", "registry/", "nexus", "nexus/",
    "sonarqube", "sonarqube/", "grafana", "grafana/", "prometheus", "prometheus/",
    "alertmanager", "alertmanager/", "loki", "loki/", "tempo", "tempo/",
    "jaeger", "jaeger/", "zipkin", "zipkin/", "thanos", "thanos/",
    "kibana", "kibana/", "elasticsearch", "elasticsearch/", "logstash", "logstash/",
    "filebrowser", "filebrowser/", "netdata", "netdata/", "cockpit", "cockpit/",
    "portainer", "portainer/", "rancher", "rancher/", "longhorn", "longhorn/",
    "argocd", "argocd/", "flux", "flux/", "tekton", "tekton/",
    "vault", "vault/", "consul", "consul/", "nomad", "nomad/",
    "traefik", "traefik/", "nginx", "nginx/", "haproxy", "haproxy/",
    "envoy", "envoy/", "istio", "istio/", "linkerd", "linkerd/",
    "cert-manager", "cert-manager/", "external-dns", "external-dns/",
    "metallb", "metallb/", "calico", "calico/", "cilium", "cilium/",
    "coredns", "coredns/", "kube-dns", "kube-dns/", "kube-proxy", "kube-proxy/"
]

STATUS_GROUPS = {
    "success": (200, 299),
    "redirect": (300, 399),
    "client_error": (400, 499),
    "server_error": (500, 599)
}


def _classify_status(status: int) -> str:
    """Klasifikasikan status HTTP."""
    for group, (low, high) in STATUS_GROUPS.items():
        if low <= status <= high:
            return group
    return "unknown"


async def _get_baseline_404(session, base_url: str) -> List[Dict[str, Any]]:
    """Dapatkan baseline soft-404 dengan 3 request acak."""
    baselines = []
    for _ in range(3):
        random_path = f"__notfound_{random.randint(100000, 999999)}__"
        url = f"{base_url.rstrip('/')}/{random_path}"
        try:
            await random_delay(0.05)
            async with session.get(url, headers=get_stealth_headers(), timeout=5, allow_redirects=False) as resp:
                body = await resp.read()
                baselines.append({
                    "status": resp.status,
                    "length": len(body),
                    "server": resp.headers.get("Server", "")
                })
        except Exception:
            baselines.append({"status": 0, "length": 0, "server": ""})
    return baselines


def _is_soft_404(resp_status: int, resp_length: int, baselines: List[Dict]) -> bool:
    """Cek apakah response mirip soft-404."""
    if resp_status == 404:
        return True
    for b in baselines:
        if b["status"] == resp_status and abs(b["length"] - resp_length) < 50:
            return True
    return False


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Directory brute force dengan baseline soft-404 dan klasifikasi status."""
    start = time.time()
    base_url = target if target.startswith("http") else f"https://{target}"
    base_url = base_url.rstrip("/")
    paths = config.get("paths", COMMON_PATHS)
    concurrency = config.get("concurrency", 30)
    extensions = config.get("extensions", ["", ".php", ".html", ".js", ".json", ".bak", ".old", ".backup", ".txt"])

    async with build_session(verify=False, timeout=10) as session:
        baselines = await _get_baseline_404(session, base_url)
        semaphore = asyncio.Semaphore(concurrency)

        async def _check_path(path: str):
            async with semaphore:
                results = []
                for ext in extensions:
                    test_path = f"{path}{ext}" if not path.endswith(ext) else path
                    url = f"{base_url}/{test_path}"
                    try:
                        await random_delay(0.1)
                        async with session.get(url, headers=get_stealth_headers(), timeout=8, allow_redirects=False) as resp:
                            body = await resp.read()
                            status = resp.status
                            group = _classify_status(status)
                            is_404 = _is_soft_404(status, len(body), baselines)
                            if not is_404 or status != 404:
                                results.append({
                                    "path": test_path,
                                    "url": str(resp.url),
                                    "status": status,
                                    "group": group,
                                    "length": len(body),
                                    "server": resp.headers.get("Server", ""),
                                    "content_type": resp.headers.get("Content-Type", ""),
                                    "soft_404": is_404
                                })
                    except Exception:
                        pass
                return results

        tasks = [_check_path(p) for p in paths]
        all_results = await asyncio.gather(*tasks)

    findings = []
    for rlist in all_results:
        for r in rlist:
            sev = "high" if r["status"] == 200 and any(kw in r["path"] for kw in [".env", ".git", "config", "backup", "dump", "admin", "login"]) else "info"
            if r["group"] == "server_error":
                sev = "medium"
            elif r["group"] == "redirect":
                sev = "low"
            findings.append({
                "path": r["path"],
                "url": r["url"],
                "status": r["status"],
                "group": r["group"],
                "length": r["length"],
                "server": r["server"],
                "content_type": r["content_type"],
                "severity": sev
            })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(findings)} path ditemukan dari {len(paths) * len(extensions)} diuji",
        "severity": "high" if any(f["severity"] == "high" for f in findings) else ("medium" if any(f["severity"] == "medium" for f in findings) else "info"),
        "duration": round(duration, 2),
        "tested": len(paths) * len(extensions),
        "baseline_samples": len(baselines)
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))