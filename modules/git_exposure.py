# modules/git_exposure.py — ~50 path + validasi signature konten (magic bytes dll) + git deep recon (logs/HEAD -> dev emails) + extract secrets masked
MODULE_NAME = "git_exposure"

import asyncio
import re
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

GIT_PATHS = [
    ".git", ".git/config", ".git/HEAD", ".git/index", ".git/logs/HEAD",
    ".git/logs/refs/heads/master", ".git/logs/refs/heads/main",
    ".git/refs/heads/master", ".git/refs/heads/main",
    ".git/refs/tags/", ".git/packed-refs", ".git/objects/info/packs",
    ".git/description", ".git/hooks/", ".git/info/exclude",
    ".git/FETCH_HEAD", ".git/ORIG_HEAD", ".git/MERGE_HEAD",
    ".git/config.bak", ".git/config.backup", ".git/config.old",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".github/", ".github/workflows/", ".gitlab/", ".gitlab-ci.yml",
    "git.log", "git.log.1", ".git/log", "git/config",
    ".hg", ".hg/store", ".hg/requires", ".hg/branch", ".hg/dirstate",
    ".svn", ".svn/entries", ".svn/wc.db", ".svn/format",
    "bzr", ".bzr", ".bzr/repository", ".bzr/checkout",
    "cvs", "CVS", "CVS/Entries", "CVS/Root", "CVS/Repository"
]

SECRET_PATTERNS = {
    "AWS Access Key": r"AKIA[0-9A-Z]{16}",
    "AWS Secret Key": r"[0-9a-zA-Z/+]{40}",
    "GitHub Token": r"ghp_[0-9a-zA-Z]{36}",
    "GitHub OAuth": r"gho_[0-9a-zA-Z]{36}",
    "GitHub App": r"ghs_[0-9a-zA-Z]{36}",
    "GitHub Refresh": r"ghr_[0-9a-zA-Z]{36}",
    "GitLab Token": r"glpat-[0-9a-zA-Z\-_]{20}",
    "Bitbucket Token": r"x-token-[0-9a-zA-Z]{32}",
    "Slack Token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
    "Discord Token": r"[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27}",
    "Telegram Bot": r"\d{8,10}:[A-Za-z0-9_-]{35}",
    "Google API Key": r"AIza[0-9A-Za-z\-_]{35}",
    "Firebase Key": r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}",
    "Stripe Key": r"sk_live_[0-9a-zA-Z]{24}",
    "PayPal Token": r"access_token\$production\$[0-9a-z]{16}\$[0-9a-f]{32}",
    "JWT Token": r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
    "Private Key": r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----",
    "SSH Key": r"ssh-(rsa|dss|ed25519) AAAA[0-9A-Za-z+/]+[=]{0,3}",
    "Generic Secret": r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"']?[0-9a-zA-Z\-_]{20,}[\"']?"
}

MAGIC_BYTES = {
    "git": b"DIRC",  # index file
    "git_pack": b"PACK",
    "git_loose": b"\x1f\x8b",  # gzip
    "zip": b"PK\x03\x04",
    "gzip": b"\x1f\x8b",
    "tar": b"ustar",
    "sqlite": b"SQLite format 3",
    "pdf": b"%PDF",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "elf": b"\x7fELF"
}


def _mask_secret(secret: str) -> str:
    """Mask secret: 4 awal + **** + 4 akhir."""
    if len(secret) <= 8:
        return "*" * len(secret)
    return secret[:4] + "****" + secret[-4:]


def _extract_secrets(content: str) -> List[Dict[str, Any]]:
    """Ekstrak dan mask secret dari konten."""
    found = []
    for name, pattern in SECRET_PATTERNS.items():
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            masked = _mask_secret(match)
            found.append({"type": name, "masked": masked, "length": len(match)})
    return found


def _parse_git_log_head(content: str) -> List[Dict[str, str]]:
    """Parse .git/logs/HEAD untuk email developer."""
    emails = []
    for line in content.split("\n"):
        match = re.search(r"<([^>]+@[^>]+)>", line)
        if match:
            email = match.group(1)
            if email not in [e["email"] for e in emails]:
                emails.append({"email": email, "source": "git_logs_HEAD"})
    return emails


def _parse_git_config(content: str) -> Dict[str, Any]:
    """Parse .git/config untuk remote URL dan user."""
    info = {"remotes": [], "user": {}}
    remote_urls = re.findall(r"url\s*=\s*(.+)", content)
    info["remotes"] = remote_urls
    user_name = re.search(r"name\s*=\s*(.+)", content)
    user_email = re.search(r"email\s*=\s*(.+)", content)
    if user_name:
        info["user"]["name"] = user_name.group(1).strip()
    if user_email:
        info["user"]["email"] = user_email.group(1).strip()
    return info


def _validate_content_signature(content: bytes, path: str) -> Dict[str, Any]:
    """Validasi signature konten (magic bytes)."""
    for name, magic in MAGIC_BYTES.items():
        if content.startswith(magic):
            return {"valid": True, "type": name, "magic": magic.hex()}
    if path.endswith(".git/config") and b"[core]" in content:
        return {"valid": True, "type": "git_config", "magic": "text"}
    if path.endswith(".git/HEAD") and (content.startswith(b"ref:") or len(content) == 41):
        return {"valid": True, "type": "git_head", "magic": "text"}
    if path.endswith(".gitignore") or path.endswith(".gitmodules"):
        return {"valid": True, "type": "git_text", "magic": "text"}
    return {"valid": False, "type": "unknown", "magic": content[:4].hex() if content else ""}


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Git exposure: ~50 path + signature validation + deep recon + secrets masked."""
    start = time.time()
    base_url = target if target.startswith("http") else f"https://{target}"
    base_url = base_url.rstrip("/")
    paths = config.get("paths", GIT_PATHS)

    async with build_session(verify=False, timeout=10) as session:
        semaphore = asyncio.Semaphore(config.get("concurrency", 20))
        findings = []
        all_emails = []
        all_secrets = []
        git_config_info = {}

        async def _check_path(path: str):
            async with semaphore:
                url = f"{base_url}/{path}"
                try:
                    await random_delay(0.1)
                    async with session.get(url, headers=get_stealth_headers(), timeout=8, allow_redirects=False) as resp:
                        if resp.status == 200:
                            content_bytes = await resp.read()
                            content = content_bytes.decode(errors="ignore")
                            sig = _validate_content_signature(content_bytes, path)
                            if sig["valid"]:
                                secrets = _extract_secrets(content)
                                emails = _parse_git_log_head(content) if "logs/HEAD" in path else []
                                config_info = _parse_git_config(content) if path.endswith(".git/config") else {}

                                finding = {
                                    "path": path,
                                    "url": str(resp.url),
                                    "status": resp.status,
                                    "size": len(content_bytes),
                                    "content_type": resp.headers.get("Content-Type", ""),
                                    "signature": sig,
                                    "secrets": secrets,
                                    "emails": emails,
                                    "config": config_info,
                                    "severity": "critical" if secrets else ("high" if path in [".git/config", ".git/HEAD", ".git/index"] else "medium")
                                }
                                findings.append(finding)
                                if secrets:
                                    all_secrets.extend(secrets)
                                if emails:
                                    all_emails.extend(emails)
                                if config_info:
                                    git_config_info.update(config_info)
                except Exception:
                    pass

        tasks = [_check_path(p) for p in paths]
        await asyncio.gather(*tasks)

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": findings,
        "summary": f"{len(findings)} file Git terekspos, {len(all_secrets)} secret, {len(all_emails)} email",
        "severity": "critical" if any(f["severity"] == "critical" for f in findings) else ("high" if any(f["severity"] == "high" for f in findings) else ("medium" if findings else "info")),
        "duration": round(duration, 2),
        "tested_paths": len(paths),
        "exposed_files": len(findings),
        "secrets_found": all_secrets,
        "emails_found": all_emails,
        "git_config": git_config_info
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))