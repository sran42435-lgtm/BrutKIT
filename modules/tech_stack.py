# modules/tech_stack.py — ~65 signature ala Wappalyzer + versi, body dibaca via resp.content.read(n)
MODULE_NAME = "tech_stack"

import asyncio
import re
import time
from typing import Dict, Any, List
from stealth_servant import get_stealth_headers, random_delay, build_session

__all__ = ["MODULE_NAME", "scan"]

TECH_SIGNATURES = {
    "WordPress": {"headers": {"x-powered-by": "wordpress"}, "body": [r"wp-content", r"wp-includes", r"wordpress"]},
    "Drupal": {"headers": {"x-generator": "drupal"}, "body": [r"drupal\.settings", r"Drupal\.behaviors"]},
    "Joomla": {"headers": {"x-content-encoded-by": "joomla"}, "body": [r"joomla", r"JFactory"]},
    "Magento": {"headers": {}, "body": [r"Mage\.Cookies", r"magento", r"skin/frontend"]},
    "Shopify": {"headers": {"x-shopify-stage": ""}, "body": [r"Shopify\.theme", r"cdn\.shopify\.com"]},
    "Wix": {"headers": {"x-wix-request-id": ""}, "body": [r"wix\.com", r"_wix"]},
    "Squarespace": {"headers": {"server": "squarespace"}, "body": [r"squarespace", r"static\.squarespace\.com"]},
    "Webflow": {"headers": {"x-webflow-enabled": ""}, "body": [r"webflow", r"webflow\.js"]},
    "Ghost": {"headers": {"x-powered-by": "ghost"}, "body": [r"ghost\.org", r"ghost-head"]},
    "Hugo": {"headers": {}, "body": [r"hugo", r"generator.*hugo"]},
    "Jekyll": {"headers": {}, "body": [r"jekyll", r"github\.io"]},
    "Next.js": {"headers": {"x-powered-by": "next\.js"}, "body": [r"_next/static", r"__NEXT_DATA__"]},
    "Nuxt.js": {"headers": {}, "body": [r"_nuxt/", r"__NUXT__"]},
    "Gatsby": {"headers": {}, "body": [r"gatsby", r"___gatsby"]},
    "React": {"headers": {}, "body": [r"react\.development\.js", r"react-dom", r"createRoot"]},
    "Vue.js": {"headers": {}, "body": [r"vue\.js", r"vue\.runtime", r"createApp"]},
    "Angular": {"headers": {}, "body": [r"ng-version", r"angular\.js", r"@angular"]},
    "Svelte": {"headers": {}, "body": [r"svelte", r"__svelte"]},
    "Ember.js": {"headers": {}, "body": [r"ember\.js", r"Ember\.Application"]},
    "Backbone.js": {"headers": {}, "body": [r"backbone\.js", r"Backbone\.Model"]},
    "jQuery": {"headers": {}, "body": [r"jquery", r"jQuery\.fn"]},
    "Bootstrap": {"headers": {}, "body": [r"bootstrap", r"bootstrap\.min\.css"]},
    "Tailwind CSS": {"headers": {}, "body": [r"tailwind", r"tailwindcss"]},
    "Bulma": {"headers": {}, "body": [r"bulma\.css", r"bulma\.min\.css"]},
    "Foundation": {"headers": {}, "body": [r"foundation", r"foundation\.min\.css"]},
    "Semantic UI": {"headers": {}, "body": [r"semantic\.ui", r"semantic\.min\.css"]},
    "Materialize": {"headers": {}, "body": [r"materialize", r"materialize\.min\.css"]},
    "Ant Design": {"headers": {}, "body": [r"antd", r"ant-design"]},
    "Element UI": {"headers": {}, "body": [r"element-ui", r"element-plus"]},
    "Vuetify": {"headers": {}, "body": [r"vuetify", r"v-application"]},
    "Quasar": {"headers": {}, "body": [r"quasar", r"quasar-framework"]},
    "Laravel": {"headers": {"x-powered-by": "laravel"}, "body": [r"laravel", r"laravel_session"]},
    "Symfony": {"headers": {"x-powered-by": "symfony"}, "body": [r"symfony", r"_sf2_attributes"]},
    "CodeIgniter": {"headers": {}, "body": [r"codeigniter", r"CI_VERSION"]},
    "Yii": {"headers": {"x-powered-by": "yii"}, "body": [r"yii", r"Yii::"]},
    "CakePHP": {"headers": {"x-powered-by": "cakephp"}, "body": [r"cakephp", r"CakePHP"]},
    "Zend Framework": {"headers": {"x-powered-by": "zend"}, "body": [r"zend", r"Zend_"]},
    "Django": {"headers": {"x-frame-options": "DENY"}, "body": [r"csrftoken", r"django"]},
    "Flask": {"headers": {}, "body": [r"flask", r"Werkzeug"]},
    "FastAPI": {"headers": {}, "body": [r"fastapi", r"uvicorn"]},
    "Starlette": {"headers": {}, "body": [r"starlette"]},
    "Tornado": {"headers": {"server": "tornado"}, "body": [r"tornado"]},
    "Express": {"headers": {"x-powered-by": "express"}, "body": [r"express"]},
    "Koa": {"headers": {}, "body": [r"koa"]},
    "Fastify": {"headers": {}, "body": [r"fastify"]},
    "NestJS": {"headers": {}, "body": [r"nest"]},
    "Hapi": {"headers": {}, "body": [r"hapi"]},
    "Spring Boot": {"headers": {}, "body": [r"spring", r"spring-boot"]},
    "ASP.NET Core": {"headers": {"x-powered-by": "asp\.net"}, "body": [r"asp\.net", r"__RequestVerificationToken"]},
    "Apache": {"headers": {"server": "apache"}, "body": []},
    "Nginx": {"headers": {"server": "nginx"}, "body": []},
    "IIS": {"headers": {"server": "iis"}, "body": []},
    "LiteSpeed": {"headers": {"server": "litespeed"}, "body": []},
    "OpenResty": {"headers": {"server": "openresty"}, "body": []},
    "Caddy": {"headers": {"server": "caddy"}, "body": []},
    "Traefik": {"headers": {"server": "traefik"}, "body": []},
    "Envoy": {"headers": {"server": "envoy"}, "body": []},
    "Varnish": {"headers": {"server": "varnish"}, "body": []},
    "Cloudflare": {"headers": {"server": "cloudflare"}, "body": []},
    "Akamai": {"headers": {"server": "akamai"}, "body": []},
    "AWS CloudFront": {"headers": {"server": "cloudfront"}, "body": []},
    "Google Cloud": {"headers": {"server": "gse"}, "body": []},
    "Firebase": {"headers": {}, "body": [r"firebase", r"__firebase"]},
    "Vercel": {"headers": {"x-vercel": ""}, "body": []},
    "Netlify": {"headers": {"server": "netlify"}, "body": []},
    "Heroku": {"headers": {"x-powered-by": "heroku"}, "body": []},
    "Docker": {"headers": {}, "body": [r"docker", r"container"]},
    "Kubernetes": {"headers": {}, "body": [r"kubernetes", r"k8s"]},
}

VERSION_PATTERNS = {
    "WordPress": r"wp-includes/[^/]+/version\.php.*?\$wp_version\s*=\s*['\"]([^'\"]+)['\"]",
    "Drupal": r"Drupal\s+([\d.]+)",
    "Joomla": r"Joomla!\s+([\d.]+)",
    "Magento": r"Magento/([\d.]+)",
    "jQuery": r"jquery[.-]([\d.]+)",
    "Bootstrap": r"bootstrap[.-]([\d.]+)",
    "React": r"react[.-]([\d.]+)",
    "Vue.js": r"vue[.-]([\d.]+)",
    "Angular": r"angular[.-]([\d.]+)",
    "Laravel": r"laravel[.-]([\d.]+)",
    "Symfony": r"symfony[.-]([\d.]+)",
    "Django": r"Django[/-]([\d.]+)",
    "Flask": r"Flask[/-]([\d.]+)",
    "Express": r"express[/-]([\d.]+)",
    "Spring Boot": r"spring-boot[.-]([\d.]+)",
    "Apache": r"Apache/([\d.]+)",
    "Nginx": r"nginx/([\d.]+)",
    "IIS": r"IIS/([\d.]+)",
    "OpenResty": r"openresty/([\d.]+)",
    "Caddy": r"Caddy/([\d.]+)",
}


async def _extract_version(name: str, body: str, headers: Dict[str, str]) -> str:
    """Coba ekstrak versi dari body atau headers."""
    pattern = VERSION_PATTERNS.get(name)
    if not pattern:
        return ""
    try:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return match.group(1)
        for h, v in headers.items():
            match = re.search(pattern, v, re.IGNORECASE)
            if match:
                return match.group(1)
    except Exception:
        pass
    return ""


async def scan(target: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deteksi tech stack dengan ~65 signature, ekstrak versi."""
    start = time.time()
    url = target if target.startswith("http") else f"https://{target}"
    max_body = config.get("max_body", 50000)

    async with build_session(verify=False, timeout=15) as session:
        await random_delay(0.2)
        try:
            async with session.get(url, headers=get_stealth_headers(), timeout=15, allow_redirects=True) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                body_bytes = await resp.content.read(max_body)
                body = body_bytes.decode(errors="ignore")
        except Exception as e:
            return {
                "module": MODULE_NAME,
                "target": target,
                "findings": [],
                "summary": f"Gagal koneksi: {e}",
                "severity": "error",
                "duration": round(time.time() - start, 2)
            }

    detected = []
    for tech, sig in TECH_SIGNATURES.items():
        matched = False
        evidence = []
        for h, pattern in sig["headers"].items():
            if h in headers:
                val = headers[h].lower()
                if not pattern or re.search(pattern, val, re.IGNORECASE):
                    matched = True
                    evidence.append(f"header:{h}={val[:100]}")
        if not matched:
            for pattern in sig["body"]:
                if re.search(pattern, body, re.IGNORECASE):
                    matched = True
                    evidence.append(f"body:{pattern}")
                    break
        if matched:
            version = await _extract_version(tech, body, headers)
            detected.append({
                "technology": tech,
                "version": version,
                "evidence": evidence[:3],
                "confidence": "high" if evidence else "medium",
                "severity": "info"
            })

    duration = time.time() - start
    return {
        "module": MODULE_NAME,
        "target": target,
        "findings": detected,
        "summary": f"{len(detected)} teknologi terdeteksi",
        "severity": "info",
        "duration": round(duration, 2),
        "tested_signatures": len(TECH_SIGNATURES),
        "body_length": len(body)
    }


if __name__ == "__main__":
    import sys
    import json
    t = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(json.dumps(asyncio.run(scan(t, {})), indent=2, ensure_ascii=False))