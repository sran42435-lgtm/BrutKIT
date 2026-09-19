// DevRoute-Inspector (AOT-Route & State Scanner) v2.0.1
// Frontend Asset & Client-Side Logic Pentesting untuk SPA modern
// (Next.js Pages/App Router, Nuxt, Remix, SvelteKit, Angular, Vite/React)
//
// Build : go build -o devroute-inspector main.go
// Pakai : ./devroute-inspector -u https://target.com
//
// HANYA untuk target yang Anda miliki izin ujinya.

package main

import (
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// ============================================================
//  KONSTANTA & WARNA TERMINAL
// ============================================================

const banner = `
  ___         _ _
 |_ _|_ _  __| (_)__ _ ___
  | || ' \/ _` + "`" + ` | / _` + "`" + ` / _ \
 |___|_||_\__,_|_\__, \___/
                 |___/
  DevRoute-Inspector v2.0.1 — AOT-Route & State Scanner
  + AST-Light | RSC/SvelteKit/Angular | GraphQL/RBAC | Diff-Probe | Burp/HTML
`

const (
	cReset  = "\033[0m"
	cRed    = "\033[1;31m"
	cYellow = "\033[1;33m"
	cCyan   = "\033[1;36m"
	cGreen  = "\033[1;32m"
	cGray   = "\033[0;37m"
	cBold   = "\033[1m"
)

var sevColor = map[string]string{
	"CRITICAL": cRed,
	"HIGH":     cYellow,
	"MEDIUM":   cCyan,
	"LOW":      cGray,
	"INFO":     cGreen,
}

// ============================================================
//  REGEX GLOBAL — DIKOMPILASI SEKALI
// ============================================================

var (
	reRouteStartAlnum  = regexp.MustCompile(`^/[a-zA-Z0-9]`)
	reTemplateLiteral  = regexp.MustCompile(`\$\{[^}]*\}`)
	reBuildID          = regexp.MustCompile(`"buildId"\s*:\s*"([A-Za-z0-9_\-]+)"`)
	reBuildManifestRef = regexp.MustCompile(`/_next/static/([A-Za-z0-9_\-]+)/_buildManifest\.js`)
	reIsHTML           = regexp.MustCompile(`(?i)<!DOCTYPE html|<html`)
)

// ===== MODUL BARU 1: AST-LIGHT — regex global =====
var (
	reTplLiteralFull = regexp.MustCompile("[\"'`](/[^\"'`]{2,160}\\$\\{[^\"'`]{0,160})[\"'`]")
	reConcatPath     = regexp.MustCompile(`["'](\/[A-Za-z0-9_\-\/]{1,80})["']\s*\+`)
	reObjRouteMap    = regexp.MustCompile(`(?i)(routes?Map|pathMap|paths?|ROUTES|PAGES|URLS?|ENDPOINTS)\s*[:=]\s*\{([^}]{1,900})\}`)
	reQuotedInside   = regexp.MustCompile(`["'](\/?[A-Za-z0-9_\-\/:\.]{2,120})["']`)
)

// ===== MODUL BARU 2: FRAMEWORK-SPECIFIC — regex global =====
var (
	reRSCActionID    = regexp.MustCompile(`(?:actionId|serverActionID|action_id)["']?\s*[:=]\s*["']([a-fA-F0-9]{8,64})["']`)
	reUseServer      = regexp.MustCompile(`use server`)
	reSvelteManifest = regexp.MustCompile(`__SVELTEKIT_|matchers\s*:\s*\{|\/_app\/immutable\/`)
	reSvelteRouteID  = regexp.MustCompile(`id\s*:\s*"(\/[A-Za-z0-9_\-$$$$$$\.]{0,120})"`)
	reAngularLazy    = regexp.MustCompile(`loadChildren[^;]{0,120}?import$\s*["']([^"']{2,120})["']`)
	reWebpackChunk   = regexp.MustCompile(`static\/js\/([A-Za-z0-9_\-~]+)\.([0-9a-f]{8,32})\.chunk\.js`)
	reChunkMapPair   = regexp.MustCompile(`\{((?:\s*"[0-9A-Za-z_\-~]+"\s*:\s*"[0-9a-f]{8,32}"\s*,?){2,})\}`)
	reChunkMapEntry  = regexp.MustCompile(`"([0-9A-Za-z_\-~]+)"\s*:\s*"([0-9a-f]{8,32})"`)
)

// ===== MODUL BARU 4: DIFF PROBE — state global =====
var (
	dummyFingerprint   httpFingerprint
	dummyFingerprintOK bool
	dummyRoute         = "/this-route-does-not-exist-12345"
)

type httpFingerprint struct {
	Code    int
	CLen    int64
	CType   string
	Title   string
	Headers string
}

// ============================================================
//  STRUKTUR DATA
// ============================================================

type Finding struct {
	Severity string `json:"severity"`
	Type     string `json:"type"`
	Value    string `json:"value"`
	Context  string `json:"context,omitempty"`
	Asset    string `json:"asset"`
	HTTPStat int    `json:"http_status,omitempty"`
	Verified string `json:"verified,omitempty"`
}

type SourceMap struct {
	Version        int      `json:"version"`
	File           string   `json:"file"`
	Sources        []string `json:"sources"`
	SourcesContent []string `json:"sourcesContent"`
}

type Report struct {
	Target      string         `json:"target"`
	StartedAt   string         `json:"started_at"`
	FinishedAt  string         `json:"finished_at"`
	AssetsTotal int            `json:"assets_scanned"`
	SMapsTotal  int            `json:"sourcemaps_parsed"`
	Findings    []Finding      `json:"findings"`
	Summary     map[string]int `json:"summary"`
	Routes      []RouteProbe   `json:"route_verification,omitempty"`

	assets int `json:"-"`
	smaps  int `json:"-"`
}

var reportMu sync.Mutex

func incAssets() int {
	reportMu.Lock()
	defer reportMu.Unlock()
	report.assets++
	return report.assets
}

func incSmaps() int {
	reportMu.Lock()
	defer reportMu.Unlock()
	report.smaps++
	return report.smaps
}

func assetCount() int {
	reportMu.Lock()
	defer reportMu.Unlock()
	return report.assets
}

type RouteProbe struct {
	Route      string `json:"route"`
	FullPath   string `json:"full_url"`
	HTTPStatus int    `json:"http_status"`
	ContentLen int64  `json:"content_length"`
	RedirectTo string `json:"redirect_to,omitempty"`
	Verdict    string `json:"verdict"`
	FromSource string `json:"from_asset"`
	AnonStatus int    `json:"anon_status,omitempty"`
	DiffNote   string `json:"diff_note,omitempty"`
}

// ============================================================
//  KONFIGURASI GLOBAL
// ============================================================

type Config struct {
	Target    *url.URL
	Headers   []string
	Conc      int
	Timeout   time.Duration
	Insecure  bool
	Proxy     string
	Verify    bool
	JSONOut   string
	DumpDir   string
	MaxAssets int
	MaxBody   int64
	// ===== MODUL BARU: opsi v2.0 =====
	AuthDiff bool
	RSCProbe bool
	BurpXML  string
	HTMLOut  string
	NoAST    bool
}

var cfg Config
var client *http.Client

// ============================================================
//  UTILITAS HTTP (dengan mitigasi SSRF: hanya host target)
// ============================================================

func buildClient() *http.Client {
	jar, _ := cookiejar.New(nil)
	tr := &http.Transport{
		TLSClientConfig:     &tls.Config{InsecureSkipVerify: cfg.Insecure},
		MaxIdleConnsPerHost: cfg.Conc * 2,
		DisableCompression:  false,
	}
	if cfg.Proxy != "" {
		if pu, err := url.Parse(cfg.Proxy); err == nil {
			tr.Proxy = http.ProxyURL(pu)
		}
	}
	return &http.Client{
		Transport: tr,
		Jar:       jar,
		Timeout:   cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return http.ErrUseLastResponse
			}
			if req.URL.Hostname() != cfg.Target.Hostname() {
				return http.ErrUseLastResponse // SSRF guard
			}
			return nil
		},
	}
}

func sameHost(u string) bool {
	pu, err := url.Parse(u)
	if err != nil {
		return false
	}
	if pu.Host == "" {
		return true
	}
	return pu.Hostname() == cfg.Target.Hostname()
}

func absURL(raw string) string {
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		return raw
	}
	pu, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return cfg.Target.ResolveReference(pu).String()
}

func fetch(rawURL string) (string, int, error) {
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (DevRoute-Inspector/1.0)")
	req.Header.Set("Accept", "*/*")
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 {
			req.Header.Set(strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, cfg.MaxBody))
	if err != nil {
		return "", resp.StatusCode, err
	}
	return string(body), resp.StatusCode, nil
}

func probeNoFollow(fullURL string) (int, int64, string) {
	tr := client.Transport.(*http.Transport)
	probe := &http.Client{
		Transport: tr,
		Timeout:   cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	req, _ := http.NewRequest("GET", fullURL, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (DevRoute-Inspector/1.0)")
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 {
			req.Header.Set(strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
	}
	resp, err := probe.Do(req)
	if err != nil {
		return 0, 0, ""
	}
	defer resp.Body.Close()
	n, _ := io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	return resp.StatusCode, n, resp.Header.Get("Location")
}

// ===== MODUL BARU 4: probe dengan kontrol header auth + fingerprint =====

func probeNoFollowEx(fullURL string, includeAuth bool) (int, int64, string) {
	tr := client.Transport.(*http.Transport)
	probe := &http.Client{
		Transport: tr, Timeout: cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	req, _ := http.NewRequest("GET", fullURL, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (DevRoute-Inspector/2.0)")
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) != 2 {
			continue
		}
		k := strings.TrimSpace(kv[0])
		if !includeAuth && isAuthHeader(k) {
			continue // strip header autentikasi untuk probe anonim
		}
		req.Header.Set(k, strings.TrimSpace(kv[1]))
	}
	resp, err := probe.Do(req)
	if err != nil {
		return 0, 0, ""
	}
	defer resp.Body.Close()
	n, _ := io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	return resp.StatusCode, n, resp.Header.Get("Location")
}

func isAuthHeader(k string) bool {
	lk := strings.ToLower(k)
	return lk == "authorization" || lk == "cookie" || lk == "x-api-key" || lk == "x-csrf-token"
}

func hasAuthHeaders() bool {
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 && isAuthHeader(strings.TrimSpace(kv[0])) {
			return true
		}
	}
	return false
}

var reTitle = regexp.MustCompile(`(?is)<title[^>]*>(.{0,200}?)</title>`)

func fingerprintRoute(fullURL string) httpFingerprint {
	fp := httpFingerprint{}
	tr := client.Transport.(*http.Transport)
	probe := &http.Client{
		Transport: tr, Timeout: cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	req, _ := http.NewRequest("GET", fullURL, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (DevRoute-Inspector/2.0)")
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 {
			req.Header.Set(strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
	}
	resp, err := probe.Do(req)
	if err != nil {
		return fp
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 128*1024))
	fp.Code = resp.StatusCode
	fp.CLen = int64(len(body))
	fp.CType = resp.Header.Get("Content-Type")
	if m := reTitle.FindSubmatch(body); m != nil {
		fp.Title = strings.TrimSpace(string(m[1]))
	}
	fp.Headers = resp.Header.Get("X-Powered-By") + "|" + resp.Header.Get("Cache-Control") + "|" + resp.Header.Get("Etag")
	return fp
}

func initDummyBaseline() {
	full := strings.TrimRight(cfg.Target.String(), "/") + dummyRoute
	dummyFingerprint = fingerprintRoute(full)
	dummyFingerprintOK = dummyFingerprint.Code != 0
	if dummyFingerprintOK {
		fmt.Printf("  %s[+]%s Baseline dummy-route: HTTP %d, %d B, title=%q\n",
			cGreen, cReset, dummyFingerprint.Code, dummyFingerprint.CLen, truncate(dummyFingerprint.Title, 40))
	}
}

func matchesDummy(fp httpFingerprint) bool {
	if !dummyFingerprintOK || fp.Code != 200 {
		return false
	}
	sameTitle := dummyFingerprint.Title != "" && fp.Title == dummyFingerprint.Title
	sameLen := dummyFingerprint.CLen > 0 && absDiff(fp.CLen, dummyFingerprint.CLen) <= dummyFingerprint.CLen/20+64
	sameCT := strings.Contains(fp.CType, "text/html") && strings.Contains(dummyFingerprint.CType, "text/html")
	return sameTitle && sameLen && sameCT
}

// ============================================================
//  MESIN EKSTRAKSI (Regex Berlapis ala jsluice matcher)
// ============================================================

type Matcher struct {
	Name     string
	Type     string
	Severity string
	Re       *regexp.Regexp
	Group    int
	Validate func(string) bool
}

func isRoutePath(s string) bool {
	if len(s) < 2 || len(s) > 200 || !strings.HasPrefix(s, "/") {
		return false
	}
	deny := []string{
		".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".css", ".woff", ".woff2",
		".ttf", ".eot", ".mp4", ".webp", ".map", ".json", ".xml", ".txt",
		"node_modules", "webpack/", "webpack://", "src/", "//",
	}
	low := strings.ToLower(s)
	for _, d := range deny {
		if strings.Contains(low, d) {
			if d == ".json" && strings.HasPrefix(low, "/api/") {
				continue
			}
			return false
		}
	}
	return reRouteStartAlnum.MatchString(s)
}

func isAPIPath(s string) bool {
	low := strings.ToLower(s)
	return strings.Contains(low, "/api/") ||
		strings.Contains(low, "/internal/") ||
		strings.Contains(low, "/graphql") ||
		strings.Contains(low, "api.") || strings.Contains(low, "internal.") ||
		strings.Contains(low, "staging.") || strings.Contains(low, "dev.")
}

func buildMatchers() []Matcher {
	q := `["'` + "`" + `]`
	Q := func(inner string) string { return q + `(` + inner + `)` + q }

	return []Matcher{
		// --- 1. ROUTE EKSAK dari pola framework ---
		{Name: "react-router-path", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`path\s*:\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\$\{\}\.]{1,120}`)), Validate: isRoutePath},
		{Name: "route-component-tag", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`<Route[^>]+path\s*=\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\.]{1,120}`)), Validate: isRoutePath},
		{Name: "createBrowserRouter-def", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`createBrowserRouter|createHashRouter|createRoutesFromElements`), Group: -1},
		{Name: "next-buildmanifest-route", Type: "route", Severity: "HIGH",
			Re: regexp.MustCompile(`"(\/[^"]{1,120})"\s*:\s*$$\s*"static\/`), Group: 1, Validate: isRoutePath},
		{Name: "next-page-route", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`["']\/_next\/(?:data|static)\/[^"']+["']|buildId["']?\s*:\s*["']([A-Za-z0-9_\-]{5,30})["']`)},
		{Name: "remix-manifest-route", Type: "route", Severity: "HIGH",
			Re: regexp.MustCompile(`"(?:routes|app)\/[A-Za-z0-9_\-\/\.]{1,120}"|(?:routeModule|parentId|id)\s*:\s*"(routes\/[^"]{1,120})"`),
			Validate: func(s string) bool {
				s = strings.Trim(s, `"`)
				return strings.HasPrefix(s, "routes/") || strings.HasPrefix(s, "app/")
			}},
		{Name: "nuxt-route-meta", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:definePageMeta|route)\s*[:(]\s*\{[^}]{0,200}?path\s*:\s*` + Q(`\/[A-Za-z0-9_\-\/:]{1,120}`)), Validate: isRoutePath},
		{Name: "push-replace-nav", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:router\.(?:push|replace)|navigate|useNavigate$$\s*$|redirect)\s*$\s*` + Q(`\/[A-Za-z0-9_\-\/:\?=&]{1,150}`)), Validate: isRoutePath},
		{Name: "generic-path", Type: "route", Severity: "LOW",
			Re: regexp.MustCompile(Q(`\/[A-Za-z0-9_\-]+(?:\/[A-Za-z0-9_\-:\.\$\{\}]{1,60}){1,5}`)), Validate: isRoutePath},

		// ===== MODUL BARU 2: SvelteKit & Angular =====
		{Name: "sveltekit-marker", Type: "route", Severity: "INFO",
			Re: reSvelteManifest, Group: -1},
		{Name: "sveltekit-route-id", Type: "route", Severity: "MEDIUM",
			Re: reSvelteRouteID, Group: 1, Validate: isRoutePath},
		{Name: "angular-router-path", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`path\s*:\s*["']([A-Za-z0-9_\-\/:]{1,120})["']`), Group: 1},
		{Name: "angular-lazy-module", Type: "route", Severity: "MEDIUM",
			Re: reAngularLazy, Group: 1},

		// ===== MODUL BARU 2: Next.js RSC / Server Actions =====
		{Name: "rsc-server-action-id", Type: "server-action", Severity: "HIGH",
			Re: reRSCActionID, Group: 1},
		{Name: "rsc-use-server", Type: "server-action", Severity: "INFO",
			Re: reUseServer, Group: -1},

		// --- 2. API / ENDPOINT INTERNAL ---
		{Name: "fetch-axios-call", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`(?:fetch|axios\.(?:get|post|put|delete|patch)|\.ajax)\s*$\s*` + Q(`[^"'`+"`"+`]{3,200}`))},
		{Name: "api-path", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(Q(`\/(?:api|internal|graphql|v[0-9]+)\/[A-Za-z0-9_\-\/:\.]{1,120}`)), Validate: isAPIPath},
		{Name: "external-api-host", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`https?:\/\/[A-Za-z0-9\.\-]*(?:api|internal|staging|dev|admin)[A-Za-z0-9\.\-]*\.[a-z]{2,}(?:\/[^\s"'` + "`" + `<>]{0,80})?`)},

		// ===== MODUL BARU 3: GraphQL extractor =====
		{Name: "gql-operation", Type: "graphql", Severity: "HIGH",
			Re: regexp.MustCompile(`(?:query|mutation|subscription)\s+([A-Za-z][A-Za-z0-9_]*)\s*[$\{\$]`), Group: 1},
		{Name: "gql-tag", Type: "graphql", Severity: "MEDIUM",
			Re: regexp.MustCompile("`\\s*(?:query|mutation|subscription)[\\s\\S]{0,60}`"), Group: -1},
		{Name: "gql-variable", Type: "graphql", Severity: "LOW",
			Re: regexp.MustCompile(`\$([A-Za-z][A-Za-z0-9_]*)\s*:\s*(Int|String|Boolean|Float|ID|[A-Z][A-Za-z0-9_]{0,40}!?)`), Group: -2},

		// ===== MODUL BARU 3: RBAC / permission mapper =====
		{Name: "rbac-permission-call", Type: "rbac", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:hasPermission|hasRole|isAllowed|can|checkPermission|authorize)\s*$\s*` + Q(`[A-Za-z_][A-Za-z0-9_:.\-]{2,60}`))},
		{Name: "rbac-role-compare", Type: "rbac", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:role|userRole|user_type|userType)\s*(?:===?|!==?)\s*` + Q(`[A-Za-z_]{3,30}`))},

		// ===== MODUL BARU 3: Storage keys inventory =====
		{Name: "webstorage-key", Type: "storage", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:localStorage|sessionStorage)\.(?:getItem|setItem|removeItem)$\s*` + Q(`[^"'`+"`"+`]{1,60}`))},
		{Name: "indexeddb-open", Type: "storage", Severity: "LOW",
			Re: regexp.MustCompile(`indexedDB\.open$\s*` + Q(`[^"'`+"`"+`]{1,60}`))},

		// --- 3. SECRET / KREDENSIAL HARDCODED ---
		{Name: "aws-access-key", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`(?:AKIA|ASIA)[0-9A-Z]{16}`)},
		{Name: "private-key-block", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----`)},
		{Name: "jwt-token", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}`)},
		{Name: "google-api-key", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`AIza[0-9A-Za-z\-_]{35}`)},
		{Name: "slack-token", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`xox[baprs]-[0-9A-Za-z\-]{10,72}`)},
		{Name: "github-token", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{36,255}`)},
		{Name: "stripe-key", Type: "secret", Severity: "CRITICAL",
			Re: regexp.MustCompile(`(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,99}`)},
		{Name: "generic-secret-assign", Type: "secret", Severity: "HIGH",
			Re: regexp.MustCompile(`(?i)(?:api[_\-]?key|secret[_\-]?key|client[_\-]?secret|auth[_\-]?token|access[_\-]?token|private[_\-]?key|password)\s*[:=]\s*` + Q(`[^"'`+"`"+`\s\$\{]{8,120}`))},

		// --- 4. STATE SCHEMA / FEATURE FLAG / ROLE ---
		{Name: "next-data-blob", Type: "state", Severity: "MEDIUM",
			Re: regexp.MustCompile(`__NEXT_DATA__\s*=\s*(\{.{0,1000})`), Group: 1},
		{Name: "nuxt-payload", Type: "state", Severity: "MEDIUM",
			Re: regexp.MustCompile(`__NUXT__|window\.__NUXT_DATA__|useNuxtApp$$\.payload`), Group: -1},
		{Name: "initial-state", Type: "state", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:initialState|preloadedState|defaultState|__PRELOADED_STATE__)\s*[:=]\s*(\{.{0,800})`), Group: 1},
		{Name: "role-flag", Type: "state", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?i)` + Q(`(role|isAdmin|isStaff|isSuperUser|permission|feature[_\-]?flag[s]?|canAccess[A-Za-z]*)`) + `\s*[:=]`),
			Group: -2},
		{Name: "debug-panel", Type: "route", Severity: "CRITICAL",
			Re: regexp.MustCompile(`(?i)` + Q(`\/[a-z0-9_\-]*(admin|debug|dev-tools|internal|console|phpinfo|actuator|backdoor)[a-z0-9_\-\/]*`)),
			Group: 1, Validate: isRoutePath},
	}
}

// ============================================================
//  ANALISIS KONTEN (bundle JS, HTML, sourcesContent sourcemap)
// ============================================================

var (
	matchers   = buildMatchers()
	dedup      = make(map[string]bool)
	dedupMutex sync.Mutex
	findings   []Finding
	findMutex  sync.Mutex
)

func addFinding(f Finding) {
	key := f.Severity + "|" + f.Type + "|" + f.Value
	dedupMutex.Lock()
	if dedup[key] {
		dedupMutex.Unlock()
		return
	}
	dedup[key] = true
	dedupMutex.Unlock()

	findMutex.Lock()
	findings = append(findings, f)
	findMutex.Unlock()

	col := sevColor[f.Severity]
	fmt.Printf("  %s[%s]%s %-22s %s%s\n", col, f.Severity, cReset, f.Type, truncate(f.Value, 90), cReset)
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func analyzeContent(content, assetURL string) {
	for _, m := range matchers {
		matches := m.Re.FindAllStringSubmatch(content, -1)
		for _, mm := range matches {
			if m.Group == -1 {
				addFinding(Finding{Severity: "INFO", Type: "framework-marker",
					Value: m.Name, Asset: assetURL, Context: truncate(mm[0], 80)})
				continue
			}
			if m.Group == -2 {
				addFinding(Finding{Severity: m.Severity, Type: m.Type,
					Value: "state-key: " + mm[1], Asset: assetURL,
					Context: truncate(mm[0], 100)})
				continue
			}
			val := mm[0]
			if m.Group >= 1 && len(mm) > m.Group {
				val = mm[m.Group]
			}
			val = strings.Trim(val, `"'`+"`"+` `)

			// ===== MODUL BARU 2: normalisasi Angular path =====
			if m.Name == "angular-router-path" && !strings.HasPrefix(val, "/") {
				val = "/" + val
			}
			// ===== MODUL BARU 3: GraphQL variable → format "$name: Type" =====
			if m.Name == "gql-variable" && len(mm) > 2 {
				val = "$" + mm[1] + ": " + mm[2]
			}

			if m.Validate != nil && !m.Validate(val) {
				continue
			}
			sev := m.Severity
			if m.Type == "route" && m.Name != "debug-panel" {
				low := strings.ToLower(val)
				if strings.Contains(low, "admin") || strings.Contains(low, "debug") ||
					strings.Contains(low, "internal") || strings.Contains(low, "config") {
					sev = "HIGH"
				}
			}
			addFinding(Finding{Severity: sev, Type: m.Type, Value: val,
				Asset: assetURL, Context: truncate(mm[0], 120)})
		}
	}

	// ===== MODUL BARU 1 & 2: hook AST-Light + chunk graph + RSC =====
	if !cfg.NoAST {
		astLightAnalyze(content, assetURL)
	}
	chunkGraphAnalyze(content, assetURL)
	rscActionAnalyze(content, assetURL)
}

// ============================================================
// ===== MODUL BARU 1: AST-LIGHT PARSER ENGINE =====
// ============================================================

func astLightAnalyze(content, assetURL string) {
	// --- 1a. Template Literal Interpolation Solver ---
	for _, mm := range reTplLiteralFull.FindAllStringSubmatch(content, -1) {
		lit := mm[1]
		if !strings.Contains(lit, "${") {
			continue
		}
		candidates := expandTemplate(lit)
		for _, cand := range candidates {
			sev := "MEDIUM"
			typ := "route-candidate"
			if isAPIPath(cand) {
				sev = "HIGH"
				typ = "api-candidate"
			}
			if !isRoutePath(cand) {
				continue
			}
			addFinding(Finding{Severity: sev, Type: typ, Value: cand,
				Asset: assetURL, Context: "template: " + truncate(lit, 80)})
		}
	}

	// --- 1b. String Concatenation Path ---
	for _, mm := range reConcatPath.FindAllStringSubmatch(content, -1) {
		base := mm[1]
		if !isRoutePath(base) {
			continue
		}
		sev := "LOW"
		if strings.Contains(strings.ToLower(base), "admin") || isAPIPath(base) {
			sev = "MEDIUM"
		}
		addFinding(Finding{Severity: sev, Type: "concat-path", Value: base,
			Asset: assetURL, Context: truncate(mm[0], 80)})
	}

	// --- 1c. Dynamic Object Path Resolver ---
	for _, mm := range reObjRouteMap.FindAllStringSubmatch(content, -1) {
		mapName := mm[1]
		body := mm[2]
		for _, q := range reQuotedInside.FindAllStringSubmatch(body, -1) {
			v := q[1]
			if !strings.HasPrefix(v, "/") {
				v = "/" + v
			}
			if !isRoutePath(v) && !isAPIPath(strings.ToLower(v)) {
				continue
			}
			addFinding(Finding{Severity: "MEDIUM", Type: "object-route-map", Value: v,
				Asset: assetURL, Context: mapName + " → " + truncate(q[0], 60)})
		}
	}
}

func expandTemplate(lit string) []string {
	segs := reTemplateLiteral.Split(lit, -1)
	nVars := len(reTemplateLiteral.FindAllString(lit, -1))
	if nVars == 0 || nVars > 2 {
		return nil
	}
	variants := []string{"1", "test"}
	out := []string{""}
	for i, seg := range segs {
		var next []string
		for _, cur := range out {
			next = append(next, cur+seg)
			if i < len(segs)-1 {
				for _, v := range variants {
					next = append(next, cur+seg+v)
				}
			}
		}
		if len(next) > 8 {
			next = next[:8]
		}
		out = next
	}
	seen := map[string]bool{}
	var res []string
	for _, o := range out {
		if o != "" && !seen[o] {
			seen[o] = true
			res = append(res, o)
		}
		if len(res) >= 4 {
			break
		}
	}
	return res
}

// ============================================================
// ===== MODUL BARU 2: FRAMEWORK-SPECIFIC ANALYZERS =====
// ============================================================

func rscActionAnalyze(content, assetURL string) {
	for _, mm := range reRSCActionID.FindAllStringSubmatch(content, -1) {
		addFinding(Finding{Severity: "HIGH", Type: "server-action", Value: "actionId: " + mm[1],
			Asset: assetURL, Context: "POST-able Server Action (uji via header Next-Action)"})
	}
}

var chunkSeen = make(map[string]bool)
var chunkMu sync.Mutex

func chunkGraphAnalyze(content, assetURL string) {
	for _, mm := range reWebpackChunk.FindAllStringSubmatch(content, -1) {
		name := "static/js/" + mm[1] + "." + mm[2] + ".chunk.js"
		registerChunk(name, assetURL)
	}
	if strings.Contains(content, "chunk.js") || strings.Contains(assetURL, "runtime") {
		for _, blk := range reChunkMapPair.FindAllStringSubmatch(content, -1) {
			for _, e := range reChunkMapEntry.FindAllStringSubmatch(blk[1], -1) {
				name := "static/js/" + e[1] + "." + e[2] + ".chunk.js"
				registerChunk(name, assetURL)
			}
		}
	}
}

func registerChunk(relPath, fromAsset string) {
	full := absURL("/_next/" + relPath)
	chunkMu.Lock()
	if chunkSeen[full] {
		chunkMu.Unlock()
		return
	}
	chunkSeen[full] = true
	chunkMu.Unlock()

	addFinding(Finding{Severity: "INFO", Type: "hidden-chunk", Value: "/_next/" + relPath,
		Asset: fromAsset, Context: "chunk tidak terdaftar di indeks HTML"})
	if sameHost(full) && assetCount() < cfg.MaxAssets && markSeen(full) {
		go scanAsset(full, 1)
	}
}

func probeRSC() {
	if !cfg.RSCProbe {
		return
	}
	fmt.Printf("\n%s[FASE 2.5]%s Next.js App Router / RSC probe...\n", cBold, cReset)
	req, err := http.NewRequest("GET", cfg.Target.String(), nil)
	if err != nil {
		return
	}
	req.Header.Set("RSC", "1")
	req.Header.Set("Next-Router-State-Tree", "%5B%22%22%5D")
	req.Header.Set("User-Agent", "Mozilla/5.0 (DevRoute-Inspector/2.0)")
	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 {
			req.Header.Set(strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	payload := string(body)
	if resp.Header.Get("Content-Type") == "" && len(payload) == 0 {
		return
	}
	fmt.Printf("  %s[+]%s RSC response: HTTP %d, %d KB\n", cGreen, cReset, resp.StatusCode, len(payload)/1024)

	for _, mm := range reRSCActionID.FindAllStringSubmatch(payload, -1) {
		addFinding(Finding{Severity: "HIGH", Type: "server-action", Value: "actionId: " + mm[1],
			Asset: "RSC payload " + cfg.Target.String()})
	}
	reRSCTree := regexp.MustCompile(`"children",\s*"([a-z0-9\-_]{1,60})"`)
	for _, mm := range reRSCTree.FindAllStringSubmatch(payload, -1) {
		addFinding(Finding{Severity: "MEDIUM", Type: "route", Value: "/" + mm[1],
			Asset: "RSC payload", Context: "segmen tree App Router"})
	}
	analyzeContent(payload, cfg.Target.String()+" (RSC)")
}

// ============================================================
//  STATIC ASSET FETCHER + SOURCE MAP UNPACKER
// ============================================================

var (
	seenAssets = make(map[string]bool)
	assetMutex sync.Mutex
)

var (
	reScriptSrc = regexp.MustCompile(`(?i)<script[^>]+src\s*=\s*["']([^"']+\.js[^"']*)["']`)
	reLinkHref  = regexp.MustCompile(`(?i)<link[^>]+href\s*=\s*["']([^"']+\.js[^"']*)["']`)
	reSourceMap = regexp.MustCompile(`//[#@]\s*sourceMappingURL\s*=\s*(\S+)`)
	reJSString  = regexp.MustCompile(`["'` + "`" + `](\/[A-Za-z0-9_\-\/\.]+?\.js(?:\?[A-Za-z0-9_=&\-\.]*)?)["'`+"`"+`]`)
	reNextChunk = regexp.MustCompile(`["'` + "`" + `](\/_next\/static\/[^"'`+"`"+`]+?\.js)["'`+"`"+`]`)
	reSvelteApp = regexp.MustCompile(`["'](\/_app\/immutable\/[^"'`+"`"+`]+?\.js)["']`)
)

func markSeen(u string) bool {
	assetMutex.Lock()
	defer assetMutex.Unlock()
	if seenAssets[u] {
		return false
	}
	seenAssets[u] = true
	return true
}

func extractAssetRefs(content, baseURL string) []string {
	var refs []string
	for _, re := range []*regexp.Regexp{reScriptSrc, reLinkHref, reJSString, reNextChunk, reSvelteApp} {
		for _, mm := range re.FindAllStringSubmatch(content, -1) {
			u := absURL(mm[1])
			if u != "" && sameHost(u) {
				refs = append(refs, u)
			}
		}
	}
	return refs
}

func handleSourceMapURL(smRef, jsURL, content string) {
	if strings.HasPrefix(smRef, "data:") {
		parts := strings.SplitN(smRef, ",", 2)
		if len(parts) == 2 {
			decoded, err := base64.StdEncoding.DecodeString(parts[1])
			if err != nil {
				return
			}
			processSourceMap(string(decoded), jsURL+" (inline)")
		}
		return
	}
	mapURL := smRef
	if pu, err := url.Parse(jsURL); err == nil {
		mu, _ := url.Parse(smRef)
		mapURL = pu.ResolveReference(mu).String()
	}
	if !sameHost(mapURL) {
		fmt.Printf("  %s[!]%s sourcemap lintas-host diblokir (SSRF guard): %s\n", cYellow, cReset, mapURL)
		return
	}
	mapBody, code, err := fetch(mapURL)
	if err != nil || code != 200 {
		return
	}
	processSourceMap(mapBody, mapURL)
}

func processSourceMap(raw, mapURL string) {
	var sm SourceMap
	if err := json.Unmarshal([]byte(raw), &sm); err != nil {
		return
	}
	if sm.Version == 0 || len(sm.Sources) == 0 {
		return
	}
	fmt.Printf("  %s[+]%s SourceMap v%d: %d modul asli — %s\n",
		cGreen, cReset, sm.Version, len(sm.Sources), truncate(mapURL, 70))

	dumpDir := ""
	if cfg.DumpDir != "" {
		dumpDir = path.Join(cfg.DumpDir, sanitizeName(mapURL))
	}
	for i, src := range sm.Sources {
		var content string
		if i < len(sm.SourcesContent) {
			content = sm.SourcesContent[i]
		}
		if content == "" {
			continue
		}
		analyzeContent(content, mapURL+" → "+src)

		if dumpDir != "" {
			fp := path.Join(dumpDir, sanitizeName(src))
			os.MkdirAll(path.Dir(fp), 0o755)
			os.WriteFile(fp, []byte(content), 0o644)
		}
		low := strings.ToLower(src)
		if strings.Contains(low, "admin") || strings.Contains(low, "debug") ||
			strings.Contains(low, "internal") || strings.Contains(low, "secret") ||
			strings.Contains(low, ".env") {
			addFinding(Finding{Severity: "HIGH", Type: "sourcemap-file",
				Value: src, Asset: mapURL})
		}
	}
	incSmaps()
}

func sanitizeName(s string) string {
	s = strings.ReplaceAll(s, "://", "_")
	s = strings.ReplaceAll(s, ":", "_")
	r := regexp.MustCompile(`[^A-Za-z0-9_\.\-/]`)
	s = r.ReplaceAllString(s, "_")
	return strings.TrimPrefix(s, "_")
}

func scanAsset(jsURL string, depth int) {
	body, code, err := fetch(jsURL)
	if err != nil || code != 200 || body == "" {
		return
	}
	if reIsHTML.MatchString(body[:min(len(body), 512)]) {
		return
	}
	fmt.Printf("  %s[*]%s Scanning asset (%d KB): %s\n", cCyan, cReset, len(body)/1024, truncate(jsURL, 80))
	n := incAssets()

	analyzeContent(body, jsURL)

	if mm := reSourceMap.FindStringSubmatch(body); mm != nil {
		handleSourceMapURL(mm[1], jsURL, body)
	} else {
		handleSourceMapURL(jsURL+".map", jsURL, "")
	}

	if depth < 1 {
		for _, ref := range extractAssetRefs(body, jsURL) {
			if n >= cfg.MaxAssets || assetCount() >= cfg.MaxAssets {
				return
			}
			if markSeen(ref) {
				scanAsset(ref, depth+1)
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================
//  DYNAMIC ROUTE RECONSTRUCT ENGINE
// ============================================================

func verifyRoutes() []RouteProbe {
	routeSet := make(map[string]string)
	findMutex.Lock()
	snapshot := append([]Finding(nil), findings...)
	findMutex.Unlock()

	hasGraphQL := false
	for _, f := range snapshot {
		if f.Type == "route" || (f.Type == "api" && strings.HasPrefix(f.Value, "/")) {
			r := normalizeRoute(f.Value)
			if r == "" {
				continue
			}
			if _, ok := routeSet[r]; !ok {
				routeSet[r] = f.Asset
			}
		}
		// ===== MODUL BARU 3: kandidat rute dari AST-Light =====
		if f.Type == "api-candidate" || f.Type == "route-candidate" || f.Type == "object-route-map" {
			r := normalizeRoute(f.Value)
			if r != "" {
				if _, ok := routeSet[r]; !ok {
					routeSet[r] = f.Asset
				}
			}
		}
		if f.Type == "graphql" {
			hasGraphQL = true
		}
	}
	// ===== MODUL BARU 3: probe endpoint GraphQL standar =====
	if hasGraphQL {
		for _, gp := range []string{"/graphql", "/api/graphql"} {
			if _, ok := routeSet[gp]; !ok {
				routeSet[gp] = "graphql-auto"
			}
		}
	}

	routes := make([]string, 0, len(routeSet))
	for r := range routeSet {
		routes = append(routes, r)
	}
	sort.Strings(routes)

	fmt.Printf("\n%s[*]%s Verifikasi dinamis %d rute (worker=%d)...\n", cCyan, cReset, len(routes), cfg.Conc)

	// ===== MODUL BARU 4: baseline dummy-route =====
	initDummyBaseline()

	var (
		probes []RouteProbe
		mu     sync.Mutex
		sem    = make(chan struct{}, cfg.Conc)
		wg     sync.WaitGroup
	)
	baseBody, _, _ := fetch(cfg.Target.String())
	baseLen := int64(len(baseBody))
	doAuthDiff := cfg.AuthDiff && hasAuthHeaders()

	for _, r := range routes {
		wg.Add(1)
		sem <- struct{}{}
		go func(route string) {
			defer wg.Done()
			defer func() { <-sem }()
			full := strings.TrimRight(cfg.Target.String(), "/") + route
			code, clen, loc := probeNoFollow(full) // fungsi asli tetap dipakai
			verdict := classifyProbe(code, clen, loc, baseLen)

			p := RouteProbe{Route: route, FullPath: full, HTTPStatus: code,
				ContentLen: clen, RedirectTo: loc, Verdict: verdict, FromSource: routeSet[route]}

			// ===== MODUL BARU 4a: SPA Differential Response Matching =====
			if verdict == "ACCESSIBLE-NO-AUTH" {
				fp := fingerprintRoute(full)
				if matchesDummy(fp) {
					verdict = "SPA-404-MASKED"
					p.Verdict = verdict
					p.DiffNote = "respons identik dengan rute dummy (404 dibungkus 200)"
				}
			}

			// ===== MODUL BARU 4b: Header & Auth Context Injection (Diff) =====
			if doAuthDiff {
				anonCode, _, _ := probeNoFollowEx(full, false)
				p.AnonStatus = anonCode
				switch {
				case code == 200 && anonCode == 200:
					p.DiffNote = appendNote(p.DiffNote, "200 dengan & tanpa auth → cek BFLA di API backend")
					addFinding(Finding{Severity: "HIGH", Type: "auth-diff", Value: route,
						Asset: routeSet[route], HTTPStat: code,
						Verified: "respons 200 identik dengan/tanpa kredensial — proteksi kemungkinan hanya client-side"})
				case code == 200 && (anonCode == 401 || anonCode == 403 || anonCode == 302):
					p.DiffNote = appendNote(p.DiffNote, "auth ditegakkan di server (anon → "+fmt.Sprint(anonCode)+")")
				case (code == 401 || code == 403) && anonCode == 200:
					p.DiffNote = appendNote(p.DiffNote, "ANOMALI: anonim justru 200")
					addFinding(Finding{Severity: "CRITICAL", Type: "auth-diff-anomaly", Value: route,
						Asset: routeSet[route], HTTPStat: anonCode,
						Verified: "tanpa kredensial = 200, dengan kredensial = ditolak"})
				}
			}

			mu.Lock()
			probes = append(probes, p)
			mu.Unlock()

			col := cGray
			switch verdict {
			case "ACCESSIBLE-NO-AUTH":
				col = cRed
			case "REDIRECT-AUTH":
				col = cYellow
			case "SPA-FALLBACK(cek manual)", "SPA-404-MASKED":
				col = cCyan
			}
			fmt.Printf("  %s%-3d%s %-46s → %s\n", col, code, cReset, truncate(route, 46), verdict)
		}(r)
	}
	wg.Wait()

	for _, p := range probes {
		if p.Verdict == "ACCESSIBLE-NO-AUTH" {
			addFinding(Finding{Severity: "CRITICAL", Type: "unprotected-route",
				Value: p.Route, Asset: p.FromSource, HTTPStat: p.HTTPStatus,
				Verified: "200 OK tanpa redirect auth — UI sensitif kemungkinan dirender"})
		}
	}
	return probes
}

func appendNote(existing, add string) string {
	if existing == "" {
		return add
	}
	return existing + "; " + add
}

func classifyProbe(code int, clen int64, loc string, baseLen int64) string {
	switch {
	case code == 0:
		return "NO-RESPONSE"
	case code >= 300 && code < 400:
		low := strings.ToLower(loc)
		if strings.Contains(low, "login") || strings.Contains(low, "auth") ||
			strings.Contains(low, "signin") || strings.Contains(low, "sso") {
			return "REDIRECT-AUTH"
		}
		return "REDIRECT"
	case code == 401 || code == 403:
		return "PROTECTED"
	case code == 404:
		return "NOT-FOUND-SERVER"
	case code == 200:
		if baseLen > 0 && clen > 0 && absDiff(clen, baseLen) < baseLen/10 {
			return "SPA-FALLBACK(cek manual)"
		}
		return "ACCESSIBLE-NO-AUTH"
	default:
		return fmt.Sprintf("HTTP-%d", code)
	}
}

func absDiff(a, b int64) int64 {
	if a > b {
		return a - b
	}
	return b - a
}

func normalizeRoute(v string) string {
	v = strings.TrimSpace(v)
	if !strings.HasPrefix(v, "/") {
		if pu, err := url.Parse(v); err == nil && pu.Host != "" {
			if pu.Hostname() != cfg.Target.Hostname() {
				return ""
			}
			v = pu.Path
		} else {
			return ""
		}
	}
	if strings.ContainsAny(v, "${") {
		v = reTemplateLiteral.ReplaceAllString(v, "1")
	}
	v = strings.Split(v, "?")[0]
	if len(v) < 2 || (!isRoutePath(v) && !strings.HasPrefix(v, "/api/")) {
		return ""
	}
	return v
}

// ============================================================
// ===== MODUL BARU 5: BURP SUITE SITEMAP XML EXPORT =====
// ============================================================

func exportBurpSitemap(outPath string) {
	var sb strings.Builder
	sb.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<items burpVersion=\"2024.x\" exportTime=\"" +
		time.Now().Format(time.RFC3339) + "\">\n")

	emit := func(u string) {
		pu, err := url.Parse(u)
		if err != nil {
			return
		}
		raw := fmt.Sprintf("GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: DevRoute-Inspector\r\nAccept: */*\r\nConnection: close\r\n\r\n",
			pu.RequestURI(), pu.Host)
		sb.WriteString("  <item>\n")
		sb.WriteString("    <time>" + time.Now().Format("Mon Jan 02 15:04:05 MST 2006") + "</time>\n")
		sb.WriteString("    <url><![CDATA[" + u + "]]></url>\n")
		sb.WriteString("    <request base64=\"true\">" + base64.StdEncoding.EncodeToString([]byte(raw)) + "</request>\n")
		sb.WriteString("  </item>\n")
	}

	for _, p := range report.Routes {
		emit(p.FullPath)
	}
	findMutex.Lock()
	snap := append([]Finding(nil), findings...)
	findMutex.Unlock()
	for _, f := range snap {
		switch f.Type {
		case "api", "api-candidate", "server-action":
			v := f.Value
			if strings.HasPrefix(v, "actionId: ") {
				emit(strings.TrimRight(cfg.Target.String(), "/") + "/?_rsc=1")
				continue
			}
			if u := absURL(v); u != "" && sameHost(u) {
				emit(u)
			}
		}
	}
	sb.WriteString("</items>\n")
	os.WriteFile(outPath, []byte(sb.String()), 0o644)
	fmt.Printf("%s[+]%s Burp Sitemap XML → %s (import: Burp → Project → Import items)\n", cGreen, cReset, outPath)
}

// ============================================================
// ===== MODUL BARU 5: HTML INTERACTIVE REPORT (SINGLE-FILE) =====
// HOTFIX: seluruh template literal JS (`...${...}`) diganti konkatenasi
// string kutip-tunggal agar tidak ada backtick di dalam raw string Go.
// ============================================================

const htmlShell = `<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>DevRoute-Inspector — {{TARGET}}</title>
<style>
 body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}
 h1{color:#58a6ff;font-size:20px} .meta{color:#8b949e;font-size:12px;margin-bottom:16px}
 .bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
 input,select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font:inherit}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #21262d;vertical-align:top}
 th{color:#8b949e;position:sticky;top:0;background:#0d1117}
 tr:hover{background:#161b22}
 .CRITICAL{color:#ff7b72;font-weight:700}.HIGH{color:#e3b341}.MEDIUM{color:#58a6ff}
 .LOW{color:#8b949e}.INFO{color:#3fb950}
 .cnt{display:inline-block;margin-right:12px;font-size:12px}
</style></head><body>
<h1>DevRoute-Inspector v2.0 — {{TARGET}}</h1>
<div class="meta">{{META}}</div>
<div class="meta" id="counts"></div>
<div class="bar">
 <input id="q" placeholder="cari value/context/asset..." size="36">
 <select id="sev"><option value="">Semua Severity</option></select>
 <select id="typ"><option value="">Semua Type</option></select>
</div>
<table><thead><tr><th>Severity</th><th>Type</th><th>Value</th><th>Context</th><th>Asset</th><th>HTTP</th><th>Catatan</th></tr></thead>
<tbody id="tb"></tbody></table>
<script>
var DATA = {{DATA}};
var tb=document.getElementById('tb'),q=document.getElementById('q'),
    sev=document.getElementById('sev'),typ=document.getElementById('typ');
function uniq(arr){var s={},o=[];for(var i=0;i<arr.length;i++){if(!s[arr[i]]){s[arr[i]]=1;o.push(arr[i]);}}return o;}
var sevs=uniq(DATA.map(function(f){return f.severity;}));
var typs=uniq(DATA.map(function(f){return f.type;}));
for(var i=0;i<sevs.length;i++){sev.add(new Option(sevs[i],sevs[i]));}
for(var i=0;i<typs.length;i++){typ.add(new Option(typs[i],typs[i]));}
function esc(x){var d=document.createElement('div');d.textContent=x||'';return d.innerHTML;}
function render(){
 var s=q.value.toLowerCase(),sv=sev.value,ty=typ.value;
 var rows=DATA.filter(function(f){
   return (!sv||f.severity===sv)&&(!ty||f.type===ty)&&
     (!s||((f.value||'')+(f.context||'')+(f.asset||'')).toLowerCase().indexOf(s)>=0);
 });
 tb.innerHTML=rows.map(function(f){
   return '<tr><td class="'+f.severity+'">'+f.severity+'</td><td>'+f.type+'</td>'+
     '<td>'+esc(f.value)+'</td><td>'+esc(f.context||'')+'</td><td>'+esc(f.asset||'')+'</td>'+
     '<td>'+(f.http_status||'')+'</td><td>'+esc(f.verified||'')+'</td></tr>';
 }).join('');
 var c={};
 for(var i=0;i<rows.length;i++){c[rows[i].severity]=(c[rows[i].severity]||0)+1;}
 var parts=[];
 for(var k in c){parts.push('<span class="cnt '+k+'">'+k+': '+c[k]+'</span>');}
 document.getElementById('counts').innerHTML=parts.join('')+'<span class="cnt">total: '+rows.length+'</span>';
}
q.oninput=render;sev.onchange=render;typ.onchange=render;render();
</script></body></html>`

func exportHTMLReport(outPath string) {
	findMutex.Lock()
	b, _ := json.Marshal(findings)
	findMutex.Unlock()
	meta := fmt.Sprintf("Dipindai: %s | Aset: %d JS, %d sourcemap | Rute diverifikasi: %d",
		report.StartedAt, report.AssetsTotal, report.SMapsTotal, len(report.Routes))
	page := strings.ReplaceAll(htmlShell, "{{TARGET}}", html.EscapeString(cfg.Target.String()))
	page = strings.ReplaceAll(page, "{{META}}", html.EscapeString(meta))
	page = strings.ReplaceAll(page, "{{DATA}}", string(b))
	os.WriteFile(outPath, []byte(page), 0o644)
	fmt.Printf("%s[+]%s Laporan HTML interaktif → %s (buka di browser: filter severity/type + pencarian)\n",
		cGreen, cReset, outPath)
}

// ============================================================
//  LAPORAN: TERMINAL BERWARNA + JSON
// ============================================================

var report Report

func printSummary() {
	summary := map[string]int{}
	for _, f := range report.Findings {
		summary[f.Severity]++
	}
	report.Summary = summary

	fmt.Printf("\n%s════════════════ RINGKASAN ════════════════%s\n", cBold, cReset)
	fmt.Printf(" Target          : %s\n", cfg.Target.String())
	reportMu.Lock()
	fmt.Printf(" Aset dipindai   : %d JS | %d sourcemap\n", report.assets, report.smaps)
	reportMu.Unlock()
	order := []string{"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
	for _, s := range order {
		if n := summary[s]; n > 0 {
			fmt.Printf(" %s%-8s%s : %d temuan\n", sevColor[s], s, cReset, n)
		}
	}
	if len(report.Routes) > 0 {
		accessible, masked := 0, 0
		for _, p := range report.Routes {
			if p.Verdict == "ACCESSIBLE-NO-AUTH" {
				accessible++
			}
			if p.Verdict == "SPA-404-MASKED" {
				masked++
			}
		}
		fmt.Printf(" Rute terverifikasi tanpa auth : %s%d%s\n", cRed, accessible, cReset)
		fmt.Printf(" Rute SPA-404-MASKED (diff)    : %s%d%s\n", cCyan, masked, cReset)
	}
	fmt.Printf("%s═══════════════════════════════════════════%s\n\n", cBold, cReset)

	if cfg.JSONOut != "" {
		b, _ := json.MarshalIndent(report, "", "  ")
		os.WriteFile(cfg.JSONOut, b, 0o644)
		fmt.Printf("%s[+]%s Laporan JSON → %s\n", cGreen, cReset, cfg.JSONOut)
	}
}

func sortFindings() {
	rank := map[string]int{"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
	sort.SliceStable(report.Findings, func(i, j int) bool {
		return rank[report.Findings[i].Severity] < rank[report.Findings[j].Severity]
	})
}

// ============================================================
//  MAIN
// ============================================================

func main() {
	var (
		fTarget   = flag.String("u", "", "URL target (wajib), mis. https://example.com")
		fHeaders  headerList
		fConc     = flag.Int("c", 10, "jumlah worker konkuren")
		fTimeout  = flag.Int("timeout", 15, "timeout HTTP per-request (detik)")
		fInsec    = flag.Bool("insecure", false, "abaikan sertifikat TLS invalid")
		fProxy    = flag.String("proxy", "", "proxy HTTP, mis. http://127.0.0.1:8080 (Burp)")
		fNoVerify = flag.Bool("no-verify", false, "lewati verifikasi rute dinamis (static-only)")
		fJSON     = flag.String("json", "devroute-report.json", "path output JSON (kosongkan untuk nonaktif)")
		fDump     = flag.String("dump", "", "direktori rekonstruksi source dari sourcemap")
		fMaxAsset = flag.Int("max-assets", 150, "batas maksimum aset JS yang dipindai")
		// ===== MODUL BARU: flag v2.0 =====
		fAuthDiff = flag.Bool("auth-diff", false, "aktifkan diff probe authenticated vs anonim (butuh -H Authorization/Cookie)")
		fRSC      = flag.Bool("rsc", false, "probe Next.js App Router (RSC payload, header RSC: 1)")
		fBurp     = flag.String("burp", "", "ekspor Burp Sitemap XML ke path ini")
		fHTML     = flag.String("html", "devroute-report.html", "ekspor laporan HTML interaktif (kosongkan untuk nonaktif)")
		fNoAST    = flag.Bool("no-ast", false, "matikan AST-Light solver (template literal / object map / concat)")
	)
	flag.Var(&fHeaders, "H", `header tambahan "Key: Value" (boleh berulang)`)
	flag.Parse()

	fmt.Print(cCyan + banner + cReset)

	if *fTarget == "" {
		fmt.Fprintln(os.Stderr, "ERROR: -u wajib diisi. Contoh: ./devroute-inspector -u https://target.com")
		flag.Usage()
		os.Exit(1)
	}
	tu, err := url.Parse(*fTarget)
	if err != nil || (tu.Scheme != "http" && tu.Scheme != "https") {
		fmt.Fprintln(os.Stderr, "ERROR: URL target tidak valid.")
		os.Exit(1)
	}

	cfg = Config{
		Target: tu, Headers: fHeaders, Conc: *fConc,
		Timeout:  time.Duration(*fTimeout) * time.Second,
		Insecure: *fInsec, Proxy: *fProxy, Verify: !*fNoVerify,
		JSONOut: *fJSON, DumpDir: *fDump, MaxAssets: *fMaxAsset,
		MaxBody: 25 * 1024 * 1024,
		AuthDiff: *fAuthDiff, RSCProbe: *fRSC,
		BurpXML: *fBurp, HTMLOut: *fHTML, NoAST: *fNoAST,
	}
	client = buildClient()

	start := time.Now()
	report.Target = cfg.Target.String()

	// --- FASE 1: Static Asset Fetcher ---
	fmt.Printf("%s[FASE 1]%s Mengunduh halaman utama & inventarisasi aset JS...\n", cBold, cReset)
	htmlBody, code, err := fetch(cfg.Target.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: gagal fetch target: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  %s[+]%s HTTP %d, %d KB\n", cGreen, cReset, code, len(htmlBody)/1024)

	analyzeContent(htmlBody, cfg.Target.String())

	assets := extractAssetRefs(htmlBody, cfg.Target.String())
	if buildID := extractBuildID(htmlBody); buildID != "" {
		assets = append(assets,
			absURL("/_next/static/"+buildID+"/_buildManifest.js"),
			absURL("/_next/static/"+buildID+"/_ssgManifest.js"),
		)
		fmt.Printf("  %s[+]%s Next.js buildId terdeteksi: %s — manifest di-seed otomatis\n",
			cGreen, cReset, buildID)
	}
	assets = append(assets, absURL("/build/manifest.json"))
	// ===== MODUL BARU 2: seed SvelteKit manifest & webpack runtime umum =====
	assets = append(assets,
		absURL("/_app/immutable/start.js"),
		absURL("/static/js/runtime~main.js"),
		absURL("/runtime~main.js"),
	)

	// --- FASE 2: AST/Regex Extractor + Source Map Unpacker (konkuren) ---
	fmt.Printf("\n%s[FASE 2]%s Ekstraksi rute, state, secret dari %d aset (+chained discovery)...\n",
		cBold, cReset, len(assets))
	var wg sync.WaitGroup
	sem := make(chan struct{}, cfg.Conc)
	for _, a := range assets {
		if assetCount() >= cfg.MaxAssets {
			break
		}
		if !markSeen(a) {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(u string) {
			defer wg.Done()
			defer func() { <-sem }()
			scanAsset(u, 0)
		}(a)
	}
	wg.Wait()
	time.Sleep(2 * time.Second) // tunggu chunk discovery goroutine

	// ===== MODUL BARU 2: FASE 2.5 — RSC / Server Actions probe =====
	probeRSC()

	// --- FASE 3: Dynamic Route Reconstruct (+ Diff Probe) ---
	if cfg.Verify {
		fmt.Printf("\n%s[FASE 3]%s Dynamic Route Verification...\n", cBold, cReset)
		report.Routes = verifyRoutes()
	}

	// --- FASE 4: Laporan ---
	report.StartedAt = start.Format(time.RFC3339)
	report.FinishedAt = time.Now().Format(time.RFC3339)
	reportMu.Lock()
	report.AssetsTotal = report.assets
	report.SMapsTotal = report.smaps
	reportMu.Unlock()

	findMutex.Lock()
	report.Findings = append([]Finding(nil), findings...)
	findMutex.Unlock()
	sortFindings()
	printSummary()

	// ===== MODUL BARU 5: FASE 5 — Ekspor integrasi workflow =====
	if cfg.BurpXML != "" {
		exportBurpSitemap(cfg.BurpXML)
	}
	if cfg.HTMLOut != "" {
		exportHTMLReport(cfg.HTMLOut)
	}
}

func extractBuildID(htmlStr string) string {
	if m := reBuildID.FindStringSubmatch(htmlStr); m != nil {
		return m[1]
	}
	if m := reBuildManifestRef.FindStringSubmatch(htmlStr); m != nil {
		return m[1]
	}
	return ""
}

type headerList []string

func (h *headerList) String() string     { return strings.Join(*h, ", ") }
func (h *headerList) Set(v string) error { *h = append(*h, v); return nil }

