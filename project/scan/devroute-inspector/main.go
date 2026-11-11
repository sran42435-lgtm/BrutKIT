// DevRoute-Inspector (AOT-Route & State Scanner)
// Frontend Asset & Client-Side Logic Pentesting untuk SPA modern
// (Next.js, Nuxt, Remix, Vite/React)
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
  DevRoute-Inspector v1.3 — AOT-Route & State Scanner
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

	// counter internal — race-safe, tidak diekspor ke JSON
	assets int `json:"-"`
	smaps  int `json:"-"`
}

// reportMu melindungi report.assets / report.smaps dari race condition
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
				return http.ErrUseLastResponse // blokir redirect keluar host (SSRF guard)
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
	if pu.Host == "" { // relatif
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

// probeNoFollow: untuk verifikasi rute, catat redirect eksplisit
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

// ============================================================
//  MESIN EKSTRAKSI (Regex Berlapis ala jsluice matcher)
// ============================================================

type Matcher struct {
	Name     string
	Type     string // "route" | "api" | "secret" | "state" | "url"
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
	q := `["'` + "`" + `]` // kutip: " ' atau backtick
	Q := func(inner string) string { return q + `(` + inner + `)` + q }

	return []Matcher{
		// --- 1. ROUTE EKSAK dari pola framework (bukan tebakan wordlist) ---
		{Name: "react-router-path", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`path\s*:\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\$\{\}\.]{1,120}`)), Validate: isRoutePath},
		{Name: "route-component-tag", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`<Route[^>]+path\s*=\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\.]{1,120}`)), Validate: isRoutePath},
		{Name: "createBrowserRouter-def", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`createBrowserRouter|createHashRouter|createRoutesFromElements`), Group: -1},

		// --- Next.js _buildManifest.js — key objek adalah rute pasti ---
		{Name: "next-buildmanifest-route", Type: "route", Severity: "HIGH",
			Re: regexp.MustCompile(`"(\/[^"]{1,120})"\s*:\s*$$\s*"static\/`), Group: 1, Validate: isRoutePath},
		{Name: "next-page-route", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`["']\/_next\/(?:data|static)\/[^"']+["']|buildId["']?\s*:\s*["']([A-Za-z0-9_\-]{5,30})["']`)},

		// --- Remix manifest ---
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

		// --- 2. API / ENDPOINT INTERNAL ---
		{Name: "fetch-axios-call", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`(?:fetch|axios\.(?:get|post|put|delete|patch)|\.ajax)\s*$\s*` + Q(`[^"'`+"`"+`]{3,200}`))},
		{Name: "api-path", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(Q(`\/(?:api|internal|graphql|v[0-9]+)\/[A-Za-z0-9_\-\/:\.]{1,120}`)), Validate: isAPIPath},
		{Name: "external-api-host", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`https?:\/\/[A-Za-z0-9\.\-]*(?:api|internal|staging|dev|admin)[A-Za-z0-9\.\-]*\.[a-z]{2,}(?:\/[^\s"'` + "`" + `<>]{0,80})?`)},

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
		// HOTFIX: {0,2000} → {0,1000} (RE2 membatasi repeat count maks 1000)
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
			if m.Group == -1 { // deteksi pola tanpa nilai (marker framework)
				addFinding(Finding{Severity: "INFO", Type: "framework-marker",
					Value: m.Name, Asset: assetURL, Context: truncate(mm[0], 80)})
				continue
			}
			if m.Group == -2 { // laporkan nama key state yang sensitif
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
	for _, re := range []*regexp.Regexp{reScriptSrc, reLinkHref, reJSString, reNextChunk} {
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
	if !sameHost(mapURL) { // SSRF guard: abaikan sourcemap lintas host
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
	incSmaps() // race-safe
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
		return // bukan JS (SPA fallback), jangan dianalisis sebagai bundle
	}
	fmt.Printf("  %s[*]%s Scanning asset (%d KB): %s\n", cCyan, cReset, len(body)/1024, truncate(jsURL, 80))
	n := incAssets() // race-safe

	// 1) analisis bundle
	analyzeContent(body, jsURL)

	// 2) sourcemap: referensi eksternal + coba <js>.map + inline data:
	if mm := reSourceMap.FindStringSubmatch(body); mm != nil {
		handleSourceMapURL(mm[1], jsURL, body)
	} else {
		handleSourceMapURL(jsURL+".map", jsURL, "")
	}

	// 3) discovery chunk tambahan (code-splitting) — 1 level saja agar efisien
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
	}
	routes := make([]string, 0, len(routeSet))
	for r := range routeSet {
		routes = append(routes, r)
	}
	sort.Strings(routes)

	fmt.Printf("\n%s[*]%s Verifikasi dinamis %d rute (worker=%d)...\n", cCyan, cReset, len(routes), cfg.Conc)

	var (
		probes []RouteProbe
		mu     sync.Mutex
		sem    = make(chan struct{}, cfg.Conc)
		wg     sync.WaitGroup
	)
	baseBody, _, _ := fetch(cfg.Target.String())
	baseLen := int64(len(baseBody))

	for _, r := range routes {
		wg.Add(1)
		sem <- struct{}{}
		go func(route string) {
			defer wg.Done()
			defer func() { <-sem }()
			full := strings.TrimRight(cfg.Target.String(), "/") + route
			code, clen, loc := probeNoFollow(full)
			verdict := classifyProbe(code, clen, loc, baseLen)
			p := RouteProbe{Route: route, FullPath: full, HTTPStatus: code,
				ContentLen: clen, RedirectTo: loc, Verdict: verdict, FromSource: routeSet[route]}
			mu.Lock()
			probes = append(probes, p)
			mu.Unlock()

			col := cGray
			switch verdict {
			case "ACCESSIBLE-NO-AUTH":
				col = cRed
			case "REDIRECT-AUTH":
				col = cYellow
			case "SPA-FALLBACK(cek manual)":
				col = cCyan
			}
			fmt.Printf("  %s%-3d%s %-50s → %s\n", col, code, cReset, truncate(route, 50), verdict)
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
				return "" // rute eksternal: jangan di-probe (scope guard)
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
//  LAPORAN: TERMINAL BERWARNA + JSON
// ============================================================

var report Report

// printSummary hanya menghitung ringkasan, mencetak, dan menulis JSON sekali.
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
		accessible := 0
		for _, p := range report.Routes {
			if p.Verdict == "ACCESSIBLE-NO-AUTH" {
				accessible++
			}
		}
		fmt.Printf(" Rute terverifikasi tanpa auth : %s%d%s\n", cRed, accessible, cReset)
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
	}
	client = buildClient()

	start := time.Now()
	report.Target = cfg.Target.String()

	// --- FASE 1: Static Asset Fetcher ---
	fmt.Printf("%s[FASE 1]%s Mengunduh halaman utama & inventarisasi aset JS...\n", cBold, cReset)
	html, code, err := fetch(cfg.Target.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: gagal fetch target: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  %s[+]%s HTTP %d, %d KB\n", cGreen, cReset, code, len(html)/1024)

	// analisis HTML utama (__NEXT_DATA__ sering ada di HTML)
	analyzeContent(html, cfg.Target.String())

	// seed otomatis _buildManifest.js & _ssgManifest.js (Next.js)
	assets := extractAssetRefs(html, cfg.Target.String())
	if buildID := extractBuildID(html); buildID != "" {
		assets = append(assets,
			absURL("/_next/static/"+buildID+"/_buildManifest.js"),
			absURL("/_next/static/"+buildID+"/_ssgManifest.js"),
		)
		fmt.Printf("  %s[+]%s Next.js buildId terdeteksi: %s — manifest di-seed otomatis\n",
			cGreen, cReset, buildID)
	}
	// Remix: manifest routes umum
	assets = append(assets, absURL("/build/manifest.json"))

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

	// --- FASE 3: Dynamic Route Reconstruct ---
	if cfg.Verify {
		fmt.Printf("\n%s[FASE 3]%s Dynamic Route Verification...\n", cBold, cReset)
		report.Routes = verifyRoutes()
	}

	// --- FASE 4: Laporan (snapshot → sort → cetak + tulis JSON sekali) ---
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
}

func extractBuildID(html string) string {
	if m := reBuildID.FindStringSubmatch(html); m != nil {
		return m[1]
	}
	if m := reBuildManifestRef.FindStringSubmatch(html); m != nil {
		return m[1]
	}
	return ""
}

// flag.Var untuk header berulang
type headerList []string

func (h *headerList) String() string     { return strings.Join(*h, ", ") }
func (h *headerList) Set(v string) error { *h = append(*h, v); return nil }

