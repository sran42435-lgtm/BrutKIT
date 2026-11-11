// DevRoute-Inspector (AOT-Route & State Scanner)
// Frontend Asset & Client-Side Logic Pentesting untuk SPA modern
// (Next.js App/Pages, Nuxt 3, SvelteKit, Vite, Remix, Webpack Micro-frontends)
//
// Build : go build -o devroute-inspector main.go
// Pakai : ./devroute-inspector -u https://target.com
//
// v3.0 — Advanced Modules:
//   [A] Hydration Diffing | Entropy Analysis | PII Leak
//   [B] Differential Probing | SPA Fallback Detection (Jaccard)
//   [C] RSC / Server Actions Exposer (Next-Action Probe)
//   [D] Authz Matrix Mapping | Client-Side Guard Analysis
//   [E] SourceMap AST-lite | In-Memory Tree Reconstruction
//   [F] Modern Bundler Support (Vite/Webpack Chunk Graph/Nuxt/Svelte)
//   [G] WAF Evasion Headers | Deep Chain Discovery (Depth 3)
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
	"math"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
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
  DevRoute-Inspector v3.0 — AOT-Route & State Scanner
  + Next.js App/Vite/Nuxt/Svelte | Webpack Chunk Graph
  + RSC Actions | Diff Probing | WAF Evasion
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

	// [F] Modern Bundler Patterns
	reModulePreload    = regexp.MustCompile(`<link[^>]+rel\s*=\s*["']modulepreload["'][^>]+href\s*=\s*["']([^"']+)["']`)
	reDynamicImport    = regexp.MustCompile(`import\s*$\s*["'` + "`" + `](\/[^"'`+"`"+`]+)["'` + "`" + `]`)
	reViteManifest     = regexp.MustCompile(`/.vite/manifest\.json|/manifest\.json`)
	reNuxtAsset        = regexp.MustCompile(`/_nuxt/[^"'` + "`" + `\s]+\.(?:js|mjs)`)
	reSvelteAsset      = regexp.MustCompile(`/_app/immutable/[^"'` + "`" + `\s]+\.(?:js|mjs)`)
	reNextAppChunk     = regexp.MustCompile(`/_next/static/chunks/app/[^"'` + "`" + `\s]+\.js`)
	reNextMainApp      = regexp.MustCompile(`/_next/static/chunks/main-app-[^"'` + "`" + `\s]+\.js`)
	reFlightData       = regexp.MustCompile(`self\.__next_f\.push$\s*$$\s*1\s*,\s*"`) // RSC Flight marker

	// [F] Webpack Chunk Graph: { id: "hash", ... } atau webpackRequire.e(id)
	reWebpackChunkMap = regexp.MustCompile(`\{\s*(\d+)\s*:\s*["']([a-f0-9]{8,32})["']`)
	reWebpackRequireE = regexp.MustCompile(`__webpack_require__\.e\s*$\s*(\d+)\s*$`)

	// [B] Tokenisasi HTML untuk Jaccard similarity
	reHTMLTag = regexp.MustCompile(`<\s*/?\s*([a-zA-Z][a-zA-Z0-9\-]*)`)

	// [C] Server Action ID extraction
	reCreateServerRef = regexp.MustCompile(`createServerReference\s*$\s*["'` + "`" + `]([A-Za-z0-9+/_=\-]{16,128})["'` + "`" + `]`)
	reActionIDKey     = regexp.MustCompile(`(?i)action[_\-]?id["']?\s*[:=]\s*["'` + "`" + `]([A-Za-z0-9+/_=\-]{16,128})["'` + "`" + `]`)
	reActionHexInRSC  = regexp.MustCompile(`["'` + "`" + `]([0-9a-f]{40,64})["'` + "`" + `]`)
	reRSCHint         = regexp.MustCompile(`server-reference|Next-Action|createServerReference|__next_f|react-server`)

	// [E] AST-lite token patterns
	reZodYupSchema   = regexp.MustCompile(`(\w+)\s*:\s*(?:z|yup|Joi|v)\.(?:string|number|boolean|object|array|enum)`)
	reHardcodedRole  = regexp.MustCompile(`(?:role|roles)\s*[!=]==?\s*["'` + "`" + `][\w\-:]{2,40}["'` + "`" + `]`)
	reClientTokenSt  = regexp.MustCompile(`(?:localStorage|sessionStorage)\.(?:getItem|setItem)\s*$\s*["'` + "`" + `][^"'` + "`" + `]*(?:token|auth|jwt|session)[^"'` + "`" + `]*["'` + "`" + `]`)
	reProcessEnv     = regexp.MustCompile(`process\.env\.([A-Z][A-Z0-9_]{2,50})`)
	reDangerousHTML  = regexp.MustCompile(`dangerouslySetInnerHTML|\beval\s*$|new\s+Function\s*$`)
	reGraphQLResolv  = regexp.MustCompile(`(?i)(?:resolvers?\s*[:=]|@Query|@Mutation|type\s+Query)`)
	reSwaggerHint    = regexp.MustCompile(`(?i)(swagger|openapi)["']?\s*[:=]|"paths"\s*:\s*\{`)

	// [A] PII patterns
	rePIIEmail = regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`)
	rePIIPhone = regexp.MustCompile(`\+\d{1,3}[\s\-]?$?\d{2,4}$?[\s\-]?\d{3,4}[\s\-]?\d{3,4}`)
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

type AuthMatrixEntry struct {
	Asset        string   `json:"asset"`
	Routes       []string `json:"ui_routes,omitempty"`
	ClientChecks []string `json:"client_role_checks,omitempty"`
	Guards       []string `json:"client_guards,omitempty"`
	APIs         []string `json:"backend_apis,omitempty"`
}

type ActionProbe struct {
	ActionID   string `json:"action_id"`
	ProbeURL   string `json:"probe_url"`
	HTTPStatus int    `json:"http_status"`
	Verdict    string `json:"verdict"`
	FromAsset  string `json:"from_asset"`
}

type Report struct {
	Target        string            `json:"target"`
	StartedAt     string            `json:"started_at"`
	FinishedAt    string            `json:"finished_at"`
	AssetsTotal   int               `json:"assets_scanned"`
	SMapsTotal    int               `json:"sourcemaps_parsed"`
	Findings      []Finding         `json:"findings"`
	Summary       map[string]int    `json:"summary"`
	Routes        []RouteProbe      `json:"route_verification,omitempty"`
	AuthMatrix    []AuthMatrixEntry `json:"auth_matrix,omitempty"`
	ServerActions []ActionProbe     `json:"server_actions,omitempty"`
	RSCDetected   bool              `json:"rsc_endpoint_detected,omitempty"`

	// counter internal
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
	Route      string  `json:"route"`
	FullPath   string  `json:"full_url"`
	HTTPStatus int     `json:"http_status"`
	ContentLen int64   `json:"content_length"`
	RedirectTo string  `json:"redirect_to,omitempty"`
	Similarity float64 `json:"baseline_similarity,omitempty"`
	Verdict    string  `json:"verdict"`
	FromSource string  `json:"from_asset"`
}

// ============================================================
//  KONFIGURASI GLOBAL
// ============================================================

type Config struct {
	Target     *url.URL
	Headers    []string
	Conc       int
	Timeout    time.Duration
	Insecure   bool
	Proxy      string
	Verify     bool
	JSONOut    string
	DumpDir    string
	MaxAssets  int
	MaxBody    int64
	MaxActions int
	EntropyMin float64
	MaxDepth   int // [F] Kedalaman chain discovery
}

var cfg Config
var client *http.Client

// ============================================================
//  UTILITAS HTTP (WAF EVASION + SSRF GUARD)
// ============================================================

// [G] Browser Headers Realistis untuk bypass WAF dasar
func applyCommonHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,id;q=0.8")
	req.Header.Set("Sec-Ch-Ua", `"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"`)
	req.Header.Set("Sec-Ch-Ua-Mobile", "?0")
	req.Header.Set("Sec-Ch-Ua-Platform", `"Windows"`)
	req.Header.Set("Sec-Fetch-Dest", "document")
	req.Header.Set("Sec-Fetch-Mode", "navigate")
	req.Header.Set("Sec-Fetch-Site", "none")
	req.Header.Set("Sec-Fetch-User", "?1")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	for _, h := range cfg.Headers {
		kv := strings.SplitN(h, ":", 2)
		if len(kv) == 2 {
			req.Header.Set(strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
	}
}

func buildClient() *http.Client {
	jar, _ := cookiejar.New(nil)
	tr := &http.Transport{
		TLSClientConfig:     &tls.Config{InsecureSkipVerify: cfg.Insecure},
		MaxIdleConnsPerHost: cfg.Conc * 2,
		DisableCompression:  false, // Biarkan server kompresi, kita handle decompress jika perlu (Go otomatis)
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
				return http.ErrUseLastResponse
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
	applyCommonHeaders(req)
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

type probeResult struct {
	Status  int
	Headers http.Header
	Body    string
	Length  int64
}

const maxProbeBody = 512 * 1024

func doProbe(method, fullURL, body string, extraHeaders map[string]string) probeResult {
	empty := probeResult{Headers: http.Header{}}
	tr, ok := client.Transport.(*http.Transport)
	if !ok {
		return empty
	}
	probe := &http.Client{
		Transport: tr,
		Timeout:   cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, fullURL, rdr)
	if err != nil {
		return empty
	}
	applyCommonHeaders(req)
	// Override Sec-Fetch untuk probe API/Action
	if method == "POST" {
		req.Header.Set("Sec-Fetch-Dest", "empty")
		req.Header.Set("Sec-Fetch-Mode", "cors")
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}
	resp, err := probe.Do(req)
	if err != nil {
		return empty
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxProbeBody))
	return probeResult{
		Status:  resp.StatusCode,
		Headers: resp.Header,
		Body:    string(raw),
		Length:  int64(len(raw)),
	}
}

// ============================================================
//  MESIN EKSTRAKSI (Regex Berlapis)
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
		// --- ROUTES ---
		{Name: "react-router-path", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`path\s*:\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\$\{\}\.]{1,120}`)), Validate: isRoutePath},
		{Name: "route-component-tag", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`<Route[^>]+path\s*=\s*` + Q(`\/[A-Za-z0-9_\-\/:\*\.]{1,120}`)), Validate: isRoutePath},
		{Name: "createBrowserRouter-def", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`createBrowserRouter|createHashRouter|createRoutesFromElements`), Group: -1},
		
		// [F] Next.js App Router & Manifest
		{Name: "next-buildmanifest-route", Type: "route", Severity: "HIGH",
			Re: regexp.MustCompile(`"(\/[^"]{1,120})"\s*:\s*$$\s*"static\/`), Group: 1, Validate: isRoutePath},
		{Name: "next-page-route", Type: "route", Severity: "MEDIUM",
			Re: regexp.MustCompile(`["']\/_next\/(?:data|static)\/[^"']+["']|buildId["']?\s*:\s*["']([A-Za-z0-9_\-]{5,30})["']`)},
		
		// [F] Remix
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

		// --- API ---
		{Name: "fetch-axios-call", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`(?:fetch|axios\.(?:get|post|put|delete|patch)|\.ajax)\s*$\s*` + Q(`[^"'`+"`"+`]{3,200}`))},
		{Name: "api-path", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(Q(`\/(?:api|internal|graphql|v[0-9]+)\/[A-Za-z0-9_\-\/:\.]{1,120}`)), Validate: isAPIPath},
		{Name: "external-api-host", Type: "api", Severity: "HIGH",
			Re: regexp.MustCompile(`https?:\/\/[A-Za-z0-9\.\-]*(?:api|internal|staging|dev|admin)[A-Za-z0-9\.\-]*\.[a-z]{2,}(?:\/[^\s"'` + "`" + `<>]{0,80})?`)},

		// --- SECRETS ---
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

		// --- STATE / AUTH ---
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

		// --- PERMISSIONS / GUARDS ---
		{Name: "permission-string", Type: "permission", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:hasPermission|checkPermission|hasRole|canAccess|isAllowed|authorize|can)\s*$\s*` + Q(`[A-Za-z0-9_:.\-*]{2,60}`))},
		{Name: "role-array-def", Type: "auth", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?i)roles?\s*[:=]\s*$$[^$$]{1,300}$$`), Group: -1},
		{Name: "vue-router-guard", Type: "guard", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?:router\.)?beforeEach\s*$|beforeResolve\s*$|defineNuxtRouteMiddleware`), Group: -1},
		{Name: "react-auth-hoc", Type: "guard", Severity: "MEDIUM",
			Re: regexp.MustCompile(`withAuth\s*$|ProtectedRoute|RequireAuth|requireAuth|AuthGate|PrivateRoute`), Group: -1},
		{Name: "client-authz-check", Type: "guard", Severity: "MEDIUM",
			Re: regexp.MustCompile(`(?i)isAuthenticated|isLoggedIn|hasValidToken|verifyToken\s*$`), Group: -1},
	}
}

// ============================================================
//  ANALISIS KONTEN
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
	if col == "" {
		col = cGray
	}
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
				sev := m.Severity
				typ := m.Type
				if typ == "guard" || typ == "auth" {
					addFinding(Finding{Severity: sev, Type: typ,
						Value: m.Name + ": " + truncate(mm[0], 60), Asset: assetURL,
						Context: truncate(mm[0], 100)})
					continue
				}
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
// [A] HYDRATION DIFFING
// ============================================================

type HydrationTarget struct {
	Path  string
	Value interface{}
}

const (
	maxFlattenEntries = 5000
	maxHydrationFind  = 120
	maxJSONScan       = 2 << 20
)

var hydrationFindCount int64

func extractBalanced(s string, idx int, open, close byte) string {
	start := strings.IndexByte(s[idx:], open)
	if start < 0 {
		return ""
	}
	start += idx
	limit := len(s)
	if start+maxJSONScan < limit {
		limit = start + maxJSONScan
	}
	depth := 0
	inStr := false
	esc := false
	for j := start; j < limit; j++ {
		c := s[j]
		if inStr {
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return s[start : j+1]
			}
		}
	}
	return ""
}

func flattenJSON(prefix string, v interface{}, out *[]HydrationTarget, depth int) {
	if depth > 12 || len(*out) >= maxFlattenEntries {
		return
	}
	switch t := v.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			p := k
			if prefix != "" {
				p = prefix + "." + k
			}
			flattenJSON(p, t[k], out, depth+1)
		}
	case []interface{}:
		for i, val := range t {
			flattenJSON(fmt.Sprintf("%s[%d]", prefix, i), val, out, depth+1)
		}
	default:
		*out = append(*out, HydrationTarget{Path: prefix, Value: v})
	}
}

func shannonEntropy(s string) float64 {
	if s == "" {
		return 0
	}
	runes := []rune(s)
	freq := make(map[rune]int, 64)
	for _, r := range runes {
		freq[r]++
	}
	n := float64(len(runes))
	ent := 0.0
	for _, f := range freq {
		p := float64(f) / n
		ent -= p * math.Log2(p)
	}
	return ent
}

var benignHydrationKeys = []string{
	"buildid", "assetprefix", "chunk", "filename", "jsfilename", "cssfilename",
	"webpack", "hash", "integrity", "src", "href", "url", "path", "route",
	"locale", "timezone", "version", "githash", "commit", "runtimeconfig.public",
}

var reHexOnly = regexp.MustCompile(`^[0-9a-fA-F]{32,64}$`)

func isBenignHydrationPath(p string) bool {
	low := strings.ToLower(p)
	for _, b := range benignHydrationKeys {
		if strings.Contains(low, b) {
			return true
		}
	}
	return false
}

func isHighEntropy(path, val string) bool {
	if len(val) < 20 || len(val) > 512 {
		return false
	}
	if reHexOnly.MatchString(val) {
		return false
	}
	if strings.Contains(val, "/") || strings.HasPrefix(val, "http") || strings.Contains(val, ".js") {
		return false
	}
	if isBenignHydrationPath(path) {
		return false
	}
	hasDigit, hasAlpha := false, false
	for _, c := range val {
		if c >= '0' && c <= '9' {
			hasDigit = true
		}
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			hasAlpha = true
		}
	}
	if !(hasDigit && hasAlpha) {
		return false
	}
	return shannonEntropy(val) >= cfg.EntropyMin
}

var sensitiveHydrationKeys = []string{
	"isadmin", "isstaff", "issuperuser", "role", "roles", "permission", "permissions",
	"internalip", "internal_ip", "privateip", "debugconfig", "debug", "featureflag",
	"featureflags", "secret", "token", "apikey", "api_key", "authorization", "password",
	"privatekey", "ssn", "creditcard", "cardnumber", "cvv", "cookie", "sessionid",
	"accesstoken", "refreshtoken", "idtoken", "awsaccesskey", "databaseurl", "dsn",
}

func hasSensitiveKey(path string) string {
	low := strings.ToLower(path)
	seg := low
	if i := strings.LastIndex(low, "."); i >= 0 {
		seg = low[i+1:]
	}
	seg = strings.TrimSuffix(seg, "]")
	for _, k := range sensitiveHydrationKeys {
		if seg == k || strings.Contains(seg, k) {
			return k
		}
	}
	return ""
}

func analyzeHydrationState(html, sourceURL string) {
	globals := []struct {
		Name string
		Re   *regexp.Regexp
	}{
		{"__NEXT_DATA__", regexp.MustCompile(`(?:window\.)?__NEXT_DATA__\s*=\s*`)},
		{"__NUXT__", regexp.MustCompile(`(?:window\.)?__NUXT(?:_DATA)?__\s*=\s*`)},
		{"__remixContext", regexp.MustCompile(`(?:window\.)?__remixContext\s*=\s*`)},
		{"__PRELOADED_STATE__", regexp.MustCompile(`(?:window\.)?__(?:PRELOADED|INITIAL)_STATE__\s*=\s*`)},
		{"self.__next_f", regexp.MustCompile(`self\.__next_f\.push$\s*$$\s*1\s*,\s*"`)},
	}
	if m := regexp.MustCompile(`(?s)<script[^>]*id\s*=\s*"__NEXT_DATA__"[^>]*>`).FindStringIndex(html); m != nil {
		globals = append(globals, struct {
			Name string
			Re   *regexp.Regexp
		}{"__NEXT_DATA__(script-tag)", nil})
	}

	found := 0
	for _, g := range globals {
		if atomic.LoadInt64(&hydrationFindCount) >= maxHydrationFind {
			return
		}
		var raw string
		if g.Re == nil {
			m := regexp.MustCompile(`(?s)<script[^>]*id\s*=\s*"__NEXT_DATA__"[^>]*>(.*?)</script>`).FindStringSubmatch(html)
			if m != nil {
				raw = strings.TrimSpace(m[1])
			}
		} else {
			loc := g.Re.FindStringIndex(html)
			if loc == nil {
				continue
			}
			raw = extractBalanced(html, loc[1], '{', '}')
			if raw == "" {
				raw = extractBalanced(html, loc[1], '[', ']')
			}
		}
		if raw == "" {
			continue
		}
		found++
		fmt.Printf("  %s[+]%s Hydration state terdeteksi: %s (%d KB)\n",
			cGreen, cReset, g.Name, len(raw)/1024)

		var data interface{}
		if err := json.Unmarshal([]byte(raw), &data); err != nil {
			addFinding(Finding{Severity: "INFO", Type: "hydration-unparsed",
				Value: g.Name + " (JSON tidak utuh)", Asset: sourceURL})
			continue
		}

		var flat []HydrationTarget
		flattenJSON("", data, &flat, 0)

		var entCount, keyCount, piiCount int
		for _, item := range flat {
			if entCount+keyCount+piiCount >= maxHydrationFind {
				break
			}
			pathStr := g.Name + ":" + item.Path
			switch v := item.Value.(type) {
			case string:
				if key := hasSensitiveKey(item.Path); key != "" && keyCount < 40 {
					keyCount++
					addFinding(Finding{Severity: "HIGH", Type: "HYDRATION_SENSITIVE_KEY",
						Value: fmt.Sprintf("%s = %s", pathStr, truncate(v, 80)),
						Asset: sourceURL, Context: "key sensitif: " + key})
				} else if isHighEntropy(item.Path, v) && entCount < 40 {
					entCount++
					addFinding(Finding{Severity: "HIGH", Type: "HIGH_ENTROPY_HYDRATION_DATA",
						Value: fmt.Sprintf("%s = %s (H=%.2f)", pathStr, truncate(v, 60), shannonEntropy(v)),
						Asset: sourceURL, Context: "kemungkinan API key/token/hash internal"})
				}
				if rePIIEmail.MatchString(v) && piiCount < 20 {
					piiCount++
					addFinding(Finding{Severity: "MEDIUM", Type: "PII_LEAK_EMAIL",
						Value: pathStr + " = " + truncate(v, 80), Asset: sourceURL})
				}
				if rePIIPhone.MatchString(v) && piiCount < 20 {
					piiCount++
					addFinding(Finding{Severity: "MEDIUM", Type: "PII_LEAK_PHONE",
						Value: pathStr + " = " + truncate(v, 80), Asset: sourceURL})
				}
			case float64, bool:
				if key := hasSensitiveKey(item.Path); key != "" && keyCount < 40 {
					keyCount++
					addFinding(Finding{Severity: "HIGH", Type: "HYDRATION_SENSITIVE_KEY",
						Value: fmt.Sprintf("%s = %v", pathStr, v),
						Asset: sourceURL, Context: "key sensitif: " + key})
				}
			}
			atomic.AddInt64(&hydrationFindCount, 1)
		}
	}
	if found == 0 {
		fmt.Printf("  %s[i]%s Tidak ada hydration state global di HTML awal\n", cGray, cReset)
	}
}

// ============================================================
// [C] SERVER ACTIONS & [E] AST-LITE
// ============================================================

type actionIDRec struct {
	ID    string
	Asset string
}

var (
	actionIDs   = make(map[string]string)
	actionMutex sync.Mutex
)

func registerActionIDs(content, asset string) {
	if len(content) == 0 {
		return
	}
	hint := reRSCHint.MatchString(content)
	var ids []string
	for _, m := range reCreateServerRef.FindAllStringSubmatch(content, -1) {
		ids = append(ids, m[1])
	}
	for _, m := range reActionIDKey.FindAllStringSubmatch(content, -1) {
		ids = append(ids, m[1])
	}
	if hint {
		for _, m := range reActionHexInRSC.FindAllStringSubmatch(content, -1) {
			ids = append(ids, m[1])
		}
	}
	if len(ids) == 0 {
		return
	}
	actionMutex.Lock()
	added := 0
	for _, id := range ids {
		if _, ok := actionIDs[id]; !ok && len(actionIDs) < 500 {
			actionIDs[id] = asset
			added++
		}
	}
	actionMutex.Unlock()
	if added > 0 {
		fmt.Printf("  %s[+]%s Server Action ID ditemukan: +%d (total kandidat %d) — %s\n",
			cGreen, cReset, added, len(actionIDs), truncate(asset, 60))
	}
}

func probeRSCAndActions(accessibleRoutes []string) []ActionProbe {
	var results []ActionProbe
	base := strings.TrimRight(cfg.Target.String(), "/")

	rscProbeURL := base + "/"
	if len(accessibleRoutes) > 0 {
		rscProbeURL = base + accessibleRoutes[0]
	}
	rsc := doProbe("GET", rscProbeURL, "", map[string]string{
		"RSC":                  "1",
		"Next-Router-State-Tree": `%5B%22%22%2C%7B%7D%5D`,
	})
	if rsc.Status > 0 {
		ct := rsc.Headers.Get("Content-Type")
		if strings.Contains(ct, "text/x-component") || strings.HasPrefix(rsc.Body, "0:") ||
			strings.Contains(rsc.Headers.Get("X-Nextjs-Cache"), "HIT") ||
			rsc.Headers.Get("X-Nextjs-Matched-Path") != "" {
			report.RSCDetected = true
			addFinding(Finding{Severity: "HIGH", Type: "RSC-ENDPOINT",
				Value: rscProbeURL, Asset: rscProbeURL, HTTPStat: rsc.Status,
				Verified: fmt.Sprintf("Content-Type=%s; server merespons mode React Server Components", ct)})
		}
	}

	actionMutex.Lock()
	ids := make([]actionIDRec, 0, len(actionIDs))
	for id, asset := range actionIDs {
		ids = append(ids, actionIDRec{ID: id, Asset: asset})
	}
	actionMutex.Unlock()
	if len(ids) == 0 {
		return results
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i].ID < ids[j].ID })
	if len(ids) > cfg.MaxActions {
		ids = ids[:cfg.MaxActions]
	}
	fmt.Printf("  %s[*]%s Probing %d Server Action ID...\n", cCyan, cReset, len(ids))

	postURL := base + "/"
	if len(accessibleRoutes) > 0 {
		postURL = base + accessibleRoutes[0]
	}

	for _, rec := range ids {
		pr := doProbe("POST", postURL, "[]", map[string]string{
			"Next-Action":  rec.ID,
			"Content-Type": "text/plain;charset=UTF-8",
			"Accept":       "text/x-component",
		})
		verdict := classifyActionProbe(pr)
		results = append(results, ActionProbe{
			ActionID: rec.ID, ProbeURL: postURL,
			HTTPStatus: pr.Status, Verdict: verdict, FromAsset: rec.Asset,
		})

		col := cGray
		switch verdict {
		case "ACTION-CALLABLE":
			col = cRed
			addFinding(Finding{Severity: "CRITICAL", Type: "SERVER-ACTION-BYPASS",
				Value: rec.ID, Asset: rec.Asset, HTTPStat: pr.Status,
				Verified: "Server Action dapat dipanggil langsung via POST + Next-Action tanpa melewati UI"})
		case "ACTION-EXISTS-ERROR":
			col = cYellow
			addFinding(Finding{Severity: "HIGH", Type: "SERVER-ACTION-EXPOSED",
				Value: rec.ID, Asset: rec.Asset, HTTPStat: pr.Status,
				Verified: "Action ID valid (server merespons x-action-error/500) — fungsi terekspos"})
		case "ACTION-PROTECTED":
			col = cCyan
		}
		fmt.Printf("  %s[%s]%s action %s → %s\n", col, verdict, cReset, truncate(rec.ID, 40), truncate(postURL, 50))
	}
	return results
}

func classifyActionProbe(pr probeResult) string {
	switch {
	case pr.Status == 0:
		return "NO-RESPONSE"
	case pr.Status == 404:
		if strings.Contains(pr.Body, "Failed to find Server Action") {
			return "ACTION-NOT-FOUND"
		}
		return "NOT-FOUND"
	case pr.Status == 401 || pr.Status == 403:
		return "ACTION-PROTECTED"
	case pr.Status == 500 || pr.Headers.Get("x-action-error") != "":
		return "ACTION-EXISTS-ERROR"
	case pr.Status >= 200 && pr.Status < 300:
		ct := pr.Headers.Get("Content-Type")
		if strings.Contains(ct, "text/x-component") || strings.Contains(ct, "application/json") ||
			strings.HasPrefix(pr.Body, "1:") || strings.HasPrefix(pr.Body, "0:") {
			return "ACTION-CALLABLE"
		}
		return "ACTION-NOOP-FALLBACK"
	case pr.Status >= 300 && pr.Status < 400:
		return "REDIRECT"
	default:
		return fmt.Sprintf("HTTP-%d", pr.Status)
	}
}

var (
	reconTree   = make(map[string]string)
	reconMutex  sync.Mutex
	reconLoaded int
)

func isSensitiveSourceFile(p string) bool {
	low := strings.ToLower(p)
	markers := []string{
		".env", "config.ts", "config.js", "constants", "middleware", "auth",
		"swagger", "openapi", "graphql", "resolver", "secret", "admin",
		"internal", "permission", "rbac", "acl", "payment", "billing",
	}
	for _, m := range markers {
		if strings.Contains(low, m) {
			return true
		}
	}
	return false
}

func astLiteScan(content, srcPath, mapURL string) {
	asset := mapURL + " → " + srcPath

	if fields := reZodYupSchema.FindAllStringSubmatch(content, 30); len(fields) > 0 {
		names := make([]string, 0, len(fields))
		for _, f := range fields {
			names = append(names, f[1])
		}
		addFinding(Finding{Severity: "MEDIUM", Type: "VALIDATION_SCHEMA",
			Value: srcPath + " → fields: " + strings.Join(names, ","),
			Asset: asset, Context: "skema zod/yup/joi — bocoran kontrak data server"})
	}
	for _, m := range reHardcodedRole.FindAllString(content, 10) {
		addFinding(Finding{Severity: "HIGH", Type: "HARDCODED_ROLE_CHECK",
			Value: truncate(m, 80), Asset: asset,
			Context: "otorisasi berbasis string role di client — uji bypass di backend"})
	}
	for _, m := range reClientTokenSt.FindAllString(content, 5) {
		addFinding(Finding{Severity: "MEDIUM", Type: "CLIENT_TOKEN_STORAGE",
			Value: truncate(m, 90), Asset: asset})
	}
	envSeen := map[string]bool{}
	for _, m := range reProcessEnv.FindAllStringSubmatch(content, 50) {
		name := m[1]
		if envSeen[name] || strings.HasPrefix(name, "NEXT_PUBLIC_") ||
			name == "NODE_ENV" || name == "PORT" {
			continue
		}
		envSeen[name] = true
		addFinding(Finding{Severity: "HIGH", Type: "SERVER_ENV_REFERENCE",
			Value: "process.env." + name, Asset: asset,
			Context: "variabel env non-publik direferensikan di kode client"})
	}
	for _, m := range reDangerousHTML.FindAllString(content, 5) {
		addFinding(Finding{Severity: "LOW", Type: "DANGEROUS_SINK",
			Value: truncate(m, 60), Asset: asset})
	}
	if reGraphQLResolv.MatchString(content) {
		addFinding(Finding{Severity: "HIGH", Type: "GRAPHQL_RESOLVER_SOURCE",
			Value: srcPath, Asset: asset,
			Context: "resolver GraphQL terekspos via sourcemap"})
	}
	if reSwaggerHint.MatchString(content) {
		addFinding(Finding{Severity: "HIGH", Type: "API_SCHEMA_SOURCE",
			Value: srcPath, Asset: asset,
			Context: "skema swagger/openapi tertinggal di source map"})
	}
}

// ============================================================
//  STATIC ASSET FETCHER + SOURCE MAP UNPACKER + [F] MODERN BUNDLER
// ============================================================

var (
	seenAssets = make(map[string]bool)
	assetMutex sync.Mutex
)

var (
	reScriptSrc   = regexp.MustCompile(`(?i)<script[^>]+src\s*=\s*["']([^"']+\.js[^"']*)["']`)
	reLinkHref    = regexp.MustCompile(`(?i)<link[^>]+href\s*=\s*["']([^"']+\.js[^"']*)["']`)
	reSourceMap   = regexp.MustCompile(`//[#@]\s*sourceMappingURL\s*=\s*(\S+)`)
	reBlockSourceMap = regexp.MustCompile(`/\*#\s*sourceMappingURL\s*=\s*(\S+)\s*\*/`)
	reJSString    = regexp.MustCompile(`["'` + "`" + `](\/[A-Za-z0-9_\-\/\.]+?\.js(?:\?[A-Za-z0-9_=&\-\.]*)?)["'`+"`"+`]`)
	reNextChunk   = regexp.MustCompile(`["'` + "`" + `](\/_next\/static\/[^"'`+"`"+`]+?\.js)["'`+"`"+`]`)
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

// [F] Ekstraksi aset dari berbagai bundler modern
func extractAssetRefs(content, baseURL string) []string {
	var refs []string
	
	// Tradisional (menggunakan Submatch karena ada grup tangkapan)
	for _, re := range []*regexp.Regexp{reScriptSrc, reLinkHref, reJSString, reNextChunk} {
		for _, mm := range re.FindAllStringSubmatch(content, -1) {
			if len(mm) > 1 {
				u := absURL(mm[1])
				if u != "" && sameHost(u) {
					refs = append(refs, u)
				}
			}
		}
	}

	// [F] Vite / Module Preload (ada grup tangkapan)
	for _, mm := range reModulePreload.FindAllStringSubmatch(content, -1) {
		if len(mm) > 1 {
			u := absURL(mm[1])
			if u != "" && sameHost(u) {
				refs = append(refs, u)
			}
		}
	}

	// [F] Dynamic Import (Vite/Webpack) (ada grup tangkapan)
	for _, mm := range reDynamicImport.FindAllStringSubmatch(content, -1) {
		if len(mm) > 1 {
			u := absURL(mm[1])
			if u != "" && sameHost(u) {
				refs = append(refs, u)
			}
		}
	}

	// [F] Next.js App Router Chunks (TIDAK ada grup tangkapan, gunakan FindAllString)
	for _, match := range reNextAppChunk.FindAllString(content, -1) {
		u := absURL(match)
		if u != "" && sameHost(u) {
			refs = append(refs, u)
		}
	}
	for _, match := range reNextMainApp.FindAllString(content, -1) {
		u := absURL(match)
		if u != "" && sameHost(u) {
			refs = append(refs, u)
		}
	}

	// [F] Nuxt 3 Assets (TIDAK ada grup tangkapan, gunakan FindAllString)
	for _, match := range reNuxtAsset.FindAllString(content, -1) {
		u := absURL(match)
		if u != "" && sameHost(u) {
			refs = append(refs, u)
		}
	}

	// [F] SvelteKit Assets (TIDAK ada grup tangkapan, gunakan FindAllString)
	for _, match := range reSvelteAsset.FindAllString(content, -1) {
		u := absURL(match)
		if u != "" && sameHost(u) {
			refs = append(refs, u)
		}
	}

	return refs
}

// [F] Ekstraksi Webpack Chunk Map: { id: "hash" }
func extractWebpackChunks(content string) []string {
	var chunks []string
	maps := reWebpackChunkMap.FindAllStringSubmatch(content, -1)
	if len(maps) == 0 {
		return chunks
	}
	
	// Cari pola require.e(id) untuk konfirmasi ID yang aktif
	activeIDs := make(map[string]bool)
	for _, m := range reWebpackRequireE.FindAllStringSubmatch(content, -1) {
		if len(m) > 1 {
			activeIDs[m[1]] = true
		}
	}

	// Rekonstruksi URL berdasarkan pola umum Next.js/Webpack
	// Pola: /_next/static/chunks/[id]-[hash].js ATAU chunk-[id].[hash].js
	for _, m := range maps {
		if len(m) < 3 {
			continue
		}
		id := m[1]
		hash := m[2]
		
		// Jika ID ini dipanggil via require.e, kemungkinan besar valid
		if activeIDs[id] || len(maps) < 20 { // Heuristik: jika sedikit map, ambil semua
			// Coba pola Next.js
			chunks = append(chunks, fmt.Sprintf("/_next/static/chunks/%s-%s.js", id, hash))
			// Coba pola Webpack standar
			chunks = append(chunks, fmt.Sprintf("/static/js/chunk-%s.%s.js", id, hash))
		}
	}
	return chunks
}

func handleSourceMapURL(smRef, jsURL, content string) {
	if strings.HasPrefix(smRef, "data:") {
		parts := strings.SplitN(smRef, ",", 2)
		if len(parts) == 2 {
			var decoded []byte
			var err error
			if strings.Contains(parts[0], "base64") {
				decoded, err = base64.StdEncoding.DecodeString(parts[1])
			} else {
				decoded, err = io.ReadAll(strings.NewReader(parts[1]))
			}
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
	sensitiveCount := 0
	for i, src := range sm.Sources {
		var content string
		if i < len(sm.SourcesContent) {
			content = sm.SourcesContent[i]
		}
		if content == "" {
			continue
		}

		if isSensitiveSourceFile(src) {
			reconMutex.Lock()
			if reconLoaded < 200 {
				reconTree[src] = content
				reconLoaded++
				sensitiveCount++
			}
			reconMutex.Unlock()
			astLiteScan(content, src, mapURL)
		}

		analyzeContent(content, mapURL+" → "+src)
		registerActionIDs(content, mapURL+" → "+src)

		if dumpDir != "" {
			fp := path.Join(dumpDir, sanitizeName(src))
			if dir := path.Dir(fp); dir != "" {
				os.MkdirAll(dir, 0o755)
			}
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
	if sensitiveCount > 0 {
		fmt.Printf("  %s[+]%s [AST-lite] %d file sensitif diprioritaskan dari sourcemap ini\n",
			cGreen, cReset, sensitiveCount)
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

// [F] scanAsset dengan Deep Chain Discovery (Depth 3)
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
	registerActionIDs(body, jsURL)

	// [F] Deteksi SourceMap via Header HTTP
	// (Simulasi: karena fetch() mengembalikan string, kita asumsikan header sudah dicek di level rendah jika perlu,
	// tapi di sini kita cek komentar blok juga)
	if mm := reSourceMap.FindStringSubmatch(body); mm != nil {
		handleSourceMapURL(mm[1], jsURL, body)
	} else if mm := reBlockSourceMap.FindStringSubmatch(body); mm != nil {
		handleSourceMapURL(mm[1], jsURL, body)
	} else {
		handleSourceMapURL(jsURL+".map", jsURL, "")
	}

	// [F] Deep Chain Discovery
	if depth < cfg.MaxDepth {
		refs := extractAssetRefs(body, jsURL)
		
		// [F] Tambahkan Webpack Chunks jika terdeteksi
		if chunks := extractWebpackChunks(body); len(chunks) > 0 {
			for _, c := range chunks {
				refs = append(refs, absURL(c))
			}
			fmt.Printf("  %s[+]%s Webpack Chunk Graph: %d chunk teridentifikasi\n", cGreen, cReset, len(chunks))
		}

		for _, ref := range refs {
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
// [B] DIFFERENTIAL PROBING
// ============================================================

type baselineFP struct {
	OK     bool
	Status int
	Length int64
	Tokens map[string]bool
}

func htmlTokenSet(body string) map[string]bool {
	matches := reHTMLTag.FindAllStringSubmatch(body, 4096)
	set := make(map[string]bool, len(matches))
	for _, m := range matches {
		set[strings.ToLower(m[1])] = true
	}
	return set
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	inter := 0
	small, large := a, b
	if len(b) < len(a) {
		small, large = b, a
	}
	for k := range small {
		if large[k] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 1.0
	}
	return float64(inter) / float64(union)
}

func captureBaseline() baselineFP {
	nonce := fmt.Sprintf("/devroute-nonexistent-%d-%d", time.Now().UnixNano(), os.Getpid()%9973)
	full := strings.TrimRight(cfg.Target.String(), "/") + nonce
	fmt.Printf("  %s[*]%s Baseline fingerprint: %s\n", cCyan, cReset, nonce)
	pr := doProbe("GET", full, "", nil)
	if pr.Status == 0 {
		return baselineFP{OK: false}
	}
	return baselineFP{
		OK:     true,
		Status: pr.Status,
		Length: pr.Length,
		Tokens: htmlTokenSet(pr.Body),
	}
}

func classifyProbeDiff(pr probeResult, base baselineFP) (string, float64) {
	sim := 0.0
	switch {
	case pr.Status == 0:
		return "NO-RESPONSE", 0
	case pr.Status >= 300 && pr.Status < 400:
		loc := pr.Headers.Get("Location")
		low := strings.ToLower(loc)
		if strings.Contains(low, "login") || strings.Contains(low, "auth") ||
			strings.Contains(low, "signin") || strings.Contains(low, "sso") {
			return "REDIRECT-AUTH", 0
		}
		return "REDIRECT", 0
	case pr.Status == 401 || pr.Status == 403:
		return "PROTECTED", 0
	case pr.Status == 404:
		return "NOT-FOUND-SERVER", 0
	case pr.Status >= 500:
		return fmt.Sprintf("HTTP-%d", pr.Status), 0
	case pr.Status == 200:
		ct := strings.ToLower(pr.Headers.Get("Content-Type"))
		if strings.Contains(ct, "application/json") || strings.Contains(ct, "text/x-component") ||
			strings.Contains(ct, "application/xml") || strings.Contains(ct, "text/plain") {
			return "ACCESSIBLE-DATA", 0
		}
		if pr.Headers.Get("X-Nextjs-Matched-Path") != "" || pr.Headers.Get("X-Nextjs-Cache") != "" {
			return "ACCESSIBLE-NO-AUTH", 0
		}
		if base.OK {
			sim = jaccard(htmlTokenSet(pr.Body), base.Tokens)
			lenClose := base.Length > 0 && absDiff(pr.Length, base.Length)*100/max64(base.Length, 1) < 10
			if sim >= 0.95 && lenClose && base.Status == 200 {
				return "SPA-FALLBACK-VIRTUAL-404", sim
			}
			if sim >= 0.95 && base.Status == 404 {
				return "SPA-FALLBACK-VIRTUAL-404", sim
			}
			return "ACCESSIBLE-NO-AUTH", sim
		}
		return "ACCESSIBLE-NO-AUTH(cek manual)", 0
	default:
		return fmt.Sprintf("HTTP-%d", pr.Status), 0
	}
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

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

	base := captureBaseline()
	if base.OK {
		fmt.Printf("  %s[+]%s Baseline: HTTP %d, %d byte, %d token DOM unik\n",
			cGreen, cReset, base.Status, base.Length, len(base.Tokens))
	} else {
		fmt.Printf("  %s[!]%s Baseline gagal — fallback ke heuristik panjang klasik\n", cYellow, cReset)
	}
	baseBody, _, _ := fetch(cfg.Target.String())
	baseLen := int64(len(baseBody))

	var (
		probes []RouteProbe
		mu     sync.Mutex
		sem    = make(chan struct{}, cfg.Conc)
		wg     sync.WaitGroup
	)

	for _, r := range routes {
		wg.Add(1)
		sem <- struct{}{}
		go func(route string) {
			defer wg.Done()
			defer func() { <-sem }()
			defer func() {
				if rec := recover(); rec != nil {
					fmt.Printf("  %s[!]%s panic saat probe %s: %v\n", cYellow, cReset, route, rec)
				}
			}()
			full := strings.TrimRight(cfg.Target.String(), "/") + route
			pr := doProbe("GET", full, "", nil)

			var verdict string
			var sim float64
			if base.OK {
				verdict, sim = classifyProbeDiff(pr, base)
			} else {
				verdict = classifyProbe(pr.Status, pr.Length, pr.Headers.Get("Location"), baseLen)
			}

			p := RouteProbe{Route: route, FullPath: full, HTTPStatus: pr.Status,
				ContentLen: pr.Length, RedirectTo: pr.Headers.Get("Location"),
				Similarity: math.Round(sim*1000) / 1000, Verdict: verdict, FromSource: routeSet[route]}
			mu.Lock()
			probes = append(probes, p)
			mu.Unlock()

			col := cGray
			switch verdict {
			case "ACCESSIBLE-NO-AUTH":
				col = cRed
			case "ACCESSIBLE-DATA":
				col = cYellow
			case "REDIRECT-AUTH":
				col = cYellow
			case "SPA-FALLBACK-VIRTUAL-404":
				col = cCyan
			}
			simStr := ""
			if sim > 0 {
				simStr = fmt.Sprintf(" (sim=%.0f%%)", sim*100)
			}
			fmt.Printf("  %s%-3d%s %-46s → %s%s\n", col, pr.Status, cReset, truncate(route, 46), verdict, simStr)
		}(r)
	}
	wg.Wait()

	for _, p := range probes {
		if p.Verdict == "ACCESSIBLE-NO-AUTH" {
			addFinding(Finding{Severity: "CRITICAL", Type: "unprotected-route",
				Value: p.Route, Asset: p.FromSource, HTTPStat: p.HTTPStatus,
				Verified: fmt.Sprintf("200 OK, deviasi signifikan dari baseline 404-virtual (sim=%.0f%%)", p.Similarity*100)})
		}
		if p.Verdict == "ACCESSIBLE-DATA" {
			addFinding(Finding{Severity: "HIGH", Type: "exposed-data-endpoint",
				Value: p.Route, Asset: p.FromSource, HTTPStat: p.HTTPStatus,
				Verified: "200 OK mengembalikan payload JSON/data murni tanpa autentikasi"})
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
// [D] AUTH MATRIX
// ============================================================

func buildAuthMatrix() []AuthMatrixEntry {
	findMutex.Lock()
	snapshot := append([]Finding(nil), findings...)
	findMutex.Unlock()

	byAsset := make(map[string]*AuthMatrixEntry)
	get := func(asset string) *AuthMatrixEntry {
		if e, ok := byAsset[asset]; ok {
			return e
		}
		e := &AuthMatrixEntry{Asset: asset}
		byAsset[asset] = e
		return e
	}
	uniq := func(list *[]string, v string) {
		for _, x := range *list {
			if x == v {
				return
			}
		}
		if len(*list) < 100 {
			*list = append(*list, v)
		}
	}

	for _, f := range snapshot {
		switch f.Type {
		case "route":
			uniq(&get(f.Asset).Routes, f.Value)
		case "permission", "state":
			uniq(&get(f.Asset).ClientChecks, f.Value)
		case "guard", "auth":
			uniq(&get(f.Asset).Guards, f.Value)
		case "api":
			uniq(&get(f.Asset).APIs, f.Value)
		}
	}

	var out []AuthMatrixEntry
	for _, e := range byAsset {
		if len(e.ClientChecks)+len(e.Guards) > 0 || (len(e.Routes) > 0 && len(e.APIs) > 0) {
			out = append(out, *e)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		score := func(e AuthMatrixEntry) int {
			return len(e.ClientChecks)*3 + len(e.Guards)*2 + len(e.APIs) + len(e.Routes)
		}
		return score(out[i]) > score(out[j])
	})
	if len(out) > 50 {
		out = out[:50]
	}

	if len(out) > 0 {
		fmt.Printf("\n%s[*]%s Authorization Matrix: %d aset memetakan rute→cek-role→API\n", cCyan, cReset, len(out))
		for i, e := range out {
			if i >= 8 {
				fmt.Printf("  %s…%d entri lainnya di laporan JSON%s\n", cGray, len(out)-8, cReset)
				break
			}
			fmt.Printf("  %s▸%s %s\n", cBold, cReset, truncate(e.Asset, 100))
			if len(e.Routes) > 0 {
				fmt.Printf("      routes : %s\n", truncate(strings.Join(e.Routes, ", "), 90))
			}
			if len(e.Guards) > 0 {
				fmt.Printf("      guards : %s\n", truncate(strings.Join(e.Guards, ", "), 90))
			}
			if len(e.ClientChecks) > 0 {
				fmt.Printf("      checks : %s\n", truncate(strings.Join(e.ClientChecks, ", "), 90))
			}
			if len(e.APIs) > 0 {
				fmt.Printf("      apis   : %s\n", truncate(strings.Join(e.APIs, ", "), 90))
			}
		}
		addFinding(Finding{Severity: "MEDIUM", Type: "AUTHZ_MATRIX_BUILT",
			Value: fmt.Sprintf("%d aset dengan pemetaan otorisasi client-side", len(out)),
			Asset: cfg.Target.String()})
	}
	return out
}

// ============================================================
//  LAPORAN
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
		accessible, fallback := 0, 0
		for _, p := range report.Routes {
			if p.Verdict == "ACCESSIBLE-NO-AUTH" || p.Verdict == "ACCESSIBLE-DATA" {
				accessible++
			}
			if p.Verdict == "SPA-FALLBACK-VIRTUAL-404" {
				fallback++
			}
		}
		fmt.Printf(" Rute terverifikasi tanpa auth : %s%d%s\n", cRed, accessible, cReset)
		fmt.Printf(" SPA fallback tersaring (anti false-positive) : %d\n", fallback)
	}
	if len(report.ServerActions) > 0 {
		callable := 0
		for _, a := range report.ServerActions {
			if a.Verdict == "ACTION-CALLABLE" || a.Verdict == "ACTION-EXISTS-ERROR" {
				callable++
			}
		}
		fmt.Printf(" Server Action di-probe : %d (terekspos/callable: %s%d%s)\n",
			len(report.ServerActions), cRed, callable, cReset)
	}
	if report.RSCDetected {
		fmt.Printf(" RSC endpoint (text/x-component) : %sTERDETEKSI%s\n", cYellow, cReset)
	}
	if len(report.AuthMatrix) > 0 {
		fmt.Printf(" Entri Authorization Matrix : %d\n", len(report.AuthMatrix))
	}
	fmt.Printf("%s═══════════════════════════════════════════%s\n\n", cBold, cReset)

	if cfg.JSONOut != "" {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: gagal marshal laporan JSON: %v\n", err)
			return
		}
		if err := os.WriteFile(cfg.JSONOut, b, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: gagal tulis %s: %v\n", cfg.JSONOut, err)
			return
		}
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
		fTarget     = flag.String("u", "", "URL target (wajib)")
		fHeaders    headerList
		fConc       = flag.Int("c", 10, "jumlah worker konkuren")
		fTimeout    = flag.Int("timeout", 15, "timeout HTTP per-request (detik)")
		fInsec      = flag.Bool("insecure", false, "abaikan sertifikat TLS invalid")
		fProxy      = flag.String("proxy", "", "proxy HTTP")
		fNoVerify   = flag.Bool("no-verify", false, "lewati verifikasi rute dinamis & action probe")
		fJSON       = flag.String("json", "devroute-report.json", "path output JSON")
		fDump       = flag.String("dump", "", "direktori rekonstruksi source dari sourcemap")
		fMaxAsset   = flag.Int("max-assets", 150, "batas maksimum aset JS")
		fMaxActions = flag.Int("max-actions", 15, "batas maksimum Server Action ID")
		fEntropy    = flag.Float64("entropy", 4.2, "ambang entropi Shannon")
		fMaxDepth   = flag.Int("depth", 3, "[F] kedalaman chain discovery (default 3)")
	)
	flag.Var(&fHeaders, "H", `header tambahan "Key: Value"`)
	flag.Parse()

	fmt.Print(cCyan + banner + cReset)

	if *fTarget == "" {
		fmt.Fprintln(os.Stderr, "ERROR: -u wajib diisi.")
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
		MaxBody: 25 * 1024 * 1024, MaxActions: *fMaxActions, EntropyMin: *fEntropy,
		MaxDepth: *fMaxDepth,
	}
	if cfg.Conc < 1 {
		cfg.Conc = 1
	}
	if cfg.MaxActions < 1 {
		cfg.MaxActions = 15
	}
	if cfg.EntropyMin < 3.0 || cfg.EntropyMin > 6.0 {
		cfg.EntropyMin = 4.2
	}
	client = buildClient()

	start := time.Now()
	report.Target = cfg.Target.String()

	// --- FASE 1: Fetch + Hydration ---
	fmt.Printf("%s[FASE 1]%s Mengunduh halaman utama & inventarisasi aset JS...\n", cBold, cReset)
	html, code, err := fetch(cfg.Target.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: gagal fetch target: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  %s[+]%s HTTP %d, %d KB\n", cGreen, cReset, code, len(html)/1024)

	analyzeContent(html, cfg.Target.String())

	fmt.Printf("\n%s[FASE 1-A]%s Hydration State Diffing...\n", cBold, cReset)
	analyzeHydrationState(html, cfg.Target.String())

	// --- Seeding Assets ---
	assets := extractAssetRefs(html, cfg.Target.String())
	
	// [F] Autodiscovery Manifest Fallbacks
	manifestSeeds := []string{
		"/asset-manifest.json", "/mix-manifest.json", 
		"/chunks-manifest.json", "/build/manifest.json",
		"/_nuxt/builds/meta/latest.json", "/_app/version.json",
	}
	if buildID := extractBuildID(html); buildID != "" {
		assets = append(assets,
			absURL("/_next/static/"+buildID+"/_buildManifest.js"),
			absURL("/_next/static/"+buildID+"/_ssgManifest.js"),
		)
		fmt.Printf("  %s[+]%s Next.js buildId terdeteksi: %s\n", cGreen, cReset, buildID)
	}
	for _, ms := range manifestSeeds {
		assets = append(assets, absURL(ms))
	}

	// --- FASE 2: Scan + Deep Chain ---
	fmt.Printf("\n%s[FASE 2]%s Ekstraksi rute, state, secret, action-ID (Depth %d)...\n",
		cBold, cReset, cfg.MaxDepth)
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
			defer func() {
				if rec := recover(); rec != nil {
					fmt.Printf("  %s[!]%s panic saat scan %s: %v\n", cYellow, cReset, truncate(u, 60), rec)
				}
			}()
			scanAsset(u, 0)
		}(a)
	}
	wg.Wait()

	// --- FASE 3: Verify Routes ---
	var accessibleRoutes []string
	if cfg.Verify {
		fmt.Printf("\n%s[FASE 3]%s Dynamic Route Verification...\n", cBold, cReset)
		report.Routes = verifyRoutes()
		for _, p := range report.Routes {
			if p.Verdict == "ACCESSIBLE-NO-AUTH" || p.Verdict == "ACCESSIBLE-DATA" {
				accessibleRoutes = append(accessibleRoutes, p.Route)
			}
		}
	}

	// --- FASE 3.5: RSC & Actions ---
	if cfg.Verify {
		fmt.Printf("\n%s[FASE 3.5]%s RSC Endpoint & Server Actions Probing...\n", cBold, cReset)
		report.ServerActions = probeRSCAndActions(accessibleRoutes)
		if len(report.ServerActions) == 0 && !report.RSCDetected {
			fmt.Printf("  %s[i]%s Tidak ada Server Action ID / RSC endpoint terdeteksi.\n", cGray, cReset)
		}
	}

	// --- FASE 4: Auth Matrix + Laporan ---
	fmt.Printf("\n%s[FASE 4]%s Authorization Mapping & Laporan...\n", cBold, cReset)
	report.AuthMatrix = buildAuthMatrix()

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

type headerList []string

func (h *headerList) String() string     { return strings.Join(*h, ", ") }
func (h *headerList) Set(v string) error { *h = append(*h, v); return nil }

