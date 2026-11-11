"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs = __importStar(require("fs"));
const acorn = __importStar(require("acorn"));
const puppeteer_1 = __importDefault(require("puppeteer"));
// --- LAYER 1: CONFIGURATION MANAGER ---
class ConfigManager {
    config;
    constructor(configPath = './config.json') {
        try {
            const raw = fs.readFileSync(configPath, 'utf-8');
            this.config = JSON.parse(raw);
        }
        catch (e) {
            this.config = {
                targetUrl: process.argv[2] || 'https://react-redux.realworld.io/',
                timeoutMs: 45000 // Increased timeout for SPA loading
            };
        }
    }
    getTarget() { return this.config.targetUrl; }
    getTimeout() { return this.config.timeoutMs || 45000; }
}
// --- LAYER 2: BROWSER ORCHESTRATION ---
class StealthCDPClient {
    browser = null;
    page = null;
    async launch() {
        console.log('[StealthCDP] Launching Chromium...');
        try {
            this.browser = await puppeteer_1.default.launch({
                headless: true,
                args: [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-background-timer-throttling' // Ensure background tabs don't sleep
                ]
            });
            this.page = await this.browser.newPage();
            console.log('[StealthCDP] Browser ready.');
        }
        catch (err) {
            console.error('[StealthCDP] Launch failed:', err.message);
            throw err;
        }
    }
    getPage() { return this.page; }
    async close() { if (this.browser)
        await this.browser.close(); }
}
// --- LAYER 3: EVASION & INTERCEPTION ---
class NetworkInterceptor {
    page;
    jsContents = new Map();
    htmlContent = '';
    constructor(page) {
        this.page = page;
    }
    async enableInterception() {
        console.log('[NetworkInterceptor] Monitoring network traffic...');
        this.page.on('response', async (response) => {
            const url = response.url();
            const contentType = response.headers()['content-type'] || '';
            if (url.endsWith('.js') || contentType.includes('javascript')) {
                try {
                    const buffer = await response.buffer();
                    const body = buffer.toString('utf-8');
                    // Only store if it's not a tiny tracking script
                    if (body.length > 500) {
                        this.jsContents.set(url, body);
                        console.log(`[Net] Captured JS (${body.length} bytes): ${url.substring(0, 50)}...`);
                    }
                }
                catch (e) { /* Ignore */ }
            }
            if (url === this.page.url() && contentType.includes('html')) {
                try {
                    this.htmlContent = await response.text();
                }
                catch (e) { /* Ignore */ }
            }
        });
    }
    getCapturedScripts() { return this.jsContents; }
    getHtmlContent() { return this.htmlContent; }
}
// --- LAYER 4: AST ANALYZER ---
class ASTAnalyzer {
    findings = [];
    // More aggressive patterns for minified code
    secretPatterns = [
        { regex: /(?:api[_-]?key|apikey|token|secret|password)\s*[:=]\s*['"]([a-zA-Z0-9_\-]{8,})['"]/i, type: 'SECRET', severity: 'HIGH' },
        { regex: /AKIA[0-9A-Z]{16}/, type: 'SECRET', severity: 'CRITICAL' },
        { regex: /ghp_[a-zA-Z0-9]{36}/, type: 'SECRET', severity: 'CRITICAL' },
        { regex: /eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*/, type: 'SECRET', severity: 'HIGH' } // JWT
    ];
    endpointPatterns = [
        /['"`](\/api\/[a-zA-Z0-9\/\._-]+)['"`]/g,
        /['"`](\/v\d+\/[a-zA-Z0-9\/\._-]+)['"`]/g,
        /axios\.(get|post|put|delete)$['"`]([^'"]+)['"`]/g,
        /fetch$['"`]([^'"]+)['"`]/g
    ];
    analyze(sourceCode, sourceUrl) {
        if (!sourceCode)
            return;
        // 1. Regex Scan
        this.secretPatterns.forEach(pattern => {
            let match;
            while ((match = pattern.regex.exec(sourceCode)) !== null) {
                this.addFinding({ type: 'SECRET', value: match[1] || match[0], sourceFile: sourceUrl, severity: pattern.severity });
            }
        });
        this.endpointPatterns.forEach(pattern => {
            let match;
            while ((match = pattern.exec(sourceCode)) !== null) {
                const endpoint = match[2] || match[1];
                if (endpoint && (endpoint.startsWith('/api') || endpoint.startsWith('/v'))) {
                    this.addFinding({ type: 'ENDPOINT', value: endpoint, sourceFile: sourceUrl, severity: 'LOW' });
                }
            }
        });
        // 2. AST Scan (Basic)
        try {
            const ast = acorn.parse(sourceCode, { ecmaVersion: 'latest', sourceType: 'module', allowReturnOutsideFunction: true });
            // In a full version, we would traverse the AST to find variable assignments
        }
        catch (e) { /* Silent fail for minified code errors */ }
    }
    addFinding(item) {
        if (!this.findings.some(f => f.value === item.value && f.type === item.type)) {
            this.findings.push(item);
        }
    }
    getFindings() { return this.findings; }
}
// --- LAYER 5: REPORT AGGREGATOR ---
class ReportAggregator {
    findings = [];
    addFindings(items) { this.findings.push(...items); }
    generateReport(target) {
        const report = {
            target,
            timestamp: new Date().toISOString(),
            summary: {
                totalEndpoints: this.findings.filter(f => f.type === 'ENDPOINT').length,
                totalSecrets: this.findings.filter(f => f.type === 'SECRET').length,
                totalParameters: 0
            },
            findings: this.findings
        };
        const fileName = `report_${new URL(target).hostname}_${Date.now()}.json`;
        fs.writeFileSync(fileName, JSON.stringify(report, null, 2));
        console.log(`[Report] Saved to ${fileName}`);
        return fileName;
    }
}
// --- MAIN ENTRY POINT ---
async function main() {
    console.log('=========================================');
    console.log('   SPECTER MAP: SPA Surface Scanner');
    console.log('=========================================');
    const configMgr = new ConfigManager();
    const target = configMgr.getTarget();
    const aggregator = new ReportAggregator();
    const analyzer = new ASTAnalyzer();
    let cdpClient = null;
    try {
        cdpClient = new StealthCDPClient();
        await cdpClient.launch();
        const page = cdpClient.getPage();
        if (!page)
            throw new Error("Page is null");
        const interceptor = new NetworkInterceptor(page);
        await interceptor.enableInterception();
        console.log(`[Main] Navigating to ${target}...`);
        // Wait for networkidle2 to ensure most chunks are loaded
        await page.goto(target, { waitUntil: 'networkidle2', timeout: configMgr.getTimeout() });
        // Extra wait for dynamic imports
        console.log('[Main] Waiting for dynamic imports...');
        await new Promise(r => setTimeout(r, 5000));
        const scripts = interceptor.getCapturedScripts();
        console.log(`[Main] Total JS Chunks Captured: ${scripts.size}`);
        if (scripts.size === 0) {
            console.log('[Main] No external JS found. Falling back to HTML analysis...');
            const html = interceptor.getHtmlContent();
            if (html)
                analyzer.analyze(html, target + '/index.html');
        }
        else {
            scripts.forEach((code, url) => analyzer.analyze(code, url));
        }
        aggregator.addFindings(analyzer.getFindings());
        aggregator.generateReport(target);
        console.log('\n--- SCAN COMPLETE ---');
        console.log(`Total Findings: ${analyzer.getFindings().length}`);
        analyzer.getFindings().forEach(f => console.log(` - [${f.type}] ${f.value}`));
    }
    catch (error) {
        console.error('[Main] Error:', error.message);
    }
    finally {
        if (cdpClient)
            await cdpClient.close();
        process.exit(0);
    }
}
main();
