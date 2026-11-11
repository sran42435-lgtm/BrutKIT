import * as fs from 'fs';
import * as path from 'path';
import * as acorn from 'acorn';
import puppeteer, { Browser, Page } from 'puppeteer';

// --- INTERFACES & TYPES ---

interface Config {
  targetUrl: string;
  timeoutMs?: number;
}

interface FoundItem {
  type: 'ENDPOINT' | 'SECRET' | 'PARAMETER';
  value: string;
  sourceFile?: string;
  severity?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

interface ReportData {
  target: string;
  timestamp: string;
  summary: {
    totalEndpoints: number;
    totalSecrets: number;
    totalParameters: number;
  };
  findings: FoundItem[];
}

// --- LAYER 1: CONFIGURATION MANAGER ---

class ConfigManager {
  private config: Config;

  constructor(configPath: string = './config.json') {
    try {
      const raw = fs.readFileSync(configPath, 'utf-8');
      this.config = JSON.parse(raw);
    } catch (e: any) {
      this.config = {
        targetUrl: process.argv[2] || 'https://react-redux.realworld.io/',
        timeoutMs: 45000 // Increased timeout for SPA loading
      };
    }
  }

  getTarget(): string { return this.config.targetUrl; }
  getTimeout(): number { return this.config.timeoutMs || 45000; }
}

// --- LAYER 2: BROWSER ORCHESTRATION ---

class StealthCDPClient {
  private browser: Browser | null = null;
  private page: Page | null = null;

  async launch(): Promise<void> {
    console.log('[StealthCDP] Launching Chromium...');
    try {
      this.browser = await puppeteer.launch({
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
    } catch (err: any) {
      console.error('[StealthCDP] Launch failed:', err.message);
      throw err;
    }
  }

  getPage(): Page | null { return this.page; }
  async close(): Promise<void> { if (this.browser) await this.browser.close(); }
}

// --- LAYER 3: EVASION & INTERCEPTION ---

class NetworkInterceptor {
  private page: Page;
  private jsContents: Map<string, string> = new Map();
  private htmlContent: string = '';

  constructor(page: Page) {
    this.page = page;
  }

  async enableInterception(): Promise<void> {
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
        } catch (e) { /* Ignore */ }
      }
      
      if (url === this.page.url() && contentType.includes('html')) {
        try {
          this.htmlContent = await response.text();
        } catch (e) { /* Ignore */ }
      }
    });
  }

  getCapturedScripts(): Map<string, string> { return this.jsContents; }
  getHtmlContent(): string { return this.htmlContent; }
}

// --- LAYER 4: AST ANALYZER ---

class ASTAnalyzer {
  private findings: FoundItem[] = [];
  
  // More aggressive patterns for minified code
  private secretPatterns = [
    { regex: /(?:api[_-]?key|apikey|token|secret|password)\s*[:=]\s*['"]([a-zA-Z0-9_\-]{8,})['"]/i, type: 'SECRET', severity: 'HIGH' },
    { regex: /AKIA[0-9A-Z]{16}/, type: 'SECRET', severity: 'CRITICAL' },
    { regex: /ghp_[a-zA-Z0-9]{36}/, type: 'SECRET', severity: 'CRITICAL' },
    { regex: /eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*/, type: 'SECRET', severity: 'HIGH' } // JWT
  ];

  private endpointPatterns = [
    /['"`](\/api\/[a-zA-Z0-9\/\._-]+)['"`]/g,
    /['"`](\/v\d+\/[a-zA-Z0-9\/\._-]+)['"`]/g,
    /axios\.(get|post|put|delete)$['"`]([^'"]+)['"`]/g,
    /fetch$['"`]([^'"]+)['"`]/g
  ];

  analyze(sourceCode: string, sourceUrl: string): void {
    if (!sourceCode) return;

    // 1. Regex Scan
    this.secretPatterns.forEach(pattern => {
      let match;
      while ((match = pattern.regex.exec(sourceCode)) !== null) {
        this.addFinding({ type: 'SECRET', value: match[1] || match[0], sourceFile: sourceUrl, severity: pattern.severity as any });
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
    } catch (e) { /* Silent fail for minified code errors */ }
  }

  private addFinding(item: FoundItem) {
    if (!this.findings.some(f => f.value === item.value && f.type === item.type)) {
      this.findings.push(item);
    }
  }

  getFindings(): FoundItem[] { return this.findings; }
}

// --- LAYER 5: REPORT AGGREGATOR ---

class ReportAggregator {
  private findings: FoundItem[] = [];
  addFindings(items: FoundItem[]) { this.findings.push(...items); }

  generateReport(target: string): string {
    const report: ReportData = {
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
  let cdpClient: StealthCDPClient | null = null;

  try {
    cdpClient = new StealthCDPClient();
    await cdpClient.launch();

    const page = cdpClient.getPage();
    if (!page) throw new Error("Page is null");

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
      if (html) analyzer.analyze(html, target + '/index.html');
    } else {
      scripts.forEach((code, url) => analyzer.analyze(code, url));
    }

    aggregator.addFindings(analyzer.getFindings());
    aggregator.generateReport(target);

    console.log('\n--- SCAN COMPLETE ---');
    console.log(`Total Findings: ${analyzer.getFindings().length}`);
    analyzer.getFindings().forEach(f => console.log(` - [${f.type}] ${f.value}`));

  } catch (error: any) {
    console.error('[Main] Error:', error.message);
  } finally {
    if (cdpClient) await cdpClient.close();
    process.exit(0);
  }
}

main();

