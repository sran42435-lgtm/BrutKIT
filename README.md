# BrutKIT

BrutKIT is an interactive Python script for **authorized web application security testing**. It discovers parameters, creates payload variants, sends HTTP requests, analyzes responses, learns from feedback, and saves reports.

> **Warning:** use this tool only on systems you own or are explicitly authorized to test. Do not test public services without permission. Proxy, throttling, and encoding features do not remove the obligation to follow applicable laws, target policies, and rate limits.

## Features

- Parameter discovery from URLs, HTML forms, hidden fields, JavaScript, internal links, robots/sitemaps, common paths, and additional endpoints.
- Category-based payload generation with grammar validation, encoding variants, polyglots, and genetic evolution.
- Response analysis that classifies results as `server_output`, `raw_html`, or `blocked`.
- Feedback learning, response-based WAF fingerprinting, adaptive mutation selection, and anomaly detection.
- Proxy rotation, TLS fingerprints, adaptive throttling, backoff retries, and synchronous/asynchronous HTTP support.
- SQLite target storage and TXT/JSON result reports.

## Requirements

- Python 3.9 or newer.
- Network access to an authorized target.
- `pip` with permission to install dependencies.
- Chromium if Playwright browser features are required.

The script checks for `numpy`, `scipy`, `scikit-learn`, `requests`, `beautifulsoup4`, `lxml`, `httpx`, `h2`, `aiohttp`, `selectolax`, `playwright`, `tldextract`, `fake-useragent`, `curl-cffi`, and `tenacity`. `torch` is used only when available for the LSTM generator.

## Installation

Clone the repository and run:

```bash
cd Brutkit/src
python3 src/brutKIT.py
```

At startup, the script checks dependencies and attempts to install missing packages with `pip`. Because this check runs every time the script starts, make sure the active Python environment is the one you intend to use.

To install dependencies manually:

```bash
python3 -m pip install numpy scipy scikit-learn requests beautifulsoup4 lxml httpx h2 aiohttp selectolax playwright tldextract fake-useragent curl-cffi tenacity
python3 -m playwright install chromium
```

## Usage

1. Run `python3 brutkit.py`.
2. Enter the target URL, domain, or IP. If no scheme is provided, the script uses `http://`.
3. Review the discovered parameters.
4. Answer `Y` to continue testing or `N` to cancel.
5. Enter a payload count, or enter `max` for 500 payloads.
6. When the process finishes, review the terminal summary and reports in `brut_results/`.

Example target:

```text
http://localhost:8080/test
```

## Lobby Commands

| Command | Function |
| --- | --- |
| `/save <url>` | Save a target to the SQLite database. |
| `/viewdb` | List saved targets. |
| `/load <id>` | Load a target by database ID. |
| `/proxy <file>` | Use proxies from a file for the next process. |
| `/back` | Confirm leaving the lobby. |
| `/cancel` | Cancel a loaded target or cancellable process. |
| `/exit` or `/quit` | Exit the script. |

Proxy files are read one line at a time. A common format is a proxy URL such as `http://127.0.0.1:8080`; blank lines and comments may be ignored by the current parser implementation.

## Internal Workflow

`BRUTPipeline` runs the process in several phases:

1. `phase1_discover()` finds parameters and endpoints within the target domain scope.
2. `phase2_generate(count)` creates validated payloads and initializes the evolution population.
3. `phase3_inject()` tests payload/parameter combinations with throttling, proxies, retries, and response analysis.
4. `phase3_advanced_retry()` creates additional variants from `blocked` or `raw_html` results.
5. `phase4_save()` writes reports, while `print_summary()` displays a summary.

## Output Files

- `brut_targets.db`: SQLite database for targets saved with `/save`.
- `brut_results/`: reports organized by domain and year.
- `.txt` files: human-readable reports containing evidence, status, proxy, and statistics.
- `.json` files: structured data for further analysis, including all payload results and the evolution log.

`server_output` results are indications that require manual verification. `raw_html` and `blocked` statuses are not proof of a vulnerability or proof that a system is secure. Use the available confirmation mechanism and document the authorization and scope of each test.

## Component Overview

- `TargetDatabase`: SQLite target storage.
- `ParameterDiscovery`, `FastHTMLParser`, and endpoint scanners: discovery.
- `MLPayloadGenerator`, `PolyglotGenerator`, and `GeneticEvolver`: payload creation and evolution.
- `Injector`, `ResponseAnalyzer`, and `SecondOrderConfirmer`: requests, classification, and result confirmation.
- `ProxyPoolManager`, `AdaptiveThrottler`, `RetryEngine`, and `TLSFingerprintEngine`: request management.
- `ReportSaver` and `DetailedLogger`: report output and terminal display.

## Troubleshooting

- **Dependency installation fails:** activate a virtual environment, upgrade `pip`, and install dependencies manually.
- **No parameters are found:** verify that the URL is reachable, the endpoint accepts input, and the target is in the authorized scope.
- **Playwright is unavailable:** browser features are skipped; install the package and Chromium if required.
- **Many 429 responses:** stop or reduce the test load according to the target policy; do not bypass service limits without permission.
- **Heavy dependencies:** Linux is recommended. Termux is not recommended unless its environment is properly configured.

## License

See [LICENSE](LICENSE).
