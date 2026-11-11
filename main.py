# main.py — rich UI: banner, input target, pilih modul, progress, jalankan orchestrator
import asyncio
import sys
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

from modules import REGISTRY, get_available_modules
from orchestrator import run_scan, get_modules_by_category

console = Console()

BANNER = r"""
███████╗████████╗ █████╗ ███████╗███████╗███╗   ██╗███████╗████████╗███████╗██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
███████╗   ██║   ███████║███████╗█████╗  ██╔██╗ ██║█████╗     ██║   █████╗  ██████╔╝
╚════██║   ██║   ██╔══██║╚════██║██╔══╝  ██║╚██╗██║██╔══╝     ██║   ██╔══╝  ██╔══██╗
███████║   ██║   ██║  ██║███████║███████╗██║ ╚████║███████╗   ██║   ███████╗██║  ██║
╚══════╝   ══╝   ══╝  ══╝╚══════╝╚══════╝══╝  ╚═══╝╚══════╝   ══╝   ╚══════╝══╝  ══╝
                                        Modular Web Security Scanner
"""


def print_banner():
    console.print(Panel(BANNER, style="bold cyan", border_style="blue"))
    console.print("[dim]ScanRider v1.0 | Legal targets only (example.com, badssl.com, testphp.vulnweb.com)[/dim]\n")


def print_module_table():
    table = Table(title="Available Modules", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Module", style="green", width=20)
    table.add_column("Category", style="yellow", width=10)
    table.add_column("Description", style="white")

    categories = ["recon", "network", "web", "vuln"]
    cat_colors = {"recon": "blue", "network": "cyan", "web": "green", "vuln": "red"}

    idx = 1
    for cat in categories:
        modules = get_modules_by_category(cat)
        for m in modules:
            info = REGISTRY[m]
            table.add_row(str(idx), m, f"[{cat_colors.get(cat, 'white')}]{cat}[/]", info["description"])
            idx += 1

    console.print(table)


def select_modules() -> List[str]:
    print_module_table()
    console.print("\n[bold]Pilih modul:[/bold]")
    console.print("  [dim]• Ketik nama modul dipisah koma (contoh: headers_scanner,tech_stack)[/dim]")
    console.print("  [dim]• Ketik kategori: recon, network, web, vuln[/dim]")
    console.print("  [dim]• Ketik 'all' untuk semua modul[/dim]")
    console.print("  [dim]• Ketik 'quick' untuk modul ringan (headers, tech_stack, ssl)[/dim]")

    choice = Prompt.ask("\n[cyan]Modules[/cyan]", default="quick").strip().lower()

    if choice == "all":
        return list(REGISTRY.keys())
    elif choice == "quick":
        return ["headers_scanner", "tech_stack", "ssl_scanner"]
    elif choice in ["recon", "network", "web", "vuln"]:
        return get_modules_by_category(choice)
    else:
        selected = [m.strip() for m in choice.split(",")]
        valid = [m for m in selected if m in REGISTRY]
        invalid = [m for m in selected if m not in REGISTRY]
        if invalid:
            console.print(f"[yellow]Warning: Modul tidak dikenal: {', '.join(invalid)}[/yellow]")
        return valid


def get_target() -> str:
    while True:
        target = Prompt.ask("\n[cyan]Target URL/domain[/cyan]", default="https://example.com").strip()
        if target:
            if not target.startswith(("http://", "https://")):
                target = "https://" + target
            return target
        console.print("[red]Target tidak boleh kosong[/red]")


def get_config() -> dict:
    if Confirm.ask("\n[cyan]Konfigurasi custom?[/cyan]", default=False):
        config = {}
        console.print("[dim]Masukkan JSON config per modul (kosong untuk skip)[/dim]")
        for name in REGISTRY:
            val = Prompt.ask(f"  Config untuk [green]{name}[/green]", default="{}")
            try:
                config[name] = json.loads(val)
            except Exception:
                console.print(f"  [red]Invalid JSON untuk {name}, skip[/red]")
        return config
    return {}


async def run_with_progress(target: str, modules: List[str], config: dict):
    """Jalankan scan dengan progress bar."""
    results = []
    completed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scanning...", total=len(modules))

        async def _run_one(module: str):
            from modules import REGISTRY
            scan_func = REGISTRY[module]["scan"]
            try:
                result = await scan_func(target, config.get(module, {}))
            except Exception as e:
                result = {
                    "module": module,
                    "target": target,
                    "findings": [{"type": "error", "detail": str(e), "severity": "error"}],
                    "summary": f"Error: {e}",
                    "severity": "error",
                    "duration": 0
                }
            completed.append(module)
            progress.update(task, advance=1, description=f"[cyan]Scanning... ({len(completed)}/{len(modules)})")
            return result

        tasks = [_run_one(m) for m in modules]
        results = await asyncio.gather(*tasks)

    return results


def print_results(results: List[dict]):
    for r in results:
        module = r.get("module", "unknown")
        severity = r.get("severity", "info")
        findings = r.get("findings", [])
        duration = r.get("duration", 0)

        sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "blue", "info": "green", "error": "red"}.get(severity, "white")

        console.print(f"\n[bold]{module}[/bold] [dim]({duration:.2f}s)[/dim]")
        console.print(f"  Summary: {r.get('summary', 'N/A')}")
        console.print(f"  Severity: [{sev_color}]{severity.upper()}[/{sev_color}]")
        console.print(f"  Findings: {len(findings)}")

        if findings:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Type", style="cyan", width=20)
            table.add_column("Detail", style="white", width=60)
            table.add_column("Severity", style="white", width=10)

            for f in findings[:10]:
                ftype = f.get("type") or f.get("header") or f.get("path") or f.get("technology") or "finding"
                detail = str(f.get("detail") or f.get("value") or f.get("evidence") or f.get("url") or "")[:80]
                fsev = f.get("severity", "info")
                fcolor = sev_color.get(fsev, "white")
                table.add_row(ftype, detail, f"[{fcolor}]{fsev}[/{fcolor}]")

            if len(findings) > 10:
                table.add_row("...", f"dan {len(findings) - 10} temuan lain", "")
            console.print(table)


async def main():
    print_banner()

    target = get_target()
    modules = select_modules()
    config = get_config()

    if not modules:
        console.print("[red]Tidak ada modul valid dipilih[/red]")
        return

    console.print(f"\n[bold green]Starting scan on {target}[/bold green]")
    console.print(f"[dim]Modules: {', '.join(modules)}[/dim]\n")

    try:
        results = await run_with_progress(target, modules, config)

        console.print("\n" + "=" * 60)
        console.print("[bold green]SCAN COMPLETE[/bold green]")
        console.print("=" * 60)

        print_results(results)

        if Confirm.ask("\n[cyan]Generate report?[/cyan]", default=True):
            from handlers import generate_report
            paths = generate_report(target, results)
            console.print("\n[green]Reports generated:[/green]")
            for fmt, path in paths.items():
                console.print(f"  {fmt.upper()}: {path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    import json
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        target = sys.argv[2] if len(sys.argv) > 2 else "https://example.com"
        modules = sys.argv[3].split(",") if len(sys.argv) > 3 else ["headers_scanner", "tech_stack"]
        config = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
        result = asyncio.run(run_scan(target, modules, config))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        asyncio.run(main())