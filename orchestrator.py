# orchestrator.py — asyncio.gather + semaphore, pilih modul dinamis, hasil -> report_generator
import asyncio
import time
from typing import Dict, Any, List, Optional

from modules import REGISTRY
from handlers import generate_report

__all__ = ["run_scan", "Orchestrator"]


class Orchestrator:
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_module(self, name: str, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Jalankan satu modul dengan semaphore."""
        if name not in REGISTRY:
            return {
                "module": name,
                "target": target,
                "findings": [{"type": "error", "detail": f"Modul '{name}' tidak ditemukan", "severity": "error"}],
                "summary": "Modul tidak terdaftar",
                "severity": "error",
                "duration": 0
            }

        module_info = REGISTRY[name]
        scan_func = module_info["scan"]

        async with self.semaphore:
            try:
                result = await scan_func(target, config.get(name, {}))
                return result
            except Exception as e:
                return {
                    "module": name,
                    "target": target,
                    "findings": [{"type": "error", "detail": f"Exception: {str(e)}", "severity": "error"}],
                    "summary": f"Error: {e}",
                    "severity": "error",
                    "duration": 0
                }

    async def run(
        self,
        target: str,
        modules: List[str],
        config: Dict[str, Any] = None,
        generate_report_flag: bool = True,
        report_formats: List[str] = None,
        report_dir: str = "storage"
    ) -> Dict[str, Any]:
        """Jalankan multiple modul concurrent, generate report."""
        config = config or {}
        report_formats = report_formats or ["json", "txt", "html"]

        start = time.time()
        tasks = [self._run_module(m, target, config) for m in modules]
        results = await asyncio.gather(*tasks)

        total_duration = time.time() - start

        report_paths = {}
        if generate_report_flag:
            try:
                report_paths = generate_report(target, results, report_dir, report_formats)
            except Exception as e:
                report_paths = {"error": str(e)}

        return {
            "target": target,
            "modules_requested": modules,
            "modules_completed": [r.get("module") for r in results],
            "results": results,
            "report_paths": report_paths,
            "total_duration": round(total_duration, 2),
            "timestamp": time.time()
        }


async def run_scan(
    target: str,
    modules: List[str],
    config: Dict[str, Any] = None,
    max_concurrent: int = 5,
    generate_report_flag: bool = True,
    report_formats: List[str] = None,
    report_dir: str = "storage"
) -> Dict[str, Any]:
    """Entry point untuk menjalankan scan."""
    orchestrator = Orchestrator(max_concurrent)
    return await orchestrator.run(
        target, modules, config, generate_report_flag, report_formats, report_dir
    )


def get_available_modules() -> List[Dict[str, Any]]:
    """Return daftar modul tersedia dari REGISTRY."""
    return [
        {
            "name": name,
            "description": info["description"],
            "category": info["category"],
            "requires": info["requires"]
        }
        for name, info in REGISTRY.items()
    ]


def get_modules_by_category(category: str) -> List[str]:
    """Return nama modul berdasarkan kategori."""
    return [
        name for name, info in REGISTRY.items()
        if info["category"] == category
    ]


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python orchestrator.py <target> <module1,module2,...>")
        print("Available modules:")
        for m in get_available_modules():
            print(f"  {m['name']} ({m['category']}): {m['description']}")
        sys.exit(1)

    target = sys.argv[1]
    modules = sys.argv[2].split(",")

    result = asyncio.run(run_scan(target, modules))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))