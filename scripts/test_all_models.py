"""Run all configured NOORIX model smoke tests from a source checkout."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noorix.model_testing import run_all_tests


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sequential NOORIX local model test suite")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "model-tests")
    args = parser.parse_args(argv)
    report = run_all_tests(args.config, args.report_dir)
    for item in report["results"]:
        print("[{}] {} ({}s): {}".format(
            item["status"], item["name"], item["duration_seconds"], item["message"]
        ))
        if item.get("details", {}).get("reason"):
            print("       " + item["details"]["reason"])
    print("TOTALS: " + json.dumps(report["totals"]))
    print("JSON REPORT: " + report["report_json"])
    print("MARKDOWN REPORT: " + report["report_markdown"])
    print("PRODUCTION READINESS: NOT CERTIFIED")
    return 1 if report["totals"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
