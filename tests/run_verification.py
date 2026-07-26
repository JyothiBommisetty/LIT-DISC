"""Master verification runner — executes all test suites sequentially."""
import sys, subprocess, time
from pathlib import Path

SUITES = [
    ("Equation Verification",       "test_equation_verification.py"),
    ("Numerical Stability",         "test_numerical_stability.py"),
    ("Determinism / Reproducibility","test_determinism.py"),
    ("Performance",                 "test_performance.py"),
    ("Memory / Temporal Validation","test_memory.py"),
    ("Edge Cases",                  "test_edge_cases.py"),
]

def run_suite(name: str, script: str) -> dict:
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / script)],
        capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - t0
    import re
    combined = result.stdout + result.stderr
    # Parse last summary line: "Total: N  |  Passed: N  |  Failed: N"
    matches = re.findall(r"Total:\s*(\d+)\s*\|\s*Passed:\s*(\d+)\s*\|\s*Failed:\s*(\d+)", combined)
    if matches:
        _, passed, failed = int(matches[-1][0]), int(matches[-1][1]), int(matches[-1][2])
    else:
        passed = sum(1 for line in combined.split("\n") if "[PASS]" in line)
        failed = sum(1 for line in combined.split("\n") if "[FAIL]" in line)
    return {"name": name, "passed": passed, "failed": failed,
            "elapsed": elapsed, "output": combined}


def main():
    print(f"\n{'='*70}")
    print(f"  LIT-DISC COMPREHENSIVE VERIFICATION SUITE")
    print(f"{'='*70}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    total_p, total_f = 0, 0
    for name, script in SUITES:
        res = run_suite(name, script)
        total_p += res["passed"]
        total_f += res["failed"]
        status = "PASS" if res["failed"] == 0 else "FAIL"
        print(f"  [{status}] {name:40s}  {res['passed']:2d} pass, "
              f"{res['failed']:2d} fail  ({res['elapsed']:.2f}s)")
        if res["failed"] > 0:
            for line in res["output"].split("\n"):
                stripped = line.strip()
                if stripped.startswith("FAIL") and "PASS" not in stripped:
                    print(f"         {stripped}")
    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE: {total_p} passed, {total_f} failed "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*70}")
    return 1 if total_f > 0 else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
