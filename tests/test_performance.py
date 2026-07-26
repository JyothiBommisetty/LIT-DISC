"""Performance profiling: pipeline runtime, scaling."""
import sys, time
sys.path.insert(0, "D:\\LIT-DISC")
from src.pipeline import LITDISCPipeline


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail, "ok": ok}


def run_all() -> list[dict]:
    results = []
    text_path = "D:\\LIT-DISC\\data\\sample.txt"  # ~400 words

    pipeline = LITDISCPipeline()

    # Cold start timing
    t0 = time.perf_counter()
    _ = pipeline.run(text_path)
    t_cold = time.perf_counter() - t0
    results.append(check("Cold start < 2s (tiny text)", t_cold < 2.0,
                         f"cold={t_cold:.3f}s"))

    # Single warm call timing
    t0 = time.perf_counter()
    _ = pipeline.run(text_path)
    t_warm = time.perf_counter() - t0
    results.append(check("Warm call < 0.5s", t_warm < 0.5,
                         f"warm={t_warm:.3f}s"))

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    FAIL {r['name']}: {r['detail']}")
