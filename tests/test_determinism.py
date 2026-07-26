"""Determinism / reproducibility checks."""
import sys
sys.path.insert(0, "D:\\LIT-DISC")
import numpy as np
from src.pipeline import LITDISCPipeline


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail, "ok": ok}


def run_all() -> list[dict]:
    results = []
    text_path = "D:\\LIT-DISC\\data\\sample.txt"

    # Run pipeline 3 times with identical config
    scores = []
    for i in range(3):
        pipeline = LITDISCPipeline()
        result = pipeline.run(text_path)
        scores.append(result.ldri.ldri)

    # Check all 3 are identical
    all_close = all(abs(s - scores[0]) < 1e-12 for s in scores)
    results.append(check("Deterministic: 3 runs identical",
                         all_close,
                         f"scores={[f'{s:.10f}' for s in scores]}"))

    # Check embedding is identical across runs
    pipeline_a = LITDISCPipeline()
    pipeline_b = LITDISCPipeline()
    ra = pipeline_a.run(text_path)
    rb = pipeline_b.run(text_path)
    vec_a = ra.embedding.to_vector()
    vec_b = rb.embedding.to_vector()
    max_diff = float(np.max(np.abs(vec_a - vec_b)))
    results.append(check("Deterministic: embeddings identical",
                         max_diff < 1e-12,
                         f"max_diff={max_diff:.2e}"))

    # Check component fractions are deterministic
    fracs_a = list(ra.ldri.fractions.values())
    fracs_b = list(rb.ldri.fractions.values())
    frac_diff = max(abs(a - b) for a, b in zip(fracs_a, fracs_b))
    results.append(check("Deterministic: fractions identical",
                         frac_diff < 1e-12,
                         f"max_frac_diff={frac_diff:.2e}"))

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"  DETERMINISM / REPRODUCIBILITY")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    FAIL {r['name']}: {r['detail']}")
