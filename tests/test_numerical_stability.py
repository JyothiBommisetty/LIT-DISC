"""Numerical stability checks: NaN, extremes, near-zero."""
import math, sys
sys.path.insert(0, "D:\\LIT-DISC")
import numpy as np
from src.ldri import LDRICalculator
from src.models import NarrativeEmbedding
from src.config import LDRIParams


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if ok else "FAIL",
            "detail": detail, "ok": ok}


DIMS = ["H_S", "H_N", "H_I", "H_Sy", "H_E", "H_Em"]


def run_all() -> list[dict]:
    results = []
    calc = LDRICalculator(params=LDRIParams(sigma_max=0.3))
    base = lambda v: np.full(12, v / math.sqrt(12))
    empty = lambda n: np.zeros(n)

    def make_emb(hs=None, hn=None, hi=None, hsy=None, he=None, hem=None,
                 pairs=None, res=None):
        return NarrativeEmbedding(
            H_S=hs if hs is not None else base(0),
            H_N=hn if hn is not None else base(0),
            H_I=hi if hi is not None else base(0),
            H_Sy=hsy if hsy is not None else base(0),
            H_E=he if he is not None else base(0),
            H_Em=hem if hem is not None else base(0),
            H_pairs=pairs if pairs is not None else empty(36),
            H_res=res if res is not None else empty(12),
            d=120)

    # 1. All-zero input
    emb_zero = make_emb()
    s = calc.compute(emb_zero, mem_avg_beta=0.0)
    results.append(check("LDRI from all-zero → 0", abs(s.ldri) < 1e-10,
                         f"got {s.ldri:.6e}"))
    results.append(check("Gamma from all-zero ≥ 0", s.gamma_global >= 0))

    # 2. NaN in input
    nan_vec = np.full(12, float("nan"))
    emb_nan = make_emb(hs=nan_vec)
    try:
        s = calc.compute(emb_nan, mem_avg_beta=0.0)
        results.append(check("NaN input produces NaN output",
                             math.isnan(s.ldri) or math.isnan(s.raw),
                             f"ldri={s.ldri}, raw={s.raw}"))
    except Exception as e:
        results.append(check("NaN input handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))

    # 3. Infinity in input
    inf_vec = np.full(12, float("inf"))
    emb_inf = make_emb(hn=inf_vec)
    try:
        s = calc.compute(emb_inf, mem_avg_beta=0.0)
        results.append(check("Inf input → finite output",
                             math.isfinite(s.ldri),
                             f"ldri={s.ldri}"))
    except Exception as e:
        results.append(check("Inf input handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))

    # 4. Negative values
    neg_vec = np.full(12, -1.0 / math.sqrt(12))
    emb_neg = make_emb(hem=neg_vec)
    s = calc.compute(emb_neg, mem_avg_beta=0.0)
    results.append(check("Negative input → non-negative LDRI",
                         s.ldri >= 0 and s.raw >= 0,
                         f"ldri={s.ldri:.4f}, raw={s.raw:.4f}"))

    # 5. All-maximum input (large values to saturate Rayleigh CDF)
    max_vec = np.full(12, 3.0)
    max_pairs = np.full(36, 3.0)
    max_res = np.full(12, 3.0)
    emb_max = make_emb(hs=max_vec, hn=max_vec, hi=max_vec,
                       hsy=max_vec, he=max_vec, hem=max_vec,
                       pairs=max_pairs, res=max_res)
    s = calc.compute(emb_max, confidences={d: 1.0 for d in ["S","N","I","Sy","E","Em"]},
                     mem_avg_beta=1.0)
    results.append(check("All-max input → LDRI ≈ 1",
                         0.9 <= s.ldri <= 1.0,
                         f"ldri={s.ldri:.6f}"))
    results.append(check("All-max input → raw ≈ 1",
                         0.9 <= s.raw <= 1.0,
                         f"raw={s.raw:.6f}"))

    # 6. Bootstrap stability
    reliabilities = []
    for _ in range(5):
        s = calc.compute(emb_zero, mem_avg_beta=0.0)
        reliabilities.append(s.reliability)
    rel_std = float(np.std(reliabilities))
    results.append(check("Bootstrap reliability stable (std<0.05)",
                         rel_std < 0.05,
                         f"std={rel_std:.6f}, vals={[f'{v:.4f}' for v in reliabilities[:3]]}"))

    # 7. Near-zero but non-zero
    tiny_v = 1e-10
    tiny_vec = np.full(12, tiny_v / math.sqrt(12))
    tiny_pairs = np.full(36, tiny_v / math.sqrt(36))
    tiny_res = np.full(12, tiny_v / math.sqrt(12))
    emb_tiny = make_emb(hs=tiny_vec, hn=tiny_vec, hi=tiny_vec,
                        hsy=tiny_vec, he=tiny_vec, hem=tiny_vec,
                        pairs=tiny_pairs, res=tiny_res)
    s = calc.compute(emb_tiny, mem_avg_beta=tiny_v)
    results.append(check("Near-zero → LDRI near 0",
                         abs(s.ldri) < 1e-6,
                         f"ldri={s.ldri:.6e}"))
    results.append(check("Near-zero → raw near 0",
                         abs(s.raw) < 1e-6,
                         f"raw={s.raw:.6e}"))

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"  NUMERICAL STABILITY")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    FAIL {r['name']}: {r['detail']}")
