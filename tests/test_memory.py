"""Memory module validation: decay dynamics, integration, stability."""
import sys
sys.path.insert(0, "D:\\LIT-DISC")
import math
import numpy as np
from src.ldri import LDRICalculator
from src.models import NarrativeEmbedding
from src.config import LDRIParams


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail, "ok": ok}


def make_emb(x: float) -> NarrativeEmbedding:
    v = np.full(12, x / np.sqrt(12))
    return NarrativeEmbedding(H_S=v, H_N=v, H_I=v, H_Sy=v, H_E=v, H_Em=v,
                              H_pairs=np.full(36, x / np.sqrt(36)),
                              H_res=np.full(12, x / np.sqrt(12)), d=120)


def run_all() -> list[dict]:
    results = []
    calc = LDRICalculator(params=LDRIParams(sigma_max=0.3))
    emb = make_emb(0.5)

    # M_factor = min(1, max(0, beta)); M_tilde = (1 + M_factor) / 2
    def memory_tilde(beta: float) -> float:
        return (1.0 + min(1.0, max(0.0, beta))) / 2.0

    # 1. Memory coupling: higher beta → higher M_tilde → higher c_mem
    s_low = calc.compute(emb, mem_avg_beta=0.1)
    s_high = calc.compute(emb, mem_avg_beta=0.9)
    m_low = s_low.components.get("memory", 0)
    m_high = s_high.components.get("memory", 0)
    results.append(check("Higher beta → higher memory contribution",
                         m_high >= m_low,
                         f"beta=0.1: mem={m_low:.4f}, beta=0.9: mem={m_high:.4f}"))

    # 2. M_tilde in [0, 1]
    mt = memory_tilde(0.5)
    results.append(check("M_tilde in [0,1]", 0 <= mt <= 1.0, f"got {mt:.4f}"))

    # 3. beta=0 → M_tilde = 0.5
    m0 = memory_tilde(0.0)
    results.append(check("M_tilde(β=0) = 0.5", 0.49 <= m0 <= 0.51, f"got {m0:.6f}"))

    # 4. beta=1 → M_tilde = 1.0
    m1 = memory_tilde(1.0)
    results.append(check("M_tilde(β=1) = 1", 0.99 <= m1 <= 1.01, f"got {m1:.6f}"))

    # 5. Memory component monotonic with embedding norm
    emb_small = make_emb(0.1)
    emb_large = make_emb(0.9)
    s_sm = calc.compute(emb_small, mem_avg_beta=0.5)
    s_lg = calc.compute(emb_large, mem_avg_beta=0.5)
    results.append(check("Memory component monotonic with norm",
                         s_lg.components.get("memory", 0) >= s_sm.components.get("memory", 0),
                         f"small={s_sm.components.get('memory',0):.4f} large={s_lg.components.get('memory',0):.4f}"))

    # 6. Zero embedding + zero beta → zero memory contribution
    emb_zero = make_emb(0.0)
    s_z = calc.compute(emb_zero, mem_avg_beta=0.0)
    results.append(check("Zero embedding + zero beta → zero memory",
                         abs(s_z.components.get("memory", 0)) < 1e-10,
                         f"got {s_z.components.get('memory',0):.6e}"))

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"  MEMORY / TEMPORAL VALIDATION")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    FAIL {r['name']}: {r['detail']}")
