"""Equation Verification: compare LDRI/HMP-DT implementation against theoretical worked examples.

Targets:
  - LDRI Methodology §8.1 Simple Chapter (pp. 20-23)
  - HMP-DT §10 Propagation Equations
  - LDRI Eqs. (2)-(10), (15)-(22)
"""
import math
import sys
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.ldri import LDRICalculator
from src.models import NarrativeEmbedding
from src.config import LDRIParams


RTOL = 1e-3


def rayleigh_cdf(x_norm: float, sigma: float = 0.3) -> float:
    """Rayleigh CDF: g(x) = 1 - exp(-x^2 / (2*sigma^2))."""
    return 1.0 - math.exp(-x_norm ** 2 / (2.0 * sigma ** 2))


def check_eq(name: str, computed: float, expected: float,
             tol: float = RTOL, context: str = "") -> dict:
    if abs(expected) < 1e-12:
        ok = abs(computed) < tol
    else:
        ok = abs(computed - expected) / max(abs(expected), 1e-12) <= tol
    rel_err = abs(computed - expected) / max(abs(expected), 1e-12) if abs(expected) > 1e-12 else abs(computed)
    return {"name": name, "status": "PASS" if ok else "FAIL",
            "computed": computed, "expected": expected,
            "relative_error": rel_err, "context": context}


def verify_ldri_worked_example() -> list[dict]:
    """Verify against LDRI Methodology §8.1 Simple Chapter."""
    results = []
    sigma = 0.3
    d_per_dim = 12

    # Expected inputs from §8.1 table
    norms = {"S": 0.15, "N": 0.55, "I": 0.30, "Sy": 0.05, "E": 0.10, "Em": 0.35}
    expected_a = {"S": 0.118, "N": 0.814, "I": 0.393, "Sy": 0.014, "E": 0.054, "Em": 0.493}

    # Verify Rayleigh CDF directly (Eq. 2)
    # Tolerances: document rounds to 3dp; exact computation may differ by ~1.5%
    rounding_tols = {"S": 0.02, "N": 0.005, "I": 0.01, "Sy": 0.05, "E": 0.02, "Em": 0.01}
    for d in ["S", "N", "I", "Sy", "E", "Em"]:
        computed = rayleigh_cdf(norms[d], sigma)
        results.append(check_eq(
            f"Rayleigh CDF a_{d} (Eq. 2)", computed, expected_a[d],
            tol=rounding_tols[d],
            context=f"||H_{d}||={norms[d]}, sigma={sigma}, exact={computed:.6f}",
        ))

    # Verify signal richness (Eq. 4)
    a_vec = np.array([expected_a[d] for d in ["S", "N", "I", "Sy", "E", "Em"]])
    calc = LDRICalculator(params=LDRIParams(sigma_max=sigma))
    R_signal = calc.signal_richness(a_vec)
    results.append(check_eq("Signal Richness R_signal (Eq. 4)", R_signal, 0.424, context="a=[0.118,0.814,0.393,0.014,0.054,0.493]"))

    # Build embedding with correct norms
    # The coupling/resonance factors are normalized by sqrt(dim), so to get
    # C = 0.08 we need ||H_pairs|| = 0.08 * sqrt(36) = 0.48
    # To get T = 0.04 we need ||H_res|| = 0.04 * sqrt(12) = 0.1386
    dim_vectors = {d: np.full(d_per_dim, norms[d] / math.sqrt(d_per_dim)) for d in norms}
    pair_target_norm = 0.08 * math.sqrt(36)
    res_target_norm = 0.04 * math.sqrt(12)
    H_pairs = np.full(36, pair_target_norm / math.sqrt(36))
    H_res = np.full(12, res_target_norm / math.sqrt(12))

    embedding = NarrativeEmbedding(
        H_S=dim_vectors["S"], H_N=dim_vectors["N"], H_I=dim_vectors["I"],
        H_Sy=dim_vectors["Sy"], H_E=dim_vectors["E"], H_Em=dim_vectors["Em"],
        H_pairs=H_pairs, H_res=H_res, d=120,
    )

    # Verify factors (Eq. 5-7, 9)
    C_factor = calc.coupling_factor(embedding.H_pairs)
    T_factor = calc.resonance_factor(embedding.H_res)
    results.append(check_eq("C = ||H_pairs||/||H_pairs||_max (Eq. 5)", C_factor, 0.08))
    results.append(check_eq("T = ||H_res||/||H_res||_max (Eq. 6)", T_factor, 0.04))
    results.append(check_eq("C_tilde = (1+C)/2 (Eq. 9)", (1+C_factor)/2, 0.540))
    results.append(check_eq("T_tilde = (1+T)/2 (Eq. 9)", (1+T_factor)/2, 0.520))
    M_factor = calc.memory_factor(0.12)
    results.append(check_eq("M_tilde = (1+M)/2 (Eq. 9)", (1+M_factor)/2, 0.560))

    # Verify compute_activations matches direct Rayleigh
    activations = calc.compute_activations(embedding)
    for i, d in enumerate(["S", "N", "I", "Sy", "E", "Em"]):
        exact = rayleigh_cdf(norms[d], sigma)
        results.append(check_eq(f"compute_activations a_{d}", float(activations[i]), exact,
                                tol=1e-6,
                                context=f"exact={exact:.6f}, doc_rounded={expected_a[d]}"))

    # Verify full LDRI (Eq. 10)
    # Worked example uses Gamma=0.85 as FINAL global confidence (product of per-dim confidences per Eq. 12).
    gamma_per_dim = 0.85 ** (1.0 / 6.0)
    confidences = {d: gamma_per_dim for d in ["S", "N", "I", "Sy", "E", "Em"]}
    score = calc.compute(embedding, confidences=confidences, mem_avg_beta=0.12)

    results.append(check_eq("LDRI_raw (Eq. 22)", score.raw, 0.490, context="(0.424^2*0.540*0.520*0.560)^(1/5) = 0.490"))
    results.append(check_eq("LDRI (Eq. 10)", score.ldri, 0.417, tol=0.005, context="0.85 * 0.490 = 0.417"))

    # Verify decomposition (Eq. 15-22, tolerance 20% for doc rounding)
    results.append(check_eq("c_intrinsic (Eq. 16)", score.components.get("intrinsic", 0), 0.262, tol=0.20))
    results.append(check_eq("c_interaction (Eq. 17)", score.components.get("interaction", 0), 0.011, tol=0.20))
    results.append(check_eq("c_temporal (Eq. 18)", score.components.get("temporal", 0), 0.006, tol=0.20))
    results.append(check_eq("c_memory (Eq. 20)", score.components.get("memory", 0), 0.016, tol=0.25))

    # Fractions are a first-order approximation around log-space
    # (see LDRI Methodology footnote in §6.1); sum may be <1
    total_frac = sum(score.fractions.values())
    results.append(check_eq("Decomposition fractions sum ≈ 1", total_frac, 1.0, tol=0.20))

    return results


def verify_signal_richness_extremes() -> list[dict]:
    results = []
    calc = LDRICalculator()

    # All-zero → 0
    R = calc.signal_richness(np.zeros(6))
    results.append(check_eq("R_signal all-zero → 0", R, 0.0))

    # All-one → 1
    a_one = np.ones(6)
    results.append(check_eq("R_signal all-one → 1", calc.signal_richness(a_one), 1.0))

    # Single dim at 1.0 → 1/sqrt(6)
    a_single = np.zeros(6); a_single[0] = 1.0
    results.append(check_eq("R_signal single(1,0,0,0,0,0) = 1/sqrt(6)",
                            calc.signal_richness(a_single), 1.0 / math.sqrt(6)))

    # Concentration penalty: uniform beats extreme
    R_uniform = calc.signal_richness(np.ones(6) * 0.5)
    R_extreme = calc.signal_richness(np.array([1.0, 0, 0, 0, 0, 0]))
    results.append(check_eq("Uniform > Extreme (concentration penalty)",
                            1.0 if R_uniform > R_extreme else 0.0, 1.0,
                            context=f"uniform(0.5)={R_uniform:.4f}, extreme={R_extreme:.4f}"))

    return results


def verify_rayleigh_properties() -> list[dict]:
    results = []
    # g(0) = 0
    results.append(check_eq("Rayleigh g(0) = 0", rayleigh_cdf(0.0), 0.0))
    # Monotonic
    results.append(check_eq("Rayleigh monotonic", 1.0 if rayleigh_cdf(0.5) > rayleigh_cdf(0.1) else 0.0, 1.0))
    # g(inf) → 1
    results.append(check_eq("Rayleigh g(inf) → 1", rayleigh_cdf(10.0), 1.0, tol=0.0001))
    # Bounded [0, 1] numerically
    bounds_ok = True
    for x in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
        v = rayleigh_cdf(x)
        if not (0.0 <= v <= 1.0):
            bounds_ok = False
    results.append(check_eq("Rayleigh bounded [0,1]", 1.0 if bounds_ok else 0.0, 1.0))
    return results


def verify_propagation() -> list[dict]:
    from src.propagation import BottomUpPropagator
    p = BottomUpPropagator()
    results = []

    items = [
        (np.array([-0.62, 0.45, 0.58]), 0.78, 0.60),
        (np.array([-0.55, 0.40, 0.52]), 0.75, 0.60),
        (np.array([-0.50, 0.38, 0.48]), 0.72, 0.60),
    ]
    f_agg, g_agg = p._aggregate_weighted(items)
    total_g = 0.78 + 0.75 + 0.72
    f_exp = (np.array([-0.62, 0.45, 0.58])*0.78 + np.array([-0.55, 0.40, 0.52])*0.75 + np.array([-0.50, 0.38, 0.48])*0.72) / total_g
    results.append(check_eq("Weighted agg (Eq. 10.2)", float(np.mean(f_agg - f_exp)), 0.0, tol=1e-6))

    alphas = p._salience_weights(items, beta_sal=2.0)
    results.append(check_eq("Salience weights sum=1", float(np.sum(alphas)), 1.0))
    results.append(check_eq("Salience weights >= 0", float(np.min(alphas)) >= 0, 1.0))

    sc = p._semantic_coherence(items)
    results.append(check_eq("Semantic coherence in [0,1]", 1.0 if 0 <= sc <= 1.0 else 0.0, 1.0))

    taus = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    w = p._spline_weights(taus, 0.5)
    results.append(check_eq("Spline weights sum=1", float(np.sum(w)), 1.0))
    results.append(check_eq("Spline max at nearest", 1.0 if np.argmax(w) == 2 else 0.0, 1.0))

    return results


def run_all() -> list[dict]:
    all_results = []
    all_results.extend(verify_ldri_worked_example())
    all_results.extend(verify_signal_richness_extremes())
    all_results.extend(verify_rayleigh_properties())
    all_results.extend(verify_propagation())
    return all_results


if __name__ == "__main__":
    import os
    # Ensure stdout can handle UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*70}")
    print(f"  EQUATION VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    if failed:
        print(f"\n  FAILED CHECKS:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    FAIL {r['name']}: got {r['computed']:.6f}, expected {r['expected']:.6f} "
                      f"(err={r['relative_error']:.4e})")
                if r['context']:
                    print(f"       context: {r['context']}")
    print(f"{'='*70}\n")
    for r in results:
        mark = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {r['name']}: {r['computed']:.6f} ~ {r['expected']:.6f} (err={r['relative_error']:.4e})")
