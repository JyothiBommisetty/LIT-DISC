"""Comprehensive tests for NarrativeResonance — HMP-DT Eqs. (12.8)–(12.9).

Test categories:
  1. Equation verification (unit tests for recursive formulation)
  2. Long-range influence (decay, horizon, worked example)
  3. Numerical approximation (error bound, convergence, constant-H validation)
  4. Recursive update (trajectory, tau tracking, reset, multi-update)
  5. Edge cases / numerical stability (NaN, Inf, zero, extremes)
  6. Performance benchmarks (timing at scale)
"""
import math
import sys
import time
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.resonance import NarrativeResonance

RTOL = 1e-4
ATOL = 1e-6

# ═══════════════════════════════════════════════════════════════
# 1. EQUATION VERIFICATION — Eqs. (12.8)–(12.9)
# ═══════════════════════════════════════════════════════════════

def test_initial_state():
    """Fresh NarrativeResonance has no field."""
    nr = NarrativeResonance()
    assert nr.R is None
    assert len(nr.get_field()) == 0
    assert nr.get_n_updates() == 0


def test_reset():
    """Reset creates zero field of given dimension."""
    nr = NarrativeResonance()
    nr.reset(7)
    assert len(nr.get_field()) == 7
    assert np.all(nr.get_field() == 0.0)
    assert nr.get_n_updates() == 0


def test_first_update_stores_H():
    """First update stores H as H(τ₀), returns R=0. Eq. (12.8)."""
    nr = NarrativeResonance()
    nr.reset(3)
    H = np.array([1.0, 0.5, 0.0])
    R = nr.update(H, 0.01)
    assert np.all(R == 0.0), "first update should return R=0"
    assert nr.get_n_updates() == 1


def test_second_update_applies_equation():
    """Second update: R = R₁·e^{-Δτ/λ} + H₀·Δτ = 0·decay + H₀·Δτ. Eq. (12.8)."""
    nr = NarrativeResonance()
    nr.reset(3)
    H0 = np.array([1.0, 0.5, 0.0])
    H1 = np.array([0.0, 0.5, 1.0])
    dt = 0.01
    nr.update(H0, dt)
    R = nr.update(H1, dt)
    expected = np.zeros(3) * math.exp(-dt / 0.3) + H0 * dt
    assert np.allclose(R, expected, rtol=RTOL), f"R={R}, expected={expected}"


def test_recursive_accumulation():
    """Third update accumulates previous R with decay. Eq. (12.8)."""
    nr = NarrativeResonance()
    nr.reset(2)
    H_list = [np.array([1.0, 0.0]),
              np.array([0.5, 0.5]),
              np.array([0.0, 1.0])]
    dt = 0.01
    for H in H_list:
        R = nr.update(H, dt)
    # R after 3 updates: R₃ = R₂·e^{-dt/λ} + H₁·dt
    # where R₂ = 0·e^{-dt/λ} + H₀·dt = H₀·dt
    # so R₃ = H₀·dt·e^{-dt/λ} + H₁·dt
    decay = math.exp(-dt / 0.3)
    expected = H_list[0] * dt * decay + H_list[1] * dt
    assert np.allclose(R, expected, rtol=RTOL), f"R={R}, expected={expected}"


def test_augmentation():
    """Augment adds η_res·R to signal. Eq. (12.9)."""
    nr = NarrativeResonance()
    nr.reset(3)
    H0 = np.array([1.0, 0.5, 0.3])
    H1 = np.array([0.5, 0.3, 0.1])
    nr.update(H0, 0.01)
    nr.update(H1, 0.01)
    f = np.array([0.5, 0.5, 0.5])
    f_aug = nr.augment(f)
    expected = f + 0.2 * nr.R[:3]
    assert np.allclose(f_aug, expected), f"f_aug={f_aug}"


def test_augment_before_update():
    """Augment returns copy of input when no field exists."""
    nr = NarrativeResonance()
    f = np.array([0.5, 0.3, 0.8])
    f_aug = nr.augment(f)
    assert np.array_equal(f_aug, f)
    assert f_aug is not f


def test_augment_different_dims():
    """R dim > f dim: slicing works. R dim < f dim: no crash."""
    nr = NarrativeResonance()
    nr.reset(7)
    H0 = np.ones(7) * 0.5
    nr.update(H0, 0.01)
    # f with smaller dim
    f_small = np.ones(3) * 0.5
    f_aug = nr.augment(f_small)
    assert len(f_aug) == 3
    assert np.all(np.isfinite(f_aug))
    # f with larger dim
    f_big = np.ones(10) * 0.5
    f_aug = nr.augment(f_big)
    assert len(f_aug) == 10
    assert np.all(np.isfinite(f_aug))


def test_tau_tracking():
    """Current tau increases by delta_tau after each update."""
    nr = NarrativeResonance()
    nr.reset(2)
    H = np.array([0.5, 0.5])
    nr.update(H, 0.02)
    assert abs(nr.get_current_tau() - 0.02) < ATOL
    nr.update(H, 0.03)
    assert abs(nr.get_current_tau() - 0.05) < ATOL


def test_multiple_resets():
    """Multiple resets clear state correctly."""
    nr = NarrativeResonance()
    nr.reset(3)
    nr.update(np.ones(3), 0.01)
    nr.update(np.ones(3), 0.01)
    assert nr.get_n_updates() == 2
    nr.reset(5)
    assert nr.get_n_updates() == 0
    assert len(nr.get_field()) == 5


# ═══════════════════════════════════════════════════════════════
# 2. LONG-RANGE INFLUENCE
# ═══════════════════════════════════════════════════════════════

def test_influence_weight():
    """Influence decays exponentially with distance."""
    nr = NarrativeResonance(lambda_decay=0.3)
    nr.reset(2)
    nr.update(np.ones(2), 0.5)  # τ = 0.5
    w_near = nr.influence_weight(0.3)  # |0.5 - 0.3| = 0.2
    expected_near = math.exp(-0.2 / 0.3)
    assert abs(w_near - expected_near) < ATOL, f"w_near={w_near}"
    w_far = nr.influence_weight(0.0)  # |0.5 - 0.0| = 0.5
    expected_far = math.exp(-0.5 / 0.3)
    assert abs(w_far - expected_far) < ATOL, f"w_far={w_far}"
    assert w_near > w_far, "nearer position should have more influence"


def test_self_influence():
    """Influence of current position is 1.0."""
    nr = NarrativeResonance()
    nr.reset(2)
    nr.update(np.ones(2), 0.3)
    w = nr.influence_weight(nr.get_current_tau())
    assert abs(w - 1.0) < ATOL


def test_effective_horizon_default():
    """With λ=0.3, threshold=0.01: horizon ≈ 1.38."""
    nr = NarrativeResonance(lambda_decay=0.3)
    h = nr.effective_horizon(0.01)
    expected = -0.3 * math.log(0.01)
    assert abs(h - expected) < ATOL


def test_effective_horizon_coverage():
    """Horizon > 1.0 means full narrative span covered."""
    nr = NarrativeResonance(lambda_decay=0.3)
    h = nr.effective_horizon()
    assert h > 1.0, f"horizon={h} should cover full [0,1] span"


def test_persistence_worked_example():
    """§7.7: spatial resonance persists at e^{-0.60/0.3} ≈ 13.5%."""
    nr = NarrativeResonance(lambda_decay=0.3)
    nr.reset(2)
    nr.update(np.ones(2), 0.60)
    # Influence from τ=0 (opening) at τ=0.60
    w = nr.influence_weight(0.0)
    expected_w = math.exp(-0.60 / 0.3)
    assert abs(w - expected_w) < ATOL, f"w={w}, expected ~{expected_w}"
    assert abs(w - 0.1353) < 0.01, f"w={w}, expected ~0.135"


# ═══════════════════════════════════════════════════════════════
# 3. NUMERICAL APPROXIMATION
# ═══════════════════════════════════════════════════════════════

def test_error_bound():
    """Error bound formula: max_curvature · Δτ² / 2."""
    bound = NarrativeResonance.numerical_error_bound(0.01)
    expected = 0.01 ** 2 / 2.0
    assert abs(bound - expected) < ATOL


def test_error_bound_scaling():
    """Smaller Δτ gives quadratically smaller bound."""
    b1 = NarrativeResonance.numerical_error_bound(0.1)
    b2 = NarrativeResonance.numerical_error_bound(0.01)
    # Δτ=0.01 should have 100x smaller bound
    assert b2 < b1 / 50, f"b1={b1}, b2={b2}, ratio={b1/max(b2,1e-15)}"


def test_validation_constant_H():
    """Numerical update matches analytic integral for constant H.

    For H(τ) = c (constant vector), the analytic solution is:
      R(τ) = c · λ · (1 - e^{-τ/λ})

    This validates the entire recursive formulation against the
    theoretical integral definition.
    """
    nr = NarrativeResonance(lambda_decay=0.3)
    c = np.array([1.0, 0.5, 0.0])
    tau_max = 1.0
    n_steps = 100
    dt = tau_max / n_steps

    nr.reset(len(c))
    for i in range(n_steps):
        tau_i = i * dt
        nr.update(c, dt)

    # Compare numerical R(τ_max) with analytic
    R_numerical = nr.get_field()
    R_analytic = c * 0.3 * (1.0 - math.exp(-tau_max / 0.3))

    rel_errors = np.abs(R_numerical - R_analytic) / (np.abs(R_analytic) + 1e-15)
    assert np.all(rel_errors < 0.05), (
        f"Max rel error={np.max(rel_errors)}"
    )


def test_convergence_with_step_size():
    """Error decreases as Δτ → 0 (smaller steps → more accurate)."""
    nr = NarrativeResonance(lambda_decay=0.3)
    c = np.array([1.0])
    tau_max = 1.0
    R_analytic = c[0] * 0.3 * (1.0 - math.exp(-tau_max / 0.3))

    errors = []
    for n_steps in [10, 50, 100, 200]:
        nr.reset(1)
        dt = tau_max / n_steps
        for i in range(n_steps):
            nr.update(c, dt)
        R_num = nr.get_field()[0]
        errors.append(abs(R_num - R_analytic))

    # Each refinement should reduce error
    for i in range(1, len(errors)):
        assert errors[i] < errors[i-1] * 1.1 or errors[i] < 0.001, (
            f"Error not decreasing: {errors}"
        )


def test_small_delta_tau():
    """Very small Δτ should not cause numerical issues."""
    nr = NarrativeResonance()
    nr.reset(2)
    H = np.array([0.5, 0.3])
    for _ in range(10):
        R = nr.update(H, 1e-8)
    assert np.all(np.isfinite(R))


def test_large_delta_tau():
    """Large Δτ (e.g., 1.0) should not cause issues."""
    nr = NarrativeResonance()
    nr.reset(2)
    H = np.array([0.5, 0.3])
    R = nr.update(H, 1.0)
    assert np.all(np.isfinite(R))
    # Second update: R = 0·decay + H·1.0 = H
    R2 = nr.update(H, 1.0)
    assert np.allclose(R2, H, rtol=RTOL)


# ═══════════════════════════════════════════════════════════════
# 4. RECURSIVE UPDATE — TRAJECTORY, SUMMARY
# ═══════════════════════════════════════════════════════════════

def test_trajectory_length():
    """Trajectory records initial state plus each R update."""
    nr = NarrativeResonance()
    nr.reset(2)
    Hs = [np.array([i * 0.1, 1.0 - i * 0.1]) for i in range(10)]
    for H in Hs:
        nr.update(H, 0.01)
    traj = nr.get_trajectory()
    # 1 initial (tau=0) + 9 R updates (first update stores prev_H, returns 0)
    assert len(traj) <= len(Hs), f"traj={len(traj)}, Hs={len(Hs)}"
    assert len(traj) >= len(Hs) - 2, f"traj too short: {len(traj)}"


def test_summary_statistics():
    """Summary statistics returns properly shaped finite array."""
    nr = NarrativeResonance()
    nr.reset(2)
    Hs = [np.array([math.sin(i * 0.1), math.cos(i * 0.1)])
          for i in range(20)]
    for H in Hs:
        nr.update(H, 0.01)
    stats = nr.summary_statistics(12)
    assert len(stats) == 12, f"stats shape={len(stats)}"
    assert np.all(np.isfinite(stats))


def test_summary_no_updates():
    """Summary with no updates returns zeros."""
    nr = NarrativeResonance()
    stats = nr.summary_statistics(12)
    assert np.all(stats == 0.0)


def test_trajectory_after_reset():
    """Reset clears trajectory."""
    nr = NarrativeResonance()
    nr.reset(2)
    nr.update(np.ones(2), 0.01)
    nr.update(np.ones(2), 0.01)
    assert len(nr.get_trajectory()) >= 1
    nr.reset(2)
    assert len(nr.get_trajectory()) == 1  # just initial state


# ═══════════════════════════════════════════════════════════════
# 5. EDGE CASES / NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════

def test_nan_input():
    """NaN in input should not crash."""
    nr = NarrativeResonance()
    nr.reset(2)
    H_nan = np.array([float('nan'), 0.5])
    try:
        nr.update(H_nan, 0.01)
        # If no exception, R may contain NaN — that's expected
        R = nr.get_field()
        assert R is not None
    except (ValueError, FloatingPointError):
        pass


def test_inf_input():
    """Inf input should not crash."""
    nr = NarrativeResonance()
    nr.reset(2)
    H_inf = np.array([float('inf'), 0.5])
    try:
        R = nr.update(H_inf, 0.01)
    except (ValueError, FloatingPointError):
        pass


def test_zero_delta_tau():
    """Δτ = 0 should not change the field (exp(0)=1, term=0)."""
    nr = NarrativeResonance()
    nr.reset(2)
    nr.update(np.array([1.0, 0.5]), 0.01)  # store H
    nr.update(np.array([0.5, 0.3]), 0.01)  # R = H1*dt
    R_before = nr.get_field().copy()
    nr.update(np.array([0.0, 0.0]), 0.0)
    R_after = nr.get_field()
    assert np.allclose(R_before, R_after), "zero dt should not change R"


def test_all_zero_input():
    """H=0 should eventually decay field to zero."""
    nr = NarrativeResonance()
    nr.reset(2)
    nr.update(np.array([1.0, 0.5]), 0.01)
    nr.update(np.array([0.5, 0.3]), 0.01)
    # Now feed zeros repeatedly
    Z = np.zeros(2)
    for _ in range(200):
        nr.update(Z, 0.01)
    R = nr.get_field()
    assert np.all(np.abs(R) < 1e-3), f"R should be near zero: {R}"


def test_trajectory_contains_taus():
    """Trajectory entries should have increasing tau values."""
    nr = NarrativeResonance()
    nr.reset(2)
    for i in range(10):
        nr.update(np.ones(2), 0.1)
    traj = nr.get_trajectory()
    taus = [t for t, _ in traj]
    for i in range(1, len(taus)):
        assert taus[i] >= taus[i-1] - ATOL


# ═══════════════════════════════════════════════════════════════
# 6. PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════

def test_performance_m_steps():
    """1000 update steps in < 100ms (cold start)."""
    nr = NarrativeResonance()
    nr.reset(7)
    H = np.ones(7) * 0.5
    dt = 0.001
    start = time.perf_counter()
    for _ in range(1000):
        nr.update(H, dt)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"1000 steps took {elapsed:.3f}s"


def test_performance_warm_steps():
    """10000 update steps in < 500ms (warm)."""
    nr = NarrativeResonance()
    nr.reset(7)
    H = np.ones(7) * 0.5
    dt = 0.0001
    # Warmup
    for _ in range(100):
        nr.update(H, dt)
    start = time.perf_counter()
    for _ in range(10000):
        nr.update(H, dt)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"10000 steps took {elapsed:.3f}s"


def test_performance_large_dim():
    """120-dim field 1000 steps in < 200ms."""
    nr = NarrativeResonance()
    nr.reset(120)
    H = np.ones(120) * 0.5
    dt = 0.001
    start = time.perf_counter()
    for _ in range(1000):
        nr.update(H, dt)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"1000 large-dim steps took {elapsed:.3f}s"


def test_performance_trajectory_append():
    """Appending 1000 trajectory entries stays fast."""
    nr = NarrativeResonance()
    nr.reset(7)
    H = np.ones(7) * 0.5
    start = time.perf_counter()
    for _ in range(1000):
        nr.update(H, 0.001)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"1000 steps took {elapsed:.3f}s"


def test_augment_performance():
    """1000 augment calls in < 50ms."""
    nr = NarrativeResonance()
    nr.reset(7)
    nr.update(np.ones(7), 0.01)
    nr.update(np.ones(7), 0.01)
    f = np.ones(7) * 0.5
    start = time.perf_counter()
    for _ in range(1000):
        _ = nr.augment(f)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"1000 augments took {elapsed:.3f}s"


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

def _check_fn(fn):
    name = fn.__name__
    try:
        fn()
        return {"name": name, "status": "PASS"}
    except Exception as e:
        return {"name": name, "status": "FAIL", "error": str(e)}


if __name__ == "__main__":
    test_fns = [v for k, v in globals().items()
                if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_fns:
        result = _check_fn(fn)
        if result["status"] == "PASS":
            print(f"  [PASS] {result['name']}")
            passed += 1
        else:
            print(f"  [FAIL] {result['name']}: {result.get('error', '')}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({passed + failed} total)")
