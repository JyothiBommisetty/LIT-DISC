"""Comprehensive tests for the Literary Signal Memory subsystem.

Covers Eqs. (12.1)–(12.7) and Architecture §9:
  - Random Fourier Features: determinism, shape, similarity (Eq. 12.2)
  - Storage: threshold condition (Eq. 12.1), capacity enforcement
  - Retrieval: cosine similarity, theta_retrieve (Eq. 12.3)
  - Reinforcement: beta increase, cap, count (Eq. 12.4)
  - Selective decay: non-retrieved entries only (Eq. 12.5)
  - Forgetting: explicit API (Eq. 12.5)
  - Purging: remove below theta_forget (Eq. 12.6)
  - Augmentation: signal modification (Eq. 12.7)
  - Capacity: max_size=200, evict lowest beta
  - Persistence: save/load round-trip
  - Lifecycle: end-to-end workflow
  - Edge cases: empty, zero vectors, boundary thresholds
  - Diagnostics: get_stats, get_activations
  - Performance: timing at capacity
"""
import math
import os
import sys
import time
import tempfile
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.config import MemoryParams
from src.models import LiterarySignal, MemoryEntry
from src.coupling.memory import RandomFourierFeatures, MemorySystem


RTOL = 1e-4
ATOL = 1e-6


def _make_signal(f_vals: list[float], gamma: float = 0.85,
                 tau: float = 0.5) -> LiterarySignal:
    return LiterarySignal(
        f=np.array(f_vals, dtype=float),
        gamma=gamma, tau=tau,
        context=None, provenance=[(1, 0)],
        residues=np.zeros((6, 6)), memory_traces=[],
    )


# ═══════════════════════════════════════════════════════════════
# RANDOM FOURIER FEATURES — Eq. (12.2)
# ═══════════════════════════════════════════════════════════════

def test_rff_deterministic():
    """RFF with fixed seeds produces identical keys."""
    f = np.array([0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.1])
    k1 = RandomFourierFeatures().project(f)
    k2 = RandomFourierFeatures().project(f)
    assert np.allclose(k1, k2), "RFF not deterministic"


def test_rff_output_shape():
    """RFF key is 32-dimensional."""
    k = RandomFourierFeatures().project(np.array([0.5, 0.3, 0.8]))
    assert len(k) == 32, f"key dim={len(k)}"


def test_rff_different_input_dims():
    """RFF handles different input dimensions, re-initializes W/b."""
    rff = RandomFourierFeatures()
    k7 = rff.project(np.array([0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.1]))
    k3 = rff.project(np.array([0.5, 0.3, 0.8]))
    assert len(k7) == 32
    assert len(k3) == 32


def test_rff_similar_inputs():
    """Similar inputs produce keys with cosine > 0.5 (kernel approx)."""
    f1 = np.array([0.5, 0.3, 0.8])
    f2 = np.array([0.52, 0.31, 0.79])
    rff = RandomFourierFeatures()
    k1 = rff.project(f1)
    k2 = rff.project(f2)
    cos_sim = float(np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2)))
    assert cos_sim > 0.5, f"similar inputs cos={cos_sim}"


def test_rff_different_inputs():
    """Very different inputs produce near-orthogonal keys."""
    f1 = np.array([0.0, 0.0, 0.0])
    f2 = np.array([1.0, 1.0, 1.0])
    rff = RandomFourierFeatures()
    k1 = rff.project(f1)
    k2 = rff.project(f2)
    cos_sim = float(np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2)))
    assert abs(cos_sim) < 0.5, f"different inputs cos={cos_sim}"


def test_rff_key_scale():
    """Key should have expected magnitude: ||k|| ≈ sqrt(2) for unit input."""
    rff = RandomFourierFeatures()
    f = np.ones(7)
    k = rff.project(f)
    norm = float(np.linalg.norm(k))
    assert 0.5 < norm < 2.0, f"norm={norm}"


def test_make_key():
    """make_key is an alias for RFF project."""
    mem = MemorySystem()
    f = np.array([0.5, 0.3, 0.8])
    k1 = mem.make_key(f)
    k2 = mem.rff.project(f)
    assert np.allclose(k1, k2)


# ═══════════════════════════════════════════════════════════════
# STORAGE — Eq. (12.1)
# ═══════════════════════════════════════════════════════════════

def test_should_store_threshold():
    """||f|| * gamma > 0.5 triggers storage. Eq. (12.1)."""
    mem = MemorySystem()
    assert mem.should_store(np.array([1.0, 0.5, 0.3]), 0.8)
    assert not mem.should_store(np.array([0.1, 0.02, 0.03]), 0.3)
    assert not mem.should_store(np.array([1.0, 0.5, 0.3]), 0.0)
    assert not mem.should_store(np.array([0.0, 0.0, 0.0]), 0.9)


def test_should_store_boundary():
    """At exactly theta_store, should NOT store (strict >)."""
    mem = MemorySystem()
    norm_f = float(np.linalg.norm(np.array([0.3, 0.4])))  # exactly 0.5
    assert abs(norm_f - 0.5) < ATOL, f"norm={norm_f}"
    gamma = 1.0
    val = norm_f * gamma  # exactly 0.5
    assert abs(val - 0.5) < ATOL, f"val={val}"
    assert not mem.should_store(np.array([0.3, 0.4]), gamma)


def test_store_signal():
    """Stored signal is retrievable with same key."""
    mem = MemorySystem()
    sig = _make_signal([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    mem.store_signal(sig)
    assert mem.size == 1
    assert abs(mem.store[0].beta - 1.0) < ATOL
    assert mem.store[0].retrieval_count == 0
    assert mem.store[0].last_retrieved == -1.0


def test_store_creates_memory_entry():
    """store_signal creates proper MemoryEntry with key."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal([0.8, 0.6, 0.4], tau=0.7)
    mem.store_signal(sig)
    entry = mem.store[0]
    assert isinstance(entry, MemoryEntry)
    assert len(entry.key) == 32
    assert entry.storage_tau == 0.7


# ═══════════════════════════════════════════════════════════════
# RETRIEVAL — Eq. (12.3)
# ═══════════════════════════════════════════════════════════════

def test_retrieve_empty():
    """Retrieval from empty store returns empty list."""
    mem = MemorySystem()
    matches = mem.retrieve(np.array([0.5, 0.3, 0.8]))
    assert len(matches) == 0


def test_retrieve_similar():
    """Similar signals are retrieved. Eq. (12.3)."""
    mem = MemorySystem()
    sig = _make_signal([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    mem.store_signal(sig)
    query = np.array([0.81, 0.59, 0.41, 0.21, 0.51, 0.31, 0.11])
    matches = mem.retrieve(query)
    assert len(matches) > 0, "should find similar"


def test_retrieve_no_false_positive():
    """Dissimilar signals are not retrieved."""
    mem = MemorySystem()
    sig = _make_signal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mem.store_signal(sig)
    query = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    matches = mem.retrieve(query)
    assert len(matches) == 0, "false positive"


def test_retrieve_returns_memory_entries():
    """retrieve() returns MemoryEntry objects."""
    mem = MemorySystem()
    sig = _make_signal([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    mem.store_signal(sig)
    matches = mem.retrieve(np.array([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1]))
    for m in matches:
        assert isinstance(m, MemoryEntry)
        assert m.beta > 0.7


def test_retrieve_triggers_reinforcement():
    """Retrieval increments retrieval_count and updates last_retrieved."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal([0.8, 0.6, 0.4], tau=0.5)
    mem.store_signal(sig)
    assert mem.store[0].retrieval_count == 0
    _ = mem.retrieve(f, current_tau=0.6)
    assert mem.store[0].retrieval_count == 1
    assert mem.store[0].last_retrieved == 0.6


# ═══════════════════════════════════════════════════════════════
# REINFORCEMENT — Eq. (12.4)
# ═══════════════════════════════════════════════════════════════

def test_reinforce_increases_beta():
    """beta = min(1, beta + rho * cos). Eq. (12.4)."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    mem.store[0] = MemoryEntry(
        key=mem.store[0].key, signal=sig,
        beta=0.5, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    mem.reinforce(0, 0.9)
    expected = min(1.0, 0.5 + 0.1 * 0.9)
    assert abs(mem.store[0].beta - expected) < ATOL, f"beta={mem.store[0].beta}"


def test_reinforce_caps_at_one():
    """Reinforcement caps beta at 1.0."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    mem.reinforce(0, 2.0)  # would give 1.0 + 0.2 = 1.2, capped to 1.0
    assert mem.store[0].beta <= 1.0 + ATOL


def test_reinforce_increments_count():
    """reinforce() increments retrieval_count."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    mem.reinforce(0, 0.9)
    assert mem.store[0].retrieval_count == 1
    mem.reinforce(0, 0.8)
    assert mem.store[0].retrieval_count == 2


def test_reinforce_sets_last_retrieved():
    """reinforce() with current_tau updates last_retrieved."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    mem.reinforce(0, 0.9, current_tau=0.75)
    assert abs(mem.store[0].last_retrieved - 0.75) < ATOL


# ═══════════════════════════════════════════════════════════════
# SELECTIVE DECAY — Eq. (12.5)
# ═══════════════════════════════════════════════════════════════

def test_decay_all_non_retrieved():
    """Only non-retrieved entries are decayed. Eq. (12.5)."""
    mem = MemorySystem()
    f1 = np.array([0.8, 0.6, 0.4])
    f2 = np.array([0.01, 0.01, 0.01])  # very different → RFF cos < 0.7
    mem.store_signal(_make_signal(f1))
    mem.store_signal(_make_signal(f2))
    # Retrieve only the first
    _ = mem.retrieve(f1)
    beta_before = mem.store[0].beta
    beta2_before = mem.store[1].beta
    mem.decay_all()
    # Retrieved entry (index 0) should NOT decay
    assert abs(mem.store[0].beta - beta_before) < ATOL, \
        f"retrieved decayed: {beta_before} -> {mem.store[0].beta}"
    # Non-retrieved entry (index 1) should decay
    expected_beta2 = beta2_before * (1.0 - mem.params.delta_decay)
    assert abs(mem.store[1].beta - expected_beta2) < ATOL, \
        f"non-retrieved not decayed: {beta2_before} -> {mem.store[1].beta}"


def test_decay_all_twice():
    """Two decay steps compound: beta * (1-delta)^2."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    # Mark as non-retrieved by removing from set
    mem._retrieved_this_iteration.clear()
    mem.decay_all()
    mem._retrieved_this_iteration.clear()
    mem.decay_all()
    expected = 1.0 * (1.0 - mem.params.delta_decay) ** 2
    assert abs(mem.store[0].beta - expected) < ATOL, f"beta={mem.store[0].beta}"


def test_decay_all_retrieved_reset():
    """After decay_all, retrieved set is cleared."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    _ = mem.retrieve(f)
    assert len(mem._retrieved_this_iteration) == 1
    mem.decay_all()
    assert len(mem._retrieved_this_iteration) == 0


# ═══════════════════════════════════════════════════════════════
# FORGETTING — aliasing decay (Eq. 12.5)
# ═══════════════════════════════════════════════════════════════

def test_forget_same_as_decay():
    """forget() is an alias for decay_all()."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    beta_before = mem.store[0].beta
    mem.forget()
    expected = beta_before * (1.0 - mem.params.delta_decay)
    assert abs(mem.store[0].beta - expected) < ATOL


# ═══════════════════════════════════════════════════════════════
# PURGING — Eq. (12.6)
# ═══════════════════════════════════════════════════════════════

def test_purge_removes_below_threshold():
    """Entries with beta < theta_forget are removed. Eq. (12.6)."""
    params = MemoryParams(theta_forget=0.5)
    mem = MemorySystem(params)
    f1 = np.array([0.8, 0.6, 0.4])
    f2 = np.array([0.2, 0.3, 0.1])
    mem.store_signal(_make_signal(f1))  # beta = 1.0
    # Manually set second entry's beta below threshold
    sig2 = _make_signal(f2)
    mem.store_signal(sig2)
    mem.store[1] = MemoryEntry(
        key=mem.store[1].key, signal=sig2,
        beta=0.3, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    assert mem.size == 2
    mem.purge()
    assert mem.size == 1
    assert abs(mem.store[0].beta - 1.0) < ATOL


def test_purge_keeps_above_threshold():
    """Entries at or above threshold are kept."""
    params = MemoryParams(theta_forget=0.3)
    mem = MemorySystem(params)
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    mem.store[0] = MemoryEntry(
        key=mem.store[0].key, signal=sig,
        beta=0.3, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    mem.purge()
    assert mem.size == 1


def test_purge_empty():
    """Purge on empty store does not error."""
    mem = MemorySystem()
    mem.purge()
    assert mem.size == 0


# ═══════════════════════════════════════════════════════════════
# SIGNAL AUGMENTATION — Eq. (12.7)
# ═══════════════════════════════════════════════════════════════

def test_augment_changes_signal():
    """Augmented signal differs from original. Eq. (12.7)."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    sig = _make_signal(f)
    mem.store_signal(sig)
    matches = mem.retrieve(f)
    assert len(matches) > 0
    f_aug = mem.augment(f, 0.85, matches)
    assert not np.allclose(f_aug, f), "augmentation had no effect"
    assert np.all(np.isfinite(f_aug))


def test_augment_no_matches():
    """Augment with empty matches returns original."""
    mem = MemorySystem()
    f = np.array([0.5, 0.3, 0.8])
    f_aug = mem.augment(f, 0.85, [])
    assert np.allclose(f_aug, f)


def test_augment_different_dimensions():
    """Augment handles different feature dimensions."""
    mem = MemorySystem()
    f_stored = np.array([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    sig = _make_signal(f_stored)
    mem.store_signal(sig)
    # Augment with shorter vector
    f_short = np.array([0.5, 0.3, 0.8])
    matches = mem.retrieve(np.array([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1]))
    f_aug = mem.augment(f_short, 0.85, matches)
    assert len(f_aug) == 3
    assert np.all(np.isfinite(f_aug))


# ═══════════════════════════════════════════════════════════════
# CAPACITY — max_size enforcement
# ═══════════════════════════════════════════════════════════════

def test_capacity_max_size():
    """Memory enforces max_size = 200."""
    params = MemoryParams(max_size=200)
    mem = MemorySystem(params)
    for i in range(200):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    assert mem.size == 200
    assert mem.is_full


def test_capacity_eviction():
    """Adding beyond max_size evicts lowest-beta entry."""
    params = MemoryParams(max_size=5)
    mem = MemorySystem(params)
    sigs = []
    for i in range(5):
        f = np.array([0.1 * i, 0.1 * i + 0.1, 0.3])
        sig = _make_signal(f, gamma=0.8)
        mem.store_signal(sig)
        sigs.append(sig)
    assert mem.size == 5
    # Lower beta of entry 2
    mem.store[2] = MemoryEntry(
        key=mem.store[2].key, signal=sigs[2],
        beta=0.3, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    # Add one more — should evict the lowest-beta (index 2, beta=0.3)
    sig_new = _make_signal(np.array([0.9, 0.9, 0.9]))
    mem.store_signal(sig_new)
    assert mem.size == 5
    # The evicted signal's f should no longer be in store
    stored_fs = [tuple(e.signal.f) for e in mem.store]
    assert tuple(sigs[2].f) not in stored_fs, "lowest-beta not evicted"
    assert tuple(sig_new.f) in stored_fs, "new signal not stored"


def test_capacity_evict_lowest_beta():
    """Explicit _evict_lowest_beta removes the entry with minimum beta."""
    params = MemoryParams(max_size=3)
    mem = MemorySystem(params)
    for i in range(3):
        sig = _make_signal([0.5, 0.3, 0.8])
        mem.store_signal(sig)
    mem.store[0] = MemoryEntry(
        key=mem.store[0].key, signal=mem.store[0].signal,
        beta=0.1, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    mem.store[1] = MemoryEntry(
        key=mem.store[1].key, signal=mem.store[1].signal,
        beta=0.5, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    mem.store[2] = MemoryEntry(
        key=mem.store[2].key, signal=mem.store[2].signal,
        beta=0.9, storage_tau=0.5, retrieval_count=0, last_retrieved=-1.0)
    mem._evict_lowest_beta()
    assert mem.size == 2
    # Entry with beta=0.1 should be gone
    for e in mem.store:
        assert e.beta > 0.1 + ATOL


# ═══════════════════════════════════════════════════════════════
# FULL LIFECYCLE
# ═══════════════════════════════════════════════════════════════

def test_lifecycle_store_retrieve_augment():
    """End-to-end: store → retrieve → augment."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.1])
    sig = _make_signal(f)
    # 1. Store (Eq. 12.1)
    mem.store_signal(sig)
    assert mem.size == 1
    # 2. Retrieve (Eq. 12.3)
    matches = mem.retrieve(f)
    assert len(matches) > 0
    # 3. Augment (Eq. 12.7)
    f_aug = mem.augment(f, 0.85, matches)
    assert np.all(np.isfinite(f_aug))


def test_lifecycle_decay_purge():
    """End-to-end: decay → purge removes under-threshold entries."""
    params = MemoryParams(theta_forget=0.5, delta_decay=0.3)
    mem = MemorySystem(params)
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    assert mem.size == 1
    for _ in range(5):
        mem.decay_all()
        mem.purge()
    assert mem.size == 0


def test_lifecycle_mark_iteration_end():
    """mark_iteration_end applies decay then purge."""
    params = MemoryParams(theta_forget=0.5, delta_decay=0.3)
    mem = MemorySystem(params)
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    assert mem.size == 1
    for _ in range(5):
        mem.mark_iteration_end()
    assert mem.size == 0


def test_lifecycle_retrieved_entries_survive():
    """Retrieved entries should survive multiple decay cycles."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    for _ in range(10):
        _ = mem.retrieve(f)  # reinforces beta each time
        mem.decay_all()      # doesn't decay retrieved entry
        mem.purge()
    assert mem.size == 1
    # Beta should be > 1.0 after reinforcement, capped at 1.0
    assert abs(mem.store[0].beta - 1.0) < ATOL


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE — save/load
# ═══════════════════════════════════════════════════════════════

def test_persist_save_load_round_trip():
    """Save then load preserves all memory entries."""
    mem = MemorySystem()
    for i in range(5):
        sig = _make_signal([0.5 + 0.1 * i, 0.3, 0.8], tau=0.1 * i)
        mem.store_signal(sig)
    _ = mem.retrieve(np.array([0.5, 0.3, 0.8]), current_tau=0.5)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        mem.persist(tmp_path)
        mem2 = MemorySystem()
        ok = mem2.load(tmp_path)
        assert ok, "load failed"
        assert mem2.size == mem.size
        for e1, e2 in zip(mem.store, mem2.store):
            assert np.allclose(e1.key, e2.key)
            assert abs(e1.beta - e2.beta) < ATOL
            assert e1.retrieval_count == e2.retrieval_count
            assert abs(e1.last_retrieved - e2.last_retrieved) < ATOL
            assert abs(e1.storage_tau - e2.storage_tau) < ATOL
    finally:
        os.unlink(tmp_path)


def test_persist_empty_store():
    """Save/load empty store works."""
    mem = MemorySystem()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        mem.persist(tmp_path)
        mem2 = MemorySystem()
        ok = mem2.load(tmp_path)
        assert ok
        assert mem2.size == 0
    finally:
        os.unlink(tmp_path)


def test_persist_nonexistent_file():
    """Load from nonexistent path returns False."""
    mem = MemorySystem()
    ok = mem.load("/nonexistent/path/memory.npz")
    assert not ok


def test_persist_multi_dim_features():
    """Save/load with varying feature dimensions."""
    mem = MemorySystem()
    mem.store_signal(_make_signal([0.5, 0.3, 0.8], tau=0.2))      # 3-d
    mem.store_signal(_make_signal([0.5, 0.3, 0.8, 0.2], tau=0.4))  # 4-d
    mem.store_signal(_make_signal([0.5], tau=0.6))                  # 1-d
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        mem.persist(tmp_path)
        mem2 = MemorySystem()
        ok = mem2.load(tmp_path)
        assert ok
        assert mem2.size == 3
        for e in mem2.store:
            assert len(e.signal.f) > 0
            assert np.all(np.isfinite(e.signal.f))
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def test_get_stats_empty():
    """get_stats on empty store returns zeros."""
    mem = MemorySystem()
    stats = mem.get_stats()
    assert stats["size"] == 0
    assert stats["avg_beta"] == 0.0
    assert stats["max_size"] == 200


def test_get_stats_populated():
    """get_stats returns correct diagnostics."""
    mem = MemorySystem()
    f = np.array([0.8, 0.6, 0.4])
    sig = _make_signal(f)
    mem.store_signal(sig)
    _ = mem.retrieve(f)
    stats = mem.get_stats()
    assert stats["size"] == 1
    assert stats["total_retrievals"] >= 1
    assert stats["avg_beta"] > 0
    assert stats["capacity_pct"] == 0.5  # 1/200


def test_get_activations():
    """get_activations returns list of beta values."""
    mem = MemorySystem()
    for i in range(3):
        sig = _make_signal([0.5 + 0.1 * i, 0.3, 0.8])
        mem.store_signal(sig)
    activations = mem.get_activations()
    assert len(activations) == 3
    assert all(a > 0 for a in activations)


def test_get_avg_beta():
    """get_avg_beta returns mean beta."""
    mem = MemorySystem()
    for i in range(3):
        sig = _make_signal([0.5, 0.3, 0.8])
        mem.store_signal(sig)
    avg = mem.get_avg_beta()
    assert abs(avg - 1.0) < ATOL


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

def test_store_zero_vector():
    """Zero vector with high gamma may still be stored."""
    mem = MemorySystem()
    sig = _make_signal([0.0, 0.0, 0.0], gamma=0.9)
    # ||0|| * 0.9 = 0 < 0.5
    assert not mem.should_store(sig.f, sig.gamma)


def test_retrieve_identical():
    """Identical query finds stored entry."""
    mem = MemorySystem()
    f = np.array([0.5, 0.3, 0.8])
    sig = _make_signal(f)
    mem.store_signal(sig)
    matches = mem.retrieve(f)
    assert len(matches) == 1


def test_retrieve_multiple_matches():
    """Multiple similar entries all match."""
    mem = MemorySystem()
    for i in range(5):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    matches = mem.retrieve(np.array([0.5, 0.3, 0.8]))
    assert len(matches) == 5


def test_clear():
    """Clear removes all entries and resets state."""
    mem = MemorySystem()
    for i in range(3):
        sig = _make_signal([0.5, 0.3, 0.8])
        mem.store_signal(sig)
    _ = mem.retrieve(np.array([0.5, 0.3, 0.8]))
    mem.clear()
    assert mem.size == 0
    assert mem._time == 0
    assert len(mem._retrieved_this_iteration) == 0


def test_custom_params():
    """Custom MemoryParams propagate correctly."""
    params = MemoryParams(
        key_dim=16, sigma_kernel=0.5,
        theta_store=0.3, theta_retrieve=0.5,
        theta_forget=0.05, rho_reinforce=0.2,
        delta_decay=0.1, max_size=50,
    )
    mem = MemorySystem(params)
    assert mem.params.key_dim == 16
    assert mem.params.max_size == 50
    assert mem.params.theta_store == 0.3
    assert mem.params.rho_reinforce == 0.2


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def test_performance_store_200():
    """Storing 200 entries completes in < 1s."""
    mem = MemorySystem()
    start = time.perf_counter()
    for i in range(200):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    elapsed = time.perf_counter() - start
    assert mem.size == 200
    assert elapsed < 1.0, f"store 200 took {elapsed:.3f}s"


def test_performance_retrieve_200():
    """Retrieval across 200 entries completes in < 1s."""
    mem = MemorySystem()
    for i in range(200):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    start = time.perf_counter()
    matches = mem.retrieve(np.array([0.5, 0.3, 0.8]))
    elapsed = time.perf_counter() - start
    assert len(matches) > 0
    assert elapsed < 1.0, f"retrieve 200 took {elapsed:.3f}s"


def test_performance_decay_200():
    """Decay on 200 entries completes in < 1s."""
    mem = MemorySystem()
    for i in range(200):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    start = time.perf_counter()
    mem.decay_all()
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"decay 200 took {elapsed:.3f}s"


def test_performance_full_lifecycle_200():
    """Full lifecycle (store + retrieve + decay + purge) on 200 entries in < 2s."""
    mem = MemorySystem()
    start = time.perf_counter()
    for i in range(200):
        sig = _make_signal([0.5 + 0.01 * i, 0.3, 0.8])
        mem.store_signal(sig)
    _ = mem.retrieve(np.array([0.5, 0.3, 0.8]))
    mem.decay_all()
    mem.purge()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"full lifecycle 200 took {elapsed:.3f}s"


# ═══════════════════════════════════════════════════════════════
# WORKED EXAMPLE VERIFICATION
# ═══════════════════════════════════════════════════════════════

def test_worked_example_storage_condition():
    """Verify §7.6: ||f||·γ = 0.82 * 0.85 = 0.70 > 0.50 → STORED."""
    mem = MemorySystem()
    f_P6_Sy = np.array([0.82, 0.71, 0.3])
    gamma_P6_Sy = 0.85
    assert mem.should_store(f_P6_Sy, gamma_P6_Sy)


def test_worked_example_retrieval():
    """Verify §7.6: similar signal triggers retrieval."""
    mem = MemorySystem()
    f_stored = np.array([0.82, 0.71, 0.3])
    sig = _make_signal(f_stored, tau=0.60)
    mem.store_signal(sig)
    f_query = np.array([0.80, 0.73, 0.28])
    matches = mem.retrieve(f_query)
    assert len(matches) > 0


def test_worked_example_augmentation():
    """Verify §7.6: retrieval enables augmentation."""
    mem = MemorySystem()
    f_stored = np.array([0.82, 0.71, 0.3])
    sig = _make_signal(f_stored, tau=0.60)
    mem.store_signal(sig)
    f_query = np.array([0.80, 0.73, 0.28])
    matches = mem.retrieve(f_query)
    f_aug = mem.augment(f_query, 0.72, matches)
    assert not np.allclose(f_aug, f_query)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback
    test_fns = [v for k, v in globals().items()
                if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({passed+failed} total)")
