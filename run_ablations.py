"""LIT-DISC v1.1 -- Ablation Study (Stage 3).

Runs the full pipeline 5 additional times on 25 passages, each with
one HMP-DT component disabled via runtime monkey-patching.

Ablations:
  1. Without Memory        (delta=0, memory store/retrieve disabled)
  2. Without Resonance     (gamma=0, resonance update disabled)
  3. Without Adaptive Coupling (beta=0, coupling kernel zeroed)
  4. Without Uncertainty   (Gamma=1.0, confidence override)
  5. Without Narrative Signals (N feature extractor disabled)

DOES NOT modify any source file in src/.

Outputs: ablation_results.csv, baseline_summary.json (updated)
"""
import json, csv, math, os, sys, time, warnings
from collections import OrderedDict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Load gold-standard results for comparison ──
with open("data/gold_standard/litdisc_results.json", encoding="utf-8") as f:
    gold = json.load(f)

# Map passage_id to passage file
PASSAGE_DIR = "data/gold_standard/passages"
passage_files = {}
for fname in os.listdir(PASSAGE_DIR):
    for p in gold:
        if fname.startswith(p["passage_id"]):
            passage_files[p["passage_id"]] = os.path.join(PASSAGE_DIR, fname)
            break

passage_ids = [p["passage_id"] for p in gold]
ldri_full = {p["passage_id"]: p["ldri"] for p in gold}
raw_full = {p["passage_id"]: p["raw_ldri"] for p in gold}
runtime_full = {p["passage_id"]: p["runtime_s"] for p in gold}

# ── Helper statistics ──
def mean(vals):
    return sum(vals)/len(vals)

def std(vals, ddof=1):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-ddof))

# ── Import pipeline ──
from src.pipeline import LITDISCPipeline
from src.config import LITDISCConfig, HMPDTConfig, LDRIParams
from dataclasses import replace

# ── Ablation runner ──
def run_ablation(abl_name: str, passage_ids: list[str],
                 modify_pipeline=None) -> list[dict]:
    """Run one ablation condition across all passages.
    """
    results = []
    for pid in passage_ids:
        path = passage_files.get(pid)
        if not path or not os.path.exists(path):
            print(f"  [WARN] {pid}: file not found, skipping")
            continue

        # Create fresh pipeline
        config = LITDISCConfig()
        pipeline = LITDISCPipeline(config=config, refinement_mode="full")

        # Apply ablation modifications
        if modify_pipeline:
            modify_pipeline(pipeline, config)

        # Run
        t0 = time.perf_counter()
        try:
            result = pipeline.run(path)
            elapsed = time.perf_counter() - t0
            ldri = float(result.ldri.ldri)
            raw = float(result.ldri.raw)
            gamma = float(result.ldri.gamma_global)
            components = {k: float(v) for k, v in result.ldri.components.items()}
            feature_means = {}
            # Extract feature means from embedding
            emb = result.embedding
            dim_keys = {"S":"semantic","N":"narrative","I":"interaction",
                        "Sy":"symbolic","E":"evolution","Em":"emotion"}
            for k, name in dim_keys.items():
                Hd = getattr(emb, f"H_{k}", None)
                if Hd is not None and len(Hd) > 0:
                    feature_means[name] = float(np.linalg.norm(Hd))
                else:
                    feature_means[name] = 0.0
            # Normalize feature means to match gold-standard scale
            # (gold standard normalizes by sqrt(k_d))
            pass
        except Exception as e:
            print(f"  [ERROR] {pid} ({abl_name}): {e}")
            elapsed = time.perf_counter() - t0
            ldri = 0
            raw = 0
            gamma = 0
            components = {}
            feature_means = {}

        results.append({
            "passage_id": pid,
            "ablation": abl_name,
            "ldri": round(ldri, 6),
            "raw_ldri": round(raw, 6),
            "gamma_global": round(gamma, 6),
            "runtime_s": round(elapsed, 3),
            **{f"comp_{k}": round(v, 6) for k, v in components.items()},
        })

        # Reset pipeline state between runs
        pipeline.reset()

    return results

# ═══════════════════════════════════════════════
# DEFINE ABLATIONS
# ═══════════════════════════════════════════════

# ── Ablation 1: Without Memory ──
def no_memory(pipeline, config):
    # Disable storage
    pipeline.memory.should_store = lambda *a, **kw: False
    pipeline.memory.store_signal = lambda *a, **kw: None
    # Disable retrieval
    pipeline.memory.retrieve = lambda *a, **kw: []
    pipeline.memory.augment = lambda f, gamma, matches: f.copy()
    pipeline.memory.decay_all = lambda: None
    pipeline.memory.clear = lambda: None
    pipeline.memory.get_avg_beta = lambda: 0.0
    # Also set LDRI memory factor to zero
    pipeline.ldri_calc.params = replace(pipeline.ldri_calc.params, delta=0.0)

# ── Ablation 2: Without Resonance ──
def no_resonance(pipeline, config):
    original_reset = pipeline.resonance.reset
    def disabled_reset(dim, initial_tau=0.0):
        import numpy as np
        pipeline.resonance.R = np.zeros(dim, dtype=float)
        pipeline.resonance._prev_H = None
        pipeline.resonance._current_tau = 0.0
        pipeline.resonance._trajectory = [(0.0, np.zeros(dim))]
        pipeline.resonance._n_updates = 0
    pipeline.resonance.reset = disabled_reset
    pipeline.resonance.update = lambda H, delta_tau: np.zeros(len(H))
    pipeline.resonance.augment = lambda f: f.copy()
    pipeline.resonance.get_trajectory_points = lambda: []
    # Set LDRI resonance factor to zero
    pipeline.ldri_calc.params = replace(pipeline.ldri_calc.params, gamma=0.0)

# ── Ablation 3: Without Adaptive Coupling ──
def no_coupling(pipeline, config):
    # Zero out the coupling kernel's compute method
    if hasattr(pipeline.bottom_up, 'kernel'):
        pipeline.bottom_up.kernel.compute = lambda *a, **kw: 0.0
    if hasattr(pipeline.controller, 'kernel'):
        pipeline.controller.kernel.compute = lambda *a, **kw: 0.0
    # Zero out coupling operator
    if hasattr(pipeline.bottom_up, 'operator'):
        old_apply = pipeline.bottom_up.operator.apply
        pipeline.bottom_up.operator.apply = lambda kappa, f, dims: f.copy()
    # Set LDRI coupling factor to zero
    pipeline.ldri_calc.params = replace(pipeline.ldri_calc.params, beta=0.0)

# ── Ablation 4: Without Uncertainty ──
def no_uncertainty(pipeline, config):
    original_extract = pipeline._extract_confidences
    pipeline._extract_confidences = lambda signals: {
        d: 1.0 for d in ["S","N","I","Sy","E","Em"]
    }
    # Also disable uncertainty propagator
    if hasattr(pipeline, 'uncertainty'):
        pipeline.uncertainty.clear = lambda: None
        pipeline.uncertainty.initialize_from_signals = lambda signals: None
        pipeline.uncertainty.propagate_node = lambda *a, **kw: (0.0, 0.0)
        pipeline.uncertainty.propagate_level = lambda *a, **kw: 0.0
        pipeline.uncertainty.propagate_global = lambda *a, **kw: 1.0
        pipeline.uncertainty.get_global_confidence = lambda: 1.0
    # Set gamma_global to 1 in LDRI result by overriding ldri_calc
    original_compute = pipeline.ldri_calc.compute
    def compute_no_uncert(embedding, confidences=None, mem_avg_beta=0.0):
        # Force confidences to 1.0
        result = original_compute(embedding, confidences={"S":1.0,"N":1.0,"I":1.0,"Sy":1.0,"E":1.0,"Em":1.0},
                                   mem_avg_beta=mem_avg_beta)
        # Override gamma_global to 1.0
        from src.models import LDRIScore
        result = LDRIScore(
            ldri=result.raw,  # LDRI without gamma
            gamma_global=1.0,
            reliability=result.reliability,
            raw=result.raw,
            components=result.components,
            fractions=result.fractions,
        )
        return result
    pipeline.ldri_calc.compute = compute_no_uncert

# ── Ablation 5: Without Narrative Signals ──
def no_narrative(pipeline, config):
    # Remove the NarrativeExtractor (key "N") from the extractors dict
    if "N" in pipeline.extractors:
        del pipeline.extractors["N"]

# ═══════════════════════════════════════════════
# RUN ALL ABLATIONS
# ═══════════════════════════════════════════════

ablations = [
    ("no_memory", "Without Memory", no_memory),
    ("no_resonance", "Without Resonance", no_resonance),
    ("no_coupling", "Without Adaptive Coupling", no_coupling),
    ("no_uncertainty", "Without Uncertainty", no_uncertainty),
    ("no_narrative", "Without Narrative Signals", no_narrative),
]

all_results = []
print("Running ablations (5 x 25 passages)...")
for abbr, name, modifier in ablations:
    print(f"  [{abbr}] {name}...")
    t0 = time.perf_counter()
    results = run_ablation(abbr, passage_ids, modify_pipeline=modifier)
    elapsed = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r["ldri"] > 0)
    print(f"    {n_ok}/{len(passage_ids)} passages OK in {elapsed:.1f}s")
    all_results.extend(results)

# ═══════════════════════════════════════════════
# ANALYZE AND WRITE RESULTS
# ═══════════════════════════════════════════════

# For each ablation, compute summary statistics
ablation_summaries = OrderedDict()
ablation_summaries["full_pipeline"] = {
    "mean_ldri": round(mean(list(ldri_full.values())), 6),
    "std_ldri": round(std(list(ldri_full.values())), 6),
    "mean_runtime_s": round(mean(list(runtime_full.values())), 3),
}

for abbr, name, _ in ablations:
    ab_results = [r for r in all_results if r["ablation"] == abbr]
    if not ab_results:
        continue
    ldri_vals = [r["ldri"] for r in ab_results]
    raw_vals = [r["raw_ldri"] for r in ab_results]
    gamma_vals = [r["gamma_global"] for r in ab_results]
    rt_vals = [r["runtime_s"] for r in ab_results]

    # Delta from full pipeline
    deltas = []
    for r in ab_results:
        full_ldri = ldri_full.get(r["passage_id"], 0)
        deltas.append(r["ldri"] - full_ldri)
    abs_deltas = [abs(d) for d in deltas]

    # Find the passage with largest change
    max_delta_idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
    max_delta_passage = ab_results[max_delta_idx]["passage_id"]

    ablation_summaries[name] = {
        "mean_ldri": round(mean(ldri_vals), 6),
        "std_ldri": round(std(ldri_vals), 6),
        "mean_runtime_s": round(mean(rt_vals), 3),
        "mean_delta_ldri": round(mean(deltas), 6),
        "mean_abs_delta_ldri": round(mean(abs_deltas), 6),
        "max_abs_delta_ldri": round(max(abs_deltas), 6),
        "max_delta_passage": max_delta_passage,
        "mean_raw_ldri": round(mean(raw_vals), 6),
        "mean_gamma_global": round(mean(gamma_vals), 6),
    }

# Write ablation_results.csv
with open("ablation_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["# LIT-DISC v1.0.0 -- Ablation Study Results"])
    header = ["passage_id","ablation","ldri","raw_ldri","gamma_global","runtime_s",
              "comp_intrinsic","comp_interaction","comp_temporal","comp_resonance",
              "comp_memory","comp_uncertainty"]
    w.writerow(header)
    for r in all_results:
        w.writerow([
            r["passage_id"], r["ablation"], r["ldri"], r["raw_ldri"],
            r["gamma_global"], r["runtime_s"],
            r.get("comp_intrinsic",""), r.get("comp_interaction",""),
            r.get("comp_temporal",""), r.get("comp_resonance",""),
            r.get("comp_memory",""), r.get("comp_uncertainty",""),
        ])
print("[OK] ablation_results.csv")

# Update baseline_summary.json with ablation data
if os.path.exists("baseline_summary.json"):
    with open("baseline_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    summary["stage_3_ablation_study"] = ablation_summaries
    with open("baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[OK] baseline_summary.json (updated with ablation data)")

# Also write as standalone
with open("ablation_summary.json", "w", encoding="utf-8") as f:
    json.dump(ablation_summaries, f, indent=2)
print("[OK] ablation_summary.json")

# ═══════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════

W = 72
print("\n" + "="*W)
print("  LIT-DISC v1.0.0 -- Ablation Study Report")
print("="*W)

print(f"\n{'='*W}")
print("  ABLATION RESULTS (averaged over 25 passages)")
print(f"{'='*W}")
print(f"  {'Condition':>28} {'LDRI':>8} {'Delta':>8} {'|Delta|':>8} {'Runtime':>8} {'Passages':>8}")
print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

ref_ldri = mean(list(ldri_full.values()))
ref_rt = mean(list(runtime_full.values()))
print(f"  {'Full Pipeline':>28} {ref_ldri:>8.4f} {'---':>8} {'---':>8} {ref_rt:>8.2f}s {'23/25':>8}")

for abbr, name, _ in ablations:
    s = ablation_summaries.get(name, {})
    if not s:
        continue
    marker = ""
    if abs(s.get("mean_abs_delta_ldri", 0)) > 0.01:
        marker = " *"
    print(f"  {name:>28} {s.get('mean_ldri',0):>8.4f} {s.get('mean_delta_ldri',0):>+8.4f} "
          f"{s.get('mean_abs_delta_ldri',0):>8.4f} {s.get('mean_runtime_s',0):>7.2f}s"
          f" {marker:>8}")

print(f"\n  * |Delta| > 0.01 indicates a meaningful ablation effect")

print(f"\n{'='*W}")
print("  DETAILED FINDINGS")
print(f"{'='*W}")

for abbr, name, _ in ablations:
    s = ablation_summaries.get(name, {})
    if not s:
        continue
    delta = s.get("mean_delta_ldri", 0)
    abs_delta = s.get("mean_abs_delta_ldri", 0)
    max_d = s.get("max_abs_delta_ldri", 0)
    max_p = s.get("max_delta_passage", "")
    rt = s.get("mean_runtime_s", 0)

    if abs_delta > 0.05:
        impact = "LARGE"
    elif abs_delta > 0.01:
        impact = "MODERATE"
    else:
        impact = "SMALL"

    print(f"\n  {name}:")
    print(f"    Impact: {impact}  (avg |delta| = {abs_delta:.6f})")
    print(f"    LDRI: {s.get('mean_ldri',0):.6f} (delta = {delta:+.6f})")
    print(f"    Runtime: {rt:.2f}s (vs {ref_rt:.2f}s baseline)")
    print(f"    Largest change: {max_p} (|delta| = {max_d:.6f})")

print(f"\n{'='*W}")
print("  Output files:")
print("    ablation_results.csv")
print("    ablation_summary.json")
print("    baseline_summary.json (updated with ablation data)")
print("="*W)
