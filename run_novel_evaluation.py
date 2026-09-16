"""LIT-DISC v1.1 — Full-Novel Scalability & Longitudinal Evaluation.

Processes The Mistress of Spices (~80k words, 105 windows of ~800 words each)
with persistent memory across the entire novel.
"""
import json, csv, math, os, sys, time, warnings, tempfile
from collections import OrderedDict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.config import LITDISCConfig
from src.pipeline import LITDISCPipeline

# ---- Paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
NOVEL = os.path.join(ROOT, "data", "mistress_spices_narrative.txt")
OUT_DIR = os.path.join(ROOT, "output", "novel_evaluation")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Config ----
config = LITDISCConfig()
print(f"Config: sigma_max={config.ldri.sigma_max}, "
      f"refinement_iters={config.hmp_dt.refinement_iterations}, "
      f"max_size={config.hmp_dt.memory.max_size}")
print()

# ---- 1. Load and split novel into windows ----
print("Loading novel...")
with open(NOVEL, encoding="utf-8") as f:
    text = f.read()

paras = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"  Paragraphs: {len(paras)}")

# Split into ~800-word windows at paragraph boundaries
windows = []
current_paras = []
current_wc = 0
for p in paras:
    pw = len(p.split())
    if current_wc + pw > 800 and current_paras:
        windows.append((current_wc, "\n\n".join(current_paras)))
        current_paras = []
        current_wc = 0
    current_paras.append(p)
    current_wc += pw
if current_paras:
    windows.append((current_wc, "\n\n".join(current_paras)))

N_windows = len(windows)
cumulative_wc = 0
for i, (wc, _) in enumerate(windows):
    cumulative_wc += wc
    windows[i] = (wc, cumulative_wc, windows[i][1])

print(f"  Windows: {N_windows}")
print(f"  Total words: {cumulative_wc}")
print(f"  Window size range: {min(w[0] for w in windows)}-{max(w[0] for w in windows)} words")
print()

# ---- 2. Sequential processing ----
print("Processing novel (memory persists across ALL windows)...")
print()

# Write each window to a temp file, process with pipeline, maintain memory
pipeline = LITDISCPipeline(config=config, refinement_mode="full")

tmpdir = tempfile.mkdtemp(prefix="novel_windows_")

window_results = []
total_start = time.perf_counter()

for i, (wc, cum_wc, window_text) in enumerate(windows):
    wid = i + 1
    tmpfile = os.path.join(tmpdir, f"window_{wid:04d}.txt")
    with open(tmpfile, "w", encoding="utf-8") as f:
        f.write(window_text)

    t0 = time.perf_counter()
    try:
        if i == 0:
            # First window: normal run (memory is empty)
            result = pipeline.run(tmpfile)
        else:
            # Subsequent windows: prevent memory.clear() so memory persists
            pipeline.memory.clear = lambda: None
            result = pipeline.run(tmpfile)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  ERROR window {wid}: {e}")
        elapsed = time.perf_counter() - t0
        pipeline.reset()
        result = None

    # Record metrics
    if result:
        ldri_obj = result.ldri
        comps = ldri_obj.components or {}
        meta = result.metadata or {}
        exp = result.explanation
        mem_acts = exp.memory_activations if exp.memory_activations else []
        retrieval_count = sum(1 for a in mem_acts if isinstance(a, dict) and a.get("activation", 0) > 0) if mem_acts else 0
        resonance_val = 0.0
        if isinstance(exp.resonance_contribution, (int, float)):
            resonance_val = float(exp.resonance_contribution)
        elif isinstance(exp.resonance_contribution, dict):
            resonance_val = float(exp.resonance_contribution.get("strength", 0))

        entry = {
            "window_id": wid,
            "word_count": wc,
            "cumulative_words": cum_wc,
            "runtime_s": round(elapsed, 4),
            "memory_size": meta.get("memory_entries", 0),
            "retrieval_count": retrieval_count,
            "ldri": round(float(ldri_obj.ldri), 8),
            "raw_ldri": round(float(ldri_obj.raw), 8),
            "gamma_global": round(float(ldri_obj.gamma_global), 8),
            "reliability": round(float(ldri_obj.reliability), 8),
            "intrinsic": round(float(comps.get("intrinsic", 0)), 8),
            "interaction": round(float(comps.get("interaction", 0)), 8),
            "temporal": round(float(comps.get("temporal", 0)), 8),
            "resonance": round(float(comps.get("resonance", 0)), 8),
            "memory": round(float(comps.get("memory", 0)), 8),
            "uncertainty": round(float(comps.get("uncertainty", 0)), 8),
            "fraction_intrinsic": round(float(ldri_obj.fractions.get("intrinsic", 0)), 8),
            "fraction_interaction": round(float(ldri_obj.fractions.get("interaction", 0)), 8),
            "fraction_temporal": round(float(ldri_obj.fractions.get("temporal", 0)), 8),
            "fraction_resonance": round(float(ldri_obj.fractions.get("resonance", 0)), 8),
            "fraction_memory": round(float(ldri_obj.fractions.get("memory", 0)), 8),
            "fraction_uncertainty": round(float(ldri_obj.fractions.get("uncertainty", 0)), 8),
            "resonance_contribution": round(resonance_val, 8),
        }
    else:
        entry = {
            "window_id": wid,
            "word_count": wc,
            "cumulative_words": cum_wc,
            "runtime_s": round(elapsed, 4),
            "memory_size": 0,
            "retrieval_count": 0,
            "ldri": 0, "raw_ldri": 0, "gamma_global": 0,
            "reliability": 0,
            "intrinsic": 0, "interaction": 0, "temporal": 0,
            "resonance": 0, "memory": 0, "uncertainty": 0,
            "fraction_intrinsic": 0, "fraction_interaction": 0,
            "fraction_temporal": 0, "fraction_resonance": 0,
            "fraction_memory": 0, "fraction_uncertainty": 0,
            "resonance_contribution": 0,
        }

    window_results.append(entry)

    if (i + 1) % 10 == 0 or i == 0 or i == N_windows - 1:
        print(f"  [{i+1:>3}/{N_windows}] w{cum_wc:>6}  "
              f"LDRI={entry['ldri']:.6f}  raw={entry['raw_ldri']:.6f}  "
              f"gamma={entry['gamma_global']:.6f}  "
              f"mem={entry['memory_size']:>3}  "
              f"rt={entry['runtime_s']:.2f}s")

total_elapsed = time.perf_counter() - total_start
print(f"\n  Total processing time: {total_elapsed:.1f}s")
print(f"  Average per window: {total_elapsed/N_windows:.2f}s")
print()

# Cleanup temp files
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ---- 3. Write chapter_statistics.csv ----
with open(os.path.join(OUT_DIR, "chapter_statistics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["window_id","cumulative_words","runtime_s","memory_size","retrieval_count",
                 "ldri","raw_ldri","gamma_global","reliability",
                 "intrinsic","interaction","temporal","resonance","memory","uncertainty",
                 "resonance_contribution"])
    for r in window_results:
        w.writerow([r["window_id"], r["cumulative_words"], r["runtime_s"],
                    r["memory_size"], r["retrieval_count"],
                    r["ldri"], r["raw_ldri"], r["gamma_global"], r["reliability"],
                    r["intrinsic"], r["interaction"], r["temporal"],
                    r["resonance"], r["memory"], r["uncertainty"],
                    r["resonance_contribution"]])
print("[OK] chapter_statistics.csv")

# ---- 4. Compute derived metrics ----
def mean(v):
    return sum(v)/len(v) if v else 0
def std(v, ddof=1):
    if len(v) < 2: return 0.0
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-ddof))
def var(v, ddof=1):
    if len(v) < 2: return 0.0
    m = mean(v)
    return sum((x-m)**2 for x in v)/(len(v)-ddof)
def cv(v):
    m = mean(v)
    return std(v)/m if m != 0 else 0

ldris = [r["ldri"] for r in window_results]
raws = [r["raw_ldri"] for r in window_results]
gammas = [r["gamma_global"] for r in window_results]
runtimes = [r["runtime_s"] for r in window_results]
memories = [r["memory_size"] for r in window_results]
retrievals = [r["retrieval_count"] for r in window_results]
intrinsics = [r["intrinsic"] for r in window_results]
interactions = [r["interaction"] for r in window_results]
temporals = [r["temporal"] for r in window_results]
resonances_comp = [r["resonance"] for r in window_results]
uncertainties = [r["uncertainty"] for r in window_results]
words = [r["word_count"] for r in window_results]
cum_words = [r["cumulative_words"] for r in window_results]

# ---- 5. Scalability ----
# Runtime scaling
rt_per_word = total_elapsed / cumulative_wc
rt_correlation = np.corrcoef(cum_words, runtimes)[0, 1]
# Check linearity — ratio of runtime-to-words in first vs second half
half = N_windows // 2
rt_per_word_first = sum(runtimes[:half]) / max(sum(words[:half]), 1)
rt_per_word_second = sum(runtimes[half:]) / max(sum(words[half:]), 1)
scalability_ratio = rt_per_word_second / max(rt_per_word_first, 1e-15)

scalability = {
    "total_windows": N_windows,
    "total_words": cumulative_wc,
    "total_runtime_s": round(total_elapsed, 2),
    "avg_runtime_per_window_s": round(total_elapsed / N_windows, 4),
    "avg_runtime_per_1000_words_s": round(rt_per_word * 1000, 4),
    "runtime_word_correlation": round(rt_correlation, 4),
    "rt_per_word_first_half": round(rt_per_word_first, 6),
    "rt_per_word_second_half": round(rt_per_word_second, 6),
    "scalability_ratio_second_vs_first": round(scalability_ratio, 4),
    "conclusion": "LINEAR" if scalability_ratio < 1.5 else "SUPER-LINEAR",
}

# Memory scaling
mem_max = max(memories)
mem_final = memories[-1]
mem_fill_ratio = mem_final / max(config.hmp_dt.memory.max_size, 1)
mem_correlation = np.corrcoef(cum_words, memories)[0, 1]

# Runtime scaling CSV
with open(os.path.join(OUT_DIR, "runtime_scaling.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cumulative_words","runtime_s","memory_size","retrieval_count",
                 "runtime_per_1000w","mem_fill_pct"])
    for i, r in enumerate(window_results):
        rp1k = r["runtime_s"] / max(r["word_count"], 1) * 1000
        mfpct = r["memory_size"] / max(config.hmp_dt.memory.max_size, 1) * 100
        w.writerow([r["cumulative_words"], r["runtime_s"], r["memory_size"],
                    r["retrieval_count"], round(rp1k, 4), round(mfpct, 2)])
print("[OK] runtime_scaling.csv")

# Memory analysis CSV
# No easy way to get per-window retrieval success without instrumenting memory.
# We record memory_size and retrieval_count from the explanation trace.
with open(os.path.join(OUT_DIR, "memory_analysis.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["window_id","cumulative_words","memory_size","retrieval_count",
                 "mem_fill_pct","new_entries"])
    prev_mem = 0
    for r in window_results:
        new_entries = max(0, r["memory_size"] - prev_mem)
        mfpct = r["memory_size"] / max(config.hmp_dt.memory.max_size, 1) * 100
        w.writerow([r["window_id"], r["cumulative_words"], r["memory_size"],
                    r["retrieval_count"], round(mfpct, 2), new_entries])
        prev_mem = r["memory_size"]
print("[OK] memory_analysis.csv")

# Longitudinal metrics CSV
with open(os.path.join(OUT_DIR, "longitudinal_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["window_id","cumulative_words","ldri","raw_ldri","gamma_global",
                 "intrinsic","interaction","temporal","resonance","memory","uncertainty",
                 "reliability","resonance_contribution"])
    for r in window_results:
        w.writerow([r["window_id"], r["cumulative_words"],
                    r["ldri"], r["raw_ldri"], r["gamma_global"],
                    r["intrinsic"], r["interaction"], r["temporal"],
                    r["resonance"], r["memory"], r["uncertainty"],
                    r["reliability"], r["resonance_contribution"]])
print("[OK] longitudinal_metrics.csv")

# ---- 6. Gamma analysis / variance decomposition ----
# Compare gold-standard vs novel
ldri_log = [math.log(max(1e-10, v)) for v in ldris]
raw_log = [math.log(max(1e-10, v)) for v in raws]
gamma_log = [math.log(max(1e-10, v)) for v in gammas]

vll = var(ldri_log)
vlr = var(raw_log)
vlg = var(gamma_log)
cov_lr = np.cov([raw_log, ldri_log])[0, 1] if len(raw_log) > 1 else 0
cov_lg = np.cov([gamma_log, ldri_log])[0, 1] if len(gamma_log) > 1 else 0

# Load gold-standard for comparison
with open(os.path.join(ROOT, "data", "gold_standard", "litdisc_results.json"), encoding="utf-8") as f:
    gs_data = json.load(f)
gs_ldris = [p["ldri"] for p in gs_data]
gs_raws = [p["raw_ldri"] for p in gs_data]
gs_gammas = [p["gamma_global"] for p in gs_data]
gs_ldri_log = [math.log(max(1e-10, v)) for v in gs_ldris]
gs_raw_log = [math.log(max(1e-10, v)) for v in gs_raws]
gs_gamma_log = [math.log(max(1e-10, v)) for v in gs_gammas]
gs_vll = var(gs_ldri_log)
gs_vlr = var(gs_raw_log)
gs_vlg = var(gs_gamma_log)

variance_decomp = {
    "novel": {
        "var_log_ldri": round(vll, 6),
        "var_log_raw": round(vlr, 6),
        "var_log_gamma": round(vlg, 6),
        "gamma_pct": round(vlg / max(vll, 1e-15) * 100, 1),
        "raw_pct": round(vlr / max(vll, 1e-15) * 100, 1),
        "cov_pct": round(2 * cov_lg / max(vll, 1e-15) * 100, 1),
    },
    "gold_standard": {
        "var_log_ldri": round(gs_vll, 6),
        "var_log_raw": round(gs_vlr, 6),
        "var_log_gamma": round(gs_vlg, 6),
        "gamma_pct": round(gs_vlg / max(gs_vll, 1e-15) * 100, 1),
        "raw_pct": round(gs_vlr / max(gs_vll, 1e-15) * 100, 1),
    },
}

with open(os.path.join(OUT_DIR, "variance_decomposition_novel.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source","var_log_ldri","gamma_pct","raw_pct","cov_pct"])
    w.writerow(["gold_standard", variance_decomp["gold_standard"]["var_log_ldri"],
                variance_decomp["gold_standard"]["gamma_pct"],
                variance_decomp["gold_standard"]["raw_pct"], 0])
    w.writerow(["full_novel", variance_decomp["novel"]["var_log_ldri"],
                variance_decomp["novel"]["gamma_pct"],
                variance_decomp["novel"]["raw_pct"],
                variance_decomp["novel"]["cov_pct"]])
print("[OK] variance_decomposition_novel.csv")

# ---- 7. Stability checks ----
# Check for NaN/Inf in all metrics
stable = True
for r in window_results:
    for k, v in r.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            stable = False
            break

# Check for monotonic runtime
runtime_jumps = sum(1 for i in range(1, len(runtimes)) if runtimes[i] > runtimes[i-1] * 3)
# Check memory doesn't exceed max_size
mem_exceeded = sum(1 for m in memories if m > config.hmp_dt.memory.max_size)
# Check component bounds
comp_bounds = {
    "intrinsic": all(0 <= v <= 1 for v in intrinsics),
    "interaction": all(0 <= v <= 1 for v in interactions),
    "temporal": all(0 <= v <= 1 for v in temporals),
    "resonance": all(0 <= v <= 1 for v in resonances_comp),
    "uncertainty": all(0 <= v <= 1 for v in uncertainties),
}

stability = {
    "numerical_stable": stable,
    "runtime_jumps_gt_3x": runtime_jumps,
    "memory_capacity_exceeded": mem_exceeded,
    "components_in_bounds_0_1": comp_bounds,
    "conclusion": "STABLE" if (stable and mem_exceeded == 0) else "UNSTABLE",
}

# ---- 8. Write novel_results.json ----
novel_results = OrderedDict()
novel_results["metadata"] = {
    "title": "The Mistress of Spices",
    "author": "Chitra Banerjee Divakaruni",
    "version": "LIT-DISC v1.1",
    "sigma_max": config.ldri.sigma_max,
    "total_words": cumulative_wc,
    "total_windows": N_windows,
    "avg_window_words": round(cumulative_wc / N_windows, 1),
    "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_runtime_s": round(total_elapsed, 2),
    "peak_memory_entries": max(memories),
    "final_memory_entries": memories[-1],
    "memory_capacity": config.hmp_dt.memory.max_size,
}
novel_results["scalability"] = scalability
novel_results["variance_decomposition"] = variance_decomp
novel_results["stability"] = stability
novel_results["summary"] = {
    "ldri_mean": round(mean(ldris), 8),
    "ldri_std": round(std(ldris), 8),
    "ldri_min": round(min(ldris), 8),
    "ldri_max": round(max(ldris), 8),
    "ldri_cv": round(cv(ldris), 4),
    "raw_ldri_mean": round(mean(raws), 8),
    "raw_ldri_std": round(std(raws), 8),
    "raw_ldri_cv": round(cv(raws), 4),
    "gamma_mean": round(mean(gammas), 8),
    "gamma_std": round(std(gammas), 8),
    "gamma_cv": round(cv(gammas), 4),
    "avg_memory_size": round(mean(memories), 1),
    "avg_runtime_s": round(mean(runtimes), 4),
    "avg_retrieval_count": round(mean(retrievals), 1),
    "avg_intrinsic": round(mean(intrinsics), 8),
    "avg_interaction": round(mean(interactions), 8),
    "avg_temporal": round(mean(temporals), 8),
    "avg_resonance": round(mean(resonances_comp), 8),
    "avg_uncertainty": round(mean(uncertainties), 8),
    "resonance_peak": round(max([r["resonance_contribution"] for r in window_results]), 8),
    "resonance_avg": round(mean([r["resonance_contribution"] for r in window_results]), 8),
}
novel_results["windows"] = window_results

with open(os.path.join(OUT_DIR, "novel_results.json"), "w", encoding="utf-8") as f:
    json.dump(novel_results, f, indent=2, ensure_ascii=False)
print("[OK] novel_results.json")

# ---- 9. Print console summary ----
W = 72
print("\n" + "="*W)
print("  FULL-NOVEL EVALUATION SUMMARY")
print("="*W)
print(f"  Novel: The Mistress of Spices ({cumulative_wc} words)")
print(f"  Windows: {N_windows} (avg {cumulative_wc//N_windows} words/window)")
print(f"  Total runtime: {total_elapsed:.1f}s")
print(f"  Average: {total_elapsed/N_windows:.2f}s/window, {rt_per_word*1000:.4f}s/1k words")
print(f"  Runtime-word correlation: {rt_correlation:.4f}")
print(f"  Scalability (second/first half ratio): {scalability_ratio:.3f}")
print(f"  Conclusion: {scalability['conclusion']}")
print()
print(f"  LDRI: mean={mean(ldris):.6f}  std={std(ldris):.6f}  CV={cv(ldris):.2%}")
print(f"  Raw LDRI: mean={mean(raws):.6f}  std={std(raws):.6f}  CV={cv(raws):.2%}")
print(f"  Gamma: mean={mean(gammas):.6f}  std={std(gammas):.6f}  CV={cv(gammas):.2%}")
print()
print(f"  Memory: peak={mem_max}, final={mem_final}, cap={config.hmp_dt.memory.max_size}, fill={mem_fill_ratio:.0%}")
print(f"  Avg retrieval: {mean(retrievals):.1f} per window")
print()
print(f"  Gamma dominance (novel): {variance_decomp['novel']['gamma_pct']:.1f}%")
print(f"  Gamma dominance (GS):    {variance_decomp['gold_standard']['gamma_pct']:.1f}%")
print(f"  Raw contribution (novel): {variance_decomp['novel']['raw_pct']:.1f}%")
print(f"  Raw contribution (GS):    {variance_decomp['gold_standard']['raw_pct']:.1f}%")
print()
print(f"  Stability: {'PASS' if stable else 'FAIL'} (NaNs/Infs), "
      f"memory bounds: {'PASS' if mem_exceeded==0 else 'FAIL'}, "
      f"components in [0,1]: {'PASS' if all(comp_bounds.values()) else 'FAIL'}")
print()
print(f"  Resonance peak: {max([r['resonance_contribution'] for r in window_results]):.8f}")
print(f"  Resonance avg: {mean([r['resonance_contribution'] for r in window_results]):.8f}")
print("="*W)

# ---- 10. Generate figures (Part 9) ----
print("\nGenerating figures...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
    })

    x = cum_words  # cumulative words as x-axis

    # Fig 1: Runtime vs cumulative words
    fig, ax = plt.subplots()
    ax.plot(x, runtimes, "b-", linewidth=0.8, alpha=0.7)
    ax.scatter(x[::5], runtimes[::5], s=10, c="blue", alpha=0.5)
    z = np.polyfit(x, runtimes, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", linewidth=1, label=f"Linear fit (r={rt_correlation:.3f})")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_runtime_scaling.png"))
    plt.close(fig)

    # Fig 2: Memory growth
    fig, ax = plt.subplots()
    ax.plot(x, memories, "g-", linewidth=0.8)
    ax.axhline(y=config.hmp_dt.memory.max_size, color="r", linestyle="--",
               label=f"Capacity ({config.hmp_dt.memory.max_size})")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Memory entries")
    ax.set_title("Memory Growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_memory_growth.png"))
    plt.close(fig)

    # Fig 3: LDRI over narrative progression
    fig, ax = plt.subplots()
    ax.plot(x, ldris, "k-", linewidth=0.8)
    ax.fill_between(x, [mean(ldris)-std(ldris)]*len(x),
                    [mean(ldris)+std(ldris)]*len(x), alpha=0.1, color="gray")
    ax.axhline(y=mean(ldris), color="r", linestyle="--", alpha=0.5,
               label=f"Mean = {mean(ldris):.5f}")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("LDRI")
    ax.set_title("LDRI Over Narrative Progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_ldri_progression.png"))
    plt.close(fig)

    # Fig 4: Raw LDRI progression
    fig, ax = plt.subplots()
    ax.plot(x, raws, "c-", linewidth=0.8)
    ax.axhline(y=mean(raws), color="r", linestyle="--", alpha=0.5,
               label=f"Mean = {mean(raws):.5f}")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Raw LDRI")
    ax.set_title("Raw LDRI Progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_raw_ldri_progression.png"))
    plt.close(fig)

    # Fig 5: Gamma progression
    fig, ax = plt.subplots()
    ax.plot(x, gammas, "m-", linewidth=0.8)
    ax.axhline(y=mean(gammas), color="r", linestyle="--", alpha=0.5,
               label=f"Mean = {mean(gammas):.5f}")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Gamma (global confidence)")
    ax.set_title("Confidence (Gamma) Progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_gamma_progression.png"))
    plt.close(fig)

    # Fig 6: Resonance progression
    res_contribs = [r["resonance_contribution"] for r in window_results]
    fig, ax = plt.subplots()
    ax.plot(x, res_contribs, "orange", linewidth=0.8)
    ax.axhline(y=mean(res_contribs), color="r", linestyle="--", alpha=0.5,
               label=f"Mean = {mean(res_contribs):.6f}")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Resonance contribution")
    ax.set_title("Resonance Evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_resonance_progression.png"))
    plt.close(fig)

    # Fig 7: Component evolution
    fig, ax = plt.subplots()
    ax.plot(x, intrinsics, label="Intrinsic", linewidth=0.8)
    ax.plot(x, interactions, label="Interaction", linewidth=0.8)
    ax.plot(x, temporals, label="Temporal", linewidth=0.8)
    ax.plot(x, resonances_comp, label="Resonance (comp)", linewidth=0.8)
    ax.plot(x, [r["memory"] for r in window_results], label="Memory", linewidth=0.8)
    ax.plot(x, uncertainties, label="Uncertainty", linewidth=0.8)
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Component value")
    ax.set_title("Component Evolution")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_component_evolution.png"))
    plt.close(fig)

    # Fig 8: Retrieval frequency
    fig, ax = plt.subplots()
    ax.bar(x[::2], retrievals[::2], width=cum_words[1]*0.6 if len(cum_words)>1 else 500,
           alpha=0.6, color="teal")
    ax.set_xlabel("Cumulative words")
    ax.set_ylabel("Retrieval count")
    ax.set_title("Retrieval Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_retrieval_frequency.png"))
    plt.close(fig)

    # Fig 9: Variance contribution comparison (Gold Standard vs Full Novel)
    fig, ax = plt.subplots()
    categories = ["Gamma", "Raw LDRI"]
    gs_vals = [variance_decomp["gold_standard"]["gamma_pct"],
               variance_decomp["gold_standard"]["raw_pct"]]
    nv_vals = [variance_decomp["novel"]["gamma_pct"],
               variance_decomp["novel"]["raw_pct"]]
    x_pos = np.arange(len(categories))
    w = 0.35
    ax.bar(x_pos - w/2, gs_vals, w, label="Gold Standard (25 passages)", color="steelblue")
    ax.bar(x_pos + w/2, nv_vals, w, label="Full Novel (105 windows)", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Contribution to var(log(LDRI)) (%)")
    ax.set_title("Variance Contribution Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_variance_comparison.png"))
    plt.close(fig)

    print(f"  Generated 9 figures in {FIG_DIR}")
except ImportError:
    print("  WARNING: matplotlib not available, skipping figures.")
except Exception as e:
    print(f"  WARNING: Figure generation error: {e}")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
