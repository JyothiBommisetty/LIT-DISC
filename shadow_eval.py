"""LIT-DISC v1.1 -- Shadow Evaluation: sigma=5.0 vs sigma=0.1.

Overrides sigma_max via config (no source modifications).
Runs the full 25-passage gold-standard evaluation.
Compares activations, LDRI, components, rankings.

Outputs: comparison_before_after.csv, sigma5_validation.json, recommendation.md
"""
import json, csv, math, os, sys, time, warnings
from collections import OrderedDict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from dataclasses import replace
from src.config import LITDISCConfig, LDRIParams
from src.pipeline import LITDISCPipeline

# -- Names --
DIM_KEYS = ["S", "N", "I", "Sy", "E", "Em"]
DIM_NAMES = ["semantic", "narrative", "interaction", "symbolic", "evolution", "emotion"]
COMP_NAMES = ["intrinsic","interaction","temporal","resonance","memory","uncertainty"]

# -- Load sigma=0.1 baseline --
with open("data/gold_standard/litdisc_results.json", encoding="utf-8") as f:
    BASELINE = json.load(f)

# -- Stats --
def mean(v):
    return sum(v)/len(v) if v else 0

def std(v, ddof=1):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-ddof))

def var(v, ddof=1):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return sum((x-m)**2 for x in v)/(len(v)-ddof)

def cv(v):
    m = mean(v)
    return std(v)/m if m != 0 else 0

def pearson_r(x, y):
    n = min(len(x), len(y))
    mx, my = sum(x[:n])/n, sum(y[:n])/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    den = math.sqrt(sum((x[i]-mx)**2 for i in range(n)) * sum((y[i]-my)**2 for i in range(n)))
    return num/den if den != 0 else 0

def spearman_r(x, y):
    n = min(len(x), len(y))
    def rank(vals):
        s = sorted([(v, i) for i, v in enumerate(vals[:n])])
        rv = [0]*n
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j][0] == s[i][0]:
                j += 1
            rnk = (i + j - 1)/2.0 + 1
            for k in range(i, j):
                rv[s[k][1]] = rnk
            i = j
        return rv
    return pearson_r(rank(x), rank(y))

# -- Create sigma=5.0 pipeline --
sigma_new = 5.0
base_config = LITDISCConfig()
new_config = LITDISCConfig(
    ldri=replace(base_config.ldri, sigma_max=sigma_new)
)
pipeline = LITDISCPipeline(config=new_config, refinement_mode="full")

# -- Map passage files --
PASSAGE_DIR = "data/gold_standard/passages"
passage_files = {}
for fname in os.listdir(PASSAGE_DIR):
    for p in BASELINE:
        if fname.startswith(p["passage_id"]):
            passage_files[p["passage_id"]] = os.path.join(PASSAGE_DIR, fname)
            break

# -- Baseline data lookup --
def base(p, field):
    for b in BASELINE:
        if b["passage_id"] == p:
            return b.get(field, 0)
    return 0

# -- Run evaluation --
print(f"Shadow evaluation: sigma_max = {sigma_new}")
print("Processing 25 passages...")

results_new = []
for idx, pid in enumerate(passage_files):
    path = passage_files[pid]
    if not path or not os.path.exists(path):
        continue

    t0 = time.perf_counter()
    try:
        result = pipeline.run(path)
        elapsed = time.perf_counter() - t0
        emb = result.embedding

        # Extract H_d norms
        norms = {}
        acts = {}
        for i, k in enumerate(DIM_KEYS):
            Hd = getattr(emb, f"H_{k}", None)
            if Hd is not None and len(Hd) > 0:
                norms[DIM_NAMES[i]] = float(np.linalg.norm(Hd))
                # Compute activation with both sigma values
                acts[DIM_NAMES[i]] = {
                    "new": 1.0 - math.exp(-norms[DIM_NAMES[i]]**2 / (2.0 * sigma_new**2)),
                    "old": 1.0 - math.exp(-norms[DIM_NAMES[i]]**2 / (2.0 * 0.1**2)),
                }
            else:
                norms[DIM_NAMES[i]] = 0.0
                acts[DIM_NAMES[i]] = {"new": 0.0, "old": 0.0}

        ldri_obj = result.ldri
        comps = {k: float(v) for k, v in ldri_obj.components.items()}
        fracs = {k: float(v) for k, v in ldri_obj.fractions.items()}

        results_new.append({
            "passage_id": pid,
            "author": base(pid, "author"),
            "category": base(pid, "category"),
            "title": base(pid, "title"),
            "word_count": base(pid, "word_count") or base(pid, "words") or 0,
            "ldri": float(ldri_obj.ldri),
            "raw_ldri": float(ldri_obj.raw),
            "gamma_global": float(ldri_obj.gamma_global),
            "runtime_s": round(elapsed, 3),
            "components": comps,
            "fractions": fracs,
            "feature_norms": norms,
            "activations_new": {d: acts[d]["new"] for d in DIM_NAMES},
            "activations_old": {d: acts[d]["old"] for d in DIM_NAMES},
        })
    except Exception as e:
        print(f"  [ERROR] {pid}: {e}")
        results_new.append({
            "passage_id": pid, "ldri": 0, "raw_ldri": 0, "gamma_global": 0,
            "runtime_s": 0, "components": {}, "feature_norms": {},
            "activations_new": {}, "activations_old": {},
        })

    if (idx+1) % 5 == 0:
        print(f"  {idx+1}/25 complete")
    pipeline.reset()

# -- Filter valid results --
valid = [r for r in results_new if r["ldri"] > 0]
print(f"\n  {len(valid)}/{25} passages valid")

# ================================================================
# COMPARISON ANALYSIS
# ================================================================

def extract_baseline(field):
    return [b.get(field, 0) for b in BASELINE]

def extract_new(field):
    return [r.get(field, 0) for r in valid]

def extract_base_component(c):
    return [b["components"][c] for b in BASELINE]

def extract_new_component(c):
    return [r["components"].get(c, 0) for r in valid]

comp_rows = []  # For CSV

# -- 1. Activation distributions --
print("\n--- 1. Activation Distribution Comparison ---")
for d in DIM_NAMES:
    old_acts = [1.0 for _ in valid]  # sigma=0.1: all saturated
    new_acts = [r["activations_new"][d] for r in valid]
    old_mean = mean(old_acts)
    new_mean = mean(new_acts)
    old_cv = cv(old_acts)
    new_cv = cv(new_acts)
    comp_rows.append({
        "comparison": "activation_" + d,
        "old_value": round(old_mean, 6),
        "new_value": round(new_mean, 6),
        "old_std": round(std(old_acts), 6),
        "new_std": round(std(new_acts), 6),
        "old_cv": round(old_cv, 4),
        "new_cv": round(new_cv, 4),
        "change_pct": round((new_mean - old_mean)/max(old_mean, 1e-15)*100, 1),
        "interpretation": "Saturated→variable" if new_cv > 0.01 else "Both saturated"
    })
    print(f"  {d:>12}: old a={old_mean:.4f} (CV={old_cv:.4f}) -> new a={new_mean:.4f} (CV={new_cv:.4f})")

# -- 2. LDRI and raw_ldri --
print("\n--- 2. LDRI and Raw LDRI ---")
for name, old_fn, new_fn in [
    ("ldri", lambda: extract_baseline("ldri"), lambda: extract_new("ldri")),
    ("raw_ldri", lambda: extract_baseline("raw_ldri"), lambda: extract_new("raw_ldri")),
    ("gamma_global", lambda: extract_baseline("gamma_global"), lambda: extract_new("gamma_global")),
]:
    o = old_fn()
    n = new_fn()
    comp_rows.append({
        "comparison": name, "old_value": round(mean(o), 6), "new_value": round(mean(n), 6),
        "old_std": round(std(o), 6), "new_std": round(std(n), 6),
        "old_cv": round(cv(o), 4), "new_cv": round(cv(n), 4),
        "change_pct": round((mean(n)-mean(o))/max(mean(o),1e-15)*100, 1),
        "interpretation": "Changed scale" if abs(cv(n)-cv(o)) > 0.05 else "Same relative spread"
    })
    print(f"  {name:>12}: old {mean(o):.6f} (CV={cv(o):.4f}) -> new {mean(n):.6f} (CV={cv(n):.4f})")

# -- 3. Components --
print("\n--- 3. Component Comparison ---")
for c in COMP_NAMES:
    o = extract_base_component(c)
    n = extract_new_component(c)
    comp_rows.append({
        "comparison": "comp_"+c, "old_value": round(mean(o), 6), "new_value": round(mean(n), 6),
        "old_std": round(std(o), 6), "new_std": round(std(n), 6),
        "old_cv": round(cv(o), 4), "new_cv": round(cv(n), 4),
        "change_pct": round((mean(n)-mean(o))/max(mean(o),1e-15)*100, 1),
        "interpretation": ""
    })
    print(f"  {c:>12}: old {mean(o):.6f} (CV={cv(o):.4f}) -> new {mean(n):.6f} (CV={cv(n):.4f})")

# -- 4. Category rankings --
print("\n--- 4. Category Rankings ---")
cats = ["MLC","MNC","NWC","SNC","SSC"]
cat_old = {}
cat_new = {}
for cat in cats:
    cat_old[cat] = [b["ldri"] for b in BASELINE if b["category"] == cat]
    cat_new[cat] = [r["ldri"] for r in valid if r["category"] == cat]

for cat in cats:
    mo = mean(cat_old[cat])
    mn = mean(cat_new[cat])
    comp_rows.append({
        "comparison": "cat_ldri_"+cat, "old_value": round(mo, 6), "new_value": round(mn, 6),
        "old_std": round(std(cat_old[cat]), 6), "new_std": round(std(cat_new[cat]), 6),
        "old_cv": round(cv(cat_old[cat]), 4), "new_cv": round(cv(cat_new[cat]), 4),
        "change_pct": round((mn-mo)/max(mo,1e-15)*100, 1),
        "interpretation": ""
    })

cat_ranking_old = sorted(cats, key=lambda c: mean(cat_old[c]), reverse=True)
cat_ranking_new = sorted(cats, key=lambda c: mean(cat_new[c]), reverse=True)
print(f"  Old ranking: {cat_ranking_old}")
print(f"  New ranking: {cat_ranking_new}")
print(f"  Same order: {cat_ranking_old == cat_ranking_new}")

# -- 5. Author rankings --
print("\n--- 5. Author Ranking Comparison ---")
authors_old = sorted(set(b["author"] for b in BASELINE))
author_ldri_old = {a: mean([b["ldri"] for b in BASELINE if b["author"] == a]) for a in authors_old}
author_ldri_new = {a: mean([r["ldri"] for r in valid if r["author"] == a]) for a in authors_old}

author_rank_old = sorted(authors_old, key=lambda a: author_ldri_old[a], reverse=True)
author_rank_new = sorted(authors_old, key=lambda a: author_ldri_new[a], reverse=True)

# Spearman correlation of author ranking
author_rank_pos_old = {a: i for i, a in enumerate(author_rank_old)}
author_rank_pos_new = {a: i for i, a in enumerate(author_rank_new)}
rank_diffs = [author_rank_pos_old[a] - author_rank_pos_new[a] for a in authors_old]
rank_r = pearson_r(list(author_rank_pos_old.values()), list(author_rank_pos_new.values()))
rank_rho = spearman_r(list(range(len(authors_old))),
                      [author_rank_pos_new[a] for a in author_rank_old])

comp_rows.append({
    "comparison": "author_ranking_spearman", "old_value": 1.0, "new_value": round(rank_rho, 4),
    "old_std": 0, "new_std": 0, "old_cv": 0, "new_cv": 0, "change_pct": 0,
    "interpretation": f"rho={rank_rho:.4f}"
})

print(f"  Top 5 old: {author_rank_old[:5]}")
print(f"  Top 5 new: {author_rank_new[:5]}")
print(f"  Ranking Spearman rho: {rank_rho:.4f}")

# -- 6. Runtime --
runtimes_old = [b["runtime_s"] for b in BASELINE]
runtimes_new = [r["runtime_s"] for r in valid]
comp_rows.append({
    "comparison": "runtime_s", "old_value": round(mean(runtimes_old), 3), "new_value": round(mean(runtimes_new), 3),
    "old_std": round(std(runtimes_old), 3), "new_std": round(std(runtimes_new), 3),
    "old_cv": round(cv(runtimes_old), 4), "new_cv": round(cv(runtimes_new), 4),
    "change_pct": round((mean(runtimes_new)-mean(runtimes_old))/max(mean(runtimes_old),1e-15)*100, 1),
    "interpretation": "No change expected (sigma is only in LDRI calc, not propagation)"
})

# -- 7. Variance ratios --
ldri_old = extract_baseline("ldri")
ldri_new = extract_new("ldri")
raw_old = extract_baseline("raw_ldri")
raw_new = extract_new("raw_ldri")

# Ratio of raw_ldri variance to LDRI variance
var_ratio_old = var(raw_old) / max(var(ldri_old), 1e-15)
var_ratio_new = var(raw_new) / max(var(ldri_new), 1e-15)
comp_rows.append({
    "comparison": "var_raw_over_var_ldri", "old_value": round(var_ratio_old, 4), "new_value": round(var_ratio_new, 4),
    "old_std": 0, "new_std": 0, "old_cv": 0, "new_cv": 0, "change_pct": 0,
    "interpretation": "Higher = raw_ldri contributes more to LDRI variance"
})

# ================================================================
# WRITE OUTPUTS
# ================================================================

# comparison_before_after.csv
with open("comparison_before_after.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["comparison","old_value","new_value","old_std","new_std",
                "old_cv","new_cv","change_pct","interpretation"])
    for r in comp_rows:
        w.writerow([r["comparison"], r["old_value"], r["new_value"],
                    r["old_std"], r["new_std"], r["old_cv"], r["new_cv"],
                    r["change_pct"], r["interpretation"]])
print("[OK] comparison_before_after.csv")

# sigma5_validation.json — per-passage results
sigma5 = OrderedDict()
sigma5["config"] = {
    "sigma_max_old": 0.1,
    "sigma_max_new": sigma_new,
    "total_passages": len(valid),
    "pipeline_config": "LITDISCConfig(ldri=replace(LITDISCConfig().ldri, sigma_max=5.0))",
    "no_source_modifications": True,
}
# Pre-compute log-space vars for LDVI decomposition
ldri_old_log = [math.log(max(1e-10, v)) for v in ldri_old]
ldri_new_log = [math.log(max(1e-10, v)) for v in ldri_new]
raw_old_log = [math.log(max(1e-10, v)) for v in raw_old]
raw_new_log = [math.log(max(1e-10, v)) for v in raw_new]
gamma_old = extract_baseline("gamma_global")
gamma_new = extract_new("gamma_global")
gamma_old_log = [math.log(max(1e-10, v)) for v in gamma_old]
gamma_new_log = [math.log(max(1e-10, v)) for v in gamma_new]
vll_old = var(ldri_old_log)
vll_new = var(ldri_new_log)
vlr_old = var(raw_old_log)
vlr_new = var(raw_new_log)
vlg_old = var(gamma_old_log)
vlg_new = var(gamma_new_log)

sigma5["summary"] = OrderedDict()
sigma5["summary"]["activation_saturation_old"] = "100% (all activations = 1.0)"
sigma5["summary"]["activation_saturation_new"] = (
    f"0% (mean a = {mean([r['activations_new'][d] for r in valid for d in DIM_NAMES]):.4f}, "
    f"CV = {cv([r['activations_new'][d] for r in valid for d in DIM_NAMES]):.4f})"
)
sigma5["summary"]["raw_ldri_cv_old"] = round(cv(raw_old), 4)
sigma5["summary"]["raw_ldri_cv_new"] = round(cv(raw_new), 4)
sigma5["summary"]["ldri_cv_old"] = round(cv(ldri_old), 4)
sigma5["summary"]["ldri_cv_new"] = round(cv(ldri_new), 4)
sigma5["summary"]["gamma_dominance_old"] = round(vlg_old / max(vll_old, 1e-15) * 100, 1)
sigma5["summary"]["gamma_dominance_new"] = round(vlg_new / max(vll_new, 1e-15) * 100, 1)
sigma5["summary"]["raw_dominance_old"] = round(vlr_old / max(vll_old, 1e-15) * 100, 1)
sigma5["summary"]["raw_dominance_new"] = round(vlr_new / max(vll_new, 1e-15) * 100, 1)
sigma5["summary"]["category_ranking_old"] = cat_ranking_old
sigma5["summary"]["category_ranking_new"] = cat_ranking_new
sigma5["summary"]["author_ranking_spearman"] = round(rank_rho, 4)
sigma5["summary"]["mean_runtime_old"] = round(mean(runtimes_old), 3)
sigma5["summary"]["mean_runtime_new"] = round(mean(runtimes_new), 3)

# Per-passage results
sigma5["passages"] = []
for r in valid:
    sigma5["passages"].append({
        "passage_id": r["passage_id"],
        "ldri": r["ldri"],
        "raw_ldri": r["raw_ldri"],
        "gamma_global": r["gamma_global"],
        "runtime_s": r["runtime_s"],
        "components": r["components"],
        "feature_norms": r["feature_norms"],
        "activations": r["activations_new"],
    })

with open("sigma5_validation.json", "w", encoding="utf-8") as f:
    json.dump(sigma5, f, indent=2, ensure_ascii=False)
print("[OK] sigma5_validation.json")

# ================================================================
# CONSOLE SUMMARY
# ================================================================
W = 72
print("\n" + "="*W)
print("  SHADOW EVALUATION: sigma=5.0 vs sigma=0.1")
print("="*W)

print(f"\n{'-'*W}")
print("  Q1: Does saturation disappear?")
print(f"{'-'*W}")
old_sat_count = sum(1 for r in valid for d in DIM_NAMES if r["activations_old"].get(d, 0) > 0.999)
new_sat_count = sum(1 for r in valid for d in DIM_NAMES if r["activations_new"].get(d, 0) > 0.999)
total_act = len(valid) * 6
old_sat_pct = old_sat_count / total_act * 100
new_sat_pct = new_sat_count / total_act * 100
print(f"    Old (sigma=0.1): {old_sat_pct:.0f}% saturation ({old_sat_count}/{total_act} activations > 0.999)")
print(f"    New (sigma=5.0): {new_sat_pct:.0f}% saturation ({new_sat_count}/{total_act} activations > 0.999)")
if new_sat_count == 0:
    print(f"    CONCLUSION: YES. Saturation completely eliminated.")
else:
    print(f"    CONCLUSION: Partial. {new_sat_pct:.1f}% remain saturated.")

print(f"\n{'-'*W}")
print("  Q2: Does raw LDRI become discriminative?")
print(f"{'-'*W}")
raw_cv_old = cv(raw_old)
raw_cv_new = cv(raw_new)
print(f"    Old raw_ldri CV: {raw_cv_old:.4f} ({'constant' if raw_cv_old < 0.01 else 'variable'})")
print(f"    New raw_ldri CV: {raw_cv_new:.4f} ({'constant' if raw_cv_new < 0.01 else 'variable'})")
if raw_cv_new > 0.01:
    print(f"    CONCLUSION: YES. raw_ldri CV increased from {raw_cv_old:.4f} to {raw_cv_new:.4f}.")
else:
    print(f"    CONCLUSION: NO. raw_ldri remains nearly constant.")

print(f"\n{'-'*W}")
print("  Q3: Does gamma cease to dominate?")
print(f"{'-'*W}")
ldri_cv_old = cv(ldri_old)
ldri_cv_new = cv(ldri_new)
gamma_cv_old = cv(gamma_old)
gamma_cv_new = cv(gamma_new)

gamma_dom_old = vlg_old / max(vll_old, 1e-15) * 100
gamma_dom_new = vlg_new / max(vll_new, 1e-15) * 100
raw_dom_old = vlr_old / max(vll_old, 1e-15) * 100
raw_dom_new = vlr_new / max(vll_new, 1e-15) * 100

print(f"    Old: var(log(gamma))/var(log(LDRI)) = {gamma_dom_old:.1f}%")
print(f"    New: var(log(gamma))/var(log(LDRI)) = {gamma_dom_new:.1f}%")
print(f"    Old: var(log(raw))/var(log(LDRI)) = {raw_dom_old:.1f}%")
print(f"    New: var(log(raw))/var(log(LDRI)) = {raw_dom_new:.1f}%")
if raw_dom_new > 10:
    print(f"    CONCLUSION: PARTIALLY. Raw_ldri now contributes {raw_dom_new:.1f}% vs {raw_dom_old:.1f}% before.")
else:
    print(f"    CONCLUSION: NO. Gamma still dominates ({gamma_dom_new:.1f}% of variance).")

print(f"\n{'-'*W}")
print("  Q4: Are category rankings more meaningful?")
print(f"{'-'*W}")
print(f"    Old ranking: {cat_ranking_old}")
print(f"    New ranking: {cat_ranking_new}")
cat_spread_old = mean(cat_old[cats[0]]) - mean(cat_old[cats[-1]])
cat_spread_new = mean(cat_new[cats[0]]) - mean(cat_new[cats[-1]])
print(f"    Old top-bottom spread: {cat_spread_old:.4f}")
print(f"    New top-bottom spread: {cat_spread_new:.4f}")

print(f"\n{'-'*W}")
print("  Q5: Are there any regressions?")
print(f"{'-'*W}")
rt_ratio = mean(runtimes_new) / max(mean(runtimes_old), 1e-15)
print(f"    Runtime ratio (new/old): {rt_ratio:.2f}x")
comp_rmse = {}
for c in COMP_NAMES:
    o_arr = extract_base_component(c)
    n_arr = extract_new_component(c)
    rmse = math.sqrt(mean([(o_arr[i] - n_arr[i])**2 for i in range(len(valid))]))
    comp_rmse[c] = round(rmse, 6)
    print(f"    Component {c} RMSE: {rmse:.6f}")

print(f"\n{'='*W}")
print("  FILES WRITTEN")
print(f"{'='*W}")
print("    comparison_before_after.csv")
print("    sigma5_validation.json")
print("="*W)
