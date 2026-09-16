"""LIT-DISC v1.1 -- Descriptive Statistical Analysis of Gold-Standard Evaluation Results.

Input:  data/gold_standard/litdisc_results.json
Output: analysis_summary.json, *statistics.csv, console summary
"""
import json, csv, math, os, sys
from collections import OrderedDict

# -- Load data --
with open("data/gold_standard/litdisc_results.json", encoding="utf-8") as f:
    data = json.load(f)

N = len(data)
# Fix: metadata word_count is accurate, but use real count from meta.wc if zero
for p in data:
    if p.get("word_count", 0) == 0:
        p["word_count"] = p.get("words", 0)

# -- Helper functions --
def mean(vals):
    return sum(vals) / len(vals)

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return (s[n//2-1] + s[n//2]) / 2
    return s[n//2]

def mode(vals):
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    maxc = max(counts.values())
    modes = [k for k, c in counts.items() if c == maxc]
    return modes if maxc > 1 else []

def variance(vals, ddof=1):
    m = mean(vals)
    return sum((x-m)**2 for x in vals) / (len(vals)-ddof)

def std(vals, ddof=1):
    return math.sqrt(variance(vals, ddof))

def quartiles(vals):
    s = sorted(vals)
    n = len(s)
    def q(p):
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    return q(0.25), q(0.50), q(0.75)

def ci95(vals):
    m = mean(vals)
    se = std(vals) / math.sqrt(len(vals))
    return m - 1.96*se, m + 1.96*se

def skewness(vals):
    m = mean(vals)
    s = std(vals, ddof=0)
    n = len(vals)
    if s == 0:
        return 0.0
    return (n / ((n-1)*(n-2))) * sum(((x-m)/s)**3 for x in vals) * (n-1) / n if n > 2 else 0.0

def kurtosis(vals):
    m = mean(vals)
    s = std(vals, ddof=0)
    n = len(vals)
    if s == 0:
        return 0.0
    num = sum((x-m)**4 for x in vals) / n
    return num / (s**4) - 3

def iqr(vals):
    q1, _, q3 = quartiles(vals)
    return q3 - q1

def outliers(vals, key_func=None):
    q1, _, q3 = quartiles(vals)
    iq = q3 - q1
    lower = q1 - 1.5*iq
    upper = q3 + 1.5*iq
    return lower, upper, [v for v in vals if v < lower or v > upper]

# -- Extract arrays --
ldris = [p["ldri"] for p in data]
raws = [p["raw_ldri"] for p in data]
gammas = [p["gamma_global"] for p in data]
runtimes = [p["runtime_s"] for p in data]
word_counts = [p["word_count"] for p in data]
categories = [p["category"] for p in data]
authors = [p["author"] for p in data]
titles = [p["title"] for p in data]
passage_ids = [p["passage_id"] for p in data]

comp_names = ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]
feat_names = ["semantic", "narrative", "interaction", "symbolic", "evolution", "emotion"]

comp_vals = {c: [p["components"][c] for p in data] for c in comp_names}
feat_vals = {f: [p["feature_means"][f] for p in data] for f in feat_names}

# ===============================================
# SECTION 1 -- Dataset Overview
# ===============================================
unique_cats = sorted(set(categories))
unique_authors = sorted(set(authors))
avg_wc = mean(word_counts)
avg_rt = mean(runtimes)
total_rt = sum(runtimes)
min_rt = min(runtimes)
max_rt = max(runtimes)

s1 = {
    "total_passages": N,
    "total_categories": len(unique_cats),
    "total_authors": len(unique_authors),
    "average_words_per_passage": round(avg_wc, 1),
    "average_runtime_s": round(avg_rt, 3),
    "total_processing_time_s": round(total_rt, 2),
    "minimum_runtime_s": round(min_rt, 3),
    "maximum_runtime_s": round(max_rt, 3),
}

# ===============================================
# SECTION 2 -- LDRI Statistics
# ===============================================
q1_ldri, med_ldri, q3_ldri = quartiles(ldris)
ldri_mode = mode([round(v, 4) for v in ldris])

s2 = {
    "mean": round(mean(ldris), 6),
    "median": round(med_ldri, 6),
    "mode": [round(v, 6) for v in ldri_mode] if ldri_mode else None,
    "std": round(std(ldris), 6),
    "variance": round(variance(ldris), 8),
    "min": round(min(ldris), 6),
    "max": round(max(ldris), 6),
    "range": round(max(ldris)-min(ldris), 6),
    "q1": round(q1_ldri, 6),
    "q3": round(q3_ldri, 6),
    "iqr": round(q3_ldri - q1_ldri, 6),
    "ci95_lower": round(ci95(ldris)[0], 6),
    "ci95_upper": round(ci95(ldris)[1], 6),
}

# ===============================================
# SECTION 3 -- Raw LDRI Statistics
# ===============================================
q1_raw, med_raw, q3_raw = quartiles(raws)

s3 = {
    "mean": round(mean(raws), 6),
    "median": round(med_raw, 6),
    "std": round(std(raws), 6),
    "variance": round(variance(raws), 8),
    "min": round(min(raws), 6),
    "max": round(max(raws), 6),
    "range": round(max(raws)-min(raws), 6),
    "q1": round(q1_raw, 6),
    "q3": round(q3_raw, 6),
    "iqr": round(q3_raw - q1_raw, 6),
    "ci95_lower": round(ci95(raws)[0], 6),
    "ci95_upper": round(ci95(raws)[1], 6),
}

# ===============================================
# SECTION 4 -- Confidence Statistics
# ===============================================
q1_g, med_g, q3_g = quartiles(gammas)
s4 = {
    "mean": round(mean(gammas), 6),
    "median": round(med_g, 6),
    "std": round(std(gammas), 6),
    "variance": round(variance(gammas), 8),
    "min": round(min(gammas), 6),
    "max": round(max(gammas), 6),
    "range": round(max(gammas)-min(gammas), 6),
    "q1": round(q1_g, 6),
    "q3": round(q3_g, 6),
    "iqr": round(q3_g - q1_g, 6),
    "ci95_lower": round(ci95(gammas)[0], 6),
    "ci95_upper": round(ci95(gammas)[1], 6),
}

# ===============================================
# SECTION 5 -- Runtime Statistics
# ===============================================
q1_rt, med_rt, q3_rt = quartiles(runtimes)
fastest_idx = runtimes.index(min_rt)
slowest_idx = runtimes.index(max_rt)
s5 = {
    "mean": round(avg_rt, 3),
    "median": round(med_rt, 3),
    "std": round(std(runtimes), 3),
    "min": round(min_rt, 3),
    "max": round(max_rt, 3),
    "fastest_passage": passage_ids[fastest_idx],
    "fastest_passage_title": f"{authors[fastest_idx]}, {titles[fastest_idx]}",
    "fastest_runtime_s": round(min_rt, 3),
    "slowest_passage": passage_ids[slowest_idx],
    "slowest_passage_title": f"{authors[slowest_idx]}, {titles[slowest_idx]}",
    "slowest_runtime_s": round(max_rt, 3),
}

# ===============================================
# SECTION 6 -- Component Statistics
# ===============================================
s6 = {}
for c in comp_names:
    v = comp_vals[c]
    s6[c] = {
        "mean": round(mean(v), 6),
        "std": round(std(v), 6),
        "min": round(min(v), 6),
        "max": round(max(v), 6),
    }
comp_means_ranked = sorted(comp_names, key=lambda c: s6[c]["mean"], reverse=True)
s6["component_rank_by_mean"] = comp_means_ranked
s6["highest_mean_component"] = comp_means_ranked[0]
s6["highest_mean_value"] = s6[comp_means_ranked[0]]["mean"]

# ===============================================
# SECTION 7 -- Feature Statistics
# ===============================================
s7 = {}
for f in feat_names:
    v = feat_vals[f]
    q1f, medf, q3f = quartiles(v)
    s7[f] = {
        "mean": round(mean(v), 4),
        "median": round(medf, 4),
        "std": round(std(v), 4),
        "variance": round(variance(v), 6),
        "min": round(min(v), 4),
        "max": round(max(v), 4),
    }
feat_ranked = sorted(feat_names, key=lambda f: s7[f]["mean"], reverse=True)
s7["feature_ranking"] = feat_ranked

# ===============================================
# SECTION 8 -- Category Analysis
# ===============================================
cat_data = {}
for cat in unique_cats:
    items = [p for p in data if p["category"] == cat]
    cat_data[cat] = {
        "count": len(items),
        "mean_ldri": round(mean([p["ldri"] for p in items]), 6),
        "mean_raw_ldri": round(mean([p["raw_ldri"] for p in items]), 6),
        "mean_gamma": round(mean([p["gamma_global"] for p in items]), 6),
        "mean_runtime": round(mean([p["runtime_s"] for p in items]), 3),
        "mean_feature_semantic": round(mean([p["feature_means"]["semantic"] for p in items]), 4),
        "mean_feature_narrative": round(mean([p["feature_means"]["narrative"] for p in items]), 4),
        "mean_feature_interaction": round(mean([p["feature_means"]["interaction"] for p in items]), 4),
        "mean_feature_symbolic": round(mean([p["feature_means"]["symbolic"] for p in items]), 4),
        "mean_feature_evolution": round(mean([p["feature_means"]["evolution"] for p in items]), 4),
        "mean_feature_emotion": round(mean([p["feature_means"]["emotion"] for p in items]), 4),
    }
cat_ranked = sorted(unique_cats, key=lambda c: cat_data[c]["mean_ldri"], reverse=True)
s8 = {
    "categories": cat_data,
    "highest_scoring_category": cat_ranked[0],
    "lowest_scoring_category": cat_ranked[-1],
}

# ===============================================
# SECTION 9 -- Author Analysis
# ===============================================
author_data = {}
for p in data:
    a = p["author"]
    if a not in author_data:
        author_data[a] = {"passage_id": p["passage_id"], "title": p["title"],
                          "category": p["category"], "ldris": [], "runtimes": [],
                          "feature_sums": {f: [] for f in feat_names}}
    author_data[a]["ldris"].append(p["ldri"])
    author_data[a]["runtimes"].append(p["runtime_s"])
    for f in feat_names:
        author_data[a]["feature_sums"][f].append(p["feature_means"][f])

author_stats = []
for a, ad in author_data.items():
    author_stats.append({
        "author": a,
        "passage_id": ad["passage_id"],
        "title": ad["title"],
        "category": ad["category"],
        "ldri": round(mean(ad["ldris"]), 6),
        "runtime_s": round(mean(ad["runtimes"]), 3),
        "feature_semantic": round(mean(ad["feature_sums"]["semantic"]), 4),
        "feature_narrative": round(mean(ad["feature_sums"]["narrative"]), 4),
        "feature_interaction": round(mean(ad["feature_sums"]["interaction"]), 4),
        "feature_symbolic": round(mean(ad["feature_sums"]["symbolic"]), 4),
        "feature_evolution": round(mean(ad["feature_sums"]["evolution"]), 4),
        "feature_emotion": round(mean(ad["feature_sums"]["emotion"]), 4),
    })
author_stats.sort(key=lambda a: a["ldri"], reverse=True)
s9 = {
    "author_ranking": author_stats,
    "top_5": [a["author"] for a in author_stats[:5]],
    "bottom_5": [a["author"] for a in author_stats[-5:]],
}

# ===============================================
# SECTION 10 -- Outlier Analysis
# ===============================================
ldri_lower, ldri_upper, ldri_out = outliers(ldris)
rt_lower, rt_upper, rt_out = outliers(runtimes)
gamma_lower, gamma_upper, gamma_out = outliers(gammas)

ldri_out_details = [{"passage_id": passage_ids[i], "value": ldris[i]}
                    for i in range(N) if ldris[i] < ldri_lower or ldris[i] > ldri_upper]
rt_out_details = [{"passage_id": passage_ids[i], "value": runtimes[i]}
                  for i in range(N) if runtimes[i] < rt_lower or runtimes[i] > rt_upper]
gamma_out_details = [{"passage_id": passage_ids[i], "value": gammas[i]}
                     for i in range(N) if gammas[i] < gamma_lower or gammas[i] > gamma_upper]

s10 = {
    "ldri": {
        "lower_fence": round(ldri_lower, 6),
        "upper_fence": round(ldri_upper, 6),
        "outlier_count": len(ldri_out_details),
        "outliers": ldri_out_details,
    },
    "runtime": {
        "lower_fence": round(rt_lower, 3),
        "upper_fence": round(rt_upper, 3),
        "outlier_count": len(rt_out_details),
        "outliers": rt_out_details,
    },
    "gamma_global": {
        "lower_fence": round(gamma_lower, 6),
        "upper_fence": round(gamma_upper, 6),
        "outlier_count": len(gamma_out_details),
        "outliers": gamma_out_details,
    },
}

# ===============================================
# SECTION 11 -- Distribution Analysis
# ===============================================
def dist_shape(vals):
    sk = skewness(vals)
    ku = kurtosis(vals)
    if abs(sk) < 0.5 and abs(ku) < 1:
        shape = "approximately normal"
    elif sk > 0.5:
        shape = "positively skewed"
    elif sk < -0.5:
        shape = "negatively skewed"
    else:
        shape = "near-symmetric"
    if ku > 2:
        shape += " (leptokurtic)"
    elif ku < -1:
        shape += " (platykurtic)"
    return shape, round(sk, 4), round(ku, 4)

s11 = {}
for name, vals in [("ldri", ldris), ("raw_ldri", raws), ("gamma_global", gammas),
                    ("runtime_s", runtimes)]:
    s, sk, ku = dist_shape(vals)
    s11[name] = {"shape": s, "skewness": sk, "kurtosis": ku}

# ===============================================
# SECTION 12 -- Executive Summary
# ===============================================
best_category = s8["highest_scoring_category"]
worst_category = s8["lowest_scoring_category"]
dominant_comp = comp_means_ranked[0]

s12 = {
    "average_ldri": round(mean(ldris), 6),
    "ldri_range": [round(min(ldris), 6), round(max(ldris), 6)],
    "average_gamma_confidence": round(mean(gammas), 6),
    "average_runtime_s": round(avg_rt, 3),
    "runtime_range_s": [round(min_rt, 3), round(max_rt, 3)],
    "feature_trend": f"Ranked by mean: {', '.join(feat_ranked)}",
    "highest_feature": feat_ranked[0],
    "lowest_feature": feat_ranked[-1],
    "category_trend": f"Highest: {best_category}, Lowest: {worst_category}",
    "category_ranking": cat_ranked,
    "author_trend": f"Top: {author_stats[0]['author']} (LDRI={author_stats[0]['ldri']}), "
                    f"Bottom: {author_stats[-1]['author']} (LDRI={author_stats[-1]['ldri']})",
    "dominant_component": dominant_comp,
    "component_mean_contributions": {c: s6[c]["mean"] for c in comp_names},
    "notable_observations": [
        f"Uncertainty dominates all component decompositions (mean={s6['uncertainty']['mean']:.4f}, min={s6['uncertainty']['min']:.4f})",
        f"Intrinsic richness is the second-largest component (mean={s6['intrinsic']['mean']:.4f})",
        f"Resonance and memory contributions are near-zero (mean={s6['resonance']['mean']:.6f}, mean={s6['memory']['mean']:.6f})",
        f"Raw LDRI is nearly constant (mean={mean(raws):.4f}, CV={std(raws)/mean(raws)*100:.1f}%)",
        f"LDRI variation is driven entirely by gamma_global (CV={std(gammas)/mean(gammas)*100:.1f}%)",
        f"Semantic density scores highest among features (mean={s7['semantic']['mean']:.3f})",
        f"Emotion scores lowest among features (mean={s7['emotion']['mean']:.3f})",
        f"Runtime averages {avg_rt:.2f}s per passage (range {min_rt:.2f}s--{max_rt:.2f}s)",
        f"SNC category shows highest mean LDRI ({cat_data['SNC']['mean_ldri']:.6f})",
        f"MNC category shows lowest mean LDRI ({cat_data['MNC']['mean_ldri']:.6f})",
    ],
}

# ===============================================
# Assemble full summary
# ===============================================
summary = OrderedDict()
summary["section_1_dataset_overview"] = s1
summary["section_2_ldri_statistics"] = s2
summary["section_3_raw_ldri_statistics"] = s3
summary["section_4_confidence_statistics"] = s4
summary["section_5_runtime_statistics"] = s5
summary["section_6_component_statistics"] = s6
summary["section_7_feature_statistics"] = s7
summary["section_8_category_analysis"] = s8
summary["section_9_author_analysis"] = s9
summary["section_10_outlier_analysis"] = s10
summary["section_11_distribution_analysis"] = s11
summary["section_12_executive_summary"] = s12

# -- Write analysis_summary.json --
with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("[OK] analysis_summary.json")

# -- Write descriptive_statistics.csv --
with open("descriptive_statistics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Mean", "Median", "Std", "Variance", "Min", "Max", "Range", "Q1", "Q3", "IQR", "CI95_Lo", "CI95_Hi"])
    for label, sd in [("LDRI", s2), ("Raw_LDRI", s3), ("Gamma_Global", s4), ("Runtime_s", s5)]:
        if label == "Runtime_s":
            w.writerow([label, sd["mean"], sd["median"], sd["std"], "", sd["min"], sd["max"], "", "", "", "", "", ""])
        else:
            w.writerow([label, sd["mean"], sd["median"], sd["std"], sd["variance"],
                       sd["min"], sd["max"], sd.get("range",""), sd.get("q1",""), sd.get("q3",""),
                       sd.get("iqr",""), sd.get("ci95_lower",""), sd.get("ci95_upper","")])
    # Features
    for f in feat_names:
        fs = s7[f]
        w.writerow([f"Feature_{f}", fs["mean"], fs["median"], fs["std"], fs["variance"],
                    fs["min"], fs["max"], "", "", "", "", "", ""])
print("[OK] descriptive_statistics.csv")

# -- Write category_statistics.csv --
with open("category_statistics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Category", "N", "Mean_LDRI", "Mean_Raw_LDRI", "Mean_Gamma",
                "Mean_Runtime", "Mean_Semantic", "Mean_Narrative", "Mean_Interaction",
                "Mean_Symbolic", "Mean_Evolution", "Mean_Emotion"])
    for cat in unique_cats:
        cd = cat_data[cat]
        w.writerow([cat, cd["count"], cd["mean_ldri"], cd["mean_raw_ldri"], cd["mean_gamma"],
                    cd["mean_runtime"], cd["mean_feature_semantic"], cd["mean_feature_narrative"],
                    cd["mean_feature_interaction"], cd["mean_feature_symbolic"],
                    cd["mean_feature_evolution"], cd["mean_feature_emotion"]])
print("[OK] category_statistics.csv")

# -- Write author_statistics.csv --
with open("author_statistics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Rank", "Author", "Passage", "Category", "LDRI", "Runtime",
                "Semantic", "Narrative", "Interaction", "Symbolic", "Evolution", "Emotion"])
    for i, a in enumerate(author_stats, 1):
        w.writerow([i, a["author"], a["passage_id"], a["category"], a["ldri"], a["runtime_s"],
                    a["feature_semantic"], a["feature_narrative"], a["feature_interaction"],
                    a["feature_symbolic"], a["feature_evolution"], a["feature_emotion"]])
print("[OK] author_statistics.csv")

# -- Write feature_statistics.csv --
with open("feature_statistics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Feature", "Mean", "Median", "Std", "Variance", "Min", "Max"])
    for fname in feat_ranked:
        fs = s7[fname]
        w.writerow([fname, fs["mean"], fs["median"], fs["std"], fs["variance"], fs["min"], fs["max"]])
print("[OK] feature_statistics.csv")

# ===============================================
# Console Summary
# ===============================================
W = 72
print("\n" + "="*W)
print("  LIT-DISC v1.1 -- Gold-Standard Evaluation: Descriptive Statistics")
print("="*W)

print(f"\n{'-'*W}")
print("  SECTION 1 -- Dataset Overview")
print(f"{'-'*W}")
print(f"  Total passages:          {s1['total_passages']}")
print(f"  Total categories:        {s1['total_categories']} ({', '.join(unique_cats)})")
print(f"  Total authors:           {s1['total_authors']}")
print(f"  Average words/passage:   {s1['average_words_per_passage']:.1f}")
print(f"  Average runtime:         {s1['average_runtime_s']:.2f}s")
print(f"  Total processing time:   {s1['total_processing_time_s']:.1f}s")
print(f"  Runtime range:           {s1['minimum_runtime_s']:.2f}s -- {s1['maximum_runtime_s']:.2f}s")

print(f"\n{'-'*W}")
print("  SECTION 2 -- LDRI Statistics")
print(f"{'-'*W}")
print(f"  Mean:     {s2['mean']:.6f}")
print(f"  Median:   {s2['median']:.6f}")
print(f"  Std:      {s2['std']:.6f}")
print(f"  Variance: {s2['variance']:.8f}")
print(f"  Min:      {s2['min']:.6f}   Max: {s2['max']:.6f}")
print(f"  Range:    {s2['range']:.6f}")
print(f"  Q1:       {s2['q1']:.6f}   Q3: {s2['q3']:.6f}   IQR: {s2['iqr']:.6f}")
print(f"  95% CI:   [{s2['ci95_lower']:.6f}, {s2['ci95_upper']:.6f}]")

print(f"\n{'-'*W}")
print("  SECTION 3 -- Raw LDRI vs Final LDRI")
print(f"{'-'*W}")
print(f"  Raw LDRI -- Mean: {s3['mean']:.6f}  Std: {s3['std']:.6f}  Min: {s3['min']:.6f}  Max: {s3['max']:.6f}")
print(f"  Final LDRI -- Mean: {s2['mean']:.6f}  Std: {s2['std']:.6f}  Min: {s2['min']:.6f}  Max: {s2['max']:.6f}")
print(f"  Ratio (final/raw): {s2['mean']/s3['mean']:.6f}")
print(f"  Effect of confidence weighting:")
print(f"    Raw LDRI is near-constant (CV={std(raws)/mean(raws)*100:.1f}%)")
print(f"    Gamma_global multiplies it down to final LDRI (CV={std(gammas)/mean(gammas)*100:.1f}%)")
print(f"    LDRI ~= raw x gamma, where gamma varies by {std(gammas)/mean(gammas)*100:.1f}% CV")

print(f"\n{'-'*W}")
print("  SECTION 4 -- Confidence (gamma_global) Statistics")
print(f"{'-'*W}")
print(f"  Mean: {s4['mean']:.6f}   Median: {s4['median']:.6f}   Std: {s4['std']:.6f}")
print(f"  Min: {s4['min']:.6f}   Max: {s4['max']:.6f}")
print(f"  Q1: {s4['q1']:.6f}   Q3: {s4['q3']:.6f}   IQR: {s4['iqr']:.6f}")

print(f"\n{'-'*W}")
print("  SECTION 5 -- Runtime Statistics")
print(f"{'-'*W}")
print(f"  Mean: {s5['mean']:.2f}s   Median: {s5['median']:.2f}s   Std: {s5['std']:.2f}s")
print(f"  Fastest: {s5['fastest_passage']} ({s5['fastest_passage_title']}) -- {s5['fastest_runtime_s']:.2f}s")
print(f"  Slowest: {s5['slowest_passage']} ({s5['slowest_passage_title']}) -- {s5['slowest_runtime_s']:.2f}s")

print(f"\n{'-'*W}")
print("  SECTION 6 -- Component Statistics")
print(f"{'-'*W}")
print(f"  {'Component':>15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for c in comp_means_ranked:
    cs = s6[c]
    print(f"  {c:>15} {cs['mean']:>10.6f} {cs['std']:>10.6f} {cs['min']:>10.6f} {cs['max']:>10.6f}")
print(f"  Highest mean component: {s6['highest_mean_component']} ({s6['highest_mean_value']:.6f})")

print(f"\n{'-'*W}")
print("  SECTION 7 -- Feature Statistics (ranked by mean)")
print(f"{'-'*W}")
print(f"  {'Rank':>4} {'Feature':>15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
print(f"  {'-'*4} {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
for i, fname in enumerate(feat_ranked, 1):
    fs = s7[fname]
    print(f"  {i:>4} {fname:>15} {fs['mean']:>8.3f} {fs['median']:>8.3f} {fs['std']:>8.3f} {fs['min']:>6.3f} {fs['max']:>6.3f}")

print(f"\n{'-'*W}")
print("  SECTION 8 -- Category Analysis")
print(f"{'-'*W}")
print(f"  {'Category':>8} {'N':>3} {'LDRI':>10} {'gamma':>8} {'Sem':>6} {'Nar':>6} {'Int':>6} {'Sym':>6} {'Evo':>6} {'Emo':>6}")
print(f"  {'-'*8} {'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for cat in cat_ranked:
    cd = cat_data[cat]
    print(f"  {cat:>8} {cd['count']:>3} {cd['mean_ldri']:>10.6f} {cd['mean_gamma']:>8.6f} "
          f"{cd['mean_feature_semantic']:>6.3f} {cd['mean_feature_narrative']:>6.3f} "
          f"{cd['mean_feature_interaction']:>6.3f} {cd['mean_feature_symbolic']:>6.3f} "
          f"{cd['mean_feature_evolution']:>6.3f} {cd['mean_feature_emotion']:>6.3f}")
print(f"  Highest: {best_category}   Lowest: {worst_category}")

print(f"\n{'-'*W}")
print("  SECTION 9 -- Author Rankings (top/bottom 5 by LDRI)")
print(f"{'-'*W}")
print(f"  {'Rank':>4} {'Author':>22} {'LDRI':>10} {'Cat':>6}")
print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*6}")
for i, a in enumerate(author_stats, 1):
    rank_str = f"{'TOP' if i <= 5 else 'BOT' if i > len(author_stats)-5 else ''}"
    print(f"  {i:>4} {a['author']:>22} {a['ldri']:>10.6f} {a['category']:>6}")
    if i == 5:
        print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*6}")

print(f"\n{'-'*W}")
print("  SECTION 10 -- Outlier Analysis")
print(f"{'-'*W}")
for metric, s10d in [("LDRI", s10["ldri"]), ("Runtime", s10["runtime"]), ("Gamma", s10["gamma_global"])]:
    print(f"  {metric}: fence [{s10d['lower_fence']:.6f}, {s10d['upper_fence']:.6f}], "
          f"{s10d['outlier_count']} outlier(s)")
    for o in s10d["outliers"]:
        print(f"    - {o['passage_id']}: {o['value']:.6f}")

print(f"\n{'-'*W}")
print("  SECTION 11 -- Distribution Analysis")
print(f"{'-'*W}")
for metric, dd in s11.items():
    print(f"  {metric:>15}: {dd['shape']:40s}  skew={dd['skewness']:+.4f}  kurt={dd['kurtosis']:+.4f}")

print(f"\n{'-'*W}")
print("  SECTION 12 -- Executive Summary")
print(f"{'-'*W}")
for obs in s12["notable_observations"]:
    print(f"  * {obs}")
print("="*W)
print("  Output files generated:")
print("    analysis_summary.json")
print("    descriptive_statistics.csv")
print("    category_statistics.csv")
print("    author_statistics.csv")
print("    feature_statistics.csv")
print("="*W)
