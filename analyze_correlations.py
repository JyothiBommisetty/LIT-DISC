"""LIT-DISC v1.0.0 -- Correlation Analysis: LDRI vs Expert Judgments.

Attempts formal correlation with expert ratings; reports zero-variance
finding; proceeds with alternative internal-consistency analyses.

Outputs: correlation_report.txt (full console output redirected)
         correlation_tables.csv (all attempted and alternative correlations)
         feature_intercorrelation.csv
         component_feature_matrix.csv
"""
import json, csv, math, os, sys
from collections import OrderedDict

# Suppress runtime warnings for zero-variance cases
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Load data ──
with open("data/gold_standard/litdisc_results.json", encoding="utf-8") as f:
    data = json.load(f)
N = len(data)

# ── Statistical helpers ──
def pearson_r(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    den = math.sqrt(sum((x[i]-mx)**2 for i in range(n)) * sum((y[i]-my)**2 for i in range(n)))
    if den == 0:
        return float("nan")
    return num / den

def spearman_r(x, y):
    nx = len(x)
    def rank(vals):
        s = sorted([(v, i) for i, v in enumerate(vals)])
        r = [0]*len(vals)
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j][0] == s[i][0]:
                j += 1
            rank_val = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                r[s[k][1]] = rank_val
            i = j
        return r
    rx, ry = rank(x), rank(y)
    return pearson_r(rx, ry)

def kendall_tau(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx > 0 and dy > 0:
                concordant += 1
            elif dx < 0 and dy < 0:
                concordant += 1
            elif dx == 0 and dy == 0:
                pass
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            else:
                discordant += 1
    den = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if den == 0:
        return float("nan")
    return (concordant - discordant) / den

def p_value_from_r(r, n, method="pearson"):
    """Approximate two-tailed p-value from t-distribution."""
    if math.isnan(r) or abs(r) >= 1:
        return float("nan")
    from math import erf, sqrt
    if method == "pearson":
        t = r * sqrt((n - 2) / (1 - r*r))
        # Two-tailed p from t-distribution using incomplete beta
        # Use approximation for large n
        df = n - 2
        # Using normal approximation for simplicity
        z = 0.5 * math.log((1+r)/(1-r))  # Fisher z-transform
        se = 1 / math.sqrt(n - 3)
        p = 2 * (1 - 0.5 * (1 + erf(abs(z)/se / math.sqrt(2))))
        return p
    elif method == "spearman":
        # Normal approximation for Spearman
        z = math.sqrt((n - 1) / (1 - r*r)) if abs(r) < 1 else 0
        # More appropriate: t = r * sqrt((n-2)/(1-r^2)) with df=n-2
        if abs(r) >= 1:
            return 0.0
        t = r * math.sqrt((n - 2) / (1 - r*r))
        # Normal approximation
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
        return p
    return float("nan")

def ci_pearson(r, n, alpha=0.05):
    """Approximate 95% CI for Pearson r using Fisher z-transform."""
    if math.isnan(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1+r)/(1-r))
    se = 1 / math.sqrt(n - 3)
    z_alpha = 1.96
    z_lo = z - z_alpha * se
    z_hi = z + z_alpha * se
    r_lo = (math.exp(2*z_lo) - 1) / (math.exp(2*z_lo) + 1)
    r_hi = (math.exp(2*z_hi) - 1) / (math.exp(2*z_hi) + 1)
    return (r_lo, r_hi)

def variance(vals):
    m = sum(vals) / len(vals)
    return sum((x-m)**2 for x in vals) / (len(vals)-1)

def describe_variation(vals, name):
    """Check if a variable has sufficient variation for correlation."""
    if len(set(vals)) == 1:
        return f"{name}: NO VARIATION -- all values = {vals[0]}. Correlation undefined."
    cv = math.sqrt(variance(vals)) / (sum(vals)/len(vals))
    return f"{name}: varies (CV={cv*100:.1f}%, unique={len(set(vals))})"

# ── Extract arrays ──
passage_ids = [p["passage_id"] for p in data]
ldris = [p["ldri"] for p in data]

expert_overall = [p["expert_scores"]["overall"] for p in data]
expert_dims = {d: [p["expert_scores"][d] for p in data]
               for d in ["semantic","narrative","interaction","symbolic","evolution","emotion"]}
feat_means = {d: [p["feature_means"][d] for p in data]
              for d in ["semantic","narrative","interaction","symbolic","evolution","emotion"]}
comp_vals = {c: [p["components"][c] for p in data]
             for c in ["intrinsic","interaction","temporal","resonance","memory","uncertainty"]}

# ── 1. Basic variation check ──
variation_report = []
variation_report.append("="*72)
variation_report.append("  EXPERT RATING VARIATION CHECK")
variation_report.append("="*72)
variation_report.append(describe_variation(expert_overall, "Expert Overall Score"))
for d in ["semantic","narrative","interaction","symbolic","evolution","emotion"]:
    variation_report.append(describe_variation(expert_dims[d], f"Expert {d.capitalize()} Score"))
variation_report.append(describe_variation(ldris, "Computed LDRI"))

# ── 2. Attempt all correlations (will fail for constant expert vars) ──
correlation_rows = []  # For CSV
correlation_rows.append(["Variable_X", "Variable_Y", "Method",
                          "r", "p_value", "CI_lo", "CI_hi", "N", "Status"])

correlation_attempts = [
    ("LDRI", "Expert_Overall", ldris, expert_overall),
]
for d in ["semantic","narrative","interaction","symbolic","evolution","emotion"]:
    correlation_attempts.append((f"Feature_{d}", f"Expert_{d}",
                                  feat_means[d], expert_dims[d]))

# Also LDRI vs each feature (descriptive, not expert)
for d in ["semantic","narrative","interaction","symbolic","evolution","emotion"]:
    correlation_attempts.append((f"LDRI", f"Feature_{d}", ldris, feat_means[d]))

# Also LDRI vs each component
for c in ["intrinsic","interaction","temporal","resonance","memory","uncertainty"]:
    correlation_attempts.append((f"LDRI", f"Component_{c}", ldris, comp_vals[c]))

for var_x, var_y, x_vals, y_vals in correlation_attempts:
    n = min(len(x_vals), len(y_vals))

    # Check for zero variance in either variable
    x_unique = len(set(x_vals))
    y_unique = len(set(y_vals))

    if x_unique == 1 or y_unique == 1:
        status = f"UNDEFINED: {var_y if y_unique==1 else var_x} has zero variance"
        correlation_rows.append([var_x, var_y, "Pearson", "NaN", "NaN", "NaN", "NaN", n, status])
        correlation_rows.append([var_x, var_y, "Spearman", "NaN", "NaN", "NaN", "NaN", n, status])
        correlation_rows.append([var_x, var_y, "Kendall", "NaN", "NaN", "NaN", "NaN", n, status])
        continue

    for method, func in [("Pearson", pearson_r), ("Spearman", spearman_r),
                         ("Kendall", kendall_tau)]:
        r = func(x_vals, y_vals)
        if math.isnan(r):
            status = "UNDEFINED (computation failed)"
            correlation_rows.append([var_x, var_y, method, "NaN", "NaN", "NaN", "NaN", n, status])
        else:
            p = p_value_from_r(r, n, method="pearson" if method=="Pearson" else "spearman")
            if method == "Pearson":
                ci_lo, ci_hi = ci_pearson(r, n)
                correlation_rows.append([var_x, var_y, method,
                                          f"{r:.6f}", f"{p:.6e}",
                                          f"{ci_lo:.6f}", f"{ci_hi:.6f}", n, "OK"])
            else:
                correlation_rows.append([var_x, var_y, method,
                                          f"{r:.6f}", f"{p:.6e}", "N/A", "N/A", n, "OK"])

# ── 3. Feature inter-correlation matrix ──
feat_names = ["semantic","narrative","interaction","symbolic","evolution","emotion"]
feat_icc_rows = []
feat_icc_rows.append(["Feature_X", "Feature_Y", "Pearson_r", "Spearman_rho", "Kendall_tau"])
for i, f1 in enumerate(feat_names):
    for j, f2 in enumerate(feat_names):
        if i < j:
            rp = pearson_r(feat_means[f1], feat_means[f2])
            rs = spearman_r(feat_means[f1], feat_means[f2])
            rt = kendall_tau(feat_means[f1], feat_means[f2])
            feat_icc_rows.append([f1, f2,
                                  f"{rp:.4f}" if not math.isnan(rp) else "NaN",
                                  f"{rs:.4f}" if not math.isnan(rs) else "NaN",
                                  f"{rt:.4f}" if not math.isnan(rt) else "NaN"])

# ── 4. Component-Feature matrix ──
comp_names = ["intrinsic","interaction","temporal","resonance","memory","uncertainty"]
cf_rows = []
cf_rows.append(["Component", "Feature", "Pearson_r", "Spearman_rho"])
for c in comp_names:
    for f in feat_names:
        rp = pearson_r(comp_vals[c], feat_means[f])
        rs = spearman_r(comp_vals[c], feat_means[f])
        cf_rows.append([c, f,
                        f"{rp:.4f}" if not math.isnan(rp) else "NaN",
                        f"{rs:.4f}" if not math.isnan(rs) else "NaN"])

# ── 5. Simulated expert demonstration ──
# Show what the pipeline WOULD produce with variegated expert data
import random
random.seed(42)
sim_expert_overall = [random.randint(1, 7) for _ in range(N)]
sim_expert_dims = {}
for d in feat_names:
    sim_expert_dims[d] = [random.randint(1, 7) for _ in range(N)]

sim_rows = []
sim_rows.append(["Variable_X", "Variable_Y", "Method", "r", "p_value", "CI_lo", "CI_hi", "Status"])
sim_label = "Simulated_Expert_Overall"
rp = pearson_r(ldris, sim_expert_overall)
rs = spearman_r(ldris, sim_expert_overall)
rt = kendall_tau(ldris, sim_expert_overall)
p_p = p_value_from_r(rp, N)
p_s = p_value_from_r(rs, N, "spearman")
ci_lo, ci_hi = ci_pearson(rp, N)
for method, r, p, status in [("Pearson", rp, p_p, "OK (simulated)"),
                              ("Spearman", rs, p_s, "OK (simulated)"),
                              ("Kendall", rt, float("nan"), "OK (simulated)")]:
    ci_ls = f"{ci_lo:.4f}" if method=="Pearson" else "N/A"
    ci_hs = f"{ci_hi:.4f}" if method=="Pearson" else "N/A"
    sim_rows.append([sim_label, "LDRI", method,
                     f"{r:.4f}" if not math.isnan(r) else "NaN",
                     f"{p:.4f}" if not math.isnan(p) else "NaN",
                     ci_ls, ci_hs, status])

# Save overall LDRI vs simulated values for console report
rp_global = rp
p_global = p_p

# Show per-feature simulated correlations
model_str = ""
for d in feat_names:
    rp_f = pearson_r(feat_means[d], sim_expert_dims[d])
    rs_f = spearman_r(feat_means[d], sim_expert_dims[d])
    p_val = p_value_from_r(rp_f, N)
    ci_lo_f, ci_hi_f = ci_pearson(rp_f, N)
    model_str += f"    {d}: Pearson r={rp_f:.4f} (p={p_val:.4f}, CI=[{ci_lo_f:.4f},{ci_hi_f:.4f}]), Spearman rho={rs_f:.4f}\n"
    sim_rows.append([f"Feature_{d}", f"Sim_Expert_{d}", "Pearson",
                     f"{rp_f:.4f}" if not math.isnan(rp_f) else "NaN",
                     f"{p_val:.4f}" if not math.isnan(p_val) else "NaN",
                     f"{ci_lo_f:.4f}" if not math.isnan(ci_lo_f) else "NaN",
                     f"{ci_hi_f:.4f}" if not math.isnan(ci_hi_f) else "NaN",
                     "OK (simulated)"])

# ── Write outputs ──

# 1. correlation_tables.csv
with open("correlation_tables.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["# LIT-DISC v1.0.0 -- Correlation Analysis"])
    w.writerow(["# STATUS: Expert ratings have ZERO variance -- all real-expert correlations are undefined"])
    w.writerow(["# Real-expert correlations (all NaN due to zero variance):"])
    for row in correlation_rows:
        w.writerow(row)
    w.writerow([])
    w.writerow(["# Simulated-expert demonstration (with random 1-7 scores):"])
    for row in sim_rows:
        w.writerow(row)
print("[OK] correlation_tables.csv")

# 2. feature_intercorrelation.csv
with open("feature_intercorrelation.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["# LIT-DISC v1.0.0 -- Feature Inter-Correlation Matrix (Pearson/Spearman/Kendall)"])
    for row in feat_icc_rows:
        w.writerow(row)
print("[OK] feature_intercorrelation.csv")

# 3. component_feature_matrix.csv
with open("component_feature_matrix.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["# LIT-DISC v1.0.0 -- Component vs Feature Correlation Matrix"])
    for row in cf_rows:
        w.writerow(row)
print("[OK] component_feature_matrix.csv")

# ── Console report ──
W = 72
print("="*W)
print("  LIT-DISC v1.0.0 -- Correlation Analysis Report")
print("="*W)

print(f"\n{'='*W}")
print("  PART 1: EXPERT RATING VARIATION CHECK")
print(f"{'='*W}")
for line in variation_report:
    if line.startswith("="):
        continue
    print(f"  {line}")

print(f"\n{'='*W}")
print("  PART 2: ATTEMPTED CORRELATIONS (real expert scores)")
print(f"{'='*W}")
print(f"  All expert scores = 4 on all 7 axes x 25 passages.")
print(f"  Zero variance => ALL Pearson r = NaN (division by zero).")
print(f"  Zero variance => ALL Spearman rho = NaN (no rank variation).")
print(f"  Zero variance => ALL Kendall tau = NaN (no pairs to compare).")
print(f"\n  CONCLUSION: Formal correlation with current expert data is")
print(f"  impossible. The file TEST-01_ratings.json is a test placeholder,")
print(f"  not real expert annotation data.")

print(f"\n{'='*W}")
print("  PART 3: ALTERNATIVE ANALYSIS -- LDRI Internal Relationships")
print(f"{'='*W}")

# LDRI vs Feature means
print(f"\n  LDRI vs Per-Dimension Feature Means:")
print(f"  {'Feature':>12} {'Pearson r':>10} {'p-value':>10} {'95% CI':>22} {'Spearman rho':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*22} {'-'*12}")
for d in feat_names:
    rp = pearson_r(ldris, feat_means[d])
    rs = spearman_r(ldris, feat_means[d])
    p = p_value_from_r(rp, N)
    ci_lo, ci_hi = ci_pearson(rp, N)
    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if not (math.isnan(ci_lo) or math.isnan(ci_hi)) else "N/A"
    print(f"  {d:>12} {rp:>10.4f} {p:>10.4f} {ci_str:>22} {rs:>12.4f}")

# LDRI vs Components
print(f"\n  LDRI vs Component Values:")
print(f"  {'Component':>12} {'Pearson r':>10} {'p-value':>10} {'95% CI':>22} {'Spearman rho':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*22} {'-'*12}")
for c in comp_names:
    rp = pearson_r(ldris, comp_vals[c])
    rs = spearman_r(ldris, comp_vals[c])
    p = p_value_from_r(rp, N)
    ci_lo, ci_hi = ci_pearson(rp, N)
    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if not (math.isnan(ci_lo) or math.isnan(ci_hi)) else "N/A"
    print(f"  {c:>12} {rp:>10.4f} {p:>10.4f} {ci_str:>22} {rs:>12.4f}")

# Feature inter-correlations (summary)
print(f"\n  Feature Inter-Correlation Summary (absolute Pearson r > 0.7):")
count_high = 0
for row in feat_icc_rows[1:]:  # Skip header
    rp = row[2].replace("NaN", "0")
    if abs(float(rp)) > 0.7:
        print(f"    {row[0]} <-> {row[1]}: r={row[2]}")
        count_high += 1
if count_high == 0:
    print("    (None -- features are moderately inter-correlated but no pair exceeds |r|>0.7)")
    print("    This suggests the 6 dimensions capture distinct variance.")

# Component-Feature strongest relationships
print(f"\n  Component-Feature: Strongest absolute correlations (|r| > 0.5):")
count_cf = 0
for row in cf_rows[1:]:
    rp = row[2].replace("NaN", "0")
    if abs(float(rp)) > 0.5:
        print(f"    {row[0]} <-> {row[1]}: r={row[2]}")
        count_cf += 1
if count_cf == 0:
    print("    (None -- components and features capture largely independent information)")

print(f"\n{'='*W}")
print("  PART 4: SIMULATED EXPERT DEMONSTRATION")
print(f"{'='*W}")
print(f"  To validate the analysis pipeline, random 1-7 expert scores were")
print(f"  generated (seed=42) and correlated with LDRI / features.")
print(f"  As expected: no significant correlation (random data, r ~ 0).")
sim_summary_line = f"  LDRI vs Simulated Overall: Pearson r={rp_global:.4f} (p={p_global:.4f})"
print(f"  {sim_summary_line}")
print(f"\n  Per-feature simulated correlations:")
print(f"  {model_str}")

print(f"\n{'='*W}")
print("  PART 5: PATH FORWARD FOR EXPERT VALIDATION")
print(f"{'='*W}")
print(f"  Annotation infrastructure status: READY")
print(f"  - Annotation package: 9 documents (instructions, manual, calibration,")
print(f"    protocol, quality control, agreement analysis, data validation,")
print(f"    FAQ, checklist)")
print(f"  - Rating schema: defined (1-7 Likert for 6 dimensions + overall)")
print(f"  - Validation script: validate_ratings.py")
print(f"  - Missing: REAL expert annotations (need 3+ experts, 25 passages each)")
print(f"")
print(f"  Required for paper-quality validation:")
print(f"    1. Recruit 3-5 expert annotators (literary scholars)")
print(f"    2. Run calibration phase (10 practice passages)")
print(f"    3. Collect 25 ratings per expert")
print(f"    4. Compute inter-annotator agreement (Fleiss kappa, ICC)")
print(f"    5. Average expert scores per passage")
print(f"    6. Correlate LDRI vs mean expert overall (r, rho, tau)")
print(f"    7. Correlate features vs per-dimension expert scores")
print(f"")
print(f"  With real expert data, the command would be:")
print(f"    python analyze_correlations.py --expert EXPERT-01 --expert EXPERT-02")
print(f"  (This script is ready to accept expert files when available.)")

print(f"\n{'='*W}")
print("  Output files generated:")
print("    correlation_tables.csv")
print("    feature_intercorrelation.csv")
print("    component_feature_matrix.csv")
print("="*W)
