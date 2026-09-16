"""Null Model Baseline Experiment for LIT-DISC v1.1.

Two null conditions:
  1. Shuffled text: paragraph-order randomized within each passage
  2. Random embeddings: Gaussian noise replaces computed embeddings

Usage: python data/experiments/null_model/run_null_model.py
"""
import sys, json, os, random, re
import numpy as np
from dataclasses import dataclass


def spearmanr(x, y):
    """Manual Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return rho, 0.0  # p-value not computed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.pipeline import LITDISCPipeline
from src.models import NarrativeEmbedding

PASSAGES_DIR = os.path.join("data", "gold_standard", "passages")
METADATA_PATH = os.path.join("data", "gold_standard", "metadata.json")
OUTPUT_DIR = os.path.join("data", "experiments", "null_model")

random.seed(42)
np.random.seed(42)


def load_passages():
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    passages = []
    for p in meta["passages"]:
        pid = p["id"]
        for fname in os.listdir(PASSAGES_DIR):
            if fname.startswith(pid):
                path = os.path.join(PASSAGES_DIR, fname)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                passages.append((pid, p["category"], p["author"], text, path))
                break
    return passages


def shuffle_paragraphs(text, seed=42):
    rng = random.Random(seed)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    rng.shuffle(paragraphs)
    return "\n\n".join(paragraphs)


def _shuffled_path(pid):
    return os.path.join(OUTPUT_DIR, f"shuffled_{pid}.txt")


def prepare_shuffled_passages(passages):
    paths = []
    for pid, cat, author, text, orig_path in passages:
        s_text = shuffle_paragraphs(text, seed=hash(pid) % (2 ** 31))
        s_path = _shuffled_path(pid)
        os.makedirs(os.path.dirname(s_path), exist_ok=True)
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(s_text)
        paths.append((pid, cat, s_path))
    return paths


def compute_noise_stats(pipeline, passages):
    norms_by_dim = {d: [] for d in ["S","N","I","Sy","E","Em","pairs","res"]}
    for pid, cat, author, text, path in passages:
        pipeline.reset()
        r = pipeline.run(path)
        for d in ["S","N","I","Sy","E","Em"]:
            norms_by_dim[d].append(np.linalg.norm(getattr(r.embedding, f"H_{d}")))
        norms_by_dim["pairs"].append(np.linalg.norm(r.embedding.H_pairs))
        norms_by_dim["res"].append(np.linalg.norm(r.embedding.H_res))
    stats = {}
    for d, vals in norms_by_dim.items():
        stats[d] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats


def random_embedding(noise_stats, d=120):
    d_per = d // 10
    comps = {}
    for dim in ["S","N","I","Sy","E","Em"]:
        s = noise_stats[dim]
        comps[f"H_{dim}"] = np.random.randn(d_per) * s["std"] + s["mean"]
    s = noise_stats["pairs"]
    comps["H_pairs"] = np.random.randn(3 * d_per) * s["std"] + s["mean"]
    s = noise_stats["res"]
    comps["H_res"] = np.random.randn(d_per) * s["std"] + s["mean"]
    comps["d"] = d
    return NarrativeEmbedding(**comps)


@dataclass
class NullResult:
    condition: str
    pid: str
    category: str
    ldri: float
    gamma: float
    raw: float
    reliability: float


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    passages = load_passages()
    pipeline = LITDISCPipeline()

    # ── Condition A: Baseline (full pipeline, original text) ──────
    print("=" * 60)
    print("Condition A: Baseline (original text)")
    print("=" * 60)
    baseline = []
    for pid, cat, author, text, path in passages:
        pipeline.reset()
        r = pipeline.run(path)
        baseline.append(NullResult("baseline", pid, cat,
            r.ldri.ldri, r.ldri.gamma_global, r.ldri.raw, r.ldri.reliability))
        print(f"  {pid:8s} LDRI={r.ldri.ldri:.6f} gamma={r.ldri.gamma_global:.6f} raw={r.ldri.raw:.6f}")

    # ── Condition B: Shuffled text ────────────────────────────────
    print("\n" + "=" * 60)
    print("Condition B: Shuffled text (paragraphs randomized)")
    print("=" * 60)
    shuffled_paths = prepare_shuffled_passages(passages)
    shuffled = []
    for pid, cat, s_path in shuffled_paths:
        pipeline.reset()
        r = pipeline.run(s_path)
        shuffled.append(NullResult("shuffled", pid, cat,
            r.ldri.ldri, r.ldri.gamma_global, r.ldri.raw, r.ldri.reliability))
        print(f"  {pid:8s} LDRI={r.ldri.ldri:.6f} gamma={r.ldri.gamma_global:.6f} raw={r.ldri.raw:.6f}")

    # ── Compute noise stats from baseline embeddings ──────────────
    pipeline2 = LITDISCPipeline()
    noise_stats = compute_noise_stats(pipeline2, passages)
    print("\nNoise stats (from baseline embeddings):")
    for d, s in noise_stats.items():
        print(f"  {d:6s}: mean={s['mean']:.3f}, std={s['std']:.3f}")

    # ── Condition C: Random embeddings ────────────────────────────
    print("\n" + "=" * 60)
    print("Condition C: Random embeddings (Gaussian noise)")
    print("=" * 60)
    random_emb = []
    rng_emb_seed = 42
    for pid, cat, author, text, path in passages:
        np.random.seed(rng_emb_seed + hash(pid) % (2 ** 31))
        pipeline.reset()
        # Replace the embedder's construct method to return random embeddings
        original_construct = pipeline.embedder.construct
        def _make_random_construct(stats):
            def _random_construct(signals, tree, resonance_trajectory=None):
                return random_embedding(stats)
            return _random_construct
        pipeline.embedder.construct = _make_random_construct(noise_stats)
        r = pipeline.run(path)
        pipeline.embedder.construct = original_construct
        random_emb.append(NullResult("random_emb", pid, cat,
            r.ldri.ldri, r.ldri.gamma_global, r.ldri.raw, r.ldri.reliability))
        print(f"  {pid:8s} LDRI={r.ldri.ldri:.6f} gamma={r.ldri.gamma_global:.6f} raw={r.ldri.raw:.6f}")

    # ── Analyze ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    def summarize(results, label):
        ldris = [r.ldri for r in results]
        gammas = [r.gamma for r in results]
        raws = [r.raw for r in results]
        mean_ldri = np.mean(ldris)
        std_ldri = np.std(ldris)
        cv_ldri = std_ldri / mean_ldri * 100 if mean_ldri > 0 else 0
        print(f"\n{label}:")
        print(f"  LDRI:     mean={mean_ldri:.6f} std={std_ldri:.6f} CV={cv_ldri:.2f}%")
        print(f"  Gamma:    mean={np.mean(gammas):.6f} std={np.std(gammas):.6f}")
        print(f"  Raw:      mean={np.mean(raws):.6f} std={np.std(raws):.6f}")

    summarize(baseline, "BASELINE (original text)")
    summarize(shuffled, "SHUFFLED TEXT")
    summarize(random_emb, "RANDOM EMBEDDINGS")

    # Correlations with baseline
    def correlate(baseline_results, test_results, label):
        b_ldri = [r.ldri for r in baseline_results]
        t_ldri = [r.ldri for r in test_results]
        rho, p = spearmanr(b_ldri, t_ldri)
        # Category means
        cats_b = {}
        cats_t = {}
        for br, tr in zip(baseline_results, test_results):
            cats_b.setdefault(br.category, []).append(br.ldri)
            cats_t.setdefault(tr.category, []).append(tr.ldri)
        cat_b_means = [np.mean(cats_b[c]) for c in sorted(cats_b)]
        cat_t_means = [np.mean(cats_t[c]) for c in sorted(cats_t)]
        rho_cat, p_cat = spearmanr(cat_b_means, cat_t_means)
        print(f"\n{label} vs BASELINE:")
        print(f"  LDRI rank correlation: rho={rho:.4f}")
        print(f"  Category mean rank correlation: rho_cat={rho_cat:.4f}")
        return rho, p

    correlate(baseline, shuffled, "SHUFFLED TEXT")
    correlate(baseline, random_emb, "RANDOM EMBEDDINGS")

    # Category means
    def category_means(results):
        cats = {}
        for r in results:
            cats.setdefault(r.category, []).append(r.ldri)
        return {c: (np.mean(vals), np.std(vals)) for c, vals in cats.items()}

    print("\n" + "=" * 60)
    print("CATEGORY MEANS (LDRI)")
    print("=" * 60)
    for label, results in [("BASELINE", baseline), ("SHUFFLED", shuffled), ("RANDOM_EMB", random_emb)]:
        cm = category_means(results)
        print(f"\n{label}:")
        for cat in sorted(cm):
            print(f"  {cat}: {cm[cat][0]:.6f} ± {cm[cat][1]:.6f}")

    # ── Save results ─────────────────────────────────────────────
    all_results = {"baseline": baseline, "shuffled": shuffled, "random_emb": random_emb}
    save_data = {}
    for cond, records in all_results.items():
        save_data[cond] = [
            {"pid": r.pid, "category": r.category,
             "ldri": r.ldri, "gamma": r.gamma, "raw": r.raw, "reliability": r.reliability}
            for r in records
        ]
    out_path = os.path.join(OUTPUT_DIR, "null_model_results.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
