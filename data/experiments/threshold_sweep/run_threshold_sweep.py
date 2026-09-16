"""Memory retrieval threshold sweep for LIT-DISC v1.1.

Sweeps theta_retrieve from 0.1 to 0.9 to determine if the
zero-cross-window-retrieval problem is architectural or a threshold artifact.

Usage: python data/experiments/threshold_sweep/run_threshold_sweep.py
"""
import sys, json, os
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.pipeline import LITDISCPipeline
from src.config import LITDISCConfig

PASSAGES_DIR = os.path.join("data", "gold_standard", "passages")
OUTPUT_DIR = os.path.join("data", "experiments", "threshold_sweep")

# Use a representative subset of gold standard passages
TEST_PASSAGES = [
    ("SNC-04", "SNC-04_woolf_lighthouse.txt"),   # High LDRI
    ("MNC-01", "MNC-01_mantel_wolf.txt"),        # Low LDRI
    ("MLC-02", "MLC-02_proust_swann.txt"),       # High LDRI
    ("SSC-04", "SSC-04_munro_bear.txt"),         # Mid LDRI
    ("NWC-05", "NWC-05_rushdie_midnight.txt"),   # Mid LDRI
]

THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for pid, fname in TEST_PASSAGES:
        path = os.path.join(PASSAGES_DIR, fname)
        print(f"\n{'='*60}")
        print(f"Passage: {pid} ({fname})")
        print(f"{'='*60}")

        passage_results = []
        for theta in THETAS:
            config = LITDISCConfig(
                hmp_dt=replace(
                    LITDISCConfig().hmp_dt,
                    memory=replace(
                        LITDISCConfig().hmp_dt.memory,
                        theta_retrieve=theta,
                    ),
                )
            )
            pipeline = LITDISCPipeline(config=config)
            r = pipeline.run(path)

            mem_entries = r.metadata.get("memory_entries", 0)
            mem_avg_beta = r.metadata.get("memory_avg_beta", 0.0)

            passage_results.append({
                "theta": theta,
                "ldri": r.ldri.ldri,
                "gamma": r.ldri.gamma_global,
                "raw": r.ldri.raw,
                "reliability": r.ldri.reliability,
                "memory_entries": mem_entries,
                "memory_avg_beta": mem_avg_beta,
            })
            print(f"  theta={theta:.1f}: LDRI={r.ldri.ldri:.6f} "
                  f"gamma={r.ldri.gamma_global:.6f} "
                  f"entries={mem_entries} avg_beta={mem_avg_beta:.4f}")

        results[pid] = passage_results

    # Save
    out_path = os.path.join(OUTPUT_DIR, "threshold_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Effect of theta_retrieve on memory retrievals")
    print("=" * 60)
    print(f"{'theta':>6s}", end="")
    for pid, _ in TEST_PASSAGES:
        print(f"  {pid:>8s}_LDRI  {pid:>8s}_beta", end="")
    print()
    for i, theta in enumerate(THETAS):
        print(f"{theta:>6.1f}", end="")
        for pid, _ in TEST_PASSAGES:
            pr = results[pid][i]
            print(f"  {pr['ldri']:>10.6f}  {pr['memory_avg_beta']:>8.4f}", end="")
        print()


if __name__ == "__main__":
    run_sweep()
