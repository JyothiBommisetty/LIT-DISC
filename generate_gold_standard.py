"""Generate data/gold_standard/litdisc_results.json with sigma=5.0."""
import json, os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import LITDISCConfig
from src.pipeline import LITDISCPipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA = os.path.join(ROOT, "data", "gold_standard", "metadata.json")
PASSAGES = os.path.join(ROOT, "data", "gold_standard", "passages")
OUTPUT = os.path.join(ROOT, "data", "gold_standard", "litdisc_results.json")
OLD = os.path.join(ROOT, "data", "gold_standard", "litdisc_results.json.old")

# Archive old results
if os.path.exists(OUTPUT) and not os.path.exists(OLD):
    os.rename(OUTPUT, OLD)
    print(f"[ARCHIVE] {OUTPUT} -> {OLD}")

with open(METADATA, encoding="utf-8") as f:
    meta = json.load(f)
passages = meta["passages"]

# Load old feature_means and expert_scores (unchanged by sigma change)
old_lookup = {}
if os.path.exists(OLD):
    with open(OLD, encoding="utf-8") as f:
        for p in json.load(f):
            old_lookup[p["passage_id"]] = {
                "feature_means": p.get("feature_means", {}),
                "expert_scores": p.get("expert_scores", {}),
            }

# Map passage IDs to files
passage_map = {}
for fname in os.listdir(PASSAGES):
    pid = fname.split("_")[0]
    passage_map[pid] = os.path.join(PASSAGES, fname)

config = LITDISCConfig()
pipeline = LITDISCPipeline(config=config, refinement_mode="full")

total_start = time.perf_counter()
results = []

for p in passages:
    pid = p["id"]
    path = passage_map.get(pid)
    if not path or not os.path.exists(path):
        print(f"  ERROR: {pid} - file not found")
        continue

    with open(path, encoding="utf-8") as f:
        text = f.read()
    wc = len(text.split())

    t0 = time.perf_counter()
    try:
        result = pipeline.run(path)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  ERROR: {pid} - {e}")
        pipeline.reset()
        continue
    pipeline.reset()

    comp = result.ldri.components or {}
    old_info = old_lookup.get(pid, {"feature_means": {}, "expert_scores": {}})

    entry = {
        "passage_id": pid,
        "author": p["author"],
        "title": p["title"],
        "category": p["category"],
        "word_count": wc,
        "runtime_s": round(elapsed, 2),
        "ldri": round(float(result.ldri.ldri), 6),
        "raw_ldri": round(float(result.ldri.raw), 6),
        "gamma_global": round(float(result.ldri.gamma_global), 6),
        "components": {
            "intrinsic": round(float(comp.get("intrinsic", 0)), 6),
            "interaction": round(float(comp.get("interaction", 0)), 6),
            "temporal": round(float(comp.get("temporal", 0)), 6),
            "resonance": round(float(comp.get("resonance", 0)), 6),
            "memory": round(float(comp.get("memory", 0)), 6),
            "uncertainty": round(float(comp.get("uncertainty", 0)), 6),
        },
        "dimension_scores": {},
        "expert_scores": old_info["expert_scores"],
        "feature_means": old_info["feature_means"],
        "sentences": len(text.split("\n")),
        "words": wc,
    }
    results.append(entry)
    print(f"  [ OK] {pid:>8}  LDRI={entry['ldri']:.6f}  raw={entry['raw_ldri']:.6f}  gamma={entry['gamma_global']:.6f}  {elapsed:.2f}s")

total_elapsed = time.perf_counter() - total_start

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print()
print("=" * 60)
print(f"  GOLD STANDARD REGENERATION COMPLETE (sigma_max=5.0)")
print(f"  {len(results)} passages processed, {len(passages)-len(results)} skipped/errors")
print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/max(len(results),1):.2f}s avg)")
print(f"  Output: {OUTPUT}")
print("=" * 60)
