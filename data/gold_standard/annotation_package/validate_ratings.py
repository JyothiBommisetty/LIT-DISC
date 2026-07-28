"""LIT-DISC Expert Rating Validator

Validates expert annotation JSON files against the gold-standard corpus schema.
Supports all QC checks defined in the Quality Control protocol.

Usage:
    python validate_ratings.py EXPERT-01_ratings.json [EXPERT-02_ratings.json ...]
"""

import json
import sys
import os
from pathlib import Path

# The 25 valid passage IDs
VALID_IDS = sorted([
    "SNC-01", "SNC-02", "SNC-03", "SNC-04", "SNC-05",
    "MNC-01", "MNC-02", "MNC-03", "MNC-04", "MNC-05",
    "SSC-01", "SSC-02", "SSC-03", "SSC-04", "SSC-05",
    "MLC-01", "MLC-02", "MLC-03", "MLC-04", "MLC-05",
    "NWC-01", "NWC-02", "NWC-03", "NWC-04", "NWC-05",
])

REQUIRED_SCORE_KEYS = {"semantic", "narrative", "interaction",
                       "symbolic", "evolution", "emotion", "overall"}


def validate_ratings(filepath: str) -> dict:
    """Validate a single rating file. Returns validation report."""
    report = {
        "file": os.path.basename(filepath),
        "status": "PASS",
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        report["status"] = "HARD REJECT"
        report["errors"].append(f"Invalid JSON: {e}")
        return report
    except FileNotFoundError:
        report["status"] = "HARD REJECT"
        report["errors"].append(f"File not found: {filepath}")
        return report

    # --- Schema Check ---
    schema_ok = True
    if not isinstance(data, dict):
        schema_ok = False
        report["errors"].append("Top-level structure must be a JSON object")
    elif "expert" not in data or "ratings" not in data:
        schema_ok = False
        report["errors"].append("Missing required keys: 'expert' and/or 'ratings'")
    else:
        expert = data["expert"]
        for key in ["id", "background", "date"]:
            if key not in expert:
                schema_ok = False
                report["errors"].append(f"Missing expert.{key}")

        ratings = data["ratings"]
        if not isinstance(ratings, list):
            schema_ok = False
            report["errors"].append("'ratings' must be a list")
        elif len(ratings) != 25:
            schema_ok = False
            report["errors"].append(f"Expected 25 ratings, got {len(ratings)}")
        else:
            for i, r in enumerate(ratings):
                if not isinstance(r, dict):
                    schema_ok = False
                    report["errors"].append(f"ratings[{i}] is not an object")
                    continue
                for key in ["passage_id", "scores", "confidence"]:
                    if key not in r:
                        schema_ok = False
                        report["errors"].append(f"ratings[{i}] missing '{key}'")
                if "scores" in r and isinstance(r["scores"], dict):
                    missing = REQUIRED_SCORE_KEYS - set(r["scores"].keys())
                    if missing:
                        schema_ok = False
                        report["errors"].append(
                            f"ratings[{i}] scores missing: {missing}")
    report["checks"]["SCHEMA"] = "PASS" if schema_ok else "FAIL"

    if not schema_ok:
        report["status"] = "HARD REJECT"
        return report

    # --- Range Check ---
    range_ok = True
    for i, r in enumerate(ratings):
        for axis, val in r["scores"].items():
            if not isinstance(val, int) or val < 1 or val > 7:
                range_ok = False
                report["errors"].append(
                    f"ratings[{i}] {axis}={val} (must be integer 1-7)")
        conf = r["confidence"]
        if not isinstance(conf, int) or conf < 1 or conf > 5:
            range_ok = False
            report["errors"].append(
                f"ratings[{i}] confidence={conf} (must be integer 1-5)")
    report["checks"]["RANGES"] = "PASS" if range_ok else "FAIL"

    if not range_ok:
        report["status"] = "HARD REJECT"
        return report

    # --- Completeness Check ---
    complete_ok = True
    seen_ids = set()
    for i, r in enumerate(ratings):
        pid = r["passage_id"]
        if pid in seen_ids:
            complete_ok = False
            report["errors"].append(f"Duplicate passage_id: {pid}")
        seen_ids.add(pid)

    missing = set(VALID_IDS) - seen_ids
    extra = seen_ids - set(VALID_IDS)
    if missing:
        complete_ok = False
        report["errors"].append(f"Missing passages: {sorted(missing)}")
    if extra:
        complete_ok = False
        report["errors"].append(f"Unknown passages: {sorted(extra)}")
    report["checks"]["COMPLETE"] = "PASS" if complete_ok else "FAIL"

    if not complete_ok:
        report["status"] = "HARD REJECT"
        return report

    # --- Consistency Check (warnings only) ---
    for i, r in enumerate(ratings):
        vals = list(r["scores"].values())
        if len(set(vals)) == 1:
            report["warnings"].append(
                f"{r['passage_id']}: Uniform scoring (all 7 axes = {vals[0]})")

    confs = [r["confidence"] for r in ratings]
    if len(set(confs)) == 1:
        report["warnings"].append(
            f"Uniform confidence ({confs[0]}) for all 25 passages")

    all_scores = [s for r in ratings for s in r["scores"].values()]
    if max(all_scores) <= 3:
        report["warnings"].append(
            "All scores <= 3 (possible range restriction)")
    if min(all_scores) >= 5:
        report["warnings"].append(
            "All scores >= 5 (possible range restriction)")

    # --- Duplicate Score Vectors ---
    vector_map = {}
    for r in ratings:
        vec = tuple(r["scores"][k] for k in REQUIRED_SCORE_KEYS)
        pid_a = r["passage_id"]
        for pid_b, v in vector_map.items():
            if vec == v:
                report["warnings"].append(
                    f"Identical score vectors: {pid_a} and {pid_b}")
        vector_map[pid_a] = vec

    # --- Final Status ---
    if report["errors"]:
        report["status"] = "HARD REJECT"
    elif report["warnings"]:
        report["status"] = "PASS WITH WARNINGS"

    return report


def print_report(report: dict):
    """Print a human-readable validation report."""
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION REPORT")
    print(f"{'=' * 60}")
    print(f"  File:   {report['file']}")
    print(f"  Status: {report['status']}")
    print(f"{'=' * 60}")
    for check, result in report["checks"].items():
        print(f"  {check:12s} {result}")
    if report["warnings"]:
        print(f"\n  WARNINGS:")
        for w in report["warnings"]:
            print(f"    - {w}")
    if report["errors"]:
        print(f"\n  ERRORS:")
        for e in report["errors"]:
            print(f"    - {e}")
    print(f"{'=' * 60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_ratings.py <rating_file.json> [...]")
        sys.exit(1)

    all_pass = True
    for filepath in sys.argv[1:]:
        report = validate_ratings(filepath)
        print_report(report)
        if report["status"] == "HARD REJECT":
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
