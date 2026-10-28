#!/usr/bin/env python3
"""Run the v2 synthetic eval battery and write evals/eval_results.json.
Usage: python scripts/run_eval.py  (from repo root, v2 deps installed)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.evals import run_eval  # noqa: E402

if __name__ == "__main__":
    results = run_eval()
    out = os.path.join(os.path.dirname(__file__), "..", "evals", "eval_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: results[k] for k in ("n_cases", "per_layer", "benign_cases",
                                              "benign_false_flag_rate")}, indent=1))
    print(f"wrote {out}")
