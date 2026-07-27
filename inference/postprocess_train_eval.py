"""Reorganise a train_eval predictions CSV into a side-by-side comparison sheet.

Rewrites the file in place with, per codebook field, the gold value, the model's
value and a verdict column, in codebook order. Verdicts mirror the scorer in
gemma3_finetune_simple.py: multi-value fields compare as order-insensitive sets,
string fields count word overlap as partial credit.

    python inference/postprocess_train_eval.py <predictions.csv>
"""

import argparse
import importlib.util
import os
import re
import sys

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_OTHER_PAREN_RE = re.compile(r"^Other\(.*\)$")


def _load_targets_spec() -> dict:
    # Loaded by file path: agent_utils/__init__ imports torch, which the mac lacks.
    path = os.path.join(_PROJECT_ROOT, "agent_utils", "africa_dataprep.py")
    spec = importlib.util.spec_from_file_location("africa_dataprep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_targets_spec()


def _words(value: str) -> set:
    return set(" ".join(str(value).strip().lower().split()).split())


def _canon_multi(value: str) -> str:
    return ";".join(sorted(a.strip() for a in str(value).split(";") if a.strip()))


def _verdict(gold: str, pred: str, spec: dict) -> str:
    if not gold:
        return ""
    if not pred:
        return "no_answer"
    if spec.get("multi_value"):
        return "match" if _canon_multi(gold) == _canon_multi(pred) else "mismatch"
    if spec.get("type") == "string":
        g, p = _words(gold), _words(pred)
        if g == p:
            return "match"
        return "partial" if g & p else "mismatch"
    return "match" if gold.strip() == pred.strip() else "mismatch"


def _in_label(value: str, spec: dict, allowed: set) -> bool:
    if not allowed:
        return True
    atoms = (
        [a.strip() for a in value.split(";") if a.strip()]
        if spec.get("multi_value")
        else [value.strip()]
    )
    allow_other = spec.get("allow_other_paren", False)
    return bool(atoms) and all(
        a in allowed or (allow_other and _OTHER_PAREN_RE.match(a)) for a in atoms
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="train_eval_predictions_*.csv to rewrite in place")
    args = parser.parse_args()

    targets = _load_targets_spec()
    df = pd.read_csv(args.csv_path, dtype=str, keep_default_na=False)

    missing = [f"{p}_{t}" for t in targets for p in ("gold", "pred") if f"{p}_{t}" not in df.columns]
    if missing:
        sys.exit(f"input is not a per-field predictions CSV; missing columns: {missing}")

    verdicts, out_of_label, field_cols = {}, {}, {}
    for target, spec in targets.items():
        gold_col, pred_col = f"gold_{target}", f"pred_{target}"
        allowed = {str(a).strip() for a in spec.get("allowed") or []}
        verdicts[target] = [_verdict(g, p, spec) for g, p in zip(df[gold_col], df[pred_col])]
        out_of_label[target] = [
            bool(p) and not _in_label(p, spec, allowed) for p in df[pred_col]
        ]
        field_cols[gold_col] = df[gold_col]
        field_cols[pred_col] = df[pred_col]
        field_cols[f"verdict_{target}"] = verdicts[target]

    graded = pd.DataFrame(verdicts)
    n_scored = (graded != "").sum(axis=1)
    n_correct = graded.isin(["match", "partial"]).sum(axis=1)
    summary = pd.DataFrame(
        {
            "id": df["id"],
            "json_ok": df["json_ok"],
            "n_scored": n_scored,
            "n_correct": n_correct,
            "n_mismatch": (graded == "mismatch").sum(axis=1),
            "n_no_answer": (graded == "no_answer").sum(axis=1),
            "n_out_of_label": pd.DataFrame(out_of_label).sum(axis=1),
            "pct_correct": (n_correct / n_scored).round(3),
        }
    )
    out = pd.concat(
        [summary, pd.DataFrame(field_cols), df["generated"]], axis=1
    )
    out.to_csv(args.csv_path, index=False)

    counts = graded.stack().value_counts()
    print(f"wrote {args.csv_path}: {len(out)} rows x {len(out.columns)} columns")
    print(f"  fields scored     : {int(n_scored.sum())}")
    for name in ("match", "partial", "mismatch", "no_answer"):
        n = int(counts.get(name, 0))
        print(f"  {name:18s}: {n:6d} ({n / max(int(n_scored.sum()), 1):.1%})")
    print(f"  rows with json_ok=False: {(df['json_ok'] != 'True').sum()}")


if __name__ == "__main__":
    main()
