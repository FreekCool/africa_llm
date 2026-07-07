---
name: dataprep-dryrun
description: Validate africa_llm data-prep changes locally on the mac (edit-only machine, no GPU stack) by loading agent_utils/africa_dataprep.py by file path and checking the migration validation gates against data_examples/AFRICA-TRAIN-DB-3jun2026.csv. Use before any data-prep change is handed to Snellius.
---

# Local data-prep dry-run

The mac has no torch/trl, and `import agent_utils` triggers `__init__.py` →
`utils` → torch, so it always fails locally. `africa_dataprep.py` itself needs only
pandas/re/json — load it **by file path**, never via the package:

```python
import importlib.util, json
spec = importlib.util.spec_from_file_location(
    "africa_dataprep", "agent_utils/africa_dataprep.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
df, targets = mod.load_3jun_training_df("data_examples/AFRICA-TRAIN-DB-3jun2026.csv")
```

Run with the `vve_nxt` conda env (`conda run -n vve_nxt python ...`).

## Validation gates (all must hold)

| Gate | Expected on the 3jun CSV |
|---|---|
| Row count | 2300, 0 dropped |
| `targets_json` keys | exactly 27 per row, label-name values (no numeric codes) |
| `NOT CODED` | 0 occurrences in `targets_json` (E4 applied) |
| `national_unity_narrow` nulls | 1589 |
| Codebook conformance | `assert_codebook_conformance` passes inside the loader (only exemptions: `Other(...)` on `language`) |
| `TARGETS` ↔ CSV columns | 27 ↔ 27 key coverage |
| String-target `allowed` fills | `resource_distribution_for_whom_ethnic1`=8, `..._region1`=351, `subgroup_unity_text`=385 distinct values |

If a gate fails after a data change: **fix the data at source, do not patch in
code** (project hard rule — the conformance guard exists to fail loudly on a bad
CSV swap).

## Scope limits

This validates data prep and scoring logic only. Anything touching the model,
tokenizer, chat template, or trainer needs Snellius — say so explicitly and hand
over the run via the snellius-handoff skill instead of claiming it's verified.
