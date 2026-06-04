# 2004 Codebook + New Dataset Migration — Checklist & Fixes

Context: moving from the old numeric codebook (`africa_prompt_2602-2.txt`) to the
label-name codebook (`africa_prompt_2004.txt`), trained on the new dataset
`AFRICA-TRAIN-DB-1jun2026.csv`. Runs on **Snellius** (`/projects/prjs1308/...`),
not locally — the files under `data_examples/` are only inspection samples.

---

## TL;DR

The new dataset is **already in 2004 label-name format** (e.g. `"politics"`,
`"not applicable"`, `"no mention of the US"`, `;`-joined multi-labels). So the
old number→label recoding is **not needed**. What remains is: wiring the job to
the new CSV (+ joining text), a handful of label string fixes, rewriting
`TARGETS`, and an accuracy change so `"not applicable"` doesn't inflate scores.

---

## ESSENTIAL — must be correct or results are wrong

### E1. Text join (the CSV has no transcripts)
- `AFRICA-TRAIN-DB-1jun2026.csv` is **labels only**, keyed by `id`.
- Transcripts come from `african_videos.json` (`text` column), joined by `id`.
- **`id` type mismatch:** CSV `id` is `float64` (e.g. `1712829139429512.0`),
  JSON `id` is int-like. Must cast both to `int64` before joining or **0 rows
  match** (this exact bug already bit during investigation).
- Rows with no matching text must be **dropped** (per decision). On the local
  sample only 1454/2296 matched — verify the real overlap on Snellius, it may be
  higher. **Check and log how many rows are dropped.**

### E2. `targets_json` built from the label columns directly
- Values are already strings — do **not** apply `topic_mapping` (topic is already
  a name). Remove/skip the old `df["topic01"] = ...map(topic_mapping)` step.
- Column is named `topic` now (not `topic01`).

### E3. Label strings: data must match codebook exactly
Three data values do NOT match the codebook and must be normalized in data-prep:
| Field | Data value | → Correct |
|---|---|---|
| `pro_russia` | `no mention of the Russia` | `no mention of Russia` |
| `pro_china` | `no mention of the China` | `no mention of China` |
| `resource_distribution_by_whom1` | `international organisations` | `international organisation` (inside `;` strings) |

`resource_distribution_by_whom1` is multi-value — replace **within** each
`;`-joined token, not the whole cell.

### E4. `national_unity_narrow == "NOT CODED"` → null
- 1589/2300 rows are `"NOT CODED"` (not a real label). Map to null so they are
  excluded from the training target and from metrics for that field.
- **Do NOT drop the whole row** — only blank that one field.

### E5. `TARGETS` rewritten to codebook label sets
- All `allowed` lists become the **label-name** sets from the codebook
  (not numbers). Conditional fields change `type` `binary` → `multiclass`.
- Rename keys: `topic01`→`topic`, `resource_distribution_for_gender`→
  `resource_distribution_gender`, `pro_mf`→`pro_imf` (CSV already uses correct names).
- Add the 3 fields absent from old TARGETS: `resource_distribution_for_whom1`,
  `national_unity_narrow`, `political_opponents_viol`.
- Every conditional field's allowed set must include `"not applicable"` and
  `"unclear"`; `pro_*` must include `"no mention of <X>"`.

### E6. Accuracy must exclude `"not applicable"` (decision: report both)
- In `run_simple_val_inference` (`gemma3_finetune_simple.py`): keep overall
  accuracy, **add** `n_applicable` + `accuracy_applicable` computed over rows
  where `gold != "not applicable"` (also exclude null gold).
- Reason: gated fields are `"not applicable"` for the majority of rows
  (e.g. `resource_distribution_by_whom1` = 1621/2300 N/A); plain accuracy is
  inflated by trivially predicting N/A.

---

## SHOULD CHECK — correctness-affecting but secondary

### S1. Multi-value (`;`) fields scored as sets, not exact strings
- Multi-value fields: `language`, `topic` (up to 3), `resource_distribution_by_whom1`,
  `resource_distribution_for_whom1`.
- Current eval does exact full-string match → order-sensitive
  (`"a;b"` ≠ `"b;a"`). Consider set-based comparison so ordering doesn't
  mis-score. (Recommended, not yet decided.)

### S2. `language` `Other(X)` values
- Free-form suffix, e.g. `Other(Malagasy)`, `Other(Amharic)`, `;`-joined.
- Allowed set can't enumerate them — treat `Other(...)` as in-label by pattern,
  or accept any `Other(*)`.

### S3. String / free-text targets
- `resource_distribution_for_whom_ethnic1`, `resource_distribution_for_whom_region1`,
  `subgroup_unity_text` stay `type: string`; their `allowed` is filled from the
  data at runtime. Verify they're populated from the **new CSV**, not the old json.

### S4. Codebook ↔ data field coverage
- Confirm every codebook field has a matching CSV column and vice-versa.
  Known: CSV has all 2004 fields incl. the 3 new ones. `topic` includes
  `"not applicable"` (gated) even though the codebook list doesn't spell it out —
  keep it in the allowed set.

### S5. Data hygiene in the CSV
- 4 duplicate `id`s and 8 rows with blank `topic` — decide drop vs keep.
- After normalization, re-run the label diff (see E3) and confirm **zero**
  out-of-codebook values remain.

### S6. System prompt deployment
- The runtime system prompt is read from
  `/projects/prjs1308/africa_llm_data/prompts/africa_prompt_system.txt` on
  Snellius — the corrected `africa_prompt_2004.txt` must be copied there for the
  fix to take effect at train/inference time.

---

## FIXED so far

### F1. Gender label typo in `africa_prompt_2004.txt` (section 10)
- The label `"resources not specifically for women"` was attached to the
  *women-specific* condition (backwards). Corrected so labels match the data:
  - `"resources for women"` → resources apply to women specifically
  - `"resources not specifically for women"` → do not apply to women exclusively
  - `"unclear"`, `"not applicable"`
- Also split the three run-together sentences onto separate lines.
- Result: **no data change needed** for `resource_distribution_gender`.

---

## RESOLVED QUESTIONS (decisions taken)

1. **Numeric vs label gold** — new dataset is already label-names; no recode needed.
2. **Text source** — join from `african_videos.json` by `id`; drop unmatched.
3. **`pro_*` "no mention" vs "not applicable"** — already distinguished in the data.
4. **Missing fields** — all present in the new CSV.
5. **Stray numeric codes** (`climate_change=3`, `anti_western=3`) — gone in new data.
6. **`national_unity_narrow == "NOT CODED"`** — ignore (null that field).
7. **Gender label conflict** — fixed in codebook (F1).

---

## IMPLEMENTATION PLAN (not yet started)

1. Job data-loading (`gemma3_finetune.py`, `gemma3_finetune_fulltrain.py`):
   load CSV, cast `id`, join `text` from json, drop unmatched, build
   `targets_json` from label columns (no `topic_mapping`).
2. Normalize labels in data-prep: 3 string swaps (E3) + `NOT CODED`→null (E4).
3. Rewrite `TARGETS` to codebook label sets (E5).
4. Add `accuracy_applicable` / `n_applicable` to `run_simple_val_inference` (E6).
