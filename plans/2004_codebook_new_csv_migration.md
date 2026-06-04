# Plan — 2004 codebook + 3jun CSV migration (UPDATED 2026-06-04)

**Goal:** Train/eval `simple_gemma3` on the **3jun** label-name dataset
(`AFRICA-TRAIN-DB-3jun2026.csv`) using the 2004 codebook (`africa_prompt_2004.txt`),
replacing the old numeric codebook + `african_videos.json` label flow.

**Companion doc:** `reports/codebook_2004_migration_checklist.md` (E/S/F item IDs below
refer to it).

> This revision supersedes the original plan. Two things changed the work materially:
> 1. **Dataset = `AFRICA-TRAIN-DB-3jun2026.csv`** (not 1jun + json join). Verified that
>    3jun is the 1jun label set with the three E3 string fixes **already applied** plus
>    an **embedded `text` column** covering all 2300 rows (text matches
>    `african_videos.json` on every overlapping id). So there is **no json join**, the
>    E1 id-cast-join bug is gone, and **E3 is already done in the data**.
> 2. **T1 and T2 are complete and validated** (see Status).

---

## 0. Status

| Task | State | Validation |
|---|---|---|
| **T1 — Codebook fixes** | ✅ **Done** | Example JSON already label-name; §26 `"mentions violent group"1` typo fixed. 27 section names == 27 example-JSON keys. |
| **T2 — Data loading (3jun)** | ✅ **Done** | `jobs/gemma3_finetune.py` loads 3jun CSV, casts `id`→int64, uses embedded `text`, builds `targets_json` from 27 label cols. 12/12 gates pass: 2300 rows / 0 dropped, 27 keys/row, label-name values, no numeric leakage, 80/20 split. |
| **T3 — E4 normalization** | ✅ **Done** | `national_unity_narrow == "NOT CODED"` → null before `targets_json`. 7/7 gates pass: 1589 rows blanked, 0 `NOT CODED` remain, 2300 rows unchanged, only that field affected (109 patriotism / 602 not-specifically-patriotic / 1589 null). |
| **T4 — `TARGETS` rewrite** | ✅ **Done** | `TARGETS` is now the data-verified label-name spec (27 fields, §1–§27 order) in the shared helper `agent_utils/africa_dataprep.py`; conditional fields multiclass, 3 new fields added, `multi_value`/`allow_other_paren` flags set. Conformance guard (key-coverage 27↔27 + atomic-subset, Other(*) exempt) passes on the 3jun CSV. D2 resolved: accept `Other(*)` by regex. |
| **T5 — `accuracy_applicable`** | ✅ **Done** | `run_simple_val_inference` computes `n_applicable` + `accuracy_applicable` per target (subset where gold ≠ `"not applicable"`); printed as `n_app`/`acc_app` columns + both in per-target CSV; overall `accuracy` untouched. Logic test: gated field overall acc 0.889 vs `accuracy_applicable` 0.000; all-N/A field → `n_applicable=0`, `accuracy_applicable` null. Metrics-time `NOT CODED` skip folded into the same change. |
| **D1/D2 — set scoring** | ✅ **Done** | `run_simple_val_inference` scores `multi_value` fields as order-insensitive sets (`"a;b"==​"b;a"`) for accuracy + applicable-subset accuracy, and accepts `Other(*)` as in_label via `allow_other_paren`. Unit-tested: order-swap 0.667 (set) vs 0.333 (old exact). |
| **T6 — Shared helper + fulltrain** | ✅ **Done** | `agent_utils/africa_dataprep.py: load_3jun_training_df()` holds all data-prep + TARGETS + conformance. Both jobs call it (fulltrain dropped the old json + numeric `topic_mapping`), so their data-prep is diff-empty. |
| **T7 — Local dry-run** | ✅ **Done** | Helper run on local 3jun sample: 2300 rows / 0 dropped, 27-key label-name `targets_json`, 0 numeric / 0 `NOT CODED`, E4 nulls=1589, conformance OK, string allowed 8/351/385. (The training-time `[SYSTEM]`/`[ASSISTANT]` print needs the model → Snellius runtime.) |
| **T8 — Harden + deploy** | 🟡 **Code done; deploy pending** | Missing-system-prompt branch in both jobs now raises `FileNotFoundError` instead of silently loading the numeric `africa_prompt_2602.txt`. **Remaining (manual, Snellius): P2 deploy** — copy `africa_prompt_2004.txt` → `prompts/africa_prompt_system.txt` and place the 3jun CSV at `CSV_PATH`, then launch under Screen on a GPU node. |

**Current state:** all code is migrated and locally validated (T3, T4, T5, D1/D2, T6, T7
done; T8 foot-gun done). The data-prep + `TARGETS` + scoring are end-to-end correct on the
3jun CSV. The **only** remaining step is the manual Snellius deploy (P2: place the
label-name codebook at `prompts/africa_prompt_system.txt` and the 3jun CSV at `CSV_PATH`),
then launch — the job now fails loudly if that codebook is missing.

---

## 0.5 Runnability prerequisites — making the script run with the two files

**Inputs:** data = `AFRICA-TRAIN-DB-3jun2026.csv`, codebook =
`codebooks/africa_prompt_2004.txt` (the **label-name** codebook; confirmed — the numeric
`africa_prompt_2602-2.txt` must **not** be used, it would teach numeric output against
label-name gold).

Two mechanics the rest of the plan assumes but never spelled out:

1. **The script never reads `codebooks/` directly.** It reads the system prompt from
   `PROMPTS_DIR/africa_prompt_system.txt` and the data from `CSV_PATH` — both currently
   Snellius paths (`gemma3_finetune.py:57,93`). So *"use codebook 2004"* literally means
   **deploy `africa_prompt_2004.txt` → `prompts/africa_prompt_system.txt`**; the local
   filename is irrelevant to the running job.
2. **Codebook format must match the data format** (label-name ↔ label-name), including the
   codebook's example output JSON.

| # | Prerequisite | Where | By |
|---|---|---|---|
| P1 | `CSV_PATH` resolves to the 3jun file | `gemma3_finetune.py:57` (Snellius) / override to `data_examples/...` for local | T2 ✅ (Snellius) |
| P2 | `africa_prompt_2004.txt` deployed to `prompts/africa_prompt_system.txt` | Snellius | **T8** |
| P3 | Deployed codebook's example JSON uses label-name keys/values | codebook | T1 ✅ |
| P4 | User-prompt template present, or rely on the built-in `{}` fallback | `prompts/inference_prompt.txt` (optional) | OK as-is |
| P5 | `targets_json` has no `NOT CODED`; `TARGETS` is label-name | code | **T3 + T4** |

- **"Executes" needs P1.** "Correct experiment" needs P2 + P3 + P5 (and the right codebook).
- **Numeric-fallback foot-gun (fix in T8 area):** if `africa_prompt_system.txt` is absent,
  the `else` branch (`gemma3_finetune.py:104-108`) silently loads the **numeric**
  `africa_prompt_2602.txt` as a single prompt. On Snellius this can silently revert the
  migration; locally it crashes (file absent). Harden this: fail loudly if the expected
  label-name system prompt is missing rather than falling back to a numeric codebook.
- **Local dry-run reality (T7):** the full job needs a GPU + model + the Snellius prompt
  paths, so it does not run end-to-end locally. Local validation = the **standalone
  data-prep checks** (already passing: 12/12 T2 gates). True end-to-end test is on Snellius
  after P1 + P2.

---

## 1. Remaining requirements

R3. `TARGETS` describes the codebook's label-name sets, correct key names, all 27 fields.
R4. `national_unity_narrow == "NOT CODED"` is excluded from target + metrics (E4).
R5. Accuracy reported both overall and excluding `"not applicable"` / null gold (E6).
R6. The codebook the model sees on Snellius matches the label-name convention (deploy).
R7. Both jobs (`gemma3_finetune.py`, `gemma3_finetune_fulltrain.py`) migrated identically.

(R1/R2 — "train on label-name strings joined to text" — satisfied by T2 via 3jun's
embedded text.)

---

## 2. Design (remaining)

### 2.1 E4 normalization — the only in-code data transform (T3)

3jun already satisfies E3 (verified: 0 occurrences of `no mention of the Russia` /
`no mention of the China` / `international organisations`). So **do not re-apply the E3
string swaps** — they would be no-ops and the data is the source of truth. The one
remaining transform:

- **E4:** `national_unity_narrow == "NOT CODED"` (1589 rows) → set the cell to `None`
  **before** building `targets_json`, so that field serializes to `null`. Blank that
  field only; do **not** drop the row.

Rationale for keeping E4 in-code (vs at source like E3): `NOT CODED` is a documented
"annotators never coded this" sentinel, mapped to null per resolved decision Q6 — a
modeling choice, not a typo fix.

> Note: a metrics-time `NOT CODED` skip already exists in
> `agent_utils/gemma3_finetune_simple.py` (uncommitted). E4 additionally removes it from
> the **training target**. Keep both.

### 2.2 `TARGETS` rewrite (T4) — allowed-sets DATA-VERIFIED

Diffed every categorical field's atomic values (`;`-split) in 3jun against the
label-name sets below: **26/27 fields match exactly, zero unexpected values.** The only
out-of-codebook values are `language`'s `Other(<lang>)` free-form entries
(`Other(Amharic)` ×63, `Other(Malagasy)` ×57, `Other(Somali)` ×49, …) — that is
**open-decision D2**, not a data error. So the sets below are locked, pending D2.

Required key/type changes (CSV already uses the new names):
- Rename `topic01`→`topic`, `resource_distribution_for_gender`→`resource_distribution_gender`,
  `pro_mf`→`pro_imf`.
- Add 3 fields: `resource_distribution_for_whom1`, `national_unity_narrow`,
  `political_opponents_viol`.
- Conditional fields: `type: binary` → `type: multiclass`.

```python
TARGETS = {
  "language": {"type": "multiclass", "allowed": [
      "English","French","Arabic","Portuguese","Swahili","Hausa","Yoruba",
      "Other","Unclear"]},  # + Other(<lang>) free-form — see D2
  "politics": {"type": "multiclass", "allowed": ["politics","not political","unclear"]},
  "domestic_politics": {"type": "multiclass", "allowed": [
      "domestic politics","not domestic politics","unclear","not applicable"]},
  "foreign_politics": {"type": "multiclass", "allowed": [
      "foreign politics","not foreign politics","unclear","not applicable"]},
  "resource_distribution": {"type": "multiclass", "allowed": [
      "resource distribution","not resource distribution","unclear","not applicable"]},
  "resource_distribution_by_whom1": {"type": "multiclass", "allowed": [
      "other state","international organisation","national government","other",
      "not specified","unclear","not applicable"]},
  "resource_distribution_for_whom1": {"type": "multiclass", "allowed": [
      "specific locality or group","country-wide","not specified","unclear",
      "not applicable"]},
  "resource_distribution_for_whom_ethnic1": {"type": "string", "allowed": [], "eval": {...}},
  "resource_distribution_for_whom_region1": {"type": "string", "allowed": [], "eval": {...}},
  "resource_distribution_gender": {"type": "multiclass", "allowed": [
      "resources for women","resources not specifically for women","unclear",
      "not applicable"]},
  "climate_change": {"type": "multiclass", "allowed": [
      "mentions climate change","mentions sustainability","unclear","not applicable"]},
  "topic": {"type": "multiclass", "allowed": [
      "NO TOPIC","ECONOMY","CIVIL RIGHTS","HEALTH","AGRICULTURE","LABOR","EDUCATION",
      "ENVIRONMENT","ENERGY","IMMIGRATION","TRANSPORTATION","LAW AND CRIME",
      "SOCIAL WELFARE","HOUSING","DOMESTIC COMMERCE","DEFENSE","TECHNOLOGY",
      "FOREIGN TRADE","INTERNATIONAL AFFAIRS","GOVERNMENT OPERATIONS","PUBLIC LANDS",
      "CULTURE","ETHNICITY","not applicable"]},  # up to 3, ;-joined
  "pro_us": {"type": "multiclass", "allowed": [
      "positive towards the US","neutral towards the US","negative towards the US",
      "unclear","no mention of the US","not applicable"]},
  "pro_russia": {"type": "multiclass", "allowed": [
      "positive towards Russia","neutral towards Russia","negative towards Russia",
      "unclear","no mention of Russia","not applicable"]},
  "pro_china": {"type": "multiclass", "allowed": [
      "positive towards China","neutral towards China","negative towards China",
      "unclear","no mention of China","not applicable"]},
  "pro_un": {"type": "multiclass", "allowed": [
      "positive towards the UN","neutral towards the UN","negative towards the UN",
      "unclear","no mention of the UN","not applicable"]},
  "pro_imf": {"type": "multiclass", "allowed": [
      "positive towards the IMF","neutral towards the IMF","negative towards the IMF",
      "unclear","no mention of the IMF","not applicable"]},
  "pro_democracy": {"type": "multiclass", "allowed": [
      "positive towards democracy","neutral towards democracy","negative towards democracy",
      "unclear","no mention of democracy","not applicable"]},
  "anti_western": {"type": "multiclass", "allowed": [
      "anti-western","not anti-western","unclear","not applicable"]},
  "national_unity": {"type": "multiclass", "allowed": [
      "national unity","no mention of the nation or national unity","unclear"]},
  "national_unity_narrow": {"type": "multiclass", "allowed": [
      "patriotism","not specifically patriotic","unclear"]},  # NOT CODED → null (E4)
  "subgroup_unity": {"type": "multiclass", "allowed": [
      "subgroup unity","no mention of specific subgroup","unclear"]},
  "subgroup_unity_text": {"type": "string", "allowed": [], "eval": {...}},
  "african_unity": {"type": "multiclass", "allowed": [
      "african unity","no mention of africa","unclear"]},
  "political_opponents": {"type": "multiclass", "allowed": [
      "mentions political opponents","no mention of political opponents","unclear",
      "not applicable"]},
  "political_opponents_viol": {"type": "multiclass", "allowed": [
      "mentions violent group","no mention of violent group","unclear","not applicable"]},
  "religion": {"type": "multiclass", "allowed": [
      "religious","no mention of religion","unclear"]},
}
```
(String `eval` blocks unchanged from current code; `allowed` filled at runtime from the
3jun CSV — verified non-empty: `..._ethnic1` 8 distinct, `..._region1` 351,
`subgroup_unity_text` 385.)

**Retained-but-absent labels are intentional.** Many allowed values never appear in 3jun
(e.g. most fields' `"unclear"`, `language`'s `Hausa`/`Yoruba`, `pro_china`'s
`"negative towards China"`). They stay in `allowed` because they are valid codebook
labels the model may legitimately predict — `in_label` scoring must accept them.

### 2.3 Codebook-conformance assertion (folded into T4)

The original plan put an out-of-codebook assertion in T3, but it needs `TARGETS`, so it
moves to T4. After building `targets_json`, assert that every **non-string** field's
atomic (`;`-split) values are a subset of its `allowed` set, with two documented
exemptions: `national_unity_narrow == "NOT CODED"` (handled by E4) and `language`'s
`Other(...)` pattern (D2). **If it fails, fix the data at source — do not patch in
code** (hard rule: no workarounds). This is the loud-failure guard for a future CSV swap.

### 2.4 Accuracy excluding "not applicable" (T5, E6)

In `run_simple_val_inference` (`agent_utils/gemma3_finetune_simple.py`), per-target loop:
additionally compute over the subset where `gold not in {"not applicable", None}`:
- `n_applicable` — count of such rows
- `accuracy_applicable` — accuracy on that subset

Add both as columns in the per-target CSV `rows` and print them. Keep overall `accuracy`
untouched. (Gated fields are mostly N/A — e.g. `resource_distribution_by_whom1`; plain
accuracy is inflated by trivially predicting N/A.)

> **Implemented (T5, 2026-06-04).** `_is_applicable_gold(g)` (False when gold is
> `"not applicable"`, case-insensitive). `n_applicable` counts applicable golds across
> all gold rows (parallel to `n_gold`); `accuracy_applicable` is computed on the answered
> applicable subset (parallel to `accuracy`, so directly comparable) and is `None` when no
> applicable example was answered. Both go to the printed table (`n_app`/`acc_app`) and the
> CSV; the `_timing` row mirrors the new keys.

---

## 3. Tasks (commit after each)

- [x] **T3 — E4 normalization (2.1).** In `gemma3_finetune.py`, blank
      `national_unity_narrow == "NOT CODED"` → `None` before `targets_json` is built.
      *(Validated 2026-06-04: present at `gemma3_finetune.py:75-77`, runs before
      `targets_json` build.)*
- [x] **T4 — Rewrite `TARGETS` (2.2) + conformance assertion (2.3).** Done in the shared
      helper `agent_utils/africa_dataprep.py`: data-verified label-name dict, numeric sets
      removed, subset assertion + key-coverage gate. D2 resolved (accept `Other(*)` regex).
- [x] **T5 — `accuracy_applicable`/`n_applicable` (2.4)** in `run_simple_val_inference`,
      plus finish committing the existing `NOT CODED` skip. *(Done 2026-06-04: added
      `_is_applicable_gold` helper, per-target `n_applicable` + `accuracy_applicable`,
      printed `n_app`/`acc_app` columns + both CSV columns, timing row mirrors keys.)*
- [x] **D1/D2 — set-based scoring + `Other(*)`.** `run_simple_val_inference` scores
      `multi_value` fields as order-insensitive sets and accepts `Other(*)` in_label via
      `allow_other_paren`. D3 resolved: keep all 2300 rows (plan default).
- [x] **T6 — Shared helper + port `gemma3_finetune_fulltrain.py`.** Done:
      `agent_utils/africa_dataprep.py: load_3jun_training_df()` returns `(df, TARGETS)`;
      both jobs call it (fulltrain dropped the json + numeric `topic_mapping` flow).
- [x] **T7 — Local dry-run** on `data_examples/AFRICA-TRAIN-DB-3jun2026.csv`: 0 dropped,
      27-key label-name `targets_json`, E4 nulls=1589, conformance passes, string allowed
      8/351/385. (The `[SYSTEM]`/`[ASSISTANT]` train-print needs the model → Snellius.)
- [x] **T8 (foot-gun) — done.** Missing-system-prompt branch in both jobs now raises
      `FileNotFoundError` instead of silently loading the numeric `africa_prompt_2602.txt`.
- [ ] **T8 (deploy, manual on Snellius).** Copy `africa_prompt_2004.txt` →
      `/projects/prjs1308/africa_llm_data/prompts/africa_prompt_system.txt` AND place
      `AFRICA-TRAIN-DB-3jun2026.csv` at the `CSV_PATH`
      (`/projects/prjs1308/africa_llm_data/`). Launch under GNU Screen on a GPU node with
      the `vve_nxt` env.

---

## 4. Validation gates

- **T3:** after E4, `targets_json` contains 0 `"NOT CODED"`; `national_unity_narrow` is
  null on ~1589 rows; row count unchanged (2300).
- **T4:** conformance assertion passes (only exemptions: `Other(...)`, handled `NOT CODED`);
  every `TARGETS` key has a matching CSV column and vice-versa (27↔27); no numeric `allowed`
  values remain.
- **T5:** per-target CSV has `n_applicable` + `accuracy_applicable`; for an always-N/A
  field, overall acc is high but `accuracy_applicable` reflects real signal.
- **T6:** both jobs import the same helper; a diff of their data-prep is empty.
- **T7:** all of the above pass against the local 3jun sample before any Snellius run.

---

## 5. Open decisions — RESOLVED 2026-06-04

- **D1 (S1) — `;` multi-value scoring.** ✅ **Set-based comparison.** `multi_value` fields
  (`language`, `topic`, `resource_distribution_by_whom1`, `resource_distribution_for_whom1`)
  are canonicalized to sorted atoms before accuracy/PRF, so `"a;b" == "b;a"`.
- **D2 (S2) — `Other(<lang>)`.** ✅ **Accept `Other(*)` by regex.** Exempt from the
  conformance assertion and counted in_label via `allow_other_paren` on `language`.
- **D3 (S5) — CSV hygiene.** ✅ **Keep all 2300 rows** (plan default). The 4 duplicate-id
  re-annotations and 8 blank-`topic` rows are retained (≈0.5%, blank topic → null).

---

## 6. Files touched (remaining)

| File | Change |
|---|---|
| `jobs/gemma3_finetune.py` | T3 (E4), T4 (TARGETS + assertion) — or delegate to shared helper |
| `jobs/gemma3_finetune_fulltrain.py` | T6 (same data-prep + TARGETS via helper) |
| `agent_utils/africa_dataprep.py` *(new, T6)* | shared `load_3jun_training_df()` |
| `agent_utils/gemma3_finetune_simple.py` | T5: `accuracy_applicable`/`n_applicable` (+ commit `NOT CODED` skip) |
| `agent_utils/utils.py` | verify-only: `_EVAL_KEY_ALIASES` (likely no change) |
| Snellius `prompts/africa_prompt_system.txt` | T8: deploy corrected codebook |
| Snellius `AFRICA-TRAIN-DB-3jun2026.csv` | T8: deploy at `CSV_PATH` |

---

## 7.5 Scientific review + hardening (2026-06-04)

An independent methodology review found three issues beyond the migration scope:

- **B1 (was experiment-voiding) — FIXED.** The codebook system prompt (~5–8k tokens)
  exceeded `max_tokens=4096`, so the transcript budget was 0 → every training transcript
  was empty and the answer JSON was right-truncated. **Rule: never truncate the codebook,
  only the transcript.** `_truncate_transcript()` now caps the transcript at
  `max_text_tokens` (default **500**) and **raises** if the codebook+answer can't fit
  `max_seq_length`; both jobs set `max_tokens=12288`. Train and inference cap text
  identically. (~18% of transcripts exceed 500 tok → raise `max_text_tokens` to keep more.)
- **B2 — accepted as-is.** 4 duplicate transcripts (some cross-`id`) land in both train and
  test under the random split. **Decision: keep all 2300 rows** (D3). Report test metrics
  with the caveat that those 4 are mildly inflated.
- **B3 — FIXED.** Removed the `train_df[:30]`/`test_df[:10]` debug slice in
  `gemma3_finetune.py`; real runs use the full data.

Non-blocking (left for judgement): severe class imbalance on several fields (report
**macro-F1 / per-class recall**, not accuracy); sawtooth LR from the per-epoch
`trainer.train()` loop; codebook example-JSON key order ≠ gold order (harmless, eval is
order-insensitive).

## 7. Done this session (for reference)

- T1: §26 typo fixed; example JSON confirmed label-name (27/27 keys).
- T2: 3jun CSV data-loading in `gemma3_finetune.py`; 12/12 validation gates pass.
- Verified 3jun = 1jun + E3 fixes + embedded text; chose 3jun as source of truth.
- Verified `TARGETS` allowed-sets against 3jun data: 26/27 exact, only `Other(...)` (D2)
  outstanding.
- T3: re-validated done — E4 blank-to-null at `gemma3_finetune.py:75-77`, before
  `targets_json`.
- T4: re-validated **not done** — `TARGETS` still the numeric/`topic01`/binary version,
  no conformance assertion. Still gated on D2.
- T5: implemented `n_applicable` + `accuracy_applicable` in `run_simple_val_inference`
  (`gemma3_finetune_simple.py`); logic-tested gated vs all-N/A field behaviour.
