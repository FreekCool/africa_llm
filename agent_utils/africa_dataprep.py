"""Shared data-prep for the 3jun training dataset + 2004 codebook.

Both Gemma-3 jobs (``jobs/gemma3_finetune.py`` and
``jobs/gemma3_finetune_fulltrain.py``) load and prepare the data through
``load_3jun_training_df`` so their data-prep can't drift. The caller does its
own train/test split (80/20 vs full-train).
"""

import re
import json

import pandas as pd

# Codebook §12 topic labels (up to 3, ;-joined). "not applicable" is retained
# even though §12 doesn't spell it out — gated rows can carry it (S4).
TOPIC_ALLOWED = [
    "NO TOPIC",
    "ECONOMY",
    "CIVIL RIGHTS",
    "HEALTH",
    "AGRICULTURE",
    "LABOR",
    "EDUCATION",
    "ENVIRONMENT",
    "ENERGY",
    "IMMIGRATION",
    "TRANSPORTATION",
    "LAW AND CRIME",
    "SOCIAL WELFARE",
    "HOUSING",
    "DOMESTIC COMMERCE",
    "DEFENSE",
    "TECHNOLOGY",
    "FOREIGN TRADE",
    "INTERNATIONAL AFFAIRS",
    "GOVERNMENT OPERATIONS",
    "PUBLIC LANDS",
    "CULTURE",
    "ETHNICITY",
    "not applicable",
]

_OTHER_PAREN_RE = re.compile(r"^Other\(.*\)$")


def build_targets_spec() -> dict:
    """Return a fresh copy of the 2004 codebook label-name target spec (§1–§27).

    Every ``allowed`` set is the codebook's label-name list, data-verified against
    the 3jun CSV: 26/27 categorical fields match exactly; the only out-of-codebook
    values are language's free-form Other(<lang>), accepted via ``allow_other_paren``
    (D2). ``multi_value`` fields are ``;``-joined and scored as order-insensitive
    sets (D1). Conditional fields are multiclass (they carry "not applicable" /
    "unclear"), not binary. String-target ``allowed`` lists are filled from the
    data by ``load_3jun_training_df``.
    """
    return {
        "language": {"type": "multiclass", "multi_value": True, "allow_other_paren": True, "allowed": [
            "English", "French", "Arabic", "Portuguese", "Swahili", "Hausa", "Yoruba",
            "Other", "Unclear"]},
        "politics": {"type": "multiclass", "allowed": [
            "politics", "not political", "unclear"]},
        "domestic_politics": {"type": "multiclass", "allowed": [
            "domestic politics", "not domestic politics", "unclear", "not applicable"]},
        "foreign_politics": {"type": "multiclass", "allowed": [
            "foreign politics", "not foreign politics", "unclear", "not applicable"]},
        "resource_distribution": {"type": "multiclass", "allowed": [
            "resource distribution", "not resource distribution", "unclear", "not applicable"]},
        "resource_distribution_by_whom1": {"type": "multiclass", "multi_value": True, "allowed": [
            "other state", "international organisation", "national government", "other",
            "not specified", "unclear", "not applicable"]},
        "resource_distribution_for_whom1": {"type": "multiclass", "multi_value": True, "allowed": [
            "specific locality or group", "country-wide", "not specified", "unclear",
            "not applicable"]},
        "resource_distribution_for_whom_ethnic1": {
            "type": "string",
            "allowed": [],
            "eval": {
                "metric": "exact",              # exact match after normalization
                "normalize": ["strip", "lower", "collapse_ws"],
                "empty_allowed": True,          # because it's only required when a condition holds
                "track_unique_incorrect": True,
                "max_unique_incorrect": 200,
            },
        },
        "resource_distribution_for_whom_region1": {
            "type": "string",
            "allowed": [],
            "eval": {
                "metric": "exact",
                "normalize": ["strip", "lower", "collapse_ws"],
                "empty_allowed": True,
                "track_unique_incorrect": True,
                "max_unique_incorrect": 200,
            },
        },
        "resource_distribution_gender": {"type": "multiclass", "allowed": [
            "resources for women", "resources not specifically for women", "unclear",
            "not applicable"]},
        "climate_change": {"type": "multiclass", "allowed": [
            "mentions climate change", "mentions sustainability", "unclear", "not applicable"]},
        "topic": {"type": "multiclass", "multi_value": True, "allowed": list(TOPIC_ALLOWED)},
        "pro_us": {"type": "multiclass", "allowed": [
            "positive towards the US", "neutral towards the US", "negative towards the US",
            "unclear", "no mention of the US", "not applicable"]},
        "pro_russia": {"type": "multiclass", "allowed": [
            "positive towards Russia", "neutral towards Russia", "negative towards Russia",
            "unclear", "no mention of Russia", "not applicable"]},
        "pro_china": {"type": "multiclass", "allowed": [
            "positive towards China", "neutral towards China", "negative towards China",
            "unclear", "no mention of China", "not applicable"]},
        "pro_un": {"type": "multiclass", "allowed": [
            "positive towards the UN", "neutral towards the UN", "negative towards the UN",
            "unclear", "no mention of the UN", "not applicable"]},
        "pro_imf": {"type": "multiclass", "allowed": [
            "positive towards the IMF", "neutral towards the IMF", "negative towards the IMF",
            "unclear", "no mention of the IMF", "not applicable"]},
        "pro_democracy": {"type": "multiclass", "allowed": [
            "positive towards democracy", "neutral towards democracy", "negative towards democracy",
            "unclear", "no mention of democracy", "not applicable"]},
        "anti_western": {"type": "multiclass", "allowed": [
            "anti-western", "not anti-western", "unclear", "not applicable"]},
        "national_unity": {"type": "multiclass", "allowed": [
            "national unity", "no mention of the nation or national unity", "unclear"]},
        "national_unity_narrow": {"type": "multiclass", "allowed": [
            "patriotism", "not specifically patriotic", "unclear"]},  # NOT CODED → null (E4)
        "subgroup_unity": {"type": "multiclass", "allowed": [
            "subgroup unity", "no mention of specific subgroup", "unclear"]},
        "subgroup_unity_text": {
            "type": "string",
            "allowed": [],
            "eval": {
                "metric": "exact",
                "normalize": ["strip", "lower", "collapse_ws"],
                "empty_allowed": True,
                "track_unique_incorrect": True,
                "max_unique_incorrect": 500,
            },
        },
        "african_unity": {"type": "multiclass", "allowed": [
            "african unity", "no mention of africa", "unclear"]},
        "political_opponents": {"type": "multiclass", "allowed": [
            "mentions political opponents", "no mention of political opponents", "unclear",
            "not applicable"]},
        "political_opponents_viol": {"type": "multiclass", "allowed": [
            "mentions violent group", "no mention of violent group", "unclear", "not applicable"]},
        "religion": {"type": "multiclass", "allowed": [
            "religious", "no mention of religion", "unclear"]},
    }


def assert_codebook_conformance(df, targets: dict, label_cols) -> None:
    """Loud-failure guard: TARGETS keys must match the CSV label columns 1-for-1,
    and every non-string field's ``;``-split atomic values must be a subset of its
    ``allowed`` set. The only exemption is language's free-form Other(<lang>)
    (``allow_other_paren`` / D2). national_unity_narrow's "NOT CODED" is already
    nulled by E4, so it never reaches here. If this fails, fix the data at source —
    do not patch in code (hard rule: no workarounds).
    """
    target_keys, csv_keys = set(targets.keys()), set(label_cols)
    if target_keys != csv_keys:
        raise ValueError(
            "TARGETS keys != CSV label columns.\n"
            f"  only in TARGETS: {sorted(target_keys - csv_keys)}\n"
            f"  only in CSV:     {sorted(csv_keys - target_keys)}"
        )

    violations = {}
    for col, spec in targets.items():
        if spec.get("type") == "string":
            continue
        allowed_set = {str(a).strip() for a in spec["allowed"]}
        allow_other = spec.get("allow_other_paren", False)
        bad = {}
        for cell in df[col].dropna().astype(str):
            for atom in cell.split(";"):
                atom = atom.strip()
                if not atom or atom in allowed_set:
                    continue
                if allow_other and _OTHER_PAREN_RE.match(atom):
                    continue
                bad[atom] = bad.get(atom, 0) + 1
        if bad:
            violations[col] = dict(sorted(bad.items()))
    if violations:
        raise ValueError(
            "Out-of-codebook values found (fix the data at source, not in code):\n"
            + "\n".join(f"  {c}: {b}" for c, b in violations.items())
        )
    n_checked = sum(1 for s in targets.values() if s.get("type") != "string")
    print(f"[conformance] OK — {n_checked} categorical fields ⊆ codebook label sets; "
          f"TARGETS keys == CSV columns ({len(target_keys)})")


def load_3jun_training_df(csv_path):
    """Load + prepare the 3jun training dataframe and its target spec.

    Returns ``(df, TARGETS)`` where ``df`` has the embedded ``text`` plus a
    ``targets_json`` column (27 label-name keys), and ``TARGETS`` is the 2004
    codebook label-name spec with string-target ``allowed`` lists filled from this
    df. The caller does its own train/test split.

    Steps: read CSV, cast ``id`` float64→int64, drop rows without a transcript
    (logged), E4 blank ``national_unity_narrow == "NOT CODED"`` → null, build
    ``targets_json`` from the 27 label columns, fill string ``allowed``, and run the
    codebook-conformance guard.
    """
    df = pd.read_csv(csv_path)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")

    # Label columns = every column except the id key and the transcript text.
    label_cols = [c for c in df.columns if c not in ("id", "text")]

    # Keep only rows with a usable transcript; log how many are dropped.
    has_text = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
    n_total = len(df)
    n_with_text = int(has_text.sum())
    print(f"[data] rows={n_total} with_text={n_with_text} dropped={n_total - n_with_text}")
    df = df.loc[has_text].reset_index(drop=True)

    # E4: "NOT CODED" in national_unity_narrow is the annotators' "never coded this
    # field" sentinel (~1589 rows), not a real label. Blank it to null so the field
    # is excluded from the training target and from metrics; the row is kept. (Q6.)
    not_coded = df["national_unity_narrow"].astype(str).str.strip() == "NOT CODED"
    print(f"[data] national_unity_narrow NOT CODED -> null: {int(not_coded.sum())} rows")
    df.loc[not_coded, "national_unity_narrow"] = None

    # Build targets_json directly from the label columns. Values are already the
    # codebook label-name strings (e.g. topic = "TRANSPORTATION"), so no numeric recode.
    df["targets_json"] = df[label_cols].apply(
        lambda row: json.dumps(
            {c: (None if pd.isna(row[c]) else row[c]) for c in label_cols},
            ensure_ascii=False,
        ),
        axis=1,
    )

    targets = build_targets_spec()

    # Fill string targets' allowed list from all annotated values in the dataset.
    for t, spec in targets.items():
        if spec.get("type") != "string" or t not in df.columns:
            continue
        vals = df[t].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        spec["allowed"] = sorted(vals.unique().tolist())
        print(f"{t}: allowed = {len(spec['allowed'])} values")

    assert_codebook_conformance(df, targets, label_cols)
    return df, targets
