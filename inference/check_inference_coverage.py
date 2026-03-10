import os
import pandas as pd


OUTPUT_DIR = "/projects/prjs1308/africa_llm_data/inference_results"


def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output directory does not exist: {OUTPUT_DIR}")
        return

    all_ids = set()
    n_files = 0

    for fname in os.listdir(OUTPUT_DIR):
        if not (fname.startswith("inference_predictions_") and fname.endswith(".csv")):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        try:
            df = pd.read_csv(path, usecols=["id"])
        except Exception as e:
            print(f"Warning: could not read ids from {path}: {e}")
            continue
        n_files += 1
        all_ids.update(df["id"].dropna().tolist())

    print(f"Scanned {n_files} inference CSV files in {OUTPUT_DIR}")
    print(f"Unique ids with inference results: {len(all_ids)}")


if __name__ == "__main__":
    main()

