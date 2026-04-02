import os
import pandas as pd

INPUT_PATH = "data/interim/movielens_step3_selected_features.csv"
OUTPUT_DIR = "data/processed"

TRAIN_RATIO = 0.35
VAL_RATIO = 0.35
TEST_RATIO = 0.30

def main():
    df = pd.read_csv(INPUT_PATH)

    # Convert and sort by datetime
    df["rating_datetime"] = pd.to_datetime(df["rating_datetime"])
    df = df.sort_values("rating_datetime").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "validate.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Time-based split complete.")
    print(f"Train size: {len(train_df)}")

if __name__ == "__main__":
    main()
