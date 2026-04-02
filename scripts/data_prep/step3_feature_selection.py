import os
import pandas as pd

INPUT_PATH = "data/interim/movielens_step1_genres_onehot.csv"
OUTPUT_PATH = "data/interim/movielens_step3_selected_features.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    print("Initial columns:", df.shape[1])

    # Columns to drop (intentional design choices)
    drop_cols = ["title", "zip", "timestamp"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    df = df.drop(columns=drop_cols)

    print("Dropped columns:", drop_cols)
    print("Remaining columns:", df.shape[1])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Step 3 complete: feature selection applied.")

if __name__ == "__main__":
    main()
