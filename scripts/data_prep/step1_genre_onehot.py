import os
import pandas as pd

INPUT_PATH = "data/rawData/merged_movielens.csv"
OUTPUT_PATH = "data/interim/movielens_step1_genres_onehot.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # Split genres like "Action|Comedy" -> ["Action", "Comedy"]
    genres_list = df["genres"].fillna("").astype(str).str.split("|")

    # Collect unique genres
    unique_genres = sorted({g for row in genres_list for g in row if g})

    # Create one-hot encoded columns
    for g in unique_genres:
        df[f"genre_{g}"] = genres_list.apply(lambda row: 1 if g in row else 0)

    # Drop original genres column
    df = df.drop(columns=["genres"])

    # Save processed file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Step 1 complete")
    print(f"Genre columns added: {len(unique_genres)}")

if __name__ == "__main__":
    main()
