import pandas as pd

INPUT_PATH = "data/interim/movielens_step1_genres_onehot.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    print("\n=== BASIC SHAPE ===")
    print("Rows, Columns:", df.shape)

    print("\n=== MISSING VALUES (Top 30) ===")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.head(30))

    print("\n=== RATING VALIDITY CHECK (Outliers) ===")
    if "rating" not in df.columns:
        print("WARNING: 'rating' column not found!")
    else:
        invalid_ratings = df[(df["rating"] < 1) | (df["rating"] > 5)]
        print("Invalid rating rows:", len(invalid_ratings))

    print("\nStep 2 checks complete.\n")

if __name__ == "__main__":
    main()
