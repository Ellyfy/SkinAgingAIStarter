"""
Build a nearest-neighbor reference index from REAL extracted features.
Do not generate fake labels for research use.
"""
from pathlib import Path
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "forehead_wrinkle",
    "left_eye_wrinkle",
    "right_eye_wrinkle",
    "left_cheek_wrinkle",
    "right_cheek_wrinkle",
    "nasolabial_wrinkle",
    "chin_wrinkle",
]

def main():
    path = Path("data/reference_features.csv")
    if not path.exists():
        raise SystemExit(
            "Create data/reference_features.csv from real extracted data first."
        )

    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    scaler = StandardScaler()
    x = scaler.fit_transform(df[FEATURES].astype(float).values)
    knn = NearestNeighbors(n_neighbors=min(25, len(df)), metric="euclidean")
    knn.fit(x)

    Path("models").mkdir(exist_ok=True)
    with open("models/reference_index.pkl", "wb") as f:
        pickle.dump(
            {
                "features": FEATURES,
                "scaler": scaler,
                "knn": knn,
                "metadata": df.drop(columns=FEATURES),
            },
            f,
        )

    print(f"Built reference index with {len(df)} rows.")

if __name__ == "__main__":
    main()
