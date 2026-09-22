"""Phase 2: builds the Phase 2 training set by running the offline,
point-in-time-correct feature pipeline (src/features/offline.py) over the
full historical data and attaching the engineered columns to the original
raw columns. Output feeds train_with_features.py.

Run from the repo root:
    python -m src.training.build_training_set
"""
from src.features.definitions import ENTITY_KEY_COLS, FEATURE_COLUMNS
from src.features.entity import build_entity_id
from src.features.offline import compute_features_batch
from src.training.common import PROCESSED, load_raw


def main():
    print("loading raw data...")
    df = load_raw()
    print(f"rows: {len(df):,}, cols: {df.shape[1]}")

    missing_key = df[ENTITY_KEY_COLS].isna().any(axis=1)
    print(f"rows missing >=1 entity key field (get singleton IDs): {missing_key.sum():,} ({missing_key.mean():.2%})")

    df["entity_id"] = build_entity_id(df)
    print(f"unique entities: {df['entity_id'].nunique():,}")

    print("computing point-in-time features...")
    feats = compute_features_batch(df)
    for col in FEATURE_COLUMNS:
        df[col] = feats[col]

    print("\nfeature summary:")
    print(df[FEATURE_COLUMNS].describe())
    print("\nnull rates:")
    print(df[FEATURE_COLUMNS].isna().mean())

    PROCESSED.mkdir(exist_ok=True, parents=True)
    out_path = PROCESSED / "train_features.parquet"
    df.to_parquet(out_path)
    print(f"\nsaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
