"""Load and split the Medical Abstracts TC Corpus.

Combines Kaggle's pre-made train/test CSVs into one pool and re-splits
with a fixed random_state, rather than using the shipped split — see
docs/architecture.md for the train/validation/test split rationale.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42
TEST_SIZE = 0.15


def load_raw() -> pd.DataFrame:
    """Read the raw CSVs, merge category names, and drop ambiguous abstracts.

    Combining Kaggle's train+test files reveals abstracts that appear more
    than once with *conflicting* category labels (documents the corpus
    creators originally multi-labeled, exploded into separate single-label
    rows by the Kaggle release) — see docs/technical-decisions.md. These are
    dropped entirely rather than arbitrarily kept under one label, since
    keeping one would inject an essentially random label for ~26% of
    documents. No duplicate group in this corpus repeats with a *matching*
    label, but the check below is written to only drop genuine conflicts.
    """
    labels = pd.read_csv(DATA_DIR / "medical_tc_labels.csv")
    train = pd.read_csv(DATA_DIR / "medical_tc_train.csv")
    test = pd.read_csv(DATA_DIR / "medical_tc_test.csv")

    df = pd.concat([train, test], ignore_index=True)
    df = df.merge(labels, on="condition_label", how="left")

    label_counts = df.groupby("medical_abstract")["condition_label"].nunique()
    ambiguous = label_counts[label_counts > 1].index
    if len(ambiguous):
        print(f"Dropped {len(ambiguous)} ambiguous (multi-labeled) abstract(s)")

    df = df[~df["medical_abstract"].isin(ambiguous)]
    df = df.drop_duplicates(subset="medical_abstract").reset_index(drop=True)

    return df


def split_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve out a stratified held-out test set; returns (pool, test)."""
    pool, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["condition_label"],
    )
    return pool.reset_index(drop=True), test.reset_index(drop=True)


def load_pool_and_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load raw data and return (pool, test)."""
    return split_test(load_raw())


if __name__ == "__main__":
    pool, test = load_pool_and_test()
    print(f"\nTotal: {len(pool) + len(test)}  Pool: {len(pool)}  Test: {len(test)}\n")
    print("Class distribution (condition_name):")
    dist = pd.DataFrame(
        {
            "pool": pool["condition_name"].value_counts(),
            "test": test["condition_name"].value_counts(),
        }
    )
    dist["pool_%"] = (dist["pool"] / dist["pool"].sum() * 100).round(1)
    dist["test_%"] = (dist["test"] / dist["test"].sum() * 100).round(1)
    print(dist)
