"""Krok 2: losowanie próbki do anotacji (warstwowo według etykiety gold).

Uruchomienie (z katalogu Lab1):  python -m src.sampling --iteration 1
Iteracja k wyklucza zdania z iteracji 1..k-1 oraz przykłady z wytycznych.
"""
import argparse

import numpy as np
import pandas as pd

import config
from src.data import load_corpus


def sampling_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["n_tokens"].between(config.MIN_TOKENS, config.MAX_TOKENS) & ~df["gold_conflict"]
    return df[keep]


def make_sample(df: pd.DataFrame, n: int, seed: int, exclude_ids=()) -> pd.DataFrame:
    """n // 4 zdań z każdej klasy gold, w losowej kolejności."""
    per_class, rest = divmod(n, len(config.SENT_LABELS))
    if rest:
        raise ValueError(f"n={n} musi być podzielne przez {len(config.SENT_LABELS)}")
    rng = np.random.default_rng(seed)
    frame = sampling_frame(df)
    frame = frame[~frame["sent_id"].isin(set(exclude_ids))]
    parts = []
    for label in config.SENT_LABELS:
        pool = frame[frame["gold"] == label]
        idx = rng.choice(len(pool), size=per_class, replace=False)
        parts.append(pool.iloc[idx])
    sample = pd.concat(parts)
    return sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)


def excluded_ids(iteration: int) -> set[int]:
    ids = {int(line) for line in config.EXAMPLE_IDS.read_text(encoding="utf-8").split()
           if line.strip().isdigit()}
    for k in range(1, iteration):
        prev = config.iter_dir(k) / "sample.csv"
        if not prev.exists():
            raise FileNotFoundError(f"Brak próbki z iteracji {k}: {prev}")
        ids |= set(pd.read_csv(prev)["sent_id"])
    return ids


def write_iteration(iteration: int, force: bool = False) -> pd.DataFrame:
    out = config.iter_dir(iteration)
    sample_path, key_path = out / "sample.csv", out / "_key_gold.csv"
    if sample_path.exists() and not force:
        raise FileExistsError(f"{sample_path} już istnieje (anotacja mogła się zacząć). Użyj --force.")
    sample = make_sample(load_corpus(), config.N_SAMPLE, config.SEEDS[iteration], excluded_ids(iteration))
    out.mkdir(parents=True, exist_ok=True)
    # Anotatorzy dostają tylko tekst; etykieta gold jest w osobnym pliku.
    sample[["sent_id", "text", "n_tokens"]].to_csv(sample_path, index=False, encoding="utf-8")
    sample[["sent_id", "gold", "gold_raw"]].to_csv(key_path, index=False, encoding="utf-8")
    return sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True, choices=sorted(config.SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    sample = write_iteration(args.iteration, args.force)
    print(f"Iteracja {args.iteration}: {len(sample)} zdań")
    print(sample["gold"].value_counts().to_string())
    print(sample["n_tokens"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
