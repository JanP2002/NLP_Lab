"""Krok 1: wczytanie wersji zdaniowej PolEmo 2.0 i przygotowanie do anotacji.

Uruchomienie (z katalogu Lab1):  python -m src.data
"""
import json

import pandas as pd

import config


def parse_line(line: str) -> tuple[str, str]:
    """'Tekst zdania . __label__z_zero' -> ('Tekst zdania .', 'z_zero')."""
    if config.LABEL_SEP not in line:
        raise ValueError(f"Brak separatora etykiety: {line!r}")
    text, label = line.rsplit(config.LABEL_SEP, 1)
    return text.strip(), label.strip()


def load_raw_lines() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("text", data_files=config.DATA_URL)["train"]
    return ds["text"]


def build_corpus(lines: list[str]) -> tuple[pd.DataFrame, dict]:
    """Zwraca tabelę zdań (bez duplikatów) i liczniki z czyszczenia.

    sent_id = numer linii w pliku źródłowym (stały identyfikator zdania).
    Tokenizacji nie zmieniamy: split() po spacjach daje tokeny korpusu.
    """
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        text, raw = parse_line(line)
        if raw not in config.GOLD_MAP:
            raise ValueError(f"Nieznana etykieta {raw!r} w linii {i}")
        rows.append((i, text, raw, config.GOLD_MAP[raw]))
    df = pd.DataFrame(rows, columns=["sent_id", "text", "gold_raw", "gold"])
    df = df[df["text"] != ""]

    # Duplikaty: zostawiamy pierwsze wystąpienie. Jeśli kopie mają różne etykiety
    # gold, zdanie oznaczamy (gold_conflict) i nie losujemy go – etykieta
    # warstwy byłaby niejednoznaczna.
    n_labels = df.groupby("text")["gold"].transform("nunique")
    df = df.assign(n_copies=df.groupby("text")["text"].transform("size"),
                   gold_conflict=n_labels > 1)
    counts = {
        "lines": len(lines),
        "parsed": len(df),
        "duplicate_rows_removed": int(df.duplicated("text").sum()),
    }
    df = df.drop_duplicates("text", keep="first").reset_index(drop=True)
    df["n_tokens"] = df["text"].str.split().str.len()
    counts["unique_sentences"] = len(df)
    counts["gold_conflict_sentences"] = int(df["gold_conflict"].sum())
    return df, counts


def corpus_stats(df: pd.DataFrame) -> dict:
    q = df["n_tokens"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    in_frame = df["n_tokens"].between(config.MIN_TOKENS, config.MAX_TOKENS) & ~df["gold_conflict"]
    return {
        "gold_counts": df["gold"].value_counts().reindex(config.SENT_LABELS).to_dict(),
        "gold_share": df["gold"].value_counts(normalize=True).round(4).reindex(config.SENT_LABELS).to_dict(),
        "n_tokens_quantiles": {f"q{int(k * 100)}": float(v) for k, v in q.items()},
        "n_tokens_mean": round(float(df["n_tokens"].mean()), 2),
        "sampling_frame_size": int(in_frame.sum()),
        "sampling_frame_gold_counts": df.loc[in_frame, "gold"].value_counts().reindex(config.SENT_LABELS).to_dict(),
    }


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED)


def main():
    df, counts = build_corpus(load_raw_lines())
    config.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED, index=False)

    stats = {"cleaning": counts, **corpus_stats(df)}
    config.CORPUS_STATS.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nZapisano {len(df)} zdań do {config.PROCESSED}")


if __name__ == "__main__":
    main()
