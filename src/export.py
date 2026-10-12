"""Punkt 2 (eksport): wypełnione arkusze → tabele i fragmenty w ustalonym formacie.

Uruchomienie (z katalogu Lab1):  python -m src.export --iteration 1

Wejście:  annotation/iterN/sample.csv, _key_gold.csv (jeśli jest),
          annotation/iterN_artefacts/iterN_A*.xlsx (wypełnione arkusze)
Wyjście:  annotation/iterN/export/
  sentences_long.csv   sent_id, annotator, sent_label, uwagi
  sentences_wide.csv   sent_id, text, A1, A2, …, gold, zgodne
  tokens_long.csv      sent_id, tok_idx, token, annotator, label
  tokens_wide.csv      sent_id, tok_idx, token, A1, A2, …, zgodne
  spans_<A>.jsonl      {"sent_id", "text", "label", "spans": [[start, end, etykieta, fragment], …], "uwagi"}

tok_idx liczymy od 1 (jak kolumny t1..tK w arkuszu). start/end to pozycje znaków
w text (od 0, end wyłącznie). Fragment = kolejne tokeny z tą samą etykietą.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

import config
from src.sheets_io import read_annotations

SHEET_RE = re.compile(r"iter(\d+)_(A\d+)\.xlsx")


def find_sheets(iteration: int) -> dict[str, Path]:
    """{anotator: ścieżka} dla plików iterN_A*.xlsx z katalogu wypełnionych arkuszy."""
    found = {}
    for path in sorted(config.filled_sheets_dir(iteration).glob(f"iter{iteration}_A*.xlsx")):
        m = SHEET_RE.fullmatch(path.name)
        if m and int(m.group(1)) == iteration:
            found[m.group(2)] = path
    if not found:
        raise FileNotFoundError(f"Brak arkuszy iter{iteration}_A*.xlsx w {config.filled_sheets_dir(iteration)}")
    return found


def load_iteration(iteration: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Próbka (z kolumną gold, jeśli jest klucz) i anotacje każdego anotatora w kolejności próbki.

    Arkusz z jakimkolwiek problemem (brak etykiety, zmienione tokeny, etykieta spoza
    listy) zatrzymuje eksport – zgodność liczona na niepełnych danych byłaby myląca.
    """
    d = config.iter_dir(iteration)
    sample = pd.read_csv(d / "sample.csv")
    key = d / "_key_gold.csv"
    if key.exists():
        sample = sample.merge(pd.read_csv(key)[["sent_id", "gold"]], on="sent_id", how="left")
    raw, problems = {}, []
    for a, path in find_sheets(iteration).items():
        df, issues = read_annotations(path, sample, a)
        raw[a] = df
        problems += [f"{path.name}: {i}" for i in issues]
    if problems:
        raise ValueError("Arkusze wymagają poprawek:\n  " + "\n  ".join(problems))
    ann = {a: df.set_index("sent_id").loc[sample["sent_id"]] for a, df in raw.items()}
    return sample, ann


def token_spans(text: str, labels: list[str]) -> list[list]:
    """Etykiety tokenów → fragmenty [start, end, etykieta, tekst] (kolejne tokeny z tą samą etykietą)."""
    tokens = text.split()
    if len(tokens) != len(labels):
        raise ValueError(f"{len(tokens)} tokenów, {len(labels)} etykiet: {text!r}")
    spans, pos, prev = [], 0, config.NO_LABEL
    for tok, lab in zip(tokens, labels):
        start = text.index(tok, pos)
        pos = start + len(tok)
        if lab != config.NO_LABEL and lab == prev:
            spans[-1][1] = pos
        elif lab != config.NO_LABEL:
            spans.append([start, pos, lab])
        prev = lab
    return [[s, e, lab, text[s:e]] for s, e, lab in spans]


def build_tables(sample: pd.DataFrame, ann: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    annotators = list(ann)
    has_gold = "gold" in sample.columns

    sent_long = pd.concat(
        [pd.DataFrame({"sent_id": sample["sent_id"].values, "annotator": a,
                       "sent_label": df["sent_label"].values, "uwagi": df["uwagi"].values})
         for a, df in ann.items()], ignore_index=True)

    sent_wide = sample[["sent_id", "text"]].copy()
    for a, df in ann.items():
        sent_wide[a] = df["sent_label"].values
    if has_gold:
        sent_wide["gold"] = sample["gold"].values
    sent_wide["zgodne"] = sent_wide[annotators].nunique(axis=1) == 1

    rows = []
    for i, (sid, text) in enumerate(zip(sample["sent_id"], sample["text"])):
        for j, tok in enumerate(text.split()):
            rows.append([sid, j + 1, tok] + [ann[a]["token_labels"].iat[i][j] for a in annotators])
    tok_wide = pd.DataFrame(rows, columns=["sent_id", "tok_idx", "token"] + annotators)
    tok_wide["zgodne"] = tok_wide[annotators].nunique(axis=1) == 1
    tok_long = (tok_wide.melt(id_vars=["sent_id", "tok_idx", "token"], value_vars=annotators,
                              var_name="annotator", value_name="label")
                .sort_values(["annotator", "sent_id", "tok_idx"], kind="stable")
                .reset_index(drop=True))
    return {"sentences_long": sent_long, "sentences_wide": sent_wide,
            "tokens_long": tok_long, "tokens_wide": tok_wide}


def build_spans(sample: pd.DataFrame, ann: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    out = {}
    for a, df in ann.items():
        out[a] = [{"sent_id": int(sid), "text": text, "label": df["sent_label"].iat[i],
                   "spans": token_spans(text, df["token_labels"].iat[i]), "uwagi": df["uwagi"].iat[i]}
                  for i, (sid, text) in enumerate(zip(sample["sent_id"], sample["text"]))]
    return out


def write_export(iteration: int) -> dict[str, Path]:
    sample, ann = load_iteration(iteration)
    out = config.export_dir(iteration)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in build_tables(sample, ann).items():
        paths[name] = out / f"{name}.csv"
        df.to_csv(paths[name], index=False, encoding="utf-8-sig")  # BOM: polskie znaki w Excelu
    for a, records in build_spans(sample, ann).items():
        paths[f"spans_{a}"] = out / f"spans_{a}.jsonl"
        with open(paths[f"spans_{a}"], "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    args = p.parse_args()
    for name, path in write_export(args.iteration).items():
        print(f"{name:15s} → {path}")


if __name__ == "__main__":
    main()
