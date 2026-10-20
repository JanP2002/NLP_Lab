"""Zadanie 1, Część II – Word Embeddings, pkt 1: osadzenia słów zaanotowanego korpusu.

Uruchomienie (z katalogu Lab1):

    python -m src.word_embeddings                          # word2vec IPI PAN + word2vec PolEmo
    python -m src.word_embeddings --models w2v fasttext    # + fastText KGR10 (drugi model, ~3,7 GB)

Wejście:  annotation/iterN/export/tokens_long.csv (tokeny z etykietami A1/A2),
          annotation/iterN/adjudicated.csv (etykieta zdania „final”), models/ (modele)
Wyjście (results/word_embeddings/):
  vocab.csv            słowa zaanotowanych zdań + statystyki anotacji + pokrycie modeli
  vectors_<model>.npz  words, vectors (float32), in_vocab – słowa korpusu, dla których jest wektor
  coverage.json/.md    pokrycie słownika: ogółem, według etykiety słowa, słowa bez wektora
  sanity.md            najbliższe słowa dla słów kontrolnych i podobieństwa wybranych par

Słowo = token korpusu zawierający literę, zapisany małymi literami (interpunkcję i same
liczby pomijamy). Wektora szukamy kolejno dla: formy małymi literami, najczęstszej
oryginalnej pisowni, formy z wielką literą na początku (słownik IPI PAN zachowuje wielkie
litery głównie w nazwach własnych).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import config
from src import text_embeddings as te

LETTER_RE = re.compile(r"[^\W\d_]")
SENT_SHARE_COLS = [f"share_{l}" for l in config.SENT_LABELS]
PROBE_WORDS = ["dobry", "polecam", "lekarz", "brud", "bardzo", "nie"]
PROBE_PAIRS = [("dobry", "zły"), ("dobry", "świetny"), ("polecam", "odradzam"),
               ("czysto", "brudno"), ("bardzo", "trochę")]


# --------------------------------------------------------------------------
# Słownik zaanotowanego korpusu
# --------------------------------------------------------------------------
def is_word(token: str) -> bool:
    return bool(LETTER_RE.search(token))


def annotated_tokens(iterations: Iterable[int] = (1, 2)) -> pd.DataFrame:
    """Wystąpienia słów z etykietami anotatorów (jeden wiersz na token × anotatora).

    Kolumny: iter, sent_id, tok_idx, token, word, annotator, label, final.
    """
    parts = []
    for it in iterations:
        path = config.export_dir(it) / "tokens_long.csv"
        if not path.exists():
            raise FileNotFoundError(f"Brak {path}. Uruchom: python -m src.export --iteration {it}")
        t = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype={"token": str})
        t["iter"] = it
        parts.append(t)
    tok = pd.concat(parts, ignore_index=True)
    tok = tok[tok["token"].map(is_word)].copy()
    tok["word"] = tok["token"].str.lower()
    final = te.load_sentences(iterations)[["sent_id", "final"]]
    return tok.merge(final, on="sent_id", how="left")


def word_label(counts: dict[str, int], n_annotations: int) -> str:
    """Etykieta słowa: najczęstsza etykieta ≠ O, gdy anotacje ≠ O to co najmniej połowa.

    Remis rozstrzyga kolejność ``config.TOKEN_LABELS``.
    """
    marked = sum(counts.get(l, 0) for l in config.TOKEN_LABELS)
    if n_annotations == 0 or marked / n_annotations < 0.5:
        return config.NO_LABEL
    return max(config.TOKEN_LABELS, key=lambda l: (counts.get(l, 0), -config.TOKEN_LABELS.index(l)))


def polemo_frequencies() -> Counter:
    """Liczba wystąpień słów (małe litery) w całym korpusie PolEmo (wersja zdaniowa)."""
    df = pd.read_parquet(config.PROCESSED, columns=["text"])
    return Counter(t.lower() for text in df["text"] for t in text.split() if is_word(t))


def annotated_vocabulary(iterations: Iterable[int] = (1, 2), polemo_freq: Counter | None = None) -> pd.DataFrame:
    """Jeden wiersz na słowo: liczności, etykiety anotatorów, kontekst zdań (``final``)."""
    tok = annotated_tokens(iterations)
    occ = tok.drop_duplicates(["sent_id", "tok_idx"])  # wystąpienia (bez powtórzeń anotatorów)
    labels = [config.NO_LABEL] + config.TOKEN_LABELS
    counts = pd.crosstab(tok["word"], tok["label"]).reindex(columns=labels, fill_value=0)

    vocab = pd.DataFrame({
        "freq": occ.groupby("word").size(),
        "n_sentences": occ.groupby("word")["sent_id"].nunique(),
        "surface": occ.groupby("word")["token"].agg(lambda s: s.value_counts().index[0]),
    })
    for l in labels:
        vocab[f"n_{l}"] = counts[l]
    n_ann = counts.sum(axis=1)
    vocab["annot_share"] = (n_ann - counts[config.NO_LABEL]) / n_ann
    vocab["word_label"] = [word_label(counts.loc[w].to_dict(), int(n_ann[w])) for w in vocab.index]
    shares = pd.crosstab(occ["word"], occ["final"], normalize="index").reindex(
        columns=config.SENT_LABELS, fill_value=0.0)
    for l in config.SENT_LABELS:
        vocab[f"share_{l}"] = shares[l].reindex(vocab.index)
    freq = polemo_freq if polemo_freq is not None else polemo_frequencies()
    vocab["polemo_freq"] = [freq.get(w, 0) for w in vocab.index]
    vocab = vocab.rename_axis("word").reset_index()
    return vocab.sort_values(["freq", "word"], ascending=[False, True], ignore_index=True)


def read_vocab(out_dir=None) -> pd.DataFrame:
    """Odczyt ``vocab.csv`` z poprawnymi typami.

    keep_default_na=False: słowa typu „null” czy „nan” nie mogą stać się brakami danych;
    kolumny liczbowe i logiczne konwertujemy jawnie.
    """
    out_dir = out_dir or config.WORD_EMB_DIR
    vocab = pd.read_csv(out_dir / "vocab.csv", encoding="utf-8-sig", keep_default_na=False,
                        dtype={"word": str, "surface": str})
    for c in vocab.columns:
        if c in SENT_SHARE_COLS or c.endswith("_rank"):
            vocab[c] = pd.to_numeric(vocab[c], errors="coerce")
        elif c.endswith("_in_vocab"):
            vocab[c] = vocab[c].astype(str).eq("True")
    return vocab


# --------------------------------------------------------------------------
# Modele i wektory
# --------------------------------------------------------------------------
@dataclass
class WordModel:
    name: str
    label: str
    loader: Callable[[], object]
    subword: bool  # fastText: wektor także dla słów spoza słownika (n-gramy znakowe)


def polemo_sentences() -> list[list[str]]:
    df = pd.read_parquet(config.PROCESSED, columns=["text"])
    return [[t.lower() for t in text.split() if is_word(t)] for text in df["text"]]


def train_polemo_w2v(sentences: list[list[str]] | None = None, params: dict | None = None,
                     path=None, force: bool = False):
    """Word2vec wytrenowany w gensim na zdaniach PolEmo (model dziedzinowy). Zapis w models/."""
    from gensim.models import KeyedVectors, Word2Vec

    path = path or config.MODELS_DIR / "w2v_polemo.kv"
    if path.exists() and not force:
        return KeyedVectors.load(str(path))
    params = params or config.W2V_POLEMO
    model = Word2Vec(sentences if sentences is not None else polemo_sentences(), **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(path))
    path.with_suffix(".json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return model.wv


MODELS = {
    "w2v": WordModel("w2v", "Word2Vec IPI PAN (nkjp+wiki, formy, skip-gram, 100 wym., 300 tys. słów)",
                     lambda: te._load_w2v(), subword=False),
    "w2v_polemo": WordModel("w2v_polemo", "Word2Vec wytrenowany na PolEmo 2.0 (gensim, skip-gram, 100 wym.)",
                            train_polemo_w2v, subword=False),
    "fasttext": WordModel("fasttext", f"fastText KGR10 ({config.FASTTEXT_FILE})",
                          lambda: te._load_fasttext(None), subword=True),
}


def lookup(kv, word: str, surface: str, subword: bool = False):
    """(wektor, forma znaleziona w słowniku albo None, czy forma jest w słowniku).

    Kolejność: małe litery → oryginalna pisownia → wielka litera na początku. Model
    z n-gramami (fastText) zwraca wektor także dla słowa spoza słownika.
    """
    for form in dict.fromkeys([word, surface, word.capitalize()]):
        if form in kv.key_to_index:
            return kv[form], form, True
    if subword:
        return kv[word], None, False
    return None, None, False


def embed_vocabulary(vocab: pd.DataFrame, kv, name: str, subword: bool = False) -> tuple[pd.DataFrame, dict]:
    """Dodaje do ``vocab`` kolumny ``<name>_in_vocab``, ``<name>_form``, ``<name>_rank``.

    Zwraca też słownik do zapisu ``.npz``: words, vectors (float32), in_vocab.
    """
    words, vecs, flags, forms, ranks = [], [], [], [], []
    for w, s in zip(vocab["word"], vocab["surface"]):
        v, form, in_vocab = lookup(kv, w, s, subword)
        flags.append(in_vocab)
        forms.append(form or "")
        ranks.append(kv.key_to_index[form] + 1 if in_vocab else np.nan)
        if v is not None:
            words.append(w)
            vecs.append(np.asarray(v, dtype=np.float32))
    vocab = vocab.copy()
    vocab[f"{name}_in_vocab"] = flags
    vocab[f"{name}_form"] = forms
    vocab[f"{name}_rank"] = ranks
    in_vocab = vocab.set_index("word")[f"{name}_in_vocab"]
    arrays = {"words": np.array(words), "vectors": np.vstack(vecs) if vecs else np.zeros((0, kv.vector_size), np.float32),
              "in_vocab": np.array([bool(in_vocab[w]) for w in words])}
    return vocab, arrays


# --------------------------------------------------------------------------
# Pokrycie i kontrola sensowności
# --------------------------------------------------------------------------
def ascii_fold(s: str) -> str:
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def oov_reason(word: str, polemo_freq: int, folded_index: dict[str, str]) -> str:
    if any(c.isdigit() for c in word):
        return "zawiera cyfrę"
    if not all(c.isalpha() or c == "-" for c in word):
        return "znak specjalny / tokenizacja"
    if word == ascii_fold(word) and word in folded_index:
        return f"brak polskich znaków (w modelu: {folded_index[word]})"
    if polemo_freq <= 2:
        return "rzadkie także w PolEmo (≤ 2 wystąpienia)"
    return "inne (literówka, forma rzadka lub złożona)"


def coverage(vocab: pd.DataFrame, name: str, kv, n_examples: int = 20) -> dict:
    hit = vocab[f"{name}_in_vocab"]
    tokens_total = int(vocab["freq"].sum())
    annotated = vocab["annot_share"] > 0
    res = {
        "types": {"n": len(vocab), "in_vocab": int(hit.sum()), "share": float(hit.mean())},
        "tokens": {"n": tokens_total, "in_vocab": int(vocab.loc[hit, "freq"].sum()),
                   "share": float(vocab.loc[hit, "freq"].sum() / tokens_total)},
        "annotated_types": {"n": int(annotated.sum()), "in_vocab": int((hit & annotated).sum()),
                            "share": float(hit[annotated].mean()) if annotated.any() else float("nan")},
        "by_word_label": {},
    }
    for l in [config.NO_LABEL] + config.TOKEN_LABELS:
        m = vocab["word_label"] == l
        res["by_word_label"][l] = {"n": int(m.sum()), "in_vocab": int((hit & m).sum()),
                                   "share": float(hit[m].mean()) if m.any() else float("nan")}
    oov = vocab[~hit].head(n_examples)
    if len(oov):
        folded = {}
        for k in kv.index_to_key:
            folded.setdefault(ascii_fold(k.lower()), k)
        res["oov_examples"] = [{"word": w, "freq": int(f), "polemo_freq": int(pf), "word_label": wl,
                                "reason": oov_reason(w, pf, folded)}
                               for w, f, pf, wl in oov[["word", "freq", "polemo_freq", "word_label"]].itertuples(index=False)]
    else:
        res["oov_examples"] = []
    return res


def sanity(kv, topn: int = 5) -> dict:
    """Najbliższe słowa dla słów kontrolnych i podobieństwa par (tylko słowa ze słownika)."""
    neighbours = {w: [(n, round(float(s), 3)) for n, s in kv.most_similar(w, topn=topn)]
                  for w in PROBE_WORDS if w in kv.key_to_index}
    pairs = {f"{a}–{b}": round(float(kv.similarity(a, b)), 3)
             for a, b in PROBE_PAIRS if a in kv.key_to_index and b in kv.key_to_index}
    return {"neighbours": neighbours, "pairs": pairs}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",") if x == x else "–"


def coverage_markdown(cov: dict[str, dict], labels: dict[str, str]) -> str:
    lines = ["# Pokrycie słownika zaanotowanego korpusu przez modele", "",
             "| Model | Typy (słowa) | Tokeny | Słowa zaanotowane¹ |", "|---|---|---|---|"]
    for name, c in cov.items():
        lines.append(f"| {labels[name]} | {_pct(c['types']['share'])} ({c['types']['in_vocab']}/{c['types']['n']}) | "
                     f"{_pct(c['tokens']['share'])} | {_pct(c['annotated_types']['share'])} "
                     f"({c['annotated_types']['in_vocab']}/{c['annotated_types']['n']}) |")
    cols = [config.NO_LABEL] + config.TOKEN_LABELS
    lines += ["", "**Pokrycie według etykiety słowa²:**", "",
              "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, c in cov.items():
        cells = [f"{_pct(c['by_word_label'][l]['share'])} (n={c['by_word_label'][l]['n']})" for l in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name, c in cov.items():
        if not c["oov_examples"]:
            continue
        lines += ["", f"**{name}: najczęstsze słowa bez wektora:**", "",
                  "| Słowo | Wystąpienia (anotowane / PolEmo) | Etykieta słowa | Prawdopodobna przyczyna |",
                  "|---|---|---|---|"]
        lines += [f"| {e['word']} | {e['freq']} / {e['polemo_freq']} | {e['word_label']} | {e['reason']} |"
                  for e in c["oov_examples"]]
    lines += ["", "¹ Słowa, którym co najmniej jedna anotacja nadała etykietę inną niż O.",
              "² Etykieta słowa = najczęstsza etykieta ≠ O, jeśli co najmniej połowa anotacji słowa jest ≠ O; "
              "inaczej O. Dla fastText „pokrycie” oznacza obecność w słowniku – wektor z n-gramów "
              "znakowych dostaje każde słowo."]
    return "\n".join(lines) + "\n"


def sanity_markdown(results: dict[str, dict], labels: dict[str, str]) -> str:
    lines = ["# Kontrola sensowności wektorów (nie zastępuje pkt 3)", ""]
    for name, r in results.items():
        lines += [f"## {labels[name]}", "", "| Słowo | 5 najbliższych (kosinus) |", "|---|---|"]
        lines += [f"| {w} | " + ", ".join(f"{n} ({s:.2f})" for n, s in nb) + " |" for w, nb in r["neighbours"].items()]
        lines += ["", "| Para | kosinus |", "|---|---|"]
        lines += [f"| {p} | {s:.3f} |" for p, s in r["pairs"].items()]
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Całość
# --------------------------------------------------------------------------
def run(models: list[str], iterations: Iterable[int] = (1, 2), retrain: bool = False) -> pd.DataFrame:
    out = config.WORD_EMB_DIR
    out.mkdir(parents=True, exist_ok=True)
    vocab = annotated_vocabulary(iterations)
    cov, checks, sizes = {}, {}, {}
    for name in models:
        spec = MODELS[name]
        kv = train_polemo_w2v(force=True) if (name == "w2v_polemo" and retrain) else spec.loader()
        vocab, arrays = embed_vocabulary(vocab, kv, name, spec.subword)
        np.savez(out / f"vectors_{name}.npz", **arrays)
        cov[name] = coverage(vocab, name, kv)
        sizes[name] = len(kv.key_to_index)
        checks[name] = sanity(kv)
        print(f"[{name}] słowa z wektorem: {len(arrays['words'])}/{len(vocab)} "
              f"(słownik: {_pct(cov[name]['types']['share'])}, tokeny: {_pct(cov[name]['tokens']['share'])})")
    labels = {n: MODELS[n].label for n in models}
    vocab.to_csv(out / "vocab.csv", index=False, encoding="utf-8-sig")
    meta = {"iterations": list(iterations), "models": labels, "vocab_size": sizes, "coverage": cov, "sanity": checks}
    (out / "coverage.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "coverage.md").write_text(coverage_markdown(cov, labels), encoding="utf-8")
    (out / "sanity.md").write_text(sanity_markdown(checks, labels), encoding="utf-8")
    return vocab


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["w2v", "w2v_polemo"], choices=list(MODELS))
    ap.add_argument("--iterations", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--retrain", action="store_true", help="wytrenuj ponownie word2vec na PolEmo")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # konsola Windows (cp1250)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    vocab = run(a.models, a.iterations, a.retrain)
    print(f"Słownik: {len(vocab)} słów, {int(vocab['freq'].sum())} wystąpień → {config.WORD_EMB_DIR}")


if __name__ == "__main__":
    main()
