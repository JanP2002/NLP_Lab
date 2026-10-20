"""Zadanie 1, Część II – Word Embeddings, pkt 3: k najbliższych słów i porównanie modeli.

Uruchomienie (z katalogu Lab1):

    python -m src.word_neighbours                                   # w2v IPI PAN vs w2v PolEmo
    python -m src.word_neighbours --models w2v w2v_polemo fasttext  # wszystkie pary trzech modeli

Wejście: results/word_embeddings/vocab.csv (pkt 1), modele z ``word_embeddings.MODELS``,
         korpus PolEmo (wydźwięk sąsiadów z etykiet gold).
Wyjście (results/word_embeddings/):
  neighbours_<model>.csv   k sąsiadów każdego słowa zapytania (kosinus, odmiana, wydźwięk)
  knn_comparison.csv       miary zgodności list dla każdego słowa i pary modeli
  knn_summary.md / .json   tabele zbiorcze, listy pokazowe, przykłady

Słowa zapytania: słowa zaanotowane (word_label ≠ O) znane wszystkim porównywanym modelom,
plus zestaw pokazowy do raportu (po kilka najczęstszych z każdej etykiety i słowa wybrane ręcznie).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from src import word_embeddings as we

ALPHA_RE = re.compile(r"^[^\W\d_]+(?:-[^\W\d_]+)*$")
SHOWCASE_EXTRA = ["polecam", "miła", "brudne", "problem", "bardzo", "trochę", "niestety", "nie"]
N_PER_LABEL = 4
EXPECTED_POLARITY = {"WYR_POZ": "+", "WYR_NEG": "-"}


# --------------------------------------------------------------------------
# Słowa zapytania
# --------------------------------------------------------------------------
def available(vocab: pd.DataFrame, model: str) -> pd.Series:
    """Czy model ma wektor słowa (fastText: zawsze – n-gramy znakowe)."""
    if we.MODELS[model].subword:
        return pd.Series(True, index=vocab.index)
    return vocab[f"{model}_in_vocab"]


def query_form(row, model: str) -> str:
    form = row.get(f"{model}_form", "")
    return form if isinstance(form, str) and form else row["word"]


def query_words(vocab: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Zestaw główny (word_label ≠ O we wszystkich modelach) i lista słów pokazowych.

    Zwraca tabelę wszystkich odpytywanych słów (główne + dodatki pokazowe spoza zestawu
    głównego, np. „nie”) z kolumną ``in_main_set``.
    """
    ok = np.logical_and.reduce([available(vocab, m) for m in models])
    main = vocab[ok & (vocab["word_label"] != config.NO_LABEL)]
    showcase = []
    for lab in config.TOKEN_LABELS:
        top = main[main["word_label"] == lab].sort_values(["freq", "word"], ascending=[False, True])
        showcase += top["word"].head(N_PER_LABEL).tolist()
    extras = vocab[ok & vocab["word"].isin(SHOWCASE_EXTRA)]
    showcase += [w for w in SHOWCASE_EXTRA if w in set(extras["word"]) and w not in showcase]
    queries = pd.concat([main, extras[~extras["word"].isin(main["word"])]]).drop_duplicates("word")
    queries = queries.assign(in_main_set=queries["word"].isin(main["word"]))
    return queries.reset_index(drop=True), showcase


# --------------------------------------------------------------------------
# Listy sąsiadów
# --------------------------------------------------------------------------
def neighbours(kv, form: str, k: int = 10) -> list[tuple[str, float]]:
    """k najbliższych słów (kosinus), tylko słowa z liter, bez samego słowa zapytania."""
    query = form.lower()
    topn = 3 * k
    while True:
        cands = kv.most_similar(form, topn=topn)
        out = [(w, float(s)) for w, s in cands if ALPHA_RE.match(w) and w.lower() != query][:k]
        if len(out) == k or len(cands) < topn:
            return out
        topn *= 3


def same_stem(a: str, b: str) -> bool:
    """Przybliżenie „to samo słowo w innej formie” (bez lematyzatora): wspólny początek
    wyrazu długości co najmniej max(3, ⌈0,6 · krótszego słowa⌉), oba słowa ≥ 4 litery."""
    a, b = a.lower(), b.lower()
    if min(len(a), len(b)) < 4:
        return False
    lcp = 0
    for x, y in zip(a, b):
        if x != y:
            break
        lcp += 1
    return lcp >= max(3, math.ceil(0.6 * min(len(a), len(b))))


def normalized_matrix(kv, words: list[str]) -> np.ndarray:
    M = np.vstack([kv[w] for w in words]).astype(np.float64)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def shared_vocabulary(kvs: list) -> list[str]:
    """Słowa z liter znane wszystkim modelom (słowniki, nie n-gramy)."""
    common = set(kvs[0].key_to_index)
    for kv in kvs[1:]:
        common &= set(kv.key_to_index)
    return sorted(w for w in common if ALPHA_RE.match(w))


def shared_neighbours(kv, vocab_words: list[str], matrix: np.ndarray, form: str, k: int = 10) -> list[str]:
    """k najbliższych słów, ale wyszukiwanych tylko w ``vocab_words`` (wspólny słownik)."""
    q = np.asarray(kv[form], dtype=np.float64)
    sims = matrix @ (q / np.linalg.norm(q))
    order = np.argsort(-sims)
    query = form.lower()
    return [vocab_words[i] for i in order if vocab_words[i].lower() != query][:k]


# --------------------------------------------------------------------------
# Miary porównania
# --------------------------------------------------------------------------
def overlap_metrics(a: list[str], b: list[str], k: int) -> dict:
    sa, sb = set(a[:k]), set(b[:k])
    inter = len(sa & sb)
    return {"overlap": inter / k, "jaccard": inter / len(sa | sb) if sa | sb else 1.0}


def second_order(kv_a, kv_b, forms_a: list[str], forms_b: list[str]) -> tuple[np.ndarray, float]:
    """Spearman ρ profili podobieństw słów zapytania (każde do pozostałych) w modelu A i B.

    Zwraca ρ dla każdego słowa i ρ dla całych macierzy (górne trójkąty).
    """
    Sa = normalized_matrix(kv_a, forms_a)
    Sb = normalized_matrix(kv_b, forms_b)
    Sa, Sb = Sa @ Sa.T, Sb @ Sb.T
    n = len(forms_a)
    per_word = np.array([spearmanr(np.delete(Sa[i], i), np.delete(Sb[i], i)).statistic for i in range(n)])
    iu = np.triu_indices(n, 1)
    return per_word, float(spearmanr(Sa[iu], Sb[iu]).statistic)


def polemo_polarity(texts=None, golds=None, min_count: int = 5) -> pd.DataFrame:
    """Wydźwięk słowa z etykiet gold PolEmo: s = (zdania POZ − zdania NEG) / wszystkie zdania ze słowem.

    Tylko słowa z co najmniej ``min_count`` zdaniami; pozostałe są „bez oceny”.
    """
    if texts is None:
        df = pd.read_parquet(config.PROCESSED, columns=["text", "gold"])
        texts, golds = df["text"], df["gold"]
    n, pos, neg = Counter(), Counter(), Counter()
    for text, gold in zip(texts, golds):
        for w in {t.lower() for t in text.split() if ALPHA_RE.match(t)}:
            n[w] += 1
            pos[w] += gold == "POZ"
            neg[w] += gold == "NEG"
    rows = [(w, c, pos[w], neg[w], (pos[w] - neg[w]) / c) for w, c in n.items() if c >= min_count]
    return pd.DataFrame(rows, columns=["word", "polemo_n", "n_POZ", "n_NEG", "polarity"]).set_index("word")


def polarity_class(score: float, tau: float) -> str:
    if score != score:  # NaN = bez oceny
        return ""
    return "+" if score >= tau else "-" if score <= -tau else "0"


# --------------------------------------------------------------------------
# Całość
# --------------------------------------------------------------------------
def build_neighbour_table(kv, model: str, queries: pd.DataFrame, k: int, polarity: pd.DataFrame,
                          tau: float) -> pd.DataFrame:
    rows = []
    for q in queries.to_dict("records"):
        form = query_form(q, model)
        for rank, (w, cos) in enumerate(neighbours(kv, form, k), start=1):
            rows.append({"query": q["word"], "query_label": q["word_label"], "query_freq": q["freq"],
                         "in_main_set": q["in_main_set"], "rank": rank, "neighbour": w, "cosine": round(cos, 4),
                         "same_stem": same_stem(q["word"], w)})
    tab = pd.DataFrame(rows)
    pol = polarity["polarity"].reindex(tab["neighbour"].str.lower())
    tab["polemo_n"] = polarity["polemo_n"].reindex(tab["neighbour"].str.lower()).fillna(0).astype(int).values
    tab["polarity"] = pol.round(3).values
    tab["polarity_class"] = [polarity_class(s, tau) for s in pol.values]
    return tab


def per_model_stats(tab: pd.DataFrame) -> pd.DataFrame:
    """Dla każdego słowa: udział odmian, wydźwięk sąsiadów (liczności, by móc sumować w grupach)."""
    g = tab.groupby("query")
    out = pd.DataFrame({"same_stem": g["same_stem"].mean(),
                        "n_scored": g["polarity_class"].apply(lambda s: int((s != "").sum()))})
    for cls, name in [("+", "n_pos"), ("-", "n_neg"), ("0", "n_neu")]:
        out[name] = g["polarity_class"].apply(lambda s, c=cls: int((s == c).sum()))
    return out


def polarity_agreement(stats: pd.DataFrame, labels: pd.Series) -> dict:
    """Sumarycznie dla zapytań WYR_POZ / WYR_NEG: udział sąsiadów o zgodnym, przeciwnym i neutralnym wydźwięku."""
    out = {}
    for lab, sign in EXPECTED_POLARITY.items():
        s = stats[labels.reindex(stats.index) == lab]
        scored = int(s["n_scored"].sum())
        same = int(s["n_pos" if sign == "+" else "n_neg"].sum())
        opp = int(s["n_neg" if sign == "+" else "n_pos"].sum())
        out[lab] = {"n_queries": int(len(s)), "scored": scored,
                    "same": same / scored if scored else float("nan"),
                    "opposite": opp / scored if scored else float("nan"),
                    "neutral": int(s["n_neu"].sum()) / scored if scored else float("nan")}
    return out


def run(models: list[str], k: int = 10, tau: float = 0.1, min_count: int = 5) -> dict:
    if len(models) < 2:
        raise ValueError("Porównanie wymaga co najmniej 2 modeli")
    out = config.WORD_EMB_DIR
    vocab = we.read_vocab(out)
    queries, showcase = query_words(vocab, models)
    main = queries[queries["in_main_set"]].reset_index(drop=True)
    kvs = {m: we.MODELS[m].loader() for m in models}
    polarity = polemo_polarity(min_count=min_count)

    tables, stats = {}, {}
    for m in models:
        tables[m] = build_neighbour_table(kvs[m], m, queries, k, polarity, tau)
        tables[m].to_csv(out / f"neighbours_{m}.csv", index=False, encoding="utf-8-sig")
        stats[m] = per_model_stats(tables[m][tables[m]["in_main_set"]])

    shared = shared_vocabulary([kvs[m] for m in models])
    shared_mats = {m: normalized_matrix(kvs[m], shared) for m in models}
    lists = {m: tables[m].groupby("query")["neighbour"].apply(list) for m in models}
    rows, global_rho = [], {}
    for a, b in itertools.combinations(models, 2):
        rho_word, rho_all = second_order(kvs[a], kvs[b], [query_form(r, a) for r in main.to_dict("records")],
                                         [query_form(r, b) for r in main.to_dict("records")])
        global_rho[f"{a} vs {b}"] = rho_all
        for i, q in enumerate(main.to_dict("records")):
            w = q["word"]
            sh_a = shared_neighbours(kvs[a], shared, shared_mats[a], query_form(q, a), k)
            sh_b = shared_neighbours(kvs[b], shared, shared_mats[b], query_form(q, b), k)
            raw, sh = overlap_metrics(lists[a][w], lists[b][w], k), overlap_metrics(sh_a, sh_b, k)
            rows.append({"query": w, "query_label": q["word_label"], "query_freq": q["freq"],
                         "model_a": a, "model_b": b, "overlap": raw["overlap"], "jaccard": raw["jaccard"],
                         "overlap_shared": sh["overlap"], "jaccard_shared": sh["jaccard"],
                         "rho2": float(rho_word[i]), "same_stem_a": stats[a].at[w, "same_stem"],
                         "same_stem_b": stats[b].at[w, "same_stem"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "knn_comparison.csv", index=False, encoding="utf-8-sig")

    labels = main.set_index("word")["word_label"]
    summary = {
        "models": {m: we.MODELS[m].label for m in models},
        "params": {"k": k, "polarity_threshold": tau, "polarity_min_count": min_count,
                   "polarity_words": int(len(polarity)), "shared_vocabulary": len(shared)},
        "n_queries": int(len(main)), "query_labels": main["word_label"].value_counts().to_dict(),
        "showcase": showcase, "global_rho2": global_rho,
        "pairs": pair_summary(comp),
        "per_model": {m: {"same_stem": by_label(stats[m]["same_stem"], labels),
                          "polarity": polarity_agreement(stats[m], labels)} for m in models},
    }
    (out / "knn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "knn_summary.md").write_text(summary_markdown(summary, tables, comp, showcase, models, k), encoding="utf-8")
    return summary


def by_label(values: pd.Series, labels: pd.Series) -> dict:
    df = pd.DataFrame({"v": values, "label": labels.reindex(values.index)})
    out = {lab: float(g["v"].mean()) for lab, g in df.groupby("label")}
    out["wszystkie"] = float(df["v"].mean())
    return out


def pair_summary(comp: pd.DataFrame) -> dict:
    metrics = ["overlap", "jaccard", "overlap_shared", "rho2"]
    out = {}
    for (a, b), g in comp.groupby(["model_a", "model_b"], sort=False):
        rows = {lab: {m: float(gg[m].mean()) for m in metrics} | {"n": int(len(gg))}
                for lab, gg in g.groupby("query_label")}
        rows["wszystkie"] = {m: float(g[m].mean()) for m in metrics} | {"n": int(len(g))}
        out[f"{a} vs {b}"] = rows
    return out


# --------------------------------------------------------------------------
# Raport
# --------------------------------------------------------------------------
def _num(x: float, d: int = 2) -> str:
    return "–" if x != x else f"{x:.{d}f}".replace(".", ",")


def _pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.0f}%"


def summary_markdown(s: dict, tables: dict, comp: pd.DataFrame, showcase: list[str], models: list[str], k: int) -> str:
    label_order = [*config.TOKEN_LABELS, "wszystkie"]
    L = [f"# Pkt 3: {k} najbliższych słów i porównanie modeli", "",
         "Modele: " + "; ".join(f"**{m}** = {lab}" for m, lab in s["models"].items()) + ".",
         f"Słowa zapytania: {s['n_queries']} słów z etykietą ≠ O znanych wszystkim modelom "
         f"({', '.join(f'{l} {n}' for l, n in s['query_labels'].items())}). "
         f"Sąsiedzi: tylko słowa z liter, z całego słownika modelu. Wspólny słownik modeli: {s['params']['shared_vocabulary']} słów.",
         "", "## 1. Zgodność list między modelami", ""]
    for pair, rows in s["pairs"].items():
        L += [f"**{pair}** (ρ drugiego rzędu dla całej macierzy podobieństw: {_num(s['global_rho2'][pair])})", "",
              f"| Etykieta | n | Wspólne@{k} | Jaccard@{k} | Wspólne@{k} na wspólnym słowniku | ρ drugiego rzędu (średnia) |",
              "|---|---|---|---|---|---|"]
        for lab in label_order:
            if lab in rows:
                r = rows[lab]
                L.append(f"| {lab} | {r['n']} | {_pct(r['overlap'])} | {_num(r['jaccard'])} | "
                         f"{_pct(r['overlap_shared'])} | {_num(r['rho2'])} |")
        L.append("")
    L += ["## 2. Charakter list w każdym modelu", "",
          "| Model | Odmiany słowa zapytania wśród sąsiadów | Sąsiedzi słów WYR_POZ: zgodny / przeciwny / neutralny wydźwięk (ocenionych) "
          "| Sąsiedzi słów WYR_NEG: zgodny / przeciwny / neutralny (ocenionych) |", "|---|---|---|---|"]
    for m in models:
        st, pol = s["per_model"][m]["same_stem"], s["per_model"][m]["polarity"]
        cells = [f"{_pct(p['same'])} / {_pct(p['opposite'])} / {_pct(p['neutral'])} ({p['scored']})"
                 for p in (pol["WYR_POZ"], pol["WYR_NEG"])]
        L.append(f"| {m} | {_pct(st['wszystkie'])} | {cells[0]} | {cells[1]} |")
    L += ["", f"Wydźwięk sąsiada: s = (zdania POZ − zdania NEG) / wszystkie zdania PolEmo ze słowem "
          f"(min. {s['params']['polarity_min_count']} zdań; próg ±{_num(s['params']['polarity_threshold'], 1)}). "
          "Model PolEmo i ta miara pochodzą z tego samego korpusu – jego przewaga jest częściowo „z założenia”.", ""]

    lists = {m: tables[m].groupby("query") for m in models}
    L += ["## 3. Listy pokazowe (5 najbliższych)", "",
          "**pogrubienie** = słowo jest na liście top-" f"{k} także w innym modelu; _kursywa_ = odmiana słowa zapytania; "
          "⊗ = wydźwięk przeciwny do etykiety słowa zapytania.", "",
          "| Słowo | Etykieta | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for w in showcase:
        cells = []
        for m in models:
            g = lists[m].get_group(w)
            others = set().union(*(set(lists[o].get_group(w)["neighbour"]) for o in models if o != m))
            exp = EXPECTED_POLARITY.get(g["query_label"].iat[0])
            parts = []
            for r in g.head(5).itertuples():
                t = r.neighbour
                t = f"_{t}_" if r.same_stem else t
                t = f"**{t}**" if r.neighbour in others else t
                if exp and r.polarity_class and r.polarity_class not in (exp, "0"):
                    t += " ⊗"
                parts.append(f"{t} ({_num(r.cosine)})")
            cells.append(", ".join(parts))
        L.append(f"| {w} | {lists[models[0]].get_group(w)['query_label'].iat[0]} | " + " | ".join(cells) + " |")

    for pair in s["pairs"]:
        a, b = pair.split(" vs ")
        c = comp[(comp["model_a"] == a) & (comp["model_b"] == b)].sort_values(["overlap_shared", "rho2"])
        fmt = lambda d: ", ".join(f"{r.query} ({_pct(r.overlap_shared)}, ρ {_num(r.rho2)})" for r in d.itertuples())  # noqa: E731
        L += ["", f"## 4. Największa i najmniejsza zgodność ({pair}, wspólne@{k} na wspólnym słowniku)", "",
              f"- najbardziej zgodne: {fmt(c.tail(5).iloc[::-1])}", f"- najmniej zgodne: {fmt(c.head(5))}"]

    L += ["", "## 5. Przykłady sąsiadów o przeciwnym wydźwięku", ""]
    for m in models:
        t = tables[m]
        exp = t["query_label"].map(EXPECTED_POLARITY)
        opp = t[t["in_main_set"] & exp.notna() & (t["polarity_class"] != "") & (t["polarity_class"] != "0")
                & (t["polarity_class"] != exp)].sort_values("cosine", ascending=False).head(10)
        ex = ", ".join(f"{r.query} → {r.neighbour} ({_num(r.cosine)}; s = {_num(r.polarity)})" for r in opp.itertuples())
        L.append(f"- **{m}**: {ex or 'brak'}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["w2v", "w2v_polemo"], choices=list(we.MODELS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--polarity-threshold", type=float, default=0.1)
    ap.add_argument("--min-count", type=int, default=5)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # konsola Windows (cp1250)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    s = run(a.models, a.k, a.polarity_threshold, a.min_count)
    print(f"Słowa zapytania: {s['n_queries']}, wspólny słownik: {s['params']['shared_vocabulary']}")
    print((config.WORD_EMB_DIR / "knn_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
