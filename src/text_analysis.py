"""Zadanie 1, Część II – Text Embeddings, pkt 3: analiza osadzeń zdań do pytań pomocniczych 1–5.

Uruchomienie (z katalogu Lab1, po: python -m src.text_embeddings --models w2v w2v_polemo):

    python -m src.text_analysis                                   # uśrednione Word2Vec (IPI PAN, PolEmo)
    python -m src.text_analysis --models w2v w2v_polemo fasttext --name trzy

Te same kroki co sekcja 3 notebooka zespołu (similar_but_different, cross_model_disagreement, probe_table,
length_stats, evaluate_embeddings, confusion_report) oraz dwie miary dodatkowe:
  * pyt. 1 – udział par o przeciwnym wydźwięku (POZ–NEG) wśród 1% najbardziej podobnych par zdań,
  * pyt. 2 – zależność podobieństwa od wspólnych słów (Spearman: kosinus vs. Jaccard zbiorów słów).
Dla porównania tabele zawierają wartości fastText / USE / TF-IDF odczytane z report/generated/emb_*.tex.

Wyjście (results/text_embeddings/): analiza_<name>.md, analiza_<name>.json, probes_<name>.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from src import text_embeddings as te

GEN = config.ROOT / "report" / "generated"
TITLES = {"w2v": "Word2Vec IPI PAN", "w2v_polemo": "Word2Vec PolEmo", "fasttext": "fastText", "use": "USE",
          "tfidf": "TF-IDF"}
TOP_SHARE = 0.01  # górny 1% par wg podobieństwa


# --------------------------------------------------------------------------
# Wyniki zespołu (fastText, USE, TF-IDF) z tabel sprawozdania
# --------------------------------------------------------------------------
def _tex_number(cell: str) -> float | str:
    s = cell.strip().replace("{,}", ".").replace("$-$", "-")
    try:
        return float(s)
    except ValueError:
        return cell.strip()


def read_tex_rows(path) -> list[list]:
    """Wiersze danych tabeli booktabs (między \\midrule i \\bottomrule; \\midrule wewnątrz pomijane)."""
    if not path.exists():
        return []
    body = path.read_text(encoding="utf-8").split("\\midrule", 1)[-1].split("\\bottomrule")[0]
    rows = []
    for line in body.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if line:
            rows.append([_tex_number(c) for c in line.split("&")])
    return rows


def team_probe_means() -> pd.DataFrame:
    rows = read_tex_rows(GEN / "emb_probe_means.tex")
    return pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=["fastText", "USE", "TF-IDF"])


def team_length() -> pd.DataFrame:
    out, model = [], None
    for r in read_tex_rows(GEN / "emb_length.tex"):
        model = r[0] or model
        if "Spearman" in str(r[1]):
            out.append({"model": model, "grupa": "Spearman", "spearman": r[5]})
        else:
            out.append({"model": model, "grupa": r[1], "n": r[2], "norma": r[3], "cos_centroid": r[4], "cos_inne": r[5]})
    return pd.DataFrame(out)


def team_classif(labels: str = "final") -> pd.DataFrame:
    out, model = [], None
    for r in read_tex_rows(GEN / "emb_classif.tex"):
        model = r[0] or model
        if r[1] == labels:
            out.append({"model": model, "accuracy": r[3], "baza": r[4], "macro_F1": r[5], "kNN10": r[6],
                        "losowy": r[7], "silhouette": r[8]})
    return pd.DataFrame(out)


def team_recall() -> pd.DataFrame:
    rows = read_tex_rows(GEN / "emb_recall.tex")
    return pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=config.SENT_LABELS)


# --------------------------------------------------------------------------
# Miary dodatkowe
# --------------------------------------------------------------------------
def jaccard_matrix(texts: list[str]) -> np.ndarray:
    """Jaccard zbiorów słów (małe litery, bez interpunkcji) dla każdej pary zdań."""
    sets = [set(te.tokens(t)) for t in texts]
    n = len(sets)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            J[i, j] = J[j, i] = len(sets[i] & sets[j]) / u if u else 0.0
    return J


def lexical_dependence(S: np.ndarray, J: np.ndarray) -> dict:
    """Pyt. 2: czy podobieństwo zależy głównie od wspólnych słów?"""
    i, j = np.triu_indices(len(S), 1)
    s, jac = S[i, j], J[i, j]
    rho = spearmanr(s, jac).statistic
    bins = {"brak wspólnych słów": jac == 0, "Jaccard (0; 0,1]": (jac > 0) & (jac <= 0.1), "Jaccard > 0,1": jac > 0.1}
    return {"spearman_cos_jaccard": float(rho),
            "mean_cos": {k: float(s[m].mean()) if m.any() else float("nan") for k, m in bins.items()},
            "n_pairs": {k: int(m.sum()) for k, m in bins.items()}}


def opposite_share_top(S: np.ndarray, labels: np.ndarray, share: float = TOP_SHARE) -> dict:
    """Pyt. 1: jaki odsetek najbardziej podobnych par to pary POZ–NEG (wobec odsetka wśród wszystkich par)."""
    i, j = np.triu_indices(len(S), 1)
    s = S[i, j]
    opp = ((labels[i] == "POZ") & (labels[j] == "NEG")) | ((labels[i] == "NEG") & (labels[j] == "POZ"))
    top = np.argsort(-s)[: max(1, int(round(share * len(s))))]
    return {"n_top": int(len(top)), "top_opposite": float(opp[top].mean()), "all_opposite": float(opp.mean()),
            "top_same": float((labels[i][top] == labels[j][top]).mean()),
            "all_same": float((labels[i] == labels[j]).mean())}


def top_pairs(df: pd.DataFrame, S: np.ndarray, k: int, opposite_only: bool) -> pd.DataFrame:
    i, j = np.triu_indices(len(S), 1)
    s = S[i, j]
    lab = df["final"].to_numpy()
    if opposite_only:
        m = ((lab[i] == "POZ") & (lab[j] == "NEG")) | ((lab[i] == "NEG") & (lab[j] == "POZ"))
        i, j, s = i[m], j[m], s[m]
    top = np.argsort(-s)[:k]
    return pd.DataFrame({"cos": np.round(s[top], 3), "final_a": lab[i[top]], "final_b": lab[j[top]],
                         "tekst_a": df["text"].to_numpy()[i[top]], "tekst_b": df["text"].to_numpy()[j[top]]})


# --------------------------------------------------------------------------
# Całość
# --------------------------------------------------------------------------
def analyse(models: list[str], iterations=(1, 2)) -> dict:
    df = te.load_sentences(iterations)
    embedders = {m: te.get_embedder(m) for m in models}
    E = {m: te.get_embeddings(embedders[m], df) for m in models}
    S = {m: te.cosine_matrix(E[m]) for m in models}
    J = jaccard_matrix(df["text"].tolist())
    probes = te.probe_table(embedders)
    res = {"n_sentences": len(df), "models": {m: embedders[m].label for m in models},
           "probes": probes, "probe_means": probes.groupby("kategoria", sort=False)[models].mean()}
    labels = df["final"].to_numpy()
    res["q1_opposite_top"] = {m: opposite_share_top(S[m], labels) for m in models}
    res["q1_pairs"] = {m: top_pairs(df, S[m], 5, opposite_only=True) for m in models}
    res["q2_lexical"] = {m: lexical_dependence(S[m], J) for m in models}
    if len(models) >= 2:
        a, b = models[:2]
        res["q2_disagreement"] = te.cross_model_disagreement(df, S[a], S[b], a, b, k=5)
    res["q4_length"] = {m: te.length_stats(df, E[m]) for m in models}
    res["q5_eval"] = {m: te.evaluation_table(df, E[m]) for m in models}
    res["q5_recall"] = {m: te.confusion_report(E[m], df["final"])[1].loc[config.SENT_LABELS, "recall"]
                        for m in models}
    return res


def _num(x, d: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}".replace(".", ",")


def _short(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def markdown(res: dict, models: list[str]) -> str:
    T = lambda m: TITLES.get(m, m)  # noqa: E731
    team_p = team_probe_means()
    L = [f"# Text Embeddings, pkt 3: analiza osadzeń ({', '.join(T(m) for m in models)})", "",
         f"{res['n_sentences']} zaanotowanych zdań (iteracje 1–2), etykieta `final` = uzgodniona. "
         "Kolumny fastText / USE / TF-IDF – wyniki zespołu z `report/generated/emb_*.tex` (ten sam zestaw par, "
         "te same zdania i miary).", "",
         "## Pary kontrolne (`PROBE_PAIRS`): średni kosinus według kategorii (pyt. 1–3)", "",
         "| Kategoria | " + " | ".join(T(m) for m in models) + " | " + " | ".join(team_p.columns) + " |",
         "|---|" + "---|" * (len(models) + len(team_p.columns))]
    for cat, row in res["probe_means"].iterrows():
        team = [_num(team_p.at[cat, c]) if cat in team_p.index else "–" for c in team_p.columns]
        L.append(f"| {cat} | " + " | ".join(_num(row[m]) for m in models) + " | " + " | ".join(team) + " |")
    L += ["", "Pary pojedynczo:", "", "| Kategoria | Zdanie a | Zdanie b | " + " | ".join(T(m) for m in models) + " |",
          "|---|---|---|" + "---|" * len(models)]
    for r in res["probes"].itertuples(index=False):
        r = r._asdict()
        L.append(f"| {r['kategoria']} | {r['zdanie_a']} | {r['zdanie_b']} | " + " | ".join(_num(r[m], 2) for m in models) + " |")

    L += ["", "## Pyt. 1: zdania uznane za podobne, a różne znaczeniowo", "",
          f"Wśród {_num(100 * TOP_SHARE, 0)}% najbardziej podobnych par zdań (etykiety `final`):", "",
          "| Model | par | POZ–NEG wśród najbliższych | POZ–NEG wśród wszystkich | ta sama etykieta wśród najbliższych | ta sama etykieta wśród wszystkich |",
          "|---|---|---|---|---|---|"]
    for m in models:
        o = res["q1_opposite_top"][m]
        L.append(f"| {T(m)} | {o['n_top']} | {_num(100 * o['top_opposite'], 1)}% | {_num(100 * o['all_opposite'], 1)}% | "
                 f"{_num(100 * o['top_same'], 1)}% | {_num(100 * o['all_same'], 1)}% |")
    for m in models:
        L += ["", f"**{T(m)}: najbardziej podobne pary POZ–NEG:**", "", "| cos | final | Zdanie a | Zdanie b |", "|---|---|---|---|"]
        L += [f"| {_num(r.cos)} | {r.final_a}/{r.final_b} | {_short(r.tekst_a)} | {_short(r.tekst_b)} |"
              for r in res["q1_pairs"][m].itertuples()]

    L += ["", "## Pyt. 2: podobne znaczeniowo, a odległe, bo inne słowa", "",
          "Zależność kosinusa od wspólnych słów (Jaccard zbiorów słów, wszystkie pary zdań):", "",
          "| Model | Spearman (cos vs Jaccard) | śr. cos: brak wspólnych słów | Jaccard (0; 0,1] | Jaccard > 0,1 |",
          "|---|---|---|---|---|"]
    for m in models:
        q = res["q2_lexical"][m]
        mc = list(q["mean_cos"].values())
        L.append(f"| {T(m)} | {_num(q['spearman_cos_jaccard'])} | {_num(mc[0])} | {_num(mc[1])} | {_num(mc[2])} |")
    n_pairs = list(res["q2_lexical"][models[0]]["n_pairs"].values())
    L += ["", f"Liczba par w grupach: {n_pairs[0]} / {n_pairs[1]} / {n_pairs[2]}."]
    if "q2_disagreement" in res:
        a, b = models[:2]
        for (x, y), tab in zip([(a, b), (b, a)], res["q2_disagreement"]):
            L += ["", f"**Blisko według {T(x)}, daleko według {T(y)} (rangi percentylowe):**", "",
                  f"| ranga {T(x)} | ranga {T(y)} | gold | Zdanie a | Zdanie b |", "|---|---|---|---|---|"]
            L += [f"| {_num(r[f'ranga_{x}'], 2)} | {_num(r[f'ranga_{y}'], 2)} | {r['gold_a']}/{r['gold_b']} | "
                  f"{_short(r['tekst_a'])} | {_short(r['tekst_b'])} |" for r in tab.to_dict("records")]

    team_l = team_length()
    L += ["", "## Pyt. 3–4: długość zdania", "",
          "| Model | Grupa (tokeny) | n | norma | cos do centroidu | śr. cos do innych |", "|---|---|---|---|---|---|"]
    for m in models:
        ls = res["q4_length"][m]
        for grp, r in ls.iterrows():
            if grp.startswith("Spearman"):
                L.append(f"| {T(m)} | Spearman (tokeny vs śr. cos) | | | | {_num(r['srednie_cos_do_innych'])} |")
            else:
                L.append(f"| {T(m)} | {grp} | {int(r['n'])} | {_num(r['norma_wektora'], 2)} | "
                         f"{_num(r['cos_do_centroidu'])} | {_num(r['srednie_cos_do_innych'])} |")
    for r in team_l.to_dict("records"):
        if r["grupa"] == "Spearman":
            L.append(f"| {r['model']} (zespół) | Spearman (tokeny vs śr. cos) | | | | {_num(r['spearman'])} |")
        else:
            L.append(f"| {r['model']} (zespół) | {r['grupa']} | {int(r['n'])} | {_num(r['norma'], 2)} | "
                     f"{_num(r['cos_centroid'])} | {_num(r['cos_inne'])} |")

    team_c, team_r = team_classif(), team_recall()
    L += ["", "## Pyt. 5: czy osadzenia nadają się do zadania (wydźwięk zdania)", "",
          "Regresja logistyczna na samych osadzeniach (5-krotna walidacja krzyżowa), etykiety `final`:", "",
          "| Model | accuracy | baza większościowa | macro-F1 | kNN-10 | losowy | silhouette |", "|---|---|---|---|---|---|---|"]
    for m in models:
        r = res["q5_eval"][m].set_index("etykiety").loc["final"]
        L.append(f"| {T(m)} | {_num(r['accuracy'])} | {_num(r['baza_wiekszosciowa'])} | {_num(r['macro_F1'])} | "
                 f"{_num(r['kNN10_czystosc'])} | {_num(r['losowy_wg_rozkladu'])} | {_num(r['silhouette_cos'])} |")
    for r in team_c.to_dict("records"):
        L.append(f"| {r['model']} (zespół) | {_num(r['accuracy'])} | {_num(r['baza'])} | {_num(r['macro_F1'])} | "
                 f"{_num(r['kNN10'])} | {_num(r['losowy'])} | {_num(r['silhouette'])} |")
    L += ["", "Czułość (recall) dla każdej klasy (`final`):", "", "| Model | " + " | ".join(config.SENT_LABELS) + " |",
          "|---|" + "---|" * len(config.SENT_LABELS)]
    for m in models:
        L.append(f"| {T(m)} | " + " | ".join(_num(res["q5_recall"][m][c], 2) for c in config.SENT_LABELS) + " |")
    for model, r in team_r.iterrows():
        L.append(f"| {model} (zespół) | " + " | ".join(_num(r[c], 2) for c in config.SENT_LABELS) + " |")
    return "\n".join(L) + "\n"


def to_json(res: dict, models: list[str]) -> dict:
    return {"models": res["models"], "n_sentences": res["n_sentences"],
            "probe_means": res["probe_means"].round(4).to_dict(),
            "q1_opposite_top": res["q1_opposite_top"], "q2_lexical": res["q2_lexical"],
            "q4_length": {m: res["q4_length"][m].rename_axis("grupa").reset_index().to_dict("records")
                          for m in models},
            "q5_eval": {m: res["q5_eval"][m].to_dict("records") for m in models},
            "q5_recall": {m: res["q5_recall"][m].to_dict() for m in models}}


def run(models: list[str], name: str) -> dict:
    res = analyse(models)
    out = te.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    res["probes"].to_csv(out / f"probes_{name}.csv", index=False, encoding="utf-8-sig")
    (out / f"analiza_{name}.md").write_text(markdown(res, models), encoding="utf-8")
    data = to_json(res, models)
    (out / f"analiza_{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["w2v", "w2v_polemo"], choices=te.MODEL_NAMES)
    ap.add_argument("--name", default="w2v", help="przyrostek plików wynikowych")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # konsola Windows (cp1250)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    run(a.models, a.name)
    print((te.OUT_DIR / f"analiza_{a.name}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
