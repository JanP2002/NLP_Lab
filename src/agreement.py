"""Punkt 2: zgodność między anotatorami – kappa Cohena i kappa Fleissa.

Uruchomienie (z katalogu Lab1):  python -m src.agreement --iteration 1
Wejście:  annotation/iterN/export/sentences_wide.csv, tokens_wide.csv  (python -m src.export)
Wyjście:  results/iterN/agreement.json, agreement_table.md,
          confusion_sentences.png, confusion_tokens.png

Poziomy:
  sentences            etykieta zdania (POZ/NEG/NEU/AMB)
  tokens               etykieta każdego tokenu, puste pole = O
  tokens_marked        tylko tokeny, którym co najmniej jeden anotator nadał etykietę ≠ O
  sentences_with_gold  uzupełniająco: anotatorzy + etykieta PolEmo jako dodatkowy oceniający

Przedziały ufności: bootstrap po zdaniach – przy tokenach losujemy całe zdania, bo tokeny
jednego zdania nie są niezależne. Ten sam seed daje te same losowania na każdym poziomie.
Przy 2 oceniających kappa Fleissa = π Scotta (wspólny rozkład etykiet obu osób).
"""
import argparse
import itertools
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa as _sm_fleiss

import config

TOKEN_CATEGORIES = [config.NO_LABEL] + config.TOKEN_LABELS


# --- miary --------------------------------------------------------------------

def cohen(a, b) -> float:
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(cohen_kappa_score(a, b))


def fleiss(ratings: np.ndarray, categories: list[str]) -> float:
    """ratings: (n_elementów, n_oceniających) etykiety tekstowe."""
    codes = pd.Categorical(ratings.ravel(), categories=categories).codes.reshape(ratings.shape)
    if (codes < 0).any():
        raise ValueError(f"Etykiety spoza listy {categories}")
    table, _ = aggregate_raters(codes, n_cat=len(categories))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(_sm_fleiss(table, method="fleiss"))


def raw_agreement(ratings: np.ndarray) -> float:
    """Odsetek elementów, którym wszyscy oceniający nadali tę samą etykietę."""
    return float((ratings == ratings[:, :1]).all(axis=1).mean())


def interpret(k: float) -> str:
    """Skala Landisa i Kocha (1977)."""
    if k is None or np.isnan(k):
        return "–"
    if k < 0:
        return "poniżej losowej"
    for upper, name in [(0.20, "nieznaczna"), (0.40, "dostateczna"), (0.60, "umiarkowana"), (0.80, "znaczna")]:
        if k <= upper:
            return name
    return "prawie pełna"


# --- poziomy i bootstrap --------------------------------------------------------

@dataclass
class Level:
    name: str
    raters: list[str]
    ratings: np.ndarray   # (n, liczba oceniających)
    groups: np.ndarray    # indeks zdania (0..n_sentences-1) dla każdego wiersza
    n_sentences: int
    categories: list[str]


def measures(lv: Level, ratings: np.ndarray) -> dict[str, float]:
    out = {"raw_agreement": raw_agreement(ratings)}
    pairs = list(itertools.combinations(range(len(lv.raters)), 2))
    for i, j in pairs:
        out[f"cohen_{lv.raters[i]}-{lv.raters[j]}"] = cohen(ratings[:, i], ratings[:, j])
    if len(pairs) > 1:
        out["cohen_mean"] = float(np.mean([out[f"cohen_{lv.raters[i]}-{lv.raters[j]}"] for i, j in pairs]))
    out["fleiss"] = fleiss(ratings, lv.categories)
    return out


def bootstrap(lv: Level, n_boot: int, seed: int) -> dict[str, list[float]]:
    """95% przedziały percentylowe; losujemy zdania ze zwracaniem."""
    rng = np.random.default_rng(seed)
    rows_of = [np.flatnonzero(lv.groups == g) for g in range(lv.n_sentences)]
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, lv.n_sentences, lv.n_sentences)
        idx = np.concatenate([rows_of[g] for g in pick])
        draws.append(measures(lv, lv.ratings[idx]))
    df = pd.DataFrame(draws)
    return {m: [float(np.nanpercentile(df[m], 2.5)), float(np.nanpercentile(df[m], 97.5))] for m in df}


def build_levels(sent: pd.DataFrame, tok: pd.DataFrame, annotators: list[str]) -> list[Level]:
    sid_index = {sid: i for i, sid in enumerate(sent["sent_id"])}
    n = len(sent)
    tok_ratings = tok[annotators].to_numpy(dtype=str)
    tok_groups = tok["sent_id"].map(sid_index).to_numpy()
    marked = (tok_ratings != config.NO_LABEL).any(axis=1)
    levels = [
        Level("sentences", annotators, sent[annotators].to_numpy(dtype=str), np.arange(n), n, config.SENT_LABELS),
        Level("tokens", annotators, tok_ratings, tok_groups, n, TOKEN_CATEGORIES),
        Level("tokens_marked", annotators, tok_ratings[marked], tok_groups[marked], n, TOKEN_CATEGORIES),
    ]
    if "gold" in sent.columns and sent["gold"].notna().all():
        raters = annotators + ["gold"]
        levels.append(Level("sentences_with_gold", raters, sent[raters].to_numpy(dtype=str),
                            np.arange(n), n, config.SENT_LABELS))
    return levels


def per_class(ratings: np.ndarray, raters: list[str], categories: list[str]) -> dict:
    """Kappa Cohena dla każdej klasy osobno (klasa kontra reszta) i liczność klasy u każdego oceniającego."""
    pairs = list(itertools.combinations(range(len(raters)), 2))
    out = {}
    for c in categories:
        b = ratings == c
        out[c] = {"cohen": float(np.mean([cohen(b[:, i], b[:, j]) for i, j in pairs])),
                  "count": {r: int(b[:, k].sum()) for k, r in enumerate(raters)}}
    return out


# --- wyniki ---------------------------------------------------------------------

def compute(iteration: int, n_boot: int, seed: int) -> dict:
    d = config.export_dir(iteration)
    if not (d / "sentences_wide.csv").exists():
        raise FileNotFoundError(f"Brak eksportu w {d}. Uruchom najpierw: python -m src.export --iteration {iteration}")
    sent = pd.read_csv(d / "sentences_wide.csv", keep_default_na=False, na_values={"gold": [""]})
    tok = pd.read_csv(d / "tokens_wide.csv", keep_default_na=False)
    annotators = [c for c in sent.columns if c.startswith("A") and c[1:].isdigit()]
    if len(annotators) < 2:
        raise ValueError(f"Do zgodności potrzeba co najmniej 2 anotatorów, jest: {annotators}")

    result = {"iteration": iteration, "annotators": annotators,
              "bootstrap": {"n": n_boot, "seed": seed, "unit": "sentence"}, "levels": {}}
    for lv in build_levels(sent, tok, annotators):
        point, ci = measures(lv, lv.ratings), bootstrap(lv, n_boot, seed)
        result["levels"][lv.name] = {
            "n": int(len(lv.ratings)), "raters": lv.raters,
            "measures": {m: {"value": v, "ci95": ci[m], **({} if m == "raw_agreement" else {"interpretation": interpret(v)})}
                         for m, v in point.items()},
        }
    result["per_class"] = {
        "sentences": per_class(sent[annotators].to_numpy(dtype=str), annotators, config.SENT_LABELS),
        "tokens": per_class(tok[annotators].to_numpy(dtype=str), annotators, config.TOKEN_LABELS),
    }
    result["confusion"] = {
        "sentences": confusion(sent, annotators[:2], config.SENT_LABELS),
        "tokens": confusion(tok, annotators[:2], TOKEN_CATEGORIES),
    }
    return result


def confusion(df: pd.DataFrame, pair: list[str], categories: list[str]) -> list[list[int]]:
    a, b = (pd.Categorical(df[r], categories=categories) for r in pair)
    return pd.crosstab(a, b, dropna=False).reindex(index=categories, columns=categories, fill_value=0).values.tolist()


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _cell(m: dict) -> str:
    lo, hi = m["ci95"]
    return f"{_num(m['value'])} [{_num(lo)}; {_num(hi)}]"


def _pct(m: dict) -> str:
    return f"{_num(100 * m['value'], 1)}%"


def markdown_table(res: dict) -> str:
    ann = res["annotators"]
    pair = f"{ann[0]}–{ann[1]}" if len(ann) == 2 else "średnia par"
    cohen_key = f"cohen_{ann[0]}-{ann[1]}" if len(ann) == 2 else "cohen_mean"
    names = {"sentences": "Zdania", "tokens": "Tokeny – wszystkie", "tokens_marked": "Tokeny – zaznaczone¹"}
    lines = [
        f"## Zgodność – iteracja {res['iteration']} (anotatorzy: {', '.join(ann)})",
        "",
        f"| Poziom | n | Zgodność surowa | κ Cohena ({pair}) [95% CI] | κ Fleissa [95% CI] | Interpretacja² |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in names.items():
        lv = res["levels"][key]
        m = lv["measures"]
        lines.append(f"| {label} | {lv['n']} | {_pct(m['raw_agreement'])} | {_cell(m[cohen_key])} | "
                     f"{_cell(m['fleiss'])} | {m['fleiss']['interpretation']} |")
    if "sentences_with_gold" in res["levels"]:
        m = res["levels"]["sentences_with_gold"]["measures"]
        lines += ["", "**Uzupełniająco – etykieta PolEmo (gold) jako dodatkowy oceniający (zdania):**", "",
                  "| Miara | Wartość [95% CI] | Interpretacja² |", "|---|---|---|",
                  f"| κ Fleissa ({', '.join(ann)}, gold) | {_cell(m['fleiss'])} | {m['fleiss']['interpretation']} |"]
        for a in ann:
            mm = m[f"cohen_{a}-gold"]
            lines.append(f"| κ Cohena ({a}–gold) | {_cell(mm)} | {mm['interpretation']} |")
    for key, title in [("sentences", "zdania"), ("tokens", "tokeny")]:
        pc = res["per_class"][key]
        lines += ["", f"**κ Cohena dla każdej klasy osobno (klasa kontra reszta) – {title}:**", "",
                  "| Klasa | κ | " + " | ".join(f"liczba {a}" for a in ann) + " |",
                  "|---|---|" + "---|" * len(ann)]
        for c, v in pc.items():
            lines.append(f"| {c} | {_num(v['cohen'])} | " + " | ".join(str(v["count"][a]) for a in ann) + " |")
    b = res["bootstrap"]
    lines += [
        "",
        "¹ Tokeny, którym co najmniej jeden anotator nadał etykietę inną niż O. Na wszystkich tokenach "
        "dominuje zgodne O, które zawyża wynik.",
        "² Skala Landisa i Kocha (1977): ≤0,20 nieznaczna, 0,21–0,40 dostateczna, 0,41–0,60 umiarkowana, "
        "0,61–0,80 znaczna, 0,81–1 prawie pełna.",
        f"Przedziały ufności: bootstrap percentylowy, {b['n']} losowań zdań ze zwracaniem (seed {b['seed']}); "
        "przy tokenach losujemy całe zdania.",
    ]
    if len(ann) == 2:
        lines.append("Przy dwóch anotatorach kappa Fleissa jest równa π Scotta: od kappy Cohena różni się tylko "
                     "tym, że oczekiwaną zgodność liczy ze wspólnego rozkładu etykiet obu osób, stąd prawie "
                     "identyczne wartości.")
    return "\n".join(lines) + "\n"


def plot_confusion(matrix: list[list[int]], categories: list[str], pair: list[str], title: str, path):
    import matplotlib
    matplotlib.use("Agg")  # zapis do pliku, bez okna
    import matplotlib.pyplot as plt

    m = np.array(matrix)
    share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)  # kolor = udział w wierszu (O dominuje)
    fig, ax = plt.subplots(figsize=(1.1 * len(categories) + 2, 1.0 * len(categories) + 1.5))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(categories)), categories, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)), categories)
    ax.set_xlabel(f"{pair[1]}   (kolor: udział w wierszu)")
    ax.set_ylabel(pair[0])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_results(iteration: int, n_boot: int, seed: int) -> dict:
    res = compute(iteration, n_boot, seed)
    out = config.results_dir(iteration)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "agreement_table.md").write_text(markdown_table(res), encoding="utf-8")
    pair = res["annotators"][:2]
    for key, cats in [("sentences", config.SENT_LABELS), ("tokens", TOKEN_CATEGORIES)]:
        k = res["levels"][key]["measures"][f"cohen_{pair[0]}-{pair[1]}"]["value"]
        title = f"Iteracja {iteration}: {'zdania' if key == 'sentences' else 'tokeny'} (κ Cohena = {_num(k)})"
        plot_confusion(res["confusion"][key], cats, pair, title, out / f"confusion_{key}.png")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP["n"])
    p.add_argument("--seed", type=int, default=config.BOOTSTRAP["seed"])
    args = p.parse_args()
    res = write_results(args.iteration, args.bootstrap, args.seed)
    print(markdown_table(res))
    print(f"Zapisano wyniki w {config.results_dir(args.iteration)}")


if __name__ == "__main__":
    main()
