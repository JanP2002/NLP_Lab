"""Zadanie 1, Część II – Word Embeddings, pkt 2: t-SNE słów z etykietami anotacji.

Uruchomienie (z katalogu Lab1):

    python -m src.word_tsne                                # w2v IPI PAN + w2v PolEmo
    python -m src.word_tsne --models fasttext              # gdy pkt 1 zapisał vectors_fasttext.npz

Wejście: wyniki pkt 1 (results/word_embeddings/vectors_<model>.npz, vocab.csv) – bez modeli.
Wyjście (results/word_embeddings/), dla każdego modelu i podzbioru („all” = wszystkie słowa
z wektorem, „annotated” = słowa z jakąkolwiek anotacją ≠ O, t-SNE liczone osobno):
  tsne_words_<model>_<subset>.html   wykres interaktywny (podpowiedź: słowo + anotacja)
  tsne_words_<model>_<subset>.csv    współrzędne i etykiety (dla raportu)
  tsne_words_<model>_<subset>.png    obraz statyczny do LaTeX
  label_separation.md / .json        czy etykiety grupują się w oryginalnej przestrzeni

Kolory: trzy barwy zespołu (zielony = pozytywne, pomarańczowy = negatywne, fioletowy)
przechodzą test rozróżnialności dla daltonistów przy wszystkich parach punktów; czwartej
barwy nie dodajemy. Modyfikatory WZM / OSL / ODW mają wspólny fiolet i różne kształty,
O / NEU są szare (tło), „mieszany” to szary pusty znacznik.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src import text_embeddings as te
from src import word_embeddings as we

POS, NEG, THIRD = "#1b9e77", "#d95f02", "#7570b3"  # paleta zespołu (LABEL_COLORS)
GREY, LIGHT_GREY = "#9e9e9e", "#bdbdbd"
MODIFIERS = ["WZM", "OSL", "ODW"]


@dataclass(frozen=True)
class Style:
    color: str
    symbol: str          # Plotly
    marker: str          # matplotlib
    background: bool = False
    hollow: bool = False


# kolejność = kolejność w legendzie; tło (O) na końcu, żeby rysowało się pod spodem
WORD_STYLES = {
    "WYR_POZ": Style(POS, "circle", "o"),
    "WYR_NEG": Style(NEG, "circle", "o"),
    "WZM": Style(THIRD, "triangle-up", "^"),
    "OSL": Style(THIRD, "triangle-down", "v"),
    "ODW": Style(THIRD, "x", "X"),
    "O": Style(LIGHT_GREY, "circle", "o", background=True),
}
CONTEXT_STYLES = {
    "POZ": Style(POS, "circle", "o"),
    "NEG": Style(NEG, "circle", "o"),
    "AMB": Style(THIRD, "diamond", "D"),
    "NEU": Style(GREY, "circle", "o", background=True),
    "mieszany": Style(GREY, "circle-open", "o", background=True, hollow=True),
    "brak": Style(LIGHT_GREY, "circle-open", "o", background=True, hollow=True),
}
SCHEMES = {"word_label": ("etykieta słowa", WORD_STYLES), "context_label": ("kontekst zdań", CONTEXT_STYLES)}
SUBSETS = {"all": "wszystkie słowa z wektorem", "annotated": "słowa z anotacją ≠ O"}
MODEL_TITLES = {"w2v": "Word2Vec IPI PAN", "w2v_polemo": "Word2Vec PolEmo (gensim)", "fasttext": "fastText KGR10"}
SHARE_COLS = [f"share_{l}" for l in config.SENT_LABELS]


# --------------------------------------------------------------------------
# Dane
# --------------------------------------------------------------------------
def context_label(vocab: pd.DataFrame) -> pd.Series:
    """Dominująca etykieta zdań (``final``), w których słowo wystąpiło; remis → „mieszany”."""
    shares = vocab[SHARE_COLS].to_numpy(dtype=float)
    out = []
    for row in shares:
        if np.isnan(row).all():
            out.append("brak")
            continue
        top = np.flatnonzero(row == np.nanmax(row))
        out.append(config.SENT_LABELS[top[0]] if len(top) == 1 else "mieszany")
    return pd.Series(out, index=vocab.index)


def load_points(model: str, subset: str = "all", out_dir=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Słowa z wektorem (kolejność jak w ``.npz``) z kolumnami ``vocab.csv`` oraz macierz wektorów."""
    out_dir = out_dir or config.WORD_EMB_DIR
    npz = out_dir / f"vectors_{model}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Brak {npz}. Najpierw pkt 1: python -m src.word_embeddings --models {model}")
    arr = np.load(npz)
    vocab = we.read_vocab(out_dir).set_index("word")
    points = vocab.loc[arr["words"]].reset_index()
    points["context_label"] = context_label(points).values
    X = arr["vectors"]
    if subset == "annotated":
        keep = (points["annot_share"] > 0).to_numpy()
        points, X = points[keep].reset_index(drop=True), X[keep]
    elif subset != "all":
        raise ValueError(f"Nieznany podzbiór {subset!r} (dostępne: {list(SUBSETS)})")
    return points, X


def tsne_coords(X: np.ndarray, perplexity: float | None = None, seed: int = 42) -> np.ndarray:
    """Ta sama metoda co dla zdań (kosinus, inicjalizacja PCA)."""
    return te.tsne_2d(X, perplexity=perplexity, seed=seed)


# --------------------------------------------------------------------------
# Wykres interaktywny
# --------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def hover_texts(points: pd.DataFrame, rank_col: str | None) -> list[str]:
    out = []
    for r in points.itertuples(index=False):
        r = r._asdict()
        counts = ", ".join(f"{l} {int(r[f'n_{l}'])}" for l in [*config.TOKEN_LABELS, config.NO_LABEL]
                           if r[f"n_{l}"] > 0)
        context = ", ".join(f"{l} {_pct(r[f'share_{l}'])}" for l in config.SENT_LABELS
                            if r[f"share_{l}"] == r[f"share_{l}"] and r[f"share_{l}"] > 0)
        surface = f" (w korpusie: „{r['surface']}”)" if r["surface"] != r["word"] else ""
        rank = f" | ranga w modelu: {int(r[rank_col]):,}".replace(",", " ") if rank_col and r[rank_col] == r[rank_col] else ""
        out.append(
            f"<b>{r['word']}</b>{surface}<br>"
            f"etykieta słowa: <b>{r['word_label']}</b> | anotacje: {counts} ({_pct(r['annot_share'])} ≠ O)<br>"
            f"wystąpienia: {int(r['freq'])} w {int(r['n_sentences'])} zdaniach | "
            f"kontekst zdań: <b>{r['context_label']}</b> ({context or 'brak'})<br>"
            f"PolEmo: {int(r['polemo_freq'])} wystąpień{rank}")
    return out


def marker_sizes(freq: pd.Series) -> np.ndarray:
    return np.clip(8 + 3 * np.log2(freq.to_numpy(dtype=float)), 8, 20)


def make_word_figure(points: pd.DataFrame, xy: np.ndarray, title: str, rank_col: str | None = None):
    """Wykres Plotly: lista rozwijana przełącza kolorowanie (etykieta słowa / kontekst zdań),
    przycisk włącza podpisy słów z etykietą ≠ O występujących ≥ 2 razy. Klik w legendę ukrywa klasę."""
    import plotly.graph_objects as go

    hover = np.array(hover_texts(points, rank_col), dtype=object)
    sizes = marker_sizes(points["freq"])
    labelled = (points["freq"] >= 2) & (points["word_label"] != config.NO_LABEL)
    label_text = np.where(labelled, points["word"], "")
    fig = go.Figure()
    traces_per_scheme = []
    for si, (col, (legend_title, styles)) in enumerate(SCHEMES.items()):
        idxs = []
        # tło najpierw, żeby było pod kolorowymi punktami
        for lab in sorted(styles, key=lambda l: not styles[l].background):
            m = (points[col] == lab).to_numpy()
            if not m.any():
                continue
            st = styles[lab]
            fig.add_trace(go.Scatter(
                x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{lab} ({int(m.sum())})",
                legendgroup=col, visible=si == 0, text=label_text[m], textposition="top center",
                textfont=dict(size=10, color="#333333"),
                marker=dict(size=np.where(st.background, 6, sizes[m]), color=st.color, symbol=st.symbol,
                            opacity=0.45 if st.background else 0.9,
                            line=dict(width=1.5 if st.hollow else 1, color=st.color if st.hollow else "white")),
                customdata=hover[m], hovertemplate="%{customdata}<extra></extra>"))
            idxs.append(len(fig.data) - 1)
        traces_per_scheme.append(idxs)

    buttons = []
    for si, (col, (legend_title, _)) in enumerate(SCHEMES.items()):
        vis = [i in traces_per_scheme[si] for i in range(len(fig.data))]
        buttons.append(dict(label=f"kolor: {legend_title}", method="update",
                            args=[{"visible": vis}, {"legend.title.text": legend_title}]))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.97),
        margin=dict(t=110), template="plotly_white", height=700,
        legend=dict(title=SCHEMES["word_label"][0]),
        xaxis=dict(title="t-SNE 1", showticklabels=False, zeroline=False),
        yaxis=dict(title="t-SNE 2", showticklabels=False, zeroline=False),
        hoverlabel=dict(align="left", bgcolor="white"),
        updatemenus=[
            dict(type="dropdown", buttons=buttons, x=0.0, y=1.08, xanchor="left", yanchor="top"),
            dict(type="buttons", direction="right", x=0.42, y=1.08, xanchor="left", yanchor="top",
                 buttons=[dict(label="podpisy: wył.", method="restyle", args=[{"mode": "markers"}]),
                          dict(label="podpisy: wł.", method="restyle", args=[{"mode": "markers+text"}])]),
        ],
    )
    return fig


# --------------------------------------------------------------------------
# Obraz statyczny do raportu (styl report/build_assets.py)
# --------------------------------------------------------------------------
def save_static_png(points: pd.DataFrame, xy: np.ndarray, title: str, path, n_labels: int = 25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True, layout="constrained")
    for ax, (col, (panel_title, styles)) in zip(axes, SCHEMES.items()):
        for lab in sorted(styles, key=lambda l: not styles[l].background):
            m = (points[col] == lab).to_numpy()
            if not m.any():
                continue
            st = styles[lab]
            size = 8 if st.background else marker_sizes(points.loc[m, "freq"]) ** 2 / 4
            ax.scatter(xy[m, 0], xy[m, 1], s=size, marker=st.marker,
                       facecolor="none" if st.hollow else st.color, edgecolor=st.color if st.hollow else "white",
                       linewidth=0.6 if st.hollow else 0.4, alpha=0.45 if st.background else 0.9,
                       label=f"{lab} ({int(m.sum())})", zorder=1 if st.background else 2)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8,
                  markerscale=0.8)
    top = points[points["word_label"] != config.NO_LABEL].nlargest(n_labels, "freq")
    for i in top.index:
        axes[0].annotate(points.at[i, "word"], xy[i], xytext=(3, 3), textcoords="offset points",
                         fontsize=7, color="#333333")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Ilościowo: czy etykiety się grupują (oryginalna przestrzeń, kosinus)
# --------------------------------------------------------------------------
def label_separation(X: np.ndarray, labels, k: int = 10, n_perm: int = 1000, seed: int = 0) -> dict:
    """Czystość k-NN (odsetek k najbliższych sąsiadów z tą samą etykietą) wobec losowego
    przetasowania etykiet oraz współczynnik sylwetki (kosinus)."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    k = min(k, len(labels) - 1)
    Xn = te.l2norm(np.asarray(X, dtype=np.float64))
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nbrs = np.argsort(-S, axis=1)[:, :k]

    def purity(lab):
        return float((lab[nbrs] == lab[:, None]).mean())

    observed = purity(labels)
    rng = np.random.default_rng(seed)
    perm = np.array([purity(rng.permutation(labels)) for _ in range(n_perm)])
    per_class = {c: float((labels[nbrs[labels == c]] == c).mean()) for c in np.unique(labels)}
    return {"n": int(len(labels)), "k": int(k), "counts": {c: int((labels == c).sum()) for c in np.unique(labels)},
            "purity": observed, "purity_per_class": per_class,
            "random_mean": float(perm.mean()), "random_ci95": [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))],
            "p_value": float((1 + (perm >= observed).sum()) / (1 + n_perm)),
            "silhouette": float(silhouette_score(X, labels, metric="cosine"))}


def polarity_probe(X: np.ndarray, labels, n_splits: int = 5, n_repeats: int = 5, n_perm: int = 100,
                   seed: int = 0) -> dict:
    """Sonda liniowa: czy etykietę da się odczytać z wektora, nawet jeśli słowa nie tworzą skupisk.

    Regresja logistyczna na znormalizowanych wektorach, zrównoważona trafność w walidacji
    krzyżowej (``n_repeats`` podziałów); odniesienie: te same etykiety przetasowane ``n_perm`` razy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    Xn, y = te.l2norm(np.asarray(X, dtype=np.float64)), np.asarray(labels)

    def score(target, split_seed):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=split_seed)
        return cross_val_score(LogisticRegression(max_iter=2000), Xn, target, cv=cv,
                               scoring="balanced_accuracy").mean()

    observed = np.array([score(y, s) for s in range(n_repeats)])
    rng = np.random.default_rng(seed)
    perm = np.array([score(rng.permutation(y), i) for i in range(n_perm)])
    return {"n": int(len(y)), "balanced_accuracy": float(observed.mean()), "sd_repeats": float(observed.std()),
            "random_mean": float(perm.mean()),
            "random_ci95": [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))],
            "p_value": float((1 + (perm >= observed.mean()).sum()) / (1 + n_perm))}


SEPARATION_SETS = {
    "WYR_POZ vs WYR_NEG (etykieta słowa)": ("word_label", ["WYR_POZ", "WYR_NEG"]),
    "5 etykiet słów (bez O)": ("word_label", config.TOKEN_LABELS),
    "POZ vs NEG (kontekst zdań)": ("context_label", ["POZ", "NEG"]),
}
PROBE_SETS = ["WYR_POZ vs WYR_NEG (etykieta słowa)", "POZ vs NEG (kontekst zdań)"]  # sonda: tylko pary klas


def separation_report(model: str, k: int = 10, n_perm: int = 1000, seed: int = 0) -> dict:
    points, X = load_points(model, "all")
    out = {}
    for name, (col, labs) in SEPARATION_SETS.items():
        m = points[col].isin(labs).to_numpy()
        out[name] = label_separation(X[m], points.loc[m, col].to_numpy(), k, n_perm, seed)
        if name in PROBE_SETS:
            out[name]["probe"] = polarity_probe(X[m], points.loc[m, col].to_numpy(), seed=seed)
    return out


def separation_markdown(results: dict[str, dict]) -> str:
    num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    lines = ["# Czy etykiety anotacji grupują się w przestrzeni osadzeń?", "",
             "Liczone w oryginalnej przestrzeni (100 wymiarów, kosinus), nie na współrzędnych t-SNE.", "",
             "| Model | Zestaw | n | Czystość k-NN | Losowo: średnia [95%] | p | Sylwetka |",
             "|---|---|---|---|---|---|---|"]
    for model, sets in results.items():
        for name, r in sets.items():
            lo, hi = r["random_ci95"]
            lines.append(f"| {MODEL_TITLES.get(model, model)} | {name} | {r['n']} | {num(r['purity'])} | "
                         f"{num(r['random_mean'])} [{num(lo)}; {num(hi)}] | {num(r['p_value'], 4)} | "
                         f"{num(r['silhouette'])} |")
    lines += ["", "**Sonda liniowa** (regresja logistyczna, 5-krotna walidacja krzyżowa ×5, zrównoważona trafność):", "",
              "| Model | Zestaw | n | Trafność | Losowo: średnia [95%] | p |", "|---|---|---|---|---|---|"]
    for model, sets in results.items():
        for name, r in sets.items():
            if "probe" in r:
                pr = r["probe"]
                lo, hi = pr["random_ci95"]
                lines.append(f"| {MODEL_TITLES.get(model, model)} | {name} | {pr['n']} | {num(pr['balanced_accuracy'])} | "
                             f"{num(pr['random_mean'])} [{num(lo)}; {num(hi)}] | {num(pr['p_value'], 3)} |")
    lines += ["", "**Czystość dla każdej etykiety osobno:**", "", "| Model | Zestaw | Etykieta (n): czystość |", "|---|---|---|"]
    for model, sets in results.items():
        for name, r in sets.items():
            cells = ", ".join(f"{c} ({r['counts'][c]}): {num(p)}" for c, p in r["purity_per_class"].items())
            lines.append(f"| {MODEL_TITLES.get(model, model)} | {name} | {cells} |")
    k = next(iter(next(iter(results.values())).values()))["k"]
    lines += ["", f"Czystość k-NN = średni odsetek k = {k} najbliższych sąsiadów (spośród słów zestawu) z tą samą "
              "etykietą. „Losowo” = ta sama miara po przetasowaniu etykiet (1000 razy): uwzględnia niezrównoważenie "
              "klas. p = odsetek przetasowań z czystością co najmniej taką jak obserwowana. Sylwetka od −1 do 1; "
              "wartości bliskie 0 oznaczają brak wyraźnych skupisk. Sonda liniowa sprawdza coś innego: czy etykietę da się "
              "odczytać z wektora prostym klasyfikatorem, nawet gdy słowa o tej samej etykiecie nie leżą blisko siebie."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Całość
# --------------------------------------------------------------------------
def run(models: list[str], subsets: list[str], perplexity: float | None = None, seed: int = 42) -> dict:
    out = config.WORD_EMB_DIR
    separation = {}
    for model in models:
        rank_col = f"{model}_rank"
        for subset in subsets:
            points, X = load_points(model, subset)
            xy = tsne_coords(X, perplexity, seed)
            stem = out / f"tsne_words_{model}_{subset}"
            cols = ["word", "word_label", "context_label", "freq", "annot_share"]
            pd.concat([points[cols], pd.DataFrame(xy, columns=["x", "y"])], axis=1).to_csv(
                stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
            title = f"t-SNE słów – {MODEL_TITLES.get(model, model)} – {SUBSETS[subset]} (n = {len(points)})"
            fig = make_word_figure(points, xy, title, rank_col if rank_col in points else None)
            fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn")
            save_static_png(points, xy, "\n".join(textwrap.wrap(title, 90)), stem.with_suffix(".png"))
            print(f"[{model}/{subset}] {len(points)} słów → {stem.name}.html/.png/.csv")
        separation[model] = separation_report(model)
    (out / "label_separation.json").write_text(json.dumps(separation, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "label_separation.md").write_text(separation_markdown(separation), encoding="utf-8")
    return separation


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["w2v", "w2v_polemo"])
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--perplexity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # konsola Windows (cp1250)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    run(a.models, a.subsets, a.perplexity, a.seed)
    print((config.WORD_EMB_DIR / "label_separation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
