#!/usr/bin/env python
"""Generuje z plików repo tabele (generated/*.tex) i wykresy (figures/*.png) dla sprawozdania.

Uruchomienie z katalogu głównego repo (tam, gdzie leży config.py):

    python report/build_assets.py            # wszystko
    python report/build_assets.py --no-boot  # bez bootstrapu różnic kappy (szybciej, bez przedziałów Δ)

Skąd biorą się liczby (nic nie jest wpisane ręcznie):
  results/iterN/agreement.json            tabele zgodności, kappa per klasa, macierze pomyłek
  results/iterN/confusion_*.png           kopiowane do report/figures/
  annotation/iterN/export/*.csv           bootstrap różnic kappy iteracja 2 - iteracja 1
  annotation/iterN/adjudicated.csv        wynik rozstrzygnięcia (superanotator)
  annotation/iterN_artefacts/disagreements_sentences.csv   rozbieżności A1/A2
  annotation/iterN/_key_gold.csv, sample.csv               etykiety PolEmo i próbka
  results/corpus_stats.json               statystyki korpusu
  results/text_embeddings/tsne_<model>.html   współrzędne t-SNE -> statyczne wykresy PNG
  notebooks/Text_Embeddings_executed.ipynb    tabele z analizy osadzeń zdań (po wykonaniu notebooka)

Część Word Embeddings: patrz funkcja build_word_embeddings() na końcu pliku (na razie pusta).
"""
import argparse
import base64
import io
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CLASSES = ["POZ", "NEG", "NEU", "AMB"]
TOKEN_CLASSES = ["WYR_POZ", "WYR_NEG", "WZM", "OSL", "ODW"]
COLORS = {"POZ": "#1b9e77", "NEG": "#d95f02", "NEU": "#7570b3", "AMB": "#e7298a"}


# --------------------------------------------------------------------------- narzędzia
def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def num(x, d=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}".replace(".", "{,}") + r"\%"


def ci(v, d=3) -> str:
    return f"[{num(v[0], d)}; {num(v[1], d)}]"


def write_tabular(name: str, header: list[str], rows: list[list], colspec: str, midrules=()):
    """Zapisuje samą tabelę (tabular) do generated/<name>.tex; opakowanie table jest w .tex."""
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        lines.append(" & ".join(str(c) for c in r) + r" \\")
        if i in midrules:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macro(name: str, value: str):
    """Makro \\name{value} do używania w tekście (generated/macros.tex)."""
    MACROS[name] = value


MACROS: dict[str, str] = {}


def interp(k):
    if k is None or np.isnan(k):
        return "--"
    if k < 0:
        return "poniżej losowej"
    for up, nm in [(0.20, "nieznaczna"), (0.40, "dostateczna"), (0.60, "umiarkowana"), (0.80, "znaczna")]:
        if k <= up:
            return nm
    return "prawie pełna"


def load_agreement(it: int) -> dict:
    return json.loads((ROOT / "results" / f"iter{it}" / "agreement.json").read_text(encoding="utf-8"))


ITERS = [1, 2]
LEVELS = [("sentences", "Zdania"), ("tokens", "Tokeny -- wszystkie"), ("tokens_marked", "Tokeny -- zaznaczone")]


# --------------------------------------------------------------------------- Część I
def build_agreement_tables():
    for it in ITERS:
        a = load_agreement(it)
        rows = []
        for key, label in LEVELS:
            lv = a["levels"][key]["measures"]
            c, f, r = lv["cohen_A1-A2"], lv["fleiss"], lv["raw_agreement"]
            rows.append([label, a["levels"][key]["n"], pct(r["value"]),
                         f"{num(c['value'])} {ci(c['ci95'])}", f"{num(f['value'])} {ci(f['ci95'])}",
                         interp(c["value"])])
        write_tabular(f"agreement_iter{it}", ["Poziom", "$n$", "Zgodność", r"$\kappa$ Cohena [95\% CI]",
                                             r"$\kappa$ Fleissa [95\% CI]", "Interpretacja"],
                      rows, "lrrllc")
        # uzupełniająco z gold
        g = a["levels"].get("sentences_with_gold")
        if g:
            m = g["measures"]
            rows = [["$\\kappa$ Fleissa (A1, A2, gold)", f"{num(m['fleiss']['value'])} {ci(m['fleiss']['ci95'])}",
                     interp(m["fleiss"]["value"])],
                    ["$\\kappa$ Cohena A1--gold", f"{num(m['cohen_A1-gold']['value'])} {ci(m['cohen_A1-gold']['ci95'])}",
                     interp(m["cohen_A1-gold"]["value"])],
                    ["$\\kappa$ Cohena A2--gold", f"{num(m['cohen_A2-gold']['value'])} {ci(m['cohen_A2-gold']['ci95'])}",
                     interp(m["cohen_A2-gold"]["value"])]]
            write_tabular(f"gold_iter{it}", ["Miara (zdania)", "Wartość [95\\% CI]", "Interpretacja"], rows, "lll")
    # makra z wartościami
    a1, a2 = load_agreement(1), load_agreement(2)
    for it, a in ((1, a1), (2, a2)):
        s = a["levels"]["sentences"]["measures"]
        t = a["levels"]["tokens"]["measures"]
        m = a["levels"]["tokens_marked"]["measures"]
        tag = "I" if it == 1 else "II"
        write_macro(f"kSentCohen{tag}", num(s["cohen_A1-A2"]["value"]))
        write_macro(f"kSentFleiss{tag}", num(s["fleiss"]["value"]))
        write_macro(f"kSentRaw{tag}", pct(s["raw_agreement"]["value"]))
        write_macro(f"kTokCohen{tag}", num(t["cohen_A1-A2"]["value"]))
        write_macro(f"kTokRaw{tag}", pct(t["raw_agreement"]["value"]))
        write_macro(f"kMarkCohen{tag}", num(m["cohen_A1-A2"]["value"]))
        write_macro(f"kMarkRaw{tag}", pct(m["raw_agreement"]["value"]))
        write_macro(f"nTok{tag}", str(a["levels"]["tokens"]["n"]))
        write_macro(f"nMark{tag}", str(a["levels"]["tokens_marked"]["n"]))


def build_perclass_tables():
    a1, a2 = load_agreement(1), load_agreement(2)
    for key, classes, name in (("sentences", CLASSES, "perclass_sent"), ("tokens", TOKEN_CLASSES, "perclass_tok")):
        rows = []
        for c in classes:
            k1, k2 = a1["per_class"][key][c]["cohen"], a2["per_class"][key][c]["cohen"]
            n1, n2 = a1["per_class"][key][c]["count"], a2["per_class"][key][c]["count"]
            rows.append([esc(c), num(k1), f"{n1['A1']}/{n1['A2']}", num(k2), f"{n2['A1']}/{n2['A2']}",
                         num(k2 - k1).replace("$-$", "$-$") if k2 - k1 < 0 else "$+$" + num(k2 - k1)])
        write_tabular(name, ["Klasa", r"$\kappa$ iter.~1", "A1/A2", r"$\kappa$ iter.~2", "A1/A2", r"$\Delta\kappa$"],
                      rows, "lrrrrr")


def _levels_for(it: int):
    from src import agreement as ag
    ex = config.export_dir(it)
    sent = pd.read_csv(ex / "sentences_wide.csv", encoding="utf-8-sig")
    tok = pd.read_csv(ex / "tokens_wide.csv", encoding="utf-8-sig", keep_default_na=False)
    return ag.build_levels(sent, tok, ["A1", "A2"]), ag


def _draws(lv, ag, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows_of = [np.flatnonzero(lv.groups == g) for g in range(lv.n_sentences)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, lv.n_sentences, lv.n_sentences)
        idx = np.concatenate([rows_of[g] for g in pick])
        m = ag.measures(lv, lv.ratings[idx])
        out.append((m["cohen_A1-A2"], m["fleiss"]))
    return np.array(out)


def build_comparison(use_boot: bool):
    a1, a2 = load_agreement(1), load_agreement(2)
    n_boot = config.BOOTSTRAP["n"]
    diffs = {}
    if use_boot:
        l1, ag = _levels_for(1)
        l2, _ = _levels_for(2)
        for x, y in zip(l1[:3], l2[:3]):
            d1 = _draws(x, ag, n_boot, config.BOOTSTRAP["seed"])
            d2 = _draws(y, ag, n_boot, config.BOOTSTRAP["seed"] + 1)
            d = d2 - d1
            diffs[x.name] = {"cohen": np.nanpercentile(d[:, 0], [2.5, 97.5]),
                             "fleiss": np.nanpercentile(d[:, 1], [2.5, 97.5])}
    rows, plot = [], []
    for key, label in LEVELS:
        m1, m2 = a1["levels"][key]["measures"], a2["levels"][key]["measures"]
        for mk, mname in (("cohen_A1-A2", "Cohen"), ("fleiss", "Fleiss")):
            k1, k2 = m1[mk]["value"], m2[mk]["value"]
            dd = diffs.get(key, {}).get("cohen" if mk.startswith("cohen") else "fleiss")
            dtxt = ci(dd) if dd is not None else "--"
            sign = "$+$" if k2 - k1 >= 0 else ""
            rows.append([label if mname == "Cohen" else "", mname, f"{num(k1)} {ci(m1[mk]['ci95'], 2)}",
                         f"{num(k2)} {ci(m2[mk]['ci95'], 2)}", f"{sign}{num(k2 - k1)}", dtxt])
            if mname == "Cohen":
                plot.append((label.replace("--", "-"), k1, m1[mk]["ci95"], k2, m2[mk]["ci95"]))
    write_tabular("comparison", ["Poziom", "Miara", "Iteracja 1 [CI]", "Iteracja 2 [CI]", r"$\Delta$", r"$\Delta$ [95\% CI]"],
                  rows, "llllrl", midrules=(1, 3))
    # wykres
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    xs = np.arange(len(plot))
    for off, idx, col, lab in ((-0.17, 1, "#4c72b0", "Iteracja 1"), (0.17, 3, "#dd8452", "Iteracja 2")):
        v = np.array([p[idx] for p in plot])
        lo = v - np.array([p[idx + 1][0] for p in plot])
        hi = np.array([p[idx + 1][1] for p in plot]) - v
        ax.errorbar(xs + off, v, yerr=[lo, hi], fmt="o", color=col, capsize=4, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in plot])
    ax.set_ylabel(r"$\kappa$ Cohena (A1--A2)".replace("--", "-"))
    ax.set_ylim(0.2, 0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "kappa_comparison.png", dpi=200)
    plt.close(fig)


def copy_confusions():
    for it in ITERS:
        for n in ("sentences", "tokens"):
            src = ROOT / "results" / f"iter{it}" / f"confusion_{n}.png"
            if src.exists():
                shutil.copy(src, FIG / f"iter{it}_confusion_{n}.png")


def pair_label(a, b):
    return "/".join(sorted([a, b], key=lambda x: CLASSES.index(x)))


def build_adjudication():
    rows_types = {}
    stats = {}
    for it in ITERS:
        dis = pd.read_csv(ROOT / "annotation" / f"iter{it}_artefacts" / "disagreements_sentences.csv", encoding="utf-8-sig")
        adj = pd.read_csv(config.iter_dir(it) / "adjudicated.csv", encoding="utf-8-sig")
        dis["sent_id"] = dis["sent_id"].astype(float).astype(int)
        adj["sent_id"] = adj["sent_id"].astype(int)
        m = dis.merge(adj, on="sent_id", how="left")
        m["para"] = [pair_label(a, b) for a, b in zip(m["sentence_label_A1"], m["sentence_label_A2"])]
        for p, c in m["para"].value_counts().items():
            rows_types.setdefault(p, {})[it] = int(c)
        pick_a1 = int((m["final"] == m["sentence_label_A1"]).sum())
        pick_a2 = int((m["final"] == m["sentence_label_A2"]).sum())
        other = len(m) - pick_a1 - pick_a2
        stats[it] = dict(n=len(adj), conf=len(m), agreed=len(adj) - len(m), a1=pick_a1, a2=pick_a2, other=other,
                         final=adj["final"].value_counts().to_dict(),
                         conf_final=m["final"].value_counts().to_dict())
        tag = "I" if it == 1 else "II"
        write_macro(f"nConf{tag}", str(len(m)))
        write_macro(f"nAgreed{tag}", str(len(adj) - len(m)))
        write_macro(f"nPickA{tag}", str(pick_a1))
        write_macro(f"nPickB{tag}", str(pick_a2))
        write_macro(f"nPickOther{tag}", str(other))
    order = sorted(rows_types, key=lambda p: -(rows_types[p].get(1, 0) + rows_types[p].get(2, 0)))
    rows = [[esc(p), rows_types[p].get(1, 0), rows_types[p].get(2, 0)] for p in order]
    rows.append([r"\textbf{razem}", sum(r[1] for r in rows), sum(r[2] for r in rows)])
    write_tabular("conflict_types", ["Para etykiet A1/A2", "Iteracja 1", "Iteracja 2"], rows, "lrr",
                  midrules=(len(rows) - 2,))
    rows = []
    for it in ITERS:
        s = stats[it]
        rows.append([it, s["n"], s["agreed"], s["conf"], s["a1"], s["a2"], s["other"]])
    write_tabular("adjudication", ["Iteracja", "Zdań", "Zgodne A1=A2", "Rozbieżne", "Decyzja = A1", "Decyzja = A2",
                                   "Inna etykieta"], rows, "rrrrrrr")
    return stats


def build_descriptive(stats):
    cs = json.loads(config.CORPUS_STATS.read_text(encoding="utf-8"))
    gold_n = cs["gold_counts"]
    tot = sum(gold_n.values())
    rows = [[c, f"{gold_n[c]:,}".replace(",", r"\,"), pct(gold_n[c] / tot)] for c in CLASSES]
    rows.append([r"\textbf{razem}", f"{tot:,}".replace(",", r"\,"), pct(1)])
    write_tabular("corpus_gold", ["Etykieta (gold)", "Zdań", "Udział"], rows, "lrr", midrules=(3,))
    q = cs["n_tokens_quantiles"]
    write_macro("corpusLines", f"{cs['cleaning']['lines']:,}".replace(",", r"\,"))
    write_macro("corpusUnique", f"{cs['cleaning']['unique_sentences']:,}".replace(",", r"\,"))
    write_macro("corpusDup", str(cs["cleaning"]["duplicate_rows_removed"]))
    write_macro("corpusConflicts", str(cs["cleaning"]["gold_conflict_sentences"]))
    write_macro("corpusFrame", f"{cs['sampling_frame_size']:,}".replace(",", r"\,"))
    write_macro("corpusMean", num(cs["n_tokens_mean"], 1))
    write_macro("corpusMedian", str(int(q["q50"])))
    write_macro("corpusQlow", str(int(q["q5"])))
    write_macro("corpusQhigh", str(int(q["q95"])))

    # etykiety final i gold w próbkach
    frames = []
    for it in ITERS:
        adj = pd.read_csv(config.iter_dir(it) / "adjudicated.csv", encoding="utf-8-sig")
        key = pd.read_csv(config.iter_dir(it) / "_key_gold.csv", encoding="utf-8-sig")
        samp = pd.read_csv(config.iter_dir(it) / "sample.csv", encoding="utf-8-sig")
        key.columns = [c.strip() for c in key.columns]
        gcol = "gold" if "gold" in key.columns else key.columns[-1]
        d = adj.merge(key[["sent_id", gcol]].rename(columns={gcol: "gold"}), on="sent_id").merge(
            samp[["sent_id", "text"]], on="sent_id")
        d["iter"] = it
        d["n_tok"] = d["text"].str.split().str.len()
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for it in ITERS + ["razem"]:
        sub = d if it == "razem" else d[d["iter"] == it]
        vc = sub["final"].value_counts()
        rows.append([it] + [vc.get(c, 0) for c in CLASSES] + [len(sub), num(sub["n_tok"].mean(), 1),
                                                              int(sub["n_tok"].min()), int(sub["n_tok"].max())])
    write_tabular("final_dist", ["Iteracja"] + CLASSES + ["razem", "śr. tokenów", "min", "maks"], rows,
                  "lrrrrrrrr", midrules=(1,))
    ct = pd.crosstab(d["gold"], d["final"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    rows = [[f"gold {c}"] + [int(ct.loc[c, k]) for k in CLASSES] + [int(ct.loc[c].sum())] for c in CLASSES]
    rows.append([r"\textbf{razem}"] + [int(ct[k].sum()) for k in CLASSES] + [int(ct.values.sum())])
    write_tabular("gold_vs_final", [r"gold $\downarrow$ / final $\rightarrow$"] + CLASSES + ["razem"], rows,
                  "lrrrrr", midrules=(3,))
    same = int(sum(ct.loc[c, c] for c in CLASSES))
    write_macro("goldFinalSame", str(same))
    write_macro("goldFinalPct", pct(same / ct.values.sum()))
    write_macro("goldAmbToNeu", str(int(ct.loc["AMB", "NEU"])))
    write_macro("goldAmbN", str(int(ct.loc["AMB"].sum())))
    write_macro("goldAmbKept", str(int(ct.loc["AMB", "AMB"])))
    write_macro("finalAmbN", str(int(ct["AMB"].sum())))


# --------------------------------------------------------------------------- Text Embeddings
def _decode(arr):
    if isinstance(arr, dict) and "bdata" in arr:
        return np.frombuffer(base64.b64decode(arr["bdata"]), dtype=np.dtype(arr["dtype"]))
    return np.asarray(arr, dtype=float)


def _load_tsne(model: str):
    h = (ROOT / "results" / "text_embeddings" / f"tsne_{model}.html").read_text(encoding="utf-8")
    i = h.find("Plotly.newPlot(")
    j = h.find("[", i)
    data, _ = json.JSONDecoder().raw_decode(h[j:])
    traces = {}
    for t in data:
        traces.setdefault(t["legendgroup"], []).append(
            dict(name=t["name"], x=_decode(t["x"]), y=_decode(t["y"]), it=[row[5] for row in t["customdata"]]))
    return traces


MODEL_TITLES = {"fasttext": "fastText (uśrednione wektory słów)", "use": "Universal Sentence Encoder",
                "tfidf": "TF-IDF + SVD (punkt odniesienia)"}


def build_tsne_figures():
    for model in ("fasttext", "use", "tfidf"):
        p = ROOT / "results" / "text_embeddings" / f"tsne_{model}.html"
        if not p.exists():
            continue
        tr = _load_tsne(model)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
        for ax, grp, ttl in zip(axes, ("final", "gold"), ("etykieta final (uzgodniona)", "etykieta gold (PolEmo)")):
            for t in tr[grp]:
                it = np.array(t["it"])
                for itv, mk in (("1", "o"), ("2", "D")):
                    sel = it == itv
                    ax.scatter(t["x"][sel], t["y"][sel], s=28 if mk == "o" else 24, marker=mk,
                               color=COLORS[t["name"]], alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c], label=c) for c in CLASSES]
        handles += [plt.Line2D([], [], marker="o", ls="", color="gray", label="iteracja 1"),
                    plt.Line2D([], [], marker="D", ls="", color="gray", label="iteracja 2")]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.suptitle(f"t-SNE: {MODEL_TITLES[model]}", fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        fig.savefig(FIG / f"tsne_{model}.png", dpi=200)
        plt.close(fig)


def _nb_tables(nb, cell_idx):
    out = []
    for o in nb["cells"][cell_idx]["outputs"]:
        h = o.get("data", {}).get("text/html")
        if h and "<table" in "".join(h):
            out.append("".join(h))
    return out


def _find_cell(nb, needle):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return i
    raise KeyError(needle)


def _clean(df):
    return df.dropna(how="all")


def build_text_embedding_tables():
    nbp = ROOT / "notebooks" / "Text_Embeddings_executed.ipynb"
    nb = json.loads(nbp.read_text(encoding="utf-8"))

    # 3.5 klasyfikacja
    h = _nb_tables(nb, _find_cell(nb, "te.evaluate_embeddings"))[0]
    t = _clean(pd.read_html(io.StringIO(h), header=0, index_col=[0, 1])[0])
    rows = []
    for (model, lab), r in t.iterrows():
        rows.append([esc(model) if lab == "final" else "", esc(lab), int(r["n"]), num(r["accuracy"]), num(r["baza_wiekszosciowa"]),
                     num(r["macro_F1"]), num(r["kNN10_czystosc"]), num(r["losowy_wg_rozkladu"]), num(r["silhouette_cos"])])
    write_tabular("emb_classif", ["Model", "Etykiety", "$n$", "accuracy", "baza większ.", "macro-F1",
                                  "kNN-10", "losowy", "silhouette"], rows, "llrrrrrrr", midrules=(3, 7))
    for (model, lab), r in t.iterrows():
        if lab == "final":
            write_macro(f"acc{model.capitalize()}", num(r["accuracy"]))
            write_macro(f"base{model.capitalize()}", num(r["baza_wiekszosciowa"]))
            write_macro(f"fone{model.capitalize()}", num(r["macro_F1"]))
            write_macro(f"sil{model.capitalize()}", num(r["silhouette_cos"]))

    # macierze pomyłek i raporty per klasa
    ci_ = _find_cell(nb, "te.confusion_report")
    tabs = _nb_tables(nb, ci_)
    models = ["fasttext", "use", "tfidf"]
    rec_rows = []
    for k, m in enumerate(models):
        cm = _clean(pd.read_html(io.StringIO(tabs[2 * k]), header=0, index_col=0)[0])
        rep = _clean(pd.read_html(io.StringIO(tabs[2 * k + 1]), header=0, index_col=0)[0])
        rows = [[f"{esc(c)}"] + [int(cm.loc[c, p]) for p in CLASSES] for c in CLASSES]
        write_tabular(f"emb_cm_{m}", [r"etykieta $\downarrow$ / predykcja $\rightarrow$"] + CLASSES, rows, "lrrrr")
        rec_rows.append([esc(m)] + [num(rep.loc[c, "recall"], 2) for c in CLASSES])
    write_tabular("emb_recall", ["Model"] + [f"recall {c}" for c in CLASSES], rec_rows, "lrrrr")

    # pary kontrolne
    tabs = _nb_tables(nb, _find_cell(nb, "te.probe_table"))
    probes = _clean(pd.read_html(io.StringIO(tabs[0]), header=0, index_col=0)[0])
    rows = [[esc(r["kategoria"]), esc(r["zdanie_a"]), esc(r["zdanie_b"]), num(r["fasttext"], 2), num(r["use"], 2), num(r["tfidf"], 2)]
            for _, r in probes.iterrows()]
    write_tabular("emb_probes", ["Kategoria", "Zdanie a", "Zdanie b", "ft", "USE", "TF-IDF"], rows, "p{2.6cm}p{4.4cm}p{4.9cm}rrr")
    means = _clean(pd.read_html(io.StringIO(tabs[1]), header=0, index_col=0)[0])
    rows = [[esc(k)] + [num(means.loc[k, c], 3) for c in ("fasttext", "use", "tfidf")] for k in means.index]
    write_tabular("emb_probe_means", ["Kategoria par", "fastText", "USE", "TF-IDF"], rows, "lrrr")

    # długość zdania
    tabs = _nb_tables(nb, _find_cell(nb, "te.length_stats"))
    names = {"fasttext": "fastText", "use": "USE", "tfidf": "TF-IDF"}
    rows = []
    for m, h in zip(("fasttext", "use", "tfidf"), tabs[:3]):
        t = _clean(pd.read_html(io.StringIO(h), header=0, index_col=0)[0])
        for i, (idx, r) in enumerate(t.iterrows()):
            if idx.startswith("Spearman"):
                rows.append(["", r"\emph{Spearman} (tokeny vs. śr. cos)", "", "", "", num(r["srednie_cos_do_innych"], 3)])
                continue
            rows.append([names[m] if i == 0 else "", esc(idx.replace("(", "(").replace("]", "]")), int(r["n"]),
                         num(r["norma_wektora"], 2) if m != "use" else "1{,}00", num(r["cos_do_centroidu"], 3),
                         num(r["srednie_cos_do_innych"], 3)])
    write_tabular("emb_length", ["Model", "Grupa (tokeny)", "$n$", "norma", "cos do centroidu", "śr. cos do innych"], rows,
                  "llrrrr", midrules=(3, 7))


# --------------------------------------------------------------------------- Word Embeddings
def build_word_embeddings():
    """Do uzupełnienia po wykonaniu części Word Embeddings.

    Miejsce na generowanie tabel (listy k-najbliższych sąsiadów, porównanie modeli) i wykresów t-SNE
    słów; pliki wejściowe dodaj do results/word_embeddings/. Sprawozdanie włącza je przez \\input{generated/...}.
    """
    return


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-boot", action="store_true", help="pomiń bootstrap różnic kappy (szybciej)")
    args = ap.parse_args()
    build_agreement_tables()
    build_perclass_tables()
    copy_confusions()
    build_comparison(not args.no_boot)
    stats = build_adjudication()
    build_descriptive(stats)
    build_tsne_figures()
    build_text_embedding_tables()
    build_word_embeddings()
    lines = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gotowe: {len(list(GEN.glob('*.tex')))} plików w {GEN}, {len(list(FIG.glob('*.png')))} wykresów w {FIG}")


if __name__ == "__main__":
    main()
