"""Zadanie 1, Część II – Text Embeddings: pliki do sprawozdania dla uśrednionego Word2Vec (pkt 1–3).

Uruchomienie (z katalogu Lab1, po python -m src.text_analysis):  python -m src.text_report

Wejście:  results/text_embeddings/analiza_w2v.json, probes_w2v.csv, emb_<model>.json (pokrycie słownika),
          report/generated/emb_*.tex (wyniki zespołu dla fastText / USE / TF-IDF)
Wyjście:  report/generated/te_w2v_macros.tex   liczby jako makra \\teW… (używa ich sections/te_w2v.tex)
          report/generated/te_w2v_probes.tex   pary kontrolne: średni kosinus, 5 modeli
          report/generated/te_w2v_length.tex   długość zdania a wektor
          report/generated/te_w2v_classif.tex  klasyfikacja wydźwięku i czułość klas, 5 modeli
Rysunki report/figures/tsne_w2v*.png tworzy report/build_assets.py (build_tsne_figures).
"""
from __future__ import annotations

import json

import pandas as pd

import config
from src import text_analysis as ta
from src import text_embeddings as te
from src.word_report import esc, num, pct

GEN = config.ROOT / "report" / "generated"
MODELS = ["w2v", "w2v_polemo"]
SUFFIX = {"w2v": "Ipi", "w2v_polemo": "Pol"}
TITLE = {"w2v": "Word2Vec IPI PAN", "w2v_polemo": "Word2Vec PolEmo"}
TEAM_TITLE = {"fasttext": "fastText", "use": "USE", "tfidf": "TF-IDF"}
CATEGORY = {"parafraza": "Para", "przeciwny sens, podobne słowa": "Opp", "przeciwny sens, te same słowa": "OppSame",
            "ten sam sens, różna długość": "Len", "kontrola: niepowiązane": "Unrel"}
EXAMPLES = {"Mila": ("Obsługa była bardzo miła .", "Obsługa była bardzo niemiła ."),
            "Polecam": ("Polecam .", None)}


def load(out_dir=None) -> dict:
    out_dir = out_dir or te.OUT_DIR
    return {"analysis": json.loads((out_dir / "analiza_w2v.json").read_text(encoding="utf-8")),
            "probes": pd.read_csv(out_dir / "probes_w2v.csv", encoding="utf-8-sig"),
            "coverage": {m: json.loads((out_dir / f"emb_{m}.json").read_text(encoding="utf-8"))["mean_vocab_coverage"]
                         for m in MODELS}}


def _length_row(rows: list[dict], group: str) -> dict:
    return next(r for r in rows if r["grupa"] == group)


def build_macros(d: dict) -> dict[str, str]:
    a = d["analysis"]
    m = {"teWBase": num(a["q5_eval"]["w2v"][0]["baza_wiekszosciowa"]),
         "teWAllOpp": pct(a["q1_opposite_top"]["w2v"]["all_opposite"], 1),
         "teWAllSame": pct(a["q1_opposite_top"]["w2v"]["all_same"], 1)}
    for model in MODELS:
        s = SUFFIX[model]
        means = a["probe_means"][model]
        m |= {f"teW{CATEGORY[c]}{s}": num(v) for c, v in means.items() if c in CATEGORY}
        m[f"teWCov{s}"] = pct(d["coverage"][model], 1)
        top = a["q1_opposite_top"][model]
        m |= {f"teWTopOpp{s}": pct(top["top_opposite"], 1), f"teWTopSame{s}": pct(top["top_same"], 1)}
        lex = a["q2_lexical"][model]
        cos = list(lex["mean_cos"].values())
        m |= {f"teWJac{s}": num(lex["spearman_cos_jaccard"], 2), f"teWNoShared{s}": num(cos[0], 2),
              f"teWShared{s}": num(cos[2], 2)}
        rows = a["q4_length"][model]
        groups = [r["grupa"] for r in rows if not r["grupa"].startswith("Spearman")]
        short, long = _length_row(rows, groups[0]), _length_row(rows, groups[-1])
        sp = next(r for r in rows if r["grupa"].startswith("Spearman"))
        m |= {f"teWNormShort{s}": num(short["norma_wektora"], 2), f"teWNormLong{s}": num(long["norma_wektora"], 2),
              f"teWCentShort{s}": num(short["cos_do_centroidu"]), f"teWCentLong{s}": num(long["cos_do_centroidu"]),
              f"teWSpearman{s}": num(sp["srednie_cos_do_innych"], 2)}
        final = next(r for r in a["q5_eval"][model] if r["etykiety"] == "final")
        m |= {f"teWAcc{s}": num(final["accuracy"]), f"teWFone{s}": num(final["macro_F1"]),
              f"teWRecNeg{s}": num(a["q5_recall"][model]["NEG"], 2),
              f"teWRecNeu{s}": num(a["q5_recall"][model]["NEU"], 2)}
        probes = d["probes"]
        for key, (sa, sb) in EXAMPLES.items():
            row = probes[(probes["zdanie_a"] == sa) & ((probes["zdanie_b"] == sb) if sb else True)].iloc[0]
            m[f"teW{key}{s}"] = num(row[model], 2)
    ft = ta.team_length()
    m["teWSpearmanFt"] = num(ft[(ft["model"] == "fastText") & (ft["grupa"] == "Spearman")]["spearman"].iat[0], 2)
    return m


def macros_tex(macros: dict[str, str]) -> str:
    head = "% Generowane przez: python -m src.text_report (nie edytować ręcznie)\n"
    return head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))


def probes_tex(d: dict) -> str:
    team = ta.team_probe_means()
    means = d["analysis"]["probe_means"]
    rows = []
    for cat in CATEGORY:
        cells = [num(means[m][cat]) for m in MODELS] + [num(team.at[cat, c]) if cat in team.index else "--"
                                                         for c in team.columns]
        rows.append(f"{esc(cat)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(["W2V IPI PAN", "W2V PolEmo", *team.columns])
    return (r"\begin{tabular}{lrrrrr}" "\n" r"\toprule" "\n" f"Kategoria par & {head} \\\\\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def length_tex(d: dict) -> str:
    rows = []
    for k, model in enumerate(MODELS):
        if k:
            rows.append(r"\midrule")
        for i, r in enumerate(d["analysis"]["q4_length"][model]):
            name = TITLE[model] if i == 0 else ""
            if r["grupa"].startswith("Spearman"):
                rows.append(f" & \\emph{{Spearman}} (tokeny vs. śr. cos) &  &  &  & {num(r['srednie_cos_do_innych'])} \\\\")
            else:
                rows.append(f"{name} & {r['grupa']} & {int(r['n'])} & {num(r['norma_wektora'], 2)} & "
                            f"{num(r['cos_do_centroidu'])} & {num(r['srednie_cos_do_innych'])} \\\\")
    return (r"\begin{tabular}{llrrrr}" "\n" r"\toprule" "\n"
            r"Model & Grupa (tokeny) & $n$ & norma & cos do centroidu & śr. cos do innych \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def classif_tex(d: dict) -> str:
    a = d["analysis"]
    rows = []
    for model in MODELS:
        r = next(x for x in a["q5_eval"][model] if x["etykiety"] == "final")
        rec = a["q5_recall"][model]
        rows.append(f"{TITLE[model]} & {num(r['accuracy'])} & {num(r['macro_F1'])} & {num(r['kNN10_czystosc'])} & "
                    f"{num(r['silhouette_cos'])} & " + " & ".join(num(rec[c], 2) for c in config.SENT_LABELS) + r" \\")
    team_c = ta.team_classif().set_index("model")
    team_r = ta.team_recall()
    rows.append(r"\midrule")
    for key, title in TEAM_TITLE.items():
        if key in team_c.index:
            r = team_c.loc[key]
            rec = [num(team_r.at[key, c], 2) if key in team_r.index else "--" for c in config.SENT_LABELS]
            rows.append(f"{title} & {num(r['accuracy'])} & {num(r['macro_F1'])} & {num(r['kNN10'])} & "
                        f"{num(r['silhouette'])} & " + " & ".join(rec) + r" \\")
    return (r"\begin{tabular}{lrrrrrrrr}" "\n" r"\toprule" "\n"
            r"Model & accuracy & macro-F1 & kNN-10 & silhouette & recall POZ & recall NEG & recall NEU & recall AMB \\"
            "\n" r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def run(out_dir=None, gen_dir=GEN) -> dict[str, str]:
    d = load(out_dir)
    macros = build_macros(d)
    gen_dir.mkdir(parents=True, exist_ok=True)
    files = {"te_w2v_macros.tex": macros_tex(macros), "te_w2v_probes.tex": probes_tex(d),
             "te_w2v_length.tex": length_tex(d), "te_w2v_classif.tex": classif_tex(d)}
    for name, content in files.items():
        (gen_dir / name).write_text(content, encoding="utf-8")
    return macros


def main() -> None:
    macros = run()
    print(f"Zapisano {len(macros)} makr i 3 tabele w {GEN}")


if __name__ == "__main__":
    main()
