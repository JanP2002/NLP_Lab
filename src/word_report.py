"""Zadanie 1, Część II – Word Embeddings: pliki do sprawozdania LaTeX (pkt 1–3, liczby do dyskusji pkt 4).

Uruchomienie (z katalogu Lab1, po pkt 1–3):  python -m src.word_report

Wejście:  results/word_embeddings/ (coverage.json, label_separation.json, knn_summary.json,
          neighbours_<model>.csv, tsne_words_<model>_all.png)
Wyjście:  report/generated/we_macros.tex   liczby jako makra \\we… (używa ich też sections/we_dyskusja.tex)
          report/generated/we_models.tex   modele i pokrycie słownika (pkt 1)
          report/generated/we_tsne.tex     rysunki t-SNE i separacja etykiet (pkt 2)
          report/generated/we_knn.tex      porównanie list najbliższych słów (pkt 3)
          report/figures/tsne_words_<model>_all.png
Formatowanie jak w report/build_assets.py: przecinek dziesiętny 0{,}648, minus $-$.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

import config
from src import text_embeddings as te
from src import word_embeddings as we
from src import word_neighbours as wn

REPORT = config.ROOT / "report"
GEN, FIG = REPORT / "generated", REPORT / "figures"
MODELS = ["w2v", "w2v_polemo"]
SUFFIX = {"w2v": "Ipi", "w2v_polemo": "Pol"}            # nazwy makr: tylko litery
TITLE = {"w2v": "IPI PAN", "w2v_polemo": "PolEmo"}
PAIR = "w2v vs w2v_polemo"
WORD_SET = "WYR_POZ vs WYR_NEG (etykieta słowa)"
CTX_SET = "POZ vs NEG (kontekst zdań)"
SHOWCASE = ["miła", "polecam", "dobrze", "ok", "brudne", "problem", "niestety", "bardzo", "trochę", "nie"]
TEX_ESCAPE = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
              "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s: str) -> str:
    return "".join(TEX_ESCAPE.get(c, c) for c in str(s))


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def pct(x: float, d: int = 0) -> str:
    """Procent bez znaku % (dopisuje go tekst: \\weCovTypesIpi\\%)."""
    return f"{100 * x:.{d}f}".replace(".", "{,}")


def integer(n: int) -> str:
    """Odstęp tysięcy jak w macros.tex (41\\,373); liczby czterocyfrowe bez odstępu."""
    return f"{n:,}".replace(",", r"\,") if n >= 10_000 else str(n)


def load_inputs(out_dir=None) -> dict:
    out_dir = out_dir or config.WORD_EMB_DIR
    read = lambda name: json.loads((out_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    return {"coverage": read("coverage.json"), "separation": read("label_separation.json"),
            "knn": read("knn_summary.json"), "vocab": we.read_vocab(out_dir),
            "neighbours": {m: pd.read_csv(out_dir / f"neighbours_{m}.csv", encoding="utf-8-sig",
                                          keep_default_na=False) for m in MODELS}}


# --------------------------------------------------------------------------
# Makra
# --------------------------------------------------------------------------
def build_macros(d: dict, n_sentences: int, polemo_sentences: int, polemo_words: int) -> dict[str, str]:
    cov, sep, knn, vocab = d["coverage"], d["separation"], d["knn"], d["vocab"]
    m = {"weNWords": integer(len(vocab)), "weNTokens": integer(int(vocab["freq"].sum())),
         "weNSentences": integer(n_sentences), "weNLabelled": integer(int((vocab["word_label"] != "O").sum())),
         "wePolemoSentences": integer(polemo_sentences), "wePolemoWords": integer(polemo_words),
         "weNQueries": integer(knn["n_queries"]), "weK": str(knn["params"]["k"]),
         "weSharedVocab": integer(knn["params"]["shared_vocabulary"])}
    for model in MODELS:
        s, c = SUFFIX[model], cov["coverage"][model]
        ws, cs = sep[model][WORD_SET], sep[model][CTX_SET]
        pol, stem = knn["per_model"][model]["polarity"], knn["per_model"][model]["same_stem"]
        m |= {f"weVocab{s}": integer(cov["vocab_size"][model]), f"weInVocab{s}": integer(c["types"]["in_vocab"]),
              f"weCovTypes{s}": pct(c["types"]["share"], 1), f"weCovTokens{s}": pct(c["tokens"]["share"], 1),
              f"weCovAnn{s}": pct(c["annotated_types"]["share"], 1),
              f"weCovNeg{s}": pct(c["by_word_label"]["WYR_NEG"]["share"]),
              f"wePurity{s}": num(ws["purity"]), f"wePurityRand{s}": num(ws["random_mean"]),
              f"weSil{s}": num(ws["silhouette"]), f"wePurityPoz{s}": num(ws["purity_per_class"]["WYR_POZ"], 2),
              f"wePurityNeg{s}": num(ws["purity_per_class"]["WYR_NEG"], 2),
              f"weCtxPurity{s}": num(cs["purity"]), f"weCtxRand{s}": num(cs["random_mean"]),
              f"weProbe{s}": num(ws["probe"]["balanced_accuracy"], 2),
              f"weProbeRand{s}": num(ws["probe"]["random_mean"], 2),
              f"weCtxProbe{s}": num(cs["probe"]["balanced_accuracy"], 2),
              f"weStem{s}": pct(stem["wszystkie"]),
              f"wePolSamePoz{s}": pct(pol["WYR_POZ"]["same"]), f"wePolOppPoz{s}": pct(pol["WYR_POZ"]["opposite"]),
              f"wePolSameNeg{s}": pct(pol["WYR_NEG"]["same"]), f"wePolOppNeg{s}": pct(pol["WYR_NEG"]["opposite"])}
    pair = knn["pairs"][PAIR]
    m |= {"weOverlap": pct(pair["wszystkie"]["overlap"]), "weOverlapShared": pct(pair["wszystkie"]["overlap_shared"]),
          "weRhoWord": num(pair["wszystkie"]["rho2"], 2), "weRhoAll": num(knn["global_rho2"][PAIR], 2),
          "weOverlapWzm": pct(pair["WZM"]["overlap"]), "weOverlapSharedWzm": pct(pair["WZM"]["overlap_shared"])}
    pairs = cov["sanity"]["w2v"]["pairs"]
    m |= {"weCosDobryZly": num(pairs["dobry–zły"], 2), "weCosDobrySwietny": num(pairs["dobry–świetny"], 2),
          "weCosPolecamOdradzam": num(pairs["polecam–odradzam"], 2)}
    return m


def macros_tex(macros: dict[str, str]) -> str:
    head = "% Generowane przez: python -m src.word_report (nie edytować ręcznie)\n"
    return head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))


# --------------------------------------------------------------------------
# Sekcje
# --------------------------------------------------------------------------
def oov_examples(cov: dict) -> str:
    """Po jednym przykładzie słowa bez wektora (IPI PAN) dla każdej przyczyny."""
    wanted = [("brak polskich znaków", "zapisy bez polskich znaków"), ("zawiera cyfrę", "formy z cyframi"),
              ("rzadkie", "słowa rzadkie")]
    parts = []
    for key, desc in wanted:
        ex = next((e for e in cov["coverage"]["w2v"]["oov_examples"] if e["reason"].startswith(key)), None)
        if ex:
            hint = ex["reason"].split("w modelu: ")[1].rstrip(")") if "w modelu: " in ex["reason"] else None
            parts.append(f"{desc} (\\emph{{{esc(ex['word'])}}}" + (f" zamiast \\emph{{{esc(hint)}}})" if hint else ")"))
    return ", ".join(parts)


def models_tex(d: dict) -> str:
    cov = d["coverage"]
    labels = config.TOKEN_LABELS
    n_lab = {l: cov["coverage"]["w2v"]["by_word_label"][l]["n"] for l in labels}
    head = " & ".join(f"{esc(l)} ($n$={n_lab[l]})" for l in labels)
    rows = []
    for model in MODELS:
        c = cov["coverage"][model]
        cells = [f"{pct(c['types']['share'], 1)}\\%", f"{pct(c['tokens']['share'], 1)}\\%",
                 f"{pct(c['annotated_types']['share'], 1)}\\%"]
        cells += [f"{pct(c['by_word_label'][l]['share'])}\\%" for l in labels]
        rows.append(f"{TITLE[model]} & " + " & ".join(cells) + r" \\")
    return rf"""\subsection{{Modele i pokrycie słownika}}\label{{sec:we-models}}

Osadzenia policzono dla wszystkich \weNWords{{}} słów (\weNTokens{{}} wystąpień; bez interpunkcji i samych liczb)
z \weNSentences{{}} zaanotowanych zdań. Porównano dwa modele word2vec~\cite{{word2vec}} (skip-gram, 100 wymiarów),
które różnią się korpusem uczącym:
\begin{{itemize}}
  \item \textbf{{IPI PAN}} -- gotowy model \texttt{{nkjp+wiki-forms-all-100-skipg-ns}}~\cite{{dsmodels}}, uczony na pełnym
        Narodowym Korpusie Języka Polskiego i polskiej Wikipedii; wczytano \weVocabIpi{{}} najczęstszych form;
  \item \textbf{{PolEmo}} -- model wytrenowany przez nas w bibliotece gensim na wszystkich \wePolemoSentences{{}} zdaniach
        PolEmo (\wePolemoWords{{}} słów; słowa występujące co najmniej 3 razy, 15 epok); słownik \weVocabPol{{}} słów.
\end{{itemize}}
Wektor słowa wyszukiwano kolejno w formie małymi literami, w oryginalnej pisowni i z wielką literą na początku.

\begin{{table}}[H]
\centering\small
\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Model & słowa & wystąpienia & słowa zaanotowane & {head} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\caption{{Pokrycie słownika: odsetek słów (wystąpień) zaanotowanego korpusu, dla których model ma wektor.
„Słowa zaanotowane'' -- z co najmniej jedną anotacją różną od O; kolumny etykiet -- słowa o danej etykiecie
(najczęstsza etykieta anotacji, jeśli co najmniej połowa anotacji słowa jest różna od O).}}
\label{{tab:we-coverage}}
\end{{table}}

Brakujące wektory w modelu IPI PAN to głównie {oov_examples(cov)}.
"""


def tsne_tex(d: dict) -> str:
    sep = d["separation"]
    rows = []
    for model in MODELS:
        for name, short in [(WORD_SET, "WYR\\_POZ / WYR\\_NEG"), (CTX_SET, "POZ / NEG (zdania)")]:
            r = sep[model][name]
            pr = r["probe"]
            rows.append(f"{TITLE[model]} & {short} & {r['n']} & {num(r['purity'])} & {num(r['random_mean'])} & "
                        f"{num(r['silhouette'])} & {num(pr['balanced_accuracy'], 2)} & {num(pr['random_mean'], 2)} \\\\")
    figs = []
    for model in MODELS:
        figs.append(rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{tsne_words_{model}_all.png}}
\caption{{t-SNE słów -- word2vec {TITLE[model]}. Kolor: etykieta słowa (lewy) i dominująca etykieta zdań, w których słowo
wystąpiło (prawy); wielkość punktu rośnie z liczbą wystąpień, słowa bez etykiety są szarym tłem.}}
\label{{fig:we-tsne-{SUFFIX[model].lower()}}}
\end{{figure}}""")
    return rf"""\subsection{{Wizualizacja t-SNE słów}}\label{{sec:we-tsne}}

Rys.~\ref{{fig:we-tsne-ipi}} i~\ref{{fig:we-tsne-pol}} pokazują t-SNE (odległość kosinusowa) wszystkich słów z wektorem.
Wersje interaktywne -- po najechaniu na punkt widać słowo, jego etykietę, liczbę anotacji każdej etykiety i rozkład
etykiet zdań; lista rozwijana zmienia kolorowanie -- są w repozytorium
(\texttt{{results/word\_embeddings/tsne\_words\_*.html}}). Ponieważ t-SNE zniekształca odległości, grupowanie
sprawdzono też w oryginalnej przestrzeni (tab.~\ref{{tab:we-separation}}).

{chr(10).join(figs)}

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Model & Etykiety & $n$ & czystość k-NN & losowo & sylwetka & sonda & losowo \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Czy etykiety grupują się w przestrzeni osadzeń (100 wymiarów, kosinus). Czystość k-NN -- średni odsetek
10 najbliższych sąsiadów z tą samą etykietą; „losowo'' -- ta sama miara po przetasowaniu etykiet (1000 razy, $p<0{{,}}01$
we wszystkich wierszach). Sonda -- zrównoważona trafność regresji logistycznej (5-krotna walidacja krzyżowa).}}
\label{{tab:we-separation}}
\end{{table}}
"""


def showcase_rows(d: dict, words: list[str], top: int = 5) -> list[str]:
    rows = []
    for w in words:
        cells, label = [], None
        for model in MODELS:
            t = d["neighbours"][model]
            g = t[t["query"] == w].sort_values("rank").head(top)
            if g.empty:
                break
            label = g["query_label"].iat[0]
            expected = wn.EXPECTED_POLARITY.get(label)
            parts = []
            for r in g.itertuples():
                txt = esc(r.neighbour)
                if expected and r.polarity_class not in ("", "0", expected):
                    txt += r"$^{\otimes}$"
                parts.append(txt)
            cells.append(", ".join(parts))
        if len(cells) == len(MODELS):
            rows.append(f"\\emph{{{esc(w)}}} & {esc(label)} & " + " & ".join(cells) + r" \\")
    return rows


def knn_tex(d: dict) -> str:
    knn = d["knn"]
    pair = knn["pairs"][PAIR]
    rows = []
    for lab in [*config.TOKEN_LABELS, "wszystkie"]:
        if lab in pair:
            r = pair[lab]
            rows.append(f"{esc(lab)} & {r['n']} & {pct(r['overlap'])}\\% & {pct(r['overlap_shared'])}\\% & "
                        f"{num(r['rho2'], 2)} \\\\")
    per_model = []
    for model in MODELS:
        pm = knn["per_model"][model]
        p, n = pm["polarity"]["WYR_POZ"], pm["polarity"]["WYR_NEG"]
        per_model.append(f"{TITLE[model]} & {pct(pm['same_stem']['wszystkie'])}\\% & {pct(p['same'])}\\% / {pct(p['opposite'])}\\% "
                         f"& {pct(n['same'])}\\% / {pct(n['opposite'])}\\% \\\\")
    return rf"""\subsection{{Listy najbliższych słów}}\label{{sec:we-knn}}

Dla \weNQueries{{}} słów z etykietą różną od O, znanych obu modelom, wyznaczono \weK{{}} najbliższych słów (kosinus;
tylko słowa z liter, z całego słownika modelu). Listy porównano miarami: odsetek wspólnych sąsiadów (także przy
wyszukiwaniu tylko wśród \weSharedVocab{{}} słów znanych obu modelom), korelacja Spearmana podobieństw między słowami
zapytania w obu modelach (podobieństwo drugiego rzędu; dla całej macierzy $\rho=\weRhoAll$), udział odmian słowa zapytania
wśród sąsiadów oraz wydźwięk sąsiadów wyznaczony z etykiet korpusu PolEmo: $s=(\#\mathrm{{POZ}}-\#\mathrm{{NEG}})/\#$zdań
ze słowem (co najmniej 5 zdań, próg $\pm 0{{,}}1$).

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{lrrrr}}
\toprule
Etykieta & $n$ & wspólne@{knn['params']['k']} & wspólne@{knn['params']['k']} (wspólny słownik) & $\rho$ drugiego rzędu \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Zgodność list najbliższych słów: IPI PAN wobec PolEmo (średnie po słowach zapytania).}}
\label{{tab:we-knn-overlap}}
\end{{table}}

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{lrrr}}
\toprule
Model & odmiany & WYR\_POZ: zgodny / przeciwny & WYR\_NEG: zgodny / przeciwny \\
\midrule
{chr(10).join(per_model)}
\bottomrule
\end{{tabular}}
\caption{{Charakter list: udział odmian słowa zapytania wśród sąsiadów (wspólny początek wyrazu) oraz wydźwięk sąsiadów słów WYR\_POZ i WYR\_NEG (odsetek sąsiadów z oceną $s$).
Model PolEmo i miara $s$ pochodzą z tego samego korpusu, więc jego wyższa zgodność wydźwięku jest częściowo „z założenia''.}}
\label{{tab:we-knn-character}}
\end{{table}}

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{llp{{5.5cm}}p{{5.5cm}}}}
\toprule
Słowo & Etykieta & IPI PAN & PolEmo \\
\midrule
{chr(10).join(showcase_rows(d, SHOWCASE))}
\bottomrule
\end{{tabular}}
\caption{{Pięć najbliższych słów w obu modelach (pełne listy: \texttt{{results/word\_embeddings/neighbours\_*.csv}},
\texttt{{knn\_summary.md}}). $^{{\otimes}}$ -- wydźwięk sąsiada przeciwny do etykiety słowa zapytania.}}
\label{{tab:we-knn-examples}}
\end{{table}}
"""


# --------------------------------------------------------------------------
# Całość
# --------------------------------------------------------------------------
def run(out_dir=None, gen_dir=GEN, fig_dir=FIG) -> dict[str, str]:
    d = load_inputs(out_dir)
    corpus = we.polemo_sentences()
    macros = build_macros(d, n_sentences=len(te.load_sentences(d["coverage"]["iterations"])),
                          polemo_sentences=len(corpus), polemo_words=sum(map(len, corpus)))
    gen_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = {"we_macros.tex": macros_tex(macros), "we_models.tex": models_tex(d),
             "we_tsne.tex": tsne_tex(d), "we_knn.tex": knn_tex(d)}
    for name, content in files.items():
        (gen_dir / name).write_text(content, encoding="utf-8")
    for model in MODELS:
        src = (out_dir or config.WORD_EMB_DIR) / f"tsne_words_{model}_all.png"
        shutil.copyfile(src, fig_dir / src.name)
    return macros


def main() -> None:
    macros = run()
    print(f"Zapisano {len(macros)} makr i 3 sekcje w {GEN}; rysunki w {FIG}")


if __name__ == "__main__":
    main()
