"""Uśrednione Word2Vec (Text Embeddings, pkt 1b, 2, 3) w notebooku zespołu ``Text_Embeddings_executed.ipynb``.

    python -m src.notebook_w2v               # wstawia komórki, uruchamia je, eksportuje HTML
    python -m src.notebook_w2v --no-html

Notebook zespołu wykonano z oryginalnym USE z TF Hub, którego nie ma w Windows, więc nie uruchamiamy go od nowa
w całości (zmieniłyby się liczby USE i układ t-SNE fastText). Zamiast tego:

1. komórki Word2Vec (``W2V_CELLS``) trafiają do sekcji 1.4, 2.1 i 3.7 obu notebooków (źródłowy zostaje bez wyników),
2. uruchamiamy tylko je, po komórkach startowych zespołu (importy, konfiguracja, wczytanie zdań), oraz komórkę
   kontrolną, która porównuje wyniki z plikami w ``results/text_embeddings/`` (z nich pochodzi sprawozdanie),
3. wyniki wpisujemy do wykonanego notebooka; komórki zespołu i ich wyniki zostają bez zmian (sprawdzane).

Nowe komórki mają znacznik ``w2v`` i stałe id, więc ponowne uruchomienie zastępuje je zamiast dublować.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

import config

NB_DIR = config.ROOT / "notebooks"
SOURCE_NB = NB_DIR / "Text_Embeddings.ipynb"
EXECUTED_NB = NB_DIR / "Text_Embeddings_executed.ipynb"
ANSWERS_MD = config.RESULTS_DIR / "text_embeddings" / "odpowiedzi_porownanie.md"
TAG = "w2v"

INTRO_ID = "ab842773"                                   # komórka tytułowa notebooka
SETUP_IDS = ["06abfd40", "085c634d", "accae549"]        # importy, konfiguracja, wczytanie zdań (df)
INTRO_NOTE = ("**Uśrednione Word2Vec (pkt 1b).** Dwa modele Word2Vec z części Word Embeddings (IPI PAN i PolEmo) "
              "mają osobne podsekcje: **1.4** (osadzenia), **2.1** (t-SNE) i **3.7** (analiza i odpowiedzi "
              "na pytania 1–5).")

# --------------------------------------------------------------------------
# Komórki: (id komórki zespołu, po której wstawiamy, [(typ, id, źródło), ...])
# --------------------------------------------------------------------------
SEC_1 = [
    ("markdown", "w2v-1-4-md", r"""### 1.4 Uśrednione wektory Word2Vec (pkt 1b)

> W `Text_Embeddings_executed.ipynb` komórki Word2Vec (1.4, 2.1 i 3.7) wykonano osobno, lokalnie 21.10.2026
> (`python -m src.notebook_w2v`). Pozostałe komórki i ich wyniki pochodzą z uruchomienia zespołu (USE z TF Hub)
> i nie były zmieniane.

Wektor zdania to **średnia wektorów jego słów** (małe litery, bez interpunkcji). Słowa bez wektora są pomijane,
dlatego podajemy pokrycie słownika. Dwa modele z części Word Embeddings, oba 100-wymiarowe:
* **Word2Vec IPI PAN** (`w2v`) – ogólny model z NKJP i Wikipedii (formy słów, skip-gram), 300 tys. najczęstszych form;
* **Word2Vec PolEmo** (`w2v_polemo`) – model wytrenowany w gensim na zdaniach PolEmo (recenzje, model dziedzinowy).

Osadzenia wczytują się z cache `results/text_embeddings/emb_<model>.npy`. Z tych samych osadzeń pochodzą liczby
w sprawozdaniu (podrozdział „Uśrednione wektory Word2Vec”)."""),
    ("code", "w2v-1-4-emb", r"""W2V_MODELS = ["w2v", "w2v_polemo"]
W2V_TITLES = {"w2v": "Word2Vec IPI PAN", "w2v_polemo": "Word2Vec PolEmo"}
emb_w2v, E_w2v, cov_w2v = {}, {}, {}
for n in W2V_MODELS:
    m = te.get_embedder(n)
    E_w2v[n] = te.get_embeddings(m, df, force=FORCE)
    # ponowne kodowanie sprawdza cache i daje pokrycie słownika dla każdego zdania (w cache go nie ma)
    assert np.allclose(m.encode(df["text"].tolist()), E_w2v[n], atol=1e-5)
    emb_w2v[n], cov_w2v[n] = m, m.last_coverage
    print(f"OK  {n:10s} {E_w2v[n].shape}  {m.label}")

display(pd.DataFrame({
    "wymiar": [E_w2v[n].shape[1] for n in W2V_MODELS],
    "pokrycie słów: średnio": [f"{cov_w2v[n].mean():.1%}" for n in W2V_MODELS],
    "pokrycie słów: najmniej": [f"{cov_w2v[n].min():.0%}" for n in W2V_MODELS],
    "zdania bez znanego słowa": [int((cov_w2v[n] == 0).sum()) for n in W2V_MODELS],
}, index=pd.Index([W2V_TITLES[n] for n in W2V_MODELS], name="model")))"""),
    ("code", "w2v-1-4-preview", r"""for n in W2V_MODELS:
    print(f"\n=== {emb_w2v[n].label} ===")
    print(f"macierz: {E_w2v[n].shape[0]} zdań × {E_w2v[n].shape[1]} wymiarów, dtype {E_w2v[n].dtype}, "
          f"średnia norma wektora: {np.linalg.norm(E_w2v[n], axis=1).mean():.2f}")
    display(te.embedding_preview(df, E_w2v[n], n_rows=5, n_dims=8))
print("\npełne osadzenia: results/text_embeddings/embeddings_w2v.csv, embeddings_w2v_polemo.csv")"""),
    ("markdown", "w2v-1-4-heat-md", r"""**Podobieństwo kosinusowe tych samych zdań co w 1.3** (po dwa z każdej klasy gold). Tak jak w fastText,
wszystkie zdania są do siebie podobne: poza przekątną cosinus wynosi średnio 0,66 (IPI PAN) i 0,61 (PolEmo),
od 0,39 do 0,84. Wysoki cosinus nie oznacza więc podobnego wydźwięku; sens ma tylko porównanie względne."""),
    ("code", "w2v-1-4-heat", r"""sel = (df[df["iter"] == df["iter"].min()].groupby("gold", sort=False).head(2)).sort_values("gold").index  # jak w 1.3
for n in W2V_MODELS:
    te.similarity_heatmap(df, E_w2v[n], sel, f"Podobieństwo kosinusowe – {emb_w2v[n].label}").show()"""),
]

SEC_2 = [
    ("markdown", "w2v-2-1-md", r"""### 2.1 t-SNE: uśrednione Word2Vec

Ta sama funkcja wykresu i te same ustawienia t-SNE co wyżej: **po najechaniu na punkt widać pełne zdanie, etykiety
gold, A1, A2 i final**, iterację i `sent_id`, a lista rozwijana nad wykresem zmienia kolorowanie. Pliki HTML tych
wykresów są już w `results/text_embeddings/tsne_w2v.html` i `tsne_w2v_polemo.html`
(`python -m src.text_embeddings --models w2v w2v_polemo --evaluate`), więc ta komórka niczego nie zapisuje."""),
    ("code", "w2v-2-1-tsne", r"""xy_w2v = {}
for n in W2V_MODELS:
    xy_w2v[n] = te.tsne_2d(E_w2v[n])
    te.make_tsne_figure(df, xy_w2v[n], f"t-SNE – {emb_w2v[n].label}").show()"""),
    ("markdown", "w2v-2-1-obs", r"""**Co widać.** Klasy wydźwięku są wymieszane, tak jak dla fastText i USE: zdania POZ i NEG leżą obok siebie,
w tych samych skupiskach tematycznych. Wyróżnia się tylko część zdań NEU, które tworzą osobne skupisko, wyraźniej
w modelu PolEmo. Liczbowo (3.7.5): wśród 10 najbliższych sąsiadów zdania ta sama klasa `final` to 33% (IPI PAN)
i 36% (PolEmo), przy 30,5% losowo."""),
]

SEC_3 = [
    ("markdown", "w2v-3-7-md", r"""### 3.7 Word2Vec: analiza osadzeń (pyt. 1–5)

Te same funkcje, pary kontrolne i miary co w 3.1–3.5, tym razem dla uśrednionego Word2Vec. Dochodzą dwie miary:
* pyt. 1: udział par POZ–NEG wśród 1% najbardziej podobnych par zdań,
* pyt. 2: zależność podobieństwa od wspólnych słów (Spearman: kosinus wobec miary Jaccarda zbiorów słów).

W tabelach porównawczych fastText, USE i TF-IDF to wyniki zespołu z sekcji 3.2–3.5, wczytane z tabel sprawozdania
(`report/generated/emb_*.tex`, funkcje `team_*` w `src/text_analysis.py`). USE z TF Hub nie działa w Windows,
więc tych modeli nie liczymy tu od nowa (wyjątek: dwie miary korpusowe w 3.7.6). Zestawienie pięciu modeli jest
w 3.7.6, a odpowiedzi na pytania, z porównaniem Word2Vec z innymi metodami, w 3.7.7."""),
    ("markdown", "w2v-3-7-1-md", r"""#### 3.7.1 Pyt. 1: zdania uznane za podobne, a różne znaczeniowo
Najbardziej podobne pary zdań o przeciwnym wydźwięku (POZ i NEG według etykiety `final`) oraz udział par POZ–NEG
wśród 1% najbardziej podobnych par zdań, w porównaniu z udziałem wśród wszystkich par."""),
    ("code", "w2v-3-7-1", r"""from src import text_analysis as ta

sims_w2v = {n: te.cosine_matrix(E_w2v[n]) for n in W2V_MODELS}
for n in W2V_MODELS:
    print(f"\n=== {emb_w2v[n].label} ===")
    display(ta.top_pairs(df, sims_w2v[n], k=8, opposite_only=True))

top_share = pd.DataFrame.from_dict(
    {W2V_TITLES[n]: ta.opposite_share_top(sims_w2v[n], df["final"].to_numpy()) for n in W2V_MODELS}, orient="index")
display(top_share.rename(columns={
    "n_top": "par w górnym 1%", "top_opposite": "POZ–NEG: górny 1%", "all_opposite": "POZ–NEG: wszystkie pary",
    "top_same": "ta sama etykieta: górny 1%", "all_same": "ta sama etykieta: wszystkie pary"}).round(3))"""),
    ("markdown", "w2v-3-7-2-md", r"""#### 3.7.2 Pyt. 2: zdania podobne znaczeniowo, a uznane za odległe (inne słowa)
**(a)** Jak bardzo podobieństwo zależy od wspólnych słów: korelacja Spearmana kosinusa z miarą Jaccarda zbiorów słów
(wszystkie pary zdań) i średni kosinus w grupach par. **(b)** Pary, które jeden model uznaje za bliskie, a drugi
za odległe (rangi percentylowe, jak w 3.2; tu IPI PAN wobec PolEmo). To kandydaci do ręcznej oceny.
**(c)** Pary kontrolne z 3.2 dla Word2Vec i średnie dla wszystkich pięciu modeli."""),
    ("code", "w2v-3-7-2", r"""J_words = ta.jaccard_matrix(df["text"].tolist())
lex_w2v = {n: ta.lexical_dependence(sims_w2v[n], J_words) for n in W2V_MODELS}
display(pd.DataFrame.from_dict(
    {W2V_TITLES[n]: {"Spearman (cos vs Jaccard)": q["spearman_cos_jaccard"],
                     **{f"śr. cos: {k}": v for k, v in q["mean_cos"].items()}} for n, q in lex_w2v.items()},
    orient="index").round(3))
print("liczba par w grupach:", lex_w2v[W2V_MODELS[0]]["n_pairs"])

close_ipi, close_pol = te.cross_model_disagreement(df, sims_w2v["w2v"], sims_w2v["w2v_polemo"],
                                                   "w2v", "w2v_polemo", k=8)
print("\nBlisko według IPI PAN, daleko według PolEmo:"); display(close_ipi)
print("Blisko według PolEmo, daleko według IPI PAN:"); display(close_pol)"""),
    ("code", "w2v-3-7-2-probes", r"""with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    probes_w2v = te.probe_table(emb_w2v)
display(probes_w2v)
print("Średnie cos. wg kategorii (fastText, USE, TF-IDF: wyniki zespołu z 3.2):")
display(probes_w2v.groupby("kategoria", sort=False)[W2V_MODELS].mean().rename(columns=W2V_TITLES)
        .join(ta.team_probe_means()).round(3))"""),
    ("markdown", "w2v-3-7-3-md", r"""#### 3.7.3 Pyt. 3: różna długość, podobny sens
Kategoria „ten sam sens, różna długość” z par kontrolnych (porównaj z parafrazami i z kontrolą niepowiązaną
w tabeli powyżej)."""),
    ("code", "w2v-3-7-3", r"""len_pairs_w2v = probes_w2v[probes_w2v["kategoria"].str.contains("różna długość")].copy()
len_pairs_w2v["dł_a"] = len_pairs_w2v["zdanie_a"].str.split().str.len()
len_pairs_w2v["dł_b"] = len_pairs_w2v["zdanie_b"].str.split().str.len()
display(len_pairs_w2v[["zdanie_a", "zdanie_b", "dł_a", "dł_b"] + W2V_MODELS])"""),
    ("markdown", "w2v-3-7-4-md", r"""#### 3.7.4 Pyt. 4: długość zdania a podobieństwo i „wyrazistość” wektora
Te same kolumny co w 3.4 (opis pod tabelami w 3.4). Wykres jest interaktywny: po najechaniu na punkt widać zdanie
i jego etykietę `final`."""),
    ("code", "w2v-3-7-4", r"""for n in W2V_MODELS:
    print(f"\n=== {emb_w2v[n].label} ===")
    display(te.length_stats(df, E_w2v[n]))
team_len = ta.team_length()
print("Spearman(n_tokens, śr. cos) w modelach zespołu (3.4):")
display(team_len.loc[team_len["grupa"] == "Spearman", ["model", "spearman"]].set_index("model"))

fig = go.Figure()
for n in W2V_MODELS:
    S = sims_w2v[n].copy(); np.fill_diagonal(S, np.nan)
    fig.add_trace(go.Scatter(x=df["n_tokens"], y=np.nanmean(S, axis=1), mode="markers", name=W2V_TITLES[n],
                             opacity=0.6, text=df["text"].str.wrap(70).str.replace("\n", "<br>"),
                             customdata=df["final"].fillna("brak"),
                             hovertemplate="%{text}<br>final: %{customdata} | tokenów: %{x}, śr. cos: %{y:.3f}"
                                           "<extra>" + W2V_TITLES[n] + "</extra>"))
fig.update_layout(title="Długość zdania vs średnie podobieństwo do pozostałych zdań – uśrednione Word2Vec",
                  template="plotly_white", xaxis_title="liczba tokenów", yaxis_title="średni cosinus do innych zdań",
                  height=520)
fig.show()"""),
    ("markdown", "w2v-3-7-5-md", r"""#### 3.7.5 Pyt. 5: czy osadzenia nadają się do zadania (wydźwięk zdania)?
Te same miary co w 3.5: regresja logistyczna na samych osadzeniach (5-krotna walidacja krzyżowa warstwowa), czystość
10 najbliższych sąsiadów wobec poziomu losowego i silhouette, dla każdego źródła etykiet. Dalej porównanie pięciu
modeli, macierze pomyłek i czułość dla każdej klasy (etykiety `final`)."""),
    ("code", "w2v-3-7-5", r"""TEAM_NAMES = {"fasttext": "fastText", "use": "USE", "tfidf": "TF-IDF"}
ev_w2v = pd.concat({n: te.evaluation_table(df, E_w2v[n]).set_index("etykiety") for n in W2V_MODELS}, names=["model"])
display(ev_w2v)

cols = {"accuracy": "accuracy", "baza_wiekszosciowa": "baza", "macro_F1": "macro_F1",
        "kNN10_czystosc": "kNN10", "losowy_wg_rozkladu": "losowy", "silhouette_cos": "silhouette"}
ours = ev_w2v.xs("final", level="etykiety")[list(cols)].rename(columns=cols, index=W2V_TITLES)
team_cls = ta.team_classif("final").set_index("model").rename(index=TEAM_NAMES)
print("Etykiety final, pięć modeli (fastText, USE, TF-IDF: wyniki zespołu z 3.5):")
display(pd.concat([ours, team_cls]))"""),
    ("code", "w2v-3-7-5-cm", r"""recall_w2v = {}
for n in W2V_MODELS:
    cm, rep = te.confusion_report(E_w2v[n], df["final"])
    print(f"\n=== {emb_w2v[n].label} ===")
    display(cm); display(rep)
    recall_w2v[W2V_TITLES[n]] = rep.loc[te.config.SENT_LABELS, "recall"]
print("Czułość (recall) dla każdej klasy, etykiety final (fastText, USE, TF-IDF: wyniki zespołu z 3.5):")
display(pd.concat([pd.DataFrame(recall_w2v).T, ta.team_recall().rename(index=TEAM_NAMES)]).round(3))"""),
    ("markdown", "w2v-3-7-6-md", r"""#### 3.7.6 Zestawienie: Word2Vec na tle fastText, USE i TF-IDF
Najważniejsze liczby z 3.1–3.5 i 3.7.1–3.7.5 w jednej tabeli, pogrupowane według pytań. Cosinusy różnych modeli
mają różne skale, więc dochodzą dwie miary względne z par kontrolnych: **pozycja na skali**, na której 0 to średnia
par niepowiązanych, a 1 to średnia parafraz (powyżej 1 = bliżej niż parafrazy), oraz różnica parafraza − niepowiązane.
Dwie miary korpusowe (udział POZ–NEG wśród 1% najbliższych par, Spearman cos vs Jaccard) liczymy tu także dla
fastText i TF-IDF. fastText to te same osadzenia CBOW co u zespołu (zapisane przy lokalnym uruchomieniu w
`results/text_embeddings_local/`; jego miary klasyfikacji są identyczne jak w 3.5), a TF-IDF liczy się od nowa.
Dla USE tych dwóch miar brak („–”), bo osadzeń z TF Hub nie ma lokalnie."""),
    ("code", "w2v-3-7-6", r"""# fastText: osadzenia CBOW zespołu zapisane przy lokalnym uruchomieniu (model nie jest wczytywany); TF-IDF od nowa
E_cmp = {**E_w2v,
         "fasttext": te.get_embeddings(te.get_embedder("fasttext"), df,
                                       out_dir=te.config.RESULTS_DIR / "text_embeddings_local"),
         "tfidf": te.get_embedder("tfidf").encode(df["text"].tolist())}
NAMES = {**W2V_TITLES, **TEAM_NAMES}
cols = [NAMES[n] for n in ["w2v", "w2v_polemo", "fasttext", "use", "tfidf"]]

pm = (probes_w2v.groupby("kategoria", sort=False)[W2V_MODELS].mean().rename(columns=W2V_TITLES)
      .join(ta.team_probe_means())[cols])
para, unrel = pm.loc["parafraza"], pm.loc["kontrola: niepowiązane"]
on_scale = lambda cat: (pm.loc[cat] - unrel) / (para - unrel)   # 0 = jak zdania niepowiązane, 1 = jak parafraza

corpus = {}
for n, e in E_cmp.items():
    S = te.cosine_matrix(e)
    corpus[NAMES[n]] = {"top": ta.opposite_share_top(S, df["final"].to_numpy())["top_opposite"],
                        "jac": ta.lexical_dependence(S, J_words)["spearman_cos_jaccard"]}
corpus = pd.DataFrame(corpus).reindex(columns=cols)

def length_row(short, long, spearman):
    return {"cos_short": short["cos"], "cos_long": long["cos"], "norm": long["norma"] / short["norma"], "rho": spearman}
lens = {}
for n in W2V_MODELS:
    ls = te.length_stats(df, E_w2v[n]).rename(columns={"cos_do_centroidu": "cos", "norma_wektora": "norma"})
    lens[W2V_TITLES[n]] = length_row(ls.iloc[0], ls.iloc[2], ls.iloc[3]["srednie_cos_do_innych"])
for m in ["fastText", "USE", "TF-IDF"]:
    t = team_len[team_len["model"] == m].rename(columns={"cos_centroid": "cos"})
    lens[m] = length_row(t.iloc[0], t.iloc[2], t.iloc[3]["spearman"])
lens = pd.DataFrame(lens)[cols]

cls = {}
for lab in ["final", "gold"]:
    ours = ev_w2v.xs(lab, level="etykiety").rename(index=W2V_TITLES).rename(
        columns={"kNN10_czystosc": "kNN10", "silhouette_cos": "silhouette"})
    team_lab = ta.team_classif(lab).set_index("model").rename(index=TEAM_NAMES)
    cls[lab] = pd.concat([ours, team_lab])[["accuracy", "macro_F1", "kNN10", "silhouette"]].T[cols]
rec = pd.concat([pd.DataFrame(recall_w2v).T, ta.team_recall().rename(index=TEAM_NAMES)]).T[cols]

Q = ["1. podobne, a różne znaczeniowo", "2. podobne znaczeniowo, inne słowa", "3. różna długość, podobny sens",
     "4. długość a wyrazistość wektora", "5. przydatność do wydźwięku"]
cmp_all = pd.DataFrame({
    (Q[0], "cos: parafraza"): para,
    (Q[0], "cos: przeciwny sens, podobne słowa"): pm.loc["przeciwny sens, podobne słowa"],
    (Q[0], "cos: przeciwny sens, te same słowa"): pm.loc["przeciwny sens, te same słowa"],
    (Q[0], "przeciwny sens na skali (0 = niepowiązane, 1 = parafraza)"): on_scale("przeciwny sens, podobne słowa"),
    (Q[0], "POZ–NEG wśród 1% najbliższych par (wszystkie pary: 0,125)"): corpus.loc["top"],
    (Q[1], "cos: niepowiązane"): unrel,
    (Q[1], "parafraza − niepowiązane"): para - unrel,
    (Q[1], "Spearman: cos vs wspólne słowa (Jaccard)"): corpus.loc["jac"],
    (Q[2], "cos: ten sam sens, różna długość"): pm.loc["ten sam sens, różna długość"],
    (Q[2], "różna długość na skali (0 = niepowiązane, 1 = parafraza)"): on_scale("ten sam sens, różna długość"),
    (Q[3], "cos do centroidu: krótkie zdania"): lens.loc["cos_short"],
    (Q[3], "cos do centroidu: długie zdania"): lens.loc["cos_long"],
    (Q[3], "norma wektora: długie / krótkie"): lens.loc["norm"],
    (Q[3], "Spearman: liczba tokenów vs śr. cos"): lens.loc["rho"],
    (Q[4], "accuracy, final (baza 0,415)"): cls["final"].loc["accuracy"],
    (Q[4], "macro-F1, final"): cls["final"].loc["macro_F1"],
    (Q[4], "accuracy, gold (baza 0,25)"): cls["gold"].loc["accuracy"],
    (Q[4], "macro-F1, gold"): cls["gold"].loc["macro_F1"],
    (Q[4], "kNN-10, final (losowo 0,305)"): cls["final"].loc["kNN10"],
    (Q[4], "silhouette, final"): cls["final"].loc["silhouette"],
    (Q[4], "czułość POZ"): rec.loc["POZ"],
    (Q[4], "czułość NEG"): rec.loc["NEG"],
    (Q[4], "czułość NEU"): rec.loc["NEU"],
}).T.astype(float)
cmp_all.index.names = ["pytanie", "miara"]
display(cmp_all.map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"))"""),
]

ANSWERS_ID = "w2v-3-7-7"
ANSWERS_HEADER = """#### 3.7.7 Odpowiedzi na pytania pomocnicze: Word2Vec w porównaniu z innymi metodami

Każda odpowiedź opiera się na konkretnych przykładach i liczbach z sekcji 3.7.1–3.7.6 (Word2Vec, zestawienie) oraz
3.1–3.5 (fastText, USE i TF-IDF, wyniki zespołu). Przy dwóch liczbach dla Word2Vec pierwsza dotyczy modelu IPI PAN,
druga PolEmo. 200 zdań i 12 par kontrolnych to ilustracja, a nie statystyka. Ten sam tekst jest
w `results/text_embeddings/odpowiedzi_porownanie.md`."""

# komórka kontrolna – tylko w uruchomieniu skryptu, nie trafia do notebooków
CHECK_CODE = r"""import json
ref = json.loads((te.OUT_DIR / "analiza_w2v.json").read_text(encoding="utf-8"))
for n in W2V_MODELS:
    ev = pd.read_csv(te.OUT_DIR / f"eval_{n}.csv", encoding="utf-8-sig").set_index("etykiety")
    assert np.allclose(ev_w2v.loc[n].loc[ev.index, ev.columns].to_numpy(float), ev.to_numpy(float)), f"eval {n}"
    ts = pd.read_csv(te.OUT_DIR / f"tsne_{n}.csv")
    assert (ts["sent_id"].astype(str).to_numpy() == df["sent_id"].astype(str).to_numpy()).all(), f"sent_id {n}"
    assert np.allclose(ts[["x", "y"]].to_numpy(), xy_w2v[n], atol=1e-3), f"t-SNE {n}"
    pm = probes_w2v.groupby("kategoria", sort=False)[n].mean()
    assert np.allclose(pm.to_numpy(), [ref["probe_means"][n][k] for k in pm.index], atol=1e-4), f"pary {n}"
    o = ta.opposite_share_top(sims_w2v[n], df["final"].to_numpy())
    assert all(np.isclose(o[k], ref["q1_opposite_top"][n][k]) for k in o), f"pyt. 1 {n}"
    assert np.isclose(lex_w2v[n]["spearman_cos_jaccard"], ref["q2_lexical"][n]["spearman_cos_jaccard"]), f"pyt. 2 {n}"
    rec = pd.Series(ref["q5_recall"][n])[te.config.SENT_LABELS].to_numpy(float)
    assert np.allclose(recall_w2v[W2V_TITLES[n]].to_numpy(float), rec), f"recall {n}"
# fastText (osadzenia z lokalnego uruchomienia) i TF-IDF (od nowa) = te same miary co u zespołu w 3.5
team_f = ta.team_classif("final").set_index("model")
for n in ["fasttext", "tfidf"]:
    r = te.evaluate_embeddings(E_cmp[n], df["final"])
    got = [r["accuracy"], r["macro_F1"], r["kNN10_czystosc"], r["silhouette_cos"]]
    assert np.allclose(got, team_f.loc[n, ["accuracy", "macro_F1", "kNN10", "silhouette"]].to_numpy(float)), n
print("ZGODNE z results/text_embeddings")"""


def answers_markdown() -> str:
    """Odpowiedzi z ``odpowiedzi_porownanie.md`` (jedno źródło tekstu), bez tytułu i wstępu pliku."""
    text = ANSWERS_MD.read_text(encoding="utf-8")
    return ANSWERS_HEADER + "\n\n" + text[text.index("**1. "):].strip()


def w2v_cells() -> list[tuple[str, list]]:
    """[(id komórki zespołu, po której wstawić, [komórki])]."""
    import nbformat

    def mk(kind, cid, src):
        new = nbformat.v4.new_code_cell if kind == "code" else nbformat.v4.new_markdown_cell
        return new(src, id=cid, metadata={"tags": [TAG]})

    sec3 = [mk(*c) for c in SEC_3] + [mk("markdown", ANSWERS_ID, answers_markdown())]
    return [("fbc01d64", [mk(*c) for c in SEC_1]),      # po mapach podobieństwa (1.3)
            ("8287da36", [mk(*c) for c in SEC_2]),      # po „Jak czytać wykresy” (2)
            ("6e410e5f", sec3)]                         # po odpowiedziach zespołu (3.6)


# --------------------------------------------------------------------------
# Wstawianie / usuwanie
# --------------------------------------------------------------------------
def is_w2v(cell) -> bool:
    return TAG in cell.get("metadata", {}).get("tags", [])


def strip_w2v(nb):
    """Notebook bez komórek Word2Vec i bez dopisku we wstępie (stan sprzed ``insert_w2v``)."""
    nb = copy.deepcopy(nb)
    nb.cells = [c for c in nb.cells if not is_w2v(c)]
    for c in nb.cells:
        if c.get("id") == INTRO_ID:
            c.source = c.source.replace("\n\n" + INTRO_NOTE, "")
    return nb


def insert_w2v(nb):
    """Wstawia komórki Word2Vec (bez wyników); wcześniejsze komórki ``w2v`` są najpierw usuwane."""
    nb = strip_w2v(nb)
    for anchor, cells in w2v_cells():
        pos = next(i for i, c in enumerate(nb.cells) if c.get("id") == anchor)
        nb.cells[pos + 1:pos + 1] = cells
    intro = next(c for c in nb.cells if c.get("id") == INTRO_ID)
    intro.source = intro.source.rstrip("\n") + "\n\n" + INTRO_NOTE
    return nb


def _dump(nb) -> str:
    return json.dumps(nb, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------
# Uruchomienie samych komórek Word2Vec
# --------------------------------------------------------------------------
def run_w2v_cells(nb, timeout: int = 1800) -> dict[str, list]:
    """Wykonuje komórki startowe zespołu + komórki Word2Vec + komórkę kontrolną. Zwraca {id: wyniki}."""
    import nbformat
    from nbclient import NotebookClient

    by_id = {c.get("id"): c for c in nb.cells}
    run = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
    run.cells = [copy.deepcopy(by_id[i]) for i in SETUP_IDS]
    run.cells += [copy.deepcopy(c) for c in nb.cells if is_w2v(c) and c.cell_type == "code"]
    run.cells.append(nbformat.v4.new_code_cell(CHECK_CODE, id="w2v-check"))
    NotebookClient(run, timeout=timeout, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    outs = {c.id: c.outputs for c in run.cells}
    errors = [c.id for c in run.cells for o in c.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError(f"błędy w komórkach: {errors}")
    check = "".join(o.get("text", "") for o in outs["w2v-check"])
    if "ZGODNE" not in check:
        raise RuntimeError(f"komórka kontrolna: {check!r}")
    print(check.strip())
    return outs


def fill_outputs(nb, outs: dict[str, list]):
    """Wpisuje wyniki do komórek Word2Vec; numery wykonania kontynuują numerację zespołu."""
    team = [c for c in nb.cells if c.cell_type == "code" and not is_w2v(c)]
    k = max((c.get("execution_count") or 0) for c in team)
    for c in nb.cells:
        if is_w2v(c) and c.cell_type == "code":
            k += 1
            c.outputs = copy.deepcopy(outs[c.id])
            c.execution_count = k
            for o in c.outputs:
                if "execution_count" in o:
                    o["execution_count"] = k
    return nb


def export_html(nb, path) -> None:
    from nbconvert import HTMLExporter

    body, _ = HTMLExporter().from_notebook_node(nb, resources={"metadata": {"name": path.stem}})
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import nbformat

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-html", action="store_true", help="bez eksportu Text_Embeddings_executed.html")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # konsola Windows (cp1250)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    src = insert_w2v(nbformat.read(SOURCE_NB, as_version=4))
    nbformat.validate(src)
    nbformat.write(src, SOURCE_NB)
    print("komórki Word2Vec w", SOURCE_NB.name)

    executed = nbformat.read(EXECUTED_NB, as_version=4)
    team_before = _dump(strip_w2v(executed))
    executed = fill_outputs(insert_w2v(executed), run_w2v_cells(src))
    if _dump(strip_w2v(executed)) != team_before:
        raise RuntimeError("zmieniły się komórki zespołu – nie zapisuję")
    nbformat.validate(executed)
    nbformat.write(executed, EXECUTED_NB)
    print("wyniki Word2Vec w", EXECUTED_NB.name)
    if not a.no_html:
        export_html(executed, EXECUTED_NB.with_suffix(".html"))
        print("eksport:", EXECUTED_NB.with_suffix(".html").name)


if __name__ == "__main__":
    main()
