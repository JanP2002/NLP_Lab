"""Zadanie 1, Część II – Text Embeddings (2 pkt), korpus PolEmo 2.0.

Plik odpowiada za trzy podpunkty zadania:

* pkt 1 (0,5 pkt) – osadzenia zdań: ``get_embedder`` / ``get_embeddings``
  (fastText i Word2Vec uśredniane po tokenach – Word2Vec w dwóch wariantach z części
  Word Embeddings: IPI PAN i wytrenowany na PolEmo –, Universal Sentence Encoder,
  dodatkowo TF-IDF+SVD jako punkt odniesienia bez pobierania modeli),
* pkt 2 (0,5 pkt) – t-SNE + interaktywny wykres Plotly: ``tsne_2d``,
  ``make_tsne_figure``,
* pkt 3 (1 pkt) – analiza do dyskusji: ``similar_but_different``,
  ``cross_model_disagreement``, ``probe_table``, ``length_stats``,
  ``evaluate_embeddings``.

Uruchomienie z linii poleceń (pkt 1 i 2, liczy i zapisuje wyniki):

    python -m src.text_embeddings --models fasttext use
    python -m src.text_embeddings --models w2v w2v_polemo # uśrednione Word2Vec (modele z models/)
    python -m src.text_embeddings --models tfidf          # test bez pobierania modeli
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import config

OUT_DIR = config.RESULTS_DIR / "text_embeddings"
MODEL_NAMES = ["fasttext", "w2v", "w2v_polemo", "use", "tfidf"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
FASTTEXT_REPO = "clarin-pl/fastText-kgr10"
USE_TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
USE_FALLBACK = "sentence-transformers/distiluse-base-multilingual-cased-v1"

LABEL_COLORS = {
    "POZ": "#1b9e77",
    "NEG": "#d95f02",
    "NEU": "#7570b3",
    "AMB": "#e6ab02",
    "brak": "#bdbdbd",
}


# --------------------------------------------------------------------------
# Dane
# --------------------------------------------------------------------------
def load_sentences(iterations: Iterable[int] = (1, 2)) -> pd.DataFrame:
    """Zdania z próbek iteracji + etykiety.

    Kolumny: sent_id, iter, text, n_tokens, gold (etykieta PolEmo), A1, A2
    (anotacje, tylko tam, gdzie jest eksport), final (rozstrzygnięcie z
    ``annotation/iterN/adjudicated.csv``, jeśli plik istnieje – inaczej NaN).
    Nie czyta parquetu, więc działa bez pyarrow.
    """
    parts = []
    for it in iterations:
        d = config.iter_dir(it)
        sample = pd.read_csv(d / "sample.csv", encoding="utf-8-sig")
        gold = pd.read_csv(d / "_key_gold.csv", encoding="utf-8-sig")[["sent_id", "gold"]]
        df = sample.merge(gold, on="sent_id", how="left")
        df["iter"] = it

        wide = config.export_dir(it) / "sentences_wide.csv"
        if wide.exists():
            w = pd.read_csv(wide, encoding="utf-8-sig")[["sent_id", "A1", "A2"]]
            df = df.merge(w, on="sent_id", how="left")
        else:
            df["A1"] = np.nan
            df["A2"] = np.nan

        adj = d / "adjudicated.csv"
        if adj.exists():
            a = pd.read_csv(adj, encoding="utf-8-sig")
            cands = ("final", "label", "adjudicated", "sent_label", "etykieta", "superanotator", "A3")
            col = next((c for c in cands if c in a.columns), None)
            if col is None:
                raise ValueError(f"{adj}: brak kolumny z etykietą (oczekiwano jednej z {cands}), "
                                 f"jest: {list(a.columns)}")
            vals = a[col].where(a[col].isna(), a[col].astype(str).str.strip().str.upper())
            bad = sorted(set(vals.dropna()) - set(config.SENT_LABELS))
            if bad:
                raise ValueError(f"{adj}: niedozwolone etykiety {bad}; dozwolone: {config.SENT_LABELS}")
            a = pd.DataFrame({"sent_id": a["sent_id"], "final": vals}).drop_duplicates("sent_id")
            df = df.merge(a, on="sent_id", how="left")
        if "final" not in df.columns:
            df["final"] = np.nan
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    cols = ["sent_id", "iter", "text", "n_tokens", "gold", "A1", "A2", "final"]
    return out[cols]


def label_columns(df: pd.DataFrame) -> list[str]:
    """Kolumny z etykietami, które mają jakiekolwiek wartości (do kolorowania)."""
    # `final` (uzgodniona anotacja) jako pierwsza = domyślny kolor wykresów
    return [c for c in ("final", "gold", "A1", "A2") if c in df and df[c].notna().any()]


def tokens(text: str) -> list[str]:
    """Tokeny słowne (bez interpunkcji), małe litery."""
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Modele osadzeń zdań
# --------------------------------------------------------------------------
class Embedder:
    """Interfejs: ``encode(list[str]) -> ndarray (n, d)`` (surowe, nieznormalizowane)."""

    name: str = "base"
    label: str = "base"      # opis do wykresów i raportu
    cacheable: bool = True

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class TfidfSvdEmbedder(Embedder):
    """Bag-of-Words/TF-IDF + SVD. Punkt odniesienia, nie wymaga żadnych pobrań.

    Dopasowuje się przy pierwszym ``encode`` (na zdaniach korpusu), potem
    tylko transformuje – dzięki temu zdania testowe trafiają do tej samej przestrzeni.
    """

    name = "tfidf"
    label = "TF-IDF + SVD (punkt odniesienia)"
    cacheable = False

    def __init__(self, dim: int = 100):
        self.dim = dim
        self._vec = None
        self._svd = None

    def encode(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = list(texts)
        if self._vec is None:
            self._vec = TfidfVectorizer(tokenizer=tokens, lowercase=False,
                                        token_pattern=None, ngram_range=(1, 2),
                                        sublinear_tf=True, min_df=1)
            X = self._vec.fit_transform(texts)
            k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=k, random_state=0)
            return self._svd.fit_transform(X)
        return self._svd.transform(self._vec.transform(texts))


class AvgVectorEmbedder(Embedder):
    """Średnia wektorów słów (fastText, Word2Vec) po tokenach zdania.

    ``loader`` zwraca obiekt gensim ``KeyedVectors``; ładowany leniwie przy
    pierwszym użyciu. Dla fastText nieznane słowa dostają wektor z n-gramów
    znakowych (OOV nie jest problemem), dla Word2Vec są pomijane.
    """

    def __init__(self, name: str, label: str, loader: Callable[[], object],
                 subword: bool):
        self.name, self.label, self._loader, self.subword = name, label, loader, subword
        self._kv = None
        self.last_coverage: np.ndarray | None = None

    @property
    def kv(self):
        if self._kv is None:
            self._kv = self._loader()
        return self._kv

    def _word_vec(self, tok: str, original: str):
        kv = self.kv
        if self.subword:
            return kv[tok]
        for cand in (tok, original):
            if cand in kv.key_to_index:
                return kv[cand]
        return None

    def encode(self, texts):
        kv = self.kv
        dim = kv.vector_size
        out = np.zeros((len(texts), dim), dtype=np.float32)
        cov = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            originals = TOKEN_RE.findall(t)
            vecs, hit = [], 0
            for o in originals:
                tok = o.lower()
                v = self._word_vec(tok, o)
                if v is not None:
                    vecs.append(v)
                if tok in kv.key_to_index or o in kv.key_to_index:
                    hit += 1
            if vecs:
                out[i] = np.mean(vecs, axis=0)
            cov[i] = hit / max(1, len(originals))
        self.last_coverage = cov
        return out


class UseEmbedder(Embedder):
    """Universal Sentence Encoder (wielojęzyczny).

    backend="tfhub"     – oryginalny USE multilingual z TF Hub (wymaga
                          tensorflow, tensorflow_hub, tensorflow_text;
                          tensorflow_text nie ma kół dla Windows – użyj WSL/Colab),
    backend="distiluse" – ``distiluse-base-multilingual-cased-v1`` z
                          sentence-transformers (model destylowany z mUSE),
    backend="auto"      – tfhub, a gdy się nie uda, distiluse (z ostrzeżeniem).
    Faktycznie użyty backend jest w ``self.backend_used`` i ``self.label``.
    """

    name = "use"

    def __init__(self, backend: str = "auto", batch_size: int = 64):
        self.backend = backend
        self.batch_size = batch_size
        self.backend_used: str | None = None
        self.label = "Universal Sentence Encoder (multilingual)"
        self._model = None

    def _load(self):
        order = {"auto": ["tfhub", "distiluse"], "tfhub": ["tfhub"],
                 "distiluse": ["distiluse"]}[self.backend]
        errors = []
        for b in order:
            try:
                if b == "tfhub":
                    import tensorflow_hub as hub
                    import tensorflow_text  # noqa: F401  (rejestruje operacje modelu)
                    self._model = hub.load(USE_TFHUB_URL)
                    self.label = "Universal Sentence Encoder multilingual (TF Hub)"
                else:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(USE_FALLBACK)
                    self.label = "USE multilingual – distiluse-base-multilingual-cased-v1 (destylat mUSE)"
                self.backend_used = b
                return
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b}: {type(e).__name__}: {e}")
        raise RuntimeError("Nie udało się załadować USE:\n  " + "\n  ".join(errors))

    def encode(self, texts):
        if self._model is None:
            self._load()
            if self.backend_used == "distiluse":
                print("UWAGA: użyto distiluse (destylat mUSE), nie oryginalnego USE z TF Hub.")
        texts = list(texts)
        if self.backend_used == "tfhub":
            chunks = [self._model(texts[i:i + self.batch_size]).numpy()
                      for i in range(0, len(texts), self.batch_size)]
            return np.vstack(chunks).astype(np.float32)
        return np.asarray(self._model.encode(texts, batch_size=self.batch_size,
                                             show_progress_bar=False), dtype=np.float32)


def _load_fasttext(path: str | None):
    from gensim.models.fasttext import load_facebook_vectors

    if path is None:
        local = config.MODELS_DIR / config.FASTTEXT_FILE
        if local.exists():
            path = str(local)
        else:
            from huggingface_hub import hf_hub_download

            # model ma 3,7 GB; plik z uwag technicznych (skip-gram, 100 wymiarów)
            print("Pobieranie fastText:", config.FASTTEXT_FILE)
            path = hf_hub_download(FASTTEXT_REPO, config.FASTTEXT_FILE)
    return load_facebook_vectors(path)


def _load_w2v(path: str | None = None, limit: int | None = config.W2V["limit"]):
    """Word2Vec z dsmodels.nlp.ipipan.waw.pl (format word2vec .txt/.txt.gz/.bin albo gensim .kv).

    Bez ścieżki: zmienna środowiskowa W2V_PATH, a potem ``models/<config.W2V["file"]>``.
    ``limit`` = liczba najczęstszych słów (plik jest posortowany według częstości).
    Pierwsze wczytanie zapisuje obok pliku kopię ``<nazwa>_<limit>.kv``, którą kolejne
    wywołania czytają w ułamku sekundy.
    """
    from gensim.models import KeyedVectors

    p = Path(path or os.environ.get("W2V_PATH") or config.MODELS_DIR / config.W2V["file"])
    if not p.exists():
        raise FileNotFoundError(
            f"Brak modelu Word2Vec: {p}. Pobierz go z dsmodels.nlp.ipipan.waw.pl, np.\n"
            f"  curl -L -o \"{config.MODELS_DIR / config.W2V['file']}\" {config.W2V['url']}\n"
            "albo podaj ścieżkę: --w2v-path PLIK.")
    if p.suffix == ".kv":
        return KeyedVectors.load(str(p))
    cache = p.with_name(f"{p.name.split('.')[0]}_{limit or 'all'}.kv")
    if cache.exists():
        return KeyedVectors.load(str(cache))
    kv = KeyedVectors.load_word2vec_format(str(p), binary=p.suffix == ".bin", limit=limit)
    kv.save(str(cache))
    return kv


def get_embedder(name: str, fasttext_path: str | None = None,
                 w2v_path: str | None = None, use_backend: str = "auto") -> Embedder:
    """Fabryka modeli: 'fasttext', 'w2v', 'w2v_polemo', 'use', 'tfidf'. Modele ładują się leniwie.

    'w2v' = Word2Vec IPI PAN (``_load_w2v``: 300 tys. najczęstszych form, bez ścieżki z ``models/``),
    'w2v_polemo' = Word2Vec wytrenowany na zdaniach PolEmo (część Word Embeddings, ``models/w2v_polemo.kv``).
    """
    if name == "fasttext":
        return AvgVectorEmbedder("fasttext", "fastText kgr10 – średnia wektorów słów",
                                 lambda: _load_fasttext(fasttext_path), subword=True)
    if name == "w2v":
        return AvgVectorEmbedder("w2v", "Word2Vec (IPI PAN) – średnia wektorów słów",
                                 lambda: _load_w2v(w2v_path), subword=False)
    if name == "w2v_polemo":
        def load_polemo():
            from src.word_embeddings import train_polemo_w2v  # import leniwy: word_embeddings importuje ten moduł
            return train_polemo_w2v()
        return AvgVectorEmbedder("w2v_polemo", "Word2Vec (PolEmo) – średnia wektorów słów",
                                 load_polemo, subword=False)
    if name == "use":
        return UseEmbedder(use_backend)
    if name == "tfidf":
        return TfidfSvdEmbedder()
    raise ValueError(f"Nieznany model: {name!r} (dostępne: {', '.join(MODEL_NAMES)})")


def _fingerprint(df: pd.DataFrame) -> str:
    return hashlib.md5("\x1f".join(df["text"]).encode("utf-8")).hexdigest()


def get_embeddings(embedder: Embedder, df: pd.DataFrame, force: bool = False,
                   out_dir: Path = OUT_DIR) -> np.ndarray:
    """Osadzenia zdań z ``df`` – z cache (``results/text_embeddings/emb_<model>.npy``).

    Cache jest unieważniany, gdy zmieni się zestaw zdań (np. dojdzie iteracja 2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"emb_{embedder.name}.npy"
    meta = out_dir / f"emb_{embedder.name}.json"
    fp = _fingerprint(df)
    if embedder.cacheable and not force and npy.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("fingerprint") == fp:
            embedder.label = info.get("label", embedder.label)  # np. faktycznie użyty backend USE
            return np.load(npy)
    emb = embedder.encode(df["text"].tolist())
    if embedder.cacheable:
        np.save(npy, emb)
        info = {"fingerprint": fp, "n": len(df), "dim": int(emb.shape[1]), "label": embedder.label}
        if getattr(embedder, "last_coverage", None) is not None:
            info["mean_vocab_coverage"] = float(np.mean(embedder.last_coverage))
        meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return emb


# --------------------------------------------------------------------------
# Pkt 2 – t-SNE i wykres interaktywny
# --------------------------------------------------------------------------
def l2norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n == 0, 1.0, n)


def embedding_preview(df: pd.DataFrame, emb: np.ndarray, n_rows: int = 5, n_dims: int = 8,
                      text_chars: int = 55) -> pd.DataFrame:
    """Podgląd macierzy osadzeń: pierwsze ``n_rows`` zdań i pierwsze ``n_dims`` wymiarów wektora.

    Pojedyncze wymiary nie mają interpretacji – tabela tylko pokazuje, jak wygląda wektor.
    """
    out = pd.DataFrame({"sent_id": df["sent_id"].to_numpy()[:n_rows],
                        "zdanie": [t[:text_chars] + ("…" if len(t) > text_chars else "")
                                   for t in df["text"].to_numpy()[:n_rows]]})
    for k in range(min(n_dims, emb.shape[1])):
        out[f"d{k}"] = np.round(emb[:n_rows, k], 3)
    out[f"… (+{max(0, emb.shape[1] - n_dims)})"] = "…"
    out["norma"] = np.round(np.linalg.norm(emb[:n_rows], axis=1), 2)
    return out


def export_embeddings_csv(df: pd.DataFrame, emb: np.ndarray, name: str,
                          out_dir: Path = OUT_DIR) -> Path:
    """Zapisuje pełne osadzenia do CSV (sent_id, iter, gold + wszystkie wymiary d0..dN)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = df[["sent_id", "iter", "gold"]].reset_index(drop=True)
    dims = pd.DataFrame(emb, columns=[f"d{k}" for k in range(emb.shape[1])])
    path = out_dir / f"embeddings_{name}.csv"
    pd.concat([meta, dims], axis=1).to_csv(path, index=False, encoding="utf-8")
    return path


def similarity_heatmap(df: pd.DataFrame, emb: np.ndarray, idx, title: str, chars: int = 38):
    """Macierz podobieństwa kosinusowego wybranych zdań (hover: pełne zdania)."""
    import plotly.graph_objects as go

    idx = list(idx)
    S = cosine_matrix(emb[idx])
    short = [f"[{df['gold'].iloc[i]}] {df['text'].iloc[i][:chars]}…" for i in idx]
    full = np.array([["<br>".join(textwrap.wrap(df["text"].iloc[i], 60)) + "<br>vs<br>" +
                      "<br>".join(textwrap.wrap(df["text"].iloc[j], 60)) for j in idx] for i in idx])
    fig = go.Figure(go.Heatmap(z=np.round(S, 3), x=short, y=short, customdata=full, zmin=-0.2, zmax=1,
                               colorscale="Viridis", text=np.round(S, 2), texttemplate="%{text}",
                               hovertemplate="%{customdata}<br>cos = %{z}<extra></extra>"))
    fig.update_layout(title=title, template="plotly_white", height=620,
                      yaxis=dict(autorange="reversed"), margin=dict(l=260, b=260))
    return fig


def tsne_2d(emb: np.ndarray, perplexity: float | None = None, seed: int = 42) -> np.ndarray:
    """t-SNE (odległość kosinusowa, inicjalizacja PCA). Perplexity domyślnie min(30, (n-1)/3)."""
    from sklearn.manifold import TSNE

    n = len(emb)
    if perplexity is None:
        perplexity = max(2.0, min(30.0, (n - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
                learning_rate="auto", random_state=seed).fit_transform(emb)


def make_tsne_figure(df: pd.DataFrame, xy: np.ndarray, title: str,
                     color_cols: list[str] | None = None, wrap: int = 70):
    """Interaktywny wykres t-SNE. Hover: pełny tekst zdania, etykiety (gold, A1, A2, final), iteracja.

    Lista rozwijana nad wykresem przełącza kolorowanie między źródłami etykiet
    (np. gold PolEmo ↔ A1 ↔ A2). Kliknięcie w legendę ukrywa/pokazuje klasę.
    Kształt znacznika: kółko = iteracja 1, romb = iteracja 2.
    """
    import plotly.graph_objects as go

    color_cols = color_cols or label_columns(df)
    wrapped = ["<br>".join(textwrap.wrap(t, wrap)) for t in df["text"]]
    custom = np.column_stack([
        wrapped,
        df["gold"].fillna("brak"), df["A1"].fillna("brak"), df["A2"].fillna("brak"),
        df["final"].fillna("brak"), df["iter"].astype(str), df["sent_id"].astype(str),
        df["n_tokens"].astype(str),
    ])
    hover = ("<b>%{customdata[0]}</b><br>"
             "gold: %{customdata[1]} | A1: %{customdata[2]} | A2: %{customdata[3]} | "
             "final: %{customdata[4]}<br>"
             "iteracja %{customdata[5]} | sent_id %{customdata[6]} | tokenów: %{customdata[7]}"
             "<extra></extra>")
    symbols = np.where(df["iter"].to_numpy() == 1, "circle", "diamond")

    fig = go.Figure()
    traces_per_col: list[list[int]] = []
    for ci, col in enumerate(color_cols):
        vals = df[col].fillna("brak")
        idxs = []
        for lab in [l for l in LABEL_COLORS if l in set(vals)]:
            m = (vals == lab).to_numpy()
            fig.add_trace(go.Scatter(
                x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{lab}",
                legendgroup=col, visible=(ci == 0),
                marker=dict(size=9, color=LABEL_COLORS[lab], symbol=symbols[m],
                            line=dict(width=0.6, color="white"), opacity=0.9),
                customdata=custom[m], hovertemplate=hover))
            idxs.append(len(fig.data) - 1)
        traces_per_col.append(idxs)

    total = len(fig.data)
    buttons = []
    for ci, col in enumerate(color_cols):
        vis = [False] * total
        for i in traces_per_col[ci]:
            vis[i] = True
        buttons.append(dict(label=f"kolor: {col}", method="update",
                            args=[{"visible": vis}, {"legend.title.text": col}]))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.97),
        margin=dict(t=110), template="plotly_white", height=680,
        legend=dict(title=color_cols[0] if color_cols else ""),
        xaxis=dict(title="t-SNE 1", showticklabels=False, zeroline=False),
        yaxis=dict(title="t-SNE 2", showticklabels=False, zeroline=False),
        hoverlabel=dict(align="left", bgcolor="white"),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.0, y=1.06,
                          xanchor="left", yanchor="top")] if len(color_cols) > 1 else [],
    )
    return fig


# --------------------------------------------------------------------------
# Pkt 3 – analiza do dyskusji
# --------------------------------------------------------------------------
def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    z = l2norm(emb)
    return z @ z.T


def _upper(S: np.ndarray):
    i, j = np.triu_indices(len(S), k=1)
    return i, j, S[i, j]


def _pairs_frame(df: pd.DataFrame, i, j, **cols) -> pd.DataFrame:
    out = pd.DataFrame({"sent_id_a": df["sent_id"].to_numpy()[i], "tekst_a": df["text"].to_numpy()[i],
                        "gold_a": df["gold"].to_numpy()[i],
                        "sent_id_b": df["sent_id"].to_numpy()[j], "tekst_b": df["text"].to_numpy()[j],
                        "gold_b": df["gold"].to_numpy()[j]})
    for k, v in cols.items():
        out[k] = v
    return out


def similar_but_different(df: pd.DataFrame, S: np.ndarray, k: int = 10,
                          label_col: str = "gold") -> pd.DataFrame:
    """Pyt. 1: pary zdań o najwyższym podobieństwie, ale o PRZECIWNYM wydźwięku (POZ vs NEG)."""
    i, j, s = _upper(S)
    lab = df[label_col].to_numpy()
    opposite = ((lab[i] == "POZ") & (lab[j] == "NEG")) | ((lab[i] == "NEG") & (lab[j] == "POZ"))
    i, j, s = i[opposite], j[opposite], s[opposite]
    top = np.argsort(-s)[:k]
    return _pairs_frame(df, i[top], j[top], cosine=np.round(s[top], 3))


def _pct_rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / max(1, len(x) - 1)


def cross_model_disagreement(df: pd.DataFrame, S_a: np.ndarray, S_b: np.ndarray,
                             name_a: str, name_b: str, k: int = 10):
    """Pyt. 2 (kandydaci): pary, które model A uważa za bliskie, a B za odległe (i odwrotnie).

    Podobieństwa z różnych modeli mają różne skale, więc porównujemy rangi
    percentylowe wszystkich par. Tabele to materiał do ręcznej oceny, czy dana para
    jest naprawdę bliska znaczeniowo (modele nie są wyrocznią).
    """
    i, j, sa = _upper(S_a)
    sb = S_b[i, j]
    ra, rb = _pct_rank(sa), _pct_rank(sb)
    d = ra - rb
    hi_a = np.argsort(-d)[:k]
    hi_b = np.argsort(d)[:k]
    def mk(idx):
        return _pairs_frame(df, i[idx], j[idx],
                            **{f"ranga_{name_a}": np.round(ra[idx], 3),
                               f"ranga_{name_b}": np.round(rb[idx], 3)})

    return mk(hi_a), mk(hi_b)


PROBE_PAIRS: list[tuple[str, str, str]] = [
    # (kategoria, zdanie A, zdanie B) – tokenizacja jak w korpusie (spacje przy interpunkcji)
    ("parafraza", "Obsługa była bardzo miła i pomocna .", "Personel okazał się niezwykle uprzejmy i życzliwy ."),
    ("parafraza", "Pokój był czysty i dobrze wyposażony .", "W pokoju panowała czystość , a wyposażenie było kompletne ."),
    ("parafraza", "Lekarz dokładnie mnie zbadał i wszystko wytłumaczył .", "Doktor przeprowadził staranne badanie i cierpliwie objaśnił wyniki ."),
    ("przeciwny sens, podobne słowa", "Polecam ten hotel każdemu .", "Nie polecam tego hotelu nikomu ."),
    ("przeciwny sens, podobne słowa", "Jedzenie było smaczne .", "Jedzenie było niesmaczne ."),
    ("przeciwny sens, podobne słowa", "Obsługa była bardzo miła .", "Obsługa była bardzo niemiła ."),
    ("przeciwny sens, te same słowa", "Obsługa była świetna , ale pokój fatalny .", "Pokój był świetny , ale obsługa fatalna ."),
    ("ten sam sens, różna długość", "Polecam .", "Szczerze polecam ten hotel wszystkim znajomym i rodzinie , bo spędziliśmy tam naprawdę udany tydzień ."),
    ("ten sam sens, różna długość", "Bardzo słaby serwis .", "Obsługa w tym miejscu była naprawdę bardzo słaba i musieliśmy długo czekać na wszystko , co zamówiliśmy ."),
    ("ten sam sens, różna długość", "Wszystko w porządku .", "Pobyt przebiegł bez żadnych problemów , wszystko było tak , jak należy , i nie mamy żadnych zastrzeżeń ."),
    ("kontrola: niepowiązane", "Obsługa była bardzo miła .", "Dojazd autostradą zajmuje około dwóch godzin ."),
    ("kontrola: niepowiązane", "Jedzenie było smaczne .", "W pokoju nie działała klimatyzacja ."),
]


def probe_table(embedders: dict[str, Embedder], pairs=PROBE_PAIRS) -> pd.DataFrame:
    """Pyt. 1–3: kontrolowane pary zdań (parafrazy, antonimy, różna długość, kontrola).

    Zwraca cosinus każdej pary dla każdego modelu. Modele muszą być już użyte
    na korpusie (TF-IDF dopasowuje się przy pierwszym ``encode``).
    """
    rows = [{"kategoria": c, "zdanie_a": a, "zdanie_b": b} for c, a, b in pairs]
    out = pd.DataFrame(rows)
    A = [a for _, a, _ in pairs]
    B = [b for _, _, b in pairs]
    for name, emb in embedders.items():
        ea, eb = l2norm(emb.encode(A)), l2norm(emb.encode(B))
        out[name] = np.round((ea * eb).sum(axis=1), 3)
    return out


def length_stats(df: pd.DataFrame, emb: np.ndarray, n_bins: int = 3) -> pd.DataFrame:
    """Pyt. 3–4: jak długość zdania wpływa na wektor.

    Dla przedziałów długości (kwantyle liczby tokenów) liczymy: średnią normę
    surowego wektora (dla uśrednianych wektorów słów – "siła" sygnału), średnie
    podobieństwo do centroidu korpusu oraz średnie podobieństwo do pozostałych
    zdań. Dodatkowo (w ostatnim wierszu) korelacja Spearmana długości ze
    średnim podobieństwem do innych zdań.
    """
    from scipy.stats import spearmanr

    z = l2norm(emb)
    S = z @ z.T
    np.fill_diagonal(S, np.nan)
    mean_sim = np.nanmean(S, axis=1)
    centroid = l2norm(z.mean(axis=0, keepdims=True))[0]
    to_centroid = z @ centroid
    norms = np.linalg.norm(emb, axis=1)
    bins = pd.qcut(df["n_tokens"], q=n_bins, duplicates="drop")
    g = pd.DataFrame({"bin": bins, "norma_wektora": norms, "cos_do_centroidu": to_centroid,
                      "srednie_cos_do_innych": mean_sim})
    res = g.groupby("bin", observed=True).agg(
        n=("norma_wektora", "size"), norma_wektora=("norma_wektora", "mean"),
        cos_do_centroidu=("cos_do_centroidu", "mean"),
        srednie_cos_do_innych=("srednie_cos_do_innych", "mean")).round(3)
    res.index = res.index.astype(str)
    res = res.astype(float)
    rho, p = spearmanr(df["n_tokens"], mean_sim)
    res.loc["Spearman(n_tokens, śr. cos)"] = [np.nan, np.nan, np.nan, round(float(rho), 3)]
    res.attrs["spearman_p"] = float(p)
    return res


def confusion_report(emb: np.ndarray, labels: pd.Series, folds: int = 5, seed: int = 0):
    """Pyt. 5: które klasy wydźwięku osadzenia mylą (macierz pomyłek + precyzja/czułość/F1 per klasa).

    Predykcje z walidacji krzyżowej (regresja logistyczna na znormalizowanych osadzeniach).
    Zwraca (macierz pomyłek: wiersze = etykieta, kolumny = predykcja; raport per klasa).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    m = labels.notna().to_numpy()
    X, y = l2norm(emb[m]), labels[m].to_numpy()
    folds = int(min(folds, pd.Series(y).value_counts().min()))
    pred = cross_val_predict(LogisticRegression(max_iter=3000), X, y,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    order = [c for c in config.SENT_LABELS if c in set(y)]
    cm = pd.crosstab(pd.Series(y, name="etykieta"), pd.Series(pred, name="predykcja")).reindex(
        index=order, columns=order, fill_value=0)
    rep = pd.DataFrame(classification_report(y, pred, labels=order, output_dict=True,
                                             zero_division=0)).T.round(3)
    return cm, rep


def evaluate_embeddings(emb: np.ndarray, labels: pd.Series, k: int = 10, folds: int = 5,
                        seed: int = 0) -> dict:
    """Pyt. 5: czy osadzenia nadają się do zadania (klasyfikacja wydźwięku zdania)?

    * regresja logistyczna, walidacja krzyżowa warstwowa → accuracy, makro-F1,
    * czystość k najbliższych sąsiadów (odsetek sąsiadów tej samej klasy) vs.
      poziom losowy (``losowy_wg_rozkladu`` = suma p_k^2),
    * baza większościowa (``baza_wiekszosciowa`` = udział najliczniejszej klasy; accuracy
      klasyfikatora, który zawsze wybiera tę klasę) – to właściwy punkt odniesienia dla accuracy,
    * silhouette (kosinusowy) względem etykiet.
    Etykiety NaN są pomijane (np. A1/A2 dla iteracji 2).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    m = labels.notna().to_numpy()
    X, y = l2norm(emb[m]), labels[m].to_numpy()
    counts = pd.Series(y).value_counts()
    folds = int(min(folds, counts.min()))
    clf = LogisticRegression(max_iter=3000, C=1.0)
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    purity = float((y[nn] == y[:, None]).mean())
    chance = float(((counts / counts.sum()) ** 2).sum())
    return {"n": int(m.sum()), "accuracy": round(accuracy_score(y, pred), 3),
            "macro_F1": round(f1_score(y, pred, average="macro"), 3),
            f"kNN{k}_czystosc": round(purity, 3), "losowy_wg_rozkladu": round(chance, 3),
            "baza_wiekszosciowa": round(float(counts.max() / counts.sum()), 3),
            "silhouette_cos": round(float(silhouette_score(X, y, metric="cosine")), 3),
            "acc_minus_baza": round(float(accuracy_score(y, pred) - counts.max() / counts.sum()), 3)}


def evaluation_table(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """``evaluate_embeddings`` dla każdego źródła etykiet (final, gold, A1, A2) – te same miary co tabela
    klasyfikacji w sprawozdaniu, więc wyniki są porównywalne między modelami."""
    return pd.DataFrame([{"etykiety": col, **evaluate_embeddings(emb, df[col])} for col in label_columns(df)])


# --------------------------------------------------------------------------
# CLI: pkt 1 i 2 (liczy osadzenia, t-SNE i zapisuje wykresy HTML)
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["fasttext", "use"],
                    choices=MODEL_NAMES)
    ap.add_argument("--iterations", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--fasttext-path", default=None, help="lokalny plik .bin fastText (inaczej pobierze z HF)")
    ap.add_argument("--w2v-path", default=None, help="lokalny plik Word2Vec (.bin/.txt/.kv) z dsmodels.nlp.ipipan.waw.pl")
    ap.add_argument("--use-backend", default="auto", choices=["auto", "tfhub", "distiluse"])
    ap.add_argument("--perplexity", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="przelicz osadzenia mimo cache")
    ap.add_argument("--evaluate", action="store_true",
                    help="miary względem etykiet (evaluate_embeddings) -> results/text_embeddings/eval_<model>.csv")
    a = ap.parse_args(argv)

    # konsola Windows (cp1250) nie zna części znaków – zamiast wyjątku podstaw '?'
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    df = load_sentences(a.iterations)
    print(f"Zdania: {len(df)} (iteracje {a.iterations}), kolumny etykiet: {label_columns(df)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        emb_model = get_embedder(name, a.fasttext_path, a.w2v_path, a.use_backend)
        emb = get_embeddings(emb_model, df, force=a.force)
        csv = export_embeddings_csv(df, emb, name)  # pkt 1: pełne osadzenia (jak w notebooku)
        cov = getattr(emb_model, "last_coverage", None)
        if cov is not None:  # tylko gdy osadzenia liczono teraz (nie z cache)
            print(f"[{name}] pokrycie słownika: średnio {cov.mean():.1%}, min {cov.min():.1%}; "
                  f"zdań bez żadnego znanego słowa: {int((cov == 0).sum())}")
        print(f"[{name}] osadzenia {emb.shape} -> {csv}")
        xy = tsne_2d(emb, a.perplexity)
        pd.DataFrame({"sent_id": df["sent_id"], "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
            OUT_DIR / f"tsne_{name}.csv", index=False)
        fig = make_tsne_figure(df, xy, f"t-SNE – {emb_model.label}")
        fig.write_html(OUT_DIR / f"tsne_{name}.html", include_plotlyjs="cdn")
        print(f"[{name}] {emb.shape} -> {OUT_DIR / f'tsne_{name}.html'}")
        if a.evaluate:
            ev = evaluation_table(df, emb)
            ev.to_csv(OUT_DIR / f"eval_{name}.csv", index=False, encoding="utf-8-sig")
            print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
