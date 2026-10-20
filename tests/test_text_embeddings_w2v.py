"""Testy Text Embeddings, pkt 1b: uśrednione wektory Word2Vec (IPI PAN i PolEmo z części Word Embeddings)."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src import text_embeddings as te

pytest.importorskip("gensim")
from gensim.models import KeyedVectors  # noqa: E402

OUT = te.OUT_DIR
W2V_FILE = config.MODELS_DIR / config.W2V["file"]


def _toy_kv():
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["lekarz", "bardzo", "miły", "Kraków"],
                   np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]], dtype=np.float32))
    return kv


def test_w2v_polemo_embedder_uses_word_embeddings_model(monkeypatch):
    from src import word_embeddings as we

    calls = []
    monkeypatch.setattr(we, "train_polemo_w2v", lambda: calls.append(1) or _toy_kv())
    emb = te.get_embedder("w2v_polemo")
    assert emb.name == "w2v_polemo" and not emb.subword and "PolEmo" in emb.label
    assert calls == []                                            # model ładuje się leniwie
    out = emb.encode(["Lekarz bardzo miły !", "W Kraków nieznane słowo"])
    assert calls == [1]
    assert np.allclose(out[0], [1 / 3, 1 / 3, 1 / 3])            # średnia 3 wektorów, „!” pominięty
    assert np.allclose(out[1], [2, 2, 2])                         # tylko „Kraków” (oryginalna pisownia)
    assert emb.last_coverage[0] == 1.0 and emb.last_coverage[1] == pytest.approx(1 / 4)


def test_w2v_embedder_loads_ipipan_with_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(te, "_load_w2v", lambda path=None: seen.setdefault("path", path) or _toy_kv())
    emb = te.get_embedder("w2v")
    emb.encode(["bardzo miły"])
    assert "path" in seen and seen["path"] is None                # bez ścieżki → models/ (config.W2V)


def test_unknown_model_lists_available():
    with pytest.raises(ValueError, match="w2v_polemo"):
        te.get_embedder("glove")


# --- dane rzeczywiste (pomijane bez wyników / modeli) ------------------------------------

@pytest.mark.parametrize("name, min_cov", [("w2v", 0.95), ("w2v_polemo", 0.85)])
def test_real_sentence_embeddings(name, min_cov):
    npy, meta = OUT / f"emb_{name}.npy", OUT / f"emb_{name}.json"
    if not npy.exists():
        pytest.skip(f"brak {npy.name} (python -m src.text_embeddings --models {name})")
    emb = np.load(npy)
    info = json.loads(meta.read_text(encoding="utf-8"))
    df = te.load_sentences((1, 2))
    assert emb.shape == (len(df), 100) == (info["n"], info["dim"])
    assert info["fingerprint"] == te._fingerprint(df)             # osadzenia dla aktualnych zdań
    assert info["mean_vocab_coverage"] >= min_cov
    assert (np.linalg.norm(emb, axis=1) > 0).all()                # żadne zdanie bez znanego słowa
    csv = pd.read_csv(OUT / f"embeddings_{name}.csv")
    assert len(csv) == len(df) and np.allclose(csv[[f"d{k}" for k in range(100)]].to_numpy(), emb, atol=1e-6)


@pytest.mark.skipif(not W2V_FILE.exists(), reason="brak modelu IPI PAN w models/")
def test_real_w2v_vector_is_mean_of_word_vectors():
    npy = OUT / "emb_w2v.npy"
    if not npy.exists():
        pytest.skip("brak emb_w2v.npy")
    emb, df, kv = np.load(npy), te.load_sentences((1, 2)), te._load_w2v()
    for i in range(3):
        words = te.TOKEN_RE.findall(df["text"].iat[i])
        vecs = [kv[w.lower()] if w.lower() in kv.key_to_index else kv[w] for w in words
                if w.lower() in kv.key_to_index or w in kv.key_to_index]
        assert np.allclose(emb[i], np.mean(vecs, axis=0), atol=1e-5)


# --- pkt 2: t-SNE zdań (wykresy interaktywne i obrazy do sprawozdania) ----------------------

def _plotly_data(path):
    html = path.read_text(encoding="utf-8")
    i = html.find("Plotly.newPlot(")
    j = html.find("[", i)
    data, end = json.JSONDecoder().raw_decode(html[j:])
    layout, _ = json.JSONDecoder().raw_decode(html[html.find("{", j + end):])
    return data, layout


@pytest.mark.parametrize("name", ["w2v", "w2v_polemo"])
def test_real_tsne_html_hover_shows_sentence_and_labels(name):
    path = OUT / f"tsne_{name}.html"
    if not path.exists():
        pytest.skip(f"brak {path.name}")
    data, layout = _plotly_data(path)
    df = te.load_sentences((1, 2)).set_index("sent_id")
    assert {t["legendgroup"] for t in data} == {"final", "gold", "A1", "A2"}
    assert [b["label"] for b in layout["updatemenus"][0]["buttons"]] == [
        "kolor: final", "kolor: gold", "kolor: A1", "kolor: A2"]
    assert all(t.get("visible", True) == (t["legendgroup"] == "final") for t in data)
    final = [t for t in data if t["legendgroup"] == "final"]
    assert sum(len(t["customdata"]) for t in final) == len(df)
    for t in final:
        assert "%{customdata[0]}" in t["hovertemplate"] and "final: %{customdata[4]}" in t["hovertemplate"]
        for row in t["customdata"]:
            sid = int(row[6])
            assert row[0].replace("<br>", " ") == df.at[sid, "text"]       # pełne zdanie w podpowiedzi
            assert row[4] == df.at[sid, "final"] == t["name"]            # etykieta = kolor śladu


@pytest.mark.parametrize("name", ["w2v", "w2v_polemo"])
def test_real_tsne_coordinates_reproducible(name):
    npy, csv = OUT / f"emb_{name}.npy", OUT / f"tsne_{name}.csv"
    if not (npy.exists() and csv.exists()):
        pytest.skip(f"brak wyników {name}")
    xy = pd.read_csv(csv)
    assert xy["sent_id"].tolist() == te.load_sentences((1, 2))["sent_id"].tolist()
    assert np.allclose(te.tsne_2d(np.load(npy)), xy[["x", "y"]].to_numpy(), atol=1e-4)


def test_report_tsne_png_for_selected_models(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(config.ROOT / "report"))
    import build_assets

    if not (OUT / "tsne_w2v.html").exists():
        pytest.skip("brak tsne_w2v.html")
    monkeypatch.setattr(build_assets, "FIG", tmp_path)
    build_assets.build_tsne_figures(("w2v",))
    assert [p.name for p in tmp_path.iterdir()] == ["tsne_w2v.png"]
    assert {"w2v", "w2v_polemo"} <= set(build_assets.MODEL_TITLES)


def test_evaluation_table_per_label_source():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal([4, 0, 0], 0.3, (20, 3)), rng.normal([0, 4, 0], 0.3, (20, 3))])
    df = pd.DataFrame({"final": ["POZ"] * 20 + ["NEG"] * 20, "gold": ["POZ", "NEG"] * 20,
                       "A1": np.nan, "A2": np.nan})
    ev = te.evaluation_table(df, emb).set_index("etykiety")
    assert list(ev.index) == ["final", "gold"]                           # A1/A2 bez etykiet pominięte
    assert ev.at["final", "accuracy"] == 1.0 and ev.at["final", "kNN10_czystosc"] == 1.0
    assert ev.at["gold", "kNN10_czystosc"] < 0.7                          # etykiety niezwiązane ze skupiskami


@pytest.mark.parametrize("name", ["w2v", "w2v_polemo"])
def test_real_evaluation_csv(name):
    path = OUT / f"eval_{name}.csv"
    if not path.exists():
        pytest.skip(f"brak {path.name} (python -m src.text_embeddings --models {name} --evaluate)")
    ev = pd.read_csv(path, encoding="utf-8-sig").set_index("etykiety")
    assert list(ev.index) == ["final", "gold", "A1", "A2"]
    assert (ev["n"] == 200).all() and ev["accuracy"].between(0, 1).all()
    assert ev.at["gold", "baza_wiekszosciowa"] == 0.25                    # próbki zbalansowane wg gold
