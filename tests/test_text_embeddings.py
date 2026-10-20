"""Testy modułu Text Embeddings – działają offline (TF-IDF i sztuczne wektory słów)."""
import numpy as np
import pytest

from src import text_embeddings as te


@pytest.fixture(scope="module")
def df():
    return te.load_sentences((1, 2))


def test_load_sentences(df):
    assert {"sent_id", "iter", "text", "n_tokens", "gold", "A1", "A2", "final"} <= set(df.columns)
    assert set(df["iter"]) == {1, 2}
    assert df["sent_id"].is_unique
    assert df["gold"].isin(["POZ", "NEG", "NEU", "AMB"]).all()
    assert df.loc[df["iter"] == 1, "A1"].notna().all()


def test_tfidf_pipeline(df):
    emb_model = te.get_embedder("tfidf")
    emb = emb_model.encode(df["text"].tolist())
    assert emb.shape[0] == len(df)
    xy = te.tsne_2d(emb)
    assert xy.shape == (len(df), 2)
    fig = te.make_tsne_figure(df, xy, "test")
    assert len(fig.data) >= 4
    assert "customdata" in fig.data[0].to_plotly_json()
    S = te.cosine_matrix(emb)
    assert te.similar_but_different(df, S, k=3).shape[0] == 3
    a, b = te.cross_model_disagreement(df, S, S[::-1, ::-1], "a", "b", k=3)
    assert len(a) == len(b) == 3
    assert te.length_stats(df, emb).shape[0] >= 2
    res = te.evaluate_embeddings(emb, df["gold"])
    assert 0 <= res["accuracy"] <= 1
    probes = te.probe_table({"tfidf": emb_model})
    assert len(probes) == len(te.PROBE_PAIRS) and "tfidf" in probes


def test_average_embedder_with_fake_vectors():
    gensim = pytest.importorskip("gensim")
    from gensim.models import KeyedVectors

    kv = KeyedVectors(vector_size=4)
    kv.add_vectors(["dobry", "hotel"], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    e = te.AvgVectorEmbedder("fake", "fake", lambda: kv, subword=False)
    out = e.encode(["Dobry hotel .", "zupełnie nieznane"])
    assert np.allclose(out[0], [0.5, 0.5, 0, 0])      # średnia dwóch wektorów, interpunkcja pominięta
    assert np.allclose(out[1], 0)                       # brak słów w słowniku → wektor zerowy
    assert e.last_coverage[0] == 1.0 and e.last_coverage[1] == 0.0
