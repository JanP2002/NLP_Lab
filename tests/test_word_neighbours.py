"""Testy Word Embeddings, pkt 3 (k najbliższych słów) – na sztucznych wektorach, bez modeli."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src import word_embeddings as we
from src import word_neighbours as wn

pytest.importorskip("gensim")
from gensim.models import KeyedVectors  # noqa: E402


def make_kv(words_vectors: dict) -> KeyedVectors:
    kv = KeyedVectors(vector_size=len(next(iter(words_vectors.values()))))
    kv.add_vectors(list(words_vectors), np.array(list(words_vectors.values()), dtype=np.float32))
    return kv


@pytest.fixture
def kv():
    return make_kv({
        "dobry": [1, 0, 0], "Dobry": [0.99, 0.1, 0], "123": [0.98, 0.2, 0], "430-31-43": [0.97, 0.25, 0],
        "świetny": [0.9, 0.3, 0], "pół-żartem": [0.85, 0.4, 0], "zły": [0.8, 0.5, 0], "dom": [0, 1, 0],
        "kot": [0, 0, 1],
    })


# --- listy sąsiadów -----------------------------------------------------------------

def test_neighbours_filter_alpha_and_self(kv):
    out = wn.neighbours(kv, "dobry", k=3)
    assert [w for w, _ in out] == ["świetny", "pół-żartem", "zły"]   # bez „Dobry”, „123”, „430-31-43”
    assert all(-1 <= s <= 1 for _, s in out)


def test_neighbours_small_vocabulary_terminates(kv):
    out = wn.neighbours(kv, "dobry", k=50)
    assert [w for w, _ in out] == ["świetny", "pół-żartem", "zły", "dom", "kot"]


@pytest.mark.parametrize("a, b, expected", [
    ("miła", "miły", True), ("dobry", "dobrze", True), ("polecam", "polecasz", True),
    ("kłopot", "kłopoty", True), ("dobry", "doskonały", False), ("trochę", "troszkę", False),
    ("nie", "niemiła", False), ("Miła", "miłą", True),
])
def test_same_stem(a, b, expected):
    assert wn.same_stem(a, b) is expected


def test_overlap_metrics():
    assert wn.overlap_metrics(["x", "y", "z"], ["y", "z", "w"], 3) == {"overlap": 2 / 3, "jaccard": 0.5}
    assert wn.overlap_metrics(["x", "y"], ["x", "y"], 2) == {"overlap": 1.0, "jaccard": 1.0}
    assert wn.overlap_metrics(["x", "y"], ["v", "w"], 2) == {"overlap": 0.0, "jaccard": 0.0}


def test_shared_vocabulary_and_neighbours(kv):
    other = make_kv({"dobry": [1, 0, 0], "zły": [0.9, 0.1, 0], "kot": [0, 0, 1], "123": [1, 0, 0],
                     "tylkotu": [0.95, 0.05, 0]})
    shared = wn.shared_vocabulary([kv, other])
    assert shared == sorted(["dobry", "zły", "kot"])                  # bez „123” i słów jednego modelu
    M = wn.normalized_matrix(kv, shared)
    assert wn.shared_neighbours(kv, shared, M, "dobry", k=2) == ["zły", "kot"]


def test_second_order_identity_and_noise():
    rng = np.random.default_rng(0)
    words = [f"s{i}" for i in range(30)]
    base = make_kv(dict(zip(words, rng.normal(size=(30, 8)))))
    per_word, total = wn.second_order(base, base, words, words)
    assert np.allclose(per_word, 1) and total == pytest.approx(1)
    noise = make_kv(dict(zip(words, rng.normal(size=(30, 8)))))
    _, total_noise = wn.second_order(base, noise, words, words)
    assert abs(total_noise) < 0.3


def test_polemo_polarity_and_classes():
    texts = ["Polecam ten hotel !"] * 3 + ["Nie polecam , hotel brudny ."] * 2 + ["Hotel stoi ."]
    golds = ["POZ"] * 3 + ["NEG"] * 2 + ["NEU"]
    p = wn.polemo_polarity(texts, golds, min_count=2)
    assert set(p.index) == {"polecam", "ten", "hotel", "nie", "brudny"}        # „stoi” < 2, „!” to nie słowo
    assert p.loc["polecam", "polarity"] == pytest.approx((3 - 2) / 5)
    assert p.loc["hotel", "polemo_n"] == 6 and p.loc["brudny", "polarity"] == -1.0
    assert [wn.polarity_class(x, 0.1) for x in (0.5, 0.05, -0.2, float("nan"))] == ["+", "0", "-", ""]


# --- zestaw zapytań i całość -----------------------------------------------------------

@pytest.fixture
def fake_models(monkeypatch):
    rng = np.random.default_rng(1)
    words = ["miła", "brudne", "bardzo", "trochę", "nie", "hotel", "lekarz", "plusy", "minusy", "dobrze",
             "kurz", "pleśń", "uprzejma", "życzliwa"]
    kv_a = make_kv(dict(zip(words, rng.normal(size=(len(words), 6)))))
    kv_b = make_kv(dict(zip(words[:-1], rng.normal(size=(len(words) - 1, 6)))))   # bez „życzliwa”
    models = {"a": we.WordModel("a", "model A", lambda: kv_a, False),
              "b": we.WordModel("b", "model B", lambda: kv_b, False)}
    monkeypatch.setattr(we, "MODELS", models)
    return kv_a, kv_b


def _toy_vocab(kv_a, kv_b):
    rows = [("miła", "WYR_POZ", 5), ("brudne", "WYR_NEG", 3), ("bardzo", "WZM", 9), ("trochę", "OSL", 2),
            ("nie", "O", 20), ("hotel", "O", 4), ("życzliwa", "WYR_POZ", 1), ("dobrze", "WYR_POZ", 2)]
    v = pd.DataFrame(rows, columns=["word", "word_label", "freq"])
    v["surface"] = v["word"]
    for m, kv in (("a", kv_a), ("b", kv_b)):
        v[f"{m}_in_vocab"] = v["word"].isin(kv.key_to_index)
        v[f"{m}_form"] = np.where(v[f"{m}_in_vocab"], v["word"], "")
    return v


def test_query_words_main_set_and_showcase(fake_models):
    v = _toy_vocab(*fake_models)
    queries, showcase = wn.query_words(v, ["a", "b"])
    main = set(queries.loc[queries["in_main_set"], "word"])
    assert main == {"miła", "brudne", "bardzo", "trochę", "dobrze"}    # „życzliwa” nie ma w modelu b
    assert "nie" in set(queries["word"]) and not queries.set_index("word").at["nie", "in_main_set"]
    assert showcase[:2] == ["miła", "dobrze"] and "nie" in showcase


def test_run_end_to_end(fake_models, tmp_path, monkeypatch):
    _toy_vocab(*fake_models).to_csv(tmp_path / "vocab.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(config, "WORD_EMB_DIR", tmp_path)
    pol = wn.polemo_polarity(["miła uprzejma hotel", "brudne kurz pleśń minusy"] * 3, ["POZ", "NEG"] * 3, 1)
    monkeypatch.setattr(wn, "polemo_polarity", lambda min_count: pol)
    s = wn.run(["a", "b"], k=3)
    for m in ("a", "b"):
        t = pd.read_csv(tmp_path / f"neighbours_{m}.csv", encoding="utf-8-sig", keep_default_na=False)
        assert t.groupby("query").size().eq(3).all()
        assert not (t["query"].str.lower() == t["neighbour"].str.lower()).any()
    comp = pd.read_csv(tmp_path / "knn_comparison.csv", encoding="utf-8-sig")
    assert len(comp) == s["n_queries"] == 5 and comp["overlap"].between(0, 1).all()
    assert json.loads((tmp_path / "knn_summary.json").read_text(encoding="utf-8"))["n_queries"] == 5
    md = (tmp_path / "knn_summary.md").read_text(encoding="utf-8")
    assert "## 1. Zgodność list" in md and "| miła | WYR_POZ |" in md
    with pytest.raises(ValueError):
        wn.run(["a"])


# --- dane rzeczywiste --------------------------------------------------------------------

def test_real_neighbour_lists():
    path = config.WORD_EMB_DIR / "neighbours_w2v.csv"
    if not path.exists():
        pytest.skip("brak wyników pkt 3")
    t = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    main = t[t["in_main_set"].astype(str) == "True"]
    assert main["query"].nunique() == 215 and main.groupby("query").size().eq(10).all()
    assert not (t["query"].str.lower() == t["neighbour"].str.lower()).any()
    assert t["neighbour"].str.fullmatch(wn.ALPHA_RE.pattern.strip("^$")).all()
    assert pd.to_numeric(t["cosine"]).le(1).all()
