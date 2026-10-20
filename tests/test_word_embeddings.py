"""Testy Word Embeddings, pkt 1 – działają bez pobierania modeli (sztuczne wektory).

Testy z prawdziwym modelem IPI PAN są pomijane, gdy nie ma go w models/.
"""
import gzip
import os
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import config
from src import text_embeddings as te
from src import word_embeddings as we

pytest.importorskip("gensim")
from gensim.models import KeyedVectors  # noqa: E402

W2V_FILE = config.MODELS_DIR / config.W2V["file"]


@pytest.fixture
def kv():
    words = ["dobry", "hotel", "Kraków", "nie", "polecam", "łazience"]
    out = KeyedVectors(vector_size=4)
    out.add_vectors(words, np.eye(6, 4, dtype=np.float32) + 0.1)
    return out


# --- wyszukiwanie wektora --------------------------------------------------------

def test_lookup_order(kv):
    v, form, hit = we.lookup(kv, "dobry", "Dobry")
    assert (form, hit) == ("dobry", True) and np.allclose(v, kv["dobry"])
    assert we.lookup(kv, "kraków", "Kraków")[1] == "Kraków"      # oryginalna pisownia
    assert we.lookup(kv, "kraków", "KRAKÓW")[1] == "Kraków"      # wielka litera na początku
    assert we.lookup(kv, "xyz", "Xyz") == (None, None, False)    # brak – nie wektor zerowy


def test_lookup_subword_model_returns_vector_outside_vocabulary():
    class FakeFastText:
        key_to_index = {"dobry": 0}

        def __getitem__(self, word):
            return np.ones(4, dtype=np.float32)

    v, form, hit = we.lookup(FakeFastText(), "dobryy", "dobryy", subword=True)
    assert v is not None and form is None and hit is False


# --- słownik zaanotowanego korpusu -------------------------------------------------

@pytest.mark.parametrize("counts, n, expected", [
    ({"O": 3, "WZM": 1}, 4, "O"),                          # 25% < 50%
    ({"O": 2, "WZM": 2}, 4, "WZM"),                        # dokładnie 50%
    ({"O": 2, "WYR_POZ": 2, "WYR_NEG": 2}, 6, "WYR_POZ"),  # remis → kolejność TOKEN_LABELS
    ({"ODW": 1}, 1, "ODW"),
    ({}, 0, "O"),
])
def test_word_label_rule(counts, n, expected):
    assert we.word_label(counts, n) == expected


@pytest.fixture
def toy_iteration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANNOT_DIR", tmp_path / "annotation")
    d = config.iter_dir(1)
    config.export_dir(1).mkdir(parents=True)
    texts = {1: "Bardzo dobry hotel , polecam !", 2: "Nie polecam , 5 gwiazdek ?"}
    pd.DataFrame({"sent_id": list(texts), "text": list(texts.values()), "n_tokens": [6, 6]}).to_csv(
        d / "sample.csv", index=False)
    pd.DataFrame({"sent_id": [1, 2], "gold": ["POZ", "NEG"], "gold_raw": ["z_plus_m", "z_minus_m"]}).to_csv(
        d / "_key_gold.csv", index=False)
    pd.DataFrame({"sent_id": [1, 2], "final": ["POZ", "NEG"], "zrodlo": ["zgodne", "zgodne"]}).to_csv(
        d / "adjudicated.csv", index=False)
    labels = {"A1": {1: ["WZM", "WYR_POZ", "O", "O", "WYR_POZ", "O"], 2: ["ODW", "WYR_POZ", "O", "O", "O", "O"]},
              "A2": {1: ["WZM", "WYR_POZ", "O", "O", "O", "O"], 2: ["O", "WYR_POZ", "O", "O", "WYR_NEG", "O"]}}
    rows = [(sid, j + 1, tok, a, labels[a][sid][j])
            for a in labels for sid, text in texts.items() for j, tok in enumerate(text.split())]
    pd.DataFrame(rows, columns=["sent_id", "tok_idx", "token", "annotator", "label"]).to_csv(
        config.export_dir(1) / "tokens_long.csv", index=False, encoding="utf-8-sig")


def test_annotated_vocabulary_on_toy_export(toy_iteration):
    v = we.annotated_vocabulary([1], polemo_freq=Counter({"polecam": 7})).set_index("word")
    assert set(v.index) == {"bardzo", "dobry", "hotel", "polecam", "nie", "gwiazdek"}  # bez „,”, „!”, „5”
    p = v.loc["polecam"]
    assert (p["freq"], p["n_sentences"], p["n_WYR_POZ"], p["n_O"]) == (2, 2, 3, 1)
    assert p["annot_share"] == pytest.approx(0.75) and p["word_label"] == "WYR_POZ"
    assert (p["share_POZ"], p["share_NEG"], p["polemo_freq"]) == (0.5, 0.5, 7)
    assert v.loc["nie", "surface"] == "Nie" and v.loc["nie", "word_label"] == "ODW"   # 1 z 2 anotacji
    assert v.loc["gwiazdek", "word_label"] == "WYR_NEG" and v.loc["hotel", "word_label"] == "O"
    assert np.allclose(v[we.SENT_SHARE_COLS].sum(axis=1), 1.0)
    assert v["freq"].tolist() == sorted(v["freq"], reverse=True)


# --- wektory, zapis i pokrycie -------------------------------------------------------

def test_embed_vocabulary_and_npz_roundtrip(kv, tmp_path):
    vocab = pd.DataFrame({"word": ["dobry", "kraków", "xyz"], "surface": ["Dobry", "Kraków", "xyz"]})
    out, arrays = we.embed_vocabulary(vocab, kv, "toy")
    assert out["toy_in_vocab"].tolist() == [True, True, False]
    assert out["toy_form"].tolist() == ["dobry", "Kraków", ""]
    assert out["toy_rank"].iloc[:2].tolist() == [1, 3] and np.isnan(out["toy_rank"].iloc[2])
    assert arrays["words"].tolist() == ["dobry", "kraków"] and arrays["vectors"].shape == (2, 4)
    np.savez(tmp_path / "v.npz", **arrays)
    back = np.load(tmp_path / "v.npz")
    assert back["words"].tolist() == ["dobry", "kraków"]
    assert np.array_equal(back["vectors"], np.vstack([kv["dobry"], kv["Kraków"]]))


def test_coverage_counts_and_oov_reasons(kv):
    vocab = pd.DataFrame({
        "word": ["dobry", "polecam", "lazience", "2osobowym", "qwerty"],
        "surface": ["dobry", "polecam", "lazience", "2osobowym", "qwerty"],
        "freq": [5, 3, 1, 1, 1], "annot_share": [1.0, 1.0, 0.0, 0.0, 0.5],
        "word_label": ["WYR_POZ", "WYR_POZ", "O", "O", "WYR_NEG"], "polemo_freq": [50, 30, 20, 1, 9],
    })
    vocab, _ = we.embed_vocabulary(vocab, kv, "toy")
    c = we.coverage(vocab, "toy", kv)
    assert c["types"] == {"n": 5, "in_vocab": 2, "share": 0.4}
    assert c["tokens"]["share"] == pytest.approx(8 / 11)
    assert c["annotated_types"]["in_vocab"] == 2 and c["annotated_types"]["n"] == 3
    assert c["by_word_label"]["WYR_POZ"]["share"] == 1.0 and c["by_word_label"]["WYR_NEG"]["share"] == 0.0
    reasons = {e["word"]: e["reason"] for e in c["oov_examples"]}
    assert reasons["lazience"] == "brak polskich znaków (w modelu: łazience)"
    assert reasons["2osobowym"] == "zawiera cyfrę"
    assert reasons["qwerty"].startswith("inne")


def test_markdown_reports_render(kv):
    vocab = pd.DataFrame({"word": ["dobry", "nie"], "surface": ["dobry", "nie"], "freq": [2, 1],
                          "annot_share": [1.0, 0.0], "word_label": ["WYR_POZ", "O"], "polemo_freq": [9, 9]})
    vocab, _ = we.embed_vocabulary(vocab, kv, "toy")
    md = we.coverage_markdown({"toy": we.coverage(vocab, "toy", kv)}, {"toy": "Model testowy"})
    assert "| Model testowy | 100,0% (2/2)" in md
    s = we.sanity_markdown({"toy": we.sanity(kv, topn=2)}, {"toy": "Model testowy"})
    assert "| dobry |" in s


# --- wczytywanie modeli --------------------------------------------------------------

def test_load_w2v_limit_and_cache(tmp_path, monkeypatch):
    path = tmp_path / "tiny.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("4 3\na 1 0 0\nb 0 1 0\nc 0 0 1\nd 1 1 0\n")
    kv = te._load_w2v(str(path), limit=2)
    assert kv.index_to_key == ["a", "b"]
    assert (tmp_path / "tiny_2.kv").exists()

    def boom(*args, **kwargs):
        raise AssertionError("powinno wczytać z cache")
    monkeypatch.setattr(KeyedVectors, "load_word2vec_format", staticmethod(boom))
    assert te._load_w2v(str(path), limit=2).index_to_key == ["a", "b"]


def test_load_w2v_missing_file_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="dsmodels"):
        te._load_w2v(str(tmp_path / "brak.txt.gz"))


TRAIN_SNIPPET = """
import numpy as np
from src import word_embeddings as we
rng = np.random.default_rng(0)
words = [f"s{i}" for i in range(30)]
sents = [list(rng.choice(words, 8)) for _ in range(200)]
params = dict(vector_size=8, sg=1, negative=5, window=2, min_count=1, epochs=3, seed=1, workers=1)
kv = we.train_polemo_w2v(sents, params, path=__import__("pathlib").Path(r"{path}"), force=True)
print(repr(float(np.abs(kv.vectors).sum())))
"""


def test_train_polemo_w2v_is_reproducible_across_hash_seeds(tmp_path):
    """Ten sam wynik w dwóch procesach z różnym PYTHONHASHSEED (seed + workers=1)."""
    outs = []
    for seed in ("1", "2"):
        code = TRAIN_SNIPPET.replace("{path}", str(tmp_path / f"m{seed}.kv"))
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]
    a = KeyedVectors.load(str(tmp_path / "m1.kv"))
    b = KeyedVectors.load(str(tmp_path / "m2.kv"))
    assert np.array_equal(a.vectors, b.vectors)


def test_models_dir_is_gitignored():
    if shutil.which("git") is None:
        pytest.skip("brak git")
    r = subprocess.run(["git", "check-ignore", "-q", "models/dowolny_model.bin"], cwd=config.ROOT)
    if r.returncode == 128:
        pytest.skip("to nie jest repozytorium git")
    assert r.returncode == 0


# --- dane rzeczywiste -------------------------------------------------------------------

def test_real_annotated_vocabulary():
    if not (config.export_dir(1) / "tokens_long.csv").exists():
        pytest.skip("brak eksportu iteracji 1")
    v = we.annotated_vocabulary([1, 2])
    assert len(v) == 1505 and v["freq"].sum() == 2783
    assert v["word"].is_unique
    assert v.loc[v["word"] == "bardzo", "word_label"].item() == "WZM"


@pytest.mark.skipif(not W2V_FILE.exists(), reason="brak modelu IPI PAN w models/")
def test_real_ipipan_model():
    kv = te._load_w2v()
    assert len(kv) == config.W2V["limit"] and kv.vector_size == 100
    for w in ["nie", "bardzo", "trochę", "polecam", "m", "śmy"]:  # „m”, „śmy”: segmentacja jak w PolEmo
        assert w in kv.key_to_index
