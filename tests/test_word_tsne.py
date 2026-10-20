"""Testy Word Embeddings, pkt 2 (t-SNE słów) – na sztucznych wektorach, bez modeli."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src import word_tsne as wt

pytest.importorskip("plotly")


def _vocab_row(word, label, shares, freq=1):
    counts = {f"n_{l}": 0 for l in [config.NO_LABEL, *config.TOKEN_LABELS]}
    if label == config.NO_LABEL:
        counts["n_O"] = 2
    else:
        counts[f"n_{label}"] = 2
    return {"word": word, "surface": word.capitalize(), "freq": freq, "n_sentences": freq, **counts,
            "annot_share": 0.0 if label == config.NO_LABEL else 1.0, "word_label": label,
            **dict(zip(wt.SHARE_COLS, shares)), "polemo_freq": 10 * freq, "toy_rank": float(freq)}


@pytest.fixture
def toy_dir(tmp_path):
    """3 skupiska po 10 słów: WYR_POZ (zdania POZ), WYR_NEG (zdania NEG), O (zdania NEU); słowo „null” w O."""
    rng = np.random.default_rng(0)
    groups = [("pos", "WYR_POZ", [1, 0, 0, 0]), ("neg", "WYR_NEG", [0, 1, 0, 0]), ("o", "O", [0, 0, 1, 0])]
    rows, vecs = [], []
    for g, (prefix, label, shares) in enumerate(groups):
        for i in range(10):
            word = "null" if (prefix == "o" and i == 0) else f"{prefix}{i}"
            rows.append(_vocab_row(word, label, shares, freq=1 + i % 3))
            vecs.append(np.eye(3, 5)[g] * 5 + rng.normal(0, 0.1, 5))
    vocab = pd.DataFrame(rows)
    vocab.loc[len(vocab)] = _vocab_row("bezwektora", "WYR_POZ", [1, 0, 0, 0])  # słowo bez wektora
    vocab.to_csv(tmp_path / "vocab.csv", index=False, encoding="utf-8-sig")
    np.savez(tmp_path / "vectors_toy.npz", words=np.array([r["word"] for r in rows]),
             vectors=np.array(vecs, dtype=np.float32), in_vocab=np.ones(len(rows), bool))
    return tmp_path


def test_context_label_rules():
    v = pd.DataFrame([[0.75, 0.25, 0, 0], [0.5, 0.5, 0, 0], [np.nan] * 4, [0, 0, 1, 0]], columns=wt.SHARE_COLS)
    assert wt.context_label(v).tolist() == ["POZ", "mieszany", "brak", "NEU"]


def test_load_points_subsets_and_na_safe_words(toy_dir):
    points, X = wt.load_points("toy", "all", out_dir=toy_dir)
    assert len(points) == len(X) == 30 and "bezwektora" not in set(points["word"])
    assert "null" in set(points["word"])                      # nie zamienione na NaN
    assert points["toy_rank"].dtype.kind == "f"
    assert points["context_label"].value_counts().to_dict() == {"POZ": 10, "NEG": 10, "NEU": 10}
    ann, X_ann = wt.load_points("toy", "annotated", out_dir=toy_dir)
    assert len(ann) == len(X_ann) == 20 and set(ann["word_label"]) == {"WYR_POZ", "WYR_NEG"}
    with pytest.raises(ValueError):
        wt.load_points("toy", "inne", out_dir=toy_dir)
    with pytest.raises(FileNotFoundError, match="pkt 1"):
        wt.load_points("brak", "all", out_dir=toy_dir)


def test_tsne_coords_deterministic(toy_dir):
    _, X = wt.load_points("toy", "all", out_dir=toy_dir)
    a, b = wt.tsne_coords(X, seed=1), wt.tsne_coords(X, seed=1)
    assert a.shape == (30, 2) and np.allclose(a, b)


def test_figure_structure_hover_and_switching(toy_dir):
    points, X = wt.load_points("toy", "all", out_dir=toy_dir)
    fig = wt.make_word_figure(points, wt.tsne_coords(X, seed=1), "test", "toy_rank")
    groups = [t.legendgroup for t in fig.data]
    word_tr = [t for t in fig.data if t.legendgroup == "word_label"]
    ctx_tr = [t for t in fig.data if t.legendgroup == "context_label"]
    assert len(word_tr) == 3 and len(ctx_tr) == 3
    assert all(t.visible for t in word_tr) and not any(t.visible for t in ctx_tr)
    assert word_tr[0].name.startswith("O")                    # tło rysowane pierwsze (pod spodem)
    assert word_tr[0].marker.color == wt.LIGHT_GREY and word_tr[0].marker.opacity == 0.45
    # lista rozwijana: każdy wariant pokazuje dokładnie swoje ślady
    dropdown = fig.layout.updatemenus[0].buttons
    for b, group in zip(dropdown, ["word_label", "context_label"]):
        assert b.args[0]["visible"] == [g == group for g in groups]
    # podpowiedź: słowo + anotacja
    pos = next(t for t in word_tr if t.name.startswith("WYR_POZ"))
    hover = next(h for h in pos.customdata if h.startswith("<b>pos1</b>"))
    assert "etykieta słowa: <b>WYR_POZ</b>" in hover and "kontekst zdań: <b>POZ</b>" in hover
    assert "ranga w modelu: 2" in hover
    # podpisy tylko dla słów z etykietą ≠ O występujących ≥ 2 razy
    labelled = {x for t in word_tr for x in t.text if x}
    assert labelled == set(points.loc[(points["freq"] >= 2) & (points["word_label"] != "O"), "word"])


def test_label_separation_detects_clusters_and_not_noise():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal([5, 0, 0], 0.3, (20, 3)), rng.normal([0, 5, 0], 0.3, (20, 3))])
    labels = np.array(["A"] * 20 + ["B"] * 20)
    r = wt.label_separation(X, labels, k=5, n_perm=200, seed=0)
    assert r["purity"] == 1.0 and r["p_value"] <= 0.01 and r["silhouette"] > 0.5
    assert r["counts"] == {"A": 20, "B": 20}
    noise = wt.label_separation(X, rng.permutation(labels), k=5, n_perm=200, seed=0)
    assert noise["random_ci95"][0] <= noise["purity"] <= noise["random_ci95"][1]
    assert abs(noise["silhouette"]) < 0.2


def test_polarity_probe_separable_vs_shuffled():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal([3, 0, 0, 0], 0.5, (30, 4)), rng.normal([0, 3, 0, 0], 0.5, (30, 4))])
    y = np.array(["A"] * 30 + ["B"] * 30)
    r = wt.polarity_probe(X, y, n_repeats=2, n_perm=20)
    assert r["balanced_accuracy"] > 0.95 and r["p_value"] <= 0.05 and r["n"] == 60
    noise = wt.polarity_probe(rng.normal(size=(60, 4)), y, n_repeats=2, n_perm=20)
    assert abs(noise["balanced_accuracy"] - 0.5) < 0.2


def test_run_writes_all_outputs(toy_dir, monkeypatch):
    monkeypatch.setattr(config, "WORD_EMB_DIR", toy_dir)
    res = wt.run(["toy"], ["all", "annotated"], seed=1)
    for subset, n in [("all", 30), ("annotated", 20)]:
        stem = toy_dir / f"tsne_words_toy_{subset}"
        for ext in (".html", ".png", ".csv"):
            assert stem.with_suffix(ext).exists()
        csv = pd.read_csv(stem.with_suffix(".csv"), encoding="utf-8-sig", keep_default_na=False)
        assert len(csv) == n and list(csv.columns) == ["word", "word_label", "context_label", "freq",
                                                       "annot_share", "x", "y"]
    sep = json.loads((toy_dir / "label_separation.json").read_text(encoding="utf-8"))
    assert sep == json.loads(json.dumps(res))
    pos_neg = sep["toy"]["WYR_POZ vs WYR_NEG (etykieta słowa)"]
    # skupiska po 10 słów, k = 10 → co najwyżej 9 sąsiadów z tą samą etykietą
    assert pos_neg["purity"] == pytest.approx(0.9) and pos_neg["p_value"] < 0.01
    assert "Czystość k-NN" in (toy_dir / "label_separation.md").read_text(encoding="utf-8")


def test_real_points_match_point1_outputs():
    if not (config.WORD_EMB_DIR / "vectors_w2v.npz").exists():
        pytest.skip("brak wyników pkt 1 dla w2v")
    points, X = wt.load_points("w2v", "all")
    assert len(points) == 1423 and X.shape == (1423, 100)
    ann, _ = wt.load_points("w2v", "annotated")
    assert len(ann) == 296 and (ann["annot_share"] > 0).all()
