"""Testy Text Embeddings, pkt 3 (analiza osadzeń zdań) – na sztucznych danych i, gdy są, na wynikach."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src import text_analysis as ta
from src import text_embeddings as te

W2V_FILE = config.MODELS_DIR / config.W2V["file"]


def test_read_tex_rows_parses_booktabs(tmp_path):
    path = tmp_path / "t.tex"
    path.write_text(
        "\\begin{tabular}{llr}\n\\toprule\nModel & Grupa & x \\\\\n\\midrule\n"
        "fastText & a & 0{,}763 \\\\\n & b & $-$0{,}123 \\\\\n\\midrule\nUSE & a & 1{,}00 \\\\\n"
        "\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    assert ta.read_tex_rows(path) == [["fastText", "a", 0.763], ["", "b", -0.123], ["USE", "a", 1.0]]
    assert ta.read_tex_rows(tmp_path / "brak.tex") == []


def test_team_tables_from_report():
    if not (ta.GEN / "emb_probe_means.tex").exists():
        pytest.skip("brak tabel zespołu w report/generated")
    p = ta.team_probe_means()
    assert p.at["parafraza", "fastText"] == pytest.approx(0.746) and list(p.columns) == ["fastText", "USE", "TF-IDF"]
    length = ta.team_length()
    assert set(length["model"]) == {"fastText", "USE", "TF-IDF"} and (length["grupa"] == "Spearman").sum() == 3
    c = ta.team_classif()
    assert list(c["model"]) == ["fasttext", "use", "tfidf"] and c["baza"].eq(0.415).all()
    assert list(ta.team_recall().columns) == config.SENT_LABELS


def test_jaccard_and_lexical_dependence():
    J = ta.jaccard_matrix(["Ala ma kota .", "ma kota Ala", "zupełnie inne słowa"])
    assert J[0, 1] == 1.0 and J[0, 2] == 0.0 and np.allclose(np.diag(J), 1)
    J4 = ta.jaccard_matrix(["a b c", "b c d", "c d e", "x y z"])
    dep = ta.lexical_dependence(J4.copy(), J4)                          # podobieństwo = Jaccard
    assert dep["spearman_cos_jaccard"] == pytest.approx(1.0)
    assert dep["n_pairs"]["brak wspólnych słów"] == 3                   # „a b c” i „c d e” mają wspólne „c”
    assert dep["n_pairs"]["Jaccard (0; 0,1]"] == 0 and np.isnan(dep["mean_cos"]["Jaccard (0; 0,1]"])


def test_opposite_share_top_and_top_pairs():
    labels = np.array(["POZ", "NEG", "POZ", "NEU"])
    S = np.eye(4)
    S[0, 1] = S[1, 0] = 0.9                                             # najbliższa para: POZ–NEG
    S[0, 2] = S[2, 0] = 0.5
    o = ta.opposite_share_top(S, labels, share=0.2)                      # 1 z 6 par
    assert o["n_top"] == 1 and o["top_opposite"] == 1.0 and o["all_opposite"] == pytest.approx(2 / 6)
    df = pd.DataFrame({"final": labels, "text": list("abcd")})
    tp = ta.top_pairs(df, S, k=5, opposite_only=True)
    assert set(zip(tp["final_a"], tp["final_b"])) <= {("POZ", "NEG"), ("NEG", "POZ")} and tp["cos"].iat[0] == 0.9


@pytest.mark.skipif(not W2V_FILE.exists(), reason="brak modelu IPI PAN w models/")
def test_real_analysis_outputs(tmp_path, monkeypatch):
    if not (te.OUT_DIR / "emb_w2v.npy").exists():
        pytest.skip("brak osadzeń w2v (python -m src.text_embeddings --models w2v w2v_polemo)")
    monkeypatch.setattr(te, "OUT_DIR", tmp_path)
    for f in ("emb_w2v.npy", "emb_w2v.json", "emb_w2v_polemo.npy", "emb_w2v_polemo.json"):   # cache osadzeń
        (tmp_path / f).write_bytes((config.RESULTS_DIR / "text_embeddings" / f).read_bytes())
    data = ta.run(["w2v", "w2v_polemo"], "test")
    md = (tmp_path / "analiza_test.md").read_text(encoding="utf-8")
    for q in ("Pyt. 1", "Pyt. 2", "Pyt. 3–4", "Pyt. 5"):
        assert q in md
    assert len(pd.read_csv(tmp_path / "probes_test.csv", encoding="utf-8-sig")) == len(te.PROBE_PAIRS)
    assert json.loads((tmp_path / "analiza_test.json").read_text(encoding="utf-8"))["n_sentences"] == 200
    assert data["q1_opposite_top"]["w2v"]["n_top"] == 199                 # 1% z 19 900 par
    assert 0 < data["q2_lexical"]["w2v"]["spearman_cos_jaccard"] < 1
