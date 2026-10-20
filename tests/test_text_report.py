"""Testy plików LaTeX części Text Embeddings dla uśrednionego Word2Vec – bez kompilatora."""
import json
import re

import pytest

import config
from src import text_report as tr
from src.word_report import num, pct

REPORT = config.ROOT / "report"
GEN = REPORT / "generated"
SECTION = REPORT / "sections" / "te_w2v.tex"
MAIN = REPORT / "sprawozdanie.tex"
TABLES = ["te_w2v_probes.tex", "te_w2v_length.tex", "te_w2v_classif.tex"]
ALLOWED_EXTRA = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ„”")


def _strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def _texts() -> dict[str, str]:
    for p in (SECTION, GEN / "te_w2v_macros.tex", *(GEN / t for t in TABLES)):
        if not p.exists():
            pytest.skip(f"brak {p.name} (python -m src.text_report)")
    return {**{t: (GEN / t).read_text(encoding="utf-8") for t in TABLES},
            "te_w2v.tex": SECTION.read_text(encoding="utf-8")}


def test_syntax_braces_environments_and_characters():
    for name, tex in _texts().items():
        body = _strip_comments(tex).replace(r"\{", "").replace(r"\}", "")
        assert body.count("{") == body.count("}"), name
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
            if kind == "begin":
                stack.append(env)
            else:
                assert stack and stack.pop() == env, f"{name}: \\end{{{env}}} bez pary"
        assert not stack, name
        clean = re.sub(r"\\(label|ref|input|includegraphics|texttt)(\[[^\]]*\])?\{[^}]*\}", "", body)
        assert not re.search(r"(?<!\\)_", clean), f"{name}: niezabezpieczone _"
        odd = {c for c in clean if ord(c) > 127 and c not in ALLOWED_EXTRA}
        assert not odd, f"{name}: znaki spoza zestawu {sorted(odd)}"


def test_tables_have_consistent_columns():
    for name in TABLES:
        tex = _texts()[name]
        spec, content = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S).groups()
        n_cols = len(re.findall(r"[lrc]", spec))
        rows = [re.sub(r"\\(toprule|midrule|bottomrule)", "", r).strip() for r in content.split("\\\\")]
        rows = [r for r in rows if r]
        assert len(rows) >= 3
        for row in rows:
            assert row.count("&") + 1 == n_cols, f"{name}: {row[:60]}"


def test_every_macro_defined_and_values_match_analysis():
    texts = _texts()
    macros = dict(re.findall(r"\\newcommand\{\\(teW[A-Za-z]+)\}\{(.*)\}", (GEN / "te_w2v_macros.tex").read_text(encoding="utf-8")))
    for name, tex in {**texts, "sprawozdanie.tex": MAIN.read_text(encoding="utf-8")}.items():
        used = set(re.findall(r"\\(teW[A-Z][A-Za-z]*)", _strip_comments(tex)))
        assert used <= set(macros), f"{name}: niezdefiniowane {sorted(used - set(macros))}"
    a = json.loads((config.RESULTS_DIR / "text_embeddings" / "analiza_w2v.json").read_text(encoding="utf-8"))
    final = next(r for r in a["q5_eval"]["w2v_polemo"] if r["etykiety"] == "final")
    assert macros["teWAccPol"] == num(final["accuracy"])
    assert macros["teWTopOppIpi"] == pct(a["q1_opposite_top"]["w2v"]["top_opposite"], 1)
    assert macros["teWParaIpi"] == num(a["probe_means"]["w2v"]["parafraza"])
    assert macros["teWRecNegIpi"] == num(a["q5_recall"]["w2v"]["NEG"], 2)


def test_figures_and_references_resolve():
    texts = _texts()
    all_tex = MAIN.read_text(encoding="utf-8") + "".join(p.read_text(encoding="utf-8") for p in GEN.glob("*.tex"))
    all_tex += "".join(p.read_text(encoding="utf-8") for p in (REPORT / "sections").glob("*.tex"))
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    for name, tex in texts.items():
        for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
            assert (REPORT / "figures" / fig).exists(), f"brak rysunku {fig}"
        for ref in re.findall(r"\\ref\{([^}]+)\}", tex):
            assert ref in labels, f"{name}: brak etykiety {ref}"
        for inp in re.findall(r"\\input\{([^}]+)\}", tex):
            assert (REPORT / f"{inp}.tex").exists(), f"{name}: brak pliku {inp}"


def test_section_answers_five_questions_concisely():
    tex = _texts()["te_w2v.tex"]
    for i in range(1, 6):
        assert f"\\textbf{{{i}." in tex
    plain = re.sub(r"\\[a-zA-Z]+", " ", _strip_comments(tex))
    plain = re.sub(r"[{}$~]", " ", plain).replace("\\", " ")
    assert 350 <= len(re.findall(r"[^\W\d_]{2,}", plain)) <= 800


def test_report_hooks():
    main = MAIN.read_text(encoding="utf-8")
    assert r"\input{generated/te_w2v_macros}" in main and r"\input{sections/te_w2v}" in main
    assert r"\ref{sec:te-w2v}" in main
    build = (REPORT / "build_assets.py").read_text(encoding="utf-8")
    assert "build_text_embeddings_w2v()" in build.split("def main")[1]


def test_build_macros_on_real_results():
    if not (config.RESULTS_DIR / "text_embeddings" / "analiza_w2v.json").exists():
        pytest.skip("brak analiza_w2v.json")
    m = tr.build_macros(tr.load())
    assert all(re.fullmatch(r"teW[A-Za-z]+", k) for k in m)
    assert {"teWMilaIpi", "teWPolecamPol", "teWSpearmanFt", "teWBase"} <= set(m)
