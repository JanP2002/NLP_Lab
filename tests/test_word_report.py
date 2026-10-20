"""Testy plików LaTeX części Word Embeddings – bez kompilatora (składnia, makra, odwołania, rysunki)."""
import json
import re

import pytest

import config
from src import word_report as wr

REPORT = config.ROOT / "report"
GEN = REPORT / "generated"
SECTION = REPORT / "sections" / "we_dyskusja.tex"
MAIN = REPORT / "sprawozdanie.tex"
WE_FILES = ["we_models.tex", "we_tsne.tex", "we_knn.tex"]
ALLOWED_EXTRA = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ„”")


def _need(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"brak {p.name} (uruchom python -m src.word_report)")


def _strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def _we_texts() -> dict[str, str]:
    _need(SECTION, *(GEN / f for f in WE_FILES))
    return {**{f: (GEN / f).read_text(encoding="utf-8") for f in WE_FILES},
            "we_dyskusja.tex": SECTION.read_text(encoding="utf-8")}


# --- formatowanie ------------------------------------------------------------------

def test_number_formatting_and_escaping():
    assert wr.num(0.6481) == "0{,}648" and wr.num(-0.035) == "$-$0{,}035"
    assert wr.pct(0.946, 1) == "94{,}6" and wr.pct(0.09) == "9"
    assert wr.integer(300000) == r"300\,000" and wr.integer(1505) == "1505"
    assert wr.esc("WYR_POZ & 50% #1") == r"WYR\_POZ \& 50\% \#1"


# --- składnia ------------------------------------------------------------------------

def test_braces_and_environments_balanced():
    for name, tex in _we_texts().items():
        body = _strip_comments(tex).replace(r"\{", "").replace(r"\}", "")
        assert body.count("{") == body.count("}"), name
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
            if kind == "begin":
                stack.append(env)
            else:
                assert stack and stack.pop() == env, f"{name}: \\end{{{env}}} bez pary"
        assert not stack, f"{name}: niezamknięte {stack}"


def test_no_unescaped_special_characters():
    for name, tex in _we_texts().items():
        body = _strip_comments(tex)
        body = re.sub(r"\\(label|ref|input|includegraphics|cite)(\[[^\]]*\])?\{[^}]*\}", "", body)
        assert not re.search(r"(?<!\\)_", body), f"{name}: niezabezpieczone _"
        assert not re.search(r"(?<!\\)#(?!\\)", body.replace(r"\#", "")), f"{name}: niezabezpieczone #"
        odd = {c for c in body if ord(c) > 127 and c not in ALLOWED_EXTRA}
        assert not odd, f"{name}: znaki spoza zestawu {sorted(odd)}"


def test_tables_have_consistent_columns():
    for name, tex in _we_texts().items():
        for spec, content in re.findall(r"\\begin\{tabular\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}(.*?)\\end\{tabular\}", tex, re.S):
            n_cols = len(re.findall(r"[lrc]|p\{", spec))
            for row in re.split(r"\\\\", content):
                row = re.sub(r"\\(toprule|midrule|bottomrule)", "", row).strip()
                if row:
                    assert row.count("&") + 1 == n_cols, f"{name}: {row[:60]}"


# --- makra, odwołania, rysunki ------------------------------------------------------------

def test_every_used_macro_is_defined():
    _need(GEN / "we_macros.tex")
    defined = set(re.findall(r"\\newcommand\{\\(we[A-Za-z]+)\}", (GEN / "we_macros.tex").read_text(encoding="utf-8")))
    texts = {**_we_texts(), "sprawozdanie.tex": MAIN.read_text(encoding="utf-8")}
    for name, tex in texts.items():
        used = set(re.findall(r"\\(we[A-Z][A-Za-z]*)", _strip_comments(tex)))
        assert used <= defined, f"{name}: niezdefiniowane {sorted(used - defined)}"


def test_macro_values_match_results():
    _need(GEN / "we_macros.tex", config.WORD_EMB_DIR / "coverage.json")
    macros = dict(re.findall(r"\\newcommand\{\\(we[A-Za-z]+)\}\{(.*)\}", (GEN / "we_macros.tex").read_text(encoding="utf-8")))
    cov = json.loads((config.WORD_EMB_DIR / "coverage.json").read_text(encoding="utf-8"))
    knn = json.loads((config.WORD_EMB_DIR / "knn_summary.json").read_text(encoding="utf-8"))
    sep = json.loads((config.WORD_EMB_DIR / "label_separation.json").read_text(encoding="utf-8"))
    assert macros["weCovTypesIpi"] == wr.pct(cov["coverage"]["w2v"]["types"]["share"], 1)
    assert macros["weOverlap"] == wr.pct(knn["pairs"][wr.PAIR]["wszystkie"]["overlap"])
    assert macros["wePurityIpi"] == wr.num(sep["w2v"][wr.WORD_SET]["purity"])
    assert macros["weProbeIpi"] == wr.num(sep["w2v"][wr.WORD_SET]["probe"]["balanced_accuracy"], 2)
    assert macros["weVocabPol"] == wr.integer(cov["vocab_size"]["w2v_polemo"])


def test_figures_refs_and_citations_resolve():
    texts = _we_texts()
    main = MAIN.read_text(encoding="utf-8")
    all_tex = "\n".join(texts.values()) + main
    for name, tex in texts.items():
        for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
            assert (REPORT / "figures" / fig).exists(), f"{name}: brak rysunku {fig}"
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    for name, tex in texts.items():
        for ref in re.findall(r"\\ref\{([^}]+)\}", tex):
            assert ref in labels, f"{name}: brak etykiety {ref}"
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", main))
    for name, tex in texts.items():
        for keys in re.findall(r"\\cite\{([^}]+)\}", tex):
            assert set(k.strip() for k in keys.split(",")) <= bib, f"{name}: brak pozycji {keys}"


def test_discussion_structure_and_length():
    tex = _we_texts()["we_dyskusja.tex"]
    for i in range(1, 5):
        assert f"\\textbf{{{i}." in tex
    plain = re.sub(r"\\[a-zA-Z]+", " ", _strip_comments(tex))
    plain = re.sub(r"[{}$~]", " ", plain).replace("\\", " ")
    assert 400 <= len(re.findall(r"[^\W\d_]{2,}", plain)) <= 750


def test_main_report_hooks():
    main = MAIN.read_text(encoding="utf-8")
    assert r"\input{generated/we_macros}" in main
    assert r"\input{sections/we_dyskusja}" in main
    assert "wnioski z części Word Embeddings" not in main
    for f in ("we_models", "we_tsne", "we_knn"):
        assert rf"\inputorTodo{{{f}}}" in main
