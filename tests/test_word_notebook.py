"""Testy notebooka Word Embeddings: wykresy t-SNE słów mają być interaktywne w Jupyterze (pkt 2)."""
import pytest

import config

nbformat = pytest.importorskip("nbformat")

NB = config.ROOT / "notebooks" / "Word_Embeddings.ipynb"
EXECUTED = config.ROOT / "notebooks" / "Word_Embeddings_executed.ipynb"
PLOTLY_MIME = "application/vnd.plotly.v1+json"
FIGURES = [("w2v", "all"), ("w2v", "annotated"), ("w2v_polemo", "all"), ("w2v_polemo", "annotated")]


def _code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _plotly_outputs(nb) -> list[dict]:
    return [o["data"] for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            if PLOTLY_MIME in o.get("data", {})]


def test_source_notebook_is_valid_clean_and_interactive():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    code = _code(nb)
    # mimetype: JupyterLab / Notebook 7 / VS Code; notebook_connected: klasyczny Notebook i eksport HTML
    assert 'pio.renderers.default = "plotly_mimetype+notebook_connected"' in code
    for model, subset in FIGURES:
        assert f'show_tsne("{model}", "{subset}")' in code
    assert ".show()" in code and "make_word_figure" in code
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")   # źródło bez wyników
    assert "RECOMPUTE = False" in code                                           # domyślnie bez modeli


def _check_figures(nb):
    figs = _plotly_outputs(nb)
    assert len(figs) == len(FIGURES)
    for data in figs:
        fig = data[PLOTLY_MIME]
        traces = [t for t in fig["data"] if t.get("legendgroup") == "word_label"]
        assert traces and all("%{customdata}" in t["hovertemplate"] for t in traces)
        hover = traces[-1]["customdata"][0]
        assert hover.startswith("<b>") and "etykieta słowa:" in hover and "kontekst zdań:" in hover
        assert [b["label"] for b in fig["layout"]["updatemenus"][0]["buttons"]] == [
            "kolor: etykieta słowa", "kolor: kontekst zdań"]
        assert "Plotly.newPlot" in data["text/html"]                 # wersja dla klasycznego Notebooka / HTML
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors


def test_executed_notebook_contains_interactive_figures():
    if not EXECUTED.exists():
        pytest.skip("brak Word_Embeddings_executed.ipynb")
    _check_figures(nbformat.read(EXECUTED, as_version=4))


def test_notebook_runs_end_to_end():
    nbclient = pytest.importorskip("nbclient")
    if not (config.WORD_EMB_DIR / "vectors_w2v.npz").exists():
        pytest.skip("brak wyników pkt 1 (results/word_embeddings)")
    nb = nbformat.read(NB, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(NB.parent)}}).execute()
    _check_figures(nb)
