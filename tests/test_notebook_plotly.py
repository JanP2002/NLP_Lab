"""Testy: wykresy Plotly w notebookach rysują się w JupyterLab / Notebook 7 / VS Code (format mimetype)."""
import pytest

import config
from src import notebook_plotly as npl

nbformat = pytest.importorskip("nbformat")
go = pytest.importorskip("plotly.graph_objects")
pio = pytest.importorskip("plotly.io")

TE_NB = config.ROOT / "notebooks" / "Text_Embeddings.ipynb"
TE_EXECUTED = config.ROOT / "notebooks" / "Text_Embeddings_executed.ipynb"


def _connected_html(fig) -> str:
    """HTML z renderera notebook_connected – tak zapisane są wykresy w wykonanym notebooku zespołu."""
    return pio.renderers["notebook_connected"].to_mimebundle(fig.to_dict())["text/html"]


def _fig():
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode="markers", name="POZ",
                                customdata=["a", "b", "c"], hovertemplate="%{customdata}<extra></extra>"),
                     layout=dict(title=dict(text="test"), height=300))


def test_figures_from_html_recovers_figure():
    figs = npl.figures_from_html(_connected_html(_fig()))
    assert len(figs) == 1
    f = go.Figure(data=figs[0]["data"], layout=figs[0]["layout"])
    assert f.layout.title.text == "test" and f.data[0].name == "POZ"
    assert list(f.data[0].customdata) == ["a", "b", "c"]
    assert figs[0]["config"].get("responsive") is True


def test_add_plotly_mimetype_is_idempotent_and_selective():
    two = _connected_html(_fig()) + _connected_html(_fig())
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("fig.show()", outputs=[
        nbformat.v4.new_output("display_data", data={"text/html": _connected_html(_fig())}),
        nbformat.v4.new_output("display_data", data={"text/html": "<b>tabela</b>", "text/plain": "x"}),
        nbformat.v4.new_output("display_data", data={"text/html": two}),          # 2 wykresy w jednym wyniku
    ])])
    assert npl.add_plotly_mimetype(nb) == 1
    outs = nb.cells[0].outputs
    assert npl.PLOTLY_MIME in outs[0].data and "text/html" in outs[0].data       # HTML zostaje
    assert npl.PLOTLY_MIME not in outs[1].data and npl.PLOTLY_MIME not in outs[2].data
    assert npl.add_plotly_mimetype(nb) == 0
    nbformat.validate(nb)


def test_text_embeddings_notebooks_render_in_jupyterlab():
    if not TE_EXECUTED.exists():
        pytest.skip("brak notebooków Text Embeddings")
    for path in (TE_NB, TE_EXECUTED):
        src = "\n".join(c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code")
        assert 'pio.renderers.default = "plotly_mimetype+notebook_connected"' in src, path.name
    nb = nbformat.read(TE_EXECUTED, as_version=4)
    html_figs = mime_figs = 0
    for c in nb.cells:
        for o in c.get("outputs", []):
            data = o.get("data", {})
            html_figs += "Plotly.newPlot" in data.get("text/html", "")
            if npl.PLOTLY_MIME in data:
                mime_figs += 1
                go.Figure(data=data[npl.PLOTLY_MIME]["data"], layout=data[npl.PLOTLY_MIME]["layout"])
    assert html_figs == mime_figs == 7
    assert npl.add_plotly_mimetype(nb) == 0                                      # nic do uzupełnienia


def test_local_text_embeddings_run_has_native_interactive_figures():
    """Lokalne ponowne uruchomienie (fastText CBOW jak u zespołu, USE = distiluse) – wykresy w obu formatach."""
    path = config.ROOT / "notebooks" / "Text_Embeddings_local_executed.ipynb"
    if not path.exists():
        pytest.skip("brak Text_Embeddings_local_executed.ipynb")
    nb = nbformat.read(path, as_version=4)
    assert "Wersja uruchomiona lokalnie" in nb.cells[1].source                  # notka o różnicach
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert 'FASTTEXT_PATH = "models/kgr10.plain.cbow.dim100.neg10.bin"' in src
    assert 'te.OUT_DIR = LOCAL_OUT' in src                                       # bez nadpisywania plików zespołu
    outs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert not [o for o in outs if o.output_type == "error"]
    figs = [o.data for o in outs if npl.PLOTLY_MIME in o.get("data", {})]
    assert len(figs) == 7 and all("Plotly.newPlot" in d["text/html"] for d in figs)
    titles = [d[npl.PLOTLY_MIME]["layout"]["title"]["text"] for d in figs]
    assert sum("t-SNE" in t for t in titles) == 3 and any("distiluse" in t for t in titles)
