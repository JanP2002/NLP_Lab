"""Testy: uśrednione Word2Vec (pkt 1b, 2, 3) w notebooku Text Embeddings zespołu (``src/notebook_w2v.py``)."""
import base64

import numpy as np
import pandas as pd
import pytest

import config
from src import notebook_plotly as npl
from src import notebook_w2v as nw

nbformat = pytest.importorskip("nbformat")

TE_DIR = config.RESULTS_DIR / "text_embeddings"
# id komórki zespołu -> pierwsza komórka Word2Vec tuż po niej
ANCHORS = {"fbc01d64": "w2v-1-4-md", "8287da36": "w2v-2-1-md", "6e410e5f": "w2v-3-7-md"}


def _w2v(nb):
    return [c for c in nb.cells if nw.is_w2v(c)]


def _decode(a):
    if isinstance(a, dict) and "bdata" in a:
        return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
    return np.asarray(a, dtype=float)


def test_insert_is_idempotent_and_reversible():
    base = nw.strip_w2v(nbformat.read(nw.SOURCE_NB, as_version=4))
    once = nw.insert_w2v(base)
    twice = nw.insert_w2v(once)
    assert nw._dump(once) == nw._dump(twice)                     # ponowne wstawienie nie dubluje komórek
    assert nw._dump(nw.strip_w2v(once)) == nw._dump(base)        # usunięcie przywraca notebook zespołu
    nbformat.validate(once)


def test_answers_come_from_markdown_file():
    md = nw.answers_markdown()
    assert md.startswith("#### 3.7.6") and "\n## " not in md
    assert "**1. Czy są zdania" in md and "**5. Czy osadzenia nadają się" in md


def test_source_notebook_has_w2v_cells_without_outputs():
    nb = nbformat.read(nw.SOURCE_NB, as_version=4)
    cells = _w2v(nb)
    assert cells and all(not c.get("outputs") for c in cells if c.cell_type == "code")
    ids = [c.get("id") for c in nb.cells]
    for anchor, first in ANCHORS.items():
        assert ids[ids.index(anchor) + 1] == first
    assert nw.INTRO_NOTE in nb.cells[0].source


def test_executed_notebook_has_interactive_w2v_figures():
    nb = nbformat.read(nw.EXECUTED_NB, as_version=4)
    ids = [c.get("id") for c in nb.cells]
    for anchor, first in ANCHORS.items():
        assert ids[ids.index(anchor) + 1] == first
    code = [c for c in _w2v(nb) if c.cell_type == "code"]
    assert len(code) == 11 and all(c.execution_count for c in code)
    outs = [o for c in code for o in c.outputs]
    assert not [o for o in outs if o.output_type == "error"]
    figs = [o.data for o in outs if npl.PLOTLY_MIME in o.get("data", {})]
    assert len(figs) == 5 and all("Plotly.newPlot" in d["text/html"] for d in figs)   # oba formaty
    titles = [d[npl.PLOTLY_MIME]["layout"]["title"]["text"] for d in figs]
    assert sum(t.startswith("t-SNE") and "Word2Vec" in t for t in titles) == 2


@pytest.mark.parametrize("model", ["w2v", "w2v_polemo"])
def test_executed_tsne_matches_saved_results(model):
    """Punkty t-SNE w notebooku = results/text_embeddings/tsne_<model>.csv (te same liczby co w sprawozdaniu)."""
    nb = nbformat.read(nw.EXECUTED_NB, as_version=4)
    cell = next(c for c in nb.cells if c.get("id") == "w2v-2-1-tsne")
    label = "IPI PAN" if model == "w2v" else "PolEmo"
    fig = next(o.data[npl.PLOTLY_MIME] for o in cell.outputs if npl.PLOTLY_MIME in o.get("data", {})
               and label in o.data[npl.PLOTLY_MIME]["layout"]["title"]["text"])
    traces = [t for t in fig["data"] if t.get("legendgroup") == "final"]
    pts = {cd[6]: (x, y) for t in traces for cd, x, y in zip(t["customdata"], _decode(t["x"]), _decode(t["y"]))}
    assert all("final:" in t["hovertemplate"] and "customdata[0]" in t["hovertemplate"] for t in traces)
    ref = pd.read_csv(TE_DIR / f"tsne_{model}.csv")
    assert len(pts) == len(ref)
    got = np.array([pts[str(s)] for s in ref["sent_id"]])
    assert np.allclose(got, ref[["x", "y"]].to_numpy(), atol=1e-3)
