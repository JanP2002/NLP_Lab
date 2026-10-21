"""Wykresy Plotly w zapisanych notebookach: dodaje format rysowany przez JupyterLab / Notebook 7 / VS Code.

Notebook wykonany z ``pio.renderers.default = "notebook_connected"`` zapisuje wykresy tylko jako ``text/html``
ze skryptem. JupyterLab i Notebook 7 nie uruchamiają skryptów w wynikach niezaufanego notebooka (np. pobranego
z repozytorium), więc wykres jest pusty. Ten skrypt odczytuje z HTML wywołanie ``Plotly.newPlot(id, data,
layout, config)`` i dopisuje do tego samego wyniku format ``application/vnd.plotly.v1+json`` – te same dane,
bez ponownego uruchamiania notebooka (wyniki się nie zmieniają). Ponowne wywołanie niczego nie dubluje.

    python -m src.notebook_plotly notebooks/Text_Embeddings_executed.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PLOTLY_MIME = "application/vnd.plotly.v1+json"
_decoder = json.JSONDecoder()


def _next_json(s: str, pos: int):
    """Następna wartość JSON od ``pos`` (pomija spacje i przecinki); zwraca (wartość, pozycja za nią)."""
    while s[pos] in " \t\r\n,":
        pos += 1
    return _decoder.raw_decode(s, pos)


def figures_from_html(html: str) -> list[dict]:
    """Figury (data, layout, config) z wywołań ``Plotly.newPlot`` w HTML wygenerowanym przez Plotly."""
    figs, start = [], 0
    while (i := html.find("Plotly.newPlot(", start)) >= 0:
        pos = i + len("Plotly.newPlot(")
        _, pos = _next_json(html, pos)            # id elementu (napis)
        data, pos = _next_json(html, pos)
        layout, pos = _next_json(html, pos)
        config, pos = _next_json(html, pos)
        figs.append({"data": data, "layout": layout, "config": config})
        start = pos
    return figs


def add_plotly_mimetype(nb) -> int:
    """Dopisuje format Plotly do każdego wyniku z jednym wykresem w HTML; zwraca liczbę uzupełnionych wyników."""
    added = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if PLOTLY_MIME in data or "Plotly.newPlot" not in data.get("text/html", ""):
                continue
            figs = figures_from_html(data["text/html"])
            if len(figs) == 1:
                data[PLOTLY_MIME] = figs[0]
                added += 1
    return added


def fix_notebook(path: Path) -> int:
    import nbformat

    nb = nbformat.read(path, as_version=4)
    added = add_plotly_mimetype(nb)
    if added:
        nbformat.validate(nb)
        nbformat.write(nb, path)
    return added


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="+", type=Path)
    for path in ap.parse_args(argv).notebooks:
        print(f"{path}: uzupełniono {fix_notebook(path)} wykresów")


if __name__ == "__main__":
    main()
