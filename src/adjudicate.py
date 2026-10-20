"""Rozstrzyganie konfliktów anotacji na poziomie zdań (superanotator).

Zasada: A1 == A2 -> etykieta uzgodniona automatycznie; A1 != A2 -> decyduje superanotator.
Superanotator NIE widzi etykiety gold PolEmo (procedura niezależna), widzi tylko tekst i etykiety A1, A2.

Krok 1 – arkusz dla superanotatora (tylko zdania z konfliktem):
    python -m src.adjudicate --iteration 2 --make-sheet
    -> annotation/iter2/to_adjudicate.xlsx  (z listą rozwijaną w kolumnie `final`; polecany, bez problemów z Excelem)
    -> annotation/iter2/to_adjudicate.csv   (kolumny: sent_id, text, A1, A2, final, uwagi)
    Superanotator wypełnia kolumnę `final` (POZ / NEG / NEU / AMB). Przy scalaniu czytany jest nowszy z obu plików
    (CSV może mieć separator , lub ; – wykrywany automatycznie). Jeśli ich nie ma, a istnieje
    annotation/iterN_artefacts/analiza_rozbieznosci.xlsx (format z iteracji 1), decyzje są czytane z kolumny „Decyzja sędziego”.

Krok 2 – scalenie:
    python -m src.adjudicate --iteration 2 --merge
    -> annotation/iter2/adjudicated.csv  (kolumny: sent_id, final, zrodlo)
       zrodlo = "zgodne" (A1 == A2) albo "superanotator"

Wymaga eksportu: python -m src.export --iteration N  (annotation/iterN/export/sentences_wide.csv).
Rozstrzyga tylko etykiety zdań (to wystarcza do Text Embeddings); rozbieżności tokenowe nie są tu ruszane.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config


def _load_wide(iteration: int) -> pd.DataFrame:
    path = config.export_dir(iteration) / "sentences_wide.csv"
    if not path.exists():
        raise FileNotFoundError(f"Brak {path}. Najpierw: python -m src.export --iteration {iteration}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("A1", "A2"):
        bad = sorted(set(df[col].dropna()) - set(config.SENT_LABELS))
        if bad or df[col].isna().any():
            raise ValueError(f"{path}: kolumna {col} ma braki lub niedozwolone etykiety {bad}")
    return df


def sheet_path(iteration: int) -> Path:
    return config.iter_dir(iteration) / "to_adjudicate.csv"


def sheet_path_xlsx(iteration: int) -> Path:
    return config.iter_dir(iteration) / "to_adjudicate.xlsx"


def write_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Arkusz z listą rozwijaną etykiet w kolumnie `final`, zawijaniem tekstu i szerokościami kolumn."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "do_rozstrzygniecia"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([("" if (isinstance(v, float) and pd.isna(v)) else v) for v in row])
    widths = {"sent_id": 9, "text": 70, "A1": 7, "A2": 7, "final": 10, "propozycja": 12,
              "uzasadnienie": 55, "uwagi": 40}
    for i, col in enumerate(df.columns, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 16)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    if "final" in df.columns:
        col = list(df.columns).index("final") + 1
        letter = ws.cell(1, col).column_letter
        dv = DataValidation(type="list", formula1='"' + ",".join(config.SENT_LABELS) + '"',
                            allow_blank=True, showErrorMessage=True,
                            errorTitle="Etykieta", error="Dozwolone: " + ", ".join(config.SENT_LABELS))
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{ws.max_row}")
        for r in range(2, ws.max_row + 1):
            ws.cell(r, col).fill = PatternFill("solid", fgColor="FFF2CC")
    wb.save(path)


def adjudicated_path(iteration: int) -> Path:
    return config.iter_dir(iteration) / "adjudicated.csv"


def analiza_path(iteration: int) -> Path:
    """Arkusz analizy rozbieżności w formacie z iteracji 1 (annotation/iterN_artefacts/)."""
    return config.filled_sheets_dir(iteration) / "analiza_rozbieznosci.xlsx"


def read_sheet(iteration: int, include_analiza: bool = True) -> pd.DataFrame:
    """Najnowszy z arkuszy rozstrzygnięć: to_adjudicate.xlsx / .csv albo (jak w iteracji 1) analiza_rozbieznosci.xlsx.

    CSV: separator , lub ; wykrywany po nagłówku. W analiza_rozbieznosci.xlsx kolumna „Decyzja sędziego”
    to `final`, a „Uzasadnienie decyzji” to `uwagi`.
    """
    cands = [sheet_path_xlsx(iteration), sheet_path(iteration)]
    if include_analiza:
        cands.append(analiza_path(iteration))
    cands = [p for p in cands if p.exists()]
    if not cands:
        raise FileNotFoundError(f"Brak arkusza rozstrzygnięć w {config.iter_dir(iteration)} ani "
                                f"{analiza_path(iteration)}. Najpierw: "
                                f"python -m src.adjudicate --iteration {iteration} --make-sheet")
    path = max(cands, key=lambda p: p.stat().st_mtime)
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        head = path.read_text(encoding="utf-8-sig").splitlines()[0]
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, sep=";" if head.count(";") > head.count(",") else ",")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Decyzja sędziego": "final", "Uzasadnienie decyzji": "uwagi"})
    print(f"czytam arkusz: {path.name}")
    return df


def make_sheet(iteration: int, force: bool = False) -> Path:
    df = _load_wide(iteration)
    conflicts = df[df["A1"] != df["A2"]]
    out = sheet_path(iteration)
    for existing in (out, sheet_path_xlsx(iteration)):
        if existing.exists() and not force:
            raise FileExistsError(f"{existing} już istnieje (może zawierać pracę superanotatora). "
                                  "Użyj --force, żeby nadpisać.")
    sheet = conflicts[["sent_id", "text", "A1", "A2"]].copy()
    sheet["final"] = ""
    sheet["uwagi"] = ""
    sheet.to_csv(out, index=False, encoding="utf-8-sig")
    write_xlsx(sheet, sheet_path_xlsx(iteration))
    print(f"iteracja {iteration}: {len(df)} zdań, zgodnych {len(df) - len(conflicts)}, "
          f"konfliktów {len(conflicts)} -> {sheet_path_xlsx(iteration)} (+ {out.name})")
    return out


def merge(iteration: int) -> Path:
    df = _load_wide(iteration)
    agreed = df[df["A1"] == df["A2"]][["sent_id", "A1"]].rename(columns={"A1": "final"})
    agreed["zrodlo"] = "zgodne"
    conflicts = df[df["A1"] != df["A2"]]

    if len(conflicts):
        sheet = read_sheet(iteration)
        sheet["sent_id"] = sheet["sent_id"].astype(int)
        sheet["final"] = sheet["final"].fillna("").str.strip().str.upper()
        sheet = sheet.drop_duplicates("sent_id").set_index("sent_id")

        missing = [i for i in conflicts["sent_id"] if i not in sheet.index or sheet.at[i, "final"] == ""]
        if missing:
            raise ValueError(f"{len(missing)} konfliktów bez decyzji superanotatora, np. sent_id {missing[:10]}")
        bad = sorted(set(sheet.loc[conflicts["sent_id"], "final"]) - set(config.SENT_LABELS))
        if bad:
            raise ValueError(f"Niedozwolone etykiety {bad}; dozwolone: {config.SENT_LABELS}")
        resolved = pd.DataFrame({"sent_id": conflicts["sent_id"].to_numpy(),
                                 "final": sheet.loc[conflicts["sent_id"], "final"].to_numpy(),
                                 "zrodlo": "superanotator"})
    else:
        resolved = pd.DataFrame(columns=["sent_id", "final", "zrodlo"])

    out = pd.concat([agreed, resolved], ignore_index=True).sort_values("sent_id")
    path = adjudicated_path(iteration)
    out.to_csv(path, index=False, encoding="utf-8")
    sup = out[out["zrodlo"] == "superanotator"]
    print(f"iteracja {iteration}: {len(out)} zdań -> {path}")
    print(f"  zgodne: {(out['zrodlo'] == 'zgodne').sum()}, rozstrzygnięte przez superanotatora: {len(sup)}")
    if len(sup):
        # po czyjej stronie stanął superanotator (informacja do raportu)
        m = conflicts.set_index("sent_id").loc[sup["sent_id"]]
        f = sup.set_index("sent_id")["final"]
        print(f"  superanotator = A1: {(f == m['A1']).sum()}, = A2: {(f == m['A2']).sum()}, "
              f"inna etykieta: {((f != m['A1']) & (f != m['A2'])).sum()}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iteration", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-sheet", action="store_true", help="arkusz konfliktów dla superanotatora")
    g.add_argument("--merge", action="store_true", help="scal zgodne + decyzje superanotatora -> adjudicated.csv")
    ap.add_argument("--force", action="store_true", help="nadpisz istniejący arkusz")
    a = ap.parse_args()
    make_sheet(a.iteration, a.force) if a.make_sheet else merge(a.iteration)


if __name__ == "__main__":
    main()
