# Zadanie 1, Część I – anotacja PolEmo 2.0

Pełny plan: [PLAN_Zadanie1_Anotacja.md](PLAN_Zadanie1_Anotacja.md). Wszystkie polecenia uruchamiamy z katalogu `Lab1`.

## Status

| Krok | Co | Stan |
|---|---|---|
| 1 | wczytanie i przygotowanie korpusu (`src/data.py`) | gotowe |
| 2 | próbka iteracji 1 (`src/sampling.py`) | gotowe |
| 3 | wytyczne v1 (`annotation/guidelines/wytyczne_v1.md`) | gotowe |
| 4 | arkusze (`src/sheets_io.py`) | gotowe |
| 5 | anotacja iteracji 1 (A1, A2) | gotowe: `annotation/iter1_artefacts/` |
| 6 | **pkt 2 zadania: eksport i zgodność** (`src/export.py`, `src/agreement.py`) | gotowe: `annotation/iter1/export/`, `results/iter1/` |
| 7–8 | analiza rozbieżności, wytyczne v2, próbka iteracji 2 | gotowe (`iter1_artefacts/`, `wytyczne_v2.md`, `annotation/iter2/`) |
| 9–10 | anotacja iteracji 2, porównanie (pkt 4), raport | do zrobienia |

## Uruchomienie

```bash
pip install -r requirements.txt
python -m src.data                                     # krok 1 → data/processed/, results/corpus_stats.json
python -m src.sampling --iteration 1                   # krok 2 → annotation/iter1/sample.csv (+ _key_gold.csv)
python -m src.sheets_io --iteration 1 --guidelines v1  # krok 4 → annotation/iter1/to_annotate/iter1_A*.xlsx
python -m src.export --iteration 1                     # pkt 2 → annotation/iter1/export/
python -m src.agreement --iteration 1                  # pkt 2 → results/iter1/ (ok. 40 s: bootstrap)
python -m pytest tests -q                              # testy
```

`sampling` nie nadpisze istniejącej próbki bez `--force`, żeby nie zmienić zdań po rozpoczęciu anotacji.

## Korpus (krok 1)

- Źródło: `clarin-pl/polemo2-official`, plik `all.sentence.train.txt`, 45 974 linii. Etykietę odcina `rsplit(" __label__", 1)`.
- Usunięto 39 powtórzonych zdań, zostało 45 935 unikalnych. 6 zdań, których kopie mają różne etykiety gold, nie jest losowanych.
- Rozkład gold: NEG 36,8%, POZ 26,7%, NEU 24,5%, AMB 12,0%. Mediana długości to 15 tokenów, wliczając interpunkcję.
- Rama losowania (5–40 tokenów, bez konfliktów): 41 373 zdań.
- `sent_id` = numer linii w pliku źródłowym.

## Próbka iteracji 1 (krok 2)

100 zdań, po 25 z każdej klasy gold (seed 20261), długość od 5 do 36 tokenów.
Wykluczone są zdania użyte jako przykłady w wytycznych (`annotation/guidelines/example_ids.txt`).
Etykiety gold leżą tylko w `_key_gold.csv` i **nie trafiają do arkuszy**.

## Co zrobić z arkuszami (koordynator)

1. Wgraj `annotation/iter1/to_annotate/iter1_A1.xlsx` na Google Drive, otwórz w Google Sheets i wybierz *Plik → Zapisz jako Arkusze Google*.
2. **Sprawdź na tym jednym pliku**, czy działają listy rozwijane (kolumna `etykieta_zdania` i wiersze `etykieta`) oraz kolorowanie po wyborze etykiety.
3. Powtórz dla A2 i A3. **Każdy plik udostępnij tylko jego anotatorowi** jako edytorowi.
4. Każdemu anotatorowi wyślij też `annotation/guidelines/wytyczne_v1.md`, np. wklejone do dokumentu Google.
5. Po skończeniu anotator pobiera arkusz (*Plik → Pobierz → Microsoft Excel (.xlsx)*). Plik zapisujemy jako `annotation/iterN_artefacts/iterN_A1.xlsx` (odpowiednio A2), bo tam szuka go `src.export`.

Wczytanie wyników (krok 6) robi `sheets_io.read_annotations(path, sample, "A1")`. Funkcja zgłasza brakujące etykiety, edycje tokenów i etykiety spoza listy.

## Punkt 2: eksport i zgodność (iteracja 1)

**Eksport** (`python -m src.export --iteration 1`) czyta wypełnione arkusze `annotation/iter1_artefacts/iter1_A*.xlsx`
i zatrzymuje się, jeśli którykolwiek ma braki. Zapisuje w `annotation/iter1/export/`:

| Plik | Zawartość |
|---|---|
| `sentences_long.csv` | `sent_id, annotator, sent_label, uwagi` |
| `sentences_wide.csv` | `sent_id, text, A1, A2, gold, zgodne` |
| `tokens_long.csv` | `sent_id, tok_idx, token, annotator, label` (tok_idx od 1, jak t1..tK w arkuszu) |
| `tokens_wide.csv` | `sent_id, tok_idx, token, A1, A2, zgodne` |
| `spans_A1.jsonl`, `spans_A2.jsonl` | zdanie, jego etykieta i fragmenty `[start, end, etykieta, tekst]` (pozycje znaków) |

**Zgodność** (`python -m src.agreement --iteration 1`) zapisuje w `results/iter1/`:
- `agreement.json`: wszystkie miary z przedziałami ufności,
- `agreement_table.md`: tabele do raportu,
- `confusion_sentences.png`, `confusion_tokens.png`: macierze pomyłek A1 × A2.

| Poziom | n | Zgodność surowa | κ Cohena [95% CI] | κ Fleissa [95% CI] | Interpretacja |
|---|---|---|---|---|---|
| Zdania | 100 | 83,0% | 0,754 [0,636; 0,851] | 0,754 [0,635; 0,851] | znaczna |
| Tokeny – wszystkie | 1706 | 94,4% | 0,730 [0,655; 0,798] | 0,730 [0,655; 0,798] | znaczna |
| Tokeny – zaznaczone | 231 | 58,9% | 0,463 [0,347; 0,573] | 0,462 [0,343; 0,571] | umiarkowana |
| Zdania: A1, A2 + gold PolEmo (uzupełniająco) | 100 | – | – | 0,636 [0,536; 0,731] | znaczna |

- Przy dwóch anotatorach kappa Fleissa jest równa π Scotta, dlatego prawie pokrywa się z kappą Cohena.
- „Zaznaczone” to tokeny, którym co najmniej jedna osoba nadała etykietę inną niż O. Zgodne O zawyża wynik liczony na wszystkich tokenach.
- Najsłabsze etykiety (kappa dla każdej klasy osobno): `OSL` 0,26 i `WYR_NEG` 0,60. Wśród zdań najsłabsze jest `NEG` (0,66).
- Interpretacja według skali Landisa i Kocha. Przedziały ufności: bootstrap, 2000 losowań całych zdań (`config.BOOTSTRAP`).
- Dla punktu 4 te same polecenia uruchamiamy z `--iteration 2`, gdy arkusze trafią do `annotation/iter2_artefacts/`.

## Struktura

```
config.py                  stałe: etykiety, seedy, ścieżki
src/data.py                krok 1
src/sampling.py            krok 2
src/sheets_io.py           krok 4 (generacja) + odczyt wypełnionych arkuszy
src/export.py              pkt 2: eksport oetykietowanych danych
src/agreement.py           pkt 2: kappa Cohena i Fleissa, bootstrap, macierze pomyłek
tests/                     34 testy (test_steps_1_4.py, test_agreement.py)
annotation/guidelines/     wytyczne_v1.md, example_ids.txt
annotation/iter1/          sample.csv, _key_gold.csv, to_annotate/, export/
annotation/iter1_artefacts/ wypełnione arkusze A1, A2 + analiza rozbieżności
data/processed/            polemo_sentences.parquet
results/corpus_stats.json  statystyki korpusu
results/iter1/             zgodność iteracji 1
```
