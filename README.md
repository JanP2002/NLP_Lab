# Zadanie 1, Część I – anotacja PolEmo 2.0

Pełny plan: [PLAN_Zadanie1_Anotacja.md](PLAN_Zadanie1_Anotacja.md). Wszystkie polecenia uruchamiamy z katalogu `Lab1`.

## Status

| Krok | Co | Stan |
|---|---|---|
| 1 | wczytanie i przygotowanie korpusu (`src/data.py`) | gotowe |
| 2 | próbka iteracji 1 (`src/sampling.py`) | gotowe |
| 3 | wytyczne v1 (`annotation/guidelines/wytyczne_v1.md`) | gotowe |
| 4 | arkusze dla A1–A3 (`src/sheets_io.py`) | gotowe, czekają na wgranie do Google Sheets |
| 5 | anotacja iteracji 1 | do zrobienia (anotatorzy) |
| 6–10 | zgodność, wytyczne v2, iteracja 2, porównanie, raport | do zrobienia |

## Uruchomienie

```bash
pip install -r requirements.txt
python -m src.data                                     # krok 1 → data/processed/, results/corpus_stats.json
python -m src.sampling --iteration 1                   # krok 2 → annotation/iter1/sample.csv (+ _key_gold.csv)
python -m src.sheets_io --iteration 1 --guidelines v1  # krok 4 → annotation/iter1/to_annotate/iter1_A*.xlsx
python -m pytest tests -q                              # testy kroków 1–4
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
5. Po skończeniu anotator pobiera arkusz (*Plik → Pobierz → Microsoft Excel (.xlsx)*). Plik zapisujemy jako `annotation/iter1/exports/iter1_A1.xlsx` (odpowiednio A2, A3).

Wczytanie wyników (krok 6) robi `sheets_io.read_annotations(path, sample, "A1")`. Funkcja zgłasza brakujące etykiety, edycje tokenów i etykiety spoza listy.

## Struktura

```
config.py                  stałe: etykiety, seedy, ścieżki
src/data.py                krok 1
src/sampling.py            krok 2
src/sheets_io.py           krok 4 (generacja) + odczyt eksportów
tests/test_steps_1_4.py    12 testów
annotation/guidelines/     wytyczne_v1.md, example_ids.txt
annotation/iter1/          sample.csv, _key_gold.csv, to_annotate/, exports/
data/processed/            polemo_sentences.parquet
results/corpus_stats.json  statystyki korpusu
```
