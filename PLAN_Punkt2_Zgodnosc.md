# Plan: punkt 2 zadania – eksport anotacji i zgodność (kappa Cohena i Fleissa)

Gałąź bazowa: `main` (`2a4c167`). Dane: iteracja 1, anotatorzy **A1** i **A2**, 100 wspólnych zdań.

## 0. Stan wyjściowy na `main`

| Element | Stan |
|---|---|
| Wypełnione arkusze | `annotation/iter1_artefacts/iter1_A1.xlsx`, `iter1_A2.xlsx`: 100/100 zdań, 0 problemów przy wczytaniu przez `sheets_io.read_annotations` |
| Wczytywanie arkuszy | jest (`src/sheets_io.py`, `read_annotations`) |
| Analiza rozbieżności (Igor) | `disagreements_sentences.csv` (17 zdań), `comparison_tokens.csv` (1706 tokenów, zgodność T/F), `analiza_rozbieznosci.xlsx` (decyzje sędziego, propozycje zmian) |
| Wytyczne v2, iteracja 2 | są |
| **Eksport danych w ustalonym formacie** | **brak** |
| **Kappa Cohena i Fleissa** | **brak**: nie ma ich ani w kodzie, ani w artefaktach |

Uwaga: `iter1_A2.xlsx` ma w zakładce *Instrukcja* wpis „Anotator: A3”, bo został wygenerowany jako arkusz A3, a potem przemianowany. Nie przeszkadza to w odczycie: nazwę anotatora podajemy przy wczytywaniu.

## 1. Założenia metodologiczne

- **Dwa poziomy zgodności** (zadanie wymaga anotacji zdań i fragmentów):
  - **zdania**: etykieta POZ / NEG / NEU / AMB, n = 100,
  - **tokeny**: etykieta każdego tokenu, puste pole = `O`, n = 1706. Ponieważ `O` dominuje i zawyża wynik, liczymy też zgodność tylko na **tokenach zaznaczonych** przez co najmniej jedną osobę (n = 231).
- **Kappa Cohena:** para A1–A2 (`sklearn.metrics.cohen_kappa_score`).
- **Kappa Fleissa:** A1 i A2 (`statsmodels.stats.inter_rater.fleiss_kappa`). **Przy 2 anotatorach kappa Fleissa jest tożsama z π Scotta.** Różni się od Cohena tylko tym, że zakłada wspólny rozkład etykiet obu osób, więc wartości są prawie równe. Trzeba to napisać w raporcie, żeby prowadzący nie uznał tego za błąd.
- **Uzupełniająco: Fleiss dla 3 ocen (A1, A2, gold PolEmo).** Etykieta korpusu to niezależna anotacja, więc daje sensownego „trzeciego oceniającego”. Zastrzeżenie: anotatorzy PolEmo widzieli całe recenzje, a my pojedyncze zdania. Decyzja sędziego **nie** nadaje się na trzeciego oceniającego, bo powstała po zobaczeniu obu anotacji i tylko dla 17 zdań.
- **Niepewność:** 95% przedział ufności metodą bootstrap. Losujemy **zdania** ze zwracaniem, 2000 razy. Przy tokenach losujemy całe zdania, a nie pojedyncze tokeny, bo tokeny w jednym zdaniu nie są niezależne. Przedziały przydadzą się w punkcie 4 (porównanie iteracji).
- **Interpretacja:** skala Landisa i Kocha (0,41–0,60 umiarkowana, 0,61–0,80 znaczna, 0,81–1 prawie pełna).

## 2. Zmiany w kodzie

| Plik | Zmiana |
|---|---|
| `config.py` | `export_dir(iteration)` → `annotation/iter{N}_artefacts/` (konwencja zespołu); `RESULTS_DIR = ROOT / "results"` |
| `src/export.py` **(nowy)** | eksport anotacji do formatów tabelarycznych (p. 3) |
| `src/agreement.py` **(nowy)** | kappa Cohena i Fleissa, zgodność surowa, przedziały bootstrap, macierze pomyłek, tabela do raportu (p. 4) |
| `tests/test_agreement.py` **(nowy)** | walidacja wzorów i zgodność z artefaktami Igora (p. 5) |
| `README.md` | polecenia i opis wyników |

Moduły korzystają z istniejącego `read_annotations` i nie wymagają zmian z gałęzi `copy1_branch`. Anotatorów wykrywamy z nazw plików `iter{N}_A*.xlsx` w `export_dir`, więc ten sam kod posłuży w iteracji 2 (punkt 4).

## 3. Eksport oetykietowanych danych (`src/export.py`)

Wynik w `annotation/iter1/export/`:

| Plik | Kolumny / format | Do czego |
|---|---|---|
| `sentences_long.csv` | `sent_id, annotator, sent_label, uwagi` | dane surowe, jedna ocena na wiersz |
| `sentences_wide.csv` | `sent_id, text, A1, A2, gold, zgodne` | przegląd i kappa na poziomie zdań |
| `tokens_long.csv` | `sent_id, tok_idx, token, annotator, label` | dane surowe tokenów |
| `tokens_wide.csv` | `sent_id, tok_idx, token, A1, A2, zgodne` | kappa na poziomie tokenów; odpowiednik `comparison_tokens.csv`, ale generowany kodem |
| `spans_A1.jsonl`, `spans_A2.jsonl` | `{"sent_id", "text", "label", "spans": [[start, end, "WYR_POZ", "bardzo miły"], …]}` | fragmenty (kolejne tokeny z tą samą etykietą) z pozycjami znaków, w formacie zbliżonym do Doccano; przyda się w Części II (słowa zaanotowane) |

Polecenie eksportu najpierw sprawdza każdy arkusz przez `read_annotations`. Jeśli arkusz ma choć jeden problem (brak etykiety, zmienione słowa), eksport się zatrzymuje.

## 4. Zgodność (`src/agreement.py`)

**Rdzeń (wymagany w punkcie 2):**

| Poziom | Miary |
|---|---|
| zdania (n = 100) | zgodność surowa, **κ Cohena** (A1–A2), **κ Fleissa** (A1, A2) |
| tokeny, wszystkie (n = 1706) | zgodność surowa, κ Cohena, κ Fleissa |
| tokeny zaznaczone (n = 231) | zgodność surowa, κ Cohena, κ Fleissa |

**Rozszerzenia (tanie, przydatne w punktach 3–4):**
- 95% przedziały bootstrap dla każdej kappy,
- κ Fleissa (A1, A2, gold) i κ Cohena każdego anotatora względem gold,
- κ dla każdej klasy osobno (klasa kontra reszta), żeby wskazać etykiety sprawiające problem,
- macierze pomyłek A1 × A2 (zdania i tokeny) jako PNG,
- interpretacja według skali Landisa i Kocha.

**Wyniki** w `results/iter1/`:
- `agreement.json`: wszystkie liczby, n, przedziały ufności i parametry (seed, liczba losowań),
- `agreement_table.md`: gotowa tabela do raportu,
- `confusion_sentences.png`, `confusion_tokens.png`.

## 5. Weryfikacja (`tests/test_agreement.py`)

1. **Wzory:** własna implementacja kappy Cohena i Fleissa (kilka linii) daje to samo co sklearn i statsmodels.
2. **Przykład z literatury:** przykład Fleissa z Wikipedii (10 elementów, 14 oceniających, 5 kategorii) daje κ = 0,210.
3. **Właściwości:** pełna zgodność daje κ = 1; zamiana nazw etykiet nie zmienia κ; przy 2 oceniających κ Fleissa = π Scotta.
4. **Fragmenty:** kolejne tokeny z tą samą etykietą łączą się w jeden fragment, a pozycje znaków wskazują dokładnie ten tekst.
5. **Bootstrap:** ten sam seed daje ten sam przedział, który zawiera wartość punktową.
6. **Zgodność z artefaktami zespołu:** na prawdziwych danych 17 zdań z rozbieżnością, czyli te same `sent_id` co w `disagreements_sentences.csv`, oraz 1611 z 1706 zgodnych tokenów, jak w `comparison_tokens.csv`.

## 6. Oczekiwane wartości

Policzone wstępnie (12.10) kodem z `main`. Implementacja musi je odtworzyć:

| Poziom | n | Zgodność surowa | κ Cohena | κ Fleissa |
|---|---|---|---|---|
| zdania | 100 | 83,0% | 0,754 | 0,754 |
| tokeny, wszystkie | 1706 | 94,4% | 0,730 | 0,730 |
| tokeny zaznaczone | 231 | 58,9% | 0,463 | 0,462 |
| zdania, A1 + A2 + gold (uzupełniająco) | 100 | – | – | 0,636 |

Wstępny wniosek do raportu: zgodność na poziomie zdań jest **znaczna** (0,61–0,80). Na poziomie fragmentów, mierzona na zaznaczonych tokenach, jest tylko **umiarkowana**, i to uzasadnia zmiany w wytycznych v2 dotyczące etykiet tokenów.

## 7. Uruchomienie (po implementacji)

```bash
python -m src.export --iteration 1        # → annotation/iter1/export/
python -m src.agreement --iteration 1     # → results/iter1/
python -m pytest tests -q
```

Dla punktu 4 te same polecenia wystarczy uruchomić z `--iteration 2`, gdy arkusze z iteracji 2 trafią do `annotation/iter2_artefacts/`.

## 8. Kolejność i czas

1. Gałąź robocza od `main`, np. `git switch -c pkt2-zgodnosc`, bo Igor też wypycha zmiany na `main`.
2. `config.py`, `src/export.py` i testy fragmentów: ok. 30 min.
3. `src/agreement.py` (rdzeń, rozszerzenia, wykresy) i testy: ok. 45 min.
4. Uruchomienie, porównanie z p. 6 i README: ok. 15 min.
5. Merge do `main` po przejrzeniu.

## 9. Decyzje do potwierdzenia

1. **Rozszerzenia z p. 4:** wszystkie (rekomendacja; przedziały ufności są potrzebne w punkcie 4) czy tylko rdzeń.
2. **Fleiss z gold jako trzecim oceniającym:** dodać jako wynik uzupełniający (rekomendacja) czy pominąć.
3. **Miejsce eksportu:** `annotation/iter1/export/` (rekomendacja) czy `annotation/iter1_artefacts/` obok plików Igora.
