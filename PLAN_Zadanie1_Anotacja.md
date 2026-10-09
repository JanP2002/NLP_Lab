# Plan: Zadanie 1, Część I – anotacja PolEmo 2.0 (punkty 1–4)

Zakres: wczytanie korpusu, wytyczne, anotacja (2 iteracje), kappa Cohena i Fleissa, porównanie iteracji.
Punkty 5–6 z PDF (anotacja reszty zbioru, statystyki opisowe) są poza zakresem, ale struktura je uwzględnia.

## 0. Decyzje projektowe (rekomendacje)

| Decyzja | Wybór | Uzasadnienie |
|---|---|---|
| Jednostka „tekstu” | **zdanie** z `all.sentence.train.txt` | zalecenie z uwag technicznych; mediana 12 słów, więc anotacja jest szybka |
| Etykieta całego tekstu (pkt 1b) | `POZ` / `NEG` / `NEU` / `AMB` | odpowiada etykietom PolEmo (`z_plus_m`, `z_minus_m`, `z_zero`, `z_amb`), więc można porównać z anotacją korpusu |
| Etykiety fragmentów (pkt 1c) | tokenowe: `WYR_POZ`, `WYR_NEG` (wyrażenia nacechowane) + `WZM`, `OSL`, `ODW` (modyfikatory: wzmocnienie, osłabienie, odwrócenie) | modyfikatory sugeruje PDF; wyrażenia polarne przydadzą się w Części II (t-SNE słów pozytywnych i negatywnych, k-najbliżsi dla słów zaanotowanych) |
| Narzędzie | **Google Sheets**, układ „siatki tokenów” (opis w kroku 4) | domyślne narzędzie z uwag technicznych. Doccano wymaga Pythona 3.11, a na tym komputerze jest tylko 3.10 i nie ma Dockera |
| Procedura | **2+1**: 3 osoby (A1, A2, A3) anotują **niezależnie całą próbkę** w obu iteracjach; A3 pełni też rolę superanotatora i rozstrzyga spory | trzech anotatorów potrzeba, żeby kappa Fleissa miała sens (dla 2 osób sprowadza się do π Scotta) |
| Próbka | **100 zdań na iterację**, losowanie warstwowe wg etykiety gold (25 na klasę), stały seed; iteracja 2 rozłączna z 1 | przy rozkładzie silnie niezrównoważonym kappa jest niestabilna („paradoks kappy”) |
| Rama losowania | zdania o długości 5–40 tokenów, po usunięciu duplikatów | przewidywalny czas anotacji; filtr zostanie opisany w raporcie |

## 1. Struktura folderu `Lab1/`

```
Lab1/
├── README.md                     # jak uruchomić, kolejność kroków
├── requirements.txt              # datasets, pandas, scikit-learn, statsmodels, openpyxl, matplotlib
├── config.py                     # SEED, N_SAMPLE, etykiety, ścieżki
├── src/
│   ├── data.py                   # krok 1: wczytanie, parsowanie, czyszczenie
│   ├── sampling.py               # krok 2: losowanie próbek (z wykluczeniem wcześniejszych id)
│   ├── sheets_io.py              # krok 4/6: generacja xlsx dla anotatorów + import eksportów
│   ├── agreement.py              # krok 6: Cohen, Fleiss, tokenowo, bootstrap, macierze pomyłek
│   └── compare.py                # krok 9: iteracja 1 vs 2
├── tests/test_agreement.py       # kappa na przykładach z podręcznika (walidacja implementacji)
├── data/processed/polemo_sentences.parquet
├── annotation/
│   ├── guidelines/wytyczne_v1.md, wytyczne_v2.md (+ changelog)
│   ├── iter1/  sample.csv (id, tekst – BEZ gold), key_gold.csv (ukryty),
│   │           to_annotate/A1.xlsx A2.xlsx A3.xlsx, exports/A1.xlsx …, adjudicated.csv
│   └── iter2/  (to samo)
├── results/iter1/, results/iter2/, results/comparison/
└── Zadanie1_Anotacja.ipynb       # raport do oddania: wywołuje src/, pokazuje tabele i wykresy
```

## 2. Kroki

### Krok 1 – wczytanie i przygotowanie korpusu [pkt 1, 1 pkt] – `src/data.py`
1. `load_dataset("text", data_files="hf://datasets/clarin-pl/polemo2-official/data/all.sentence.train.txt")`.
2. Podział każdej linii: `text, label = line.rsplit(" __label__", 1)`.
   **Uwaga:** separator ma **podwójne** podkreślniki (`__label__`). W skopiowanym tekście widać `_label_`, bo markdown zjadł podkreślniki. W PDF jest wersja poprawna.
3. Mapowanie `z_plus_m→POZ`, `z_minus_m→NEG`, `z_zero→NEU`, `z_amb→AMB`; asercja, że nie ma innych etykiet.
4. Czyszczenie: `strip`, usunięcie pustych linii i dokładnych duplikatów, kolumna `n_tokens = len(text.split())`.
5. **Tokenizacji nie zmieniamy** („Była m”, „mieli śmy”). Dzięki temu podział po spacjach daje dokładnie tokeny korpusu, a anotacja fragmentów mapuje się 1:1 na tokeny. W wytycznych trzeba to wyjaśnić anotatorom.
6. Stały identyfikator `sent_id` (numer linii), zapis do parquet. Podstawowe statystyki (rozkład etykiet, długości) przydadzą się też w pkt 6.

### Krok 2 – losowanie próbki – `src/sampling.py`
- `make_sample(df, n=100, seed, strata="gold", exclude_ids=...)`: losowanie warstwowe (25 na klasę) z ramy 5–40 tokenów.
- Wynik to `sample.csv` (tylko `sent_id`, `text`) i osobny `key_gold.csv`. **Anotatorzy nie widzą etykiet gold.**
- Iteracja 2 wywołuje to samo z `exclude_ids` = id z iteracji 1 i innym seedem. Daje to rozłączną próbkę o tej samej konstrukcji, więc obie iteracje da się porównać.

### Krok 3 – wytyczne v1 [pkt 1a] – `annotation/guidelines/wytyczne_v1.md`
Krótka notatka (1–2 strony) zawierająca:
- **cel** i jednostkę (pojedyncze zdanie oceniane bez kontekstu recenzji),
- **etykietę zdania**: definicje POZ/NEG/NEU/AMB, reguła rozstrzygania (np. „AMB tylko, gdy obecne są oba bieguny”), 2–3 przykłady na klasę z korpusu,
- **etykiety tokenów**: definicje 5 klas, zasada zakresu (zaznaczamy minimalny wyzwalacz, np. samo „nie”, „bardzo”, „polecam”; fragment wielotokenowy = kolejne tokeny z tą samą etykietą), przykłady,
- **artefakty tokenizacji**: „Była m” = „Byłam”, interpunkcja oddzielona spacją,
- **zasady niezależności**: brak konsultacji między anotatorami, brak zaglądania do etykiet korpusu, brak używania LLM, notowanie wątpliwości w kolumnie `uwagi`.

### Krok 4 – arkusze do anotacji [pkt 1b–1d] – `src/sheets_io.py`
Generacja przez `openpyxl` osobnego pliku xlsx dla każdego anotatora, w **innej losowej kolejności zdań** (ogranicza efekt kolejności). Każde zdanie zajmuje 2 wiersze:

| sent_id | etykieta zdania ▼ | wiersz | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | uwagi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1734 | `NEG` | tekst | Lekarz | bardzo | miły | , | ale | nie | polecam | . | |
| | | etykieta | | `WZM` | `WYR_POZ` | | | `ODW` | `WYR_POZ` | | |

- Listy rozwijane (walidacja danych) w kolumnie etykiety zdania i w wierszu etykiet tokenów; puste pole oznacza `O`.
- Zalety: anotacja na poziomie tekstu i fragmentów w jednym arkuszu, granice fragmentów są dokładne (bez dopasowywania przepisanych napisów), a zgodność tokenowa liczy się bezpośrednio.
- Pliki wgrywamy na Google Drive (konwersja xlsx→Sheets zachowuje listy rozwijane; trzeba to sprawdzić na jednym pliku). **Każdy anotator dostaje wyłącznie swój plik**, a po skończeniu pobiera go jako .xlsx do `iterN/exports/`.
- Przy medianie 12 tokenów szacunkowy czas to ok. 40 s na zdanie, czyli ok. 70 min na osobę na iterację.

### Krok 5 – anotacja, iteracja 1
Trzy osoby anotują niezależnie 100 zdań według wytycznych v1.

### Krok 6 – eksport i zgodność [pkt 2, 0.5 pkt] – `src/agreement.py`
Import: `read_annotations(xlsx) → DataFrame(sent_id, annotator, sent_label, token_labels[])`. Walidacja wykrywa braki, etykiety spoza listy i niezgodną liczbę tokenów.

**Poziom zdania:**
- **kappa Cohena** (`sklearn.metrics.cohen_kappa_score`) dla par A1–A2, A1–A3, A2–A3 oraz średnia,
- **kappa Fleissa** (`statsmodels.stats.inter_rater.aggregate_raters` + `fleiss_kappa`) dla trzech anotatorów,
- dodatkowo: zgodność surowa (% zgodności), kappa każdego anotatora względem gold PolEmo, kappa per klasa (jedna kontra reszta).

**Poziom tokenów (fragmentów):**
- Cohen i Fleiss na wszystkich tokenach. Tokeny `O` dominują i zawyżają wynik, dlatego liczymy też kappę na tokenach zaznaczonych przez co najmniej jednego anotatora oraz F1 fragmentów między parami (dopasowanie dokładne i częściowe).

**Niepewność i diagnostyka:**
- 95% przedział ufności metodą bootstrap (2000 losowań zdań ze zwracaniem) dla każdej kappy,
- macierze pomyłek dla par, tabela `disagreements_iter1.csv` (zdania z niezgodnością oraz uwagi anotatorów),
- interpretacja według skali Landisa i Kocha.

**Walidacja implementacji** (`tests/`): kappa Fleissa na przykładzie z artykułu Fleissa (1971) / Wikipedii, Cohen porównany z ręcznym wzorem.

### Krok 7 – analiza rozbieżności i wytyczne v2 [pkt 3, część 1]
1. Przegląd `disagreements_iter1.csv`, podział rozbieżności na typy (np. AMB↔NEU, brak kontekstu, ironia, zakres negacji, fakt kontra ocena: „Pokój miał 20 m2”, „raczej” jako osłabienie czy asekuracja).
2. A3 jako superanotator ustala wersję uzgodnioną iteracji 1 (`adjudicated.csv`: głosowanie większościowe, a przy remisie decyzja A3).
3. Plik `wytyczne_v2.md` z **changelogiem**, w którym każda zmiana odnosi się do konkretnego typu rozbieżności i ma przykład z iteracji 1.

### Krok 8 – anotacja, iteracja 2 [pkt 3, część 2, razem 1 pkt]
Nowa, rozłączna próbka 100 zdań (krok 2 z `exclude_ids`), ci sami anotatorzy, wytyczne v2, nowa losowa kolejność. Kroki 4–6 bez zmian w kodzie, inny jest tylko katalog `iter2/`.

### Krok 9 – porównanie iteracji [pkt 4, 0.5 pkt] – `src/compare.py`
- Tabela: Cohen (3 pary i średnia) oraz Fleiss, na poziomie zdań i tokenów, iteracja 1 obok iteracji 2, z przedziałami ufności.
- Δ = kappa₂ − kappa₁ z przedziałem bootstrap różnicy (próbki są niezależne, więc bootstrap liczymy osobno dla każdej i bierzemy różnicę).
- Wykres: kappa z przedziałami ufności dla obu iteracji, oddzielnie dla zdań i tokenów.
- Uczciwy komentarz: zmiana kappy łączy efekt nowych wytycznych, efekt uczenia się anotatorów i inną próbkę. Przy n=100 przedziały mają szerokość ok. ±0,1, więc małe różnice nie są rozstrzygające.

### Krok 10 – raport `Zadanie1_Anotacja.ipynb`
Sekcje odpowiadające punktacji: (1) korpus i przygotowanie, (2) wytyczne v1 i anotacja, (3) zgodność w iteracji 1, (4) rozbieżności → wytyczne v2 i iteracja 2, (5) porównanie. Na końcu krótkie podsumowanie i odnośnik do pkt 5–6.

## 3. Pułapki

- `__label__` (podwójne podkreślniki) zamiast `_label_`.
- Paradoks kappy przy niezrównoważonych klasach: losowanie warstwowe, raportowanie surowej zgodności i kappy per klasa.
- Tokenowa kappa zawyżona przez `O`: dodatkowa kappa na tokenach zaznaczonych i F1 fragmentów.
- Niezależność: osobne pliki, brak współdzielenia, ukryty gold, losowa kolejność zdań.
- Google Sheets: przed rozesłaniem sprawdzić, czy listy rozwijane przetrwały konwersję; eksportować z powrotem do .xlsx, nie do CSV (CSV gubi strukturę dwóch wierszy na zdanie).
- Brakujące pakiety: `openpyxl` (potrzebny w kroku 4). W Części II przydadzą się `gensim` i `plotly`.

## 4. Decyzje do potwierdzenia

1. **Anotatorzy:** kto jest A1, A2, A3? Rzetelna zgodność międzyanotatorska wymaga trzech osób. Gdyby rolę któregoś anotatora miał pełnić LLM, trzeba to uzgodnić z prowadzącym i jawnie opisać.
2. **Schemat fragmentów:** wyrażenia polarne + modyfikatory (rekomendacja) czy aspekty ABSA. Aspekty różnią się między dziedzinami PolEmo (hotele, medycyna, szkoła, produkty), więc wytyczne byłyby dłuższe.
3. **Narzędzie:** Google Sheets (rekomendacja) czy Doccano (wymaga instalacji Pythona 3.11; projekt typu „Intent Detection and Slot Filling” łączy etykietę tekstu z fragmentami).
4. **Wielkość próbki:** 100 zdań na iterację (ok. 70 min na osobę) czy inna.
