# Plan: Część II, Word Embeddings, pkt 1: osadzenia słów word2vec

Treść punktu: *„Wykonać osadzenia słów w anotowanym korpusie (co najmniej 2 różne modele) [1 pkt]. a. Word2vec: i. dla języka polskiego dsmodels.nlp.ipipan.waw.pl, ii. dla języka angielskiego gensim.”*

Gałąź bazowa: `main` (`d29c890`). Gałąź robocza: `czesc2-w2v`.

## 0. Stan wyjściowy (2026-10-20)

| Element | Stan |
|---|---|
| Text Embeddings (osadzenia zdań, część II) | zrobione przez zespół (`src/text_embeddings.py`, notebook): fastText, USE (distiluse), TF-IDF; **word2vec nie był uruchamiany** (`W2V_PATH = None`) |
| Wczytywanie word2vec w kodzie | jest `text_embeddings._load_w2v(path)`. **Brak `limit`** (cały plik ~771 MB, kilka mln słów), brak cache |
| Wczytywanie fastText | `text_embeddings._load_fasttext`. Bez ścieżki bierze `sorted(bins)[0]` = **`kgr10.plain.cbow.dim100.neg10.bin`** (CBOW), a uwagi techniczne zalecają `kgr10.plain.skipgram.dim100.neg10.bin` |
| **Word Embeddings** (słowa, część II, 3 pkt) | **nie rozpoczęte** |
| Na tym komputerze | brak `gensim` (`pip` zainstaluje 4.4.0 bez zmiany wersji numpy 2.2.6); brak pobranych modeli; 475 GB wolnego dysku, 31 GB RAM |
| Zaanotowany korpus (teraz) | iteracje 1–2: **200 zdań, 2783 słowa (tokeny z literą), 1505 typów** (małe litery). Etykiety: A1/A2 na tokenach (`iterN/export/tokens_long.csv`), etykieta zdania `final` (`iterN/adjudicated.csv`) |

## 1. Zakres i założenia

- **„Dla języka angielskiego (gensim)” nie dotyczy naszego korpusu.** PolEmo jest polski, więc ten podpunkt pomijamy z jednym zdaniem uzasadnienia w raporcie. `gensim` i tak jest **biblioteką**, którą wczytujemy model IPI PAN (`KeyedVectors`), a opcjonalnie trenujemy własny model (p. 6).
- **„Co najmniej 2 różne modele”:** word2vec (ten plan) + **fastText KGR10**, którego funkcja wczytująca już jest. Moduł projektuję tak, żeby fastText był drugim wpisem tego samego rejestru modeli. Jego uruchomienie to osobny, krótki krok (p. 9, decyzja D1).
- **„W anotowanym korpusie”:** osadzamy **słowa występujące w zaanotowanych zdaniach** razem z informacją z anotacji: etykietą słowa i etykietą zdania. Dzięki temu pkt 2 (t-SNE z etykietami i podpowiedzią) i pkt 3 (k-najbliższych dla słów zaanotowanych) korzystają wprost z wyniku pkt 1.

## 2. Wybór modelu word2vec (IPI PAN)

Nazwy plików mają schemat `korpus-typ-zakres-wymiar-architektura-algorytm.txt.gz`. Rekomendacja: **`nkjp+wiki-forms-all-100-skipg-ns.txt.gz` (771 MB)**, plik wskazany w uwagach technicznych.

| Cecha | Wybór | Dlaczego |
|---|---|---|
| typ | **forms** (nie lemmas) | nasz korpus to formy wyrazowe; lematyzacja wymagałaby dodatkowego narzędzia (Morfeusz, spaCy-pl), którego nie mamy |
| zakres | **all** (nie restricted) | wersja `restricted` obejmuje tylko wybrane części mowy. Ryzyko: braki wektorów dla „nie”, „bardzo”, „trochę”, czyli dokładnie słów `ODW`, `WZM`, `OSL` |
| korpus | **nkjp+wiki** | największy, ogólny; NKJP dzieli formy tak samo jak PolEmo („była m”, „mieli śmy”), więc tokeny pasują (**do sprawdzenia** po pobraniu) |
| wymiar | **100** | zalecenie z uwag technicznych; ten sam wymiar co fastText dim100 |
| architektura / algorytm | **skipg-ns** | zalecenie z uwag technicznych. Alternatywa `cbow-ns` (759 MB) daje porównanie 1:1 z fastText CBOW, którego użył zespół (decyzja D1) |

Adres: `http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-skipg-ns.txt.gz` (format tekstowy word2vec, gzip).

## 3. Pobranie i wczytanie

1. Katalog `models/` (**dopisać do `.gitignore`**, bo GitHub nie przyjmie pliku 771 MB):
   ```bash
   mkdir -p models
   curl -L -o models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-skipg-ns.txt.gz
   ```
   Pobranie zajmuje ok. 5–15 min. Każda osoba pobiera model u siebie; repozytorium go nie zawiera.
2. Kontrola pliku: `gzip -t` (plik nie jest uszkodzony), a pierwszy wiersz to `<liczba_słów> 100`.
3. **Wczytanie z `limit=300000`** (zalecenie z uwag technicznych):
   - `KeyedVectors.load_word2vec_format(plik, binary=False, limit=300000)`,
   - plik word2vec jest posortowany według częstości, więc wczytujemy 300 tys. najczęstszych form, co daje ok. 120 MB RAM,
   - pierwsze wczytanie trwa ok. 1–3 min; potem **cache** `models/w2v_ipipan_300k.kv` (format gensim) wczytuje się w mniej niż 1 s.
4. **Wielkość liter:** po wczytaniu policzymy, jaki odsetek słów modelu zawiera wielkie litery. Od tego zależy kolejność wyszukiwania (p. 5).
5. Zmiana w kodzie zespołu: `_load_w2v(path, limit=config.W2V["limit"])` z cache. Ta sama funkcja posłuży osadzeniom zdań i słów, więc wektory będą spójne. **Do uzgodnienia z autorem `text_embeddings.py` (21bp37)**; zmiana jest wstecznie zgodna.

## 4. Słownik zaanotowanego korpusu

Źródło: `annotation/iter{1,2}/export/tokens_long.csv` (tokeny korpusu z etykietami A1/A2) i `iterN/adjudicated.csv` (etykieta zdania `final`). Później dojdzie iteracja 3 z pkt 5 Części I (ten sam kod, parametr `--iterations`).

- **Słowo** = token korpusu zawierający literę, zapisany małymi literami. Interpunkcję i same liczby pomijamy. Biorę tokeny z eksportu, a nie ponownie dzielony tekst, żeby każde wystąpienie było przypisane do swojej etykiety.
- **Kolumny `results/word_embeddings/vocab.csv`** (jeden wiersz na słowo):

| Kolumna | Znaczenie |
|---|---|
| `word` | forma małymi literami |
| `surface` | najczęstsza oryginalna pisownia (np. „Polecam”) |
| `freq`, `n_sentences` | liczba wystąpień i zdań w zaanotowanym korpusie |
| `polemo_freq` | liczba wystąpień w całym PolEmo (821 tys. tokenów), żeby odróżnić słowa rzadkie |
| `n_WYR_POZ` … `n_ODW`, `n_O` | liczba anotacji danej etykiety (A1 i A2 łącznie) |
| `annot_share` | odsetek anotacji innych niż O |
| `word_label` | etykieta słowa do kolorowania: najczęstsza etykieta różna od O, jeśli `annot_share` ≥ 0,5, inaczej `O` (decyzja D5) |
| `share_POZ` … `share_AMB` | rozkład etykiety `final` zdań, w których słowo wystąpiło |
| `w2v_in_vocab`, `w2v_form`, `w2v_rank` | czy model zna słowo, w jakiej pisowni je znaleziono, pozycja w rankingu częstości modelu |

## 5. Osadzenia word2vec

1. **Wyszukiwanie wektora:** kolejno `word` (małe litery), potem `surface` (oryginalna pisownia), potem pisownia z wielką literą na początku. Do `w2v_form` zapisujemy formę, którą znaleziono. Ta sama kolejność co w `AvgVectorEmbedder`, więc wyniki dla zdań i słów są spójne.
2. **Wynik:**
   - `results/word_embeddings/vectors_w2v.npz` (`words`, `vectors` float32 n×100) tylko dla słów korpusu. Plik ma ok. 0,6 MB, więc **trafia do repozytorium**: zespół może robić t-SNE (pkt 2) bez pobierania modelu,
   - `results/word_embeddings/coverage.json` i `coverage.md` z pokryciem słownika:
     - odsetek typów i tokenów, dla których model ma wektor,
     - **osobno dla słów z każdą etykietą** (np. ile słów `ODW` ma wektor),
     - 20 najczęstszych słów bez wektora z przyczyną: literówka, brak polskich znaków („przyjezdza”), rzadkie słowo, tokenizacja.

     Te liczby są potrzebne w dyskusji (pkt 4: „czy osadzenia dobrze odwzorowują znaczenie słów w korpusie dziedzinowym?”).
3. **Sprawdzenie, że wektory mają sens** (nie zastępuje pkt 3):
   - 5 najbliższych słów dla kilku słów kontrolnych: „dobry”, „polecam”, „lekarz”, „brud”, „bardzo”,
   - podobieństwa cos(dobry, zły) i cos(dobry, świetny), co pokazuje, że antonimy w word2vec są blisko (przyda się w pytaniu 2 dyskusji),
   - zapis do `sanity.md`.

## 6. Opcjonalnie: drugi word2vec wytrenowany w gensim na PolEmo (model dziedzinowy)

Tani dodatek (ok. 2 min treningu), który bezpośrednio odpowiada na pytania 1 i 4 dyskusji: czy osadzenia ogólne pasują do dziedziny i jak je poprawić.
- Dane: wszystkie 45 935 zdań PolEmo (`data/processed/polemo_sentences.parquet`), słowa małymi literami (`text_embeddings.tokens`).
- `gensim.models.Word2Vec(vector_size=100, sg=1, negative=5, window=5, min_count=3, epochs=15, seed=2026, workers=1)`. Parametry są dobrane pod porównanie z modelem IPI PAN (100 wymiarów, skip-gram, negative sampling). `workers=1` daje powtarzalny wynik.
- Wynik: `models/w2v_polemo.kv` (~15 MB, poza repozytorium, odtwarzalny), `vectors_w2v_polemo.npz` i kolumny `w2v_polemo_*` w `vocab.csv`.
- Zastrzeżenie: model widzi też zdania zaanotowane. To uczenie bez etykiet, więc nie ma wycieku etykiet, ale trzeba to zaznaczyć w raporcie.

## 7. Zmiany w kodzie

| Plik | Zmiana |
|---|---|
| `.gitignore` | `models/` |
| `config.py` | `MODELS_DIR`; `W2V = {"file": "nkjp+wiki-forms-all-100-skipg-ns.txt.gz", "url": …, "limit": 300_000}`; `WORD_EMB_DIR = RESULTS_DIR / "word_embeddings"`; parametry modelu z p. 6. Ścieżkę można nadpisać zmienną środowiskową `W2V_PATH`. |
| `src/text_embeddings.py` | `_load_w2v(path, limit)` + cache `.kv` (uzgodnić z 21bp37) |
| `src/word_embeddings.py` **(nowy)** | `annotated_vocabulary(iterations)`, `lookup(kv, word, surface)`, `embed_vocabulary(model_name)`, `coverage(...)`, `sanity_report(...)`, `train_polemo_w2v()`; rejestr modeli `{"w2v": …, "w2v_polemo": …, "fasttext": …}`; CLI |
| `tests/test_word_embeddings.py` **(nowy)** | testy bez pobierania modeli (p. 8) |
| `README_WordEmbeddings.md` | polecenia, opis plików wynikowych |

## 8. Testy (bez pobierania modeli)

- Mały model `KeyedVectors` budowany w pamięci (np. 6 słów, 4 wymiary):
  - kolejność wyszukiwania: małe litery, potem oryginał, potem wielka litera na początku,
  - brak wektora jest zgłaszany, a nie zamieniany po cichu na wektor zerowy.
- `annotated_vocabulary` na sztucznym `tokens_long.csv`:
  - liczności się zgadzają,
  - interpunkcja i liczby są pominięte,
  - reguła `word_label` (D5) działa, a remis ma ustalone rozstrzygnięcie,
  - `share_*` sumują się do 1.
- Pokrycie: znane liczby na małych danych; procenty dla poszczególnych etykiet.
- `.npz`: zapis i odczyt dają te same słowa i wektory, wymiar 100.
- `train_polemo_w2v` na mini-korpusie: dwa treningi z tym samym seedem dają identyczne wektory.
- `git check-ignore models/x` potwierdza, że modele nie trafią do repozytorium.
- Testy z prawdziwym modelem są **pomijane**, gdy brak pliku w `models/`. Gdy plik jest, sprawdzają:
  - wymiar 100, 300 000 słów,
  - słowa „nie”, „bardzo”, „polecam” są w słowniku,
  - słowniki IPI PAN i PolEmo (`vocab.csv`) mają tę samą liczbę wierszy.

## 9. Uruchomienie (po implementacji)

```bash
pip install -r requirements_embeddings.txt                 # gensim 4.4.0, plotly, …
curl -L -o models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-skipg-ns.txt.gz
python -m src.word_embeddings --models w2v                 # pkt 1: słownik, wektory, pokrycie, kontrola sensowności
python -m src.word_embeddings --models w2v w2v_polemo      # + opcjonalny model dziedzinowy (p. 6)
python -m src.word_embeddings --models w2v fasttext        # + drugi wymagany model (fastText KGR10, ~3,7 GB)
python -m pytest tests/test_word_embeddings.py -q
```

**Gotowe, gdy:**
- `vocab.csv` ma wiersz dla każdego słowa zaanotowanego korpusu (teraz 1505),
- `vectors_w2v.npz` zawiera wszystkie słowa znalezione w modelu,
- `coverage.md` podaje pokrycie ogółem i dla każdej etykiety oraz przykłady słów bez wektora,
- najbliższe słowa w `sanity.md` mają sens,
- testy przechodzą.

## 10. Powiązanie z dalszymi punktami części II (Word Embeddings)

| Pkt | Jak korzysta z pkt 1 |
|---|---|
| 2: t-SNE słów z etykietami, interaktywnie | `vectors_*.npz` + `vocab.csv` (`word_label`, `share_*`); użyjemy `text_embeddings.tsne_2d` i `make_tsne_figure` (podpowiedź: słowo, etykieta, liczba wystąpień) |
| 3: k-najbliższych słów, porównanie modeli | wymaga wczytanego modelu, bo szukamy w całym słowniku 300 tys. słów, a nie tylko w słowach korpusu; lista słów zaanotowanych z `vocab.csv` (`annot_share` > 0) |
| 4: dyskusja | pokrycie i słowa bez wektora (pytanie 1), kontrola sensowności i antonimy (pytanie 2), różnice IPI PAN / fastText / model PolEmo (pytania 3–4) |

## 11. Kolejność i czas

| Krok | Czas |
|---|---|
| decyzje D1–D6 | – |
| instalacja `gensim` i pobranie modelu (można zacząć od razu, niezależnie od reszty) | 5–15 min |
| `config`, `.gitignore`, `_load_w2v` z limitem i cache | 15 min |
| `word_embeddings.py` (słownik, wyszukiwanie, wektory, pokrycie, kontrola sensowności) + testy | ~1,5 h |
| opcjonalny model PolEmo (p. 6) | +20 min |
| uruchomienie, przegląd pokrycia, README | 20 min |

Uwaga o terminie: treść zadania podaje *„Czas wykonania: 2 tygodnie (oddanie lab 3)”* dla **całego** Zadania 1, czyli obu części. Jeśli to ok. 23.10, Word Embeddings (3 pkt) trzeba zrobić równolegle z domknięciem Części I.

## 12. Decyzje do potwierdzenia

| # | Pytanie | Opcje | Rekomendacja |
|---|---|---|---|
| D1 | Architektura modeli | (a) w2v **skipg-ns** + fastText **skipgram** dim100 (zgodnie z uwagami technicznymi; wymaga przypięcia pliku w `_load_fasttext`, a Text Embeddings warto policzyć ponownie, żeby były spójne); (b) w2v **cbow-ns** + fastText **cbow**, tak jak już użył zespół; bez zmian w Text Embeddings; (c) skipg-ns + cbow (różna architektura, porównanie miesza efekty) | **(a)**; (b), jeśli nie ma czasu na ponowne liczenie Text Embeddings |
| D2 | `limit` słownika word2vec | 300 tys. (uwagi techniczne) / 1 mln (więcej rzadkich słów, ok. 400 MB RAM) / bez limitu | **300 tys.**; jeśli pokrycie słów zaanotowanych spadnie poniżej ok. 95%, sprawdzić 1 mln i opisać różnicę |
| D3 | Własny word2vec na PolEmo (p. 6) | tak / nie | **tak**: tani i daje mocny materiał do dyskusji (pytania 1 i 4) |
| D4 | Zakres zdań | teraz iteracje 1–2 (200 zdań), ponownie po iteracji 3 / czekać na zbiór końcowy | **teraz iteracje 1–2**, ponowne uruchomienie po pkt 5 Części I (jedno polecenie) |
| D5 | Etykieta słowa do kolorowania | najczęstsza etykieta ≠ O, gdy `annot_share` ≥ 0,5 / dowolna anotacja ≠ O / tylko zgodne A1 i A2 | **≥ 0,5**: słowa zaznaczane sporadycznie nie dostają etykiety |
| D6 | Pliki w repozytorium | wektory słów korpusu (`.npz`, ~0,6 MB) i `vocab.csv` w repozytorium, modele (`models/`) poza nim / nic z wyników | **wektory i słownik w repozytorium, modele poza nim** |
