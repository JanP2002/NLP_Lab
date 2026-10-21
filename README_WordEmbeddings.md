# Zadanie 1, Część II – Word Embeddings, pkt 1: osadzenia słów (word2vec)

Plan i decyzje: [PLAN_Czesc2_Word2Vec.md](PLAN_Czesc2_Word2Vec.md). Polecenia uruchamiamy z katalogu głównego repozytorium.

## Notebook z wykresami interaktywnymi (pkt 1–4)

[`notebooks/Word_Embeddings.ipynb`](notebooks/Word_Embeddings.ipynb) wyświetla pod komórkami **interaktywne wykresy
t-SNE słów** (pkt 2). Po najechaniu na punkt widać słowo i jego anotację, lista rozwijana zmienia kolorowanie,
przycisk „podpisy” pokazuje napisy przy słowach. Notebook pokazuje też wyniki pkt 1 i 3 oraz skrót dyskusji (pkt 4).
Czyta zapisane wyniki z `results/word_embeddings/`, więc **nie wymaga pobierania modeli**; „Run All” trwa ok. 40 s.

```bash
pip install -r requirements_embeddings.txt          # jupyter, plotly, gensim, …
jupyter lab notebooks/Word_Embeddings.ipynb         # albo: jupyter notebook …, albo otwarcie w VS Code
```

* **Wyświetlanie wykresów.** Notebook ustawia `pio.renderers.default = "plotly_mimetype+notebook_connected"`:
  * format `plotly_mimetype` rysują JupyterLab, Notebook 7 i VS Code,
  * format `notebook_connected` (plotly.js z CDN, potrzebny internet) rysuje klasyczny Notebook i eksport HTML.

  Bez tego ustawienia Plotly poza Jupyterem otwiera wykres w przeglądarce zamiast pod komórką.
* **Wersje z wynikami** (bez uruchamiania):
  * `notebooks/Word_Embeddings_executed.ipynb` – wykresy widać od razu po otwarciu w JupyterLab lub VS Code; jeśli
    klasyczny Notebook ich nie pokazuje, trzeba zaufać notebookowi (*File → Trust Notebook*) albo uruchomić go ponownie;
  * `notebooks/Word_Embeddings_executed.html` – w przeglądarce, z interaktywnymi wykresami.

  Obie powstają poleceniami:
  `jupyter nbconvert --to notebook --execute notebooks/Word_Embeddings.ipynb --output Word_Embeddings_executed.ipynb`
  i `jupyter nbconvert --to html notebooks/Word_Embeddings_executed.ipynb`.
* Testy: `python -m pytest tests/test_word_notebook.py -q`. Sprawdzają 4 wykresy, oba formaty i podpowiedzi
  ze słowem i anotacją, a także uruchamiają cały notebook.

## Uruchomienie

```bash
pip install -r requirements_embeddings.txt      # gensim >= 4.4 (działa z numpy 2), plotly, …

# model word2vec IPI PAN (771 MB) – plik z uwag technicznych; katalog models/ jest w .gitignore
mkdir -p models
curl -L -o models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-skipg-ns.txt.gz

python -m src.word_embeddings                          # word2vec IPI PAN + word2vec wytrenowany na PolEmo
python -m src.word_embeddings --models w2v fasttext    # + fastText KGR10 (drugi model; pobiera ~3,7 GB)
python -m pytest tests/test_word_embeddings.py -q      # testy (bez modeli działają na sztucznych wektorach)
```

- **Wczytanie modelu:** `text_embeddings._load_w2v` wczytuje **300 tys. najczęstszych form** (`config.W2V["limit"]`, zgodnie z uwagami technicznymi). Pierwsze wczytanie trwa ok. 30 s i zapisuje kopię `models/…_300000.kv`, którą kolejne uruchomienia czytają w ok. 3 s. Inną ścieżkę do modelu można podać w zmiennej `W2V_PATH`.
- **Model PolEmo** (`w2v_polemo`): word2vec wytrenowany w gensim na wszystkich 45 935 zdaniach PolEmo. Parametry (`config.W2V_POLEMO`): 100 wymiarów, skip-gram, negative sampling, `min_count=3`, 15 epok, `seed=2026`, `workers=1`, więc wynik jest powtarzalny. Trening trwa ok. 15 s, model jest zapisywany w `models/w2v_polemo.kv`. Ponowny trening: `--retrain`.
- **Wersja angielska (gensim):** nie dotyczy, bo PolEmo jest polski. gensim służy tu jako biblioteka do wczytania i trenowania modeli.

## Pliki wynikowe (`results/word_embeddings/`)

| Plik | Zawartość |
|---|---|
| `vocab.csv` | 1505 słów z 200 zaanotowanych zdań (iteracje 1–2): `freq`, `n_sentences`, `surface`, liczby etykiet anotatorów `n_O … n_ODW`, `annot_share`, `word_label`, rozkład etykiety zdania `share_POZ … share_AMB`, `polemo_freq`, a dla każdego modelu `<model>_in_vocab`, `<model>_form`, `<model>_rank` |
| `vectors_w2v.npz`, `vectors_w2v_polemo.npz` | `words`, `vectors` (float32, n×100), `in_vocab`. Zawierają tylko słowa korpusu, więc t-SNE (pkt 2) nie wymaga pobierania modelu |
| `coverage.md`, `coverage.json` | pokrycie słownika ogółem i dla każdej etykiety słowa, najczęstsze słowa bez wektora z prawdopodobną przyczyną |
| `sanity.md` | najbliższe słowa dla słów kontrolnych i podobieństwa par (dobry–zły itd.) |

- **Słowo** to token zaanotowanego korpusu zawierający literę, zapisany małymi literami; interpunkcję i same liczby pomijamy.
- **Wyszukiwanie wektora:** kolejno forma małymi literami, oryginalna pisownia i forma z wielką literą na początku. W słowniku IPI PAN wielkie litery mają głównie nazwy własne, np. „polecam” jest w słowniku, a „Polecam” nie.
- **`word_label`:** najczęstsza etykieta różna od O, jeśli co najmniej połowa anotacji danego słowa jest różna od O; w przeciwnym razie O. Przy remisie decyduje kolejność `config.TOKEN_LABELS`. Przykład: słowo „nie” anotatorzy oznaczyli jako `ODW` w 44% anotacji, więc dostaje O, bo „nie” w zdaniach neutralnych celowo zostawiano bez etykiety.

## Wyniki (iteracje 1–2)

| Model | Pokrycie typów | Pokrycie tokenów | Słowa zaanotowane |
|---|---|---|---|
| Word2Vec IPI PAN (nkjp+wiki, formy, skip-gram, 300 tys.) | 94,6% (1423/1505) | 97,1% | 94,9% (296/312) |
| Word2Vec PolEmo (gensim) | 85,2% (1283/1505) | 92,0% | 83,7% (261/312) |

- **Brakujące wektory w IPI PAN** dotyczą głównie słów rzadkich (pojedyncze wystąpienia także w PolEmo), form z cyframi („2osobowym”) i zapisów bez polskich znaków („lazience”, w modelu jest „łazience”). Słowa `WZM`, `OSL` i `ODW` są pokryte w 100%.
- **Model PolEmo** ma niższe pokrycie, bo pomija słowa występujące rzadziej niż 3 razy. Najbardziej odczuwają to słowa `WYR_NEG`: pokrycie 65%.
- **Kontrola sensowności** (`sanity.md`):
  - sąsiedzi w IPI PAN są trafni: „lekarz” → chirurg, internista, neurolog; „brud” → smród, kurz,
  - **antonimy leżą blisko** (cos(dobry, zły) = 0,73; cos(polecam, odradzam) = 0,79), bo word2vec uczy się z kontekstu, a nie z wydźwięku. To materiał do pytania 2 w dyskusji,
  - model PolEmo jest wyraźnie bardziej zaszumiony.

## Uwaga: spójność z Text Embeddings

`text_embeddings._load_fasttext` wczytuje teraz plik **`kgr10.plain.skipgram.dim100.neg10.bin`** (`config.FASTTEXT_FILE`, zalecenie z uwag technicznych), najpierw z `models/`, a jeśli go tam nie ma, z Hugging Face. Wcześniej bez ścieżki brał pierwszy plik alfabetycznie, czyli model CBOW. Osadzenia zdań fastText w `results/text_embeddings/` policzono jeszcze modelem CBOW. Żeby obie części używały tego samego modelu, trzeba je przeliczyć:

```bash
python -m src.text_embeddings --models fasttext use --force
```

## Pkt 2: t-SNE słów z etykietami anotacji

Plan: [PLAN_Czesc2_Word2Vec_tSNE.md](PLAN_Czesc2_Word2Vec_tSNE.md). Skrypt korzysta tylko z plików pkt 1, więc nie wymaga modeli.

```bash
python -m src.word_tsne                       # w2v IPI PAN + w2v PolEmo, oba podzbiory (~25 s)
python -m src.word_tsne --models fasttext     # gdy pkt 1 zapisze vectors_fasttext.npz
python -m pytest tests/test_word_tsne.py -q
```

| Plik (`results/word_embeddings/`) | Zawartość |
|---|---|
| `tsne_words_<model>_all.html` | **wykres interaktywny**: wszystkie słowa z wektorem (IPI PAN 1423, PolEmo 1283); O = szare tło |
| `tsne_words_<model>_annotated.html` | słowa z jakąkolwiek anotacją ≠ O (296 / 261); t-SNE liczone osobno |
| `tsne_words_<model>_<podzbiór>.png` | obraz statyczny do raportu: panele „etykieta słowa” i „kontekst zdań”, podpisy 25 najczęstszych słów z etykietą ≠ O |
| `tsne_words_<model>_<podzbiór>.csv` | `word, word_label, context_label, freq, annot_share, x, y` |
| `label_separation.md` / `.json` | czy etykiety grupują się w oryginalnej przestrzeni 100-wym. |

**Obsługa wykresu:**
- **Podpowiedź po najechaniu** pokazuje słowo (i jego pisownię w korpusie), etykietę słowa, liczbę anotacji każdej etykiety, liczbę wystąpień, dominującą etykietę zdań z rozkładem procentowym, częstość w PolEmo i rangę w modelu.
- **Lista rozwijana** przełącza kolorowanie: *etykieta słowa* (WYR_POZ / WYR_NEG / WZM / OSL / ODW / O) albo *kontekst zdań* (dominująca etykieta `final` zdań, w których słowo wystąpiło; remis = „mieszany”).
- **Przycisk „podpisy”** pokazuje napisy przy słowach z etykietą ≠ O występujących co najmniej 2 razy.
- **Kliknięcie w legendę** ukrywa lub pokazuje klasę.
- Wielkość punktu rośnie z liczbą wystąpień.
- t-SNE: ta sama metoda co dla zdań (`text_embeddings.tsne_2d`: kosinus, inicjalizacja PCA, perplexity 30, seed 42).

**Kolory.** Trzy barwy zespołu (zielony = pozytywne, pomarańczowy = negatywne, fioletowy) przechodzą test rozróżnialności dla osób z zaburzeniami widzenia barw przy wszystkich parach punktów. Czwartej barwy nie dodajemy, dlatego:
- WZM, OSL i ODW mają wspólny fiolet i różne kształty,
- AMB to fioletowy romb,
- O i NEU są szarym tłem.

Plik HTML wczytuje Plotly z CDN (jak wykresy zespołu), więc do wyświetlenia potrzebny jest internet.

**Wyniki: czy etykiety się grupują** (`label_separation.md`, przestrzeń 100-wym., kosinus, k = 10):

| Model | Zestaw | Czystość k-NN | Losowo [95%] | p | Sylwetka |
|---|---|---|---|---|---|
| IPI PAN | WYR_POZ vs WYR_NEG | 0,648 | 0,536 [0,510; 0,562] | 0,001 | 0,033 |
| IPI PAN | POZ vs NEG (kontekst zdań) | 0,545 | 0,502 [0,487; 0,519] | 0,001 | 0,004 |
| PolEmo | WYR_POZ vs WYR_NEG | 0,671 | 0,571 [0,535; 0,605] | 0,001 | 0,038 |
| PolEmo | POZ vs NEG (kontekst zdań) | 0,559 | 0,504 [0,489; 0,519] | 0,001 | 0,006 |

**Wniosek do dyskusji (pkt 4):** słowa o tej samej etykiecie są sobie bliższe niż przy losowym przypisaniu etykiet (p = 0,001), ale **wyraźnych skupisk nie ma** (sylwetka ≈ 0).
- Grupują się głównie słowa pozytywne (czystość WYR_POZ 0,75, a WYR_NEG tylko 0,47).
- Słowa negatywne mieszają się z pozytywnymi. word2vec uczy się z kontekstu, a antonimy pojawiają się w tych samych kontekstach (cos(dobry, zły) = 0,73).
- Modyfikatory (bardzo, trochę, całkiem, bardziej) tworzą zwarte skupisko widoczne na wykresie. Grupują się według **funkcji składniowej**, a nie wydźwięku.

## Pkt 3: k najbliższych słów i porównanie modeli

Plan: [PLAN_Czesc2_Word2Vec_kNN.md](PLAN_Czesc2_Word2Vec_kNN.md). Ten krok wymaga lokalnych modeli (`models/`). Wyniki (CSV, MD) są małe i trafiają do repozytorium.

```bash
python -m src.word_neighbours                                   # w2v IPI PAN vs w2v PolEmo (~12 s)
python -m src.word_neighbours --models w2v w2v_polemo fasttext  # po pobraniu fastText: wszystkie pary
python -m pytest tests/test_word_neighbours.py -q
```

| Plik (`results/word_embeddings/`) | Zawartość |
|---|---|
| `neighbours_<model>.csv` | 10 sąsiadów każdego słowa zapytania: `query, query_label, query_freq, in_main_set, rank, neighbour, cosine, same_stem, polemo_n, polarity, polarity_class` |
| `knn_comparison.csv` | dla każdego słowa i pary modeli: wspólne@10, Jaccard@10, wspólne@10 na wspólnym słowniku, ρ drugiego rzędu, udział odmian |
| `knn_summary.md` | tabele zbiorcze, **listy pokazowe obok siebie** (22 słowa), słowa najbardziej i najmniej zgodne, przykłady sąsiadów o przeciwnym wydźwięku |
| `knn_summary.json` | wszystkie liczby i parametry |

**Słowa zapytania i listy:**
- **Słowa zapytania:** 215 słów z `word_label ≠ O` znanych obu modelom (WYR_POZ 125, WYR_NEG 55, WZM 18, OSL 13, ODW 4).
- **Zestaw pokazowy:** po 4 najczęstsze słowa z każdej etykiety oraz *polecam, miła, brudne, problem, bardzo, trochę, niestety, nie*. Słowo „nie” ma etykietę O, ale jest kluczowe dla negacji.
- **Sąsiedzi:** k = 10 z całego słownika modelu (IPI PAN 300 tys., PolEmo 21,7 tys.), tylko słowa z liter, bez samego słowa zapytania.

**Miary:**
- **Wspólne@10:** udział sąsiadów obecnych w obu listach.
- **Wspólne@10 na wspólnym słowniku:** to samo, ale sąsiadów szukamy tylko wśród 20 206 słów znanych obu modelom, więc różnica wielkości słowników nie zaniża wyniku.
- **ρ drugiego rzędu:** korelacja Spearmana podobieństw między słowami zapytania w obu modelach, czyli czy modele podobnie układają słowa względem siebie.
- **Odmiany (`same_stem`):** reguła przybliżona bez lematyzatora: wspólny początek wyrazu ≥ max(3, ⌈0,6 · krótszego słowa⌉), oba słowa mają co najmniej 4 litery. Próg 3 zamiast 4 z planu przyjąłem, żeby para „miła”/„miły” była rozpoznana.
- **Wydźwięk sąsiada:** s = (zdania POZ − zdania NEG) / wszystkie zdania PolEmo ze słowem, dla słów z co najmniej 5 zdaniami; próg ±0,1.

### Wyniki (w2v IPI PAN vs w2v PolEmo)

| Miara | Wartość |
|---|---|
| wspólne@10 (surowe / na wspólnym słowniku) | **9% / 12%**; najwyżej dla WZM (16% / 18%) |
| ρ drugiego rzędu (średnia dla słów / cała macierz) | 0,29 / 0,27 |
| odmiany słowa zapytania wśród sąsiadów | IPI PAN 13%, PolEmo 2% |
| sąsiedzi słów WYR_POZ: zgodny / **przeciwny** wydźwięk | IPI PAN 47% / **41%**; PolEmo 59% / 32% |
| sąsiedzi słów WYR_NEG: zgodny / przeciwny wydźwięk | IPI PAN 67% / 17%; PolEmo 70% / 20% |

**Wnioski do dyskusji (pkt 4, pytanie 3: różnice list i ich przyczyny):**
1. **Modele prawie się nie zgadzają** (wspólne@10 ≈ 10%, ρ ≈ 0,3). Różnica nie wynika z wielkości słowników, bo na wspólnym słowniku wspólnych sąsiadów jest tylko nieco więcej (12%).
2. **Korpus ogólny (IPI PAN)** daje synonimy i słowa tej samej kategorii, ale także **antonimy**: polecam → odradzam, zadowolona → niezadowolona, pozytywne → negatywne, plusem → minusem, dobrze → słabo. Dlatego aż 41% ocenionych sąsiadów słów pozytywnych ma wydźwięk negatywny. W modelu „forms” częste są też odmiany tego samego słowa (13%).
3. **Korpus dziedzinowy (PolEmo)** daje skojarzenia tematyczne z recenzji: miła → pomocna, uprzejma, profesjonalna (lekarze); brudne → wykładziny, kafelki, kurz, pleśń (hotele). Przy rzadszych słowach daje jednak szum (problem → kulinarnie, zdrowotny; nie → oj, wroce), bo korpus ma tylko 0,8 mln tokenów.
4. **Homonimia i pisownia:**
   - „ok” anotatorzy oznaczyli jako WYR_POZ (w porządku), a oba modele rozumieją je jako skrót „około” (około, ponad, prawie),
   - formy bez polskich znaków tworzą w IPI PAN własne sąsiedztwo (zaden → kazdy, zwykly, doskonaly).
5. **Modyfikatory** (WZM) mają najwięcej wspólnych sąsiadów (bardzo → dość, niezbyt w obu modelach). Ich znaczenie jest funkcyjne i mniej zależy od dziedziny.

**Zastrzeżenia:**
- Model PolEmo i miara wydźwięku pochodzą z tego samego korpusu, więc jego lepsza zgodność wydźwięku jest częściowo „z założenia”.
- s(w) opiera się na etykiecie całego zdania i nie uwzględnia negacji. Przykład: „polecę” ma wydźwięk ujemny, bo zwykle występuje w „nie polecę”.

## Pkt 4: dyskusja (sprawozdanie LaTeX)

Plan: [PLAN_Czesc2_Word2Vec_Dyskusja.md](PLAN_Czesc2_Word2Vec_Dyskusja.md). Dyskusja jest w sprawozdaniu (`report/sprawozdanie.tex`, rozdział „Część II: Word Embeddings”).

```bash
python -m src.word_tsne        # (jeśli zmieniły się wyniki) separacja etykiet + sonda liniowa
python -m src.word_report      # report/generated/we_*.tex + report/figures/tsne_words_*.png
python -m pytest tests/test_word_report.py -q   # składnia LaTeX, makra, odwołania, rysunki (bez kompilatora)
```

| Plik | Zawartość |
|---|---|
| `report/generated/we_macros.tex` | 60 liczb jako makra `\we…` (np. `\weCovTypesIpi` = 94,6). Generowany, nie edytować ręcznie |
| `report/generated/we_models.tex` | pkt 1: opis modeli, tabela pokrycia słownika |
| `report/generated/we_tsne.tex` | pkt 2: rysunki t-SNE (IPI PAN, PolEmo), tabela separacji etykiet z sondą liniową |
| `report/generated/we_knn.tex` | pkt 3: zgodność list, charakter list (odmiany, wydźwięk), 5 najbliższych słów dla 10 słów |
| `report/sections/we_dyskusja.tex` | **pkt 4: dyskusja**, pisana ręcznie (ok. 550 słów, 4 pytania pomocnicze); liczby wyłącznie przez makra |

Zmiany w `report/sprawozdanie.tex` (osobny commit, do przejrzenia przez autora szablonu):
1. wczytanie `we_macros`,
2. wstęp rozdziału Word Embeddings zamiast `	odo` i wpięcie dyskusji,
3. 3 punkty we „Wnioskach”,
4. dopisek w części o narzędziach opartych na modelach językowych,
5. bibliografia: word2vec (Mikolov i in. 2013), modele IPI PAN (strona zasobu, bo nie podaje publikacji do cytowania), retrofitting (Faruqui i in. 2015).

**Na tym komputerze nie ma kompilatora LaTeX**, więc PDF trzeba zbudować u osoby z LaTeX-em albo w Overleaf (`pdflatex sprawozdanie.tex`, 2 razy). Testy sprawdzają składnię bez kompilacji.

**Nowy wynik do dyskusji (sonda liniowa, `label_separation.md`):** regresja logistyczna odczytuje WYR_POZ / WYR_NEG z wektora IPI PAN z trafnością 0,72, a przy losowych etykietach 0,50. Wydźwięk jest więc częściowo zakodowany w wektorach, choć nie widać go w sąsiedztwie słów. Dla modelu PolEmo trafność wynosi 0,58.

## Co dalej (Word Embeddings)

| Pkt | Wejście |
|---|---|
| fastText jako trzeci model (rodzaj modelu inny niż word2vec) | `python -m src.word_embeddings --models w2v w2v_polemo fasttext`, potem `word_tsne` i `word_neighbours` z `--models … fasttext` (~3,7 GB do pobrania); `word_report` i dyskusję trzeba wtedy rozszerzyć o trzeci model |
