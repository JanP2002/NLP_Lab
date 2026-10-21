# Zadanie 1, Część II – Text Embeddings (2 pkt)

Pliki i punkty zadania:

| Plik | Co robi | Punkt zadania |
|---|---|---|
| `src/text_embeddings.py` | osadzenia zdań (fastText / Word2Vec / USE / TF-IDF), t-SNE, wykres Plotly, funkcje analizy | pkt 1, 2, 3 |
| `notebooks/Text_Embeddings.ipynb` | raport: sekcja 1 = pkt 1, sekcja 2 = pkt 2, sekcja 3 = pkt 3 (pytania 1–5); Word2Vec w 1.4, 2.1, 3.7 | pkt 1, 2, 3 |
| `src/notebook_w2v.py` | dopisuje komórki Word2Vec do notebooka i wykonuje tylko je | pkt 1b, 2, 3 |
| `tests/test_text_embeddings.py`, `tests/test_text_embeddings_w2v.py`, `tests/test_te_notebook_w2v.py` | testy offline (+ uśrednione Word2Vec) | – |
| `requirements_embeddings.txt` | zależności (Windows/Linux/macOS) | – |
| `requirements_use_tfhub.txt` | opcjonalnie: oryginalny USE z TF Hub (tylko Linux/macOS/WSL/Colab) | – |

## Uruchomienie (z katalogu głównego repo, czyli tam, gdzie leży `config.py`)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements_embeddings.txt

python -m pytest tests/test_text_embeddings.py -q      # opcjonalnie: sprawdzenie bez pobierania modeli
python -m src.text_embeddings --models tfidf           # szybki test całego potoku (bez pobierania)

# pkt 1 i 2 z linii poleceń: osadzenia + t-SNE + wykresy HTML w results/text_embeddings/
python -m src.text_embeddings --models fasttext use
# pkt 1b: uśrednione wektory Word2Vec (modele z części Word Embeddings, katalog models/)
python -m src.text_embeddings --models w2v w2v_polemo
# z własnymi plikami modeli:
python -m src.text_embeddings --models fasttext w2v use --fasttext-path /sciezka/kgr10.bin --w2v-path /sciezka/model.bin

# pełny raport (pkt 1–3):
jupyter lab notebooks/Text_Embeddings.ipynb        # albo: jupyter notebook …, albo VS Code
```

* **Wykresy interaktywne w notebooku.** Notebook ustawia `pio.renderers.default = "plotly_mimetype+notebook_connected"`:
  * format `plotly_mimetype` rysują JupyterLab, Notebook 7 i VS Code, także w niezaufanym notebooku,
  * format `notebook_connected` rysuje klasyczny Notebook i eksport HTML (plotly.js z CDN, potrzebny internet).

  Wcześniej było samo `notebook_connected`, a jego wykresy (HTML ze skryptem) JupyterLab i Notebook 7 pokazują jako
  puste pole, bo w wynikach niezaufanego notebooka nie uruchamiają skryptów.
* `notebooks/Text_Embeddings_executed.ipynb` ma już dopisany format `plotly_mimetype` dla 7 wykresów zespołu
  (3 mapy podobieństwa, 3 t-SNE, wykres długości). Dopisał go `python -m src.notebook_plotly <notebook>`, który odtwarza
  figury z zapisanego HTML, bez ponownego uruchamiania notebooka; wyniki się nie zmieniły. Ten sam skrypt naprawi
  inne notebooki wykonane z `notebook_connected`.
* **Word2Vec w notebooku (pkt 1b, 2, 3).** Uśrednione Word2Vec (IPI PAN i PolEmo) ma w obu notebookach własne
  podsekcje:
  * **1.4**: osadzenia, pokrycie słownika, podgląd macierzy, 2 mapy podobieństwa;
  * **2.1**: 2 interaktywne wykresy t-SNE (hover: zdanie, etykiety gold, A1, A2, final);
  * **3.7**: analiza do pytań 1–5 (3.7.1–3.7.5: te same funkcje co 3.1–3.5, interaktywny wykres długości);
    zestawienie pięciu modeli w jednej tabeli według pytań (3.7.6); odpowiedzi na pytania 1–5, Word2Vec
    w porównaniu z fastText, USE i TF-IDF (3.7.7, tekst z `results/text_embeddings/odpowiedzi_porownanie.md`).
    W zestawieniu dwie miary korpusowe (POZ–NEG wśród 1% najbliższych par, Spearman cos vs Jaccard) liczone są też
    dla fastText (osadzenia CBOW z `results/text_embeddings_local/`, miary jak u zespołu) i TF-IDF; dla USE ich brak.

  Dopisuje je `python -m src.notebook_w2v` (ok. 30 s). Wykonanego notebooka zespołu nie uruchamiamy od nowa, bo
  w Windows nie ma USE z TF Hub. Skrypt wykonuje tylko komórki Word2Vec (po komórkach startowych zespołu), wpisuje ich
  wyniki i eksportuje `Text_Embeddings_executed.html`. Przed zapisem sprawdza dwie rzeczy:
  * komórki zespołu i ich wyniki się nie zmieniły;
  * liczby zgadzają się z `results/text_embeddings/` (`eval_*.csv`, `tsne_*.csv`, `analiza_w2v.json`), czyli
    ze sprawozdaniem.

  Razem w notebooku jest 12 wykresów interaktywnych (7 zespołu i 5 Word2Vec). Skrypt można uruchamiać wielokrotnie:
  stare komórki Word2Vec (znacznik `w2v`) są zastępowane.
* **Wersja uruchomiona lokalnie:** `notebooks/Text_Embeddings_local_executed.ipynb` / `.html` (21.10). To ponowne
  uruchomienie tego samego notebooka z nowym ustawieniem wykresów, więc wszystkie 7 wykresów ma oba formaty bez
  dodatkowej obróbki. Różnice względem wersji zespołu (opisane też w pierwszej komórce):
  * fastText: ten sam plik (`models/kgr10.plain.cbow.dim100.neg10.bin`, pobrany z HF). Osadzenia i miary są
    identyczne jak u zespołu, ale układ t-SNE jest inny przy innej wersji scikit-learn;
  * USE: `distiluse-base-multilingual-cased-v1` zamiast USE z TF Hub (w Windows brak `tensorflow_text`), więc liczby
    USE różnią się od sprawozdania, np. accuracy 0,525 wobec 0,520;
  * wyniki (osadzenia, CSV, HTML) trafiły do `results/text_embeddings_local/`; pliki zespołu w `results/text_embeddings/`
    zostały nietknięte.

  **Liczby w sprawozdaniu nadal pochodzą z wersji zespołu** (`Text_Embeddings_executed.ipynb`).

* fastText kgr10 pobiera się z Hugging Face (`clarin-pl/fastText-kgr10`) i waży kilka GB.
* Word2Vec z https://dsmodels.nlp.ipipan.waw.pl trzeba pobrać ręcznie (wybierz model „forms”, bez znaczników POS).
  Plik `models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz` (polecenie `curl` w `README_WordEmbeddings.md`) jest
  wczytywany bez podawania ścieżki; inny plik: `--w2v-path` / `W2V_PATH`. Wczytujemy 300 tys. najczęstszych form
  (`config.W2V["limit"]`, zalecenie z uwag technicznych), a pierwsze wczytanie zapisuje szybką kopię `.kv`.
* **Uśrednione Word2Vec (pkt 1b)** – dwa warianty z części Word Embeddings:
  * `w2v`: IPI PAN (ogólny NKJP + Wikipedia),
  * `w2v_polemo`: model wytrenowany na zdaniach PolEmo (`models/w2v_polemo.kv`; jeśli go nie ma, trenuje się w ok. 15 s).

  Wektor zdania to średnia wektorów jego słów (małe litery, bez interpunkcji). Słowa bez wektora są pomijane.
  Polecenie wypisuje pokrycie i zapisuje `embeddings_<model>.csv` (jak notebook). Wyniki dla 200 zdań:

  | Model | Pokrycie słów (średnio / min) | Zdania bez znanego słowa | Śr. kosinus par zdań | Śr. kosinus do średniego wektora |
  |---|---|---|---|---|
  | `w2v` (IPI PAN) | 96,7% / 70% | 0 | 0,674 | 0,822 |
  | `w2v_polemo` | 91,0% / 50% | 0 | 0,618 | 0,787 |

  Jak w fastText, uśrednianie zbliża wektory zdań do siebie (wysoki średni kosinus), więc podobieństwo trzeba
  porównywać względnie, a nie bezwzględnie.
* **t-SNE dla Word2Vec (pkt 2)** – to samo polecenie z `--evaluate`:

  ```bash
  python -m src.text_embeddings --models w2v w2v_polemo --evaluate
  python -c "import sys; sys.path.insert(0, 'report'); import build_assets as b; b.build_tsne_figures(('w2v', 'w2v_polemo'))"
  ```

  Pliki:
  * `results/text_embeddings/tsne_w2v.html`, `tsne_w2v_polemo.html`: wykresy interaktywne (ta sama funkcja
    `make_tsne_figure` co dla fastText i USE). Po najechaniu na punkt widać pełne zdanie, etykiety gold, A1, A2
    i final, iterację, `sent_id` i liczbę tokenów. Lista rozwijana zmienia kolorowanie (final, gold, A1, A2).
    Kształt punktu oznacza iterację.
  * `tsne_<model>.csv`: współrzędne; t-SNE z seedem 42 daje przy ponownym uruchomieniu te same wartości.
  * `report/figures/tsne_w2v.png`, `tsne_w2v_polemo.png`: obrazy statyczne w stylu wykresów fastText i USE,
    z panelami „final” i „gold” (`report/build_assets.py`; `python report/build_assets.py` tworzy je razem
    z resztą).
  * `eval_<model>.csv`: te same miary co tabela klasyfikacji w sprawozdaniu (`evaluate_embeddings`), dla każdego
    źródła etykiet.

  Porównanie dla etykiet final (n = 200, baza większościowa 0,415). Wartości fastText i USE pochodzą
  z `report/generated/emb_classif.tex`:

  | Model | accuracy | macro-F1 | kNN-10 (losowo 0,305) | silhouette |
  |---|---|---|---|---|
  | fastText | 0,505 | 0,322 | 0,321 | −0,123 |
  | USE | 0,520 | 0,330 | 0,334 | −0,013 |
  | **Word2Vec IPI PAN** | 0,505 | 0,310 | 0,329 | −0,065 |
  | **Word2Vec PolEmo** | **0,540** | **0,348** | **0,356** | −0,048 |

  Na wykresach t-SNE klasy są wymieszane, tak samo jak dla fastText i USE. Model PolEmo, mały, ale uczony
  na recenzjach, wypada nieco lepiej od pozostałych (na etykietach gold macro-F1 0,487 wobec 0,464 dla USE),
  ale przewaga nad bazą większościową jest nadal niewielka (+0,125). Widać w nim też skupisko zdań NEU.
  W sprawozdaniu Word2Vec opisuje podrozdział „Uśrednione wektory Word2Vec” (`report/sections/te_w2v.tex`, na końcu
  rozdziału Text Embeddings). Zawiera rysunki t-SNE, 3 tabele i odpowiedzi na pytania 1–5. Tabele i liczby
  (`report/generated/te_w2v_*.tex`) generuje `python -m src.text_report`; robi to też `python report/build_assets.py`.
* **Analiza i odpowiedzi na pytania 1–5 dla Word2Vec (pkt 3)**:

  ```bash
  python -m src.text_analysis                      # → results/text_embeddings/analiza_w2v.md / .json, probes_w2v.csv
  python -m pytest tests/test_text_analysis.py -q
  ```

  `src/text_analysis.py` powtarza kroki sekcji 3 notebooka (te same funkcje i 12 par kontrolnych `PROBE_PAIRS`)
  dla uśrednionego Word2Vec. Wartości fastText, USE i TF-IDF czyta z `report/generated/emb_*.tex`, więc tabele od
  razu porównują pięć modeli. Dochodzą dwie miary:
  * pyt. 1: udział par POZ–NEG wśród 1% najbardziej podobnych par zdań,
  * pyt. 2: zależność podobieństwa od wspólnych słów (Spearman: kosinus wobec Jaccarda zbiorów słów).

  **Odpowiedzi (pisane ręcznie, z liczbami):** [`results/text_embeddings/odpowiedzi_w2v.md`](results/text_embeddings/odpowiedzi_w2v.md).
  W skrócie:
  * pary o przeciwnym sensie są bliższe niż parafrazy (0,88 wobec 0,71),
  * wśród najbardziej podobnych par zdań pary POZ–NEG są nawet nadreprezentowane (15% wobec 12,5%),
  * długość zdania silnie zbliża je do średniej korpusu (Spearman 0,77, w fastText 0,59),
  * klasyfikator na osadzeniach niemal nie rozpoznaje NEG (czułość 7–11%),
  * Word2Vec PolEmo ma najwyższą trafność z pięciu modeli (0,540), ale uczono go na tym samym korpusie.
* USE: domyślnie `auto` = oryginał z TF Hub, a gdy go brak (np. Windows) – `distiluse` (destylat mUSE z sentence-transformers).
  Oryginał wymaga `pip install -r requirements_use_tfhub.txt` (bez Windows).
  W raporcie wpisz, który backend faktycznie użyto (notebook podaje to w opisie modelu).
* Osadzenia są cache'owane w `results/text_embeddings/`; po dodaniu iteracji 2 do analizy cache przelicza się sam.
* Po rozstrzygnięciu anotacji zapisz `annotation/iterN/adjudicated.csv` (`sent_id,final`) – pojawi się kolor `final`.
