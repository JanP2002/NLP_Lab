# Zadanie 1, Część II – Text Embeddings (2 pkt)

Pliki i punkty zadania:

| Plik | Co robi | Punkt zadania |
|---|---|---|
| `src/text_embeddings.py` | osadzenia zdań (fastText / Word2Vec / USE / TF-IDF), t-SNE, wykres Plotly, funkcje analizy | pkt 1, 2, 3 |
| `notebooks/Zadanie1_Text_Embeddings.ipynb` | raport: sekcja 1 = pkt 1, sekcja 2 = pkt 2, sekcja 3 = pkt 3 (pytania 1–5) | pkt 1, 2, 3 |
| `tests/test_text_embeddings.py` | testy offline | – |
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
# z własnymi plikami modeli:
python -m src.text_embeddings --models fasttext w2v use --fasttext-path /sciezka/kgr10.bin --w2v-path /sciezka/model.bin

# pełny raport (pkt 1–3):
jupyter notebook notebooks/Zadanie1_Text_Embeddings.ipynb
```

* fastText kgr10 pobiera się z Hugging Face (`clarin-pl/fastText-kgr10`) i waży kilka GB.
* Word2Vec z https://dsmodels.nlp.ipipan.waw.pl trzeba pobrać ręcznie (wybierz model „forms”, bez znaczników POS)
  i podać `--w2v-path` / `W2V_PATH` w notebooku.
* USE: domyślnie `auto` = oryginał z TF Hub, a gdy go brak (np. Windows) – `distiluse` (destylat mUSE z sentence-transformers).
  Oryginał wymaga `pip install -r requirements_use_tfhub.txt` (bez Windows).
  W raporcie wpisz, który backend faktycznie użyto (notebook podaje to w opisie modelu).
* Osadzenia są cache'owane w `results/text_embeddings/`; po dodaniu iteracji 2 do analizy cache przelicza się sam.
* Po rozstrzygnięciu anotacji zapisz `annotation/iterN/adjudicated.csv` (`sent_id,final`) – pojawi się kolor `final`.
