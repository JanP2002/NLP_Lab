# Plan: Część II, Word Embeddings, pkt 2: t-SNE słów word2vec z etykietami anotacji

Treść punktu: *„Wizualizacja danych za pomocą t-SNE z odniesieniem do etykiet w ramach anotacji. Wykonać wykres interaktywny, w którym po najechaniu na punkt można odczytać słowo i / lub anotację. [0,5 pkt]”*

Gałąź: `czesc2-w2v` (kontynuacja pkt 1; zmiany z pkt 1 nie są jeszcze w commicie).

## 0. Punkt wyjścia

| Element | Stan |
|---|---|
| Wynik pkt 1 | `results/word_embeddings/vocab.csv` (1505 słów, etykiety anotacji, kontekst zdań) i `vectors_w2v.npz` / `vectors_w2v_polemo.npz` (wektory 100-wym. słów korpusu) |
| Słowa z wektorem | **w2v IPI PAN: 1423** (O 1172, WYR_POZ 137, WYR_NEG 78, WZM 19, OSL 13, ODW 4; 296 z jakąkolwiek anotacją ≠ O). **w2v PolEmo: 1283** (O 1067, WYR_POZ 125, WYR_NEG 56, WZM 18, OSL 13, ODW 4) |
| Częstość | 1138 z 1423 słów występuje w zaanotowanym korpusie **raz**; 144 słowa co najmniej 3 razy |
| Gotowe narzędzia zespołu | `text_embeddings.tsne_2d` (t-SNE: kosinus, inicjalizacja PCA, perplexity = min(30, (n−1)/3), seed 42); paleta `LABEL_COLORS`; wykres zdań `make_tsne_figure` z listą rozwijaną zmieniającą kolorowanie |
| Raport LaTeX | `report/build_assets.py` robi obrazki PNG w matplotlib (dwa panele: final / gold) na podstawie wykresów HTML; `kaleido` nie jest zainstalowane |

**Zasada:** pkt 2 korzysta **tylko z plików pkt 1** (`.npz` i `vocab.csv`), więc działa bez pobierania modeli (771 MB / 3,7 GB). Każdy w zespole może go uruchomić.

## 1. Co pokazuje wykres

### 1.1 Punkty
- **Wykres A, „wszystkie słowa”:** wszystkie słowa z wektorem (1423 dla IPI PAN). Słowa bez etykiety (O) są małymi, półprzezroczystymi szarymi punktami, a słowa zaanotowane dużymi kolorowymi. Widać, gdzie słowa oceniające leżą na tle całego słownictwa. Kliknięcie „O” w legendzie ukrywa tło.
- **Wykres B, „słowa zaanotowane”:** tylko słowa z jakąkolwiek anotacją ≠ O (296). t-SNE jest liczone **osobno** na tym podzbiorze, bo daje wtedy lepszą lokalną strukturę: czy WYR_POZ i WYR_NEG tworzą osobne skupiska.

### 1.2 Kolorowanie (lista rozwijana nad wykresem, jak w wykresie zdań zespołu)
| Wariant | Źródło | Znaczenie |
|---|---|---|
| **etykieta słowa** (domyślny) | `word_label` (anotacja tokenów) | WYR_POZ / WYR_NEG / WZM / OSL / ODW / O |
| **kontekst zdań** | dominująca etykieta `final` zdań, w których słowo wystąpiło (największe `share_*`; przy remisie „mieszany”) | POZ / NEG / NEU / AMB / mieszany. Pokazuje, czy słowa z zdań pozytywnych i negatywnych się grupują, także te nieoznaczone |
| bilans wydźwięku (opcjonalnie) | `share_POZ − share_NEG`, skala ciągła od −1 do +1 | odcień zamiast klas; dobry dla słów częstych |

Kolory zgodne z paletą zespołu (`LABEL_COLORS`): POZ i WYR_POZ to ten sam zielony, NEG i WYR_NEG ten sam pomarańczowy, NEU fioletowy, AMB żółty. WZM, OSL i ODW dostają trzy odróżnialne kolory spoza tej palety, a O i „mieszany” szary. Przy implementacji paletę sprawdzę pod kątem kontrastu i czytelności dla osób z zaburzeniami widzenia barw.

### 1.3 Podpowiedź po najechaniu na punkt (wymóg zadania: słowo i/lub anotacja)
```
polecam   (pisownia w korpusie: „Polecam”)
etykieta słowa: WYR_POZ | anotacje: WYR_POZ 8, O 0 (100% ≠ O)
wystąpienia: 4 w 4 zdaniach | kontekst zdań: POZ 75%, NEG 25%, NEU 0%, AMB 0%
PolEmo: 1 234 wystąpień | ranga w modelu: 5 321
```

### 1.4 Wielkość punktu i podpisy
- Wielkość punktu rośnie z logarytmem liczby wystąpień, więc częste słowa są widoczne.
- Wykres B: opcjonalne podpisy tekstowe przy słowach występujących co najmniej 2 razy (przycisk „podpisy: wł./wył.”).

## 2. Ilościowe uzupełnienie: czy etykiety się grupują?

t-SNE zniekształca odległości: odstępy między skupiskami i ich wielkość nie mają znaczenia. Dlatego wrażenie z wykresu warto sprawdzić **w oryginalnej przestrzeni 100-wymiarowej** (kosinus). Przyda się to też w pkt 4 (pytanie 2: „czy występuje grupowanie na słowa pozytywne i negatywne?”).

- **Czystość k-NN:** dla każdego słowa WYR_POZ i WYR_NEG (n ≈ 215) odsetek jego k = 10 najbliższych sąsiadów (spośród tych słów) z tą samą etykietą, uśredniony. Porównanie z **losowym przypisaniem etykiet**: 1000 permutacji, średnia, 95% przedział i wartość p.
- **Współczynnik sylwetki** (kosinus) dla WYR_POZ i WYR_NEG oraz dla wszystkich 5 etykiet słów.
- To samo dla wariantu „kontekst zdań” (POZ i NEG).
- Wynik: `label_separation.md` i `.json`, osobno dla każdego modelu.

Oczekiwanie do sprawdzenia: kontrast pozytywne/negatywne będzie **słaby**, bo word2vec stawia antonimy blisko siebie (cos(dobry, zły) = 0,73 w pkt 1). Wtedy wykres i liczby razem dają argument do dyskusji.

## 3. Implementacja: `src/word_tsne.py` (nowy)

| Funkcja | Zadanie |
|---|---|
| `load_points(model, subset)` | łączy `vectors_<model>.npz` z `vocab.csv`; `subset ∈ {"all", "annotated"}`; zwraca tabelę słów i macierz wektorów |
| `context_label(vocab)` | dominująca etykieta zdań z kolumn `share_*`; remis → „mieszany” |
| `tsne_coords(X, seed, perplexity)` | `text_embeddings.tsne_2d` (ta sama metoda co dla zdań) |
| `make_word_figure(points, xy, title)` | Plotly: grupy punktów dla każdego wariantu kolorowania, lista rozwijana, podpowiedź z p. 1.3, wielkość z liczby wystąpień, O jako tło |
| `save_static_png(points, xy, path)` | matplotlib w stylu `report/build_assets.py`: 2 panele (etykieta słowa / kontekst zdań), legenda pod spodem, podpisy 25 najczęstszych słów zaanotowanych; do raportu LaTeX |
| `label_separation(X, labels, k, n_perm, seed)` | czystość k-NN z permutacjami i sylwetka (p. 2) |
| `main()` | CLI: `--models w2v w2v_polemo`, `--subsets all annotated`, `--perplexity`, `--seed` |

**Wyniki** (`results/word_embeddings/`) dla każdego modelu i podzbioru:
- `tsne_words_<model>_<subset>.html`: wykres interaktywny. Plotly ładowane z CDN, jak u zespołu, więc plik jest mały, ale do wyświetlenia potrzebny jest internet.
- `tsne_words_<model>_<subset>.csv`: `word, x, y, word_label, context_label, freq`. Raport może z niego zrobić własny PNG.
- `tsne_words_<model>_<subset>.png`: obraz statyczny do LaTeX.
- `label_separation.md` i `label_separation.json`.

## 4. Testy (`tests/test_word_tsne.py`, bez modeli)

1. Sztuczne wektory z 3 wyraźnymi skupiskami:
   - wykres ma osobną grupę punktów dla każdej etykiety w każdym wariancie kolorowania,
   - podpowiedź zawiera słowo i etykietę,
   - lista rozwijana ma tyle opcji, ile wariantów,
   - grupa O jest szara i półprzezroczysta.
2. `tsne_coords` z tym samym seedem daje identyczne współrzędne.
3. `context_label`: najczęstsza etykieta wygrywa, remis daje „mieszany”, wynik dla brakującego `final` jest określony.
4. `label_separation`: dla idealnie rozdzielonych skupisk czystość wynosi 1, a p jest małe; dla przetasowanych etykiet czystość jest bliska wartości losowej.
5. Pliki CSV, HTML i PNG powstają, a CSV ma oczekiwane kolumny.
6. Dane rzeczywiste (pomijane, gdy brak plików pkt 1): 1423 punkty dla `w2v/all`, 296 dla `w2v/annotated`; każdy punkt ma słowo z `vocab.csv`.

## 5. Uruchomienie (po implementacji)

```bash
python -m src.word_tsne --models w2v w2v_polemo     # wykresy A i B dla obu modeli word2vec + separacja etykiet
python -m pytest tests/test_word_tsne.py -q
```

Ten sam skrypt obsłuży fastText (`--models fasttext`), gdy w pkt 1 powstanie `vectors_fasttext.npz`.

**Gotowe, gdy:**
- 4 pliki HTML (2 modele × 2 podzbiory) otwierają się w przeglądarce,
- po najechaniu na punkt widać słowo i anotację,
- lista rozwijana przełącza kolorowanie,
- jest `label_separation.md` z wartościami dla obu modeli,
- są pliki PNG do raportu,
- testy przechodzą.

## 6. Czas

| Krok | Czas |
|---|---|
| `word_tsne.py`: punkty, t-SNE, wykres interaktywny | ~45 min |
| PNG do raportu, separacja etykiet | ~30 min |
| testy | ~30 min |
| uruchomienie, przegląd wykresów, README | ~15 min |

## 7. Decyzje do potwierdzenia

| # | Pytanie | Opcje | Rekomendacja |
|---|---|---|---|
| D1 | Które słowa na wykresie | (a) wykres A „wszystkie” + wykres B „zaanotowane” / (b) tylko wszystkie / (c) tylko zaanotowane | **(a)**: A daje tło, B czytelny obraz etykiet |
| D2 | Warianty kolorowania | etykieta słowa + kontekst zdań / + bilans wydźwięku / tylko etykieta słowa | **etykieta słowa + kontekst zdań**. Bilans wydźwięku opcjonalnie, przy większości słów występujących raz niewiele dodaje |
| D3 | Modele | w2v IPI PAN + w2v PolEmo / tylko IPI PAN | **oba**; to tylko dwa dodatkowe pliki z tego samego kodu, a porównanie przyda się w dyskusji |
| D4 | Ilościowa separacja etykiet (p. 2) | tak / nie | **tak**: wykres t-SNE sam w sobie nie rozstrzyga, czy etykiety się grupują |
| D5 | PNG do raportu | własny PNG w `results/word_embeddings/` / dopisanie funkcji do `report/build_assets.py` | **własny PNG**: bez ingerencji w plik autora raportu; dodanie `\includegraphics` uzgodnić z Wysciguvvka |
| D6 | Sprawdzenie stabilności t-SNE (perplexity 5/15/30/50, kilka seedów) | tak / nie | **nie** na teraz (0,5 pkt); ewentualnie później jako materiał do dyskusji |
