# Plan: Część II, Word Embeddings, pkt 3: listy k najbliższych słów i porównanie modeli

Treść punktu: *„Wygenerować listy k-najbardziej podobnych dla słów zaanotowanych w korpusie i porównać otrzymane listy dla co najmniej 2 różnych modeli przestrzeni wektorowych [1 pkt]”.*

Gałąź: `czesc2-w2v` (kontynuacja pkt 1–2; zmiany z pkt 1–2 nie są jeszcze w commicie).

## 0. Punkt wyjścia

| Element | Stan |
|---|---|
| Modele lokalne | **w2v IPI PAN**: 300 tys. słów (`models/…_300000.kv`, wczytanie ok. 3 s); **w2v PolEmo**: 21 658 słów (`models/w2v_polemo.kv`); fastText KGR10 nie jest pobrany (3,7 GB) |
| Słowa zaanotowane (`vocab.csv`) | 267 słów z etykietą ≠ O; **215 jest w obu modelach word2vec** (WYR_POZ 125, WYR_NEG 55, WZM 18, OSL 13, ODW 4). Tylko 39 z nich występuje w zaanotowanym korpusie co najmniej 2 razy |
| Wydźwięk słów z etykiet PolEmo | 13 337 słów PolEmo ma co najmniej 5 wystąpień; z 50 tys. najczęstszych słów IPI PAN 10 714 ma taki wydźwięk |

Przykłady (8 najbliższych słów, tylko słowa z liter):

| Słowo | IPI PAN (ogólny) | PolEmo (dziedzinowy) | Wspólne |
|---|---|---|---|
| miła | sympatyczna, życzliwa, serdeczna, przyjemna, przemiła, **niemiła**, wspaniała, uczynna | pomocna, przemiła, uprzejma, sympatyczna, **profesjonalna**, życzliwa, **kompetentna**, przyjazna | 3/8 |
| brudne | śmierdzące, cuchnące, dziurawe, gołe, podarte, ohydne, rozwalone, odrapane | **wykładziny, kafelki, kurz**, zniszczone, **pleśń**, śmierdzące, pajęczyny, ciemne | 1/8 |
| problem | dylemat, kłopot, nierozwiązany, aspekt, argument, paradoks, … | gorszy, błąd, kulinarnie, zdrowotny, pech, zadzwonilam, … | 0/8 |
| trochę | troszkę, troszeczkę, ciut, odrobinkę, odrobinę, ciutkę, … | nieco, troszkę, zwyczajnie, ciągnie, wysłużone, … | 1/8 |

Już te przykłady pokazują główne różnice:
- model ogólny daje **synonimy i antonimy** (niemiła),
- model dziedzinowy daje **słowa tematyczne** (wykładziny, kafelki: hotele; profesjonalna, kompetentna: lekarze),
- przy rzadszych słowach model dziedzinowy daje **szum** (problem), bo jego korpus ma tylko 0,8 mln tokenów.

## 1. Zakres: „co najmniej 2 różne modele”

- **Teraz:** w2v IPI PAN i w2v PolEmo. To ten sam algorytm (skip-gram, 100 wymiarów), ale **różne przestrzenie wektorowe**: inny korpus (ogólny NKJP + Wikipedia kontra recenzje PolEmo), inna wielkość korpusu i słownika.
- **Ryzyko interpretacji wymogu:** prowadzący może oczekiwać modeli różnych **typów** (word2vec, fastText, GloVe). Dlatego moduł porówna **dowolną liczbę modeli parami**, a fastText KGR10 dołączy się tym samym poleceniem (`--models w2v w2v_polemo fasttext`) po pobraniu 3,7 GB (decyzja D1).

## 2. Słowa zapytania

- **Pełny zestaw:** słowa z `word_label ≠ O` obecne we wszystkich porównywanych modelach (dla dwóch modeli word2vec: 215). Wszystkie listy trafiają do pliku CSV.
- **Zestaw pokazowy do raportu (~24 słowa):** po 4 najczęstsze słowa z każdej etykiety (ODW ma tylko 4) oraz słowa wybrane ręcznie, które dobrze ilustrują różnice: *polecam, miła, brudne, problem, bardzo, trochę, niestety, nie*. Słowo „nie” ma `word_label = O` (44% anotacji ODW), ale jest kluczowe dla negacji, więc dodajemy je jawnie.
- Wektor słowa zapytania bierzemy w formie znalezionej w pkt 1 (`<model>_form`), np. z wielką literą dla nazw.

## 3. Generowanie list

- **k = 10** najbliższych słów (kosinus) w **całym słowniku modelu** (IPI PAN 300 tys., PolEmo 21,7 tys.).
- **Filtr sąsiadów:** tylko słowa z liter (dopuszczalny łącznik), co usuwa liczby i kody typu „430-31-43”. Pomijamy samo słowo zapytania bez względu na wielkość liter. Gensim pobiera 3·k kandydatów, a po filtrze bierzemy k.
- **Formy fleksyjne zostają, ale z flagą `same_stem`.** Modele „forms” często dają jako sąsiadów inne formy tego samego słowa (miła → miły, miłą). Reguła przybliżona, bo nie mamy lematyzatora: wspólny początek wyrazu długości co najmniej max(4, 60% krótszego słowa). Udział takich sąsiadów to jedna z miar porównania.

## 4. Porównanie list i modeli

Dla każdego słowa zapytania i każdej pary modeli, potem średnio według etykiety i ogółem:

| Miara | Co mierzy | Uwagi |
|---|---|---|
| **Pokrycie@k** = \|N_A ∩ N_B\| / k oraz **Jaccard@k** | czy modele wskazują tych samych sąsiadów | zaniżone przez różnicę słowników: sąsiad z IPI PAN może nie istnieć w PolEmo |
| **Pokrycie@k na wspólnym słowniku** | to samo, ale sąsiadów szukamy tylko wśród słów znanych obu modelom | uczciwsze porównanie; własne wyszukiwanie macierzowe (kosinus na wspólnym słowniku) |
| **Podobieństwo drugiego rzędu** (Spearman ρ) | czy modele podobnie *układają* słowa zapytania względem siebie: dla każdego słowa porównujemy jego wektor podobieństw do pozostałych 214 słów w modelu A i B; dodatkowo ρ dla całych macierzy 215×215 | niezależne od różnic słowników |
| **Udział form fleksyjnych** (`same_stem`) | ile listy to odmiany tego samego słowa | duży udział = mniej informacji semantycznej |
| **Zgodność wydźwięku sąsiadów** | czy sąsiedzi słowa WYR_POZ (WYR_NEG) mają wydźwięk pozytywny (negatywny) | wydźwięk sąsiada z etykiet całego PolEmo (p. 4.1); udział zgodnych, przeciwnych („wtrącenie antonimów”) i neutralnych oraz pokrycie |

### 4.1 Wydźwięk słowa z PolEmo (do oceny sąsiadów)
- s(w) = (liczba zdań POZ − liczba zdań NEG) / liczba wszystkich zdań z w, liczone na 45 935 zdaniach PolEmo z etykietami gold, tylko dla słów z **co najmniej 5 wystąpieniami**. Pozostałe słowa są „bez oceny”.
- Próg: s ≥ +0,1 oznacza pozytywny, s ≤ −0,1 negatywny, pomiędzy neutralny. Do sprawdzenia, czy wnioski nie zmieniają się przy progu 0,2.
- **Zastrzeżenie:** model PolEmo i miara s(w) pochodzą z tego samego korpusu, więc przewaga modelu PolEmo w tej mierze jest częściowo „z założenia”. Trzeba to napisać w raporcie.

## 5. Pliki wynikowe (`results/word_embeddings/`)

| Plik | Zawartość |
|---|---|
| `neighbours_<model>.csv` | `query, query_label, query_freq, rank, neighbour, cosine, same_stem, polemo_freq, polarity` (215 × 10 wierszy na model) |
| `knn_comparison.csv` | dla każdego słowa i pary modeli: pokrycie@k, Jaccard@k, pokrycie na wspólnym słowniku, ρ drugiego rzędu, udział form, zgodność wydźwięku |
| `knn_summary.md` | (1) tabela zbiorcza miar według etykiety i ogółem; (2) **listy pokazowe obok siebie** (top 5 z każdego modelu dla ~24 słów); (3) 5 słów z największą i 5 z najmniejszą zgodnością modeli; (4) przykłady sąsiadów o przeciwnym wydźwięku; (5) krótki komentarz do dyskusji |
| `knn_summary.json` | wszystkie liczby i parametry (k, próg wydźwięku, minimalna częstość) |

Pliki CSV i MD trafiają do repozytorium, więc zespół może pisać dyskusję bez pobierania modeli.

## 6. Implementacja: `src/word_neighbours.py` (nowy)

| Funkcja | Zadanie |
|---|---|
| `query_words(vocab, models, extra)` | zestaw pełny (word_label ≠ O we wszystkich modelach) i pokazowy |
| `neighbours(kv, form, k, exclude)` | sąsiedzi z filtrem słów z liter, bez samego słowa |
| `same_stem(a, b)` | reguła wspólnego początku wyrazu (p. 3) |
| `shared_vocab_neighbours(kv_a, kv_b, queries, k)` | wyszukiwanie na wspólnym słowniku (macierz znormalizowanych wektorów) |
| `overlap_metrics(list_a, list_b)` | pokrycie@k, Jaccard@k |
| `second_order(kv_a, kv_b, queries)` | ρ dla każdego słowa i dla całych macierzy |
| `polemo_polarity(min_count)` | s(w) z korpusu PolEmo (`data/processed/polemo_sentences.parquet`) |
| `summary_markdown(...)` | `knn_summary.md` |
| `main()` | CLI: `--models` (≥ 2, wszystkie pary), `--k`, `--polarity-threshold`, `--min-count` |

Modele wczytują istniejące funkcje z pkt 1 (`word_embeddings.MODELS`). Czas działania dla dwóch modeli word2vec to kilkanaście sekund.

## 7. Testy (`tests/test_word_neighbours.py`, na sztucznych wektorach)

1. Filtr sąsiadów: liczby i kody są pomijane, słowo zapytania (także wielką literą) nie trafia na listę, lista ma dokładnie k słów.
2. `same_stem`: miła/miły → tak; dobry/dobrze → tak; dobry/doskonały → nie; krótkie słowa (≤ 3 litery) → nie.
3. Pokrycie i Jaccard na ręcznie policzonych przykładach; identyczne listy dają 1, rozłączne 0.
4. Wspólny słownik: sąsiad spoza słownika jednego z modeli nie pojawia się w wynikach.
5. ρ drugiego rzędu: model porównany sam ze sobą daje 1; model z przetasowanymi wektorami daje wyraźnie mniej.
6. `polemo_polarity` na małym korpusie: wartości s(w), próg minimalnej częstości, klasy przy progu.
7. `knn_summary.md` zawiera tabelę zbiorczą i listy pokazowe.
8. Dane rzeczywiste (pomijane bez modeli):
   - 215 słów zapytania, po 10 sąsiadów,
   - żadne słowo nie jest swoim sąsiadem,
   - wszystkie wartości kosinusa ≤ 1.

## 8. Uruchomienie (po implementacji)

```bash
python -m src.word_neighbours --models w2v w2v_polemo            # pkt 3 dla word2vec
python -m src.word_neighbours --models w2v w2v_polemo fasttext   # po pobraniu fastText: 3 modele, 3 pary
python -m pytest tests/test_word_neighbours.py -q
```

**Gotowe, gdy:**
- `neighbours_*.csv` ma 215 × 10 wierszy na model,
- `knn_summary.md` zawiera tabelę porównania, listy pokazowe i przykłady,
- liczby z `knn_summary.json` zgadzają się z CSV,
- testy przechodzą.

## 9. Związek z pkt 4 (dyskusja)

| Pytanie pomocnicze | Materiał z pkt 3 |
|---|---|
| 1. Czy osadzenia ogólne dobrze odwzorowują znaczenie słów w korpusie dziedzinowym? | IPI PAN: synonimy ogólne; PolEmo: znaczenie „dziedzinowe” (miła → kompetentna, profesjonalna) |
| 2. Czy słowa pozytywne i negatywne się grupują? | zgodność wydźwięku sąsiadów, wtrącenia antonimów (miła → niemiła) |
| **3. Różnice między listami i ich przyczyny** | pokrycie@k (surowe i na wspólnym słowniku), ρ drugiego rzędu, udział form fleksyjnych; przyczyny: korpus (ogólny / dziedzinowy), jego wielkość (szum dla rzadkich słów), modele „forms” (odmiany), kontekstowa natura word2vec (antonimy) |
| 4. Jak ulepszyć osadzenia? | dotrenowanie modelu ogólnego na PolEmo, lematy zamiast form, fastText (n-gramy), osadzenia uczone z etykietami wydźwięku |

## 10. Czas

| Krok | Czas |
|---|---|
| `word_neighbours.py`: listy, filtr, formy, wspólny słownik | ~45 min |
| miary porównania i wydźwięk z PolEmo | ~30 min |
| raport `knn_summary.md` | ~20 min |
| testy | ~30 min |
| uruchomienie, przegląd, README | ~15 min |

## 11. Decyzje do potwierdzenia

| # | Pytanie | Opcje | Rekomendacja |
|---|---|---|---|
| D1 | Które modele | (a) w2v IPI PAN + w2v PolEmo teraz, fastText dołączyć po pobraniu tym samym poleceniem / (b) tylko dwa modele word2vec / (c) czekać z pkt 3 na fastText | **(a)**: dwa modele word2vec spełniają wymóg formalnie (różne przestrzenie), a fastText usuwa wątpliwość co do „różnych modeli” |
| D2 | Słowa zapytania | `word_label ≠ O` we wszystkich modelach (215) / `annot_share > 0` (szerszy, ok. 260) / tylko słowa o częstości ≥ 2 (39) | **`word_label ≠ O` (215)** oraz zestaw pokazowy z „nie” |
| D3 | k | 5 / 10 / 20 | **10** w CSV i miarach; w raporcie top 5 |
| D4 | Formy fleksyjne wśród sąsiadów | zostawić i oznaczyć / usunąć z list | **zostawić i oznaczyć**: ich udział to ważna różnica między modelami |
| D5 | Miary porównania | pokrycie (surowe i na wspólnym słowniku) + ρ drugiego rzędu / tylko pokrycie / + RBO (miara ważąca kolejność na liście) | **pokrycie surowe i na wspólnym słowniku + ρ**; RBO niepotrzebne |
| D6 | Zgodność wydźwięku sąsiadów (wydźwięk z PolEmo) | tak / nie | **tak**: łączy listy z anotacją i odpowiada na pytanie 2; z zastrzeżeniem o wspólnym korpusie |
| D7 | Zestaw pokazowy | automatyczny (top 4 na etykietę) + słowa wybrane ręcznie / tylko automatyczny | **automatyczny + ręczny** (polecam, miła, brudne, problem, bardzo, trochę, niestety, nie) |
