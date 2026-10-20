# Plan: Część II, Word Embeddings, pkt 4: dyskusja wyników (word2vec)

Treść punktu: *„Opracować krótką dyskusję otrzymanych rezultatów [0,5 pkt]”* z czterema pytaniami pomocniczymi:
1. Czy osadzenia pre-trenowane na dużych korpusach dobrze odwzorowują znaczenie słów w korpusie dziedzinowym?
2. Czy w korpusie występuje grupowanie słów związanych z anotacją (pozytywne / negatywne)? Jeśli nie, dlaczego osadzenia mogą nie oddawać emocjonalnego znaczenia słów?
3. Na czym polegają różnice między listami podobieństwa z różnych modeli i z czego wynikają?
4. Jak można ulepszyć osadzenia dla tego zadania?

Gałąź: `czesc2-w2v` (pkt 1–3 zrobione, bez commita).

## 0. Punkt wyjścia

| Element | Stan |
|---|---|
| Wyniki pkt 1–3 | `results/word_embeddings/`: `coverage.md`, `sanity.md`, `label_separation.md`, `knn_summary.md`, wykresy `tsne_words_*.png/.html`, `vocab.csv`, `neighbours_*.csv` |
| Sprawozdanie (`report/sprawozdanie.tex`, autor szablonu: Wysciguvvka) | rozdział „Część II: Word Embeddings” ma ogólne `\todo` i trzy miejsca `\inputorTodo{we_models}`, `{we_tsne}`, `{we_knn}` (plik `report/generated/<nazwa>.tex`, a gdy go brak, ramka „DO UZUPEŁNIENIA”). **Brak miejsca na dyskusję**; w „Wnioskach” jest `\todo{wnioski z części Word Embeddings.}` |
| Styl zespołu (Text Embeddings) | pogrubione numerowane pytania, odpowiedź 4–8 zdań z liczbami w tekście, tabele z `generated/*.tex`, liczby jako makra (`generated/macros.tex`, np. `\accFasttext`) generowane przez `report/build_assets.py` |
| Narzędzia | **brak kompilatora LaTeX** na tym komputerze, więc pliki `.tex` sprawdzamy testami składni |
| Bibliografia | `thebibliography{9}` z 6 pozycjami (Cohen, Fleiss, Landis i Koch, PolEmo, fastText, USE); brak word2vec i modeli IPI PAN |

**Wstępna analiza, sprawdzona 20.10 (do decyzji D4):**
- **Sonda liniowa** (regresja logistyczna, 5-krotna walidacja, zrównoważona trafność) WYR_POZ vs WYR_NEG z wektorów słów: IPI PAN **0,72**, przy losowych etykietach 0,51 [0,48; 0,54]; PolEmo 0,58. Wydźwięk **jest częściowo zakodowany** w wektorach, choć nie tworzy skupisk ani sąsiedztw.
- **Połączenie wektorów IPI PAN i PolEmo:** czystość k-NN 0,70 wobec 0,67 i 0,68, sonda bez poprawy. Brak wyraźnego zysku, więc nie warto tego pokazywać.

## 1. Forma

Dyskusja trafia do **sprawozdania LaTeX** (to jest oddawany dokument), w stylu części Text Embeddings, i ma ok. **1 strony**. Rozdział „Część II: Word Embeddings” dostaje 4 podrozdziały:

| Podrozdział | Pkt | Plik | Źródło |
|---|---|---|---|
| Modele i pokrycie słownika | 1 | `generated/we_models.tex` (istniejące miejsce) | generowany: opis 2 modeli, tabela pokrycia (ogółem, dla każdej etykiety), przykłady słów bez wektora |
| t-SNE słów | 2 | `generated/we_tsne.tex` (istniejące miejsce) | generowany: rysunek `tsne_words_w2v_all.png` (+ PolEmo), tabela separacji etykiet (+ sonda, D4) |
| Listy najbliższych słów | 3 | `generated/we_knn.tex` (istniejące miejsce) | generowany: tabela zgodności modeli + lista pokazowa (ok. 10 słów, top 5) |
| **Dyskusja** | **4** | **`sections/we_dyskusja.tex` (nowy, pisany ręcznie)** | tekst z odpowiedziami na 4 pytania; liczby jako makra z `generated/we_macros.tex` |

- Liczby w tekście pochodzą z makr, więc po dodaniu fastText lub ponownym uruchomieniu pkt 1–3 wystarczy przegenerować pliki, bez ręcznej edycji tekstu.
- Makra mają osobny plik `we_macros.tex`, a nie dopisek do `macros.tex` zespołu, więc skrypty nie nadpisują sobie nawzajem plików.

## 2. Zarys treści dyskusji (odpowiedzi do zatwierdzenia)

**1. Czy osadzenia ogólne dobrze odwzorowują znaczenie słów w korpusie dziedzinowym?** *Częściowo.*
- Pokrycie jest wysokie: 94,6% słów, 97,1% tokenów, 100% WZM/OSL/ODW. Braki to słowa rzadkie, z cyframi i bez polskich znaków („lazience”).
- Znaczenie ogólne jest trafne: lekarz → chirurg, internista; brud → smród, kurz.
- Model ogólny podaje jednak znaczenie słownikowe, a nie oceniające ani dziedzinowe (miła → sympatyczna, życzliwa). Mały model dziedzinowy łapie kontekst recenzji (miła → profesjonalna, kompetentna; brudne → wykładziny, kafelki, pleśń), ale ma słownik tylko 21,7 tys. słów i pokrycie 85%.
- Homonimia: „ok” (WYR_POZ) oba modele odczytują jako „około”.

**2. Czy słowa grupują się według anotacji?** *Słabo; wyraźnych skupisk nie ma.*
- Na wykresie t-SNE punkty WYR_POZ i WYR_NEG są wymieszane.
- Czystość 10 najbliższych sąsiadów: 0,65 wobec 0,54 przy losowych etykietach (p = 0,001), ale sylwetka ≈ 0,03. Skupiają się głównie słowa pozytywne (0,75), negatywne nie (0,47).
- Antonimy leżą blisko: cos(dobry, zły) = 0,73. Aż 41% sąsiadów słów pozytywnych w IPI PAN ma wydźwięk negatywny (polecam → odradzam, zadowolona → niezadowolona).
- **Dlaczego:** word2vec uczy się przewidywać kontekst (hipoteza dystrybucyjna), a antonimy występują w tych samych kontekstach („obsługa była miła/niemiła”).
- Wydźwięk zależy od dziedziny i negacji („nie polecam”). Etykieta słowa zależy od zdania („nie” jest ODW tylko w 44% anotacji).
- Mimo to sonda liniowa odczytuje wydźwięk z trafnością 0,72 (D4): informacja jest w wektorach, ale nie w sąsiedztwie.
- Słowa grupują się raczej według **funkcji**: modyfikatory (bardzo, trochę, całkiem) tworzą zwarte skupisko.

**3. Różnice między listami i ich przyczyny:**
- Modele prawie się nie zgadzają: wspólne sąsiedztwo 9% (12% na wspólnym słowniku), ρ drugiego rzędu 0,27.
- Przyczyny:
  - **korpus**: ogólny daje relacje słownikowe (synonimy, antonimy, kohiponimy), recenzje dają skojarzenia tematyczne,
  - **wielkość korpusu**: 0,8 mln tokenów daje szum dla rzadszych słów (problem → kulinarnie, zdrowotny),
  - **słownik**: model form IPI PAN w 13% podaje odmiany tego samego słowa, model PolEmo w 2%,
  - **pisownia**: formy bez polskich znaków tworzą osobne sąsiedztwa (zaden → kazdy).
- Najbardziej zgodne są wzmacniacze (16–18%), bo ich użycie mało zależy od dziedziny.

**4. Jak ulepszyć osadzenia:**
- **adaptacja do dziedziny:** dotrenowanie modelu ogólnego na recenzjach (pełne recenzje `klej-polemo2-in`) zamiast osobnego, małego modelu,
- **wiedza o wydźwięku:** *retrofitting* lub *counter-fitting* z leksykonem wydźwięku (np. emocjonalna Słowosieć), który rozsuwa antonimy, albo osadzenia uczone z etykietami (*sentiment-specific embeddings*) na 46 tys. zdań PolEmo,
- **fastText** (n-gramy znakowe) dla literówek i form bez polskich znaków albo normalizacja diakrytyków,
- **lematyzacja** (Morfeusz) albo modele „lemmas”, żeby odmiany nie zajmowały list,
- **negacja jako jedna jednostka** („nie_polecam”) przed treningiem,
- **modele kontekstowe** (HerBERT, Polish RoBERTa), które rozróżniają znaczenia („ok” jako aprobata a skrót „około”).

**Wnioski (2–3 punkty do „Wniosków”, zamiast `\todo`):**
- osadzenia word2vec dobrze pokrywają słownictwo i oddają znaczenie ogólne, ale nie wydźwięk: antonimy są blisko, a grupowanie według etykiet jest słabe,
- model ogólny i dziedzinowy dają niemal rozłączne listy podobieństwa: relacje słownikowe kontra skojarzenia tematyczne,
- do zadań wydźwięku potrzebna jest adaptacja do dziedziny i informacja o wydźwięku (lub modele kontekstowe).

## 3. Implementacja

| Plik | Zmiana |
|---|---|
| `src/word_report.py` **(nowy)** | buduje z wyników pkt 1–3: `report/generated/we_macros.tex` (liczby), `we_models.tex`, `we_tsne.tex`, `we_knn.tex`; kopiuje `tsne_words_w2v_all.png` (+ PolEmo) do `report/figures/`; zamienia `_`, `%`, `&` na postać bezpieczną dla LaTeX; liczby z przecinkiem dziesiętnym (`0{,}648`, jak w `macros.tex`) |
| `src/word_tsne.py` | (D4) `polarity_probe()`: sonda liniowa z permutacyjnym punktem odniesienia; wynik w `label_separation.md/json` |
| `report/sections/we_dyskusja.tex` **(nowy)** | tekst dyskusji (p. 2), liczby wyłącznie przez makra `\we…` |
| `report/sprawozdanie.tex` (D3) | (1) w preambule `\IfFileExists{generated/we_macros.tex}{\input{generated/we_macros}}{}`; (2) ogólne `\todo` w rozdziale WE → krótki wstęp, a po `we_knn` dyskusja (`\IfFileExists{sections/we_dyskusja.tex}{…}{\todo{…}}`); (3) w „Wnioskach” `\todo` → 3 punkty; (4) w części o narzędziach opartych na modelach językowych: dopisek o kodzie Word Embeddings; (5) bibliografia: Mikolov i in. 2013 (word2vec), modele IPI PAN (do zweryfikowania na dsmodels.nlp.ipipan.waw.pl, jaką publikację cytować), opcjonalnie Faruqui i in. 2015 (retrofitting); `thebibliography{99}` |
| `tests/test_word_report.py` **(nowy)** | testy z p. 4 |
| `README_WordEmbeddings.md` | odnośnik do sprawozdania, polecenie budowania |

Polecenia (po implementacji):
```bash
python -m src.word_tsne                    # (D4) przelicza label_separation z sondą liniową
python -m src.word_report                  # report/generated/we_*.tex + figures/tsne_words_*.png
python -m pytest tests/test_word_report.py -q
# kompilacja sprawozdania u osoby z LaTeX-em / w Overleaf: pdflatex sprawozdanie.tex (2×)
```

## 4. Testy (bez kompilatora LaTeX)

1. Wszystkie makra użyte w `sections/we_dyskusja.tex` i `generated/we_*.tex` są zdefiniowane w `we_macros.tex` i odwrotnie (wyrażenia regularne).
2. Wygenerowane `.tex`:
   - nawiasy `{}` są zbalansowane,
   - każde `\begin{…}` ma `\end{…}`,
   - nie ma niezabezpieczonych `_`, `%`, `&` poza tabelami i komentarzami,
   - liczby mają przecinek dziesiętny.
3. Wartości makr zgadzają się z `coverage.json`, `label_separation.json`, `knn_summary.json` (np. `\weCovTypes` = 94,6).
4. Rysunki, do których odwołują się `.tex`, istnieją w `report/figures/`.
5. Każde z 4 pytań ma w `we_dyskusja.tex` pogrubiony nagłówek, a tekst ma ok. 1 strony (liczba słów w przedziale 400–750).
6. (D4) `polarity_probe` na sztucznych danych: separowalne skupiska dają trafność bliską 1, przetasowane etykiety bliską 0,5.

## 5. Kolejność i czas

| Krok | Czas |
|---|---|
| (D4) sonda liniowa w `word_tsne.py` + test | 15 min |
| `word_report.py`: makra i 3 pliki `.tex` + kopie rysunków | 45 min |
| `sections/we_dyskusja.tex` (tekst, ok. 1 strony) | 40 min |
| zmiany w `sprawozdanie.tex`, bibliografia | 15 min |
| testy, README | 25 min |

Kompilację PDF musi wykonać osoba z LaTeX-em (np. Overleaf), bo tu nie ma kompilatora. Zalecam przekazać to autorowi szablonu razem z prośbą o przegląd tekstu.

## 6. Decyzje do potwierdzenia

| # | Pytanie | Opcje | Rekomendacja |
|---|---|---|---|
| D1 | Gdzie dyskusja | sprawozdanie LaTeX / tylko Markdown w repozytorium / oba | **sprawozdanie LaTeX** (oddawany dokument); README zostaje skrótem |
| D2 | Wypełnić też miejsca na wyniki pkt 1–3 (`we_models`, `we_tsne`, `we_knn`) | tak / nie (tylko dyskusja) | **tak**: dyskusja odwołuje się do tabel i rysunków, a miejsca na nie już czekają |
| D3 | Zmiany w `sprawozdanie.tex` autora szablonu | 5 drobnych zmian (p. 3) / tylko pliki w `generated/` i `sections/`, a wpięcie zrobi autor | **5 drobnych zmian w osobnym commicie** do przejrzenia przez Wysciguvvka |
| D4 | Dodatkowa analiza | sonda liniowa (0,72 wobec 0,51) / + połączone wektory / nic | **sonda liniowa**: pokazuje, że wydźwięk jest w wektorach, choć nie w sąsiedztwie; połączone wektory pominąć (brak zysku) |
| D5 | fastText | dyskusja teraz dla word2vec, rozszerzenie po dodaniu fastText (makra) / czekać na fastText | **teraz word2vec**; struktura tabel i makr gotowa na trzeci model |
| D6 | Długość i styl | ok. 1 strony, 4 pogrubione pytania + 3 punkty we „Wnioskach” / dłużej | **ok. 1 strony**, jak w części Text Embeddings |
| D7 | Bibliografia | dodać word2vec + modele IPI PAN (+ retrofitting) / bez zmian | **dodać 2–3 pozycje**; publikację dla modeli IPI PAN sprawdzić na ich stronie przed wpisaniem |
