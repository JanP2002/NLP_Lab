# Text Embeddings, pkt 3: odpowiedzi na pytania pomocnicze (uśrednione Word2Vec)

Modele:
- **IPI PAN**: ogólny word2vec, NKJP + Wikipedia, 300 tys. form,
- **PolEmo**: word2vec wytrenowany na recenzjach PolEmo.

Wektor zdania to średnia wektorów jego słów. Liczby pochodzą z [`analiza_w2v.md`](analiza_w2v.md) (`python -m src.text_analysis`). Dla porównania podaję wyniki zespołu dla fastText i USE z tych samych par kontrolnych, zdań i miar. Odpowiedzi opierają się na 200 zdaniach i 12 parach kontrolnych, więc są ilustracją, a nie statystyką. Tekst pisany ręcznie: po ponownym uruchomieniu analizy trzeba sprawdzić, czy liczby się zgadzają.

## 1. Czy są zdania, które model uznał za podobne, a różnią się znaczeniem?

**Tak, i to systematycznie.**
- **Pary kontrolne.** Pary o przeciwnym sensie i podobnych słowach są bliższe niż prawdziwe parafrazy:
  - średnio 0,882 wobec 0,710 (IPI PAN) i 0,877 wobec 0,700 (PolEmo),
  - „Obsługa była bardzo miła” i „…bardzo niemiła”: 0,97 / 0,96,
  - „Jedzenie było smaczne” i „…niesmaczne”: 0,93 / 0,97,
  - „Polecam ten hotel każdemu” i „Nie polecam tego hotelu nikomu”: 0,74 / 0,70, czyli tyle co parafrazy,
  - zamiana ocen („obsługa świetna, pokój fatalny” i odwrotnie): 0,95 / 0,93. Średnia nie widzi kolejności słów, więc to prawie ten sam wektor.

  Zespół zaobserwował to samo w fastText (0,859 wobec 0,746) i USE.
- **Pary z korpusu.** Wśród 1% najbardziej podobnych par (199 par):
  - pary POZ–NEG stanowią 15,1% (IPI PAN) i 16,1% (PolEmo), więcej niż wśród wszystkich par (12,5%),
  - pary o tej samej etykiecie stanowią 29–31%, tyle co przypadkiem (30,2%).

  Bliskość wektorów w ogóle nie odróżnia więc wydźwięku. Najbliższe pary POZ–NEG (cos 0,90–0,92) to długie opinie o lekarzach, np. „…jest to wyjątkowy lekarz…” (POZ) i „Myślę, że lekarz powinien dokładnie wysłuchać pacjenta…” (NEG): wspólny temat i słownictwo, przeciwna ocena.
- **Przyczyna.** Średnia wektorów słów koduje temat, a nie wydźwięk:
  - słowa o przeciwnym wydźwięku leżą w word2vec blisko siebie (cos(dobry, zły) = 0,73, część Word Embeddings),
  - „nie” to tylko jedno słowo w średniej, które nie zmienia wektorów pozostałych słów.

## 2. Czy są zdania podobne znaczeniowo, a uznane za odległe, bo użyto innych słów?

**Częściowo.**
- **Parafrazy kontra zdania niepowiązane.** Parafrazy dostają 0,71 / 0,70, wyraźnie więcej niż pary niepowiązane (0,43 / 0,43). Ten kontrast jest większy niż w fastText (0,746 wobec 0,506), bo word2vec zna synonimy (miła ≈ uprzejma).
- **Zależność od wspólnych słów.**
  - Korelacja Spearmana podobieństwa ze wspólnymi słowami (Jaccard zbiorów słów) wynosi 0,45 / 0,50.
  - Pary bez wspólnych słów mają średnio 0,62 / 0,57, a pary o Jaccardzie > 0,1 mają 0,77 / 0,74.
  - Inne słowa obniżają więc podobieństwo, choć nie do zera, bo wszystkie zdania leżą blisko średniego wektora (pyt. 4).
- **Przykład odległości z powodu innych form słów.** Zdania „Ogólnie Czystość i pokoje na 10, atmosfera na 3.” i „Czysto, schludnie, bardzo wygodne łóżka.”:
  - w IPI PAN leżą daleko od siebie (ranga 0,15 wśród wszystkich par),
  - w PolEmo blisko (0,94).

  Model form traktuje „czystość” i „czysto” jak różne słowa, a model uczony na recenzjach łączy je przez kontekst.
- **Przykład odwrotny.** IPI PAN zbliża zdania o czasie („Używam jej od ok 2 lat.” i „A było to 2 miesiące temu.”, ranga 0,95), których PolEmo nie łączy (0,14). Czy para jest naprawdę podobna znaczeniowo, trzeba ocenić ręcznie, bo modele różnią się tym, które relacje między słowami znają.

## 3. Czy zdania o różnej długości i podobnym sensie są traktowane podobnie?

**Nie w pełni.**
- **Średnie wartości.** Pary „krótkie zdanie i jego długa wersja” mają średnio 0,58 (IPI PAN) i 0,66 (PolEmo). To mniej niż parafrazy (0,71 / 0,70), choć więcej niż zdania niepowiązane (0,43).
- **Przykład.** „Polecam.” i długa pochwała hotelu mają 0,50 / 0,60.
- **Przyczyna.** Krótkie zdanie to wektor jednego lub kilku słów. Długie to średnia wielu słów (hotel, znajomi, rodzina, tydzień), które odciągają wektor od oceny i zbliżają go do przeciętnego zdania korpusu.
- **Różnica między modelami.** Model PolEmo radzi sobie trochę lepiej, bo jego słowa niosą kontekst recenzji („polecam” leży blisko słów o hotelach).
- **Porównanie z innymi modelami.** fastText daje 0,60, a USE tylko 0,19.

## 4. Czy długość zdania wpływa na podobieństwo i „wyrazistość” wektora?

**Tak, wyraźnie, i silniej niż w fastText.** Od zdań krótkich (5–11 tokenów) do długich (21–36):
- **norma wektora maleje:** 2,13 → 1,81 (IPI PAN) i 2,35 → 1,91 (PolEmo),
- **podobieństwo do średniego wektora korpusu rośnie:** 0,747 → 0,885 i 0,718 → 0,847,
- **średnie podobieństwo do pozostałych zdań rośnie:** 0,61 → 0,73,
- **korelacja Spearmana** długości ze średnim podobieństwem: 0,77 / 0,76, wobec 0,59 w fastText i 0,32 w USE.

**Przyczyna.** W średniej wielu słów ich wkłady częściowo się znoszą, więc wektor długiego zdania jest krótszy i bliższy „przeciętnemu zdaniu”, czyli mniej wyrazisty.

**Konsekwencje:**
- najbardziej podobne pary w korpusie to głównie długie zdania (pyt. 1),
- wysokie podobieństwo może wynikać z samej długości, a nie z sensu,
- pomogłoby ważenie słów (rzadsze ważniejsze) i usunięcie wspólnej składowej wszystkich zdań, jak w metodzie SIF.

## 5. Czy osadzenia nadają się do wybranego zadania (wydźwięk zdania)?

**Nie samodzielnie.**
- **Wyniki klasyfikatora.** Regresja logistyczna na samych osadzeniach (etykiety `final`):

  | Model | Trafność | Baza większościowa | Macro-F1 | Czystość 10 sąsiadów (losowo 0,305) | Sylwetka |
  |---|---|---|---|---|---|
  | IPI PAN | 0,505 | 0,415 | 0,31 | 0,33 | ujemna |
  | PolEmo | 0,540 | 0,415 | 0,35 | 0,36 | ujemna |

  Przewaga nad bazą większościową to +0,09 i +0,13.
- **Czułość dla poszczególnych klas.** Klasyfikator wybiera głównie NEU (czułość 0,89). POZ rozpoznaje w 44% / 54% przypadków, NEG prawie wcale (7% / 11%), AMB w ogóle. To zgodne z pyt. 1: negacja i antonimy nie zmieniają średniej.
- **Porównanie z innymi modelami.** Model PolEmo wypada najlepiej z pięciu porównanych (USE 0,520, fastText 0,505, TF-IDF 0,490). Uczono go jednak na tym samym korpusie recenzji, więc ma przewagę „dziedzinową”.
- **Do czego się nadają.** Najwyżej jako cecha pomocnicza (temat, dziedzina recenzji).
- **Co jest potrzebne zamiast tego:**
  - model uwzględniający kolejność słów i negację, np. dostrojony transformer (HerBERT),
  - albo osadzenia uczone z etykietami wydźwięku.

## Właściwości badanych modeli: podsumowanie

| | Word2Vec IPI PAN (ogólny) | Word2Vec PolEmo (dziedzinowy) |
|---|---|---|
| co koduje wektor zdania | temat i słownictwo; relacje słownikowe (synonimy, wyrażenia czasu) | temat i skojarzenia z recenzji (czystość, łazienka, lekarz) |
| wydźwięk | prawie niewidoczny (antonimy blisko, negacja ginie w średniej) | nieco lepszy (najwyższa trafność), nadal słaby dla NEG |
| długość zdania | silny efekt: długie zdania zbliżają się do średniej korpusu | to samo |
| słownik | 96,7% słów w zdaniach ma wektor | 91,0%; słowa rzadkie (< 3 wystąpienia) pominięte |
