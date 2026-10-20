# Text Embeddings, pkt 3: analiza osadzeń (Word2Vec IPI PAN, Word2Vec PolEmo)

200 zaanotowanych zdań (iteracje 1–2), etykieta `final` = uzgodniona. Kolumny fastText / USE / TF-IDF – wyniki zespołu z `report/generated/emb_*.tex` (ten sam zestaw par, te same zdania i miary).

## Pary kontrolne (`PROBE_PAIRS`): średni kosinus według kategorii (pyt. 1–3)

| Kategoria | Word2Vec IPI PAN | Word2Vec PolEmo | fastText | USE | TF-IDF |
|---|---|---|---|---|---|
| parafraza | 0,710 | 0,700 | 0,746 | 0,492 | 0,376 |
| przeciwny sens, podobne słowa | 0,882 | 0,877 | 0,859 | 0,794 | 0,744 |
| przeciwny sens, te same słowa | 0,950 | 0,927 | 0,892 | 0,924 | 0,770 |
| ten sam sens, różna długość | 0,578 | 0,658 | 0,604 | 0,191 | 0,389 |
| kontrola: niepowiązane | 0,428 | 0,431 | 0,506 | 0,111 | 0,004 |

Pary pojedynczo:

| Kategoria | Zdanie a | Zdanie b | Word2Vec IPI PAN | Word2Vec PolEmo |
|---|---|---|---|---|
| parafraza | Obsługa była bardzo miła i pomocna . | Personel okazał się niezwykle uprzejmy i życzliwy . | 0,65 | 0,73 |
| parafraza | Pokój był czysty i dobrze wyposażony . | W pokoju panowała czystość , a wyposażenie było kompletne . | 0,70 | 0,57 |
| parafraza | Lekarz dokładnie mnie zbadał i wszystko wytłumaczył . | Doktor przeprowadził staranne badanie i cierpliwie objaśnił wyniki . | 0,78 | 0,81 |
| przeciwny sens, podobne słowa | Polecam ten hotel każdemu . | Nie polecam tego hotelu nikomu . | 0,74 | 0,70 |
| przeciwny sens, podobne słowa | Jedzenie było smaczne . | Jedzenie było niesmaczne . | 0,93 | 0,97 |
| przeciwny sens, podobne słowa | Obsługa była bardzo miła . | Obsługa była bardzo niemiła . | 0,97 | 0,96 |
| przeciwny sens, te same słowa | Obsługa była świetna , ale pokój fatalny . | Pokój był świetny , ale obsługa fatalna . | 0,95 | 0,93 |
| ten sam sens, różna długość | Polecam . | Szczerze polecam ten hotel wszystkim znajomym i rodzinie , bo spędziliśmy tam naprawdę udany tydzień . | 0,50 | 0,60 |
| ten sam sens, różna długość | Bardzo słaby serwis . | Obsługa w tym miejscu była naprawdę bardzo słaba i musieliśmy długo czekać na wszystko , co zamówiliśmy . | 0,57 | 0,69 |
| ten sam sens, różna długość | Wszystko w porządku . | Pobyt przebiegł bez żadnych problemów , wszystko było tak , jak należy , i nie mamy żadnych zastrzeżeń . | 0,66 | 0,68 |
| kontrola: niepowiązane | Obsługa była bardzo miła . | Dojazd autostradą zajmuje około dwóch godzin . | 0,49 | 0,34 |
| kontrola: niepowiązane | Jedzenie było smaczne . | W pokoju nie działała klimatyzacja . | 0,37 | 0,52 |

## Pyt. 1: zdania uznane za podobne, a różne znaczeniowo

Wśród 1% najbardziej podobnych par zdań (etykiety `final`):

| Model | par | POZ–NEG wśród najbliższych | POZ–NEG wśród wszystkich | ta sama etykieta wśród najbliższych | ta sama etykieta wśród wszystkich |
|---|---|---|---|---|---|
| Word2Vec IPI PAN | 199 | 15,1% | 12,5% | 29,1% | 30,2% |
| Word2Vec PolEmo | 199 | 16,1% | 12,5% | 30,7% | 30,2% |

**Word2Vec IPI PAN: najbardziej podobne pary POZ–NEG:**

| cos | final | Zdanie a | Zdanie b |
|---|---|---|---|
| 0,924 | POZ/NEG | U doktora można się bardzo dużo nauczyć owszem jest wymagający ale jak ma się z czymś pro… | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,912 | POZ/NEG | Była m u niego i prywatnie , i w szpitalu - kobiety w kolejce może i narzekały na czas oc… | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,907 | POZ/NEG | Kilka sytuacji w trakcie mojej ciąży potwierdziły , że jest to wyjątkowy lekarz , taki , … | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,904 | POZ/NEG | U doktora można się bardzo dużo nauczyć owszem jest wymagający ale jak ma się z czymś pro… | Nie wiem jak można pracować na oddziale , TAKIM oddziale i nie mieć dla ludzi ani trochę … |
| 0,901 | POZ/NEG | Pan Bartłomiej wysłuchał długiej historii mojego urazu zadając przy tym pytania o istotne… | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |

**Word2Vec PolEmo: najbardziej podobne pary POZ–NEG:**

| cos | final | Zdanie a | Zdanie b |
|---|---|---|---|
| 0,890 | POZ/NEG | Kilka sytuacji w trakcie mojej ciąży potwierdziły , że jest to wyjątkowy lekarz , taki , … | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,884 | POZ/NEG | Pan Bartłomiej wysłuchał długiej historii mojego urazu zadając przy tym pytania o istotne… | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,884 | POZ/NEG | Była m u niego i prywatnie , i w szpitalu - kobiety w kolejce może i narzekały na czas oc… | Myśle , że lekarz powinien dokładnie wysłuchać pacjenta , co ten ma do powiedzenia na tem… |
| 0,874 | POZ/NEG | U doktora można się bardzo dużo nauczyć owszem jest wymagający ale jak ma się z czymś pro… | Nie wiem jak można pracować na oddziale , TAKIM oddziale i nie mieć dla ludzi ani trochę … |
| 0,868 | POZ/NEG | U doktora można się bardzo dużo nauczyć owszem jest wymagający ale jak ma się z czymś pro… | Podsumowanie : dla ładnego wglądu nie opłaca się wydawać tyle kasy . . . to pierwszy tele… |

## Pyt. 2: podobne znaczeniowo, a odległe, bo inne słowa

Zależność kosinusa od wspólnych słów (Jaccard zbiorów słów, wszystkie pary zdań):

| Model | Spearman (cos vs Jaccard) | śr. cos: brak wspólnych słów | Jaccard (0; 0,1] | Jaccard > 0,1 |
|---|---|---|---|---|
| Word2Vec IPI PAN | 0,452 | 0,624 | 0,724 | 0,771 |
| Word2Vec PolEmo | 0,503 | 0,566 | 0,670 | 0,738 |

Liczba par w grupach: 10221 / 9279 / 400.

**Blisko według Word2Vec IPI PAN, daleko według Word2Vec PolEmo (rangi percentylowe):**

| ranga Word2Vec IPI PAN | ranga Word2Vec PolEmo | gold | Zdanie a | Zdanie b |
|---|---|---|---|---|
| 0,89 | 0,05 | NEG/NEU | To nie jest hotel 4 * , max . 2 * . | Używam jej od ok 2 lat . |
| 0,95 | 0,14 | NEU/NEU | Używam jej od ok 2 lat . | A było to 2 miesiące temu . |
| 0,86 | 0,08 | NEG/NEU | Masakra nie polecam tego miejsca ! ! ! | Przynajmniej wtedy wiadomo co np się dzieje albo co nabroil w moim wypadku . |
| 0,89 | 0,13 | NEG/NEG | Nie tutaj , może raz w tygodniu . | Można po nim się pochorować . |
| 0,77 | 0,01 | AMB/NEU | Ogólnie Czystość i pokoje na 10 , atmosfera na 3 . | Używam jej od ok 2 lat . |

**Blisko według Word2Vec PolEmo, daleko według Word2Vec IPI PAN (rangi percentylowe):**

| ranga Word2Vec PolEmo | ranga Word2Vec IPI PAN | gold | Zdanie a | Zdanie b |
|---|---|---|---|---|
| 0,93 | 0,14 | NEG/POZ | Przy poziomie Gołębiewskiego to rudera . | Przy ktorej mnostwo przeroznych lokalnych Tawern i sklepow . |
| 0,94 | 0,15 | AMB/POZ | Ogólnie Czystość i pokoje na 10 , atmosfera na 3 . | Czysto , schludnie , bardzo wygodne łóżka . |
| 0,82 | 0,05 | NEG/NEG | Nie wiem jak można pracować na oddziale , TAKIM oddziale i nie mieć dla ludzi ani trochę … | Pieniedzy oczywiscie nie chca wrocic |
| 0,89 | 0,12 | AMB/POZ | Przeprasza że tak dość obrazowo , ale konkretnie . | Bardzo dobra rozdzielczość optyczna szkła ! |
| 0,94 | 0,17 | NEU/NEG | jak macie kogoś lepszego to piszcie nazwiska , chętnie skorzystam . | Jeżeli ktoś wybiera się na wakacje z dziećmi tym bardziej ten hotel się nie nadaje , drog… |

## Pyt. 3–4: długość zdania

| Model | Grupa (tokeny) | n | norma | cos do centroidu | śr. cos do innych |
|---|---|---|---|---|---|
| Word2Vec IPI PAN | (4.999, 11.333] | 67 | 2,13 | 0,747 | 0,612 |
| Word2Vec IPI PAN | (11.333, 20.0] | 70 | 1,93 | 0,836 | 0,686 |
| Word2Vec IPI PAN | (20.0, 36.0] | 63 | 1,81 | 0,885 | 0,726 |
| Word2Vec IPI PAN | Spearman (tokeny vs śr. cos) | | | | 0,774 |
| Word2Vec PolEmo | (4.999, 11.333] | 67 | 2,35 | 0,718 | 0,563 |
| Word2Vec PolEmo | (11.333, 20.0] | 70 | 2,05 | 0,800 | 0,628 |
| Word2Vec PolEmo | (20.0, 36.0] | 63 | 1,91 | 0,847 | 0,665 |
| Word2Vec PolEmo | Spearman (tokeny vs śr. cos) | | | | 0,757 |
| fastText (zespół) | (4.999, 11.333] | 67 | 30,30 | 0,763 | 0,624 |
| fastText (zespół) | (11.333, 20.0] | 70 | 28,62 | 0,809 | 0,662 |
| fastText (zespół) | (20.0, 36.0] | 63 | 27,80 | 0,893 | 0,731 |
| fastText (zespół) | Spearman (tokeny vs śr. cos) | | | | 0,588 |
| USE (zespół) | (4.999, 11.333] | 67 | 1,00 | 0,147 | 0,024 |
| USE (zespół) | (11.333, 20.0] | 70 | 1,00 | 0,214 | 0,038 |
| USE (zespół) | (20.0, 36.0] | 63 | 1,00 | 0,235 | 0,042 |
| USE (zespół) | Spearman (tokeny vs śr. cos) | | | | 0,318 |
| TF-IDF (zespół) | (4.999, 11.333] | 67 | 0,76 | 0,120 | 0,014 |
| TF-IDF (zespół) | (11.333, 20.0] | 70 | 0,74 | 0,153 | 0,019 |
| TF-IDF (zespół) | (20.0, 36.0] | 63 | 0,74 | 0,192 | 0,025 |
| TF-IDF (zespół) | Spearman (tokeny vs śr. cos) | | | | 0,589 |

## Pyt. 5: czy osadzenia nadają się do zadania (wydźwięk zdania)

Regresja logistyczna na samych osadzeniach (5-krotna walidacja krzyżowa), etykiety `final`:

| Model | accuracy | baza większościowa | macro-F1 | kNN-10 | losowy | silhouette |
|---|---|---|---|---|---|---|
| Word2Vec IPI PAN | 0,505 | 0,415 | 0,310 | 0,329 | 0,305 | -0,065 |
| Word2Vec PolEmo | 0,540 | 0,415 | 0,348 | 0,356 | 0,305 | -0,048 |
| fasttext (zespół) | 0,505 | 0,415 | 0,322 | 0,321 | 0,305 | -0,123 |
| use (zespół) | 0,520 | 0,415 | 0,330 | 0,334 | 0,305 | -0,013 |
| tfidf (zespół) | 0,490 | 0,415 | 0,318 | 0,335 | 0,305 | -0,008 |

Czułość (recall) dla każdej klasy (`final`):

| Model | POZ | NEG | NEU | AMB |
|---|---|---|---|---|
| Word2Vec IPI PAN | 0,44 | 0,07 | 0,89 | 0,00 |
| Word2Vec PolEmo | 0,54 | 0,11 | 0,89 | 0,00 |
| fasttext (zespół) | 0,33 | 0,15 | 0,92 | 0,00 |
| use (zespół) | 0,50 | 0,09 | 0,88 | 0,00 |
| tfidf (zespół) | 0,32 | 0,17 | 0,88 | 0,00 |
