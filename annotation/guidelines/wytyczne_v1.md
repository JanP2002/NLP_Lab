# Wytyczne anotacji – wersja 1 (iteracja 1)

Korpus: PolEmo 2.0, wersja zdaniowa (recenzje hoteli, lekarzy, uczelni i produktów).
Czas: ok. 70 min na 100 zdań. Można robić przerwy.

## 1. Zadanie

Każde zdanie oceniasz **samodzielnie, bez kontekstu reszty recenzji**. Anotacja ma dwa poziomy:

- **A. Etykieta całego zdania**: obowiązkowa, dokładnie jedna.
- **B. Etykiety słów (fragmentów)**: zaznaczasz tylko słowa pełniące jedną z pięciu funkcji opisanych w p. 3. Pozostałe słowa zostawiasz puste.

## 2. Etykieta zdania (kolumna `etykieta_zdania`)

| Etykieta | Kiedy |
|---|---|
| `POZ` | zdanie wyraża pozytywną ocenę lub opinię autora o przedmiocie recenzji albo jego części |
| `NEG` | zdanie wyraża negatywną ocenę lub opinię |
| `NEU` | brak oceny: fakt, opis, relacja wydarzeń, informacja ogólna |
| `AMB` | zdanie zawiera jednocześnie ocenę pozytywną i negatywną i żadna nie przeważa |

Reguły:
1. Oceniasz **opinię wyrażoną w zdaniu**, a nie to, czy opisany fakt jest obiektywnie dobry. Jeśli zdanie podaje fakt, który dla klienta jest wyraźnie dobry lub zły (np. długie czekanie, brak ciepłej wody), wybierz etykietę, którą przypisałby mu typowy czytelnik recenzji.
2. Jeśli w zdaniu są plusy i minusy, ale jeden biegun wyraźnie dominuje (np. konkluzja po „ale”), wybierz dominujący. `AMB` stosuj tylko wtedy, gdy oba bieguny są równoważne.
3. Ironię i sarkazm oceniasz według intencji autora, a nie dosłownego sensu.
4. W razie wątpliwości wybierz najlepiej pasującą etykietę i opisz wątpliwość w kolumnie `uwagi`.

## 3. Etykiety słów (wiersz `etykieta` pod słowami zdania)

| Etykieta | Funkcja | Typowe słowa |
|---|---|---|
| `WYR_POZ` | wyrażenie, które samo niesie pozytywną ocenę | polecam, miły, czysto, rewelacyjny, ok, niezła |
| `WYR_NEG` | wyrażenie, które samo niesie negatywną ocenę | okropne, arogancki, brud, problem, niemiły, najgorszy |
| `WZM` | wzmacnia siłę oceny | bardzo, niezwykle, absolutnie, zdecydowanie, jak najbardziej, w żadnym wypadku |
| `OSL` | osłabia siłę oceny | trochę, nieco, raczej, dość, całkiem (w znaczeniu „w miarę”) |
| `ODW` | odwraca polaryzację oceny | nie, brak, bez, ani, żaden, nigdy |

Reguły:
1. **Minimalny zakres.** Zaznaczasz tylko słowa pełniące daną funkcję. Nie zaznaczasz rzeczowników oznaczających przedmiot oceny (np. „pokój”, „lekarz”, „obsługa”) ani interpunkcji.
2. **Wyrażenie wielowyrazowe** (np. „do niczego”, „na medal”, „strata czasu”, „jak najbardziej”): każde słowo dostaje tę samą etykietę.
3. **Wydźwięk wyrażenia przed odwróceniem.** Wyrażenie oznaczasz według jego wydźwięku w danym kontekście, ale **bez uwzględnienia negacji**, która dostaje osobną etykietę `ODW`. Przykład: „nie polecam” daje *nie* = `ODW` i *polecam* = `WYR_POZ`, a całe zdanie dostaje `NEG`.
4. **Przeczenie wewnątrz słowa** („niemiły”, „niezła”, „nieprofesjonalny”) nie jest `ODW`. Całe słowo oznaczasz według jego znaczenia: *niemiły* = `WYR_NEG`, *niezła* = `WYR_POZ`.
5. **Modyfikatory (`WZM`, `OSL`, `ODW`) zaznaczasz tylko wtedy, gdy modyfikują ocenę.** W zdaniu „Taka funkcja nie wyklucza opcji …” słowo „nie” nie odwraca żadnej oceny, więc zostaje puste. Tak samo „bardzo” w „bardzo dużo ludzi”, jeśli liczba ludzi nie jest oceniana.
6. Etykiety słów i etykieta zdania są niezależne. Zdanie `POZ` lub `NEG` może nie mieć żadnego zaznaczonego słowa (ocena wynika z faktów), a zdanie `NEU` może mieć zaznaczone słowa.

## 4. Tokenizacja

Teksty są już podzielone na tokeny i tego podziału nie zmieniamy:
- „Była m” = „Byłam”, „mieli śmy” = „mieliśmy”. Jeśli takie słowo zaznaczasz, obie części dostają tę samą etykietę.
- Interpunkcja jest osobnym tokenem i nigdy jej nie zaznaczamy.
- Literówki i brak polskich znaków („przyjezdza”) ignorujesz.

## 5. Przykłady (zdania spoza próbek do anotacji)

| Zdanie | Etykieta zdania | Etykiety słów |
|---|---|---|
| Pani doktor była bardzo serdeczna . | `POZ` | bardzo = `WZM`, serdeczna = `WYR_POZ` |
| Jak najbardziej polecam ten hotel . | `POZ` | Jak, najbardziej = `WZM`; polecam = `WYR_POZ` |
| Z odnalezieniem rezerwacji nie było problemu . | `POZ` | nie = `ODW`, problemu = `WYR_NEG` |
| W żadnym wypadku nie polecam . | `NEG` | W, żadnym, wypadku = `WZM`; nie = `ODW`; polecam = `WYR_POZ` |
| Najgorszy lekarz u jakiego dotychczas była m . | `NEG` | Najgorszy = `WYR_NEG` |
| Obsługa klienta raczej słaba - przedmiotowa . | `NEG` | raczej = `OSL`, słaba = `WYR_NEG` |
| Przychody w turystyce do 2020 roku mają wzrosnąć o 5 mld zł . | `NEU` | – |
| Taka funkcja nie wyklucza opcji indywidualnej blokady poszczególnych kanałów według naszych potrzeb . | `NEU` | – (reguła 3.5) |
| Kuchnia całkiem niezła , każdy znajdzie coś dla siebie . | `POZ` | całkiem = `OSL`, niezła = `WYR_POZ` |
| Ogólnie hotel wewnątrz nie powala , ale jest bardzo dobrze zlokalizowany . | `AMB` | nie = `ODW`, powala = `WYR_POZ`, bardzo = `WZM`, dobrze = `WYR_POZ` |
| Jeden kelner i kelnerka ok , ale jeden ( młodszy ) niemiły . | `AMB` | ok = `WYR_POZ`, niemiły = `WYR_NEG` |

## 6. Praca w arkuszu

- Każde zdanie zajmuje dwa wiersze. W wierszu `tekst` są słowa zdania, a pod nimi, w wierszu `etykieta`, wybierasz etykiety z listy rozwijanej. Puste pole oznacza brak etykiety.
- Etykietę zdania wybierasz w kolumnie `etykieta_zdania` (wiersz `tekst`).
- Wątpliwości opisujesz w kolumnie `uwagi`.
- **Nie zmieniasz** słów, kolejności wierszy ani kolumn i nie dodajesz wierszy.
- Po skończeniu: *Plik → Pobierz → Microsoft Excel (.xlsx)* i wyślij plik koordynatorowi.

## 7. Zasady niezależności

- Anotujesz **sam**. Do zakończenia iteracji nie rozmawiasz o zdaniach z innymi anotatorami.
- Nie sprawdzasz etykiet PolEmo i nie używasz LLM-ów, tłumaczy ani słowników wydźwięku.
- Nie udostępniasz nikomu swojego arkusza i nie otwierasz cudzych.
