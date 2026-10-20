# Pkt 3: 10 najbliższych słów i porównanie modeli

Modele: **w2v** = Word2Vec IPI PAN (nkjp+wiki, formy, skip-gram, 100 wym., 300 tys. słów); **w2v_polemo** = Word2Vec wytrenowany na PolEmo 2.0 (gensim, skip-gram, 100 wym.).
Słowa zapytania: 215 słów z etykietą ≠ O znanych wszystkim modelom (WYR_POZ 125, WYR_NEG 55, WZM 18, OSL 13, ODW 4). Sąsiedzi: tylko słowa z liter, z całego słownika modelu. Wspólny słownik modeli: 20206 słów.

## 1. Zgodność list między modelami

**w2v vs w2v_polemo** (ρ drugiego rzędu dla całej macierzy podobieństw: 0,27)

| Etykieta | n | Wspólne@10 | Jaccard@10 | Wspólne@10 na wspólnym słowniku | ρ drugiego rzędu (średnia) |
|---|---|---|---|---|---|
| WYR_POZ | 125 | 8% | 0,04 | 11% | 0,32 |
| WYR_NEG | 55 | 9% | 0,05 | 12% | 0,26 |
| WZM | 18 | 16% | 0,10 | 18% | 0,24 |
| OSL | 13 | 8% | 0,05 | 12% | 0,23 |
| ODW | 4 | 8% | 0,04 | 10% | 0,41 |
| wszystkie | 215 | 9% | 0,05 | 12% | 0,29 |

## 2. Charakter list w każdym modelu

| Model | Odmiany słowa zapytania wśród sąsiadów | Sąsiedzi słów WYR_POZ: zgodny / przeciwny / neutralny wydźwięk (ocenionych) | Sąsiedzi słów WYR_NEG: zgodny / przeciwny / neutralny (ocenionych) |
|---|---|---|---|
| w2v | 13% | 47% / 41% / 11% (490) | 67% / 17% / 16% (256) |
| w2v_polemo | 2% | 59% / 32% / 9% (744) | 70% / 20% / 10% (312) |

Wydźwięk sąsiada: s = (zdania POZ − zdania NEG) / wszystkie zdania PolEmo ze słowem (min. 5 zdań; próg ±0,1). Model PolEmo i ta miara pochodzą z tego samego korpusu – jego przewaga jest częściowo „z założenia”.

## 3. Listy pokazowe (5 najbliższych)

**pogrubienie** = słowo jest na liście top-10 także w innym modelu; _kursywa_ = odmiana słowa zapytania; ⊗ = wydźwięk przeciwny do etykiety słowa zapytania.

| Słowo | Etykieta | w2v | w2v_polemo |
|---|---|---|---|
| miła | WYR_POZ | **sympatyczna** (0,85), **życzliwa** (0,82), serdeczna (0,79), przyjemna (0,79), **przemiła** (0,79) | pomocna (0,85), **przemiła** (0,81), uprzejma (0,81), **sympatyczna** (0,80), profesjonalna (0,79) |
| ok | WYR_POZ | **około** (0,96), tys (0,79), **ponad** ⊗ (0,79), blisko (0,74), tysięcy ⊗ (0,69) | **około** (0,63), **ponad** ⊗ (0,54), min ⊗ (0,51), piechotą (0,50), **prawie** ⊗ (0,49) |
| polecam | WYR_POZ | **odradzam** ⊗ (0,79), _polecasz_ (0,76), zalecam ⊗ (0,76), doradzam (0,72), radzę ⊗ (0,71) | **odradzam** ⊗ (0,77), **_polecamy_** (0,65), _polecić_ (0,62), **_polecę_** ⊗ (0,59), pragnie (0,58) |
| dobrze | WYR_POZ | słabo ⊗ (0,84), **świetnie** (0,79), **doskonale** (0,75), znakomicie (0,75), wspaniale (0,74) | fajnie (0,63), ładnie (0,59), **świetnie** (0,57), porządnie (0,54), estetycznie ⊗ (0,52) |
| niestety | WYR_NEG | wprawdzie ⊗ (0,79), niemniej (0,73), bo (0,72), chyba (0,71), zresztą (0,71) | **szczęście** (0,48), przyznać ⊗ (0,47), **owszem** (0,45), prawie (0,45), uwierzcie ⊗ (0,45) |
| krótko | WYR_NEG | potem (0,70), najpierw (0,70), później (0,68), zakończeniu (0,67), ponownie (0,67) | opóźniła (0,64), wczesna (0,64), momentem (0,64), przebiegla (0,64), najwidoczniej (0,64) |
| problem | WYR_NEG | dylemat (0,73), kłopot (0,71), nierozwiązany (0,68), aspekt (0,68), argument (0,68) | gorszy (0,53), błąd (0,53), kulinarnie (0,53), zdrowotny (0,52), pech (0,51) |
| problemy | WYR_NEG | **kłopoty** (0,89), trudności (0,78), komplikacje (0,77), perturbacje (0,76), dylematy (0,73) | oddychaniem (0,70), **poważne** (0,63), **kłopoty** (0,62), skórą (0,61), zniknęły ⊗ (0,60) |
| bardzo | WZM | **dość** (0,91), stosunkowo (0,85), **niezwykle** (0,84), **niezbyt** (0,83), zbyt (0,82) | **dość** (0,73), **niezbyt** (0,66), niesamowicie (0,66), **dosyć** (0,65), naprawdę (0,64) |
| bardziej | WZM | **mniej** (0,83), **znacznie** (0,79), **nieco** (0,76), coraz (0,74), **niż** (0,72) | **mniej** (0,52), niezbyt (0,51), **niż** (0,51), częstsze (0,50), trochę (0,50) |
| duży | WZM | **ogromny** (0,86), niewielki (0,86), **mały** (0,85), **olbrzymi** (0,84), największy (0,79) | spory (0,76), **mały** (0,66), ubogi (0,64), **olbrzymi** (0,64), wybór (0,63) |
| ani | WZM | **nie** (0,84), nigdy (0,79), **żadnego** (0,77), tylko (0,73), żadnych (0,72) | razu (0,54), wietrzenie (0,52), norm (0,50), **nie** (0,50), uśmiechnęła (0,49) |
| trochę | OSL | **troszkę** (0,91), troszeczkę (0,88), ciut (0,86), odrobinkę (0,84), odrobinę (0,82) | nieco (0,63), **troszkę** (0,55), zwyczajnie (0,53), ciągnie (0,53), wysłużone (0,53) |
| całkiem | OSL | zupełnie (0,79), **dosyć** (0,74), niecałkiem (0,73), trochę (0,72), dośc (0,72) | wizualnie (0,63), miarę (0,63), bardzo (0,63), chodź (0,62), aczkolwiek (0,61) |
| choćby | OSL | chociażby (0,85), oczywiście (0,64), zresztą (0,64), conajwyżej (0,62), ot (0,61) | dokupienia (0,63), jakiejś (0,62), gospodarza (0,61), utrzymywania (0,60), dostarczenie (0,60) |
| dosyć | OSL | **dość** (0,88), **niezbyt** (0,83), dośc (0,83), _dosć_ (0,82), nadzwyczaj (0,81) | **dość** (0,70), **niezbyt** (0,65), bardzo (0,65), **stosunkowo** (0,64), głośny (0,63) |
| nigdy | ODW | **nie** (0,85), ani (0,79), dotąd (0,71), jednak (0,69), już (0,69) | _nigdzie_ (0,64), skorzystam (0,63), wroce (0,63), odwiedze (0,62), udam (0,60) |
| niezbyt | ODW | dość (0,87), nadzwyczaj (0,86), wyjątkowo (0,86), nader (0,84), **dosyć** (0,83) | niesamowicie (0,66), **bardzo** (0,66), **dosyć** (0,65), _niezłe_ (0,63), swietne (0,63) |
| nikogo | ODW | kogokolwiek (0,75), kogoś (0,70), niczego (0,69), kogo (0,67), _nikt_ (0,65) | wroce (0,70), reagował (0,70), wolnego (0,69), puścić (0,68), pojadę (0,68) |
| zaden | ODW | kazdy (0,86), zwykly (0,82), doskonaly (0,81), przeciez (0,80), twoj (0,80) | moglem (0,83), udalo (0,83), wczesniejszych (0,82), zly (0,82), twierdzac (0,82) |
| brudne | WYR_NEG | **śmierdzące** (0,88), cuchnące (0,82), dziurawe (0,80), gołe (0,79), podarte (0,79) | wykładziny (0,81), kafelki (0,81), kurz (0,79), zniszczone (0,79), pleśń (0,78) |
| nie | O | nigdy (0,85), ani (0,84), jednak (0,82), ale (0,81), gdyż (0,79) | oj (0,67), wroce (0,66), odpisać (0,66), uważasz (0,65), niewiem (0,65) |

## 4. Największa i najmniejsza zgodność (w2v vs w2v_polemo, wspólne@10 na wspólnym słowniku)

- najbardziej zgodne: świetne (60%, ρ 0,43), dosyć (50%, ρ 0,50), przeszkadzało (50%, ρ 0,43), znacznie (50%, ρ 0,25), miał (50%, ρ 0,23)
- najmniej zgodne: pozornie (0%, ρ -0,14), potencjałem (0%, ρ -0,13), pośród (0%, ρ -0,06), zdrową (0%, ρ -0,03), paprochy (0%, ρ -0,01)

## 5. Przykłady sąsiadów o przeciwnym wydźwięku

- **w2v**: minusów → plusów (0,96; s = 0,29), mogli → musieli (0,91; s = -0,76), duże → niewielkie (0,90; s = 0,28), nauczyć → uczyć (0,89; s = -0,18), niskim → wysokim (0,89; s = 0,56), plusem → minusem (0,88; s = -0,72), zadowolona → niezadowolona (0,87; s = -0,92), pozytywne → negatywne (0,87; s = -0,61), chciała → wolała (0,87; s = -0,31), mogli → chcieli (0,87; s = -0,21)
- **w2v_polemo**: znała → roztrzęsiona (0,88; s = -1,00), ciekawie → monotonny (0,87; s = -0,50), skromne → obfite (0,86; s = 0,86), zrozumiale → studentow (0,85; s = -0,25), milej → obslugi (0,85; s = -0,50), znała → janiszewska (0,85; s = -1,00), poprawiło → pomoze (0,84; s = -0,33), chińskie → zadowalająca (0,84; s = 0,33), znała → załamana (0,84; s = -0,80), ciekawie → monotonnie (0,84; s = -0,67)
