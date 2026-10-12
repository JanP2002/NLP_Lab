## Zgodność – iteracja 1 (anotatorzy: A1, A2)

| Poziom | n | Zgodność surowa | κ Cohena (A1–A2) [95% CI] | κ Fleissa [95% CI] | Interpretacja² |
|---|---|---|---|---|---|
| Zdania | 100 | 83,0% | 0,754 [0,636; 0,851] | 0,754 [0,635; 0,851] | znaczna |
| Tokeny – wszystkie | 1706 | 94,4% | 0,730 [0,655; 0,798] | 0,730 [0,655; 0,798] | znaczna |
| Tokeny – zaznaczone¹ | 231 | 58,9% | 0,463 [0,347; 0,573] | 0,462 [0,343; 0,571] | umiarkowana |

**Uzupełniająco – etykieta PolEmo (gold) jako dodatkowy oceniający (zdania):**

| Miara | Wartość [95% CI] | Interpretacja² |
|---|---|---|
| κ Fleissa (A1, A2, gold) | 0,636 [0,536; 0,731] | znaczna |
| κ Cohena (A1–gold) | 0,587 [0,468; 0,704] | umiarkowana |
| κ Cohena (A2–gold) | 0,587 [0,469; 0,700] | umiarkowana |

**κ Cohena dla każdej klasy osobno (klasa kontra reszta) – zdania:**

| Klasa | κ | liczba A1 | liczba A2 |
|---|---|---|---|
| POZ | 0,851 | 27 | 29 |
| NEG | 0,661 | 22 | 24 |
| NEU | 0,732 | 42 | 41 |
| AMB | 0,784 | 9 | 6 |

**κ Cohena dla każdej klasy osobno (klasa kontra reszta) – tokeny:**

| Klasa | κ | liczba A1 | liczba A2 |
|---|---|---|---|
| WYR_POZ | 0,828 | 88 | 77 |
| WYR_NEG | 0,604 | 54 | 53 |
| WZM | 0,695 | 29 | 31 |
| OSL | 0,264 | 9 | 6 |
| ODW | 0,685 | 17 | 15 |

¹ Tokeny, którym co najmniej jeden anotator nadał etykietę inną niż O. Na wszystkich tokenach dominuje zgodne O, które zawyża wynik.
² Skala Landisa i Kocha (1977): ≤0,20 nieznaczna, 0,21–0,40 dostateczna, 0,41–0,60 umiarkowana, 0,61–0,80 znaczna, 0,81–1 prawie pełna.
Przedziały ufności: bootstrap percentylowy, 2000 losowań zdań ze zwracaniem (seed 2026); przy tokenach losujemy całe zdania.
Przy dwóch anotatorach kappa Fleissa jest równa π Scotta: od kappy Cohena różni się tylko tym, że oczekiwaną zgodność liczy ze wspólnego rozkładu etykiet obu osób, stąd prawie identyczne wartości.
