# Pokrycie słownika zaanotowanego korpusu przez modele

| Model | Typy (słowa) | Tokeny | Słowa zaanotowane¹ |
|---|---|---|---|
| Word2Vec IPI PAN (nkjp+wiki, formy, skip-gram, 100 wym., 300 tys. słów) | 94,6% (1423/1505) | 97,1% | 94,9% (296/312) |
| Word2Vec wytrenowany na PolEmo 2.0 (gensim, skip-gram, 100 wym.) | 85,2% (1283/1505) | 92,0% | 83,7% (261/312) |

**Pokrycie według etykiety słowa²:**

| Model | O | WYR_POZ | WYR_NEG | WZM | OSL | ODW |
|---|---|---|---|---|---|---|
| w2v | 94,7% (n=1238) | 94,5% (n=145) | 90,7% (n=86) | 100,0% (n=19) | 100,0% (n=13) | 100,0% (n=4) |
| w2v_polemo | 86,2% (n=1238) | 86,2% (n=145) | 65,1% (n=86) | 94,7% (n=19) | 100,0% (n=13) | 100,0% (n=4) |

**w2v: najczęstsze słowa bez wektora:**

| Słowo | Wystąpienia (anotowane / PolEmo) | Etykieta słowa | Prawdopodobna przyczyna |
|---|---|---|---|
| 2osobowym | 1 / 1 | O | zawiera cyfrę |
| acellularne | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| antipaxos | 1 / 3 | O | inne (literówka, forma rzadka lub złożona) |
| autologiczne | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| barga | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| beznapięciowy | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| chłodziarkę | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| córcie | 1 / 4 | O | inne (literówka, forma rzadka lub złożona) |
| doczekalismy | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| doprosiła | 1 / 2 | WYR_POZ | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| dzieciusiami | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| gloeniu | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| groos | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| hallmann | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| ibisach | 1 / 3 | O | inne (literówka, forma rzadka lub złożona) |
| jednoręczną | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| kartkówce | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| klifami | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| lazience | 1 / 20 | O | brak polskich znaków (w modelu: łazience) |
| lazienek | 1 / 1 | O | brak polskich znaków (w modelu: łazienek) |

**w2v_polemo: najczęstsze słowa bez wektora:**

| Słowo | Wystąpienia (anotowane / PolEmo) | Etykieta słowa | Prawdopodobna przyczyna |
|---|---|---|---|
| jadalnią | 2 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| 1zł | 1 / 1 | WYR_POZ | zawiera cyfrę |
| 2mm | 1 / 2 | O | zawiera cyfrę |
| 2osobowym | 1 / 1 | O | zawiera cyfrę |
| acellularne | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| altanie | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| aspektach | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| autologiczne | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| badaczki | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| barga | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| beznapięciowy | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| beztroskim | 1 / 1 | WYR_NEG | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| białkowa | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| bieszczadzkie | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| budownictwo | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| bugs | 1 / 1 | WYR_NEG | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| bullet | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| błyszczące | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| choinkowe | 1 / 1 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |
| chłodziarki | 1 / 2 | O | rzadkie także w PolEmo (≤ 2 wystąpienia) |

¹ Słowa, którym co najmniej jedna anotacja nadała etykietę inną niż O.
² Etykieta słowa = najczęstsza etykieta ≠ O, jeśli co najmniej połowa anotacji słowa jest ≠ O; inaczej O. Dla fastText „pokrycie” oznacza obecność w słowniku – wektor z n-gramów znakowych dostaje każde słowo.
