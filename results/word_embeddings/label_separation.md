# Czy etykiety anotacji grupują się w przestrzeni osadzeń?

Liczone w oryginalnej przestrzeni (100 wymiarów, kosinus), nie na współrzędnych t-SNE.

| Model | Zestaw | n | Czystość k-NN | Losowo: średnia [95%] | p | Sylwetka |
|---|---|---|---|---|---|---|
| Word2Vec IPI PAN | WYR_POZ vs WYR_NEG (etykieta słowa) | 215 | 0,648 | 0,536 [0,510; 0,562] | 0,0010 | 0,033 |
| Word2Vec IPI PAN | 5 etykiet słów (bez O) | 251 | 0,531 | 0,401 [0,377; 0,425] | 0,0010 | -0,035 |
| Word2Vec IPI PAN | POZ vs NEG (kontekst zdań) | 622 | 0,545 | 0,502 [0,487; 0,519] | 0,0010 | 0,004 |
| Word2Vec PolEmo (gensim) | WYR_POZ vs WYR_NEG (etykieta słowa) | 181 | 0,671 | 0,571 [0,535; 0,605] | 0,0010 | 0,038 |
| Word2Vec PolEmo (gensim) | 5 etykiet słów (bez O) | 216 | 0,549 | 0,411 [0,381; 0,446] | 0,0010 | -0,015 |
| Word2Vec PolEmo (gensim) | POZ vs NEG (kontekst zdań) | 569 | 0,559 | 0,504 [0,489; 0,519] | 0,0010 | 0,006 |

**Sonda liniowa** (regresja logistyczna, 5-krotna walidacja krzyżowa ×5, zrównoważona trafność):

| Model | Zestaw | n | Trafność | Losowo: średnia [95%] | p |
|---|---|---|---|---|---|
| Word2Vec IPI PAN | WYR_POZ vs WYR_NEG (etykieta słowa) | 215 | 0,720 | 0,501 [0,469; 0,543] | 0,010 |
| Word2Vec IPI PAN | POZ vs NEG (kontekst zdań) | 622 | 0,616 | 0,503 [0,467; 0,541] | 0,010 |
| Word2Vec PolEmo (gensim) | WYR_POZ vs WYR_NEG (etykieta słowa) | 181 | 0,581 | 0,500 [0,490; 0,517] | 0,010 |
| Word2Vec PolEmo (gensim) | POZ vs NEG (kontekst zdań) | 569 | 0,619 | 0,497 [0,459; 0,542] | 0,010 |

**Czystość dla każdej etykiety osobno:**

| Model | Zestaw | Etykieta (n): czystość |
|---|---|---|
| Word2Vec IPI PAN | WYR_POZ vs WYR_NEG (etykieta słowa) | WYR_NEG (78): 0,474, WYR_POZ (137): 0,747 |
| Word2Vec IPI PAN | 5 etykiet słów (bez O) | ODW (4): 0,100, OSL (13): 0,200, WYR_NEG (78): 0,387, WYR_POZ (137): 0,684, WZM (19): 0,342 |
| Word2Vec IPI PAN | POZ vs NEG (kontekst zdań) | NEG (287): 0,521, POZ (335): 0,567 |
| Word2Vec PolEmo (gensim) | WYR_POZ vs WYR_NEG (etykieta słowa) | WYR_NEG (56): 0,370, WYR_POZ (125): 0,806 |
| Word2Vec PolEmo (gensim) | 5 etykiet słów (bez O) | ODW (4): 0,050, OSL (13): 0,146, WYR_NEG (56): 0,332, WYR_POZ (125): 0,765, WZM (18): 0,128 |
| Word2Vec PolEmo (gensim) | POZ vs NEG (kontekst zdań) | NEG (259): 0,579, POZ (310): 0,543 |

Czystość k-NN = średni odsetek k = 10 najbliższych sąsiadów (spośród słów zestawu) z tą samą etykietą. „Losowo” = ta sama miara po przetasowaniu etykiet (1000 razy): uwzględnia niezrównoważenie klas. p = odsetek przetasowań z czystością co najmniej taką jak obserwowana. Sylwetka od −1 do 1; wartości bliskie 0 oznaczają brak wyraźnych skupisk. Sonda liniowa sprawdza coś innego: czy etykietę da się odczytać z wektora prostym klasyfikatorem, nawet gdy słowa o tej samej etykiecie nie leżą blisko siebie.
