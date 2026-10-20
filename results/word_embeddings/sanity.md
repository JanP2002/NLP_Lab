# Kontrola sensowności wektorów (nie zastępuje pkt 3)

## Word2Vec IPI PAN (nkjp+wiki, formy, skip-gram, 100 wym., 300 tys. słów)

| Słowo | 5 najbliższych (kosinus) |
|---|---|
| dobry | świetny (0.86), doskonały (0.83), kiepski (0.81), niezły (0.80), wspaniały (0.80) |
| polecam | odradzam (0.79), polecasz (0.76), zalecam (0.76), doradzam (0.72), radzę (0.71) |
| lekarz | chirurg (0.89), internista (0.88), neurolog (0.87), położnik (0.86), kardiolog (0.85) |
| brud | smród (0.77), kurz (0.76), zaduch (0.72), nieład (0.70), syf (0.70) |
| bardzo | dość (0.91), stosunkowo (0.85), niezwykle (0.84), niezbyt (0.83), zbyt (0.82) |
| nie | nigdy (0.85), ani (0.84), jednak (0.82), ale (0.81), gdyż (0.79) |

| Para | kosinus |
|---|---|
| dobry–zły | 0.733 |
| dobry–świetny | 0.860 |
| polecam–odradzam | 0.794 |
| czysto–brudno | 0.251 |
| bardzo–trochę | 0.427 |

## Word2Vec wytrenowany na PolEmo 2.0 (gensim, skip-gram, 100 wym.)

| Słowo | 5 najbliższych (kosinus) |
|---|---|
| dobry | słaby (0.64), świetny (0.63), genialny (0.58), dyski (0.57), pozytywny (0.57) |
| polecam | odradzam (0.77), polecamy (0.65), polecić (0.62), polecę (0.59), pragnie (0.58) |
| lekarz | specjalista (0.73), ginekolog (0.69), laryngolog (0.67), kompetentny (0.66), rzeczowy (0.65) |
| brud | bród (0.84), syf (0.83), wilgocią (0.81), pajeczyny (0.81), prlu (0.80) |
| bardzo | dość (0.73), niezbyt (0.66), niesamowicie (0.66), dosyć (0.65), naprawdę (0.64) |
| nie | oj (0.67), wroce (0.66), odpisać (0.66), uważasz (0.65), niewiem (0.65) |

| Para | kosinus |
|---|---|
| dobry–zły | 0.560 |
| dobry–świetny | 0.629 |
| polecam–odradzam | 0.766 |
| czysto–brudno | 0.583 |
| bardzo–trochę | 0.346 |
