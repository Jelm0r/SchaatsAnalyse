# Opname-advies: gratis nauwkeurigheid

De hoekmeting is direct begrensd door wat er op de pixels staat. Op afstand is een
onderbeen maar ~30–60 px lang; **2 px keypointfout = 2–4° hoekfout**. Betere opnames
leveren daarom meer op dan welke algoritme-verbetering dan ook. Richtlijnen, in
volgorde van effect:

1. **Film in 4K** (3840×2160) i.p.v. 1080p. Verdubbelt de lineaire pixelresolutie:
   het onderbeen van een verre schaatser wordt 60–120 px i.p.v. 30–60 px, en elke
   pixelfout telt half zo zwaar door in de hoek. De analyse wordt er trager van
   (grotere frames decoderen), maar de verfijningspass kijkt toch alleen naar de
   uitsnede rond de schaatser.
2. **Korte sluitertijd** (sport-/actiestand, of handmatig ≥ 1/500 s). Bewegings-
   onscherpte smeert de benen uit tot vage vegen waar élk pose-model op stukloopt —
   dit is de grootste bron van "skelet naast het been"-frames. Op een ijsbaan is
   meestal licht genoeg; een hogere ISO (wat ruis) is véél minder erg dan blur.
3. **Progressief opnemen, niet interlaced** — geldt alleen voor camcorders; telefoons
   en actiecams doen dit al goed. Zoek in het menu naar `PS`, `1080/50p` of "Opnamestand"
   en vermijd alles met een `i` erin (`1080/50i`). Een interlaced camera schiet 50 keer
   per seconde een *half* beeld (eerst de even beeldrijen, dan de oneven) en weeft die
   twee 1/50 s uit elkaar liggende momenten tot één frame. Op een bewegend been staan de
   twee helften daardoor op een andere plek: de kamtanden die je in beeld ziet.
   **Gemeten** op `00005.MTS` (AVCHD 1080i50) tegen de progressieve clips: de twee helften
   liggen op knieën en enkels **6,1 px** uit elkaar (p90 13,4; max 44), tegen 0,06 px op
   progressief materiaal — met "2 px = 2–4° hoekfout" is dat in zulke opnames de grootste
   ruisbron die er is. De app filtert het weg (`deinterlace` in `schaats_analyse.py`,
   automatisch herkend per video), maar dat is reparatie: de helft van de beeldrijen op het
   bewegende deel wordt dan geïnterpoleerd in plaats van opgenomen. Meteen goed opnemen
   levert echte pixels op én scheelt de halve temporele resolutie die het filter weggooit.

4. **50 of 60 fps** i.p.v. 24/30. Meer frames per slag = betere smoothing, strakkere
   afzetdetectie (de gewichtsoverdracht duurt maar enkele frames) en kleinere gaten
   bij een gemiste detectie. Bonus: 60 fps dwingt op de meeste telefoons al een
   kortere sluitertijd af.
5. **Vaste, horizontale camera, recht van voren** (statief of steun). Dit is al de
   aanname van de pijplijn (geen horizoncorrectie nodig); een schommelende camera
   voegt een foutbron toe die achteraf maar half te repareren is.
6. **Contrast helpt de tracking**: een pak dat afsteekt tegen het ijs én tegen de
   andere rijders maakt de pakkleur-poortwachter (doelkeuze) betrouwbaarder. Twee
   rijders in identieke pakken die elkaar kruisen blijft het moeilijkste geval.
7. **Zon/lampen achter de camera**, niet erachter de schaatser: tegenlicht maakt de
   schaatser een silhouet en drukt de keypoint-scores.

Validatie: neem één sessie op met de oude én nieuwe instellingen en vergelijk
`python schaats_eval.py vergelijk oud.npz nieuw.npz` (botlengte-stabiliteit, jitter)
— zie `schaats_eval.py` voor het meetprotocol.
