# Bugrapport — code-review juli 2026

Review van `schaats_analyse.py`, `schaats_yolo.py`, `schaats_gui.py`, `schaats_db.py`,
`schaats_eval.py`, `schaats_perspectief.py`, met **nadruk op het volgen van de
schaatser**. Elke bevinding met **[bewezen]** is met een reproductiescript of op de
echte analyses in `Documenten\SchaatsAnalyse\media\*\landmarks.npz` aangetoond;
**[latent]** = het codepad bestaat en is bereikbaar, maar kwam in de huidige
bibliotheek niet voor.

Zelftests die wél in orde zijn: `python schaats_db.py` → OK, `python
schaats_perspectief.py` → PASS, alle zes modules compileren.

---

## Hercontrole 26 jul 2026 — alle 24 bevindingen nagemeten

Elke fix is onafhankelijk getoetst: de oude code is uit git gehaald
(`git show HEAD:schaats_analyse.py`) en naast de nieuwe gedraaid op dezelfde data, zodat
"opgelost" niet op de fix-tekst maar op een meting rust.

**Alle 24 bevindingen zijn weg.** Bewijs per blok:

| | bewijs |
|---|---|
| **A1** | `ev.hoek` − ruwe hoek op het eindframe = **+0,0° in alle 100 events van 18 clips** (was +0,2…+10,0). |
| **A2** | "clips waar de laatste, doorlopende afzet níet als afgekapt gemarkeerd is: **GEEN**". Gemiddelde excl. afgekapt: `521f6916` 51,8→38,2°, `4b066ea3` 58,2→51,4°, `542bdc04` 56,3→42,1°. |
| **A3** | `b9896526` 13→8 events, `542bdc04` 9→8, `6c3eacd5`/`bbf6fc5c` 6→5. |
| **B1** | Reproductie: coast op frames 6–10, pakt het doel op frame 11 weer op, **0× de omstander** (was: permanent vast op de omstander). |
| **B2** | 3 én 4 gemiste frames herstellen nu direct (was: vanaf 3 permanent kwijt). |
| **B3** | Grote stilstaande omstander (x=0,75) vs. kleinere rijdende schaatser (x=0,10) → kiest **0,10**; de oude "grootste pose" koos 0,75. |
| **B4** | Decisief: oud draait de onbetrouwbare frames 7 en 13 tégen hun buren in (`RRRRRRRLRRRRRLRRRRRR`), nieuw houdt de reeks consistent (`RRRR…`). |
| **B5** | Op echte landmarks: botlengte-CV **4× beter, 0× slechter**; `db2ad2a7` gekruiste frames **330 → 171**, CV tibia 0,136→0,106 en 0,145→0,120. |
| **B6** | Stilstaande heupen geven nu geen valse `heup_dx = 60` meer; tiebreaker antwoordt niet langer altijd `'links'`. |
| **C1–C7** | Synthetisch op de helpers: geen-seed → `RuntimeError`; klik-gemist gemeld; korte seed 1→16 detecties aangedikt; kleur aan de ketenkant gemeten (weigert én accepteert het spiegelgeval correct); bbox-terugval knipt niet meer, écht ander pak knipt nog wél; 2-frame-overlap wordt gestitcht. Alle call-sites van de gewijzigde signatures nagelopen. |
| **D1** | `open_db` op een v3-database gooit nu `BibliotheekTeNieuw` i.p.v. hem naar v2 te verlagen. |
| **D2–D6** | In de diff geverifieerd: coöperatief afbreken + `wait()` vóór sluiten, `gestopt`-vlag in `annoteer`, `_meld_leesfout`, alleen `schaats*.db` als conflictkopie. |
| **E** | Alle 9 punten in de diff terug te vinden. |
| **regressie** | Gouden referentie op de júiste clips (`Schaats frontaal.MOV`): hoekfout en puntfout **exact gelijk** oud vs. nieuw — geen regressie op landmark-nauwkeurigheid. Zelftests `schaats_db.py` → OK, `schaats_perspectief.py` → PASS, alle zes modules compileren. |

### Wat er ná de fixes nog staat (nieuw gevonden)

Stand 26 jul 2026: **R1 en R3 zijn opgelost** (zie de fix-blokken hieronder). R2 en R4
blijven bewust staan: R2 is op echte data per saldo een verbetering (zie B5) en R4 is een
afweging, geen fout.

**R1. Een korte stand-run kan nog steeds het "rechtop komen" als afzet rapporteren.**
**[bewezen]** — ✅ OPGELOST (26 jul 2026)

> **Fix:** de afkap-vlag is gegeneraliseerd naar **één vlag mét reden**:
> `FrameResultaat.afzet_onvolledig` → `AfzetEvent.onvolledig`, met `ONV_AFGEKAPT`
> ("afgekapt") en de nieuwe `ONV_GEEN_PUSH` ("geen volledige push"). Elke plek die de
> statistiek filtert hoeft alleen op waarheid te toetsen, terwijl de GUI-tooltip de
> gebruiker de juiste vraag stelt — "de video hield op" vraagt om een langere opname,
> "geen volledige afzet waargenomen" om een blik op de been-toewijzing. `AFGEKAPT_MARKER`
> blijft letterlijk `"afgekapt"`, dus bestaande events-caches werken ongewijzigd door;
> `lijst_analyses` zet nu één `NOT LIKE` per reden (`ONVOLLEDIG_MARKERS`).
>
> Het criterium is **meetkunde, geen tuning-getal**: staat het onderbeen bij de gekozen
> voltooiing minder dan `STREK_MIN_HELLING_DEG` = 20° uit het lood (dus afzethoek > 70°),
> dan is de zijwaartse component van de afzet sin(20°) ≈ 0,34 — er is simpelweg niet opzij
> geduwd, en het plateau besloeg alleen de opricht-fase.
>
> **Gemeten over de 18 opgeslagen analyses (100 events):** de drie genoemde events krijgen
> `ONV_GEEN_PUSH` en vallen uit gem/min/max; alle **84** eerder-gezonde events blijven
> meetellen, de 13 al-afgekapte houden hun eigen reden. Aantal events, L/R-volgorde en
> alle hoeken per clip zijn onveranderd — het is puur een vlag. `c9fcd0be` gem
> 60,6 → **41,8°**, `db2ad2a7` 45,0 → **41,7°**, de overige 16 clips exact gelijk. De
> gouden referentie op `1e9ae474`/`2651bcdb`/`542bdc04`/`cb13bb1e` geeft dezelfde hoek- en
> puntfout als vóór de wijziging (landmarks onaangeroerd).
>
> **Twee dingen om te weten.** (1) De in dit rapport voorgestelde maat — de afstand tussen
> de gekozen hoek en het rechtop-komen binnen dezelfde run — bleek op de echte data níet te
> scheiden: gezonde events zitten daar op 0–56° en de drie verdachte op 3–22°, dus de
> groepen overlappen volledig (nagemeten voor `max_hoek` van het event, het maximum over de
> hele run, en het maximum tot de voltooiing). Vandaar de meetkundige bovengrens.
> (2) De dichtstbijzijnde buur onder de grens is `db2ad2a7` ev9 (66,7°) — zelf ook een
> twijfelgeval: een run van 77 frames (1,5 s) waarin de been-toewijzing nooit omsloeg, dus
> drie halve slagen in één "run". Die 3,3° marge is de krapste plek van deze drempel; wordt
> de been-toewijzing beter, dan wordt de marge ruimer.

`min_run` en de afkap-vlag vingen de meeste spookafzetten, maar een run die net lang
genoeg is en waarvan het strek-plateau alleen de opricht-fase beslaat, leverde een
gewone, meetellende afzet met een hoek van 74–83°. Nagemeten: **3 van de 100 events**
in de bibliotheek — `c9fcd0be` ev1 (8 fr / 0,27 s, 79,4°), `db2ad2a7` ev19 (23 fr, 83,1°)
en ev21 (46 fr, 74,3°). Effect: `c9fcd0be` gem 41,8 → **60,6°**, `db2ad2a7` 41,7 → 45,0°.
Herkenbaar patroon: korte duur + hoek boven ~70°. Een bovengrens op de afzethoek, of
eisen dat het plateau een dálende hoekflank bevat, zou dit sluiten.

**R2. De gezamenlijke L/R-beslissing kan een fout in één gewrichtspaar niet meer
repareren.** Verwisselt de detector alléén de knielabels (of alléén de enkels), dan is
`kost_sw ≈ kost_id` en wisselt de nieuwe code niets — het gekruiste skelet blijft staan.
De oude code repareerde dat geval wél. Aangetoond synthetisch (5 frames gekruist, beide
richtingen). Op de echte data is de nieuwe versie per saldo duidelijk beter (zie B5
hierboven), dus dit is geen reden om terug te draaien — wel om te weten dat het gat er is.

**R3. `schaats_eval._botlengte_cv` kan een betekenisloos getal opleveren.**
**[bewezen]** — ✅ OPGELOST (26 jul 2026)

> **Fix:** de klem `np.maximum(trend, 1e-6)` is weg. Frames waar de trend onder
> `TREND_MIN_FRAC` (0,25) × de mediane botlengte zakt hebben geen bruikbare noemer en
> worden **overgeslagen**; `_botlengte_cv` retourneert nu `(cv, n_gebruikt, n_overgeslagen)`
> en `print_metrics` zet dat achter de waarde: `CV tibia_l: 0.098 (n=170, 3 overgeslagen)`.
> Blijven er minder dan 10 bruikbare frames over, dan komt er géén getal maar
> "onbetrouwbaar/te weinig metingen". Nagemeten over alle 18 npz's: elke CV ligt nu tussen
> **0,023 en 0,238** (of expliciet geen uitspraak) — precies het geval uit dit rapport,
> `4b066ea3` `tibia_l`, gaat van **981184,372 → 0,098** met 3 overgeslagen frames.

Op `4b066ea3` stond er `CV tibia_l: 981184.372` (n=173). Oorzaak: de Savitzky–Golay-trend
wordt in één frame **negatief** (−0,87 px), waarna `np.maximum(trend, 1e-6)` de ratio naar
1,3·10⁷ laat exploderen. De onderliggende data is ook wankel (tibia 10,4 px vs. mediaan
54,9 — zwaaibeen achter het standbeen), maar de metric hoort dat te *melden* in plaats van
een getal van zeven cijfers te printen. Dit is de tool waarmee toekomstige wijzigingen
getoetst worden, dus het is de moeite waard: trendwaarden onder een fractie van de
mediaan overslaan en het aantal overgeslagen frames rapporteren.

**R4. Bewuste afweging in de nieuwe herseed-poort.** `TRACK_HERSEED_GATE` (0,25) weigert
een herseed buiten de geëxtrapoleerde laatst bekende plek, en `_verwacht` klemt de
extrapolatie op `hervind_frames`. Raakt de schaatser lang genoeg kwijt en duikt hij ver
weg weer op, dan wordt hij niet meer opgepakt — de tracker geeft dan liever niets dan het
verkeerde skelet. Dat is de juiste keuze, maar het betekent dat een lange occlusie nu een
blijvend gat in de dekking geeft in plaats van een (mogelijk foute) lock.

---

## A. Meetfouten — deze raken direct de cijfers in de tabel

### A1. De afzethoek is een naijlend gemiddelde, niet de hoek op het gekozen frame **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:1565` en `:1550`

> **Fix**: `segmenteer_afzetten` vult `hoeken` nu met `r.hoek` i.p.v. `r.smooth_hoek`, dus
> `hoek`/`min_hoek`/`max_hoek` komen alle drie van dezelfde per-frame-reeks en `hoek` is de
> hoek van het voltooiingsframe zelf. Daarnaast is `smooth_hoek` van een trailing deque naar
> een **gecentreerd** (zero-lag) gemiddelde gegaan (`_zet_smooth_hoek`), per aaneengesloten
> reeks van hetzelfde standbeen — dat maakt de HUD-waarde eerlijk en neemt D4 (hoek-buffer
> mengt twee benen) meteen mee. De GUI-grafiek plot nu `r.hoek` (zelfde grootheid als de
> tabel, zie de E-tabel). Nagemeten over de bibliotheek: afwijking `ev.hoek` − ruwe hoek op
> het eindframe is voor **alle 100 events exact 0,0°** (was gem +3,3°, max +12,1°); de
> segmentatie zelf verandert niet. De gerapporteerde hoeken zakken dus gem 3,3°.

`bepaal_afzet_uit_strek` kiest binnen het strek-plateau bewust het frame met de
**vlakste onderbeenhoek** ([schaats_analyse.py:1120-1127](schaats_analyse.py#L1120-L1127)).
Maar `segmenteer_afzetten` rapporteert niet de hoek van dát frame: het vult
`huidig['hoeken']` met `r.smooth_hoek` en neemt `hoek=hoeken[-1]`. En `smooth_hoek`
is een **achterwaarts** gemiddelde over de laatste `smooth_n` (default 5) frames
([schaats_analyse.py:1255](schaats_analyse.py#L1255), `:1281-1282`). Omdat de hoek naar
dat minimum toe daalt, ligt het gemiddelde er structureel bóven.

Gemeten op alle 18 opgeslagen analyses — `ev.hoek` min de ruwe hoek op het eindframe:

| analyse | afwijking per event |
|---|---|
| `1e9ae474` | +2.3 +5.3 +3.6 +4.3 +1.6 +2.7 |
| `4fdb2b0e` | +5.8 +6.0 +7.7 |
| `521f6916` | +7.7 +6.1 +7.0 |
| `542bdc04` | +5.3 +5.0 **+10.0** +3.5 +8.0 +2.3 +5.2 +3.9 +6.1 |
| `b9896526` | +1.7 +1.3 +1.5 +0.6 +1.6 +1.5 +1.2 +7.5 |

**Altijd positief (te steil), in 100% van de events**, tot +10°. Voor een tool die
afzethoeken op 0,1° rapporteert is dit de grootste foutbron in het hele programma.

Fix: laat het gekozen voltooiingsframe zijn eigen (eventueel gesmoothde) hoek
dragen, bv. `hoek = r.hoek` van het eindframe, of bereken `smooth_hoek` **gecentreerd**
(zero-lag, zoals de landmark-smoothing zelf al doet) i.p.v. met een trailing deque.

### A2. Afgekapte laatste stand-run levert een volwaardige afzet op **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:1076-1089` + `:1128-1129`

> **Fix:** een run die niet op een beenwissel eindigt maar op het einde
> van de video/het pose-segment zet `FrameResultaat.afzet_afgekapt` → `AfzetEvent.afgekapt`.
> Zo'n event blijft zichtbaar (grijze rij + tooltip, `afgekapt`-kolom in de CSV) maar valt
> buiten gem/min/max in de GUI en buiten `AVG(hoek)` in de bibliotheeklijst. Runs die aan
> het *begin* afgekapt zijn tellen wél mee (daar mist alleen de load-fase). Effect op de
> opgeslagen analyses, bv. `4b066ea3`: gem 55,1 → 51,4°, max 73,7 → 55,0°.

`bepaal_afzet_uit_strek` behandelt elke stand-run gelijk, ook de run die door het
**einde van de video** (of van een pose-segment) wordt afgekapt. Daar is de push nog
niet af, dus het strek-plateau bevat alleen de eerste helft en de "vlakste hoek" is
in werkelijkheid het rechtop-komen.

In **18 van de 18** clips raakt de laatste run het laatste pose-frame. Het effect op
de laatste afzet:

| analyse | overige afzetten | laatste afzet |
|---|---|---|
| `4b066ea3` | 49.8 57.3 52.2 55.5 57.6 | **79.5** |
| `7e7d0a66` | 49.7 57.2 52.2 55.4 57.5 | **79.4** |
| `b9896526` | 38.9 36.3 39.2 40.6 39.7 40.5 42.7 | **55.3** |
| `521f6916` | 40.8 49.3 | **58.7** |
| `4fdb2b0e` | 46.7 51.0 | **63.0** |

Die waarde gaat mee in `gem`/`max` in de statusregel ([schaats_gui.py:2192](schaats_gui.py#L2192))
én in `AVG(hoek)` van de bibliotheek-lijst ([schaats_db.py:236](schaats_db.py#L236)).

Fix: markeer een run waarvan het laatste frame samenvalt met het einde van het
pose-segment als onvolledig — géén event, of een event met
`opmerking="afgekapt"` dat buiten de statistiek valt. Idem voor de run die op
frame 0 begint (minder schadelijk, want de piek zit aan het eind van een push).

### A3. De slagtijd-prior ondermijnt zichzelf **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:1091-1093`

> **Fix:** de mediaan loopt alleen nog over runs van minstens
> `STREK_MIN_RUN_S` (0,2 s); zijn die er niet, dan valt hij terug op alle runs. Gemeten op
> de opgeslagen analyses: `1edc50ac` `min_run` 5 → 11 fr (halve slag 14,5 → 32),
> `8725ef71` 4 → 10 fr (12 → 27,5), `542bdc04` 3 → 5 fr (9 → 11,5), `b9896526` 12 → 16 fr.
> In `542bdc04` verdwijnen de vier kortste spook-runs (4–5 fr) daarmee.

```python
lengtes = [len(run['idx']) for run in runs]
half_periode = float(np.median(lengtes)) if lengtes else 0.0
min_run = max(2, int(round(STREK_MIN_SLAG_FRAC * half_periode))) if half_periode else 2
```

De mediaan wordt over **álle** runs genomen, inclusief de ruis-runs die de prior
juist moet wegfilteren. Zijn er veel L/R-flips, dan zakt de mediaan en zakt de
drempel mee — precies wanneer je hem nodig hebt:

| analyse | run-lengtes | mediaan → `min_run` | echte halve slag |
|---|---|---|---|
| `1edc50ac` | `[1,1,1,3,29,34,36,32,26,2]` | 14.5 → 5 fr | **32 fr** |
| `8725ef71` | `[24,2,12,1,31,32,3]` | 12.0 → 4 fr | **27.5 fr** |
| `542bdc04` | `[5,5,4,5,9,23,19,16,14]` | 9.0 → 3 fr | 11.5 fr |
| `db2ad2a7` | `[1,1,12,1,20,35,34,…]` | 24.5 → 9 fr | 29.5 fr |

Gevolg in `542bdc04`: negen "afzetten", waarvan vijf van 4–9 frames (0,13–0,34 s) met
hoeken tot 86,4° — dat is geen afzet maar het rechtop-komen tijdens een ruis-omslag.

Fix: schat `half_periode` robuust, bv. mediaan van alleen de runs ≥ een absolute
ondergrens (0,2 s), of de mediaan van de bovenste helft van de run-lengtes; eventueel
iteratief (drempel toepassen → opnieuw schatten).

---

## B. Volgen van de schaatser — MediaPipe-backend (`DoelTracker`)

### B1. Herseeden gebruikt de verouderde klik van frame 0; de "laatst bekende plek"-tak is dode code **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:588-607` + `:609-647`

> **Fix:** `DoelTracker.laatste_bekend` staat nu naast `self.centroid` (lock-vlag) en wordt
> bij verlies níet gewist, dus de herseed-tak werkt. Die tak kiest de kandidaat het dichtst
> bij de **mee-geëxtrapoleerde** laatst bekende plek (`_verwacht()`) en weigert een herseed
> buiten `TRACK_HERSEED_GATE` (0.25) — dan liever géén pose dan een skelet op een omstander.
> Het klikpunt geldt alleen nog bij de koude start (`laatste_bekend is None`).
> Verificatie: reproductie hierboven kiest nu 0× de omstander en pakt het doel op frame 11
> weer op.

`_seed` heeft drie takken, met als tweede:

```python
if self.centroid is not None:
    # Net kwijt geweest: pak de schaatser het dichtst bij de laatst bekende plek.
```

Die tak wordt **nooit** bereikt. Herseeden gebeurt bij
`self.centroid is None or self.kwijt > self.hervind_frames` (`:620`), maar op precies
het moment dat `kwijt` de drempel passeert wordt `self.centroid = None` gezet — in
beide verliestakken (`:616` en `:637-638`). Bij het herseeden is `centroid` dus altijd
`None`, en valt hij terug op ofwel het **klikpunt van frame 0**, ofwel "grootste pose".

Reproductie (`scratchpad/test_reseed.py`): doel start op x=0.20 (daar klikt de
gebruiker) en rijdt naar rechts; een omstander staat stil op x=0.90; frames 6–10 is
het doel niet detecteerbaar.

```
 frame | doel op | gekozen | wat is dat?
    5  |  0.35   |  0.35   | DOEL
    6  |  0.38   |  -      | kwijt (coast)
   ...
   10  |  0.50   |  0.90   | << OMSTANDER
   15  |  0.65   |  0.90   | << OMSTANDER
```

Vanaf frame 11 is het doel weer zichtbaar op 0.53 — veel dichter bij de laatst
bekende plek (0.35) dan de omstander (0.90) — maar de tracker zit al vast op de
omstander en herstelt nooit meer. In de GUI zie je dan een skelet op de verkeerde
persoon met plausibele hoeken.

Fix: bewaar de laatste positie apart van de "heb ik lock"-vlag (bv. `self.laatste_bekend`
naast `self.centroid`), zodat de tweede tak werkt; en weiger een herseed die verder
dan een ruime poort van de laatst bekende plek ligt.

### B2. Tijdens coast wordt de voorspelling niet mee-geëxtrapoleerd **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:627-639`

> **Fix:** de voorspelling schuift mee (`_verwacht(centroid, kwijt+1)`, horizon begrensd op
> `hervind_frames` + geklemd op het beeld) en de poort groeit mee:
> `min(gate × (1 + TRACK_GATE_GROEI × kwijt), TRACK_GATE_MAX)` (plafond gelijk aan
> `TRACK_HERSEED_GATE`, zodat de acceptatie bij de overgang coast→herseed niet verspringt)
> — de tegenhanger van de
> YOLO-`STITCH_GATE_*`. Bij een match wordt de snelheid gedeeld door de gat-lengte, anders
> zou één gat de schatting N× opblazen. Verificatie: gaten van 1 t/m 6 frames herstellen nu
> allemaal (`++++......++++++`), en de snelheidsschatting blijft na een gat van 4 frames
> 0.0298 vs. echt 0.03.

De voorspelling is altijd `centroid + 1 × snelheid` (`:628-629`), ook na N gemiste
frames — terwijl de echte sprong dan `(N+1) × snelheid` is. `self.centroid` blijft op
de laatste match staan. De poort `TRACK_GATE` groeit ook niet mee.

Reproductie (`scratchpad/test_coast.py`, snelheid 0.045/frame = ⅓ van de poort):

```
1 frame gemist  -> ++++.++++++    herstel: JA
2 frames gemist -> ++++..++++++   herstel: JA
3 frames gemist -> ++++...XXXXXX  herstel: NEE
4 frames gemist -> ++++....XXXXXX herstel: NEE
```
(`X` = de schaatser wordt wél gedetecteerd, maar valt buiten de poort.)

Bij ≥3 gemiste frames is de schaatser **permanent** kwijt: hij blijft `hervind_frames`
(bij 30 fps = 15 frames) coasten en herseedt daarna via B1 op het verouderde
klikpunt. Drie gemiste frames op een rij is bij bewegingsonscherpte niets bijzonders.

Fix: schuif de voorspelling per coast-frame mee (`centroid += snelheid` bij verlies),
of schaal de poort met het aantal gemiste frames — precies zoals de YOLO-backend het
al doet met `STITCH_GATE_BASIS + STITCH_GATE_GROEI × gat`
([schaats_yolo.py:389](schaats_yolo.py#L389)).

### B3. `DoelTracker` heeft geen "de schaatser beweegt"-criterium **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:601-607`

> **Fix:** bij een koude start **zonder klik** kijkt `analyseer_frames` eerst
> `SEED_WARMUP_S` (1 s) mee. Die gebufferde frames gaan door `_volg_kandidaten` (ruwe
> nearest-neighbour-sporen) en `_kies_bewegend_doel` wijst de grootste *beweger* aan:
> mediane bbox-oppervlakte × afgelegde weg, met `SEED_MIN_VERPLAATSING` als ondergrens —
> dezelfde regel als de YOLO-backend. Dat punt gaat als `doel_punt` de tracker in, waarna
> de gebufferde frames alsnog worden afgespeeld (geen frame gaat verloren); frames vóór
> het startframe van het gekozen spoor blijven leeg, want daar is het doel nog niet in
> beeld. Mét muisklik verandert er niets. Verificatie (`test_stap6_analyse.py`, TEST 1):
> grote stilstaande omstander op x=0.90 vs. rijdende schaatser op x=0.20 — de oude regel
> koos de omstander, de nieuwe volgt de schaatser het hele venster.

Bij een koude start zonder klik kiest hij `max(..., key=_bbox_oppervlak)`: de
**grootste** pose. Een omstander langs de boarding is in beeld geregeld groter dan de
schaatser die verder weg rijdt. De YOLO-backend lost dit expliciet op met
`MIN_VERPLAATSING` en "mediane area × padlengte"
([schaats_yolo.py:320-325](schaats_yolo.py#L320-L325)); de MediaPipe-tracker mist dat.
Ook per frame telt alleen nabijheid, dus een stilstaande omstander die dichter bij de
voorspelling ligt wint van het bewegende doel.

Fix: bij de koude start over de eerste ~1 s de verplaatsing per kandidaat meewegen.

### B4. De L/R-meerderheidsstem draait óók niet-besliste frames om **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:836-844`

> **Fix:** `_fix_lr_swaps` houdt nu een `besloten`-masker bij; de meerderheid wordt over
> alleen die frames genomen en de inversie raakt alleen die frames. Frames die wegens
> lage zichtbaarheid zijn overgeslagen blijven ongemoeid.

```python
if gewisseld.mean() > 0.5:           # ketting verkeerd verankerd: labels omdraaien
    gewisseld = ~gewisseld
```

`gewisseld[t]` blijft `False` voor frames die bewust **niet beslist** zijn omdat de
zichtbaarheid te laag is (`:825-826` doet `continue`). Na de inversie worden juist die
frames als "wissel" gemarkeerd en worden L/R daar omgedraaid — zonder enige
onderbouwing, en tegen de rest van de reeks in.

Reproductie (`scratchpad/test_tracking.py`, TEST 3): twee frames met
`visibility = 0.05` krijgen na de fix hun knieën verwisseld terwijl alle omringende
frames correct staan. Dat zet in die frames een links/rechts-fout die de
botlengte-check en het l−r-enkelsignaal van de afzetcyclus vervuilt.

Fix: houd een aparte `besloten`-mask bij en inverteer alleen `gewisseld & besloten`,
of neem de meerderheid alleen over de besliste frames en laat de rest ongemoeid.

### B5. Kniën en enkels worden onafhankelijk van elkaar omgewisseld **[latent]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:818-819`

> **Fix:** `_fix_lr_swaps` neemt per frame nog één beslissing voor het hele been — knie én
> enkel, met hiel en teen mee — op de **opgetelde** continuïteitskosten van beide
> gewrichtsparen. Verificatie (`test_stap6_analyse.py`, TEST 2): bij anatomisch correcte
> invoer met flikkerende L/R-labels (benen vrijwel over elkaar) leverde de oude fixer in
> 3 van 8 runs 6–21 van de 30 frames met de knie van been A aan de enkel van been B; de
> nieuwe nul in alle acht. TEST 3 bewaakt dat een aanhoudende verwisseling nog steeds
> gewoon hersteld wordt. De kruisings-ambiguïteit hieronder blijft staan: dat is een
> andere aanname (continuïteit), geen ontbrekende koppeling.

`paren = ((L_KNEE, R_KNEE, ()), (L_ANKLE, R_ANKLE, (hiel, teen)))` — twee losse
beslissingen. Vallen ze verschillend uit, dan hangt de linkerknie aan de rechterenkel:
een anatomisch onmogelijk skelet met een onzinnige tibialengte. In mijn synthetische
test viel het samen goed uit, dus niet aangetoond — maar de koppeling ontbreekt.

Ook opvallend: de continuïteitskost kiest bij écht kruisende benen de
**niet-kruisende** interpretatie (zichtbaar in `scratchpad/test_lr_koppel.py`, waar de
fix de echte kruising terugdraait). Bij frontaal filmen op het rechte stuk is dat
zelden fataal; bij bochtwerk/oversteken wel.

Fix: één gezamenlijke beslissing per been (knie+enkel+hiel+teen tegelijk), met de som
van de kosten.

### B6. `bepaal_afzetbeen` vergelijkt de rechterheup met de **linker**heup **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_analyse.py:333`

> **Fix:** de tiebreaker vergelijkt nu het heup-*midden* van nu met dat van 3 frames
> terug, zoals `detecteer_gewicht_op_been` al deed. Bij stilstaande heupen komt er 0 uit
> en antwoordt hij niet langer altijd `'links'`.

```python
heup_dx = lm_data['r_heup'][0] - heup_history[-3][0]
```

`heup_hist` bevat tuples `(l_heup_x, r_heup_x)` ([schaats_analyse.py:1268](schaats_analyse.py#L1268)),
dus `[-3][0]` is de **linker**heup van 3 frames terug. Er wordt dus geen verschuiving
gemeten maar de constante heupbreedte.

Reproductie (TEST 4): bij volledig stilstaande heupen (l=120, r=180) komt er
`heup_dx = 60` uit i.p.v. 0, en de tiebreaker antwoordt altijd `'links'`.

Dit zit in de per-frame fallback (`cyclus=False`), dus in de normale pijplijn niet
actief — maar wel in de diagnose-stand. Fix: `heup_history[-3][1]`, of vergelijk de
heup-middens zoals `detecteer_gewicht_op_been` doet.

---

## C. Volgen van de schaatser — YOLO-backend

### C1. Geen seed gevonden → stille, volledig lege analyse — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:696-701`

> **Fix:** `analyseer()` raiset nu een `RuntimeError` met uitleg als `_kies_seed` niets
> oplevert; de GUI toont dat als "Fout bij analyseren" en er belandt geen lege analyse
> in de bibliotheek.

```python
seed = _kies_seed(tracklets, frames, doel_punt)
doel_per_frame, ref = {}, KleurReferentie()
if seed is not None:
    ...
```

Is `seed` `None` (geen enkel tracklet, bv. omdat ByteTrack geen ID's uitdeelde), dan
blijft `doel_per_frame` leeg, krijgt géén frame `pose_gevonden`, en loopt de rest van
de pijplijn zonder fout door. De GUI meldt vervolgens gewoon "0 afzetten gevonden"
zonder te zeggen dat er niemand gevolgd is — en slaat die lege analyse op in de
bibliotheek.

Fix: `raise` met een begrijpelijke melding, of een waarschuwing terug naar de GUI.

### C2. Een klik die niemand raakt valt stil terug op "grootste beweger" — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:312-319`

> **Fix:** `_kies_seed` retourneert `(tracklet, klik_gemist)`; bij een gemiste klik gaat
> er een melding via de nieuwe `waarschuwing_callback` van `analyseer()` naar de GUI
> (`AnalyseWorker.waarschuwing` → box na afloop; batch → in het eindoverzicht).

De klik wordt alleen in de eerste `KLIK_ZOEK_FRAMES` (60) frames gezocht. Raakt hij
niemand, dan volgt zonder enige melding de grootste beweger — mogelijk de andere
schaatser. De gebruiker denkt dat zijn keuze is opgevolgd.

Fix: signaleer "je klik kon niet aan een schaatser gekoppeld worden; er wordt nu de
grootste beweger gevolgd" richting de GUI.

### C3. Het seed-tracklet krijgt geen minimumlengte — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:299-325`

> **Fix:** twee kanten. (1) `_kies_seed` geeft bij een klik die meerdere tracklets raakt
> voorrang aan een tracklet van minstens `SEED_MIN_LEN` (5) detecties; raakt de klik
> alleen korte fragmenten, dan telt de klik gewoon. (2) `_stik_keten` begint met
> `_bootstrap()`: zolang de keten korter is dan `SEED_MIN_LEN` worden direct
> aansluitende fragmenten (gat ≤ `BOOTSTRAP_MAX_GAP`) puur **op positie** aangehaakt —
> de kleurreferentie is daar immers nog te dun om iets mee te toetsen. Pas daarna wordt
> de kleur poortwachter. Verificatie (`test_stap6_yolo.py`): een seed van één detectie
> groeit naar een keten van 14 frames met een referentie uit 14 histogrammen.

De kleurreferentie in `_stik_keten` wordt uitsluitend uit het seed-tracklet gevuld
(`:350-352`). Landt de klik op een fragment van 1–2 frames (goed mogelijk ná
`_splits_op_kleur`), dan is de referentie één histogram en is de hele keten-stitching
daarop gebouwd.

Fix: eis een minimale tracklet-lengte voor de seed, of neem bij een korte seed de
kleur van de best passende buur-fragmenten mee.

### C4. `_kleur_sim` kijkt bij achterwaarts stitchen naar de verkeerde kant van het tracklet — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:355-357`

> **Fix:** `_kleur_sim(t, richting)` meet `t[:10]` bij richting +1 en `t[-10:]` bij −1 —
> altijd de kant die aan de keten grenst. Verificatie (`test_stap6_yolo.py`): een
> kandidaat wiens pakkleur meedrijft scoort 0,38 aan de beginkant (onder
> `KLEUR_MATCH_MIN` = 0,45) en 0,75 aan de eindkant; hij werd eerst geweigerd en wordt nu
> teruggestikt.

```python
def _kleur_sim(t):
    sims = [s for s in (ref.sim(d.hist) for d in t[:10]) if s is not None]
```

Altijd de **eerste** 10 detecties. Bij `_probeer(-1)` grenst juist het **eind** van de
kandidaat aan het begin van de keten; daar is de kleur (belichting, schaal) het beste
vergelijkbaar. Voor een lang tracklet waarin de kleur meedrijft kan dat een correcte
kandidaat onder `KLEUR_MATCH_MIN` (0.45) duwen.

Fix: `t[:10]` bij richting +1, `t[-10:]` bij richting −1.

### C5. Masker- en bbox-histogrammen worden onderling vergeleken — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:141-169`

> **Fix:** `_torso_hist` retourneert nu `(hist, uit_masker)` en `Detectie` draagt die
> herkomst (`hist_masker`, plus de property `ref_hist` = het histogram voor zover het als
> bewijs mag dienen). Gevolgen: de kleurreferentie wordt alleen uit masker-histogrammen
> opgebouwd; `_splits_op_kleur` laat een bbox-terugval géén knip veroorzaken (hij telt
> niet mee vóór de knip, maar reset de teller ook niet — er is simpelweg geen oordeel);
> in `_stik_keten` geldt een bbox-oordeel als "onzeker" en vervalt de kleurdrempel ten
> gunste van een halve afstandspoort; en in beide verfijningsroutes mag een
> bbox-histogram een schatting niet meer afwijzen. Verificatie (`test_stap6_yolo.py`):
> vijf frames bbox-terugval midden in een tracklet knippen niet meer (1 stuk), terwijl
> hetzelfde patroon mét masker-histogrammen nog gewoon knipt (3 stukken).

`_torso_hist` levert óf een histogram over de **torso-polygon** (alleen pak-pixels),
óf — als de torso-keypoints onder 0.3 zitten — over een **rechthoek uit de bbox**
(inclusief achtergrond: ijs, boarding, publiek). Die twee zijn niet uitwisselbaar,
maar worden wel met dezelfde drempels (`KLEUR_SPLIT_MIN`, `KLEUR_MATCH_MIN`) tegen
elkaar en tegen de referentie gehouden. Een frame dat op de bbox-terugval valt, zakt
daardoor makkelijk onder de split-drempel → onnodige tracklet-knip → gat in de keten.

Fix: markeer de herkomst op het histogram en vergelijk alleen soortgelijke, of laat
bbox-histogrammen niet meetellen voor de split-beslissing (alleen als zwak bewijs).

### C6. Tracklets die in tijd met de keten overlappen kunnen nooit gestitcht worden — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:365` en `:371`

> **Fix:** een kandidaat mag tot `STITCH_MAX_OVERLAP` (2) frames met de keten overlappen;
> hij moet die alleen wél écht verlengen (`t[-1].frame > keten-eind`, resp.
> `t[0].frame < keten-begin`). De poort groeit met `max(gat, 0)`, en de bestaande
> uitdunning per frame (beste kleurmatch wint) ruimt de dubbele frames op. Verificatie
> (`test_stap6_yolo.py`): een kandidaat die 1 frame overlapt wordt nu gestitcht en de
> keten houdt 12 unieke frames.

`0 < t[0].frame - eind.frame` eist dat de kandidaat volledig ná het keten-eind begint.
Een kandidaat die één frame overlapt (komt voor rond occlusies, waar twee ID's kort
naast elkaar bestaan) valt buiten de selectie en het gat blijft staan.

### C7. Ongebruikte parameter — ✅ OPGELOST (26 jul 2026)
`schaats_yolo.py:423` — `_interpoleer_doel(doel_per_frame, n_frames, fps)` gebruikt
`n_frames` niet. Verwarrend, want de aanroeper geeft `info.totaal` mee terwijl de
resultatenlijst op `len(frames)` gebaseerd is (`:690`, `:712`). Als die twee ooit
verschillen (VFR-.MOV rapporteert `CAP_PROP_FRAME_COUNT` regelmatig verkeerd) suggereert
de signatuur een consistentie die er niet is.

> **Fix:** parameter geschrapt; de signatuur is nu `_interpoleer_doel(doel_per_frame, fps)`.

---

## D. Robuustheid — bibliotheek en GUI

### D1. `open_db` verlaagt `user_version` van een nieuwere database **[bewezen]** — ✅ OPGELOST (26 jul 2026)
`schaats_db.py:154-166`

> **Fix**: `open_db` weigert een nieuwere database met de nieuwe uitzondering
> `schaats_db.BibliotheekTeNieuw` ("gemaakt met een nieuwere versie van de app … werk de app
> bij"), en schrijft `PRAGMA user_version` alleen nog ná een geslaagde aanmaak of migratie —
> een DB die al op de huidige versie staat wordt niet meer aangeraakt. De GUI vangt dat type
> apart af in `_zet_bibliotheek`: eigen dialoogtitel + terugval op de standaard-bibliotheekmap,
> zodat er niet in de nieuwere gedeelde map geschreven wordt. Zelftest uitgebreid
> (`python schaats_db.py`): een v3-DB met een extra kolom blijft ná `open_db` op v3 staan,
> mét die kolom, en de uitzondering wordt geworpen.

```python
if versie == 0:      ...
elif versie < SCHEMA_VERSIE: _migreer(con, versie)
if versie != SCHEMA_VERSIE:  con.execute(f"PRAGMA user_version = {SCHEMA_VERSIE}")
```

Bij `versie > SCHEMA_VERSIE` — een collega met een nieuwere app-versie heeft naar de
gedeelde Drive-map geschreven — wordt niet gemigreerd, maar de versie wél **omlaag**
gezet. Reproductie (`scratchpad/test_db_downgrade.py`):

```
voor open_db :  user_version = 3
na  open_db :  user_version = 2   (app SCHEMA_VERSIE = 2)
v3-kolom staat er nog: True
```

De v3-app die daarna opent, ziet v2 en draait `_migreer(van=2)` opnieuw →
`ALTER TABLE analyse ADD COLUMN video_bytes` → `duplicate column name` → `open_db`
faalt en de GUI valt terug op de standaardmap. In een gedeelde cloudmap met
verschillende app-versies is dat een reële manier om de bibliotheek onbruikbaar te
maken.

Fix: `if versie > SCHEMA_VERSIE: raise` met "deze bibliotheek is gemaakt met een
nieuwere versie van de app", en alleen `PRAGMA user_version` schrijven na een
geslaagde migratie of aanmaak.

### D2. Afsluiten tijdens een lopende analyse: QThread wordt vernietigd terwijl hij draait — ✅ OPGELOST (26 jul 2026)
`schaats_gui.py:2775-2779`

> **Fix**: coöperatief afbreken. Beide workers kregen `breek_af()` + een vlag die de
> voortgangs-callback — die elke pass per frame aanroept — `AnalyseAfgebroken` laat gooien;
> de analyse stopt dus binnen één frame, zonder signaal en zonder op te slaan. `closeEvent`
> roept `_stop_workers()`: draait er een worker, dan een bevestigingsvraag, daarna
> `blockSignals` + `breek_af` + `_wacht_op_worker` (wachtcursor, UI blijft hertekenen, geen
> muis/toets). Stopt de thread niet binnen de deadline — vrijwel altijd een lopende
> videokopie, die bewust **niet** halverwege wordt afgekapt — dan wordt het sluiten geweigerd
> (`event.ignore()`) i.p.v. de thread te slopen. Een `_afsluiten`-vlag zorgt dat een signaal
> dat al in de wachtrij stond geen dialoog of paginawissel meer opent. `requestInterruption()`
> ('Stop na deze video') houdt z'n oude betekenis: de lopende video wordt afgemaakt en
> opgeslagen. Getest met een neppe analyse in de scratchpad (`test_afbreken.py`): beide workers
> stoppen binnen ~16 ms, zonder signaal en zonder opslag; de 'stop na deze video'-route slaat
> video A wél op en slaat B over.

`closeEvent` stopt de speeltimer en sluit de capture, maar wacht niet op
`self.worker` / `self.batch_worker`. Sluit je het venster tijdens een (batch-)analyse —
en die duurt bij 2 s/frame lang — dan wordt de QThread bij interpreter-teardown
vernietigd terwijl hij loopt (`QThread: Destroyed while thread is still running`,
meestal een harde crash). Erger: `sla_analyse_op` kan halverwege de videokopie
afgebroken worden.

Fix: in `closeEvent` `requestInterruption()` + `wait(...)` op een actieve worker, of
het sluiten weigeren met een vraag aan de gebruiker.

### D3. `schaats_eval.py annoteer`: 'q' stopt de annotatielus niet — ✅ OPGELOST (26 jul 2026)
`schaats_eval.py:309-312`

> **Fix:** een `gestopt`-vlag; na het afhandelen van het frame `break`t de for-lus daarop.
> Het al geannoteerde werk wordt nog gewoon weggeschreven.

```python
if res == 'q':
    punten = None
    doelen = []
    break
```

`doelen = []` **herbindt** de naam; de `for doel in doelen`-lus (`:282`) itereert over
het originele lijst-object en gaat gewoon door naar het volgende frame. De gebruiker
blijft frames voorgeschoteld krijgen na 'q'. (De teller in het label,
`len(doelen)` op `:299`, wordt daarna ook 0.)

Fix: een `gestopt`-vlag zetten en na de `while` `break`-en, of `del doelen[:]` +
expliciete controle.

### D4. Bij een detectiegat wordt de hoek-buffer geleegd, bij een beenwissel niet **[latent]** — ✅ vervallen met A1 (26 jul 2026)
> De trailing deque is weg; `_zet_smooth_hoek` breekt de reeks bij een gat **én** bij een
> beenwissel, en de tabel rapporteert de hoek van één frame.
`schaats_analyse.py:1259-1262`

`hoek_buffer.clear()` gebeurt alleen als er géén pose is. Bij een wissel van standbeen
blijft de deque de hoeken van het **andere** been houden. Eindigt een event binnen
`smooth_n` frames na de wissel, dan is de gerapporteerde hoek een mengsel van beide
benen. In de huidige bibliotheek gebeurde dat niet (de events eindigen laat in de run,
en waar het bijna misging — `542bdc04` event 2 — zat er toevallig een detectiegat vlak
vóór), maar bij korte runs is het bereikbaar. Verdwijnt automatisch als A1 wordt
opgelost door de hoek van één frame te rapporteren.

### D5. Mislukt frame lezen laat de weergave desynchroniseren — ✅ OPGELOST (26 jul 2026)
`schaats_gui.py:2264-2267`

> **Fix:** `_meld_leesfout(idx)` stopt het afspelen, zet de weergave terug op het laatst
> geldige frame (waardoor slider, tabelmarkering en grafiekmarker weer kloppen) en meldt
> het in de tijdregel: "Frame N kon niet gelezen worden — beeld staat nog op M".

Geeft `_lees_frame_exact` `None` (voorbij het einde, of een decodefout), dan `return`t
`_toon_frame` meteen: `huidige_idx`, de slider, de tabelmarkering en de statusregel
blijven op het vorige frame staan terwijl de gebruiker denkt verder te zijn gesprongen.
Fix: melden of terugvallen op het laatst geldige frame-index.

### D6. `detecteer_conflictkopieen` meldt élk `.db`-bestand — ✅ OPGELOST (26 jul 2026)
`schaats_db.py:340-347`

> **Fix:** alleen namen die met de stam van `DB_NAAM` beginnen (`schaats….db`) tellen —
> syncers hangen hun markering achter de bestandsnaam. De zelftest controleert nu dat
> `schaats-LAPTOP.db` wél en `adressen.db` níet gemeld wordt.

Elk bestand op `.db` behalve `schaats.db` geldt als conflictkopie — ook een volstrekt
onverwante database die iemand in de map zet. In een gedeelde map levert dat een
waarschuwing bij elke keer openen én bij elke "Vernieuwen". Fix: matchen op de
`schaats*.db`-vorm.

---

## E. Klein / opruimen

Alle punten hieronder zijn opgelost op 26 jul 2026.

| Waar | Wat | Hoe opgelost |
|---|---|---|
| `schaats_analyse.py` `detecteer_gewicht_op_been` | `been_kant` wordt berekend en nooit gebruikt. | Regel geschrapt. |
| `schaats_analyse.py` `teken_been_overlay` | `if hoek > 0:` — bij een negatieve hoek (knie onder de enkel, kapotte detectie) wordt de hoeklijn stil weggelaten i.p.v. het probleem te tonen. | Lijn wordt altijd getekend, **rood** bij hoek ≤ 0. |
| `schaats_analyse.py` `smooth_landmarks_offline` | Pose-segmenten < 3 frames worden overgeslagen: géén L/R-fix, géén uitschieter-verwerping, géén smoothing. Bij versnipperde detectie blijft zo ruwe data staan zonder dat dat ergens blijkt. | De `< 3`-guard is weg; de filters degraderen zelf netjes (SG geeft een te kort venster ongewijzigd terug, de botlengte-check slaat een te korte reeks over) en de L/R-fix doet wél zijn werk. |
| `schaats_analyse.py` `segmenteer_afzetten` | `hoek` en `min_hoek`/`max_hoek` komen alle drie uit de `smooth_hoek`-reeks, terwijl `bepaal_afzet_uit_strek` op ruwe hoeken werkt. | Vervallen met **A1**: alle drie komen nu uit `r.hoek`. |
| `schaats_analyse.py` `analyseer_video` | CLI-voortgang deelt door `totaal` uit `CAP_PROP_FRAME_COUNT`; op VFR-.MOV kan dat >100% geven. | Percentage geklemd op 100, noemer op `max(totaal, frame_nr)`. |
| `schaats_gui.py` (import) | De GUI importeert `_torso_centroid` (privé) uit `schaats_analyse`. | Hernoemd naar de publieke `torso_centroid`. |
| `schaats_gui.py` `_zoom_wiel` | `wheelEvent` van het videolabel wordt overschreven; buiten een geladen analyse doet het muiswiel niets én scrollt de pagina niet. | Zonder analyse (of bij delta 0) gaat het event door naar `QLabel.wheelEvent`. |
| `schaats_gui.py` (grafiek) | De grafiek plot `smooth_hoek` (naijlend); na A1 hoort dit dezelfde grootheid te tonen als de tabel. | Vervallen met **A1**: de grafiek plot `r.hoek`. |
| `schaats_eval.py` `bereken_metrics` | `cv, n = _botlengte_cv(...)`; `n` wordt weggegooid terwijl het aantal metingen juist zegt of de CV betekenis heeft. | `n` gaat mee als `n_<naam>` en `print_metrics` toont `(n=…)` achter elke CV. |

---

## Voorstel voor de volgorde van oplossen

1. ✅ **A1** (hoek van het juiste frame) — grootste en meest systematische meetfout, kleine
   ingreep. Toets met `python schaats_eval.py metrics ... --golden goud_schaats_frontaal.json`;
   die gouden referentie staat er al.
2. ✅ **A2 + A3** (afgekapte runs, robuuste slagperiode) — samen halen ze de spookafzetten
   en de vervuilde gemiddelden eruit.
3. ✅ **B1 + B2** (herseeden + coast-extrapolatie) — de twee tracking-fouten die de
   verkeerde persoon kunnen laten volgen. B2 is een tweeregelige wijziging.
4. ✅ **D1 + D2** (schema-downgrade, afsluiten tijdens analyse) — kunnen data kosten.
5. ✅ **B4, B6, C1, C2** — foutieve of stille correcties.
6. ✅ **De rest** (26 jul 2026): B3, B5, C3–C7, D3, D5, D6 en de complete E-tabel.
   Daarmee is elke bevinding uit deze review afgehandeld.

Reproductiescripts staan in
`%LOCALAPPDATA%\Temp\claude\c--Apps-SchaatsAnalyse\9b13189b-…\scratchpad\`
(`test_reseed.py`, `test_coast.py`, `test_tracking.py`, `test_lr_koppel.py`,
`test_hoek.py`, `test_echt.py`, `test_runs.py`, `test_db_downgrade.py`,
`test_buffer.py`). Als ze blijvend moeten worden, is dat een goede aanleiding voor een
echte testmap in de repo.

De verificatie van **B1 + B2** staat in `…\c1bbebfb-…\scratchpad\test_b1_b2.py` (herseed,
coast van 1–6 frames, kruisende schaatsers als regressie, snelheidsschatting na een gat).

Die van **stap 6** staat in `…\88e1387b-…\scratchpad\`: `test_stap6_analyse.py` (B3 + B5,
draait op de MediaPipe-venv), `test_stap6_yolo.py` (C3–C6, draait op `.venv-yolo`) en
`test_stap6_e2e.py` (hele YOLO-pijplijn op "Schaats frontaal.MOV").

Die van **R1 + R3** staat in `…\137015c8-…\scratchpad\`: `meet.py` (alle 18 npz's door de
pijplijn → JSON, één keer op de oude en één keer op de nieuwe code), `accept.py` (de vijf
acceptatiecriteria van R1 op die twee dumps), `diag.py`/`diag2.py`/`dump_run.py` (het zoeken
naar een scheidende maat, incl. het bewijs dat de hoekdaling-over-het-plateau níet scheidt),
`cv_check.py` (R3: alle CV's over alle npz's) en `gui_smoke.py` (headless PySide6-test van
de tabelkleur, beide tooltips en de CSV-kolom).
