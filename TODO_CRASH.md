# Te doen na de crash van 13-8-2026 (knippen + batch)

## Opnieuw opgetreden: 25-8-2026, nu in de gebundelde app (EXE.md stap 5)

Zelfde flow als 13 augustus — opname knippen, daarna de batch-analyse op het fragment —
dus **dit is geen regressie van de bundeling**, maar dezelfde bug in een omgeving waar hij
slechter zichtbaar is. Wat er precies gebeurde:

- Fragment `00005 13-16.mp4` geknipt uit `00005.MTS` (84 frames, 1920×1080 @ 25 fps, 5,0 MB —
  **het knippen zelf ging goed**, het bestand is compleet en leesbaar), daarna batch-analyse
  gestart met nog een geopende analyse (`IMG_9001.mov`) op de weergavepagina.
- Crash tijdens pass 1, bij frame 21 van 84 (de balk toont 21/168 omdat detectie- en
  verfijningspass elk een schijf krijgen).
- Vlak vóór de crash tekende Qt al niet meer correct: kop over de uitlegtekst heen, de
  voortgangsbalk van de batch midden in de opnametabel, twee statusregels onder elkaar.
- Windows-gebeurtenislogboek: `SchaatsAnalyse.exe` — **foutmodule `Qt6Gui.dll` 6.11.1.0,
  uitzonderingscode `0xc0000005` (access violation), offset `0x00000000000e3880`**, gevolgd
  door een tweede melding `0xc000041d` (fout in een callback). Dus een harde crash in de
  Qt-tekenlaag, niet in de analyse: onnxruntime/DirectML deden op dat moment gewoon hun werk.
- `%LOCALAPPDATA%\SchaatsAnalyse\schaatsanalyse.log` eindigt na
  `Loading …yolo26x-pose-dml.onnx` **zonder traceback** — precies wat je bij een C++-crash
  verwacht: Python komt er niet meer aan te pas. **Punt 3 hieronder is daarmee de eerste stap**,
  want in de exe is er geen console en is dit logboek het enige spoor dat wij zelf schrijven.
- Punt 4 is inmiddels achterhaald: het wisselbestand staat op automatisch beheerd
  (commit limit 29,7 GB bij 15,2 GB RAM, piekgebruik 477 MB), dus geheugenkrapte is hier
  geen waarschijnlijke verklaring meer.

### Waar de crash zit: Qt's schermbeheer (25-8-2026, vastgesteld)

De crash-offset uit het WER-rapport is terug te vertalen naar een functie door de
**exporttabel van `Qt6Gui.dll`** te lezen en de dichtstbijzijnde export vóór de offset te
zoeken (script in de scratchpad; PE-header → DataDirectory → exports, ~60 regels stdlib).
Dat wijst beide keren dezelfde kant op:

| crash | offset | dichtstbijzijnde export |
|---|---|---|
| 20:46 | `0xe3880` | exact het begin van `QScreen::geometry()` |
| 21:25 | `0xe4a79` | `QScreen::virtualSiblings()` + `0x29` |

Dat is het bekende beeld van een **dangling `QScreen`**: Windows herbouwt de
schermconfiguratie, Qt gooit de oude `QScreen`-objecten weg, en wie er dan nog naar wijst
valt om met `0xc0000005` (QTBUG-42985 en verwanten). Onze eigen code raakt `QScreen` maar
op één plek aan — `zet_venstergrootte` in schaats_gui.py, mét `None`-check — en roept
`geometry()`/`virtualSiblings()` nergens zelf aan: die gebruikt Qt intern bij het plaatsen
van vensters. En de crashende flow opent er een paar achter elkaar (knipvenster,
batch-dialoog, doelkiezer per clip).

**Verdachte trigger: het Knipprogramma.** In het systeemlogboek staat op **20:45:49**
activiteit van `Microsoft.ScreenSketch`, 19 seconden vóór de crash van 20:46:08 — een
schermoverlay is precies het soort gebeurtenis dat de schermlijst laat herbouwen. Nog
**niet getoetst**: na de crashlog-build liep de flow drie keer zonder crash (fragmenten van
84, 252 en 62 frames), maar in geen van die runs is een screenshot gemaakt. De test die
het beslist is dus: batch starten en er halverwege een paar keer Win+Shift+S doorheen.

### Bevestigd met de stack (25-8-2026, 22:19-sessie)

Reproductie: knippen → batch-analyse, en er tijdens de analyse een paar keer **Win+Shift+S**
doorheen. Drie eerdere runs zónder screenshots liepen door; deze crashte. Wat het logboek
opving:

```
Windows fatal exception: access violation

Thread 0x00005d74 (most recent call first):     <- de analyse-worker, gewoon bezig
  File "schaats_yolo.py", line 692 in _detecteer_alles
  File "schaats_gui.py", line 1417 in run

Current thread 0x00005528 (most recent call first):   <- hier knalde het
  File "schaats_gui.py", line 6715 in main            (sys.exit(app.exec()))
```

**De hoofdthread valt om in `app.exec()` met een Python-stack van één regel.** Dat is het
bewijs dat de fout niet in onze code zit: zat hij in een slot van ons, dan stonden onze
functies in die stack. De access violation treedt dus op in Qt-interne C++-code die door een
Windows-bericht wordt aangeroepen (schermwijziging) — precies passend bij de
`QScreen`-offsets hierboven. De worker stond ondertussen in `cap.read()` en is alleen
meegesleurd.

Dit is de bekende Qt-bugfamilie op Windows: QTBUG-81359 (access violation in
`QWindowsWindow::checkForScreenChanged` bij schermwijzigingen) en QTBUG-42985. Hier draait
**PySide6 6.11.1**; 6.11.2 bestaat, maar de release notes noemen geen fix hiervoor.

**Wat dit praktisch betekent:** de schade blijft beperkt tot de lopende video — `BatchWorker`
slaat per clip op, dus eerder afgeronde video's van dezelfde batch staan al in de
bibliotheek. En de crash kost geen meting: hij zit in de tekenlaag, niet in de pijplijn.

### Poging tot een minimale reproductie: mislukt (25-8-2026, 22:25-22:50)

Doel was een test van 30 seconden i.p.v. twee minuten, om PySide6-versies zuiver te kunnen
vergelijken. Vier opstellingen, elk geprikkeld met 10-15 knip-overlays (`ms-screenclip:` +
Esc, dus dezelfde gebeurtenis als Win+Shift+S):

| opstelling | uitkomst |
|---|---|
| kaal Qt-venster, 20 repaints/s + werkthread | 10× overleefd |
| idem + elke 0,7 s een dialoog openen/sluiten (via `screen().availableGeometry()`) | 12× overleefd |
| idem + **het echte YOLO-model op DirectML** in een thread (provider bevestigd) | 15× overleefd |
| **de app zelf**, idle op de bibliotheekpagina | 15× overleefd |

Dus: repaints, dialoogvensters, GPU-belasting via DirectML en de app-in-rust zijn **elk
afzonderlijk niet genoeg**. De crash heeft de echte analyse-flow nodig (knippen → batch,
mét de overlay ertussendoor). Dat maakt hem duur om te reproduceren — reken op twee minuten
per poging — maar het sluit wel de goedkope verklaringen uit, en dat is precies wat een
bugrapport aan Qt zou moeten vermelden.

Het testscript staat in de scratchpad (`qtcrash/minimaal.py` + `prikkel.ps1`); niet in de
repo gezet omdat het gereedschap is, geen onderdeel van de app.

**Let op bij het debuggen:** twee draaiende sessies (exe én los script) schrijven in
hetzelfde logboek, en dan is de volgorde niet meer te lezen. Dat heeft op 25-8 al één keer
een verkeerde conclusie opgeleverd — de melding onder een exe-kop bleek van het losse
script te komen. Eén tegelijk.

**Volgende poging het beste in de venv** (`start_gui.bat`): daar is er wél een console met
traceback en kost een codewijziging geen herbouw van 5 minuten.

### Aangepakt (25-8-2026, avond): minder vensters + PySide6 6.11.2

Twee ingrepen, allebei op de enige twee plekken waar wij invloed hebben. De access violation
zélf blijft Qt-intern — daar valt niets af te vangen — maar de **voorwaarde** ervoor is een
verouderde `QScreen`-verwijzing in een venster dat Qt bij een schermwijziging afloopt, en het
aantal van die vensters is wél van ons.

**1. Vensterlek gedicht (was punt 1 hieronder, en het was groter dan gedacht).** Een `QDialog`
met een parent blijft na `exec()` bestaan als verborgen top-level venster, mét native
Windows-venster en `QScreen`-pointer. Alle elf modale kiezers lopen nu via
`toon_dialoog(dlg)` in `schaats_gui.py` (`try: return dlg.exec()` / `finally:
dlg.deleteLater()`), plus twee vensters die niet in die lijst stonden: de knip-voortgangs-
`QProgressDialog` (`close()` verbergt alleen) en het **opstartscherm**, dat na `finish()` de
héle sessie bleef leven. In de crashende flow — knippen → batch van zeven clips — scheelde
dat ~16 achtergebleven vensters, waarvan een paar met een eigen `VideoSpeler` en
`VideoCapture` erin. Dat verklaart vermoedelijk ook de **verhaspelde tekening** vlak vóór de
crash (kop over de uitlegtekst, voortgangsbalk midden in de opnametabel).

*Nagemeten* (offscreen, `MainWindow` erbij; scripts in de scratchpad): drie kale `exec()`-en
laten drie kiezers achter, zes aanroepen via de helper nul. `deleteLater()` en niet
`WA_DeleteOnClose`, want elke aanroepplek leest de uitkomst pás ná `exec()` — apart getoetst
dat de dialoog leesbaar blijft dwars door een `QProgressDialog` (die `processEvents` doet) en
door een geneste dialoog heen, en pas verdwijnt zodra we terug zijn in de hoofd-event-lus,
dus ruim vóór de analyse begint. Voor het opstartscherm geldt hetzelfde langs de andere weg:
zijn `deleteLater()` staat vóór `app.exec()` (lus-niveau 0) en wordt opgeruimd zodra de lus
start — ook dat is apart gemeten.

**2. PySide6 6.11.1 → 6.11.2** in `.venv-yolo`. Een gok: de release notes noemen deze bug
niet. Rooktest gedaan (app start schoon offscreen), maar de exe is nog **niet** herbouwd.

**Wat hiermee nog niet bewezen is:** dat de crash weg is. De reproductie kost twee minuten per
poging (knippen → batch, mét Win+Shift+S ertussendoor) en die is na deze wijziging nog niet
gedraaid. Doe hem in de venv (`start_gui.bat`), en pas als hij een paar rondes overleeft is
een herbouw van de exe de moeite. Blijft hij crashen, dan is optie 3 aan de beurt: melden bij
Qt (QTBUG-81359/QTBUG-42985-familie) — het rapport ligt met dit document zo goed als klaar.

### Crash nummer vijf (25-8-2026, 23:45) — en die was zonder Win+Shift+S

**Belangrijkste voorbehoud vooraf: dit was de óúde exe.** `Report.wer` geeft
`TargetAppVer=2026//08//25:19:33:50` en `Qt6Gui.dll 6.11.1.0`, terwijl de fix hierboven ná
19:33 in de bron is gezet en `bouw.bat` niet is gedraaid. Er heeft dus geen enkele regel van
de opruimactie meegedraaid; deze crash zegt niets over of die helpt.

**De offsets zelf nagerekend** (eigen PE-exporttabellezer op `Qt6Gui.dll` uit de bundel, 10.590
exports), want "dichtstbijzijnde export" is een heuristiek en die wilde ik niet erven:

| tijd | offset | functie | delta |
|---|---|---|---|
| 20:46:08 | `0xe3880` | `QScreen::geometry()` | **+0** |
| 20:46:34 | `0xe3880` | `QScreen::geometry()` | **+0** |
| 21:25:07 | `0xe4a79` | `QScreen::virtualSiblings()` | +0x29 |
| 22:21:50 | `0xe4a79` | `QScreen::virtualSiblings()` | +0x29 |
| 23:45:26 | `0xe4a7c` | `QScreen::virtualSiblings()` | +0x2c |

`virtualSiblings()` loopt van `0xe4a50` tot `0xe4e60`, dus die drie offsets liggen er ruim
binnen. En een crash op **+0** van een member-functie betekent dat de `this`-pointer zelf
stuk is — niet een lege `d`-pointer maar een **weggegooid `QScreen`-object**. Use-after-free,
hard bewijs.

**Waarom er geen screenshot nodig was: deze machine heeft twee beeldschermen.** Het
laptoppaneel (`\\.\DISPLAY1`, 1280×800) én een **EIZO EV2480** (1920×1080, boven het
laptopscherm gepositioneerd: x=319, y=−1080). Beide op 100% DPI, dus geen gemengde schaling.
De EIZO hangt aan **USB-C**: in `Microsoft-Windows-DeviceSetupManager/Admin` staat om 20:57
een `USB Billboard Device` (dat is precies wat een DisplayPort-alt-mode-verbinding aanmeldt),
een VIA-USB-hub (`VID_2109&PID_2817`) en twee EIZO-apparaten (`VID_056D` = EIZO), gevolgd om
20:58:14 door de container `Generic Monitor (EV2480)`.

Daarmee valt de hele "trigger"-vraag anders uit dan gedacht: **Win+Shift+S was nooit de
oorzaak, alleen één manier om de schermlijst te laten herbouwen.** Een USB-C-beeldscherm
levert er van zichzelf meer: de DP-link kan opnieuw trainen, de monitor kan in en uit
energiebesparing zakken, de hub kan even wegvallen. Geen daarvan logt iets in het
gebeurtenislogboek — nagekeken, tussen 23:30 en 23:47 staat er in `System` **niets**.

**Nog een detail dat de moeite is:** de AMD-driver is `31.0.22048.7002` van **25-3-2024**, op
een Windows-build uit 2026. Een ruim twee jaar oude driver die twee schermen aanstuurt
waarvan één over USB-C, terwijl DirectML op diezelfde iGPU staat te rekenen. Een
driver-update is geen bewezen oplossing maar wel de goedkoopste externe variabele die nog
open staat.

**Wat hierop gedaan is: het zichtbaar maken.** `_schermen_naar_logboek()` in `schaats_gui.py`
hangt aan `screenAdded`/`screenRemoved`/`primaryScreenChanged` en per scherm aan
`geometryChanged`/`availableGeometryChanged`/`refreshRateChanged`/`logicalDotsPerInchChanged`,
en schrijft elke wijziging mét tijdstempel naar het logboek. Het staat direct achter het
aanmaken van de `QApplication`, dus ook een wijziging tijdens de zware imports komt erin. Bij
de volgende crash staat er dus zwart op wit óf er vlak ervoor een scherm kwam, ging of van
maat veranderde — precies de regel die nu ontbreekt. Puur meten; het repareert niets.
Getoetst tegen de echte opstelling:
`[scherm 23:52:32] bij start: \\.\DISPLAY1 1280x800 op (0,0) @60Hz | EV2480 1920x1080 op (319,-1080) @60Hz`

**De opruimactie wint hierdoor wél aan plausibiliteit.** Wat er dangelt is een `QScreen*` die
iemand nog vasthoudt. Elk verborgen top-level venster is zo'n houder, en een knip→batch liet
er zestien achter. Met twee schermen waarvan één over USB-C is de kans dat de schermlijst
tijdens een analyse van minuten herbouwd wordt, veel groter dan bij één vast paneel — en dat
verklaart ook waarom de minimale reproducties van 22:25–22:50 niets deden: die openden geen
kiezers en lieten dus niets achter om te dangelen.

### De test van 25-8 23:57 — geen crash, maar het logboek bleef leeg (opgelost)

De eerste knip→batch-run mét de opruimactie **liep gewoon door** (batch klaar, analyse
opgeslagen). Alleen stond er in het logboek van die sessie niets: geen `[scherm ...]`-regel,
geen ultralytics-uitvoer, alleen de sessiekop en de bekende afgehandelde `0x8001010d`.

Oorzaak, en het is een gat dat elke volgende diagnose zou hebben gekost: die sessie draaide
op **`pythonw.exe`**. Dan is `sys.stderr` gewoon `None`, en `start_logboek()` leidde alleen
om als de app bevroren was of `SCHAATSANALYSE_LOG` gezet was — geen van beide. Elke
`sys.stderr.write` gooide dus een `AttributeError`, die in de diagnose-code netjes wordt
weggevangen. Verwarrend genoeg stáát er wél een sessiekop: die schrijft `start_crashlog()`
zelf als er niet omgeleid is, dus het logboek leest als "de app logt" terwijl er niets
binnenkomt.

`start_logboek()` leidt nu ook om als `sys.stderr`/`sys.stdout` `None` is. Getoetst door
`sys.stderr` op `None` te zetten en de route te draaien: omleiding actief, en zowel een
`[scherm ...]`-regel als een gewone `print` komen in het bestand terecht. NB: `start_gui.bat`
gebruikt `python.exe` en had dit probleem niet — het treedt op via een snelkoppeling of
starter die `pythonw.exe` aanroept.

1. ~~**Kiezers opruimen.**~~ ✅ *25-8-2026* — zie hierboven; niet alleen de vier genoemde
   kiezers maar alle elf `exec()`-plekken, plus de knip-voortgangsdialoog en het
   opstartscherm.
2. **Knip-voortgang fijner.** In `_knip_naar_tijdelijk` (`_melden`) niet in hele procenten
   melden maar in promille, of `setValue` op frame-basis met een tijdsdrempel.
3. ~~**Crashlog.**~~ ✅ *25-8-2026* — zit in `schaats_omgeving.start_crashlog()` (niet in
   `main()` maar naast `start_logboek()`, dus vóór alle zware imports: een crash tijdens het
   laden van torch of Qt telt ook). `faulthandler` op een eigen filedescriptor naar het
   logbestand, plus `sys.excepthook` én `threading.excepthook`. Elke sessie eindigt met
   `=== netjes afgesloten … ===`, zodat het ontbreken daarvan de crash aanwijst — nodig,
   want Qt levert bij het opstarten stelselmatig een afgehandelde `0x8001010d` op die
   faulthandler tóch meldt. Getoetst in de zelftest met een echte access violation.
   **De volgende crash laat dus een stack achter; reproduceer hem in de venv.**
4. ~~**Wisselbestand.**~~ Achterhaald: staat inmiddels op automatisch beheerd
   (commit limit 29,7 GB bij 15,2 GB RAM).
