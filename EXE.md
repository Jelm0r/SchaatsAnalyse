# SchaatsAnalyse als installeerbare .exe

*Plan, opgesteld 24 augustus 2026. Stap 1 en 2 uitgevoerd op 24 augustus 2026; stap 3 t/m 6 nog niet.*

## Context

De app draait nu alleen op een machine waar met de hand twee Python-omgevingen zijn
opgebouwd. Dat is een drempel voor de andere trainers: zij moeten Python 3.11 installeren,
een venv maken, torch/ultralytics/PySide6/rtmlib erin zetten en — per merk GPU verschillend
— de juiste onnxruntime-variant kiezen. Die GPU-installatie reist bovendien niet mee via git
(`.venv-yolo/` staat in `.gitignore`), dus iedereen doet dat werk opnieuw en meestal fout.

Doel: **één bestand downloaden en de app werkt**, inclusief GPU-versnelling, zonder dat er
ook maar iets geïnstalleerd hoeft te worden.

Bijvangst die het meeste waard is: door `onnxruntime-directml` in te bakken krijgt iedereen
de gemeten 2,2× versnelling zonder installatiewerk — op Intel, AMD én NVIDIA.

## Uitgangspunten

| keuze | besluit |
|---|---|
| Distributievorm | **Installer-exe** (Inno Setup). Eén .exe downloaden, één keer uitpakken, daarna start de app in ~2,5 s zoals nu. Géén onefile — die pakt bij elke start ~1,5 GB uit naar `%TEMP%` en maakt het opstartwerk (opstartscherm + luie backend-import) ongedaan. |
| GPU | **Alleen DirectML.** Eén build voor iedereen, 2,2× sneller dan CPU, valt terug op CPU waar geen GPU is. Geen aparte NVIDIA/CUDA-build (+2,5 GB en twee builds onderhouden). |
| Modellen | **Meeleveren.** Werkt meteen en offline, en het modelbestand ligt vast — een later gedownload ander model zou de metingen stilzwijgend kunnen verschuiven. |
| MediaPipe-backend | **Weglaten.** `IS_YOLO` is in dit pakket altijd waar, dus MediaPipe zou dode ballast zijn. Wel de terugvaltak dichttimmeren (zie stap 1.6). |

Gemeten omvang (24 aug 2026): `.venv-yolo` = 1,9 GB, site-packages = 1.838 MB
(PySide6 634, torch 496, polars-runtime 176, cv2 112, onnxruntime 71, sympy 72).
Modellen samen 557 MB. Verwachting na uitkleden: **~1,3–1,6 GB geïnstalleerd, ~700–900 MB
download.**

---

## Stap 1 — Codefixes: zes plekken, twee helpers ✅ *uitgevoerd 24 aug 2026*

### Wat er nu staat

Alle zes plekken om, plus de terugvalregel. De helpers heten `app_dir()`, `data_dir()` en
`is_bevroren()` en staan boven in [schaats_analyse.py](schaats_analyse.py); `data_dir()` maakt
de map aan (`makedirs(exist_ok=True)`, fouten ingeslikt — de schrijfactie zelf faalt dan wel).

**In de repo-omgeving lost alles naar exact dezelfde bestanden als voorheen**, dus er verandert
niets aan de meting. Nagemeten op deze machine:

```
app_dir     C:\Apps\SchaatsAnalyse          data_dir  C:\Users\<u>\AppData\Local\SchaatsAnalyse
yolo model  C:\Apps\SchaatsAnalyse\yolo26x-pose.pt        (bestaat → geen herdownload)
onnx pad    C:\Apps\SchaatsAnalyse\yolo26x-pose-dml.onnx  (fictief .pt → data_dir)
rtmpose     https://download.openmmlab.com/...            (geen lokaal bestand → URL)
app_versie  2026-08-24 · 85ba31fc+                        (git-route, ongewijzigd)
```

`_laad_yolo()` op het nieuwe absolute pad pakt de bestaande DirectML-export en exporteert of
downloadt niets opnieuw. Zelftests van `schaats_db.py`, `schaats_yolo.py` en
`schaats_perspectief.py` draaien, `py_compile` in beide venvs, en de GUI start. Het bevroren
pad is gesimuleerd met `sys.frozen` + een handgeschreven `_versie.py` → `2026-08-24 · 4df9ab5a`
(en `+` bij vuil): hetzelfde formaat als de git-route.

Twee dingen die de latere stappen hieruit moeten overnemen:

- **`_versie.py` moet `COMMIT`, `DATUM` en `VUIL` definiëren** (str, str, bool) — dat leest
  `_versie_uit_bundel()` in [schaats_db.py](schaats_db.py). Een leeg of ontbrekend `COMMIT`
  betekent "geen stempel" en valt terug op `_git()`.
- **Het RTMPose-bestand moet `rtmpose-x-halpe26-384x288.onnx` heten** en naast de exe staan;
  die naam staat als `RTMPOSE_LOKAAL` in [schaats_yolo.py](schaats_yolo.py). Staat hij er niet,
  dan pakt de app stilzwijgend de URL en downloadt 178 MB bij het eerste gebruik.

De terugvalregel is iets strenger uitgevallen dan hierboven beschreven: bevroren blijft
`IS_YOLO`/`BACKEND_NAAM` staan en levert `_laad_backend()` een `_backend_stuk` op dat één
duidelijke `RuntimeError` gooit, en `_waarschuw_backend_terugval()` toont daar een
**blokkerende** melding ("er kan nu niet geanalyseerd worden; bibliotheek en opnames bekijken
werkt wel") in plaats van de MediaPipe-waarschuwing.

### Het oorspronkelijke plan

Zes plaatsen gaan ervan uit dat er een scriptmap en een git-repo naast de code staan. Alle
zes lossen op met twee kleine helpers in **`schaats_analyse.py`** — dat is de laagste
gemeenschappelijke module: `schaats_gui.py`, `schaats_yolo.py` én `schaats_db.py` importeren
er alle drie uit, dus één plek volstaat.

```python
def app_dir():
    """Map met de meegeleverde bestanden (modellen). Bevroren: naast de exe."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def data_dir():
    """Schrijfbare map voor wat de app zelf aanmaakt (%LOCALAPPDATA%\\SchaatsAnalyse)."""
```

`data_dir()` volgt het bestaande patroon van `_config_pad()` in
[schaats_db.py:67](schaats_db.py#L67), dat al `%APPDATA%` met `expanduser("~")` als terugval
gebruikt. De bibliotheekconfig zelf blijft ongewijzigd in `%APPDATA%` staan — een bestaande
installatie vindt zijn Drive-bibliotheek dus gewoon terug.

De zes aanpassingen:

1. **[schaats_gui.py:392](schaats_gui.py#L392)** — `_MODEL_DIR = dirname(__file__)` → `app_dir()`.
2. **[schaats_analyse.py:2434](schaats_analyse.py#L2434)** — CLI-modelpad → `app_dir()`.
3. **[schaats_yolo.py:204](schaats_yolo.py#L204)** — `STANDAARD_YOLO_MODEL = "yolo26x-pose.pt"`
   is een kále bestandsnaam en hangt dus aan de werkmap. Vanuit een snelkoppeling gestart
   zou ultralytics het model niet vinden en **126 MB opnieuw downloaden** naar een
   willekeurige map. Absoluut maken via `app_dir()` op de gebruikplek
   ([schaats_yolo.py:1279](schaats_yolo.py#L1279)).
4. **`_onnx_pad` [schaats_yolo.py:215](schaats_yolo.py#L215)** — schrijft de DirectML-export
   naast het `.pt`-bestand. We leveren die export mee, dus normaal wordt er niets geschreven;
   maar als het bestand ooit ontbreekt moet de export niet stuklopen op een read-only
   installatiemap. Regel: bestaat hij naast het `.pt` → gebruiken; anders `data_dir()`.
5. **`RTMPOSE_MODEL` [schaats_yolo.py:401](schaats_yolo.py#L401)** — nu een URL. rtmlib's
   `BaseTool.__init__` doet `if not os.path.exists(onnx_model): download_checkpoint(...)`,
   dus **een lokaal pad werkt zonder patch**. Het meegeleverde `.onnx` als pad meegeven op
   de twee aanroepplekken ([schaats_yolo.py:999](schaats_yolo.py#L999) en
   [1007](schaats_yolo.py#L1007)), met de URL als terugval.
6. **`app_versie()` [schaats_db.py:132](schaats_db.py#L132)** — draait `git` in de scriptmap.
   In een exe is er geen repo, dus zou elke analyse van een collega **zonder versiestempel**
   in de bibliotheek belanden — precies het gegeven waarop de Info-dialoog en de
   titel-tooltip leunen. Oplossing: het buildscript genereert een `_versie.py` met commit,
   datum en de vuil-vlag; `app_versie()` leest die eerst als `sys.frozen` waar is en valt
   anders terug op `_git()` zoals nu. **Het `label`-formaat blijft exact
   `"2026-08-24 · 4df9ab5a"`**, zodat oude en nieuwe analyses vergelijkbaar blijven.

Plus één robuustheidsregel: `_laad_backend()`
([schaats_gui.py:349](schaats_gui.py#L349)) valt bij een mislukte YOLO-import terug op
MediaPipe. Die zit niet in het pakket, dus die terugval zou in een exe pas bij de eerste
analyse crashen. Als `sys.frozen` waar is hoort daar een nette melding te komen in plaats
van een terugval — `BACKEND_FOUT` bestaat al en wordt al door
`_waarschuw_backend_terugval()` getoond.

---

## Stap 2 — De uitvoer omleiden (anders crasht een `print`) ✅ *uitgevoerd 24 aug 2026*

### Wat er nu staat

Een nieuwe, **stdlib-only** module [schaats_omgeving.py](schaats_omgeving.py): `is_bevroren()`,
`app_dir()` en `data_dir()` (verhuisd uit [schaats_analyse.py](schaats_analyse.py), dat ze
doorgeeft zodat élke bestaande import ongewijzigd blijft werken) plus `start_logboek()`. Die
verhuizing is de kern van deze stap: de omleiding heeft `data_dir()` nodig op precies het moment
dat cv2/numpy nog niet geladen mógen worden.

[schaats_gui.py](schaats_gui.py) roept `start_logboek()` aan **vóór de Qt-import en dus vóór
`_start_opstartscherm()`**, en alleen als `__name__ == "__main__"` — dezelfde regel als het
opstartscherm zelf, zodat `import schaats_gui` in een meetscript niets kaapt. Kosten: **~1 ms**
(de 8 ms die `-X importtime` toont is bijna helemaal `threading`, en dat stond er al).

Het logboek staat in `%LOCALAPPDATA%\SchaatsAnalyse\schaatsanalyse.log`: regelgebufferd (een
crash laat de laatste complete regels dus wél achter), roterend op 1 MB naar `.log.1`, met per
start een kopblok (tijd, programma, `app_dir`, `data_dir`, pythonversie, bevroren ja/nee).
`sys.stdout` en `sys.stderr` wijzen naar dezelfde stroom, zodat de volgorde klopt. Loggen mag de
app nooit slopen: elke schrijfactie zit in een try/except, en is het bestand niet te openen
(read-only map, volle schijf) dan gaat de uitvoer naar een stille stroom — uitvoer kwijt, maar
géén crash, en dat laatste was hier de hele bedoeling.

**Aangetoond dat het het probleem oplost**, met een gesimuleerde exe (`sys.frozen` + stdout en
stderr op `None`):

| | zonder omleiding | met |
|---|---|---|
| kale `print()` | **stil** — Python slikt hem, geen fout | in het logboek |
| `sys.stdout.write(...)` | `AttributeError: 'NoneType' object has no attribute 'write'` | in het logboek |
| tqdm-balk (ultralytics-download/-export, rtmlib) | **`AttributeError`** | in het logboek |
| logging-handler op `sys.stdout` | stil | in het logboek |
| traceback van een onafgevangen fout | stil | in het logboek |

De aanname bovenaan deze stap klopte dus net niet: een kale `print()` crasht niet (CPython laat
hem vallen als `sys.stdout` None is), maar tqdm en elke rechtstreekse `.write` wél — en alles wat
niet crasht verdwijnt spoorloos, precies wat je nodig hebt als een collega belt dat het niet werkt.

**Getest**: zelftest `python schaats_omgeving.py` (omleiden, een tweede sessie erachteraan,
rotatie naar `.log.1`, en de terugval als het logbestand niet te openen is) in beide venvs; de
zelftests van `schaats_db`/`schaats_yolo`/`schaats_perspectief` ongewijzigd; `py_compile` in
beide venvs; en de GUI echt gestart met `SCHAATSANALYSE_LOG=1` — het kopblok stond in het
logboek. Aan de meting verandert er niets: in de repo-omgeving wordt er zonder die env-var
niets omgeleid.

Twee dingen die de latere stappen hieruit moeten overnemen:

- **`SCHAATSANALYSE_LOG=1` zet de omleiding ook in de gewone venv aan.** Zo is deze route te
  testen zonder eerst een exe te bouwen; zonder de env-var blijft alle uitvoer op de console.
- **De blokkerende backend-melding noemt nu het logpad** (alleen als er echt gelogd wordt), want
  een logboek dat niemand kan vinden is niets waard. `INSTALLEREN.md` (stap 6) hoort datzelfde
  pad te noemen: "stuur me `%LOCALAPPDATA%\SchaatsAnalyse\schaatsanalyse.log`".

### Het oorspronkelijke plan

Een PyInstaller-build met `--windowed` heeft geen console: `sys.stdout` is dan `None` en een
kale `print()` gooit `AttributeError`. Deze codebase print op meerdere plekken — onder meer
`_meld()` in [schaats_yolo.py](schaats_yolo.py) wanneer er géén `waarschuwing_callback` is,
en de eenmalige export-melding in `_laad_yolo`. Die zouden in de exe kunnen klappen.

In het bevroren entry point, vóór alle andere imports:
`sys.stdout` en `sys.stderr` naar een logbestand in `data_dir()` (`schaatsanalyse.log`,
roterend op grootte). Dat lost het probleem op én levert meteen iets om naar te vragen als
een collega meldt dat het niet werkt.

Let op de volgorde: dit moet vóór `_start_opstartscherm()` in
[schaats_gui.py](schaats_gui.py), want dat is al een neveneffect op moduleniveau.

---

## Stap 3 — PyInstaller-spec

Bouwen ín `.venv-yolo` (Python 3.11), met `pip install pyinstaller`. Een `schaatsanalyse.spec`
in de repo (niet een lange commandoregel), zodat de build reproduceerbaar is en in git staat.

- **Modus:** `onedir`, `--windowed`, `--noupx` (UPX sloopt Qt- en torch-DLL's).
- **Entry:** `schaats_gui.py`.
- **Modellen: níet in de bundel.** Die worden door Inno naast de exe gezet (stap 4). Scheelt
  bij elke herbouw honderden MB kopieerwerk en maakt een model vervangen mogelijk zonder
  opnieuw te builden. `app_dir()` vindt ze daar.
- **Meenemen:** `--collect-data ultralytics` (de yaml-configs, waaronder `bytetrack.yaml` en
  `default.cfg`, worden niet automatisch gevonden), `--collect-data rtmlib`.
- **Uitsluiten** — dit is waar de winst zit: `PySide6` levert alle Qt-modules mee (634 MB)
  terwijl de app er vier gebruikt (`QtCore`, `QtGui`, `QtWidgets`, `QtCharts`, geverifieerd
  op de imports in [schaats_gui.py](schaats_gui.py)). Expliciet uitsluiten:
  `QtWebEngine*`, `QtQuick*`, `QtQml`, `Qt3D*`, `QtMultimedia*`, `QtDesigner`, `QtTest`.
  Verder `mediapipe`, `tkinter`, `pytest`, `IPython`, `torch.distributed`.
- **Waarschijnlijk gedoe** (inplannen, niet wegdenken): torch' DLL-verzameling en de
  polars-runtime van 176 MB die ultralytics meebrengt. Reken op een paar rondjes
  bouwen → starten → ontbrekende module toevoegen. `matplotlib` (31 MB) kan waarschijnlijk
  níet weg: `ultralytics.utils.plotting` importeert het bij het laden.

Een `bouw.bat` eromheen die achtereenvolgens `_versie.py` genereert, PyInstaller draait en
Inno aanroept.

---

## Stap 4 — Inno Setup-script

`installer.iss`, resultaat `SchaatsAnalyse-setup.exe`.

- **`PrivilegesRequired=lowest`** → installeert naar `{localappdata}\Programs\SchaatsAnalyse`.
  Twee redenen: geen beheerdersrechten nodig (belangrijk als de trainers op een
  werk-laptop zitten), en de map is schrijfbaar, zodat de terugval uit stap 1.4 werkt.
- `[Files]`: de PyInstaller-uitvoer plus de drie modellen — `yolo26x-pose.pt`,
  `yolo26x-pose-dml.onnx` en het RTMPose-model uit
  `%USERPROFILE%\.cache\rtmlib\hub\checkpoints\`, **hernoemd naar
  `rtmpose-x-halpe26-384x288.onnx`** (= `RTMPOSE_LOKAAL`, zie stap 1). **Niet**
  meenemen: `pose_landmarker_*.task`, `yolo11*-pose.pt` — die horen bij backends die niet
  in dit pakket zitten.
- `Compression=lzma2/max`, `SolidCompression=yes`.
- Snelkoppeling in startmenu + optioneel bureaublad; uninstaller.
- De bibliotheekmap wordt **niet** aangeraakt: die staat in Google Drive en het pad ernaartoe
  in `%APPDATA%\SchaatsAnalyse\config.json`. Verwijderen van de app mag nooit
  trainingsdata raken.

---

## Stap 5 — Verificatie: bewijzen dat het dezelfde meting is

Dit is de belangrijkste stap. Een andere omgeving mag de metingen niet verschuiven — precies
de discipline uit [GPU.md](GPU.md), en die is hier één-op-één herbruikbaar.

1. **Referentie vastleggen** — analyseer `Schaats frontaal.MOV` (103 frames, mét doelklik) in
   de huidige `.venv-yolo`. Noteer dekking, aantal afzetten, benen, framegrenzen en hoeken.
   Bewaar de npz.
2. **Zelfde clip via de exe.** Verwacht: **dezelfde dekking, dezelfde 8 afzetten met dezelfde
   benen en dezelfde framegrenzen**, en hoeken die tot op een tiende gelijk zijn.
3. **Hard maken met de bestaande meetbasis:**
   `python schaats_eval.py vergelijk oud.npz nieuw.npz` — verwacht 0,0 px mediaan op knieën
   en enkels, en **lees het standbeen-getal**, niet het gemengde gemiddelde.
4. **Controleer dat DirectML écht aanstaat in de exe** (niet stilzwijgend CPU): de analysetijd
   moet in de buurt van 0,96 s/frame liggen, niet 2,14. Tegenproef met
   `SCHAATSANALYSE_CPU=1`.
5. **Schone machine** — een pc zonder Python, zonder Visual C++ runtime, zonder handmatig
   geïnstalleerde GPU-pakketten. Zonder deze test weet je alleen dat het bij jou werkt.
6. **Functionele rondgang:** bibliotheek in Drive openen, een analyse heropenen, Info-dialoog
   controleren (moet de versie tonen, niet "onbekend"), een opname bekijken in het
   kijkvenster, een fragment knippen, twee analyses vergelijken. Kijk daarna in
   `%LOCALAPPDATA%\SchaatsAnalyse\schaatsanalyse.log`: er hoort één kopblok per start te
   staan, en verder geen tracebacks (stap 2).
7. **Opstarttijd meten** — moet in de buurt van de huidige 2,5 s liggen.
8. `python schaats_db.py` (zelftest) draaien ná de `app_versie`-wijziging: die controleert al
   de vorm van `app_versie()` en de automatische `app_versie`/`app_commit` in de opgeslagen
   instellingen.

---

## Stap 6 — SmartScreen

De exe is niet gesigneerd, dus Windows toont bij de eerste start "Windows heeft uw pc
beschermd" (doorklikken via *Meer informatie → Toch uitvoeren*), en Defender markeert
PyInstaller-builds met enige regelmaat als verdacht. Een certificaat kost ~€200–400 per jaar
en is voor een handvol trainers waarschijnlijk niet de moeite — maar de installatie-instructie
moet dit **wél** noemen, anders denkt de eerste collega dat er een virus in zit.

Actie: een kort `INSTALLEREN.md` met de download, de SmartScreen-stap, de instructie om de
bibliotheekmap op de gedeelde Drive te kiezen, en — als er tóch iets misgaat — waar het
logboek staat: `%LOCALAPPDATA%\SchaatsAnalyse\schaatsanalyse.log` (stap 2).

---

## Bestanden

| bestand | wat |
|---|---|
| [schaats_omgeving.py](schaats_omgeving.py) | nieuw — `is_bevroren()`/`app_dir()`/`data_dir()` + het logboek; stdlib-only zodat het vóór het opstartscherm kan |
| [schaats_analyse.py](schaats_analyse.py) | geeft de drie helpers door uit `schaats_omgeving`; fix CLI-modelpad |
| [schaats_gui.py](schaats_gui.py) | `_MODEL_DIR` via `app_dir()`; `start_logboek()` vóór alle imports; backend-terugval |
| [schaats_yolo.py](schaats_yolo.py) | modelpad, `_onnx_pad`, RTMPose-pad |
| [schaats_db.py](schaats_db.py) | `app_versie()` leest `_versie.py` als bevroren |
| `schaatsanalyse.spec` | nieuw — PyInstaller-recept |
| `installer.iss` | nieuw — Inno Setup |
| `bouw.bat` | nieuw — versie genereren → PyInstaller → Inno |
| `_versie.py` | nieuw, **gegenereerd**, in `.gitignore` |
| `INSTALLEREN.md` | nieuw — instructie voor de trainers |
| `.gitignore` | `build/`, `dist/`, `_versie.py`, `*-setup.exe` |

## Tijdsinschatting

- **Stap 1 + 2 (codefixes)** — ~1 uur. Deze zijn ook zonder exe geen kwaad; het modelpad uit
  1.3 is zelfs nu al een latent probleem.
- **Stap 3 (spec werkend krijgen)** — een halve tot hele dag, vrijwel volledig
  hidden-import-gepuzzel met torch en ultralytics. Dit is het onvoorspelbare deel.
- **Stap 4 + 6 (installer + instructie)** — ~2 uur.
- **Stap 5 (verificatie incl. schone machine)** — ~een halve dag.

Volgorde-advies: stap 1 en 2 eerst afronden en gewoon in de huidige venv testen. Daarna een
kale spec bouwen en kijken of hij überhaupt start — dat weet je binnen een uur, en dat is het
moment waarop blijkt of dit een middag of twee dagen wordt.
