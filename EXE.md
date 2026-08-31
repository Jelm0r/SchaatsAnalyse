# SchaatsAnalyse als installeerbare .exe

*Plan, opgesteld 24 augustus 2026. Stap 1 en 2 uitgevoerd op 24 augustus 2026, stap 3 en 4 op
25 augustus 2026, stap 5 en 6 op 25 augustus en de schone-machinetest op 26 augustus 2026.
Wat er nog met de hand moet gebeuren staat per stap onder "Nog te doen".*

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

## Stap 3 — PyInstaller-spec ✅ *uitgevoerd 25 augustus 2026*

### Wat er nu staat

Drie nieuwe bestanden, en **geen enkele regel in de bestaande modules gewijzigd** — de losse
scripts en beide venvs draaien onveranderd, de exe staat ernaast. Alles wat de bundel nodig
heeft had de code al gekregen in stap 1 en 2.

| bestand | wat |
|---|---|
| `schaatsanalyse.spec` | het PyInstaller-recept: onedir, windowed, noupx, modellen erbuiten |
| `maak_versie.py` | genereert `_versie.py` met de git-stempel (commit/datum/vuil), zelfde vlaggen als `app_versie()` |
| `bouw.bat` | versiestempel → PyInstaller → de modellen naast de exe zetten |

Bouwen: **`bouw.bat`** in de repomap (PyInstaller 6.22.2, geïnstalleerd in `.venv-yolo`).
Duurt ~5 minuten en levert `dist\SchaatsAnalyse\SchaatsAnalyse.exe`.

**Gemeten (25 augustus 2026):** app zonder modellen **787 MB** (`_internal` 740 MB + exe
47 MB), met de drie modellen erbij 1,3 GB. Grootste brokken: torch 365, cv2 112, PySide6 104,
onnxruntime 64, numpy.libs 21, matplotlib 15, PIL 13, torchvision 11 MB. Daarmee onder de
geschatte 1,3–1,6 GB, door twee excludes die allebei zijn nagemeten:

- **PySide6: 634 → 104 MB.** De app gebruikt vier Qt-modules (QtCore/QtGui/QtWidgets/QtCharts);
  de rest staat expliciet in `excludes` omdat matplotlib graag Qt- en tk-backends meesleept.
- **polars weg: −180 MB.** De polars-runtime (177 MB) komt via ultralytics mee, maar álle
  polars-imports daar staan in trainings-, benchmark-, plot- en dataframe-exportpaden — in
  ultralytics zelf becommentarieerd als *"scope for faster 'import ultralytics'"* — en deze app
  doet alleen inferentie. Nagemeten met polars hard geblokkeerd via `sys.meta_path`:
  `import schaats_yolo`, `_laad_yolo()` (de DirectML-route) en daarna `model.track()` én
  `model.predict()` op een frame draaien alle drie zonder polars ooit te laden.

**Wat aangetoond is dat werkt** (twee keer, vóór en ná de polars-exclude):

- De exe **start bevroren**: het kopblok in het logboek meldt `bevroren=True` en `app_dir` =
  de distmap, dus stap 1 lost daar naar de meegeleverde modellen. Geen traceback in het logboek.
- **De luie backend-import slaagt in de bundel** — de riskantste plek van deze stap. 30 s na de
  start zitten `Qt6Charts.dll`, `torch_cpu.dll` en `onnxruntime_pybind11_state.pyd` in het
  proces (`Get-Process ... .Modules`). Dat is meteen het bewijs dat de `find_spec`-vraag van
  `_backend_beschikbaar()` onder PyInstaller's FrozenImporter werkt (anders zou `IS_YOLO` stil
  op False vallen en zou er nóóit torch geladen zijn) en dat `_warm_backend_op()` z'n
  `import schaats_yolo` erdoorheen krijgt.
- De **analyse-kritieke databestanden** zitten erin: `ultralytics/cfg/default.yaml`,
  `ultralytics/cfg/trackers/bytetrack.yaml` en `onnxruntime/capi/DirectML.dll`. rtmlib heeft
  geen hook maar is pure Python en zit compleet in de PYZ.
- **`_versie` staat als PYMODULE in de bundel**, dus de exe kan een versiestempel meegeven aan
  elke analyse (het gedrag van `_versie_uit_bundel()` zelf is in stap 1 al gesimuleerd getest).

**Wat hiermee nog níet bewezen is:** dat een échte analyse in de exe dezelfde meting oplevert,
en dat DirectML aanstaat in plaats van stilzwijgend CPU. Dat valt niet met een startcheck te
doen — dat is stap 5, en het is niet voor niets de belangrijkste stap.

**Vijf dingen om te onthouden:**

- **`--noconfirm` wist de hele uitvoermap**, dus `bouw.bat` kopieert de 557 MB modellen bij elke
  build opnieuw (~30 s van schijf naar schijf). Dat is de prijs voor het buiten de bundel houden:
  ze hoeven niet door de PyInstaller-molen, en een model vervangen kan zonder opnieuw te bouwen.
- **`torch.distributed` is bewust níet uitgesloten**, hoewel het plan het noemde: ultralytics'
  trainer importeert het en die keten hangt aan `from ultralytics import YOLO`. De winst zou
  hooguit **6 MB** zijn (het is Python-broncode; de 315 MB in `torch/lib` zijn de DLL's en die
  moeten mee), dus dat risico is de moeite niet.
- **matplotlib blijft** (31 → 15 MB): `ultralytics.utils.plotting` importeert het bij het laden,
  maar PyInstaller ontdekte zelf dat alleen de **Agg**-backend gebruikt wordt.
- **Geen icoon**: de exe draagt het standaard PyInstaller-icoon. Een `.ico` erbij zetten en
  `icon=` in de spec invullen is genoeg — cosmetisch, dus voor stap 4 bewaard.
- De waarschuwing **`Library nvcuda.dll required via ctypes not found`** hoort erbij: dit is de
  DirectML-build, er zit geen CUDA in. Idem `Hidden import "tzdata" not found` (polars-restant).

### Het oorspronkelijke plan

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

## Stap 4 — Inno Setup-script ✅ *uitgevoerd 25 augustus 2026*

### Wat er nu staat

`installer.iss` verpakt de map uit stap 3 tot **één download van 649 MB**
(`dist\SchaatsAnalyse-setup.exe`, 1.313 MB uitgepakt). Compileren duurt ~5 minuten en hangt
als stap 4/4 achter `bouw.bat`; ontbreekt Inno Setup, dan wordt die stap overgeslagen met een
melding en is de gebouwde map nog gewoon te starten. Inno Setup 6.7, per gebruiker
geïnstalleerd (`winget install JRSoftware.InnoSetup`).

Drie kleine dingen eromheen:

- **`maak_versie.py --toon`** drukt de stempel **ASCII** af (`2026-08-24.a4be1f2a+`) zonder iets
  te schrijven. `bouw.bat` geeft die als `/DVersie=` aan de compiler, zodat "Apps en onderdelen"
  laat zien wélke build er staat. De middenstip van `label` overleeft de console-codepage en een
  `for /f`-lus in cmd.exe niet; het label in de bibliotheek blijft ongewijzigd.
- **`schaatsanalyse.ico`** (het bewaarde punt uit stap 3): `icon=` in de spec, `SetupIconFile` in
  de installer, en daarmee ook het icoon van elke snelkoppeling. Vervangen = het bestand
  overschrijven en opnieuw bouwen.
- **De installer weigert te compileren als de build niet compleet is.** Vier
  `#if !FileExists`-controles op de exe en de drie modellen; zonder die controles zou er een
  installer uitrollen die er goed uitziet en bij de eerste analyse stilzwijgend 178 MB gaat
  downloaden (stap 1.5).

**Rooktest** (stil installeren naar een tijdelijke map, starten, weer verwijderen):

| | uitkomst |
|---|---|
| installeren (`/VERYSILENT`) | exitcode 0, **43 s**, 3.313 bestanden, 1.313 MB |
| modellen | alle drie aanwezig, met de namen die `app_dir()` verwacht |
| snelkoppeling startmenu | wijst naar de geïnstalleerde exe (bureaublad-vinkje overgeslagen met `/TASKS=""`) |
| Apps en onderdelen | `SchaatsAnalyse` · versie `2026-08-24.a4be1f2a+` |
| app starten | draait, venster "Schaats Analyse"; logboek meldt `bevroren=True` en `app_dir` = de installatiemap, geen traceback |
| verwijderen | exitcode 0, map weg, beide snelkoppelingen weg |

**Wat hiermee nog níet bewezen is:** dat een analyse uit de installatie dezelfde meting oplevert
en dat DirectML aanstaat in plaats van stilzwijgend CPU — dat is stap 5. En dat het op een
schone machine werkt (ook stap 5) of hoe SmartScreen zich gedraagt (stap 6).

**Vier dingen om te onthouden:**

- **Sluit een draaiende `dist\SchaatsAnalyse\SchaatsAnalyse.exe` vóór een herbouw.** PyInstaller
  wist met `--noconfirm` de uitvoermap, loopt op de vergrendelde exe stuk met
  `PermissionError: [WinError 5]` — en heeft dan de rest van de map al half opgeruimd.
- **De installatiemap is `{localappdata}\Programs\SchaatsAnalyse`** (`PrivilegesRequired=lowest`):
  geen beheerdersrechten nodig, en schrijfbaar, dus de terugval uit stap 1.4 komt er niet aan te pas.
- **Verwijderen raakt `%LOCALAPPDATA%\SchaatsAnalyse` niet aan** (logboek, en een eventuele eigen
  ONNX-export) en de bibliotheek in Drive al helemaal niet. Het logboek overleeft dus een
  herinstallatie, en `config.json` in `%APPDATA%` houdt het pad naar de Drive-map vast.
- **Inno Setup 6.3 of nieuwer is nodig**: `ArchitecturesAllowed=x64compatible` bestaat daarvóór
  niet. `SetupLogging=yes` staat aan, dus bij een mislukte installatie is er een
  `Setup Log*.txt` in `%TEMP%` om naar te vragen — zelfde gedachte als het logboek uit stap 2.

### Het oorspronkelijke plan

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

## Stap 5 — Verificatie: bewijzen dat het dezelfde meting is ✅ *meetdeel uitgevoerd 25 augustus 2026*

### Wat er nu staat

De kernvraag van deze stap — **levert de gebundelde app dezelfde meting als de venv?** — is
beantwoord op de vaste testclip (`Schaats frontaal.MOV`, 103 frames, zónder doelklik,
`bocht=True`, smoothing 5 / drempel 0,015, dus exact de instellingen van de opgeslagen
analyse). Drie runs naast elkaar gelegd: een verse referentie in `.venv-yolo` op DirectML, een
tegenproef met `SCHAATSANALYSE_CPU=1`, en de analyse die via `dist\SchaatsAnalyse\SchaatsAnalyse.exe`
in de bibliotheek belandde.

| | venv · DirectML | **exe · DirectML** | venv · CPU (tegenproef) |
|---|---|---|---|
| Analysetijd | 90,7 s | **~90 s** (gestopwatcht) | 202,5 s |
| Per frame | **0,88 s** | **~0,9 s** | **1,97 s** |
| Dekking | 103/103 | 103/103 | 103/103 |
| Afzetten | 6 · RLRLRL | 6 · RLRLRL | 6 · RLRLRL |
| Framegrenzen | 0-11, 12-30, 36-47, 48-66, 67-83, 84-102 | idem | idem |
| Hoeken | 42,2 / 42,5 / 42,9 / 40,0 / 45,6 / 50,0\* | idem | idem |

\* = `afgekapt`, telt niet mee in de statistiek.

**Dit zijn 6 afzetten en niet de 8 uit [GPU.md](GPU.md) hoofdstuk 3-4**: die referentie draaide
*mét* doelklik en met `bocht=False`. Hier telt niet welke van de twee "beter" is, maar dat alle
drie de runs hierboven **dezelfde** instellingen gebruiken — die van de opgeslagen analyse, want
anders vergelijk je twee verschillende metingen (GPU.md hoofdstuk 5, valkuil twee).

**De exe-analyse is niet "vergelijkbaar" maar identiek**: alle x/y-landmarks van alle 33
punten over alle 103 frames wijken **≤0,0001 px** af van de verse venv-run (het enige noemens-
waardige verschil zit in de derde kolom, `visibility`, op ≤6·10⁻⁵ — float-afronding, en die
kolom voedt geen enkele meting; `middellijn_dev` heeft hetzelfde NaN-patroon en verschilt 0,0).
Twee exe-runs gemeten: een die op 0,0000 px uitkwam en de controle-run van 25 augustus 20:33 op
0,0001 px — het verschil tussen twee DirectML-runs onderling is dus van dezelfde orde als tussen
exe en venv, oftewel een tienduizendste pixel. `schaats_eval.py vergelijk` geeft dan ook 0,0 px mediaan én p95 op knieën en
enkels. De CPU-tegenproef wijkt zoals verwacht wél een fractie af (mediaan 0,0001 px, max
0,48 px op een hiel, 0,28 px op de metingspunten) **zonder dat de meting verandert** — precies
het beeld uit [GPU.md](GPU.md) hoofdstuk 4.

**Waarom dat geen toeval is, en de reden dat deze stap zo goed afliep**: de bundel bevat
letterlijk dezelfde code en dezelfde rekenkern als de venv. Nagemeten:

- **Alle projectmodules bytecode-identiek.** `_versie`, `schaats_analyse`, `schaats_yolo`,
  `schaats_db`, `schaats_omgeving` en `schaats_perspectief` uit de PYZ, plus het entry-script
  `schaats_gui` uit het CArchive, tegen een verse `compile()` van de repo-bron: alle zeven
  gelijk op de instructiebytes. `_versie` zit er dus ook echt in — zonder die module zou elke
  analyse uit de exe "onbekend" in de Info-dialoog krijgen (stap 3).
- **Alle binaries byte-identiek.** 122 `.dll`/`.pyd` in `_internal` die ook in
  `.venv-yolo\Lib\site-packages` staan: **0 verschillen**, inclusief `DirectML.dll`,
  `onnxruntime.dll`, `onnxruntime_pybind11_state.pyd`, de negen torch-DLL's, `cv2.pyd` en de
  vijftien numpy-binaries.

**DirectML staat aan in de exe.** Het logboek van een exe-analyse toont
`Loading …\yolo26x-pose-dml.onnx for ONNX Runtime inference…`, en dat pad wordt in `_laad_yolo`
**alleen** genomen als `yolo_dml()` waar is; mislukt de route, dan komt er een expliciete
"GPU-route (DirectML) kon niet worden opgezet"-melding in datzelfde logboek, en die staat er
niet. **Let op de regel eronder:** `Using ONNX Runtime 1.24.4 with CPUExecutionProvider` is
géén bewijs van het tegendeel — ultralytics logt daar zijn eigen aanvraag, vlak vóór de
`_dml_sessies()`-patch de provider vervangt (ultralytics kent DirectML niet, zie CLAUDE.md).
Wie hier alsnog aan twijfelt, meet de tijd: 0,88 vs. 1,97 s/frame is geen subtiel verschil — en
dat is precies wat de stopwatch op een analyse uit de exe zelf bevestigde (**~90 s** voor 103
frames, niet ~200 s).

**Opstarttijd** (drie starts achter elkaar, warme machine, bibliotheek op de Drive-map):
opstartscherm na **1,6 s**, hoofdvenster na **2,7 s** — naast de 1,3 s / 2,5 s van de losse
scripts, dus de bundel kost ~0,2 s extra. Het logboek klopt: één kopblok per start, twaalf
starts, **nul tracebacks**.

**Zelftest** `python schaats_db.py` na de `app_versie`-wijziging: *Zelftest OK*.

**Alvast voor de schone machine (punt 5):** de C-runtime zit in de bundel — `vcruntime140.dll`,
`vcruntime140_1.dll`, `msvcp140.dll`, `MSVCP140_ATOMIC_WAIT.dll`, `ucrtbase.dll` en 40
`api-ms-win-*`-stubs. Een pc zonder Visual C++ redistributable hoort dus gewoon te starten.

### Functionele rondgang: hoever gekomen (25 augustus 2026)

| handeling | uitkomst |
|---|---|
| bibliotheek in Drive openen | ✅ opnames + analyses zichtbaar, statuswijziging ("bezig") opgeslagen |
| analyse heropenen | ✅ `IMG_9001.mov` geladen — dus npz, videokopie én **QtCharts** werken bevroren |
| opname bekijken (kijkvenster) | ✅ twee punten gezet en teruggelezen uit `bron_markering` |
| **fragment knippen** | ✅ `00005 13-16.mp4`: 84 frames, 1920×1080 @ 25 fps, 5,0 MB, leesbaar — de `mp4v`-`VideoWriter` uit `opencv_videoio_ffmpeg500_64.dll` doet het in de bundel. Dit was het enige pad dat geen enkele andere test raakt |
| batch-analyse op het fragment | ✅ drie keer gedraaid en opgeslagen (84, 252 en 62 frames) — ❌ maar hij **crasht** als er tijdens de analyse een schermgebeurtenis komt, zie hieronder |
| Info-dialoog | ✅ |
| twee analyses vergelijken | ✅ |
| logboek achteraf | ✅ één kopblok per start, geen tracebacks |

**De rondgang is dus geslaagd**, met één uitzondering die géén bundelprobleem is.

**De crash.** Foutmodule `Qt6Gui.dll`, `0xc0000005` — en met het vangnet uit TODO_CRASH punt 3
(nu ingebouwd) viel hij te lokaliseren: de **hoofdthread valt om in `app.exec()`** met een
Python-stack van één regel, en de offsets wijzen op `QScreen::geometry()` en
`QScreen::virtualSiblings()`. Het is dus Qt-interne code die op een Windows-bericht reageert —
een schermwijziging — en niet onze code; de analyse-worker stond ondertussen gewoon in
`cap.read()`. Reproduceerbaar door tijdens de analyse Win+Shift+S te doen; drie runs zonder
schermafbeelding liepen door. Dezelfde flow crashte op 13 augustus al met de losse scripts.
Volledige waarneming, de offsetanalyse en vier mislukte pogingen tot een minimale reproductie
staan in [TODO_CRASH.md](TODO_CRASH.md).

Wat de bundeling er wél mee te maken heeft: bevroren is er geen console, dus zo'n C++-crash
liet **niets** achter — Python komt er niet aan te pas — en het antwoord moest uit het
Windows-gebeurtenislogboek komen. Daarom staat `faulthandler` nu naast `start_logboek()` in
`schaats_omgeving.py`, samen met de Qt-meldingen die op Windows anders naar de debugger gaan.
Elke sessie eindigt met `=== netjes afgesloten … ===`; ontbreekt die regel, dan is de app daar
gecrasht.

### De schone machine ✅ *26 augustus 2026*

Gedaan op een tweede laptop zonder Python, zonder VC++ runtime en zonder GPU-pakketten: setup
uit Drive gehaald, geïnstalleerd, gestart, **één video geanalyseerd** en rondgeklikt — zonder
problemen. Daarmee is het deel van deze stap gehaald dat je op je eigen machine principieel
niet kunt toetsen: de bundel vindt zijn modellen, de C-runtime zit erin, en de app draait
zonder dat er ooit een venv is opgebouwd.

**Welke build het precies was, is niet meer vast te stellen.** De Info-dialoog van de analyse
daar toonde `43014c8b`, maar dat veld noemt de versie waarmee die **analyse** gemaakt is en
niet de geïnstalleerde app — en de laptop is inmiddels niet meer beschikbaar om
*Instellingen → Apps* na te kijken. Het was dus `43014c8` of de latere `6f96bf14`. Voor de
conclusie maakt dat niet uit: tussen die twee commits is geen enkel bundelingsbestand
aangeraakt (`schaatsanalyse.spec`, `installer.iss`, `bouw.bat`, `maak_versie.py` en
`schaats_omgeving.py` zijn ongewijzigd; het verschil zit in `schaats_analyse`, `schaats_gui`,
`schaats_yolo` en `schaats_db`). Wat een schone machine moest bewijzen — bundelen,
installeren, paden, modellen, starten, meten — is in beide gevallen hetzelfde.

**Leerpunt voor de volgende keer:** noteer de versie uit *Instellingen → Apps → Geïnstalleerde
apps* zolang de machine er nog staat. Die komt uit de `AppVersion` van de installer en zegt
wélke app er draait; de versie in de Info-dialoog van een analyse is bevroren op het moment
van analyseren en beantwoordt die vraag dus niet.

### Het oorspronkelijke plan

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

## Stap 6 — SmartScreen + installatie-instructie ✅ *uitgevoerd 25 augustus 2026*

### Wat er nu staat

[INSTALLEREN.md](INSTALLEREN.md) — de instructie voor de trainers, geschreven voor iemand
zonder Python en zonder beheerdersrechten. Volgorde van het document is de volgorde waarin een
collega tegen dingen aanloopt: downloaden → browserwaarschuwing → SmartScreen → installeren →
eerste start → Drive offline zetten.

Vier keuzes die er inhoudelijk toe doen:

- **De SmartScreen-stap staat al in de inleiding**, niet pas halverwege. Wie het venster
  onverwacht ziet, stopt — en belt of hij een virus binnenhaalt. De uitleg is dus niet "klik
  op Toch uitvoeren" maar **waarom** het venster er is: een certificaat kost €200–400 per jaar,
  onbekend ≠ onveilig. Met erbij het enige wat écht fout kan gaan: *zet nooit je virusscanner
  uit*; bij een quarantaine staat de route via Beveiligingsgeschiedenis → Toestaan op apparaat
  erin, en anders eerst bellen.
- **Downloaden vanaf de Drive-map**, dus staat er meteen bij dat je het bestand eerst offline
  beschikbaar maakt: een setup van 650 MB starten vanaf de streaming-schijf gaat traag en kan
  halverwege afbreken. Dezelfde valkuil als bij de opnames, alleen dan bij de installatie.
- **De bibliotheekmap is als "de belangrijkste stap" gemarkeerd**, met erbij wat er misgaat
  als je hem overslaat (je werkt dan in je Documenten-map en niemand ziet je analyses). Dat is
  de enige instelling waarbij een verkeerde keuze stil blijft en pas weken later opvalt.
- **Het logboek is als eerste-hulpmiddel opgeschreven, niet als voetnoot**: hoe je de map
  opent (`Windows + R` → `%LOCALAPPDATA%\SchaatsAnalyse`), wat je erbij vermeldt, en het
  criterium uit stap 2 — ontbreekt `=== netjes afgesloten ... ===` achter je laatste sessie,
  dan is de app gecrasht. Een trainer kan daarmee zélf zien of er iets te melden valt.

Verder staan de dingen erin die geen instructie zijn maar wel de eerste vragen: analysetijd
(~1 s/frame, GPU automatisch via DirectML, CPU is ~2× trager), de bocht die wordt overgeslagen,
bijwerken (over de oude heen installeren, bibliotheek blijft), verwijderen (raakt Drive en
logboek niet) en de crash bij Win+Shift+S tijdens een analyse uit [TODO_CRASH.md](TODO_CRASH.md)
— als bekende hebbelijkheid mét de workaround, want die vinden ze anders zelf en dan is het een
mysterie.

**Geen certificaat gekocht.** ~€200–400 per jaar voor een handvol trainers weegt niet op tegen
één keer doorklikken, en een EV-certificaat (dat SmartScreen wél meteen vertrouwt) is nog
duurder. Als het aantal gebruikers ooit groeit is dit de plek om terug te komen.

### Nog te doen (kan alleen met de hand)

1. ~~`bouw.bat` opnieuw draaien~~ ✅ — `dist\SchaatsAnalyse-setup.exe` is nu de build van
   26 aug 19:28, `2026-08-26.6f96bf14`, mét het crashlog-vangnet. Nagemeten dat de stempel aan
   **beide** kanten klopt: het `_versie`-moduletje in het PYZ-archief van de exe zegt
   `COMMIT = '6f96bf14', VUIL = False`, en de `ProductVersion` van de installer zegt hetzelfde.
   Die twee worden los van elkaar opgehaald (stap 1 en stap 4 van `bouw.bat`, ~5 min uit
   elkaar), dus een commit tíjdens het bouwen kan ze uit elkaar laten lopen — lees ze daarom
   allebei en niet alleen de bestandseigenschappen.
2. **De setup op de juiste plek in Drive zetten.** Hij staat nu in de wortel van `Mijn Drive`,
   terwijl INSTALLEREN.md naar `Mijn Drive\SchaatsAnalyse\app\SchaatsAnalyse-setup.exe`
   verwijst. Dat is niet alleen een padverschil: de map die met de trainers gedeeld is, is
   `SchaatsAnalyse` — controleer of ze in de wortel van jouw Mijn Drive überhaupt bij het
   bestand kunnen. De map `app\` bestaat daar nog niet; hij zit de bibliotheek niet in de weg
   (`synchroniseer_bronmap` kijkt alleen in `opnames\`, de conflictcheck alleen naar
   `schaats*.db` in de hoofdmap).
3. **INSTALLEREN.md meesturen** — bij de setup in dezelfde Drive-map, want een collega die het
   venster van SmartScreen ziet heeft de uitleg op dát moment nodig.
4. **Let op oude installaties nu de gedeelde bibliotheek op schema v5 staat** (sinds de
   interlacing-commit `2246151`; de analyse van 26 aug 17:26 in Drive is er al mee gemaakt).
   Een installatie van vóór die commit kent v4 en weigert de bibliotheek te openen met
   `BibliotheekTeNieuw` — netjes afgevangen en zonder schade, maar het is wél hét signaal dat
   die machine de nieuwe setup nodig heeft. Deel dus geen v4-build meer uit.

### Het oorspronkelijke plan

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
| `schaatsanalyse.ico` | nieuw — icoon van de exe, de installer en de snelkoppelingen; vervangbaar |
| `installer.iss` | nieuw — Inno Setup: PyInstaller-uitvoer + modellen → `dist\SchaatsAnalyse-setup.exe` |
| `maak_versie.py` | nieuw — schrijft de git-stempel in `_versie.py`; `--toon` drukt hem ASCII af voor de installer |
| `bouw.bat` | nieuw — versie genereren → PyInstaller → modellen erbij → Inno Setup |
| `_versie.py` | nieuw, **gegenereerd**, in `.gitignore` |
| [INSTALLEREN.md](INSTALLEREN.md) | nieuw — instructie voor de trainers: downloaden uit Drive, de SmartScreen-stap, bibliotheekmap kiezen, Drive offline zetten, waar het logboek staat |
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
