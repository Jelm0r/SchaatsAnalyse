# GPU.md — de analyse op de videokaart laten rekenen

Stand: 13 augustus 2026. Dit bestand beschrijft **wat er in de code zit** en **wat er per laptop
geïnstalleerd moet worden** — voor een NVIDIA-machine (CUDA) én voor een machine met een
integrated GPU (DirectML).

De aanleiding: de analyse duurde ~2 s/frame en dat bleek volledig op de CPU te draaien. In
`.venv-yolo` stond `torch 2.13.0+cpu` — de CPU-only build, die principieel geen CUDA kan
aanspreken, hoe goed de videokaart ook is. Op de laptop met een **RTX 3050** is dat opgelost met
CUDA (~10× sneller); op de laptop met een **AMD Radeon integrated GPU** loopt het via DirectML
(~2,2× sneller, hoofdstuk 4).

---

## 1. Wat er in de code zit (op elke machine hetzelfde)

Alles staat in [`schaats_yolo.py`](schaats_yolo.py) — de MediaPipe-backend is niet aangeraakt en
draait onveranderd op de CPU. **Er valt niets in te stellen**: de code kiest zelf, per pass.

| Onderdeel | Wat het doet |
|---|---|
| `yolo_device()` | `'cuda'` als torch een bruikbare GPU ziet, anders `'cpu'`. Voor de detectiepass (ultralytics). |
| `yolo_dml()` | True als de detectiepass via **DirectML** kan — de route zonder NVIDIA-kaart. CUDA gaat vóór. |
| `rtmpose_device()` | `'cuda'` / `'dml'` / `'cpu'` voor de verfijningspass, naar wat ONNXRuntime aan providers heeft. |
| `_laad_yolo(...)` | Laadt het `.pt`-model (CPU/CUDA) óf, op de DirectML-route, een ONNX-export van dezelfde gewichten. Exporteert die één keer per model (~15 s) naar `<model>-dml.onnx`. |
| `_dml_sessies()` | Contextmanager die ONNXRuntime-sessies binnen het blok op DirectML zet. Nodig omdat ultralytics maar drie providers kent (CUDA, CoreML, CPU) en er geen knop voor DirectML heeft. |
| `_DmlYolo` | Schil om het ONNX-model: bouwt de sessie ín de patch op (ultralytics doet dat pas bij de eerste inferentie) en valt terug op het `.pt`-model op de CPU als DirectML halverwege afhaakt. |
| `_infereer(aanroep, ...)` | Voert een ultralytics-aanroep uit op het gekozen apparaat en zet bij een **CUDA-OOM** de rest van de run blijvend op de CPU voort, i.p.v. de analyse te laten sneuvelen. |
| `_maak_rtmpose(...)` | Valt terug op de CPU als de GPU-sessie niet opbouwt. Nodig omdat `get_available_providers()` zegt dat CUDA/DirectML *meegecompileerd* is — niet dat de DLL's ook laden. |
| `SCHAATSANALYSE_CPU=1` | Dwingt beide passes naar de CPU. Voor A/B-metingen zonder de omgeving te slopen. |

**Waarom de twee passes apart worden vastgesteld:** ze draaien op verschillende motoren. De
detectiepass gaat via torch/CUDA (of via ONNX/DirectML), de RTMPose-verfijning via ONNXRuntime —
en die haalt zijn GPU-ondersteuning uit een ánder pakket. Een machine kan dus prima de ene wél en
de andere niet hebben, en dan moet elke pass los van de andere kunnen terugvallen.

**`YOLO_AUTOINSTALL=false`** staat bovenin `schaats_yolo.py`, vóór de ultralytics-import. De
ONNX-backend van ultralytics doet namelijk `check_requirements("onnxruntime")` en installeert dat
pakket ongevraagd — dwars over `onnxruntime-directml` heen, waarmee de GPU-route stilzwijgend
verdwijnt. Dat is tijdens het bouwen van deze route één keer echt gebeurd (de meting werd ineens
2× trager en de logregel meldde een ander ONNXRuntime-versienummer).

---

## 2. Installatie is **per laptop** — dit reist niet mee via git

`.venv-yolo/` staat in `.gitignore`. Wat je uit GitHub haalt is dus alleen de code; de
CUDA-pakketten moet je op elke machine apart installeren. Op een machine zonder NVIDIA-GPU sla je
dit hele hoofdstuk over — de code valt vanzelf terug op de CPU.

**Op een NVIDIA-machine** (uitgevoerd op de RTX 3050-laptop):

```bash
# 1. Welke CUDA-versie kan de driver aan? Staat rechtsboven in de uitvoer.
nvidia-smi

# 2. Vervang de CPU-build van torch door de CUDA-build. Houd de torch-VERSIE gelijk
#    (hier 2.13.0), alleen het +cuXXX-deel verandert — dan kan ultralytics er niet
#    over struikelen. Kies een cuXXX-index op of onder wat nvidia-smi meldde.
.venv-yolo\Scripts\python.exe -m pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130

# 3. ONNXRuntime apart, voor de RTMPose-verfijning. Dit is echt een losse stap:
#    een CUDA-torch zegt niets over wat ONNXRuntime kan.
.venv-yolo\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv-yolo\Scripts\python.exe -m pip install onnxruntime-gpu
```

**Op een machine zonder NVIDIA-GPU** (uitgevoerd op de laptop met AMD Radeon Graphics) — hier
loopt alles via DirectML, dus torch blijft ongemoeid:

```bash
# 1. Wat zit erin? CUDA heeft alleen zin bij een NVIDIA-kaart.
powershell -c "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion"

# 2. Vervang ONNXRuntime door de DirectML-build. Eerst de oude weg: beide pakketten
#    leveren dezelfde `onnxruntime`-module en kunnen niet naast elkaar staan.
.venv-yolo\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv-yolo\Scripts\python.exe -m pip install onnxruntime-directml

# 3. Gereedschap voor de eenmalige ONNX-export van het YOLO-model.
.venv-yolo\Scripts\python.exe -m pip install onnx onnxslim
```

Bij de eerste analyse daarna exporteert de code zelf `yolo26x-pose-dml.onnx` (~15 s, 220 MB,
blijft naast het `.pt`-bestand staan; staat in `.gitignore`).

**Controleren waar je op draait:**

```bash
.venv-yolo\Scripts\python.exe -c "import schaats_yolo as s; print('yolo:', s.yolo_device(), '| dml:', s.yolo_dml(), '| rtmpose:', s.rtmpose_device())"
```

Op een NVIDIA-machine hoort er twee keer `cuda` te staan; staat er `cpu` terwijl er wél een
NVIDIA-kaart in zit, controleer dan eerst `torch.__version__` — eindigt die op `+cpu`, dan is
stap 2 niet gelukt. Op een machine met integrated GPU hoort er `yolo: cpu | dml: True |
rtmpose: dml` te staan: torch blijft daar op de CPU (dat klopt), het rekenwerk gaat via ONNX.

---

## 3. Referentiemeting (RTX 3050 Laptop, 4 GB)

"Schaats frontaal.MOV", 103 frames, mét doelklik, `bocht=False`, dezelfde machine op CPU vs. GPU:

| | CPU | GPU |
|---|---|---|
| Analysetijd | 194,0 s | **19,5 s** |
| Per frame | 1,88 s | 0,19 s |
| Dekking | 100/103 | 100/103 |
| Afzetten | 8 | 8 |

**9,9× sneller, en de meting is identiek**: dezelfde 8 afzetten, dezelfde benen, dezelfde
framegrenzen, dezelfde hoeken tot op 0,00°. Landmarks verschillen subpixel (mediaan 0,048 px,
p95 0,23 px); de enige uitschieters tot ~3,6 px zitten op de **polsen** en de schouder — punten
die geen enkele meting voedt. GPU-versnelling is hier dus puur tijdwinst, geen stille
meetwijziging, en oude analyses blijven vergelijkbaar met nieuwe.

---

## 4. De laptop met integrated GPU: DirectML (gedaan, 13 augustus 2026)

**CUDA is NVIDIA-only.** Op een Intel Iris Xe/UHD of AMD Radeon Graphics geeft
`torch.cuda.is_available()` gewoon `False`. Torch praat sowieso niet met een iGPU, dus de weg
loopt niet via `device=` maar via een **geëxporteerd ONNX-model** dat door ONNXRuntime met de
**DirectML**-provider wordt uitgevoerd (merkonafhankelijk: elke DirectX-12-GPU, dus ook AMD).
OpenVINO — het alternatief — viel af: de GPU-plugin daarvan is Intel-only.

### Meting (AMD Radeon Graphics, integrated; zelfde clip en protocol als hoofdstuk 3)

| | CPU | DirectML |
|---|---|---|
| Analysetijd | 220,6 s | **98,4 s** |
| Per frame | 2,14 s | 0,96 s |
| Dekking | 100/103 | 100/103 |
| Afzetten | 8 | 8 |

**2,2× sneller bij een identieke meting**: dezelfde acht afzetten, dezelfde benen en
framegrenzen, dezelfde hoeken, en 0,0 px mediaan verschil op knieën en enkels (p95 0,1 px). De
RTMPose-verfijning op DirectML is zelfs **bit-identiek** aan diezelfde pass op de CPU.

Vooraf gemeten op het kale model (1280×1280, ONNXRuntime zonder de rest van de pijplijn):
detectiepass 3,26 s → 0,60 s per frame, RTMPose 0,098 s → 0,035 s. Dat de hele analyse "maar"
2,2× sneller wordt, komt doordat de CPU-referentie via torch draait (2,14 s/frame) en niet via
ONNXRuntime-CPU, en doordat het lezen/decoderen en de kleurmachinerie op de CPU blijven.

### De valkuil die het bijna stilzwijgend fout liet gaan: **exporteer met `dynamic=True`**

Ultralytics letterboxt een `.pt`-model **rechthoekig** (alleen tot een veelvoud van de stride),
maar een ONNX-model met een **vaste** invoervorm krijgt het beeld in een **vierkant** geplakt,
met een brede grijze rand erbij. Het net ziet dan een ander plaatje. Gemeten op deze clip met een
statische export:

- dekking **89/103** in plaats van 100/103,
- eventgrenzen verschoven, twee afzethoeken **18° anders** (58,9° → 40,4° en 53,9° → 64,7°),
- en dat terwijl de detecties zelf per frame vrijwel gelijk waren — het verschil liep via de
  gat-opvulling en de kleurpoort in de verfijningspass.

Met `dynamic=True` valt ultralytics terug op precies dezelfde rechthoekige letterbox als bij het
`.pt`-model en is de meting weer gelijk. Het kost **geen snelheid**: alle frames van één video
hebben dezelfde vorm, dus DirectML bouwt zijn graaf één keer op (98,4 s dynamisch tegen 124,4 s
statisch — dynamisch was zelfs sneller).

Dat dit aan de export lag en niet aan de rekenkunde van DirectML is met een bisect vastgesteld:
hetzelfde statische ONNX-model op de **CPU**-provider gaf óók 89/103. Dat is het gereedschap dat
hoofdstuk 5 beschrijft, en het is precies waarvoor het bedoeld is.

### Wat er nog te winnen valt

Een iGPU deelt zijn geheugenbandbreedte met de CPU, dus de 10× van de RTX 3050 zit er niet in.
Wil je hier écht sneller, dan is een kleiner model (`yolo26m-pose.pt`, zie
`STANDAARD_YOLO_MODEL`) waarschijnlijk een grotere winst dan verdere GPU-tuning — maar dát is wél
een meetwijziging en hoort dus langs het protocol hieronder.

---

## 5. Meetprotocol — hoe je bewijst dat een versnelling geen meetwijziging is

De regel voor dit project: **snelheid mag veranderen, de uitkomst niet.** Een versnelling die de
hoeken een halve graad verschuift is geen versnelling maar een stille regressie.

1. Draai dezelfde video twee keer: één keer met de nieuwe route, één keer met
   `SCHAATSANALYSE_CPU=1` als referentie. Wijkt het af, **bisect dan**: draai het nieuwe *model*
   op de oude *provider* (of andersom). Zo bleek de afwijking van de DirectML-route in de
   ONNX-export te zitten en niet in de GPU (hoofdstuk 4).
2. Gebruik **dezelfde instellingen**, inclusief het `doel_punt` van de oorspronkelijke analyse
   (te vinden in `analyse.instellingen_json` in de bibliotheek-DB).
3. Vergelijk met de bestaande tooling: `python schaats_eval.py vergelijk cpu.npz nieuw.npz`.
   **Lees altijd het standbeen-getal** — zie de `schaats_eval.py`-paragraaf in CLAUDE.md voor
   waarom het gemengde getal misleidt.
4. Geslaagd = dezelfde dekking, hetzelfde aantal afzetten met dezelfde benen en framegrenzen, en
   hoeken die tot op ~0,0° gelijk zijn.

**Twee valkuilen die bij het opzetten van deze meting daadwerkelijk zijn misgegaan** — beide
kosten je een run van vier minuten voordat je doorhebt dat je eigen testscript fout was:

- `segmenteer_afzetten(resultaten, min_lengte=3)` — de tweede parameter is **`min_lengte`, geen
  fps**. Geef je er per ongeluk `info.fps` (30) aan mee, dan wordt élk event weggefilterd en
  krijg je "0 afzetten" terwijl de analyse prima is.
- **Zonder `doel_punt`** kiest de automatische doelkeuze de grootste beweger, en dat is op deze
  clip niet dezelfde schaatser als bij de opgeslagen analyse. Je vergelijkt dan twee verschillende
  metingen met elkaar.

---

## 6. Kun je CPU en GPU tegelijk laten rekenen?

Technisch kan het, praktisch levert het bijna niets op — en dat is geen kwestie van smaak
maar van rekenkunde. De GPU doet een frame in ~0,15 s, de CPU in ~1,75 s. Verdeel je het werk
optimaal over die twee, dan kan de trage kant hooguit **~9%** van de frames voor zijn rekening
nemen; meer, en hij wordt zelf de vertrager terwijl de snelle staat te wachten. De hele winst is
dus die 9% (19,5 s → ~17,8 s). **Hoe sneller je GPU, hoe minder een CPU er nog bij kan
bijdragen.**

**Waar de tijd nu heen gaat** (gemeten op de RTX 3050, 103 frames, `bocht=False`, door de tijd
binnen de modelaanroepen af te zetten tegen de totale analysetijd):

| | tijd | aandeel |
|---|---|---|
| YOLO-detectie | 15,50 s | 78% |
| RTMPose-verfijning | 2,44 s | 12% |
| Al het overige (decoderen, kleur, tracking, smoothing, afgeleiden) | 1,77 s | **9%** |

Dat laatste getal is het belangrijkste van de tabel: de CPU zit **niet** werkeloos naast een
wachtende GPU. Was het 40% geweest, dan liep de aanvoer achter en viel er wél iets te winnen —
maar dan door de aanvoer te repareren, niet door er inferentie bij te proppen. (Kanttekening:
de 150 ms per YOLO-aanroep bevat ook CPU-voorbereiding binnen ultralytics, dus het pure GPU-deel
is iets kleiner dan 78%. Aan de conclusie verandert dat niets.)

**Twee struikelblokken die specifiek voor dit programma gelden:**

- **De tracking is van nature volgordelijk.** De detectiepass draait ByteTrack met
  `persist=True`: elk frame bouwt voort op het vorige, zo houdt elke schaatser een doorlopend ID.
  Frames over twee werkers verdelen breekt die keten. Je zou de video in blokken moeten knippen en
  de sporen daarna weer aaneen moeten naaien — precies de robuustheid raken waar dit programma het
  bij kruisende schaatsers van moet hebben, voor een winst van 9%.
- **De CPU is al bezet** met decoderen, aanleveren en verwerken. Laad je hem óók vol met eigen
  inferentie, dan gaat het aanvoeren naar de GPU trager en kun je netto langzamer uitkomen.

**Op een integrated GPU is het idee nóg minder kansrijk**, om een reden die niet meteen opvalt:
een iGPU heeft geen eigen geheugen maar deelt het werkgeheugen met de processor. Allebei tegelijk
laten rekenen betekent dat ze om dezelfde geheugenbandbreedte vechten.

**Wat wél helpt is minder werk, niet meer apparaten:** de bochtdetectie die er al in zit (gemeten
45% tijdwinst op "Kim tempo") en eventueel een lichter model (`yolo26m-pose`) — maar dat laatste
is een **meetwijziging** en hoort dus langs het protocol in hoofdstuk 5.

---

## 7. Wat je beter niet doet

- **Geen `half=True` / fp16.** Het is verleidelijk (het is op een GPU gratis snelheid), maar het
  verandert de keypoints in de laatste decimalen en daarmee de gemeten hoeken. Dat is een
  meetwijziging en hoort niet als bijvangst van een snelheidsmaatregel binnen te sluipen.
- **Niet tegelijk de torch-versie ophogen** bij het wisselen naar een CUDA-build. Verander alleen
  het `+cuXXX`-deel, dan blijft ultralytics-compatibiliteit buiten schot.
- **rtmlib altijd met `--no-deps` installeren** — het declareert `opencv-contrib-python` en
  overschrijft anders de bestaande cv2-installatie (staat ook in CLAUDE.md).
- **Geen apparaatkeuze in de GUI bouwen.** De gebruiker (een trainer) kan niet weten wat hier het
  juiste antwoord is; de code hoort dat zelf vast te stellen, zoals nu.
- **Nooit een ultralytics-commando draaien zonder `YOLO_AUTOINSTALL=false`** op een
  DirectML-machine — ook niet even snel vanaf de opdrachtregel. Het installeert dan de CPU-build
  van ONNXRuntime over `onnxruntime-directml` heen en de GPU is stilletjes weg. Herstellen:
  `pip uninstall -y onnxruntime` + `pip install --force-reinstall onnxruntime-directml`.
- **Geen statische ONNX-export** (zie hoofdstuk 4): dat verandert de letterbox en daarmee de
  meting.
- **Het opwarm-frame van `_DmlYolo` niet verkleinen** om tijd te besparen. Het model is
  dynamisch, dus elke vorm mág — maar de end2end-kop doet een TopK over `max_det` (300)
  posities, en die zijn er op een klein beeld niet: op 64 px klapt de DirectML-sessie er
  meteen op stuk en draait de hele analyse via de terugval alsnog op de CPU (uitgeprobeerd:
  99 s → 224 s, mét de juiste uitkomst).

---

## 8. Werk verdelen over de twee laptops

De bibliotheek staat in Google Drive, dus analyseren en bekijken hoeven niet op dezelfde machine.
De RTX 3050-laptop blijft met ~0,19 s/frame veruit de snelste voor **analyses**; de laptop met de
integrated GPU doet er met DirectML ~0,96 s/frame over (was ~2,14 s) en is daarmee prima voor
**kijken en beoordelen** — afspelen, skeletten corrigeren, vergelijken, knippen — en voortaan ook
bruikbaar voor een losse analyse tussendoor.
