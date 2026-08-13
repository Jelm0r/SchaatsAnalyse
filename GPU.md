# GPU.md — de analyse op de videokaart laten rekenen

Stand: 13 augustus 2026. Dit bestand beschrijft **wat er in de code zit**, **wat er per laptop
geïnstalleerd moet worden**, en **wat er op de laptop met de integrated GPU nog te winnen valt**.

De aanleiding: de analyse duurde ~2 s/frame en dat bleek volledig op de CPU te draaien. In
`.venv-yolo` stond `torch 2.13.0+cpu` — de CPU-only build, die principieel geen CUDA kan
aanspreken, hoe goed de videokaart ook is. Op de laptop met een **RTX 3050** is dat nu opgelost;
op de laptop met een **integrated GPU** verandert er niets (zie "Deze laptop" hieronder).

---

## 1. Wat er in de code zit (op elke machine hetzelfde)

Alles staat in [`schaats_yolo.py`](schaats_yolo.py) — de MediaPipe-backend is niet aangeraakt en
draait onveranderd op de CPU. **Er valt niets in te stellen**: de code kiest zelf, per pass.

| Onderdeel | Wat het doet |
|---|---|
| `yolo_device()` | `'cuda'` als torch een bruikbare GPU ziet, anders `'cpu'`. Voor de detectiepass (ultralytics). |
| `rtmpose_device()` | `'cuda'` als ONNXRuntime een CUDA-provider heeft, anders `'cpu'`. Voor de verfijningspass. |
| `_infereer(aanroep, ...)` | Voert een ultralytics-aanroep uit op het gekozen apparaat en zet bij een **CUDA-OOM** de rest van de run blijvend op de CPU voort, i.p.v. de analyse te laten sneuvelen. |
| `_maak_rtmpose(...)` | Valt terug op de CPU als de CUDA-sessie niet opbouwt. Nodig omdat `get_available_providers()` zegt dat CUDA *meegecompileerd* is — niet dat de DLL's ook laden. |
| `SCHAATSANALYSE_CPU=1` | Dwingt beide passes naar de CPU. Voor A/B-metingen zonder de omgeving te slopen. |

**Waarom de twee passes apart worden vastgesteld:** ze draaien op verschillende motoren. De
detectiepass gaat via torch/CUDA, de RTMPose-verfijning via ONNXRuntime — en die haalt zijn
CUDA-ondersteuning uit een ánder pakket. Een machine kan dus prima de ene wél en de andere niet
hebben, en dan moet elke pass los van de andere kunnen terugvallen.

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

**Controleren waar je op draait:**

```bash
.venv-yolo\Scripts\python.exe -c "import schaats_yolo as s; print('yolo:', s.yolo_device(), '| rtmpose:', s.rtmpose_device())"
```

Twee keer `cuda` = goed. Staat er `cpu` terwijl er wél een NVIDIA-kaart in zit, controleer dan
eerst `torch.__version__`: eindigt die op `+cpu`, dan is stap 2 niet gelukt.

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

## 4. Deze laptop: integrated GPU

**CUDA is NVIDIA-only.** Op een Intel Iris Xe/UHD of AMD Radeon Graphics geeft
`torch.cuda.is_available()` gewoon `False`, dus beide passes melden `cpu` en draait de analyse
precies zoals voorheen: ~2 s/frame. Er gaat niets stuk — dat is de terugval uit hoofdstuk 1 —
maar er is ook geen winst. Alles hieronder is dus **nog te doen werk**, geen bestaande situatie.

**Eerst vaststellen wat erin zit** (nog niet gedaan, want die laptop was er niet bij):

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
```

### De route die wél kan

Een iGPU is te benaderen via **OpenVINO** (Intel) of **DirectML** (merkonafhankelijk, elke
DirectX-12-GPU, dus ook AMD). Torch praat niet met een iGPU, dus de weg loopt via een
geëxporteerd model in plaats van via `device='cuda'`.

Wat al vaststaat na inspectie van de geïnstalleerde ultralytics (8.4.118):

- Ultralytics heeft een **ingebouwde OpenVINO-backend** en accepteert `device='intel:gpu'`. Het
  controleert zelf of dat apparaat bestaat en **waarschuwt + valt terug op CPU/AUTO** als het er
  niet is — dus een verkeerde gok kan de analyse niet stukmaken.
- rtmlib kent `backend='openvino'` met `device='gpu'` naast de `onnxruntime`-route die we nu
  gebruiken, dus de verfijningspass is met een kleine ingreep om te leiden.

**Volgorde van werken — en dit is de belangrijkste zin van dit document: begin bij de
detectiepass.** Die is **~94% van de analysetijd**. Alleen RTMPose versnellen voelt makkelijker
(rtmlib heeft de knop al zitten), maar levert je hooguit een paar procent op. De winst zit in het
YOLO-deel, en dat vraagt een export:

```bash
# yolo26x-pose naar OpenVINO IR exporteren (eenmalig, levert een map *_openvino_model/)
.venv-yolo\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26x-pose.pt').export(format='openvino')"
```

Daarna zou `yolo_device()` moeten worden uitgebreid van een tweekeuze (`cuda`/`cpu`) naar een
derde tak die het geëxporteerde model + `device='intel:gpu'` kiest. Let op: dat raakt ook de plek
waar `YOLO(...)` wordt geladen in `analyseer()`, want het OpenVINO-model is een **andere map**,
niet hetzelfde `.pt`-bestand.

### Verwachtingen, eerlijk

Reken **niet** op de 10× van de RTX 3050. Een iGPU deelt zijn geheugenbandbreedte met de CPU, en
yolo26x-pose op 1280 px is een zwaar model; realistisch is eerder **1,5–3×**, en het kan op een
gelijkspel uitkomen. Dat is precies waarom hoofdstuk 5 bestaat: **eerst meten, dan bouwen.** Als
de meting tegenvalt, is een kleiner model (`yolo26m-pose.pt`, zie `STANDAARD_YOLO_MODEL`)
waarschijnlijk een grotere winst dan de iGPU — maar dát is wél een meetwijziging en hoort dus
langs het protocol hieronder.

---

## 5. Meetprotocol — hoe je bewijst dat een versnelling geen meetwijziging is

De regel voor dit project: **snelheid mag veranderen, de uitkomst niet.** Een versnelling die de
hoeken een halve graad verschuift is geen versnelling maar een stille regressie.

1. Draai dezelfde video twee keer: één keer met de nieuwe route, één keer met
   `SCHAATSANALYSE_CPU=1` als referentie.
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

## 6. Wat je beter niet doet

- **Geen `half=True` / fp16.** Het is verleidelijk (het is op een GPU gratis snelheid), maar het
  verandert de keypoints in de laatste decimalen en daarmee de gemeten hoeken. Dat is een
  meetwijziging en hoort niet als bijvangst van een snelheidsmaatregel binnen te sluipen.
- **Niet tegelijk de torch-versie ophogen** bij het wisselen naar een CUDA-build. Verander alleen
  het `+cuXXX`-deel, dan blijft ultralytics-compatibiliteit buiten schot.
- **rtmlib altijd met `--no-deps` installeren** — het declareert `opencv-contrib-python` en
  overschrijft anders de bestaande cv2-installatie (staat ook in CLAUDE.md).
- **Geen apparaatkeuze in de GUI bouwen.** De gebruiker (een trainer) kan niet weten wat hier het
  juiste antwoord is; de code hoort dat zelf vast te stellen, zoals nu.

---

## 7. Praktisch advies zolang de iGPU-route er niet is

De bibliotheek staat in Google Drive, dus analyseren en bekijken hoeven niet op dezelfde machine.
Draai de **analyses** op de laptop met de RTX 3050 en gebruik deze laptop voor het **kijken en
beoordelen** — afspelen, skeletten corrigeren, vergelijken, knippen. Dat werk is licht en merkt
van het verschil niets.
