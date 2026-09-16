"""
Schaats-analyse YOLO-backend
============================
Detectie/tracking-backend op basis van YOLO-pose + ByteTrack (ultralytics), met
daarbovenop een **offline doelkeuze**: omdat dit een batch-tool is, verzamelen we
eerst álle detecties van álle frames en kiezen we daarná globaal welke detectie in
elk frame de doelschaatser is. Dat is veel robuuster dan per frame streaming kiezen:

1. **Detectiepass op hoge resolutie** (`DETECT_IMGSZ`): kleine/bewegingsonscherpe
   schaatsers ver weg worden anders simpelweg niet gevonden.
2. **Tracklets + pakkleur-splits**: ByteTrack-ID's zijn meestal stabiel, maar bij
   kruisende schaatsers "steelt" een ID geregeld de andere schaatser. Per tracklet
   bewaken we het HSV-histogram van de torso (het pak); verspringt de kleur
   aanhoudend, dan wordt het tracklet dáár geknipt.
3. **Keten-stitching met kleur + rijrichting**: vanaf het seed-tracklet (muisklik of
   grootste beweger) worden tracklets aaneengeregen. Een kandidaat telt alleen mee
   als de pakkleur bij de referentie past én zijn startpositie strookt met de
   voorspelde positie (constante snelheid over het gat — de rijrichting van een
   schaatser is voorspelbaar).
4. **Verfijningspass** (`verfijn`): per doel-frame wordt de pose opnieuw geschat met
   een **top-down model op de bekende bounding box** — bij voorkeur **RTMPose-26**
   (Halpe26, via rtmlib/ONNXRuntime): wezenlijk nauwkeuriger dan yolo26x-pose
   (~76 vs ~69,5 COCO-AP, subpixel SimCC-decodering) én met échte hiel/teen-
   keypoints, en bovendien veel sneller op CPU. Detectiegaten worden via
   geïnterpoleerde bboxes alsnog gevuld. De pakkleur blijft de poortwachter, zodat
   de verfijning nooit stiekem de andere schaatser pakt. Is rtmlib niet
   geïnstalleerd, dan valt de pass terug op de oude vierkante-crop + yolo26x-route.

Beide zware passes draaien **op de GPU zodra die er is** en anders gewoon op de CPU;
zie `yolo_device()`/`rtmpose_device()` verderop voor hoe dat per pass wordt bepaald.
Aan de metingen verandert dat niets — dezelfde gewichten in dezelfde fp32-precisie —
alleen aan de looptijd.

De rest van de pijplijn (offline smoothing, afgeleiden, tekenen, GUI) uit
`schaats_analyse.py` wordt ongewijzigd hergebruikt. Vereist torch/ultralytics
(zie .venv-yolo). YOLO levert COCO-17 keypoints; die mappen we in de MediaPipe-33-
indeling. Hiel/teen bestaan niet in COCO en worden daar op de enkel gelegd met
visibility 0; RTMPose-26 levert ze wél echt (Halpe26 → MediaPipe 29–32).
"""
import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

# Vóór de ultralytics-import: de ONNX-backend van ultralytics doet
# `check_requirements("onnxruntime")` en zou daarmee ongevraagd de CPU-build van
# ONNXRuntime installeren — precies over `onnxruntime-directml` heen, het pakket waar de
# iGPU-route hieronder op steunt. Auto-installeren staat daarom uit; de afhankelijkheden
# van dit project worden met de hand beheerd (zie GPU.md).
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

from ultralytics import YOLO   # op moduleniveau: zo faalt de import meteen als torch/
                               # ultralytics ontbreekt, en kiest de GUI netjes MediaPipe.

try:                           # optioneel: RTMPose-verfijning (pip install rtmlib onnxruntime)
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.base import RTMLIB_SETTINGS as _RTMLIB_SETTINGS
    # rtmlib kent van huis uit alleen cpu/cuda/rocm/mps; DirectML — de GPU-route op een
    # Windows-machine zónder NVIDIA — ontbreekt in die tabel. Eén regel eraan toevoegen
    # volstaat: rtmlib zoekt de provider verder gewoon op naam op.
    _RTMLIB_SETTINGS['onnxruntime'].setdefault('dml', 'DmlExecutionProvider')
    IS_RTMPOSE = True
except ImportError:
    IS_RTMPOSE = False

from schaats_analyse import (
    FrameResultaat, Landmark, VideoInfo, video_info,
    smooth_landmarks_offline, verwerk_afgeleiden, zet_horizon, fase_voortgang,
    bocht_ratio, bepaal_bocht_reeks, NUM_POSES_DEFAULT, BOCHT_IN, BOCHT_UIT,
    app_dir, data_dir, open_video, is_interlaced,
)

BACKEND_NAAM = ("YOLO-pose + ByteTrack + RTMPose-verfijning" if IS_RTMPOSE
                else "YOLO-pose + ByteTrack")


# ── Rekenapparaat: GPU waar die er is, anders CPU ───────────────────────────────
# De twee zware passes draaien op verschillende motoren — de detectiepass op
# torch/CUDA (ultralytics), de RTMPose-verfijning op ONNXRuntime — en die halen hun
# GPU-ondersteuning uit verschillende pakketten (een CUDA-build van torch, resp.
# `onnxruntime-gpu`). Een machine kan dus prima de ene wél en de andere niet hebben,
# en daarom wordt het apparaat per pass apart vastgesteld en valt elke pass los van de
# andere terug op de CPU. Aan de uitkomst verandert dat niets: dezelfde gewichten in
# dezelfde fp32-precisie, alleen sneller. (Half precision zou nóg sneller zijn, maar
# verandert de keypoints in de laatste decimalen en daarmee de gemeten hoeken — dat is
# een meetwijziging en hoort niet als bijvangst van een snelheidsmaatregel.)
def _cpu_afgedwongen():
    """`SCHAATSANALYSE_CPU=1` dwingt beide passes naar de CPU — nodig om een analyse op
    GPU tegen een analyse op CPU af te zetten zonder de omgeving te moeten slopen."""
    return bool(os.environ.get('SCHAATSANALYSE_CPU'))


@lru_cache(maxsize=1)
def yolo_device():
    """
    Apparaat voor de YOLO-passes in ultralytics-notatie: 'cuda' als torch een bruikbare
    GPU ziet, anders 'cpu'. Torch wordt hier **lokaal** geïmporteerd (ultralytics heeft
    het al binnengehaald) en de uitkomst wordt per proces gecacht: `cuda.is_available()`
    initialiseert de CUDA-driver, en dat hoeft niet bij elk frame opnieuw.
    """
    if _cpu_afgedwongen():
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


@lru_cache(maxsize=1)
def _ort_providers():
    """De ONNXRuntime-providers van deze installatie, of een lege verzameling."""
    try:
        import onnxruntime as ort
        return frozenset(ort.get_available_providers())
    except Exception:
        return frozenset()


@lru_cache(maxsize=1)
def rtmpose_device():
    """
    Apparaat voor de RTMPose-verfijning: 'cuda' als ONNXRuntime een CUDA-provider heeft,
    'dml' als het er een DirectML-provider heeft, anders 'cpu'. Dat is een ándere vraag
    dan `yolo_device()` — deze pass draait niet op torch, dus een CUDA-torch zegt niets
    over wat ONNXRuntime kan. Staat alleen het gewone `onnxruntime` geïnstalleerd (zonder
    `onnxruntime-gpu`/`onnxruntime-directml`), dan is er wél een GPU maar kan deze pass er
    niet bij, en blijft hij stilletjes op de CPU.
    """
    if _cpu_afgedwongen():
        return 'cpu'
    providers = _ort_providers()
    if 'CUDAExecutionProvider' in providers:
        return 'cuda'
    if 'DmlExecutionProvider' in providers:
        return 'dml'
    return 'cpu'


@lru_cache(maxsize=1)
def yolo_dml():
    """
    True als de **detectiepass** via DirectML op de GPU kan — de route voor een machine
    zonder NVIDIA-kaart (integrated Intel/AMD-GPU), waar `yolo_device()` altijd 'cpu'
    zegt omdat CUDA daar principieel niets doet.

    DirectML is geen torch-apparaat: het bestaat alleen binnen ONNXRuntime. De weg loopt
    dus niet via `device=`, maar via het **geëxporteerde ONNX-model** dat ultralytics ook
    kan laden (zie `_laad_yolo`). CUDA gaat vóór: heeft deze machine een bruikbare
    NVIDIA-GPU, dan is de gewone torch-route sneller én dichter bij de referentiemeting
    in GPU.md.
    """
    if _cpu_afgedwongen() or yolo_device() == 'cuda':
        return False
    return 'DmlExecutionProvider' in _ort_providers()


_gpu_uitgevallen = False   # na een CUDA-OOM: de rest van deze run draait op de CPU


def _infereer(aanroep, waarschuwing_callback=None):
    """
    Voer een ultralytics-aanroep uit op het gekozen apparaat, met de CPU als vangnet
    wanneer het GPU-geheugen volloopt. `aanroep(device)` doet het echte werk.

    Een laptop-GPU heeft weinig VRAM en deelt dat met het bureaublad, dus yolo26x-pose
    op `DETECT_IMGSZ` kan er nét niet in passen — en dan zou een analyse van tien
    minuten halverwege alsnog sneuvelen op een OOM. Na zo'n fout schakelt de hele run
    **blijvend** over op de CPU: per frame terugvallen zou het geheugen telkens opnieuw
    laten vollopen. Halverwege van apparaat wisselen is voor de meting onschadelijk —
    het zijn dezelfde gewichten in dezelfde precisie.
    """
    global _gpu_uitgevallen
    device = 'cpu' if _gpu_uitgevallen else yolo_device()
    try:
        return aanroep(device)
    except Exception as exc:
        if device == 'cpu' or 'out of memory' not in str(exc).lower():
            raise
        _gpu_uitgevallen = True
        try:                       # geef het vastgelopen geheugen terug vóór de retry
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        melding = ("GPU-geheugen vol; de analyse gaat verder op de CPU en duurt "
                   "daardoor langer.")
        if waarschuwing_callback:
            waarschuwing_callback(melding)
        else:
            print(melding)
        return aanroep('cpu')


# yolo26x-pose = meest nauwkeurig (traagst op CPU). Alternatief: "yolo26m-pose.pt"
# (sneller, iets minder nauwkeurig). Ultralytics downloadt het model bij eerste gebruik.
# NB: YOLOv12 is nooit als pose-model uitgebracht (alleen detectie); YOLO26 is het
# nieuwste pose-model in ultralytics, opvolger van yolo11x-pose.
# Bewust alleen de bestandsnaam; `analyseer()` maakt hem absoluut t.o.v. `app_dir()`.
# Een kále naam hangt aan de wérkmap, en vanuit een snelkoppeling gestart zou ultralytics
# het model daar niet vinden en 126 MB opnieuw downloaden naar een willekeurige map.
STANDAARD_YOLO_MODEL = "yolo26x-pose.pt"

# ── Detectie ────────────────────────────────────────────────────────────────────
DETECT_IMGSZ = 1280       # inferentieresolutie detectiepass; 640 mist verre/blurry schaatsers

# opset 17 voor de ONNX-export: hoger levert operatoren op die de DirectML-provider niet
# allemaal kent en die dan stilletjes op de CPU worden uitgevoerd — precies de winst die
# we hier komen halen.
ONNX_OPSET = 17


def _onnx_pad(pt_pad):
    """Pad van het DirectML-model: naast het .pt-bestand als het daar staat, anders in de
    schrijfbare datamap.

    Normaal wordt er niets geschreven — in de repo staat de export er na de eerste keer,
    en een installatie levert hem mee. Maar ontbreekt hij toch, dan moet de eenmalige
    export ergens heen waar zeker geschreven mag worden: een installatiemap kan read-only
    zijn, en dan zou de GPU-route bij elke analyse opnieuw stuklopen.
    """
    stam, _ = os.path.splitext(pt_pad)
    naast = f"{stam}-dml.onnx"
    if os.path.exists(naast):
        return naast
    return os.path.join(data_dir(), os.path.basename(naast))


@contextmanager
def _dml_sessies():
    """
    Laat ONNXRuntime-sessies die binnen dit blok worden opgebouwd op DirectML draaien.

    Ultralytics kiest zijn eigen provider en kent er precies drie: CUDA, CoreML en CPU
    (`ultralytics/nn/backends/onnx.py`). DirectML zit daar niet bij, en er is geen knop
    om het mee te geven — dus wordt `InferenceSession` hier tijdelijk vervangen door een
    variant die de DirectML-provider vooraan zet. Alleen een aanvraag die het bij de CPU
    zou laten wordt omgeleid; vraagt de aanroeper zelf al om een provider (rtmlib doet
    dat), dan blijft die keuze staan.
    """
    import onnxruntime as ort
    origineel = ort.InferenceSession

    def maak(pad, sess_options=None, providers=None, **kw):
        if not providers or list(providers) == ['CPUExecutionProvider']:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        return origineel(pad, sess_options, providers=providers, **kw)

    ort.InferenceSession = maak
    try:
        yield
    finally:
        ort.InferenceSession = origineel


class _DmlYolo:
    """
    Het YOLO-detectiemodel op DirectML, met het gewone .pt-model op de CPU als vangnet.

    Naar buiten toe een gewoon YOLO-model: `track()` en `predict()` gaan ongewijzigd door.
    Twee dingen doet deze schil wél zelf:

    - **De sessie binnen de patch opbouwen.** Ultralytics maakt de ONNXRuntime-sessie pas
      bij de eerste inferentie aan, dus alleen het model laden binnen `_dml_sessies()` is
      niet genoeg — er gaat één dummy-frame doorheen zolang de patch actief is. Daarna
      hergebruikt `track()` diezelfde predictor en dus dezelfde sessie.
    - **Terugvallen op de CPU** als DirectML halverwege afhaakt (geheugen vol, een
      driver die de sessie loslaat). Zelfde afweging als bij de CUDA-OOM in `_infereer`:
      een analyse van tien minuten hoort niet te sneuvelen op een apparaatprobleem.
      ByteTrack begint na zo'n wissel met nieuwe ID's, maar de doelkeuze is offline en
      rijgt tracklets aaneen op pakkleur en rijrichting — daar is een ID-breuk het
      normale geval, geen uitzondering.
    """

    def __init__(self, onnx_pad, pt_pad, waarschuwing_callback=None):
        self._pt_pad = pt_pad
        self._waarschuwing = waarschuwing_callback
        self._dml = True
        # Hoe vaak de ByteTrack-toestand opnieuw is begonnen. `BYTETracker.__init__` roept
        # `reset_id()` aan, dus na een modelwissel tellen de ID's weer vanaf 1 en zou
        # `_bouw_tracklets` (dat puur op ID groepeert) een oud en een nieuw spoor aan
        # elkaar plakken. `_detecteer_alles` leest deze teller en houdt de ID-ruimtes
        # daarom uit elkaar.
        self.generatie = 0
        with _dml_sessies():
            self._model = YOLO(onnx_pad, task='pose')
            # Op de volle DETECT_IMGSZ, ook al is dit maar een dummy: het model is
            # dynamisch, maar de end2end-kop doet een TopK over `max_det` (300) posities
            # en die zijn er op een klein beeld niet — op 64 px klapt de DirectML-sessie
            # er meteen op stuk. Kost één extra graafopbouw (de echte frames worden
            # rechthoekig geletterboxt), en dat is een paar seconden per analyse.
            self._model.predict(np.zeros((DETECT_IMGSZ, DETECT_IMGSZ, 3), np.uint8),
                                imgsz=DETECT_IMGSZ, verbose=False, device='cpu')

    def __getattr__(self, naam):
        # Alles wat deze schil niet zelf afhandelt gaat naar het onderliggende model —
        # `predictor` bijvoorbeeld, want daar hangt ultralytics de trackers aan en die
        # moet `_herstart_tracker` kunnen bereiken. Via `__dict__` i.p.v. `self._model`,
        # anders zou een opzoeking vóórdat `_model` bestaat zichzelf eindeloos aanroepen.
        model = self.__dict__.get('_model')
        if model is None:
            raise AttributeError(naam)
        return getattr(model, naam)

    def track(self, *args, **kw):
        return self._roep('track', *args, **kw)

    def predict(self, *args, **kw):
        return self._roep('predict', *args, **kw)

    def _roep(self, naam, *args, **kw):
        try:
            return getattr(self._model, naam)(*args, **kw)
        except np.linalg.LinAlgError:
            # Niet de GPU: dit komt uit de Kalman-update van ByteTrack, die met
            # `np.linalg.solve` op de geprojecteerde covariantie werkt en op een ontaarde
            # matrix afgaat met "Singular matrix" (ultralytics/trackers/utils/
            # kalman_filter.py). Dat is rekenwerk ná de inferentie en zegt niets over het
            # apparaat. Doorlaten, zodat `_detecteer_alles` de tracker kan herstarten en
            # het apparaat houdt wat het is — de brede tak hieronder schreef zo'n
            # rekenfout op naam van DirectML, meldde dat aan de gebruiker en zette de rest
            # van de analyse op de CPU (hier ~2x trager) terwijl er niets mis was.
            raise
        except Exception as exc:
            if not self._dml:
                raise
            self._dml = False
            self._model = YOLO(self._pt_pad)
            self.generatie += 1        # nieuw model = nieuwe BYTETracker = ID's vanaf 1
            _meld(self._waarschuwing,
                  f"De GPU (DirectML) haakte af ({exc}); de analyse gaat verder op de "
                  "CPU en duurt daardoor langer.")
            return getattr(self._model, naam)(*args, **kw)


def _herstart_tracker(model):
    """Maakt de ByteTrack-toestand leeg, zonder van model of apparaat te wisselen.

    Ultralytics hangt de trackers aan de predictor (`on_predict_start`), dus ze bestaan pas
    ná de eerste `track()` — precies het moment waarop ze stuk kunnen gaan. Retourneert
    False als er niets te herstarten valt; de caller gooit de fout dan door in plaats van
    hem stil te slikken.
    """
    trackers = getattr(getattr(model, 'predictor', None), 'trackers', None)
    if not trackers:
        return False
    gedaan = False
    for t in trackers:
        reset = getattr(t, 'reset', None)
        if callable(reset):
            reset()
            gedaan = True
    return gedaan


def _meld(waarschuwing_callback, tekst):
    """Een stille terugval hoort de gebruiker te bereiken — in de GUI via de callback,
    op de CLI via de uitvoer."""
    if waarschuwing_callback:
        waarschuwing_callback(tekst)
    else:
        print(tekst)


def _laad_yolo(pt_pad, waarschuwing_callback=None):
    """
    Het detectiemodel, op het snelste apparaat dat deze machine biedt.

    Zonder DirectML-route (NVIDIA-machine, of geen `onnxruntime-directml`) is dit gewoon
    `YOLO(pt_pad)`: torch kiest zelf CPU of CUDA via `yolo_device()`. Mét DirectML gaat
    het via een ONNX-export van dezelfde gewichten, die één keer per model wordt gemaakt
    (~15 s) en daarna naast het .pt-bestand blijft staan.

    **`dynamic=True` is geen detail maar de kern van de meetgelijkheid.** Ultralytics
    letterboxt een .pt-model *rechthoekig* (alleen tot een veelvoud van de stride), maar
    een ONNX-model met een vaste invoervorm krijgt het beeld in een **vierkant** van
    `DETECT_IMGSZ` geplakt — dus met een brede grijze rand erbij. Het net ziet dan een
    ander plaatje, en dat is op deze clip geen theoretisch verschil: gemeten zakte de
    dekking van 100/103 naar 89/103, verschoven de eventgrenzen en veranderden twee
    afzethoeken met 18°. Met een dynamische invoervorm valt ultralytics terug op precies
    dezelfde rechthoekige letterbox als bij het .pt-model, en is de meting weer gelijk
    (0,0 px mediaan verschil, dezelfde acht afzetten met dezelfde hoeken — zie GPU.md).
    Het scheelt bovendien niets in snelheid: alle frames van één video hebben dezelfde
    vorm, dus DirectML bouwt zijn graaf één keer op.

    Elke stap kan mislukken — geen `onnx`/`onnxslim` voor de export, een sessie die niet
    opbouwt — en dan is het antwoord steeds hetzelfde: melden en op de CPU verder. Een
    tragere analyse is beter dan geen analyse.
    """
    if not yolo_dml():
        return YOLO(pt_pad)
    onnx_pad = _onnx_pad(pt_pad)
    try:
        if not os.path.exists(onnx_pad):
            _meld(waarschuwing_callback,
                  f"Eenmalig het model exporteren voor de GPU ({os.path.basename(onnx_pad)})...")
            uit = YOLO(pt_pad).export(format='onnx', imgsz=DETECT_IMGSZ,
                                      opset=ONNX_OPSET, dynamic=True)
            os.replace(uit, onnx_pad)
        return _DmlYolo(onnx_pad, pt_pad, waarschuwing_callback)
    except Exception as exc:
        _meld(waarschuwing_callback,
              f"De GPU-route (DirectML) kon niet worden opgezet ({exc}); de analyse "
              "draait op de CPU.")
        return YOLO(pt_pad)

# ── Bocht overslaan (tijdwinst) ─────────────────────────────────────────────────
# De detectiepass is ~94% van de analysetijd, en in de bocht levert die tijd niets op:
# daar is geen bruikbare frontale meting te doen. Zodra `_BochtWacht` zegt dat we in de
# bocht zitten, draait er alleen nog elke `BOCHT_CHECK_S` inferentie om te kijken of het
# rechte stuk alweer begonnen is — de rest van de frames wordt wél gelezen (decoderen is
# verwaarloosbaar, en zo blijft de framenummering exact) maar niet geïnfereerd.
#
# Waarom dit veilig kan: de verfijningspass vult detectiegaten tot GAP_VUL_S (1,0 s) met
# geïnterpoleerde bboxes en schat de pose daar alsnog top-down. De gaten die het
# overslaan achterlaat zijn `BOCHT_CHECK_S` lang, dus ruim daarbinnen: hebben we ergens
# ten onrechte overgeslagen, dan herstelt pass 2 die frames gewoon. Te weinig overslaan
# kost tijd, te veel overslaan kost (bijna) geen dekking.
BOCHT_CHECK_S    = 0.33   # hoe vaak er in de bocht nog geïnfereerd wordt (≈ elke 10 frames
                          # bij 30 fps), maar dan fps-onafhankelijk
BOCHT_START_S    = 0.5    # zolang moet er bochtbewijs zijn (iemand in beeld, maar gedraaid)
                          # vóór we frames gaan overslaan
BOCHT_STIL_S     = 3.0    # ... of zolang helemaal niemand meetbaar in beeld. Ruim boven het
                          # langste detectiegat op een recht stuk in de bibliotheek (2,1 s),
                          # zodat een blur-gat de analyse niet in de skip-stand duwt
BOCHT_BEWEEG_VENSTER_S = 1.0  # venster waarover "beweegt deze persoon?" wordt gemeten
BOCHT_MIN_BEWEGING = 0.10 # verplaatsing + groei van de bbox in dat venster, als fractie van
                          # de eigen lichaamshoogte. Een omstander langs de boarding staat
                          # frontaal in beeld en zou de analyse anders eindeloos op vol tempo
                          # houden (zelfde motief als MIN_VERPLAATSING bij de doelkeuze). Een
                          # schaatser die recht op de camera af komt verplaatst in beeld
                          # nauwelijks maar gróeit ~18%/s — vandaar dat groei meetelt

# ── Pakkleur (torso-HSV-histogram) ──────────────────────────────────────────────
KLEUR_BINS      = (8, 4, 3)  # H, S, V — compact, robuust bij schaal/belichting
KLEUR_MATCH_MIN = 0.45       # min. similarity (1 − Bhattacharyya) om als doel te gelden
KLEUR_SPLIT_MIN = 0.35       # binnen een tracklet: aanhoudend hieronder → ID-diefstal, knip
KLEUR_SPLIT_N   = 3          # aantal opeenvolgende afwijkende frames vóór de knip
REF_HIST_N      = 25         # referentie = gemiddelde van de recentste N doel-histogrammen

# ── Keten-stitching (kleur + rijrichting) ───────────────────────────────────────
STITCH_MAX_GAP_S  = 2.0      # max. tijdsgat dat gestitcht mag worden
STITCH_GATE_BASIS = 0.06     # afstandspoort (genormaliseerd) bij gat 0 ...
STITCH_GATE_GROEI = 0.015    # ... die per gat-frame groeit (onzekerheid van de voorspelling)
STITCH_OVERLAP_GATE = 0.05   # overlapt een kandidaat de keten, dan moet hij op de gedeelde
                             # frames binnen deze afstand liggen: dezelfde schaatser onder
                             # twee ID's, en niet de omstander die de hele clip in beeld staat
SNELHEID_VENSTER  = 5        # aantal detecties waarover de snelheid wordt geschat
MIN_VERPLAATSING  = 0.06     # tracklet-padlengte hieronder = statische omstander
KLIK_ZOEK_S       = 6.0      # zolang zoeken we (in seconden) naar de aangeklikte schaatser
KLIK_POORT_BASIS  = 0.05     # zo ver mag de klik naast een schaatser liggen om hem nog te ...
KLIK_POORT_GROEI  = 0.02     # ... bedoelen, plus dit per seconde die sinds de klik om is
SEED_MIN_LEN      = 5        # detecties; een kortere seed geeft een te dunne kleurreferentie
BOOTSTRAP_MAX_GAP = 3        # frames; zo dichtbij mag een fragment een korte seed aanvullen

# ── Verfijning ──────────────────────────────────────────────────────────────────
GAP_VUL_S       = 1.0        # max. detectiegat dat via geïnterpoleerde bboxes wordt gevuld
# RTMPose-26 (Halpe26 = COCO-17 + hoofd/nek/heupcentrum + voeten), top-down op de
# doel-bbox. 'body7'-gewichten = getraind op 7 datasets, robuust op sportbeelden.
RTMPOSE_MODEL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                 'onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288'
                 '-7fb6e239_20230606.zip')
# Naam van het meegeleverde model naast de app (installatie); rtmlib's `BaseTool` doet
# `if not os.path.exists(onnx_model): download_checkpoint(...)`, dus een lokaal pad werkt
# zonder patch en de URL blijft de terugval voor een kale repo-omgeving.
RTMPOSE_LOKAAL = "rtmpose-x-halpe26-384x288.onnx"
RTMPOSE_INPUT = (288, 384)   # (breedte, hoogte) van de modelinvoer
RTMPOSE_MIN_SCORE = 0.3      # min. gemiddelde been-keypointscore om de schatting te vertrouwen
# Terugvalroute zonder rtmlib (vierkante crop door yolo26x):
VERFIJN_IMGSZ   = 640        # inferentiegrootte op de uitsnede
VERFIJN_MARGE   = 1.9        # cropzijde = marge × grootste bbox-zijde
VERFIJN_MIN_PX  = 256        # ondergrens cropzijde (pixels)

# ── Kijkglas: een te kleine schaatser volgen vanaf een getekend kader ───────────
# De detectiepass pikt een schaatser pas op bij ~80–130 px hoogte (1080p); daaronder
# is er geen bbox en dus ook geen verfijning. Met een kader van de gebruiker op het
# eerste frame propageert het kijkglas de bbox van frame naar frame uit de eigen
# verfijnde keypoints (zie `_Kijkglas`). Geen tweede detector: RTMPose op de vorige
# bbox is ~30–100 ms, en een YOLO-`predict` op een crop tijdens de `track`-sessie zou
# ByteTrack met crop-coördinaten voeden.
KIJKGLAS_COAST_S    = 1.0    # zolang loopt de propagatie zonder treffer door (constante
                             # snelheid) voor de run sterft; zelfde orde als GAP_VUL_S
KIJKGLAS_MARGE      = 0.10   # rand om de keypoint-extent voor de volgende bbox (rtmlib pad
                             # zelf nog eens ×1,25, dus dit blijft klein)
KIJKGLAS_KOPPEL_IOU = 0.3    # koppeltoets: IoU van de gepropageerde bbox met de keten-bbox
                             # op het frame waar de run de keten raakt
KIJKGLAS_SPRONG     = 0.5    # plausibiliteit per stap: het midden mag hoogstens zo'n
                             # fractie van de bbox-hoogte verspringen ...
KIJKGLAS_GROEI      = 1.5    # ... en de hoogte hoogstens deze factor groeien/krimpen; meer
                             # betekent dat RTMPose op iemand anders is gaan zitten
KIJKGLAS_START_SCHALEN = (0.7, 1.0, 1.4)   # eerste frame: drie schalingen van het hand-
                             # getekende kader, hoogste been-score wint (een hand tekent
                             # zelden strak)
KADER_MAAT_MAX      = 3.0    # seed-poort mét kader: een kandidaat mag hoogstens zo veel
                             # hoger zijn dan het kader (omstander naast de klik uitsluiten;
                             # in 6 s groeit een aanrijdende schaatser ~1,7×, gemeten)
KADER_MIN_HOOGTE_PX = 70     # daaronder valt er niets te meten: op `00000 16-14` (schaatser
                             # 35–57 px) gaf RTMPose been-scores van 0,1–0,2 met minder dan
                             # vier bruikbare keypoints, en YOLO op een 5× uitvergrote crop
                             # zag alleen sporadisch een blob — de benen zijn dan ~15 px, en
                             # 2 px fout is al 4° (OPNAME.md). Alleen een waarschuwing.

# COCO-17 keypoint-index → MediaPipe 33-landmark-index.
COCO_NAAR_MP = {
    0: 0,             # neus
    5: 11, 6: 12,     # schouders  (L, R)
    7: 13, 8: 14,     # ellebogen
    9: 15, 10: 16,    # polsen
    11: 23, 12: 24,   # heupen
    13: 25, 14: 26,   # knieën
    15: 27, 16: 28,   # enkels
}
# COCO-indices van torso-hoekpunten in polygonvolgorde (voor het kleur-masker).
# Halpe26 heeft dezelfde eerste 17 indices als COCO, dus dit geldt voor beide.
TORSO_COCO = (5, 6, 12, 11)

# Halpe26-extra's (na de 17 COCO-punten) → MediaPipe-index. Kleine tenen (22/23)
# hebben geen MediaPipe-equivalent en blijven ongebruikt.
HALPE_NAAR_MP = {20: 31, 21: 32,   # grote tenen (L, R) → foot_index
                 24: 29, 25: 30}   # hielen (L, R)


def _coco_naar_landmarks(kp_xy, kp_conf, w, h):
    """COCO-17 keypoints (pixels + conf) → lijst van 33 genormaliseerde Landmarks."""
    lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for c, m in COCO_NAAR_MP.items():
        x, y = kp_xy[c]
        lm[m] = Landmark(float(x) / w, float(y) / h, 0.0, float(kp_conf[c]))
    # Hiel/teen op de enkel leggen met visibility 0 (bestaan niet in COCO).
    for m_foot, m_ankle in ((29, 27), (31, 27), (30, 28), (32, 28)):
        a = lm[m_ankle]
        lm[m_foot] = Landmark(a.x, a.y, 0.0, 0.0)
    return lm


def _halpe26_naar_landmarks(kp_xy, kp_conf, w, h):
    """Halpe26 keypoints (pixels + conf) → 33 genormaliseerde Landmarks, mét voeten."""
    lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for c, m in list(COCO_NAAR_MP.items()) + list(HALPE_NAAR_MP.items()):
        x, y = kp_xy[c]
        vis = float(np.clip(kp_conf[c], 0.0, 1.0))
        lm[m] = Landmark(float(x) / w, float(y) / h, 0.0, vis)
    return lm


# ── Pakkleur ────────────────────────────────────────────────────────────────────
def _torso_hist(frame_bgr, kp_xy, kp_conf, bbox_px=None):
    """
    HSV-histogram van de torso (polygon schouders→heupen) — de "kleur van het pak".
    Valt terug op het centrale bovenstuk van de bounding box als de torso-keypoints
    onbetrouwbaar zijn. Retourneert `(hist_of_None, uit_masker)`.

    Die tweede waarde is de **herkomst**, en die telt: een masker-histogram bevat
    alleen pak-pixels, een bbox-terugval óók achtergrond (ijs, boarding, publiek).
    De twee zijn niet uitwisselbaar, dus een bbox-histogram mag niet met dezelfde
    drempels tegen een masker-referentie worden gehouden — anders zakt zo'n frame
    onterecht onder de split-drempel en knipt de tracklet-splitser een gat in een
    verder prima keten.
    """
    h, w = frame_bgr.shape[:2]
    mask = None
    if min(float(kp_conf[i]) for i in TORSO_COCO) >= 0.3:
        pts = np.array([kp_xy[i] for i in TORSO_COCO], dtype=np.int32)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0) + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, pts - [x0, y0], 255)
    if mask is None and bbox_px is not None:
        bx0, by0, bx1, by1 = bbox_px
        bw, bh = bx1 - bx0, by1 - by0
        x0 = int(max(0, bx0 + 0.25 * bw)); x1 = int(min(w, bx1 - 0.25 * bw))
        y0 = int(max(0, by0 + 0.15 * bh)); y1 = int(min(h, by0 + 0.55 * bh))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None, False
    if mask is None and bbox_px is None:
        return None, False
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(KLEUR_BINS),
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
    return hist, mask is not None


def _hist_sim(a, b):
    """Kleur-similarity in [0, 1]: 1 − Bhattacharyya-afstand."""
    if a is None or b is None:
        return None
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class KleurReferentie:
    """Lopende pakkleur-referentie: gemiddelde van de recentste N doel-histogrammen."""
    def __init__(self):
        self.hists = deque(maxlen=REF_HIST_N)
        self._som = None

    def voeg_toe(self, hist):
        if hist is None:
            return
        if len(self.hists) == self.hists.maxlen:
            self._som -= self.hists[0]
        self._som = hist.copy() if self._som is None else self._som + hist
        self.hists.append(hist)

    @property
    def hist(self):
        if not self.hists:
            return None
        ref = self._som / len(self.hists)
        return ref

    def sim(self, hist):
        return _hist_sim(self.hist, hist)


# ── Detecties & tracklets ───────────────────────────────────────────────────────
@dataclass
class Detectie:
    """Eén persoon-detectie in één frame (alles genormaliseerd op framegrootte)."""
    frame: int
    tid: object                 # ByteTrack-ID of None
    centroid: tuple
    bbox: tuple                 # (x1, y1, x2, y2)
    area: float
    lm: list                    # MediaPipe-33 Landmarks
    hist: object = None         # torso-HSV-histogram of None
    hist_masker: bool = False   # True = uit de torso-polygon, False = bbox-terugval

    @property
    def ref_hist(self):
        """Het histogram voor zover het als bewijs mag dienen: alleen de masker-
        variant. De bbox-terugval bevat achtergrond en zou de referentie vervuilen
        én bij vergelijking met een masker-referentie stelselmatig te laag scoren."""
        return self.hist if self.hist_masker else None


class _BochtWacht:
    """
    Beslist tijdens de detectiepass welk frame nog inferentie krijgt. De bocht is voor
    deze tool onbruikbaar beeld, dus daar hoeft het dure model niet over elk frame — één
    controle per `BOCHT_CHECK_S` volstaat om te merken dat het rechte stuk weer begint.

    Toestand per geanalyseerd frame, uit de detecties van dát frame:
    - **frontaal** — iemand met `bocht_ratio` ≥ `BOCHT_UIT` die ook echt bewéégt. Zet de
      wacht meteen terug op vol tempo.
    - **gedraaid** — iemand meetbaar in beeld, maar met `bocht_ratio` < `BOCHT_IN`: de
      bocht. Na `BOCHT_START_S` overslaan.
    - **niets** — niemand meetbaar. Kan de bocht zijn (schaatser te ver/te klein), maar
      net zo goed een blur-gat op het rechte stuk; daarom pas na `BOCHT_STIL_S`.
    - Zit iedereen tússen de twee drempels in (de hysterese-band), dan zegt dit frame
      niets en blijven de tellers staan waar ze stonden — "onbeslist" is nadrukkelijk
      niet hetzelfde als "niemand in beeld".

    De bewegingseis houdt omstanders langs de boarding buiten de "frontaal"-stem — die
    staan frontaal in beeld en zouden de analyse anders eindeloos op vol tempo houden.
    Een spoor met te weinig historie krijgt het voordeel van de twijfel (telt als
    bewegend), zodat we nooit gaan overslaan puur omdat we iemand nog niet lang genoeg
    zien.
    """

    def __init__(self, fps, w, h, aan=True):
        self.aan = aan
        self.w, self.h = w, h
        fps = fps or 30.0
        self.check    = max(1, int(round(BOCHT_CHECK_S * fps)))
        self.n_start  = max(1, int(round(BOCHT_START_S * fps)))
        self.n_stil   = max(1, int(round(BOCHT_STIL_S * fps)))
        self.venster  = max(2, int(round(BOCHT_BEWEEG_VENSTER_S * fps)))
        self.skip     = False
        self.bocht_n  = 0
        self.stil_n   = 0
        self.laatste  = None          # laatst geïnfereerde frame
        self.sporen   = {}            # tid → deque van (frame, cx_px, cy_px, hoogte_px)

    def analyseren(self, f):
        """Krijgt frame `f` inferentie?"""
        if not self.aan or not self.skip:
            return True
        return self.laatste is None or (f - self.laatste) >= self.check

    def _beweegt(self, d):
        """Verplaatsing + groei van deze persoon over het laatste venster, als fractie
        van de eigen lichaamshoogte. Groei telt mee omdat een schaatser die recht op de
        camera af komt in beeld nauwelijks van z'n plaats komt maar wél groeit."""
        spoor = self.sporen.get(d.tid)
        if d.tid is None or spoor is None or len(spoor) < 2:
            return True                                   # te weinig historie: voordeel van de twijfel
        f0, x0, y0, h0 = spoor[0]
        f1, x1, y1, h1 = spoor[-1]
        if (f1 - f0) < max(2, self.venster // 2) or h1 <= 0:
            return True                                   # te kort stuk om iets te zeggen
        # Verplaatsing + groei, geschaald naar "per seconde" (`venster` = 1 s aan frames)
        # en uitgedrukt in de eigen lichaamshoogte, zodat afstand tot de camera wegvalt.
        beweging = (np.hypot(x1 - x0, y1 - y0) + abs(h1 - h0)) / h1
        return beweging * self.venster / (f1 - f0) >= BOCHT_MIN_BEWEGING

    def voed(self, f, dets):
        """Verwerk de detecties van een geïnfereerd frame."""
        self.laatste = f
        if not self.aan:
            return
        frontaal = gedraaid = gezien = False
        for d in dets:
            hoogte = (d.bbox[3] - d.bbox[1]) * self.h
            if d.tid is not None:
                spoor = self.sporen.setdefault(d.tid, deque(maxlen=self.venster))
                spoor.append((f, d.centroid[0] * self.w, d.centroid[1] * self.h, hoogte))
            ratio = bocht_ratio(d.lm, self.w, self.h)
            if ratio is None:
                continue
            gezien = True
            if ratio >= BOCHT_UIT and self._beweegt(d):
                frontaal = True
            elif ratio < BOCHT_IN:
                gedraaid = True

        # In de skip-stand staat er `check` frames tussen twee metingen; de tellers lopen
        # in frames, dus tel dan ook de overgeslagen frames mee.
        stap = self.check if self.skip else 1
        if frontaal:
            self.skip = False
            self.bocht_n = self.stil_n = 0
        elif gedraaid:
            self.stil_n = 0
            self.bocht_n += stap
            if self.bocht_n >= self.n_start:
                self.skip = True
        elif not gezien:
            self.bocht_n = 0
            self.stil_n += stap
            if self.stil_n >= self.n_stil:
                self.skip = True


def _detecteer_alles(input_pad, model, info, imgsz=DETECT_IMGSZ, progress_callback=None,
                     bocht=True, waarschuwing_callback=None, deinterlacen=False):
    """
    Pass 1: YOLO-pose + ByteTrack over de video op hoge resolutie. Retourneert
    `(frames, buiten_meting)`: per frame een lijst Detectie's (alle personen, met
    torso-kleurhistogram) en per frame of het buiten de meting valt.

    **`buiten_meting` dekt twee soorten frames**, en die horen allebei bij "de bocht is
    niet geanalyseerd": de frames die zijn overgeslagen (géén inferentie), én de
    **controleframes** — de frames die in de overslaan-stand wél zijn geïnfereerd, puur
    om te kijken of het rechte stuk alweer begonnen is. Zo'n controleframe heeft dus wél
    een skelet, maar het is een kijkje en geen meting: het staat midden in een stuk dat
    verder niet bekeken is, dus de buurframes die een afzet zouden moeten aantonen
    ontbreken. De verdikking van dat frame telt wél mee voor het óórdeel (het mag de
    bocht beëindigen — daar is het voor), maar het levert zelf nooit een afzethoek.

    De lus leest de frames zelf i.p.v. `model.track(source=pad, stream=True)` te laten
    streamen — anders is er geen manier om een frame wél te lezen maar niet te
    infereren, en dat is precies wat `_BochtWacht` in de bocht wil (zie daar). Elk frame
    wordt gelezen, dus de framenummering blijft exact gelijk aan die van de video;
    decoderen is verwaarloosbaar naast de ~2 s inferentie per frame.
    """
    w, h = info.w, info.h
    frames, buiten_meting = [], []
    # ByteTrack begint na elke herstart weer bij ID 1, en `_bouw_tracklets` groepeert puur
    # op ID — zonder eigen ID-ruimte per generatie zou een oud en een nieuw spoor tot één
    # tracklet samensmelten, wat in het ergste geval twee verschillende schaatsers aan
    # elkaar plakt. Vandaar: elke generatie begint boven de hoogste ID die al vergeven is.
    herstarts, vorige_gen, id_basis, hoogste_tid = 0, 0, 0, 0
    wacht = _BochtWacht(info.fps, w, h, aan=bocht)
    cap = open_video(input_pad, deinterlacen)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {input_pad}")
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not wacht.analyseren(f):
            frames.append([])
            buiten_meting.append(True)
        else:
            # Stond de wacht in de overslaan-stand, dan is dít een controleframe.
            buiten_meting.append(wacht.skip)

            def volg(dev, _f=frame):
                return model.track(_f, persist=True, imgsz=imgsz,
                                   tracker='bytetrack.yaml', classes=[0],
                                   verbose=False, device=dev)

            try:
                res = _infereer(volg, waarschuwing_callback)[0]
            except np.linalg.LinAlgError as exc:
                # De Kalman-update van ByteTrack loopt op een ontaarde covariantie stuk
                # ("Singular matrix"). Eén frame is dat niet waard: tracker leegmaken en
                # dit frame overdoen, op hetzelfde apparaat. Bewust **geen** melding aan
                # de gebruiker — er wisselt geen apparaat, er verandert niets aan de
                # meting, en het enige dat echt breekt zijn de ByteTrack-ID's, waar de
                # offline doelkeuze juist op gebouwd is (tracklets worden aaneengeregen
                # op pakkleur en rijrichting). Wél in het logboek: als dit vaak gebeurt
                # zegt dat iets over de opname, en dan wil je het kunnen terugzien.
                if not _herstart_tracker(model):
                    raise
                herstarts += 1
                print(f"[tracker] frame {f}: {exc}; ByteTrack opnieuw gestart "
                      f"({herstarts}x in deze analyse)")
                res = _infereer(volg, waarschuwing_callback)[0]

            gen = getattr(model, 'generatie', 0) + herstarts
            if gen != vorige_gen:
                vorige_gen, id_basis = gen, hoogste_tid
            dets = []
            kps, boxes = res.keypoints, res.boxes
            if kps is not None and boxes is not None and kps.xy is not None and len(boxes) > 0:
                xy = kps.xy.cpu().numpy()                                # (N, 17, 2) pixels
                conf = (kps.conf.cpu().numpy() if kps.conf is not None
                        else np.ones(xy.shape[:2], dtype=float))
                xywh = boxes.xywh.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                for i in range(len(xy)):
                    cx, cy, bw, bh = xywh[i]
                    bbox_px = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                    hist, uit_masker = _torso_hist(frame, xy[i], conf[i], bbox_px)
                    dets.append(Detectie(
                        frame=f,
                        tid=(id_basis + int(ids[i])) if ids is not None else None,
                        centroid=(cx / w, cy / h),
                        bbox=(bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h),
                        area=(bw * bh) / (w * h),
                        lm=_coco_naar_landmarks(xy[i], conf[i], w, h),
                        hist=hist,
                        hist_masker=uit_masker,
                    ))
            for d in dets:
                if d.tid is not None and d.tid > hoogste_tid:
                    hoogste_tid = d.tid
            wacht.voed(f, dets)
            frames.append(dets)
        f += 1
        if progress_callback is not None:
            progress_callback(f, info.totaal)
    cap.release()
    return frames, buiten_meting


def _bouw_tracklets(frames):
    """Groepeer detecties per ByteTrack-ID tot tracklets (frame-gesorteerd)."""
    tracklets = {}
    for dets in frames:
        for d in dets:
            if d.tid is not None:
                tracklets.setdefault(d.tid, []).append(d)
    return list(tracklets.values())


def _splits_op_kleur(tracklet):
    """
    Knip een tracklet op de plekken waar de pakkleur aanhoudend verspringt — dat is
    vrijwel altijd ByteTrack die na een kruising/occlusie de andere schaatser aan
    hetzelfde ID hangt. Retourneert een lijst deel-tracklets.

    Alleen masker-histogrammen (`Detectie.ref_hist`) mogen knippen: een bbox-terugval
    bevat achtergrond en scoort daardoor stelselmatig laag tegen een masker-referentie
    — die zou anders een gat knippen waar niets aan de hand is. Zo'n frame telt dus
    niet mee vóór de knip, maar reset de teller ook niet: het geeft simpelweg geen
    oordeel.
    """
    stukken, huidig = [], []
    ref = KleurReferentie()
    laag = []                       # detecties sinds het eerste afwijkende frame
    n_laag = 0                      # daarvan: het aantal met een écht oordeel
    for d in tracklet:
        if d.ref_hist is None:
            (laag if laag else huidig).append(d)   # geen bruikbaar histogram: geen oordeel
            continue
        s = ref.sim(d.ref_hist)                    # None zolang de referentie leeg is
        if s is not None and s < KLEUR_SPLIT_MIN:
            laag.append(d)
            n_laag += 1
            if n_laag >= KLEUR_SPLIT_N:
                # Aanhoudend een ander pak: knip vóór het eerste afwijkende frame.
                if huidig:
                    stukken.append(huidig)
                huidig, ref, laag, n_laag = list(laag), KleurReferentie(), [], 0
                for d2 in huidig:
                    ref.voeg_toe(d2.ref_hist)
            continue
        huidig.extend(laag); laag, n_laag = [], 0   # korte dip (occlusie-mix): behouden
        huidig.append(d)
        ref.voeg_toe(d.ref_hist)
    huidig.extend(laag)
    if huidig:
        stukken.append(huidig)
    return stukken


def _pad_lengte(tracklet):
    """Totale afgelegde weg van de centroid (genormaliseerd)."""
    return float(sum(
        np.hypot(b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1])
        for a, b in zip(tracklet, tracklet[1:])))


def _afstand_tot_box(punt, bbox):
    """Kortste afstand van een punt tot een bounding box; 0 als het punt erin ligt."""
    x, y = punt
    return float(np.hypot(max(bbox[0] - x, 0.0, x - bbox[2]),
                          max(bbox[1] - y, 0.0, y - bbox[3])))


def _kies_seed(tracklets, frames, doel_punt, fps=25.0, doel_kader=None):
    """
    Kies het start-tracklet. Met muisklik in twee stappen; zonder klik: de grootste
    *beweger* — mediane oppervlakte × padlengte — zodat statische omstanders langs de
    boarding nooit gekozen worden.

    Met een getekend **kader** (`doel_kader`, genormaliseerd xyxy) gelden twee dingen
    extra. Een kandidaat mag hoogstens `KADER_MAAT_MAX`× zo hoog zijn als het kader —
    de gebruiker heeft de grootte van de schaatser aangegeven, dus een omstander van
    vier keer die maat naast de klik is niet wie bedoeld werd. En er is **geen terugval
    op de grootste beweger**: wie een kader tekent wijst één schaatser aan, en als de
    detectiepass die nergens in het zoekvenster ziet is het aan het kijkglas
    (`_Kijkglas`) om hem vanaf het kader te volgen, niet aan een gok.

    De klik staat op het eerste frame, maar de schaatser die je aanwijst hoeft daar nog
    niet gedetecteerd te zijn: ver weg en klein duurt het op echt materiaal secondenlang
    voor het model hem oppikt (gemeten: 3,4 s op `00005 8-41`). Daarom wordt er
    `KLIK_ZOEK_S` seconden lang doorgezocht op de plek waar geklikt is:

    1. **Op de klik.** Loop die frames chronologisch af en pak het eerste tracklet waarvan
       de bounding box het klikpunt bevat (dichtstbijzijnde centroid bij meerdere).
    2. **Naast de klik.** Nog niemand? Dan wint de detectie waarvan de bounding box het
       dichtst bij de klik ligt, binnen een poort die per seconde meegroeit — de schaatser
       is sinds de klik immers opgeschoven. Van de kandidaten wint de kleinste afstand,
       niet de afstand gedeeld door de poort: dat laatste beloont juist de late gok (zie
       de nearest-first-regel in `_stik_keten`).

    **Er wordt bewust niet met een constante snelheid teruggerekend** naar het klikframe,
    zoals `_stik_keten` over zijn gaten doet. Over een gat van een paar frames klopt dat
    model, maar over seconden niet: de romp zwaait met elke slag zijwaarts mee, en die
    slagbeweging is veel groter dan de netto verplaatsing. Nagemeten op `00005 8-41`
    (frontale opname, schaatser komt aanrijden): de x van de romp slingert tussen 0,33 en
    0,50 terwijl hij netto nauwelijks opschuift, en de snelheid uit vijf frames zet hem
    teruggerekend op x = −0,001 tot −1,563 — buiten beeld dus. De plek in beeld is hier
    het betrouwbare signaal, en de klik zelf de beste schatting daarvan.

    Retourneert `(tracklet_of_None, klik_gemist)`. `klik_gemist` is True als er wél
    geklikt is maar ook stap 2 niemand opleverde — dan is stilzwijgend de grootste
    beweger gekozen en dat hoort de gebruiker te weten (het kan de verkeerde schaatser
    zijn).
    """
    klik_gemist = False
    if doel_punt is None and doel_kader is not None:
        doel_punt = ((doel_kader[0] + doel_kader[2]) / 2, (doel_kader[1] + doel_kader[3]) / 2)
    if doel_punt is not None:
        dx, dy = doel_punt
        max_hoogte = (KADER_MAAT_MAX * (doel_kader[3] - doel_kader[1])
                      if doel_kader is not None else None)
        venster = min(int(round(max(fps, 1.0) * KLIK_ZOEK_S)), len(frames))
        per_frame = {}
        for t in tracklets:
            for d in t:
                if d.frame < venster and (max_hoogte is None
                                          or d.bbox[3] - d.bbox[1] <= max_hoogte):
                    per_frame.setdefault(d.frame, []).append((d, t))
        for f in range(venster):
            raak = [(d, t) for d, t in per_frame.get(f, ())
                    if d.bbox[0] - 0.03 <= dx <= d.bbox[2] + 0.03
                    and d.bbox[1] - 0.03 <= dy <= d.bbox[3] + 0.03]
            if raak:
                # Bij meerdere treffers eerst een tracklet dat lang genoeg is voor een
                # bruikbare kleurreferentie; is alles kort, dan telt de klik gewoon.
                lang = [dt for dt in raak if len(dt[1]) >= SEED_MIN_LEN]
                return min(lang or raak, key=lambda dt: (dt[0].centroid[0] - dx) ** 2
                                                        + (dt[0].centroid[1] - dy) ** 2)[1], False
        # Stap 2: niemand stond óp de klik — pak wie er het dichtst naast stond, binnen
        # een poort die meegroeit met de tijd die sinds de klik verstreken is.
        bij = []
        for f in range(venster):
            for d, t in per_frame.get(f, ()):
                afstand = _afstand_tot_box((dx, dy), d.bbox)
                if afstand <= KLIK_POORT_BASIS + KLIK_POORT_GROEI * f / max(fps, 1.0):
                    bij.append((afstand, f, t))
        if bij:
            lang = [k for k in bij if len(k[2]) >= SEED_MIN_LEN]
            return min(lang or bij, key=lambda k: (k[0], k[1]))[2], False
        # Ook zo niemand → val terug op de grootste beweger, maar meld het. Met een
        # kader niet: dan volgt het kijkglas de aangewezen schaatser vanaf het kader.
        klik_gemist = True
        if doel_kader is not None:
            return None, True
    bewegers = [t for t in tracklets if _pad_lengte(t) >= MIN_VERPLAATSING]
    kandidaten = bewegers or tracklets
    if not kandidaten:
        return None, klik_gemist
    return max(kandidaten, key=lambda t: float(np.median([d.area for d in t]))
                                         * max(_pad_lengte(t), 1e-6)), klik_gemist


def _snelheid(dets):
    """Gemiddelde centroid-verplaatsing per frame over de laatste detecties."""
    dets = dets[-SNELHEID_VENSTER:]
    if len(dets) < 2:
        return (0.0, 0.0)
    dt = dets[-1].frame - dets[0].frame
    if dt <= 0:
        return (0.0, 0.0)
    return ((dets[-1].centroid[0] - dets[0].centroid[0]) / dt,
            (dets[-1].centroid[1] - dets[0].centroid[1]) / dt)


def _stik_keten(seed, tracklets, fps):
    """
    Rijg tracklets aaneen tot één doel-keten, voor- en achterwaarts vanaf het seed-
    tracklet. Een kandidaat wordt alleen geaccepteerd als (a) zijn pakkleur bij de
    lopende referentie past (≥ KLEUR_MATCH_MIN) en (b) zijn aansluitpositie binnen de
    poort van de constante-snelheid-voorspelling over het gat ligt. Retourneert
    (keten-detecties gesorteerd op frame, KleurReferentie).

    Drie fijnere punten:
    - de kleur wordt gemeten aan de **kant van de kandidaat die aan de keten grenst**
      (begin bij vooruit stitchen, eind bij achteruit) — daar zijn belichting en schaal
      het best vergelijkbaar;
    - een kandidaat mag een paar frames met de keten **overlappen**
      rond een occlusie bestaan twee ID's een tijd naast elkaar, en met een strikte
      "moet ná het einde beginnen"-eis bleef dat gat voorgoed staan. Toegestaan zolang
      de kandidaat op de gedeelde frames vrijwel samenvalt met de keten
      (`STITCH_OVERLAP_GATE`); dubbele frames worden onderaan op kleur uitgedund;
    - een **korte seed** (fragment van 1–2 detecties, goed mogelijk ná `_splits_op_kleur`)
      geeft een kleurreferentie van één histogram; die wordt eerst op positie aangedikt
      (`_bootstrap`) vóór de kleur als poortwachter gaat dienen.
    """
    max_gap = int(round(STITCH_MAX_GAP_S * fps))
    keten = list(seed)
    ref = KleurReferentie()
    for d in keten:
        ref.voeg_toe(d.ref_hist)
    rest = [t for t in tracklets if t is not seed]

    def _kleur_sim(t, richting):
        """(similarity, zeker) van de ketenkant van kandidaat `t`. `zeker` is False als
        er alleen bbox-terugval-histogrammen zijn: die zijn niet met een masker-
        referentie vergelijkbaar, dus dan zegt het getal niets."""
        rand = t[:10] if richting > 0 else t[-10:]
        sims = [s for s in (ref.sim(d.ref_hist) for d in rand) if s is not None]
        if sims:
            return float(np.median(sims)), True
        sims = [s for s in (ref.sim(d.hist) for d in rand) if s is not None]
        return (float(np.median(sims)), False) if sims else (None, False)

    def _kandidaten(richting):
        """[(tracklet, gat, aansluitende detectie)] voor deze richting.

        Het aansluitpunt is de eerste detectie **voorbij de ketenrand**, niet blind
        `t[0]`/`t[-1]`. Dat verschil is wezenlijk. Vroeger moest een kandidaat vrijwel
        helemaal buiten de keten liggen (de inmiddels vervallen `STITCH_MAX_OVERLAP`), en een tracklet
        dat de rand overlápte viel daardoor in béide richtingen af: vooruit lag zijn
        eerste frame te ver terug, achteruit eindigde hij ná het begin van de keten.
        Zo'n tracklet kon dus nooit meedoen, ook al was het aantoonbaar dezelfde
        schaatser — en dat kost dekking: op één clip bleven de tracklets 23-38, 41-66,
        49-55 en 57-68 alle vier buiten de keten, samen een gat van 34 frames.

        Overlap zomaar toestaan mag niet: er zijn tracklets die de héle clip beslaan
        (stilstaande omstanders langs de boarding, `pad` ≈ 0,1). Daarom een **echte
        toets op de gedeelde frames**: valt de kandidaat samen met frames die de keten
        al heeft, dan moet hij daar op vrijwel dezelfde plek staan
        (`STITCH_OVERLAP_GATE`). Twee detecties in hetzelfde frame zijn per definitie
        twee verschillende boxen — liggen ze bovenop elkaar, dan is het dezelfde
        schaatser die rond een occlusie kort twee ID's kreeg; liggen ze uit elkaar, dan
        is het iemand anders en valt hij af. Dat is een scherpere toets dan welke
        voorspelling over een gat ook, want er komt geen extrapolatie aan te pas.
        """
        op_frame = {}
        for d in keten:
            op_frame.setdefault(d.frame, d)
        rand = keten[-1].frame if richting > 0 else keten[0].frame
        uit = []
        for t in rest:
            verlengt = t[-1].frame > rand if richting > 0 else t[0].frame < rand
            if not verlengt:
                continue
            gedeeld = [(d, op_frame[d.frame]) for d in t if d.frame in op_frame]
            if gedeeld:
                afw = float(np.median([np.hypot(a.centroid[0] - b.centroid[0],
                                                a.centroid[1] - b.centroid[1])
                                       for a, b in gedeeld]))
                if afw > STITCH_OVERLAP_GATE:
                    continue
            d0 = next(d for d in (t if richting > 0 else reversed(t))
                      if (d.frame > rand if richting > 0 else d.frame < rand))
            g = (d0.frame - rand) if richting > 0 else (rand - d0.frame)
            if g <= max_gap:
                uit.append((t, g, d0))
        return uit

    def _voorspel(richting, g):
        """Positie waar de keten na `g` frames verwacht wordt (constante snelheid)."""
        anker = keten[-1] if richting > 0 else keten[0]
        v = _snelheid(keten) if richting > 0 else _snelheid(keten[:SNELHEID_VENSTER])
        return (anker.centroid[0] + richting * v[0] * g,
                anker.centroid[1] + richting * v[1] * g)

    def _afstand(d0, richting, g):
        px, py = _voorspel(richting, g)
        return float(np.hypot(d0.centroid[0] - px, d0.centroid[1] - py))

    def _poort(g):
        return STITCH_GATE_BASIS + STITCH_GATE_GROEI * max(g, 0)

    def _opneem(t, richting):
        rest.remove(t)
        keten.extend(t)
        keten.sort(key=lambda d: d.frame)
        for d in (t if richting > 0 else reversed(t)):
            ref.voeg_toe(d.ref_hist)

    def _bootstrap():
        """Dik een te korte seed aan met direct aansluitende fragmenten, op positie —
        de kleurreferentie is hier immers nog te dun om iets mee te toetsen."""
        while len(keten) < SEED_MIN_LEN:
            keuze = None
            for richting in (+1, -1):
                for t, g, d0 in _kandidaten(richting):
                    if g > BOOTSTRAP_MAX_GAP:
                        continue
                    afst = _afstand(d0, richting, g)
                    if afst <= _poort(g) and (keuze is None or afst < keuze[0]):
                        keuze = (afst, t, richting)
            if keuze is None:
                return
            _opneem(keuze[1], keuze[2])

    def _probeer(richting):
        """richting=+1: aan het eind doorstikken; -1: vóór het begin.

        Kiest de **dichtstbijzijnde** bruikbare kandidaat, niet de best scorende. Dat is
        geen detail: de keten groeit met stapjes, en wie over een tussenliggend tracklet
        heen springt maakt dat tracklet onbereikbaar — het ligt daarna middenin de keten
        en verlengt hem dus in geen van beide richtingen meer. De oude score
        (`sim − 0,5·afst/poort`) deelde de afstand door een poort die mét het gat
        meegroeit en beloonde daarmee juist de grote sprong: op de testclip won
        66-182 (gat 35, kleur 0,72, score 0,701) van 41-66 (gat 10, kleur 0,71, score
        0,642), waarna 41-66 voorgoed buiten de keten viel en er een gat van 34 frames
        overbleef — te groot voor `GAP_VUL_S`, dus 34 frames zonder skelet.

        Kleur en afstand blijven poorten; bij een gelijk gat beslist de oude score.
        """
        while True:
            beste, beste_sleutel = None, None
            for t, g, d0 in _kandidaten(richting):
                sim, zeker = _kleur_sim(t, richting)
                if zeker and sim < KLEUR_MATCH_MIN:
                    continue
                afst = _afstand(d0, richting, g)
                # Zonder bruikbaar kleuroordeel telt alleen de positie, en dan willen we
                # de kandidaat echt dichtbij hebben.
                poort = _poort(g) * (1.0 if zeker else 0.5)
                if afst > poort:
                    continue
                score = (sim if zeker else KLEUR_MATCH_MIN) - 0.5 * afst / poort
                sleutel = (-g, score)          # klein gat eerst, dan pas de kwaliteit
                if beste_sleutel is None or sleutel > beste_sleutel:
                    beste, beste_sleutel = t, sleutel
            if beste is None:
                return
            _opneem(beste, richting)

    _bootstrap()
    _probeer(+1)
    _probeer(-1)
    _probeer(+1)          # na terugstikken kan er vooraan óf achteraan meer passen
    keten.sort(key=lambda d: d.frame)

    # Dubbele frames (overlappende tracklets): houd per frame de detectie die het best
    # bij de referentie past.
    per_frame = {}
    for d in keten:
        z = per_frame.get(d.frame)
        if z is None:
            per_frame[d.frame] = d
        else:
            sd, sz = ref.sim(d.ref_hist), ref.sim(z.ref_hist)
            if (sd or 0.0) > (sz or 0.0):
                per_frame[d.frame] = d
    return [per_frame[f] for f in sorted(per_frame)], ref


# ── Verfijningspass ─────────────────────────────────────────────────────────────
def _interpoleer_doel(doel_per_frame, fps, bocht=None):
    """
    Vul detectiegaten ≤ GAP_VUL_S met lineair geïnterpoleerde bboxes, zodat de
    verfijningspass daar tóch een schatting kan proberen. Retourneert
    {frame: (bbox_norm_xyxy, echt)} — `echt` False voor geïnterpoleerde plekken.

    `bocht` (per frame True/False) houdt de bochtstukken buiten die opvulling: daar zijn
    de gaten met opzet gemaakt door de detectiepass, en ze alsnog laten verfijnen zou de
    bespaarde tijd meteen weer opsouperen aan beeld waar toch niets te meten valt.
    """
    plan = {}
    frames = sorted(doel_per_frame)
    for f in frames:
        plan[f] = (doel_per_frame[f].bbox, True)
    max_gap = int(round(GAP_VUL_S * fps))
    for a, b in zip(frames, frames[1:]):
        g = b - a
        if 1 < g <= max_gap:
            ba = np.array(doel_per_frame[a].bbox)
            bb = np.array(doel_per_frame[b].bbox)
            for f in range(a + 1, b):
                if bocht is not None and f < len(bocht) and bocht[f]:
                    continue
                t = (f - a) / g
                plan[f] = (tuple(ba + t * (bb - ba)), False)
    return plan


def _bbox_uit_lm(lm, marge=KIJKGLAS_MARGE, min_vis=RTMPOSE_MIN_SCORE):
    """Genormaliseerde bbox (x0, y0, x1, y1) om de zichtbare landmarks, met een rand van
    `marge` × hoogte; None als er te weinig te zien is. Dit is de bbox die het kijkglas
    naar het volgende frame draagt."""
    pts = [(p.x, p.y) for p in lm if p.visibility >= min_vis]
    if len(pts) < 4:
        return None
    xs, ys = zip(*pts)
    hoogte = max(ys) - min(ys)
    if hoogte <= 0:
        return None
    rand = marge * hoogte
    return (max(0.0, min(xs) - rand), max(0.0, min(ys) - rand),
            min(1.0, max(xs) + rand), min(1.0, max(ys) + rand))


def _iou(a, b):
    """Intersection-over-union van twee (x0, y0, x1, y1)-boxen."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    snede = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    opp = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    unie = opp(a) + opp(b) - snede
    return snede / unie if unie > 0 else 0.0


def _schaal_bbox(bbox, factor):
    """Dezelfde bbox, `factor`× zo groot om z'n eigen midden."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    hw, hh = (bbox[2] - bbox[0]) / 2 * factor, (bbox[3] - bbox[1]) / 2 * factor
    return (cx - hw, cy - hh, cx + hw, cy + hh)


class _Kijkglas:
    """
    Volgt de doelschaatser door de frames waar de keten hem níet heeft, door de bbox van
    frame naar frame te propageren uit de eigen verfijnde keypoints. Ontstaan voor de
    schaatser die te klein is voor de detectiepass: die pikt hem pas op bij ~80–130 px
    hoogte (1080p), terwijl de verfijning (RTMPose top-down op een bbox, opgeschaald naar
    288×384) op zo'n schaatser prima werkt — het probleem is het vínden van de bbox, niet
    de keypoints. Het getekende kader van de gebruiker levert die bbox op het eerste
    frame; daarna levert elk verfijnd frame de bbox voor het volgende.

    Pure toestand, geen video en geen model: de verfijning komt binnen als een callable
    (`schat`), zodat dit met synthetische landmarks te toetsen is (`_zelftest_kijkglas`).

    **Runs en hun herkomst.** Een run begint óf op het kader (`bron='kader'`) óf op een
    ketenframe (`'keten'`; `anker` herstart de run op élk ketenframe, zodat de coast bij
    het volgende gat met een verse snelheid vertrekt). Treffers komen in `hangend` en
    worden pas definitief (`gevuld`) als de identiteit vaststaat:
    - raakt de run de keten (`anker` op een ketenframe), dan beslist de **koppeltoets**:
      IoU van de voorspelde bbox met de keten-bbox ≥ `KIJKGLAS_KOPPEL_IOU` → geaccepteerd,
      anders is de run naar iemand anders afgedwaald en gaat alles weg;
    - sterft de run (`KIJKGLAS_COAST_S` zonder treffer) of houdt de video op, dan telt de
      herkomst: een keten-run wordt geaccepteerd (de identiteit kwam van een geverifieerd
      ketenpunt), een kader-run alleen als de keten níet aantoonbaar dezelfde schaatser
      is (`keten_gekoppeld=False`: geen keten, óf een keten uit de grootste-beweger-
      terugval omdat de detectiepass in het kader niemand zag — dan zijn kader en keten
      twee losse gissingen over verschillende stukken van de clip, en is de ene niet
      beter dan de andere). Is de keten wél via het kader gevonden, dan is een kader-run
      die hem niet haalt verdacht: een handgetekend kader is geen bewezen identiteit, en
      zonder koppelpunt valt drift niet uit te sluiten.

    `kader_uitkomst` bewaart hoe de kader-run is afgelopen (`ok`, `reden`, `n`), zodat
    `analyseer` de gebruiker kan vertellen dat het kijkglas de schaatser is kwijtgeraakt
    — dat is het verschil tussen "0 afzetten" en weten waarom.

    **Plausibiliteit per stap** (`KIJKGLAS_SPRONG`/`KIJKGLAS_GROEI`): RTMPose levert
    altijd een skelet, ook als er iemand anders in de bbox is geschoven; springt het
    midden meer dan een halve lichaamshoogte of verandert de hoogte meer dan ×1,5 in één
    frame, dan is dat geen schaatser maar een verwisseling en telt het als misser.
    """

    def __init__(self, fps, keten_gekoppeld):
        fps = fps or 30.0
        self.max_gemist = max(1, int(round(KIJKGLAS_COAST_S * fps)))
        self.keten_gekoppeld = keten_gekoppeld
        self.kader_uitkomst = None  # {'ok', 'reden', 'n'} zodra de kader-run is afgesloten
        self.bbox = None            # laatst bekende bbox (genormaliseerd); None = geen run
        self.v = (0.0, 0.0)         # verplaatsing van het midden per frame
        self.laatste_f = None       # frame van de laatste treffer of het laatste anker
        self.gemist = 0             # opeenvolgende frames zonder treffer
        self.bron = None            # 'kader' | 'keten'
        self.eerste = False         # eerste stap van een kader-run: schaalveger
        self.hangend = {}           # frame → (lm, dev), wacht op de koppeltoets
        self.gevuld = {}            # frame → (lm, dev), definitief
        self.logboek = []           # regels voor het logboek
        self._treffers = deque(maxlen=SNELHEID_VENSTER)   # (frame, cx, cy)

    @property
    def leeft(self):
        return self.bbox is not None

    # ── toestand ───────────────────────────────────────────────────────────────
    @staticmethod
    def _midden(bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _zet(self, f, bbox):
        """Nieuwe bekende positie op frame `f`; werkt de snelheid bij."""
        cx, cy = self._midden(bbox)
        self._treffers.append((f, cx, cy))
        if len(self._treffers) >= 2:
            f0, x0, y0 = self._treffers[0]
            f1, x1, y1 = self._treffers[-1]
            if f1 > f0:
                self.v = ((x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0))
        self.bbox = tuple(bbox)
        self.laatste_f = f
        self.gemist = 0

    def _voorspel(self, f):
        """Waar de bbox op frame `f` verwacht wordt (constante snelheid sinds de laatste
        bekende positie)."""
        dt = f - self.laatste_f
        cx, cy = self._midden(self.bbox)
        cx, cy = cx + self.v[0] * dt, cy + self.v[1] * dt
        hw, hh = (self.bbox[2] - self.bbox[0]) / 2, (self.bbox[3] - self.bbox[1]) / 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def _sluit_run(self, ok, reden):
        n = len(self.hangend)
        if self.bron == 'kader':
            self.kader_uitkomst = {'ok': ok, 'reden': reden, 'n': n}
        if n:
            if ok:
                self.gevuld.update(self.hangend)
            eerste, laatste = min(self.hangend), max(self.hangend)
            self.logboek.append(
                f"[kijkglas] run vanaf {self.bron} frames {eerste}-{laatste} ({n} gevuld): "
                f"{'geaccepteerd' if ok else 'verworpen'} — {reden}")
        self.hangend = {}

    # ── gebeurtenissen ─────────────────────────────────────────────────────────
    def start_kader(self, bbox, f=0):
        """Begin de run op het getekende kader."""
        self._treffers.clear()
        self.v = (0.0, 0.0)
        self._zet(f, bbox)
        self.bron, self.eerste, self.hangend = 'kader', True, {}

    def anker(self, f, bbox):
        """De keten heeft de schaatser op frame `f`: koppeltoets voor een hangende run,
        daarna herstart vanaf dit geverifieerde punt."""
        if self.leeft and self.hangend:
            voorspeld = self._voorspel(f)
            iou = _iou(voorspeld, bbox)
            ok = iou >= KIJKGLAS_KOPPEL_IOU
            self._sluit_run(ok, f"koppeling op frame {f}: IoU {iou:.2f}")
            if not ok:
                self._treffers.clear()  # dat was iemand anders: zijn snelheid ook weg
        elif self.leeft:
            self.hangend = {}
        self._zet(f, bbox)
        self.bron, self.eerste = 'keten', False

    def volg(self, f, bbox):
        """Een frame dat pass 2 zelf al vulde (kort gat, geïnterpoleerd): alleen de
        positie meenemen, zodat de coast straks van de juiste plek vertrekt."""
        if self.leeft:
            self._zet(f, bbox)

    def stap(self, f, schat):
        """
        Eén frame zonder keten. `schat(bbox) → (lm, dev, score, hist)` is de verfijning
        op die bbox (lm None = afgewezen door score- of kleurpoort). Retourneert de
        geaccepteerde `(lm, dev, hist)`, of None.
        """
        if not self.leeft:
            return None
        verwacht = self._voorspel(f)
        # Eerste stap van een kader-run: een hand tekent zelden strak, dus drie
        # schalingen proberen en de zekerste nemen.
        kandidaten = ([_schaal_bbox(verwacht, k) for k in KIJKGLAS_START_SCHALEN]
                      if self.eerste else [verwacht])
        beste = None
        for bb in kandidaten:
            uit = schat(bb)
            if uit is None or uit[0] is None:
                continue
            lm, dev, score, hist = uit
            nieuw = _bbox_uit_lm(lm)
            if nieuw is None:
                continue
            if not self.eerste and not self._plausibel(verwacht, nieuw):
                continue
            if beste is None or score > beste[0]:
                beste = (score, lm, dev, hist, nieuw)
        if beste is None:
            self.gemist += 1
            if self.gemist > self.max_gemist:
                self._sterf(f)
            return None
        _, lm, dev, hist, nieuw = beste
        self.eerste = False
        self._zet(f, nieuw)
        self.hangend[f] = (lm, dev)
        return lm, dev, hist

    def _plausibel(self, verwacht, nieuw):
        h_v = verwacht[3] - verwacht[1]
        h_n = nieuw[3] - nieuw[1]
        if h_v <= 0 or h_n <= 0:
            return False
        (vx, vy), (nx, ny) = self._midden(verwacht), self._midden(nieuw)
        if float(np.hypot(nx - vx, ny - vy)) > KIJKGLAS_SPRONG * h_v:
            return False
        groei = h_n / h_v
        return 1.0 / KIJKGLAS_GROEI <= groei <= KIJKGLAS_GROEI

    def _sterf(self, f):
        ok = self.bron == 'keten' or not self.keten_gekoppeld
        self._sluit_run(ok, f"run gestopt op frame {f} na {self.gemist} frames zonder "
                            f"treffer" + ("" if ok else "; kader-run zonder koppeling"))
        self.bbox = None
        self._treffers.clear()          # een later anker begint met een verse snelheid

    def afsluiten(self):
        """Einde van de video: wat er nog hangt afhandelen op herkomst."""
        if self.leeft:
            ok = self.bron == 'keten' or not self.keten_gekoppeld
            self._sluit_run(ok, "einde video" + ("" if ok else "; kader-run zonder koppeling"))
            self.bbox = None


def _rtmpose_model():
    r"""Het meegeleverde RTMPose-bestand als dat er staat, anders de URL — waarna rtmlib
    het zelf downloadt en cachet in %USERPROFILE%\.cache\rtmlib."""
    pad = os.path.join(app_dir(), RTMPOSE_LOKAAL)
    return pad if os.path.exists(pad) else RTMPOSE_MODEL


def _maak_rtmpose(waarschuwing_callback=None):
    """
    RTMPose-26-model voor de verfijningspass, of None zonder rtmlib.

    De CPU-terugval is hier geen luxe: `rtmpose_device()` leest af of ONNXRuntime een
    CUDA- of DirectML-provider heeft **meegecompileerd**, wat iets anders is dan of de
    bijbehorende DLL's op deze machine ook echt laden. Blijkt dat laatste niet zo, dan
    faalt pas het opbouwen van de sessie — en dat mag geen analyse kosten die verder
    prima op de CPU had gekund.
    """
    if not IS_RTMPOSE:
        return None
    device = rtmpose_device()
    model = _rtmpose_model()     # één keer bepalen: beide takken hetzelfde gewichtenbestand
    try:
        return _RTMPose(model, model_input_size=RTMPOSE_INPUT,
                        backend='onnxruntime', device=device)
    except Exception as exc:
        if device == 'cpu':
            raise
        _meld(waarschuwing_callback,
              f"RTMPose kon niet op de GPU starten ({exc}); de verfijningspass "
              "draait op de CPU.")
        return _RTMPose(model, model_input_size=RTMPOSE_INPUT,
                        backend='onnxruntime', device='cpu')


def _verfijn_landmarks(input_pad, model, info, doel_per_frame, ref,
                       progress_callback=None, rtmpose=None, bocht=None,
                       deinterlacen=False, kijkglas=None):
    """
    Pass 2: lees de video opnieuw en schat per doel-frame de pose opnieuw, nu met de
    schaatser beeldvullend in het inferentiebeeld → aanzienlijk nauwkeurigere
    keypoints dan in de volledige-frame-pass. Met `rtmpose` gaat dat top-down op de
    doel-bbox (RTMPose-26: subpixel-decodering + echte hiel/teen); anders via een
    vierkante crop door het YOLO-model. De pakkleur (referentie `ref`) bewaakt in
    beide routes dat nooit stilletjes een andere persoon wordt overgenomen.
    Retourneert `(uit, devs, gevuld)`: {frame: lm} met verfijnde (of herstelde)
    landmarks, {frame: middellijn-dev}, en de frames die het kijkglas erbij heeft gezet.

    `bocht` (per frame True/False) houdt de gat-opvulling weg uit de bochtstukken; zie
    `_interpoleer_doel`.

    `kijkglas` (een `_Kijkglas`, al gestart op het getekende kader) vult de frames die
    noch de keten noch de interpolatie dekt: de aanloop vóór de eerste detectie, gaten
    langer dan `GAP_VUL_S`, de uitloop, en de stukken die de bochtwacht heeft
    overgeslagen. Het negeert `bocht` bewust: de propagatie is goedkoop, en of een
    gevuld frame bocht is beslist daarna de ratio (`bepaal_bocht_reeks`). Op
    ketenframes wordt het kijkglas geankerd (koppeltoets + verse start), op
    geïnterpoleerde frames alleen bijgewerkt; zie `_Kijkglas`.
    """
    plan = _interpoleer_doel(doel_per_frame, info.fps, bocht)
    w, h = info.w, info.h
    uit, devs = {}, {}

    def verfijn(frame, bbox, poort):
        """→ (lm, dev, score, hist); lm None als de schatting is afgewezen."""
        if rtmpose is not None:
            return _verfijn_rtmpose(rtmpose, frame, bbox, poort, ref, w, h)
        lm = _verfijn_yolo_crop(model, frame, bbox, ref, w, h)
        return lm, None, (1.0 if lm is not None else 0.0), None

    cap = open_video(input_pad, deinterlacen)
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f in plan:
            bbox, echt = plan[f]
            lm, dev, _, _ = verfijn(frame, bbox, 'veto' if echt else 'match')
            if dev is not None:
                devs[f] = dev
            if lm is not None:
                uit[f] = lm
            if kijkglas is not None:
                # De keten-bbox als de verfijning afviel: ook dan is dit een geverifieerd
                # punt (pass-1-detectie) en hoort de coast van hier te vertrekken.
                bekend = (_bbox_uit_lm(lm) if lm is not None else None) or bbox
                if echt:
                    kijkglas.anker(f, bekend)
                elif lm is not None:
                    kijkglas.volg(f, bekend)
        elif kijkglas is not None and kijkglas.leeft:
            treffer = kijkglas.stap(f, lambda bb, _fr=frame: verfijn(_fr, bb, 'veto'))
            if treffer is not None and not doel_per_frame:
                # Kijkglas-only (geen keten): de referentie moet uit de eigen treffers
                # komen, anders heeft de kleurpoort nooit iets om tegen te toetsen.
                ref.voeg_toe(treffer[2])
        f += 1
        if progress_callback is not None:
            progress_callback(f, info.totaal)
    cap.release()

    gevuld = set()
    if kijkglas is not None:
        kijkglas.afsluiten()
        for f2, (lm, dev) in kijkglas.gevuld.items():
            uit[f2] = lm
            if dev is not None:
                devs[f2] = dev
        gevuld = set(kijkglas.gevuld)
        for regel in kijkglas.logboek:
            print(regel)
        print(f"[kijkglas] {len(gevuld)} frames erbij gevuld")
    return uit, devs, gevuld


# Halpe26-indices van de been-keypoints (heupen t/m enkels) voor de kwaliteitscheck.
_HALPE_BENEN = (11, 12, 13, 14, 15, 16)

# ── Middellijn-kwaliteitsvlag ───────────────────────────────────────────────────
# Frontaal gefilmd hoort een gewricht horizontaal in het midden van het been te
# liggen. We meten per knie de afwijking t.o.v. de middellijn van het been in een
# kleurmasker. De kleur komt uit een **dij-zelfsample van hetzelfde been in
# hetzelfde frame** (niet uit de torso-referentie: een pak is geregeld tweekleurig
# — witte torso, zwarte broek — maar dij en knie zijn altijd dezelfde stof, met
# dezelfde belichting). Puur een kwaliteitsvlag: grote afwijking = wankel frame.
# Bewust géén automatische correctie zolang niet gemeten is dat die de hoeken
# verbetert.
MIDDELLIJN_ROIJEN     = 5      # aantal beeldrijen rond de knie-y waarover gemiddeld wordt
MIDDELLIJN_BP_DREMPEL = 0.2    # maskerdrempel als fractie van het back-projection-maximum
MIDDELLIJN_RUN_MIN    = 0.08   # min. runbreedte als fractie van de tibialengte (ruis)
MIDDELLIJN_RUN_MAX    = 0.8    # max. runbreedte — breder = benen/arm samengesmolten: overslaan
MIDDELLIJN_MIN_RIJEN  = 0.6    # min. fractie bruikbare rijen voor een geldige meting


def _been_hist(frame, heup_xy, knie_xy, tibia_len):
    """HSV-histogram (0–255-genormaliseerd, voor back-projection) van een blokje
    midden op het bovenbeen — de kleur van de broekspijp van dít been."""
    mid_x = (float(heup_xy[0]) + float(knie_xy[0])) / 2
    mid_y = (float(heup_xy[1]) + float(knie_xy[1])) / 2
    half = int(np.clip(0.12 * tibia_len, 4, 30))
    x0, y0, x1, y1 = int(mid_x - half), int(mid_y - half), int(mid_x + half), int(mid_y + half)
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0] or x1 - x0 < 4:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(KLEUR_BINS),
                        [0, 180, 0, 256, 0, 256])
    # 0–255 i.p.v. L1: calcBackProject geeft uint8 terug — met een L1-histogram
    # (waarden ≪ 1) trunceert álles naar 0 en is het masker altijd leeg.
    return cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)


def _middellijn_afwijking(frame, been_hist, knie_xy, tibia_len):
    """
    Horizontale afwijking (px, getekend: middellijn − keypoint) van één kniepunt
    t.o.v. het midden van het been in het kleurmasker, of None als de meting
    niet lukt (been niet vrijstaand, kleur onduidelijk, beeldrand).
    """
    if been_hist is None or tibia_len < 12:
        return None
    h, w = frame.shape[:2]
    kx, ky = float(knie_xy[0]), float(knie_xy[1])
    half_b = int(np.clip(0.7 * tibia_len, 12, 90))
    half_r = MIDDELLIJN_ROIJEN // 2
    x0, x1 = int(kx - half_b), int(kx + half_b + 1)
    y0, y1 = int(ky - half_r), int(ky + half_r + 1)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    bp = cv2.calcBackProject([hsv], [0, 1, 2], been_hist,
                             [0, 180, 0, 256, 0, 256], scale=1)
    piek = float(bp.max())
    if piek <= 0:
        return None
    masker = bp >= MIDDELLIJN_BP_DREMPEL * piek

    centra = []
    kx_lokaal = kx - x0
    for rij in masker:
        # De aaneengesloten run van pak-pixels die het kniepunt bevat.
        idx = np.flatnonzero(rij)
        if len(idx) == 0:
            continue
        # runs = groepen opeenvolgende indices
        splitsingen = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splitsingen + 1)
        run = next((rn for rn in runs if rn[0] - 2 <= kx_lokaal <= rn[-1] + 2), None)
        if run is None:
            continue
        breedte = run[-1] - run[0] + 1
        if not (MIDDELLIJN_RUN_MIN * tibia_len <= breedte <= MIDDELLIJN_RUN_MAX * tibia_len):
            continue
        centra.append((run[0] + run[-1]) / 2.0)
    if len(centra) < MIDDELLIJN_MIN_RIJEN * masker.shape[0]:
        return None
    return round(float(np.median(centra) - kx_lokaal), 1)


def _verfijn_rtmpose(rtmpose, frame, bbox, poort, ref, w, h):
    """
    Top-down verfijning van één frame: RTMPose-26 op de doel-bbox (pixels).
    De kleurpoort houdt de andere schaatser buiten, in twee standen:
    - `'veto'` — alleen een dúidelijk ander pak (< `KLEUR_SPLIT_MIN`) wordt geweigerd.
      Voor een echte detectie (pass-1-vangnet: de landmarks blijven dan staan) én voor
      het kijkglas, waar continuïteit, scorepoort, plausibiliteit en koppeltoets het
      vangnet zijn en de torso van een schaatser van 90 px (~15×25 px) te ruizig is
      voor een positieve drempel;
    - `'match'` — een geïnterpoleerd gat-frame eist juist een positieve kleurmatch
      (≥ `KLEUR_MATCH_MIN`), want daar is geen enkel ander vangnet.
    Retourneert `(lm | None, middellijn-dev-dict | None, been-score, masker-hist | None)`.
    """
    bbox_px = (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)
    kps, scores = rtmpose(frame, [list(bbox_px)])
    kp, sc = kps[0], scores[0]

    score = float(sc[list(_HALPE_BENEN)].mean())
    if score < RTMPOSE_MIN_SCORE:
        return None, None, score, None     # benen niet gezien (occlusie): niet vertrouwen
    hist, uit_masker = _torso_hist(frame, kp, sc, bbox_px)
    sim = ref.sim(hist)
    if poort == 'veto':
        # Alleen een masker-histogram mag een verfijning afwijzen: de bbox-terugval
        # bevat achtergrond en scoort ook bij de júiste schaatser laag.
        if uit_masker and sim is not None and sim < KLEUR_SPLIT_MIN:
            return None, None, score, None     # duidelijk een ander pak in de bbox
    else:
        if sim is None or sim < KLEUR_MATCH_MIN:
            return None, None, score, None     # gat-frame: alleen vullen bij zékere match

    # Kwaliteitsvlag: knie t.o.v. de middellijn van het been (Halpe: 11/13/15 =
    # L heup/knie/enkel, 12/14/16 = R). Alleen meten, niet corrigeren.
    dev = {}
    for naam, h_i, k_i, e_i in (('l_knie', 11, 13, 15), ('r_knie', 12, 14, 16)):
        dev[naam] = None
        if min(sc[h_i], sc[k_i], sc[e_i]) >= RTMPOSE_MIN_SCORE:
            tibia = float(np.hypot(*(kp[k_i] - kp[e_i])))
            been_hist = _been_hist(frame, kp[h_i], kp[k_i], tibia)
            dev[naam] = _middellijn_afwijking(frame, been_hist, kp[k_i], tibia)
    if dev['l_knie'] is None and dev['r_knie'] is None:
        dev = None
    return _halpe26_naar_landmarks(kp, sc, w, h), dev, score, (hist if uit_masker else None)


def _verfijn_yolo_crop(model, frame, bbox, ref, w, h):
    """Terugvalroute zonder rtmlib: vierkante crop rond de bbox door het YOLO-model."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zijde = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    kant = int(max(VERFIJN_MIN_PX, VERFIJN_MARGE * zijde * max(w, h)))
    kant = min(kant, min(w, h))
    x0 = int(np.clip(cx * w - kant / 2, 0, w - kant))
    y0 = int(np.clip(cy * h - kant / 2, 0, h - kant))
    crop = frame[y0:y0 + kant, x0:x0 + kant]
    res = _infereer(lambda dev: model.predict(
        crop, imgsz=VERFIJN_IMGSZ, classes=[0], verbose=False, device=dev))[0]
    keuze = _kies_in_crop(res, crop, (cx * w - x0, cy * h - y0), kant, ref)
    if keuze is None:
        return None
    kp_xy, kp_conf = keuze
    return _coco_naar_landmarks(kp_xy + [x0, y0], kp_conf, w, h)


def _kies_in_crop(res, crop, verwacht_xy, kant, ref):
    """
    Kies in het crop-resultaat de persoon die het doel is: pakkleur moet bij de
    referentie passen; bij meerdere matches wint de dichtstbijzijnde bij de verwachte
    positie. Retourneert (kp_xy, kp_conf) in cropcoördinaten, of None.
    """
    kps, boxes = res.keypoints, res.boxes
    if kps is None or boxes is None or kps.xy is None or len(boxes) == 0:
        return None
    xy = kps.xy.cpu().numpy()
    conf = (kps.conf.cpu().numpy() if kps.conf is not None
            else np.ones(xy.shape[:2], dtype=float))
    xywh = boxes.xywh.cpu().numpy()

    kandidaten = []
    for i in range(len(xy)):
        cx, cy, bw, bh = xywh[i]
        hist, uit_masker = _torso_hist(crop, xy[i], conf[i],
                                       (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        # Zonder masker is de similarity niet met de referentie vergelijkbaar; dan geldt
        # "onbekend" (None) i.p.v. een kunstmatig lage score.
        sim = ref.sim(hist) if uit_masker else None
        afst = float(np.hypot(cx - verwacht_xy[0], cy - verwacht_xy[1])) / kant
        kandidaten.append((i, sim, afst))

    passend = [k for k in kandidaten if k[1] is not None and k[1] >= KLEUR_MATCH_MIN]
    if passend:
        i = min(passend, key=lambda k: k[2])[0]
        return xy[i], conf[i]
    # Geen kleurmatch (bv. torso deels buiten beeld): accepteer alleen een kandidaat
    # die vrijwel exact op de verwachte plek staat — anders liever geen verfijning.
    dichtbij = [k for k in kandidaten if k[2] <= 0.12 and (k[1] is None or k[1] >= KLEUR_SPLIT_MIN)]
    if dichtbij:
        i = min(dichtbij, key=lambda k: k[2])[0]
        return xy[i], conf[i]
    return None


def _bocht_met_controleframes(resultaten, buiten_meting, info):
    """
    Classificeer de bocht (`bepaal_bocht_reeks`) en houd daarna vast dat een frame dat de
    detectiepass niet echt geanalyseerd heeft ook nooit een meting oplevert.

    Dat laatste gaat over de **controleframes**: in de overslaan-stand infereert de wacht
    elke `BOCHT_CHECK_S` één frame om te zien of het rechte stuk alweer begonnen is. Dat
    frame krijgt dus een skelet, maar het is een kijkje en geen meting — de buurframes
    die samen een afzet zouden moeten aantonen zijn juist overgeslagen. Zijn óórdeel telt
    wél mee (het mag de bocht beëindigen, daar is het voor); alleen zijn eigen hoek niet.

    Zonder deze regel zou zo'n frame zichzelf op z'n eigen heupstand kunnen vrijpleiten,
    en midden in een bocht draait een schaatser af en toe kort bijna frontaal (zichtbaar
    in de Ellia- en Fran-clips). Beëindigt een controleframe de bocht echt, dan draait de
    detectiepass daarná weer op vol tempo en worden díe frames gewoon gemeten — er gaat
    dan hooguit dit ene frame aan het begin van de eerstvolgende stand-run verloren, en
    een aan het begin afgekapte run telt sowieso mee (zie `bepaal_afzet_uit_strek`).
    """
    bepaal_bocht_reeks(resultaten, info.w, info.h, info.fps)
    for r, buiten in zip(resultaten, buiten_meting):
        if buiten:
            r.bocht = True


# ── Hoofd-API ───────────────────────────────────────────────────────────────────
def analyseer(input_pad, model_pad=None, smooth_n=5, threshold=0.015, force_fps=None,
              num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
              progress_callback=None, yolo_model=None, horizon_deg=0.0,
              auto_horizon=False, verfijn=True, perspectief=None,
              waarschuwing_callback=None, bocht=True, deinterlacen=None,
              doel_kader=None, bocht_data=False):
    """
    Volledige analyse via YOLO-pose + ByteTrack + offline doelkeuze + crop-verfijning.
    Signatuur-compatibel met schaats_analyse.analyseer() (`model_pad` — het MediaPipe
    .task — en `num_poses` worden genegeerd; YOLO detecteert altijd alle personen).
    Retourneert (VideoInfo, lijst[FrameResultaat]).

    `doel_kader` (genormaliseerd (x0, y0, x1, y1) om de schaatser op het eerste frame)
    zet het **kijkglas** aan: de schaatser wordt ook gevolgd waar de detectiepass hem
    niet ziet — te klein, ver weg — door de bbox uit het kader van frame naar frame te
    propageren in de verfijningspass (zie `_Kijkglas`). Het middelpunt dient als
    `doel_punt` als dat niet apart gegeven is. Zonder kader is het gedrag byte-voor-byte
    als voorheen: het kijkglas is bewust opt-in, omdat het ook gaten en overgeslagen
    bochtstukken van elke analyse zou vullen en dat een meetwijziging is.

    `perspectief` (PerspectiefConfig) werkt identiek aan de MediaPipe-backend: de
    gedeelde stappen `zet_horizon`/`verwerk_afgeleiden` doen al het werk.

    `waarschuwing_callback(tekst)` krijgt meldingen over stille terugvallen in de
    doelkeuze (nu: een muisklik die niemand raakte). Lukt de doelkeuze helemaal niet,
    dan is dat geen waarschuwing maar een fout — dan volgt een RuntimeError i.p.v. een
    lege analyse die er als een geldig resultaat uitziet.

    Met `bocht` (default) slaat de detectiepass de bocht grotendeels over (`_BochtWacht`)
    en leveren bochtframes geen afzetmeting. Dat is hier vooral een snelheidsmaatregel:
    de detectiepass is het leeuwendeel van de analysetijd en in de bocht valt er niets te
    meten. Met `bocht=False` wordt elk frame geïnfereerd en gemeten, zoals voorheen.
    Met `bocht_data=True` wordt elk frame geïnfereerd en verfijnd (dus óók in de
    bocht), maar krijgt elk bochtframe zijn `bocht`-vlag: landmarks worden gewoon
    opgeslagen (datacollectie voor een toekomstige bochtmeting), terwijl
    `segmenteer_afzetten` ze weigert — er komen dus géén bochtafzetten uit.
    """
    info = video_info(input_pad, force_fps)
    # None = zelf uitzoeken (CLI-gemak); de GUI bepaalt het in de dialoog en geeft een
    # expliciete bool, zodat de keuze zichtbaar is en in de instellingen belandt.
    # Béíde passes moeten dezelfde pixels zien: verfijnt pass 2 op geweven beeld terwijl
    # pass 1 gefilterd is, dan meet je twee verschillende video's door elkaar.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    model = _laad_yolo(yolo_model or os.path.join(app_dir(), STANDAARD_YOLO_MODEL),
                       waarschuwing_callback)
    if perspectief is not None:
        auto_horizon = False     # vaste camera per aanname; kalibratie kent de kanteling al
    if doel_kader is not None:
        doel_kader = tuple(float(v) for v in doel_kader)
        if doel_punt is None:
            doel_punt = ((doel_kader[0] + doel_kader[2]) / 2,
                         (doel_kader[1] + doel_kader[3]) / 2)
        kader_px = (doel_kader[3] - doel_kader[1]) * info.h
        if kader_px < KADER_MIN_HOOGTE_PX:
            # Geen blokkade — misschien groeit hij snel genoeg — maar wel zeggen waarom
            # er straks (bijna) niets uitkomt, want dat is anders niet te zien.
            _meld(waarschuwing_callback,
                  f"Het getekende kader is maar {kader_px:.0f} px hoog. Onder ongeveer "
                  f"{KADER_MIN_HOOGTE_PX} px ziet ook de verfijning geen benen meer (de "
                  f"benen zijn dan zo'n 15 px), dus daar valt niets te meten — ook niet met "
                  f"het kijkglas. Begin het fragment later, op het moment dat de schaatser "
                  f"groter in beeld staat.")

    # Meerdere passes → één doorlopende voortgangsbalk via fase-schijven.
    n_fasen = 1 + (1 if verfijn else 0) + (1 if auto_horizon else 0)
    fase = 0
    det_cb = fase_voortgang(progress_callback, fase, n_fasen); fase += 1
    ver_cb = fase_voortgang(progress_callback, fase, n_fasen) if verfijn else None
    fase += 1 if verfijn else 0
    hor_cb = fase_voortgang(progress_callback, fase, n_fasen) if auto_horizon else None

    # Pass 1: alle detecties verzamelen (hoge resolutie). In de bocht slaat de wacht
    # frames over en zijn de frames die hij nog wél infereert enkel controleframes;
    # allebei staan ze in `buiten_meting` en gaan zo de rest van de pijplijn in.
    frames, buiten_meting = _detecteer_alles(input_pad, model, info,
                                             progress_callback=det_cb,
                                             bocht=(False if bocht_data else bocht),
                                             waarschuwing_callback=waarschuwing_callback,
                                             deinterlacen=deinterlacen)
    n_frames = len(frames)

    # Offline doelkeuze: tracklets → kleur-splits → seed → stitching.
    tracklets = []
    for t in _bouw_tracklets(frames):
        tracklets.extend(_splits_op_kleur(t))
    seed, klik_gemist = _kies_seed(tracklets, frames, doel_punt, info.fps, doel_kader)
    terugval = False
    if seed is None and doel_kader is not None:
        # De detectiepass zag in het kader niemand. Dan tóch de grootste beweger volgen
        # (precies wat een klik in dat geval doet), náást het kijkglas vanaf het kader:
        # het kader mag nooit slechter uitpakken dan een klik. Op `00000 16-14` (schaatser
        # 35–57 px, pas op frame 187 van 200 gedetecteerd) leverde kijkglas-only één
        # frame op, terwijl de terugval de dertien echte detecties aan het eind pakt.
        seed, _ = _kies_seed(tracklets, frames, None, info.fps)
        terugval = seed is not None
    if seed is None and doel_kader is None:
        # Zonder seed blijft doel_per_frame leeg en loopt de rest van de pijplijn
        # gewoon door: geen enkel frame krijgt een pose en de GUI meldt "0 afzetten"
        # alsof dat een meting is. Liever hard falen met een begrijpelijke reden.
        raise RuntimeError(
            "Geen schaatser gevonden om te volgen: de detectie leverde in deze video "
            "geen enkele persoon op. Controleer of de schaatser in beeld is en of de "
            "video leesbaar is (codec/resolutie), en probeer eventueel een andere clip.")
    if klik_gemist and waarschuwing_callback is not None:
        if doel_kader is not None:
            vervolg = (f"Het kijkglas volgt de schaatser vanaf het kader; daarnaast is de "
                       f"grootste beweger in beeld gevolgd (vanaf frame {seed[0].frame}) — "
                       f"controleer of dat dezelfde schaatser is."
                       if terugval else
                       "De schaatser is nu alleen met het kijkglas vanaf het kader gevolgd; "
                       "controleer het resultaat.")
            waarschuwing_callback(
                f"Het getekende kader kon niet aan een detectie gekoppeld worden — op die "
                f"plek heeft de detectiepass in de eerste {KLIK_ZOEK_S:.0f} seconden "
                f"niemand van die maat gezien. " + vervolg)
        else:
            waarschuwing_callback(
                f"Je klik kon niet aan een schaatser gekoppeld worden — op die plek is in "
                f"de eerste {KLIK_ZOEK_S:.0f} seconden niemand gedetecteerd, ook niet vlak "
                f"ernaast. Er is nu de grootste beweger in beeld gevolgd; controleer of dat "
                f"de bedoelde schaatser is.")

    if seed is not None:
        keten, ref = _stik_keten(seed, tracklets, info.fps)
    else:
        keten, ref = [], KleurReferentie()      # kijkglas-only: alles komt van het kader
    doel_per_frame = {d.frame: d for d in keten}
    kijkglas = None
    if doel_kader is not None and verfijn:
        # De keten is alleen "gekoppeld" als hij via het kader zelf gevonden is; een
        # terugval-keten is een tweede gissing en mag een kader-run niet afkeuren.
        kijkglas = _Kijkglas(info.fps, keten_gekoppeld=bool(doel_per_frame) and not terugval)
        kijkglas.start_kader(doel_kader, 0)

    resultaten = []
    for f in range(n_frames):
        r = FrameResultaat(frame_nr=f, tijd=f / info.fps if info.fps > 0 else 0)
        r.bocht = bool(buiten_meting[f])
        if f in doel_per_frame:
            r.lm = doel_per_frame[f].lm
            r.pose_gevonden = True
        resultaten.append(r)

    # Bocht bepalen op de rúwe pass-1-landmarks, vóór de verfijning: dan hoeft die
    # verfijning niet meer over de bocht. De controleframes die de detectiepass in de
    # bocht wél heeft geïnfereerd geven daar het oordeel; zeggen die dat het rechte stuk
    # alweer bezig is, dan draait de detectiepass daarná weer op vol tempo en worden
    # díe frames wél gewoon gemeten.
    if bocht or bocht_data:
        _bocht_met_controleframes(resultaten, buiten_meting, info)
        # bocht_data: verfijn ook de bochtframes, dus geen skip-lijst doorgeven.
        bocht_per_frame = None if bocht_data else [r.bocht for r in resultaten]
    else:
        bocht_per_frame = None

    # Pass 2: verfijning (nauwkeurigere keypoints + gaten vullen), top-down met
    # RTMPose-26 als rtmlib beschikbaar is, anders de oude YOLO-crop-route. Mét kader
    # loopt het kijkglas hier mee en vult het de frames die de keten niet heeft.
    if verfijn and (doel_per_frame or kijkglas is not None):
        verfijnd, devs, gevuld = _verfijn_landmarks(
            input_pad, model, info, doel_per_frame, ref, progress_callback=ver_cb,
            rtmpose=_maak_rtmpose(waarschuwing_callback), bocht=bocht_per_frame,
            deinterlacen=deinterlacen, kijkglas=kijkglas)
    else:
        verfijnd, devs, gevuld = {}, {}, set()
    if kijkglas is not None and not doel_per_frame and not gevuld:
        raise RuntimeError(
            "Geen schaatser gevonden om te volgen: de detectie zag in het getekende "
            "kader niemand, en ook het kijkglas kon er vanaf het kader geen schaatser in "
            "volgen. Teken het kader strak om de schaatser op het eerste frame (inzoomen "
            "met het muiswiel) en controleer of hij daar echt in beeld staat.")
    uitkomst = kijkglas.kader_uitkomst if kijkglas is not None else None
    if uitkomst is not None and 'koppeling' not in uitkomst['reden']:
        # De kader-run heeft de keten niet gehaald: gestorven of einde video. Dat kan
        # gewoon goed zijn (kijkglas-only tot het eind), maar meestal betekent het dat de
        # schaatser te klein of te onduidelijk was — en dat hoort de gebruiker te horen
        # in plaats van het uit "0 afzetten" te moeten raden.
        _meld(waarschuwing_callback,
              f"Het kijkglas heeft de schaatser vanaf het kader {uitkomst['n']} frame(s) "
              f"kunnen volgen ({uitkomst['reden']}). Raakt het hem zo snel kwijt, dan is "
              f"hij te klein of te onduidelijk in beeld voor de verfijning.")
    # Een frame dat het kijkglas gevuld heeft ís geanalyseerd, ook als de bochtwacht het
    # in pass 1 had overgeslagen: de tweede bochtclassificatie hieronder beoordeelt het
    # dan op zijn eigen heupstand, zoals elk ander frame met een skelet.
    for f in gevuld:
        if f < len(buiten_meting):
            buiten_meting[f] = False

    for f, r in enumerate(resultaten):
        lm = verfijnd.get(f)
        if lm is not None:
            r.lm = lm
            r.pose_gevonden = True
            r.middellijn_dev = devs.get(f)

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)
    if bocht or bocht_data:
        # Nog eens, nu op de verfijnde landmarks: de frames die pass 2 erbij heeft
        # gevonden krijgen zo alsnog hun eigen oordeel.
        _bocht_met_controleframes(resultaten, buiten_meting, info)
    zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief, deinterlacen=deinterlacen)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten


# ── Zelftest ────────────────────────────────────────────────────────────────────
def _zelftest_bochtwacht():
    """
    Toetst de toestandsmachine van `_BochtWacht` op synthetische detecties — geen video,
    geen model, dus in een seconde te draaien. Dit is het stuk waar een fout stil is: te
    weinig overslaan kost alleen tijd, maar te véél overslaan haalt frames uit de meting.
    Draaien met `python schaats_yolo.py`.
    """
    W, H, FPS = 1000, 1000, 30.0

    def _pose(ratio, romp, cx):
        """Landmark-lijst met precies deze heupbreedte/romplengte-verhouding."""
        lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
        hb = ratio * romp
        lm[11] = Landmark(cx - 0.05, 0.5 - romp / H / 2, 0, 1.0)
        lm[12] = Landmark(cx + 0.05, 0.5 - romp / H / 2, 0, 1.0)
        lm[23] = Landmark(cx - hb / W / 2, 0.5 + romp / H / 2, 0, 1.0)
        lm[24] = Landmark(cx + hb / W / 2, 0.5 + romp / H / 2, 0, 1.0)
        return lm

    def _det(ratio, tid=1, cx=0.5, romp=100):
        return Detectie(frame=0, tid=tid, centroid=(cx, 0.5),
                        bbox=(cx - 0.05, 0.3, cx + 0.05, 0.7), area=0.04,
                        lm=_pose(ratio, romp, cx))

    def _loop(n, maak_dets, aan=True):
        """Retourneert (aantal geanalyseerd, aantal overgeslagen, frame waarop het weer
        op vol tempo ging na de laatste skip)."""
        wacht = _BochtWacht(FPS, W, H, aan=aan)
        geanalyseerd = overgeslagen = 0
        for f in range(n):
            if not wacht.analyseren(f):
                overgeslagen += 1
                continue
            geanalyseerd += 1
            wacht.voed(f, maak_dets(f))
        return geanalyseerd, overgeslagen

    # 1. Frontale, groeiende schaatser: nooit overslaan (hij komt recht op de camera af,
    #    dus hij verplaatst in beeld nauwelijks — de groei moet hem redden).
    _, over = _loop(200, lambda f: [_det(0.9, romp=100 + f * 0.6)])
    assert over == 0, f"frontale schaatser werd {over} frames overgeslagen"

    # 2. Bocht: het overgrote deel moet worden overgeslagen.
    an, _ = _loop(300, lambda f: [_det(0.15)])
    assert an < 60, f"bocht: nog {an} van 300 frames geanalyseerd"

    # 3. Terug op het rechte stuk wordt binnen één controle-interval opgepakt.
    wacht = _BochtWacht(FPS, W, H)
    hervat = None
    for f in range(400):
        if not wacht.analyseren(f):
            continue
        wacht.voed(f, [_det(0.15 if f < 200 else 0.9, romp=100 + max(0, f - 200) * 0.6)])
        if f >= 200 and hervat is None:
            hervat = f
    assert hervat is not None and hervat - 200 <= wacht.check, f"hervat pas op frame {hervat}"

    # 4. Een stilstaande omstander staat frontaal in beeld, maar mag de bocht niet
    #    openhouden.
    an, _ = _loop(300, lambda f: [_det(0.9, tid=7, cx=0.2), _det(0.15, tid=1, cx=0.6)])
    assert an < 120, f"omstander hield de analyse {an} van 300 frames op vol tempo"

    # 5. Een blur-gat van 2 s op een recht stuk is géén bocht (BOCHT_STIL_S = 3 s).
    _, over = _loop(300, lambda f: [] if 100 <= f < 160 else [_det(0.9, romp=100 + f * 0.6)])
    assert over == 0, f"blur-gat leidde tot {over} overgeslagen frames"

    # 6. Uitgezet = alles analyseren, ook in de bocht.
    _, over = _loop(200, lambda f: [_det(0.1)], aan=False)
    assert over == 0, f"met bocht=False werden er toch {over} frames overgeslagen"

    # 7. Een controleframe is een kijkje, geen meting — ook niet als de schaatser er
    #    toevallig frontaal op staat (dat gebeurt: midden in een bocht draait hij af en
    #    toe kort bijna frontaal). We spelen de bocht na, laten één controleframe er
    #    frontaal uitzien en eisen dat het frame géén afgeleiden krijgt.
    wacht = _BochtWacht(FPS, W, H)
    resultaten, buiten_meting, frontaal_op = [], [], None
    for f in range(300):
        r = FrameResultaat(frame_nr=f, tijd=f / FPS)
        controle = wacht.skip                      # infereren we dit frame alleen als check?
        if not wacht.analyseren(f):
            buiten_meting.append(True)             # overgeslagen: geen inferentie
        else:
            buiten_meting.append(controle)
            r.pose_gevonden = True
            # Eén controleframe halverwege krijgt een frontale heupstand mee.
            if controle and frontaal_op is None and f > 150:
                r.lm, frontaal_op = _pose(1.0, 100, 0.5), f
            else:
                r.lm = _pose(0.15, 100, 0.5)
            wacht.voed(f, [_det(0.15)])
        resultaten.append(r)

    assert frontaal_op is not None, "geen controleframe om te toetsen"
    _bocht_met_controleframes(resultaten, buiten_meting, VideoInfo(W, H, FPS, len(resultaten)))
    assert resultaten[frontaal_op].bocht, (
        f"controleframe {frontaal_op} pleitte zichzelf vrij op z'n eigen heupstand")
    verwerk_afgeleiden(resultaten, W, H, FPS)
    assert all(r.lm_data is None for r in resultaten), \
        "een frame in de bocht leverde tóch een meting op"

    print("Zelftest _BochtWacht OK")


def _zelftest_seed():
    """
    Toetst `_kies_seed` op synthetische tracklets — geen video, geen model. Het geval dat
    ertoe doet is de schaatser die op het klikframe nog te ver weg is om gedetecteerd te
    worden: dan moet de klik alsnog bij hém uitkomen en niet stilzwijgend bij de grootste
    beweger. De cijfers hieronder zijn die van `00005 8-41`, de clip waarop dit misging.
    Draaien met `python schaats_yolo.py`.
    """
    FPS = 25.0
    KLIK = (0.391, 0.212)

    def _tracklet(tid, start, n, x0, y0, zwaai=0.0, drift=(0.0, 0.0),
                  breedte=0.03, hoogte=0.09):
        """Schaatser die om (x0, y0) heen zwaait (één slag per 25 frames) en langzaam
        wegdrijft — de beweging van een frontaal aanrijdende schaatser."""
        dets = []
        for k in range(n):
            x = x0 + zwaai * float(np.sin(2 * np.pi * k / 25)) + drift[0] * k
            y = y0 + drift[1] * k
            dets.append(Detectie(frame=start + k, tid=tid, centroid=(x, y),
                                 bbox=(x - breedte / 2, y - hoogte / 2,
                                       x + breedte / 2, y + hoogte / 2),
                                 area=breedte * hoogte,
                                 lm=[Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]))
        return dets

    N = 281
    frames = [[] for _ in range(N)]
    # De aangeklikte schaatser: pas vanaf frame 84 (3,4 s) groot genoeg om gedetecteerd
    # te worden, dan vlak naast de klik, zwaaiend met elke slag. Daarnaast een grotere
    # schaatser die vanaf frame 0 wél gezien wordt — de oude terugval koos díe.
    doel = _tracklet(1, 84, 197, 0.40, 0.23, zwaai=0.06, drift=(0.0, 0.0007))
    ander = _tracklet(2, 0, 200, 0.75, 0.55, zwaai=0.05, drift=(-0.001, 0.0005),
                      breedte=0.10, hoogte=0.30)
    for t in (doel, ander):
        for d in t:
            frames[d.frame].append(d)

    # 1. De klik hoort bij de schaatser die pas seconden later gedetecteerd wordt.
    seed, gemist = _kies_seed([doel, ander], frames, KLIK, FPS)
    assert not gemist, "klik op een nog niet gedetecteerde schaatser gold als gemist"
    assert seed is doel, "de klik kwam bij de verkeerde schaatser uit"

    # 2. Een klik op niemand blijft een gemiste klik: liever de grootste beweger mét
    #    melding dan een willekeurige schaatser als 'jouw klik'.
    seed, gemist = _kies_seed([doel, ander], frames, (0.03, 0.95), FPS)
    assert gemist and seed is ander, "klik in het niets leverde geen nette terugval op"

    # 3. Staat er wél iemand op de klik, dan wint die onverkort — het naast-de-klik-zoeken
    #    mag een gewone treffer nooit overrulen.
    seed, gemist = _kies_seed([doel, ander], frames, (0.75, 0.55), FPS)
    assert not gemist and seed is ander, "directe treffer op de klik ging verloren"

    # 4. De poort moet ook echt sluiten: een schaatser die aan de andere kant van het
    #    beeld rijdt hoort niet alsnog als 'jouw klik' te gelden.
    ver = _tracklet(3, 20, 100, 0.90, 0.80)
    seed, gemist = _kies_seed([ver], frames, (0.10, 0.15), FPS)
    assert gemist, "een schaatser ver buiten de poort werd toch als de klik gelezen"

    # 5. En het zoekvenster is eindig: wie pas ná KLIK_ZOEK_S over de klik rijdt, is een
    #    voorbijganger en niet de schaatser die je aanwees.
    laat = _tracklet(4, int(FPS * KLIK_ZOEK_S) + 10, 60, KLIK[0], KLIK[1])
    seed, gemist = _kies_seed([laat, ander], frames, KLIK, FPS)
    assert gemist, "een schaatser buiten het zoekvenster werd toch als de klik gelezen"

    # 6. Met een kader telt de maat mee: een omstander van vier keer de kaderhoogte die
    #    wél op de klik staat is niet wie er aangewezen is — en zonder passende
    #    kandidaat geeft _kies_seed géén terugval (`analyseer` beslist zelf of hij naast
    #    het kijkglas alsnog de grootste beweger volgt, en meldt dat dan).
    kader = (KLIK[0] - 0.02, KLIK[1] - 0.05, KLIK[0] + 0.02, KLIK[1] + 0.05)   # 0,10 hoog
    groot = _tracklet(5, 0, 120, KLIK[0], KLIK[1], breedte=0.14, hoogte=0.42)
    seed, gemist = _kies_seed([groot, doel, ander], frames, KLIK, FPS, doel_kader=kader)
    assert not gemist and seed is doel, "de maatpoort liet de grote omstander door"
    seed, gemist = _kies_seed([groot, ander], frames, KLIK, FPS, doel_kader=kader)
    assert gemist and seed is None, "met kader hoort er geen terugval op de beweger te zijn"
    # Zonder kader wint de omstander op de klik gewoon (bestaand gedrag, stap 1).
    seed, gemist = _kies_seed([groot, doel, ander], frames, KLIK, FPS)
    assert not gemist and seed is groot

    print("Zelftest _kies_seed OK")


def _zelftest_kijkglas():
    """
    Toetst `_Kijkglas` met een nep-verfijning — geen video, geen model. De verfijning
    levert een skelet op de plek waar de synthetische schaatser staat (of niets, om een
    misser na te bootsen); de toets is of het kijkglas de juiste frames vult, de
    koppeltoets goed toepast en op tijd stopt. Draaien met `python schaats_yolo.py`.
    """
    FPS = 25.0

    def skelet(cx, cy, hoogte):
        """33 landmarks op een rechthoek rond (cx, cy): genoeg voor `_bbox_uit_lm`."""
        lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
        hw, hh = hoogte * 0.2, hoogte * 0.5 / (1 + 2 * KIJKGLAS_MARGE)
        for i, (dx, dy) in zip((11, 12, 23, 24, 27, 28, 0),
                               ((-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1))):
            lm[i] = Landmark(cx + dx * hw, cy + dy * hh, 0.0, 0.9)
        return lm

    def schaatser(f):
        """Positie van de synthetische schaatser: rijdt langzaam naar rechtsonder en
        groeit (komt op de camera af)."""
        return 0.30 + 0.002 * f, 0.25 + 0.001 * f, 0.08 + 0.0004 * f

    def maak_schat(missers=(), verschoven=None):
        """Nep-verfijning: skelet op de echte plek als de bbox hem bevat; None op de
        opgegeven frames (occlusie); op `verschoven` frames een skelet dat een halve
        beeldbreedte verderop staat (RTMPose die op iemand anders gaat zitten)."""
        aanroepen = []

        def schat(f, bbox):
            aanroepen.append(f)
            if f in missers:
                return None, None, 0.1, None
            cx, cy, hoogte = schaatser(f)
            if verschoven and f in verschoven:
                cx += 0.5
            if not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
                return None, None, 0.1, None
            return skelet(cx, cy, hoogte), None, 0.8, None
        return schat, aanroepen

    def kader_op(f, ruim=1.0):
        cx, cy, hoogte = schaatser(f)
        hw, hh = 0.2 * hoogte * ruim, 0.5 * hoogte * ruim
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def loop(kg, schat, frames, keten):
        for f in frames:
            if f in keten:
                kg.anker(f, kader_op(f))
            elif kg.leeft:
                kg.stap(f, lambda bb, _f=f: schat(_f, bb))
        kg.afsluiten()

    # 1. Aanloop vanaf een (te ruim getekend) kader tot de keten op frame 40: de
    #    koppeltoets slaagt en alle 40 aanloopframes zijn gevuld.
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    kg.start_kader(kader_op(0, ruim=1.6), 0)
    schat, aanroepen = maak_schat()
    loop(kg, schat, range(60), keten=set(range(40, 60)))
    assert set(kg.gevuld) == set(range(40)), sorted(kg.gevuld)
    assert aanroepen[:3] == [0, 0, 0], "eerste stap hoort de drie startschalingen te proberen"
    assert any("geaccepteerd" in r for r in kg.logboek)

    # 2. Dezelfde aanloop, maar de keten zit op iemand anders (bbox aan de andere kant
    #    van het beeld): koppeltoets faalt, niets gevuld, reden in het logboek.
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    kg.start_kader(kader_op(0), 0)
    schat, _ = maak_schat()
    for f in range(40):
        kg.stap(f, lambda bb, _f=f: schat(_f, bb))
    kg.anker(40, (0.8, 0.8, 0.9, 0.95))
    assert not kg.gevuld and any("verworpen" in r for r in kg.logboek)

    # 3. Coast: een occlusie korter dan KIJKGLAS_COAST_S wordt overbrugd; daarna vult
    #    het kijkglas gewoon weer, vanaf een keten-anker (uitloop na frame 10).
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    schat, _ = maak_schat(missers=set(range(20, 30)))
    loop(kg, schat, range(60), keten=set(range(0, 11)))
    verwacht = set(range(11, 20)) | set(range(30, 60))
    assert set(kg.gevuld) == verwacht, sorted(set(kg.gevuld) ^ verwacht)

    # 4. Een occlusie langer dan de coast laat de run sterven; wat er vóór hing is van
    #    een keten-run en blijft staan, en na de dood wordt er niets meer gevuld.
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    schat, _ = maak_schat(missers=set(range(20, 60)))
    loop(kg, schat, range(80), keten=set(range(0, 11)))
    assert set(kg.gevuld) == set(range(11, 20)), sorted(kg.gevuld)
    assert not kg.leeft

    # 5. Een kader-run die sterft zonder de keten te raken wordt verworpen — een
    #    handgetekend kader is geen bewezen identiteit ...
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    kg.start_kader(kader_op(0), 0)
    schat, _ = maak_schat(missers=set(range(15, 80)))
    loop(kg, schat, range(80), keten=set())
    assert not kg.gevuld and any("zonder koppeling" in r for r in kg.logboek)
    assert kg.kader_uitkomst == {'ok': False, 'reden': kg.kader_uitkomst['reden'], 'n': 15}
    assert "koppeling" not in kg.kader_uitkomst['reden'].split(";")[0]   # gestorven, niet gekoppeld
    # ... behalve als er helemaal geen keten is (kijkglas-only): dan is het kader alles.
    kg = _Kijkglas(FPS, keten_gekoppeld=False)
    kg.start_kader(kader_op(0), 0)
    schat, _ = maak_schat()
    loop(kg, schat, range(50), keten=set())
    assert set(kg.gevuld) == set(range(50))
    assert kg.kader_uitkomst['ok'] and kg.kader_uitkomst['n'] == 50
    # ... en met een terugval-keten (niet gekoppeld) blijft een gestorven kader-run ook
    # staan: kader en keten zijn dan twee losse gissingen over verschillende stukken.
    kg = _Kijkglas(FPS, keten_gekoppeld=False)
    kg.start_kader(kader_op(0), 0)
    schat, _ = maak_schat(missers=set(range(15, 80)))
    loop(kg, schat, range(120), keten=set(range(100, 120)))
    assert set(kg.gevuld) == set(range(15)), sorted(kg.gevuld)

    # 6. Plausibiliteit: een skelet dat ineens een halve beeldbreedte verderop staat is
    #    een verwisseling en telt als misser, niet als treffer.
    kg = _Kijkglas(FPS, keten_gekoppeld=True)
    schat, _ = maak_schat(verschoven={12, 13})
    loop(kg, schat, range(30), keten=set(range(0, 11)))
    assert 12 not in kg.gevuld and 13 not in kg.gevuld and 14 in kg.gevuld

    print("Zelftest _Kijkglas OK")


if __name__ == '__main__':
    _zelftest_bochtwacht()
    _zelftest_seed()
    _zelftest_kijkglas()
