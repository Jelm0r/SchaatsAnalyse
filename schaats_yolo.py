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

De rest van de pijplijn (offline smoothing, afgeleiden, tekenen, GUI) uit
`schaats_analyse.py` wordt ongewijzigd hergebruikt. Vereist torch/ultralytics
(zie .venv-yolo). YOLO levert COCO-17 keypoints; die mappen we in de MediaPipe-33-
indeling. Hiel/teen bestaan niet in COCO en worden daar op de enkel gelegd met
visibility 0; RTMPose-26 levert ze wél echt (Halpe26 → MediaPipe 29–32).
"""
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO   # op moduleniveau: zo faalt de import meteen als torch/
                               # ultralytics ontbreekt, en kiest de GUI netjes MediaPipe.

try:                           # optioneel: RTMPose-verfijning (pip install rtmlib onnxruntime)
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    IS_RTMPOSE = True
except ImportError:
    IS_RTMPOSE = False

from schaats_analyse import (
    FrameResultaat, Landmark, video_info,
    smooth_landmarks_offline, verwerk_afgeleiden, zet_horizon, fase_voortgang,
    NUM_POSES_DEFAULT,
)

BACKEND_NAAM = ("YOLO-pose + ByteTrack + RTMPose-verfijning" if IS_RTMPOSE
                else "YOLO-pose + ByteTrack")

# yolo26x-pose = meest nauwkeurig (traagst op CPU). Alternatief: "yolo26m-pose.pt"
# (sneller, iets minder nauwkeurig). Ultralytics downloadt het model bij eerste gebruik.
# NB: YOLOv12 is nooit als pose-model uitgebracht (alleen detectie); YOLO26 is het
# nieuwste pose-model in ultralytics, opvolger van yolo11x-pose.
STANDAARD_YOLO_MODEL = "yolo26x-pose.pt"

# ── Detectie ────────────────────────────────────────────────────────────────────
DETECT_IMGSZ = 1280       # inferentieresolutie detectiepass; 640 mist verre/blurry schaatsers

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
SNELHEID_VENSTER  = 5        # aantal detecties waarover de snelheid wordt geschat
MIN_VERPLAATSING  = 0.06     # tracklet-padlengte hieronder = statische omstander
KLIK_ZOEK_FRAMES  = 60       # zolang zoeken we (in frames) naar de aangeklikte schaatser

# ── Verfijning ──────────────────────────────────────────────────────────────────
GAP_VUL_S       = 1.0        # max. detectiegat dat via geïnterpoleerde bboxes wordt gevuld
# RTMPose-26 (Halpe26 = COCO-17 + hoofd/nek/heupcentrum + voeten), top-down op de
# doel-bbox. 'body7'-gewichten = getraind op 7 datasets, robuust op sportbeelden.
RTMPOSE_MODEL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                 'onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288'
                 '-7fb6e239_20230606.zip')
RTMPOSE_INPUT = (288, 384)   # (breedte, hoogte) van de modelinvoer
RTMPOSE_MIN_SCORE = 0.3      # min. gemiddelde been-keypointscore om de schatting te vertrouwen
# Terugvalroute zonder rtmlib (vierkante crop door yolo26x):
VERFIJN_IMGSZ   = 640        # inferentiegrootte op de uitsnede
VERFIJN_MARGE   = 1.9        # cropzijde = marge × grootste bbox-zijde
VERFIJN_MIN_PX  = 256        # ondergrens cropzijde (pixels)

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
    onbetrouwbaar zijn. Retourneert een L1-genormaliseerd histogram, of None.
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
            return None
    if mask is None and bbox_px is None:
        return None
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(KLEUR_BINS),
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
    return hist


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


def _detecteer_alles(input_pad, model, info, imgsz=DETECT_IMGSZ, progress_callback=None):
    """
    Pass 1: YOLO-pose + ByteTrack over de hele video op hoge resolutie. Retourneert
    per frame een lijst Detectie's (alle personen, met torso-kleurhistogram).
    """
    w, h = info.w, info.h
    frames = []
    for res in model.track(source=input_pad, stream=True, persist=True, imgsz=imgsz,
                           tracker='bytetrack.yaml', classes=[0], verbose=False):
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
                dets.append(Detectie(
                    frame=len(frames),
                    tid=int(ids[i]) if ids is not None else None,
                    centroid=(cx / w, cy / h),
                    bbox=(bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h),
                    area=(bw * bh) / (w * h),
                    lm=_coco_naar_landmarks(xy[i], conf[i], w, h),
                    hist=_torso_hist(res.orig_img, xy[i], conf[i], bbox_px),
                ))
        frames.append(dets)
        if progress_callback is not None:
            progress_callback(len(frames), info.totaal)
    return frames


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
    """
    stukken, huidig = [], []
    ref = KleurReferentie()
    laag = []                       # opeenvolgende detecties onder de split-drempel
    for d in tracklet:
        s = ref.sim(d.hist)
        if s is not None and s < KLEUR_SPLIT_MIN:
            laag.append(d)
            if len(laag) >= KLEUR_SPLIT_N:
                # Aanhoudend een ander pak: knip vóór het eerste afwijkende frame.
                if huidig:
                    stukken.append(huidig)
                huidig, ref, laag = list(laag), KleurReferentie(), []
                for d2 in huidig:
                    ref.voeg_toe(d2.hist)
            continue
        huidig.extend(laag); laag = []          # korte dip (occlusie-mix): behouden
        huidig.append(d)
        ref.voeg_toe(d.hist)
    huidig.extend(laag)
    if huidig:
        stukken.append(huidig)
    return stukken


def _pad_lengte(tracklet):
    """Totale afgelegde weg van de centroid (genormaliseerd)."""
    return float(sum(
        np.hypot(b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1])
        for a, b in zip(tracklet, tracklet[1:])))


def _kies_seed(tracklets, frames, doel_punt):
    """
    Kies het start-tracklet. Met muisklik: loop de eerste frames chronologisch af en
    pak het eerste tracklet waarvan de bounding box het klikpunt bevat (dichtstbijzijnde
    centroid bij meerdere). Zonder klik: de grootste *beweger* — mediane oppervlakte ×
    padlengte — zodat statische omstanders langs de boarding nooit gekozen worden.
    """
    if doel_punt is not None:
        dx, dy = doel_punt
        per_frame = {}
        for t in tracklets:
            for d in t:
                per_frame.setdefault(d.frame, []).append((d, t))
        for f in range(min(KLIK_ZOEK_FRAMES, len(frames))):
            raak = [(d, t) for d, t in per_frame.get(f, ())
                    if d.bbox[0] - 0.03 <= dx <= d.bbox[2] + 0.03
                    and d.bbox[1] - 0.03 <= dy <= d.bbox[3] + 0.03]
            if raak:
                return min(raak, key=lambda dt: (dt[0].centroid[0] - dx) ** 2
                                                + (dt[0].centroid[1] - dy) ** 2)[1]
        # Niemand onder de klik gevonden → val terug op de grootste beweger.
    bewegers = [t for t in tracklets if _pad_lengte(t) >= MIN_VERPLAATSING]
    kandidaten = bewegers or tracklets
    if not kandidaten:
        return None
    return max(kandidaten, key=lambda t: float(np.median([d.area for d in t]))
                                         * max(_pad_lengte(t), 1e-6))


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
    lopende referentie past (≥ KLEUR_MATCH_MIN) en (b) zijn startpositie binnen de
    poort van de constante-snelheid-voorspelling over het gat ligt. Retourneert
    (keten-detecties gesorteerd op frame, KleurReferentie).
    """
    max_gap = int(round(STITCH_MAX_GAP_S * fps))
    keten = list(seed)
    ref = KleurReferentie()
    for d in keten:
        ref.voeg_toe(d.hist)
    rest = [t for t in tracklets if t is not seed]

    def _kleur_sim(t):
        sims = [s for s in (ref.sim(d.hist) for d in t[:10]) if s is not None]
        return float(np.median(sims)) if sims else None

    def _probeer(richting):
        """richting=+1: aan het eind doorstikken; -1: vóór het begin."""
        while True:
            if richting > 0:
                eind = keten[-1]
                v = _snelheid(keten)                 # snelheid aan het keteneinde
                kandidaten = [t for t in rest if 0 < t[0].frame - eind.frame <= max_gap]
                def gat(t): return t[0].frame - eind.frame
                def start_det(t): return t[0]
            else:
                begin = keten[0]
                v = _snelheid(keten[:SNELHEID_VENSTER])   # snelheid aan het ketenbegin
                kandidaten = [t for t in rest if 0 < begin.frame - t[-1].frame <= max_gap]
                def gat(t): return begin.frame - t[-1].frame
                def start_det(t): return t[-1]

            beste, beste_score = None, -1.0
            for t in kandidaten:
                sim = _kleur_sim(t)
                if sim is None or sim < KLEUR_MATCH_MIN:
                    continue
                g = gat(t)
                if richting > 0:
                    px = keten[-1].centroid[0] + v[0] * g
                    py = keten[-1].centroid[1] + v[1] * g
                else:
                    px = keten[0].centroid[0] - v[0] * g
                    py = keten[0].centroid[1] - v[1] * g
                d0 = start_det(t)
                afst = float(np.hypot(d0.centroid[0] - px, d0.centroid[1] - py))
                poort = STITCH_GATE_BASIS + STITCH_GATE_GROEI * g
                if afst > poort:
                    continue
                score = sim - 0.5 * afst / poort
                if score > beste_score:
                    beste, beste_score = t, score
            if beste is None:
                return
            rest.remove(beste)
            keten.extend(beste)
            keten.sort(key=lambda d: d.frame)
            for d in (beste if richting > 0 else reversed(beste)):
                ref.voeg_toe(d.hist)

    _probeer(+1)
    _probeer(-1)
    _probeer(+1)          # na terugstikken kan er vooraan óf achteraan meer passen
    keten.sort(key=lambda d: d.frame)

    # Dubbele frames (licht overlappende tracklets): houd per frame de detectie die
    # het best bij de referentie past.
    per_frame = {}
    for d in keten:
        z = per_frame.get(d.frame)
        if z is None:
            per_frame[d.frame] = d
        else:
            sd, sz = ref.sim(d.hist), ref.sim(z.hist)
            if (sd or 0.0) > (sz or 0.0):
                per_frame[d.frame] = d
    return [per_frame[f] for f in sorted(per_frame)], ref


# ── Verfijningspass ─────────────────────────────────────────────────────────────
def _interpoleer_doel(doel_per_frame, n_frames, fps):
    """
    Vul detectiegaten ≤ GAP_VUL_S met lineair geïnterpoleerde bboxes, zodat de
    verfijningspass daar tóch een schatting kan proberen. Retourneert
    {frame: (bbox_norm_xyxy, echt)} — `echt` False voor geïnterpoleerde plekken.
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
                t = (f - a) / g
                plan[f] = (tuple(ba + t * (bb - ba)), False)
    return plan


def _maak_rtmpose():
    """RTMPose-26-model voor de verfijningspass, of None zonder rtmlib."""
    if not IS_RTMPOSE:
        return None
    return _RTMPose(RTMPOSE_MODEL, model_input_size=RTMPOSE_INPUT)


def _verfijn_landmarks(input_pad, model, info, doel_per_frame, ref,
                       progress_callback=None, rtmpose=None):
    """
    Pass 2: lees de video opnieuw en schat per doel-frame de pose opnieuw, nu met de
    schaatser beeldvullend in het inferentiebeeld → aanzienlijk nauwkeurigere
    keypoints dan in de volledige-frame-pass. Met `rtmpose` gaat dat top-down op de
    doel-bbox (RTMPose-26: subpixel-decodering + echte hiel/teen); anders via een
    vierkante crop door het YOLO-model. De pakkleur (referentie `ref`) bewaakt in
    beide routes dat nooit stilletjes een andere persoon wordt overgenomen.
    Retourneert {frame: lm} met verfijnde (of herstelde) landmarks.
    """
    plan = _interpoleer_doel(doel_per_frame, info.totaal, info.fps)
    w, h = info.w, info.h
    uit, devs = {}, {}
    cap = cv2.VideoCapture(input_pad)
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f in plan:
            bbox, echt = plan[f]
            if rtmpose is not None:
                lm, dev = _verfijn_rtmpose(rtmpose, frame, bbox, echt, ref, w, h)
                if dev is not None:
                    devs[f] = dev
            else:
                lm = _verfijn_yolo_crop(model, frame, bbox, ref, w, h)
            if lm is not None:
                uit[f] = lm
        f += 1
        if progress_callback is not None:
            progress_callback(f, info.totaal)
    cap.release()
    return uit, devs


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


def _verfijn_rtmpose(rtmpose, frame, bbox, echt, ref, w, h):
    """
    Top-down verfijning van één frame: RTMPose-26 op de doel-bbox (pixels).
    De kleurpoort houdt de andere schaatser buiten: matcht de torso-kleur van de
    schatting niet met de referentie, dan vervalt de verfijning (bij een echte
    detectie blijven de pass-1-landmarks staan; een geïnterpoleerd gat-frame eist
    juist een positieve kleurmatch, want daar is geen pass-1-vangnet).
    Retourneert (lm | None, middellijn-dev-dict | None).
    """
    bbox_px = (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)
    kps, scores = rtmpose(frame, [list(bbox_px)])
    kp, sc = kps[0], scores[0]

    if float(sc[list(_HALPE_BENEN)].mean()) < RTMPOSE_MIN_SCORE:
        return None, None                  # benen niet gezien (occlusie): niet vertrouwen
    sim = ref.sim(_torso_hist(frame, kp, sc, bbox_px))
    if echt:
        if sim is not None and sim < KLEUR_SPLIT_MIN:
            return None, None              # duidelijk een ander pak in de bbox
    else:
        if sim is None or sim < KLEUR_MATCH_MIN:
            return None, None              # gat-frame: alleen vullen bij zékere match

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
    return _halpe26_naar_landmarks(kp, sc, w, h), dev


def _verfijn_yolo_crop(model, frame, bbox, ref, w, h):
    """Terugvalroute zonder rtmlib: vierkante crop rond de bbox door het YOLO-model."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zijde = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    kant = int(max(VERFIJN_MIN_PX, VERFIJN_MARGE * zijde * max(w, h)))
    kant = min(kant, min(w, h))
    x0 = int(np.clip(cx * w - kant / 2, 0, w - kant))
    y0 = int(np.clip(cy * h - kant / 2, 0, h - kant))
    crop = frame[y0:y0 + kant, x0:x0 + kant]
    res = model.predict(crop, imgsz=VERFIJN_IMGSZ, classes=[0], verbose=False)[0]
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
        hist = _torso_hist(crop, xy[i], conf[i],
                           (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        sim = ref.sim(hist)
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


# ── Hoofd-API ───────────────────────────────────────────────────────────────────
def analyseer(input_pad, model_pad=None, smooth_n=5, threshold=0.015, force_fps=None,
              num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
              progress_callback=None, yolo_model=None, horizon_deg=0.0,
              auto_horizon=False, verfijn=True, perspectief=None):
    """
    Volledige analyse via YOLO-pose + ByteTrack + offline doelkeuze + crop-verfijning.
    Signatuur-compatibel met schaats_analyse.analyseer() (`model_pad` — het MediaPipe
    .task — en `num_poses` worden genegeerd; YOLO detecteert altijd alle personen).
    Retourneert (VideoInfo, lijst[FrameResultaat]).

    `perspectief` (PerspectiefConfig) werkt identiek aan de MediaPipe-backend: de
    gedeelde stappen `zet_horizon`/`verwerk_afgeleiden` doen al het werk.
    """
    info = video_info(input_pad, force_fps)
    model = YOLO(yolo_model or STANDAARD_YOLO_MODEL)
    if perspectief is not None:
        auto_horizon = False     # vaste camera per aanname; kalibratie kent de kanteling al

    # Meerdere passes → één doorlopende voortgangsbalk via fase-schijven.
    n_fasen = 1 + (1 if verfijn else 0) + (1 if auto_horizon else 0)
    fase = 0
    det_cb = fase_voortgang(progress_callback, fase, n_fasen); fase += 1
    ver_cb = fase_voortgang(progress_callback, fase, n_fasen) if verfijn else None
    fase += 1 if verfijn else 0
    hor_cb = fase_voortgang(progress_callback, fase, n_fasen) if auto_horizon else None

    # Pass 1: alle detecties verzamelen (hoge resolutie).
    frames = _detecteer_alles(input_pad, model, info, progress_callback=det_cb)
    n_frames = len(frames)

    # Offline doelkeuze: tracklets → kleur-splits → seed → stitching.
    tracklets = []
    for t in _bouw_tracklets(frames):
        tracklets.extend(_splits_op_kleur(t))
    seed = _kies_seed(tracklets, frames, doel_punt)

    doel_per_frame, ref = {}, KleurReferentie()
    if seed is not None:
        keten, ref = _stik_keten(seed, tracklets, info.fps)
        doel_per_frame = {d.frame: d for d in keten}

    # Pass 2: verfijning (nauwkeurigere keypoints + gaten vullen), top-down met
    # RTMPose-26 als rtmlib beschikbaar is, anders de oude YOLO-crop-route.
    if verfijn and doel_per_frame:
        verfijnd, devs = _verfijn_landmarks(input_pad, model, info, doel_per_frame, ref,
                                            progress_callback=ver_cb, rtmpose=_maak_rtmpose())
    else:
        verfijnd, devs = {}, {}

    resultaten = []
    for f in range(n_frames):
        r = FrameResultaat(frame_nr=f, tijd=f / info.fps if info.fps > 0 else 0)
        lm = verfijnd.get(f)
        if lm is None and f in doel_per_frame:
            lm = doel_per_frame[f].lm
        if lm is not None:
            r.lm = lm
            r.pose_gevonden = True
            r.middellijn_dev = devs.get(f)
        resultaten.append(r)

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)
    zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten
