"""
Schaatser Analyse Tool
======================
Detecteert het afzetbeen, berekent de afzethoek t.o.v. het ijs,
en geeft aan zolang het gewicht op het afzetbeen zit.

Gebruik:
    python schaats_analyse.py --input video.mp4 --output resultaat.mp4

Vereisten:
    pip install mediapipe opencv-python numpy
    Download het pose-landmarker modelbestand (eenmalig):
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
    Zet het naast dit script, of geef het pad door met --model.

Opties:
    --input     Pad naar invoervideo
    --output    Pad naar uitvoervideo (default: output.mp4)
    --model     Pad naar pose_landmarker .task modelbestand
    --fps       Forceer output FPS (default: zelfde als invoer)
    --smooth    Aantal frames voor hoek-smoothing (default: 5)
    --threshold Minimale heup-verschuiving voor gewichtsdetectie (default: 0.015)
"""

import os
import cv2
import numpy as np
import argparse
from collections import deque, namedtuple
from dataclasses import dataclass, field

import schaats_perspectief   # puur numpy — veilig in beide venvs

# MediaPipe wordt bewust NIET op moduleniveau geïmporteerd: dan kan dit bestand ook
# geladen worden in een omgeving zónder mediapipe (bv. de YOLO-venv, die de gedeelde
# functies hergebruikt). De import gebeurt lokaal in analyseer_frames().

# Vaste MediaPipe Pose 33-landmark connections (voorheen uit mp_vision gehaald),
# hier hardcoded zodat het tekenen geen mediapipe-import vereist.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32),
]

# Lichtgewicht landmark-vervanger: heeft dezelfde .x/.y/.z/.visibility-velden als
# een MediaPipe NormalizedLandmark, zodat get_landmarks() en de tekenfuncties er
# ongewijzigd mee werken. We gebruiken 'm voor gesmoothde punten.
Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])

# ── Detectie / tracking ─────────────────────────────────────────────────────────
# We laten MediaPipe meerdere personen detecteren en kiezen zelf de juiste met een
# eigen tracker (DoelTracker), zodat de detectie niet naar een andere schaatser springt.
NUM_POSES_DEFAULT = 5     # max. aantal gelijktijdig te detecteren schaatsers
TRACK_GATE        = 0.14  # max. genormaliseerde sprong van de torso-centroïde per frame
TRACK_HERVIND_S   = 0.5   # zolang de doelschaatser kwijt is voordat we opnieuw seeden (s)
TRACK_MIN_VIS     = 0.3   # minimale zichtbaarheid om een landmark mee te tellen
TORSO_IDX         = (11, 12, 23, 24)  # schouders + heupen: stabiele identiteits-centroïde

# ── Offline landmark-smoothing (Savitzky–Golay) ─────────────────────────────────
# Omdat dit een batch-tool is en we álle frames hebben, smoothen we bidirectioneel
# (zero-lag) i.p.v. causaal. SG onderdrukt jitter maar behoudt pieken (bv. volledige
# strekking bij afzet) beter dan een gewoon voortschrijdend gemiddelde.
SMOOTH_WINDOW_S = 0.25    # vensterlengte in seconden (wordt omgezet naar oneven # frames)
SMOOTH_POLY     = 2       # polynoomorde van de SG-fit
HORIZON_SMOOTH_S = 0.5    # vensterlengte (s) voor het smoothen van de per-frame horizon-schatting

# ── Uitschieter-verwerping (occlusie) ───────────────────────────────────────────
# Als een ledemaat een ander verbergt (bv. arm vóór heup/been) verspringt een landmark
# kortstondig. Zulke frames herkennen we aan lage zichtbaarheid óf een sprong die ver
# van de lokale mediaan ligt (Hampel), en interpoleren we weg vóór de smoothing.
VIS_MIN        = 0.2      # landmark onbetrouwbaar onder deze zichtbaarheid
HAMPEL_WINDOW  = 7        # vensterlengte (frames) voor de mediaan/MAD
HAMPEL_K       = 4.5      # aantal robuuste standaarddeviaties voor "uitschieter"
INTERP_MAX_S   = 0.1      # max. duur (s) van een onbetrouwbare reeks die weg-geïnterpoleerd
                          # wordt; langere reeksen (bv. bewegingsonscherpte over meerdere
                          # frames) houden de ruwe detectie — die zit óp de schaatser, een
                          # lange rechte-lijn-interpolatie zet het hele skelet ernaast
BLUR_FRAME_FRAC = 0.5     # is minder dan deze fractie van de data-dragende gewrichten in een
                          # frame "betrouwbaar", dan is dat een gecorreleerde confidence-dip
                          # (bewegingsonscherpte) — vertrouw dan de hele ruwe detectie i.p.v.
                          # het complete skelet weg te interpoleren

# ── Cyclus-bewuste afzetbeen-bepaling ───────────────────────────────────────────
# Het enkel-hoogteverschil (links vs rechts) oscilleert met de schaatsslag. We
# smoothen dat signaal en passen het met hysterese toe, zodat het standbeen alleen
# wisselt bij een echte gewichtsoverdracht (geen per-frame geflikker rond de wissel).
STANCE_SMOOTH_S = 0.18    # smoothing-venster (s) van het stand-signaal
STANCE_BAND_FRAC = 0.20   # hysterese-band als fractie van de signaalamplitude

# ── Perspectiefcorrectie (fase 7) ───────────────────────────────────────────────
# De 3D-reconstructie zelf zit in schaats_perspectief.py; hier alleen de koppeling.
ENKEL_HOOGTE_M       = 0.10   # enkel-landmark ligt op malleolus + schaats, niet óp het ijs
RIJRICHTING_VENSTER_S = 0.4   # venster (s) voor de traject-richting uit wereldposities
RIJRICHTING_MIN_M     = 0.15  # minimale verplaatsing in het venster om de richting te vertrouwen

# ── Landmark indices (MediaPipe Pose) ──────────────────────────────────────────
L_HIP, R_HIP     = 23, 24
L_KNEE, R_KNEE   = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL   = 29, 30
L_TOE,  R_TOE    = 31, 32

# ── Kleuren (BGR) ──────────────────────────────────────────────────────────────
GROEN    = (80, 200, 80)
ROOD     = (60, 60, 220)
WIT      = (255, 255, 255)
GEEL     = (0, 210, 230)
DONKER   = (20, 20, 20)
PAARS    = (200, 100, 220)
SKELET   = (255, 200, 0)   # helder cyaan-blauw; goed zichtbaar en botst niet met groen/rood afzetbeen


@dataclass
class VideoInfo:
    """Video-eigenschappen, los uit te lezen zonder frames te decoderen."""
    w: int
    h: int
    fps: float
    totaal: int


@dataclass
class FrameResultaat:
    """Analyseresultaat van één frame, zonder pixeldata (goedkoop te cachen)."""
    frame_nr: int
    tijd: float
    lm: object = None           # ruwe mediapipe landmarks (genormaliseerd), voor skelet-tekening
    lm_data: dict = None        # pixelcoördinaten per landmark
    been: str = None
    hoek: float = None
    smooth_hoek: float = None
    kniehoek: float = None
    gewicht_erop: bool = None
    signalen: list = field(default_factory=list)
    pose_gevonden: bool = False
    horizon_deg: float = 0.0    # camerakanteling t.o.v. het ijs bij dit frame (per-frame bij auto)
    # Perspectiefcorrectie (alleen gevuld mét kalibratie; None/True = geen correctie actief):
    hoek_correctie: float = None    # gecorrigeerde − oude beeldvlak-hoek (kwaliteitsindicator)
    hoek_betrouwbaar: bool = True   # False: been bijna in de kijkrichting / geometrie sloot niet
    wereld_xy: tuple = None         # positie op de baan in meters (voor snelheid/slaglengte)
    snelheid: float = None          # m/s (alleen als de lijnafstand-schaal bekend is)


@dataclass
class AfzetEvent:
    """Eén samenhangende afzetbeweging (gewicht op hetzelfde been)."""
    index: int
    been: str
    start_frame: int
    eind_frame: int
    start_tijd: float
    eind_tijd: float
    hoek: float       # hoek bij afzet-voltooiing (representatief)
    min_hoek: float
    max_hoek: float
    opmerking: str = None   # bv. "alternatie?" als L/R niet klopt, of "samengevoegd"
    # Perspectiefcorrectie (None zonder kalibratie):
    correctie: float = None      # toegepaste correctie bij afzet-voltooiing (graden)
    betrouwbaar: bool = True     # False: hoek gemeten met been bijna in de kijkrichting
    snelheid: float = None       # gemiddelde snelheid tijdens de afzet (m/s)
    slaglengte: float = None     # afgelegde afstand tijdens de afzet (m)


@dataclass
class PerspectiefConfig:
    """Opt-in perspectiefcorrectie via baanlijnen (fase 7): een kalibratie uit
    `schaats_perspectief.kalibreer_uit_lijnen` plus de reconstructie-keuzes.
    Zonder deze config gedraagt de pijplijn zich exact als voorheen."""
    kalibratie: object              # schaats_perspectief.PerspectiefKalibratie
    methode: str = "onderbeen"      # 'onderbeen' (bol-snijding) | 'beenvlak' (rijrichting-vlak)
    onderbeen_l: float = None       # onderbeenlengte in m (verplicht bij 'onderbeen')
    enkel_hoogte: float = ENKEL_HOOGTE_M


def video_info(input_pad, force_fps=None):
    """Leest video-eigenschappen zonder frames te decoderen."""
    cap = cv2.VideoCapture(input_pad)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {input_pad}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = force_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(w, h, fps, totaal)


def bereken_hoek_tov_ijs(enkel_xy, knie_xy, horizon_deg=0.0):
    """
    Bereken de hoek van het been t.o.v. het ijs.
    Retourneert hoek in graden (0° = evenwijdig aan het ijs, 90° = loodrecht erop).

    `horizon_deg` is de kanteling van de echte ijslijn t.o.v. de horizontale beeldas
    (positief = de ijslijn loopt naar rechts omhoog); die wordt van de gemeten hoek
    afgetrokken zodat een scheefstaande camera de afzethoek niet vervuilt. Voor kleine
    kantelingen (de praktijk) is dit aftrekken nauwkeurig genoeg; bij grote kanteling
    zou je de landmarks eerst moeten roteren (de `abs(dx)`-aanname gaat dan wringen).
    """
    dx = knie_xy[0] - enkel_xy[0]
    dy = enkel_xy[1] - knie_xy[1]   # y-as omgekeerd in beeldcoördinaten
    hoek = np.degrees(np.arctan2(dy, abs(dx)))
    return round(hoek - horizon_deg, 1)


def horizon_hoek_uit_lijn(p1, p2):
    """
    Kantelhoek (graden) van een referentielijn (bv. langs het ijs of de boarding)
    t.o.v. de horizontale beeldas. Positief = de lijn loopt naar rechts omhoog.
    `p1`/`p2` zijn pixel-(x, y)-punten; de richting wordt genormaliseerd (p2 rechts).
    Schaal-invariant, dus punten in geschaalde weergavecoördinaten mogen ook.
    """
    (x1, y1), (x2, y2) = p1, p2
    if x2 < x1:                                  # p2 altijd rechts van p1
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
    return round(float(np.degrees(np.arctan2(-(y2 - y1), (x2 - x1) or 1e-9))), 2)


def detecteer_ijslijn(frame_bgr, max_kanteling=20.0, roi_onder=0.45, min_lijnen=2):
    """
    Schat de camerakanteling automatisch uit één frame: zoekt sterke, bijna-horizontale
    randen (ijslijn, boarding, reclameband) met een Hough-transform en neemt de
    lengte-gewogen mediaan van hun kanteling. Retourneert `horizon_deg` (float) of
    None als er geen betrouwbare horizontale lijn is.

    Alleen het onderste deel van het beeld wordt bekeken (`roi_onder` = fractie hoogte,
    daar ligt het ijs); lijnen steiler dan `max_kanteling`° worden verworpen (verticale
    boarding-randen, benen). Dit is een *suggestie* — laat de gebruiker hem bevestigen.
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * (1.0 - roi_onder))
    roi = frame_bgr[y0:h, :]
    grijs = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    randen = cv2.Canny(grijs, 50, 150, apertureSize=3)
    lijnen = cv2.HoughLinesP(randen, 1, np.pi / 180, threshold=80,
                             minLineLength=int(w * 0.25), maxLineGap=20)
    if lijnen is None:
        return None

    hoeken, gewichten = [], []
    for x1, y1, x2, y2 in lijnen[:, 0]:
        hoek = horizon_hoek_uit_lijn((x1, y1), (x2, y2))
        if abs(hoek) <= max_kanteling:
            hoeken.append(hoek)
            gewichten.append(float(np.hypot(x2 - x1, y2 - y1)))   # langere lijn = sterker
    if len(hoeken) < min_lijnen:
        return None

    order = np.argsort(hoeken)
    hoeken = np.array(hoeken)[order]
    cum = np.cumsum(np.array(gewichten)[order])
    mediaan = hoeken[int(np.searchsorted(cum, cum[-1] / 2.0))]   # lengte-gewogen mediaan
    return round(float(mediaan), 2)


def bereken_kniehoek(heup, knie, enkel):
    """Hoek in het kniegewricht (heup–knie–enkel)."""
    a = np.array(heup)
    b = np.array(knie)
    c = np.array(enkel)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cos_a, -1, 1))), 1)


def get_landmarks(lm, w, h):
    """Converteer genormaliseerde landmarks naar pixelcoördinaten."""
    def pt(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))
    def vis(idx):
        return lm[idx].visibility

    return {
        'l_heup':   pt(L_HIP),   'r_heup':   pt(R_HIP),
        'l_knie':   pt(L_KNEE),  'r_knie':   pt(R_KNEE),
        'l_enkel':  pt(L_ANKLE), 'r_enkel':  pt(R_ANKLE),
        'l_hiel':   pt(L_HEEL),  'r_hiel':   pt(R_HEEL),
        'l_teen':   pt(L_TOE),   'r_teen':   pt(R_TOE),
        'vis_l_enkel': vis(L_ANKLE), 'vis_r_enkel': vis(R_ANKLE),
        'vis_l_knie':  vis(L_KNEE),  'vis_r_knie':  vis(R_KNEE),
    }


def bepaal_afzetbeen(lm_data, heup_history, w):
    """
    Bepaalt welk been het afzetbeen is o.b.v.:
    1. Welk been lager/verder naar buiten staat (laagste enkel = dichtstbij ijs)
    2. Richting van heupverschuiving
    """
    l_enkel_y = lm_data['l_enkel'][1]
    r_enkel_y = lm_data['r_enkel'][1]

    # Hogere y-waarde = lager in beeld = dichter bij ijs
    if abs(l_enkel_y - r_enkel_y) < 10:
        # Ankels op gelijke hoogte: gebruik heupverschuiving als tiebreaker
        if len(heup_history) >= 3:
            heup_dx = lm_data['r_heup'][0] - heup_history[-3][0]
            return 'links' if heup_dx > 0 else 'rechts'
        return 'rechts'

    return 'links' if l_enkel_y > r_enkel_y else 'rechts'


def detecteer_gewicht_op_been(been, lm_data, enkel_history, heup_history, w, h, threshold):
    """
    Bepaal of het gewicht nog op het afzetbeen zit.

    Signalen dat gewicht WEG is van het been:
    - Enkel stijgt plotseling (been verlaat ijs)
    - Knie is vrijwel gestrekt (extensie > 160°)
    - Heup beweegt snel weg van het been
    """
    if been == 'links':
        enkel = lm_data['l_enkel']
        knie  = lm_data['l_knie']
        heup  = lm_data['l_heup']
    else:
        enkel = lm_data['r_enkel']
        knie  = lm_data['r_knie']
        heup  = lm_data['r_heup']

    gewicht_signalen = []

    # Signaal 1: enkel-y stabiel? (relatief t.o.v. framegrootte)
    if len(enkel_history[been]) >= 4:
        recente_y = [e[1] for e in list(enkel_history[been])[-4:]]
        stijging = recente_y[0] - recente_y[-1]   # positief = omhoog
        if stijging / h > 0.015:
            gewicht_signalen.append('enkel_omhoog')

    # Signaal 2: been gestrekt? (knie extensie)
    kniehoek = bereken_kniehoek(
        (heup[0]/w, heup[1]/h),
        (knie[0]/w, knie[1]/h),
        (enkel[0]/w, enkel[1]/h)
    )
    if kniehoek > 162:
        gewicht_signalen.append('been_gestrekt')

    # Signaal 3: heup verschuift weg van het been
    if len(heup_history) >= 4:
        heup_midden_huidig  = (lm_data['l_heup'][0] + lm_data['r_heup'][0]) / 2 / w
        heup_midden_eerder  = (heup_history[-4][0] + heup_history[-4][1]) / 2 / w
        been_kant = lm_data['l_enkel'][0]/w if been == 'links' else lm_data['r_enkel'][0]/w

        verschuiving = heup_midden_huidig - heup_midden_eerder
        # Bij linkerben: heup verschuift rechts (positief) als gewicht overgaat
        if been == 'links' and verschuiving > threshold:
            gewicht_signalen.append('heup_weg')
        elif been == 'rechts' and verschuiving < -threshold:
            gewicht_signalen.append('heup_weg')

    # Gewicht weg als 2+ signalen aanwezig
    gewicht_erop = len(gewicht_signalen) < 2
    return gewicht_erop, kniehoek, gewicht_signalen


def teken_been_overlay(frame, lm_data, been, hoek, gewicht_erop, kniehoek, horizon_deg=0.0):
    """Teken de been-overlay met hoek en kleurcodering."""
    kleur = GROEN if gewicht_erop else ROOD

    if been == 'links':
        heup  = lm_data['l_heup']
        knie  = lm_data['l_knie']
        enkel = lm_data['l_enkel']
        hiel  = lm_data['l_hiel']
    else:
        heup  = lm_data['r_heup']
        knie  = lm_data['r_knie']
        enkel = lm_data['r_enkel']
        hiel  = lm_data['r_hiel']

    # Been-lijn dikker dan normaal
    cv2.line(frame, heup, knie, kleur, 4)
    cv2.line(frame, knie, enkel, kleur, 4)

    # IJslijn bij enkel — gekanteld volgens de ingestelde horizon (0° = horizontaal)
    ijslijn_len = 60
    rad = np.radians(horizon_deg)
    ex = int(np.cos(rad) * ijslijn_len)
    ey = int(np.sin(rad) * ijslijn_len)   # positieve horizon = naar rechts omhoog (y omlaag)
    cv2.line(frame,
             (enkel[0] - ex, enkel[1] + ey),
             (enkel[0] + ex, enkel[1] - ey),
             WIT, 2)

    # Hoeklijn verlengd (enkel → richting knie, maar projectie op grond)
    if hoek > 0:
        lijn_len = 80
        richting_x = int(np.sin(np.radians(hoek)) * lijn_len * (-1 if been == 'links' else 1))
        richting_y = -int(np.cos(np.radians(hoek)) * lijn_len)
        eind = (enkel[0] + richting_x, enkel[1] + richting_y)
        cv2.line(frame, enkel, eind, GEEL, 2, cv2.LINE_AA)

    # Knooppunten
    for punt in [heup, knie, enkel]:
        cv2.circle(frame, punt, 7, kleur, -1)
        cv2.circle(frame, punt, 7, WIT, 1)

    # Hoektekst bij enkel
    tekst_pos = (enkel[0] + 14, enkel[1] - 14)
    cv2.putText(frame, f"{hoek} deg", tekst_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, DONKER, 4)
    cv2.putText(frame, f"{hoek} deg", tekst_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, GEEL, 2)


def teken_hud(frame, been, hoek, gewicht_erop, kniehoek, smooth_hoek, frame_nr, fps, w, h,
              correctie=None, betrouwbaar=True, snelheid=None):
    """Teken het HUD-paneel linksboven. `correctie`/`betrouwbaar`/`snelheid` zijn de
    perspectiefcorrectie-velden (alleen getoond als er een kalibratie actief was)."""
    tijd = frame_nr / fps if fps > 0 else 0
    kleur_status = GROEN if gewicht_erop else ROOD
    status_tekst = "GEWICHT OP BEEN" if gewicht_erop else "AFZET VOLTOOID"

    regels = [
        (f"t = {tijd:.2f}s  |  frame {frame_nr}", WIT, 0.45),
        (f"Afzetbeen: {been.upper()}", WIT, 0.55),
        (f"Afzethoek: {hoek} deg  (gem: {smooth_hoek} deg)", GEEL, 0.65),
        (f"Kniehoek:  {kniehoek} deg", PAARS, 0.55),
        (status_tekst, kleur_status, 0.6),
    ]
    if correctie is not None:
        regels.insert(3, (f"Persp.corr.: {correctie:+.1f} deg", WIT, 0.5))
        if snelheid is not None:
            regels.insert(4, (f"Snelheid: {snelheid:.1f} m/s", WIT, 0.5))
        if not betrouwbaar:
            regels.append(("HOEK ONBETROUWBAAR (kijkrichting)", ROOD, 0.45))

    paneel_h = 26 * len(regels) + 40      # 5 regels → 170, zoals voorheen
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, paneel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (280, paneel_h), (80, 80, 80), 1)

    y = 36
    for tekst, kleur, schaal in regels:
        cv2.putText(frame, tekst, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, schaal, DONKER, 3)
        cv2.putText(frame, tekst, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, schaal, kleur, 1)
        y += 26


def teken_alle_landmarks(frame, landmarks, w, h, min_vis=0.2):
    """
    Teken alle pose-landmarks licht op de achtergrond. Landmarks met te lage
    zichtbaarheid worden overgeslagen (zo tekent de YOLO-backend, die ontbrekende
    33-slots op visibility 0 zet, geen lijnen naar (0,0)).
    """
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    zichtbaar = [getattr(l, 'visibility', 1.0) >= min_vis for l in landmarks]
    for a, b in POSE_CONNECTIONS:
        if zichtbaar[a] and zichtbaar[b]:
            cv2.line(frame, pts[a], pts[b], SKELET, 2, cv2.LINE_AA)
    for p, zb in zip(pts, zichtbaar):
        if zb:
            cv2.circle(frame, p, 3, SKELET, -1, cv2.LINE_AA)


def teken_overlay_op_frame(frame, resultaat, fps, toon_skelet=True, toon_afzetbeen=True,
                           toon_hud=True):
    """
    Tekent de overlay voor één frame o.b.v. een FrameResultaat, met per laag
    aan/uit te zetten. Gedeeld door de CLI-video-export en de GUI-live-weergave.
    De getekende ijslijn kantelt mee met `resultaat.horizon_deg` (per frame), zodat
    de overlay de gebruikte referentie toont — ook bij een schommelende camera.
    """
    h, w = frame.shape[:2]

    if not resultaat.pose_gevonden:
        cv2.putText(frame, "Geen pose gedetecteerd", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROOD, 2)
        return

    if toon_skelet:
        teken_alle_landmarks(frame, resultaat.lm, w, h)

    if toon_afzetbeen:
        teken_been_overlay(frame, resultaat.lm_data, resultaat.been, resultaat.hoek,
                            resultaat.gewicht_erop, resultaat.kniehoek, resultaat.horizon_deg)

    if toon_hud:
        teken_hud(frame, resultaat.been, resultaat.hoek, resultaat.gewicht_erop,
                  resultaat.kniehoek, resultaat.smooth_hoek, resultaat.frame_nr, fps, w, h,
                  correctie=resultaat.hoek_correctie, betrouwbaar=resultaat.hoek_betrouwbaar,
                  snelheid=resultaat.snelheid)


def _zichtbare_xy(lm, idxs=None, min_vis=TRACK_MIN_VIS):
    """Genormaliseerde (x,y) van voldoende zichtbare landmarks; optioneel beperkt tot idxs."""
    bron = lm if idxs is None else [lm[i] for i in idxs]
    return [(l.x, l.y) for l in bron if l.visibility >= min_vis]


def _torso_centroid(lm):
    """
    Stabiele identiteits-centroïde (schouders + heupen), genormaliseerd. De romp
    beweegt rustiger dan de ledematen, dus dit is een betrouwbaar anker om dezelfde
    schaatser frame na frame te herkennen. None als er te weinig zichtbaar is.
    """
    pts = _zichtbare_xy(lm, TORSO_IDX)
    if not pts:
        pts = _zichtbare_xy(lm)          # val terug op alle zichtbare punten
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _bbox(lm):
    """Bounding box (minx, miny, maxx, maxy) rond de zichtbare landmarks, of None."""
    pts = _zichtbare_xy(lm)
    if len(pts) < 4:
        return None
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_oppervlak(lm):
    """Oppervlak van de bounding box rond de zichtbare landmarks (genormaliseerd)."""
    box = _bbox(lm)
    if box is None:
        return 0.0
    return (box[2] - box[0]) * (box[3] - box[1])


class DoelTracker:
    """
    Volgt één doelschaatser door de frames heen. MediaPipe levert per frame een
    lijst poses (meerdere schaatsers); deze tracker kiest telkens de pose die het
    best bij de voorspelde positie van het doel past, met een afstandspoort zodat
    de tracking niet naar een andere schaatser overspringt als ze elkaar kruisen.

    Seeden: als `doel_punt` (genormaliseerd (x,y), bv. een muisklik) gegeven is,
    wordt de schaatser het dichtst daarbij gekozen; anders de grootste (meest
    prominente) schaatser in beeld.
    """
    def __init__(self, doel_punt=None, gate=TRACK_GATE, hervind_frames=15):
        self.doel_punt = doel_punt
        self.gate = gate
        self.hervind_frames = hervind_frames
        self.centroid = None         # laatst bekende torso-centroïde
        self.snelheid = (0.0, 0.0)   # geschatte verplaatsing per frame
        self.kwijt = 0               # aantal opeenvolgende frames zonder match

    def _seed(self, centroids):
        """Kies een startpose uit [(centroid, pose), ...] (alle centroids != None)."""
        if self.doel_punt is not None and self.centroid is None:
            dx, dy = self.doel_punt
            # Bij een muisklik telt vooral wélke schaatser je aanwees: geef voorrang aan
            # de schaatser wiens bounding box het klikpunt bevat, zodat hij niet op een
            # andere schaatser lockt wiens romp-centroïde toevallig dichter bij de klik
            # ligt (bv. als je op de schaatsen/onderbenen klikt i.p.v. de romp).
            def _in_box(cp):
                box = _bbox(cp[1])
                return box is not None and box[0] <= dx <= box[2] and box[1] <= dy <= box[3]
            binnen = [cp for cp in centroids if _in_box(cp)]
            kandidaten = binnen if binnen else centroids
            return min(kandidaten, key=lambda cp: (cp[0][0]-dx)**2 + (cp[0][1]-dy)**2)
        if self.centroid is not None:
            # Net kwijt geweest: pak de schaatser het dichtst bij de laatst bekende plek.
            cx, cy = self.centroid
            return min(centroids, key=lambda cp: (cp[0][0]-cx)**2 + (cp[0][1]-cy)**2)
        # Koude start zonder klik: volg de grootste (meest prominente) schaatser.
        return max(centroids, key=lambda cp: _bbox_oppervlak(cp[1]))

    def update(self, poses):
        """Kies de doel-pose voor dit frame; retourneert de landmarklijst of None."""
        centroids = [(_torso_centroid(p), p) for p in poses]
        centroids = [(c, p) for c, p in centroids if c is not None]
        if not centroids:
            self.kwijt += 1
            if self.kwijt > self.hervind_frames:
                self.centroid = None
            return None

        # Nog geen lock, of te lang kwijt → (her)seed.
        if self.centroid is None or self.kwijt > self.hervind_frames:
            c, p = self._seed(centroids)
            self.centroid = c
            self.snelheid = (0.0, 0.0)
            self.kwijt = 0
            return p

        # Voorspel de positie en kies de dichtstbijzijnde pose binnen de poort.
        px = self.centroid[0] + self.snelheid[0]
        py = self.centroid[1] + self.snelheid[1]
        (bc, bp), afstand = min(
            (((c, p), ((c[0]-px)**2 + (c[1]-py)**2) ** 0.5) for c, p in centroids),
            key=lambda t: t[1],
        )
        if afstand > self.gate:
            # Beste kandidaat te ver → waarschijnlijk de andere schaatser; coast.
            self.kwijt += 1
            if self.kwijt > self.hervind_frames:
                self.centroid = None
            return None

        # Match: snelheid en positie licht gedempt bijwerken.
        vx, vy = bc[0] - self.centroid[0], bc[1] - self.centroid[1]
        self.snelheid = (0.5 * self.snelheid[0] + 0.5 * vx,
                         0.5 * self.snelheid[1] + 0.5 * vy)
        self.centroid = bc
        self.kwijt = 0
        return bp


def analyseer_frames(input_pad, model_pad, force_fps=None, num_poses=NUM_POSES_DEFAULT,
                      doel_punt=None, progress_callback=None):
    """
    Generator: detecteert per frame álle schaatsers (multi-pose) en volgt met een
    DoelTracker de doelschaatser. Yield't per frame een FrameResultaat met alleen de
    ruwe landmarks (lm) van het doel + pose_gevonden — de afgeleide grootheden
    (afzetbeen/hoek/knie/gewicht) worden pas ná de offline smoothing ingevuld door
    verwerk_afgeleiden(). `progress_callback(frame_nr, totaal)` wordt na elk frame
    aangeroepen.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    cap = cv2.VideoCapture(input_pad)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {input_pad}")

    fps    = force_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_options = mp_python.BaseOptions(model_asset_path=model_pad)
    landmarker_options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tracker  = DoelTracker(doel_punt=doel_punt,
                           hervind_frames=int(max(1, fps * TRACK_HERVIND_S)))
    frame_nr = 0

    with mp_vision.PoseLandmarker.create_from_options(landmarker_options) as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_nr * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            poses = results.pose_landmarks or []
            doel = tracker.update(poses)

            resultaat = FrameResultaat(frame_nr=frame_nr, tijd=frame_nr / fps if fps > 0 else 0)
            if doel is not None:
                resultaat.lm = doel
                resultaat.pose_gevonden = True

            yield resultaat
            frame_nr += 1

            if progress_callback is not None:
                progress_callback(frame_nr, totaal)

    cap.release()


# ── Offline landmark-smoothing ──────────────────────────────────────────────────
def _savgol_coeffs(window, poly):
    """SG-coëfficiënten voor de gladde waarde in het midden van het venster."""
    half = window // 2
    k = np.arange(-half, half + 1)
    A = np.vander(k, poly + 1, increasing=True)   # kolommen k^0 .. k^poly
    return np.linalg.pinv(A)[0]                    # rij voor polynoomcoëfficiënt 0


def _savgol(y, window, poly):
    """Savitzky–Golay smoothing (numpy) met polynomiale randafhandeling."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if window % 2 == 0:
        window += 1
    if n < poly + 2:
        return y.copy()
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= poly:
        return y.copy()

    half = window // 2
    coeffs = _savgol_coeffs(window, poly)
    out = y.copy()
    out[half:n - half] = np.convolve(y, coeffs[::-1], mode='valid')

    # Randen: lokale polynoomfit over het eerste/laatste venster.
    xw = np.arange(window)
    pL = np.polyfit(xw, y[:window], poly)
    out[:half] = np.polyval(pL, xw[:half])
    pR = np.polyfit(xw, y[-window:], poly)
    out[n - half:] = np.polyval(pR, xw[window - half:])
    return out


def _hampel_uitschieters(y, window=HAMPEL_WINDOW, k=HAMPEL_K):
    """
    Booleaanse mask van uitschieters: punten die meer dan k robuuste standaard-
    deviaties (via mediaan/MAD in een lokaal venster) van de lokale mediaan liggen.
    Vangt occlusie-sprongen die een gewricht kortstondig ergens anders neerzetten.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    mask = np.zeros(n, dtype=bool)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med = np.median(y[lo:hi])
        mad = np.median(np.abs(y[lo:hi] - med))
        if mad > 0 and abs(y[i] - med) > k * 1.4826 * mad:
            mask[i] = True
    return mask


def _interpoleer_onbetrouwbaar(y, betrouwbaar):
    """Vervang niet-betrouwbare posities door lineaire interpolatie uit de rest."""
    y = np.asarray(y, dtype=float).copy()
    idx = np.flatnonzero(betrouwbaar)
    if len(idx) == 0 or len(idx) == len(y):
        return y                      # niets (of alles) betrouwbaar: laat staan
    ontbrekend = np.flatnonzero(~betrouwbaar)
    y[ontbrekend] = np.interp(ontbrekend, idx, y[idx])   # np.interp klemt aan de randen
    return y


def _begrens_interpolatie(betrouwbaar, max_run):
    """
    Laat lange onbetrouwbare reeksen en reeksen aan een segmentrand tóch als
    betrouwbaar gelden (ruwe detectie behouden). Interpolatie repareert korte
    haperingen (occlusie-sprong, één slecht frame) prima, maar over een lange reeks
    is de rechte-lijn-gok slechter dan de detectie zelf, en aan een rand kan
    np.interp alleen vastklemmen — dan bevriest het skelet naast de schaatser.
    """
    b = betrouwbaar.copy()
    n = len(b)
    i = 0
    while i < n:
        if b[i]:
            i += 1
            continue
        j = i
        while j < n and not b[j]:
            j += 1
        if i == 0 or j == n or (j - i) > max_run:
            b[i:j] = True
        i = j
    return b


def smooth_landmarks_offline(resultaten, w, h, window_s=SMOOTH_WINDOW_S,
                             poly=SMOOTH_POLY, fps=30.0):
    """
    Maakt de landmark-trajecten schoon en smooth ze bidirectioneel (zero-lag), per
    aaneengesloten segment van frames-mét-pose (zodat er niet over detectiegaten heen
    wordt geïnterpoleerd). Per gewricht worden eerst onbetrouwbare frames (lage
    zichtbaarheid of occlusie-uitschieter) weg-geïnterpoleerd, daarna Savitzky–Golay
    toegepast. Overschrijft resultaat.lm met de opgeschoonde, gladde landmarks.
    """
    window = max(poly + 2, int(round(window_s * fps)))
    if window % 2 == 0:
        window += 1
    max_run = max(1, int(round(INTERP_MAX_S * fps)))

    # Aaneengesloten segmenten van frames met pose bepalen.
    segmenten, huidig, n_lm = [], [], None
    for i, r in enumerate(resultaten):
        if r.pose_gevonden and r.lm is not None:
            huidig.append(i)
            if n_lm is None:
                n_lm = len(r.lm)
        elif huidig:
            segmenten.append(huidig); huidig = []
    if huidig:
        segmenten.append(huidig)
    if n_lm is None:
        return

    for seg in segmenten:
        if len(seg) < 3:
            continue
        X = np.array([[resultaten[i].lm[j].x for j in range(n_lm)] for i in seg])
        Y = np.array([[resultaten[i].lm[j].y for j in range(n_lm)] for i in seg])
        V = np.array([[resultaten[i].lm[j].visibility for j in range(n_lm)] for i in seg])
        # Onbetrouwbaar = slecht zichtbaar óf een positie-uitschieter (occlusie).
        Bet = np.empty((len(seg), n_lm), dtype=bool)
        for j in range(n_lm):
            b = (V[:, j] >= VIS_MIN)
            b &= ~_hampel_uitschieters(X[:, j])
            b &= ~_hampel_uitschieters(Y[:, j])
            Bet[:, j] = b
        # Blurframe-vangnet: als het merendeel van de data-dragende gewrichten in een
        # frame tegelijk onbetrouwbaar heet, is dat een gecorreleerde confidence-dip
        # (bewegingsonscherpte), geen occlusie van één gewricht. De ruwe detectie zit
        # dan wél op de schaatser — hele frame vertrouwen, niet weg-interpoleren.
        meet = V.max(axis=0) >= VIS_MIN       # gewrichten die überhaupt data dragen
        if meet.any():
            blur = (V[:, meet] >= VIS_MIN).mean(axis=1) < BLUR_FRAME_FRAC
            Bet[blur, :] = True
        for j in range(n_lm):
            betrouwbaar = _begrens_interpolatie(Bet[:, j], max_run)
            xs = _interpoleer_onbetrouwbaar(X[:, j], betrouwbaar)
            ys = _interpoleer_onbetrouwbaar(Y[:, j], betrouwbaar)
            X[:, j] = _savgol(xs, window, poly)
            Y[:, j] = _savgol(ys, window, poly)
        for t, i in enumerate(seg):
            oud = resultaten[i].lm
            resultaten[i].lm = [
                Landmark(float(X[t, j]), float(Y[t, j]),
                         getattr(oud[j], 'z', 0.0), oud[j].visibility)
                for j in range(n_lm)
            ]


def bepaal_horizon_reeks(input_pad, n_frames, fps, force_fps=None, progress_callback=None):
    """
    Detecteert de ijslijn-kanteling **per frame** (voor een schommelende camera) en
    maakt er een stabiel signaal van: elke frame krijgt `detecteer_ijslijn()`, daarna
    worden gaten (geen lijn gevonden) en uitschieters (verkeerd gedetecteerde lijn,
    Hampel) weg-geïnterpoleerd en wordt het geheel over de tijd gesmoothd (Savitzky–
    Golay, `HORIZON_SMOOTH_S`). Zo volgt de horizon de trage schommeling zonder de
    per-frame Hough-jitter. Retourneert een lijst van lengte `n_frames` met graden.

    Draait als losse video-pass (backend-onafhankelijk); de decode-kosten vallen weg
    tegen de pose-detectie. Bij géén enkele betrouwbare lijn: alles 0.0.
    """
    cap = cv2.VideoCapture(input_pad)
    ruw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ruw.append(detecteer_ijslijn(frame))     # float of None
        if progress_callback is not None:
            progress_callback(len(ruw), n_frames)
    cap.release()

    # Lengte gelijktrekken aan de resultatenlijst.
    if len(ruw) < n_frames:
        ruw += [None] * (n_frames - len(ruw))
    else:
        ruw = ruw[:n_frames]

    betrouwbaar = np.array([v is not None for v in ruw])
    if not betrouwbaar.any():
        return [0.0] * n_frames

    y = np.array([v if v is not None else np.nan for v in ruw], dtype=float)
    y = _interpoleer_onbetrouwbaar(y, betrouwbaar)          # vul detectiegaten
    betr2 = betrouwbaar & ~_hampel_uitschieters(y)          # gooi foute lijnen eruit
    y = _interpoleer_onbetrouwbaar(y, betr2)
    window = max(SMOOTH_POLY + 2, int(round(HORIZON_SMOOTH_S * fps)))
    y = _savgol(y, window, SMOOTH_POLY)
    return [round(float(v), 2) for v in y]


def wijs_afzetbeen_cyclus(resultaten, h, fps):
    """
    Wijst per pose-frame het afzetbeen (standbeen) toe op basis van de schaatscyclus
    i.p.v. per frame onafhankelijk. Signaal = enkel-hoogteverschil (links − rechts;
    hogere y = lager in beeld = dichter bij ijs), dus positief → linkerenkel op ijs →
    links is standbeen. Dat signaal wordt gesmoothd en met **hysterese** omgezet naar
    een been, zodat het alleen wisselt bij een echte gewichtsoverdracht.

    Werkt per aaneengesloten segment van pose-frames (niet over detectiegaten heen);
    de hysterese-toestand blijft binnen een segment behouden.
    """
    # Aaneengesloten segmenten van frames met lm_data.
    segmenten, huidig = [], []
    for i, r in enumerate(resultaten):
        if r.pose_gevonden and r.lm_data is not None:
            huidig.append(i)
        elif huidig:
            segmenten.append(huidig); huidig = []
    if huidig:
        segmenten.append(huidig)

    win = max(SMOOTH_POLY + 2, int(round(STANCE_SMOOTH_S * fps)))
    if win % 2 == 0:
        win += 1

    for seg in segmenten:
        sig = np.array([(resultaten[i].lm_data['l_enkel'][1]
                         - resultaten[i].lm_data['r_enkel'][1]) / h for i in seg])
        sig_s = _savgol(sig, win, SMOOTH_POLY) if len(seg) >= 3 else sig
        band = max(1e-4, STANCE_BAND_FRAC * float(np.percentile(np.abs(sig_s), 80)))

        staat = 'links' if (len(sig_s) and sig_s[0] >= 0) else 'rechts'
        for t, i in enumerate(seg):
            v = sig_s[t]
            if v > band:
                staat = 'links'
            elif v < -band:
                staat = 'rechts'
            resultaten[i].been = staat


def _lm_px(r, idx, w, h):
    """Float-pixelcoördinaten van landmark `idx` — géén int-truncatie zoals in
    `get_landmarks`: op grote afstand is het onderbeen maar tientallen pixels en
    kost een hele-pixel-afronding al gauw enkele graden in de 3D-reconstructie."""
    return (r.lm[idx].x * w, r.lm[idx].y * h)


def _wereldtraject(resultaten, perspectief, fps, w, h):
    """
    Wereldposities (m) + rijrichting per frame uit de gekalibreerde enkelposities.

    Positie = de enkel van het stándbeen (die staat op het ijs; de zweefbeen-enkel
    hangt tientallen cm hoger en zou de positie systematisch vertekenen), via de
    kijkstraal op het vlak `enkel_hoogte` boven het ijs geprikt. De richting is de
    verplaatsing over een venster van `RIJRICHTING_VENSTER_S`; is die te klein om te
    vertrouwen (stilstand, gat), dan valt hij terug op de baanlijn-richting (wereld-y)
    met het teken van de netto verplaatsing — de baanlijnen zíjn immers de rijrichting.
    """
    kal = perspectief.kalibratie
    n = len(resultaten)
    pos = np.full((n, 2), np.nan)
    for i, r in enumerate(resultaten):
        if r.lm_data is None:
            continue
        if r.been in ('links', 'rechts'):
            e_idx = L_ANKLE if r.been == 'links' else R_ANKLE
        else:                             # geen cyclus-toewijzing: laagste enkel in beeld
            e_idx = L_ANKLE if r.lm[L_ANKLE].y >= r.lm[R_ANKLE].y else R_ANKLE
        P = schaats_perspectief.punt_op_ijs(kal, _lm_px(r, e_idx, w, h),
                                            hoogte=perspectief.enkel_hoogte)
        if P is not None:
            pos[i] = P[:2]

    geldig = ~np.isnan(pos[:, 0])
    idx = np.where(geldig)[0]
    richting = np.zeros((n, 2))
    if len(idx) < 2:
        richting[:] = (0.0, 1.0)
        return pos, richting

    # gaten dichtinterpoleren voor een doorlopend traject (alleen voor de richting)
    vol = np.column_stack([np.interp(np.arange(n), idx, pos[idx, k]) for k in (0, 1)])
    netto = vol[idx[-1]] - vol[idx[0]]
    basis = np.array([0.0, 1.0 if netto[1] >= 0 else -1.0])
    k = max(1, int(round(RIJRICHTING_VENSTER_S * fps / 2)))
    for i in range(n):
        d = vol[min(i + k, n - 1)] - vol[max(i - k, 0)]
        lengte = float(np.hypot(d[0], d[1]))
        richting[i] = d / lengte if lengte >= RIJRICHTING_MIN_M else basis
    return pos, richting


def _perspectief_hoek(r, enkel_px, knie_px, perspectief, richting):
    """
    Gecorrigeerde afzethoek voor één frame via de 3D-reconstructie. Vult op `r` ook
    `hoek_correctie` (t.o.v. de oude meting: beeldvlak-hoek minus horizonaftrek) en
    `hoek_betrouwbaar`. Valt bij een mislukte reconstructie (enkel boven de horizon —
    kan alleen bij een ontspoorde detectie) terug op de oude beeldvlak-meting.
    """
    rec = schaats_perspectief.reconstrueer_hoek(
        perspectief.kalibratie, enkel_px, knie_px,
        methode=perspectief.methode, onderbeen_l=perspectief.onderbeen_l,
        vlak_richting=richting if perspectief.methode == 'beenvlak' else None,
        rijrichting=richting, enkel_hoogte=perspectief.enkel_hoogte)
    oude_hoek = bereken_hoek_tov_ijs(enkel_px, knie_px, r.horizon_deg)
    if rec is None:
        r.hoek_correctie = None
        r.hoek_betrouwbaar = False
        return oude_hoek
    hoek = round(rec.hoek, 1)
    r.hoek_correctie = round(hoek - oude_hoek, 1)
    r.hoek_betrouwbaar = rec.betrouwbaar
    return hoek


def verwerk_afgeleiden(resultaten, w, h, fps, smooth_n=5, threshold=0.015, cyclus=True,
                       perspectief=None):
    """
    Vult per frame de afgeleide grootheden in (afzetbeen, hoek, kniehoek, gewicht),
    berekend uit resultaat.lm. Wordt ná de offline smoothing aangeroepen zodat alles
    op de gladde landmarks is gebaseerd.

    Het afzetbeen wordt cyclus-bewust toegewezen (`wijs_afzetbeen_cyclus`) als
    `cyclus`, anders per frame (`bepaal_afzetbeen`). De rest is sequentieel i.v.m. de
    tijd-histories; bij een detectiegat worden die geleegd. De afzethoek wordt
    gecorrigeerd met de per-frame `r.horizon_deg` (door `analyseer` gezet: constant of
    per-frame bij auto-horizon).

    Met `perspectief` (PerspectiefConfig) vervangt de 3D-reconstructie de beeldvlak-
    hoek: de echte hoek t.o.v. het ijsvlak komt in `r.hoek`, de toegepaste correctie
    in `r.hoek_correctie` en de kwaliteitsvlag in `r.hoek_betrouwbaar`. Bijvangst:
    `r.wereld_xy` en (met bekende schaal) `r.snelheid`. De horizonaftrek vervalt dan
    voor de hóek (de kalibratie kent de camerastand exact); `r.horizon_deg` blijft de
    ware-horizonkanteling voor de overlay.
    """
    # Pass 1: pixelcoördinaten voor alle pose-frames.
    for r in resultaten:
        r.lm_data = get_landmarks(r.lm, w, h) if (r.pose_gevonden and r.lm is not None) else None

    # Been-toewijzing (globaal, cyclus-bewust) vóór de per-frame afgeleiden.
    if cyclus:
        wijs_afzetbeen_cyclus(resultaten, h, fps)

    # Wereldtraject + snelheid (alleen mét kalibratie).
    if perspectief is not None:
        posities, richtingen = _wereldtraject(resultaten, perspectief, fps, w, h)
        k = max(1, int(round(RIJRICHTING_VENSTER_S * fps / 2)))
        for i, r in enumerate(resultaten):
            if np.isnan(posities[i, 0]):
                continue
            r.wereld_xy = (float(posities[i, 0]), float(posities[i, 1]))
            if perspectief.kalibratie.schaal_bekend:
                j0, j1 = max(i - k, 0), min(i + k, len(resultaten) - 1)
                d = posities[j1] - posities[j0]
                if not np.isnan(d[0]) and j1 > j0:
                    r.snelheid = round(float(np.hypot(d[0], d[1])) * fps / (j1 - j0), 2)

    # Pass 2: hoek, kniehoek en gewicht per frame.
    hoek_buffer = deque(maxlen=smooth_n)
    enkel_hist  = {'links': deque(maxlen=10), 'rechts': deque(maxlen=10)}
    heup_hist   = deque(maxlen=10)

    for i, r in enumerate(resultaten):
        if not (r.pose_gevonden and r.lm_data is not None):
            hoek_buffer.clear(); heup_hist.clear()
            enkel_hist['links'].clear(); enkel_hist['rechts'].clear()
            continue

        lm_data = r.lm_data
        enkel_hist['links'].append(lm_data['l_enkel'])
        enkel_hist['rechts'].append(lm_data['r_enkel'])
        heup_hist.append((lm_data['l_heup'][0], lm_data['r_heup'][0]))

        been = r.been if cyclus else bepaal_afzetbeen(lm_data, heup_hist, w)
        if perspectief is not None:
            # float-pixels uit r.lm (niet de int-getrunceerde lm_data): op afstand
            # kost hele-pixel-afronding al gauw enkele graden in de reconstructie
            e_idx, k_idx = (L_ANKLE, L_KNEE) if been == 'links' else (R_ANKLE, R_KNEE)
            hoek = _perspectief_hoek(r, _lm_px(r, e_idx, w, h), _lm_px(r, k_idx, w, h),
                                     perspectief, richtingen[i])
        elif been == 'links':
            hoek = bereken_hoek_tov_ijs(lm_data['l_enkel'], lm_data['l_knie'], r.horizon_deg)
        else:
            hoek = bereken_hoek_tov_ijs(lm_data['r_enkel'], lm_data['r_knie'], r.horizon_deg)
        hoek_buffer.append(hoek)
        smooth_hoek = round(float(np.mean(hoek_buffer)), 1)

        gewicht_erop, kniehoek, signalen = detecteer_gewicht_op_been(
            been, lm_data, enkel_hist, heup_hist, w, h, threshold)

        r.been = been
        r.hoek = hoek
        r.smooth_hoek = smooth_hoek
        r.kniehoek = kniehoek
        r.gewicht_erop = gewicht_erop
        r.signalen = signalen


def fase_voortgang(progress_callback, fase, n_fasen):
    """
    Wikkelt een `progress_callback(i, totaal)` zó dat meerdere video-passes samen één
    doorlopende balk vormen: pass `fase` (0-based) van `n_fasen` mapt op zijn eigen
    schijf [fase/n_fasen, (fase+1)/n_fasen]. None als er geen callback is.
    """
    if progress_callback is None:
        return None
    return lambda i, totaal: progress_callback(fase * totaal + i, n_fasen * totaal)


def analyseer(input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
              num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
              progress_callback=None, horizon_deg=0.0, auto_horizon=False,
              perspectief=None):
    """
    Volledige analyse-pijplijn: multi-pose detectie + doel-tracking (streaming),
    daarna offline landmark-smoothing en het berekenen van de afgeleide grootheden.
    Retourneert (VideoInfo, lijst[FrameResultaat]).

    De horizoncorrectie is óf een vaste `horizon_deg` (handmatige lijn), óf — bij
    `auto_horizon` — **per frame** gedetecteerd (`bepaal_horizon_reeks`, voor een
    schommelende camera). In beide gevallen komt de waarde per frame in `r.horizon_deg`.
    Bij auto is er een tweede video-pass; die deelt de voortgangsbalk met de detectie
    (elk een helft) via `fase_voortgang`, zodat het één doorlopende balk blijft.

    Met `perspectief` (PerspectiefConfig) vervangt de baanlijn-kalibratie de horizon-
    machinerie volledig (zie `zet_horizon`/`verwerk_afgeleiden`).
    """
    info = video_info(input_pad, force_fps)
    if perspectief is not None:
        auto_horizon = False     # vaste camera per aanname; kalibratie kent de kanteling al

    # Auto-horizon = twee passes → één doorlopende balk (detectie 0–50%, horizon 50–100%).
    det_cb = fase_voortgang(progress_callback, 0, 2) if auto_horizon else progress_callback
    hor_cb = fase_voortgang(progress_callback, 1, 2) if auto_horizon else progress_callback

    resultaten = list(analyseer_frames(
        input_pad, model_pad, force_fps=force_fps, num_poses=num_poses,
        doel_punt=doel_punt, progress_callback=det_cb))

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)

    zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten


def zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps=None,
                progress_callback=None, perspectief=None):
    """
    Vult `r.horizon_deg` per frame: per-frame gedetecteerd bij `auto_horizon`
    (`bepaal_horizon_reeks`), anders de constante `horizon_deg` overal. Gedeeld door
    de MediaPipe- en YOLO-backend, zodat de per-frame horizon backend-onafhankelijk is.

    Met `perspectief` komt de kanteling uit de kalibratie zelf (de ware horizon =
    verdwijnlijn van het ijsvlak); auto-horizon is dan niet van toepassing (vaste
    camera). De hoek gebruikt `horizon_deg` dan overigens niet meer (de reconstructie
    kent de camerastand exact) — dit stuurt alleen nog de getekende ijslijn.
    """
    if perspectief is not None:
        hz = perspectief.kalibratie.horizon_deg
        for r in resultaten:
            r.horizon_deg = hz
        return
    if auto_horizon:
        horizons = bepaal_horizon_reeks(input_pad, len(resultaten), info.fps,
                                        force_fps=force_fps, progress_callback=progress_callback)
        for r, hz in zip(resultaten, horizons):
            r.horizon_deg = hz
    else:
        for r in resultaten:
            r.horizon_deg = horizon_deg


def _merge_events(a, b):
    """Voeg twee gelijk-been events samen tot één afzet (a vóór b)."""
    snelheden = [v for v in (a.snelheid, b.snelheid) if v is not None]
    slagen = [s for s in (a.slaglengte, b.slaglengte) if s is not None]
    return AfzetEvent(
        index=a.index, been=a.been,
        start_frame=a.start_frame, eind_frame=b.eind_frame,
        start_tijd=a.start_tijd, eind_tijd=b.eind_tijd,
        hoek=b.hoek,                              # hoek bij afzet-voltooiing = laatste
        min_hoek=min(a.min_hoek, b.min_hoek),
        max_hoek=max(a.max_hoek, b.max_hoek),
        opmerking="samengevoegd",
        correctie=b.correctie, betrouwbaar=a.betrouwbaar and b.betrouwbaar,
        snelheid=round(float(np.mean(snelheden)), 2) if snelheden else None,
        slaglengte=round(float(np.sum(slagen)), 2) if slagen else None,
    )


def forceer_alternerend(events, resultaten, merge_gap_s=0.35, min_tegen=2):
    """
    Feedback op basis van de schaats-regel 'afzetten wisselen altijd links-rechts':
    twee gelijk-been events achter elkaar is onmogelijk. Per zo'n paar onderscheiden we:

    - **Opgesplitste afzet** — in de tussenruimte was het andere been níet het afzetbeen
      (puur een detectie-dip) én het gat is klein → de twee events worden **samengevoegd**.
    - **Gemiste tegen-afzet** — het andere been wás kort afzetbeen in de tussenruimte (die
      korte afzet is door `min_lengte` weggefilterd) óf het gat is groot → we **markeren**
      het event met "gemiste tegenafzet?" i.p.v. het weg te poetsen, zodat jij het ziet.

    Retourneert een nieuwe, opnieuw geïndexeerde eventlijst.
    """
    if not events:
        return events

    per_frame = {r.frame_nr: r for r in resultaten}
    uit = [events[0]]
    for ev in events[1:]:
        vorige = uit[-1]
        if ev.been != vorige.been:
            uit.append(ev)
            continue

        tegen = 'rechts' if ev.been == 'links' else 'links'
        tegen_frames = sum(
            1 for f in range(vorige.eind_frame + 1, ev.start_frame)
            if f in per_frame and per_frame[f].been == tegen
        )
        gat = ev.start_tijd - vorige.eind_tijd

        if tegen_frames < min_tegen and gat <= merge_gap_s:
            uit[-1] = _merge_events(vorige, ev)          # opgesplitste afzet → samenvoegen
        else:
            ev.opmerking = "gemiste tegenafzet?"         # markeren, niet samenvoegen
            uit.append(ev)

    for i, ev in enumerate(uit):
        ev.index = i
    return uit


def segmenteer_afzetten(resultaten, min_lengte=3, alternerend=True):
    """
    Groepeert per-frame resultaten tot afzet-events: aaneengesloten frames
    waarin hetzelfde been afzetbeen is én het gewicht er nog op zit.
    `min_lengte` filtert ruis (te korte, onbetrouwbare detecties) eruit.
    Als `alternerend`, wordt daarna de L/R-alternatie-regel toegepast
    (`forceer_alternerend`): opgesplitste afzetten samenvoegen, onmogelijke
    herhalingen markeren.
    """
    events = []
    huidig = None

    def _sluit_af():
        if huidig is not None and len(huidig['hoeken']) >= min_lengte:
            start, laatste, hoeken = huidig['start'], huidig['laatste'], huidig['hoeken']
            frames = huidig['frames']
            # Perspectief-bijvangst (None zonder kalibratie): correctie + vlag van het
            # voltooiingsframe, snelheid gemiddeld over de afzet, slaglengte = afgelegde
            # afstand tussen start- en eindpositie op de baan.
            snelheden = [f.snelheid for f in frames if f.snelheid is not None]
            slaglengte = None
            if start.wereld_xy is not None and laatste.wereld_xy is not None:
                slaglengte = round(float(np.hypot(
                    laatste.wereld_xy[0] - start.wereld_xy[0],
                    laatste.wereld_xy[1] - start.wereld_xy[1])), 2)
            events.append(AfzetEvent(
                index=len(events),
                been=huidig['been'],
                start_frame=start.frame_nr,
                eind_frame=laatste.frame_nr,
                start_tijd=start.tijd,
                eind_tijd=laatste.tijd,
                hoek=hoeken[-1],
                min_hoek=min(hoeken),
                max_hoek=max(hoeken),
                correctie=laatste.hoek_correctie,
                betrouwbaar=laatste.hoek_betrouwbaar,
                snelheid=round(float(np.mean(snelheden)), 2) if snelheden else None,
                slaglengte=slaglengte,
            ))

    for r in resultaten:
        actief = r.pose_gevonden and r.gewicht_erop
        if actief:
            if huidig is None or huidig['been'] != r.been:
                _sluit_af()
                huidig = {'been': r.been, 'start': r, 'laatste': r, 'hoeken': [], 'frames': []}
            huidig['hoeken'].append(r.smooth_hoek)
            huidig['frames'].append(r)
            huidig['laatste'] = r
        else:
            _sluit_af()
            huidig = None

    _sluit_af()
    if alternerend:
        events = forceer_alternerend(events, resultaten)
    return events


def analyseer_video(input_pad, output_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                    num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
                    horizon_deg=0.0, auto_horizon=False):
    """
    CLI-analyse in twee passes: eerst detecteren/tracken/smoothen (nodig omdat de
    offline smoothing álle frames vereist), daarna de video opnieuw lezen en de
    overlay erop tekenen en wegschrijven.
    """
    def toon_voortgang(frame_nr, totaal):
        if frame_nr % 30 == 0:
            pct = frame_nr / totaal * 100 if totaal > 0 else 0
            print(f"  {frame_nr}/{totaal} frames ({pct:.0f}%)")

    print("[INFO] Pass 1/2: detectie + tracking + smoothing ...")
    info, resultaten = analyseer(
        input_pad, model_pad, smooth_n, threshold, force_fps,
        num_poses=num_poses, doel_punt=doel_punt, smooth_landmarks=smooth_landmarks,
        progress_callback=toon_voortgang, horizon_deg=horizon_deg, auto_horizon=auto_horizon)

    print(f"[INFO] Video: {info.w}×{info.h} @ {info.fps:.1f}fps, {info.totaal} frames")
    print(f"[INFO] Pass 2/2: overlay tekenen → {output_pad}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_pad, fourcc, info.fps, (info.w, info.h))
    cap    = cv2.VideoCapture(input_pad)
    idx    = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx < len(resultaten):
            teken_overlay_op_frame(frame, resultaten[idx], info.fps)
        out.write(frame)
        idx += 1
        if idx % 30 == 0:
            print(f"  {idx}/{info.totaal} frames getekend")
    cap.release()
    out.release()

    events = segmenteer_afzetten(resultaten)
    print(f"\n[KLAAR] Resultaat opgeslagen: {output_pad}")
    print(f"        {idx} frames verwerkt, {len(events)} afzetten gevonden")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schaatser afzethoek analyse")
    parser.add_argument("--input", required=False, default=None, help="...")
    parser.add_argument("--output",    default="output.mp4",  help="Uitvoervideo")
    parser.add_argument("--model",     default=None,
                        help="Pad naar pose_landmarker .task modelbestand")
    parser.add_argument("--fps",       type=float, default=None, help="Forceer FPS")
    parser.add_argument("--smooth",    type=int,   default=5,    help="Smoothing frames")
    parser.add_argument("--threshold", type=float, default=0.015,
                        help="Gevoeligheid gewichtsdetectie (0.01–0.03)")
    parser.add_argument("--num-poses", type=int, default=NUM_POSES_DEFAULT,
                        help="Max. aantal gelijktijdig te detecteren schaatsers")
    parser.add_argument("--heavy", action="store_true",
                        help="Gebruik het heavy-model (nauwkeuriger, trager)")
    parser.add_argument("--target", default=None, metavar="X,Y",
                        help="Genormaliseerd startpunt (0-1) van de te volgen schaatser, "
                             "bv. 0.5,0.4; standaard: grootste schaatser")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Offline landmark-smoothing (Savitzky–Golay) uitzetten")
    parser.add_argument("--horizon", type=float, default=0.0, metavar="GRADEN",
                        help="Kanteling van de ijslijn t.o.v. de horizontale beeldas "
                             "(positief = naar rechts omhoog); corrigeert een scheve camera. "
                             "Standaard 0 (ijs = horizontaal).")
    parser.add_argument("--auto-horizon", action="store_true",
                        help="Detecteer de ijslijn-kanteling automatisch, per frame (voor een "
                             "schommelende camera); overschrijft --horizon.")
    args = parser.parse_args()

    # Als er geen --input is meegegeven, vraag er interactief om
    if not args.input:
        args.input = input("Welke video wil je analyseren? (pad naar bestand): ").strip().strip('"')

    # Check of het bestand echt bestaat
    if not os.path.isfile(args.input):
        print(f"Bestand niet gevonden: {args.input}")
        exit(1)

    standaard_naam = "pose_landmarker_heavy.task" if args.heavy else "pose_landmarker_full.task"
    model_pad = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), standaard_naam)
    if not os.path.isfile(model_pad):
        variant = "heavy" if args.heavy else "full"
        print(f"Modelbestand niet gevonden: {model_pad}")
        print(f"Download het via: https://storage.googleapis.com/mediapipe-models/"
              f"pose_landmarker/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")
        exit(1)

    doel_punt = None
    if args.target:
        try:
            dx, dy = (float(v) for v in args.target.split(","))
            doel_punt = (dx, dy)
        except ValueError:
            print(f"Ongeldig --target: {args.target!r} (verwacht bv. 0.5,0.4)")
            exit(1)

    analyseer_video(
        input_pad=args.input,
        output_pad=args.output,
        model_pad=model_pad,
        smooth_n=args.smooth,
        threshold=args.threshold,
        force_fps=args.fps,
        num_poses=args.num_poses,
        doel_punt=doel_punt,
        smooth_landmarks=not args.no_smooth,
        horizon_deg=args.horizon,
        auto_horizon=args.auto_horizon,
    )
