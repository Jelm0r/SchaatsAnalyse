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
import sys
import cv2
import numpy as np
import argparse
from collections import deque, namedtuple
from dataclasses import dataclass, field

import schaats_perspectief   # puur numpy — veilig in beide venvs


# ── Waar staan de bestanden? ────────────────────────────────────────────────────
# Deze drie hoorden hier (de laagste gedeelde module: schaats_gui, schaats_yolo én
# schaats_db importeren er alle drie uit), maar staan nu in schaats_omgeving.py — dat is
# stdlib-only en dus laadbaar vóór het opstartscherm, waar `data_dir()` al nodig is om de
# uitvoer om te leiden (EXE.md stap 2) terwijl deze module juist cv2+numpy binnentrekt.
# Ze worden hier doorgegeven, zodat elke bestaande import ongewijzigd blijft werken.
from schaats_omgeving import is_bevroren, app_dir, data_dir     # noqa: F401


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
TRACK_GATE_GROEI  = 0.25  # ... die per gemist frame groeit: na een gat is de voorspelling
                          # onzekerder (de snelheidsschatting veroudert mee)
TRACK_GATE_MAX    = 0.25  # bovengrens van de gegroeide poort — daarboven is "de dichtstbij"
                          # geen bewijs meer dat het dezelfde schaatser is
TRACK_HERSEED_GATE = 0.25 # bij herseeden na langdurig verlies: max. afstand tot de
                          # (geëxtrapoleerde) laatst bekende plek. Ruim t.o.v. TRACK_GATE
                          # (~8 frames rijden) maar niet ruimer, want de voorspelling is
                          # al mee-geëxtrapoleerd. Bewust gelijk aan TRACK_GATE_MAX, zodat de
                          # acceptatie bij de overgang coast→herseed niet verspringt. Ligt er
                          # niemand binnen deze poort, dan
                          # liever géén pose dan een skelet op de verkeerde persoon (dat
                          # levert plausibele maar onjuiste hoeken op).
TRACK_HERVIND_S   = 0.5   # zolang de doelschaatser kwijt is voordat we opnieuw seeden (s)
TRACK_MIN_VIS     = 0.3   # minimale zichtbaarheid om een landmark mee te tellen
# Koude start zónder muisklik: niet blind "de grootste pose" nemen — een omstander langs
# de boarding is in beeld geregeld groter dan de schaatser die verder weg rijdt. We kijken
# eerst een seconde mee en kiezen dan de grootste *beweger* (mediane bbox × afgelegde weg),
# net als de YOLO-backend met MIN_VERPLAATSING doet.
SEED_WARMUP_S     = 1.0   # zolang kijken we mee voordat de doelschaatser gekozen wordt (s)
SEED_MIN_VERPLAATSING = 0.02  # kortere afgelegde weg in dat venster = statische omstander
TORSO_IDX         = (11, 12, 23, 24)  # schouders + heupen: stabiele identiteits-centroïde

# ── Offline landmark-smoothing (Savitzky–Golay) ─────────────────────────────────
# Omdat dit een batch-tool is en we álle frames hebben, smoothen we bidirectioneel
# (zero-lag) i.p.v. causaal. SG onderdrukt jitter maar behoudt pieken (bv. volledige
# strekking bij afzet) beter dan een gewoon voortschrijdend gemiddelde.
SMOOTH_WINDOW_S = 0.25    # vensterlengte in seconden (wordt omgezet naar oneven # frames)
SMOOTH_POLY     = 2       # polynoomorde van de SG-fit
HORIZON_SMOOTH_S = 0.5    # vensterlengte (s) voor het smoothen van de per-frame horizon-schatting
# Ruimte die de schaatser per frame in beeld inneemt (voor de automatische zoom in de GUI).
# De omvang golft mee met de schaatscyclus (door de knieën zakken, benen spreiden), dus
# smoothen we over ruwweg één slag — wat overblijft is de trage verandering van de afstand
# tot de camera.
KADER_SMOOTH_S   = 1.5    # vensterlengte (s) voor het smoothen van de kadergrootte
KADER_MIDDEN_S   = 0.5    # vensterlengte (s) voor het smoothen van het kader-middelpunt;
                          # korter, want het midden moet de schaatser echt volgen — alleen
                          # het meebewegen met armen en benen moet eruit
KADER_POLY       = 1      # lineair, niet SMOOTH_POLY: de afstand tot de camera verandert
                          # lokaal recht-toe-recht-aan, terwijl de kwadratische randfit van
                          # SG de cyclus-golf naar buiten toe extrapoleert — gemeten 15% mis
                          # op het eerste frame tegen 3% met een lineaire fit
KADER_MIN_FRAMES = 5      # minder bruikbare frames = geen zinnig kader-signaal
KADER_GAT_S      = 1.0    # zolang mag een detectiegat overbrugd worden met de laatst bekende
                          # kadering; duurt het langer, dan weten we niet waar de schaatser
                          # is en zoomt het kader vloeiend terug naar het volledige beeld —
                          # liever alles zien dan een uitvergroting van de verkeerde plek

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

# ── Voorspelbaarheids-checks (schaatsen is een voorspelbare beweging) ───────────
# 1. L/R-verwisseling: bij kruisende/overlappende benen hangt het pose-model links
#    en rechts geregeld verkeerd om. Dat vangt Hampel alleen kortstondig — een
#    aanhoudende verwisseling vergiftigt de afzetbeen-cyclus (l−r-enkelsignaal).
#    We volgen daarom per gewrichtspaar de continuïteit van de twee trajecten en
#    kiezen per frame de toewijzing met de kleinste sprong.
LR_SWAP_FACTOR = 0.8      # wissel alleen als de gewisselde toewijzing dúidelijk beter
                          # past (kostenratio); voorkomt geflikker als de benen kruisen
# 2. Botlengte: onder- en bovenbeen zijn star — hun pixellengte hoort alleen traag
#    te veranderen (afstand tot de camera). Een plotse lengtesprong is een
#    detectiefout, óók als x en y elk binnen hun eigen Hampel-band blijven.

# ── Cyclus-bewuste afzetbeen-bepaling ───────────────────────────────────────────
# Het enkel-hoogteverschil (links vs rechts) oscilleert met de schaatsslag. We
# smoothen dat signaal en passen het met hysterese toe, zodat het standbeen alleen
# wisselt bij een echte gewichtsoverdracht (geen per-frame geflikker rond de wissel).
STANCE_SMOOTH_S = 0.18    # smoothing-venster (s) van het stand-signaal
STANCE_BAND_FRAC = 0.20   # hysterese-band als fractie van de signaalamplitude

# ── Afzet-voltooiing uit beenstrekking ──────────────────────────────────────────
# "Been maximaal gestrekt = afzet klaar." I.p.v. een per-frame 2-van-3-stem met vaste
# drempels detecteren we het strek-máximum als een piek van een glad, schaalvrij
# signaal (rechte-lijn heup→enkel / som botlengtes). Geen magische drempel nodig en de
# hoek wordt precies op de piek afgelezen. De slagtijd is voorspelbaar: uit de mediane
# stand-run-lengte schatten we de halve slagperiode als zachte prior tegen spookafzetten.
STREK_SMOOTH_S      = 0.15   # smoothing-venster (s) van het strek-ratio-signaal
STREK_MIN_SLAG_FRAC = 0.35   # een stand-run korter dan deze fractie van de halve slag-
                             # periode is (bijna zeker) een ruis-omslag → geen afzet.
                             # Laag gehouden zodat snelle openingsslagen blijven staan.
STREK_MIN_RUN_S     = 0.20   # absolute ondergrens (s) voor een stand-run die als échte slag
                             # meetelt. Nodig omdat de halve slagperiode uit de mediane run-
                             # lengte komt: tellen de ruis-runs daarin mee, dan zakt de mediaan
                             # — en dus de drempel — juist bij véél L/R-flips (zie BUGS.md A3).
                             # Een halve schaatsslag duurt nooit minder dan ~0,2 s.
STREK_PLATEAU_BAND  = 0.02   # het been is een hele fase "gestrekt" (van rechtop komen tot
                             # volle zijwaartse push). Frames waarin de strek-ratio binnen
                             # deze band onder het per-run maximum zit tellen als "gestrekt";
                             # binnen dat plateau lezen we de vlakste (laagste) onderbeenhoek
                             # af = de eigenlijke afzethoek (niet het rechtop-komen).
STREK_MIN_HELLING_DEG = 20.0 # minimale helling van het onderbeen t.o.v. de vertícaal bij
                             # afzet-voltooiing (dus: afzethoek ≤ 90 − 20 = 70°). Beslaat het
                             # strek-plateau van een run alléén de opricht-fase, dan is de
                             # "vlakste hoek" binnen dat plateau nog steeds het rechtop-komen
                             # en rapporteert de tool een afzet van 74–83°. Dit is geen
                             # tuning-getal maar meetkunde: bij een onderbeen dat maar 20°
                             # uit het lood staat is de zijwaartse component van de afzet
                             # sin(20°) ≈ 0,34 — er is dan simpelweg niet opzij geduwd.
                             # Op de 18 opgeslagen analyses scheidt deze grens de drie
                             # rechtop-kom-events (74,3 / 79,4 / 83,1°) van alle 84 gezonde
                             # afzetten; de steilste gezonde meting staat op 66,7°.
                             # Zo'n event verdwijnt niet — het wordt gemarkeerd (zie
                             # ONV_GEEN_PUSH) en valt buiten gem/min/max.

# Redenen waarom een afzet wél zichtbaar blijft maar buiten de statistiek valt
# (`FrameResultaat.afzet_onvolledig` → `AfzetEvent.onvolledig`). Bewust één vlag mét
# reden i.p.v. twee losse booleans: elke plek die de statistiek filtert hoeft alleen op
# waarheid te toetsen, terwijl de GUI de gebruiker kan vertellen wát er mis was — "de
# video hield op" vraagt om een langere opname, "geen volledige afzet waargenomen" om
# een kritische blik op de been-toewijzing. De teksten zijn tegelijk de marker die in
# de events-cache van de bibliotheek meereist (schaats_db).
ONV_AFGEKAPT  = "afgekapt"              # run loopt door tot het einde van video/pose-segment
ONV_GEEN_PUSH = "geen volledige push"   # plateau beslaat alleen het rechtop-komen

# ── Bochtdetectie ───────────────────────────────────────────────────────────────
# In de bocht draait het lichaam om de verticale as: de heupen staan niet langer naast
# elkaar maar achter elkaar, dus hun horizontale afstand in beeld stort in terwijl de
# romp even lang blijft. `bocht_ratio` = heupbreedte / romplengte is daarmee een
# schaalvrij "sta ik frontaal in beeld"-signaal (zelfde principe als `_strek_ratio`):
# onafhankelijk van de afstand tot de camera, en juist gevoelig voor precies de rotatie
# die de bocht maakt. Gemeten over de 22 analyses in de bibliotheek:
#   - 16 frontale clips (recht stuk): mediaan 0,75–1,20, laagste 0,5 s-mediaan 0,57
#   - bochtdeel van vier lange clips: mediaan 0,21–0,24, laagste 0,05
# Marge dus ruim 3×. Alternatieve noemers (femur, heel been, schouderbreedte) gaven
# allemaal minder scheiding (1,7–2,7×).
BOCHT_IN        = 0.40   # onder deze (gesmoothte) ratio: bocht — ruim onder 0,57
BOCHT_UIT       = 0.50   # boven deze ratio weer recht stuk (hysterese, zoals STANCE_BAND_FRAC)
BOCHT_SMOOTH_S  = 0.5    # smoothing-venster (s); de ratio golft licht mee met de slag
BOCHT_MIN_S     = 0.6    # korter dan dit is geen bocht maar ruis → laten staan
BOCHT_MIN_TORSO_PX = 12  # onder deze romplengte is de ratio pixelruis (de verste schaatser
                         # in de bibliotheek meet 25–35 px)

# ── Perspectiefcorrectie (fase 7) ───────────────────────────────────────────────
# De 3D-reconstructie zelf zit in schaats_perspectief.py; hier alleen de koppeling.
ENKEL_HOOGTE_M       = 0.10   # enkel-landmark ligt op malleolus + schaats, niet óp het ijs
RIJRICHTING_VENSTER_S = 0.4   # venster (s) voor de traject-richting uit wereldposities
RIJRICHTING_MIN_M     = 0.15  # minimale verplaatsing in het venster om de richting te vertrouwen

# ── Landmark indices (MediaPipe Pose) ──────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
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
    smooth_hoek: float = None   # gecentreerd gemiddelde van `hoek` (weergave; zero-lag)
    kniehoek: float = None
    gewicht_erop: bool = None
    signalen: list = field(default_factory=list)
    strek_ratio: float = None   # strekking standbeen (0–1, ~1 = gestrekt); piek = einde afzet
    afzet_onvolledig: str = None  # None = volwaardige afzet; anders de reden waarom de hoek
                                  # van deze stand-run niet als meting telt (ONV_AFGEKAPT /
                                  # ONV_GEEN_PUSH) — beide leveren een te steile hoek
    pose_gevonden: bool = False
    bocht: bool = False         # de schaatser staat hier niet frontaal in beeld (bocht) — er
                                # worden geen afgeleiden berekend, dus dit frame levert geen
                                # afzetmeting. Bij de YOLO-backend zijn dit tevens de frames
                                # waarvan de detectiepass de inferentie heeft overgeslagen.
    horizon_deg: float = 0.0    # camerakanteling t.o.v. het ijs bij dit frame (per-frame bij auto)
    # Kwaliteitsvlag uit de verfijningspass (alleen YOLO+RTMPose-backend): horizontale
    # afwijking (px) van het kniepunt t.o.v. de middellijn van het been in het pak-
    # kleurmasker. Frontaal gefilmd hoort het gewricht in het midden van het been te
    # liggen; een grote afwijking markeert frames waar de meting wankel is.
    middellijn_dev: dict = None     # {'l_knie': px, 'r_knie': px} (None = niet gemeten)
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
    hoek: float       # hoek op het voltooiingsframe zelf (niet gemiddeld — zie segmenteer_afzetten)
    min_hoek: float
    max_hoek: float
    opmerking: str = None   # bv. "alternatie?" als L/R niet klopt, of "samengevoegd"
    onvolledig: str = None  # None = telt mee; anders de reden waarom deze afzet buiten
                            # gem/min/max valt (ONV_AFGEKAPT / ONV_GEEN_PUSH). Het event
                            # blijft zichtbaar — er is immers iets gebeurd — maar de hoek
                            # is in beide gevallen systematisch te steil om te meten.
    # Perspectiefcorrectie (None zonder kalibratie):
    correctie: float = None      # toegepaste correctie bij afzet-voltooiing (graden)
    betrouwbaar: bool = True     # False: hoek gemeten met been bijna in de kijkrichting
    snelheid: float = None       # gemiddelde snelheid tijdens de afzet (m/s)
    slaglengte: float = None     # afgelegde afstand tijdens de afzet (m)


@dataclass
class PerspectiefConfig:
    """Opt-in perspectiefcorrectie via baanlijnen (fase 7): een kalibratie uit
    `schaats_perspectief.kalibreer_uit_lijnen` plus de reconstructie-keuzes.
    Zonder deze config gedraagt de pijplijn zich exact als voorheen.

    `invoer` (KalibratieInvoer) is de bewaarbare herkomst van `kalibratie`. Hij is
    optioneel omdat de kern ook met een los opgebouwde kalibratie werkt (zelftests),
    maar zónder invoer kan de config niet opgeslagen worden — `naar_dict` weigert dat
    dan expliciet in plaats van stilzwijgend een correctie te laten verdampen."""
    kalibratie: object              # schaats_perspectief.PerspectiefKalibratie
    methode: str = "onderbeen"      # 'onderbeen' (bol-snijding) | 'beenvlak' (rijrichting-vlak)
    onderbeen_l: float = None       # onderbeenlengte in m (verplicht bij 'onderbeen')
    enkel_hoogte: float = ENKEL_HOOGTE_M
    invoer: object = None           # schaats_perspectief.KalibratieInvoer

    def naar_dict(self):
        """JSON-bare vorm voor `analyse.instellingen_json`."""
        if self.invoer is None:
            raise ValueError("deze PerspectiefConfig heeft geen KalibratieInvoer en "
                             "kan dus niet opgeslagen worden")
        return {
            "invoer": self.invoer.naar_dict(),
            "methode": self.methode,
            "onderbeen_l": None if self.onderbeen_l is None else float(self.onderbeen_l),
            "enkel_hoogte": float(self.enkel_hoogte),
        }

    @classmethod
    def uit_dict(cls, d):
        """Herbouwt de config uit `naar_dict`, inclusief het herberekenen van de
        kalibratie uit de bewaarde lijnen. Gooit ValueError als dat niet lukt."""
        invoer = schaats_perspectief.KalibratieInvoer.uit_dict(d["invoer"])
        return cls(kalibratie=invoer.kalibreer(),
                   methode=d.get("methode", "onderbeen"),
                   onderbeen_l=d.get("onderbeen_l"),
                   enkel_hoogte=d.get("enkel_hoogte", ENKEL_HOOGTE_M),
                   invoer=invoer)


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


# ---------------------------------------------------------------------------
# Interlacing (kamtanden)
# ---------------------------------------------------------------------------
# Een camcorder die 1080i50 opneemt schiet 50 keer per seconde een hálf beeld (eerst de
# even rijen, dan de oneven) en weeft die twee 1/50 s uit elkaar liggende momenten tot
# één frame. Op stilstaande delen zie je daar niets van; op een bewegend been staan de
# even en de oneven rijen op een ándere plek — de kamtanden. Een mediaspeler deïnterlacet
# bij het afspelen (via de GPU), OpenCV niet: die levert het geweven frame precies zoals
# het in het bestand staat, dus zowel het beeld áls de pose-detectie krijgt de kam te zien.
#
# Gemeten op `00005.MTS` (AVCHD 1080i50) tegen de progressieve telefoonclips: de twee
# velden liggen op knieën en enkels **6,1 px** uit elkaar (p90 13,4; max 44), tegen
# 0,06 px op progressief materiaal. Met "2 px keypointfout = 2–4° hoekfout" (OPNAME.md)
# is dat in dat materiaal de grootste ruisbron die er is — groter dan wat er
# algoritmisch nog te winnen valt.
DEINT_DREMPEL       = 8      # per-pixel kamdrempel op grijswaarden
DEINT_MIN_KAMPIXELS = 2000   # minder kam in een frame = te weinig beweging om te oordelen
DEINT_PROEF_FRAMES  = 24     # metingen die `is_interlaced` verzamelt
DEINT_PROEF_MAX     = 300    # frames die het daarvoor hoogstens doorleest
DEINT_VERSCHUIVING  = 0.5    # px veldverschuiving waarboven een video interlaced heet
_DEINT_VENSTER      = 64     # halve venstermaat voor de faseCorrelatie
_DEINT_KERN = np.ones((7, 3), np.uint8)


def _kam_masker(grijs_i16, drempel=DEINT_DREMPEL):
    """
    Per pixel: wijkt deze rij van BEIDE verticale buren dezelfde kant op af? Dat is de
    handtekening van een kam — bij gewoon beelddetail ligt een rij tússen zijn buren in.
    Verwacht int16 (uint8 loopt over op het verschil).
    """
    m = grijs_i16[1:-1]
    # In plaats van `(a * b) > d²` met drie tijdelijke arrays van 2 megapixel: het product
    # in `a` zelf schrijven. Bit-identiek (int16 wrapt in beide gevallen even hard) en op
    # 1080p ~2 ms sneller — wat in de analyse niets voorstelt, maar tijdens het afspelen
    # van camcorderbeeld telt elke millisecond van het budget van 40 ms per frame.
    a = m - grijs_i16[:-2]
    a *= m - grijs_i16[2:]
    return a > drempel * drempel


def deinterlace(frame, drempel=DEINT_DREMPEL):
    """
    Haalt de kamtanden uit één frame: waar het beeld kamt worden de **oneven** rijen
    weggegooid en uit de even rijen geïnterpoleerd, zodat het bewegende deel uit precies
    één moment komt. Stilstaande delen blijven onaangeroerd op volle verticale resolutie.

    **Eén veld moet winnen.** De voor de hand liggende variant — beide velden middelen —
    haalt de kam er wél uit maar laat het temporele mengsel staan: elke uitvoerrij is dan
    nog steeds een mengsel van twee momenten. Gemeten ging de veldverschuiving op
    knie/enkel daarmee van 6,07 naar 5,52 px, tegen **0,03 px** met deze versie (ffmpeg's
    `yadif` haalt 0,07 px). Kosten ~12 ms per 1080p-frame — verwaarloosbaar naast de
    ~2 s/frame van de detectiepass, maar wél merkbaar tijdens het afspelen: daar is het
    hele budget 40 ms per frame bij 25 fps, en decoderen (~8 ms) plus schalen naar een
    HiDPI-scherm (~15 ms) zit daar al in. Vandaar dat de twee zwaarste stappen zuinig
    geschreven zijn; zie `_kam_masker` en de `cv2.copyTo` hieronder.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    # Verticaal uitsmeren zodat een kamgebied als geheel behandeld wordt en er geen losse
    # rijen tussenuit vallen; [0::2] houdt daarna de oneven absolute rijen over.
    kam = cv2.dilate(_kam_masker(g, drempel).view(np.uint8), _DEINT_KERN).view(bool)[0::2]
    uit = frame.copy()
    interp = cv2.addWeighted(frame[0:-2:2], 0.5, frame[2::2], 0.5, 0.0)
    # De samenvoegstap via OpenCV in plaats van numpy: `uit[1:-1:2] = np.where(kam[:,:,None],
    # interp, origineel)` kost 8,9 ms, `cv2.copyTo` op een contigue kopie 1,1 ms — dezelfde
    # uitvoer, byte voor byte. Het verschil is dat numpy hier een bool-masker over drie
    # kanalen broadcast en een complete nieuwe array bouwt, waar OpenCV met SIMD over 16
    # threads alleen de gemaskeerde bytes schrijft. De omweg via een kopie is nodig omdat
    # cv2 niet in een strided view (elke tweede rij) kan schrijven; die kopie is goedkoop.
    oneven = frame[1:-1:2].copy()
    cv2.copyTo(interp, kam.view(np.uint8), oneven)
    uit[1:-1:2] = oneven
    return uit


def _veldverschuiving(grijs_i16, cy, cx):
    """
    Hoeveel pixels het beeld tussen de twee velden opschuift, gemeten rond (cy, cx) door
    de even en de oneven rijen met faseCorrelatie op elkaar te leggen. None als het frame
    te klein is of de correlatie niets oplevert.
    """
    h, w = grijs_i16.shape
    if h < 2 * _DEINT_VENSTER or w < 2 * _DEINT_VENSTER:
        return None
    cy = int(np.clip(cy, _DEINT_VENSTER, h - _DEINT_VENSTER))
    cx = int(np.clip(cx, _DEINT_VENSTER, w - _DEINT_VENSTER))
    crop = grijs_i16[cy - _DEINT_VENSTER:cy + _DEINT_VENSTER,
                     cx - _DEINT_VENSTER:cx + _DEINT_VENSTER].astype(np.float32)
    a, b = crop[0::2], crop[1::2]
    n = min(len(a), len(b))
    venster = cv2.createHanningWindow((a.shape[1], n), cv2.CV_32F)
    (dx, _), respons = cv2.phaseCorrelate(a[:n] * venster, b[:n] * venster, venster)
    return abs(dx) if respons > 0.15 else None


def is_interlaced(input_pad, drempel_px=DEINT_VERSCHUIVING):
    """
    Bepaalt of een video interlaced is: liggen de twee velden van een frame op hetzelfde
    moment (progressief) of 1/50 s uit elkaar (interlaced)?

    Per frame wordt de **dichtste kamcluster** opgezocht — dat is het bewegende object —
    en daar worden de even en de oneven rijen met faseCorrelatie op elkaar gelegd. Mikken
    op de mediáne kampixel werkt niet: bij weinig beweging is de kam verspreide
    compressieruis en landt het venster op stilstaand ijs (gemeten: 6 van 27 clips fout).

    De **kamfractie alleen volstaat evenmin**, hoe verleidelijk goedkoop ook: een kleine,
    sterk gecomprimeerde clip (832×464) haalde daarop 0,035 tegen ≤ 0,0025 voor al het
    andere progressieve materiaal, en zou dus onterecht gefilterd worden. Alleen de
    verschuiving zelf scheidt de twee gevallen; die is per definitie 0 als beide velden
    van hetzelfde moment komen, wat er ook aan compressie overheen is gegaan.

    Gekalibreerd over de 27 video's in de bibliotheek (26-8-2026): de elf
    camcorderfragmenten meten 1,59–17,41 px, de zestien progressieve clips 0,00–0,15 px —
    0 fouten, ~3× marge aan beide kanten van de drempel. Te weinig bewegende frames om te
    oordelen → False: liever niet filteren dan onnodig pixels aanraken.
    """
    cap = cv2.VideoCapture(input_pad)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {input_pad}")
    metingen, bekeken = [], 0
    try:
        while len(metingen) < DEINT_PROEF_FRAMES and bekeken < DEINT_PROEF_MAX:
            ret, frame = cap.read()
            if not ret:
                break
            bekeken += 1
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
            kam = _kam_masker(g)
            if kam.sum() < DEINT_MIN_KAMPIXELS:
                continue                      # te weinig beweging: dit frame zegt niets
            dichtheid = cv2.blur(kam.astype(np.float32), (2 * _DEINT_VENSTER,) * 2)
            _, _, _, (mx, my) = cv2.minMaxLoc(dichtheid)
            d = _veldverschuiving(g, my + 1, mx)   # +1: `kam` begint op rij 1
            if d is not None:
                metingen.append(d)
    finally:
        cap.release()
    if len(metingen) < 5:
        return False
    # p75 en niet de mediaan: de vraag is of het beeld óóit veldverschuiving vertoont, en
    # een deel van de frames vangt nu eenmaal een moment waarop er weinig beweegt.
    return float(np.percentile(metingen, 75)) > drempel_px


class VideoLezer:
    """
    `cv2.VideoCapture` met het kamfilter op read()/retrieve().

    Al het andere (`grab`, `set`, `get`, `isOpened`, `release`) gaat via `__getattr__`
    door naar de capture, zodat elke bestaande leeslus onveranderd blijft werken —
    hetzelfde doorgeefluik-patroon als de DirectML-schil in schaats_yolo.
    """

    def __init__(self, input_pad, drempel=DEINT_DREMPEL):
        self._cap = cv2.VideoCapture(input_pad)
        self._drempel = drempel

    def read(self):
        ret, frame = self._cap.read()
        return (True, deinterlace(frame, self._drempel)) if ret else (ret, frame)

    def retrieve(self, *args):
        ret, frame = self._cap.retrieve(*args)
        return (True, deinterlace(frame, self._drempel)) if ret else (ret, frame)

    def __getattr__(self, naam):
        if naam == "_cap":                    # nog niet gezet: geen oneindige recursie
            raise AttributeError(naam)
        return getattr(self._cap, naam)


def open_video(input_pad, deinterlacen=False):
    """
    Opent een video, met of zonder kamfilter. **Zonder is het letterlijk een
    `cv2.VideoCapture`** — voor progressief materiaal verandert er dus niets, ook geen
    extra Python-aanroep per frame.
    """
    return VideoLezer(input_pad) if deinterlacen else cv2.VideoCapture(input_pad)


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
    hoek = float(np.degrees(np.arctan2(dy, abs(dx))))
    return round(hoek - horizon_deg, 1)     # gewone float: gaat zo de DB/CSV/JSON in


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
        # Enkels op gelijke hoogte: gebruik heupverschuiving als tiebreaker. `heup_history`
        # bevat tuples (l_heup_x, r_heup_x), dus vergelijk het heup-*midden* met dat van 3
        # frames terug — net als `detecteer_gewicht_op_been`. (Eerder werd hier de rechter-
        # heup van nu tegen de línkerheup van toen gezet: dat mat niet de verschuiving maar
        # de constante heupbreedte, waardoor de tiebreaker altijd 'links' antwoordde.)
        if len(heup_history) >= 3:
            midden_nu     = (lm_data['l_heup'][0] + lm_data['r_heup'][0]) / 2
            midden_eerder = (heup_history[-3][0] + heup_history[-3][1]) / 2
            return 'links' if midden_nu - midden_eerder > 0 else 'rechts'
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
    if lm_data is None or been not in ('links', 'rechts'):
        return                        # bochtframe of nog niet doorgerekend: niets te tekenen
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

    # Hoeklijn verlengd (enkel → richting knie, maar projectie op grond). Een hoek ≤ 0
    # betekent dat de knie onder de enkel zit: geen schaatshouding maar een kapotte
    # detectie. Die lijn tekenen we juist wél, in rood — stilzwijgend weglaten verbergt
    # het probleem terwijl de tabel de waarde gewoon rapporteert.
    lijn_len = 80
    richting_x = int(np.sin(np.radians(hoek)) * lijn_len * (-1 if been == 'links' else 1))
    richting_y = -int(np.cos(np.radians(hoek)) * lijn_len)
    eind = (enkel[0] + richting_x, enkel[1] + richting_y)
    cv2.line(frame, enkel, eind, GEEL if hoek > 0 else ROOD, 2, cv2.LINE_AA)

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
              correctie=None, betrouwbaar=True, snelheid=None, strek_ratio=None):
    """Teken het HUD-paneel linksboven. `correctie`/`betrouwbaar`/`snelheid` zijn de
    perspectiefcorrectie-velden (alleen getoond als er een kalibratie actief was).
    `strek_ratio` toont de beenstrekking (afzet-detectie: piek = einde afzet)."""
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

    if strek_ratio is not None:           # beenstrekking: piek = afzet klaar
        regels.insert(len(regels) - 1, (f"Strekking: {strek_ratio:.2f}", PAARS, 0.5))

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

    if resultaat.bocht:
        # Het skelet blijft staan (je wilt zien dát daar iemand rijdt), maar er is hier
        # niets gemeten — zeg dat er dan ook bij i.p.v. een lege HUD te tonen.
        cv2.putText(frame, "BOCHT - niet gemeten", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GEEL, 2)
        return

    if toon_afzetbeen:
        teken_been_overlay(frame, resultaat.lm_data, resultaat.been, resultaat.hoek,
                            resultaat.gewicht_erop, resultaat.kniehoek, resultaat.horizon_deg)

    if toon_hud:
        teken_hud(frame, resultaat.been, resultaat.hoek, resultaat.gewicht_erop,
                  resultaat.kniehoek, resultaat.smooth_hoek, resultaat.frame_nr, fps, w, h,
                  correctie=resultaat.hoek_correctie, betrouwbaar=resultaat.hoek_betrouwbaar,
                  snelheid=resultaat.snelheid, strek_ratio=resultaat.strek_ratio)


def _zichtbare_xy(lm, idxs=None, min_vis=TRACK_MIN_VIS):
    """Genormaliseerde (x,y) van voldoende zichtbare landmarks; optioneel beperkt tot idxs."""
    bron = lm if idxs is None else [lm[i] for i in idxs]
    return [(l.x, l.y) for l in bron if l.visibility >= min_vis]


def torso_centroid(lm):
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


def bocht_ratio(lm, w, h, min_vis=VIS_MIN):
    """
    "Sta ik frontaal in beeld?" als schaalvrij getal: heupbreedte gedeeld door de
    romplengte (schoudermidden → heupmidden), beide in pixels. Frontaal staan de heupen
    naast elkaar (~0,6–1,3); draait het lichaam de bocht in, dan komen ze achter elkaar
    te staan en stort de breedte in (~0,2) terwijl de romp even lang blijft.

    Beide maten in pixels (níet genormaliseerd), anders zou de beeldverhouding de ratio
    scheeftrekken. None als de vier landmarks niet zichtbaar zijn of de romp te klein is
    om nog iets te kunnen zeggen (`BOCHT_MIN_TORSO_PX`) — op die afstand is de breedte
    pixelruis. Werkt op elke lijst van landmark-objecten met .x/.y/.visibility, dus ook
    op een YOLO-`Detectie.lm` tijdens de detectiepass.
    """
    try:
        sl, sr, hl, hr = (lm[L_SHOULDER], lm[R_SHOULDER], lm[L_HIP], lm[R_HIP])
    except (TypeError, IndexError):
        return None
    if min(p.visibility for p in (sl, sr, hl, hr)) < min_vis:
        return None
    heup_b = abs(hl.x - hr.x) * w
    romp = float(np.hypot(((sl.x + sr.x) - (hl.x + hr.x)) / 2 * w,
                          ((sl.y + sr.y) - (hl.y + hr.y)) / 2 * h))
    if romp < BOCHT_MIN_TORSO_PX:
        return None
    return heup_b / romp


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


def _volg_kandidaten(buffer, gate=TRACK_GATE, max_gat=3):
    """
    Rijg de poses uit een reeks frames tot ruwe kandidaat-sporen: elke pose wordt aan
    het spoor gekoppeld waarvan de laatste torso-centroïde het dichtst bij ligt (binnen
    `gate`, en niet ouder dan `max_gat` frames), anders begint er een nieuw spoor.

    Dit is bewust simpeler dan `DoelTracker` — het hoeft bij de koude start alleen goed
    genoeg te zijn om "rijdt" van "staat stil langs de boarding" te onderscheiden.
    Retourneert dicts met `start` (eerste frame-index), `punten` en `opp`.
    """
    sporen = []
    for i, poses in enumerate(buffer):
        centroids = [(c, p) for c, p in ((torso_centroid(p), p) for p in poses)
                     if c is not None]
        for c, p in centroids:
            beste, beste_afst = None, gate
            for s in sporen:
                if s['laatst'] == i or i - s['laatst'] > max_gat:
                    continue            # dit frame al vergeven, of het spoor is verlopen
                d = ((s['punten'][-1][0] - c[0]) ** 2 + (s['punten'][-1][1] - c[1]) ** 2) ** 0.5
                if d < beste_afst:
                    beste, beste_afst = s, d
            if beste is None:
                sporen.append({'start': i, 'laatst': i, 'punten': [c],
                               'opp': [_bbox_oppervlak(p)]})
            else:
                beste['punten'].append(c)
                beste['opp'].append(_bbox_oppervlak(p))
                beste['laatst'] = i
    return sporen


def _pad_lengte(punten):
    """Totale afgelegde weg langs een reeks genormaliseerde punten."""
    return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
               for a, b in zip(punten, punten[1:]))


def _kies_bewegend_doel(buffer):
    """
    Kies uit de eerste frames de doelschaatser als **grootste beweger**: mediane
    bbox-oppervlakte × afgelegde weg, met `SEED_MIN_VERPLAATSING` als ondergrens.
    Zonder dat criterium wint bij een koude start simpelweg de grootste pose in beeld —
    en dat is geregeld een omstander langs de boarding, die dichter bij de camera staat
    dan de schaatser die verderop rijdt.

    Retourneert `(startframe, centroïde op dat frame)` of None als er niets te kiezen is.
    """
    sporen = _volg_kandidaten(buffer)
    if not sporen:
        return None
    bewegers = [s for s in sporen if _pad_lengte(s['punten']) >= SEED_MIN_VERPLAATSING]
    beste = max(bewegers or sporen,
                key=lambda s: float(np.median(s['opp']))
                              * max(_pad_lengte(s['punten']), 1e-6))
    return beste['start'], beste['punten'][0]


class DoelTracker:
    """
    Volgt één doelschaatser door de frames heen. MediaPipe levert per frame een
    lijst poses (meerdere schaatsers); deze tracker kiest telkens de pose die het
    best bij de voorspelde positie van het doel past, met een afstandspoort zodat
    de tracking niet naar een andere schaatser overspringt als ze elkaar kruisen.

    Seeden: als `doel_punt` (genormaliseerd (x,y)) gegeven is, wordt de schaatser het
    dichtst daarbij gekozen. Dat punt komt van een muisklik óf — bij een koude start
    zonder klik — van `_kies_bewegend_doel`, dat na een warmup-venster de grootste
    *beweger* aanwijst. Ontbreekt het punt alsnog, dan valt de tracker terug op de
    grootste pose in beeld.

    Twee dingen zijn expliciet **coast-bewust** (een detectiegat van een paar frames is
    bij bewegingsonscherpte niets bijzonders):
    - de voorspelling schuift per gemist frame mee (`n × snelheid`, niet 1 × snelheid) en
      de poort groeit met de gat-lengte — anders valt de schaatser na ~3 gemiste frames
      buiten de poort en is hij permanent kwijt, ook al wordt hij netjes gedetecteerd;
    - de laatst bekende plek (`laatste_bekend`) wordt apart van de lock-vlag (`centroid`)
      bijgehouden, zodat een herseed ná langdurig verlies dáárop kan aansluiten i.p.v. op
      het (verouderde) klikpunt van frame 0 of op "de grootste pose in beeld".
    """
    def __init__(self, doel_punt=None, gate=TRACK_GATE, hervind_frames=15):
        self.doel_punt = doel_punt
        self.gate = gate
        self.hervind_frames = hervind_frames
        self.centroid = None         # torso-centroïde bij de laatste match; None = geen lock
        self.laatste_bekend = None   # idem, maar blijft ook ná verlies staan (voor herseed)
        self.snelheid = (0.0, 0.0)   # geschatte verplaatsing per frame
        self.kwijt = 0               # aantal opeenvolgende frames zonder match

    def _verwacht(self, vanaf, stappen):
        """Constante-snelheid-voorspelling `stappen` frames vooruit vanaf `vanaf`. De
        horizon wordt begrensd op `hervind_frames` en het resultaat op het beeld geklemd:
        een oude snelheidsschatting × een lang gat rekent de schaatser anders het beeld
        uit, waarna niemand meer binnen welke poort dan ook valt."""
        stap = min(max(0, stappen), self.hervind_frames)
        return (min(1.0, max(0.0, vanaf[0] + stap * self.snelheid[0])),
                min(1.0, max(0.0, vanaf[1] + stap * self.snelheid[1])))

    def _seed(self, centroids):
        """Kies een startpose uit [(centroid, pose), ...] (alle centroids != None), of
        None als er niets geloofwaardigs bij zit (alleen bij een herseed)."""
        if self.laatste_bekend is not None:
            # Herseed na langdurig verlies: pak de schaatser het dichtst bij de laatst
            # bekende plek, mee-geëxtrapoleerd over de verliesduur. Niemand binnen de
            # ruime poort → geen lock (liever een gat dan de verkeerde persoon volgen).
            ex, ey = self._verwacht(self.laatste_bekend, self.kwijt)
            beste = min(centroids, key=lambda cp: (cp[0][0]-ex)**2 + (cp[0][1]-ey)**2)
            afstand = ((beste[0][0]-ex)**2 + (beste[0][1]-ey)**2) ** 0.5
            return beste if afstand <= TRACK_HERSEED_GATE else None
        if self.doel_punt is not None:
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
        # Koude start zonder klik: volg de grootste (meest prominente) schaatser.
        return max(centroids, key=lambda cp: _bbox_oppervlak(cp[1]))

    def update(self, poses):
        """Kies de doel-pose voor dit frame; retourneert de landmarklijst of None."""
        centroids = [(torso_centroid(p), p) for p in poses]
        centroids = [(c, p) for c, p in centroids if c is not None]
        if not centroids:
            self.kwijt += 1
            return None

        # Nog geen lock, of te lang kwijt → (her)seed.
        if self.centroid is None or self.kwijt > self.hervind_frames:
            gekozen = self._seed(centroids)
            if gekozen is None:
                self.kwijt += 1      # herseed geweigerd: blijf coasten (geen pose dit frame)
                return None
            c, p = gekozen
            self.centroid = self.laatste_bekend = c
            self.snelheid = (0.0, 0.0)
            self.kwijt = 0
            return p

        # Voorspel de positie — mee-geëxtrapoleerd over de gemiste frames — en kies de
        # dichtstbijzijnde pose binnen de poort, die met de gat-lengte meegroeit.
        n = self.kwijt + 1                     # frames sinds de laatste match
        px, py = self._verwacht(self.centroid, n)
        poort = min(self.gate * (1.0 + TRACK_GATE_GROEI * self.kwijt), TRACK_GATE_MAX)
        (bc, bp), afstand = min(
            (((c, p), ((c[0]-px)**2 + (c[1]-py)**2) ** 0.5) for c, p in centroids),
            key=lambda t: t[1],
        )
        if afstand > poort:
            # Beste kandidaat te ver → waarschijnlijk de andere schaatser; coast.
            self.kwijt += 1
            return None

        # Match: snelheid (per frame, dus gedeeld door de gat-lengte) en positie licht
        # gedempt bijwerken.
        vx, vy = (bc[0] - self.centroid[0]) / n, (bc[1] - self.centroid[1]) / n
        self.snelheid = (0.5 * self.snelheid[0] + 0.5 * vx,
                         0.5 * self.snelheid[1] + 0.5 * vy)
        self.centroid = self.laatste_bekend = bc
        self.kwijt = 0
        return bp


def analyseer_frames(input_pad, model_pad, force_fps=None, num_poses=NUM_POSES_DEFAULT,
                      doel_punt=None, progress_callback=None, deinterlacen=False):
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

    cap = open_video(input_pad, deinterlacen)
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
    # Koude start zonder klik: eerst een seconde meekijken en dán pas kiezen (zie
    # _kies_bewegend_doel). Met een klik is er niets te kiezen en analyseren we direct.
    warmup = 0 if doel_punt is not None else int(max(1, round(fps * SEED_WARMUP_S)))
    buffer = [] if warmup else None

    def _maak(nr, doel):
        r = FrameResultaat(frame_nr=nr, tijd=nr / fps if fps > 0 else 0)
        if doel is not None:
            r.lm = doel
            r.pose_gevonden = True
        return r

    def _leeg_buffer():
        """Kies de doelschaatser uit de gebufferde frames en speel die frames daarna
        alsnog door de tracker af, zodat er geen enkel frame verloren gaat."""
        keuze = _kies_bewegend_doel(buffer)
        start = 0
        if keuze is not None:
            start, tracker.doel_punt = keuze
        for i, poses in enumerate(buffer):
            # Vóór het startframe van het gekozen spoor is het doel nog niet in beeld;
            # de tracker zou daar op een omstander locken, dus die frames blijven leeg.
            yield _maak(i, tracker.update(poses) if i >= start else None)

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
            if buffer is not None:
                buffer.append(poses)
                if len(buffer) >= warmup:
                    yield from _leeg_buffer()
                    buffer = None
            else:
                yield _maak(frame_nr, tracker.update(poses))

            frame_nr += 1

            if progress_callback is not None:
                progress_callback(frame_nr, totaal)

        if buffer:                       # video korter dan het warmup-venster
            yield from _leeg_buffer()

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


def _gat_afstand(betrouwbaar):
    """Per frame: hoeveel frames het van het dichtstbijzijnde betrouwbare frame af ligt
    (0 waar het zelf betrouwbaar is). Maat voor hoe diep je in een detectiegat zit."""
    idx = np.flatnonzero(np.asarray(betrouwbaar, dtype=bool))
    n = len(betrouwbaar)
    if len(idx) == 0:
        return np.full(n, float(n))
    alle = np.arange(n)
    dichtstbij = np.searchsorted(idx, alle).clip(0, len(idx) - 1)
    vorige = (dichtstbij - 1).clip(0, len(idx) - 1)
    return np.minimum(np.abs(alle - idx[dichtstbij]), np.abs(alle - idx[vorige])).astype(float)


def _lopend_max(y, window):
    """Gecentreerd lopend maximum: de envelope van een golvend signaal."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = window // 2
    return np.array([y[max(0, i - half):min(n, i + half + 1)].max() for i in range(n)])


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


def _fix_lr_swaps(X, Y, V, w, h):
    """
    Herstel links/rechts-verwisselingen van de beengewrichten binnen één segment.
    Werkt in-place op de segment-arrays X/Y/V (T×33). De twee beentrajecten worden op
    continuïteit gevolgd: past op frame t de gewisselde toewijzing dúidelijk beter bij
    de vorige posities (kosten < LR_SWAP_FACTOR × ongewisseld), dan worden L en R daar
    omgedraaid.

    De beslissing geldt voor het **hele been tegelijk** — knie én enkel, met hiel en
    teen mee — op de opgetelde kosten van beide gewrichtsparen. Zouden knie en enkel
    los van elkaar beslissen, dan kan de ene wél en de andere niet wisselen: dan hangt
    de linkerknie aan de rechterenkel, een anatomisch onmogelijk skelet met een
    onzinnige tibialengte (die vervolgens de botlengte-check laat aanslaan).

    Omdat het kettinkje op het eerste frame is verankerd, kan een verkeerd eerste
    frame de hele reeks omgekeerd labelen; daarom achteraf een meerderheidsstem
    tegen de ruwe detectorlabels — de detector heeft het meestal goed, wij
    repareren alleen de minderheids-stukken. Die stem loopt uitsluitend over de
    frames waar we écht een beslissing hebben genomen (zie `besloten`).
    """
    paren   = ((L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE))
    volgers = ((L_HEEL, R_HEEL), (L_TOE, R_TOE))     # zitten aan de enkel vast
    T = len(X)
    gewisseld = np.zeros(T, dtype=bool)
    besloten  = np.zeros(T, dtype=bool)   # frames waar de continuïteitskost een oordeel gaf
    vorige = {}                           # paar → (laatste linker-, laatste rechterpositie)
    for t in range(T):
        huidig = {(l, r): (np.array([X[t, l] * w, Y[t, l] * h]),
                           np.array([X[t, r] * w, Y[t, r] * h]))
                  for l, r in paren
                  if V[t, l] >= VIS_MIN and V[t, r] >= VIS_MIN}
        if not huidig:
            continue                      # onbetrouwbaar frame: niet beslissen
        besloten[t] = True
        kost_id = kost_sw = 0.0
        vergeleken = False
        for paar, (pl, pr) in huidig.items():
            if paar not in vorige:
                continue
            ql, qr = vorige[paar]
            kost_id += np.linalg.norm(pl - ql) + np.linalg.norm(pr - qr)
            kost_sw += np.linalg.norm(pl - qr) + np.linalg.norm(pr - ql)
            vergeleken = True
        if vergeleken and kost_sw < LR_SWAP_FACTOR * kost_id:
            gewisseld[t] = True
            huidig = {paar: (pr, pl) for paar, (pl, pr) in huidig.items()}
        vorige.update(huidig)
    if not gewisseld.any():
        return
    # Meerderheidsstem alléén over de besliste frames. Een overgeslagen frame staat
    # op False omdat er géén oordeel is, niet omdat er "niet gewisseld" moest worden;
    # zou de inversie die frames meenemen, dan kregen juist de onbetrouwbaarste
    # frames een L/R-wissel zonder enige onderbouwing, tegen hun buren in.
    if gewisseld[besloten].mean() > 0.5:  # ketting verkeerd verankerd: labels omdraaien
        gewisseld[besloten] = ~gewisseld[besloten]
    for t in np.flatnonzero(gewisseld):
        for a, b in paren + volgers:
            for A in (X, Y, V):
                A[t, a], A[t, b] = A[t, b], A[t, a]


def _botlengte_uitschieters(X, Y, V, w, h):
    """
    Booleaans masker (T×33): gewrichten waarvan een aangrenzend bot (femur/tibia)
    in dat frame een lengte-uitschieter heeft. Botten zijn star; hun beeldlengte
    verandert alleen traag met de afstand tot de camera. Een sprong betekent dat
    (minstens) één eindpunt fout gedetecteerd is — welke weten we niet, dus beide
    eindpunten gelden daar als onbetrouwbaar.
    """
    T = len(X)
    mask = np.zeros((T, X.shape[1]), dtype=bool)
    botten = ((L_HIP, L_KNEE), (R_HIP, R_KNEE), (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE))
    for a, b in botten:
        geldig = (V[:, a] >= VIS_MIN) & (V[:, b] >= VIS_MIN)
        idx = np.flatnonzero(geldig)
        if len(idx) < HAMPEL_WINDOW:
            continue
        lengte = np.hypot((X[idx, a] - X[idx, b]) * w, (Y[idx, a] - Y[idx, b]) * h)
        fout = _hampel_uitschieters(lengte)
        for t in idx[fout]:
            mask[t, a] = mask[t, b] = True
    return mask


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
        # Ook heel korte segmenten (versnipperde detectie) gaan door de molen. De
        # filters degraderen daar netjes: Savitzky–Golay geeft een venster < poly+2
        # ongewijzigd terug, `_begrens_interpolatie` laat een randreeks staan en de
        # botlengte-check slaat een te korte reeks over — maar de L/R-fix doet wél zijn
        # werk. Ze overslaan liet daar ruwe, ongecontroleerde data staan zonder dat dat
        # ergens uit bleek.
        X = np.array([[resultaten[i].lm[j].x for j in range(n_lm)] for i in seg])
        Y = np.array([[resultaten[i].lm[j].y for j in range(n_lm)] for i in seg])
        V = np.array([[resultaten[i].lm[j].visibility for j in range(n_lm)] for i in seg])
        # Eerst links/rechts-verwisselingen herstellen: die zien er voor de per-
        # gewricht-filters uit als (dubbele) sprongen, maar zijn exact herstelbaar.
        _fix_lr_swaps(X, Y, V, w, h)
        # Onbetrouwbaar = slecht zichtbaar óf een positie-uitschieter (occlusie).
        Bet = np.empty((len(seg), n_lm), dtype=bool)
        for j in range(n_lm):
            b = (V[:, j] >= VIS_MIN)
            b &= ~_hampel_uitschieters(X[:, j])
            b &= ~_hampel_uitschieters(Y[:, j])
            Bet[:, j] = b
        # Botlengte-check: een femur/tibia-lengtesprong markeert beide eindpunten.
        Bet &= ~_botlengte_uitschieters(X, Y, V, w, h)
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
            # Visibility uit V (niet uit oud): bij een L/R-wissel is die meegewisseld.
            resultaten[i].lm = [
                Landmark(float(X[t, j]), float(Y[t, j]),
                         getattr(oud[j], 'z', 0.0), float(V[t, j]))
                for j in range(n_lm)
            ]


def bepaal_horizon_reeks(input_pad, n_frames, fps, force_fps=None, progress_callback=None,
                         deinterlacen=False):
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
    cap = open_video(input_pad, deinterlacen)
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


def kader_reeks(resultaten, fps):
    """
    Het kader dat de schaatser per frame nodig heeft: `(midden_x, midden_y, straal)`, alles
    genormaliseerd. De GUI leidt hier de automatische zoom én het volgpunt uit af — de
    uitsnede is in genormaliseerde coördinaten symmetrisch (0.5/zoom in x én y), dus één
    straal rond één middelpunt is precies wat er in beeld moet passen.

    Het middelpunt is het midden van álle zichtbare landmarks, níet `torso_centroid`: dat
    laatste ligt hoog in het lichaam (schouders + heupen), zodat een kader eromheen boven
    het hoofd net zoveel ruimte krijgt als onder de schaatsen — een kwart van het beeld
    verspild. De straal is de afstand van dat midden tot het verste punt, dus tot een bij
    de afzet ver uitgestrekt been, niet enkel de lichaamslengte.

    Opschoning als in `bepaal_horizon_reeks`: gaten (geen pose) en uitschieters (een ledemaat
    dat kort wegvalt of verspringt) weg-geïnterpoleerd, daarna smoothen. Voor de straal
    eerst een **lopend maximum** over `KADER_SMOOTH_S` en pas daarna Savitzky–Golay; die
    volgorde is essentieel, want gladstrijken alléén vlakt de piek af en dan valt een ver
    uitgestrekt been net buiten beeld. Het maximum over ruwweg één slag is een envelope die
    de breedste stand van dat moment altijd dekt en toch traag beweegt, zodat het kader niet
    met elke slag meepompt. Het middelpunt wordt korter gesmoothd (`KADER_MIDDEN_S`) — dat
    moet de schaatser wél volgen — en de straal wordt er ná die smoothing tegen gemeten,
    zodat de twee bij elkaar passen.

    Retourneert een lijst tupels van dezelfde lengte als `resultaten`, of None als er te
    weinig pose is om iets zinnigs te zeggen (de caller valt dan terug op een vaste zoom).
    """
    punten_per_frame = [
        _zichtbare_xy(r.lm) if (r.pose_gevonden and r.lm is not None) else []
        for r in resultaten
    ]
    betrouwbaar = np.array([len(p) > 0 for p in punten_per_frame])
    if betrouwbaar.sum() < KADER_MIN_FRAMES:
        return None

    def _opschonen(waarden):
        """Gaten + uitschieters eruit — het recept van bepaal_horizon_reeks, zonder smoothing."""
        y = np.array([v if v is not None else np.nan for v in waarden], dtype=float)
        y = _interpoleer_onbetrouwbaar(y, betrouwbaar)
        betr2 = betrouwbaar & ~_hampel_uitschieters(y)
        return _interpoleer_onbetrouwbaar(y, betr2), betr2

    midden_window = max(KADER_POLY + 2, int(round(KADER_MIDDEN_S * fps)))
    straal_window = max(KADER_POLY + 2, int(round(KADER_SMOOTH_S * fps)))
    cx, _ = _opschonen([(min(x for x, _ in p) + max(x for x, _ in p)) / 2 if p else None
                        for p in punten_per_frame])
    cy, _ = _opschonen([(min(y for _, y in p) + max(y for _, y in p)) / 2 if p else None
                        for p in punten_per_frame])
    cx = _savgol(cx, midden_window, KADER_POLY)
    cy = _savgol(cy, midden_window, KADER_POLY)

    # Straal t.o.v. het gesmoothte midden (niet het ruwe), anders sluiten ze niet op elkaar aan.
    ruw_straal = [max((max(abs(x - cx[i]), abs(y - cy[i])) for x, y in p), default=None)
                  for i, p in enumerate(punten_per_frame)]
    straal, betr2 = _opschonen(ruw_straal)
    straal = _savgol(_lopend_max(straal, straal_window), straal_window, KADER_POLY)
    # De SG-randfit kan doorschieten tot ≤ 0; dat zou verderop een deling door nul geven.
    ondergrens = max(1e-3, float(np.median(straal[betr2] if betr2.any() else straal)) * 0.05)
    straal = np.maximum(ondergrens, straal)

    # Lange detectiegaten: het kader vloeiend openen tot het volledige beeld (straal 0.5).
    # Buiten een gat is `f` 0 en verandert er niets. Het mengen gebeurt in "zoom"-ruimte
    # (0.5/straal) en niet in de straal zelf: dicht bij de schaatser is de straal klein, en
    # daar zou een lineaire menging de zoom in een paar frames laten instorten. Het midden
    # blijft staan waar het stond — bij straal 0.5 valt de uitsnede toch over het hele beeld.
    f = np.clip(_gat_afstand(betrouwbaar) / max(1.0, KADER_GAT_S * fps), 0.0, 1.0)
    straal = 0.5 / ((0.5 / straal) * (1 - f) + 1.0 * f)
    return [(float(cx[i]), float(cy[i]), float(straal[i])) for i in range(len(resultaten))]


def bepaal_bocht_reeks(resultaten, w, h, fps):
    """
    Markeert per frame of de schaatser in de bocht rijdt (`FrameResultaat.bocht`), zodat
    die frames geen afzetmeting meer opleveren. Recept van `bepaal_horizon_reeks` /
    `kader_reeks`: ruw signaal → uitschieters eruit → smoothen → beslissen.

    1. `bocht_ratio` per frame met een pose (heupbreedte / romplengte).
    2. Hampel-uitschieters weg (een frame waarin een heup kort verspringt) en gaten
       lineair overbruggen, daarna Savitzky–Golay over `BOCHT_SMOOTH_S` — de ratio golft
       licht mee met de slag en moet niet per frame kunnen omslaan.
    3. Hysterese: onder `BOCHT_IN` de bocht in, pas boven `BOCHT_UIT` er weer uit
       (hetzelfde patroon als de stand-toewijzing in `wijs_afzetbeen_cyclus`).
    4. Bochtstukken korter dan `BOCHT_MIN_S` zijn ruis en vervallen.

    Frames **zonder** pose kunnen niet gemeten worden: die erven de lopende toestand, en
    een `bocht`-vlag die er al op stond blijft staan (bij de YOLO-backend zijn dat de
    frames die de detectiepass heeft overgeslagen — daar is niets te meten en dat blijft
    zo). Frames **mét** pose worden op hun eigen ratio beoordeeld en kunnen een al gezette
    vlag dus ook weer **wissen**: precies wat er moet gebeuren als de detectiepass een
    stuk onterecht heeft overgeslagen maar de controleframes daarbinnen een keurig
    frontale schaatser laten zien.
    """
    n = len(resultaten)
    if n == 0:
        return

    def _ratio(r):
        if not (r.pose_gevonden and r.lm is not None):
            return np.nan
        v = bocht_ratio(r.lm, w, h)
        return np.nan if v is None else v

    ruw = np.array([_ratio(r) for r in resultaten], dtype=float)
    meetbaar = ~np.isnan(ruw)
    if not meetbaar.any():
        return                       # geen enkel oordeel mogelijk — laat de vlaggen staan
    y = _interpoleer_onbetrouwbaar(ruw, meetbaar)
    betr = meetbaar & ~_hampel_uitschieters(y)
    if betr.any():
        y = _interpoleer_onbetrouwbaar(y, betr)
    win = max(SMOOTH_POLY + 2, int(round(BOCHT_SMOOTH_S * fps)))
    if win % 2 == 0:
        win += 1
    y = _savgol(y, win, SMOOTH_POLY) if n >= win else y

    # Hysterese. De starttoestand komt van het eerste meetbare frame, zodat een clip die
    # ín de bocht begint meteen goed staat (i.p.v. pas na de eerste onderschrijding).
    eerste = int(np.argmax(meetbaar))
    staat = bool(y[eerste] < BOCHT_IN)
    vlag = np.zeros(n, dtype=bool)
    for i in range(n):
        if meetbaar[i]:
            if y[i] < BOCHT_IN:
                staat = True
            elif y[i] > BOCHT_UIT:
                staat = False
            vlag[i] = staat
        else:
            vlag[i] = staat or resultaten[i].bocht

    # Te korte bochtjes zijn ruis. Andersom geldt hetzelfde: een paar frames "recht stuk"
    # midden in de bocht is geen recht stuk, en zou een schijnmeting kunnen opleveren.
    min_len = max(1, int(round(BOCHT_MIN_S * fps)))
    _wis_korte_runs(vlag, min_len)

    for r, b in zip(resultaten, vlag):
        r.bocht = bool(b)


def _wis_korte_runs(vlag, min_len):
    """Zet aaneengesloten runs korter dan `min_len` op de waarde van hun buren (in place).
    Runs aan de rand tellen alleen mee als ze aan hun ene buur-run grenzen."""
    n = len(vlag)
    i = 0
    while i < n:
        j = i
        while j < n and vlag[j] == vlag[i]:
            j += 1
        if (j - i) < min_len and not (i == 0 and j == n):
            buur = vlag[i - 1] if i > 0 else vlag[j]
            vlag[i:j] = buur
        i = j


def maak_voorvulling(resultaten, idx, fps, n_lm=33):
    """
    Een startskelet voor frame `idx`, dat zelf géén pose heeft: de GUI zet dit neer als
    de gebruiker handmatig een skelet gaat plaatsen, zodat hij bestaande punten alleen
    hoeft te corrigeren in plaats van alle acht opnieuw aan te wijzen.

    Ligt het frame tússen twee pose-frames en is het gat kort (≤ `INTERP_MAX_S` — dezelfde
    grens die `_begrens_interpolatie` in de smoothing hanteert), dan wordt er lineair
    tussen die twee geïnterpoleerd. Over een langer gat is een blend van twee poses
    anatomische onzin (de ledematen smelten door elkaar) en is de dichtstbijzijnde pose,
    hoe verouderd ook, een eerlijker startpunt. Is er in de hele analyse geen enkele pose,
    dan komt alles in het beeldmidden te staan met visibility 0 — onzichtbaar, zodat er
    niets op het scherm staat dat er niet is.

    Retourneert altijd een lijst van precies `n_lm` `Landmark`s (nooit None): de
    serialisatie schrijft in een vaste (n, 33, 3)-array.
    """
    if not (0 <= idx < len(resultaten)):
        return [Landmark(0.5, 0.5, 0.0, 0.0) for _ in range(n_lm)]

    def _bruikbaar(i):
        r = resultaten[i]
        return r.pose_gevonden and isinstance(r.lm, (list, tuple)) and len(r.lm) >= n_lm

    voor = next((i for i in range(idx - 1, -1, -1) if _bruikbaar(i)), None)
    na = next((i for i in range(idx + 1, len(resultaten)) if _bruikbaar(i)), None)
    if voor is None and na is None:
        return [Landmark(0.5, 0.5, 0.0, 0.0) for _ in range(n_lm)]

    max_gat = max(1, int(round(INTERP_MAX_S * (fps or 30.0))))
    if voor is not None and na is not None and (na - voor - 1) <= max_gat:
        t = (idx - voor) / (na - voor)
        a, b = resultaten[voor].lm, resultaten[na].lm
        return [Landmark(a[j].x + (b[j].x - a[j].x) * t,
                         a[j].y + (b[j].y - a[j].y) * t,
                         0.0,
                         a[j].visibility + (b[j].visibility - a[j].visibility) * t)
                for j in range(n_lm)]

    bron = voor if na is None else (na if voor is None else
                                    (voor if idx - voor <= na - idx else na))
    return [Landmark(p.x, p.y, 0.0, p.visibility) for p in resultaten[bron].lm[:n_lm]]


class KnipAfgebroken(Exception):
    """De gebruiker heeft het knippen gestopt (zie knip_fragmenten/stop_check)."""


def _veilige_bestandsnaam(naam):
    """Maakt van een fragmenttitel een bestandsnaam die Windows accepteert."""
    schoon = "".join(c if c.isalnum() or c in " -_." else "_" for c in (naam or "").strip())
    return schoon.strip(" .")[:80]


def knip_fragmenten(bron_pad, fragmenten, doelmap, progress_callback=None,
                    stop_check=None, fps=None, deinterlacen=False):
    """
    Schrijft de gemarkeerde stukken van een lange opname weg als losse videobestanden
    (ROADMAP fase 8) en retourneert de paden, in dezelfde volgorde als `fragmenten`.

    `fragmenten` is een lijst `(start_frame, eind_frame, naam)` — beide grenzen **inclusief**,
    `naam` wordt de bestandsnaam (zonder extensie). **Er wordt exact op de gemarkeerde frames
    geknipt**: geen marge erbij of eraf. De trainer kijkt tijdens het markeren naar het beeld
    en bepaalt de grenzen zelf; het programma hoort daar niet stilzwijgend seconden bij te
    doen. (Aan de voorkant zou lucht de doelkeuze zelfs moeilijker maken: `DoelKiezer` krijgt
    frame 0 van de clip, en dat is nu precies het beeld waarop "start" gedrukt werd.)

    **Eén sequentiële pass**: elk frame gaat naar de writer van elk fragment waarin het valt,
    zodat de video precies één keer gedecodeerd wordt en er nérgens geseekt hoeft te worden —
    zelfde motief als de eigen leeslus in `schaats_yolo._detecteer_alles`. Overlappende
    fragmenten mogen daardoor gewoon (het frame gaat dan naar twee writers).

    Codec `mp4v`: die zit in de opencv-python-wheel, terwijl `avc1` op Windows vaak ontbreekt.
    Er wordt dus her-gecodeerd; voor pose-detectie is dat kwaliteitsverlies verwaarloosbaar.
    Een stream-copy (ffmpeg `-c copy`) zou dat vermijden maar kan alleen op een keyframe
    beginnen — dat is precies de stilzwijgende marge die hier niet gewenst is, en het maakt
    frame 0 van de clip een ánder beeld dan waarop je "start" drukte.

    `progress_callback(frame_nr, totaal)` en `stop_check() -> bool` (afbreken; de reeds
    geschreven bestanden worden dan opgeruimd).
    """
    cap = open_video(bron_pad, deinterlacen)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {bron_pad}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(doelmap, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    taken, gebruikt = [], set()
    for i, (start, eind, naam) in enumerate(fragmenten):
        stam = _veilige_bestandsnaam(naam) or f"fragment_{i + 1}"
        if stam.lower() in gebruikt:                # twee fragmenten met dezelfde titel
            stam = f"{stam}_{i + 1}"
        gebruikt.add(stam.lower())
        taken.append({"start": int(start), "eind": int(eind), "writer": None,
                      "pad": os.path.join(doelmap, f"{stam}.mp4")})

    # Voorbij het laatste eindframe hoeft er niets meer gedecodeerd te worden — bij een
    # fragment aan het begin van een opname van een half uur scheelt dat vrijwel alles.
    laatste = max((t["eind"] for t in taken), default=-1)
    geschreven = set()
    try:
        idx = 0
        while idx <= laatste:
            if stop_check is not None and stop_check():
                raise KnipAfgebroken()
            # grab/retrieve i.p.v. read(): frames die in géén enkel fragment vallen hoeven
            # alleen doorgeschoven te worden, niet gedecodeerd. Gemeten op een 1080p-opname
            # scheelt dat ~9 → ~3 ms per frame, en juist bij een fragment ver in een opname
            # van 23 minuten is dat het leeuwendeel van het werk.
            if not cap.grab():
                break                               # video korter dan CAP_PROP_FRAME_COUNT meldde
            actief = [t for t in taken if t["start"] <= idx <= t["eind"]]
            if actief:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                for t in actief:
                    if t["writer"] is None:
                        t["writer"] = cv2.VideoWriter(t["pad"], fourcc, fps, (w, h))
                        if not t["writer"].isOpened():
                            raise IOError(f"Kan fragment niet schrijven: {t['pad']}")
                        geschreven.add(t["pad"])
                    t["writer"].write(frame)
            idx += 1
            if progress_callback is not None:
                progress_callback(idx, max(totaal, laatste + 1))
    except BaseException:
        for t in taken:
            if t["writer"] is not None:
                t["writer"].release()
        cap.release()
        for pad in geschreven:
            try:
                os.remove(pad)
            except OSError:
                pass
        raise
    for t in taken:
        if t["writer"] is not None:
            t["writer"].release()
    cap.release()

    ontbreekt = [t for t in taken if t["writer"] is None]
    if ontbreekt:
        # Een fragment dat volledig voorbij het einde van de video lag: melden i.p.v. een
        # onbestaand pad de batch-flow in te sturen.
        raise IOError(f"{len(ontbreekt)} fragment(en) vielen buiten de video "
                      f"({os.path.basename(bron_pad)}) en konden niet geknipt worden.")
    return [t["pad"] for t in taken]


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


def _strek_ratio(r, been, w, h):
    """Beenstrekking als schaalvrije ratio: rechte-lijn heup→enkel gedeeld door de som
    van de botlengtes (heup→knie + knie→enkel). ~1.0 = volledig gestrekt (knie op de
    lijn), lager = gebogen. Schaalvrij (dichtbij/ver weg maakt niet uit) én zonder vaste
    pixeldrempel — precies wat nodig is om het strek-máximum (= einde afzet) als piek te
    vinden i.p.v. met een harde hoekgrens. Op float-pixels: op afstand is het onderbeen
    maar tientallen pixels."""
    h_idx, k_idx, e_idx = (L_HIP, L_KNEE, L_ANKLE) if been == 'links' else (R_HIP, R_KNEE, R_ANKLE)
    heup  = np.array(_lm_px(r, h_idx, w, h))
    knie  = np.array(_lm_px(r, k_idx, w, h))
    enkel = np.array(_lm_px(r, e_idx, w, h))
    bot = float(np.hypot(*(knie - heup)) + np.hypot(*(enkel - knie)))
    return float(np.hypot(*(enkel - heup)) / bot) if bot > 1e-6 else 0.0


def bepaal_afzet_uit_strek(resultaten, w, h, fps):
    """
    Bepaalt de afzet-voltooiing uit de **beenstrekking** i.p.v. de per-frame 2-van-3-
    stem (`detecteer_gewicht_op_been`). Het standbeen is een hele fase "gestrekt": van
    rechtop komen na de plaatsing (been ≈ verticaal → onderbeenhoek ~80-90°) tot de
    volledige zijwaartse afzet (been plat → ~50°). We nemen als afzet-voltooiing niet
    het strek-máximum (dat valt op het rechtop-komen → een misleidend hoge hoek), maar
    binnen die gestrekte fase het frame met de **vlakste (laagste) onderbeenhoek** = de
    eigenlijke, meest horizontale push. Dat volgt de observatie "been maximaal = afzet
    klaar" én leest de hoek op het betekenisvolle moment; het rechtop-komen (hoge hoek)
    valt automatisch af en de recovery zit al buiten het strek-plateau.

    Werkt per aaneengesloten **stand-run** (frames waarin `r.been` gelijk blijft, met
    pose + lm_data — de cyclus-toewijzing levert precies één push per run). Per run:
    het gestrekte plateau = het aaneengesloten venster rond `argmax(strek_ratio)` waar
    de ratio binnen `STREK_PLATEAU_BAND` onder z'n maximum blijft; de afzet-voltooiing
    `c` = het plateau-frame met de kleinste onderbeenhoek. Zet per frame `r.strek_ratio`
    (gesmoothd, voor HUD/debug) en `r.gewicht_erop` (True t/m `c`, daarna False → het
    event spant de hele load+push en eindigt op de vlakste-hoek-frame).

    "Voorspelbare slagtijd" zit erin als **zachte prior**: uit de mediane run-lengte
    schatten we de halve slagperiode; een run die daar een fractie (STREK_MIN_SLAG_FRAC)
    korter dan is, is vrijwel zeker een ruis-omslag van de been-toewijzing en levert
    géén afzet (voorkomt spookafzetten door kortstondige L/R-flips). LET OP: bij een
    versnellende start zijn de slagen korter — de fractie staat daarom laag zodat alleen
    écht korte (ruis-)runs sneuvelen, niet de snelle openingsslagen. De mediaan loopt
    alleen over runs van minstens `STREK_MIN_RUN_S`: nemen de ruis-runs eraan deel, dan
    zakt de schatting — en dus de drempel — precies wanneer je hem nodig hebt.

    Twee soorten runs leveren wél een event — je wilt zien dát er iets gebeurde — maar met
    `afzet_onvolledig` gemarkeerd, zodat de GUI en de bibliotheek-statistiek ze buiten
    gem/min/max houden:

    - **`ONV_AFGEKAPT`** — de run eindigt niet op een beenwissel maar op het einde van de
      video of van het pose-segment. De push is daar niet afgemaakt, het plateau bevat
      alleen het rechtop-komen en de "vlakste hoek" is systematisch veel te steil (gemeten:
      +15 tot +25° op de laatste afzet). Een run die aan het *begin* van een segment is
      afgekapt telt gewoon mee: daar mist alleen de load-fase, terwijl de push-voltooiing
      (waar de hoek vandaan komt) wél in beeld is.
    - **`ONV_GEEN_PUSH`** — de run is lang genoeg (`min_run`) en eindigt netjes op een
      beenwissel, maar het strek-plateau beslaat alléén de opricht-fase; de vlakste hoek
      binnen dat plateau is dan nog steeds het rechtop-komen. Herkenbaar aan de meetkunde:
      het onderbeen staat bij "voltooiing" minder dan `STREK_MIN_HELLING_DEG` uit het lood,
      dus er is niet opzij geduwd. Kwam voor in 3 van de 100 events in de bibliotheek.

    Vereist de globale cyclus-been-toewijzing. Retourneert de geschatte slagperiode
    (frames), puur informatief.
    """
    # Aaneengesloten runs van gelijk standbeen (binnen frames met pose + lm_data).
    # `afgekapt` = de run eindigt niet op een beenwissel maar op een detectiegat of het
    # einde van de video; de afzet is dan niet uit-geobserveerd.
    runs, huidig = [], None
    for i, r in enumerate(resultaten):
        heeft = r.pose_gevonden and r.lm_data is not None and r.been in ('links', 'rechts')
        if not heeft:
            if huidig is not None:
                huidig['afgekapt'] = True
                runs.append(huidig); huidig = None
            continue
        if huidig is None or huidig['been'] != r.been:
            if huidig is not None:
                runs.append(huidig)
            huidig = {'been': r.been, 'idx': [], 'afgekapt': False}
        huidig['idx'].append(i)
    if huidig is not None:
        huidig['afgekapt'] = True          # video is op → laatste push niet afgemaakt
        runs.append(huidig)

    # Halve slagperiode robuust schatten: alleen runs die lang genoeg zijn om überhaupt
    # een halve slag te kunnen zijn (zie STREK_MIN_RUN_S). Zonder zulke runs valt hij
    # terug op álle runs — dan is er niets beters.
    min_abs = max(2, int(round(STREK_MIN_RUN_S * fps))) if fps > 0 else 2
    lengtes = [len(run['idx']) for run in runs]
    echte = [n for n in lengtes if n >= min_abs] or lengtes
    half_periode = float(np.median(echte)) if echte else 0.0
    min_run = max(min_abs, int(round(STREK_MIN_SLAG_FRAC * half_periode))) if half_periode else min_abs

    win = max(SMOOTH_POLY + 2, int(round(STREK_SMOOTH_S * fps)))
    if win % 2 == 0:
        win += 1

    for run in runs:
        idx, been = run['idx'], run['been']
        ratio = np.array([_strek_ratio(resultaten[i], been, w, h) for i in idx])
        ratio_s = _savgol(ratio, win, SMOOTH_POLY) if len(idx) >= 3 else ratio
        for t, i in enumerate(idx):
            resultaten[i].strek_ratio = round(float(ratio_s[t]), 3)
            resultaten[i].afzet_onvolledig = ONV_AFGEKAPT if run['afgekapt'] else None
        if len(idx) < min_run:                    # te kort → ruis-omslag, geen afzet
            for i in idx:
                resultaten[i].gewicht_erop = False
            continue

        # Gestrekte fase = het aaneengesloten plateau rond de strek-piek (ratio binnen de
        # band onder z'n maximum). Recovery (knie buigt → ratio zakt) valt er buiten.
        piek = int(np.argmax(ratio_s))
        drempel = ratio_s[piek] - STREK_PLATEAU_BAND
        lo = hi = piek
        while lo - 1 >= 0 and ratio_s[lo - 1] >= drempel:
            lo -= 1
        while hi + 1 < len(idx) and ratio_s[hi + 1] >= drempel:
            hi += 1

        # Binnen het plateau de vlakste (laagste) onderbeenhoek = afzet-voltooiing. Het
        # rechtop-komen (hoge hoek) valt zo automatisch af.
        e_idx, k_idx = (L_ANKLE, L_KNEE) if been == 'links' else (R_ANKLE, R_KNEE)
        hoeken = [bereken_hoek_tov_ijs(_lm_px(resultaten[idx[t]], e_idx, w, h),
                                       _lm_px(resultaten[idx[t]], k_idx, w, h),
                                       resultaten[idx[t]].horizon_deg)
                  for t in range(lo, hi + 1)]
        c = lo + int(np.argmin(hoeken))
        for t, i in enumerate(idx):
            resultaten[i].gewicht_erop = (t <= c)

        # Meetkundige eindtoets: staat het onderbeen bij de "voltooiing" nog vrijwel
        # rechtop, dan besloeg het plateau alleen het opricht-moment en is er geen
        # zijwaartse afzet waargenomen. Het event blijft staan, maar de hoek is geen
        # meting. Een al afgekapte run houdt zijn eigen (bekende) reden.
        if not run['afgekapt'] and (90.0 - min(hoeken)) < STREK_MIN_HELLING_DEG:
            for i in idx:
                resultaten[i].afzet_onvolledig = ONV_GEEN_PUSH

    return half_periode * 2.0


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
    hoek = round(float(rec.hoek), 1)     # gewone float: gaat zo de DB/CSV/JSON in
    r.hoek_correctie = round(hoek - oude_hoek, 1)
    r.hoek_betrouwbaar = rec.betrouwbaar
    return hoek


def _zet_smooth_hoek(resultaten, smooth_n):
    """
    Vult `r.smooth_hoek`: een **gecentreerd** (zero-lag) gemiddelde van `r.hoek` over
    `smooth_n` frames, per aaneengesloten reeks van hetzelfde standbeen.

    Bewust géén achterwaartse deque meer: zo'n trailing gemiddelde ijlt een halve
    venster na, en omdat de afzethoek naar z'n minimum toe daalt lag de gerapporteerde
    hoek daardoor systematisch te steil. Een detectiegat óf een beenwissel breekt de
    reeks, zodat er nooit over een gat heen of tussen twee benen door wordt gemiddeld.
    Puur een weergave-grootheid (HUD/grafiek); de tabel rapporteert de hoek van één
    frame (zie `segmenteer_afzetten`).
    """
    half = max(0, (int(smooth_n) - 1) // 2)

    def vul(run):
        hoeken = np.array([r.hoek for r in run], dtype=float)
        for t, r in enumerate(run):
            lo, hi = max(0, t - half), min(len(run), t + half + 1)
            r.smooth_hoek = round(float(hoeken[lo:hi].mean()), 1)

    run = []
    for r in list(resultaten) + [None]:
        heeft_hoek = r is not None and r.hoek is not None
        aansluitend = heeft_hoek and (not run or (r.been == run[-1].been
                                                 and r.frame_nr == run[-1].frame_nr + 1))
        if run and not aansluitend:
            vul(run)
            run = []
        if heeft_hoek:
            run.append(r)


def verwerk_afgeleiden(resultaten, w, h, fps, smooth_n=5, threshold=0.015, cyclus=True,
                       perspectief=None, afzet_strek=True):
    """
    Vult per frame de afgeleide grootheden in (afzetbeen, hoek, kniehoek, gewicht),
    berekend uit resultaat.lm. Wordt ná de offline smoothing aangeroepen zodat alles
    op de gladde landmarks is gebaseerd.

    Het afzetbeen wordt cyclus-bewust toegewezen (`wijs_afzetbeen_cyclus`) als
    `cyclus`, anders per frame (`bepaal_afzetbeen`). De afzet-voltooiing komt met
    `afzet_strek` (default) uit de beenstrekking (`bepaal_afzet_uit_strek`: strek-
    máximum = einde afzet); zonder (of zonder `cyclus`) uit de per-frame gewicht-stem
    (`detecteer_gewicht_op_been`). De rest is sequentieel i.v.m. de
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
    # Pass 1: pixelcoördinaten voor alle pose-frames. De onvolledig-vlag hoort bij de
    # been-runs van déze doorrekening (na een skelet-edit kunnen die verschuiven), dus
    # eerst schoon.
    #
    # Een bochtframe krijgt bewust géén `lm_data`: dáár zit de hele uitsluiting. De
    # been-toewijzing, de afzet-voltooiing en de event-segmentatie bouwen hun segmenten
    # allemaal op "pose én lm_data", dus zij zien de bocht vanzelf als een detectiegat —
    # zonder dat er ook maar iets aan de meetlogica verandert. Het skelet blijft wél
    # getekend (dat leest `r.lm`), zodat je in de weergave ziet wat er gebeurde.
    for r in resultaten:
        bruikbaar = r.pose_gevonden and r.lm is not None and not r.bocht
        r.lm_data = get_landmarks(r.lm, w, h) if bruikbaar else None
        r.afzet_onvolledig = None

    # Been-toewijzing (globaal, cyclus-bewust) vóór de per-frame afgeleiden.
    if cyclus:
        wijs_afzetbeen_cyclus(resultaten, h, fps)

    # Afzet-voltooiing uit de beenstrekking (strek-máximum = einde afzet). Vereist de
    # globale cyclus-been-toewijzing; zonder cyclus (diagnose-stand) valt hij terug op
    # de per-frame gewicht-stem hieronder.
    gebruik_strek = afzet_strek and cyclus
    if gebruik_strek:
        bepaal_afzet_uit_strek(resultaten, w, h, fps)

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
    enkel_hist  = {'links': deque(maxlen=10), 'rechts': deque(maxlen=10)}
    heup_hist   = deque(maxlen=10)

    for i, r in enumerate(resultaten):
        if not (r.pose_gevonden and r.lm_data is not None):
            heup_hist.clear()
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

        gewicht_stem, kniehoek, signalen = detecteer_gewicht_op_been(
            been, lm_data, enkel_hist, heup_hist, w, h, threshold)

        r.been = been
        r.hoek = hoek
        r.kniehoek = kniehoek
        if not gebruik_strek:            # met strek staat r.gewicht_erop al globaal gezet
            r.gewicht_erop = gewicht_stem
        r.signalen = signalen           # de 3 oude signalen blijven als HUD-diagnose

    # Pass 3: het weergave-gemiddelde van de hoek — gecentreerd, dus zonder naijlen.
    _zet_smooth_hoek(resultaten, smooth_n)


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
              perspectief=None, waarschuwing_callback=None, bocht=True,
              deinterlacen=None):
    """
    Volledige analyse-pijplijn: multi-pose detectie + doel-tracking (streaming),
    daarna offline landmark-smoothing en het berekenen van de afgeleide grootheden.
    Retourneert (VideoInfo, lijst[FrameResultaat]).

    Met `bocht` (default) worden bochtframes gemarkeerd (`bepaal_bocht_reeks`) en leveren
    ze geen afzetmeting. Deze backend detecteert streaming per frame en slaat — anders
    dan de YOLO-backend — geen frames over; hier is het dus puur een meetfilter.

    `waarschuwing_callback(tekst)` bestaat voor signatuur-compatibiliteit met de YOLO-
    backend (die meldt er stille terugvallen in de doelkeuze mee). Deze backend kiest
    zijn doel streaming per frame en heeft nog geen melding die hier past.

    De horizoncorrectie is óf een vaste `horizon_deg` (handmatige lijn), óf — bij
    `auto_horizon` — **per frame** gedetecteerd (`bepaal_horizon_reeks`, voor een
    schommelende camera). In beide gevallen komt de waarde per frame in `r.horizon_deg`.
    Bij auto is er een tweede video-pass; die deelt de voortgangsbalk met de detectie
    (elk een helft) via `fase_voortgang`, zodat het één doorlopende balk blijft.

    Met `perspectief` (PerspectiefConfig) vervangt de baanlijn-kalibratie de horizon-
    machinerie volledig (zie `zet_horizon`/`verwerk_afgeleiden`).
    """
    info = video_info(input_pad, force_fps)
    # None = zelf uitzoeken (CLI-gemak); de GUI bepaalt het in de dialoog en geeft een
    # expliciete bool, zodat de keuze zichtbaar is en in de instellingen belandt.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    if perspectief is not None:
        auto_horizon = False     # vaste camera per aanname; kalibratie kent de kanteling al

    # Auto-horizon = twee passes → één doorlopende balk (detectie 0–50%, horizon 50–100%).
    det_cb = fase_voortgang(progress_callback, 0, 2) if auto_horizon else progress_callback
    hor_cb = fase_voortgang(progress_callback, 1, 2) if auto_horizon else progress_callback

    resultaten = list(analyseer_frames(
        input_pad, model_pad, force_fps=force_fps, num_poses=num_poses,
        doel_punt=doel_punt, progress_callback=det_cb, deinterlacen=deinterlacen))

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)

    if bocht:
        bepaal_bocht_reeks(resultaten, info.w, info.h, info.fps)

    zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief, deinterlacen=deinterlacen)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten


def zet_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps=None,
                progress_callback=None, perspectief=None, deinterlacen=False):
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
                                        force_fps=force_fps, progress_callback=progress_callback,
                                        deinterlacen=deinterlacen)
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
        # Is een van beide delen onvolledig, dan geldt dat voor het geheel; de reden van
        # het láátste deel weegt het zwaarst, want dáár wordt de hoek afgelezen.
        onvolledig=b.onvolledig or a.onvolledig,
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


# ── Serialisatie (fase 0) ────────────────────────────────────────────────────────
# Een analyse opslaan/terugladen zonder de video opnieuw te analyseren. Alleen de
# landmarks + pose_gevonden + horizon zijn bron van waarheid; de afgeleiden
# (been/hoek/gewicht/events) worden bij het terugladen herberekend met
# verwerk_afgeleiden() + segmenteer_afzetten(). Backend-onafhankelijk: r.lm is óf een
# MediaPipe-landmarklijst óf een Landmark-lijst — beide met .x/.y/.visibility.

def resultaten_naar_arrays(resultaten, info):
    """
    Serialiseert een geanalyseerde FrameResultaat-lijst naar platte numpy-arrays.
    z wordt bewust weggelaten (nergens gebruikt). Retourneert een dict geschikt voor
    np.savez_compressed; de video-eigenschappen gaan als meta mee zodat het terugladen
    geen video nodig heeft.
    """
    n = len(resultaten)
    landmarks     = np.zeros((n, 33, 3), dtype=np.float32)   # x, y, visibility
    pose_gevonden = np.zeros(n, dtype=bool)
    horizon       = np.zeros(n, dtype=np.float32)
    middellijn    = np.full((n, 2), np.nan, dtype=np.float32)  # dev l_knie, r_knie (px)
    bocht         = np.zeros(n, dtype=bool)
    for i, r in enumerate(resultaten):
        pose_gevonden[i] = r.pose_gevonden
        horizon[i]       = r.horizon_deg
        bocht[i]         = r.bocht
        if r.middellijn_dev:
            for k, naam in enumerate(('l_knie', 'r_knie')):
                if r.middellijn_dev.get(naam) is not None:
                    middellijn[i, k] = r.middellijn_dev[naam]
        if r.pose_gevonden and r.lm is not None:
            for j, p in enumerate(r.lm):
                landmarks[i, j, 0] = p.x
                landmarks[i, j, 1] = p.y
                landmarks[i, j, 2] = p.visibility
    return {
        'landmarks':     landmarks,
        'pose_gevonden': pose_gevonden,
        'horizon_deg':   horizon,
        'middellijn_dev': middellijn,
        # De bocht-vlag is géén afgeleide: bij de YOLO-backend markeert hij ook de frames
        # waarvan de detectiepass de inferentie heeft overgeslagen, en dat valt uit de
        # landmarks niet te herleiden (die zijn er juist niet). Dus opslaan.
        'bocht':         bocht,
        'w':      np.int32(info.w),
        'h':      np.int32(info.h),
        'fps':    np.float32(info.fps),
        'totaal': np.int32(info.totaal),
    }


def arrays_naar_resultaten(arrays):
    """
    Inverse van resultaten_naar_arrays: bouwt een verse (VideoInfo, FrameResultaat-lijst)
    met platte Landmark-tuples in .lm (z=0). De afgeleiden zijn nog leeg — draai
    verwerk_afgeleiden() + segmenteer_afzetten() om ze te vullen. frame_nr/tijd worden
    exact gereconstrueerd uit de frame-index en fps, zoals analyseer_frames ze zet.
    """
    landmarks     = arrays['landmarks']
    pose_gevonden = arrays['pose_gevonden']
    horizon       = arrays['horizon_deg']
    middellijn    = arrays['middellijn_dev'] if 'middellijn_dev' in arrays else None
    bocht         = arrays['bocht'] if 'bocht' in arrays else None   # ontbreekt in oude npz's
    info = VideoInfo(int(arrays['w']), int(arrays['h']),
                     float(arrays['fps']), int(arrays['totaal']))
    fps = info.fps
    resultaten = []
    for i in range(len(landmarks)):
        r = FrameResultaat(frame_nr=i, tijd=i / fps if fps > 0 else 0.0)
        r.pose_gevonden = bool(pose_gevonden[i])
        r.horizon_deg   = float(horizon[i])
        r.bocht         = bool(bocht[i]) if bocht is not None else False
        if middellijn is not None and not np.all(np.isnan(middellijn[i])):
            r.middellijn_dev = {
                naam: (float(middellijn[i, k]) if not np.isnan(middellijn[i, k]) else None)
                for k, naam in enumerate(('l_knie', 'r_knie'))}
        if r.pose_gevonden:
            r.lm = [Landmark(float(x), float(y), 0.0, float(v))
                    for x, y, v in landmarks[i]]
        resultaten.append(r)
    return info, resultaten


def sla_landmarks_op(pad, resultaten, info):
    """Schrijft de analyse (landmarks + horizon + meta) gecomprimeerd naar een .npz."""
    np.savez_compressed(pad, **resultaten_naar_arrays(resultaten, info))


def laad_landmarks(pad):
    """Laadt een .npz terug naar (VideoInfo, FrameResultaat-lijst); afgeleiden nog leeg."""
    with np.load(pad) as arrays:
        return arrays_naar_resultaten(arrays)


def segmenteer_afzetten(resultaten, min_lengte=3, alternerend=True):
    """
    Groepeert per-frame resultaten tot afzet-events: aaneengesloten frames
    waarin hetzelfde been afzetbeen is én het gewicht er nog op zit.
    `min_lengte` filtert ruis (te korte, onbetrouwbare detecties) eruit.
    Als `alternerend`, wordt daarna de L/R-alternatie-regel toegepast
    (`forceer_alternerend`): opgesplitste afzetten samenvoegen, onmogelijke
    herhalingen markeren.

    `hoek` is de hoek van het **laatste frame** van het event — precies het frame dat
    `bepaal_afzet_uit_strek` als afzet-voltooiing koos (de vlakste onderbeenhoek binnen
    het strek-plateau). Dus `r.hoek`, niet `r.smooth_hoek`: een gemiddelde over de
    frames vóór dat moment ligt op een dalende hoek per definitie te hoog (te steil).
    `min_hoek`/`max_hoek` komen uit dezelfde reeks, zodat de tabel één definitie van
    "de hoek" gebruikt.

    Een event erft `onvolledig` van zijn frames (`FrameResultaat.afzet_onvolledig`): de
    afzet liep nog toen de video/het pose-segment ophield (`ONV_AFGEKAPT`), of er is
    binnen de run helemaal geen zijwaartse push waargenomen (`ONV_GEEN_PUSH`). In beide
    gevallen is de hoek het rechtop-komen i.p.v. de voltooide push; zo'n event blijft
    zichtbaar, maar hoort buiten gemiddelde/min/max te vallen (GUI + bibliotheeklijst
    doen dat).
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
                onvolledig=laatste.afzet_onvolledig,
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
            huidig['hoeken'].append(r.hoek)
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
                    horizon_deg=0.0, auto_horizon=False, save_npz=None, from_npz=None,
                    bocht=True, deinterlacen=None):
    """
    CLI-analyse in twee passes: eerst detecteren/tracken/smoothen (nodig omdat de
    offline smoothing álle frames vereist), daarna de video opnieuw lezen en de
    overlay erop tekenen en wegschrijven.

    Met `save_npz` worden de landmarks (fase 0) na de analyse weggeschreven. Met
    `from_npz` wordt de analyse overgeslagen: de landmarks worden uit dat .npz geladen
    en alleen de afgeleiden opnieuw berekend — zo toon je aan dat een teruggeladen
    analyse dezelfde tabel/overlay geeft zónder de video opnieuw te analyseren.
    """
    def toon_voortgang(frame_nr, totaal):
        if frame_nr % 30 == 0:
            # `totaal` komt uit CAP_PROP_FRAME_COUNT en klopt op VFR-.MOV's geregeld niet;
            # klem het percentage zodat de CLI geen 103% meldt.
            pct = min(100.0, frame_nr / totaal * 100) if totaal > 0 else 0
            print(f"  {frame_nr}/{max(totaal, frame_nr)} frames ({pct:.0f}%)")

    # Eén keer bepalen en aan beide passes meegeven: tekent pass 2 de overlay op ándere
    # pixels dan pass 1 gemeten heeft, dan klopt het skelet niet meer met het beeld.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    if deinterlacen:
        print("[INFO] Interlaced bron gedetecteerd: kamtanden worden weggefilterd.")

    if from_npz:
        print(f"[INFO] Landmarks laden uit {from_npz} (geen detectie) ...")
        info, resultaten = laad_landmarks(from_npz)
        # Alleen de afgeleiden herberekenen; horizon_deg zit al per frame in het .npz.
        verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold)
    else:
        print("[INFO] Pass 1/2: detectie + tracking + smoothing ...")
        info, resultaten = analyseer(
            input_pad, model_pad, smooth_n, threshold, force_fps,
            num_poses=num_poses, doel_punt=doel_punt, smooth_landmarks=smooth_landmarks,
            progress_callback=toon_voortgang, horizon_deg=horizon_deg, auto_horizon=auto_horizon,
            bocht=bocht, deinterlacen=deinterlacen)
        n_bocht = sum(1 for r in resultaten if r.bocht)
        if n_bocht:
            print(f"[INFO] {n_bocht} van {len(resultaten)} frames als bocht gemarkeerd "
                  f"(geen meting)")
        if save_npz:
            sla_landmarks_op(save_npz, resultaten, info)
            print(f"[INFO] Landmarks opgeslagen: {save_npz}")

    print(f"[INFO] Video: {info.w}×{info.h} @ {info.fps:.1f}fps, {info.totaal} frames")
    print(f"[INFO] Pass 2/2: overlay tekenen → {output_pad}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_pad, fourcc, info.fps, (info.w, info.h))
    cap    = open_video(input_pad, deinterlacen)
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
    parser.add_argument("--deinterlace", dest="deinterlace", action="store_true", default=None,
                        help="Filter kamtanden weg (interlaced camcorderbeeld). Standaard "
                             "wordt dit per video zelf bepaald; deze vlag forceert het aan.")
    parser.add_argument("--no-deinterlace", dest="deinterlace", action="store_false",
                        help="Nooit deinterlacen, ook niet als de video interlaced lijkt "
                             "(om een A/B te draaien).")
    parser.add_argument("--no-bocht", action="store_true",
                        help="Bochtdetectie uitzetten; ook bochtframes leveren dan (onbruikbare) "
                             "afzetmetingen op.")
    parser.add_argument("--save-npz", default=None, metavar="PAD",
                        help="Schrijf na de analyse de landmarks weg naar dit .npz (fase 0).")
    parser.add_argument("--from-npz", default=None, metavar="PAD",
                        help="Sla de detectie over en laad de landmarks uit dit .npz; "
                             "berekent alleen de afgeleiden en tekent de overlay opnieuw.")
    args = parser.parse_args()

    # Als er geen --input is meegegeven, vraag er interactief om
    if not args.input:
        args.input = input("Welke video wil je analyseren? (pad naar bestand): ").strip().strip('"')

    # Check of het bestand echt bestaat
    if not os.path.isfile(args.input):
        print(f"Bestand niet gevonden: {args.input}")
        exit(1)

    standaard_naam = "pose_landmarker_heavy.task" if args.heavy else "pose_landmarker_full.task"
    model_pad = args.model or os.path.join(app_dir(), standaard_naam)
    if not args.from_npz and not os.path.isfile(model_pad):
        # Bij --from-npz wordt niet gedetecteerd, dus is het model niet nodig.
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
        deinterlacen=args.deinterlace,
        save_npz=args.save_npz,
        from_npz=args.from_npz,
        bocht=not args.no_bocht,
    )
