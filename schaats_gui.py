"""
Schaatser Analyse GUI
=====================
Visuele interface bij schaats_analyse.py: speelt de video af met de
skelet/afzetbeen-overlay live erop, en toont een tabel met alle afzethoeken.

Gebruik:
    python schaats_gui.py

Vereisten (naast schaats_analyse.py z'n dependencies):
    pip install PySide6
"""

import os
import sys
import csv
import math
import shutil
import tempfile
import threading
import time

# ── Uitvoer eerst: bevroren is er geen console ──────────────────────────────────
# Vóór álle andere imports, want een gebundelde .exe (PyInstaller --windowed) heeft geen
# console: sys.stdout/stderr zijn dan None en alles wat die stroom écht aanspreekt loopt
# stuk — de tqdm-balk van ultralytics/rtmlib, de logging-handler die ultralytics bij de
# import aanhaakt, elke sys.stdout.write. Dat moet dus geregeld zijn vóór de eerste van
# die imports, en dus ook vóór _start_opstartscherm() hieronder, dat al op moduleniveau
# een venster neerzet. schaats_omgeving is stdlib-only: ~1 ms, threading stond er al.
# In de repo-omgeving gebeurt er niets, tenzij SCHAATSANALYSE_LOG gezet is.
import schaats_omgeving

LOGPAD = schaats_omgeving.start_logboek() if __name__ == "__main__" else None

# ── Qt eerst, en meteen een opstartscherm ───────────────────────────────────────
# Bewust vóór alle andere imports: de rest van deze module trekt cv2/numpy binnen en
# (bij het eerste gebruik) torch/ultralytics, en juist op een koude machine kost dat
# seconden waarin er niets op het scherm gebeurt en de gebruiker denkt dat de app niet
# opgestart is. Alleen de Qt-import (~0,1 s) gaat eraan vooraf, zodat er binnen een
# fractie van een seconde een venstertje staat dat vertelt wat er gebeurt.
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QPointF, QEventLoop, QEvent, QSize, QRect, QPoint,
    QMargins,
)
from PySide6.QtGui import (
    QImage, QPixmap, QAction, QColor, QPainter, QPen, QShortcut, QKeySequence, QFont,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QCheckBox, QFileDialog, QMessageBox, QProgressBar,
    QHeaderView, QAbstractItemView, QToolBar, QStackedWidget, QSpinBox,
    QDoubleSpinBox, QDialog, QRadioButton, QComboBox, QFormLayout,
    QListWidget, QListWidgetItem, QLineEdit, QPlainTextEdit, QInputDialog,
    QDialogButtonBox, QToolTip, QLayout, QSizePolicy, QTabWidget, QProgressDialog,
    QSplashScreen,
)


class Opstartscherm(QSplashScreen):
    """Het venstertje dat tijdens het opstarten laat zien dát er iets gebeurt.

    Getekend in code (geen afbeeldingsbestand): een plaatje laden zou weer een schijf-
    toegang zijn op precies het moment dat we die willen vermijden.
    """
    BREEDTE, HOOGTE = 460, 180

    def __init__(self):
        super().__init__(self._achtergrond())
        # QSplashScreen staat standaard altijd-bovenop. Tijdens het opstarten kan er een
        # modale melding komen (bibliotheek onbereikbaar, conflictkopie) en die zou dán
        # áchter het opstartscherm vallen — een app die vastgelopen lijkt. Vandaar uit.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)

    @classmethod
    def _achtergrond(cls):
        pm = QPixmap(cls.BREEDTE, cls.HOOGTE)
        pm.fill(QColor(24, 40, 66))
        p = QPainter(pm)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont(p.font().family(), 22, QFont.Bold))
        p.drawText(QRect(0, 40, cls.BREEDTE, 44), Qt.AlignCenter, "Schaats Analyse")
        p.setPen(QColor(150, 180, 220))
        p.setFont(QFont(p.font().family(), 9))
        p.drawText(QRect(0, 84, cls.BREEDTE, 22), Qt.AlignCenter, "bezig met opstarten...")
        p.end()
        return pm

    def melding(self, tekst):
        """Zet de statusregel en tékent hem ook meteen: tussen twee meldingen door draait
        er geen event-loop (we zitten nog in de opstartcode), dus zonder processEvents
        blijft het scherm op de eerste tekst staan."""
        self.showMessage(f"  {tekst}", Qt.AlignBottom | Qt.AlignLeft,
                         QColor(220, 232, 248))
        QApplication.processEvents()


def _start_opstartscherm():
    """Maakt de QApplication en zet het opstartscherm neer. Retourneert (app, scherm).

    Wordt op moduleniveau aangeroepen — vóór de zware imports hieronder — en alleen als
    dit bestand als programma draait; bij `import schaats_gui` (zelftests, meetscripts)
    gebeurt er niets.
    """
    app = QApplication(sys.argv)
    scherm = Opstartscherm()
    scherm.show()
    scherm.melding("Onderdelen laden...")
    return app, scherm


_APP, _SPLASH = _start_opstartscherm() if __name__ == "__main__" else (None, None)

import cv2
import numpy as np

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

import schaats_db
import schaats_perspectief
from schaats_analyse import (
    segmenteer_afzetten, teken_overlay_op_frame, horizon_hoek_uit_lijn,
    detecteer_ijslijn, PerspectiefConfig, verwerk_afgeleiden, Landmark,
    torso_centroid, kader_reeks, maak_voorvulling, bepaal_bocht_reeks,
    FrameResultaat, video_info, knip_fragmenten, KnipAfgebroken,
    ONV_AFGEKAPT, ONV_GEEN_PUSH, app_dir, is_bevroren,
)

# Skelet-editor (fase 3)
# De handle-/grijpradius schaalt mee met de schaatser: een vaste fractie van de torso-lengte
# op het scherm, geklemd op [GRIJP_MIN_PX, GRIJP_MAX_PX]. Zo lijken de bolletjes bij elke
# schaatsergrootte én zoomstand even groot en overlappen ze niet meer als de schaatser klein
# in beeld staat. Tuning-knoppen:
GRIJP_FRAC     = 0.06   # handle-radius als fractie van de torso-lengte (torso ~200 px → ~12 px)
GRIJP_MIN_PX   = 4      # ondergrens in schermpixels (kleine schaatser houdt een aanklikbare handle)
GRIJP_MAX_PX   = 14     # bovengrens in schermpixels (close-up geen enorme bollen)
HANDLE_MIN_VIS = 0.2    # onder deze zichtbaarheid geen sleepbare handle (zoals teken_alle_landmarks)

# MediaPipe-33 landmark-index → naam van het lichaamsdeel (voor de hover-tekst in de editor).
# "links"/"rechts" is anatomisch (de eigen linker-/rechterkant van de schaatser), net als in
# de detectie/ L-R-fixer. Indices die niet voorkomen krijgen een generieke terugval.
LANDMARK_NAMEN = {
    0: "neus",
    1: "linkeroog (binnen)", 2: "linkeroog", 3: "linkeroog (buiten)",
    4: "rechteroog (binnen)", 5: "rechteroog", 6: "rechteroog (buiten)",
    7: "linkeroor", 8: "rechteroor", 9: "mond links", 10: "mond rechts",
    11: "linkerschouder", 12: "rechterschouder",
    13: "linkerelleboog", 14: "rechterelleboog",
    15: "linkerpols", 16: "rechterpols",
    17: "linkerpink", 18: "rechterpink",
    19: "linkerwijsvinger", 20: "rechterwijsvinger",
    21: "linkerduim", 22: "rechterduim",
    23: "linkerheup", 24: "rechterheup",
    25: "linkerknie", 26: "rechterknie",
    27: "linkerenkel", 28: "rechterenkel",
    29: "linkerhiel", 30: "rechterhiel",
    31: "linkerteen", 32: "rechterteen",
}

# Handmatig skelet plaatsen op een frame zonder pose. De volgorde loopt van boven naar
# beneden en per paar links-eerst, zodat de gebruiker een vast ritme krijgt. Meer punten
# vragen heeft geen zin: dit is precies wat de metingen gebruiken (heup/knie/enkel voor de
# afzet-, knie- en strekhoek) plus de schouders voor de torso, waar de grijpradius en de
# automatische zoom op rekenen. Hiel en teen (29-32) blijven op visibility 0 staan — net als
# bij de YOLO-backend zonder RTMPose, die ze ook niet kent.
PLAATS_VOLGORDE  = (11, 12, 23, 24, 25, 26, 27, 28)
# Zonder deze zes is er geen meting mogelijk; het skelet mag pas vastgelegd worden als ze
# allemaal een zichtbare positie hebben (aangeklikt óf overgenomen uit de voorvulling).
PLAATS_VERPLICHT = (23, 24, 25, 26, 27, 28)

# Inzoomen op de schaatser in de weergave
ZOOM_MAX  = 5.0        # maximale zoomfactor van de weergave-uitsnede (handmatig: slider/wiel)
ZOOM_STAP = 1.25       # muiswiel-factor per notch
# De automatische zoom mag verder inzoomen dan de handmatige grens: een schaatser die aan
# het begin van de clip ver weg is, moet echt vergroot worden om beeldvullend te zijn.
# Op 4K is 1/8 uitsnede nog 480×270 px; daaronder wordt het te zacht.
ZOOM_AUTO_MAX = 8.0    # bovengrens van de automatische zoom
# ...maar nooit verder dan waar er nog pixels zijn. Wat telt is niet de videoresolutie maar
# hoeveel de uitsnede op het scherm wordt opgeblazen: staande telefoonbeelden staan in een
# liggend paneel al fors gekrompen (veel ruimte om in te zoomen), 4K-liggend nauwelijks.
# De gebruiker kiest de zoom hier niet zelf, dus hoort het programma geen pap af te leveren.
KADER_MAX_VERGROTING = 2.5   # max. schermpixels per videopixel bij automatische zoom
# Lucht rondom de schaatser, als fractie van de ruimte die hij zelf nodig heeft. De
# landmarks houden op bij de neus en de tenen, terwijl de bovenkant van het hoofd en de
# ijzers er nog buiten steken — en een schaatser strak tegen de rand kijkt niet prettig.
KADER_MARGE = 0.15

# Bochtdetectie: uitleg bij de checkbox in beide analyse-dialogen (één tekst, twee plekken).
BOCHT_TOOLTIP = (
    "Herkent aan de stand van de heupen wanneer de schaatser niet frontaal in beeld is\n"
    "(in de bocht staan ze achter elkaar i.p.v. naast elkaar).\n"
    "\n"
    "Die frames worden dan grotendeels niet meer door de detector gehaald — dat scheelt\n"
    "flink in analysetijd — en ze leveren geen afzetmeting op. Elke ~0,3 s wordt gekeken\n"
    "of het rechte stuk alweer begonnen is, dus een clip die ín de bocht begint pakt de\n"
    "meting vanzelf op zodra de schaatser recht op de camera af komt.\n"
    "\n"
    "Uitzetten alleen om te zien wat er in de bocht gebeurt; die hoeken zijn niet bruikbaar.")

# Perspectiefcorrectie (fase 7). Nog experimenteel: de wiskunde en de koppeling staan er
# en de kalibratie wordt nu bewaard, maar de correctie is nog niet op echt materiaal
# gevalideerd (ROADMAP fase 7, stap 3). Vandaar "experimenteel" en niet "werkt niet" —
# aan staat hij alleen als je er bewust mee meet.
PERSPECTIEF_TOOLTIP = (
    "Voor een vaste, schuin geplaatste camera. Trek vóór de analyse de baanlijnen na;\n"
    "daaruit wordt de camerastand gekalibreerd en wordt de afzethoek per frame\n"
    "teruggerekend naar het echte ijsvlak i.p.v. het vertekende beeldvlak.\n"
    "Bijvangst: snelheid en slaglengte in de tabel.\n"
    "\n"
    "Nodig: minstens 2 baanlijnen + 1 dwarslijn (2+1 alleen met opgegeven\n"
    "brandpuntsafstand; anders 3+1 of 2+2), en een camera die niet beweegt.\n"
    "\n"
    "De kalibratie wordt bij de analyse bewaard, dus heropenen herstelt de correctie\n"
    "en een volgende clip uit dezelfde camerastand kan hem overnemen.\n"
    "\n"
    "EXPERIMENTEEL: nog niet op echt materiaal gevalideerd. Wordt er frontaal met een\n"
    "horizontale camera gefilmd, dan is de vertekening klein en heb je dit niet nodig.")

PERSPECTIEF_TOOLTIP_BATCH = (
    PERSPECTIEF_TOOLTIP + "\n"
    "\n"
    "In een batch wordt de kalibratie ÉÉN keer gevraagd en op alle clips toegepast —\n"
    "ze komen immers uit dezelfde camerastand. Dat is ook de voorwaarde om hun hoeken\n"
    "onderling te mogen vergelijken.")


def _kalibratie_rijen(inst):
    """Info-rijen over de bewaarde perspectiefkalibratie (leeg als er geen is).

    Toont de invoer (aantal lijnen, lijnafstand, methode, onderbeenlengte) en de
    hérberekende uitkomst (f, camerahoogte, residu). Die laatste komt niet uit de
    opslag maar wordt hier opnieuw uitgerekend — precies zoals bij het openen van de
    analyse — zodat je in de Info ziet wat de analyse nú zou gebruiken."""
    p = inst.get("perspectief")
    if not p or not p.get("invoer"):
        return []
    inv = p["invoer"]
    n_rij, n_dwars = len(inv.get("rijlijnen") or []), len(inv.get("dwarslijnen") or [])
    methode = {"onderbeen": "onderbeenlengte (bol-snijding)",
               "beenvlak": "beenvlak (rijrichting)"}.get(p.get("methode"), p.get("methode"))
    rijen = [
        ("Kalibratie:", f"{n_rij} baanlijnen + {n_dwars} dwarslijnen, "
                        f"{inv.get('lijnafstand', '?')} m uit elkaar, "
                        f"op beeld {inv.get('beeld_w')}×{inv.get('beeld_h')}",
         "De nagetrokken lijnen worden bewaard; de camerastand wordt eruit herberekend."),
        ("Reconstructie:", methode
         + (f", onderbeen {p['onderbeen_l'] * 100:.1f} cm" if p.get("onderbeen_l") else ""),
         None),
    ]
    if inv.get("notitie"):
        rijen.append(("Kalibratie-notitie:", inv["notitie"], None))
    try:
        kal = schaats_perspectief.KalibratieInvoer.uit_dict(inv).kalibreer()
        rijen.append(("Camerastand:",
                      f"f = {kal.f:.0f} px{' (geschat)' if kal.f_geschat else ''}, "
                      f"hoogte {kal.camera_hoogte:.1f} m, horizon {kal.horizon_deg:+.2f}°, "
                      f"residu {kal.residu_px:.1f} px", None))
    except Exception as e:
        rijen.append(("Camerastand:", f"niet herberekenbaar: {e}", None))
    return rijen

# Kolommen van de opnametabel (fase 8). Als getal genoemd omdat er cel-widgets en een
# itemChanged-filter op hangen: een kolom erbij mag geen stille verschuiving worden.
OPNAME_KOL_NAAM, OPNAME_KOL_DUUR, OPNAME_KOL_LOKAAL = 0, 1, 2
OPNAME_KOL_STATUS, OPNAME_KOL_TELLING, OPNAME_KOL_NOTITIE = 3, 4, 5

# Weergave van `schaats_db.bestand_lokaal`: (tekst, kleur, uitleg). Zonder deze kolom is er
# geen enkel signaal dat een opname nog in de cloud staat — het bestand ís er immers, hij
# komt alleen tergend traag binnen. Zie `_opname_beschikbaar` voor het waarom van de cijfers.
LOKAAL_WEERGAVE = {
    "lokaal": ("✓ ja", QColor(60, 140, 60),
               "Deze opname staat op deze pc: doorbladeren en knippen gaan op volle "
               "snelheid."),
    "deels":  ("⏳ deels", QColor(190, 130, 0),
               "Een deel staat lokaal, de rest nog niet — waarschijnlijk is de cloudmap nog "
               "aan het downloaden. Wacht daarop, anders blijft doorbladeren traag."),
    "cloud":  ("☁ nee — nog in de cloud", QColor(190, 60, 60),
               "Deze opname staat niet offline op deze pc; elk stuk beeld moet eerst "
               "gedownload worden.\n"
               "Gemeten: 5 tot 20 seconden per sprong in het knipvenster, tegen "
               "0,1 seconde als hij lokaal staat.\n\n"
               "Oplossing: rechtsklik de map 'opnames' in Verkenner → Google Drive "
               "→ 'Offline beschikbaar maken'."),
}

# Knipvenster (fase 8): boven deze sprong seekt een `snel_zoeken`-speler i.p.v. sequentieel
# door te spoelen. Kleine sprongen sequentieel laten lopen houdt frame-voor-frame-stappen en
# gewoon afspelen exact — en juist rond een fragmentgrens tik je frame voor frame.
SEEK_DREMPEL_FRAMES = 30

# Afspeelsnelheden: (label, factor op de fps). 1.0 = echte snelheid, lager = slow motion.
# Boven 1× is bedoeld om door een lange opname te scannen (het knipvenster van fase 8): daar
# wordt niet sneller gedecodeerd maar worden frames **overgeslagen** (zie _speel_tick), want
# 8× echte snelheid haalt geen enkele decoder.
SNELHEDEN = [("8×", 8.0), ("4×", 4.0), ("2×", 2.0),
             ("1×", 1.0), ("½×", 0.5), ("¼×", 0.25), ("⅛×", 0.125), ("1/16×", 0.0625)]


def _snelheid_idx(factor):
    """Index van een snelheid in SNELHEDEN, opgezocht op factor i.p.v. hard genummerd —
    anders verschuift elke toegevoegde snelheid stilzwijgend de defaults."""
    return next(i for i, (_, f) in enumerate(SNELHEDEN) if f == factor)


SNELHEID_DEFAULT_IDX = _snelheid_idx(1.0)

# Doorspoelen met . en , in het handmatige kijkvenster. 6× de opnamesnelheid: snel genoeg
# om een half uur door te komen, langzaam genoeg om te zien wanneer je erlangs schiet. De
# tik is een bovengrens op de vloeiendheid — het doelframe volgt uit de wandklok, dus het
# blijft 6× ook als de decoder het niet bijhoudt (zie BekijkVenster._spoel_tick).
SPOEL_FACTOR = 6.0
SPOEL_TICK_MS = 40

# Breedte van de transportknoppen (⏮ ⏪ ▶ ⏩ ⏭): ze dragen één teken, dus de
# Qt-standaardbreedte voor tekstknoppen is verspilde ruimte op een smal scherm.
TRANSPORT_KNOP_BREEDTE = 46

# Vergelijkpagina: interval van de masterklok die beide video's tegelijk aanstuurt. Dit is
# alleen een bovengrens op de vloeiendheid — het doelframe volgt uit de wandkloktijd, dus
# de klok corrigeert zichzelf en er ontstaat geen drift.
ALLES_TICK_MS = 30
# Twee video's tegelijk decoderen haalt 1× toch niet, en een trainer kijkt naar techniek:
# standaard ¼×.
ALLES_SNELHEID_IDX = _snelheid_idx(0.25)

# ── Backend-selectie ────────────────────────────────────────────────────────────
# Gebruik YOLO-pose + ByteTrack als torch/ultralytics beschikbaar is (draai de app dan
# onder de .venv-yolo), val anders terug op de MediaPipe-backend.
#
# `import schaats_yolo` trekt torch + ultralytics binnen: ~2,8 s op een warme machine en
# een veelvoud daarvan koud (Windows scant die honderden MB's aan DLL's). Dat is
# tweederde van de opstarttijd, terwijl de startpagina alleen de bibliotheek toont — de
# backend is pas nodig als er écht een analyse begint. Daarom hier alleen de goedkope
# vraag "staat het pakket geïnstalleerd?" (find_spec: ~1 ms, importeert niets) en de
# echte import lui, via `analyseer_backend()`. Meteen na het tonen van het venster wordt
# hij op de achtergrond alvast warmgedraaid (`_warm_backend_op`), zodat de eerste analyse
# er niets van merkt.
def _backend_beschikbaar():
    """(yolo?, rtmpose?) puur op basis van geïnstalleerde pakketten, zonder ze te laden."""
    from importlib.util import find_spec
    try:
        yolo = find_spec("ultralytics") is not None and find_spec("torch") is not None
        return yolo, yolo and find_spec("rtmlib") is not None
    except (ImportError, ValueError):     # kapotte installatie: dan MediaPipe
        return False, False


_HEEFT_YOLO, _HEEFT_RTMPOSE = _backend_beschikbaar()
IS_YOLO = _HEEFT_YOLO
# Voorspelling van schaats_yolo.BACKEND_NAAM (die module is nog niet geladen). Zodra hij
# er wél is, wordt deze naam vervangen door de zijne — een verschil corrigeert zichzelf
# dus, en de naam die in een analyse wordt opgeslagen komt altijd van de backend zelf.
BACKEND_NAAM = (("YOLO-pose + ByteTrack + RTMPose-verfijning" if _HEEFT_RTMPOSE
                 else "YOLO-pose + ByteTrack") if IS_YOLO else "MediaPipe")

_backend_slot = threading.Lock()
_backend_fn = None
BACKEND_FOUT = ""      # gevuld als de YOLO-import geïnstalleerd leek maar toch mislukte


def _backend_stuk(*_args, **_kwargs):
    """Analyse-ingang als de enige meegeleverde backend niet laadde (alleen bevroren).

    MediaPipe zit niet in het gebundelde pakket, dus daar terugvallen zou pas midden in
    de eerste analyse als ImportError opduiken. Liever hier meteen één duidelijke fout;
    `_waarschuw_backend_terugval()` heeft de gebruiker dan al gewaarschuwd."""
    raise RuntimeError("De analyse-backend kon niet geladen worden:\n\n"
                       f"{BACKEND_FOUT}")


def _laad_backend():
    """Importeert de gekozen backend (eenmalig) en retourneert zijn `analyseer`-functie.

    Faalt de YOLO-import alsnog — het pakket stond er wel maar is stuk, bv. een torch met
    ontbrekende DLL's — dan valt de app hier terug op MediaPipe in plaats van de analyse
    te laten stuklopen, en worden `IS_YOLO`/`BACKEND_NAAM` bijgetrokken. Dat mag níet
    stilzwijgend gebeuren (het is een andere detector en dus een andere meting), dus de
    reden wordt bewaard in `BACKEND_FOUT` en door de GUI gemeld zodra er een analyse start.

    In een gebundelde .exe bestaat die terugval niet: MediaPipe zit niet in het pakket
    (`IS_YOLO` is daar altijd waar). Dan blijft de backendnaam staan en levert dit
    `_backend_stuk` op, zodat er één duidelijke fout komt i.p.v. een ImportError diep in
    de eerste analyse.
    """
    global _backend_fn, IS_YOLO, BACKEND_NAAM, BACKEND_FOUT
    with _backend_slot:
        if _backend_fn is None:
            if IS_YOLO:
                try:
                    import schaats_yolo
                    BACKEND_NAAM = schaats_yolo.BACKEND_NAAM
                    _backend_fn = schaats_yolo.analyseer
                except Exception as e:
                    BACKEND_FOUT = f"{type(e).__name__}: {e}"
                    if not is_bevroren():
                        IS_YOLO = False
                        BACKEND_NAAM = "MediaPipe"
                    print(f"YOLO-backend kon niet geladen worden ({BACKEND_FOUT}); "
                          + ("er kan nu niet geanalyseerd worden." if is_bevroren()
                             else "de app werkt verder met MediaPipe."),
                          file=sys.stderr)
            if _backend_fn is None:
                if is_bevroren():
                    _backend_fn = _backend_stuk      # MediaPipe zit niet in dit pakket
                else:
                    from schaats_analyse import analyseer as mp_analyseer
                    _backend_fn = mp_analyseer
        return _backend_fn


def analyseer_backend(*args, **kwargs):
    """De analyse-ingang van de GUI; laadt de backend bij het eerste gebruik."""
    return _laad_backend()(*args, **kwargs)


def _warm_backend_op():
    """Laadt de backend alvast op de achtergrond, direct nadat het venster in beeld staat.

    Een daemon-thread, want het is puur vooruitwerken: gaat de gebruiker meteen een
    analyse starten, dan blokkeert diens import gewoon op dezelfde lock tot deze klaar is.
    """
    if IS_YOLO:
        threading.Thread(target=_laad_backend, name="backend-warmup", daemon=True).start()

_MODEL_DIR = app_dir()          # naast de scripts, of naast de exe
STANDAARD_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
HEAVY_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_heavy.task")


# Ruimte die de vensterrand (titelbalk + kaders) buiten de inhoud inneemt. `resize()` stelt
# de inhoudsmaat in, dus zonder deze marge steekt een venster op schermhoogte onderlangs weg
# achter de taakbalk. Ruim genomen; het gaat om een ondergrens, niet om precisie.
VENSTER_RAND = QMargins(8, 40, 8, 8)


def zet_venstergrootte(venster, gewenste_breedte, gewenste_hoogte, maximaliseer=False):
    """Past de venstergrootte aan het beschikbare scherm aan en centreert het venster.

    Een vaste pixelmaat (1400x820 voor het hoofdvenster) valt op een kleiner laptopscherm
    buiten beeld. Past de gewenste maat niet, dan gaat het hoofdvenster **gemaximaliseerd**
    open (`maximaliseer=True`): dat vult de hoogte precies en scheelt de gebruiker het
    handmatig goedzetten bij elke start. Dialogen worden alleen geklemd en gecentreerd.

    Let op: `resize()` kan de layout niet overrulen — is de `minimumSizeHint` van de inhoud
    breder dan het scherm, dan wordt het venster alsnog te groot. Daarom breken de brede
    bedieningsbalken in `VideoSpeler` af met een `WrapBalk`; zie daar.
    """
    scherm = venster.screen() or QApplication.primaryScreen()
    if scherm is None:
        venster.resize(gewenste_breedte, gewenste_hoogte)
        return
    beschikbaar = scherm.availableGeometry()
    if maximaliseer and (gewenste_breedte > beschikbaar.width()
                         or gewenste_hoogte > beschikbaar.height()):
        # setWindowState i.p.v. showMaximized(): het venster mag hier nog niet in beeld
        # springen — de caller bepaalt wanneer er getoond wordt. Echt maximaliseren i.p.v.
        # naar de schermmaat resizen, want `resize()` zet de *inhoud*: de titelbalk komt daar
        # nog bovenop en zou de statusbalk onder de taakbalk schuiven.
        venster.resize(beschikbaar.size().shrunkBy(VENSTER_RAND))
        venster.setWindowState(venster.windowState() | Qt.WindowMaximized)
        return
    # Ruimte laten voor de vensterrand: `resize()` gaat over de inhoud, de titelbalk zit
    # daarbuiten — zonder marge valt de onderrand achter de taakbalk.
    breedte = min(gewenste_breedte, beschikbaar.width() - VENSTER_RAND.left()
                  - VENSTER_RAND.right())
    hoogte = min(gewenste_hoogte, beschikbaar.height() - VENSTER_RAND.top()
                 - VENSTER_RAND.bottom())
    venster.resize(breedte, hoogte)
    x = beschikbaar.x() + (beschikbaar.width() - breedte) // 2
    y = beschikbaar.y() + (beschikbaar.height() - hoogte) // 2
    venster.move(x, y)


class FlowLayout(QLayout):
    """Layout die zijn items op een regel zet en **afbreekt** als de breedte niet meelukt.

    Nodig omdat de bedieningsbalken van `VideoSpeler` (laag-toggles + zoomregelaars, samen
    ~774 px) als `QHBoxLayout` een minimumbreedte van 774 px eisen. Twee spelers naast
    elkaar op de vergelijkpagina maakten daar 1607 px van — breder dan een 1280 px
    laptopscherm, en een `QMainWindow` kan niet kleiner dan zijn `minimumSizeHint`, dus
    `resize()` werd domweg genegeerd. Afbrekend is de minimumbreedte die van het bréédste
    losse item (~138 px) en past het venster op elk scherm; op een breed scherm blijft het
    één regel en ziet het er precies zo uit als voorheen.
    """

    def __init__(self, parent=None, marge=0, tussenruimte=6, min_breedte=0):
        super().__init__(parent)
        self._items = []
        self._tussenruimte = tussenruimte
        self._min_breedte = min_breedte
        self.setContentsMargins(marge, marge, marge, marge)

    # ── QLayout-plichten ─────────────────────────────────────────────────
    def addItem(self, item):
        self._items.append(item)

    def addStretch(self, _factor=1):
        """No-op: een afbrekende balk lijnt links uit, een rekstuk heeft geen betekenis.
        Bestaat zodat aanroepers die van een QHBoxLayout komen niet hoeven te veranderen."""

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    # ── Hoogte volgt uit de breedte ──────────────────────────────────────
    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, breedte):
        return self._leg_uit(QRect(0, 0, breedte, 0), alleen_meten=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._leg_uit(rect, alleen_meten=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        # De breedte van het breedste item — daaronder past geen enkele regel meer.
        maat = QSize()
        for item in self._items:
            maat = maat.expandedTo(item.minimumSize())
        marges = self.contentsMargins()
        maat = maat + QSize(marges.left() + marges.right(), marges.top() + marges.bottom())
        # Deze breedte is niet alleen een ondergrens: Qt vraagt de minimumhóogte van een
        # hoogte-volgt-breedte-item op door `heightForWidth()` hier op te roepen. Met de
        # breedte van één item breekt de balk in elf regels af en groeit het venster-minimum
        # met ~280 px in de hoogte. Vandaar een realistische ondergrens (zie WrapBalk).
        return maat.expandedTo(QSize(self._min_breedte, 0))

    def _leg_uit(self, rect, alleen_meten):
        """Plaatst de items regel voor regel; retourneert de benodigde totale hoogte.

        Twee doorgangen: eerst de regelindeling (en dus de hoogte van elke regel), daarna het
        plaatsen. Dat is nodig om **verticaal te centreren** — een label van 16 px hoort niet
        bovenaan een regel met vinkjes van 24 px te bungelen.
        """
        marges = self.contentsMargins()
        vak = rect.adjusted(marges.left(), marges.top(), -marges.right(), -marges.bottom())

        regels = []                       # [(items, regelhoogte)]
        huidig, breedte, regelhoogte = [], 0, 0
        for item in self._items:
            maat = item.sizeHint()
            erbij = maat.width() if not huidig else self._tussenruimte + maat.width()
            if huidig and breedte + erbij > vak.width():
                regels.append((huidig, regelhoogte))
                huidig, breedte, regelhoogte = [], 0, 0
                erbij = maat.width()
            huidig.append(item)
            breedte += erbij
            regelhoogte = max(regelhoogte, maat.height())
        if huidig:
            regels.append((huidig, regelhoogte))

        y = vak.y()
        for items, hoogte in regels:
            if not alleen_meten:
                x = vak.x()
                for item in items:
                    maat = item.sizeHint()
                    item.setGeometry(QRect(QPoint(x, y + (hoogte - maat.height()) // 2), maat))
                    x += maat.width() + self._tussenruimte
            y += hoogte + self._tussenruimte
        totaal = (y - self._tussenruimte - vak.y()) if regels else 0
        return totaal + marges.top() + marges.bottom()


class WrapBalk(QWidget):
    """Draagwidget voor een `FlowLayout`, zodat een QVBoxLayout de afbrekende balk als
    gewoon item kan opnemen (en de hoogte-uit-breedte netjes doorgeeft)."""

    # Ondergrens voor de breedte van de balk. Niet cosmetisch: een QVBoxLayout leidt de
    # minimumhoogte van een hoogte-volgt-breedte-item af door `heightForWidth()` op te vragen
    # bij de *minimale* breedte. Zonder ondergrens is dat de breedte van één item (138 px),
    # waar de balk in acht regels afbreekt — 250 px hoogte die permanent in het
    # venster-minimum gaat zitten en het venster boven de schermhoogte tilt. Bij 280 px zijn
    # het drie regels (~90 px), en 280 blijft onder de 320/400 px die het videobeeld zelf al
    # eist, dus deze grens kost geen enkele extra breedte.
    MIN_BREEDTE = 280

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, min_breedte=self.MIN_BREEDTE)
        beleid = self.sizePolicy()
        beleid.setHeightForWidth(True)
        beleid.setVerticalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(beleid)
        self.setMinimumWidth(self.MIN_BREEDTE)

    def addWidget(self, w):
        self.flow.addWidget(w)

    def addStretch(self, factor=1):
        self.flow.addStretch(factor)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, breedte):
        return self.flow.heightForWidth(breedte)


class DoelKiezer(QDialog):
    """
    Toont het eerste frame en laat de gebruiker op de te volgen schaatser klikken.
    Retourneert een genormaliseerd (x, y)-punt in `doel_punt`, of None ('volg grootste').
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kies de schaatser om te volgen")
        self.doel_punt = None
        self._frame = frame_bgr
        self._scaled_size = None

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Klik op de schaatser die je wilt volgen "
                           "(of gebruik de knop hieronder)."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._klik
        v.addWidget(self.label, 1)

        knoppen = QHBoxLayout()
        knoppen.addStretch(1)
        btn_skip = QPushButton("Volg grootste schaatser")
        btn_skip.clicked.connect(self.accept)     # doel_punt blijft None
        knoppen.addWidget(btn_skip)
        v.addLayout(knoppen)

        zet_venstergrootte(self, 900, 640)

        h, w = frame_bgr.shape[:2]
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _klik(self, event):
        if self._scaled_size is None:
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        x = (event.position().x() - offx) / sw
        y = (event.position().y() - offy) / sh
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.doel_punt = (float(x), float(y))
            self.accept()


class HorizonKiezer(QDialog):
    """
    Toont het eerste frame en laat de gebruiker twee punten langs de ijslijn (of een
    andere horizontale referentie: boarding, reclameband, baanlijn) klikken. Daaruit
    volgt de kanteling van de camera t.o.v. de horizon. Retourneert `horizon_deg`
    (float, graden) — 0.0 als er geen kanteling wordt ingesteld.
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stel de horizon / ijslijn in")
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self._frame = frame_bgr
        self._punten = []            # originele-pixel (x, y) van de referentielijn
        self._scaled_size = None
        self._scale = 1.0

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Klik twee punten langs het ijs (of de boarding/reclameband) om de\n"
            "camerakanteling te bepalen, of laat hem automatisch detecteren.\n"
            "Klik opnieuw om de lijn te hertekenen."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._klik
        v.addWidget(self.label, 1)

        self.lbl_hoek = QLabel("Kanteling: 0.00°  (nog geen lijn getekend)")
        self.lbl_hoek.setStyleSheet("font-weight: bold;")
        v.addWidget(self.lbl_hoek)

        self.chk_per_frame = QCheckBox(
            "Automatisch per frame herkennen (voor een schommelende camera) (werkt niet)")
        self.chk_per_frame.setToolTip(
            "WERKT NIET / niet in gebruik: sinds juli 2026 staat de camera altijd\n"
            "precies horizontaal, dus er is geen kanteling om per frame te volgen.\n"
            "Laat deze optie uit.")
        self.chk_per_frame.stateChanged.connect(self._wissel_per_frame)
        v.addWidget(self.chk_per_frame)

        knoppen = QHBoxLayout()
        self.btn_auto = QPushButton("Detecteer (dit frame)")
        self.btn_auto.clicked.connect(self._detecteer)
        knoppen.addWidget(self.btn_auto)
        btn_geen = QPushButton("Geen kanteling (0°)")
        btn_geen.clicked.connect(self._geen_kanteling)
        knoppen.addWidget(btn_geen)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Bevestig")
        self.btn_ok.clicked.connect(self._bevestig)
        knoppen.addWidget(self.btn_ok)
        v.addLayout(knoppen)

        zet_venstergrootte(self, 900, 680)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        if self._punten:
            painter = QPainter(scaled)
            pen = QPen(QColor(60, 200, 255), 3)
            painter.setPen(pen)
            pts = [(int(x * self._scale), int(y * self._scale)) for x, y in self._punten]
            for px, py in pts:
                painter.drawEllipse(px - 4, py - 4, 8, 8)
            if len(pts) == 2:
                painter.drawLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _klik(self, event):
        if self._scaled_size is None or self.chk_per_frame.isChecked():
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if len(self._punten) >= 2:           # derde klik → nieuwe lijn beginnen
            self._punten = []
        self._punten.append((ox, oy))
        if len(self._punten) == 2:
            self.horizon_deg = horizon_hoek_uit_lijn(self._punten[0], self._punten[1])
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: klik het tweede punt …")
        self._render()

    def _detecteer(self):
        graden = detecteer_ijslijn(self._frame)
        if graden is None:
            QMessageBox.information(
                self, "Geen ijslijn gevonden",
                "Kon geen betrouwbare horizontale lijn detecteren. Teken de lijn "
                "handmatig, of kies 'Geen kanteling'.")
            return
        # Synthetiseer een weergavelijn dwars door het beeld op de gevonden hoek.
        w, h = self._orig_w, self._orig_h
        cx, cy = w / 2.0, h / 2.0
        helling = np.tan(np.radians(graden))          # y daalt naar rechts bij positieve hoek
        self._punten = [(0.0, cy + helling * cx), (float(w), cy - helling * (w - cx))]
        self.horizon_deg = graden
        self.lbl_hoek.setText(f"Kanteling: {graden:+.2f}°  (automatisch — controleer de lijn)")
        self._render()

    def _wissel_per_frame(self, _state):
        """Bij per-frame auto is de handmatige/constante lijn niet van toepassing."""
        aan = self.chk_per_frame.isChecked()
        self.label.setEnabled(not aan)
        self.btn_auto.setEnabled(not aan)
        if aan:
            self.lbl_hoek.setText("Kanteling: automatisch per frame — "
                                  "wordt tijdens de analyse bepaald.")
        elif len(self._punten) == 2:
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: 0.00°  (nog geen lijn getekend)")

    def _bevestig(self):
        self.auto_per_frame = self.chk_per_frame.isChecked()
        self.accept()

    def _geen_kanteling(self):
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self.accept()


class KalibratieKiezer(QDialog):
    """
    Perspectiefkalibratie via baanlijnen (fase 7, vaste camera). De gebruiker trekt
    op het eerste frame lijnen na (elke lijn = twee klikken): baanlijnen die in
    werkelijkheid evenwijdig in de rijrichting lopen, en dwarslijnen die er haaks op
    staan. De dialoog kalibreert live mee en tekent de gevonden ware horizon; de
    Bevestig-knop kan pas als de kalibratie slaagt. Resultaat in `self.perspectief`
    (PerspectiefConfig).

    Minimaal nodig: 2 baanlijnen + 2 dwarslijnen, óf 3 baanlijnen + 1 dwarslijn, óf
    2 baanlijnen + 1 dwarslijn + een opgegeven brandpuntsafstand (frontale camera's
    kúnnen alleen met opgegeven brandpuntsafstand).
    """
    KLEUR_RIJ = QColor(60, 200, 255)     # cyaan
    KLEUR_DWARS = QColor(255, 170, 40)   # oranje
    KLEUR_HORIZON = QColor(240, 240, 240)

    def __init__(self, frame_bgr, parent=None, invoer=None, config=None):
        """`invoer`/`config`: een eerder gemaakte kalibratie om mee te beginnen (zelfde
        camerastand hergebruiken). De lijnen staan dan al getekend en zijn nog te
        corrigeren — hergebruiken en aanpassen is één en dezelfde handeling."""
        super().__init__(parent)
        self.setWindowTitle("Perspectiefkalibratie: trek de baanlijnen na")
        self.perspectief = None
        self._frame = frame_bgr
        self._rijlijnen = []             # [((x,y),(x,y))] in originele pixels
        self._dwarslijnen = []
        self._klik_punt = None           # eerste punt van een lijn-in-wording
        self._kalibratie = None
        self._invoer = None              # KalibratieInvoer van de huidige lijnen
        self._scaled_size = None
        self._scale = 1.0

        hoofd = QHBoxLayout(self)

        links = QVBoxLayout()
        links.addWidget(QLabel(
            "Trek elke lijn met twee klikken. Baanlijnen: evenwijdig in de rijrichting "
            "(volgorde maakt niet uit).\nDwarslijnen: haaks erop (start-/finishlijn, "
            "bochtmarkering). Trek zo lang mogelijke lijnen — dat is nauwkeuriger."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 400)
        self.label.mousePressEvent = self._klik
        links.addWidget(self.label, 1)
        hoofd.addLayout(links, 1)

        rechts = QVBoxLayout()

        soort_groep = QGroupBox("Lijnsoort (voor de volgende lijn)")
        sv = QVBoxLayout(soort_groep)
        self.radio_rij = QRadioButton("Baanlijn (rijrichting)")
        self.radio_dwars = QRadioButton("Dwarslijn (haaks op de baan)")
        self.radio_rij.setChecked(True)
        sv.addWidget(self.radio_rij)
        sv.addWidget(self.radio_dwars)
        rechts.addWidget(soort_groep)

        knoppen_lijn = QHBoxLayout()
        btn_wis_laatste = QPushButton("Laatste lijn wissen")
        btn_wis_laatste.clicked.connect(self._wis_laatste)
        btn_wis_alles = QPushButton("Alles wissen")
        btn_wis_alles.clicked.connect(self._wis_alles)
        knoppen_lijn.addWidget(btn_wis_laatste)
        knoppen_lijn.addWidget(btn_wis_alles)
        rechts.addLayout(knoppen_lijn)

        # Alleen-hoeken is de standaard: de afzethoek is schaalvrij, dus zonder bekende
        # lijnafstand is hij exact (nagemeten: 0,00° fout, of je nu 0,5 of 50 m invult).
        # Meters heb je alleen nodig voor snelheid/slaglengte — en de méthode 'onderbeen'
        # rekent met een onderbeenlengte in échte meters, dus die kan hier niet mee (een
        # verzonnen afstand gaf daar 49° fout, stilzwijgend). Vandaar de koppeling
        # hieronder: alleen-hoeken ⇒ 'beenvlak', dat geen enkele lengte gebruikt.
        self.chk_alleen_hoeken = QCheckBox("Alleen hoeken (geen snelheid/slaglengte)")
        self.chk_alleen_hoeken.setChecked(True)
        self.chk_alleen_hoeken.setToolTip(
            "De afzethoek is schaalvrij: hij komt alléén uit de richtingen van de lijnen,\n"
            "niet uit hun afstand. Je hoeft dus geen enkele maat te weten of op te meten.\n"
            "\n"
            "Uitzetten alleen als je snelheid (m/s) en slaglengte (m) in de tabel wilt, óf\n"
            "als je met de reconstructiemethode 'onderbeenlengte' wilt werken — die rekent\n"
            "met een lengte in echte meters en heeft dus een echte lijnafstand nodig.")
        self.chk_alleen_hoeken.toggled.connect(self._schaal_gewijzigd)
        rechts.addWidget(self.chk_alleen_hoeken)

        vorm = QFormLayout()
        self.spin_lijnafstand = QDoubleSpinBox()
        self.spin_lijnafstand.setRange(0.5, 30.0)
        self.spin_lijnafstand.setSingleStep(0.5)
        self.spin_lijnafstand.setValue(schaats_perspectief.STANDAARD_LIJNAFSTAND)
        self.spin_lijnafstand.setSuffix(" m")
        self.spin_lijnafstand.valueChanged.connect(self._herkalibreer)
        self.lbl_lijnafstand = QLabel("Afstand tussen baanlijnen:")
        vorm.addRow(self.lbl_lijnafstand, self.spin_lijnafstand)

        self.spin_f = QSpinBox()
        self.spin_f.setRange(0, 100000)
        self.spin_f.setValue(0)
        self.spin_f.setSpecialValueText("automatisch")
        self.spin_f.setToolTip(
            "Brandpuntsafstand in pixels. Normaal schat de kalibratie hem zelf uit de\n"
            "lijnen; bij een (bijna) frontale camera kan dat principieel niet en moet\n"
            "hij hier ingevuld worden (typisch 1–2× de beeldbreedte voor een telefoon).")
        self.spin_f.valueChanged.connect(self._herkalibreer)
        vorm.addRow("Brandpuntsafstand (px):", self.spin_f)

        self.combo_methode = QComboBox()
        self.combo_methode.addItem("Onderbeenlengte (bol-snijding)", "onderbeen")
        self.combo_methode.addItem("Beenvlak (rijrichting)", "beenvlak")
        self.combo_methode.setToolTip(
            "Hoe de knie-diepte wordt gereconstrueerd. Beide zijn experimenteel te\n"
            "vergelijken; 'onderbeenlengte' heeft de lengte hieronder nodig.")
        self.combo_methode.currentIndexChanged.connect(
            lambda _: self._schaal_gewijzigd(self.chk_alleen_hoeken.isChecked()))
        vorm.addRow("Reconstructie:", self.combo_methode)

        self.spin_lengte = QDoubleSpinBox()
        self.spin_lengte.setRange(1.0, 2.30)
        self.spin_lengte.setSingleStep(0.01)
        self.spin_lengte.setValue(1.80)
        self.spin_lengte.setSuffix(" m")
        vorm.addRow("Lichaamslengte schaatser:", self.spin_lengte)

        self.spin_onderbeen = QDoubleSpinBox()
        self.spin_onderbeen.setRange(0.0, 70.0)
        self.spin_onderbeen.setSingleStep(0.5)
        self.spin_onderbeen.setValue(0.0)
        self.spin_onderbeen.setSuffix(" cm")
        self.spin_onderbeen.setSpecialValueText("uit lichaamslengte")
        self.spin_onderbeen.setToolTip(
            "Opgemeten onderbeenlengte (knieholte tot enkelknobbel). Laat op\n"
            "'uit lichaamslengte' staan om hem te schatten als 0.246 × lichaamslengte.")
        self.lbl_onderbeen = QLabel("Onderbeenlengte:")
        vorm.addRow(self.lbl_onderbeen, self.spin_onderbeen)
        self.lbl_lengte = vorm.labelForField(self.spin_lengte)
        rechts.addLayout(vorm)

        self.lbl_status = QLabel("Nog geen lijnen getekend.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight: bold;")
        rechts.addWidget(self.lbl_status)
        rechts.addStretch(1)

        knoppen = QHBoxLayout()
        btn_annuleer = QPushButton("Annuleren")
        btn_annuleer.clicked.connect(self.reject)
        knoppen.addWidget(btn_annuleer)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Bevestig")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self._bevestig)
        knoppen.addWidget(self.btn_ok)
        rechts.addLayout(knoppen)

        paneel = QWidget()
        paneel.setLayout(rechts)
        paneel.setFixedWidth(340)
        hoofd.addWidget(paneel)

        zet_venstergrootte(self, 1150, 700)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)

        # Bewust helemaal aan het eind: `_schaal_gewijzigd` verzet de methode-combo, en
        # dat signaal loopt door naar `_herkalibreer`, die `lbl_status`, `btn_ok` én
        # `_orig_w` nodig heeft. Alles moet dus al bestaan.
        self._schaal_gewijzigd(self.chk_alleen_hoeken.isChecked())

        if config is not None and invoer is None:
            invoer = config.invoer
        if invoer is not None:
            self._vul_voor(invoer, config)
        self._render()

    def _vul_voor(self, invoer, config=None):
        """Zet een bestaande kalibratie in de dialoog. De beeldmaat moet kloppen: de
        lijnen staan in pixels, dus op een andersgrote video zouden ze er stilzwijgend
        naast liggen en een plausibele maar foute kalibratie opleveren."""
        if not invoer.past_bij(self._orig_w, self._orig_h):
            QMessageBox.warning(
                self, "Kalibratie past niet",
                f"Die kalibratie is gemaakt op beeld van {invoer.beeld_w}×{invoer.beeld_h} "
                f"en deze video is {self._orig_w}×{self._orig_h}. De lijnen staan in "
                f"pixels, dus overnemen zou ze verkeerd neerleggen. Trek ze opnieuw na.")
            return
        self._rijlijnen = list(invoer.rijlijnen)
        self._dwarslijnen = list(invoer.dwarslijnen)
        self.spin_lijnafstand.setValue(invoer.lijnafstand)
        self.spin_f.setValue(int(invoer.f_px or 0))
        # Eerst de schaalvlag, dan pas methode/lengte: `_schaal_gewijzigd` zet de methode
        # vast op 'beenvlak' zodra alleen-hoeken aan staat, en zou een daarvóór gezette
        # keuze weer overschrijven.
        self.chk_alleen_hoeken.setChecked(not invoer.schaal_bekend)
        if config is not None:
            idx = self.combo_methode.findData(config.methode)
            if idx >= 0:
                self.combo_methode.setCurrentIndex(idx)
            if config.onderbeen_l:
                self.spin_onderbeen.setValue(config.onderbeen_l * 100.0)
        self._herkalibreer()

    # ── tekenen ──────────────────────────────────────────────────────────
    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        painter = QPainter(scaled)
        s = self._scale
        for lijnen, kleur, prefix in ((self._rijlijnen, self.KLEUR_RIJ, "R"),
                                      (self._dwarslijnen, self.KLEUR_DWARS, "D")):
            painter.setPen(QPen(kleur, 3))
            for i, (p1, p2) in enumerate(lijnen, start=1):
                x1, y1 = p1[0] * s, p1[1] * s
                x2, y2 = p2[0] * s, p2[1] * s
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.drawText(int((x1 + x2) / 2) + 6, int((y1 + y2) / 2) - 6,
                                 f"{prefix}{i}")
        if self._klik_punt is not None:
            kleur = self.KLEUR_RIJ if self.radio_rij.isChecked() else self.KLEUR_DWARS
            painter.setPen(QPen(kleur, 3))
            px, py = self._klik_punt[0] * s, self._klik_punt[1] * s
            painter.drawEllipse(int(px) - 4, int(py) - 4, 8, 8)
        if self._kalibratie is not None:
            # ware horizon (verdwijnlijn van het ijsvlak) als visuele controle
            a, b, c = self._kalibratie.horizonlijn
            if abs(b) > 1e-9:
                y0 = -(c + a * 0.0) / b * s
                y1 = -(c + a * self._orig_w) / b * s
                painter.setPen(QPen(self.KLEUR_HORIZON, 1, Qt.DashLine))
                painter.drawLine(0, int(y0), int(self._orig_w * s), int(y1))
        painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    # ── interactie ───────────────────────────────────────────────────────
    def _klik(self, event):
        if self._scaled_size is None:
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if self._klik_punt is None:
            self._klik_punt = (ox, oy)
        else:
            lijn = (self._klik_punt, (ox, oy))
            self._klik_punt = None
            if np.hypot(lijn[1][0] - lijn[0][0], lijn[1][1] - lijn[0][1]) < 10:
                self.lbl_status.setText("Lijn te kort — klik twee punten verder uit elkaar.")
            elif self.radio_rij.isChecked():
                self._rijlijnen.append(lijn)
            else:
                self._dwarslijnen.append(lijn)
            self._herkalibreer()
        self._render()

    def _wis_laatste(self):
        if self._klik_punt is not None:
            self._klik_punt = None
        elif self._dwarslijnen and (self.radio_dwars.isChecked() or not self._rijlijnen):
            self._dwarslijnen.pop()
        elif self._rijlijnen:
            self._rijlijnen.pop()
        self._herkalibreer()
        self._render()

    def _wis_alles(self):
        self._rijlijnen = []
        self._dwarslijnen = []
        self._klik_punt = None
        self._herkalibreer()
        self._render()

    @staticmethod
    def _lijn_hint(n_rij):
        """Extra uitleg bij een mislukte kalibratie met 3+ baanlijnen.

        Met drie of meer baanlijnen bouwt de kalibratie de verdwijnlijn met de
        kruisverhouding, en die gebruikt hun onderlinge afstanden — de dialoog neemt aan
        dat ze **gelijkmatig** verdeeld zijn. Zijn ze dat niet (blauwe baanlijn, ijsrand
        en boardingvoet liggen zelden op gelijke afstand), dan is het stelsel niet met
        één camera te rijmen en volgt een 'f² ≤ 0'-melding die die oorzaak niet noemt.
        Nagemeten: ongelijk verdeeld + verteld als gelijk = geweigerd; met de juiste
        onderlinge afstanden = 0,00° fout. Het gaat dus nooit stilzwijgend mis."""
        if n_rij < 3:
            return ""
        return ("\n\nTip: met 3+ baanlijnen wordt aangenomen dat ze GELIJKMATIG verdeeld "
                "zijn. Zijn ze dat niet, gebruik dan precies 2 baanlijnen + 2 "
                "dwarslijnen — dan doet hun onderlinge afstand niet meer mee.")

    # ── schaal aan/uit ───────────────────────────────────────────────────
    def _schaal_gewijzigd(self, alleen_hoeken):
        """Koppelt 'alleen hoeken' aan de velden die een echte schaal veronderstellen.

        Zonder bekende lijnafstand zijn de wereldcoördinaten op een willekeurige factor
        na bepaald. Voor `beenvlak` maakt dat niets uit (de hoek volgt uit richtingen),
        maar `onderbeen` snijdt met een bol van een lengte in échte meters — die factor
        werkt daar direct door in de hoek. Daarom wordt de methode dan vastgezet op
        `beenvlak` in plaats van de gebruiker een stille foutbron te laten kiezen."""
        for w in (self.spin_lijnafstand, self.lbl_lijnafstand):
            w.setEnabled(not alleen_hoeken)
        if alleen_hoeken:
            idx = self.combo_methode.findData("beenvlak")
            if idx >= 0:
                self.combo_methode.setCurrentIndex(idx)
        self.combo_methode.setEnabled(not alleen_hoeken)
        self.combo_methode.setToolTip(
            "Vastgezet op 'beenvlak' omdat die geen enkele lengte in meters gebruikt.\n"
            "Zet 'Alleen hoeken' uit en vul de echte lijnafstand in om 'onderbeenlengte'\n"
            "te kunnen kiezen (die heeft een echte schaal nodig)."
            if alleen_hoeken else
            "Hoe de knie-diepte wordt gereconstrueerd. Beide zijn experimenteel te\n"
            "vergelijken; 'onderbeenlengte' heeft de lengte hieronder nodig.")
        # De lengtevelden horen alleen bij 'onderbeen'.
        lengte_nodig = (not alleen_hoeken
                        and self.combo_methode.currentData() == "onderbeen")
        for w in (self.spin_lengte, self.spin_onderbeen, self.lbl_onderbeen):
            w.setEnabled(lengte_nodig)
        if self.lbl_lengte is not None:
            self.lbl_lengte.setEnabled(lengte_nodig)
        self._herkalibreer()

    # ── kalibratie ───────────────────────────────────────────────────────
    @staticmethod
    def _sorteer_rijlijnen(lijnen):
        """Sorteer de baanlijnen ruimtelijk (aangrenzend), zodat de gelijkmatige
        offsets kloppen ongeacht de tekenvolgorde: projecteer de lijnmiddens op de
        richting loodrecht op de gemiddelde lijnrichting."""
        richtingen = []
        for p1, p2 in lijnen:
            d = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
            d /= np.hypot(d[0], d[1]) or 1.0
            if richtingen and float(d @ richtingen[0]) < 0:
                d = -d
            richtingen.append(d)
        gem = np.mean(richtingen, axis=0)
        gem /= np.hypot(gem[0], gem[1]) or 1.0
        n = np.array([-gem[1], gem[0]])
        return sorted(lijnen, key=lambda seg: float(
            (seg[0][0] + seg[1][0]) / 2 * n[0] + (seg[0][1] + seg[1][1]) / 2 * n[1]))

    def _herkalibreer(self):
        self._kalibratie = None
        self._invoer = None
        n_rij, n_dwars = len(self._rijlijnen), len(self._dwarslijnen)
        if n_rij < 2 or n_dwars < 1:
            self.lbl_status.setText(
                f"Getekend: {n_rij} baanlijn(en), {n_dwars} dwarslijn(en).\n"
                f"Nodig: minstens 2 baanlijnen + 1 dwarslijn "
                f"(2+1 alleen met opgegeven brandpuntsafstand; anders 3+1 of 2+2).")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        # Via de invoer kalibreren (niet rechtstreeks): dan loopt wat hier live te zien
        # is langs exact dezelfde weg als een later heropende analyse.
        alleen_hoeken = self.chk_alleen_hoeken.isChecked()
        invoer = schaats_perspectief.KalibratieInvoer(
            rijlijnen=self._sorteer_rijlijnen(self._rijlijnen),
            dwarslijnen=list(self._dwarslijnen),
            beeld_w=self._orig_w, beeld_h=self._orig_h,
            lijnafstand=self.spin_lijnafstand.value(),
            schaal_bekend=not alleen_hoeken,
            f_px=self.spin_f.value() or None)
        try:
            self._kalibratie = invoer.kalibreer()
            self._invoer = invoer
        except ValueError as e:
            self.lbl_status.setText(f"Kalibratie lukt nog niet: {e}{self._lijn_hint(n_rij)}")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        kal = self._kalibratie
        # Zonder bekende schaal is de camerahoogte in willekeurige eenheden; die dan
        # in meters tonen zou een precisie suggereren die er niet is.
        hoogte = (f"camerahoogte {kal.camera_hoogte:.1f} m, " if kal.schaal_bekend
                  else "")
        # Met precies 2 baanlijnen + 2 dwarslijnen is het stelsel exact bepaald: het
        # residu is dan per constructie 0,00 px en zegt niets over de kwaliteit —
        # tonen zou als "perfect gekalibreerd" gelezen worden. Een derde dwarslijn
        # maakt V2 een kleinste-kwadraten-fit en het residu wél informatief.
        overbepaald = len(self._dwarslijnen) >= 3 or len(self._rijlijnen) >= 3
        residu = (f", residu {kal.residu_px:.1f} px" if overbepaald else "")
        tekst = (f"Kalibratie OK — f = {kal.f:.0f} px"
                 f"{' (geschat)' if kal.f_geschat else ''}, {hoogte}"
                 f"horizon {kal.horizon_deg:+.2f}°{residu}.")
        if not overbepaald:
            tekst += ("\nPrecies genoeg lijnen: er is géén controle mogelijk. Teken een "
                      "derde dwarslijn om te zien of de kalibratie klopt.")
        if not kal.schaal_bekend:
            tekst += ("\nAlleen hoeken: die zijn schaalvrij en dus exact; snelheid en "
                      "slaglengte blijven leeg.")
        if kal.waarschuwingen:
            tekst += "\n⚠ " + "\n⚠ ".join(kal.waarschuwingen)
        self.lbl_status.setText(tekst)
        self.btn_ok.setEnabled(True)
        self._render()

    def _bevestig(self):
        if self._kalibratie is None:
            return
        methode = self.combo_methode.currentData()
        # 'beenvlak' gebruikt geen lengte; er dan tóch een meegeven zou in de opslag en
        # de Info-dialoog suggereren dat hij de meting beïnvloedt.
        if methode == "onderbeen":
            onderbeen_l = (self.spin_onderbeen.value() / 100.0
                           if self.spin_onderbeen.value() > 0
                           else schaats_perspectief.onderbeen_uit_lichaamslengte(
                               self.spin_lengte.value()))
        else:
            onderbeen_l = None
        self.perspectief = PerspectiefConfig(
            kalibratie=self._kalibratie,
            methode=methode,
            onderbeen_l=onderbeen_l,
            invoer=self._invoer)
        self.accept()


class AnalyseAfgebroken(Exception):
    """Coöperatief afbreken van een lopende (batch-)analyse: `breek_af()` zet een vlag en
    de voortgangs-callback — die elke pass per frame aanroept — gooit deze uitzondering.
    Zo stopt een analyse binnen één frame i.p.v. dat de thread bij het afsluiten van de
    app vernietigd wordt terwijl hij nog draait (Qt: 'Destroyed while thread is still
    running'). Het opslaan zelf wordt nooit halverwege afgebroken — een halve videokopie
    in de mediamap is erger dan even wachten."""


class AnalyseWorker(QThread):
    """Draait de analyse op de achtergrond, zodat de GUI niet blokkeert, en slaat het
    resultaat daarna automatisch op in de bibliotheek (fase 1). Het opslaan gebeurt
    bewust ook in deze thread: de videokopie naar de mediamap kan lang duren."""
    voortgang = Signal(int, int)
    status = Signal(str)                     # tekst voor de voortgangsdialoog (busy-fase)
    klaar = Signal(object, object, object, object)   # info, resultaten, events, analyse_id
    fout = Signal(str)                       # analyse zelf mislukt
    opslag_fout = Signal(str)                # alléén het opslaan mislukt (analyse is er wel)
    waarschuwing = Signal(str)               # stille terugval in de analyse (bv. klik raakte niemand)

    def __init__(self, input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                 doel_punt=None, horizon_deg=0.0, auto_horizon=False, smooth_landmarks=True,
                 perspectief=None, bieb=None, schaatser_id=None, titel=None,
                 instellingen=None, backend=None, aangemaakt_door="", bocht=True):
        super().__init__()
        self.input_pad = input_pad
        self.model_pad = model_pad
        self.smooth_n = smooth_n
        self.threshold = threshold
        self.force_fps = force_fps
        self.doel_punt = doel_punt
        self.horizon_deg = horizon_deg
        self.auto_horizon = auto_horizon
        self.smooth_landmarks = smooth_landmarks
        self.bocht = bocht
        self.perspectief = perspectief
        self.bieb = bieb
        self.schaatser_id = schaatser_id
        self.titel = titel
        self.instellingen = instellingen
        self.backend = backend
        self.aangemaakt_door = aangemaakt_door
        self.afbreken = False

    def breek_af(self):
        """Vraagt de analyse te stoppen (afsluiten van de app). De thread eindigt bij de
        eerstvolgende frame-callback, zonder signaal en zonder op te slaan."""
        self.afbreken = True

    def run(self):
        try:
            def toon_voortgang(frame_nr, totaal):
                if self.afbreken:
                    raise AnalyseAfgebroken()
                self.voortgang.emit(frame_nr, totaal)

            info, resultaten = analyseer_backend(
                self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                self.force_fps, doel_punt=self.doel_punt, progress_callback=toon_voortgang,
                horizon_deg=self.horizon_deg, auto_horizon=self.auto_horizon,
                smooth_landmarks=self.smooth_landmarks, perspectief=self.perspectief,
                waarschuwing_callback=self.waarschuwing.emit, bocht=self.bocht,
            )
            events = segmenteer_afzetten(resultaten)
        except AnalyseAfgebroken:
            return                    # afsluiten: niets melden, niets opslaan
        except Exception as e:
            self.fout.emit(str(e))
            return
        if self.afbreken:
            return                    # niet meer aan een lange videokopie beginnen

        # Opslaan in de bibliotheek; faalt dit, dan gaat de (lange) analyse niet
        # verloren — de resultaten worden alsnog getoond, alleen niet bewaard.
        analyse_id = None
        if self.bieb is not None and self.schaatser_id is not None:
            self.status.emit("Opslaan in bibliotheek...")
            try:
                analyse_id = schaats_db.sla_analyse_op(
                    self.bieb, self.schaatser_id, self.titel, self.input_pad,
                    info, resultaten, events,
                    backend=self.backend, instellingen=self.instellingen,
                    aangemaakt_door=self.aangemaakt_door)
            except Exception as e:
                self.opslag_fout.emit(str(e))
        self.klaar.emit(info, resultaten, events, analyse_id)


class BatchWorker(QThread):
    """Draait een reeks analyses achter elkaar op de achtergrond en slaat elke video
    automatisch op in de bibliotheek. Eén slechte clip stopt de batch niet — die wordt als
    mislukt gemeld en de rest loopt door. 'Stop na deze video' vraagt via requestInterruption()
    een nette stop aan die tussen de video's wordt afgehandeld (de lopende video wordt eerst
    afgemaakt en opgeslagen)."""
    taak_start  = Signal(int, int, str)          # index (0-based), totaal, titel
    voortgang   = Signal(int, int)               # frame_nr, totaal van de huidige video
    status      = Signal(str)                     # busy-tekst (videokopie naar de bibliotheek)
    taak_klaar  = Signal(int, object)            # index, analyse_id (of None)
    taak_fout   = Signal(int, str)               # index, foutmelding — batch gaat door
    alles_klaar = Signal(list, list, list)       # geslaagde titels, [(titel, melding)] mislukt,
                                                 # [(titel, melding)] waarschuwingen

    def __init__(self, taken, bieb, backend, aangemaakt_door=""):
        super().__init__()
        self.taken = taken
        self.bieb = bieb
        self.backend = backend
        self.aangemaakt_door = aangemaakt_door
        self.afbreken = False

    def breek_af(self):
        """Hard stoppen (afsluiten van de app): ook de lopende video wordt afgebroken.
        Bewust iets anders dan `requestInterruption()` ('Stop na deze video'), dat de
        huidige video juist netjes laat afmaken en opslaan."""
        self.afbreken = True

    def _voortgang(self, frame_nr, totaal):
        if self.afbreken:
            raise AnalyseAfgebroken()
        self.voortgang.emit(frame_nr, totaal)

    def run(self):
        n = len(self.taken)
        geslaagd, fouten, waarschuwingen = [], [], []
        for i, taak in enumerate(self.taken):
            if self.afbreken or self.isInterruptionRequested():
                break                            # 'Stop na deze video' — rest overslaan
            self.taak_start.emit(i, n, taak["titel"])
            try:
                # Waarschuwingen (bv. een klik die niemand raakte) niet per video in een
                # modale box gooien — een batch draait juist onbewaakt; verzamelen en aan
                # het eind in het overzicht melden, mét de titel erbij.
                def _waarschuw(tekst, titel=taak["titel"]):
                    waarschuwingen.append((titel, tekst))

                info, resultaten = analyseer_backend(
                    taak["input_pad"], taak["model_pad"], taak["smooth_n"], taak["threshold"],
                    doel_punt=taak["doel_punt"],
                    progress_callback=self._voortgang,
                    horizon_deg=taak["horizon_deg"], auto_horizon=taak["auto_horizon"],
                    smooth_landmarks=taak["smooth_landmarks"],
                    perspectief=taak.get("perspectief"),
                    waarschuwing_callback=_waarschuw, bocht=taak.get("bocht", True),
                )
                events = segmenteer_afzetten(resultaten)
                if resultaten and all(r.bocht for r in resultaten):
                    # Anders staat deze clip straks als "0 afzetten" in de lijst zonder dat
                    # iemand weet waarom.
                    _waarschuw("De schaatser staat nergens frontaal in beeld; de hele video "
                               "is als bocht aangemerkt en er is niets gemeten.")
                if self.afbreken:
                    break                        # niet meer aan een lange videokopie beginnen
                self.status.emit("Opslaan in bibliotheek...")
                analyse_id = schaats_db.sla_analyse_op(
                    self.bieb, taak["schaatser_id"], taak["titel"], taak["input_pad"],
                    info, resultaten, events,
                    backend=self.backend, instellingen=taak["instellingen"],
                    aangemaakt_door=self.aangemaakt_door,
                    bron_id=taak.get("bron_id"),
                    bron_start_frame=taak.get("bron_start_frame"),
                    bron_eind_frame=taak.get("bron_eind_frame"))
                geslaagd.append(taak["titel"])
                self.taak_klaar.emit(i, analyse_id)
            except AnalyseAfgebroken:
                break                            # afsluiten: rest van de rij vervalt
            except Exception as e:
                # sla_analyse_op ruimt zijn eigen halve mediamap op; hier alleen registreren.
                fouten.append((taak["titel"], str(e)))
                self.taak_fout.emit(i, str(e))
        if not self.afbreken:
            self.alles_klaar.emit(geslaagd, fouten, waarschuwingen)


class SchaatserDialog(QDialog):
    """Schaatser-profiel aanmaken of bewerken: naam, geboortejaar, notities."""

    def __init__(self, parent=None, naam="", geboortejaar=None, notities=""):
        super().__init__(parent)
        self.setWindowTitle("Schaatser")
        form = QFormLayout(self)

        self.veld_naam = QLineEdit(naam)
        form.addRow("Naam:", self.veld_naam)

        self.veld_jaar = QSpinBox()
        self.veld_jaar.setRange(0, 2100)
        self.veld_jaar.setSpecialValueText("—")   # 0 = niet ingevuld
        self.veld_jaar.setValue(geboortejaar or 0)
        form.addRow("Geboortejaar:", self.veld_jaar)

        self.veld_notities = QPlainTextEdit(notities or "")
        self.veld_notities.setFixedHeight(70)
        form.addRow("Notities:", self.veld_notities)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        form.addRow(knoppen)

        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setEnabled(bool(naam.strip()))
        self.veld_naam.textChanged.connect(lambda t: self._ok.setEnabled(bool(t.strip())))

    @property
    def naam(self):
        return self.veld_naam.text().strip()

    @property
    def geboortejaar(self):
        return self.veld_jaar.value() or None

    @property
    def notities(self):
        return self.veld_notities.toPlainText().strip()


class NieuweAnalyseDialog(QDialog):
    """Verzamelt alles voor één nieuwe analyse: schaatser, video, titel en de
    analyse-instellingen (verhuisd van de oude startpagina, fase 1)."""

    def __init__(self, schaatsers, voorkeur_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nieuwe analyse")
        self.video_pad = None
        v = QVBoxLayout(self)

        form = QFormLayout()
        self.combo_schaatser = QComboBox()
        for s in schaatsers:
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_schaatser.addItem(tekst, s["id"])
        if voorkeur_id is not None:
            idx = self.combo_schaatser.findData(voorkeur_id)
            if idx >= 0:
                self.combo_schaatser.setCurrentIndex(idx)
        form.addRow("Schaatser:", self.combo_schaatser)

        rij_video = QHBoxLayout()
        knop_video = QPushButton("Video kiezen...")
        knop_video.clicked.connect(self._kies_video)
        self.lbl_video = QLabel("Geen video gekozen")
        rij_video.addWidget(knop_video)
        rij_video.addWidget(self.lbl_video, stretch=1)
        form.addRow("Video:", rij_video)

        self.veld_titel = QLineEdit()
        self.veld_titel.setPlaceholderText("standaard: naam van het videobestand")
        form.addRow("Titel:", self.veld_titel)
        v.addLayout(form)

        instellingen = QGroupBox("Instellingen")
        fv = QVBoxLayout(instellingen)

        rij_smooth = QHBoxLayout()
        rij_smooth.addWidget(QLabel("Smoothing (frames):"))
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 30)
        self.spin_smooth.setValue(5)
        rij_smooth.addStretch(1)
        rij_smooth.addWidget(self.spin_smooth)
        fv.addLayout(rij_smooth)

        rij_threshold = QHBoxLayout()
        rij_threshold.addWidget(QLabel("Gewicht-drempel:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 0.2)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.015)
        rij_threshold.addStretch(1)
        rij_threshold.addWidget(self.spin_threshold)
        fv.addLayout(rij_threshold)

        self.chk_heavy = QCheckBox("Heavy-model (nauwkeuriger, trager)")
        self.chk_heavy.setVisible(not IS_YOLO)   # alleen relevant voor de MediaPipe-backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Bocht overslaan (sneller)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(BOCHT_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        self.chk_perspectief = QCheckBox("Perspectiefcorrectie via baanlijnen (experimenteel)")
        self.chk_perspectief.setToolTip(PERSPECTIEF_TOOLTIP)
        fv.addWidget(self.chk_perspectief)

        self.chk_geen_smoothing = QCheckBox("Geen landmark-smoothing (ruwe detecties)")
        self.chk_geen_smoothing.setToolTip(
            "Slaat het opschonen + Savitzky–Golay-smoothen van de landmark-trajecten\n"
            "over: het skelet volgt de detecties exact (kan trillen), maar kan nooit\n"
            "achterlopen door interpolatie. Handig om te zien of een achterlopend\n"
            "skelet uit de smoothing komt of uit de detectie zelf.")
        fv.addWidget(self.chk_geen_smoothing)

        v.addWidget(instellingen)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setText("Start analyse")
        self._ok.setEnabled(False)               # pas actief mét gekozen video

    def _kies_video(self):
        pad, _ = QFileDialog.getOpenFileName(
            self, "Kies video", "", "Video's (*.mp4 *.mov *.avi *.mkv);;Alle bestanden (*)")
        if not pad:
            return
        self.video_pad = pad
        self.lbl_video.setText(os.path.basename(pad))
        if not self.veld_titel.text().strip():
            self.veld_titel.setText(os.path.splitext(os.path.basename(pad))[0])
        self._ok.setEnabled(True)

    @property
    def schaatser_id(self):
        return self.combo_schaatser.currentData()

    @property
    def titel(self):
        tekst = self.veld_titel.text().strip()
        if tekst:
            return tekst
        return os.path.splitext(os.path.basename(self.video_pad or "analyse"))[0]


class BatchAnalyseDialog(QDialog):
    """Verzamelt een hele batch in één dialoog: meerdere video's tegelijk, elk met een
    eigen schaatser en titel, plus gedeelde analyse-instellingen. De doel- en horizon-keuze
    gebeurt daarna per video in de verzamellus (MainWindow._nieuwe_batch_analyse)."""

    def __init__(self, schaatsers, voorkeur_id=None, voorgevuld=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fragmenten analyseren" if voorgevuld else "Batch-analyse")
        self.resize(760, 500)
        self._schaatsers = schaatsers
        v = QVBoxLayout(self)

        # Video's kiezen + standaard-schaatser die je in één keer op alle rijen zet.
        rij_top = QHBoxLayout()
        knop_videos = QPushButton("Video's kiezen...")
        knop_videos.clicked.connect(self._kies_videos)
        rij_top.addWidget(knop_videos)
        rij_top.addWidget(QLabel("Standaard schaatser:"))
        self.combo_standaard = QComboBox()
        for s in schaatsers:
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_standaard.addItem(tekst, s["id"])
        if voorkeur_id is not None:
            idx = self.combo_standaard.findData(voorkeur_id)
            if idx >= 0:
                self.combo_standaard.setCurrentIndex(idx)
        rij_top.addWidget(self.combo_standaard, stretch=1)
        knop_toepassen = QPushButton("Toepassen op alle rijen")
        knop_toepassen.clicked.connect(self._pas_standaard_toe)
        rij_top.addWidget(knop_toepassen)
        v.addLayout(rij_top)

        # Video's + per rij een schaatser (combobox) en een bewerkbare titel.
        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["Video", "Schaatser", "Titel"])
        self.tabel.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tabel.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tabel.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self.tabel, stretch=1)

        knop_verwijder = QPushButton("Geselecteerde rij verwijderen")
        knop_verwijder.clicked.connect(self._verwijder_rij)
        v.addWidget(knop_verwijder)

        # Gedeelde instellingen (dezelfde widgets/waarden als NieuweAnalyseDialog).
        instellingen = QGroupBox("Instellingen (gelden voor de hele batch)")
        fv = QVBoxLayout(instellingen)

        rij_smooth = QHBoxLayout()
        rij_smooth.addWidget(QLabel("Smoothing (frames):"))
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 30)
        self.spin_smooth.setValue(5)
        rij_smooth.addStretch(1)
        rij_smooth.addWidget(self.spin_smooth)
        fv.addLayout(rij_smooth)

        rij_threshold = QHBoxLayout()
        rij_threshold.addWidget(QLabel("Gewicht-drempel:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 0.2)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.015)
        rij_threshold.addStretch(1)
        rij_threshold.addWidget(self.spin_threshold)
        fv.addLayout(rij_threshold)

        self.chk_heavy = QCheckBox("Heavy-model (nauwkeuriger, trager)")
        self.chk_heavy.setVisible(not IS_YOLO)   # alleen relevant voor de MediaPipe-backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Bocht overslaan (sneller)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(BOCHT_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        self.chk_perspectief = QCheckBox("Perspectiefcorrectie via baanlijnen (experimenteel)")
        self.chk_perspectief.setToolTip(PERSPECTIEF_TOOLTIP_BATCH)
        fv.addWidget(self.chk_perspectief)

        self.chk_geen_smoothing = QCheckBox("Geen landmark-smoothing (ruwe detecties)")
        fv.addWidget(self.chk_geen_smoothing)
        v.addWidget(instellingen)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setText("Start batch")
        self._ok.setEnabled(False)               # pas actief met minstens één video

        # Fase 8: rijen die al vaststaan (zojuist uit een opname geknipte fragmenten).
        # De trainer hoeft alleen nog schaatser + titel in te vullen — verder is dit
        # exact de bestaande batch-flow.
        for f in (voorgevuld or []):
            self._voeg_rij(f["input_pad"], f.get("titel"), bron={
                "bron_id": f.get("bron_id"),
                "bron_start_frame": f.get("bron_start_frame"),
                "bron_eind_frame": f.get("bron_eind_frame")})

    def _voeg_rij(self, pad, titel=None, bron=None):
        """Eén videorij: pad achter de eerste cel, schaatser-combo, bewerkbare titel.
        `bron` (fase 8) reist mee zodat de analyse weet uit welk stuk van welke opname
        deze clip komt."""
        r = self.tabel.rowCount()
        self.tabel.insertRow(r)
        item_pad = QTableWidgetItem(os.path.basename(pad))
        item_pad.setData(Qt.UserRole, pad)                 # volledige pad achter de rij
        item_pad.setData(Qt.UserRole + 1, bron)
        item_pad.setFlags(item_pad.flags() & ~Qt.ItemIsEditable)
        self.tabel.setItem(r, 0, item_pad)
        self.tabel.setCellWidget(r, 1, self._maak_schaatser_combo())
        self.tabel.setItem(r, 2, QTableWidgetItem(
            titel or os.path.splitext(os.path.basename(pad))[0]))
        self._ok.setEnabled(True)

    def _maak_schaatser_combo(self):
        """Een per-rij schaatser-keuze, voorgeselecteerd op de huidige standaard."""
        combo = QComboBox()
        for s in self._schaatsers:
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            combo.addItem(tekst, s["id"])
        idx = combo.findData(self.combo_standaard.currentData())
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    def _kies_videos(self):
        paden, _ = QFileDialog.getOpenFileNames(
            self, "Kies video's", "", "Video's (*.mp4 *.mov *.avi *.mkv);;Alle bestanden (*)")
        for pad in paden:
            self._voeg_rij(pad)
        self._ok.setEnabled(self.tabel.rowCount() > 0)

    def _pas_standaard_toe(self):
        sid = self.combo_standaard.currentData()
        for r in range(self.tabel.rowCount()):
            combo = self.tabel.cellWidget(r, 1)
            if combo is not None:
                idx = combo.findData(sid)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _verwijder_rij(self):
        r = self.tabel.currentRow()
        if r >= 0:
            self.tabel.removeRow(r)
        self._ok.setEnabled(self.tabel.rowCount() > 0)

    @property
    def taken(self):
        """Lijst van {input_pad, schaatser_id, titel} — één per video-rij."""
        rijen = []
        for r in range(self.tabel.rowCount()):
            item_pad = self.tabel.item(r, 0)
            pad = item_pad.data(Qt.UserRole)
            combo = self.tabel.cellWidget(r, 1)
            schaatser_id = combo.currentData() if combo is not None else None
            titel_item = self.tabel.item(r, 2)
            titel = titel_item.text().strip() if titel_item is not None else ""
            if not titel:
                titel = os.path.splitext(os.path.basename(pad))[0]
            taak = {"input_pad": pad, "schaatser_id": schaatser_id, "titel": titel}
            taak.update(item_pad.data(Qt.UserRole + 1) or {})   # bron_* (fase 8), of niets
            rijen.append(taak)
        return rijen

    @property
    def smooth_n(self):
        return self.spin_smooth.value()

    @property
    def threshold(self):
        return self.spin_threshold.value()

    @property
    def heavy_gevraagd(self):
        return self.chk_heavy.isChecked()

    @property
    def geen_smoothing(self):
        return self.chk_geen_smoothing.isChecked()


class AnalyseKiezer(QDialog):
    """
    Kiest één opgeslagen analyse: eerst de schaatser, dan een van diens analyses.
    Tweemaal achter elkaar gebruikt om een vergelijking op te zetten, en daarna per kant
    om van analyse te wisselen.
    """

    def __init__(self, bieb, titel="Kies analyse", voorkeur_schaatser_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titel)
        self.bieb = bieb

        form = QFormLayout(self)
        self.combo_schaatser = QComboBox()
        # Schaatsers zonder analyses overslaan — dan kan de analyse-combo nooit leeg zijn.
        for s in schaats_db.lijst_schaatsers(bieb):
            if not s["aantal_analyses"]:
                continue
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_schaatser.addItem(tekst, s["id"])
        if voorkeur_schaatser_id is not None:
            idx = self.combo_schaatser.findData(voorkeur_schaatser_id)
            if idx >= 0:
                self.combo_schaatser.setCurrentIndex(idx)
        self.combo_schaatser.currentIndexChanged.connect(self._vul_analyses)
        form.addRow("Schaatser:", self.combo_schaatser)

        self.combo_analyse = QComboBox()
        self.combo_analyse.setMinimumWidth(380)
        form.addRow("Analyse:", self.combo_analyse)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        form.addRow(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setText("Kiezen")

        self._vul_analyses()

    def _vul_analyses(self, _idx=None):
        self.combo_analyse.clear()
        sid = self.combo_schaatser.currentData()
        if sid is not None:
            for a in schaats_db.lijst_analyses(self.bieb, sid):
                gem = f"{a['gem_hoek']:.1f}°" if a["gem_hoek"] is not None else "—"
                self.combo_analyse.addItem(
                    f"{a['datum']} — {a['titel']}  "
                    f"({a['aantal_afzetten']} afzetten, gem {gem})", a["id"])
        self._ok.setEnabled(self.combo_analyse.count() > 0)

    @property
    def analyse_id(self):
        return self.combo_analyse.currentData()

    @property
    def schaatser_naam(self):
        sid = self.combo_schaatser.currentData()
        if sid is None:
            return ""
        # de combotekst draagt evt. het geboortejaar; voor de kop willen we alleen de naam
        naam = next((s["naam"] for s in schaats_db.lijst_schaatsers(self.bieb)
                     if s["id"] == sid), "")
        return naam


def _ja_nee(waarde):
    return "ja" if waarde else "nee"


def _duur_tekst(a):
    """Videoduur + aantal afzetten voor de bibliotheeklijst: "5,4s (4 afzetten)".

    De duur zegt op één oogopslag wat voor clip dit is (losse slag of hele ronde); het
    aantal afzetten blijft er tussen haakjes bij staan. Boven de minuut wordt het m:ss,
    want "83,2s" leest niemand als anderhalve minuut. Ontbreekt fps of framecount (een
    onvolledig weggeschreven analyse), dan alleen een streepje — geen deling door nul."""
    fps = a.get("fps") or 0
    frames = a.get("totaal_frames") or 0
    if fps > 0 and frames > 0:
        sec = frames / fps
        duur = (f"{sec:.1f}s".replace(".", ",") if sec < 60
                else f"{int(sec) // 60}:{int(sec) % 60:02d}")
    else:
        duur = "—"
    n = a.get("aantal_afzetten") or 0
    return f"{duur} ({n} afzet{'ten' if n != 1 else ''})"


class AnalyseInfoDialog(QDialog):
    """
    Read-only overzicht van één opgeslagen analyse: met welke appversie/backend hij
    gedraaid is, wanneer en door wie, en met welke instellingen.

    Waarom: de trackinglogica wijzigt tijdens het ontwikkelen regelmatig, dus een vreemde
    meting moet te verklaren zijn ("dit is nog met de oude L/R-fixer gedaan"). Puur
    informatief — geen invoerveld. De waarden zijn selecteerbaar zodat een commit-hash
    te kopiëren is.
    """

    APPVERSIE_TIP = ("Git-commit waarmee deze analyse gedraaid is (commitdatum · hash).\n"
                     "Een '+' betekent: er stonden op dat moment ongecommitte wijzigingen\n"
                     "in de code, dus de hash beschrijft de analyse niet volledig.")

    def __init__(self, meta, schaatser_naam="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Info over deze analyse")
        inst = meta.get("instellingen") or {}
        form = QFormLayout(self)

        gemaakt = meta.get("aangemaakt_op") or ""
        datum = meta.get("datum") or "—"
        if gemaakt:
            datum += f"   (opgeslagen {gemaakt})"

        # Smoothing uit = de diagnosestand (ruwe detecties, CLI --no-smooth).
        smoothing = (f"{inst.get('smooth_n', '?')} frames"
                     if inst.get("smooth_landmarks", True) else "uit (ruwe detecties)")
        horizon = ("automatisch per frame" if inst.get("auto_horizon")
                   else f"vast {inst.get('horizon_deg', 0.0):.1f}°")
        # `heavy` kiest tussen pose_landmarker_heavy en _full en bestaat dus alleen in de
        # MediaPipe-backend; YOLO heeft één model en negeert de vlag — daar zou de rij
        # ("nee") alleen suggereren dat er een zwaarder model te kiezen viel.
        heavy_rij = ([] if meta.get("backend") == "yolo" else
                     [("Heavy-model:", _ja_nee(inst.get("heavy")),
                       "MediaPipe: pose_landmarker_heavy.task i.p.v. _full.task.")])

        # Herkomst (fase 8): alleen bij een uit een opname geknipt fragment. Een losse clip
        # heeft geen bron, en dan zegt een lege rij niets.
        herkomst_rij = []
        if meta.get("bron_id") and meta.get("bron_start_frame") is not None:
            fps = meta.get("fps") or 0
            plek = (f" ({_tijd_tekst(meta['bron_start_frame'], fps)}–"
                    f"{_tijd_tekst(meta['bron_eind_frame'], fps)})" if fps else "")
            herkomst_rij = [("Uit opname:",
                             f"{meta.get('bron_naam') or 'onbekend'}{plek}",
                             "Dit fragment is met het knipvenster uit een langere "
                             "trainingsopname geknipt.")]

        rijen = [
            ("Titel:", meta.get("titel") or "—", None),
            ("Schaatser:", schaatser_naam or "—", None),
            ("Analysedatum:", datum, None),
            ("Aangemaakt door:", meta.get("aangemaakt_door") or "—", None),
            ("Appversie:", inst.get("app_versie") or "onbekend (van vóór deze functie)",
             self.APPVERSIE_TIP),
            ("Backend:", inst.get("backend_naam") or meta.get("backend") or "—", None),
            ("Video:", f"{os.path.basename(meta.get('video_bestand') or '')}  —  "
                       f"{meta.get('w')}×{meta.get('h')} @ "
                       f"{(meta.get('fps') or 0):.1f} fps, "
                       f"{meta.get('totaal_frames')} frames", None),
        ] + herkomst_rij + [
            ("Handmatig bewerkt:", _ja_nee(meta.get("bewerkt")),
             "Zijn er met de skelet-editor punten verplaatst of skeletten geplaatst?"),
            ("Smoothing:", smoothing, None),
            ("Drempel:", f"{inst.get('threshold', '?')}", None),
        ] + heavy_rij + [
            ("Bocht overslaan:", _ja_nee(inst.get("bocht_overslaan")), None),
            ("Horizon:", horizon, None),
            ("Perspectiefcorrectie:", _ja_nee(inst.get("perspectief_gebruikt")), None),
        ] + _kalibratie_rijen(inst)
        for label, waarde, tip in rijen:
            w = QLabel(str(waarde))
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tip:
                w.setToolTip(tip)
            form.addRow(label, w)

        knoppen = QDialogButtonBox(QDialogButtonBox.Close)
        knoppen.rejected.connect(self.reject)
        form.addRow(knoppen)


class VideoSpeler(QWidget):
    """
    Videopaneel met een eigen capture, afspeeltimer en zoom/pan-state: beeldlabel +
    transportknoppen + afspeelsnelheid + scrub-slider + laag-toggles + zoomregelaars.

    Zelfstandig, zodat er meerdere naast elkaar kunnen bestaan (de analysepagina heeft er
    één, de vergelijkpagina twee). De speler is de **enige** eigenaar van `video_info`,
    `resultaten`, `huidige_idx` en `video_pad`; MainWindow kijkt er via read-only properties
    naar, zodat er nooit stilzwijgend een tweede kopie ontstaat die uit de pas loopt.

    De eigenaar haakt in met plain callables — géén signalen, want een QMouseEvent overleeft
    een queued connectie niet en er is per speler precies één eigenaar:
        op_frame_getoond(idx)     — ná het tekenen van een frame (grafiek/tabel/statusbalk)
        overlay_tekenaar(pixmap)  — vlak vóór setPixmap (de skelet-editor tekent z'n handles)
        op_muis_druk/_beweeg/_los(event)
                                  — alleen als de speler het event niet zelf als pan-sleep
                                    heeft opgeslokt
    """

    def __init__(self, min_grootte=(480, 320), toon_snelheid=True, snel_zoeken=False,
                 toon_overlay=True, parent=None):
        super().__init__(parent)

        # Zonder analyse valt er niets te tekenen: het knipvenster (fase 8) voedt de speler
        # met lege FrameResultaat-objecten, en dan zou de overlay op élk frame "Geen pose
        # gedetecteerd" zetten. `toon_overlay=False` slaat het tekenen over en verbergt de
        # laag-vinkjes, die daar toch niets doen.
        self.toon_overlay = toon_overlay

        # `snel_zoeken` ruilt frame-exactheid in voor bruikbaarheid op een lange opname —
        # zie _lees_frame_exact. Alleen aanzetten waar het beeld een kijkje is en geen meting
        # (het knipvenster van fase 8); de weergavepagina laat hem uit.
        self.snel_zoeken = snel_zoeken

        # Weergave-state (per speler, zodat er meerdere tegelijk kunnen draaien)
        self.video_pad = None
        self.video_info = None
        self.resultaten = []
        self.huidige_idx = -1
        self.cap = None
        self._weergave_pos = 0        # frames al gelezen door cap (sequentiële cursor)
        self._laatste_frame = None    # ruwe kopie van het huidige frame (voor laag-toggles)
        self._weergave_scaled = None  # QSize van de getoonde (geschaalde) pixmap, voor omrekening

        # Inzoomen op de schaatser. Twee zoomwaarden, bewust uit elkaar gehouden:
        # `_zoom` is wat de gebruiker instelde (slider/wiel, 1–ZOOM_MAX), `_zoom_eff` is wat
        # er daadwerkelijk getoond wordt. Zonder automatische zoom zijn ze gelijk.
        self._zoom = 1.0            # 1.0 = passend (geen crop); tot ZOOM_MAX
        self._zoom_eff = 1.0        # toegepaste zoom van het huidige frame (tot ZOOM_AUTO_MAX)
        self._pan_cx = 0.5          # genormaliseerd middelpunt van de uitsnede (volledig frame)
        self._pan_cy = 0.5
        self._zoom_volg = True      # auto-centreren op de schaatser (spiegel van chk_volg)
        self._volg_forceren = False  # eenmalig centreren zonder framewissel (na een zoom-actie)
        self._zoom_auto = False     # zoom door het programma laten bepalen (spiegel van chk_auto)
        self._kader = None          # (midden_x, midden_y, straal) per frame, of None
        self._crop_norm = (0.0, 0.0, 1.0, 1.0)  # (x0n, y0n, breedten, hoogten): getoonde crop
        self._pan_sleep = None      # laatste muispositie tijdens een handmatige pan-sleep

        # Haken voor de eigenaar (zie de klasse-docstring)
        self.op_frame_getoond = None
        self.overlay_tekenaar = None
        self.op_muis_druk = None
        self.op_muis_beweeg = None
        self.op_muis_los = None
        self.bewerk_modus = False     # stuurt de pan-vs-editor-voorrang van de muis
        self.volgen_bevroren = False  # tijdens een editor-sleep: uitsnede niet laten verspringen

        # Scrub-samenvoeging (zie _scrub_gevraagd); moet vóór _bouw_ui bestaan, want die
        # verbindt de slider er eventueel aan.
        self._scrub_doel = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._scrub_tick)

        self._bouw_ui(min_grootte, toon_snelheid, toon_overlay)
        self.speeltimer = QTimer(self)
        self.speeltimer.timeout.connect(self._speel_tick)
        self.zet_besturing_actief(False)

    # ── UI opbouw ────────────────────────────────────────────────────────
    def _bouw_ui(self, min_grootte, toon_snelheid=True, toon_overlay=True):
        self._hoofd = QVBoxLayout(self)

        self.label = QLabel("Geen video geladen")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #111; color: #888;")
        # De expliciete minimumSize is wat de layout als ondergrens gebruikt (hij wint van de
        # minimumSizeHint, die bij een QLabel mét pixmap de pixmapgrootte is). Houd `min_grootte`
        # dus laag: hij bepaalt rechtstreeks hoe hoog het venster minimaal wordt.
        self.label.setMinimumSize(*min_grootte)
        self._hoofd.addWidget(self.label, stretch=1)

        knoppen = QHBoxLayout()
        self.btn_start = QPushButton("⏮")
        self.btn_frame_terug = QPushButton("⏪")
        self.btn_play = QPushButton("▶")
        self.btn_frame_verder = QPushButton("⏩")
        self.btn_eind = QPushButton("⏭")
        self.lbl_tijd = QLabel("t=0.00s  frame 0/0")

        self.btn_start.clicked.connect(lambda: self.ga_naar(0))
        self.btn_frame_terug.clicked.connect(lambda: self.ga_naar(self.huidige_idx - 1))
        self.btn_play.clicked.connect(self._toggle_afspelen)
        self.btn_frame_verder.clicked.connect(lambda: self.ga_naar(self.huidige_idx + 1))
        self.btn_eind.clicked.connect(lambda: self.ga_naar(len(self.resultaten) - 1))

        for w in (self.btn_start, self.btn_frame_terug, self.btn_play,
                  self.btn_frame_verder, self.btn_eind):
            # Eén teken breed: de Qt-standaardbreedte (81 px) is bedoeld voor knoppen mét
            # tekst en eiste met vijf transportknoppen 405 px per speler — twee spelers naast
            # elkaar op de vergelijkpagina paste daarmee niet op een smal laptopscherm.
            w.setMaximumWidth(TRANSPORT_KNOP_BREEDTE)
            knoppen.addWidget(w)

        # Afspeelsnelheid (slow motion): factor waarmee de fps vermenigvuldigd wordt.
        # De combo bestaat altijd (hij is de bron voor `_speel_interval_ms`), maar hoeft
        # niet zichtbaar te zijn: op de vergelijkpagina stuurt één gedeelde regelaar
        # beide kanten, zodat de video's altijd even snel lopen.
        self.lbl_snelheid = QLabel("Snelheid")
        knoppen.addWidget(self.lbl_snelheid)
        self.combo_snelheid = QComboBox()
        self.combo_snelheid.setToolTip("Afspeelsnelheid — kies een lagere factor voor slow motion.")
        for label, factor in SNELHEDEN:
            self.combo_snelheid.addItem(label, factor)
        self.combo_snelheid.setCurrentIndex(SNELHEID_DEFAULT_IDX)
        self.combo_snelheid.currentIndexChanged.connect(self._zet_snelheid)
        knoppen.addWidget(self.combo_snelheid)
        self.lbl_snelheid.setVisible(toon_snelheid)
        self.combo_snelheid.setVisible(toon_snelheid)

        knoppen.addWidget(self.lbl_tijd, stretch=1)
        self._hoofd.addLayout(knoppen)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        # Op een lange opname kost één sprong ~80 ms (seek) + tekenen, terwijl een sleep over
        # de tijdlijn honderden valueChanged-signalen afvuurt. Die stapelen zich op en de GUI
        # lijkt vast te lopen. `_scrub_gevraagd` bewaart alleen het laatst gevraagde frame en
        # tekent dat via een timer met interval 0: die vuurt pas als de wachtrij leeg is, dus
        # alle tussenliggende waarden vallen vanzelf weg en er wordt precies zo vaak getekend
        # als de machine aankan. Zonder snel_zoeken (korte clips) blijft de directe weg staan.
        self.slider.valueChanged.connect(
            self._scrub_gevraagd if self.snel_zoeken else self.ga_naar)
        self._hoofd.addWidget(self.slider)

        # Afbrekende balk i.p.v. QHBoxLayout: deze rij is met al zijn regelaars te breed voor
        # een laptopscherm (zeker twee spelers naast elkaar) en moet kunnen inklappen.
        self._balk_toggles = WrapBalk()
        self._rij_toggles = self._balk_toggles
        self.chk_skelet = QCheckBox("Skelet")
        self.chk_afzetbeen = QCheckBox("Afzetbeen")
        self.chk_hud = QCheckBox("HUD")
        for chk in (self.chk_skelet, self.chk_afzetbeen, self.chk_hud):
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _=None: self.toon_huidig_frame())
            chk.setVisible(toon_overlay)
            self._rij_toggles.addWidget(chk)

        # Inzoomen op de schaatser (muiswiel boven de video werkt ook — zie onder).
        self.chk_volg = QCheckBox("Volg schaatser")
        self.chk_volg.setChecked(True)
        self.chk_volg.setToolTip("Houd de schaatser gecentreerd in beeld tijdens het inzoomen.")
        self.chk_volg.toggled.connect(self._zet_zoom_volg)
        self._rij_toggles.addWidget(self.chk_volg)
        self.chk_auto = QCheckBox("Automatische zoom")
        self.chk_auto.setToolTip(
            "Het programma kiest de zoom: de schaatser staat helemaal in beeld met wat ruimte "
            "eromheen, de hele clip lang. Rijdt hij naar de camera toe, dan zoomt het beeld "
            "vanzelf uit.\nZolang dit aan staat is de zoomregelaar buiten werking; aan het "
            "muiswiel draaien neemt de zoom weer over.")
        self.chk_auto.toggled.connect(self._zet_zoom_auto)
        self._rij_toggles.addWidget(self.chk_auto)
        # Volgen en automatische zoom leven van de gedetecteerde pose; zonder analyse
        # (knipvenster) zouden het vinkjes zijn die niets doen.
        self.chk_volg.setVisible(toon_overlay)
        self.chk_auto.setVisible(toon_overlay)

        # De zoomregelaars als één blok in de balk: zouden ze los meedoen, dan kan het
        # label "Zoom" op de vorige regel achterblijven terwijl zijn schuif afbreekt.
        zoom_blok = QWidget()
        zoom_rij = QHBoxLayout(zoom_blok)
        zoom_rij.setContentsMargins(0, 0, 0, 0)
        zoom_rij.addWidget(QLabel("Zoom"))
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(100, int(ZOOM_MAX * 100))   # 100 = 1.0×
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(120)
        self.slider_zoom.setToolTip("Zoomniveau. Muiswiel boven de video werkt ook.")
        self.slider_zoom.valueChanged.connect(lambda v: self._zet_zoom(v / 100.0))
        zoom_rij.addWidget(self.slider_zoom)
        self.lbl_zoom = QLabel("1.0×")
        self.lbl_zoom.setFixedWidth(38)
        zoom_rij.addWidget(self.lbl_zoom)
        self.btn_zoom_reset = QPushButton("Passend")
        self.btn_zoom_reset.setToolTip("Zoom herstellen naar passend beeld.")
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        zoom_rij.addWidget(self.btn_zoom_reset)
        self._rij_toggles.addWidget(zoom_blok)

        self._hoofd.addWidget(self._balk_toggles)

        # Muis-events op het videolabel: pannen doet de speler zelf, de rest gaat naar de
        # haken van de eigenaar (de skelet-editor).
        self.label.mousePressEvent = self._muis_druk
        self.label.mouseMoveEvent = self._muis_beweeg
        self.label.mouseReleaseEvent = self._muis_los
        self.label.wheelEvent = self._zoom_wiel   # muiswiel = in-/uitzoomen
        # tracking aan: mouseMoveEvent vuurt ook zónder ingedrukte knop, nodig voor de
        # hover-tekst die het lichaamsdeel onder de cursor benoemt in de bewerk-modus.
        self.label.setMouseTracking(True)
        # Rechts-slepen pant (ook in de bewerk-modus, waar links bezet is). Zonder dit
        # propageert contextMenuEvent naar het QMainWindow, dat er zijn toolbar-/dock-menu
        # op opent — dan klapt er bij elke pan een menu open.
        self.label.setContextMenuPolicy(Qt.PreventContextMenu)

    def voeg_bedieningsknop(self, w):
        """Hangt een eigenaar-specifieke knop rechts in de toggles-rij (bv. '✏ Bewerken')."""
        self._rij_toggles.addWidget(w)

    def voeg_onderbalk(self, w):
        """Hangt een eigenaar-specifieke balk onderaan het paneel (bv. de editor-balk)."""
        self._hoofd.addWidget(w)

    # ── Laden / sluiten ──────────────────────────────────────────────────
    @property
    def crop_norm(self):
        return self._crop_norm

    @property
    def weergave_scaled(self):
        return self._weergave_scaled

    def laad(self, info, resultaten, video_pad):
        """
        Neemt een analyse in gebruik: capture heropenen, zoom resetten, besturing aan.

        Toont bewust nog géén frame — de caller roept als laatste `ga_naar(0)` aan. Alleen
        zo staan de tabel en de grafiek van de eigenaar al klaar wanneer `op_frame_getoond`
        voor het eerste frame vuurt.
        """
        if not video_pad:
            raise ValueError("VideoSpeler.laad() zonder videopad")
        self.video_info = info
        self.resultaten = resultaten
        self.video_pad = video_pad

        # Zoom resetten (geen zoom-lekkage tussen analyses). De stand van "Automatische zoom"
        # blijft wél staan: dat is een voorkeur van de kijker, geen eigenschap van de clip.
        self._zoom = self._zoom_eff = 1.0
        self._pan_cx = self._pan_cy = 0.5
        self._zoom_volg = True
        self._volg_forceren = True     # bij het eerste frame meteen op de schaatser richten
        self._pan_sleep = None
        self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        # Eén keer offline: welk kader heeft de schaatser per frame nodig? Kost een fractie
        # van een seconde en maakt de automatische zoom onafhankelijk van de afspeelrichting
        # (scrubben geeft exact dezelfde uitsnede als ernaartoe afspelen).
        self._kader = kader_reeks(resultaten, info.fps or 30.0)
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(100)
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom.setText("1.0×")
        self.chk_volg.blockSignals(True)
        self.chk_volg.setChecked(True)
        self.chk_volg.blockSignals(False)

        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_pad)
        self._weergave_pos = 0
        self._laatste_frame = None
        self.huidige_idx = -1

        # blockSignals: setRange klemt een te hoge sliderwaarde en zou anders valueChanged
        # vuren → een volle seek op een tabel die nog van de vórige analyse is.
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(resultaten) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.zet_besturing_actief(True)

    def herbereken_kader(self):
        """Het auto-zoom-kader opnieuw afleiden uit de huidige resultaten.

        Alleen nodig als er frames zijn bíjgekomen die eerst geen pose hadden (handmatig
        geplaatst skelet): `kader_reeks` opent bij een gat > KADER_GAT_S naar het volle
        beeld, dus zonder herberekening blijft juist het net gevulde frame uitgezoomd.
        Bewust niet na elke sleep-correctie — dan zou de zoom bij elke drop verspringen."""
        if not self.resultaten or self.video_info is None:
            return
        self._kader = kader_reeks(self.resultaten, self.video_info.fps or 30.0)

    def sluit(self):
        """Laat het videobestand los (nodig voordat de mediamap gewist kan worden) en
        maakt het paneel leeg."""
        self.pauzeer()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._laatste_frame = None
        self._weergave_pos = 0
        self._kader = None
        self.video_info = None
        self.resultaten = []
        self.huidige_idx = -1
        self.video_pad = None
        self.label.setText("Geen video geladen")
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.blockSignals(False)
        self.lbl_tijd.setText("t=0.00s  frame 0/0")
        self.zet_besturing_actief(False)

    def zet_besturing_actief(self, actief):
        for w in (self.btn_start, self.btn_frame_terug, self.btn_play,
                  self.btn_frame_verder, self.btn_eind, self.slider,
                  self.chk_volg, self.chk_auto, self.slider_zoom, self.btn_zoom_reset):
            w.setEnabled(actief)
        self._zet_handzoom_actief(actief)

    def _zet_handzoom_actief(self, actief=None):
        """Zet de handmatige zoomregelaars aan/uit: bepaalt het programma de zoom, dan zijn
        ze buiten werking (grijs) — dat is eerlijker dan een slider die niets doet."""
        if actief is None:
            actief = self.slider.isEnabled()
        aan = bool(actief) and not self._zoom_auto
        self.slider_zoom.setEnabled(aan)
        self.btn_zoom_reset.setEnabled(aan)

    # ── Navigeren + tekenen ──────────────────────────────────────────────
    def ga_naar(self, idx):
        if not self.resultaten:
            return
        idx = max(0, min(idx, len(self.resultaten) - 1))
        self._toon_frame(idx)

    def toon_huidig_frame(self):
        if self.huidige_idx >= 0:
            self._toon_frame(self.huidige_idx)

    def _scrub_gevraagd(self, idx):
        """Scrub-aanvraag van de slider (alleen in `snel_zoeken`-modus). Zie de opmerking bij
        de slider: alleen het láátste doel telt, de rest laten we vallen."""
        self._scrub_doel = idx
        if not self._scrub_timer.isActive():
            self._scrub_timer.start(0)

    def _scrub_tick(self):
        doel = self._scrub_doel
        if doel != self.huidige_idx:
            self.ga_naar(doel)
        if self._scrub_doel != doel:      # tijdens het tekenen verder gesleept
            self._scrub_timer.start(0)

    def _lees_frame_exact(self, idx):
        """
        Lees frame `idx` frame-exact, uitsluitend via sequentieel lezen.
        Een CAP_PROP_POS_FRAMES-seek is op VFR-video's (bv. iPhone-.MOV) níet
        frame-exact: het gedecodeerde beeld kan er enkele frames naast zitten
        terwijl OpenCV wél het gevraagde framenummer rapporteert. Het skelet
        (van het júiste frame) lijkt dan achter te lopen op het beeld — ook
        tijdens het afspelen erna, want de fout blijft constant. Daarom houden
        we zelf de cursor bij: vooruit spoelen met grab(), achteruit door de
        video te heropenen. Op de korte clips waar deze tool voor is, is dat
        ruim snel genoeg.

        Met `snel_zoeken` (het knipvenster van fase 8) mag er wél geseekt worden. Daar is
        het beeld een kijkje en geen meting: op een opname van een half uur (50.000 frames)
        zou terugspoelen vanaf frame 0 het doorbladeren onmogelijk maken, terwijl een
        fragmentgrens die je met het oog bepaalt best een paar frames (~0,1 s) mag schelen.
        Ná de seek wordt de eigen cursor hersteld, zodat vooruit afspelen weer klopt.

        **Elke stap achteruit seekt**, hoe klein ook: de sequentiële route spoelt daarvoor
        vanaf frame 0 opnieuw door, en één frame terug op frame 20.000 van een opname van
        23 minuten kostte zo bijna een minuut waarin de GUI volledig vastliep (gemeten op
        `00005.MTS`, 34.728 frames). Alleen kléíne sprongen vooruit blijven sequentieel —
        die zijn goedkoop (~3 ms per overgeslagen frame tegen ~80 ms voor een seek) en zo
        blijft frame-voor-frame stappen rond een fragmentgrens exact.
        """
        if self.cap is None:
            return None
        if self.snel_zoeken and (idx < self._weergave_pos
                                 or idx - self._weergave_pos > SEEK_DREMPEL_FRAMES):
            if self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                self._weergave_pos = idx      # cursor herstellen: vooruit lezen klopt weer
            # Mislukt de seek, dan valt de code hieronder terug op sequentieel spoelen.
        if idx < self._weergave_pos:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_pad)
            self._weergave_pos = 0
        while self._weergave_pos < idx:
            if not self.cap.grab():
                return None
            self._weergave_pos += 1
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._weergave_pos += 1
        return frame

    def _meld_leesfout(self, idx):
        """
        Frame `idx` kon niet gelezen worden (voorbij het einde, of een decodefout).
        Stilzwijgend terugkeren is misleidend: beeld, slider, tabelmarkering en tijd
        blijven dan op het vórige frame staan terwijl de gebruiker denkt te zijn
        gesprongen. Dus: afspelen stoppen, de besturing terugzetten op het laatst
        geldige frame en het melden.
        """
        self.pauzeer()
        if self.huidige_idx >= 0 and self._laatste_frame is not None:
            self._toon_frame(self.huidige_idx)       # zet slider/tabel weer in de pas
            self.lbl_tijd.setText(
                f"Frame {idx} kon niet gelezen worden — beeld staat nog op {self.huidige_idx}")
        else:
            self.lbl_tijd.setText(f"Frame {idx} kon niet gelezen worden")

    def _toon_frame(self, idx):
        if idx == self.huidige_idx and self._laatste_frame is not None:
            frame = self._laatste_frame.copy()      # alleen overlay opnieuw tekenen
            nieuw_frame = False
        else:
            frame = self._lees_frame_exact(idx)
            if frame is None:
                self._meld_leesfout(idx)
                return
            self._laatste_frame = frame
            frame = frame.copy()
            nieuw_frame = True
        self.huidige_idx = idx

        resultaat = self.resultaten[idx]
        # De automatische zoom verschilt per frame; label (en de uitgeschakelde slider als
        # aflezing) tonen daarom de toegepaste factor. Eerst rekenen, dán pas het volgen:
        # automatisch kan er ook zonder handmatige zoom een uitsnede zijn, en die moet
        # mee-centreren.
        self._zoom_eff = self._bereken_zoom_eff(idx)
        self.lbl_zoom.setText(f"{self._zoom_eff:.1f}×")
        if self._zoom_auto:
            self.slider_zoom.blockSignals(True)
            self.slider_zoom.setValue(int(round(min(ZOOM_MAX, self._zoom_eff) * 100)))
            self.slider_zoom.blockSignals(False)
        # Auto-volgen: centreer de zoom-uitsnede op de schaatser, maar alleen bij een echte
        # framewissel en niet tijdens een handle-sleep — anders verspringt de uitsnede onder
        # de cursor bij het verslepen of het togglen van een laag.
        if ((nieuw_frame or self._volg_forceren) and self._zoom_eff > 1.0
                and self._zoom_volg and not self.volgen_bevroren):
            # Automatisch: op het kader-middelpunt, want de zoom is op datzelfde kader
            # gemeten — daar staat de schaatser dus gegarandeerd compleet in beeld.
            # Handmatig: op de romp, die rustiger beweegt dan de armen en benen.
            kader = self._kader_op(idx) if self._zoom_auto else None
            c = (kader[:2] if kader is not None
                 else torso_centroid(resultaat.lm) if resultaat.pose_gevonden else None)
            if c is not None:
                self._pan_cx, self._pan_cy = c   # klemmen gebeurt in _toon_pixmap
        self._volg_forceren = False
        if self.toon_overlay:
            teken_overlay_op_frame(
                frame, resultaat, self.video_info.fps,
                toon_skelet=self.chk_skelet.isChecked(),
                toon_afzetbeen=self.chk_afzetbeen.isChecked(),
                toon_hud=self.chk_hud.isChecked(),
            )
        self._toon_pixmap(frame)

        if idx != self.slider.value():
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        self.lbl_tijd.setText(f"t={resultaat.tijd:.2f}s  frame {idx}/{len(self.resultaten) - 1}")
        if self.op_frame_getoond is not None:
            self.op_frame_getoond(idx)

    def _toon_pixmap(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        # Inzoomen = een uitsnede rond het pan-middelpunt opschalen. De uitsnede houdt
        # dezelfde beeldverhouding als het frame, zodat de KeepAspectRatio-letterbox
        # (en dus de coördinaat-omrekening van de editor) onveranderd blijft.
        z = max(1.0, self._zoom_eff)
        if z > 1.0:
            cw, ch = w / z, h / z
            x0 = min(max(self._pan_cx * w - cw / 2, 0.0), w - cw)   # crop binnen het frame klemmen
            y0 = min(max(self._pan_cy * h - ch / 2, 0.0), h - ch)
            ix0, iy0 = int(round(x0)), int(round(y0))
            icw = min(int(round(cw)), w - ix0)
            ich = min(int(round(ch)), h - iy0)
            # .copy() maakt de slice C-contigu (nodig voor de QImage-stride) en laat
            # _laatste_frame gegarandeerd op volle resolutie staan.
            frame_bgr = frame_bgr[iy0:iy0 + ich, ix0:ix0 + icw].copy()
            self._crop_norm = (ix0 / w, iy0 / h, icw / w, ich / h)
            h, w = frame_bgr.shape[:2]
        else:
            self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Volgorde is niet cosmetisch: de overlay-tekenaar (skelet-editor) rekent via
        # norm_naar_widget met _weergave_scaled, dus die moet al bijgewerkt zijn.
        self._weergave_scaled = pixmap.size()
        if self.overlay_tekenaar is not None:
            self.overlay_tekenaar(pixmap)
        self.label.setPixmap(pixmap)

    # ── Inzoomen op de schaatser ─────────────────────────────────────────────
    def _kader_op(self, idx):
        """Kader `(midden_x, midden_y, straal)` van frame `idx` (offline reeks), of None."""
        if self._kader is None or not (0 <= idx < len(self._kader)):
            return None
        return self._kader[idx]

    def _zoom_plafond(self):
        """Hoe ver de automaat mag inzoomen. Bij zoom 1× past het frame met factor `s` op het
        paneel; bij zoom z wordt dat `z·s` schermpixels per videopixel. Boven
        `KADER_MAX_VERGROTING` wordt dat zichtbaar pap, dus daar houdt de automaat op."""
        info = self.video_info
        if info is None or not info.w or not info.h:
            return ZOOM_AUTO_MAX
        s = min(self.label.width() / info.w, self.label.height() / info.h)
        if s <= 0:
            return ZOOM_AUTO_MAX
        return min(ZOOM_AUTO_MAX, max(1.0, KADER_MAX_VERGROTING / s))

    def _bereken_zoom_eff(self, idx):
        """De zoom die op frame `idx` daadwerkelijk toegepast wordt.

        Handmatig is dat simpelweg de ingestelde zoom. Automatisch bepaalt het programma hem
        uit de schaatser zelf: de uitsnede is (genormaliseerd) 0.5/zoom groot rondom het
        kader-middelpunt, dus vullen we die met de ruimte die de schaatser nodig heeft plus
        `KADER_MARGE` lucht. Verder uitzoomen dan het volledige beeld kan niet, dus dichtbij
        blijft de zoom gewoon op 1× staan."""
        z = min(ZOOM_MAX, max(1.0, self._zoom))
        if not self._zoom_auto:
            return z
        kader = self._kader_op(idx)
        if kader is None or not kader[2]:
            return z            # geen bruikbare pose: laat de handmatige zoom staan
        return min(self._zoom_plafond(), max(1.0, 0.5 / (kader[2] * (1.0 + KADER_MARGE))))

    def _zet_zoom(self, z):
        """Centrale zoom-setter: klemt, werkt slider+label bij (zonder signaal-lus) en
        hertekent het huidige frame goedkoop (geen herlezen van de video)."""
        z = min(ZOOM_MAX, max(1.0, float(z)))
        self._zoom = z
        if z <= 1.0:
            self._pan_cx = self._pan_cy = 0.5
        self._volg_forceren = True   # bewuste zoom-actie: meteen op de schaatser richten
        self.lbl_zoom.setText(f"{z:.1f}×")     # _toon_frame zet er zo de effectieve zoom in
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(int(round(z * 100)))
        self.slider_zoom.blockSignals(False)
        self.toon_huidig_frame()

    def _zoom_wiel(self, event):
        """Muiswiel boven de video: in-/uitzoomen. Auto-volgen blijft aan, dus de uitsnede
        blijft op de schaatser (geen zoom-naar-cursor, dat zou met 'volg schaatser' vechten)."""
        delta = event.angleDelta().y()
        if not self.resultaten or delta == 0:
            # Niets te zoomen: het wiel teruggeven aan Qt, anders slikt het videolabel
            # de scroll en gebeurt er buiten een geladen analyse helemaal niets.
            QLabel.wheelEvent(self.label, event)
            return
        if self._zoom_auto:
            # Aan het wiel draaien = de zoom overnemen, net zoals handmatig slepen het
            # auto-volgen overneemt. `_zet_zoom_auto` neemt de huidige stand over, dus het
            # beeld springt niet — er wordt vanaf hier alleen niet meer bijgestuurd.
            self.chk_auto.setChecked(False)
        factor = ZOOM_STAP if delta > 0 else 1.0 / ZOOM_STAP
        self._zet_zoom(self._zoom * factor)
        event.accept()

    def _zet_zoom_volg(self, aan):
        self._zoom_volg = bool(aan)
        self._volg_forceren = True
        self.toon_huidig_frame()

    def _zet_zoom_auto(self, aan):
        """Zet de automatische zoom aan/uit. Bij uitzetten wordt de laatst getoonde zoom de
        handmatige stand, zodat het beeld op dat moment niet verspringt — behalve boven
        `ZOOM_MAX`, waar de handmatige regelaar nu eenmaal ophoudt."""
        self._zoom_auto = bool(aan)
        self._zet_handzoom_actief()
        if not self._zoom_auto:
            self._zet_zoom(self._zoom_eff)     # neemt over, hertekent en herstelt de slider
            return
        self._volg_forceren = True
        self.toon_huidig_frame()

    def _zoom_reset(self):
        """Terug naar passend beeld en auto-volgen weer aan."""
        self._pan_cx = self._pan_cy = 0.5
        self._zoom_volg = True
        self.chk_volg.blockSignals(True)
        self.chk_volg.setChecked(True)
        self.chk_volg.blockSignals(False)
        self._zet_zoom(1.0)

    # ── Coördinaat-omrekening (letterbox + zoom-uitsnede) ────────────────────
    def widget_naar_norm(self, pos):
        """Muispositie op het videolabel → genormaliseerde (x, y) in het frame (0–1).
        Buiten het getekende beeld kan het resultaat buiten [0,1] liggen (caller checkt)."""
        if self._weergave_scaled is None:
            return None
        sw, sh = self._weergave_scaled.width(), self._weergave_scaled.height()
        if sw <= 0 or sh <= 0:
            return None
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        fx = (pos.x() - offx) / sw          # fractie binnen de getoonde uitsnede
        fy = (pos.y() - offy) / sh
        x0n, y0n, wn, hn = self._crop_norm  # bij zoom==1 is dit (0,0,1,1) → oude formule
        return (x0n + fx * wn, y0n + fy * hn)

    def norm_naar_widget(self, nx, ny):
        """Inverse: genormaliseerde (x, y) → positie op het videolabel (voor hittesten)."""
        sw, sh = self._weergave_scaled.width(), self._weergave_scaled.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        x0n, y0n, wn, hn = self._crop_norm  # bij zoom==1 is dit (0,0,1,1) → oude formule
        return QPointF(offx + (nx - x0n) / wn * sw, offy + (ny - y0n) / hn * sh)

    # ── Muis: pannen doet de speler zelf, de rest gaat naar de eigenaar ──────
    def _muis_druk(self, event):
        # Deze takken moeten bovenaan blijven: buiten de bewerk-modus is links-slepen bedoeld
        # om het ingezoomde beeld te verschuiven (pannen), in de bewerk-modus is het = punt
        # verplaatsen of plaatsen (dat handelt de eigenaar af). Rechts-slepen pant áltijd —
        # dat is de enige manier om te schuiven terwijl de linkerknop bezet is.
        if self._zoom_eff > 1.0 and (event.button() == Qt.RightButton
                                     or (not self.bewerk_modus
                                         and event.button() == Qt.LeftButton)):
            self._pan_sleep = event.position()
            return
        if self.op_muis_druk is not None:
            self.op_muis_druk(event)

    def _muis_beweeg(self, event):
        if self._pan_sleep is not None:
            if self._weergave_scaled is None:
                return
            d = event.position() - self._pan_sleep
            self._pan_sleep = event.position()
            sw, sh = self._weergave_scaled.width(), self._weergave_scaled.height()
            _, _, wn, hn = self._crop_norm
            if sw > 0 and sh > 0:
                half = 0.5 / max(1.0, self._zoom_eff)
                # slepen naar rechts toont de linkerkant → uitsnede-midden schuift mee
                self._pan_cx = min(1.0 - half, max(half, self._pan_cx - d.x() / sw * wn))
                self._pan_cy = min(1.0 - half, max(half, self._pan_cy - d.y() / sh * hn))
            self._zoom_volg = False
            self.chk_volg.blockSignals(True)
            self.chk_volg.setChecked(False)
            self.chk_volg.blockSignals(False)
            self.toon_huidig_frame()
            return
        if self.op_muis_beweeg is not None:
            self.op_muis_beweeg(event)

    def _muis_los(self, event):
        if self._pan_sleep is not None:
            self._pan_sleep = None
            return
        if self.op_muis_los is not None:
            self.op_muis_los(event)

    # ── Afspelen ─────────────────────────────────────────────────────────
    def _speel_stap(self):
        """Hoeveel frames er per timer-tik opgeschoven wordt.

        Tot en met 1× is dat er één en regelt de timer het tempo. Sneller dan echte snelheid
        kan geen decoder bijbenen (op een 1080p-opname ~9 ms per frame, dus 4× = 100 fps
        halen we niet), dus daarboven slaan we frames óver: bij 4× vier frames per tik op het
        normale fps-interval. Overgeslagen frames kosten alleen een `grab()` (~3 ms) i.p.v.
        een volledige decode."""
        factor = self.combo_snelheid.currentData() or 1.0
        return max(1, int(round(factor))) if factor > 1.0 else 1

    def _speel_interval_ms(self):
        """Timer-interval per tik, geschaald met de gekozen afspeelsnelheid — en met de
        stapgrootte erin verrekend, zodat 4× met stap 4 gewoon op het fps-tempo loopt."""
        factor = self.combo_snelheid.currentData() or 1.0
        tikken_per_s = (self.video_info.fps or 30.0) * factor / self._speel_stap()
        return max(1, int(1000 / tikken_per_s))

    def _zet_snelheid(self, _idx=None):
        # Draait de video al, herstart de timer meteen met het nieuwe tempo.
        if self.speeltimer.isActive():
            self.speeltimer.start(self._speel_interval_ms())

    def speelt(self):
        return self.speeltimer.isActive()

    def speel(self):
        if not self.resultaten or self.speeltimer.isActive():
            return
        if self.huidige_idx >= len(self.resultaten) - 1:
            self.ga_naar(0)
        self.speeltimer.start(self._speel_interval_ms())
        self.btn_play.setText("⏸")

    def pauzeer(self):
        if self.speeltimer.isActive():
            self.speeltimer.stop()
        self.btn_play.setText("▶")

    def _toggle_afspelen(self):
        if self.speeltimer.isActive():
            self.pauzeer()
        else:
            self.speel()

    def _speel_tick(self):
        volgende = self.huidige_idx + self._speel_stap()
        if volgende >= len(self.resultaten):
            # Bij een stap > 1 zou het einde anders overgeslagen worden; nog even het
            # laatste frame tonen en dan pas stoppen.
            if self.huidige_idx < len(self.resultaten) - 1:
                self._toon_frame(len(self.resultaten) - 1)
            self.pauzeer()
            return
        self._toon_frame(volgende)


class VergelijkKant(QWidget):
    """
    Eén kant van de vergelijkpagina: kop met de gekozen analyse, een eigen VideoSpeler,
    een sync-punt (startframe voor 'Start alles') en een minimale afzettabel.

    De tabel toont bewust weinig — nummer, been en hoek — maar markeert wél onvolledige
    afzetten: naast elkaar gezet nodigt de weergave uit om twee hoeken te vergelijken, en
    juist een onvolledige afzet is systematisch te steil.
    """

    def __init__(self, naam, kies_callback, parent=None):
        super().__init__(parent)
        self.naam = naam
        self.analyse_id = None
        self.events = []
        self.sync_frame = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        kop = QHBoxLayout()
        self.lbl_titel = QLabel(f"{naam} — nog geen analyse gekozen")
        self.lbl_titel.setStyleSheet("font-weight: bold; padding: 2px;")
        kop.addWidget(self.lbl_titel, stretch=1)
        self.btn_kies = QPushButton("Kies analyse...")
        self.btn_kies.clicked.connect(kies_callback)
        kop.addWidget(self.btn_kies)
        # Leegmaken wordt van buiten bedraad (zie _bouw_vergelijkpagina): de masterklok
        # moet eerst los, en die kent de kant niet andersom.
        self.btn_leeg = QPushButton("✕")
        self.btn_leeg.setToolTip("Deze kant leegmaken.")
        self.btn_leeg.setEnabled(False)
        kop.addWidget(self.btn_leeg)
        v.addLayout(kop)

        # Zonder eigen snelheidsregelaar: de gedeelde regelaar onderaan de vergelijkpagina
        # stuurt beide kanten, zodat twee video's nooit op verschillend tempo lopen.
        self.speler = VideoSpeler(min_grootte=(320, 200), toon_snelheid=False)
        # De HUD wordt op vaste vol-frame-posities getekend en is in een half paneel
        # onleesbaar; per kant weer aan te zetten.
        self.speler.chk_hud.setChecked(False)
        v.addWidget(self.speler, stretch=1)

        rij_sync = QHBoxLayout()
        self.btn_sync = QPushButton("⚑ Zet sync hier")
        self.btn_sync.setToolTip(
            "Legt het huidige frame vast als startpunt voor 'Start alles', zodat beide\n"
            "video's op dezelfde fase van de slag beginnen.\n"
            "Let op: sync-punten gelden voor deze sessie en worden niet opgeslagen.")
        self.btn_sync.clicked.connect(self._zet_sync)
        rij_sync.addWidget(self.btn_sync)
        self.lbl_sync = QLabel("sync: frame 0")
        self.lbl_sync.setStyleSheet("color: #888;")
        rij_sync.addWidget(self.lbl_sync)
        rij_sync.addStretch(1)
        v.addLayout(rij_sync)

        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["#", "Been", "Hoek (°)"])
        self.tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.setMaximumHeight(180)
        self.tabel.cellClicked.connect(self._klik_op_rij)
        v.addWidget(self.tabel)

        self.btn_sync.setEnabled(False)

    # ── Vullen / legen ───────────────────────────────────────────────────
    def toon(self, analyse_id, schaatser_naam, data):
        """Neemt een geladen analyse (dict uit MainWindow._laad_analyse_data) in gebruik.

        Dezelfde analyse opnieuw laden (bv. na een edit) houdt het sync-punt: dat hoort bij
        de video, niet bij het laden. Een ándere analyse begint weer op frame 0."""
        zelfde = analyse_id == self.analyse_id
        self.analyse_id = analyse_id
        self.events = data["events"]
        self.sync_frame = (min(self.sync_frame, max(0, len(data["resultaten"]) - 1))
                           if zelfde else 0)
        self.lbl_titel.setText(f"{schaatser_naam} — {data['titel']}")
        self.speler.laad(data["info"], data["resultaten"], data["video_pad"])
        self._vul_tabel()
        self.btn_kies.setText("Wisselen...")
        self.btn_sync.setEnabled(True)
        self.btn_leeg.setEnabled(True)
        self.speler.ga_naar(self.sync_frame)
        self._toon_sync_label()

    def leeg(self):
        """Laat de video los (nodig voordat de mediamap gewist kan worden)."""
        self.speler.sluit()
        self.analyse_id = None
        self.events = []
        self.sync_frame = 0
        self.tabel.setRowCount(0)
        self.lbl_titel.setText(f"{self.naam} — nog geen analyse gekozen")
        self.lbl_sync.setText("sync: frame 0")
        self.btn_kies.setText("Kies analyse...")
        self.btn_sync.setEnabled(False)
        self.btn_leeg.setEnabled(False)

    def heeft_analyse(self):
        return self.analyse_id is not None and bool(self.speler.resultaten)

    # ── Sync-punt ────────────────────────────────────────────────────────
    def _zet_sync(self):
        if not self.heeft_analyse():
            return
        self.sync_frame = max(0, self.speler.huidige_idx)
        self._toon_sync_label()

    def _toon_sync_label(self):
        if not self.heeft_analyse():
            self.lbl_sync.setText("sync: frame 0")
            return
        tijd = self.speler.resultaten[self.sync_frame].tijd
        self.lbl_sync.setText(f"sync: frame {self.sync_frame}  (t={tijd:.2f}s)")

    def naar_sync(self):
        if self.heeft_analyse():
            self.speler.ga_naar(self.sync_frame)

    # ── Tabel ────────────────────────────────────────────────────────────
    def _vul_tabel(self):
        self.tabel.setRowCount(len(self.events))
        onvolledig_kleur = QColor(70, 70, 70)
        for i, ev in enumerate(self.events):
            waarden = [str(i + 1), ev.been.capitalize(), f"{ev.hoek:.1f}"]
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setTextAlignment(Qt.AlignCenter)
                if ev.onvolledig:
                    item.setBackground(onvolledig_kleur)
                    item.setToolTip(
                        f"Onvolledige afzet ({ev.onvolledig}) — de push is niet "
                        "afgemaakt, dus deze hoek is te steil en niet vergelijkbaar.")
                self.tabel.setItem(i, kolom, item)

    def _klik_op_rij(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self.speler.ga_naar(self.events[rij].start_frame)


# ── Fragmenten knippen uit een lange opname (fase 8) ───────────────────────────

def _tijd_tekst(frames, fps):
    """Framenummer → "m:ss" (of "h:mm:ss" op een lange opname)."""
    sec = int(round(frames / (fps or 30.0)))
    if sec >= 3600:
        return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60}:{sec % 60:02d}"


def _lees_tijd(tekst, fps):
    """"m:ss", "h:mm:ss" of een aantal seconden → framenummer; None als het niet leest."""
    delen = (tekst or "").strip().replace(",", ".").split(":")
    try:
        waarden = [float(d) for d in delen if d != ""]
    except ValueError:
        return None
    if not waarden or len(waarden) != len(delen):
        return None
    sec = 0.0
    for w in waarden:
        sec = sec * 60 + w
    return int(round(sec * (fps or 30.0)))


class FragmentBalk(QWidget):
    """
    De balk onder de tijdlijn van het knipvenster: per gemarkeerd stuk een gekleurd blok op
    zijn plek in de opname. Dit is de enige echt nieuwe teken-code van fase 8.

    Kleuren: **groen** = zojuist gemarkeerd, **grijs** = in een eerdere sessie al geanalyseerd
    (uit `bron_fragmenten`), **oranje** = het lopende, nog niet gestopte fragment. Overlappen
    twee gemarkeerde stukken elkaar, dan wordt het overlappende deel **rood** gearceerd —
    zichtbaar gemaakt, maar er wordt níets automatisch samengevoegd of ingekort. De trainer
    past het zelf aan of laat het zoals het is (het knippen is volledig handmatig).
    """
    KLIK = Signal(int)        # index in `fragmenten` van het aangeklikte blok (−1 = ernaast)

    HOOGTE = 26
    KLEUR_ACHTERGROND = QColor(45, 45, 45)
    KLEUR_GEDAAN = QColor(120, 120, 120)
    KLEUR_NIEUW = QColor(60, 160, 80)
    KLEUR_LOPEND = QColor(220, 150, 40)
    KLEUR_OVERLAP = QColor(200, 60, 60)
    KLEUR_CURSOR = QColor(240, 240, 240)
    KLEUR_SELECTIE = QColor(255, 255, 255)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HOOGTE)
        self.setToolTip(
            "Gemarkeerde stukken van deze opname.\n"
            "Groen = zojuist gemarkeerd · grijs = al geanalyseerd · oranje = loopt nog · "
            "rood = overlap.\nKlik op een blok om erheen te springen.")
        self.totaal = 1
        self.fragmenten = []      # [(start, eind)] — zojuist gemarkeerd
        self.gedaan = []          # [(start, eind, label)] — uit een eerdere sessie
        self.lopend = None        # startframe van het nog niet gestopte fragment
        self.cursor = 0
        self.selectie = -1

    def zet(self, totaal=None, fragmenten=None, gedaan=None, lopend=..., cursor=None,
            selectie=None):
        """Alles wat de balk toont in één aanroep bijwerken (en hertekenen)."""
        if totaal is not None:
            self.totaal = max(1, int(totaal))
        if fragmenten is not None:
            self.fragmenten = list(fragmenten)
        if gedaan is not None:
            self.gedaan = list(gedaan)
        if lopend is not ...:
            self.lopend = lopend
        if cursor is not None:
            self.cursor = int(cursor)
        if selectie is not None:
            self.selectie = int(selectie)
        self.update()

    def _x(self, frame):
        return int(round(frame / self.totaal * max(1, self.width() - 1)))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.KLEUR_ACHTERGROND)
        h = self.height()

        for start, eind, _label in self.gedaan:
            self._blok(p, start, eind, self.KLEUR_GEDAAN, 4, h - 8)
        for i, (start, eind) in enumerate(self.fragmenten):
            self._blok(p, start, eind, self.KLEUR_NIEUW, 2, h - 4)
            if i == self.selectie:
                p.setPen(QPen(self.KLEUR_SELECTIE, 2))
                p.setBrush(Qt.NoBrush)
                x0, x1 = self._x(start), self._x(eind)
                p.drawRect(QRect(x0, 1, max(2, x1 - x0), h - 3))
        # Overlap ná de blokken, zodat de arcering er bovenop ligt.
        for i, (a0, a1) in enumerate(self.fragmenten):
            for b0, b1 in self.fragmenten[i + 1:]:
                s, e = max(a0, b0), min(a1, b1)
                if s <= e:
                    self._blok(p, s, e, self.KLEUR_OVERLAP, 2, h - 4)
        if self.lopend is not None:
            self._blok(p, self.lopend, max(self.lopend, self.cursor),
                       self.KLEUR_LOPEND, 2, h - 4)

        p.setPen(QPen(self.KLEUR_CURSOR, 1))
        x = self._x(self.cursor)
        p.drawLine(x, 0, x, h)

    def _blok(self, p, start, eind, kleur, y, hoogte):
        x0, x1 = self._x(start), self._x(eind)
        p.fillRect(QRect(x0, y, max(2, x1 - x0), hoogte), kleur)

    def mousePressEvent(self, event):
        frame = int(event.position().x() / max(1, self.width() - 1) * self.totaal)
        for i, (start, eind) in enumerate(self.fragmenten):
            if start <= frame <= eind:
                self.KLIK.emit(i)
                return
        self.KLIK.emit(-1)


class FragmentKiezer(QDialog):
    """
    Het knipvenster (ROADMAP fase 8): een opname van een half uur doorlopen en de bruikbare
    stukken markeren. Levert een lijst `(start_frame, eind_frame, naam)`; het knippen zelf
    doet `schaats_analyse.knip_fragmenten`, en daarna gaan de clips als voorgevulde rijen de
    bestaande batch-flow in.

    **Dit is een knipprogramma en het knippen is volledig handmatig.** De app bepaalt niets
    zelf: niet wanneer de schaatser in beeld is, niet waar een stuk begint of eindigt, en er
    komt geen seconde marge bij of af. De trainer kijkt, drukt op start en stop, en dát zijn
    de grenzen.

    Hergebruikt `VideoSpeler` voor het afspelen (transportknoppen, snelheidcombo om op 4×
    door een half uur te scannen, zoom), met twee afwijkingen van de weergavepagina:
    `snel_zoeken=True` — achteruit scrubben mag hier seeken, want het beeld is een kijkje en
    geen meting — en `toon_overlay=False`, want er is geen analyse om te tekenen.
    """

    def __init__(self, bron_pad, info, gedaan=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Fragmenten knippen — {os.path.basename(bron_pad)}")
        self.info = info
        self.fps = info.fps or 30.0
        self._fragmenten = []          # [(start, eind)] in markeervolgorde
        self._start_open = None        # startframe van het lopende fragment
        self._stam = os.path.splitext(os.path.basename(bron_pad))[0]

        # Krap houden: dit venster moet op een laptopscherm (1280×800, werkgebied 752 px)
        # in z'n geheel passen — anders zakt de knoppenbalk onderlangs weg en is "Klaar"
        # onvindbaar. Elk minimum hieronder is daarom bewust laag; het beeld rekt vanzelf
        # mee als er ruimte is.
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        uitleg = QLabel("Markeer de bruikbare stukken: <b>Start</b> (S) — <b>Stop</b> (E). "
                        "Er wordt exact op die frames geknipt.")
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)

        self.speler = VideoSpeler(min_grootte=(400, 200), snel_zoeken=True,
                                  toon_overlay=False)
        self.speler.op_frame_getoond = self._frame_getoond
        v.addWidget(self.speler, stretch=1)

        # De balk hangt ín de speler (onder de scrub-slider), zodat hij dezelfde breedte
        # als de tijdlijn heeft en er samen mee opschuift.
        self.balk = FragmentBalk()
        self.balk.KLIK.connect(self._klik_op_balk)
        self.speler.voeg_onderbalk(self.balk)

        # Navigatiehulp: op een half uur is de slider te grof om een afzet terug te vinden.
        rij_nav = WrapBalk()
        for label, sec in (("−1 min", -60), ("−10 s", -10), ("−1 s", -1),
                           ("+1 s", 1), ("+10 s", 10), ("+1 min", 60)):
            knop = QPushButton(label)
            knop.setMaximumWidth(72)
            knop.clicked.connect(lambda _=False, s=sec: self._spring(s))
            rij_nav.addWidget(knop)
        rij_nav.addWidget(QLabel("Ga naar"))
        self.veld_tijd = QLineEdit()
        self.veld_tijd.setPlaceholderText("m:ss")
        self.veld_tijd.setFixedWidth(80)
        self.veld_tijd.returnPressed.connect(self._ga_naar_tijd)
        rij_nav.addWidget(self.veld_tijd)
        knop_ga = QPushButton("Ga")
        knop_ga.setMaximumWidth(48)
        knop_ga.clicked.connect(self._ga_naar_tijd)
        rij_nav.addWidget(knop_ga)
        rij_nav.addStretch(1)
        v.addWidget(rij_nav)

        # Markeerknoppen + de lijst met gemarkeerde stukken.
        onder = QHBoxLayout()
        links = QVBoxLayout()
        self.btn_start = QPushButton("● Start bruikbaar beeld  (S)")
        self.btn_start.clicked.connect(self._start_fragment)
        links.addWidget(self.btn_start)
        self.btn_stop = QPushButton("■ Stop bruikbaar beeld  (E)")
        self.btn_stop.clicked.connect(self._stop_fragment)
        links.addWidget(self.btn_stop)
        self.lbl_lopend = QLabel("")
        self.lbl_lopend.setStyleSheet("color: #d89828;")
        links.addWidget(self.lbl_lopend)
        links.addStretch(1)
        onder.addLayout(links)

        rechts = QVBoxLayout()
        rechts.addWidget(QLabel("Gemarkeerde fragmenten  (klik = erheen springen)"))
        self.tabel = QTableWidget(0, 4)
        self.tabel.setHorizontalHeaderLabels(["#", "Van – tot", "Duur", ""])
        kop = self.tabel.horizontalHeader()
        kop.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(1, QHeaderView.Stretch)
        kop.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.setMinimumHeight(70)
        self.tabel.cellClicked.connect(self._klik_op_rij)
        rechts.addWidget(self.tabel, stretch=1)
        onder.addLayout(rechts, stretch=1)
        v.addLayout(onder)

        if gedaan:
            info_gedaan = QLabel(
                f"Grijs in de balk: {len(gedaan)} stuk(ken) van deze opname zijn al "
                f"geanalyseerd.")
            info_gedaan.setStyleSheet("color: #888;")
            v.addWidget(info_gedaan)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self._bevestig)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)

        QShortcut(QKeySequence("S"), self, activated=self._start_fragment)
        QShortcut(QKeySequence("E"), self, activated=self._stop_fragment)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._verwijder_selectie)

        # Lege FrameResultaat-lijst: de speler wil er één (sliderlengte, tijdlabel), maar er
        # is nog niets geanalyseerd. `kader_reeks` geeft dan None en de zoom blijft handmatig.
        resultaten = [FrameResultaat(i, i / self.fps) for i in range(max(1, info.totaal))]
        self.speler.laad(info, resultaten, bron_pad)
        self.balk.zet(totaal=len(resultaten),
                      gedaan=[(f["start_frame"], f["eind_frame"],
                               f["titel"] or "") for f in gedaan])
        self.speler.ga_naar(0)
        self._werk_bij()

        # Pas nadat alles er staat: dan kan de klem in zet_venstergrootte tegen een
        # definitieve layout aan rekenen (en een `resize()` wordt genegeerd zodra de inhoud
        # groter is dan gevraagd — vandaar dat alle minima hierboven laag zijn).
        zet_venstergrootte(self, 1100, 720)

    # ── Markeren ─────────────────────────────────────────────────────────
    def _start_fragment(self):
        if self._start_open is not None:
            return
        self._start_open = self.speler.huidige_idx
        self._werk_bij()

    def _stop_fragment(self):
        """Sluit het lopende fragment af. Een stop vóór de start is een vergissing, geen
        fragment: dan liever niets vastleggen dan een omgekeerd stuk knippen."""
        if self._start_open is None:
            return
        eind = self.speler.huidige_idx
        if eind < self._start_open:
            QMessageBox.information(
                self, "Stop ligt vóór de start",
                "Het eind van een fragment ligt vóór het begin. Spoel verder door en druk "
                "opnieuw op Stop, of begin dit fragment opnieuw.")
            return
        self._fragmenten.append((self._start_open, eind))
        self._start_open = None
        self._werk_bij(selectie=len(self._fragmenten) - 1)

    def _verwijder_selectie(self):
        rij = self.tabel.currentRow()
        if 0 <= rij < len(self._fragmenten):
            self._fragmenten.pop(rij)
            self._werk_bij()

    def _klik_op_rij(self, rij, _kolom=0):
        if 0 <= rij < len(self._fragmenten):
            self.speler.ga_naar(self._fragmenten[rij][0])
            self.balk.zet(selectie=rij)

    def _klik_op_balk(self, index):
        if index < 0:
            return
        self.tabel.selectRow(index)
        self._klik_op_rij(index)

    # ── Navigeren ────────────────────────────────────────────────────────
    def _spring(self, seconden):
        self.speler.ga_naar(self.speler.huidige_idx + int(round(seconden * self.fps)))

    def _ga_naar_tijd(self):
        frame = _lees_tijd(self.veld_tijd.text(), self.fps)
        if frame is None:
            QMessageBox.information(self, "Tijd", "Gebruik m:ss (bijvoorbeeld 12:30).")
            return
        self.speler.ga_naar(frame)

    def _frame_getoond(self, idx):
        self.balk.zet(cursor=idx, lopend=self._start_open)
        if self._start_open is not None:
            self.lbl_lopend.setText(
                f"Loopt vanaf {_tijd_tekst(self._start_open, self.fps)} — "
                f"nu {_tijd_tekst(idx, self.fps)}")

    # ── Weergave bijwerken ───────────────────────────────────────────────
    def _werk_bij(self, selectie=None):
        self.btn_start.setEnabled(self._start_open is None)
        self.btn_stop.setEnabled(self._start_open is not None)
        self.lbl_lopend.setText(
            "" if self._start_open is None
            else f"Loopt vanaf {_tijd_tekst(self._start_open, self.fps)}")

        self.tabel.setRowCount(len(self._fragmenten))
        for i, (start, eind) in enumerate(self._fragmenten):
            duur = (eind - start + 1) / self.fps
            waarden = [str(i + 1),
                       f"{_tijd_tekst(start, self.fps)} – {_tijd_tekst(eind, self.fps)}",
                       f"{duur:.1f}s".replace(".", ",")]
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setToolTip(f"frame {start}–{eind}")
                self.tabel.setItem(i, kolom, item)
            knop = QPushButton("✕")
            knop.setMaximumWidth(30)
            knop.setToolTip("Dit fragment verwijderen")
            knop.clicked.connect(lambda _=False, r=i: self._verwijder(r))
            self.tabel.setCellWidget(i, 3, knop)
        if selectie is not None and 0 <= selectie < len(self._fragmenten):
            self.tabel.selectRow(selectie)

        self.balk.zet(fragmenten=self._fragmenten, lopend=self._start_open,
                      selectie=selectie if selectie is not None else -1)
        n = len(self._fragmenten)
        self._ok.setText(f"Klaar — analyseer {n} fragment{'en' if n != 1 else ''}"
                         if n else "Klaar")
        self._ok.setEnabled(n > 0)

    def _verwijder(self, rij):
        if 0 <= rij < len(self._fragmenten):
            self._fragmenten.pop(rij)
            self._werk_bij()

    # ── Afsluiten ────────────────────────────────────────────────────────
    def _bevestig(self):
        if self._start_open is not None:
            antwoord = QMessageBox.question(
                self, "Fragment loopt nog",
                "Er staat nog een fragment open (wel Start, geen Stop). Dat stuk wordt niet "
                "geknipt.\n\nToch doorgaan met de fragmenten die er staan?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if antwoord != QMessageBox.Yes:
                return
        self.accept()

    @property
    def fragmenten(self):
        """[(start_frame, eind_frame, naam)] op startframe gesorteerd. De naam wordt zowel de
        bestandsnaam van de clip als de voorgestelde analysetitel; de starttijd erin maakt
        hem herkenbaar én in de praktijk uniek (knip_fragmenten dedupliceert de rest)."""
        return [(start, eind, f"{self._stam} {_tijd_tekst(start, self.fps).replace(':', '-')}")
                for start, eind in sorted(self._fragmenten)]

    def done(self, resultaat):
        # Niet closeEvent: een modale dialoog die via accept()/reject() sluit krijgt er geen.
        # Het videobestand moet los, anders houdt Windows de opname vast.
        self.speler.sluit()
        super().done(resultaat)


class PuntenBalk(QWidget):
    """
    De balk onder de tijdlijn van het kijkvenster: één streepje per bewaard punt, op zijn
    plek in de opname. De tegenhanger van `FragmentBalk` — die tekent stukken (start–eind),
    dit zijn losse momenten.

    Klikken op (of vlak naast) een streepje springt erheen; dat is de snelste weg terug naar
    hetzelfde beeld, terwijl de lijst ernaast vooral dient om te zien wát een punt is.
    """
    KLIK = Signal(int)        # index in `punten` van het aangeklikte streepje (−1 = ernaast)

    HOOGTE = 22
    RAAK_PX = 6               # hoe ver naast een streepje een klik nog telt
    KLEUR_ACHTERGROND = QColor(45, 45, 45)
    KLEUR_PUNT = QColor(90, 170, 240)
    KLEUR_SELECTIE = QColor(255, 255, 255)
    KLEUR_CURSOR = QColor(240, 240, 240)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HOOGTE)
        self.setToolTip("Bewaarde punten in deze opname. Klik op een streepje om erheen "
                        "te springen.")
        self.totaal = 1
        self.punten = []          # [(frame, label)] op framenummer gesorteerd
        self.cursor = 0
        self.selectie = -1

    def zet(self, totaal=None, punten=None, cursor=None, selectie=None):
        """Alles wat de balk toont in één aanroep bijwerken (en hertekenen)."""
        if totaal is not None:
            self.totaal = max(1, int(totaal))
        if punten is not None:
            self.punten = list(punten)
        if cursor is not None:
            self.cursor = int(cursor)
        if selectie is not None:
            self.selectie = int(selectie)
        self.update()

    def _x(self, frame):
        return int(round(frame / self.totaal * max(1, self.width() - 1)))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.KLEUR_ACHTERGROND)
        h = self.height()
        for i, (frame, _label) in enumerate(self.punten):
            x = self._x(frame)
            kleur = self.KLEUR_SELECTIE if i == self.selectie else self.KLEUR_PUNT
            p.fillRect(QRect(max(0, x - 1), 3, 3, h - 6), kleur)
            # Het nummer erbij zolang het er een van de eerste negen is: dat is tevens de
            # toets waarmee je erheen springt, dus het staat er niet voor de sier.
            if i < 9:
                p.setPen(QPen(kleur))
                p.drawText(QRect(x - 10, 2, 20, h - 4),
                           Qt.AlignHCenter | Qt.AlignTop, str(i + 1))
        p.setPen(QPen(self.KLEUR_CURSOR, 1))
        x = self._x(self.cursor)
        p.drawLine(x, 0, x, h)

    def mousePressEvent(self, event):
        klik_x = event.position().x()
        dichtst, beste = -1, self.RAAK_PX + 1
        for i, (frame, _label) in enumerate(self.punten):
            afstand = abs(self._x(frame) - klik_x)
            if afstand < beste:
                dichtst, beste = i, afstand
        self.KLIK.emit(dichtst)


class BekijkVenster(QDialog):
    """
    Een opname handmatig bekijken: beelden rechtstreeks uit de camera doorlopen, zónder
    analyse. **Er wordt niets gedetecteerd, gevolgd of gemeten** — dit is puur een speler,
    en juist daarom bruikbaar op materiaal waar de tracking niets van zou maken (de bocht,
    meerdere schaatsers door elkaar, een warming-up) en op een opname die nog niet geknipt is.

    Vier dingen maken het meer dan een speler:
      * **volledig scherm** — je kijkt naar techniek, niet naar knoppen (F11 → venster);
      * **inzoomen en vertragen** komen ongewijzigd uit `VideoSpeler` (muiswiel/zoomregelaar
        en de snelheidcombo tot 1/16×);
      * **`.` en `,` spoelen op 6×** door de opname, met de wandklok als maat (zie
        `_spoel_tick`), zodat het echt 6× is en niet "zo snel als de decoder toevallig kan";
      * **punten** die je op een frame zet en die bewaard blijven (`bron_markering`), zodat
        dezelfde sprong of afzet er de volgende sessie nog staat — en in de gedeelde
        bibliotheek ook voor een collega.

    Hergebruikt `VideoSpeler` met dezelfde twee afwijkingen als het knipvenster:
    `snel_zoeken=True` (achteruit mag seeken — dit is een kijkje, geen meting) en
    `toon_overlay=False` (er is geen analyse om te tekenen).
    """

    PANEEL_BREEDTE = 260

    def __init__(self, bron, info, bieb, trainer_naam="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Bekijken — {bron['naam']}")
        self.bron = bron
        self.bieb = bieb
        self.trainer_naam = trainer_naam
        self.info = info
        self.fps = info.fps or 30.0
        self._punten = []           # rijen uit bron_markering, op framenummer gesorteerd
        self._vullen = False        # onderdrukt itemChanged tijdens het opbouwen
        self._spoel_richting = 0    # −1 terug, 0 stil, +1 vooruit
        self._spoel_vanaf = 0
        self._spoel_t0 = 0.0

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        kop = QHBoxLayout()
        kop.addWidget(QLabel(f"<b>{bron['naam']}</b>"))
        self.lbl_spoel = QLabel("")
        self.lbl_spoel.setStyleSheet("color: #5aaaf0;")
        kop.addWidget(self.lbl_spoel)
        kop.addStretch(1)
        self.btn_paneel = QPushButton("Punten verbergen")
        self.btn_paneel.clicked.connect(self._toggle_paneel)
        kop.addWidget(self.btn_paneel)
        btn_venster = QPushButton("Venstermodus (F11)")
        btn_venster.clicked.connect(self._toggle_volledig_scherm)
        kop.addWidget(btn_venster)
        btn_sluit = QPushButton("Sluiten (Esc)")
        btn_sluit.clicked.connect(self.accept)
        kop.addWidget(btn_sluit)
        v.addLayout(kop)

        self.splitter = QSplitter(Qt.Horizontal)
        self.speler = VideoSpeler(min_grootte=(400, 200), snel_zoeken=True,
                                  toon_overlay=False)
        self.speler.op_frame_getoond = self._frame_getoond
        self.splitter.addWidget(self.speler)

        # De balk hangt ín de speler, zodat hij dezelfde breedte als de tijdlijn houdt.
        self.balk = PuntenBalk()
        self.balk.KLIK.connect(self._klik_op_balk)
        self.speler.voeg_onderbalk(self.balk)

        self.splitter.addWidget(self._bouw_puntenpaneel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        v.addWidget(self.splitter, stretch=1)

        hulp = QLabel(
            "<b>Spatie</b> pauze · <b>.</b> doorspoelen 6× · <b>,</b> terugspoelen 6× · "
            "<b>&larr;/&rarr;</b> één frame · <b>P</b> punt zetten · <b>1&ndash;9</b> naar "
            "punt · <b>Del</b> punt weg · muiswiel zoomt · <b>F11</b> venster · "
            "<b>Esc</b> sluiten")
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        v.addWidget(hulp)

        self._spoel_timer = QTimer(self)
        self._spoel_timer.timeout.connect(self._spoel_tick)

        # Lege FrameResultaat-lijst: de speler wil er één (sliderlengte, tijdlabel), maar er
        # is hier per definitie niets geanalyseerd — dat is de hele bedoeling.
        resultaten = [FrameResultaat(i, i / self.fps) for i in range(max(1, info.totaal))]
        self.speler.laad(info, resultaten, bron["pad"])
        self.balk.zet(totaal=len(resultaten))
        self._vernieuw_punten()
        self.speler.ga_naar(0)

        # De toetsen moeten werken waar de focus ook staat (een knop slikt spatie, een
        # slider de pijltjes), dus filteren we op app-niveau zolang dit venster actief is.
        QApplication.instance().installEventFilter(self)

        # Eerst een normale maat zetten en dán pas volledig scherm: F11 heeft anders geen
        # zinnige geometrie om naar terug te vallen.
        zet_venstergrootte(self, 1280, 800)
        self.setWindowState(self.windowState() | Qt.WindowFullScreen)

    def _bouw_puntenpaneel(self):
        self.paneel = QWidget()
        p = QVBoxLayout(self.paneel)
        p.setContentsMargins(6, 0, 0, 0)
        p.addWidget(QLabel("<b>Punten</b>  (klik = erheen springen)"))

        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["#", "Tijd", "Naam"])
        kop = self.tabel.horizontalHeader()
        kop.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(2, QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.cellClicked.connect(self._klik_op_rij)
        # Alleen de naamkolom is editeerbaar (de vlaggen worden per item gezet in
        # _vernieuw_punten); _vullen onderdrukt itemChanged tijdens het opbouwen.
        self.tabel.itemChanged.connect(self._punt_hernoemd)
        p.addWidget(self.tabel, stretch=1)

        btn_zet = QPushButton("➕ Punt zetten  (P)")
        btn_zet.setToolTip("Onthoudt het frame dat nu in beeld staat. Het punt blijft bij "
                           "deze opname bewaard, ook na het sluiten van dit venster.")
        btn_zet.clicked.connect(self._zet_punt)
        p.addWidget(btn_zet)
        btn_weg = QPushButton("✕ Punt verwijderen  (Del)")
        btn_weg.clicked.connect(self._verwijder_punt)
        p.addWidget(btn_weg)

        self.lbl_punten = QLabel("")
        self.lbl_punten.setStyleSheet("color: #888;")
        self.lbl_punten.setWordWrap(True)
        p.addWidget(self.lbl_punten)

        self.paneel.setMinimumWidth(self.PANEEL_BREEDTE)
        return self.paneel

    # ── Punten (bewaard in de bibliotheek) ───────────────────────────────
    def _vernieuw_punten(self, selectie=None):
        """Leest de punten opnieuw uit de database en vult tabel + balk. De database is de
        waarheid: zo staat er nooit een punt in beeld dat niet bewaard is."""
        try:
            self._punten = schaats_db.lijst_markeringen(self.bieb, self.bron["id"])
        except Exception as e:
            self._punten = []
            self.lbl_punten.setText(f"Punten konden niet gelezen worden: {e}")
            return
        self._vullen = True
        try:
            self.tabel.setRowCount(len(self._punten))
            for rij, punt in enumerate(self._punten):
                nr = QTableWidgetItem(str(rij + 1))
                nr.setFlags(nr.flags() & ~Qt.ItemIsEditable)
                self.tabel.setItem(rij, 0, nr)

                tijd = QTableWidgetItem(_tijd_tekst(punt["frame"], self.fps))
                tijd.setFlags(tijd.flags() & ~Qt.ItemIsEditable)
                tijd.setToolTip(f"frame {punt['frame']}")
                self.tabel.setItem(rij, 1, tijd)

                naam = QTableWidgetItem(punt["label"] or "")
                naam.setData(Qt.UserRole, punt["id"])
                naam.setToolTip("Dubbelklik om te hernoemen"
                                + (f" · gezet door {punt['aangemaakt_door']}"
                                   if punt["aangemaakt_door"] else ""))
                self.tabel.setItem(rij, 2, naam)
        finally:
            self._vullen = False

        self.balk.zet(punten=[(p["frame"], p["label"]) for p in self._punten],
                      selectie=selectie if selectie is not None else -1)
        n = len(self._punten)
        self.lbl_punten.setText(
            "Nog geen punten gezet." if not n
            else f"{n} punt{'en' if n != 1 else ''} bewaard bij deze opname.")
        if selectie is not None and 0 <= selectie < n:
            self.tabel.selectRow(selectie)

    def _zet_punt(self):
        frame = self.speler.huidige_idx
        if frame < 0:
            return
        if any(p["frame"] == frame for p in self._punten):
            self.lbl_punten.setText("Op dit frame staat al een punt.")
            return
        try:
            schaats_db.voeg_markering_toe(self.bieb, self.bron["id"], frame,
                                          f"Punt {len(self._punten) + 1}",
                                          self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"Het punt kon niet bewaard worden:\n{e}")
            return
        # Herlezen en dán pas selecteren: de lijst staat op framenummer, dus een punt dat
        # je halverwege terugzet komt niet onderaan te staan.
        self._vernieuw_punten()
        index = next((i for i, punt in enumerate(self._punten)
                      if punt["frame"] == frame), None)
        if index is not None:
            self.tabel.selectRow(index)
            self.balk.zet(selectie=index)

    def _verwijder_punt(self):
        rij = self.tabel.currentRow()
        if not 0 <= rij < len(self._punten):
            return
        try:
            schaats_db.verwijder_markering(self.bieb, self._punten[rij]["id"])
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"Het punt kon niet verwijderd worden:\n{e}")
            return
        self._vernieuw_punten()

    def _punt_hernoemd(self, item):
        if self._vullen or item.column() != 2:
            return
        try:
            schaats_db.wijzig_markering(self.bieb, item.data(Qt.UserRole), label=item.text())
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"De naam kon niet bewaard worden:\n{e}")

    def _ga_naar_punt(self, index):
        if 0 <= index < len(self._punten):
            self.speler.ga_naar(self._punten[index]["frame"])
            self.tabel.selectRow(index)
            self.balk.zet(selectie=index)

    def _klik_op_rij(self, rij, _kolom=0):
        self._ga_naar_punt(rij)

    def _klik_op_balk(self, index):
        if index >= 0:
            self.tabel.selectRow(index)
            self._ga_naar_punt(index)

    # ── Doorspoelen met . en , (6×) ──────────────────────────────────────
    def _start_spoelen(self, richting):
        """Begint te spoelen zolang de toets ingedrukt blijft. Het eerste frame gaat er
        meteen af, zodat een tíkje op de toets één frame opschuift en vasthouden 6× spoelt —
        allebei manieren waarop zo'n toets gebruikt wordt."""
        if self._spoel_richting == richting or not self.speler.resultaten:
            return
        self.speler.pauzeer()
        self._spoel_richting = richting
        self._spoel_vanaf = self.speler.huidige_idx
        self._spoel_t0 = time.monotonic()
        self.speler.ga_naar(self._spoel_vanaf + richting)
        self._spoel_timer.start(SPOEL_TICK_MS)
        self.lbl_spoel.setText(f"{'▶▶' if richting > 0 else '◀◀'} {SPOEL_FACTOR:g}×")

    def _stop_spoelen(self, richting=None):
        if self._spoel_richting == 0 or (richting is not None
                                         and richting != self._spoel_richting):
            return
        self._spoel_timer.stop()
        self._spoel_richting = 0
        self.lbl_spoel.setText("")

    def _spoel_tick(self):
        """Het doelframe volgt uit de **wandkloktijd** sinds de toetsdruk, niet uit een vaste
        stap per tik — hetzelfde motief als de masterklok van de vergelijkpagina. Zo is het
        echt 6× de opnamesnelheid: haalt de decoder dat niet (achteruit kost elke stap een
        seek), dan worden er meer frames overgeslagen in plaats van dat het spoelen
        vertraagt, en er stapelt zich niets op."""
        verstreken = time.monotonic() - self._spoel_t0
        stap = max(1, int(round(verstreken * self.fps * SPOEL_FACTOR)))
        doel = self._spoel_vanaf + self._spoel_richting * stap
        laatste = len(self.speler.resultaten) - 1
        if doel <= 0 or doel >= laatste:
            self.speler.ga_naar(max(0, min(doel, laatste)))
            self._stop_spoelen()          # begin/eind bereikt: er valt niets meer te spoelen
            return
        self.speler.ga_naar(doel)

    # ── Weergave ─────────────────────────────────────────────────────────
    def _frame_getoond(self, idx):
        self.balk.zet(cursor=idx)

    def _toggle_paneel(self):
        zichtbaar = not self.paneel.isVisible()
        self.paneel.setVisible(zichtbaar)
        self.btn_paneel.setText("Punten verbergen" if zichtbaar else "Punten tonen")

    def _toggle_volledig_scherm(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ── Toetsen ──────────────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        """App-brede toetsafhandeling zolang dít venster het actieve is.

        Waarom geen gewone `keyPressEvent`: na één muisklik staat de focus op een knop of op
        de tijdlijn, en die slikken respectievelijk spatie en de pijltjestoetsen voordat het
        venster ze ziet. Een filter op de applicatie krijgt ze als eerste. Tekstinvoer is de
        uitzondering — daar hoort een punt of komma gewoon in de tekst te belanden.
        """
        # Dit filter krijgt élk event van de hele applicatie langs, dus deze eerste tak
        # moet kort zijn: geen toets → meteen terug (False = "niet afgehandeld", precies wat
        # QObject.eventFilter ook zou doen).
        soort = event.type()
        if soort not in (QEvent.KeyPress, QEvent.KeyRelease) or not self.isActiveWindow():
            return False
        if isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit)):
            return False

        toets = event.key()
        richting = {Qt.Key_Period: 1, Qt.Key_Comma: -1}.get(toets)
        if richting is not None:
            # Autorepeat overslaan: tijdens het vasthouden stuurt Windows een stroom
            # press/release-paren, en die zouden het spoelen elke ~30 ms opnieuw starten —
            # waarmee de wandklok telkens op nul valt en er niets meer opschiet.
            if not event.isAutoRepeat():
                if soort == QEvent.KeyPress:
                    self._start_spoelen(richting)
                else:
                    self._stop_spoelen(richting)
            return True
        if soort != QEvent.KeyPress:
            return False

        if toets == Qt.Key_Space:
            self._stop_spoelen()
            if self.speler.speelt():
                self.speler.pauzeer()
            else:
                self.speler.speel()
        elif toets in (Qt.Key_Left, Qt.Key_Right):
            self._stop_spoelen()
            self.speler.pauzeer()
            self.speler.ga_naar(self.speler.huidige_idx
                                + (1 if toets == Qt.Key_Right else -1))
        elif toets == Qt.Key_Home:
            self.speler.ga_naar(0)
        elif toets == Qt.Key_End:
            self.speler.ga_naar(len(self.speler.resultaten) - 1)
        elif toets == Qt.Key_P:
            self._zet_punt()
        elif toets == Qt.Key_Delete:
            self._verwijder_punt()
        elif toets == Qt.Key_F11:
            self._toggle_volledig_scherm()
        elif Qt.Key_1 <= toets <= Qt.Key_9:
            self._ga_naar_punt(toets - Qt.Key_1)
        else:
            return False
        return True

    def changeEvent(self, event):
        # Gaat het venster van actief naar inactief (alt-tab, een melding ervoor), dan komt
        # de key-release nooit meer binnen en zou het spoelen eindeloos doorlopen.
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self._stop_spoelen()
        super().changeEvent(event)

    def done(self, resultaat):
        # Niet closeEvent: een modale dialoog die via accept()/reject() sluit krijgt er geen.
        # Het videobestand moet los, anders houdt Windows de opname vast.
        self._stop_spoelen()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.speler.sluit()
        super().done(resultaat)


class LokaalProef(QThread):
    """Meet op de achtergrond welke opnames echt op deze pc staan (`bestand_lokaal`).

    Op een achtergrondthread omdat de proef juist in het interessante geval geld kost: een
    opname die nog in de cloud staat levert per monster een netwerkronde op, dus een lijst
    van vijf opnames zou de bibliotheek seconden laten bevriezen. Elke uitkomst gaat los
    naar de tabel, zodat de kolom zich vult terwijl je al kunt kijken."""

    gemeten = Signal(int, str)          # bron_id, status uit schaats_db.bestand_lokaal

    def __init__(self, paden, parent=None):
        super().__init__(parent)
        self._paden = list(paden)      # [(bron_id, pad)]

    def run(self):
        for bron_id, pad in self._paden:
            if self.isInterruptionRequested():
                return
            try:
                status = schaats_db.bestand_lokaal(pad)
            except Exception:
                status = None          # een onleesbaar bestand meldt de sync-check al
            if status:
                self.gemeten.emit(bron_id, status)


class MainWindow(QMainWindow):
    def __init__(self, melding=None):
        super().__init__()
        # Statusregel van het opstartscherm (of None): de opbouw hieronder duurt op een
        # koude machine een paar seconden en dat mag te zien zijn.
        self._melding = melding or (lambda tekst: None)
        self.setWindowTitle("Schaats Analyse")
        zet_venstergrootte(self, 1400, 820, maximaliseer=True)

        self.input_pad = None
        self.model_pad = STANDAARD_MODEL
        self.smooth_n = 5
        self.threshold = 0.015
        self.doel_punt = None
        self.horizon_deg = 0.0
        self.auto_horizon = False
        self.perspectief = None
        self.geen_smoothing = False
        self.bocht_overslaan = True   # bochtframes niet analyseren/meten (checkbox in de dialoog)
        # video_info / resultaten / huidige_idx wonen in self.speler (zie de properties
        # hieronder); die wordt in _bouw_ui() aangemaakt en niets vóór die aanroep leest ze.
        self.events = []
        self.worker = None
        self.batch_worker = None
        self.bieb = None            # bibliotheekpad (gezet door _zet_bibliotheek)
        self._opnames = []          # fase 8: bronvideo-rijen achter de opnametabel
        self._lokaal = {}           # bron_id -> 'lokaal'/'deels'/'cloud' (snelheidsproef)
        self._lokaal_proef = None   # lopende LokaalProef-thread
        self._knip_tmpmap = None    # tijdelijke map met zojuist geknipte fragmenten
        self.knip_worker = None
        self.trainer_naam = schaats_db.trainer_naam()  # fase 4: gaat mee als aangemaakt_door
        self.analyse_id = None      # id van de geopende analyse in de bibliotheek
        # Bij wie de geopende analyse hoort — nodig voor de kop op de vergelijkpagina
        # (en als voorkeur in de analysekiezer); de DB kent alleen het id.
        self.analyse_schaatser_id = None
        self.analyse_schaatser_naam = ""
        self._pending_opslag = None # {schaatser_id, titel, instellingen} voor de worker
        self._bezig = False         # draait er een (batch-)analyse op de achtergrond?
        self._afsluiten = False     # venster gaat dicht: worker-slots niets meer laten doen
        self._auto_toon_klaar = True  # mag de verse analyse bij afronden vanzelf getoond?
        self._analyse_waarschuwingen = []   # meldingen uit de lopende analyse (na afloop tonen)
        self._backend_gemeld = False        # is een backend-terugval al gemeld? (zie
                                            # _waarschuw_backend_terugval)

        # Skelet-editor (fase 3) — de zoom/pan-state zit in de VideoSpeler
        self._editor_actief = False
        self._sleep = None          # {'idx', 'j', 'start_lm': Landmark} tijdens een sleep
        # Undo-items zijn getypeerd: 'sleep' verplaatst één landmark over een uitvloei-venster,
        # 'skelet' zet een compleet handmatig geplaatst skelet neer (of weer weg).
        self._undo = []
        self._redo = []
        self._handmatig = {}        # {frame_idx: set(landmark_idx)} — alleen voor de overlay-markering
        self._plaats = None         # lopende plaats-reeks, zie _start_plaatsen

        # Vergelijkpagina: één masterklok voor "Start alles" (zie _alles_tick)
        self._alles_timer = QTimer(self)
        self._alles_timer.timeout.connect(self._alles_tick)
        self._alles_lopend = []     # [(VergelijkKant, basisframe)] tijdens het samen afspelen
        self._alles_t0 = 0.0
        self._alles_factor = 1.0

        self._melding("Venster opbouwen...")
        self._bouw_ui()
        self._melding("Bibliotheek openen...")
        self._zet_bibliotheek(schaats_db.bibliotheek_pad())

    # De VideoSpeler is de enige eigenaar van deze drie; hier alleen doorkijkjes, zodat de
    # bestaande editor-/tabelcode ongewijzigd blijft werken én een stille tweede kopie
    # structureel onmogelijk is (een stray toewijzing geeft meteen een AttributeError).
    @property
    def resultaten(self):
        return self.speler.resultaten

    @property
    def video_info(self):
        return self.speler.video_info

    @property
    def huidige_idx(self):
        return self.speler.huidige_idx

    # ── UI opbouw ────────────────────────────────────────────────────────
    def _bouw_ui(self):
        toolbar = QToolBar("Hoofd")
        self.addToolBar(toolbar)
        self.actie_bibliotheek = QAction("Bibliotheek", self)
        self.actie_bibliotheek.triggered.connect(self._terug_naar_start)
        toolbar.addAction(self.actie_bibliotheek)

        self.stack = QStackedWidget()
        self.pagina_start = self._bouw_startpagina()
        self.pagina_analyse = self._bouw_analysepagina()
        self.pagina_vergelijk = self._bouw_vergelijkpagina()
        self.stack.addWidget(self.pagina_start)
        self.stack.addWidget(self.pagina_analyse)
        self.stack.addWidget(self.pagina_vergelijk)
        self.stack.setCurrentWidget(self.pagina_start)
        # Eén haak i.p.v. bij elke setCurrentWidget-aanroep: een verlaten pagina mag niet
        # doordecoderen op de achtergrond.
        self.stack.currentChanged.connect(self._paginawissel)

        # Centraal = de stack + een blijvende voortgangsbalk onderin (verborgen tenzij
        # er een analyse/batch draait). Doordat de balk buiten de stack staat, blijft ze
        # over paginawissels heen zichtbaar en blokkeert ze het venster niet — zo kan er
        # gebrowst worden terwijl een analyse op de achtergrond rekent.
        self.voortgang_balk = self._bouw_voortgangsbalk()
        centraal = QWidget()
        cv = QVBoxLayout(centraal)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.stack, stretch=1)
        cv.addWidget(self.voortgang_balk)
        self.setCentralWidget(centraal)

        # Twee permanente statusbalk-widgets: hoeveel frames een skelet hebben (dekking) en
        # de live-status van het huidige frame. Permanent, want showMessage() overschrijft
        # de gewone statusbalk-tekst en de dekking moet altijd afleesbaar blijven.
        self.lbl_dekking = QLabel("")
        self.lbl_dekking.setToolTip(
            "Aantal frames met een skelet (gedetecteerd of handmatig geplaatst).\n"
            "Frames zonder skelet breken een afzetmeting af — met '✏ Bewerken' zijn ze "
            "handmatig aan te vullen.")
        self.statusBar().addPermanentWidget(self.lbl_dekking)
        self.lbl_live = QLabel("")
        self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_live)
        self.statusBar().showMessage(
            f"Kies een schaatser en start of open een analyse.  ·  backend: {BACKEND_NAAM}")

    def _pauzeer_alles(self):
        """Stopt elke lopende weergave (analysepagina én beide vergelijk-spelers).
        Idempotent, dus veilig om overal aan te roepen."""
        self._stop_alles()
        self.speler.pauzeer()
        for kant in (self.kant_links, self.kant_rechts):
            kant.speler.pauzeer()

    def _paginawissel(self, _idx=None):
        """Bij het verlaten van een pagina: alles pauzeren, en de bewerk-modus uitzetten —
        de undo-sneltoetsen zijn venster-breed, dus Ctrl+Z op een andere pagina zou anders
        een onzichtbare analyse bewerken én naar de bibliotheek wegschrijven."""
        self._pauzeer_alles()
        if self.stack.currentWidget() is not self.pagina_analyse:
            if self.btn_bewerken.isChecked():
                self.btn_bewerken.setChecked(False)   # triggert _toggle_bewerken(False)
            self._stop_plaatsen()                     # faalveilig: geen half skelet achterlaten
            self.lbl_live.setText("")                 # geen stale status van een andere pagina
            self.lbl_dekking.setText("")

    def _bouw_startpagina(self):
        """De bibliotheek (fase 1): links de schaatsers, rechts hun analyses.
        Fase 8 zet daar een tweede tabblad naast: de nog te knippen opnames."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        titel = QLabel("Schaatser Analyse — bibliotheek")
        titel.setStyleSheet("font-size: 22px; font-weight: bold; padding: 4px;")
        v.addWidget(titel)

        # Twee tabbladen i.p.v. een derde kolom: de opnames horen bij niemand in het
        # bijzonder (het is een werklijst voor het team), dus ze hangen niet aan de
        # schaatserselectie links.
        self.tabs_bieb = QTabWidget()
        v.addWidget(self.tabs_bieb, stretch=1)

        splitter = QSplitter(Qt.Horizontal)

        # Links: schaatsers.
        links = QWidget()
        lv = QVBoxLayout(links)
        lv.addWidget(QLabel("Schaatsers"))
        self.lijst_schaatsers = QListWidget()
        self.lijst_schaatsers.currentItemChanged.connect(lambda *_: self._vernieuw_analyses())
        lv.addWidget(self.lijst_schaatsers, stretch=1)
        rij_s = QHBoxLayout()
        self.btn_nieuwe_schaatser = QPushButton("Nieuwe schaatser...")
        self.btn_nieuwe_schaatser.clicked.connect(self._nieuwe_schaatser)
        self.btn_bewerk_schaatser = QPushButton("Bewerken...")
        self.btn_bewerk_schaatser.clicked.connect(self._bewerk_schaatser)
        self.btn_verwijder_schaatser = QPushButton("Verwijderen")
        self.btn_verwijder_schaatser.clicked.connect(self._verwijder_schaatser)
        for b in (self.btn_nieuwe_schaatser, self.btn_bewerk_schaatser,
                  self.btn_verwijder_schaatser):
            rij_s.addWidget(b)
        lv.addLayout(rij_s)
        splitter.addWidget(links)

        # Rechts: analyses van de geselecteerde schaatser (uit de events-cache).
        rechts = QWidget()
        rv = QVBoxLayout(rechts)
        rv.addWidget(QLabel("Analyses  (dubbelklik om te openen)"))
        self.tabel_analyses = QTableWidget(0, 4)
        self.tabel_analyses.setHorizontalHeaderLabels(["Datum", "Titel", "Duur", ""])
        kop = self.tabel_analyses.horizontalHeader()
        # Alleen de titel rekt mee; datum/duur/knoppen krijgen precies wat ze nodig hebben,
        # anders staan vier knoppen in een kwart van de tabelbreedte geperst.
        kop.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(1, QHeaderView.Stretch)
        kop.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tabel_analyses.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel_analyses.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel_analyses.cellDoubleClicked.connect(
            lambda *_: self._open_analyse_uit_bibliotheek())
        rv.addWidget(self.tabel_analyses, stretch=1)
        rij_a = QHBoxLayout()
        self.btn_nieuwe_analyse = QPushButton("Nieuwe analyse...")
        self.btn_nieuwe_analyse.clicked.connect(self._nieuwe_analyse)
        self.btn_batch_analyse = QPushButton("Batch-analyse...")
        self.btn_batch_analyse.setToolTip(
            "Meerdere video's tegelijk kiezen en achter elkaar analyseren. Je stelt vooraf "
            "per video de doelschaatser en horizon in; daarna draait de hele rij onbewaakt.")
        # lambda: `clicked` geeft anders zijn `checked`-bool door als `voorgevuld`.
        self.btn_batch_analyse.clicked.connect(lambda: self._nieuwe_batch_analyse())
        self.btn_vergelijk = QPushButton("Vergelijk schaatsers...")
        self.btn_vergelijk.setToolTip(
            "Twee opgeslagen analyses naast elkaar zetten. Elke video is los te bedienen;\n"
            "met een sync-punt per kant en 'Start alles' lopen ze vanaf dezelfde fase\n"
            "van de slag tegelijk.")
        self.btn_vergelijk.clicked.connect(self._vergelijk_schaatsers)
        # Openen/Info/Hernoemen/Verwijderen horen bij één analyse en staan daarom in de rij
        # zelf (zie _maak_rij_knoppen); hieronder blijven alleen de bibliotheek-brede acties.
        for b in (self.btn_nieuwe_analyse, self.btn_batch_analyse, self.btn_vergelijk):
            rij_a.addWidget(b)
        rij_a.addStretch(1)
        rv.addLayout(rij_a)
        splitter.addWidget(rechts)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.tabs_bieb.addTab(splitter, "Schaatsers && analyses")
        self.tabs_bieb.addTab(self._bouw_opnamespaneel(), "Opnames")

        # Onderin, bovenste rij: vernieuwen + trainersnaam (delen via een cloudmap, fase 4).
        rij_deel = QHBoxLayout()
        knop_vernieuw = QPushButton("Vernieuwen")
        knop_vernieuw.setToolTip(
            "Lees de bibliotheek opnieuw in — toont analyses die collega's intussen aan de\n"
            "gedeelde cloudmap hebben toegevoegd, zonder de app te herstarten.")
        knop_vernieuw.clicked.connect(self._vernieuw_bibliotheek)
        rij_deel.addWidget(knop_vernieuw)
        knop_naam = QPushButton("Jouw naam...")
        knop_naam.setToolTip(
            "Je naam wordt bij nieuwe analyses bewaard (aangemaakt door), zodat in een\n"
            "gedeelde bibliotheek zichtbaar is wie welke analyse maakte.")
        knop_naam.clicked.connect(self._kies_trainer_naam)
        rij_deel.addWidget(knop_naam)
        self.lbl_trainer = QLabel("")
        self.lbl_trainer.setStyleSheet("color: #888;")
        rij_deel.addWidget(self.lbl_trainer)
        rij_deel.addStretch(1)
        v.addLayout(rij_deel)
        self._toon_trainer_naam()

        # Onderste rij: de bibliotheekmap (deelbaar via een cloudmap, zie ROADMAP fase 4).
        rij_b = QHBoxLayout()
        knop_bieb = QPushButton("Bibliotheekmap...")
        knop_bieb.setToolTip(
            "De map met de database en alle video's/landmarks. Zet deze map in een\n"
            "gesynchroniseerde cloudmap (Google Drive/OneDrive/Dropbox) om de\n"
            "bibliotheek met andere trainers te delen; elke trainer wijst dezelfde\n"
            "map aan.")
        knop_bieb.clicked.connect(self._kies_bibliotheekmap)
        rij_b.addWidget(knop_bieb)
        self.lbl_bieb = QLabel("")
        self.lbl_bieb.setStyleSheet("color: #888;")
        rij_b.addWidget(self.lbl_bieb, stretch=1)
        v.addLayout(rij_b)

        return paneel

    def _bouw_opnamespaneel(self):
        """Fase 8: de werklijst met ruwe opnames uit `<bibliotheek>/opnames/`.

        De **map** is de waarheid over welke bestanden er zijn (bij elke keer openen en bij
        'Vernieuwen' opnieuw gescand), de **database** over wat wij ervan weten: status,
        notitie en welke stukken al geanalyseerd zijn. Omdat de map net als de rest van de
        bibliotheek in de gedeelde Drive staat, is dit geen persoonlijk lijstje maar een
        werklijst voor het team."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        uitleg = QLabel(
            "Ruwe trainingsopnames die nog geknipt moeten worden. Zet ze in de map "
            "<b>opnames</b> in de bibliotheek; ze verschijnen hier vanzelf (of na "
            "'Vernieuwen').<br>Dubbelklik op een opname om er fragmenten uit te knippen, "
            "of open hem met <b>Bekijken</b> om alleen te kijken — volledig scherm, geen "
            "analyse.")
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)

        self.tabel_opnames = QTableWidget(0, 6)
        self.tabel_opnames.setHorizontalHeaderLabels(
            ["Opname", "Duur", "Op deze pc", "Status", "Fragmenten / punten", "Notitie"])
        kop = self.tabel_opnames.horizontalHeader()
        kop.setSectionResizeMode(0, QHeaderView.Stretch)
        for k in (1, 2, 3, 4):
            kop.setSectionResizeMode(k, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(OPNAME_KOL_NOTITIE, QHeaderView.Stretch)
        self.tabel_opnames.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel_opnames.cellDoubleClicked.connect(lambda *_: self._knip_opname())
        # De notitie is ter plekke te bewerken; alleen die kolom is editeerbaar (zie
        # _vul_opnames, dat de vlaggen per item zet).
        self.tabel_opnames.itemChanged.connect(self._opname_notitie_gewijzigd)
        self._vullen_opnames = False   # onderdrukt itemChanged tijdens het opbouwen
        v.addWidget(self.tabel_opnames, stretch=1)

        rij = QHBoxLayout()
        self.btn_bekijken = QPushButton("👁 Bekijken (volledig scherm)...")
        self.btn_bekijken.setToolTip(
            "Speelt de gekozen opname af zoals hij uit de camera komt: geen detectie, geen\n"
            "tracking, alleen beeld. Vertragen, inzoomen, met . en , op 6× door- en\n"
            "terugspoelen, en punten zetten die bewaard blijven.")
        self.btn_bekijken.clicked.connect(self._bekijk_opname)
        rij.addWidget(self.btn_bekijken)
        self.btn_knippen = QPushButton("✂ Fragmenten knippen...")
        self.btn_knippen.setToolTip(
            "Open de gekozen opname, markeer de bruikbare stukken (start/stop) en laat ze\n"
            "daarna in één keer analyseren — dezelfde batch-flow als 'Batch-analyse...'.")
        self.btn_knippen.clicked.connect(self._knip_opname)
        rij.addWidget(self.btn_knippen)
        knop_map = QPushButton("Map met opnames openen")
        knop_map.setToolTip("Opent de map waar de ruwe opnames in horen te staan.")
        knop_map.clicked.connect(self._open_opnamesmap)
        rij.addWidget(knop_map)
        rij.addStretch(1)
        v.addLayout(rij)
        return paneel

    def _vernieuw_opnames(self):
        """Scant `opnames/` en vult de tabel. Schrijft alleen als er echt nieuwe bestanden
        zijn (synchroniseer_bronmap), zodat de gedeelde DB niet bij elke app-start van elke
        trainer wordt aangeraakt."""
        try:
            schaats_db.synchroniseer_bronmap(self.bieb)
            opnames = schaats_db.lijst_bronvideos(self.bieb)
        except Exception as e:
            self.tabel_opnames.setRowCount(0)
            self.statusBar().showMessage(f"Opnames konden niet gelezen worden: {e}", 6000)
            return
        self._opnames = opnames

        self._vullen_opnames = True
        try:
            self.tabel_opnames.setRowCount(len(opnames))
            for rij, b in enumerate(opnames):
                ontbreekt = b["sync"] == "ontbreekt"
                naam = QTableWidgetItem(b["naam"] + ("  (bestand niet gevonden)"
                                                     if ontbreekt else ""))
                naam.setData(Qt.UserRole, b["id"])
                naam.setFlags(naam.flags() & ~Qt.ItemIsEditable)
                if b["bijgewerkt_door"]:
                    naam.setToolTip(f"Status gezet door {b['bijgewerkt_door']}")
                if ontbreekt:
                    naam.setForeground(QColor(150, 150, 150))
                elif b["sync"] == "onvolledig":
                    naam.setToolTip("De cloudsync is dit bestand nog aan het downloaden.")
                self.tabel_opnames.setItem(rij, 0, naam)

                duur = QTableWidgetItem(
                    _tijd_tekst(b["totaal_frames"], b["fps"])
                    if (b["totaal_frames"] and b["fps"]) else "—")
                duur.setFlags(duur.flags() & ~Qt.ItemIsEditable)
                self.tabel_opnames.setItem(rij, 1, duur)

                # De status zet je zélf: er wordt nooit automatisch iets op 'klaar' gezet,
                # want het programma kan niet weten of jij een opname af vindt.
                combo = QComboBox()
                combo.addItems(schaats_db.BRON_STATUSSEN)
                idx = combo.findText(b["status"])
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.currentTextChanged.connect(
                    lambda tekst, bid=b["id"]: self._zet_opname_status(bid, tekst))
                self.tabel_opnames.setCellWidget(rij, OPNAME_KOL_STATUS, combo)

                n_frag, n_sch = b["aantal_fragmenten"], b["aantal_schaatsers"]
                n_pt = b.get("aantal_punten", 0)
                telling = QTableWidgetItem(
                    f"{n_frag} fragment{'en' if n_frag != 1 else ''}"
                    + (f" · {n_sch} schaatser{'s' if n_sch != 1 else ''}" if n_frag else "")
                    + (f" · {n_pt} punt{'en' if n_pt != 1 else ''}" if n_pt else ""))
                telling.setFlags(telling.flags() & ~Qt.ItemIsEditable)
                self.tabel_opnames.setItem(rij, OPNAME_KOL_TELLING, telling)

                notitie = QTableWidgetItem(b["notitie"] or "")
                notitie.setToolTip("Dubbelklik om te bewerken (bv. 'training 3 aug, "
                                   "tempo-serie').")
                self.tabel_opnames.setItem(rij, OPNAME_KOL_NOTITIE, notitie)

                self._zet_lokaal_cel(rij, self._lokaal.get(b["id"]))
        finally:
            self._vullen_opnames = False
        if opnames and self.tabel_opnames.currentRow() < 0:
            self.tabel_opnames.selectRow(0)   # 'Knippen...' werkt dan meteen
        self.tabs_bieb.setTabText(1, f"Opnames ({len(opnames)})" if opnames else "Opnames")
        self._start_lokaal_proef(opnames)

    def _geselecteerde_opname(self):
        rij = self.tabel_opnames.currentRow()
        if rij < 0:
            return None
        item = self.tabel_opnames.item(rij, 0)
        bron_id = item.data(Qt.UserRole) if item else None
        return next((b for b in getattr(self, "_opnames", []) if b["id"] == bron_id), None)

    def _start_lokaal_proef(self, opnames):
        """Laat op de achtergrond meten welke opnames op deze pc staan.

        Alleen voor bestanden die er zijn: bij 'ontbreekt' zegt de sync-check het al. Een
        lopende meting wordt afgebroken — na een verversing kunnen de rijen anders zijn, en
        een uitkomst van een oude lijst hoort niet meer in de tabel."""
        self._stop_lokaal_proef()
        paden = [(b["id"], b["pad"]) for b in opnames if b["sync"] != "ontbreekt"]
        if not paden:
            return
        self._lokaal_proef = LokaalProef(paden, self)
        self._lokaal_proef.gemeten.connect(self._lokaal_gemeten)
        self._lokaal_proef.start()

    def _stop_lokaal_proef(self):
        """Breekt een lopende meting af. `wait` mag hier: de thread controleert de vlag
        tussen twee opnames door en één monster duurt hooguit een seconde."""
        proef = self._lokaal_proef
        self._lokaal_proef = None
        if proef is not None and proef.isRunning():
            proef.requestInterruption()
            proef.wait(3000)

    def _lokaal_gemeten(self, bron_id, status):
        """Eén uitkomst binnen: onthouden en de cel bijwerken (de rij kan intussen weg zijn)."""
        self._lokaal[bron_id] = status
        for b in self._opnames:
            if b["id"] == bron_id:
                b["lokaal"] = status
        for rij in range(self.tabel_opnames.rowCount()):
            item = self.tabel_opnames.item(rij, OPNAME_KOL_NAAM)
            if item is not None and item.data(Qt.UserRole) == bron_id:
                self._zet_lokaal_cel(rij, status)
                return

    def _zet_lokaal_cel(self, rij, status):
        """Vult de kolom 'Op deze pc'. Zonder uitkomst een streepje: de meting loopt nog, en
        'nee' zou dan een bewering zijn die we nog niet kunnen doen."""
        tekst, kleur, uitleg = LOKAAL_WEERGAVE.get(
            status, ("—", QColor(150, 150, 150), "De snelheidsproef loopt nog."))
        cel = QTableWidgetItem(tekst)
        cel.setFlags(cel.flags() & ~Qt.ItemIsEditable)
        cel.setForeground(kleur)
        cel.setToolTip(uitleg)
        self.tabel_opnames.setItem(rij, OPNAME_KOL_LOKAAL, cel)

    def _zet_opname_status(self, bron_id, status):
        if self._vullen_opnames:
            return
        try:
            schaats_db.wijzig_bronvideo(self.bieb, bron_id, status=status,
                                        bijgewerkt_door=self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Opname", f"Status opslaan mislukte:\n{e}")
            return
        for b in getattr(self, "_opnames", []):
            if b["id"] == bron_id:
                b["status"] = status
        self.statusBar().showMessage(f"Status → {status}", 3000)

    def _opname_notitie_gewijzigd(self, item):
        if self._vullen_opnames or item.column() != OPNAME_KOL_NOTITIE:
            return
        naam_item = self.tabel_opnames.item(item.row(), 0)
        if naam_item is None:
            return
        try:
            schaats_db.wijzig_bronvideo(self.bieb, naam_item.data(Qt.UserRole),
                                        notitie=item.text(),
                                        bijgewerkt_door=self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Opname", f"Notitie opslaan mislukte:\n{e}")

    def _opname_beschikbaar(self, bron):
        """Staat het bestand er, en helemaal? Meldt zelf wat eraan schort en geeft False.

        Zowel het knipvenster als het kijkvenster openen de opname rechtstreeks van schijf.
        In een gedeelde cloudmap is een opname van een half uur minutenlang onderweg, en dan
        is één nette melding beter dan een venster dat op een half bestand stukloopt."""
        if bron["sync"] == "ontbreekt":
            QMessageBox.warning(
                self, "Opname niet gevonden",
                f"Het bestand staat niet (meer) op schijf:\n{bron['pad']}\n\n"
                "Is de bibliotheek een gedeelde cloudmap, dan is de opname mogelijk nog niet "
                "gesynchroniseerd.")
            return False
        if bron["sync"] == "onvolledig":
            QMessageBox.warning(
                self, "Opname wordt nog gedownload",
                f"'{bron['naam']}' is op deze pc nog kleiner dan bij de collega die hem "
                "toevoegde — de cloudsync is er nog mee bezig.\n\n"
                "Probeer het opnieuw zodra de download klaar is.")
            return False
        return self._waarschuw_niet_lokaal(bron)

    def _waarschuw_niet_lokaal(self, bron):
        """Waarschuwt als de opname niet offline op deze pc staat; True = doorgaan.

        Het bestand is er wél — de cloudmap laat hem gewoon zien — maar het beeld komt
        er per stukje overheen. Gemeten op deze bibliotheek (24 aug 2026, Google Drive
        in streaming-stand): één sprong in het knipvenster haalde ~40 MB op en kostte
        5 tot 20 s, tegen 30-120 ms als dezelfde opname lokaal staat. Dat valt niet met
        code te verhelpen — de speler springt al gericht i.p.v. door te spoelen (zie
        SEEK_DREMPEL_FRAMES), en die 40 MB is wat ffmpeg nodig heeft om in een MPEG-TS
        zonder index het juiste tijdstip te vinden. Het enige zinnige is het zéggen,
        vóórdat iemand denkt dat het programma hangt.

        Doorgaan mag: soms wil je alleen even het begin zien, en wat je al bekeken hebt
        zit in de cloudcache en is daarna wél meteen terug."""
        status = self._lokaal.get(bron["id"])
        if status is None:                 # de achtergrondmeting was nog niet zover
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                status = schaats_db.bestand_lokaal(bron["pad"])
            except Exception:
                status = None
            finally:
                QApplication.restoreOverrideCursor()
            if status:
                self._lokaal[bron["id"]] = status
        if status not in ("cloud", "deels"):
            return True                    # lokaal, of niet te meten → niet zeuren

        antwoord = QMessageBox.warning(
            self, "Opname staat nog in de cloud",
            f"'{bron['naam']}' staat "
            + ("nog maar gedeeltelijk" if status == "deels" else "niet")
            + " offline op deze pc; het beeld wordt tijdens het kijken uit de cloud "
              "gedownload.\n\n"
              "Doorbladeren wordt daardoor traag: een sprong naar een ander moment "
              "kost al gauw 5 tot 20 seconden, tegen een tiende seconde als de opname "
              "lokaal staat. Ook het knippen leest de opname helemaal door.\n\n"
              "Beter: rechtsklik in Verkenner de map 'opnames' → Google Drive → "
              "'Offline beschikbaar maken', wacht tot hij binnen is en druk daarna "
              "op 'Vernieuwen'.\n\n"
              "Toch nu openen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return antwoord == QMessageBox.Yes

    def _bekijk_opname(self):
        """Een opname handmatig doorkijken: alleen beeld, geen analyse.

        Bewust zónder de controles die het knippen wél doet (draait er een analyse, bestaat
        er al een schaatser): er wordt niets gemeten en er komt niets in de bibliotheek
        terecht behalve de punten, en die horen bij de opname zelf."""
        bron = self._geselecteerde_opname()
        if bron is None:
            QMessageBox.information(
                self, "Opname bekijken",
                "Kies eerst een opname in de lijst.\n\n"
                "Staat er niets? Zet je opnames in de map 'opnames' in de bibliotheek en "
                "druk op 'Vernieuwen'.")
            return
        if not self._opname_beschikbaar(bron):
            return
        try:
            info = video_info(bron["pad"])
        except Exception as e:
            QMessageBox.critical(self, "Opname", f"Kan de opname niet openen:\n{e}")
            return
        dlg = BekijkVenster(bron, info, self.bieb, self.trainer_naam, parent=self)
        dlg.exec()
        self._vernieuw_opnames()      # de puntentelling in de lijst bijwerken

    def _open_opnamesmap(self):
        pad = schaats_db.opnames_pad(self.bieb)
        try:
            os.startfile(pad)                      # Windows; elders valt hij netjes terug
        except Exception:
            QMessageBox.information(self, "Map met opnames", pad)

    def _bouw_analysepagina(self):
        paneel = QWidget()
        layout = QVBoxLayout(paneel)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._bouw_videopaneel())
        splitter.addWidget(self._bouw_datapaneel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return paneel

    def _bouw_vergelijkpagina(self):
        """Twee analyses naast elkaar: elke kant los te bedienen, plus een gedeelde
        'Start alles' die beide vanaf hun sync-punt tegelijk laat lopen."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        titel = QLabel("Vergelijk schaatsers")
        titel.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px;")
        v.addWidget(titel)

        splitter = QSplitter(Qt.Horizontal)
        self.kant_links = VergelijkKant(
            "Links", lambda: self._kies_vergelijk_kant(self.kant_links))
        self.kant_rechts = VergelijkKant(
            "Rechts", lambda: self._kies_vergelijk_kant(self.kant_rechts))
        for kant in (self.kant_links, self.kant_rechts):
            splitter.addWidget(kant)
            # Zelf op ▶ drukken = handmatige besturing overnemen: de masterklok laten los.
            kant.speler.btn_play.clicked.connect(self._stop_alles)
            # Een kant leegmaken terwijl de masterklok loopt: eerst de klok los.
            kant.btn_leeg.clicked.connect(
                lambda _=False, k=kant: (self._stop_alles(), k.leeg()))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        v.addWidget(splitter, stretch=1)

        balk = QHBoxLayout()
        self.btn_start_alles = QPushButton("▶ Start alles")
        self.btn_start_alles.setToolTip(
            "Speelt beide video's tegelijk af vanaf hun sync-punt, elk op z'n eigen fps.")
        self.btn_start_alles.clicked.connect(self._start_alles)
        balk.addWidget(self.btn_start_alles)
        self.btn_pauzeer_alles = QPushButton("⏸ Pauzeer alles")
        self.btn_pauzeer_alles.clicked.connect(self._pauzeer_alles)
        balk.addWidget(self.btn_pauzeer_alles)
        self.btn_naar_sync = QPushButton("⏮ Beide naar sync")
        self.btn_naar_sync.clicked.connect(self._beide_naar_sync)
        balk.addWidget(self.btn_naar_sync)
        self.chk_vanaf_sync = QCheckBox("vanaf sync-punt")
        self.chk_vanaf_sync.setChecked(True)
        self.chk_vanaf_sync.setToolTip(
            "Uit: 'Start alles' hervat waar beide video's nu staan, zonder terug te springen.")
        balk.addWidget(self.chk_vanaf_sync)

        balk.addWidget(QLabel("Snelheid"))
        self.combo_alles_snelheid = QComboBox()
        self.combo_alles_snelheid.setToolTip(
            "Afspeelsnelheid op deze pagina — geldt voor 'Start alles' én voor een kant "
            "die je los afspeelt, zodat de video's altijd even snel lopen.")
        for label, factor in SNELHEDEN:
            self.combo_alles_snelheid.addItem(label, factor)
        self.combo_alles_snelheid.setCurrentIndex(ALLES_SNELHEID_IDX)
        self.combo_alles_snelheid.currentIndexChanged.connect(self._zet_alles_snelheid)
        balk.addWidget(self.combo_alles_snelheid)
        self._zet_alles_snelheid()      # kanten meteen op de startsnelheid zetten

        balk.addStretch(1)
        v.addLayout(balk)
        return paneel

    def _bouw_videopaneel(self):
        """De gedeelde VideoSpeler plus de editor-onderdelen die alléén op de analysepagina
        horen (de vergelijkpagina gebruikt dezelfde speler, zonder editor)."""
        # Bescheiden ondergrens: het beeld rekt toch mee met het venster, en een hoge
        # ondergrens tilt het venster-minimum boven de beschikbare schermhoogte uit —
        # dan negeert Qt de gevraagde venstergrootte (zie zet_venstergrootte).
        self.speler = VideoSpeler(min_grootte=(400, 240))
        self.speler.op_frame_getoond = self._speler_frame_getoond
        self.speler.overlay_tekenaar = self._teken_handles
        self.speler.op_muis_druk = self._editor_muis_druk
        self.speler.op_muis_beweeg = self._editor_muis_beweeg
        self.speler.op_muis_los = self._editor_muis_los

        self.btn_bewerken = QPushButton("✏ Bewerken")
        self.btn_bewerken.setCheckable(True)
        self.btn_bewerken.setToolTip(
            "Skelet-editor: sleep foute landmarkpunten naar de juiste plek.\n"
            "De correctie vloeit uit naar de buurframes (instelbaar) en wordt\n"
            "direct opgeslagen.")
        self.btn_bewerken.toggled.connect(self._toggle_bewerken)
        self.speler.voeg_bedieningsknop(self.btn_bewerken)

        self.btn_vergelijk_deze = QPushButton("⇄ Vergelijk met...")
        self.btn_vergelijk_deze.setToolTip(
            "Zet deze analyse links op de vergelijkpagina en kies er een andere naast.")
        self.btn_vergelijk_deze.clicked.connect(self._vergelijk_met_deze)
        self.btn_vergelijk_deze.setEnabled(False)
        self.speler.voeg_bedieningsknop(self.btn_vergelijk_deze)

        self.btn_info = QPushButton("ℹ Info...")
        self.btn_info.setToolTip(
            "Met welke appversie, backend en instellingen is deze analyse gemaakt?")
        # lambda: clicked() geeft anders `checked=False` door als analyse_id.
        self.btn_info.clicked.connect(lambda: self._toon_analyse_info())
        self.btn_info.setEnabled(False)
        self.speler.voeg_bedieningsknop(self.btn_info)

        self.btn_bocht_nu = QPushButton("Bocht bepalen")
        self.btn_bocht_nu.setToolTip(
            "Voor analyses van vóór de bochtdetectie: bepaal alsnog welke frames in de\n"
            "bocht liggen en haal ze uit de meting.\n"
            "\n"
            "Er wordt niets opnieuw geanalyseerd — de landmarks van de hele clip staan al\n"
            "opgeslagen, dus de heupstand valt er zo uit af te lezen. Je krijgt eerst te\n"
            "zien wat het met de tabel doet en kunt dan pas beslissen.")
        self.btn_bocht_nu.clicked.connect(self._bepaal_bocht_nu)
        self.btn_bocht_nu.setEnabled(False)
        self.speler.voeg_bedieningsknop(self.btn_bocht_nu)

        # Editor-balk (fase 3): alleen zichtbaar in bewerk-modus. Afbrekend (WrapBalk), want
        # met de plaats-knoppen erbij past hij op een laptopscherm niet meer op één regel —
        # en een te brede balk tilt het venster-minimum boven de schermhoogte uit.
        self.editor_balk = WrapBalk()
        self.editor_balk.addWidget(QLabel("Uitvloeien ±"))
        self.spin_uitvloei = QSpinBox()
        self.spin_uitvloei.setRange(0, 60)
        self.spin_uitvloei.setValue(8)
        self.spin_uitvloei.setSuffix(" frames")
        self.spin_uitvloei.setToolTip(
            "Hoe ver de correctie naar de buurframes uitvloeit (cosinus-afbouw).\n"
            "0 = alleen dit frame. Stopt bij een detectiegat.")
        self.editor_balk.addWidget(self.spin_uitvloei)
        self.btn_undo = QPushButton("↶ Ongedaan")
        self.btn_undo.clicked.connect(self._undo_edit)
        self.editor_balk.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("↷ Opnieuw")
        self.btn_redo.clicked.connect(self._redo_edit)
        self.editor_balk.addWidget(self.btn_redo)
        self.btn_volgend_gat = QPushButton("⏭ Volgend gat")
        self.btn_volgend_gat.setToolTip(
            "Spring naar het eerstvolgende frame zonder skelet.\n"
            "Na het laatste gat begint de zoektocht weer vooraan.\n"
            "Frames in de bocht worden overgeslagen — daar wordt toch niet gemeten.")
        self.btn_volgend_gat.clicked.connect(self._ga_naar_volgend_gat)
        self.editor_balk.addWidget(self.btn_volgend_gat)
        self.btn_maak_skelet = QPushButton("➕ Maak skelet")
        self.btn_maak_skelet.setToolTip(
            "Zet een skelet op dit frame. Het wordt overgenomen van de buurframes,\n"
            "daarna sleep je de punten naar de juiste plek — net als op elk ander frame.\n"
            "Valt er niets over te nemen, dan vraagt het programma de punten\n"
            "één voor één (schouders, heupen, knieën, enkels).\n"
            "Alleen beschikbaar op een frame zonder gedetecteerde pose dat niet in de bocht ligt.")
        self.btn_maak_skelet.clicked.connect(self._start_plaatsen)
        self.editor_balk.addWidget(self.btn_maak_skelet)
        self.btn_herstel = QPushButton("Herstel origineel")
        self.btn_herstel.setToolTip("Zet alle landmarks terug naar de oorspronkelijke detectie.")
        self.btn_herstel.clicked.connect(self._herstel_origineel)
        self.editor_balk.addWidget(self.btn_herstel)
        self.lbl_editor_hint = QLabel("")
        self.lbl_editor_hint.setStyleSheet("color: #888;")
        self.editor_balk.addWidget(self.lbl_editor_hint)
        self.editor_balk.setVisible(False)
        self.speler.voeg_onderbalk(self.editor_balk)

        # Plaats-balk: alleen zichtbaar tijdens een lopende klikreeks. Apart van de
        # editor-balk zodat de gewone bewerk-knoppen niet met de reeks-knoppen mengen.
        self.plaats_balk = WrapBalk()
        self.lbl_plaats = QLabel("")
        self.lbl_plaats.setStyleSheet("font-weight: bold;")
        self.plaats_balk.addWidget(self.lbl_plaats)
        self.btn_plaats_vorige = QPushButton("← Vorige punt")
        self.btn_plaats_vorige.clicked.connect(self._plaats_vorige)
        self.plaats_balk.addWidget(self.btn_plaats_vorige)
        self.btn_plaats_over = QPushButton("Overslaan →")
        self.btn_plaats_over.setToolTip(
            "Dit punt overslaan — het houdt de positie uit de voorvulling.")
        self.btn_plaats_over.clicked.connect(self._plaats_overslaan)
        self.plaats_balk.addWidget(self.btn_plaats_over)
        self.btn_plaats_klaar = QPushButton("✔ Klaar")
        self.btn_plaats_klaar.setToolTip(
            "Het skelet vastleggen. Kan pas als heup, knie en enkel van beide benen\n"
            "een zichtbare positie hebben — daar rusten alle metingen op.")
        self.btn_plaats_klaar.clicked.connect(self._plaats_klaar)
        self.plaats_balk.addWidget(self.btn_plaats_klaar)
        self.btn_plaats_annuleer = QPushButton("✕ Annuleren")
        self.btn_plaats_annuleer.clicked.connect(self._plaats_annuleren)
        self.plaats_balk.addWidget(self.btn_plaats_annuleer)
        self.plaats_balk.setVisible(False)
        self.speler.voeg_onderbalk(self.plaats_balk)

        # Sneltoetsen voor undo/redo (alleen actief in bewerk-modus, zie de handlers).
        QShortcut(QKeySequence.Undo, self).activated.connect(self._undo_edit)
        QShortcut(QKeySequence.Redo, self).activated.connect(self._redo_edit)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo_edit)
        return self.speler

    def _bouw_datapaneel(self):
        paneel = QSplitter(Qt.Vertical)

        tabel_groep = QGroupBox("Afzethoeken")
        tv = QVBoxLayout(tabel_groep)
        self.tabel = QTableWidget(0, 4)
        self.tabel.setHorizontalHeaderLabels(["#", "Tijd (s)", "Been", "Hoek (°)"])
        self.tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.cellClicked.connect(self._klik_op_rij)
        tv.addWidget(self.tabel)

        self.lbl_stats = QLabel("gem — | min — | max —")
        tv.addWidget(self.lbl_stats)

        knoppen = QHBoxLayout()
        self.btn_export = QPushButton("Exporteer CSV")
        self.btn_export.clicked.connect(self._exporteer_csv)
        self.btn_export.setEnabled(False)
        knoppen.addWidget(self.btn_export)
        tv.addLayout(knoppen)

        paneel.addWidget(tabel_groep)

        grafiek_groep = QGroupBox("Hoek per tijd")
        gv = QVBoxLayout(grafiek_groep)
        self.chart = QChart()
        self.chart.legend().hide()
        self.serie_hoek = QLineSeries()
        self.serie_marker = QLineSeries()
        self.chart.addSeries(self.serie_hoek)
        self.chart.addSeries(self.serie_marker)
        self.as_x = QValueAxis()
        self.as_y = QValueAxis()
        self.as_x.setTitleText("tijd (s)")
        self.as_y.setTitleText("hoek (°)")
        self.chart.addAxis(self.as_x, Qt.AlignBottom)
        self.chart.addAxis(self.as_y, Qt.AlignLeft)
        self.serie_hoek.attachAxis(self.as_x)
        self.serie_hoek.attachAxis(self.as_y)
        self.serie_marker.attachAxis(self.as_x)
        self.serie_marker.attachAxis(self.as_y)
        chart_view = QChartView(self.chart)
        gv.addWidget(chart_view)
        paneel.addWidget(grafiek_groep)

        paneel.setStretchFactor(0, 2)
        paneel.setStretchFactor(1, 1)
        return paneel

    # ── Voortgangsbalk (niet-blokkerend, blijft over paginawissels heen staan) ──
    def _bouw_voortgangsbalk(self):
        """Een dunne balk onderin met label + voortgang en (voor batch) een stopknop.
        Vervangt de vroegere modale voortgangsdialoog die het hele venster blokkeerde."""
        balk = QWidget()
        h = QHBoxLayout(balk)
        h.setContentsMargins(10, 4, 10, 4)
        self.lbl_voortgang = QLabel("")
        self.bar_voortgang = QProgressBar()
        self.bar_voortgang.setRange(0, 100)
        self.bar_voortgang.setFixedWidth(260)
        self.btn_voortgang_stop = QPushButton("Stop na deze video")
        self.btn_voortgang_stop.clicked.connect(self._batch_stop_gevraagd)
        self.btn_voortgang_stop.hide()
        h.addWidget(self.lbl_voortgang, stretch=1)
        h.addWidget(self.bar_voortgang)
        h.addWidget(self.btn_voortgang_stop)
        balk.hide()
        return balk

    def _toon_voortgangsbalk(self, tekst, met_stop=False):
        self.lbl_voortgang.setText(tekst)
        self.bar_voortgang.setRange(0, 100)
        self.bar_voortgang.setValue(0)
        self.btn_voortgang_stop.setEnabled(True)
        self.btn_voortgang_stop.setVisible(met_stop)
        self.voortgang_balk.show()

    def _verberg_voortgangsbalk(self):
        self.voortgang_balk.hide()

    def _zet_bezig(self, bezig):
        """Schakelt tijdens een lopende (batch-)analyse de knoppen uit die met de worker
        kunnen botsen (een tweede analyse/batch starten, of de schaatser weggooien onder
        wie straks wordt opgeslagen). Openen/afspelen blijft bewust bruikbaar, zodat er
        gebrowst kan worden terwijl de analyse draait."""
        self._bezig = bezig
        self.btn_nieuwe_analyse.setEnabled(not bezig)
        self.btn_batch_analyse.setEnabled(not bezig)
        if bezig:
            self.btn_verwijder_schaatser.setEnabled(False)
        else:
            self._vernieuw_analyses()   # selectie-afhankelijke knoppen terug in hun stand

    # ── Bibliotheek (fase 1) ─────────────────────────────────────────────
    def _zet_bibliotheek(self, pad):
        """Opent (of maakt) de bibliotheek op `pad` en vult de lijsten. Faalt het pad
        (bv. verdwenen netwerkmap), dan valt de app terug op de standaardmap."""
        try:
            schaats_db.open_db(pad)
        except schaats_db.BibliotheekTeNieuw as e:
            # Gedeelde cloudmap waarin een collega met een nieuwere app heeft geschreven:
            # niet aanraken (schrijven zou z'n schema kunnen slopen), wel duidelijk melden.
            QMessageBox.critical(
                self, "Bibliotheek is nieuwer dan deze app",
                f"{e}\n\nMap:\n{pad}\n\n"
                "Er wordt zolang met de standaard-bibliotheekmap gewerkt.")
            standaard = schaats_db.standaard_bibliotheek()
            if pad != standaard:
                return self._zet_bibliotheek(standaard)
            raise
        except Exception as e:
            QMessageBox.critical(
                self, "Bibliotheek",
                f"Kan de bibliotheek niet openen in:\n{pad}\n\n{e}")
            standaard = schaats_db.standaard_bibliotheek()
            if pad != standaard:
                return self._zet_bibliotheek(standaard)
            raise
        self.bieb = pad
        self.lbl_bieb.setText(pad)
        self._waarschuw_conflictkopieen()
        self._vernieuw_schaatsers()
        # Tijdens het opstarten de traagste stap apart melden: bij een nieuwe opname leest
        # de scan de videometa, en op een cloudmap kan dat seconden duren.
        self._melding("Opnames scannen...")
        self._vernieuw_opnames()      # fase 8: werklijst met nog te knippen opnames

    def _waarschuw_conflictkopieen(self):
        """Fase 4: waarschuwt als de cloudsync naast schaats.db conflictkopieën van de
        database heeft achtergelaten (zie schaats_db.detecteer_conflictkopieen)."""
        try:
            kopieen = schaats_db.detecteer_conflictkopieen(self.bieb)
        except Exception:
            return
        if not kopieen:
            return
        QMessageBox.warning(
            self, "Mogelijke conflictkopie van de database",
            "In de bibliotheekmap staan naast 'schaats.db' nog andere database­bestanden:\n\n"
            "• " + "\n• ".join(kopieen) + "\n\n"
            "Zulke kopieën ontstaan als de cloudsync (Google Drive/OneDrive/Dropbox) een "
            "conflict maakt doordat twee trainers bijna tegelijk schreven. 'schaats.db' "
            "blijft de actieve bibliotheek; bekijk de kopie(ën) en verwijder of hernoem ze "
            "om verwarring te voorkomen.")

    def _vernieuw_bibliotheek(self):
        """Fase 4: leest de bibliotheek opnieuw van schijf, zodat analyses van collega's
        (via de gedeelde cloudmap) zichtbaar worden zonder herstart."""
        self._waarschuw_conflictkopieen()
        self._vernieuw_schaatsers()
        self._vernieuw_opnames()      # ook nieuwe opnames van collega's oppikken
        self.statusBar().showMessage("Bibliotheek vernieuwd.", 4000)

    def _toon_trainer_naam(self):
        self.lbl_trainer.setText(
            f"jij: {self.trainer_naam}" if self.trainer_naam else "jij: (naam niet ingesteld)")

    def _kies_trainer_naam(self):
        naam, ok = QInputDialog.getText(
            self, "Jouw naam",
            "Je naam (wordt bij nieuwe analyses bewaard als 'aangemaakt door'):",
            text=self.trainer_naam)
        if not ok:
            return
        self.trainer_naam = naam.strip()
        cfg = schaats_db.laad_config()
        cfg["trainer_naam"] = self.trainer_naam
        schaats_db.bewaar_config(cfg)
        self._toon_trainer_naam()

    def _kies_bibliotheekmap(self):
        pad = QFileDialog.getExistingDirectory(self, "Kies bibliotheekmap", self.bieb or "")
        if not pad:
            return
        cfg = schaats_db.laad_config()
        cfg["bibliotheek_pad"] = pad
        schaats_db.bewaar_config(cfg)
        self._zet_bibliotheek(pad)

    def _geselecteerde_schaatser_id(self):
        item = self.lijst_schaatsers.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _schaatser_naam(self, schaatser_id):
        """Naam bij een schaatser-id, of "" als die er niet (meer) is."""
        if schaatser_id is None:
            return ""
        s = next((x for x in schaats_db.lijst_schaatsers(self.bieb)
                  if x["id"] == schaatser_id), None)
        return s["naam"] if s else ""

    def _geselecteerde_analyse_id(self):
        rij = self.tabel_analyses.currentRow()
        if rij < 0:
            return None
        item = self.tabel_analyses.item(rij, 0)
        return item.data(Qt.UserRole) if item else None

    def _geselecteerde_titel(self):
        item = self.tabel_analyses.item(self.tabel_analyses.currentRow(), 1)
        return item.text() if item else ""

    def _vernieuw_schaatsers(self, selecteer_id=None):
        """Herlaadt de schaatserslijst uit de database (en daarmee de analysetabel)."""
        if selecteer_id is None:
            selecteer_id = self._geselecteerde_schaatser_id()
        self.lijst_schaatsers.blockSignals(True)
        self.lijst_schaatsers.clear()
        selecteer_rij = None
        schaatsers = schaats_db.lijst_schaatsers(self.bieb)
        # Vergelijken werkt over schaatsers heen, dus niet aan de selectie hangen maar aan
        # "staat er ergens een analyse".
        self.btn_vergelijk.setEnabled(any(s["aantal_analyses"] for s in schaatsers))
        for rij, s in enumerate(schaatsers):
            tekst = s["naam"]
            if s["geboortejaar"]:
                tekst += f" ({s['geboortejaar']})"
            n = s["aantal_analyses"]
            tekst += f"  ·  {n} analyse{'s' if n != 1 else ''}"
            item = QListWidgetItem(tekst)
            item.setData(Qt.UserRole, s["id"])
            item.setData(Qt.UserRole + 1, s["naam"])
            if s["notities"]:
                item.setToolTip(s["notities"])
            self.lijst_schaatsers.addItem(item)
            if s["id"] == selecteer_id:
                selecteer_rij = rij
        self.lijst_schaatsers.blockSignals(False)
        if selecteer_rij is None and self.lijst_schaatsers.count():
            selecteer_rij = 0
        if selecteer_rij is not None:
            self.lijst_schaatsers.setCurrentRow(selecteer_rij)  # triggert _vernieuw_analyses
        else:
            self._vernieuw_analyses()

    def _maak_rij_knoppen(self, analyse_id, titel):
        """De vier per-analyse acties als widget voor de laatste tabelkolom.
        Elke knop draagt zijn eigen analyse-id mee (default-argument in de lambda —
        anders zou de laatste lus-waarde gelden voor álle rijen), zodat een klik werkt
        op de rij waar hij op staat en niet op de toevallige tabelselectie."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)
        knoppen = (
            ("Openen", "Deze analyse openen op de weergavepagina.",
             lambda _=False, a=analyse_id: self._open_analyse_uit_bibliotheek(a)),
            ("ℹ Info...", "Met welke appversie, backend en instellingen is deze "
                          "analyse gemaakt?",
             lambda _=False, a=analyse_id: self._toon_analyse_info(a)),
            ("Hernoemen...", "Deze analyse een andere titel geven.",
             lambda _=False, a=analyse_id, t=titel: self._hernoem_analyse(a, t)),
            ("Verwijderen", "Deze analyse verwijderen, inclusief video en landmarks.",
             lambda _=False, a=analyse_id, t=titel: self._verwijder_analyse(a, t)),
        )
        for tekst, tip, slot in knoppen:
            b = QPushButton(tekst)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        return w

    def _vernieuw_analyses(self):
        """Vult de analysetabel voor de geselecteerde schaatser uit de events-cache
        (geen npz/video nodig — daarom is de bibliotheek direct snel)."""
        sid = self._geselecteerde_schaatser_id()
        self.tabel_analyses.setRowCount(0)
        if sid is not None:
            analyses = schaats_db.lijst_analyses(self.bieb, sid)
            self.tabel_analyses.setRowCount(len(analyses))
            for rij, a in enumerate(analyses):
                # Herkomst in de tooltip: wie hem maakte (fase 4) en met welke appversie —
                # zo is zonder openen te zien of een analyse nog met oude code is gedraaid.
                door = (a.get("aangemaakt_door") or "").strip()
                versie = ((a.get("instellingen") or {}).get("app_versie") or "").strip()
                tip = "\n".join(r for r in (f"Aangemaakt door {door}" if door else "",
                                            f"Appversie: {versie}" if versie else "") if r)
                for kolom, tekst in enumerate(
                        [a["datum"], a["titel"], _duur_tekst(a)]):
                    item = QTableWidgetItem(tekst)
                    if kolom == 0:
                        item.setData(Qt.UserRole, a["id"])
                    if kolom == 1 and tip:
                        item.setToolTip(tip)
                    self.tabel_analyses.setItem(rij, kolom, item)
                knoppen = self._maak_rij_knoppen(a["id"], a["titel"])
                self.tabel_analyses.setCellWidget(rij, 3, knoppen)
                # Rijhoogte volgt de knoppen niet vanzelf; zonder dit worden ze afgeknepen.
                self.tabel_analyses.setRowHeight(rij, knoppen.sizeHint().height() + 4)
        self.btn_bewerk_schaatser.setEnabled(sid is not None)
        self.btn_verwijder_schaatser.setEnabled(sid is not None)

    def _nieuwe_schaatser(self):
        dlg = SchaatserDialog(self)
        if dlg.exec() != QDialog.Accepted or not dlg.naam:
            return
        sid = schaats_db.maak_schaatser(self.bieb, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._vernieuw_schaatsers(selecteer_id=sid)

    def _bewerk_schaatser(self):
        sid = self._geselecteerde_schaatser_id()
        if sid is None:
            return
        s = next((x for x in schaats_db.lijst_schaatsers(self.bieb) if x["id"] == sid), None)
        if s is None:
            return
        dlg = SchaatserDialog(self, naam=s["naam"], geboortejaar=s["geboortejaar"],
                              notities=s["notities"])
        if dlg.exec() != QDialog.Accepted or not dlg.naam:
            return
        schaats_db.wijzig_schaatser(self.bieb, sid, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._vernieuw_schaatsers(selecteer_id=sid)

    def _verwijder_schaatser(self):
        sid = self._geselecteerde_schaatser_id()
        if sid is None:
            return
        naam = self.lijst_schaatsers.currentItem().data(Qt.UserRole + 1)
        analyses = schaats_db.lijst_analyses(self.bieb, sid)
        tekst = f"Schaatser '{naam}' verwijderen?"
        if analyses:
            tekst += (f"\n\nDe {len(analyses)} bijbehorende analyse"
                      f"{'s' if len(analyses) != 1 else ''} (inclusief video's en "
                      "landmarks) worden dan ook verwijderd.")
        tekst += "\n\nDit kan niet ongedaan worden gemaakt."
        if QMessageBox.question(self, "Schaatser verwijderen", tekst) != QMessageBox.Yes:
            return
        ids = {a["id"] for a in analyses}
        if self.analyse_id in ids:
            self._sluit_weergave()   # laat de geopende video los vóór het wissen
        self._sluit_vergelijk_voor(ids)
        schaats_db.verwijder_schaatser(self.bieb, sid)
        self._vernieuw_schaatsers()

    def _hernoem_analyse(self, aid=None, huidig=""):
        # aid/huidig komen van de knop in de rij zelf; zonder die twee valt hij terug op
        # de tabelselectie (bv. een sneltoets die er ooit bij komt).
        if aid is None:
            aid, huidig = self._geselecteerde_analyse_id(), self._geselecteerde_titel()
        if aid is None:
            return
        titel, ok = QInputDialog.getText(self, "Analyse hernoemen", "Nieuwe titel:",
                                         text=huidig)
        if not ok or not titel.strip():
            return
        schaats_db.hernoem_analyse(self.bieb, aid, titel.strip())
        self._vernieuw_analyses()

    def _verwijder_analyse(self, aid=None, titel=""):
        if aid is None:
            aid, titel = self._geselecteerde_analyse_id(), self._geselecteerde_titel()
        if aid is None:
            return
        if QMessageBox.question(
                self, "Analyse verwijderen",
                f"Analyse '{titel}' verwijderen, inclusief de gekopieerde video en "
                "landmarks?\n\nDit kan niet ongedaan worden gemaakt.") != QMessageBox.Yes:
            return
        if aid == self.analyse_id:
            self._sluit_weergave()   # Windows weigert een nog geopende video te wissen
        self._sluit_vergelijk_voor({aid})
        schaats_db.verwijder_analyse(self.bieb, aid)
        self._vernieuw_schaatsers()

    def _sluit_weergave(self):
        """Maakt de weergavepagina leeg en laat het videobestand los (nodig voordat de
        mediamap van de geopende analyse verwijderd kan worden)."""
        self.speler.sluit()
        self.events = []
        self.analyse_id = None
        self.analyse_schaatser_id = None
        self.analyse_schaatser_naam = ""
        self.btn_vergelijk_deze.setEnabled(False)
        self.input_pad = None
        self.tabel.setRowCount(0)
        self.serie_hoek.clear()
        self.serie_marker.clear()
        self.lbl_stats.setText("gem — | min — | max —")
        self.lbl_live.setText("")
        self.btn_export.setEnabled(False)

    # ── Nieuwe analyse + openen ──────────────────────────────────────────
    def _nieuwe_analyse(self):
        schaatsers = schaats_db.lijst_schaatsers(self.bieb)
        if not schaatsers:
            QMessageBox.information(
                self, "Nieuwe analyse",
                "Maak eerst een schaatser aan — elke analyse hoort bij een profiel.")
            return
        dlg = NieuweAnalyseDialog(schaatsers, voorkeur_id=self._geselecteerde_schaatser_id(),
                                  parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.input_pad = dlg.video_pad

        # Modelkeuze is alleen relevant voor de MediaPipe-backend; YOLO gebruikt zijn
        # eigen model (yolo26x-pose.pt) en negeert model_pad.
        heavy = False
        if not IS_YOLO:
            if dlg.chk_heavy.isChecked():
                if os.path.isfile(HEAVY_MODEL):
                    self.model_pad = HEAVY_MODEL
                    heavy = True
                else:
                    QMessageBox.warning(
                        self, "Heavy-model ontbreekt",
                        "pose_landmarker_heavy.task staat niet naast het script.\n\n"
                        "Download het via:\nhttps://storage.googleapis.com/mediapipe-models/"
                        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                        "pose_landmarker_heavy.task\n\nEr wordt nu met het full-model gewerkt.")
                    self.model_pad = STANDAARD_MODEL
            else:
                self.model_pad = STANDAARD_MODEL

            if not os.path.isfile(self.model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Kies pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                self.model_pad = gekozen

        # Eerste frame lezen; delen door doel- en horizon-kiezer.
        frame0 = self._lees_eerste_frame()
        if frame0 is None:
            return

        # Doelschaatser laten kiezen op het eerste frame.
        self.doel_punt = self._kies_doelschaatser(frame0)
        if self.doel_punt is False:      # dialoog afgebroken
            return

        # Perspectiefkalibratie (baanlijnen) óf de klassieke horizon-stap.
        self.perspectief = None
        if dlg.chk_perspectief.isChecked():
            self.perspectief = self._kies_perspectief(frame0)
            if self.perspectief is None:     # dialoog afgebroken
                return
            self.horizon_deg, self.auto_horizon = 0.0, False   # kalibratie vervangt de horizon
        else:
            horizon = self._kies_horizon(frame0)
            if horizon is False:         # dialoog afgebroken
                return
            self.horizon_deg, self.auto_horizon = horizon

        self.smooth_n = dlg.spin_smooth.value()
        self.threshold = dlg.spin_threshold.value()
        self.geen_smoothing = dlg.chk_geen_smoothing.isChecked()
        self.bocht_overslaan = dlg.chk_bocht.isChecked()

        # Wat het .npz níet bevat maar heropenen wél nodig heeft/wil documenteren.
        # (De bocht-vlag per frame zit wél in het npz; dit is puur de instelling.)
        instellingen = {
            "smooth_n": self.smooth_n,
            "threshold": self.threshold,
            "smooth_landmarks": not self.geen_smoothing,
            "bocht_overslaan": self.bocht_overslaan,
            "doel_punt": list(self.doel_punt) if self.doel_punt else None,
            "horizon_deg": self.horizon_deg,
            "auto_horizon": self.auto_horizon,
            "heavy": heavy,
            "backend_naam": BACKEND_NAAM,
            "perspectief_gebruikt": self.perspectief is not None,
            # De kalibratie-invoer (lijnen + parameters), zodat heropenen de correctie
            # herberekent i.p.v. hem te laten verdampen — en zodat een volgende analyse
            # uit dezelfde camerastand hem kan overnemen.
            "perspectief": self.perspectief.naar_dict() if self.perspectief else None,
        }
        self._pending_opslag = {"schaatser_id": dlg.schaatser_id, "titel": dlg.titel,
                                "instellingen": instellingen}

        # Op de bibliotheek blijven; de voortgangsbalk loopt onderin en bij afronden
        # springt de weergave vanzelf naar het resultaat (zolang er niets anders geopend is).
        self._start_analyse()

    def _kies_perspectief(self, frame0):
        """Perspectiefkalibratie voor één video: eerst aanbieden om er een uit een
        eerdere analyse over te nemen (zelfde camerastand), dan de `KalibratieKiezer`
        — voorgevuld als er iets overgenomen is, zodat controleren en corrigeren
        dezelfde handeling blijft. Retourneert een PerspectiefConfig, of None bij
        afbreken.

        Hergebruik is hier geen gemak maar een meetkundige voorwaarde: analyses die je
        onderling wilt vergelijken moeten op dezelfde kalibratie rusten, anders meet je
        de spreiding tussen zeven keer natrekken in plaats van het effect van de
        correctie.
        """
        h, w = frame0.shape[:2]
        invoer = config = None
        try:
            eerdere = schaats_db.lijst_kalibraties(self.bieb, beeld_w=w, beeld_h=h)
        except Exception:
            eerdere = []
        if eerdere:
            keuzes = ["Nieuwe kalibratie (lijnen zelf natrekken)"]
            for k in eerdere[:15]:
                datum = k["datum"] or ""
                naam = k["schaatser"] or "?"
                extra = f" — {k['notitie']}" if k["notitie"] else ""
                keuzes.append(f"{k['titel']} ({naam}, {datum}){extra}")
            keuze, ok = QInputDialog.getItem(
                self, "Kalibratie overnemen?",
                "Er staan al kalibraties in de bibliotheek voor beeld van "
                f"{w}×{h}. Een kalibratie hoort bij één camerastand, dus clips uit\n"
                "dezelfde opname horen dezelfde te gebruiken — alleen dan zijn hun "
                "hoeken onderling vergelijkbaar.\n\nOvernemen van:",
                keuzes, 0, False)
            if not ok:
                return None
            idx = keuzes.index(keuze)
            if idx > 0:
                try:
                    config = PerspectiefConfig.uit_dict(eerdere[idx - 1]["perspectief"])
                    invoer = config.invoer
                except Exception as e:
                    QMessageBox.warning(
                        self, "Kalibratie onbruikbaar",
                        f"Die opgeslagen kalibratie is niet te herberekenen:\n\n{e}\n\n"
                        "Trek de lijnen opnieuw na.")
                    config = invoer = None

        kdlg = KalibratieKiezer(frame0, self, invoer=invoer, config=config)
        if kdlg.exec() != QDialog.Accepted:
            return None
        return kdlg.perspectief

    def _laad_analyse_data(self, analyse_id):
        """
        Laadt één analyse uit de bibliotheek en herberekent de afgeleiden met de ópgeslagen
        instellingen (de fase 0-naad). Retourneert een dict, of None als het niet lukt (dan
        is de melding al getoond).

        Raakt bewust géén MainWindow-state aan — `smooth_n`/`threshold` komen als waarde
        terug i.p.v. op self gezet te worden. Zo kan de vergelijkpagina analyses laden
        zonder de instellingen van de geopende analyse (die `_na_edit` gebruikt) te
        overschrijven.
        """
        try:
            data = schaats_db.laad_analyse(self.bieb, analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Fout bij openen",
                                 f"Kan de analyse niet laden:\n\n{e}")
            return None

        sync = schaats_db.video_sync_status(data["video_pad"], data["meta"].get("video_bytes"))
        if sync == "ontbreekt":
            QMessageBox.warning(
                self, "Video ontbreekt",
                "Het videobestand van deze analyse staat (nog) niet op schijf — "
                "mogelijk is de cloudmap nog aan het synchroniseren.\n\n"
                "Probeer het later opnieuw.")
            return None
        if sync == "onvolledig":
            QMessageBox.warning(
                self, "Video nog niet gesynchroniseerd",
                "Het videobestand is nog niet volledig binnengehaald uit de gedeelde "
                "cloudmap (het is kleiner dan bij het opslaan).\n\n"
                "Wacht tot de synchronisatie klaar is en probeer het opnieuw.")
            return None

        info, resultaten = data["info"], data["resultaten"]
        inst = data["meta"]["instellingen"]
        smooth_n = int(inst.get("smooth_n", 5))
        threshold = float(inst.get("threshold", 0.015))

        # Perspectiefkalibratie terug uit de instellingen (de lijnen zijn bewaard, de
        # matrices worden hier herberekend). Faalt dat, dan gaat het openen gewoon door
        # zónder correctie — met een melding, want de hoeken wijken dan af.
        perspectief, persp_fout = None, None
        if inst.get("perspectief"):
            try:
                perspectief = PerspectiefConfig.uit_dict(inst["perspectief"])
            except Exception as e:
                persp_fout = str(e)

        # De horizon zit al per frame in het .npz; alleen de afgeleiden herberekenen.
        verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                           perspectief=perspectief)
        events = segmenteer_afzetten(resultaten)
        # De events-cache is een momentopname van de berekening bij het opslaan; wat je
        # hier ziet is vers herberekend. Bijwerken houdt de lijstweergave (aantal afzetten,
        # gemiddelde hoek) gelijk aan de tabel — ook voor analyses van vóór een
        # algoritme-verbetering. Mislukt het (bv. DB even op slot in de cloudmap), dan is
        # dat geen reden om het openen af te breken.
        try:
            schaats_db.ververs_events_cache(self.bieb, analyse_id, events)
        except Exception:
            pass

        return {
            "meta": data["meta"],
            "info": info,
            "resultaten": resultaten,
            "events": events,
            "video_pad": data["video_pad"],
            "titel": data["meta"]["titel"],
            "smooth_n": smooth_n,
            "threshold": threshold,
            "perspectief_gebruikt": bool(inst.get("perspectief_gebruikt")),
            "perspectief": perspectief,
            "perspectief_fout": persp_fout,
        }

    def _open_analyse_uit_bibliotheek(self, analyse_id=None):
        """Opent een opgeslagen analyse op de weergavepagina."""
        if analyse_id is None:
            analyse_id = self._geselecteerde_analyse_id()
        if analyse_id is None:
            return
        data = self._laad_analyse_data(analyse_id)
        if data is None:
            return

        # Deze twee stuurt de skelet-editor aan (_na_edit herberekent ermee).
        self.smooth_n = data["smooth_n"]
        self.threshold = data["threshold"]
        self.perspectief = data["perspectief"]
        if data["perspectief_fout"]:
            QMessageBox.warning(
                self, "Kalibratie kon niet herberekend worden",
                "Deze analyse is met perspectiefcorrectie gedraaid, maar de bewaarde "
                "kalibratie levert nu geen geldige camerastand op:\n\n"
                f"{data['perspectief_fout']}\n\n"
                "De hoeken zijn zonder correctie herberekend en wijken dus af.")
        elif data["perspectief_gebruikt"] and self.perspectief is None:
            QMessageBox.information(
                self, "Zonder perspectiefcorrectie",
                "Deze analyse is destijds met perspectiefcorrectie gedraaid, van vóór "
                "het bewaren van de kalibratie. De hoeken zijn nu zonder correctie "
                "herberekend en kunnen dus afwijken.")

        self.input_pad = data["video_pad"]
        self.analyse_id = analyse_id
        self.analyse_schaatser_id = data["meta"]["schaatser_id"]
        self.analyse_schaatser_naam = self._schaatser_naam(self.analyse_schaatser_id)
        # De gebruiker bekijkt nu bewust deze analyse; een op de achtergrond lopende
        # analyse mag hem hier straks niet uit wegrukken.
        self._auto_toon_klaar = False
        self.stack.setCurrentWidget(self.pagina_analyse)
        self._toon_resultaten(data["info"], data["resultaten"], data["events"],
                              bron=data["titel"])

    # ── Vergelijken (twee analyses naast elkaar) ─────────────────────────
    def _vergelijk_schaatsers(self):
        """Opent de vergelijkpagina; vraagt eerst om de analyses die nog ontbreken."""
        if not any(s["aantal_analyses"] for s in schaats_db.lijst_schaatsers(self.bieb)):
            QMessageBox.information(
                self, "Nog niets te vergelijken",
                "Er staan nog geen analyses in de bibliotheek.")
            return
        for kant in (self.kant_links, self.kant_rechts):
            if kant.heeft_analyse():
                continue          # al gevuld (bv. bij terugkeren) — laten staan
            if not self._kies_vergelijk_kant(kant):
                break             # afgebroken: met wat er ligt doorgaan
        if not (self.kant_links.heeft_analyse() or self.kant_rechts.heeft_analyse()):
            return                # helemaal niets gekozen → niet naar een lege pagina
        self.stack.setCurrentWidget(self.pagina_vergelijk)
        self.statusBar().showMessage(
            "Vergelijken: zet per kant een sync-punt op dezelfde fase van de slag en "
            "druk op 'Start alles'.")

    def _toon_analyse_info(self, analyse_id=None):
        """Info over een analyse: appversie/commit, backend, datum, maker en de
        belangrijkste instellingen. Gedeeld door de knop op de weergavepagina (de geopende
        analyse) en die op de startpagina (de rij die in de bibliotheek geselecteerd is).
        De meta wordt hier vers uit de DB gehaald (goedkoop — `analyse_meta` leest het npz
        niet), zodat er geen kopie op MainWindow hoeft te leven die na een verse analyse
        én na het openen bijgehouden moet worden."""
        if analyse_id is None:
            analyse_id = self.analyse_id
        if analyse_id is None:
            return
        try:
            meta = schaats_db.analyse_meta(self.bieb, analyse_id)
        except Exception as e:
            QMessageBox.warning(self, "Info", f"Kon de analysegegevens niet lezen:\n{e}")
            return
        AnalyseInfoDialog(meta, self._schaatser_naam(meta.get("schaatser_id")),
                          parent=self).exec()

    def _vergelijk_met_deze(self):
        """Vanaf de weergavepagina rechtstreeks vergelijken: de geopende analyse gaat
        links, en voor rechts wordt (als daar nog niets bruikbaars staat) meteen om een
        analyse gevraagd. Scheelt de omweg via de bibliotheek."""
        if self.analyse_id is None:
            return
        self._pauzeer_alles()
        if not self._zet_vergelijk_kant(self.kant_links, self.analyse_id,
                                        self.analyse_schaatser_naam):
            return          # melding is al getoond
        # Een andere analyse rechts blijft staan (inclusief sync-punt); dezelfde analyse
        # twee keer naast elkaar heeft geen zin.
        if (not self.kant_rechts.heeft_analyse()
                or self.kant_rechts.analyse_id == self.analyse_id):
            self._kies_vergelijk_kant(self.kant_rechts,
                                      voorkeur_id=self.analyse_schaatser_id)
        self.stack.setCurrentWidget(self.pagina_vergelijk)
        self.statusBar().showMessage(
            "Vergelijken: zet per kant een sync-punt op dezelfde fase van de slag en "
            "druk op 'Start alles'.")

    def _kies_vergelijk_kant(self, kant, voorkeur_id=None):
        """Laat één kant een analyse kiezen en laadt die. True als het gelukt is."""
        if voorkeur_id is None:
            voorkeur_id = self._geselecteerde_schaatser_id()
        dlg = AnalyseKiezer(self.bieb, titel=f"{kant.naam}: kies analyse",
                            voorkeur_schaatser_id=voorkeur_id, parent=self)
        if dlg.exec() != QDialog.Accepted or dlg.analyse_id is None:
            return False
        return self._zet_vergelijk_kant(kant, dlg.analyse_id, dlg.schaatser_naam)

    def _zet_vergelijk_kant(self, kant, analyse_id, schaatser_naam):
        """Laadt een analyse uit de bibliotheek in één kant. True als het gelukt is.

        Bewust opnieuw laden i.p.v. de resultatenlijst van de weergavepagina delen: de
        skelet-editor muteert die objecten in place, en elke speler heeft z'n eigen
        VideoCapture."""
        self._stop_alles()
        data = self._laad_analyse_data(analyse_id)
        if data is None:
            return False
        kant.toon(analyse_id, schaatser_naam, data)
        return True

    def _sluit_vergelijk_voor(self, ids):
        """Laat de video's van deze analyses los op de vergelijkpagina — Windows weigert
        een nog geopende video te wissen, en rmtree faalt dan stílzwijgend."""
        for kant in (self.kant_links, self.kant_rechts):
            if kant.analyse_id in ids:
                kant.leeg()

    def _beide_naar_sync(self):
        self._stop_alles()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for kant in (self.kant_links, self.kant_rechts):
                kant.naar_sync()
        finally:
            QApplication.restoreOverrideCursor()

    def _start_alles(self):
        """
        Beide video's tegelijk afspelen, aangestuurd door één masterklok.

        Twee losse frame-timers lopen binnen enkele seconden uit de pas — twee decodes +
        overlay + rescale kosten meer dan één timerinterval — en zouden bij verschillende
        fps sowieso niet kloppen. Daarom rekent `_alles_tick` het doelframe per kant uit de
        verstreken wandkloktijd × de eigen fps: zelfcorrigerend, dus geen drift.
        """
        kanten = [k for k in (self.kant_links, self.kant_rechts) if k.heeft_analyse()]
        if not kanten:
            QMessageBox.information(self, "Niets te starten",
                                    "Kies eerst voor beide kanten een analyse.")
            return
        self._pauzeer_alles()
        if self.chk_vanaf_sync.isChecked():
            # Terugspoelen heropent de video en spoelt sequentieel; die wachttijd zit zo
            # eenmalig vooraan in plaats van in de eerste tick.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for kant in kanten:
                    kant.naar_sync()
            finally:
                QApplication.restoreOverrideCursor()
        self._alles_lopend = [(k, max(0, k.speler.huidige_idx)) for k in kanten]
        self._alles_factor = self.combo_alles_snelheid.currentData() or 1.0
        self._alles_t0 = time.monotonic()
        self._alles_timer.start(ALLES_TICK_MS)

    def _alles_tick(self):
        t = (time.monotonic() - self._alles_t0) * self._alles_factor
        klaar = True
        for kant, basis in self._alles_lopend:
            info = kant.speler.video_info
            if info is None or not kant.speler.resultaten:
                continue
            laatste = len(kant.speler.resultaten) - 1
            doel = basis + int(round(t * (info.fps or 30.0)))
            kant.speler.ga_naar(min(doel, laatste))
            if doel < laatste:
                klaar = False
        if klaar:
            self._stop_alles()

    def _stop_alles(self):
        if self._alles_timer.isActive():
            self._alles_timer.stop()
        self._alles_lopend = []

    def _zet_alles_snelheid(self, _idx=None):
        """De gedeelde snelheid van de vergelijkpagina toepassen.

        Beide kanten krijgen dezelfde factor — ook voor los afspelen, want twee video's
        op verschillend tempo naast elkaar zijn niet te vergelijken. Draait de masterklok,
        dan wordt die opnieuw geijkt vanaf de huidige stand, anders zou het doelframe
        terugspringen."""
        idx = self.combo_alles_snelheid.currentIndex()
        for kant in (self.kant_links, self.kant_rechts):
            kant.speler.combo_snelheid.setCurrentIndex(idx)   # herstart een lopende timer
        if not self._alles_timer.isActive():
            return
        self._alles_lopend = [(k, max(0, k.speler.huidige_idx))
                              for k, _ in self._alles_lopend]
        self._alles_factor = self.combo_alles_snelheid.currentData() or 1.0
        self._alles_t0 = time.monotonic()

    def _lees_eerste_frame(self, pad=None):
        """Leest het eerste frame van de gekozen video, of None bij een fout.
        Zonder `pad` de huidige `self.input_pad`; met `pad` een willekeurige video
        (gebruikt door de batch-verzamellus voor elke clip apart)."""
        cap = cv2.VideoCapture(pad or self.input_pad)
        ret, frame0 = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "Fout", "Kan het eerste frame niet lezen.")
            return None
        return frame0

    def _kies_doelschaatser(self, frame0):
        """Toont het eerste frame in een kiezer. Retourneert (x,y), None, of False (afgebroken)."""
        dlg = DoelKiezer(frame0, self)
        if dlg.exec() != QDialog.Accepted:
            return False
        return dlg.doel_punt

    def _kies_horizon(self, frame0):
        """
        Laat de ijslijn/kanteling instellen. Retourneert (graden, auto_per_frame) of
        False (afgebroken).
        """
        dlg = HorizonKiezer(frame0, self)
        if dlg.exec() != QDialog.Accepted:
            return False
        return dlg.horizon_deg, dlg.auto_per_frame

    def _terug_naar_start(self):
        # pauzeren gebeurt via de stack.currentChanged-haak (_pauzeer_alles)
        self._vernieuw_schaatsers()   # nieuwe/gewijzigde analyses direct zichtbaar
        self.stack.setCurrentWidget(self.pagina_start)

    def _waarschuw_backend_terugval(self):
        """Meldt (één keer) dat de YOLO-backend niet geladen kon worden en er dus met
        MediaPipe gemeten wordt — een andere detector geeft andere hoeken, dus dat mag
        niet onopgemerkt blijven. Het warmdraaien start bij het tonen van het venster, dus
        op het moment dat hier een analyse begint is de uitkomst allang bekend.

        In een gebundelde .exe is er geen MediaPipe om op terug te vallen; daar is het
        geen waarschuwing maar een blokkade, en zegt de melding dat ook."""
        if not BACKEND_FOUT or self._backend_gemeld:
            return
        self._backend_gemeld = True
        if is_bevroren():
            # In het gebundelde pakket is er geen tweede backend om op terug te vallen:
            # er valt nu niets te meten (bibliotheek en opnames bekijken werken wel).
            QMessageBox.critical(
                self, "Analyse-backend niet beschikbaar",
                "De meegeleverde analyse-backend liet zich niet laden:\n\n"
                f"{BACKEND_FOUT}\n\n"
                "Er kan nu niet geanalyseerd worden. De bibliotheek openen en opnames "
                "bekijken werkt wel. Geef deze melding door aan de beheerder van de app."
                # Het logboek is alleen iets waard als de gebruiker weet waar het staat.
                + (f"\n\nHet volledige logboek staat in:\n{LOGPAD}" if LOGPAD else ""))
            return
        QMessageBox.warning(
            self, "YOLO-backend niet beschikbaar",
            "torch/ultralytics is wel geïnstalleerd, maar liet zich niet laden:\n\n"
            f"{BACKEND_FOUT}\n\n"
            "De analyse draait daarom met de MediaPipe-backend. Die meet minder "
            "nauwkeurig, dus vergelijk deze analyse niet zomaar met eerdere.")

    def _start_analyse(self):
        self._waarschuw_backend_terugval()
        self.speler.zet_besturing_actief(False)
        self.btn_export.setEnabled(False)
        self._auto_toon_klaar = True          # nog niets anders geopend → resultaat straks tonen
        self._analyse_waarschuwingen = []     # meldingen uit de analyse zelf (na afloop tonen)
        self._zet_bezig(True)                 # geen tweede worker/botsende bewerking eroverheen
        self._toon_voortgangsbalk("Video analyseren...")

        opslag = self._pending_opslag or {}
        self.worker = AnalyseWorker(self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                                    doel_punt=self.doel_punt, horizon_deg=self.horizon_deg,
                                    auto_horizon=self.auto_horizon,
                                    smooth_landmarks=not self.geen_smoothing,
                                    perspectief=self.perspectief,
                                    bieb=self.bieb,
                                    schaatser_id=opslag.get("schaatser_id"),
                                    titel=opslag.get("titel"),
                                    instellingen=opslag.get("instellingen"),
                                    backend=BACKEND_NAAM,
                                    aangemaakt_door=self.trainer_naam,
                                    bocht=self.bocht_overslaan)
        self.worker.voortgang.connect(self._analyse_voortgang)
        self.worker.status.connect(self._analyse_status)
        self.worker.opslag_fout.connect(self._opslag_fout)
        self.worker.waarschuwing.connect(self._analyse_waarschuwing)
        self.worker.klaar.connect(self._analyse_klaar)
        self.worker.fout.connect(self._analyse_fout)
        self.worker.start()

    def _analyse_voortgang(self, frame_nr, totaal):
        if totaal > 0:
            self.bar_voortgang.setRange(0, 100)
            self.bar_voortgang.setValue(int(frame_nr / totaal * 100))
        self.lbl_voortgang.setText(f"Video analyseren... ({frame_nr}/{totaal})")

    def _analyse_status(self, tekst):
        # Busy-fase zonder bekende duur (videokopie naar de bibliotheek).
        self.bar_voortgang.setRange(0, 0)
        self.lbl_voortgang.setText(tekst)

    def _analyse_waarschuwing(self, tekst):
        """Melding uit de analyse zelf (bv. 'je klik raakte niemand, nu volgt de grootste
        beweger'). Verzamelen i.p.v. meteen tonen: een modale box halverwege zou de
        gebruiker midden in een lange analyse overvallen. `_analyse_klaar`/`_analyse_fout`
        legen de lijst weer."""
        self._analyse_waarschuwingen.append(tekst)

    def _toon_analyse_waarschuwingen(self):
        meldingen = getattr(self, "_analyse_waarschuwingen", [])
        self._analyse_waarschuwingen = []
        if meldingen:
            QMessageBox.warning(self, "Let op bij deze analyse", "\n\n".join(meldingen))

    def _meld_bocht(self, resultaten):
        """Zeg hoeveel van de clip als bocht is overgeslagen. Is dat álles, dan is een
        lege tabel geen meting maar een verkeerd gekozen clip (of een te strenge
        drempel) — dat verdient een echte waarschuwing i.p.v. '0 afzetten'."""
        n = len(resultaten)
        if not n:
            return
        bocht = sum(1 for r in resultaten if r.bocht)
        if bocht == n:
            self._analyse_waarschuwingen.append(
                "In deze video staat de schaatser nergens frontaal in beeld — hij is dus "
                "volledig als 'bocht' aangemerkt en er zijn geen afzetten gemeten.\n"
                "Klopt dat niet, analyseer de video dan opnieuw met 'Bocht overslaan' uit.")
        elif bocht:
            self.statusBar().showMessage(
                f"{bocht} van {n} frames ({bocht / n:.0%}) overgeslagen: bocht.", 10000)

    def _opslag_fout(self, bericht):
        if self._afsluiten:
            return
        QMessageBox.warning(
            self, "Niet opgeslagen in bibliotheek",
            "De analyse is gelukt, maar kon niet in de bibliotheek worden opgeslagen:\n\n"
            f"{bericht}\n\nDe resultaten zijn nu wel zichtbaar, maar niet bewaard.")

    def _analyse_fout(self, bericht):
        if self._afsluiten:
            return
        self._verberg_voortgangsbalk()
        self._zet_bezig(False)
        self._pending_opslag = None
        self._analyse_waarschuwingen = []      # de fout zegt al genoeg
        QMessageBox.critical(self, "Fout bij analyseren", bericht)
        self.speler.zet_besturing_actief(False)

    def _analyse_klaar(self, info, resultaten, events, analyse_id):
        if self._afsluiten:
            return          # signaal van vlak vóór het afsluiten: niets meer opbouwen
        self._verberg_voortgangsbalk()
        self._zet_bezig(False)
        opslag = self._pending_opslag or {}
        self._pending_opslag = None
        self._meld_bocht(resultaten)
        self._toon_analyse_waarschuwingen()

        if not self._auto_toon_klaar:
            # De gebruiker is intussen met een andere analyse bezig → niet uit z'n
            # weergave rukken; alleen de lijst verversen en het melden.
            self._vernieuw_schaatsers()
            titel = opslag.get("titel") or "analyse"
            self.statusBar().showMessage(
                f"Analyse '{titel}' klaar en opgeslagen in de bibliotheek.", 10000)
            return

        self.analyse_id = analyse_id
        self.analyse_schaatser_id = opslag.get("schaatser_id")
        self.analyse_schaatser_naam = self._schaatser_naam(self.analyse_schaatser_id)
        if analyse_id is not None:
            # Weergave leest voortaan de bibliotheekkopie; het origineel mag weg.
            try:
                self.input_pad = schaats_db.analyse_video_pad(self.bieb, analyse_id)
            except Exception:
                pass   # terugvallen op de bronvideo (alleen weergave)
        self.stack.setCurrentWidget(self.pagina_analyse)
        self._toon_resultaten(info, resultaten, events)

    # ---- Batch-analyse (meerdere video's achter elkaar) --------------------------

    # ── Fragmenten knippen uit een lange opname (fase 8) ─────────────────────
    def _knip_opname(self):
        """Opname → knipvenster → clips wegschrijven → de bestaande batch-flow in.

        Ná het knippen gebeurt er niets nieuws: elk fragment is een gewoon videobestandje,
        dus `BatchAnalyseDialog` (voorgevuld) + `_nieuwe_batch_analyse` doen de rest. Geen
        tweede analyse-pijplijn en geen tweede opslagroute."""
        if self._bezig:
            QMessageBox.information(
                self, "Even geduld",
                "Er draait al een analyse. Wacht daarop voordat je nieuwe fragmenten knipt.")
            return
        bron = self._geselecteerde_opname()
        if bron is None:
            QMessageBox.information(
                self, "Fragmenten knippen",
                "Kies eerst een opname in de lijst.\n\n"
                "Staat er niets? Zet je opnames in de map 'opnames' in de bibliotheek en "
                "druk op 'Vernieuwen'.")
            return
        # Cloudsync: melden en niet openen, i.p.v. het knipvenster op een half bestand laten
        # stuklopen. Een opname van een half uur in 4K is minutenlang onderweg.
        if not self._opname_beschikbaar(bron):
            return
        # Vóór het markeerwerk, niet erna: zonder profiel valt er straks niets op te slaan.
        schaatsers = schaats_db.lijst_schaatsers(self.bieb)
        if not schaatsers:
            QMessageBox.information(
                self, "Fragmenten knippen",
                "Maak eerst een schaatser aan — elke analyse hoort bij een profiel.")
            return
        try:
            info = video_info(bron["pad"])
        except Exception as e:
            QMessageBox.critical(self, "Opname", f"Kan de opname niet openen:\n{e}")
            return

        gedaan = schaats_db.bron_fragmenten(self.bieb, bron["id"])
        dlg = FragmentKiezer(bron["pad"], info, gedaan=gedaan, parent=self)
        if dlg.exec() != QDialog.Accepted or not dlg.fragmenten:
            return

        paden = self._knip_naar_tijdelijk(bron["pad"], dlg.fragmenten, info)
        if not paden:
            return
        voorgevuld = [
            {"input_pad": pad, "titel": naam, "bron_id": bron["id"],
             "bron_start_frame": start, "bron_eind_frame": eind}
            for pad, (start, eind, naam) in zip(paden, dlg.fragmenten)]
        self._nieuwe_batch_analyse(voorgevuld=voorgevuld)

    def _knip_naar_tijdelijk(self, bron_pad, fragmenten, info):
        """Schrijft de gemarkeerde stukken naar een tijdelijke map en retourneert de paden
        (of [] bij afbreken/fout). `sla_analyse_op` kopieert ze daarna zoals altijd naar
        `media/<uuid>/` — één extra kopie van een kort bestandje, niet de moeite om die
        opslagroute voor open te breken."""
        self._ruim_knipmap_op()
        self._knip_tmpmap = tempfile.mkdtemp(prefix="schaats_fragmenten_")

        voortgang = QProgressDialog("Fragmenten knippen...", "Stoppen", 0, 100, self)
        voortgang.setWindowTitle("Knippen")
        voortgang.setWindowModality(Qt.WindowModal)
        voortgang.setMinimumDuration(0)
        voortgang.setValue(0)

        def _melden(gedaan, totaal):
            voortgang.setValue(int(gedaan / max(1, totaal) * 100))

        try:
            paden = knip_fragmenten(bron_pad, fragmenten, self._knip_tmpmap,
                                    progress_callback=_melden,
                                    stop_check=voortgang.wasCanceled, fps=info.fps)
        except KnipAfgebroken:
            self._ruim_knipmap_op()
            return []
        except Exception as e:
            self._ruim_knipmap_op()
            QMessageBox.critical(self, "Knippen mislukt", str(e))
            return []
        finally:
            voortgang.close()
        return paden

    def _ruim_knipmap_op(self):
        """Gooit de tijdelijke fragmentmap weg (de clips staan dan in de bibliotheek)."""
        if self._knip_tmpmap:
            shutil.rmtree(self._knip_tmpmap, ignore_errors=True)
            self._knip_tmpmap = None

    def _nieuwe_batch_analyse(self, voorgevuld=None):
        schaatsers = schaats_db.lijst_schaatsers(self.bieb)
        if not schaatsers:
            QMessageBox.information(
                self, "Batch-analyse",
                "Maak eerst een schaatser aan — elke analyse hoort bij een profiel.")
            return
        dlg = BatchAnalyseDialog(schaatsers, voorkeur_id=self._geselecteerde_schaatser_id(),
                                 voorgevuld=voorgevuld, parent=self)
        if dlg.exec() != QDialog.Accepted:
            self._ruim_knipmap_op()      # geknipte clips zonder batch zijn nutteloos
            return

        # Model resolven — gedeeld voor de hele batch, alleen relevant voor MediaPipe.
        model_pad, heavy = STANDAARD_MODEL, False
        if not IS_YOLO:
            if dlg.heavy_gevraagd:
                if os.path.isfile(HEAVY_MODEL):
                    model_pad, heavy = HEAVY_MODEL, True
                else:
                    QMessageBox.warning(
                        self, "Heavy-model ontbreekt",
                        "pose_landmarker_heavy.task staat niet naast het script.\n\n"
                        "Er wordt nu met het full-model gewerkt.")
                    model_pad = STANDAARD_MODEL
            if not os.path.isfile(model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Kies pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                model_pad = gekozen

        smooth_n, threshold = dlg.smooth_n, dlg.threshold
        geen_smoothing = dlg.geen_smoothing
        bocht = dlg.chk_bocht.isChecked()

        # Perspectiefkalibratie: één keer voor de héle batch. Alle clips van een batch
        # komen in de praktijk uit dezelfde opname (fase 8 knipt ze zo aan), dus dezelfde
        # camerastand — en alleen op één gedeelde kalibratie zijn hun hoeken onderling
        # vergelijkbaar. Per clip opnieuw natrekken zou zeven nét andere kalibraties geven.
        batch_perspectief = None
        if dlg.chk_perspectief.isChecked():
            eerste_frame = None
            for taak in dlg.taken:
                eerste_frame = self._lees_eerste_frame(taak["input_pad"])
                if eerste_frame is not None:
                    break
            if eerste_frame is None:
                return
            batch_perspectief = self._kies_perspectief(eerste_frame)
            if batch_perspectief is None:        # dialoog afgebroken
                self._ruim_knipmap_op()
                return

        # Verzamel-lus: per video het eerste frame + doelschaatser + horizon uitvragen.
        taken = []
        for taak in dlg.taken:
            pad, schaatser_id, titel = taak["input_pad"], taak["schaatser_id"], taak["titel"]
            frame0 = self._lees_eerste_frame(pad)
            if frame0 is None:
                continue                         # _lees_eerste_frame heeft al gemeld

            doel = self._kies_doelschaatser(frame0)
            if doel is False:                    # dialoog afgebroken
                if self._overslaan_of_afbreken(titel):
                    continue
                return
            if batch_perspectief is not None:
                # De kalibratie levert de kanteling zelf; de horizon-stap vervalt, net
                # als bij een enkele analyse.
                horizon_deg, auto_horizon = 0.0, False
                if not batch_perspectief.invoer.past_bij(frame0.shape[1], frame0.shape[0]):
                    QMessageBox.warning(
                        self, "Kalibratie past niet",
                        f"'{titel}' is {frame0.shape[1]}×{frame0.shape[0]} en de "
                        f"kalibratie is gemaakt op {batch_perspectief.invoer.beeld_w}×"
                        f"{batch_perspectief.invoer.beeld_h}. Deze clip wordt overgeslagen.")
                    continue
            else:
                horizon = self._kies_horizon(frame0)
                if horizon is False:             # dialoog afgebroken
                    if self._overslaan_of_afbreken(titel):
                        continue
                    return
                horizon_deg, auto_horizon = horizon

            instellingen = {
                "smooth_n": smooth_n,
                "threshold": threshold,
                "smooth_landmarks": not geen_smoothing,
                "bocht_overslaan": bocht,
                "doel_punt": list(doel) if doel else None,
                "horizon_deg": horizon_deg,
                "auto_horizon": auto_horizon,
                "heavy": heavy,
                "backend_naam": BACKEND_NAAM,
                "perspectief_gebruikt": batch_perspectief is not None,
                "perspectief": batch_perspectief.naar_dict() if batch_perspectief else None,
            }
            taken.append({
                "input_pad": pad, "schaatser_id": schaatser_id, "titel": titel,
                "doel_punt": doel, "horizon_deg": horizon_deg, "auto_horizon": auto_horizon,
                "smooth_landmarks": not geen_smoothing, "smooth_n": smooth_n,
                "threshold": threshold, "model_pad": model_pad, "instellingen": instellingen,
                "bocht": bocht, "perspectief": batch_perspectief,
                # Fase 8: uit welk stuk van welke opname deze clip komt (None bij een
                # losse video) — levert straks de grijze blokken in het knipvenster.
                "bron_id": taak.get("bron_id"),
                "bron_start_frame": taak.get("bron_start_frame"),
                "bron_eind_frame": taak.get("bron_eind_frame"),
            })

        if not taken:
            self._ruim_knipmap_op()
            return
        # Op de bibliotheek blijven terwijl de batch draait, zodat er gebrowst kan worden.
        self._start_batch(taken)

    def _overslaan_of_afbreken(self, titel):
        """Bij een afgebroken doel-/horizon-kiezer: alleen deze video overslaan (True) of
        de hele batch afbreken (False)."""
        antwoord = QMessageBox.question(
            self, "Video overslaan?",
            f"De instelling voor '{titel}' is afgebroken.\n\n"
            "Wil je alleen deze video overslaan en met de rest doorgaan?\n"
            "(Nee = de hele batch afbreken.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return antwoord == QMessageBox.Yes

    def _start_batch(self, taken):
        self.speler.zet_besturing_actief(False)
        self.btn_export.setEnabled(False)
        self._auto_toon_klaar = False        # batch toont zelf geen resultaten
        self._zet_bezig(True)

        self._batch_index, self._batch_totaal, self._batch_huidig = 0, len(taken), ""

        # Voortgangsbalk mét knop 'Stop na deze video' — niet-blokkerend, dus de
        # bibliotheek blijft ondertussen bruikbaar.
        self._toon_voortgangsbalk("Batch starten...", met_stop=True)

        self._waarschuw_backend_terugval()
        self.batch_worker = BatchWorker(taken, self.bieb, BACKEND_NAAM, self.trainer_naam)
        self.batch_worker.taak_start.connect(self._batch_taak_start)
        self.batch_worker.voortgang.connect(self._batch_voortgang)
        self.batch_worker.status.connect(self._analyse_status)   # busy-fase hergebruiken
        self.batch_worker.taak_klaar.connect(self._batch_taak_klaar)  # nieuwe analyse live tonen
        self.batch_worker.alles_klaar.connect(self._batch_klaar)
        self.batch_worker.start()

    def _batch_stop_gevraagd(self):
        if self.batch_worker is not None:
            self.batch_worker.requestInterruption()
        self.lbl_voortgang.setText("Stopt na de huidige video...")
        self.btn_voortgang_stop.setEnabled(False)

    def _batch_taak_start(self, index, totaal, titel):
        self._batch_index, self._batch_totaal, self._batch_huidig = index, totaal, titel
        self.bar_voortgang.setRange(0, 100)
        self.bar_voortgang.setValue(0)
        self.lbl_voortgang.setText(f"Video {index + 1}/{totaal} — {titel}")

    def _batch_voortgang(self, frame_nr, totaal):
        if totaal > 0:
            self.bar_voortgang.setRange(0, 100)
            self.bar_voortgang.setValue(int(frame_nr / totaal * 100))
        self.lbl_voortgang.setText(
            f"Video {self._batch_index + 1}/{self._batch_totaal} — {self._batch_huidig} "
            f"({frame_nr}/{totaal})")

    def _batch_taak_klaar(self, index, analyse_id):
        # Elke afgeronde video meteen in de bibliotheek laten opduiken (raakt de
        # eventueel geopende weergave niet — dat zijn andere widgets).
        if self._afsluiten:
            return
        self._vernieuw_schaatsers()

    def _batch_klaar(self, geslaagd, fouten, waarschuwingen=()):
        if self._afsluiten:
            return
        self._verberg_voortgangsbalk()
        self._zet_bezig(False)
        self._vernieuw_schaatsers()              # nieuwe analyses direct zichtbaar
        self._vernieuw_opnames()                 # fragmenttelling per opname bijwerken
        # De clips staan nu als kopie in media/<uuid>/; de tijdelijke map mag weg.
        self._ruim_knipmap_op()

        n_ok = len(geslaagd)
        n_tot = n_ok + len(fouten)
        # Waarschuwingen (bv. een klik die niemand raakte) horen bij een geslaagde video:
        # de analyse is er, maar mogelijk van de verkeerde schaatser — dus wél melden.
        extra = ""
        if waarschuwingen:
            regels = "\n".join(f"• {t}: {m}" for t, m in waarschuwingen)
            extra = f"\n\nLet op:\n{regels}"
        if fouten:
            regels = "\n".join(f"• {t}: {m}" for t, m in fouten)
            QMessageBox.warning(
                self, "Batch klaar",
                f"{n_ok} van {n_tot} video's geslaagd en opgeslagen.\n\nMislukt:\n{regels}{extra}")
        elif waarschuwingen:
            QMessageBox.warning(
                self, "Batch klaar",
                f"Alle {n_ok} video's zijn geanalyseerd en opgeslagen in de bibliotheek.{extra}")
        else:
            QMessageBox.information(
                self, "Batch klaar",
                f"Alle {n_ok} video's zijn geanalyseerd en opgeslagen in de bibliotheek.")

    def _toon_resultaten(self, info, resultaten, events, bron=None):
        """
        Vult de weergavepagina met een resultatenlijst. Gedeeld door een verse analyse
        en door een uit .npz geladen analyse (`bron` = de bestandsnaam, voor de statusbalk).
        """
        self.events = events

        # Editor-status resetten (geen edit-lekkage tussen analyses); niet via de
        # toggle-handler, want de weergave wordt hieronder toch opnieuw opgebouwd.
        self._stop_plaatsen()       # nog vóór de reset: hoort bij de vórige resultatenlijst
        self._editor_actief = False
        self._sleep = None
        self.speler.bewerk_modus = False
        self.speler.volgen_bevroren = False
        self._undo.clear()
        self._redo.clear()
        self._handmatig.clear()
        self.btn_bewerken.blockSignals(True)
        self.btn_bewerken.setChecked(False)
        self.btn_bewerken.blockSignals(False)
        self.editor_balk.setVisible(False)
        self._sluit_plaats_balk()

        # Capture heropenen, zoom resetten, besturing aan — toont nog géén frame.
        self.speler.laad(info, resultaten, self.input_pad)

        self._vul_tabel()
        self._vul_grafiek()
        self._update_dekking()
        self.btn_export.setEnabled(bool(events))
        # Vergelijken kan alleen met een analyse die in de bibliotheek staat — de
        # vergelijkkant laadt hem daar opnieuw uit. Hetzelfde voor de info: de meta
        # (appversie, instellingen) komt uit de DB-rij.
        self.btn_vergelijk_deze.setEnabled(self.analyse_id is not None)
        self.btn_info.setEnabled(self.analyse_id is not None)
        # Alleen aanbieden waar er iets te winnen valt: een analyse die de bocht al kent
        # hoeft niets, en zonder bibliotheek-id valt het resultaat nergens te bewaren.
        self.btn_bocht_nu.setEnabled(
            self.analyse_id is not None and not any(r.bocht for r in resultaten))

        herkomst = f"  ·  geladen uit {bron}" if bron else ""
        self.statusBar().showMessage(
            f"{os.path.basename(self.input_pad)} — {info.w}×{info.h} @ {info.fps:.1f}fps, "
            f"{len(resultaten)} frames, {len(events)} afzetten gevonden{herkomst}")

        # Pas nu tekenen: tabel en grafiek staan klaar voor de op_frame_getoond-haak.
        self.speler.ga_naar(0)

    # ── Tabel + grafiek vullen ───────────────────────────────────────────
    def _vul_tabel(self):
        # Kolommen dynamisch: perspectiefcorrectie en snelheid/slaglengte alleen als
        # er een kalibratie actief was (de events dragen die velden dan).
        met_corr = any(ev.correctie is not None for ev in self.events)
        met_metrisch = any(ev.snelheid is not None for ev in self.events)
        koppen = ["#", "Tijd (s)", "Been", "Hoek (°)"]
        if met_corr:
            koppen.append("Corr. (°)")
        if met_metrisch:
            koppen += ["v (m/s)", "Slag (m)"]
        self.tabel.setColumnCount(len(koppen))
        self.tabel.setHorizontalHeaderLabels(koppen)

        self.tabel.setRowCount(len(self.events))
        markeer_kleur = QColor(120, 60, 20)     # waarschuwing: onmogelijke L/R-herhaling
        onbetrouwbaar_kleur = QColor(40, 60, 120)  # hoek gemeten met been ~in kijkrichting
        onvolledig_kleur = QColor(70, 70, 70)   # afzet niet uit-geobserveerd: geen meting
        # De reden verschilt voor de gebruiker: bij "afgekapt" helpt een langere opname,
        # bij "geen volledige push" is de been-toewijzing of de slag zelf de vraag.
        onvolledig_uitleg = {
            ONV_AFGEKAPT:
                "Deze afzet liep nog toen de video (of de detectie) ophield — de push is "
                "niet afgemaakt, dus de hoek is te steil. Telt niet mee in "
                "gemiddelde/min/max.",
            ONV_GEEN_PUSH:
                "Het been kwam in deze slag wel rechtop, maar er is geen zijwaartse afzet "
                "waargenomen: bij volledige strekking stond het onderbeen nog vrijwel "
                "verticaal. De hoek is dus het rechtop-komen en telt niet mee in "
                "gemiddelde/min/max. Controleer of hier echt een afzet zat.",
        }
        for i, ev in enumerate(self.events):
            waarden = [str(i + 1), f"{ev.start_tijd:.2f}", ev.been.capitalize(), f"{ev.hoek:.1f}"]
            if met_corr:
                waarden.append(f"{ev.correctie:+.1f}" if ev.correctie is not None else "—")
            if met_metrisch:
                waarden.append(f"{ev.snelheid:.1f}" if ev.snelheid is not None else "—")
                waarden.append(f"{ev.slaglengte:.1f}" if ev.slaglengte is not None else "—")
            gemarkeerd = ev.opmerking == "gemiste tegenafzet?"
            onbetrouwbaar = met_corr and not ev.betrouwbaar
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setTextAlignment(Qt.AlignCenter)
                if ev.onvolledig:
                    item.setBackground(onvolledig_kleur)
                    item.setToolTip(onvolledig_uitleg.get(
                        ev.onvolledig,
                        f"Onvolledige afzet ({ev.onvolledig}) — telt niet mee in "
                        "gemiddelde/min/max."))
                elif gemarkeerd:
                    item.setBackground(markeer_kleur)
                    item.setToolTip("Zelfde been als de vorige afzet — onmogelijk in het "
                                    "schaatsritme. Waarschijnlijk een gemiste tegen-afzet.")
                elif onbetrouwbaar:
                    item.setBackground(onbetrouwbaar_kleur)
                    item.setToolTip("Been stond bij afzet-voltooiing bijna in de kijkrichting "
                                    "— de perspectiefcorrectie (en dus de hoek) is hier "
                                    "onbetrouwbaar.")
                self.tabel.setItem(i, kolom, item)

        if self.events:
            # Onvolledige afzetten tellen niet mee: hun hoek is het rechtop-komen en trekt
            # gem/max omhoog. De reden staat erbij, want die vraagt om iets anders van de
            # gebruiker (langere opname vs. de slag zelf nakijken).
            meetbaar = [ev for ev in self.events if not ev.onvolledig]
            hoeken = [ev.hoek for ev in meetbaar]
            n_mark = sum(1 for ev in self.events if ev.opmerking == "gemiste tegenafzet?")
            redenen = {}
            for ev in self.events:
                if ev.onvolledig:
                    redenen[ev.onvolledig] = redenen.get(ev.onvolledig, 0) + 1
            if hoeken:
                tekst = (f"gem {np.mean(hoeken):.1f}°  |  min {min(hoeken):.1f}°  "
                         f"|  max {max(hoeken):.1f}°")
            else:
                tekst = "geen volledige afzet gemeten"
            if redenen:
                tekst += "   ·  " + ", ".join(f"{n}× {reden}" for reden, n in redenen.items())
                tekst += " (niet meegeteld)"
            if n_mark:
                tekst += f"   ·  ⚠ {n_mark} mogelijke L/R-fout"
            if met_corr:
                n_onb = sum(1 for ev in meetbaar if not ev.betrouwbaar)
                if n_onb:
                    tekst += f"   ·  ⚠ {n_onb} onbetrouwbare hoek (kijkrichting)"
            self.lbl_stats.setText(tekst)
        else:
            self.lbl_stats.setText("Geen afzetten gedetecteerd")

    def _vul_grafiek(self):
        self.serie_hoek.clear()
        # Dezelfde grootheid als de tabel (de hoek per frame), zodat een tabelrij
        # precies op de curve valt.
        punten = [(r.tijd, r.hoek) for r in self.resultaten
                  if r.pose_gevonden and r.hoek is not None]
        if not punten:
            return
        for t, hoek in punten:
            self.serie_hoek.append(t, hoek)

        tijden = [t for t, _ in punten]
        hoeken = [h for _, h in punten]
        self.as_x.setRange(0, max(tijden) if tijden else 1)
        marge = 5
        self.as_y.setRange(min(hoeken) - marge, max(hoeken) + marge)

    def _update_grafiek_marker(self, tijd):
        y_min, y_max = self.as_y.min(), self.as_y.max()
        self.serie_marker.clear()
        self.serie_marker.append(tijd, y_min)
        self.serie_marker.append(tijd, y_max)

    # ── Navigatie / weergave ─────────────────────────────────────────────
    def _speler_frame_getoond(self, idx):
        """Haak van de VideoSpeler: alles wat de analysepagina aan een frame ophangt."""
        # Wegnavigeren tijdens een plaats-reeks sluit die eerst netjes af. De idx-toets is
        # nodig omdat _herbereken() zelf hertekent en dus hier terugkomt op hetzelfde frame.
        if self._plaats is not None and self._plaats['idx'] != idx:
            self._stop_plaatsen()
        resultaat = self.resultaten[idx]
        self._update_grafiek_marker(resultaat.tijd)
        self._markeer_actieve_rij(idx)
        self._update_live_status(resultaat)
        if self._editor_actief:
            self._update_editor_knoppen()   # "Maak skelet" alleen op een frame zonder pose

    def _update_live_status(self, resultaat):
        if not resultaat.pose_gevonden:
            tekst = "Bocht — niet geanalyseerd" if resultaat.bocht else "Geen pose gedetecteerd"
            self.lbl_live.setText(tekst)
            self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px; color: #c33;")
            return
        if resultaat.bocht:
            # Wel een skelet, maar geen afgeleiden: `verwerk_afgeleiden` slaat bochtframes
            # over, dus been/hoek zijn hier None en er valt niets te tonen.
            self.lbl_live.setText("Bocht — geen meting")
            self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px; color: #c80;")
            return

        status = "GEWICHT OP BEEN" if resultaat.gewicht_erop else "AFZET VOLTOOID"
        kleur = "#2a2" if resultaat.gewicht_erop else "#c33"
        extra = ""
        if resultaat.hoek_correctie is not None:
            extra = f"Corr: {resultaat.hoek_correctie:+.1f}°   "
            if resultaat.snelheid is not None:
                extra += f"v: {resultaat.snelheid:.1f} m/s   "
            if not resultaat.hoek_betrouwbaar:
                extra += "⚠ onbetrouwbaar   "
        strek = f"Strekking: {resultaat.strek_ratio:.2f}   " if resultaat.strek_ratio is not None else ""
        self.lbl_live.setText(
            f"Afzetbeen: {resultaat.been.upper()}   "
            f"Afzethoek: {resultaat.hoek}°   "
            f"Kniehoek: {resultaat.kniehoek}°   "
            f"{strek}{extra}{status}"
        )
        self.lbl_live.setStyleSheet(f"font-weight: bold; padding-right: 10px; color: {kleur};")

    # ── Skelet-editor: handles op de VideoSpeler ─────────────────────────────
    def _handle_straal(self):
        """Handle-/grijpradius in (geschaalde) schermpixels, evenredig met de schaatser:
        GRIJP_FRAC × torso-lengte-op-het-scherm, geklemd op [GRIJP_MIN_PX, GRIJP_MAX_PX].
        Via _norm_naar_widget zit de crop/zoom-schaal er al in (de letterbox-offset valt bij
        een afstand weg), dus dit klopt op elke zoomstand en is exact consistent met het
        hittesten. Val terug op GRIJP_MAX_PX als er geen bruikbare pose/torso is."""
        if (not (0 <= self.huidige_idx < len(self.resultaten))
                or self.speler.weergave_scaled is None):
            return float(GRIJP_MAX_PX)
        r = self.resultaten[self.huidige_idx]
        if not (r.pose_gevonden and isinstance(r.lm, list)):
            return float(GRIJP_MAX_PX)
        lm = r.lm

        def _mid(a, b):
            pts = [lm[i] for i in (a, b)
                   if getattr(lm[i], 'visibility', 1.0) >= HANDLE_MIN_VIS]
            if not pts:
                return None
            return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))

        schouder, heup = _mid(11, 12), _mid(23, 24)   # schouder-midden → heup-midden
        if schouder is None or heup is None:
            return float(GRIJP_MAX_PX)
        p1 = self.speler.norm_naar_widget(*schouder)
        p2 = self.speler.norm_naar_widget(*heup)
        torso = math.hypot(p1.x() - p2.x(), p1.y() - p2.y())
        return min(float(GRIJP_MAX_PX), max(float(GRIJP_MIN_PX), GRIJP_FRAC * torso))

    def _teken_handles(self, pixmap):
        """Tekent sleepbare ringen op elke zichtbare landmark van het huidige frame,
        rechtstreeks op de geschaalde pixmap (dus vaste grootte in schermpixels).

        Hangt permanent als overlay_tekenaar aan de speler; de bewerk-modus-guard zit
        daarom hier (die vlag wordt op twee plekken uitgezet — één guard is faalveilig)."""
        if not self._editor_actief:
            return
        if not (0 <= self.huidige_idx < len(self.resultaten)):
            return
        r = self.resultaten[self.huidige_idx]
        if not (r.pose_gevonden and isinstance(r.lm, list)):
            return
        pw, ph = pixmap.width(), pixmap.height()
        x0n, y0n, wn, hn = self.speler.crop_norm  # bij zoom==1 (0,0,1,1) → lm.x*pw, lm.y*ph
        straal = self._handle_straal()      # schaalt mee met de schaatser + zoom
        gemarkeerd = self._handmatig.get(self.huidige_idx, set())
        sleep_j = (self._sleep['j'] if self._sleep and self._sleep['idx'] == self.huidige_idx
                   else None)
        doel_j = self._plaats_doelpunt()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            for j, lm in enumerate(r.lm):
                if getattr(lm, 'visibility', 1.0) < HANDLE_MIN_VIS and j != doel_j:
                    continue
                # buiten de uitsnede valt de ring buiten [0,pw]; de painter clipt hem
                middel = QPointF((lm.x - x0n) / wn * pw, (lm.y - y0n) / hn * ph)
                if j == sleep_j:
                    painter.setPen(QPen(QColor(255, 255, 0), 3))     # actief gesleept
                elif j in gemarkeerd:
                    painter.setPen(QPen(QColor(0, 255, 120), 2))     # handmatig gezet
                else:
                    painter.setPen(QPen(QColor(255, 255, 255), 1))   # gewoon
                painter.drawEllipse(middel, straal, straal)
                if j == doel_j:
                    # Het punt dat nu gevraagd wordt: dikke oranje ring met kruisdraad op de
                    # voorgevulde plek. De hint-tekst alleen laat de gebruiker zoeken; dit
                    # zegt "ongeveer hier, corrigeer maar".
                    painter.setPen(QPen(QColor(255, 150, 0), 3))
                    buiten = straal * 1.8
                    painter.drawEllipse(middel, buiten, buiten)
                    painter.drawLine(QPointF(middel.x() - buiten * 1.5, middel.y()),
                                     QPointF(middel.x() + buiten * 1.5, middel.y()))
                    painter.drawLine(QPointF(middel.x(), middel.y() - buiten * 1.5),
                                     QPointF(middel.x(), middel.y() + buiten * 1.5))
        finally:
            painter.end()

    def _plaats_doelpunt(self):
        """Het landmark dat de lopende plaats-reeks nu vraagt, of None."""
        if not self._plaats or self._plaats['idx'] != self.huidige_idx:
            return None
        stap = self._plaats['stap']
        return PLAATS_VOLGORDE[stap] if 0 <= stap < len(PLAATS_VOLGORDE) else None

    # ── Skelet-editor: bewerk-modus + slepen (fase 3) ────────────────────────
    def _toggle_bewerken(self, actief):
        if not actief:
            self._stop_plaatsen()           # nooit een halve reeks achterlaten
        self._editor_actief = actief
        self.speler.bewerk_modus = actief   # stuurt de pan-vs-editor-voorrang van de muis
        self.editor_balk.setVisible(actief)
        self._sleep = None
        self.speler.volgen_bevroren = False
        if actief:
            self.speler.pauzeer()
            self.lbl_editor_hint.setText("Sleep een punt naar de juiste plek.")
            self._update_editor_knoppen()
        self.speler.toon_huidig_frame()

    def _update_editor_knoppen(self):
        bezig = self._plaats is not None
        self.btn_undo.setEnabled(bool(self._undo) and not bezig)
        self.btn_redo.setEnabled(bool(self._redo) and not bezig)
        self.btn_herstel.setEnabled(self.analyse_id is not None and not bezig)
        self.btn_volgend_gat.setEnabled(not bezig)
        # Alleen aanbieden waar het zin heeft: op een frame dat al een pose heeft is
        # slepen het gereedschap, niet plaatsen — en in de bocht wordt er toch niet gemeten.
        self.btn_maak_skelet.setEnabled(not bezig and self._is_gat(self.huidige_idx))

    def _frame_bewerkbaar(self, idx):
        """Een frame is bewerkbaar als het een pose heeft die als lijst van (muteerbare)
        Landmark-tuples in geheugen staat — geldt voor alle uit de bibliotheek geladen
        analyses. Ruwe MediaPipe-objecten (diagnose-stand 'geen smoothing') niet."""
        if not (0 <= idx < len(self.resultaten)):
            return False
        r = self.resultaten[idx]
        return bool(r.pose_gevonden and isinstance(r.lm, list))

    def _zet_landmark(self, idx, j, nx, ny, vis=None):
        """Vervangt landmark j in frame idx (Landmark is immutable)."""
        lm = self.resultaten[idx].lm[j]
        self.resultaten[idx].lm[j] = Landmark(nx, ny, lm.z,
                                              lm.visibility if vis is None else vis)

    def _uitvloei_frames(self, idx, N):
        """Frame-indices waarover de correctie uitvloeit: idx plus tot ±N buurframes,
        stoppend bij een detectiegat (onbewerkbaar frame) in elke richting."""
        frames = [idx]
        for richting in (-1, 1):
            for k in range(1, N + 1):
                f = idx + richting * k
                if not self._frame_bewerkbaar(f):
                    break
                frames.append(f)
        return frames

    def _zoek_landmark(self, pos):
        """Index van de dichtstbijzijnde zichtbare landmark binnen de handle-radius
        (_handle_straal) van de muispositie (schermruimte), of None."""
        if not self._frame_bewerkbaar(self.huidige_idx) or self.speler.weergave_scaled is None:
            return None
        straal = self._handle_straal()      # zelfde radius als de getekende ring
        beste, beste_d2 = None, float(straal * straal)
        for j, lm in enumerate(self.resultaten[self.huidige_idx].lm):
            if getattr(lm, 'visibility', 1.0) < HANDLE_MIN_VIS:
                continue
            w = self.speler.norm_naar_widget(lm.x, lm.y)
            d2 = (w.x() - pos.x()) ** 2 + (w.y() - pos.y()) ** 2
            if d2 <= beste_d2:
                beste, beste_d2 = j, d2
        return beste

    def _toon_hover_naam(self, event):
        """Toont in de bewerk-modus een tooltip met het lichaamsdeel van het punt onder de
        cursor (zelfde trefradius als selecteren). Geen punt in de buurt → tooltip weg."""
        j = self._zoek_landmark(event.position())
        if j is None:
            QToolTip.hideText()
            return
        naam = LANDMARK_NAMEN.get(j, f"punt {j}")
        # iets naast de cursor zodat de tekst het punt zelf niet afdekt
        pos = (event.globalPosition() + QPointF(14, 10)).toPoint()
        QToolTip.showText(pos, naam, self.speler.label)

    # De pan-tak (links-slepen bij zoom > 1 buiten bewerk-modus) zit in de VideoSpeler;
    # deze haken krijgen het event alleen als de speler het niet zelf heeft opgeslokt.
    def _editor_muis_druk(self, event):
        if not self._editor_actief:
            return
        # De plaats-reeks krijgt voorrang en slokt de klik altijd op: het frame ís tijdens
        # de reeks bewerkbaar, dus zonder deze tak zou een klik een sleep starten op een
        # voorgevuld punt in plaats van het gevraagde punt neer te zetten.
        if self._plaats is not None:
            self._plaats_klik(event)
            return
        if not self._frame_bewerkbaar(self.huidige_idx):
            self.lbl_editor_hint.setText("Dit frame heeft geen bewerkbare pose.")
            return
        j = self._zoek_landmark(event.position())
        if j is None:
            return
        self._sleep = {'idx': self.huidige_idx, 'j': j,
                       'start_lm': self.resultaten[self.huidige_idx].lm[j]}
        # auto-volgen bevriezen: anders verspringt de uitsnede onder de cursor
        self.speler.volgen_bevroren = True

    def _editor_muis_beweeg(self, event):
        if not self._editor_actief or self._plaats is not None:
            return
        if not self._sleep:
            # geen sleep bezig → toon bij hover het lichaamsdeel onder de cursor
            self._toon_hover_naam(event)
            return
        norm = self.speler.widget_naar_norm(event.position())
        if norm is None:
            return
        nx = min(1.0, max(0.0, norm[0]))
        ny = min(1.0, max(0.0, norm[1]))
        idx, j = self._sleep['idx'], self._sleep['j']
        self._zet_landmark(idx, j, nx, ny, vis=1.0)   # live feedback; nog geen herbereken
        self.speler.ga_naar(idx)

    def _editor_muis_los(self, event):
        if self._plaats is not None:
            return                          # de klik is al bij het indrukken afgehandeld
        if not (self._editor_actief and self._sleep):
            self.speler.volgen_bevroren = False
            return
        sleep, self._sleep = self._sleep, None
        self.speler.volgen_bevroren = False
        idx, j, start_lm = sleep['idx'], sleep['j'], sleep['start_lm']
        eind = self.resultaten[idx].lm[j]
        dx, dy = eind.x - start_lm.x, eind.y - start_lm.y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            self.speler.ga_naar(idx)              # geen echte verplaatsing: alleen hertekenen
            return
        # Centrum terug op pre-edit zodat het hele venster gelijk begint.
        self.resultaten[idx].lm[j] = start_lm
        N = self.spin_uitvloei.value()
        frames = self._uitvloei_frames(idx, N)
        oud = {f: self.resultaten[f].lm[j] for f in frames}
        for f in frames:
            k = abs(f - idx)
            gewicht = 1.0 if k == 0 else 0.5 * (1.0 + math.cos(math.pi * k / N))
            lm = self.resultaten[f].lm[j]
            vis = 1.0 if f == idx else lm.visibility     # alleen het gesleepte punt is zeker
            self.resultaten[f].lm[j] = Landmark(lm.x + dx * gewicht, lm.y + dy * gewicht,
                                                lm.z, vis)
        nieuw = {f: self.resultaten[f].lm[j] for f in frames}
        self._undo.append({'type': 'sleep', 'j': j, 'oud': oud, 'nieuw': nieuw})
        self._redo.clear()
        self._handmatig.setdefault(idx, set()).add(j)
        self._na_edit()

    # ── Skelet plaatsen op een frame zonder pose ─────────────────────────────
    def _is_gat(self, idx):
        """Een frame dat een skelet mist én waar een skelet iets oplevert. Bochtframes
        vallen af: die leveren toch geen meting, dus ze met de hand dichten is werk voor
        niets."""
        if not (0 <= idx < len(self.resultaten)):
            return False
        r = self.resultaten[idx]
        return not r.pose_gevonden and not r.bocht

    def _gat_positie(self, idx):
        """(hoeveelste, totaal) van frame `idx` binnen zijn aaneengesloten reeks frames
        zónder skelet. Voor de hint: een half gedicht gat verandert de tabel nog niet,
        want `bepaal_afzet_uit_strek` breekt de stand-run op élk skeletloos frame af."""
        if not self._is_gat(idx):
            return (0, 0)
        start = idx
        while start > 0 and self._is_gat(start - 1):
            start -= 1
        eind = idx
        while self._is_gat(eind + 1):
            eind += 1
        return (idx - start + 1, eind - start + 1)

    def _ga_naar_volgend_gat(self):
        """Springt naar het eerstvolgende frame zonder skelet; wrapt na het laatste."""
        n = len(self.resultaten)
        if not n:
            return
        volgorde = list(range(self.huidige_idx + 1, n)) + list(range(0, self.huidige_idx + 1))
        doel = next((i for i in volgorde if self._is_gat(i)), None)
        if doel is None:
            self.lbl_editor_hint.setText("Elk frame heeft een skelet — niets meer te doen.")
            return
        self.speler.ga_naar(doel)
        hoeveelste, totaal = self._gat_positie(doel)
        self.lbl_editor_hint.setText(
            f"Frame {doel} — {hoeveelste} van {totaal} zonder skelet in dit gat.")

    def _start_plaatsen(self):
        """"Maak skelet" op een frame zonder pose.

        Twee wegen, en de eerste is verreweg de gewone: leveren de buurframes een bruikbare
        voorvulling, dan komt dat skelet er meteen op en corrigeer je het met de normale
        sleep-editor — dat is intuïtiever dan acht keer een naam lezen en klikken, en het
        houdt één manier van werken voor álle frames. Alleen als er níets is om over te
        nemen (nergens in de analyse een pose) valt er niets te verslepen en vraagt het
        programma de punten één voor één op in een vaste volgorde."""
        idx = self.huidige_idx
        if self._plaats is not None or not (0 <= idx < len(self.resultaten)):
            return
        r = self.resultaten[idx]
        if r.pose_gevonden:
            self.lbl_editor_hint.setText(
                "Dit frame heeft al een skelet — sleep de punten die niet kloppen.")
            return
        info = self.video_info
        oud_lm, oud_pose = r.lm, r.pose_gevonden
        voorvulling = maak_voorvulling(self.resultaten, idx, info.fps or 30.0)
        bruikbaar = all(voorvulling[j].visibility >= HANDLE_MIN_VIS for j in PLAATS_VERPLICHT)

        r.lm = voorvulling
        r.pose_gevonden = True
        # Het kader is berekend toen dit nog een gat was — op een gat > KADER_GAT_S staat de
        # automatische zoom volledig uit, precies wanneer je nauwkeurig moet werken.
        self.speler.herbereken_kader()

        if bruikbaar:
            # Het skelet staat er; vanaf hier is dit een doodgewoon bewerkbaar frame.
            self._undo.append({'type': 'skelet', 'idx': idx,
                               'oud_lm': oud_lm, 'oud_pose': oud_pose,
                               'nieuw_lm': list(r.lm), 'nieuw_pose': True,
                               'geklikt': set()})
            self._redo.clear()
            self._na_edit()      # doorrekenen + opslaan; het frame telt nu mee
            self.lbl_editor_hint.setText(
                "Skelet overgenomen van de buurframes — sleep de punten naar de juiste plek. "
                f"(Uitvloeien staat op ±{self.spin_uitvloei.value()} frames.)")
            return

        # Niets om over te nemen: punt voor punt vragen. Doorrekenen moet hier al, want
        # zonder lm_data/been/hoek loopt de eerstvolgende hertekening (teken_been_overlay,
        # _update_live_status) stuk op een frame dat zegt een pose te hebben. Opslaan nog
        # niet — de gebruiker kan de reeks nog afbreken.
        self._plaats = {'idx': idx, 'stap': 0, 'geklikt': set(),
                        'oud_lm': oud_lm, 'oud_pose': oud_pose}
        self._herbereken()
        self.speler.volgen_bevroren = True   # uitsnede mag niet verspringen tussen klikken
        self.plaats_balk.setVisible(True)
        self._toon_plaats_stap()

    def _toon_plaats_stap(self):
        """Hint + knopstatus voor de huidige stap; ververst ook het beeld (doelpunt-ring)."""
        if self._plaats is None:
            return
        stap, n = self._plaats['stap'], len(PLAATS_VOLGORDE)
        if stap < n:
            naam = LANDMARK_NAMEN.get(PLAATS_VOLGORDE[stap], f"punt {PLAATS_VOLGORDE[stap]}")
            self.lbl_plaats.setText(f"Klik: {naam}  ({stap + 1} van {n})")
        else:
            self.lbl_plaats.setText(f"Alle {n} punten gehad — leg het skelet vast.")
        self.btn_plaats_vorige.setEnabled(stap > 0)
        self.btn_plaats_over.setEnabled(stap < n)
        self.btn_plaats_klaar.setEnabled(self._plaats_compleet())
        hoeveelste, totaal = self._gat_positie(self._plaats['idx'])
        rest = (f"  ·  frame {hoeveelste} van {totaal} in dit gat" if totaal > 1 else "")
        self.lbl_editor_hint.setText(
            "Rechts slepen = beeld verschuiven, muiswiel = zoomen." + rest)
        self._update_editor_knoppen()
        self.speler.toon_huidig_frame()

    def _plaats_compleet(self):
        """Mag het skelet vastgelegd worden? Alleen als elk meetpunt een zichtbare positie
        heeft — anders belandt er een frame met heup, knie en enkel op één punt in de
        tabel, en dat leest als een afzethoek van 0°."""
        if self._plaats is None:
            return False
        lm = self.resultaten[self._plaats['idx']].lm
        return all(lm[j].visibility >= HANDLE_MIN_VIS for j in PLAATS_VERPLICHT)

    def _plaats_klik(self, event):
        if self._plaats is None:
            return
        stap = self._plaats['stap']
        if stap >= len(PLAATS_VOLGORDE):
            self.lbl_plaats.setText("Alle punten gehad — klik op ✔ Klaar.")
            return
        norm = self.speler.widget_naar_norm(event.position())
        if norm is None or not (0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0):
            # Niet klemmen: dat zou de knie stilzwijgend op de beeldrand leggen.
            self.lbl_editor_hint.setText("Klik binnen het beeld.")
            return
        j = PLAATS_VOLGORDE[stap]
        self._zet_landmark(self._plaats['idx'], j, norm[0], norm[1], vis=1.0)
        self._plaats['geklikt'].add(j)
        self._handmatig.setdefault(self._plaats['idx'], set()).add(j)
        self._plaats['stap'] = stap + 1
        self._toon_plaats_stap()

    def _plaats_vorige(self):
        if self._plaats is not None and self._plaats['stap'] > 0:
            self._plaats['stap'] -= 1
            self._toon_plaats_stap()

    def _plaats_overslaan(self):
        if self._plaats is not None and self._plaats['stap'] < len(PLAATS_VOLGORDE):
            self._plaats['stap'] += 1
            self._toon_plaats_stap()

    def _plaats_klaar(self):
        if self._plaats is None:
            return
        if not self._plaats_compleet():
            ontbreekt = ", ".join(
                LANDMARK_NAMEN.get(j, str(j)) for j in PLAATS_VERPLICHT
                if self.resultaten[self._plaats['idx']].lm[j].visibility < HANDLE_MIN_VIS)
            self.lbl_editor_hint.setText(f"Nog aan te wijzen: {ontbreekt}.")
            return
        # Eerst de state loslaten: _na_edit hertekent, en dat vuurt _speler_frame_getoond
        # weer terug hierheen — met een gevulde _plaats zou dat een lus worden.
        plaats, self._plaats = self._plaats, None
        idx = plaats['idx']
        self._sluit_plaats_balk()
        self._undo.append({'type': 'skelet', 'idx': idx,
                           'oud_lm': plaats['oud_lm'], 'oud_pose': plaats['oud_pose'],
                           'nieuw_lm': list(self.resultaten[idx].lm), 'nieuw_pose': True,
                           'geklikt': set(plaats['geklikt'])})
        self._redo.clear()
        self.speler.herbereken_kader()
        self._na_edit()

    def _plaats_annuleren(self):
        """Terug naar de toestand vóór het plaatsen — het frame is weer een gat."""
        if self._plaats is None:
            return
        plaats, self._plaats = self._plaats, None
        idx = plaats['idx']
        r = self.resultaten[idx]
        r.lm, r.pose_gevonden = plaats['oud_lm'], plaats['oud_pose']
        self._handmatig.pop(idx, None)
        self._sluit_plaats_balk()
        self.speler.herbereken_kader()
        self._herbereken()               # bewust niet opslaan: er is niets veranderd
        self.lbl_editor_hint.setText("Skelet plaatsen geannuleerd.")

    def _sluit_plaats_balk(self):
        self.plaats_balk.setVisible(False)
        self.lbl_plaats.setText("")
        self.speler.volgen_bevroren = False

    def _stop_plaatsen(self):
        """Faalveilige uitgang voor elk pad dat de reeks kan onderbreken (wegnavigeren,
        bewerk-modus uit, paginawissel, andere analyse, venster sluiten). Nooit een half
        skelet laten staan.

        Heeft de gebruiker punten aangewezen en is het skelet bruikbaar, dan blijft dat
        werk behouden. Heeft hij nog niets aangeklikt, dan gaat het weg — anders zou per
        ongeluk wegscrubben stilzwijgend een voorvulling als meting vastleggen (inclusief
        `analyse.bewerkt = 1`) terwijl de gebruiker niets heeft besloten."""
        if self._plaats is None:
            return
        if self._plaats['geklikt'] and self._plaats_compleet():
            self._plaats_klaar()
        else:
            self._plaats_annuleren()

    def _herbereken(self):
        """Afgeleiden + events opnieuw uit de huidige landmarks halen (géén smoothing) en
        de hele weergave bijwerken — zónder op te slaan.

        Los van `_bewaar` omdat een handmatig skelet dat nog geplaatst wordt al wél
        doorgerekend moet zijn (anders tekent de overlay op een lege `lm_data` en klapt
        `teken_been_overlay`/`_update_live_status` eruit), maar nog niét opgeslagen mag
        worden: `bewaar_bewerkte_landmarks` zet `analyse.bewerkt = 1` en legt de pristine
        backup aan, en dat is onomkeerbaar als de gebruiker de reeks annuleert."""
        info = self.video_info
        # `perspectief` moet mee: sinds de kalibratie bewaard wordt, draagt een heropende
        # analyse er een, en zonder dit argument zouden de hoeken na één sleepbeweging
        # stilzwijgend terugvallen op het onvertekende beeldvlak.
        verwerk_afgeleiden(self.resultaten, info.w, info.h, info.fps,
                           self.smooth_n, self.threshold,
                           perspectief=self.perspectief)
        self.events = segmenteer_afzetten(self.resultaten)
        self._vul_tabel()
        self._vul_grafiek()
        self.btn_export.setEnabled(bool(self.events))
        self._update_dekking()
        self.speler.toon_huidig_frame()
        self._update_editor_knoppen()

    def _update_dekking(self):
        """Statusbalk-teller: hoeveel frames hebben een skelet? Frames zonder breken een
        afzetmeting af, dus dit is de maat voor 'hoeveel werk ligt er nog'.

        Bochtframes tellen niet mee — daar valt sowieso niets te meten, dus ze horen niet
        als openstaand werk in de noemer te staan."""
        if not self.resultaten:
            self.lbl_dekking.setText("")
            return
        bocht = sum(1 for r in self.resultaten if r.bocht)
        totaal = len(self.resultaten) - bocht
        met = sum(1 for r in self.resultaten if r.pose_gevonden and not r.bocht)
        tekst = f"Skelet: {met} van {totaal} frames"
        if bocht:
            tekst += f" · {bocht} in de bocht"
        self.lbl_dekking.setText(tekst)
        kleur = "#888" if met == totaal else "#c80"
        self.lbl_dekking.setStyleSheet(f"padding-right: 14px; color: {kleur};")

    def _bepaal_bocht_nu(self):
        """
        Bepaalt de bocht alsnog op een analyse die er nog geen markering voor heeft, en
        haalt die frames uit de meting. Voor alles wat vóór de bochtdetectie is gedraaid:
        er wordt niets opnieuw geanalyseerd, want de landmarks van de héle clip staan al
        in het .npz — daar valt de heupstand zo uit af te lezen. Op zo'n analyse is het
        signaal zelfs betrouwbaarder dan op een verse, waar de bocht juist dun bemeten is.

        Eerst rekenen, dán pas vragen: de gebruiker ziet wat het met zijn tabel doet
        voordat er iets naar de bibliotheek gaat. Bij "nee" gaat alles exact terug.
        """
        if not self.resultaten or self.analyse_id is None:
            return
        info = self.video_info
        oude_vlaggen = [r.bocht for r in self.resultaten]
        oude_events = len(self.events)

        bepaal_bocht_reeks(self.resultaten, info.w, info.h, info.fps)
        n_bocht = sum(1 for r in self.resultaten if r.bocht)
        if n_bocht == 0:
            QMessageBox.information(
                self, "Geen bocht gevonden",
                "In deze analyse staat de schaatser overal frontaal in beeld — er is geen "
                "bocht om uit te sluiten. Er verandert dus niets.")
            self.btn_bocht_nu.setEnabled(False)
            return

        self._herbereken()          # tabel/grafiek tonen al wat het wordt
        verdwenen = oude_events - len(self.events)
        antwoord = QMessageBox.question(
            self, "Bocht bepalen",
            f"{n_bocht} van de {len(self.resultaten)} frames "
            f"({n_bocht / len(self.resultaten):.0%}) liggen in de bocht.\n\n"
            f"Afzetten: {oude_events} → {len(self.events)}"
            + (f" ({verdwenen} vervallen — die zijn in de bocht gemeten "
               f"en dus niet bruikbaar)" if verdwenen > 0 else "") + ".\n\n"
            "Toepassen en opslaan? De landmarks blijven ongewijzigd; alleen de markering "
            "welke frames buiten de meting vallen wordt bewaard.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if antwoord != QMessageBox.Yes:
            for r, b in zip(self.resultaten, oude_vlaggen):
                r.bocht = b
            self._herbereken()
            return

        try:
            schaats_db.bewaar_bochtmarkering(
                self.bieb, self.analyse_id, self.resultaten, info, self.events)
        except Exception as e:
            QMessageBox.warning(
                self, "Niet opgeslagen",
                f"De bochtmarkering kon niet worden opgeslagen:\n\n{e}\n\n"
                "Je ziet hem nu wel, maar bij het opnieuw openen is hij weg.")
            return
        self.btn_bocht_nu.setEnabled(False)
        self._vernieuw_schaatsers()          # gemiddelde hoek in de bibliotheeklijst
        self.statusBar().showMessage(
            f"Bocht bepaald: {n_bocht} frames uitgesloten, opgeslagen.", 8000)

    def _na_edit(self):
        """Na een edit/undo/redo: herberekenen én auto-opslaan naar de bibliotheek."""
        self._herbereken()
        self._bewaar()

    def _bewaar(self):
        info = self.video_info
        if self.analyse_id is not None:
            try:
                schaats_db.bewaar_bewerkte_landmarks(
                    self.bieb, self.analyse_id, self.resultaten, info, self.events)
                self.lbl_editor_hint.setText("Correctie opgeslagen.")
            except Exception as e:
                self.lbl_editor_hint.setText(f"Opslaan mislukt: {e}")
        else:
            self.lbl_editor_hint.setText("Niet opgeslagen (geen bibliotheek-analyse).")

    def _pas_edit_toe(self, edit, kant):
        """Zet één undo-item terug of opnieuw; `kant` is 'oud' of 'nieuw'."""
        if edit.get('type') == 'skelet':
            idx = edit['idx']
            lm, pose = edit[f'{kant}_lm'], edit[f'{kant}_pose']
            r = self.resultaten[idx]
            r.lm = list(lm) if lm is not None else None
            r.pose_gevonden = pose
            # De groene "handmatig"-markering hoort bij een skelet dat er staat.
            if pose and lm is not None:
                self._handmatig[idx] = set(edit.get('geklikt', ()))
            else:
                self._handmatig.pop(idx, None)
            self.speler.herbereken_kader()   # de dekking veranderde, dus de auto-zoom ook
            return
        j = edit['j']
        for f, lm in edit[kant].items():
            self.resultaten[f].lm[j] = lm

    def _undo_edit(self):
        if not (self._editor_actief and self._undo) or self._plaats is not None:
            return
        edit = self._undo.pop()
        self._pas_edit_toe(edit, 'oud')
        self._redo.append(edit)
        self._na_edit()

    def _redo_edit(self):
        if not (self._editor_actief and self._redo) or self._plaats is not None:
            return
        edit = self._redo.pop()
        self._pas_edit_toe(edit, 'nieuw')
        self._undo.append(edit)
        self._na_edit()

    def _herstel_origineel(self):
        if self.analyse_id is None:
            return
        if QMessageBox.question(
                self, "Herstel origineel",
                "Alle handmatige correcties van deze analyse ongedaan maken en terug naar "
                "de oorspronkelijke detectie?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            hersteld = schaats_db.herstel_originele_landmarks(self.bieb, self.analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Herstel origineel", f"Mislukt:\n\n{e}")
            return
        if not hersteld:
            QMessageBox.information(
                self, "Herstel origineel",
                "Deze analyse is nog niet bewerkt — er is niets te herstellen.")
            return
        try:
            data = schaats_db.laad_analyse(self.bieb, self.analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Herstel origineel", f"Herladen mislukt:\n\n{e}")
            return
        info, resultaten = data["info"], data["resultaten"]
        # Ook hier de kalibratie meegeven — "origineel herstellen" gaat over de
        # landmarks, niet over de perspectiefcorrectie.
        verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, self.smooth_n,
                           self.threshold, perspectief=self.perspectief)
        events = segmenteer_afzetten(resultaten)
        try:
            schaats_db.ververs_events_cache(self.bieb, self.analyse_id, events)
        except Exception:
            pass
        self._undo.clear()
        self._redo.clear()
        self._handmatig.clear()
        self._toon_resultaten(info, resultaten, events, bron=data["meta"]["titel"])
        self._update_editor_knoppen()
        self.lbl_editor_hint.setText("Origineel hersteld.")

    def _markeer_actieve_rij(self, idx):
        for i, ev in enumerate(self.events):
            if ev.start_frame <= idx <= ev.eind_frame:
                if self.tabel.currentRow() != i:
                    self.tabel.blockSignals(True)
                    self.tabel.selectRow(i)
                    self.tabel.blockSignals(False)
                return

    def _klik_op_rij(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self.speler.ga_naar(self.events[rij].start_frame)

    # ── Export ───────────────────────────────────────────────────────────
    def _exporteer_csv(self):
        pad, _ = QFileDialog.getSaveFileName(self, "Exporteer afzethoeken", "afzethoeken.csv", "CSV (*.csv)")
        if not pad:
            return
        met_corr = any(ev.correctie is not None for ev in self.events)
        with open(pad, "w", newline="", encoding="utf-8") as f:
            schrijver = csv.writer(f)
            # `onvolledig` draagt de réden (leeg = volwaardige meting); dat is meer waard in
            # een sheet dan de oude 0/1-kolom `afgekapt`, die maar één van de twee dekte.
            kop = ["#", "been", "start_tijd_s", "eind_tijd_s", "hoek_deg", "min_hoek_deg",
                   "max_hoek_deg", "onvolledig"]
            if met_corr:
                kop += ["correctie_deg", "betrouwbaar", "snelheid_ms", "slaglengte_m"]
            schrijver.writerow(kop)
            for i, ev in enumerate(self.events):
                rij = [i + 1, ev.been, f"{ev.start_tijd:.3f}", f"{ev.eind_tijd:.3f}",
                       f"{ev.hoek:.1f}", f"{ev.min_hoek:.1f}", f"{ev.max_hoek:.1f}",
                       ev.onvolledig or ""]
                if met_corr:
                    rij += [f"{ev.correctie:+.1f}" if ev.correctie is not None else "",
                            int(bool(ev.betrouwbaar)),
                            f"{ev.snelheid:.2f}" if ev.snelheid is not None else "",
                            f"{ev.slaglengte:.2f}" if ev.slaglengte is not None else ""]
                schrijver.writerow(rij)
        self.statusBar().showMessage(f"Geëxporteerd naar {pad}", 5000)

    def _actieve_workers(self):
        return [w for w in (self.worker, self.batch_worker)
                if w is not None and w.isRunning()]

    def _wacht_op_worker(self, worker, seconden=120):
        """Wacht tot de thread écht gestopt is, met een wachtcursor en een levende UI.
        De afbreek-check zit op de framegrens (bij YOLO ~2 s/frame) en een al begonnen
        videokopie wordt bewust afgemaakt — daarom een ruime deadline i.p.v. terminate()."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            deadline = time.monotonic() + seconden
            while not worker.wait(100):
                # Alleen hertekenen; geen muis/toets, anders klikt de gebruiker in een
                # venster dat al aan het sluiten is.
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                if time.monotonic() > deadline:
                    return False
        finally:
            QApplication.restoreOverrideCursor()
        return True

    def _stop_workers(self):
        """Breekt een lopende (batch-)analyse netjes af vóór het afsluiten. Retourneert
        False als er niet afgesloten mag worden (gebruiker ziet ervan af, of de thread
        is nog niet gestopt) — een QThread vernietigen terwijl hij draait is een crash."""
        actief = self._actieve_workers()
        if not actief:
            return True
        antwoord = QMessageBox.question(
            self, "Analyse loopt nog",
            "Er draait nog een analyse op de achtergrond.\n\n"
            "Afsluiten breekt die af; de video wordt dan niet in de bibliotheek "
            "opgeslagen. Al afgeronde video's van een batch blijven wel bewaard.\n\n"
            "Toch afsluiten?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if antwoord != QMessageBox.Yes:
            return False

        # Vlag + blockSignals: nieuwe signalen komen niet meer, en een signaal dat al in
        # de wachtrij stond mag geen dialoog of paginawissel meer opleveren tijdens het
        # afsluiten (de slots controleren `_afsluiten`).
        self._afsluiten = True
        for w in actief:
            w.blockSignals(True)
            w.breek_af()
        self.lbl_voortgang.setText("Analyse afbreken...")
        for w in actief:
            if not self._wacht_op_worker(w):
                self._afsluiten = False
                for x in actief:
                    x.blockSignals(False)
                QMessageBox.warning(
                    self, "Analyse stopt nog niet",
                    "De analyse reageert nog niet op het afbreken — waarschijnlijk wordt "
                    "de video nog naar de bibliotheek gekopieerd. Die kopie wordt niet "
                    "halverwege afgekapt.\n\nHet venster blijft open; probeer het zo nog "
                    "een keer af te sluiten.")
                return False
        return True

    def closeEvent(self, event):
        if not self._stop_workers():
            event.ignore()
            return
        self._stop_lokaal_proef()
        self._stop_plaatsen()   # een lopende reeks nog vastleggen of terugdraaien
        self._pauzeer_alles()
        self.speler.sluit()
        self.kant_links.leeg()
        self.kant_rechts.leeg()
        self._ruim_knipmap_op()   # geknipte fragmenten die niet meer geanalyseerd worden
        super().closeEvent(event)


def main():
    # De QApplication en het opstartscherm bestaan al sinds de import bovenaan dit bestand
    # (zie _start_opstartscherm); alleen als deze module via een omweg wordt gestart, zijn
    # ze er niet.
    app = _APP or QApplication(sys.argv)
    venster = MainWindow(melding=_SPLASH.melding if _SPLASH else None)
    venster.show()
    venster._melding = lambda tekst: None    # het opstartscherm gaat nu dicht
    if _SPLASH:
        _SPLASH.finish(venster)
    # Pas nu torch/ultralytics binnenhalen: het venster staat er, de gebruiker kan al door
    # de bibliotheek bladeren, en tegen de tijd dat hij een analyse start is de backend er.
    _warm_backend_op()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
