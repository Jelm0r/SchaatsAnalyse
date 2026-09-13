"""
SkateAnalysis GUI
=================
Visual interface for skate_analysis.py: plays the video back with the
skeleton/push-leg overlay live on top, and shows a table of all push angles.

Usage:
    python skate_gui.py

Requirements (besides skate_analysis.py's own dependencies):
    pip install PySide6
"""

import os
import sys
import csv
import math
import shutil
import tempfile
import queue
import threading
import time

# ── Output first: when frozen there's no console ────────────────────────────────
# Before *all* other imports, because a bundled .exe (PyInstaller --windowed) has no
# console: sys.stdout/stderr are then None and anything that actually touches that
# stream breaks — the tqdm bar from ultralytics/rtmlib, the logging handler ultralytics
# hooks up on import, every sys.stdout.write. So this must be sorted before the first of
# those imports, and thus before _start_splash_screen() below, which already puts up a
# window at module level. skate_environment is stdlib-only: ~1 ms, threading was already
# imported. In the repo environment nothing happens unless SKATEANALYSIS_LOG is set.
import skate_environment

LOGPATH = skate_environment.start_log() if __name__ == "__main__" else None

# The safety net below, for the same reason and at the same moment: a crash in Qt or in
# a numerics library happens in C++ and leaves nothing behind without this — no
# traceback, no shutdown message, and when frozen no console either (25-8-2026:
# 0xc0000005 in Qt6Gui.dll during a batch analysis, see TODO_CRASH.md). `start_crashlog()`
# writes the stack to the same log file and also catches unhandled errors from plain
# threads, which would otherwise vanish without a trace here (the backend warmup, the
# local-probe on the recordings). Unlike the log, this also runs in the repo environment:
# that's exactly where debugging happens.
if __name__ == "__main__":
    skate_environment.start_crashlog()

# ── Qt first, and put up a splash screen right away ─────────────────────────────
# Deliberately before all other imports: the rest of this module pulls in cv2/numpy and
# (on first use) torch/ultralytics, and on a cold machine that costs seconds during which
# nothing happens on screen and the user thinks the app didn't start. Only the Qt import
# (~0.1 s) precedes it, so that within a fraction of a second there's a little window
# telling the user what's happening.
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QPointF, QEventLoop, QEvent, QSize, QRect, QPoint,
    QMargins, QObject, QtMsgType, qInstallMessageHandler,
)
from PySide6.QtGui import (
    QImage, QPixmap, QAction, QColor, QPainter, QPen, QPainterPath, QShortcut,
    QKeySequence, QFont,
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


class SplashScreen(QSplashScreen):
    """The little window that shows, during startup, that something is happening.

    Drawn in code (no image file): loading a picture would be disk access again at
    exactly the moment we want to avoid that.
    """
    WIDTH, HEIGHT = 460, 180

    def __init__(self):
        super().__init__(self._background())
        # QSplashScreen is always-on-top by default. During startup a modal message can
        # come up (library unreachable, conflict copy) and it would then fall *behind*
        # the splash screen — an app that looks frozen. Hence turning that off.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)

    @classmethod
    def _background(cls):
        pm = QPixmap(cls.WIDTH, cls.HEIGHT)
        pm.fill(QColor(24, 40, 66))
        p = QPainter(pm)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont(p.font().family(), 22, QFont.Bold))
        p.drawText(QRect(0, 40, cls.WIDTH, 44), Qt.AlignCenter, "Schaats Analyse")
        p.setPen(QColor(150, 180, 220))
        p.setFont(QFont(p.font().family(), 9))
        p.drawText(QRect(0, 84, cls.WIDTH, 22), Qt.AlignCenter, "bezig met opstarten...")
        p.end()
        return pm

    def melding(self, tekst):
        """Sets the status line and paints it right away: between two messages no event
        loop runs (we're still in startup code), so without processEvents the screen
        would stay on the first piece of text."""
        self.showMessage(f"  {tekst}", Qt.AlignBottom | Qt.AlignLeft,
                         QColor(220, 232, 248))
        QApplication.processEvents()


def _start_splash_screen():
    """Creates the QApplication and puts up the splash screen. Returns (app, screen).

    Called at module level — before the heavy imports below — and only when this file
    is run as a program; on `import skate_gui` (self-tests, measurement scripts)
    nothing happens.
    """
    app = QApplication(sys.argv)
    scherm = SplashScreen()
    scherm.show()
    scherm.melding("Onderdelen laden...")
    return app, scherm


def _qt_to_log():
    """Sends Qt's own messages to the log.

    Needed because on Windows, without a console, Qt writes to the debugger
    (OutputDebugString) rather than to stderr: precisely the warnings that precede a
    crash in the render layer — "Cannot set parent, new parent is in a different
    thread", "It is not safe to use pixmaps outside the GUI thread", "Timers cannot be
    stopped from another thread" — are therefore invisible, while they point exactly at
    what's going wrong. Captured from here on, so also during the heavy imports.
    """
    kinds = {QtMsgType.QtDebugMsg: "debug", QtMsgType.QtInfoMsg: "info",
             QtMsgType.QtWarningMsg: "WARNING", QtMsgType.QtCriticalMsg: "CRITICAL",
             QtMsgType.QtFatalMsg: "FATAL"}

    def handler(kind, context, text):
        try:
            location = ""
            if context is not None and context.file:
                location = " (%s:%s)" % (context.file, context.line)
            sys.stderr.write("[Qt %s] %s%s\n" % (kinds.get(kind, "?"), text, location))
        except (OSError, ValueError, AttributeError):
            pass

    qInstallMessageHandler(handler)


def _screens_to_log(app):
    """Writes every change in the screen list to the log.

    The crash from TODO_CRASH.md is a `QScreen` that Qt had already thrown away:
    `0xc0000005` on `QScreen::geometry()`+0 and on `QScreen::virtualSiblings()`+0x29 — a
    crash on the very first bytes of a member function, so a broken `this` rather than a
    null reference. That's use-after-free, and Qt only discards a `QScreen` when Windows
    rebuilds the screen list.

    On this machine (two displays) that also happens without anyone doing anything: a
    monitor dropping into power save, a connection retraining, a driver doing a mode
    switch under GPU load. Win+Shift+S was thus never the cause, only a handy way to
    trigger it — which explains why the crash also occurs when nothing is touched.

    Without these lines that's invisible: the log just stops. With these lines, the next
    crash will show in black and white whether a screen appeared, disappeared, or
    changed size just before it, and how many seconds before. Pure measurement — nothing
    is fixed by this.
    """
    def write_log(text):
        try:
            sys.stderr.write("[screen %s] %s\n" % (time.strftime("%H:%M:%S"), text))
        except (OSError, ValueError, AttributeError):
            pass

    def describe(screen):
        # On 'removed' the QScreen object is still valid during the signal, but we read
        # it defensively: this is diagnostic code and must never itself cause a crash.
        try:
            g = screen.geometry()
            return "%s %dx%d at (%d,%d) @%.0fHz" % (screen.name(), g.width(), g.height(),
                                                    g.x(), g.y(), screen.refreshRate())
        except (RuntimeError, AttributeError):
            return "<screen no longer readable>"

    def track(screen):
        s = screen      # default argument in every lambda: otherwise they'd all capture the last loop value
        s.geometryChanged.connect(lambda _v, s=s: write_log("geometry: " + describe(s)))
        s.availableGeometryChanged.connect(lambda _v, s=s: write_log("work area: " + describe(s)))
        s.refreshRateChanged.connect(lambda _v, s=s: write_log("refresh rate: " + describe(s)))
        s.logicalDotsPerInchChanged.connect(lambda _v, s=s: write_log("DPI: " + describe(s)))

    for screen in app.screens():
        track(screen)
    write_log("at start: " + " | ".join(describe(s) for s in app.screens()))

    def added(screen):
        track(screen)
        write_log("SCREEN ADDED: " + describe(screen))

    app.screenAdded.connect(added)
    app.screenRemoved.connect(lambda s: write_log("SCREEN REMOVED: " + describe(s)))
    app.primaryScreenChanged.connect(lambda s: write_log("primary screen now: " + describe(s)))


if __name__ == "__main__":
    _qt_to_log()

_APP, _SPLASH = _start_splash_screen() if __name__ == "__main__" else (None, None)

if _APP is not None:
    # Right after creating the QApplication, so a screen change during the heavy
    # imports is visible too.
    _screens_to_log(_APP)

import cv2
import numpy as np

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

import skate_db
import skate_perspective
from skate_analysis import (
    segment_pushes, draw_overlay_on_frame, horizon_angle_from_line,
    detect_ice_line, PerspectiveConfig, process_derivatives, Landmark,
    torso_centroid, box_sequence, make_prefill, determine_corner_sequence,
    FrameResult, video_info, trim_fragments, TrimAborted,
    INCOMPLETE_TRUNCATED, INCOMPLETE_NO_PUSH, app_dir, is_frozen,
    open_video, is_interlaced,
)

# Interlacing (combing): a camcorder recording 1080i weaves two moments 1/50 s apart into
# one frame. A media player deinterlaces on playback, OpenCV doesn't — so without a
# filter both the trainer and the pose detector see the comb. See `deinterlace` in
# skate_analysis for the measurement behind this.
DEINT_TOOLTIP = (
    "Camcorderbeeld (1080i) bestaat uit twee halve beelden van 1/50 s uit elkaar,\n"
    "samengeweven tot één frame. Op een bewegend been staan die twee helften op een\n"
    "andere plek — de kamtanden die je in beeld ziet.\n\n"
    "Gemeten op dit soort materiaal liggen de twee helften op knieën en enkels 6 px\n"
    "uit elkaar; met '2 px keypointfout = 2-4° hoekfout' is dat de grootste ruisbron\n"
    "die er in zulke opnames zit. Het filter haalt ze eruit.\n\n"
    "Wordt per video zelf vastgesteld; progressief materiaal (telefoon, GoPro) wordt\n"
    "niet aangeraakt. Uitzetten alleen om een A/B te draaien.")

# Skeleton editor (phase 3)
# The handle/grab radius scales with the skater: a fixed fraction of the on-screen torso
# length, clamped to [HANDLE_MIN_PX, HANDLE_MAX_PX]. That way the little dots look the
# same size at every skater size *and* zoom level, and stop overlapping when the skater
# is small in frame. Tuning knobs:
HANDLE_FRAC     = 0.06   # handle radius as a fraction of the torso length (torso ~200 px -> ~12 px)
HANDLE_MIN_PX   = 4      # lower bound in screen pixels (a small skater keeps a clickable handle)
HANDLE_MAX_PX   = 14     # upper bound in screen pixels (no giant blobs on a close-up)
HANDLE_MIN_VIS = 0.2    # below this visibility, no draggable handle (same as draw_all_landmarks)

# MediaPipe-33 landmark index -> body part name (for the hover text in the editor).
# "left"/"right" is anatomical (the skater's own left/right side), same as in the
# detection/L-R fixer. Indices that don't occur get a generic fallback.
LANDMARK_NAMES = {
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

# Manually placing a skeleton on a frame with no pose. The order runs top to bottom and
# left-first per pair, so the user gets a steady rhythm. Asking for more points has no
# point: this is exactly what the measurements use (hip/knee/ankle for the push, knee and
# extension angle) plus the shoulders for the torso, which the handle radius and the
# automatic zoom rely on. Heel and toe (29-32) stay at visibility 0 -- same as the YOLO
# backend without RTMPose, which doesn't know them either.
PLACEMENT_ORDER  = (11, 12, 23, 24, 25, 26, 27, 28)
# Without these six there's no measurement possible; the skeleton can only be committed
# once all of them have a visible position (clicked, or taken over from the prefill).
PLACEMENT_REQUIRED = (23, 24, 25, 26, 27, 28)

# Zooming in on the skater in the view
ZOOM_MAX  = 5.0        # max zoom factor of the view crop (manual: slider/wheel)
ZOOM_STEP = 1.25       # mouse-wheel factor per notch
# Automatic zoom is allowed to go in further than the manual limit: a skater who's far
# away at the start of the clip genuinely needs to be magnified to fill the frame.
# At 4K a 1/8 crop is still 480x270 px; below that it gets too soft.
ZOOM_AUTO_MAX = 8.0    # upper bound of automatic zoom
# ...but never further than there are still pixels for. What counts isn't the video
# resolution but how much the crop gets blown up on screen: a portrait phone clip
# already sits heavily shrunk in a landscape panel (lots of room to zoom into), a
# landscape 4K clip barely at all. The user doesn't choose the zoom here themselves, so
# the program shouldn't serve up mush.
BOX_MAX_MAGNIFICATION = 2.5   # max. screen pixels per video pixel during automatic zoom
# Breathing room around the skater, as a fraction of the space it needs itself. The
# landmarks stop at the nose and the toes, while the top of the head and the skate
# blades stick out beyond that -- and a skater pinned right against the edge doesn't
# look good.
BOX_MARGIN = 0.15

# ── Drawing on the image ─────────────────────────────────────────────────────────
# Trainer annotations: a sketch or a straight line over the image ("look, your knee
# caves in here"). They belong to the clip, not to one frame -- you draw something and
# then play on to see if it holds up -- so they stay put while the video keeps playing.
# They're stored in **frame-normalized** coordinates (like the landmarks), so they stay
# glued to their spot on the image when zooming and panning; in widget coordinates they'd
# end up next to the skater after the very first zoom step.
DRAW_PAN, DRAW_SKETCH, DRAW_LINE = "pan", "sketch", "line"
DRAW_MODES = (("✋ Schuiven", DRAW_PAN),
              ("✏ Schetsen", DRAW_SKETCH),
              ("📏 Lijn", DRAW_LINE))
DRAW_TOOLTIP = (
    "Wat de linkermuisknop op het beeld doet:\n"
    "  ✋ Schuiven — het ingezoomde beeld verslepen (en in de bewerk-modus punten slepen)\n"
    "  ✏ Schetsen — vrij tekenen zolang je de knop ingedrukt houdt\n"
    "  📏 Lijn — een rechte lijn van indrukken tot loslaten\n"
    "\n"
    "Rechts slepen schuift het beeld altijd, ook midden in het tekenen.\n"
    "Een tekening hoort bij de clip en niet bij één frame: hij blijft staan terwijl de\n"
    "video doorloopt, en schuift mee met zoomen en pannen.")
# Magenta doesn't clash with the overlay (white skeleton, green/red push leg, yellow
# handles) and stays visible on both white ice and a dark suit.
DRAW_COLOR = (255, 45, 210)
DRAW_THICKNESS = 3        # screen pixels, so equally thick at every zoom level (same as the handles)
DRAW_OPACITY = 0.7    # 70%: the image must still be visible through it
# Below this amount of movement (as a fraction of the image size) a drag is just a
# click. Otherwise it would leave behind a dot that's only in the way.
DRAW_MIN_DRAG = 0.005


def _drag_distance(punten):
    """Largest distance to the start point of a stroke (normalized)."""
    x0, y0 = punten[0]
    return max(math.hypot(x - x0, y - y0) for x, y in punten)


# Corner detection: explanation for the checkbox in both analysis dialogs (one text, two spots).
CORNER_TOOLTIP = (
    "Herkent aan de stand van de heupen wanneer de schaatser niet frontaal in beeld is\n"
    "(in de bocht staan ze achter elkaar i.p.v. naast elkaar).\n"
    "\n"
    "Die frames worden dan grotendeels niet meer door de detector gehaald — dat scheelt\n"
    "flink in analysetijd — en ze leveren geen afzetmeting op. Elke ~0,3 s wordt gekeken\n"
    "of het rechte stuk alweer begonnen is, dus een clip die ín de bocht begint pakt de\n"
    "meting vanzelf op zodra de schaatser recht op de camera af komt.\n"
    "\n"
    "Uitzetten alleen om te zien wat er in de bocht gebeurt; die hoeken zijn niet bruikbaar.")

# Perspective correction (phase 7). Still experimental: the math and the pipeline hookup
# are there and the calibration now gets saved, but the correction hasn't been validated
# on real material yet (ROADMAP phase 7, step 3). Hence "experimental" and not "doesn't
# work" -- it's only on when you deliberately measure with it.
PERSPECTIVE_TOOLTIP = (
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

PERSPECTIVE_TOOLTIP_BATCH = (
    PERSPECTIVE_TOOLTIP + "\n"
    "\n"
    "In een batch wordt de kalibratie ÉÉN keer gevraagd en op alle clips toegepast —\n"
    "ze komen immers uit dezelfde camerastand. Dat is ook de voorwaarde om hun hoeken\n"
    "onderling te mogen vergelijken.")


def _calibration_rows(inst):
    """Info rows about the saved perspective calibration (empty if there isn't one).

    Shows the input (number of lines, line distance, method, lower-leg length) and the
    *recomputed* outcome (f, camera height, residual). The latter doesn't come from
    storage but is worked out again here -- exactly like when the analysis is opened --
    so the Info dialog shows what the analysis would actually use *now*.

    The top-level `perspectief` key (and the row labels/text below) are
    `instellingen_json` content and stay Dutch here on purpose, deferred to Phase 8d
    together with `doel_punt`/`doel_kader`/etc (see TRANSLATION_PROGRESS.md). The
    *nested* calibration-input dict, though, is read through
    `CalibrationInput.from_dict()` rather than by indexing raw keys directly: that
    dict's keys have been English since Phase 3 (`track_lines`, `image_w`, ... -- see
    `CalibrationInput.to_dict()`), and `from_dict()` already dual-reads the old Dutch
    spellings for calibrations saved before that. **Found and fixed a real bug here**:
    the previous version of this function indexed the old Dutch keys directly
    (`invoer.get("rijlijnen")` etc.), which no longer matched anything `to_dict()`
    actually writes -- so the Info dialog silently showed zero calibration rows for
    every perspective-corrected analysis saved since Phase 3 landed (confirmed with a
    throwaway round-trip: the old code returned `[]` for a `PerspectiveConfig.to_dict()`
    dict, this version returns the expected rows)."""
    p = inst.get("perspectief")
    if not p:
        return []
    inv_dict = p.get("calibration_input") or p.get("invoer")
    if not inv_dict:
        return []
    inv = skate_perspective.CalibrationInput.from_dict(inv_dict)
    # `method`/`methode` may still hold an old on-disk value ('onderbeen'/'beenvlak',
    # from an analysis saved before Phase 3) -- same normalization as the shim in
    # `skate_perspective.reconstruct_angle()`.
    methode_raw = p.get("method", p.get("methode"))
    methode = {"lower_leg": "onderbeenlengte (bol-snijding)", "onderbeen": "onderbeenlengte (bol-snijding)",
              "leg_plane": "beenvlak (rijrichting)", "beenvlak": "beenvlak (rijrichting)"
              }.get(methode_raw, methode_raw)
    onderbeen_l = p.get("lower_leg_l", p.get("onderbeen_l"))
    rows = [
        ("Kalibratie:", f"{len(inv.track_lines)} baanlijnen + {len(inv.cross_lines)} dwarslijnen, "
                        f"{inv.line_distance} m uit elkaar, "
                        f"op beeld {inv.image_w}×{inv.image_h}",
         "De nagetrokken lijnen worden bewaard; de camerastand wordt eruit herberekend."),
        ("Reconstructie:", methode
         + (f", onderbeen {onderbeen_l * 100:.1f} cm" if onderbeen_l else ""),
         None),
    ]
    if inv.note:
        rows.append(("Kalibratie-notitie:", inv.note, None))
    try:
        kal = inv.calibrate()
        rows.append(("Camerastand:",
                      f"f = {kal.f:.0f} px{' (geschat)' if kal.f_estimated else ''}, "
                      f"hoogte {kal.camera_height:.1f} m, horizon {kal.horizon_deg:+.2f}°, "
                      f"residu {kal.residual_px:.1f} px", None))
    except Exception as e:
        rows.append(("Camerastand:", f"niet herberekenbaar: {e}", None))
    return rows

# File filter for every video picker, derived from the extensions the library itself
# accepts — otherwise `.mts` (AVCHD camcorder, exactly the interlaced material from
# OPNAME.md) is present in the recordings folder but not in the file picker.
VIDEO_FILTER = ("Video's (" + " ".join("*" + e for e in skate_db.VIDEO_EXTS) + ");;"
                "Alle bestanden (*)")

# Columns of the recordings table (phase 8). Named as numbers because cell widgets and an
# itemChanged filter hang off them: an extra column must never become a silent shift.
# NOTE: kept Dutch (`OPNAME_` = recording) for now, along with the other VideoPlayer/
# opnametab-adjacent identifiers below down to the backend-selection section — they
# belong with a future Phase 8 session that translates the recordings tab/VideoPlayer,
# not this module-infrastructure one. See TRANSLATION_PROGRESS.md.
OPNAME_KOL_NAAM, OPNAME_KOL_DUUR, OPNAME_KOL_LOKAAL = 0, 1, 2
OPNAME_KOL_STATUS, OPNAME_KOL_TELLING, OPNAME_KOL_NOTITIE = 3, 4, 5


def _opname_sleutel(bron):
    """Identity of a row in the recordings table: (library, id). The id alone isn't
    enough — the list shows both the shared work list and the loose videos from the
    local library, two databases whose ids both start at 1."""
    return (bron["bieb"], bron["id"])

# Display for `skate_db.file_is_local`: (text, color, explanation). Without this column
# there's no signal at all that a recording is still in the cloud — the file *is* there,
# after all, it just comes in agonizingly slowly. See `_opname_beschikbaar` for why these
# numbers.
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

# Trim window (phase 8): above this jump, a `fast_seek` ("fast seek") player seeks
# instead of scrubbing sequentially. Letting small jumps run sequentially keeps
# frame-by-frame stepping and plain playback exact — and it's right around a fragment
# boundary that you step frame by frame.
SEEK_THRESHOLD_FRAMES = 30

# Playback speeds: (label, factor on the fps). 1.0 = real speed, lower = slow motion.
# Above 1x is meant for scanning through a long recording (the phase 8 trim window):
# there, decoding doesn't speed up but frames get **skipped** (see _play_tick), because
# 8x real speed outruns every decoder.
SPEEDS = [("8×", 8.0), ("4×", 4.0), ("2×", 2.0),
             ("1×", 1.0), ("½×", 0.5), ("¼×", 0.25), ("⅛×", 0.125), ("1/16×", 0.0625)]


def _speed_idx(factor):
    """Index of a speed in SPEEDS, looked up by factor instead of hardcoded —
    otherwise every speed added later silently shifts the defaults."""
    return next(i for i, (_, f) in enumerate(SPEEDS) if f == factor)


SPEED_DEFAULT_IDX = _speed_idx(1.0)

# The playback timer fires **faster than the frame rate**. `_play_tick` reads the target
# frame off the wall clock and returns immediately if no new frame is due yet, so an
# empty tick costs microseconds. If the timer fired exactly once per frame, every tick
# that's a few ms late would immediately cost a whole frame — and that's not an edge
# case: at 59.22 fps (a screen recording of a TV broadcast) the interval is 16 ms while a
# frame takes 16.886 ms, so the slack is 0.9 ms and ordinary Windows timer jitter eats
# that up. At 25 fps this wasn't noticeable, because there the slack is over 20 ms —
# which is why it only showed up on high-fps material.
#
# Measured on "kjeld in inzell" (59.22 fps, 1180x670, this screen at dpr 2, three 6 s runs
# per setting): one tick per frame delivers 92-93% of frames with 13-17 **double steps**,
# a third of that (oversampling) 96-97% with 2. The number of double steps is the metric
# that matters here — that's what reads as stutter; the tick itself only costs ~5 ms, so
# there was no shortage of compute time, only of hitting the window, and the percentage
# alone wouldn't reveal the bug. Oversampling further (1/4) gained nothing more.
PLAY_OVERSAMPLE = 3
# Lower bound, so an extremely high frame rate doesn't set the timer to hundreds of empty
# ticks per second.
PLAY_TICK_MIN_MS = 4

# Scrubbing with . and , — everywhere in the app, see `PlayerKeys`. 6x the recording
# speed: fast enough to get through half an hour, slow enough to see when you've shot
# past something. The tick is an upper bound on smoothness — the target frame follows
# from the wall clock, so it stays 6x even if the decoder can't keep up (see
# PlayerKeys._scrub_tick).
SCRUB_FACTOR = 6.0
SCRUB_TICK_MS = 40

# One single list of the default keys, so every video window can show the same line and
# no window grows its own (and thus, over time, diverging) list. A window's extras get
# appended with `keys_help()`.
VIDEO_KEYS_HELP = (
    "<b>Spatie</b> afspelen/pauze · <b>.</b> doorspoelen 6× · <b>,</b> terugspoelen 6× · "
    "<b>&larr;/&rarr;</b> één frame · <b>Home/End</b> begin/eind · muiswiel zoomt · "
    "<b>F11</b> volledig scherm")

# The same list as plain text, for a tooltip (which doesn't understand HTML markup).
VIDEO_KEYS_TOOLTIP = (
    "Toetsen: spatie = afspelen/pauze, ← → = één frame, . en , = spoelen op 6×\n"
    "zolang je de toets ingedrukt houdt, Home/End = begin/eind, F11 = volledig scherm.")


def keys_help(*extra):
    """The default keys plus the window's own keys, as one help line."""
    return " · ".join((VIDEO_KEYS_HELP,) + tuple(extra))


def toggle_fullscreen(venster):
    """F11 on every video window. Deliberately `setWindowState` and not `showNormal()`:
    the latter also clears a maximized state, so the main window would suddenly be small
    after leaving F11."""
    if venster.isFullScreen():
        venster.setWindowState(venster.windowState() & ~Qt.WindowFullScreen)
    else:
        venster.setWindowState(venster.windowState() | Qt.WindowFullScreen)

# Width of the transport buttons (⏮ ⏪ ▶ ⏩ ⏭): they carry a single glyph, so Qt's
# default width for text buttons is wasted space on a narrow screen.
TRANSPORT_BUTTON_WIDTH = 46

# Compare page: **upper bound** on the interval of the master clock driving both videos
# at once. The target frame follows from the wall-clock time, so the clock self-corrects
# and no drift builds up — but it can never show more frames than it ticks, and a fixed
# 30 ms is less than two frames on high-fps material. Measured on two Kjeld analyses
# (59.22 fps) side by side at 1x: 55% of the frames, 32 fps on screen, every second frame
# dropped. `MasterClock._interval_ms` therefore computes the real interval from the
# fastest side; this value is now only the ceiling for slow clips and slow motion.
ALL_TICK_MS = 30
# Decoding two videos at once can't hit 1x anyway, and a trainer is watching technique:
# default ¼×.
ALL_SPEED_IDX = _speed_idx(0.25)

# ── Backend selection ────────────────────────────────────────────────────────────
# Use YOLO-pose + ByteTrack if torch/ultralytics is available (then run the app under
# .venv-yolo), otherwise fall back to the MediaPipe backend.
#
# `import skate_yolo` pulls in torch + ultralytics: ~2.8 s on a warm machine and a
# multiple of that cold (Windows scans those hundreds of MB of DLLs). That's two-thirds
# of the startup time, while the start page only shows the library — the backend is only
# needed once an analysis actually begins. Hence only the cheap question "is the package
# installed?" here (find_spec: ~1 ms, imports nothing) and the real import lazily, via
# `analyze_backend()`. Right after the window is shown it gets warmed up in the
# background already (`_warm_backend_up`), so the first analysis doesn't notice.
def _backend_available():
    """(yolo?, rtmpose?) purely based on installed packages, without loading them."""
    from importlib.util import find_spec
    try:
        yolo = find_spec("ultralytics") is not None and find_spec("torch") is not None
        return yolo, yolo and find_spec("rtmlib") is not None
    except (ImportError, ValueError):     # broken install: then MediaPipe
        return False, False


_HAS_YOLO, _HAS_RTMPOSE = _backend_available()
IS_YOLO = _HAS_YOLO
# Prediction of skate_yolo.BACKEND_NAME (that module isn't loaded yet). Once it is, this
# name gets replaced by its own — so a mismatch corrects itself, and the name saved with
# an analysis always comes from the backend itself.
BACKEND_NAME = (("YOLO-pose + ByteTrack + RTMPose-verfijning" if _HAS_RTMPOSE
                 else "YOLO-pose + ByteTrack") if IS_YOLO else "MediaPipe")

_backend_slot = threading.Lock()
_backend_fn = None
BACKEND_ERROR = ""      # filled in if the YOLO import looked installed but still failed


def _backend_broken(*_args, **_kwargs):
    """Analysis entry point if the only bundled backend failed to load (frozen only).

    MediaPipe isn't in the bundled package, so falling back to it would only surface as
    an ImportError halfway through the first analysis. Better to raise one clear error
    right here; `_warn_backend_fallback()` will already have warned the user by then."""
    raise RuntimeError("The analysis backend could not be loaded:\n\n"
                       f"{BACKEND_ERROR}")


def _load_backend():
    """Imports the chosen backend (once) and returns its `analyze` function.

    If the YOLO import fails after all — the package was there but broken, e.g. a torch
    with missing DLLs — the app falls back to MediaPipe here instead of letting the
    analysis crash, and `IS_YOLO`/`BACKEND_NAME` get pulled along. That must *not* happen
    silently (it's a different detector and thus a different measurement), so the reason
    is kept in `BACKEND_ERROR` and reported by the GUI as soon as an analysis starts.

    In a bundled .exe that fallback doesn't exist: MediaPipe isn't in the package
    (`IS_YOLO` is always true there). Then the backend name stays put and this yields
    `_backend_broken`, so there's one clear error instead of an ImportError deep inside
    the first analysis.
    """
    global _backend_fn, IS_YOLO, BACKEND_NAME, BACKEND_ERROR
    with _backend_slot:
        if _backend_fn is None:
            if IS_YOLO:
                try:
                    import skate_yolo
                    BACKEND_NAME = skate_yolo.BACKEND_NAME
                    _backend_fn = skate_yolo.analyze
                except Exception as e:
                    BACKEND_ERROR = f"{type(e).__name__}: {e}"
                    if not is_frozen():
                        IS_YOLO = False
                        BACKEND_NAME = "MediaPipe"
                    print(f"YOLO backend could not be loaded ({BACKEND_ERROR}); "
                          + ("no analysis is possible right now." if is_frozen()
                             else "the app continues with MediaPipe."),
                          file=sys.stderr)
            if _backend_fn is None:
                if is_frozen():
                    _backend_fn = _backend_broken      # MediaPipe isn't in this package
                else:
                    from skate_analysis import analyze as mp_analyze
                    _backend_fn = mp_analyze
        return _backend_fn


def analyze_backend(*args, **kwargs):
    """The GUI's analysis entry point; loads the backend on first use."""
    return _load_backend()(*args, **kwargs)


def _warm_backend_up():
    """Loads the backend in the background ahead of time, right after the window is shown.

    A daemon thread, because this is pure lookahead work: if the user starts an analysis
    right away, its import just blocks on the same lock until this one is done.
    """
    if IS_YOLO:
        threading.Thread(target=_load_backend, name="backend-warmup", daemon=True).start()

_MODEL_DIR = app_dir()          # naast de scripts, of naast de exe
DEFAULT_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
HEAVY_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_heavy.task")


# Room that the window frame (title bar + borders) takes up outside the content.
# `resize()` sets the content size, so without this margin a window at screen height
# sticks out below, behind the taskbar. Taken generously; it's a lower bound, not
# precision.
WINDOW_MARGIN = QMargins(8, 40, 8, 8)


def set_window_size(window, wanted_width, wanted_height, maximize=False):
    """Fits the window size to the available screen and centers the window.

    A fixed pixel size (1400x820 for the main window) falls outside the visible area on
    a smaller laptop screen. If the wanted size doesn't fit, the main window opens
    **maximized** instead (`maximize=True`): that fills the height exactly and saves the
    user from manually fixing it up at every start. Dialogs are only clamped and centered.

    Note: `resize()` can't override the layout — if the content's `minimumSizeHint` is
    wider than the screen, the window ends up too big anyway. That's why the wide
    control bars in `VideoPlayer` wrap with a `WrapBar`; see there.
    """
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        window.resize(wanted_width, wanted_height)
        return
    available = screen.availableGeometry()
    if maximize and (wanted_width > available.width()
                     or wanted_height > available.height()):
        # setWindowState instead of showMaximized(): the window must not jump into view
        # here yet — the caller decides when it's shown. Actually maximize instead of
        # resizing to the screen size, because `resize()` sets the *content*: the title
        # bar sits on top of that and would push the status bar under the taskbar.
        window.resize(available.size().shrunkBy(WINDOW_MARGIN))
        window.setWindowState(window.windowState() | Qt.WindowMaximized)
        return
    # Leave room for the window frame: `resize()` is about the content, the title bar
    # sits outside that — without a margin the bottom edge falls behind the taskbar.
    width = min(wanted_width, available.width() - WINDOW_MARGIN.left()
                - WINDOW_MARGIN.right())
    height = min(wanted_height, available.height() - WINDOW_MARGIN.top()
                 - WINDOW_MARGIN.bottom())
    window.resize(width, height)
    # `move()` sets the corner of the *frame* (title bar included), so the margin needs
    # to be accounted for here too. If only the content were centered, the title bar
    # would push the window down ~30 px and, with a clamped height, the bottom 5 px
    # would fall behind the taskbar — measured on 12-9-2026 on the trim window (+5 px)
    # and `CalibrationPicker` (+3 px).
    x = available.x() + max(0, available.width() - width - WINDOW_MARGIN.left()
                            - WINDOW_MARGIN.right()) // 2
    y = available.y() + max(0, available.height() - height - WINDOW_MARGIN.top()
                            - WINDOW_MARGIN.bottom()) // 2
    window.move(x, y)


def show_dialog(dlg):
    """Runs a modal dialog and then cleans it up. Returns the exec() code.

    The cleanup isn't tidiness, it's a crash fix (TODO_CRASH.md). A `QDialog` with a
    parent simply keeps existing after `exec()` — as a **hidden top-level window**,
    including the native Windows window behind it and the `QScreen` reference inside
    that. If Windows rebuilds the screen list (a snip overlay via Win+Shift+S, a display
    added, a DPI switch), Qt walks all those windows (`QWindowsWindow::checkForScreenChanged`)
    and falls over on a `QScreen` that no longer exists: `0xc0000005` in
    `QScreen::geometry()` / `QScreen::virtualSiblings()`, right in the middle of
    `app.exec()` and thus outside the reach of any `try`. One trim-then-batch round of
    seven clips left sixteen of those windows behind this way (target and horizon picker
    per clip, plus the trim window and the batch dialog), some of them with their own
    `VideoPlayer` and `VideoCapture` inside.

    `deleteLater()` and not `WA_DeleteOnClose`, because the caller reads the outcome
    (`dlg.doel_punt`, `dlg.fragmenten`, ...) only *after* `exec()`. Verified: the dialog
    stays alive until we're back in the main event loop — straight through a
    `QProgressDialog` (which does `processEvents`) and through a nested dialog — so every
    call site can safely read it, while it's cleaned up well before the analysis starts.
    """
    try:
        return dlg.exec()
    finally:
        dlg.deleteLater()


class FlowLayout(QLayout):
    """Layout that lines its items up on a row and **wraps** when the width doesn't fit.

    Needed because `VideoPlayer`'s control bars (layer toggles + zoom controls, ~774 px
    together) demand a minimum width of 774 px as a `QHBoxLayout`. Two players side by
    side on the compare page made that 1607 px — wider than a 1280 px laptop screen, and
    a `QMainWindow` can't be smaller than its `minimumSizeHint`, so `resize()` was simply
    ignored. Wrapping, the minimum width is that of the *widest single item* (~138 px)
    and the window fits on every screen; on a wide screen it stays one row and looks
    exactly like it did before.
    """

    def __init__(self, parent=None, marge=0, tussenruimte=6, min_breedte=0):
        super().__init__(parent)
        self._items = []
        self._tussenruimte = tussenruimte
        self._min_breedte = min_breedte
        self.setContentsMargins(marge, marge, marge, marge)

    # ── QLayout duties ────────────────────────────────────────────────────
    def addItem(self, item):
        self._items.append(item)

    def addStretch(self, _factor=1):
        """No-op: a wrapping bar left-aligns, a stretch item has no meaning here.
        Exists so callers coming from a QHBoxLayout don't have to change."""

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    # ── Height follows from the width ────────────────────────────────────
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
        # The width of the widest item — below that not a single row fits any more.
        maat = QSize()
        for item in self._items:
            maat = maat.expandedTo(item.minimumSize())
        marges = self.contentsMargins()
        maat = maat + QSize(marges.left() + marges.right(), marges.top() + marges.bottom())
        # This width isn't just a lower bound: Qt asks for the minimum height of a
        # height-follows-width item by calling `heightForWidth()` here. At the width of
        # a single item the bar wraps into eleven rows and the window minimum grows by
        # ~280 px in height. Hence a realistic lower bound (see WrapBar).
        return maat.expandedTo(QSize(self._min_breedte, 0))

    def _leg_uit(self, rect, alleen_meten):
        """Places the items row by row; returns the total height needed.

        Two passes: first the row layout (and thus the height of each row), then the
        placing. That's needed to **center vertically** — a 16 px label shouldn't hang
        at the top of a row with 24 px checkboxes.
        """
        marges = self.contentsMargins()
        vak = rect.adjusted(marges.left(), marges.top(), -marges.right(), -marges.bottom())

        regels = []                       # [(items, row_height)]
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


class WrapBar(QWidget):
    """Carrier widget for a `FlowLayout`, so a QVBoxLayout can take the wrapping bar in
    as a plain item (and passes the height-from-width along properly)."""

    # Lower bound for the width of the bar: below 280 px wrapping stops making sense,
    # and 280 stays below the 320/400 px the video image itself already demands, so this
    # limit costs no extra width at all.
    MIN_BREEDTE = 280

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, min_breedte=self.MIN_BREEDTE)
        beleid = self.sizePolicy()
        beleid.setHeightForWidth(True)
        # `Preferred`, not `Minimum`. With `Minimum` the sizeHint counts as a lower
        # bound, and here that's `heightForWidth(MIN_BREEDTE)` -- the bar wrapped at its
        # *narrowest* width: five rows (150 px) in the main window while it needs one or
        # two at its actual width. That phantom sat permanently in the window minimum
        # and pushed the main window above a 1280x720 screen (measured 12-9-2026). How
        # many rows are *actually* needed is decided by the owner at its own minimum
        # width, see `VideoPlayer.minimumSizeHint`; QBoxLayout, when placing the bar,
        # always gives it via heightForWidth the rows it needs at that moment.
        beleid.setVerticalPolicy(QSizePolicy.Preferred)
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


def minimum_with_wrapping(widget):
    """`minimumSizeHint` for a widget with wrapping bars (`WrapBar`) in its QVBoxLayout:
    the normal layout lower bound, but with the height the bars need at the widget's
    **minimum width** — it can't get any narrower than that anyway, so that's the honest
    minimum. Qt only does this itself for top-level windows; in a QSplitter (analysis and
    compare page) either the phantom of the narrowest wrapping counts instead (policy
    `Minimum`) or there's no wrapping accounted for at all (policy `Preferred`), and then
    the bar can vanish below the edge at the window's minimum width."""
    maat = QWidget.minimumSizeHint(widget)
    lay = widget.layout()
    if lay is not None and lay.hasHeightForWidth():
        hoogte = lay.totalMinimumHeightForWidth(maat.width())
        if hoogte > maat.height():
            maat.setHeight(hoogte)
    return maat


class ElideLabel(QLabel):
    """A QLabel that shortens long text with '…' instead of making the window wider.

    A plain QLabel without wordwrap demands its full text width as a minimum, and that
    carries straight through into the window minimum: two skaters with a long name and
    title side by side on the compare page made the main window 1489 px wide on a
    1280 px screen, and a long Drive path under the library 908 px (measured 12-9-2026).
    The full text stays available as a tooltip.
    """

    def __init__(self, tekst="", modus=Qt.ElideRight, parent=None):
        super().__init__(parent)
        self._volledig = ""
        self._modus = modus
        beleid = self.sizePolicy()
        beleid.setHorizontalPolicy(QSizePolicy.Ignored)   # never demand width
        self.setSizePolicy(beleid)
        self.setText(tekst)

    def setText(self, tekst):
        self._volledig = tekst
        self.setToolTip(tekst)
        self._pas_aan()

    def tekst(self):
        return self._volledig

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._pas_aan()

    def _pas_aan(self):
        fm = self.fontMetrics()
        marge = self.contentsMargins()
        # `indent`/padding from a stylesheet aren't in contentsMargins; a few px of
        # slack keeps the last letter from getting cut off right at the edge.
        breedte = max(0, self.width() - marge.left() - marge.right() - 6)
        super().setText(fm.elidedText(self._volledig, self._modus, breedte))


# Everything below, from the target/horizon/calibration picker dialogs through
# `AnalysisInfoDialog` (ending at `class ForwardReader`), is Phase 8a session 2:
# identifiers/comments/docstrings only -- UI text (window titles, labels, tooltips,
# messages) is still Dutch. See TRANSLATION_PROGRESS.md.
BOX_MIN_DRAG_PX  = 5       # shorter drag in TargetPicker = click (point), longer = box
BOX_ZOOM_MAX     = 8.0     # zoom range of TargetPicker (mouse wheel)
BOX_MIN_HEIGHT_PX = 70     # = skate_yolo.BOX_MIN_HEIGHT_PX (that module is lazy-loaded here):
                           # below this height in video pixels there's nothing to measure,
                           # not even with the spyglass -- measured on `00000 16-14`, see
                           # CLAUDE.md. The picker says so before the analysis, the backend
                           # once more afterwards.


class TargetPicker(QDialog):
    """
    Shows the first frame and lets the user point out the skater to follow: with a
    **click** (a point, as always) or by dragging a **box** around them with the left
    button. The box is for a skater who's small in frame: the YOLO backend uses it to
    switch on the spyglass (`skate_yolo._Spyglass`), which follows them from the box
    wherever the detection pass doesn't see them yet (measured: only from about
    80-130 px tall). For that the box needs to be reasonably tight, and on a 900 px
    dialog such a skater is only about 45 px tall -- hence the **mouse wheel to zoom**
    around the cursor and **right-drag to pan**. Crop-and-magnify, the same recipe as
    `VideoPlayer._show_pixmap`: `_crop_norm` is the single source of truth for the
    conversion, and everything that gets saved is normalized to the frame, so the zoom
    level doesn't matter for the outcome.

    Outcome after `exec()`: `doel_punt` (normalized (x, y), or None = 'follow largest')
    and `doel_kader` (normalized (x0, y0, x1, y1), or None). With a box, `doel_punt` is
    its center point. A click accepts immediately (existing behavior); a box waits for
    "Follow this box" so it can be redrawn first.

    `doel_punt`/`doel_kader` keep their Dutch names deliberately -- they flow straight
    into `instellingen_json`'s still-Dutch `doel_punt`/`doel_kader` keys elsewhere in
    this file (deferred to Phase 8d, see TRANSLATION_PROGRESS.md); renaming just the
    attribute here would split one logical key into two spellings.
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kies de schaatser om te volgen")
        self.doel_punt = None
        self.doel_kader = None
        self._frame = frame_bgr
        self._scaled_size = None
        self._zoom = 1.0
        self._pan = (0.5, 0.5)                    # center of the crop (normalized)
        self._crop_norm = (0.0, 0.0, 1.0, 1.0)    # (x0n, y0n, widthn, heightn) of the crop
        self._drag_start = None                   # left button: QPointF of the press
        self._drag_norm = None                    # ... and that point normalized
        self._box = None                          # drawn box (normalized xyxy)
        self._pan_start = None                    # right-drag: (QPointF, pan at the press)

        v = QVBoxLayout(self)
        uitleg = QLabel("Klik op de schaatser die je wilt volgen. Staat hij klein in "
                        "beeld, sleep dan een kader om hem heen: dan wordt hij ook "
                        "gevolgd waar de detectie hem nog niet ziet.\n"
                        "Muiswiel = inzoomen rond de cursor, rechts-slepen = beeld "
                        "verschuiven.")
        # Wrap it, otherwise the longest line claims the dialog width (855 px, and 1240 px
        # at a larger system font size — wider than a projector or a 1366 laptop at 125%).
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._press
        self.label.mouseMoveEvent = self._move
        self.label.mouseReleaseEvent = self._release
        self.label.wheelEvent = self._wheel
        # Right-drag pans; without this a context menu pops up on every pan.
        self.label.setContextMenuPolicy(Qt.PreventContextMenu)
        v.addWidget(self.label, 1)

        knoppen = QHBoxLayout()
        self.lbl_zoom = QLabel("Zoom 1,0×")
        knoppen.addWidget(self.lbl_zoom)
        knoppen.addStretch(1)
        self.btn_box = QPushButton("Volg dit kader")
        self.btn_box.setEnabled(False)
        self.btn_box.clicked.connect(self._confirm_box)
        knoppen.addWidget(self.btn_box)
        btn_skip = QPushButton("Volg grootste schaatser")
        btn_skip.clicked.connect(self.accept)     # doel_punt and doel_kader stay None
        knoppen.addWidget(btn_skip)
        v.addLayout(knoppen)

        set_window_size(self, 900, 640)

        h, w = frame_bgr.shape[:2]
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    # ── display ──────────────────────────────────────────────────────────────
    def _render(self):
        if not hasattr(self, "_pix"):
            return                    # resizeEvent before __init__ finishes
        pw, ph = self._pix.width(), self._pix.height()
        z = max(1.0, self._zoom)
        if z > 1.0:
            cw, ch = pw / z, ph / z
            x0 = min(max(self._pan[0] * pw - cw / 2, 0.0), pw - cw)   # crop within the frame
            y0 = min(max(self._pan[1] * ph - ch / 2, 0.0), ph - ch)
            ix0, iy0 = int(round(x0)), int(round(y0))
            icw, ich = min(int(round(cw)), pw - ix0), min(int(round(ch)), ph - iy0)
            bron = self._pix.copy(ix0, iy0, icw, ich)
            self._crop_norm = (ix0 / pw, iy0 / ph, icw / pw, ich / ph)
        else:
            bron = self._pix
            self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        scaled = bron.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        if self._box is not None:
            x0, y0 = self._norm_to_pixmap(self._box[0], self._box[1])
            x1, y1 = self._norm_to_pixmap(self._box[2], self._box[3])
            painter = QPainter(scaled)
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            painter.drawRect(QRect(QPoint(int(x0), int(y0)), QPoint(int(x1), int(y1))))
            painter.end()
        self.label.setPixmap(scaled)
        self.lbl_zoom.setText(f"Zoom {z:.1f}×".replace(".", ","))

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _widget_to_norm(self, pos):
        """Mouse position on the label → normalized (x, y) in the frame; can lie outside
        [0, 1] (caller clamps or rejects)."""
        if self._scaled_size is None:
            return None
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        if sw <= 0 or sh <= 0:
            return None
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        fx, fy = (pos.x() - offx) / sw, (pos.y() - offy) / sh
        x0n, y0n, wn, hn = self._crop_norm
        return (x0n + fx * wn, y0n + fy * hn)

    def _norm_to_pixmap(self, nx, ny):
        """Normalized (x, y) → position on the scaled pixmap (without the label offset)."""
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        x0n, y0n, wn, hn = self._crop_norm
        return ((nx - x0n) / wn * sw, (ny - y0n) / hn * sh)

    # ── zoom and pan ─────────────────────────────────────────────────────────
    def _wheel(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        nieuw = min(BOX_ZOOM_MAX, max(1.0, self._zoom * factor))
        if nieuw == self._zoom:
            return
        onder = self._widget_to_norm(event.position())
        if nieuw <= 1.0 or onder is None:
            self._pan = (0.5, 0.5)
        else:
            # Zoom around the cursor: the point under the cursor stays in the same spot
            # in the window, so the cursor's fraction within the crop stays the same and
            # the new center point follows from that.
            sw, sh = self._scaled_size.width(), self._scaled_size.height()
            fx = (event.position().x() - (self.label.width() - sw) / 2) / sw
            fy = (event.position().y() - (self.label.height() - sh) / 2) / sh
            cw, ch = 1.0 / nieuw, 1.0 / nieuw
            self._pan = (onder[0] - fx * cw + cw / 2, onder[1] - fy * ch + ch / 2)
        self._zoom = nieuw
        self._render()
        event.accept()

    # ── mouse ────────────────────────────────────────────────────────────────
    def _press(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_start = (event.position(), self._pan)
            return
        if event.button() != Qt.LeftButton:
            return
        norm = self._widget_to_norm(event.position())
        if norm is None:
            return
        self._drag_start = event.position()
        self._drag_norm = norm
        if self._box is not None:                  # a new drag clears the old box
            self._box = None
            self.btn_box.setEnabled(False)
            self._render()

    def _move(self, event):
        if self._pan_start is not None and self._scaled_size is not None:
            start, pan0 = self._pan_start
            sw, sh = self._scaled_size.width(), self._scaled_size.height()
            x0n, y0n, wn, hn = self._crop_norm
            dx = (event.position().x() - start.x()) / sw * wn
            dy = (event.position().y() - start.y()) / sh * hn
            self._pan = (min(max(pan0[0] - dx, wn / 2), 1 - wn / 2),
                         min(max(pan0[1] - dy, hn / 2), 1 - hn / 2))
            self._render()
            return
        if self._drag_start is None:
            return
        if (event.position() - self._drag_start).manhattanLength() < BOX_MIN_DRAG_PX:
            return
        norm = self._widget_to_norm(event.position())
        if norm is None:
            return
        # Dragging into the letterbox clamps to the frame edge (like drawing in the
        # viewing window): a box that ends just outside the frame is still intentional.
        x0, y0 = self._drag_norm
        x1, y1 = min(max(norm[0], 0.0), 1.0), min(max(norm[1], 0.0), 1.0)
        self._box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._render()

    def _release(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan_start = None
            return
        if event.button() != Qt.LeftButton or self._drag_start is None:
            return
        start, self._drag_start = self._drag_start, None
        if (event.position() - start).manhattanLength() < BOX_MIN_DRAG_PX:
            # A click: the point, as always — accept right away.
            x, y = self._drag_norm
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.doel_punt = (float(x), float(y))
                self.doel_kader = None
                self.accept()
            return
        if self._box is not None and (self._box[2] - self._box[0] > 0.005
                                      and self._box[3] - self._box[1] > 0.005):
            self.btn_box.setEnabled(True)
            self.btn_box.setFocus()
        else:
            self._box = None
            self._render()

    def _confirm_box(self):
        if self._box is None:
            return
        x0, y0, x1, y1 = (float(v) for v in self._box)
        hoogte_px = (y1 - y0) * self._pix.height()
        if hoogte_px < BOX_MIN_HEIGHT_PX:
            # Say so now, not after three minutes of computing: starting the fragment
            # later is the only remedy, and that means going back to the trim window.
            antwoord = QMessageBox.question(
                self, "Kader erg klein",
                f"Het kader is maar {hoogte_px:.0f} pixels hoog. Onder ongeveer "
                f"{BOX_MIN_HEIGHT_PX} pixels zijn er geen benen meer om te meten — ook "
                f"niet met het kijkglas — en levert de analyse vrijwel zeker niets op.\n\n"
                f"Tip: begin het fragment later, op het moment dat de schaatser groter in "
                f"beeld staat.\n\nToch doorgaan met dit kader?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if antwoord != QMessageBox.Yes:
                return
        self.doel_kader = (x0, y0, x1, y1)
        self.doel_punt = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.accept()


class HorizonPicker(QDialog):
    """
    Shows the first frame and lets the user click two points along the ice line (or
    another horizontal reference: boarding, ad board, track line). That gives the
    camera's tilt relative to the horizon. Returns `horizon_deg` (float, degrees) — 0.0
    if no tilt is set.
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stel de horizon / ijslijn in")
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self._frame = frame_bgr
        self._points = []            # original-pixel (x, y) of the reference line
        self._scaled_size = None
        self._scale = 1.0

        v = QVBoxLayout(self)
        uitleg = QLabel(
            "Klik twee punten langs het ijs (of de boarding/reclameband) om de\n"
            "camerakanteling te bepalen, of laat hem automatisch detecteren.\n"
            "Klik opnieuw om de lijn te hertekenen.")
        uitleg.setWordWrap(True)      # see TargetPicker: no dialog width from one text line
        v.addWidget(uitleg)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._click
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
        self.chk_per_frame.stateChanged.connect(self._toggle_per_frame)
        v.addWidget(self.chk_per_frame)

        knoppen = QHBoxLayout()
        self.btn_auto = QPushButton("Detecteer (dit frame)")
        self.btn_auto.clicked.connect(self._detect)
        knoppen.addWidget(self.btn_auto)
        btn_geen = QPushButton("Geen kanteling (0°)")
        btn_geen.clicked.connect(self._no_tilt)
        knoppen.addWidget(btn_geen)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Bevestig")
        self.btn_ok.clicked.connect(self._confirm)
        knoppen.addWidget(self.btn_ok)
        v.addLayout(knoppen)

        set_window_size(self, 900, 680)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        if self._points:
            painter = QPainter(scaled)
            pen = QPen(QColor(60, 200, 255), 3)
            painter.setPen(pen)
            pts = [(int(x * self._scale), int(y * self._scale)) for x, y in self._points]
            for px, py in pts:
                painter.drawEllipse(px - 4, py - 4, 8, 8)
            if len(pts) == 2:
                painter.drawLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _click(self, event):
        if self._scaled_size is None or self.chk_per_frame.isChecked():
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if len(self._points) >= 2:           # third click → start a new line
            self._points = []
        self._points.append((ox, oy))
        if len(self._points) == 2:
            self.horizon_deg = horizon_angle_from_line(self._points[0], self._points[1])
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: klik het tweede punt …")
        self._render()

    def _detect(self):
        degrees = detect_ice_line(self._frame)
        if degrees is None:
            QMessageBox.information(
                self, "Geen ijslijn gevonden",
                "Kon geen betrouwbare horizontale lijn detecteren. Teken de lijn "
                "handmatig, of kies 'Geen kanteling'.")
            return
        # Synthesize a display line straight across the frame at the found angle.
        w, h = self._orig_w, self._orig_h
        cx, cy = w / 2.0, h / 2.0
        slope = np.tan(np.radians(degrees))            # y drops to the right at a positive angle
        self._points = [(0.0, cy + slope * cx), (float(w), cy - slope * (w - cx))]
        self.horizon_deg = degrees
        self.lbl_hoek.setText(f"Kanteling: {degrees:+.2f}°  (automatisch — controleer de lijn)")
        self._render()

    def _toggle_per_frame(self, _state):
        """With per-frame auto, the manual/constant line doesn't apply."""
        on = self.chk_per_frame.isChecked()
        self.label.setEnabled(not on)
        self.btn_auto.setEnabled(not on)
        if on:
            self.lbl_hoek.setText("Kanteling: automatisch per frame — "
                                  "wordt tijdens de analyse bepaald.")
        elif len(self._points) == 2:
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: 0.00°  (nog geen lijn getekend)")

    def _confirm(self):
        self.auto_per_frame = self.chk_per_frame.isChecked()
        self.accept()

    def _no_tilt(self):
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self.accept()


class CalibrationPicker(QDialog):
    """
    Perspective calibration via track lines (phase 7, fixed camera). The user traces
    lines on the first frame (each line = two clicks): track lines that in reality run
    parallel in the direction of travel, and cross lines perpendicular to them. The
    dialog calibrates live as you go and draws the found true horizon; the Confirm
    button only becomes available once the calibration succeeds. Result in
    `self.perspectief` (PerspectiveConfig).

    Minimum needed: 2 track lines + 2 cross lines, or 3 track lines + 1 cross line, or
    2 track lines + 1 cross line + a given focal length (a frontal camera can only work
    with a given focal length).
    """
    COLOR_TRACK = QColor(60, 200, 255)     # cyan
    COLOR_CROSS = QColor(255, 170, 40)     # orange
    COLOR_HORIZON = QColor(240, 240, 240)

    def __init__(self, frame_bgr, parent=None, calibration_input=None, config=None):
        """`calibration_input`/`config`: a previously made calibration to start from
        (reusing the same camera pose). The lines are then already drawn and can still
        be corrected — reusing and adjusting is one and the same action."""
        super().__init__(parent)
        self.setWindowTitle("Perspectiefkalibratie: trek de baanlijnen na")
        self.perspectief = None
        self._frame = frame_bgr
        self._track_lines = []           # [((x,y),(x,y))] in original pixels
        self._cross_lines = []
        self._click_point = None         # first point of a line being drawn
        self._calibration = None
        self._calibration_input = None   # CalibrationInput of the current lines
        self._scaled_size = None
        self._scale = 1.0

        hoofd = QHBoxLayout(self)

        links = QVBoxLayout()
        uitleg = QLabel(
            "Trek elke lijn met twee klikken. Baanlijnen: evenwijdig in de rijrichting "
            "(volgorde maakt niet uit).\nDwarslijnen: haaks erop (start-/finishlijn, "
            "bochtmarkering). Trek zo lang mogelijke lijnen — dat is nauwkeuriger.")
        uitleg.setWordWrap(True)      # see TargetPicker: no dialog width from one text line
        links.addWidget(uitleg)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 400)
        self.label.mousePressEvent = self._click
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
        btn_wis_laatste.clicked.connect(self._undo_last)
        btn_wis_alles = QPushButton("Alles wissen")
        btn_wis_alles.clicked.connect(self._clear_all)
        knoppen_lijn.addWidget(btn_wis_laatste)
        knoppen_lijn.addWidget(btn_wis_alles)
        rechts.addLayout(knoppen_lijn)

        # Angles-only is the default: the push angle is scale-free, so without a known
        # line distance it's exact (measured: 0.00° off, whether you enter 0.5 m or 50 m).
        # Meters are only needed for speed/stroke length — and the 'lower_leg' method
        # computes with a lower-leg length in real meters, so it can't work here (a made-up
        # distance gave 49° off there, silently). Hence the coupling below: angles-only ⇒
        # 'leg_plane', which uses no length at all.
        self.chk_alleen_hoeken = QCheckBox("Alleen hoeken (geen snelheid/slaglengte)")
        self.chk_alleen_hoeken.setChecked(True)
        self.chk_alleen_hoeken.setToolTip(
            "De afzethoek is schaalvrij: hij komt alléén uit de richtingen van de lijnen,\n"
            "niet uit hun afstand. Je hoeft dus geen enkele maat te weten of op te meten.\n"
            "\n"
            "Uitzetten alleen als je snelheid (m/s) en slaglengte (m) in de tabel wilt, óf\n"
            "als je met de reconstructiemethode 'onderbeenlengte' wilt werken — die rekent\n"
            "met een lengte in echte meters en heeft dus een echte lijnafstand nodig.")
        self.chk_alleen_hoeken.toggled.connect(self._scale_changed)
        rechts.addWidget(self.chk_alleen_hoeken)

        vorm = QFormLayout()
        self.spin_lijnafstand = QDoubleSpinBox()
        self.spin_lijnafstand.setRange(0.5, 30.0)
        self.spin_lijnafstand.setSingleStep(0.5)
        self.spin_lijnafstand.setValue(skate_perspective.DEFAULT_LINE_DISTANCE)
        self.spin_lijnafstand.setSuffix(" m")
        self.spin_lijnafstand.valueChanged.connect(self._recalibrate)
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
        self.spin_f.valueChanged.connect(self._recalibrate)
        vorm.addRow("Brandpuntsafstand (px):", self.spin_f)

        self.combo_methode = QComboBox()
        self.combo_methode.addItem("Onderbeenlengte (bol-snijding)", "lower_leg")
        self.combo_methode.addItem("Beenvlak (rijrichting)", "leg_plane")
        self.combo_methode.setToolTip(
            "Hoe de knie-diepte wordt gereconstrueerd. Beide zijn experimenteel te\n"
            "vergelijken; 'onderbeenlengte' heeft de lengte hieronder nodig.")
        self.combo_methode.currentIndexChanged.connect(
            lambda _: self._scale_changed(self.chk_alleen_hoeken.isChecked()))
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
        self.btn_ok.clicked.connect(self._confirm)
        knoppen.addWidget(self.btn_ok)
        rechts.addLayout(knoppen)

        paneel = QWidget()
        paneel.setLayout(rechts)
        paneel.setFixedWidth(340)
        hoofd.addWidget(paneel)

        set_window_size(self, 1150, 700)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)

        # Deliberately right at the end: `_scale_changed` moves the method combo, and
        # that signal flows through to `_recalibrate`, which needs `lbl_status`, `btn_ok`
        # and `_orig_w`. So everything must already exist.
        self._scale_changed(self.chk_alleen_hoeken.isChecked())

        if config is not None and calibration_input is None:
            calibration_input = config.invoer
        if calibration_input is not None:
            self._prefill(calibration_input, config)
        self._render()

    def _prefill(self, calibration_input, config=None):
        """Puts an existing calibration into the dialog. The image size must match: the
        lines are in pixels, so on a differently-sized video they'd silently land in the
        wrong place and produce a plausible but wrong calibration."""
        if not calibration_input.fits(self._orig_w, self._orig_h):
            QMessageBox.warning(
                self, "Kalibratie past niet",
                f"Die kalibratie is gemaakt op beeld van {calibration_input.image_w}×"
                f"{calibration_input.image_h} en deze video is {self._orig_w}×"
                f"{self._orig_h}. De lijnen staan in pixels, dus overnemen zou ze "
                f"verkeerd neerleggen. Trek ze opnieuw na.")
            return
        self._track_lines = list(calibration_input.track_lines)
        self._cross_lines = list(calibration_input.cross_lines)
        self.spin_lijnafstand.setValue(calibration_input.line_distance)
        self.spin_f.setValue(int(calibration_input.f_px or 0))
        # Scale flag first, method/length after: `_scale_changed` pins the method to
        # 'leg_plane' as soon as angles-only is on, and would overwrite a choice set
        # before it.
        self.chk_alleen_hoeken.setChecked(not calibration_input.scale_known)
        if config is not None:
            # `config.methode` may still hold an old on-disk value ('onderbeen'/
            # 'beenvlak', from an analysis saved before Phase 3) -- same normalization
            # as the shim in `skate_perspective.reconstruct_angle()`.
            methode = {"onderbeen": "lower_leg", "beenvlak": "leg_plane"}.get(
                config.methode, config.methode)
            idx = self.combo_methode.findData(methode)
            if idx >= 0:
                self.combo_methode.setCurrentIndex(idx)
            if config.onderbeen_l:
                self.spin_onderbeen.setValue(config.onderbeen_l * 100.0)
        self._recalibrate()

    # ── drawing ──────────────────────────────────────────────────────────
    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        painter = QPainter(scaled)
        s = self._scale
        for lijnen, kleur, prefix in ((self._track_lines, self.COLOR_TRACK, "R"),
                                      (self._cross_lines, self.COLOR_CROSS, "D")):
            painter.setPen(QPen(kleur, 3))
            for i, (p1, p2) in enumerate(lijnen, start=1):
                x1, y1 = p1[0] * s, p1[1] * s
                x2, y2 = p2[0] * s, p2[1] * s
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.drawText(int((x1 + x2) / 2) + 6, int((y1 + y2) / 2) - 6,
                                 f"{prefix}{i}")
        if self._click_point is not None:
            kleur = self.COLOR_TRACK if self.radio_rij.isChecked() else self.COLOR_CROSS
            painter.setPen(QPen(kleur, 3))
            px, py = self._click_point[0] * s, self._click_point[1] * s
            painter.drawEllipse(int(px) - 4, int(py) - 4, 8, 8)
        if self._calibration is not None:
            # true horizon (vanishing line of the ice plane) as a visual check
            a, b, c = self._calibration.horizon_line
            if abs(b) > 1e-9:
                y0 = -(c + a * 0.0) / b * s
                y1 = -(c + a * self._orig_w) / b * s
                painter.setPen(QPen(self.COLOR_HORIZON, 1, Qt.DashLine))
                painter.drawLine(0, int(y0), int(self._orig_w * s), int(y1))
        painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    # ── interaction ──────────────────────────────────────────────────────
    def _click(self, event):
        if self._scaled_size is None:
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if self._click_point is None:
            self._click_point = (ox, oy)
        else:
            lijn = (self._click_point, (ox, oy))
            self._click_point = None
            if np.hypot(lijn[1][0] - lijn[0][0], lijn[1][1] - lijn[0][1]) < 10:
                self.lbl_status.setText("Lijn te kort — klik twee punten verder uit elkaar.")
            elif self.radio_rij.isChecked():
                self._track_lines.append(lijn)
            else:
                self._cross_lines.append(lijn)
            self._recalibrate()
        self._render()

    def _undo_last(self):
        if self._click_point is not None:
            self._click_point = None
        elif self._cross_lines and (self.radio_dwars.isChecked() or not self._track_lines):
            self._cross_lines.pop()
        elif self._track_lines:
            self._track_lines.pop()
        self._recalibrate()
        self._render()

    def _clear_all(self):
        self._track_lines = []
        self._cross_lines = []
        self._click_point = None
        self._recalibrate()
        self._render()

    @staticmethod
    def _line_hint(n_track):
        """Extra explanation for a failed calibration with 3+ track lines.

        With three or more track lines the calibration builds the vanishing line using
        the cross-ratio, and that uses their mutual distances — the dialog assumes
        they're **evenly** spaced. If they're not (a blue track line, ice edge, and
        boarding foot rarely sit at equal distances), the system can't be reconciled
        with a single camera and an 'f² ≤ 0' message follows that doesn't name that
        cause. Measured: unevenly spaced + told it's even = refused; with the correct
        mutual distances = 0.00° off. So it never fails silently."""
        if n_track < 3:
            return ""
        return ("\n\nTip: met 3+ baanlijnen wordt aangenomen dat ze GELIJKMATIG verdeeld "
                "zijn. Zijn ze dat niet, gebruik dan precies 2 baanlijnen + 2 "
                "dwarslijnen — dan doet hun onderlinge afstand niet meer mee.")

    # ── scale on/off ───────────────────────────────────────────────────────
    def _scale_changed(self, alleen_hoeken):
        """Couples 'angles only' to the fields that assume a real scale.

        Without a known line distance, the world coordinates are determined up to an
        arbitrary factor. For `leg_plane` that doesn't matter (the angle follows from
        directions), but `lower_leg` intersects with a sphere of a length in real
        meters — that factor feeds straight into the angle there. That's why the method
        gets pinned to `leg_plane` then, instead of letting the user pick a silent
        source of error."""
        for w in (self.spin_lijnafstand, self.lbl_lijnafstand):
            w.setEnabled(not alleen_hoeken)
        if alleen_hoeken:
            idx = self.combo_methode.findData("leg_plane")
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
        # The length fields only belong with 'lower_leg'.
        lengte_nodig = (not alleen_hoeken
                        and self.combo_methode.currentData() == "lower_leg")
        for w in (self.spin_lengte, self.spin_onderbeen, self.lbl_onderbeen):
            w.setEnabled(lengte_nodig)
        if self.lbl_lengte is not None:
            self.lbl_lengte.setEnabled(lengte_nodig)
        self._recalibrate()

    # ── calibration ──────────────────────────────────────────────────────
    @staticmethod
    def _sort_track_lines(lijnen):
        """Sort the track lines spatially (adjacent), so the even offsets are correct
        regardless of drawing order: project the line midpoints onto the direction
        perpendicular to the average line direction."""
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

    def _recalibrate(self):
        self._calibration = None
        self._calibration_input = None
        n_rij, n_dwars = len(self._track_lines), len(self._cross_lines)
        if n_rij < 2 or n_dwars < 1:
            self.lbl_status.setText(
                f"Getekend: {n_rij} baanlijn(en), {n_dwars} dwarslijn(en).\n"
                f"Nodig: minstens 2 baanlijnen + 1 dwarslijn "
                f"(2+1 alleen met opgegeven brandpuntsafstand; anders 3+1 of 2+2).")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        # Calibrate via the input (not directly): that way what's shown live here
        # travels exactly the same path as a later reopened analysis.
        alleen_hoeken = self.chk_alleen_hoeken.isChecked()
        calibration_input = skate_perspective.CalibrationInput(
            track_lines=self._sort_track_lines(self._track_lines),
            cross_lines=list(self._cross_lines),
            image_w=self._orig_w, image_h=self._orig_h,
            line_distance=self.spin_lijnafstand.value(),
            scale_known=not alleen_hoeken,
            f_px=self.spin_f.value() or None)
        try:
            self._calibration = calibration_input.calibrate()
            self._calibration_input = calibration_input
        except ValueError as e:
            self.lbl_status.setText(f"Kalibratie lukt nog niet: {e}{self._line_hint(n_rij)}")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        kal = self._calibration
        # Without a known scale, the camera height is in arbitrary units; showing that
        # in meters would suggest a precision that isn't there.
        hoogte = (f"camerahoogte {kal.camera_height:.1f} m, " if kal.scale_known
                  else "")
        # With exactly 2 track lines + 2 cross lines the system is exactly determined:
        # the residual is then 0.00 px by construction and says nothing about quality —
        # showing it would read as "perfectly calibrated". A third cross line turns V2
        # into a least-squares fit and makes the residual actually informative.
        overbepaald = len(self._cross_lines) >= 3 or len(self._track_lines) >= 3
        residu = (f", residu {kal.residual_px:.1f} px" if overbepaald else "")
        tekst = (f"Kalibratie OK — f = {kal.f:.0f} px"
                 f"{' (geschat)' if kal.f_estimated else ''}, {hoogte}"
                 f"horizon {kal.horizon_deg:+.2f}°{residu}.")
        if not overbepaald:
            tekst += ("\nPrecies genoeg lijnen: er is géén controle mogelijk. Teken een "
                      "derde dwarslijn om te zien of de kalibratie klopt.")
        if not kal.scale_known:
            tekst += ("\nAlleen hoeken: die zijn schaalvrij en dus exact; snelheid en "
                      "slaglengte blijven leeg.")
        if kal.warnings:
            tekst += "\n⚠ " + "\n⚠ ".join(kal.warnings)
        self.lbl_status.setText(tekst)
        self.btn_ok.setEnabled(True)
        self._render()

    def _confirm(self):
        if self._calibration is None:
            return
        method = self.combo_methode.currentData()
        # 'leg_plane' uses no length; passing one anyway would suggest in storage and
        # the Info dialog that it affects the measurement.
        if method == "lower_leg":
            lower_leg_l = (self.spin_onderbeen.value() / 100.0
                           if self.spin_onderbeen.value() > 0
                           else skate_perspective.lower_leg_from_body_height(
                               self.spin_lengte.value()))
        else:
            lower_leg_l = None
        self.perspectief = PerspectiveConfig(
            calibration=self._calibration,
            method=method,
            lower_leg_l=lower_leg_l,
            calibration_input=self._calibration_input)
        self.accept()


class AnalysisAborted(Exception):
    """Cooperative abort of a running (batch) analysis: `abort()` sets a flag and the
    progress callback -- which every pass calls per frame -- raises this exception. That
    way an analysis stops within one frame instead of the thread being destroyed while
    still running when the app closes (Qt: 'Destroyed while thread is still running').
    Saving itself is never aborted halfway -- a half video copy in the media folder is
    worse than waiting a moment."""


class AnalysisWorker(QThread):
    """Runs the analysis in the background so the GUI doesn't block, then automatically
    saves the result to the library (phase 1). Saving deliberately happens in this
    thread too: the video copy to the media folder can take a while."""
    progress = Signal(int, int)
    status = Signal(str)                     # text for the progress dialog (busy phase)
    done = Signal(object, object, object, object)    # info, resultaten, events, analyse_id
    error = Signal(str)                      # the analysis itself failed
    save_error = Signal(str)                 # only the save failed (the analysis exists)
    warning = Signal(str)                    # silent fallback in the analysis (e.g. a click hit nobody)

    def __init__(self, input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                 doel_punt=None, horizon_deg=0.0, auto_horizon=False, smooth_landmarks=True,
                 perspectief=None, bieb=None, schaatser_id=None, titel=None,
                 instellingen=None, backend=None, aangemaakt_door="", bocht=True,
                 deinterlacen=False, doel_kader=None):
        super().__init__()
        self.input_pad = input_pad
        self.model_pad = model_pad
        self.smooth_n = smooth_n
        self.threshold = threshold
        self.force_fps = force_fps
        self.doel_punt = doel_punt
        self.doel_kader = doel_kader
        self.horizon_deg = horizon_deg
        self.auto_horizon = auto_horizon
        self.smooth_landmarks = smooth_landmarks
        self.bocht = bocht
        self.deinterlacen = deinterlacen
        self.perspectief = perspectief
        self.bieb = bieb
        self.schaatser_id = schaatser_id
        self.titel = titel
        self.instellingen = instellingen
        self.backend = backend
        self.aangemaakt_door = aangemaakt_door
        self.cancelled = False

    def abort(self):
        """Asks the analysis to stop (app shutting down). The thread ends at the next
        frame callback, without a signal and without saving."""
        self.cancelled = True

    def run(self):
        try:
            def show_progress(frame_nr, totaal):
                if self.cancelled:
                    raise AnalysisAborted()
                self.progress.emit(frame_nr, totaal)

            info, resultaten = analyze_backend(
                self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                self.force_fps, doel_punt=self.doel_punt, progress_callback=show_progress,
                horizon_deg=self.horizon_deg, auto_horizon=self.auto_horizon,
                smooth_landmarks=self.smooth_landmarks, perspectief=self.perspectief,
                waarschuwing_callback=self.warning.emit, bocht=self.bocht,
                deinterlacen=self.deinterlacen, doel_kader=self.doel_kader,
            )
            events = segment_pushes(resultaten)
        except AnalysisAborted:
            return                    # shutting down: report nothing, save nothing
        except Exception as e:
            self.error.emit(str(e))
            return
        if self.cancelled:
            return                    # don't start a long video copy anymore

        # Save to the library; if this fails, the (long) analysis isn't lost -- the
        # results are still shown, just not kept.
        analyse_id = None
        if self.bieb is not None and self.schaatser_id is not None:
            self.status.emit("Opslaan in bibliotheek...")
            try:
                analyse_id = skate_db.save_analysis(
                    self.bieb, self.schaatser_id, self.titel, self.input_pad,
                    info, resultaten, events,
                    backend=self.backend, instellingen=self.instellingen,
                    aangemaakt_door=self.aangemaakt_door)
            except Exception as e:
                self.save_error.emit(str(e))
        self.done.emit(info, resultaten, events, analyse_id)


class BatchWorker(QThread):
    """Runs a series of analyses back-to-back in the background and automatically saves
    each video to the library. One bad clip doesn't stop the batch -- it's reported as
    failed and the rest keeps going. 'Stop after this video' requests a clean stop via
    requestInterruption() that's handled between videos (the video in progress is
    finished and saved first)."""
    task_start = Signal(int, int, str)           # index (0-based), total, titel
    progress   = Signal(int, int)                # frame_nr, total of the current video
    status     = Signal(str)                     # busy text (video copy to the library)
    task_done  = Signal(int, object)             # index, analyse_id (or None)
    task_error = Signal(int, str)                # index, error message -- the batch continues
    all_done   = Signal(list, list, list)        # succeeded titles, [(titel, message)] failed,
                                                 # [(titel, message)] warnings

    def __init__(self, taken, bieb, backend, aangemaakt_door=""):
        super().__init__()
        self.taken = taken
        self.bieb = bieb
        self.backend = backend
        self.aangemaakt_door = aangemaakt_door
        self.cancelled = False

    def abort(self):
        """Hard stop (app shutting down): the video in progress is aborted too.
        Deliberately different from `requestInterruption()` ('Stop after this video'),
        which lets the current video finish and save cleanly."""
        self.cancelled = True

    def _progress(self, frame_nr, totaal):
        if self.cancelled:
            raise AnalysisAborted()
        self.progress.emit(frame_nr, totaal)

    def run(self):
        n = len(self.taken)
        geslaagd, fouten, waarschuwingen = [], [], []
        for i, taak in enumerate(self.taken):
            if self.cancelled or self.isInterruptionRequested():
                break                            # 'Stop after this video' -- skip the rest
            self.task_start.emit(i, n, taak["titel"])
            try:
                # Don't throw a warning (e.g. a click that hit nobody) into a modal box
                # per video -- a batch runs unattended by design; collect them and report
                # in the summary at the end, with the titel alongside.
                def _warn(tekst, titel=taak["titel"]):
                    waarschuwingen.append((titel, tekst))

                info, resultaten = analyze_backend(
                    taak["input_pad"], taak["model_pad"], taak["smooth_n"], taak["threshold"],
                    doel_punt=taak["doel_punt"],
                    progress_callback=self._progress,
                    horizon_deg=taak["horizon_deg"], auto_horizon=taak["auto_horizon"],
                    smooth_landmarks=taak["smooth_landmarks"],
                    perspectief=taak.get("perspectief"),
                    waarschuwing_callback=_warn, bocht=taak.get("bocht", True),
                    deinterlacen=taak.get("deinterlacen", False),
                    doel_kader=taak.get("doel_kader"),
                )
                events = segment_pushes(resultaten)
                if resultaten and all(r.bocht for r in resultaten):
                    # Otherwise this clip would show up as "0 pushes" in the list with no
                    # one knowing why.
                    _warn("De schaatser staat nergens frontaal in beeld; de hele video "
                          "is als bocht aangemerkt en er is niets gemeten.")
                if self.cancelled:
                    break                        # don't start a long video copy anymore
                self.status.emit("Opslaan in bibliotheek...")
                analyse_id = skate_db.save_analysis(
                    self.bieb, taak["schaatser_id"], taak["titel"], taak["input_pad"],
                    info, resultaten, events,
                    backend=self.backend, instellingen=taak["instellingen"],
                    aangemaakt_door=self.aangemaakt_door,
                    bron_id=taak.get("bron_id"),
                    bron_start_frame=taak.get("bron_start_frame"),
                    bron_eind_frame=taak.get("bron_eind_frame"))
                geslaagd.append(taak["titel"])
                self.task_done.emit(i, analyse_id)
            except AnalysisAborted:
                break                            # shutting down: the rest of the row is dropped
            except Exception as e:
                # save_analysis cleans up its own half media folder; just record it here.
                fouten.append((taak["titel"], str(e)))
                self.task_error.emit(i, str(e))
        if not self.cancelled:
            self.all_done.emit(geslaagd, fouten, waarschuwingen)


class SkaterDialog(QDialog):
    """Create or edit a skater profile: name, birth year, notes."""

    def __init__(self, parent=None, naam="", geboortejaar=None, notities=""):
        super().__init__(parent)
        self.setWindowTitle("Schaatser")
        form = QFormLayout(self)

        self.veld_naam = QLineEdit(naam)
        form.addRow("Naam:", self.veld_naam)

        self.veld_jaar = QSpinBox()
        self.veld_jaar.setRange(0, 2100)
        self.veld_jaar.setSpecialValueText("—")   # 0 = not filled in
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


class NewAnalysisDialog(QDialog):
    """Collects everything for one new analysis: skater, video, titel and the analysis
    settings (moved here from the old start page, phase 1)."""

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
        knop_video.clicked.connect(self._choose_video)
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
        self.chk_heavy.setVisible(not IS_YOLO)   # only relevant for the MediaPipe backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Bocht overslaan (sneller)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(CORNER_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        self.chk_deint = QCheckBox("Interlacing wegfilteren (kamtanden)")
        self.chk_deint.setToolTip(DEINT_TOOLTIP)
        self.chk_deint.setEnabled(False)         # only usable once a video is chosen
        fv.addWidget(self.chk_deint)

        self.chk_perspectief = QCheckBox("Perspectiefcorrectie via baanlijnen (experimenteel)")
        self.chk_perspectief.setToolTip(PERSPECTIVE_TOOLTIP)
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
        self._ok.setEnabled(False)               # only enabled once a video is chosen

    def _choose_video(self):
        pad, _ = QFileDialog.getOpenFileName(
            self, "Kies video", "", VIDEO_FILTER)
        if not pad:
            return
        self.video_pad = pad
        self.lbl_video.setText(os.path.basename(pad))
        if not self.veld_titel.text().strip():
            self.veld_titel.setText(os.path.splitext(os.path.basename(pad))[0])
        self._ok.setEnabled(True)
        self._check_interlacing(pad)

    def _check_interlacing(self, pad):
        """Sets the comb-filter checkbox to what this video needs. Right here, not only
        during the analysis, so the user sees what's about to happen and can override it
        -- and so the choice ends up as an explicit value in the settings."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            interlaced = is_interlaced(pad)
        except Exception:
            interlaced = False                   # unable to measure isn't a reason to filter
        finally:
            QApplication.restoreOverrideCursor()
        self.chk_deint.setEnabled(True)
        self.chk_deint.setChecked(interlaced)
        self.lbl_video.setText(os.path.basename(pad)
                               + ("  —  interlaced" if interlaced else ""))

    @property
    def schaatser_id(self):
        return self.combo_schaatser.currentData()

    @property
    def deinterlacen(self):
        return self.chk_deint.isChecked()

    @property
    def titel(self):
        tekst = self.veld_titel.text().strip()
        if tekst:
            return tekst
        return os.path.splitext(os.path.basename(self.video_pad or "analyse"))[0]


class BatchAnalysisDialog(QDialog):
    """Collects a whole batch in one dialog: multiple videos at once, each with its own
    skater and titel, plus shared analysis settings. The target/horizon choice then
    happens per video in the collection loop (MainWindow._nieuwe_batch_analyse)."""

    def __init__(self, schaatsers, voorkeur_id=None, voorgevuld=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fragmenten analyseren" if voorgevuld else "Batch-analyse")
        self.resize(760, 500)
        self._schaatsers = schaatsers
        v = QVBoxLayout(self)

        # Choose videos + the default skater you apply to all rows at once.
        rij_top = QHBoxLayout()
        knop_videos = QPushButton("Video's kiezen...")
        knop_videos.clicked.connect(self._choose_videos)
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
        knop_toepassen.clicked.connect(self._apply_default)
        rij_top.addWidget(knop_toepassen)
        v.addLayout(rij_top)

        # Videos + a skater (combobox) and an editable titel per row.
        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["Video", "Schaatser", "Titel"])
        self.tabel.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tabel.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tabel.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self.tabel, stretch=1)

        knop_verwijder = QPushButton("Geselecteerde rij verwijderen")
        knop_verwijder.clicked.connect(self._remove_row)
        v.addWidget(knop_verwijder)

        # Gedeelde instellingen (dezelfde widgets/waarden als NewAnalysisDialog).
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
        self.chk_heavy.setVisible(not IS_YOLO)   # only relevant for the MediaPipe backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Bocht overslaan (sneller)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(CORNER_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        # Determined per clip, not once for the whole batch: a batch can hold clips from
        # different cameras, and the answer is cheap per file.
        self.chk_deint = QCheckBox("Interlacing automatisch wegfilteren (kamtanden)")
        self.chk_deint.setToolTip(DEINT_TOOLTIP)
        self.chk_deint.setChecked(True)
        fv.addWidget(self.chk_deint)

        self.chk_perspectief = QCheckBox("Perspectiefcorrectie via baanlijnen (experimenteel)")
        self.chk_perspectief.setToolTip(PERSPECTIVE_TOOLTIP_BATCH)
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
        self._ok.setEnabled(False)               # only enabled with at least one video

        # Phase 8: rows that are already fixed (fragments just cut from a recording).
        # The trainer only needs to fill in skater + titel -- the rest is exactly the
        # existing batch flow.
        for f in (voorgevuld or []):
            self._add_row(f["input_pad"], f.get("titel"), bron={
                "bron_id": f.get("bron_id"),
                "bron_start_frame": f.get("bron_start_frame"),
                "bron_eind_frame": f.get("bron_eind_frame")})

    def _add_row(self, pad, titel=None, bron=None):
        """One video row: pad behind the first cell, skater combo, editable titel.
        `bron` (phase 8) rides along so the analysis knows which piece of which
        recording this clip comes from."""
        r = self.tabel.rowCount()
        self.tabel.insertRow(r)
        item_pad = QTableWidgetItem(os.path.basename(pad))
        item_pad.setData(Qt.UserRole, pad)                 # full path behind the row
        item_pad.setData(Qt.UserRole + 1, bron)
        item_pad.setFlags(item_pad.flags() & ~Qt.ItemIsEditable)
        self.tabel.setItem(r, 0, item_pad)
        self.tabel.setCellWidget(r, 1, self._make_skater_combo())
        self.tabel.setItem(r, 2, QTableWidgetItem(
            titel or os.path.splitext(os.path.basename(pad))[0]))
        self._ok.setEnabled(True)

    def _make_skater_combo(self):
        """A per-row skater choice, preselected on the current default."""
        combo = QComboBox()
        for s in self._schaatsers:
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            combo.addItem(tekst, s["id"])
        idx = combo.findData(self.combo_standaard.currentData())
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    def _choose_videos(self):
        paden, _ = QFileDialog.getOpenFileNames(
            self, "Kies video's", "", VIDEO_FILTER)
        for pad in paden:
            self._add_row(pad)
        self._ok.setEnabled(self.tabel.rowCount() > 0)

    def _apply_default(self):
        sid = self.combo_standaard.currentData()
        for r in range(self.tabel.rowCount()):
            combo = self.tabel.cellWidget(r, 1)
            if combo is not None:
                idx = combo.findData(sid)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _remove_row(self):
        r = self.tabel.currentRow()
        if r >= 0:
            self.tabel.removeRow(r)
        self._ok.setEnabled(self.tabel.rowCount() > 0)

    @property
    def taken(self):
        """List of {input_pad, schaatser_id, titel} -- one per video row."""
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
            taak.update(item_pad.data(Qt.UserRole + 1) or {})   # bron_* (phase 8), or nothing
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


class AnalysisPicker(QDialog):
    """
    Picks one saved analysis: skater first, then one of their analyses. Used twice in a
    row to set up a comparison, and after that per side to switch analyses.
    """

    def __init__(self, bieb, titel="Kies analyse", voorkeur_schaatser_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titel)
        self.bieb = bieb

        form = QFormLayout(self)
        self.combo_schaatser = QComboBox()
        # Skip skaters with no analyses -- that way the analysis combo can never be empty.
        for s in skate_db.list_skaters(bieb):
            if not s["aantal_analyses"]:
                continue
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_schaatser.addItem(tekst, s["id"])
        if voorkeur_schaatser_id is not None:
            idx = self.combo_schaatser.findData(voorkeur_schaatser_id)
            if idx >= 0:
                self.combo_schaatser.setCurrentIndex(idx)
        self.combo_schaatser.currentIndexChanged.connect(self._fill_analyses)
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

        self._fill_analyses()

    def _fill_analyses(self, _idx=None):
        self.combo_analyse.clear()
        sid = self.combo_schaatser.currentData()
        if sid is not None:
            for a in skate_db.list_analyses(self.bieb, sid):
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
        # the combo text may carry the birth year too; for the heading we only want the name
        naam = next((s["naam"] for s in skate_db.list_skaters(self.bieb)
                     if s["id"] == sid), "")
        return naam


def _yes_no(waarde):
    return "ja" if waarde else "nee"


def _duration_text(a):
    """Video duration + number of pushes for the library list: "5.4s (4 pushes)".

    The duration says at a glance what kind of clip this is (a single stroke or a whole
    lap); the number of pushes stays alongside it in parentheses. Above one minute it
    becomes m:ss, since nobody reads "83.2s" as a minute and a half. Missing fps or frame
    count (an incompletely-written analysis) gives just a dash -- no division by zero."""
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


class AnalysisInfoDialog(QDialog):
    """
    Read-only overview of one saved analysis: which app version/backend it ran with,
    when and by whom, and with which settings.

    Why: the tracking logic changes regularly during development, so a strange
    measurement needs to be explainable ("this was still done with the old L/R fixer").
    Purely informational -- no input field. The values are selectable so a commit hash
    can be copied.
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

        # Smoothing off = the diagnostic mode (raw detections, CLI --no-smooth).
        smoothing = (f"{inst.get('smooth_n', '?')} frames"
                     if inst.get("smooth_landmarks", True) else "uit (ruwe detecties)")
        horizon = ("automatisch per frame" if inst.get("auto_horizon")
                   else f"vast {inst.get('horizon_deg', 0.0):.1f}°")
        # `heavy` chooses between pose_landmarker_heavy and _full and so only exists for
        # the MediaPipe backend; YOLO has one model and ignores the flag -- there the row
        # ("no") would only suggest a heavier model could have been chosen.
        heavy_rij = ([] if meta.get("backend") == "yolo" else
                     [("Heavy-model:", _yes_no(inst.get("heavy")),
                       "MediaPipe: pose_landmarker_heavy.task i.p.v. _full.task.")])

        # Origin (phase 8): only for a fragment cut from a recording. A loose clip has no
        # source, and then an empty row says nothing.
        herkomst_rij = []
        if meta.get("bron_id") and meta.get("bron_start_frame") is not None:
            fps = meta.get("fps") or 0
            plek = (f" ({_time_text(meta['bron_start_frame'], fps)}–"
                    f"{_time_text(meta['bron_eind_frame'], fps)})" if fps else "")
            herkomst_rij = [("Uit opname:",
                             f"{meta.get('bron_naam') or 'onbekend'}{plek}",
                             "Dit fragment is met het knipvenster uit een langere "
                             "trainingsopname geknipt.")]

        # How the target skater was pointed out. A box switches on the spyglass in the
        # YOLO backend (the skater is followed from the box wherever the detection
        # doesn't see them), so that's a measurement-relevant difference from a click.
        if inst.get("doel_kader"):
            doel = "kader getekend (kijkglas aan)"
        elif "doel_punt" not in inst:
            doel = "onbekend (van vóór deze functie)"
        elif inst.get("doel_punt"):
            doel = "aangeklikt"
        else:
            doel = "grootste beweger"

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
            ("Handmatig bewerkt:", _yes_no(meta.get("bewerkt")),
             "Zijn er met de skelet-editor punten verplaatst of skeletten geplaatst?"),
            ("Smoothing:", smoothing, None),
            ("Drempel:", f"{inst.get('threshold', '?')}", None),
        ] + heavy_rij + [
            ("Doelschaatser:", doel,
             "Klik = de detectiepass zoekt de schaatser op die plek. Kader = daarbovenop\n"
             "volgt het kijkglas hem vanaf het kader waar de detectie hem (nog) niet ziet,\n"
             "bv. omdat hij klein in beeld staat."),
            ("Bocht overslaan:", _yes_no(inst.get("bocht_overslaan")), None),
            ("Interlacing gefilterd:",
             _yes_no(inst.get("deinterlaced")) if "deinterlaced" in inst
             else "onbekend (van vóór deze functie)",
             "Camcorderbeeld (1080i) weeft twee momenten van 1/50 s uit elkaar in één\n"
             "frame. Stond dit aan, dan zijn die kamtanden vóór de detectie weggefilterd."),
            ("Horizon:", horizon, None),
            ("Perspectiefcorrectie:", _yes_no(inst.get("perspectief_gebruikt")), None),
        ] + _calibration_rows(inst)
        for label, waarde, tip in rijen:
            w = QLabel(str(waarde))
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tip:
                w.setToolTip(tip)
            form.addRow(label, w)

        knoppen = QDialogButtonBox(QDialogButtonBox.Close)
        knoppen.rejected.connect(self.reject)
        form.addRow(knoppen)


# Read-ahead during playback (see `ForwardReader`): how much memory the buffer may cost
# per player at most, and how many frames it may hold at most. A 1080p frame is 6.2 MB
# and a 4K frame 24.9 MB, so without a byte cap a fixed frame cap would cost hundreds of
# megabytes on 4K material -- and the compare page has two players. The goal is to
# absorb a load spike of a few frames, not to read in the whole video: precomputing
# everything would cost 216 GB and a 9-minute wait for a 23-minute recording.
READAHEAD_MAX_BYTES = 96 * 1024 * 1024
READAHEAD_MAX_FRAMES = 16


class ForwardReader(QThread):
    """Reads frames ahead on its own thread while playback is running.

    Showing a frame has two halves that need nothing from each other: **reading**
    (decoding + the comb filter if applicable, measured at 15.9 ms on 1080i camcorder
    footage) and **showing** (crop + QImage + scaling to a HiDPI screen, 12.0 ms). Back
    to back on the GUI thread that's 27.9 ms of the 40 ms available at 25 fps -- enough
    margin for the average, too little for a spike, and every spike costs a frame right
    away. Side by side, the floor is the slower half: 15.9 ms, i.e. ~63 fps worth of
    headroom.

    That this works at all isn't a given in Python: it only does because OpenCV releases
    the GIL during `read()` and during the filter steps, so real parallel work happens
    instead of turn-taking.

    **Whoever holds the capture reads.** A `cv2.VideoCapture` isn't thread-safe, so the
    player hands it off when it starts and gets it back when it stops -- there are never
    two readers. On handback the capture sits past the last frame it read, i.e. further
    than what has been shown; the frames not yet shown travel back with it (`remaining()`)
    so they don't need to be decoded again and the player doesn't need to rewind -- on the
    playback page that would reopen the video and spool from frame 0.
    """

    def __init__(self, cap, start_pos, count, carryover=(), parent=None):
        super().__init__(parent)
        self.cap = cap
        self.pos = start_pos          # index of the next frame to read
        self.at_end = False           # video exhausted; the queue may still hold frames
        self._q = queue.Queue(maxsize=max(2, count))
        for idx, frame in carryover:  # what was left over from the previous round
            try:
                self._q.put_nowait((idx, frame))
            except queue.Full:
                break

    def run(self):
        while not self.isInterruptionRequested():
            ok, frame = self.cap.read()
            if not ok:
                self.at_end = True
                return
            idx, self.pos = self.pos, self.pos + 1
            while not self.isInterruptionRequested():
                try:
                    self._q.put((idx, frame), timeout=0.05)
                    break
                except queue.Full:
                    pass              # buffer full: wait until the player takes one

    def take(self, wanted):
        """The ready frame closest to `wanted`, or `(None, None)`.

        Everything before `wanted` is discarded -- those frames have already passed
        according to the playback clock, and returning them would make the picture lag
        behind. If only something older is ready (the reader is behind), the newest of
        those is the best answer: it keeps the picture moving."""
        taken = (None, None)
        while True:
            try:
                idx, frame = self._q.get_nowait()
            except queue.Empty:
                return taken
            taken = (idx, frame)
            if idx >= wanted:
                return taken

    def remaining(self):
        """What's still in the buffer, by frame number -- for handing back to the player."""
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return sorted(out)


class VideoPlayer(QWidget):
    """
    Video panel with its own capture, playback timer, and zoom/pan state: image label +
    transport buttons + playback speed + scrub slider + layer toggles + zoom controls.

    Self-contained, so several can exist side by side (the analysis page has one, the
    compare page two). The player is the **sole** owner of `video_info`, `resultaten`,
    `huidige_idx`, and `video_pad`; MainWindow only looks at them through read-only
    properties, so a second, silently diverging copy can never appear.

    The owner hooks in with plain callables -- no signals, since a QMouseEvent doesn't
    survive a queued connection and there is exactly one owner per player:
        on_frame_shown(idx)     -- after drawing a frame (chart/table/status bar)
        overlay_drawer(pixmap)  -- right before setPixmap (the skeleton editor draws its
                                    handles; the trainer's annotations already sit under it)
        on_mouse_press/_move/_release(event)
                                  -- only if the player itself didn't already swallow the
                                    event as a pan drag
    """

    def __init__(self, min_size=(480, 320), show_speed=True, fast_seek=False,
                 show_overlay=True, show_drawing=False, parent=None):
        super().__init__(parent)

        # Without an analysis there's nothing to draw: the trim window (phase 8) feeds the
        # player empty FrameResult objects, and then the overlay would put "No pose
        # detected" on every single frame. `show_overlay=False` skips the drawing and hides
        # the layer checkboxes, which wouldn't do anything there anyway.
        self.show_overlay = show_overlay

        # Drawing on the image (see DRAW_TOOLTIP) exists only where you're purely looking:
        # the viewing window. Off by default, and then the controls aren't even created --
        # `FlowLayout` doesn't skip hidden items, so an invisible drawing block would cost
        # every other video window a gap and ~28 px of window minimum for something that
        # can't be used there.
        self.drawing_on = bool(show_drawing)

        # Comb filter for an interlaced source (see DEINT_TOOLTIP). When on, the player
        # shows exactly the pixels the measurement was taken from -- otherwise the skeleton
        # would sit on a different image than the one it was computed from.
        self.deinterlacen = False

        # `fast_seek` trades frame-exactness for usability on a long recording -- see
        # _read_frame_exact. Only turn it on where the image is a look, not a measurement
        # (the phase-8 trim window); the viewing page leaves it off.
        self.fast_seek = fast_seek

        # Display state (per player, so several can run at once)
        self.video_pad = None
        self.video_info = None
        self.resultaten = []
        self.huidige_idx = -1
        self.cap = None
        self._display_pos = 0        # frames already read by cap (sequential cursor)
        self._last_frame = None    # raw copy of the current frame (for layer toggles)
        self._display_scaled = None  # QSize of the shown (scaled) pixmap, for coordinate conversion

        # Zooming in on the skater. Two zoom values, deliberately kept apart:
        # `_zoom` is what the user set (slider/wheel, 1-ZOOM_MAX), `_zoom_eff` is what's
        # actually shown. Without automatic zoom they're equal.
        self._zoom = 1.0            # 1.0 = fitted (no crop); up to ZOOM_MAX
        self._zoom_eff = 1.0        # zoom applied to the current frame (up to ZOOM_AUTO_MAX)
        self._pan_cx = 0.5          # normalized center of the crop (full frame)
        self._pan_cy = 0.5
        self._zoom_follow = True      # auto-center on the skater (mirrors chk_follow)
        self._force_follow = False  # center once without a frame change (after a zoom action)
        self._zoom_auto = False     # let the program determine the zoom (mirrors chk_auto)
        self._box = None          # (center_x, center_y, radius) per frame, or None
        self._crop_norm = (0.0, 0.0, 1.0, 1.0)  # (x0n, y0n, widthn, heightn): the shown crop
        self._pan_drag = None      # last mouse position during a manual pan drag

        # Drawing on the image (see DRAW_TOOLTIP). The strokes are lists of
        # frame-normalized points and belong to the clip, not to a frame: they stay put
        # while the video keeps playing.
        self.draw_mode = DRAW_PAN
        self._drawing = []          # completed strokes: [[(nx, ny), ...], ...]
        self._stroke = None          # the stroke currently being dragged
        self._base_pixmap = None    # scaled image without the drawing (fast redraw)

        # Hooks for the owner (see the class docstring)
        self.on_frame_shown = None
        self.overlay_drawer = None
        self.on_mouse_press = None
        self.on_mouse_move = None
        self.on_mouse_release = None
        # Directly on the field: the setter below touches `combo_draw`, which doesn't
        # exist yet until after `_build_ui`.
        self._edit_mode = False    # drives the pan-vs-editor priority of the mouse
        self.follow_frozen = False  # during an editor drag: don't let the crop jump

        # Scrub coalescing (see _scrub_requested); must exist before `_build_ui`, since
        # that connects the slider to it.
        self._scrub_target = 0
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._scrub_tick)

        self._build_ui(min_size, show_speed, show_overlay)
        self.play_timer = QTimer(self)
        # PreciseTimer, and that's not fine-tuning here but the difference between smooth
        # and choppy. A regular Qt timer is a CoarseTimer and on Windows hangs off the
        # clock granularity of ~15.6 ms: measured, a 40 ms timer then fires once every
        # 46.5 ms. At 25 fps that's structurally 6.5 ms per frame too late, and since
        # `_play_tick` counts on the wall clock it pays for that in skipped frames -- 15%
        # of all frames, while there was plenty of compute time to spare (27.9 ms of work
        # out of the 40 ms). With PreciseTimer it fires at 40.1 ms and the skip rate drops
        # to 1%.
        self.play_timer.setTimerType(Qt.PreciseTimer)
        self.play_timer.timeout.connect(self._play_tick)
        # Calibration point of the playback clock (see `_play_tick`): from which frame and
        # which moment elapsed time is counted, and which frame we ourselves last showed.
        self._play_base_idx = 0
        self._play_base_t = 0.0
        self._play_last = -1
        # Read-ahead (see `ForwardReader`): the running thread, and the frames it handed
        # back on stopping that haven't been shown yet ({index: frame}).
        self._reader = None
        self._readahead_rest = {}
        self.set_controls_active(False)

    # ── UI construction ─────────────────────────────────────────────────
    def _build_ui(self, min_size, show_speed=True, show_overlay=True):
        self._main = QVBoxLayout(self)

        self.label = QLabel("Geen video geladen")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #111; color: #888;")
        # The explicit minimumSize is what the layout uses as the lower bound (it wins over
        # the minimumSizeHint, which for a QLabel with a pixmap is the pixmap size). So keep
        # `min_size` low: it directly determines how short the window can be.
        self.label.setMinimumSize(*min_size)
        self._main.addWidget(self.label, stretch=1)

        buttons = QHBoxLayout()
        self.btn_start = QPushButton("⏮")
        self.btn_frame_back = QPushButton("⏪")
        self.btn_play = QPushButton("▶")
        self.btn_frame_forward = QPushButton("⏩")
        self.btn_end = QPushButton("⏭")
        self.lbl_time = QLabel("t=0.00s  frame 0/0")

        self.btn_start.clicked.connect(lambda: self.go_to(0))
        self.btn_frame_back.clicked.connect(lambda: self.go_to(self.huidige_idx - 1))
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_frame_forward.clicked.connect(lambda: self.go_to(self.huidige_idx + 1))
        self.btn_end.clicked.connect(lambda: self.go_to(len(self.resultaten) - 1))

        # Putting the keyboard shortcut on the button is the one place everyone runs into
        # it: this bar sits in all four places where a video is shown.
        for button, tip in ((self.btn_start, "Naar het begin (Home)"),
                          (self.btn_frame_back, "Eén frame terug (←)"),
                          (self.btn_play, "Afspelen / pauze (spatie)"),
                          (self.btn_frame_forward, "Eén frame verder (→)"),
                          (self.btn_end, "Naar het eind (End)")):
            button.setToolTip(f"{tip}\n\n{VIDEO_KEYS_TOOLTIP}")

        for w in (self.btn_start, self.btn_frame_back, self.btn_play,
                  self.btn_frame_forward, self.btn_end):
            # One character wide: the Qt default width (81 px) is meant for buttons with
            # text and demanded 405 px per player for five transport buttons -- two players
            # side by side on the compare page then didn't fit a narrow laptop screen.
            w.setMaximumWidth(TRANSPORT_BUTTON_WIDTH)
            buttons.addWidget(w)

        # Playback speed (slow motion): the factor the fps gets multiplied by. The combo
        # always exists (it's the source for `_play_interval_ms`), but doesn't have to be
        # visible: on the compare page a single shared control drives both sides, so the
        # videos always run at the same speed.
        self.lbl_speed = QLabel("Snelheid")
        buttons.addWidget(self.lbl_speed)
        self.combo_speed = QComboBox()
        self.combo_speed.setToolTip("Afspeelsnelheid — kies een lagere factor voor slow motion.")
        for label, factor in SPEEDS:
            self.combo_speed.addItem(label, factor)
        self.combo_speed.setCurrentIndex(SPEED_DEFAULT_IDX)
        self.combo_speed.currentIndexChanged.connect(self._set_speed)
        buttons.addWidget(self.combo_speed)
        self.lbl_speed.setVisible(show_speed)
        self.combo_speed.setVisible(show_speed)

        buttons.addWidget(self.lbl_time, stretch=1)
        self._main.addLayout(buttons)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setToolTip(f"Sleep om door de video te scrubben.\n\n{VIDEO_KEYS_TOOLTIP}")
        # On a long recording, one jump costs ~80 ms (seek) + drawing, while a drag across
        # the timeline fires hundreds of valueChanged signals. Those pile up and the GUI
        # seems to lock up. `_scrub_requested` only stores the last requested frame and
        # draws it via a timer with interval 0: that only fires once the queue is empty, so
        # all the intermediate values automatically drop away and drawing happens exactly
        # as often as the machine can keep up with. Without fast_seek (short clips) the
        # direct path is left as is.
        self.slider.valueChanged.connect(
            self._scrub_requested if self.fast_seek else self.go_to)
        self._main.addWidget(self.slider)

        # A wrapping bar instead of QHBoxLayout: with all its controls this row is too wide
        # for a laptop screen (certainly two players side by side) and needs to be able to
        # collapse.
        self._toggles_bar = WrapBar()
        self._toggles_row = self._toggles_bar
        self.chk_skeleton = QCheckBox("Skelet")
        self.chk_push_leg = QCheckBox("Afzetbeen")
        self.chk_hud = QCheckBox("HUD")
        for chk in (self.chk_skeleton, self.chk_push_leg, self.chk_hud):
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _=None: self.show_current_frame())
            chk.setVisible(show_overlay)
            self._toggles_row.addWidget(chk)

        # Zooming in on the skater (the mouse wheel over the video also works -- see below).
        self.chk_follow = QCheckBox("Volg schaatser")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("Houd de schaatser gecentreerd in beeld tijdens het inzoomen.")
        self.chk_follow.toggled.connect(self._set_zoom_follow)
        self._toggles_row.addWidget(self.chk_follow)
        self.chk_auto = QCheckBox("Automatische zoom")
        self.chk_auto.setToolTip(
            "Het programma kiest de zoom: de schaatser staat helemaal in beeld met wat ruimte "
            "eromheen, de hele clip lang. Rijdt hij naar de camera toe, dan zoomt het beeld "
            "vanzelf uit.\nZolang dit aan staat is de zoomregelaar buiten werking; aan het "
            "muiswiel draaien neemt de zoom weer over.")
        self.chk_auto.toggled.connect(self._set_zoom_auto)
        self._toggles_row.addWidget(self.chk_auto)
        # Follow and automatic zoom live off the detected pose; without an analysis
        # (trim window) they'd be checkboxes that don't do anything.
        self.chk_follow.setVisible(show_overlay)
        self.chk_auto.setVisible(show_overlay)

        # The zoom controls as a single block in the bar: if they took part loose, the
        # "Zoom" label could get left behind on the previous line while its slider wraps.
        zoom_block = QWidget()
        zoom_row = QHBoxLayout(zoom_block)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.addWidget(QLabel("Zoom"))
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(100, int(ZOOM_MAX * 100))   # 100 = 1.0×
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(120)
        self.slider_zoom.setToolTip("Zoomniveau. Muiswiel boven de video werkt ook.")
        self.slider_zoom.valueChanged.connect(lambda v: self._set_zoom(v / 100.0))
        zoom_row.addWidget(self.slider_zoom)
        self.lbl_zoom = QLabel("1.0×")
        self.lbl_zoom.setFixedWidth(38)
        zoom_row.addWidget(self.lbl_zoom)
        self.btn_zoom_reset = QPushButton("Passend")
        self.btn_zoom_reset.setToolTip("Zoom herstellen naar passend beeld.")
        self.btn_zoom_reset.clicked.connect(self._reset_zoom)
        # Only with an analysis: there it also turns auto-follow back on. Without an
        # analysis (viewing/trim window) the slider at 1x is the same thing, and the
        # button would seem to do nothing.
        self.btn_zoom_reset.setVisible(show_overlay)
        zoom_row.addWidget(self.btn_zoom_reset)
        self._toggles_row.addWidget(zoom_block)

        # Drawing on the image -- only where it's turned on (the viewing window). One
        # block in the bar (same reason as zoom_block: otherwise the label gets left
        # behind on the previous line when the bar wraps). The combo decides what the
        # left button does; right-drag always keeps panning.
        self.combo_draw = self.btn_draw_undo = self.btn_draw_clear = None
        if self.drawing_on:
            draw_block = QWidget()
            draw_row = QHBoxLayout(draw_block)
            draw_row.setContentsMargins(0, 0, 0, 0)
            draw_row.addWidget(QLabel("Muis"))
            self.combo_draw = QComboBox()
            for label, mode in DRAW_MODES:
                self.combo_draw.addItem(label, mode)
            self.combo_draw.setToolTip(DRAW_TOOLTIP)
            self.combo_draw.currentIndexChanged.connect(self._set_draw_mode)
            draw_row.addWidget(self.combo_draw)
            self.btn_draw_undo = QPushButton("↶")
            self.btn_draw_undo.setMaximumWidth(TRANSPORT_BUTTON_WIDTH)
            self.btn_draw_undo.setToolTip("Laatst getekende streek weghalen.")
            self.btn_draw_undo.clicked.connect(self.clear_last_stroke)
            draw_row.addWidget(self.btn_draw_undo)
            self.btn_draw_clear = QPushButton("🧹")
            self.btn_draw_clear.setMaximumWidth(TRANSPORT_BUTTON_WIDTH)
            self.btn_draw_clear.setToolTip("Alle aantekeningen van het beeld halen.")
            # lambda: otherwise clicked() passes its `checked=False` through as
            # `redraw`, which does clear the drawing but leaves it on screen.
            self.btn_draw_clear.clicked.connect(lambda: self.clear_drawing())
            draw_row.addWidget(self.btn_draw_clear)
            self._toggles_row.addWidget(draw_block)

        self._main.addWidget(self._toggles_bar)

        # Mouse events on the video label: the player itself does panning, the rest goes
        # to the owner's hooks (the skeleton editor).
        self.label.mousePressEvent = self._mouse_press
        self.label.mouseMoveEvent = self._mouse_move
        self.label.mouseReleaseEvent = self._mouse_release
        self.label.wheelEvent = self._zoom_wheel   # mouse wheel = zoom in/out
        # tracking on: mouseMoveEvent also fires without a button held, needed for the
        # hover text that names the body part under the cursor in edit mode.
        self.label.setMouseTracking(True)
        # Right-drag pans (also in edit mode, where the left button is taken). Without
        # this, contextMenuEvent propagates to the QMainWindow, which opens its
        # toolbar/dock menu on it -- then a menu would pop open on every pan.
        self.label.setContextMenuPolicy(Qt.PreventContextMenu)

    def add_control_button(self, w):
        """Adds an owner-specific button to the right of the toggles row (e.g. '✏ Edit')."""
        self._toggles_row.addWidget(w)

    def add_bottom_bar(self, w):
        """Adds an owner-specific bar to the bottom of the panel (e.g. the editor bar)."""
        self._main.addWidget(w)

    def minimumSizeHint(self):
        # The wrapping bars (toggles/zoom, and the editor bar via add_bottom_bar) count
        # with the number of lines they need at this panel's minimum width -- not their
        # narrowest wrap and not zero lines. See minimum_with_wrapping.
        return minimum_with_wrapping(self)

    # ── Loading / releasing ──────────────────────────────────────────────
    @property
    def crop_norm(self):
        return self._crop_norm

    @property
    def display_scaled(self):
        return self._display_scaled

    def load(self, info, resultaten, video_pad, deinterlacen=False):
        """
        Puts an analysis into use: reopen the capture, reset zoom, controls on.

        Deliberately shows no frame yet -- the caller calls `go_to(0)` last. Only that way
        are the owner's table and chart already in place by the time `on_frame_shown`
        fires for the first frame.
        """
        if not video_pad:
            raise ValueError("VideoPlayer.load() without a video path")
        self.video_info = info
        self.resultaten = resultaten
        self.video_pad = video_pad

        # Reset zoom (no zoom leaking between analyses). The "Automatic zoom" setting does
        # stay put: that's a viewer preference, not a property of the clip.
        self._zoom = self._zoom_eff = 1.0
        self._pan_cx = self._pan_cy = 0.5
        self._zoom_follow = True
        self._force_follow = True     # aim at the skater right away on the first frame
        self._pan_drag = None
        self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        # Annotations belong to the clip underneath: a different analysis starts clean.
        self.clear_drawing(redraw=False)
        # Once, offline: what box does the skater need per frame? Costs a fraction of a
        # second and makes the automatic zoom independent of playback direction (scrubbing
        # gives exactly the same crop as playing toward it).
        self._box = box_sequence(resultaten, info.fps or 30.0)
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(100)
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom.setText("1.0×")
        self.chk_follow.blockSignals(True)
        self.chk_follow.setChecked(True)
        self.chk_follow.blockSignals(False)

        self._stop_readahead()         # get the capture back first, only then release it
        if self.cap is not None:
            self.cap.release()
        self.deinterlacen = bool(deinterlacen)
        self.cap = open_video(video_pad, self.deinterlacen)
        self._display_pos = 0
        self._readahead_rest = {}
        self._last_frame = None
        self.huidige_idx = -1

        # blockSignals: setRange clamps a too-high slider value and would otherwise fire
        # valueChanged -> a full seek on a table that's still from the previous analysis.
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(resultaten) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.set_controls_active(True)

    def recompute_box(self):
        """Re-derives the auto-zoom box from the current results.

        Only needed when frames have been added that didn't have a pose before (a
        manually placed skeleton): `box_sequence` opens up to the full image on a gap >
        BOX_GAP_S, so without recomputing, exactly the just-filled-in frame stays zoomed
        out. Deliberately not done after every drag correction -- that would make the
        zoom jump on every drop."""
        if not self.resultaten or self.video_info is None:
            return
        self._box = box_sequence(self.resultaten, self.video_info.fps or 30.0)

    def release(self):
        """Lets go of the video file (needed before the media folder can be deleted) and
        clears the panel."""
        self.pause()                 # also gets the capture back from the read-ahead reader
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._last_frame = None
        self._display_pos = 0
        self._readahead_rest = {}
        self._box = None
        self.clear_drawing(redraw=False)
        self.video_info = None
        self.resultaten = []
        self.huidige_idx = -1
        self.video_pad = None
        self.label.setText("Geen video geladen")
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.blockSignals(False)
        self.lbl_time.setText("t=0.00s  frame 0/0")
        self.set_controls_active(False)

    def set_controls_active(self, active):
        for w in (self.btn_start, self.btn_frame_back, self.btn_play,
                  self.btn_frame_forward, self.btn_end, self.slider,
                  self.chk_follow, self.chk_auto, self.slider_zoom, self.btn_zoom_reset):
            w.setEnabled(active)
        if self.drawing_on:
            self.btn_draw_undo.setEnabled(active)
            self.btn_draw_clear.setEnabled(active)
            # The combo also hangs off edit mode (which claims the left button).
            self.combo_draw.setEnabled(bool(active) and not self._edit_mode)
        self._set_manual_zoom_active(active)

    def _set_manual_zoom_active(self, active=None):
        """Turns the manual zoom controls on/off: if the program determines the zoom,
        they're disabled (greyed out) -- fairer than a slider that doesn't do anything."""
        if active is None:
            active = self.slider.isEnabled()
        on = bool(active) and not self._zoom_auto
        self.slider_zoom.setEnabled(on)
        self.btn_zoom_reset.setEnabled(on)

    # ── Navigation + drawing ─────────────────────────────────────────────
    def go_to(self, idx):
        if not self.resultaten:
            return
        # Jumping elsewhere is random access; the read-ahead reader has no business there
        # (it sits further along and is holding the capture). If the video keeps running,
        # `_play_tick` starts it up again on its own at the new spot.
        self._stop_readahead()
        idx = max(0, min(idx, len(self.resultaten) - 1))
        self._show_frame(idx)

    def show_current_frame(self):
        if self.huidige_idx >= 0:
            self._show_frame(self.huidige_idx)

    def show_on_clock(self, idx):
        """Show a frame under an **external** clock -- the compare page's master clock.

        Unlike `go_to`, this isn't random access but sequential-forward, so the
        read-ahead reader should indeed be running here. That's where it's needed most:
        two players run side by side, so all the work counts double -- measured, one tick
        with two 1080i recordings at 1x cost **56.4 ms** against a 30 ms tick interval. If
        the reader is still behind, we show nothing and the next tick picks it up; the
        clock keeps running, so that costs at most one frame."""
        if not self.resultaten:
            return
        idx = max(0, min(idx, len(self.resultaten) - 1))
        if idx < self.huidige_idx:
            self.go_to(idx)          # backward really is random access
            return
        if idx == self.huidige_idx:
            return
        if self._reader is None:
            self._start_readahead()
        if self._reader is None:       # above 1x: grab-skipping is cheaper there
            self._show_frame(idx)
            return
        read_idx, frame = self._reader.take(idx)
        if frame is not None:
            self._show_frame(min(read_idx, len(self.resultaten) - 1), frame)

    def _scrub_requested(self, idx):
        """Scrub request from the slider (only in `fast_seek` mode). See the note by the
        slider: only the latest target counts, the rest we drop."""
        self._scrub_target = idx
        if not self._scrub_timer.isActive():
            self._scrub_timer.start(0)

    def _scrub_tick(self):
        target = self._scrub_target
        if target != self.huidige_idx:
            self.go_to(target)
        if self._scrub_target != target:      # dragged further while drawing
            self._scrub_timer.start(0)

    def _read_frame_exact(self, idx):
        """
        Reads frame `idx` frame-exactly, exclusively via sequential reading.
        A CAP_PROP_POS_FRAMES seek is NOT frame-exact on VFR videos (e.g. iPhone .MOV):
        the decoded image can be off by a few frames while OpenCV does report the
        requested frame number. The skeleton (of the *correct* frame) then appears to
        lag behind the image -- also during playback afterward, since the error stays
        constant. That's why we track the cursor ourselves: spool forward with grab(),
        backward by reopening the video. On the short clips this tool is for, that's
        comfortably fast enough.

        With `fast_seek` (the phase-8 trim window) seeking is allowed. There the image is
        a look, not a measurement: on a half-hour recording (50,000 frames), rewinding
        from frame 0 would make browsing impossible, while a fragment boundary you set by
        eye can easily be a few frames (~0.1 s) off. After the seek, our own cursor is
        restored so forward playback is correct again.

        **Every step backward seeks**, however small: the sequential route otherwise
        re-spools from frame 0, and one frame back at frame 20,000 of a 23-minute
        recording cost nearly a minute during which the GUI completely locked up
        (measured on `00005.MTS`, 34,728 frames). Only *small* jumps forward stay
        sequential -- those are cheap (~3 ms per skipped frame against ~80 ms for a seek)
        and that keeps stepping frame by frame around a fragment boundary exact.
        """
        if self.cap is None:
            return None
        # Frames the read-ahead reader had already decoded when it handed back the
        # capture: those lie before the capture position, so without this branch a step
        # forward would count as a jump back and reopen and re-spool the video.
        if self._readahead_rest:
            frame = self._readahead_rest.pop(idx, None)
            for old in [i for i in self._readahead_rest if i <= idx]:
                del self._readahead_rest[old]
            if frame is not None:
                return frame
            self._readahead_rest = {}   # we're going elsewhere
        if self.fast_seek and (idx < self._display_pos
                                 or idx - self._display_pos > SEEK_THRESHOLD_FRAMES):
            if self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                self._display_pos = idx      # restore the cursor: forward reading is correct again
            # If the seek fails, the code below falls back to sequential spooling.
        if idx < self._display_pos:
            self.cap.release()
            self.cap = open_video(self.video_pad, self.deinterlacen)
            self._display_pos = 0
        while self._display_pos < idx:
            if not self.cap.grab():
                return None
            self._display_pos += 1
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._display_pos += 1
        return frame

    def _report_read_error(self, idx):
        """
        Frame `idx` could not be read (past the end, or a decode error).
        Silently returning is misleading: the image, slider, table marker, and time then
        stay on the previous frame while the user thinks they jumped. So: stop playback,
        set the controls back to the last valid frame, and report it.
        """
        self.pause()
        if self.huidige_idx >= 0 and self._last_frame is not None:
            self._show_frame(self.huidige_idx)       # put the slider/table back in step
            self.lbl_time.setText(
                f"Frame {idx} kon niet gelezen worden — beeld staat nog op {self.huidige_idx}")
        else:
            self.lbl_time.setText(f"Frame {idx} kon niet gelezen worden")

    def _show_frame(self, idx, frame=None):
        if frame is not None:                       # ready-made from the read-ahead reader
            self._last_frame = frame
            frame = frame.copy()
            new_frame = True
        elif idx == self.huidige_idx and self._last_frame is not None:
            frame = self._last_frame.copy()      # only redraw the overlay
            new_frame = False
        else:
            frame = self._read_frame_exact(idx)
            if frame is None:
                self._report_read_error(idx)
                return
            self._last_frame = frame
            frame = frame.copy()
            new_frame = True
        self.huidige_idx = idx

        result = self.resultaten[idx]
        # The automatic zoom differs per frame; the label (and the disabled slider, as a
        # readout) therefore show the applied factor. Compute first, then follow: even
        # without a manual zoom there can be an automatic crop, and that has to re-center
        # along with it.
        self._zoom_eff = self._compute_effective_zoom(idx)
        self.lbl_zoom.setText(f"{self._zoom_eff:.1f}×")
        if self._zoom_auto:
            self.slider_zoom.blockSignals(True)
            self.slider_zoom.setValue(int(round(min(ZOOM_MAX, self._zoom_eff) * 100)))
            self.slider_zoom.blockSignals(False)
        # Auto-follow: center the zoom crop on the skater, but only on an actual frame
        # change and not during a handle drag -- otherwise the crop would jump out from
        # under the cursor while dragging or toggling a layer.
        if ((new_frame or self._force_follow) and self._zoom_eff > 1.0
                and self._zoom_follow and not self.follow_frozen):
            # Automatic: on the box's center point, since the zoom was measured against
            # that same box -- so the skater is guaranteed to be fully in frame there.
            # Manual: on the torso, which moves more calmly than the arms and legs.
            box = self._box_at(idx) if self._zoom_auto else None
            c = (box[:2] if box is not None
                 else torso_centroid(result.lm) if result.pose_found else None)
            if c is not None:
                self._pan_cx, self._pan_cy = c   # clamping happens in _show_pixmap
        self._force_follow = False
        if self.show_overlay:
            draw_overlay_on_frame(
                frame, result, self.video_info.fps,
                toon_skelet=self.chk_skeleton.isChecked(),
                toon_afzetbeen=self.chk_push_leg.isChecked(),
                toon_hud=self.chk_hud.isChecked(),
            )
        self._show_pixmap(frame)

        if idx != self.slider.value():
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        self.lbl_time.setText(f"t={result.time:.2f}s  frame {idx}/{len(self.resultaten) - 1}")
        if self.on_frame_shown is not None:
            self.on_frame_shown(idx)

    def _show_pixmap(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        # Zooming in = scaling up a crop around the pan center. The crop keeps the same
        # aspect ratio as the frame, so the KeepAspectRatio letterbox (and hence the
        # editor's coordinate conversion) stays unchanged.
        z = max(1.0, self._zoom_eff)
        if z > 1.0:
            cw, ch = w / z, h / z
            x0 = min(max(self._pan_cx * w - cw / 2, 0.0), w - cw)   # clamp the crop within the frame
            y0 = min(max(self._pan_cy * h - ch / 2, 0.0), h - ch)
            ix0, iy0 = int(round(x0)), int(round(y0))
            icw = min(int(round(cw)), w - ix0)
            ich = min(int(round(ch)), h - iy0)
            # .copy() makes the slice C-contiguous (needed for the QImage stride) and
            # guarantees _last_frame stays at full resolution.
            frame_bgr = frame_bgr[iy0:iy0 + ich, ix0:ix0 + icw].copy()
            self._crop_norm = (ix0 / w, iy0 / h, icw / w, ich / h)
            h, w = frame_bgr.shape[:2]
        else:
            self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        # No .copy() on the QImage: it points at the numpy buffer, but `QPixmap.fromImage`
        # converts the image to the platform's pixmap format and thus copies it itself --
        # and `frame_bgr` stays alive until that line is done. The extra copy used to be a
        # memcpy of the whole image every frame (measured 0.6 ms of a 5.3 ms tick).
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # The order isn't cosmetic: the overlay drawer (skeleton editor) computes via
        # norm_to_widget using _display_scaled, so that must already be up to date.
        self._display_scaled = pixmap.size()
        # Keep the bare scaled image around, so a mouse move while drawing only has to
        # redo the drawing layer (see _refresh_drawing). Only when something is actually
        # being drawn or could be -- otherwise it costs a copy of the whole pixmap every
        # frame for nothing.
        self._base_pixmap = (pixmap.copy()
                              if self._drawing or self.draw_mode != DRAW_PAN
                              else None)
        self._draw_layers(pixmap)

    # ── Zooming in on the skater ──────────────────────────────────────────
    def _box_at(self, idx):
        """Box `(center_x, center_y, radius)` of frame `idx` (offline sequence), or None."""
        if self._box is None or not (0 <= idx < len(self._box)):
            return None
        return self._box[idx]

    def _zoom_ceiling(self):
        """How far the automaton may zoom in. At zoom 1x the frame fits the panel with
        factor `s`; at zoom z that becomes `z*s` screen pixels per video pixel. Above
        `BOX_MAX_MAGNIFICATION` that turns into visible mush, so that's where the
        automaton stops."""
        info = self.video_info
        if info is None or not info.w or not info.h:
            return ZOOM_AUTO_MAX
        s = min(self.label.width() / info.w, self.label.height() / info.h)
        if s <= 0:
            return ZOOM_AUTO_MAX
        return min(ZOOM_AUTO_MAX, max(1.0, BOX_MAX_MAGNIFICATION / s))

    def _compute_effective_zoom(self, idx):
        """The zoom actually applied to frame `idx`.

        Manually, that's simply the set zoom. Automatically, the program determines it
        from the skater itself: the crop is (normalized) 0.5/zoom in size around the box's
        center point, so we fill that with the room the skater needs plus `BOX_MARGIN` of
        air. It can't zoom out further than the full image, so up close the zoom simply
        stays at 1x."""
        z = min(ZOOM_MAX, max(1.0, self._zoom))
        if not self._zoom_auto:
            return z
        box = self._box_at(idx)
        if box is None or not box[2]:
            return z            # no usable pose: leave the manual zoom as is
        return min(self._zoom_ceiling(), max(1.0, 0.5 / (box[2] * (1.0 + BOX_MARGIN))))

    def _set_zoom(self, z):
        """Central zoom setter: clamps, updates the slider+label (without a signal loop),
        and redraws the current frame cheaply (no re-reading the video)."""
        z = min(ZOOM_MAX, max(1.0, float(z)))
        self._zoom = z
        if z <= 1.0:
            self._pan_cx = self._pan_cy = 0.5
        self._force_follow = True   # a deliberate zoom action: aim at the skater right away
        self.lbl_zoom.setText(f"{z:.1f}×")     # _show_frame fills in the effective zoom this way
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(int(round(z * 100)))
        self.slider_zoom.blockSignals(False)
        self.show_current_frame()

    def _zoom_wheel(self, event):
        """Mouse wheel over the video: zoom in/out. Auto-follow stays on, so the crop
        stays on the skater (no zoom-to-cursor, that would fight with 'follow skater')."""
        delta = event.angleDelta().y()
        if not self.resultaten or delta == 0:
            # Nothing to zoom: give the wheel back to Qt, otherwise the video label
            # swallows the scroll and, outside of a loaded analysis, nothing happens at all.
            QLabel.wheelEvent(self.label, event)
            return
        if self._zoom_auto:
            # Turning the wheel = taking over the zoom, just like manual dragging takes
            # over auto-follow. `_set_zoom_auto` picks up the current setting, so the
            # image doesn't jump -- from here on it just no longer gets steered.
            self.chk_auto.setChecked(False)
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        self._set_zoom(self._zoom * factor)
        event.accept()

    def _set_zoom_follow(self, on):
        self._zoom_follow = bool(on)
        self._force_follow = True
        self.show_current_frame()

    def _set_zoom_auto(self, on):
        """Turns automatic zoom on/off. On turning it off, the last shown zoom becomes the
        manual setting, so the image doesn't jump at that moment -- except above
        `ZOOM_MAX`, where the manual control simply stops."""
        self._zoom_auto = bool(on)
        self._set_manual_zoom_active()
        if not self._zoom_auto:
            self._set_zoom(self._zoom_eff)     # takes over, redraws, and restores the slider
            return
        self._force_follow = True
        self.show_current_frame()

    def _reset_zoom(self):
        """Back to a fitted image and auto-follow on again."""
        self._pan_cx = self._pan_cy = 0.5
        self._zoom_follow = True
        self.chk_follow.blockSignals(True)
        self.chk_follow.setChecked(True)
        self.chk_follow.blockSignals(False)
        self._set_zoom(1.0)

    # ── Coordinate conversion (letterbox + zoom crop) ─────────────────────
    def widget_to_norm(self, pos):
        """Mouse position on the video label -> normalized (x, y) in the frame (0-1).
        Outside the drawn image the result can lie outside [0,1] (the caller checks)."""
        if self._display_scaled is None:
            return None
        sw, sh = self._display_scaled.width(), self._display_scaled.height()
        if sw <= 0 or sh <= 0:
            return None
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        fx = (pos.x() - offx) / sw          # fraction within the shown crop
        fy = (pos.y() - offy) / sh
        x0n, y0n, wn, hn = self._crop_norm  # at zoom==1 this is (0,0,1,1) -> the old formula
        return (x0n + fx * wn, y0n + fy * hn)

    def norm_to_widget(self, nx, ny):
        """Inverse: normalized (x, y) -> position on the video label (for hit testing)."""
        sw, sh = self._display_scaled.width(), self._display_scaled.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        x0n, y0n, wn, hn = self._crop_norm  # at zoom==1 this is (0,0,1,1) -> the old formula
        return QPointF(offx + (nx - x0n) / wn * sw, offy + (ny - y0n) / hn * sh)

    # ── Drawing on the image ───────────────────────────────────────────────
    @property
    def edit_mode(self):
        return self._edit_mode

    @edit_mode.setter
    def edit_mode(self, active):
        """The skeleton editor claims the left button, so drawing can't happen with it at
        the same time. A drawing mode that silently stops doing anything (or worse:
        swallows a drag that was meant to move a landmark) is the worse of the two
        outcomes, so we put the mouse back to panning and lock the combo for as long as
        the editor is on. The drawing itself is left exactly as is.

        Only matters where drawing is on (the viewing window) -- and that's exactly where
        there is no skeleton editor, so in practice they never meet. The rule is here
        because the flag can be turned on from elsewhere."""
        self._edit_mode = bool(active)
        if not self.drawing_on:
            return
        if self._edit_mode and self.draw_mode != DRAW_PAN:
            self.combo_draw.setCurrentIndex(0)      # -> _set_draw_mode
        self.combo_draw.setEnabled(not self._edit_mode and self.slider.isEnabled())

    def _set_draw_mode(self, _idx=None):
        self.draw_mode = self.combo_draw.currentData() or DRAW_PAN
        self._stroke = None
        # The cursor says what the left button does right now.
        self.label.setCursor(Qt.ArrowCursor if self.draw_mode == DRAW_PAN
                             else Qt.CrossCursor)
        self.show_current_frame()      # get the base pixmap ready right away (or clear it)

    def clear_drawing(self, redraw=True):
        """Removes all annotations from the image."""
        was_something = bool(self._drawing) or self._stroke is not None
        self._drawing = []
        self._stroke = None
        if redraw and was_something:
            self._refresh_drawing()

    def clear_last_stroke(self):
        """Takes back only the last drawn stroke -- the usual correction, since one
        botched line shouldn't cost the whole annotation."""
        if not self._drawing:
            return
        self._drawing.pop()
        self._refresh_drawing()

    def _draw_point(self, pos):
        """Mouse position -> frame-normalized point, clamped to the image.

        Clamp rather than reject: drag all the way into the black bar next to the image
        and the line should end at the image edge instead of vanishing outside the frame
        -- where it would suddenly reappear on zooming out."""
        point = self.widget_to_norm(pos)
        if point is None:
            return None
        return (min(1.0, max(0.0, point[0])), min(1.0, max(0.0, point[1])))

    def _start_stroke(self, pos):
        point = self._draw_point(pos)
        if point is None:
            return
        self._stroke = [point, point]   # the second point follows the mouse
        self._refresh_drawing()

    def _extend_stroke(self, pos):
        point = self._draw_point(pos)
        if point is None or not self._stroke:
            return
        if self.draw_mode == DRAW_LINE:
            self._stroke[-1] = point   # straight line: only the end point moves
        else:
            self._stroke.append(point)
        self._refresh_drawing()

    def _stop_stroke(self):
        stroke, self._stroke = self._stroke, None
        if stroke and _drag_distance(stroke) >= DRAW_MIN_DRAG:
            self._drawing.append(stroke)
        self._refresh_drawing()

    def _paint_drawing(self, pixmap):
        """Puts the annotations onto the scaled pixmap.

        Normalized -> pixmap pixels via the same crop the skeleton editor uses, so the
        drawing stays glued to the image while zooming and panning. The transparency sits
        on the painter, not in the pen: a stroke is drawn as a single path, so an
        overlapping bend doesn't come out darker than the rest of the line."""
        strokes = self._drawing + ([self._stroke] if self._stroke else [])
        if not strokes:
            return
        pw, ph = pixmap.width(), pixmap.height()
        x0n, y0n, wn, hn = self._crop_norm   # at zoom==1 (0,0,1,1) -> nx*pw, ny*ph
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(DRAW_OPACITY)
        pen = QPen(QColor(*DRAW_COLOR), DRAW_THICKNESS)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        try:
            for stroke in strokes:
                path = QPainterPath()
                path.moveTo((stroke[0][0] - x0n) / wn * pw, (stroke[0][1] - y0n) / hn * ph)
                for nx, ny in stroke[1:]:
                    path.lineTo((nx - x0n) / wn * pw, (ny - y0n) / hn * ph)
                painter.drawPath(path)
        finally:
            painter.end()

    def _draw_layers(self, pixmap):
        """The two layers onto the scaled image and only then to the screen: first the
        annotations, then the owner's hook -- the skeleton editor must keep its handles on
        top, since those are clickable."""
        self._paint_drawing(pixmap)
        if self.overlay_drawer is not None:
            self.overlay_drawer(pixmap)
        self.label.setPixmap(pixmap)

    def _refresh_drawing(self):
        """Only redoes the drawing layer, without touching the video.

        While dragging, a redraw happens on every mouse move, and the full path (copy the
        frame, crop, QImage, scale) costs tens of milliseconds on 4K material -- then the
        line lags behind the cursor. `_base_pixmap` is the already-scaled image without
        the drawing; a copy of that is a memcpy."""
        if self._base_pixmap is None:
            self.show_current_frame()
            return
        self._draw_layers(self._base_pixmap.copy())

    # ── Mouse: the player itself does panning, the rest goes to the owner ────
    def _mouse_press(self, event):
        # These branches must stay at the top, and in this order. Right-drag *always*
        # pans -- that's the only way to shift the view while the left button is taken
        # (by the skeleton editor or by drawing). What the left button does is chosen by
        # the user in the mouse combo: in a drawing mode that's drawing, and otherwise
        # shifting the zoomed image (outside edit mode) or moving a point (in edit mode,
        # which the owner handles).
        if self._zoom_eff > 1.0 and event.button() == Qt.RightButton:
            self._pan_drag = event.position()
            return
        if event.button() == Qt.LeftButton and self.draw_mode != DRAW_PAN:
            self._start_stroke(event.position())
            return
        if (self._zoom_eff > 1.0 and event.button() == Qt.LeftButton
                and not self.edit_mode):
            self._pan_drag = event.position()
            return
        if self.on_mouse_press is not None:
            self.on_mouse_press(event)

    def _mouse_move(self, event):
        if self._pan_drag is not None:
            if self._display_scaled is None:
                return
            d = event.position() - self._pan_drag
            self._pan_drag = event.position()
            sw, sh = self._display_scaled.width(), self._display_scaled.height()
            _, _, wn, hn = self._crop_norm
            if sw > 0 and sh > 0:
                half = 0.5 / max(1.0, self._zoom_eff)
                # dragging right shows the left side -> the crop center moves along
                self._pan_cx = min(1.0 - half, max(half, self._pan_cx - d.x() / sw * wn))
                self._pan_cy = min(1.0 - half, max(half, self._pan_cy - d.y() / sh * hn))
            self._zoom_follow = False
            self.chk_follow.blockSignals(True)
            self.chk_follow.setChecked(False)
            self.chk_follow.blockSignals(False)
            self.show_current_frame()
            return
        if self._stroke is not None:
            self._extend_stroke(event.position())
            return
        if self.on_mouse_move is not None:
            self.on_mouse_move(event)

    def _mouse_release(self, event):
        if self._pan_drag is not None:
            self._pan_drag = None
            return
        if self._stroke is not None:
            self._stop_stroke()
            return
        if self.on_mouse_release is not None:
            self.on_mouse_release(event)

    # ── Playback ─────────────────────────────────────────────────────────
    def _play_step(self):
        """How many frames on average get advanced per timer tick -- the tick frequency
        follows from this (`_play_interval_ms`); *which* frame gets shown is decided by
        the wall clock in `_play_tick`.

        Up to and including 1x that's one, and the timer sets the pace. No decoder can
        keep up faster than real speed (on a 1080p recording ~9 ms per frame, so 4x = 100
        fps isn't achievable), so above that we skip frames: at 4x, four frames per tick
        on the normal fps interval. Skipped frames only cost a `grab()` (~3 ms) instead of
        a full decode."""
        factor = self.combo_speed.currentData() or 1.0
        return max(1, int(round(factor))) if factor > 1.0 else 1

    def _play_interval_ms(self):
        """Timer interval per tick, scaled by the chosen playback speed -- with the step
        size factored in, so that 4x with a step of 4 simply runs at the fps pace.

        `PLAY_OVERSAMPLE` sits on top of that: the timer fires a few times per frame, so
        a late tick costs at most part of a frame instead of a whole frame. See the note
        by that constant."""
        factor = self.combo_speed.currentData() or 1.0
        ticks_per_s = (self.video_info.fps or 30.0) * factor / self._play_step()
        return max(PLAY_TICK_MIN_MS, int(1000 / (ticks_per_s * PLAY_OVERSAMPLE)))

    def _set_speed(self, _idx=None):
        # If the video is already playing, restart the timer right away at the new pace.
        # The playback clock has to be recalibrated along with it: it computes elapsed
        # time * factor, and without recalibrating, the new factor would apply
        # retroactively to the time that already elapsed.
        if self.play_timer.isActive():
            self._calibrate_play_clock()
            self.play_timer.start(self._play_interval_ms())

    def is_playing(self):
        return self.play_timer.isActive()

    def play(self):
        if not self.resultaten or self.play_timer.isActive():
            return
        if self.huidige_idx >= len(self.resultaten) - 1:
            self.go_to(0)
        self._calibrate_play_clock()
        self._start_readahead()
        self.play_timer.start(self._play_interval_ms())
        self.btn_play.setText("⏸")

    def _calibrate_play_clock(self):
        """Records from which frame and which moment `_play_tick` counts time."""
        self._play_base_idx = self.huidige_idx
        self._play_base_t = time.perf_counter()
        self._play_last = self.huidige_idx

    # ── Read-ahead ───────────────────────────────────────────────────────
    def _start_readahead(self):
        """Hands the capture to a `ForwardReader`, unless that would gain nothing.

        **Only up to and including 1x.** Above that the player skips frames and
        `_read_frame_exact` reads the ones in between with a bare `grab()` (~3 ms) instead
        of decoding them; a read-ahead reader would decode all of them and thereby become
        the bottleneck -- at 8x, 200 frames per second are needed and it manages ~63."""
        if (self._reader is not None or self.cap is None or self.video_info is None
                or (self.combo_speed.currentData() or 1.0) > 1.0):
            return
        frame_bytes = max(1, self.video_info.w * self.video_info.h * 3)
        count = max(2, min(READAHEAD_MAX_FRAMES, READAHEAD_MAX_BYTES // frame_bytes))
        # What's left over from the previous round goes back into the buffer: those
        # frames lie before the capture position and would otherwise be skipped.
        carryover = sorted((i, f) for i, f in self._readahead_rest.items() if i > self.huidige_idx)
        self._readahead_rest = {}
        self._reader = ForwardReader(self.cap, self._display_pos, count, carryover, self)
        self._reader.start()

    def _stop_readahead(self):
        """Takes the capture back from the read-ahead reader, along with anything it
        already had ready."""
        if self._reader is None:
            return
        reader, self._reader = self._reader, None
        reader.requestInterruption()
        reader.wait(2000)
        # The capture now sits past the last frame it read -- further than what's been
        # shown. Keep the not-yet-shown frames, otherwise the next step forward would
        # look like a jump back and reopen and re-spool the video.
        self._display_pos = reader.pos
        self._readahead_rest = {i: f for i, f in reader.remaining() if i > self.huidige_idx}

    def pause(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
        self._stop_readahead()
        self.btn_play.setText("▶")

    def _toggle_playback(self):
        if self.play_timer.isActive():
            self.pause()
        else:
            self.play()

    def _play_tick(self):
        """Shows the frame that's due on the **wall clock**, not simply the next one.

        "Previous + 1 per tick" works fine as long as one frame fits inside the tick
        interval, and silently falls apart the moment it doesn't: the video then plays
        back slowed down *and* choppy, since every spike in decode time lands directly on
        top. That's not an edge case -- a 1080i camcorder recording costs ~8 ms to decode,
        ~12 ms for the comb filter, and on a HiDPI screen another ~15 ms to scale,
        together against the 40 ms available at 25 fps. By deriving the target from
        elapsed time, the pace stays correct and a shortfall gets paid in frames instead
        of in delay; a skipped frame only costs a `grab()` (~3 ms) since
        `_read_frame_exact` already spools past it sequentially anyway. Same motive (and
        same shape) as the compare page's master clock and the scrubbing in
        `PlayerKeys`.
        """
        if self.huidige_idx != self._play_last:
            self._calibrate_play_clock()     # scrubbed or jumped in the meantime: recalibrate
        factor = self.combo_speed.currentData() or 1.0
        fps = (self.video_info.fps if self.video_info else None) or 30.0
        elapsed = time.perf_counter() - self._play_base_t
        target = self._play_base_idx + int(elapsed * fps * factor)
        if target >= len(self.resultaten):
            # The end must not be skipped over; still show the last frame for a moment.
            if self.huidige_idx < len(self.resultaten) - 1:
                self._show_frame(len(self.resultaten) - 1)
            self.pause()
            return
        if target <= self.huidige_idx:  # below 1x the target stands still for several ticks
            return
        if self._reader is not None:
            idx, frame = self._reader.take(target)
            if frame is None:
                # The read-ahead reader has nothing ready yet. Show nothing and try again
                # on the next tick: the clock keeps running, so this costs at most one frame.
                if self._reader.at_end:
                    self._show_frame(len(self.resultaten) - 1)
                    self.pause()
                return
            self._show_frame(min(idx, len(self.resultaten) - 1), frame)
        else:
            self._start_readahead()       # e.g. after a jump during playback
            self._show_frame(target)
        self._play_last = self.huidige_idx


class PlayerKeys(QObject):
    """
    The keys used to view a video — the same everywhere in the app.

    Video plays back in four places (the viewing page, the compare page, the trim window,
    and the viewing window), and they only shared the mouse: the viewing window had
    space/`.`/`,`/arrow keys, the trim window only `S` and `E`, and the two pages in the
    main window had nothing — there, everything had to go through the buttons. This class
    *is* that handling, written once; each place keeps only its own extras (`extra`).

    Why a filter on the **application** and not `keyPressEvent` or `QShortcut`: after one
    mouse click, focus sits on a button or on the timeline, and those swallow space and
    the arrow keys respectively before the window ever sees them. A filter on
    QApplication gets them first. Three things go with that:
      * the non-key branch must be **short** — every event of the whole app passes
        through here;
      * a focused **text field** keeps its own keys, otherwise there's no way to type a
        name anymore and a period never makes it into the text;
      * keys **with a modifier** pass through untouched, so Ctrl+Z (undo) and Alt+F4 keep
        working and Ctrl+Left doesn't silently rewind a frame.

    `players` is a callable, because which players there are to control depends on the
    state of the window (on the compare page: which sides are filled); `active` is the
    condition on top of that — is the right page even open. If `on_play` is set,
    play/pause goes there instead of straight to the players: the compare page runs on
    one master clock, and that must not be bypassed by two separate timers. Whoever
    passes `on_play` must also pass `is_playing`, because then a player's own play_timer
    no longer means anything: under the master clock it stands still while the picture
    keeps playing, and without that answer, space would restart playback instead of
    pausing it.
    """

    def __init__(self, window, players, extra=None, active=None,
                 on_play=None, is_playing=None, on_scrub=None):
        super().__init__(window)
        self._window = window
        self._players = players
        self._extra = dict(extra or {})
        self._active = active
        self._on_play = on_play
        self._is_playing = is_playing
        self._on_scrub = on_scrub

        self._direction = 0     # -1 back, 0 idle, +1 forward
        self._running = []      # [(player, start_frame)] while scrubbing
        self._t0 = 0.0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)   # see VideoPlayer.play_timer
        self._timer.timeout.connect(self._scrub_tick)
        QApplication.instance().installEventFilter(self)

    def detach(self):
        """When the window closes: stop scrubbing and release the app-wide filter."""
        self.stop_scrubbing()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    # ── Which players ────────────────────────────────────────────────────
    def _active_players(self):
        players = self._players() if callable(self._players) else self._players
        if isinstance(players, VideoPlayer):
            players = [players]
        return [p for p in players if p is not None and p.resultaten]

    def _is_running(self, players):
        """Is anything playing? With its own playback route, play_timer is no longer the
        answer — under the compare page's master clock it stands still while the picture
        keeps playing."""
        if self._is_playing is not None:
            return self._is_playing()
        return any(p.is_playing() for p in players)

    def _pause(self, players):
        if self._on_play is not None:
            self._on_play(False)
        for p in players:
            p.pause()

    # ── Scrubbing with . and , ───────────────────────────────────────────
    def start_scrubbing(self, direction):
        """Starts scrubbing for as long as the key stays down. The first frame is taken
        right away, so a quick tap on the key advances exactly one frame and holding it
        down scrubs at 6x — both are ways this key gets used."""
        if self._direction == direction:
            return
        players = self._active_players()
        if not players:
            return
        # Stop and pause first, THEN set our own state: `_pause` goes through `on_play`
        # to the owner, and the owner calls `stop_scrubbing()` from there in turn
        # (`_pause_all` does that). The other way around would immediately reset the
        # direction we just set back to 0, and nothing would scrub.
        self.stop_scrubbing()
        self._pause(players)
        self._direction = direction
        self._running = [(p, p.huidige_idx) for p in players]
        self._t0 = time.monotonic()
        for p, start in self._running:
            p.go_to(start + direction)
        self._timer.start(SCRUB_TICK_MS)
        self._report(f"{'▶▶' if direction > 0 else '◀◀'} {SCRUB_FACTOR:g}×")

    def stop_scrubbing(self, direction=None):
        if self._direction == 0 or (direction is not None and direction != self._direction):
            return
        self._timer.stop()
        self._direction = 0
        self._running = []
        self._report("")

    def _scrub_tick(self):
        """The target frame follows from the **wall-clock time** since the key was
        pressed, not from a fixed step per tick — same motive as the compare page's
        master clock. That way it's genuinely 6x the recording speed: if the decoder
        can't keep up (going backward costs a seek per step), more frames get skipped
        instead of the scrubbing slowing down, and nothing piles up."""
        elapsed = time.monotonic() - self._t0
        done = True
        for p, start in self._running:
            info = p.video_info
            fps = (info.fps if info is not None else 0) or 30.0
            step = max(1, int(round(elapsed * fps * SCRUB_FACTOR)))
            last = len(p.resultaten) - 1
            target = start + self._direction * step
            p.go_to(max(0, min(target, last)))
            if 0 < target < last:
                done = False
        if done:
            self.stop_scrubbing()   # start/end reached: nothing left to scrub

    def _report(self, text):
        if self._on_scrub is not None:
            self._on_scrub(text)

    # ── The filter itself ──────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        # This first branch must be short: every event of the whole application passes
        # through here. False = "not handled", exactly what QObject.eventFilter would do
        # too.
        kind = event.type()
        if kind not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if not self._window.isActiveWindow():
            return False
        if self._active is not None and not self._active():
            return False
        if isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit)):
            return False
        if event.modifiers() & ~Qt.KeypadModifier:
            return False

        key = event.key()
        direction = {Qt.Key_Period: 1, Qt.Key_Comma: -1}.get(key)
        if direction is not None:
            # Skip autorepeat: while a key is held down, Windows sends a stream of
            # press/release pairs, and those would restart scrubbing roughly every
            # 30 ms — resetting the wall clock to zero each time so nothing advances.
            if not event.isAutoRepeat():
                if kind == QEvent.KeyPress:
                    self.start_scrubbing(direction)
                else:
                    self.stop_scrubbing(direction)
            return True
        if kind != QEvent.KeyPress:
            return False

        # A window's own keys go first: a place is allowed to take over a default key.
        handler = self._extra.get(key)
        if handler is not None:
            handler()
            return True

        if key == Qt.Key_F11:
            toggle_fullscreen(self._window)
            return True

        players = self._active_players()
        if not players:
            return False

        if key == Qt.Key_Space:
            self.stop_scrubbing()
            # One decision for all players: if anything is playing, everything stops.
            # Deciding per player separately lets two videos side by side drift apart.
            if self._is_running(players):
                self._pause(players)
            elif self._on_play is not None:
                self._on_play(True)
            else:
                for p in players:
                    p.play()
        elif key in (Qt.Key_Left, Qt.Key_Right):
            self.stop_scrubbing()
            self._pause(players)
            step = 1 if key == Qt.Key_Right else -1
            for p in players:
                p.go_to(p.huidige_idx + step)
        elif key == Qt.Key_Home:
            self.stop_scrubbing()
            for p in players:
                p.go_to(0)
        elif key == Qt.Key_End:
            self.stop_scrubbing()
            for p in players:
                p.go_to(len(p.resultaten) - 1)
        else:
            return False
        return True


class MasterClock(QObject):
    """
    One clock that runs two (or more) players at the same time.

    Two separate frame timers drift apart within a few seconds — two decodes + overlay +
    rescale cost more than one timer interval — and wouldn't line up at different fps
    anyway. So `_tick` computes each player's target frame from the elapsed wall-clock
    time x its own fps: self-correcting, so no drift, and under slow decoding, frames get
    skipped instead of the two sides drifting apart.

    Shared by the compare page (two analyses) and the viewing window (two raw videos);
    the owner handles what happens before starting (jumping to the sync point, pausing
    other views) and passes in the players. `factor` is a callable that returns the
    playback speed, since that lives in a combo box owned by the caller.
    """

    def __init__(self, parent, factor, on_done=None):
        super().__init__(parent)
        self._factor_source = factor
        self._on_done = on_done     # called when the clock itself reaches the end
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)   # see VideoPlayer.play_timer
        self._timer.timeout.connect(self._tick)
        self._running = []    # [(VideoPlayer, base_frame)] while playing together
        self._t0 = 0.0
        self._factor = 1.0

    def is_running(self):
        return self._timer.isActive()

    def start(self, players):
        """Runs these players together, starting from their current frame."""
        self._running = [(p, max(0, p.huidige_idx)) for p in players]
        self._factor = self._factor_source() or 1.0
        self._t0 = time.monotonic()
        self._timer.start(self._interval_ms())

    def stop(self):
        """Stops the clock and pauses only the players that were running under it. Their
        own timer already stood still, but their forward reader didn't, and that holds
        on to the capture plus up to 96 MB of decoded frames nobody's waiting on anymore.
        Deliberately not all of the owner's players: this method also hangs off each
        side's own play button ("take over manually"), and that button already ran its
        own `_toggle_playback` before us — pausing everything here would make each
        side's own play button useless."""
        if self._timer.isActive():
            self._timer.stop()
        running, self._running = self._running, []
        for p, _ in running:
            p.pause()

    def recalibrate(self):
        """On a speed change: recalibrate from the current position, otherwise the target
        frame would jump backward — the factor would otherwise apply retroactively to
        time already elapsed. The speed also changes how many frames per second need to
        pass, i.e. how often the clock needs to tick."""
        if not self._timer.isActive():
            return
        self._running = [(p, max(0, p.huidige_idx)) for p, _ in self._running]
        self._factor = self._factor_source() or 1.0
        self._t0 = time.monotonic()
        self._timer.start(self._interval_ms())

    def _interval_ms(self):
        """Clock interval: a fraction of the *shortest* frame among the players running
        together, since the clock has to be able to serve every player — exactly the
        same trade-off as in `_play_interval_ms`, including `PLAY_OVERSAMPLE`. Never
        slower than `ALL_TICK_MS`, so slow motion doesn't tick needlessly often."""
        fps = max((p.video_info.fps or 30.0
                   for p, _ in self._running if p.video_info is not None), default=30.0)
        fps *= self._factor
        return max(PLAY_TICK_MIN_MS, min(ALL_TICK_MS, int(1000 / (fps * PLAY_OVERSAMPLE))))

    def _tick(self):
        t = (time.monotonic() - self._t0) * self._factor
        done = True
        for p, base in self._running:
            info = p.video_info
            if info is None or not p.resultaten:
                continue
            last = len(p.resultaten) - 1
            target = base + int(round(t * (info.fps or 30.0)))
            if target < last:
                # `show_on_clock`, not `go_to`: this is sequential-forward, so the
                # forward reader is allowed to help — two players run side by side here.
                p.show_on_clock(target)
                done = False
            else:
                # The end: show exactly the last frame. `show_on_clock` would leave a
                # frame the reader doesn't have yet to the next tick, and that tick never
                # comes — the clock stops below.
                p.go_to(last)
        if done:
            self.stop()
            if self._on_done is not None:
                self._on_done()


class CompareSide(QWidget):
    """
    One side of the compare page: a header with the chosen analysis, its own VideoPlayer,
    a sync point (start frame for 'Start alles') and a minimal push table.

    The table deliberately shows little -- number, leg and angle -- but does mark
    incomplete pushes: putting two analyses side by side invites comparing two angles, and
    an incomplete push is exactly the one that's systematically too steep.
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
        # ElideLabel: a long name + title must not make this side (and so the main window)
        # wider than the screen -- see the class.
        self.lbl_titel = ElideLabel(f"{naam} — nog geen analyse gekozen")
        self.lbl_titel.setStyleSheet("font-weight: bold; padding: 2px;")
        kop.addWidget(self.lbl_titel, stretch=1)
        self.btn_kies = QPushButton("Kies analyse...")
        self.btn_kies.clicked.connect(kies_callback)
        kop.addWidget(self.btn_kies)
        # Clearing is wired up from outside (see _bouw_vergelijkpagina): the master clock
        # must let go first, and it doesn't know this side the other way around.
        self.btn_leeg = QPushButton("✕")
        self.btn_leeg.setToolTip("Deze kant leegmaken.")
        self.btn_leeg.setEnabled(False)
        kop.addWidget(self.btn_leeg)
        v.addLayout(kop)

        # No speed control of its own: the shared control at the bottom of the compare
        # page drives both sides, so the two videos never run at a different tempo. Lower
        # bound 440x180, not 320x200: at 320 wide the toggle bar breaks into three rows,
        # into two from ~435, and that row plus 20 px of picture is exactly what the
        # compare page (the tallest of the three) had too much of on a 1280x720 screen.
        # Two 440-wide sides still fit comfortably on 1280 (see schaats_schermtest.py).
        self.speler = VideoPlayer(min_size=(440, 180), show_speed=False)
        # The HUD is drawn at fixed full-frame positions and is unreadable in a half
        # panel; can be switched back on per side.
        self.speler.chk_hud.setChecked(False)
        v.addWidget(self.speler, stretch=1)

        rij_sync = QHBoxLayout()
        self.btn_sync = QPushButton("⚑ Zet sync hier")
        self.btn_sync.setToolTip(
            "Legt het huidige frame vast als startpunt voor 'Start alles', zodat beide\n"
            "video's op dezelfde fase van de slag beginnen.\n"
            "Let op: sync-punten gelden voor deze sessie en worden niet opgeslagen.")
        self.btn_sync.clicked.connect(self._set_sync)
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
        self.tabel.cellClicked.connect(self._click_row)
        v.addWidget(self.tabel)

        self.btn_sync.setEnabled(False)

    # ── Filling / emptying ───────────────────────────────────────────────
    def toon(self, analyse_id, schaatser_naam, data):
        """Takes a loaded analysis (dict from MainWindow._laad_analyse_data) into use.

        Loading the same analysis again (e.g. after an edit) keeps the sync point: that
        belongs to the video, not to the loading. A *different* analysis starts over at
        frame 0."""
        zelfde = analyse_id == self.analyse_id
        self.analyse_id = analyse_id
        self.events = data["events"]
        self.sync_frame = (min(self.sync_frame, max(0, len(data["resultaten"]) - 1))
                           if zelfde else 0)
        self.lbl_titel.setText(f"{schaatser_naam} — {data['titel']}")
        self.speler.load(data["info"], data["resultaten"], data["video_pad"],
                         data.get("deinterlaced", False))
        self._fill_table()
        self.btn_kies.setText("Wisselen...")
        self.btn_sync.setEnabled(True)
        self.btn_leeg.setEnabled(True)
        self.speler.go_to(self.sync_frame)
        self._show_sync_label()

    def leeg(self):
        """Releases the video (needed before the media folder can be deleted)."""
        self.speler.release()
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

    # ── Sync point ───────────────────────────────────────────────────────
    def _set_sync(self):
        if not self.heeft_analyse():
            return
        self.sync_frame = max(0, self.speler.huidige_idx)
        self._show_sync_label()

    def _show_sync_label(self):
        if not self.heeft_analyse():
            self.lbl_sync.setText("sync: frame 0")
            return
        tijd = self.speler.resultaten[self.sync_frame].tijd
        self.lbl_sync.setText(f"sync: frame {self.sync_frame}  (t={tijd:.2f}s)")

    def naar_sync(self):
        if self.heeft_analyse():
            self.speler.go_to(self.sync_frame)

    # ── Table ────────────────────────────────────────────────────────────
    def _fill_table(self):
        self.tabel.setRowCount(len(self.events))
        incomplete_color = QColor(70, 70, 70)
        for i, ev in enumerate(self.events):
            waarden = [str(i + 1), ev.been.capitalize(), f"{ev.hoek:.1f}"]
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setTextAlignment(Qt.AlignCenter)
                if ev.onvolledig:
                    item.setBackground(incomplete_color)
                    item.setToolTip(
                        f"Onvolledige afzet ({ev.onvolledig}) — de push is niet "
                        "afgemaakt, dus deze hoek is te steil en niet vergelijkbaar.")
                self.tabel.setItem(i, kolom, item)

    def _click_row(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self.speler.go_to(self.events[rij].start_frame)


# ── Fragmenten knippen uit een lange opname (fase 8) ───────────────────────────

def _time_text(frames, fps):
    """Framenummer → "m:ss" (of "h:mm:ss" op een lange opname)."""
    sec = int(round(frames / (fps or 30.0)))
    if sec >= 3600:
        return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60}:{sec % 60:02d}"


def _read_time(tekst, fps):
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


class FragmentBar(QWidget):
    """
    The bar under the trim window's timeline: one colored block per marked stretch, at
    its own spot in the recording. This is the only genuinely new drawing code in phase 8.

    Colors: **green** = just marked, **gray** = already analyzed in an earlier session
    (from `bron_fragmenten`), **orange** = the running, not-yet-stopped fragment. Where two
    marked stretches overlap, the overlapping part is hatched **red** -- made visible, but
    nothing is merged or shortened automatically. The trainer adjusts it themself or leaves
    it as it is (the trimming is entirely manual).
    """
    CLICKED = Signal(int)        # index into `fragmenten` of the clicked block (-1 = beside it)

    HEIGHT = 26
    COLOR_BACKGROUND = QColor(45, 45, 45)
    COLOR_DONE = QColor(120, 120, 120)
    COLOR_NEW = QColor(60, 160, 80)
    COLOR_RUNNING = QColor(220, 150, 40)
    COLOR_OVERLAP = QColor(200, 60, 60)
    COLOR_CURSOR = QColor(240, 240, 240)
    COLOR_SELECTION = QColor(255, 255, 255)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
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
        """Update everything the bar shows in one call (and redraw)."""
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
        p.fillRect(self.rect(), self.COLOR_BACKGROUND)
        h = self.height()

        for start, eind, _label in self.gedaan:
            self._blok(p, start, eind, self.COLOR_DONE, 4, h - 8)
        for i, (start, eind) in enumerate(self.fragmenten):
            self._blok(p, start, eind, self.COLOR_NEW, 2, h - 4)
            if i == self.selectie:
                p.setPen(QPen(self.COLOR_SELECTION, 2))
                p.setBrush(Qt.NoBrush)
                x0, x1 = self._x(start), self._x(eind)
                p.drawRect(QRect(x0, 1, max(2, x1 - x0), h - 3))
        # Overlap after the blocks, so the hatching lies on top of them.
        for i, (a0, a1) in enumerate(self.fragmenten):
            for b0, b1 in self.fragmenten[i + 1:]:
                s, e = max(a0, b0), min(a1, b1)
                if s <= e:
                    self._blok(p, s, e, self.COLOR_OVERLAP, 2, h - 4)
        if self.lopend is not None:
            self._blok(p, self.lopend, max(self.lopend, self.cursor),
                       self.COLOR_RUNNING, 2, h - 4)

        p.setPen(QPen(self.COLOR_CURSOR, 1))
        x = self._x(self.cursor)
        p.drawLine(x, 0, x, h)

    def _blok(self, p, start, eind, kleur, y, hoogte):
        x0, x1 = self._x(start), self._x(eind)
        p.fillRect(QRect(x0, y, max(2, x1 - x0), hoogte), kleur)

    def mousePressEvent(self, event):
        frame = int(event.position().x() / max(1, self.width() - 1) * self.totaal)
        for i, (start, eind) in enumerate(self.fragmenten):
            if start <= frame <= eind:
                self.CLICKED.emit(i)
                return
        self.CLICKED.emit(-1)


class FragmentPicker(QDialog):
    """
    The trim window (ROADMAP phase 8): go through a half-hour recording and mark the
    usable stretches. Yields a list `(start_frame, eind_frame, naam)`; the actual trim is
    done by `schaats_analyse.trim_fragments`, and the clips then enter the existing batch
    flow as pre-filled rows.

    **This is a trimming tool and the trimming is entirely manual.** The app decides
    nothing itself: not when the skater is in frame, not where a stretch starts or ends,
    and no margin is added or removed. The trainer watches, presses start and stop, and
    that's the boundary.

    Reuses `VideoPlayer` for playback (transport buttons, speed combo to scan through a
    half hour at 4x, zoom), with two departures from the viewing page: `fast_seek=True` --
    scrubbing backward is allowed to seek here, since the picture is just a look, not a
    measurement -- and `show_overlay=False`, since there's no analysis to draw.
    """

    def __init__(self, bron_pad, info, gedaan=(), deinterlacen=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Fragmenten knippen — {os.path.basename(bron_pad)}")
        self.info = info
        self.fps = info.fps or 30.0
        self._fragmenten = []          # [(start, eind)] in markeervolgorde
        self._start_open = None        # startframe van het lopende fragment
        self._stam = os.path.splitext(os.path.basename(bron_pad))[0]

        # Keep it tight: this window has to fit entirely on a laptop screen (1280x800,
        # working area 752 px) -- otherwise the button bar sinks below the edge and
        # "Klaar" becomes unreachable. Every minimum below is therefore deliberately low;
        # the picture stretches on its own when there's room.
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        uitleg = QLabel("Markeer de bruikbare stukken: <b>Start</b> (S) — <b>Stop</b> (E). "
                        "Er wordt exact op die frames geknipt.")
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)

        self.lbl_spoel = QLabel("")
        self.lbl_spoel.setStyleSheet("color: #5aaaf0;")

        self.speler = VideoPlayer(min_size=(400, 200), fast_seek=True,
                                  show_overlay=False)
        self.speler.on_frame_shown = self._frame_shown
        v.addWidget(self.speler, stretch=1)

        # The bar hangs inside the player (under the scrub slider), so it has the same
        # width as the timeline and shifts along with it.
        self.balk = FragmentBar()
        self.balk.CLICKED.connect(self._click_bar)
        self.speler.add_bottom_bar(self.balk)

        # Navigation aid: on a half-hour recording the slider is too coarse to find a
        # push again.
        rij_nav = WrapBar()
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
        rij_nav.addWidget(self.lbl_spoel)
        rij_nav.addStretch(1)
        v.addWidget(rij_nav)

        # Marking buttons + the list of marked stretches.
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
        self.tabel.cellClicked.connect(self._click_row)
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
        knoppen.accepted.connect(self._confirm)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)

        hulp = QLabel(keys_help("<b>S</b> start fragment", "<b>E</b> stop fragment",
                                   "<b>Del</b> fragment weg"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        v.addWidget(hulp)

        # Empty FrameResult list: the player wants one (slider length, time label), but
        # nothing has been analyzed yet. `box_sequence` then returns None and the zoom
        # stays manual.
        resultaten = [FrameResult(i, i / self.fps) for i in range(max(1, info.totaal))]
        self.speler.load(info, resultaten, bron_pad, deinterlacen)
        self.balk.zet(totaal=len(resultaten),
                      gedaan=[(f["start_frame"], f["eind_frame"],
                               f["titel"] or "") for f in gedaan])
        self.speler.go_to(0)
        self._refresh()

        # The same keys as everywhere else, plus S/E/Del for marking. Deliberately no
        # QShortcut anymore for those three: a letter shortcut would also fire while
        # typing in the "go to" field, and the filter leaves a focused text field alone.
        self.toetsen = PlayerKeys(
            self, lambda: [self.speler],
            extra={Qt.Key_S: self._start_fragment, Qt.Key_E: self._stop_fragment,
                   Qt.Key_Delete: self._delete_selection},
            on_scrub=self.lbl_spoel.setText)

        # Only after everything is in place: that way the clamp in set_window_size can
        # work against a final layout (and a `resize()` is ignored once the content is
        # bigger than what's requested -- hence every minimum above being kept low).
        set_window_size(self, 1100, 720)

    # ── Marking ──────────────────────────────────────────────────────────
    def _start_fragment(self):
        if self._start_open is not None:
            return
        self._start_open = self.speler.huidige_idx
        self._refresh()

    def _stop_fragment(self):
        """Closes the running fragment. A stop before the start is a mistake, not a
        fragment: better to record nothing than to trim a reversed stretch."""
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
        self._refresh(selectie=len(self._fragmenten) - 1)

    def _delete_selection(self):
        rij = self.tabel.currentRow()
        if 0 <= rij < len(self._fragmenten):
            self._fragmenten.pop(rij)
            self._refresh()

    def _click_row(self, rij, _kolom=0):
        if 0 <= rij < len(self._fragmenten):
            self.speler.go_to(self._fragmenten[rij][0])
            self.balk.zet(selectie=rij)

    def _click_bar(self, index):
        if index < 0:
            return
        self.tabel.selectRow(index)
        self._click_row(index)

    # ── Navigation ───────────────────────────────────────────────────────
    def _spring(self, seconden):
        self.speler.go_to(self.speler.huidige_idx + int(round(seconden * self.fps)))

    def _ga_naar_tijd(self):
        frame = _read_time(self.veld_tijd.text(), self.fps)
        if frame is None:
            QMessageBox.information(self, "Tijd", "Gebruik m:ss (bijvoorbeeld 12:30).")
            return
        self.speler.go_to(frame)

    def _frame_shown(self, idx):
        self.balk.zet(cursor=idx, lopend=self._start_open)
        if self._start_open is not None:
            self.lbl_lopend.setText(
                f"Loopt vanaf {_time_text(self._start_open, self.fps)} — "
                f"nu {_time_text(idx, self.fps)}")

    # ── Refreshing the view ──────────────────────────────────────────────
    def _refresh(self, selectie=None):
        self.btn_start.setEnabled(self._start_open is None)
        self.btn_stop.setEnabled(self._start_open is not None)
        self.lbl_lopend.setText(
            "" if self._start_open is None
            else f"Loopt vanaf {_time_text(self._start_open, self.fps)}")

        self.tabel.setRowCount(len(self._fragmenten))
        for i, (start, eind) in enumerate(self._fragmenten):
            duur = (eind - start + 1) / self.fps
            waarden = [str(i + 1),
                       f"{_time_text(start, self.fps)} – {_time_text(eind, self.fps)}",
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
            self._refresh()

    # ── Closing ──────────────────────────────────────────────────────────
    def _confirm(self):
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
        """[(start_frame, eind_frame, naam)] sorted by start frame. The name doubles as
        both the clip's file name and the suggested analysis title; the start time in it
        makes it recognizable and, in practice, unique (trim_fragments dedupes the rest)."""
        return [(start, eind, f"{self._stam} {_time_text(start, self.fps).replace(':', '-')}")
                for start, eind in sorted(self._fragmenten)]

    def changeEvent(self, event):
        # No longer active (alt-tab, a message box in front): the key-release then never
        # arrives and scrubbing would run forever.
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.toetsen.stop_scrubbing()
        super().changeEvent(event)

    def done(self, resultaat):
        # Not closeEvent: a modal dialog that closes via accept()/reject() doesn't get one.
        # The video file must be released, or Windows keeps holding the recording.
        self.toetsen.detach()
        self.speler.release()
        super().done(resultaat)


class PointsBar(QWidget):
    """
    The bar under the viewing window's timeline: one tick per saved point, at its own
    spot in the recording. The counterpart of `FragmentBar` -- which draws stretches
    (start-end), these are single moments.

    Clicking on (or right next to) a tick jumps to it; that's the fastest way back to the
    same picture, while the list beside it mainly serves to show *what* a point is.
    """
    CLICKED = Signal(int)        # index into `punten` of the clicked tick (-1 = beside it)

    HEIGHT = 22
    HIT_PX = 6               # how far off a tick a click still counts
    COLOR_BACKGROUND = QColor(45, 45, 45)
    COLOR_POINT = QColor(90, 170, 240)
    COLOR_SELECTION = QColor(255, 255, 255)
    COLOR_CURSOR = QColor(240, 240, 240)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setToolTip("Bewaarde punten in deze opname. Klik op een streepje om erheen "
                        "te springen.")
        self.totaal = 1
        self.punten = []          # [(frame, label)] op framenummer gesorteerd
        self.cursor = 0
        self.selectie = -1

    def zet(self, totaal=None, punten=None, cursor=None, selectie=None):
        """Update everything the bar shows in one call (and redraw)."""
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
        p.fillRect(self.rect(), self.COLOR_BACKGROUND)
        h = self.height()
        for i, (frame, _label) in enumerate(self.punten):
            x = self._x(frame)
            kleur = self.COLOR_SELECTION if i == self.selectie else self.COLOR_POINT
            p.fillRect(QRect(max(0, x - 1), 3, 3, h - 6), kleur)
            # The number goes with it as long as it's one of the first nine: that's also
            # the key you press to jump there, so it's not there just for decoration.
            if i < 9:
                p.setPen(QPen(kleur))
                p.drawText(QRect(x - 10, 2, 20, h - 4),
                           Qt.AlignHCenter | Qt.AlignTop, str(i + 1))
        p.setPen(QPen(self.COLOR_CURSOR, 1))
        x = self._x(self.cursor)
        p.drawLine(x, 0, x, h)

    def mousePressEvent(self, event):
        klik_x = event.position().x()
        dichtst, beste = -1, self.HIT_PX + 1
        for i, (frame, _label) in enumerate(self.punten):
            afstand = abs(self._x(frame) - klik_x)
            if afstand < beste:
                dichtst, beste = i, afstand
        self.CLICKED.emit(dichtst)


class ViewSide(QWidget):
    """
    One video in the viewing window: a header with the name (and a ✕ once there are two),
    its own `VideoPlayer` with the `PointsBar` under it, and a sync row for "Start alles".

    The counterpart of `CompareSide`, without analysis and without a table: nothing is
    measured here. The header and sync row are only visible when there are two sides (see
    `ViewWindow._zet_modus`) -- with one video the name is already in the window title,
    and there's nothing to sync.
    """

    def __init__(self, bron, info, parent=None):
        super().__init__(parent)
        self.bron = bron
        self.info = info
        self.fps = info.fps or 30.0
        self.sync_frame = 0
        self.punten_aan = bron.get("id") is not None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        self.kop = QWidget()
        kop = QHBoxLayout(self.kop)
        kop.setContentsMargins(0, 0, 0, 0)
        self.lbl_titel = ElideLabel(bron["naam"])   # long file name: shorten, don't widen
        self.lbl_titel.setStyleSheet("font-weight: bold;")
        kop.addWidget(self.lbl_titel, stretch=1)
        # Closing is wired up from outside (see ViewWindow._voeg_kant): the clock must
        # let go first, and it doesn't know this side the other way around.
        self.btn_leeg = QPushButton("✕")
        self.btn_leeg.setToolTip("Deze video sluiten; de andere blijft staan.")
        kop.addWidget(self.btn_leeg)
        v.addWidget(self.kop)

        # The same flags as the trim window, plus drawing: here -- and only here -- you
        # can draw over the picture. This is the window where you look and point things
        # out; on the viewing and compare pages the left button is already taken (panning,
        # the skeleton editor), and in the trim window you're setting boundaries.
        self.speler = VideoPlayer(min_size=(400, 200), fast_seek=True,
                                  show_overlay=False, show_drawing=True)
        v.addWidget(self.speler, stretch=1)
        # The bar hangs inside the player, so it has the same width as the timeline.
        self.balk = PointsBar()
        self.speler.add_bottom_bar(self.balk)

        self.rij_sync = QWidget()
        rij = QHBoxLayout(self.rij_sync)
        rij.setContentsMargins(0, 0, 0, 0)
        self.btn_sync = QPushButton("⚑ Zet sync hier")
        self.btn_sync.setToolTip(
            "Legt het huidige frame vast als startpunt voor 'Start alles', zodat beide\n"
            "video's op dezelfde fase van de slag beginnen.\n"
            "Let op: sync-punten gelden voor deze sessie en worden niet opgeslagen.")
        self.btn_sync.clicked.connect(self._set_sync)
        rij.addWidget(self.btn_sync)
        self.lbl_sync = QLabel("")
        self.lbl_sync.setStyleSheet("color: #888;")
        rij.addWidget(self.lbl_sync)
        rij.addStretch(1)
        v.addWidget(self.rij_sync)

        # Empty FrameResult list: the player wants one (slider length, time label), but
        # by definition nothing has been analyzed here -- that's the whole point.
        resultaten = [FrameResult(i, i / self.fps) for i in range(max(1, info.totaal))]
        self.speler.load(info, resultaten, bron["pad"], bool(bron.get("interlaced")))
        self.balk.zet(totaal=len(resultaten))
        self._show_sync_label()

    def release(self):
        self.speler.release()

    # ── Sync point (per session, not saved) ──────────────────────────────
    def _set_sync(self):
        self.sync_frame = max(0, self.speler.huidige_idx)
        self._show_sync_label()

    def _show_sync_label(self):
        self.lbl_sync.setText(
            f"sync: frame {self.sync_frame}  ({_time_text(self.sync_frame, self.fps)})")

    def naar_sync(self):
        self.speler.go_to(self.sync_frame)


class ViewWindow(QDialog):
    """
    Watch a recording by hand: step through footage straight from the camera, without
    analysis. **Nothing is detected, tracked, or measured** -- this is a pure player, and
    that's exactly why it's usable on material the tracking would make nothing of (the
    corner, several skaters crossing, a warm-up) and on a recording that hasn't been
    trimmed yet.

    Four things make it more than a player:
      * **full screen** -- you're watching technique, not buttons (F11 -> window);
      * **zooming and slowing down** come unchanged from `VideoPlayer` (mouse wheel/zoom
        control and the speed combo down to 1/16x);
      * the **default keys** (space, `.`/`,`, arrows, Home/End) from `PlayerKeys` -- devised
        here, but the same everywhere in the app ever since;
      * **points** you set on a frame and that stay saved (`bron_markering`), so the same
        jump or push is still there next session -- and, in the shared library, for a
        colleague too.

    **Two videos side by side.** The window shows one or two `ViewSide`s; the second comes
    from the recordings list (two rows selected) or via "➕ Tweede video ernaast..."
    (`kies_tweede`, a callable from MainWindow that does the file picker, the registration
    as a loose video, and the availability check -- those don't belong in this window).
    With two sides, playback runs on the same `MasterClock` as the compare page, with a
    sync point per side and one shared speed; space then drives the clock. The **points
    only exist with one video** -- with two, each video has exactly one sync point (per
    session, not saved), and as soon as a side closes with ✕ the points of the remaining
    video come back. `_zet_modus` is the one place that handles that difference.

    `bron` is a row from `skate_db` -- a recording from `opnames/` or a loose video from
    this pc (`bronvideo_voor_pad`); the window doesn't care which. Only when there's no
    row (`bron['id'] is None`, registration failed) do the points fall away.
    """

    PANEL_WIDTH = 260

    def __init__(self, paren, trainer_naam="", kies_tweede=None, parent=None):
        """`paren` = one or two `(bron, info)`; `kies_tweede` supplies one more (or None)
        on request, for the "Tweede video ernaast" button."""
        super().__init__(parent)
        self.trainer_naam = trainer_naam
        self.kies_tweede = kies_tweede
        self.kanten = []
        self._punt_kant = None      # the side the points panel is attached to
        self._punten = []           # rows from bron_markering, sorted by frame number
        self._vullen = False        # suppresses itemChanged while building

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        kop = QHBoxLayout()
        self.lbl_naam = QLabel("")
        kop.addWidget(self.lbl_naam)
        self.lbl_spoel = QLabel("")
        self.lbl_spoel.setStyleSheet("color: #5aaaf0;")
        kop.addWidget(self.lbl_spoel)
        kop.addStretch(1)
        self.lbl_geen_punten = QLabel("punten worden niet bewaard")
        self.lbl_geen_punten.setStyleSheet("color: #888;")
        kop.addWidget(self.lbl_geen_punten)
        self.btn_tweede = QPushButton("➕ Tweede video ernaast...")
        self.btn_tweede.setToolTip(
            "Zet een tweede video naast deze, om ze synchroon te bekijken. De video komt\n"
            "daarna als 'losse video' in de opnamelijst. Met twee video's zijn er geen\n"
            "punten, wel een sync-punt per video.")
        self.btn_tweede.clicked.connect(self._voeg_tweede_toe)
        kop.addWidget(self.btn_tweede)
        self.btn_paneel = QPushButton("Punten verbergen")
        self.btn_paneel.clicked.connect(self._toggle_paneel)
        kop.addWidget(self.btn_paneel)
        btn_venster = QPushButton("Venstermodus (F11)")
        btn_venster.clicked.connect(self._toggle_volledig_scherm)
        kop.addWidget(btn_venster)
        # Full screen has no title bar, so no Windows minimize button either.
        btn_min = QPushButton("Minimaliseren")
        btn_min.setToolTip("Zet het venster even in de taakbalk; de video's blijven staan.")
        btn_min.clicked.connect(self._minimaliseer)
        kop.addWidget(btn_min)
        btn_sluit = QPushButton("Sluiten (Esc)")
        btn_sluit.clicked.connect(self.accept)
        kop.addWidget(btn_sluit)
        v.addLayout(kop)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter_kanten = QSplitter(Qt.Horizontal)      # de video's
        self.splitter.addWidget(self.splitter_kanten)
        self.splitter.addWidget(self._bouw_puntenpaneel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        v.addWidget(self.splitter, stretch=1)

        # The shared controls for two sides -- same pattern as the compare page.
        self.balk_alles = QWidget()
        balk = QHBoxLayout(self.balk_alles)
        balk.setContentsMargins(0, 0, 0, 0)
        self.btn_start_alles = QPushButton("▶ Start alles")
        self.btn_start_alles.setToolTip(
            "Speelt beide video's tegelijk af vanaf hun sync-punt, elk op z'n eigen fps.\n"
            "Spatie speelt ook beide tegelijk, maar hervat waar ze nu staan.")
        # lambda: clicked() would otherwise pass `checked=False` as vanaf_sync.
        self.btn_start_alles.clicked.connect(lambda: self._start_alles())
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
            "Uit: 'Start alles' hervat waar beide video's nu staan, zonder terug te springen\n"
            "(spatie doet dat altijd).")
        balk.addWidget(self.chk_vanaf_sync)
        balk.addWidget(QLabel("Snelheid"))
        self.combo_alles_snelheid = QComboBox()
        self.combo_alles_snelheid.setToolTip(
            "Afspeelsnelheid voor beide video's — ook als je er één los afspeelt, zodat ze "
            "altijd even snel lopen.")
        for label, factor in SPEEDS:
            self.combo_alles_snelheid.addItem(label, factor)
        self.combo_alles_snelheid.setCurrentIndex(ALL_SPEED_IDX)
        self.combo_alles_snelheid.currentIndexChanged.connect(self._zet_alles_snelheid)
        balk.addWidget(self.combo_alles_snelheid)
        balk.addStretch(1)
        v.addWidget(self.balk_alles)

        self.hulp = QLabel("")
        self.hulp.setWordWrap(True)
        self.hulp.setStyleSheet("color: #888;")
        v.addWidget(self.hulp)

        self.klok = MasterClock(
            self, factor=lambda: self.combo_alles_snelheid.currentData() or 1.0,
            on_done=self._stop_alles)

        for bron, info in paren:
            self._voeg_kant(bron, info)

        # The default keys, plus the points keys that only exist here. The latter do
        # nothing as long as there are two videos (see _zet_punt etc.).
        extra = {Qt.Key_P: self._zet_punt, Qt.Key_Delete: self._verwijder_punt}
        for n in range(9):
            extra[Qt.Key_1 + n] = lambda i=n: self._ga_naar_punt(i)
        self.toetsen = PlayerKeys(
            self, lambda: [k.speler for k in self.kanten], extra=extra,
            on_play=self._toetsen_afspelen,
            is_playing=lambda: self.klok.is_running() or any(k.speler.is_playing() for k in self.kanten),
            on_scrub=self.lbl_spoel.setText)

        self._zet_modus()

        # First set a normal size, only then full screen: otherwise F11 has no sensible
        # geometry to fall back to.
        set_window_size(self, 1280, 800)
        self.setWindowState(self.windowState() | Qt.WindowFullScreen)

    # ── Sides ────────────────────────────────────────────────────────────
    def _voeg_kant(self, bron, info):
        kant = ViewSide(bron, info)
        kant.btn_leeg.clicked.connect(lambda _=False, k=kant: self._verwijder_kant(k))
        # Pressing ▶ yourself = taking over manual control: let the clock go.
        kant.speler.btn_play.clicked.connect(self._stop_alles)
        kant.balk.CLICKED.connect(self._click_bar)   # the bar is only visible with points
        self.kanten.append(kant)
        self.splitter_kanten.addWidget(kant)
        kant.speler.go_to(0)
        return kant

    def _verwijder_kant(self, kant):
        """✕ on a side: release that video and carry on with the other -- keeping points."""
        if len(self.kanten) < 2 or kant not in self.kanten:
            return
        self._pauzeer_alles()
        self.kanten.remove(kant)
        kant.release()
        kant.setParent(None)
        kant.deleteLater()
        self._zet_modus()

    def _voeg_tweede_toe(self):
        if self.kies_tweede is None or len(self.kanten) != 1:
            return
        self._pauzeer_alles()
        paar = self.kies_tweede()
        if not paar:
            return
        self._voeg_kant(*paar)
        self._zet_modus()

    def _zet_modus(self):
        """One video or two -- the one place that handles the difference.

        One video: the points panel and points bar (if the video has a row in the
        library), the player's own speed control, and the button for a second video. Two
        videos: per side a header (name + ✕) and sync row, the shared bottom bar with
        "Start alles" and one speed, and no points -- the handlers stay attached to the
        keys but do nothing then."""
        twee = len(self.kanten) > 1
        self.lbl_naam.setText("  |  ".join(f"<b>{k.bron['naam']}</b>" for k in self.kanten))
        self.setWindowTitle("Bekijken — " + " | ".join(k.bron["naam"] for k in self.kanten))

        for kant in self.kanten:
            kant.kop.setVisible(twee)
            kant.rij_sync.setVisible(twee)
            kant.speler.lbl_speed.setVisible(not twee)
            kant.speler.combo_speed.setVisible(not twee)
        self.balk_alles.setVisible(twee)
        self.btn_tweede.setVisible(not twee and self.kies_tweede is not None)

        self._koppel_punten(None if twee else self.kanten[0])
        for kant in self.kanten:
            kant.balk.setVisible(kant is self._punt_kant)
        if twee:
            self._zet_alles_snelheid()
            self.hulp.setText(keys_help("beide video's tegelijk",
                                           "<b>Esc</b> sluiten"))
        else:
            punt_toetsen = (("<b>P</b> punt zetten", "<b>1&ndash;9</b> naar punt",
                             "<b>Del</b> punt weg") if self._punt_kant is not None else ())
            self.hulp.setText(keys_help(*punt_toetsen, "<b>Esc</b> sluiten"))

    def _koppel_punten(self, kant):
        """Attaches the points panel to this side (or to none: `None`)."""
        if kant is not None and not kant.punten_aan:
            kant = None
        vorige, self._punt_kant = self._punt_kant, kant
        if vorige is not None and vorige in self.kanten:
            vorige.speler.on_frame_shown = None
        aan = kant is not None
        self.paneel.setVisible(aan and (self.btn_paneel.text() == "Punten verbergen"))
        self.btn_paneel.setVisible(aan)
        # "Punten worden niet bewaard" only with one video without a row: with two videos
        # there are no points anyway, and the help line already says so.
        self.lbl_geen_punten.setVisible(len(self.kanten) == 1 and not aan)
        if not aan:
            self._punten = []
            return
        kant.speler.on_frame_shown = self._frame_shown
        self._vernieuw_punten()
        kant.balk.zet(cursor=max(0, kant.speler.huidige_idx))

    # ── Playing together (two sides, MasterClock) ─────────────────────────
    def _toetsen_afspelen(self, play):
        """Space: the clock with two videos, just the player with one. Space is
        play/pause and so **resumes wherever the videos are**; only the "Start alles"
        button first jumps to the sync points."""
        if not play:
            self._pauzeer_alles()
        elif len(self.kanten) > 1:
            self._start_alles(vanaf_sync=False)
        else:
            self.kanten[0].speler.play()

    def _start_alles(self, vanaf_sync=None):
        """`vanaf_sync`: None = what the checkbox says (the button), False = resume
        (space)."""
        if len(self.kanten) < 2:
            if self.kanten:
                self.kanten[0].speler.play()
            return
        self._pauzeer_alles()
        if vanaf_sync is None:
            vanaf_sync = self.chk_vanaf_sync.isChecked()
        if vanaf_sync:
            # Rewinding costs a seek per side here; that wait sits up front once instead
            # of in the first tick.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for kant in self.kanten:
                    kant.naar_sync()
            finally:
                QApplication.restoreOverrideCursor()
        self.klok.start([k.speler for k in self.kanten])

    def _stop_alles(self):
        self.klok.stop()      # pauses the players that were running under it itself

    def _pauzeer_alles(self):
        """Everything still: clock, scrubbing, and each player released. Idempotent."""
        self.klok.stop()
        self.toetsen.stop_scrubbing()
        for kant in self.kanten:
            kant.speler.pause()

    def _beide_naar_sync(self):
        self._pauzeer_alles()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for kant in self.kanten:
                kant.naar_sync()
        finally:
            QApplication.restoreOverrideCursor()

    def _zet_alles_snelheid(self, _idx=None):
        """Sets the shared speed on both players -- also for playing one alone, since two
        videos at a different tempo next to each other can't be compared."""
        idx = self.combo_alles_snelheid.currentIndex()
        for kant in self.kanten:
            kant.speler.combo_speed.setCurrentIndex(idx)   # herstart een lopende timer
        self.klok.recalibrate()

    # ── Points panel ─────────────────────────────────────────────────────
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
        self.tabel.cellClicked.connect(self._click_row)
        # Only the name column is editable (the flags are set per item in
        # _vernieuw_punten); _vullen suppresses itemChanged while building.
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

        self.paneel.setMinimumWidth(self.PANEL_WIDTH)
        return self.paneel

    # ── Points (saved in the library) ────────────────────────────────────
    # Every handler below operates on `_punt_kant` and does nothing if it's absent -- with
    # two videos, or with a loose video that couldn't be registered.
    @property
    def bieb(self):
        return self._punt_kant.bron.get("bieb") if self._punt_kant else None

    @property
    def bron(self):
        return self._punt_kant.bron if self._punt_kant else None

    def _vernieuw_punten(self, selectie=None):
        """Re-reads the points from the database and fills the table + bar. The database
        is the truth: that way a point is never shown that isn't saved."""
        kant = self._punt_kant
        if kant is None:
            return
        try:
            self._punten = skate_db.list_markings(self.bieb, self.bron["id"])
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

                tijd = QTableWidgetItem(_time_text(punt["frame"], kant.fps))
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

        kant.balk.zet(punten=[(p["frame"], p["label"]) for p in self._punten],
                      selectie=selectie if selectie is not None else -1)
        n = len(self._punten)
        self.lbl_punten.setText(
            "Nog geen punten gezet." if not n
            else f"{n} punt{'en' if n != 1 else ''} bewaard bij deze opname.")
        if selectie is not None and 0 <= selectie < n:
            self.tabel.selectRow(selectie)

    def _zet_punt(self):
        kant = self._punt_kant
        if kant is None:
            return
        frame = kant.speler.huidige_idx
        if frame < 0:
            return
        if any(p["frame"] == frame for p in self._punten):
            self.lbl_punten.setText("Op dit frame staat al een punt.")
            return
        try:
            skate_db.add_marking(self.bieb, self.bron["id"], frame,
                                          f"Punt {len(self._punten) + 1}",
                                          self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"Het punt kon niet bewaard worden:\n{e}")
            return
        # Re-read and only then select: the list is ordered by frame number, so a point
        # you set halfway back doesn't end up at the bottom.
        self._vernieuw_punten()
        index = next((i for i, punt in enumerate(self._punten)
                      if punt["frame"] == frame), None)
        if index is not None:
            self.tabel.selectRow(index)
            kant.balk.zet(selectie=index)

    def _verwijder_punt(self):
        if self._punt_kant is None:
            return
        rij = self.tabel.currentRow()
        if not 0 <= rij < len(self._punten):
            return
        try:
            skate_db.delete_marking(self.bieb, self._punten[rij]["id"])
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"Het punt kon niet verwijderd worden:\n{e}")
            return
        self._vernieuw_punten()

    def _punt_hernoemd(self, item):
        if self._vullen or item.column() != 2 or self._punt_kant is None:
            return
        try:
            skate_db.edit_marking(self.bieb, item.data(Qt.UserRole), label=item.text())
        except Exception as e:
            QMessageBox.warning(self, "Punt", f"De naam kon niet bewaard worden:\n{e}")

    def _ga_naar_punt(self, index):
        kant = self._punt_kant
        if kant is not None and 0 <= index < len(self._punten):
            kant.speler.go_to(self._punten[index]["frame"])
            self.tabel.selectRow(index)
            kant.balk.zet(selectie=index)

    def _click_row(self, rij, _kolom=0):
        self._ga_naar_punt(rij)

    def _click_bar(self, index):
        if index >= 0:
            self.tabel.selectRow(index)
            self._ga_naar_punt(index)

    # ── Display ──────────────────────────────────────────────────────────
    def _frame_shown(self, idx):
        if self._punt_kant is not None:
            self._punt_kant.balk.zet(cursor=idx)

    def _toggle_paneel(self):
        zichtbaar = not self.paneel.isVisible()
        self.paneel.setVisible(zichtbaar)
        self.btn_paneel.setText("Punten verbergen" if zichtbaar else "Punten tonen")

    def _toggle_volledig_scherm(self):
        toggle_fullscreen(self)

    def _minimaliseer(self):
        # Everything still first: a video running on in the taskbar decodes for nothing.
        self._pauzeer_alles()
        self.showMinimized()

    def changeEvent(self, event):
        # If the window goes from active to inactive (alt-tab, a message box in front),
        # the key-release never arrives anymore and scrubbing would run forever.
        if (event.type() == QEvent.ActivationChange and not self.isActiveWindow()
                and hasattr(self, "toetsen")):
            self.toetsen.stop_scrubbing()
        super().changeEvent(event)

    def done(self, resultaat):
        # Not closeEvent: a modal dialog that closes via accept()/reject() doesn't get one.
        # The video files must be released, or Windows keeps holding the recording.
        self.toetsen.detach()
        self.klok.stop()
        for kant in self.kanten:
            kant.release()
        super().done(resultaat)


class LocalProbe(QThread):
    """Measures in the background which recordings really sit on this pc (`bestand_lokaal`).

    On a background thread because the probe costs time precisely in the interesting
    case: a recording still in the cloud costs a network round trip per sample, so a list
    of five recordings would freeze the library for seconds. Every outcome goes to the
    table on its own, so the column fills in while you're already looking around."""

    # Keyed on the path and not the source id: the outcome is a property of the file, and
    # the list contains rows from two databases (shared + local) whose ids overlap.
    measured = Signal(str, str)          # path, status from skate_db.file_is_local

    def __init__(self, paden, parent=None):
        super().__init__(parent)
        self._paden = list(paden)

    def run(self):
        for pad in self._paden:
            if self.isInterruptionRequested():
                return
            try:
                status = skate_db.file_is_local(pad)
            except Exception:
                status = None          # an unreadable file is already flagged by the sync check
            if status:
                self.measured.emit(pad, status)


class CopyWorker(QThread):
    """Copies recordings from the camera to `opnames/` in the background
    (`skate_db.copy_to_recordings`). On a thread because a single 4 GB recording takes
    minutes and the progress bar has to keep moving meanwhile; 'Stoppen' =
    `requestInterruption`, which the copy loop reads as `stop_check` -- it then cleans up
    the partial file itself."""

    progress = Signal(int, int, int, str)     # bytes done, bytes total, idx, name
    done = Signal(list, bool, object)         # copied paths, aborted?, error or None

    def __init__(self, bieb, plan, parent=None):
        super().__init__(parent)
        self.bieb = bieb
        self.plan = plan                       # updated in the thread (reason on failure)

    def run(self):
        paden, afgebroken, fout = [], False, None
        try:
            paden = skate_db.copy_to_recordings(
                self.bieb, self.plan, progress_callback=self.progress.emit,
                stop_check=self.isInterruptionRequested)
        except skate_db.CopyAborted:
            afgebroken = True
        except Exception as e:                 # everything must reach the GUI
            fout = e
        self.done.emit(paden, afgebroken, fout)


class CopyDialog(QDialog):
    """Progress of the copying: which file, how far along, how fast, and how much longer.

    The remaining time comes from the speed over the **last `COPY_WINDOW_S`** rather than
    the average since the start: a memory card reads the first seconds from cache and a
    Drive folder writes in fits and starts, and with the overall average the estimate
    would lag behind there for minutes. The display is roughly rounded (`_remaining_text`)
    so it doesn't jump back and forth on every tick. The window only closes once the
    thread is done -- closing via ✕ or Esc is 'Stoppen', since destroying a running
    QThread is a crash."""

    def __init__(self, worker, aantal, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Opnames naar de bibliotheek kopiëren")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(520)
        self._worker = worker
        self._aantal = aantal
        self._monsters = []                   # (time, bytes) of the last few seconds
        self.paden, self.afgebroken, self.fout = [], False, None   # set by _on_done

        v = QVBoxLayout(self)
        self.lbl_bestand = QLabel("Voorbereiden...")
        self.lbl_bestand.setWordWrap(True)
        v.addWidget(self.lbl_bestand)
        self.balk = QProgressBar()
        self.balk.setRange(0, 1000)
        self.balk.setValue(0)
        self.balk.setTextVisible(True)
        v.addWidget(self.balk)
        self.lbl_detail = QLabel("")
        v.addWidget(self.lbl_detail)
        rij = QHBoxLayout()
        rij.addStretch(1)
        self.btn_stop = QPushButton("Stoppen")
        self.btn_stop.setToolTip("Breekt het kopiëren af. Wat al helemaal gekopieerd is "
                                 "blijft staan; het half gekopieerde bestand wordt weggehaald.")
        self.btn_stop.clicked.connect(self._stop)
        rij.addWidget(self.btn_stop)
        v.addLayout(rij)

        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_done)

    def _on_progress(self, gedaan, totaal, idx, naam):
        nu = time.monotonic()
        self._monsters.append((nu, gedaan))
        while len(self._monsters) > 1 and nu - self._monsters[0][0] > COPY_WINDOW_S:
            self._monsters.pop(0)
        self.lbl_bestand.setText(f"Bestand {idx + 1} van {self._aantal}: <b>{naam}</b>")
        self.balk.setValue(int(gedaan / max(1, totaal) * 1000))
        delen = [f"{_bytes_text(gedaan)} van {_bytes_text(totaal)}"]
        t0, b0 = self._monsters[0]
        if nu - t0 >= 1.5:
            snelheid = (gedaan - b0) / (nu - t0)
            if snelheid > 0:
                delen.append(f"{_bytes_text(snelheid)}/s")
                delen.append(_remaining_text((totaal - gedaan) / snelheid))
        else:
            delen.append("snelheid meten...")
        self.lbl_detail.setText(" · ".join(delen))

    def _stop(self):
        self.btn_stop.setEnabled(False)
        self.lbl_detail.setText("Stoppen... (het lopende blok wordt afgemaakt)")
        self._worker.requestInterruption()

    def _on_done(self, paden, afgebroken, fout):
        # Save the outcome here rather than fetching it via a second connection on
        # `done`: accept() ends exec(), and a second signal already queued up isn't
        # delivered yet by the time the caller has already moved on.
        self.paden, self.afgebroken, self.fout = list(paden), afgebroken, fout
        self.accept()

    def reject(self):
        # ✕ or Esc: don't close while the thread is running -- stop it cleanly first.
        if self._worker.isRunning():
            self._stop()
        else:
            super().reject()


COPY_WINDOW_S = 8.0     # speed for the remaining time: over the last 8 s


def _bytes_text(n):
    """4,3 GB / 820 MB / 12 kB -- with a comma, like the rest of the app."""
    n = float(n)
    for eenheid in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or eenheid == "TB":
            break
        n /= 1000.0
    if eenheid == "B":
        return f"{int(n)} B"
    tekst = f"{n:.1f}" if n < 10 else f"{n:.0f}"
    return f"{tekst.replace('.', ',')} {eenheid}"


def _remaining_text(seconden):
    """"nog ongeveer 3 min" -- roughly rounded, so the estimate doesn't jump on every tick."""
    if seconden < 10:
        return "bijna klaar"
    if seconden < 60:
        return f"nog ongeveer {int(round(seconden / 5.0) * 5)} s"
    minuten = int(math.ceil(seconden / 60.0))
    if minuten < 60:
        return f"nog ongeveer {minuten} min"
    uren, rest = divmod(minuten, 60)
    return f"nog ongeveer {uren} u {rest} min" if rest else f"nog ongeveer {uren} u"


class MainWindow(QMainWindow):
    def __init__(self, melding=None):
        super().__init__()
        # Statusregel van het opstartscherm (of None): de opbouw hieronder duurt op een
        # koude machine een paar seconden en dat mag te zien zijn.
        self._melding = melding or (lambda tekst: None)
        self.setWindowTitle("Schaats Analyse")
        set_window_size(self, 1400, 820, maximize=True)

        self.input_pad = None
        self.model_pad = DEFAULT_MODEL
        self.smooth_n = 5
        self.threshold = 0.015
        self.doel_punt = None
        self.doel_kader = None       # getekend kader → kijkglas in de YOLO-backend
        self.horizon_deg = 0.0
        self.auto_horizon = False
        self.perspectief = None
        self.geen_smoothing = False
        self.bocht_overslaan = True   # bochtframes niet analyseren/meten (checkbox in de dialoog)
        self.deinterlacen = False     # kamfilter voor interlaced bron (per video vastgesteld)
        # video_info / resultaten / huidige_idx wonen in self.speler (zie de properties
        # hieronder); die wordt in _bouw_ui() aangemaakt en niets vóór die aanroep leest ze.
        self.events = []
        self.worker = None
        self.batch_worker = None
        self.bieb = None            # bibliotheekpad (gezet door _zet_bibliotheek)
        self.lokaal = None          # de lokale bibliotheek (losse video's), zie _zet_bibliotheek
        self._opnames = []          # fase 8: bronvideo-rijen achter de opnametabel
        self._lokaal = {}           # pad -> 'lokaal'/'deels'/'cloud' (snelheidsproef)
        self._lokaal_proef = None   # lopende LocalProbe-thread
        self._knip_tmpmap = None    # tijdelijke map met zojuist geknipte fragmenten
        self.knip_worker = None
        self.trainer_naam = skate_db.trainer_name()  # fase 4: gaat mee als aangemaakt_door
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
                                            # _warn_backend_fallback)

        # Skelet-editor (fase 3) — de zoom/pan-state zit in de VideoPlayer
        self._editor_actief = False
        self._sleep = None          # {'idx', 'j', 'start_lm': Landmark} tijdens een sleep
        # Undo-items zijn getypeerd: 'sleep' verplaatst één landmark over een uitvloei-venster,
        # 'skelet' zet een compleet handmatig geplaatst skelet neer (of weer weg).
        self._undo = []
        self._redo = []
        self._handmatig = {}        # {frame_idx: set(landmark_idx)} — alleen voor de overlay-markering
        self._plaats = None         # lopende plaats-reeks, zie _start_plaatsen

        # De toetsafhandeling van de twee pagina's met beeld (gevuld in _bouw_ui). Als lijst,
        # want `changeEvent` kan door Qt aangeroepen worden vóórdat het venster er staat.
        self._toetsen = []

        # Vergelijkpagina: één masterklok voor "Start alles" (zie MasterClock). De factor
        # komt uit de gedeelde snelheidscombo, die pas in _bouw_ui ontstaat — de callable
        # wordt pas bij het starten gelezen.
        self.klok = MasterClock(
            self, factor=lambda: self.combo_alles_snelheid.currentData() or 1.0,
            on_done=self._stop_alles)

        self._melding("Venster opbouwen...")
        self._bouw_ui()
        self._melding("Bibliotheek openen...")
        self._zet_bibliotheek(skate_db.library_path())

    # De VideoPlayer is de enige eigenaar van deze drie; hier alleen doorkijkjes, zodat de
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

        # Drie permanente statusbalk-widgets: hoeveel frames een skelet hebben (dekking), of
        # er gespoeld wordt, en de live-status van het huidige frame. Permanent, want
        # showMessage() overschrijft de gewone statusbalk-tekst en de dekking moet altijd
        # afleesbaar blijven.
        self.lbl_dekking = QLabel("")
        self.lbl_dekking.setToolTip(
            "Aantal frames met een skelet (gedetecteerd of handmatig geplaatst).\n"
            "Frames zonder skelet breken een afzetmeting af — met '✏ Bewerken' zijn ze "
            "handmatig aan te vullen.")
        self.statusBar().addPermanentWidget(self.lbl_dekking)
        self.lbl_spoel = QLabel("")
        self.lbl_spoel.setStyleSheet("color: #5aaaf0; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_spoel)
        self.lbl_live = QLabel("")
        self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_live)
        self.statusBar().showMessage(
            f"Kies een schaatser en start of open een analyse.  ·  backend: {BACKEND_NAME}")

        # Dezelfde toetsen als in het kijk- en knipvenster, hier voor de twee pagina's met
        # beeld. Twee losse objecten en niet één met een pagina-tak halverwege de
        # afhandeling: op de vergelijkpagina zijn er twee spelers tegelijk te sturen en
        # bedient spatie de masterklok (anders lopen de kanten binnen seconden uit de pas),
        # en dat verschil hoort in de opbouw te staan.
        #
        # Beide moeten bestaan vóór de `currentChanged`-haak hieronder, want die loopt via
        # `_pauzeer_alles` en dat stopt ook een lopende spoelactie.
        self.toetsen_analyse = PlayerKeys(
            self, lambda: [self.speler], on_scrub=self.lbl_spoel.setText,
            active=lambda: self.stack.currentWidget() is self.pagina_analyse)
        self.toetsen_vergelijk = PlayerKeys(
            self,
            lambda: [k.speler for k in (self.kant_links, self.kant_rechts)
                     if k.heeft_analyse()],
            on_scrub=self.lbl_spoel.setText,
            on_play=self._toetsen_vergelijk_afspelen,
            is_playing=lambda: (self.klok.is_running()
                            or self.kant_links.speler.is_playing()
                            or self.kant_rechts.speler.is_playing()),
            active=lambda: self.stack.currentWidget() is self.pagina_vergelijk)
        self._toetsen = [self.toetsen_analyse, self.toetsen_vergelijk]

        # Eén haak i.p.v. bij elke setCurrentWidget-aanroep: een verlaten pagina mag niet
        # doordecoderen op de achtergrond.
        self.stack.currentChanged.connect(self._paginawissel)
        self._alleen_zichtbare_pagina_telt()

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

    def _toetsen_vergelijk_afspelen(self, play):
        """Spatie op de vergelijkpagina bedient de masterklok, niet de twee spelers los —
        twee losse frame-timers lopen binnen seconden uit de pas (zie `_start_alles`).
        Spatie is afspelen/pauze en hervat dus waar de video's staan; alleen de knop
        "Start alles" springt eerst naar de sync-punten."""
        if play:
            self._start_alles(vanaf_sync=False)
        else:
            self._pauzeer_alles()

    def _pauzeer_alles(self):
        """Stopt elke lopende weergave (analysepagina én beide vergelijk-spelers), inclusief
        een lopende spoelactie. Idempotent, dus veilig om overal aan te roepen."""
        self._stop_alles()
        for toetsen in self._toetsen:
            toetsen.stop_scrubbing()
        self.speler.pause()
        for kant in (self.kant_links, self.kant_rechts):
            kant.speler.pause()

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
        self._alleen_zichtbare_pagina_telt()

    def _alleen_zichtbare_pagina_telt(self):
        """Laat alleen de getoonde pagina meetellen in het venster-minimum.

        Een QStackedLayout neemt het maximum over álle pagina's, ook de verborgen: de
        startpagina is 318 px hoog, maar het hoofdvenster eiste 643 omdat de (verborgen)
        vergelijkpagina 598 nodig heeft — en na één keer vergelijken met twee lange namen
        bleef het venster 1489 px breed op een 1280 px-scherm, óók terug op de startpagina
        (gemeten 12-9-2026). Een verborgen pagina op `Ignored` eist niets; bij het tonen
        krijgt hij zijn eigen beleid terug. Het venster kan daardoor bij een paginawissel
        gróeien (naar het minimum van de nieuwe pagina), maar niet meer voor een pagina die
        niemand ziet."""
        huidig = self.stack.currentWidget()
        for i in range(self.stack.count()):
            pagina = self.stack.widget(i)
            if pagina is huidig:
                pagina.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            else:
                pagina.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.stack.layout().invalidate()

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
        # Afkorten in het midden: een lang Drive-pad maakte de startpagina anders 1039 px
        # breed (de mapnaam aan het eind is het informatieve deel, dus die blijft staan).
        self.lbl_bieb = ElideLabel("", Qt.ElideMiddle)
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
            "analyse; selecteer <b>twee</b> rijen (Ctrl+klik) om ze naast elkaar te zien."
            "<br>Wil je een video bekijken die hier niet in staat, waar hij ook op "
            "deze pc staat? Gebruik <b>Nieuwe video bekijken</b>; hij komt daarna als "
            "<i>losse video</i> onderaan deze lijst — alleen op deze pc, niet bij collega's. "
            "Met <b>Van camera naar bibliotheek</b> kopieer je hele opnames van de camera "
            "hierheen, mét voortgangsbalk.")
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
        # Twee rijen mogen tegelijk geselecteerd zijn: 'Bekijken' zet ze dan naast elkaar.
        # Knippen, status en notitie blijven op de huidige rij werken.
        self.tabel_opnames.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Dubbelklik = knippen, behalve op de notitie: daar is dubbelklik al 'bewerken'
        # en hoort niet ook nog het knipvenster open te gaan.
        self.tabel_opnames.cellDoubleClicked.connect(self._opname_dubbelklik)
        # De notitie is ter plekke te bewerken; alleen die kolom is editeerbaar (zie
        # _vul_opnames, dat de vlaggen per item zet).
        self.tabel_opnames.itemChanged.connect(self._opname_notitie_gewijzigd)
        self._vullen_opnames = False   # onderdrukt itemChanged tijdens het opbouwen
        v.addWidget(self.tabel_opnames, stretch=1)

        # Een afbrekende balk: vijf knoppen op één regel eisen ~900 px en tilden de
        # startpagina van 700 naar 897 px minimumbreedte (gemeten met schaats_schermtest);
        # afgebroken kost een knop erbij hoogstens een regel.
        rij = WrapBar()
        self.btn_bekijken = QPushButton("👁 Bekijken (volledig scherm)...")
        self.btn_bekijken.setToolTip(
            "Speelt de gekozen opname af zoals hij uit de camera komt: geen detectie, geen\n"
            "tracking, alleen beeld. Vertragen, inzoomen, met . en , op 6× door- en\n"
            "terugspoelen, en punten zetten die bewaard blijven.\n"
            "Twee rijen geselecteerd (Ctrl+klik)? Dan komen ze naast elkaar, met een\n"
            "sync-punt per video en 'Start alles' om ze synchroon af te spelen.")
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
        self.btn_losse_video = QPushButton("🎬 Nieuwe video bekijken...")
        self.btn_losse_video.setToolTip(
            "Bekijk een video die ergens anders op deze pc staat — net van de camera\n"
            "gehaald, van een collega gekregen — zonder hem eerst in de bibliotheek te\n"
            "zetten. Hetzelfde kijkvenster: volledig scherm, vertragen, inzoomen, spoelen\n"
            "en punten die bewaard blijven — en in het venster kun je er een tweede video\n"
            "naast zetten. Er wordt niets geanalyseerd en niets gekopieerd;\n"
            "de video komt als 'losse video' onderaan de lijst hierboven, zodat je hem de\n"
            "volgende keer daar terugvindt. Dat wordt alleen op deze pc onthouden — het pad\n"
            "komt niet in de gedeelde bibliotheek en collega's zien hem niet.")
        self.btn_losse_video.clicked.connect(self._bekijk_losse_video)
        rij.addWidget(self.btn_losse_video)
        self.btn_importeer = QPushButton("📥 Van camera naar bibliotheek...")
        self.btn_importeer.setToolTip(
            "Kopieert hele opnames van de camera of geheugenkaart naar de map 'opnames' in\n"
            "de bibliotheek, met een voortgangsbalk en de resterende tijd — zodat dat niet\n"
            "buiten de app in de Verkenner hoeft. Ze staan daarna in deze lijst en Google\n"
            "Drive zet ze vanzelf op de gedeelde schijf.\n"
            "Op een camcorder staan de opnames meestal in PRIVATE\\AVCHD\\BDMV\\STREAM\n"
            "(bestanden als 00005.MTS). Een bestand dat er al staat wordt nooit overschreven.")
        self.btn_importeer.clicked.connect(self._importeer_van_camera)
        rij.addWidget(self.btn_importeer)
        v.addWidget(rij)
        return paneel

    def _vernieuw_opnames(self, selecteer=None):
        """Scant `opnames/` en vult de tabel. Schrijft alleen als er echt nieuwe bestanden
        zijn (synchroniseer_bronmap), zodat de gedeelde DB niet bij elke app-start van elke
        trainer wordt aangeraakt.

        Losse video's van deze pc ("Nieuwe video bekijken") komen eronder, uit de **lokale**
        bibliotheek (`_lokale_opnames`), gemarkeerd en met het volledige pad in de tooltip:
        het bestand staat buiten de bibliotheek, dus de naam alleen zegt niet waar hij is.
        De rij-identiteit is `_opname_sleutel` (bibliotheek + id), want de id's van de
        twee databases overlappen. `selecteer` = de sleutel die na het vullen geselecteerd
        moet zijn (de zojuist geopende video), anders blijft de selectie waar hij was."""
        try:
            skate_db.sync_source_dir(self.bieb)
            opnames = skate_db.list_source_videos(self.bieb)
        except Exception as e:
            self.tabel_opnames.setRowCount(0)
            self.statusBar().showMessage(f"Opnames konden niet gelezen worden: {e}", 6000)
            return
        opnames += self._lokale_opnames()
        self._opnames = opnames

        self._vullen_opnames = True
        try:
            self.tabel_opnames.setRowCount(len(opnames))
            for rij, b in enumerate(opnames):
                ontbreekt = b["sync"] == "ontbreekt"
                naam = QTableWidgetItem(
                    b["naam"] + ("  (losse video)" if b["extern"] else "")
                    + ("  (bestand niet gevonden)" if ontbreekt else ""))
                naam.setData(Qt.UserRole, _opname_sleutel(b))
                naam.setFlags(naam.flags() & ~Qt.ItemIsEditable)
                uitleg = []
                if b["extern"]:
                    uitleg.append(
                        f"Losse video, geopend via 'Nieuwe video bekijken':\n{b['pad']}\n"
                        "Alleen op deze pc onthouden (met de punten) — niet in de gedeelde "
                        "bibliotheek, dus collega's zien hem niet. Verdwijnt uit de lijst "
                        "zolang het bestand er niet staat.")
                if b["bijgewerkt_door"]:
                    uitleg.append(f"Status gezet door {b['bijgewerkt_door']}")
                if ontbreekt:
                    naam.setForeground(QColor(150, 150, 150))
                elif b["sync"] == "onvolledig":
                    uitleg.append("De cloudsync is dit bestand nog aan het downloaden.")
                if uitleg:
                    naam.setToolTip("\n\n".join(uitleg))
                self.tabel_opnames.setItem(rij, 0, naam)

                duur = QTableWidgetItem(
                    _time_text(b["totaal_frames"], b["fps"])
                    if (b["totaal_frames"] and b["fps"]) else "—")
                duur.setFlags(duur.flags() & ~Qt.ItemIsEditable)
                self.tabel_opnames.setItem(rij, 1, duur)

                # De status zet je zélf: er wordt nooit automatisch iets op 'klaar' gezet,
                # want het programma kan niet weten of jij een opname af vindt.
                combo = QComboBox()
                combo.addItems(skate_db.SOURCE_STATUSES)
                idx = combo.findText(b["status"])
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.currentTextChanged.connect(
                    lambda tekst, bron=b: self._zet_opname_status(bron, tekst))
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

                self._zet_lokaal_cel(rij, self._lokaal.get(b["pad"]))
        finally:
            self._vullen_opnames = False
        gevraagd = next((rij for rij, b in enumerate(opnames)
                         if selecteer is not None and _opname_sleutel(b) == selecteer), -1)
        if gevraagd >= 0:
            self.tabel_opnames.selectRow(gevraagd)
            self.tabel_opnames.scrollToItem(self.tabel_opnames.item(gevraagd, 0))
        elif opnames and self.tabel_opnames.currentRow() < 0:
            self.tabel_opnames.selectRow(0)   # 'Knippen...' werkt dan meteen
        self.tabs_bieb.setTabText(1, f"Opnames ({len(opnames)})" if opnames else "Opnames")
        self._start_lokaal_proef(opnames)

    def _lokale_opnames(self):
        """De losse video's van deze pc uit de lokale bibliotheek, voor onder de werklijst.

        Eerst de opruiming (`verhuis_losse_videos`): wat een oudere versie van de app nog
        met een absoluut pad in de gedeelde database zette, gaat mét punten naar de lokale —
        de gebruiker wil die paden niet in de Drive hebben, en een collega zag er alleen
        "bestand niet gevonden" van. Een losse video waarvan het bestand er (nu) niet is
        wordt **niet getoond** maar ook niet gewist: een USB-stick die even niet in zit of
        een hernoemd bestand mag de punten niet kosten, en zo'n rij in de lijst laten staan
        is precies waar niemand op zit te wachten. Faalt het lezen, dan alleen de werklijst —
        de gedeelde bibliotheek mag niet stranden op de lokale."""
        if self.lokaal is None:
            return []
        try:
            skate_db.migrate_loose_videos(self.bieb, self.lokaal)
            return [b for b in skate_db.list_source_videos(self.lokaal, extern=True)
                    if b["sync"] != "ontbreekt"]
        except Exception as e:
            self.statusBar().showMessage(f"Losse video's konden niet gelezen worden: {e}",
                                         6000)
            return []

    def _geselecteerde_opname(self):
        rij = self.tabel_opnames.currentRow()
        if rij < 0:
            return None
        item = self.tabel_opnames.item(rij, 0)
        return self._opname_bij_sleutel(item.data(Qt.UserRole) if item else None)

    def _geselecteerde_opnames(self):
        """Alle geselecteerde rijen (op rijvolgorde) als bron-dicts — voor 'Bekijken', dat
        er twee naast elkaar kan zetten."""
        rijen = sorted({idx.row() for idx in self.tabel_opnames.selectedIndexes()})
        bronnen = []
        for rij in rijen:
            item = self.tabel_opnames.item(rij, 0)
            bron = self._opname_bij_sleutel(item.data(Qt.UserRole) if item else None)
            if bron is not None:
                bronnen.append(bron)
        return bronnen

    def _opname_bij_sleutel(self, sleutel):
        """De bron-dict achter een tabelrij (`Qt.UserRole` van de naamcel), of None."""
        if sleutel is None:
            return None
        return next((b for b in getattr(self, "_opnames", [])
                     if _opname_sleutel(b) == tuple(sleutel)), None)

    def _start_lokaal_proef(self, opnames):
        """Laat op de achtergrond meten welke opnames op deze pc staan.

        Alleen voor bestanden die er zijn: bij 'ontbreekt' zegt de sync-check het al. Een
        lopende meting wordt afgebroken — na een verversing kunnen de rijen anders zijn, en
        een uitkomst van een oude lijst hoort niet meer in de tabel."""
        self._stop_lokaal_proef()
        paden = [b["pad"] for b in opnames if b["sync"] != "ontbreekt"]
        if not paden:
            return
        self._lokaal_proef = LocalProbe(paden, self)
        self._lokaal_proef.measured.connect(self._lokaal_gemeten)
        self._lokaal_proef.start()

    def _stop_lokaal_proef(self):
        """Breekt een lopende meting af. `wait` mag hier: de thread controleert de vlag
        tussen twee opnames door en één monster duurt hooguit een seconde."""
        proef = self._lokaal_proef
        self._lokaal_proef = None
        if proef is not None and proef.isRunning():
            proef.requestInterruption()
            proef.wait(3000)

    def _lokaal_gemeten(self, pad, status):
        """Eén uitkomst binnen: onthouden en de cel bijwerken (de rij kan intussen weg zijn)."""
        self._lokaal[pad] = status
        for rij, b in enumerate(self._opnames):
            if b["pad"] == pad:
                b["lokaal"] = status
                if rij < self.tabel_opnames.rowCount():
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

    def _zet_opname_status(self, bron, status):
        if self._vullen_opnames:
            return
        try:
            skate_db.edit_source_video(bron["bieb"], bron["id"], status=status,
                                        bijgewerkt_door=self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Opname", f"Status opslaan mislukte:\n{e}")
            return
        bron["status"] = status
        self.statusBar().showMessage(f"Status → {status}", 3000)

    def _opname_dubbelklik(self, rij, kolom):
        if kolom != OPNAME_KOL_NOTITIE:
            self._knip_opname()

    def _opname_notitie_gewijzigd(self, item):
        if self._vullen_opnames or item.column() != OPNAME_KOL_NOTITIE:
            return
        naam_item = self.tabel_opnames.item(item.row(), 0)
        bron = self._opname_bij_sleutel(naam_item.data(Qt.UserRole)) if naam_item else None
        if bron is None:
            return
        try:
            skate_db.edit_source_video(bron["bieb"], bron["id"], notitie=item.text(),
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
        if not self._waarschuw_niet_lokaal(bron):
            return False
        # Nu vaststellen, zodat zowel het kijkvenster als het knipvenster het antwoord al
        # heeft en er niet halverwege een meting van een paar seconden tussen valt.
        self._bron_interlaced(bron)
        return True

    def _bron_interlaced(self, bron):
        """Is deze opname interlaced (kamtanden)? Eén keer meten per opname en het antwoord
        in de bibliotheek bewaren: het kost een paar seconden — op een streaming Drive meer —
        terwijl het antwoord nooit verandert. Lukt het meten niet, dan niet filteren: liever
        het ruwe beeld dan pixels aanraken op grond van een mislukte meting."""
        if bron.get("interlaced") is not None:
            return bool(bron["interlaced"])
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            uitkomst = is_interlaced(bron["pad"])
        except Exception:
            uitkomst = False
        finally:
            QApplication.restoreOverrideCursor()
        try:
            if bron.get("id") is not None:
                skate_db.set_source_interlaced(bron["bieb"], bron["id"], uitkomst)
        except Exception:
            pass                      # meten lukte; alleen het onthouden niet
        bron["interlaced"] = 1 if uitkomst else 0
        return uitkomst

    def _waarschuw_niet_lokaal(self, bron):
        """Waarschuwt als de opname niet offline op deze pc staat; True = doorgaan.

        Het bestand is er wél — de cloudmap laat hem gewoon zien — maar het beeld komt
        er per stukje overheen. Gemeten op deze bibliotheek (24 aug 2026, Google Drive
        in streaming-stand): één sprong in het knipvenster haalde ~40 MB op en kostte
        5 tot 20 s, tegen 30-120 ms als dezelfde opname lokaal staat. Dat valt niet met
        code te verhelpen — de speler springt al gericht i.p.v. door te spoelen (zie
        SEEK_THRESHOLD_FRAMES), en die 40 MB is wat ffmpeg nodig heeft om in een MPEG-TS
        zonder index het juiste tijdstip te vinden. Het enige zinnige is het zéggen,
        vóórdat iemand denkt dat het programma hangt.

        Doorgaan mag: soms wil je alleen even het begin zien, en wat je al bekeken hebt
        zit in de cloudcache en is daarna wél meteen terug."""
        # De cache is op pad gesleuteld, dus ook een losse video die niet geregistreerd kon
        # worden (geen id) heeft er gewoon zijn eigen plek in.
        status = self._lokaal.get(bron["pad"])
        if status is None:                 # de achtergrondmeting was nog niet zover
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                status = skate_db.file_is_local(bron["pad"])
            except Exception:
                status = None
            finally:
                QApplication.restoreOverrideCursor()
            if status:
                self._lokaal[bron["pad"]] = status
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
        """Een opname handmatig doorkijken: alleen beeld, geen analyse. Twee geselecteerde
        rijen komen naast elkaar.

        Bewust zónder de controles die het knippen wél doet (draait er een analyse, bestaat
        er al een schaatser): er wordt niets gemeten en er komt niets in de bibliotheek
        terecht behalve de punten, en die horen bij de opname zelf."""
        bronnen = self._geselecteerde_opnames()
        if not bronnen:
            QMessageBox.information(
                self, "Opname bekijken",
                "Kies eerst een opname in de lijst.\n\n"
                "Staat er niets? Zet je opnames in de map 'opnames' in de bibliotheek en "
                "druk op 'Vernieuwen'. Wil je een video bekijken die daar niet in staat, "
                "gebruik dan 'Nieuwe video bekijken...'.")
            return
        if len(bronnen) > 2:
            QMessageBox.information(
                self, "Opname bekijken",
                "Kies één opname, of twee om ze naast elkaar te bekijken.")
            return
        self._open_bekijkvenster(bronnen, "Opname")
        self._vernieuw_opnames()      # de puntentelling in de lijst bijwerken

    def _open_bekijkvenster(self, bronnen, titel):
        """Het gedeelde stuk van `_bekijk_opname` en `_bekijk_losse_video`: beschikbaarheid
        controleren, de video('s) openen en het kijkvenster tonen. `titel` is de kop van
        een eventuele foutmelding. Faalt er één van twee, dan gaat het venster niet open —
        beter dan stilzwijgend één video tonen waar er twee gevraagd zijn."""
        paren = []
        for bron in bronnen:
            paar = self._open_bron(bron, titel)
            if paar is None:
                return
            paren.append(paar)
        self._losse_toegevoegd = False
        dlg = ViewWindow(paren, self.trainer_naam,
                            kies_tweede=self._kies_tweede_video, parent=self)
        show_dialog(dlg)
        if self._losse_toegevoegd:
            self._vernieuw_opnames()  # de video die in het venster is toegevoegd, in de lijst

    def _open_bron(self, bron, titel):
        """Beschikbaarheidscheck + `video_info` voor één bron → `(bron, info)` of None
        (melding is dan al getoond). De punten gaan naar de bibliotheek waar de rij in
        staat: gedeeld voor een opname, lokaal voor een losse video (`bron["bieb"]`)."""
        if not self._opname_beschikbaar(bron):
            return None
        try:
            return bron, video_info(bron["pad"])
        except Exception as e:
            QMessageBox.critical(self, titel, f"Kan de video niet openen:\n{e}")
            return None

    def _kies_losse_video(self):
        """Bestandskiezer voor een video ergens op deze pc + registratie als losse video
        (zie `_bekijk_losse_video`). Geeft de bron-dict, of None bij annuleren."""
        cfg = skate_db.load_config()
        pad, _ = QFileDialog.getOpenFileName(
            self, "Kies een video om te bekijken", cfg.get("laatste_videomap", ""),
            VIDEO_FILTER)
        if not pad:
            return None
        cfg["laatste_videomap"] = os.path.dirname(pad)
        try:
            skate_db.save_config(cfg)
        except Exception:
            pass                      # de map onthouden is comfort, geen voorwaarde

        try:
            if self.lokaal is None:
                raise RuntimeError("de lokale bibliotheek kon bij het opstarten niet "
                                   "geopend worden (zie het logboek)")
            bron = skate_db.loose_video(self.bieb, self.lokaal, pad)
        except Exception as e:
            QMessageBox.warning(
                self, "Video bekijken",
                f"De video kan bekeken worden, maar wordt niet onthouden en punten kunnen "
                f"nu niet bewaard worden:\n{e}")
            bron = {"id": None, "bieb": None, "naam": os.path.basename(pad), "pad": pad,
                    "sync": None, "interlaced": None}
        return bron

    def _kies_tweede_video(self):
        """Voor "➕ Tweede video ernaast..." in het kijkvenster: dezelfde route als een
        losse video (kiezer, registratie, beschikbaarheid), maar het venster staat al open.
        Geeft `(bron, info)` of None; onthoudt dat de lijst straks ververst moet worden."""
        bron = self._kies_losse_video()
        if bron is None:
            return None
        if bron["id"] is not None:
            self._losse_toegevoegd = True    # de rij bestaat al, ook als het openen faalt
        return self._open_bron(bron, "Video bekijken")

    def _bekijk_losse_video(self):
        """Een video ergens anders op deze pc bekijken, zonder hem naar de bibliotheek te
        kopiëren.

        Voor "alleen kijken" is er niets uit de bibliotheek nodig — geen schaatser, geen
        analyse, geen kopie — maar je kwam er tot nu toe alleen in via een rij in de
        opnamelijst, en die bestaat per definitie uit bestanden in `opnames/`. Een clip die
        net van de camera komt of van een collega, was dus niet te bekijken.

        De video krijgt wél een rij, maar in de **lokale** bibliotheek (`skate_db.
        losse_video` → `lokale_bibliotheek`): het pad is van deze pc en hoort niet in de
        gedeelde Drive, terwijl de punten die je zet wél bewaard moeten blijven én de video
        de volgende keer gewoon in de lijst moet staan i.p.v. opnieuw via de bestandskiezer
        opgezocht te worden. Daarom wordt de lijst ná het kijken ververst met deze rij
        geselecteerd — ook als het venster uiteindelijk niet openging (cloudwaarschuwing
        geweigerd), want de rij is er dan al. Lukt het registreren niet (lokale bibliotheek
        niet te openen), dan gaat het kijken gewoon door zonder punten — daar hoort de
        database niet tussen te komen."""
        bron = self._kies_losse_video()
        if bron is None:
            return
        try:
            self._open_bekijkvenster([bron], "Video bekijken")
        finally:
            if bron["id"] is not None:
                self._vernieuw_opnames(selecteer=_opname_sleutel(bron))

    def _importeer_van_camera(self):
        """Hele opnames van de camera/geheugenkaart naar `opnames/` kopiëren, ín de app.

        Tot nu toe moest dat in de Verkenner, terwijl alles wat erna komt (scannen,
        knippen, bekijken) hier zit; en een kopie van 4 GB naar een Drive-map is precies
        het soort wachten waar je een balk met resterende tijd bij wilt. De route:
        `kopieer_plan` beslist vooraf wat er wél en niet gaat (bestaande bestanden worden
        nooit overschreven — daar hangen fragmenten en punten aan, zie skate_db), een
        ruimtecheck, dan `CopyWorker` + `CopyDialog`, en ná afloop de gewone
        `_vernieuw_opnames`, want vanaf dat moment is het een opname als elke andere:
        de scan registreert hem, Drive uploadt hem, collega's zien hem verschijnen."""
        cfg = skate_db.load_config()
        paden, _ = QFileDialog.getOpenFileNames(
            self, "Kies de opnames op de camera of geheugenkaart",
            cfg.get("laatste_cameramap", ""), VIDEO_FILTER)
        if not paden:
            return
        cfg["laatste_cameramap"] = os.path.dirname(paden[0])
        try:
            skate_db.save_config(cfg)
        except Exception:
            pass                      # de map onthouden is comfort, geen voorwaarde

        try:
            plan = skate_db.copy_plan(self.bieb, paden)
        except Exception as e:
            QMessageBox.critical(self, "Kopiëren", f"Kan de map 'opnames' niet bereiken:\n{e}")
            return
        te_doen = [i for i in plan if i["reden"] is None]
        overgeslagen = [i for i in plan if i["reden"] is not None]
        if not te_doen:
            QMessageBox.information(
                self, "Kopiëren",
                "Er valt niets te kopiëren:\n\n" + self._kopieer_redenen(overgeslagen))
            return
        totaal = sum(i["bytes"] for i in te_doen)

        # Ruimte: op een Drive-map is de vrije ruimte die van de lokale cache/schijf, en
        # een kopie die op 90% strandt kost een kwartier voor niets.
        try:
            vrij = shutil.disk_usage(skate_db.recordings_path(self.bieb)).free
        except OSError:
            vrij = None
        if vrij is not None and totaal > vrij:
            QMessageBox.warning(
                self, "Te weinig ruimte",
                f"Deze opnames zijn samen {_bytes_text(totaal)}, maar op de schijf van de "
                f"bibliotheek is nog {_bytes_text(vrij)} vrij.\n\nMaak ruimte (of kies "
                f"minder opnames) en probeer het opnieuw.")
            return
        if overgeslagen:
            antwoord = QMessageBox.question(
                self, "Kopiëren",
                f"{len(te_doen)} van de {len(plan)} gekozen bestanden worden gekopieerd "
                f"({_bytes_text(totaal)}). De rest niet:\n\n"
                + self._kopieer_redenen(overgeslagen) + "\n\nDoorgaan?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if antwoord != QMessageBox.Yes:
                return

        worker = CopyWorker(self.bieb, plan, self)
        dlg = CopyDialog(worker, len(te_doen), self)
        worker.start()
        try:
            show_dialog(dlg)
            worker.wait()             # accept() komt uit klaar(), dus dit is meteen klaar
        finally:
            worker.deleteLater()

        gekopieerd = dlg.paden
        mislukt = [i for i in te_doen if i["reden"] is not None]   # reden gezet door de worker
        sleutel = None
        if gekopieerd:
            try:
                skate_db.sync_source_dir(self.bieb)
                sleutel = _opname_sleutel(skate_db.source_video_for_path(self.bieb, gekopieerd[0]))
            except Exception:
                sleutel = None
        self._vernieuw_opnames(selecteer=sleutel)

        regels = []
        if gekopieerd:
            regels.append(f"{len(gekopieerd)} opname(s) gekopieerd naar de bibliotheek "
                          f"({_bytes_text(sum(os.path.getsize(p) for p in gekopieerd))}). "
                          f"Ze staan nu in de lijst; staat de bibliotheek in Google Drive, dan "
                          f"uploadt Drive ze vanzelf en zien collega's ze daarna ook.")
        if dlg.afgebroken:
            regels.append("Het kopiëren is gestopt; het half gekopieerde bestand is weggehaald.")
        if dlg.fout is not None:
            regels.append(f"Het kopiëren is onverwacht gestopt:\n{dlg.fout}")
        if mislukt:
            regels.append("Niet gelukt:\n" + self._kopieer_redenen(mislukt))
        if not regels:
            regels.append("Er is niets gekopieerd.")
        (QMessageBox.warning if (mislukt or dlg.fout is not None)
         else QMessageBox.information)(self, "Kopiëren", "\n\n".join(regels))

    @staticmethod
    def _kopieer_redenen(items):
        return "\n".join(f"• {i['naam']} — {i['reden']}" for i in items)

    def _open_opnamesmap(self):
        pad = skate_db.recordings_path(self.bieb)
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
        self.kant_links = CompareSide(
            "Links", lambda: self._kies_vergelijk_kant(self.kant_links))
        self.kant_rechts = CompareSide(
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
        # lambda: clicked() zou anders `checked=False` als vanaf_sync doorgeven.
        self.btn_start_alles.clicked.connect(lambda: self._start_alles())
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
        for label, factor in SPEEDS:
            self.combo_alles_snelheid.addItem(label, factor)
        self.combo_alles_snelheid.setCurrentIndex(ALL_SPEED_IDX)
        self.combo_alles_snelheid.currentIndexChanged.connect(self._zet_alles_snelheid)
        balk.addWidget(self.combo_alles_snelheid)
        self._zet_alles_snelheid()      # kanten meteen op de startsnelheid zetten

        balk.addStretch(1)
        v.addLayout(balk)

        # Dezelfde toetsen als elders, maar ze sturen hier beide kanten tegelijk — dat is
        # het enige wat op deze pagina anders is en hoort er dus bij te staan.
        hulp = QLabel(keys_help("beide kanten tegelijk"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        v.addWidget(hulp)
        return paneel

    def _bouw_videopaneel(self):
        """De gedeelde VideoPlayer plus de editor-onderdelen die alléén op de analysepagina
        horen (de vergelijkpagina gebruikt dezelfde speler, zonder editor)."""
        # Bescheiden ondergrens: het beeld rekt toch mee met het venster, en een hoge
        # ondergrens tilt het venster-minimum boven de beschikbare schermhoogte uit —
        # dan negeert Qt de gevraagde venstergrootte (zie set_window_size).
        self.speler = VideoPlayer(min_size=(400, 240))
        self.speler.on_frame_shown = self._speler_frame_getoond
        self.speler.overlay_drawer = self._teken_handles
        self.speler.on_mouse_press = self._editor_muis_druk
        self.speler.on_mouse_move = self._editor_muis_beweeg
        self.speler.on_mouse_release = self._editor_muis_los

        self.btn_bewerken = QPushButton("✏ Bewerken")
        self.btn_bewerken.setCheckable(True)
        self.btn_bewerken.setToolTip(
            "Skelet-editor: sleep foute landmarkpunten naar de juiste plek.\n"
            "De correctie vloeit uit naar de buurframes (instelbaar) en wordt\n"
            "direct opgeslagen.")
        self.btn_bewerken.toggled.connect(self._toggle_bewerken)
        self.speler.add_control_button(self.btn_bewerken)

        self.btn_vergelijk_deze = QPushButton("⇄ Vergelijk met...")
        self.btn_vergelijk_deze.setToolTip(
            "Zet deze analyse links op de vergelijkpagina en kies er een andere naast.")
        self.btn_vergelijk_deze.clicked.connect(self._vergelijk_met_deze)
        self.btn_vergelijk_deze.setEnabled(False)
        self.speler.add_control_button(self.btn_vergelijk_deze)

        self.btn_info = QPushButton("ℹ Info...")
        self.btn_info.setToolTip(
            "Met welke appversie, backend en instellingen is deze analyse gemaakt?")
        # lambda: clicked() geeft anders `checked=False` door als analyse_id.
        self.btn_info.clicked.connect(lambda: self._toon_analyse_info())
        self.btn_info.setEnabled(False)
        self.speler.add_control_button(self.btn_info)

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
        self.speler.add_control_button(self.btn_bocht_nu)

        # Editor-balk (fase 3): alleen zichtbaar in bewerk-modus. Afbrekend (WrapBar), want
        # met de plaats-knoppen erbij past hij op een laptopscherm niet meer op één regel —
        # en een te brede balk tilt het venster-minimum boven de schermhoogte uit.
        self.editor_balk = WrapBar()
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
        self.speler.add_bottom_bar(self.editor_balk)

        # Plaats-balk: alleen zichtbaar tijdens een lopende klikreeks. Apart van de
        # editor-balk zodat de gewone bewerk-knoppen niet met de reeks-knoppen mengen.
        self.plaats_balk = WrapBar()
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
        self.speler.add_bottom_bar(self.plaats_balk)

        # Dezelfde regel als onder het knip- en kijkvenster: de toetsen zijn overal gelijk,
        # dus hoort de opsomming dat ook te zijn (zie VIDEO_KEYS_HELP).
        hulp = QLabel(keys_help("<b>Ctrl+Z / Ctrl+Y</b> bewerking terug/opnieuw"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        self.speler.add_bottom_bar(hulp)

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
            skate_db.open_db(pad)
        except skate_db.LibraryTooNew as e:
            # Gedeelde cloudmap waarin een collega met een nieuwere app heeft geschreven:
            # niet aanraken (schrijven zou z'n schema kunnen slopen), wel duidelijk melden.
            QMessageBox.critical(
                self, "Bibliotheek is nieuwer dan deze app",
                f"{e}\n\nMap:\n{pad}\n\n"
                "Er wordt zolang met de standaard-bibliotheekmap gewerkt.")
            standaard = skate_db.default_library()
            if pad != standaard:
                return self._zet_bibliotheek(standaard)
            raise
        except Exception as e:
            QMessageBox.critical(
                self, "Bibliotheek",
                f"Kan de bibliotheek niet openen in:\n{pad}\n\n{e}")
            standaard = skate_db.default_library()
            if pad != standaard:
                return self._zet_bibliotheek(standaard)
            raise
        self.bieb = pad
        self.lbl_bieb.setText(pad)
        # De lokale bibliotheek (losse video's van deze pc) staat los van de gedeelde en
        # verandert niet mee bij het wisselen van bibliotheekmap: één keer openen. Lukt dat
        # niet, dan blijft alles werken behalve het onthouden van losse video's.
        if self.lokaal is None:
            try:
                self.lokaal = skate_db.local_library()
            except Exception as e:
                self.statusBar().showMessage(
                    f"Lokale bibliotheek niet beschikbaar (losse video's worden niet "
                    f"onthouden): {e}", 8000)
        self._waarschuw_conflictkopieen()
        self._vernieuw_schaatsers()
        # Tijdens het opstarten de traagste stap apart melden: bij een nieuwe opname leest
        # de scan de videometa, en op een cloudmap kan dat seconden duren.
        self._melding("Opnames scannen...")
        self._vernieuw_opnames()      # fase 8: werklijst met nog te knippen opnames

    def _waarschuw_conflictkopieen(self):
        """Fase 4: waarschuwt als de cloudsync naast schaats.db conflictkopieën van de
        database heeft achtergelaten (zie skate_db.detect_conflict_copies)."""
        try:
            kopieen = skate_db.detect_conflict_copies(self.bieb)
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
        cfg = skate_db.load_config()
        cfg["trainer_naam"] = self.trainer_naam
        skate_db.save_config(cfg)
        self._toon_trainer_naam()

    def _kies_bibliotheekmap(self):
        pad = QFileDialog.getExistingDirectory(self, "Kies bibliotheekmap", self.bieb or "")
        if not pad:
            return
        cfg = skate_db.load_config()
        cfg["bibliotheek_pad"] = pad
        skate_db.save_config(cfg)
        self._zet_bibliotheek(pad)

    def _geselecteerde_schaatser_id(self):
        item = self.lijst_schaatsers.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _schaatser_naam(self, schaatser_id):
        """Naam bij een schaatser-id, of "" als die er niet (meer) is."""
        if schaatser_id is None:
            return ""
        s = next((x for x in skate_db.list_skaters(self.bieb)
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
        schaatsers = skate_db.list_skaters(self.bieb)
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
            analyses = skate_db.list_analyses(self.bieb, sid)
            self.tabel_analyses.setRowCount(len(analyses))
            for rij, a in enumerate(analyses):
                # Herkomst in de tooltip: wie hem maakte (fase 4) en met welke appversie —
                # zo is zonder openen te zien of een analyse nog met oude code is gedraaid.
                door = (a.get("aangemaakt_door") or "").strip()
                versie = ((a.get("instellingen") or {}).get("app_versie") or "").strip()
                tip = "\n".join(r for r in (f"Aangemaakt door {door}" if door else "",
                                            f"Appversie: {versie}" if versie else "") if r)
                for kolom, tekst in enumerate(
                        [a["datum"], a["titel"], _duration_text(a)]):
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
        dlg = SkaterDialog(self)
        if show_dialog(dlg) != QDialog.Accepted or not dlg.naam:
            return
        sid = skate_db.create_skater(self.bieb, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._vernieuw_schaatsers(selecteer_id=sid)

    def _bewerk_schaatser(self):
        sid = self._geselecteerde_schaatser_id()
        if sid is None:
            return
        s = next((x for x in skate_db.list_skaters(self.bieb) if x["id"] == sid), None)
        if s is None:
            return
        dlg = SkaterDialog(self, naam=s["naam"], geboortejaar=s["geboortejaar"],
                          notities=s["notities"])
        if show_dialog(dlg) != QDialog.Accepted or not dlg.naam:
            return
        skate_db.edit_skater(self.bieb, sid, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._vernieuw_schaatsers(selecteer_id=sid)

    def _verwijder_schaatser(self):
        sid = self._geselecteerde_schaatser_id()
        if sid is None:
            return
        naam = self.lijst_schaatsers.currentItem().data(Qt.UserRole + 1)
        analyses = skate_db.list_analyses(self.bieb, sid)
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
        skate_db.delete_skater(self.bieb, sid)
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
        skate_db.rename_analysis(self.bieb, aid, titel.strip())
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
        skate_db.delete_analysis(self.bieb, aid)
        self._vernieuw_schaatsers()

    def _sluit_weergave(self):
        """Maakt de weergavepagina leeg en laat het videobestand los (nodig voordat de
        mediamap van de geopende analyse verwijderd kan worden)."""
        self.speler.release()
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
        schaatsers = skate_db.list_skaters(self.bieb)
        if not schaatsers:
            QMessageBox.information(
                self, "Nieuwe analyse",
                "Maak eerst een schaatser aan — elke analyse hoort bij een profiel.")
            return
        dlg = NewAnalysisDialog(schaatsers, voorkeur_id=self._geselecteerde_schaatser_id(),
                                parent=self)
        if show_dialog(dlg) != QDialog.Accepted:
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
                    self.model_pad = DEFAULT_MODEL
            else:
                self.model_pad = DEFAULT_MODEL

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

        # Doelschaatser laten kiezen op het eerste frame (klik of kader).
        keuze = self._kies_doelschaatser(frame0)
        if keuze is False:               # dialoog afgebroken
            return
        self.doel_punt, self.doel_kader = keuze

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
        self.deinterlacen = dlg.deinterlacen

        # Wat het .npz níet bevat maar heropenen wél nodig heeft/wil documenteren.
        # (De bocht-vlag per frame zit wél in het npz; dit is puur de instelling.)
        instellingen = {
            "smooth_n": self.smooth_n,
            "threshold": self.threshold,
            "smooth_landmarks": not self.geen_smoothing,
            "bocht_overslaan": self.bocht_overslaan,
            "deinterlaced": self.deinterlacen,
            "doel_punt": list(self.doel_punt) if self.doel_punt else None,
            "doel_kader": list(self.doel_kader) if self.doel_kader else None,
            "horizon_deg": self.horizon_deg,
            "auto_horizon": self.auto_horizon,
            "heavy": heavy,
            "backend_naam": BACKEND_NAME,
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
        eerdere analyse over te nemen (zelfde camerastand), dan de `CalibrationPicker`
        — voorgevuld als er iets overgenomen is, zodat controleren en corrigeren
        dezelfde handeling blijft. Retourneert een PerspectiveConfig, of None bij
        afbreken.

        Hergebruik is hier geen gemak maar een meetkundige voorwaarde: analyses die je
        onderling wilt vergelijken moeten op dezelfde kalibratie rusten, anders meet je
        de spreiding tussen zeven keer natrekken in plaats van het effect van de
        correctie.
        """
        h, w = frame0.shape[:2]
        invoer = config = None
        try:
            eerdere = skate_db.list_calibrations(self.bieb, beeld_w=w, beeld_h=h)
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
                    config = PerspectiveConfig.uit_dict(eerdere[idx - 1]["perspectief"])
                    invoer = config.invoer
                except Exception as e:
                    QMessageBox.warning(
                        self, "Kalibratie onbruikbaar",
                        f"Die opgeslagen kalibratie is niet te herberekenen:\n\n{e}\n\n"
                        "Trek de lijnen opnieuw na.")
                    config = invoer = None

        kdlg = CalibrationPicker(frame0, self, calibration_input=invoer, config=config)
        if show_dialog(kdlg) != QDialog.Accepted:
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
            data = skate_db.load_analysis(self.bieb, analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Fout bij openen",
                                 f"Kan de analyse niet laden:\n\n{e}")
            return None

        sync = skate_db.video_sync_status(data["video_pad"], data["meta"].get("video_bytes"))
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
                perspectief = PerspectiveConfig.uit_dict(inst["perspectief"])
            except Exception as e:
                persp_fout = str(e)

        # De horizon zit al per frame in het .npz; alleen de afgeleiden herberekenen.
        process_derivatives(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                           perspectief=perspectief)
        events = segment_pushes(resultaten)
        # De events-cache is een momentopname van de berekening bij het opslaan; wat je
        # hier ziet is vers herberekend. Bijwerken houdt de lijstweergave (aantal afzetten,
        # gemiddelde hoek) gelijk aan de tabel — ook voor analyses van vóór een
        # algoritme-verbetering. Mislukt het (bv. DB even op slot in de cloudmap), dan is
        # dat geen reden om het openen af te breken.
        try:
            skate_db.refresh_events_cache(self.bieb, analyse_id, events)
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
            # Zo krijgt de speler exact de pixels te zien waarop gemeten is. Een analyse
            # van vóór deze functie mist de sleutel en toont dus het ruwe beeld — precies
            # wat er toen ook gemeten is.
            "deinterlaced": bool(inst.get("deinterlaced")),
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
        self.deinterlacen = data["deinterlaced"]
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
        if not any(s["aantal_analyses"] for s in skate_db.list_skaters(self.bieb)):
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
            meta = skate_db.analysis_meta(self.bieb, analyse_id)
        except Exception as e:
            QMessageBox.warning(self, "Info", f"Kon de analysegegevens niet lezen:\n{e}")
            return
        show_dialog(AnalysisInfoDialog(
            meta, self._schaatser_naam(meta.get("schaatser_id")), parent=self))

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
        dlg = AnalysisPicker(self.bieb, titel=f"{kant.naam}: kies analyse",
                             voorkeur_schaatser_id=voorkeur_id, parent=self)
        if show_dialog(dlg) != QDialog.Accepted or dlg.analyse_id is None:
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

    def _start_alles(self, vanaf_sync=None):
        """Beide video's tegelijk afspelen, aangestuurd door één `MasterClock`.
        `vanaf_sync`: None = wat het vinkje zegt (de knop), False = hervatten (spatie)."""
        kanten = [k for k in (self.kant_links, self.kant_rechts) if k.heeft_analyse()]
        if not kanten:
            QMessageBox.information(self, "Niets te starten",
                                    "Kies eerst voor beide kanten een analyse.")
            return
        self._pauzeer_alles()
        if vanaf_sync is None:
            vanaf_sync = self.chk_vanaf_sync.isChecked()
        if vanaf_sync:
            # Terugspoelen heropent de video en spoelt sequentieel; die wachttijd zit zo
            # eenmalig vooraan in plaats van in de eerste tick.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for kant in kanten:
                    kant.naar_sync()
            finally:
                QApplication.restoreOverrideCursor()
        self.klok.start([k.speler for k in kanten])

    def _stop_alles(self):
        # De klok pauzeert zelf de spelers die eronder liepen (zie MasterClock.stop); de
        # andere met rust laten, anders werkt ▶ per kant niet meer.
        self.klok.stop()

    def _zet_alles_snelheid(self, _idx=None):
        """De gedeelde snelheid van de vergelijkpagina toepassen.

        Beide kanten krijgen dezelfde factor — ook voor los afspelen, want twee video's
        op verschillend tempo naast elkaar zijn niet te vergelijken. Draait de masterklok,
        dan wordt die opnieuw geijkt vanaf de huidige stand (`MasterClock.recalibrate`)."""
        idx = self.combo_alles_snelheid.currentIndex()
        for kant in (self.kant_links, self.kant_rechts):
            kant.speler.combo_speed.setCurrentIndex(idx)   # herstart een lopende timer
        self.klok.recalibrate()

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
        """Toont het eerste frame in een kiezer. Retourneert `(doel_punt, doel_kader)` —
        elk genormaliseerd of None ('volg grootste') — of False (afgebroken)."""
        dlg = TargetPicker(frame0, self)
        if show_dialog(dlg) != QDialog.Accepted:
            return False
        return dlg.doel_punt, dlg.doel_kader

    def _kies_horizon(self, frame0):
        """
        Laat de ijslijn/kanteling instellen. Retourneert (graden, auto_per_frame) of
        False (afgebroken).
        """
        dlg = HorizonPicker(frame0, self)
        if show_dialog(dlg) != QDialog.Accepted:
            return False
        return dlg.horizon_deg, dlg.auto_per_frame

    def _terug_naar_start(self):
        # pauzeren gebeurt via de stack.currentChanged-haak (_pauzeer_alles)
        self._vernieuw_schaatsers()   # nieuwe/gewijzigde analyses direct zichtbaar
        self.stack.setCurrentWidget(self.pagina_start)

    def _warn_backend_fallback(self):
        """Meldt (één keer) dat de YOLO-backend niet geladen kon worden en er dus met
        MediaPipe gemeten wordt — een andere detector geeft andere hoeken, dus dat mag
        niet onopgemerkt blijven. Het warmdraaien start bij het tonen van het venster, dus
        op het moment dat hier een analyse begint is de uitkomst allang bekend.

        In een gebundelde .exe is er geen MediaPipe om op terug te vallen; daar is het
        geen waarschuwing maar een blokkade, en zegt de melding dat ook."""
        if not BACKEND_ERROR or self._backend_gemeld:
            return
        self._backend_gemeld = True
        if is_frozen():
            # In het gebundelde pakket is er geen tweede backend om op terug te vallen:
            # er valt nu niets te meten (bibliotheek en opnames bekijken werken wel).
            QMessageBox.critical(
                self, "Analyse-backend niet beschikbaar",
                "De meegeleverde analyse-backend liet zich niet laden:\n\n"
                f"{BACKEND_ERROR}\n\n"
                "Er kan nu niet geanalyseerd worden. De bibliotheek openen en opnames "
                "bekijken werkt wel. Geef deze melding door aan de beheerder van de app."
                # Het logboek is alleen iets waard als de gebruiker weet waar het staat.
                + (f"\n\nHet volledige logboek staat in:\n{LOGPATH}" if LOGPATH else ""))
            return
        QMessageBox.warning(
            self, "YOLO-backend niet beschikbaar",
            "torch/ultralytics is wel geïnstalleerd, maar liet zich niet laden:\n\n"
            f"{BACKEND_ERROR}\n\n"
            "De analyse draait daarom met de MediaPipe-backend. Die meet minder "
            "nauwkeurig, dus vergelijk deze analyse niet zomaar met eerdere.")

    def _start_analyse(self):
        self._warn_backend_fallback()
        self.speler.set_controls_active(False)
        self.btn_export.setEnabled(False)
        self._auto_toon_klaar = True          # nog niets anders geopend → resultaat straks tonen
        self._analyse_waarschuwingen = []     # meldingen uit de analyse zelf (na afloop tonen)
        self._zet_bezig(True)                 # geen tweede worker/botsende bewerking eroverheen
        self._toon_voortgangsbalk("Video analyseren...")

        opslag = self._pending_opslag or {}
        self.worker = AnalysisWorker(self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                                     doel_punt=self.doel_punt, horizon_deg=self.horizon_deg,
                                     auto_horizon=self.auto_horizon,
                                     smooth_landmarks=not self.geen_smoothing,
                                     perspectief=self.perspectief,
                                     bieb=self.bieb,
                                     schaatser_id=opslag.get("schaatser_id"),
                                     titel=opslag.get("titel"),
                                     instellingen=opslag.get("instellingen"),
                                     backend=BACKEND_NAME,
                                     aangemaakt_door=self.trainer_naam,
                                     bocht=self.bocht_overslaan,
                                     deinterlacen=self.deinterlacen,
                                     doel_kader=self.doel_kader)
        self.worker.progress.connect(self._analyse_voortgang)
        self.worker.status.connect(self._analyse_status)
        self.worker.save_error.connect(self._opslag_fout)
        self.worker.warning.connect(self._analyse_waarschuwing)
        self.worker.done.connect(self._analyse_klaar)
        self.worker.error.connect(self._analyse_fout)
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
        self.speler.set_controls_active(False)

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
                self.input_pad = skate_db.analysis_video_path(self.bieb, analyse_id)
            except Exception:
                pass   # terugvallen op de bronvideo (alleen weergave)
        self.stack.setCurrentWidget(self.pagina_analyse)
        self._toon_resultaten(info, resultaten, events)

    # ---- Batch-analyse (meerdere video's achter elkaar) --------------------------

    # ── Fragmenten knippen uit een lange opname (fase 8) ─────────────────────
    def _knip_opname(self):
        """Opname → knipvenster → clips wegschrijven → de bestaande batch-flow in.

        Ná het knippen gebeurt er niets nieuws: elk fragment is een gewoon videobestandje,
        dus `BatchAnalysisDialog` (voorgevuld) + `_nieuwe_batch_analyse` doen de rest. Geen
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
        schaatsers = skate_db.list_skaters(self.bieb)
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

        # Een losse video staat in de lokale bibliotheek, de analyses komen in de gedeelde:
        # daar is geen bron-rij om naar te wijzen, dus die fragmenten worden analyses
        # zonder herkomst (zoals een batch van losse clips) en het knipvenster kan er geen
        # al-geknipte stukken bij tekenen.
        gedeeld = bron["bieb"] == self.bieb
        gedaan = skate_db.source_fragments(self.bieb, bron["id"]) if gedeeld else []
        dlg = FragmentPicker(bron["pad"], info, gedaan=gedaan,
                             deinterlacen=self._bron_interlaced(bron), parent=self)
        if show_dialog(dlg) != QDialog.Accepted or not dlg.fragmenten:
            return

        paden = self._knip_naar_tijdelijk(bron["pad"], dlg.fragmenten, info,
                                          deinterlacen=self._bron_interlaced(bron))
        if not paden:
            return
        voorgevuld = [
            {"input_pad": pad, "titel": naam,
             "bron_id": bron["id"] if gedeeld else None,
             "bron_start_frame": start if gedeeld else None,
             "bron_eind_frame": eind if gedeeld else None}
            for pad, (start, eind, naam) in zip(paden, dlg.fragmenten)]
        self._nieuwe_batch_analyse(voorgevuld=voorgevuld)

    def _knip_naar_tijdelijk(self, bron_pad, fragmenten, info, deinterlacen=False):
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
            paden = trim_fragments(bron_pad, fragmenten, self._knip_tmpmap,
                                    progress_callback=_melden,
                                    stop_check=voortgang.wasCanceled, fps=info.fps,
                                    deinterlacen=deinterlacen)
        except TrimAborted:
            self._ruim_knipmap_op()
            return []
        except Exception as e:
            self._ruim_knipmap_op()
            QMessageBox.critical(self, "Knippen mislukt", str(e))
            return []
        finally:
            # close() verbergt alleen; zonder deleteLater blijft dit venster (mét zijn
            # QScreen-verwijzing) achter in precies de flow die crashte — zie show_dialog.
            voortgang.close()
            voortgang.deleteLater()
        return paden

    def _ruim_knipmap_op(self):
        """Gooit de tijdelijke fragmentmap weg (de clips staan dan in de bibliotheek)."""
        if self._knip_tmpmap:
            shutil.rmtree(self._knip_tmpmap, ignore_errors=True)
            self._knip_tmpmap = None

    def _nieuwe_batch_analyse(self, voorgevuld=None):
        schaatsers = skate_db.list_skaters(self.bieb)
        if not schaatsers:
            QMessageBox.information(
                self, "Batch-analyse",
                "Maak eerst een schaatser aan — elke analyse hoort bij een profiel.")
            return
        dlg = BatchAnalysisDialog(schaatsers, voorkeur_id=self._geselecteerde_schaatser_id(),
                                  voorgevuld=voorgevuld, parent=self)
        if show_dialog(dlg) != QDialog.Accepted:
            self._ruim_knipmap_op()      # geknipte clips zonder batch zijn nutteloos
            return

        # Model resolven — gedeeld voor de hele batch, alleen relevant voor MediaPipe.
        model_pad, heavy = DEFAULT_MODEL, False
        if not IS_YOLO:
            if dlg.heavy_gevraagd:
                if os.path.isfile(HEAVY_MODEL):
                    model_pad, heavy = HEAVY_MODEL, True
                else:
                    QMessageBox.warning(
                        self, "Heavy-model ontbreekt",
                        "pose_landmarker_heavy.task staat niet naast het script.\n\n"
                        "Er wordt nu met het full-model gewerkt.")
                    model_pad = DEFAULT_MODEL
            if not os.path.isfile(model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Kies pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                model_pad = gekozen

        smooth_n, threshold = dlg.smooth_n, dlg.threshold
        geen_smoothing = dlg.geen_smoothing
        bocht = dlg.chk_bocht.isChecked()
        deint_auto = dlg.chk_deint.isChecked()

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

            keuze = self._kies_doelschaatser(frame0)
            if keuze is False:                   # dialoog afgebroken
                if self._overslaan_of_afbreken(titel):
                    continue
                return
            doel, kader = keuze
            if batch_perspectief is not None:
                # De kalibratie levert de kanteling zelf; de horizon-stap vervalt, net
                # als bij een enkele analyse.
                horizon_deg, auto_horizon = 0.0, False
                if not batch_perspectief.invoer.fits(frame0.shape[1], frame0.shape[0]):
                    QMessageBox.warning(
                        self, "Kalibratie past niet",
                        f"'{titel}' is {frame0.shape[1]}×{frame0.shape[0]} en de "
                        f"kalibratie is gemaakt op {batch_perspectief.invoer.image_w}×"
                        f"{batch_perspectief.invoer.image_h}. Deze clip wordt overgeslagen.")
                    continue
            else:
                horizon = self._kies_horizon(frame0)
                if horizon is False:             # dialoog afgebroken
                    if self._overslaan_of_afbreken(titel):
                        continue
                    return
                horizon_deg, auto_horizon = horizon

            # Per clip, want een batch kan clips uit verschillende camera's bevatten.
            # Uitgezet in de dialoog = nergens filteren (om een A/B te kunnen draaien).
            deint = False
            if deint_auto:
                try:
                    deint = is_interlaced(pad)
                except Exception:
                    deint = False

            instellingen = {
                "smooth_n": smooth_n,
                "threshold": threshold,
                "smooth_landmarks": not geen_smoothing,
                "bocht_overslaan": bocht,
                "deinterlaced": deint,
                "doel_punt": list(doel) if doel else None,
                "doel_kader": list(kader) if kader else None,
                "horizon_deg": horizon_deg,
                "auto_horizon": auto_horizon,
                "heavy": heavy,
                "backend_naam": BACKEND_NAME,
                "perspectief_gebruikt": batch_perspectief is not None,
                "perspectief": batch_perspectief.naar_dict() if batch_perspectief else None,
            }
            taken.append({
                "input_pad": pad, "schaatser_id": schaatser_id, "titel": titel,
                "doel_punt": doel, "doel_kader": kader,
                "horizon_deg": horizon_deg, "auto_horizon": auto_horizon,
                "smooth_landmarks": not geen_smoothing, "smooth_n": smooth_n,
                "threshold": threshold, "model_pad": model_pad, "instellingen": instellingen,
                "bocht": bocht, "perspectief": batch_perspectief, "deinterlacen": deint,
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
        self.speler.set_controls_active(False)
        self.btn_export.setEnabled(False)
        self._auto_toon_klaar = False        # batch toont zelf geen resultaten
        self._zet_bezig(True)

        self._batch_index, self._batch_totaal, self._batch_huidig = 0, len(taken), ""

        # Voortgangsbalk mét knop 'Stop na deze video' — niet-blokkerend, dus de
        # bibliotheek blijft ondertussen bruikbaar.
        self._toon_voortgangsbalk("Batch starten...", met_stop=True)

        self._warn_backend_fallback()
        self.batch_worker = BatchWorker(taken, self.bieb, BACKEND_NAME, self.trainer_naam)
        self.batch_worker.task_start.connect(self._batch_taak_start)
        self.batch_worker.progress.connect(self._batch_voortgang)
        self.batch_worker.status.connect(self._analyse_status)   # reuse the busy phase
        self.batch_worker.task_done.connect(self._batch_taak_klaar)  # show the new analysis live
        self.batch_worker.all_done.connect(self._batch_klaar)
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
        self.speler.edit_mode = False
        self.speler.follow_frozen = False
        self._undo.clear()
        self._redo.clear()
        self._handmatig.clear()
        self.btn_bewerken.blockSignals(True)
        self.btn_bewerken.setChecked(False)
        self.btn_bewerken.blockSignals(False)
        self.editor_balk.setVisible(False)
        self._sluit_plaats_balk()

        # Capture heropenen, zoom resetten, besturing aan — toont nog géén frame.
        self.speler.load(info, resultaten, self.input_pad, self.deinterlacen)

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

        # Pas nu tekenen: tabel en grafiek staan klaar voor de on_frame_shown-haak.
        self.speler.go_to(0)

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
            INCOMPLETE_TRUNCATED:
                "Deze afzet liep nog toen de video (of de detectie) ophield — de push is "
                "niet afgemaakt, dus de hoek is te steil. Telt niet mee in "
                "gemiddelde/min/max.",
            INCOMPLETE_NO_PUSH:
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
        """Haak van de VideoPlayer: alles wat de analysepagina aan een frame ophangt."""
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
            # Wel een skelet, maar geen afgeleiden: `process_derivatives` slaat bochtframes
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

    # ── Skelet-editor: handles op de VideoPlayer ─────────────────────────────
    def _handle_straal(self):
        """Handle-/grijpradius in (geschaalde) schermpixels, evenredig met de schaatser:
        HANDLE_FRAC × torso-lengte-op-het-scherm, geklemd op [HANDLE_MIN_PX, HANDLE_MAX_PX].
        Via _norm_naar_widget zit de crop/zoom-schaal er al in (de letterbox-offset valt bij
        een afstand weg), dus dit klopt op elke zoomstand en is exact consistent met het
        hittesten. Val terug op HANDLE_MAX_PX als er geen bruikbare pose/torso is."""
        if (not (0 <= self.huidige_idx < len(self.resultaten))
                or self.speler.display_scaled is None):
            return float(HANDLE_MAX_PX)
        r = self.resultaten[self.huidige_idx]
        if not (r.pose_gevonden and isinstance(r.lm, list)):
            return float(HANDLE_MAX_PX)
        lm = r.lm

        def _mid(a, b):
            pts = [lm[i] for i in (a, b)
                   if getattr(lm[i], 'visibility', 1.0) >= HANDLE_MIN_VIS]
            if not pts:
                return None
            return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))

        schouder, heup = _mid(11, 12), _mid(23, 24)   # schouder-midden → heup-midden
        if schouder is None or heup is None:
            return float(HANDLE_MAX_PX)
        p1 = self.speler.norm_to_widget(*schouder)
        p2 = self.speler.norm_to_widget(*heup)
        torso = math.hypot(p1.x() - p2.x(), p1.y() - p2.y())
        return min(float(HANDLE_MAX_PX), max(float(HANDLE_MIN_PX), HANDLE_FRAC * torso))

    def _teken_handles(self, pixmap):
        """Tekent sleepbare ringen op elke zichtbare landmark van het huidige frame,
        rechtstreeks op de geschaalde pixmap (dus vaste grootte in schermpixels).

        Hangt permanent als overlay_drawer aan de speler; de bewerk-modus-guard zit
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
        return PLACEMENT_ORDER[stap] if 0 <= stap < len(PLACEMENT_ORDER) else None

    # ── Skelet-editor: bewerk-modus + slepen (fase 3) ────────────────────────
    def _toggle_bewerken(self, actief):
        if not actief:
            self._stop_plaatsen()           # nooit een halve reeks achterlaten
        self._editor_actief = actief
        self.speler.edit_mode = actief   # stuurt de pan-vs-editor-voorrang van de muis
        self.editor_balk.setVisible(actief)
        self._sleep = None
        self.speler.follow_frozen = False
        if actief:
            self.speler.pause()
            self.lbl_editor_hint.setText("Sleep een punt naar de juiste plek.")
            self._update_editor_knoppen()
        self.speler.show_current_frame()

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
        if not self._frame_bewerkbaar(self.huidige_idx) or self.speler.display_scaled is None:
            return None
        straal = self._handle_straal()      # zelfde radius als de getekende ring
        beste, beste_d2 = None, float(straal * straal)
        for j, lm in enumerate(self.resultaten[self.huidige_idx].lm):
            if getattr(lm, 'visibility', 1.0) < HANDLE_MIN_VIS:
                continue
            w = self.speler.norm_to_widget(lm.x, lm.y)
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
        naam = LANDMARK_NAMES.get(j, f"punt {j}")
        # iets naast de cursor zodat de tekst het punt zelf niet afdekt
        pos = (event.globalPosition() + QPointF(14, 10)).toPoint()
        QToolTip.showText(pos, naam, self.speler.label)

    # De pan-tak (links-slepen bij zoom > 1 buiten bewerk-modus) zit in de VideoPlayer;
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
        self.speler.follow_frozen = True

    def _editor_muis_beweeg(self, event):
        if not self._editor_actief or self._plaats is not None:
            return
        if not self._sleep:
            # geen sleep bezig → toon bij hover het lichaamsdeel onder de cursor
            self._toon_hover_naam(event)
            return
        norm = self.speler.widget_to_norm(event.position())
        if norm is None:
            return
        nx = min(1.0, max(0.0, norm[0]))
        ny = min(1.0, max(0.0, norm[1]))
        idx, j = self._sleep['idx'], self._sleep['j']
        self._zet_landmark(idx, j, nx, ny, vis=1.0)   # live feedback; nog geen herbereken
        self.speler.go_to(idx)

    def _editor_muis_los(self, event):
        if self._plaats is not None:
            return                          # de klik is al bij het indrukken afgehandeld
        if not (self._editor_actief and self._sleep):
            self.speler.follow_frozen = False
            return
        sleep, self._sleep = self._sleep, None
        self.speler.follow_frozen = False
        idx, j, start_lm = sleep['idx'], sleep['j'], sleep['start_lm']
        eind = self.resultaten[idx].lm[j]
        dx, dy = eind.x - start_lm.x, eind.y - start_lm.y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            self.speler.go_to(idx)              # geen echte verplaatsing: alleen hertekenen
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
        self.speler.go_to(doel)
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
        voorvulling = make_prefill(self.resultaten, idx, info.fps or 30.0)
        bruikbaar = all(voorvulling[j].visibility >= HANDLE_MIN_VIS for j in PLACEMENT_REQUIRED)

        r.lm = voorvulling
        r.pose_gevonden = True
        # Het kader is berekend toen dit nog een gat was — op een gat > KADER_GAT_S staat de
        # automatische zoom volledig uit, precies wanneer je nauwkeurig moet werken.
        self.speler.recompute_box()

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
        self.speler.follow_frozen = True   # uitsnede mag niet verspringen tussen klikken
        self.plaats_balk.setVisible(True)
        self._toon_plaats_stap()

    def _toon_plaats_stap(self):
        """Hint + knopstatus voor de huidige stap; ververst ook het beeld (doelpunt-ring)."""
        if self._plaats is None:
            return
        stap, n = self._plaats['stap'], len(PLACEMENT_ORDER)
        if stap < n:
            naam = LANDMARK_NAMES.get(PLACEMENT_ORDER[stap], f"punt {PLACEMENT_ORDER[stap]}")
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
        self.speler.show_current_frame()

    def _plaats_compleet(self):
        """Mag het skelet vastgelegd worden? Alleen als elk meetpunt een zichtbare positie
        heeft — anders belandt er een frame met heup, knie en enkel op één punt in de
        tabel, en dat leest als een afzethoek van 0°."""
        if self._plaats is None:
            return False
        lm = self.resultaten[self._plaats['idx']].lm
        return all(lm[j].visibility >= HANDLE_MIN_VIS for j in PLACEMENT_REQUIRED)

    def _plaats_klik(self, event):
        if self._plaats is None:
            return
        stap = self._plaats['stap']
        if stap >= len(PLACEMENT_ORDER):
            self.lbl_plaats.setText("Alle punten gehad — klik op ✔ Klaar.")
            return
        norm = self.speler.widget_to_norm(event.position())
        if norm is None or not (0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0):
            # Niet klemmen: dat zou de knie stilzwijgend op de beeldrand leggen.
            self.lbl_editor_hint.setText("Klik binnen het beeld.")
            return
        j = PLACEMENT_ORDER[stap]
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
        if self._plaats is not None and self._plaats['stap'] < len(PLACEMENT_ORDER):
            self._plaats['stap'] += 1
            self._toon_plaats_stap()

    def _plaats_klaar(self):
        if self._plaats is None:
            return
        if not self._plaats_compleet():
            ontbreekt = ", ".join(
                LANDMARK_NAMES.get(j, str(j)) for j in PLACEMENT_REQUIRED
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
        self.speler.recompute_box()
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
        self.speler.recompute_box()
        self._herbereken()               # bewust niet opslaan: er is niets veranderd
        self.lbl_editor_hint.setText("Skelet plaatsen geannuleerd.")

    def _sluit_plaats_balk(self):
        self.plaats_balk.setVisible(False)
        self.lbl_plaats.setText("")
        self.speler.follow_frozen = False

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
        process_derivatives(self.resultaten, info.w, info.h, info.fps,
                           self.smooth_n, self.threshold,
                           perspectief=self.perspectief)
        self.events = segment_pushes(self.resultaten)
        self._vul_tabel()
        self._vul_grafiek()
        self.btn_export.setEnabled(bool(self.events))
        self._update_dekking()
        self.speler.show_current_frame()
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

        determine_corner_sequence(self.resultaten, info.w, info.h, info.fps)
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
            skate_db.save_corner_marking(
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
                skate_db.save_edited_landmarks(
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
            self.speler.recompute_box()   # de dekking veranderde, dus de auto-zoom ook
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
            hersteld = skate_db.restore_original_landmarks(self.bieb, self.analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Herstel origineel", f"Mislukt:\n\n{e}")
            return
        if not hersteld:
            QMessageBox.information(
                self, "Herstel origineel",
                "Deze analyse is nog niet bewerkt — er is niets te herstellen.")
            return
        try:
            data = skate_db.load_analysis(self.bieb, self.analyse_id)
        except Exception as e:
            QMessageBox.critical(self, "Herstel origineel", f"Herladen mislukt:\n\n{e}")
            return
        info, resultaten = data["info"], data["resultaten"]
        # Ook hier de kalibratie meegeven — "origineel herstellen" gaat over de
        # landmarks, niet over de perspectiefcorrectie.
        process_derivatives(resultaten, info.w, info.h, info.fps, self.smooth_n,
                           self.threshold, perspectief=self.perspectief)
        events = segment_pushes(resultaten)
        try:
            skate_db.refresh_events_cache(self.bieb, self.analyse_id, events)
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
            self.speler.go_to(self.events[rij].start_frame)

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
            w.abort()
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

    def changeEvent(self, event):
        # Niet meer het actieve venster (alt-tab, een modale dialoog ervoor): de key-release
        # van . of , komt dan nooit binnen en het spoelen zou eindeloos doorlopen.
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            for toetsen in self._toetsen:
                toetsen.stop_scrubbing()
        super().changeEvent(event)

    def closeEvent(self, event):
        if not self._stop_workers():
            event.ignore()
            return
        self._stop_lokaal_proef()
        self._stop_plaatsen()   # een lopende reeks nog vastleggen of terugdraaien
        self._pauzeer_alles()
        for toetsen in self._toetsen:
            toetsen.detach()
        self.speler.release()
        self.kant_links.leeg()
        self.kant_rechts.leeg()
        self._ruim_knipmap_op()   # geknipte fragmenten die niet meer geanalyseerd worden
        super().closeEvent(event)


def main():
    # De QApplication en het opstartscherm bestaan al sinds de import bovenaan dit bestand
    # (zie _start_splash_screen); alleen als deze module via een omweg wordt gestart, zijn
    # ze er niet.
    app = _APP or QApplication(sys.argv)
    venster = MainWindow(melding=_SPLASH.melding if _SPLASH else None)
    venster.show()
    venster._melding = lambda tekst: None    # het opstartscherm gaat nu dicht
    if _SPLASH:
        _SPLASH.finish(venster)
        # finish() verbergt het opstartscherm alleen. Zonder dit blijft het de héle sessie
        # als top-level venster bestaan — met een QScreen-verwijzing die bij een
        # schermwijziging verouderd raakt; zie show_dialog.
        _SPLASH.deleteLater()
    # Pas nu torch/ultralytics binnenhalen: het venster staat er, de gebruiker kan al door
    # de bibliotheek bladeren, en tegen de tijd dat hij een analyse start is de backend er.
    _warm_backend_up()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
