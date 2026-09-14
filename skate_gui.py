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
        p.drawText(QRect(0, 40, cls.WIDTH, 44), Qt.AlignCenter, "SkateAnalysis")
        p.setPen(QColor(150, 180, 220))
        p.setFont(QFont(p.font().family(), 9))
        p.drawText(QRect(0, 84, cls.WIDTH, 22), Qt.AlignCenter, "starting up...")
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
    scherm.melding("Loading components...")
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
    "Camcorder footage (1080i) consists of two half-frames 1/50 s apart, woven\n"
    "together into one frame. On a moving leg those two halves sit in a different\n"
    "spot — the combing you see in the image.\n\n"
    "Measured on this kind of material, the two halves sit 6 px apart on knees and\n"
    "ankles; with '2 px keypoint error = 2-4° angle error' that's the biggest source\n"
    "of noise there is in such recordings. The filter removes it.\n\n"
    "Determined per video automatically; progressive material (phone, GoPro) is\n"
    "left untouched. Turn off only to run an A/B comparison.")

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
    0: "nose",
    1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)",
    4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)",
    7: "left ear", 8: "right ear", 9: "mouth left", 10: "mouth right",
    11: "left shoulder", 12: "right shoulder",
    13: "left elbow", 14: "right elbow",
    15: "left wrist", 16: "right wrist",
    17: "left pinky", 18: "right pinky",
    19: "left index", 20: "right index",
    21: "left thumb", 22: "right thumb",
    23: "left hip", 24: "right hip",
    25: "left knee", 26: "right knee",
    27: "left ankle", 28: "right ankle",
    29: "left heel", 30: "right heel",
    31: "left toe", 32: "right toe",
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
DRAW_MODES = (("✋ Pan", DRAW_PAN),
              ("✏ Sketch", DRAW_SKETCH),
              ("📏 Line", DRAW_LINE))
DRAW_TOOLTIP = (
    "What the left mouse button does on the image:\n"
    "  ✋ Pan — drag the zoomed-in image (and, in edit mode, drag points)\n"
    "  ✏ Sketch — draw freehand for as long as you hold the button down\n"
    "  📏 Line — a straight line from press to release\n"
    "\n"
    "Right-drag always pans the image, even in the middle of drawing.\n"
    "A drawing belongs to the clip, not to one frame: it stays put while the\n"
    "video keeps playing, and moves along with zooming and panning.")
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
    "Recognizes from the position of the hips when the skater isn't frontal in frame\n"
    "(in the corner they stand behind each other instead of side by side).\n"
    "\n"
    "Those frames then mostly no longer go through the detector — that saves a good\n"
    "deal of analysis time — and they yield no push measurement. Every ~0.3 s it checks\n"
    "whether the straight stretch has started again, so a clip that begins in the\n"
    "corner picks up the measurement automatically once the skater comes straight at\n"
    "the camera.\n"
    "\n"
    "Turn off only to see what happens in the corner; those angles aren't usable.")

# Perspective correction (phase 7). Still experimental: the math and the pipeline hookup
# are there and the calibration now gets saved, but the correction hasn't been validated
# on real material yet (ROADMAP phase 7, step 3). Hence "experimental" and not "doesn't
# work" -- it's only on when you deliberately measure with it.
PERSPECTIVE_TOOLTIP = (
    "For a fixed, angled camera. Trace the track lines before the analysis;\n"
    "the camera pose is calibrated from them and the push angle is recomputed per\n"
    "frame onto the real ice plane instead of the distorted image plane.\n"
    "Bonus: speed and stroke length in the table.\n"
    "\n"
    "Needed: at least 2 track lines + 1 cross line (2+1 only with a given\n"
    "focal length; otherwise 3+1 or 2+2), and a camera that doesn't move.\n"
    "\n"
    "The calibration is saved with the analysis, so reopening restores the correction\n"
    "and a following clip from the same camera pose can reuse it.\n"
    "\n"
    "EXPERIMENTAL: not yet validated on real material. If filmed frontally with a\n"
    "horizontal camera, the distortion is small and you don't need this.")

PERSPECTIVE_TOOLTIP_BATCH = (
    PERSPECTIVE_TOOLTIP + "\n"
    "\n"
    "In a batch the calibration is asked ONCE and applied to all clips — after all,\n"
    "they come from the same camera pose. That's also the condition for comparing\n"
    "their angles against each other.")


def _calibration_rows(inst):
    """Info rows about the saved perspective calibration (empty if there isn't one).

    Shows the input (number of lines, line distance, method, lower-leg length) and the
    *recomputed* outcome (f, camera height, residual). The latter doesn't come from
    storage but is worked out again here -- exactly like when the analysis is opened --
    so the Info dialog shows what the analysis would actually use *now*.

    Reads `inst["perspective"]`, falling back to the old `inst["perspectief"]` spelling
    for an analysis saved before this dual-read was added (Pattern E -- see
    TRANSLATION_PROGRESS.md's "Deferred to Phase 8" section: `perspective` is the
    settings-json *storage* key, unlike `perspectief=` the keyword argument into
    `analyze()`, which stays Dutch permanently and is a separate concern from this
    dict key). The *nested* calibration-input dict is read through
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
    p = inst.get("perspective") or inst.get("perspectief")
    if not p:
        return []
    inv_dict = p.get("calibration_input") or p.get("invoer")
    if not inv_dict:
        return []
    inv = skate_perspective.CalibrationInput.from_dict(inv_dict)
    # `method`/`methode` may still hold an old on-disk value ('onderbeen'/'beenvlak',
    # from an analysis saved before Phase 3) -- same normalization as the shim in
    # `skate_perspective.reconstruct_angle()`.
    method_raw = p.get("method", p.get("methode"))
    method = {"lower_leg": "lower-leg length (sphere intersection)",
              "onderbeen": "lower-leg length (sphere intersection)",
              "leg_plane": "leg plane (direction of travel)",
              "beenvlak": "leg plane (direction of travel)"
              }.get(method_raw, method_raw)
    lower_leg_l = p.get("lower_leg_l", p.get("onderbeen_l"))
    rows = [
        ("Calibration:", f"{len(inv.track_lines)} track lines + {len(inv.cross_lines)} cross lines, "
                        f"{inv.line_distance} m apart, "
                        f"on a {inv.image_w}×{inv.image_h} image",
         "The traced lines are saved; the camera pose is recomputed from them."),
        ("Reconstruction:", method
         + (f", lower leg {lower_leg_l * 100:.1f} cm" if lower_leg_l else ""),
         None),
    ]
    if inv.note:
        rows.append(("Calibration note:", inv.note, None))
    try:
        kal = inv.calibrate()
        rows.append(("Camera pose:",
                      f"f = {kal.f:.0f} px{' (estimated)' if kal.f_estimated else ''}, "
                      f"height {kal.camera_height:.1f} m, horizon {kal.horizon_deg:+.2f}°, "
                      f"residual {kal.residual_px:.1f} px", None))
    except Exception as e:
        rows.append(("Camera pose:", f"cannot be recomputed: {e}", None))
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


def _opname_sleutel(source):
    """Identity of a row in the recordings table: (library, id). The id alone isn't
    enough — the list shows both the shared work list and the loose videos from the
    local library, two databases whose ids both start at 1."""
    return (source["library"], source["id"])

# Display for `skate_db.file_is_local`: (text, color, explanation). Without this column
# there's no signal at all that a recording is still in the cloud — the file *is* there,
# after all, it just comes in agonizingly slowly. See `_recording_available` for why these
# numbers.
LOKAAL_WEERGAVE = {
    "lokaal": ("✓ yes", QColor(60, 140, 60),
               "This recording is on this pc: browsing and trimming run at full "
               "speed."),
    "deels":  ("⏳ partly", QColor(190, 130, 0),
               "Part of it is local, the rest isn't yet — probably the cloud folder is "
               "still downloading. Wait for that, otherwise browsing stays slow."),
    "cloud":  ("☁ no — still in the cloud", QColor(190, 60, 60),
               "This recording isn't available offline on this pc; every piece of "
               "footage must be downloaded first.\n"
               "Measured: 5 to 20 seconds per jump in the trim window, versus "
               "0.1 second when it's local.\n\n"
               "Fix: right-click the 'opnames' folder in Explorer → Google Drive "
               "→ 'Make available offline'."),
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
    "<b>Space</b> play/pause · <b>.</b> scrub forward 6× · <b>,</b> scrub backward 6× · "
    "<b>&larr;/&rarr;</b> one frame · <b>Home/End</b> start/end · mouse wheel zooms · "
    "<b>F11</b> full screen")

# The same list as plain text, for a tooltip (which doesn't understand HTML markup).
VIDEO_KEYS_TOOLTIP = (
    "Keys: space = play/pause, ← → = one frame, . and , = scrub at 6×\n"
    "for as long as you hold the key, Home/End = start/end, F11 = full screen.")


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
    (`dlg.doel_punt`, `dlg.fragments`, ...) only *after* `exec()`. Verified: the dialog
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
        self.setWindowTitle("Choose the skater to follow")
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
        uitleg = QLabel("Click the skater you want to follow. If they're small in "
                        "frame, drag a box around them instead: that way they're also "
                        "followed where detection doesn't see them yet.\n"
                        "Mouse wheel = zoom around the cursor, right-drag = pan the "
                        "image.")
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
        self.lbl_zoom = QLabel("Zoom 1.0×")
        knoppen.addWidget(self.lbl_zoom)
        knoppen.addStretch(1)
        self.btn_box = QPushButton("Follow this box")
        self.btn_box.setEnabled(False)
        self.btn_box.clicked.connect(self._confirm_box)
        knoppen.addWidget(self.btn_box)
        btn_skip = QPushButton("Follow largest skater")
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
            source = self._pix.copy(ix0, iy0, icw, ich)
            self._crop_norm = (ix0 / pw, iy0 / ph, icw / pw, ich / ph)
        else:
            source = self._pix
            self._crop_norm = (0.0, 0.0, 1.0, 1.0)
        scaled = source.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        if self._box is not None:
            x0, y0 = self._norm_to_pixmap(self._box[0], self._box[1])
            x1, y1 = self._norm_to_pixmap(self._box[2], self._box[3])
            painter = QPainter(scaled)
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            painter.drawRect(QRect(QPoint(int(x0), int(y0)), QPoint(int(x1), int(y1))))
            painter.end()
        self.label.setPixmap(scaled)
        self.lbl_zoom.setText(f"Zoom {z:.1f}×")

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
                self, "Box very small",
                f"The box is only {hoogte_px:.0f} pixels tall. Below about "
                f"{BOX_MIN_HEIGHT_PX} pixels there are no legs left to measure — not "
                f"even with the spyglass — and the analysis is almost certain to "
                f"yield nothing.\n\n"
                f"Tip: start the fragment later, at the moment the skater is bigger "
                f"in frame.\n\nContinue with this box anyway?",
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
        self.setWindowTitle("Set the horizon / ice line")
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self._frame = frame_bgr
        self._points = []            # original-pixel (x, y) of the reference line
        self._scaled_size = None
        self._scale = 1.0

        v = QVBoxLayout(self)
        uitleg = QLabel(
            "Click two points along the ice (or the boarding/ad board) to determine\n"
            "the camera tilt, or have it detected automatically.\n"
            "Click again to redraw the line.")
        uitleg.setWordWrap(True)      # see TargetPicker: no dialog width from one text line
        v.addWidget(uitleg)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._click
        v.addWidget(self.label, 1)

        self.lbl_hoek = QLabel("Tilt: 0.00°  (no line drawn yet)")
        self.lbl_hoek.setStyleSheet("font-weight: bold;")
        v.addWidget(self.lbl_hoek)

        self.chk_per_frame = QCheckBox(
            "Detect automatically per frame (for a wobbling camera) (doesn't work)")
        self.chk_per_frame.setToolTip(
            "DOESN'T WORK / not in use: since July 2026 the camera always stands\n"
            "exactly horizontal, so there's no tilt to track per frame.\n"
            "Leave this option off.")
        self.chk_per_frame.stateChanged.connect(self._toggle_per_frame)
        v.addWidget(self.chk_per_frame)

        knoppen = QHBoxLayout()
        self.btn_auto = QPushButton("Detect (this frame)")
        self.btn_auto.clicked.connect(self._detect)
        knoppen.addWidget(self.btn_auto)
        btn_geen = QPushButton("No tilt (0°)")
        btn_geen.clicked.connect(self._no_tilt)
        knoppen.addWidget(btn_geen)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Confirm")
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
            self.lbl_hoek.setText(f"Tilt: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Tilt: click the second point …")
        self._render()

    def _detect(self):
        degrees = detect_ice_line(self._frame)
        if degrees is None:
            QMessageBox.information(
                self, "No ice line found",
                "Could not detect a reliable horizontal line. Draw the line "
                "manually, or choose 'No tilt'.")
            return
        # Synthesize a display line straight across the frame at the found angle.
        w, h = self._orig_w, self._orig_h
        cx, cy = w / 2.0, h / 2.0
        slope = np.tan(np.radians(degrees))            # y drops to the right at a positive angle
        self._points = [(0.0, cy + slope * cx), (float(w), cy - slope * (w - cx))]
        self.horizon_deg = degrees
        self.lbl_hoek.setText(f"Tilt: {degrees:+.2f}°  (automatic — check the line)")
        self._render()

    def _toggle_per_frame(self, _state):
        """With per-frame auto, the manual/constant line doesn't apply."""
        on = self.chk_per_frame.isChecked()
        self.label.setEnabled(not on)
        self.btn_auto.setEnabled(not on)
        if on:
            self.lbl_hoek.setText("Tilt: automatic per frame — "
                                  "determined during the analysis.")
        elif len(self._points) == 2:
            self.lbl_hoek.setText(f"Tilt: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Tilt: 0.00°  (no line drawn yet)")

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
        self.setWindowTitle("Perspective calibration: trace the track lines")
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
            "Trace each line with two clicks. Track lines: parallel to the direction "
            "of travel\n(order doesn't matter). Cross lines: perpendicular to them "
            "(start/finish line, corner marking). Trace lines as long as possible — "
            "that's more accurate.")
        uitleg.setWordWrap(True)      # see TargetPicker: no dialog width from one text line
        links.addWidget(uitleg)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 400)
        self.label.mousePressEvent = self._click
        links.addWidget(self.label, 1)
        hoofd.addLayout(links, 1)

        rechts = QVBoxLayout()

        soort_groep = QGroupBox("Line type (for the next line)")
        sv = QVBoxLayout(soort_groep)
        self.radio_rij = QRadioButton("Track line (direction of travel)")
        self.radio_dwars = QRadioButton("Cross line (perpendicular to the track)")
        self.radio_rij.setChecked(True)
        sv.addWidget(self.radio_rij)
        sv.addWidget(self.radio_dwars)
        rechts.addWidget(soort_groep)

        knoppen_lijn = QHBoxLayout()
        btn_wis_laatste = QPushButton("Clear last line")
        btn_wis_laatste.clicked.connect(self._undo_last)
        btn_wis_alles = QPushButton("Clear all")
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
        self.chk_alleen_hoeken = QCheckBox("Angles only (no speed/stroke length)")
        self.chk_alleen_hoeken.setChecked(True)
        self.chk_alleen_hoeken.setToolTip(
            "The push angle is scale-free: it comes only from the directions of the "
            "lines,\nnot from their distance. So you don't need to know or measure "
            "any distance at all.\n"
            "\n"
            "Turn off only if you want speed (m/s) and stroke length (m) in the "
            "table, or\nif you want to use the 'lower-leg length' reconstruction "
            "method — that one\ncomputes with a length in real meters and so needs a "
            "real line distance.")
        self.chk_alleen_hoeken.toggled.connect(self._scale_changed)
        rechts.addWidget(self.chk_alleen_hoeken)

        vorm = QFormLayout()
        self.spin_lijnafstand = QDoubleSpinBox()
        self.spin_lijnafstand.setRange(0.5, 30.0)
        self.spin_lijnafstand.setSingleStep(0.5)
        self.spin_lijnafstand.setValue(skate_perspective.DEFAULT_LINE_DISTANCE)
        self.spin_lijnafstand.setSuffix(" m")
        self.spin_lijnafstand.valueChanged.connect(self._recalibrate)
        self.lbl_lijnafstand = QLabel("Distance between track lines:")
        vorm.addRow(self.lbl_lijnafstand, self.spin_lijnafstand)

        self.spin_f = QSpinBox()
        self.spin_f.setRange(0, 100000)
        self.spin_f.setValue(0)
        self.spin_f.setSpecialValueText("automatic")
        self.spin_f.setToolTip(
            "Focal length in pixels. Normally the calibration estimates it itself "
            "from the\nlines; with a (near-)frontal camera that's fundamentally not "
            "possible and it\nmust be filled in here (typically 1-2x the image width "
            "for a phone).")
        self.spin_f.valueChanged.connect(self._recalibrate)
        vorm.addRow("Focal length (px):", self.spin_f)

        self.combo_methode = QComboBox()
        self.combo_methode.addItem("Lower-leg length (sphere intersection)", "lower_leg")
        self.combo_methode.addItem("Leg plane (direction of travel)", "leg_plane")
        self.combo_methode.setToolTip(
            "How the knee depth is reconstructed. Both are experimental to\n"
            "compare; 'lower-leg length' needs the length below.")
        self.combo_methode.currentIndexChanged.connect(
            lambda _: self._scale_changed(self.chk_alleen_hoeken.isChecked()))
        vorm.addRow("Reconstruction:", self.combo_methode)

        self.spin_lengte = QDoubleSpinBox()
        self.spin_lengte.setRange(1.0, 2.30)
        self.spin_lengte.setSingleStep(0.01)
        self.spin_lengte.setValue(1.80)
        self.spin_lengte.setSuffix(" m")
        vorm.addRow("Skater's body height:", self.spin_lengte)

        self.spin_onderbeen = QDoubleSpinBox()
        self.spin_onderbeen.setRange(0.0, 70.0)
        self.spin_onderbeen.setSingleStep(0.5)
        self.spin_onderbeen.setValue(0.0)
        self.spin_onderbeen.setSuffix(" cm")
        self.spin_onderbeen.setSpecialValueText("from body height")
        self.spin_onderbeen.setToolTip(
            "Measured lower-leg length (back of the knee to the ankle bone). Leave "
            "on\n'from body height' to estimate it as 0.246 × body height.")
        self.lbl_onderbeen = QLabel("Lower-leg length:")
        vorm.addRow(self.lbl_onderbeen, self.spin_onderbeen)
        self.lbl_lengte = vorm.labelForField(self.spin_lengte)
        rechts.addLayout(vorm)

        self.lbl_status = QLabel("No lines drawn yet.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight: bold;")
        rechts.addWidget(self.lbl_status)
        rechts.addStretch(1)

        knoppen = QHBoxLayout()
        btn_annuleer = QPushButton("Cancel")
        btn_annuleer.clicked.connect(self.reject)
        knoppen.addWidget(btn_annuleer)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Confirm")
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
                self, "Calibration doesn't fit",
                f"That calibration was made on a {calibration_input.image_w}×"
                f"{calibration_input.image_h} image and this video is "
                f"{self._orig_w}×{self._orig_h}. The lines are stored in pixels, so "
                f"reusing them would place them wrong. Trace them again.")
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
                self.lbl_status.setText("Line too short — click two points further apart.")
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
        return ("\n\nTip: with 3+ track lines it's assumed they're EVENLY spaced. "
                "If they're not, use exactly 2 track lines + 2 cross lines instead "
                "— then their mutual distance no longer matters.")

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
            "Pinned to 'leg plane' because it doesn't use any length in meters.\n"
            "Turn off 'Angles only' and fill in the real line distance to be able\n"
            "to choose 'lower-leg length' (that one needs a real scale)."
            if alleen_hoeken else
            "How the knee depth is reconstructed. Both are experimental to\n"
            "compare; 'lower-leg length' needs the length below.")
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
                f"Drawn: {n_rij} track line(s), {n_dwars} cross line(s).\n"
                f"Needed: at least 2 track lines + 1 cross line "
                f"(2+1 only with a given focal length; otherwise 3+1 or 2+2).")
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
            self.lbl_status.setText(f"Calibration not yet possible: {e}{self._line_hint(n_rij)}")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        kal = self._calibration
        # Without a known scale, the camera height is in arbitrary units; showing that
        # in meters would suggest a precision that isn't there.
        hoogte = (f"camera height {kal.camera_height:.1f} m, " if kal.scale_known
                  else "")
        # With exactly 2 track lines + 2 cross lines the system is exactly determined:
        # the residual is then 0.00 px by construction and says nothing about quality —
        # showing it would read as "perfectly calibrated". A third cross line turns V2
        # into a least-squares fit and makes the residual actually informative.
        overbepaald = len(self._cross_lines) >= 3 or len(self._track_lines) >= 3
        residu = (f", residual {kal.residual_px:.1f} px" if overbepaald else "")
        tekst = (f"Calibration OK — f = {kal.f:.0f} px"
                 f"{' (estimated)' if kal.f_estimated else ''}, {hoogte}"
                 f"horizon {kal.horizon_deg:+.2f}°{residu}.")
        if not overbepaald:
            tekst += ("\nExactly enough lines: no check is possible. Draw a third "
                      "cross line to see whether the calibration checks out.")
        if not kal.scale_known:
            tekst += ("\nAngles only: those are scale-free and thus exact; speed "
                      "and stroke length stay empty.")
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
    done = Signal(object, object, object, object)    # info, resultaten, events, analysis_id
    error = Signal(str)                      # the analysis itself failed
    save_error = Signal(str)                 # only the save failed (the analysis exists)
    warning = Signal(str)                    # silent fallback in the analysis (e.g. a click hit nobody)

    def __init__(self, input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                 doel_punt=None, horizon_deg=0.0, auto_horizon=False, smooth_landmarks=True,
                 perspectief=None, library=None, schaatser_id=None, titel=None,
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
        self.library = library
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
        analysis_id = None
        if self.library is not None and self.schaatser_id is not None:
            self.status.emit("Saving to the library...")
            try:
                analysis_id = skate_db.save_analysis(
                    self.library, self.schaatser_id, self.titel, self.input_pad,
                    info, resultaten, events,
                    backend=self.backend, instellingen=self.instellingen,
                    aangemaakt_door=self.aangemaakt_door)
            except Exception as e:
                self.save_error.emit(str(e))
        self.done.emit(info, resultaten, events, analysis_id)


class BatchWorker(QThread):
    """Runs a series of analyses back-to-back in the background and automatically saves
    each video to the library. One bad clip doesn't stop the batch -- it's reported as
    failed and the rest keeps going. 'Stop after this video' requests a clean stop via
    requestInterruption() that's handled between videos (the video in progress is
    finished and saved first)."""
    task_start = Signal(int, int, str)           # index (0-based), total, titel
    progress   = Signal(int, int)                # frame_nr, total of the current video
    status     = Signal(str)                     # busy text (video copy to the library)
    task_done  = Signal(int, object)             # index, analysis_id (or None)
    task_error = Signal(int, str)                # index, error message -- the batch continues
    all_done   = Signal(list, list, list)        # succeeded titles, [(titel, message)] failed,
                                                 # [(titel, message)] warnings

    def __init__(self, taken, library, backend, aangemaakt_door=""):
        super().__init__()
        self.taken = taken
        self.library = library
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
                    _warn("The skater is never frontal in frame; the whole video "
                          "was marked as a corner and nothing was measured.")
                if self.cancelled:
                    break                        # don't start a long video copy anymore
                self.status.emit("Saving to the library...")
                analysis_id = skate_db.save_analysis(
                    self.library, taak["schaatser_id"], taak["titel"], taak["input_pad"],
                    info, resultaten, events,
                    backend=self.backend, instellingen=taak["instellingen"],
                    aangemaakt_door=self.aangemaakt_door,
                    bron_id=taak.get("bron_id"),
                    bron_start_frame=taak.get("bron_start_frame"),
                    bron_eind_frame=taak.get("bron_eind_frame"))
                geslaagd.append(taak["titel"])
                self.task_done.emit(i, analysis_id)
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
        self.setWindowTitle("Skater")
        form = QFormLayout(self)

        self.veld_naam = QLineEdit(naam)
        form.addRow("Name:", self.veld_naam)

        self.veld_jaar = QSpinBox()
        self.veld_jaar.setRange(0, 2100)
        self.veld_jaar.setSpecialValueText("—")   # 0 = not filled in
        self.veld_jaar.setValue(geboortejaar or 0)
        form.addRow("Birth year:", self.veld_jaar)

        self.veld_notities = QPlainTextEdit(notities or "")
        self.veld_notities.setFixedHeight(70)
        form.addRow("Notes:", self.veld_notities)

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
        self.setWindowTitle("New analysis")
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
        form.addRow("Skater:", self.combo_schaatser)

        rij_video = QHBoxLayout()
        knop_video = QPushButton("Choose video...")
        knop_video.clicked.connect(self._choose_video)
        self.lbl_video = QLabel("No video chosen")
        rij_video.addWidget(knop_video)
        rij_video.addWidget(self.lbl_video, stretch=1)
        form.addRow("Video:", rij_video)

        self.veld_titel = QLineEdit()
        self.veld_titel.setPlaceholderText("default: video file name")
        form.addRow("Title:", self.veld_titel)
        v.addLayout(form)

        instellingen = QGroupBox("Settings")
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
        rij_threshold.addWidget(QLabel("Weight threshold:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 0.2)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.015)
        rij_threshold.addStretch(1)
        rij_threshold.addWidget(self.spin_threshold)
        fv.addLayout(rij_threshold)

        self.chk_heavy = QCheckBox("Heavy model (more accurate, slower)")
        self.chk_heavy.setVisible(not IS_YOLO)   # only relevant for the MediaPipe backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Skip corner (faster)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(CORNER_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        self.chk_deint = QCheckBox("Filter out interlacing (combing)")
        self.chk_deint.setToolTip(DEINT_TOOLTIP)
        self.chk_deint.setEnabled(False)         # only usable once a video is chosen
        fv.addWidget(self.chk_deint)

        self.chk_perspectief = QCheckBox("Perspective correction via track lines (experimental)")
        self.chk_perspectief.setToolTip(PERSPECTIVE_TOOLTIP)
        fv.addWidget(self.chk_perspectief)

        self.chk_geen_smoothing = QCheckBox("No landmark smoothing (raw detections)")
        self.chk_geen_smoothing.setToolTip(
            "Skips cleaning up + Savitzky-Golay-smoothing the landmark tracks: the\n"
            "skeleton follows the detections exactly (may jitter), but can never lag\n"
            "behind through interpolation. Useful to see whether a lagging skeleton\n"
            "comes from the smoothing or from the detection itself.")
        fv.addWidget(self.chk_geen_smoothing)

        v.addWidget(instellingen)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setText("Start analysis")
        self._ok.setEnabled(False)               # only enabled once a video is chosen

    def _choose_video(self):
        pad, _ = QFileDialog.getOpenFileName(
            self, "Choose video", "", VIDEO_FILTER)
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
    happens per video in the collection loop (MainWindow._new_batch_analysis)."""

    def __init__(self, schaatsers, voorkeur_id=None, voorgevuld=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analyze fragments" if voorgevuld else "Batch analysis")
        self.resize(760, 500)
        self._schaatsers = schaatsers
        v = QVBoxLayout(self)

        # Choose videos + the default skater you apply to all rows at once.
        rij_top = QHBoxLayout()
        knop_videos = QPushButton("Choose videos...")
        knop_videos.clicked.connect(self._choose_videos)
        rij_top.addWidget(knop_videos)
        rij_top.addWidget(QLabel("Default skater:"))
        self.combo_standaard = QComboBox()
        for s in schaatsers:
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_standaard.addItem(tekst, s["id"])
        if voorkeur_id is not None:
            idx = self.combo_standaard.findData(voorkeur_id)
            if idx >= 0:
                self.combo_standaard.setCurrentIndex(idx)
        rij_top.addWidget(self.combo_standaard, stretch=1)
        knop_toepassen = QPushButton("Apply to all rows")
        knop_toepassen.clicked.connect(self._apply_default)
        rij_top.addWidget(knop_toepassen)
        v.addLayout(rij_top)

        # Videos + a skater (combobox) and an editable titel per row.
        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["Video", "Skater", "Title"])
        self.tabel.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tabel.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tabel.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self.tabel, stretch=1)

        knop_verwijder = QPushButton("Remove selected row")
        knop_verwijder.clicked.connect(self._remove_row)
        v.addWidget(knop_verwijder)

        # Gedeelde instellingen (dezelfde widgets/waarden als NewAnalysisDialog).
        instellingen = QGroupBox("Settings (apply to the whole batch)")
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
        rij_threshold.addWidget(QLabel("Weight threshold:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 0.2)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.015)
        rij_threshold.addStretch(1)
        rij_threshold.addWidget(self.spin_threshold)
        fv.addLayout(rij_threshold)

        self.chk_heavy = QCheckBox("Heavy model (more accurate, slower)")
        self.chk_heavy.setVisible(not IS_YOLO)   # only relevant for the MediaPipe backend
        fv.addWidget(self.chk_heavy)

        self.chk_bocht = QCheckBox("Skip corner (faster)")
        self.chk_bocht.setChecked(True)
        self.chk_bocht.setToolTip(CORNER_TOOLTIP)
        fv.addWidget(self.chk_bocht)

        # Determined per clip, not once for the whole batch: a batch can hold clips from
        # different cameras, and the answer is cheap per file.
        self.chk_deint = QCheckBox("Filter out interlacing automatically (combing)")
        self.chk_deint.setToolTip(DEINT_TOOLTIP)
        self.chk_deint.setChecked(True)
        fv.addWidget(self.chk_deint)

        self.chk_perspectief = QCheckBox("Perspective correction via track lines (experimental)")
        self.chk_perspectief.setToolTip(PERSPECTIVE_TOOLTIP_BATCH)
        fv.addWidget(self.chk_perspectief)

        self.chk_geen_smoothing = QCheckBox("No landmark smoothing (raw detections)")
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
            self._add_row(f["input_pad"], f.get("titel"), source={
                "bron_id": f.get("bron_id"),
                "bron_start_frame": f.get("bron_start_frame"),
                "bron_eind_frame": f.get("bron_eind_frame")})

    def _add_row(self, pad, titel=None, source=None):
        """One video row: pad behind the first cell, skater combo, editable titel.
        `source` (phase 8) rides along so the analysis knows which piece of which
        recording this clip comes from."""
        r = self.tabel.rowCount()
        self.tabel.insertRow(r)
        item_pad = QTableWidgetItem(os.path.basename(pad))
        item_pad.setData(Qt.UserRole, pad)                 # full path behind the row
        item_pad.setData(Qt.UserRole + 1, source)
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
            self, "Choose videos", "", VIDEO_FILTER)
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

    def __init__(self, library, titel="Choose analysis", voorkeur_schaatser_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titel)
        self.library = library

        form = QFormLayout(self)
        self.combo_schaatser = QComboBox()
        # Skip skaters with no analyses -- that way the analysis combo can never be empty.
        for s in skate_db.list_skaters(library):
            if not s["aantal_analyses"]:
                continue
            tekst = s["naam"] + (f" ({s['geboortejaar']})" if s["geboortejaar"] else "")
            self.combo_schaatser.addItem(tekst, s["id"])
        if voorkeur_schaatser_id is not None:
            idx = self.combo_schaatser.findData(voorkeur_schaatser_id)
            if idx >= 0:
                self.combo_schaatser.setCurrentIndex(idx)
        self.combo_schaatser.currentIndexChanged.connect(self._fill_analyses)
        form.addRow("Skater:", self.combo_schaatser)

        self.combo_analyse = QComboBox()
        self.combo_analyse.setMinimumWidth(380)
        form.addRow("Analysis:", self.combo_analyse)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self.accept)
        knoppen.rejected.connect(self.reject)
        form.addRow(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)
        self._ok.setText("Choose")

        self._fill_analyses()

    def _fill_analyses(self, _idx=None):
        self.combo_analyse.clear()
        sid = self.combo_schaatser.currentData()
        if sid is not None:
            for a in skate_db.list_analyses(self.library, sid):
                gem = f"{a['gem_hoek']:.1f}°" if a["gem_hoek"] is not None else "—"
                self.combo_analyse.addItem(
                    f"{a['datum']} — {a['titel']}  "
                    f"({a['aantal_afzetten']} pushes, avg {gem})", a["id"])
        self._ok.setEnabled(self.combo_analyse.count() > 0)

    @property
    def analysis_id(self):
        return self.combo_analyse.currentData()

    @property
    def schaatser_naam(self):
        sid = self.combo_schaatser.currentData()
        if sid is None:
            return ""
        # the combo text may carry the birth year too; for the heading we only want the name
        naam = next((s["naam"] for s in skate_db.list_skaters(self.library)
                     if s["id"] == sid), "")
        return naam


def _yes_no(waarde):
    return "yes" if waarde else "no"


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
        duur = (f"{sec:.1f}s" if sec < 60
                else f"{int(sec) // 60}:{int(sec) % 60:02d}")
    else:
        duur = "—"
    n = a.get("aantal_afzetten") or 0
    return f"{duur} ({n} push{'es' if n != 1 else ''})"


class AnalysisInfoDialog(QDialog):
    """
    Read-only overview of one saved analysis: which app version/backend it ran with,
    when and by whom, and with which settings.

    Why: the tracking logic changes regularly during development, so a strange
    measurement needs to be explainable ("this was still done with the old L/R fixer").
    Purely informational -- no input field. The values are selectable so a commit hash
    can be copied.
    """

    APPVERSIE_TIP = ("Git commit this analysis was run with (commit date · hash).\n"
                     "A '+' means: there were uncommitted changes to the code at that\n"
                     "time, so the hash doesn't fully describe the analysis.")

    def __init__(self, meta, schaatser_naam="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Info about this analysis")
        inst = meta.get("instellingen") or {}
        form = QFormLayout(self)

        gemaakt = meta.get("aangemaakt_op") or ""
        datum = meta.get("datum") or "—"
        if gemaakt:
            datum += f"   (saved {gemaakt})"

        # Smoothing off = the diagnostic mode (raw detections, CLI --no-smooth).
        smoothing = (f"{inst.get('smooth_n', '?')} frames"
                     if inst.get("smooth_landmarks", True) else "off (raw detections)")
        horizon = ("automatic per frame" if inst.get("auto_horizon")
                   else f"fixed {inst.get('horizon_deg', 0.0):.1f}°")
        # `heavy` chooses between pose_landmarker_heavy and _full and so only exists for
        # the MediaPipe backend; YOLO has one model and ignores the flag -- there the row
        # ("no") would only suggest a heavier model could have been chosen.
        heavy_rij = ([] if meta.get("backend") == "yolo" else
                     [("Heavy model:", _yes_no(inst.get("heavy")),
                       "MediaPipe: pose_landmarker_heavy.task instead of _full.task.")])

        # Origin (phase 8): only for a fragment cut from a recording. A loose clip has no
        # source, and then an empty row says nothing.
        herkomst_rij = []
        if meta.get("bron_id") and meta.get("bron_start_frame") is not None:
            fps = meta.get("fps") or 0
            plek = (f" ({_time_text(meta['bron_start_frame'], fps)}–"
                    f"{_time_text(meta['bron_eind_frame'], fps)})" if fps else "")
            herkomst_rij = [("From recording:",
                             f"{meta.get('bron_naam') or 'unknown'}{plek}",
                             "This fragment was trimmed from a longer training "
                             "recording with the trim window.")]

        # How the target skater was pointed out. A box switches on the spyglass in the
        # YOLO backend (the skater is followed from the box wherever the detection
        # doesn't see them), so that's a measurement-relevant difference from a click.
        if inst.get("doel_kader"):
            doel = "box drawn (spyglass on)"
        elif "doel_punt" not in inst:
            doel = "unknown (from before this feature)"
        elif inst.get("doel_punt"):
            doel = "clicked"
        else:
            doel = "largest mover"

        rijen = [
            ("Title:", meta.get("titel") or "—", None),
            ("Skater:", schaatser_naam or "—", None),
            ("Analysis date:", datum, None),
            ("Created by:", meta.get("aangemaakt_door") or "—", None),
            ("App version:", inst.get("app_version", inst.get("app_versie"))
             or "unknown (from before this feature)", self.APPVERSIE_TIP),
            ("Backend:", inst.get("backend_name", inst.get("backend_naam"))
             or meta.get("backend") or "—", None),
            ("Video:", f"{os.path.basename(meta.get('video_bestand') or '')}  —  "
                       f"{meta.get('w')}×{meta.get('h')} @ "
                       f"{(meta.get('fps') or 0):.1f} fps, "
                       f"{meta.get('totaal_frames')} frames", None),
        ] + herkomst_rij + [
            ("Manually edited:", _yes_no(meta.get("bewerkt")),
             "Were any points moved or skeletons placed with the skeleton editor?"),
            ("Smoothing:", smoothing, None),
            ("Threshold:", f"{inst.get('threshold', '?')}", None),
        ] + heavy_rij + [
            ("Target skater:", doel,
             "Click = the detection pass looks for the skater at that spot. Box = on top\n"
             "of that, the spyglass follows them from the box wherever detection doesn't\n"
             "see them (yet), e.g. because they're small in frame."),
            ("Skip corner:", _yes_no(inst.get("skip_corner", inst.get("bocht_overslaan"))), None),
            ("Interlacing filtered:",
             _yes_no(inst.get("deinterlaced")) if "deinterlaced" in inst
             else "unknown (from before this feature)",
             "Camcorder footage (1080i) weaves two moments 1/50 s apart into one\n"
             "frame. If this was on, that combing was filtered out before detection."),
            ("Horizon:", horizon, None),
            ("Perspective correction:",
             _yes_no(inst.get("perspective_used", inst.get("perspectief_gebruikt"))), None),
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

        self.label = QLabel("No video loaded")
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
        for button, tip in ((self.btn_start, "To the start (Home)"),
                          (self.btn_frame_back, "One frame back (←)"),
                          (self.btn_play, "Play / pause (space)"),
                          (self.btn_frame_forward, "One frame forward (→)"),
                          (self.btn_end, "To the end (End)")):
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
        self.lbl_speed = QLabel("Speed")
        buttons.addWidget(self.lbl_speed)
        self.combo_speed = QComboBox()
        self.combo_speed.setToolTip("Playback speed — pick a lower factor for slow motion.")
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
        self.slider.setToolTip(f"Drag to scrub through the video.\n\n{VIDEO_KEYS_TOOLTIP}")
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
        self.chk_skeleton = QCheckBox("Skeleton")
        self.chk_push_leg = QCheckBox("Push leg")
        self.chk_hud = QCheckBox("HUD")
        for chk in (self.chk_skeleton, self.chk_push_leg, self.chk_hud):
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _=None: self.show_current_frame())
            chk.setVisible(show_overlay)
            self._toggles_row.addWidget(chk)

        # Zooming in on the skater (the mouse wheel over the video also works -- see below).
        self.chk_follow = QCheckBox("Follow skater")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("Keep the skater centered in frame while zoomed in.")
        self.chk_follow.toggled.connect(self._set_zoom_follow)
        self._toggles_row.addWidget(self.chk_follow)
        self.chk_auto = QCheckBox("Automatic zoom")
        self.chk_auto.setToolTip(
            "The program picks the zoom: the skater stays fully in frame with some room "
            "around them, the whole clip long. If they skate toward the camera, the image "
            "zooms out automatically.\nWhile this is on the zoom slider is disabled; turning "
            "the mouse wheel takes over the zoom again.")
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
        self.slider_zoom.setToolTip("Zoom level. The mouse wheel over the video also works.")
        self.slider_zoom.valueChanged.connect(lambda v: self._set_zoom(v / 100.0))
        zoom_row.addWidget(self.slider_zoom)
        self.lbl_zoom = QLabel("1.0×")
        self.lbl_zoom.setFixedWidth(38)
        zoom_row.addWidget(self.lbl_zoom)
        self.btn_zoom_reset = QPushButton("Fit")
        self.btn_zoom_reset.setToolTip("Reset zoom to fit the frame.")
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
            draw_row.addWidget(QLabel("Mouse"))
            self.combo_draw = QComboBox()
            for label, mode in DRAW_MODES:
                self.combo_draw.addItem(label, mode)
            self.combo_draw.setToolTip(DRAW_TOOLTIP)
            self.combo_draw.currentIndexChanged.connect(self._set_draw_mode)
            draw_row.addWidget(self.combo_draw)
            self.btn_draw_undo = QPushButton("↶")
            self.btn_draw_undo.setMaximumWidth(TRANSPORT_BUTTON_WIDTH)
            self.btn_draw_undo.setToolTip("Remove the last stroke drawn.")
            self.btn_draw_undo.clicked.connect(self.clear_last_stroke)
            draw_row.addWidget(self.btn_draw_undo)
            self.btn_draw_clear = QPushButton("🧹")
            self.btn_draw_clear.setMaximumWidth(TRANSPORT_BUTTON_WIDTH)
            self.btn_draw_clear.setToolTip("Remove all annotations from the image.")
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
        self.label.setText("No video loaded")
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
                f"Frame {idx} could not be read — still showing {self.huidige_idx}")
        else:
            self.lbl_time.setText(f"Frame {idx} could not be read")

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
    a sync point (start frame for 'Start all') and a minimal push table.

    The table deliberately shows little -- number, leg and angle -- but does mark
    incomplete pushes: putting two analyses side by side invites comparing two angles, and
    an incomplete push is exactly the one that's systematically too steep.
    """

    def __init__(self, name, kies_callback, parent=None):
        super().__init__(parent)
        self.name = name
        self.analysis_id = None
        self.events = []
        self.sync_frame = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        kop = QHBoxLayout()
        # ElideLabel: a long name + title must not make this side (and so the main window)
        # wider than the screen -- see the class.
        self.lbl_titel = ElideLabel(f"{name} — no analysis chosen yet")
        self.lbl_titel.setStyleSheet("font-weight: bold; padding: 2px;")
        kop.addWidget(self.lbl_titel, stretch=1)
        self.btn_kies = QPushButton("Choose analysis...")
        self.btn_kies.clicked.connect(kies_callback)
        kop.addWidget(self.btn_kies)
        # Clearing is wired up from outside (see _build_compare_page): the master clock
        # must let go first, and it doesn't know this side the other way around.
        self.btn_clear = QPushButton("✕")
        self.btn_clear.setToolTip("Clear this side.")
        self.btn_clear.setEnabled(False)
        kop.addWidget(self.btn_clear)
        v.addLayout(kop)

        # No speed control of its own: the shared control at the bottom of the compare
        # page drives both sides, so the two videos never run at a different tempo. Lower
        # bound 440x180, not 320x200: at 320 wide the toggle bar breaks into three rows,
        # into two from ~435, and that row plus 20 px of picture is exactly what the
        # compare page (the tallest of the three) had too much of on a 1280x720 screen.
        # Two 440-wide sides still fit comfortably on 1280 (see skate_screentest.py).
        self.player = VideoPlayer(min_size=(440, 180), show_speed=False)
        # The HUD is drawn at fixed full-frame positions and is unreadable in a half
        # panel; can be switched back on per side.
        self.player.chk_hud.setChecked(False)
        v.addWidget(self.player, stretch=1)

        rij_sync = QHBoxLayout()
        self.btn_sync = QPushButton("⚑ Set sync here")
        self.btn_sync.setToolTip(
            "Fixes the current frame as the start point for 'Start all', so both\n"
            "videos begin at the same phase of the stroke.\n"
            "Note: sync points apply to this session and aren't saved.")
        self.btn_sync.clicked.connect(self._set_sync)
        rij_sync.addWidget(self.btn_sync)
        self.lbl_sync = QLabel("sync: frame 0")
        self.lbl_sync.setStyleSheet("color: #888;")
        rij_sync.addWidget(self.lbl_sync)
        rij_sync.addStretch(1)
        v.addLayout(rij_sync)

        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["#", "Leg", "Angle (°)"])
        self.tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.setMaximumHeight(180)
        self.tabel.cellClicked.connect(self._click_row)
        v.addWidget(self.tabel)

        self.btn_sync.setEnabled(False)

    # ── Filling / emptying ───────────────────────────────────────────────
    def show_analysis(self, analysis_id, schaatser_naam, data):
        """Takes a loaded analysis (dict from MainWindow._load_analysis_data) into use.

        Loading the same analysis again (e.g. after an edit) keeps the sync point: that
        belongs to the video, not to the loading. A *different* analysis starts over at
        frame 0."""
        zelfde = analysis_id == self.analysis_id
        self.analysis_id = analysis_id
        self.events = data["events"]
        self.sync_frame = (min(self.sync_frame, max(0, len(data["resultaten"]) - 1))
                           if zelfde else 0)
        self.lbl_titel.setText(f"{schaatser_naam} — {data['titel']}")
        self.player.load(data["info"], data["resultaten"], data["video_pad"],
                         data.get("deinterlaced", False))
        self._fill_table()
        self.btn_kies.setText("Switch...")
        self.btn_sync.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.player.go_to(self.sync_frame)
        self._show_sync_label()

    def clear(self):
        """Releases the video (needed before the media folder can be deleted)."""
        self.player.release()
        self.analysis_id = None
        self.events = []
        self.sync_frame = 0
        self.tabel.setRowCount(0)
        self.lbl_titel.setText(f"{self.name} — no analysis chosen yet")
        self.lbl_sync.setText("sync: frame 0")
        self.btn_kies.setText("Choose analysis...")
        self.btn_sync.setEnabled(False)
        self.btn_clear.setEnabled(False)

    def has_analysis(self):
        return self.analysis_id is not None and bool(self.player.resultaten)

    # ── Sync point ───────────────────────────────────────────────────────
    def _set_sync(self):
        if not self.has_analysis():
            return
        self.sync_frame = max(0, self.player.huidige_idx)
        self._show_sync_label()

    def _show_sync_label(self):
        if not self.has_analysis():
            self.lbl_sync.setText("sync: frame 0")
            return
        tijd = self.player.resultaten[self.sync_frame].tijd
        self.lbl_sync.setText(f"sync: frame {self.sync_frame}  (t={tijd:.2f}s)")

    def to_sync(self):
        if self.has_analysis():
            self.player.go_to(self.sync_frame)

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
                        f"Incomplete push ({ev.onvolledig}) — the push wasn't "
                        "finished, so this angle is too steep and not comparable.")
                self.tabel.setItem(i, kolom, item)

    def _click_row(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self.player.go_to(self.events[rij].start_frame)


# ── Trimming fragments from a long recording (phase 8) ─────────────────────────

def _time_text(frames, fps):
    """Frame number → "m:ss" (or "h:mm:ss" on a long recording)."""
    sec = int(round(frames / (fps or 30.0)))
    if sec >= 3600:
        return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60}:{sec % 60:02d}"


def _read_time(tekst, fps):
    """"m:ss", "h:mm:ss" or a number of seconds → frame number; None if it doesn't parse."""
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
    CLICKED = Signal(int)        # index into `fragments` of the clicked block (-1 = beside it)

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
            "Marked stretches of this recording.\n"
            "Green = just marked · gray = already analyzed · orange = still running · "
            "red = overlap.\nClick a block to jump to it.")
        self.totaal = 1
        self.fragments = []      # [(start, eind)] — just marked
        self.analyzed = []          # [(start, eind, label)] — from an earlier session
        self.running = None        # start frame of the not-yet-stopped fragment
        self.cursor = 0
        self.selection = -1

    def zet(self, totaal=None, fragments=None, analyzed=None, running=..., cursor=None,
            selection=None):
        """Update everything the bar shows in one call (and redraw)."""
        if totaal is not None:
            self.totaal = max(1, int(totaal))
        if fragments is not None:
            self.fragments = list(fragments)
        if analyzed is not None:
            self.analyzed = list(analyzed)
        if running is not ...:
            self.running = running
        if cursor is not None:
            self.cursor = int(cursor)
        if selection is not None:
            self.selection = int(selection)
        self.update()

    def _x(self, frame):
        return int(round(frame / self.totaal * max(1, self.width() - 1)))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.COLOR_BACKGROUND)
        h = self.height()

        for start, eind, _label in self.analyzed:
            self._blok(p, start, eind, self.COLOR_DONE, 4, h - 8)
        for i, (start, eind) in enumerate(self.fragments):
            self._blok(p, start, eind, self.COLOR_NEW, 2, h - 4)
            if i == self.selection:
                p.setPen(QPen(self.COLOR_SELECTION, 2))
                p.setBrush(Qt.NoBrush)
                x0, x1 = self._x(start), self._x(eind)
                p.drawRect(QRect(x0, 1, max(2, x1 - x0), h - 3))
        # Overlap after the blocks, so the hatching lies on top of them.
        for i, (a0, a1) in enumerate(self.fragments):
            for b0, b1 in self.fragments[i + 1:]:
                s, e = max(a0, b0), min(a1, b1)
                if s <= e:
                    self._blok(p, s, e, self.COLOR_OVERLAP, 2, h - 4)
        if self.running is not None:
            self._blok(p, self.running, max(self.running, self.cursor),
                       self.COLOR_RUNNING, 2, h - 4)

        p.setPen(QPen(self.COLOR_CURSOR, 1))
        x = self._x(self.cursor)
        p.drawLine(x, 0, x, h)

    def _blok(self, p, start, eind, kleur, y, hoogte):
        x0, x1 = self._x(start), self._x(eind)
        p.fillRect(QRect(x0, y, max(2, x1 - x0), hoogte), kleur)

    def mousePressEvent(self, event):
        frame = int(event.position().x() / max(1, self.width() - 1) * self.totaal)
        for i, (start, eind) in enumerate(self.fragments):
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

    def __init__(self, source_path, info, analyzed=(), deinterlacen=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Trim fragments — {os.path.basename(source_path)}")
        self.info = info
        self.fps = info.fps or 30.0
        self._fragments = []          # [(start, eind)] in markeervolgorde
        self._start_open = None        # start frame of the running fragment
        self._stam = os.path.splitext(os.path.basename(source_path))[0]

        # Keep it tight: this window has to fit entirely on a laptop screen (1280x800,
        # working area 752 px) -- otherwise the button bar sinks below the edge and
        # "Klaar" becomes unreachable. Every minimum below is therefore deliberately low;
        # the picture stretches on its own when there's room.
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        uitleg = QLabel("Mark the usable stretches: <b>Start</b> (S) — <b>Stop</b> (E). "
                        "The trim happens exactly on those frames.")
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)

        self.lbl_scrub = QLabel("")
        self.lbl_scrub.setStyleSheet("color: #5aaaf0;")

        self.player = VideoPlayer(min_size=(400, 200), fast_seek=True,
                                  show_overlay=False)
        self.player.on_frame_shown = self._frame_shown
        v.addWidget(self.player, stretch=1)

        # The bar hangs inside the player (under the scrub slider), so it has the same
        # width as the timeline and shifts along with it.
        self.balk = FragmentBar()
        self.balk.CLICKED.connect(self._click_bar)
        self.player.add_bottom_bar(self.balk)

        # Navigation aid: on a half-hour recording the slider is too coarse to find a
        # push again.
        rij_nav = WrapBar()
        for label, sec in (("−1 min", -60), ("−10 s", -10), ("−1 s", -1),
                           ("+1 s", 1), ("+10 s", 10), ("+1 min", 60)):
            knop = QPushButton(label)
            knop.setMaximumWidth(72)
            knop.clicked.connect(lambda _=False, s=sec: self._jump(s))
            rij_nav.addWidget(knop)
        rij_nav.addWidget(QLabel("Go to"))
        self.veld_tijd = QLineEdit()
        self.veld_tijd.setPlaceholderText("m:ss")
        self.veld_tijd.setFixedWidth(80)
        self.veld_tijd.returnPressed.connect(self._go_to_time)
        rij_nav.addWidget(self.veld_tijd)
        knop_ga = QPushButton("Go")
        knop_ga.setMaximumWidth(48)
        knop_ga.clicked.connect(self._go_to_time)
        rij_nav.addWidget(knop_ga)
        rij_nav.addWidget(self.lbl_scrub)
        rij_nav.addStretch(1)
        v.addWidget(rij_nav)

        # Marking buttons + the list of marked stretches.
        onder = QHBoxLayout()
        links = QVBoxLayout()
        self.btn_start = QPushButton("● Start usable footage  (S)")
        self.btn_start.clicked.connect(self._start_fragment)
        links.addWidget(self.btn_start)
        self.btn_stop = QPushButton("■ Stop usable footage  (E)")
        self.btn_stop.clicked.connect(self._stop_fragment)
        links.addWidget(self.btn_stop)
        self.lbl_lopend = QLabel("")
        self.lbl_lopend.setStyleSheet("color: #d89828;")
        links.addWidget(self.lbl_lopend)
        links.addStretch(1)
        onder.addLayout(links)

        rechts = QVBoxLayout()
        rechts.addWidget(QLabel("Marked fragments  (click = jump to it)"))
        self.tabel = QTableWidget(0, 4)
        self.tabel.setHorizontalHeaderLabels(["#", "From – to", "Duration", ""])
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

        if analyzed:
            info_gedaan = QLabel(
                f"Gray in the bar: {len(analyzed)} stretch(es) of this recording are "
                f"already analyzed.")
            info_gedaan.setStyleSheet("color: #888;")
            v.addWidget(info_gedaan)

        knoppen = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        knoppen.accepted.connect(self._confirm)
        knoppen.rejected.connect(self.reject)
        v.addWidget(knoppen)
        self._ok = knoppen.button(QDialogButtonBox.Ok)

        hulp = QLabel(keys_help("<b>S</b> start fragment", "<b>E</b> stop fragment",
                                   "<b>Del</b> remove fragment"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        v.addWidget(hulp)

        # Empty FrameResult list: the player wants one (slider length, time label), but
        # nothing has been analyzed yet. `box_sequence` then returns None and the zoom
        # stays manual.
        resultaten = [FrameResult(i, i / self.fps) for i in range(max(1, info.totaal))]
        self.player.load(info, resultaten, source_path, deinterlacen)
        self.balk.zet(totaal=len(resultaten),
                      analyzed=[(f["start_frame"], f["eind_frame"],
                               f["titel"] or "") for f in analyzed])
        self.player.go_to(0)
        self._refresh()

        # The same keys as everywhere else, plus S/E/Del for marking. Deliberately no
        # QShortcut anymore for those three: a letter shortcut would also fire while
        # typing in the "go to" field, and the filter leaves a focused text field alone.
        self.keys = PlayerKeys(
            self, lambda: [self.player],
            extra={Qt.Key_S: self._start_fragment, Qt.Key_E: self._stop_fragment,
                   Qt.Key_Delete: self._delete_selection},
            on_scrub=self.lbl_scrub.setText)

        # Only after everything is in place: that way the clamp in set_window_size can
        # work against a final layout (and a `resize()` is ignored once the content is
        # bigger than what's requested -- hence every minimum above being kept low).
        set_window_size(self, 1100, 720)

    # ── Marking ──────────────────────────────────────────────────────────
    def _start_fragment(self):
        if self._start_open is not None:
            return
        self._start_open = self.player.huidige_idx
        self._refresh()

    def _stop_fragment(self):
        """Closes the running fragment. A stop before the start is a mistake, not a
        fragment: better to record nothing than to trim a reversed stretch."""
        if self._start_open is None:
            return
        eind = self.player.huidige_idx
        if eind < self._start_open:
            QMessageBox.information(
                self, "Stop is before the start",
                "The end of a fragment lies before its start. Scrub further ahead and "
                "press Stop again, or start this fragment over.")
            return
        self._fragments.append((self._start_open, eind))
        self._start_open = None
        self._refresh(selection=len(self._fragments) - 1)

    def _delete_selection(self):
        rij = self.tabel.currentRow()
        if 0 <= rij < len(self._fragments):
            self._fragments.pop(rij)
            self._refresh()

    def _click_row(self, rij, _kolom=0):
        if 0 <= rij < len(self._fragments):
            self.player.go_to(self._fragments[rij][0])
            self.balk.zet(selection=rij)

    def _click_bar(self, index):
        if index < 0:
            return
        self.tabel.selectRow(index)
        self._click_row(index)

    # ── Navigation ───────────────────────────────────────────────────────
    def _jump(self, seconden):
        self.player.go_to(self.player.huidige_idx + int(round(seconden * self.fps)))

    def _go_to_time(self):
        frame = _read_time(self.veld_tijd.text(), self.fps)
        if frame is None:
            QMessageBox.information(self, "Time", "Use m:ss (for example 12:30).")
            return
        self.player.go_to(frame)

    def _frame_shown(self, idx):
        self.balk.zet(cursor=idx, running=self._start_open)
        if self._start_open is not None:
            self.lbl_lopend.setText(
                f"Running from {_time_text(self._start_open, self.fps)} — "
                f"now {_time_text(idx, self.fps)}")

    # ── Refreshing the view ──────────────────────────────────────────────
    def _refresh(self, selection=None):
        self.btn_start.setEnabled(self._start_open is None)
        self.btn_stop.setEnabled(self._start_open is not None)
        self.lbl_lopend.setText(
            "" if self._start_open is None
            else f"Running from {_time_text(self._start_open, self.fps)}")

        self.tabel.setRowCount(len(self._fragments))
        for i, (start, eind) in enumerate(self._fragments):
            duur = (eind - start + 1) / self.fps
            waarden = [str(i + 1),
                       f"{_time_text(start, self.fps)} – {_time_text(eind, self.fps)}",
                       f"{duur:.1f}s"]
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setToolTip(f"frame {start}–{eind}")
                self.tabel.setItem(i, kolom, item)
            knop = QPushButton("✕")
            knop.setMaximumWidth(30)
            knop.setToolTip("Remove this fragment")
            knop.clicked.connect(lambda _=False, r=i: self._remove(r))
            self.tabel.setCellWidget(i, 3, knop)
        if selection is not None and 0 <= selection < len(self._fragments):
            self.tabel.selectRow(selection)

        self.balk.zet(fragments=self._fragments, running=self._start_open,
                      selection=selection if selection is not None else -1)
        n = len(self._fragments)
        self._ok.setText(f"Done — analyze {n} fragment{'s' if n != 1 else ''}"
                         if n else "Done")
        self._ok.setEnabled(n > 0)

    def _remove(self, rij):
        if 0 <= rij < len(self._fragments):
            self._fragments.pop(rij)
            self._refresh()

    # ── Closing ──────────────────────────────────────────────────────────
    def _confirm(self):
        if self._start_open is not None:
            antwoord = QMessageBox.question(
                self, "Fragment still running",
                "There's still a fragment open (Start pressed, no Stop). That stretch "
                "won't be trimmed.\n\nContinue anyway with the fragments already listed?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if antwoord != QMessageBox.Yes:
                return
        self.accept()

    @property
    def fragments(self):
        """[(start_frame, eind_frame, naam)] sorted by start frame. The name doubles as
        both the clip's file name and the suggested analysis title; the start time in it
        makes it recognizable and, in practice, unique (trim_fragments dedupes the rest)."""
        return [(start, eind, f"{self._stam} {_time_text(start, self.fps).replace(':', '-')}")
                for start, eind in sorted(self._fragments)]

    def changeEvent(self, event):
        # No longer active (alt-tab, a message box in front): the key-release then never
        # arrives and scrubbing would run forever.
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.keys.stop_scrubbing()
        super().changeEvent(event)

    def done(self, resultaat):
        # Not closeEvent: a modal dialog that closes via accept()/reject() doesn't get one.
        # The video file must be released, or Windows keeps holding the recording.
        self.keys.detach()
        self.player.release()
        super().done(resultaat)


class PointsBar(QWidget):
    """
    The bar under the viewing window's timeline: one tick per saved point, at its own
    spot in the recording. The counterpart of `FragmentBar` -- which draws stretches
    (start-end), these are single moments.

    Clicking on (or right next to) a tick jumps to it; that's the fastest way back to the
    same picture, while the list beside it mainly serves to show *what* a point is.
    """
    CLICKED = Signal(int)        # index into `points` of the clicked tick (-1 = beside it)

    HEIGHT = 22
    HIT_PX = 6               # how far off a tick a click still counts
    COLOR_BACKGROUND = QColor(45, 45, 45)
    COLOR_POINT = QColor(90, 170, 240)
    COLOR_SELECTION = QColor(255, 255, 255)
    COLOR_CURSOR = QColor(240, 240, 240)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setToolTip("Saved points in this recording. Click a tick to jump to it.")
        self.totaal = 1
        self.points = []          # [(frame, label)] sorted by frame number
        self.cursor = 0
        self.selection = -1

    def zet(self, totaal=None, points=None, cursor=None, selection=None):
        """Update everything the bar shows in one call (and redraw)."""
        if totaal is not None:
            self.totaal = max(1, int(totaal))
        if points is not None:
            self.points = list(points)
        if cursor is not None:
            self.cursor = int(cursor)
        if selection is not None:
            self.selection = int(selection)
        self.update()

    def _x(self, frame):
        return int(round(frame / self.totaal * max(1, self.width() - 1)))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.COLOR_BACKGROUND)
        h = self.height()
        for i, (frame, _label) in enumerate(self.points):
            x = self._x(frame)
            kleur = self.COLOR_SELECTION if i == self.selection else self.COLOR_POINT
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
        for i, (frame, _label) in enumerate(self.points):
            afstand = abs(self._x(frame) - klik_x)
            if afstand < beste:
                dichtst, beste = i, afstand
        self.CLICKED.emit(dichtst)


class ViewSide(QWidget):
    """
    One video in the viewing window: a header with the name (and a ✕ once there are two),
    its own `VideoPlayer` with the `PointsBar` under it, and a sync row for "Start all".

    The counterpart of `CompareSide`, without analysis and without a table: nothing is
    measured here. The header and sync row are only visible when there are two sides (see
    `ViewWindow._set_mode`) -- with one video the name is already in the window title,
    and there's nothing to sync.
    """

    def __init__(self, source, info, parent=None):
        super().__init__(parent)
        self.source = source
        self.info = info
        self.fps = info.fps or 30.0
        self.sync_frame = 0
        self.points_enabled = source.get("id") is not None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        self.kop = QWidget()
        kop = QHBoxLayout(self.kop)
        kop.setContentsMargins(0, 0, 0, 0)
        self.lbl_titel = ElideLabel(source["naam"])   # long file name: shorten, don't widen
        self.lbl_titel.setStyleSheet("font-weight: bold;")
        kop.addWidget(self.lbl_titel, stretch=1)
        # Closing is wired up from outside (see ViewWindow._add_side): the clock must
        # let go first, and it doesn't know this side the other way around.
        self.btn_clear = QPushButton("✕")
        self.btn_clear.setToolTip("Close this video; the other one stays.")
        kop.addWidget(self.btn_clear)
        v.addWidget(self.kop)

        # The same flags as the trim window, plus drawing: here -- and only here -- you
        # can draw over the picture. This is the window where you look and point things
        # out; on the viewing and compare pages the left button is already taken (panning,
        # the skeleton editor), and in the trim window you're setting boundaries.
        self.player = VideoPlayer(min_size=(400, 200), fast_seek=True,
                                  show_overlay=False, show_drawing=True)
        v.addWidget(self.player, stretch=1)
        # The bar hangs inside the player, so it has the same width as the timeline.
        self.balk = PointsBar()
        self.player.add_bottom_bar(self.balk)

        self.rij_sync = QWidget()
        rij = QHBoxLayout(self.rij_sync)
        rij.setContentsMargins(0, 0, 0, 0)
        self.btn_sync = QPushButton("⚑ Set sync here")
        self.btn_sync.setToolTip(
            "Fixes the current frame as the start point for 'Start all', so both\n"
            "videos begin at the same phase of the stroke.\n"
            "Note: sync points apply to this session and aren't saved.")
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
        self.player.load(info, resultaten, source["pad"], bool(source.get("interlaced")))
        self.balk.zet(totaal=len(resultaten))
        self._show_sync_label()

    def release(self):
        self.player.release()

    # ── Sync point (per session, not saved) ──────────────────────────────
    def _set_sync(self):
        self.sync_frame = max(0, self.player.huidige_idx)
        self._show_sync_label()

    def _show_sync_label(self):
        self.lbl_sync.setText(
            f"sync: frame {self.sync_frame}  ({_time_text(self.sync_frame, self.fps)})")

    def to_sync(self):
        self.player.go_to(self.sync_frame)


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
    from the recordings list (two rows selected) or via "➕ Second video alongside..."
    (`choose_second`, a callable from MainWindow that does the file picker, the registration
    as a loose video, and the availability check -- those don't belong in this window).
    With two sides, playback runs on the same `MasterClock` as the compare page, with a
    sync point per side and one shared speed; space then drives the clock. The **points
    only exist with one video** -- with two, each video has exactly one sync point (per
    session, not saved), and as soon as a side closes with ✕ the points of the remaining
    video come back. `_set_mode` is the one place that handles that difference.

    `source` is a row from `skate_db` -- a recording from `opnames/` or a loose video from
    this pc (`bronvideo_voor_pad`); the window doesn't care which. Only when there's no
    row (`source['id'] is None`, registration failed) do the points fall away.
    """

    PANEL_WIDTH = 260

    def __init__(self, pairs, trainer_naam="", choose_second=None, parent=None):
        """`pairs` = one or two `(source, info)`; `choose_second` supplies one more (or None)
        on request, for the "Second video alongside" button."""
        super().__init__(parent)
        self.trainer_naam = trainer_naam
        self.choose_second = choose_second
        self.sides = []
        self._points_side = None      # the side the points panel is attached to
        self._points = []           # rows from bron_markering, sorted by frame number
        self._vullen = False        # suppresses itemChanged while building

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        kop = QHBoxLayout()
        self.lbl_naam = QLabel("")
        kop.addWidget(self.lbl_naam)
        self.lbl_scrub = QLabel("")
        self.lbl_scrub.setStyleSheet("color: #5aaaf0;")
        kop.addWidget(self.lbl_scrub)
        kop.addStretch(1)
        self.lbl_no_points = QLabel("points aren't saved")
        self.lbl_no_points.setStyleSheet("color: #888;")
        kop.addWidget(self.lbl_no_points)
        self.btn_tweede = QPushButton("➕ Second video alongside...")
        self.btn_tweede.setToolTip(
            "Put a second video next to this one, to view them in sync. The video then\n"
            "ends up as a 'loose video' in the recordings list. With two videos there\n"
            "are no points, but there is a sync point per video.")
        self.btn_tweede.clicked.connect(self._add_second)
        kop.addWidget(self.btn_tweede)
        self.btn_paneel = QPushButton("Hide points")
        self.btn_paneel.clicked.connect(self._toggle_panel)
        kop.addWidget(self.btn_paneel)
        btn_venster = QPushButton("Window mode (F11)")
        btn_venster.clicked.connect(self._toggle_fullscreen)
        kop.addWidget(btn_venster)
        # Full screen has no title bar, so no Windows minimize button either.
        btn_min = QPushButton("Minimize")
        btn_min.setToolTip("Send the window to the taskbar for a moment; the videos keep running.")
        btn_min.clicked.connect(self._minimize)
        kop.addWidget(btn_min)
        btn_sluit = QPushButton("Close (Esc)")
        btn_sluit.clicked.connect(self.accept)
        kop.addWidget(btn_sluit)
        v.addLayout(kop)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter_kanten = QSplitter(Qt.Horizontal)      # the videos
        self.splitter.addWidget(self.splitter_kanten)
        self.splitter.addWidget(self._build_points_panel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        v.addWidget(self.splitter, stretch=1)

        # The shared controls for two sides -- same pattern as the compare page.
        self.balk_alles = QWidget()
        balk = QHBoxLayout(self.balk_alles)
        balk.setContentsMargins(0, 0, 0, 0)
        self.btn_start_all = QPushButton("▶ Start all")
        self.btn_start_all.setToolTip(
            "Plays both videos at once from their sync point, each at its own fps.\n"
            "Space also plays both at once, but resumes where they currently are.")
        # lambda: clicked() would otherwise pass `checked=False` as vanaf_sync.
        self.btn_start_all.clicked.connect(lambda: self._start_all())
        balk.addWidget(self.btn_start_all)
        self.btn_pause_all = QPushButton("⏸ Pause all")
        self.btn_pause_all.clicked.connect(self._pause_all)
        balk.addWidget(self.btn_pause_all)
        self.btn_to_sync = QPushButton("⏮ Both to sync")
        self.btn_to_sync.clicked.connect(self._both_to_sync)
        balk.addWidget(self.btn_to_sync)
        self.chk_from_sync = QCheckBox("from sync point")
        self.chk_from_sync.setChecked(True)
        self.chk_from_sync.setToolTip(
            "Off: 'Start all' resumes where both videos currently are, without jumping\n"
            "back (space always does that).")
        balk.addWidget(self.chk_from_sync)
        balk.addWidget(QLabel("Speed"))
        self.combo_all_speed = QComboBox()
        self.combo_all_speed.setToolTip(
            "Playback speed for both videos — even if you play just one on its own, so "
            "they always run at the same speed.")
        for label, factor in SPEEDS:
            self.combo_all_speed.addItem(label, factor)
        self.combo_all_speed.setCurrentIndex(ALL_SPEED_IDX)
        self.combo_all_speed.currentIndexChanged.connect(self._set_all_speed)
        balk.addWidget(self.combo_all_speed)
        balk.addStretch(1)
        v.addWidget(self.balk_alles)

        self.hulp = QLabel("")
        self.hulp.setWordWrap(True)
        self.hulp.setStyleSheet("color: #888;")
        v.addWidget(self.hulp)

        self.clock = MasterClock(
            self, factor=lambda: self.combo_all_speed.currentData() or 1.0,
            on_done=self._stop_all)

        for source, info in pairs:
            self._add_side(source, info)

        # The default keys, plus the points keys that only exist here. The latter do
        # nothing as long as there are two videos (see _set_point etc.).
        extra = {Qt.Key_P: self._set_point, Qt.Key_Delete: self._remove_point}
        for n in range(9):
            extra[Qt.Key_1 + n] = lambda i=n: self._go_to_point(i)
        self.keys = PlayerKeys(
            self, lambda: [k.player for k in self.sides], extra=extra,
            on_play=self._keys_play,
            is_playing=lambda: self.clock.is_running() or any(k.player.is_playing() for k in self.sides),
            on_scrub=self.lbl_scrub.setText)

        self._set_mode()

        # First set a normal size, only then full screen: otherwise F11 has no sensible
        # geometry to fall back to.
        set_window_size(self, 1280, 800)
        self.setWindowState(self.windowState() | Qt.WindowFullScreen)

    # ── Sides ────────────────────────────────────────────────────────────
    def _add_side(self, source, info):
        side = ViewSide(source, info)
        side.btn_clear.clicked.connect(lambda _=False, k=side: self._remove_side(k))
        # Pressing ▶ yourself = taking over manual control: let the clock go.
        side.player.btn_play.clicked.connect(self._stop_all)
        side.balk.CLICKED.connect(self._click_bar)   # the bar is only visible with points
        self.sides.append(side)
        self.splitter_kanten.addWidget(side)
        side.player.go_to(0)
        return side

    def _remove_side(self, side):
        """✕ on a side: release that video and carry on with the other -- keeping points."""
        if len(self.sides) < 2 or side not in self.sides:
            return
        self._pause_all()
        self.sides.remove(side)
        side.release()
        side.setParent(None)
        side.deleteLater()
        self._set_mode()

    def _add_second(self):
        if self.choose_second is None or len(self.sides) != 1:
            return
        self._pause_all()
        paar = self.choose_second()
        if not paar:
            return
        self._add_side(*paar)
        self._set_mode()

    def _set_mode(self):
        """One video or two -- the one place that handles the difference.

        One video: the points panel and points bar (if the video has a row in the
        library), the player's own speed control, and the button for a second video. Two
        videos: per side a header (name + ✕) and sync row, the shared bottom bar with
        "Start all" and one speed, and no points -- the handlers stay attached to the
        keys but do nothing then."""
        twee = len(self.sides) > 1
        self.lbl_naam.setText("  |  ".join(f"<b>{k.source['naam']}</b>" for k in self.sides))
        self.setWindowTitle("Viewing — " + " | ".join(k.source["naam"] for k in self.sides))

        for side in self.sides:
            side.kop.setVisible(twee)
            side.rij_sync.setVisible(twee)
            side.player.lbl_speed.setVisible(not twee)
            side.player.combo_speed.setVisible(not twee)
        self.balk_alles.setVisible(twee)
        self.btn_tweede.setVisible(not twee and self.choose_second is not None)

        self._attach_points(None if twee else self.sides[0])
        for side in self.sides:
            side.balk.setVisible(side is self._points_side)
        if twee:
            self._set_all_speed()
            self.hulp.setText(keys_help("both videos at once",
                                           "<b>Esc</b> close"))
        else:
            punt_toetsen = (("<b>P</b> set point", "<b>1&ndash;9</b> to point",
                             "<b>Del</b> remove point") if self._points_side is not None else ())
            self.hulp.setText(keys_help(*punt_toetsen, "<b>Esc</b> close"))

    def _attach_points(self, side):
        """Attaches the points panel to this side (or to none: `None`)."""
        if side is not None and not side.points_enabled:
            side = None
        vorige, self._points_side = self._points_side, side
        if vorige is not None and vorige in self.sides:
            vorige.player.on_frame_shown = None
        aan = side is not None
        self.paneel.setVisible(aan and (self.btn_paneel.text() == "Hide points"))
        self.btn_paneel.setVisible(aan)
        # "points aren't saved" only with one video without a row: with two videos
        # there are no points anyway, and the help line already says so.
        self.lbl_no_points.setVisible(len(self.sides) == 1 and not aan)
        if not aan:
            self._points = []
            return
        side.player.on_frame_shown = self._frame_shown
        self._refresh_points()
        side.balk.zet(cursor=max(0, side.player.huidige_idx))

    # ── Playing together (two sides, MasterClock) ─────────────────────────
    def _keys_play(self, play):
        """Space: the clock with two videos, just the player with one. Space is
        play/pause and so **resumes wherever the videos are**; only the "Start all"
        button first jumps to the sync points."""
        if not play:
            self._pause_all()
        elif len(self.sides) > 1:
            self._start_all(vanaf_sync=False)
        else:
            self.sides[0].player.play()

    def _start_all(self, vanaf_sync=None):
        """`vanaf_sync`: None = what the checkbox says (the button), False = resume
        (space)."""
        if len(self.sides) < 2:
            if self.sides:
                self.sides[0].player.play()
            return
        self._pause_all()
        if vanaf_sync is None:
            vanaf_sync = self.chk_from_sync.isChecked()
        if vanaf_sync:
            # Rewinding costs a seek per side here; that wait sits up front once instead
            # of in the first tick.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for side in self.sides:
                    side.to_sync()
            finally:
                QApplication.restoreOverrideCursor()
        self.clock.start([k.player for k in self.sides])

    def _stop_all(self):
        self.clock.stop()      # pauses the players that were running under it itself

    def _pause_all(self):
        """Everything still: clock, scrubbing, and each player released. Idempotent."""
        self.clock.stop()
        self.keys.stop_scrubbing()
        for side in self.sides:
            side.player.pause()

    def _both_to_sync(self):
        self._pause_all()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for side in self.sides:
                side.to_sync()
        finally:
            QApplication.restoreOverrideCursor()

    def _set_all_speed(self, _idx=None):
        """Sets the shared speed on both players -- also for playing one alone, since two
        videos at a different tempo next to each other can't be compared."""
        idx = self.combo_all_speed.currentIndex()
        for side in self.sides:
            side.player.combo_speed.setCurrentIndex(idx)   # restarts a running timer
        self.clock.recalibrate()

    # ── Points panel ─────────────────────────────────────────────────────
    def _build_points_panel(self):
        self.paneel = QWidget()
        p = QVBoxLayout(self.paneel)
        p.setContentsMargins(6, 0, 0, 0)
        p.addWidget(QLabel("<b>Points</b>  (click = jump to it)"))

        self.tabel = QTableWidget(0, 3)
        self.tabel.setHorizontalHeaderLabels(["#", "Time", "Name"])
        kop = self.tabel.horizontalHeader()
        kop.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(2, QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.cellClicked.connect(self._click_row)
        # Only the name column is editable (the flags are set per item in
        # _refresh_points); _vullen suppresses itemChanged while building.
        self.tabel.itemChanged.connect(self._point_renamed)
        p.addWidget(self.tabel, stretch=1)

        btn_zet = QPushButton("➕ Set point  (P)")
        btn_zet.setToolTip("Remembers the frame currently in view. The point stays saved "
                           "with this recording, even after closing this window.")
        btn_zet.clicked.connect(self._set_point)
        p.addWidget(btn_zet)
        btn_weg = QPushButton("✕ Remove point  (Del)")
        btn_weg.clicked.connect(self._remove_point)
        p.addWidget(btn_weg)

        self.lbl_points = QLabel("")
        self.lbl_points.setStyleSheet("color: #888;")
        self.lbl_points.setWordWrap(True)
        p.addWidget(self.lbl_points)

        self.paneel.setMinimumWidth(self.PANEL_WIDTH)
        return self.paneel

    # ── Points (saved in the library) ────────────────────────────────────
    # Every handler below operates on `_points_side` and does nothing if it's absent -- with
    # two videos, or with a loose video that couldn't be registered.
    @property
    def library(self):
        return self._points_side.source.get("library") if self._points_side else None

    @property
    def source(self):
        return self._points_side.source if self._points_side else None

    def _refresh_points(self, selection=None):
        """Re-reads the points from the database and fills the table + bar. The database
        is the truth: that way a point is never shown that isn't saved."""
        side = self._points_side
        if side is None:
            return
        try:
            self._points = skate_db.list_markings(self.library, self.source["id"])
        except Exception as e:
            self._points = []
            self.lbl_points.setText(f"Points could not be read: {e}")
            return
        self._vullen = True
        try:
            self.tabel.setRowCount(len(self._points))
            for rij, punt in enumerate(self._points):
                nr = QTableWidgetItem(str(rij + 1))
                nr.setFlags(nr.flags() & ~Qt.ItemIsEditable)
                self.tabel.setItem(rij, 0, nr)

                tijd = QTableWidgetItem(_time_text(punt["frame"], side.fps))
                tijd.setFlags(tijd.flags() & ~Qt.ItemIsEditable)
                tijd.setToolTip(f"frame {punt['frame']}")
                self.tabel.setItem(rij, 1, tijd)

                naam = QTableWidgetItem(punt["label"] or "")
                naam.setData(Qt.UserRole, punt["id"])
                naam.setToolTip("Double-click to rename"
                                + (f" · set by {punt['aangemaakt_door']}"
                                   if punt["aangemaakt_door"] else ""))
                self.tabel.setItem(rij, 2, naam)
        finally:
            self._vullen = False

        side.balk.zet(points=[(p["frame"], p["label"]) for p in self._points],
                      selection=selection if selection is not None else -1)
        n = len(self._points)
        self.lbl_points.setText(
            "No points set yet." if not n
            else f"{n} point{'s' if n != 1 else ''} saved with this recording.")
        if selection is not None and 0 <= selection < n:
            self.tabel.selectRow(selection)

    def _set_point(self):
        side = self._points_side
        if side is None:
            return
        frame = side.player.huidige_idx
        if frame < 0:
            return
        if any(p["frame"] == frame for p in self._points):
            self.lbl_points.setText("There's already a point on this frame.")
            return
        try:
            skate_db.add_marking(self.library, self.source["id"], frame,
                                          f"Point {len(self._points) + 1}",
                                          self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Point", f"The point could not be saved:\n{e}")
            return
        # Re-read and only then select: the list is ordered by frame number, so a point
        # you set halfway back doesn't end up at the bottom.
        self._refresh_points()
        index = next((i for i, punt in enumerate(self._points)
                      if punt["frame"] == frame), None)
        if index is not None:
            self.tabel.selectRow(index)
            side.balk.zet(selection=index)

    def _remove_point(self):
        if self._points_side is None:
            return
        rij = self.tabel.currentRow()
        if not 0 <= rij < len(self._points):
            return
        try:
            skate_db.delete_marking(self.library, self._points[rij]["id"])
        except Exception as e:
            QMessageBox.warning(self, "Point", f"The point could not be removed:\n{e}")
            return
        self._refresh_points()

    def _point_renamed(self, item):
        if self._vullen or item.column() != 2 or self._points_side is None:
            return
        try:
            skate_db.edit_marking(self.library, item.data(Qt.UserRole), label=item.text())
        except Exception as e:
            QMessageBox.warning(self, "Point", f"The name could not be saved:\n{e}")

    def _go_to_point(self, index):
        side = self._points_side
        if side is not None and 0 <= index < len(self._points):
            side.player.go_to(self._points[index]["frame"])
            self.tabel.selectRow(index)
            side.balk.zet(selection=index)

    def _click_row(self, rij, _kolom=0):
        self._go_to_point(rij)

    def _click_bar(self, index):
        if index >= 0:
            self.tabel.selectRow(index)
            self._go_to_point(index)

    # ── Display ──────────────────────────────────────────────────────────
    def _frame_shown(self, idx):
        if self._points_side is not None:
            self._points_side.balk.zet(cursor=idx)

    def _toggle_panel(self):
        zichtbaar = not self.paneel.isVisible()
        self.paneel.setVisible(zichtbaar)
        self.btn_paneel.setText("Hide points" if zichtbaar else "Show points")

    def _toggle_fullscreen(self):
        toggle_fullscreen(self)

    def _minimize(self):
        # Everything still first: a video running on in the taskbar decodes for nothing.
        self._pause_all()
        self.showMinimized()

    def changeEvent(self, event):
        # If the window goes from active to inactive (alt-tab, a message box in front),
        # the key-release never arrives anymore and scrubbing would run forever.
        if (event.type() == QEvent.ActivationChange and not self.isActiveWindow()
                and hasattr(self, "keys")):
            self.keys.stop_scrubbing()
        super().changeEvent(event)

    def done(self, resultaat):
        # Not closeEvent: a modal dialog that closes via accept()/reject() doesn't get one.
        # The video files must be released, or Windows keeps holding the recording.
        self.keys.detach()
        self.clock.stop()
        for side in self.sides:
            side.release()
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

    def __init__(self, library, plan, parent=None):
        super().__init__(parent)
        self.library = library
        self.plan = plan                       # updated in the thread (reason on failure)

    def run(self):
        paden, afgebroken, fout = [], False, None
        try:
            paden = skate_db.copy_to_recordings(
                self.library, self.plan, progress_callback=self.progress.emit,
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
        self.setWindowTitle("Copy recordings to the library")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(520)
        self._worker = worker
        self._aantal = aantal
        self._monsters = []                   # (time, bytes) of the last few seconds
        self.paden, self.afgebroken, self.fout = [], False, None   # set by _on_done

        v = QVBoxLayout(self)
        self.lbl_bestand = QLabel("Preparing...")
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
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Aborts the copy. Anything already fully copied stays; "
                                 "the half-copied file is removed.")
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
        self.lbl_bestand.setText(f"File {idx + 1} of {self._aantal}: <b>{naam}</b>")
        self.balk.setValue(int(gedaan / max(1, totaal) * 1000))
        delen = [f"{_bytes_text(gedaan)} of {_bytes_text(totaal)}"]
        t0, b0 = self._monsters[0]
        if nu - t0 >= 1.5:
            snelheid = (gedaan - b0) / (nu - t0)
            if snelheid > 0:
                delen.append(f"{_bytes_text(snelheid)}/s")
                delen.append(_remaining_text((totaal - gedaan) / snelheid))
        else:
            delen.append("measuring speed...")
        self.lbl_detail.setText(" · ".join(delen))

    def _stop(self):
        self.btn_stop.setEnabled(False)
        self.lbl_detail.setText("Stopping... (the current block is being finished)")
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
    """4.3 GB / 820 MB / 12 kB."""
    n = float(n)
    for eenheid in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or eenheid == "TB":
            break
        n /= 1000.0
    if eenheid == "B":
        return f"{int(n)} B"
    tekst = f"{n:.1f}" if n < 10 else f"{n:.0f}"
    return f"{tekst} {eenheid}"


def _remaining_text(seconden):
    """"about 3 min left" -- roughly rounded, so the estimate doesn't jump on every tick."""
    if seconden < 10:
        return "almost done"
    if seconden < 60:
        return f"about {int(round(seconden / 5.0) * 5)} s left"
    minuten = int(math.ceil(seconden / 60.0))
    if minuten < 60:
        return f"about {minuten} min left"
    uren, rest = divmod(minuten, 60)
    return f"about {uren} h {rest} min left" if rest else f"about {uren} h left"


class MainWindow(QMainWindow):
    def __init__(self, melding=None):
        super().__init__()
        # Status line from the splash screen (or None): the setup below takes a few
        # seconds on a cold machine, and that's allowed to be visible.
        self._melding = melding or (lambda tekst: None)
        self.setWindowTitle("SkateAnalysis")
        set_window_size(self, 1400, 820, maximize=True)

        self.input_pad = None
        self.model_pad = DEFAULT_MODEL
        self.smooth_n = 5
        self.threshold = 0.015
        self.doel_punt = None
        self.doel_kader = None       # drawn box -> spyglass in the YOLO backend
        self.horizon_deg = 0.0
        self.auto_horizon = False
        self.perspectief = None
        self.geen_smoothing = False
        self.bocht_overslaan = True   # don't analyze/measure corner frames (checkbox in the dialog)
        self.deinterlacen = False     # comb filter for an interlaced source (determined per video)
        # video_info / resultaten / huidige_idx live on self.player (see the properties
        # below); it's created in _build_ui() and nothing before that call reads them.
        self.events = []
        self.worker = None
        self.batch_worker = None
        self.library = None            # library path (set by _set_library)
        self.lokaal = None          # the local library (loose videos), see _set_library
        self._recordings = []          # phase 8: source-video rows behind the recordings table
        self._lokaal = {}           # path -> 'local'/'partial'/'cloud' (speed probe)
        self._local_probe = None   # running LocalProbe thread
        self._clip_tmp_dir = None    # temp folder with the just-trimmed fragments
        self.clip_worker = None
        self.trainer_naam = skate_db.trainer_name()  # phase 4: passed along as aangemaakt_door
        self.analysis_id = None      # id of the analysis currently open in the library
        # Who the open analysis belongs to -- needed for the heading on the compare page
        # (and as a preference in the analysis picker); the DB only knows the id.
        self.open_analysis_skater_id = None
        self.open_analysis_skater_name = ""
        self._pending_save = None # {schaatser_id, titel, instellingen} for the worker
        self._busy = False         # is a (batch) analysis running in the background?
        self._shutting_down = False     # window is closing: worker slots should no longer act
        self._auto_show_done = True  # may the fresh analysis show itself automatically when done?
        self._analysis_warnings = []   # messages from the running analysis (shown afterwards)
        self._backend_reported = False        # has a backend fallback already been reported? (see
                                            # _warn_backend_fallback)

        # Skeleton editor (phase 3) -- the zoom/pan state lives on the VideoPlayer
        self._editor_active = False
        self._drag = None          # {'idx', 'j', 'start_lm': Landmark} during a drag
        # Undo items are typed: 'sleep' moves one landmark over a blend-out window,
        # 'skelet' places a fully manually-placed skeleton (or removes it again).
        self._undo = []
        self._redo = []
        self._manual = {}        # {frame_idx: set(landmark_idx)} -- only for the overlay marker
        self._place = None         # running placement sequence, see _start_placing

        # The key handling of the two pages with video (filled in _build_ui). As a list,
        # because `changeEvent` can be called by Qt before the window even exists.
        self._key_handlers = []

        # Compare page: one master clock for "Start all" (see MasterClock). The factor
        # comes from the shared speed combo, which only exists after _build_ui -- the
        # callable is only read when starting.
        self.clock = MasterClock(
            self, factor=lambda: self.combo_all_speed.currentData() or 1.0,
            on_done=self._stop_all)

        self._melding("Building window...")
        self._build_ui()
        self._melding("Opening library...")
        self._set_library(skate_db.library_path())

    # The VideoPlayer is the sole owner of these three; here just pass-throughs, so the
    # existing table/editor code keeps working unchanged and a silent second copy is
    # structurally impossible (a stray assignment immediately raises AttributeError).
    @property
    def resultaten(self):
        return self.player.resultaten

    @property
    def video_info(self):
        return self.player.video_info

    @property
    def huidige_idx(self):
        return self.player.huidige_idx

    # -- UI setup --------------------------------------------------------
    def _build_ui(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        self.action_library = QAction("Library", self)
        self.action_library.triggered.connect(self._back_to_start)
        toolbar.addAction(self.action_library)

        self.stack = QStackedWidget()
        self.page_start = self._build_start_page()
        self.page_analysis = self._build_analysis_page()
        self.page_compare = self._build_compare_page()
        self.stack.addWidget(self.page_start)
        self.stack.addWidget(self.page_analysis)
        self.stack.addWidget(self.page_compare)
        self.stack.setCurrentWidget(self.page_start)

        # Three permanent status-bar widgets: how many frames have a skeleton (coverage),
        # whether scrubbing is happening, and the live status of the current frame.
        # Permanent, because showMessage() overwrites the regular status-bar text and the
        # coverage must always stay readable.
        self.lbl_coverage = QLabel("")
        self.lbl_coverage.setToolTip(
            "Number of frames with a skeleton (detected or manually placed).\n"
            "Frames without a skeleton break off a push measurement -- use '✏ Edit' to "
            "fill them in by hand.")
        self.statusBar().addPermanentWidget(self.lbl_coverage)
        self.lbl_scrub = QLabel("")
        self.lbl_scrub.setStyleSheet("color: #5aaaf0; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_scrub)
        self.lbl_live = QLabel("")
        self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_live)
        self.statusBar().showMessage(
            f"Choose a skater and start or open an analysis.  ·  backend: {BACKEND_NAME}")

        # The same keys as in the view and trim windows, here for the two pages with
        # video. Two separate objects rather than one with a page-branch halfway through
        # the handling: on the compare page there are two players to drive at once and
        # space controls the master clock (otherwise the two sides drift apart within
        # seconds), and that difference belongs in the setup.
        #
        # Both must exist before the `currentChanged` hook below, since that runs through
        # `_pause_all`, which also stops a running scrub.
        self.keys_analysis = PlayerKeys(
            self, lambda: [self.player], on_scrub=self.lbl_scrub.setText,
            active=lambda: self.stack.currentWidget() is self.page_analysis)
        self.keys_compare = PlayerKeys(
            self,
            lambda: [k.player for k in (self.side_left, self.side_right)
                     if k.has_analysis()],
            on_scrub=self.lbl_scrub.setText,
            on_play=self._keys_compare_play,
            is_playing=lambda: (self.clock.is_running()
                            or self.side_left.player.is_playing()
                            or self.side_right.player.is_playing()),
            active=lambda: self.stack.currentWidget() is self.page_compare)
        self._key_handlers = [self.keys_analysis, self.keys_compare]

        # One hook instead of one at every setCurrentWidget call: a page being left
        # shouldn't keep decoding in the background.
        self.stack.currentChanged.connect(self._page_switch)
        self._only_visible_page_counts()

        # Central = the stack plus a persistent progress bar at the bottom (hidden unless
        # an analysis/batch is running). Because the bar sits outside the stack, it stays
        # visible across page switches and doesn't block the window -- so you can browse
        # while an analysis is crunching in the background.
        self.progress_row = self._build_progress_bar()
        centraal = QWidget()
        cv = QVBoxLayout(centraal)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.stack, stretch=1)
        cv.addWidget(self.progress_row)
        self.setCentralWidget(centraal)

    def _keys_compare_play(self, play):
        """Space on the compare page drives the master clock, not the two players
        separately -- two separate frame timers drift apart within seconds (see
        `_start_all`). Space is play/pause and so resumes wherever the videos are;
        only the "Start all" button jumps to the sync points first."""
        if play:
            self._start_all(vanaf_sync=False)
        else:
            self._pause_all()

    def _pause_all(self):
        """Stops any running playback (analysis page as well as both compare players),
        including a running scrub. Idempotent, so safe to call from anywhere."""
        self._stop_all()
        for keys in self._key_handlers:
            keys.stop_scrubbing()
        self.player.pause()
        for kant in (self.side_left, self.side_right):
            kant.player.pause()

    def _page_switch(self, _idx=None):
        """When leaving a page: pause everything, and turn off edit mode -- the undo
        shortcuts are window-wide, so Ctrl+Z on another page would otherwise edit an
        invisible analysis and write it to the library."""
        self._pause_all()
        if self.stack.currentWidget() is not self.page_analysis:
            if self.btn_edit.isChecked():
                self.btn_edit.setChecked(False)   # triggers _toggle_editing(False)
            self._stop_placing()                     # fail-safe: don't leave a half skeleton
            self.lbl_live.setText("")                 # no stale status from another page
            self.lbl_coverage.setText("")
        self._only_visible_page_counts()

    def _only_visible_page_counts(self):
        """Makes only the shown page count toward the window minimum.

        A QStackedLayout takes the maximum over ALL pages, including hidden ones: the
        start page is 318 px tall, but the main window demanded 643 because the (hidden)
        compare page needs 598 -- and after comparing once with two long names the window
        stayed 1489 px wide on a 1280 px screen, even back on the start page (measured
        12-9-2026). A hidden page on `Ignored` demands nothing; when shown it gets its own
        policy back. The window can therefore grow on a page switch (to the new page's
        minimum), but never for a page nobody sees."""
        huidig = self.stack.currentWidget()
        for i in range(self.stack.count()):
            pagina = self.stack.widget(i)
            if pagina is huidig:
                pagina.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            else:
                pagina.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.stack.layout().invalidate()

    def _build_start_page(self):
        """The library (phase 1): skaters on the left, their analyses on the right.
        Phase 8 adds a second tab next to it: the recordings still to be trimmed."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        titel = QLabel("SkateAnalysis — library")
        titel.setStyleSheet("font-size: 22px; font-weight: bold; padding: 4px;")
        v.addWidget(titel)

        # Two tabs instead of a third column: the recordings don't belong to anyone in
        # particular (it's a worklist for the team), so they don't hang off the skater
        # selection on the left.
        self.tabs_library = QTabWidget()
        v.addWidget(self.tabs_library, stretch=1)

        splitter = QSplitter(Qt.Horizontal)

        # Left: skaters.
        links = QWidget()
        lv = QVBoxLayout(links)
        lv.addWidget(QLabel("Skaters"))
        self.list_skaters_widget = QListWidget()
        self.list_skaters_widget.currentItemChanged.connect(lambda *_: self._refresh_analyses())
        lv.addWidget(self.list_skaters_widget, stretch=1)
        rij_s = QHBoxLayout()
        self.btn_new_skater = QPushButton("New skater...")
        self.btn_new_skater.clicked.connect(self._new_skater)
        self.btn_edit_skater = QPushButton("Edit...")
        self.btn_edit_skater.clicked.connect(self._edit_skater)
        self.btn_delete_skater = QPushButton("Delete")
        self.btn_delete_skater.clicked.connect(self._delete_skater)
        for b in (self.btn_new_skater, self.btn_edit_skater,
                  self.btn_delete_skater):
            rij_s.addWidget(b)
        lv.addLayout(rij_s)
        splitter.addWidget(links)

        # Right: analyses of the selected skater (from the events cache).
        rechts = QWidget()
        rv = QVBoxLayout(rechts)
        rv.addWidget(QLabel("Analyses  (double-click to open)"))
        self.table_analyses = QTableWidget(0, 4)
        self.table_analyses.setHorizontalHeaderLabels(["Date", "Title", "Duration", ""])
        kop = self.table_analyses.horizontalHeader()
        # Only the title stretches; date/duration/buttons get exactly what they need,
        # otherwise four buttons get squeezed into a quarter of the table width.
        kop.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(1, QHeaderView.Stretch)
        kop.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_analyses.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_analyses.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_analyses.cellDoubleClicked.connect(
            lambda *_: self._open_analysis_from_library())
        rv.addWidget(self.table_analyses, stretch=1)
        rij_a = QHBoxLayout()
        self.btn_new_analysis = QPushButton("New analysis...")
        self.btn_new_analysis.clicked.connect(self._new_analysis)
        self.btn_batch_analysis = QPushButton("Batch analysis...")
        self.btn_batch_analysis.setToolTip(
            "Choose several videos at once and analyze them one after another. You set "
            "the target skater and horizon per video up front; then the whole row runs "
            "unattended.")
        # lambda: `clicked` would otherwise pass its `checked` bool as `voorgevuld`.
        self.btn_batch_analysis.clicked.connect(lambda: self._new_batch_analysis())
        self.btn_compare = QPushButton("Compare skaters...")
        self.btn_compare.setToolTip(
            "Put two saved analyses side by side. Each video is controlled separately;\n"
            "with a sync point per side and 'Start all' they run from the same phase\n"
            "of the stroke at the same time.")
        self.btn_compare.clicked.connect(self._compare_skaters)
        # Open/Info/Rename/Delete belong to one analysis and therefore sit in the row
        # itself (see _make_row_buttons); below stay only the library-wide actions.
        for b in (self.btn_new_analysis, self.btn_batch_analysis, self.btn_compare):
            rij_a.addWidget(b)
        rij_a.addStretch(1)
        rv.addLayout(rij_a)
        splitter.addWidget(rechts)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.tabs_library.addTab(splitter, "Skaters && analyses")
        self.tabs_library.addTab(self._build_recordings_panel(), "Recordings")

        # At the bottom, top row: refresh + trainer name (sharing via a cloud folder, phase 4).
        rij_deel = QHBoxLayout()
        knop_vernieuw = QPushButton("Refresh")
        knop_vernieuw.setToolTip(
            "Re-read the library -- shows analyses that colleagues have meanwhile added\n"
            "to the shared cloud folder, without restarting the app.")
        knop_vernieuw.clicked.connect(self._refresh_library)
        rij_deel.addWidget(knop_vernieuw)
        knop_naam = QPushButton("Your name...")
        knop_naam.setToolTip(
            "Your name is stored with new analyses (created by), so in a shared\n"
            "library it's visible who made which analysis.")
        knop_naam.clicked.connect(self._choose_trainer_naam)
        rij_deel.addWidget(knop_naam)
        self.lbl_trainer = QLabel("")
        self.lbl_trainer.setStyleSheet("color: #888;")
        rij_deel.addWidget(self.lbl_trainer)
        rij_deel.addStretch(1)
        v.addLayout(rij_deel)
        self._show_trainer_naam()

        # Bottom row: the library folder (shareable via a cloud folder, see ROADMAP phase 4).
        rij_b = QHBoxLayout()
        knop_bieb = QPushButton("Library folder...")
        knop_bieb.setToolTip(
            "The folder with the database and all videos/landmarks. Put this folder in\n"
            "a synced cloud folder (Google Drive/OneDrive/Dropbox) to share the\n"
            "library with other trainers; each trainer points at the same folder.")
        knop_bieb.clicked.connect(self._choose_library_folder)
        rij_b.addWidget(knop_bieb)
        # Elide in the middle: a long Drive path otherwise made the start page 1039 px
        # wide (the folder name at the end is the informative part, so that stays put).
        self.lbl_library = ElideLabel("", Qt.ElideMiddle)
        self.lbl_library.setStyleSheet("color: #888;")
        rij_b.addWidget(self.lbl_library, stretch=1)
        v.addLayout(rij_b)

        return paneel

    def _build_recordings_panel(self):
        """Phase 8: the worklist of raw recordings from `<library>/opnames/`.

        The **folder** is the truth about which files exist (rescanned every time it's
        opened and on 'Refresh'), the **database** about what we know of them: status,
        note, and which parts have already been analyzed. Because the folder, like the
        rest of the library, lives in the shared Drive, this isn't a personal list but a
        worklist for the team."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        uitleg = QLabel(
            "Raw training recordings still to be trimmed. Put them in the "
            "<b>opnames</b> folder in the library; they'll show up here automatically "
            "(or after 'Refresh').<br>Double-click a recording to trim fragments from "
            "it, or open it with <b>View</b> to just watch -- full screen, no "
            "analysis; select <b>two</b> rows (Ctrl+click) to see them side by side."
            "<br>Want to view a video that isn't listed here, wherever it may be on "
            "this PC? Use <b>View new video</b>; it then appears as a "
            "<i>loose video</i> at the bottom of this list -- only on this PC, not for "
            "colleagues. With <b>From camera to library</b> you copy whole recordings from "
            "the camera here, with a progress bar.")
        uitleg.setWordWrap(True)
        v.addWidget(uitleg)

        self.table_recordings = QTableWidget(0, 6)
        self.table_recordings.setHorizontalHeaderLabels(
            ["Recording", "Duration", "On this PC", "Status", "Fragments / points", "Note"])
        kop = self.table_recordings.horizontalHeader()
        kop.setSectionResizeMode(0, QHeaderView.Stretch)
        for k in (1, 2, 3, 4):
            kop.setSectionResizeMode(k, QHeaderView.ResizeToContents)
        kop.setSectionResizeMode(OPNAME_KOL_NOTITIE, QHeaderView.Stretch)
        self.table_recordings.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Two rows may be selected at once: 'View' then puts them side by side.
        # Trimming, status and note keep working on the current row.
        self.table_recordings.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Double-click = trim, except on the note: there double-click is already 'edit'
        # and shouldn't also open the trim window.
        self.table_recordings.cellDoubleClicked.connect(self._recording_double_click)
        # The note can be edited in place; only that column is editable (see
        # _vul_opnames, which sets the flags per item).
        self.table_recordings.itemChanged.connect(self._recording_note_changed)
        self._filling_recordings = False   # suppresses itemChanged while building
        v.addWidget(self.table_recordings, stretch=1)

        # A wrapping bar: five buttons on one line demand ~900 px and pushed the start
        # page's minimum width from 700 to 897 px (measured with skate_screentest);
        # wrapping costs at most one extra line per added button.
        rij = WrapBar()
        self.btn_view = QPushButton("👁 View (full screen)...")
        self.btn_view.setToolTip(
            "Plays the chosen recording back exactly as it came from the camera: no\n"
            "detection, no tracking, just video. Slow down, zoom in, scrub forward/back\n"
            "at 6x with . and ,, and set points that are kept.\n"
            "Two rows selected (Ctrl+click)? Then they appear side by side, with a\n"
            "sync point per video and 'Start all' to play them in sync.")
        self.btn_view.clicked.connect(self._view_recording)
        rij.addWidget(self.btn_view)
        self.btn_clip = QPushButton("✂ Trim fragments...")
        self.btn_clip.setToolTip(
            "Open the chosen recording, mark the usable parts (start/stop), and then "
            "have them\nanalyzed in one go -- the same batch flow as 'Batch analysis...'.")
        self.btn_clip.clicked.connect(self._clip_recording)
        rij.addWidget(self.btn_clip)
        knop_map = QPushButton("Open recordings folder")
        knop_map.setToolTip("Opens the folder where the raw recordings belong.")
        knop_map.clicked.connect(self._open_recordings_folder)
        rij.addWidget(knop_map)
        self.btn_loose_video = QPushButton("🎬 View new video...")
        self.btn_loose_video.setToolTip(
            "View a video that's somewhere else on this PC -- just taken from the\n"
            "camera, received from a colleague -- without first putting it in the\n"
            "library. The same view window: full screen, slow down, zoom, scrub, and\n"
            "points that are kept -- and in the window you can put a second video\n"
            "next to it. Nothing is analyzed and nothing is copied;\n"
            "the video shows up as a 'loose video' at the bottom of the list above, so "
            "you\n"
            "find it there again next time. That's only remembered on this PC -- the "
            "path\n"
            "doesn't go into the shared library and colleagues won't see it.")
        self.btn_loose_video.clicked.connect(self._view_loose_video)
        rij.addWidget(self.btn_loose_video)
        self.btn_import = QPushButton("📥 From camera to library...")
        self.btn_import.setToolTip(
            "Copies whole recordings from the camera or memory card to the 'opnames'\n"
            "folder in the library, with a progress bar and remaining time -- so that "
            "doesn't\n"
            "have to happen outside the app in Explorer. They then show up in this list "
            "and\n"
            "Google Drive uploads them by itself.\n"
            "On a camcorder the recordings are usually in PRIVATE\\AVCHD\\BDMV\\STREAM\n"
            "(files like 00005.MTS). A file that's already there is never overwritten.")
        self.btn_import.clicked.connect(self._import_from_camera)
        rij.addWidget(self.btn_import)
        v.addWidget(rij)
        return paneel

    def _refresh_recordings(self, selecteer=None):
        """Scans `opnames/` and fills the table. Only writes if there really are new
        files (sync_source_dir), so the shared DB isn't touched at every app start of
        every trainer.

        Loose videos from this PC ("View new video") appear below it, from the
        **local** library (`_local_recordings`), marked and with the full path in the
        tooltip: the file is outside the library, so the name alone doesn't say where it
        is. Row identity is `_opname_sleutel` (library + id), since the ids of the two
        databases overlap. `selecteer` = the key that should be selected after filling
        (the video just opened); otherwise the selection stays where it was."""
        try:
            skate_db.sync_source_dir(self.library)
            opnames = skate_db.list_source_videos(self.library)
        except Exception as e:
            self.table_recordings.setRowCount(0)
            self.statusBar().showMessage(f"Could not read recordings: {e}", 6000)
            return
        opnames += self._local_recordings()
        self._recordings = opnames

        self._filling_recordings = True
        try:
            self.table_recordings.setRowCount(len(opnames))
            for rij, b in enumerate(opnames):
                ontbreekt = b["sync"] == "ontbreekt"
                naam = QTableWidgetItem(
                    b["naam"] + ("  (loose video)" if b["extern"] else "")
                    + ("  (file not found)" if ontbreekt else ""))
                naam.setData(Qt.UserRole, _opname_sleutel(b))
                naam.setFlags(naam.flags() & ~Qt.ItemIsEditable)
                uitleg = []
                if b["extern"]:
                    uitleg.append(
                        f"Loose video, opened via 'View new video':\n{b['pad']}\n"
                        "Only remembered on this PC (with the points) -- not in the shared "
                        "library, so colleagues won't see it. Disappears from the list as "
                        "long as the file isn't there.")
                if b["bijgewerkt_door"]:
                    uitleg.append(f"Status set by {b['bijgewerkt_door']}")
                if ontbreekt:
                    naam.setForeground(QColor(150, 150, 150))
                elif b["sync"] == "onvolledig":
                    uitleg.append("The cloud sync is still downloading this file.")
                if uitleg:
                    naam.setToolTip("\n\n".join(uitleg))
                self.table_recordings.setItem(rij, 0, naam)

                duur = QTableWidgetItem(
                    _time_text(b["totaal_frames"], b["fps"])
                    if (b["totaal_frames"] and b["fps"]) else "—")
                duur.setFlags(duur.flags() & ~Qt.ItemIsEditable)
                self.table_recordings.setItem(rij, 1, duur)

                # You set the status yourself: nothing is ever automatically set to
                # 'done', since the program can't know whether you consider a recording
                # finished.
                combo = QComboBox()
                combo.addItems(skate_db.SOURCE_STATUSES)
                idx = combo.findText(b["status"])
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.currentTextChanged.connect(
                    lambda tekst, source=b: self._set_recording_status(source, tekst))
                self.table_recordings.setCellWidget(rij, OPNAME_KOL_STATUS, combo)

                n_frag, n_sch = b["aantal_fragmenten"], b["aantal_schaatsers"]
                n_pt = b.get("aantal_punten", 0)
                telling = QTableWidgetItem(
                    f"{n_frag} fragment{'s' if n_frag != 1 else ''}"
                    + (f" · {n_sch} skater{'s' if n_sch != 1 else ''}" if n_frag else "")
                    + (f" · {n_pt} point{'s' if n_pt != 1 else ''}" if n_pt else ""))
                telling.setFlags(telling.flags() & ~Qt.ItemIsEditable)
                self.table_recordings.setItem(rij, OPNAME_KOL_TELLING, telling)

                notitie = QTableWidgetItem(b["notitie"] or "")
                notitie.setToolTip("Double-click to edit (e.g. 'training Aug 3, "
                                   "tempo series').")
                self.table_recordings.setItem(rij, OPNAME_KOL_NOTITIE, notitie)

                self._set_local_cell(rij, self._lokaal.get(b["pad"]))
        finally:
            self._filling_recordings = False
        gevraagd = next((rij for rij, b in enumerate(opnames)
                         if selecteer is not None and _opname_sleutel(b) == selecteer), -1)
        if gevraagd >= 0:
            self.table_recordings.selectRow(gevraagd)
            self.table_recordings.scrollToItem(self.table_recordings.item(gevraagd, 0))
        elif opnames and self.table_recordings.currentRow() < 0:
            self.table_recordings.selectRow(0)   # so 'Trim...' works right away
        self.tabs_library.setTabText(1, f"Recordings ({len(opnames)})" if opnames else "Recordings")
        self._start_local_probe(opnames)

    def _local_recordings(self):
        """The loose videos from this PC out of the local library, for under the worklist.

        First the cleanup (`migrate_loose_videos`): what an older version of the app put
        into the shared database with an absolute path moves, with its points, to the
        local one -- the user doesn't want those paths in the Drive, and a colleague only
        ever saw "file not found" from it. A loose video whose file is (now) not there
        is **not shown** but also not deleted: a USB stick that's momentarily unplugged
        or a renamed file shouldn't cost the points, and leaving such a row in the list
        is exactly what nobody wants. If reading fails, only the worklist -- the shared
        library must not get stuck on the local one."""
        if self.lokaal is None:
            return []
        try:
            skate_db.migrate_loose_videos(self.library, self.lokaal)
            return [b for b in skate_db.list_source_videos(self.lokaal, extern=True)
                    if b["sync"] != "ontbreekt"]
        except Exception as e:
            self.statusBar().showMessage(f"Could not read loose videos: {e}",
                                         6000)
            return []

    def _selected_recording(self):
        rij = self.table_recordings.currentRow()
        if rij < 0:
            return None
        item = self.table_recordings.item(rij, 0)
        return self._recording_by_key(item.data(Qt.UserRole) if item else None)

    def _selected_recordings(self):
        """All selected rows (in row order) as source dicts -- for 'View', which can put
        two of them side by side."""
        rijen = sorted({idx.row() for idx in self.table_recordings.selectedIndexes()})
        bronnen = []
        for rij in rijen:
            item = self.table_recordings.item(rij, 0)
            source = self._recording_by_key(item.data(Qt.UserRole) if item else None)
            if source is not None:
                bronnen.append(source)
        return bronnen

    def _recording_by_key(self, sleutel):
        """The source dict behind a table row (`Qt.UserRole` of the name cell), or None."""
        if sleutel is None:
            return None
        return next((b for b in getattr(self, "_recordings", [])
                     if _opname_sleutel(b) == tuple(sleutel)), None)

    def _start_local_probe(self, opnames):
        """Measures in the background which recordings are on this PC.

        Only for files that exist: for 'missing' the sync check already says so. A
        running measurement is aborted -- after a refresh the rows may differ, and a
        result from an old list no longer belongs in the table."""
        self._stop_local_probe()
        paden = [b["pad"] for b in opnames if b["sync"] != "ontbreekt"]
        if not paden:
            return
        self._local_probe = LocalProbe(paden, self)
        self._local_probe.measured.connect(self._locally_measured)
        self._local_probe.start()

    def _stop_local_probe(self):
        """Aborts a running measurement. `wait` is fine here: the thread checks the flag
        between two recordings and one sample takes at most a second."""
        proef = self._local_probe
        self._local_probe = None
        if proef is not None and proef.isRunning():
            proef.requestInterruption()
            proef.wait(3000)

    def _locally_measured(self, pad, status):
        """One result in: remember it and update the cell (the row may meanwhile be gone)."""
        self._lokaal[pad] = status
        for rij, b in enumerate(self._recordings):
            if b["pad"] == pad:
                b["lokaal"] = status
                if rij < self.table_recordings.rowCount():
                    self._set_local_cell(rij, status)
                return

    def _set_local_cell(self, rij, status):
        """Fills the 'On this PC' column. Without a result yet, a dash: the measurement is
        still running, and 'no' would then be a claim we can't make yet."""
        tekst, kleur, uitleg = LOKAAL_WEERGAVE.get(
            status, ("—", QColor(150, 150, 150), "The speed probe is still running."))
        cel = QTableWidgetItem(tekst)
        cel.setFlags(cel.flags() & ~Qt.ItemIsEditable)
        cel.setForeground(kleur)
        cel.setToolTip(uitleg)
        self.table_recordings.setItem(rij, OPNAME_KOL_LOKAAL, cel)

    def _set_recording_status(self, source, status):
        if self._filling_recordings:
            return
        try:
            skate_db.edit_source_video(source["library"], source["id"], status=status,
                                        bijgewerkt_door=self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Recording", f"Saving the status failed:\n{e}")
            return
        source["status"] = status
        self.statusBar().showMessage(f"Status → {status}", 3000)

    def _recording_double_click(self, rij, kolom):
        if kolom != OPNAME_KOL_NOTITIE:
            self._clip_recording()

    def _recording_note_changed(self, item):
        if self._filling_recordings or item.column() != OPNAME_KOL_NOTITIE:
            return
        naam_item = self.table_recordings.item(item.row(), 0)
        source = self._recording_by_key(naam_item.data(Qt.UserRole)) if naam_item else None
        if source is None:
            return
        try:
            skate_db.edit_source_video(source["library"], source["id"], notitie=item.text(),
                                        bijgewerkt_door=self.trainer_naam)
        except Exception as e:
            QMessageBox.warning(self, "Recording", f"Saving the note failed:\n{e}")

    def _recording_available(self, source):
        """Is the file there, and complete? Reports itself what's wrong and returns False.

        Both the trim window and the view window open the recording directly from disk.
        In a shared cloud folder a half-hour recording is on its way for minutes, and then
        one clean message beats a window that crashes on a half file."""
        if source["sync"] == "ontbreekt":
            QMessageBox.warning(
                self, "Recording not found",
                f"The file is no longer on disk:\n{source['pad']}\n\n"
                "If the library is a shared cloud folder, the recording may not have "
                "synced yet.")
            return False
        if source["sync"] == "onvolledig":
            QMessageBox.warning(
                self, "Recording still downloading",
                f"'{source['naam']}' is on this PC still smaller than at the colleague who "
                "added it -- the cloud sync is still working on it.\n\n"
                "Try again once the download is finished.")
            return False
        if not self._warn_not_local(source):
            return False
        # Determine this now, so both the view window and the trim window already have
        # the answer and there's no measurement of a few seconds partway through.
        self._source_interlaced(source)
        return True

    def _source_interlaced(self, source):
        """Is this recording interlaced (combing)? Measure once per recording and store
        the answer in the library: it costs a few seconds -- more on a streaming Drive --
        while the answer never changes. If the measurement fails, don't filter: better
        the raw image than touching pixels based on a failed measurement."""
        if source.get("interlaced") is not None:
            return bool(source["interlaced"])
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            uitkomst = is_interlaced(source["pad"])
        except Exception:
            uitkomst = False
        finally:
            QApplication.restoreOverrideCursor()
        try:
            if source.get("id") is not None:
                skate_db.set_source_interlaced(source["library"], source["id"], uitkomst)
        except Exception:
            pass                      # measuring worked; only remembering it didn't
        source["interlaced"] = 1 if uitkomst else 0
        return uitkomst
    def _warn_not_local(self, source):
        """Warns if the recording isn't stored offline on this PC; True = go ahead.

        The file IS there -- the cloud folder just shows it -- but the image arrives
        piece by piece. Measured on this library (24 Aug 2026, Google Drive in streaming
        mode): one jump in the trim window fetched ~40 MB and took 5 to 20 s, versus
        30-120 ms when the same recording is local. That can't be fixed in code -- the
        player already seeks precisely instead of scrubbing through (see
        SEEK_THRESHOLD_FRAMES), and those 40 MB are what ffmpeg needs to find the right
        moment in an MPEG-TS without an index. The only sensible thing is to say so,
        before anyone thinks the program has hung.

        Continuing is allowed: sometimes you just want to see the start, and whatever
        you've already viewed sits in the cloud cache and comes right back."""
        # The cache is keyed on path, so even a loose video that couldn't be registered
        # (no id) simply gets its own spot in it.
        status = self._lokaal.get(source["pad"])
        if status is None:                 # the background measurement hadn't gotten there yet
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                status = skate_db.file_is_local(source["pad"])
            except Exception:
                status = None
            finally:
                QApplication.restoreOverrideCursor()
            if status:
                self._lokaal[source["pad"]] = status
        if status not in ("cloud", "deels"):
            return True                    # local, or unmeasurable -> don't make a fuss

        antwoord = QMessageBox.warning(
            self, "Recording is still in the cloud",
            f"'{source['naam']}' is "
            + ("still only partly" if status == "deels" else "not")
            + " offline on this PC; the image is downloaded from the cloud while "
              "viewing.\n\n"
              "Browsing will therefore be slow: jumping to another moment easily "
              "costs 5 to 20 seconds, versus a tenth of a second when the recording "
              "is local. Trimming also reads through the whole recording.\n\n"
              "Better: right-click the 'opnames' folder in Explorer -> Google Drive -> "
              "'Make available offline', wait until it's downloaded, then press "
              "'Refresh'.\n\n"
              "Open it now anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return antwoord == QMessageBox.Yes

    def _view_recording(self):
        """Manually view a recording: video only, no analysis. Two selected rows appear
        side by side.

        Deliberately without the checks that trimming does (is an analysis running,
        does a skater profile already exist): nothing is measured and nothing ends up in
        the library except the points, and those belong to the recording itself."""
        bronnen = self._selected_recordings()
        if not bronnen:
            QMessageBox.information(
                self, "View recording",
                "First choose a recording in the list.\n\n"
                "Nothing there? Put your recordings in the 'opnames' folder in the "
                "library and press 'Refresh'. Want to view a video that isn't there, "
                "use 'View new video...'.")
            return
        if len(bronnen) > 2:
            QMessageBox.information(
                self, "View recording",
                "Choose one recording, or two to view them side by side.")
            return
        self._open_view_window(bronnen, "Recording")
        self._refresh_recordings()      # update the point count in the list

    def _open_view_window(self, bronnen, titel):
        """The shared part of `_view_recording` and `_view_loose_video`: check
        availability, open the video(s), and show the view window. `titel` is the
        heading of a possible error message. If one of two fails, the window doesn't
        open -- better than silently showing one video when two were requested."""
        paren = []
        for source in bronnen:
            paar = self._open_source(source, titel)
            if paar is None:
                return
            paren.append(paar)
        self._loose_added = False
        dlg = ViewWindow(paren, self.trainer_naam,
                            choose_second=self._choose_second_video, parent=self)
        show_dialog(dlg)
        if self._loose_added:
            self._refresh_recordings()  # the video added in the window, into the list

    def _open_source(self, source, titel):
        """Availability check + `video_info` for one source -> `(source, info)` or None
        (the message has then already been shown). The points go to the library the row
        belongs to: shared for a recording, local for a loose video (`source["library"]`)."""
        if not self._recording_available(source):
            return None
        try:
            return source, video_info(source["pad"])
        except Exception as e:
            QMessageBox.critical(self, titel, f"Can't open the video:\n{e}")
            return None

    def _choose_loose_video(self):
        """File picker for a video somewhere on this PC + registering it as a loose video
        (see `_view_loose_video`). Returns the source dict, or None on cancel."""
        cfg = skate_db.load_config()
        pad, _ = QFileDialog.getOpenFileName(
            self, "Choose a video to view", cfg.get("laatste_videomap", ""),
            VIDEO_FILTER)
        if not pad:
            return None
        cfg["laatste_videomap"] = os.path.dirname(pad)
        try:
            skate_db.save_config(cfg)
        except Exception:
            pass                      # remembering the folder is a convenience, not a requirement

        try:
            if self.lokaal is None:
                raise RuntimeError("the local library could not be opened at startup "
                                   "(see the log file)")
            source = skate_db.loose_video(self.library, self.lokaal, pad)
        except Exception as e:
            QMessageBox.warning(
                self, "View video",
                f"The video can be viewed, but won't be remembered and points can't "
                f"be saved right now:\n{e}")
            source = {"id": None, "library": None, "naam": os.path.basename(pad), "pad": pad,
                    "sync": None, "interlaced": None}
        return source

    def _choose_second_video(self):
        """For "➕ Second video alongside..." in the view window: the same route as a
        loose video (picker, registration, availability), but the window is already
        open. Returns `(source, info)` or None; remembers that the list needs a refresh
        afterwards."""
        source = self._choose_loose_video()
        if source is None:
            return None
        if source["id"] is not None:
            self._loose_added = True    # the row already exists, even if opening fails
        return self._open_source(source, "View video")

    def _view_loose_video(self):
        """View a video somewhere else on this PC, without copying it into the library.

        For "just watching" nothing from the library is needed -- no skater, no
        analysis, no copy -- but until now you could only get there via a row in the
        recordings list, and that by definition consists of files in `opnames/`. A clip
        fresh from the camera or from a colleague couldn't be viewed.

        The video does get a row, but in the **local** library (`skate_db.loose_video`
        -> `local_library`): the path belongs to this PC and doesn't belong in the
        shared Drive, while the points you set do need to be kept and the video should
        simply appear in the list next time instead of being sought out again via the
        file picker. That's why the list is refreshed after viewing with this row
        selected -- even if the window ultimately didn't open (cloud warning declined),
        since the row already exists by then. If registering fails (local library can't
        be opened), viewing just proceeds without points -- the database shouldn't get
        in the way there."""
        source = self._choose_loose_video()
        if source is None:
            return
        try:
            self._open_view_window([source], "View video")
        finally:
            if source["id"] is not None:
                self._refresh_recordings(selecteer=_opname_sleutel(source))

    def _import_from_camera(self):
        """Copy whole recordings from the camera/memory card to `opnames/`, inside the app.

        Until now that had to happen in Explorer, while everything that follows
        (scanning, trimming, viewing) lives here; and a 4 GB copy to a Drive folder is
        exactly the kind of wait where you want a bar with remaining time. The route:
        `copy_plan` decides in advance what will and won't happen (existing files are
        never overwritten -- fragments and points hang off them, see skate_db), a space
        check, then `CopyWorker` + `CopyDialog`, and afterwards the regular
        `_refresh_recordings`, since from that point it's a recording like any other:
        the scan registers it, Drive uploads it, colleagues see it appear."""
        cfg = skate_db.load_config()
        paden, _ = QFileDialog.getOpenFileNames(
            self, "Choose the recordings on the camera or memory card",
            cfg.get("laatste_cameramap", ""), VIDEO_FILTER)
        if not paden:
            return
        cfg["laatste_cameramap"] = os.path.dirname(paden[0])
        try:
            skate_db.save_config(cfg)
        except Exception:
            pass                      # remembering the folder is a convenience, not a requirement

        try:
            plan = skate_db.copy_plan(self.library, paden)
        except Exception as e:
            QMessageBox.critical(self, "Copying", f"Can't reach the 'opnames' folder:\n{e}")
            return
        te_doen = [i for i in plan if i["reden"] is None]
        overgeslagen = [i for i in plan if i["reden"] is not None]
        if not te_doen:
            QMessageBox.information(
                self, "Copying",
                "There's nothing to copy:\n\n" + self._copy_reasons(overgeslagen))
            return
        totaal = sum(i["bytes"] for i in te_doen)

        # Space: on a Drive folder the free space is that of the local cache/disk, and
        # a copy that runs aground at 90% costs a quarter of an hour for nothing.
        try:
            vrij = shutil.disk_usage(skate_db.recordings_path(self.library)).free
        except OSError:
            vrij = None
        if vrij is not None and totaal > vrij:
            QMessageBox.warning(
                self, "Not enough space",
                f"These recordings together are {_bytes_text(totaal)}, but the "
                f"library's disk only has {_bytes_text(vrij)} free.\n\nMake space (or "
                f"choose fewer recordings) and try again.")
            return
        if overgeslagen:
            antwoord = QMessageBox.question(
                self, "Copying",
                f"{len(te_doen)} of the {len(plan)} chosen files will be copied "
                f"({_bytes_text(totaal)}). The rest won't:\n\n"
                + self._copy_reasons(overgeslagen) + "\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if antwoord != QMessageBox.Yes:
                return

        worker = CopyWorker(self.library, plan, self)
        dlg = CopyDialog(worker, len(te_doen), self)
        worker.start()
        try:
            show_dialog(dlg)
            worker.wait()             # accept() comes from done(), so this is already finished
        finally:
            worker.deleteLater()

        gekopieerd = dlg.paden
        mislukt = [i for i in te_doen if i["reden"] is not None]   # reason set by the worker
        sleutel = None
        if gekopieerd:
            try:
                skate_db.sync_source_dir(self.library)
                sleutel = _opname_sleutel(skate_db.source_video_for_path(self.library, gekopieerd[0]))
            except Exception:
                sleutel = None
        self._refresh_recordings(selecteer=sleutel)

        regels = []
        if gekopieerd:
            regels.append(f"{len(gekopieerd)} recording(s) copied to the library "
                          f"({_bytes_text(sum(os.path.getsize(p) for p in gekopieerd))}). "
                          f"They're now in the list; if the library is in Google Drive, "
                          f"Drive uploads them by itself and colleagues will see them too.")
        if dlg.afgebroken:
            regels.append("Copying was stopped; the half-copied file has been removed.")
        if dlg.fout is not None:
            regels.append(f"Copying stopped unexpectedly:\n{dlg.fout}")
        if mislukt:
            regels.append("Failed:\n" + self._copy_reasons(mislukt))
        if not regels:
            regels.append("Nothing was copied.")
        (QMessageBox.warning if (mislukt or dlg.fout is not None)
         else QMessageBox.information)(self, "Copying", "\n\n".join(regels))

    @staticmethod
    def _copy_reasons(items):
        return "\n".join(f"• {i['naam']} — {i['reden']}" for i in items)

    def _open_recordings_folder(self):
        pad = skate_db.recordings_path(self.library)
        try:
            os.startfile(pad)                      # Windows; falls back gracefully elsewhere
        except Exception:
            QMessageBox.information(self, "Recordings folder", pad)

    def _build_analysis_page(self):
        paneel = QWidget()
        layout = QVBoxLayout(paneel)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_data_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return paneel

    def _build_compare_page(self):
        """Two analyses side by side: each side controlled separately, plus a shared
        'Start all' that runs both from their sync point at the same time."""
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        titel = QLabel("Compare skaters")
        titel.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px;")
        v.addWidget(titel)

        splitter = QSplitter(Qt.Horizontal)
        self.side_left = CompareSide(
            "Left", lambda: self._choose_compare_side(self.side_left))
        self.side_right = CompareSide(
            "Right", lambda: self._choose_compare_side(self.side_right))
        for kant in (self.side_left, self.side_right):
            splitter.addWidget(kant)
            # Pressing ▶ yourself = taking over manual control: let go of the master clock.
            kant.player.btn_play.clicked.connect(self._stop_all)
            # Clearing a side while the master clock runs: stop the clock first.
            kant.btn_clear.clicked.connect(
                lambda _=False, k=kant: (self._stop_all(), k.clear()))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        v.addWidget(splitter, stretch=1)

        balk = QHBoxLayout()
        self.btn_start_all = QPushButton("▶ Start all")
        self.btn_start_all.setToolTip(
            "Plays both videos at once from their sync point, each at its own fps.")
        # lambda: clicked() would otherwise pass `checked=False` as vanaf_sync.
        self.btn_start_all.clicked.connect(lambda: self._start_all())
        balk.addWidget(self.btn_start_all)
        self.btn_pause_all = QPushButton("⏸ Pause all")
        self.btn_pause_all.clicked.connect(self._pause_all)
        balk.addWidget(self.btn_pause_all)
        self.btn_to_sync = QPushButton("⏮ Both to sync")
        self.btn_to_sync.clicked.connect(self._both_to_sync)
        balk.addWidget(self.btn_to_sync)
        self.chk_from_sync = QCheckBox("from sync point")
        self.chk_from_sync.setChecked(True)
        self.chk_from_sync.setToolTip(
            "Off: 'Start all' resumes wherever both videos currently are, without "
            "jumping back.")
        balk.addWidget(self.chk_from_sync)

        balk.addWidget(QLabel("Speed"))
        self.combo_all_speed = QComboBox()
        self.combo_all_speed.setToolTip(
            "Playback speed on this page -- applies to 'Start all' as well as to a side "
            "you play on its own, so the videos always run at the same speed.")
        for label, factor in SPEEDS:
            self.combo_all_speed.addItem(label, factor)
        self.combo_all_speed.setCurrentIndex(ALL_SPEED_IDX)
        self.combo_all_speed.currentIndexChanged.connect(self._set_all_speed)
        balk.addWidget(self.combo_all_speed)
        self._set_all_speed()      # set both sides to the start speed right away

        balk.addStretch(1)
        v.addLayout(balk)

        # The same keys as elsewhere, but here they drive both sides at once -- that's
        # the only thing different on this page, and belongs here.
        hulp = QLabel(keys_help("both sides at once"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        v.addWidget(hulp)
        return paneel

    def _build_video_panel(self):
        """The shared VideoPlayer plus the editor parts that belong only on the analysis
        page (the compare page uses the same player, without the editor)."""
        # A modest minimum: the video stretches with the window anyway, and a high
        # minimum pushes the window minimum above the available screen height -- Qt then
        # ignores the requested window size (see set_window_size).
        self.player = VideoPlayer(min_size=(400, 240))
        self.player.on_frame_shown = self._player_frame_shown
        self.player.overlay_drawer = self._draw_handles
        self.player.on_mouse_press = self._editor_mouse_press
        self.player.on_mouse_move = self._editor_mouse_move
        self.player.on_mouse_release = self._editor_mouse_release

        self.btn_edit = QPushButton("✏ Edit")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolTip(
            "Skeleton editor: drag wrong landmark points to the right spot.\n"
            "The correction blends into neighboring frames (adjustable) and is\n"
            "saved right away.")
        self.btn_edit.toggled.connect(self._toggle_editing)
        self.player.add_control_button(self.btn_edit)

        self.btn_compare_this = QPushButton("⇄ Compare with...")
        self.btn_compare_this.setToolTip(
            "Put this analysis on the left of the compare page and choose another one "
            "next to it.")
        self.btn_compare_this.clicked.connect(self._compare_with_this)
        self.btn_compare_this.setEnabled(False)
        self.player.add_control_button(self.btn_compare_this)

        self.btn_info = QPushButton("ℹ Info...")
        self.btn_info.setToolTip(
            "With which app version, backend and settings was this analysis made?")
        # lambda: clicked() would otherwise pass `checked=False` as analysis_id.
        self.btn_info.clicked.connect(lambda: self._show_analysis_info())
        self.btn_info.setEnabled(False)
        self.player.add_control_button(self.btn_info)

        self.btn_corner_now = QPushButton("Determine corner")
        self.btn_corner_now.setToolTip(
            "For analyses from before corner detection: determine which frames are in\n"
            "the corner after the fact, and remove them from the measurement.\n"
            "\n"
            "Nothing is re-analyzed -- the landmarks of the whole clip are already\n"
            "saved, so the hip stance can be read straight from them. You'll first "
            "see\n"
            "what it does to the table before deciding.")
        self.btn_corner_now.clicked.connect(self._determine_corner_now)
        self.btn_corner_now.setEnabled(False)
        self.player.add_control_button(self.btn_corner_now)

        # Editor bar (phase 3): only visible in edit mode. Wrapping (WrapBar), because
        # with the placement buttons added it no longer fits a laptop screen on one line
        # -- and a too-wide bar pushes the window minimum above the screen height.
        self.editor_bar = WrapBar()
        self.editor_bar.addWidget(QLabel("Blend ±"))
        self.spin_uitvloei = QSpinBox()
        self.spin_uitvloei.setRange(0, 60)
        self.spin_uitvloei.setValue(8)
        self.spin_uitvloei.setSuffix(" frames")
        self.spin_uitvloei.setToolTip(
            "How far the correction blends into neighboring frames (cosine falloff).\n"
            "0 = this frame only. Stops at a detection gap.")
        self.editor_bar.addWidget(self.spin_uitvloei)
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.clicked.connect(self._undo_edit)
        self.editor_bar.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.clicked.connect(self._redo_edit)
        self.editor_bar.addWidget(self.btn_redo)
        self.btn_next_gap = QPushButton("⏭ Next gap")
        self.btn_next_gap.setToolTip(
            "Jump to the next frame without a skeleton.\n"
            "After the last gap the search starts over from the front.\n"
            "Frames in the corner are skipped -- nothing is measured there anyway.")
        self.btn_next_gap.clicked.connect(self._go_to_next_gap)
        self.editor_bar.addWidget(self.btn_next_gap)
        self.btn_create_skeleton = QPushButton("➕ Make skeleton")
        self.btn_create_skeleton.setToolTip(
            "Places a skeleton on this frame. It's taken over from the neighboring "
            "frames,\n"
            "then you drag the points into place -- just like on any other frame.\n"
            "If there's nothing to take over, the program asks for the points\n"
            "one at a time (shoulders, hips, knees, ankles).\n"
            "Only available on a frame without a detected pose that isn't in the corner.")
        self.btn_create_skeleton.clicked.connect(self._start_placing)
        self.editor_bar.addWidget(self.btn_create_skeleton)
        self.btn_restore = QPushButton("Restore original")
        self.btn_restore.setToolTip("Resets all landmarks to the original detection.")
        self.btn_restore.clicked.connect(self._restore_original)
        self.editor_bar.addWidget(self.btn_restore)
        self.lbl_editor_hint = QLabel("")
        self.lbl_editor_hint.setStyleSheet("color: #888;")
        self.editor_bar.addWidget(self.lbl_editor_hint)
        self.editor_bar.setVisible(False)
        self.player.add_bottom_bar(self.editor_bar)

        # Placement bar: only visible during a running click sequence. Separate from the
        # editor bar so the regular edit buttons don't mix with the sequence buttons.
        self.place_bar = WrapBar()
        self.lbl_place = QLabel("")
        self.lbl_place.setStyleSheet("font-weight: bold;")
        self.place_bar.addWidget(self.lbl_place)
        self.btn_place_previous = QPushButton("← Previous point")
        self.btn_place_previous.clicked.connect(self._place_previous)
        self.place_bar.addWidget(self.btn_place_previous)
        self.btn_place_skip = QPushButton("Skip →")
        self.btn_place_skip.setToolTip(
            "Skip this point -- it keeps the position from the prefill.")
        self.btn_place_skip.clicked.connect(self._place_skip)
        self.place_bar.addWidget(self.btn_place_skip)
        self.btn_place_done = QPushButton("✔ Done")
        self.btn_place_done.setToolTip(
            "Commits the skeleton. Only possible once hip, knee and ankle of both legs\n"
            "have a visible position -- every measurement rests on those.")
        self.btn_place_done.clicked.connect(self._place_done)
        self.place_bar.addWidget(self.btn_place_done)
        self.btn_place_cancel = QPushButton("✕ Cancel")
        self.btn_place_cancel.clicked.connect(self._place_cancel)
        self.place_bar.addWidget(self.btn_place_cancel)
        self.place_bar.setVisible(False)
        self.player.add_bottom_bar(self.place_bar)

        # The same line as under the trim and view windows: the keys are the same
        # everywhere, so the summary should be too (see VIDEO_KEYS_HELP).
        hulp = QLabel(keys_help("<b>Ctrl+Z / Ctrl+Y</b> undo/redo an edit"))
        hulp.setWordWrap(True)
        hulp.setStyleSheet("color: #888;")
        self.player.add_bottom_bar(hulp)

        # Shortcuts for undo/redo (only active in edit mode, see the handlers).
        QShortcut(QKeySequence.Undo, self).activated.connect(self._undo_edit)
        QShortcut(QKeySequence.Redo, self).activated.connect(self._redo_edit)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo_edit)
        return self.player

    def _build_data_panel(self):
        paneel = QSplitter(Qt.Vertical)

        tabel_groep = QGroupBox("Push angles")
        tv = QVBoxLayout(tabel_groep)
        self.tabel = QTableWidget(0, 4)
        self.tabel.setHorizontalHeaderLabels(["#", "Time (s)", "Leg", "Angle (°)"])
        self.tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.cellClicked.connect(self._click_on_row)
        tv.addWidget(self.tabel)

        self.lbl_stats = QLabel("avg — | min — | max —")
        tv.addWidget(self.lbl_stats)

        knoppen = QHBoxLayout()
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        knoppen.addWidget(self.btn_export)
        tv.addLayout(knoppen)

        paneel.addWidget(tabel_groep)

        grafiek_groep = QGroupBox("Angle over time")
        gv = QVBoxLayout(grafiek_groep)
        self.chart = QChart()
        self.chart.legend().hide()
        self.series_angle = QLineSeries()
        self.series_marker = QLineSeries()
        self.chart.addSeries(self.series_angle)
        self.chart.addSeries(self.series_marker)
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self.axis_x.setTitleText("time (s)")
        self.axis_y.setTitleText("angle (°)")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series_angle.attachAxis(self.axis_x)
        self.series_angle.attachAxis(self.axis_y)
        self.series_marker.attachAxis(self.axis_x)
        self.series_marker.attachAxis(self.axis_y)
        chart_view = QChartView(self.chart)
        gv.addWidget(chart_view)
        paneel.addWidget(grafiek_groep)

        paneel.setStretchFactor(0, 2)
        paneel.setStretchFactor(1, 1)
        return paneel

    # -- Progress bar (non-blocking, stays put across page switches) --
    def _build_progress_bar(self):
        """A thin bar at the bottom with a label + progress and (for a batch) a stop
        button. Replaces the earlier modal progress dialog that blocked the whole window."""
        balk = QWidget()
        h = QHBoxLayout(balk)
        h.setContentsMargins(10, 4, 10, 4)
        self.lbl_progress = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(260)
        self.btn_progress_stop = QPushButton("Stop after this video")
        self.btn_progress_stop.clicked.connect(self._batch_stop_requested)
        self.btn_progress_stop.hide()
        h.addWidget(self.lbl_progress, stretch=1)
        h.addWidget(self.progress_bar)
        h.addWidget(self.btn_progress_stop)
        balk.hide()
        return balk

    def _show_progress_bar(self, tekst, met_stop=False):
        self.lbl_progress.setText(tekst)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.btn_progress_stop.setEnabled(True)
        self.btn_progress_stop.setVisible(met_stop)
        self.progress_row.show()

    def _hide_progress_bar(self):
        self.progress_row.hide()

    def _set_busy(self, bezig):
        """During a running (batch) analysis, disables the buttons that could clash with
        the worker (starting a second analysis/batch, or deleting the skater the result
        is about to be saved under). Opening/playing stays deliberately usable, so
        browsing is possible while the analysis is running."""
        self._busy = bezig
        self.btn_new_analysis.setEnabled(not bezig)
        self.btn_batch_analysis.setEnabled(not bezig)
        if bezig:
            self.btn_delete_skater.setEnabled(False)
        else:
            self._refresh_analyses()   # selection-dependent buttons back to their state

    # -- Library (phase 1) ------------------------------------------------
    def _set_library(self, pad):
        """Opens (or creates) the library at `pad` and fills the lists. If the path fails
        (e.g. a vanished network folder), the app falls back to the default folder."""
        try:
            skate_db.open_db(pad)
        except skate_db.LibraryTooNew as e:
            # A shared cloud folder a colleague with a newer app has written to: don't
            # touch it (writing could break its schema), but do report it clearly.
            QMessageBox.critical(
                self, "Library is newer than this app",
                f"{e}\n\nFolder:\n{pad}\n\n"
                "Working with the default library folder for now.")
            standaard = skate_db.default_library()
            if pad != standaard:
                return self._set_library(standaard)
            raise
        except Exception as e:
            QMessageBox.critical(
                self, "Library",
                f"Can't open the library at:\n{pad}\n\n{e}")
            standaard = skate_db.default_library()
            if pad != standaard:
                return self._set_library(standaard)
            raise
        self.library = pad
        self.lbl_library.setText(pad)
        # The local library (loose videos from this PC) is separate from the shared one
        # and doesn't change when the library folder is switched: open it once. If that
        # fails, everything else keeps working except remembering loose videos.
        if self.lokaal is None:
            try:
                self.lokaal = skate_db.local_library()
            except Exception as e:
                self.statusBar().showMessage(
                    f"Local library not available (loose videos won't be "
                    f"remembered): {e}", 8000)
        self._warn_conflict_copies()
        self._refresh_skaters()
        # During startup, report the slowest step separately: for a new recording the
        # scan reads the video metadata, and on a cloud folder that can take seconds.
        self._melding("Scanning recordings...")
        self._refresh_recordings()      # phase 8: worklist of recordings still to be trimmed

    def _warn_conflict_copies(self):
        """Phase 4: warns if the cloud sync has left conflict copies of the database
        alongside schaats.db (see skate_db.detect_conflict_copies)."""
        try:
            kopieen = skate_db.detect_conflict_copies(self.library)
        except Exception:
            return
        if not kopieen:
            return
        QMessageBox.warning(
            self, "Possible database conflict copy",
            "There are other database files in the library folder next to "
            "'schaats.db':\n\n"
            "• " + "\n• ".join(kopieen) + "\n\n"
            "Such copies are created when the cloud sync (Google Drive/OneDrive/Dropbox) "
            "creates a conflict because two trainers wrote at nearly the same time. "
            "'schaats.db' stays the active library; check the copy/copies and delete or "
            "rename them to avoid confusion.")

    def _refresh_library(self):
        """Phase 4: re-reads the library from disk, so colleagues' analyses (via the
        shared cloud folder) become visible without restarting."""
        self._warn_conflict_copies()
        self._refresh_skaters()
        self._refresh_recordings()      # also pick up new recordings from colleagues
        self.statusBar().showMessage("Library refreshed.", 4000)

    def _show_trainer_naam(self):
        self.lbl_trainer.setText(
            f"you: {self.trainer_naam}" if self.trainer_naam else "you: (name not set)")

    def _choose_trainer_naam(self):
        naam, ok = QInputDialog.getText(
            self, "Your name",
            "Your name (stored with new analyses as 'created by'):",
            text=self.trainer_naam)
        if not ok:
            return
        self.trainer_naam = naam.strip()
        cfg = skate_db.load_config()
        cfg["trainer_name"] = self.trainer_naam
        skate_db.save_config(cfg)
        self._show_trainer_naam()

    def _choose_library_folder(self):
        pad = QFileDialog.getExistingDirectory(self, "Choose library folder", self.library or "")
        if not pad:
            return
        cfg = skate_db.load_config()
        cfg["library_path"] = pad
        skate_db.save_config(cfg)
        self._set_library(pad)

    def _selected_schaatser_id(self):
        item = self.list_skaters_widget.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _skater_name(self, schaatser_id):
        """Name for a skater id, or "" if it no longer exists."""
        if schaatser_id is None:
            return ""
        s = next((x for x in skate_db.list_skaters(self.library)
                  if x["id"] == schaatser_id), None)
        return s["naam"] if s else ""

    def _selected_analyse_id(self):
        rij = self.table_analyses.currentRow()
        if rij < 0:
            return None
        item = self.table_analyses.item(rij, 0)
        return item.data(Qt.UserRole) if item else None

    def _selected_titel(self):
        item = self.table_analyses.item(self.table_analyses.currentRow(), 1)
        return item.text() if item else ""

    def _refresh_skaters(self, selecteer_id=None):
        """Reloads the skater list from the database (and with it the analysis table)."""
        if selecteer_id is None:
            selecteer_id = self._selected_schaatser_id()
        self.list_skaters_widget.blockSignals(True)
        self.list_skaters_widget.clear()
        selecteer_rij = None
        schaatsers = skate_db.list_skaters(self.library)
        # Comparing works across skaters, so hang it off "is there an analysis
        # anywhere" rather than off the selection.
        self.btn_compare.setEnabled(any(s["aantal_analyses"] for s in schaatsers))
        for rij, s in enumerate(schaatsers):
            tekst = s["naam"]
            if s["geboortejaar"]:
                tekst += f" ({s['geboortejaar']})"
            n = s["aantal_analyses"]
            tekst += f"  ·  {n} analys{'e' if n == 1 else 'es'}"
            item = QListWidgetItem(tekst)
            item.setData(Qt.UserRole, s["id"])
            item.setData(Qt.UserRole + 1, s["naam"])
            if s["notities"]:
                item.setToolTip(s["notities"])
            self.list_skaters_widget.addItem(item)
            if s["id"] == selecteer_id:
                selecteer_rij = rij
        self.list_skaters_widget.blockSignals(False)
        if selecteer_rij is None and self.list_skaters_widget.count():
            selecteer_rij = 0
        if selecteer_rij is not None:
            self.list_skaters_widget.setCurrentRow(selecteer_rij)  # triggers _refresh_analyses
        else:
            self._refresh_analyses()

    def _make_row_buttons(self, analysis_id, titel):
        """The four per-analysis actions as a widget for the last table column.
        Each button carries its own analysis id (default argument in the lambda --
        otherwise the last loop value would apply to ALL rows), so a click works on the
        row it's on and not on whatever the table happens to have selected."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)
        knoppen = (
            ("Open", "Open this analysis on the view page.",
             lambda _=False, a=analysis_id: self._open_analysis_from_library(a)),
            ("ℹ Info...", "With which app version, backend and settings was this "
                          "analysis made?",
             lambda _=False, a=analysis_id: self._show_analysis_info(a)),
            ("Rename...", "Give this analysis a different title.",
             lambda _=False, a=analysis_id, t=titel: self._rename_analysis(a, t)),
            ("Delete", "Delete this analysis, including video and landmarks.",
             lambda _=False, a=analysis_id, t=titel: self._delete_analysis(a, t)),
        )
        for tekst, tip, slot in knoppen:
            b = QPushButton(tekst)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        return w

    def _refresh_analyses(self):
        """Fills the analysis table for the selected skater from the events cache
        (no npz/video needed -- that's why the library is instantly fast)."""
        sid = self._selected_schaatser_id()
        self.table_analyses.setRowCount(0)
        if sid is not None:
            analyses = skate_db.list_analyses(self.library, sid)
            self.table_analyses.setRowCount(len(analyses))
            for rij, a in enumerate(analyses):
                # Provenance in the tooltip: who made it (phase 4) and with which app
                # version -- so without opening it you can see whether an analysis was
                # still run with old code.
                door = (a.get("aangemaakt_door") or "").strip()
                inst_a = a.get("instellingen") or {}
                versie = (inst_a.get("app_version", inst_a.get("app_versie")) or "").strip()
                tip = "\n".join(r for r in (f"Created by {door}" if door else "",
                                            f"App version: {versie}" if versie else "") if r)
                for kolom, tekst in enumerate(
                        [a["datum"], a["titel"], _duration_text(a)]):
                    item = QTableWidgetItem(tekst)
                    if kolom == 0:
                        item.setData(Qt.UserRole, a["id"])
                    if kolom == 1 and tip:
                        item.setToolTip(tip)
                    self.table_analyses.setItem(rij, kolom, item)
                knoppen = self._make_row_buttons(a["id"], a["titel"])
                self.table_analyses.setCellWidget(rij, 3, knoppen)
                # Row height doesn't follow the buttons automatically; without this
                # they get squeezed.
                self.table_analyses.setRowHeight(rij, knoppen.sizeHint().height() + 4)
        self.btn_edit_skater.setEnabled(sid is not None)
        self.btn_delete_skater.setEnabled(sid is not None)

    def _new_skater(self):
        dlg = SkaterDialog(self)
        if show_dialog(dlg) != QDialog.Accepted or not dlg.naam:
            return
        sid = skate_db.create_skater(self.library, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._refresh_skaters(selecteer_id=sid)

    def _edit_skater(self):
        sid = self._selected_schaatser_id()
        if sid is None:
            return
        s = next((x for x in skate_db.list_skaters(self.library) if x["id"] == sid), None)
        if s is None:
            return
        dlg = SkaterDialog(self, naam=s["naam"], geboortejaar=s["geboortejaar"],
                          notities=s["notities"])
        if show_dialog(dlg) != QDialog.Accepted or not dlg.naam:
            return
        skate_db.edit_skater(self.library, sid, dlg.naam, dlg.geboortejaar, dlg.notities)
        self._refresh_skaters(selecteer_id=sid)

    def _delete_skater(self):
        sid = self._selected_schaatser_id()
        if sid is None:
            return
        naam = self.list_skaters_widget.currentItem().data(Qt.UserRole + 1)
        analyses = skate_db.list_analyses(self.library, sid)
        tekst = f"Delete skater '{naam}'?"
        if analyses:
            tekst += (f"\n\nThe {len(analyses)} associated analys"
                      f"{'is' if len(analyses) == 1 else 'es'} (including videos and "
                      "landmarks) will then also be deleted.")
        tekst += "\n\nThis cannot be undone."
        if QMessageBox.question(self, "Delete skater", tekst) != QMessageBox.Yes:
            return
        ids = {a["id"] for a in analyses}
        if self.analysis_id in ids:
            self._close_view()   # let go of the open video before deleting
        self._close_compare_for(ids)
        skate_db.delete_skater(self.library, sid)
        self._refresh_skaters()

    def _rename_analysis(self, aid=None, huidig=""):
        # aid/huidig come from the button in the row itself; without those two it falls
        # back to the table selection (e.g. a shortcut that gets added someday).
        if aid is None:
            aid, huidig = self._selected_analyse_id(), self._selected_titel()
        if aid is None:
            return
        titel, ok = QInputDialog.getText(self, "Rename analysis", "New title:",
                                         text=huidig)
        if not ok or not titel.strip():
            return
        skate_db.rename_analysis(self.library, aid, titel.strip())
        self._refresh_analyses()

    def _delete_analysis(self, aid=None, titel=""):
        if aid is None:
            aid, titel = self._selected_analyse_id(), self._selected_titel()
        if aid is None:
            return
        if QMessageBox.question(
                self, "Delete analysis",
                f"Delete analysis '{titel}', including the copied video and "
                "landmarks?\n\nThis cannot be undone.") != QMessageBox.Yes:
            return
        if aid == self.analysis_id:
            self._close_view()   # Windows refuses to delete a still-open video
        self._close_compare_for({aid})
        skate_db.delete_analysis(self.library, aid)
        self._refresh_skaters()

    def _close_view(self):
        """Clears the view page and lets go of the video file (needed before the media
        folder of the open analysis can be deleted)."""
        self.player.release()
        self.events = []
        self.analysis_id = None
        self.open_analysis_skater_id = None
        self.open_analysis_skater_name = ""
        self.btn_compare_this.setEnabled(False)
        self.input_pad = None
        self.tabel.setRowCount(0)
        self.series_angle.clear()
        self.series_marker.clear()
        self.lbl_stats.setText("avg — | min — | max —")
        self.lbl_live.setText("")
        self.btn_export.setEnabled(False)
    # -- New analysis + opening --------------------------------------------
    def _new_analysis(self):
        schaatsers = skate_db.list_skaters(self.library)
        if not schaatsers:
            QMessageBox.information(
                self, "New analysis",
                "First create a skater -- every analysis belongs to a profile.")
            return
        dlg = NewAnalysisDialog(schaatsers, voorkeur_id=self._selected_schaatser_id(),
                                parent=self)
        if show_dialog(dlg) != QDialog.Accepted:
            return
        self.input_pad = dlg.video_pad

        # Model choice only matters for the MediaPipe backend; YOLO uses its own model
        # (yolo26x-pose.pt) and ignores model_pad.
        heavy = False
        if not IS_YOLO:
            if dlg.chk_heavy.isChecked():
                if os.path.isfile(HEAVY_MODEL):
                    self.model_pad = HEAVY_MODEL
                    heavy = True
                else:
                    QMessageBox.warning(
                        self, "Heavy model missing",
                        "pose_landmarker_heavy.task isn't next to the script.\n\n"
                        "Download it from:\nhttps://storage.googleapis.com/mediapipe-models/"
                        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                        "pose_landmarker_heavy.task\n\nUsing the full model for now.")
                    self.model_pad = DEFAULT_MODEL
            else:
                self.model_pad = DEFAULT_MODEL

            if not os.path.isfile(self.model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Choose pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                self.model_pad = gekozen

        # Read the first frame; shared by the target and horizon pickers.
        frame0 = self._read_first_frame()
        if frame0 is None:
            return

        # Have the target skater chosen on the first frame (click or box).
        keuze = self._choose_target_skater(frame0)
        if keuze is False:               # dialog cancelled
            return
        self.doel_punt, self.doel_kader = keuze

        # Perspective calibration (track lines) or the classic horizon step.
        self.perspectief = None
        if dlg.chk_perspectief.isChecked():
            self.perspectief = self._choose_perspectief(frame0)
            if self.perspectief is None:     # dialog cancelled
                return
            self.horizon_deg, self.auto_horizon = 0.0, False   # calibration replaces the horizon
        else:
            horizon = self._choose_horizon(frame0)
            if horizon is False:         # dialog cancelled
                return
            self.horizon_deg, self.auto_horizon = horizon

        self.smooth_n = dlg.spin_smooth.value()
        self.threshold = dlg.spin_threshold.value()
        self.geen_smoothing = dlg.chk_geen_smoothing.isChecked()
        self.bocht_overslaan = dlg.chk_bocht.isChecked()
        self.deinterlacen = dlg.deinterlacen

        # What the .npz does NOT contain but reopening needs/wants documented.
        # (The per-frame corner flag IS in the npz; this is purely the setting.)
        instellingen = {
            "smooth_n": self.smooth_n,
            "threshold": self.threshold,
            "smooth_landmarks": not self.geen_smoothing,
            "skip_corner": self.bocht_overslaan,
            "deinterlaced": self.deinterlacen,
            "doel_punt": list(self.doel_punt) if self.doel_punt else None,
            "doel_kader": list(self.doel_kader) if self.doel_kader else None,
            "horizon_deg": self.horizon_deg,
            "auto_horizon": self.auto_horizon,
            "heavy": heavy,
            "backend_name": BACKEND_NAME,
            "perspective_used": self.perspectief is not None,
            # The calibration input (lines + parameters), so reopening recomputes the
            # correction instead of letting it evaporate -- and so a next analysis from
            # the same camera position can reuse it.
            "perspective": self.perspectief.naar_dict() if self.perspectief else None,
        }
        self._pending_save = {"schaatser_id": dlg.schaatser_id, "titel": dlg.titel,
                                "instellingen": instellingen}

        # Stay on the library; the progress bar runs at the bottom and when done the
        # view jumps to the result automatically (as long as nothing else is open).
        self._start_analysis()

    def _choose_perspectief(self, frame0):
        """Perspective calibration for one video: first offers to take one over from an
        earlier analysis (same camera position), then the `CalibrationPicker` --
        prefilled if something was taken over, so checking and correcting stays one
        action. Returns a PerspectiveConfig, or None on cancel.

        Reuse here isn't a convenience but a geometric requirement: analyses you want to
        compare against each other must rest on the same calibration, otherwise you're
        measuring the spread between seven separate tracings instead of the effect of
        the correction.
        """
        h, w = frame0.shape[:2]
        invoer = config = None
        try:
            eerdere = skate_db.list_calibrations(self.library, beeld_w=w, beeld_h=h)
        except Exception:
            eerdere = []
        if eerdere:
            keuzes = ["New calibration (trace the lines yourself)"]
            for k in eerdere[:15]:
                datum = k["datum"] or ""
                naam = k["schaatser"] or "?"
                extra = f" — {k['note']}" if k["note"] else ""
                keuzes.append(f"{k['titel']} ({naam}, {datum}){extra}")
            keuze, ok = QInputDialog.getItem(
                self, "Reuse a calibration?",
                "There are already calibrations in the library for "
                f"{w}×{h} video. A calibration belongs to one camera position, so clips "
                "from\nthe same recording should use the same one -- only then are "
                "their\nangles comparable to each other.\n\nReuse from:",
                keuzes, 0, False)
            if not ok:
                return None
            idx = keuzes.index(keuze)
            if idx > 0:
                try:
                    config = PerspectiveConfig.uit_dict(eerdere[idx - 1]["perspective"])
                    invoer = config.invoer
                except Exception as e:
                    QMessageBox.warning(
                        self, "Calibration unusable",
                        f"That saved calibration can't be recomputed:\n\n{e}\n\n"
                        "Trace the lines again.")
                    config = invoer = None

        kdlg = CalibrationPicker(frame0, self, calibration_input=invoer, config=config)
        if show_dialog(kdlg) != QDialog.Accepted:
            return None
        return kdlg.perspectief

    def _load_analysis_data(self, analysis_id):
        """
        Loads one analysis from the library and recomputes the derivatives with the
        saved settings (the phase 0 seam). Returns a dict, or None if it fails (the
        message has then already been shown).

        Deliberately touches no MainWindow state -- `smooth_n`/`threshold` come back as
        a value instead of being set on self. That way the compare page can load
        analyses without overwriting the settings of the open analysis (which `_after_edit`
        uses).
        """
        try:
            data = skate_db.load_analysis(self.library, analysis_id)
        except Exception as e:
            QMessageBox.critical(self, "Error opening",
                                 f"Can't load the analysis:\n\n{e}")
            return None

        sync = skate_db.video_sync_status(data["video_pad"], data["meta"].get("video_bytes"))
        if sync == "ontbreekt":
            QMessageBox.warning(
                self, "Video missing",
                "The video file for this analysis isn't (yet) on disk -- "
                "the cloud folder may still be syncing.\n\n"
                "Try again later.")
            return None
        if sync == "onvolledig":
            QMessageBox.warning(
                self, "Video not fully synced yet",
                "The video file hasn't been fully downloaded from the shared "
                "cloud folder yet (it's smaller than when it was saved).\n\n"
                "Wait for the sync to finish and try again.")
            return None

        info, resultaten = data["info"], data["resultaten"]
        inst = data["meta"]["instellingen"]
        smooth_n = int(inst.get("smooth_n", 5))
        threshold = float(inst.get("threshold", 0.015))

        # Perspective calibration back from the settings (the lines are saved, the
        # matrices are recomputed here). If that fails, opening just continues without
        # correction -- with a warning, since the angles then deviate.
        # "perspective" is the settings-json storage key (English since this session);
        # "perspectief" is the old spelling, read as a fallback for an analysis saved
        # before this rename -- not to be confused with the `perspectief=` keyword
        # argument into analyze()/process_derivatives, which is a separate, permanently
        # Dutch concern (see TRANSLATION_PROGRESS.md's "Deferred to Phase 8" section).
        perspectief_dict = inst.get("perspective") or inst.get("perspectief")
        perspectief, persp_fout = None, None
        if perspectief_dict:
            try:
                perspectief = PerspectiveConfig.uit_dict(perspectief_dict)
            except Exception as e:
                persp_fout = str(e)

        # The horizon is already per frame in the .npz; only the derivatives need recomputing.
        process_derivatives(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                           perspectief=perspectief)
        events = segment_pushes(resultaten)
        # The events cache is a snapshot of the computation at save time; what you see
        # here is freshly recomputed. Updating it keeps the list view (push count,
        # average angle) in sync with the table -- even for analyses from before an
        # algorithm improvement. If it fails (e.g. the DB briefly locked in the cloud
        # folder), that's no reason to abort opening.
        try:
            skate_db.refresh_events_cache(self.library, analysis_id, events)
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
            "perspectief_gebruikt": bool(inst.get("perspective_used", inst.get("perspectief_gebruikt"))),
            "perspectief": perspectief,
            "perspectief_fout": persp_fout,
            # This way the player sees exactly the pixels the measurement was made on. An
            # analysis from before this function is missing the key and so shows the raw
            # image -- exactly what was measured back then too.
            "deinterlaced": bool(inst.get("deinterlaced")),
        }

    def _open_analysis_from_library(self, analysis_id=None):
        """Opens a saved analysis on the view page."""
        if analysis_id is None:
            analysis_id = self._selected_analyse_id()
        if analysis_id is None:
            return
        data = self._load_analysis_data(analysis_id)
        if data is None:
            return

        # These two drive the skeleton editor (_after_edit recomputes with them).
        self.smooth_n = data["smooth_n"]
        self.threshold = data["threshold"]
        self.deinterlacen = data["deinterlaced"]
        self.perspectief = data["perspectief"]
        if data["perspectief_fout"]:
            QMessageBox.warning(
                self, "Calibration couldn't be recomputed",
                "This analysis was run with perspective correction, but the saved "
                "calibration no longer produces a valid camera position:\n\n"
                f"{data['perspectief_fout']}\n\n"
                "The angles have been recomputed without correction and so deviate.")
        elif data["perspectief_gebruikt"] and self.perspectief is None:
            QMessageBox.information(
                self, "Without perspective correction",
                "This analysis was run with perspective correction at the time, from "
                "before calibrations were saved. The angles have now been recomputed "
                "without correction and may therefore deviate.")

        self.input_pad = data["video_pad"]
        self.analysis_id = analysis_id
        self.open_analysis_skater_id = data["meta"]["schaatser_id"]
        self.open_analysis_skater_name = self._skater_name(self.open_analysis_skater_id)
        # The user is now deliberately viewing this analysis; an analysis running in
        # the background shouldn't yank it out from under them in a moment.
        self._auto_show_done = False
        self.stack.setCurrentWidget(self.page_analysis)
        self._show_results(data["info"], data["resultaten"], data["events"],
                              source=data["titel"])

    # -- Comparing (two analyses side by side) --------------------------
    def _compare_skaters(self):
        """Opens the compare page; first asks for whichever analyses are still missing."""
        if not any(s["aantal_analyses"] for s in skate_db.list_skaters(self.library)):
            QMessageBox.information(
                self, "Nothing to compare yet",
                "There are no analyses in the library yet.")
            return
        for kant in (self.side_left, self.side_right):
            if kant.has_analysis():
                continue          # already filled (e.g. on returning) -- leave it
            if not self._choose_compare_side(kant):
                break             # cancelled: continue with what's there
        if not (self.side_left.has_analysis() or self.side_right.has_analysis()):
            return                # nothing chosen at all -> don't go to an empty page
        self.stack.setCurrentWidget(self.page_compare)
        self.statusBar().showMessage(
            "Comparing: set a sync point per side on the same phase of the stroke and "
            "press 'Start all'.")

    def _show_analysis_info(self, analysis_id=None):
        """Info about an analysis: app version/commit, backend, date, creator and the
        main settings. Shared by the button on the view page (the open analysis) and the
        one on the start page (the row selected in the library). The metadata is fetched
        fresh from the DB here (cheap -- `analysis_meta` doesn't read the npz), so no
        copy needs to live on MainWindow that both a fresh analysis and opening one
        would have to keep in sync."""
        if analysis_id is None:
            analysis_id = self.analysis_id
        if analysis_id is None:
            return
        try:
            meta = skate_db.analysis_meta(self.library, analysis_id)
        except Exception as e:
            QMessageBox.warning(self, "Info", f"Could not read the analysis data:\n{e}")
            return
        show_dialog(AnalysisInfoDialog(
            meta, self._skater_name(meta.get("schaatser_id")), parent=self))

    def _compare_with_this(self):
        """Compare directly from the view page: the open analysis goes on the left, and
        for the right side (if nothing usable is there yet) an analysis is asked for
        right away. Skips the detour through the library."""
        if self.analysis_id is None:
            return
        self._pause_all()
        if not self._set_compare_side(self.side_left, self.analysis_id,
                                        self.open_analysis_skater_name):
            return          # message already shown
        # A different analysis on the right stays put (sync point included); the same
        # analysis twice side by side makes no sense.
        if (not self.side_right.has_analysis()
                or self.side_right.analysis_id == self.analysis_id):
            self._choose_compare_side(self.side_right,
                                      voorkeur_id=self.open_analysis_skater_id)
        self.stack.setCurrentWidget(self.page_compare)
        self.statusBar().showMessage(
            "Comparing: set a sync point per side on the same phase of the stroke and "
            "press 'Start all'.")

    def _choose_compare_side(self, kant, voorkeur_id=None):
        """Lets one side choose an analysis and loads it. True if it succeeded."""
        if voorkeur_id is None:
            voorkeur_id = self._selected_schaatser_id()
        dlg = AnalysisPicker(self.library, titel=f"{kant.name}: choose analysis",
                             voorkeur_schaatser_id=voorkeur_id, parent=self)
        if show_dialog(dlg) != QDialog.Accepted or dlg.analysis_id is None:
            return False
        return self._set_compare_side(kant, dlg.analysis_id, dlg.schaatser_naam)

    def _set_compare_side(self, kant, analysis_id, schaatser_naam):
        """Loads an analysis from the library into one side. True if it succeeded.

        Deliberately reloading rather than sharing the view page's results list: the
        skeleton editor mutates those objects in place, and each player has its own
        VideoCapture."""
        self._stop_all()
        data = self._load_analysis_data(analysis_id)
        if data is None:
            return False
        kant.show_analysis(analysis_id, schaatser_naam, data)
        return True

    def _close_compare_for(self, ids):
        """Lets go of the videos for these analyses on the compare page -- Windows
        refuses to delete a still-open video, and rmtree then fails silently."""
        for kant in (self.side_left, self.side_right):
            if kant.analysis_id in ids:
                kant.clear()

    def _both_to_sync(self):
        self._stop_all()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for kant in (self.side_left, self.side_right):
                kant.to_sync()
        finally:
            QApplication.restoreOverrideCursor()

    def _start_all(self, vanaf_sync=None):
        """Plays both videos at the same time, driven by one `MasterClock`.
        `vanaf_sync`: None = whatever the checkbox says (the button), False = resume
        (space)."""
        kanten = [k for k in (self.side_left, self.side_right) if k.has_analysis()]
        if not kanten:
            QMessageBox.information(self, "Nothing to start",
                                    "First choose an analysis for both sides.")
            return
        self._pause_all()
        if vanaf_sync is None:
            vanaf_sync = self.chk_from_sync.isChecked()
        if vanaf_sync:
            # Rewinding reopens the video and scrubs sequentially; that wait then sits
            # once up front instead of in the first tick.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for kant in kanten:
                    kant.to_sync()
            finally:
                QApplication.restoreOverrideCursor()
        self.clock.start([k.player for k in kanten])

    def _stop_all(self):
        # The clock itself pauses the players that were running under it (see
        # MasterClock.stop); leave the others alone, otherwise ▶ per side stops working.
        self.clock.stop()

    def _set_all_speed(self, _idx=None):
        """Applies the compare page's shared speed.

        Both sides get the same factor -- also for playing separately, since two videos
        at different speeds side by side can't be compared. If the master clock is
        running, it's recalibrated from the current position (`MasterClock.recalibrate`)."""
        idx = self.combo_all_speed.currentIndex()
        for kant in (self.side_left, self.side_right):
            kant.player.combo_speed.setCurrentIndex(idx)   # restarts a running timer
        self.clock.recalibrate()

    def _read_first_frame(self, pad=None):
        """Reads the first frame of the chosen video, or None on failure.
        Without `pad`, the current `self.input_pad`; with `pad`, an arbitrary video
        (used by the batch collection loop for each clip separately)."""
        cap = cv2.VideoCapture(pad or self.input_pad)
        ret, frame0 = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "Error", "Can't read the first frame.")
            return None
        return frame0

    def _choose_target_skater(self, frame0):
        """Shows the first frame in a picker. Returns `(doel_punt, doel_kader)` -- each
        normalized or None ('follow largest') -- or False (cancelled)."""
        dlg = TargetPicker(frame0, self)
        if show_dialog(dlg) != QDialog.Accepted:
            return False
        return dlg.doel_punt, dlg.doel_kader

    def _choose_horizon(self, frame0):
        """
        Lets the ice line/tilt be set. Returns (degrees, auto_per_frame) or
        False (cancelled).
        """
        dlg = HorizonPicker(frame0, self)
        if show_dialog(dlg) != QDialog.Accepted:
            return False
        return dlg.horizon_deg, dlg.auto_per_frame

    def _back_to_start(self):
        # Pausing happens via the stack.currentChanged hook (_pause_all)
        self._refresh_skaters()   # new/changed analyses immediately visible
        self.stack.setCurrentWidget(self.page_start)

    def _warn_backend_fallback(self):
        """Reports (once) that the YOLO backend couldn't be loaded and so measurements
        are being made with MediaPipe -- a different detector gives different angles, so
        that can't go unnoticed. Warming up starts when the window is shown, so by the
        time an analysis starts here the outcome is already known.

        In a bundled .exe there's no MediaPipe to fall back to; there it's not a warning
        but a blockage, and the message says so too."""
        if not BACKEND_ERROR or self._backend_reported:
            return
        self._backend_reported = True
        if is_frozen():
            # In the bundled package there's no second backend to fall back to: nothing
            # can be measured now (the library and viewing recordings still work).
            QMessageBox.critical(
                self, "Analysis backend not available",
                "The bundled analysis backend failed to load:\n\n"
                f"{BACKEND_ERROR}\n\n"
                "Analysis isn't possible right now. Opening the library and viewing "
                "recordings still works. Please pass this message on to whoever manages "
                "the app."
                # The log file is only worth mentioning if the user knows where it is.
                + (f"\n\nThe full log file is at:\n{LOGPATH}" if LOGPATH else ""))
            return
        QMessageBox.warning(
            self, "YOLO backend not available",
            "torch/ultralytics is installed, but failed to load:\n\n"
            f"{BACKEND_ERROR}\n\n"
            "The analysis will therefore run on the MediaPipe backend. That measures "
            "less accurately, so don't casually compare this analysis with earlier ones.")

    def _start_analysis(self):
        self._warn_backend_fallback()
        self.player.set_controls_active(False)
        self.btn_export.setEnabled(False)
        self._auto_show_done = True          # nothing else open yet -> show the result later
        self._analysis_warnings = []     # messages from the analysis itself (shown afterwards)
        self._set_busy(True)                 # no second worker/clashing edit on top of it
        self._show_progress_bar("Analyzing video...")

        opslag = self._pending_save or {}
        self.worker = AnalysisWorker(self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                                     doel_punt=self.doel_punt, horizon_deg=self.horizon_deg,
                                     auto_horizon=self.auto_horizon,
                                     smooth_landmarks=not self.geen_smoothing,
                                     perspectief=self.perspectief,
                                     library=self.library,
                                     schaatser_id=opslag.get("schaatser_id"),
                                     titel=opslag.get("titel"),
                                     instellingen=opslag.get("instellingen"),
                                     backend=BACKEND_NAME,
                                     aangemaakt_door=self.trainer_naam,
                                     bocht=self.bocht_overslaan,
                                     deinterlacen=self.deinterlacen,
                                     doel_kader=self.doel_kader)
        self.worker.progress.connect(self._analysis_progress)
        self.worker.status.connect(self._analysis_status)
        self.worker.save_error.connect(self._save_error)
        self.worker.warning.connect(self._analysis_warning)
        self.worker.done.connect(self._analysis_done)
        self.worker.error.connect(self._analysis_error)
        self.worker.start()

    def _analysis_progress(self, frame_nr, totaal):
        if totaal > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(frame_nr / totaal * 100))
        self.lbl_progress.setText(f"Analyzing video... ({frame_nr}/{totaal})")

    def _analysis_status(self, tekst):
        # Busy phase with no known duration (video copy to the library).
        self.progress_bar.setRange(0, 0)
        self.lbl_progress.setText(tekst)

    def _analysis_warning(self, tekst):
        """A message from the analysis itself (e.g. 'your click missed everyone, now
        following the largest mover'). Collected instead of shown right away: a modal
        box halfway through would ambush the user in the middle of a long analysis.
        `_analysis_done`/`_analysis_error` empty the list again."""
        self._analysis_warnings.append(tekst)

    def _show_analysis_warnings(self):
        meldingen = getattr(self, "_analysis_warnings", [])
        self._analysis_warnings = []
        if meldingen:
            QMessageBox.warning(self, "Note about this analysis", "\n\n".join(meldingen))

    def _report_corner(self, resultaten):
        """Says how much of the clip was skipped as corner. If that's everything, an
        empty table isn't a measurement but a wrongly chosen clip (or too strict a
        threshold) -- that deserves a real warning instead of '0 pushes'."""
        n = len(resultaten)
        if not n:
            return
        bocht = sum(1 for r in resultaten if r.bocht)
        if bocht == n:
            self._analysis_warnings.append(
                "In this video the skater is never facing the camera anywhere -- so the "
                "whole thing has been marked as 'corner' and no pushes were measured.\n"
                "If that's not right, re-analyze the video with 'Skip corner' turned off.")
        elif bocht:
            self.statusBar().showMessage(
                f"{bocht} of {n} frames ({bocht / n:.0%}) skipped: corner.", 10000)

    def _save_error(self, bericht):
        if self._shutting_down:
            return
        QMessageBox.warning(
            self, "Not saved in library",
            "The analysis succeeded, but couldn't be saved in the library:\n\n"
            f"{bericht}\n\nThe results are visible now, but not kept.")

    def _analysis_error(self, bericht):
        if self._shutting_down:
            return
        self._hide_progress_bar()
        self._set_busy(False)
        self._pending_save = None
        self._analysis_warnings = []      # the error already says enough
        QMessageBox.critical(self, "Error during analysis", bericht)
        self.player.set_controls_active(False)

    def _analysis_done(self, info, resultaten, events, analysis_id):
        if self._shutting_down:
            return          # signal from just before closing: build nothing more
        self._hide_progress_bar()
        self._set_busy(False)
        opslag = self._pending_save or {}
        self._pending_save = None
        self._report_corner(resultaten)
        self._show_analysis_warnings()

        if not self._auto_show_done:
            # The user is meanwhile busy with a different analysis -> don't yank them
            # out of their view; just refresh the list and report it.
            self._refresh_skaters()
            titel = opslag.get("titel") or "analysis"
            self.statusBar().showMessage(
                f"Analysis '{titel}' done and saved in the library.", 10000)
            return

        self.analysis_id = analysis_id
        self.open_analysis_skater_id = opslag.get("schaatser_id")
        self.open_analysis_skater_name = self._skater_name(self.open_analysis_skater_id)
        if analysis_id is not None:
            # The view now reads the library copy from here on; the original may go.
            try:
                self.input_pad = skate_db.analysis_video_path(self.library, analysis_id)
            except Exception:
                pass   # fall back to the source video (view only)
        self.stack.setCurrentWidget(self.page_analysis)
        self._show_results(info, resultaten, events)

    # ---- Batch analysis (several videos one after another) --------------------------

    # -- Trimming fragments from a long recording (phase 8) --------------------
    def _clip_recording(self):
        """Recording -> trim window -> clips written out -> into the existing batch flow.

        Nothing new happens after trimming: each fragment is a plain video file, so
        `BatchAnalysisDialog` (prefilled) + `_new_batch_analysis` do the rest. No
        second analysis pipeline and no second save route."""
        if self._busy:
            QMessageBox.information(
                self, "Just a moment",
                "An analysis is already running. Wait for it before trimming new "
                "fragments.")
            return
        source = self._selected_recording()
        if source is None:
            QMessageBox.information(
                self, "Trim fragments",
                "First choose a recording in the list.\n\n"
                "Nothing there? Put your recordings in the 'opnames' folder in the "
                "library and press 'Refresh'.")
            return
        # Cloud sync: report and don't open, rather than letting the trim window crash
        # on a half file. A half-hour recording in 4K is on its way for minutes.
        if not self._recording_available(source):
            return
        # Before the marking work, not after: without a profile there'll be nothing to
        # save afterwards.
        schaatsers = skate_db.list_skaters(self.library)
        if not schaatsers:
            QMessageBox.information(
                self, "Trim fragments",
                "First create a skater -- every analysis belongs to a profile.")
            return
        try:
            info = video_info(source["pad"])
        except Exception as e:
            QMessageBox.critical(self, "Recording", f"Can't open the recording:\n{e}")
            return

        # A loose video sits in the local library, the analyses go into the shared one:
        # there's no source row to point at there, so those fragments become analyses
        # without provenance (like a batch of loose clips) and the trim window can't
        # draw any already-trimmed parts for it.
        gedeeld = source["library"] == self.library
        analyzed = skate_db.source_fragments(self.library, source["id"]) if gedeeld else []
        dlg = FragmentPicker(source["pad"], info, analyzed=analyzed,
                             deinterlacen=self._source_interlaced(source), parent=self)
        if show_dialog(dlg) != QDialog.Accepted or not dlg.fragments:
            return

        paden = self._clip_to_temp(source["pad"], dlg.fragments, info,
                                          deinterlacen=self._source_interlaced(source))
        if not paden:
            return
        voorgevuld = [
            {"input_pad": pad, "titel": naam,
             "bron_id": source["id"] if gedeeld else None,
             "bron_start_frame": start if gedeeld else None,
             "bron_eind_frame": eind if gedeeld else None}
            for pad, (start, eind, naam) in zip(paden, dlg.fragments)]
        self._new_batch_analysis(voorgevuld=voorgevuld)

    def _clip_to_temp(self, source_path, fragments, info, deinterlacen=False):
        """Writes the marked parts to a temporary folder and returns the paths (or []
        on cancel/error). `sla_analyse_op` copies them afterwards as always to
        `media/<uuid>/` -- one extra copy of a short file, not worth breaking open that
        save route for."""
        self._clean_up_clip_folder()
        self._clip_tmp_dir = tempfile.mkdtemp(prefix="schaats_fragmenten_")

        voortgang = QProgressDialog("Trimming fragments...", "Stop", 0, 100, self)
        voortgang.setWindowTitle("Trimming")
        voortgang.setWindowModality(Qt.WindowModal)
        voortgang.setMinimumDuration(0)
        voortgang.setValue(0)

        def _report_progress(gedaan, totaal):
            voortgang.setValue(int(gedaan / max(1, totaal) * 100))

        try:
            paden = trim_fragments(source_path, fragments, self._clip_tmp_dir,
                                    progress_callback=_report_progress,
                                    stop_check=voortgang.wasCanceled, fps=info.fps,
                                    deinterlacen=deinterlacen)
        except TrimAborted:
            self._clean_up_clip_folder()
            return []
        except Exception as e:
            self._clean_up_clip_folder()
            QMessageBox.critical(self, "Trimming failed", str(e))
            return []
        finally:
            # close() only hides it; without deleteLater this window (with its
            # QScreen reference) stays behind in exactly the flow that used to crash --
            # see show_dialog.
            voortgang.close()
            voortgang.deleteLater()
        return paden

    def _clean_up_clip_folder(self):
        """Discards the temporary fragment folder (the clips are then in the library)."""
        if self._clip_tmp_dir:
            shutil.rmtree(self._clip_tmp_dir, ignore_errors=True)
            self._clip_tmp_dir = None

    def _new_batch_analysis(self, voorgevuld=None):
        schaatsers = skate_db.list_skaters(self.library)
        if not schaatsers:
            QMessageBox.information(
                self, "Batch analysis",
                "First create a skater -- every analysis belongs to a profile.")
            return
        dlg = BatchAnalysisDialog(schaatsers, voorkeur_id=self._selected_schaatser_id(),
                                  voorgevuld=voorgevuld, parent=self)
        if show_dialog(dlg) != QDialog.Accepted:
            self._clean_up_clip_folder()      # trimmed clips without a batch are useless
            return

        # Resolve the model -- shared for the whole batch, only relevant for MediaPipe.
        model_pad, heavy = DEFAULT_MODEL, False
        if not IS_YOLO:
            if dlg.heavy_gevraagd:
                if os.path.isfile(HEAVY_MODEL):
                    model_pad, heavy = HEAVY_MODEL, True
                else:
                    QMessageBox.warning(
                        self, "Heavy model missing",
                        "pose_landmarker_heavy.task isn't next to the script.\n\n"
                        "Using the full model for now.")
                    model_pad = DEFAULT_MODEL
            if not os.path.isfile(model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Choose pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                model_pad = gekozen

        smooth_n, threshold = dlg.smooth_n, dlg.threshold
        geen_smoothing = dlg.geen_smoothing
        bocht = dlg.chk_bocht.isChecked()
        deint_auto = dlg.chk_deint.isChecked()

        # Perspective calibration: once for the whole batch. In practice all clips in a
        # batch come from the same recording (that's how phase 8 trims them), so the
        # same camera position -- and only on one shared calibration are their angles
        # comparable to each other. Retracing per clip would give seven slightly
        # different calibrations.
        batch_perspectief = None
        if dlg.chk_perspectief.isChecked():
            eerste_frame = None
            for taak in dlg.taken:
                eerste_frame = self._read_first_frame(taak["input_pad"])
                if eerste_frame is not None:
                    break
            if eerste_frame is None:
                return
            batch_perspectief = self._choose_perspectief(eerste_frame)
            if batch_perspectief is None:        # dialog cancelled
                self._clean_up_clip_folder()
                return

        # Collection loop: per video ask for the first frame + target skater + horizon.
        taken = []
        for taak in dlg.taken:
            pad, schaatser_id, titel = taak["input_pad"], taak["schaatser_id"], taak["titel"]
            frame0 = self._read_first_frame(pad)
            if frame0 is None:
                continue                         # _read_first_frame already reported it

            keuze = self._choose_target_skater(frame0)
            if keuze is False:                   # dialog cancelled
                if self._skip_or_abort(titel):
                    continue
                return
            doel, kader = keuze
            if batch_perspectief is not None:
                # The calibration provides the tilt itself; the horizon step drops out,
                # same as for a single analysis.
                horizon_deg, auto_horizon = 0.0, False
                if not batch_perspectief.invoer.fits(frame0.shape[1], frame0.shape[0]):
                    QMessageBox.warning(
                        self, "Calibration doesn't fit",
                        f"'{titel}' is {frame0.shape[1]}×{frame0.shape[0]} and the "
                        f"calibration was made on {batch_perspectief.invoer.image_w}×"
                        f"{batch_perspectief.invoer.image_h}. This clip is skipped.")
                    continue
            else:
                horizon = self._choose_horizon(frame0)
                if horizon is False:             # dialog cancelled
                    if self._skip_or_abort(titel):
                        continue
                    return
                horizon_deg, auto_horizon = horizon

            # Per clip, since a batch can contain clips from different cameras.
            # Turned off in the dialog = filter nowhere (to allow an A/B run).
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
                "skip_corner": bocht,
                "deinterlaced": deint,
                "doel_punt": list(doel) if doel else None,
                "doel_kader": list(kader) if kader else None,
                "horizon_deg": horizon_deg,
                "auto_horizon": auto_horizon,
                "heavy": heavy,
                "backend_name": BACKEND_NAME,
                "perspective_used": batch_perspectief is not None,
                "perspective": batch_perspectief.naar_dict() if batch_perspectief else None,
            }
            taken.append({
                "input_pad": pad, "schaatser_id": schaatser_id, "titel": titel,
                "doel_punt": doel, "doel_kader": kader,
                "horizon_deg": horizon_deg, "auto_horizon": auto_horizon,
                "smooth_landmarks": not geen_smoothing, "smooth_n": smooth_n,
                "threshold": threshold, "model_pad": model_pad, "instellingen": instellingen,
                "bocht": bocht, "perspectief": batch_perspectief, "deinterlacen": deint,
                # Phase 8: which part of which recording this clip comes from (None for
                # a loose video) -- yields the grey blocks in the trim window later.
                "bron_id": taak.get("bron_id"),
                "bron_start_frame": taak.get("bron_start_frame"),
                "bron_eind_frame": taak.get("bron_eind_frame"),
            })

        if not taken:
            self._clean_up_clip_folder()
            return
        # Stay on the library while the batch runs, so browsing is possible.
        self._start_batch(taken)

    def _skip_or_abort(self, titel):
        """When a target/horizon picker is cancelled: only skip this video (True) or
        abort the whole batch (False)."""
        antwoord = QMessageBox.question(
            self, "Skip video?",
            f"The setup for '{titel}' was cancelled.\n\n"
            "Do you want to skip only this video and continue with the rest?\n"
            "(No = abort the whole batch.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return antwoord == QMessageBox.Yes

    def _start_batch(self, taken):
        self.player.set_controls_active(False)
        self.btn_export.setEnabled(False)
        self._auto_show_done = False        # the batch doesn't show results itself
        self._set_busy(True)

        self._batch_index, self._batch_total, self._batch_current = 0, len(taken), ""

        # Progress bar with a 'Stop after this video' button -- non-blocking, so the
        # library stays usable in the meantime.
        self._show_progress_bar("Starting batch...", met_stop=True)

        self._warn_backend_fallback()
        self.batch_worker = BatchWorker(taken, self.library, BACKEND_NAME, self.trainer_naam)
        self.batch_worker.task_start.connect(self._batch_task_start)
        self.batch_worker.progress.connect(self._batch_progress)
        self.batch_worker.status.connect(self._analysis_status)   # reuse the busy phase
        self.batch_worker.task_done.connect(self._batch_task_done)  # show the new analysis live
        self.batch_worker.all_done.connect(self._batch_done)
        self.batch_worker.start()

    def _batch_stop_requested(self):
        if self.batch_worker is not None:
            self.batch_worker.requestInterruption()
        self.lbl_progress.setText("Stopping after the current video...")
        self.btn_progress_stop.setEnabled(False)

    def _batch_task_start(self, index, totaal, titel):
        self._batch_index, self._batch_total, self._batch_current = index, totaal, titel
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText(f"Video {index + 1}/{totaal} — {titel}")

    def _batch_progress(self, frame_nr, totaal):
        if totaal > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(frame_nr / totaal * 100))
        self.lbl_progress.setText(
            f"Video {self._batch_index + 1}/{self._batch_total} — {self._batch_current} "
            f"({frame_nr}/{totaal})")

    def _batch_task_done(self, index, analysis_id):
        # Let each finished video pop up in the library right away (doesn't touch a
        # possibly open view -- those are different widgets).
        if self._shutting_down:
            return
        self._refresh_skaters()

    def _batch_done(self, geslaagd, fouten, waarschuwingen=()):
        if self._shutting_down:
            return
        self._hide_progress_bar()
        self._set_busy(False)
        self._refresh_skaters()              # new analyses immediately visible
        self._refresh_recordings()                 # update the fragment count per recording
        # The clips now exist as a copy in media/<uuid>/; the temp folder may go.
        self._clean_up_clip_folder()

        n_ok = len(geslaagd)
        n_tot = n_ok + len(fouten)
        # Warnings (e.g. a click that missed everyone) belong to a successful video:
        # the analysis exists, but possibly of the wrong skater -- so do report it.
        extra = ""
        if waarschuwingen:
            regels = "\n".join(f"• {t}: {m}" for t, m in waarschuwingen)
            extra = f"\n\nNote:\n{regels}"
        if fouten:
            regels = "\n".join(f"• {t}: {m}" for t, m in fouten)
            QMessageBox.warning(
                self, "Batch done",
                f"{n_ok} of {n_tot} videos succeeded and were saved.\n\nFailed:\n{regels}{extra}")
        elif waarschuwingen:
            QMessageBox.warning(
                self, "Batch done",
                f"All {n_ok} videos have been analyzed and saved in the library.{extra}")
        else:
            QMessageBox.information(
                self, "Batch done",
                f"All {n_ok} videos have been analyzed and saved in the library.")
    def _show_results(self, info, resultaten, events, source=None):
        """
        Fills the view page with a results list. Shared by a fresh analysis and one
        loaded from .npz (`source` = the file name, for the status bar).
        """
        self.events = events

        # Reset editor status (no edit leakage between analyses); not via the toggle
        # handler, since the view is rebuilt below anyway.
        self._stop_placing()       # still before the reset: belongs to the PREVIOUS results list
        self._editor_active = False
        self._drag = None
        self.player.edit_mode = False
        self.player.follow_frozen = False
        self._undo.clear()
        self._redo.clear()
        self._manual.clear()
        self.btn_edit.blockSignals(True)
        self.btn_edit.setChecked(False)
        self.btn_edit.blockSignals(False)
        self.editor_bar.setVisible(False)
        self._close_place_bar()

        # Reopen the capture, reset zoom, controls on -- doesn't show a frame yet.
        self.player.load(info, resultaten, self.input_pad, self.deinterlacen)

        self._fill_table()
        self._fill_chart()
        self._update_coverage()
        self.btn_export.setEnabled(bool(events))
        # Comparing only works with an analysis that's in the library -- the compare
        # side reloads it from there. Same for the info: the metadata (app version,
        # settings) comes from the DB row.
        self.btn_compare_this.setEnabled(self.analysis_id is not None)
        self.btn_info.setEnabled(self.analysis_id is not None)
        # Only offer it where there's something to gain: an analysis that already knows
        # the corner needs nothing, and without a library id the result can't be saved
        # anywhere.
        self.btn_corner_now.setEnabled(
            self.analysis_id is not None and not any(r.bocht for r in resultaten))

        herkomst = f"  ·  loaded from {source}" if source else ""
        self.statusBar().showMessage(
            f"{os.path.basename(self.input_pad)} — {info.w}×{info.h} @ {info.fps:.1f}fps, "
            f"{len(resultaten)} frames, {len(events)} pushes found{herkomst}")

        # Only draw now: table and graph are ready for the on_frame_shown hook.
        self.player.go_to(0)

    # -- Filling the table + graph -----------------------------------------
    def _fill_table(self):
        # Columns are dynamic: perspective correction and speed/stroke length only if a
        # calibration was active (the events then carry those fields).
        met_corr = any(ev.correctie is not None for ev in self.events)
        met_metrisch = any(ev.snelheid is not None for ev in self.events)
        koppen = ["#", "Time (s)", "Leg", "Angle (°)"]
        if met_corr:
            koppen.append("Corr. (°)")
        if met_metrisch:
            koppen += ["v (m/s)", "Stroke (m)"]
        self.tabel.setColumnCount(len(koppen))
        self.tabel.setHorizontalHeaderLabels(koppen)

        self.tabel.setRowCount(len(self.events))
        markeer_kleur = QColor(120, 60, 20)     # warning: impossible L/R repeat
        onbetrouwbaar_kleur = QColor(40, 60, 120)  # angle measured with leg ~along the sightline
        onvolledig_kleur = QColor(70, 70, 70)   # push not fully observed: not a measurement
        # The reason matters to the user differently: with "truncated" a longer
        # recording helps, with "no full push" the question is the leg assignment or
        # the stroke itself.
        onvolledig_uitleg = {
            INCOMPLETE_TRUNCATED:
                "This push was still ongoing when the video (or the detection) ended -- "
                "the push wasn't finished, so the angle is too steep. Not counted in "
                "average/min/max.",
            INCOMPLETE_NO_PUSH:
                "The leg did come upright in this stroke, but no sideways push was "
                "observed: at full extension the lower leg was still nearly vertical. "
                "The angle is therefore the coming-upright and isn't counted in "
                "average/min/max. Check whether there really was a push here.",
        }
        for i, ev in enumerate(self.events):
            waarden = [str(i + 1), f"{ev.start_tijd:.2f}", ev.been.capitalize(), f"{ev.hoek:.1f}"]
            if met_corr:
                waarden.append(f"{ev.correctie:+.1f}" if ev.correctie is not None else "—")
            if met_metrisch:
                waarden.append(f"{ev.snelheid:.1f}" if ev.snelheid is not None else "—")
                waarden.append(f"{ev.slaglengte:.1f}" if ev.slaglengte is not None else "—")
            gemarkeerd = ev.opmerking == "missed counter-push?"
            onbetrouwbaar = met_corr and not ev.betrouwbaar
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setTextAlignment(Qt.AlignCenter)
                if ev.onvolledig:
                    item.setBackground(onvolledig_kleur)
                    item.setToolTip(onvolledig_uitleg.get(
                        ev.onvolledig,
                        f"Incomplete push ({ev.onvolledig}) -- not counted in "
                        "average/min/max."))
                elif gemarkeerd:
                    item.setBackground(markeer_kleur)
                    item.setToolTip("Same leg as the previous push -- impossible in the "
                                    "skating rhythm. Probably a missed counter-push.")
                elif onbetrouwbaar:
                    item.setBackground(onbetrouwbaar_kleur)
                    item.setToolTip("Leg was nearly along the sightline at push "
                                    "completion -- the perspective correction (and "
                                    "hence the angle) is unreliable here.")
                self.tabel.setItem(i, kolom, item)

        if self.events:
            # Incomplete pushes don't count: their angle is the coming-upright and
            # pulls avg/max up. The reason is shown, since it calls for something
            # different from the user (a longer recording vs. checking the stroke itself).
            meetbaar = [ev for ev in self.events if not ev.onvolledig]
            hoeken = [ev.hoek for ev in meetbaar]
            n_mark = sum(1 for ev in self.events if ev.opmerking == "missed counter-push?")
            redenen = {}
            for ev in self.events:
                if ev.onvolledig:
                    redenen[ev.onvolledig] = redenen.get(ev.onvolledig, 0) + 1
            if hoeken:
                tekst = (f"avg {np.mean(hoeken):.1f}°  |  min {min(hoeken):.1f}°  "
                         f"|  max {max(hoeken):.1f}°")
            else:
                tekst = "no full push measured"
            if redenen:
                tekst += "   ·  " + ", ".join(f"{n}× {reden}" for reden, n in redenen.items())
                tekst += " (not counted)"
            if n_mark:
                tekst += f"   ·  ⚠ {n_mark} possible L/R error"
            if met_corr:
                n_onb = sum(1 for ev in meetbaar if not ev.betrouwbaar)
                if n_onb:
                    tekst += f"   ·  ⚠ {n_onb} unreliable angle (sightline)"
            self.lbl_stats.setText(tekst)
        else:
            self.lbl_stats.setText("No pushes detected")

    def _fill_chart(self):
        self.series_angle.clear()
        # Same quantity as the table (the angle per frame), so a table row falls
        # exactly on the curve.
        punten = [(r.tijd, r.hoek) for r in self.resultaten
                  if r.pose_gevonden and r.hoek is not None]
        if not punten:
            return
        for t, hoek in punten:
            self.series_angle.append(t, hoek)

        tijden = [t for t, _ in punten]
        hoeken = [h for _, h in punten]
        self.axis_x.setRange(0, max(tijden) if tijden else 1)
        marge = 5
        self.axis_y.setRange(min(hoeken) - marge, max(hoeken) + marge)

    def _update_chart_marker(self, tijd):
        y_min, y_max = self.axis_y.min(), self.axis_y.max()
        self.series_marker.clear()
        self.series_marker.append(tijd, y_min)
        self.series_marker.append(tijd, y_max)

    # -- Navigation / view --------------------------------------------------
    def _player_frame_shown(self, idx):
        """VideoPlayer hook: everything the analysis page hangs off a frame."""
        # Navigating away during a placement sequence closes it cleanly first. The idx
        # check is needed because _recompute() redraws itself and so comes back here on
        # the same frame.
        if self._place is not None and self._place['idx'] != idx:
            self._stop_placing()
        resultaat = self.resultaten[idx]
        self._update_chart_marker(resultaat.tijd)
        self._mark_active_row(idx)
        self._update_live_status(resultaat)
        if self._editor_active:
            self._update_editor_buttons()   # "Make skeleton" only on a frame without a pose

    def _update_live_status(self, resultaat):
        if not resultaat.pose_gevonden:
            tekst = "Corner — not analyzed" if resultaat.bocht else "No pose detected"
            self.lbl_live.setText(tekst)
            self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px; color: #c33;")
            return
        if resultaat.bocht:
            # There's a skeleton, but no derivatives: `process_derivatives` skips corner
            # frames, so leg/angle are None here and there's nothing to show.
            self.lbl_live.setText("Corner — not measured")
            self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px; color: #c80;")
            return

        status = "WEIGHT ON LEG" if resultaat.gewicht_erop else "PUSH COMPLETE"
        kleur = "#2a2" if resultaat.gewicht_erop else "#c33"
        extra = ""
        if resultaat.hoek_correctie is not None:
            extra = f"Corr: {resultaat.hoek_correctie:+.1f}°   "
            if resultaat.snelheid is not None:
                extra += f"v: {resultaat.snelheid:.1f} m/s   "
            if not resultaat.hoek_betrouwbaar:
                extra += "⚠ unreliable   "
        strek = f"Extension: {resultaat.strek_ratio:.2f}   " if resultaat.strek_ratio is not None else ""
        self.lbl_live.setText(
            f"Push leg: {resultaat.been.upper()}   "
            f"Push angle: {resultaat.hoek}°   "
            f"Knee angle: {resultaat.kniehoek}°   "
            f"{strek}{extra}{status}"
        )
        self.lbl_live.setStyleSheet(f"font-weight: bold; padding-right: 10px; color: {kleur};")

    # -- Skeleton editor: handles on the VideoPlayer -----------------------
    def _handle_radius(self):
        """Handle/grab radius in (scaled) screen pixels, proportional to the skater:
        HANDLE_FRAC × on-screen torso length, clamped to [HANDLE_MIN_PX, HANDLE_MAX_PX].
        Via _norm_naar_widget the crop/zoom scale is already included (the letterbox
        offset drops out over a distance), so this is correct at any zoom level and
        exactly consistent with the hit test. Falls back to HANDLE_MAX_PX if there's no
        usable pose/torso."""
        if (not (0 <= self.huidige_idx < len(self.resultaten))
                or self.player.display_scaled is None):
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

        schouder, heup = _mid(11, 12), _mid(23, 24)   # shoulder-mid -> hip-mid
        if schouder is None or heup is None:
            return float(HANDLE_MAX_PX)
        p1 = self.player.norm_to_widget(*schouder)
        p2 = self.player.norm_to_widget(*heup)
        torso = math.hypot(p1.x() - p2.x(), p1.y() - p2.y())
        return min(float(HANDLE_MAX_PX), max(float(HANDLE_MIN_PX), HANDLE_FRAC * torso))

    def _draw_handles(self, pixmap):
        """Draws draggable rings on every visible landmark of the current frame,
        directly on the scaled pixmap (so a fixed size in screen pixels).

        Hangs permanently as overlay_drawer on the player; the edit-mode guard is
        therefore here (that flag is cleared in two places -- one guard is fail-safe)."""
        if not self._editor_active:
            return
        if not (0 <= self.huidige_idx < len(self.resultaten)):
            return
        r = self.resultaten[self.huidige_idx]
        if not (r.pose_gevonden and isinstance(r.lm, list)):
            return
        pw, ph = pixmap.width(), pixmap.height()
        x0n, y0n, wn, hn = self.player.crop_norm  # at zoom==1 (0,0,1,1) -> lm.x*pw, lm.y*ph
        straal = self._handle_radius()      # scales with the skater + zoom
        gemarkeerd = self._manual.get(self.huidige_idx, set())
        drag_j = (self._drag['j'] if self._drag and self._drag['idx'] == self.huidige_idx
                  else None)
        doel_j = self._place_target_point()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            for j, lm in enumerate(r.lm):
                if getattr(lm, 'visibility', 1.0) < HANDLE_MIN_VIS and j != doel_j:
                    continue
                # outside the crop the ring falls outside [0,pw]; the painter clips it
                middel = QPointF((lm.x - x0n) / wn * pw, (lm.y - y0n) / hn * ph)
                if j == drag_j:
                    painter.setPen(QPen(QColor(255, 255, 0), 3))     # actively dragged
                elif j in gemarkeerd:
                    painter.setPen(QPen(QColor(0, 255, 120), 2))     # manually placed
                else:
                    painter.setPen(QPen(QColor(255, 255, 255), 1))   # regular
                painter.drawEllipse(middel, straal, straal)
                if j == doel_j:
                    # The point currently being asked for: a thick orange ring with
                    # crosshair on the prefilled spot. The hint text alone lets the user
                    # hunt; this says "roughly here, correct it".
                    painter.setPen(QPen(QColor(255, 150, 0), 3))
                    buiten = straal * 1.8
                    painter.drawEllipse(middel, buiten, buiten)
                    painter.drawLine(QPointF(middel.x() - buiten * 1.5, middel.y()),
                                     QPointF(middel.x() + buiten * 1.5, middel.y()))
                    painter.drawLine(QPointF(middel.x(), middel.y() - buiten * 1.5),
                                     QPointF(middel.x(), middel.y() + buiten * 1.5))
        finally:
            painter.end()

    def _place_target_point(self):
        """The landmark the running placement sequence is currently asking for, or None."""
        if not self._place or self._place['idx'] != self.huidige_idx:
            return None
        stap = self._place['stap']
        return PLACEMENT_ORDER[stap] if 0 <= stap < len(PLACEMENT_ORDER) else None

    # -- Skeleton editor: edit mode + dragging (phase 3) --------------------
    def _toggle_editing(self, actief):
        if not actief:
            self._stop_placing()           # never leave a half sequence behind
        self._editor_active = actief
        self.player.edit_mode = actief   # drives the pan-vs-editor priority of the mouse
        self.editor_bar.setVisible(actief)
        self._drag = None
        self.player.follow_frozen = False
        if actief:
            self.player.pause()
            self.lbl_editor_hint.setText("Drag a point to the right spot.")
            self._update_editor_buttons()
        self.player.show_current_frame()

    def _update_editor_buttons(self):
        bezig = self._place is not None
        self.btn_undo.setEnabled(bool(self._undo) and not bezig)
        self.btn_redo.setEnabled(bool(self._redo) and not bezig)
        self.btn_restore.setEnabled(self.analysis_id is not None and not bezig)
        self.btn_next_gap.setEnabled(not bezig)
        # Only offer it where it makes sense: on a frame that already has a pose,
        # dragging is the tool, not placing -- and in the corner nothing is measured
        # anyway.
        self.btn_create_skeleton.setEnabled(not bezig and self._is_gap(self.huidige_idx))

    def _frame_editable(self, idx):
        """A frame is editable if it has a pose stored in memory as a list of (mutable)
        Landmark tuples -- true for every analysis loaded from the library. Raw
        MediaPipe objects (diagnostic mode 'no smoothing') are not."""
        if not (0 <= idx < len(self.resultaten)):
            return False
        r = self.resultaten[idx]
        return bool(r.pose_gevonden and isinstance(r.lm, list))

    def _set_landmark(self, idx, j, nx, ny, vis=None):
        """Replaces landmark j in frame idx (Landmark is immutable)."""
        lm = self.resultaten[idx].lm[j]
        self.resultaten[idx].lm[j] = Landmark(nx, ny, lm.z,
                                              lm.visibility if vis is None else vis)

    def _taper_frames(self, idx, N):
        """Frame indices the correction blends over: idx plus up to ±N neighboring
        frames, stopping at a detection gap (uneditable frame) in each direction."""
        frames = [idx]
        for richting in (-1, 1):
            for k in range(1, N + 1):
                f = idx + richting * k
                if not self._frame_editable(f):
                    break
                frames.append(f)
        return frames

    def _find_landmark(self, pos):
        """Index of the nearest visible landmark within the handle radius
        (_handle_radius) of the mouse position (screen space), or None."""
        if not self._frame_editable(self.huidige_idx) or self.player.display_scaled is None:
            return None
        straal = self._handle_radius()      # same radius as the drawn ring
        beste, beste_d2 = None, float(straal * straal)
        for j, lm in enumerate(self.resultaten[self.huidige_idx].lm):
            if getattr(lm, 'visibility', 1.0) < HANDLE_MIN_VIS:
                continue
            w = self.player.norm_to_widget(lm.x, lm.y)
            d2 = (w.x() - pos.x()) ** 2 + (w.y() - pos.y()) ** 2
            if d2 <= beste_d2:
                beste, beste_d2 = j, d2
        return beste

    def _show_hover_name(self, event):
        """In edit mode, shows a tooltip with the body part of the point under the
        cursor (same hit radius as selecting). No point nearby -> tooltip goes away."""
        j = self._find_landmark(event.position())
        if j is None:
            QToolTip.hideText()
            return
        naam = LANDMARK_NAMES.get(j, f"point {j}")
        # slightly offset from the cursor so the text doesn't cover the point itself
        pos = (event.globalPosition() + QPointF(14, 10)).toPoint()
        QToolTip.showText(pos, naam, self.player.label)

    # The pan branch (left-drag at zoom > 1 outside edit mode) lives in VideoPlayer;
    # these hooks only get the event if the player hasn't already swallowed it itself.
    def _editor_mouse_press(self, event):
        if not self._editor_active:
            return
        # The placement sequence takes priority and always swallows the click: the
        # frame IS editable during the sequence, so without this branch a click would
        # start a drag on a prefilled point instead of placing the requested point.
        if self._place is not None:
            self._place_click(event)
            return
        if not self._frame_editable(self.huidige_idx):
            self.lbl_editor_hint.setText("This frame has no editable pose.")
            return
        j = self._find_landmark(event.position())
        if j is None:
            return
        self._drag = {'idx': self.huidige_idx, 'j': j,
                       'start_lm': self.resultaten[self.huidige_idx].lm[j]}
        # freeze auto-follow: otherwise the crop jumps out from under the cursor
        self.player.follow_frozen = True

    def _editor_mouse_move(self, event):
        if not self._editor_active or self._place is not None:
            return
        if not self._drag:
            # no drag in progress -> show the body part under the cursor on hover
            self._show_hover_name(event)
            return
        norm = self.player.widget_to_norm(event.position())
        if norm is None:
            return
        nx = min(1.0, max(0.0, norm[0]))
        ny = min(1.0, max(0.0, norm[1]))
        idx, j = self._drag['idx'], self._drag['j']
        self._set_landmark(idx, j, nx, ny, vis=1.0)   # live feedback; no recompute yet
        self.player.go_to(idx)

    def _editor_mouse_release(self, event):
        if self._place is not None:
            return                          # the click was already handled on press
        if not (self._editor_active and self._drag):
            self.player.follow_frozen = False
            return
        drag, self._drag = self._drag, None
        self.player.follow_frozen = False
        idx, j, start_lm = drag['idx'], drag['j'], drag['start_lm']
        eind = self.resultaten[idx].lm[j]
        dx, dy = eind.x - start_lm.x, eind.y - start_lm.y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            self.player.go_to(idx)              # no real movement: just redraw
            return
        # Reset the center to pre-edit so the whole window starts equal.
        self.resultaten[idx].lm[j] = start_lm
        N = self.spin_uitvloei.value()
        frames = self._taper_frames(idx, N)
        oud = {f: self.resultaten[f].lm[j] for f in frames}
        for f in frames:
            k = abs(f - idx)
            gewicht = 1.0 if k == 0 else 0.5 * (1.0 + math.cos(math.pi * k / N))
            lm = self.resultaten[f].lm[j]
            vis = 1.0 if f == idx else lm.visibility     # only the dragged point is certain
            self.resultaten[f].lm[j] = Landmark(lm.x + dx * gewicht, lm.y + dy * gewicht,
                                                lm.z, vis)
        nieuw = {f: self.resultaten[f].lm[j] for f in frames}
        self._undo.append({'type': 'sleep', 'j': j, 'oud': oud, 'nieuw': nieuw})
        self._redo.clear()
        self._manual.setdefault(idx, set()).add(j)
        self._after_edit()

    # -- Placing a skeleton on a frame without a pose ------------------------
    def _is_gap(self, idx):
        """A frame that's missing a skeleton AND where a skeleton would actually help.
        Corner frames don't count: those don't produce a measurement anyway, so closing
        them by hand is work for nothing."""
        if not (0 <= idx < len(self.resultaten)):
            return False
        r = self.resultaten[idx]
        return not r.pose_gevonden and not r.bocht

    def _gap_position(self, idx):
        """(how-manieth, total) of frame `idx` within its contiguous run of frames
        WITHOUT a skeleton. For the hint: a half-closed gap doesn't change the table
        yet, since `bepaal_afzet_uit_strek` breaks the stance run off on EVERY
        skeleton-less frame."""
        if not self._is_gap(idx):
            return (0, 0)
        start = idx
        while start > 0 and self._is_gap(start - 1):
            start -= 1
        eind = idx
        while self._is_gap(eind + 1):
            eind += 1
        return (idx - start + 1, eind - start + 1)

    def _go_to_next_gap(self):
        """Jumps to the next frame without a skeleton; wraps after the last one."""
        n = len(self.resultaten)
        if not n:
            return
        volgorde = list(range(self.huidige_idx + 1, n)) + list(range(0, self.huidige_idx + 1))
        doel = next((i for i in volgorde if self._is_gap(i)), None)
        if doel is None:
            self.lbl_editor_hint.setText("Every frame has a skeleton -- nothing left to do.")
            return
        self.player.go_to(doel)
        hoeveelste, totaal = self._gap_position(doel)
        self.lbl_editor_hint.setText(
            f"Frame {doel} — {hoeveelste} of {totaal} without a skeleton in this gap.")

    def _start_placing(self):
        """"Make skeleton" on a frame without a pose.

        Two routes, and the first is by far the usual one: if the neighboring frames
        yield a usable prefill, that skeleton goes right on and you correct it with the
        normal drag editor -- more intuitive than reading and clicking eight names, and
        it keeps one way of working for ALL frames. Only if there's NOTHING to take over
        (no pose anywhere in the analysis) is there nothing to drag, and the program asks
        for the points one at a time in a fixed order."""
        idx = self.huidige_idx
        if self._place is not None or not (0 <= idx < len(self.resultaten)):
            return
        r = self.resultaten[idx]
        if r.pose_gevonden:
            self.lbl_editor_hint.setText(
                "This frame already has a skeleton -- drag the points that are off.")
            return
        info = self.video_info
        oud_lm, oud_pose = r.lm, r.pose_gevonden
        voorvulling = make_prefill(self.resultaten, idx, info.fps or 30.0)
        bruikbaar = all(voorvulling[j].visibility >= HANDLE_MIN_VIS for j in PLACEMENT_REQUIRED)

        r.lm = voorvulling
        r.pose_gevonden = True
        # The box was computed while this was still a gap -- on a gap > KADER_GAT_S the
        # automatic zoom goes all the way out, exactly when precision is needed.
        self.player.recompute_box()

        if bruikbaar:
            # The skeleton is in place; from here it's a perfectly ordinary editable frame.
            self._undo.append({'type': 'skelet', 'idx': idx,
                               'oud_lm': oud_lm, 'oud_pose': oud_pose,
                               'nieuw_lm': list(r.lm), 'nieuw_pose': True,
                               'geklikt': set()})
            self._redo.clear()
            self._after_edit()      # recompute + save; the frame now counts
            self.lbl_editor_hint.setText(
                "Skeleton taken over from the neighboring frames -- drag the points into "
                "place. "
                f"(Blend is set to ±{self.spin_uitvloei.value()} frames.)")
            return

        # Nothing to take over: ask point by point. It has to be recomputed here
        # already, since without lm_data/leg/angle the next redraw (teken_been_overlay,
        # _update_live_status) would break on a frame that claims to have a pose. Not
        # saved yet -- the user can still cancel the sequence.
        self._place = {'idx': idx, 'stap': 0, 'geklikt': set(),
                        'oud_lm': oud_lm, 'oud_pose': oud_pose}
        self._recompute()
        self.player.follow_frozen = True   # the crop mustn't jump between clicks
        self.place_bar.setVisible(True)
        self._show_place_step()

    def _show_place_step(self):
        """Hint + button state for the current step; also refreshes the image (target
        ring)."""
        if self._place is None:
            return
        stap, n = self._place['stap'], len(PLACEMENT_ORDER)
        if stap < n:
            naam = LANDMARK_NAMES.get(PLACEMENT_ORDER[stap], f"point {PLACEMENT_ORDER[stap]}")
            self.lbl_place.setText(f"Click: {naam}  ({stap + 1} of {n})")
        else:
            self.lbl_place.setText(f"All {n} points done -- commit the skeleton.")
        self.btn_place_previous.setEnabled(stap > 0)
        self.btn_place_skip.setEnabled(stap < n)
        self.btn_place_done.setEnabled(self._place_complete())
        hoeveelste, totaal = self._gap_position(self._place['idx'])
        rest = (f"  ·  frame {hoeveelste} of {totaal} in this gap" if totaal > 1 else "")
        self.lbl_editor_hint.setText(
            "Right-drag = pan the image, mouse wheel = zoom." + rest)
        self._update_editor_buttons()
        self.player.show_current_frame()

    def _place_complete(self):
        """May the skeleton be committed? Only if every measurement point has a visible
        position -- otherwise a frame with hip, knee and ankle stacked on one point ends
        up in the table, and that reads as a push angle of 0°."""
        if self._place is None:
            return False
        lm = self.resultaten[self._place['idx']].lm
        return all(lm[j].visibility >= HANDLE_MIN_VIS for j in PLACEMENT_REQUIRED)

    def _place_click(self, event):
        if self._place is None:
            return
        stap = self._place['stap']
        if stap >= len(PLACEMENT_ORDER):
            self.lbl_place.setText("All points done -- click ✔ Done.")
            return
        norm = self.player.widget_to_norm(event.position())
        if norm is None or not (0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0):
            # Don't clamp: that would silently put the knee on the edge of the frame.
            self.lbl_editor_hint.setText("Click inside the frame.")
            return
        j = PLACEMENT_ORDER[stap]
        self._set_landmark(self._place['idx'], j, norm[0], norm[1], vis=1.0)
        self._place['geklikt'].add(j)
        self._manual.setdefault(self._place['idx'], set()).add(j)
        self._place['stap'] = stap + 1
        self._show_place_step()

    def _place_previous(self):
        if self._place is not None and self._place['stap'] > 0:
            self._place['stap'] -= 1
            self._show_place_step()

    def _place_skip(self):
        if self._place is not None and self._place['stap'] < len(PLACEMENT_ORDER):
            self._place['stap'] += 1
            self._show_place_step()

    def _place_done(self):
        if self._place is None:
            return
        if not self._place_complete():
            ontbreekt = ", ".join(
                LANDMARK_NAMES.get(j, str(j)) for j in PLACEMENT_REQUIRED
                if self.resultaten[self._place['idx']].lm[j].visibility < HANDLE_MIN_VIS)
            self.lbl_editor_hint.setText(f"Still to place: {ontbreekt}.")
            return
        # Let go of the state first: _after_edit redraws, and that fires
        # _player_frame_shown back to here -- with _place still set that would loop.
        plaats, self._place = self._place, None
        idx = plaats['idx']
        self._close_place_bar()
        self._undo.append({'type': 'skelet', 'idx': idx,
                           'oud_lm': plaats['oud_lm'], 'oud_pose': plaats['oud_pose'],
                           'nieuw_lm': list(self.resultaten[idx].lm), 'nieuw_pose': True,
                           'geklikt': set(plaats['geklikt'])})
        self._redo.clear()
        self.player.recompute_box()
        self._after_edit()

    def _place_cancel(self):
        """Back to the state before placing -- the frame is a gap again."""
        if self._place is None:
            return
        plaats, self._place = self._place, None
        idx = plaats['idx']
        r = self.resultaten[idx]
        r.lm, r.pose_gevonden = plaats['oud_lm'], plaats['oud_pose']
        self._manual.pop(idx, None)
        self._close_place_bar()
        self.player.recompute_box()
        self._recompute()               # deliberately not saving: nothing has changed
        self.lbl_editor_hint.setText("Skeleton placement cancelled.")

    def _close_place_bar(self):
        self.place_bar.setVisible(False)
        self.lbl_place.setText("")
        self.player.follow_frozen = False

    def _stop_placing(self):
        """Fail-safe exit for any path that can interrupt the sequence (navigating away,
        edit mode off, page switch, another analysis, window closing). Never leave a
        half skeleton standing.

        If the user has placed points and the skeleton is usable, that work is kept. If
        nothing has been clicked yet, it's discarded -- otherwise scrubbing away by
        accident would silently commit a prefill as a measurement (including
        `analyse.bewerkt = 1`) while the user decided nothing."""
        if self._place is None:
            return
        if self._place['geklikt'] and self._place_complete():
            self._place_done()
        else:
            self._place_cancel()

    def _recompute(self):
        """Recomputes derivatives + events from the current landmarks (NO smoothing) and
        refreshes the whole view -- WITHOUT saving.

        Separate from `_save` because a manually placed skeleton still being placed
        already needs to be computed (otherwise the overlay draws on an empty `lm_data`
        and `teken_been_overlay`/`_update_live_status` breaks), but must NOT be saved
        yet: `bewaar_bewerkte_landmarks` sets `analyse.bewerkt = 1` and creates the
        pristine backup, and that's irreversible if the user cancels the sequence."""
        info = self.video_info
        # `perspectief` must go along: since the calibration is saved, a reopened
        # analysis carries one, and without this argument the angles would silently
        # fall back to the uncorrected image plane after a single drag.
        process_derivatives(self.resultaten, info.w, info.h, info.fps,
                           self.smooth_n, self.threshold,
                           perspectief=self.perspectief)
        self.events = segment_pushes(self.resultaten)
        self._fill_table()
        self._fill_chart()
        self.btn_export.setEnabled(bool(self.events))
        self._update_coverage()
        self.player.show_current_frame()
        self._update_editor_buttons()

    def _update_coverage(self):
        """Status-bar counter: how many frames have a skeleton? Frames without one break
        off a push measurement, so this is the measure of "how much work is left".

        Corner frames don't count -- nothing can be measured there anyway, so they
        shouldn't sit in the denominator as outstanding work."""
        if not self.resultaten:
            self.lbl_coverage.setText("")
            return
        bocht = sum(1 for r in self.resultaten if r.bocht)
        totaal = len(self.resultaten) - bocht
        met = sum(1 for r in self.resultaten if r.pose_gevonden and not r.bocht)
        tekst = f"Skeleton: {met} of {totaal} frames"
        if bocht:
            tekst += f" · {bocht} in the corner"
        self.lbl_coverage.setText(tekst)
        kleur = "#888" if met == totaal else "#c80"
        self.lbl_coverage.setStyleSheet(f"padding-right: 14px; color: {kleur};")

    def _determine_corner_now(self):
        """
        Determines the corner after the fact on an analysis that doesn't yet have a
        marking for it, and removes those frames from the measurement. For anything run
        before corner detection existed: nothing is re-analyzed, since the landmarks of
        the whole clip are already in the .npz -- the hip stance can be read straight
        from that. On such an analysis the signal is even more reliable than on a fresh
        one, where the corner is measured only thinly.

        Compute first, then ask: the user sees what it does to their table before
        anything goes to the library. On "no" everything goes back exactly as it was.
        """
        if not self.resultaten or self.analysis_id is None:
            return
        info = self.video_info
        oude_vlaggen = [r.bocht for r in self.resultaten]
        oude_events = len(self.events)

        determine_corner_sequence(self.resultaten, info.w, info.h, info.fps)
        n_bocht = sum(1 for r in self.resultaten if r.bocht)
        if n_bocht == 0:
            QMessageBox.information(
                self, "No corner found",
                "In this analysis the skater faces the camera everywhere -- there's no "
                "corner to exclude. So nothing changes.")
            self.btn_corner_now.setEnabled(False)
            return

        self._recompute()          # table/graph already show what it'll become
        verdwenen = oude_events - len(self.events)
        antwoord = QMessageBox.question(
            self, "Determine corner",
            f"{n_bocht} of the {len(self.resultaten)} frames "
            f"({n_bocht / len(self.resultaten):.0%}) are in the corner.\n\n"
            f"Pushes: {oude_events} → {len(self.events)}"
            + (f" ({verdwenen} dropped -- those were measured in the corner "
               f"and so unusable)" if verdwenen > 0 else "") + ".\n\n"
            "Apply and save? The landmarks stay unchanged; only the marking of which "
            "frames fall outside the measurement is saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if antwoord != QMessageBox.Yes:
            for r, b in zip(self.resultaten, oude_vlaggen):
                r.bocht = b
            self._recompute()
            return

        try:
            skate_db.save_corner_marking(
                self.library, self.analysis_id, self.resultaten, info, self.events)
        except Exception as e:
            QMessageBox.warning(
                self, "Not saved",
                f"The corner marking couldn't be saved:\n\n{e}\n\n"
                "You see it now, but it'll be gone the next time you open it.")
            return
        self.btn_corner_now.setEnabled(False)
        self._refresh_skaters()          # average angle in the library list
        self.statusBar().showMessage(
            f"Corner determined: {n_bocht} frames excluded, saved.", 8000)

    def _after_edit(self):
        """After an edit/undo/redo: recompute AND auto-save to the library."""
        self._recompute()
        self._save()

    def _save(self):
        info = self.video_info
        if self.analysis_id is not None:
            try:
                skate_db.save_edited_landmarks(
                    self.library, self.analysis_id, self.resultaten, info, self.events)
                self.lbl_editor_hint.setText("Correction saved.")
            except Exception as e:
                self.lbl_editor_hint.setText(f"Saving failed: {e}")
        else:
            self.lbl_editor_hint.setText("Not saved (no library analysis).")

    def _apply_edit(self, edit, kant):
        """Reverts or reapplies one undo item; `kant` is 'oud' (old) or 'nieuw' (new)."""
        if edit.get('type') == 'skelet':
            idx = edit['idx']
            lm, pose = edit[f'{kant}_lm'], edit[f'{kant}_pose']
            r = self.resultaten[idx]
            r.lm = list(lm) if lm is not None else None
            r.pose_gevonden = pose
            # The green "manual" marker belongs to a skeleton that's actually there.
            if pose and lm is not None:
                self._manual[idx] = set(edit.get('geklikt', ()))
            else:
                self._manual.pop(idx, None)
            self.player.recompute_box()   # coverage changed, so the auto-zoom does too
            return
        j = edit['j']
        for f, lm in edit[kant].items():
            self.resultaten[f].lm[j] = lm

    def _undo_edit(self):
        if not (self._editor_active and self._undo) or self._place is not None:
            return
        edit = self._undo.pop()
        self._apply_edit(edit, 'oud')
        self._redo.append(edit)
        self._after_edit()

    def _redo_edit(self):
        if not (self._editor_active and self._redo) or self._place is not None:
            return
        edit = self._redo.pop()
        self._apply_edit(edit, 'nieuw')
        self._undo.append(edit)
        self._after_edit()

    def _restore_original(self):
        if self.analysis_id is None:
            return
        if QMessageBox.question(
                self, "Restore original",
                "Undo all manual corrections of this analysis and go back to the "
                "original detection?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            hersteld = skate_db.restore_original_landmarks(self.library, self.analysis_id)
        except Exception as e:
            QMessageBox.critical(self, "Restore original", f"Failed:\n\n{e}")
            return
        if not hersteld:
            QMessageBox.information(
                self, "Restore original",
                "This analysis hasn't been edited yet -- there's nothing to restore.")
            return
        try:
            data = skate_db.load_analysis(self.library, self.analysis_id)
        except Exception as e:
            QMessageBox.critical(self, "Restore original", f"Reloading failed:\n\n{e}")
            return
        info, resultaten = data["info"], data["resultaten"]
        # Pass the calibration along here too -- "restore original" is about the
        # landmarks, not about the perspective correction.
        process_derivatives(resultaten, info.w, info.h, info.fps, self.smooth_n,
                           self.threshold, perspectief=self.perspectief)
        events = segment_pushes(resultaten)
        try:
            skate_db.refresh_events_cache(self.library, self.analysis_id, events)
        except Exception:
            pass
        self._undo.clear()
        self._redo.clear()
        self._manual.clear()
        self._show_results(info, resultaten, events, source=data["meta"]["titel"])
        self._update_editor_buttons()
        self.lbl_editor_hint.setText("Original restored.")

    def _mark_active_row(self, idx):
        for i, ev in enumerate(self.events):
            if ev.start_frame <= idx <= ev.eind_frame:
                if self.tabel.currentRow() != i:
                    self.tabel.blockSignals(True)
                    self.tabel.selectRow(i)
                    self.tabel.blockSignals(False)
                return

    def _click_on_row(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self.player.go_to(self.events[rij].start_frame)

    # -- Export ---------------------------------------------------------
    def _export_csv(self):
        pad, _ = QFileDialog.getSaveFileName(self, "Export push angles", "afzethoeken.csv", "CSV (*.csv)")
        if not pad:
            return
        met_corr = any(ev.correctie is not None for ev in self.events)
        with open(pad, "w", newline="", encoding="utf-8") as f:
            schrijver = csv.writer(f)
            # `onvolledig` carries the reason (empty = a full measurement); that's worth
            # more in a sheet than the old 0/1 `afgekapt` column, which only covered one
            # of the two.
            kop = ["#", "leg", "start_time_s", "end_time_s", "angle_deg", "min_angle_deg",
                   "max_angle_deg", "incomplete"]
            if met_corr:
                kop += ["correction_deg", "reliable", "speed_ms", "stroke_length_m"]
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
        self.statusBar().showMessage(f"Exported to {pad}", 5000)

    def _active_workers(self):
        return [w for w in (self.worker, self.batch_worker)
                if w is not None and w.isRunning()]

    def _wait_for_worker(self, worker, seconden=120):
        """Waits until the thread has really stopped, with a wait cursor and a live UI.
        The abort check sits at the frame boundary (~2 s/frame with YOLO), and a video
        copy already in progress is deliberately allowed to finish -- hence a generous
        deadline instead of terminate()."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            deadline = time.monotonic() + seconden
            while not worker.wait(100):
                # Only redraw; no mouse/keys, otherwise the user clicks in a window
                # that's already closing.
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                if time.monotonic() > deadline:
                    return False
        finally:
            QApplication.restoreOverrideCursor()
        return True

    def _stop_workers(self):
        """Cleanly aborts a running (batch) analysis before closing. Returns False if
        closing shouldn't happen (the user backs out, or the thread hasn't stopped yet)
        -- destroying a running QThread is a crash."""
        actief = self._active_workers()
        if not actief:
            return True
        antwoord = QMessageBox.question(
            self, "Analysis still running",
            "An analysis is still running in the background.\n\n"
            "Closing aborts it; the video won't be saved in the library. Videos "
            "already finished in a batch are kept.\n\n"
            "Close anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if antwoord != QMessageBox.Yes:
            return False

        # Flag + blockSignals: no new signals will arrive, and one already queued may
        # not produce a dialog or page switch anymore while closing (the slots check
        # `_shutting_down`).
        self._shutting_down = True
        for w in actief:
            w.blockSignals(True)
            w.abort()
        self.lbl_progress.setText("Aborting analysis...")
        for w in actief:
            if not self._wait_for_worker(w):
                self._shutting_down = False
                for x in actief:
                    x.blockSignals(False)
                QMessageBox.warning(
                    self, "Analysis not stopping yet",
                    "The analysis isn't responding to the abort yet -- probably the "
                    "video is still being copied to the library. That copy won't be "
                    "cut off halfway.\n\nThe window stays open; try closing it again "
                    "in a moment.")
                return False
        return True

    def changeEvent(self, event):
        # No longer the active window (alt-tab, a modal dialog in front): the key
        # release of . or , never arrives then and scrubbing would run forever.
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            for keys in self._key_handlers:
                keys.stop_scrubbing()
        super().changeEvent(event)

    def closeEvent(self, event):
        if not self._stop_workers():
            event.ignore()
            return
        self._stop_local_probe()
        self._stop_placing()   # still commit or roll back a running sequence
        self._pause_all()
        for keys in self._key_handlers:
            keys.detach()
        self.player.release()
        self.side_left.clear()
        self.side_right.clear()
        self._clean_up_clip_folder()   # trimmed fragments that will no longer be analyzed
        super().closeEvent(event)


def main():
    # The QApplication and the splash screen already exist since the import at the top
    # of this file (see _start_splash_screen); only if this module is started via a
    # detour are they missing.
    app = _APP or QApplication(sys.argv)
    venster = MainWindow(melding=_SPLASH.melding if _SPLASH else None)
    venster.show()
    venster._melding = lambda tekst: None    # the splash screen closes now
    if _SPLASH:
        _SPLASH.finish(venster)
        # finish() only hides the splash screen. Without this it stays alive as a
        # top-level window for the whole session -- with a QScreen reference that goes
        # stale on a screen change; see show_dialog.
        _SPLASH.deleteLater()
    # Only pull in torch/ultralytics now: the window is up, the user can already browse
    # the library, and by the time they start an analysis the backend is ready.
    _warm_backend_up()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
