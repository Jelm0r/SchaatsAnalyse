"""
Skater Analysis Tool
=====================
Detects the push leg, calculates the push angle relative to the ice,
and reports as long as the weight is on the push leg.

Usage:
    python skate_analysis.py --input video.mp4 --output result.mp4

Requirements:
    pip install mediapipe opencv-python numpy
    Download the pose-landmarker model file (once):
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
    Put it next to this script, or pass the path with --model.

Options:
    --input     Path to input video
    --output    Path to output video (default: output.mp4)
    --model     Path to the pose_landmarker .task model file
    --fps       Force output FPS (default: same as input)
    --smooth    Number of frames for angle smoothing (default: 5)
    --threshold Minimum hip shift for weight detection (default: 0.015)
"""

import os
import sys
import cv2
import numpy as np
import argparse
from collections import deque, namedtuple
from dataclasses import dataclass, field

import skate_perspective   # pure numpy — safe in both venvs


# ── Where do the files live? ────────────────────────────────────────────────────
# These three belong here (the lowest shared module: schaats_gui, schaats_yolo, and
# schaats_db all import them from here), but now live in skate_environment.py — that
# module is stdlib-only and thus loadable before the splash screen, where `data_dir()`
# is already needed to redirect output (EXE.md step 2) while this module pulls in
# cv2+numpy. They're re-exported here, so every existing import keeps working unchanged.
from skate_environment import is_frozen, app_dir, data_dir     # noqa: F401


# MediaPipe is deliberately NOT imported at module level: that way this file can also
# be loaded in an environment without mediapipe (e.g. the YOLO venv, which reuses the
# shared functions). The import happens locally, inside analyze_frames().

# Fixed MediaPipe Pose 33-landmark connections (previously pulled from mp_vision),
# hardcoded here so drawing doesn't require a mediapipe import.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32),
]

# Lightweight landmark stand-in: has the same .x/.y/.z/.visibility fields as a
# MediaPipe NormalizedLandmark, so get_landmarks() and the drawing functions work with
# it unchanged. Used for smoothed points.
Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])

# ── Detection / tracking ─────────────────────────────────────────────────────────
# We let MediaPipe detect multiple people and pick the right one ourselves with our
# own tracker (TargetTracker), so detection doesn't jump to a different skater.
NUM_POSES_DEFAULT = 5     # max. number of skaters to detect at once
TRACK_GATE        = 0.14  # max. normalized jump of the torso centroid per frame
TRACK_GATE_GROWTH = 0.25  # ... which grows per missed frame: after a gap the prediction
                          # is less certain (the velocity estimate ages along with it)
TRACK_GATE_MAX    = 0.25  # upper bound on the grown gate — beyond that, "nearest" is no
                          # longer evidence it's the same skater
TRACK_RESEED_GATE = 0.25  # when reseeding after a long loss: max. distance to the
                          # (extrapolated) last known spot. Generous relative to
                          # TRACK_GATE (~8 frames of skating) but not more than that,
                          # since the prediction is already extrapolated along. Deliberately
                          # equal to TRACK_GATE_MAX, so acceptance doesn't jump at the
                          # coast→reseed transition. Nobody within this gate → better no
                          # pose than a skeleton on the wrong person (that produces
                          # plausible but wrong angles).
TRACK_LOST_S      = 0.5   # how long the target skater may be lost before we reseed (s)
TRACK_MIN_VIS     = 0.3   # minimum visibility for a landmark to count
# Cold start without a mouse click: don't blindly take "the biggest pose" — a bystander
# along the boards is regularly bigger in frame than the skater riding further away. We
# watch for a second first and then pick the biggest *mover* (median bbox × distance
# covered), the same way the YOLO backend does with MIN_MOVEMENT.
SEED_WARMUP_S     = 1.0   # how long we watch before the target skater is chosen (s)
SEED_MIN_MOVEMENT = 0.02  # less distance covered in that window = a stationary bystander
TORSO_IDX         = (11, 12, 23, 24)  # shoulders + hips: stable identity centroid

# ── Offline landmark smoothing (Savitzky-Golay) ─────────────────────────────────
# Because this is a batch tool and we have ALL the frames, we smooth bidirectionally
# (zero-lag) instead of causally. SG suppresses jitter but preserves peaks (e.g. full
# extension at the push) better than a plain moving average.
SMOOTH_WINDOW_S = 0.25    # window length in seconds (converted to an odd number of frames)
SMOOTH_POLY     = 2       # polynomial order of the SG fit
HORIZON_SMOOTH_S = 0.5    # window length (s) for smoothing the per-frame horizon estimate
# Space the skater takes up in frame per frame (for the automatic zoom in the GUI). The
# extent ripples along with the skating cycle (knees drop, legs spread), so we smooth
# over roughly one stroke — what's left is the slow change in distance to the camera.
BOX_SMOOTH_S   = 1.5      # window length (s) for smoothing the box size
BOX_CENTER_S   = 0.5      # window length (s) for smoothing the box center point;
                          # shorter, because the center really needs to follow the
                          # skater — only the wobble from arms and legs needs to come out
BOX_POLY       = 1        # linear, not SMOOTH_POLY: the distance to the camera changes
                          # locally in a straight line, whereas SG's quadratic edge fit
                          # extrapolates the cycle's ripple outward — measured 15% off
                          # on the first frame vs. 3% with a linear fit
BOX_MIN_FRAMES = 5        # fewer usable frames = no meaningful box signal
BOX_GAP_S      = 1.0      # a detection gap may be bridged with the last known box for
                          # this long; if it takes longer, we no longer know where the
                          # skater is and the box smoothly zooms back out to the full
                          # frame — better to see everything than a magnified view of
                          # the wrong spot

# ── Outlier rejection (occlusion) ───────────────────────────────────────────────
# If one limb hides another (e.g. an arm in front of a hip/leg) a landmark jumps
# briefly. We recognize such frames by low visibility or a jump far from the local
# median (Hampel), and interpolate them away before smoothing.
VIS_MIN        = 0.2      # landmark unreliable below this visibility
HAMPEL_WINDOW  = 7        # window length (frames) for the median/MAD
HAMPEL_K       = 4.5      # number of robust standard deviations for "outlier"
INTERP_MAX_S   = 0.1      # max. duration (s) of an unreliable stretch that gets
                          # interpolated away; longer stretches (e.g. motion blur over
                          # several frames) keep the raw detection — that sits ON the
                          # skater, a long straight-line interpolation puts the whole
                          # skeleton next to them
BLUR_FRAME_FRAC = 0.5     # if fewer than this fraction of the data-bearing joints in a
                          # frame counts as "reliable", that's a correlated confidence
                          # dip (motion blur) — trust the whole raw detection instead of
                          # interpolating the complete skeleton away

# ── Predictability checks (skating is a predictable motion) ────────────────────
# 1. L/R swap: with crossing/overlapping legs the pose model regularly flips left and
#    right. Hampel only catches that briefly — a persistent swap poisons the push-leg
#    cycle (the l-r ankle signal). We therefore track the continuity of the two leg
#    trajectories per joint pair and pick, per frame, the assignment with the smallest jump.
LR_SWAP_FACTOR = 0.8      # only swap if the swapped assignment clearly fits better
                          # (cost ratio); prevents flicker when the legs cross
# 2. Bone length: lower and upper leg are rigid — their pixel length should only change
#    slowly (distance to the camera). A sudden length jump is a detection error, even if
#    x and y each stay within their own Hampel band.

# ── Cycle-aware push-leg determination ───────────────────────────────────────────
# The ankle height difference (left vs. right) oscillates with the skating stroke. We
# smooth that signal and apply it with hysteresis, so the stance leg only switches on a
# real weight transfer (no per-frame flicker around the switch).
STANCE_SMOOTH_S = 0.18    # smoothing window (s) of the stance signal
STANCE_BAND_FRAC = 0.20   # hysteresis band as a fraction of the signal amplitude

# ── Push completion from leg extension ──────────────────────────────────────────
# "Leg fully extended = push done." Instead of a per-frame 2-of-3 vote with fixed
# thresholds, we detect the extension *maximum* as a peak of a smooth, scale-free
# signal (straight line hip→ankle / sum of bone lengths). No magic threshold needed and
# the angle is read exactly at the peak. Stroke timing is predictable: from the median
# stance-run length we estimate the half-stroke period as a soft prior against phantom pushes.
EXTENSION_SMOOTH_S      = 0.15   # smoothing window (s) of the extension-ratio signal
EXTENSION_MIN_STROKE_FRAC = 0.35 # a stance run shorter than this fraction of the half-
                             # stroke period is (almost certainly) a noise flip → no
                             # push. Kept low so fast opening strokes still stand.
EXTENSION_MIN_RUN_S     = 0.20   # absolute lower bound (s) for a stance run to count
                             # toward the "real stroke" median. Needed because the half-
                             # stroke period comes from the median run length: if the
                             # noise runs take part in that, the estimate — and thus the
                             # threshold — drops exactly when lots of L/R flips happen
                             # (see BUGS.md A3). A half skating stroke never takes less
                             # than ~0.2 s.
EXTENSION_PLATEAU_BAND  = 0.02   # the leg is "extended" for a whole phase (from standing
                             # up after placement to full sideways push). Frames where the
                             # extension ratio stays within this band under the per-run
                             # maximum count as "extended"; within that plateau we read
                             # off the flattest (lowest) lower-leg angle = the actual push
                             # angle (not the standing-up).
EXTENSION_MIN_SLOPE_DEG = 20.0   # minimum slope of the lower leg relative to vertical at
                             # push completion (so: push angle ≤ 90 - 20 = 70°). If a
                             # run's extension plateau covers only the standing-up phase,
                             # the "flattest angle" within that plateau is still the
                             # standing-up, and the tool would report a push of 74-83°.
                             # This isn't a tuning number but geometry: with a lower leg
                             # only 20° out of vertical, the sideways component of the
                             # push is sin(20°) ≈ 0.34 — there simply wasn't a sideways
                             # push. On the 18 saved analyses this threshold separates the
                             # three standing-up events (74.3 / 79.4 / 83.1°) from all 84
                             # healthy pushes; the steepest healthy measurement is 66.7°.
                             # Such an event doesn't disappear — it gets flagged (see
                             # INCOMPLETE_NO_PUSH) and falls outside avg/min/max.

# Reasons a push stays visible but falls outside the statistics
# (`FrameResult.push_incomplete` → `PushEvent.incomplete`). Deliberately one flag with a
# reason instead of two separate booleans: every place that filters the statistics only
# needs to check truthiness, while the GUI can tell the user WHAT went wrong — "the
# video ended" calls for a longer recording, "no full push observed" calls for a
# critical look at the leg assignment. The texts double as the marker that travels along
# in the library's events cache (schaats_db).
INCOMPLETE_TRUNCATED = "truncated"      # run runs to the end of the video/pose segment
INCOMPLETE_NO_PUSH   = "no full push"   # plateau covers only the standing-up

# ── Corner detection ───────────────────────────────────────────────────────────
# In a corner the body rotates around the vertical axis: the hips are no longer next to
# each other but one behind the other, so their horizontal distance in frame collapses
# while the torso stays the same length. `corner_ratio` = hip width / torso length is
# thus a scale-free "am I facing the camera?" signal (same principle as
# `_extension_ratio`): independent of distance to the camera, and specifically sensitive
# to exactly the rotation a corner produces. Measured over the 22 analyses in the library:
#   - 16 straight-section clips: median 0.75-1.20, lowest 0.5 s median 0.57
#   - the corner part of four long clips: median 0.21-0.24, lowest 0.05
# So a margin of well over 3×. Alternative denominators (femur, whole leg, shoulder
# width) all gave less separation (1.7-2.7×).
CORNER_IN        = 0.40  # below this (smoothed) ratio: corner — well under 0.57
CORNER_OUT       = 0.50  # above this ratio, straight again (hysteresis, like STANCE_BAND_FRAC)
CORNER_SMOOTH_S  = 0.5   # smoothing window (s); the ratio ripples slightly with the stroke
CORNER_MIN_S     = 0.6   # shorter than this isn't a corner but noise → leave as is
CORNER_MIN_TORSO_PX = 12 # below this torso length the ratio is pixel noise (the
                         # farthest skater in the library measures 25-35 px)

# ── Perspective correction (phase 7) ───────────────────────────────────────────
# The 3D reconstruction itself lives in skate_perspective.py; only the glue is here.
ANKLE_HEIGHT_M     = 0.10  # the ankle landmark sits at malleolus + skate height, not ON the ice
TRAVEL_WINDOW_S    = 0.4   # window (s) for the trajectory direction from world positions
TRAVEL_MIN_M       = 0.15  # minimum displacement in the window to trust the direction

# ── Landmark indices (MediaPipe Pose) ──────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP     = 23, 24
L_KNEE, R_KNEE   = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL   = 29, 30
L_TOE,  R_TOE    = 31, 32

# ── Colors (BGR) ────────────────────────────────────────────────────────────────
GREEN   = (80, 200, 80)
RED     = (60, 60, 220)
WHITE   = (255, 255, 255)
YELLOW  = (0, 210, 230)
DARK    = (20, 20, 20)
PURPLE  = (200, 100, 220)
SKELETON = (255, 200, 0)  # bright cyan-blue; clearly visible and doesn't clash with a green/red push leg


def _alias(new_name):
    """Builds a get/set property that mirrors `new_name`, so a call site that still
    uses the original Dutch field name keeps working without change. Transitional —
    see the translate-to-english plan: schaats_gui.py/schaats_yolo.py/schaats_db.py/
    schaats_eval.py/schaats_schermtest.py aren't translated yet and still read/write
    these fields by their old names. Remove each alias once nothing references it
    anymore. A plain (un-annotated) class attribute like this is not picked up by
    @dataclass as a field, so it coexists safely with the real ones."""
    def getter(self):
        return getattr(self, new_name)

    def setter(self, value):
        setattr(self, new_name, value)

    return property(getter, setter)


@dataclass
class VideoInfo:
    """Video properties, read without decoding frames."""
    w: int
    h: int
    fps: float
    totaal: int


@dataclass
class FrameResult:
    """Analysis result for one frame, without pixel data (cheap to cache)."""
    frame_nr: int
    time: float
    lm: object = None           # raw mediapipe landmarks (normalized), for skeleton drawing
    lm_data: dict = None        # pixel coordinates per landmark
    leg: str = None
    angle: float = None
    smooth_angle: float = None  # centered average of `angle` (display; zero-lag)
    knee_angle: float = None
    weight_on: bool = None
    signals: list = field(default_factory=list)
    extension_ratio: float = None  # stance-leg extension (0-1, ~1 = extended); peak = end of push
    push_incomplete: str = None  # None = a full push; otherwise the reason this stance
                                  # run's angle doesn't count as a measurement
                                  # (INCOMPLETE_TRUNCATED / INCOMPLETE_NO_PUSH) — both
                                  # produce an angle that's too steep
    pose_found: bool = False
    corner: bool = False        # the skater isn't facing the camera here (a corner) — no
                                 # derivatives are computed, so this frame yields no push
                                 # measurement. On the YOLO backend these are also the
                                 # frames the detection pass skipped inference on.
    horizon_deg: float = 0.0    # camera tilt relative to the ice at this frame (per-frame under auto)
    # Quality flag from the refinement pass (YOLO+RTMPose backend only): horizontal
    # deviation (px) of the knee point from the midline of the leg in the suit-color
    # mask. Filmed frontally, the joint should sit in the middle of the leg; a large
    # deviation flags frames where the measurement is shaky.
    midline_dev: dict = None    # {'l_knee': px, 'r_knee': px} (None = not measured)
    # Perspective correction (only filled with a calibration; None/True = no correction active):
    angle_correction: float = None  # corrected minus old image-plane angle (quality indicator)
    angle_reliable: bool = True     # False: leg nearly in the viewing direction / geometry didn't close
    world_xy: tuple = None      # position on the track in meters (for speed/stroke length)
    speed: float = None         # m/s (only if the line-distance scale is known)

    # Transitional Dutch-name aliases (see `_alias` above); remove once
    # schaats_gui.py/schaats_yolo.py/schaats_db.py/schaats_eval.py/schaats_schermtest.py
    # are translated and nothing references them anymore. `frame_nr` and `lm`/`lm_data`
    # keep their names outright — short and already language-neutral, not worth the
    # churn of renaming every call site across the whole project for them.
    tijd = _alias("time")
    been = _alias("leg")
    hoek = _alias("angle")
    smooth_hoek = _alias("smooth_angle")
    kniehoek = _alias("knee_angle")
    gewicht_erop = _alias("weight_on")
    signalen = _alias("signals")
    strek_ratio = _alias("extension_ratio")
    afzet_onvolledig = _alias("push_incomplete")
    pose_gevonden = _alias("pose_found")
    bocht = _alias("corner")
    middellijn_dev = _alias("midline_dev")
    hoek_correctie = _alias("angle_correction")
    hoek_betrouwbaar = _alias("angle_reliable")
    wereld_xy = _alias("world_xy")
    snelheid = _alias("speed")


@dataclass
class PushEvent:
    """One coherent push motion (weight on the same leg)."""
    index: int
    leg: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    angle: float      # angle at the completion frame itself (not averaged — see segment_pushes)
    min_angle: float
    max_angle: float
    note: str = None       # e.g. "alternation?" if L/R doesn't check out, or "merged"
    incomplete: str = None  # None = counts; otherwise the reason this push falls
                             # outside avg/min/max (INCOMPLETE_TRUNCATED /
                             # INCOMPLETE_NO_PUSH). The event stays visible — after all,
                             # something happened — but in both cases the angle is
                             # systematically too steep to measure.
    # Perspective correction (None without a calibration):
    correction: float = None    # correction applied at push completion (degrees)
    reliable: bool = True       # False: angle measured with the leg nearly in the viewing direction
    speed: float = None         # average speed during the push (m/s)
    stroke_length: float = None  # distance covered during the push (m)

    # Transitional Dutch-name aliases (see `_alias` above).
    been = _alias("leg")
    eind_frame = _alias("end_frame")
    start_tijd = _alias("start_time")
    eind_tijd = _alias("end_time")
    hoek = _alias("angle")
    min_hoek = _alias("min_angle")
    max_hoek = _alias("max_angle")
    opmerking = _alias("note")
    onvolledig = _alias("incomplete")
    correctie = _alias("correction")
    betrouwbaar = _alias("reliable")
    snelheid = _alias("speed")
    slaglengte = _alias("stroke_length")


@dataclass
class PerspectiveConfig:
    """Opt-in perspective correction via track lines (phase 7): a calibration from
    `skate_perspective.calibrate_from_lines` plus the reconstruction choices. Without
    this config the pipeline behaves exactly as before.

    `calibration_input` (CalibrationInput) is the storable origin of `calibration`. It's
    optional because the core also works with a calibration built up on its own
    (self-tests), but without a calibration_input the config can't be saved — `to_dict`
    explicitly refuses that instead of silently letting a correction evaporate."""
    calibration: object              # skate_perspective.PerspectiveCalibration
    method: str = "lower_leg"        # 'lower_leg' (sphere intersection) | 'leg_plane' (direction-of-travel plane)
    lower_leg_l: float = None        # lower-leg length in m (required for 'lower_leg')
    ankle_height: float = ANKLE_HEIGHT_M
    calibration_input: object = None  # skate_perspective.CalibrationInput

    def to_dict(self):
        """JSON-able form for `analyse.instellingen_json`."""
        if self.calibration_input is None:
            raise ValueError("this PerspectiveConfig has no CalibrationInput and "
                             "can therefore not be saved")
        return {
            "calibration_input": self.calibration_input.to_dict(),
            "method": self.method,
            "lower_leg_l": None if self.lower_leg_l is None else float(self.lower_leg_l),
            "ankle_height": float(self.ankle_height),
        }

    @classmethod
    def from_dict(cls, d):
        """Reads either the current English keys or the original Dutch ones (both the
        top-level keys and, via CalibrationInput.from_dict, the nested ones), so a
        config saved before the English rename still loads correctly."""
        inv_dict = d["calibration_input"] if "calibration_input" in d else d["invoer"]
        calibration_input = skate_perspective.CalibrationInput.from_dict(inv_dict)
        return cls(calibration=calibration_input.calibrate(),
                   method=d.get("method", d.get("methode", "lower_leg")),
                   lower_leg_l=d.get("lower_leg_l", d.get("onderbeen_l")),
                   ankle_height=d.get("ankle_height", d.get("enkel_hoogte", ANKLE_HEIGHT_M)),
                   calibration_input=calibration_input)

    # Transitional Dutch-name aliases (see `_alias` above) — schaats_gui.py still reads
    # and constructs these by their old names.
    kalibratie = _alias("calibration")
    methode = _alias("method")
    onderbeen_l = _alias("lower_leg_l")
    enkel_hoogte = _alias("ankle_height")
    invoer = _alias("calibration_input")
    naar_dict = to_dict
    uit_dict = classmethod(from_dict.__func__)


def video_info(input_pad, force_fps=None):
    """Reads video properties without decoding frames."""
    cap = cv2.VideoCapture(input_pad)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {input_pad}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = force_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(w, h, fps, totaal)


# ---------------------------------------------------------------------------
# Interlacing (combing)
# ---------------------------------------------------------------------------
# A camcorder recording 1080i50 fires a half image 50 times a second (first the even
# rows, then the odd ones) and weaves those two moments, 1/50 s apart, into one frame.
# On still parts you don't notice; on a moving leg the even and odd rows sit at a
# different spot — the combing. A media player deinterlaces on playback (via the GPU),
# OpenCV does not: it delivers the woven frame exactly as it sits in the file, so both
# the image AND the pose detection see the comb.
#
# Measured on `00005.MTS` (AVCHD 1080i50) against the progressive phone clips: the two
# fields sit **6.1 px** apart on knees and ankles (p90 13.4; max 44), against 0.06 px on
# progressive material. With "2 px keypoint error = 2-4° angle error" (OPNAME.md), that's
# the single largest source of noise in that material — bigger than anything left to
# gain algorithmically.
DEINT_THRESHOLD    = 8      # per-pixel comb threshold on grayscale values
DEINT_MIN_COMB_PX  = 2000   # less combing in a frame = too little motion to judge
DEINT_TEST_FRAMES  = 24     # measurements `is_interlaced` collects
DEINT_TEST_MAX     = 300    # frames it reads through at most to get them
DEINT_SHIFT        = 0.5    # px field shift above which a video counts as interlaced
_DEINT_WINDOW      = 64     # half window size for the phase correlation
_DEINT_KERNEL = np.ones((7, 3), np.uint8)


def _comb_mask(gray_i16, threshold=DEINT_THRESHOLD):
    """
    Per pixel: does this row deviate from BOTH vertical neighbors in the same
    direction? That's the signature of a comb — with ordinary image detail a row sits
    somewhere between its neighbors. Expects int16 (uint8 overflows on the difference).
    """
    m = gray_i16[1:-1]
    # Instead of `(a * b) > d**2` with three 2-megapixel temporary arrays: write the
    # product into `a` itself. Bit-identical (int16 wraps just as hard either way) and
    # ~2 ms faster on 1080p — nothing in the analysis, but during playback of camcorder
    # footage every millisecond of the 40 ms per-frame budget counts.
    a = m - gray_i16[:-2]
    a *= m - gray_i16[2:]
    return a > threshold * threshold


def deinterlace(frame, threshold=DEINT_THRESHOLD):
    """
    Removes the combing from one frame: wherever the image combs, the **odd** rows are
    discarded and interpolated from the even rows, so the moving part comes from
    exactly one moment. Still parts stay untouched at full vertical resolution.

    **One field has to win.** The obvious variant — averaging both fields — does
    remove the comb but leaves the temporal blend in place: every output row is then
    still a mix of two moments. Measured, the field shift on knee/ankle went from 6.07
    to 5.52 px that way, against **0.03 px** with this version (ffmpeg's `yadif`
    reaches 0.07 px). Costs ~12 ms per 1080p frame — negligible next to the ~2 s/frame
    of the detection pass, but noticeable during playback: there the whole budget is
    40 ms per frame at 25 fps, and decoding (~8 ms) plus scaling to a HiDPI screen
    (~15 ms) is already part of that. Hence the two heaviest steps are written to be
    cheap; see `_comb_mask` and the `cv2.copyTo` below.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    # Smear vertically so a comb region is treated as a whole and no isolated rows drop
    # out; [0::2] then keeps the odd absolute rows.
    comb = cv2.dilate(_comb_mask(g, threshold).view(np.uint8), _DEINT_KERNEL).view(bool)[0::2]
    out = frame.copy()
    interp = cv2.addWeighted(frame[0:-2:2], 0.5, frame[2::2], 0.5, 0.0)
    # The merge step via OpenCV instead of numpy: `out[1:-1:2] = np.where(comb[:,:,None],
    # interp, original)` costs 8.9 ms, `cv2.copyTo` on a contiguous copy 1.1 ms —
    # identical output, byte for byte. The difference is that numpy broadcasts a bool
    # mask over three channels here and builds a whole new array, whereas OpenCV writes
    # only the masked bytes with SIMD across 16 threads. The detour via a copy is needed
    # because cv2 can't write into a strided view (every other row); that copy is cheap.
    odd = frame[1:-1:2].copy()
    cv2.copyTo(interp, comb.view(np.uint8), odd)
    out[1:-1:2] = odd
    return out


def _field_shift(gray_i16, cy, cx):
    """
    How many pixels the image shifts between the two fields, measured around (cy, cx)
    by laying the even and odd rows over each other with phase correlation. None if the
    frame is too small or the correlation yields nothing.
    """
    h, w = gray_i16.shape
    if h < 2 * _DEINT_WINDOW or w < 2 * _DEINT_WINDOW:
        return None
    cy = int(np.clip(cy, _DEINT_WINDOW, h - _DEINT_WINDOW))
    cx = int(np.clip(cx, _DEINT_WINDOW, w - _DEINT_WINDOW))
    crop = gray_i16[cy - _DEINT_WINDOW:cy + _DEINT_WINDOW,
                     cx - _DEINT_WINDOW:cx + _DEINT_WINDOW].astype(np.float32)
    a, b = crop[0::2], crop[1::2]
    n = min(len(a), len(b))
    window = cv2.createHanningWindow((a.shape[1], n), cv2.CV_32F)
    (dx, _), response = cv2.phaseCorrelate(a[:n] * window, b[:n] * window, window)
    return abs(dx) if response > 0.15 else None


def is_interlaced(input_pad, threshold_px=DEINT_SHIFT):
    """
    Determines whether a video is interlaced: do the two fields of a frame come from
    the same moment (progressive) or 1/50 s apart (interlaced)?

    Per frame the **densest comb cluster** is located — that's the moving object — and
    there the even and odd rows are laid over each other with phase correlation.
    Aiming for the median comb pixel doesn't work: with little motion the comb is
    scattered compression noise and the window lands on still ice (measured: 6 of 27
    clips wrong).

    The **comb fraction alone is equally insufficient**, however tempting and cheap:
    one small, heavily compressed clip (832x464) scored 0.035 on it against ≤0.0025 for
    all the other progressive material, and would thus be filtered incorrectly. Only
    the shift itself separates the two cases, since it's 0 by definition when both
    fields come from the same moment, whatever compression was applied on top.

    Calibrated over the 27 videos in the library (2026-08-26): the eleven camcorder
    clips measure 1.59-17.41 px, the sixteen progressive clips 0.00-0.15 px — 0
    errors, ~3x margin on both sides of the threshold. Too few moving frames to judge →
    False: better not to filter than to touch pixels needlessly.
    """
    cap = cv2.VideoCapture(input_pad)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {input_pad}")
    measurements, seen = [], 0
    try:
        while len(measurements) < DEINT_TEST_FRAMES and seen < DEINT_TEST_MAX:
            ret, frame = cap.read()
            if not ret:
                break
            seen += 1
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
            comb = _comb_mask(g)
            if comb.sum() < DEINT_MIN_COMB_PX:
                continue                      # too little motion: this frame says nothing
            density = cv2.blur(comb.astype(np.float32), (2 * _DEINT_WINDOW,) * 2)
            _, _, _, (mx, my) = cv2.minMaxLoc(density)
            d = _field_shift(g, my + 1, mx)   # +1: `comb` starts at row 1
            if d is not None:
                measurements.append(d)
    finally:
        cap.release()
    if len(measurements) < 5:
        return False
    # p75, not the median: the question is whether the footage EVER shows a field
    # shift, and some of the frames simply catch a moment with little motion.
    return float(np.percentile(measurements, 75)) > threshold_px


class VideoReader:
    """
    `cv2.VideoCapture` with the comb filter on read()/retrieve().

    Everything else (`grab`, `set`, `get`, `isOpened`, `release`) goes through
    `__getattr__` to the capture, so every existing read loop keeps working unchanged —
    the same pass-through pattern as the DirectML shell in schaats_yolo.
    """

    def __init__(self, input_pad, threshold=DEINT_THRESHOLD):
        self._cap = cv2.VideoCapture(input_pad)
        self._threshold = threshold

    def read(self):
        ret, frame = self._cap.read()
        return (True, deinterlace(frame, self._threshold)) if ret else (ret, frame)

    def retrieve(self, *args):
        ret, frame = self._cap.retrieve(*args)
        return (True, deinterlace(frame, self._threshold)) if ret else (ret, frame)

    def __getattr__(self, name):
        if name == "_cap":                    # not set yet: no infinite recursion
            raise AttributeError(name)
        return getattr(self._cap, name)


def open_video(input_pad, deinterlacen=False):
    """
    Opens a video, with or without the comb filter. **Without, it's literally a
    `cv2.VideoCapture`** — so nothing changes for progressive material, not even an
    extra Python call per frame.
    """
    return VideoReader(input_pad) if deinterlacen else cv2.VideoCapture(input_pad)


def calculate_angle_to_ice(enkel_xy, knie_xy, horizon_deg=0.0):
    """
    Calculate the angle of the leg relative to the ice.
    Returns the angle in degrees (0° = parallel to the ice, 90° = perpendicular to it).

    `horizon_deg` is the tilt of the true ice line relative to the horizontal image
    axis (positive = the ice line rises to the right); it's subtracted from the
    measured angle so a tilted camera doesn't pollute the push angle. For small tilts
    (the practical case) this subtraction is accurate enough; at a large tilt you'd
    have to rotate the landmarks first (the `abs(dx)` assumption starts to strain then).
    """
    dx = knie_xy[0] - enkel_xy[0]
    dy = enkel_xy[1] - knie_xy[1]   # y-axis flipped in image coordinates
    hoek = float(np.degrees(np.arctan2(dy, abs(dx))))
    return round(hoek - horizon_deg, 1)     # a plain float: goes straight into the DB/CSV/JSON


def horizon_angle_from_line(p1, p2):
    """
    Tilt angle (degrees) of a reference line (e.g. along the ice or the boards)
    relative to the horizontal image axis. Positive = the line rises to the right.
    `p1`/`p2` are pixel (x, y) points; the direction is normalized (p2 on the right).
    Scale-invariant, so points in scaled display coordinates work too.
    """
    (x1, y1), (x2, y2) = p1, p2
    if x2 < x1:                                  # p2 always to the right of p1
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
    return round(float(np.degrees(np.arctan2(-(y2 - y1), (x2 - x1) or 1e-9))), 2)


def detect_ice_line(frame_bgr, max_kanteling=20.0, roi_onder=0.45, min_lijnen=2):
    """
    Automatically estimates the camera tilt from one frame: looks for strong, nearly
    horizontal edges (ice line, boards, ad banner) with a Hough transform and takes the
    length-weighted median of their tilt. Returns `horizon_deg` (float), or None if
    there's no reliable horizontal line.

    Only the bottom part of the frame is examined (`roi_onder` = fraction of height,
    that's where the ice is); lines steeper than `max_kanteling`° are rejected
    (vertical board edges, legs). This is a *suggestion* — let the user confirm it.
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * (1.0 - roi_onder))
    roi = frame_bgr[y0:h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=int(w * 0.25), maxLineGap=20)
    if lines is None:
        return None

    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        hoek = horizon_angle_from_line((x1, y1), (x2, y2))
        if abs(hoek) <= max_kanteling:
            angles.append(hoek)
            weights.append(float(np.hypot(x2 - x1, y2 - y1)))   # longer line = stronger
    if len(angles) < min_lijnen:
        return None

    order = np.argsort(angles)
    angles = np.array(angles)[order]
    cum = np.cumsum(np.array(weights)[order])
    median = angles[int(np.searchsorted(cum, cum[-1] / 2.0))]   # length-weighted median
    return round(float(median), 2)


def calculate_knee_angle(heup, knie, enkel):
    """Angle at the knee joint (hip-knee-ankle)."""
    a = np.array(heup)
    b = np.array(knie)
    c = np.array(enkel)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cos_a, -1, 1))), 1)


def get_landmarks(lm, w, h):
    """Converts normalized landmarks to pixel coordinates."""
    def pt(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))
    def vis(idx):
        return lm[idx].visibility

    return {
        'l_hip':    pt(L_HIP),   'r_hip':    pt(R_HIP),
        'l_knee':   pt(L_KNEE),  'r_knee':   pt(R_KNEE),
        'l_ankle':  pt(L_ANKLE), 'r_ankle':  pt(R_ANKLE),
        'l_heel':   pt(L_HEEL),  'r_heel':   pt(R_HEEL),
        'l_toe':    pt(L_TOE),   'r_toe':    pt(R_TOE),
        'vis_l_ankle': vis(L_ANKLE), 'vis_r_ankle': vis(R_ANKLE),
        'vis_l_knee':  vis(L_KNEE),  'vis_r_knee':  vis(R_KNEE),
    }


def determine_push_leg(lm_data, heup_history, w):
    """
    Determines which leg is the push leg based on:
    1. Which leg is lower/further outward (lowest ankle = closest to the ice)
    2. Direction of hip shift
    """
    l_ankle_y = lm_data['l_ankle'][1]
    r_ankle_y = lm_data['r_ankle'][1]

    # Higher y value = lower in frame = closer to the ice
    if abs(l_ankle_y - r_ankle_y) < 10:
        # Ankles at equal height: use the hip shift as a tiebreaker. `heup_history`
        # holds tuples (l_hip_x, r_hip_x), so compare the hip *midpoint* with the one
        # from 3 frames ago — same as `detect_weight_on_leg`. (Previously the
        # current right hip was compared against the left hip of back then: that
        # measured not the shift but the constant hip width, so the tiebreaker always
        # answered 'links'.)
        if len(heup_history) >= 3:
            mid_now     = (lm_data['l_hip'][0] + lm_data['r_hip'][0]) / 2
            mid_earlier = (heup_history[-3][0] + heup_history[-3][1]) / 2
            return 'links' if mid_now - mid_earlier > 0 else 'rechts'
        return 'rechts'

    return 'links' if l_ankle_y > r_ankle_y else 'rechts'


def detect_weight_on_leg(been, lm_data, enkel_history, heup_history, w, h, threshold):
    """
    Determines whether the weight is still on the push leg.

    Signals that weight is OFF the leg:
    - Ankle rises suddenly (leg leaves the ice)
    - Knee is nearly straight (extension > 160°)
    - Hip moves quickly away from the leg
    """
    if been == 'links':
        enkel = lm_data['l_ankle']
        knie  = lm_data['l_knee']
        heup  = lm_data['l_hip']
    else:
        enkel = lm_data['r_ankle']
        knie  = lm_data['r_knee']
        heup  = lm_data['r_hip']

    weight_signals = []

    # Signal 1: ankle-y stable? (relative to frame size)
    if len(enkel_history[been]) >= 4:
        recent_y = [e[1] for e in list(enkel_history[been])[-4:]]
        rise = recent_y[0] - recent_y[-1]   # positive = up
        if rise / h > 0.015:
            weight_signals.append('ankle_up')

    # Signal 2: leg extended? (knee extension)
    kniehoek = calculate_knee_angle(
        (heup[0]/w, heup[1]/h),
        (knie[0]/w, knie[1]/h),
        (enkel[0]/w, enkel[1]/h)
    )
    if kniehoek > 162:
        weight_signals.append('leg_extended')

    # Signal 3: hip shifts away from the leg
    if len(heup_history) >= 4:
        hip_mid_now     = (lm_data['l_hip'][0] + lm_data['r_hip'][0]) / 2 / w
        hip_mid_earlier = (heup_history[-4][0] + heup_history[-4][1]) / 2 / w

        shift = hip_mid_now - hip_mid_earlier
        # For the left leg: the hip shifts right (positive) as weight transfers
        if been == 'links' and shift > threshold:
            weight_signals.append('hip_away')
        elif been == 'rechts' and shift < -threshold:
            weight_signals.append('hip_away')

    # Weight off if 2+ signals are present
    weight_on = len(weight_signals) < 2
    return weight_on, kniehoek, weight_signals


def draw_leg_overlay(frame, lm_data, been, hoek, gewicht_erop, kniehoek, horizon_deg=0.0):
    """Draws the leg overlay with angle and color coding."""
    if lm_data is None or been not in ('links', 'rechts'):
        return                        # corner frame or not yet processed: nothing to draw
    kleur = GREEN if gewicht_erop else RED

    if been == 'links':
        heup  = lm_data['l_hip']
        knie  = lm_data['l_knee']
        enkel = lm_data['l_ankle']
        hiel  = lm_data['l_heel']
    else:
        heup  = lm_data['r_hip']
        knie  = lm_data['r_knee']
        enkel = lm_data['r_ankle']
        hiel  = lm_data['r_heel']

    # Leg line thicker than normal
    cv2.line(frame, heup, knie, kleur, 4)
    cv2.line(frame, knie, enkel, kleur, 4)

    # Ice line at the ankle — tilted according to the configured horizon (0° = horizontal)
    ice_line_len = 60
    rad = np.radians(horizon_deg)
    ex = int(np.cos(rad) * ice_line_len)
    ey = int(np.sin(rad) * ice_line_len)   # positive horizon = rising to the right (y down)
    cv2.line(frame,
             (enkel[0] - ex, enkel[1] + ey),
             (enkel[0] + ex, enkel[1] - ey),
             WHITE, 2)

    # Angle line, extended (ankle → toward the knee, but projected onto the ground). An
    # angle ≤ 0 means the knee sits below the ankle: not a skating stance but a broken
    # detection. We draw that line anyway, in red — silently omitting it would hide the
    # problem while the table just reports the value.
    line_len = 80
    dir_x = int(np.sin(np.radians(hoek)) * line_len * (-1 if been == 'links' else 1))
    dir_y = -int(np.cos(np.radians(hoek)) * line_len)
    end = (enkel[0] + dir_x, enkel[1] + dir_y)
    cv2.line(frame, enkel, end, YELLOW if hoek > 0 else RED, 2, cv2.LINE_AA)

    # Joints
    for punt in [heup, knie, enkel]:
        cv2.circle(frame, punt, 7, kleur, -1)
        cv2.circle(frame, punt, 7, WHITE, 1)

    # Angle text at the ankle
    tekst_pos = (enkel[0] + 14, enkel[1] - 14)
    cv2.putText(frame, f"{hoek} deg", tekst_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, DARK, 4)
    cv2.putText(frame, f"{hoek} deg", tekst_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 2)


def draw_hud(frame, been, hoek, gewicht_erop, kniehoek, smooth_hoek, frame_nr, fps, w, h,
              correctie=None, betrouwbaar=True, snelheid=None, strek_ratio=None):
    """Draws the HUD panel top-left. `correctie`/`betrouwbaar`/`snelheid` are the
    perspective-correction fields (only shown when a calibration was active).
    `strek_ratio` shows the leg extension (push detection: peak = end of push)."""
    tijd = frame_nr / fps if fps > 0 else 0
    status_color = GREEN if gewicht_erop else RED
    status_text = "WEIGHT ON LEG" if gewicht_erop else "PUSH COMPLETE"

    lines = [
        (f"t = {tijd:.2f}s  |  frame {frame_nr}", WHITE, 0.45),
        (f"Push leg: {been.upper()}", WHITE, 0.55),
        (f"Push angle: {hoek} deg  (avg: {smooth_hoek} deg)", YELLOW, 0.65),
        (f"Knee angle:  {kniehoek} deg", PURPLE, 0.55),
        (status_text, status_color, 0.6),
    ]
    if correctie is not None:
        lines.insert(3, (f"Persp. corr.: {correctie:+.1f} deg", WHITE, 0.5))
        if snelheid is not None:
            lines.insert(4, (f"Speed: {snelheid:.1f} m/s", WHITE, 0.5))
        if not betrouwbaar:
            lines.append(("ANGLE UNRELIABLE (viewing direction)", RED, 0.45))

    if strek_ratio is not None:           # leg extension: peak = push done
        lines.insert(len(lines) - 1, (f"Extension: {strek_ratio:.2f}", PURPLE, 0.5))

    panel_h = 26 * len(lines) + 40      # 5 lines → 170, as before
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (280, panel_h), (80, 80, 80), 1)

    y = 36
    for tekst, kleur, schaal in lines:
        cv2.putText(frame, tekst, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, schaal, DARK, 3)
        cv2.putText(frame, tekst, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, schaal, kleur, 1)
        y += 26


def draw_all_landmarks(frame, landmarks, w, h, min_vis=0.2):
    """
    Draws all pose landmarks lightly in the background. Landmarks with too low
    visibility are skipped (so the YOLO backend, which sets missing 33-slots to
    visibility 0, doesn't draw lines to (0,0)).
    """
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    visible = [getattr(l, 'visibility', 1.0) >= min_vis for l in landmarks]
    for a, b in POSE_CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(frame, pts[a], pts[b], SKELETON, 2, cv2.LINE_AA)
    for p, vb in zip(pts, visible):
        if vb:
            cv2.circle(frame, p, 3, SKELETON, -1, cv2.LINE_AA)


def draw_overlay_on_frame(frame, resultaat, fps, toon_skelet=True, toon_afzetbeen=True,
                           toon_hud=True):
    """
    Draws the overlay for one frame from a FrameResult, with each layer toggleable.
    Shared by the CLI video export and the GUI live display. The drawn ice line tilts
    along with `resultaat.horizon_deg` (per frame), so the overlay shows the reference
    used — even with a wobbling camera.
    """
    h, w = frame.shape[:2]

    if not resultaat.pose_gevonden:
        cv2.putText(frame, "No pose detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        return

    if toon_skelet:
        draw_all_landmarks(frame, resultaat.lm, w, h)

    if resultaat.bocht:
        # The skeleton stays put (you want to see THAT someone's riding there), but
        # nothing was measured here — say so instead of showing an empty HUD.
        cv2.putText(frame, "CORNER - not measured", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        return

    if toon_afzetbeen:
        draw_leg_overlay(frame, resultaat.lm_data, resultaat.been, resultaat.hoek,
                            resultaat.gewicht_erop, resultaat.kniehoek, resultaat.horizon_deg)

    if toon_hud:
        draw_hud(frame, resultaat.been, resultaat.hoek, resultaat.gewicht_erop,
                  resultaat.kniehoek, resultaat.smooth_hoek, resultaat.frame_nr, fps, w, h,
                  correctie=resultaat.hoek_correctie, betrouwbaar=resultaat.hoek_betrouwbaar,
                  snelheid=resultaat.snelheid, strek_ratio=resultaat.strek_ratio)


def _visible_xy(lm, idxs=None, min_vis=TRACK_MIN_VIS):
    """Normalized (x,y) of sufficiently visible landmarks; optionally restricted to idxs."""
    bron = lm if idxs is None else [lm[i] for i in idxs]
    return [(l.x, l.y) for l in bron if l.visibility >= min_vis]


def torso_centroid(lm):
    """
    Stable identity centroid (shoulders + hips), normalized. The torso moves more
    calmly than the limbs, so this is a reliable anchor for recognizing the same
    skater frame after frame. None if too little is visible.
    """
    pts = _visible_xy(lm, TORSO_IDX)
    if not pts:
        pts = _visible_xy(lm)          # fall back to all visible points
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def corner_ratio(lm, w, h, min_vis=VIS_MIN):
    """
    "Am I facing the camera?" as a scale-free number: hip width divided by torso
    length (shoulder midpoint → hip midpoint), both in pixels. Facing the camera, the
    hips sit side by side (~0.6-1.3); rotate into the corner and they line up one
    behind the other, so the width collapses (~0.2) while the torso stays the same length.

    Both measures in pixels (not normalized), otherwise the aspect ratio would skew the
    ratio. None if the four landmarks aren't visible or the torso is too small to say
    anything (`CORNER_MIN_TORSO_PX`) — at that distance the width is pixel noise. Works
    on any list of landmark objects with .x/.y/.visibility, so also on a YOLO
    `Detection.lm` during the detection pass.
    """
    try:
        sl, sr, hl, hr = (lm[L_SHOULDER], lm[R_SHOULDER], lm[L_HIP], lm[R_HIP])
    except (TypeError, IndexError):
        return None
    if min(p.visibility for p in (sl, sr, hl, hr)) < min_vis:
        return None
    hip_w = abs(hl.x - hr.x) * w
    torso = float(np.hypot(((sl.x + sr.x) - (hl.x + hr.x)) / 2 * w,
                          ((sl.y + sr.y) - (hl.y + hr.y)) / 2 * h))
    if torso < CORNER_MIN_TORSO_PX:
        return None
    return hip_w / torso


def _bbox(lm):
    """Bounding box (minx, miny, maxx, maxy) around the visible landmarks, or None."""
    pts = _visible_xy(lm)
    if len(pts) < 4:
        return None
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(lm):
    """Area of the bounding box around the visible landmarks (normalized)."""
    box = _bbox(lm)
    if box is None:
        return 0.0
    return (box[2] - box[0]) * (box[3] - box[1])


def _track_candidates(buffer, gate=TRACK_GATE, max_gat=3):
    """
    Strings the poses from a sequence of frames into raw candidate tracks: each pose is
    linked to the track whose latest torso centroid is closest (within `gate`, and no
    older than `max_gat` frames), otherwise a new track starts.

    Deliberately simpler than `TargetTracker` — for the cold start it only needs to be
    good enough to tell "skating" apart from "standing still along the boards".
    Returns dicts with `start` (first frame index), `punten`, and `opp`.
    """
    sporen = []
    for i, poses in enumerate(buffer):
        centroids = [(c, p) for c, p in ((torso_centroid(p), p) for p in poses)
                     if c is not None]
        for c, p in centroids:
            beste, beste_afst = None, gate
            for s in sporen:
                if s['laatst'] == i or i - s['laatst'] > max_gat:
                    continue            # this frame already spoken for, or the track expired
                d = ((s['punten'][-1][0] - c[0]) ** 2 + (s['punten'][-1][1] - c[1]) ** 2) ** 0.5
                if d < beste_afst:
                    beste, beste_afst = s, d
            if beste is None:
                sporen.append({'start': i, 'laatst': i, 'punten': [c],
                               'opp': [_bbox_area(p)]})
            else:
                beste['punten'].append(c)
                beste['opp'].append(_bbox_area(p))
                beste['laatst'] = i
    return sporen


def _path_length(punten):
    """Total distance covered along a sequence of normalized points."""
    return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
               for a, b in zip(punten, punten[1:]))


def _choose_moving_target(buffer):
    """
    From the first frames, picks the target skater as the **biggest mover**: median
    bbox area x distance covered, with `SEED_MIN_MOVEMENT` as a lower bound. Without
    that criterion, a cold start simply picks the biggest pose in frame — and that's
    regularly a bystander along the boards, who's closer to the camera than the skater
    riding further away.

    Returns `(start frame, centroid at that frame)`, or None if there's nothing to pick.
    """
    sporen = _track_candidates(buffer)
    if not sporen:
        return None
    movers = [s for s in sporen if _path_length(s['punten']) >= SEED_MIN_MOVEMENT]
    beste = max(movers or sporen,
                key=lambda s: float(np.median(s['opp']))
                              * max(_path_length(s['punten']), 1e-6))
    return beste['start'], beste['punten'][0]


class TargetTracker:
    """
    Tracks one target skater through the frames. MediaPipe delivers a list of poses per
    frame (multiple skaters); this tracker always picks the pose that best matches the
    target's predicted position, with a distance gate so tracking doesn't jump to a
    different skater when they cross.

    Seeding: if `doel_punt` (normalized (x,y)) is given, the skater closest to it is
    chosen. That point comes from a mouse click, or — on a cold start without a click —
    from `_choose_moving_target`, which points out the biggest *mover* after a warmup
    window. If the point is still missing, the tracker falls back to the biggest pose
    in frame.

    Two things are explicitly **coast-aware** (a detection gap of a few frames is
    nothing special with motion blur):
    - the prediction shifts along per missed frame (`n x speed`, not `1 x speed`) and
      the gate grows with the gap length — otherwise the skater falls outside the gate
      after ~3 missed frames and is permanently lost, even though they're being
      detected just fine;
    - the last known spot (`laatste_bekend`) is tracked separately from the lock flag
      (`centroid`), so a reseed after a long loss can pick up from there instead of
      from the (stale) click point of frame 0 or "the biggest pose in frame".
    """
    def __init__(self, doel_punt=None, gate=TRACK_GATE, hervind_frames=15):
        self.doel_punt = doel_punt
        self.gate = gate
        self.hervind_frames = hervind_frames
        self.centroid = None         # torso centroid at the last match; None = no lock
        self.laatste_bekend = None   # same, but also stays set after a loss (for reseeding)
        self.snelheid = (0.0, 0.0)   # estimated displacement per frame
        self.kwijt = 0               # number of consecutive frames without a match

    def _verwacht(self, vanaf, stappen):
        """Constant-speed prediction `stappen` frames ahead from `vanaf`. The horizon is
        capped at `hervind_frames` and the result is clamped to the frame: an aging
        speed estimate x a long gap would otherwise put the skater off-frame, after
        which nobody falls within any gate at all."""
        stap = min(max(0, stappen), self.hervind_frames)
        return (min(1.0, max(0.0, vanaf[0] + stap * self.snelheid[0])),
                min(1.0, max(0.0, vanaf[1] + stap * self.snelheid[1])))

    def _seed(self, centroids):
        """Picks a starting pose from [(centroid, pose), ...] (all centroids != None),
        or None if nothing credible is among them (only on a reseed)."""
        if self.laatste_bekend is not None:
            # Reseed after a long loss: take the skater closest to the last known spot,
            # extrapolated over the loss duration. Nobody within the generous gate → no
            # lock (better a gap than following the wrong person).
            ex, ey = self._verwacht(self.laatste_bekend, self.kwijt)
            beste = min(centroids, key=lambda cp: (cp[0][0]-ex)**2 + (cp[0][1]-ey)**2)
            afstand = ((beste[0][0]-ex)**2 + (beste[0][1]-ey)**2) ** 0.5
            return beste if afstand <= TRACK_RESEED_GATE else None
        if self.doel_punt is not None:
            dx, dy = self.doel_punt
            # On a mouse click, what mainly counts is WHICH skater you pointed at: give
            # priority to the skater whose bounding box contains the click point, so it
            # doesn't lock onto a different skater whose torso centroid happens to be
            # closer to the click (e.g. if you click on the skates/lower legs instead
            # of the torso).
            def _in_box(cp):
                box = _bbox(cp[1])
                return box is not None and box[0] <= dx <= box[2] and box[1] <= dy <= box[3]
            binnen = [cp for cp in centroids if _in_box(cp)]
            kandidaten = binnen if binnen else centroids
            return min(kandidaten, key=lambda cp: (cp[0][0]-dx)**2 + (cp[0][1]-dy)**2)
        # Cold start without a click: follow the biggest (most prominent) skater.
        return max(centroids, key=lambda cp: _bbox_area(cp[1]))

    def update(self, poses):
        """Picks the target pose for this frame; returns the landmark list or None."""
        centroids = [(torso_centroid(p), p) for p in poses]
        centroids = [(c, p) for c, p in centroids if c is not None]
        if not centroids:
            self.kwijt += 1
            return None

        # No lock yet, or lost too long → (re)seed.
        if self.centroid is None or self.kwijt > self.hervind_frames:
            gekozen = self._seed(centroids)
            if gekozen is None:
                self.kwijt += 1      # reseed refused: keep coasting (no pose this frame)
                return None
            c, p = gekozen
            self.centroid = self.laatste_bekend = c
            self.snelheid = (0.0, 0.0)
            self.kwijt = 0
            return p

        # Predict the position — extrapolated over the missed frames — and pick the
        # closest pose within the gate, which grows along with the gap length.
        n = self.kwijt + 1                     # frames since the last match
        px, py = self._verwacht(self.centroid, n)
        poort = min(self.gate * (1.0 + TRACK_GATE_GROWTH * self.kwijt), TRACK_GATE_MAX)
        (bc, bp), afstand = min(
            (((c, p), ((c[0]-px)**2 + (c[1]-py)**2) ** 0.5) for c, p in centroids),
            key=lambda t: t[1],
        )
        if afstand > poort:
            # Best candidate too far → probably the other skater; coast.
            self.kwijt += 1
            return None

        # Match: update speed (per frame, so divided by the gap length) and position,
        # both lightly damped.
        vx, vy = (bc[0] - self.centroid[0]) / n, (bc[1] - self.centroid[1]) / n
        self.snelheid = (0.5 * self.snelheid[0] + 0.5 * vx,
                         0.5 * self.snelheid[1] + 0.5 * vy)
        self.centroid = self.laatste_bekend = bc
        self.kwijt = 0
        return bp


def analyze_frames(input_pad, model_pad, force_fps=None, num_poses=NUM_POSES_DEFAULT,
                      doel_punt=None, progress_callback=None, deinterlacen=False):
    """
    Generator: detects ALL skaters per frame (multi-pose) and tracks the target skater
    with a TargetTracker. Yields a FrameResult per frame with only the raw landmarks
    (lm) of the target + pose_found — the derived quantities (push leg/angle/knee/
    weight) are only filled in after offline smoothing, by process_derivatives().
    `progress_callback(frame_nr, totaal)` is called after every frame.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    cap = open_video(input_pad, deinterlacen)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {input_pad}")

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

    tracker  = TargetTracker(doel_punt=doel_punt,
                           hervind_frames=int(max(1, fps * TRACK_LOST_S)))
    frame_nr = 0
    # Cold start without a click: watch for a second first and only then choose (see
    # _choose_moving_target). With a click there's nothing to choose and we analyze right away.
    warmup = 0 if doel_punt is not None else int(max(1, round(fps * SEED_WARMUP_S)))
    buffer = [] if warmup else None

    def _maak(nr, doel):
        r = FrameResult(frame_nr=nr, time=nr / fps if fps > 0 else 0)
        if doel is not None:
            r.lm = doel
            r.pose_found = True
        return r

    def _leeg_buffer():
        """Picks the target skater from the buffered frames and then still plays those
        frames through the tracker, so not a single frame is lost."""
        keuze = _choose_moving_target(buffer)
        start = 0
        if keuze is not None:
            start, tracker.doel_punt = keuze
        for i, poses in enumerate(buffer):
            # Before the start frame of the chosen track the target isn't in frame yet;
            # the tracker would lock onto a bystander there, so those frames stay empty.
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

        if buffer:                       # video shorter than the warmup window
            yield from _leeg_buffer()

    cap.release()


# ── Offline landmark smoothing ──────────────────────────────────────────────────
def _savgol_coeffs(window, poly):
    """SG coefficients for the smooth value at the center of the window."""
    half = window // 2
    k = np.arange(-half, half + 1)
    A = np.vander(k, poly + 1, increasing=True)   # columns k^0 .. k^poly
    return np.linalg.pinv(A)[0]                    # row for polynomial coefficient 0


def _savgol(y, window, poly):
    """Savitzky-Golay smoothing (numpy) with polynomial edge handling."""
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

    # Edges: local polynomial fit over the first/last window.
    xw = np.arange(window)
    pL = np.polyfit(xw, y[:window], poly)
    out[:half] = np.polyval(pL, xw[:half])
    pR = np.polyfit(xw, y[-window:], poly)
    out[n - half:] = np.polyval(pR, xw[window - half:])
    return out


def _hampel_outliers(y, window=HAMPEL_WINDOW, k=HAMPEL_K):
    """
    Boolean mask of outliers: points more than k robust standard deviations (via
    median/MAD in a local window) from the local median. Catches occlusion jumps that
    briefly put a joint somewhere else.
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


def _gap_distance(betrouwbaar):
    """Per frame: how many frames it sits from the nearest reliable frame (0 where it's
    reliable itself). A measure of how deep you are in a detection gap."""
    idx = np.flatnonzero(np.asarray(betrouwbaar, dtype=bool))
    n = len(betrouwbaar)
    if len(idx) == 0:
        return np.full(n, float(n))
    alle = np.arange(n)
    dichtstbij = np.searchsorted(idx, alle).clip(0, len(idx) - 1)
    vorige = (dichtstbij - 1).clip(0, len(idx) - 1)
    return np.minimum(np.abs(alle - idx[dichtstbij]), np.abs(alle - idx[vorige])).astype(float)


def _running_max(y, window):
    """Centered running maximum: the envelope of a rippling signal."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = window // 2
    return np.array([y[max(0, i - half):min(n, i + half + 1)].max() for i in range(n)])


def _interpolate_unreliable(y, betrouwbaar):
    """Replaces unreliable positions with linear interpolation from the rest."""
    y = np.asarray(y, dtype=float).copy()
    idx = np.flatnonzero(betrouwbaar)
    if len(idx) == 0 or len(idx) == len(y):
        return y                      # nothing (or everything) reliable: leave as is
    missing = np.flatnonzero(~betrouwbaar)
    y[missing] = np.interp(missing, idx, y[idx])   # np.interp clamps at the edges
    return y


def _limit_interpolation(betrouwbaar, max_run):
    """
    Lets long unreliable stretches and stretches at a segment edge count as reliable
    after all (keeps the raw detection). Interpolation repairs short hiccups
    (occlusion jump, one bad frame) fine, but over a long stretch the straight-line
    guess is worse than the detection itself, and at an edge np.interp can only clamp —
    then the skeleton freezes next to the skater.
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
    Repairs left/right swaps of the leg joints within one segment. Works in-place on
    the segment arrays X/Y/V (Tx33). The two leg trajectories are followed for
    continuity: if at frame t the swapped assignment clearly fits the previous
    positions better (cost < LR_SWAP_FACTOR x unswapped), L and R are flipped there.

    The decision applies to the **whole leg at once** — knee and ankle, with heel and
    toe along for the ride — on the combined cost of both joint pairs. Were knee and
    ankle to decide independently, one might swap and the other not: then the left knee
    ends up hanging off the right ankle, an anatomically impossible skeleton with a
    nonsensical tibia length (which then trips the bone-length check).

    Because the little chain is anchored on the first frame, a wrong first frame can
    label the whole sequence inverted; hence a majority vote afterward against the raw
    detector labels — the detector is usually right, we only repair the minority
    stretches. That vote runs exclusively over the frames where we actually made a
    decision (see `besloten`).
    """
    paren   = ((L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE))
    volgers = ((L_HEEL, R_HEEL), (L_TOE, R_TOE))     # attached to the ankle
    T = len(X)
    gewisseld = np.zeros(T, dtype=bool)
    besloten  = np.zeros(T, dtype=bool)   # frames where the continuity cost gave a verdict
    vorige = {}                           # pair → (last left position, last right position)
    for t in range(T):
        huidig = {(l, r): (np.array([X[t, l] * w, Y[t, l] * h]),
                           np.array([X[t, r] * w, Y[t, r] * h]))
                  for l, r in paren
                  if V[t, l] >= VIS_MIN and V[t, r] >= VIS_MIN}
        if not huidig:
            continue                      # unreliable frame: don't decide
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
    # Majority vote only over the decided frames. A skipped frame is False because
    # there's NO verdict, not because "no swap" was the call; were the inversion to
    # include those frames, the least reliable frames would get an L/R swap with no
    # basis at all, against their neighbors.
    if gewisseld[besloten].mean() > 0.5:  # chain anchored wrong: flip the labels
        gewisseld[besloten] = ~gewisseld[besloten]
    for t in np.flatnonzero(gewisseld):
        for a, b in paren + volgers:
            for A in (X, Y, V):
                A[t, a], A[t, b] = A[t, b], A[t, a]


def _bone_length_outliers(X, Y, V, w, h):
    """
    Boolean mask (Tx33): joints whose adjacent bone (femur/tibia) has a length outlier
    in that frame. Bones are rigid; their image length only changes slowly with
    distance to the camera. A jump means (at least) one endpoint was detected wrong —
    which one, we don't know, so both endpoints count as unreliable there.
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
        fout = _hampel_outliers(lengte)
        for t in idx[fout]:
            mask[t, a] = mask[t, b] = True
    return mask


def smooth_landmarks_offline(resultaten, w, h, window_s=SMOOTH_WINDOW_S,
                             poly=SMOOTH_POLY, fps=30.0):
    """
    Cleans up the landmark trajectories and smooths them bidirectionally (zero-lag),
    per contiguous segment of frames-with-pose (so nothing gets interpolated across
    detection gaps). Per joint, unreliable frames (low visibility or occlusion outlier)
    are first interpolated away, then Savitzky-Golay is applied. Overwrites
    resultaat.lm with the cleaned, smooth landmarks.
    """
    window = max(poly + 2, int(round(window_s * fps)))
    if window % 2 == 0:
        window += 1
    max_run = max(1, int(round(INTERP_MAX_S * fps)))

    # Determine contiguous segments of frames with a pose.
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
        # Even very short segments (fragmented detection) go through the mill. The
        # filters degrade gracefully there: Savitzky-Golay returns a window < poly+2
        # unchanged, `_limit_interpolation` leaves an edge stretch as is, and the
        # bone-length check skips a too-short stretch — but the L/R fix still does its
        # job. Skipping them would leave raw, unchecked data in place without that
        # showing up anywhere.
        X = np.array([[resultaten[i].lm[j].x for j in range(n_lm)] for i in seg])
        Y = np.array([[resultaten[i].lm[j].y for j in range(n_lm)] for i in seg])
        V = np.array([[resultaten[i].lm[j].visibility for j in range(n_lm)] for i in seg])
        # First repair left/right swaps: to the per-joint filters those look like
        # (double) jumps, but they're exactly repairable.
        _fix_lr_swaps(X, Y, V, w, h)
        # Unreliable = poorly visible OR a position outlier (occlusion).
        Bet = np.empty((len(seg), n_lm), dtype=bool)
        for j in range(n_lm):
            b = (V[:, j] >= VIS_MIN)
            b &= ~_hampel_outliers(X[:, j])
            b &= ~_hampel_outliers(Y[:, j])
            Bet[:, j] = b
        # Bone-length check: a femur/tibia length jump flags both endpoints.
        Bet &= ~_bone_length_outliers(X, Y, V, w, h)
        # Blur-frame safety net: if most of the data-bearing joints in a frame are
        # unreliable at the same time, that's a correlated confidence dip (motion
        # blur), not an occlusion of one joint. The raw detection does sit on the
        # skater there — trust the whole frame instead of interpolating the complete
        # skeleton away.
        meet = V.max(axis=0) >= VIS_MIN       # joints that carry any data at all
        if meet.any():
            blur = (V[:, meet] >= VIS_MIN).mean(axis=1) < BLUR_FRAME_FRAC
            Bet[blur, :] = True
        for j in range(n_lm):
            betrouwbaar = _limit_interpolation(Bet[:, j], max_run)
            xs = _interpolate_unreliable(X[:, j], betrouwbaar)
            ys = _interpolate_unreliable(Y[:, j], betrouwbaar)
            X[:, j] = _savgol(xs, window, poly)
            Y[:, j] = _savgol(ys, window, poly)
        for t, i in enumerate(seg):
            oud = resultaten[i].lm
            # Visibility from V (not from oud): on an L/R swap it was swapped along too.
            resultaten[i].lm = [
                Landmark(float(X[t, j]), float(Y[t, j]),
                         getattr(oud[j], 'z', 0.0), float(V[t, j]))
                for j in range(n_lm)
            ]


def determine_horizon_sequence(input_pad, n_frames, fps, force_fps=None, progress_callback=None,
                         deinterlacen=False):
    """
    Detects the ice-line tilt **per frame** (for a wobbling camera) and turns it into a
    stable signal: every frame gets `detect_ice_line()`, then gaps (no line found) and
    outliers (a wrongly detected line, Hampel) are interpolated away and the whole thing
    is smoothed over time (Savitzky-Golay, `HORIZON_SMOOTH_S`). This way the horizon
    follows the slow wobble without the per-frame Hough jitter. Returns a list of
    length `n_frames` in degrees.

    Runs as a separate video pass (backend-independent); the decode cost disappears
    against the pose detection. With no reliable line at all: everything 0.0.
    """
    cap = open_video(input_pad, deinterlacen)
    ruw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ruw.append(detect_ice_line(frame))     # float or None
        if progress_callback is not None:
            progress_callback(len(ruw), n_frames)
    cap.release()

    # Match the length to the results list.
    if len(ruw) < n_frames:
        ruw += [None] * (n_frames - len(ruw))
    else:
        ruw = ruw[:n_frames]

    betrouwbaar = np.array([v is not None for v in ruw])
    if not betrouwbaar.any():
        return [0.0] * n_frames

    y = np.array([v if v is not None else np.nan for v in ruw], dtype=float)
    y = _interpolate_unreliable(y, betrouwbaar)          # fill detection gaps
    betr2 = betrouwbaar & ~_hampel_outliers(y)              # throw out wrong lines
    y = _interpolate_unreliable(y, betr2)
    window = max(SMOOTH_POLY + 2, int(round(HORIZON_SMOOTH_S * fps)))
    y = _savgol(y, window, SMOOTH_POLY)
    return [round(float(v), 2) for v in y]


def box_sequence(resultaten, fps):
    """
    The box the skater needs per frame: `(center_x, center_y, radius)`, all normalized.
    The GUI derives both the automatic zoom and the follow point from this — the crop
    is symmetric in normalized coordinates (0.5/zoom in both x and y), so one radius
    around one center point is exactly what needs to fit in frame.

    The center is the midpoint of ALL visible landmarks, NOT `torso_centroid`: the
    latter sits high in the body (shoulders + hips), so a box around it would give as
    much room above the head as below the skates — wasting a quarter of the frame. The
    radius is the distance from that center to the farthest point, so to a leg fully
    extended at the push, not just body height.

    Cleaned up as in `determine_horizon_sequence`: gaps (no pose) and outliers (a limb that
    briefly drops out or jumps) interpolated away, then smoothed. For the radius, first
    a **running maximum** over `BOX_SMOOTH_S` and only then Savitzky-Golay; that order
    is essential, because smoothing alone flattens the peak and then a fully extended
    leg falls just outside the frame. The maximum over roughly one stroke is an
    envelope that always covers the widest stance of that moment yet still moves
    slowly, so the box doesn't pump with every stroke. The center is smoothed over a
    shorter window (`BOX_CENTER_S`) — that does need to follow the skater — and the
    radius is measured against it after that smoothing, so the two match up.

    Returns a list of tuples the same length as `resultaten`, or None if there's too
    little pose to say anything sensible (the caller then falls back to a fixed zoom).
    """
    punten_per_frame = [
        _visible_xy(r.lm) if (r.pose_gevonden and r.lm is not None) else []
        for r in resultaten
    ]
    betrouwbaar = np.array([len(p) > 0 for p in punten_per_frame])
    if betrouwbaar.sum() < BOX_MIN_FRAMES:
        return None

    def _opschonen(waarden):
        """Gaps + outliers out — the recipe from determine_horizon_sequence, without smoothing."""
        y = np.array([v if v is not None else np.nan for v in waarden], dtype=float)
        y = _interpolate_unreliable(y, betrouwbaar)
        betr2 = betrouwbaar & ~_hampel_outliers(y)
        return _interpolate_unreliable(y, betr2), betr2

    midden_window = max(BOX_POLY + 2, int(round(BOX_CENTER_S * fps)))
    straal_window = max(BOX_POLY + 2, int(round(BOX_SMOOTH_S * fps)))
    cx, _ = _opschonen([(min(x for x, _ in p) + max(x for x, _ in p)) / 2 if p else None
                        for p in punten_per_frame])
    cy, _ = _opschonen([(min(y for _, y in p) + max(y for _, y in p)) / 2 if p else None
                        for p in punten_per_frame])
    cx = _savgol(cx, midden_window, BOX_POLY)
    cy = _savgol(cy, midden_window, BOX_POLY)

    # Radius relative to the smoothed center (not the raw one), otherwise they wouldn't line up.
    ruw_straal = [max((max(abs(x - cx[i]), abs(y - cy[i])) for x, y in p), default=None)
                  for i, p in enumerate(punten_per_frame)]
    straal, betr2 = _opschonen(ruw_straal)
    straal = _savgol(_running_max(straal, straal_window), straal_window, BOX_POLY)
    # The SG edge fit can overshoot to ≤ 0; that would cause a division by zero further on.
    ondergrens = max(1e-3, float(np.median(straal[betr2] if betr2.any() else straal)) * 0.05)
    straal = np.maximum(ondergrens, straal)

    # Long detection gaps: smoothly open the box up to the full frame (radius 0.5).
    # Outside a gap `f` is 0 and nothing changes. The blend happens in "zoom" space
    # (0.5/radius), not in the radius itself: close to the skater the radius is small,
    # and there a linear blend would collapse the zoom within a few frames. The center
    # stays where it was — at radius 0.5 the crop covers the whole frame anyway.
    f = np.clip(_gap_distance(betrouwbaar) / max(1.0, BOX_GAP_S * fps), 0.0, 1.0)
    straal = 0.5 / ((0.5 / straal) * (1 - f) + 1.0 * f)
    return [(float(cx[i]), float(cy[i]), float(straal[i])) for i in range(len(resultaten))]


def determine_corner_sequence(resultaten, w, h, fps):
    """
    Flags per frame whether the skater is riding a corner (`FrameResult.corner`), so
    those frames no longer yield a push measurement. Recipe from `determine_horizon_sequence`
    / `box_sequence`: raw signal → outliers out → smooth → decide.

    1. `corner_ratio` per frame with a pose (hip width / torso length).
    2. Hampel outliers out (a frame where a hip briefly jumps) and gaps bridged
       linearly, then Savitzky-Golay over `CORNER_SMOOTH_S` — the ratio ripples
       slightly with the stroke and shouldn't be able to flip every frame.
    3. Hysteresis: below `CORNER_IN` into the corner, only above `CORNER_OUT` back out
       (the same pattern as the stance assignment in `assign_push_leg_cyclic`).
    4. Corner stretches shorter than `CORNER_MIN_S` are noise and are cleared.

    Frames **without** a pose can't be measured: they inherit the running state, and a
    `corner` flag already set stays set (on the YOLO backend these are the frames the
    detection pass skipped — nothing to measure there, and that stays so). Frames
    **with** a pose are judged on their own ratio and can therefore also **clear** an
    already-set flag: exactly what needs to happen when the detection pass wrongly
    skipped a stretch but the check frames within it show a neatly frontal skater.
    """
    n = len(resultaten)
    if n == 0:
        return

    def _ratio(r):
        if not (r.pose_gevonden and r.lm is not None):
            return np.nan
        v = corner_ratio(r.lm, w, h)
        return np.nan if v is None else v

    ruw = np.array([_ratio(r) for r in resultaten], dtype=float)
    meetbaar = ~np.isnan(ruw)
    if not meetbaar.any():
        return                       # no verdict possible at all — leave the flags as is
    y = _interpolate_unreliable(ruw, meetbaar)
    betr = meetbaar & ~_hampel_outliers(y)
    if betr.any():
        y = _interpolate_unreliable(y, betr)
    win = max(SMOOTH_POLY + 2, int(round(CORNER_SMOOTH_S * fps)))
    if win % 2 == 0:
        win += 1
    y = _savgol(y, win, SMOOTH_POLY) if n >= win else y

    # Hysteresis. The starting state comes from the first measurable frame, so a clip
    # that begins IN the corner is correct right away (instead of only after the first
    # time it drops below the threshold).
    eerste = int(np.argmax(meetbaar))
    staat = bool(y[eerste] < CORNER_IN)
    vlag = np.zeros(n, dtype=bool)
    for i in range(n):
        if meetbaar[i]:
            if y[i] < CORNER_IN:
                staat = True
            elif y[i] > CORNER_OUT:
                staat = False
            vlag[i] = staat
        else:
            vlag[i] = staat or resultaten[i].bocht

    # Corners that are too short are noise. Conversely the same applies: a few "straight"
    # frames in the middle of a corner isn't a straight section either, and could
    # produce a bogus measurement.
    min_len = max(1, int(round(CORNER_MIN_S * fps)))
    _clear_short_runs(vlag, min_len)

    for r, b in zip(resultaten, vlag):
        r.bocht = bool(b)


def _clear_short_runs(vlag, min_len):
    """Sets contiguous runs shorter than `min_len` to their neighbors' value (in
    place). Runs at the edge only count if they border their one neighboring run."""
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


def make_prefill(resultaten, idx, fps, n_lm=33):
    """
    A starting skeleton for frame `idx`, which itself has no pose: the GUI puts this
    down when the user is about to manually place a skeleton, so they only need to
    correct existing points instead of pointing out all eight again.

    If the frame sits between two pose frames and the gap is short (≤ `INTERP_MAX_S` —
    the same limit `_limit_interpolation` uses in the smoothing), it's linearly
    interpolated between those two. Over a longer gap, blending two poses is
    anatomical nonsense (the limbs melt into each other), and the nearest pose, however
    stale, is a fairer starting point. If there's no pose at all in the whole analysis,
    everything sits in the center of the frame with visibility 0 — invisible, so
    nothing shows on screen that isn't there.

    Always returns a list of exactly `n_lm` `Landmark`s (never None): the serialization
    writes into a fixed (n, 33, 3) array.
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


class TrimAborted(Exception):
    """The user stopped the trimming (see trim_fragments/stop_check)."""


def _safe_filename(naam):
    """Turns a fragment title into a filename Windows will accept."""
    schoon = "".join(c if c.isalnum() or c in " -_." else "_" for c in (naam or "").strip())
    return schoon.strip(" .")[:80]


def trim_fragments(bron_pad, fragmenten, doelmap, progress_callback=None,
                    stop_check=None, fps=None, deinterlacen=False):
    """
    Writes the marked stretches of a long recording out as separate video files
    (ROADMAP phase 8) and returns the paths, in the same order as `fragmenten`.

    `fragmenten` is a list of `(start_frame, eind_frame, naam)` — both bounds
    **inclusive**, `naam` becomes the filename (without extension). **The cut is
    exactly on the marked frames**: no margin added or subtracted. The trainer watches
    the footage while marking and decides the bounds themselves; the program shouldn't
    silently add seconds. (Padding at the front would even make target selection
    harder: `DoelKiezer` gets frame 0 of the clip, and that's now precisely the frame
    "start" was pressed on.)

    **One sequential pass**: every frame goes to the writer of every fragment it falls
    in, so the video is decoded exactly once and there's never any seeking — same
    motive as the read loop in `schaats_yolo._detecteer_alles`, and overlapping
    fragments are therefore simply fine (the frame then goes to two writers).

    Codec `mp4v`: bundled with the opencv-python wheel, whereas `avc1` is often missing
    on Windows. So this re-encodes; for pose detection that quality loss is negligible.
    A stream copy (ffmpeg `-c copy`) would avoid that but can only start on a keyframe —
    exactly the silent margin that isn't wanted here, and it would make frame 0 of the
    clip a different image than the one "start" was pressed on.

    `progress_callback(frame_nr, totaal)` and `stop_check() -> bool` (abort; the files
    already written are then cleaned up).
    """
    cap = open_video(bron_pad, deinterlacen)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {bron_pad}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(doelmap, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    taken, gebruikt = [], set()
    for i, (start, eind, naam) in enumerate(fragmenten):
        stam = _safe_filename(naam) or f"fragment_{i + 1}"
        if stam.lower() in gebruikt:                # two fragments with the same title
            stam = f"{stam}_{i + 1}"
        gebruikt.add(stam.lower())
        taken.append({"start": int(start), "eind": int(eind), "writer": None,
                      "pad": os.path.join(doelmap, f"{stam}.mp4")})

    # Past the last end frame nothing needs to be decoded any more — for a fragment at
    # the start of a half-hour recording that saves almost everything.
    laatste = max((t["eind"] for t in taken), default=-1)
    geschreven = set()
    try:
        idx = 0
        while idx <= laatste:
            if stop_check is not None and stop_check():
                raise TrimAborted()
            # grab/retrieve instead of read(): frames that fall in no fragment at all
            # only need to be advanced past, not decoded. Measured on a 1080p
            # recording, that saves ~9 → ~3 ms per frame, and precisely for a fragment
            # far into a 23-minute recording that's the bulk of the work.
            if not cap.grab():
                break                               # video shorter than CAP_PROP_FRAME_COUNT reported
            actief = [t for t in taken if t["start"] <= idx <= t["eind"]]
            if actief:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                for t in actief:
                    if t["writer"] is None:
                        t["writer"] = cv2.VideoWriter(t["pad"], fourcc, fps, (w, h))
                        if not t["writer"].isOpened():
                            raise IOError(f"Can't write fragment: {t['pad']}")
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
        # A fragment that lay entirely past the end of the video: report it instead of
        # sending a nonexistent path into the batch flow.
        raise IOError(f"{len(ontbreekt)} fragment(s) fell outside the video "
                      f"({os.path.basename(bron_pad)}) and could not be trimmed.")
    return [t["pad"] for t in taken]


def assign_push_leg_cyclic(resultaten, h, fps):
    """
    Assigns the push leg (stance leg) per pose frame based on the skating cycle instead
    of independently per frame. Signal = ankle height difference (left minus right;
    higher y = lower in frame = closer to the ice), so positive → left ankle on the ice
    → left is the stance leg. That signal is smoothed and turned into a leg with
    **hysteresis**, so it only switches on a real weight transfer.

    Works per contiguous segment of pose frames (not across detection gaps); the
    hysteresis state is kept within a segment.
    """
    # Contiguous segments of frames with lm_data.
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
        sig = np.array([(resultaten[i].lm_data['l_ankle'][1]
                         - resultaten[i].lm_data['r_ankle'][1]) / h for i in seg])
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
    """Float pixel coordinates of landmark `idx` — no int truncation like in
    `get_landmarks`: at a distance the lower leg is only tens of pixels, and a whole-
    pixel rounding already costs a few degrees in the 3D reconstruction."""
    return (r.lm[idx].x * w, r.lm[idx].y * h)


def _extension_ratio(r, been, w, h):
    """Leg extension as a scale-free ratio: straight line hip→ankle divided by the sum
    of the bone lengths (hip→knee + knee→ankle). ~1.0 = fully extended (knee on the
    line), lower = bent. Scale-free (near or far doesn't matter) and without a fixed
    pixel threshold — exactly what's needed to find the extension *maximum* (= end of
    push) as a peak instead of with a hard angle limit. On float pixels: at a distance
    the lower leg is only tens of pixels."""
    h_idx, k_idx, e_idx = (L_HIP, L_KNEE, L_ANKLE) if been == 'links' else (R_HIP, R_KNEE, R_ANKLE)
    heup  = np.array(_lm_px(r, h_idx, w, h))
    knie  = np.array(_lm_px(r, k_idx, w, h))
    enkel = np.array(_lm_px(r, e_idx, w, h))
    bot = float(np.hypot(*(knie - heup)) + np.hypot(*(enkel - knie)))
    return float(np.hypot(*(enkel - heup)) / bot) if bot > 1e-6 else 0.0


def determine_push_from_extension(resultaten, w, h, fps):
    """
    Determines push completion from **leg extension** instead of the per-frame 2-of-3
    vote (`detect_weight_on_leg`). The stance leg is "extended" for a whole phase:
    from standing up after placement (leg ≈ vertical → lower-leg angle ~80-90°) to the
    full sideways push (leg flat → ~50°). We don't take the extension *maximum* as push
    completion (that falls on the standing-up → a misleadingly high angle), but within
    that extended phase the frame with the **flattest (lowest) lower-leg angle** = the
    actual, most horizontal push. That follows the observation "leg fully extended =
    push done" and reads the angle at the meaningful moment; the standing-up (high
    angle) falls away automatically and the recovery already sits outside the extension
    plateau.

    Works per contiguous **stance run** (frames where `r.been` stays the same, with
    pose + lm_data — the cyclic assignment yields exactly one push per run). Per run:
    the extended plateau = the contiguous window around `argmax(strek_ratio)` where the
    ratio stays within `EXTENSION_PLATEAU_BAND` under its maximum; push completion `c` =
    the plateau frame with the smallest lower-leg angle. Sets `r.strek_ratio` (smoothed,
    for HUD/debug) and `r.gewicht_erop` (True through `c`, False after → the event spans
    the whole load+push and ends on the flattest-angle frame) per frame.

    "Predictable stroke timing" is in there as a **soft prior**: from the median run
    length we estimate the half-stroke period; a run shorter than a fraction
    (EXTENSION_MIN_STROKE_FRAC) of that is almost certainly a noise flip of the leg
    assignment and yields NO push (prevents phantom pushes from brief L/R flips). NOTE:
    with an accelerating start the strokes are shorter — the fraction is therefore kept
    low so only truly short (noise) runs get dropped, not fast opening strokes. The
    median runs only over runs of at least `EXTENSION_MIN_RUN_S`: were the noise runs to
    take part in it, the estimate — and thus the threshold — would drop exactly when
    you need it, with lots of L/R flips.

    Two kinds of runs DO yield an event — you want to see that something happened —
    but flagged with `afzet_onvolledig`, so the GUI and the library statistics keep
    them out of avg/min/max:

    - **`INCOMPLETE_TRUNCATED`** — the run doesn't end on a leg switch but on the end of
      the video or of the pose segment. The push wasn't finished there, the plateau
      only contains the standing-up, and the "flattest angle" is systematically far too
      steep (measured: +15 to +25° on the final push). A run truncated at the *start*
      of a segment counts fine: only the load phase is missing there, while the push
      completion (where the angle comes from) is in frame.
    - **`INCOMPLETE_NO_PUSH`** — the run is long enough (`min_run`) and ends neatly on a
      leg switch, but the extension plateau covers only the standing-up phase; the
      flattest angle within that plateau is then still the standing-up. Recognizable by
      the geometry: the lower leg stands less than `EXTENSION_MIN_SLOPE_DEG` out of
      vertical at "completion", so there was no sideways push. Occurred in 3 of the 100
      events in the library.

    Requires the global cyclic leg assignment. Returns the estimated stroke period
    (frames), purely informational.
    """
    # Contiguous runs of the same stance leg (within frames with pose + lm_data).
    # `afgekapt` = the run doesn't end on a leg switch but on a detection gap or the end
    # of the video; the push wasn't fully observed then.
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
        huidig['afgekapt'] = True          # video ended → last push not completed
        runs.append(huidig)

    # Robustly estimate the half-stroke period: only runs long enough to even be a half
    # stroke at all (see EXTENSION_MIN_RUN_S). Without such runs it falls back to all
    # runs — then there's nothing better.
    min_abs = max(2, int(round(EXTENSION_MIN_RUN_S * fps))) if fps > 0 else 2
    lengtes = [len(run['idx']) for run in runs]
    echte = [n for n in lengtes if n >= min_abs] or lengtes
    half_periode = float(np.median(echte)) if echte else 0.0
    min_run = max(min_abs, int(round(EXTENSION_MIN_STROKE_FRAC * half_periode))) if half_periode else min_abs

    win = max(SMOOTH_POLY + 2, int(round(EXTENSION_SMOOTH_S * fps)))
    if win % 2 == 0:
        win += 1

    for run in runs:
        idx, been = run['idx'], run['been']
        ratio = np.array([_extension_ratio(resultaten[i], been, w, h) for i in idx])
        ratio_s = _savgol(ratio, win, SMOOTH_POLY) if len(idx) >= 3 else ratio
        for t, i in enumerate(idx):
            resultaten[i].strek_ratio = round(float(ratio_s[t]), 3)
            resultaten[i].afzet_onvolledig = INCOMPLETE_TRUNCATED if run['afgekapt'] else None
        if len(idx) < min_run:                    # too short → noise flip, no push
            for i in idx:
                resultaten[i].gewicht_erop = False
            continue

        # Extended phase = the contiguous plateau around the extension peak (ratio
        # within the band under its maximum). Recovery (knee bends → ratio drops) falls
        # outside it.
        piek = int(np.argmax(ratio_s))
        drempel = ratio_s[piek] - EXTENSION_PLATEAU_BAND
        lo = hi = piek
        while lo - 1 >= 0 and ratio_s[lo - 1] >= drempel:
            lo -= 1
        while hi + 1 < len(idx) and ratio_s[hi + 1] >= drempel:
            hi += 1

        # Within the plateau, the flattest (lowest) lower-leg angle = push completion.
        # The standing-up (high angle) falls away automatically that way.
        e_idx, k_idx = (L_ANKLE, L_KNEE) if been == 'links' else (R_ANKLE, R_KNEE)
        hoeken = [calculate_angle_to_ice(_lm_px(resultaten[idx[t]], e_idx, w, h),
                                       _lm_px(resultaten[idx[t]], k_idx, w, h),
                                       resultaten[idx[t]].horizon_deg)
                  for t in range(lo, hi + 1)]
        c = lo + int(np.argmin(hoeken))
        for t, i in enumerate(idx):
            resultaten[i].gewicht_erop = (t <= c)

        # Geometric final check: if the lower leg is still nearly upright at
        # "completion", the plateau only covered the standing-up moment and no sideways
        # push was observed. The event stays, but the angle isn't a measurement. A run
        # already truncated keeps its own (known) reason.
        if not run['afgekapt'] and (90.0 - min(hoeken)) < EXTENSION_MIN_SLOPE_DEG:
            for i in idx:
                resultaten[i].afzet_onvolledig = INCOMPLETE_NO_PUSH

    return half_periode * 2.0


def _world_trajectory(resultaten, perspectief, fps, w, h):
    """
    World positions (m) + direction of travel per frame from the calibrated ankle positions.

    Position = the ankle of the **stance leg** (it sits on the ice; the swing-leg ankle
    hangs tens of cm higher and would systematically distort the position), pinned via
    the line of sight onto the plane `ankle_height` above the ice. The direction is the
    displacement over a `TRAVEL_WINDOW_S` window; if that's too small to trust
    (standing still, a gap), it falls back to the track-line direction (world-y) with
    the sign of the net displacement — after all, the track lines ARE the direction of travel.
    """
    kal = perspectief.kalibratie
    n = len(resultaten)
    pos = np.full((n, 2), np.nan)
    for i, r in enumerate(resultaten):
        if r.lm_data is None:
            continue
        if r.been in ('links', 'rechts'):
            e_idx = L_ANKLE if r.been == 'links' else R_ANKLE
        else:                             # no cyclic assignment: lowest ankle in frame
            e_idx = L_ANKLE if r.lm[L_ANKLE].y >= r.lm[R_ANKLE].y else R_ANKLE
        P = skate_perspective.point_on_ice(kal, _lm_px(r, e_idx, w, h),
                                           height=perspectief.enkel_hoogte)
        if P is not None:
            pos[i] = P[:2]

    geldig = ~np.isnan(pos[:, 0])
    idx = np.where(geldig)[0]
    richting = np.zeros((n, 2))
    if len(idx) < 2:
        richting[:] = (0.0, 1.0)
        return pos, richting

    # Interpolate gaps closed for a continuous trajectory (direction only).
    vol = np.column_stack([np.interp(np.arange(n), idx, pos[idx, k]) for k in (0, 1)])
    netto = vol[idx[-1]] - vol[idx[0]]
    basis = np.array([0.0, 1.0 if netto[1] >= 0 else -1.0])
    k = max(1, int(round(TRAVEL_WINDOW_S * fps / 2)))
    for i in range(n):
        d = vol[min(i + k, n - 1)] - vol[max(i - k, 0)]
        lengte = float(np.hypot(d[0], d[1]))
        richting[i] = d / lengte if lengte >= TRAVEL_MIN_M else basis
    return pos, richting


def _perspective_angle(r, enkel_px, knie_px, perspectief, richting):
    """
    Corrected push angle for one frame via the 3D reconstruction. Also fills in
    `hoek_correctie` (relative to the old measurement: image-plane angle minus horizon
    subtraction) and `hoek_betrouwbaar` on `r`. Falls back to the old image-plane
    measurement on a failed reconstruction (ankle above the horizon — can only happen
    with a derailed detection).
    """
    # Transitional: PerspectiveConfig.methode/perspectief.methode still hold the
    # original Dutch method names here until schaats_gui.py is translated in its own
    # phase (see the translate-to-english plan). Normalize once, locally, so the
    # comparison below and the call into skate_perspective agree — the same shim
    # `reconstruct_angle` itself falls back on.
    method = {"onderbeen": "lower_leg", "beenvlak": "leg_plane"}.get(
        perspectief.methode, perspectief.methode)
    rec = skate_perspective.reconstruct_angle(
        perspectief.kalibratie, enkel_px, knie_px,
        method=method, lower_leg_l=perspectief.onderbeen_l,
        plane_direction=richting if method == 'leg_plane' else None,
        travel_direction=richting, ankle_height=perspectief.enkel_hoogte)
    oude_hoek = calculate_angle_to_ice(enkel_px, knie_px, r.horizon_deg)
    if rec is None:
        r.hoek_correctie = None
        r.hoek_betrouwbaar = False
        return oude_hoek
    hoek = round(float(rec.angle), 1)     # a plain float: goes straight into the DB/CSV/JSON
    r.hoek_correctie = round(hoek - oude_hoek, 1)
    r.hoek_betrouwbaar = rec.reliable
    return hoek


def _set_smooth_angle(resultaten, smooth_n):
    """
    Fills `r.smooth_hoek`: a **centered** (zero-lag) average of `r.hoek` over
    `smooth_n` frames, per contiguous run of the same stance leg.

    Deliberately no trailing deque any more: such a trailing average lags half a
    window behind, and because the push angle drops toward its minimum, the reported
    angle was therefore systematically too steep. A detection gap or a leg switch
    breaks the run, so it's never averaged across a gap or between two legs. Purely a
    display quantity (HUD/graph); the table reports the angle of a single frame (see
    segment_pushes).
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


def process_derivatives(resultaten, w, h, fps, smooth_n=5, threshold=0.015, cyclus=True,
                       perspectief=None, afzet_strek=True):
    """
    Fills in the derived quantities per frame (push leg, angle, knee angle, weight),
    computed from resultaat.lm. Called after the offline smoothing so everything is
    based on the smooth landmarks.

    The push leg is assigned cycle-aware (`assign_push_leg_cyclic`) if `cyclus`,
    otherwise per frame (`determine_push_leg`). Push completion comes, with `afzet_strek`
    (default), from the leg extension (`determine_push_from_extension`: extension maximum =
    end of push); without it (or without `cyclus`) from the per-frame weight vote
    (`detect_weight_on_leg`). The rest is sequential because of the time
    histories; on a detection gap those are cleared. The push angle is corrected with
    the per-frame `r.horizon_deg` (set by `analyze`: constant or per-frame with auto-horizon).

    With `perspectief` (PerspectiveConfig) the 3D reconstruction replaces the
    image-plane angle: the real angle relative to the ice plane goes into `r.hoek`, the
    correction applied into `r.hoek_correctie`, and the quality flag into
    `r.hoek_betrouwbaar`. Bycatch: `r.wereld_xy` and (with a known scale) `r.snelheid`.
    The horizon subtraction then no longer applies to the ANGLE (the calibration knows
    the camera pose exactly); `r.horizon_deg` remains the true-horizon tilt for the overlay.
    """
    # Pass 1: pixel coordinates for all pose frames. The incomplete flag belongs to the
    # leg runs of THIS pass (they can shift after a skeleton edit), so clean first.
    #
    # A corner frame deliberately gets no `lm_data`: that's the entire exclusion right
    # there. The leg assignment, push completion, and event segmentation all build
    # their segments on "pose AND lm_data", so they automatically see the corner as a
    # detection gap — without a single change to the measurement logic. The skeleton
    # still gets drawn (that reads `r.lm`), so you can see what happened in the display.
    for r in resultaten:
        bruikbaar = r.pose_gevonden and r.lm is not None and not r.bocht
        r.lm_data = get_landmarks(r.lm, w, h) if bruikbaar else None
        r.afzet_onvolledig = None

    # Leg assignment (global, cycle-aware) before the per-frame derivatives.
    if cyclus:
        assign_push_leg_cyclic(resultaten, h, fps)

    # Push completion from the leg extension (extension maximum = end of push).
    # Requires the global cyclic leg assignment; without cyclus (diagnostic mode) it
    # falls back to the per-frame weight vote below.
    gebruik_strek = afzet_strek and cyclus
    if gebruik_strek:
        determine_push_from_extension(resultaten, w, h, fps)

    # World trajectory + speed (only with a calibration).
    if perspectief is not None:
        posities, richtingen = _world_trajectory(resultaten, perspectief, fps, w, h)
        k = max(1, int(round(TRAVEL_WINDOW_S * fps / 2)))
        for i, r in enumerate(resultaten):
            if np.isnan(posities[i, 0]):
                continue
            r.wereld_xy = (float(posities[i, 0]), float(posities[i, 1]))
            if perspectief.kalibratie.scale_known:
                j0, j1 = max(i - k, 0), min(i + k, len(resultaten) - 1)
                d = posities[j1] - posities[j0]
                if not np.isnan(d[0]) and j1 > j0:
                    r.snelheid = round(float(np.hypot(d[0], d[1])) * fps / (j1 - j0), 2)

    # Pass 2: angle, knee angle and weight per frame.
    enkel_hist  = {'links': deque(maxlen=10), 'rechts': deque(maxlen=10)}
    heup_hist   = deque(maxlen=10)

    for i, r in enumerate(resultaten):
        if not (r.pose_gevonden and r.lm_data is not None):
            heup_hist.clear()
            enkel_hist['links'].clear(); enkel_hist['rechts'].clear()
            continue

        lm_data = r.lm_data
        enkel_hist['links'].append(lm_data['l_ankle'])
        enkel_hist['rechts'].append(lm_data['r_ankle'])
        heup_hist.append((lm_data['l_hip'][0], lm_data['r_hip'][0]))

        been = r.been if cyclus else determine_push_leg(lm_data, heup_hist, w)
        if perspectief is not None:
            # Float pixels from r.lm (not the int-truncated lm_data): at a distance,
            # whole-pixel rounding already costs a few degrees in the reconstruction
            e_idx, k_idx = (L_ANKLE, L_KNEE) if been == 'links' else (R_ANKLE, R_KNEE)
            hoek = _perspective_angle(r, _lm_px(r, e_idx, w, h), _lm_px(r, k_idx, w, h),
                                     perspectief, richtingen[i])
        elif been == 'links':
            hoek = calculate_angle_to_ice(lm_data['l_ankle'], lm_data['l_knee'], r.horizon_deg)
        else:
            hoek = calculate_angle_to_ice(lm_data['r_ankle'], lm_data['r_knee'], r.horizon_deg)

        gewicht_stem, kniehoek, signalen = detect_weight_on_leg(
            been, lm_data, enkel_hist, heup_hist, w, h, threshold)

        r.been = been
        r.hoek = hoek
        r.kniehoek = kniehoek
        if not gebruik_strek:            # with extension, r.gewicht_erop is already set globally
            r.gewicht_erop = gewicht_stem
        r.signalen = signalen           # the 3 old signals remain as HUD diagnostics

    # Pass 3: the display average of the angle — centered, so without lag.
    _set_smooth_angle(resultaten, smooth_n)


def phase_progress(progress_callback, fase, n_fasen):
    """
    Wraps a `progress_callback(i, totaal)` so that several video passes together form
    one continuous bar: pass `fase` (0-based) of `n_fasen` maps onto its own slice
    [fase/n_fasen, (fase+1)/n_fasen]. None if there's no callback.
    """
    if progress_callback is None:
        return None
    return lambda i, totaal: progress_callback(fase * totaal + i, n_fasen * totaal)


def analyze(input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
              num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
              progress_callback=None, horizon_deg=0.0, auto_horizon=False,
              perspectief=None, waarschuwing_callback=None, bocht=True,
              deinterlacen=None, doel_kader=None):
    """
    Full analysis pipeline: multi-pose detection + target tracking (streaming), then
    offline landmark smoothing and computing the derived quantities. Returns
    (VideoInfo, list[FrameResult]).

    `doel_kader` (normalized (x0, y0, x1, y1) around the skater on the first frame)
    exists for signature compatibility with the YOLO backend, where it turns on the
    peephole (following a too-small skater from the box). This backend doesn't have
    that; its center point here only serves as `doel_punt` if that isn't given.

    With `bocht` (default) corner frames are flagged (`determine_corner_sequence`) and yield
    no push measurement. This backend detects per frame, streaming, and — unlike the
    YOLO backend — never skips frames; here it's therefore purely a measurement filter.

    `waarschuwing_callback(tekst)` exists for signature compatibility with the YOLO
    backend (which uses it to report silent fallbacks in target selection). This
    backend picks its target per frame, streaming, and has no message yet that would
    fit here.

    The horizon correction is either a fixed `horizon_deg` (manual line), or — with
    `auto_horizon` — detected **per frame** (`determine_horizon_sequence`, for a wobbling
    camera). Either way the value ends up per frame in `r.horizon_deg`. With auto
    there's a second video pass; it shares the progress bar with detection (half each)
    via `phase_progress`, so it stays one continuous bar.

    With `perspectief` (PerspectiveConfig) the track-line calibration replaces the
    horizon machinery entirely (see `set_horizon`/`process_derivatives`).
    """
    info = video_info(input_pad, force_fps)
    # None = figure it out ourselves (CLI convenience); the GUI determines it in the
    # dialog and passes an explicit bool, so the choice is visible and ends up in the settings.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    if perspectief is not None:
        auto_horizon = False     # a fixed camera is assumed; the calibration already knows the tilt
    if doel_punt is None and doel_kader is not None:
        doel_punt = ((doel_kader[0] + doel_kader[2]) / 2, (doel_kader[1] + doel_kader[3]) / 2)

    # Auto-horizon = two passes → one continuous bar (detection 0-50%, horizon 50-100%).
    det_cb = phase_progress(progress_callback, 0, 2) if auto_horizon else progress_callback
    hor_cb = phase_progress(progress_callback, 1, 2) if auto_horizon else progress_callback

    resultaten = list(analyze_frames(
        input_pad, model_pad, force_fps=force_fps, num_poses=num_poses,
        doel_punt=doel_punt, progress_callback=det_cb, deinterlacen=deinterlacen))

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)

    if bocht:
        determine_corner_sequence(resultaten, info.w, info.h, info.fps)

    set_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief, deinterlacen=deinterlacen)
    process_derivatives(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten


def set_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps=None,
                progress_callback=None, perspectief=None, deinterlacen=False):
    """
    Fills `r.horizon_deg` per frame: detected per frame with `auto_horizon`
    (`determine_horizon_sequence`), otherwise the constant `horizon_deg` everywhere. Shared by
    the MediaPipe and YOLO backends, so the per-frame horizon is backend-independent.

    With `perspectief`, the tilt comes from the calibration itself (the true horizon =
    the ice plane's vanishing line); auto-horizon then doesn't apply (fixed camera).
    The angle no longer uses `horizon_deg` at that point either (the reconstruction
    knows the camera pose exactly) — this then only drives the drawn ice line.
    """
    if perspectief is not None:
        hz = perspectief.kalibratie.horizon_deg
        for r in resultaten:
            r.horizon_deg = hz
        return
    if auto_horizon:
        horizons = determine_horizon_sequence(input_pad, len(resultaten), info.fps,
                                        force_fps=force_fps, progress_callback=progress_callback,
                                        deinterlacen=deinterlacen)
        for r, hz in zip(resultaten, horizons):
            r.horizon_deg = hz
    else:
        for r in resultaten:
            r.horizon_deg = horizon_deg


def _merge_events(a, b):
    """Merges two same-leg events into one push (a before b)."""
    snelheden = [v for v in (a.snelheid, b.snelheid) if v is not None]
    slagen = [s for s in (a.slaglengte, b.slaglengte) if s is not None]
    return PushEvent(
        index=a.index, leg=a.been,
        start_frame=a.start_frame, end_frame=b.eind_frame,
        start_time=a.start_tijd, end_time=b.eind_tijd,
        angle=b.hoek,                              # angle at push completion = the last one
        min_angle=min(a.min_hoek, b.min_hoek),
        max_angle=max(a.max_hoek, b.max_hoek),
        note="merged",
        # If either part is incomplete, that applies to the whole; the reason of the
        # LAST part weighs heaviest, since that's where the angle is read from.
        incomplete=b.onvolledig or a.onvolledig,
        correction=b.correctie, reliable=a.betrouwbaar and b.betrouwbaar,
        speed=round(float(np.mean(snelheden)), 2) if snelheden else None,
        stroke_length=round(float(np.sum(slagen)), 2) if slagen else None,
    )


def force_alternating(events, resultaten, merge_gap_s=0.35, min_tegen=2):
    """
    Feedback based on the skating rule "pushes always alternate left-right": two
    same-leg events back to back is impossible. Per such a pair we distinguish:

    - **Split push** — in the gap between them the other leg was NOT the push leg
      (purely a detection dip) AND the gap is small → the two events are **merged**.
    - **Missed counter-push** — the other leg WAS briefly the push leg in the gap
      (that short push got filtered out by `min_lengte`), or the gap is large → we
      **flag** the event with "missed counter-push?" instead of erasing it, so you see it.

    Returns a new, freshly re-indexed event list.
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
            uit[-1] = _merge_events(vorige, ev)          # split push → merge
        else:
            ev.opmerking = "missed counter-push?"         # flag, don't merge
            uit.append(ev)

    for i, ev in enumerate(uit):
        ev.index = i
    return uit


# ── Serialization (phase 0) ────────────────────────────────────────────────────────
# Save/reload an analysis without re-analyzing the video. Only the landmarks +
# pose_found + horizon are the source of truth; the derivatives (leg/angle/weight/
# events) get recomputed on reload with process_derivatives() + segment_pushes().
# Backend-independent: r.lm is either a MediaPipe landmark list or a Landmark list —
# both with .x/.y/.visibility.

def results_to_arrays(resultaten, info):
    """
    Serializes an analyzed FrameResult list to flat numpy arrays. z is deliberately
    left out (never used anywhere). Returns a dict suitable for np.savez_compressed;
    the video properties travel along as meta so reloading needs no video.
    """
    n = len(resultaten)
    landmarks     = np.zeros((n, 33, 3), dtype=np.float32)   # x, y, visibility
    pose_found    = np.zeros(n, dtype=bool)
    horizon       = np.zeros(n, dtype=np.float32)
    midline       = np.full((n, 2), np.nan, dtype=np.float32)  # dev l_knee, r_knee (px)
    corner        = np.zeros(n, dtype=bool)
    for i, r in enumerate(resultaten):
        pose_found[i]    = r.pose_gevonden
        horizon[i]       = r.horizon_deg
        corner[i]        = r.bocht
        if r.middellijn_dev:
            for k, naam in enumerate(('l_knee', 'r_knee')):
                if r.middellijn_dev.get(naam) is not None:
                    midline[i, k] = r.middellijn_dev[naam]
        if r.pose_gevonden and r.lm is not None:
            for j, p in enumerate(r.lm):
                landmarks[i, j, 0] = p.x
                landmarks[i, j, 1] = p.y
                landmarks[i, j, 2] = p.visibility
    return {
        'landmarks':     landmarks,
        # English key names, written going forward; arrays_to_results() reads
        # either these or the original Dutch names, so a not-yet-resaved analysis from
        # before this rename still loads (see the translate-to-english plan).
        'pose_found':    pose_found,
        'horizon_deg':   horizon,
        'midline_dev':   midline,
        # The corner flag is NOT a derivative: on the YOLO backend it also flags the
        # frames the detection pass skipped inference on, and that can't be recovered
        # from the landmarks (which are precisely absent there). So it's saved.
        'corner':        corner,
        'w':      np.int32(info.w),
        'h':      np.int32(info.h),
        'fps':    np.float32(info.fps),
        'total':  np.int32(info.totaal),
    }


def arrays_to_results(arrays):
    """
    Inverse of results_to_arrays: builds a fresh (VideoInfo, FrameResult list)
    with plain Landmark tuples in .lm (z=0). The derivatives are still empty — run
    process_derivatives() + segment_pushes() to fill them. frame_nr/time are
    reconstructed exactly from the frame index and fps, the way analyze_frames sets them.

    Reads either the new English npz keys or the original Dutch ones, so an npz saved
    before the English rename still loads without any data loss (see the
    translate-to-english plan). `landmarks`, `horizon_deg`, `w`, `h`, `fps` were always
    English and need no fallback.
    """
    landmarks  = arrays['landmarks']
    pose_found = arrays['pose_found'] if 'pose_found' in arrays else arrays['pose_gevonden']
    horizon    = arrays['horizon_deg']
    if 'midline_dev' in arrays:
        midline = arrays['midline_dev']
    elif 'middellijn_dev' in arrays:
        midline = arrays['middellijn_dev']
    else:
        midline = None                     # missing in old npz's, from before this field existed
    if 'corner' in arrays:
        corner = arrays['corner']
    elif 'bocht' in arrays:
        corner = arrays['bocht']
    else:
        corner = None                      # missing in old npz's, from before this field existed
    total = int(arrays['total']) if 'total' in arrays else int(arrays['totaal'])
    info = VideoInfo(int(arrays['w']), int(arrays['h']), float(arrays['fps']), total)
    fps = info.fps
    resultaten = []
    for i in range(len(landmarks)):
        r = FrameResult(frame_nr=i, time=i / fps if fps > 0 else 0.0)
        r.pose_gevonden = bool(pose_found[i])
        r.horizon_deg   = float(horizon[i])
        r.bocht         = bool(corner[i]) if corner is not None else False
        if midline is not None and not np.all(np.isnan(midline[i])):
            r.middellijn_dev = {
                naam: (float(midline[i, k]) if not np.isnan(midline[i, k]) else None)
                for k, naam in enumerate(('l_knee', 'r_knee'))}
        if r.pose_gevonden:
            r.lm = [Landmark(float(x), float(y), 0.0, float(v))
                    for x, y, v in landmarks[i]]
        resultaten.append(r)
    return info, resultaten


def save_landmarks(pad, resultaten, info):
    """Writes the analysis (landmarks + horizon + meta) compressed to an .npz."""
    np.savez_compressed(pad, **results_to_arrays(resultaten, info))


def load_landmarks(pad):
    """Loads an .npz back to (VideoInfo, FrameResult list); derivatives still empty."""
    with np.load(pad) as arrays:
        return arrays_to_results(arrays)


def segment_pushes(resultaten, min_lengte=3, alternerend=True):
    """
    Groups per-frame results into push events: contiguous frames where the same leg is
    the push leg AND the weight is still on it. `min_lengte` filters out noise (too-
    short, unreliable detections). If `alternerend`, the L/R alternation rule is then
    applied (`force_alternating`): merging split pushes, flagging impossible repeats.

    `hoek` is the angle of the event's **last frame** — exactly the frame
    `determine_push_from_extension` picked as push completion (the flattest lower-leg angle
    within the extension plateau). So `r.hoek`, not `r.smooth_hoek`: an average over
    the frames before that moment sits, by definition, too high (too steep) on a
    dropping angle. `min_hoek`/`max_hoek` come from the same sequence, so the table
    uses one definition of "the angle".

    An event inherits `onvolledig` from its frames (`FrameResult.afzet_onvolledig`):
    the push was still running when the video/pose segment ended
    (`INCOMPLETE_TRUNCATED`), or no sideways push was observed at all within the run
    (`INCOMPLETE_NO_PUSH`). In both cases the angle is the standing-up rather than the
    completed push; such an event stays visible, but belongs outside average/min/max
    (the GUI + library list do that).
    """
    events = []
    huidig = None

    def _sluit_af():
        if huidig is not None and len(huidig['hoeken']) >= min_lengte:
            start, laatste, hoeken = huidig['start'], huidig['laatste'], huidig['hoeken']
            frames = huidig['frames']
            # Perspective bycatch (None without a calibration): correction + flag of
            # the completion frame, speed averaged over the push, stroke length =
            # distance covered between the start and end position on the track.
            snelheden = [f.snelheid for f in frames if f.snelheid is not None]
            slaglengte = None
            if start.wereld_xy is not None and laatste.wereld_xy is not None:
                slaglengte = round(float(np.hypot(
                    laatste.wereld_xy[0] - start.wereld_xy[0],
                    laatste.wereld_xy[1] - start.wereld_xy[1])), 2)
            events.append(PushEvent(
                index=len(events),
                leg=huidig['been'],
                start_frame=start.frame_nr,
                end_frame=laatste.frame_nr,
                start_time=start.tijd,
                end_time=laatste.tijd,
                angle=hoeken[-1],
                min_angle=min(hoeken),
                max_angle=max(hoeken),
                incomplete=laatste.afzet_onvolledig,
                correction=laatste.hoek_correctie,
                reliable=laatste.hoek_betrouwbaar,
                speed=round(float(np.mean(snelheden)), 2) if snelheden else None,
                stroke_length=slaglengte,
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
        events = force_alternating(events, resultaten)
    return events


def analyze_video(input_pad, output_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                    num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
                    horizon_deg=0.0, auto_horizon=False, save_npz=None, from_npz=None,
                    bocht=True, deinterlacen=None):
    """
    CLI analysis in two passes: first detect/track/smooth (needed because the offline
    smoothing requires ALL the frames), then read the video again and draw + write out
    the overlay.

    With `save_npz` the landmarks (phase 0) are written out after the analysis. With
    `from_npz` the analysis is skipped: the landmarks are loaded from that .npz and
    only the derivatives are recomputed — this is how you show that a reloaded analysis
    gives the same table/overlay WITHOUT re-analyzing the video.
    """
    def toon_voortgang(frame_nr, totaal):
        if frame_nr % 30 == 0:
            # `totaal` comes from CAP_PROP_FRAME_COUNT and is regularly wrong on
            # VFR .MOVs; clamp the percentage so the CLI doesn't report 103%.
            pct = min(100.0, frame_nr / totaal * 100) if totaal > 0 else 0
            print(f"  {frame_nr}/{max(totaal, frame_nr)} frames ({pct:.0f}%)")

    # Determine once and pass to both passes: if pass 2 draws the overlay on different
    # pixels than pass 1 measured, the skeleton no longer matches the image.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    if deinterlacen:
        print("[INFO] Interlaced source detected: combing will be filtered out.")

    if from_npz:
        print(f"[INFO] Loading landmarks from {from_npz} (no detection) ...")
        info, resultaten = load_landmarks(from_npz)
        # Only recompute the derivatives; horizon_deg is already in the .npz per frame.
        process_derivatives(resultaten, info.w, info.h, info.fps, smooth_n, threshold)
    else:
        print("[INFO] Pass 1/2: detection + tracking + smoothing ...")
        info, resultaten = analyze(
            input_pad, model_pad, smooth_n, threshold, force_fps,
            num_poses=num_poses, doel_punt=doel_punt, smooth_landmarks=smooth_landmarks,
            progress_callback=toon_voortgang, horizon_deg=horizon_deg, auto_horizon=auto_horizon,
            bocht=bocht, deinterlacen=deinterlacen)
        n_bocht = sum(1 for r in resultaten if r.bocht)
        if n_bocht:
            print(f"[INFO] {n_bocht} of {len(resultaten)} frames flagged as a corner "
                  f"(not measured)")
        if save_npz:
            save_landmarks(save_npz, resultaten, info)
            print(f"[INFO] Landmarks saved: {save_npz}")

    print(f"[INFO] Video: {info.w}x{info.h} @ {info.fps:.1f}fps, {info.totaal} frames")
    print(f"[INFO] Pass 2/2: drawing overlay -> {output_pad}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_pad, fourcc, info.fps, (info.w, info.h))
    cap    = open_video(input_pad, deinterlacen)
    idx    = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx < len(resultaten):
            draw_overlay_on_frame(frame, resultaten[idx], info.fps)
        out.write(frame)
        idx += 1
        if idx % 30 == 0:
            print(f"  {idx}/{info.totaal} frames drawn")
    cap.release()
    out.release()

    events = segment_pushes(resultaten)
    print(f"\n[DONE] Result saved: {output_pad}")
    print(f"        {idx} frames processed, {len(events)} pushes found")


# ── Transitional Dutch-name aliases (module level) ──────────────────────────────
# schaats_gui.py/schaats_yolo.py/schaats_db.py/schaats_eval.py/schaats_schermtest.py
# aren't translated yet (see the translate-to-english plan) and still do
# `from schaats_analyse import <dutch name>` — now `from skate_analysis import
# <dutch name>`, since only the module's own file got renamed in those import lines,
# not the names inside them (that's each file's own phase). These aliases keep every
# such import resolving to the same object under its new English name. Remove each one
# once nothing imports it anymore.
FrameResultaat = FrameResult
AfzetEvent = PushEvent
PerspectiefConfig = PerspectiveConfig
KnipAfgebroken = TrimAborted
ONV_AFGEKAPT = INCOMPLETE_TRUNCATED
ONV_GEEN_PUSH = INCOMPLETE_NO_PUSH
BOCHT_IN = CORNER_IN
BOCHT_UIT = CORNER_OUT
sla_landmarks_op = save_landmarks
laad_landmarks = load_landmarks
arrays_naar_resultaten = arrays_to_results
resultaten_naar_arrays = results_to_arrays
verwerk_afgeleiden = process_derivatives
segmenteer_afzetten = segment_pushes
bereken_hoek_tov_ijs = calculate_angle_to_ice
bocht_ratio = corner_ratio
bepaal_bocht_reeks = determine_corner_sequence
teken_overlay_op_frame = draw_overlay_on_frame
horizon_hoek_uit_lijn = horizon_angle_from_line
detecteer_ijslijn = detect_ice_line
kader_reeks = box_sequence
maak_voorvulling = make_prefill
knip_fragmenten = trim_fragments
zet_horizon = set_horizon
fase_voortgang = phase_progress
analyseer = analyze


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skater push-angle analysis")
    parser.add_argument("--input", required=False, default=None, help="...")
    parser.add_argument("--output",    default="output.mp4",  help="Output video")
    parser.add_argument("--model",     default=None,
                        help="Path to the pose_landmarker .task model file")
    parser.add_argument("--fps",       type=float, default=None, help="Force FPS")
    parser.add_argument("--smooth",    type=int,   default=5,    help="Smoothing frames")
    parser.add_argument("--threshold", type=float, default=0.015,
                        help="Weight-detection sensitivity (0.01-0.03)")
    parser.add_argument("--num-poses", type=int, default=NUM_POSES_DEFAULT,
                        help="Max. number of skaters to detect at once")
    parser.add_argument("--heavy", action="store_true",
                        help="Use the heavy model (more accurate, slower)")
    parser.add_argument("--target", default=None, metavar="X,Y",
                        help="Normalized starting point (0-1) of the skater to track, "
                             "e.g. 0.5,0.4; default: biggest skater")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Turn off offline landmark smoothing (Savitzky-Golay)")
    parser.add_argument("--horizon", type=float, default=0.0, metavar="DEGREES",
                        help="Tilt of the ice line relative to the horizontal image axis "
                             "(positive = rising to the right); corrects a tilted camera. "
                             "Default 0 (ice = horizontal).")
    parser.add_argument("--auto-horizon", action="store_true",
                        help="Detect the ice-line tilt automatically, per frame (for a "
                             "wobbling camera); overrides --horizon.")
    parser.add_argument("--deinterlace", dest="deinterlace", action="store_true", default=None,
                        help="Filter out combing (interlaced camcorder footage). By "
                             "default this is determined per video automatically; this "
                             "flag forces it on.")
    parser.add_argument("--no-deinterlace", dest="deinterlace", action="store_false",
                        help="Never deinterlace, even if the video looks interlaced "
                             "(to run an A/B comparison).")
    parser.add_argument("--no-corner", action="store_true",
                        help="Turn off corner detection; corner frames then also yield "
                             "(unusable) push measurements.")
    parser.add_argument("--save-npz", default=None, metavar="PATH",
                        help="Write the landmarks out to this .npz after the analysis (phase 0).")
    parser.add_argument("--from-npz", default=None, metavar="PATH",
                        help="Skip the detection and load the landmarks from this .npz; "
                             "only recomputes the derivatives and redraws the overlay.")
    args = parser.parse_args()

    # If no --input was given, ask for it interactively
    if not args.input:
        args.input = input("Which video do you want to analyze? (path to file): ").strip().strip('"')

    # Check that the file actually exists
    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}")
        exit(1)

    standaard_naam = "pose_landmarker_heavy.task" if args.heavy else "pose_landmarker_full.task"
    model_pad = args.model or os.path.join(app_dir(), standaard_naam)
    if not args.from_npz and not os.path.isfile(model_pad):
        # With --from-npz there's no detection, so the model isn't needed.
        variant = "heavy" if args.heavy else "full"
        print(f"Model file not found: {model_pad}")
        print(f"Download it from: https://storage.googleapis.com/mediapipe-models/"
              f"pose_landmarker/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")
        exit(1)

    doel_punt = None
    if args.target:
        try:
            dx, dy = (float(v) for v in args.target.split(","))
            doel_punt = (dx, dy)
        except ValueError:
            print(f"Invalid --target: {args.target!r} (expected e.g. 0.5,0.4)")
            exit(1)

    analyze_video(
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
        bocht=not args.no_corner,
    )
