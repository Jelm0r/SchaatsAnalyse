"""
Skate analysis YOLO backend
============================
Detection/tracking backend built on YOLO-pose + ByteTrack (ultralytics), with an
**offline target selection** on top of it: because this is a batch tool, we first
collect *all* detections from *all* frames, and only afterwards decide globally which
detection in each frame is the target skater. That is far more robust than choosing
per frame while streaming:

1. **Detection pass at high resolution** (`DETECT_IMGSZ`): small/motion-blurred skaters
   far away would otherwise simply not be found.
2. **Tracklets + suit-color splits**: ByteTrack IDs are usually stable, but with
   crossing skaters an ID regularly "steals" the other skater. Per tracklet we watch
   the torso's HSV histogram (the suit); if the color jumps persistently, the tracklet
   gets cut right there.
3. **Chain stitching with color + direction of travel**: starting from the seed
   tracklet (a mouse click, or the biggest mover), tracklets get strung together. A
   candidate only counts if its suit color matches the reference *and* its start
   position agrees with the predicted position (constant velocity across the gap — a
   skater's direction of travel is predictable).
4. **Refinement pass** (`refine`): for each target frame the pose gets re-estimated
   with a **top-down model on the known bounding box** — preferably **RTMPose-26**
   (Halpe26, via rtmlib/ONNXRuntime): substantially more accurate than yolo26x-pose
   (~76 vs ~69.5 COCO-AP, subpixel SimCC decoding) and with *real* heel/toe keypoints,
   plus a lot faster on CPU. Detection gaps still get filled via interpolated bboxes.
   The suit color stays the gatekeeper, so the refinement never quietly latches onto
   the other skater. Without rtmlib installed, this pass falls back to the older
   square-crop + yolo26x route.

Both heavy passes run **on the GPU wherever one is available**, and otherwise just on
the CPU; see `yolo_device()`/`rtmpose_device()` below for how that's decided per pass.
Nothing about the measurements changes because of that — the same weights at the same
fp32 precision — only the running time.

The rest of the pipeline (offline smoothing, derivatives, drawing, GUI) from
`skate_analysis.py` is reused unchanged. Requires torch/ultralytics (see .venv-yolo).
YOLO delivers COCO-17 keypoints; we map those onto the MediaPipe-33 layout. Heel/toe
don't exist in COCO and get placed on the ankle there with visibility 0; RTMPose-26
delivers them for real (Halpe26 → MediaPipe 29-32).
"""
import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

# Before the ultralytics import: ultralytics' ONNX backend does
# `check_requirements("onnxruntime")` and would thereby install the CPU build of
# ONNXRuntime unasked — right on top of `onnxruntime-directml`, the package the iGPU
# route below depends on. Auto-install is therefore off; this project's dependencies
# are managed by hand (see GPU.md).
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

from ultralytics import YOLO   # at module level: this way the import fails right away
                               # if torch/ultralytics is missing, and the GUI cleanly
                               # picks MediaPipe instead.

try:                           # optional: RTMPose refinement (pip install rtmlib onnxruntime)
    from rtmlib.tools.pose_estimation import RTMPose as _RTMPose
    from rtmlib.tools.base import RTMLIB_SETTINGS as _RTMLIB_SETTINGS
    # rtmlib natively only knows cpu/cuda/rocm/mps; DirectML — the GPU route on a
    # Windows machine *without* an NVIDIA card — is missing from that table. Adding one
    # line is enough: rtmlib just looks the provider up by name afterwards.
    _RTMLIB_SETTINGS['onnxruntime'].setdefault('dml', 'DmlExecutionProvider')
    IS_RTMPOSE = True
except ImportError:
    IS_RTMPOSE = False

from skate_analysis import (
    FrameResult, Landmark, VideoInfo, video_info,
    smooth_landmarks_offline, process_derivatives, set_horizon, phase_progress,
    corner_ratio, determine_corner_sequence, NUM_POSES_DEFAULT, CORNER_IN, CORNER_OUT,
    app_dir, data_dir, open_video, is_interlaced,
)

BACKEND_NAME = ("YOLO-pose + ByteTrack + RTMPose refinement" if IS_RTMPOSE
                else "YOLO-pose + ByteTrack")


# ── Compute device: GPU where there is one, otherwise CPU ───────────────────────
# The two heavy passes run on different engines — the detection pass on torch/CUDA
# (ultralytics), the RTMPose refinement on ONNXRuntime — and they get their GPU
# support from different packages (a CUDA build of torch, resp. `onnxruntime-gpu`). A
# machine can easily have one but not the other, so the device is determined per pass
# separately and each pass falls back to the CPU independently of the other. Nothing
# about the outcome changes: the same weights at the same fp32 precision, just faster.
# (Half precision would be faster still, but changes the keypoints in the last decimal
# places and thereby the measured angles — that's a measurement change and shouldn't
# be a side effect of a speed optimization.)
def _cpu_forced():
    """`SCHAATSANALYSE_CPU=1` forces both passes to the CPU — needed to compare an
    analysis on GPU against an analysis on CPU without tearing down the environment."""
    return bool(os.environ.get('SCHAATSANALYSE_CPU'))


@lru_cache(maxsize=1)
def yolo_device():
    """
    Device for the YOLO passes in ultralytics notation: 'cuda' if torch sees a usable
    GPU, otherwise 'cpu'. Torch gets imported **locally** here (ultralytics has already
    pulled it in), and the outcome is cached per process: `cuda.is_available()`
    initializes the CUDA driver, and that doesn't need to happen again every frame.
    """
    if _cpu_forced():
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
    """This installation's ONNXRuntime providers, or an empty set."""
    try:
        import onnxruntime as ort
        return frozenset(ort.get_available_providers())
    except Exception:
        return frozenset()


@lru_cache(maxsize=1)
def rtmpose_device():
    """
    Device for the RTMPose refinement: 'cuda' if ONNXRuntime has a CUDA provider,
    'dml' if it has a DirectML provider, otherwise 'cpu'. That's a *different* question
    than `yolo_device()` — this pass doesn't run on torch, so a CUDA-capable torch says
    nothing about what ONNXRuntime can do. If only the plain `onnxruntime` is installed
    (without `onnxruntime-gpu`/`onnxruntime-directml`), there is a GPU but this pass
    can't reach it, and it quietly stays on the CPU.
    """
    if _cpu_forced():
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
    True if the **detection pass** can go via DirectML on the GPU — the route for a
    machine without an NVIDIA card (integrated Intel/AMD GPU), where `yolo_device()`
    always says 'cpu' because CUDA can't do anything there in principle.

    DirectML is not a torch device: it only exists inside ONNXRuntime. The path
    therefore doesn't go via `device=`, but via the **exported ONNX model** that
    ultralytics can also load (see `_load_yolo`). CUDA wins if available: if this
    machine has a usable NVIDIA GPU, the ordinary torch route is faster and closer to
    the reference measurement in GPU.md.
    """
    if _cpu_forced() or yolo_device() == 'cuda':
        return False
    return 'DmlExecutionProvider' in _ort_providers()


_gpu_fallen_back = False   # after a CUDA OOM: the rest of this run runs on the CPU


def _infer(call, warning_callback=None):
    """
    Run an ultralytics call on the chosen device, with the CPU as a safety net for
    when the GPU memory fills up. `call(device)` does the actual work.

    A laptop GPU has little VRAM and shares it with the desktop, so yolo26x-pose at
    `DETECT_IMGSZ` can just fail to fit — and then a ten-minute analysis would still
    crash halfway through on an OOM. After such an error the whole run switches over
    to the CPU **permanently**: falling back per frame would let the memory fill up
    again every time. Switching devices halfway is harmless for the measurement — it's
    the same weights at the same precision.
    """
    global _gpu_fallen_back
    device = 'cpu' if _gpu_fallen_back else yolo_device()
    try:
        return call(device)
    except Exception as exc:
        if device == 'cpu' or 'out of memory' not in str(exc).lower():
            raise
        _gpu_fallen_back = True
        try:                       # give back the stuck memory before the retry
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        message = ("GPU memory full; the analysis continues on the CPU and will "
                   "therefore take longer.")
        if warning_callback:
            warning_callback(message)
        else:
            print(message)
        return call('cpu')


# yolo26x-pose = most accurate (slowest on CPU). Alternative: "yolo26m-pose.pt"
# (faster, slightly less accurate). Ultralytics downloads the model on first use.
# Note: YOLOv12 was never released as a pose model (detection only); YOLO26 is
# ultralytics' newest pose model, the successor to yolo11x-pose.
# Deliberately just the filename; `analyze()` makes it absolute relative to `app_dir()`.
# A bare name is anchored to the working directory, and started from a shortcut
# ultralytics wouldn't find the model there and would download 126 MB again to some
# arbitrary folder.
DEFAULT_YOLO_MODEL = "yolo26x-pose.pt"

# ── Detection ────────────────────────────────────────────────────────────────────
DETECT_IMGSZ = 1280       # inference resolution for the detection pass; 640 misses far/blurry skaters

# opset 17 for the ONNX export: higher yields operators the DirectML provider doesn't
# all know, which then silently run on the CPU — exactly the gain we're here to get.
ONNX_OPSET = 17


def _onnx_pad(pt_pad):
    """Path of the DirectML model: next to the .pt file if it's there, otherwise in the
    writable data folder.

    Normally nothing gets written — in the repo the export is already there after the
    first time, and an installation ships it. But should it still be missing, the
    one-time export needs somewhere that's certainly writable: an installation folder
    can be read-only, and then the GPU route would fail again on every analysis.
    """
    stem, _ = os.path.splitext(pt_pad)
    beside = f"{stem}-dml.onnx"
    if os.path.exists(beside):
        return beside
    return os.path.join(data_dir(), os.path.basename(beside))


@contextmanager
def _dml_sessions():
    """
    Makes ONNXRuntime sessions built within this block run on DirectML.

    Ultralytics picks its own provider and knows exactly three: CUDA, CoreML and CPU
    (`ultralytics/nn/backends/onnx.py`). DirectML isn't among them, and there's no knob
    to pass it — so `InferenceSession` gets temporarily replaced here by a variant that
    puts the DirectML provider first. Only a request that would otherwise land on the
    CPU gets redirected; if the caller already asks for a provider itself (rtmlib
    does), that choice is left alone.
    """
    import onnxruntime as ort
    original = ort.InferenceSession

    def make(path, sess_options=None, providers=None, **kw):
        if not providers or list(providers) == ['CPUExecutionProvider']:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        return original(path, sess_options, providers=providers, **kw)

    ort.InferenceSession = make
    try:
        yield
    finally:
        ort.InferenceSession = original


class _DmlYolo:
    """
    The YOLO detection model on DirectML, with the plain .pt model on the CPU as a
    safety net.

    From the outside it looks like an ordinary YOLO model: `track()` and `predict()`
    pass through unchanged. Two things this shell does handle itself:

    - **Building the session inside the patch.** Ultralytics only creates the
      ONNXRuntime session on the first inference, so loading the model within
      `_dml_sessions()` alone isn't enough — one dummy frame goes through while the
      patch is active. After that, `track()` reuses that same predictor and thus the
      same session.
    - **Falling back to the CPU** if DirectML drops out halfway (memory full, a driver
      letting go of the session). Same trade-off as the CUDA OOM in `_infer`: a
      ten-minute analysis shouldn't crash on a device problem. ByteTrack starts with
      fresh IDs after such a switch, but target selection is offline and strings
      tracklets together on suit color and direction of travel — an ID break is the
      normal case there, not an exception.
    """

    def __init__(self, onnx_pad, pt_pad, warning_callback=None):
        self._pt_pad = pt_pad
        self._warning = warning_callback
        self._dml = True
        # How many times the ByteTrack state has been restarted. `BYTETracker.__init__`
        # calls `reset_id()`, so after a model switch the IDs count from 1 again, and
        # `_build_tracklets` (which groups purely by ID) would glue an old and a new
        # track together. `_detect_all` reads this counter and therefore keeps the ID
        # spaces of each generation apart.
        self.generation = 0
        with _dml_sessions():
            self._model = YOLO(onnx_pad, task='pose')
            # At the full DETECT_IMGSZ, even though this is only a dummy: the model is
            # dynamic, but the end2end head does a TopK over `max_det` (300) positions
            # and those aren't there on a small image — at 64 px the DirectML session
            # crashes right on it. Costs one extra graph build (the real frames are
            # letterboxed rectangularly), and that's a few seconds per analysis.
            self._model.predict(np.zeros((DETECT_IMGSZ, DETECT_IMGSZ, 3), np.uint8),
                                imgsz=DETECT_IMGSZ, verbose=False, device='cpu')

    def __getattr__(self, name):
        # Anything this shell doesn't handle itself goes to the underlying model —
        # `predictor`, for example, since that's where ultralytics hangs the trackers,
        # and `_restart_tracker` needs to reach them. Via `__dict__` rather than
        # `self._model`, otherwise a lookup before `_model` exists would call itself
        # forever.
        model = self.__dict__.get('_model')
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def track(self, *args, **kw):
        return self._call('track', *args, **kw)

    def predict(self, *args, **kw):
        return self._call('predict', *args, **kw)

    def _call(self, name, *args, **kw):
        try:
            return getattr(self._model, name)(*args, **kw)
        except np.linalg.LinAlgError:
            # Not the GPU: this comes from ByteTrack's Kalman update, which uses
            # `np.linalg.solve` on the projected covariance and hits a degenerate
            # matrix with "Singular matrix" (ultralytics/trackers/utils/
            # kalman_filter.py). That's arithmetic *after* inference and says nothing
            # about the device. Let it through, so `_detect_all` can restart the
            # tracker and the device stays what it is — the broad branch below used to
            # blame this arithmetic error on DirectML, report it to the user and put
            # the rest of the analysis on the CPU (~2x slower here) for nothing.
            raise
        except Exception as exc:
            if not self._dml:
                raise
            self._dml = False
            self._model = YOLO(self._pt_pad)
            self.generation += 1        # new model = new BYTETracker = IDs from 1
            _notify(self._warning,
                  f"The GPU (DirectML) dropped out ({exc}); the analysis continues on "
                  "the CPU and will therefore take longer.")
            return getattr(self._model, name)(*args, **kw)


def _restart_tracker(model):
    """Clears the ByteTrack state, without switching model or device.

    Ultralytics hangs the trackers off the predictor (`on_predict_start`), so they only
    exist *after* the first `track()` — precisely the moment they can break. Returns
    False if there's nothing to restart; the caller then re-raises the error instead of
    swallowing it silently.
    """
    trackers = getattr(getattr(model, 'predictor', None), 'trackers', None)
    if not trackers:
        return False
    done = False
    for t in trackers:
        reset = getattr(t, 'reset', None)
        if callable(reset):
            reset()
            done = True
    return done


def _notify(warning_callback, text):
    """A silent fallback should reach the user — via the callback in the GUI, via the
    output on the CLI."""
    if warning_callback:
        warning_callback(text)
    else:
        print(text)


def _load_yolo(pt_pad, warning_callback=None):
    """
    The detection model, on the fastest device this machine offers.

    Without the DirectML route (an NVIDIA machine, or no `onnxruntime-directml`) this
    is just `YOLO(pt_pad)`: torch itself picks CPU or CUDA via `yolo_device()`. With
    DirectML it goes via an ONNX export of the same weights, made once per model
    (~15 s) and then kept next to the .pt file.

    **`dynamic=True` is not a detail but the crux of the measurement equivalence.**
    Ultralytics letterboxes a .pt model *rectangularly* (only up to a multiple of the
    stride), but an ONNX model with a fixed input shape gets the image pasted into a
    **square** of `DETECT_IMGSZ` — so with a wide gray border added. The net then sees
    a different picture, and on this clip that's no theoretical difference: measured,
    coverage dropped from 100/103 to 89/103, event boundaries shifted and two push
    angles changed by 18°. With a dynamic input shape, ultralytics falls back to
    exactly the same rectangular letterbox as with the .pt model, and the measurement
    is equal again (0.0 px median difference, the same eight pushes with the same
    angles — see GPU.md). It also costs nothing in speed: all frames of one video have
    the same shape, so DirectML builds its graph once.

    Every step can fail — no `onnx`/`onnxslim` for the export, a session that won't
    build — and the answer is always the same: report it and continue on the CPU. A
    slower analysis is better than no analysis.
    """
    if not yolo_dml():
        return YOLO(pt_pad)
    onnx_pad = _onnx_pad(pt_pad)
    try:
        if not os.path.exists(onnx_pad):
            _notify(warning_callback,
                  f"Exporting the model for the GPU, one time ({os.path.basename(onnx_pad)})...")
            uit = YOLO(pt_pad).export(format='onnx', imgsz=DETECT_IMGSZ,
                                      opset=ONNX_OPSET, dynamic=True)
            os.replace(uit, onnx_pad)
        return _DmlYolo(onnx_pad, pt_pad, warning_callback)
    except Exception as exc:
        _notify(warning_callback,
              f"The GPU route (DirectML) couldn't be set up ({exc}); the analysis "
              "runs on the CPU.")
        return YOLO(pt_pad)

# ── Skipping the corner (time savings) ──────────────────────────────────────────
# The detection pass is ~94% of the analysis time, and in the corner that time buys
# nothing: there's no usable frontal measurement to be made there. As soon as
# `_CornerGuard` says we're in the corner, only one inference every `CORNER_CHECK_S`
# still runs, to check whether the straight section has started again — the rest of
# the frames still get *read* (decoding is negligible, and this way the frame
# numbering stays exact) but not run through inference.
#
# Why this is safe: the refinement pass fills detection gaps up to GAP_FILL_S (1.0 s)
# with interpolated bboxes and estimates the pose there top-down anyway. The gaps that
# skipping leaves behind are `CORNER_CHECK_S` long, well within that: if we skipped
# somewhere wrongly, pass 2 simply recovers those frames. Skipping too little costs
# time, skipping too much costs (almost) no coverage.
CORNER_CHECK_S    = 0.33   # how often inference still runs in the corner (~ every 10
                          # frames at 30 fps), but frame-rate independent
CORNER_START_S    = 0.5    # that much corner evidence is needed (someone in frame, but
                          # turned) before we start skipping frames
CORNER_STILL_S    = 3.0    # ... or that long with nobody measurable in frame at all. Well
                          # above the longest detection gap on a straight section in the
                          # library (2.1 s), so a blur gap doesn't push the analysis
                          # into skip mode
CORNER_MOVE_WINDOW_S = 1.0  # window over which "is this person moving?" is measured
CORNER_MIN_MOVEMENT = 0.10 # displacement + growth of the bbox in that window, as a
                          # fraction of the person's own body height. A bystander along
                          # the boards stands frontal in frame and would otherwise keep
                          # the analysis at full speed forever (same motive as
                          # MIN_MOVEMENT in target selection). A skater coming straight
                          # at the camera barely moves in frame but *grows* ~18%/s —
                          # hence growth counts too

# ── Suit color (torso HSV histogram) ────────────────────────────────────────────
COLOR_BINS      = (8, 4, 3)  # H, S, V — compact, robust to scale/lighting
COLOR_MATCH_MIN = 0.45       # min. similarity (1 - Bhattacharyya) to count as the target
COLOR_SPLIT_MIN = 0.35       # within a tracklet: persistently below this → ID theft, cut
COLOR_SPLIT_N   = 3          # number of consecutive deviating frames before the cut
REF_HIST_N      = 25         # reference = average of the most recent N target histograms

# ── Chain stitching (color + direction of travel) ───────────────────────────────
STITCH_MAX_GAP_S  = 2.0      # max. time gap that may be stitched
STITCH_GATE_BASIS = 0.06     # distance gate (normalized) at gap 0 ...
STITCH_GATE_GROWTH = 0.015   # ... which grows per gap frame (uncertainty of the prediction)
STITCH_OVERLAP_GATE = 0.05   # if a candidate overlaps the chain, it must lie within this
                             # distance on the shared frames: the same skater under two
                             # IDs, not the bystander who stands in frame the whole clip
SPEED_WINDOW      = 5        # number of detections over which the speed is estimated
MIN_MOVEMENT      = 0.06     # tracklet path length below this = stationary bystander
CLICK_SEARCH_S    = 6.0      # how long (in seconds) we keep looking for the clicked skater
CLICK_GATE_BASE   = 0.05     # how far the click may lie next to a skater to still mean ...
CLICK_GATE_GROWTH = 0.02     # ... them, plus this much per second elapsed since the click
SEED_MIN_LEN      = 5        # detections; a shorter seed gives too thin a color reference
BOOTSTRAP_MAX_GAP = 3        # frames; this close, a fragment may fill out a short seed

# ── Refinement ───────────────────────────────────────────────────────────────────
GAP_FILL_S      = 1.0        # max. detection gap that gets filled via interpolated bboxes
# RTMPose-26 (Halpe26 = COCO-17 + head/neck/hip-center + feet), top-down on the target
# bbox. 'body7' weights = trained on 7 datasets, robust on sports footage.
RTMPOSE_MODEL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                 'onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288'
                 '-7fb6e239_20230606.zip')
# Name of the model shipped alongside the app (installation); rtmlib's `BaseTool` does
# `if not os.path.exists(onnx_model): download_checkpoint(...)`, so a local path works
# without a patch and the URL stays the fallback for a bare repo environment.
RTMPOSE_LOCAL = "rtmpose-x-halpe26-384x288.onnx"
RTMPOSE_INPUT = (288, 384)   # (width, height) of the model input
RTMPOSE_MIN_SCORE = 0.3      # min. average leg keypoint score to trust the estimate
# Fallback route without rtmlib (square crop through yolo26x):
REFINE_IMGSZ    = 640        # inference size on the crop
REFINE_MARGIN   = 1.9        # crop side = margin × largest bbox side
REFINE_MIN_PX   = 256        # lower bound on the crop side (pixels)

# ── Spyglass: following a too-small skater from a drawn box ────────────────────
# The detection pass only picks up a skater at roughly 80-130 px height (1080p); below
# that there's no bbox and hence no refinement either. With a box the user draws on
# the first frame, the spyglass propagates the bbox from frame to frame out of its own
# refined keypoints (see `_Spyglass`). No second detector: RTMPose on the previous bbox
# is ~30-100 ms, and a YOLO `predict` on a crop during the `track` session would feed
# ByteTrack with crop coordinates.
SPYGLASS_COAST_S    = 1.0    # how long the propagation keeps coasting without a hit
                             # (constant velocity) before the run dies; same order as
                             # GAP_FILL_S
SPYGLASS_MARGIN     = 0.10   # border around the keypoint extent for the next bbox
                             # (the rtmlib path itself adds another ×1.25, so this
                             # stays small)
SPYGLASS_LINK_IOU   = 0.3    # link test: IoU of the propagated bbox with the chain
                             # bbox on the frame where the run touches the chain
SPYGLASS_JUMP       = 0.5    # plausibility per step: the center may jump at most this
                             # fraction of the bbox height ...
SPYGLASS_GROWTH     = 1.5    # ... and the height at most grow/shrink by this factor;
                             # more means RTMPose has landed on someone else
SPYGLASS_START_SCALES = (0.7, 1.0, 1.4)   # first frame: three scalings of the hand-
                             # drawn box, the highest leg score wins (a hand rarely
                             # draws tight)
BOX_SIZE_MAX        = 3.0    # seed gate *with* a box: a candidate may be at most this
                             # much taller than the box (excludes a bystander next to
                             # the click; in 6 s an approaching skater grows ~1.7x,
                             # measured)
BOX_MIN_HEIGHT_PX   = 70     # below this there's nothing to measure: on `00000 16-14`
                             # (skater 35-57 px) RTMPose gave leg scores of 0.1-0.2 with
                             # fewer than four usable keypoints, and YOLO on a 5x
                             # magnified crop only saw a sporadic blob — the legs are
                             # then ~15 px, and 2 px of error is already 4°
                             # (OPNAME.md). Only a warning.

# COCO-17 keypoint index → MediaPipe 33-landmark index.
COCO_TO_MP = {
    0: 0,             # nose
    5: 11, 6: 12,     # shoulders  (L, R)
    7: 13, 8: 14,     # elbows
    9: 15, 10: 16,    # wrists
    11: 23, 12: 24,   # hips
    13: 25, 14: 26,   # knees
    15: 27, 16: 28,   # ankles
}
# COCO indices of the torso corner points in polygon order (for the color mask).
# Halpe26 has the same first 17 indices as COCO, so this holds for both.
TORSO_COCO = (5, 6, 12, 11)

# Halpe26 extras (after the 17 COCO points) → MediaPipe index. Small toes (22/23) have
# no MediaPipe equivalent and stay unused.
HALPE_TO_MP = {20: 31, 21: 32,   # big toes (L, R) → foot_index
               24: 29, 25: 30}   # heels (L, R)


def _coco_to_landmarks(kp_xy, kp_conf, w, h):
    """COCO-17 keypoints (pixels + conf) → list of 33 normalized Landmarks."""
    lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for c, m in COCO_TO_MP.items():
        x, y = kp_xy[c]
        lm[m] = Landmark(float(x) / w, float(y) / h, 0.0, float(kp_conf[c]))
    # Put heel/toe on the ankle with visibility 0 (they don't exist in COCO).
    for m_foot, m_ankle in ((29, 27), (31, 27), (30, 28), (32, 28)):
        a = lm[m_ankle]
        lm[m_foot] = Landmark(a.x, a.y, 0.0, 0.0)
    return lm


def _halpe26_to_landmarks(kp_xy, kp_conf, w, h):
    """Halpe26 keypoints (pixels + conf) → 33 normalized Landmarks, with feet."""
    lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for c, m in list(COCO_TO_MP.items()) + list(HALPE_TO_MP.items()):
        x, y = kp_xy[c]
        vis = float(np.clip(kp_conf[c], 0.0, 1.0))
        lm[m] = Landmark(float(x) / w, float(y) / h, 0.0, vis)
    return lm


# ── Suit color ───────────────────────────────────────────────────────────────────
def _torso_hist(frame_bgr, kp_xy, kp_conf, bbox_px=None):
    """
    HSV histogram of the torso (polygon shoulders→hips) — the "color of the suit".
    Falls back to the central upper part of the bounding box if the torso keypoints
    are unreliable. Returns `(hist_or_None, from_mask)`.

    That second value is the **origin**, and it matters: a mask histogram contains
    only suit pixels, a bbox fallback also contains background (ice, boards,
    audience). The two aren't interchangeable, so a bbox histogram may not be held to
    the same thresholds against a mask reference — otherwise such a frame would drop
    below the split threshold unfairly and the tracklet splitter would cut a gap where
    nothing is wrong.
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
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(COLOR_BINS),
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
    return hist, mask is not None


def _hist_sim(a, b):
    """Color similarity in [0, 1]: 1 - Bhattacharyya distance."""
    if a is None or b is None:
        return None
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class ColorReference:
    """Running suit-color reference: average of the most recent N target histograms."""
    def __init__(self):
        self.hists = deque(maxlen=REF_HIST_N)
        self._sum = None

    def add(self, hist):
        if hist is None:
            return
        if len(self.hists) == self.hists.maxlen:
            self._sum -= self.hists[0]
        self._sum = hist.copy() if self._sum is None else self._sum + hist
        self.hists.append(hist)

    @property
    def hist(self):
        if not self.hists:
            return None
        ref = self._sum / len(self.hists)
        return ref

    def sim(self, hist):
        return _hist_sim(self.hist, hist)


# ── Detections & tracklets ───────────────────────────────────────────────────────
@dataclass
class Detection:
    """One person detection in one frame (everything normalized to frame size)."""
    frame: int
    tid: object                 # ByteTrack ID or None
    centroid: tuple
    bbox: tuple                 # (x1, y1, x2, y2)
    area: float
    lm: list                    # MediaPipe-33 Landmarks
    hist: object = None         # torso HSV histogram or None
    hist_mask: bool = False     # True = from the torso polygon, False = bbox fallback

    @property
    def ref_hist(self):
        """The histogram to the extent it may serve as evidence: only the mask
        variant. The bbox fallback contains background and would pollute the
        reference, and would score systematically too low against a mask reference."""
        return self.hist if self.hist_mask else None


class _CornerGuard:
    """
    Decides during the detection pass which frame still gets inference. The corner is
    unusable footage for this tool, so the expensive model doesn't need to run over
    every frame there — one check per `CORNER_CHECK_S` is enough to notice the straight
    section has started again.

    State per analyzed frame, from that frame's own detections:
    - **frontal** — someone with `corner_ratio` >= `CORNER_OUT` who is also actually
      moving. Puts the guard straight back to full speed.
    - **turned** — someone measurable in frame, but with `corner_ratio` < `CORNER_IN`:
      the corner. Starts skipping after `CORNER_START_S`.
    - **nothing** — nobody measurable. Could be the corner (skater too far/too small),
      but just as easily a blur gap on the straight section; hence only after
      `CORNER_STILL_S`.
    - If everybody sits between the two thresholds (the hysteresis band), this frame
      says nothing and the counters stay where they were — "undecided" is deliberately
      not the same as "nobody in frame".

    The movement requirement keeps bystanders along the boards out of the "frontal"
    vote — they stand frontal in frame and would otherwise keep the analysis at full
    speed forever. A track with too little history gets the benefit of the doubt
    (counts as moving), so we never start skipping purely because we haven't seen
    someone long enough yet.
    """

    def __init__(self, fps, w, h, on=True):
        self.enabled = on
        self.w, self.h = w, h
        fps = fps or 30.0
        self.check    = max(1, int(round(CORNER_CHECK_S * fps)))
        self.n_start  = max(1, int(round(CORNER_START_S * fps)))
        self.n_still  = max(1, int(round(CORNER_STILL_S * fps)))
        self.window   = max(2, int(round(CORNER_MOVE_WINDOW_S * fps)))
        self.skip     = False
        self.corner_n = 0
        self.still_n  = 0
        self.last     = None          # last frame that got inference
        self.tracks   = {}            # tid → deque of (frame, cx_px, cy_px, height_px)

    def should_infer(self, f):
        """Does frame `f` get inference?"""
        if not self.enabled or not self.skip:
            return True
        return self.last is None or (f - self.last) >= self.check

    def _is_moving(self, d):
        """Displacement + growth of this person over the last window, as a fraction of
        their own body height. Growth counts because a skater coming straight at the
        camera barely moves in frame but does grow."""
        track = self.tracks.get(d.tid)
        if d.tid is None or track is None or len(track) < 2:
            return True                                   # too little history: benefit of the doubt
        f0, x0, y0, h0 = track[0]
        f1, x1, y1, h1 = track[-1]
        if (f1 - f0) < max(2, self.window // 2) or h1 <= 0:
            return True                                   # too short a stretch to say anything
        # Displacement + growth, scaled to "per second" (`window` = 1 s worth of
        # frames) and expressed in the person's own body height, so distance to the
        # camera drops out.
        movement = (np.hypot(x1 - x0, y1 - y0) + abs(h1 - h0)) / h1
        return movement * self.window / (f1 - f0) >= CORNER_MIN_MOVEMENT

    def feed(self, f, dets):
        """Process the detections of an inferred frame."""
        self.last = f
        if not self.enabled:
            return
        frontal = turned = seen = False
        for d in dets:
            height = (d.bbox[3] - d.bbox[1]) * self.h
            if d.tid is not None:
                track = self.tracks.setdefault(d.tid, deque(maxlen=self.window))
                track.append((f, d.centroid[0] * self.w, d.centroid[1] * self.h, height))
            ratio = corner_ratio(d.lm, self.w, self.h)
            if ratio is None:
                continue
            seen = True
            if ratio >= CORNER_OUT and self._is_moving(d):
                frontal = True
            elif ratio < CORNER_IN:
                turned = True

        # In skip mode there are `check` frames between two measurements; the counters
        # run in frames, so count the skipped frames too when we're skipping.
        step = self.check if self.skip else 1
        if frontal:
            self.skip = False
            self.corner_n = self.still_n = 0
        elif turned:
            self.still_n = 0
            self.corner_n += step
            if self.corner_n >= self.n_start:
                self.skip = True
        elif not seen:
            self.corner_n = 0
            self.still_n += step
            if self.still_n >= self.n_still:
                self.skip = True


def _detect_all(input_pad, model, info, imgsz=DETECT_IMGSZ, progress_callback=None,
                bocht=True, waarschuwing_callback=None, deinterlacen=False):
    """
    Pass 1: YOLO-pose + ByteTrack over the video at high resolution. Returns
    `(frames, excluded)`: per frame a list of Detections (all people, with a torso
    color histogram) and per frame whether it falls outside the measurement.

    **`excluded` covers two kinds of frames**, and both belong under "the corner
    wasn't analyzed": the frames that got skipped (no inference), and the **check
    frames** — the frames that, in skip mode, did get inference, purely to see whether
    the straight section has started again. Such a check frame does have a skeleton,
    but it's a glance, not a measurement: it sits in the middle of a stretch that
    otherwise wasn't looked at, so the neighboring frames that would need to prove a
    push are missing. That frame's own verdict does count (it may end the corner —
    that's what it's for), but it never yields a push angle of its own.

    The loop reads the frames itself instead of letting `model.track(source=pad,
    stream=True)` stream — otherwise there's no way to read a frame without running
    inference on it, and that's exactly what `_CornerGuard` wants in the corner (see
    there). Every frame gets read, so the frame numbering stays exactly equal to the
    video's; decoding is negligible next to the ~2 s of inference per frame.
    """
    w, h = info.w, info.h
    frames, excluded = [], []
    # ByteTrack starts back at ID 1 after every restart, and `_build_tracklets` groups
    # purely by ID — without its own ID space per generation, an old and a new track
    # would melt into one tracklet, which in the worst case glues two different skaters
    # together. Hence: every generation starts above the highest ID already handed out.
    restarts, prev_gen, id_base, highest_tid = 0, 0, 0, 0
    guard = _CornerGuard(info.fps, w, h, on=bocht)
    cap = open_video(input_pad, deinterlacen)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {input_pad}")
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not guard.should_infer(f):
            frames.append([])
            excluded.append(True)
        else:
            # If the guard was in skip mode, this is a check frame.
            excluded.append(guard.skip)

            def run_track(dev, _f=frame):
                return model.track(_f, persist=True, imgsz=imgsz,
                                   tracker='bytetrack.yaml', classes=[0],
                                   verbose=False, device=dev)

            try:
                res = _infer(run_track, waarschuwing_callback)[0]
            except np.linalg.LinAlgError as exc:
                # ByteTrack's Kalman update hits a degenerate covariance ("Singular
                # matrix"). One frame isn't worth that: clear the tracker and redo this
                # frame, on the same device. Deliberately **no** message to the user —
                # no device changes, nothing about the measurement changes, and the
                # only thing that really breaks is the ByteTrack IDs, which the offline
                # target selection is built to handle anyway (tracklets get strung
                # together on suit color and direction of travel). Do log it though: if
                # this happens often it says something about the recording, and then
                # you want to be able to look it up.
                if not _restart_tracker(model):
                    raise
                restarts += 1
                print(f"[tracker] frame {f}: {exc}; ByteTrack restarted "
                      f"({restarts}x in this analysis)")
                res = _infer(run_track, waarschuwing_callback)[0]

            gen = getattr(model, 'generation', 0) + restarts
            if gen != prev_gen:
                prev_gen, id_base = gen, highest_tid
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
                    hist, from_mask = _torso_hist(frame, xy[i], conf[i], bbox_px)
                    dets.append(Detection(
                        frame=f,
                        tid=(id_base + int(ids[i])) if ids is not None else None,
                        centroid=(cx / w, cy / h),
                        bbox=(bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h),
                        area=(bw * bh) / (w * h),
                        lm=_coco_to_landmarks(xy[i], conf[i], w, h),
                        hist=hist,
                        hist_mask=from_mask,
                    ))
            for d in dets:
                if d.tid is not None and d.tid > highest_tid:
                    highest_tid = d.tid
            guard.feed(f, dets)
            frames.append(dets)
        f += 1
        if progress_callback is not None:
            progress_callback(f, info.totaal)
    cap.release()
    return frames, excluded


def _build_tracklets(frames):
    """Group detections per ByteTrack ID into tracklets (frame-sorted)."""
    tracklets = {}
    for dets in frames:
        for d in dets:
            if d.tid is not None:
                tracklets.setdefault(d.tid, []).append(d)
    return list(tracklets.values())


def _split_by_color(tracklet):
    """
    Cut a tracklet at the spots where the suit color persistently jumps — that's
    almost always ByteTrack hanging the other skater onto the same ID after a
    crossing/occlusion. Returns a list of sub-tracklets.

    Only mask histograms (`Detection.ref_hist`) may cut: a bbox fallback contains
    background and therefore scores systematically low against a mask reference — it
    would otherwise cut a gap where nothing is wrong. Such a frame therefore doesn't
    count towards the cut, but doesn't reset the counter either: it simply gives no
    verdict.
    """
    pieces, current = [], []
    ref = ColorReference()
    low = []                        # detections since the first deviating frame
    n_low = 0                       # of those: the number with an actual verdict
    for d in tracklet:
        if d.ref_hist is None:
            (low if low else current).append(d)   # no usable histogram: no verdict
            continue
        s = ref.sim(d.ref_hist)                    # None while the reference is empty
        if s is not None and s < COLOR_SPLIT_MIN:
            low.append(d)
            n_low += 1
            if n_low >= COLOR_SPLIT_N:
                # Persistently a different suit: cut before the first deviating frame.
                if current:
                    pieces.append(current)
                current, ref, low, n_low = list(low), ColorReference(), [], 0
                for d2 in current:
                    ref.add(d2.ref_hist)
            continue
        current.extend(low); low, n_low = [], 0   # short dip (occlusion mix): keep it
        current.append(d)
        ref.add(d.ref_hist)
    current.extend(low)
    if current:
        pieces.append(current)
    return pieces


def _path_length(tracklet):
    """Total distance covered by the centroid (normalized)."""
    return float(sum(
        np.hypot(b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1])
        for a, b in zip(tracklet, tracklet[1:])))


def _distance_to_box(point, bbox):
    """Shortest distance from a point to a bounding box; 0 if the point is inside it."""
    x, y = point
    return float(np.hypot(max(bbox[0] - x, 0.0, x - bbox[2]),
                          max(bbox[1] - y, 0.0, y - bbox[3])))


def _choose_seed(tracklets, frames, doel_punt, fps=25.0, doel_kader=None):
    """
    Choose the starting tracklet. With a mouse click, in two steps; without a click:
    the biggest *mover* — median area × path length — so stationary bystanders along
    the boards never get chosen.

    With a drawn **box** (`doel_kader`, normalized xyxy), two extra rules apply. A
    candidate may be at most `BOX_SIZE_MAX`x as tall as the box — the user indicated
    the skater's size, so a bystander four times that size next to the click isn't who
    was meant. And there is **no fallback to the biggest mover**: whoever draws a box
    points at one specific skater, and if the detection pass doesn't see them anywhere
    in the search window, it's up to the spyglass (`_Spyglass`) to follow them from the
    box, not up to a guess.

    The click sits on the first frame, but the skater being pointed at doesn't have to
    be detected there yet: far away and small, it can take seconds on real footage
    before the model picks them up (measured: 3.4 s on `00005 8-41`). So we keep
    searching for `CLICK_SEARCH_S` seconds at the clicked spot:

    1. **On the click.** Walk those frames chronologically and take the first tracklet
       whose bounding box contains the click point (closest centroid if there are
       several).
    2. **Next to the click.** Still nobody? Then the detection whose bounding box lies
       closest to the click wins, within a gate that grows per second — the skater has
       after all moved on since the click. Among the candidates the smallest distance
       wins, not the distance divided by the gate: that would reward the late guess
       (see the nearest-first rule in `_stitch_chain`).

    **We deliberately don't extrapolate back at a constant velocity** to the click
    frame, the way `_stitch_chain` does over its gaps. Over a few frames that model
    holds, but not over seconds: the torso sways side to side with every stroke, and
    that stroke motion is much bigger than the net displacement. Measured on
    `00005 8-41` (frontal footage, skater skating in): the torso's x swings between
    0.33 and 0.50 while it barely moves net, and the velocity from five frames places
    it, extrapolated back, at x = -0.001 to -1.563 — outside the frame. The position in
    the frame is the reliable signal here, and the click itself is the best estimate
    of it.

    Returns `(tracklet_or_None, click_missed)`. `click_missed` is True if there was a
    click but step 2 also found nobody — then the biggest mover has been chosen
    silently, and the user should know that (it could be the wrong skater).
    """
    click_missed = False
    if doel_punt is None and doel_kader is not None:
        doel_punt = ((doel_kader[0] + doel_kader[2]) / 2, (doel_kader[1] + doel_kader[3]) / 2)
    if doel_punt is not None:
        dx, dy = doel_punt
        max_height = (BOX_SIZE_MAX * (doel_kader[3] - doel_kader[1])
                      if doel_kader is not None else None)
        window = min(int(round(max(fps, 1.0) * CLICK_SEARCH_S)), len(frames))
        per_frame = {}
        for t in tracklets:
            for d in t:
                if d.frame < window and (max_height is None
                                          or d.bbox[3] - d.bbox[1] <= max_height):
                    per_frame.setdefault(d.frame, []).append((d, t))
        for f in range(window):
            hits = [(d, t) for d, t in per_frame.get(f, ())
                    if d.bbox[0] - 0.03 <= dx <= d.bbox[2] + 0.03
                    and d.bbox[1] - 0.03 <= dy <= d.bbox[3] + 0.03]
            if hits:
                # With several hits, first prefer a tracklet long enough for a usable
                # color reference; if everything is short, the click just counts.
                long_enough = [dt for dt in hits if len(dt[1]) >= SEED_MIN_LEN]
                return min(long_enough or hits, key=lambda dt: (dt[0].centroid[0] - dx) ** 2
                                                        + (dt[0].centroid[1] - dy) ** 2)[1], False
        # Step 2: nobody stood right on the click — take whoever stood closest next to
        # it, within a gate that grows with the time elapsed since the click.
        nearby = []
        for f in range(window):
            for d, t in per_frame.get(f, ()):
                distance = _distance_to_box((dx, dy), d.bbox)
                if distance <= CLICK_GATE_BASE + CLICK_GATE_GROWTH * f / max(fps, 1.0):
                    nearby.append((distance, f, t))
        if nearby:
            long_enough = [k for k in nearby if len(k[2]) >= SEED_MIN_LEN]
            return min(long_enough or nearby, key=lambda k: (k[0], k[1]))[2], False
        # Still nobody either way → fall back to the biggest mover, but report it. Not
        # with a box: then the spyglass follows the pointed-at skater from the box.
        click_missed = True
        if doel_kader is not None:
            return None, True
    movers = [t for t in tracklets if _path_length(t) >= MIN_MOVEMENT]
    candidates = movers or tracklets
    if not candidates:
        return None, click_missed
    return max(candidates, key=lambda t: float(np.median([d.area for d in t]))
                                         * max(_path_length(t), 1e-6)), click_missed


def _speed(dets):
    """Average centroid displacement per frame over the most recent detections."""
    dets = dets[-SPEED_WINDOW:]
    if len(dets) < 2:
        return (0.0, 0.0)
    dt = dets[-1].frame - dets[0].frame
    if dt <= 0:
        return (0.0, 0.0)
    return ((dets[-1].centroid[0] - dets[0].centroid[0]) / dt,
            (dets[-1].centroid[1] - dets[0].centroid[1]) / dt)


def _stitch_chain(seed, tracklets, fps):
    """
    String tracklets together into one target chain, forward and backward from the
    seed tracklet. A candidate only gets accepted if (a) its suit color matches the
    running reference (>= COLOR_MATCH_MIN) and (b) its joining position lies within the
    gate of the constant-velocity prediction across the gap. Returns (chain detections
    sorted by frame, ColorReference).

    Three finer points:
    - the color is measured on the **side of the candidate bordering the chain**
      (start when stitching forward, end when stitching backward) — that's where
      lighting and scale are most comparable;
    - a candidate may **overlap** the chain by a few frames (around an occlusion two
      IDs can coexist for a while, and with a strict "must start after the end" rule
      that gap would stay forever). Allowed as long as the candidate lines up on the
      shared frames with the chain (`STITCH_OVERLAP_GATE`);
    - a **short seed** (a fragment of 1-2 detections, quite possible after
      `_split_by_color`) gives a color reference of a single histogram; it first gets
      padded out on position (`_bootstrap`) before color starts serving as a
      gatekeeper.
    """
    max_gap = int(round(STITCH_MAX_GAP_S * fps))
    chain = list(seed)
    ref = ColorReference()
    for d in chain:
        ref.add(d.ref_hist)
    rest = [t for t in tracklets if t is not seed]

    def _color_sim(t, direction):
        """(similarity, certain) of candidate `t`'s chain-facing side. `certain` is
        False if there are only bbox-fallback histograms: those aren't comparable to a
        mask reference, so then the number means nothing."""
        edge = t[:10] if direction > 0 else t[-10:]
        sims = [s for s in (ref.sim(d.ref_hist) for d in edge) if s is not None]
        if sims:
            return float(np.median(sims)), True
        sims = [s for s in (ref.sim(d.hist) for d in edge) if s is not None]
        return (float(np.median(sims)), False) if sims else (None, False)

    def _candidates(direction):
        """[(tracklet, gap, joining detection)] for this direction.

        The joining point is the first detection **past the chain edge**, not blindly
        `t[0]`/`t[-1]`. That distinction matters. It used to require a candidate to lie
        almost entirely outside the chain (the since-retired `STITCH_MAX_OVERLAP`), and
        a tracklet that overlapped the edge would then fall out in *both* directions:
        forward its first frame lay too far back, backward it ended *after* the chain's
        start. Such a tracklet could then never join, even if it was demonstrably the
        same skater — and that costs coverage: on one clip the tracklets 23-38, 41-66,
        49-55 and 57-68 all four stayed outside the chain, together a gap of 34 frames.

        Simply allowing overlap isn't safe: there are tracklets that span the *entire*
        clip (stationary bystanders along the boards, `path` ~ 0.1). Hence a **real
        test on the shared frames**: if the candidate coincides with frames the chain
        already has, it must sit there at nearly the same spot (`STITCH_OVERLAP_GATE`).
        Two detections in the same frame are by definition two different boxes — if
        they sit on top of each other, it's the same skater who briefly got two IDs
        around an occlusion; if they sit apart, it's someone else and the candidate
        falls out. That's a sharper test than any prediction across a gap, since no
        extrapolation is involved.
        """
        on_frame = {}
        for d in chain:
            on_frame.setdefault(d.frame, d)
        edge = chain[-1].frame if direction > 0 else chain[0].frame
        out = []
        for t in rest:
            extends = t[-1].frame > edge if direction > 0 else t[0].frame < edge
            if not extends:
                continue
            shared = [(d, on_frame[d.frame]) for d in t if d.frame in on_frame]
            if shared:
                deviation = float(np.median([np.hypot(a.centroid[0] - b.centroid[0],
                                                a.centroid[1] - b.centroid[1])
                                       for a, b in shared]))
                if deviation > STITCH_OVERLAP_GATE:
                    continue
            d0 = next(d for d in (t if direction > 0 else reversed(t))
                      if (d.frame > edge if direction > 0 else d.frame < edge))
            g = (d0.frame - edge) if direction > 0 else (edge - d0.frame)
            if g <= max_gap:
                out.append((t, g, d0))
        return out

    def _predict(direction, g):
        """Position where the chain is expected after `g` frames (constant velocity)."""
        anchor = chain[-1] if direction > 0 else chain[0]
        v = _speed(chain) if direction > 0 else _speed(chain[:SPEED_WINDOW])
        return (anchor.centroid[0] + direction * v[0] * g,
                anchor.centroid[1] + direction * v[1] * g)

    def _distance(d0, direction, g):
        px, py = _predict(direction, g)
        return float(np.hypot(d0.centroid[0] - px, d0.centroid[1] - py))

    def _gate(g):
        return STITCH_GATE_BASIS + STITCH_GATE_GROWTH * max(g, 0)

    def _absorb(t, direction):
        rest.remove(t)
        chain.extend(t)
        chain.sort(key=lambda d: d.frame)
        for d in (t if direction > 0 else reversed(t)):
            ref.add(d.ref_hist)

    def _bootstrap():
        """Pad out a too-short seed with directly adjoining fragments, on position —
        the color reference is still too thin here to test anything against."""
        while len(chain) < SEED_MIN_LEN:
            choice = None
            for direction in (+1, -1):
                for t, g, d0 in _candidates(direction):
                    if g > BOOTSTRAP_MAX_GAP:
                        continue
                    dist = _distance(d0, direction, g)
                    if dist <= _gate(g) and (choice is None or dist < choice[0]):
                        choice = (dist, t, direction)
            if choice is None:
                return
            _absorb(choice[1], choice[2])

    def _try(direction):
        """direction=+1: stitch onward at the end; -1: before the start.

        Picks the **closest** usable candidate, not the best-scoring one. That's not a
        detail: the chain grows in small steps, and whoever jumps over an
        in-between tracklet makes that tracklet unreachable — it then sits in the
        middle of the chain and no longer extends it in either direction. The old
        score (`sim - 0.5*dist/gate`) divided the distance by a gate that grows *with*
        the gap, and thereby rewarded exactly the big jump: on the test clip, 66-182
        (gap 35, color 0.72, score 0.701) beat 41-66 (gap 10, color 0.71, score 0.642),
        after which 41-66 was permanently locked out of the chain and a 34-frame gap
        remained — too big for GAP_FILL_S, so 34 frames without a skeleton.

        Color and distance stay gates; on an equal gap the old score decides.
        """
        while True:
            best, best_key = None, None
            for t, g, d0 in _candidates(direction):
                sim, certain = _color_sim(t, direction)
                if certain and sim < COLOR_MATCH_MIN:
                    continue
                dist = _distance(d0, direction, g)
                # Without a usable color verdict, only position counts, and then we
                # really want the candidate close by.
                gate = _gate(g) * (1.0 if certain else 0.5)
                if dist > gate:
                    continue
                score = (sim if certain else COLOR_MATCH_MIN) - 0.5 * dist / gate
                key = (-g, score)          # smallest gap first, then quality
                if best_key is None or key > best_key:
                    best, best_key = t, key
            if best is None:
                return
            _absorb(best, direction)

    _bootstrap()
    _try(+1)
    _try(-1)
    _try(+1)          # after stitching backward, more may fit at the front or the back

    chain.sort(key=lambda d: d.frame)

    # Duplicate frames (overlapping tracklets): for each frame, keep the detection that
    # fits the reference best.
    per_frame = {}
    for d in chain:
        z = per_frame.get(d.frame)
        if z is None:
            per_frame[d.frame] = d
        else:
            sd, sz = ref.sim(d.ref_hist), ref.sim(z.ref_hist)
            if (sd or 0.0) > (sz or 0.0):
                per_frame[d.frame] = d
    return [per_frame[f] for f in sorted(per_frame)], ref


# ── Refinement pass ──────────────────────────────────────────────────────────────
def _interpolate_target(target_per_frame, fps, bocht=None):
    """
    Fill detection gaps <= GAP_FILL_S with linearly interpolated bboxes, so the
    refinement pass can still attempt an estimate there. Returns
    {frame: (bbox_norm_xyxy, real)} — `real` False for interpolated spots.

    `bocht` (per frame True/False) keeps corner stretches out of that fill: the gaps
    there were made on purpose by the detection pass, and refining them anyway would
    immediately spend the time saved on footage where nothing is measurable regardless.
    """
    plan = {}
    frames = sorted(target_per_frame)
    for f in frames:
        plan[f] = (target_per_frame[f].bbox, True)
    max_gap = int(round(GAP_FILL_S * fps))
    for a, b in zip(frames, frames[1:]):
        g = b - a
        if 1 < g <= max_gap:
            ba = np.array(target_per_frame[a].bbox)
            bb = np.array(target_per_frame[b].bbox)
            for f in range(a + 1, b):
                if bocht is not None and f < len(bocht) and bocht[f]:
                    continue
                t = (f - a) / g
                plan[f] = (tuple(ba + t * (bb - ba)), False)
    return plan


def _bbox_from_lm(lm, margin=SPYGLASS_MARGIN, min_vis=RTMPOSE_MIN_SCORE):
    """Normalized bbox (x0, y0, x1, y1) around the visible landmarks, with a border of
    `margin` × height; None if there's too little to see. This is the bbox the
    spyglass carries forward to the next frame."""
    pts = [(p.x, p.y) for p in lm if p.visibility >= min_vis]
    if len(pts) < 4:
        return None
    xs, ys = zip(*pts)
    height = max(ys) - min(ys)
    if height <= 0:
        return None
    border = margin * height
    return (max(0.0, min(xs) - border), max(0.0, min(ys) - border),
            min(1.0, max(xs) + border), min(1.0, max(ys) + border))


def _iou(a, b):
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _scale_bbox(bbox, factor):
    """The same bbox, `factor`x as big around its own center."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    hw, hh = (bbox[2] - bbox[0]) / 2 * factor, (bbox[3] - bbox[1]) / 2 * factor
    return (cx - hw, cy - hh, cx + hw, cy + hh)


class _Spyglass:
    """
    Follows the target skater through the frames where the chain doesn't have them, by
    propagating the bbox from frame to frame out of its own refined keypoints. Born for
    the skater too small for the detection pass: that only picks them up at roughly
    80-130 px height (1080p), while the refinement (RTMPose top-down on a bbox, scaled
    up to 288x384) works fine on such a skater — the problem is *finding* the bbox, not
    the keypoints. The user's drawn box supplies that bbox on the first frame; after
    that, every refined frame supplies the bbox for the next one.

    Pure state, no video and no model: the refinement comes in as a callable
    (`estimate`), so this is testable with synthetic landmarks (`_self_test_spyglass`).

    **Runs and their origin.** A run starts either on the box (`source='box'`) or on a
    chain frame (`'chain'`; `anchor` restarts the run on *every* chain frame, so the
    coast starts the next gap with a fresh velocity). Hits land in `pending` and only
    become final (`filled`) once the identity is settled:
    - if the run touches the chain (`anchor` on a chain frame), the **link test**
      decides: IoU of the predicted bbox with the chain bbox >= `SPYGLASS_LINK_IOU` →
      accepted, otherwise the run has drifted onto someone else and everything gets
      dropped;
    - if the run dies (`SPYGLASS_COAST_S` without a hit) or the video ends, the origin
      decides: a chain run gets accepted (the identity came from a verified chain
      point), a box run only if the chain is **not** demonstrably the same skater
      (`chain_linked=False`: no chain, or a chain that came from the biggest-mover
      fallback because the detection pass saw nobody in the box — then box and chain
      are two separate guesses about different stretches of the clip, and one is no
      better than the other). If the chain *was* found via the box, then a box run that
      doesn't reach it is suspicious: a hand-drawn box is not a proven identity, and
      without a joining point drift can't be ruled out.

    `box_outcome` records how the box run ended (`ok`, `reason`, `n`), so `analyze` can
    tell the user the spyglass lost the skater — that's the difference between "0
    pushes" and knowing why.

    **Plausibility per step** (`SPYGLASS_JUMP`/`SPYGLASS_GROWTH`): RTMPose always
    delivers a skeleton, even if someone else has moved into the bbox; if the center
    jumps more than half a body height, or the height changes by more than x1.5 in one
    frame, that's not a skater but a swap, and counts as a miss.
    """

    def __init__(self, fps, chain_linked):
        fps = fps or 30.0
        self.max_missed = max(1, int(round(SPYGLASS_COAST_S * fps)))
        self.chain_linked = chain_linked
        self.box_outcome = None     # {'ok', 'reason', 'n'} once the box run has closed
        self.bbox = None            # last known bbox (normalized); None = no run
        self.v = (0.0, 0.0)         # displacement of the center per frame
        self.last_f = None          # frame of the last hit or the last anchor
        self.missed = 0             # consecutive frames without a hit
        self.source = None          # 'box' | 'chain'
        self.first = False          # first step of a box run: scale sweep
        self.pending = {}           # frame → (lm, dev), waiting for the link test
        self.filled = {}            # frame → (lm, dev), final
        self.log = []               # lines for the log file
        self._hits = deque(maxlen=SPEED_WINDOW)   # (frame, cx, cy)

    @property
    def alive(self):
        return self.bbox is not None

    # ── state ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _center(bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _set(self, f, bbox):
        """New known position on frame `f`; updates the velocity."""
        cx, cy = self._center(bbox)
        self._hits.append((f, cx, cy))
        if len(self._hits) >= 2:
            f0, x0, y0 = self._hits[0]
            f1, x1, y1 = self._hits[-1]
            if f1 > f0:
                self.v = ((x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0))
        self.bbox = tuple(bbox)
        self.last_f = f
        self.missed = 0

    def _predict(self, f):
        """Where the bbox is expected on frame `f` (constant velocity since the last
        known position)."""
        dt = f - self.last_f
        cx, cy = self._center(self.bbox)
        cx, cy = cx + self.v[0] * dt, cy + self.v[1] * dt
        hw, hh = (self.bbox[2] - self.bbox[0]) / 2, (self.bbox[3] - self.bbox[1]) / 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def _close_run(self, ok, reason):
        n = len(self.pending)
        if self.source == 'box':
            self.box_outcome = {'ok': ok, 'reason': reason, 'n': n}
        if n:
            if ok:
                self.filled.update(self.pending)
            first_f, last_f = min(self.pending), max(self.pending)
            self.log.append(
                f"[spyglass] run from {self.source} frames {first_f}-{last_f} ({n} filled): "
                f"{'accepted' if ok else 'rejected'} — {reason}")
        self.pending = {}

    # ── events ─────────────────────────────────────────────────────────────────
    def start_box(self, bbox, f=0):
        """Start the run on the drawn box."""
        self._hits.clear()
        self.v = (0.0, 0.0)
        self._set(f, bbox)
        self.source, self.first, self.pending = 'box', True, {}

    def anchor(self, f, bbox):
        """The chain has the skater on frame `f`: link test for a pending run, then
        restart from this verified point."""
        if self.alive and self.pending:
            predicted = self._predict(f)
            iou = _iou(predicted, bbox)
            ok = iou >= SPYGLASS_LINK_IOU
            self._close_run(ok, f"link at frame {f}: IoU {iou:.2f}")
            if not ok:
                self._hits.clear()  # that was someone else: their velocity too
        elif self.alive:
            self.pending = {}
        self._set(f, bbox)
        self.source, self.first = 'chain', False

    def follow(self, f, bbox):
        """A frame pass 2 already filled itself (short gap, interpolated): only take
        the position along, so the coast later starts from the right spot."""
        if self.alive:
            self._set(f, bbox)

    def step(self, f, estimate):
        """
        One frame without the chain. `estimate(bbox) → (lm, dev, score, hist)` is the
        refinement on that bbox (lm None = rejected by the score or color gate).
        Returns the accepted `(lm, dev, hist)`, or None.
        """
        if not self.alive:
            return None
        expected = self._predict(f)
        # First step of a box run: a hand rarely draws tight, so try three scalings
        # and take the surest one.
        candidates = ([_scale_bbox(expected, k) for k in SPYGLASS_START_SCALES]
                      if self.first else [expected])
        best = None
        for bb in candidates:
            out = estimate(bb)
            if out is None or out[0] is None:
                continue
            lm, dev, score, hist = out
            new_box = _bbox_from_lm(lm)
            if new_box is None:
                continue
            if not self.first and not self._plausible(expected, new_box):
                continue
            if best is None or score > best[0]:
                best = (score, lm, dev, hist, new_box)
        if best is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self._die(f)
            return None
        _, lm, dev, hist, new_box = best
        self.first = False
        self._set(f, new_box)
        self.pending[f] = (lm, dev)
        return lm, dev, hist

    def _plausible(self, expected, new_box):
        h_v = expected[3] - expected[1]
        h_n = new_box[3] - new_box[1]
        if h_v <= 0 or h_n <= 0:
            return False
        (vx, vy), (nx, ny) = self._center(expected), self._center(new_box)
        if float(np.hypot(nx - vx, ny - vy)) > SPYGLASS_JUMP * h_v:
            return False
        growth = h_n / h_v
        return 1.0 / SPYGLASS_GROWTH <= growth <= SPYGLASS_GROWTH

    def _die(self, f):
        ok = self.source == 'chain' or not self.chain_linked
        self._close_run(ok, f"run stopped on frame {f} after {self.missed} frames without "
                            f"a hit" + ("" if ok else "; box run without a link"))
        self.bbox = None
        self._hits.clear()          # a later anchor starts with a fresh velocity

    def close(self):
        """End of the video: settle whatever's still pending, based on origin."""
        if self.alive:
            ok = self.source == 'chain' or not self.chain_linked
            self._close_run(ok, "end of video" + ("" if ok else "; box run without a link"))
            self.bbox = None


def _rtmpose_model():
    r"""The shipped RTMPose file if present, otherwise the URL — after which rtmlib
    downloads and caches it itself in %USERPROFILE%\.cache\rtmlib."""
    pad = os.path.join(app_dir(), RTMPOSE_LOCAL)
    return pad if os.path.exists(pad) else RTMPOSE_MODEL


def _make_rtmpose(waarschuwing_callback=None):
    """
    RTMPose-26 model for the refinement pass, or None without rtmlib.

    The CPU fallback here isn't a luxury: `rtmpose_device()` reads off whether
    ONNXRuntime got **compiled with** a CUDA or DirectML provider, which is different
    from whether the matching DLLs actually load on this machine. If they don't,
    building the session fails right there — and that shouldn't cost an analysis that
    would otherwise have run fine on the CPU.
    """
    if not IS_RTMPOSE:
        return None
    device = rtmpose_device()
    model = _rtmpose_model()     # decide once: both branches use the same weights file
    try:
        return _RTMPose(model, model_input_size=RTMPOSE_INPUT,
                        backend='onnxruntime', device=device)
    except Exception as exc:
        if device == 'cpu':
            raise
        _notify(waarschuwing_callback,
              f"RTMPose couldn't start on the GPU ({exc}); the refinement pass "
              "runs on the CPU.")
        return _RTMPose(model, model_input_size=RTMPOSE_INPUT,
                        backend='onnxruntime', device='cpu')


def _refine_landmarks(input_pad, model, info, target_per_frame, ref,
                       progress_callback=None, rtmpose=None, bocht=None,
                       deinterlacen=False, spyglass=None):
    """
    Pass 2: read the video again and re-estimate the pose for each target frame, now
    with the skater filling the inference image → substantially more accurate
    keypoints than in the full-frame pass. With `rtmpose` that goes top-down on the
    target bbox (RTMPose-26: subpixel decoding + real heel/toe); otherwise via a square
    crop through the YOLO model. The suit color (reference `ref`) guards in both routes
    that another person never quietly gets picked up.
    Returns `(out, devs, filled)`: {frame: lm} with refined (or restored) landmarks,
    {frame: midline-dev}, and the frames the spyglass added.

    `bocht` (per frame True/False) keeps the gap-filling out of the corner stretches;
    see `_interpolate_target`.

    `spyglass` (a `_Spyglass`, already started on the drawn box) fills the frames
    neither the chain nor the interpolation covers: the run-up before the first
    detection, gaps longer than `GAP_FILL_S`, the tail, and the stretches the corner
    guard skipped. It deliberately ignores `bocht`: the propagation is cheap, and
    whether a filled frame is a corner gets decided afterwards by the ratio
    (`determine_corner_sequence`). On chain frames the spyglass gets anchored (link
    test + fresh start), on interpolated frames it's only updated; see `_Spyglass`.
    """
    plan = _interpolate_target(target_per_frame, info.fps, bocht)
    w, h = info.w, info.h
    out, devs = {}, {}

    def refine(frame, bbox, gate):
        """→ (lm, dev, score, hist); lm None if the estimate was rejected."""
        if rtmpose is not None:
            return _refine_rtmpose(rtmpose, frame, bbox, gate, ref, w, h)
        lm = _refine_yolo_crop(model, frame, bbox, ref, w, h)
        return lm, None, (1.0 if lm is not None else 0.0), None

    cap = open_video(input_pad, deinterlacen)
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f in plan:
            bbox, real = plan[f]
            lm, dev, _, _ = refine(frame, bbox, 'veto' if real else 'match')
            if dev is not None:
                devs[f] = dev
            if lm is not None:
                out[f] = lm
            if spyglass is not None:
                # The chain bbox if the refinement fell through: even then this is a
                # verified point (pass-1 detection) and the coast should start from
                # here.
                known = (_bbox_from_lm(lm) if lm is not None else None) or bbox
                if real:
                    spyglass.anchor(f, known)
                elif lm is not None:
                    spyglass.follow(f, known)
        elif spyglass is not None and spyglass.alive:
            hit = spyglass.step(f, lambda bb, _fr=frame: refine(_fr, bb, 'veto'))
            if hit is not None and not target_per_frame:
                # Spyglass-only (no chain): the reference has to come from the
                # spyglass's own hits, otherwise the color gate never has anything to
                # test against.
                ref.add(hit[2])
        f += 1
        if progress_callback is not None:
            progress_callback(f, info.totaal)
    cap.release()

    filled = set()
    if spyglass is not None:
        spyglass.close()
        for f2, (lm, dev) in spyglass.filled.items():
            out[f2] = lm
            if dev is not None:
                devs[f2] = dev
        filled = set(spyglass.filled)
        for line in spyglass.log:
            print(line)
        print(f"[spyglass] {len(filled)} frames added")
    return out, devs, filled


# Halpe26 indices of the leg keypoints (hips through ankles) for the quality check.
_HALPE_LEGS = (11, 12, 13, 14, 15, 16)

# ── Midline quality flag ─────────────────────────────────────────────────────────
# Filmed frontally, a joint should sit horizontally in the middle of the leg. We
# measure, per knee, the deviation from the leg's midline in a color mask. The color
# comes from a **thigh self-sample of the same leg in the same frame** (not from the
# torso reference: a suit is regularly two-toned — white torso, black pants — but
# thigh and knee are always the same fabric, under the same lighting). Purely a
# quality flag: a large deviation = a shaky frame. Deliberately no automatic
# correction until it's measured that one would improve the angles.
MIDLINE_ROWS       = 5      # number of image rows around the knee y over which is averaged
MIDLINE_BP_THRESHOLD = 0.2  # mask threshold as a fraction of the back-projection maximum
MIDLINE_RUN_MIN    = 0.08   # min. run width as a fraction of the tibia length (noise)
MIDLINE_RUN_MAX    = 0.8    # max. run width — wider = legs/arm merged: skip
MIDLINE_MIN_ROWS   = 0.6    # min. fraction of usable rows for a valid measurement


def _leg_hist(frame, hip_xy, knee_xy, tibia_len):
    """HSV histogram (0-255 normalized, for back-projection) of a small block in the
    middle of the upper leg — the color of *this* leg's trouser fabric."""
    mid_x = (float(hip_xy[0]) + float(knee_xy[0])) / 2
    mid_y = (float(hip_xy[1]) + float(knee_xy[1])) / 2
    half = int(np.clip(0.12 * tibia_len, 4, 30))
    x0, y0, x1, y1 = int(mid_x - half), int(mid_y - half), int(mid_x + half), int(mid_y + half)
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0] or x1 - x0 < 4:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(COLOR_BINS),
                        [0, 180, 0, 256, 0, 256])
    # 0-255 instead of L1: calcBackProject returns uint8 — with an L1 histogram
    # (values << 1) everything truncates to 0 and the mask is always empty.
    return cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)


def _midline_deviation(frame, leg_hist, knee_xy, tibia_len):
    """
    Horizontal deviation (px, signed: midline - keypoint) of one knee point relative
    to the middle of the leg in the color mask, or None if the measurement doesn't
    work out (leg not free-standing, color unclear, frame edge).
    """
    if leg_hist is None or tibia_len < 12:
        return None
    h, w = frame.shape[:2]
    kx, ky = float(knee_xy[0]), float(knee_xy[1])
    half_b = int(np.clip(0.7 * tibia_len, 12, 90))
    half_r = MIDLINE_ROWS // 2
    x0, x1 = int(kx - half_b), int(kx + half_b + 1)
    y0, y1 = int(ky - half_r), int(ky + half_r + 1)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    bp = cv2.calcBackProject([hsv], [0, 1, 2], leg_hist,
                             [0, 180, 0, 256, 0, 256], scale=1)
    peak = float(bp.max())
    if peak <= 0:
        return None
    mask = bp >= MIDLINE_BP_THRESHOLD * peak

    centers = []
    kx_local = kx - x0
    for row in mask:
        # The contiguous run of suit pixels that contains the knee point.
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            continue
        # runs = groups of consecutive indices
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        run = next((rn for rn in runs if rn[0] - 2 <= kx_local <= rn[-1] + 2), None)
        if run is None:
            continue
        width = run[-1] - run[0] + 1
        if not (MIDLINE_RUN_MIN * tibia_len <= width <= MIDLINE_RUN_MAX * tibia_len):
            continue
        centers.append((run[0] + run[-1]) / 2.0)
    if len(centers) < MIDLINE_MIN_ROWS * mask.shape[0]:
        return None
    return round(float(np.median(centers) - kx_local), 1)


def _refine_rtmpose(rtmpose, frame, bbox, gate, ref, w, h):
    """
    Top-down refinement of one frame: RTMPose-26 on the target bbox (pixels). The
    color gate keeps the other skater out, in two modes:
    - `'veto'` — only a clearly different suit (< `COLOR_SPLIT_MIN`) gets rejected.
      For a real detection (pass-1 safety net: the landmarks stay as they were) and
      for the spyglass, where continuity, the score gate, plausibility and the link
      test are the safety net and the torso of a 90 px skater (~15x25 px) is too
      noisy for a positive threshold;
    - `'match'` — an interpolated gap frame requires a positive color match instead
      (>= `COLOR_MATCH_MIN`), because there's no other safety net there.
    Returns `(lm | None, midline-dev-dict | None, leg-score, mask-hist | None)`.
    """
    bbox_px = (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)
    kps, scores = rtmpose(frame, [list(bbox_px)])
    kp, sc = kps[0], scores[0]

    score = float(sc[list(_HALPE_LEGS)].mean())
    if score < RTMPOSE_MIN_SCORE:
        return None, None, score, None     # legs not seen (occlusion): don't trust it
    hist, from_mask = _torso_hist(frame, kp, sc, bbox_px)
    sim = ref.sim(hist)
    if gate == 'veto':
        # Only a mask histogram may reject a refinement: the bbox fallback contains
        # background and scores low even for the correct skater.
        if from_mask and sim is not None and sim < COLOR_SPLIT_MIN:
            return None, None, score, None     # clearly a different suit in the bbox
    else:
        if sim is None or sim < COLOR_MATCH_MIN:
            return None, None, score, None     # gap frame: only fill on a sure match

    # Quality flag: knee relative to the leg's midline (Halpe: 11/13/15 = L hip/knee/
    # ankle, 12/14/16 = R). Measurement only, no correction.
    dev = {}
    for name, h_i, k_i, e_i in (('l_knee', 11, 13, 15), ('r_knee', 12, 14, 16)):
        dev[name] = None
        if min(sc[h_i], sc[k_i], sc[e_i]) >= RTMPOSE_MIN_SCORE:
            tibia = float(np.hypot(*(kp[k_i] - kp[e_i])))
            leg_hist = _leg_hist(frame, kp[h_i], kp[k_i], tibia)
            dev[name] = _midline_deviation(frame, leg_hist, kp[k_i], tibia)
    if dev['l_knee'] is None and dev['r_knee'] is None:
        dev = None
    return _halpe26_to_landmarks(kp, sc, w, h), dev, score, (hist if from_mask else None)


def _refine_yolo_crop(model, frame, bbox, ref, w, h):
    """Fallback route without rtmlib: square crop around the bbox through the YOLO
    model."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    crop_side = int(max(REFINE_MIN_PX, REFINE_MARGIN * side * max(w, h)))
    crop_side = min(crop_side, min(w, h))
    x0 = int(np.clip(cx * w - crop_side / 2, 0, w - crop_side))
    y0 = int(np.clip(cy * h - crop_side / 2, 0, h - crop_side))
    crop = frame[y0:y0 + crop_side, x0:x0 + crop_side]
    res = _infer(lambda dev: model.predict(
        crop, imgsz=REFINE_IMGSZ, classes=[0], verbose=False, device=dev))[0]
    choice = _choose_in_crop(res, crop, (cx * w - x0, cy * h - y0), crop_side, ref)
    if choice is None:
        return None
    kp_xy, kp_conf = choice
    return _coco_to_landmarks(kp_xy + [x0, y0], kp_conf, w, h)


def _choose_in_crop(res, crop, expected_xy, crop_side, ref):
    """
    In the crop result, choose the person who is the target: suit color must match the
    reference; among several matches the one closest to the expected position wins.
    Returns (kp_xy, kp_conf) in crop coordinates, or None.
    """
    kps, boxes = res.keypoints, res.boxes
    if kps is None or boxes is None or kps.xy is None or len(boxes) == 0:
        return None
    xy = kps.xy.cpu().numpy()
    conf = (kps.conf.cpu().numpy() if kps.conf is not None
            else np.ones(xy.shape[:2], dtype=float))
    xywh = boxes.xywh.cpu().numpy()

    candidates = []
    for i in range(len(xy)):
        cx, cy, bw, bh = xywh[i]
        hist, from_mask = _torso_hist(crop, xy[i], conf[i],
                                       (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        # Without a mask, the similarity isn't comparable to the reference; then it
        # counts as "unknown" (None) instead of an artificially low score.
        sim = ref.sim(hist) if from_mask else None
        dist = float(np.hypot(cx - expected_xy[0], cy - expected_xy[1])) / crop_side
        candidates.append((i, sim, dist))

    matching = [k for k in candidates if k[1] is not None and k[1] >= COLOR_MATCH_MIN]
    if matching:
        i = min(matching, key=lambda k: k[2])[0]
        return xy[i], conf[i]
    # No color match (e.g. torso partly out of frame): only accept a candidate that
    # sits almost exactly at the expected spot — otherwise no refinement at all.
    nearby = [k for k in candidates if k[2] <= 0.12 and (k[1] is None or k[1] >= COLOR_SPLIT_MIN)]
    if nearby:
        i = min(nearby, key=lambda k: k[2])[0]
        return xy[i], conf[i]
    return None


def _corner_with_check_frames(resultaten, excluded, info):
    """
    Classify the corner (`determine_corner_sequence`) and then enforce that a frame
    the detection pass didn't actually analyze never yields a measurement either.

    That last part is about the **check frames**: in skip mode, the guard runs
    inference on one frame every `CORNER_CHECK_S` to see whether the straight section
    has started again. That frame does get a skeleton, but it's a glance, not a
    measurement — the neighboring frames that would need to prove a push were just
    skipped. Its verdict does count (it may end the corner — that's what it's for);
    only its own angle doesn't.

    Without this rule, such a frame could clear itself on its own hip stance, and in
    the middle of a corner a skater does occasionally turn briefly almost frontal
    (visible in the Ellia and Fran clips). If a check frame really does end the
    corner, the detection pass runs at full speed again afterwards and those frames
    get measured normally — at most this one frame at the start of the next stance run
    is lost, and a run truncated at its start counts anyway (see
    `determine_push_from_extension`).
    """
    determine_corner_sequence(resultaten, info.w, info.h, info.fps)
    for r, out in zip(resultaten, excluded):
        if out:
            r.corner = True


# ── Main API ──────────────────────────────────────────────────────────────────────
def analyze(input_pad, model_pad=None, smooth_n=5, threshold=0.015, force_fps=None,
              num_poses=NUM_POSES_DEFAULT, doel_punt=None, smooth_landmarks=True,
              progress_callback=None, yolo_model=None, horizon_deg=0.0,
              auto_horizon=False, refine=True, perspectief=None,
              waarschuwing_callback=None, bocht=True, deinterlacen=None,
              doel_kader=None):
    """
    Full analysis via YOLO-pose + ByteTrack + offline target selection + crop
    refinement. Signature-compatible with skate_analysis.analyze() (`model_pad` — the
    MediaPipe .task — and `num_poses` are ignored; YOLO always detects everyone).
    Returns (VideoInfo, list[FrameResult]).

    `doel_kader` (normalized (x0, y0, x1, y1) around the skater on the first frame)
    turns on the **spyglass**: the skater also gets followed where the detection pass
    doesn't see them — too small, far away — by propagating the bbox from the box from
    frame to frame in the refinement pass (see `_Spyglass`). Its center point serves as
    `doel_punt` if that isn't given separately. Without a box the behavior is
    byte-for-byte as before: the spyglass is deliberately opt-in, because it would
    otherwise also fill the gaps and skipped corner stretches of every analysis, and
    that's a measurement change.

    `perspectief` (PerspectiveConfig) works identically to the MediaPipe backend: the
    shared steps `set_horizon`/`process_derivatives` do all the work.

    `waarschuwing_callback(tekst)` receives messages about silent fallbacks in target
    selection (currently: a mouse click that hit nobody). If target selection fails
    entirely, that's not a warning but an error — then a RuntimeError follows instead
    of an empty analysis that looks like a valid result.

    With `bocht` (default) the detection pass mostly skips the corner (`_CornerGuard`)
    and corner frames yield no push measurement. Here that's mainly a speed measure:
    the detection pass is the bulk of the analysis time, and in the corner there's
    nothing to measure anyway. With `bocht=False`, every frame gets inferred and
    measured, as before.
    """
    info = video_info(input_pad, force_fps)
    # None = figure it out ourselves (CLI convenience); the GUI determines it in the
    # dialog and passes an explicit bool, so the choice is visible and ends up in the
    # settings. *Both* passes must see the same pixels: if pass 2 refines on woven
    # footage while pass 1 was filtered, you're measuring two different videos through
    # each other.
    if deinterlacen is None:
        deinterlacen = is_interlaced(input_pad)
    model = _load_yolo(yolo_model or os.path.join(app_dir(), DEFAULT_YOLO_MODEL),
                       waarschuwing_callback)
    if perspectief is not None:
        auto_horizon = False     # a fixed camera is assumed; the calibration already knows the tilt
    if doel_kader is not None:
        doel_kader = tuple(float(v) for v in doel_kader)
        if doel_punt is None:
            doel_punt = ((doel_kader[0] + doel_kader[2]) / 2,
                         (doel_kader[1] + doel_kader[3]) / 2)
        kader_px = (doel_kader[3] - doel_kader[1]) * info.h
        if kader_px < BOX_MIN_HEIGHT_PX:
            # Not a block — maybe it grows fast enough — but it should say why the
            # result will be (nearly) empty, since otherwise that isn't visible.
            _notify(waarschuwing_callback,
                  f"The box you drew is only {kader_px:.0f} px tall. Below roughly "
                  f"{BOX_MIN_HEIGHT_PX} px, even the refinement no longer sees legs "
                  f"(the legs are then about 15 px), so there's nothing to measure "
                  f"there either — not even with the spyglass. Start the fragment "
                  f"later, at the point where the skater is bigger in frame.")

    # Multiple passes → one continuous progress bar via phase slices.
    n_phases = 1 + (1 if refine else 0) + (1 if auto_horizon else 0)
    phase = 0
    det_cb = phase_progress(progress_callback, phase, n_phases); phase += 1
    ver_cb = phase_progress(progress_callback, phase, n_phases) if refine else None
    phase += 1 if refine else 0
    hor_cb = phase_progress(progress_callback, phase, n_phases) if auto_horizon else None

    # Pass 1: collect all detections (high resolution). In the corner, the guard skips
    # frames, and the frames it still does infer there are only check frames; both end
    # up in `excluded` and flow into the rest of the pipeline that way.
    frames, excluded = _detect_all(input_pad, model, info,
                                             progress_callback=det_cb, bocht=bocht,
                                             waarschuwing_callback=waarschuwing_callback,
                                             deinterlacen=deinterlacen)
    n_frames = len(frames)

    # Offline target selection: tracklets → color splits → seed → stitching.
    tracklets = []
    for t in _build_tracklets(frames):
        tracklets.extend(_split_by_color(t))
    seed, click_missed = _choose_seed(tracklets, frames, doel_punt, info.fps, doel_kader)
    fallback = False
    if seed is None and doel_kader is not None:
        # The detection pass saw nobody in the box. Then follow the biggest mover
        # anyway (exactly what a click does in that case), *alongside* the spyglass
        # from the box: the box may never come out worse than a click. On
        # `00000 16-14` (skater 35-57 px, detected only from frame 187 of 200),
        # spyglass-only yielded one frame, while the fallback grabs the thirteen real
        # detections at the end.
        seed, _ = _choose_seed(tracklets, frames, None, info.fps)
        fallback = seed is not None
    if seed is None and doel_kader is None:
        # Without a seed, target_per_frame stays empty and the rest of the pipeline just
        # runs through: not a single frame gets a pose, and the GUI reports "0 pushes"
        # as if that were a measurement. Better to fail hard with an understandable
        # reason.
        raise RuntimeError(
            "No skater found to track: detection didn't yield a single person in this "
            "video. Check that the skater is in frame and that the video is readable "
            "(codec/resolution), and consider trying a different clip.")
    if click_missed and waarschuwing_callback is not None:
        if doel_kader is not None:
            followup = (f"The spyglass follows the skater from the box; the biggest "
                       f"mover in frame has also been followed alongside it (from "
                       f"frame {seed[0].frame}) — check whether that's the same "
                       f"skater."
                       if fallback else
                       "The skater has now only been followed with the spyglass from "
                       "the box; check the result.")
            waarschuwing_callback(
                f"The box you drew couldn't be linked to a detection — the detection "
                f"pass didn't see anyone that size there in the first "
                f"{CLICK_SEARCH_S:.0f} seconds. " + followup)
        else:
            waarschuwing_callback(
                f"Your click couldn't be linked to a skater — nobody was detected "
                f"there in the first {CLICK_SEARCH_S:.0f} seconds, not even right next "
                f"to it. The biggest mover in frame has now been followed; check "
                f"whether that's the intended skater.")

    if seed is not None:
        chain, ref = _stitch_chain(seed, tracklets, info.fps)
    else:
        chain, ref = [], ColorReference()      # spyglass-only: everything comes from the box
    target_per_frame = {d.frame: d for d in chain}
    spyglass = None
    if doel_kader is not None and refine:
        # The chain only counts as "linked" if it was found via the box itself; a
        # fallback chain is a second guess and may not disqualify a box run.
        spyglass = _Spyglass(info.fps, chain_linked=bool(target_per_frame) and not fallback)
        spyglass.start_box(doel_kader, 0)

    resultaten = []
    for f in range(n_frames):
        r = FrameResult(frame_nr=f, time=f / info.fps if info.fps > 0 else 0)
        r.corner = bool(excluded[f])
        if f in target_per_frame:
            r.lm = target_per_frame[f].lm
            r.pose_found = True
        resultaten.append(r)

    # Determine the corner on the *raw* pass-1 landmarks, before refinement: then that
    # refinement doesn't have to run over the corner at all. The check frames the
    # detection pass did infer in the corner give the verdict there; if they say the
    # straight section is already back, the detection pass runs at full speed again
    # afterwards and those frames get measured normally.
    if bocht:
        _corner_with_check_frames(resultaten, excluded, info)
        corner_per_frame = [r.corner for r in resultaten]
    else:
        corner_per_frame = None

    # Pass 2: refinement (more accurate keypoints + fill gaps), top-down with
    # RTMPose-26 if rtmlib is available, otherwise the older YOLO-crop route. With a
    # box, the spyglass runs alongside and fills the frames the chain doesn't have.
    if refine and (target_per_frame or spyglass is not None):
        refined, devs, filled = _refine_landmarks(
            input_pad, model, info, target_per_frame, ref, progress_callback=ver_cb,
            rtmpose=_make_rtmpose(waarschuwing_callback), bocht=corner_per_frame,
            deinterlacen=deinterlacen, spyglass=spyglass)
    else:
        refined, devs, filled = {}, {}, set()
    if spyglass is not None and not target_per_frame and not filled:
        raise RuntimeError(
            "No skater found to track: detection saw nobody in the drawn box, and the "
            "spyglass couldn't follow a skater there from the box either. Draw the box "
            "tightly around the skater on the first frame (zoom in with the mouse "
            "wheel) and check that they're really in frame there.")
    outcome = spyglass.box_outcome if spyglass is not None else None
    if outcome is not None and 'link' not in outcome['reason']:
        # The box run didn't reach the chain: it died, or the video ended. That can be
        # perfectly fine (spyglass-only to the end), but usually it means the skater
        # was too small or too unclear — and the user should hear that instead of
        # having to guess it from "0 pushes".
        _notify(waarschuwing_callback,
              f"The spyglass could only follow the skater from the box for "
              f"{outcome['n']} frame(s) ({outcome['reason']}). If it loses them this "
              f"quickly, they're too small or too unclear in frame for the "
              f"refinement.")
    # A frame the spyglass filled *was* analyzed, even if the corner guard had skipped
    # it in pass 1: the second corner classification below judges it on its own hip
    # stance then, like any other frame with a skeleton.
    for f in filled:
        if f < len(excluded):
            excluded[f] = False

    for f, r in enumerate(resultaten):
        lm = refined.get(f)
        if lm is not None:
            r.lm = lm
            r.pose_found = True
            r.midline_dev = devs.get(f)

    if smooth_landmarks:
        smooth_landmarks_offline(resultaten, info.w, info.h, fps=info.fps)
    if bocht:
        # Once more, now on the refined landmarks: the frames pass 2 added get their
        # own verdict this way too.
        _corner_with_check_frames(resultaten, excluded, info)
    set_horizon(resultaten, input_pad, info, horizon_deg, auto_horizon, force_fps, hor_cb,
                perspectief=perspectief, deinterlacen=deinterlacen)
    process_derivatives(resultaten, info.w, info.h, info.fps, smooth_n, threshold,
                       perspectief=perspectief)
    return info, resultaten


# ── Self-test ──────────────────────────────────────────────────────────────────
def _self_test_corner_guard():
    """
    Tests `_CornerGuard`'s state machine on synthetic detections — no video, no model,
    so it runs in about a second. This is the kind of place where a bug stays silent:
    skipping too little only costs time, but skipping too *much* removes frames from
    the measurement. Run with `python skate_yolo.py`.
    """
    W, H, FPS = 1000, 1000, 30.0

    def _pose(ratio, torso, cx):
        """Landmark list with exactly this hip-width/torso-length ratio."""
        lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
        hb = ratio * torso
        lm[11] = Landmark(cx - 0.05, 0.5 - torso / H / 2, 0, 1.0)
        lm[12] = Landmark(cx + 0.05, 0.5 - torso / H / 2, 0, 1.0)
        lm[23] = Landmark(cx - hb / W / 2, 0.5 + torso / H / 2, 0, 1.0)
        lm[24] = Landmark(cx + hb / W / 2, 0.5 + torso / H / 2, 0, 1.0)
        return lm

    def _det(ratio, tid=1, cx=0.5, torso=100):
        return Detection(frame=0, tid=tid, centroid=(cx, 0.5),
                        bbox=(cx - 0.05, 0.3, cx + 0.05, 0.7), area=0.04,
                        lm=_pose(ratio, torso, cx))

    def _loop(n, make_dets, on=True):
        """Returns (number analyzed, number skipped, frame where it went back to full
        speed after the last skip)."""
        guard = _CornerGuard(FPS, W, H, on=on)
        analyzed = skipped = 0
        for f in range(n):
            if not guard.should_infer(f):
                skipped += 1
                continue
            analyzed += 1
            guard.feed(f, make_dets(f))
        return analyzed, skipped

    # 1. A frontal, growing skater: never skip (they're coming straight at the camera,
    #    so they barely move in frame — growth has to save them).
    _, skipped = _loop(200, lambda f: [_det(0.9, torso=100 + f * 0.6)])
    assert skipped == 0, f"frontal skater got {skipped} frames skipped"

    # 2. Corner: the vast majority should get skipped.
    an, _ = _loop(300, lambda f: [_det(0.15)])
    assert an < 60, f"corner: still {an} of 300 frames analyzed"

    # 3. Back on the straight section, it gets picked up within one check interval.
    guard = _CornerGuard(FPS, W, H)
    resumed = None
    for f in range(400):
        if not guard.should_infer(f):
            continue
        guard.feed(f, [_det(0.15 if f < 200 else 0.9, torso=100 + max(0, f - 200) * 0.6)])
        if f >= 200 and resumed is None:
            resumed = f
    assert resumed is not None and resumed - 200 <= guard.check, f"resumed only on frame {resumed}"

    # 4. A stationary bystander stands frontal in frame, but shouldn't keep the corner
    #    open.
    an, _ = _loop(300, lambda f: [_det(0.9, tid=7, cx=0.2), _det(0.15, tid=1, cx=0.6)])
    assert an < 120, f"bystander held the analysis at full speed for {an} of 300 frames"

    # 5. A 2 s blur gap on a straight section is NOT a corner (CORNER_STILL_S = 3 s).
    _, skipped = _loop(300, lambda f: [] if 100 <= f < 160 else [_det(0.9, torso=100 + f * 0.6)])
    assert skipped == 0, f"blur gap caused {skipped} skipped frames"

    # 6. Turned off = analyze everything, corner included.
    _, skipped = _loop(200, lambda f: [_det(0.1)], on=False)
    assert skipped == 0, f"with bocht=False, {skipped} frames still got skipped"

    # 7. A check frame is a glance, not a measurement — even if the skater happens to
    #    be frontal on it (that happens: in the middle of a corner they occasionally
    #    turn briefly almost frontal). We replay the corner, let one check frame look
    #    frontal, and require that frame to get no derivatives.
    guard = _CornerGuard(FPS, W, H)
    resultaten, excluded, frontal_at = [], [], None
    for f in range(300):
        r = FrameResult(frame_nr=f, time=f / FPS)
        check = guard.skip                      # do we only infer this frame as a check?
        if not guard.should_infer(f):
            excluded.append(True)             # skipped: no inference
        else:
            excluded.append(check)
            r.pose_found = True
            # One check frame halfway gets a frontal hip stance.
            if check and frontal_at is None and f > 150:
                r.lm, frontal_at = _pose(1.0, 100, 0.5), f
            else:
                r.lm = _pose(0.15, 100, 0.5)
            guard.feed(f, [_det(0.15)])
        resultaten.append(r)

    assert frontal_at is not None, "no check frame to test"
    _corner_with_check_frames(resultaten, excluded, VideoInfo(W, H, FPS, len(resultaten)))
    assert resultaten[frontal_at].corner, (
        f"check frame {frontal_at} cleared itself on its own hip stance")
    process_derivatives(resultaten, W, H, FPS)
    assert all(r.lm_data is None for r in resultaten), \
        "a frame in the corner still produced a measurement"

    print("Self-test _CornerGuard OK")


def _self_test_seed():
    """
    Tests `_choose_seed` on synthetic tracklets — no video, no model. The case that
    matters is the skater who's still too far away to be detected on the click frame:
    then the click still has to land on *them* and not silently on the biggest mover.
    The numbers below are those of `00005 8-41`, the clip where this went wrong. Run
    with `python skate_yolo.py`.
    """
    FPS = 25.0
    CLICK = (0.391, 0.212)

    def _tracklet(tid, start, n, x0, y0, sway=0.0, drift=(0.0, 0.0),
                  width=0.03, height=0.09):
        """Skater swaying around (x0, y0) (one stroke per 25 frames) and slowly
        drifting away — the motion of a skater approaching frontally."""
        dets = []
        for k in range(n):
            x = x0 + sway * float(np.sin(2 * np.pi * k / 25)) + drift[0] * k
            y = y0 + drift[1] * k
            dets.append(Detection(frame=start + k, tid=tid, centroid=(x, y),
                                 bbox=(x - width / 2, y - height / 2,
                                       x + width / 2, y + height / 2),
                                 area=width * height,
                                 lm=[Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]))
        return dets

    N = 281
    frames = [[] for _ in range(N)]
    # The clicked skater: only big enough to be detected from frame 84 (3.4 s), then
    # right next to the click, swaying with every stroke. Plus a bigger skater who is
    # seen from frame 0 onward — the old fallback picked *that one*.
    target = _tracklet(1, 84, 197, 0.40, 0.23, sway=0.06, drift=(0.0, 0.0007))
    other = _tracklet(2, 0, 200, 0.75, 0.55, sway=0.05, drift=(-0.001, 0.0005),
                      width=0.10, height=0.30)
    for t in (target, other):
        for d in t:
            frames[d.frame].append(d)

    # 1. The click belongs to the skater who only gets detected seconds later.
    seed, missed = _choose_seed([target, other], frames, CLICK, FPS)
    assert not missed, "a click on a not-yet-detected skater counted as missed"
    assert seed is target, "the click landed on the wrong skater"

    # 2. A click on nobody stays a missed click: better the biggest mover with a
    #    warning than an arbitrary skater passed off as 'your click'.
    seed, missed = _choose_seed([target, other], frames, (0.03, 0.95), FPS)
    assert missed and seed is other, "a click into empty space didn't get a clean fallback"

    # 3. If someone *does* stand on the click, they win outright — the next-to-the-
    #    click search must never overrule a direct hit.
    seed, missed = _choose_seed([target, other], frames, (0.75, 0.55), FPS)
    assert not missed and seed is other, "a direct hit on the click got lost"

    # 4. The gate really has to close too: a skater on the other side of the frame
    #    shouldn't still count as 'your click'.
    far = _tracklet(3, 20, 100, 0.90, 0.80)
    seed, missed = _choose_seed([far], frames, (0.10, 0.15), FPS)
    assert missed, "a skater well outside the gate still got read as the click"

    # 5. And the search window is finite: whoever only skates past the click after
    #    CLICK_SEARCH_S is a passer-by, not the skater you pointed at.
    late = _tracklet(4, int(FPS * CLICK_SEARCH_S) + 10, 60, CLICK[0], CLICK[1])
    seed, missed = _choose_seed([late, other], frames, CLICK, FPS)
    assert missed, "a skater outside the search window still got read as the click"

    # 6. With a box, size counts: a bystander four times the box height who does stand
    #    on the click isn't who was pointed at — and without a fitting candidate
    #    `_choose_seed` gives no fallback (`analyze` decides for itself whether to
    #    follow the biggest mover anyway, alongside the spyglass, and reports that).
    box = (CLICK[0] - 0.02, CLICK[1] - 0.05, CLICK[0] + 0.02, CLICK[1] + 0.05)   # 0.10 tall
    big = _tracklet(5, 0, 120, CLICK[0], CLICK[1], width=0.14, height=0.42)
    seed, missed = _choose_seed([big, target, other], frames, CLICK, FPS, doel_kader=box)
    assert not missed and seed is target, "the size gate let the big bystander through"
    seed, missed = _choose_seed([big, other], frames, CLICK, FPS, doel_kader=box)
    assert missed and seed is None, "with a box there should be no fallback to the mover"
    # Without a box, the bystander on the click simply wins (existing behavior, step 1).
    seed, missed = _choose_seed([big, target, other], frames, CLICK, FPS)
    assert not missed and seed is big

    print("Self-test _choose_seed OK")


def _self_test_spyglass():
    """
    Tests `_Spyglass` with a fake refinement — no video, no model. The refinement
    yields a skeleton at the spot where the synthetic skater stands (or nothing, to
    mimic a miss); the test checks whether the spyglass fills the right frames,
    applies the link test correctly and stops on time. Run with `python skate_yolo.py`.
    """
    FPS = 25.0

    def skeleton(cx, cy, height):
        """33 landmarks on a rectangle around (cx, cy): enough for `_bbox_from_lm`."""
        lm = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
        hw, hh = height * 0.2, height * 0.5 / (1 + 2 * SPYGLASS_MARGIN)
        for i, (dx, dy) in zip((11, 12, 23, 24, 27, 28, 0),
                               ((-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1))):
            lm[i] = Landmark(cx + dx * hw, cy + dy * hh, 0.0, 0.9)
        return lm

    def skater(f):
        """Position of the synthetic skater: skates slowly toward the bottom right and
        grows (approaching the camera)."""
        return 0.30 + 0.002 * f, 0.25 + 0.001 * f, 0.08 + 0.0004 * f

    def make_estimate(misses=(), shifted=None):
        """Fake refinement: skeleton at the real spot if the bbox contains it; None on
        the given frames (occlusion); on `shifted` frames a skeleton that sits half a
        frame width away (RTMPose landing on someone else)."""
        calls = []

        def estimate(f, bbox):
            calls.append(f)
            if f in misses:
                return None, None, 0.1, None
            cx, cy, height = skater(f)
            if shifted and f in shifted:
                cx += 0.5
            if not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
                return None, None, 0.1, None
            return skeleton(cx, cy, height), None, 0.8, None
        return estimate, calls

    def box_at(f, generous=1.0):
        cx, cy, height = skater(f)
        hw, hh = 0.2 * height * generous, 0.5 * height * generous
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def run(sg, estimate, frames, chain):
        for f in frames:
            if f in chain:
                sg.anchor(f, box_at(f))
            elif sg.alive:
                sg.step(f, lambda bb, _f=f: estimate(_f, bb))
        sg.close()

    # 1. Run-up from a (too generously drawn) box to the chain at frame 40: the link
    #    test succeeds and all 40 run-up frames are filled.
    sg = _Spyglass(FPS, chain_linked=True)
    sg.start_box(box_at(0, generous=1.6), 0)
    estimate, calls = make_estimate()
    run(sg, estimate, range(60), chain=set(range(40, 60)))
    assert set(sg.filled) == set(range(40)), sorted(sg.filled)
    assert calls[:3] == [0, 0, 0], "the first step should try the three start scalings"
    assert any("accepted" in r for r in sg.log)

    # 2. Same run-up, but the chain sits on someone else (bbox on the other side of the
    #    frame): the link test fails, nothing filled, reason in the log.
    sg = _Spyglass(FPS, chain_linked=True)
    sg.start_box(box_at(0), 0)
    estimate, _ = make_estimate()
    for f in range(40):
        sg.step(f, lambda bb, _f=f: estimate(_f, bb))
    sg.anchor(40, (0.8, 0.8, 0.9, 0.95))
    assert not sg.filled and any("rejected" in r for r in sg.log)

    # 3. Coast: an occlusion shorter than SPYGLASS_COAST_S gets bridged; afterwards the
    #    spyglass simply fills again, from a chain anchor (a tail after frame 10).
    sg = _Spyglass(FPS, chain_linked=True)
    estimate, _ = make_estimate(misses=set(range(20, 30)))
    run(sg, estimate, range(60), chain=set(range(0, 11)))
    expected = set(range(11, 20)) | set(range(30, 60))
    assert set(sg.filled) == expected, sorted(set(sg.filled) ^ expected)

    # 4. An occlusion longer than the coast lets the run die; what was pending before
    #    it belongs to a chain run and stays, and nothing more gets filled after the
    #    death.
    sg = _Spyglass(FPS, chain_linked=True)
    estimate, _ = make_estimate(misses=set(range(20, 60)))
    run(sg, estimate, range(80), chain=set(range(0, 11)))
    assert set(sg.filled) == set(range(11, 20)), sorted(sg.filled)
    assert not sg.alive

    # 5. A box run that dies without touching the chain gets rejected — a hand-drawn
    #    box is not a proven identity ...
    sg = _Spyglass(FPS, chain_linked=True)
    sg.start_box(box_at(0), 0)
    estimate, _ = make_estimate(misses=set(range(15, 80)))
    run(sg, estimate, range(80), chain=set())
    assert not sg.filled and any("without a link" in r for r in sg.log)
    assert sg.box_outcome == {'ok': False, 'reason': sg.box_outcome['reason'], 'n': 15}
    assert "link" not in sg.box_outcome['reason'].split(";")[0]   # died, not linked
    # ... except if there's no chain at all (spyglass-only): then the box is everything.
    sg = _Spyglass(FPS, chain_linked=False)
    sg.start_box(box_at(0), 0)
    estimate, _ = make_estimate()
    run(sg, estimate, range(50), chain=set())
    assert set(sg.filled) == set(range(50))
    assert sg.box_outcome['ok'] and sg.box_outcome['n'] == 50
    # ... and with a fallback chain (not linked), a dead box run also stays as is: box
    # and chain are then two separate guesses about different parts of the clip.
    sg = _Spyglass(FPS, chain_linked=False)
    sg.start_box(box_at(0), 0)
    estimate, _ = make_estimate(misses=set(range(15, 80)))
    run(sg, estimate, range(120), chain=set(range(100, 120)))
    assert set(sg.filled) == set(range(15)), sorted(sg.filled)

    # 6. Plausibility: a skeleton that suddenly sits half a frame width away is a swap
    #    and counts as a miss, not a hit.
    sg = _Spyglass(FPS, chain_linked=True)
    estimate, _ = make_estimate(shifted={12, 13})
    run(sg, estimate, range(30), chain=set(range(0, 11)))
    assert 12 not in sg.filled and 13 not in sg.filled and 14 in sg.filled

    print("Self-test _Spyglass OK")


# ── Transitional Dutch-name aliases ───────────────────────────────────────────────
# schaats_gui.py isn't translated yet (see the translate-to-english plan) and does a
# bare `import schaats_yolo`, then reads `schaats_yolo.BACKEND_NAAM` and calls
# `schaats_yolo.analyseer(...)` (in `_laad_backend()`). `schaats_yolo.py` itself now
# only exists as a tiny compatibility shim (`from skate_yolo import *`), so these
# aliases are what makes those two names available on this module for that shim to
# re-export. Remove both (and eventually the shim file) once schaats_gui.py is
# translated and imports skate_yolo directly under its real names.
BACKEND_NAAM = BACKEND_NAME
analyseer = analyze


if __name__ == '__main__':
    _self_test_corner_guard()
    _self_test_seed()
    _self_test_spyglass()
