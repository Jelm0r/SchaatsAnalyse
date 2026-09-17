"""
Evaluation of tracking accuracy (measurement basis for improvements)
=======================================================================
Measures the quality of a stored analysis (.npz from phase 0) with proxy metrics
that need no ground truth, and — if there's a golden reference — the real angle
error in degrees. That way every pipeline change can be tested rigorously instead
of by eye (same discipline as the phase 6 measurement protocol in ROADMAP.md).

Usage (works in both venvs; only needs numpy + cv2 + skate_analysis):

    # Proxy metrics for one analysis
    python skate_eval.py metrics analysis.npz [--golden golden.json]

    # Two analyses (e.g. before/after a change) side by side + per-joint difference
    python skate_eval.py compare old.npz new.npz [--golden golden.json]

    # Create a golden reference: click knees + ankles in N spread-out frames
    python skate_eval.py annotate video.mp4 --out golden.json [--n 15]

    # Check the corner signal (calibrate thresholds on new footage)
    python skate_eval.py corner analysis.npz [--step 10]

The proxy metrics:
- **coverage** — fraction of frames with a pose.
- **bone-length CV** — coefficient of variation (std/mean) of the tibia and femur
  pixel length per leg. Bones are rigid: their length in the image should only
  change slowly (distance to the camera). A high CV means keypoints jumping back
  and forth on the leg.
- **jitter** — average |second derivative| (px/frame²) of knee/ankle within
  contiguous pose segments. Measures shake that isn't real motion.
- **events** — number of pushes, L-R order, and flagged alternation errors (via
  the normal `process_derivatives` + `segment_pushes`).

The golden reference is a JSON with manually clicked knee/ankle positions in a
number of frames; the angle error is then |measured angle - golden angle| of the
ankle→knee segment per leg (image plane, including the analysis's horizon
subtraction). Annotation clicks in two stages (coarse → zoomed-in) for subpixel
precision. That error is **split into stance leg and swing leg**: only the stance
leg produces the angle that ends up in the table and in `PushEvent.angle`, while
the swing leg is (partly) hidden behind the stance leg when filmed frontally and
is no more than a guess there. **The stance-leg number is the measure that
matters** — including where `compare` prints an A/B difference; see
`_golden_errors` for the numbers behind that choice.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from skate_analysis import (
    load_landmarks, process_derivatives, segment_pushes, calculate_angle_to_ice,
    corner_ratio, determine_corner_sequence, CORNER_IN, CORNER_OUT,
    VIS_MIN, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, _savgol,
)

# Order in which the annotation is clicked per frame (name, landmark index).
ANNOTATION_POINTS = [
    ('l_knee', L_KNEE), ('l_ankle', L_ANKLE),
    ('r_knee', R_KNEE), ('r_ankle', R_ANKLE),
]
ZOOM = 6           # magnification of the precision click
ZOOM_REGION = 100  # side length (px) of the crop around the coarse click
# Lower bound for the bone-length trend as a fraction of the median bone length. A bone
# can shrink in the image because the skater is moving away or the leg turns into the
# viewing direction, but not to a quarter of its own median — if the trend dips below
# that, the fit was pulled off by an outlier and the denominator is meaningless (see
# _bone_length_cv).
TREND_MIN_FRAC = 0.25


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _segments(resultaten):
    """Contiguous runs of frame indices with a pose."""
    segments, current = [], []
    for i, r in enumerate(resultaten):
        if r.pose_found and r.lm is not None:
            current.append(i)
        elif current:
            segments.append(current); current = []
    if current:
        segments.append(current)
    return segments


def _px(r, idx, w, h):
    return np.array([r.lm[idx].x * w, r.lm[idx].y * h])


def _visible(r, *idxs):
    return all(r.lm[i].visibility >= VIS_MIN for i in idxs)


def _bone_length_cv(resultaten, w, h, idx_a, idx_b, fps, leg=None):
    """
    Bone-length instability: relative spread around the slow trend.
    The trend (Savitzky-Golay, ~0.7 s) captures the real scale change caused by the
    skater approaching the camera; what's left over is measurement noise on the
    keypoints.

    With `leg` ('links'/'rechts') only frames where that leg is the actual stánce
    leg are counted — the leg the measurement is actually taken on. The swing leg
    is regularly (partly) hidden behind the stance leg when filmed frontally;
    detection errors there aren't measurement errors.

    Returns `(cv, n_used, n_skipped)`. Skipped are frames where the trend dips
    below `TREND_MIN_FRAC` × the median bone length: a polynomial fit through a
    deep outlier (swing leg disappearing behind the stance leg) can dip locally to
    zero or even **negative**, and then `length / trend` blows the CV up to a
    seven-digit nonsense number (BUGS.md R3: `CV tibia_l: 981184.372`). Such
    frames have no usable denominator; they should be counted, not divided by.
    """
    lengths = [float(np.linalg.norm(_px(r, idx_a, w, h) - _px(r, idx_b, w, h)))
               for r in resultaten
               if r.pose_found and r.lm is not None and _visible(r, idx_a, idx_b)
               and (leg is None or r.leg == leg)]
    if len(lengths) < 10:
        return None, len(lengths), 0
    lengths = np.array(lengths)
    window = max(5, int(round(0.7 * fps)) | 1)
    trend = _savgol(lengths, window, 2)
    valid = trend >= TREND_MIN_FRAC * float(np.median(lengths))
    n_skipped = int((~valid).sum())
    if int(valid.sum()) < 10:      # too few usable denominators -> no verdict
        return None, int(valid.sum()), n_skipped
    return float(np.std(lengths[valid] / trend[valid] - 1.0)), int(valid.sum()), n_skipped


def _jitter(resultaten, w, h, idxs):
    """Average |second derivative| (px/frame²) of the given landmarks, per segment."""
    values = []
    for seg in _segments(resultaten):
        if len(seg) < 3:
            continue
        for idx in idxs:
            P = np.array([_px(resultaten[i], idx, w, h) for i in seg])
            acc = np.diff(P, n=2, axis=0)
            values.extend(np.linalg.norm(acc, axis=1))
    return float(np.mean(values)) if values else None


def _load_and_process(pad):
    info, resultaten = load_landmarks(pad)
    process_derivatives(resultaten, info.w, info.h, info.fps)
    events = segment_pushes(resultaten)
    return info, resultaten, events


# ── Metrics ─────────────────────────────────────────────────────────────────────
def calculate_metrics(pad, golden_pad=None):
    """All metrics for one npz as a dict (for printing or comparing)."""
    info, resultaten, events = _load_and_process(pad)
    w, h = info.w, info.h
    m = {'path': pad, 'frames': len(resultaten)}
    m['coverage'] = sum(1 for r in resultaten if r.pose_found) / max(1, len(resultaten))

    for naam, (a, b) in (('tibia_l', (L_KNEE, L_ANKLE)), ('tibia_r', (R_KNEE, R_ANKLE)),
                         ('femur_l', (L_HIP, L_KNEE)), ('femur_r', (R_HIP, R_KNEE))):
        cv, n, skipped = _bone_length_cv(resultaten, w, h, a, b, info.fps)
        m[f'cv_{naam}'], m[f'n_{naam}'], m[f'skipped_{naam}'] = cv, n, skipped
    # Stance-leg variants: only frames where this leg is the leg being measured.
    for naam, leg, (a, b) in (('stance_l', 'links', (L_KNEE, L_ANKLE)),
                              ('stance_r', 'rechts', (R_KNEE, R_ANKLE))):
        cv, n, skipped = _bone_length_cv(resultaten, w, h, a, b, info.fps, leg=leg)
        m[f'cv_{naam}'], m[f'n_{naam}'], m[f'skipped_{naam}'] = cv, n, skipped
    m['jitter_knee_ankle'] = _jitter(resultaten, w, h, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE))

    m['n_events'] = len(events)
    m['order'] = ''.join('L' if e.leg == 'links' else 'R' for e in events)
    m['alternation_errors'] = sum(1 for e in events if e.note == 'missed counter-push?')
    m['angles'] = [e.angle for e in events]
    # Incomplete pushes (the video ended mid-push, or no sideways push was observed at
    # all) have an angle that's too steep and shouldn't be read as a measurement — only
    # flagged here with a reason, because the metrics are a diagnostic tool: you want to
    # see THAT they're there and WHY.
    m['incomplete'] = {i: e.incomplete for i, e in enumerate(events) if e.incomplete}

    # Midline quality flag (only present in newer analyses).
    devs = [abs(v) for r in resultaten if getattr(r, 'midline_dev', None)
            for v in r.midline_dev.values() if v is not None]
    m['midline_dev_px'] = float(np.mean(devs)) if devs else None

    if golden_pad:
        m.update(_golden_errors(resultaten, w, h, golden_pad))
    return m


def _golden_errors(resultaten, w, h, golden_pad):
    """
    Angle and position errors relative to manually annotated frames.

    The angle error is **split by stance/swing leg** (`r.leg`, set by
    `assign_push_leg_cyclic`) — same measurement philosophy as the `leg` parameter
    of `_bone_length_cv` above. Filmed frontally, the swing leg is (partly) hidden
    behind the stance leg; both the detector and the human clicking the golden
    reference are guessing there, which produces "errors" of tens of degrees. And
    that angle never ends up in the result anyway: only the stance leg produces
    `PushEvent.angle` and the table. Lumped together, the average is over six
    times too pessimistic (measured on "Skate frontal.MOV": 8.47° mixed vs. 1.34°
    stance leg), and it mostly tracks those guesses in an A/B comparison — which
    drowns out a real improvement of a few tenths of a degree.
    **The stance-leg number is the measure that matters.**

    Frames without a stance leg (`r.leg is None`: corner frames and detection
    gaps) fall out of both averages and are only counted, so it stays visible how
    many of the annotated frames actually take part.
    """
    with open(golden_pad, encoding='utf-8') as f:
        golden = json.load(f)
    leg_code = {'links': 'l', 'rechts': 'r'}
    stance, swing, point_errors = [], [], []
    n_no_leg = 0
    for frame_nr_s, points in golden.get('frames', {}).items():
        frame_nr = int(frame_nr_s)
        if frame_nr >= len(resultaten):
            continue
        r = resultaten[frame_nr]
        if not (r.pose_found and r.lm is not None):
            continue
        for naam, idx in ANNOTATION_POINTS:
            if naam in points:
                point_errors.append(float(np.linalg.norm(
                    _px(r, idx, w, h) - np.array(points[naam]))))
        stance_code = leg_code.get(r.leg)
        for leg, (k_idx, e_idx) in (('l', (L_KNEE, L_ANKLE)), ('r', (R_KNEE, R_ANKLE))):
            knee, ankle = points.get(f'{leg}_knee'), points.get(f'{leg}_ankle')
            if knee is None or ankle is None:
                continue
            golden_angle = calculate_angle_to_ice(ankle, knee, r.horizon_deg)
            measured_angle = calculate_angle_to_ice(_px(r, e_idx, w, h), _px(r, k_idx, w, h),
                                                     r.horizon_deg)
            error = abs(measured_angle - golden_angle)
            if stance_code is None:
                n_no_leg += 1
            elif leg == stance_code:
                stance.append(error)
            else:
                swing.append(error)
    n_total = len(stance) + len(swing) + n_no_leg
    if not n_total:
        return {'golden_n': 0}
    m = {'golden_n': n_total, 'golden_n_stance': len(stance), 'golden_n_swing': len(swing),
         'golden_n_no_leg': n_no_leg}
    if stance:
        m['golden_angle_error_stance_avg'] = float(np.mean(stance))
        m['golden_angle_error_stance_max'] = float(np.max(stance))
    if swing:
        m['golden_angle_error_swing_avg'] = float(np.mean(swing))
        m['golden_angle_error_swing_max'] = float(np.max(swing))
    if point_errors:
        m['golden_point_error_px'] = float(np.mean(point_errors))
    return m


def print_metrics(m):
    print(f"\n== {os.path.basename(m['path'])} ==")
    print(f"  coverage:         {m['coverage']:.1%}  ({m['frames']} frames)")
    for naam in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r', 'stance_l', 'stance_r'):
        cv, n = m[f'cv_{naam}'], m.get(f'n_{naam}', 0)
        # The number of measurements alongside it: a CV over a handful of frames says
        # little. Same for the number of frames without a usable trend — there are many
        # of those for a leg that regularly disappears behind the other, and then the
        # CV needs a grain of salt too.
        skipped = m.get(f'skipped_{naam}', 0)
        count = f"(n={n}" + (f", {skipped} skipped)" if skipped else ")")
        print(f"  CV {naam}:       {cv:.3f}  {count}" if cv is not None
              else f"  CV {naam}:       unreliable/too few measurements  {count}")
    j = m['jitter_knee_ankle']
    print(f"  jitter knee/ankle: {j:.2f} px/frame^2" if j is not None else "  jitter: -")
    print(f"  events: {m['n_events']}  order {m['order']}  "
          f"alternation errors {m['alternation_errors']}")
    incomplete = m.get('incomplete', {})
    angle_text = ', '.join(f"{round(x, 1)}{'*' if i in incomplete else ''}"
                           for i, x in enumerate(m['angles']))
    print(f"  angles at completion: [{angle_text}]"
          + (f"   (* = {'/'.join(sorted(set(incomplete.values())))}, doesn't count)"
             if incomplete else ""))
    if m.get('midline_dev_px') is not None:
        print(f"  knee midline deviation: {m['midline_dev_px']:.1f} px avg")
    if m.get('golden_n'):
        # Stance leg first, with an arrow: that's the only number that's actually about
        # the measurement (see _golden_errors). The swing-leg line stays because the
        # spread is diagnostic — dropping it would trade a misleading number for a
        # hidden one.
        if m.get('golden_angle_error_stance_avg') is not None:
            print(f"  GOLDEN stance leg ({m['golden_n_stance']}): "
                  f"angle error avg {m['golden_angle_error_stance_avg']:5.2f}°  "
                  f"max {m['golden_angle_error_stance_max']:5.2f}°   <-- this is the measure")
        if m.get('golden_angle_error_swing_avg') is not None:
            print(f"  GOLDEN swing leg ({m['golden_n_swing']}): "
                  f"angle error avg {m['golden_angle_error_swing_avg']:5.2f}°  "
                  f"max {m['golden_angle_error_swing_max']:5.2f}°   "
                  f"(hidden behind the stance leg - not actionable)")
        if m.get('golden_point_error_px') is not None:
            no_leg = m.get('golden_n_no_leg', 0)
            print(f"  GOLDEN point error avg {m['golden_point_error_px']:.1f} px"
                  + (f"   ({no_leg} measurements without a stance leg: corner/gap, not counted)"
                     if no_leg else ""))


# ── Compare ─────────────────────────────────────────────────────────────────────
def compare(pad_a, pad_b, golden_pad=None):
    ma = calculate_metrics(pad_a, golden_pad)
    mb = calculate_metrics(pad_b, golden_pad)
    print_metrics(ma)
    print_metrics(mb)

    info_a, res_a, _ = _load_and_process(pad_a)
    info_b, res_b, _ = _load_and_process(pad_b)
    w, h = info_a.w, info_a.h
    n = min(len(res_a), len(res_b))
    print(f"\n== difference per joint (px, over frames with a pose in both) ==")
    for naam, idx in (('knee L', L_KNEE), ('knee R', R_KNEE),
                      ('ankle L', L_ANKLE), ('ankle R', R_ANKLE)):
        d = [float(np.linalg.norm(_px(res_a[i], idx, w, h) - _px(res_b[i], idx, w, h)))
             for i in range(n)
             if res_a[i].pose_found and res_a[i].lm is not None
             and res_b[i].pose_found and res_b[i].lm is not None]
        if d:
            d = np.array(d)
            print(f"  {naam}: median {np.median(d):.1f}  avg {d.mean():.1f}  "
                  f"p95 {np.percentile(d, 95):.1f}")

    # The stance-leg number goes side by side: that's what the A/B conclusion rests on,
    # and hunting it down across two separate metrics blocks invites misreading it (see
    # _golden_errors).
    if (ma.get('golden_angle_error_stance_avg') is not None
            and mb.get('golden_angle_error_stance_avg') is not None):
        print("\n== GOLDEN stance leg ==")
        for label, key in (('angle error avg', 'golden_angle_error_stance_avg'),
                           ('angle error max', 'golden_angle_error_stance_max')):
            a, b = ma[key], mb[key]
            print(f"  {label}  {a:5.2f} -> {b:5.2f}°   ({b - a:+.2f})")
        print(f"  n             {ma['golden_n_stance']} -> {mb['golden_n_stance']}")


# ── Annotation (golden reference) ────────────────────────────────────────────────
def annotate(video_pad, out_pad, n_frames=15):
    """
    Interactive clicker: pick n frames evenly spread across the video and click
    the four points in ANNOTATION_POINTS order for each frame. Each click happens
    in two stages: coarse on the full frame, then precise in a zoomed-in crop.
    Keys: u = redo last point, s = skip frame, q = stop and save.
    """
    cap = cv2.VideoCapture(video_pad)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {video_pad}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, n_frames)))

    existing = {}
    if os.path.exists(out_pad):
        with open(out_pad, encoding='utf-8') as f:
            existing = json.load(f).get('frames', {})
        print(f"({len(existing)} previously annotated frames loaded; those are kept)")

    frames_out = dict(existing)
    window = 'annotate'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    click = {}

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['xy'] = (x, y)

    cv2.setMouseCallback(window, _mouse)

    def _wait_click(image, text):
        """Show image + text, wait for a click or key. Returns ('click', (x,y)) or ('key', k)."""
        shown = image.copy()
        cv2.putText(shown, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(shown, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow(window, shown)
        click.clear()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if 'xy' in click:
                return 'click', click['xy']
            if k in (ord('u'), ord('s'), ord('q')):
                return 'key', chr(k)

    frame_nr = -1
    ok = True
    stopped = False          # 'q' must stop the WHOLE loop, not just the current point
    for target in targets:
        if str(target) in frames_out:
            continue
        while frame_nr < target and ok:
            ok, frame = cap.read()
            frame_nr += 1
        if not ok:
            break
        h, w = frame.shape[:2]
        points, i = {}, 0
        while i < len(ANNOTATION_POINTS):
            naam, _ = ANNOTATION_POINTS[i]
            base = frame.copy()
            for nm, (px, py) in points.items():
                cv2.drawMarker(base, (int(px), int(py)), (0, 200, 0),
                               cv2.MARKER_CROSS, 18, 2)
            kind, res = _wait_click(
                base, f"frame {target} ({len(frames_out)+1}/{len(targets)}): click {naam}"
                      "   [u=redo s=skip q=stop]")
            if kind == 'key':
                if res == 'u' and points:
                    i -= 1
                    points.pop(ANNOTATION_POINTS[i][0], None)
                    continue
                if res == 's':
                    points = None
                    break
                if res == 'q':
                    # Only `targets = []` would rebind the name; the for-loop iterates
                    # over the original list object and would just carry on to the next
                    # frame. Hence an explicit flag.
                    points, stopped = None, True
                    break
                continue
            gx, gy = res
            # Stage 2: zoomed-in crop around the coarse click, for the precise position.
            x0 = int(np.clip(gx - ZOOM_REGION // 2, 0, w - ZOOM_REGION))
            y0 = int(np.clip(gy - ZOOM_REGION // 2, 0, h - ZOOM_REGION))
            crop = cv2.resize(frame[y0:y0 + ZOOM_REGION, x0:x0 + ZOOM_REGION],
                              (ZOOM_REGION * ZOOM, ZOOM_REGION * ZOOM),
                              interpolation=cv2.INTER_NEAREST)
            kind2, res2 = _wait_click(crop, f"precise: {naam}")
            if kind2 == 'key':
                continue                      # back to the coarse click for this same point
            zx, zy = res2
            points[naam] = [x0 + zx / ZOOM, y0 + zy / ZOOM]
            i += 1
        if points:
            frames_out[str(target)] = points
        if stopped:
            break

    cap.release()
    cv2.destroyAllWindows()
    with open(out_pad, 'w', encoding='utf-8') as f:
        json.dump({'video': os.path.basename(video_pad), 'frames': frames_out},
                  f, indent=1)
    print(f"{len(frames_out)} frames annotated -> {out_pad}")


# ── Corner signal ─────────────────────────────────────────────────────────────────
def corner_report(pad, step=None):
    """
    Print an analysis's corner signal: per frame the raw `corner_ratio`
    (hip width / torso length) plus the segments `determine_corner_sequence`
    finds in it. This lets you re-derive the threshold on new footage without the
    GUI — needed because CORNER_IN/CORNER_OUT are calibrated on whatever clips
    happen to be in the library.
    """
    info, resultaten = load_landmarks(pad)
    w, h, fps = info.w, info.h, info.fps
    raw = [corner_ratio(r.lm, w, h) if (r.pose_found and r.lm is not None) else None
           for r in resultaten]
    determine_corner_sequence(resultaten, w, h, fps)

    measured = [v for v in raw if v is not None]
    print(f"{os.path.basename(pad)}: {len(resultaten)} frames @ {fps:.1f} fps, "
          f"{len(measured)} measurable")
    if measured:
        q = np.percentile(measured, [5, 50, 95])
        print(f"  ratio  min={min(measured):.2f}  p05={q[0]:.2f}  median={q[1]:.2f}  "
              f"p95={q[2]:.2f}  max={max(measured):.2f}   (thresholds: "
              f"corner < {CORNER_IN}, straight > {CORNER_OUT})")

    segments, start = [], None
    for i, r in enumerate(resultaten):
        if r.corner and start is None:
            start = i
        elif not r.corner and start is not None:
            segments.append((start, i - 1)); start = None
    if start is not None:
        segments.append((start, len(resultaten) - 1))
    n_corner = sum(1 for r in resultaten if r.corner)
    print(f"  corner: {n_corner} frames ({n_corner / max(1, len(resultaten)):.0%}) "
          f"in {len(segments)} segment(s)")
    for a, b in segments:
        print(f"    frames {a}-{b}  ({a / fps:.1f}-{b / fps:.1f} s, {(b - a + 1) / fps:.1f} s)")

    # Deliberately ASCII: the Windows console runs on cp1252 here and chokes on block
    # characters.
    step = step or max(1, len(resultaten) // 60)
    print(f"  timeline (every {step} frames; . = no pose, # = corner):")
    for i in range(0, len(resultaten), step):
        v = raw[i]
        bar = "#" if resultaten[i].corner else " "
        print(f"    f{i:5d} {i / fps:6.1f}s  {'  . ' if v is None else f'{v:4.2f}'} {bar}")


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('metrics', help='proxy metrics for a single analysis npz')
    pm.add_argument('npz')
    pm.add_argument('--golden', help='golden-reference json (from "annotate")')

    pv = sub.add_parser('compare', help='two analyses side by side')
    pv.add_argument('npz_old')
    pv.add_argument('npz_new')
    pv.add_argument('--golden')

    pa = sub.add_parser('annotate', help='click a golden reference')
    pa.add_argument('video')
    pa.add_argument('--out', required=True, help='output json')
    pa.add_argument('--n', type=int, default=15, help='number of frames (default 15)')

    pc = sub.add_parser('corner', help='corner signal + detected corner segments')
    pc.add_argument('npz')
    pc.add_argument('--step', type=int, default=None,
                    help='print the timeline every this many frames')

    args = p.parse_args()
    if args.cmd == 'metrics':
        print_metrics(calculate_metrics(args.npz, args.golden))
    elif args.cmd == 'compare':
        compare(args.npz_old, args.npz_new, args.golden)
    elif args.cmd == 'annotate':
        annotate(args.video, args.out, args.n)
    elif args.cmd == 'corner':
        corner_report(args.npz, args.step)


if __name__ == '__main__':
    main()
