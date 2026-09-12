# -*- coding: utf-8 -*-
"""
skate_perspective.py — calibration + 3D angle reconstruction for perspective correction
(ROADMAP phase 7, step 1: the math core, separate from the GUI and the pipeline).

Problem: the push angle is measured in the image plane; if the camera isn't
perpendicular to the leg's plane of motion, perspective distorts the angle, and that
distortion changes with the skater's position in frame.

Core idea: the track lines on the ice are straight, parallel lines with a known
distance between them. From those we calibrate the camera relative to the ice plane
(fixed camera, one calibration per video):

  1. Parallel lines in the direction of travel meet in the image at vanishing point V1;
     cross lines give V2. The line V1-V2 is the vanishing line of the ice plane (= the
     true horizon).
  2. From two orthogonal vanishing points follows the focal length (assumptions:
     principal point at the image center, square pixels): f² = -(V1-pp)·(V2-pp).
  3. From that: the camera rotation relative to the ice plane, and (with the line
     distance as scale) the ice-plane ↔ world homography, the camera position, and the
     camera height.

Required line configurations — with only 2 track lines + 1 cross line the system is
one degree of freedom short (7 unknowns: f + rotation + translation; 6 constraints):
  - ≥2 track lines + ≥2 cross lines (V2 = intersection of the cross lines), or
  - ≥3 track lines with known mutual distances (vanishing line via the cross-ratio
    construction) + ≥1 cross line, or
  - ≥2 track lines + ≥1 cross line + a given focal length `f_px`.
With a (near-)frontal camera (viewing axis in the vertical plane of the direction of
travel), V2 lies at infinity and self-calibrating f is fundamentally impossible — then
supplying `f_px` is the only route; the rest of the calibration works as normal.

Angle reconstruction: the ankle sits on the ice → world position via the homography;
the knee is only a line of sight. To pin that ray down in 3D there are two
interchangeable assumptions behind one interface (`reconstruct_angle(method=...)`),
to be compared on real footage before either becomes the default:
  - 'lower_leg': constant lower-leg length — intersect the knee ray with the sphere
    around the ankle (two intersections → tie-break rule, see `reconstruct_angle`). The
    length preferably comes from a measurement on the skater or from body height
    (`lower_leg_from_body_height()`); `calibrate_lower_leg_length()` estimates it from
    the video but is fundamentally a lower bound — see its docstring.
  - 'leg_plane': the lower leg lies in a vertical plane with a given direction (e.g.
    the direction of travel from the trajectory) — intersect the knee ray with that
    plane.

Without a known line distance in meters all the angles still come out right (scale
cancels out); only the derived meters (speed, stroke length) don't.

Pure numpy (no scipy/cv2/mediapipe/torch): importable in both venvs.
Self-test: `python skate_perspective.py` — synthetic cameras with known pose project
track lines + a "leg" with a known 3D angle; verified that the module reconstructs the
angle to within < 0.5°, for multiple camera positions and leg positions in frame.
"""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_LINE_DISTANCE = 4.0   # m — distance between consecutive track lines
CONDITION_MIN_DEG = 25.0      # leg↔ray angle below which the reconstruction counts as unreliable
PLANE_CONDITION_MIN_DEG = 10.0  # ray↔leg-plane angle below which method 'leg_plane' counts as unreliable
F_MIN_FRAC, F_MAX_FRAC = 0.4, 15.0  # plausible f range as a fraction of the image width
# Conditioning of f's self-calibration: the further a vanishing point lies from the
# image center, the less f can be extracted from it (f² = -(V1-pp)·(V2-pp) then gets
# dominated by one distant, poorly-determined point). Distance measured in units of
# f0 = (width+height)/2. Calibrated on the self-test cameras — those reach 1.0 / 1.5 /
# 9.3 and all recover the correct f — against a real track recording where the camera
# looks along the track: there V2 sits at 133 and f comes out at 3.3× the image width
# (~17° field of view, impossible for such a shot), while a 5 px shift of one line
# endpoint makes f jump from 3827 px to "impossible".
VP_CONDITION_WARN = 5.0       # above this: f is sensitive, report it
VP_CONDITION_MAX = 30.0       # above this: f is meaningless, refuse and ask for f_px
LOWER_LEG_FRACTION = 0.246    # lower-leg length (knee-ankle) as a fraction of body height
                              # (Winter's anthropometric table: knee 0.285·H - ankle 0.039·H)
KNEE_Z_MIN_FRAC = -0.05       # knee may be at most this fraction of L below the ice (noise)


# ---------------------------------------------------------------------------
# homogeneous helpers
# ---------------------------------------------------------------------------

def _unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def _line_hom(p1, p2):
    """Homogeneous line through two points, normalized so |ax+by+c| = distance."""
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    n = float(np.hypot(l[0], l[1]))
    if n < 1e-12:
        raise ValueError("line with (nearly) coincident endpoints")
    return l / n


def _vp_distance(v):
    """Distance of a vanishing point to the image center, in units of f0 = (w+h)/2
    (the normalization `calibrate_from_lines` works in). Infinitely far = parallel
    lines in the image; that's exactly the case where f can't be extracted."""
    denom = abs(v[2])
    return float("inf") if denom < 1e-15 else float(np.hypot(v[0], v[1]) / denom)


def _vanishing_point(lines):
    """Least-squares intersection of ≥2 homogeneous lines (SVD). Homogeneous result
    with |v| = 1; the w component can be ~0 (vanishing point at infinity)."""
    _, _, vt = np.linalg.svd(np.asarray(lines, dtype=float))
    return _unit(vt[-1])


def _vanishing_line_from_offsets(track_lines_hom, offsets, v1):
    """
    Vanishing line of the ice plane from ≥3 parallel lines with known mutual
    distances (cross-ratio construction). Idea: pick a transversal image line; it
    intersects the track lines at points that, in the world, march evenly along with
    the offsets. The 1D homography offset→position-on-transversal is determined from
    ≥3 pairs; the image of offset=∞ is the vanishing point of the transversal
    direction. Vanishing line = the line through V1 and that point.
    """
    # transversal through the origin (image center in normalized coordinates),
    # perpendicular to the direction toward V1 — cuts the whole fan cleanly
    if abs(v1[2]) > 1e-9:
        dirv = _unit(v1[:2] / v1[2])
    else:
        dirv = _unit(v1[:2])
    u = np.array([-dirv[1], dirv[0]])
    t = np.array([-u[1], u[0], 0.0])          # line through (0,0) with direction u

    scale = max(abs(o) for o in offsets) or 1.0
    rows = []
    for l, off in zip(track_lines_hom, offsets):
        a = np.cross(t, l)
        if abs(a[2]) < 1e-12:
            raise ValueError("transversal parallel to a track line")
        s = float((a[:2] / a[2]) @ u)          # position along the transversal
        x = off / scale
        rows.append([x, 1.0, -s * x, -s])      # s = (αx+β)/(γx+δ)
    _, _, vt = np.linalg.svd(np.asarray(rows))
    alpha, _, gamma, _ = vt[-1]
    if abs(gamma) < 1e-12 * max(abs(alpha), 1e-30):
        vp_pt = np.array([u[0], u[1], 0.0])    # vanishing point at infinity
    else:
        s_inf = alpha / gamma
        vp_pt = np.array([s_inf * u[0], s_inf * u[1], 1.0])
    return _unit(np.cross(v1, vp_pt))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@dataclass
class PerspectiveCalibration:
    """Camera ↔ ice-plane calibration for one (fixed) camera pose.

    World frame: x = across the track lines (from line 1 to line 2), y = direction of
    travel, z = up; the ice is z=0 and the origin is at the intersection of track
    line 1 × cross line 1.
    Camera frame: x = right in image, y = down in image, z = viewing direction.
    """
    w: int
    h: int
    f: float                    # focal length in pixels
    pp: tuple                   # principal point (assumed: image center)
    R: np.ndarray                # world→camera rotation; columns = world axes in camera frame
    t: np.ndarray                # world origin in camera frame
    C: np.ndarray                # camera position in world coordinates
    H: np.ndarray                # homography ice plane (X, Y, 1) → image pixels
    H_inv: np.ndarray
    horizon_line: np.ndarray     # vanishing line of the ice plane in pixel coords (a,b,c)
    scale_known: bool             # True if the line distance was given in real meters
    f_estimated: bool             # True if f comes from the vanishing points (not given)
    residual_px: float            # RMS distance from the drawn line points to the reprojection
    warnings: list = field(default_factory=list)

    @property
    def camera_height(self):
        return float(self.C[2])

    @property
    def horizon_deg(self):
        """Tilt of the true horizon (vanishing line) relative to the image x-axis,
        same convention as `horizon_angle_from_line`: positive = rises to the right."""
        a, b, _ = self.horizon_line
        dx, dy = -b, a                       # the line's direction vector
        if dx < 0:
            dx, dy = -dx, -dy
        return float(np.degrees(np.arctan2(-dy, dx)))


@dataclass
class CalibrationInput:
    """The traced lines + parameters from which a `PerspectiveCalibration` follows.

    This is what gets saved, NOT the calibration itself: `PerspectiveCalibration`
    consists almost entirely of derived matrices (R, t, C, H, H_inv) that can be
    recomputed exactly from this input. That keeps the storage JSON-able (fits in
    `analyse.instellingen_json`, no schema bump), readable by a human, and an old
    analysis automatically gets the benefit of a later improvement in the calibration
    math — the same reasoning as the events cache, which is also freshly recomputed
    on open.

    A calibration belongs to one **camera pose**, not one video: every clip from the
    same fixed setup may share it (see `fits`).
    """
    track_lines: list                # [((x1,y1),(x2,y2))] in original pixels
    cross_lines: list
    image_w: int
    image_h: int
    line_distance: float = DEFAULT_LINE_DISTANCE
    track_offsets: list = None
    scale_known: bool = True
    f_px: float = None
    note: str = ""                   # free text, e.g. "Deventer track, camera at 100m"

    def to_dict(self):
        """JSON-able form; points become floats, no numpy."""
        def _lines(ls):
            return [[[float(p[0]), float(p[1])] for p in line] for line in ls]
        return {
            "track_lines": _lines(self.track_lines),
            "cross_lines": _lines(self.cross_lines),
            "image_w": int(self.image_w),
            "image_h": int(self.image_h),
            "line_distance": float(self.line_distance),
            "track_offsets": None if self.track_offsets is None
                           else [float(o) for o in self.track_offsets],
            "scale_known": bool(self.scale_known),
            "f_px": None if self.f_px is None else float(self.f_px),
            "note": self.note or "",
        }

    @classmethod
    def from_dict(cls, d):
        """Reads either the current English keys or the original Dutch keys, so a
        calibration saved before the English rename still loads correctly."""
        def _lines(ls):
            return [tuple((float(p[0]), float(p[1])) for p in line) for line in (ls or [])]
        return cls(
            track_lines=_lines(d.get("track_lines", d.get("rijlijnen"))),
            cross_lines=_lines(d.get("cross_lines", d.get("dwarslijnen"))),
            image_w=int(d["image_w"] if "image_w" in d else d["beeld_w"]),
            image_h=int(d["image_h"] if "image_h" in d else d["beeld_h"]),
            line_distance=float(d.get("line_distance", d.get("lijnafstand", DEFAULT_LINE_DISTANCE))),
            track_offsets=d.get("track_offsets", d.get("rij_offsets")),
            scale_known=bool(d.get("scale_known", d.get("schaal_bekend", True))),
            f_px=d.get("f_px"),
            note=d.get("note", d.get("notitie", "")))

    def fits(self, w, h):
        """May this calibration be used on a video of w×h? Only with an identical
        image size — the lines are in pixels, so a different resolution or crop shifts
        them silently and yields a plausible but wrong calibration."""
        return int(w) == int(self.image_w) and int(h) == int(self.image_h)

    def calibrate(self):
        """Recompute the `PerspectiveCalibration`. Raises the same ValueError as
        `calibrate_from_lines` for an unusable configuration."""
        return calibrate_from_lines(
            self.track_lines, self.cross_lines, self.image_w, self.image_h,
            line_distance=self.line_distance, track_offsets=self.track_offsets,
            scale_known=self.scale_known, f_px=self.f_px)


def calibrate_from_lines(track_lines, cross_lines, image_w, image_h,
                         line_distance=DEFAULT_LINE_DISTANCE, track_offsets=None,
                         scale_known=True, f_px=None):
    """
    Calibrate camera ↔ ice plane from traced track lines.

    track_lines : list of ((x1,y1),(x2,y2)) pixel-point pairs of lines that run
                  parallel in the direction of travel in the world, in order (adjacent).
    cross_lines : same, perpendicular to the direction of travel (start/finish line
                  etc.); ≥1 required.
    track_offsets : world distance (m) of each track line relative to the first;
                  default evenly spaced `line_distance` apart in the given order.
    scale_known : False if the line distance is an assumption — angles remain valid,
                  meters (camera height, speed) do not.
    f_px        : known focal length in pixels; required for configurations where
                  self-calibration is underdetermined (see the module docstring).

    Returns PerspectiveCalibration; raises ValueError with an explanation for an
    unusable line configuration or degenerate geometry.
    """
    if len(track_lines) < 2:
        raise ValueError("at least two parallel track lines (direction of travel) needed")
    if len(cross_lines) < 1:
        raise ValueError("at least one cross line needed (start/finish line or corner "
                         "marking, perpendicular to the track lines)")
    if f_px is None and len(cross_lines) == 1 and len(track_lines) < 3:
        raise ValueError(
            "underdetermined: with two track lines and one cross line the calibration "
            "is one degree of freedom short — draw a third track line, or a second "
            "cross line, or supply the focal length (f_px)")

    cx, cy = image_w / 2.0, image_h / 2.0
    f0 = (image_w + image_h) / 2.0           # conditioning: work in ~O(1) coordinates

    def norm_pt(p):
        return ((p[0] - cx) / f0, (p[1] - cy) / f0)

    rl = [_line_hom(norm_pt(p1), norm_pt(p2)) for p1, p2 in track_lines]
    dl = [_line_hom(norm_pt(p1), norm_pt(p2)) for p1, p2 in cross_lines]
    offsets = list(track_offsets) if track_offsets is not None \
        else [i * line_distance for i in range(len(rl))]
    if len(offsets) != len(rl):
        raise ValueError("track_offsets must have as many values as track_lines")
    offsets = [o - offsets[0] for o in offsets]

    warnings = []

    # --- vanishing points -----------------------------------------------------
    v1 = _vanishing_point(rl)                                # direction of travel
    if len(dl) >= 2:
        v2 = _vanishing_point(dl)                             # cross direction
    elif f_px is None:
        lh = _vanishing_line_from_offsets(rl, offsets, v1)   # ≥3 track lines (checked)
        v2 = _unit(np.cross(lh, dl[0]))
    else:
        # f known: V2 = intersection of the cross line with the "orthogonal-complement
        # line" ω·v1 (all points w with v1ᵀ·ω·w = 0), ω = diag(1/fn², 1/fn², 1)
        fn = f_px / f0
        omega_v1 = np.array([v1[0] / fn**2, v1[1] / fn**2, v1[2]])
        v2 = _unit(np.cross(omega_v1, dl[0]))

    # --- focal length --------------------------------------------------------
    if f_px is not None:
        fn = f_px / f0
        f_estimated = False
    else:
        denom = v1[2] * v2[2]
        numer = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(denom) < 1e-9:
            raise ValueError(
                "vanishing point (nearly) at infinity — the camera is (almost) frontal "
                "to, or perpendicular to, the direction of travel; self-calibrating the "
                "focal length is then impossible. Supply f_px.")
        f2 = -numer / denom
        if f2 <= 0:
            raise ValueError(
                "vanishing points not consistent with a camera (f² ≤ 0) — are the "
                "cross line(s) actually perpendicular to the track lines?")
        # Conditioning: if a vanishing point lies very far away, f can't be extracted
        # from it — the lines it belongs to run nearly parallel in the image, and a
        # couple of pixels of drawing error shifts the point (and thus f) enormously.
        # Better to stop here than to hand back a plausible-looking but meaningless
        # camera pose.
        far = max(_vp_distance(v1), _vp_distance(v2))
        if far > VP_CONDITION_MAX:
            which = "the track lines" if _vp_distance(v1) > _vp_distance(v2) else "the cross lines"
            raise ValueError(
                f"focal length can't be estimated from these lines: {which} run nearly "
                f"parallel in the image, so their vanishing point lies ~{far:.0f}× the "
                f"image size away (usable is < {VP_CONDITION_MAX:.0f}). The camera is "
                f"then looking almost along that direction and f doesn't follow from it "
                f"— a couple of pixels of drawing error already changes it by a factor. "
                f"Supply f_px (checkerboard calibration or camera spec); the rest of the "
                f"calibration works as normal.")
        if far > VP_CONDITION_WARN:
            warnings.append(
                f"focal length poorly determined: farthest vanishing point at ~{far:.0f}× "
                f"the image size — f is sensitive to a couple of pixels of drawing error; "
                f"consider supplying f_px")
        fn = float(np.sqrt(f2))
        f_estimated = True
    f_pix = fn * f0
    if not (F_MIN_FRAC * image_w <= f_pix <= F_MAX_FRAC * image_w):
        warnings.append(
            f"implausible focal length ({f_pix:.0f} px at image width "
            f"{image_w}) — the calibration is poorly conditioned (camera nearly "
            f"frontal?); consider supplying f_px")

    # --- rotation ------------------------------------------------------------
    def k_inv(v):
        return _unit(np.array([v[0] / fn, v[1] / fn, v[2]]))

    r_y = k_inv(v1)                          # direction of travel (world y) in camera frame
    r_x = k_inv(v2)
    r_x = _unit(r_x - (r_x @ r_y) * r_y)     # make exactly orthogonal
    r_z = np.cross(r_x, r_y)

    def K(v):
        return np.array([fn * v[0], fn * v[1], v[2]])

    # --- translation + scale ------------------------------------------------
    # world origin = intersection of track line 1 × cross line 1; second track line sets the scale
    o = np.cross(rl[0], dl[0])
    if abs(o[2]) < 1e-12:
        raise ValueError("first track line and cross line don't intersect in the image "
                         "(drawn parallel?)")
    u0 = k_inv(o)
    if u0[2] < 0:
        u0 = -u0                             # origin lies in front of the camera
    d = offsets[1]
    if d == 0:
        raise ValueError("two track lines with the same offset — check the distances")
    p2 = np.cross(rl[1], dl[0])              # image of world point (d, 0)
    c1 = np.cross(p2, K(d * r_x))
    c2 = np.cross(p2, K(u0))
    n2 = float(c2 @ c2)
    if n2 < 1e-18:
        raise ValueError("degenerate line configuration while determining the scale")
    mu = -float(c1 @ c2) / n2
    if mu < 0:                               # x-axis pointed the wrong way
        r_x = -r_x
        r_z = np.cross(r_x, r_y)
        mu = -mu
    t = mu * u0
    R = np.column_stack([r_x, r_y, r_z])
    C = -R.T @ t
    if C[2] < 0:                             # camera belongs ABOVE the ice (direction-of-travel sign is free)
        r_y = -r_y
        r_z = np.cross(r_x, r_y)
        R = np.column_stack([r_x, r_y, r_z])
        C = -R.T @ t

    # --- homography + horizon in pixel coordinates ---------------------------
    K_px = np.array([[f_pix, 0.0, cx], [0.0, f_pix, cy], [0.0, 0.0, 1.0]])
    H = K_px @ np.column_stack([r_x, r_y, t])
    H /= np.linalg.norm(H)
    H_inv = np.linalg.inv(H)

    def to_px(v):                            # normalized homogeneous point → pixels
        return np.array([v[0] * f0 + cx * v[2], v[1] * f0 + cy * v[2], v[2]])

    horizon_line = np.cross(to_px(v1), to_px(v2))
    horizon_line /= np.hypot(horizon_line[0], horizon_line[1])

    # --- residual: drawn points vs. reprojection of the world lines ----
    H_inv_T = H_inv.T
    dist = []
    world_lines = [(np.array([1.0, 0.0, -off]), line)
                    for off, line in zip(offsets, track_lines)]
    world_lines.append((np.array([0.0, 1.0, 0.0]), cross_lines[0]))
    for wl, (p1, p2) in world_lines:
        l_img = H_inv_T @ wl
        l_img /= np.hypot(l_img[0], l_img[1])
        for p in (p1, p2):
            dist.append(float(l_img @ [p[0], p[1], 1.0]))
    residual = float(np.sqrt(np.mean(np.square(dist))))

    kal = PerspectiveCalibration(
        w=image_w, h=image_h, f=f_pix, pp=(cx, cy), R=R, t=t, C=C,
        H=H, H_inv=H_inv, horizon_line=horizon_line,
        scale_known=scale_known, f_estimated=f_estimated,
        residual_px=residual, warnings=warnings)

    # extra cross lines: only used for V2 — check that in the world they do indeed
    # come out ~perpendicular to the direction of travel (a drawing-quality signal)
    for i, (p1, p2) in enumerate(cross_lines[1:], start=2):
        w1 = point_on_ice(kal, p1)
        w2 = point_on_ice(kal, p2)
        if w1 is None or w2 is None:
            continue
        direction = _unit((w2 - w1)[:2])
        skew = abs(np.degrees(np.arcsin(np.clip(direction[1], -1, 1))))
        if skew > 3.0:
            kal.warnings.append(
                f"cross line {i} is {skew:.1f}° off perpendicular in the world — "
                f"drawn sloppily, or not really a cross line?")
    return kal


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def _ray(kal, px):
    """Line of sight through a pixel: (camera position, unit direction) in world coords."""
    v = np.array([(px[0] - kal.pp[0]) / kal.f, (px[1] - kal.pp[1]) / kal.f, 1.0])
    return kal.C, _unit(kal.R.T @ v)


def point_on_ice(kal, px, height=0.0):
    """World position (3-vector) of a pixel on the (horizontal) plane z=`height` —
    default the ice plane itself. `height` > 0 corrects for the ankle landmark not
    sitting ON the ice but at malleolus/boot height. None if the line of sight doesn't
    hit the plane (pixel on/above the horizon)."""
    C, d = _ray(kal, px)
    if d[2] >= -1e-9 or height >= C[2]:
        return None
    s = (height - C[2]) / d[2]
    return C + s * d


def _image_angle(ankle_px, knee_px):
    """Image-plane angle exactly as the current pipeline measures it
    (calculate_angle_to_ice with horizon 0, unrounded) — a reference to show the
    correction being applied."""
    dx = knee_px[0] - ankle_px[0]
    dy = ankle_px[1] - knee_px[1]
    return float(np.degrees(np.arctan2(dy, abs(dx))))


@dataclass
class AngleReconstruction:
    """Result of one 3D angle reconstruction."""
    angle: float                # real angle relative to the ice plane (degrees)
    image_angle: float          # uncorrected image-plane angle
    correction: float           # angle - image_angle (quality indicator: large = heavily corrected)
    condition_deg: float        # angle between lower leg and line of sight; small = leg in viewing direction
    reliable: bool               # False on poor conditioning or a degenerate intersection
    method: str
    X_ankle: np.ndarray          # world coordinates (meters if scale is known)
    X_knee: np.ndarray
    angle_alternative: float = None   # method 'lower_leg': angle of the sphere solution not chosen
    plane_condition_deg: float = None  # method 'leg_plane': angle ray↔plane; small = unstable intersection


def reconstruct_angle(kal, ankle_px, knee_px, method="lower_leg",
                      lower_leg_l=None, plane_direction=None, travel_direction=None,
                      ankle_height=0.0):
    """
    Reconstruct the real push angle relative to the ice plane from ankle and knee pixels.

    method 'lower_leg': intersect the knee line of sight with the sphere (radius
      `lower_leg_l`) around the ankle. Two intersection points; tie-break rule in
      order of availability:
      1. `plane_direction` (2D world direction): the solution closest to the vertical
         plane through the ankle in that direction;
      2. `travel_direction` (2D world direction of the trajectory): the solution whose
         knee leans forward the most;
      3. otherwise: smallest |correction| (conservative — least deviation from the image).
    method 'leg_plane': intersect the knee ray with the vertical plane through the
      ankle with direction `plane_direction` (required).

    `ankle_height` (m) places the ankle not ON the ice but at that height above it
    (malleolus + skate ≈ 0.10 m); the angle stays relative to the (horizontal) ice plane.

    Returns AngleReconstruction, or None if the ankle can't be placed on the ice
    (pixel above the horizon). `reliable=False` marks frames where the leg is nearly
    in the viewing direction (condition < {:.0f}°) or the geometry didn't close.
    """.format(CONDITION_MIN_DEG)
    # Transitional: schaats_analyse.py/schaats_gui.py haven't been translated to
    # English yet (see the translate-to-english plan) and still pass the original
    # Dutch method names through PerspectiefConfig/the calibration dialog. Accept both
    # spellings here so this module can be finished on its own without a cross-file
    # break; drop this once every caller uses 'lower_leg'/'leg_plane' directly.
    method = {"onderbeen": "lower_leg", "beenvlak": "leg_plane"}.get(method, method)

    A = point_on_ice(kal, ankle_px, height=ankle_height)
    if A is None:
        return None
    C, dk = _ray(kal, knee_px)
    image_angle = _image_angle(ankle_px, knee_px)
    degenerate = False
    alternative = None
    plane_condition = None

    if method == "lower_leg":
        if lower_leg_l is None:
            raise ValueError("method 'lower_leg' requires lower_leg_l (meters, or "
                             "via calibrate_lower_leg_length)")
        L = float(lower_leg_l)
        w0 = C - A
        b = float(dk @ w0)
        c = float(w0 @ w0) - L * L
        disc = b * b - c
        if disc < 0:
            # ray misses the sphere: image knee farther from the ankle than L can
            # explain (noise/wrong L) → take the closest point and mark unreliable
            candidates = [C + (-b) * dk]
            degenerate = True
        else:
            w_disc = float(np.sqrt(disc))
            candidates = [C + s * dk for s in (-b - w_disc, -b + w_disc) if s > 1e-9]
            candidates = [X for X in candidates if X[2] - A[2] > KNEE_Z_MIN_FRAC * L]
            if not candidates:
                candidates = [C + (-b) * dk]
                degenerate = True
        if len(candidates) == 1:
            X = candidates[0]
        else:
            if plane_direction is not None:
                u = _unit([plane_direction[0], plane_direction[1], 0.0])
                m = np.cross(u, [0.0, 0.0, 1.0])
                X = min(candidates, key=lambda Xk: abs(float(m @ (Xk - A))))
            elif travel_direction is not None:
                r = _unit([travel_direction[0], travel_direction[1], 0.0])
                X = max(candidates, key=lambda Xk: float(r @ (Xk - A)))
            else:
                X = min(candidates, key=lambda Xk: abs(_angle_to_ice(A, Xk) - image_angle))
            other = candidates[0] if candidates[1] is X else candidates[1]
            alternative = _angle_to_ice(A, other)

    elif method == "leg_plane":
        if plane_direction is None:
            raise ValueError("method 'leg_plane' requires plane_direction (2D world direction)")
        u = _unit([plane_direction[0], plane_direction[1], 0.0])
        m = np.cross(u, [0.0, 0.0, 1.0])     # normal of the vertical leg plane
        denom = float(m @ dk)
        # conditioning of the intersection: angle between the ray and the plane — if
        # the ray lies (nearly) in the plane, any calibration/pixel error blows up
        # enormously; that happens exactly with a frontal camera and the plane in the
        # direction of travel
        plane_condition = float(np.degrees(np.arcsin(np.clip(abs(denom), 0.0, 1.0))))
        if abs(denom) < 1e-9:
            return AngleReconstruction(
                angle=image_angle, image_angle=image_angle, correction=0.0,
                condition_deg=0.0, reliable=False, method=method,
                X_ankle=A, X_knee=A, plane_condition_deg=plane_condition)
        if plane_condition < PLANE_CONDITION_MIN_DEG:
            degenerate = True
        s = float(m @ (A - C)) / denom
        if s <= 0:
            degenerate = True
            s = abs(s)
        X = C + s * dk
    else:
        raise ValueError(f"unknown method: {method!r}")

    leg = X - A
    leg_n = float(np.linalg.norm(leg))
    if leg_n < 1e-9:
        return AngleReconstruction(
            angle=image_angle, image_angle=image_angle, correction=0.0, condition_deg=0.0,
            reliable=False, method=method, X_ankle=A, X_knee=X,
            plane_condition_deg=plane_condition)
    angle = _angle_to_ice(A, X)
    condition = float(np.degrees(np.arccos(np.clip(abs(leg / leg_n @ dk), 0.0, 1.0))))
    reliable = (not degenerate) and condition >= CONDITION_MIN_DEG
    return AngleReconstruction(
        angle=angle, image_angle=image_angle, correction=angle - image_angle,
        condition_deg=condition, reliable=reliable, method=method,
        X_ankle=A, X_knee=X, angle_alternative=alternative,
        plane_condition_deg=plane_condition)


def _angle_to_ice(A, X):
    """Angle (degrees) of the segment A→X relative to the ice plane z=0."""
    d = X - A
    return float(np.degrees(np.arcsin(np.clip(d[2] / np.linalg.norm(d), -1.0, 1.0))))


def lower_leg_from_body_height(body_height_m):
    """Anthropometric estimate of the lower-leg length (knee-ankle):
    0.246 × body height (Winter's table). Measuring it on the skater directly (back
    of the knee to the ankle bone) is even better; both are more reliable than
    `calibrate_lower_leg_length` (see the bias explanation there)."""
    return LOWER_LEG_FRACTION * float(body_height_m)


def calibrate_lower_leg_length(kal, pairs, percentile=95.0, ankle_height=0.0):
    """
    Estimate the lower-leg length (world units) from a series of (ankle_px, knee_px)
    pairs.

    Per frame, the perpendicular distance from the 3D ankle to the knee line of sight
    is a lower bound for the length (foreshortening can only ever shorten a
    projection); a high percentile of those lower bounds approximates the real
    length — but only if the lower leg is actually ~perpendicular to the viewing
    direction somewhere in the series.

    NOTE — systematic underestimation: in skating, the ankle angle (dorsiflexion)
    keeps the lower leg permanently leaned forward, so seen from the front the lower
    leg always looks shorter than it is, and that perpendicular moment may never
    occur. Synthetically measured bias at realistic orientations (elevation 45-70°,
    lean within direction of travel ± 30-60°): 2-6% too short, growing as the camera
    gets more frontal and the spread of lean angles gets smaller — and that carries
    through as angle errors of a few degrees in method 'lower_leg'. So prefer a
    measured length, or `lower_leg_from_body_height()`; this estimate is only a lower
    bound / sanity check. Returns None with no usable frames.
    """
    lower = []
    for ankle_px, knee_px in pairs:
        A = point_on_ice(kal, ankle_px, height=ankle_height)
        if A is None:
            continue
        C, dk = _ray(kal, knee_px)
        w0 = A - C
        lower.append(float(np.linalg.norm(w0 - (w0 @ dk) * dk)))
    if not lower:
        return None
    return float(np.percentile(lower, percentile))


# ---------------------------------------------------------------------------
# self-test: synthetic cameras with known pose
# ---------------------------------------------------------------------------

class _SynthCamera:
    """Virtual camera with a known pose, for the self-test."""

    def __init__(self, name, C, target, roll_deg, f, w=1920, h=1080):
        self.name, self.f, self.w, self.h = name, float(f), w, h
        self.C = np.asarray(C, dtype=float)
        z = _unit(np.asarray(target, float) - self.C)
        x = _unit(np.cross(z, [0.0, 0.0, 1.0]))
        y = np.cross(z, x)
        r = np.radians(roll_deg)
        x, y = np.cos(r) * x + np.sin(r) * y, -np.sin(r) * x + np.cos(r) * y
        self.R_wc = np.vstack([x, y, z])     # rows = camera axes in world coords

    def project(self, Xw):
        Xc = self.R_wc @ (np.asarray(Xw, dtype=float) - self.C)
        assert Xc[2] > 0.2, f"{self.name}: point {Xw} behind the camera"
        return (self.f * Xc[0] / Xc[2] + self.w / 2.0,
                self.f * Xc[1] / Xc[2] + self.h / 2.0)

    def segment(self, P1, P2):
        return (self.project(P1), self.project(P2))

    def in_frame(self, px, margin=0.0):
        return (-margin <= px[0] < self.w + margin) and (-margin <= px[1] < self.h + margin)

    def true_horizon_line(self):
        r_z = self.R_wc[:, 2]                # world ẑ in camera frame
        K = np.array([[self.f, 0, self.w / 2.0], [0, self.f, self.h / 2.0], [0, 0, 1.0]])
        l = np.linalg.inv(K).T @ r_z
        return l / np.hypot(l[0], l[1])


# scene: track lines x = 0, 4, 8 (along y), cross lines y = 4 and 18
_TRACK_X = [0.0, 4.0, 8.0]
_CROSS_Y = [4.0, 18.0]
_Y_RANGE = (2.0, 22.0)
_X_RANGE = (-1.0, 9.0)
_L_LEG = 0.45


def _scene_lines(cam):
    track = [cam.segment((x, _Y_RANGE[0], 0), (x, _Y_RANGE[1], 0)) for x in _TRACK_X]
    cross = [cam.segment((_X_RANGE[0], y, 0), (_X_RANGE[1], y, 0)) for y in _CROSS_Y]
    return track, cross


def _leg(ankle_xy, alpha_deg, azimuth_deg, length=_L_LEG):
    """Ankle on the ice + knee with a known 3D angle `alpha` and lean direction `azimuth`."""
    A = np.array([ankle_xy[0], ankle_xy[1], 0.0])
    a, fi = np.radians(alpha_deg), np.radians(azimuth_deg)
    K = A + length * np.array([np.cos(a) * np.cos(fi), np.cos(a) * np.sin(fi), np.sin(a)])
    return A, K


def _line_angle_error(l1, l2):
    """Angle difference (degrees) between two normalized image lines."""
    c = abs(l1[0] * l2[0] + l1[1] * l2[1])
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def _calibration_checks(cam, kal, errors):
    name = cam.name
    f_err = abs(kal.f - cam.f) / cam.f
    # calibration world frame = scene frame shifted by cross line 1 (y -= _CROSS_Y[0])
    C_expected = cam.C - np.array([0.0, _CROSS_Y[0], 0.0])
    c_err = float(np.linalg.norm(kal.C - C_expected))
    h_err = abs(kal.camera_height - cam.C[2]) / cam.C[2]
    hz_err = _line_angle_error(kal.horizon_line, cam.true_horizon_line())
    reproj = 0.0
    for x in np.linspace(0, 8, 5):
        for y in np.linspace(*_Y_RANGE, 5):
            px = np.array(cam.project((x, y, 0)))
            q = kal.H @ [x, y - _CROSS_Y[0], 1.0]
            reproj = max(reproj, float(np.linalg.norm(q[:2] / q[2] - px)))
    if f_err > 0.005:
        errors.append(f"{name}: f error {f_err:.2%}")
    if c_err > 0.02:
        errors.append(f"{name}: camera-position error {c_err:.3f} m")
    if hz_err > 0.05:
        errors.append(f"{name}: horizon error {hz_err:.3f}°")
    if reproj > 0.1:
        errors.append(f"{name}: homography reprojection {reproj:.3f} px")
    return f_err, h_err, hz_err, reproj


def self_test(extended=True):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    rng = np.random.default_rng(7)
    errors = []

    cameras = [
        _SynthCamera("center-frontal  ", C=(4, -18, 1.8), target=(4, 8, 0), roll_deg=0.0, f=1400),
        _SynthCamera("off-center      ", C=(-8, -14, 2.5), target=(6, 8, 0), roll_deg=0.0, f=1400),
        _SynthCamera("far off + roll  ", C=(16, -9, 3.2), target=(2, 12, 0), roll_deg=2.5, f=1050),
        _SynthCamera("nearly frontal  ", C=(7, -16, 2.2), target=(4, 10, 0), roll_deg=-1.0, f=1600),
    ]

    print("== Calibration from projected track lines ==")
    print("   configurations: A = 3 track lines + 1 cross line (cross-ratio vanishing line)")
    print("                   B = 2 track lines + 2 cross lines (V2 from the cross lines)")
    print("                   C = 2 track lines + 1 cross line + given f")
    print(f"{'camera':<17} {'cfg':<4} {'f-error':>9} {'height-error':>12} "
          f"{'horizon-error':>13} {'reproj (px)':>12}")
    kals = {}
    for cam in cameras:
        track, cross = _scene_lines(cam)
        for cfg, kwargs in [
                ("A", dict(track_lines=track, cross_lines=cross[:1])),
                ("B", dict(track_lines=track[:2], cross_lines=cross)),
                ("C", dict(track_lines=track[:2], cross_lines=cross[:1], f_px=cam.f))]:
            try:
                kal = calibrate_from_lines(image_w=cam.w, image_h=cam.h, **kwargs)
            except ValueError as e:
                message = str(e).split("—")[0].strip()
                print(f"{cam.name:<17} {cfg:<4} self-calibration refused: {message} …")
                if cfg == "C" or cam.name.strip() != "center-frontal":
                    errors.append(f"{cam.name}/{cfg}: unexpected refusal: {e}")
                continue
            f_e, h_e, hz_e, rp = _calibration_checks(cam, kal, errors)
            ws = "  ⚠ " + kal.warnings[0][:40] if kal.warnings else ""
            print(f"{cam.name:<17} {cfg:<4} {f_e:>9.2%} {h_e:>12.2%} "
                  f"{hz_e:>12.4f}° {rp:>12.4f}{ws}")
            kals.setdefault(cam.name, (cam, kal))  # first successful calibration per camera

    print("\n== Angle reconstruction (exact projections, leg in frame) ==")
    print(f"   lower leg L = {_L_LEG} m; angles 35/50/65°; lean directions 0-330°; "
          f"requirement: error < 0.5° on frames marked reliable")
    print(f"{'camera':<17} {'n':>4} {'image error min/avg/max':>24} "
          f"{'error (b)':>10} {'(b) flag':>9} {'error (a) best':>14} "
          f"{'choice trav./least':>18}")
    ankles = [(1, 4), (4, 8), (6, 14), (3, 18), (7, 10), (2, 12)]
    alphas = [35.0, 50.0, 65.0]
    azimuths = list(range(0, 360, 30))
    for cam, kal in kals.values():
        image_e, fa, fb = [], [], []
        b_flagged = choice_trav = choice_least = n = 0
        for ankle in ankles:
            for alpha in alphas:
                for az in azimuths:
                    A, Kn = _leg(ankle, alpha, az)
                    if Kn[2] <= 0:
                        continue
                    e_px, k_px = cam.project(A), cam.project(Kn)
                    if not (cam.in_frame(e_px) and cam.in_frame(k_px)):
                        continue
                    n += 1
                    image_e.append(abs(_image_angle(e_px, k_px) - alpha))
                    rb = reconstruct_angle(kal, e_px, k_px, method="leg_plane",
                                           plane_direction=(np.cos(np.radians(az)),
                                                          np.sin(np.radians(az))))
                    if rb.reliable:
                        fb.append(abs(rb.angle - alpha))
                    else:
                        b_flagged += 1        # leg plane ~ parallel to the line of sight
                    ra = reconstruct_angle(kal, e_px, k_px, method="lower_leg",
                                           lower_leg_l=_L_LEG, travel_direction=(0, 1))
                    best = min(abs(ra.angle - alpha),
                                abs(ra.angle_alternative - alpha)
                                if ra.angle_alternative is not None else np.inf)
                    fa.append(best)
                    if abs(ra.angle - alpha) < 0.5:
                        choice_trav += 1
                    rd = reconstruct_angle(kal, e_px, k_px, method="lower_leg",
                                           lower_leg_l=_L_LEG)
                    if abs(rd.angle - alpha) < 0.5:
                        choice_least += 1
        name = cam.name
        if n < 20:
            errors.append(f"{name}: too few legs in frame (n={n})")
        if fb and max(fb) > 0.5:
            errors.append(f"{name}: method (b) error {max(fb):.3f}°")
        if max(fa) > 0.5:
            errors.append(f"{name}: method (a) best-solution error {max(fa):.3f}°")
        print(f"{name:<17} {n:>4} {min(image_e):>7.2f}/{np.mean(image_e):>6.2f}/"
              f"{max(image_e):>6.2f}°  {max(fb) if fb else 0.0:>9.4f}° "
              f"{b_flagged / n:>8.0%} {max(fa):>13.4f}° "
              f"{choice_trav / n:>7.0%} /{choice_least / n:>5.0%}")

    print("\n== Lower-leg-length self-calibration ==")
    print("   ideal case (uniform lean directions 0-330°, so also ~perpendicular to")
    print("   the line of sight) validates the math; 'realistic' limits the lean")
    print("   direction to direction of travel ± 45° (the ankle angle keeps the lower")
    print("   leg leaned forward) and shows the systematic underestimate — which is")
    print("   why a measured length is the preferred route and this estimate is only")
    print("   a lower bound / sanity check.")
    travel_az = -90.0                        # skater travels in -y (toward the cameras)
    for cam, kal in kals.values():
        pairs_ideal, pairs_real = [], []
        for ankle in ankles:
            for alpha in alphas:
                for az in azimuths:
                    A, Kn = _leg(ankle, alpha, az)
                    if Kn[2] <= 0:
                        continue
                    e_px, k_px = cam.project(A), cam.project(Kn)
                    if cam.in_frame(e_px) and cam.in_frame(k_px):
                        pairs_ideal.append((e_px, k_px))
                        d_az = (az - travel_az + 180) % 360 - 180
                        if abs(d_az) <= 45:
                            pairs_real.append((e_px, k_px))
        L_i = calibrate_lower_leg_length(kal, pairs_ideal, percentile=100.0)
        L_r = calibrate_lower_leg_length(kal, pairs_real, percentile=100.0)
        rel_i = abs(L_i - _L_LEG) / _L_LEG
        bias_r = (L_r - _L_LEG) / _L_LEG
        if rel_i > 0.01:
            errors.append(f"{cam.name}: lower-leg-length error (ideal) {rel_i:.2%}")
        if L_i > _L_LEG * 1.001 or L_r > _L_LEG * 1.001:
            errors.append(f"{cam.name}: lower-leg-length estimate above the real "
                          f"length — no longer a lower bound")
        print(f"{cam.name:<17} ideal: {L_i:.4f} m (error {rel_i:.2%})   "
              f"realistic: {L_r:.4f} m (bias {bias_r:+.1%})")

    print("\n== Serialization: CalibrationInput round-trip via JSON ==")
    cam = cameras[1]
    track, cross = _scene_lines(cam)
    inv = CalibrationInput(track_lines=track, cross_lines=cross, image_w=cam.w, image_h=cam.h,
                           line_distance=DEFAULT_LINE_DISTANCE, note="self-test")
    kal_a = inv.calibrate()
    import json as _json
    blob = _json.dumps(inv.to_dict())
    kal_b = CalibrationInput.from_dict(_json.loads(blob)).calibrate()
    # Byte-identical, not 'approximately': the recomputation must take the same path,
    # otherwise a reopened analysis would silently give slightly different angles than
    # a fresh one.
    diffs = [n for n in ("f", "H", "H_inv", "R", "t", "C", "horizon_line")
                   if not np.array_equal(np.asarray(getattr(kal_a, n), float),
                                         np.asarray(getattr(kal_b, n), float))]
    print(f"json {len(blob)} bytes; recomputed byte-identical: "
          f"{'yes' if not diffs else 'NO — ' + ', '.join(diffs)}")
    if diffs:
        errors.append(f"serialization: {', '.join(diffs)} differ after round-trip")
    # The image size must travel along: the same lines at a different resolution
    # would give a plausible but wrong calibration.
    if not (inv.fits(cam.w, cam.h) and not inv.fits(cam.w // 2, cam.h)):
        errors.append("serialization: fits() doesn't guard the image size")
    # Also confirm the old Dutch key names still load (pre-rename saved calibrations).
    old_blob = _json.dumps({
        "rijlijnen": inv.to_dict()["track_lines"], "dwarslijnen": inv.to_dict()["cross_lines"],
        "beeld_w": cam.w, "beeld_h": cam.h, "lijnafstand": DEFAULT_LINE_DISTANCE,
        "rij_offsets": None, "schaal_bekend": True, "f_px": None, "notitie": "old format",
    })
    kal_old = CalibrationInput.from_dict(_json.loads(old_blob)).calibrate()
    if not np.array_equal(np.asarray(kal_old.H, float), np.asarray(kal_a.H, float)):
        errors.append("serialization: loading the old Dutch key names gave a different result")

    print("\n== Quality flag: leg nearly in the viewing direction ==")
    cam, kal = kals[cameras[1].name]
    A = np.array([4.0, 8.0, 0.0])
    to_cam = _unit(cam.C - A)                # upward toward the camera
    perp = _unit(np.cross(to_cam, [0, 0, 1.0]))
    d_leg = _unit(np.cos(np.radians(10)) * to_cam + np.sin(np.radians(10)) * perp)
    Kn = A + _L_LEG * d_leg
    r = reconstruct_angle(kal, cam.project(A), cam.project(Kn),
                          method="lower_leg", lower_leg_l=_L_LEG)
    print(f"leg 10° from the line of sight: condition = {r.condition_deg:.1f}°, "
          f"reliable = {r.reliable} (threshold {CONDITION_MIN_DEG:.0f}°)")
    if r.reliable or r.condition_deg > 15:
        errors.append("quality flag: nearly-in-viewing-direction not flagged")

    if extended:
        print("\n== Noise sensitivity (informational): σ = 1 px on all line endpoints, "
              "300 trials, leg at (4,8), 50°, lean direction 150° ==")
        az = 150.0
        A_w, Kn_w = _leg((4, 8), 50.0, az)
        plane = (np.cos(np.radians(az)), np.sin(np.radians(az)))
        for cam_i in (1, 3):                 # off-center and nearly-frontal
            cam = cameras[cam_i]
            track, cross = _scene_lines(cam)
            e_px, k_px = cam.project(A_w), cam.project(Kn_w)
            f_est, err_b, err_a, flagged, failed = [], [], [], 0, 0
            for _ in range(300):
                noise = lambda seg: tuple(
                    (p[0] + rng.normal(0, 1.0), p[1] + rng.normal(0, 1.0)) for p in seg)
                try:
                    kal_n = calibrate_from_lines([noise(s) for s in track],
                                                 [noise(s) for s in cross[:1]],
                                                 cam.w, cam.h)
                except ValueError:
                    failed += 1
                    continue
                f_est.append(kal_n.f)
                rb = reconstruct_angle(kal_n, e_px, k_px, method="leg_plane",
                                       plane_direction=plane)
                if rb.reliable:
                    err_b.append(abs(rb.angle - 50.0))
                else:
                    flagged += 1
                ra = reconstruct_angle(kal_n, e_px, k_px, method="lower_leg",
                                       lower_leg_l=_L_LEG, travel_direction=(0, 1))
                err_a.append(min(abs(ra.angle - 50.0),
                                  abs(ra.angle_alternative - 50.0)
                                  if ra.angle_alternative is not None else np.inf))
            if f_est:
                print(f"{cam.name:<17} f p5/p50/p95 = {np.percentile(f_est, 5):.0f}/"
                      f"{np.percentile(f_est, 50):.0f}/{np.percentile(f_est, 95):.0f} px "
                      f"(real {cam.f:.0f}); refused {failed}, flagged {flagged}")
                if err_b:
                    print(f"{'':<17} error (b) p50/p95 = {np.percentile(err_b, 50):.2f}/"
                          f"{np.percentile(err_b, 95):.2f}°   "
                          f"error (a, best) p50/p95 = {np.percentile(err_a, 50):.2f}/"
                          f"{np.percentile(err_a, 95):.2f}°")
            else:
                print(f"{cam.name:<17} all {failed} trials refused")

        # demonstration: a leg plane (nearly) parallel to the line of sight gets flagged
        cam, kal = kals[cameras[1].name]
        A_d, Kn_d = _leg((4, 8), 50.0, 60.0)    # line-of-sight azimuth ≈ 61° for this camera
        rd = reconstruct_angle(kal, cam.project(A_d), cam.project(Kn_d),
                               method="leg_plane",
                               plane_direction=(np.cos(np.radians(60)), np.sin(np.radians(60))))
        print(f"\ndegenerate leg plane (lean direction ≈ viewing direction, off-center): "
              f"plane condition = {rd.plane_condition_deg:.1f}°, reliable = {rd.reliable}")
        if rd.reliable:
            errors.append("quality flag: degenerate leg plane not flagged")

    print()
    if errors:
        print(f"FAIL — {len(errors)} problem(s):")
        for f in errors:
            print(f"  - {f}")
        return 1
    print("PASS — calibration, both reconstruction methods (< 0.5°), length "
          "self-calibration, and the quality flag are all in order.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
