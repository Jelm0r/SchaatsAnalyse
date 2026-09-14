"""Does every GUI window fit on screen? — regression test for the window sizes.

    .venv-yolo\\Scripts\\python.exe skate_screentest.py                       # the floor: fails if anything doesn't fit
    .venv-yolo\\Scripts\\python.exe skate_screentest.py --all                 # report across all screens and font sizes
    .venv-yolo\\Scripts\\python.exe skate_screentest.py --all --font-size 11  # same, one font size

**The rule enforced here:** every window fits — with its Windows 11 frame (31 px title
bar, 1 px side borders) — in the work area of a screen of **1280×720 logical pixels
with a 48 px taskbar**, at the default font size (Segoe UI 9 pt). That's an FHD laptop
at 150% scaling (the common setting on a 13-14" screen), and the smallest size you
still run into on a Windows laptop; 1366×768 at 125% (1092×566) falls outside it and
also fights with other programs. "Fits" means: the window's effective **minimum
size** — where Qt can never take it below, no matter how you `resize()` it — plus the
frame stays within the work area, and the size `set_window_size` actually opens the
window at fits within it too.

Why a test and not a one-off measurement: the minimum silently creeps up with every
button or row that gets added. On 12-9-2026 the main window was 643 px tall (CLAUDE.md
still said 631) and no longer fit on 1280×720 — the hidden compare page counted, the
wrapping bars counted at their narrowest wrap, and two long names on the compare page
made the window 1489 px wide. See the findings in CLAUDE.md ("Passen op elk scherm").

**Measuring method.** The Qt offscreen platform with a screen configuration in a JSON
file (`QT_QPA_PLATFORM=offscreen:configfile=...`; the path must **not** contain a
drive letter — the colon breaks the parser and Qt crashes with 0xc0000409, hence the
chdir + relative path). Screen sizes in that file are *logical* pixels. Two things
without which no number is correct: `QT_QPA_FONTDIR=C:\\Windows\\Fonts` (without a
font directory, offscreen knows no fonts at all on Windows and measures everything
twice too wide) and the `windows11` style + Segoe UI of the real platform (offscreen
otherwise picks Fusion with "Sans Serif"). Validated against the real platform with
hidden windows (`WA_DontShowOnScreen`): difference ≤ 1%. Each scenario runs in a
**child process**, because the platform reads the screen configuration only once, at
startup.

Every window is built against a real test library (a synthetic 40-frame video, two
analyses with long names, one recording, a long library path) — the stress cases are
therefore part of the measurement by default.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The floor (see the docstring): logical screen size and taskbar height.
FLOOR = ("1280x720 logical (FHD laptop at 150%)", 1280, 720, 48)
FONT_STANDARD = 9.0

# Windows 11 window frame at 100%: title bar 31 px (measured 29 at dpr 2), 1 px side.
TITLE_BAR, SIDE_BORDER = 31, 1

# The report grid: (name, logical width, logical height, taskbar).
GRID = [
    ("1024x768 @100% (projector)",  1024,  768, 48),
    ("1280x720 @100%",              1280,  720, 48),
    ("1280x800 @100/200%",          1280,  800, 48),
    ("1366x768 @100%",              1366,  768, 48),
    ("1366x768 @125%",              1092,  614, 48),
    ("1440x900 @100%",              1440,  900, 48),
    ("1920x1080 @100%",             1920, 1080, 48),
    ("1920x1080 @125%",             1536,  864, 48),
    ("1920x1080 @150%",             1280,  720, 48),
    ("1920x1200 @125%",             1536,  960, 48),
    ("2560x1440 @150%",             1706,  960, 48),
    ("2256x1504 @150% (Surface)",   1504, 1002, 48),
    ("2736x1824 @200% (Surface)",   1368,  912, 48),
    ("3840x2160 @200%",             1920, 1080, 48),
    ("3840x2160 @300%",             1280,  720, 48),
]
FONT_SIZES = (9.0, 10.0, 11.0, 12.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Child process: one screen, measure every window
# ═══════════════════════════════════════════════════════════════════════════
def _measure(name, width, height, taskbar, font_size, out_path):
    tmp = tempfile.mkdtemp(prefix="skate_screentest_")
    json.dump({"screens": [{"name": name, "x": 0, "y": 0, "width": width,
                            "height": height - taskbar,
                            "logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1.0}]},
              open(os.path.join(tmp, "screen.json"), "w"))
    os.chdir(tmp)
    os.environ["QT_QPA_PLATFORM"] = "offscreen:configfile=screen.json"
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
    # A long library path is one of the stress cases (the label under the start page).
    bieb = os.path.join(tmp, "Google Drive", "Shared drives",
                        "IJsster Amersfoort Skating Club", "Trainers",
                        "Technique analysis 2026-2027", "library")
    lokaal = os.path.join(tmp, "local")
    os.environ["SKATEANALYSIS_LIBRARY"] = bieb
    os.environ["SKATEANALYSIS_LOCAL"] = lokaal
    os.environ["SCHAATSANALYSE_CPU"] = "1"
    sys.path.insert(0, HERE)

    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt
    app = QApplication([])
    style = QStyleFactory.create("windows11")
    if style is not None:
        app.setStyle(style)
    app.setFont(QFont("Segoe UI", font_size))

    import math
    import shutil
    import numpy as np
    import cv2
    import skate_db
    import skate_gui as G
    from skate_analysis import (FrameResult, Landmark, process_derivatives,
                                 segment_pushes, video_info)

    # ── fixture ──────────────────────────────────────────────────────────
    def make_video(pad, w=640, h=360, n=40, fps=25.0):
        wr = cv2.VideoWriter(pad, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            f = np.full((h, w, 3), 200, np.uint8)
            cv2.circle(f, (w // 2 + i * 3, h // 2), 40, (30, 30, 30), -1)
            wr.write(f)
        wr.release()

    def make_results(n, fps):
        res = []
        for i in range(n):
            t = i / fps
            lm = [Landmark(0.5, 0.5, 0.0, 0.0)] * 33
            s = math.sin(2 * math.pi * t)
            pts = {0: (0.50, 0.15), 11: (0.44, 0.28), 12: (0.56, 0.28),
                   13: (0.40, 0.40), 14: (0.60, 0.40), 15: (0.38, 0.50), 16: (0.62, 0.50),
                   23: (0.46, 0.50), 24: (0.54, 0.50),
                   25: (0.43 - 0.03 * s, 0.70), 26: (0.57 + 0.03 * s, 0.70),
                   27: (0.42 - 0.05 * s, 0.90 - 0.04 * max(s, 0)),
                   28: (0.58 + 0.05 * s, 0.90 - 0.04 * max(-s, 0))}
            pts[29] = (pts[27][0] - 0.01, pts[27][1] + 0.02)
            pts[30] = (pts[28][0] + 0.01, pts[28][1] + 0.02)
            pts[31] = (pts[27][0] + 0.02, pts[27][1] + 0.02)
            pts[32] = (pts[28][0] - 0.02, pts[28][1] + 0.02)
            for j, (x, y) in pts.items():
                lm[j] = Landmark(x, y, 0.0, 1.0)
            res.append(FrameResult(frame_nr=i, time=t, lm=lm, pose_found=True))
        return res

    skate_db.open_db(bieb)
    skate_db.open_db(lokaal)
    video = os.path.join(tmp, "Testvideo.mp4")
    make_video(video)
    info = video_info(video)
    resultaten = make_results(info.totaal, info.fps)
    process_derivatives(resultaten, info.w, info.h, info.fps)
    events = segment_pushes(resultaten)
    long_name = "Second Skater with a fairly long name"
    sid = skate_db.create_skater(bieb, "Test Skater", 2010)
    sid2 = skate_db.create_skater(bieb, long_name, 2008)
    inst = {"smooth_n": 5, "threshold": 0.015, "smooth_landmarks": True,
            "skip_corner": True, "doel_punt": [0.5, 0.5], "horizon_deg": 0.0,
            "auto_horizon": False, "heavy": False, "deinterlaced": False}
    aid1 = skate_db.save_analysis(bieb, sid, "Analysis one", video, info, resultaten,
                                   events, "yolo", dict(inst), aangemaakt_door="Tester")
    aid2 = skate_db.save_analysis(bieb, sid2, "Analysis two with a long title for the header",
                                   video, info, resultaten, events, "yolo", dict(inst))
    os.makedirs(os.path.join(bieb, "opnames"), exist_ok=True)
    recording = os.path.join(bieb, "opnames", "Recording.mp4")
    shutil.copy2(video, recording)
    skate_db.sync_source_dir(bieb)
    source = skate_db.list_source_videos(bieb)[0]
    source_l = skate_db.loose_video(bieb, lokaal, video)

    # ── measure ──────────────────────────────────────────────────────────
    screen = app.primaryScreen().availableGeometry()
    out = {"scenario": name, "font_size": font_size, "style": app.style().objectName(),
           "font": app.font().family(), "workarea": [screen.width(), screen.height()],
           "windows": []}

    def pump(n=6):
        for _ in range(n):
            app.processEvents()

    def effective_min(w):
        # Where Qt never lets the window come in below: an explicit minimumSize wins
        # over minimumSizeHint, per axis.
        ms, mh = w.minimumSize(), w.minimumSizeHint()
        return [ms.width() if ms.width() > 0 else mh.width(),
                ms.height() if ms.height() > 0 else mh.height()]

    def measure(label, w):
        pump()
        g = w.geometry()
        out["windows"].append({
            "window": label, "min": effective_min(w),
            "geom": [g.x(), g.y(), g.width(), g.height()],
            "maximized": bool(w.windowState() & Qt.WindowMaximized),
            "fullscreen": bool(w.windowState() & Qt.WindowFullScreen)})

    def dialog(label, make):
        d = make()
        d.show()
        measure(label, d)
        d.close()
        d.deleteLater()
        pump()

    frame = np.full((1080, 1920, 3), 120, np.uint8)
    skaters = skate_db.list_skaters(bieb)
    dialog("TargetPicker", lambda: G.TargetPicker(frame))
    dialog("HorizonPicker", lambda: G.HorizonPicker(frame))
    dialog("CalibrationPicker", lambda: G.CalibrationPicker(frame))
    dialog("SkaterDialog", lambda: G.SkaterDialog())
    dialog("NewAnalysisDialog", lambda: G.NewAnalysisDialog(skaters))
    dialog("BatchAnalysisDialog", lambda: G.BatchAnalysisDialog(skaters))
    dialog("AnalysisPicker", lambda: G.AnalysisPicker(bieb))
    dialog("AnalysisInfoDialog", lambda: G.AnalysisInfoDialog(
        skate_db.analysis_meta(bieb, aid1), "Test Skater"))
    dialog("FragmentPicker", lambda: G.FragmentPicker(
        recording, info, analyzed=[{"start_frame": 2, "eind_frame": 10, "titel": "x"}]))
    dialog("ViewWindow (1 video)", lambda: G.ViewWindow([(source, info)], "Tester"))
    dialog("ViewWindow (2 videos)",
           lambda: G.ViewWindow([(source, info), (source_l, info)], "Tester"))

    mw = G.MainWindow()
    mw.show()
    pump(10)
    measure("MainWindow: start page", mw)
    mw._open_analysis_from_library(aid1)
    pump(10)
    measure("MainWindow: analysis open", mw)
    mw.btn_edit.setChecked(True)
    pump(10)
    measure("MainWindow: analysis, edit mode", mw)
    mw.btn_edit.setChecked(False)
    pump()
    mw._set_compare_side(mw.side_left, aid2, long_name)
    mw._set_compare_side(mw.side_right, aid2, long_name)
    mw.stack.setCurrentWidget(mw.page_compare)
    pump(10)
    measure("MainWindow: compare, 2 long names", mw)
    mw.stack.setCurrentWidget(mw.page_start)
    pump(10)
    measure("MainWindow: back on start (compare filled)", mw)
    mw.close()
    pump(10)

    json.dump(out, open(out_path, "w"))
    app.quit()
    shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Parent process: run scenarios and assess them
# ═══════════════════════════════════════════════════════════════════════════
def measure_scenario(name, width, height, taskbar, font_size):
    """Runs one scenario in a child process and returns the measurement."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--measure", name, str(width),
             str(height), str(taskbar), str(font_size), out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"measurement '{name}' failed:\n{proc.stderr[-3000:]}")
        return json.load(open(out_path))
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def assess(v, work_w, work_h):
    """Does window `v` fit in the work area? Returns (fits, width_shortfall,
    height_shortfall, reason). A fullscreen window has no frame."""
    mw, mh = v["min"]
    if v["fullscreen"]:
        tw, th = mw - work_w, mh - work_h
    else:
        tw, th = mw + 2 * SIDE_BORDER - work_w, mh + TITLE_BAR - work_h
    if tw > 0 or th > 0:
        return False, max(0, tw), max(0, th), "minimum size"
    # The size the window actually opened at (after set_window_size), with the frame:
    # `move()` sets the frame corner, so the title bar sits above geom.y and the side
    # borders next to it.
    x, y, w, h = v["geom"]
    if not (v["fullscreen"] or v["maximized"]):
        bottom = y + TITLE_BAR + h + SIDE_BORDER - work_h
        right = x + w + 2 * SIDE_BORDER - work_w
        if bottom > 0 or right > 0 or x < 0 or y < 0:
            return False, max(0, right), max(0, bottom), "opened size"
    return True, 0, 0, ""


def report(measurement):
    work_w, work_h = measurement["workarea"]
    print(f"\n=== {measurement['scenario']} — work area {work_w}×{work_h}, "
          f"{measurement['font_size']:g} pt {measurement['font']} / {measurement['style']} ===")
    errors = 0
    for v in measurement["windows"]:
        fits, tw, th, reason = assess(v, work_w, work_h)
        state = ("fullscreen" if v["fullscreen"]
                 else "maximized" if v["maximized"]
                 else f"opens {v['geom'][2]}×{v['geom'][3]}")
        if fits:
            margin_h = work_h - v["min"][1] - (0 if v["fullscreen"] else TITLE_BAR)
            print(f"  ok    {v['window']:<46} min {v['min'][0]:>4}×{v['min'][1]:<4} "
                  f"{state:<18} height margin {margin_h:>3} px")
        else:
            errors += 1
            print(f"  FAIL  {v['window']:<46} min {v['min'][0]:>4}×{v['min'][1]:<4} "
                  f"{state:<18} {reason}: {tw} px too wide, {th} px too tall")
    return errors


def main(argv):
    if argv[:1] == ["--measure"]:
        name, w, h, tb, font_size, out_path = argv[1:7]
        _measure(name, int(w), int(h), int(tb), float(font_size), out_path)
        return 0

    if "--all" in argv:
        sizes = FONT_SIZES
        if "--font-size" in argv:
            sizes = (float(argv[argv.index("--font-size") + 1]),)
        total = 0
        for font_size in sizes:
            for name, w, h, tb in GRID:
                total += report(measure_scenario(f"{name}, {font_size:g} pt", w, h, tb, font_size))
        print(f"\n{total} window(s) don't fit somewhere in the whole grid "
              f"(the floor scenario is excluded — it's informative, not a failure).")
        return 0

    name, w, h, tb = FLOOR
    errors = report(measure_scenario(name, w, h, tb, FONT_STANDARD))
    if errors:
        print(f"\nFAIL: {errors} window(s) don't fit on the agreed floor "
              f"({name}). See CLAUDE.md, 'Passen op elk scherm'.")
        return 1
    print(f"\nOK: every window fits on the floor ({name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
