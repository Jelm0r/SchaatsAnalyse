"""Past elk venster van de GUI op het scherm? — regressietest voor de venstermaten.

    .venv-yolo\\Scripts\\python.exe schaats_schermtest.py            # de ondergrens: faalt als iets niet past
    .venv-yolo\\Scripts\\python.exe schaats_schermtest.py --alles    # rapport over alle schermen en letters
    .venv-yolo\\Scripts\\python.exe schaats_schermtest.py --alles --letter 11   # idem, één lettergrootte

**De afspraak die hier wordt afgedwongen:** elk venster past — met zijn Windows 11-rand
(titelbalk 31 px, zijkanten 1 px) — in het werkgebied van een scherm van **1280×720
logische pixels met een taakbalk van 48 px**, bij de standaardletter (Segoe UI 9 pt). Dat
is een FHD-laptop op 150% schaal (de gangbare stand op 13–14"), en de kleinste maat die
je op een Windows-laptop nog tegenkomt; 1366×768 op 125% (1092×566) valt erbuiten en
vecht ook met andere programma's. "Past" betekent: de effectieve **minimummaat** van het
venster — daar kan Qt het nooit onder krijgen, hoe je ook `resize()`t — plus de rand blijft
binnen het werkgebied, én de maat waarmee `zet_venstergrootte` het venster werkelijk opent
staat er ook binnen.

Waarom een test en geen eenmalige meting: het minimum schuift stilzwijgend omhoog met elke
knop of regel die erbij komt. Op 12-9-2026 was het hoofdvenster 643 px hoog (CLAUDE.md
zei nog 631) en paste het niet meer op 1280×720 — de verborgen vergelijkpagina telde mee,
de afbrekende balken telden met hun smalste afbreking mee, en twee lange namen op de
vergelijkpagina maakten het venster 1489 px breed. Zie de bevindingen in CLAUDE.md
("Passen op elk scherm").

**Meetmethode.** Het Qt-offscreen-platform met een schermconfiguratie in een JSON-bestand
(`QT_QPA_PLATFORM=offscreen:configfile=...`; het pad mag **geen** drive-letter bevatten,
de dubbele punt breekt de parser en Qt crasht met 0xc0000409 — daarom chdir + relatief
pad). Schermmaten in dat bestand zijn *logische* pixels. Twee dingen zonder welke geen
enkel getal klopt: `QT_QPA_FONTDIR=C:\\Windows\\Fonts` (zonder fontmap kent offscreen op
Windows geen enkel lettertype en meet alles twee keer te breed) en de stijl `windows11` +
Segoe UI van het echte platform (offscreen kiest anders Fusion met "Sans Serif").
Gevalideerd tegen het echte platform met verborgen vensters (`WA_DontShowOnScreen`):
verschil ≤ 1%. Per scenario draait een **kindproces**, want het platform leest de
schermconfiguratie één keer bij het opstarten.

Elk venster wordt opgebouwd op een echte testbibliotheek (synthetische video van 40
frames, twee analyses met lange namen, één opname, een lang bibliotheekpad) — de
stress-gevallen zitten dus standaard in de meting.
"""
import json
import os
import subprocess
import sys
import tempfile

HIER = os.path.dirname(os.path.abspath(__file__))

# De ondergrens (zie de docstring): logische schermmaat en taakbalkhoogte.
ONDERGRENS = ("1280x720 logisch (FHD-laptop op 150%)", 1280, 720, 48)
LETTER_STANDAARD = 9.0

# Windows 11-vensterrand op 100%: titelbalk 31 px (nagemeten 29 op dpr 2), 1 px zijkant.
TITELBALK, ZIJRAND = 31, 1

# Het rapport-raster: (naam, logische breedte, logische hoogte, taakbalk).
RASTER = [
    ("1024x768 @100% (beamer)",     1024,  768, 48),
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
LETTERS = (9.0, 10.0, 11.0, 12.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Kindproces: één scherm, alle vensters meten
# ═══════════════════════════════════════════════════════════════════════════
def _meet(naam, breedte, hoogte, taakbalk, letter, uit_pad):
    tmp = tempfile.mkdtemp(prefix="schaats_schermtest_")
    json.dump({"screens": [{"name": naam, "x": 0, "y": 0, "width": breedte,
                            "height": hoogte - taakbalk,
                            "logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1.0}]},
              open(os.path.join(tmp, "scherm.json"), "w"))
    os.chdir(tmp)
    os.environ["QT_QPA_PLATFORM"] = "offscreen:configfile=scherm.json"
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
    # Een lang bibliotheekpad is een van de stress-gevallen (het label onder de startpagina).
    bieb = os.path.join(tmp, "Google Drive", "Gedeelde drives",
                        "Schaatsvereniging IJsster Amersfoort", "Trainers",
                        "Techniekanalyse 2026-2027", "bieb")
    lokaal = os.path.join(tmp, "lokaal")
    os.environ["SKATEANALYSIS_LIBRARY"] = bieb
    os.environ["SKATEANALYSIS_LOCAL"] = lokaal
    os.environ["SCHAATSANALYSE_CPU"] = "1"
    sys.path.insert(0, HIER)

    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt
    app = QApplication([])
    stijl = QStyleFactory.create("windows11")
    if stijl is not None:
        app.setStyle(stijl)
    app.setFont(QFont("Segoe UI", letter))

    import math
    import shutil
    import numpy as np
    import cv2
    import schaats_db
    import skate_gui as G
    from skate_analysis import (FrameResultaat, Landmark, verwerk_afgeleiden,
                                 segmenteer_afzetten, video_info)

    # ── fixture ──────────────────────────────────────────────────────────
    def maak_video(pad, w=640, h=360, n=40, fps=25.0):
        wr = cv2.VideoWriter(pad, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            f = np.full((h, w, 3), 200, np.uint8)
            cv2.circle(f, (w // 2 + i * 3, h // 2), 40, (30, 30, 30), -1)
            wr.write(f)
        wr.release()

    def maak_resultaten(n, fps):
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
            res.append(FrameResultaat(frame_nr=i, time=t, lm=lm, pose_found=True))
        return res

    schaats_db.open_db(bieb)
    schaats_db.open_db(lokaal)
    video = os.path.join(tmp, "Testvideo.mp4")
    maak_video(video)
    info = video_info(video)
    resultaten = maak_resultaten(info.totaal, info.fps)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps)
    events = segmenteer_afzetten(resultaten)
    lang = "Tweede Schaatser met een behoorlijk lange naam"
    sid = schaats_db.maak_schaatser(bieb, "Test Schaatser", 2010)
    sid2 = schaats_db.maak_schaatser(bieb, lang, 2008)
    inst = {"smooth_n": 5, "threshold": 0.015, "smooth_landmarks": True,
            "bocht_overslaan": True, "doel_punt": [0.5, 0.5], "horizon_deg": 0.0,
            "auto_horizon": False, "heavy": False, "deinterlaced": False}
    aid1 = schaats_db.sla_analyse_op(bieb, sid, "Analyse één", video, info, resultaten,
                                     events, "yolo", dict(inst), aangemaakt_door="Tester")
    aid2 = schaats_db.sla_analyse_op(bieb, sid2, "Analyse twee met een lange titel voor de kop",
                                     video, info, resultaten, events, "yolo", dict(inst))
    os.makedirs(os.path.join(bieb, "opnames"), exist_ok=True)
    opname = os.path.join(bieb, "opnames", "Opname.mp4")
    shutil.copy2(video, opname)
    schaats_db.synchroniseer_bronmap(bieb)
    bron = schaats_db.lijst_bronvideos(bieb)[0]
    bron_l = schaats_db.losse_video(bieb, lokaal, video)

    # ── meten ────────────────────────────────────────────────────────────
    scherm = app.primaryScreen().availableGeometry()
    uit = {"scenario": naam, "letter": letter, "stijl": app.style().objectName(),
           "font": app.font().family(), "werkgebied": [scherm.width(), scherm.height()],
           "vensters": []}

    def pomp(n=6):
        for _ in range(n):
            app.processEvents()

    def eff_min(w):
        # Waar Qt het venster nooit onder laat komen: expliciete minimumSize wint van de
        # minimumSizeHint, per as.
        ms, mh = w.minimumSize(), w.minimumSizeHint()
        return [ms.width() if ms.width() > 0 else mh.width(),
                ms.height() if ms.height() > 0 else mh.height()]

    def meet(label, w):
        pomp()
        g = w.geometry()
        uit["vensters"].append({
            "venster": label, "min": eff_min(w),
            "geom": [g.x(), g.y(), g.width(), g.height()],
            "gemaximaliseerd": bool(w.windowState() & Qt.WindowMaximized),
            "volledig_scherm": bool(w.windowState() & Qt.WindowFullScreen)})

    def dialoog(label, maak):
        d = maak()
        d.show()
        meet(label, d)
        d.close()
        d.deleteLater()
        pomp()

    frame = np.full((1080, 1920, 3), 120, np.uint8)
    schaatsers = schaats_db.lijst_schaatsers(bieb)
    dialoog("DoelKiezer", lambda: G.DoelKiezer(frame))
    dialoog("HorizonKiezer", lambda: G.HorizonKiezer(frame))
    dialoog("KalibratieKiezer", lambda: G.KalibratieKiezer(frame))
    dialoog("SchaatserDialog", lambda: G.SchaatserDialog())
    dialoog("NieuweAnalyseDialog", lambda: G.NieuweAnalyseDialog(schaatsers))
    dialoog("BatchAnalyseDialog", lambda: G.BatchAnalyseDialog(schaatsers))
    dialoog("AnalyseKiezer", lambda: G.AnalyseKiezer(bieb))
    dialoog("AnalyseInfoDialog", lambda: G.AnalyseInfoDialog(
        schaats_db.analyse_meta(bieb, aid1), "Test Schaatser"))
    dialoog("FragmentKiezer", lambda: G.FragmentKiezer(
        opname, info, gedaan=[{"start_frame": 2, "eind_frame": 10, "titel": "x"}]))
    dialoog("BekijkVenster (1 video)", lambda: G.BekijkVenster([(bron, info)], "Tester"))
    dialoog("BekijkVenster (2 video's)",
            lambda: G.BekijkVenster([(bron, info), (bron_l, info)], "Tester"))

    mw = G.MainWindow()
    mw.show()
    pomp(10)
    meet("MainWindow: startpagina", mw)
    mw._open_analyse_uit_bibliotheek(aid1)
    pomp(10)
    meet("MainWindow: analyse geopend", mw)
    mw.btn_bewerken.setChecked(True)
    pomp(10)
    meet("MainWindow: analyse, bewerk-modus", mw)
    mw.btn_bewerken.setChecked(False)
    pomp()
    mw._zet_vergelijk_kant(mw.kant_links, aid2, lang)
    mw._zet_vergelijk_kant(mw.kant_rechts, aid2, lang)
    mw.stack.setCurrentWidget(mw.pagina_vergelijk)
    pomp(10)
    meet("MainWindow: vergelijk, 2 lange namen", mw)
    mw.stack.setCurrentWidget(mw.pagina_start)
    pomp(10)
    meet("MainWindow: terug op start (vergelijk gevuld)", mw)
    mw.close()
    pomp(10)

    json.dump(uit, open(uit_pad, "w"))
    app.quit()
    shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Ouderproces: scenario's draaien en beoordelen
# ═══════════════════════════════════════════════════════════════════════════
def meet_scenario(naam, breedte, hoogte, taakbalk, letter):
    """Draait één scenario in een kindproces en geeft de meting terug."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        uit_pad = f.name
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--meet", naam, str(breedte),
             str(hoogte), str(taakbalk), str(letter), uit_pad],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"meting '{naam}' mislukt:\n{proc.stderr[-3000:]}")
        return json.load(open(uit_pad))
    finally:
        try:
            os.remove(uit_pad)
        except OSError:
            pass


def beoordeel(v, werk_b, werk_h):
    """Past venster `v` in het werkgebied? Geeft (past, tekort_breedte, tekort_hoogte,
    reden). Een venster in volledig scherm heeft geen rand."""
    mb, mh = v["min"]
    if v["volledig_scherm"]:
        tb, th = mb - werk_b, mh - werk_h
    else:
        tb, th = mb + 2 * ZIJRAND - werk_b, mh + TITELBALK - werk_h
    if tb > 0 or th > 0:
        return False, max(0, tb), max(0, th), "minimummaat"
    # De werkelijk geopende maat (na zet_venstergrootte), mét rand: `move()` zet de
    # framehoek, dus de titelbalk komt bóven geom.y en de zijranden ernaast.
    x, y, b, h = v["geom"]
    if not (v["volledig_scherm"] or v["gemaximaliseerd"]):
        onder = y + TITELBALK + h + ZIJRAND - werk_h
        rechts = x + b + 2 * ZIJRAND - werk_b
        if onder > 0 or rechts > 0 or x < 0 or y < 0:
            return False, max(0, rechts), max(0, onder), "geopende maat"
    return True, 0, 0, ""


def rapport(meting):
    werk_b, werk_h = meting["werkgebied"]
    print(f"\n=== {meting['scenario']} — werkgebied {werk_b}×{werk_h}, "
          f"{meting['letter']:g} pt {meting['font']} / {meting['stijl']} ===")
    fouten = 0
    for v in meting["vensters"]:
        past, tb, th, reden = beoordeel(v, werk_b, werk_h)
        toestand = ("volledig scherm" if v["volledig_scherm"]
                    else "gemaximaliseerd" if v["gemaximaliseerd"]
                    else f"opent {v['geom'][2]}×{v['geom'][3]}")
        if past:
            marge_h = werk_h - v["min"][1] - (0 if v["volledig_scherm"] else TITELBALK)
            print(f"  ok    {v['venster']:<46} min {v['min'][0]:>4}×{v['min'][1]:<4} "
                  f"{toestand:<18} hoogtemarge {marge_h:>3} px")
        else:
            fouten += 1
            print(f"  FOUT  {v['venster']:<46} min {v['min'][0]:>4}×{v['min'][1]:<4} "
                  f"{toestand:<18} {reden}: {tb} px te breed, {th} px te hoog")
    return fouten


def main(argv):
    if argv[:1] == ["--meet"]:
        naam, b, h, tb, letter, uit_pad = argv[1:7]
        _meet(naam, int(b), int(h), int(tb), float(letter), uit_pad)
        return 0

    if "--alles" in argv:
        letters = LETTERS
        if "--letter" in argv:
            letters = (float(argv[argv.index("--letter") + 1]),)
        totaal = 0
        for letter in letters:
            for naam, b, h, tb in RASTER:
                totaal += rapport(meet_scenario(f"{naam}, {letter:g} pt", b, h, tb, letter))
        print(f"\n{totaal} venster(s) passen niet in het hele raster "
              f"(ondergrens-scenario's uitgezonderd is dat informatief, geen fout).")
        return 0

    naam, b, h, tb = ONDERGRENS
    fouten = rapport(meet_scenario(naam, b, h, tb, LETTER_STANDAARD))
    if fouten:
        print(f"\nFOUT: {fouten} venster(s) passen niet op de afgesproken ondergrens "
              f"({naam}). Zie CLAUDE.md, 'Passen op elk scherm'.")
        return 1
    print(f"\nOK: alle vensters passen op de ondergrens ({naam}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
