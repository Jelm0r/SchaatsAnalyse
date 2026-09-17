"""
Fasen-marker (annotatie-tool)
=============================
Handmatig markeer je per slag de fase-overgangen, voor de ontwikkeling van de
techniek-metriek. Je stapt frame voor frame en drukt op de fasenknop op het moment
dat de overgang valt:

  1 = start POSITIONERING   — nieuw mes op het ijs, nog inefficiënt: het lichaamsgewicht
                              staat nog niet over het duwbeen (met netto binnenrand —
                              definitie volgt later).
  2 = start DUWFASE         — gewicht over het duwbeen; efficiënt duwen begint.
  3 = start EIND-DUW        — de heupas draait voorbij de rijrichting: het loodrechte
                              "vooruit" van de lijn tussen de heupbenen draait voorbij
                              de richtingslijn van de reis; vanaf dan staat het
                              zwaartepunt niet meer goed boven de duw.
  4 = einde SLAG            — het schaatscontact verlaat het ijs.

Bediening:
  ← / →    één frame           ↑ / ↓   tien frames      Spatie  afspelen/pauze
  1 / 2 / 3 / 4  faseovergang markeren (op het huidige frame, in de huidige slag)
  N        nieuwe slag starten op dit frame
  U        laatste markering ongedaan maken (of de vorige slag terugopenen)
  S        opslaan naar JSON                    Esc/Q  afsluiten (S eerst!)

De JSON bevat per slag de vier framenummers. Fase-intervallen zijn de stukken ertussen:
positionering = [positioning_start, pushing_start), duw = [pushing_start, endpush_start),
eind-duw = [endpush_start, end_frame].

Hulplijnen op het beeld (gids, geen oordeel):
  geel   heupas (lijn tussen de heupbenen)
  cyaan  loodrecht op de heupas (de "vooruit"-as, twee kanten getekend)
  groen  rijrichting, uit de baan van het heupmidden over ±12 frames
  bovenin de ruwe kantelhoek van beide schaatsen (zie schaats_techniek.py)

CLI:
    python schaats_fasen.py --input video.mov --npz analyse.npz [--uit fasen.json] [--rook png]
"""
import argparse
import json
import math
import sys

import cv2

import schaats_analyse as sa
import schaats_techniek

# lm_data-sleutels per landmark-index (schouders zitten níet in lm_data)
NAAM = {
    sa.L_HIP: 'l_heup', sa.R_HIP: 'r_heup',
    sa.L_KNEE: 'l_knie', sa.R_KNEE: 'r_knie',
    sa.L_ANKLE: 'l_enkel', sa.R_ANKLE: 'r_enkel',
    sa.L_HEEL: 'l_hiel', sa.R_HEEL: 'r_hiel',
    sa.L_TOE: 'l_teen', sa.R_TOE: 'r_teen',
}
SKEL = [
    ('l_heup', 'r_heup'), ('l_heup', 'l_knie'), ('r_heup', 'r_knie'),
    ('l_knie', 'l_enkel'), ('r_knie', 'r_enkel'),
    ('l_hiel', 'l_teen'), ('r_hiel', 'r_teen'),
]
FASEN_KEYS = {
    '1': 'positioning_start',
    '2': 'pushing_start',
    '3': 'endpush_start',
    '4': 'end_frame',
}
FASEN_KLEUR = {
    'positioning_start': (0, 90, 255),     # oranje (BGR)
    'pushing_start': (0, 200, 0),          # groen
    'endpush_start': (255, 120, 0),        # blauw
    'end_frame': (0, 0, 255),              # rood
}


def teken_hulplijnen(frame, r, heup_hist=None):
    """Mini-skelet + heupas + loodrechte vooruit-as + rijrichting + kantelhoeken."""
    lm = r.lm_data if r is not None else None
    if lm is None:
        return frame

    for a, b in SKEL:
        pa, pb = lm.get(a), lm.get(b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, (255, 255, 255), 1)
            for p in (pa, pb):
                cv2.circle(frame, p, 3, (255, 255, 255), -1)

    lh, rh = lm['l_heup'], lm['r_heup']
    mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    L = math.hypot(rh[0] - lh[0], rh[1] - lh[1]) or 1.0

    # heupas (geel) + loodrechte vooruit-as (cyaan, beide richtingen)
    cv2.line(frame, lh, rh, (0, 255, 255), 2)
    hoek = math.atan2(rh[1] - lh[1], rh[0] - lh[0]) + math.pi / 2
    dx, dy = math.cos(hoek) * L, math.sin(hoek) * L
    cv2.line(frame, (int(mid[0] - dx), int(mid[1] - dy)), (int(mid[0] + dx), int(mid[1] + dy)),
             (255, 255, 0), 2)

    # rijrichting (groen) uit de heupmidden-baan over het ±12-frames-venster
    if heup_hist and len(heup_hist) >= 2:
        (x0, y0), (x1, y1) = heup_hist[0], heup_hist[-1]
        d = math.hypot(x1 - x0, y1 - y0)
        if d > 3:
            s = L * 1.5 / d
            cv2.line(frame, (int(mid[0] - (x1 - x0) * s), int(mid[1] - (y1 - y0) * s)),
                     (int(mid[0] + (x1 - x0) * s), int(mid[1] + (y1 - y0) * s)),
                     (0, 255, 0), 2)

    # ruwe kantelhoek van beide schaatsen bovenin
    for kant, y in (('l', 30), ('r', 58)):
        u = schaats_techniek.kantelhoek(lm, kant, 0.0)   # geen dode zone: ruwe hoek
        if u:
            tekst = f'{"L" if kant == "l" else "R"} kantel {u[0]:+.1f}'
            cv2.putText(frame, tekst, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if kant == 'l' else (0, 180, 255), 2)
    return frame


def laad_context(npz_pad):
    info, resultaten = sa.laad_landmarks(npz_pad)
    sa.verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, 5, 0.015)
    n = len(resultaten)
    for i, r in enumerate(resultaten):
        pts = []
        for j in range(max(0, i - 12), min(n, i + 13)):
            lm = resultaten[j].lm_data
            if lm is not None:
                pts.append(((lm['l_heup'][0] + lm['r_heup'][0]) / 2.0,
                            (lm['l_heup'][1] + lm['r_heup'][1]) / 2.0))
        r.heup_hist = pts if len(pts) >= 2 else None
    return info, resultaten


def main_rook(npz_pad, video_pad, uit_png, frame_nr=100):
    """Rooktest zonder GUI: rendert één frame met hulplijnen naar een PNG."""
    info, resultaten = laad_context(npz_pad)
    r = resultaten[frame_nr] if frame_nr < len(resultaten) else None
    cap = cv2.VideoCapture(video_pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'frame {frame_nr} onleesbaar')
    cv2.imwrite(uit_png, teken_hulplijnen(fr, r, r.heup_hist if r else None))
    print(f'[ROOK] {uit_png}')


def run_gui(args):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QVBoxLayout, QWidget

    info, resultaten = laad_context(args.npz)

    class Venster(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cap = cv2.VideoCapture(args.input)
            self.f = 0
            self.slag = {}
            self.slagen = []
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.volgende)
            self.label = QLabel(' laden…')
            self.label.setAlignment(Qt.AlignCenter)
            lay = QVBoxLayout()
            lay.addWidget(self.label)
            w = QWidget()
            w.setLayout(lay)
            self.setCentralWidget(w)
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.setWindowTitle('Fasen-marker — 1/2/3/4 markeert, ←/→ frame, N nieuwe slag, U ongedaan, S opslaan')
            self.resize(1280, 820)
            self.label.resize(self.label.width(), 760)
            self.toon()

        def toon(self):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
            ok, fr = self.cap.read()
            if not ok:
                self.timer.stop()
                return
            r = resultaten[self.f] if self.f < len(resultaten) else None
            if r is not None:
                fr = teken_hulplijnen(fr, r, r.heup_hist)
            # eerder gemarkeerde grenzen + huidige slag tekenen
            for slag in self.slagen + [self.slag]:
                for toets, sleutel in FASEN_KEYS.items():
                    if sleutel in slag:
                        kleur = FASEN_KLEUR[sleutel]
                        cv2.line(fr, (slag[sleutel], 0), (slag[sleutel], fr.shape[0]), kleur, 2)
            schaal = (self.label.height() or 760) / fr.shape[0]
            klein = cv2.resize(fr, (int(fr.shape[1] * schaal), int(fr.shape[0] * schaal)))
            rgb = cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
            self.label.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                                          rgb.strides[0], QImage.Format_RGB888)))
            marks = ', '.join(f'{k[0]}={v}' for k, v in sorted(self.slag.items(), key=lambda kv: kv[1])) or '—'
            self.status.showMessage(f'frame {self.f}/{info.totaal} ({self.f / (info.fps or 30):.2f}s)   '
                                    f'huidige slag: {marks}   slagen: {len(self.slagen)}')

        def volgende(self):
            if self.f < info.totaal - 1:
                self.f += 1
                self.toon()
            else:
                self.timer.stop()

        def keyPressEvent(self, ev):
            k = ev.key()
            if k == Qt.Key_Left:
                self.f = max(0, self.f - 1)
                self.toon()
            elif k == Qt.Key_Right:
                self.f = min(self.f + 1, info.totaal - 1)
                self.toon()
            elif k == Qt.Key_Up:
                self.f = max(0, self.f - 10)
                self.toon()
            elif k == Qt.Key_Down:
                self.f = min(self.f + 10, info.totaal - 1)
                self.toon()
            elif k == Qt.Key_Space:
                if self.timer.isActive():
                    self.timer.stop()
                else:
                    self.timer.start(int(1000 / (info.fps or 30)))
            elif k == Qt.Key_N:
                if self.slag:
                    self.slagen.append(dict(self.slag))
                self.slag = {'positioning_start': self.f}
                self.toon()
            elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
                self.slag[FASEN_KEYS[ev.text()]] = self.f
                self.toon()
            elif k == Qt.Key_U:
                if self.slag:
                    laatst = max(self.slag.items(), key=lambda kv: kv[1])
                    del self.slag[laatst[0]]
                elif self.slagen:
                    self.slag = self.slagen.pop()
                self.toon()
            elif k == Qt.Key_S:
                data = {'video': args.input, 'fps': info.fps,
                        'strokes': self.slagen + ([dict(self.slag)] if self.slag else [])}
                with open(args.uit, 'w') as fh:
                    json.dump(data, fh, indent=1)
                self.status.showMessage(f'opgeslagen: {args.uit}', 5000)
            elif k in (Qt.Key_Escape, Qt.Key_Q):
                self.close()

    app = QApplication(sys.argv)
    venster = Venster()
    venster.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasen-marker')
    parser.add_argument('--input', required=True)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--uit', default='fasen.json')
    parser.add_argument('--rook', default=None, help='rooktest: render frame 100 naar PNG, geen GUI')
    parser.add_argument('--rook-frame', type=int, default=100)
    args = parser.parse_args()
    if args.rook:
        main_rook(args.npz, args.input, args.rook, args.rook_frame)
        sys.exit(0)
    run_gui(args)
