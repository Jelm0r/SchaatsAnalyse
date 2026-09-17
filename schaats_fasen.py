"""
Fasen-marker v2 — automatische fase-detectie + correctie-GUI
============================================================
Berekent per slag (straight) automatisch de fasen uit de landmarks volgens de
algoritmische definitie, toont ze in een zijpaneel met tijdstempels en het
efficientie-percentage, en laat je ze per slag corrigeren. Correcties worden
opgeslagen en zijn de trainingsdata voor de automatische detector.

Definities per slag (slag = één afzetgebeurtenis):
  positioning_start  nieuw mes op het ijs (start van de afzetgebeurtenis).
  pushing_start      eerste frame waarop het heupmidden boven het contactpunt van
                     het duwbeen hangt (binnen GEWICHT_FRAC × heupbreedte).
  endpush_start      eerste frame na pushing_start waarop de heupas verder dan
                     ROTATIE_GRADEN gedraaid is t.o.v. de face-on-richting van
                     deze slag (de heupas draait voorbij de rijrichting).
  end_frame          einde van de afzetgebeurtenis (mes verlaat het ijs).

Efficientie-percentage = (endpush_start − pushing_start) / (end_frame − positioning_start),
dus de duur van de efficiënte duwfase als deel van de hele slag.

Bediening:
  ← / → één frame, ↑ / ↓ tien frames, Spatie afspelen/pauze
  1 / 2 / 3 / 4  faseovergang van de HUIDIGE slag op dit frame zetten (corrigeert)
  N    nieuwe slag starten op dit frame (handmatig)
  U    laatste correctie in de slag ongedaan maken
  S    opslaan (auto-fasen mét correcties, elke slag met 'auto'/'corrected'-vlag)
  Klik in het zijpaneel  naar die slag springen
  Esc/Q  afsluiten (S eerst!)

CLI:
    python schaats_fasen.py --input video.mov --npz analyse.npz [--uit fasen.json]
                             [--rotatie 12.0] [--gewicht-frac 0.5] [--dump] [--rook png]
"""
import argparse
import json
import math
import sys

import cv2

import schaats_analyse as sa
import schaats_techniek

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
    'positioning_start': (0, 90, 255),
    'pushing_start': (0, 200, 0),
    'endpush_start': (255, 120, 0),
    'end_frame': (0, 0, 255),
}
BAND_KLEUR = {
    'positioning': (0, 90, 255),
    'duw': (0, 200, 0),
    'eind': (255, 120, 0),
}


def hipaxis_hoek(lm):
    """Hoek van de heupas in het beeldvlak; 0 = verticale heuplijn in beeld."""
    lh, rh = lm['l_heup'], lm['r_heup']
    return math.degrees(math.atan2(rh[0] - lh[0], rh[1] - lh[1]))


def hoek_verschil(a, b):
    """Circulair verschil in graden (−180..180)."""
    return (a - b + 180.0) % 360.0 - 180.0


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


def automatische_fasen(resultaten, info, rotatie_graden=12.0, gewicht_frac=0.5):
    """Per afzetgebeurtenis de vier fasegrenzen uit de algoritmische definitie."""
    events = sa.segmeneer_afzetten(resultaten)
    slagen = []
    for ev in events:
        kant = 'l' if ev.been == 'links' else 'r'
        meetbaar = [f for f in range(ev.start_frame, min(ev.eind_frame + 1, len(resultaten)))
                    if resultaten[f].lm_data is not None]
        if not meetbaar:
            continue
        # face-on-baseline: eerste meetbare frames van deze slag
        basis = sorted(hipaxis_hoek(resultaten[f].lm_data) for f in meetbaar[:6])
        basis_hoek = basis[len(basis) // 2]
        duw_start = None
        eind_start = None
        for f in meetbaar:
            lm = resultaten[f].lm_data
            hiel, teen = lm[f'{kant}_hiel'], lm[f'{kant}_teen']
            contact = ((hiel[0] + teen[0]) / 2.0, (hiel[1] + teen[1]) / 2.0)
            heupm = ((lm['l_heup'][0] + lm['r_heup'][0]) / 2.0,
                     (lm['l_heup'][1] + lm['r_heup'][1]) / 2.0)
            heupbreedte = abs(lm['r_heup'][0] - lm['l_heup'][0]) or 1.0
            boven = abs(heupm[0] - contact[0]) <= gewicht_frac * heupbreedte
            if duw_start is None and boven:
                duw_start = f
            if duw_start is not None and abs(hoek_verschil(hipaxis_hoek(lm), basis_hoek)) >= rotatie_graden:
                eind_start = f
                break
        slagen.append({
            'been': ev.been,
            'positioning_start': ev.start_frame,
            'pushing_start': duw_start if duw_start is not None else ev.start_frame,
            'endpush_start': eind_start if eind_start is not None else ev.eind_frame,
            'end_frame': ev.eind_frame,
            'corrected': False,
        })
    return slagen


def duw_pct(slag, fps):
    """Efficientie-percentage: duur van de duwfase als deel van de hele slag (in tijd)."""
    try:
        a = slag['pushing_start']
        b = slag['endpush_start']
        c = slag['positioning_start']
        d = slag['end_frame']
        if None in (a, b, c, d) or d <= c:
            return None
        return round(100.0 * (b - a) / (d - c), 1)
    except (KeyError, TypeError):
        return None


def teken_hulplijnen(frame, r, slag=None):
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
    cv2.line(frame, lh, rh, (0, 255, 255), 2)
    hoek = math.atan2(rh[1] - lh[1], rh[0] - lh[0]) + math.pi / 2
    dx, dy = math.cos(hoek) * L, math.sin(hoek) * L
    cv2.line(frame, (int(mid[0] - dx), int(mid[1] - dy)), (int(mid[0] + dx), int(mid[1] + dy)),
             (255, 255, 0), 2)
    if getattr(r, 'heup_hist', None) and len(r.heup_hist) >= 2:
        (x0, y0), (x1, y1) = r.heup_hist[0], r.heup_hist[-1]
        d = math.hypot(x1 - x0, y1 - y0)
        if d > 3:
            s = L * 1.5 / d
            cv2.line(frame, (int(mid[0] - (x1 - x0) * s), int(mid[1] - (y1 - y0) * s)),
                     (int(mid[0] + (x1 - x0) * s), int(mid[1] + (y1 - y0) * s)),
                     (0, 255, 0), 2)

    for kant, y in (('l', 30), ('r', 58)):
        u = schaats_techniek.kantelhoek(lm, kant, 0.0)
        if u:
            cv2.putText(frame, f'{"L" if kant == "l" else "R"} kantel {u[0]:+.1f}',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if kant == 'l' else (0, 180, 255), 2)
    return frame


def fase_banden(fr, slagen, huidige):
    """Kleurband bovenin per fase van de huidige slag + grenslijnen door het beeld."""
    h = fr.shape[0]
    slag = huidige if huidige is not None else None
    for i, s in enumerate(slagen):
        for sleutel in FASEN_KEYS.values():
            if sleutel in s and s.get('corrected'):
                kleur = FASEN_KLEUR[sleutel]
            elif sleutel in s:
                kleur = tuple(int(c * 0.6) for c in FASEN_KLEUR[sleutel])
            else:
                continue
            cv2.line(fr, (s[sleutel], 0), (s[sleutel], h), kleur, 2)
    if slag:
        grenzen = sorted((slag[k], FASEN_KLEUR[k]) for k in FASEN_KEYS.values() if k in slag)
        for i in range(len(grenzen) - 1):
            f0, k0 = grenzen[i]
            f1, _ = grenzen[i + 1]
            fase = FASEN_KEYS['1' if k0 == FASEN_KLEUR['positioning_start'] else '2'] if False else None
        # fasebanden: pos=oranje, duw=groen, eind=blauw
        if 'positioning_start' in slag and 'pushing_start' in slag:
            cv2.rectangle(fr, (slag['positioning_start'], 0), (slag['pushing_start'], 14),
                          BAND_KLEUR['positioning'], -1)
        if 'pushing_start' in slag and 'endpush_start' in slag:
            cv2.rectangle(fr, (slag['pushing_start'], 0), (slag['endpush_start'], 14),
                          BAND_KLEUR['duw'], -1)
        if 'endpush_start' in slag and 'end_frame' in slag:
            cv2.rectangle(fr, (slag['endpush_start'], 0), (slag['end_frame'], 14),
                          BAND_KLEUR['eind'], -1)
    return fr


def main_rook(npz_pad, video_pad, uit_png, frame_nr=100, rotatie=12.0, gewicht_frac=0.5):
    info, resultaten = laad_context(npz_pad)
    slagen = automatische_fasen(resultaten, info, rotatie, gewicht_frac)
    print(f'[AUTO] {len(slagen)} slagen:')
    for i, s in enumerate(slagen):
        print(f"  slag {i}: {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
              f"duw f{s['pushing_start']}–f{s['endpush_start']}  efficientie {duw_pct(s, info.fps)}%")
    r = resultaten[frame_nr] if frame_nr < len(resultaten) else None
    cap = cv2.VideoCapture(video_pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ok, fr = cap.read()
    cap.release()
    huidige = next((s for s in slagen if s['positioning_start'] <= frame_nr <= s['end_frame']), None)
    fr = fase_banden(teken_hulplijnen(fr, r), slagen, huidige)
    cv2.imwrite(uit_png, fr)
    print(f'[ROOK] {uit_png}')


def run_gui(args):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget, QListWidgetItem,
                                   QLabel, QMainWindow, QStatusBar, QVBoxLayout, QWidget)

    info, resultaten = laad_context(args.npz)
    slagen = automatische_fasen(resultaten, info, args.rotatie, args.gewicht_frac)
    for s in slagen:
        s['auto'] = True

    def slag_op_frame(f):
        huid = None
        for s in slagen:
            if s['positioning_start'] <= f <= s['end_frame']:
                huid = s
                break
        return huid

    app = QApplication(sys.argv)

    class Venster(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cap = cv2.VideoCapture(args.input)
            self.f = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.volgende)
            self.video_label = QLabel(' laden…')
            self.video_label.setAlignment(Qt.AlignCenter)
            self.lijst = QListWidget()
            self.lijst.itemClicked.connect(self.klik_slag)
            self.lijst.setFixedWidth(420)
            lay = QHBoxLayout()
            lay.addWidget(self.video_label, 1)
            lay.addWidget(self.lijst)
            w = QWidget()
            w.setLayout(lay)
            self.setCentralWidget(w)
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.setWindowTitle('Fasen-marker — 1/2/3/4 corrigeert, N nieuwe slag, U ongedaan, S opslaan')
            self.resize(1500, 860)
            self.vul_lijst()
            self.toon()

        def vul_lijst(self):
            self.lijst.blockSignals(True)
            self.lijst.clear()
            for i, s in enumerate(slagen):
                pct = duw_pct(s, info.fps)
                vlag = ' (gecorrigeerd)' if s.get('corrected') else ''
                self.lijst.addItem(QListWidgetItem(
                    f"slag {i:>2} {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
                    f"duw f{s['pushing_start']}–f{s['endpush_start']}  "
                    f"efficientie {pct if pct is not None else '—'}%{vlag}"))
            self.lijst.blockSignals(False)

        def klik_slag(self, item):
            self.f = slagen[self.lijst.row(item)]['positioning_start']
            self.timer.stop()
            self.toon()

        def toon(self):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
            ok, fr = self.cap.read()
            if not ok:
                self.timer.stop()
                return
            r = resultaten[self.f] if self.f < len(resultaten) else None
            fr = teken_hulplijnen(fr, r)
            huidige = slag_op_frame(self.f)
            fr = fase_banden(fr, slagen, huidige)
            schaal = (self.video_label.height() or 800) / fr.shape[0]
            klein = cv2.resize(fr, (int(fr.shape[1] * schaal), int(fr.shape[0] * schaal)))
            rgb = cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
            self.video_label.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                                                rgb.strides[0], QImage.Format_RGB888)))
            # huidige slag in het paneel markeren
            if huidige is not None:
                i = slagen.index(huidige)
                self.lijst.blockSignals(True)
                self.lijst.setCurrentRow(i)
                self.lijst.blockSignals(False)
            pct = duw_pct(huidige, info.fps) if huidige else None
            self.status.showMessage(f'frame {self.f}/{info.totaal} ({self.f / (info.fps or 30):.2f}s)   '
                                    f'slag: {"—" if huidige is None else slagen.index(huidige)}   '
                                    f'efficientie: {"—" if pct is None else str(pct) + "%"}')

        def volgende(self):
            if self.f < info.totaal - 1:
                self.f += 1
                self.toon()
            else:
                self.timer.stop()

        def huidige_of_geen(self):
            s = slag_op_frame(self.f)
            if s is None:
                self.status.showMessage('geen slag op dit frame — N om er een te starten', 3000)
            return s

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
                s = {'been': '?', 'positioning_start': self.f,
                     'pushing_start': self.f, 'endpush_start': self.f,
                     'end_frame': self.f, 'corrected': True}
                slagen.append(s)
                slagen.sort(key=lambda x: x['positioning_start'])
                self.vul_lijst()
                self.toon()
            elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
                s = self.huidige_of_geen()
                if s is not None:
                    s[FASEN_KEYS[ev.text()]] = self.f
                    s['corrected'] = True
                    slagen.sort(key=lambda x: x['positioning_start'])
                    self.vul_lijst()
                    self.toon()
            elif k == Qt.Key_U:
                s = slag_op_frame(self.f)
                if s is not None and s.get('corrected'):
                    vers = automatische_fasen(resultaten, info, args.rotatie, args.gewicht_frac)
                    for v in vers:
                        if v['positioning_start'] == s['positioning_start']:
                            s.update(v)
                            s['auto'] = True
                            break
                    self.vul_lijst()
                    self.toon()
            elif k == Qt.Key_S:
                data = {'video': args.input, 'fps': info.fps,
                        'rotatie_graden': args.rotatie, 'gewicht_frac': args.gewicht_frac,
                        'strokes': slagen}
                with open(args.uit, 'w') as fh:
                    json.dump(data, fh, indent=1)
                self.status.showMessage(f'opgeslagen: {args.uit}', 5000)
            elif k in (Qt.Key_Escape, Qt.Key_Q):
                self.close()

    venster = Venster()
    venster.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasen-marker v2')
    parser.add_argument('--input', required=True)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--uit', default='fasen.json')
    parser.add_argument('--rotatie', type=float, default=12.0,
                        help='heuprotatie in graden die eind-duw triggert (standaard 12.0)')
    parser.add_argument('--gewicht-frac', type=float, default=0.5,
                        help='heupmidden mag zo veel × heupbreedte van het contactpunt zitten (0.5)')
    parser.add_argument('--dump', action='store_true', help='alleen auto-fasen printen, geen GUI')
    parser.add_argument('--rook', default=None, help='rooktest: render frame naar PNG, geen GUI')
    parser.add_argument('--rook-frame', type=int, default=100)
    args = parser.parse_args()
    if args.dump:
        info, resultaten = laad_context(args.npz)
        for i, s in enumerate(automatische_fasen(resultaten, info, args.rotatie, args.gewicht_frac)):
            print(f"slag {i:>2} {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
                  f"duw f{s['pushing_start']}–f{s['endpush_start']}  efficientie {duw_pct(s, info.fps)}%")
        sys.exit(0)
    if args.rook:
        main_rook(args.npz, args.input, args.rook, args.rook_frame, args.rotatie, args.gewicht_frac)
        sys.exit(0)
    run_gui(args)
