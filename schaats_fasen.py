"""
Fasen-marker v3 — automatische fase-detectie + correctie-GUI
============================================================
Berekent per slag (straight) automatisch de fasen uit de landmarks volgens de
algoritmische definitie, toont ze in een zijpaneel (tijdstempels + efficientie-
percentage) en op een tijdlijn onder de video, en laat je slagen corrigeren.
Correcties worden opgeslagen en zijn de trainingsdata voor de detector.

Definities per slag (slag = één afzetgebeurtenis):
  positioning_start  nieuw mes op het ijs (start van de afzetgebeurtenis).
  pushing_start      frame waarop het heupmidden het dichtst boven het contactpunt
                     van het duwbeen hangt.
  endpush_start      eerste frame na pushing_start waarop de heupas verder dan
                     --rotatie graden gedraaid is t.o.v. de face-on-richting;
                     fallback: het rotatiepiek.
  end_frame          einde van de afzetgebeurtenis (mes verlaat het ijs).

Efficientie-percentage = duur van de duwfase (pushing→endpush) als deel van de slag.

UI:
  links    video met hulplijnen (heupas geel, vooruit-as cyaan, rijrichting groen,
           kantelhoeken bovenin) en daaronder de TIJDLIJN: elke slag als
           oranje/groen/blauwe segmenten (positionering/duw/eind-duw), gedimde
           kleuren = automatische waarde, volle kleur = gecorrigeerd, witte
           speelkop = huidige frame, gele kader = slag in bewerking.
  rechts   slaglijst (klik = springen), knoppen Redefine/Nieuw/Ongedaan/Opslaan
           en de sneltoetsen-legenda.

Bediening:
  ← / → één frame, ↑ / ↓ tien frames, Spatie afspelen/pauze
  klik in de slaglijst of op de tijdlijn   naar die slag/frame springen
  Redefine (E)  slag-in-bewerking aan/uit (geel kader); 1/2/3/4 zetten dan de
                fasegrenzen van DIE slag op het huidige frame
  N  nieuwe slag op dit frame, U ongedaan, S opslaan, Esc/Q afsluiten

CLI:
    python schaats_fasen.py --input video.mov --npz analyse.npz [--uit fasen.json]
                             [--rotatie 10.0] [--dump] [--rook png]
"""
import argparse
import json
import math
import sys

import cv2
import numpy as np

import schaats_analyse as sa
import schaats_techniek

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
FASEN_NAAM = {
    'positioning_start': 'positionering',
    'pushing_start': 'duwfase',
    'endpush_start': 'eind-duw',
    'end_frame': 'einde slag',
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
LEGENDA = """Sneltoetsen
  ← / →    één frame
  ↑ / ↓    tien frames
  Spatie   afspelen / pauze
  1 / 2 / 3 / 4   fasegrens zetten (in Redefine-modus)
  E        Redefine-modus aan/uit voor de geselecteerde slag
  N        nieuwe slag op dit frame
  U        ongedaan maken
  S        opslaan
  Esc / Q  afsluiten

Fasen (per slag)
  oranje   positionering (inefficiënt)
  groen    duwfase (efficiënt: gewicht boven duwbeen,
           heupas nog in de rijrichting)
  blauw    eind-duw (heupas gedraaid, inefficiënt)
Tijdlijn: gedimd = automatisch, vol = gecorrigeerd,
witte speelkop = huidige frame, geel kader = bewerken"""


def hipaxis_hoek(lm):
    lh, rh = lm['l_heup'], lm['r_heup']
    return math.degrees(math.atan2(rh[0] - lh[0], rh[1] - lh[1]))


def hoek_verschil(a, b):
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


def automatische_fasen(resultaten, info, rotatie_graden=10.0):
    """Per afzetgebeurtenis de vier fasegrenzen uit de algoritmische definitie."""
    events = sa.segmenteer_afzetten(resultaten)
    slagen = []
    for ev in events:
        kant = 'l' if ev.been == 'links' else 'r'
        meetbaar = [f for f in range(ev.start_frame, min(ev.eind_frame + 1, len(resultaten)))
                    if resultaten[f].lm_data is not None]
        if not meetbaar:
            continue
        basis_hoek = hipaxis_hoek(resultaten[meetbaar[0]].lm_data)
        afstanden = []
        rotaties = []
        for f in meetbaar:
            lm = resultaten[f].lm_data
            hiel, teen = lm[f'{kant}_hiel'], lm[f'{kant}_teen']
            contact = ((hiel[0] + teen[0]) / 2.0, (hiel[1] + teen[1]) / 2.0)
            heupm_x = (lm['l_heup'][0] + lm['r_heup'][0]) / 2.0
            afstanden.append(abs(heupm_x - contact[0]))
            rotaties.append(abs(hoek_verschil(hipaxis_hoek(lm), basis_hoek)))
        duw_start = meetbaar[afstanden.index(min(afstanden))]
        na_duw = [(f, rot) for f, rot in zip(meetbaar, rotaties) if f >= duw_start]
        gepasseerd = next((f for f, rot in na_duw if rot >= rotatie_graden), None)
        eind_start = gepasseerd if gepasseerd is not None else (
            max(na_duw, key=lambda fr: fr[1])[0] if na_duw else ev.eind_frame)
        slagen.append({
            'been': ev.been,
            'positioning_start': ev.start_frame,
            'pushing_start': duw_start,
            'endpush_start': eind_start,
            'end_frame': ev.eind_frame,
            'corrected': False,
        })
    return slagen


def duw_pct(slag, fps=None):
    try:
        a, b, c, d = slag['pushing_start'], slag['endpush_start'], slag['positioning_start'], slag['end_frame']
        if None in (a, b, c, d) or d <= c:
            return None
        return round(100.0 * (b - a) / (d - c), 1)
    except (KeyError, TypeError):
        return None


def fase_van(slag, f):
    """Fasenaam waarin frame f zich bevindt binnen deze slag (of None)."""
    if slag is None:
        return None
    if f < slag.get('pushing_start', slag.get('positioning_start', 0)):
        return 'positionering'
    if f < slag.get('endpush_start', 0):
        return 'duw'
    if f <= slag.get('end_frame', 0):
        return 'eind'
    return None


def teken_fase_banner(fr, fase, bewerken=False, gezet=0, tekst_extra=None):
    """Grote kleurenbalk bovenin: in welke fase zit dit frame / wat definieer je."""
    if fase is None:
        return fr
    kleur = BAND_KLEUR['positioning' if fase == 'positionering' else fase]
    tekst = fase.upper()
    if bewerken:
        tekst += f'  (DEFINIEER {gezet}/4)'
    if tekst_extra:
        tekst += '  — ' + tekst_extra
    w = fr.shape[1]
    cv2.rectangle(fr, (w // 2 - 380, 10), (w // 2 + 380, 96), kleur, -1)
    cv2.putText(fr, tekst, (w // 2 - 360, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 5)
    return fr


def teken_hulplijnen(frame, r):
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


def tijdlijn_afbeelding(slagen, totaal, huidige_i=None, bewerk_i=None,
                        speelkop=None, w=1200, h=36):
    """Tijdlijn: alle slagen als fase-segmenten, speelkop, bewerk-kader."""
    img = np.full((h, w, 3), 38, np.uint8)

    def x(f):
        return int(round(f / max(1, totaal - 1) * (w - 1)))

    for i, s in enumerate(slagen):
        a, b = s.get('positioning_start'), s.get('end_frame')
        if a is None or b is None:
            continue
        dim = 0.5 if not s.get('corrected') else 1.0
        for k0, k1, base in (('positioning_start', 'pushing_start', BAND_KLEUR['positioning']),
                             ('pushing_start', 'endpush_start', BAND_KLEUR['duw']),
                             ('endpush_start', 'end_frame', BAND_KLEUR['eind'])):
            if k0 in s and k1 in s:
                kleur = tuple(int(c * dim) for c in base)
                x0, x1 = x(s[k0]), x(s[k1])
                cv2.rectangle(img, (x0, 2), (max(x0 + 1, x1), h - 6), kleur, -1)
        if i == bewerk_i:
            cv2.rectangle(img, (x(a), 0), (x(b), h - 1), (0, 255, 255), 2)
        elif i == huidige_i:
            cv2.rectangle(img, (x(a), 0), (x(b), h - 1), (255, 255, 255), 1)
    if speelkop is not None:
        xp = x(speelkop)
        cv2.line(img, (xp, 0), (xp, h), (255, 255, 255), 1)
        cv2.circle(img, (xp, h // 2), 4, (255, 255, 255), -1)
    return img


def main_rook(npz_pad, video_pad, uit_png, frame_nr=100, rotatie=10.0):
    info, resultaten = laad_context(npz_pad)
    slagen = automatische_fasen(resultaten, info, rotatie)
    print(f'[AUTO] {len(slagen)} slagen:')
    for i, s in enumerate(slagen):
        print(f"  slag {i}: {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
              f"duw f{s['pushing_start']}–f{s['endpush_start']}  efficientie {duw_pct(s)}%")
    r = resultaten[frame_nr] if frame_nr < len(resultaten) else None
    cap = cv2.VideoCapture(video_pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'frame {frame_nr} onleesbaar')
    fr = teken_hulplijnen(fr, r)
    lijn = tijdlijn_afbeelding(slagen, info.totaal, speelkop=frame_nr, w=fr.shape[1])
    combined = np.vstack([fr, lijn])
    cv2.imwrite(uit_png, combined)
    print(f'[ROOK] {uit_png}')


def run_gui(args):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget,
                                   QListWidgetItem, QMainWindow, QPushButton,
                                   QStatusBar, QVBoxLayout, QWidget)

    info, resultaten = laad_context(args.npz)
    slagen = automatische_fasen(resultaten, info, args.rotatie)

    def slag_index_op_frame(f):
        for i, s in enumerate(slagen):
            if s['positioning_start'] <= f <= s['end_frame']:
                return i
        return None

    app = QApplication(sys.argv)

    class Venster(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cap = cv2.VideoCapture(args.input)
            self.f = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.volgende)
            self.bewerk_i = None
            self.bewerk_fase = 0   # 0..3: welke fasegrens je aan het definiëren bent

            self.video_label = QLabel(' laden…')
            self.video_label.setAlignment(Qt.AlignCenter)
            self.tijd_label = QLabel()
            self.tijd_label.setFixedHeight(40)
            self.lijst = QListWidget()
            self.lijst.itemClicked.connect(self.klik_slag)
            self.lijst.setFixedWidth(430)

            self.fase_label = QLabel('geen fase')
            self.fase_label.setAlignment(Qt.AlignCenter)
            self.fase_label.setFixedHeight(46)
            self.fase_label.setStyleSheet('font-size: 20px; font-weight: bold; color: #222; background: #555;')

            self.b_redefine = QPushButton('Redefine (E)')
            self.b_redefine.clicked.connect(self.toggle_bewerk)
            self.b_new = QPushButton('Nieuw (N)')
            self.b_new.clicked.connect(self.nieuwe_slag)
            self.b_undo = QPushButton('Ongedaan (U)')
            self.b_undo.clicked.connect(self.ongedaan)
            self.b_save = QPushButton('Opslaan (S)')
            self.b_save.clicked.connect(self.opslaan)
            knoppen = QHBoxLayout()
            for b in (self.b_redefine, self.b_new, self.b_undo, self.b_save):
                knoppen.addWidget(b)

            legend = QLabel(LEGENDA)
            legend.setStyleSheet('font-size: 12px; color: #ccc;')

            rechts = QVBoxLayout()
            rechts.addWidget(self.fase_label)
            rechts.addWidget(self.lijst, 1)
            rechts.addLayout(knoppen)
            rechts.addWidget(legend)
            rechts_w = QWidget()
            rechts_w.setLayout(rechts)

            links = QVBoxLayout()
            links.addWidget(self.video_label, 1)
            links.addWidget(self.tijd_label)
            links_w = QWidget()
            links_w.setLayout(links)

            top = QHBoxLayout()
            top.addWidget(links_w, 1)
            top.addWidget(rechts_w)
            w = QWidget()
            w.setLayout(top)
            self.setCentralWidget(w)
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.setWindowTitle('Fasen-marker v3')
            self.resize(1550, 900)
            self.vul_lijst()
            self.toon()

        def vul_lijst(self):
            self.lijst.blockSignals(True)
            self.lijst.clear()
            for i, s in enumerate(slagen):
                pct = duw_pct(s)
                vlag = '  [GEWIJZIGD]' if s.get('corrected') else ''
                item = QListWidgetItem(
                    f"slag {i:>2} {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
                    f"duw f{s['pushing_start']}–f{s['endpush_start']}  "
                    f"efficientie {pct if pct is not None else '—'}%{vlag}")
                if s.get('corrected'):
                    item.setBackground(Qt.yellow)
                if i == self.bewerk_i:
                    item.setText(item.text() + '  << BEWERKEN (1/2/3/4) >>')
                    item.setBackground(Qt.red)
                self.lijst.addItem(item)
            self.lijst.blockSignals(False)

        def toggle_bewerk(self):
            row = self.lijst.currentRow()
            if self.bewerk_i is not None:
                self.bewerk_i = None
            elif row >= 0:
                self.bewerk_i = row
                self.bewerk_fase = 0
                self.f = slagen[row]['positioning_start']
                self.timer.stop()
            self.vul_lijst()
            self.toon()

        def nieuwe_slag(self):
            s = {'been': '?', 'positioning_start': self.f, 'pushing_start': self.f,
                 'endpush_start': self.f, 'end_frame': self.f, 'corrected': True}
            slagen.append(s)
            slagen.sort(key=lambda x: x['positioning_start'])
            self.bewerk_i = slagen.index(s)
            self.timer.stop()
            self.vul_lijst()
            self.toon()

        def ongedaan(self):
            if self.bewerk_i is not None:
                s = slagen[self.bewerk_i]
                vers = automatische_fasen(resultaten, info, args.rotatie)
                for v in vers:
                    if v['positioning_start'] == s['positioning_start']:
                        corrected = s.get('corrected')
                        s.update(v)
                        s['corrected'] = False
                        break
                self.vul_lijst()
                self.toon()

        def opslaan(self):
            data = {'video': args.input, 'fps': info.fps, 'rotatie_graden': args.rotatie,
                    'strokes': slagen}
            with open(args.uit, 'w') as fh:
                json.dump(data, fh, indent=1)
            self.status.showMessage(f'opgeslagen: {args.uit}', 5000)

        def klik_slag(self, item):
            self.timer.stop()
            self.f = slagen[self.lijst.row(item)]['positioning_start']
            self.toon()

        def toon(self):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
            ok, fr = self.cap.read()
            if not ok:
                self.timer.stop()
                return
            r = resultaten[self.f] if self.f < len(resultaten) else None
            fr = teken_hulplijnen(fr, r)
            bewerk_slag = slagen[self.bewerk_i] if self.bewerk_i is not None else None
            if bewerk_slag is not None:
                # definieer-modus: banner toont de fase die je NU definieert (1→2→3→4)
                volgorde = list(FASEN_KEYS.values())
                fase_key = volgorde[self.bewerk_fase]
                fase = {'positioning_start': 'positionering', 'pushing_start': 'duw',
                        'endpush_start': 'eind', 'end_frame': 'eind'}[fase_key]
                gezet = [k for k in volgorde if k in bewerk_slag]
                fr = teken_fase_banner(
                    fr, fase, bewerken=True, gezet=len(gezet),
                    tekst_extra=f"druk {self.bewerk_fase + 1} op frame {self.f}")
            else:
                fase = fase_van(slagen[slag_index_op_frame(self.f)]
                                if slag_index_op_frame(self.f) is not None else None, self.f)
                fr = teken_fase_banner(fr, fase, bewerken=False)
            kleuren = {'positionering': '#e05a00', 'duw': '#00c800', 'eind': '#ff7800', None: '#555'}
            namen = {'positionering': 'POSITIONERING (inefficiënt)', 'duw': 'DUWFASE (efficiënt)',
                     'eind': 'EIND-DUW (inefficiënt)', None: 'geen fase'}
            if bewerk_slag is not None:
                volgorde = list(FASEN_KEYS.values())
                fase_key = volgorde[self.bewerk_fase]
                gezet = [k for k in volgorde if k in bewerk_slag]
                self.fase_label.setText(
                    f"DEFINIEER slag {self.bewerk_i}: {namen[fase]} — druk {self.bewerk_fase + 1} "
                    f"({len(gezet)}/4 gezet)")
            else:
                self.fase_label.setText(namen[fase])
            self.fase_label.setStyleSheet(
                f'font-size: 20px; font-weight: bold; color: white; background: {kleuren[fase]};')
            schaal = (self.video_label.height() or 700) / fr.shape[0]
            klein = cv2.resize(fr, (int(fr.shape[1] * schaal), int(fr.shape[0] * schaal)))
            rgb = cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
            self.video_label.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                                                rgb.strides[0], QImage.Format_RGB888)))
            huidige_i = slag_index_op_frame(self.f)
            tijd = tijdlijn_afbeelding(slagen, info.totaal, huidige_i, self.bewerk_i, self.f,
                                       w=max(600, self.video_label.width()))
            rgbt = cv2.cvtColor(tijd, cv2.COLOR_BGR2RGB)
            self.tijd_label.setPixmap(QPixmap.fromImage(QImage(rgbt.data, rgbt.shape[1], rgbt.shape[0],
                                                               rgbt.strides[0], QImage.Format_RGB888)))
            self.lijst.blockSignals(True)
            if huidige_i is not None and self.bewerk_i is None:
                self.lijst.setCurrentRow(huidige_i)
            self.lijst.blockSignals(False)
            bewerk_tekst = ''
            if self.bewerk_i is not None:
                s = slagen[self.bewerk_i]
                bewerk_tekst = (f'   BEWERKEN slag {self.bewerk_i} ('
                                + ', '.join(f'{FASEN_NAAM[k]}={s[v]}' for k, v in FASEN_KEYS.items() if v in s)
                                + ')')
            self.status.showMessage(f'frame {self.f}/{info.totaal} '
                                    f'({self.f / (info.fps or 30):.2f}s){bewerk_tekst}')

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
            elif k == Qt.Key_E:
                self.toggle_bewerk()
            elif k == Qt.Key_N:
                self.nieuwe_slag()
            elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
                if self.bewerk_i is None:
                    self.status.showMessage('Redefine-modus uit (E) — geen slag te wijzigen', 3000)
                    return
                s = slagen[self.bewerk_i]
                s[FASEN_KEYS[ev.text()]] = self.f
                s['corrected'] = True
                # de definieerwijzer volgt de toets die je drukte, niet de fase van dit frame
                self.bewerk_fase = max(self.bewerk_fase, int(ev.text()) - 1)
                self.vul_lijst()
                self.toon()
            elif k == Qt.Key_U:
                self.ongedaan()
            elif k == Qt.Key_S:
                self.opslaan()
            elif k in (Qt.Key_Escape, Qt.Key_Q):
                self.close()

    venster = Venster()
    venster.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasen-marker v3')
    parser.add_argument('--input', required=True)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--uit', default='fasen.json')
    parser.add_argument('--rotatie', type=float, default=10.0)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--rook', default=None)
    parser.add_argument('--rook-frame', type=int, default=100)
    args = parser.parse_args()
    if args.dump:
        info, resultaten = laad_context(args.npz)
        for i, s in enumerate(automatische_fasen(resultaten, info, args.rotatie)):
            print(f"slag {i:>2} {s['been']:<6} f{s['positioning_start']}-{s['end_frame']}  "
                  f"duw f{s['pushing_start']}–f{s['endpush_start']}  efficientie {duw_pct(s)}%")
        sys.exit(0)
    if args.rook:
        main_rook(args.npz, args.input, args.rook, args.rook_frame, args.rotatie)
        sys.exit(0)
    run_gui(args)
