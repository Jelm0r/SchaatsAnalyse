"""
Schaatser Analyse GUI
=====================
Visuele interface bij schaats_analyse.py: speelt de video af met de
skelet/afzetbeen-overlay live erop, en toont een tabel met alle afzethoeken.

Gebruik:
    python schaats_gui.py

Vereisten (naast schaats_analyse.py z'n dependencies):
    pip install PySide6
"""

import os
import sys
import csv

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QAction, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QCheckBox, QFileDialog, QMessageBox, QProgressDialog,
    QHeaderView, QAbstractItemView, QToolBar, QStackedWidget, QSpinBox,
    QDoubleSpinBox, QDialog, QRadioButton, QComboBox, QFormLayout,
)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

import schaats_perspectief
from schaats_analyse import (
    video_info, segmenteer_afzetten, teken_overlay_op_frame, horizon_hoek_uit_lijn,
    detecteer_ijslijn, PerspectiefConfig,
)

# Backend-selectie: gebruik YOLO-pose + ByteTrack als torch/ultralytics beschikbaar is
# (draai de app dan onder de .venv-yolo), val anders terug op de MediaPipe-backend.
try:
    from schaats_yolo import analyseer as analyseer_backend, BACKEND_NAAM
except Exception:
    from schaats_analyse import analyseer as analyseer_backend
    BACKEND_NAAM = "MediaPipe"

IS_YOLO = BACKEND_NAAM.startswith("YOLO")

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
STANDAARD_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
HEAVY_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_heavy.task")


class DoelKiezer(QDialog):
    """
    Toont het eerste frame en laat de gebruiker op de te volgen schaatser klikken.
    Retourneert een genormaliseerd (x, y)-punt in `doel_punt`, of None ('volg grootste').
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kies de schaatser om te volgen")
        self.doel_punt = None
        self._frame = frame_bgr
        self._scaled_size = None

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Klik op de schaatser die je wilt volgen "
                           "(of gebruik de knop hieronder)."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._klik
        v.addWidget(self.label, 1)

        knoppen = QHBoxLayout()
        knoppen.addStretch(1)
        btn_skip = QPushButton("Volg grootste schaatser")
        btn_skip.clicked.connect(self.accept)     # doel_punt blijft None
        knoppen.addWidget(btn_skip)
        v.addLayout(knoppen)

        self.resize(900, 640)

        h, w = frame_bgr.shape[:2]
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _klik(self, event):
        if self._scaled_size is None:
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        x = (event.position().x() - offx) / sw
        y = (event.position().y() - offy) / sh
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.doel_punt = (float(x), float(y))
            self.accept()


class HorizonKiezer(QDialog):
    """
    Toont het eerste frame en laat de gebruiker twee punten langs de ijslijn (of een
    andere horizontale referentie: boarding, reclameband, baanlijn) klikken. Daaruit
    volgt de kanteling van de camera t.o.v. de horizon. Retourneert `horizon_deg`
    (float, graden) — 0.0 als er geen kanteling wordt ingesteld.
    """
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stel de horizon / ijslijn in")
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self._frame = frame_bgr
        self._punten = []            # originele-pixel (x, y) van de referentielijn
        self._scaled_size = None
        self._scale = 1.0

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Klik twee punten langs het ijs (of de boarding/reclameband) om de\n"
            "camerakanteling te bepalen, of laat hem automatisch detecteren.\n"
            "Klik opnieuw om de lijn te hertekenen."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.mousePressEvent = self._klik
        v.addWidget(self.label, 1)

        self.lbl_hoek = QLabel("Kanteling: 0.00°  (nog geen lijn getekend)")
        self.lbl_hoek.setStyleSheet("font-weight: bold;")
        v.addWidget(self.lbl_hoek)

        self.chk_per_frame = QCheckBox(
            "Automatisch per frame herkennen (voor een schommelende camera) (werkt niet)")
        self.chk_per_frame.setToolTip(
            "WERKT NIET / niet in gebruik: sinds juli 2026 staat de camera altijd\n"
            "precies horizontaal, dus er is geen kanteling om per frame te volgen.\n"
            "Laat deze optie uit.")
        self.chk_per_frame.stateChanged.connect(self._wissel_per_frame)
        v.addWidget(self.chk_per_frame)

        knoppen = QHBoxLayout()
        self.btn_auto = QPushButton("Detecteer (dit frame)")
        self.btn_auto.clicked.connect(self._detecteer)
        knoppen.addWidget(self.btn_auto)
        btn_geen = QPushButton("Geen kanteling (0°)")
        btn_geen.clicked.connect(self._geen_kanteling)
        knoppen.addWidget(btn_geen)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Bevestig")
        self.btn_ok.clicked.connect(self._bevestig)
        knoppen.addWidget(self.btn_ok)
        v.addLayout(knoppen)

        self.resize(900, 680)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        if self._punten:
            painter = QPainter(scaled)
            pen = QPen(QColor(60, 200, 255), 3)
            painter.setPen(pen)
            pts = [(int(x * self._scale), int(y * self._scale)) for x, y in self._punten]
            for px, py in pts:
                painter.drawEllipse(px - 4, py - 4, 8, 8)
            if len(pts) == 2:
                painter.drawLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def _klik(self, event):
        if self._scaled_size is None or self.chk_per_frame.isChecked():
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if len(self._punten) >= 2:           # derde klik → nieuwe lijn beginnen
            self._punten = []
        self._punten.append((ox, oy))
        if len(self._punten) == 2:
            self.horizon_deg = horizon_hoek_uit_lijn(self._punten[0], self._punten[1])
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: klik het tweede punt …")
        self._render()

    def _detecteer(self):
        graden = detecteer_ijslijn(self._frame)
        if graden is None:
            QMessageBox.information(
                self, "Geen ijslijn gevonden",
                "Kon geen betrouwbare horizontale lijn detecteren. Teken de lijn "
                "handmatig, of kies 'Geen kanteling'.")
            return
        # Synthetiseer een weergavelijn dwars door het beeld op de gevonden hoek.
        w, h = self._orig_w, self._orig_h
        cx, cy = w / 2.0, h / 2.0
        helling = np.tan(np.radians(graden))          # y daalt naar rechts bij positieve hoek
        self._punten = [(0.0, cy + helling * cx), (float(w), cy - helling * (w - cx))]
        self.horizon_deg = graden
        self.lbl_hoek.setText(f"Kanteling: {graden:+.2f}°  (automatisch — controleer de lijn)")
        self._render()

    def _wissel_per_frame(self, _state):
        """Bij per-frame auto is de handmatige/constante lijn niet van toepassing."""
        aan = self.chk_per_frame.isChecked()
        self.label.setEnabled(not aan)
        self.btn_auto.setEnabled(not aan)
        if aan:
            self.lbl_hoek.setText("Kanteling: automatisch per frame — "
                                  "wordt tijdens de analyse bepaald.")
        elif len(self._punten) == 2:
            self.lbl_hoek.setText(f"Kanteling: {self.horizon_deg:+.2f}°")
        else:
            self.lbl_hoek.setText("Kanteling: 0.00°  (nog geen lijn getekend)")

    def _bevestig(self):
        self.auto_per_frame = self.chk_per_frame.isChecked()
        self.accept()

    def _geen_kanteling(self):
        self.horizon_deg = 0.0
        self.auto_per_frame = False
        self.accept()


class KalibratieKiezer(QDialog):
    """
    Perspectiefkalibratie via baanlijnen (fase 7, vaste camera). De gebruiker trekt
    op het eerste frame lijnen na (elke lijn = twee klikken): baanlijnen die in
    werkelijkheid evenwijdig in de rijrichting lopen, en dwarslijnen die er haaks op
    staan. De dialoog kalibreert live mee en tekent de gevonden ware horizon; de
    Bevestig-knop kan pas als de kalibratie slaagt. Resultaat in `self.perspectief`
    (PerspectiefConfig).

    Minimaal nodig: 2 baanlijnen + 2 dwarslijnen, óf 3 baanlijnen + 1 dwarslijn, óf
    2 baanlijnen + 1 dwarslijn + een opgegeven brandpuntsafstand (frontale camera's
    kúnnen alleen met opgegeven brandpuntsafstand).
    """
    KLEUR_RIJ = QColor(60, 200, 255)     # cyaan
    KLEUR_DWARS = QColor(255, 170, 40)   # oranje
    KLEUR_HORIZON = QColor(240, 240, 240)

    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Perspectiefkalibratie: trek de baanlijnen na")
        self.perspectief = None
        self._frame = frame_bgr
        self._rijlijnen = []             # [((x,y),(x,y))] in originele pixels
        self._dwarslijnen = []
        self._klik_punt = None           # eerste punt van een lijn-in-wording
        self._kalibratie = None
        self._scaled_size = None
        self._scale = 1.0

        hoofd = QHBoxLayout(self)

        links = QVBoxLayout()
        links.addWidget(QLabel(
            "Trek elke lijn met twee klikken. Baanlijnen: evenwijdig in de rijrichting "
            "(volgorde maakt niet uit).\nDwarslijnen: haaks erop (start-/finishlijn, "
            "bochtmarkering). Trek zo lang mogelijke lijnen — dat is nauwkeuriger."))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 400)
        self.label.mousePressEvent = self._klik
        links.addWidget(self.label, 1)
        hoofd.addLayout(links, 1)

        rechts = QVBoxLayout()

        soort_groep = QGroupBox("Lijnsoort (voor de volgende lijn)")
        sv = QVBoxLayout(soort_groep)
        self.radio_rij = QRadioButton("Baanlijn (rijrichting)")
        self.radio_dwars = QRadioButton("Dwarslijn (haaks op de baan)")
        self.radio_rij.setChecked(True)
        sv.addWidget(self.radio_rij)
        sv.addWidget(self.radio_dwars)
        rechts.addWidget(soort_groep)

        knoppen_lijn = QHBoxLayout()
        btn_wis_laatste = QPushButton("Laatste lijn wissen")
        btn_wis_laatste.clicked.connect(self._wis_laatste)
        btn_wis_alles = QPushButton("Alles wissen")
        btn_wis_alles.clicked.connect(self._wis_alles)
        knoppen_lijn.addWidget(btn_wis_laatste)
        knoppen_lijn.addWidget(btn_wis_alles)
        rechts.addLayout(knoppen_lijn)

        vorm = QFormLayout()
        self.spin_lijnafstand = QDoubleSpinBox()
        self.spin_lijnafstand.setRange(0.5, 30.0)
        self.spin_lijnafstand.setSingleStep(0.5)
        self.spin_lijnafstand.setValue(schaats_perspectief.STANDAARD_LIJNAFSTAND)
        self.spin_lijnafstand.setSuffix(" m")
        self.spin_lijnafstand.valueChanged.connect(self._herkalibreer)
        vorm.addRow("Afstand tussen baanlijnen:", self.spin_lijnafstand)

        self.spin_f = QSpinBox()
        self.spin_f.setRange(0, 100000)
        self.spin_f.setValue(0)
        self.spin_f.setSpecialValueText("automatisch")
        self.spin_f.setToolTip(
            "Brandpuntsafstand in pixels. Normaal schat de kalibratie hem zelf uit de\n"
            "lijnen; bij een (bijna) frontale camera kan dat principieel niet en moet\n"
            "hij hier ingevuld worden (typisch 1–2× de beeldbreedte voor een telefoon).")
        self.spin_f.valueChanged.connect(self._herkalibreer)
        vorm.addRow("Brandpuntsafstand (px):", self.spin_f)

        self.combo_methode = QComboBox()
        self.combo_methode.addItem("Onderbeenlengte (bol-snijding)", "onderbeen")
        self.combo_methode.addItem("Beenvlak (rijrichting)", "beenvlak")
        self.combo_methode.setToolTip(
            "Hoe de knie-diepte wordt gereconstrueerd. Beide zijn experimenteel te\n"
            "vergelijken; 'onderbeenlengte' heeft de lengte hieronder nodig.")
        vorm.addRow("Reconstructie:", self.combo_methode)

        self.spin_lengte = QDoubleSpinBox()
        self.spin_lengte.setRange(1.0, 2.30)
        self.spin_lengte.setSingleStep(0.01)
        self.spin_lengte.setValue(1.80)
        self.spin_lengte.setSuffix(" m")
        vorm.addRow("Lichaamslengte schaatser:", self.spin_lengte)

        self.spin_onderbeen = QDoubleSpinBox()
        self.spin_onderbeen.setRange(0.0, 70.0)
        self.spin_onderbeen.setSingleStep(0.5)
        self.spin_onderbeen.setValue(0.0)
        self.spin_onderbeen.setSuffix(" cm")
        self.spin_onderbeen.setSpecialValueText("uit lichaamslengte")
        self.spin_onderbeen.setToolTip(
            "Opgemeten onderbeenlengte (knieholte tot enkelknobbel). Laat op\n"
            "'uit lichaamslengte' staan om hem te schatten als 0.246 × lichaamslengte.")
        vorm.addRow("Onderbeenlengte:", self.spin_onderbeen)
        rechts.addLayout(vorm)

        self.lbl_status = QLabel("Nog geen lijnen getekend.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight: bold;")
        rechts.addWidget(self.lbl_status)
        rechts.addStretch(1)

        knoppen = QHBoxLayout()
        btn_annuleer = QPushButton("Annuleren")
        btn_annuleer.clicked.connect(self.reject)
        knoppen.addWidget(btn_annuleer)
        knoppen.addStretch(1)
        self.btn_ok = QPushButton("Bevestig")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self._bevestig)
        knoppen.addWidget(self.btn_ok)
        rechts.addLayout(knoppen)

        paneel = QWidget()
        paneel.setLayout(rechts)
        paneel.setFixedWidth(340)
        hoofd.addWidget(paneel)

        self.resize(1150, 700)

        h, w = frame_bgr.shape[:2]
        self._orig_w, self._orig_h = w, h
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._render()

    # ── tekenen ──────────────────────────────────────────────────────────
    def _render(self):
        scaled = self._pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self._scale = scaled.width() / self._orig_w if self._orig_w else 1.0

        painter = QPainter(scaled)
        s = self._scale
        for lijnen, kleur, prefix in ((self._rijlijnen, self.KLEUR_RIJ, "R"),
                                      (self._dwarslijnen, self.KLEUR_DWARS, "D")):
            painter.setPen(QPen(kleur, 3))
            for i, (p1, p2) in enumerate(lijnen, start=1):
                x1, y1 = p1[0] * s, p1[1] * s
                x2, y2 = p2[0] * s, p2[1] * s
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.drawText(int((x1 + x2) / 2) + 6, int((y1 + y2) / 2) - 6,
                                 f"{prefix}{i}")
        if self._klik_punt is not None:
            kleur = self.KLEUR_RIJ if self.radio_rij.isChecked() else self.KLEUR_DWARS
            painter.setPen(QPen(kleur, 3))
            px, py = self._klik_punt[0] * s, self._klik_punt[1] * s
            painter.drawEllipse(int(px) - 4, int(py) - 4, 8, 8)
        if self._kalibratie is not None:
            # ware horizon (verdwijnlijn van het ijsvlak) als visuele controle
            a, b, c = self._kalibratie.horizonlijn
            if abs(b) > 1e-9:
                y0 = -(c + a * 0.0) / b * s
                y1 = -(c + a * self._orig_w) / b * s
                painter.setPen(QPen(self.KLEUR_HORIZON, 1, Qt.DashLine))
                painter.drawLine(0, int(y0), int(self._orig_w * s), int(y1))
        painter.end()

        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    # ── interactie ───────────────────────────────────────────────────────
    def _klik(self, event):
        if self._scaled_size is None:
            return
        sw, sh = self._scaled_size.width(), self._scaled_size.height()
        offx = (self.label.width() - sw) / 2
        offy = (self.label.height() - sh) / 2
        ox = (event.position().x() - offx) / self._scale
        oy = (event.position().y() - offy) / self._scale
        if not (0 <= ox <= self._orig_w and 0 <= oy <= self._orig_h):
            return
        if self._klik_punt is None:
            self._klik_punt = (ox, oy)
        else:
            lijn = (self._klik_punt, (ox, oy))
            self._klik_punt = None
            if np.hypot(lijn[1][0] - lijn[0][0], lijn[1][1] - lijn[0][1]) < 10:
                self.lbl_status.setText("Lijn te kort — klik twee punten verder uit elkaar.")
            elif self.radio_rij.isChecked():
                self._rijlijnen.append(lijn)
            else:
                self._dwarslijnen.append(lijn)
            self._herkalibreer()
        self._render()

    def _wis_laatste(self):
        if self._klik_punt is not None:
            self._klik_punt = None
        elif self._dwarslijnen and (self.radio_dwars.isChecked() or not self._rijlijnen):
            self._dwarslijnen.pop()
        elif self._rijlijnen:
            self._rijlijnen.pop()
        self._herkalibreer()
        self._render()

    def _wis_alles(self):
        self._rijlijnen = []
        self._dwarslijnen = []
        self._klik_punt = None
        self._herkalibreer()
        self._render()

    # ── kalibratie ───────────────────────────────────────────────────────
    @staticmethod
    def _sorteer_rijlijnen(lijnen):
        """Sorteer de baanlijnen ruimtelijk (aangrenzend), zodat de gelijkmatige
        offsets kloppen ongeacht de tekenvolgorde: projecteer de lijnmiddens op de
        richting loodrecht op de gemiddelde lijnrichting."""
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

    def _herkalibreer(self):
        self._kalibratie = None
        n_rij, n_dwars = len(self._rijlijnen), len(self._dwarslijnen)
        if n_rij < 2 or n_dwars < 1:
            self.lbl_status.setText(
                f"Getekend: {n_rij} baanlijn(en), {n_dwars} dwarslijn(en).\n"
                f"Nodig: minstens 2 baanlijnen + 1 dwarslijn "
                f"(2+1 alleen met opgegeven brandpuntsafstand; anders 3+1 of 2+2).")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        try:
            self._kalibratie = schaats_perspectief.kalibreer_uit_lijnen(
                self._sorteer_rijlijnen(self._rijlijnen), self._dwarslijnen,
                self._orig_w, self._orig_h,
                lijnafstand=self.spin_lijnafstand.value(),
                f_px=self.spin_f.value() or None)
        except ValueError as e:
            self.lbl_status.setText(f"Kalibratie lukt nog niet: {e}")
            self.btn_ok.setEnabled(False)
            self._render()
            return
        kal = self._kalibratie
        tekst = (f"Kalibratie OK — f = {kal.f:.0f} px"
                 f"{' (geschat)' if kal.f_geschat else ''}, camerahoogte "
                 f"{kal.camera_hoogte:.1f} m, horizon {kal.horizon_deg:+.2f}°, "
                 f"residu {kal.residu_px:.1f} px.")
        if kal.waarschuwingen:
            tekst += "\n⚠ " + "\n⚠ ".join(kal.waarschuwingen)
        self.lbl_status.setText(tekst)
        self.btn_ok.setEnabled(True)
        self._render()

    def _bevestig(self):
        if self._kalibratie is None:
            return
        if self.spin_onderbeen.value() > 0:
            onderbeen_l = self.spin_onderbeen.value() / 100.0
        else:
            onderbeen_l = schaats_perspectief.onderbeen_uit_lichaamslengte(
                self.spin_lengte.value())
        self.perspectief = PerspectiefConfig(
            kalibratie=self._kalibratie,
            methode=self.combo_methode.currentData(),
            onderbeen_l=onderbeen_l)
        self.accept()


class AnalyseWorker(QThread):
    """Draait de analyse op de achtergrond, zodat de GUI niet blokkeert."""
    voortgang = Signal(int, int)
    klaar = Signal(object, object, object)   # info, resultaten, events
    fout = Signal(str)

    def __init__(self, input_pad, model_pad, smooth_n=5, threshold=0.015, force_fps=None,
                 doel_punt=None, horizon_deg=0.0, auto_horizon=False, smooth_landmarks=True,
                 perspectief=None):
        super().__init__()
        self.input_pad = input_pad
        self.model_pad = model_pad
        self.smooth_n = smooth_n
        self.threshold = threshold
        self.force_fps = force_fps
        self.doel_punt = doel_punt
        self.horizon_deg = horizon_deg
        self.auto_horizon = auto_horizon
        self.smooth_landmarks = smooth_landmarks
        self.perspectief = perspectief

    def run(self):
        try:
            def toon_voortgang(frame_nr, totaal):
                self.voortgang.emit(frame_nr, totaal)

            info, resultaten = analyseer_backend(
                self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                self.force_fps, doel_punt=self.doel_punt, progress_callback=toon_voortgang,
                horizon_deg=self.horizon_deg, auto_horizon=self.auto_horizon,
                smooth_landmarks=self.smooth_landmarks, perspectief=self.perspectief,
            )
            events = segmenteer_afzetten(resultaten)
            self.klaar.emit(info, resultaten, events)
        except Exception as e:
            self.fout.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Schaats Analyse")
        self.resize(1400, 820)

        self.input_pad = None
        self.model_pad = STANDAARD_MODEL
        self.smooth_n = 5
        self.threshold = 0.015
        self.doel_punt = None
        self.horizon_deg = 0.0
        self.auto_horizon = False
        self.perspectief = None
        self.video_info = None
        self.resultaten = []
        self.events = []
        self.huidige_idx = -1
        self.cap_weergave = None
        self._weergave_pos = 0      # frames al gelezen door cap_weergave (sequentiële cursor)
        self._laatste_frame = None  # ruwe kopie van het huidige frame (voor laag-toggles)
        self.worker = None

        self._bouw_ui()
        self.speeltimer = QTimer(self)
        self.speeltimer.timeout.connect(self._speel_tick)

    # ── UI opbouw ────────────────────────────────────────────────────────
    def _bouw_ui(self):
        toolbar = QToolBar("Hoofd")
        self.addToolBar(toolbar)
        self.actie_nieuwe_video = QAction("Nieuwe video...", self)
        self.actie_nieuwe_video.triggered.connect(self._terug_naar_start)
        toolbar.addAction(self.actie_nieuwe_video)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.pagina_start = self._bouw_startpagina()
        self.pagina_analyse = self._bouw_analysepagina()
        self.stack.addWidget(self.pagina_start)
        self.stack.addWidget(self.pagina_analyse)
        self.stack.setCurrentWidget(self.pagina_start)

        self.lbl_live = QLabel("")
        self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_live)
        self.statusBar().showMessage(f"Kies een video om te beginnen.  ·  backend: {BACKEND_NAAM}")

    def _bouw_startpagina(self):
        paneel = QWidget()
        v = QVBoxLayout(paneel)
        v.addStretch(2)

        titel = QLabel("Schaatser Analyse")
        titel.setAlignment(Qt.AlignCenter)
        titel.setStyleSheet("font-size: 28px; font-weight: bold;")
        v.addWidget(titel)

        subtitel = QLabel("Kies een video om de afzethoek-analyse te starten.")
        subtitel.setAlignment(Qt.AlignCenter)
        subtitel.setStyleSheet("color: #888; padding-bottom: 16px;")
        v.addWidget(subtitel)

        self.lbl_gekozen_video = QLabel("Geen video gekozen")
        self.lbl_gekozen_video.setAlignment(Qt.AlignCenter)
        v.addWidget(self.lbl_gekozen_video)

        knop_kies = QPushButton("Video kiezen...")
        knop_kies.setFixedWidth(220)
        knop_kies.clicked.connect(self._kies_video)
        v.addWidget(knop_kies, alignment=Qt.AlignHCenter)

        v.addSpacing(16)

        instellingen = QGroupBox("Instellingen")
        instellingen.setFixedWidth(320)
        fv = QVBoxLayout(instellingen)

        rij_smooth = QHBoxLayout()
        rij_smooth.addWidget(QLabel("Smoothing (frames):"))
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 30)
        self.spin_smooth.setValue(self.smooth_n)
        rij_smooth.addStretch(1)
        rij_smooth.addWidget(self.spin_smooth)
        fv.addLayout(rij_smooth)

        rij_threshold = QHBoxLayout()
        rij_threshold.addWidget(QLabel("Gewicht-drempel:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 0.2)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(self.threshold)
        rij_threshold.addStretch(1)
        rij_threshold.addWidget(self.spin_threshold)
        fv.addLayout(rij_threshold)

        self.chk_heavy = QCheckBox("Heavy-model (nauwkeuriger, trager)")
        self.chk_heavy.setVisible(not IS_YOLO)   # alleen relevant voor de MediaPipe-backend
        fv.addWidget(self.chk_heavy)

        self.chk_perspectief = QCheckBox("Perspectiefcorrectie via baanlijnen (werkt niet)")
        self.chk_perspectief.setToolTip(
            "WERKT NIET / niet in gebruik: sinds juli 2026 wordt er altijd recht van\n"
            "voren met een horizontale camera gefilmd, dus perspectiefvertekening is\n"
            "klein. Deze optie is niet afgebouwd/gevalideerd — laat hem uit.\n"
            "\n"
            "Bedoeld (voor een vaste, schuin geplaatste camera): trek vóór de analyse de\n"
            "baanlijnen na; daaruit wordt de camerastand gekalibreerd en wordt de\n"
            "afzethoek per frame teruggerekend naar het echte ijsvlak (i.p.v. het\n"
            "vertekende beeldvlak). Bijvangst: snelheid en slaglengte in de tabel.")
        fv.addWidget(self.chk_perspectief)

        self.chk_geen_smoothing = QCheckBox("Geen landmark-smoothing (ruwe detecties)")
        self.chk_geen_smoothing.setToolTip(
            "Slaat het opschonen + Savitzky–Golay-smoothen van de landmark-trajecten\n"
            "over: het skelet volgt de detecties exact (kan trillen), maar kan nooit\n"
            "achterlopen door interpolatie. Handig om te zien of een achterlopend\n"
            "skelet uit de smoothing komt of uit de detectie zelf.")
        fv.addWidget(self.chk_geen_smoothing)

        v.addWidget(instellingen, alignment=Qt.AlignHCenter)

        v.addSpacing(16)

        self.btn_start_analyse = QPushButton("Start analyse")
        self.btn_start_analyse.setFixedWidth(220)
        self.btn_start_analyse.setEnabled(False)
        self.btn_start_analyse.clicked.connect(self._start_analyse_vanaf_start)
        v.addWidget(self.btn_start_analyse, alignment=Qt.AlignHCenter)

        v.addStretch(3)
        return paneel

    def _bouw_analysepagina(self):
        paneel = QWidget()
        layout = QVBoxLayout(paneel)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._bouw_videopaneel())
        splitter.addWidget(self._bouw_datapaneel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return paneel

    def _bouw_videopaneel(self):
        paneel = QWidget()
        v = QVBoxLayout(paneel)

        self.video_label = QLabel("Geen video geladen")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")
        self.video_label.setMinimumSize(480, 320)
        v.addWidget(self.video_label, stretch=1)

        knoppen = QHBoxLayout()
        self.btn_start = QPushButton("⏮")
        self.btn_frame_terug = QPushButton("⏪")
        self.btn_play = QPushButton("▶")
        self.btn_frame_verder = QPushButton("⏩")
        self.btn_eind = QPushButton("⏭")
        self.lbl_tijd = QLabel("t=0.00s  frame 0/0")

        self.btn_start.clicked.connect(lambda: self._ga_naar(0))
        self.btn_frame_terug.clicked.connect(lambda: self._ga_naar(self.huidige_idx - 1))
        self.btn_play.clicked.connect(self._toggle_afspelen)
        self.btn_frame_verder.clicked.connect(lambda: self._ga_naar(self.huidige_idx + 1))
        self.btn_eind.clicked.connect(lambda: self._ga_naar(len(self.resultaten) - 1))

        for w in (self.btn_start, self.btn_frame_terug, self.btn_play,
                  self.btn_frame_verder, self.btn_eind):
            knoppen.addWidget(w)
        knoppen.addWidget(self.lbl_tijd, stretch=1)
        v.addLayout(knoppen)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._ga_naar)
        v.addWidget(self.slider)

        toggles = QHBoxLayout()
        self.chk_skelet = QCheckBox("Skelet")
        self.chk_afzetbeen = QCheckBox("Afzetbeen")
        self.chk_hud = QCheckBox("HUD")
        for chk in (self.chk_skelet, self.chk_afzetbeen, self.chk_hud):
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _=None: self._toon_huidig_frame())
            toggles.addWidget(chk)
        toggles.addStretch(1)
        v.addLayout(toggles)

        self._zet_besturing_actief(False)
        return paneel

    def _bouw_datapaneel(self):
        paneel = QSplitter(Qt.Vertical)

        tabel_groep = QGroupBox("Afzethoeken")
        tv = QVBoxLayout(tabel_groep)
        self.tabel = QTableWidget(0, 4)
        self.tabel.setHorizontalHeaderLabels(["#", "Tijd (s)", "Been", "Hoek (°)"])
        self.tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabel.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabel.cellClicked.connect(self._klik_op_rij)
        tv.addWidget(self.tabel)

        self.lbl_stats = QLabel("gem — | min — | max —")
        tv.addWidget(self.lbl_stats)

        self.btn_export = QPushButton("Exporteer CSV")
        self.btn_export.clicked.connect(self._exporteer_csv)
        self.btn_export.setEnabled(False)
        tv.addWidget(self.btn_export)

        paneel.addWidget(tabel_groep)

        grafiek_groep = QGroupBox("Hoek per tijd")
        gv = QVBoxLayout(grafiek_groep)
        self.chart = QChart()
        self.chart.legend().hide()
        self.serie_hoek = QLineSeries()
        self.serie_marker = QLineSeries()
        self.chart.addSeries(self.serie_hoek)
        self.chart.addSeries(self.serie_marker)
        self.as_x = QValueAxis()
        self.as_y = QValueAxis()
        self.as_x.setTitleText("tijd (s)")
        self.as_y.setTitleText("hoek (°)")
        self.chart.addAxis(self.as_x, Qt.AlignBottom)
        self.chart.addAxis(self.as_y, Qt.AlignLeft)
        self.serie_hoek.attachAxis(self.as_x)
        self.serie_hoek.attachAxis(self.as_y)
        self.serie_marker.attachAxis(self.as_x)
        self.serie_marker.attachAxis(self.as_y)
        chart_view = QChartView(self.chart)
        gv.addWidget(chart_view)
        paneel.addWidget(grafiek_groep)

        paneel.setStretchFactor(0, 2)
        paneel.setStretchFactor(1, 1)
        return paneel

    def _zet_besturing_actief(self, actief):
        for w in (self.btn_start, self.btn_frame_terug, self.btn_play,
                  self.btn_frame_verder, self.btn_eind, self.slider):
            w.setEnabled(actief)

    # ── Video laden + analyseren ─────────────────────────────────────────
    def _kies_video(self):
        pad, _ = QFileDialog.getOpenFileName(
            self, "Kies video", "", "Video's (*.mp4 *.mov *.avi *.mkv);;Alle bestanden (*)")
        if not pad:
            return
        self.input_pad = pad
        self.lbl_gekozen_video.setText(os.path.basename(pad))
        self.btn_start_analyse.setEnabled(True)

    def _start_analyse_vanaf_start(self):
        # Modelkeuze is alleen relevant voor de MediaPipe-backend; YOLO gebruikt zijn
        # eigen model (yolo11x-pose.pt) en negeert model_pad.
        if not IS_YOLO:
            if self.chk_heavy.isChecked():
                if os.path.isfile(HEAVY_MODEL):
                    self.model_pad = HEAVY_MODEL
                else:
                    QMessageBox.warning(
                        self, "Heavy-model ontbreekt",
                        "pose_landmarker_heavy.task staat niet naast het script.\n\n"
                        "Download het via:\nhttps://storage.googleapis.com/mediapipe-models/"
                        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                        "pose_landmarker_heavy.task\n\nEr wordt nu met het full-model gewerkt.")
                    self.model_pad = STANDAARD_MODEL
            else:
                self.model_pad = STANDAARD_MODEL

            if not os.path.isfile(self.model_pad):
                gekozen, _ = QFileDialog.getOpenFileName(
                    self, "Kies pose_landmarker .task model", "", "Model (*.task)")
                if not gekozen:
                    return
                self.model_pad = gekozen

        # Eerste frame lezen; delen door doel- en horizon-kiezer.
        frame0 = self._lees_eerste_frame()
        if frame0 is None:
            return

        # Doelschaatser laten kiezen op het eerste frame.
        self.doel_punt = self._kies_doelschaatser(frame0)
        if self.doel_punt is False:      # dialoog afgebroken
            return

        # Perspectiefkalibratie (baanlijnen) óf de klassieke horizon-stap.
        self.perspectief = None
        if self.chk_perspectief.isChecked():
            dlg = KalibratieKiezer(frame0, self)
            if dlg.exec() != QDialog.Accepted:
                return
            self.perspectief = dlg.perspectief
            self.horizon_deg, self.auto_horizon = 0.0, False   # kalibratie vervangt de horizon
        else:
            horizon = self._kies_horizon(frame0)
            if horizon is False:         # dialoog afgebroken
                return
            self.horizon_deg, self.auto_horizon = horizon

        self.smooth_n = self.spin_smooth.value()
        self.threshold = self.spin_threshold.value()

        self.stack.setCurrentWidget(self.pagina_analyse)
        self._start_analyse()

    def _lees_eerste_frame(self):
        """Leest het eerste frame van de gekozen video, of None bij een fout."""
        cap = cv2.VideoCapture(self.input_pad)
        ret, frame0 = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "Fout", "Kan het eerste frame niet lezen.")
            return None
        return frame0

    def _kies_doelschaatser(self, frame0):
        """Toont het eerste frame in een kiezer. Retourneert (x,y), None, of False (afgebroken)."""
        dlg = DoelKiezer(frame0, self)
        if dlg.exec() != QDialog.Accepted:
            return False
        return dlg.doel_punt

    def _kies_horizon(self, frame0):
        """
        Laat de ijslijn/kanteling instellen. Retourneert (graden, auto_per_frame) of
        False (afgebroken).
        """
        dlg = HorizonKiezer(frame0, self)
        if dlg.exec() != QDialog.Accepted:
            return False
        return dlg.horizon_deg, dlg.auto_per_frame

    def _terug_naar_start(self):
        if self.speeltimer.isActive():
            self.speeltimer.stop()
            self.btn_play.setText("▶")
        self.stack.setCurrentWidget(self.pagina_start)

    def _start_analyse(self):
        self._zet_besturing_actief(False)
        self.btn_export.setEnabled(False)

        self.progress = QProgressDialog("Video analyseren...", None, 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)

        self.worker = AnalyseWorker(self.input_pad, self.model_pad, self.smooth_n, self.threshold,
                                    doel_punt=self.doel_punt, horizon_deg=self.horizon_deg,
                                    auto_horizon=self.auto_horizon,
                                    smooth_landmarks=not self.chk_geen_smoothing.isChecked(),
                                    perspectief=self.perspectief)
        self.worker.voortgang.connect(self._analyse_voortgang)
        self.worker.klaar.connect(self._analyse_klaar)
        self.worker.fout.connect(self._analyse_fout)
        self.worker.start()

    def _analyse_voortgang(self, frame_nr, totaal):
        if totaal > 0:
            self.progress.setValue(int(frame_nr / totaal * 100))
        self.progress.setLabelText(f"Video analyseren... ({frame_nr}/{totaal})")

    def _analyse_fout(self, bericht):
        self.progress.close()
        QMessageBox.critical(self, "Fout bij analyseren", bericht)
        self._zet_besturing_actief(False)

    def _analyse_klaar(self, info, resultaten, events):
        self.progress.close()
        self.video_info = info
        self.resultaten = resultaten
        self.events = events

        if self.cap_weergave is not None:
            self.cap_weergave.release()
        self.cap_weergave = cv2.VideoCapture(self.input_pad)
        self._weergave_pos = 0
        self._laatste_frame = None
        self.huidige_idx = -1

        self.slider.setRange(0, max(0, len(resultaten) - 1))
        self._vul_tabel()
        self._vul_grafiek()
        self._zet_besturing_actief(True)
        self.btn_export.setEnabled(bool(events))

        self.statusBar().showMessage(
            f"{os.path.basename(self.input_pad)} — {info.w}×{info.h} @ {info.fps:.1f}fps, "
            f"{len(resultaten)} frames, {len(events)} afzetten gevonden")

        self._ga_naar(0)

    # ── Tabel + grafiek vullen ───────────────────────────────────────────
    def _vul_tabel(self):
        # Kolommen dynamisch: perspectiefcorrectie en snelheid/slaglengte alleen als
        # er een kalibratie actief was (de events dragen die velden dan).
        met_corr = any(ev.correctie is not None for ev in self.events)
        met_metrisch = any(ev.snelheid is not None for ev in self.events)
        koppen = ["#", "Tijd (s)", "Been", "Hoek (°)"]
        if met_corr:
            koppen.append("Corr. (°)")
        if met_metrisch:
            koppen += ["v (m/s)", "Slag (m)"]
        self.tabel.setColumnCount(len(koppen))
        self.tabel.setHorizontalHeaderLabels(koppen)

        self.tabel.setRowCount(len(self.events))
        markeer_kleur = QColor(120, 60, 20)     # waarschuwing: onmogelijke L/R-herhaling
        onbetrouwbaar_kleur = QColor(40, 60, 120)  # hoek gemeten met been ~in kijkrichting
        for i, ev in enumerate(self.events):
            waarden = [str(i + 1), f"{ev.start_tijd:.2f}", ev.been.capitalize(), f"{ev.hoek:.1f}"]
            if met_corr:
                waarden.append(f"{ev.correctie:+.1f}" if ev.correctie is not None else "—")
            if met_metrisch:
                waarden.append(f"{ev.snelheid:.1f}" if ev.snelheid is not None else "—")
                waarden.append(f"{ev.slaglengte:.1f}" if ev.slaglengte is not None else "—")
            gemarkeerd = ev.opmerking == "gemiste tegenafzet?"
            onbetrouwbaar = met_corr and not ev.betrouwbaar
            for kolom, waarde in enumerate(waarden):
                item = QTableWidgetItem(waarde)
                item.setTextAlignment(Qt.AlignCenter)
                if gemarkeerd:
                    item.setBackground(markeer_kleur)
                    item.setToolTip("Zelfde been als de vorige afzet — onmogelijk in het "
                                    "schaatsritme. Waarschijnlijk een gemiste tegen-afzet.")
                elif onbetrouwbaar:
                    item.setBackground(onbetrouwbaar_kleur)
                    item.setToolTip("Been stond bij afzet-voltooiing bijna in de kijkrichting "
                                    "— de perspectiefcorrectie (en dus de hoek) is hier "
                                    "onbetrouwbaar.")
                self.tabel.setItem(i, kolom, item)

        if self.events:
            hoeken = [ev.hoek for ev in self.events]
            n_mark = sum(1 for ev in self.events if ev.opmerking == "gemiste tegenafzet?")
            tekst = f"gem {np.mean(hoeken):.1f}°  |  min {min(hoeken):.1f}°  |  max {max(hoeken):.1f}°"
            if n_mark:
                tekst += f"   ·  ⚠ {n_mark} mogelijke L/R-fout"
            if met_corr:
                n_onb = sum(1 for ev in self.events if not ev.betrouwbaar)
                if n_onb:
                    tekst += f"   ·  ⚠ {n_onb} onbetrouwbare hoek (kijkrichting)"
            self.lbl_stats.setText(tekst)
        else:
            self.lbl_stats.setText("Geen afzetten gedetecteerd")

    def _vul_grafiek(self):
        self.serie_hoek.clear()
        punten = [(r.tijd, r.smooth_hoek) for r in self.resultaten if r.pose_gevonden]
        if not punten:
            return
        for t, hoek in punten:
            self.serie_hoek.append(t, hoek)

        tijden = [t for t, _ in punten]
        hoeken = [h for _, h in punten]
        self.as_x.setRange(0, max(tijden) if tijden else 1)
        marge = 5
        self.as_y.setRange(min(hoeken) - marge, max(hoeken) + marge)

    def _update_grafiek_marker(self, tijd):
        y_min, y_max = self.as_y.min(), self.as_y.max()
        self.serie_marker.clear()
        self.serie_marker.append(tijd, y_min)
        self.serie_marker.append(tijd, y_max)

    # ── Navigatie / weergave ─────────────────────────────────────────────
    def _ga_naar(self, idx):
        if not self.resultaten:
            return
        idx = max(0, min(idx, len(self.resultaten) - 1))
        self._toon_frame(idx)

    def _toon_huidig_frame(self):
        if self.huidige_idx >= 0:
            self._toon_frame(self.huidige_idx)

    def _lees_frame_exact(self, idx):
        """
        Lees frame `idx` frame-exact, uitsluitend via sequentieel lezen.
        Een CAP_PROP_POS_FRAMES-seek is op VFR-video's (bv. iPhone-.MOV) níet
        frame-exact: het gedecodeerde beeld kan er enkele frames naast zitten
        terwijl OpenCV wél het gevraagde framenummer rapporteert. Het skelet
        (van het júiste frame) lijkt dan achter te lopen op het beeld — ook
        tijdens het afspelen erna, want de fout blijft constant. Daarom houden
        we zelf de cursor bij: vooruit spoelen met grab(), achteruit door de
        video te heropenen. Op de korte clips waar deze tool voor is, is dat
        ruim snel genoeg.
        """
        if idx < self._weergave_pos:
            self.cap_weergave.release()
            self.cap_weergave = cv2.VideoCapture(self.input_pad)
            self._weergave_pos = 0
        while self._weergave_pos < idx:
            if not self.cap_weergave.grab():
                return None
            self._weergave_pos += 1
        ret, frame = self.cap_weergave.read()
        if not ret:
            return None
        self._weergave_pos += 1
        return frame

    def _toon_frame(self, idx):
        if idx == self.huidige_idx and self._laatste_frame is not None:
            frame = self._laatste_frame.copy()      # alleen overlay opnieuw tekenen
        else:
            frame = self._lees_frame_exact(idx)
            if frame is None:
                return
            self._laatste_frame = frame
            frame = frame.copy()
        self.huidige_idx = idx

        resultaat = self.resultaten[idx]
        teken_overlay_op_frame(
            frame, resultaat, self.video_info.fps,
            toon_skelet=self.chk_skelet.isChecked(),
            toon_afzetbeen=self.chk_afzetbeen.isChecked(),
            toon_hud=self.chk_hud.isChecked(),
        )
        self._toon_pixmap(frame)

        if idx != self.slider.value():
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        self.lbl_tijd.setText(f"t={resultaat.tijd:.2f}s  frame {idx}/{len(self.resultaten) - 1}")
        self._update_grafiek_marker(resultaat.tijd)
        self._markeer_actieve_rij(idx)
        self._update_live_status(resultaat)

    def _update_live_status(self, resultaat):
        if not resultaat.pose_gevonden:
            self.lbl_live.setText("Geen pose gedetecteerd")
            self.lbl_live.setStyleSheet("font-weight: bold; padding-right: 10px; color: #c33;")
            return

        status = "GEWICHT OP BEEN" if resultaat.gewicht_erop else "AFZET VOLTOOID"
        kleur = "#2a2" if resultaat.gewicht_erop else "#c33"
        extra = ""
        if resultaat.hoek_correctie is not None:
            extra = f"Corr: {resultaat.hoek_correctie:+.1f}°   "
            if resultaat.snelheid is not None:
                extra += f"v: {resultaat.snelheid:.1f} m/s   "
            if not resultaat.hoek_betrouwbaar:
                extra += "⚠ onbetrouwbaar   "
        self.lbl_live.setText(
            f"Afzetbeen: {resultaat.been.upper()}   "
            f"Afzethoek: {resultaat.hoek}°   "
            f"Kniehoek: {resultaat.kniehoek}°   "
            f"{extra}{status}"
        )
        self.lbl_live.setStyleSheet(f"font-weight: bold; padding-right: 10px; color: {kleur};")

    def _toon_pixmap(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def _markeer_actieve_rij(self, idx):
        for i, ev in enumerate(self.events):
            if ev.start_frame <= idx <= ev.eind_frame:
                if self.tabel.currentRow() != i:
                    self.tabel.blockSignals(True)
                    self.tabel.selectRow(i)
                    self.tabel.blockSignals(False)
                return

    def _klik_op_rij(self, rij, _kolom):
        if 0 <= rij < len(self.events):
            self._ga_naar(self.events[rij].start_frame)

    # ── Afspelen ─────────────────────────────────────────────────────────
    def _toggle_afspelen(self):
        if self.speeltimer.isActive():
            self.speeltimer.stop()
            self.btn_play.setText("▶")
        else:
            if self.huidige_idx >= len(self.resultaten) - 1:
                self._ga_naar(0)
            interval_ms = max(1, int(1000 / (self.video_info.fps or 30.0)))
            self.speeltimer.start(interval_ms)
            self.btn_play.setText("⏸")

    def _speel_tick(self):
        volgende = self.huidige_idx + 1
        if volgende >= len(self.resultaten):
            self.speeltimer.stop()
            self.btn_play.setText("▶")
            return
        self._toon_frame(volgende)

    # ── Export ───────────────────────────────────────────────────────────
    def _exporteer_csv(self):
        pad, _ = QFileDialog.getSaveFileName(self, "Exporteer afzethoeken", "afzethoeken.csv", "CSV (*.csv)")
        if not pad:
            return
        met_corr = any(ev.correctie is not None for ev in self.events)
        with open(pad, "w", newline="", encoding="utf-8") as f:
            schrijver = csv.writer(f)
            kop = ["#", "been", "start_tijd_s", "eind_tijd_s", "hoek_deg", "min_hoek_deg", "max_hoek_deg"]
            if met_corr:
                kop += ["correctie_deg", "betrouwbaar", "snelheid_ms", "slaglengte_m"]
            schrijver.writerow(kop)
            for i, ev in enumerate(self.events):
                rij = [i + 1, ev.been, f"{ev.start_tijd:.3f}", f"{ev.eind_tijd:.3f}",
                       f"{ev.hoek:.1f}", f"{ev.min_hoek:.1f}", f"{ev.max_hoek:.1f}"]
                if met_corr:
                    rij += [f"{ev.correctie:+.1f}" if ev.correctie is not None else "",
                            int(bool(ev.betrouwbaar)),
                            f"{ev.snelheid:.2f}" if ev.snelheid is not None else "",
                            f"{ev.slaglengte:.2f}" if ev.slaglengte is not None else ""]
                schrijver.writerow(rij)
        self.statusBar().showMessage(f"Geëxporteerd naar {pad}", 5000)

    def closeEvent(self, event):
        self.speeltimer.stop()
        if self.cap_weergave is not None:
            self.cap_weergave.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    venster = MainWindow()
    venster.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
