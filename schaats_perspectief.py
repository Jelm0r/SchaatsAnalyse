# -*- coding: utf-8 -*-
"""
schaats_perspectief.py — kalibratie + 3D-hoekreconstructie voor perspectiefcorrectie
(ROADMAP fase 7, stap 1: de wiskundekern, los van GUI en pijplijn).

Probleem: de afzethoek wordt in het beeldvlak gemeten; staat de camera niet loodrecht
op het bewegingsvlak van het been, dan vertekent het perspectief de hoek, en die
afwijking verandert met de positie van de schaatser in beeld.

Kern: de baanlijnen in het ijs zijn rechte, evenwijdige lijnen met bekende onderlinge
afstand. Daaruit kalibreren we de camera t.o.v. het ijsvlak (vaste camera, één
kalibratie per video):

  1. Evenwijdige rijrichting-lijnen snijden in beeld in verdwijnpunt V1; dwarslijnen
     geven V2. De lijn V1–V2 is de verdwijnlijn van het ijsvlak (= de ware horizon).
  2. Uit twee orthogonale verdwijnpunten volgt de brandpuntsafstand (aannames:
     principal point in het beeldmidden, vierkante pixels): f² = −(V1−pp)·(V2−pp).
  3. Daarmee de camerarotatie t.o.v. het ijsvlak en (met de lijnafstand als schaal)
     de homografie ijsvlak ↔ wereld, de camerapositie en de camerahoogte.

Vereiste lijnconfiguraties — met alleen 2 rijlijnen + 1 dwarslijn is het stelsel
één vrijheidsgraad te kort (7 onbekenden: f + rotatie + translatie; 6 constraints):
  - ≥2 rijlijnen + ≥2 dwarslijnen (V2 = snijpunt van de dwarslijnen), of
  - ≥3 rijlijnen met bekende onderlinge afstanden (verdwijnlijn via de
    cross-ratio-constructie) + ≥1 dwarslijn, of
  - ≥2 rijlijnen + ≥1 dwarslijn + een opgegeven brandpuntsafstand `f_px`.
Bij een (vrijwel) frontale camera (kijkas in het verticale vlak van de rijrichting)
ligt V2 op oneindig en is zelfkalibratie van f principieel onmogelijk — dan is
`f_px` opgeven de enige route; de rest van de kalibratie werkt dan gewoon.

Hoekreconstructie: de enkel staat op het ijs → wereldpositie via de homografie; de
knie is alleen een kijkstraal. Om die straal in 3D te prikken zijn er twee
inwisselbare aannames achter één interface (`reconstrueer_hoek(methode=...)`),
op testmateriaal te vergelijken vóór er één default wordt:
  - 'onderbeen': constante onderbeenlengte — snijd de kniestraal met de bol rond de
    enkel (twee snijpunten → keuzeregel, zie `reconstrueer_hoek`). De lengte komt bij
    voorkeur van een meting aan de schaatser of uit de lichaamslengte
    (`onderbeen_uit_lichaamslengte()`); `kalibreer_onderbeenlengte()` schat hem uit
    de video maar is principieel een óndergrens — zie de docstring daar.
  - 'beenvlak': het onderbeen ligt in een verticaal vlak met opgegeven richting
    (bv. de rijrichting uit het traject) — snijd de kniestraal met dat vlak.

Zonder bekende lijnafstand in meters kloppen alle hóeken nog steeds (schaal valt
tegen elkaar weg); alleen afgeleide meters (snelheid, slaglengte) niet.

Puur numpy (geen scipy/cv2/mediapipe/torch): importeerbaar in beide venvs.
Zelftest: `python schaats_perspectief.py` — synthetische camera's met bekende stand
projecteren baanlijnen + een "been" met bekende 3D-hoek; geverifieerd wordt dat de
module de hoek tot op < 0.5° terugrekent, voor meerdere cameraposities en
beenposities in beeld.
"""

from dataclasses import dataclass, field

import numpy as np

STANDAARD_LIJNAFSTAND = 4.0   # m — onderlinge afstand van opeenvolgende baanlijnen
CONDITIE_MIN_DEG = 25.0       # been↔kijkstraal-hoek waaronder de reconstructie onbetrouwbaar geldt
VLAK_CONDITIE_MIN_DEG = 10.0  # kijkstraal↔beenvlak-hoek waaronder methode 'beenvlak' onbetrouwbaar geldt
F_MIN_FRAC, F_MAX_FRAC = 0.4, 15.0  # plausibel f-bereik als fractie van de beeldbreedte
# Conditionering van de zelfkalibratie van f: hoe verder een verdwijnpunt van het
# beeldmidden ligt, hoe minder f eruit te halen valt (f² = −(V1−pp)·(V2−pp) wordt dan
# door één ver, slecht bepaald punt gedomineerd). Afstand gemeten in eenheden
# f0 = (breedte+hoogte)/2. Geijkt op de zelftest-camera's — die halen 1,0 / 1,5 / 9,3
# en leveren allemaal de juiste f — tegen een echte baanopname waar de camera langs de
# baan kijkt: daar ligt V2 op 133 en komt f op 3,3× de beeldbreedte (≈17° beeldhoek,
# onmogelijk voor zo'n shot), terwijl 5 px verschuiving van één lijnuiteinde f van
# 3827 px naar "onmogelijk" laat springen.
VP_CONDITIE_WAARSCHUW = 5.0   # hierboven: f is gevoelig, meld het
VP_CONDITIE_MAX = 30.0        # hierboven: f is betekenisloos, weiger en vraag om f_px
ONDERBEEN_FRACTIE = 0.246     # onderbeenlengte (knie–enkel) als fractie van de lichaamslengte
                              # (antropometrische tabel van Winter: knie 0.285·H − enkel 0.039·H)
KNIE_Z_MIN_FRAC = -0.05       # knie mag hooguit deze fractie van L onder het ijs (ruis)


# ---------------------------------------------------------------------------
# homogene hulpjes
# ---------------------------------------------------------------------------

def _eenheid(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def _lijn_hom(p1, p2):
    """Homogene lijn door twee punten, genormaliseerd zodat |ax+by+c| = afstand."""
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    n = float(np.hypot(l[0], l[1]))
    if n < 1e-12:
        raise ValueError("lijn met (vrijwel) samenvallende eindpunten")
    return l / n


def _vp_afstand(v):
    """Afstand van een verdwijnpunt tot het beeldmidden, in eenheden f0 = (b+h)/2
    (de normalisatie waarin `kalibreer_uit_lijnen` rekent). Oneindig ver = evenwijdige
    lijnen in beeld; dat is precies het geval waarin f er niet uit te halen valt."""
    noemer = abs(v[2])
    return float("inf") if noemer < 1e-15 else float(np.hypot(v[0], v[1]) / noemer)


def _verdwijnpunt(lijnen):
    """Kleinste-kwadraten-snijpunt van ≥2 homogene lijnen (SVD). Homogeen resultaat
    met |v| = 1; de w-component kan ~0 zijn (verdwijnpunt op oneindig)."""
    _, _, vt = np.linalg.svd(np.asarray(lijnen, dtype=float))
    return _eenheid(vt[-1])


def _verdwijnlijn_uit_offsets(rijlijnen_hom, offsets, v1):
    """
    Verdwijnlijn van het ijsvlak uit ≥3 evenwijdige lijnen met bekende onderlinge
    afstanden (cross-ratio-constructie). Idee: kies een transversale beeldlijn; die
    snijdt de rijlijnen in punten die in de wereld gelijkmatig met de offsets meelopen.
    De 1D-homografie offset→positie-op-transversaal is uit ≥3 paren bepaald; het beeld
    van offset=∞ is het verdwijnpunt van de transversaal-richting. Verdwijnlijn =
    lijn door V1 en dat punt.
    """
    # transversaal door de oorsprong (beeldmidden in genormaliseerde coördinaten),
    # loodrecht op de richting naar V1 — snijdt de hele waaier netjes
    if abs(v1[2]) > 1e-9:
        dirv = _eenheid(v1[:2] / v1[2])
    else:
        dirv = _eenheid(v1[:2])
    u = np.array([-dirv[1], dirv[0]])
    t = np.array([-u[1], u[0], 0.0])          # lijn door (0,0) met richting u

    schaal = max(abs(o) for o in offsets) or 1.0
    rijen = []
    for l, off in zip(rijlijnen_hom, offsets):
        a = np.cross(t, l)
        if abs(a[2]) < 1e-12:
            raise ValueError("transversaal evenwijdig aan een baanlijn")
        s = float((a[:2] / a[2]) @ u)          # positie langs de transversaal
        x = off / schaal
        rijen.append([x, 1.0, -s * x, -s])     # s = (αx+β)/(γx+δ)
    _, _, vt = np.linalg.svd(np.asarray(rijen))
    alfa, _, gamma, _ = vt[-1]
    if abs(gamma) < 1e-12 * max(abs(alfa), 1e-30):
        vt_pt = np.array([u[0], u[1], 0.0])    # verdwijnpunt op oneindig
    else:
        s_inf = alfa / gamma
        vt_pt = np.array([s_inf * u[0], s_inf * u[1], 1.0])
    return _eenheid(np.cross(v1, vt_pt))


# ---------------------------------------------------------------------------
# kalibratie
# ---------------------------------------------------------------------------

@dataclass
class PerspectiefKalibratie:
    """Camera ↔ ijsvlak-kalibratie voor één (vaste) camerastand.

    Wereldframe: x = dwars op de rijlijnen (van lijn 1 naar lijn 2), y = rijrichting,
    z = omhoog; het ijs is z=0 en de oorsprong ligt op snijpunt rijlijn1 × dwarslijn1.
    Cameraframe: x = rechts in beeld, y = omlaag in beeld, z = kijkrichting.
    """
    w: int
    h: int
    f: float                    # brandpuntsafstand in pixels
    pp: tuple                   # principal point (aangenomen: beeldmidden)
    R: np.ndarray               # wereld→camera-rotatie; kolommen = wereldassen in cameraframe
    t: np.ndarray               # wereld-oorsprong in cameraframe
    C: np.ndarray               # camerapositie in wereldcoördinaten
    H: np.ndarray               # homografie ijsvlak (X, Y, 1) → beeldpixels
    H_inv: np.ndarray
    horizonlijn: np.ndarray     # verdwijnlijn van het ijsvlak in pixelcoörds (a,b,c)
    schaal_bekend: bool         # True als de lijnafstand in echte meters is opgegeven
    f_geschat: bool             # True als f uit de verdwijnpunten komt (niet opgegeven)
    residu_px: float            # RMS-afstand van de getekende lijnpunten tot de terugprojectie
    waarschuwingen: list = field(default_factory=list)

    @property
    def camera_hoogte(self):
        return float(self.C[2])

    @property
    def horizon_deg(self):
        """Kanteling van de ware horizon (verdwijnlijn) t.o.v. de beeld-x-as, zelfde
        conventie als `horizon_hoek_uit_lijn`: positief = loopt naar rechts omhoog."""
        a, b, _ = self.horizonlijn
        dx, dy = -b, a                       # richtingsvector van de lijn
        if dx < 0:
            dx, dy = -dx, -dy
        return float(np.degrees(np.arctan2(-dy, dx)))


@dataclass
class KalibratieInvoer:
    """De nagetrokken lijnen + parameters waaruit een `PerspectiefKalibratie` volgt.

    Dit is wat er bewaard wordt, níet de kalibratie zelf: `PerspectiefKalibratie`
    bestaat vrijwel geheel uit afgeleide matrices (R, t, C, H, H_inv) die exact uit
    deze invoer te herberekenen zijn. Zo blijft de opslag JSON-baar (past in
    `analyse.instellingen_json`, geen schemabump), leesbaar voor een mens, en krijgt
    een oude analyse automatisch de winst van een latere verbetering in de
    kalibratiewiskunde — dezelfde redenering als de events-cache, die bij het openen
    ook vers herberekend wordt.

    Een kalibratie hoort bij één **camerastand**, niet bij één video: alle clips die
    uit dezelfde vaste opstelling komen mogen hem delen (zie `past_bij`).
    """
    rijlijnen: list                 # [((x1,y1),(x2,y2))] in originele pixels
    dwarslijnen: list
    beeld_w: int
    beeld_h: int
    lijnafstand: float = STANDAARD_LIJNAFSTAND
    rij_offsets: list = None
    schaal_bekend: bool = True
    f_px: float = None
    notitie: str = ""               # vrije tekst, bv. "baan Deventer, camera bij 100m"

    def naar_dict(self):
        """JSON-bare vorm; punten worden floats, geen numpy."""
        def _lijnen(ls):
            return [[[float(p[0]), float(p[1])] for p in lijn] for lijn in ls]
        return {
            "rijlijnen": _lijnen(self.rijlijnen),
            "dwarslijnen": _lijnen(self.dwarslijnen),
            "beeld_w": int(self.beeld_w),
            "beeld_h": int(self.beeld_h),
            "lijnafstand": float(self.lijnafstand),
            "rij_offsets": None if self.rij_offsets is None
                           else [float(o) for o in self.rij_offsets],
            "schaal_bekend": bool(self.schaal_bekend),
            "f_px": None if self.f_px is None else float(self.f_px),
            "notitie": self.notitie or "",
        }

    @classmethod
    def uit_dict(cls, d):
        def _lijnen(ls):
            return [tuple((float(p[0]), float(p[1])) for p in lijn) for lijn in (ls or [])]
        return cls(
            rijlijnen=_lijnen(d.get("rijlijnen")),
            dwarslijnen=_lijnen(d.get("dwarslijnen")),
            beeld_w=int(d["beeld_w"]), beeld_h=int(d["beeld_h"]),
            lijnafstand=float(d.get("lijnafstand", STANDAARD_LIJNAFSTAND)),
            rij_offsets=d.get("rij_offsets"),
            schaal_bekend=bool(d.get("schaal_bekend", True)),
            f_px=d.get("f_px"),
            notitie=d.get("notitie", ""))

    def past_bij(self, w, h):
        """Mag deze kalibratie op een video van w×h? Alleen bij gelijke beeldmaat —
        de lijnen staan in pixels, dus een andere resolutie of bijsnijding verschuift
        ze stilzwijgend en levert een plausibele maar foute kalibratie op."""
        return int(w) == int(self.beeld_w) and int(h) == int(self.beeld_h)

    def kalibreer(self):
        """Herbereken de `PerspectiefKalibratie`. Gooit dezelfde ValueError als
        `kalibreer_uit_lijnen` bij een onbruikbare configuratie."""
        return kalibreer_uit_lijnen(
            self.rijlijnen, self.dwarslijnen, self.beeld_w, self.beeld_h,
            lijnafstand=self.lijnafstand, rij_offsets=self.rij_offsets,
            schaal_bekend=self.schaal_bekend, f_px=self.f_px)


def kalibreer_uit_lijnen(rijlijnen, dwarslijnen, beeld_w, beeld_h,
                         lijnafstand=STANDAARD_LIJNAFSTAND, rij_offsets=None,
                         schaal_bekend=True, f_px=None):
    """
    Kalibreer camera ↔ ijsvlak uit nagetrokken baanlijnen.

    rijlijnen   : lijst ((x1,y1),(x2,y2)) pixelpunt-paren van lijnen die in de wereld
                  evenwijdig in de rijrichting lopen, in volgorde (aangrenzend).
    dwarslijnen : idem, haaks op de rijrichting (start-/finishlijn e.d.); ≥1 vereist.
    rij_offsets : wereld-afstand (m) van elke rijlijn t.o.v. de eerste; default
                  gelijkmatig `lijnafstand` uit elkaar in de opgegeven volgorde.
    schaal_bekend: False als de lijnafstand een aanname is — hoeken blijven geldig,
                  meters (camerahoogte, snelheid) niet.
    f_px        : bekende brandpuntsafstand in pixels; verplicht bij configuraties
                  waar zelfkalibratie onderbepaald is (zie moduledocstring).

    Retourneert PerspectiefKalibratie; ValueError met uitleg bij een onbruikbare
    lijnconfiguratie of gedegenereerde geometrie.
    """
    if len(rijlijnen) < 2:
        raise ValueError("minstens twee evenwijdige baanlijnen (rijrichting) nodig")
    if len(dwarslijnen) < 1:
        raise ValueError("minstens één dwarslijn nodig (start-/finishlijn of "
                         "bochtmarkering, haaks op de baanlijnen)")
    if f_px is None and len(dwarslijnen) == 1 and len(rijlijnen) < 3:
        raise ValueError(
            "onderbepaald: met twee baanlijnen en één dwarslijn is de kalibratie "
            "één vrijheidsgraad te kort — teken een derde baanlijn óf een tweede "
            "dwarslijn, óf geef de brandpuntsafstand op (f_px)")

    cx, cy = beeld_w / 2.0, beeld_h / 2.0
    f0 = (beeld_w + beeld_h) / 2.0           # conditionering: werk in ~O(1)-coördinaten

    def norm_pt(p):
        return ((p[0] - cx) / f0, (p[1] - cy) / f0)

    rl = [_lijn_hom(norm_pt(p1), norm_pt(p2)) for p1, p2 in rijlijnen]
    dl = [_lijn_hom(norm_pt(p1), norm_pt(p2)) for p1, p2 in dwarslijnen]
    offsets = list(rij_offsets) if rij_offsets is not None \
        else [i * lijnafstand for i in range(len(rl))]
    if len(offsets) != len(rl):
        raise ValueError("rij_offsets moet evenveel waarden hebben als rijlijnen")
    offsets = [o - offsets[0] for o in offsets]

    waarschuwingen = []

    # --- verdwijnpunten -----------------------------------------------------
    v1 = _verdwijnpunt(rl)                                   # rijrichting
    if len(dl) >= 2:
        v2 = _verdwijnpunt(dl)                               # dwarsrichting
    elif f_px is None:
        lh = _verdwijnlijn_uit_offsets(rl, offsets, v1)      # ≥3 rijlijnen (gecheckt)
        v2 = _eenheid(np.cross(lh, dl[0]))
    else:
        # f bekend: V2 = snijpunt van de dwarslijn met de "orthocomplement-lijn"
        # ω·v1 (alle punten w met v1ᵀ·ω·w = 0), ω = diag(1/fn², 1/fn², 1)
        fn = f_px / f0
        omega_v1 = np.array([v1[0] / fn**2, v1[1] / fn**2, v1[2]])
        v2 = _eenheid(np.cross(omega_v1, dl[0]))

    # --- brandpuntsafstand --------------------------------------------------
    if f_px is not None:
        fn = f_px / f0
        f_geschat = False
    else:
        noemer = v1[2] * v2[2]
        teller = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(noemer) < 1e-9:
            raise ValueError(
                "verdwijnpunt (vrijwel) op oneindig — de camera staat (bijna) frontaal "
                "op of loodrecht op de rijrichting; zelfkalibratie van de brandpunts"
                "afstand kan dan niet. Geef f_px op.")
        f2 = -teller / noemer
        if f2 <= 0:
            raise ValueError(
                "verdwijnpunten niet consistent met een camera (f² ≤ 0) — staan de "
                "dwarslijn(en) in werkelijkheid wel haaks op de baanlijnen?")
        # Conditionering: ligt een verdwijnpunt heel ver weg, dan is f er niet uit te
        # halen — de lijnen die erbij horen lopen in beeld vrijwel evenwijdig, en een
        # paar pixels tekenfout verschuift het punt (en dus f) enorm. Liever hier
        # stoppen dan een plausibel ogende, betekenisloze camerastand afleveren.
        ver = max(_vp_afstand(v1), _vp_afstand(v2))
        if ver > VP_CONDITIE_MAX:
            welke = "de baanlijnen" if _vp_afstand(v1) > _vp_afstand(v2) else "de dwarslijnen"
            raise ValueError(
                f"brandpuntsafstand niet te schatten uit deze lijnen: {welke} lopen in "
                f"beeld vrijwel evenwijdig, dus hun verdwijnpunt ligt ~{ver:.0f}× de "
                f"beeldmaat weg (bruikbaar is < {VP_CONDITIE_MAX:.0f}). De camera kijkt "
                f"dan bijna langs die richting en f volgt er niet uit — een paar pixels "
                f"tekenfout verandert hem al met een factor. Geef f_px op (schaakbord"
                f"kalibratie of cameraspecificatie); de rest van de kalibratie werkt dan "
                f"gewoon.")
        if ver > VP_CONDITIE_WAARSCHUW:
            waarschuwingen.append(
                f"brandpuntsafstand slecht bepaald: verste verdwijnpunt op ~{ver:.0f}× "
                f"de beeldmaat — f is gevoelig voor een paar pixels tekenfout; overweeg "
                f"f_px op te geven")
        fn = float(np.sqrt(f2))
        f_geschat = True
    f_pix = fn * f0
    if not (F_MIN_FRAC * beeld_w <= f_pix <= F_MAX_FRAC * beeld_w):
        waarschuwingen.append(
            f"onwaarschijnlijke brandpuntsafstand ({f_pix:.0f} px bij beeldbreedte "
            f"{beeld_w}) — kalibratie is slecht geconditioneerd (camera bijna "
            f"frontaal?); overweeg f_px op te geven")

    # --- rotatie ------------------------------------------------------------
    def k_inv(v):
        return _eenheid(np.array([v[0] / fn, v[1] / fn, v[2]]))

    r_y = k_inv(v1)                          # rijrichting (wereld-y) in cameraframe
    r_x = k_inv(v2)
    r_x = _eenheid(r_x - (r_x @ r_y) * r_y)  # exact orthogonaal maken
    r_z = np.cross(r_x, r_y)

    def K(v):
        return np.array([fn * v[0], fn * v[1], v[2]])

    # --- translatie + schaal ------------------------------------------------
    # wereld-oorsprong = snijpunt rijlijn1 × dwarslijn1; tweede rijlijn zet de schaal
    o = np.cross(rl[0], dl[0])
    if abs(o[2]) < 1e-12:
        raise ValueError("eerste rijlijn en dwarslijn snijden elkaar niet in beeld "
                         "(evenwijdig getekend?)")
    u0 = k_inv(o)
    if u0[2] < 0:
        u0 = -u0                             # oorsprong ligt vóór de camera
    d = offsets[1]
    if d == 0:
        raise ValueError("twee rijlijnen met dezelfde offset — afstanden controleren")
    p2 = np.cross(rl[1], dl[0])              # beeld van wereldpunt (d, 0)
    c1 = np.cross(p2, K(d * r_x))
    c2 = np.cross(p2, K(u0))
    n2 = float(c2 @ c2)
    if n2 < 1e-18:
        raise ValueError("gedegenereerde lijnconfiguratie bij het bepalen van de schaal")
    mu = -float(c1 @ c2) / n2
    if mu < 0:                               # x-as wees de verkeerde kant op
        r_x = -r_x
        r_z = np.cross(r_x, r_y)
        mu = -mu
    t = mu * u0
    R = np.column_stack([r_x, r_y, r_z])
    C = -R.T @ t
    if C[2] < 0:                             # camera hoort bóven het ijs (rijrichting-teken is vrij)
        r_y = -r_y
        r_z = np.cross(r_x, r_y)
        R = np.column_stack([r_x, r_y, r_z])
        C = -R.T @ t

    # --- homografie + horizon in pixelcoördinaten ---------------------------
    K_px = np.array([[f_pix, 0.0, cx], [0.0, f_pix, cy], [0.0, 0.0, 1.0]])
    H = K_px @ np.column_stack([r_x, r_y, t])
    H /= np.linalg.norm(H)
    H_inv = np.linalg.inv(H)

    def naar_px(v):                          # homogeen genormaliseerd punt → pixels
        return np.array([v[0] * f0 + cx * v[2], v[1] * f0 + cy * v[2], v[2]])

    horizonlijn = np.cross(naar_px(v1), naar_px(v2))
    horizonlijn /= np.hypot(horizonlijn[0], horizonlijn[1])

    # --- residu: getekende punten vs. terugprojectie van de wereldlijnen ----
    H_inv_T = H_inv.T
    afst = []
    wereldlijnen = [(np.array([1.0, 0.0, -off]), lijn)
                    for off, lijn in zip(offsets, rijlijnen)]
    wereldlijnen.append((np.array([0.0, 1.0, 0.0]), dwarslijnen[0]))
    for wl, (p1, p2) in wereldlijnen:
        l_img = H_inv_T @ wl
        l_img /= np.hypot(l_img[0], l_img[1])
        for p in (p1, p2):
            afst.append(float(l_img @ [p[0], p[1], 1.0]))
    residu = float(np.sqrt(np.mean(np.square(afst))))

    kal = PerspectiefKalibratie(
        w=beeld_w, h=beeld_h, f=f_pix, pp=(cx, cy), R=R, t=t, C=C,
        H=H, H_inv=H_inv, horizonlijn=horizonlijn,
        schaal_bekend=schaal_bekend, f_geschat=f_geschat,
        residu_px=residu, waarschuwingen=waarschuwingen)

    # extra dwarslijnen: alleen voor V2 gebruikt — check dat ze in de wereld inderdaad
    # ~haaks op de rijrichting uitkomen (tekenkwaliteit-signaal)
    for i, (p1, p2) in enumerate(dwarslijnen[1:], start=2):
        w1 = punt_op_ijs(kal, p1)
        w2 = punt_op_ijs(kal, p2)
        if w1 is None or w2 is None:
            continue
        richting = _eenheid((w2 - w1)[:2])
        scheef = abs(np.degrees(np.arcsin(np.clip(richting[1], -1, 1))))
        if scheef > 3.0:
            kal.waarschuwingen.append(
                f"dwarslijn {i} staat in de wereld {scheef:.1f}° uit haaks — "
                f"slordig getekend of niet echt een dwarslijn?")
    return kal


# ---------------------------------------------------------------------------
# reconstructie
# ---------------------------------------------------------------------------

def _straal(kal, px):
    """Kijkstraal door een pixel: (camerapositie, eenheidsrichting) in wereldcoörds."""
    v = np.array([(px[0] - kal.pp[0]) / kal.f, (px[1] - kal.pp[1]) / kal.f, 1.0])
    return kal.C, _eenheid(kal.R.T @ v)


def punt_op_ijs(kal, px, hoogte=0.0):
    """Wereldpositie (3-vector) van een pixel op het (horizontale) vlak z=`hoogte` —
    default het ijsvlak zelf. `hoogte` > 0 corrigeert ervoor dat het enkel-landmark
    niet óp het ijs ligt maar op malleolus-/schoenhoogte. None als de kijkstraal het
    vlak niet raakt (pixel op/boven de horizon)."""
    C, d = _straal(kal, px)
    if d[2] >= -1e-9 or hoogte >= C[2]:
        return None
    s = (hoogte - C[2]) / d[2]
    return C + s * d


def _beeldhoek(enkel_px, knie_px):
    """Beeldvlak-hoek zoals de huidige pijplijn hem meet (bereken_hoek_tov_ijs met
    horizon 0, ongerond) — referentie om de toegepaste correctie te tonen."""
    dx = knie_px[0] - enkel_px[0]
    dy = enkel_px[1] - knie_px[1]
    return float(np.degrees(np.arctan2(dy, abs(dx))))


@dataclass
class HoekReconstructie:
    """Resultaat van één 3D-hoekreconstructie."""
    hoek: float                 # echte hoek t.o.v. het ijsvlak (graden)
    hoek_beeld: float           # ongecorrigeerde beeldvlak-hoek
    correctie: float            # hoek − hoek_beeld (kwaliteitsindicator: groot = veel gecorrigeerd)
    conditie_deg: float         # hoek tussen onderbeen en kijkstraal; klein = been in kijkrichting
    betrouwbaar: bool           # False bij slechte conditie of gedegenereerde snijding
    methode: str
    X_enkel: np.ndarray         # wereldcoördinaten (meters als schaal bekend)
    X_knie: np.ndarray
    hoek_alternatief: float = None  # methode 'onderbeen': hoek van de niet-gekozen boloplossing
    vlak_conditie_deg: float = None # methode 'beenvlak': hoek kijkstraal↔vlak; klein = instabiele snijding


def reconstrueer_hoek(kal, enkel_px, knie_px, methode="onderbeen",
                      onderbeen_l=None, vlak_richting=None, rijrichting=None,
                      enkel_hoogte=0.0):
    """
    Reconstrueer de echte afzethoek t.o.v. het ijsvlak uit enkel- en knie-pixels.

    methode 'onderbeen': snijd de knie-kijkstraal met de bol (straal `onderbeen_l`)
      rond de enkel. Twee snijpunten; keuzeregel in volgorde van beschikbaarheid:
      1. `vlak_richting` (2D wereldrichting): oplossing het dichtst bij het verticale
         vlak door de enkel in die richting;
      2. `rijrichting` (2D wereldrichting van het traject): oplossing waarvan de knie
         het meest naar voren leunt;
      3. anders: kleinste |correctie| (conservatief — minste afwijking van het beeld).
    methode 'beenvlak': snijd de kniestraal met het verticale vlak door de enkel met
      richting `vlak_richting` (verplicht).

    `enkel_hoogte` (m) legt de enkel niet óp het ijs maar op die hoogte erboven
    (malleolus + schaats ≈ 0.10 m); de hoek blijft t.o.v. het (horizontale) ijsvlak.

    Retourneert HoekReconstructie, of None als de enkel niet op het ijs te plaatsen
    is (pixel boven de horizon). `betrouwbaar=False` markeert frames waar het been
    bijna in de kijkrichting staat (conditie < {:.0f}°) of de geometrie niet sloot.
    """.format(CONDITIE_MIN_DEG)
    A = punt_op_ijs(kal, enkel_px, hoogte=enkel_hoogte)
    if A is None:
        return None
    C, dk = _straal(kal, knie_px)
    hoek_beeld = _beeldhoek(enkel_px, knie_px)
    gedegenereerd = False
    alternatief = None
    vlak_conditie = None

    if methode == "onderbeen":
        if onderbeen_l is None:
            raise ValueError("methode 'onderbeen' vereist onderbeen_l (meters, of "
                             "via kalibreer_onderbeenlengte)")
        L = float(onderbeen_l)
        w0 = C - A
        b = float(dk @ w0)
        c = float(w0 @ w0) - L * L
        disc = b * b - c
        if disc < 0:
            # straal mist de bol: beeld-knie verder van de enkel dan L kan verklaren
            # (ruis/verkeerde L) → raakpunt nemen en als onbetrouwbaar markeren
            kandidaten = [C + (-b) * dk]
            gedegenereerd = True
        else:
            w_disc = float(np.sqrt(disc))
            kandidaten = [C + s * dk for s in (-b - w_disc, -b + w_disc) if s > 1e-9]
            kandidaten = [X for X in kandidaten if X[2] - A[2] > KNIE_Z_MIN_FRAC * L]
            if not kandidaten:
                kandidaten = [C + (-b) * dk]
                gedegenereerd = True
        if len(kandidaten) == 1:
            X = kandidaten[0]
        else:
            if vlak_richting is not None:
                u = _eenheid([vlak_richting[0], vlak_richting[1], 0.0])
                m = np.cross(u, [0.0, 0.0, 1.0])
                X = min(kandidaten, key=lambda Xk: abs(float(m @ (Xk - A))))
            elif rijrichting is not None:
                r = _eenheid([rijrichting[0], rijrichting[1], 0.0])
                X = max(kandidaten, key=lambda Xk: float(r @ (Xk - A)))
            else:
                X = min(kandidaten, key=lambda Xk: abs(_hoek_tov_ijs(A, Xk) - hoek_beeld))
            ander = kandidaten[0] if kandidaten[1] is X else kandidaten[1]
            alternatief = _hoek_tov_ijs(A, ander)

    elif methode == "beenvlak":
        if vlak_richting is None:
            raise ValueError("methode 'beenvlak' vereist vlak_richting (2D wereldrichting)")
        u = _eenheid([vlak_richting[0], vlak_richting[1], 0.0])
        m = np.cross(u, [0.0, 0.0, 1.0])     # normaal van het verticale beenvlak
        noemer = float(m @ dk)
        # conditie van de snijding: hoek tussen kijkstraal en vlak — ligt de straal
        # (bijna) in het vlak, dan vergroot elke kalibratie-/pixelfout enorm uit;
        # dat gebeurt juist bij een frontale camera met het vlak in de rijrichting
        vlak_conditie = float(np.degrees(np.arcsin(np.clip(abs(noemer), 0.0, 1.0))))
        if abs(noemer) < 1e-9:
            return HoekReconstructie(
                hoek=hoek_beeld, hoek_beeld=hoek_beeld, correctie=0.0,
                conditie_deg=0.0, betrouwbaar=False, methode=methode,
                X_enkel=A, X_knie=A, vlak_conditie_deg=vlak_conditie)
        if vlak_conditie < VLAK_CONDITIE_MIN_DEG:
            gedegenereerd = True
        s = float(m @ (A - C)) / noemer
        if s <= 0:
            gedegenereerd = True
            s = abs(s)
        X = C + s * dk
    else:
        raise ValueError(f"onbekende methode: {methode!r}")

    been = X - A
    been_n = float(np.linalg.norm(been))
    if been_n < 1e-9:
        return HoekReconstructie(
            hoek=hoek_beeld, hoek_beeld=hoek_beeld, correctie=0.0, conditie_deg=0.0,
            betrouwbaar=False, methode=methode, X_enkel=A, X_knie=X,
            vlak_conditie_deg=vlak_conditie)
    hoek = _hoek_tov_ijs(A, X)
    conditie = float(np.degrees(np.arccos(np.clip(abs(been / been_n @ dk), 0.0, 1.0))))
    betrouwbaar = (not gedegenereerd) and conditie >= CONDITIE_MIN_DEG
    return HoekReconstructie(
        hoek=hoek, hoek_beeld=hoek_beeld, correctie=hoek - hoek_beeld,
        conditie_deg=conditie, betrouwbaar=betrouwbaar, methode=methode,
        X_enkel=A, X_knie=X, hoek_alternatief=alternatief,
        vlak_conditie_deg=vlak_conditie)


def _hoek_tov_ijs(A, X):
    """Hoek (graden) van het segment A→X t.o.v. het ijsvlak z=0."""
    d = X - A
    return float(np.degrees(np.arcsin(np.clip(d[2] / np.linalg.norm(d), -1.0, 1.0))))


def onderbeen_uit_lichaamslengte(lichaamslengte_m):
    """Antropometrische schatting van de onderbeenlengte (knie–enkel):
    0.246 × lichaamslengte (tabel van Winter). Zelf opmeten bij de schaatser
    (knieholte tot enkelknobbel) is nóg beter; beide zijn betrouwbaarder dan
    `kalibreer_onderbeenlengte` (zie de bias-uitleg daar)."""
    return ONDERBEEN_FRACTIE * float(lichaamslengte_m)


def kalibreer_onderbeenlengte(kal, paren, percentiel=95.0, enkel_hoogte=0.0):
    """
    Schat de onderbeenlengte (wereld-eenheden) uit een reeks (enkel_px, knie_px)-paren.

    Per frame is de loodrechte afstand van de 3D-enkel tot de knie-kijkstraal een
    ondergrens voor de lengte (foreshortening kan een projectie alléén verkorten);
    een hoog percentiel van die ondergrenzen benadert de echte lengte — máár alleen
    als het onderbeen ergens in de reeks écht ~loodrecht op de kijkrichting staat.

    LET OP — systematische onderschatting: bij schaatsen houdt de enkelhoek
    (dorsiflexie) het onderbeen permanent voorover geleund, dus frontaal gezien
    lijkt het onderbeen áltijd korter dan het is en komt dat loodrechte moment er
    mogelijk nooit. Synthetisch gemeten bias bij realistische oriëntaties
    (elevatie 45–70°, lean binnen rijrichting ± 30–60°): 2–6% te kort, oplopend
    naarmate de camera frontaler staat en de leunspreiding kleiner is — en dat
    werkt door als hoekfouten van enkele graden in methode 'onderbeen'. Gebruik
    daarom bij voorkeur een opgemeten lengte of `onderbeen_uit_lichaamslengte()`;
    deze schatting is een óndergrens/sanity-check. Retourneert None zonder
    bruikbare frames.
    """
    onder = []
    for enkel_px, knie_px in paren:
        A = punt_op_ijs(kal, enkel_px, hoogte=enkel_hoogte)
        if A is None:
            continue
        C, dk = _straal(kal, knie_px)
        w0 = A - C
        onder.append(float(np.linalg.norm(w0 - (w0 @ dk) * dk)))
    if not onder:
        return None
    return float(np.percentile(onder, percentiel))


# ---------------------------------------------------------------------------
# zelftest: synthetische camera's met bekende stand
# ---------------------------------------------------------------------------

class _SynthCamera:
    """Virtuele camera met bekende stand voor de zelftest."""

    def __init__(self, naam, C, doel, roll_deg, f, w=1920, h=1080):
        self.naam, self.f, self.w, self.h = naam, float(f), w, h
        self.C = np.asarray(C, dtype=float)
        z = _eenheid(np.asarray(doel, float) - self.C)
        x = _eenheid(np.cross(z, [0.0, 0.0, 1.0]))
        y = np.cross(z, x)
        r = np.radians(roll_deg)
        x, y = np.cos(r) * x + np.sin(r) * y, -np.sin(r) * x + np.cos(r) * y
        self.R_wc = np.vstack([x, y, z])     # rijen = camera-assen in wereldcoörds

    def project(self, Xw):
        Xc = self.R_wc @ (np.asarray(Xw, dtype=float) - self.C)
        assert Xc[2] > 0.2, f"{self.naam}: punt {Xw} achter de camera"
        return (self.f * Xc[0] / Xc[2] + self.w / 2.0,
                self.f * Xc[1] / Xc[2] + self.h / 2.0)

    def lijnstuk(self, P1, P2):
        return (self.project(P1), self.project(P2))

    def in_beeld(self, px, marge=0.0):
        return (-marge <= px[0] < self.w + marge) and (-marge <= px[1] < self.h + marge)

    def ware_horizonlijn(self):
        r_z = self.R_wc[:, 2]                # wereld-ẑ in cameraframe
        K = np.array([[self.f, 0, self.w / 2.0], [0, self.f, self.h / 2.0], [0, 0, 1.0]])
        l = np.linalg.inv(K).T @ r_z
        return l / np.hypot(l[0], l[1])


# scène: rijlijnen x = 0, 4, 8 (langs y), dwarslijnen y = 4 en 18
_RIJ_X = [0.0, 4.0, 8.0]
_DWARS_Y = [4.0, 18.0]
_Y_BEREIK = (2.0, 22.0)
_X_BEREIK = (-1.0, 9.0)
_L_BEEN = 0.45


def _scene_lijnen(cam):
    rij = [cam.lijnstuk((x, _Y_BEREIK[0], 0), (x, _Y_BEREIK[1], 0)) for x in _RIJ_X]
    dwars = [cam.lijnstuk((_X_BEREIK[0], y, 0), (_X_BEREIK[1], y, 0)) for y in _DWARS_Y]
    return rij, dwars


def _been(enkel_xy, alfa_deg, azimut_deg, lengte=_L_BEEN):
    """Enkel op het ijs + knie met bekende 3D-hoek `alfa` en leunrichting `azimut`."""
    A = np.array([enkel_xy[0], enkel_xy[1], 0.0])
    a, fi = np.radians(alfa_deg), np.radians(azimut_deg)
    K = A + lengte * np.array([np.cos(a) * np.cos(fi), np.cos(a) * np.sin(fi), np.sin(a)])
    return A, K


def _hoeklijn_fout(l1, l2):
    """Hoekverschil (graden) tussen twee genormaliseerde beeldlijnen."""
    c = abs(l1[0] * l2[0] + l1[1] * l2[1])
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def _kalibratie_checks(cam, kal, fouten):
    naam = cam.naam
    f_fout = abs(kal.f - cam.f) / cam.f
    # kalibratie-wereldframe = scène-frame verschoven met dwarslijn 1 (y −= _DWARS_Y[0])
    C_verwacht = cam.C - np.array([0.0, _DWARS_Y[0], 0.0])
    c_fout = float(np.linalg.norm(kal.C - C_verwacht))
    h_fout = abs(kal.camera_hoogte - cam.C[2]) / cam.C[2]
    hz_fout = _hoeklijn_fout(kal.horizonlijn, cam.ware_horizonlijn())
    reproj = 0.0
    for x in np.linspace(0, 8, 5):
        for y in np.linspace(*_Y_BEREIK, 5):
            px = np.array(cam.project((x, y, 0)))
            q = kal.H @ [x, y - _DWARS_Y[0], 1.0]
            reproj = max(reproj, float(np.linalg.norm(q[:2] / q[2] - px)))
    if f_fout > 0.005:
        fouten.append(f"{naam}: f-fout {f_fout:.2%}")
    if c_fout > 0.02:
        fouten.append(f"{naam}: camerapositie-fout {c_fout:.3f} m")
    if hz_fout > 0.05:
        fouten.append(f"{naam}: horizon-fout {hz_fout:.3f}°")
    if reproj > 0.1:
        fouten.append(f"{naam}: homografie-reprojectie {reproj:.3f} px")
    return f_fout, h_fout, hz_fout, reproj


def zelftest(uitgebreid=True):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows-console is cp1252
    rng = np.random.default_rng(7)
    fouten = []

    cameras = [
        _SynthCamera("midden-frontaal ", C=(4, -18, 1.8), doel=(4, 8, 0), roll_deg=0.0, f=1400),
        _SynthCamera("excentrisch     ", C=(-8, -14, 2.5), doel=(6, 8, 0), roll_deg=0.0, f=1400),
        _SynthCamera("sterk exc.+roll ", C=(16, -9, 3.2), doel=(2, 12, 0), roll_deg=2.5, f=1050),
        _SynthCamera("bijna-frontaal  ", C=(7, -16, 2.2), doel=(4, 10, 0), roll_deg=-1.0, f=1600),
    ]

    print("== Kalibratie uit geprojecteerde baanlijnen ==")
    print("   configuraties: A = 3 rijlijnen + 1 dwarslijn (cross-ratio-verdwijnlijn)")
    print("                  B = 2 rijlijnen + 2 dwarslijnen (V2 uit dwarslijnen)")
    print("                  C = 2 rijlijnen + 1 dwarslijn + opgegeven f")
    print(f"{'camera':<17} {'cfg':<4} {'f-fout':>9} {'hoogte-fout':>12} "
          f"{'horizon-fout':>13} {'reproj (px)':>12}")
    kals = {}
    for cam in cameras:
        rij, dwars = _scene_lijnen(cam)
        for cfg, kwargs in [
                ("A", dict(rijlijnen=rij, dwarslijnen=dwars[:1])),
                ("B", dict(rijlijnen=rij[:2], dwarslijnen=dwars)),
                ("C", dict(rijlijnen=rij[:2], dwarslijnen=dwars[:1], f_px=cam.f))]:
            try:
                kal = kalibreer_uit_lijnen(beeld_w=cam.w, beeld_h=cam.h, **kwargs)
            except ValueError as e:
                melding = str(e).split("—")[0].strip()
                print(f"{cam.naam:<17} {cfg:<4} zelfkalibratie geweigerd: {melding} …")
                if cfg == "C" or cam.naam.strip() != "midden-frontaal":
                    fouten.append(f"{cam.naam}/{cfg}: onverwachte weigering: {e}")
                continue
            f_f, h_f, hz_f, rp = _kalibratie_checks(cam, kal, fouten)
            ws = "  ⚠ " + kal.waarschuwingen[0][:40] if kal.waarschuwingen else ""
            print(f"{cam.naam:<17} {cfg:<4} {f_f:>9.2%} {h_f:>12.2%} "
                  f"{hz_f:>12.4f}° {rp:>12.4f}{ws}")
            kals.setdefault(cam.naam, (cam, kal))  # eerste geslaagde kalibratie per camera

    print("\n== Hoekreconstructie (exacte projecties, been in beeld) ==")
    print(f"   onderbeen L = {_L_BEEN} m; hoeken 35/50/65°; leunrichtingen 0–330°; "
          f"eis: fout < 0.5° op betrouwbaar gemarkeerde frames")
    print(f"{'camera':<17} {'n':>4} {'beeldfout min/gem/max':>24} "
          f"{'fout (b)':>10} {'(b) vlag':>9} {'fout (a) best':>14} "
          f"{'keuze rijr./minst':>18}")
    enkels = [(1, 4), (4, 8), (6, 14), (3, 18), (7, 10), (2, 12)]
    alfas = [35.0, 50.0, 65.0]
    azimuts = list(range(0, 360, 30))
    for cam, kal in kals.values():
        beeldf, fa, fb = [], [], []
        b_gevlagd = keuze_rijr = keuze_minst = n = 0
        for enkel in enkels:
            for alfa in alfas:
                for az in azimuts:
                    A, Kn = _been(enkel, alfa, az)
                    if Kn[2] <= 0:
                        continue
                    e_px, k_px = cam.project(A), cam.project(Kn)
                    if not (cam.in_beeld(e_px) and cam.in_beeld(k_px)):
                        continue
                    n += 1
                    beeldf.append(abs(_beeldhoek(e_px, k_px) - alfa))
                    rb = reconstrueer_hoek(kal, e_px, k_px, methode="beenvlak",
                                           vlak_richting=(np.cos(np.radians(az)),
                                                          np.sin(np.radians(az))))
                    if rb.betrouwbaar:
                        fb.append(abs(rb.hoek - alfa))
                    else:
                        b_gevlagd += 1        # beenvlak ~ evenwijdig aan kijkstraal
                    ra = reconstrueer_hoek(kal, e_px, k_px, methode="onderbeen",
                                           onderbeen_l=_L_BEEN, rijrichting=(0, 1))
                    beste = min(abs(ra.hoek - alfa),
                                abs(ra.hoek_alternatief - alfa)
                                if ra.hoek_alternatief is not None else np.inf)
                    fa.append(beste)
                    if abs(ra.hoek - alfa) < 0.5:
                        keuze_rijr += 1
                    rd = reconstrueer_hoek(kal, e_px, k_px, methode="onderbeen",
                                           onderbeen_l=_L_BEEN)
                    if abs(rd.hoek - alfa) < 0.5:
                        keuze_minst += 1
        naam = cam.naam
        if n < 20:
            fouten.append(f"{naam}: te weinig benen in beeld (n={n})")
        if fb and max(fb) > 0.5:
            fouten.append(f"{naam}: methode (b) fout {max(fb):.3f}°")
        if max(fa) > 0.5:
            fouten.append(f"{naam}: methode (a) beste-oplossing-fout {max(fa):.3f}°")
        print(f"{naam:<17} {n:>4} {min(beeldf):>7.2f}/{np.mean(beeldf):>6.2f}/"
              f"{max(beeldf):>6.2f}°  {max(fb) if fb else 0.0:>9.4f}° "
              f"{b_gevlagd / n:>8.0%} {max(fa):>13.4f}° "
              f"{keuze_rijr / n:>7.0%} /{keuze_minst / n:>5.0%}")

    print("\n== Onderbeenlengte-autokalibratie ==")
    print("   ideaal geval (uniforme leunrichtingen 0–330°, dus ook ~loodrecht op de")
    print("   kijkstraal) valideert de wiskunde; 'realistisch' beperkt de leunrichting")
    print("   tot rijrichting ± 45° (enkelhoek houdt het onderbeen voorover) en toont")
    print("   de systematische onderschatting — daarom is een opgemeten lengte de")
    print("   voorkeursroute en deze schatting alleen een ondergrens/sanity-check.")
    rijrichting_az = -90.0                   # schaatser rijdt in -y (richting camera's)
    for cam, kal in kals.values():
        paren_ideaal, paren_reeel = [], []
        for enkel in enkels:
            for alfa in alfas:
                for az in azimuts:
                    A, Kn = _been(enkel, alfa, az)
                    if Kn[2] <= 0:
                        continue
                    e_px, k_px = cam.project(A), cam.project(Kn)
                    if cam.in_beeld(e_px) and cam.in_beeld(k_px):
                        paren_ideaal.append((e_px, k_px))
                        d_az = (az - rijrichting_az + 180) % 360 - 180
                        if abs(d_az) <= 45:
                            paren_reeel.append((e_px, k_px))
        L_i = kalibreer_onderbeenlengte(kal, paren_ideaal, percentiel=100.0)
        L_r = kalibreer_onderbeenlengte(kal, paren_reeel, percentiel=100.0)
        rel_i = abs(L_i - _L_BEEN) / _L_BEEN
        bias_r = (L_r - _L_BEEN) / _L_BEEN
        if rel_i > 0.01:
            fouten.append(f"{cam.naam}: onderbeenlengte-fout (ideaal) {rel_i:.2%}")
        if L_i > _L_BEEN * 1.001 or L_r > _L_BEEN * 1.001:
            fouten.append(f"{cam.naam}: onderbeenlengte-schatting boven de echte "
                          f"lengte — geen ondergrens meer")
        print(f"{cam.naam:<17} ideaal: {L_i:.4f} m (fout {rel_i:.2%})   "
              f"realistisch: {L_r:.4f} m (bias {bias_r:+.1%})")

    print("\n== Serialisatie: KalibratieInvoer round-trip via JSON ==")
    cam = cameras[1]
    rij, dwars = _scene_lijnen(cam)
    inv = KalibratieInvoer(rijlijnen=rij, dwarslijnen=dwars, beeld_w=cam.w, beeld_h=cam.h,
                           lijnafstand=STANDAARD_LIJNAFSTAND, notitie="zelftest")
    kal_a = inv.kalibreer()
    import json as _json
    blob = _json.dumps(inv.naar_dict())
    kal_b = KalibratieInvoer.uit_dict(_json.loads(blob)).kalibreer()
    # Byte-identiek, niet 'ongeveer': de herberekening moet dezelfde weg lopen, anders
    # zou een heropende analyse stilletjes iets andere hoeken geven dan de verse.
    verschillen = [n for n in ("f", "H", "H_inv", "R", "t", "C", "horizonlijn")
                   if not np.array_equal(np.asarray(getattr(kal_a, n), float),
                                         np.asarray(getattr(kal_b, n), float))]
    print(f"json {len(blob)} bytes; byte-identiek herberekend: "
          f"{'ja' if not verschillen else 'NEE — ' + ', '.join(verschillen)}")
    if verschillen:
        fouten.append(f"serialisatie: {', '.join(verschillen)} wijken af na round-trip")
    # De beeldmaat hoort mee te reizen: dezelfde lijnen op een andere resolutie leggen
    # zou een plausibele maar foute kalibratie geven.
    if not (inv.past_bij(cam.w, cam.h) and not inv.past_bij(cam.w // 2, cam.h)):
        fouten.append("serialisatie: past_bij() bewaakt de beeldmaat niet")

    print("\n== Kwaliteitsvlag: been bijna in de kijkrichting ==")
    cam, kal = kals[cameras[1].naam]
    A = np.array([4.0, 8.0, 0.0])
    naar_cam = _eenheid(cam.C - A)           # omhoog richting camera
    perp = _eenheid(np.cross(naar_cam, [0, 0, 1.0]))
    d_leg = _eenheid(np.cos(np.radians(10)) * naar_cam + np.sin(np.radians(10)) * perp)
    Kn = A + _L_BEEN * d_leg
    r = reconstrueer_hoek(kal, cam.project(A), cam.project(Kn),
                          methode="onderbeen", onderbeen_l=_L_BEEN)
    print(f"been 10° van de kijkstraal: conditie = {r.conditie_deg:.1f}°, "
          f"betrouwbaar = {r.betrouwbaar} (drempel {CONDITIE_MIN_DEG:.0f}°)")
    if r.betrouwbaar or r.conditie_deg > 15:
        fouten.append("kwaliteitsvlag: bijna-in-kijkrichting niet gemarkeerd")

    if uitgebreid:
        print("\n== Ruisgevoeligheid (informatief): σ = 1 px op alle lijn-eindpunten, "
              "300 trials, been op (4,8), 50°, leunrichting 150° ==")
        az = 150.0
        A_w, Kn_w = _been((4, 8), 50.0, az)
        vlak = (np.cos(np.radians(az)), np.sin(np.radians(az)))
        for cam_i in (1, 3):                 # excentrisch en bijna-frontaal
            cam = cameras[cam_i]
            rij, dwars = _scene_lijnen(cam)
            e_px, k_px = cam.project(A_w), cam.project(Kn_w)
            f_est, fout_b, fout_a, gevlagd, mislukt = [], [], [], 0, 0
            for _ in range(300):
                ruis = lambda seg: tuple(
                    (p[0] + rng.normal(0, 1.0), p[1] + rng.normal(0, 1.0)) for p in seg)
                try:
                    kal_n = kalibreer_uit_lijnen([ruis(s) for s in rij],
                                                 [ruis(s) for s in dwars[:1]],
                                                 cam.w, cam.h)
                except ValueError:
                    mislukt += 1
                    continue
                f_est.append(kal_n.f)
                rb = reconstrueer_hoek(kal_n, e_px, k_px, methode="beenvlak",
                                       vlak_richting=vlak)
                if rb.betrouwbaar:
                    fout_b.append(abs(rb.hoek - 50.0))
                else:
                    gevlagd += 1
                ra = reconstrueer_hoek(kal_n, e_px, k_px, methode="onderbeen",
                                       onderbeen_l=_L_BEEN, rijrichting=(0, 1))
                fout_a.append(min(abs(ra.hoek - 50.0),
                                  abs(ra.hoek_alternatief - 50.0)
                                  if ra.hoek_alternatief is not None else np.inf))
            if f_est:
                print(f"{cam.naam:<17} f p5/p50/p95 = {np.percentile(f_est, 5):.0f}/"
                      f"{np.percentile(f_est, 50):.0f}/{np.percentile(f_est, 95):.0f} px "
                      f"(echt {cam.f:.0f}); geweigerd {mislukt}, gevlagd {gevlagd}")
                if fout_b:
                    print(f"{'':<17} fout (b) p50/p95 = {np.percentile(fout_b, 50):.2f}/"
                          f"{np.percentile(fout_b, 95):.2f}°   "
                          f"fout (a, beste) p50/p95 = {np.percentile(fout_a, 50):.2f}/"
                          f"{np.percentile(fout_a, 95):.2f}°")
            else:
                print(f"{cam.naam:<17} alle {mislukt} trials geweigerd")

        # demonstratie: beenvlak (bijna) evenwijdig aan de kijkstraal wordt gevlagd
        cam, kal = kals[cameras[1].naam]
        A_d, Kn_d = _been((4, 8), 50.0, 60.0)   # kijkstraal-azimut ≈ 61° voor deze camera
        rd = reconstrueer_hoek(kal, cam.project(A_d), cam.project(Kn_d),
                               methode="beenvlak",
                               vlak_richting=(np.cos(np.radians(60)), np.sin(np.radians(60))))
        print(f"\ngedegenereerd beenvlak (leunrichting ≈ kijkrichting, excentrisch): "
              f"vlak-conditie = {rd.vlak_conditie_deg:.1f}°, betrouwbaar = {rd.betrouwbaar}")
        if rd.betrouwbaar:
            fouten.append("kwaliteitsvlag: gedegenereerd beenvlak niet gemarkeerd")

    print()
    if fouten:
        print(f"FAIL — {len(fouten)} probleem/problemen:")
        for f in fouten:
            print(f"  - {f}")
        return 1
    print("PASS — kalibratie, beide reconstructiemethodes (< 0.5°), lengte-"
          "autokalibratie en kwaliteitsvlag in orde.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(zelftest())
