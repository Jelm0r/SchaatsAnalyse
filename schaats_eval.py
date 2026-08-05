"""
Evaluatie van de tracking-nauwkeurigheid (meetbasis voor verbeteringen)
=======================================================================
Meet de kwaliteit van een opgeslagen analyse (.npz uit fase 0) met proxy-metrics
die géén ground truth nodig hebben, en — als er een gouden referentie is — de
echte hoekfout in graden. Zo is elke pijplijn-wijziging hard te toetsen i.p.v.
op oog (zelfde discipline als het fase 6-meetprotocol in ROADMAP.md).

Gebruik (werkt in beide venvs; alleen numpy + cv2 + schaats_analyse):

    # Proxy-metrics van één analyse
    python schaats_eval.py metrics analyse.npz [--golden goud.json]

    # Twee analyses (bv. vóór/na een wijziging) naast elkaar + per-gewricht verschil
    python schaats_eval.py vergelijk oud.npz nieuw.npz [--golden goud.json]

    # Gouden referentie aanmaken: in N verspreide frames knieën + enkels aanklikken
    python schaats_eval.py annoteer video.mp4 --uit goud.json [--n 15]

    # Bochtsignaal nakijken (drempels ijken op nieuw materiaal)
    python schaats_eval.py bocht analyse.npz [--stap 10]

De proxy-metrics:
- **dekking** — fractie frames met pose.
- **botlengte-CV** — variatiecoëfficiënt (std/gemiddelde) van de tibia- en
  femur-pixellengte per been. Botten zijn star: hun beeldlengte hoort alleen
  traag te veranderen (afstand tot de camera). Hoge CV = keypoints die op het
  been heen en weer springen.
- **jitter** — gemiddelde |tweede afgeleide| (px/frame²) van knie/enkel binnen
  aaneengesloten pose-segmenten. Meet trilling die geen echte beweging is.
- **events** — aantal afzetten, L-R-volgorde en gemarkeerde alternatiefouten
  (via de normale `verwerk_afgeleiden` + `segmenteer_afzetten`).

De gouden referentie is een JSON met handmatig aangeklikte knie/enkel-posities
in een aantal frames; de hoekfout is dan |gemeten hoek − gouden hoek| van het
segment enkel→knie per been (beeldvlak, incl. horizonaftrek van de analyse).
De annotatie klikt in twee trappen (grof → uitvergroting) voor subpixel-precisie.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from schaats_analyse import (
    laad_landmarks, verwerk_afgeleiden, segmenteer_afzetten, bereken_hoek_tov_ijs,
    bocht_ratio, bepaal_bocht_reeks, BOCHT_IN, BOCHT_UIT,
    VIS_MIN, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, _savgol,
)

# Volgorde waarin de annotatie per frame wordt aangeklikt (naam, landmark-index).
ANNOTATIE_PUNTEN = [
    ('l_knie', L_KNEE), ('l_enkel', L_ANKLE),
    ('r_knie', R_KNEE), ('r_enkel', R_ANKLE),
]
ZOOM = 6          # uitvergroting van de precisie-klik
ZOOM_REGIO = 100  # zijde (px) van de uitsnede rond de grove klik
# Ondergrens voor de botlengte-trend als fractie van de mediane botlengte. Een bot kan
# in beeld krimpen doordat de schaatser wegrijdt of het been in de kijkrichting draait,
# maar niet tot een kwart van zijn eigen mediaan — zakt de trend daar onder, dan is de
# fit door een uitschieter getrokken en is de noemer betekenisloos (zie _botlengte_cv).
TREND_MIN_FRAC = 0.25


# ── Hulpjes ─────────────────────────────────────────────────────────────────────
def _segmenten(resultaten):
    """Aaneengesloten runs van frame-indices met pose."""
    segs, huidig = [], []
    for i, r in enumerate(resultaten):
        if r.pose_gevonden and r.lm is not None:
            huidig.append(i)
        elif huidig:
            segs.append(huidig); huidig = []
    if huidig:
        segs.append(huidig)
    return segs


def _px(r, idx, w, h):
    return np.array([r.lm[idx].x * w, r.lm[idx].y * h])


def _zichtbaar(r, *idxs):
    return all(r.lm[i].visibility >= VIS_MIN for i in idxs)


def _botlengte_cv(resultaten, w, h, idx_a, idx_b, fps, been=None):
    """
    Instabiliteit van de botlengte: relatieve spreiding rond de trage trend.
    De trend (Savitzky–Golay, ~0.7 s) vangt de echte schaalverandering op doordat
    de schaatser de camera nadert; wat overblijft is meetruis op de keypoints.

    Met `been` ('links'/'rechts') tellen alleen frames waarin dat been het
    stándbeen is — het been waarop daadwerkelijk gemeten wordt. Het zweefbeen is
    frontaal geregeld (deels) verborgen achter het standbeen; detectiefouten dáár
    zijn geen meetfouten.

    Retourneert `(cv, n_gebruikt, n_overgeslagen)`. Overgeslagen zijn frames waar de
    trend onder `TREND_MIN_FRAC` × de mediane botlengte zakt: een polynoomfit door een
    diepe uitschieter (zweefbeen dat achter het standbeen wegvalt) kan lokaal naar nul
    of zelfs **negatief** duiken, en dan blaast `lengte / trend` de CV op tot een
    zevencijferig onzingetal (BUGS.md R3: `CV tibia_l: 981184.372`). Zulke frames
    hebben geen bruikbare noemer; ze horen geteld te worden, niet gedeeld.
    """
    lengtes = [float(np.linalg.norm(_px(r, idx_a, w, h) - _px(r, idx_b, w, h)))
               for r in resultaten
               if r.pose_gevonden and r.lm is not None and _zichtbaar(r, idx_a, idx_b)
               and (been is None or r.been == been)]
    if len(lengtes) < 10:
        return None, len(lengtes), 0
    lengtes = np.array(lengtes)
    venster = max(5, int(round(0.7 * fps)) | 1)
    trend = _savgol(lengtes, venster, 2)
    geldig = trend >= TREND_MIN_FRAC * float(np.median(lengtes))
    n_over = int((~geldig).sum())
    if int(geldig.sum()) < 10:      # te weinig bruikbare noemers → geen uitspraak
        return None, int(geldig.sum()), n_over
    return float(np.std(lengtes[geldig] / trend[geldig] - 1.0)), int(geldig.sum()), n_over


def _jitter(resultaten, w, h, idxs):
    """Gemiddelde |tweede afgeleide| (px/frame²) van de gegeven landmarks, per segment."""
    waarden = []
    for seg in _segmenten(resultaten):
        if len(seg) < 3:
            continue
        for idx in idxs:
            P = np.array([_px(resultaten[i], idx, w, h) for i in seg])
            acc = np.diff(P, n=2, axis=0)
            waarden.extend(np.linalg.norm(acc, axis=1))
    return float(np.mean(waarden)) if waarden else None


def _laad_en_verwerk(pad):
    info, resultaten = laad_landmarks(pad)
    verwerk_afgeleiden(resultaten, info.w, info.h, info.fps)
    events = segmenteer_afzetten(resultaten)
    return info, resultaten, events


# ── Metrics ─────────────────────────────────────────────────────────────────────
def bereken_metrics(pad, golden_pad=None):
    """Alle metrics van één npz als dict (voor printen of vergelijken)."""
    info, resultaten, events = _laad_en_verwerk(pad)
    w, h = info.w, info.h
    m = {'pad': pad, 'frames': len(resultaten)}
    m['dekking'] = sum(1 for r in resultaten if r.pose_gevonden) / max(1, len(resultaten))

    for naam, (a, b) in (('tibia_l', (L_KNEE, L_ANKLE)), ('tibia_r', (R_KNEE, R_ANKLE)),
                         ('femur_l', (L_HIP, L_KNEE)), ('femur_r', (R_HIP, R_KNEE))):
        cv, n, over = _botlengte_cv(resultaten, w, h, a, b, info.fps)
        m[f'cv_{naam}'], m[f'n_{naam}'], m[f'over_{naam}'] = cv, n, over
    # Standbeen-varianten: alleen frames waarin dit been het meetbeen is.
    for naam, been, (a, b) in (('stand_l', 'links', (L_KNEE, L_ANKLE)),
                               ('stand_r', 'rechts', (R_KNEE, R_ANKLE))):
        cv, n, over = _botlengte_cv(resultaten, w, h, a, b, info.fps, been=been)
        m[f'cv_{naam}'], m[f'n_{naam}'], m[f'over_{naam}'] = cv, n, over
    m['jitter_knie_enkel'] = _jitter(resultaten, w, h, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE))

    m['n_events'] = len(events)
    m['volgorde'] = ''.join('L' if e.been == 'links' else 'R' for e in events)
    m['alternatie_fouten'] = sum(1 for e in events if e.opmerking == 'gemiste tegenafzet?')
    m['hoeken'] = [e.hoek for e in events]
    # Onvolledige afzetten (video hield op tijdens de push, of er is helemaal geen
    # zijwaartse push waargenomen) hebben een te steile hoek en horen niet als meting
    # gelezen te worden — hier alleen gemarkeerd mét reden, want de metrics zijn een
    # diagnosemiddel: je wilt zien dát ze er zijn en waaróm.
    m['onvolledig'] = {i: e.onvolledig for i, e in enumerate(events) if e.onvolledig}

    # Middellijn-kwaliteitsvlag (alleen aanwezig in nieuwere analyses).
    devs = [abs(v) for r in resultaten if getattr(r, 'middellijn_dev', None)
            for v in r.middellijn_dev.values() if v is not None]
    m['middellijn_dev_px'] = float(np.mean(devs)) if devs else None

    if golden_pad:
        m.update(_golden_fouten(resultaten, w, h, golden_pad))
    return m


def _golden_fouten(resultaten, w, h, golden_pad):
    """Hoek- en positiefouten t.o.v. handmatig geannoteerde frames."""
    with open(golden_pad, encoding='utf-8') as f:
        goud = json.load(f)
    hoekfouten, puntfouten = [], []
    for fnr_s, punten in goud.get('frames', {}).items():
        fnr = int(fnr_s)
        if fnr >= len(resultaten):
            continue
        r = resultaten[fnr]
        if not (r.pose_gevonden and r.lm is not None):
            continue
        for naam, idx in ANNOTATIE_PUNTEN:
            if naam in punten:
                puntfouten.append(float(np.linalg.norm(
                    _px(r, idx, w, h) - np.array(punten[naam]))))
        for been, (k_idx, e_idx) in (('l', (L_KNEE, L_ANKLE)), ('r', (R_KNEE, R_ANKLE))):
            kn, en = punten.get(f'{been}_knie'), punten.get(f'{been}_enkel')
            if kn is None or en is None:
                continue
            goud_hoek = bereken_hoek_tov_ijs(en, kn, r.horizon_deg)
            meet_hoek = bereken_hoek_tov_ijs(_px(r, e_idx, w, h), _px(r, k_idx, w, h),
                                             r.horizon_deg)
            hoekfouten.append(abs(meet_hoek - goud_hoek))
    if not hoekfouten:
        return {'goud_n': 0}
    return {
        'goud_n': len(hoekfouten),
        'goud_hoekfout_gem': float(np.mean(hoekfouten)),
        'goud_hoekfout_max': float(np.max(hoekfouten)),
        'goud_puntfout_px': float(np.mean(puntfouten)),
    }


def print_metrics(m):
    print(f"\n== {os.path.basename(m['pad'])} ==")
    print(f"  dekking:          {m['dekking']:.1%}  ({m['frames']} frames)")
    for naam in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r', 'stand_l', 'stand_r'):
        cv, n = m[f'cv_{naam}'], m.get(f'n_{naam}', 0)
        # Het aantal metingen erbij: een CV over een handvol frames zegt weinig. Idem het
        # aantal frames zonder bruikbare trend — dat zijn er veel bij een been dat
        # regelmatig achter het andere wegvalt, en dan is ook de CV met een korrel zout.
        over = m.get(f'over_{naam}', 0)
        tel = f"(n={n}" + (f", {over} overgeslagen)" if over else ")")
        print(f"  CV {naam}:       {cv:.3f}  {tel}" if cv is not None
              else f"  CV {naam}:       onbetrouwbaar/te weinig metingen  {tel}")
    j = m['jitter_knie_enkel']
    print(f"  jitter knie/enkel: {j:.2f} px/frame^2" if j is not None else "  jitter: -")
    print(f"  events: {m['n_events']}  volgorde {m['volgorde']}  "
          f"alternatiefouten {m['alternatie_fouten']}")
    onvolledig = m.get('onvolledig', {})
    hoek_tekst = ', '.join(f"{round(x, 1)}{'*' if i in onvolledig else ''}"
                           for i, x in enumerate(m['hoeken']))
    print(f"  hoeken bij voltooiing: [{hoek_tekst}]"
          + (f"   (* = {'/'.join(sorted(set(onvolledig.values())))}, telt niet mee)"
             if onvolledig else ""))
    if m.get('middellijn_dev_px') is not None:
        print(f"  middellijn-afwijking knie: {m['middellijn_dev_px']:.1f} px gem.")
    if m.get('goud_n'):
        print(f"  GOUD ({m['goud_n']} metingen): hoekfout gem {m['goud_hoekfout_gem']:.2f}°"
              f"  max {m['goud_hoekfout_max']:.2f}°  puntfout gem {m['goud_puntfout_px']:.1f} px")


# ── Vergelijken ─────────────────────────────────────────────────────────────────
def vergelijk(pad_a, pad_b, golden_pad=None):
    ma = bereken_metrics(pad_a, golden_pad)
    mb = bereken_metrics(pad_b, golden_pad)
    print_metrics(ma)
    print_metrics(mb)

    info_a, res_a, _ = _laad_en_verwerk(pad_a)
    info_b, res_b, _ = _laad_en_verwerk(pad_b)
    w, h = info_a.w, info_a.h
    n = min(len(res_a), len(res_b))
    print(f"\n== verschil per gewricht (px, over frames met pose in beide) ==")
    for naam, idx in (('knie L', L_KNEE), ('knie R', R_KNEE),
                      ('enkel L', L_ANKLE), ('enkel R', R_ANKLE)):
        d = [float(np.linalg.norm(_px(res_a[i], idx, w, h) - _px(res_b[i], idx, w, h)))
             for i in range(n)
             if res_a[i].pose_gevonden and res_a[i].lm is not None
             and res_b[i].pose_gevonden and res_b[i].lm is not None]
        if d:
            d = np.array(d)
            print(f"  {naam}: mediaan {np.median(d):.1f}  gem {d.mean():.1f}  "
                  f"p95 {np.percentile(d, 95):.1f}")


# ── Annotatie (gouden referentie) ───────────────────────────────────────────────
def annoteer(video_pad, uit_pad, n_frames=15):
    """
    Interactieve clicker: kies n frames gelijkmatig verspreid over de video en klik
    per frame de vier punten in ANNOTATIE_PUNTEN-volgorde aan. Elke klik gaat in twee
    trappen: grof op het hele beeld, daarna precies in een uitvergrote uitsnede.
    Toetsen: u = laatste punt opnieuw, s = frame overslaan, q = stoppen en opslaan.
    """
    cap = cv2.VideoCapture(video_pad)
    if not cap.isOpened():
        raise IOError(f"Kan video niet openen: {video_pad}")
    totaal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    doelen = sorted(set(int(round(x)) for x in np.linspace(0, totaal - 1, n_frames)))

    bestaand = {}
    if os.path.exists(uit_pad):
        with open(uit_pad, encoding='utf-8') as f:
            bestaand = json.load(f).get('frames', {})
        print(f"({len(bestaand)} eerder geannoteerde frames geladen; die blijven staan)")

    frames_uit = dict(bestaand)
    venster = 'annoteer'
    cv2.namedWindow(venster, cv2.WINDOW_NORMAL)
    klik = {}

    def _muis(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            klik['xy'] = (x, y)

    cv2.setMouseCallback(venster, _muis)

    def _wacht_klik(beeld, tekst):
        """Toon beeld + tekst, wacht op klik of toets. Retourneert ('klik', (x,y)) of ('toets', k)."""
        getoond = beeld.copy()
        cv2.putText(getoond, tekst, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(getoond, tekst, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow(venster, getoond)
        klik.clear()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if 'xy' in klik:
                return 'klik', klik['xy']
            if k in (ord('u'), ord('s'), ord('q')):
                return 'toets', chr(k)

    frame_nr = -1
    ok = True
    gestopt = False          # 'q' moet de héle lus stoppen, niet alleen het huidige punt
    for doel in doelen:
        if str(doel) in frames_uit:
            continue
        while frame_nr < doel and ok:
            ok, frame = cap.read()
            frame_nr += 1
        if not ok:
            break
        h, w = frame.shape[:2]
        punten, i = {}, 0
        while i < len(ANNOTATIE_PUNTEN):
            naam, _ = ANNOTATIE_PUNTEN[i]
            basis = frame.copy()
            for nm, (px, py) in punten.items():
                cv2.drawMarker(basis, (int(px), int(py)), (0, 200, 0),
                               cv2.MARKER_CROSS, 18, 2)
            soort, res = _wacht_klik(
                basis, f"frame {doel} ({len(frames_uit)+1}/{len(doelen)}): klik {naam}"
                       "   [u=opnieuw s=overslaan q=stop]")
            if soort == 'toets':
                if res == 'u' and punten:
                    i -= 1
                    punten.pop(ANNOTATIE_PUNTEN[i][0], None)
                    continue
                if res == 's':
                    punten = None
                    break
                if res == 'q':
                    # Alleen `doelen = []` herbindt de naam; de for-lus itereert over het
                    # oorspronkelijke lijst-object en ging daardoor gewoon door naar het
                    # volgende frame. Vandaar een expliciete vlag.
                    punten, gestopt = None, True
                    break
                continue
            gx, gy = res
            # Trap 2: uitvergrote uitsnede rond de grove klik voor de precieze positie.
            x0 = int(np.clip(gx - ZOOM_REGIO // 2, 0, w - ZOOM_REGIO))
            y0 = int(np.clip(gy - ZOOM_REGIO // 2, 0, h - ZOOM_REGIO))
            uitsnede = cv2.resize(frame[y0:y0 + ZOOM_REGIO, x0:x0 + ZOOM_REGIO],
                                  (ZOOM_REGIO * ZOOM, ZOOM_REGIO * ZOOM),
                                  interpolation=cv2.INTER_NEAREST)
            soort2, res2 = _wacht_klik(uitsnede, f"precies: {naam}")
            if soort2 == 'toets':
                continue                      # terug naar de grove klik van ditzelfde punt
            zx, zy = res2
            punten[naam] = [x0 + zx / ZOOM, y0 + zy / ZOOM]
            i += 1
        if punten:
            frames_uit[str(doel)] = punten
        if gestopt:
            break

    cap.release()
    cv2.destroyAllWindows()
    with open(uit_pad, 'w', encoding='utf-8') as f:
        json.dump({'video': os.path.basename(video_pad), 'frames': frames_uit},
                  f, indent=1)
    print(f"{len(frames_uit)} frames geannoteerd → {uit_pad}")


# ── Bochtsignaal ────────────────────────────────────────────────────────────────
def bocht_rapport(pad, stap=None):
    """
    Print het bochtsignaal van een analyse: per frame de ruwe `bocht_ratio`
    (heupbreedte / romplengte) plus de segmenten die `bepaal_bocht_reeks` eruit haalt.
    Hiermee is de drempel op nieuw materiaal na te rekenen zonder de GUI — nodig omdat
    BOCHT_IN/BOCHT_UIT nu geijkt zijn op de clips die tóevallig in de bibliotheek staan.
    """
    info, resultaten = laad_landmarks(pad)
    w, h, fps = info.w, info.h, info.fps
    ruw = [bocht_ratio(r.lm, w, h) if (r.pose_gevonden and r.lm is not None) else None
           for r in resultaten]
    bepaal_bocht_reeks(resultaten, w, h, fps)

    gemeten = [v for v in ruw if v is not None]
    print(f"{os.path.basename(pad)}: {len(resultaten)} frames @ {fps:.1f} fps, "
          f"{len(gemeten)} meetbaar")
    if gemeten:
        q = np.percentile(gemeten, [5, 50, 95])
        print(f"  ratio  min={min(gemeten):.2f}  p05={q[0]:.2f}  mediaan={q[1]:.2f}  "
              f"p95={q[2]:.2f}  max={max(gemeten):.2f}   (drempels: "
              f"bocht < {BOCHT_IN}, recht stuk > {BOCHT_UIT})")

    segmenten, start = [], None
    for i, r in enumerate(resultaten):
        if r.bocht and start is None:
            start = i
        elif not r.bocht and start is not None:
            segmenten.append((start, i - 1)); start = None
    if start is not None:
        segmenten.append((start, len(resultaten) - 1))
    n_bocht = sum(1 for r in resultaten if r.bocht)
    print(f"  bocht: {n_bocht} frames ({n_bocht / max(1, len(resultaten)):.0%}) "
          f"in {len(segmenten)} segment(en)")
    for a, b in segmenten:
        print(f"    frames {a}-{b}  ({a / fps:.1f}-{b / fps:.1f} s, {(b - a + 1) / fps:.1f} s)")

    # Bewust ASCII: de Windows-console draait hier op cp1252 en slikt geen blokjes.
    stap = stap or max(1, len(resultaten) // 60)
    print(f"  verloop (elke {stap} frames; . = geen pose, # = bocht):")
    for i in range(0, len(resultaten), stap):
        v = ruw[i]
        balk = "#" if resultaten[i].bocht else " "
        print(f"    f{i:5d} {i / fps:6.1f}s  {'  . ' if v is None else f'{v:4.2f}'} {balk}")


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('metrics', help='proxy-metrics van één analyse-npz')
    pm.add_argument('npz')
    pm.add_argument('--golden', help='gouden-referentie-json (van "annoteer")')

    pv = sub.add_parser('vergelijk', help='twee analyses naast elkaar')
    pv.add_argument('npz_oud')
    pv.add_argument('npz_nieuw')
    pv.add_argument('--golden')

    pa = sub.add_parser('annoteer', help='gouden referentie aanklikken')
    pa.add_argument('video')
    pa.add_argument('--uit', required=True, help='uitvoer-json')
    pa.add_argument('--n', type=int, default=15, help='aantal frames (default 15)')

    pb = sub.add_parser('bocht', help='bochtsignaal + gevonden bochtsegmenten')
    pb.add_argument('npz')
    pb.add_argument('--stap', type=int, default=None,
                    help='om de hoeveel frames het verloop wordt geprint')

    args = p.parse_args()
    if args.cmd == 'metrics':
        print_metrics(bereken_metrics(args.npz, args.golden))
    elif args.cmd == 'vergelijk':
        vergelijk(args.npz_oud, args.npz_nieuw, args.golden)
    elif args.cmd == 'annoteer':
        annoteer(args.video, args.uit, args.n)
    elif args.cmd == 'bocht':
        bocht_rapport(args.npz, args.stap)


if __name__ == '__main__':
    main()
