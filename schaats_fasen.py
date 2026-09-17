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
import csv
import json
import shlex
import math
import os
import subprocess
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
  R        rapport + overlayvideo
  A        auto-detectie opnieuw (npz)
  P        video naar PinkBox sturen (voorbewerken)
  Esc      actie annuleren
  Q        afsluiten

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
    import numpy as np
    info, resultaten = sa.laad_landmarks(npz_pad)
    raw = np.load(npz_pad, allow_pickle=True)['landmarks']   # (n, 33, 3), genormaliseerd
    for i, r in enumerate(resultaten):
        r.raw_lm = raw[i] if i < len(raw) else None
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
    """Efficientie-score: 1.0 = 50% van de slag is de efficiënte duwfase (2→3).

    score = aandeel efficiënte frames / 0.5 → 2.0 = hele slag efficiënt,
    1.0 = helft, 0.5 = kwart.
    """
    try:
        a, b, c, d = slag['pushing_start'], slag['endpush_start'], slag['positioning_start'], slag['end_frame']
        if None in (a, b, c, d) or d <= c:
            return None
        return round(2.0 * (b - a) / (d - c), 2)
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


def teken_fase_banner(fr, fase, bewerken=False, gezet=0, tekst_extra=None, fase_frame=None):
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
    # live tweede regel in definitie-modus: in welke fase zit DIT frame
    if bewerken and fase_frame is not None:
        fkleur = BAND_KLEUR['positioning' if fase_frame == 'positionering' else fase_frame]
        cv2.rectangle(fr, (w // 2 - 380, 104), (w // 2 + 380, 156), fkleur, -1)
        cv2.putText(fr, f'dit frame: {fase_frame.upper()}', (w // 2 - 350, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return fr


_datum_cache = {}


def bestandsdatum(video_pad):
    """Opnamedatum uit de mp4-metadata (ffprobe creation_time), niet de exportdatum."""
    import subprocess
    import datetime
    try:
        mtime = os.path.getmtime(video_pad)
    except OSError:
        return ''
    if video_pad in _datum_cache and _datum_cache[video_pad][0] == mtime:
        return _datum_cache[video_pad][1]
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format_tags=creation_time',
             '-of', 'json', video_pad],
            capture_output=True, text=True, timeout=10).stdout
        ct = json.loads(out)['format']['tags']['creation_time']
        dt = datetime.datetime.strptime(ct[:19], '%Y-%m-%dT%H:%M:%S')
        datum = dt.strftime('%d %b %Y %H:%M')
    except Exception:
        datum = ''
    _datum_cache[video_pad] = (mtime, datum)
    return datum


def scan_videos(d):
    if not os.path.isdir(d):
        return []
    vids = []
    for f in sorted(os.listdir(d)):
        if f.startswith('._') or not f.lower().endswith(('.mp4', '.mov', '.m4v')):
            continue
        vids.append(os.path.join(d, f))
    return vids


def video_paden(video_pad):
    stem = os.path.splitext(os.path.basename(video_pad))[0]
    d = os.path.dirname(video_pad)
    return {
        'video': video_pad,
        'npz': os.path.join(d, 'npz', stem + '.npz'),
        'fasen': os.path.join(d, 'fasen', stem + '.fasen.json'),
        'rapport': os.path.join(d, 'reports', stem),
    }


def laad_fasen_json(pad):
    with open(pad) as fh:
        return json.load(fh).get('strokes', [])


def hervorm_resultaten(resultaten, info, target_n):
    """Landmark-reeks hersamplen naar `target_n` frames (bv. 2x-RIFE-video).

    Meettechnisch zijn hersamplede frames geen nieuwe metingen — ze zijn een
    tijd-interpolatie van de gemeten frames, net als de geinterpolerde beelden.
    """
    import numpy as np
    if not resultaten or target_n <= len(resultaten):
        return info, resultaten
    n = len(resultaten)
    arr = np.array([r.raw_lm if r.raw_lm is not None else np.full((33, 3), np.nan)
                    for r in resultaten], dtype=float)          # (n, 33, 3)
    xs = np.arange(n)
    xs2 = np.linspace(0, n - 1, target_n)
    arr2 = np.empty((target_n, 33, 3), dtype=float)
    for i in range(33):
        for ch in range(3):
            kolom = arr[:, i, ch]
            geldig = ~np.isnan(kolom)
            if geldig.sum() >= 2:
                arr2[:, i, ch] = np.interp(xs2, xs[geldig], kolom[geldig])
            else:
                arr2[:, i, ch] = kolom[0] if n else np.nan

    gev = np.array([r.pose_gevonden for r in resultaten], dtype=float)
    gev2 = np.interp(xs2, xs, gev) >= 0.5
    bocht = np.array([r.bocht for r in resultaten], dtype=float)
    bocht2 = np.interp(xs2, xs, bocht) >= 0.5
    horizon = np.array([r.horizon_deg for r in resultaten], dtype=float)
    horizon2 = np.interp(xs2, xs, horizon)
    def _mid(r):
        # middellijn_dev kan tuple of dict (per knie) zijn, afhankelijk van backend
        md = r.middellijn_dev
        if md is None:
            return (0.0, 0.0)
        if isinstance(md, dict):
            vals = [v for v in md.values() if isinstance(v, (int, float))]
            a = vals[0] if len(vals) > 0 else 0.0
            b = vals[1] if len(vals) > 1 else a
            return (float(a), float(b))
        return (float(md[0]), float(md[1]))

    mid = np.array([_mid(r) for r in resultaten], dtype=float)
    mid2 = np.stack([np.interp(xs2, xs, mid[:, 0]), np.interp(xs2, xs, mid[:, 1])], axis=1)

    info2 = sa.VideoInfo(w=info.w, h=info.h, fps=(info.fps * target_n / n), totaal=target_n)
    uit = []
    for i in range(target_n):
        r = sa.FrameResultaat(frame_nr=i, tijd=i / (info2.fps or 60.0))
        lm = arr2[i]
        if np.isnan(lm).all():
            r.lm_data = None
        else:
            r.lm_data = maak_lm_data(lm, info.w, info.h)
        r.pose_gevonden = bool(gev2[i])
        r.bocht = bool(bocht2[i])
        r.horizon_deg = float(horizon2[i])
        r.middellijn_dev = (float(mid2[i, 0]), float(mid2[i, 1]))
        r.raw_lm = lm
        uit.append(r)
    # heupmidden-baan opnieuw voor de rijrichting-lijn
    for i, r in enumerate(uit):
        pts = []
        for j in range(max(0, i - 24), min(target_n, i + 25)):
            lm = uit[j].lm_data
            if lm is not None:
                pts.append(((lm['l_heup'][0] + lm['r_heup'][0]) / 2.0,
                            (lm['l_heup'][1] + lm['r_heup'][1]) / 2.0))
        r.heup_hist = pts if len(pts) >= 2 else None
    if info2.fps and uit:
        sa.verwerk_afgeleiden(uit, info2.w or 1, info2.h or 1, info2.fps, 9, 0.015)
    return info2, uit


def maak_lm_data(lm, w, h):
    def pt(i):
        return (int(round(lm[i][0] * w)), int(round(lm[i][1] * h)))

    d = {
        'l_heup': pt(sa.L_HIP), 'r_heup': pt(sa.R_HIP),
        'l_knie': pt(sa.L_KNEE), 'r_knie': pt(sa.R_KNEE),
        'l_enkel': pt(sa.L_ANKLE), 'r_enkel': pt(sa.R_ANKLE),
        'l_hiel': pt(sa.L_HEEL), 'r_hiel': pt(sa.R_HEEL),
        'l_teen': pt(sa.L_TOE), 'r_teen': pt(sa.R_TOE),
        'vis_l_enkel': float(lm[sa.L_ANKLE][2]), 'vis_r_enkel': float(lm[sa.R_ANKLE][2]),
        'vis_l_knie': float(lm[sa.L_KNEE][2]), 'vis_r_knie': float(lm[sa.R_KNEE][2]),
    }
    return d


def lege_resultaten(video_pad, fps=None):
    """Frames zonder landmarks: handmatige annotatie van een onbewerkte video."""
    cap = cv2.VideoCapture(video_pad)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_v = cap.get(cv2.CAP_PROP_FPS) or (fps or 30.0)
    cap.release()
    info = sa.VideoInfo(w=0, h=0, fps=fps_v, totaal=n)
    res = [sa.FrameResultaat(frame_nr=i, tijd=i / fps_v) for i in range(n)]
    for r in res:
        r.pose_gevonden = False
        r.bocht = False
    return info, res


def fase_banden_op_frame(fr, slagen, slag):
    if slag:
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


def rapport(slagen, info, resultaten, out_dir, video_pad):
    """Push- en efficientierapport: CSV per slag + overlayvideo met fasebanden."""
    os.makedirs(out_dir, exist_ok=True)
    events = sa.segmenteer_afzetten(resultaten) if resultaten and resultaten[0].lm_data else []
    csv_pad = os.path.join(out_dir, 'rapport.csv')
    with open(csv_pad, 'w', newline='', encoding='utf-8') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['slag', 'been', 'positionering_start', 'pushing_start',
                       'endpush_start', 'end_frame', 'duur_pos_s', 'duur_duw_s',
                       'duur_eind_s', 'efficientie_pct', 'afzethoek', 'gecorrigeerd'])
        fps = info.fps or 30.0
        for i, s in enumerate(slagen):
            a = s.get('positioning_start', s.get('pushing_start', 0))
            b = s.get('endpush_start', s.get('pushing_start', 0))
            c = s.get('endpush_start', s.get('end_frame', 0))
            d = s.get('end_frame', 0)
            ev = next((e for e in events if e.start_frame == s.get('pushing_start')), None)
            wcsv.writerow([i, s.get('been', '?'), s.get('positioning_start', ''),
                           s.get('pushing_start', ''), s.get('endpush_start', ''),
                           s.get('end_frame', ''),
                           round((b - a) / fps, 2), round((c - b) / fps, 2),
                           round((d - c) / fps, 2), duw_pct(s),
                           ev.hoek if ev else '', bool(s.get('corrected'))])
    mp4 = os.path.join(out_dir, 'overlay.mp4')
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*'mp4v'), info.fps or 30.0,
                             (info.w, info.h))
    cap = cv2.VideoCapture(video_pad)
    for idx, r in enumerate(resultaten):
        ok, fr = cap.read()
        if not ok:
            break
        fr = teken_hulplijnen(fr, r, info.w, info.h)
        slag = None
        for s in slagen:
            if s.get('positioning_start', -1) <= idx <= s.get('end_frame', -1):
                slag = s
                break
        fr = fase_banden_op_frame(fr, slagen, slag)
        writer.write(fr)
    cap.release()
    writer.release()
    return csv_pad, mp4


def teken_hulplijnen(frame, r, w=None, h=None, duw_kant=None, doel=None):
    """Mini-skelet (incl. schouders/hoofd) + heupas + vooruit-as + been-assen + kantelhoeken."""
    lm = r.lm_data if r is not None else None
    if lm is None:
        return frame
    for a, b in SKEL:
        pa, pb = lm.get(a), lm.get(b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, (255, 255, 255), 1)
            for p in (pa, pb):
                cv2.circle(frame, p, 3, (255, 255, 255), -1)
    # schouders + hoofd uit de ruwe landmarks (genormaliseerd → pixels)
    if getattr(r, 'raw_lm', None) is not None and w and h:
        def rp(idx):
            return (int(r.raw_lm[idx][0] * w), int(r.raw_lm[idx][1] * h))
        ls, rs = rp(11), rp(12)                      # L/R schouder
        if r.raw_lm[11][2] > 0.3 and r.raw_lm[12][2] > 0.3:
            cv2.line(frame, ls, rs, (255, 255, 255), 1)
            for p in (ls, rs):
                cv2.circle(frame, p, 3, (255, 255, 255), -1)
            for sidx, heup in ((11, lm['l_heup']), (12, lm['r_heup'])):
                if r.raw_lm[sidx][2] > 0.3:
                    cv2.line(frame, rp(sidx), (lm['l_heup'] if sidx == 11 else lm['r_heup']),
                             (255, 255, 255), 1)
        neus = rp(0)
        if r.raw_lm[0][2] > 0.3:
            cv2.circle(frame, neus, 7, (255, 255, 255), 1)
    # gekozen doelpositie (kies-schaatser): doorlopende kruis-marker
    if doel is not None and w and h:
        px, py = int(doel[0] * w), int(doel[1] * h)
        L = int(0.08 * h) or 20
        cv2.line(frame, (px - L, py), (px + L, py), (0, 255, 0), 3)
        cv2.line(frame, (px, py - L), (px, py + L), (0, 255, 0), 3)
        cv2.circle(frame, (px, py), int(0.11 * h) or 28, (0, 255, 0), 3)

    # COM-proxy: gemiddelde van beide heupen + hoofd (neus)
    neus = None
    if getattr(r, 'raw_lm', None) is not None and w and h and r.raw_lm[0][2] > 0.3:
        neus = (int(r.raw_lm[0][0] * w), int(r.raw_lm[0][1] * h))
    com = None
    if neus is not None:
        com = (int((lm['l_heup'][0] + lm['r_heup'][0] + neus[0]) / 3.0),
               int((lm['l_heup'][1] + lm['r_heup'][1] + neus[1]) / 3.0))
    if com is not None:
        cv2.circle(frame, com, 6, (255, 100, 0), -1)   # oranje-rode stip: zwaartepunt-proxy

    # been-as (magenta): alleen op het duwbeen, door midden(enkel,teen) en de knie
    for kant in ([duw_kant] if duw_kant in ('l', 'r') else []):
        enkel = lm[f'{kant}_enkel']
        teen = lm[f'{kant}_teen']
        knie = lm[f'{kant}_knie']
        voetm = (int((enkel[0] + teen[0]) / 2.0), int((enkel[1] + teen[1]) / 2.0))
        dxl, dyl = knie[0] - voetm[0], knie[1] - voetm[1]
        if abs(dxl) + abs(dyl) < 5:
            continue
        eind = (int(knie[0] + 1.5 * dxl), int(knie[1] + 1.5 * dyl))
        cv2.line(frame, voetm, eind, (255, 0, 255), 2)
        cv2.circle(frame, voetm, 4, (255, 0, 255), -1)
        # tweede lijn (blauw): door hetzelfde voetmidden en de zwaartepunt-proxy,
        # verlengd voorbij de COM — de hoek tussen magenta en blauw is de kandidaat-feature
        if com is not None:
            dcx, dcy = com[0] - voetm[0], com[1] - voetm[1]
            eind2 = (int(com[0] + 0.8 * dcx), int(com[1] + 0.8 * dcy))
            cv2.line(frame, voetm, eind2, (255, 100, 0), 2)

    lh, rh = lm['l_heup'], lm['r_heup']
    cv2.line(frame, lh, rh, (0, 255, 255), 2)
    L = math.hypot(rh[0] - lh[0], rh[1] - lh[1]) or 1.0
    mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    hoek = math.atan2(rh[1] - lh[1], rh[0] - lh[0]) + math.pi / 2
    dx, dy = math.cos(hoek) * L, math.sin(hoek) * L
    cv2.line(frame, (int(mid[0] - dx), int(mid[1] - dy)), (int(mid[0] + dx), int(mid[1] + dy)),
             (255, 255, 0), 2)
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
    huidige = next((s for s in slagen if s['positioning_start'] <= frame_nr <= s['end_frame']), None)
    duw_kant = ('l' if huidige['been'] == 'links' else 'r') if huidige else None
    fr = teken_hulplijnen(fr, r, info.w, info.h, duw_kant)
    lijn = tijdlijn_afbeelding(slagen, info.totaal, speelkop=frame_nr, w=fr.shape[1])
    combined = np.vstack([fr, lijn])
    cv2.imwrite(uit_png, combined)
    print(f'[ROOK] {uit_png}')


try:
    from PySide6.QtWidgets import QLabel  # noqa: F401  (module-level voor KlikLabel)
except ImportError:
    QLabel = object  # rook/--dump-modus zonder Qt importeert dan nog


class KlikLabel(QLabel):
    def __init__(self, *a, **kw):
        super().__init__(*a)
        self.klik_callback = None
        self.klik_modus = False

    def mousePressEvent(self, ev):
        if self.klik_modus and self.klik_callback:
            self.klik_callback(ev.position())
        super().mousePressEvent(ev)


def run_gui(args):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                                   QListWidget, QListWidgetItem, QMainWindow,
                                   QPushButton, QStatusBar, QVBoxLayout, QWidget)

    def slag_index_in(slagen, f):
        for i, s in enumerate(slagen):
            if s.get('positioning_start', -1) <= f <= s.get('end_frame', -1):
                return i
        return None

    app = QApplication(sys.argv)

    class Venster(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cap = None
            self.video_pad = None
            self.info = None
            self.resultaten = []
            self.slagen = []
            self.f = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.volgende)
            self.bewerk_i = None
            self.bewerk_fase = 0
            self.doel = None

            # links: videolijst
            self.videolijst = QListWidget()
            self.videolijst.itemClicked.connect(self.klik_video)
            self.videolijst.setFixedWidth(300)
            self.b_map = QPushButton('Kies map…')
            self.b_map.clicked.connect(self.kies_map)
            self.b_rapport = QPushButton('Rapport + overlay (R)')
            self.b_rapport.clicked.connect(self.maak_rapport)
            self.b_auto = QPushButton('Auto-detect (A)')
            self.b_auto.clicked.connect(self.auto_detect)
            self.b_voor = QPushButton('Voorbewerken (P)')
            self.b_voor.clicked.connect(self.voorbewerk)
            self.b_doel = QPushButton('Kies schaatser (K)')
            self.b_doel.clicked.connect(self.kies_schaatser)
            self.kies_modus = False
            links = QVBoxLayout()
            links.addWidget(self.b_map)
            links.addWidget(self.videolijst, 1)
            links.addWidget(self.b_auto)
            links.addWidget(self.b_voor)
            links.addWidget(self.b_doel)
            links.addWidget(self.b_rapport)
            links_w = QWidget()
            links_w.setLayout(links)

            # midden: video + tijdlijn
            self.video_label = KlikLabel(' kies een video…')
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.klik_callback = self.video_klik
            self.tijd_label = QLabel()
            self.tijd_label.setFixedHeight(40)
            midden = QVBoxLayout()
            midden.addWidget(self.video_label, 1)
            midden.addWidget(self.tijd_label)
            midden_w = QWidget()
            midden_w.setLayout(midden)

            # rechts: fasen
            self.fase_label = QLabel('geen fase')
            self.fase_label.setAlignment(Qt.AlignCenter)
            self.fase_label.setFixedHeight(50)
            self.fase_label.setStyleSheet(
                'font-size: 18px; font-weight: bold; color: white; background: #555;')
            self.lijst = QListWidget()
            self.lijst.itemClicked.connect(self.klik_slag)
            self.lijst.setFixedWidth(430)
            self.b_redefine = QPushButton('Redefine (E)')
            self.b_redefine.clicked.connect(self.toggle_bewerk)
            self.b_new = QPushButton('Nieuw (N)')
            self.b_new.clicked.connect(self.nieuwe_slag)
            self.b_undo = QPushButton('Ongedaan (U)')
            self.b_undo.clicked.connect(self.ongedaan)
            self.b_save = QPushButton('Opslaan (S)')
            self.b_save.clicked.connect(self.opslaan)
            self.b_fasen = []
            for toets, tekst in (('1', '1 Positionering'), ('2', '2 Duwfase'),
                                 ('3', '3 Eind-duw'), ('4', '4 Einde slag')):
                b = QPushButton(tekst)
                b.clicked.connect(lambda _, t=toets: self.zet_fase(t))
                self.b_fasen.append(b)
            fase_knoppen = QHBoxLayout()
            for b in self.b_fasen:
                fase_knoppen.addWidget(b)
            knoppen = QHBoxLayout()
            for b in (self.b_redefine, self.b_new, self.b_undo, self.b_save):
                knoppen.addWidget(b)
            legend = QLabel(LEGENDA)
            legend.setStyleSheet('font-size: 12px; color: #ccc;')
            rechts = QVBoxLayout()
            rechts.addWidget(self.fase_label)
            rechts.addWidget(self.lijst, 1)
            rechts.addLayout(fase_knoppen)
            rechts.addLayout(knoppen)
            rechts.addWidget(legend)
            rechts_w = QWidget()
            rechts_w.setLayout(rechts)

            top = QHBoxLayout()
            top.addWidget(links_w)
            top.addWidget(midden_w, 1)
            top.addWidget(rechts_w)
            w = QWidget()
            w.setLayout(top)
            self.setCentralWidget(w)
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.setWindowTitle('Schaats-werkbank — fasen per slag')
            self.resize(1750, 900)
            for wdgt in (self.videolijst, self.lijst, self.b_redefine, self.b_new,
                         self.b_undo, self.b_save, self.b_map, self.b_rapport,
                         self.b_auto, self.b_voor, self.b_doel):
                wdgt.setFocusPolicy(Qt.NoFocus)
            for b in self.b_fasen:
                b.setFocusPolicy(Qt.NoFocus)
            self.setFocusPolicy(Qt.StrongFocus)
            self.refreshtimer = QTimer(self)
            self.refreshtimer.timeout.connect(self.rij_verversen)
            self.refreshtimer.start(10000)

            self.map = args.map or os.path.expanduser('~/SchaatsAnalyse-album')
            self.vul_videolijst()
            if self.videos:
                self.laad_video(self.videos[0])
            self.toon()

        def vul_videolijst(self):
            self.videos = scan_videos(self.map)
            self.videolijst.blockSignals(True)
            self.videolijst.clear()
            for v in (self.videos or []):
                stem = os.path.splitext(os.path.basename(v))[0]
                p = video_paden(v)
                naam = os.path.basename(v)
                datum = bestandsdatum(v)
                rapport_klaar = ' 📄 rapport' if os.path.exists(
                    os.path.join(p['rapport'], 'rapport.csv')) else ''
                if os.path.exists(p['fasen']):
                    st = laad_fasen_json(p['fasen'])
                    ncorr = sum(1 for s in st if s.get('corrected'))
                    if ncorr:
                        status = f'✍ {ncorr}/{len(st)} slagen gecorrigeerd{rapport_klaar}'
                    else:
                        status = f'✓ {len(st)} slagen gemarkeerd (auto){rapport_klaar}'
                elif os.path.exists(p['npz']):
                    status = f'🤖 auto-detect klaar — nog niet gemarkeerd{rapport_klaar}'
                elif os.path.exists(p['npz'] + '.preparing'):
                    verstreken = int(time.time() - os.path.getmtime(p['npz'] + '.preparing'))
                    if verstreken > 900:
                        status = f'⚠ voorbewerken loopt al {verstreken // 60} min — check PinkBox'
                    else:
                        status = f'⏳ wordt voorbereid… ({verstreken}s)'
                else:
                    status = '○ geen analyse — handmatig of P voor automatisch'
                self.videolijst.addItem(QListWidgetItem(f'{naam}  {datum}\n{status}'))
            self.videolijst.blockSignals(False)
            # geladen video opnieuw selecteren zodat de lijst meegaat
            if getattr(self, 'video_pad', None):
                for i, v in enumerate(self.videos):
                    if v == self.video_pad:
                        self.lijst_row = i
                        self.videolijst.setCurrentRow(i)
                        break

        def kies_map(self):
            d = QFileDialog.getExistingDirectory(self, 'Kies videomap', self.map)
            if d:
                self.map = d
                self.vul_videolijst()
                if self.videos:
                    self.laad_video(self.videos[0])
                    self.toon()

        def klik_video(self, item):
            self.timer.stop()
            self.laad_video(self.videos[self.videolijst.row(item)])
            self.toon()

        def laad_video(self, video_pad):
            # data (npz/fasen) hoort bij het ORIGINELE pad; de 2x-versie is alleen de speel-kopie
            self.video_pad = video_pad
            stem2x = os.path.splitext(os.path.basename(video_pad))[0] + '_2x.mp4'
            kandidaat2x = os.path.join(self.map, '2x', stem2x)
            self.speel_pad = kandidaat2x if os.path.exists(kandidaat2x) else video_pad
            p = video_paden(video_pad)
            os.makedirs(os.path.join(self.map, 'npz'), exist_ok=True)
            os.makedirs(os.path.join(self.map, 'fasen'), exist_ok=True)
            self.cap = cv2.VideoCapture(self.speel_pad)
            if not self.cap.isOpened():
                self.status.showMessage(f'video niet te lezen: {self.speel_pad}', 6000)
                return
            self.bewerk_i = None
            self.bewerk_fase = 0
            self.doel = None
            self.f = 0
            self.timer.stop()
            if os.path.exists(p['npz']):
                try:
                    self.info, self.resultaten = laad_context(p['npz'])
                except Exception as e:
                    self.status.showMessage(f'npz onleesbaar ({e}) — handmatige modus', 6000)
                    self.info, self.resultaten = lege_resultaten(video_pad)
                n_vid = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                if n_vid and n_vid < self.info.totaal:
                    # gedeeltelijke/kapotte 2x-kopie: terugvallen op het origineel
                    self.speel_pad = video_pad
                    self.cap.release()
                    self.cap = cv2.VideoCapture(video_pad)
                    n_vid = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                if n_vid in (0, self.info.totaal):
                    # mp4v-metadata bevat vaak geen frame count: tellen met grab()
                    n_vid = 0
                    while self.cap.grab():
                        n_vid += 1
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if n_vid and self.info.totaal and n_vid != self.info.totaal:
                    try:
                        self.info, self.resultaten = hervorm_resultaten(
                            self.resultaten, self.info, n_vid)
                    except Exception as e:
                        self.status.showMessage(f'hersampling mislukt ({e}) — origineel geladen', 6000)
                        self.info, self.resultaten = laad_context(p['npz'])
            else:
                self.info, self.resultaten = lege_resultaten(video_pad)
            if os.path.exists(p['fasen']):
                self.slagen = laad_fasen_json(p['fasen'])
            elif self.resultaten and self.resultaten[0].lm_data is not None:
                self.slagen = automatische_fasen(self.resultaten, self.info, args.rotatie)
            else:
                self.slagen = []
            self.vul_lijst()
            self.status.showMessage(f'geladen: {os.path.basename(video_pad)}', 4000)

        def vul_lijst(self):
            self.lijst.blockSignals(True)
            self.lijst.clear()
            for i, s in enumerate(self.slagen):
                pct = duw_pct(s)
                vlag = '  [GEWIJZIGD]' if s.get('corrected') else ''
                item = QListWidgetItem(
                    f"slag {i:>2} {s.get('been', '?'):<6} f{s.get('positioning_start', '?')}-"
                    f"{s.get('end_frame', '?')}  efficientie {pct if pct is not None else '—'}%{vlag}")
                if s.get('corrected'):
                    item.setBackground(Qt.yellow)
                if i == self.bewerk_i:
                    item.setText(item.text() + '  << BEWERKEN >>')
                    item.setBackground(Qt.red)
                self.lijst.addItem(item)
            self.lijst.blockSignals(False)

        def auto_detect(self):
            if self.video_pad is None:
                return
            p = video_paden(self.video_pad)
            if not os.path.exists(p['npz']):
                self.status.showMessage('geen npz — gebruik eerst Voorbewerken', 4000)
                return
            self.info, self.resultaten = laad_context(p['npz'])
            self.slagen = automatische_fasen(self.resultaten, self.info, args.rotatie)
            self.bewerk_i = None
            self.bewerk_fase = 0
            self.vul_lijst()
            self.vul_videolijst()
            self.toon()

        def voorbewerk(self):
            if self.video_pad is None:
                return
            p = video_paden(self.video_pad)
            if os.path.exists(p['npz']):
                self.status.showMessage('deze video is al voorbereid', 3000)
                return
            import subprocess, shlex
            self.status.showMessage('video naar PinkBox sturen voor voorbewerking…')
            QApplication.processEvents()
            cmd = (f'/usr/bin/scp -q {shlex.quote(self.video_pad)} '
                   f'"pinkbox:C:/Users/danie/GitHub/SchaatsAnalyse/runs/batch/" '
                   '&& /usr/bin/ssh pinkbox "schtasks /Run /TN SchaatsBatchA"')
            os.makedirs(os.path.join(self.map, 'npz'), exist_ok=True)
            open(p['npz'] + '.preparing', 'w').write(str(int(time.time())))
            subprocess.Popen(['/bin/zsh', '-c', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.vul_videolijst()
            self.status.showMessage('voorbewerking gestart op PinkBox — npz volgt via sync, daarna A', 8000)

        def kies_schaatser(self):
            if self.video_pad is None:
                self.status.showMessage('geen video geladen', 3000)
                return
            self.kies_modus = not self.kies_modus
            self.video_label.klik_modus = self.kies_modus
            self.status.showMessage(
                'KLIK op de schaatser die gevolgd moet worden — daarna wordt de clip '
                'opnieuw geanalyseerd (paar minuten op PinkBox, daarna A)'
                if self.kies_modus else 'kies-modus uit', 8000)

        def video_klik(self, label_pos):
            """Klik in 'kies schaatser'-modus → doel opgeven en heranalyseren op PinkBox."""
            if not self.kies_modus or self.video_pad is None or not self.info:
                return
            pm = self.video_label.pixmap()
            if pm is None or not pm.width():
                return
            px, py = label_pos.x(), label_pos.y()
            if not (0 <= px < pm.width() and 0 <= py < pm.height()):
                return
            doel = (round(px / pm.width(), 4), round(py / pm.height(), 4))
            stem = os.path.splitext(os.path.basename(self.video_pad))[0]
            req = json.dumps({'stem': stem, 'doel': list(doel)})
            reqpad = f'/tmp/doel_verzoek_{stem}.json'
            with open(reqpad, 'w') as fh:
                fh.write(req)
            p = video_paden(self.video_pad)
            os.makedirs(os.path.dirname(p['npz']), exist_ok=True)
            open(p['npz'] + '.preparing', 'w').write(str(int(time.time())))
            QApplication.processEvents()
            cmd = (f'/usr/bin/scp -q {shlex.quote(reqpad)} '
                   '"pinkbox:C:/Users/danie/GitHub/SchaatsAnalyse/runs/doel_verzoek.json" '
                   '&& /usr/bin/ssh pinkbox "schtasks /Run /TN SchaatsDoel"')
            subprocess.Popen(['/bin/zsh', '-c', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.kies_modus = False
            self.video_label.klik_modus = False
            self.doel = doel
            self.vul_videolijst()
            self.toon()
            self.status.showMessage(
                f'doel gezet op {doel} — heranalyse gestart op PinkBox, npz volgt via sync (daarna A)',
                10000)

        def rij_verversen(self):
            # status van de lijst periodiek verversen (npz kan tussentijds aankomen)
            if self.map and self.videos:
                self.vul_videolijst()

        def maak_rapport(self):
            if self.video_pad is None or not self.info:
                self.status.showMessage('geen video geladen', 3000)
                return
            p = video_paden(self.video_pad)
            self.status.showMessage('rapport wordt gegenereerd (overlayvideo kan even duren)…')
            QApplication.processEvents()
            try:
                csv_pad, mp4 = rapport(self.slagen, self.info, self.resultaten,
                                       p['rapport'], self.video_pad)
            except Exception as e:
                self.status.showMessage(f'rapport mislukt: {e}', 6000)
                return
            self.opslaan(stil=True)
            self.status.showMessage(f'rapport: {csv_pad} + {mp4}', 8000)

        def opslaan(self, stil=False):
            if self.video_pad is None:
                return
            p = video_paden(self.video_pad)
            os.makedirs(os.path.dirname(p['fasen']), exist_ok=True)
            data = {'video': self.video_pad, 'fps': self.info.fps,
                    'rotatie_graden': args.rotatie, 'strokes': self.slagen}
            with open(p['fasen'], 'w') as fh:
                json.dump(data, fh, indent=1)
            if not stil:
                self.status.showMessage(f'opgeslagen: {p["fasen"]}', 5000)

        def klik_slag(self, item):
            self.timer.stop()
            self.f = self.slagen[self.lijst.row(item)].get('positioning_start', self.f)
            self.toon()

        def toggle_bewerk(self):
            if self.bewerk_i is not None:
                self.bewerk_i = None
            else:
                row = self.lijst.currentRow()
                if row < 0:
                    row = slag_index_in(self.slagen, self.f)
                    if row is None:
                        vorige = [i for i, s in enumerate(self.slagen)
                                  if s.get('positioning_start', 0) <= self.f]
                        row = vorige[-1] if vorige else (0 if self.slagen else None)
                if row is None:
                    self.status.showMessage('geen slagen — druk N voor een nieuwe slag', 4000)
                    return
                self.lijst.setCurrentRow(row)
                self.bewerk_i = row
                self.bewerk_fase = 0
                self.f = self.slagen[row]['positioning_start']
                self.timer.stop()
            self.vul_lijst()
            self.toon()

        def nieuwe_slag(self):
            s = {'been': '?', 'positioning_start': self.f, 'pushing_start': self.f,
                 'endpush_start': self.f, 'end_frame': self.f, 'corrected': True}
            self.slagen.append(s)
            self.slagen.sort(key=lambda x: x.get('positioning_start', 0))
            self.bewerk_i = self.slagen.index(s)
            self.bewerk_fase = 0
            self.timer.stop()
            self.vul_lijst()
            self.toon()

        def zet_fase(self, toets):
            if self.bewerk_i is None:
                i = slag_index_in(self.slagen, self.f)
                if i is None:
                    self.status.showMessage('geen slag op dit frame — E of N eerst', 3000)
                    return
                self.bewerk_i = i
                self.bewerk_fase = 0
            s = self.slagen[self.bewerk_i]
            s[FASEN_KEYS[toets]] = self.f
            s['corrected'] = True
            self.bewerk_fase = min(3, int(toets))
            self.vul_lijst()
            self.toon()

        def ongedaan(self):
            if self.bewerk_i is not None:
                s = self.slagen[self.bewerk_i]
                if self.resultaten and self.resultaten[0].lm_data is not None:
                    vers = automatische_fasen(self.resultaten, self.info, args.rotatie)
                    for v in vers:
                        if v['positioning_start'] == s['positioning_start']:
                            s.update(v)
                            s['corrected'] = False
                            break
                self.vul_lijst()
                self.toon()

        def toon(self):
            if self.video_pad is None or not self.info:
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
            ok, fr = self.cap.read()
            if not ok:
                self.timer.stop()
                return
            r = self.resultaten[self.f] if self.f < len(self.resultaten) else None
            duw_slag = self.slagen[self.bewerk_i] if self.bewerk_i is not None else None
            if duw_slag is None:
                i = slag_index_in(self.slagen, self.f)
                duw_slag = self.slagen[i] if i is not None else None
            duw_kant = ('l' if duw_slag.get('been') == 'links' else 'r') \
                if duw_slag and duw_slag.get('been') in ('links', 'rechts') else None
            fr = teken_hulplijnen(fr, r, self.info.w, self.info.h, duw_kant, doel=self.doel)
            if self.kies_modus:
                cv2.rectangle(fr, (self.info.w // 2 - 380, 10), (self.info.w // 2 + 380, 96),
                              (0, 255, 0), -1)
                cv2.putText(fr, 'KIES SCHAATSER: KLIK OP DE SCHAATSER', (self.info.w // 2 - 350, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4)
            namen = {'positionering': 'POSITIONERING (inefficiënt)', 'duw': 'DUWFASE (efficiënt)',
                     'eind': 'EIND-DUW (inefficiënt)', None: 'geen fase'}
            if self.bewerk_i is not None:
                volgorde = list(FASEN_KEYS.values())
                fase = {'positioning_start': 'positionering', 'pushing_start': 'duw',
                        'endpush_start': 'eind', 'end_frame': 'eind'}[volgorde[self.bewerk_fase]]
                gezet = [kk for kk in volgorde if kk in duw_slag]
                fr = teken_fase_banner(fr, fase, bewerken=True, gezet=len(gezet),
                                       tekst_extra=f'druk {self.bewerk_fase + 1} op frame {self.f}',
                                       fase_frame=fase_van(duw_slag, self.f))
            else:
                fase = fase_van(duw_slag, self.f)
                fr = teken_fase_banner(fr, fase, bewerken=False)
            fr = fase_banden_op_frame(fr, self.slagen, duw_slag)
            schaal = (self.video_label.height() or 700) / fr.shape[0]
            klein = cv2.resize(fr, (int(fr.shape[1] * schaal), int(fr.shape[0] * schaal)))
            rgb = cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
            self.video_label.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                                                rgb.strides[0], QImage.Format_RGB888)))
            tijd = tijdlijn_afbeelding(self.slagen, self.info.totaal,
                                       slag_index_in(self.slagen, self.f), self.bewerk_i, self.f,
                                       w=max(600, self.video_label.width()))
            rgbt = cv2.cvtColor(tijd, cv2.COLOR_BGR2RGB)
            self.tijd_label.setPixmap(QPixmap.fromImage(QImage(rgbt.data, rgbt.shape[1], rgbt.shape[0],
                                                               rgbt.strides[0], QImage.Format_RGB888)))
            self.lijst.blockSignals(True)
            if self.bewerk_i is not None:
                self.lijst.setCurrentRow(self.bewerk_i)
            elif duw_slag is not None:
                self.lijst.setCurrentRow(self.slagen.index(duw_slag))
            self.lijst.blockSignals(False)
            if self.bewerk_i is not None:
                volgorde = list(FASEN_KEYS.values())
                fase_naam = {'positioning_start': 'POSITIONERING (inefficiënt)',
                             'pushing_start': 'DUWFASE (efficiënt)',
                             'endpush_start': 'EIND-DUW (inefficiënt)',
                             'end_frame': 'EINDE SLAG'}[volgorde[self.bewerk_fase]]
                gezet = [kk for kk in volgorde if kk in duw_slag]
                self.fase_label.setText(
                    f'DEFINIEER slag {self.bewerk_i}: {fase_naam} — druk {self.bewerk_fase + 1} '
                    f'({len(gezet)}/4 gezet)\ndit frame: {namen[fase_van(duw_slag, self.f)]}')
            else:
                self.fase_label.setText(namen[fase_van(duw_slag, self.f)])
            self.status.showMessage(
                f'{os.path.basename(self.video_pad)}  frame {self.f}/{self.info.totaal} '
                f'({self.f / (self.info.fps or 30):.2f}s)')

        def volgende(self):
            if self.info and self.f < (self.info.totaal or 1) - 1:
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
                self.f = min(self.f + 1, (self.info.totaal or 1) - 1)
                self.toon()
            elif k == Qt.Key_Up:
                self.f = max(0, self.f - 10)
                self.toon()
            elif k == Qt.Key_Down:
                self.f = min(self.f + 10, (self.info.totaal or 1) - 1)
                self.toon()
            elif k == Qt.Key_Space:
                if self.timer.isActive():
                    self.timer.stop()
                else:
                    self.timer.start(int(1000 / (self.info.fps or 30)))
            elif k == Qt.Key_E:
                self.toggle_bewerk()
            elif k == Qt.Key_N:
                self.nieuwe_slag()
            elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
                self.zet_fase(ev.text())
            elif k == Qt.Key_U:
                self.ongedaan()
            elif k == Qt.Key_R:
                self.maak_rapport()
            elif k == Qt.Key_A:
                self.auto_detect()
            elif k == Qt.Key_P:
                self.voorbewerk()
            elif k == Qt.Key_K:
                self.kies_schaatser()
            elif k == Qt.Key_S:
                self.opslaan()
            elif k == Qt.Key_Escape:
                if self.bewerk_i is not None:
                    self.bewerk_i = None
                    self.bewerk_fase = 0
                    self.status.showMessage('Redefine-modus afgebroken', 3000)
                    self.vul_lijst()
                    self.toon()
            elif k == Qt.Key_Q:
                self.close()

    venster = Venster()
    venster.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasen-marker v3')
    parser.add_argument('--input', default=None)
    parser.add_argument('--npz', default=None)
    parser.add_argument('--uit', default='fasen.json')
    parser.add_argument('--map', default=None, help='werkbank: map met videos + npz/ + fasen/')
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
