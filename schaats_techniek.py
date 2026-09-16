"""
Schaats-techniek: kantel-metriek (straight-line mode)
=====================================================
Meet per slag hoeveel procent van de frames de schaats kantelt naar het
lichaamszwaartepunt (binnenkant-rollen): linkerschaats naar rechts (naar het midden),
rechterschaats naar links.

Signaal (frontaal beeld): de enkel staat bovenop de schaats. Rolts de schaats naar een
binnenrand, dan schuift de enkel náár het lichaamsmidden toe t.o.v. het contactpunt van
de schaats (midden tussen hiel en teen). De kantelhoek is de hoek van de lijn
contactpunt→enkel met de verticaal in het beeldvlak; naar-binnen = de horizontale
kantelrichting wijst naar het heupmidden.

Beperkingen (bewust genoteerd, zie ook de meetdiscipline in GPU.md):
- 2D-projectie: de kantelhoek is de *beeld*hoek, geen echte rollhoek in 3D.
- Zwaartepunt-proxy: midden van beide heupen. Bij een sterke heupverplaatsing in de
  afzet beweegt de referentie mee — dat is precies de bedoeling ("waar de body is"),
  maar het maakt de metriek gevoelig voor de heup-schatting.
- Alleen meetbaar waar RTMPose hiel/teen écht gezien heeft; op de YOLO-route liggen
  hiel/teen op de enkel en is de hoek ongedefinieerd.

CLI:
    python schaats_techniek.py analyse.npz [--doodzone 3.0] [--grafiek uit.png]

Uitvoer: per afzet (slag) het percentage meetbare frames met binnenkant-kanteling,
plus de gemiddelde/maximale kantelhoek, en per schaats de totalen over de hele
video. Grafiek: kantelhoek over tijd met de naar-binnen-frames gemarkeerd.
"""
import argparse
import json
import math
import sys

import schaats_analyse as sa

MIN_AFSTAND_PX = 8.0   # hiel/teen minder dan zoveel px van de enkel = niet gezien (YOL-route)
MIN_DY_PX = 2.0        # enkel moet echt boven het contactpunt staan


def _afstand(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def kantelhoek(lm, kant, doodzone_graden=3.0):
    """Kantelhoek van één schaats in één frame.

    kant: 'l' of 'r'. Retourneert None als het frame niet meetbaar is, anders
    (hoek_in_graden, naar_binnen: bool). Positieve hoek = kantelt naar beeldrechts.
    """
    if lm is None:
        return None
    enkel = lm[f'{kant}_enkel']
    hiel = lm[f'{kant}_hiel']
    teen = lm[f'{kant}_teen']
    # Op de YOLO-route liggen hiel/teen óp de enkel (visibility 0): geen kantelsignaal.
    if _afstand(hiel, enkel) < MIN_AFSTAND_PX and _afstand(teen, enkel) < MIN_AFSTAND_PX:
        return None
    contact = ((hiel[0] + teen[0]) / 2.0, (hiel[1] + teen[1]) / 2.0)
    dy = contact[1] - enkel[1]        # hoogte van de enkel boven het contactpunt (y omlaag)
    if dy <= MIN_DY_PX:
        return None
    hoek = math.degrees(math.atan2(enkel[0] - contact[0], dy))
    zwaarte = (lm['l_heup'][0] + lm['r_heup'][0]) / 2.0 - contact[0]
    naar_binnen = abs(hoek) > doodzone_graden and (hoek * zwaarte) > 0
    return hoek, naar_binnen


def slag_overzicht(resultaten, events, doodzone_graden=3.0):
    """Per afzetgebeurtenis het kanteloverzicht van de afzetbeen-schaats."""
    rijen = []
    for ev in events:
        kant = 'l' if ev.been == 'links' else 'r'
        tot = mee = binnen = 0
        h_sum = 0.0
        h_maxabs = 0.0
        for f in range(ev.start_frame, min(ev.eind_frame + 1, len(resultaten))):
            r = resultaten[f]
            if r.bocht or not r.pose_gevonden:
                continue
            u = kantelhoek(r.lm_data, kant, doodzone_graden)
            if u is None:
                continue
            hoek, naar_binnen = u
            tot += 1
            h_sum += hoek
            h_maxabs = max(h_maxabs, abs(hoek))
            if naar_binnen:
                binnen += 1
        rijen.append({
            'index': ev.index, 'been': ev.been,
            'start_frame': ev.start_frame, 'eind_frame': ev.eind_frame,
            'meetbaar': tot,
            'pct_naar_binnen': round(100.0 * binnen / tot, 1) if tot else None,
            'gem_hoek': round(h_sum / tot, 1) if tot else None,
            'max_abs_hoek': round(h_maxabs, 1) if tot else None,
        })
    return rijen


def totaal_per_schaats(resultaten, doodzone_graden=3.0):
    """Kantelpercentage over álle meetbare, niet-bocht frames, per schaats."""
    totaal = {}
    for kant in ('l', 'r'):
        tot = mee = binnen = 0
        for r in resultaten:
            if r.bocht or not r.pose_gevonden:
                continue
            u = kantelhoek(r.lm_data, kant, doodzone_graden)
            if u is None:
                continue
            hoek, naar_binnen = u
            tot += 1
            if naar_binnen:
                binnen += 1
        totaal[kant] = {
            'meetbaar': tot,
            'pct_naar_binnen': round(100.0 * binnen / tot, 1) if tot else None,
        }
    return totaal


def grafiek(resultaten, pad, doodzone_graden=3.0):
    """Kantelhoek over tijd voor beide schaatsen; naar-binnen-frames gemarkeerd."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t, l_hoek, r_hoek, l_in, r_in = [], [], [], [], []
    for r in resultaten:
        t.append(r.tijd)
        for kant, h, i in (('l', l_hoek, l_in), ('r', r_hoek, r_in)):
            u = kantelhoek(r.lm_data, kant, doodzone_graden) if r.pose_gevonden and not r.bocht else None
            h.append(u[0] if u else np.nan)
            i.append(u[1] if u else False)

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.array(t)
    for kant, h, i, kleur in (('l', l_hoek, l_in, 'tab:blue'), ('r', r_hoek, r_in, 'tab:red')):
        hh = np.array(h, dtype=float)
        ii = np.array(i, dtype=bool)
        ax.plot(t, hh, color=kleur, lw=0.8, alpha=0.9, label=f'{"links" if kant=="l" else "rechts"}')
        ax.fill_between(t, 0, hh, where=ii, color=kleur, alpha=0.25,
                        label=f'{"links" if kant=="l" else "rechts"} → naar zwaartepunt')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(doodzone_graden, color='gray', lw=0.5, ls='--')
    ax.axhline(-doodzone_graden, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('tijd (s)')
    ax.set_ylabel('kantelhoek (°, + = naar beeldrechts)')
    ax.set_title('Schaatskanteling — gevulde gebieden = kantelt naar het zwaartepunt')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kantel-metriek per slag (straight)')
    parser.add_argument('npz')
    parser.add_argument('--doodzone', type=float, default=3.0,
                        help='kantelhoeken kleiner dan dit aantal graden tellen als vlak (standaard 3.0)')
    parser.add_argument('--grafiek', default=None, metavar='PNG')
    args = parser.parse_args()

    info, resultaten = sa.laad_landmarks(args.npz)
    sa.verwerk_afgeleiden(resultaten, info.w, info.h, info.fps, 5, 0.015)
    events = sa.segmenteer_afzetten(resultaten)

    rijen = slag_overzicht(resultaten, events, args.doodzone)
    totaal = totaal_per_schaats(resultaten, args.doodzone)

    print(f'\n== Kantel-metriek: {args.npz} ==')
    print(f'  dekking: {sum(1 for r in resultaten if r.pose_gevonden)}/{info.totaal} frames\n')
    print('  slag | been   | frames | meetbaar | %% naar zwaartepunt | gem. hoek | max |hoek|')
    for r in rijen:
        if r['meetbaar'] == 0:
            print(f"  {r['index']:>4} | {r['been']:<6} | geen meetbare frames")
            continue
        print(f"  {r['index']:>4} | {r['been']:<6} | f{r['start_frame']}-{r['eind_frame']:<4} | "
              f"{r['meetbaar']:>8} | {r['pct_naar_binnen']:>5.1f}%% "
              f"(gem {r['gem_hoek']:+.1f}°, max {r['max_abs_hoek']:.1f}°)")
    print(f'\n  Totaal (hele video, niet-bocht frames):')
    for kant, naam in (('l', 'linkerschaats'), ('r', 'rechterschaats')):
        t = totaal[kant]
        if t['pct_naar_binnen'] is not None:
            print(f"  {naam}: {t['pct_naar_binnen']}% van {t['meetbaar']} meetbare frames naar zwaartepunt")

    if args.grafiek:
        fig = grafiek(resultaten, args.grafiek)
        fig.savefig(args.grafiek, dpi=150)
        print(f'[INFO] grafiek: {args.grafiek}')
