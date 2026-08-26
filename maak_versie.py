"""
maak_versie.py — legt de git-stempel vast in een gegenereerde `_versie.py`.

Waarom dit bestaat: `app_versie()` in schaats_db.py draait `git` in de map naast het
script om te bepalen met wélke code een analyse gemaakt is (Info-dialoog, titel-tooltip,
`instellingen_json`). In een gebundelde .exe is er geen git en geen repo, dus zou elke
analyse van een collega **zonder versiestempel** in de bibliotheek belanden. Het
buildscript draait daarom dit script vlak voor PyInstaller; `_versie_uit_bundel()` leest
het resultaat als `sys.frozen` waar is, en valt anders terug op de git-route.

Het label-formaat blijft exact `"2026-08-24 · 4df9ab5a"`, zodat analyses uit de exe en
uit de repo onderling vergelijkbaar blijven — daarom worden hier dezelfde git-commando's
met dezelfde vlaggen gedraaid als in `app_versie()` (`--abbrev=8`, `-uno`).

Het gegenereerde bestand staat in `.gitignore`: het is een build-artefact, en in de
repo-omgeving wordt het nooit gelezen (`_versie_uit_bundel()` draait alleen bevroren).

Gebruik:  python maak_versie.py [doelpad]
          python maak_versie.py --toon      (alleen afdrukken, niets schrijven)

Die tweede vorm is er voor `bouw.bat`: Inno Setup krijgt de stempel als /DVersie=…
mee en zet hem in "Apps en onderdelen". Hij drukt bewust **ASCII** af (punt in
plaats van de middenstip, geen spatie), want dit gaat door een for-lus in cmd.exe
en over een commandoregel — het scheidingsteken van `label` overleeft de
console-codepage niet. Het label in de bibliotheek blijft ongewijzigd.
"""

import os
import subprocess
import sys

HIER = os.path.dirname(os.path.abspath(__file__))
STANDAARD_DOEL = os.path.join(HIER, "_versie.py")


def _git(*args):
    """Zelfde helper als in schaats_db.py: "" bij elke fout (geen git, geen repo)."""
    try:
        r = subprocess.run(["git", "-C", HIER] + list(args),
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def git_stempel():
    """(commit, datum, vuil) uit de repo naast dit script; lege strings buiten een repo."""
    commit = datum = ""
    uit = _git("log", "-1", "--abbrev=8", "--format=%h%x09%cs")
    if "\t" in uit:
        commit, datum = uit.split("\t", 1)
    # Untracked bestanden tellen niet als "vuil" (-uno): video's en npz's zeggen niets
    # over de gedraaide logica.
    vuil = bool(commit) and bool(_git("status", "--porcelain", "-uno"))
    return commit, datum, vuil


def ascii_stempel():
    """`2026-08-25.a4be1f2b+`, of "onbekend" buiten een git-repo. Zie de docstring."""
    commit, datum, vuil = git_stempel()
    if not commit:
        return "onbekend"
    return f"{datum}.{commit}{'+' if vuil else ''}"


def schrijf(doel=STANDAARD_DOEL):
    commit, datum, vuil = git_stempel()
    inhoud = (
        '"""Gegenereerd door maak_versie.py — niet met de hand bewerken, niet in git."""\n'
        f'COMMIT = {commit!r}\n'
        f'DATUM = {datum!r}\n'
        f'VUIL = {vuil!r}\n'
    )
    with open(doel, "w", encoding="utf-8") as f:
        f.write(inhoud)
    label = f"{datum} · {commit}{'+' if vuil else ''}" if commit else "(geen git-stempel)"
    return doel, label


if __name__ == "__main__":
    if "--toon" in sys.argv[1:]:
        print(ascii_stempel())
        raise SystemExit(0)
    doel = sys.argv[1] if len(sys.argv) > 1 else STANDAARD_DOEL
    pad, label = schrijf(doel)
    print(f"{pad}  ->  {label}")
