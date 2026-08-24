"""
schaats_db.py — de bibliotheek (fase 1): schaatser-profielen + opgeslagen analyses.

Eén bibliotheekmap (pad instelbaar via config, later deelbaar via een cloudmap):

    <bibliotheek>/
      schaats.db                  ← SQLite: schaatsers, analyses, events-cache
      media/<analyse-uuid>/
        <originele videonaam>     ← gekopieerd origineel
        landmarks.npz             ← gesmoothte landmarks (fase 0-serialisatie)
      opnames/                    ← ruwe trainingsopnames, nog te knippen (fase 8)

Alle SQL en padlogica leeft hier; de GUI praat alleen met deze module.

Cloudmap-discipline (maakt delen via Google Drive/OneDrive/Dropbox mogelijk, fase 4):
- journal_mode=DELETE (géén WAL: -wal/-shm-nevenbestanden syncen half → corruptiegevaar);
- verbindingen worden per aanroep geopend en gesloten, nooit open gehouden — het
  DB-bestand is vrijwel altijd "in rust" voor de syncer, en sqlite3-objecten kruisen
  zo ook nooit een thread (opslaan draait in de GUI-workerthread);
- busy_timeout voor het zeldzame geval dat twee trainers tegelijk schrijven;
- alle paden in de DB relatief t.o.v. de bibliotheekmap, met forward slashes.

Stdlib + numpy (indirect, via schaats_analyse); importeerbaar in beide venvs.
Zelftest zonder video of GUI: `python schaats_db.py`.
"""

import json
import os
import random
import shutil
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import date

from schaats_analyse import (sla_landmarks_op, laad_landmarks, video_info,
                             ONV_AFGEKAPT, ONV_GEEN_PUSH)

DB_NAAM     = "schaats.db"
MEDIA_MAP   = "media"
OPNAMES_MAP = "opnames"              # ruwe, nog niet geknipte opnames (fase 8)
NPZ_NAAM    = "landmarks.npz"
NPZ_RUW_NAAM = "landmarks_ruw.npz"   # pristine landmarks vóór de eerste handmatige edit (fase 3)
SCHEMA_VERSIE = 4   # v2 (fase 4): analyse.video_bytes; v3 (fase 8): bronvideo + analyse.bron_*;
                    # v4: bron_markering (punten van het handmatige kijkvenster)
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".wmv")

# Status van een opname in de werklijst (fase 8). Bewust handmatig: het programma kan niet
# weten of de trainer een opname áf vindt, dus er wordt nooit automatisch iets op 'klaar'
# gezet — het toont alleen de telling (`3 fragmenten · 2 analyses`).
BRON_STATUSSEN = ("nog doen", "bezig", "klaar", "onbruikbaar")
BRON_STATUS_DEFAULT = BRON_STATUSSEN[0]
# Tekstvlaggen in afzet_event_cache.opmerking voor een afzet die zichtbaar blijft maar
# buiten de gemiddelde hoek valt (`AfzetEvent.onvolledig`). De teksten zijn die van
# schaats_analyse zelf, zodat caches van vóór de tweede reden gewoon blijven werken:
# `ONV_AFGEKAPT` is nog steeds letterlijk "afgekapt".
AFGEKAPT_MARKER = ONV_AFGEKAPT
ONVOLLEDIG_MARKERS = (ONV_AFGEKAPT, ONV_GEEN_PUSH)
ENV_BIBLIOTHEEK = "SCHAATSANALYSE_BIBLIOTHEEK"   # override voor tests


# ── Config (per gebruiker, dus lokaal — niet in de gedeelde map) ───────────────

def config_pad():
    basis = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(basis, "SchaatsAnalyse", "config.json")


def standaard_bibliotheek():
    return os.path.join(os.path.expanduser("~"), "Documents", "SchaatsAnalyse")


def laad_config():
    """Leest config.json; onleesbaar/afwezig → defaults (nooit crashen bij opstarten)."""
    try:
        with open(config_pad(), encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config is geen dict")
    except Exception:
        cfg = {}
    cfg.setdefault("bibliotheek_pad", standaard_bibliotheek())
    cfg.setdefault("trainer_naam", "")    # fase 4: gaat mee als analyse.aangemaakt_door
    return cfg


def bewaar_config(cfg):
    """Schrijft config.json atomair (tmp + os.replace), zodat een crash halverwege
    nooit een half/corrupt configbestand achterlaat."""
    pad = config_pad()
    os.makedirs(os.path.dirname(pad), exist_ok=True)
    tmp = pad + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, pad)


def bibliotheek_pad():
    env = os.environ.get(ENV_BIBLIOTHEEK)
    if env:
        return env
    return laad_config()["bibliotheek_pad"]


def trainer_naam():
    """De naam van de huidige trainer (fase 4), leeg als niet ingesteld. Wordt bij
    nieuwe analyses als aangemaakt_door bewaard, zodat in een gedeelde bibliotheek
    zichtbaar is wie welke analyse maakte."""
    return (laad_config().get("trainer_naam") or "").strip()


# ── Appversie (welke code heeft deze analyse gedraaid?) ────────────────────────

_app_versie_cache = None


def _git(*args):
    """Draait een git-commando in de repomap; "" bij elke fout (geen git, geen repo,
    timeout). CREATE_NO_WINDOW voorkomt een console-flits vanuit de GUI op Windows."""
    try:
        r = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__))] + list(args),
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def app_versie():
    """Met welke versie van de app draait deze analyse? → dict met `commit` (korte
    hash), `datum` (commitdatum ISO), `vuil` (ongecommitte wijzigingen) en `label`
    ("2026-08-05 · 7e013fb2+"). Buiten een git-repo zijn alle velden leeg.

    De trackinglogica wijzigt tijdens het ontwikkelen regelmatig, dus van een opgeslagen
    analyse moet achteraf te zien zijn welke code hem maakte. Bewust uit git en niet uit
    een handmatig opgehoogde constante: die loopt juist tijdens snel ontwikkelen achter
    en liegt dan. De commitdatum is het mens-leesbare deel voor een trainer, de hash het
    precieze deel om `git show` op te doen. `vuil` (de `+`) telt untracked bestanden niet
    mee — video's en npz's naast de code zeggen niets over de gedraaide logica.

    Eén keer per proces gemeten (subprocess kost tijd; de code wijzigt niet tijdens een
    draaiende sessie)."""
    global _app_versie_cache
    if _app_versie_cache is None:
        commit = datum = ""
        uit = _git("log", "-1", "--abbrev=8", "--format=%h%x09%cs")
        if "\t" in uit:
            commit, datum = uit.split("\t", 1)
        vuil = bool(commit) and bool(_git("status", "--porcelain", "-uno"))
        label = f"{datum} · {commit}{'+' if vuil else ''}" if commit else ""
        _app_versie_cache = {"commit": commit, "datum": datum,
                             "vuil": vuil, "label": label}
    return dict(_app_versie_cache)


# ── Verbinding + schema ────────────────────────────────────────────────────────

@contextmanager
def _verbind(bieb):
    """Kortstondige verbinding: openen → transactie → sluiten (zie moduledocstring)."""
    con = sqlite3.connect(os.path.join(bieb, DB_NAAM), timeout=5.0)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA foreign_keys=ON")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


_SCHEMA = """
CREATE TABLE schaatser(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    naam          TEXT NOT NULL,
    geboortejaar  INTEGER,
    notities      TEXT NOT NULL DEFAULT '',
    aangemaakt_op TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE bronvideo(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bestand         TEXT NOT NULL UNIQUE,        -- relatief pad ('opnames/…'), forward slashes
    naam            TEXT NOT NULL,
    bytes           INTEGER,                     -- alleen de sync-check, géén identiteit
    fps             REAL,
    totaal_frames   INTEGER,
    status          TEXT NOT NULL DEFAULT 'nog doen',
    notitie         TEXT NOT NULL DEFAULT '',
    bijgewerkt_door TEXT NOT NULL DEFAULT '',    -- wie de status/notitie het laatst zette
    toegevoegd_op   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE bron_markering(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bron_id         INTEGER NOT NULL REFERENCES bronvideo(id) ON DELETE CASCADE,
    frame           INTEGER NOT NULL,            -- framenummer in de opname
    label           TEXT NOT NULL DEFAULT '',
    aangemaakt_door TEXT NOT NULL DEFAULT '',    -- wie het punt zette (gedeelde bibliotheek)
    aangemaakt_op   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE analyse(
    id                TEXT PRIMARY KEY,          -- UUID, tevens mapnaam onder media/
    schaatser_id      INTEGER NOT NULL REFERENCES schaatser(id) ON DELETE CASCADE,
    titel             TEXT NOT NULL,
    datum             TEXT NOT NULL,             -- ISO (YYYY-MM-DD)
    video_bestand     TEXT NOT NULL,             -- relatief pad, forward slashes
    w                 INTEGER,
    h                 INTEGER,
    fps               REAL,
    totaal_frames     INTEGER,
    backend           TEXT NOT NULL,             -- 'yolo' | 'mediapipe'
    instellingen_json TEXT NOT NULL DEFAULT '{}',
    aangemaakt_door   TEXT NOT NULL DEFAULT '',  -- trainersnaam (fase 4)
    bewerkt           INTEGER NOT NULL DEFAULT 0,-- handmatige skelet-edits (fase 3)
    video_bytes       INTEGER,                   -- grootte van de gekopieerde video (fase 4, cloud-sync-check)
    bron_id           INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL,
    bron_start_frame  INTEGER,                   -- uit welk stuk van de opname deze clip komt
    bron_eind_frame   INTEGER,                   -- (alle drie NULL bij een losse clip)
    aangemaakt_op     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE afzet_event_cache(
    analyse_id  TEXT NOT NULL REFERENCES analyse(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    been        TEXT,
    start_frame INTEGER,
    eind_frame  INTEGER,
    hoek        REAL,
    min_hoek    REAL,
    max_hoek    REAL,
    opmerking   TEXT,
    PRIMARY KEY (analyse_id, idx)
);
"""


class BibliotheekTeNieuw(RuntimeError):
    """De bibliotheek is met een nieuwere versie van de app gemaakt (user_version >
    SCHEMA_VERSIE). We raken hem dan niet aan: het schema kan kolommen/tabellen hebben
    die deze versie niet kent, en zou de versie omlaag zetten (zie open_db)."""


def open_db(bieb):
    """Maakt de bibliotheekmap + database + schema aan als die nog niet bestaan, en
    migreert een oudere database naar het huidige schema. Idempotent; aanroepen bij het
    opstarten en na het wisselen van bibliotheekpad.

    Een **nieuwere** database (gedeelde cloudmap, collega met een recentere app) wordt
    geweigerd met `BibliotheekTeNieuw` in plaats van stilzwijgend te worden 'gedowngrade':
    `PRAGMA user_version` omlaag zetten zou de nieuwere app bij het volgende openen z'n
    eigen migratie opnieuw laten draaien (`duplicate column name`) en de bibliotheek
    onbruikbaar maken. `user_version` wordt daarom alleen geschreven ná een geslaagde
    aanmaak of migratie."""
    os.makedirs(os.path.join(bieb, MEDIA_MAP), exist_ok=True)
    os.makedirs(os.path.join(bieb, OPNAMES_MAP), exist_ok=True)   # werklijst-map (fase 8)
    with _verbind(bieb) as con:
        versie = con.execute("PRAGMA user_version").fetchone()[0]
        if versie > SCHEMA_VERSIE:
            raise BibliotheekTeNieuw(
                f"Deze bibliotheek is gemaakt met een nieuwere versie van de app "
                f"(schema v{versie}; deze app kent v{SCHEMA_VERSIE}). "
                f"Werk de app bij om hem te kunnen openen.")
        if versie == 0:
            con.executescript(_SCHEMA)
        elif versie < SCHEMA_VERSIE:
            _migreer(con, versie)
        else:
            return                        # al bij; niets te schrijven
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSIE}")


def _migreer(con, van):
    """Werkt een bestaande database stapsgewijs bij naar SCHEMA_VERSIE. Cloud-veilig:
    ALTER TABLE ADD COLUMN is een kleine, in-place wijziging die één DB-bestand houdt."""
    if van < 2:
        # v1 → v2 (fase 4): kolom voor de video-grootte; oude rijen krijgen NULL en
        # slaan de sync-groottecheck bij het openen dus over (alleen bestaanscheck).
        con.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")
    if van < 3:
        # v2 → v3 (fase 8): de nog niet geknipte opnames als werklijst, plus de
        # herkomst van een analyse (welk stuk van welke opname). Oude analyses houden
        # bron_id NULL — dat klopt ook: die kwamen van een losse clip, niet uit een
        # opname. Nergens een migratie die iets moet raden.
        con.execute(_tabel_ddl("bronvideo"))
        # De REFERENCES-clausule mag mee in ADD COLUMN zolang de default NULL is (SQLite),
        # zodat een gemigreerde bibliotheek exact hetzelfde schema krijgt als een verse.
        for kolom in ("bron_id INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL",
                      "bron_start_frame INTEGER", "bron_eind_frame INTEGER"):
            con.execute(f"ALTER TABLE analyse ADD COLUMN {kolom}")
    if van < 4:
        # v3 → v4: de punten die de trainer in het handmatige kijkvenster zet. Een losse
        # tabel en geen kolom op bronvideo: het zijn er meerdere per opname, en ze horen
        # bij de opname (niet bij een analyse), dus ze verdwijnen mee als die rij ooit weg
        # zou vallen. Oude bibliotheken krijgen simpelweg een lege tabel.
        con.execute(_tabel_ddl("bron_markering"))


def _tabel_ddl(naam):
    """De CREATE TABLE van `naam` uit _SCHEMA, als IF NOT EXISTS — zo staat het schema op
    één plek en gebruikt de migratie gegarandeerd dezelfde definitie als een verse
    bibliotheek."""
    kop = f"CREATE TABLE {naam}("
    begin = _SCHEMA.index(kop)
    eind = _SCHEMA.index(");", begin) + 2
    return _SCHEMA[begin:eind].replace(kop, f"CREATE TABLE IF NOT EXISTS {naam}(")


def _abs_pad(bieb, rel):
    """Relatief DB-pad (forward slashes) → absoluut pad op dit OS."""
    return os.path.join(bieb, *rel.split("/"))


def _normaliseer_backend(naam):
    return "yolo" if (naam or "").lower().startswith("yolo") else "mediapipe"


def _meta_uit_rij(rij):
    """DB-rij → dict met de geparste `instellingen` erbij (corrupte JSON → {}).
    Gedeeld door lijst_analyses, laad_analyse en analyse_meta."""
    meta = dict(rij)
    try:
        meta["instellingen"] = json.loads(meta.get("instellingen_json") or "{}")
    except ValueError:
        meta["instellingen"] = {}
    return meta


# ── Schaatsers ─────────────────────────────────────────────────────────────────

def maak_schaatser(bieb, naam, geboortejaar=None, notities=""):
    with _verbind(bieb) as con:
        cur = con.execute(
            "INSERT INTO schaatser(naam, geboortejaar, notities) VALUES (?, ?, ?)",
            (naam, geboortejaar, notities))
        return cur.lastrowid


def wijzig_schaatser(bieb, schaatser_id, naam, geboortejaar=None, notities=""):
    with _verbind(bieb) as con:
        con.execute(
            "UPDATE schaatser SET naam = ?, geboortejaar = ?, notities = ? WHERE id = ?",
            (naam, geboortejaar, notities, schaatser_id))


def lijst_schaatsers(bieb):
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM analyse a WHERE a.schaatser_id = s.id)"
            "       AS aantal_analyses "
            "FROM schaatser s ORDER BY s.naam COLLATE NOCASE").fetchall()
        return [dict(r) for r in rijen]


def verwijder_schaatser(bieb, schaatser_id):
    """Verwijdert de schaatser én al zijn analyses (DB-cascade + mediamappen)."""
    with _verbind(bieb) as con:
        analyse_ids = [r["id"] for r in con.execute(
            "SELECT id FROM analyse WHERE schaatser_id = ?", (schaatser_id,))]
        con.execute("DELETE FROM schaatser WHERE id = ?", (schaatser_id,))
    # Mediamappen pas ná de geslaagde DB-verwijdering; een achterblijvende map
    # zonder DB-rij is onschadelijk (andersom niet).
    for aid in analyse_ids:
        shutil.rmtree(os.path.join(bieb, MEDIA_MAP, aid), ignore_errors=True)


# ── Analyses ───────────────────────────────────────────────────────────────────

def lijst_analyses(bieb, schaatser_id):
    """Lijstweergave uit de events-cache (geen npz nodig): titel, datum, videoduur
    (`totaal_frames`/`fps`), aantal afzetten, gemiddelde hoek.

    De gemiddelde hoek slaat **onvolledige** afzetten over — de push liep nog toen de video
    ophield, of er is binnen de stand-run geen zijwaartse push waargenomen; beide geven een
    veel te steile hoek die het gemiddelde omhoog trekt. Ze tellen wél mee in het aantal,
    want ze zijn gebeurd.

    `instellingen_json` gaat mee (geparst als `instellingen`) zodat de lijstweergave de
    appversie kan tonen zonder een analyse te openen — dat kost één json.loads per rij,
    verwaarloosbaar naast de subselects hieronder."""
    niet_like = " AND ".join(["c.opmerking NOT LIKE ?"] * len(ONVOLLEDIG_MARKERS))
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT a.id, a.titel, a.datum, a.backend, a.bewerkt, a.aangemaakt_op,"
            "       a.aangemaakt_door, a.instellingen_json, a.totaal_frames, a.fps,"
            "       (SELECT COUNT(*)  FROM afzet_event_cache c WHERE c.analyse_id = a.id)"
            "       AS aantal_afzetten,"
            "       (SELECT AVG(hoek) FROM afzet_event_cache c WHERE c.analyse_id = a.id"
            f"         AND (c.opmerking IS NULL OR ({niet_like}))) AS gem_hoek "
            "FROM analyse a WHERE a.schaatser_id = ? "
            "ORDER BY a.datum DESC, a.aangemaakt_op DESC",
            tuple(f"%{m}%" for m in ONVOLLEDIG_MARKERS) + (schaatser_id,)).fetchall()
        return [_meta_uit_rij(r) for r in rijen]


def lijst_kalibraties(bieb, beeld_w=None, beeld_h=None):
    """Analyses die een bewaarde perspectiefkalibratie dragen (fase 7), nieuwste eerst.

    Bedoeld om een kalibratie te hérgebruiken: een kalibratie hoort bij één camerastand,
    niet bij één clip, dus alle fragmenten uit dezelfde opname mogen hem delen. Dat is
    ook een meetkundige voorwaarde om analyses onderling te kunnen vergelijken — zeven
    keer met de hand dezelfde lijnen natrekken geeft zeven nét andere kalibraties, en
    dan meet je die spreiding in plaats van het effect van de correctie.

    Met `beeld_w`/`beeld_h` worden alleen kalibraties van diezelfde beeldmaat
    teruggegeven: de lijnen staan in pixels, dus op een andersgrote video liggen ze
    ernaast. Retourneert dicts met id/titel/schaatser/datum/`perspectief` (de ruwe dict)
    en `notitie`.

    Er is bewust géén aparte kalibratietabel: de kalibratie zit in `instellingen_json`,
    dus dit kost één json.loads per analyse en geen schemabump."""
    uit = []
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT a.id, a.titel, a.datum, a.aangemaakt_op, a.w, a.h,"
            "       a.instellingen_json, s.naam AS schaatser "
            "FROM analyse a LEFT JOIN schaatser s ON s.id = a.schaatser_id "
            "ORDER BY a.datum DESC, a.aangemaakt_op DESC").fetchall()
    for r in rijen:
        try:
            inst = json.loads(r["instellingen_json"] or "{}")
        except (ValueError, TypeError):
            continue
        p = inst.get("perspectief")
        if not p or not p.get("invoer"):
            continue
        inv = p["invoer"]
        if beeld_w is not None and int(inv.get("beeld_w", -1)) != int(beeld_w):
            continue
        if beeld_h is not None and int(inv.get("beeld_h", -1)) != int(beeld_h):
            continue
        uit.append({"id": r["id"], "titel": r["titel"], "datum": r["datum"],
                    "schaatser": r["schaatser"], "w": inv.get("beeld_w"),
                    "h": inv.get("beeld_h"), "perspectief": p,
                    "notitie": inv.get("notitie", "")})
    return uit


def sla_analyse_op(bieb, schaatser_id, titel, video_pad, info, resultaten, events,
                   backend, instellingen, datum=None, aangemaakt_door="",
                   bron_id=None, bron_start_frame=None, bron_eind_frame=None):
    """
    Slaat een afgeronde analyse op in de bibliotheek: kopieert de video, schrijft de
    landmarks als .npz en insert de DB-rij + events-cache in één transactie.
    De DB-insert is bewust de láátste stap: gaat er iets mis (video onleesbaar, schijf
    vol), dan wordt de mediamap opgeruimd en staat er nooit een halve analyse in de
    bibliotheek. Retourneert het analyse-id (UUID).
    Draait in de praktijk in de workerthread — de videokopie kan lang duren.
    `aangemaakt_door` (fase 4) is de trainersnaam; `video_bytes` (de grootte van de
    gekopieerde video) wordt bewaard zodat een collega die de analyse opent terwijl de
    cloudsync nog loopt een halve download kan herkennen.

    De **appversie** en de volledige `backend_naam` worden hier aan `instellingen`
    toegevoegd — één plek, dus elke opslagroute (enkele analyse, batch, zelftest) legt
    het vast zonder eraan te hoeven denken. Het gaat mee in `instellingen_json`, dus
    zonder schemabump. `setdefault`: een caller die het zelf al invult wint.

    `bron_id` + `bron_start_frame`/`bron_eind_frame` (fase 8) leggen vast uit welk stuk van
    welke opname deze clip geknipt is; bij een losse video blijven ze NULL. Daarmee is "welke
    stukken van deze opname zijn al gedaan" één query (`bron_fragmenten`).
    """
    inst = dict(instellingen or {})
    inst.setdefault("backend_naam", backend or "")   # de kolom `backend` is genormaliseerd
    versie = app_versie()
    inst.setdefault("app_versie", versie["label"])
    inst.setdefault("app_commit", versie["commit"])

    analyse_id = str(uuid.uuid4())
    doelmap = os.path.join(bieb, MEDIA_MAP, analyse_id)
    videonaam = os.path.basename(video_pad)
    try:
        os.makedirs(doelmap)
        videokopie = os.path.join(doelmap, videonaam)
        shutil.copy2(video_pad, videokopie)
        sla_landmarks_op(os.path.join(doelmap, NPZ_NAAM), resultaten, info)
        video_bytes = os.path.getsize(videokopie)

        rel_video = f"{MEDIA_MAP}/{analyse_id}/{videonaam}"
        with _verbind(bieb) as con:
            con.execute(
                "INSERT INTO analyse(id, schaatser_id, titel, datum, video_bestand,"
                "                    w, h, fps, totaal_frames, backend, instellingen_json,"
                "                    aangemaakt_door, video_bytes,"
                "                    bron_id, bron_start_frame, bron_eind_frame)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (analyse_id, schaatser_id, titel,
                 datum or date.today().isoformat(), rel_video,
                 info.w, info.h, info.fps, info.totaal,
                 _normaliseer_backend(backend),
                 json.dumps(inst, ensure_ascii=False),
                 aangemaakt_door or "", video_bytes,
                 bron_id, bron_start_frame, bron_eind_frame))
            _schrijf_events_cache(con, analyse_id, events)
        return analyse_id
    except Exception:
        shutil.rmtree(doelmap, ignore_errors=True)
        raise


def _schrijf_events_cache(con, analyse_id, events):
    """Vervangt de events-cache van één analyse (DELETE + INSERT) binnen een lopende
    transactie. Gedeeld door sla_analyse_op, bewaar_bewerkte_landmarks en
    ververs_events_cache — de cache is puur voor snelle lijstweergave."""
    con.execute("DELETE FROM afzet_event_cache WHERE analyse_id = ?", (analyse_id,))
    con.executemany(
        "INSERT INTO afzet_event_cache(analyse_id, idx, been, start_frame,"
        "                              eind_frame, hoek, min_hoek, max_hoek, opmerking)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(analyse_id, i, ev.been, ev.start_frame, ev.eind_frame,
          ev.hoek, ev.min_hoek, ev.max_hoek, _cache_opmerking(ev))
         for i, ev in enumerate(events)])


def _cache_opmerking(ev):
    """Opmerkingstekst voor de cache. De onvolledig-vlag (`AfzetEvent.onvolledig`: de push
    liep nog toen de video ophield, óf er is geen zijwaartse push waargenomen — beide geven
    een te steile hoek) krijgt geen eigen kolom — dat zou een schemamigratie kosten voor
    iets wat bij het openen tóch vers herberekend wordt — maar reist als tekst mee, zodat
    `lijst_analyses` zo'n afzet buiten de gemiddelde hoek kan houden
    (`ONVOLLEDIG_MARKERS`)."""
    reden = getattr(ev, 'onvolledig', None)
    if reden:
        return f"{reden} · {ev.opmerking}" if ev.opmerking else reden
    return ev.opmerking


def analyse_meta(bieb, analyse_id):
    """Alleen de DB-rij van één analyse (incl. geparste `instellingen`), zónder het npz
    te lezen — genoeg voor een info-overzicht (appversie, backend, instellingen).
    `bron_naam` komt er los bij: de herkomst tonen ("uit opname X, 12:30–13:05") vraagt
    de bestandsnaam, en die staat in de bronvideo-tabel."""
    with _verbind(bieb) as con:
        rij = con.execute(
            "SELECT a.*, b.naam AS bron_naam FROM analyse a "
            "LEFT JOIN bronvideo b ON b.id = a.bron_id WHERE a.id = ?",
            (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analyse {analyse_id} staat niet in de bibliotheek.")
    return _meta_uit_rij(rij)


def laad_analyse(bieb, analyse_id):
    """Laadt een analyse: DB-meta (incl. geparste instellingen) + landmarks uit het
    .npz. Retourneert {'meta', 'info', 'resultaten', 'video_pad'}; de afgeleiden zijn
    nog leeg — draai verwerk_afgeleiden() + segmenteer_afzetten() (fase 0-naad)."""
    meta = analyse_meta(bieb, analyse_id)
    # npz-lezen buiten de DB-verbinding houden (verbinding zo kort mogelijk).
    info, resultaten = laad_landmarks(os.path.join(bieb, MEDIA_MAP, analyse_id, NPZ_NAAM))
    return {"meta": meta, "info": info, "resultaten": resultaten,
            "video_pad": _abs_pad(bieb, meta["video_bestand"])}


def analyse_video_pad(bieb, analyse_id):
    """Absoluut pad van de gekopieerde video van deze analyse."""
    with _verbind(bieb) as con:
        rij = con.execute("SELECT video_bestand FROM analyse WHERE id = ?",
                          (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analyse {analyse_id} staat niet in de bibliotheek.")
    return _abs_pad(bieb, rij["video_bestand"])


# ── Opnames: de nog niet geknipte bronvideo's (fase 8) ─────────────────────────
#
# De ontwerpregel: **de map is de waarheid over wélke bestanden er zijn, de database over
# wat wij ervan weten.** De bestandslijst wordt bij het openen van schijf gescand, niet uit
# de DB gelezen — anders loopt de DB scheef zodra iemand een bestand hernoemt of weggooit en
# zit je aan opruimwerk vast. In de DB staat alleen wat je nooit van schijf kunt aflezen:
# status, notitie, en welke analyses uit welk stuk van welke opname komen.

def opnames_pad(bieb, maak_aan=True):
    """Absoluut pad van `<bibliotheek>/opnames/`. Binnen de bibliotheekmap, zodat alle paden
    relatief blijven (fase 1-discipline) en er géén extra pad-instelling per trainer nodig
    is; de map wordt aangemaakt als hij er nog niet staat."""
    pad = os.path.join(bieb, OPNAMES_MAP)
    if maak_aan:
        os.makedirs(pad, exist_ok=True)
    return pad


def synchroniseer_bronmap(bieb, meta_lezer=None):
    """Scant `opnames/` en zet nieuwe bestanden in de bronvideo-tabel. Retourneert het
    aantal toegevoegde opnames.

    - **Identiteit is het relatieve pad** (`opnames/<naam>`, UNIQUE): één map kan geen twee
      bestanden met dezelfde naam bevatten, en omdat de map ín de bibliotheek zit is dat pad
      bij elke trainer hetzelfde. Een hernoemd bestand geldt als nieuw; de oude rij blijft
      met zijn analyses bestaan en wordt getoond als "bestand niet gevonden".
    - **`bytes` hoort níet in de sleutel**, alleen in de sync-check: een opname die bij een
      collega nog binnenkomt is op dat moment *kleiner* dan wat er in de DB staat. Zat de
      grootte in de sleutel, dan zag de scan een half gedownload bestand aan voor een nieuwe
      opname en kwam er een tweede rij bij — precies wanneer je de fragmentgeschiedenis nodig
      hebt.
    - `INSERT OR IGNORE`, zodat twee trainers die tegelijk dezelfde nieuwe opname zien niet
      botsen, en er wordt **alleen geschreven als er echt iets nieuws is**: anders zou elke
      app-start van elke trainer de gedeelde DB aanraken, terwijl die volgens de fase
      4-discipline zo veel mogelijk in rust hoort te zijn voor de syncer.

    `meta_lezer(pad)` levert `(fps, totaal_frames)`; standaard via OpenCV. Een onleesbaar
    bestand (nog aan het downloaden) komt gewoon in de lijst met lege meta — dat is beter dan
    het overslaan, want dan zie je niet dát er een opname is.
    """
    map_pad = opnames_pad(bieb)
    try:
        namen = sorted(n for n in os.listdir(map_pad)
                       if os.path.splitext(n)[1].lower() in VIDEO_EXTS)
    except OSError:
        return 0
    with _verbind(bieb) as con:
        bekend = {r["bestand"] for r in con.execute("SELECT bestand FROM bronvideo")}
        nieuw = [n for n in namen if f"{OPNAMES_MAP}/{n}" not in bekend]
        for naam in nieuw:
            pad = os.path.join(map_pad, naam)
            fps, totaal = (meta_lezer or _video_meta)(pad)
            try:
                grootte = os.path.getsize(pad)
            except OSError:
                grootte = None
            con.execute(
                "INSERT OR IGNORE INTO bronvideo(bestand, naam, bytes, fps, totaal_frames,"
                "                                status) VALUES (?, ?, ?, ?, ?, ?)",
                (f"{OPNAMES_MAP}/{naam}", naam, grootte, fps, totaal, BRON_STATUS_DEFAULT))
    return len(nieuw)


def _video_meta(pad):
    """(fps, totaal_frames) van een videobestand; (None, None) als het niet te lezen is."""
    try:
        info = video_info(pad)
        return info.fps, info.totaal
    except Exception:
        return None, None


def lijst_bronvideos(bieb):
    """Alle bekende opnames met hun werklijst-gegevens: status, notitie, hoeveel fragmenten
    er al uit geknipt zijn (= analyses met deze bron), of het bestand er staat en of de
    cloudsync nog bezig is.

    `aantal_fragmenten` telt de analyses die uit deze opname komen; `aantal_schaatsers`
    hoeveel verschillende schaatsers dat betreft; `aantal_punten` de bewaarde punten uit het
    handmatige kijkvenster. Het onderscheid "fragmenten vs. analyses"
    uit de roadmap valt hier samen — elk gemarkeerd fragment wordt precies één analyse."""
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT b.*,"
            "       (SELECT COUNT(*) FROM analyse a WHERE a.bron_id = b.id)"
            "       AS aantal_fragmenten,"
            "       (SELECT COUNT(DISTINCT a.schaatser_id) FROM analyse a"
            "         WHERE a.bron_id = b.id) AS aantal_schaatsers,"
            "       (SELECT COUNT(*) FROM bron_markering m WHERE m.bron_id = b.id)"
            "       AS aantal_punten "
            "FROM bronvideo b ORDER BY b.naam COLLATE NOCASE").fetchall()
    uit = []
    for r in rijen:
        d = dict(r)
        d["pad"] = _abs_pad(bieb, d["bestand"])
        d["sync"] = video_sync_status(d["pad"], d["bytes"])
        uit.append(d)
    return uit


def bronvideo(bieb, bron_id):
    """Eén opname-rij (incl. absoluut pad + sync-status), of KeyError."""
    for b in lijst_bronvideos(bieb):
        if b["id"] == bron_id:
            return b
    raise KeyError(f"Opname {bron_id} staat niet in de bibliotheek.")


def wijzig_bronvideo(bieb, bron_id, status=None, notitie=None, bijgewerkt_door=""):
    """Zet status en/of notitie van een opname. `bijgewerkt_door` (de trainersnaam) gaat mee
    zodat in een gedeelde bibliotheek zichtbaar is wie een opname op 'klaar' zette —
    hetzelfde motief als `aangemaakt_door` bij een analyse."""
    velden, waarden = [], []
    if status is not None:
        velden.append("status = ?")
        waarden.append(status)
    if notitie is not None:
        velden.append("notitie = ?")
        waarden.append(notitie)
    if not velden:
        return
    velden.append("bijgewerkt_door = ?")
    waarden.append(bijgewerkt_door or "")
    with _verbind(bieb) as con:
        con.execute(f"UPDATE bronvideo SET {', '.join(velden)} WHERE id = ?",
                    waarden + [bron_id])


def bron_fragmenten(bieb, bron_id):
    """De stukken van deze opname die al geanalyseerd zijn: [{analyse_id, titel, schaatser,
    start_frame, eind_frame}], op startframe gesorteerd. Dit levert de grijze blokken in het
    knipvenster — één query in plaats van een scan door alle instellingen_json-velden."""
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT a.id AS analyse_id, a.titel, s.naam AS schaatser,"
            "       a.bron_start_frame AS start_frame, a.bron_eind_frame AS eind_frame "
            "FROM analyse a LEFT JOIN schaatser s ON s.id = a.schaatser_id "
            "WHERE a.bron_id = ? AND a.bron_start_frame IS NOT NULL "
            "ORDER BY a.bron_start_frame", (bron_id,)).fetchall()
    return [dict(r) for r in rijen]


# ── Punten in een opname (handmatig kijkvenster) ───────────────────────────────
# Een trainer die een opname doorkijkt wil een plek kunnen terugvinden — de sprong die
# hij nog eens wil zien, het moment waarop de serie begint. Dat zijn losse framenummers
# met een naampje, en ze horen bij de **opname**: ze overleven het afsluiten van het
# venster en staan (net als status en notitie) voor het hele team in de gedeelde
# database. Bewust géén analyse: er wordt niets gemeten, alleen onthouden waar je was.

def lijst_markeringen(bieb, bron_id):
    """De bewaarde punten van een opname, op framenummer gesorteerd."""
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT * FROM bron_markering WHERE bron_id = ? ORDER BY frame, id",
            (bron_id,)).fetchall()
    return [dict(r) for r in rijen]


def voeg_markering_toe(bieb, bron_id, frame, label="", aangemaakt_door=""):
    """Zet een punt op `frame` in deze opname; retourneert het nieuwe id."""
    with _verbind(bieb) as con:
        cur = con.execute(
            "INSERT INTO bron_markering(bron_id, frame, label, aangemaakt_door) "
            "VALUES (?, ?, ?, ?)",
            (bron_id, int(frame), label or "", aangemaakt_door or ""))
        return cur.lastrowid


def wijzig_markering(bieb, markering_id, label=None, frame=None):
    """Hernoemt een punt en/of verplaatst het naar een ander frame."""
    velden, waarden = [], []
    if label is not None:
        velden.append("label = ?")
        waarden.append(label)
    if frame is not None:
        velden.append("frame = ?")
        waarden.append(int(frame))
    if not velden:
        return
    with _verbind(bieb) as con:
        con.execute(f"UPDATE bron_markering SET {', '.join(velden)} WHERE id = ?",
                    waarden + [markering_id])


def verwijder_markering(bieb, markering_id):
    with _verbind(bieb) as con:
        con.execute("DELETE FROM bron_markering WHERE id = ?", (markering_id,))


# ── Gedeelde cloudmap (fase 4) ──────────────────────────────────────────────────

def detecteer_conflictkopieen(bieb):
    """Zoekt naar conflictkopieën van de database die een cloudsyncer (Google Drive,
    OneDrive, Dropbox) kan achterlaten wanneer twee trainers bijna tegelijk schrijven —
    bv. 'schaats-DESKTOP.db', 'schaats (1).db' of 'schaats (conflicted copy).db'.
    Retourneert de bestandsnamen (zonder pad), gesorteerd; leeg als alles in orde is.

    Bewust detectie, geen preventie: de app kan zulke kopieën niet veilig samenvoegen,
    maar waarschuwt zodat de trainer ze handmatig kan opruimen. 'schaats.db' zelf en de
    kortstondige '-journal' (DELETE-mode) worden overgeslagen.

    Alleen namen die op onze eigen database lijken tellen mee ('schaats….db'): een
    syncer hangt zijn markering áchter de bestandsnaam. Een willekeurige andere
    database die iemand in de map zet is geen conflictkopie, en zou anders bij elke
    keer openen én elke 'Vernieuwen' opnieuw een waarschuwing opleveren."""
    try:
        namen = os.listdir(bieb)
    except OSError:
        return []
    hoofd = DB_NAAM.lower()
    stam  = os.path.splitext(hoofd)[0]
    kopieen = [n for n in namen
               if n.lower().endswith(".db") and n.lower() != hoofd
               and n.lower().startswith(stam)]
    return sorted(kopieen)


def video_sync_status(video_pad, verwacht_bytes):
    """Sync-status van een gekopieerde video in een gedeelde cloudmap (fase 4):
    - 'ontbreekt'  : het bestand staat (nog) niet op schijf;
    - 'onvolledig' : het bestand is kleiner dan bij het opslaan (cloud downloadt nog);
    - None         : in orde, of de verwachte grootte is onbekend (analyse van vóór v2)."""
    if not os.path.isfile(video_pad):
        return "ontbreekt"
    try:
        if verwacht_bytes and os.path.getsize(video_pad) < verwacht_bytes:
            return "onvolledig"
    except OSError:
        return "ontbreekt"
    return None


# Snelheidsproef (zie `bestand_lokaal`): grootte van één leesblokje en de tijd waarboven
# we het bestand als "niet op deze pc" beschouwen. Gemeten 24 aug 2026 op deze bibliotheek:
# van een lokale schijf kost zo'n blokje 0,3-2 ms, uit een Google Drive streaming-map 614 ms.
# De drempel ligt daar ruim tussenin, zodat ook een trage USB- of netwerkschijf nog "lokaal"
# heet — het gaat om de factor 100, niet om de precieze grens.
LOKAAL_BLOK_BYTES = 64 * 1024
LOKAAL_DREMPEL_MS = 100.0


def bestand_lokaal(pad, monsters=3, drempel_ms=LOKAAL_DREMPEL_MS):
    """Staat dit videobestand écht op deze pc, of wordt het per stukje uit de cloud gehaald?

    - 'lokaal' : elk monster kwam meteen binnen;
    - 'deels'  : een deel wel, een deel niet (cloudmap is nog aan het downloaden);
    - 'cloud'  : geen enkel monster kwam van schijf;
    - None     : het bestand is er niet, of is niet te lezen.

    **Meten, niet vragen.** Windows kent wel een attribuut voor cloud-placeholders
    (FILE_ATTRIBUTE_OFFLINE / RECALL_ON_DATA_ACCESS), maar Google Drive voor desktop zet dat
    niet: zijn schijf meldt zich als een gewone vaste schijf en een niet-gedownloade opname
    van 4 GB heeft attribuut `Normal`. Wat je wél kunt zien is de tijd — een blokje van een
    plek die nog niet lokaal staat kost een netwerkronde.

    Waarom dit ertoe doet: op een streaming-map moet elke sprong in het knipvenster eerst
    tientallen MB ophalen (gemeten: 5-20 s per sprong, ~40 MB), terwijl diezelfde sprong op
    een lokale kopie 30-120 ms kost. Zie `schaats_gui._opname_beschikbaar`.

    De monsters liggen op **willekeurige** plekken: een gelezen stuk zit daarna in de
    cloudcache, dus steeds dezelfde plek proeven zou de tweede keer altijd 'lokaal' zeggen.
    Kosten: verwaarloosbaar als het bestand lokaal staat, en anders een paar seconden —
    daarom draait dit in de GUI op een achtergrondthread.
    """
    try:
        grootte = os.path.getsize(pad)
    except OSError:
        return None
    if grootte <= 0:
        return None

    traag = gemeten = 0
    try:
        with open(pad, "rb") as f:
            for _ in range(max(1, monsters)):
                speling = grootte - LOKAAL_BLOK_BYTES
                off = random.randrange(speling) if speling > 0 else 0
                begin = time.perf_counter()
                f.seek(off)
                if not f.read(LOKAAL_BLOK_BYTES):
                    continue
                gemeten += 1
                traag += (time.perf_counter() - begin) * 1000.0 > drempel_ms
    except OSError:
        return None
    if not gemeten:
        return None
    if traag == 0:
        return "lokaal"
    return "cloud" if traag == gemeten else "deels"


# ── Skelet-editor (fase 3) ──────────────────────────────────────────────────────

def bewaar_bewerkte_landmarks(bieb, analyse_id, resultaten, info, events):
    """
    Overschrijft de landmarks van een analyse met handmatig gecorrigeerde (skelet-
    editor). Bij de éérste edit wordt de originele landmarks.npz eenmalig veiliggesteld
    als landmarks_ruw.npz, zodat 'herstel origineel' altijd terug kan. Zet analyse.bewerkt
    = 1 en ververst de events-cache in één transactie. Draait per drop, dus houd het licht.
    """
    doelmap = os.path.join(bieb, MEDIA_MAP, analyse_id)
    npz = os.path.join(doelmap, NPZ_NAAM)
    ruw = os.path.join(doelmap, NPZ_RUW_NAAM)
    if not os.path.isfile(ruw) and os.path.isfile(npz):
        shutil.copy2(npz, ruw)          # pristine origineel, alleen bij de eerste edit
    sla_landmarks_op(npz, resultaten, info)
    with _verbind(bieb) as con:
        con.execute("UPDATE analyse SET bewerkt = 1 WHERE id = ?", (analyse_id,))
        _schrijf_events_cache(con, analyse_id, events)


def bewaar_bochtmarkering(bieb, analyse_id, resultaten, info, events):
    """
    Schrijft een alsnog bepaalde bochtmarkering (`FrameResultaat.bocht`) weg en ververst
    de events-cache. Voor analyses van vóór de bochtdetectie: hun npz heeft de vlag nog
    niet, terwijl de landmarks van de héle clip er wél in staan — daar valt de bocht dus
    prima uit af te leiden zonder opnieuw te analyseren.

    Bewust níet `bewaar_bewerkte_landmarks`: dat is voor handwerk met de skelet-editor en
    zet `analyse.bewerkt = 1` plus een pristine backup. Hier verandert geen enkel
    landmark — alleen de vlag die zegt welke frames buiten de meting vallen — dus die
    analyse blijft "niet bewerkt" en de eval-metrics blijven vergelijkbaar met andere
    onbewerkte analyses (zie de waarschuwing in schaats_eval).
    """
    npz = os.path.join(bieb, MEDIA_MAP, analyse_id, NPZ_NAAM)
    sla_landmarks_op(npz, resultaten, info)
    with _verbind(bieb) as con:
        _schrijf_events_cache(con, analyse_id, events)


def herstel_originele_landmarks(bieb, analyse_id):
    """Zet de landmarks terug naar vóór de eerste edit (landmarks_ruw.npz → landmarks.npz)
    en analyse.bewerkt = 0. Retourneert True als er een origineel was, anders False (de
    analyse is nooit bewerkt). De GUI herlaadt daarna en ververst zelf de events-cache."""
    doelmap = os.path.join(bieb, MEDIA_MAP, analyse_id)
    ruw = os.path.join(doelmap, NPZ_RUW_NAAM)
    if not os.path.isfile(ruw):
        return False
    shutil.copy2(ruw, os.path.join(doelmap, NPZ_NAAM))
    with _verbind(bieb) as con:
        con.execute("UPDATE analyse SET bewerkt = 0 WHERE id = ?", (analyse_id,))
    return True


def ververs_events_cache(bieb, analyse_id, events):
    """Herschrijft alleen de events-cache (na een herberekening, bv. na herstel origineel)."""
    with _verbind(bieb) as con:
        _schrijf_events_cache(con, analyse_id, events)


def hernoem_analyse(bieb, analyse_id, titel):
    with _verbind(bieb) as con:
        con.execute("UPDATE analyse SET titel = ? WHERE id = ?", (titel, analyse_id))


def verwijder_analyse(bieb, analyse_id):
    """Verwijdert DB-rij (cascade wist de events-cache) + mediamap."""
    with _verbind(bieb) as con:
        con.execute("DELETE FROM analyse WHERE id = ?", (analyse_id,))
    shutil.rmtree(os.path.join(bieb, MEDIA_MAP, analyse_id), ignore_errors=True)


# ── Zelftest ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from schaats_analyse import arrays_naar_resultaten, resultaten_naar_arrays, AfzetEvent

    tmp = tempfile.mkdtemp(prefix="schaats_db_test_")
    try:
        bieb = os.path.join(tmp, "bieb")
        open_db(bieb)
        open_db(bieb)   # idempotent
        with _verbind(bieb) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSIE

        # Migratie v1 → v4: een oude DB zonder video_bytes (v2), zonder bronvideo/bron_*
        # (v3) en zonder bron_markering (v4) wordt in één keer bijgewerkt zonder dataverlies. Het oude schema staat hier
        # bewust letterlijk: v1 ís bevroren, en het uit _SCHEMA weg filteren wordt met elke
        # bump fragieler.
        v1_schema = """
        CREATE TABLE schaatser(
            id INTEGER PRIMARY KEY AUTOINCREMENT, naam TEXT NOT NULL,
            geboortejaar INTEGER, notities TEXT NOT NULL DEFAULT '',
            aangemaakt_op TEXT NOT NULL DEFAULT (datetime('now', 'localtime')));
        CREATE TABLE analyse(
            id TEXT PRIMARY KEY,
            schaatser_id INTEGER NOT NULL REFERENCES schaatser(id) ON DELETE CASCADE,
            titel TEXT NOT NULL, datum TEXT NOT NULL, video_bestand TEXT NOT NULL,
            w INTEGER, h INTEGER, fps REAL, totaal_frames INTEGER,
            backend TEXT NOT NULL, instellingen_json TEXT NOT NULL DEFAULT '{}',
            aangemaakt_door TEXT NOT NULL DEFAULT '', bewerkt INTEGER NOT NULL DEFAULT 0,
            aangemaakt_op TEXT NOT NULL DEFAULT (datetime('now', 'localtime')));
        CREATE TABLE afzet_event_cache(
            analyse_id TEXT NOT NULL REFERENCES analyse(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL, been TEXT, start_frame INTEGER, eind_frame INTEGER,
            hoek REAL, min_hoek REAL, max_hoek REAL, opmerking TEXT,
            PRIMARY KEY (analyse_id, idx));
        """
        oud = os.path.join(tmp, "oud_v1")
        os.makedirs(os.path.join(oud, MEDIA_MAP))
        with sqlite3.connect(os.path.join(oud, DB_NAAM)) as c:
            c.executescript(v1_schema)
            c.execute("PRAGMA user_version = 1")
        open_db(oud)   # migreert
        with _verbind(oud) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSIE
            kolommen = [r[1] for r in c.execute("PRAGMA table_info(analyse)")]
            assert "video_bytes" in kolommen
            assert {"bron_id", "bron_start_frame", "bron_eind_frame"} <= set(kolommen)
            assert c.execute("SELECT COUNT(*) FROM bronvideo").fetchone()[0] == 0
            assert c.execute("SELECT COUNT(*) FROM bron_markering").fetchone()[0] == 0

        # Migratie v2 → v3 apart: alleen de fase 8-stap, zonder de v2-stap ervoor.
        oud2 = os.path.join(tmp, "oud_v2")
        os.makedirs(os.path.join(oud2, MEDIA_MAP))
        with sqlite3.connect(os.path.join(oud2, DB_NAAM)) as c:
            c.executescript(v1_schema)
            c.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")   # = v2
            c.execute("PRAGMA user_version = 2")
        open_db(oud2)
        with _verbind(oud2) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSIE
            assert "bron_eind_frame" in [r[1] for r in c.execute("PRAGMA table_info(analyse)")]

        # Migratie v3 → v4 apart: een bibliotheek die alleen de puntentabel nog mist, en
        # waarvan de bestaande opnamerijen ongemoeid moeten blijven.
        oud3 = os.path.join(tmp, "oud_v3")
        os.makedirs(os.path.join(oud3, MEDIA_MAP))
        with sqlite3.connect(os.path.join(oud3, DB_NAAM)) as c:
            c.executescript(v1_schema)
            c.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")   # = v2
            c.execute(_tabel_ddl("bronvideo"))                                # = v3
            for kolom in ("bron_id INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL",
                          "bron_start_frame INTEGER", "bron_eind_frame INTEGER"):
                c.execute(f"ALTER TABLE analyse ADD COLUMN {kolom}")
            c.execute("INSERT INTO bronvideo(bestand, naam) VALUES ('opnames/x.mp4', 'x.mp4')")
            c.execute("PRAGMA user_version = 3")
        open_db(oud3)
        with _verbind(oud3) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSIE
            assert c.execute("SELECT COUNT(*) FROM bron_markering").fetchone()[0] == 0
            assert c.execute("SELECT naam FROM bronvideo").fetchone()[0] == "x.mp4"

        # Nieuwere DB (collega met een recentere app): weigeren, niét downgraden.
        nieuw = os.path.join(tmp, "nieuw_v99")
        os.makedirs(os.path.join(nieuw, MEDIA_MAP))
        with sqlite3.connect(os.path.join(nieuw, DB_NAAM)) as c:
            c.executescript(_SCHEMA)
            c.execute("ALTER TABLE analyse ADD COLUMN iets_nieuws TEXT")
            c.execute(f"PRAGMA user_version = {SCHEMA_VERSIE + 1}")
        try:
            open_db(nieuw)
            raise AssertionError("nieuwere bibliotheek had geweigerd moeten worden")
        except BibliotheekTeNieuw:
            pass
        with _verbind(nieuw) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSIE + 1
            assert "iets_nieuws" in [r[1] for r in c.execute("PRAGMA table_info(analyse)")]

        # Env-var override voor het bibliotheekpad.
        os.environ[ENV_BIBLIOTHEEK] = bieb
        assert bibliotheek_pad() == bieb
        del os.environ[ENV_BIBLIOTHEEK]

        # Synthetische analyse (fase 0-serialisatievorm) + dummy-videobestand.
        n = 20
        rng = np.random.default_rng(1)
        arrays = {
            "landmarks":     rng.random((n, 33, 3)).astype(np.float32),
            "pose_gevonden": np.ones(n, dtype=bool),
            "horizon_deg":   np.zeros(n, dtype=np.float32),
            "w": np.int32(64), "h": np.int32(48),
            "fps": np.float32(25.0), "totaal": np.int32(n),
        }
        info, resultaten = arrays_naar_resultaten(arrays)
        events = [
            AfzetEvent(0, "links",  0,  5, 0.00, 0.20, 40.0, 38.0, 44.0),
            AfzetEvent(1, "rechts", 6, 12, 0.24, 0.48, 42.0, 40.0, 45.0,
                       opmerking="gemiste tegenafzet?"),
            # Onvolledig: tellen mee in het aantal, maar niet in de gemiddelde hoek —
            # anders trekken die veel te steile hoeken het gemiddelde op. Beide redenen,
            # want `lijst_analyses` moet ze allebei uitfilteren.
            AfzetEvent(2, "links",  13, 19, 0.52, 0.76, 79.0, 60.0, 80.0,
                       onvolledig=ONV_AFGEKAPT),
            AfzetEvent(3, "rechts", 20, 26, 0.80, 1.04, 83.0, 74.0, 88.0,
                       onvolledig=ONV_GEEN_PUSH),
        ]
        video = os.path.join(tmp, "Testvideo.mp4")
        with open(video, "wb") as f:
            f.write(b"geen echte video, copy2 kijkt niet naar de inhoud")

        sid = maak_schaatser(bieb, "Test Schaatser", 2010, "proefkonijn")
        wijzig_schaatser(bieb, sid, "Test Schaatser", 2011, "proefkonijn 2.0")
        s = lijst_schaatsers(bieb)
        assert len(s) == 1 and s[0]["geboortejaar"] == 2011 and s[0]["aantal_analyses"] == 0

        # Appversie: buiten een git-repo zijn de velden leeg — daarom alleen de vorm
        # controleren, niet de inhoud.
        versie = app_versie()
        assert set(versie) == {"commit", "datum", "vuil", "label"}
        assert bool(versie["label"]) == bool(versie["commit"])

        # `backend_naam` bewust weggelaten: sla_analyse_op hoort hem zelf in te vullen.
        instellingen = {"smooth_n": 5, "threshold": 0.015, "smooth_landmarks": True,
                        "bocht_overslaan": True,
                        "doel_punt": [0.5, 0.5], "horizon_deg": 0.0, "auto_horizon": False,
                        "heavy": False, "perspectief_gebruikt": False}
        aid = sla_analyse_op(bieb, sid, "Proefanalyse", video, info, resultaten, events,
                             backend="YOLO-pose + ByteTrack", instellingen=instellingen,
                             aangemaakt_door="Coach Tester")
        assert os.path.isfile(os.path.join(bieb, MEDIA_MAP, aid, "Testvideo.mp4"))
        assert os.path.isfile(os.path.join(bieb, MEDIA_MAP, aid, NPZ_NAAM))

        la = lijst_analyses(bieb, sid)
        assert len(la) == 1 and la[0]["aantal_afzetten"] == 4
        # gemiddelde over (40.0, 42.0); de onvolledige 79.0 en 83.0 vallen erbuiten
        assert abs(la[0]["gem_hoek"] - 41.0) < 1e-9 and la[0]["backend"] == "yolo"
        assert la[0]["aangemaakt_door"] == "Coach Tester"
        # De lijstweergave draagt de instellingen mee (voor de appversie-tooltip).
        assert la[0]["instellingen"]["app_versie"] == versie["label"]
        assert lijst_schaatsers(bieb)[0]["aantal_analyses"] == 1

        # analyse_meta = dezelfde meta zonder het npz te lezen.
        m = analyse_meta(bieb, aid)
        assert m["titel"] == "Proefanalyse" and m["instellingen"]["smooth_n"] == 5
        try:
            analyse_meta(bieb, "bestaat-niet")
            raise AssertionError("analyse_meta hoort KeyError te geven")
        except KeyError:
            pass

        data = laad_analyse(bieb, aid)
        assert data["meta"]["titel"] == "Proefanalyse"
        # Wat de caller meegaf staat er onveranderd in; appversie en de volledige
        # backendnaam heeft sla_analyse_op er zelf bij gezet.
        opgeslagen = data["meta"]["instellingen"]
        assert all(opgeslagen[k] == v for k, v in instellingen.items())
        assert opgeslagen["backend_naam"] == "YOLO-pose + ByteTrack"
        assert opgeslagen["app_versie"] == versie["label"]
        assert opgeslagen["app_commit"] == versie["commit"]
        assert data["meta"]["aangemaakt_door"] == "Coach Tester"
        assert data["meta"]["video_bytes"] == os.path.getsize(video)
        assert os.path.isfile(data["video_pad"])
        assert data["video_pad"] == analyse_video_pad(bieb, aid)

        # Cloud-sync-check (fase 4): grootte klopt → None; kleiner → onvolledig; weg → ontbreekt.
        assert video_sync_status(data["video_pad"], data["meta"]["video_bytes"]) is None
        assert video_sync_status(data["video_pad"], data["meta"]["video_bytes"] + 999) == "onvolledig"
        assert video_sync_status(os.path.join(tmp, "weg.mp4"), 100) == "ontbreekt"

        # Snelheidsproef: een bestand in een tempmap staat per definitie lokaal, en een
        # bestand dat er niet is levert None (geen exceptie de GUI in). De cloud-kant valt
        # niet na te bootsen zonder cloudmap — die is met de hand gemeten, zie de docstring.
        assert bestand_lokaal(data["video_pad"]) == "lokaal"
        assert bestand_lokaal(os.path.join(tmp, "weg.mp4")) is None

        # ── Opnames / bronvideo's (fase 8) ────────────────────────────────────────
        # De map is de waarheid over wélke bestanden er zijn: scannen levert de rijen,
        # en een tweede scan zonder nieuwe bestanden schrijft niets (rust voor de syncer).
        assert os.path.isdir(opnames_pad(bieb))
        assert lijst_bronvideos(bieb) == []
        opname = os.path.join(opnames_pad(bieb), "Training 3 aug.mp4")
        with open(opname, "wb") as f:
            f.write(b"nep-opname van een half uur")
        with open(os.path.join(opnames_pad(bieb), "aantekeningen.txt"), "w") as f:
            f.write("geen video, hoort niet in de lijst")
        # meta_lezer geïnjecteerd: het nepbestand is geen echte video.
        assert synchroniseer_bronmap(bieb, meta_lezer=lambda p: (30.0, 54000)) == 1
        assert synchroniseer_bronmap(bieb, meta_lezer=lambda p: (30.0, 54000)) == 0  # idempotent
        bronnen = lijst_bronvideos(bieb)
        assert len(bronnen) == 1                     # het .txt-bestand telt niet mee
        bron = bronnen[0]
        assert bron["bestand"] == f"{OPNAMES_MAP}/Training 3 aug.mp4"
        assert bron["totaal_frames"] == 54000 and bron["status"] == BRON_STATUS_DEFAULT
        assert bron["sync"] is None and bron["aantal_fragmenten"] == 0
        assert os.path.isfile(bron["pad"])

        wijzig_bronvideo(bieb, bron["id"], status="bezig", notitie="tempo-serie",
                         bijgewerkt_door="Coach Tester")
        bron = bronvideo(bieb, bron["id"])
        assert bron["status"] == "bezig" and bron["notitie"] == "tempo-serie"
        assert bron["bijgewerkt_door"] == "Coach Tester"

        # Een analyse die uit een stuk van deze opname geknipt is → grijze blokken.
        assert bron_fragmenten(bieb, bron["id"]) == []
        aid_f = sla_analyse_op(bieb, sid, "Fragment 1", video, info, resultaten, events,
                               "MediaPipe", {}, bron_id=bron["id"],
                               bron_start_frame=1200, bron_eind_frame=1560)
        frag = bron_fragmenten(bieb, bron["id"])
        assert len(frag) == 1 and frag[0]["start_frame"] == 1200
        assert frag[0]["eind_frame"] == 1560 and frag[0]["schaatser"] == "Test Schaatser"
        assert lijst_bronvideos(bieb)[0]["aantal_fragmenten"] == 1
        assert analyse_meta(bieb, aid_f)["bron_id"] == bron["id"]
        # Een losse clip houdt bron_id NULL — er valt niets te raden.
        assert analyse_meta(bieb, aid)["bron_id"] is None
        verwijder_analyse(bieb, aid_f)
        assert bron_fragmenten(bieb, bron["id"]) == []

        # Punten uit het handmatige kijkvenster: bewaren, hernoemen, verplaatsen, wissen.
        assert lijst_markeringen(bieb, bron["id"]) == []
        assert lijst_bronvideos(bieb)[0]["aantal_punten"] == 0
        p2 = voeg_markering_toe(bieb, bron["id"], 900, "sprong", "Coach Tester")
        p1 = voeg_markering_toe(bieb, bron["id"], 300, "start serie")
        punten = lijst_markeringen(bieb, bron["id"])
        assert [p["frame"] for p in punten] == [300, 900]        # op frame gesorteerd
        assert punten[1]["id"] == p2 and punten[1]["label"] == "sprong"
        assert punten[1]["aangemaakt_door"] == "Coach Tester"
        assert lijst_bronvideos(bieb)[0]["aantal_punten"] == 2
        wijzig_markering(bieb, p1, label="warming-up", frame=250)
        punten = lijst_markeringen(bieb, bron["id"])
        assert punten[0]["frame"] == 250 and punten[0]["label"] == "warming-up"
        wijzig_markering(bieb, p1)                               # niets op te geven = niets doen
        assert lijst_markeringen(bieb, bron["id"])[0]["frame"] == 250
        verwijder_markering(bieb, p1)
        assert [p["id"] for p in lijst_markeringen(bieb, bron["id"])] == [p2]
        # De foreign key wordt echt gehandhaafd (PRAGMA foreign_keys=ON in _verbind).
        try:
            voeg_markering_toe(bieb, 999999, 10, "nergens bij")
            raise AssertionError("een punt bij een onbekende opname had moeten falen")
        except sqlite3.IntegrityError:
            pass
        verwijder_markering(bieb, p2)

        # De sync-check van fase 4 werkt één op één op een opname die nog binnenkomt.
        with open(bron["pad"], "wb") as f:
            f.write(b"half")
        assert lijst_bronvideos(bieb)[0]["sync"] == "onvolledig"
        os.remove(bron["pad"])
        assert lijst_bronvideos(bieb)[0]["sync"] == "ontbreekt"   # rij blijft staan

        # Conflictkopie-detectie (fase 4): een tweede .db-bestand wordt gemeld.
        assert detecteer_conflictkopieen(bieb) == []
        with open(os.path.join(bieb, "schaats-LAPTOP.db"), "wb") as f:
            f.write(b"nep-conflictkopie")
        assert detecteer_conflictkopieen(bieb) == ["schaats-LAPTOP.db"]
        # ... maar een willekeurige andere database in de map is géén conflictkopie.
        with open(os.path.join(bieb, "adressen.db"), "wb") as f:
            f.write(b"iets heel anders")
        assert detecteer_conflictkopieen(bieb) == ["schaats-LAPTOP.db"]
        os.remove(os.path.join(bieb, "adressen.db"))
        os.remove(os.path.join(bieb, "schaats-LAPTOP.db"))
        terug = resultaten_naar_arrays(data["resultaten"], data["info"])
        assert np.array_equal(terug["landmarks"], arrays["landmarks"])

        hernoem_analyse(bieb, aid, "Hernoemd")
        assert lijst_analyses(bieb, sid)[0]["titel"] == "Hernoemd"

        # Skelet-editor (fase 3): bewerken → ruw-backup ontstaat, npz wijzigt,
        # bewerkt=1, cache bijgewerkt; herstel origineel → npz == origineel, bewerkt=0.
        map_a = os.path.join(bieb, MEDIA_MAP, aid)
        assert not os.path.isfile(os.path.join(map_a, NPZ_RUW_NAAM))   # nog niet bewerkt
        bewerkt_res = list(data["resultaten"])
        r0 = bewerkt_res[0]
        r0.lm[26] = r0.lm[26]._replace(x=0.123, y=0.456, visibility=1.0)  # r_knie verschoven
        bewerkte_events = [AfzetEvent(0, "links", 0, 5, 0.00, 0.20, 30.0, 28.0, 33.0)]
        bewaar_bewerkte_landmarks(bieb, aid, bewerkt_res, data["info"], bewerkte_events)
        assert os.path.isfile(os.path.join(map_a, NPZ_RUW_NAAM))
        with np.load(os.path.join(map_a, NPZ_NAAM)) as gew:
            assert abs(float(gew["landmarks"][0, 26, 0]) - 0.123) < 1e-6
        with np.load(os.path.join(map_a, NPZ_RUW_NAAM)) as orig:
            assert np.array_equal(orig["landmarks"], arrays["landmarks"])
        la_b = lijst_analyses(bieb, sid)
        assert la_b[0]["bewerkt"] == 1 and la_b[0]["aantal_afzetten"] == 1

        assert herstel_originele_landmarks(bieb, aid) is True
        with np.load(os.path.join(map_a, NPZ_NAAM)) as hersteld:
            assert np.array_equal(hersteld["landmarks"], arrays["landmarks"])
        ververs_events_cache(bieb, aid, events)   # zoals de GUI na herstel doet
        la_h = lijst_analyses(bieb, sid)
        assert la_h[0]["bewerkt"] == 0 and la_h[0]["aantal_afzetten"] == 4
        assert abs(la_h[0]["gem_hoek"] - 41.0) < 1e-9   # beide onvolledig-redenen weer eruit

        # Fout-injectie: onleesbare video → geen DB-rij, geen (extra) mediamap.
        try:
            sla_analyse_op(bieb, sid, "Kapot", os.path.join(tmp, "bestaat_niet.mp4"),
                           info, resultaten, events, "MediaPipe", {})
            raise AssertionError("sla_analyse_op had moeten falen")
        except OSError:
            pass
        assert len(lijst_analyses(bieb, sid)) == 1
        assert os.listdir(os.path.join(bieb, MEDIA_MAP)) == [aid]

        verwijder_analyse(bieb, aid)
        assert lijst_analyses(bieb, sid) == []
        assert not os.path.isdir(os.path.join(bieb, MEDIA_MAP, aid))

        # Cascade: schaatser weg → analyses + events-cache + mediamappen weg.
        aid2 = sla_analyse_op(bieb, sid, "Nog een", video, info, resultaten, events,
                              "MediaPipe", {})
        verwijder_schaatser(bieb, sid)
        assert lijst_schaatsers(bieb) == []
        assert not os.path.isdir(os.path.join(bieb, MEDIA_MAP, aid2))
        with _verbind(bieb) as con:
            assert con.execute("SELECT COUNT(*) FROM afzet_event_cache").fetchone()[0] == 0

        print("Zelftest OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
