"""
schaats_db.py — de bibliotheek (fase 1): schaatser-profielen + opgeslagen analyses.

Eén bibliotheekmap (pad instelbaar via config, later deelbaar via een cloudmap):

    <bibliotheek>/
      schaats.db                  ← SQLite: schaatsers, analyses, events-cache
      media/<analyse-uuid>/
        <originele videonaam>     ← gekopieerd origineel
        landmarks.npz             ← gesmoothte landmarks (fase 0-serialisatie)

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
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date

from schaats_analyse import sla_landmarks_op, laad_landmarks

DB_NAAM     = "schaats.db"
MEDIA_MAP   = "media"
NPZ_NAAM    = "landmarks.npz"
SCHEMA_VERSIE = 1
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


def open_db(bieb):
    """Maakt de bibliotheekmap + database + schema aan als die nog niet bestaan.
    Idempotent; aanroepen bij het opstarten en na het wisselen van bibliotheekpad."""
    os.makedirs(os.path.join(bieb, MEDIA_MAP), exist_ok=True)
    with _verbind(bieb) as con:
        versie = con.execute("PRAGMA user_version").fetchone()[0]
        if versie == 0:
            con.executescript(_SCHEMA)
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSIE}")


def _abs_pad(bieb, rel):
    """Relatief DB-pad (forward slashes) → absoluut pad op dit OS."""
    return os.path.join(bieb, *rel.split("/"))


def _normaliseer_backend(naam):
    return "yolo" if (naam or "").lower().startswith("yolo") else "mediapipe"


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
    """Lijstweergave uit de events-cache (geen npz nodig): titel, datum,
    aantal afzetten, gemiddelde hoek."""
    with _verbind(bieb) as con:
        rijen = con.execute(
            "SELECT a.id, a.titel, a.datum, a.backend, a.bewerkt, a.aangemaakt_op,"
            "       (SELECT COUNT(*)  FROM afzet_event_cache c WHERE c.analyse_id = a.id)"
            "       AS aantal_afzetten,"
            "       (SELECT AVG(hoek) FROM afzet_event_cache c WHERE c.analyse_id = a.id)"
            "       AS gem_hoek "
            "FROM analyse a WHERE a.schaatser_id = ? "
            "ORDER BY a.datum DESC, a.aangemaakt_op DESC", (schaatser_id,)).fetchall()
        return [dict(r) for r in rijen]


def sla_analyse_op(bieb, schaatser_id, titel, video_pad, info, resultaten, events,
                   backend, instellingen, datum=None):
    """
    Slaat een afgeronde analyse op in de bibliotheek: kopieert de video, schrijft de
    landmarks als .npz en insert de DB-rij + events-cache in één transactie.
    De DB-insert is bewust de láátste stap: gaat er iets mis (video onleesbaar, schijf
    vol), dan wordt de mediamap opgeruimd en staat er nooit een halve analyse in de
    bibliotheek. Retourneert het analyse-id (UUID).
    Draait in de praktijk in de workerthread — de videokopie kan lang duren.
    """
    analyse_id = str(uuid.uuid4())
    doelmap = os.path.join(bieb, MEDIA_MAP, analyse_id)
    videonaam = os.path.basename(video_pad)
    try:
        os.makedirs(doelmap)
        shutil.copy2(video_pad, os.path.join(doelmap, videonaam))
        sla_landmarks_op(os.path.join(doelmap, NPZ_NAAM), resultaten, info)

        rel_video = f"{MEDIA_MAP}/{analyse_id}/{videonaam}"
        with _verbind(bieb) as con:
            con.execute(
                "INSERT INTO analyse(id, schaatser_id, titel, datum, video_bestand,"
                "                    w, h, fps, totaal_frames, backend, instellingen_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (analyse_id, schaatser_id, titel,
                 datum or date.today().isoformat(), rel_video,
                 info.w, info.h, info.fps, info.totaal,
                 _normaliseer_backend(backend),
                 json.dumps(instellingen or {}, ensure_ascii=False)))
            con.executemany(
                "INSERT INTO afzet_event_cache(analyse_id, idx, been, start_frame,"
                "                              eind_frame, hoek, min_hoek, max_hoek, opmerking)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(analyse_id, i, ev.been, ev.start_frame, ev.eind_frame,
                  ev.hoek, ev.min_hoek, ev.max_hoek, ev.opmerking)
                 for i, ev in enumerate(events)])
        return analyse_id
    except Exception:
        shutil.rmtree(doelmap, ignore_errors=True)
        raise


def laad_analyse(bieb, analyse_id):
    """Laadt een analyse: DB-meta (incl. geparste instellingen) + landmarks uit het
    .npz. Retourneert {'meta', 'info', 'resultaten', 'video_pad'}; de afgeleiden zijn
    nog leeg — draai verwerk_afgeleiden() + segmenteer_afzetten() (fase 0-naad)."""
    with _verbind(bieb) as con:
        rij = con.execute("SELECT * FROM analyse WHERE id = ?", (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analyse {analyse_id} staat niet in de bibliotheek.")
    meta = dict(rij)
    try:
        meta["instellingen"] = json.loads(rij["instellingen_json"] or "{}")
    except ValueError:
        meta["instellingen"] = {}
    # npz-lezen buiten de DB-verbinding houden (verbinding zo kort mogelijk).
    info, resultaten = laad_landmarks(os.path.join(bieb, MEDIA_MAP, analyse_id, NPZ_NAAM))
    return {"meta": meta, "info": info, "resultaten": resultaten,
            "video_pad": _abs_pad(bieb, rij["video_bestand"])}


def analyse_video_pad(bieb, analyse_id):
    """Absoluut pad van de gekopieerde video van deze analyse."""
    with _verbind(bieb) as con:
        rij = con.execute("SELECT video_bestand FROM analyse WHERE id = ?",
                          (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analyse {analyse_id} staat niet in de bibliotheek.")
    return _abs_pad(bieb, rij["video_bestand"])


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
        ]
        video = os.path.join(tmp, "Testvideo.mp4")
        with open(video, "wb") as f:
            f.write(b"geen echte video, copy2 kijkt niet naar de inhoud")

        sid = maak_schaatser(bieb, "Test Schaatser", 2010, "proefkonijn")
        wijzig_schaatser(bieb, sid, "Test Schaatser", 2011, "proefkonijn 2.0")
        s = lijst_schaatsers(bieb)
        assert len(s) == 1 and s[0]["geboortejaar"] == 2011 and s[0]["aantal_analyses"] == 0

        instellingen = {"smooth_n": 5, "threshold": 0.015, "smooth_landmarks": True,
                        "doel_punt": [0.5, 0.5], "horizon_deg": 0.0, "auto_horizon": False,
                        "heavy": False, "backend_naam": "YOLO-pose + ByteTrack",
                        "perspectief_gebruikt": False}
        aid = sla_analyse_op(bieb, sid, "Proefanalyse", video, info, resultaten, events,
                             backend="YOLO-pose + ByteTrack", instellingen=instellingen)
        assert os.path.isfile(os.path.join(bieb, MEDIA_MAP, aid, "Testvideo.mp4"))
        assert os.path.isfile(os.path.join(bieb, MEDIA_MAP, aid, NPZ_NAAM))

        la = lijst_analyses(bieb, sid)
        assert len(la) == 1 and la[0]["aantal_afzetten"] == 2
        assert abs(la[0]["gem_hoek"] - 41.0) < 1e-9 and la[0]["backend"] == "yolo"
        assert lijst_schaatsers(bieb)[0]["aantal_analyses"] == 1

        data = laad_analyse(bieb, aid)
        assert data["meta"]["titel"] == "Proefanalyse"
        assert data["meta"]["instellingen"] == instellingen
        assert os.path.isfile(data["video_pad"])
        assert data["video_pad"] == analyse_video_pad(bieb, aid)
        terug = resultaten_naar_arrays(data["resultaten"], data["info"])
        assert np.array_equal(terug["landmarks"], arrays["landmarks"])

        hernoem_analyse(bieb, aid, "Hernoemd")
        assert lijst_analyses(bieb, sid)[0]["titel"] == "Hernoemd"

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
