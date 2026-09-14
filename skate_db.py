"""
skate_db.py — the library (phase 1): skater profiles + saved analyses.

One library folder (path configurable, later shareable via a cloud folder):

    <library>/
      schaats.db                  <- SQLite: skaters, analyses, events cache
      media/<analysis-uuid>/
        <original video name>     <- copied original
        landmarks.npz             <- smoothed landmarks (phase 0 serialization)
      opnames/                    <- raw training recordings, still to be trimmed (phase 8)

All SQL and path logic lives here; the GUI only talks to this module.

Cloud-folder discipline (makes sharing via Google Drive/OneDrive/Dropbox possible, phase 4):
- journal_mode=DELETE (no WAL: -wal/-shm side files sync only half -> corruption risk);
- connections are opened and closed per call, never held open — the DB file is nearly
  always "at rest" for the syncer, and sqlite3 objects therefore also never cross a
  thread (saving runs in the GUI worker thread);
- busy_timeout for the rare case of two trainers writing at the same time;
- every path in the DB is relative to the library folder, with forward slashes.

Stdlib + numpy (indirectly, via skate_analysis); importable in both venvs.
Self-test without video or GUI: `python skate_db.py`.

Note on file/folder names: `schaats.db`, `media/`, and `opnames/` keep their original
names deliberately (see the translate-to-english plan) — they're storage internals
inside a live, Google-Drive-synced library that other trainers already have on disk;
renaming them would be an in-place rename inside a folder several machines are actively
syncing, for zero readability benefit (nobody browses this by hand, everything goes
through this module). The SQL schema itself, which only this module ever reads by
name, is fully translated below via a v5->v6 migration.
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

from skate_analysis import (sla_landmarks_op, laad_landmarks, video_info,
                             ONV_AFGEKAPT, ONV_GEEN_PUSH, is_frozen, data_dir)

DB_NAME      = "schaats.db"
MEDIA_DIR    = "media"
RECORDINGS_DIR = "opnames"           # raw, not-yet-trimmed recordings (phase 8)
NPZ_NAME     = "landmarks.npz"
NPZ_RAW_NAME = "landmarks_ruw.npz"   # pristine landmarks from before the first manual edit (phase 3)
SCHEMA_VERSION = 6  # v2 (phase 4): analyse.video_bytes; v3 (phase 8): bronvideo + analyse.bron_*;
                    # v4: bron_markering (points from the manual viewing window);
                    # v6: whole schema translated to English (tables, columns, the
                    # source_video.status enum, the event cache's stored text, and
                    # several settings_json content keys)
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".wmv")

# Status of a recording in the work list (phase 8). Deliberately manual: the program
# can't know whether a trainer considers a recording done, so nothing ever gets set to
# 'done' automatically — it only shows the count ("3 fragments - 2 analyses").
SOURCE_STATUSES = ("todo", "in_progress", "done", "unusable")
SOURCE_STATUS_DEFAULT = SOURCE_STATUSES[0]
# Text flags in push_event_cache.note for a push that stays visible but falls outside
# the average angle (`PushEvent.incomplete`). The texts are skate_analysis's own, so
# caches from before the second reason just keep working: `ONV_AFGEKAPT` is still
# literally "truncated" (translated from "afgekapt" together with skate_analysis.py;
# see the v5->v6 migration for how already-cached rows from before that rename catch up).
TRUNCATED_MARKER = ONV_AFGEKAPT
INCOMPLETE_MARKERS = (ONV_AFGEKAPT, ONV_GEEN_PUSH)
ENV_LIBRARY = "SKATEANALYSIS_LIBRARY"   # override for tests
ENV_LOCAL = "SKATEANALYSIS_LOCAL"       # same, for the local library
LOCAL_DIR = "local"                     # under data_dir(): the local library


# ── Config (per user, so local — not in the shared folder) ─────────────────────

def config_path():
    """Where config.json lives (per-user, %APPDATA%, not the shared library folder).

    One-time migration: earlier versions (before the translate-to-english rename)
    wrote this under the old folder name "SchaatsAnalyse". If the new folder's
    config.json doesn't exist yet but the old one does, copy it forward — otherwise a
    trainer who already picked a library folder and set their name would appear to
    lose both the moment this version runs. Same pattern as `data_dir()` in
    skate_environment.py; deliberately a file copy here (not a folder rename) since
    this folder holds exactly one file and a copy is safe even if something still has
    the old file open."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "SkateAnalysis", "config.json")
    if not os.path.isfile(path):
        old_path = os.path.join(base, "SchaatsAnalyse", "config.json")
        if os.path.isfile(old_path):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.copy2(old_path, path)
            except OSError:
                pass
    return path


def default_library():
    return os.path.join(os.path.expanduser("~"), "Documents", "SkateAnalysis")


def local_library():
    """The **local** library: the same database layout as the shared one, but on this
    PC (`%LOCALAPPDATA%\\SkateAnalysis\\local`, next to the log) and never in the Drive.

    This is where the **loose videos** opened via "View new video" live, with their
    points. Those rows carry an absolute path from THIS pc, and that doesn't belong in
    the shared database: a colleague gets nothing out of it except a "file not found"
    row. It's deliberately a second library, not a second storage path — every function
    in this module works on it unchanged (`bronvideo_voor_pad`, `voeg_markering_toe`,
    ...), so the viewing window just needs to be given a different `bieb` path.
    `open_db` also creates an empty `media/` and `opnames/` next to it; those cost
    nothing and it keeps a single creation path. No sync discipline needed (no cloud
    folder), but it comes along for free."""
    path = os.environ.get(ENV_LOCAL) or os.path.join(data_dir(), LOCAL_DIR)
    open_db(path)
    return path


def load_config():
    """Reads config.json; unreadable/missing -> defaults (never crash on startup)."""
    try:
        with open(config_path(), encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config is not a dict")
    except Exception:
        cfg = {}
    # Dual-read (translate-to-english, phase 8): an existing config.json on a trainer's
    # machine still has the old Dutch keys. Carry the value over to the new key (without
    # touching the old one — nothing here rewrites the file), so nobody's saved library
    # path or trainer name silently vanishes the first time this version runs. Every
    # write from here on (skate_gui.py's two call sites) uses only the new keys, so a
    # config.json saved once under this version never has the old keys again.
    if "library_path" not in cfg and "bibliotheek_pad" in cfg:
        cfg["library_path"] = cfg["bibliotheek_pad"]
    if "trainer_name" not in cfg and "trainer_naam" in cfg:
        cfg["trainer_name"] = cfg["trainer_naam"]
    cfg.setdefault("library_path", default_library())
    cfg.setdefault("trainer_name", "")    # phase 4: travels along as analysis.created_by
    return cfg


def save_config(cfg):
    """Writes config.json atomically (tmp + os.replace), so a crash halfway never
    leaves a half/corrupt config file behind."""
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def library_path():
    env = os.environ.get(ENV_LIBRARY)
    if env:
        return env
    return load_config()["library_path"]


def trainer_name():
    """The name of the current trainer (phase 4), empty if not set. Saved as
    created_by on new analyses, so in a shared library it's visible who made which
    analysis."""
    return (load_config().get("trainer_name") or "").strip()


# ── App version (which code produced this analysis?) ───────────────────────────

_app_version_cache = None


def _git(*args):
    """Runs a git command in the repo folder; "" on any error (no git, no repo,
    timeout). CREATE_NO_WINDOW avoids a console flash from the GUI on Windows."""
    try:
        r = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__))] + list(args),
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _version_from_bundle():
    """The version as the build script set it in `_versie.py`, or None.

    In a bundled .exe there's no git and no repo, so `_git()` would return empty
    everywhere and every colleague's analysis would land in the library **without a
    version stamp** — exactly what the Info dialog and the title tooltip rely on. The
    build script therefore records the same fields in a generated `_versie.py`, and the
    label format stays exactly the same ("2026-08-24 . 4df9ab5a") so analyses from the
    exe and from the repo stay comparable.
    """
    try:
        import _versie
    except Exception:
        return None
    commit = str(getattr(_versie, "COMMIT", "") or "").strip()
    if not commit:
        return None
    datum = str(getattr(_versie, "DATUM", "") or "").strip()
    dirty = bool(getattr(_versie, "VUIL", False))
    return {"commit": commit, "datum": datum, "vuil": dirty,
            "label": f"{datum} . {commit}{'+' if dirty else ''}"}


def app_version():
    """Which version of the app produced this analysis? -> dict with `commit` (short
    hash), `datum` (commit date, ISO), `vuil` (uncommitted changes), and `label`
    ("2026-08-05 . 7e013fb2+"). Outside a git repo, every field is empty.

    The tracking logic changes regularly during development, so a saved analysis must
    show afterward which code made it. Deliberately from git and not a manually bumped
    constant: that falls behind precisely while developing quickly, and then lies.
    The commit date is the human-readable part for a trainer, the hash the precise part
    to run `git show` on. `vuil` (the `+`) doesn't count untracked files — videos and
    npz's next to the code say nothing about the logic that ran.

    In a bundled .exe the answer comes from `_versie.py` (see `_version_from_bundle`)
    instead of from git; the format is identical.

    Measured once per process (subprocess costs time; the code doesn't change during a
    running session)."""
    global _app_version_cache
    if _app_version_cache is None:
        bundled = _version_from_bundle() if is_frozen() else None
        if bundled is not None:
            _app_version_cache = bundled
        else:
            commit = datum = ""
            uit = _git("log", "-1", "--abbrev=8", "--format=%h%x09%cs")
            if "\t" in uit:
                commit, datum = uit.split("\t", 1)
            dirty = bool(commit) and bool(_git("status", "--porcelain", "-uno"))
            label = f"{datum} . {commit}{'+' if dirty else ''}" if commit else ""
            _app_version_cache = {"commit": commit, "datum": datum,
                                 "vuil": dirty, "label": label}
    return dict(_app_version_cache)


# ── Connection + schema ──────────────────────────────────────────────────────────

@contextmanager
def _connect(bieb):
    """Short-lived connection: open -> transaction -> close (see the module docstring)."""
    con = sqlite3.connect(os.path.join(bieb, DB_NAME), timeout=5.0)
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
CREATE TABLE skater(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    birth_year    INTEGER,
    notes         TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE source_video(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file            TEXT NOT NULL UNIQUE,        -- relative path ('opnames/...'), forward slashes
    name            TEXT NOT NULL,
    bytes           INTEGER,                     -- sync check only, NOT identity
    fps             REAL,
    total_frames    INTEGER,
    status          TEXT NOT NULL DEFAULT 'todo',
    note            TEXT NOT NULL DEFAULT '',
    updated_by      TEXT NOT NULL DEFAULT '',    -- who last set the status/note
    interlaced      INTEGER,                     -- 1/0, NULL = not determined yet (v5)
    added_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE source_marking(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    frame           INTEGER NOT NULL,            -- frame number in the recording
    label           TEXT NOT NULL DEFAULT '',
    created_by      TEXT NOT NULL DEFAULT '',    -- who set the point (shared library)
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE analysis(
    id                TEXT PRIMARY KEY,          -- UUID, also the folder name under media/
    skater_id         INTEGER NOT NULL REFERENCES skater(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    date              TEXT NOT NULL,             -- ISO (YYYY-MM-DD)
    video_file        TEXT NOT NULL,             -- relative path, forward slashes
    w                 INTEGER,
    h                 INTEGER,
    fps               REAL,
    total_frames      INTEGER,
    backend           TEXT NOT NULL,             -- 'yolo' | 'mediapipe'
    settings_json     TEXT NOT NULL DEFAULT '{}',
    created_by        TEXT NOT NULL DEFAULT '',  -- trainer name (phase 4)
    edited            INTEGER NOT NULL DEFAULT 0,-- manual skeleton edits (phase 3)
    video_bytes       INTEGER,                   -- size of the copied video (phase 4, cloud-sync check)
    source_id         INTEGER REFERENCES source_video(id) ON DELETE SET NULL,
    source_start_frame INTEGER,                  -- which part of the recording this clip comes from
    source_end_frame  INTEGER,                   -- (all three NULL for a loose clip)
    created_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE push_event_cache(
    analysis_id TEXT NOT NULL REFERENCES analysis(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    leg         TEXT,
    start_frame INTEGER,
    end_frame   INTEGER,
    angle       REAL,
    min_angle   REAL,
    max_angle   REAL,
    note        TEXT,
    PRIMARY KEY (analysis_id, idx)
);
"""


class LibraryTooNew(RuntimeError):
    """The library was created with a newer version of the app (user_version >
    SCHEMA_VERSION). We leave it untouched: the schema may have columns/tables this
    version doesn't know about, and setting the version back down would be wrong
    (see open_db)."""


def open_db(bieb):
    """Creates the library folder + database + schema if they don't exist yet, and
    migrates an older database to the current schema. Idempotent; call it on startup
    and after switching the library path.

    A **newer** database (shared cloud folder, a colleague with a more recent app) is
    refused with `LibraryTooNew` instead of being silently 'downgraded': setting
    `PRAGMA user_version` back down would make the newer app redo its own migration the
    next time it opens ("duplicate column name") and break the library. `user_version`
    is therefore only ever written after a successful creation or migration."""
    os.makedirs(os.path.join(bieb, MEDIA_DIR), exist_ok=True)
    os.makedirs(os.path.join(bieb, RECORDINGS_DIR), exist_ok=True)   # work-list folder (phase 8)
    with _connect(bieb) as con:
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise LibraryTooNew(
                f"This library was created with a newer version of the app "
                f"(schema v{version}; this app knows v{SCHEMA_VERSION}). "
                f"Update the app to be able to open it.")
        if version == 0:
            con.executescript(_SCHEMA)
        elif version < SCHEMA_VERSION:
            _migrate(con, version, bieb)
        else:
            return                        # already up to date; nothing to write
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


# Frozen historical DDL for the two tables the v2->v3 and v3->v4 steps below create.
# These must stay byte-for-byte what `_table_ddl("bronvideo"/"bron_markering")` used to
# produce (Dutch names, `interlaced` already included) BEFORE the v5->v6 rename below —
# exactly the same reasoning as the frozen `v1_schema` fixture in the self-test: an old
# library migrating through v3/v4 needs the schema *as it was*, not today's English one.
# `_table_ddl()` itself now only ever needs to look up *current* (English) table names;
# without this, it would raise `ValueError: substring not found` the instant `_SCHEMA`
# stopped containing a table literally called "bronvideo".
_OLD_BRONVIDEO_DDL = """
CREATE TABLE IF NOT EXISTS bronvideo(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bestand         TEXT NOT NULL UNIQUE,
    naam            TEXT NOT NULL,
    bytes           INTEGER,
    fps             REAL,
    totaal_frames   INTEGER,
    status          TEXT NOT NULL DEFAULT 'nog doen',
    notitie         TEXT NOT NULL DEFAULT '',
    bijgewerkt_door TEXT NOT NULL DEFAULT '',
    interlaced      INTEGER,
    toegevoegd_op   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""
_OLD_BRON_MARKERING_DDL = """
CREATE TABLE IF NOT EXISTS bron_markering(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bron_id         INTEGER NOT NULL REFERENCES bronvideo(id) ON DELETE CASCADE,
    frame           INTEGER NOT NULL,
    label           TEXT NOT NULL DEFAULT '',
    aangemaakt_door TEXT NOT NULL DEFAULT '',
    aangemaakt_op   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


def _migrate(con, van, bieb):
    """Updates an existing database step by step to SCHEMA_VERSION. Cloud-safe:
    ALTER TABLE ADD COLUMN/RENAME are small, in-place edits that keep a single DB file."""
    if van < 2:
        # v1 -> v2 (phase 4): column for the video size; old rows get NULL and thus
        # skip the sync-size check on open (existence check only).
        con.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")
    if van < 3:
        # v2 -> v3 (phase 8): the not-yet-trimmed recordings as a work list, plus the
        # origin of an analysis (which part of which recording). Old analyses keep
        # bron_id NULL — that's correct too: those came from a loose clip, not from a
        # recording. No migration that has to guess anything.
        con.execute(_OLD_BRONVIDEO_DDL)
        # The REFERENCES clause may come along in ADD COLUMN as long as the default is
        # NULL (SQLite), so a migrated library ends up with exactly the same schema as
        # a fresh one.
        for column in ("bron_id INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL",
                      "bron_start_frame INTEGER", "bron_eind_frame INTEGER"):
            con.execute(f"ALTER TABLE analyse ADD COLUMN {column}")
    if van < 4:
        # v3 -> v4: the points a trainer sets in the manual viewing window. A separate
        # table, not a column on bronvideo: there are several per recording, and they
        # belong to the recording (not to an analysis), so they disappear along if that
        # row were ever removed. Old libraries simply get an empty table.
        con.execute(_OLD_BRON_MARKERING_DDL)
    if van < 5:
        # v4 -> v5: remember whether a recording is interlaced. Measuring it costs a
        # couple of seconds per file (more on a streaming Drive), while the answer
        # never changes — so determine it once and keep it. NULL = not measured yet;
        # nothing to guess for existing rows.
        #
        # Conditional, because `_table_ddl` always produces the *newest* definition: a
        # library that only got `bronvideo` above (v2 -> v3) already has the column and
        # would trip over a duplicate column name.
        if not _has_column(con, "bronvideo", "interlaced"):
            con.execute("ALTER TABLE bronvideo ADD COLUMN interlaced INTEGER")
    if van < 6:
        # v5 -> v6: translate the whole schema — and the persisted values that are more
        # than just column labels — to English, as part of the translate-to-english
        # effort (see CLAUDE.md/the plan). Table/column renames are pure ALTER TABLE
        # metadata edits (fast, no row data rewritten); the status enum and the event
        # cache need their actual VALUES remapped too.

        # 1. Tables first. SQLite 3.25+ (this project ships 3.45.1 in the bundled exe;
        #    see GPU.md) auto-rewrites every other table's REFERENCES clause when the
        #    referenced table is renamed, so the order relative to the column renames
        #    below doesn't matter — every FK here targets `id`, which never changes.
        for old, new in (("schaatser", "skater"), ("bronvideo", "source_video"),
                          ("bron_markering", "source_marking"), ("analyse", "analysis"),
                          ("afzet_event_cache", "push_event_cache")):
            con.execute(f"ALTER TABLE {old} RENAME TO {new}")

        # 2. Columns, one RENAME COLUMN each.
        column_renames = {
            "skater": [("naam", "name"), ("geboortejaar", "birth_year"),
                       ("notities", "notes"), ("aangemaakt_op", "created_at")],
            "source_video": [("bestand", "file"), ("naam", "name"),
                              ("totaal_frames", "total_frames"), ("notitie", "note"),
                              ("bijgewerkt_door", "updated_by"),
                              ("toegevoegd_op", "added_at")],
            "source_marking": [("bron_id", "source_id"),
                                ("aangemaakt_door", "created_by"),
                                ("aangemaakt_op", "created_at")],
            "analysis": [("schaatser_id", "skater_id"), ("titel", "title"),
                         ("datum", "date"), ("video_bestand", "video_file"),
                         ("totaal_frames", "total_frames"),
                         ("instellingen_json", "settings_json"),
                         ("aangemaakt_door", "created_by"), ("bewerkt", "edited"),
                         ("bron_id", "source_id"),
                         ("bron_start_frame", "source_start_frame"),
                         ("bron_eind_frame", "source_end_frame"),
                         ("aangemaakt_op", "created_at")],
            "push_event_cache": [("analyse_id", "analysis_id"), ("been", "leg"),
                                  ("eind_frame", "end_frame"), ("hoek", "angle"),
                                  ("min_hoek", "min_angle"), ("max_hoek", "max_angle"),
                                  ("opmerking", "note")],
        }
        for table, columns in column_renames.items():
            for old, new in columns:
                con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")

        # 3. Status enum values. Persisted directly (also used verbatim as the status
        #    combobox's items in schaats_gui.py) with NO recompute path of its own —
        #    unlike the event cache below, nothing ever regenerates this column, so a
        #    row left on the old spelling would sit outside the new enum forever.
        con.execute(
            "UPDATE source_video SET status = CASE status "
            "WHEN 'nog doen' THEN 'todo' WHEN 'bezig' THEN 'in_progress' "
            "WHEN 'klaar' THEN 'done' WHEN 'onbruikbaar' THEN 'unusable' "
            "ELSE status END")

        # 4. `settings_json` content keys — not just the column that holds it (renamed
        #    above), but the JSON *inside* it. `doel_punt`/`doel_kader`/`perspectief`
        #    (the analyze()-parameter spelling) are deliberately NOT touched here: they
        #    mirror keyword arguments into skate_analysis.py's/skate_yolo.py's
        #    `analyze()` that stay Dutch permanently (see TRANSLATION_PROGRESS.md's
        #    glossary entry for `doel_kader`), so renaming the storage key would just
        #    invent a second spelling for the exact same concept. `perspective` (the
        #    saved calibration blob, a different concern from the live `perspectief=`
        #    argument that only ever exists in memory) DOES get renamed, along with the
        #    four keys below with no such tie to a function signature. A malformed or
        #    unreadable settings_json is left alone rather than aborting the whole
        #    migration for every other analysis.
        _key_renames = (("bocht_overslaan", "skip_corner"),
                        ("perspectief_gebruikt", "perspective_used"),
                        ("perspectief", "perspective"),
                        ("backend_naam", "backend_name"),
                        ("app_versie", "app_version"))
        for row in con.execute("SELECT id, settings_json FROM analysis").fetchall():
            try:
                settings = json.loads(row["settings_json"] or "{}")
            except (ValueError, TypeError):
                continue
            if not isinstance(settings, dict):
                continue
            gewijzigd = False
            for old, new in _key_renames:
                if old in settings and new not in settings:
                    settings[new] = settings.pop(old)
                    gewijzigd = True
            if gewijzigd:
                con.execute("UPDATE analysis SET settings_json = ? WHERE id = ?",
                           (json.dumps(settings, ensure_ascii=False), row["id"]))

        # 5. The event cache is disposable (the npz is the real source of truth,
        #    recomputed fresh every time an analysis is opened), but leaving it stale
        #    is worse than cosmetic: skate_analysis.py's own English rename (an earlier
        #    step of this same effort) already means a freshly computed cache row's
        #    `note` reads "truncated"/"no full push" rather than the original Dutch
        #    marker text — so a stale pre-migration row's `NOT LIKE '%truncated%'`
        #    would wrongly succeed and pull an actually-incomplete push into the
        #    average. Recompute every analysis through the exact pipeline a normal
        #    reopen uses; if that fails for one analysis (moved media, a cloud file
        #    still syncing), fall back to a plain text remap for just that row instead
        #    of leaving it on the pre-rename vocabulary.
        _recompute_all_caches(con, bieb)


def _has_column(con, table, column):
    return column in [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _table_ddl(name):
    """The CREATE TABLE for `name` out of _SCHEMA, as IF NOT EXISTS — so the schema
    lives in one place and a migration is guaranteed to use the same definition as a
    fresh library. Only meaningful for *current* (English) table names; a historical
    Dutch-named table needed by an old migration step has its own frozen DDL above,
    since slicing it out of today's _SCHEMA would raise (the name isn't in there
    anymore)."""
    kop = f"CREATE TABLE {name}("
    begin = _SCHEMA.index(kop)
    eind = _SCHEMA.index(");", begin) + 2
    return _SCHEMA[begin:eind].replace(kop, f"CREATE TABLE IF NOT EXISTS {name}(")


def _recompute_all_caches(con, bieb):
    """Rebuilds push_event_cache for every analysis from its .npz, the same pipeline a
    normal reopen uses (see load_analysis). Used by the v5->v6 migration so existing
    analyses' cached text catches up with skate_analysis.py's English rename; falls
    back to a plain marker-text remap for any single analysis this can't do right now
    (its npz isn't reachable), rather than letting one bad analysis abort the whole
    migration for every other analysis in the library."""
    from skate_analysis import (
        load_landmarks, process_derivatives, segment_pushes, PerspectiveConfig)
    rows = con.execute("SELECT id, settings_json FROM analysis").fetchall()
    for row in rows:
        analysis_id = row["id"]
        npz_path = os.path.join(bieb, MEDIA_DIR, analysis_id, NPZ_NAME)
        try:
            settings = json.loads(row["settings_json"] or "{}")
            info, results = load_landmarks(npz_path)
            perspective = None
            # "perspective" is the current spelling (step 4 above already renamed it
            # for every row by the time this runs); "perspectief" is read as a
            # fallback only for the rare case this recompute runs against
            # settings_json step 4 didn't reach (malformed JSON, since step 4 skips
            # those rows too).
            persp_dict = settings.get("perspective") or settings.get("perspectief")
            if persp_dict:
                try:
                    perspective = PerspectiveConfig.uit_dict(persp_dict)
                except Exception:
                    perspective = None
            process_derivatives(
                results, info.w, info.h, info.fps,
                smooth_n=settings.get("smooth_n", 5),
                threshold=settings.get("threshold", 0.015),
                perspectief=perspective)
            events = segment_pushes(results)
            _write_events_cache(con, analysis_id, events)
        except Exception:
            con.execute(
                "UPDATE push_event_cache SET note = "
                "REPLACE(REPLACE(note, 'afgekapt', 'truncated'), "
                "'geen volledige push', 'no full push') WHERE analysis_id = ?",
                (analysis_id,))


def _abs_path(bieb, rel):
    """Relative DB path (forward slashes) -> absolute path on this OS.

    One exception: a **loose video from this pc** (see `source_video_for_path`) is
    stored as an absolute path and must therefore not be glued onto the library —
    `os.path.join` would otherwise turn (`D:\\bieb`, `C:/film.mp4`) into `C:film.mp4`, a
    drive-relative path pointing at something else entirely."""
    if _is_external(rel):
        return os.path.normpath(rel)
    return os.path.join(bieb, *rel.split("/"))


def _is_external(bestand):
    """Is this DB path a loose video from this pc, rather than a file in the library?

    Everything the library manages itself is stored relative (`opnames/<name>`, the
    phase 1 discipline); an absolute path can therefore only ever be a loose video.
    That means no extra column — and thus no migration — is needed to tell the two
    kinds apart."""
    return os.path.isabs(bestand) or bool(os.path.splitdrive(bestand)[0])


def _normalize_backend(name):
    return "yolo" if (name or "").lower().startswith("yolo") else "mediapipe"


def _add_legacy_keys(d, **old_equals_new):
    """Adds the original Dutch keys back onto an already-built dict, each pointing at
    the same value as its new English column name, so schaats_gui.py (not translated
    yet, see the translate-to-english plan) can keep reading a returned row by its
    original key — module-level function/class aliases (see the bottom of this file)
    only cover `schaats_db.some_function`, not the *shape* of what a function returns,
    so this is the other half of that same compatibility story. Call as
    `_add_legacy_keys(d, oude_naam="new_name", ...)`; a mapping is only applied where
    `new_name` is actually present in `d`. Mutates and returns `d`. Remove each call
    once its caller reads the new names directly."""
    for old, new in old_equals_new.items():
        if new in d:
            d[old] = d[new]
    return d


def _meta_from_row(row):
    """DB row -> dict with the parsed `instellingen` added (corrupt JSON -> {}), plus
    the original Dutch keys for the columns that were renamed in the v6 migration.
    Shared by list_analyses, load_analysis, and analysis_meta."""
    meta = dict(row)
    try:
        meta["instellingen"] = json.loads(meta.get("settings_json") or "{}")
    except ValueError:
        meta["instellingen"] = {}
    return _add_legacy_keys(
        meta, titel="title", datum="date", video_bestand="video_file",
        totaal_frames="total_frames", aangemaakt_door="created_by",
        aangemaakt_op="created_at", bewerkt="edited", schaatser_id="skater_id",
        instellingen_json="settings_json", bron_id="source_id",
        bron_start_frame="source_start_frame", bron_eind_frame="source_end_frame")


# ── Skaters ────────────────────────────────────────────────────────────────────

def create_skater(bieb, naam, geboortejaar=None, notities=""):
    with _connect(bieb) as con:
        cur = con.execute(
            "INSERT INTO skater(name, birth_year, notes) VALUES (?, ?, ?)",
            (naam, geboortejaar, notities))
        return cur.lastrowid


def edit_skater(bieb, schaatser_id, naam, geboortejaar=None, notities=""):
    with _connect(bieb) as con:
        con.execute(
            "UPDATE skater SET name = ?, birth_year = ?, notes = ? WHERE id = ?",
            (naam, geboortejaar, notities, schaatser_id))


def list_skaters(bieb):
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM analysis a WHERE a.skater_id = s.id)"
            "       AS aantal_analyses "
            "FROM skater s ORDER BY s.name COLLATE NOCASE").fetchall()
        return [_add_legacy_keys(dict(r), naam="name", geboortejaar="birth_year",
                                  notities="notes", aangemaakt_op="created_at")
                for r in rijen]


def delete_skater(bieb, schaatser_id):
    """Deletes the skater and all their analyses (DB cascade + media folders)."""
    with _connect(bieb) as con:
        analyse_ids = [r["id"] for r in con.execute(
            "SELECT id FROM analysis WHERE skater_id = ?", (schaatser_id,))]
        con.execute("DELETE FROM skater WHERE id = ?", (schaatser_id,))
    # Media folders only AFTER the successful DB delete; a leftover folder without a DB
    # row is harmless (the other way around isn't).
    for aid in analyse_ids:
        shutil.rmtree(os.path.join(bieb, MEDIA_DIR, aid), ignore_errors=True)


# ── Analyses ───────────────────────────────────────────────────────────────────

def list_analyses(bieb, schaatser_id):
    """List view from the events cache (no npz needed): title, date, video duration
    (`total_frames`/`fps`), number of pushes, average angle.

    The average angle skips **incomplete** pushes — the push was still running when the
    video ended, or no sideways push was observed within the stance run; both give a
    far too steep angle that would drag the average up. They DO count toward the
    number, since they happened.

    `settings_json` comes along (parsed as `instellingen`) so the list view can show the
    app version without opening an analysis — that costs one json.loads per row,
    negligible next to the subselects below."""
    niet_like = " AND ".join(["c.note NOT LIKE ?"] * len(INCOMPLETE_MARKERS))
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT a.id, a.title, a.date, a.backend, a.edited, a.created_at,"
            "       a.created_by, a.settings_json, a.total_frames, a.fps,"
            "       (SELECT COUNT(*)  FROM push_event_cache c WHERE c.analysis_id = a.id)"
            "       AS aantal_afzetten,"
            "       (SELECT AVG(angle) FROM push_event_cache c WHERE c.analysis_id = a.id"
            f"         AND (c.note IS NULL OR ({niet_like}))) AS gem_hoek "
            "FROM analysis a WHERE a.skater_id = ? "
            "ORDER BY a.date DESC, a.created_at DESC",
            tuple(f"%{m}%" for m in INCOMPLETE_MARKERS) + (schaatser_id,)).fetchall()
        return [_meta_from_row(r) for r in rijen]


def list_calibrations(bieb, beeld_w=None, beeld_h=None):
    """Analyses carrying a saved perspective calibration (phase 7), newest first.

    Meant for reusing a calibration: a calibration belongs to one camera pose, not one
    clip, so every fragment from the same recording may share it. That's also a
    geometric requirement for comparing analyses with each other — tracing the same
    lines by hand seven times gives seven slightly different calibrations, and then
    you're measuring that spread instead of the correction's effect.

    With `beeld_w`/`beeld_h`, only calibrations of that same image size are returned:
    the lines are in pixels, so on a differently-sized video they'd sit next to where
    they should. Returns dicts with id/title/skater/date/`perspective` (the raw dict)
    and `note`.

    There's deliberately no separate calibration table: the calibration lives in
    `settings_json`, so this costs one json.loads per analysis and no schema bump."""
    # Local import, same reasoning as `_recompute_all_caches` below: keeps skate_db.py
    # importable without pulling in skate_perspective.py at module load. `from_dict()`
    # dual-reads image_w/image_h/note against the old beeld_w/beeld_h/notitie spelling
    # (Pattern E) -- reading `inv["beeld_w"]` directly here was a real, pre-existing bug
    # (found while translating this function): CalibrationInput.to_dict() has written
    # `image_w`/`image_h`/`note` since phase 3, so every calibration saved since then
    # had this function silently treat it as sizeless and note-less. Same class of bug
    # as the one already fixed in skate_gui.py's `_calibration_rows` (see the phase 8a
    # session 2 log in TRANSLATION_PROGRESS.md) -- that fix never touched this function,
    # since this one lives in skate_db.py.
    from skate_perspective import CalibrationInput
    uit = []
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT a.id, a.title, a.date, a.created_at, a.w, a.h,"
            "       a.settings_json, s.name AS schaatser "
            "FROM analysis a LEFT JOIN skater s ON s.id = a.skater_id "
            "ORDER BY a.date DESC, a.created_at DESC").fetchall()
    for r in rijen:
        try:
            inst = json.loads(r["settings_json"] or "{}")
        except (ValueError, TypeError):
            continue
        # `perspective` is the current settings-json storage key (skate_gui.py writes
        # it as of this session); `perspectief` is read as a fallback for an analysis
        # saved before this rename. Distinct from `perspectief=`, the keyword argument
        # into analyze(), which stays Dutch permanently -- see TRANSLATION_PROGRESS.md's
        # glossary entry for `doel_kader`, which the same reasoning applies to.
        p = inst.get("perspective") or inst.get("perspectief")
        if not p or "calibration_input" not in p and "invoer" not in p:
            continue
        try:
            inv = CalibrationInput.from_dict(p.get("calibration_input", p.get("invoer")))
        except (KeyError, TypeError, ValueError):
            continue
        if beeld_w is not None and inv.image_w != int(beeld_w):
            continue
        if beeld_h is not None and inv.image_h != int(beeld_h):
            continue
        uit.append({"id": r["id"], "titel": r["title"], "datum": r["date"],
                    "schaatser": r["schaatser"], "w": inv.image_w,
                    "h": inv.image_h, "perspective": p,
                    "note": inv.note})
    return uit


def save_analysis(bieb, schaatser_id, titel, video_pad, info, resultaten, events,
                   backend, instellingen, datum=None, aangemaakt_door="",
                   bron_id=None, bron_start_frame=None, bron_eind_frame=None):
    """
    Saves a finished analysis in the library: copies the video, writes the landmarks
    as .npz, and inserts the DB row + events cache in a single transaction. The DB
    insert is deliberately the LAST step: if anything goes wrong (unreadable video,
    disk full), the media folder is cleaned up and a half-finished analysis never ends
    up in the library. Returns the analysis id (UUID).
    In practice runs in the worker thread — the video copy can take a while.
    `aangemaakt_door` (phase 4) is the trainer name; `video_bytes` (the size of the
    copied video) is kept so a colleague opening the analysis while the cloud sync is
    still running can recognize a half download.

    The **app version** and the full `backend_name` are added to `instellingen` here —
    one place, so every save route (single analysis, batch, self-test) records it
    without having to think about it. It travels in `settings_json`, so no schema bump.
    `setdefault`: a caller that already filled it in wins.

    `bron_id` + `bron_start_frame`/`bron_eind_frame` (phase 8) record which part of
    which recording this clip was trimmed from; for a loose video they stay NULL. That
    makes "which parts of this recording are already done" a single query
    (`source_fragments`).
    """
    inst = dict(instellingen or {})
    # translate-to-english (phase 8): `backend_name`/`app_version` (English keys, both
    # renamed together with skate_gui.py's own reader in the same commit — the
    # v5->v6 migration above rewrites these two keys in every already-saved analysis,
    # so there's no old spelling left to dual-read here). `app_commit` needed no
    # rename, it was already English.
    inst.setdefault("backend_name", backend or "")   # the `backend` column is normalized
    versie = app_version()
    inst.setdefault("app_version", versie["label"])
    inst.setdefault("app_commit", versie["commit"])

    analyse_id = str(uuid.uuid4())
    doelmap = os.path.join(bieb, MEDIA_DIR, analyse_id)
    videonaam = os.path.basename(video_pad)
    try:
        os.makedirs(doelmap)
        videokopie = os.path.join(doelmap, videonaam)
        shutil.copy2(video_pad, videokopie)
        sla_landmarks_op(os.path.join(doelmap, NPZ_NAME), resultaten, info)
        video_bytes = os.path.getsize(videokopie)

        rel_video = f"{MEDIA_DIR}/{analyse_id}/{videonaam}"
        with _connect(bieb) as con:
            con.execute(
                "INSERT INTO analysis(id, skater_id, title, date, video_file,"
                "                    w, h, fps, total_frames, backend, settings_json,"
                "                    created_by, video_bytes,"
                "                    source_id, source_start_frame, source_end_frame)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (analyse_id, schaatser_id, titel,
                 datum or date.today().isoformat(), rel_video,
                 info.w, info.h, info.fps, info.totaal,
                 _normalize_backend(backend),
                 json.dumps(inst, ensure_ascii=False),
                 aangemaakt_door or "", video_bytes,
                 bron_id, bron_start_frame, bron_eind_frame))
            _write_events_cache(con, analyse_id, events)
        return analyse_id
    except Exception:
        shutil.rmtree(doelmap, ignore_errors=True)
        raise


def _write_events_cache(con, analyse_id, events):
    """Replaces the events cache of one analysis (DELETE + INSERT) within a running
    transaction. Shared by save_analysis, save_edited_landmarks, and
    refresh_events_cache — the cache is purely for a fast list view."""
    con.execute("DELETE FROM push_event_cache WHERE analysis_id = ?", (analyse_id,))
    con.executemany(
        "INSERT INTO push_event_cache(analysis_id, idx, leg, start_frame,"
        "                              end_frame, angle, min_angle, max_angle, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(analyse_id, i, ev.been, ev.start_frame, ev.eind_frame,
          ev.hoek, ev.min_hoek, ev.max_hoek, _cache_note(ev))
         for i, ev in enumerate(events)])


def _cache_note(ev):
    """Note text for the cache. The incomplete flag (`PushEvent.incomplete`: the push
    was still running when the video ended, or no sideways push was observed — both
    give a too-steep angle) gets no column of its own — that would cost a schema
    migration for something that's freshly recomputed on open anyway — but travels
    along as text, so `list_analyses` can keep such a push out of the average angle
    (`INCOMPLETE_MARKERS`)."""
    reden = getattr(ev, 'onvolledig', None)
    if reden:
        return f"{reden} . {ev.opmerking}" if ev.opmerking else reden
    return ev.opmerking


def analysis_meta(bieb, analyse_id):
    """Just the DB row of one analysis (incl. parsed `instellingen`), WITHOUT reading
    the npz — enough for an info overview (app version, backend, settings).
    `bron_naam` is joined in separately: showing the origin ("from recording X,
    12:30-13:05") needs the filename, and that lives in the source_video table."""
    with _connect(bieb) as con:
        rij = con.execute(
            "SELECT a.*, b.name AS bron_naam FROM analysis a "
            "LEFT JOIN source_video b ON b.id = a.source_id WHERE a.id = ?",
            (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analysis {analyse_id} is not in the library.")
    return _meta_from_row(rij)


def load_analysis(bieb, analyse_id):
    """Loads an analysis: DB meta (incl. parsed settings) + landmarks from the .npz.
    Returns {'meta', 'info', 'resultaten', 'video_pad'}; the derivatives are still
    empty — run process_derivatives() + segment_pushes() (the phase 0 seam)."""
    meta = analysis_meta(bieb, analyse_id)
    # Keep the npz read outside the DB connection (keep the connection as short as possible).
    info, resultaten = laad_landmarks(os.path.join(bieb, MEDIA_DIR, analyse_id, NPZ_NAME))
    return {"meta": meta, "info": info, "resultaten": resultaten,
            "video_pad": _abs_path(bieb, meta["video_file"])}


def analysis_video_path(bieb, analyse_id):
    """Absolute path of this analysis's copied video."""
    with _connect(bieb) as con:
        rij = con.execute("SELECT video_file FROM analysis WHERE id = ?",
                          (analyse_id,)).fetchone()
    if rij is None:
        raise KeyError(f"Analysis {analyse_id} is not in the library.")
    return _abs_path(bieb, rij["video_file"])


# ── Recordings: the not-yet-trimmed source videos (phase 8) ────────────────────
#
# The design rule: **the folder is the truth about which files exist, the database
# about what we know of them.** The file list is scanned from disk when opened, not
# read from the DB — otherwise the DB drifts the moment someone renames or deletes a
# file, and you're stuck with cleanup. The DB only holds what you can never read off
# disk: status, note, and which analyses come from which part of which recording.

def recordings_path(bieb, maak_aan=True):
    """Absolute path of `<library>/opnames/`. Inside the library folder, so every path
    stays relative (the phase 1 discipline) and no extra per-trainer path setting is
    needed; the folder is created if it doesn't exist yet."""
    pad = os.path.join(bieb, RECORDINGS_DIR)
    if maak_aan:
        os.makedirs(pad, exist_ok=True)
    return pad


def sync_source_dir(bieb, meta_lezer=None):
    """Scans `opnames/` and puts new files into the source_video table. Returns the
    number of recordings added.

    - **Identity is the relative path** (`opnames/<name>`, UNIQUE): one folder can't
      hold two files with the same name, and because the folder sits inside the
      library that path is the same for every trainer. A renamed file counts as new;
      the old row stays with its analyses and shows as "file not found".
    - **`bytes` does NOT belong in the key**, only in the sync check: a recording still
      arriving for a colleague is at that moment *smaller* than what's in the DB. Were
      the size part of the key, the scan would mistake a half-downloaded file for a new
      recording and add a second row — exactly when you need the fragment history most.
    - `INSERT OR IGNORE`, so two trainers who see the same new recording at the same
      time don't collide, and it **only writes when there's actually something new**:
      otherwise every trainer's every app start would touch the shared DB, while the
      phase 4 discipline wants it at rest as much as possible for the syncer.

    `meta_lezer(pad)` returns `(fps, total_frames)`; via OpenCV by default. An unreadable
    file (still downloading) simply shows up in the list with empty meta — better than
    skipping it, since then you wouldn't see that a recording exists at all.
    """
    map_pad = recordings_path(bieb)
    try:
        namen = sorted(n for n in os.listdir(map_pad)
                       if os.path.splitext(n)[1].lower() in VIDEO_EXTS)
    except OSError:
        return 0
    with _connect(bieb) as con:
        bekend = {r["file"] for r in con.execute("SELECT file FROM source_video")}
        nieuw = [n for n in namen if f"{RECORDINGS_DIR}/{n}" not in bekend]
        for naam in nieuw:
            pad = os.path.join(map_pad, naam)
            fps, totaal = (meta_lezer or _video_meta)(pad)
            try:
                grootte = os.path.getsize(pad)
            except OSError:
                grootte = None
            con.execute(
                "INSERT OR IGNORE INTO source_video(file, name, bytes, fps, total_frames,"
                "                                status) VALUES (?, ?, ?, ?, ?, ?)",
                (f"{RECORDINGS_DIR}/{naam}", naam, grootte, fps, totaal, SOURCE_STATUS_DEFAULT))
    return len(nieuw)


def _video_meta(pad):
    """(fps, total_frames) of a video file; (None, None) if it can't be read."""
    try:
        info = video_info(pad)
        return info.fps, info.totaal
    except Exception:
        return None, None


COPY_BLOCK = 4 * 1024 * 1024      # read step while copying from the camera (~10 updates/s)
COPY_PART = ".part"               # temporary name while copying — no video extension


class CopyAborted(Exception):
    """The user aborted the copy (`stop_check` returned True)."""


def copy_plan(bieb, paden):
    """What of `paden` (files on the camera/memory card) goes to `opnames/`, and what
    doesn't and why. One dict per path: `pad`, `naam`, `bytes`, `doel`, and `reden` —
    `None` = copy it, otherwise the reason to skip it. The GUI shows those reasons
    BEFORE the copy starts, since a thread runs during the copy that can't ask anything.

    **Nothing is ever overwritten.** The identity of a recording is `opnames/<name>`
    (`sync_source_dir`), and the trimmed fragments, status, and the team's points all
    hang off that row. Putting a file with the same name but a different size on top
    would silently point all that work at a different recording; the same size is
    almost certainly the same recording, and then there's nothing to copy. Both cases
    are reported and skipped — renaming on the camera is up to the trainer."""
    doelmap = recordings_path(bieb)
    plan = []
    gezien = set()
    for pad in paden:
        naam = os.path.basename(pad)
        doel = os.path.join(doelmap, naam)
        item = {"pad": pad, "naam": naam, "bytes": 0, "doel": doel, "reden": None}
        plan.append(item)
        if not os.path.isfile(pad):
            item["reden"] = "bestand niet gevonden"
            continue
        item["bytes"] = os.path.getsize(pad)
        if os.path.splitext(naam)[1].lower() not in VIDEO_EXTS:
            item["reden"] = "geen videobestand"
        elif _in_recordings_dir(bieb, pad):
            item["reden"] = "staat al in de map opnames"
        elif os.path.normcase(naam) in gezien:
            item["reden"] = "twee keer gekozen"
        elif os.path.exists(doel):
            try:
                bestaand = os.path.getsize(doel)
            except OSError:
                bestaand = None
            if bestaand == item["bytes"]:
                item["reden"] = "staat al in de bibliotheek"
            else:
                item["reden"] = ("er staat al een ánder bestand met deze naam in de "
                                 "bibliotheek — hernoem het eerst op de camera")
        gezien.add(os.path.normcase(naam))
    return plan


def copy_to_recordings(bieb, plan, progress_callback=None, stop_check=None):
    """Copies the items from `copy_plan` without a `reden` to `opnames/` and returns
    the paths of the successful copies. One failed file doesn't stop the rest: it gets
    its error as `reden` and the next one proceeds.

    - **Block by block** (`COPY_BLOCK`) instead of `shutil.copy2`, since that gives no
      progress: `progress_callback(gedaan_bytes, totaal_bytes, idx, naam)` after each
      block, and `stop_check()` in between — a 4 GB recording takes minutes and must be
      abortable.
    - **First under a temporary name** (`<naam>.part`, no video extension): the folder
      is the truth about which recordings exist, so a half-copied file under its real
      name would already be registered as a recording by `sync_source_dir` — on this pc
      AND for a colleague via Drive. Only after the last byte does `os.replace` give it
      the real name; on abort or an error the partial file is removed.
    - `copystat` afterward, so the recording date from the camera stays on the file."""
    totaal = sum(i["bytes"] for i in plan if i["reden"] is None)
    gedaan = 0
    geslaagd = []
    for idx, item in enumerate(plan):
        if item["reden"] is not None:
            continue
        tmp = item["doel"] + COPY_PART
        try:
            with open(item["pad"], "rb") as bron, open(tmp, "wb") as doel:
                while True:
                    if stop_check and stop_check():
                        raise CopyAborted()
                    blok = bron.read(COPY_BLOCK)
                    if not blok:
                        break
                    doel.write(blok)
                    gedaan += len(blok)
                    if progress_callback:
                        progress_callback(gedaan, totaal, idx, item["naam"])
            try:
                shutil.copystat(item["pad"], tmp)
            except OSError:
                pass                        # copying the date over is nice-to-have, not required
            os.replace(tmp, item["doel"])
            geslaagd.append(item["doel"])
        except CopyAborted:
            _remove_quietly(tmp)
            raise
        except OSError as e:
            _remove_quietly(tmp)
            item["reden"] = f"mislukt: {e.strerror or e}"
            gedaan += item["bytes"]        # the bar shouldn't stay stuck on this file
            if progress_callback:
                progress_callback(gedaan, totaal, idx, item["naam"])
    return geslaagd


def _remove_quietly(pad):
    try:
        os.remove(pad)
    except OSError:
        pass


def list_source_videos(bieb, extern=False):
    """Every known recording with its work-list data: status, note, how many fragments
    have already been trimmed from it (= analyses with this source), whether the file
    is present, and whether the cloud sync is still in progress.

    `aantal_fragmenten` counts the analyses that come from this recording;
    `aantal_schaatsers` how many different skaters that involves; `aantal_punten` the
    saved points from the manual viewing window. The "fragments vs. analyses"
    distinction from the roadmap collapses here — every marked fragment becomes exactly
    one analysis.

    Every row carries `library` (which library it's in — the GUI shows the shared and the
    local one mixed together and needs to write back to the right one on a change) and
    `extern`: True for a **loose video from this pc** (absolute path,
    `source_video_for_path`), False for a recording from `opnames/`. `extern` filters:
    `False` (default) is the **work list**, `True` only the loose videos, `None` both.
    The default is deliberately the work list: since `local_library`, loose videos no
    longer belong in the shared database, and whatever is still in there from before
    that (`migrate_loose_videos` only takes the rows whose file is on this pc) reads as
    a "file not found" for everyone else."""
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT b.*,"
            "       (SELECT COUNT(*) FROM analysis a WHERE a.source_id = b.id)"
            "       AS aantal_fragmenten,"
            "       (SELECT COUNT(DISTINCT a.skater_id) FROM analysis a"
            "         WHERE a.source_id = b.id) AS aantal_schaatsers,"
            "       (SELECT COUNT(*) FROM source_marking m WHERE m.source_id = b.id)"
            "       AS aantal_punten "
            "FROM source_video b ORDER BY b.name COLLATE NOCASE").fetchall()
    uit = []
    for r in rijen:
        d = _add_legacy_keys(dict(r), bestand="file", naam="name",
                              totaal_frames="total_frames", notitie="note",
                              bijgewerkt_door="updated_by", toegevoegd_op="added_at")
        d["extern"] = _is_external(d["file"])
        if extern is not None and d["extern"] != extern:
            continue
        d["library"] = bieb
        d["pad"] = _abs_path(bieb, d["file"])
        d["sync"] = video_sync_status(d["pad"], d["bytes"])
        uit.append(d)
    return uit


def source_video(bieb, bron_id):
    """One recording row (incl. absolute path + sync status), or KeyError."""
    for b in list_source_videos(bieb, extern=None):
        if b["id"] == bron_id:
            return b
    raise KeyError(f"Recording {bron_id} is not in the library.")


def source_video_for_path(bieb, pad, meta_lezer=None):
    """The row for a **loose video** — somewhere outside `opnames/` — in library
    `bieb`, creating it if it doesn't exist yet. Returns the same dict as
    `list_source_videos`. The GUI doesn't call this directly but through `loose_video`,
    which picks the **local** library; only the row logic lives here.

    Meant for "just watching" (`ViewWindow`): nothing is measured or copied there,
    but the points a trainer sets must still be kept, and those hang off a
    source_video row via `source_marking.source_id`. Hence a real row, with two
    differences from a recording out of `opnames/`:

    - **`bestand` is the absolute path** (forward slashes, normalized with `abspath`).
      That's immediately the trait `_is_external` uses to tell the two kinds apart, so
      no extra column and no migration is needed. Picking the same video again later
      finds the same row again through the UNIQUE on `bestand` — with the points from
      last time.
    - **`bytes` stays NULL.** That size only serves to recognize a half-downloaded
      cloud copy (`video_sync_status`); on a file that simply lives on this pc the
      comparison is meaningless and would backfire — the same video later re-encoded
      smaller would count as "still downloading" and the window wouldn't open.
    """
    sleutel = os.path.abspath(pad).replace("\\", "/")
    grootte = None
    # Whoever browses to the recordings folder in the file picker anyway should get the
    # row the scan already has for it: the same video under two keys would split its points.
    # The rules of a recording then simply apply too (relative path, `bytes` for the sync).
    if _in_recordings_dir(bieb, pad):
        sleutel = f"{RECORDINGS_DIR}/{os.path.basename(pad)}"
        try:
            grootte = os.path.getsize(pad)
        except OSError:
            grootte = None
    with _connect(bieb) as con:
        rij = con.execute("SELECT id FROM source_video WHERE file = ?", (sleutel,)).fetchone()
        if rij is None:
            fps, totaal = (meta_lezer or _video_meta)(pad)
            cur = con.execute(
                "INSERT INTO source_video(file, name, bytes, fps, total_frames, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sleutel, os.path.basename(pad), grootte, fps, totaal, SOURCE_STATUS_DEFAULT))
            bron_id = cur.lastrowid
        else:
            bron_id = rij["id"]
    return source_video(bieb, bron_id)


def _in_recordings_dir(bieb, pad):
    """Does `pad` sit in `<bieb>/opnames/`? Then it's a recording of the team, not a
    loose video."""
    return (os.path.normcase(os.path.dirname(os.path.abspath(pad)))
            == os.path.normcase(os.path.abspath(recordings_path(bieb, maak_aan=False))))


def loose_video(bieb, lokaal, pad, meta_lezer=None):
    """The row for "View new video": in the **local** library, unless the user browsed
    to the shared library's recordings folder in the file picker — then it's simply the
    recording the scan already has for it (the same video under two keys would split
    its points). The dict carries `library`, so the caller doesn't need to know itself
    which of the two it ended up in."""
    doel = bieb if _in_recordings_dir(bieb, pad) else lokaal
    return source_video_for_path(doel, pad, meta_lezer=meta_lezer)


def migrate_loose_videos(bieb, lokaal):
    """One-time cleanup: loose videos that ended up with their absolute path in the
    **shared** database before `local_library` (2026-09-11) move to the local one —
    with their points. Returns the number of rows moved.

    Only rows whose file **is on this pc**: that's the pc that registered them (an
    absolute path is by definition from one machine). A colleague's row stays put until
    THEY update their app and refresh — deleting it here would cost their points, and
    `list_source_videos` already doesn't show it anyway. Whatever never gets picked up
    again (file gone before the owner refreshed) stays invisible in the shared
    database; that's a row of a few hundred bytes, not a problem.

    If the same video is already local (opened once more through the old app after the
    move), the points join that row. Analyses trimmed from such a row lose their
    `source_id` (ON DELETE SET NULL): in the shared database there's then no source
    left to point at. Reads without writing if there's nothing to move — the shared
    database should stay at rest (phase 4)."""
    with _connect(bieb) as con:
        rijen = [dict(r) for r in con.execute("SELECT * FROM source_video").fetchall()]
    te_verhuizen = [r for r in rijen
                    if _is_external(r["file"]) and os.path.isfile(_abs_path(bieb, r["file"]))]
    for r in te_verhuizen:
        with _connect(bieb) as con:
            punten = con.execute(
                "SELECT frame, label, created_by, created_at FROM source_marking "
                "WHERE source_id = ? ORDER BY frame", (r["id"],)).fetchall()
        with _connect(lokaal) as lcon:
            bestaand = lcon.execute("SELECT id FROM source_video WHERE file = ?",
                                    (r["file"],)).fetchone()
            if bestaand is None:
                cur = lcon.execute(
                    "INSERT INTO source_video(file, name, bytes, fps, total_frames, status, "
                    "                      note, updated_by, interlaced, added_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["file"], r["name"], r["bytes"], r["fps"], r["total_frames"],
                     r["status"], r["note"], r["updated_by"], r["interlaced"],
                     r["added_at"]))
                nieuw_id = cur.lastrowid
            else:
                nieuw_id = bestaand["id"]
            lcon.executemany(
                "INSERT INTO source_marking(source_id, frame, label, created_by, "
                "                           created_at) VALUES (?, ?, ?, ?, ?)",
                [(nieuw_id, p["frame"], p["label"], p["created_by"], p["created_at"])
                 for p in punten])
        with _connect(bieb) as con:
            con.execute("DELETE FROM source_video WHERE id = ?", (r["id"],))   # points cascade
    return len(te_verhuizen)


def edit_source_video(bieb, bron_id, status=None, notitie=None, bijgewerkt_door=""):
    """Sets the status and/or note of a recording. `bijgewerkt_door` (the trainer name)
    travels along so in a shared library it's visible who set a recording to 'done' —
    the same motive as `created_by` on an analysis."""
    velden, waarden = [], []
    if status is not None:
        velden.append("status = ?")
        waarden.append(status)
    if notitie is not None:
        velden.append("note = ?")
        waarden.append(notitie)
    if not velden:
        return
    velden.append("updated_by = ?")
    waarden.append(bijgewerkt_door or "")
    with _connect(bieb) as con:
        con.execute(f"UPDATE source_video SET {', '.join(velden)} WHERE id = ?",
                    waarden + [bron_id])


def set_source_interlaced(bieb, bron_id, interlaced):
    """Records whether this recording is interlaced (combing), so it doesn't need to
    be measured again every session. Deliberately a separate function and not a field
    on `edit_source_video`: this is a measured property of the file, not a trainer's
    choice, so `updated_by` should NOT change because of it."""
    with _connect(bieb) as con:
        con.execute("UPDATE source_video SET interlaced = ? WHERE id = ?",
                    (1 if interlaced else 0, bron_id))


def source_fragments(bieb, bron_id):
    """The parts of this recording already analyzed: [{analyse_id, titel, schaatser,
    start_frame, eind_frame}], sorted by start frame. This produces the gray blocks in
    the trim window — a single query instead of scanning every settings_json field."""
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT a.id AS analyse_id, a.title AS titel, s.name AS schaatser,"
            "       a.source_start_frame AS start_frame, a.source_end_frame AS eind_frame "
            "FROM analysis a LEFT JOIN skater s ON s.id = a.skater_id "
            "WHERE a.source_id = ? AND a.source_start_frame IS NOT NULL "
            "ORDER BY a.source_start_frame", (bron_id,)).fetchall()
    return [dict(r) for r in rijen]


# ── Points in a recording (manual viewing window) ───────────────────────────────
# A trainer watching through a recording wants to be able to find a spot again — the
# jump they want to see once more, the moment a run starts. Those are loose frame
# numbers with a little name, and they belong to the **recording**: they survive
# closing the window and (like status and note) are visible to the whole team in the
# shared database. Deliberately not an analysis: nothing is measured, only where you
# were is remembered.

def list_markings(bieb, bron_id):
    """The saved points of a recording, sorted by frame number."""
    with _connect(bieb) as con:
        rijen = con.execute(
            "SELECT * FROM source_marking WHERE source_id = ? ORDER BY frame, id",
            (bron_id,)).fetchall()
    return [_add_legacy_keys(dict(r), bron_id="source_id", aangemaakt_door="created_by",
                              aangemaakt_op="created_at")
            for r in rijen]


def add_marking(bieb, bron_id, frame, label="", aangemaakt_door=""):
    """Sets a point at `frame` in this recording; returns the new id."""
    with _connect(bieb) as con:
        cur = con.execute(
            "INSERT INTO source_marking(source_id, frame, label, created_by) "
            "VALUES (?, ?, ?, ?)",
            (bron_id, int(frame), label or "", aangemaakt_door or ""))
        return cur.lastrowid


def edit_marking(bieb, markering_id, label=None, frame=None):
    """Renames a point and/or moves it to a different frame."""
    velden, waarden = [], []
    if label is not None:
        velden.append("label = ?")
        waarden.append(label)
    if frame is not None:
        velden.append("frame = ?")
        waarden.append(int(frame))
    if not velden:
        return
    with _connect(bieb) as con:
        con.execute(f"UPDATE source_marking SET {', '.join(velden)} WHERE id = ?",
                    waarden + [markering_id])


def delete_marking(bieb, markering_id):
    with _connect(bieb) as con:
        con.execute("DELETE FROM source_marking WHERE id = ?", (markering_id,))


# ── Shared cloud folder (phase 4) ────────────────────────────────────────────────

def detect_conflict_copies(bieb):
    """Looks for conflict copies of the database that a cloud syncer (Google Drive,
    OneDrive, Dropbox) can leave behind when two trainers write at almost the same
    time — e.g. 'schaats-DESKTOP.db', 'schaats (1).db', or 'schaats (conflicted
    copy).db'. Returns the filenames (no path), sorted; empty if everything is fine.

    Deliberately detection, not prevention: the app can't safely merge such copies, but
    warns so the trainer can clean them up by hand. 'schaats.db' itself and the
    short-lived '-journal' (DELETE mode) are skipped.

    Only names that look like our own database count ('schaats....db'): a syncer
    appends its marker after the filename. A random other database someone puts in the
    folder isn't a conflict copy, and would otherwise trigger a warning every time it's
    opened and on every 'Refresh'."""
    try:
        namen = os.listdir(bieb)
    except OSError:
        return []
    hoofd = DB_NAME.lower()
    stam  = os.path.splitext(hoofd)[0]
    kopieen = [n for n in namen
               if n.lower().endswith(".db") and n.lower() != hoofd
               and n.lower().startswith(stam)]
    return sorted(kopieen)


def video_sync_status(video_pad, verwacht_bytes):
    """Sync status of a copied video in a shared cloud folder (phase 4):
    - 'ontbreekt'  : the file isn't on disk (yet);
    - 'onvolledig' : the file is smaller than when it was saved (cloud still downloading);
    - None         : fine, or the expected size is unknown (an analysis from before v2)."""
    if not os.path.isfile(video_pad):
        return "ontbreekt"
    try:
        if verwacht_bytes and os.path.getsize(video_pad) < verwacht_bytes:
            return "onvolledig"
    except OSError:
        return "ontbreekt"
    return None


# Speed probe (see `file_is_local`): size of one read block and the time above which we
# consider the file "not on this pc". Measured 2026-08-24 on this library: from a local
# disk such a block costs 0.3-2 ms, from a Google Drive streaming folder 614 ms. The
# threshold sits comfortably in between, so a slow USB or network drive still counts as
# "local" — it's about the factor of 100, not the exact cutoff.
LOCAL_BLOCK_BYTES = 64 * 1024
LOCAL_THRESHOLD_MS = 100.0


def file_is_local(pad, monsters=3, drempel_ms=LOCAL_THRESHOLD_MS):
    """Does this video file really live on this pc, or is it fetched from the cloud
    piece by piece?

    - 'lokaal' : every sample came back immediately;
    - 'deels'  : some did, some didn't (the cloud folder is still downloading);
    - 'cloud'  : not a single sample came from disk;
    - None     : the file isn't there, or can't be read.

    **Measure, don't ask.** Windows does have an attribute for cloud placeholders
    (FILE_ATTRIBUTE_OFFLINE / RECALL_ON_DATA_ACCESS), but Google Drive for desktop
    doesn't set it: its drive presents itself as an ordinary fixed disk, and a
    not-yet-downloaded 4 GB recording has attribute `Normal`. What you CAN see is the
    time — a block from a spot that isn't local yet costs a network round trip.

    Why this matters: on a streaming folder, every jump in the trim window first has to
    fetch tens of MB (measured: 5-20 s per jump, ~40 MB), while the same jump on a local
    copy costs 30-120 ms. See `skate_gui._recording_available`.

    The samples sit at **random** spots: a piece that's been read then sits in the
    cloud cache, so probing the same spot every time would say 'lokaal' the second time
    around regardless. Cost: negligible if the file is local, otherwise a couple of
    seconds — hence this runs on a background thread in the GUI.
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
                speling = grootte - LOCAL_BLOCK_BYTES
                off = random.randrange(speling) if speling > 0 else 0
                begin = time.perf_counter()
                f.seek(off)
                if not f.read(LOCAL_BLOCK_BYTES):
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


# ── Skeleton editor (phase 3) ─────────────────────────────────────────────────────

def save_edited_landmarks(bieb, analyse_id, resultaten, info, events):
    """
    Overwrites an analysis's landmarks with manually corrected ones (skeleton editor).
    On the FIRST edit, the original landmarks.npz is preserved once as
    landmarks_ruw.npz, so 'restore original' can always go back. Sets analysis.edited =
    1 and refreshes the events cache in one transaction. Runs per drop, so keep it light.
    """
    doelmap = os.path.join(bieb, MEDIA_DIR, analyse_id)
    npz = os.path.join(doelmap, NPZ_NAME)
    ruw = os.path.join(doelmap, NPZ_RAW_NAME)
    if not os.path.isfile(ruw) and os.path.isfile(npz):
        shutil.copy2(npz, ruw)          # pristine original, only on the first edit
    sla_landmarks_op(npz, resultaten, info)
    with _connect(bieb) as con:
        con.execute("UPDATE analysis SET edited = 1 WHERE id = ?", (analyse_id,))
        _write_events_cache(con, analyse_id, events)


def save_corner_marking(bieb, analyse_id, resultaten, info, events):
    """
    Writes a corner marking determined after the fact (`FrameResult.corner`) and
    refreshes the events cache. For analyses from before corner detection existed:
    their npz doesn't have the flag yet, while the landmarks for the whole clip ARE in
    there — so the corner can perfectly well be derived without re-analyzing.

    Deliberately NOT `save_edited_landmarks`: that's for manual work with the skeleton
    editor and sets `analysis.edited = 1` plus a pristine backup. Here not a single
    landmark changes — only the flag that says which frames fall outside the
    measurement — so that analysis stays "not edited" and the eval metrics stay
    comparable with other unedited analyses (see the warning in skate_eval).
    """
    npz = os.path.join(bieb, MEDIA_DIR, analyse_id, NPZ_NAME)
    sla_landmarks_op(npz, resultaten, info)
    with _connect(bieb) as con:
        _write_events_cache(con, analyse_id, events)


def restore_original_landmarks(bieb, analyse_id):
    """Restores the landmarks to before the first edit (landmarks_ruw.npz ->
    landmarks.npz) and analysis.edited = 0. Returns True if there was an original,
    otherwise False (the analysis was never edited). The GUI reloads afterward and
    refreshes the events cache itself."""
    doelmap = os.path.join(bieb, MEDIA_DIR, analyse_id)
    ruw = os.path.join(doelmap, NPZ_RAW_NAME)
    if not os.path.isfile(ruw):
        return False
    shutil.copy2(ruw, os.path.join(doelmap, NPZ_NAME))
    with _connect(bieb) as con:
        con.execute("UPDATE analysis SET edited = 0 WHERE id = ?", (analyse_id,))
    return True


def refresh_events_cache(bieb, analyse_id, events):
    """Rewrites just the events cache (after a recompute, e.g. after restoring the original)."""
    with _connect(bieb) as con:
        _write_events_cache(con, analyse_id, events)


def rename_analysis(bieb, analyse_id, titel):
    with _connect(bieb) as con:
        con.execute("UPDATE analysis SET title = ? WHERE id = ?", (titel, analyse_id))


def delete_analysis(bieb, analyse_id):
    """Deletes the DB row (cascade clears the events cache) + media folder."""
    with _connect(bieb) as con:
        con.execute("DELETE FROM analysis WHERE id = ?", (analyse_id,))
    shutil.rmtree(os.path.join(bieb, MEDIA_DIR, analyse_id), ignore_errors=True)


# ── Transitional Dutch-name aliases ──────────────────────────────────────────────
# schaats_gui.py/schaats_schermtest.py aren't translated yet (see the
# translate-to-english plan) and do a bare `import schaats_db`, then call
# `schaats_db.maak_schaatser(...)` etc. throughout — 75+ call sites in schaats_gui.py
# alone, far too many to update from this phase. `schaats_db.py` itself now only
# exists as a tiny compatibility shim (`from skate_db import *`) so that bare import
# keeps resolving; these aliases are what makes every individual Dutch name it's still
# calling actually available on this module for that shim to re-export. Remove each
# alias (and eventually the shim file) once schaats_gui.py/schaats_schermtest.py are
# translated and import skate_db directly under its real names.
BibliotheekTeNieuw = LibraryTooNew
KopieerAfgebroken = CopyAborted
BRON_STATUSSEN = SOURCE_STATUSES
BRON_STATUS_DEFAULT = SOURCE_STATUS_DEFAULT
DB_NAAM = DB_NAME
MEDIA_MAP = MEDIA_DIR
OPNAMES_MAP = RECORDINGS_DIR
NPZ_NAAM = NPZ_NAME
NPZ_RUW_NAAM = NPZ_RAW_NAME
ONVOLLEDIG_MARKERS = INCOMPLETE_MARKERS
AFGEKAPT_MARKER = TRUNCATED_MARKER
ENV_BIBLIOTHEEK = ENV_LIBRARY
ENV_LOKAAL = ENV_LOCAL
LOKAAL_MAP = LOCAL_DIR
KOPIEER_BLOK = COPY_BLOCK
KOPIEER_DEEL = COPY_PART
LOKAAL_BLOK_BYTES = LOCAL_BLOCK_BYTES
LOKAAL_DREMPEL_MS = LOCAL_THRESHOLD_MS
config_pad = config_path
standaard_bibliotheek = default_library
lokale_bibliotheek = local_library
laad_config = load_config
bewaar_config = save_config
bibliotheek_pad = library_path
trainer_naam = trainer_name
app_versie = app_version
maak_schaatser = create_skater
wijzig_schaatser = edit_skater
lijst_schaatsers = list_skaters
verwijder_schaatser = delete_skater
lijst_analyses = list_analyses
lijst_kalibraties = list_calibrations
sla_analyse_op = save_analysis
analyse_meta = analysis_meta
laad_analyse = load_analysis
analyse_video_pad = analysis_video_path
opnames_pad = recordings_path
synchroniseer_bronmap = sync_source_dir
lijst_bronvideos = list_source_videos
bronvideo = source_video
bronvideo_voor_pad = source_video_for_path
losse_video = loose_video
verhuis_losse_videos = migrate_loose_videos
wijzig_bronvideo = edit_source_video
zet_bron_interlaced = set_source_interlaced
bron_fragmenten = source_fragments
lijst_markeringen = list_markings
voeg_markering_toe = add_marking
wijzig_markering = edit_marking
verwijder_markering = delete_marking
detecteer_conflictkopieen = detect_conflict_copies
bestand_lokaal = file_is_local
bewaar_bewerkte_landmarks = save_edited_landmarks
bewaar_bochtmarkering = save_corner_marking
herstel_originele_landmarks = restore_original_landmarks
ververs_events_cache = refresh_events_cache
hernoem_analyse = rename_analysis
verwijder_analyse = delete_analysis
kopieer_plan = copy_plan
kopieer_naar_opnames = copy_to_recordings


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from skate_analysis import arrays_naar_resultaten, resultaten_naar_arrays, AfzetEvent

    tmp = tempfile.mkdtemp(prefix="skate_db_test_")
    try:
        bieb = os.path.join(tmp, "bieb")
        open_db(bieb)
        open_db(bieb)   # idempotent
        with _connect(bieb) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []

        # Migration v1 -> v6: an old DB without video_bytes (v2), without
        # source_video/analysis.source_* (v3), without source_marking (v4), and still
        # on the Dutch v5 schema (v6) gets updated in one go without data loss. The old
        # schema is deliberately literal here: v1 IS frozen, and filtering it out of
        # _SCHEMA gets more fragile with every bump.
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

        def _kolommen(c, tabel):
            return [r[1] for r in c.execute(f"PRAGMA table_info({tabel})")]

        oud = os.path.join(tmp, "oud_v1")
        os.makedirs(os.path.join(oud, MEDIA_DIR))
        with sqlite3.connect(os.path.join(oud, DB_NAME)) as c:
            c.executescript(v1_schema)
            c.execute("PRAGMA user_version = 1")
        open_db(oud)   # migrates all the way to v6
        with _connect(oud) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []
            kolommen = _kolommen(c, "analysis")
            assert "video_bytes" in kolommen
            assert {"source_id", "source_start_frame", "source_end_frame"} <= set(kolommen)
            assert "settings_json" in kolommen and "instellingen_json" not in kolommen
            assert c.execute("SELECT COUNT(*) FROM source_video").fetchone()[0] == 0
            assert c.execute("SELECT COUNT(*) FROM source_marking").fetchone()[0] == 0
            # A database migrated all the way from v1 must be indistinguishable in
            # shape from a freshly created v6 one -- same tables, same columns. Column
            # ORDER can legitimately differ (ALTER TABLE ADD COLUMN always appends at
            # the end, so a column added by an old migration step sits earlier than it
            # would in a fresh CREATE TABLE) -- this codebase always addresses columns
            # by name, never by position, in every query, so only the column SET is
            # the actual invariant worth asserting here.
            fresh = os.path.join(tmp, "fresh_v6")
            open_db(fresh)
            with _connect(fresh) as fc:
                for tabel in ("skater", "source_video", "source_marking", "analysis",
                              "push_event_cache"):
                    assert set(_kolommen(c, tabel)) == set(_kolommen(fc, tabel)), tabel

        # Migration v2 -> v3 separately: only the phase 8 step, without the v2 step before it.
        oud2 = os.path.join(tmp, "oud_v2")
        os.makedirs(os.path.join(oud2, MEDIA_DIR))
        with sqlite3.connect(os.path.join(oud2, DB_NAME)) as c:
            c.executescript(v1_schema)
            c.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")   # = v2
            c.execute("PRAGMA user_version = 2")
        open_db(oud2)
        with _connect(oud2) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert "source_end_frame" in _kolommen(c, "analysis")

        # Migration v3 -> v4 separately: a library only missing the points table, whose
        # existing recording rows must stay untouched.
        oud3 = os.path.join(tmp, "oud_v3")
        os.makedirs(os.path.join(oud3, MEDIA_DIR))
        with sqlite3.connect(os.path.join(oud3, DB_NAME)) as c:
            c.executescript(v1_schema)
            c.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")   # = v2
            c.execute(_OLD_BRONVIDEO_DDL)                                    # = v3
            for kolom in ("bron_id INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL",
                          "bron_start_frame INTEGER", "bron_eind_frame INTEGER"):
                c.execute(f"ALTER TABLE analyse ADD COLUMN {kolom}")
            c.execute("INSERT INTO bronvideo(bestand, naam) VALUES ('opnames/x.mp4', 'x.mp4')")
            c.execute("PRAGMA user_version = 3")
        open_db(oud3)
        with _connect(oud3) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert c.execute("SELECT COUNT(*) FROM source_marking").fetchone()[0] == 0
            assert c.execute("SELECT name FROM source_video").fetchone()[0] == "x.mp4"
            assert _has_column(c, "source_video", "interlaced")
            assert c.execute("SELECT interlaced FROM source_video").fetchone()[0] is None

        # A v5 database (the Dutch schema, one release before this rename) with a real
        # analysis: the v5->v6 step must rename everything, remap the status enum, and
        # recompute the event cache so `note` reads the new marker text.
        oud5 = os.path.join(tmp, "oud_v5")
        os.makedirs(os.path.join(oud5, MEDIA_DIR))
        with sqlite3.connect(os.path.join(oud5, DB_NAME)) as c:
            c.executescript(v1_schema)
            c.execute("ALTER TABLE analyse ADD COLUMN video_bytes INTEGER")
            c.execute(_OLD_BRONVIDEO_DDL)
            for kolom in ("bron_id INTEGER REFERENCES bronvideo(id) ON DELETE SET NULL",
                          "bron_start_frame INTEGER", "bron_eind_frame INTEGER"):
                c.execute(f"ALTER TABLE analyse ADD COLUMN {kolom}")
            c.execute(_OLD_BRON_MARKERING_DDL)
            c.execute(
                "INSERT INTO bronvideo(bestand, naam, status) "
                "VALUES ('opnames/oud.mp4', 'oud.mp4', 'klaar')")
            # A skater must exist first so the analyse row's foreign key holds.
            c.execute("INSERT INTO schaatser(naam) VALUES ('Oude Schaatser')")
            sid_v5 = c.execute("SELECT id FROM schaatser").fetchone()[0]
            oude_instellingen = json.dumps({
                "smooth_n": 5, "bocht_overslaan": True, "perspectief_gebruikt": True,
                "perspectief": {"invoer": {"beeld_w": 64}}, "backend_naam": "YOLO-pose + ByteTrack",
                "app_versie": "2026-01-01 - oud1234",
                # doel_punt/doel_kader must survive the migration UNCHANGED -- they
                # stay Dutch permanently (see step 4's comment in _migrate).
                "doel_punt": [0.5, 0.5]})
            c.execute(
                "INSERT INTO analyse(id, schaatser_id, titel, datum, video_bestand, "
                "w, h, fps, totaal_frames, backend, instellingen_json) VALUES "
                "('a-v5', ?, 'Oude analyse', '2026-01-01', 'media/a-v5/v.mp4', "
                "64, 48, 25.0, 20, 'yolo', ?)", (sid_v5, oude_instellingen))
            c.execute(
                "INSERT INTO afzet_event_cache(analyse_id, idx, been, start_frame, "
                "eind_frame, hoek, min_hoek, max_hoek, opmerking) VALUES "
                "('a-v5', 0, 'links', 0, 19, 79.0, 60.0, 80.0, 'afgekapt')")
            c.execute("PRAGMA user_version = 5")
        # No npz on disk for 'a-v5' -- the recompute must fall back to a text remap
        # for this one analysis instead of raising and aborting the whole migration.
        open_db(oud5)
        with _connect(oud5) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []
            assert c.execute("SELECT status FROM source_video").fetchone()[0] == "done"
            row = c.execute(
                "SELECT leg, note FROM push_event_cache WHERE analysis_id = 'a-v5'"
            ).fetchone()
            assert row["leg"] == "links"          # leg VALUES stay Dutch for now, by design
            assert row["note"] == "truncated"      # marker TEXT is remapped even on fallback
            # Step 4: settings_json content keys renamed in place, doel_punt untouched.
            inst_v5 = json.loads(
                c.execute("SELECT settings_json FROM analysis WHERE id = 'a-v5'"
                         ).fetchone()[0])
            assert inst_v5["skip_corner"] is True and "bocht_overslaan" not in inst_v5
            assert inst_v5["perspective_used"] is True and "perspectief_gebruikt" not in inst_v5
            assert inst_v5["perspective"] == {"invoer": {"beeld_w": 64}} and "perspectief" not in inst_v5
            assert inst_v5["backend_name"] == "YOLO-pose + ByteTrack" and "backend_naam" not in inst_v5
            assert inst_v5["app_version"] == "2026-01-01 - oud1234" and "app_versie" not in inst_v5
            assert inst_v5["doel_punt"] == [0.5, 0.5]

        # Newer DB (a colleague with a more recent app): refuse, don't downgrade.
        nieuw = os.path.join(tmp, "nieuw_v99")
        os.makedirs(os.path.join(nieuw, MEDIA_DIR))
        with sqlite3.connect(os.path.join(nieuw, DB_NAME)) as c:
            c.executescript(_SCHEMA)
            c.execute("ALTER TABLE analysis ADD COLUMN iets_nieuws TEXT")
            c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        try:
            open_db(nieuw)
            raise AssertionError("a newer library should have been refused")
        except LibraryTooNew:
            pass
        with _connect(nieuw) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 1
            assert "iets_nieuws" in _kolommen(c, "analysis")

        # Env-var override for the library path.
        os.environ[ENV_LIBRARY] = bieb
        assert library_path() == bieb
        del os.environ[ENV_LIBRARY]

        # config.json: the one-time "SchaatsAnalyse" -> "SkateAnalysis" folder copy
        # (config_path) and the old-key -> new-key dual read (load_config), both
        # added together with the phase-8 settings-key rename. A trainer upgrading
        # from before this change has a config.json with the old folder AND the old
        # keys; both must still resolve to the value they saved.
        oud_appdata = os.environ.get("APPDATA")
        appdata_tmp = os.path.join(tmp, "appdata_migratie")
        os.environ["APPDATA"] = appdata_tmp
        try:
            oude_map = os.path.join(appdata_tmp, "SchaatsAnalyse")
            os.makedirs(oude_map)
            with open(os.path.join(oude_map, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"bibliotheek_pad": "D:/Oude Bieb", "trainer_naam": "Oude Trainer"}, f)
            pad = config_path()   # should copy the old file into the new folder
            assert os.path.isfile(pad) and "SkateAnalysis" in pad
            assert os.path.isfile(os.path.join(oude_map, "config.json")), \
                "the old file should be copied, not moved"
            cfg = load_config()
            assert cfg["library_path"] == "D:/Oude Bieb" and cfg["trainer_name"] == "Oude Trainer"
            assert library_path() == "D:/Oude Bieb" and trainer_name() == "Oude Trainer"
            # A second call must not re-copy over a config.json that already exists
            # (e.g. one already updated by a newer save_config() call).
            with open(pad, "w", encoding="utf-8") as f:
                json.dump({"library_path": "D:/Nieuwe Bieb"}, f)
            assert config_path() == pad
            assert load_config()["library_path"] == "D:/Nieuwe Bieb"
            # A brand-new config.json (no old folder at all) just gets the defaults --
            # no crash from an absent SchaatsAnalyse folder.
            os.remove(pad)
            os.rmdir(os.path.dirname(pad))
            shutil.rmtree(oude_map)
            cfg = load_config()
            assert cfg["library_path"] == default_library() and cfg["trainer_name"] == ""
        finally:
            if oud_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = oud_appdata

        # Synthetic analysis (phase 0 serialization form) + dummy video file.
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
                       note="gemiste tegenafzet?"),
            # Incomplete: count toward the total, but not toward the average angle —
            # otherwise those far-too-steep angles would drag the average up. Both
            # reasons, since `list_analyses` must filter out both.
            AfzetEvent(2, "links",  13, 19, 0.52, 0.76, 79.0, 60.0, 80.0,
                       incomplete=ONV_AFGEKAPT),
            AfzetEvent(3, "rechts", 20, 26, 0.80, 1.04, 83.0, 74.0, 88.0,
                       incomplete=ONV_GEEN_PUSH),
        ]
        video = os.path.join(tmp, "Testvideo.mp4")
        with open(video, "wb") as f:
            f.write(b"geen echte video, copy2 kijkt niet naar de inhoud")

        sid = create_skater(bieb, "Test Schaatser", 2010, "proefkonijn")
        edit_skater(bieb, sid, "Test Schaatser", 2011, "proefkonijn 2.0")
        s = list_skaters(bieb)
        assert len(s) == 1 and s[0]["birth_year"] == 2011 and s[0]["aantal_analyses"] == 0

        # App version: outside a git repo the fields are empty — so only check the
        # shape, not the content.
        versie = app_version()
        assert set(versie) == {"commit", "datum", "vuil", "label"}
        assert bool(versie["label"]) == bool(versie["commit"])

        # `backend_name` deliberately left out: save_analysis should fill it in itself.
        # `doel_punt`/`doel_kader` stay Dutch on purpose (settings-json storage keys
        # that mirror analyze()'s own permanently-Dutch keyword arguments -- see
        # TRANSLATION_PROGRESS.md's glossary entry for `doel_kader`); every other key
        # here is English since this session's phase-8 settings-key rename.
        instellingen = {"smooth_n": 5, "threshold": 0.015, "smooth_landmarks": True,
                        "skip_corner": True,
                        "doel_punt": [0.5, 0.5], "doel_kader": [0.45, 0.4, 0.55, 0.6],
                        "horizon_deg": 0.0, "auto_horizon": False,
                        "heavy": False, "perspective_used": False}
        aid = save_analysis(bieb, sid, "Proefanalyse", video, info, resultaten, events,
                             backend="YOLO-pose + ByteTrack", instellingen=instellingen,
                             aangemaakt_door="Coach Tester")
        assert os.path.isfile(os.path.join(bieb, MEDIA_DIR, aid, "Testvideo.mp4"))
        assert os.path.isfile(os.path.join(bieb, MEDIA_DIR, aid, NPZ_NAME))

        la = list_analyses(bieb, sid)
        assert len(la) == 1 and la[0]["aantal_afzetten"] == 4
        # average over (40.0, 42.0); the incomplete 79.0 and 83.0 fall outside it
        assert abs(la[0]["gem_hoek"] - 41.0) < 1e-9 and la[0]["backend"] == "yolo"
        assert la[0]["created_by"] == "Coach Tester"
        # The list view carries the settings along (for the app-version tooltip).
        assert la[0]["instellingen"]["app_version"] == versie["label"]
        assert list_skaters(bieb)[0]["aantal_analyses"] == 1

        # analysis_meta = the same meta without reading the npz.
        m = analysis_meta(bieb, aid)
        assert m["titel"] == "Proefanalyse" and m["instellingen"]["smooth_n"] == 5
        try:
            analysis_meta(bieb, "bestaat-niet")
            raise AssertionError("analysis_meta should raise KeyError")
        except KeyError:
            pass

        data = load_analysis(bieb, aid)
        assert data["meta"]["titel"] == "Proefanalyse"
        # What the caller supplied is stored unchanged; save_analysis itself added the
        # app version and the full backend name.
        opgeslagen = data["meta"]["instellingen"]
        assert all(opgeslagen[k] == v for k, v in instellingen.items())
        assert opgeslagen["backend_name"] == "YOLO-pose + ByteTrack"
        assert opgeslagen["app_version"] == versie["label"]
        assert opgeslagen["app_commit"] == versie["commit"]
        assert data["meta"]["created_by"] == "Coach Tester"
        assert data["meta"]["video_bytes"] == os.path.getsize(video)
        assert os.path.isfile(data["video_pad"])
        assert data["video_pad"] == analysis_video_path(bieb, aid)

        # `list_calibrations`: a real calibration through the real PerspectiveConfig/
        # CalibrationInput round-trip (not a hand-built dict), since indexing the
        # nested calibration-input dict by hand is exactly what the real pre-existing
        # bug this function had did wrong (see its docstring). Also covers the
        # `perspective`/`perspectief` top-level dual-read together in one library.
        # On its own skater + cleaned up via delete_skater afterwards, so it doesn't
        # perturb the analysis counts the rest of this self-test asserts against `sid`.
        from skate_analysis import PerspectiveConfig
        from skate_perspective import CalibrationInput, _SynthCamera, _scene_lines
        sid_persp = create_skater(bieb, "Perspectief Tester")
        # Same pose as skate_perspective.py's own serialization self-test -- known to
        # produce a well-conditioned (non-degenerate) calibration.
        cam = _SynthCamera("test", C=(-8, -14, 2.5), target=(6, 8, 0), roll_deg=0.0, f=1400)
        track, cross = _scene_lines(cam)
        inv = CalibrationInput(track_lines=track, cross_lines=cross,
                               image_w=cam.w, image_h=cam.h, note="kalibratie-notitie")
        cfg = PerspectiveConfig(calibration=inv.calibrate(), calibration_input=inv,
                                lower_leg_l=0.44)
        instellingen_persp = dict(instellingen, perspective=cfg.to_dict())
        aid_p = save_analysis(bieb, sid_persp, "Met perspectief", video, info, resultaten,
                              events, backend="YOLO-pose + ByteTrack",
                              instellingen=instellingen_persp)
        # A second analysis using the OLD top-level key + the old nested key spelling
        # (as if saved before this session's rename), to confirm the dual-read still
        # resolves both at once, not just whichever one is currently written.
        def _line_to_list(line):
            return [[float(p[0]), float(p[1])] for p in line]
        oud_persp = {"invoer": {"rijlijnen": [_line_to_list(l) for l in track],
                                "dwarslijnen": [_line_to_list(l) for l in cross],
                                "beeld_w": cam.w, "beeld_h": cam.h,
                                "lijnafstand": 4.0, "notitie": "oude stijl"}}
        aid_oud = save_analysis(bieb, sid_persp, "Oude stijl", video, info, resultaten,
                                events, backend="YOLO-pose + ByteTrack",
                                instellingen=dict(instellingen, perspectief=oud_persp))

        kals = list_calibrations(bieb, beeld_w=cam.w, beeld_h=cam.h)
        gevonden = {k["id"]: k for k in kals}
        assert aid_p in gevonden and aid_oud in gevonden
        assert gevonden[aid_p]["note"] == "kalibratie-notitie"
        assert gevonden[aid_p]["w"] == cam.w and gevonden[aid_p]["h"] == cam.h
        assert gevonden[aid_p]["perspective"] == cfg.to_dict()
        assert gevonden[aid_oud]["note"] == "oude stijl"
        # A different image size must exclude both (the lines are in pixels).
        assert list_calibrations(bieb, beeld_w=cam.w + 1, beeld_h=cam.h) == []
        # `aid` (the very first analysis, no calibration at all) is never returned.
        assert aid not in gevonden
        delete_skater(bieb, sid_persp)   # cascade: both analyses + their media folders

        # Cloud-sync check (phase 4): size matches -> None; smaller -> incomplete; gone -> missing.
        assert video_sync_status(data["video_pad"], data["meta"]["video_bytes"]) is None
        assert video_sync_status(data["video_pad"], data["meta"]["video_bytes"] + 999) == "onvolledig"
        assert video_sync_status(os.path.join(tmp, "weg.mp4"), 100) == "ontbreekt"

        # Speed probe: a file in a temp folder is local by definition, and a file that
        # doesn't exist returns None (no exception into the GUI). The cloud side can't
        # be simulated without a real cloud folder — that was measured by hand, see the docstring.
        assert file_is_local(data["video_pad"]) == "lokaal"
        assert file_is_local(os.path.join(tmp, "weg.mp4")) is None

        # ── Recordings / source videos (phase 8) ──────────────────────────────────
        # The folder is the truth about which files exist: scanning produces the rows,
        # and a second scan with no new files writes nothing (rest for the syncer).
        assert os.path.isdir(recordings_path(bieb))
        assert list_source_videos(bieb) == []
        opname = os.path.join(recordings_path(bieb), "Training 3 aug.mp4")
        with open(opname, "wb") as f:
            f.write(b"nep-opname van een half uur")
        with open(os.path.join(recordings_path(bieb), "aantekeningen.txt"), "w") as f:
            f.write("geen video, hoort niet in de lijst")
        # meta_lezer injected: the fake file isn't a real video.
        assert sync_source_dir(bieb, meta_lezer=lambda p: (30.0, 54000)) == 1
        assert sync_source_dir(bieb, meta_lezer=lambda p: (30.0, 54000)) == 0  # idempotent
        bronnen = list_source_videos(bieb)
        assert len(bronnen) == 1                     # the .txt file doesn't count
        bron = bronnen[0]
        assert bron["bestand"] == f"{RECORDINGS_DIR}/Training 3 aug.mp4"
        assert bron["total_frames"] == 54000 and bron["status"] == SOURCE_STATUS_DEFAULT
        assert bron["sync"] is None and bron["aantal_fragmenten"] == 0
        assert os.path.isfile(bron["pad"])

        edit_source_video(bieb, bron["id"], status="in_progress", notitie="tempo-serie",
                         bijgewerkt_door="Coach Tester")
        bron = source_video(bieb, bron["id"])
        assert bron["status"] == "in_progress" and bron["note"] == "tempo-serie"
        assert bron["updated_by"] == "Coach Tester"

        assert bron["interlaced"] is None            # not measured yet
        set_source_interlaced(bieb, bron["id"], True)
        bron = source_video(bieb, bron["id"])
        assert bron["interlaced"] == 1
        assert bron["updated_by"] == "Coach Tester"   # measured, not changed
        set_source_interlaced(bieb, bron["id"], False)
        assert source_video(bieb, bron["id"])["interlaced"] == 0

        # An analysis trimmed from a part of this recording -> gray blocks.
        assert source_fragments(bieb, bron["id"]) == []
        aid_f = save_analysis(bieb, sid, "Fragment 1", video, info, resultaten, events,
                               "MediaPipe", {}, bron_id=bron["id"],
                               bron_start_frame=1200, bron_eind_frame=1560)
        frag = source_fragments(bieb, bron["id"])
        assert len(frag) == 1 and frag[0]["start_frame"] == 1200
        assert frag[0]["eind_frame"] == 1560 and frag[0]["schaatser"] == "Test Schaatser"
        assert list_source_videos(bieb)[0]["aantal_fragmenten"] == 1
        assert analysis_meta(bieb, aid_f)["source_id"] == bron["id"]
        # A loose clip keeps source_id NULL — nothing to guess.
        assert analysis_meta(bieb, aid)["source_id"] is None
        delete_analysis(bieb, aid_f)
        assert source_fragments(bieb, bron["id"]) == []

        # Points from the manual viewing window: save, rename, move, delete.
        assert list_markings(bieb, bron["id"]) == []
        assert list_source_videos(bieb)[0]["aantal_punten"] == 0
        p2 = add_marking(bieb, bron["id"], 900, "sprong", "Coach Tester")
        p1 = add_marking(bieb, bron["id"], 300, "start serie")
        punten = list_markings(bieb, bron["id"])
        assert [p["frame"] for p in punten] == [300, 900]        # sorted by frame
        assert punten[1]["id"] == p2 and punten[1]["label"] == "sprong"
        assert punten[1]["created_by"] == "Coach Tester"
        assert list_source_videos(bieb)[0]["aantal_punten"] == 2
        edit_marking(bieb, p1, label="warming-up", frame=250)
        punten = list_markings(bieb, bron["id"])
        assert punten[0]["frame"] == 250 and punten[0]["label"] == "warming-up"
        edit_marking(bieb, p1)                               # nothing given = nothing done
        assert list_markings(bieb, bron["id"])[0]["frame"] == 250
        delete_marking(bieb, p1)
        assert [p["id"] for p in list_markings(bieb, bron["id"])] == [p2]
        # The foreign key really is enforced (PRAGMA foreign_keys=ON in _connect).
        try:
            add_marking(bieb, 999999, 10, "nergens bij")
            raise AssertionError("a point on an unknown recording should have failed")
        except sqlite3.IntegrityError:
            pass
        delete_marking(bieb, p2)

        # The phase 4 sync check works one-to-one on a recording still coming in.
        with open(bron["pad"], "wb") as f:
            f.write(b"half")
        assert list_source_videos(bieb)[0]["sync"] == "onvolledig"
        os.remove(bron["pad"])
        assert list_source_videos(bieb)[0]["sync"] == "ontbreekt"   # row stays

        # ── Loose video from this pc (viewing only) ───────────────────────────────
        # Outside the library, so with an absolute path — and that path belongs not in
        # the shared database but in the **local** one (`local_library`): `loose_video`
        # puts it there, with the points, and the shared work list stays untouched.
        lokaal = os.path.join(tmp, "lokaal")
        os.environ[ENV_LOCAL] = lokaal
        assert local_library() == lokaal and os.path.isfile(os.path.join(lokaal, DB_NAME))
        assert all(b["extern"] is False and b["library"] == bieb for b in list_source_videos(bieb))
        los = os.path.join(tmp, "van de camera.mp4")
        with open(los, "wb") as f:
            f.write(b"nep-video ergens op de laptop")
        lb = loose_video(bieb, lokaal, los, meta_lezer=lambda p: (25.0, 1500))
        assert lb["library"] == lokaal and lb["extern"] is True
        assert os.path.isabs(lb["bestand"]) and "/" in lb["bestand"]
        assert os.path.normpath(lb["pad"]) == os.path.normpath(los)
        assert lb["bytes"] is None and lb["sync"] is None   # no cloud size check
        assert lb["naam"] == "van de camera.mp4" and lb["total_frames"] == 1500
        # Same video a second time: the same row, not a second one.
        assert loose_video(bieb, lokaal, los)["id"] == lb["id"]
        # Shared: still only the recording; local: only the loose video.
        assert [b["id"] for b in list_source_videos(bieb, extern=None)] == [bron["id"]]
        assert [b["id"] for b in list_source_videos(lokaal, extern=None)] == [lb["id"]]
        assert source_video(lokaal, lb["id"])["id"] == lb["id"]
        # The points hang off it just like a recording and survive reopening.
        add_marking(lokaal, lb["id"], 42, "mooie afzet", "Coach Tester")
        assert loose_video(bieb, lokaal, los)["aantal_punten"] == 1
        assert list_markings(lokaal, lb["id"])[0]["label"] == "mooie afzet"
        # A smaller re-encoded file must NOT count as 'incomplete' (bytes is NULL).
        with open(los, "wb") as f:
            f.write(b"kort")
        assert loose_video(bieb, lokaal, los)["sync"] is None
        # And the comb-filter answer is remembered here just as well.
        set_source_interlaced(lokaal, lb["id"], True)
        assert loose_video(bieb, lokaal, los)["interlaced"] == 1
        # Whoever browses to the shared library's recordings folder in the file picker
        # gets the existing (shared) row, not a local one with an absolute path — that
        # would split that recording's points.
        zelfde = loose_video(bieb, lokaal, os.path.join(recordings_path(bieb),
                                                        "Training 3 aug.mp4"))
        assert zelfde["id"] == bron["id"] and zelfde["library"] == bieb
        assert zelfde["bestand"] == f"{RECORDINGS_DIR}/Training 3 aug.mp4"
        assert len(list_source_videos(lokaal, extern=None)) == 1   # nothing added locally

        # Migration of rows the old app still put in the shared database: only the ones
        # whose file is here, with points; the rest (a colleague's) stays put but
        # outside the work list.
        oud = os.path.join(tmp, "oude losse.mp4")
        with open(oud, "wb") as f:
            f.write(b"stond al in de drive")
        ob = source_video_for_path(bieb, oud, meta_lezer=lambda p: (30.0, 900))
        add_marking(bieb, ob["id"], 7, "start", "Coach Tester")
        add_marking(bieb, ob["id"], 99, "eind", "Coach Tester")
        cb = source_video_for_path(bieb, os.path.join(tmp, "van collega.mp4"),
                                meta_lezer=lambda p: (30.0, 10))   # file doesn't exist here
        assert [b["id"] for b in list_source_videos(bieb)] == [bron["id"]]   # both hidden
        assert migrate_loose_videos(bieb, lokaal) == 1
        assert migrate_loose_videos(bieb, lokaal) == 0                     # idempotent
        gedeeld = {b["id"] for b in list_source_videos(bieb, extern=None)}
        assert gedeeld == {bron["id"], cb["id"]}                            # ob is gone
        verhuisd = loose_video(bieb, lokaal, oud)
        assert verhuisd["library"] == lokaal and verhuisd["total_frames"] == 900
        assert [(m["frame"], m["label"]) for m in list_markings(lokaal, verhuisd["id"])] \
            == [(7, "start"), (99, "eind")]
        assert len(list_source_videos(lokaal, extern=None)) == 2
        del os.environ[ENV_LOCAL]

        # Conflict-copy detection (phase 4): a second .db file gets reported.
        assert detect_conflict_copies(bieb) == []
        with open(os.path.join(bieb, "schaats-LAPTOP.db"), "wb") as f:
            f.write(b"nep-conflictkopie")
        assert detect_conflict_copies(bieb) == ["schaats-LAPTOP.db"]
        # ... but a random other database in the folder is NOT a conflict copy.
        with open(os.path.join(bieb, "adressen.db"), "wb") as f:
            f.write(b"iets heel anders")
        assert detect_conflict_copies(bieb) == ["schaats-LAPTOP.db"]
        os.remove(os.path.join(bieb, "adressen.db"))
        os.remove(os.path.join(bieb, "schaats-LAPTOP.db"))
        terug = resultaten_naar_arrays(data["resultaten"], data["info"])
        assert np.array_equal(terug["landmarks"], arrays["landmarks"])

        rename_analysis(bieb, aid, "Hernoemd")
        assert list_analyses(bieb, sid)[0]["titel"] == "Hernoemd"

        # Skeleton editor (phase 3): edit -> raw backup appears, npz changes, edited=1,
        # cache refreshed; restore original -> npz == original, edited=0.
        map_a = os.path.join(bieb, MEDIA_DIR, aid)
        assert not os.path.isfile(os.path.join(map_a, NPZ_RAW_NAME))   # not edited yet
        bewerkt_res = list(data["resultaten"])
        r0 = bewerkt_res[0]
        r0.lm[26] = r0.lm[26]._replace(x=0.123, y=0.456, visibility=1.0)  # r_knee moved
        bewerkte_events = [AfzetEvent(0, "links", 0, 5, 0.00, 0.20, 30.0, 28.0, 33.0)]
        save_edited_landmarks(bieb, aid, bewerkt_res, data["info"], bewerkte_events)
        assert os.path.isfile(os.path.join(map_a, NPZ_RAW_NAME))
        with np.load(os.path.join(map_a, NPZ_NAME)) as gew:
            assert abs(float(gew["landmarks"][0, 26, 0]) - 0.123) < 1e-6
        with np.load(os.path.join(map_a, NPZ_RAW_NAME)) as orig:
            assert np.array_equal(orig["landmarks"], arrays["landmarks"])
        la_b = list_analyses(bieb, sid)
        assert la_b[0]["edited"] == 1 and la_b[0]["aantal_afzetten"] == 1

        assert restore_original_landmarks(bieb, aid) is True
        with np.load(os.path.join(map_a, NPZ_NAME)) as hersteld:
            assert np.array_equal(hersteld["landmarks"], arrays["landmarks"])
        refresh_events_cache(bieb, aid, events)   # as the GUI does after restoring
        la_h = list_analyses(bieb, sid)
        assert la_h[0]["edited"] == 0 and la_h[0]["aantal_afzetten"] == 4
        assert abs(la_h[0]["gem_hoek"] - 41.0) < 1e-9   # both incomplete reasons filtered out again

        # Error injection: unreadable video -> no DB row, no (extra) media folder.
        try:
            save_analysis(bieb, sid, "Kapot", os.path.join(tmp, "bestaat_niet.mp4"),
                           info, resultaten, events, "MediaPipe", {})
            raise AssertionError("save_analysis should have failed")
        except OSError:
            pass
        assert len(list_analyses(bieb, sid)) == 1
        assert os.listdir(os.path.join(bieb, MEDIA_DIR)) == [aid]

        delete_analysis(bieb, aid)
        assert list_analyses(bieb, sid) == []
        assert not os.path.isdir(os.path.join(bieb, MEDIA_DIR, aid))

        # Cascade: skater gone -> analyses + events cache + media folders gone.
        aid2 = save_analysis(bieb, sid, "Nog een", video, info, resultaten, events,
                              "MediaPipe", {})
        delete_skater(bieb, sid)
        assert list_skaters(bieb) == []
        assert not os.path.isdir(os.path.join(bieb, MEDIA_DIR, aid2))
        with _connect(bieb) as con:
            assert con.execute("SELECT COUNT(*) FROM push_event_cache").fetchone()[0] == 0

        # Copying from the camera to opnames/: plan (what will/won't and why), a
        # block-by-block copy with progress, a temporary name the scan doesn't see, and
        # aborting without leaving a half file behind.
        camera = os.path.join(tmp, "camera")
        os.makedirs(camera)
        groot = os.path.join(camera, "00007.MTS")
        with open(groot, "wb") as f:
            f.write(os.urandom(COPY_BLOCK * 2 + 12345))    # three blocks, last one half
        dubbel = os.path.join(camera, "al aanwezig.mp4")   # already in opnames/, same size
        with open(dubbel, "wb") as f:
            f.write(b"dezelfde opname nog een keer")
        shutil.copy2(dubbel, os.path.join(recordings_path(bieb), "al aanwezig.mp4"))
        anders = os.path.join(camera, "naamgenoot.mp4")    # same name, different size
        with open(anders, "wb") as f:
            f.write(b"andere inhoud, andere grootte")
        with open(os.path.join(recordings_path(bieb), "naamgenoot.mp4"), "wb") as f:
            f.write(b"kort")
        with open(os.path.join(camera, "notitie.txt"), "w") as f:
            f.write("geen video")
        sync_source_dir(bieb)              # those two are already known by now
        plan = copy_plan(bieb, [groot, dubbel, anders, os.path.join(camera, "weg.mp4"),
                                   os.path.join(camera, "notitie.txt"), groot])
        assert [i["reden"] is None for i in plan] == [True, False, False, False, False, False]
        assert plan[1]["reden"] == "staat al in de bibliotheek"
        assert "hernoem" in plan[2]["reden"] and plan[3]["reden"] == "bestand niet gevonden"
        assert plan[4]["reden"] == "geen videobestand" and plan[5]["reden"] == "twee keer gekozen"
        # Abort after the first block: no file, no partial file, nothing in the scan.
        tellers = []
        try:
            copy_to_recordings(bieb, plan, progress_callback=lambda *a: tellers.append(a),
                                 stop_check=lambda: len(tellers) >= 1)
            raise AssertionError("should have aborted")
        except CopyAborted:
            pass
        assert tellers == [(COPY_BLOCK, plan[0]["bytes"], 0, "00007.MTS")]
        assert not os.path.exists(plan[0]["doel"])
        assert not os.path.exists(plan[0]["doel"] + COPY_PART)
        assert sync_source_dir(bieb) == 0
        # Complete: byte-identical, progress runs to the total, the scan picks it up.
        tellers = []
        geslaagd = copy_to_recordings(bieb, plan,
                                        progress_callback=lambda *a: tellers.append(a))
        assert geslaagd == [plan[0]["doel"]] and len(tellers) == 3
        assert tellers[-1][:2] == (plan[0]["bytes"], plan[0]["bytes"])
        with open(groot, "rb") as a, open(plan[0]["doel"], "rb") as b:
            assert a.read() == b.read()
        assert not os.path.exists(plan[0]["doel"] + COPY_PART)
        assert sync_source_dir(bieb) == 1
        assert source_video_for_path(bieb, plan[0]["doel"])["bestand"] == f"{RECORDINGS_DIR}/00007.MTS"
        # One more time: now it's "already in the library".
        assert copy_plan(bieb, [groot])[0]["reden"] == "staat al in de bibliotheek"

        print("Zelftest OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
