# Translating SchaatsAnalyse to English — progress & handoff

**Read this file first in any new session working on this effort.** It's written so a
session with zero memory of earlier ones can resume without re-deriving decisions
already made. Branch: `translate-to-english` (based on `main`, not yet merged).

## How to resume

1. `git log --oneline main..translate-to-english` to see exactly what's landed.
2. Read this whole file — especially "Patterns established" and "Glossary" below, since
   later phases must stay consistent with earlier ones.
3. Pick up at the first unchecked phase in "Phase status".
4. For each phase: rename the file (`git mv`), translate its content, apply whichever
   compatibility patterns apply (see below), verify (see "Verification per phase"),
   commit, update this document's checklist and commit that too.

Do not skip the compatibility patterns to save time — every phase so far has produced a
real, silent-failure-class bug when a pattern was skipped or applied incompletely, and
those were only caught by actually *running* things (self-tests, the GUI screen test),
never by `py_compile` alone. Budget for that verification step; it is not optional
polish.

## Why this is happening

Non-Dutch developers are joining the project; the whole codebase (~21,000 lines across
8 `schaats_*.py` modules), 9 markdown docs, the product name, and the SQLite schema are
in Dutch. Full scope, as chosen by the user: translate everything, including file/
product names and the database schema. See `CLAUDE.md` for the pre-translation
architecture description (still accurate for untranslated files; increasingly stale for
translated ones — update it in Phase 11).

The **actual end users of the shipped app are Dutch-speaking, non-technical trainers**
(see `INSTALLEREN.md`) — this is a deliberate, accepted trade-off the user chose
knowingly, not an oversight. Nothing about that changes the plan; it's just why the GUI
text (Phase 8) is being done carefully rather than skipped.

## Phase status

Each phase = one file renamed, translated, and committed on its own, verified before
moving on. Numbers match the original plan (`C:\Users\jelme\.claude\plans\reactive-wibbling-naur.md`
on the machine this was planned on — that plan file has the original design; this
document supersedes it for anything about actual progress and lessons learned).

- [x] **Phase 1 — Setup**: branch created.
- [x] **Phase 2 — `schaats_omgeving.py` → `skate_environment.py`** (commit `1958d7d`).
      Leaf module. Added one-time `%LOCALAPPDATA%\SchaatsAnalyse` → `...\SkateAnalysis`
      folder migration in `data_dir()`.
- [x] **Phase 3 — `schaats_perspectief.py` → `skate_perspective.py`** (commit `a076df5`).
      `PerspectiefKalibratie`→`PerspectiveCalibration`, `KalibratieInvoer`→
      `CalibrationInput`, `HoekReconstructie`→`AngleReconstruction`. Dual-key JSON
      read added directly in `CalibrationInput.from_dict()`. Method-name strings
      `'onderbeen'/'beenvlak'` → `'lower_leg'/'leg_plane'`, with a normalization shim
      in `reconstruct_angle()` accepting both spellings (needed since
      `schaats_analyse.py`/`schaats_gui.py` still pass the Dutch ones through).
- [x] **Phase 4 — `schaats_analyse.py` → `skate_analysis.py`** (commit `498a4c7`).
      The core module — biggest single-file diff so far. `FrameResultaat`→
      `FrameResult`, `AfzetEvent`→`PushEvent`, `PerspectiefConfig`→`PerspectiveConfig`
      (again, now fully owned by this file), `DoelTracker`→`TargetTracker`. ~50
      function names translated. `.npz` keys `pose_gevonden/middellijn_dev/bocht/
      totaal` → `pose_found/midline_dev/corner/total` (dual-read). `ONV_AFGEKAPT`/
      `ONV_GEEN_PUSH` → `INCOMPLETE_TRUNCATED`/`INCOMPLETE_NO_PUSH` (name **and**
      value translated — safe because every consumer imports the constant, none
      hardcodes the literal string). `get_landmarks()`'s dict keys (`l_heup`→`l_hip`
      etc.) translated outright — confirmed via grep that nothing outside this file
      indexes into that dict. **The stored `leg`/`been` VALUE stays `'links'/'rechts'`
      for now** — only the field *name* changed; translating the value too would
      silently break every `== 'links'` comparison in the four other files. Two real
      bugs caught only by a hand-written runtime script (not by `py_compile`/import):
      `PushEvent` was being constructed with the *old* Dutch keyword names in two
      places inside this same file (`_merge_events`, `segment_pushes`'s own event
      builder).
- [x] **Phase 5 — `schaats_db.py` → `skate_db.py`** (commit `15a39d6`). Real SQL
      migration, `SCHEMA_VERSIE`/`SCHEMA_VERSION` 5→6: every table and column renamed
      (`schaatser`→`skater`, `bronvideo`→`source_video`, `bron_markering`→
      `source_marking`, `analyse`→`analysis`, `afzet_event_cache`→`push_event_cache`),
      `status` enum values remapped, whole event cache recomputed through the real
      pipeline (with a text-remap fallback if an analysis's npz isn't reachable).
      Fixed a latent crash bug (`_tabel_ddl`/`_table_ddl` slicing `_SCHEMA` by table
      name — breaks the instant `_SCHEMA` stops containing the Dutch names an old
      migration step still needs) by freezing the old DDL as literals, same treatment
      as the existing frozen `v1_schema` test fixture.
      **Two new compatibility patterns had to be invented here — read "Patterns
      established" below before touching Phase 6, since it needs both again.**

- [ ] **Phase 6 — `schaats_yolo.py` → `skate_yolo.py`** (2,282 lines). **Confirmed:
      `schaats_gui.py:605` does a bare `import schaats_yolo` inside `_laad_backend()`
      (lazy-loaded for startup time) — this needs the exact same shim treatment as
      `schaats_db.py` (see Pattern B below), not just from-import aliases.**
      `.venv-yolo\Scripts\python.exe skate_yolo.py`'s self-test is the *only*
      verification path that actually imports this file under the plain venv never
      touches it — treat that self-test as mandatory, not optional, and also try to
      run one real analysis end-to-end under `.venv-yolo` if a test video is
      available.
- [ ] **Phase 7 — `schaats_eval.py` → `skate_eval.py`** (527 lines). Also rename
      `goud_schaats_frontaal.json` → `golden_skate_frontal.json` and its internal
      Dutch keys (`l_knie` etc.) — this file is a dev-only tool, not used by trainers,
      so lower stakes. Check whether `schaats_eval.py` is imported anywhere else
      (unlikely; it's a standalone CLI tool) before assuming no shim is needed.
- [ ] **Phase 8 — `schaats_gui.py` → `skate_gui.py`** (8,839 lines — bigger than every
      phase so far *combined*). Do this in the sub-steps from the original plan:
      - 8a. Rename file, translate identifiers/comments/docstrings only (structural
        pass; UI text still Dutch after this step).
      - 8b. Translate the main dialogs' window titles/labels/tooltips.
      - 8c. Translate the ~150 `QMessageBox` calls and ~178 tooltips across the rest
        of the file.
      - 8d. Update the three brand-string spellings found ("Schaats Analyse" splash +
        window title, "Schaatser Analyse" docstring headers + library header) to
        "SkateAnalysis". **This is also the point where `instellingen_json`'s
        content and `config.json`'s dict keys finally get translated** (deferred
        from Phases 4/5 specifically to be done together with this file — see
        "Deferred to Phase 8" below for the exact keys and why).
      - 8e. Re-run `.venv-yolo\Scripts\python.exe skate_screentest.py --alles` once
        8a-8d are committed — English strings are often a different length than the
        Dutch originals, and the window-size minimums documented in `CLAUDE.md` were
        measured against the Dutch text.
      Given the size, expect this phase alone to span several sessions. Commit after
      each sub-step, not just at the end of 8e — do not let this become one giant
      uncommitted diff.
- [ ] **Phase 9 — `schaats_schermtest.py` → `skate_screentest.py`** (342 lines). Do
      this *last* among the code files — it imports gui+db+analysis and is the best
      regression canary once everything else is renamed.
- [ ] **Phase 10 — Build, installer, branding**. Rename `schaatsanalyse.spec` →
      `skate_analysis.spec`, `schaatsanalyse.ico` → `skateanalysis.ico`,
      `maak_versie.py` → `make_version.py` (which generates `_versie.py` →
      `_version.py`), `bouw.bat` → `build.bat`. Update `installer.iss` (`AppName`/
      `AppPublisher`/paths → "SkateAnalysis", **keep the `AppId` GUID byte-identical**
      — it's commented "never change" in the source and is what Windows uses for
      upgrade-in-place detection, unrelated to the display name). Rename the
      remaining `SCHAATSANALYSE_CPU` env var (read in `schaats_yolo.py`/
      `skate_yolo.py` — should already be `skate_yolo.py` by this point from Phase 6)
      to `SKATEANALYSIS_CPU`; the other two (`_BIBLIOTHEEK`/`_LOKAAL`) were already
      renamed in Phase 5.
- [ ] **Phase 11 — Documentation**. Translate `CLAUDE.md`, `ROADMAP.md`, `BUGS.md`,
      `EXE.md`, `GPU.md`, `TODO_CRASH.md`, `INSTALLEREN.md`→`INSTALL.md`,
      `OPNAME.md`→`RECORDING.md`, `README.md`. Do this only once the actual names/
      paths they describe are final (i.e., after Phase 10), or you'll be translating
      stale Dutch names into stale English ones and redoing it. **Note**: there is
      currently an uncommitted, unrelated pending edit to `INSTALLEREN.md` on this
      branch (predates this effort, carried along from `main`) — read it and fold its
      *content* (about downloading the installer via the Drive app vs. a browser
      download) into the translated `INSTALL.md`, don't discard it.
- [ ] **Phase 12 — Final sweep**. `python -m py_compile` across the whole repo, every
      self-test once more, `grep -ri schaats` repo-wide (expect only the explicit
      exceptions below to still match), confirm `start_gui.bat`/`build.bat` point at
      the renamed files, and **delete the `schaats_db.py`/`schaats_yolo.py`
      compatibility shims** (and any leftover `_add_legacy_keys()`/`_alias()` calls
      whose caller has by now been translated directly) — they're transitional, not
      permanent design.

## Explicit exceptions — never rename these

| Item | Why |
|---|---|
| Physical file `schaats.db` | Storage internal inside a live Google-Drive-synced folder on every trainer's machine; renaming risks sync conflicts for zero benefit — nobody reads it by name. |
| `media/`, `opnames/` folders | Same reasoning. |
| `landmarks.npz`, `landmarks_ruw.npz` filenames | Same reasoning — `landmarks_ruw.npz` is a real file already sitting in every previously-hand-edited analysis; renaming it breaks "restore original" for those. Only the *keys inside* the npz got translated (Phase 4), never the filename. |
| GitHub repo name (`Jelm0r/SchaatsAnalyse`) | User's explicit choice. |
| `installer.iss`'s `AppId` GUID | Already commented "never change" — Windows upgrade-in-place identity, unrelated to the display name. |
| `v1_schema` fixture in `skate_db.py`'s self-test | Frozen historical fixture representing real old databases in the wild; must stay byte-for-byte Dutch. |
| `_OLD_BRONVIDEO_DDL` / `_OLD_BRON_MARKERING_DDL` in `skate_db.py` | Same reasoning, added in Phase 5 for the same class of problem (`_table_ddl` needing the *historical* Dutch shape for old migration steps). |

## Glossary (locked in — do not deviate, do not re-litigate)

| Dutch | English | Notes |
|---|---|---|
| afzet / afzethoek / afzetbeen | push / push angle / push leg | User-specified: "just use Push as in 'your push was not fast enough'". `AfzetEvent`→`PushEvent`. |
| bocht | corner | `_BochtWacht`→`_CornerGuard` (not done yet — lives in `schaats_yolo.py`, Phase 6). |
| slag (skating stroke cycle) | stroke | **Distinct from "push" above — do not conflate.** `STREK_MIN_SLAG_FRAC`→`EXTENSION_MIN_STROKE_FRAC`. |
| strek / strek_ratio | extension / extension_ratio | Leg-straightening signal. |
| standbeen | stance leg | |
| zweefbeen | swing leg | Standard biomechanics term, maps cleanly. |
| gewicht_erop | weight_on | |
| schaatser | skater | Product name "SchaatsAnalyse" → **"SkateAnalysis"** (user-chosen). |
| bronvideo | source_video | |
| bron_markering | source_marking | |
| kader (user-drawn box) | box | `doel_kader`→`target_box`, `KADER_MARGE`→`BOX_MARGIN`. Not done yet — lives in `schaats_gui.py`/`schaats_yolo.py`. |
| kijkglas (small-target tracking) | spyglass | `_Kijkglas`→`_Spyglass`. Claude's own naming call (low-stakes internal class), not user-specified. Not done yet — `schaats_yolo.py`. |
| doel / doelpunt / DoelTracker | target / target_point / TargetTracker | Done in Phase 4. |
| kamtanden / kam_masker | combing / comb_mask | Standard video-engineering term. Done in Phase 4 (`_comb_mask`, `deinterlace`). |
| opname | recording | `OPNAME.md`→`RECORDING.md` (Phase 11). |
| knippen / fragment | trim/cut / fragment | `knip_fragmenten`→`trim_fragments`. Done in Phase 4. |
| bibliotheek | library | Done in Phase 5 (`library_path`, etc). |
| instellingen | settings | DB column `instellingen_json`→`settings_json` done (Phase 5); the JSON *content*'s keys are not (deferred to Phase 8, see below). |
| kalibratie / verdwijnpunt | calibration / vanishing_point | Done in Phase 3. |
| baanlijn / dwarslijn | track_line / cross_line | Done in Phase 3. |
| pakkleur | suit_color | Not done yet — `schaats_yolo.py`. |
| onvolledig / afgekapt / geen volledige push | incomplete / truncated / no full push | Done in Phase 4 (`INCOMPLETE_TRUNCATED`, `INCOMPLETE_NO_PUSH`). |
| nog doen / bezig / klaar / onbruikbaar (recording status) | todo / in_progress / done / unusable | Done in Phase 5 (`SOURCE_STATUSES`), including the DB value migration. |

Already-English, left alone everywhere: `landmarks`, `horizon`, `smooth_n`, `threshold`,
`heavy`, `interlaced`/`deinterlace`, `recovery`, `tracklet`, `frame_nr` (kept
permanently — see below), `info`.

**`frame_nr` is a deliberate, permanent exception**, decided in Phase 4: "nr" is
cross-language comprehensible (used in English too, if old-fashioned), and this field
is used in probably 100+ places across every file. Renaming it would be enormous,
low-value churn. Do not rename it in any later phase.

## Patterns established (apply these consistently — do not improvise a new one per file)

Every phase needs SOME subset of these, depending on how the file is imported and what
it returns. Figure out which apply **before** writing any translated code, the same way
Phases 5 and 6 required checking import style first.

### Pattern A — `from X import name1, name2` in another untranslated file

The importing file's module *path* must change (`from schaats_analyse import ...` →
`from skate_analysis import ...`) regardless — the old module literally won't exist.
The *names* inside that import list can stay Dutch if the renamed module exports a
module-level alias for each one (`old_name = new_name` near the bottom of the file, in
a clearly marked section). Used for `skate_environment.py`, `skate_perspective.py`,
`skate_analysis.py`.

**Before assuming this pattern applies, grep for how the file is actually imported —
see Pattern B for why that matters.**

### Pattern B — bare `import X`, then `X.name(...)` used many times

A from-import alias does nothing here — `import schaats_db` needs a module literally
importable *as* `schaats_db`. Confirmed needed for `schaats_db.py` (75+ call sites in
`schaats_gui.py`) and **will be needed again for `schaats_yolo.py`** (confirmed via
`grep -n "import schaats_yolo"` → `schaats_gui.py:605`, inside `_laad_backend()`).

Fix: leave a tiny shim file at the OLD name that just does `from new_module import *`,
with a docstring explaining it's transitional and naming exactly when to delete it
(once the importing file is itself translated to `import new_module` directly). See
`schaats_db.py` for the exact template — copy it, don't reinvent it.

**Before deciding between Pattern A and B for a new file, always grep first**:
`grep -n "import schaats_X\|from schaats_X import" *.py`. Guessing wrong wastes an
entire rewrite pass.

### Pattern C — dataclass field renames

Add a small property-alias helper once per file (see `_alias()` in `skate_analysis.py`,
`skate_perspective.py`) so `instance.oude_naam` still reads/writes `instance.new_name`
transparently. A **plain, un-annotated class attribute** (`oude_naam = _alias("new_name")`)
inside a `@dataclass` body is not picked up as a dataclass field, so it coexists safely
with the real ones — confirmed working in both files above.

**Known trap**: this fixes attribute *access* (`obj.oude_naam`), not *constructor
keyword arguments* (`Klass(oude_naam=...)`) — a dataclass's generated `__init__` only
ever accepts the real (new) field names. Grep for every `ClassName(` construction site
across the whole repo and check for keyword args using old names; fix those directly
(there are usually only a handful — positional construction is unaffected by a field
rename, since it goes by position not name). This exact class of bug was caught twice
in Phase 4, both times *inside the same file being translated*, not by py_compile.

**Known trap #2**: aliasing a `@classmethod` needs `new_name = classmethod(old_name.__func__)`,
not a bare assignment (`new_name = old_name` loses the classmethod binding — calling it
on the class rather than an instance then passes the wrong thing as the first
argument). Caught and fixed once in `skate_perspective.py`'s `PerspectiveConfig`.

### Pattern D — functions returning dicts/rows whose keys come from renamed SQL columns

**The one that isn't obvious until it bites you.** Patterns A/B/C all fix "does
`module.name` resolve correctly" — none of them touch the *shape* of what a function
returns. If `list_analyses()` now returns `dict(row)` with column `title` instead of
`titel`, `schaats_gui.py` reading `row["titel"]` directly breaks, and no alias
mechanism above catches that, because the function name (`list_analyses`, aliased as
`lijst_analyses`) resolved just fine — the *dict it handed back* is what changed shape.

Fix: `_add_legacy_keys(d, old_name="new_name", ...)` in `skate_db.py` — mutates a dict
in place, adding each old key pointing at the same value as its new counterpart
(applied only where the new key is actually present). Apply it to every dict a
still-untranslated file might read fields from. **Only found this pattern by running
the self-test and the full GUI screen test** — grepping the *implementation* for old
column names doesn't catch it, since the bug is in the *caller* (`schaats_gui.py`)
reading a key that no longer exists. The way it was actually found: grep
`schaats_gui.py` itself for `["dutch_word"]` patterns matching the columns you just
renamed, and check each hit against what the called function now actually returns.

This pattern will very likely recur in Phase 6 if `schaats_yolo.py` returns any
dict/row-shaped data consumed by `schaats_gui.py` — check for it explicitly, don't
assume it only applies to `skate_db.py`.

### Pattern E — persisted values with no live consumer to keep in sync

`.npz` array keys, DB column values (`bronvideo.status`), and marker text
(`ONV_AFGEKAPT`'s value) needed **dual-read, English-only-write**, not a hard rename:
old files/rows already exist with old values, and nothing forces every one of them to
be rewritten before the new code runs against them. Read with a fallback
(`arrays['new_key'] if 'new_key' in arrays else arrays['old_key']`), always write only
the new key/value going forward. For genuinely orphaned old data (nothing left ever
reads it), a **migration** that rewrites it in place (Phase 5's schema migration) is
better than carrying the fallback forever — but only do that once you're certain
nothing else still *writes* the old form (see Pattern F for why writers matter).

### Pattern F — string VALUES compared across files, where the writer isn't translated yet

`PushEvent.leg`'s stored value stays `'links'`/`'rechts'` deliberately (Phase 4) even
though the *field name* is now `leg` — because `schaats_gui.py` (untranslated) does
`== 'links'` comparisons against it in several places, and those would silently always
fail if the stored value became `'left'`. Same story for the `'onderbeen'`/`'beenvlak'`
method-name strings in Phase 3 (fixed with a local normalization dict instead, since
that specific value only flows through one function call, not a stored field). **Rule
of thumb**: if a file you haven't translated yet still *writes* a Dutch value your
translated code will *read and compare*, keep accepting/producing the Dutch value until
that writer's own phase — translate the field/column *name*, not the value, until both
ends of the pipe are being translated together.

## Deferred to Phase 8 (do not attempt piecemeal before then)

`instellingen_json`'s JSON *content* keys (`doel_punt`, `doel_kader`, `bocht_overslaan`,
`perspectief_gebruikt`, `perspectief`, `backend_naam`, `app_versie`, `app_commit`) and
`config.json`'s dict keys (`bibliotheek_pad`, `trainer_naam`) are **intentionally still
Dutch** after Phase 5, even though `skate_db.py` itself is fully translated. Reason:
the code that *constructs* these dicts lives in `schaats_gui.py`, which isn't
translated yet. Renaming what `skate_db.py`'s `save_analysis()`/`load_config()` expect
without also updating `schaats_gui.py`'s construction sites in the same commit would
silently split a single logical key into two (one written under the old name by
`schaats_gui.py`, one defaulted under the new name by `skate_db.py`), permanently
losing whatever `schaats_gui.py` saved. Do this translation **together with** Phase 8,
in one coordinated change, with a proper one-time rewrite of existing analyses'
`instellingen_json` (the ~48 analyses currently in the library) — do not attempt a
"rewrite now, fix the writer later" sequence, for the same reason.

## Verification per phase (do all of these, in this order, every time)

1. `python -m py_compile <every touched file>` — syntax only, catches typos.
2. A real `import <module>` (not just py_compile) on every touched file, under **both**
   venvs if the file is reachable from `schaats_yolo.py`'s dependency chain. `py_compile`
   does not execute imports and will not catch a broken cross-file reference.
3. The module's own self-test if it has one (`python skate_environment.py`,
   `python skate_perspective.py`, `python skate_db.py`,
   `.venv-yolo\Scripts\python.exe skate_yolo.py`). For `skate_yolo.py` specifically,
   this is **mandatory**, not optional — nothing else in the verification chain ever
   imports it under the plain venv.
4. For files with no self-test (`skate_analysis.py`, and `skate_gui.py` once renamed):
   write a small throwaway runtime script that actually calls the changed functions
   with real-ish data and checks the output — not just that it imports. This is what
   caught the `PushEvent` construction bugs in Phase 4 and would have caught the dict-
   key bugs in Phase 5 faster than waiting for the full GUI test to fail.
5. `.venv-yolo\Scripts\python.exe schaats_schermtest.py` (quick mode, ~2s) after any
   phase touching something `schaats_gui.py` depends on — it exercises a large fraction
   of the GUI end-to-end (opening the library, an analysis, several dialogs) and has
   caught real bugs twice already (Phase 5's dict-key issue was confirmed fixed this
   way). Run `--alles` (~1.5 min) specifically after Phase 8d, since English strings
   changing length can push a dialog past its measured minimum size.
6. Commit only after all of the above pass. Write a commit message that states what
   was renamed, what compatibility pattern was applied and why, and what was verified —
   future sessions (including future you) will read these messages instead of
   re-deriving the reasoning. See the four existing commits on this branch for the
   expected level of detail.

## Where the original plan lives

The plan this was scoped from, including the background-agent validation findings
that shaped Phase 5's migration design, is at
`C:\Users\jelme\.claude\plans\reactive-wibbling-naur.md` on the machine this was
authored on — not guaranteed to be readable from a different machine or a fresh
container. Everything from it that still matters for future phases has been folded
into this document; treat this file, not that one, as authoritative for anything about
current status.
