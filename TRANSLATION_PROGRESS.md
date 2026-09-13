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

**Process amendment, 13 Sep 2026 (user's explicit choice — "maximum speed"):** the
compatibility patterns above are still mandatory and still get applied per class/section
exactly as before — they're the actual substance of this work, not overhead, and are what
caught every real bug so far. What changes is everything *around* them, to cut tokens/
session count:
- **Batch much bigger chunks per session** — a whole run of remaining classes at once
  (e.g. all of `VergelijkKant` through `KopieerDialoog` in one pass) instead of one class
  at a time. `MainWindow` (~3,460 lines) will still likely need to be split, purely
  because of its size, not out of caution.
- **No bespoke throwaway verification script for a class with no real behavioral risk**
  (no dataclass/constructor-keyword rename, no dict/row-shape change, nothing another
  file reads by structure) — for those, `py_compile` + a real `import skate_gui` under
  both venvs + a repo-wide grep for every old identifier + the screen test (quick mode
  during the pass, `--alles` only at the true end of Phase 8) is enough. Still write a
  real script for anything touching a worker's signals, a dialog's constructor kwargs, or
  any dict a still-untranslated caller reads by key — those are exactly the categories
  that have produced real bugs.
- **Verify once at the end of a large chunk, not after every class** — commit checkpoints
  still happen (don't let this become one giant uncommitted diff across sessions), just
  less often than "every class."
- **Stop duplicating the full narrative in both the commit message and this document.**
  Write the detailed "what/why/what was found" story **once**, in the commit message
  (git already keeps that forever). This document's own session-log entries from here on
  should be short: what got renamed (the rename table), which patterns applied and why,
  any bug found, a one-line verification summary, and the commit hash to read for the
  full story — not a second copy of the same paragraphs. (Sessions 1-6's entries above
  predate this amendment and are already written the old, fuller way — leave them as
  they are rather than rewriting history.)

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

- [x] **Phase 6 — `schaats_yolo.py` → `skate_yolo.py`** (2,282 → 2,325 lines).
      Pattern B shim confirmed needed and applied exactly like `schaats_db.py`
      (`schaats_gui.py:605`'s bare `import schaats_yolo` inside `_laad_backend()`
      reads `schaats_yolo.BACKEND_NAAM` and calls `schaats_yolo.analyseer(...)`;
      both now exist as Dutch aliases at the bottom of `skate_yolo.py` —
      `BACKEND_NAAM = BACKEND_NAME`, `analyseer = analyze` — re-exported by the
      `schaats_yolo.py` shim). `_BochtWacht`→`_CornerGuard`, `_Kijkglas`→`_Spyglass`,
      `KleurReferentie`→`ColorReference`, `Detectie`→`Detection`, and ~40 more
      functions/constants translated (kader→box, kleur→color/suit color, keten→chain,
      klik→click, kijkglas→spyglass, verfijn→refine, doel→target — see the glossary
      additions below). Followed the precedent already set by `skate_analysis.py`'s
      own `analyze()`: kept `bocht`, `doel_punt`, `doel_kader`, `perspectief`,
      `waarschuwing_callback` as the literal parameter names of the public
      `analyze()` function (and, for `bocht`, of every internal helper it flows
      through too — simpler and lower-risk than splitting internal/external naming
      for one identifier) because `schaats_gui.py`'s two call sites
      (`AnalyseWorker.run`, `BatchWorker.run`) pass them as keyword arguments;
      confirmed via `inspect.signature(...).bind()` with both call sites' exact
      kwargs before relying on it. Also kept the pre-existing project-wide
      conventions of never translating `_pad`-suffixed names, `resultaten`, or
      `deinterlacen` (all three still appear untranslated throughout the already-
      "complete" `skate_analysis.py`/`skate_db.py` — checked by grep before assuming
      Phase 4/5 had done a 100%-pure pass; they hadn't, and matching that bar avoids
      inventing a stricter standard than the rest of the codebase follows).
      **Found and fixed a real Pattern-D bug that predates this phase**: RTMPose's
      midline-quality dict used Dutch keys (`'l_knie'`/`'r_knie'`), but
      `skate_analysis.py`'s `results_to_arrays`/`arrays_to_results` (Phase 4) already
      expected English (`'l_knee'`/`'r_knee'`) — so every YOLO-backend analysis with
      RTMPose refinement was silently writing an all-NaN midline-deviation array to
      the npz. Fixed by writing the English keys directly (confirmed via a throwaway
      round-trip script). User-facing warning/message strings (the ones passed to
      `waarschuwing_callback` or embedded in raised exceptions) were translated to
      English too, matching the precedent already set in `skate_perspective.py`
      (Phase 3) of translating these even though they'll show up mixed with
      still-Dutch GUI text until Phase 8 — confirmed by finding
      `schaats_gui.py:1696` interpolates a `skate_perspective` exception's English
      text straight into a Dutch status label today.
      **Verified**: `py_compile` across the repo; `.venv-yolo\Scripts\python.exe
      skate_yolo.py` self-test (`_CornerGuard`/`_choose_seed`/`_Spyglass`, all pass);
      `inspect.signature().bind()` against both real `schaats_gui.py` call sites;
      a throwaway script confirming the `l_knee`/`r_knee` fix round-trips through
      `results_to_arrays`/`arrays_to_results`; `import schaats_gui` under
      `.venv-yolo` (exercises the full shim chain); `.venv-yolo\Scripts\python.exe
      schaats_schermtest.py` (all windows OK); and — going further than the
      "mandatory" bar — a full real end-to-end `analyze()` run on
      `Downloads\Schaats frontaal.MOV` (RTMPose refinement, cached weights, no
      target click/box), which reproduced *exactly* the reference numbers already
      documented in `CLAUDE.md` for this clip: 103/103 coverage, 0 corner frames,
      six push events at 42.2/42.5/42.9/40.0/45.6/50.0°, perfect R-L-R-L-R-L
      alternation, last one `INCOMPLETE_TRUNCATED` — a 92 s run, byte-for-byte the
      documented measurement.
- [x] **Phase 7 — `schaats_eval.py` → `skate_eval.py`** (commit `4302a08`, 527 → 533
      lines). Confirmed via `grep -rn "import schaats_eval\|from schaats_eval"` that
      nothing imports it anywhere else (standalone dev CLI) — **no shim needed**, full
      rename/translation in one pass. Also renamed `goud_schaats_frontaal.json` →
      `golden_skate_frontal.json` and translated its internal keys (`l_knie/l_enkel/
      r_knie/r_enkel` → `l_knee/l_ankle/r_knee/r_ankle`) directly, since nothing else
      reads this fixture. Reads `skate_analysis.py`'s English names directly
      (`load_landmarks`, `process_derivatives`, `segment_pushes`,
      `calculate_angle_to_ice`, `corner_ratio`, `determine_corner_sequence`,
      `CORNER_IN`/`CORNER_OUT`) and `FrameResult`/`PushEvent` fields directly
      (`r.leg`, `r.pose_found`, `r.corner`, `r.midline_dev`, `e.leg`, `e.angle`,
      `e.note`, `e.incomplete`) rather than through the Dutch aliases — safe since
      there's no other consumer to keep in sync with. CLI subcommands translated too
      (`vergelijk`/`annoteer`/`bocht` → `compare`/`annotate`/`corner`, flags
      `--uit`/`--stap` → `--out`/`--step`), since this is a dev-only tool with no
      trainer-facing surface. Kept the lower-bar conventions (`pad`/`_pad`-suffixed
      names, `naam`, `resultaten` left untranslated) — with one deliberate exception:
      `uit_pad` → `out_pad`, since (unlike `bron_pad`/`video_pad`) that name doesn't
      already exist elsewhere in the codebase to stay consistent with; it's introduced
      fresh in this file and mirrors the existing `output_pad` convention.
      **Found and fixed a real latent bug that predates this phase** (Pattern F, but
      running backwards): `calculate_metrics`'s alternation-error count compared
      `e.opmerking` against the literal Dutch string `'gemiste tegenafzet?'`, but
      Phase 4 already changed the *value* `skate_analysis.py` writes for that note to
      English (`"missed counter-push?"`) — so on every npz produced since Phase 4
      landed, this comparison silently always evaluated to zero. Fixed by comparing
      against the current value; confirmed the fix actually catches a case (an
      `LRRLRLRL` sequence pulled from the library now reports 1 alternation error
      instead of 0).
      **Verified**: `py_compile` across the touched + dependency files; real
      `import skate_eval` under both venvs; `--help` on every subcommand; full
      end-to-end `metrics`/`compare`/`corner` runs against real npz files from the
      library (output correctly formatted, no crashes); `metrics --golden` against
      the translated `golden_skate_frontal.json` reproduced the exact documented
      reference number from `CLAUDE.md` (GOLDEN stance leg avg 1.34° on "Schaats
      frontaal.MOV"), confirming the JSON key translation and `_golden_errors` are
      behaviorally identical to the original; `annotate()` exercised end-to-end with
      a mocked `cv2` (the 'q'-stop path, and a full click run through both the coarse
      and precise stages plus a redo) since it needs a real display otherwise,
      confirming it writes valid English-keyed JSON with no leftover Dutch
      identifiers; grepped the finished file for common Dutch words — no hits.
- [ ] **Phase 8 — `schaats_gui.py` → `skate_gui.py`** (8,839 lines — bigger than every
      phase so far *combined*). Do this in the sub-steps from the original plan:
      - 8a. Rename file, translate identifiers/comments/docstrings only (structural
        pass; UI text still Dutch after this step). **In progress, spans several
        sessions on its own — see "8a session log" right below before continuing.**
      - 8b. Translate the main dialogs' window titles/labels/tooltips.
      - 8c. Translate the ~150 `QMessageBox` calls and ~178 tooltips across the rest
        of the file.
      - 8d. Update the three brand-string spellings found ("Schaats Analyse" splash +
        window title, "Schaatser Analyse" docstring headers + library header) to
        "SkateAnalysis" (the docstring-header instance is already done, see below —
        the splash screen's painted text and the main window title remain). **This is
        also the point where `instellingen_json`'s content and `config.json`'s dict
        keys finally get translated** (deferred from Phases 4/5 specifically to be
        done together with this file — see "Deferred to Phase 8" below for the exact
        keys and why).
      - 8e. Re-run `.venv-yolo\Scripts\python.exe skate_screentest.py --alles` once
        8a-8d are committed — English strings are often a different length than the
        Dutch originals, and the window-size minimums documented in `CLAUDE.md` were
        measured against the Dutch text.
      Given the size, expect this phase alone to span several sessions. Commit after
      each sub-step, not just at the end of 8e — do not let this become one giant
      uncommitted diff.

      **8a session log — read this before resuming 8a.** 8a itself is too big for one
      session; it's being done class-by-class/section-by-section, each its own commit.
      `git log --oneline` on this branch shows the commits; here's what's landed and
      exactly where the next one picks up.

      - **Session 1 (commit `769f590`)** — file renamed, and lines 1 through ~948
        (everything *before* `class DoelKiezer`) fully translated: module docstring,
        the startup/splash-screen sequence, the skeleton-editor and on-screen-drawing
        constants, corner/perspective tooltip constants + `_calibration_rows`
        (`_kalibratie_rijen`), and the generic window/dialog infrastructure
        (`set_window_size`, `show_dialog`, `FlowLayout`, `WrapBar`, `ElideLabel`,
        `minimum_with_wrapping`). Also did the **global** Pattern A/B import cleanup
        for the *whole file* regardless of section (this had to be file-wide, not
        session-scoped, because Python identifiers must stay consistent): switched
        `import schaats_db`/`import schaats_yolo` to `import skate_db`/`import
        skate_yolo` and every one of their ~75 and 2 call sites respectively to the
        real English names (no more Dutch aliases anywhere in this file for those two
        modules), and switched `from skate_analysis import (...)` from the 14 Dutch
        aliases to the real English names, renaming every usage throughout the file.
        One real bug from this (Pattern G): `MainWindow.__init__` (untranslated, far
        outside the edited range) called `set_window_size(..., maximaliseer=True)` —
        fixed that one keyword. Also updated `start_gui.bat` and
        `schaats_schermtest.py`'s `import schaats_gui as G` to point at `skate_gui.py`
        — required just to keep the app runnable and the verification loop working
        through the rest of Phase 8, ahead of Phases 9/10's own pointer-file updates.
        Full detail (exact rename tables, what was deliberately left Dutch and why) is
        in the commit message — read it with `git show 769f590`.

      - **Left Dutch on purpose, with an inline code comment pointing back to this
        file, at one remaining spot** (so a future session doesn't have to rediscover
        this by grepping):
        1. `OPNAME_KOL_*`, `LOKAAL_WEERGAVE`, `SEEK_DREMPEL_FRAMES`,
           `SNELHEDEN`/`_snelheid_idx`/`SNELHEID_DEFAULT_IDX`, `SPEEL_*`, `SPOEL_*`,
           `VIDEO_TOETSEN_HULP`/`VIDEO_TOETSEN_TOOLTIP`/`toetsen_hulp`,
           `wissel_volledig_scherm`, `TRANSPORT_KNOP_BREEDTE`, `ALLES_TICK_MS`,
           `ALLES_SNELHEID_IDX` (lines ~430-560ish) — comments around them are already
           translated, but the identifiers themselves belong with whichever future
           session translates `VideoSpeler`/`SpelerToetsen`/`MasterKlok`/the recordings
           tab, since that's where they're actually used.

        (Session 1's other two items are now done — see session 2 below: `KADER_*` →
        `BOX_*`, and `DoelKiezer`/`HorizonKiezer`/`KalibratieKiezer` themselves.)

      - **Session 2 (this session)** — `DoelKiezer`→`TargetPicker`,
        `HorizonKiezer`→`HorizonPicker`, `KalibratieKiezer`→`CalibrationPicker`
        (~942-1772: the module-level `KADER_*`→`BOX_*` constants through the end of
        `CalibrationPicker`, right up to `class AnalyseAfgebroken`). Identifiers,
        docstrings, and comments only — window titles/labels/tooltips/messages are
        still Dutch (8b/8c). `doel_punt`/`doel_kader` (on `TargetPicker`) and
        `perspectief` (on `CalibrationPicker`) were deliberately kept as-is — per
        "Deferred to Phase 8" below, they flow straight into still-Dutch
        `instellingen_json` keys built elsewhere in this file (not yet reached by this
        session); renaming just the attribute would split one logical key into two
        spellings. `horizon_deg`/`auto_per_frame` needed no change (already English).
        Confirmed via `grep` that all three classes are referenced from exactly one
        `MainWindow` call site each plus `schaats_schermtest.py` (updated too), and
        fixed those.

        **Found and fixed two real, pre-existing bugs while translating
        `KalibratieKiezer`/`CalibrationPicker`** (both Pattern C/D, both predate this
        session — introduced when Phase 3/4 translated `skate_perspective.py`/
        `skate_analysis.py`'s `CalibrationInput`/`PerspectiveConfig` out from under this
        still-Dutch file, and never caught since because nothing in the verification
        chain up to now ever exercised the calibration flow end-to-end):
        1. **`_bevestig` (now `_confirm`) constructed `PerspectiveConfig` with the old
           Dutch keyword names** (`kalibratie=`, `methode=`, `onderbeen_l=`, `invoer=`).
           Since Phase 4, `PerspectiveConfig`'s real dataclass fields are
           `calibration`/`method`/`lower_leg_l`/`calibration_input` — the Dutch names
           only exist as attribute-access aliases (Pattern C), which do **not** cover
           constructor keyword arguments (this is Pattern C's documented "known trap").
           Confirmed with a throwaway script that the old call raises `TypeError`
           immediately: **every attempt to confirm a perspective calibration through
           this dialog on this branch has been crashing** since Phase 4 landed
           (`498a4c7`), not just producing a subtly wrong result — it simply never got
           far enough to reach `self.accept()`. Fixed by using the real field names.
        2. **`_calibration_rows` indexed the saved calibration dict by its old Dutch
           keys** (`invoer.get("rijlijnen")`, `.get("dwarslijnen")`, `.get("beeld_w")`,
           etc.), but `CalibrationInput.to_dict()` has written English keys
           (`track_lines`, `image_w`, ...) since Phase 3 — so for any
           perspective-corrected analysis, the Info dialog's calibration section
           silently rendered as empty (`p.get("invoer")` found nothing, since the real
           top-level key is `calibration_input`). Fixed by reading through
           `CalibrationInput.from_dict()` (which already dual-reads old/new spellings,
           Pattern E) instead of indexing the raw dict, while leaving the *displayed*
           row text in Dutch (still deferred). Also added the same
           `'onderbeen'/'beenvlak'` → `'lower_leg'/'leg_plane'` value normalization
           (matching the existing shim in `skate_perspective.reconstruct_angle()`) to
           `_prefill`'s (formerly `_vul_voor`'s) method-combo lookup, since the combo's
           item data is now the English spelling and a real on-disk analysis from before
           Phase 3 still has the old value in its `method`/`methode` field.

        **Verified**: `py_compile`; real `import skate_gui` under both venvs
        (offscreen Qt platform); a throwaway script (with a real `QApplication` and a
        synthetic camera from `skate_perspective`'s own self-test helpers,
        `_SynthCamera`/`_scene_lines`) that drives `CalibrationPicker` end-to-end —
        draws lines, recalibrates, calls `_confirm()`, confirms a `PerspectiveConfig`
        comes out (previously: `TypeError`) — then feeds its real `to_dict()` output
        into `_calibration_rows()` and confirms non-empty rows (previously: `[]`), then
        feeds a hand-built *old-format* Dutch-keyed dict through the same function and
        confirms it *still* reads correctly (dual-read preserved), and finally confirms
        `_prefill()` correctly maps an old `method="onderbeen"` value onto the new
        `"lower_leg"` combo entry; `.venv-yolo\Scripts\python.exe schaats_schermtest.py`
        (quick mode) — all 16 windows pass, including the three renamed dialogs by
        their new names.

      - **Session 3 (same day as session 2)** — `AnalyseAfgebroken`→`AnalysisAborted`,
        `AnalyseWorker`→`AnalysisWorker`, `BatchWorker` (name unchanged, already
        English), `SchaatserDialog`→`SkaterDialog`, `NieuweAnalyseDialog`→
        `NewAnalysisDialog`, `BatchAnalyseDialog`→`BatchAnalysisDialog`,
        `AnalyseKiezer`→`AnalysisPicker`, `_ja_nee`→`_yes_no`, `_duur_tekst`→
        `_duration_text`, `AnalyseInfoDialog`→`AnalysisInfoDialog` (~1801-2558, up to
        but not including `class VooruitLezer`). Identifiers/docstrings/comments only,
        same as every 8a session -- row labels inside `AnalysisInfoDialog` and every
        other UI string stay Dutch (8b/8c).

        `AnalysisWorker`'s and `BatchWorker`'s Signals were renamed too (`voortgang`→
        `progress`, `klaar`→`done`, `fout`→`error`, `opslag_fout`→`save_error`,
        `waarschuwing`→`warning`, `taak_start`→`task_start`, `taak_klaar`→`task_done`,
        `taak_fout`→`task_error`, `alles_klaar`→`all_done`) and `breek_af()`→`abort()`,
        with the internal `afbreken` flag→`cancelled` (confirmed via grep this flag has
        no external readers -- safe to rename freely, unlike `bieb`/`analyse_id`/etc
        below). This meant touching the two `MainWindow` call sites that construct these
        workers and connect to their signals (~11 lines total, the same bounded-touch
        pattern as session 2's three picker call sites) -- grepped for every
        `.connect(...)`/`.emit(...)`/construction site first, confirmed exactly two
        instantiation points, updated both.

        **Deliberately left Dutch** (checked case-by-case, same judgment call as
        `doel_punt`/`doel_kader`/`perspectief` in session 2, for a different reason this
        time): `bieb`, `schaatser_id`, `titel`, `instellingen`, `aangemaakt_door`,
        `analyse_id`, `naam`, `geboortejaar`, `notities`, `taken`/`taak`,
        `schaatser_naam`, `heavy_gevraagd`, `geen_smoothing`. These aren't deferred
        JSON-key concerns -- they're plain Python parameter/attribute names -- but they
        turned out to be genuinely **shared vocabulary spanning `MainWindow`** (tens to
        hundreds of call sites each, confirmed with `grep -c` before deciding, e.g.
        `analyse_id` alone has 30+ hits across `MainWindow`/`VergelijkKant`) rather than
        contained to the classes in this chunk. Translating them here would rename maybe
        10% of their occurrences and leave the rest mismatched until whichever session
        finally does `MainWindow` -- exactly the split-personality risk Pattern F/G warn
        about, just for ordinary identifiers instead of persisted values this time. Also
        confirmed `skate_db.py`'s own public functions (`create_skater`, `save_analysis`,
        ...) still take these exact Dutch parameter names (Phase 5 translated function/
        table/column *names* but not parameter names -- same "lower bar" as
        `resultaten`/`_pad`/`naam` elsewhere), so leaving them Dutch here doesn't even
        introduce a fresh inconsistency; it matches the rest of the already-"finished"
        codebase.

        **Verified**: `py_compile`; real `import skate_gui` under both venvs (offscreen
        Qt platform), confirming every renamed class/Signal resolves and every Signal's
        C++ type signature looks right; a throwaway script exercising `AnalysisWorker`
        and `BatchWorker` end-to-end with a mocked `analyze_backend`/`skate_db` (not a
        real video/model) -- confirms `progress`/`warning`/`done` fire correctly on a
        normal run, `abort()`/`cancelled` correctly stops `run()` silently via
        `AnalysisAborted` (both before and during a run), and `BatchWorker` correctly
        emits `task_start`/`task_done`/`task_error`/`all_done` across a two-task batch
        where one task fails; a second throwaway script confirming `SkaterDialog`/
        `NewAnalysisDialog`/`BatchAnalysisDialog`'s properties (`naam`, `schaatser_id`,
        `titel`, `taken`, `smooth_n`, ...) still read correctly after the rename;
        `.venv-yolo\Scripts\python.exe schaats_schermtest.py` (quick mode) -- all 16
        windows pass, including all five renamed dialogs by their new names.

      - **Session 4** — `VooruitLezer`→`ForwardReader` only (~2547-2554, 2557-2632, plus
        its ~13 call sites scattered through the not-yet-translated `VideoSpeler`
        class). Deliberately scoped smaller than the "VooruitLezer + all of
        `VideoSpeler`" suggestion below — `VideoSpeler` itself is ~1,100 lines, too big
        for one sitting per the user's "keep sessions small" request this session opened
        with, so it's split off as its own future session (see "Next up" below).
        `VOORUIT_MAX_BYTES`/`VOORUIT_MAX_FRAMES`→`READAHEAD_MAX_BYTES`/
        `READAHEAD_MAX_FRAMES`, `pak()`→`take()`, `resterend()`→`remaining()`,
        `einde`→`at_end`, constructor params `aantal`/`voorraad`→`count`/`carryover`.
        In `VideoSpeler` (otherwise still fully Dutch): renamed only the attributes/
        methods that are this class's own state (`self._lezer`→`self._reader`,
        `self._vooruit_rest`→`self._readahead_rest`, `_start_vooruitlezen`/
        `_stop_vooruitlezen`→`_start_readahead`/`_stop_readahead`) and translated only
        the comments/docstrings directly about read-ahead at each of those ~13 call
        sites (`toon_op_klok`'s full docstring, the `_start_readahead`/`_stop_readahead`
        method bodies+docstrings, and one-line comments elsewhere) — left every
        surrounding line (method names like `speel`/`pauzeer`/`ga_naar`/`_speel_tick`,
        their own docstrings, unrelated local variables) untouched in Dutch, since those
        belong to the general `VideoSpeler` translation pass, not this one. Confirmed via
        `grep` that every `_lezer`/`lezer`/`.pak(`/`.resterend(` occurrence was one of
        these call sites (no collision with unrelated names like `meta_lezer`) before
        editing, and that exactly two remaining Dutch-language *mentions* of the concept
        (not the identifier) sit inside `MasterKlok`'s still-untranslated docstring/
        comments (~3983, ~4027) — deliberately left alone, out of scope until
        `MasterKlok`'s own session.
        **Verified**: `py_compile`; real `import skate_gui` under both venvs (offscreen
        Qt); a throwaway script instantiating `ForwardReader` directly against a fake
        `cv2`-shaped capture (no real video needed) confirming `take()`'s discard-before-
        wanted and newest-if-behind semantics, `remaining()`'s carry-back, and `at_end`
        all behave identically to the original `pak()`/`resterend()`/`einde`;
        `.venv-yolo\Scripts\python.exe schaats_schermtest.py` (quick mode) — all 16
        windows pass.

      - **Session 5** — the rest of `VideoSpeler`→`VideoPlayer` (~2633-3736, everything
        left after session 4 split off `ForwardReader`): full identifier/docstring/comment
        translation, ~50 methods and ~40 attributes. Class itself renamed
        `VideoSpeler`→`VideoPlayer`; constructor params `min_grootte/toon_snelheid/
        snel_zoeken/toon_overlay/toon_tekenen`→`min_size/show_speed/fast_seek/
        show_overlay/show_drawing`. Because this class's public surface (methods,
        callback-hook attributes, widget handles) is read from many other not-yet-
        translated classes in the *same file* (`MainWindow`, `VergelijkKant`,
        `FragmentKiezer`, `BekijkVenster`), the risk here isn't Pattern A/B (no module
        boundary — it's all one file) but plain missed call sites. Handled it as a
        same-file variant of Pattern G: `grep -c` every candidate identifier first,
        split into "safe to rename file-wide" (0 external refs, or external refs that
        are themselves unambiguous — e.g. `snel_zoeken` only ever appears as a
        `VideoSpeler(...)` keyword at 2 call sites) vs. "leave alone" when the name
        turned out to be shared vocabulary spanning `MainWindow`'s own not-yet-
        translated state (see below). Renamed via a scripted whole-file word-boundary
        regex pass (~90 identifier pairs) rather than by hand, specifically *because*
        it's all one file and grep could verify every hit — followed by a full manual
        read-through translating every comment/docstring the script doesn't touch.
        Renamed (methods): `laad`→`load`, `sluit`→`release` (not `close` — would shadow
        `QWidget.close()`'s different meaning), `ga_naar`→`go_to`, `toon_huidig_frame`→
        `show_current_frame`, `toon_op_klok`→`show_on_clock`, `speel`/`pauzeer`/
        `speelt`→`play`/`pause`/`is_playing`, `zet_besturing_actief`→
        `set_controls_active`, `voeg_bedieningsknop`/`voeg_onderbalk`→
        `add_control_button`/`add_bottom_bar`, `herbereken_kader`→`recompute_box`,
        the whole zoom/pan family (`_zet_zoom`→`_set_zoom`, `_kader_op`→`_box_at`,
        `_zoom_plafond`→`_zoom_ceiling`, `_bereken_zoom_eff`→`_compute_effective_zoom`,
        `widget_naar_norm`/`norm_naar_widget`→`widget_to_norm`/`norm_to_widget`, ...),
        the whole drawing family (`wis_tekening`→`clear_drawing`, `_teken_*`→`_draw_*`/
        `_paint_drawing`, `_muis_*`→`_mouse_*`, `bewerk_modus`→`edit_mode`, `_kader`→
        `_box` matching the existing `kader`→`box` glossary entry from `skate_yolo.py`),
        the whole playback-clock family (`_speel_tick`→`_play_tick`,
        `_ijk_speelklok`→`_calibrate_play_clock`, `speeltimer`→`play_timer`), plus the
        owner-hook attributes (`op_frame_getoond`→`on_frame_shown`,
        `overlay_tekenaar`→`overlay_drawer`, `op_muis_druk/_beweeg/_los`→
        `on_mouse_press/_move/_release`) and their ~10 external assignment sites in
        `MainWindow`/`VergelijkKant`/`BekijkVenster`. Also renamed the handful of
        VideoPlayer-owned module-level constants from session 1's "left Dutch on
        purpose" list that this class actually depends on: `SNELHEDEN`/`_snelheid_idx`/
        `SNELHEID_DEFAULT_IDX`/`ALLES_SNELHEID_IDX`→`SPEEDS`/`_speed_idx`/
        `SPEED_DEFAULT_IDX`/`ALL_SPEED_IDX`, `VIDEO_TOETSEN_HULP`/`VIDEO_TOETSEN_TOOLTIP`/
        `toetsen_hulp`→`VIDEO_KEYS_HELP`/`VIDEO_KEYS_TOOLTIP`/`keys_help`,
        `wissel_volledig_scherm`→`toggle_fullscreen`, `TRANSPORT_KNOP_BREEDTE`→
        `TRANSPORT_BUTTON_WIDTH`, `SEEK_DREMPEL_FRAMES`→`SEEK_THRESHOLD_FRAMES`,
        `SPEEL_OVERSAMPLE`/`SPEEL_TIK_MIN_MS`→`PLAY_OVERSAMPLE`/`PLAY_TICK_MIN_MS` (all
        of these are also used by the still-Dutch `SpelerToetsen`/`MasterKlok`/
        `BekijkVenster`/`FragmentKiezer`, so those call sites got the mechanical rename
        too, without translating anything else in those classes). Also renamed the
        `_bouw_ui` method — but **only** `VideoPlayer`'s own copy, scoped by line range
        rather than the whole-file regex, since `MainWindow` independently defines its
        own unrelated `_bouw_ui` that stays Dutch until `MainWindow`'s own session.
        Switched the two `resultaat.pose_gevonden`/`resultaat.tijd` reads inside this
        class to the real `FrameResult` fields (`result.pose_found`/`result.time`)
        instead of the Dutch aliases, matching the Phase 6/7 precedent of preferring
        real names over aliases in freshly translated code.
        **Deliberately left Dutch** (checked individually, same judgment call as
        session 3's `bieb`/`schaatser_id`/etc.): `deinterlacen` (both the constructor
        parameter and the attribute) -- confirmed via grep this is genuinely shared
        vocabulary: `AnalysisWorker.__init__` (already "translated" in session 3) still
        takes a `deinterlacen=` keyword, `MainWindow` has its own separate
        `self.deinterlacen` instance attribute, and `FragmentKiezer`/`BekijkVenster`
        pass it positionally into `VideoPlayer.load(...)` too -- renaming it here would
        split one concept across `MainWindow`'s eventual session. `huidige_idx` for the
        same reason and more so: `MainWindow` declares its own read-only property
        `def huidige_idx(self): return self.speler.huidige_idx` and then uses
        `self.huidige_idx` as if it were its own attribute in ~15 more places inside the
        still fully-Dutch skeleton-editor section of `MainWindow` (~8100-8400) --
        renaming the player's copy alone would desynchronize the property's name from
        what it proxies. `resultaten`/`video_pad`/`video_info`/`info` per the
        project-wide lower-bar convention (`resultaten` and `_pad`-suffixed names stay
        Dutch everywhere; `video_info`/`info` were already English). Left the plain Qt
        widget handles alone too (`self.label`, `self.slider`, `self.cap`,
        `self.btn_*`, `self.chk_*`, `self.lbl_*`, `self.combo_*`) except where a name
        was actually a Dutch word carrying real meaning (`chk_skelet`→`chk_skeleton`,
        `chk_afzetbeen`→`chk_push_leg`, `lbl_tijd`→`lbl_time`, `lbl_snelheid`/
        `combo_snelheid`→`lbl_speed`/`combo_speed`, `btn_frame_terug/_verder/_eind`→
        `btn_frame_back/_forward/_end`, `combo_teken`/`btn_teken_terug`/`btn_teken_wis`→
        `combo_draw`/`btn_draw_undo`/`btn_draw_clear`) -- matches the precedent already
        set by `TargetPicker` (session predates this log but is in the committed code:
        `self.label`, `self._pix` stayed put while `_box`/`_drag_start`/`_pan_start`
        were translated). All Dutch **strings** (tooltips, button labels, status-bar
        text) untouched throughout -- that's 8b/8c.
        **One mechanical-rename side effect worth knowing about for later sessions**:
        the whole-file regex pass also rewrites the *word* wherever it's used as plain
        prose inside a still-Dutch comment, not just where it names the actual
        identifier (e.g. a comment explaining a dialog "closes via accept()/reject()"
        used the literal word `sluit` as an ordinary verb, and the mechanical pass
        turned that into nonsense before the manual translation pass overwrote it with
        real English anyway). Caught here only because this session immediately
        followed the rename with a full hand-translation of every comment in range;
        a future session that runs a similar mechanical pass over a *wider* range than
        it intends to hand-translate in the same sitting should re-read the affected
        comments before leaving them, not trust the script's output as prose.
        **Verified**: `py_compile` across the repo; real `import skate_gui` under both
        venvs (offscreen Qt); a throwaway script driving `VideoPlayer` end-to-end
        against a fake capture (`load`/`go_to`/`crop_norm`/the `on_frame_shown` hook/
        `_set_zoom`/`_reset_zoom`/the `edit_mode` property/`play`/`pause`/`is_playing`/
        `release`) -- all pass under the new names; grepped the whole file for every
        renamed identifier's old spelling (zero hits) and for the new names' external
        call sites (`VideoPlayer(`, `.on_frame_shown`, `.overlay_drawer`, `.edit_mode`,
        `.follow_frozen`, `.combo_speed`, `.lbl_speed`) to confirm every external
        assignment/keyword-call site was updated consistently;
        `.venv-yolo\Scripts\python.exe schaats_schermtest.py` (quick mode) -- all 16
        windows pass, including the three `VideoPlayer`-backed windows
        (`FragmentKiezer`, `BekijkVenster` x2, `MainWindow` analysis/compare pages).

      - **Session 6** — `SpelerToetsen`→`PlayerKeys`, `MasterKlok`→`MasterClock`
        (~3742-4046 at session 5's end state): full identifier/docstring/comment
        translation of both classes, plus the module-level constants that are their own
        dependencies noted as still-Dutch in session 1's list: `SPOEL_FACTOR`/
        `SPOEL_TICK_MS`→`SCRUB_FACTOR`/`SCRUB_TICK_MS`, `ALLES_TICK_MS`→`ALL_TICK_MS`
        (`VIDEO_TOETSEN_HULP` was already done in session 5, as `VIDEO_KEYS_HELP`). This
        also closed out the two remaining Dutch "vooruitlezer" mentions session 4 had
        flagged as living in these classes' comments.

        `PlayerKeys` constructor params: `venster/spelers/actief/op_afspelen/op_spoel`→
        `window/players/active/on_play/on_scrub` (`extra` and `is_playing` were already
        English, unchanged). `MasterKlok`'s `op_klaar`→`on_done` (`parent`/`factor`
        already English). Unlike session 3's `bieb`/`schaatser_id`/etc or session 5's
        `deinterlacen`/`huidige_idx` (left Dutch as shared vocabulary spanning tens to
        hundreds of call sites in not-yet-translated `MainWindow` code), these
        constructor parameters are passed by keyword from only **4** call sites total for
        `PlayerKeys` (`FragmentKiezer`, `BekijkVenster`, `MainWindow` x2) and **2** for
        `MasterKlok` (`BekijkVenster`, `MainWindow`) — grepped and confirmed via
        `extra=`/`op_spoel=`/`actief=`/`op_afspelen=`/`is_playing=`/`factor=`/`op_klaar=`
        before deciding, same bounded-touch judgment call as session 2's three picker
        renames, not the wider "shared vocabulary" exception. All 6 call sites updated in
        the same commit (Pattern G, applied within a single file rather than across a
        module boundary — same variant session 5 used for `VideoPlayer`'s own public
        surface).

        Methods renamed and every external call site fixed (grepped per name before and
        after): `losmaken`→`detach`, `start_spoelen`/`stop_spoelen`→`start_scrubbing`/
        `stop_scrubbing`, `_spoel_tick`→`_scrub_tick`, `_actieve_spelers`→
        `_active_players`, `_loopt`→`_is_running`, `_pauzeer`→`_pause`, `_meld`→`_report`
        (all `PlayerKeys`); `loopt`→`is_running`, `herijk`→`recalibrate` (`MasterKlok`;
        `start`/`stop`/`_interval_ms`/`_tick` were already English). Internal attributes
        translated throughout both classes (`_venster`→`_window`, `_spelers`→`_players`,
        `_richting`→`_direction`, `_lopend`→`_running`, `_factor_bron`→`_factor_source`,
        etc.) — confirmed via grep that every one of these (all leading-underscore,
        private) has zero external readers, unlike the `self.speler`/`self.klok`/
        `self.toetsen` attribute names that *hold instances* of these classes in
        `FragmentKiezer`/`BekijkVenster`/`MainWindow`, which were deliberately left Dutch
        (same precedent as session 5 leaving `self.speler` pointing at a `VideoPlayer`
        instance) since renaming those belongs to each owner class's own future session.
        Local variables inside method bodies translated too (`verstreken`→`elapsed`,
        `stap`→`step`, `doel`→`target`, `klaar`→`done`, `soort`→`kind`, `toets`→`key`,
        `laatste`→`last`, `vanaf`→`start`, etc.) — `p.resultaten`/`p.huidige_idx` (the
        `VideoPlayer` properties these classes read) were left as-is, matching the
        project-wide convention that those two names stay Dutch everywhere until
        `VideoPlayer`'s own already-translated code is the only place that defines them
        (it is; these classes just read them).

        Three bare mentions of the old class names in still-Dutch prose elsewhere in the
        file (session 4/5's own docstrings and comments inside `VideoPlayer`,
        `BekijkVenster`, `MainWindow`) were updated to the new names as plain factual
        pointers, without translating the surrounding still-Dutch sentence — same
        judgment call as leaving a comment's prose alone while fixing a renamed
        identifier it happens to reference by name.

        **Verified**: `py_compile`; real `import skate_gui` under both venvs (offscreen
        Qt); a throwaway script (fake `VideoPlayer`-shaped objects, no real video/model)
        driving `PlayerKeys` through a real `QApplication` event filter — space
        play/pause, arrow-key stepping, Home/End, the `extra` override, a focused
        `QLineEdit` swallowing its own keys, a modifier key passing through untouched,
        and `start_scrubbing`/`stop_scrubbing` actually advancing a fake player's frame
        over real wall-clock time — and `MasterClock` end-to-end (`start`/`is_running`/
        `recalibrate`/`stop`, frames advancing via `show_on_clock`, landing exactly on
        the last frame and firing `on_done` when a run reaches the end); grepped the
        whole file for every old name (`SpelerToetsen`, `MasterKlok`, `SPOEL_FACTOR`,
        `SPOEL_TICK_MS`, `ALLES_TICK_MS`, `stop_spoelen`, `start_spoelen`, `_spoel_tick`,
        `.losmaken(`, `.loopt(`, `.herijk(`) — zero hits repo-wide outside this progress
        document itself; `.venv-yolo\Scripts\python.exe schaats_schermtest.py` (quick
        mode) — all 16 windows pass, including both `BekijkVenster` variants and the
        `MainWindow` compare page, which exercise both renamed classes end-to-end.

      - **Session 7** (first session under the "maximum speed" process amendment above) —
        `VergelijkKant`→`CompareSide`, `FragmentBalk`→`FragmentBar`, `FragmentKiezer`→
        `FragmentPicker`, `PuntenBalk`→`PointsBar`, `BekijkKant`→`ViewSide`,
        `BekijkVenster`→`ViewWindow`, `LokaalProef`→`LocalProbe`, `KopieerWorker`→
        `CopyWorker`, `KopieerDialoog`→`CopyDialog`, plus `_tijd_tekst`/`_lees_tijd`/
        `_bytes_tekst`/`_resterend_tekst`/`KOPIEER_VENSTER_S` →
        `_time_text`/`_read_time`/`_bytes_text`/`_remaining_text`/`COPY_WINDOW_S`
        (~4051-5417, one whole-file mechanical rename pass + per-class docstring/comment
        translation). Also renamed, scoped to this range only (same Dutch name is reused
        independently by later still-Dutch classes, so a file-wide regex would have been
        unsafe): the `KLIK` signal → `CLICKED` and each class's own `KLEUR_*`/`HOOGTE`/
        `RAAK_PX`/`PANEEL_BREEDTE` constants → `COLOR_*`/`HEIGHT`/`HIT_PX`/`PANEL_WIDTH`,
        `_klik_op_rij`/`_klik_op_balk`/`_frame_getoond`/`_bevestig`/`_werk_bij`/
        `_verwijder_selectie`/`_zet_sync`/`_toon_sync_label` → `_click_row`/`_click_bar`/
        `_frame_shown`/`_confirm`/`_refresh`/`_delete_selection`/`_set_sync`/
        `_show_sync_label`. `CopyWorker`'s and `LocalProbe`'s Signals renamed too
        (`voortgang`/`klaar`→`progress`/`done`, `gemeten`→`measured`, matching session 3's
        precedent for `AnalysisWorker`/`BatchWorker`), with the one external connect site
        each has in `MainWindow` updated. Left Dutch (same "shared vocabulary spanning
        MainWindow" judgment as sessions 3/5): `toon`, `analyse_id`, `heeft_analyse`,
        `leeg`, `naar_sync`, `events`, `sync_frame`, `naam`, `gedaan`, `fragmenten`,
        `bron_pad`, `_spring`, `_ga_naar_tijd`, `_verwijder`, `punten`, `bron`, `bieb`.
        **Found and fixed two real, pre-existing bugs**, both from session 5's whole-file
        mechanical regex pass having rewritten common Dutch verbs used as ordinary prose
        elsewhere in the file (the exact risk flagged in session 5's own log): (1) a
        **user-facing tooltip** on the ViewWindow "▶ Start alles" button literally read
        "Spatie is_playing ook beide tegelijk..." (should be "speelt", Dutch for "plays")
        — `speelt`→`is_playing` had been applied file-wide, not scoped to VideoPlayer;
        (2) a comment in `MainWindow` read "...een plaats-reeks release die eerst netjes
        af" (should be "sluit", Dutch for "closes") from the same over-broad `sluit`→
        `release` pass — reverted both to correct Dutch (not translated, since neither
        section is due for translation yet). Also fixed one stale cross-reference each in
        `skate_db.py` (`BekijkVenster`→`ViewWindow` in a docstring) and
        `schaats_schermtest.py` (`G.FragmentKiezer`/`G.BekijkVenster` call sites, which
        would otherwise `AttributeError` on the next screen-test run).
        **Verified once, at the end of the whole chunk** (per the process amendment):
        `py_compile`; real `import skate_gui` under both venvs; a repo-wide grep for
        every old name (zero hits outside this progress document); a single throwaway
        script covering the classes with actual behavioral risk — `CopyWorker`/
        `CopyDialog`'s renamed `progress`/`done` signals, `LocalProbe`'s renamed
        `measured` signal, `FragmentBar`/`PointsBar`'s `CLICKED` signal + renamed color
        constants (incl. a real `mousePressEvent` hit-test), and `CompareSide`'s renamed
        private methods — all pass; `schaats_schermtest.py` full run (not just quick
        mode, since this was the true end of a multi-class chunk) — all 16 windows pass,
        including `FragmentPicker` and both `ViewWindow` variants.

      - **Next up for 8a**, in order (line numbers will have drifted after session 7 —
        re-`grep -n "^class "` first, don't trust these verbatim):
        - `MainWindow` (~5413-8877, ~3,460 lines — will very likely need to be split
          across more than one session by itself; it owns most of the ~150
          `QMessageBox` calls and ~178 tooltips that 8c is nominally about, so some
          8c work will naturally happen while translating its identifiers/docstrings/
          comments here in 8a — that's fine, just don't call 8c "done" until a
          dedicated pass confirms every dialog/tooltip string is translated too. This is
          also where `bieb`/`schaatser_id`/`titel`/`instellingen`/`aangemaakt_door`/
          `analyse_id`/`naam`/`geboortejaar`/`notities`/`taken`/`schaatser_naam` from
          session 3's "deliberately left Dutch" list, and `deinterlacen`/`huidige_idx`
          from session 5's, actually live and could finally be renamed in one
          coordinated pass, if a future session decides that's worth doing -- check
          first whether `skate_db.py`'s own parameter names would need to move
          together for `deinterlacen`/etc., since right now they match on purpose).
        - `main()` (~8877-8898).
        Line numbers are from session 7's end state and will drift as each chunk is
        translated — re-`grep -n "^class \|^def "` at the start of each session rather
        than trusting the numbers above verbatim.
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
| kader (user-drawn box) | box | Done in `skate_yolo.py` (Phase 6) for every *internal-only* name: `KADER_MAAT_MAX`→`BOX_SIZE_MAX`, `KADER_MIN_HOOGTE_PX`→`BOX_MIN_HEIGHT_PX`, `_Kijkglas.start_kader`→`start_box`, etc. **`doel_kader` the parameter itself stays Dutch** — superseding the `target_box` this row originally planned — because `schaats_gui.py`'s two `analyze()` call sites pass it as a keyword argument (`doel_kader=...`); see the `bocht`/`doel_punt`/`perspectief`/`waarschuwing_callback` note in the Phase 6 entry above. Still Dutch in `schaats_gui.py` itself (Phase 8). |
| kijkglas (small-target tracking) | spyglass | `_Kijkglas`→`_Spyglass`. Claude's own naming call (low-stakes internal class), not user-specified. Done in Phase 6 (`skate_yolo.py`) — the local variable holding an instance is also `spyglass` now (was left as `kijkglas` in an early pass of this phase; fixed the same session, see "Patterns established" note on residual locals). |
| doel / doelpunt / DoelTracker | target / target_point / TargetTracker | Done in Phase 4. |
| kamtanden / kam_masker | combing / comb_mask | Standard video-engineering term. Done in Phase 4 (`_comb_mask`, `deinterlace`). |
| opname | recording | `OPNAME.md`→`RECORDING.md` (Phase 11). |
| knippen / fragment | trim/cut / fragment | `knip_fragmenten`→`trim_fragments`. Done in Phase 4. |
| bibliotheek | library | Done in Phase 5 (`library_path`, etc). |
| instellingen | settings | DB column `instellingen_json`→`settings_json` done (Phase 5); the JSON *content*'s keys are not (deferred to Phase 8, see below). |
| kalibratie / verdwijnpunt | calibration / vanishing_point | Done in Phase 3. |
| baanlijn / dwarslijn | track_line / cross_line | Done in Phase 3. |
| pakkleur | suit color | Done in Phase 6 (`KleurReferentie`→`ColorReference`, `KLEUR_MATCH_MIN`→`COLOR_MATCH_MIN`, etc.). |
| onvolledig / afgekapt / geen volledige push | incomplete / truncated / no full push | Done in Phase 4 (`INCOMPLETE_TRUNCATED`, `INCOMPLETE_NO_PUSH`). |
| nog doen / bezig / klaar / onbruikbaar (recording status) | todo / in_progress / done / unusable | Done in Phase 5 (`SOURCE_STATUSES`), including the DB value migration. |
| keten (tracklet chain) | chain | Done in Phase 6. `_stik_keten`→`_stitch_chain`, `keten_gekoppeld`→`chain_linked`, `_Kijkglas.bron` values `'kader'`/`'keten'`→`'box'`/`'chain'`. |
| klik / klik_gemist | click / click_missed | Done in Phase 6. |
| verfijnen (top-down re-estimation pass) | refine | Done in Phase 6. `_verfijn_landmarks`→`_refine_landmarks`, etc. The `analyze()` parameter (`verfijn=True`) was safe to rename to `refine=True` — confirmed via grep that no untranslated caller passes it by keyword. |
| poort (a gate/threshold check) / koppeling (linking two tracks) | gate / link | Done in Phase 6, including the `'veto'`/`'match'` gate-mode strings (already English) and the `_Spyglass` log/outcome text — the `'link'` substring is load-bearing: `analyze()` and the self-test both check for it inside a reason string. |

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

### Pattern G — plain function keyword arguments (Pattern C's trap, without a dataclass)

Pattern C's "known trap" (a rename fixes attribute *access* but not constructor
*keyword arguments*) applies just as much to an ordinary function with no dataclass in
sight. `skate_analysis.py`'s `analyze()` and `skate_yolo.py`'s `analyze()` are both
called from `schaats_gui.py` (untranslated) with several arguments passed by keyword
(`AnalyseWorker.run`/`BatchWorker.run` in `schaats_gui.py`, both calling through the
`analyseer_backend()` indirection at `schaats_gui.py:626`). Renaming any of those
*parameter names* breaks the call immediately with a `TypeError`, not a
silent-failure-class bug — so this one at least fails loud, but it fails loud only once
someone runs the actual GUI, since none of `skate_yolo.py`'s own verification (its
self-test, a bare import) ever calls `analyze()` the way `schaats_gui.py` does.

**Rule of thumb**: before renaming a public function's parameters, grep every call
site across the whole repo for `functionname(` and check whether any of them pass
arguments by keyword using the old name — not just whether the function itself is
imported under an alias (Pattern A/B only fix *that* the name resolves, not what
keywords its signature accepts). Confirm with
`inspect.signature(fn).bind(*args, **kwargs)` using the *exact* call sites found, since
`bind()` raises the same `TypeError` a real call would without needing to actually run
the function. Both `skate_yolo.analyze()` call sites were checked this way in Phase 6
before relying on it.

For parameters no untranslated file calls by keyword (confirmed by the same grep
coming up empty), rename freely — e.g. `skate_yolo.py`'s `verfijn=True` became
`refine=True` because nothing outside the file passes `verfijn=`.

### Note: earlier "fully translated" phases still leave many local variables Dutch

Before assuming Phase 4/5 achieved a 100%-pure translation and calibrating Phase 6+'s
effort to match that (nonexistent) bar, grep the *supposedly finished* files for
plain-language local variable names. `resultaten` (87 hits in `skate_analysis.py`, 17
in `skate_db.py`), every `_pad`-suffixed name (40 + 25 hits), and `naam` (11 + 29 hits)
are all still untranslated **inside function bodies** in files whose public API
(function/class names, parameters that cross no external boundary, docstrings) is
fully English. This isn't a defect to fix retroactively — it's the actual, lower bar
those phases were held to, apparently because a 2000+-line file's every single local
variable isn't worth the added risk/time for a batch/analysis tool nobody but a
developer reads the source of. Phase 6 followed the same bar: public identifiers,
docstrings, comments, and most meaningful locals got translated, but `resultaten` and
`_pad` names were deliberately left alone throughout `skate_yolo.py` too, for
consistency with the rest of the (still Dutch-parameter-name) codebase rather than as
an oversight. Do the same grep-first check before Phase 7/8/9 rather than assuming the
already-committed phases set a stricter precedent than they actually did.

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
