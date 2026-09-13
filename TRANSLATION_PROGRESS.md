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
