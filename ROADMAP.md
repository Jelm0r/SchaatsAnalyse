# SkateAnalysis Roadmap

Plan for the next development phase, in build order. Decisions made (July 2026):

- **Sharing**: one shared cloud folder (OneDrive/Dropbox/network drive) containing a SQLite file + media files. No server, no accounts.
- **Videos**: copied along into the shared folder, so every trainer can watch the analysis together with the footage.
- **Scale**: one team (±5–30 skaters, 1–5 trainers). The design can lean on that; no permissions/roles system needed.
- **Skeleton editor**: corrections flow out to neighboring frames with an **adjustable window** (0 = only the edited frame).
- **Recording setup (assumption since July 2026)**: skaters are **always filmed straight-on from the front** and the camera is **always exactly horizontal**. That removes the need for camera-tilt correction (horizon) and perspective correction from the daily workflow. Phases 5 and 7 remain as **nice-to-haves** for a possible later setup (tilted/wobbling camera), but are **not a priority right now**.

The phases build on each other: 0 → 1 → 2 can't swap order; 3 (skeleton editor) and 4 (sharing) can then be built independently of each other. Phase 6 (faster analysis) is independent of the rest and can happen any time, even earlier. Phase 5 (horizon tracking) and 7 (perspective correction) are, under the fixed recording setup, **nice-to-haves** (see above); phase 7 shares building blocks with phase 5 (marking/tracking lines) and includes horizon correction as a special case. **Phase 8** (trimming fragments in the app + the recordings themselves in the library, requested 10 August 2026) was likewise independent and has been **done since 11 August 2026**; on the analysis side it leans entirely on the existing batch flow from phase 1 and added one schema bump (`source_video`, v3) for the work list of not-yet-trimmed recordings. That makes the **next** phase either phase 6 (faster analysis) or phase 2 (profile view, once the measurement is done).

---

## Phase 0 — Preparation: making results serializable ✅

> **Done (16 July 2026)** — verified on a real video through the GUI: save, restart the GUI, reload → identical table/graph/overlay, in a second instead of a full detection run.
>
> **What's there:**
> - `skate_analysis.py`, section "Serialization (phase 0)" before `segment_pushes`: `results_to_arrays()` / `arrays_to_results()` + `save_landmarks()` / `load_landmarks()` (`np.savez_compressed`).
> - CLI: `--save-npz PATH` (write out landmarks after the analysis) and `--from-npz PATH` (skip detection, only recompute derivatives + draw the overlay; no model needed).
> - GUI: buttons "Save landmarks (.npz)" (data panel) and "Load landmarks (.npz)..." (start page). **Temporary scaffolding** — phase 1 replaces manual file picking with the library; the functions underneath stay the same.
> - `_analysis_done` is split: the display part is now `_show_results(info, resultaten, events, source=None)`, shared by a fresh and a loaded analysis. **This is the seam phase 1 reuses.**
>
> **Deviations from the plan below:**
> - Step 2's "plain landmark type" turned out to already exist (the `Landmark` namedtuple, `skate_analysis.py` line ~51, already used by the YOLO backend) — no new type needed.
> - `arrays_to_results(arrays)` takes **no** `info` argument: the video metadata (w/h/fps/total) travels inside the `.npz` and comes back out as `VideoInfo` → `(VideoInfo, resultaten)`. Reloading therefore doesn't require the video.
> - Step 4 held up: `process_derivatives()` had no hidden dependency on the detection pass.
>
> **Not serialized:** the perspective calibration (phase 7). It's off on load — harmless under the frontal-camera assumption.

Everything after this stands or falls on being able to **save and reload** an analysis. Right now the `FrameResult` list only lives in the GUI's memory, and the `lm` field holds a MediaPipe landmark object that can't be written to disk directly.

**To build (in `skate_analysis.py`):**

1. `results_to_arrays(resultaten)` → dict of numpy arrays:
   - `landmarks`: `(n_frames, 33, 3)` float32 — normalized x, y, visibility (z is unused anywhere);
   - `pose_found`: `(n_frames,)` bool; `horizon_deg`: `(n_frames,)` float32.
   - Derivatives (leg/angle/weight/events) **not** saved as the source of truth: they're recomputable from the landmarks via `process_derivatives()` + `segment_pushes()`. They are cached (see phase 1) for fast display.
2. `arrays_to_results(arrays, info)` → a fresh `FrameResult` list with a **plain landmark type** (e.g. a small `Landmark` dataclass with `x/y/visibility`) instead of the MediaPipe object. All existing code (`get_landmarks`, `draw_all_landmarks`, smoothing) only reads `.x/.y/.visibility`, so this works without further changes — worth verifying, though. The YOLO backend (`_coco_to_landmarks`) already builds its own landmark objects, so this brings both backends in line.
3. Storage as **`.npz`** (`np.savez_compressed`): a 3000-frame video is ± 1–2 MB. A separate file next to the video, not a blob in SQLite — keeps the database small and cloud-sync-friendly.
4. `process_derivatives()` must be callable on its own on a reloaded list (it essentially already is — check there's no hidden dependency on the detection pass).

**Done when:** writing an analysis to `.npz`, restarting the GUI, reloading it, and seeing an identical table/graph/overlay — without re-analyzing the video.

---

## Phase 1 — Profiles + database ✅

> **Done (18 July 2026)** — self-test (`python skate_db.py`) + headless GUI smoke test green in both venvs; the full cycle (create skater → analyze → reopen) works.
>
> **What's there:** `skate_db.py` (config, schema `user_version=1`, CRUD, `save_analysis` with video+npz first and the DB insert as the last step, self-test); GUI start page = the library (skaters on the left, analyses on the right from the events cache, double-click = open, rename/delete with confirmation); `NewAnalysisDialog`; `AnalysisWorker` automatically saves after the analysis (in the worker thread, with a "Saving to library..." phase); reopening via the phase-0 seam with the **saved** `smooth_n`/`threshold` from `settings_json`. The phase-0 scaffolding buttons (save/load npz) are gone.
>
> **Deviations from the plan below:**
> - The settings group box moved into `NewAnalysisDialog` (no third stack page — the flow was already a chain of modal dialogs).
> - Pulled forward from phase 4 (lightly): library-path config in `%APPDATA%\SchaatsAnalyse\config.json` + a "Library folder..." button, and the cloud-safe SQLite discipline (journal DELETE, short per-call connections, busy_timeout, relative paths with forward slashes). The library can therefore already live in a shared cloud folder (Google Drive/OneDrive/Dropbox); phase 4 only adds trainer name + conflict detection.
> - Only if saving itself fails does the (long) analysis stay visible with a warning — it's then just not saved.
> - ~~The perspective calibration is not serialized (like phase 0)~~ — **superseded since 11 August 2026**: the calibration input (the traced lines + parameters) travels in `settings_json` and the camera pose is recomputed on open. See phase 7 below. Analyses from before that date still get the old message.
> - Settled open questions: video is always copied (the original stays put, filename kept in the uuid folder); no import of old loose analyses.

**Goal:** every skater gets a profile; every analysis belongs to a profile.

### Storage layout (the "library")

One folder, shareable later via the cloud (phase 4). Path configurable; local by default, e.g. `Documents\SkateAnalysis`:

```
<library>/
  skate.db                   ← SQLite: profiles, analyses, events cache
  media/
    <analysis-id>/
      video.mp4              ← copied original (keeping the name is fine)
      landmarks.npz           ← smoothed landmarks (phase 0)
      landmarks_raw.npz       ← same, before manual edits (phase 3)
```

`analysis-id` = a UUID, so two trainers never create clashing folder names.

### Database schema (stdlib `sqlite3`, works in both venvs, no new dependency)

```sql
skater(id, name, birth_year, notes, created_at)
analysis(id TEXT PRIMARY KEY,          -- UUID, also the folder name under media/
        skater_id, title, date,
        video_file,                     -- relative path within the library
        w, h, fps, total_frames,
        backend,                        -- 'yolo' | 'mediapipe'
        settings_json,                  -- target_point, horizon, smooth, ...
        created_by,                     -- trainer name (free text, see phase 4)
        edited,                         -- 0/1: are there manual skeleton edits
        created_at)
push_event_cache(analysis_id, idx, leg, start_frame, end_frame,
                  angle, min_angle, max_angle, note)
```

`push_event_cache` is purely for fast list display ("last analysis: avg. 42°") without loading the `.npz` first; opening an analysis recomputes everything fresh from the landmarks.

New module **`skate_db.py`**: `open_db(path)` (creates the schema if needed), CRUD for skaters/analyses, `save_analysis(...)` (copies the video + writes the npz + inserts in one transaction), `load_analysis(id)`. Keep all SQL here; the GUI only talks to this module.

### GUI changes (`skate_gui.py`)

- **New start page = the library**: the skater list on the left (+ a "new skater" button), that skater's analyses on the right (date, title, number of pushes, avg. angle from the cache). Double-click → open the analysis.
- **"New analysis" flow**: choose (or create) a skater → choose a video → the existing `TargetPicker` → `AnalysisWorker` runs → on `done` automatically save to the library → open the analysis view. The existing view page stays largely unchanged; only `cap_display` now reads the copied video from `media/<id>/`.
- **Seam from phase 0**: opening an analysis = `load_landmarks()` + `process_derivatives()` + `segment_pushes()` → `_show_results(...)`. That path already works (the temporary "Load landmarks" button does exactly this); phase 1 only swaps the file dialog for the library selection and then removes both temporary buttons.
- Rename/delete an analysis (delete = DB row + media folder, with confirmation).
- Video copying can take a while for large files → do it in the worker thread, not the UI thread.

**Done when:** the full cycle works — create a skater, analyze a video, close, reopen, watch the analysis back from the profile.

---

## Phase 2 — Enrich the profile view

> **Waiting on the measurement (decided 8 August 2026).** This phase is about *interpreting* skating technique: progress over time, notes, export. That's only valuable once the underlying measurement is solid — a progress graph of angles that still shift with every tracking improvement is misleading, and old analyses would end up looking different from new ones. Phase 2 is therefore only picked up **once the analysis is finished**; it's technically a small job, but not the next one.

Small but worthwhile follow-up to phase 1 (can also come later):

- **Progress over time**: a small graph per skater with the average/best push angle per analysis date (the data's already in `push_event_cache`).
- A notes field per analysis ("practiced left corners, headwind").
- CSV export per skater (all analyses) alongside the existing per-analysis export.

---

## Extra — App version per analysis + info tab ✅

> **Done (6 August 2026)** — self-test (`python skate_db.py`) green in both venvs, version manually checked against `git log -1`, and a headless smoke test of the dialog + the library tooltip (also with an analysis from before this feature).
>
> **Goal:** during development the tracking logic changes regularly; for a saved analysis you need to be able to see afterward which version of the app it was made with, so a strange measurement can be explained ("this was done with the old L/R fixer").
>
> **What's there:** `skate_db.app_version()` reads the git repo next to the script (`git log -1 --abbrev=8 --format=%h%x09%cs` + `git status --porcelain -uno`) and returns `{commit, date, dirty, label}`, cached once per process; outside a repo everything is empty. `save_analysis` sets `app_version`, `app_commit` and the full `backend_name` itself in `settings_json` (`setdefault`, so a caller that fills them in wins) — one place, so a single analysis, a batch, and the self-test all record all three without having to think about it. New in the GUI: an **"ℹ Info..."** button in two places — in the transport bar next to "Edit"/"⇄ Compare with..." (the open analysis) and on the start page next to "Open" (the selected row, so without having to open the analysis) → `AnalysisInfoDialog` with title, skater, analysis date, creator, app version, backend, video format, whether it's been manually edited, and the settings (smoothing, threshold, skip corner, horizon, perspective; the heavy-model row only for a MediaPipe analysis, since the YOLO backend has one model and ignores that flag). The values are selectable so the hash can be copied. The library list shows the version as a second line in the existing title tooltip.
>
> **Choices:** the open question of hash-vs-label became **both, but both automatic**: the label is `commit-date · short hash` (`2026-08-05 · 7e013fb7`), with a `+` if there were uncommitted changes. The commit date is the readable part for a trainer, the hash the precise part for running `git show`. A manually bumped `APP_VERSION` constant was deliberately dropped: it lags behind during fast development and then lies. Untracked files don't count as "dirty" — videos and npz files next to the code say nothing about the logic that ran. `skate_db` also got an `analysis_meta()` (the DB row + parsed settings, **without** reading the npz), and `list_analyses` includes the settings; the Info button fetches its metadata there fresh instead of a copy living on `MainWindow`.
>
> **Deliberately not done:** no schema bump and no dedicated column — this rides on the existing JSON field. Analyses from before this change therefore get no version retroactively; those show "unknown (from before this feature)" and keep exactly their old tooltip in the library. The version is **not** updated on a manual skeleton edit: it records what version the analysis was **run** with (that an analysis has been edited is shown separately in the dialog).

---

## Extra — Library list: video duration instead of pushes/angle ✅

> **Done (6 August 2026)** — headless smoke test of the library page (columns, row buttons, row height) in the YOLO venv + `python skate_db.py` green.
>
> Implemented as described below, with two additions that came out of using it:
> - **Duration and push count in one column**: `"5.4s (4 pushes)"` (above a minute, `"1:23 (12 pushes)"` — nobody reads "83.2s" as a minute and a half). So the push count didn't need to disappear; it just no longer sits where the eye lands first. Formatting in `_duration_text()` (`skate_gui.py`); `list_analyses()` only needs to supply `total_frames`+`fps` extra for it.
> - **The per-analysis buttons move onto the row itself** (`_make_row_buttons` → `setCellWidget` in the fourth column): "Open", "ℹ Info...", "Rename...", "Delete" belong to one analysis, while "New analysis..."/"Batch analysis..."/"Compare skaters..." are library-wide — those used to sit together in one row at the bottom. Each button carries its own analysis id (a default argument in the lambda, otherwise every row would capture the last loop value), so a click acts on its own row rather than the table's incidental selection. `_rename_analysis`/`_delete_analysis` were given `(aid, title)` arguments for that, with the old selection route as a fallback. Side effect: the bottom button row went from seven buttons to three, which brings the **window minimum from 1738 → 1406 px** wide (measured headless).

**Goal:** the analysis table on the start page (`tabel_analyses`) currently shows "number of pushes" and "avg. angle" per row. Those two columns aren't what a trainer looks at first; **how long the video runs** (seconds) is more useful, and the average angle can go away entirely.

**Approach:**

- **Duration instead of push count**: the duration (`total_frames / fps`) is already in the `analysis` table (phase 1 schema, columns `total_frames`+`fps`), so this is a pure display change in `list_analyses()` (`skate_db.py`) + the column setup in `skate_gui.py` — no schema bump, no new calculation. Format as `m:ss` (or `s` for short clips).
- **Remove the avg. angle column**: drop the column from `tabel_analyses`. The underlying calculation (`AVG(angle)` with the `INCOMPLETE_MARKERS` filtering in `list_analyses`) can stay for phase 2's progress graph — only the column in *this* table disappears.
- **Push count**: might still be useful elsewhere (e.g. as a tooltip), but isn't needed as a column once duration is there — to be decided whether it disappears entirely or stays alongside the duration.

**Done when:** the library table shows title/date/duration (and optionally push count) per analysis, without the average-angle column.

---

## Extra — Corner detection: the corner is no longer analyzed ✅

> **Done (5 August 2026)** — outside the phasing. The corner cost analysis time without ever producing a usable measurement; both problems are now solved.
>
> **What the signal is:** `corner_ratio` in `skate_analysis.py` = **hip width / torso length** (shoulder midpoint→hip midpoint), in pixels. Scale-free, like `_extension_ratio` — and specifically sensitive to the rotation around the vertical axis that a corner causes: frontal, the hips sit side by side; in the corner they sit front-to-back while the torso stays the same length. **Measured over all 22 analyses in the library:** 18 frontal clips never dip below 0.57 (median 0.75–1.20); the corner section of four long clips sits at 0.21–0.24. Margin over 3×. Four alternative denominators (femur, whole leg, shoulder width, combinations) all gave less separation (1.7–2.7×). The classification (`determine_corner_sequence`, following the recipe of `determine_horizon_sequence`: Hampel → Savitzky–Golay → hysteresis 0.40 in / 0.50 out → runs < 0.6 s cleared) marks **zero** frames as corner on those 18 frontal clips.
>
> **Where the time savings come from:** `_CornerGuard` in `skate_yolo.py`. The detection pass is ~94% of the analysis time, so it had to skip the corner — and that couldn't be done with `model.track(source=path, stream=True)`, since ultralytics reads and infers every frame itself there. The loop now reads the frames itself (decoding is negligible, and this way the frame numbering stays exact) and only infers roughly every 0.3 s in the corner, exactly as proposed. The guard starts skipping after 0.5 s of corner evidence (someone in frame, but turned) or 3 s with nobody measurable at all — that 3 s deliberately sits above the longest detection gap on a straight section in the library (2.1 s, IMG_9001). Bystanders along the boarding stand frontal in view and would keep the analysis at full speed forever; that's why only a person who's moving **or** growing counts (a skater heading straight at the camera barely shifts in frame but grows ~18%/s).
>
> **Why the skipping is safe:** the refinement pass fills detection gaps up to `GAP_FILL_S` (1.0 s) with interpolated bboxes and still estimates the pose there top-down. The gaps the skipping leaves behind are 0.33 s, well within that. The corner is therefore **determined on the raw pass-1 landmarks, before refinement**: if the guard wrongly skipped a stretch, the next check frame within 0.33 s reports a frontal skater and the detection pass immediately returns to full speed. Too much skipping costs (almost) no coverage; too little skipping only costs time.
>
> **A check frame doesn't count as a measurement.** The frame that still gets inferred in skip mode gets a skeleton — but it sits in the middle of a stretch that otherwise wasn't looked at, so the neighboring frames a push would need to prove itself are missing. `_detect_all` therefore returns those frames in `excluded`, and `_corner_with_check_frames` keeps them at `corner=True` after each classification. Their *verdict* still counts (they're allowed to end the corner — that's what they're there for), their own angle doesn't. Without that rule, such a frame could clear itself on its own hip stance: mid-corner a skater sometimes briefly turns almost frontal.
>
> **Measured (5 Aug 2026):**
>
> | | | |
> |---|---|---|
> | **"Kim tempo"** (888 frames, 52% corner) | 2266 s → **1253 s** | **45% faster** (1.8×) |
> | ... pushes | 32 → 16 | the five angles of 74–87° at the end (the corner) are gone |
> | ... straight section (frame 0–427) | landmarks **identical** (median 0.00 px) | only the last 7 frames before the corner differ |
> | **"Schaats frontaal"** (crossing skaters) | corner on vs. off: **byte-identical** | 0 frames marked as corner |
> | **"7e ronde"** (portrait phone footage) | 0 corner, 100% coverage, same 5 pushes | no rotation problem from the own read loop |
> | **The own read loop itself** | with corner off: **byte-identical to the saved analysis** | replacing `model.track(source=...)` changes nothing |
>
> The angle differences left over on the straight section of Kim tempo (up to 3.8°, one fewer push) don't come from detection but from the measurement logic: the estimated stroke period (`EXTENSION_MIN_STROKE_FRAC`) and the L/R alternation check used to run partly over corner noise too. Setting the corner flag on the already-saved landmarks produces the exact same event list — so this is a gain, not a deviation.
>
> **What "corner" means for the measurement:** one line in `process_derivatives` — a corner frame gets no `lm_data`. Leg assignment, push completion, and event segmentation all build their segments on "pose and lm_data", so they automatically see the corner as a detection gap; nothing about the measurement logic changed. The skeleton is still drawn, with "CORNER — not measured" on screen. Marking instead of hard-cutting is needed because some clips start right **in** the corner (the last strokes of the previous lap), and because a video with multiple laps this way keeps producing every straight section.
>
> **Existing analyses** don't change on their own: their npz doesn't know the flag and it isn't computed retroactively on open. If you do want such an analysis cleaned up, the **"Determine corner"** button does that without re-analyzing — the landmarks for the whole clip are already there. It shows what it would do to the table first and only then asks; on "Kim tempo" that's 32 → 16 pushes in a fraction of a second instead of 21 minutes. Deliberately a button, not an automatism: it changes what the trainer already saw.
>
> **Further:** `FrameResult.corner` travels in the npz (old npz files load unchanged — the runtime skip can't be derived from landmarks, so it has to be stored); checkbox **"Skip corner (faster)"** (on by default) in both analysis dialogs; the coverage counter doesn't count corner frames as outstanding work and "⏭ Next gap" doesn't jump into them; `python skate_eval.py corner analysis.npz` prints the signal + the found segments. CLI: `--no-corner`.
>
> **Still open:** the thresholds are calibrated on four TV clips that end in the corner. There's still **no clip from our own setup that starts in the corner** — once there is one, recheck with `skate_eval.py corner` and adjust `CORNER_IN`/`CORNER_OUT` if needed. `CORNER_MIN_MOVEMENT` (the bystander filter) is the most likely second thing to tune.

---

## Extra — Batch analysis ✅

> **Done (20 July 2026)** — outside the phasing, on top of the phase 1 flow. Analyze several videos at once.
>
> **What's there:** `BatchAnalysisDialog` (videos + a skater/title per row + shared settings) and `BatchWorker` (`QThread`) in `skate_gui.py`. The target/horizon choice happens up front, per video, in `_new_batch_analysis`; the whole row then runs unattended and each analysis saves itself via `save_analysis`. One failed clip doesn't stop the batch (it's reported, the rest continues); "Stop after this video" = a clean stop between clips. Not a roadmap phase, but a logical follow-on to the library — noted here for that reason.

---

## Extra — Compare skaters (two analyses side by side) ✅

> **Done (27 July 2026)** — outside the phasing. Two saved analyses side by side to compare skaters (or the same skater at two points in time).
>
> **Groundwork — the video panel factored out.** All playback state lived as loose `self.*` attributes on `MainWindow`, so two players side by side was impossible. A new `VideoPlayer(QWidget)` with its own capture/timer/zoom/pan and all the controls; `MainWindow` still reaches `video_info`/`resultaten`/`huidige_idx` through read-only properties, so the existing editor and table code stayed unchanged. The skeleton editor hooks in via `overlay_drawer` + `on_mouse_press/_move/_release` (plain callables); panning stays inside the player, ahead of the callback, so the priority rule holds by construction. The analysis page then works identically — including a fix that came along for the ride: `slider.setRange` in `load()` is now wrapped in `blockSignals`, so a short analysis after a long one no longer produces a phantom seek.
>
> **What's there:** a "Compare skaters..." button on the start page → a third page with two `CompareSide` widgets (header, own `VideoPlayer`, "Choose analysis...", sync point, a minimal push table `#`/`Leg`/`Angle` with gray highlighting for incomplete pushes). `AnalysisPicker` (skater → analysis) is used twice on open and per side to switch. Each side can be played independently; **"Start all"** plays both from their sync point via **one master clock** that computes the target frame per side from wall-clock time × its own fps — self-correcting and correct across different fps (two separate timers would drift apart within seconds). Default ¼×. The shared load helper `_load_analysis_data` touches no `MainWindow` state, so comparing doesn't overwrite the settings of the currently open analysis.
>
> **Deliberately not yet:** sync points aren't saved (no DB column, would need a schema bump + migration); no combined statistic or graph across the two sides; the table is deliberately minimal — see what a trainer actually misses in practice first.

---

## Extra — Automatic zoom ✅

> **Done (28 July 2026)** — the program determines the zoom itself: the skater stays fully in frame with some air around them for the whole clip. Outside the phasing; part of the three loose improvement points from 27 July.
>
> **Why:** measured across the library, a skater grows 1.4–5.8× larger in frame over the course of a clip. One fixed zoom factor is therefore only right for a few seconds, and the trainer ended up riding the slider during playback. The trainer also doesn't want to have to choose the framing themselves — that's work the program can do.
>
> **What's there:** `box_sequence()` in `skate_analysis.py` computes, once at load time, an offline per-frame `(center, radius)` of the skater — up front rather than online, so scrubbing gives exactly the same crop as playing toward it. In `VideoPlayer`, `_zoom` (set, 1–5×) and `_zoom_eff` (applied, up to 8×) are kept separate; with the **"Automatic zoom"** checkbox (off by default), `_zoom_eff` follows the box radius plus 15% margin, and the crop follows the box center point. As long as the automation is on, the zoom slider and "Fit" are disabled; turning the mouse wheel hands control back. It lives in `VideoPlayer`, so the compare page gets it per side too — two skaters at different distances only really become comparable this way. See CLAUDE.md (the "Automatic zoom" bullet) for details.
>
> **What the measurement showed (on real analyses, every frame checked):**
> - Framing around `torso_centroid` wastes a quarter of the frame (that point sits high in the body) — hence the center of all *visible* landmarks instead: frame fill 0.40 → 0.56.
> - Smoothing alone flattens the peak, and then an outstretched leg falls outside the crop; first a running maximum over one stroke, then smoothing. Result: no frame with the skater outside the crop, at < 2% zoom change per frame.
> - `BOX_POLY=1` instead of the shared `SMOOTH_POLY=2`: the quadratic edge fit extrapolates the stroke wave and is 15% off on the first frame (linear: 3%).
> - The ceiling shouldn't sit on the video resolution but on the on-screen magnification: portrait phone clips already sit shrunk inside a landscape panel and may zoom in a lot, landscape 4K barely at all.
> - On a long detection gap the box held its last known framing — that produced an 8× close-up of the spot where the skater *used to be* (visible in the first second of IMG_9002). Now the box smoothly opens up to the full frame after `BOX_GAP_S`.
>
> **Deliberately not done:** no persistence (zoom state and the checkbox belong to watching, not to the analysis); no automatic decision on when the automation should be on — that stays a checkbox.

---

## Extra — Smoother comparing ✅

> **Done (28 July 2026)** — the two loose improvement points from 27 July on the compare page.
>
> **Comparing directly from an open analysis:** the **"⇄ Compare with..."** button in the transport bar of the view page (next to "Edit", via `add_control_button`). The open analysis always goes to the **left side** — from an open analysis there's never an empty side to be clever about, so predictable is better — and for the right side you're immediately asked for an analysis, preselected on the same skater (so "this skater then vs. now" is two clicks). If the right side already had a different analysis, it stays, sync point included; canceling the picker just opens the page with only the left side filled. The side **reloads the analysis from the library** rather than sharing the view page's result list: the skeleton editor mutates those `FrameResult` objects in place. For that, `_choose_compare_side` was split into the dialog and a reusable `_set_compare_side(side, analysis_id, name)`; the button is disabled as long as there's no saved analysis open (an unsaved analysis can't be loaded from the library).
>
> **Switching one side** turned out to already exist: every `CompareSide` had its own "Choose analysis..." button that only replaces that side (it came along with the `VideoPlayer` refactor, but was never taken off this list). What did need adding: the button reads **"Switch..."** once a side has an analysis, there's an **✕** button to clear a side (wired from the page so the master clock is released first), and reloading the same analysis **keeps the sync point** — that belongs to the video, not to loading it. A different analysis still starts at frame 0.
>
> **Two raw videos side by side (12 September 2026):** the viewing window (`ViewWindow`, Recordings tab) can now also show two videos at once — select two rows, or "➕ Second video alongside..." in the window — on the same master clock as this page, which was pulled out of `MainWindow` as `MasterClock` for that reason. A sync point per video, one shared speed, no analysis needed; points only with a single video. See CLAUDE.md.

> **One speed for both sides:** the per-side speed control is gone (`VideoPlayer(show_speed=False)`); the shared control at the bottom of the page now also drives a side playing on its own. Running two videos side by side at different speeds is exactly what you don't want while comparing, and the per-side combo invited exactly that.
>
> **Deliberately not done:** still no saved sync points (see above — schema bump); no third side.

---

## Extra — Following a small skater from a drawn box (spyglass) ✅

> **Done (11 September 2026)** — outside the phasing, on request: "on some footage the skater is too small".
>
> **Why:** the detection pass (yolo26x-pose at 1280) only sees a skater from ~80–130 px in height (measured across the 48 analyses in the library). A skater who starts far away therefore loses their run-up — on `00005 8-41`, frames 84–112 (3.4–4.5 s), even though they're already ~100 px in frame at frame 0 — and `_CornerGuard` additionally puts that run-up into skip mode as "nothing to see", so it shows up as corner in the table. The refinement pass (RTMPose on a bbox) does work on such a skater; the problem is *finding* the bbox.
>
> **What's there:** in `TargetPicker`, besides clicking, you can also **drag a box around the skater** (mouse wheel zooms around the cursor, right-drag pans — on a 900 px dialog such a skater is ~45 px tall). That box turns on the **spyglass** in the YOLO backend (`_Spyglass` in `skate_yolo.py`): in the refinement pass the bbox is propagated frame to frame from the RTMPose keypoints themselves, from the box through the run-up, and from the chain through gaps > `GAP_FILL_S`, the tail, and the corner skips. No second detector (VRAM; and a YOLO `predict` during the `track` session feeds ByteTrack crop coordinates). Safety nets: a score gate, a veto color gate, per-step plausibility, and a link test (IoU against the chain bbox where the run meets the chain); a run starting from the box is rejected without a link. The box is stored as `doel_kader` in `settings_json` and shown in the Info dialog. See CLAUDE.md (point 7 under the YOLO backend and the `TargetPicker` bullet).
>
> **Measured** (A/B with the same code, only the box differing): `8-41` coverage 185 → 269/269 and 8 → 12 pushes, `10-57` 119 → 132/132, `14-48` 120 → 183/183 and 6 → 7 pushes; link IoU 0.86–0.92; on the shared frames **0.0 px** difference on knees and ankles; +5 to +7 s analysis time. Visually checked that the skeleton sits on the intended skater, even on `14-48` where a second skater in the same suit appears right behind him.
>
> **The floor is measured and hard:** on `00000 16-14` (skater 35–57 px, 7.5 of the 8 s) RTMPose is blind (leg scores 0.1–0.2) and YOLO on a 5× magnified crop only sporadically sees a blob — below ~70 px there's nothing to measure with any finder. That's why the app warns on such a box (`BOX_MIN_HEIGHT_PX`), still follows the largest mover alongside the spyglass (the box may never do worse than a click), and reports if the spyglass loses the skater. The advice then is: start the fragment later.
>
> **Also measured and rejected: a higher detection resolution.** The same clip with `DETECT_IMGSZ` 1920 (native) and 2560 (1.33× upscaled): YOLO then finds the skater at ~73 px instead of ~115 px (frame 144 instead of 187 — 1.7 s earlier), but nowhere below 73 px at any resolution, and 5.8 of this clip's 8 s sit below that. Costs 2.0× resp. 3.2× the detection pass (0.94 → 1.92 → 3.05 s/frame on DirectML), and 1280 already just barely fits on the 4 GB NVIDIA laptop. The 73–115 px band that 1920 gains is already covered by the box + spyglass at 1280 for free. Decision (11 September 2026): 1280 stays; no opt-in built.
>
> **Deliberately not (yet):** turning the spyglass on without a box (it would then also fill every analysis's gaps and corner skips — a measurement change that deserves its own A/B first); short gaps ≤ `GAP_FILL_S` using propagation instead of interpolation (same reason); a YOLO re-detection on a crop if the spyglass dies (only once the measurement calls for it — `schat` is already a callable); a golden angle error on the run-up frames (`skate_eval.py annotate` on `8-41`).

---

## Phase 3 — Skeleton editor (dragging points) ✅

> **Done (20 July 2026)** — self-test (`python skate_db.py`) covers the save round-trip (edit → `landmarks_raw.npz` backup + `edited=1` + a fresh events cache; restore original → npz restored + `edited=0`); GUI tested by hand.
>
> **What's there:** an "Edit" button on the view page (`_toggle_editing`) with draggable handles per landmark (`_draw_handles`/`_find_landmark`/`_handle_radius`), dragging + flowing out to ± N neighboring frames with a cosine falloff (`_editor_mouse_press`/`_move`/`_release`, `_set_landmark`, `_flow_frames`), live recomputation (`_recompute` → `process_derivatives` + `segment_pushes`, no smoothing), undo/redo (`_undo`/`_redo`), "restore original". Storage side in `skate_db.py`: `save_edited_landmarks` (overwrite the npz + a one-time `landmarks_raw.npz` backup + `edited=1` + cache), `restore_original_landmarks`, `refresh_events_cache`, a shared `_write_events_cache`.
>
> **Deviations from the plan below:**
> - Grab radius scales with the on-screen torso length (`HANDLE_MIN_PX`/`HANDLE_MAX_PX`) instead of a fixed 12 px — legible for a distant skater and at any zoom level.
> - Pixel-correct editing at any zoom level first needed **crop-and-magnify zoom** (built separately, same commit; see CLAUDE.md `_show_pixmap`/`_crop_norm`).
> - *Placing* points on frames without a pose: not in the first version — **built afterward, 31 July 2026**, see below.

> **Addition: placing a skeleton on a frame without a pose (31 July 2026)** — headless smoke test (click sequence, undo/redo, canceling, navigating away) + `python skate_db.py` green.
>
> **Why:** a frame without a pose breaks the stance run in `determine_push_from_extension` and produces `INCOMPLETE_TRUNCATED` — two missing frames can cost a whole push measurement that way. If the trainer can close the gap by hand, the measurement comes back. That's the actual payoff; the skeleton itself is just the means.
>
> **What's there:** in edit mode, **"➕ Make skeleton"** is active on a gap frame. Normally the skeleton is then **taken over from the neighboring frames** (`make_prefill` in `skate_analysis.py`: interpolate on a short gap, copy on a long gap) and you correct it with the regular drag editor — one way of working for **every** frame. Only if there's nothing to take over (no pose anywhere in the analysis) is there also nothing to drag; then the program asks for the points one at a time in a fixed order (shoulders, hips, knees, ankles — `PLACEMENT_ORDER`, 8 clicks) with the requested point shown as an orange ring-with-crosshair. Zooming and panning stay available — the mouse wheel zooms and **right-drag pans** (new in `VideoPlayer`, also on the compare page). Status-bar counter **"Skeleton: 123 of 126 frames"** plus a **"⏭ Next gap"** button. Undo/redo reverts a placed skeleton in one step. See CLAUDE.md for details.
>
> **Choices:** dragging over clicking — reading eight names during a click sequence is more work than adjusting a skeleton that's already roughly right, and it keeps the controls the same as for any other frame. 8 points, not 33, in the click sequence — exactly what the measurements use, plus the torso that grab radius and auto-zoom rely on; heel/toe stay at visibility 0, just like the YOLO backend without RTMPose. A click sequence can only be committed once hip/knee/ankle of both legs are set (otherwise a 0° angle would land in the table), and navigating away without a single click cancels rather than silently committing the prefill as a measurement.
>
> **Deliberately not done:** no "fill the whole gap at once" button (interpolating over a gap is exactly what the detector already couldn't do); no persistent marker for which frame was hand-placed (the green rings are per session).

**Goal:** after an analysis, fix small detection errors by dragging landmark points; works on any analysis loaded from the database.

### Interaction

- An **Edit button** on the view page turns the editor on: playback pauses, every landmark of the current frame gets a draggable handle (small circles; grab radius ± 12 px at screen resolution).
- Mouse events on the video label: convert widget coordinates → frame coordinates (mind the scaling/letterboxing of the view label — this conversion path already half exists in `TargetPicker`/`HorizonPicker`, worth making reusable).
- Dragging works on the **smoothed** landmarks (what you see is what you edit); the smoothing step is therefore **not** re-run after an edit, otherwise the correction would immediately be smoothed away again.
- Dragged points get `visibility = 1.0` (a manually set point is by definition reliable) and a "manual" marker so they can get a different outline in the overlay.

### Flowing out to neighboring frames (adjustable)

- A "flow out: ± N frames" spinbox (default 8, 0 = this frame only).
- The displacement (delta-x, delta-y of that point) is applied over the window with a cosine falloff: a frame at distance `k` gets `delta * 0.5*(1+cos(pi*k/N))`. No jump in the motion, and at distance N the effect is exactly 0.
- Flowing out stops at a detection gap (frames without a pose) — never smear across gaps, same principle as the smoothing.

### Recomputing + saving

- After every drop (mouse button released): rerun `process_derivatives()` + `segment_pushes()` over the results list → table, graph, and HUD refresh live. This is pure numpy work over already-detected landmarks, plenty fast.
- An **undo/redo** stack (store per edit: landmark index, window, deltas) — doing manual work on 12 px points, you will miss at least once.
- **Saving**: changed landmarks → overwrite `landmarks.npz`; the original lives in `landmarks_raw.npz` (created on the first edit) so there can be a **"restore original"** button. `analysis.edited = 1` in the DB, and refresh the events cache.

**Done when:** correcting an obviously wrong knee point in one drag motion, seeing the angle table update immediately, saving, reopening — the correction is still there; "restore original" reverts everything.

---

## Phase 4 — Sharing with multiple trainers (shared cloud folder) ✅

> **Done (20 July 2026)** — this team's library lives in a **Google Drive** folder (Mirror). Self-test (`python skate_db.py`) extended with the v1→v2 migration, conflict-copy detection, and the created_by/video_bytes round-trip; headless GUI smoke test green in the YOLO venv.
>
> **What's there:**
> - **Trainer name**: a "Your name..." button on the start page (next to "Refresh"), stored in `config.json` (`trainer_name`, per user — not in the shared folder). Travels along as `analysis.created_by` on both new and batch analyses (via `AnalysisWorker`/`BatchWorker`), and shows up as a "Created by …" tooltip on the title in the analysis table.
> - **Conflict detection** (`skate_db.detect_conflict_copies`): opening/switching a library, and "Refresh", warns if there are other `*.db` files next to `skate.db` (a conflict copy from the syncer, e.g. `skate-DESKTOP.db`). Detection, not prevention — the trainer cleans it up by hand.
> - **Refresh button**: rereads the library from disk (`_refresh_library`) so colleagues' analyses become visible without a restart; also repeats the conflict check.
> - **Video sync check**: a schema migration to `user_version=2` adds `analysis.video_bytes` (the size of the copied video at save time). On open, `video_sync_status()` determines whether the video is missing (not downloaded yet) or incomplete (smaller than what was saved → still syncing from the cloud) and shows a clean message instead of loading half a video. Old analyses (v1, `video_bytes` NULL) skip the size check.
>
> **Deviations from the plan below:**
> - The cloud-safe SQLite discipline (journal DELETE, short connections, busy_timeout, relative paths) and the library-path config were already pulled forward in phase 1 — only trainer name, conflict detection, refresh, and the sync size check were left for here.
> - The optional lock file (`media/<id>/.lock`) against simultaneous editing of the same analysis was **deliberately not built**: the roadmap flagged it as "if desired", and the odds are negligible with UUID folders + a small team ("last writer wins" remains the accepted limitation).

**Goal:** the whole team looks at the same library.

### Approach

- **Settings screen**: choose the library path. Every trainer points at the same OneDrive/Dropbox/network folder. Plus a "your name" field → goes into `analysis.created_by`. Both remembered in a small local config file (`%APPDATA%\SchaatsAnalyse\config.json` — not in the shared folder, since it's per user).
- Because phase 1 already stores everything relative to the library folder, sharing afterward is mostly *configuration*, not a rebuild. That's the reason to keep that path discipline strict from phase 1 onward.

### SQLite on a synced folder — the pitfalls and countermeasures

SQLite isn't designed for concurrent writes via cloud sync. At team scale this is entirely manageable, provided:

1. **No WAL mode** (`journal_mode=DELETE`): WAL creates `-wal`/`-shm` side files that cloud syncers can sync only partially → corruption risk. DELETE mode keeps it to a single file (plus a short-lived journal).
2. **Short connections**: open a connection → transaction → close immediately, never leave a connection open while browsing. That way the DB file is almost always "at rest" for the syncer.
3. A `busy_timeout` of a few seconds for the rare case where two trainers write at the same moment over a real network drive.
4. **Conflict detection instead of prevention**: if OneDrive does create a conflict copy anyway (`skate-<pc-name>.db`), detect it on startup and warn. Since analyses are UUID folders and trainers rarely write within the same minute, the practical odds are small; media files (video/npz) are only ever created, never written to by two people at once.
5. A **Refresh button** in the library view (reread the DB) so you see a colleague's new analyses without restarting. No need for automatic polling.

### Deliberately accepted limitations (fine at this scale)

- No accounts/permissions: anyone with the folder can see and delete everything.
- "Last writer wins" when editing exactly the same analysis at the same time (e.g. both in the skeleton editor) — rare; can be mitigated if desired with a simple lock file (`media/<id>/.lock` with the trainer's name) that shows a warning.
- Large videos sync slowly; whoever opens a colleague's analysis while the video is still coming in gets a clean "video not synced yet" message (checks whether the file exists + whether the file size matches).

**Upgrade path**: should this ever need to become club-wide (accounts, permissions, concurrent writes), the step to a hosted Postgres (e.g. Supabase) is limited to swapping out `skate_db.py` + uploading the media — the rest of the app notices nothing. That's the reason to keep all SQL in one module.

---

## Phase 5 — Better stabilization: horizon via two tracked points

> **Nice-to-have (not now).** Since July 2026 the assumption is that the camera is **always exactly horizontal** — then there's no camera tilt to correct and this phase is unnecessary. Kept for a possible later setup with a tilted/wobbling camera; only pick this up once that situation actually arises.

**Goal:** make the auto-horizon more reliable. The current `determine_horizon_sequence()` runs a Hough line detection on the bottom of the frame per frame — it sometimes grabs the wrong line (boarding ads, a shadow edge). New idea: the user marks **two points in the first frame that are, in reality, horizontally apart** (e.g. two markings on the boarding); those two points are tracked through the whole video, and the angle of the line connecting them **is**, per frame, the camera tilt.

### Interaction

- A third horizon mode next to "fixed" and "automatic per frame": **"track two points"**. The existing `HorizonPicker` dialog gets extended: the user clicks two points (as a line is already drawn today), but now chooses "track these points through the video".
- Tips in the dialog: pick points on **stationary, high-contrast** details (boarding edge, a line transition, a pillar) that stay in frame the whole video — not on ice (reflective) or on people.

### Technique

1. **Tracking**: sparse Lucas–Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) per point, with a **forward-backward check** (track the point back; if the return trip deviates > 1–2 px, mark the frame as unreliable). LK is subpixel-accurate and cheap (negligible next to pose detection).
2. **Fallback per point**: if LK fails (occlusion — e.g. the skater slides in front of the point), template matching (`cv2.matchTemplate`) in a search window around the predicted position; if that fails too → skip the frame and interpolate later.
3. **Angle sequence**: per frame, `atan2(dy, dx)` of the two tracked points, minus the angle in the reference frame (the clicked stance = by definition the true horizontal, so the measured angle is directly the tilt). Then the same cleanup that already exists: Hampel outliers + Savitzky–Golay (reuse the `determine_horizon_sequence` machinery), result per frame in `r.horizon_deg` — the rest of the pipeline (subtraction in `calculate_angle_to_ice`, the tilting ice line in the overlay) then works unchanged.
4. **A point leaves the frame** (panning camera): detect when a point nears the frame edge and then **hand off to fresh anchor points** — `cv2.goodFeaturesToTrack` in the same image band looks for new high-contrast points, which inherit the angle calibration valid at that moment. That way the measurement keeps running without the user having to click again. This is the hardest step; a first version may skip it and simply warn + hold the last angle.
5. As a separate, fast video pass integrated into `phase_progress()` (like the existing auto-horizon pass), in both backends.

**Done when:** on a test video with a visibly wobbling camera, the tracked-points mode gives a smooth, believable `horizon_deg` sequence (the white ice line in the overlay stays on the real ice edge), even when the skater briefly passes one of the points.

---

## Phase 6 — Faster analysis (getting more out of the CPU/iGPU)

**Goal:** substantially speed up the YOLO analysis (currently ~2 s/frame on CPU with yolo26x-pose at 1280). Hardware here: a **Ryzen 7 7735U** (8 cores/16 threads) with an **integrated Radeon 680M** — no NVIDIA, so no CUDA; the realistic route is optimized CPU inference and possibly the iGPU via DirectML.

> **Groundwork: measured on the target machine (19 July 2026).** A profiling session was run on "Schaats frontaal.MOV" (1920×1080, torch 2.12.1+**cpu**, onnxruntime CPU-only). The findings below are noted here because they reverse the order of this phase; they were never turned into code, so phase 6 is still fully open.
>
> **Where the time goes:** the detection pass yolo11x-pose @1280 = **~2333 ms/frame = 94% of the compute time**; RTMPose refinement ~140 ms/bbox (only on target frames). Everything that isn't the detection pass is noise.
>
> **Dead end: throwing more hardware at it.** Both nets are **memory-bandwidth bound**, not compute bound. Measured: yolo11x@1280 is **flat from 1 → 16 threads** (~1950 ms/frame — more threads does nothing); two processes at once each cost ~3080 ms (together only **1.3×**); four processes each ~7000 ms (slower than serial). RTMPose behaves the same (~1.3× with two processes). Multiprocessing, threading, and tuning thread settings therefore give **at most ~1.3×** — the reason step 6 below moved from "quick win" to "probably pointless".
>
> **The lever: shrink the detection model.** Measured per frame @1280: `yolo11m` = 810 ms (**2.9×** faster than x), `yolo11n` = 162 ms (**14×**, but 5 of 6 detections). Over the whole pipeline that's roughly 2.6× (m) to 8× (n). This is possible because the pass-1 keypoints are **overwritten anyway** by the RTMPose refinement (`verfijnd.get(f)` wins; the pass-1 `lm` is only a fallback): the detection pass only has to deliver bbox + track ID + torso color + centroid. The risk therefore sits **not** in angle precision but in **detection coverage of the target skater, tracking robustness, and the color histogram** — exactly what `skate_eval.py` measures. Shrinking `DETECT_IMGSZ` (1280 → 960) roughly halves the detection cost but hits the same risk: 1280 was specifically chosen *because* 640 completely missed distant/motion-blurred skaters.
>
> **Happened since, outside this list:** the swap to `yolo26x-pose` (21 Jul 2026) gained ~12% and cost nothing in accuracy (stance-leg angle error 1.34° vs. 1.54° against the golden reference — equal within noise), and **corner detection** (5 Aug 2026) removed 45% of the time on a clip that's half corner. The latter is effectively step 2's "skip frames" idea, applied where it was free.
>
> **Not measured, so still an estimate:** OpenVINO export and DirectML on the iGPU (step 4). Those numbers below come from the literature, not a test on this machine.

In increasing order of effort, cumulative — after each step, measure with a fixed test video (see the measurement protocol below). **Order after the July 2026 measurement: step 3 first** — that's the only step with a big number underneath it; 1 and 2 are edge work on the 6% that isn't the detection pass.

1. **Batch inference in the refinement pass** (`_refine_landmarks`): crops currently go through `model.predict()` one at a time; ultralytics accepts a list of images. Collect crops and predict in batches of e.g. 8–16 → less overhead per frame, better core utilization. Little code, no quality loss. *Note: refinement is only ~6% of the runtime, so this is at most a few percent overall — and "better core utilization" is questionable for a bandwidth-bound net.*
2. **A prefetch thread for reading video**: `cv2.VideoCapture.read()` + resize on a separate thread with a small queue, so decoding and inference overlap instead of alternating. Applies to every pass (detection, refinement, auto-horizon). *Since corner detection (Aug 2026), `_detect_all` reads the frames itself instead of via `model.track(source=...)`, so this step can now also apply to the detection pass.*

   **A nearby opportunity, found while doing corner detection:** the refinement pass fills detection gaps up to `GAP_FILL_S` (1.0 s) with interpolated bboxes and recovers the pose there. Skipping 1 in N frames on a straight section in pass 1 would therefore largely be caught by pass 2 — the corner guard already does exactly that in the corner. Only to be done alongside the measurement protocol: this is about frames that are actually being measured.
3. **A lighter model for the detection pass, x for the refinement** — ⭐ **start here**: pass 1 only needs to deliver bboxes/track IDs and global keypoints; the precise angles come from the crop pass. `yolo26m-pose` (or even `s`) at 1280 for pass 1 + RTMPose/`yolo26x-pose` for the crops. **Measured prediction** (see groundwork): m ≈ 2.6×, n ≈ 8× over the whole pipeline — by far the biggest number on this list, and the only step that touches the 94%. **Must validate** that pass 1 still finds the distant/motion-blurred skater (that was the whole reason for 1280 × x): on the test video, check that coverage stays 100%, the events stay identical, and target selection/stitching doesn't get worse (`yolo11n` missed 1 in 6 detections in the measurement — that's where tracking breaks first, not the angle). The color histogram depends on bbox quality, so `_split_by_color` and `_stitch_chain` are the first places a too-small model shows itself. One A/B run against the golden reference settles this.
4. **An exported model instead of PyTorch**: ultralytics' `model.export(format=...)`, then infer with:
   - **OpenVINO** (`format="openvino"`): an optimized CPU runtime, also works on AMD CPUs; typically 1.5–3× faster than torch-CPU, same weights so the same output (small numeric deviations).
   - **ONNX Runtime + DirectML** (`format="onnx"`, `onnxruntime-directml`): runs on the Radeon iGPU. Potentially the biggest jump, but iGPU drivers/DirectML are the most unpredictable item on this list — plan it as an experiment, with the CPU path as a fallback.
   Both fit into `skate_yolo.py` behind a small abstraction layer around `model.track`/`model.predict`; ByteTrack tracking keeps working through ultralytics with an exported model.
5. **A GUI "fast / accurate" choice**: a configurable profile on the start page (fast = m model + a smaller `DETECT_IMGSZ`; accurate = the current settings). The user chooses per video whether they want a quick impression or a precise measurement.
6. **Quick wins** — largely superseded by the measurement; what's left of them, in descending order:
   - **Power mode to "Best performance"** (Settings → System → Power & battery → Power mode, **not** `powercfg`: on this Windows 11 install there's only one scheme, "Balanced", and the performance mode is an overlay from the Settings app). *Checked 8 Aug 2026: the machine was on "Balanced" while plugged in.* The 7735U is a 15 W chip with configurable TDP up to 28 W; a lower sustained package power also drags down the fabric/memory-controller clock, and that is exactly the bottleneck. **The only knob on this list where 10–30% is realistically on the table, and never measured.** One toggle + one test run.
   - **`cv2.setNumThreads(2)` during the YOLO pass.** July's profiling measured yolo *in isolation*; in the real pipeline, decoding, resizing, and the color histogram compete for the same cores. Two lines, expect a few percent, no risk to the measurement.
   - **Above-normal process priority** for the analysis worker. Marginal, free.
   - **A Defender exclusion on the library folder.** Doesn't touch inference, but does touch writing: `save_analysis` copies the whole video and that gets scanned along the way. Felt wait time, not analysis time.
   - ~~`torch.set_num_threads(16)`~~ — **dropped**: thread scaling is flat from 1 → 16, there's no core utilization to gain.

   **Two dead ends, explicitly noted here so they aren't investigated again:**
   - **Upgrading memory can't and doesn't need to happen.** *Checked 8 Aug 2026:* 4× 4 GB **LPDDR5-6400 soldered to the motherboard** (16 GB, full bus width). No single-channel mistake to fix, no SODIMM to swap — the memory subsystem is already running at spec. The bandwidth bottleneck is therefore a given, not a defect.
   - **Running several videos from a batch at once.** A natural thought, but exactly the measured scenario: two processes together 1.3×, four processes slower than serial. `BatchWorker` runs them one at a time and should keep doing so.

**Measurement protocol**: one fixed test video ("Schaats frontaal.MOV"), record per step: total analysis time, pose coverage (%), and whether the push events (count, leg order, angles ±1°) stay equal to the reference run. A speedup that changes the measurement isn't a speedup. For a model swap, add `python skate_eval.py compare old.npz new.npz` + the golden reference — but compare **unedited** analyses (`analysis.edited = 0`), since a hand-placed skeleton counts as a detection in the coverage metric.

**Expectation (revised after the July 2026 measurement)**: step 3 is the main prize — **2.6× (m model) to 8× (n model)**, provided coverage holds up. Step 4 (OpenVINO) is theoretically 1.5–3× on top of that, but untested on this machine. Steps 1 and 2 give at most a few percent, since they touch the 6% that isn't the detection pass. Step 6 has one exception worth doing: the **power mode** does touch the bottleneck (package power → memory clock) and could be 10–30% — start there even, since it costs no code. The earlier estimate "1.5–2× from 1+2+6" was made before the profiling and turned out too optimistic.

**Done when:** the total analysis time for the test video is at least halved with no loss of coverage or measurement quality, and the fastest acceptable configuration is the default.

---

## Phase 7 — Perspective correction via track lines

> **Nice-to-have for the daily workflow, but validation is under way (Aug 2026).** Since July 2026 the assumption is that filming is **always straight-on from the front** with a **horizontal** camera. Under that setup the camera looks nearly perpendicular to the plane of motion and the perspective distortion is small, so the daily workflow doesn't need this correction. Steps 1 and 2 (the math core + hooking it into the pipeline and GUI) are done and opt-in; since 11 August 2026 the calibration is also **saved and reused**. What's left is step 3, validation on real footage — and that footage now exists (see "Status" at the end of this phase).

**Problem:** the push and knee angle are currently measured in the **image plane** — the 2D projection of the leg. That's only correct if the camera looks perpendicular to the leg's plane of motion. If the camera isn't centered on the track (or the skater doesn't come straight at the camera), you're looking at an angle and perspective foreshortens the leg in one direction: the measured angle deviates structurally from the real one, and — more treacherously — the deviation **changes with the skater's position in frame**. The same push can then appear to have a different angle at the start of a pass than at the end. The existing horizon correction only fixes camera rotation around the viewing axis (roll), not this distortion.

**Core idea:** the lines in the ice are straight, parallel lines with a known spacing (standard track width ± 4 m). From that, the camera's pose relative to the ice plane can be calibrated, and with that calibration the angles can be reprojected per frame onto the real, undistorted plane.

### Steps

1. **Marking lines (interaction).** An extension of the existing `HorizonPicker` dialog: on one frame, the user traces two (or more) track lines that in reality run parallel in the direction of travel, plus preferably one cross line (start/finish line, corner marking). Optionally assisted with Hough detection (that machinery already exists in `detect_ice_line()`), but manual tracing is the reliable baseline — lines in the ice are low-contrast and partly scratched.
2. **Calibration from the lines.** Parallel lines meet in frame at a vanishing point; the direction-of-travel lines give vanishing point V1, the cross line(s) give V2. The line V1–V2 is the **vanishing line of the ice plane = the true horizon** (a bonus: this replaces the Hough horizon hack with something more principled). With the usual assumptions (principal point at the image center, square pixels), two orthogonal vanishing points yield an estimate of the focal length, and with that the full **ice-plane ↔ world-plane homography**. The known track width puts scale (meters) on it.
3. **Angle correction.** The ankle sits (nearly) on the ice → its world position follows from the homography. The knee is a ray from the camera; pinning it down in 3D needs one extra assumption. Two candidates, to be chosen after experiment:
   - **(a) Constant lower-leg length**: the ankle–knee distance is fixed per skater. Calibrate that length on frames where the leg is nearly perpendicular to the viewing direction (there the projection is undistorted), then per frame intersect the knee ray with the sphere of that radius around the ankle. Geometrically the cleanest.
   - **(b) Leg-plane assumption**: assume the lower leg lies in a vertical plane with a known orientation (e.g. the direction of travel from the tracked trajectory, or perpendicular to it during the sideways push). Simpler, but the assumption is debatable for a skating push (diagonally sideways-backward) — verify on test material which variant is more stable first.
   From the reconstructed 3D points follows the real angle relative to the ice plane; that replaces the image-plane angle at the `calculate_angle_to_ice()` spot (the same place in the pipeline as the current horizon subtraction, so steps 2/3 of the core stay unchanged).
4. **A quality indicator.** The correction is large and sensitive when the skater is far from the camera axis; show per frame (HUD) and per push event (table) how large the applied correction was, and flag measurements where the geometry becomes unreliable (leg nearly along the viewing direction — then no correction can save it).
5. **A bonus (free once you're here):** with a metric ice-plane homography, the skater's position on the track per frame is known → real **speed (m/s)** and **stroke length per push** in the table.
6. **Camera motion.** First version: only a **fixed camera (tripod)** — one calibration for the whole video. For a panning/wobbling camera the homography would need to move per frame: track the marked lines with the same LK optical-flow machinery as phase 5. That's a logical follow-up, not part of the first version.

**Done when:** the same skater passing at different places in frame (near/far, left/right) gets a stable push angle after correction (± 2°), where the uncorrected measurement visibly drifts with frame position. Test recording: one skater, several laps past the same fixed camera, compare angles per pass.

### Status (11 August 2026)

**Steps 1 and 2 are done** (7 July 2026): the math core `skate_perspective.py` with a self-test, and the opt-in hookup into both backends + the GUI. **Step 3 (validation on real footage) is still open.**

**Test material found.** The July video didn't qualify (moving camera, no cross line, frontal, 4.2 s). The usable material is the **seven fragments trimmed from `opnames/00005.MTS`** (`source_id` 1, titles "00005 8-41" through "00005 11-54", source frames 13015–18106). Checked:

| Requirement | Previous video | These seven fragments |
|---|---|---|
| Fixed camera | ✗ ~38 px drift (≈2°) | ✅ **0–1 px over 3½ minutes**, all seven |
| Oblique viewing angle (f self-calibratable) | ✗ frontal | ✅ clearly oblique along the track |
| ≥2 direction-of-travel lines | ✗ 1 | ✅ **3** (blue track line, ice/snow edge, boarding base) |
| ≥1 cross line | ✗ none | ✅ visible to the eye (manual tracing; Hough doesn't find them on scratched ice) |
| Multiple passes | ✗ one | ✅ **seven, ~50 pushes total** |

Two findings from that check that steer the plan:
- **The distortion is already visible in the uncorrected analyses**: within each pass, the push angle rises as the skater gets closer (11-08: 44.4° at ankle-y 319 → 55.5° at y 640; 10-25: 38.0 → 41.9). So there really is something to correct.
- **But that drift varies a lot from pass to pass over the same image region** (+2.7° at 8-41 vs. +11° at 11-08), so part of it is technique change or noise. Validation therefore can't lean on "the angle should become constant" and needs an explicit null model.
- **The building columns stand perfectly vertical in frame** (0–2 px over 60 px height) → camera roll ≈ 0. It follows that the ice plane's vanishing line runs exactly horizontal through V1 — a free extra check on the calibration, and a fallback route should a cross line ever be missing.

**Validation plan — four tests, cheapest to the done-criterion:**
- **A — calibration without ground truth (first).** The stance-leg ankle travels over the ice, so the reconstructed `r.wereld_xy` should be a **straight line** with a smooth speed. Straightness residual + plausibility (pace ≈ 10–12 m/s, stroke length 5–8 m, reconstructed hip width vs. measured) rejects the homography before anything gets annotated. If this fails, the rest is pointless.
- **B — robustness.** Leave-one-line-out: recalibrate on varying subsets of the track lines and measure how much f, the horizon, and the final angles move. If they move more than the claimed gain, the calibration is too wobbly for practice.
- **C — depth drift (the done criterion).** Regress the push angle on frame position per pass, before and after correction; the slope should go to zero. Null model: the spread within one small image region is the noise floor, and only a slope reduction clearly above that counts.
- **D — consistency across passes.** The seven passes cover different lateral bands (x = 31 to x = 1219); after correction, the spread of the per-pass average should shrink. Least sensitive to technique drift within a single pass — and the reason one shared calibration across all seven is a hard requirement (see below).

**Rejected as a test:** left/right antisymmetry. Checked: the L−R difference is already only −0.9° to +2.0°, so no discriminating power.

**No length measurement needed for the angle at all (checked 11 August 2026).** The trainer only wants correct angles, not speed or stroke length — and that's possible, because the push angle is **scale-free**: it follows from the directions of the lines, not their spacing. Measured with 2 track lines + 2 cross lines and `leg_plane`: **0.00° error whether you enter 0.5 m, 4 m, or 50 m** as the line spacing. The GUI therefore has "Angles only (no speed/stroke length)" as the **default** (`scale_known=False`). Two things follow from that:
- **`lower_leg` can't work without a real scale**: it intersects with a sphere of a length in real meters, so a made-up line spacing gave **49° error, silently**. The dialog therefore locks the method to `leg_plane` as long as angles-only is on. For the A/B of both methods (tests A/C/D above) you have to turn the checkbox off and enter the real 4 m.
- **With 3+ track lines their spacing does matter**, even with `leg_plane`: the vanishing line then comes from the cross-ratio. Unevenly spaced lines entered as evenly spaced produce a **refusal** (f² ≤ 0) — never silently wrong — and with the correct `track_offsets`, 0.00° again. `track_offsets` exists in the module but not in the GUI; the error message therefore points to the way out: exactly **2 track lines + 2 cross lines**, since then V2 comes from the cross lines.

**First practical test on `00005 11-23` (11/12 August 2026): the correction made the angles worse, and that's been tracked down.** 2 track lines + 2 cross lines were traced, "angles only", method `leg_plane`. Result: corrections from −22.9° to +71.2°, status bar "unreliable". Three causes, all three verified:
1. **`f` is fundamentally not estimable from this camera pose.** The two cross lines run in frame almost parallel (slope −0.0219 vs. −0.0204), so V2 sits at **133× the image size** — practically at infinity, exactly the case for which the module docstring requires `f_px`. Consequence: f = 7081 px = 3.3× the image width ≈ **17° field of view**, impossible for a recording like this. And it's not just wrong but *meaningless*: **shifting one line endpoint 5 px sends f from 3827 px to "impossible"**. The camera here looks almost along the track — my earlier assessment "clearly oblique" was wrong; it's oblique enough for a clean V1, but the cross direction lies nearly parallel to the image plane.
2. **`leg_plane` is degenerate for exactly this pose.** That method puts the lower leg in a vertical plane in the direction of travel; if the camera looks along the track, the viewing ray lies *inside* that plane (the existing `plane_condition_deg` flag). Measured over 164 frames: `leg_plane` flags **25–61 as unreliable**, `lower_leg` only **5–7**. The recommendation to use `leg_plane` (because it needs no measurements) was therefore wrong for this material.
3. **But even with the better method, the correction doesn't win.** Sweeping f from 1200 to 6380 px: uncorrected, the spread of the counted push angles is **sd 1.6°**; the best corrected case is `lower_leg` at **sd 2.0°**, `leg_plane` stays at 3.8–7.4°. On this clip there's little to gain — makes sense, since a camera looking along the track has relatively little perspective distortion to begin with (the original roadmap premise). Note: sd over four pushes is a weak number, and "consistent" isn't the same as "correct".

**A gate built in as a result:** `calibrate_from_lines` now refuses self-calibration of f if the farthest vanishing point lies above `VP_CONDITION_MAX` (30× the image size), explaining that `f_px` must be supplied; above `VP_CONDITION_WARN` (5×) it gives a warning. Calibrated against the self-test cameras, which sit at 1.0 / 1.5 / 9.3 and all yield the correct f. The `CalibrationPicker` also no longer shows the **residual** at exactly 2+2 lines: the system is then exactly determined, so the residual is by construction 0.00 px and read as "perfectly calibrated" — it now says no check is possible and that a third cross line would provide one.

**The follow-up is therefore: determine `f_px` separately** (a checkerboard calibration with the same camcorder/zoom setting, or the camera spec), and only then measure again — preferably on `11-08`, since that pass does show clear drift (44.4° → 55.5°) while `11-23` is already flat uncorrected.

**Lower-leg length vs. hip width.** The proposal to use hip width as an anchor has been checked and is not recommended as a ruler: hip width measures 28–44 px against 47–85 px for the lower leg (~60%, so ~1.7× noisier), gets shortened by torso rotation itself (that's exactly the `corner_ratio` signal), and pins down the pelvis rather than the knee — chaining to the knee then adds the femur length rather than removing it. It's still useful as an independent check in test A. Note the misunderstanding underneath it: the **3D** lower-leg length **is** constant; only the projection varies, and that variation is exactly the signal the sphere intersection works on. The practical worry is valid though — deriving the length from the video is unreliable (see `calibrate_lower_leg_length`). Way out: `method='leg_plane'` needs **no length at all**; run both methods side by side and let A/C/D decide.

**A precondition built in (11 August 2026): the calibration is saved and reusable.** Without that, test D would require seven manual tracings by hand — seven slightly different calibrations, measuring that spread instead of the effect of the correction. What gets saved is the input (`CalibrationInput`: lines + line spacing/`f_px`/offsets/note + image size) in `settings_json`; the camera pose is recomputed from it. Reopening restores the correction, the batch flow asks for the calibration once for the whole row, and `_choose_perspective` offers earlier calibrations of the same image size to reuse. See CLAUDE.md for details.

---

## Phase 8 — Long recordings: trimming usable fragments in the app ✅

> **Done (11 August 2026)** — self-test (`python skate_db.py`) green in both venvs incl. the v1→v3 and v2→v3 migration and the recordings round-trip; headless smoke test of the recordings list, the trim window (marking, S/E/Delete shortcuts, drawing the bar, fragment list), and the batch hookup (prefilled rows carry `source_*` through to `save_analysis`).
>
> **What's there**, exactly per the plan below — trimming supplies the input for the existing batch flow, so nothing changed on the analysis side:
> - **`skate_db`, schema v3**: a `source_video` table + `analysis.source_id`/`source_start_frame`/`source_end_frame`, with `sync_source_dir` / `list_source_videos` / `source_video` / `edit_source_video` / `source_fragments`. `open_db` creates `opnames/`.
> - **`skate_analysis.trim_fragments()`**: one sequential pass, exactly on the marked frames, `mp4v`.
> - **GUI**: a second tab **"Recordings"** on the start page (status + note editable in place, count "3 fragments · 2 skaters"), **`FragmentPicker`** + **`FragmentBar`**, `VideoPlayer(fast_seek=, show_overlay=)`, and `BatchAnalysisDialog(voorgevuld=...)`. The Info dialog for an analysis now shows **"From recording: … (12:30–13:05)"**.
>
> **Workable on a real recording (checked again 11 August 2026, after the first practical test — the GUI froze on `00005.MTS`, a 4.2 GB AVCHD 1920×1080 @ 25 fps recording, 34,728 frames ≈ 23 min).** Decoding wasn't the problem (~9 ms per frame, a `grab()` ~3 ms, a seek ~80 ms) — the number of calls was. Three causes, all three fixed:
> - **One frame back meant rescanning the whole video from frame 0.** `fast_seek` only seeked on a jump > 30 frames, so the small backward step fell into the sequential route: at frame 20,000 that cost ~84 s with a fully frozen window. Now **every** backward step seeks → 99 ms. Small forward jumps stay sequential (cheaper than a seek, and stepping frame by frame around a boundary stays exact).
> - **Dragging the timeline stacked up hundreds of seeks.** The slider now only queues the last requested frame; a `QTimer` with interval 0 draws it once the queue is empty, so every intermediate value drops out. Measured: 300 slider signals processed in 1 ms, then drawn once.
> - **Scanning through wasn't possible at all.** The speed choice stopped at 1×, so watching through once took 23 minutes. There are now **2×/4×/8×**, implemented by **skipping** frames (4 frames per tick at the fps rate) instead of decoding faster — the latter can't outrun the decoder.
> - **The window didn't fit the laptop screen.** At 1280×800 (workarea 752 px) the trim window required 723 px minimum; with the title bar added, the button bar sank below the edge and "Done" was unreachable. The minimums were lowered (video floor 400×200, a smaller fragment table, tighter margins) → **553 px**, and `set_window_size` now runs last, against a complete layout. The other dialogs were rechecked and fit — **including `CalibrationPicker`** (rechecked 11 August 2026: minimum 1008×460, opens at 1150×700; the earlier claim of 1724 px was wrong).
> - Bonus: `trim_fragments` uses `grab()` without `retrieve()` for frames outside every fragment. Trimming 10 s at minute 20 costs 96 s instead of ~270 s that way — negligible next to the analysis (~2 s/frame) that follows.
>
> **Measured (11 August 2026):**
> - **Seek deviation on real iPhone .MOV files** (the price of `fast_seek`): on `IMG_8997.mov` and `IMG_9001.mov`, **0–1 frame** (0–33 ms). On `Schaats frontaal.MOV`, where `CAP_PROP_FRAME_COUNT` reports 108 frames but only 103 are readable, it grows to **4 frames (168 ms)** near the end of the clip — exactly the VFR drift the view page never seeks over. For a boundary picked by eye that's acceptable; the fragment itself stays exact, since `trim_fragments` counts sequentially from frame 0.
> - **Recoding** (the A/B step B asked for): `Schaats frontaal.MOV` trimmed from itself (103 frames, 1920×1080) and both analyzed with the same code. **100% coverage both ways, six pushes both ways, the same L-R order (RLRLRL), zero alternation errors**, and the event boundaries identical bar one frame (35 vs. 36). The angles: 42.2→41.0 · 42.5→42.5 · 42.9→42.5 · 40.0→40.5 · 45.6→45.9 (and the truncated 50.0→51.5, which doesn't count anyway) — so **at most 1.2° on a counted push, mostly ≤ 0.5°**. Joint positions differ 1–2 px median (p95 6–11 px) on a 1920-px-wide frame. Conclusion: **cv2 with `mp4v` stays the default.** The deviation sits in the same order as the ±1° phase 6 proposes as acceptable, and the alternative (ffmpeg stream copy) would bring back a 1–2 s GOP margin at the front — exactly what's not wanted here. If this ever becomes a real problem, the honest fix is a better codec setting or ffmpeg **with** recoding, not `-c copy`.
>
> **Deviations from the plan below:**
> - Recordings sit in a **tab** next to the skater list (not a third column): the work list doesn't hang off the skater selection.
> - **The target skater isn't picked in the trim window** (the open question below). `TargetPicker` therefore stands on frame 0 of every clip — which is exactly the frame "start" was pressed on, since trimming happens exactly on the marked frames. The backend rework (`_choose_seed(..., doel_frame)`) hasn't been done for this yet.
> - `VideoPlayer` got **`show_overlay`** alongside `fast_seek`: with empty `FrameResult` objects, the overlay would show "No pose detected" on every frame, and "Follow skater"/"Automatic zoom" are checkboxes that can't do anything without an analysis. Small jumps (≤ `SEEK_THRESHOLD_FRAMES`, 30) stay sequential, so stepping frame by frame around a boundary stays exact.
> - Bonus: the **Info dialog** shows a fragment's origin (`analysis_meta` pulls in `bron_naam` with a LEFT JOIN).
>
> **Motivation from practice:** one training session produces one half-hour recording. It currently gets cut into usable pieces by hand outside the app (Clipchamp) — that takes longer than the analysis itself and the external program works poorly. Trimming belongs in the app, right next to the video you're already watching anyway.

**Goal:** open a half-hour recording, mark the usable stretches in it (start/stop per stretch), and then have those stretches analyzed in one go — exactly like the app already analyzes a loose clip.

> **Starting point: this is a trimming tool, and the trimming is entirely manual** (fixed 10 August 2026). The app decides **nothing** on its own: not when the skater is in frame, not where a stretch starts or ends, and no second of margin is added or removed. The trainer watches, presses start and stop, and those are the boundaries. Everything the program does is remember those boundaries, show them, and write clips from them. Every proposal to build "smartness" in here has been rejected up front — see "Deliberately considered and not chosen".

### The requested flow

1. Choose a half-hour recording from the new **"Recordings"** list on the start page (`<library>/opnames/`, shared via Drive — see below).
2. Scrub through it; at a usable stretch: **"Start usable footage"** → **"Stop usable footage"**. Repeat for stretch 2, 3, … x.
3. While marking, it's **visible which stretches are already marked** (colored blocks on a bar under the timeline) and which stretches of this source recording were **already analyzed in an earlier session**.
4. On **"Done — analyze x fragments"**: choose a skater (and title) per fragment, then pick the target skater per fragment, and the analysis runs as it does now.

### Architecture: trimming supplies the input for the existing batch flow

The core of this design is that **nothing new** needs to happen after trimming: `BatchAnalysisDialog.taken` is already a list of `{input_pad, schaatser_id, titel}`, and `_new_batch_analysis` already asks for the target skater + horizon per video, after which `BatchWorker` runs the row and each analysis saves itself. Once every fragment is a plain video file, the whole new feature collapses into **two steps placed before that dialog**:

- **A. `FragmentPicker`** (new dialog) — marking on the source video → a list of `(start_frame, end_frame)`.
- **B. `trim_fragments()`** (new helper) — write those frame ranges out as loose clips → a list of file paths.

Then: open `BatchAnalysisDialog` with those paths **prefilled** (the rows already exist, the trainer only fills in skater + title). No second analysis pipeline and no second storage route — nothing changes on the analysis side. The schema bump below is therefore not about analyzing but about keeping track of the recordings themselves.

### A. `FragmentPicker` — the trim dialog

- **Reuse `VideoPlayer`** for playback: the scrub slider, transport buttons, speed combo (at 4× for scanning through half an hour), and zoom are already there. The player expects a `resultaten` list (for the overlay, the box, and slider length); a list of `total_frames` empty `FrameResult` objects is enough — `box_sequence` then returns `None` and automatic zoom falls back to a fixed zoom. **Verify first** that `load()` doesn't choke on that; if it does, a small dedicated player like `TargetPicker` already has.
- **Two buttons + shortcuts**: "Start usable footage" (`S`) and "Stop usable footage" (`E`). After "Stop" the fragment is **added to the list immediately** and shown on the bar; the button jumps back to "Start" for the next stretch. As long as a start is pending, only "Stop" is active (and vice versa) — so a half fragment can't happen.
- **A fragment bar** under the timeline: one widget as wide as the slider, with a colored block per fragment at `start/total … end/total`. Green = just marked, gray = already analyzed in an earlier session (see below), orange = the running (not yet stopped) fragment. Clicking a block jumps to it and selects it; `Delete` removes it. This is the only genuinely new drawing code in this phase.
- **A list alongside it** with, per fragment, `#`, start–end as `m:ss`, duration, and a delete button. If two fragments overlap, that's **made visible** (the overlapping part in a different color) but **nothing** is automatically merged or shortened — the trainer adjusts it themselves or leaves it as is.
- **Navigation help**: ±1 s / ±10 s / ±1 min buttons and a time-entry field. On half an hour, the slider is too coarse to find a push again.
- **Optional: pick the target skater right here** (to be decided during the build). Whoever is marking a fragment is, at that moment, looking at the skater they mean — that's the natural moment to click them, and it saves x separate `TargetPicker` dialogs later. Technically this fits the YOLO backend well: it collects all detections offline and stitches from the seed tracklet **both forward and backward** (`_stitch_chain`), so a seed in the middle of the clip is just as good as a seed at frame 0. What's needed is `_choose_seed(..., doel_punt, doel_frame)` that searches from `doel_frame` instead of from 0, plus sending the frame number along in `settings_json`. The MediaPipe backend is streaming and can't do this without rework — it just keeps the existing frame-0 route (it's the fallback backend). Until this exists, `TargetPicker` stays on frame 0 of the clip — exactly the frame "start" was pressed on, since trimming happens exactly on the marked frames (see below).

### B. `trim_fragments()` — writing out the clips

- **One sequential pass** over the source video with `cv2.VideoCapture`, sending each frame to the `VideoWriter` of the fragment it falls into. That way the video is decoded exactly once and there's **no seeking anywhere** — the frame numbering stays exactly that of the source (same motive as the dedicated read loop in `_detect_all`). A progress dialog around it; decoding half an hour costs a few minutes, negligible next to the analysis that follows.
- **Codec**: `mp4v` (ships in the opencv-python wheel; `avc1` is often unavailable on Windows). So this **does recode** — a quality loss that's negligible for pose detection, but measure it once: trim a known clip from itself and compare the analysis with `python skate_eval.py compare old.npz new.npz`. If that deviates noticeably, the way out is **ffmpeg with `-c copy`** (no recoding, essentially instant) if `shutil.which("ffmpeg")` finds something, with the cv2 route as a fallback. **But note — that clashes with "no margins":** stream copy can only start on keyframes, so the fragment ends up longer at the front by a GOP (~1–2 s) than what was marked. That's exactly the silent margin that isn't wanted here, and it makes frame 0 of the clip a different image than the one "start" was pressed on — with consequences for target selection. So: **cv2 with recoding is the default**, and ffmpeg only if the A/B shows recoding genuinely affects the measurement. In that case the honest fix isn't stream copy but ffmpeg **with** recoding of just the first GOP (`-ss` after `-i`), or a better codec setting in cv2.
- **Write to a temporary folder**; `save_analysis` then copies the clip as always to `media/<uuid>/`. That's one extra copy of a short file — not worth opening up `save_analysis` for.
- **No automatic margins — trimming happens exactly on the marked frames** (decided 10 August 2026). The trainer watches the footage while marking and decides the boundaries themselves; the program has no business silently adding or removing seconds. Two things worth knowing here, but no reason for automation:
  - There's nothing to gain at the **front** anyway: a stance run truncated at the start still counts (`determine_push_from_extension`, `run['afgekapt']`) — only the load phase is missing there, while the push completion the angle comes from is in frame. Air at the front would also make target selection harder: `TargetPicker` gets **frame 0 of the clip** (`_read_first_frame(pad)` in `_new_batch_analysis`) and `_choose_seed` looks for the click point in the first `CLICK_SEARCH_S` (6) seconds — since 26 August 2026, since with the old 60 frames the click missed a skater who wasn't detected until 3.4 s in — so the further away the skater stands there, the greater the odds of a miss, and on a miss the analysis follows the largest mover with a warning. That's how frame 0 ends up being exactly the frame "start" was pressed on.
  - At the **back**, the last stance run ends at the clip's end instead of on a leg switch; that push gets `INCOMPLETE_TRUNCATED`, stays visible gray in the table but falls outside avg/min/max. If you want that last push counted, press "stop" after the leg switch — a choice the trainer makes while watching, not the program.

### The recordings themselves in the library: `bronvideo` (schema v3)

> **Decided 11 August 2026.** The database currently only knows analyzed clips. There should also be a place for the **not-yet-analyzed recordings** — that folder already exists and, like the library, sits **in the shared Google Drive**. That makes "what still needs trimming" not a personal list but a **work list for the team**, and that's what justifies the schema bump: it delivers not just the gray blocks in the trim window, but also an overview of outstanding work.

**The design rule: the folder is the truth about which files exist, the database about what we know of them.** The file list is scanned from disk on open, not read from the DB — otherwise the DB would drift the moment someone renames or deletes a file, and you'd be stuck cleaning up after it. The DB only holds what you can never read off disk: status, a note, and which analyses came from which stretch of which recording.

**Where recordings live:** `<library>/opnames/`, i.e. **inside** the library folder. That way every path stays relative with forward slashes (the phase 1 discipline) and no extra per-trainer path setting is needed. If the folder doesn't exist, the app creates it.

**Schema `user_version=3`** — one new table plus three columns, both cloud-safe via the existing `_migrate` pattern (`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN`, as in v1→v2):

```sql
source_video(id INTEGER PRIMARY KEY,
          file,               -- relative path within the library (opnames/…)
          name, bytes,        -- identity: UNIQUE(name, bytes)
          fps, total_frames,  -- read once, saves reopening every time
          status,             -- 'todo' | 'in_progress' | 'done' | 'unusable'
          note,               -- free text ("training 3 aug, tempo series")
          updated_by, added_at)

analysis … + source_id, source_start_frame, source_end_frame   -- NULL for a loose clip
```

- **Identity = the relative path** (`opnames/<filename>`), `UNIQUE(file)`. One folder can't hold two files with the same name, and because the folder sits inside the library, that path is the same on every trainer's machine — exactly why the phase 1 discipline (everything relative, forward slashes) pays off here. A renamed file counts as new; the old row stays with its analyses and shows as "file not found".
- **`bytes` is not identity, only the sync check.** That distinction matters: a recording still coming in from a colleague is, at that moment, *smaller* than what's in the DB. If the size were part of the key, the scan would take a half-downloaded file for a new recording and add a second row — a messy list and lost fragment history, exactly when you need it most.
- **Old analyses keep `source_id` NULL** — which is also correct: they came from a loose clip, not from a recording. No migration has to guess anything.
- **`sync_source_dir(bieb)`** scans `opnames/`, adds new files with `INSERT OR IGNORE` (two trainers spotting the same new recording at once don't collide) and leaves rows for vanished files in place. Runs when the library opens and on "Refresh" (the phase 4 button, which just does this too). **Only writes when there's genuinely something new** — otherwise every app start by every trainer would touch the shared DB, and per the phase 4 discipline it should stay at rest as much as possible for the syncer.
- **You set the status yourself.** Nothing gets automatically set to "done" once every fragment is analyzed: the program can't know whether you consider the recording finished. It shows the count (`3 fragments · 2 analyses`), you set the status. Same principle as the rest of this phase.

**Sync status, now genuinely needed.** Google Drive Mirror puts every file locally, but a half-hour 4K recording takes minutes to arrive. The `video_bytes` trick from phase 4 works here one-to-one: the trainer who adds the recording first records `bytes`, and if a colleague's local file is smaller, the download is still in progress. `video_sync_status()` can be reused unchanged for this — reporting and not opening, instead of letting the trim window crash on a half file.

**What the GUI does with it:** the start page gets a **second view "Recordings"** (tab or button) next to the skater list: filename, duration, status, note, and per recording `3 fragments · 2 analyses`. Double-click → the `FragmentPicker`. Status and note are editable in place; `bijgewerkt_door` alongside, so it's visible who marked a recording "done" — the same motive as `created_by` on an analysis.

**And it delivers the gray blocks:** with `source_id` + `source_start_frame`/`source_end_frame` on the analysis, "which stretches of this recording are already done" is one query instead of a scan through every `settings_json` field. The trim window draws them gray, and reopening the same recording immediately shows where you left off. Bonus for phase 2: analyses from the same training session become recognizable as such.

### Pitfalls

- **Scrubbing backward on half an hour is currently unworkable.** `VideoPlayer._read_frame_exact` deliberately never seeks (VFR videos give a frame-inaccurate seek) and, on a backward jump, rewinds the video from frame 0 and replays. On a 10 s clip that's nothing, on 50,000 frames it's unusable. **This is the only real blocker of this phase**, and the fix is that the trim window makes a different trade-off than the view page: here the footage is a **look, not a measurement**, so a `CAP_PROP_POS_FRAMES` seek is allowed. Build that as an explicit flag on the player (e.g. `fast_seek=True`) so the view page stays untouched, and restore the internal cursor (`_display_pos`) after the seek so forward playback is correct again afterward.

  **What that costs:** on a VFR source, the shown frame can differ by a few frames from the reported number, so the trim lands at most a few frames from the picture you pressed on. For a boundary picked by eye that's invisible (~0.1 s) — and the alternative, never seeking, makes browsing half an hour impossible. The fragment itself stays exact: `trim_fragments()` counts sequentially from frame 0, so nothing drifts inside the clip. Measure once on an iPhone .MOV how big the deviation is in practice.
- **VFR sources**: fragments are written out using the reported `info.fps`. The whole app already computes with constant fps, so this isn't a new deviation — worth noting anyway, since a VFR drift runs further on half an hour than on 10 s.
- **`_read_first_frame` + `TargetPicker` per fragment** work unchanged once every fragment is a real file; they simply read frame 0 of that clip. That's the very argument for genuinely trimming instead of piping frame ranges through the pipeline.

### Deliberately considered and not chosen

- **Not trimming, but pointing analyses at the source recording + a frame range.** Tempting now that the recording is already in the library: zero extra bytes, no recoding. Still not done, and the reason is **display**, not storage. `VideoPlayer._read_frame_exact` deliberately never seeks and rewinds sequentially; an analysis starting at frame 40,000 of the source recording would have to scan through half an hour of video on every open and every jump back. On top of that it breaks `media/<uuid>/` as a unit: cascading delete (deleting one analysis mustn't take five other analyses' recording with it), `video_bytes`, the sync message, the duration column, and the compare page all rely on one analysis having one video file of its own. The extra storage is minor anyway: the fragments together are a fraction of the recording they came from.
- **Automatically suggesting usable stretches** (a cheap detection pass that looks for "frontal skater in frame" — the `_CornerGuard` machinery effectively already does this classification): **rejected, and not "for later".** The requested feature is a trimming tool; the trainer sees perfectly well what's usable and doesn't want a computer's suggestion laid over it. Such a pre-pass would also cost exactly the detection time this phase is meant to save. If this ever comes back, it should be a separate idea with its own motivation — not part of phase 8.

**Done when:** dropping a half-hour recording into `opnames/`, seeing it show up in the new recordings list as "todo", marking say six usable stretches in it, pressing "Done", and finding six analyses in the library without any further manual work — and, on reopening that same recording, seeing which stretches are already done, even on a colleague's PC.

**Scope:** 2 sessions, built in two separate pieces. (a) Schema v3 + `sync_source_dir` + the recordings list on the start page — that's `skate_db` work with a self-test extension and stands on its own. (b) The trim window + `trim_fragments()` + the hookup to `BatchAnalysisDialog`; the fragment bar and fast seeking are the real work there, the rest is wiring existing pieces together.

---

## Loose ends in target selection (open, found 26 August 2026)

Both surfaced while fixing the target click (`CLICK_SEARCH_S`, see the fix in
BUGS.md C2). Neither was addressed at the time: they fell outside that question and
deserve their own A/B.

### 1. The stitch gate grows to almost the whole frame width

`_stitch_chain` lets its distance gate grow linearly with the gap
(`STITCH_GATE_BASIS` + `STITCH_GATE_GROWTH`·gap). At a gap just under
`STITCH_MAX_GAP_S` (2.0 s), that's **0.06 + 0.015 × 49 = 0.795** — on a
normalized frame 1.0 wide there's effectively no positional requirement left, and
only the color gate still holds anything back.

**Measured on `00005 8-41`** (an offline replay on a dumped detection pass): the chain
starts with frames 27 and 35 of ByteTrack ID 16 at x ≈ 0.865, and glues that across a
gap of 49 frames (1.96 s) onto the target skater standing at frame 84 at x = 0.378 — an
actual jump of **0.490**, comfortably inside that gate of 0.795. That this is two
different people can be shown without even looking at the footage: **that same ID 16
is demonstrably somewhere else in frames 43–119**, namely at x = 0.871 → 0.963 with a
steadily growing bbox (area 0.0136 → 0.0294). The chain thus claims one person is in
two places in frame at once. That the color gate let it through is because
`_split_by_color` had already cut ID 16 itself in two (27–35 vs. 43–136) — the head
was, by suit color, no longer the same as the rest of that ID.

**The damage here was zero**: both frames fall in the corner, so they produce no
`lm_data` and no measurement. That's this clip's luck. In a fragment without a corner,
such a head would put a skeleton on the wrong person right at the start — precisely
where the first stance run begins.

**Directions** (nothing chosen yet): a **ceiling** on the gate, the way the MediaPipe
tracker already does with `TRACK_GATE_MAX`; or coupling the growth to the **predicted
displacement** instead of linearly to the frame count; or tightening the color
requirement as the gap grows. Note that this mirrors the nearest-first rule already in
`_stitch_chain`: that rule deliberately chose the smallest gap because a long jump
allows too much freedom — here that same long jump gets that freedom back through the
gate.

**Scope**: small in code, but the A/B is the real work — this touches every existing
analysis, so measure on several clips (coverage, events, L-R alternation, bone-length
CV) before and after, with `skate_eval.py compare`.

### 2. The MediaPipe backend has no click-search mechanism

`TargetTracker._seed` (in `skate_analysis.py`) takes, on a mouse click, the pose
closest to the click point in the **first frame with any detection at all** — no
distance gate, no search window, and no warning. A click there therefore always
"succeeds", even on a bystander ten meters away, and the user hears nothing about it.
The YOLO backend has, since 26 August 2026, a window (`CLICK_SEARCH_S`), a gate
(`CLICK_GATE_BASE`/`_GROWTH`), and a warning.

**Low priority**, since this backend isn't used in practice: the GUI picks YOLO as
soon as torch/ultralytics is present (`IS_YOLO`), and the bundled app doesn't even
include MediaPipe. It stays the fallback for an environment without torch, so fixing
it isn't urgent — but knowing the difference is there matters, since an analysis from
that backend would then be based on a different target than the one the trainer
pointed at.

**Scope**: small (the same gate/window logic in `_seed`), but with no test material in
that venv there's little to validate.

## Idea — OpenCV filters as a check on the neural net (not yet worked out, raised 1 September 2026)

Raised in response to the question of whether the tool uses classic CV filters (Canny,
background subtraction/silhouette) to *find* the skater — it currently doesn't (see
CLAUDE.md: Canny is only used for the ice line/horizon, color histograms only for
target selection/tracking and the `midline_dev` quality flag). The idea here is
different: not replacing the net with a filter, but putting a filter **alongside** it
as an independent check on what the net returns.

**What that could look like:**
- A silhouette/background mask (e.g. `cv2.createBackgroundSubtractorMOG2`/KNN, or
  simple frame differencing) around the reported bbox: does it match a moving object,
  or does it sit on stationary ice/boarding? Could flag an early wrong target choice or
  a tracker that jumped onto a bystander, alongside the existing color gate
  (`COLOR_MATCH_MIN`/`COLOR_SPLIT_MIN` in `skate_yolo.py`).
- Edge detection (Canny/contour) to check whether a leg keypoint lands on a real
  edge/contour rather than in thin air — similar to what `_midline_deviation` already
  does with color backprojection for the knee, but as a generic edge check instead of
  color-specific.
- Could serve as an **extra quality flag** alongside existing signals (`midline_dev`,
  the color gates, the RTMPose keypoint score) — not to auto-correct, the same way
  `_midline_deviation` currently only measures and doesn't correct.

**What makes this more promising since July 2026:** the fixed **frontal + horizontal
camera** assumption (see the top of this document) usually also means a fixed camera
position per clip — the background (ice, boarding, audience) then only changes because
of the moving skater(s), which makes background subtraction/frame differencing a good
deal more reliable than with a panning camera.

**Still fully open:**
- No measurement at all — this is purely a direction, not a design. It would first
  need to show that such a filter signals something the existing color/score signals
  don't already catch (e.g. on the clips from "Loose ends in target selection" above,
  where the color gate let a wrong target choice through).
- Which OpenCV technique (MOG2/KNN/frame-diff/Canny-contour) and at what level
  (bbox plausibility? a single keypoint? the whole frame?) — to be chosen after a
  first trial, not up front.
- As with any change to the measurement: only introduce after an A/B with
  `skate_eval.py` (coverage, events, bone-length CV, golden reference) — a control
  filter that adds its own noise is worse than no control at all.
- **Priority: low** relative to phase 2 (the next planned step, see the top of this
  document) — this is a note for later, not a planned phase.

## Order & scope (rough estimate)

| Phase | What | Scope |
|---|---|---|
| 0 | Serialization (`npz` + plain landmarks) | ✅ **done** (16 Jul 2026) |
| 1 | `skate_db.py` + library GUI + new-analysis flow | ✅ **done** (18 Jul 2026) |
| 2 | Progress graph, notes, export | small, 1 session — *only after the analysis is done* (see phase 2) |
| — | Batch analysis (extra, outside the phasing) | ✅ **done** (20 Jul 2026) |
| — | Compare skaters + `VideoPlayer` refactor (extra) | ✅ **done** (27 Jul 2026) |
| — | Corner detection: no longer analyzing the corner (extra) | ✅ **done** (5 Aug 2026) |
| — | App version per analysis + info dialog (extra) | ✅ **done** (6 Aug 2026) |
| 3 | Skeleton editor with flow-out + undo | ✅ **done** (20 Jul 2026; placing skeletons on gap frames 31 Jul 2026) |
| 4 | Settings, shared folder, conflict handling | ✅ **done** (20 Jul 2026) |
| 5 | Horizon via two tracked points | *nice-to-have (not now — horizontal camera)*; medium, 1–2 sessions (step 4, handing off points, is most of the work) |
| 6 | Faster analysis | **step 3 (a lighter pass-1 model) first** — 1 session incl. an A/B against the golden reference; export/DirectML as a separate experiment. Steps 1/2/6 are edge work (see the measurement in phase 6) |
| 7 | Perspective correction via track lines | *nice-to-have (not now — frontal, horizontal camera)*; large, 2–3 sessions (step 3, the 3D reconstruction, is research work — validate on test material first) |
| 8 | Trimming fragments in the app + `bronvideo`/recordings in the library | ✅ **done** (11 Aug 2026) |
| — | Loose ends in target selection (stitch gate, MediaPipe click) | **open** — small code, the A/B is the work; see its own section above |
| — | Following a small skater from a drawn box (spyglass) | ✅ **done** (11 Sep 2026) — see its own section above |
| — | OpenCV filters as a check on the neural net (idea) | **open, unplanned** — no design yet, low priority; see its own section above |

## Open questions (decide when the phase begins)

- ~~**Phase 1**: always copy the video?~~ **Decided (Jul 2026): always copy, leave the original in place.**
- ~~**Phase 1**: import old "loose" analyses?~~ **Decided (Jul 2026): not needed.**
- ~~**Phase 3**: also allow editing points on frames without a detected pose (point "placement" instead of dragging)?~~ **Built (31 Jul 2026): yes** — a guided click sequence of 8 points with a prefill from the neighboring frames, plus a coverage counter. See the phase 3 addition.
- ~~**Phase 4**: which cloud provider does the team actually use?~~ **Decided (Jul 2026): Google Drive** (Mirror mode, so every file local on disk). Conflict detection is provider-agnostic (any `*.db` next to `skate.db`).
- **Phase 5** *(nice-to-have, not now)*: under the current assumption (horizontal camera) this phase isn't needed. Becomes relevant only if filming ever does happen with a tilted/wobbling camera; then also: does the camera pan along (handing off points needed) or does it sit on a tripod?
- **Phase 6**: how much measurement deviation is acceptable for the "fast" profile? (Proposal: events must stay identical, angles may differ ±1°.) And: is a lighter pass-1 model even a *profile*, or simply the new default? If the A/B shows `yolo26m` gives the same coverage and events, there's nothing to choose — then step 5 falls away.
- **Corner detection**: there's still no clip from our own setup that **starts** in the corner; the thresholds (`CORNER_IN`/`CORNER_OUT`) are currently set on the margin from four TV clips that **end** in the corner. Once such a clip exists: `python skate_eval.py corner analysis.npz` and adjust if needed.
- **Phase 8**: ~~may the trim dialog use a `CAP_PROP_POS_FRAMES` seek?~~ **Built (11 Aug 2026): yes**, with an explicit `fast_seek` flag on `VideoPlayer`; measured deviation 0–1 frame on two iPhone .MOV files and up to 4 frames (168 ms) on a VFR clip where the frame count itself is already off. ~~Recode with cv2 or stream-copy with ffmpeg?~~ **Decided (11 Aug 2026): cv2/`mp4v`** — see the A/B under phase 8. ~~How much margin around a fragment?~~ **Decided (10 Aug 2026): no margin at all** — trim exactly on the marked frames; the trainer judges the boundaries themselves while marking. **Still open:** may the trainer pick the target skater already in the trim window, on a frame of their own choosing, instead of afterward on frame 0 of the clip? Not built; it requires `_choose_seed(..., doel_frame)` in the YOLO backend plus the frame number in `settings_json`, and frame 0 of a fragment is already exactly the frame "start" was pressed on — so the urgency has become small.
