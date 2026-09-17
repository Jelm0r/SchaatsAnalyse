# Bug report — code review July 2026

Review of `skate_analysis.py`, `skate_yolo.py`, `skate_gui.py`, `skate_db.py`,
`skate_eval.py`, `skate_perspective.py`, with **emphasis on tracking the skater**. Every
finding marked **[proven]** was demonstrated with a reproduction script or on the real
analyses in `Documents\SkateAnalysis\media\*\landmarks.npz`; **[latent]** = the code
path exists and is reachable, but didn't occur in the current library.

Self-tests that were already fine: `python skate_db.py` → OK, `python
skate_perspective.py` → PASS, all six modules compile.

---

## Recheck 26 Jul 2026 — all 24 findings re-measured

Every fix was independently verified: the old code was pulled out of git
(`git show HEAD:skate_analysis.py`) and run side by side with the new code on the same
data, so "resolved" rests on a measurement, not on the fix description.

**All 24 findings are gone.** Evidence per block:

| | evidence |
|---|---|
| **A1** | `ev.angle` minus the raw angle on the end frame = **+0.0° in all 100 events across 18 clips** (was +0.2…+10.0). |
| **A2** | "clips where the last, still-running push is NOT flagged as truncated: **NONE**". Average excl. truncated: `521f6916` 51.8→38.2°, `4b066ea3` 58.2→51.4°, `542bdc04` 56.3→42.1°. |
| **A3** | `b9896526` 13→8 events, `542bdc04` 9→8, `6c3eacd5`/`bbf6fc5c` 6→5. |
| **B1** | Reproduction: coasts on frames 6–10, picks the target back up on frame 11, **0×** the bystander (was: permanently stuck on the bystander). |
| **B2** | 3 and 4 missed frames now recover immediately (was: permanently lost from 3 onward). |
| **B3** | Big stationary bystander (x=0.75) vs. a smaller moving skater (x=0.10) → picks **0.10**; the old "biggest pose" picked 0.75. |
| **B4** | Decisive: old flips the unreliable frames 7 and 13 against their neighbors (`RRRRRRRLRRRRRLRRRRRR`), new keeps the sequence consistent (`RRRR…`). |
| **B5** | On real landmarks: bone-length CV **4× better, 0× worse**; `db2ad2a7` crossed frames **330 → 171**, tibia CV 0.136→0.106 and 0.145→0.120. |
| **B6** | Stationary hips no longer give a false `hip_dx = 60`; the tiebreaker no longer always answers `'left'`. |
| **C1–C7** | Synthetic tests on the helpers: no seed → `RuntimeError`; missed click reported; short seed 1→16 detections padded out; color measured on the chain-facing side (correctly rejects and accepts the mirror case); a bbox fallback no longer cuts, a genuinely different suit still does; a 2-frame overlap gets stitched. Every call site of the changed signatures checked. |
| **D1** | `open_db` on a v3 database now raises `LibraryTooNew` instead of downgrading it to v2. |
| **D2–D6** | Verified in the diff: cooperative abort + `wait()` before closing, `stopped` flag in `annotate`, `_report_read_error`, only `schaats*.db` counted as a conflict copy. |
| **E** | All 9 items found back in the diff. |
| **regression** | Golden reference on the right clips (`Schaats frontaal.MOV`): angle error and point error **exactly equal** old vs. new — no regression on landmark accuracy. Self-tests `skate_db.py` → OK, `skate_perspective.py` → PASS, all six modules compile. |

### What's still standing after the fixes (newly found)

State as of 26 Jul 2026: **R1 and R3 are resolved** (see the fix blocks below). R2 and
R4 deliberately remain: R2 is, on the balance of real data, an improvement (see B5) and
R4 is a trade-off, not a bug.

**R1. A short stance run can still report the "standing up" as a push.**
**[proven]** — ✅ RESOLVED (26 Jul 2026)

> **Fix:** the truncation flag has been generalized into **one flag with a reason**:
> `FrameResult.push_incomplete` → `PushEvent.incomplete`, with `INCOMPLETE_TRUNCATED`
> ("truncated") and the new `INCOMPLETE_NO_PUSH` ("no full push"). Every place that
> filters the statistics only has to check for truth, while the GUI tooltip asks the
> user the right question — "the video ended" calls for a longer recording, "no full
> push observed" calls for a look at the leg assignment. `TRUNCATED_MARKER` stays
> literally `"truncated"`, so existing event caches keep working unchanged;
> `list_analyses` now applies one `NOT LIKE` per reason (`INCOMPLETE_MARKERS`).
>
> The criterion is **geometry, not a tuning number**: if the lower leg at the chosen
> completion stands less than `EXTENSION_MIN_SLOPE_DEG` = 20° out of vertical (i.e.
> push angle > 70°), the sideways component of the push is sin(20°) ≈ 0.34 — there
> simply wasn't a sideways push, and the plateau only covered the standing-up phase.
>
> **Measured across the 18 saved analyses (100 events):** the three named events get
> `INCOMPLETE_NO_PUSH` and drop out of avg/min/max; all **84** previously-healthy
> events still count, the 13 already-truncated ones keep their own reason. Event count,
> L/R order, and every angle per clip are unchanged — it's purely a flag. `c9fcd0be`
> avg 60.6 → **41.8°**, `db2ad2a7` 45.0 → **41.7°**, the other 16 clips exactly the
> same. The golden reference on `1e9ae474`/`2651bcdb`/`542bdc04`/`cb13bb1e` gives the
> same angle and point error as before the change (landmarks untouched).
>
> **Two things worth knowing.** (1) The measure this report originally proposed — the
> distance between the chosen angle and the standing-up within the same run — turned
> out on the real data NOT to separate: healthy events sit at 0–56° there and the three
> suspect ones at 3–22°, so the groups fully overlap (checked for the event's
> `max_angle`, the maximum over the whole run, and the maximum up to completion). Hence
> the geometric upper bound instead. (2) The closest neighbor under the threshold is
> `db2ad2a7` ev9 (66.7°) — itself a borderline case too: a run of 77 frames (1.5 s)
> where the leg assignment never switched, so three half-strokes in one "run". That
> 3.3° margin is the tightest spot of this threshold; if the leg assignment improves,
> the margin widens.

`min_run` and the truncation flag caught most of the phantom pushes, but a run that's
just long enough, whose extension plateau covers only the standing-up phase, produced
an ordinary, counted push with an angle of 74–83°. Measured: **3 of the 100 events** in
the library — `c9fcd0be` ev1 (8 fr / 0.27 s, 79.4°), `db2ad2a7` ev19 (23 fr, 83.1°) and
ev21 (46 fr, 74.3°). Effect: `c9fcd0be` avg 41.8 → **60.6°**, `db2ad2a7` 41.7 → 45.0°.
Recognizable pattern: short duration + angle above ~70°. An upper bound on the push
angle, or requiring the plateau to contain a falling angle flank, would close this.

**R2. The joint L/R decision can no longer repair an error in a single joint pair.**
If the detector swaps ONLY the knee labels (or only the ankles), then
`swap_cost ≈ identity_cost` and the new code doesn't swap anything — the crossed
skeleton stays as is. The old code did repair that case. Demonstrated synthetically (5
frames crossed, both directions). On the real data the new version is clearly better
on balance (see B5 above), so this isn't a reason to revert — just something to be
aware the gap exists.

**R3. `skate_eval._bone_length_cv` can produce a meaningless number.**
**[proven]** — ✅ RESOLVED (26 Jul 2026)

> **Fix:** the clamp `np.maximum(trend, 1e-6)` is gone. Frames where the trend dips
> below `TREND_MIN_FRAC` (0.25) × the median bone length have no usable denominator and
> are now **skipped**; `_bone_length_cv` now returns `(cv, n_used, n_skipped)` and
> `print_metrics` appends that after the value: `CV tibia_l: 0.098 (n=170, 3 skipped)`.
> If fewer than 10 usable frames remain, no number is printed at all, just
> "unreliable/too few measurements". Checked across all 18 npz's: every CV now falls
> between **0.023 and 0.238** (or explicitly no verdict) — exactly the case from this
> report, `4b066ea3` `tibia_l`, goes from **981184.372 → 0.098** with 3 skipped frames.

On `4b066ea3` it read `CV tibia_l: 981184.372` (n=173). Cause: the Savitzky-Golay trend
goes **negative** in one frame (−0.87 px), after which `np.maximum(trend, 1e-6)`
explodes the ratio to 1.3·10⁷. The underlying data is shaky too (tibia 10.4 px vs.
median 54.9 — swing leg behind the stance leg), but the metric ought to *report* that
instead of printing a seven-digit number. This is the tool future changes get measured
against, so it's worth the effort: skip trend values under a fraction of the median and
report the number of skipped frames.

**R4. A deliberate trade-off in the new reseed gate.** `TRACK_RESEED_GATE` (0.25)
refuses a reseed outside the extrapolated last known spot, and `_verwacht` caps the
extrapolation at `hervind_frames`. If the skater is lost long enough and resurfaces far
away, they no longer get picked up — the tracker would rather produce nothing than the
wrong skeleton. That's the right call, but it means a long occlusion now leaves a
permanent gap in coverage instead of a (possibly wrong) lock.

---

## A. Measurement errors — these directly hit the numbers in the table

### A1. The push angle is a lagging average, not the angle at the chosen frame **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix**: `segment_pushes` now fills `angles` with `r.angle` instead of `r.smooth_angle`,
> so `angle`/`min_angle`/`max_angle` all three come from the same per-frame sequence and
> `angle` is the angle of the completion frame itself. On top of that, `smooth_angle`
> went from a trailing deque to a **centered** (zero-lag) average (`_set_smooth_angle`),
> per contiguous run of the same stance leg — which makes the HUD value honest and
> takes care of D4 (angle buffer mixing two legs) along the way. The GUI graph now plots
> `r.angle` (the same quantity as the table, see the E table). Checked across the
> library: the deviation `ev.angle` minus the raw angle on the end frame is **exactly
> 0.0° for all 100 events** (was avg +3.3°, max +12.1°); the segmentation itself doesn't
> change. Reported angles therefore drop by an average of 3.3°.

`determine_push_from_extension` deliberately picks the frame with the **flattest
lower-leg angle** within the extension plateau. But `segment_pushes` didn't report that
frame's angle: it filled `huidig['hoeken']` with `r.smooth_hoek` and took
`angle=hoeken[-1]`. And `smooth_hoek` was a **trailing** average over the last
`smooth_n` (default 5) frames. Because the angle drops toward that minimum, the average
sat systematically above it.

Measured across all 18 saved analyses — `ev.angle` minus the raw angle on the end
frame:

| analysis | deviation per event |
|---|---|
| `1e9ae474` | +2.3 +5.3 +3.6 +4.3 +1.6 +2.7 |
| `4fdb2b0e` | +5.8 +6.0 +7.7 |
| `521f6916` | +7.7 +6.1 +7.0 |
| `542bdc04` | +5.3 +5.0 **+10.0** +3.5 +8.0 +2.3 +5.2 +3.9 +6.1 |
| `b9896526` | +1.7 +1.3 +1.5 +0.6 +1.6 +1.5 +1.2 +7.5 |

**Always positive (too steep), in 100% of events**, up to +10°. For a tool that reports
push angles to 0.1°, this was the biggest error source in the whole program.

Fix: let the chosen completion frame carry its own (possibly smoothed) angle, e.g.
`angle = r.angle` of the end frame, or compute `smooth_angle` **centered** (zero-lag,
the way the landmark smoothing already does) instead of with a trailing deque.

### A2. A truncated last stance run produces a fully-fledged push **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** a run that doesn't end on a leg switch but on the end of the
> video/pose segment sets `FrameResult.push_incomplete` → `PushEvent.incomplete`. Such
> an event stays visible (gray row + tooltip, `incomplete` column in the CSV) but falls
> outside avg/min/max in the GUI and outside `AVG(angle)` in the library list. Runs
> truncated at the *start* still count (there only the load phase is missing). Effect
> on the saved analyses, e.g. `4b066ea3`: avg 55.1 → 51.4°, max 73.7 → 55.0°.

`determine_push_from_extension` treated every stance run the same, including the run
cut off by the **end of the video** (or of a pose segment). There the push wasn't done
yet, so the extension plateau only contains the first half and the "flattest angle" is
actually the standing-up.

In **18 of 18** clips the last run hits the last pose frame. The effect on the last
push:

| analysis | other pushes | last push |
|---|---|---|
| `4b066ea3` | 49.8 57.3 52.2 55.5 57.6 | **79.5** |
| `7e7d0a66` | 49.7 57.2 52.2 55.4 57.5 | **79.4** |
| `b9896526` | 38.9 36.3 39.2 40.6 39.7 40.5 42.7 | **55.3** |
| `521f6916` | 40.8 49.3 | **58.7** |
| `4fdb2b0e` | 46.7 51.0 | **63.0** |

That value fed into `avg`/`max` in the status line and into `AVG(angle)` of the library
list.

Fix: mark a run whose last frame coincides with the end of the pose segment as
incomplete — no event, or an event with `note="truncated"` that falls outside the
statistics. Same for the run that starts at frame 0 (less harmful, since the peak sits
at the end of a push).

### A3. The stroke-time prior undermines itself **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** the median now only runs over runs of at least `EXTENSION_MIN_RUN_S`
> (0.2 s); if there are none, it falls back to all runs. Measured on the saved
> analyses: `1edc50ac` `min_run` 5 → 11 fr (half stroke 14.5 → 32), `8725ef71` 4 → 10 fr
> (12 → 27.5), `542bdc04` 3 → 5 fr (9 → 11.5), `b9896526` 12 → 16 fr. In `542bdc04` the
> four shortest phantom runs (4–5 fr) disappear as a result.

```python
lengths = [len(run['idx']) for run in runs]
half_period = float(np.median(lengths)) if lengths else 0.0
min_run = max(2, int(round(EXTENSION_MIN_STROKE_FRAC * half_period))) if half_period else 2
```

The median was taken over **all** runs, including the noise runs the prior is
supposed to filter out. With lots of L/R flips, the median drops and the threshold
drops with it — right when you need it most:

| analysis | run lengths | median → `min_run` | real half stroke |
|---|---|---|---|
| `1edc50ac` | `[1,1,1,3,29,34,36,32,26,2]` | 14.5 → 5 fr | **32 fr** |
| `8725ef71` | `[24,2,12,1,31,32,3]` | 12.0 → 4 fr | **27.5 fr** |
| `542bdc04` | `[5,5,4,5,9,23,19,16,14]` | 9.0 → 3 fr | 11.5 fr |
| `db2ad2a7` | `[1,1,12,1,20,35,34,…]` | 24.5 → 9 fr | 29.5 fr |

Result in `542bdc04`: nine "pushes", five of them 4–9 frames (0.13–0.34 s) with angles
up to 86.4° — not a push but the standing-up during a noise flip.

Fix: estimate `half_period` robustly, e.g. the median of only the runs at or above an
absolute lower bound (0.2 s), or the median of the top half of the run lengths;
possibly iteratively (apply the threshold → re-estimate).

---

## B. Tracking the skater — MediaPipe backend (`TargetTracker`)

### B1. Reseeding uses the stale click from frame 0; the "last known spot" branch is dead code **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** `TargetTracker.laatste_bekend` now lives alongside `self.centroid` (the lock
> flag) and is NOT cleared on loss, so the reseed branch works. That branch picks the
> candidate closest to the **extrapolated** last known spot (`_verwacht()`) and refuses
> a reseed outside `TRACK_RESEED_GATE` (0.25) — better no pose at all than a skeleton on
> a bystander. The click point now only counts on a cold start (`laatste_bekend is
> None`). Verification: the reproduction above now picks the bystander 0× and picks the
> target back up on frame 11.

`_seed` had three branches, with as the second:

```python
if self.centroid is not None:
    # Just lost: take the skater closest to the last known spot.
```

That branch was **never** reached. Reseeding happens on
`self.centroid is None or self.kwijt > self.hervind_frames`, but at exactly the moment
`kwijt` crosses the threshold, `self.centroid` gets set to `None` — in both loss
branches. So when reseeding, `centroid` was always `None`, falling back to either the
**click point of frame 0**, or "biggest pose".

Reproduction (`scratchpad/test_reseed.py`): the target starts at x=0.20 (where the user
clicks) and moves right; a bystander stands still at x=0.90; the target isn't
detectable on frames 6–10.

```
 frame | target at | picked | what is that?
    5  |   0.35    |  0.35  | TARGET
    6  |   0.38    |   -    | lost (coast)
   ...
   10  |   0.50    |  0.90  | << BYSTANDER
   15  |   0.65    |  0.90  | << BYSTANDER
```

From frame 11 the target is visible again at 0.53 — much closer to the last known spot
(0.35) than the bystander (0.90) — but the tracker is already locked onto the bystander
and never recovers. In the GUI you'd see a skeleton on the wrong person with plausible
angles.

Fix: store the last position separately from the "do I have a lock" flag (e.g.
`self.laatste_bekend` alongside `self.centroid`), so the second branch works; and
refuse a reseed that lies beyond a generous gate from the last known spot.

### B2. During coast the prediction isn't extrapolated forward **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** the prediction now shifts along (`_verwacht(centroid, kwijt+1)`, horizon
> capped at `hervind_frames` + clamped to the frame) and the gate grows with it:
> `min(gate × (1 + TRACK_GATE_GROWTH × kwijt), TRACK_GATE_MAX)` (ceiling equal to
> `TRACK_RESEED_GATE`, so acceptance doesn't jump at the coast→reseed transition) — the
> counterpart of YOLO's `STITCH_GATE_*`. On a match the speed is divided by the gap
> length, otherwise a single gap would blow up the estimate N-fold. Verification: gaps
> of 1 through 6 frames all recover now (`++++......++++++`), and the speed estimate
> after a 4-frame gap stays at 0.0298 vs. the real 0.03.

The prediction was always `centroid + 1 × speed`, even after N missed frames — while
the real jump by then is `(N+1) × speed`. `self.centroid` stayed put at the last match.
The gate `TRACK_GATE` didn't grow either.

Reproduction (`scratchpad/test_coast.py`, speed 0.045/frame = ⅓ of the gate):

```
1 frame missed  -> ++++.++++++    recovers: YES
2 frames missed -> ++++..++++++   recovers: YES
3 frames missed -> ++++...XXXXXX  recovers: NO
4 frames missed -> ++++....XXXXXX recovers: NO
```
(`X` = the skater IS detected, but falls outside the gate.)

At ≥3 missed frames the skater is **permanently** lost: they keep coasting for
`hervind_frames` (at 30 fps = 15 frames) and then reseed via B1 onto the stale click
point. Three missed frames in a row is nothing unusual with motion blur.

Fix: shift the prediction forward per coast frame (`centroid += speed` on loss), or
scale the gate with the number of missed frames — exactly what the YOLO backend
already does with `STITCH_GATE_BASIS + STITCH_GATE_GROWTH × gap`
([skate_yolo.py](skate_yolo.py)).

### B3. `TargetTracker` has no "the skater is moving" criterion **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** on a cold start **without a click**, `analyze_frames` first watches for
> `SEED_WARMUP_S` (1 s). Those buffered frames go through `_track_candidates` (raw
> nearest-neighbor tracks) and `_choose_moving_target` picks out the biggest *mover*:
> median bbox area × distance covered, with `SEED_MIN_MOVEMENT` as a lower bound — the
> same rule as the YOLO backend. That point goes into the tracker as `doel_punt`, after
> which the buffered frames are still played back (no frame is lost); frames before the
> chosen track's start frame stay empty, since the target isn't in frame there yet.
> Nothing changes with a mouse click. Verification (`test_stap6_analyse.py`, TEST 1): a
> big stationary bystander at x=0.90 vs. a moving skater at x=0.20 — the old rule picked
> the bystander, the new one follows the skater the whole window.

On a cold start without a click, it picked `max(..., key=_bbox_area)`: the **biggest**
pose. A bystander along the boards is regularly bigger in frame than the skater riding
farther away. The YOLO backend explicitly solves this with `MIN_MOVEMENT` and "median
area × path length" ([skate_yolo.py](skate_yolo.py)); the MediaPipe tracker lacked
that. Per frame, only proximity counted too, so a stationary bystander closer to the
prediction beat the moving target.

Fix: weigh the displacement per candidate over the first ~1 s of a cold start.

### B4. The L/R majority vote also flips undecided frames **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** `_fix_lr_swaps` now keeps a `besloten` (decided) mask; the majority is taken
> only over those frames and the inversion only touches them. Frames skipped due to low
> visibility stay untouched.

```python
if gewisseld.mean() > 0.5:           # chain anchored wrong: flip the labels
    gewisseld = ~gewisseld
```

`gewisseld[t]` stayed `False` for frames deliberately **not decided** because
visibility was too low (a `continue`). After the inversion, exactly those frames got
marked as "swap" and had L/R flipped there — with no basis at all, and against the rest
of the sequence.

Reproduction (`scratchpad/test_tracking.py`, TEST 3): two frames with
`visibility = 0.05` had their knees swapped after the fix while every surrounding frame
stayed correct. That plants an L/R error in those frames that pollutes the bone-length
check and the L-R ankle signal of the push cycle.

Fix: keep a separate `besloten` mask and only invert `gewisseld & besloten`, or take
the majority only over the decided frames and leave the rest untouched.

### B5. Knees and ankles get swapped independently of each other **[latent]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** `_fix_lr_swaps` now makes one decision per frame for the **whole leg** — knee
> and ankle, with heel and toe riding along — on the **combined** continuity cost of
> both joint pairs. Verification (`test_stap6_analyse.py`, TEST 2): on anatomically
> correct input with flickering L/R labels (legs nearly on top of each other) the old
> fixer produced, in 3 of 8 runs, 6–21 of the 30 frames with leg A's knee on leg B's
> ankle; the new one zero in all eight. TEST 3 confirms a persistent swap is still
> repaired as before. The crossing ambiguity below still stands: that's a different
> assumption (continuity), not a missing link.

`pairs = ((L_KNEE, R_KNEE, ()), (L_ANKLE, R_ANKLE, (heel, toe)))` — two separate
decisions. If they came out differently, the left knee ends up hanging off the right
ankle: an anatomically impossible skeleton with a nonsensical tibia length. In my
synthetic test they happened to come out the same, so it wasn't demonstrated — but the
link is missing.

Also notable: the continuity cost picks the **non-crossing** interpretation even on
legs that genuinely cross (visible in `scratchpad/test_lr_koppel.py`, where the fix
reverses the real crossing). Filming frontally on the straight, that's rarely fatal; in
corner work / overtaking it is.

Fix: one joint decision per leg (knee+ankle+heel+toe together), on the summed cost.

### B6. `determine_push_leg` compares the right hip with the **left** hip **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_analysis.py](skate_analysis.py)

> **Fix:** the tiebreaker now compares the hip *midpoint* now with the one from 3
> frames ago, the way `detect_weight_on_leg` already did. With stationary hips it now
> comes out at 0 and no longer always answers `'left'`.

```python
heup_dx = lm_data['r_heup'][0] - heup_history[-3][0]
```

`heup_hist` holds tuples `(l_hip_x, r_hip_x)`, so `[-3][0]` is the **left** hip from 3
frames ago. So it measured no shift at all, just the constant hip width.

Reproduction (TEST 4): with fully stationary hips (l=120, r=180) this produced
`heup_dx = 60` instead of 0, and the tiebreaker always answered `'left'`.

This sits in the per-frame fallback (`cyclus=False`), so it's not active in the normal
pipeline — but it is in the diagnostic mode. Fix: `heup_history[-3][1]`, or compare the
hip midpoints the way `detect_weight_on_leg` does.

---

## C. Tracking the skater — YOLO backend

### C1. No seed found → silent, fully empty analysis — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** `analyze()` now raises a `RuntimeError` with an explanation if `_choose_seed`
> yields nothing; the GUI shows that as "Error during analysis" and no empty analysis
> ends up in the library.

```python
seed = _choose_seed(tracklets, frames, doel_punt)
target_per_frame, ref = {}, ColorReference()
if seed is not None:
    ...
```

If `seed` is `None` (not a single tracklet, e.g. because ByteTrack handed out no IDs),
`target_per_frame` stayed empty, no frame got `pose_found`, and the rest of the
pipeline ran through without error. The GUI would then just report "0 pushes found"
without saying nobody had been tracked — and would save that empty analysis to the
library.

Fix: `raise` with an understandable message, or a warning back to the GUI.

### C2. A click that hits nobody silently falls back to "biggest mover" — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** `_choose_seed` returns `(tracklet, click_missed)`; on a missed click, a
> message goes to the GUI via `analyze()`'s new `waarschuwing_callback`
> (`AnalyseWorker.warning` → box after completion; batch → in the summary).
>
> **Addition 26 August 2026 — the click missed far too often.** The 60-frame search
> window was too short: the skater you point at is often still too far away to be
> detected on frame 0. On `00005 8-41` they only showed up on frame 84 (3.4 s), so the
> click by definition couldn't hit anyone, and the trainer saw the warning while
> nothing was actually wrong with their click. Now: window `CLICK_SEARCH_S` (6 s,
> fps-independent), and if the click still hits nobody within that, the detection
> closest *next to* it counts, within a gate that grows per second. Checked on the
> dumped detections of that clip: the click sits on frame 84 inside the intended
> skater's bbox (distance 0.000) and the chain is identical to the old fallback's (197
> frames, 27–280) — only the message disappears. A click that already hit within 60
> frames behaves unchanged: that loop returns on the first hit.

The click was only searched for in the first `KLIK_ZOEK_FRAMES` (60) frames. If it hit
nobody, the biggest mover followed without any message — possibly the other skater.
The user thinks their choice was honored.

Fix: signal "your click couldn't be linked to a skater; now following the biggest
mover" to the GUI.

### C3. The seed tracklet gets no minimum length — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** two sides. (1) `_choose_seed` gives priority, when a click hits several
> tracklets, to one of at least `SEED_MIN_LEN` (5) detections; if the click only hits
> short fragments, the click just counts. (2) `_stitch_chain` starts with
> `_bootstrap()`: as long as the chain is shorter than `SEED_MIN_LEN`, directly
> adjoining fragments (gap ≤ `BOOTSTRAP_MAX_GAP`) get attached purely **on position** —
> the color reference is still too thin there to test anything against. Only after that
> does color become the gatekeeper. Verification (`test_stap6_yolo.py`): a seed of one
> detection grows into a chain of 14 frames with a reference built from 14 histograms.

The color reference in `_stitch_chain` was filled exclusively from the seed tracklet.
If the click lands on a fragment of 1–2 frames (quite possible after
`_split_by_color`), the reference is a single histogram and the whole chain stitching
is built on it.

Fix: require a minimum tracklet length for the seed, or fold in the color of the
best-fitting neighbor fragments on a short seed.

### C4. `_color_sim` looks at the wrong side of the tracklet when stitching backward — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** `_color_sim(t, direction)` measures `t[:10]` at direction +1 and `t[-10:]` at
> −1 — always the side bordering the chain. Verification (`test_stap6_yolo.py`): a
> candidate whose suit color drifts scores 0.38 at the start (below `COLOR_MATCH_MIN` =
> 0.45) and 0.75 at the end; it was rejected before and now gets stitched back in.

```python
def _kleur_sim(t):
    sims = [s for s in (ref.sim(d.hist) for d in t[:10]) if s is not None]
```

Always the **first** 10 detections. In `_probeer(-1)` it's the **end** of the candidate
that borders the start of the chain; there the color (lighting, scale) is best
comparable. For a long tracklet where the color drifts, that could push a correct
candidate below `COLOR_MATCH_MIN` (0.45).

Fix: `t[:10]` at direction +1, `t[-10:]` at direction −1.

### C5. Mask and bbox histograms get compared against each other — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** `_torso_hist` now returns `(hist, hist_mask)` and `Detection` carries that
> origin (`hist_mask`, plus the property `ref_hist` = the histogram to the extent it
> may serve as evidence). Consequences: the color reference is now only built from mask
> histograms; `_split_by_color` no longer lets a bbox fallback cause a cut (it doesn't
> count toward the cut, but doesn't reset the counter either — there's simply no
> verdict); in `_stitch_chain` a bbox verdict counts as "uncertain" and the color
> threshold gives way to half a distance gate; and in both refinement routes a bbox
> histogram may no longer reject an estimate. Verification (`test_stap6_yolo.py`): five
> frames of bbox fallback in the middle of a tracklet no longer cut it (1 piece), while
> the same pattern with mask histograms still cuts as before (3 pieces).

`_torso_hist` produced either a histogram over the **torso polygon** (suit pixels
only), or — if the torso keypoints were below 0.3 — over a **rectangle from the bbox**
(including background: ice, boards, audience). Those two aren't interchangeable, but
they were held to the same thresholds (`COLOR_SPLIT_MIN`, `COLOR_MATCH_MIN`) against
each other and against the reference. A frame that fell onto the bbox fallback would
then easily drop below the split threshold → unnecessary tracklet cut → gap in the
chain.

Fix: mark the origin on the histogram and only compare like with like, or don't let
bbox histograms count toward the split decision (weak evidence only).

### C6. Tracklets overlapping the chain in time can never be stitched — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py)

> **Fix:** a candidate may overlap the chain by up to `STITCH_OVERLAP_GATE`-compatible
> frames as long as it genuinely extends it (`t[-1].frame > chain-end`, resp.
> `t[0].frame < chain-start`). The gate grows with `max(gap, 0)`, and the existing
> per-frame de-duplication (best color match wins) cleans up the duplicate frames.
> Verification (`test_stap6_yolo.py`): a candidate overlapping by 1 frame now gets
> stitched and the chain keeps 12 unique frames.

`0 < t[0].frame - end.frame` required the candidate to start entirely after the chain's
end. A candidate overlapping by one frame (happens around occlusions, where two IDs
briefly coexist) fell outside the selection and the gap stayed.

### C7. Unused parameter — ✅ RESOLVED (26 Jul 2026)
[skate_yolo.py](skate_yolo.py) — `_interpolate_target(target_per_frame, n_frames, fps)`
didn't use `n_frames`. Confusing, because the caller passed `info.total` while the
result list is based on `len(frames)`. If those two ever differ (a VFR .MOV regularly
misreports `CAP_PROP_FRAME_COUNT`), the signature suggests a consistency that isn't
there.

> **Fix:** parameter dropped; the signature is now
> `_interpolate_target(target_per_frame, fps)`.

---

## D. Robustness — library and GUI

### D1. `open_db` downgrades `user_version` of a newer database **[proven]** — ✅ RESOLVED (26 Jul 2026)
[skate_db.py](skate_db.py)

> **Fix**: `open_db` now refuses a newer database with the new exception
> `skate_db.LibraryTooNew` ("made with a newer version of the app … update the app"),
> and only writes `PRAGMA user_version` after a successful creation or migration — a DB
> already at the current version is no longer touched. The GUI catches that type
> separately in `_set_library`: its own dialog title + fallback to the default library
> folder, so nothing gets written into the newer shared folder. Self-test extended
> (`python skate_db.py`): a v3 DB with an extra column stays at v3 after `open_db`, with
> that column intact, and the exception gets raised.

```python
if versie == 0:      ...
elif versie < SCHEMA_VERSIE: _migreer(con, versie)
if versie != SCHEMA_VERSIE:  con.execute(f"PRAGMA user_version = {SCHEMA_VERSIE}")
```

At `versie > SCHEMA_VERSIE` — a colleague with a newer app version has written to the
shared Drive folder — nothing was migrated, but the version DID get set **downward**.
Reproduction (`scratchpad/test_db_downgrade.py`):

```
before open_db:  user_version = 3
after  open_db:  user_version = 2   (app SCHEMA_VERSIE = 2)
v3 column still there: True
```

The v3 app that opens it afterward sees v2 and runs `_migreer(from=2)` again →
`ALTER TABLE analyse ADD COLUMN video_bytes` → `duplicate column name` → `open_db`
fails and the GUI falls back to the default folder. In a shared cloud folder with
different app versions, that's a real way to break the library.

Fix: `if versie > SCHEMA_VERSIE: raise` with "this library was made with a newer
version of the app", and only write `PRAGMA user_version` after a successful migration
or creation.

### D2. Closing while an analysis is running: the QThread gets destroyed while it's still running — ✅ RESOLVED (26 Jul 2026)
[skate_gui.py](skate_gui.py)

> **Fix**: cooperative abort. Both workers got `abort()` + a flag that lets the
> progress callback — which each pass calls per frame — raise `AnalysisAborted`; the
> analysis stops within a single frame, without a signal and without saving.
> `closeEvent` calls `_stop_workers()`: if a worker is running, a confirmation prompt,
> then `blockSignals` + `abort` + `_wait_for_worker` (wait cursor, UI keeps redrawing,
> no mouse/keys). If the thread doesn't stop within the deadline — almost always a video
> copy in progress, which is deliberately **not** cut off halfway — closing is refused
> (`event.ignore()`) instead of destroying the thread. A `_shutting_down` flag makes
> sure a signal already queued no longer opens a dialog or switches pages. Batch's
> `requestInterruption()` ('Stop after this video') keeps its old meaning: the running
> video finishes and gets saved. Tested with a fake analysis in the scratchpad
> (`test_afbreken.py`): both workers stop within ~16 ms, without a signal and without
> saving; the 'stop after this video' route does save video A and skips B.

`closeEvent` stopped the play timer and closed the capture, but didn't wait on
`self.worker` / `self.batch_worker`. Closing the window during a (batch) analysis — and
that takes a long time at 2 s/frame — meant the QThread got destroyed at interpreter
teardown while it was still running (`QThread: Destroyed while thread is still
running`, usually a hard crash). Worse: `sla_analyse_op` could get aborted halfway
through the video copy.

Fix: in `closeEvent`, `requestInterruption()` + `wait(...)` on an active worker, or
refuse to close with a question to the user.

### D3. `skate_eval.py annotate`: 'q' doesn't stop the annotation loop — ✅ RESOLVED (26 Jul 2026)
[skate_eval.py](skate_eval.py)

> **Fix:** a `stopped` flag; after handling the frame the for-loop `break`s on it.
> Already-annotated work still gets written out.

```python
if res == 'q':
    points = None
    targets = []
    break
```

`targets = []` **rebinds** the name; the `for target in targets` loop iterates over the
original list object and just moves on to the next frame. The user kept getting served
frames after 'q'. (The counter in the label, `len(targets)`, then also became 0.)

Fix: set a `stopped` flag and `break` after the `while`, or `del targets[:]` + an
explicit check.

### D4. A detection gap clears the angle buffer, a leg switch doesn't **[latent]** — ✅ superseded by A1 (26 Jul 2026)
> The trailing deque is gone; `_set_smooth_angle` breaks the sequence on a gap **and**
> on a leg switch, and the table reports the angle of a single frame.
[skate_analysis.py](skate_analysis.py)

`hoek_buffer.clear()` only happened when there was no pose. On a stance-leg switch, the
deque kept the angles of the **other** leg. If an event ended within `smooth_n` frames
after the switch, the reported angle was a mix of both legs. That never happened in the
current library (events end late in the run, and where it nearly went wrong —
`542bdc04` event 2 — there happened to be a detection gap right before it), but it's
reachable on short runs. Disappears automatically once A1 is fixed by reporting the
angle of a single frame.

### D5. A failed frame read desyncs the display — ✅ RESOLVED (26 Jul 2026)
[skate_gui.py](skate_gui.py)

> **Fix:** `_report_read_error(idx)` stops playback, resets the display to the last
> valid frame (so the slider, table highlight and graph marker line up again), and
> reports it in the time bar: "Frame N could not be read — display stayed on M".

If `_read_frame_exact` returned `None` (past the end, or a decode error),
`_show_frame` would `return` immediately: `huidige_idx`, the slider, the table
highlight and the status line stayed on the previous frame while the user thinks
they've jumped further. Fix: report it, or fall back to the last valid frame index.

### D6. `detect_conflict_copies` reports every `.db` file — ✅ RESOLVED (26 Jul 2026)
[skate_db.py](skate_db.py)

> **Fix:** only names starting with the stem of `DB_NAME` count (`schaats….db`) —
> syncers append their marker after the filename. The self-test now checks that
> `schaats-LAPTOP.db` DOES and `adressen.db` does NOT get flagged.

Every `.db` file except `schaats.db` counted as a conflict copy — even a completely
unrelated database someone drops into the folder. In a shared folder that produces a
warning on every open AND on every "Refresh". Fix: match on the `schaats*.db` shape.

---

## E. Small / cleanup

All items below were resolved on 26 Jul 2026.

| Where | What | How resolved |
|---|---|---|
| `skate_analysis.py` `detect_weight_on_leg` | `been_kant` was computed and never used. | Line removed. |
| `skate_analysis.py` `draw_leg_overlay` | `if hoek > 0:` — on a negative angle (knee below the ankle, broken detection) the angle line was silently omitted instead of showing the problem. | The line is now always drawn, **red** at angle ≤ 0. |
| `skate_analysis.py` `smooth_landmarks_offline` | Pose segments < 3 frames were skipped: no L/R fix, no outlier rejection, no smoothing. On fragmented detection, raw data was left in place with nothing showing that anywhere. | The `< 3` guard is gone; the filters degrade gracefully on their own (SG returns a too-short window unchanged, the bone-length check skips a too-short stretch) and the L/R fix does its job. |
| `skate_analysis.py` `segment_pushes` | `angle` and `min_angle`/`max_angle` all three came from the `smooth_angle` sequence, while `determine_push_from_extension` works on raw angles. | Superseded by **A1**: all three now come from `r.angle`. |
| `skate_analysis.py` `analyze_video` | CLI progress divided by `total` from `CAP_PROP_FRAME_COUNT`; on VFR .MOV that could exceed 100%. | Percentage clamped at 100, denominator at `max(total, frame_nr)`. |
| `skate_gui.py` (import) | The GUI imported `_torso_centroid` (private) from `skate_analysis`. | Renamed to the public `torso_centroid`. |
| `skate_gui.py` `_zoom_wheel` | The video label's `wheelEvent` was overridden; without a loaded analysis, the mouse wheel did nothing AND didn't scroll the page either. | Without an analysis (or at delta 0), the event now passes through to `QLabel.wheelEvent`. |
| `skate_gui.py` (graph) | The graph plotted `smooth_angle` (lagging); after A1 this should show the same quantity as the table. | Superseded by **A1**: the graph plots `r.angle`. |
| `skate_eval.py` `calculate_metrics` | `cv, n = _bone_length_cv(...)`; `n` was discarded even though the number of measurements is exactly what says whether the CV means anything. | `n` is carried along as `n_<name>` and `print_metrics` shows `(n=…)` after every CV. |

---

## Proposed fix order

1. ✅ **A1** (angle of the correct frame) — the biggest and most systematic measurement
   error, small fix. Test with `python skate_eval.py metrics ... --golden
   golden_skate_frontal.json`; that golden reference already exists.
2. ✅ **A2 + A3** (truncated runs, robust stroke period) — together they remove the
   phantom pushes and the polluted averages.
3. ✅ **B1 + B2** (reseeding + coast extrapolation) — the two tracking bugs that can
   make it follow the wrong person. B2 is a two-line change.
4. ✅ **D1 + D2** (schema downgrade, closing during an analysis) — can cost data.
5. ✅ **B4, B6, C1, C2** — wrong or silent corrections.
6. ✅ **The rest** (26 Jul 2026): B3, B5, C3–C7, D3, D5, D6 and the full E table. With
   that, every finding in this review has been handled.

Reproduction scripts live in
`%LOCALAPPDATA%\Temp\claude\c--Apps-SchaatsAnalyse\9b13189b-…\scratchpad\`
(`test_reseed.py`, `test_coast.py`, `test_tracking.py`, `test_lr_koppel.py`,
`test_hoek.py`, `test_echt.py`, `test_runs.py`, `test_db_downgrade.py`,
`test_buffer.py`). If they need to stick around, that's a good reason for a real test
folder in the repo.

The verification of **B1 + B2** lives in `…\c1bbebfb-…\scratchpad\test_b1_b2.py`
(reseed, coast of 1–6 frames, crossing skaters as a regression check, speed estimate
after a gap).

That of **step 6** lives in `…\88e1387b-…\scratchpad\`: `test_stap6_analyse.py` (B3 +
B5, runs on the MediaPipe venv), `test_stap6_yolo.py` (C3–C6, runs on `.venv-yolo`) and
`test_stap6_e2e.py` (the whole YOLO pipeline on "Schaats frontaal.MOV").

That of **R1 + R3** lives in `…\137015c8-…\scratchpad\`: `meet.py` (all 18 npz's
through the pipeline → JSON, once on the old and once on the new code), `accept.py`
(R1's five acceptance criteria on those two dumps), `diag.py`/`diag2.py`/`dump_run.py`
(the search for a separating measure, including the proof that the angle-drop-across-
the-plateau does NOT separate), `cv_check.py` (R3: all CVs across all npz's) and
`gui_smoke.py` (a headless PySide6 test of the table color, both tooltips and the CSV
column).
