# To do after the crash of 13 Aug 2026 (cut + batch)

## Recurred: 25 Aug 2026, now in the bundled app (EXE.md step 5)

Same flow as 13 August — cut a recording, then run batch analysis on the fragment —
so **this is not a regression from bundling**, but the same bug in an environment
where it's less visible. What exactly happened:

- Fragment `00005 13-16.mp4` cut from `00005.MTS` (84 frames, 1920×1080 @ 25 fps, 5.0
  MB — **the cutting itself went fine**, the file is complete and readable), then a
  batch analysis was started with another analysis (`IMG_9001.mov`) still open on the
  view page.
- Crash during pass 1, at frame 21 of 84 (the bar shows 21/168 because the detection
  and refinement passes each get a half).
- Just before the crash Qt was already drawing incorrectly: header over the
  explanatory text, the batch progress bar in the middle of the recordings table, two
  status lines stacked on top of each other.
- Windows event log: `SkateAnalysis.exe` — **faulting module `Qt6Gui.dll` 6.11.1.0,
  exception code `0xc0000005` (access violation), offset `0x00000000000e3880`**,
  followed by a second notice `0xc000041d` (error in a callback). So a hard crash in
  Qt's drawing layer, not in the analysis: onnxruntime/DirectML were just doing their
  job at that moment.
- `%LOCALAPPDATA%\SkateAnalysis\skateanalysis.log` ends after
  `Loading …yolo26x-pose-dml.onnx` **with no traceback** — exactly what you'd expect
  from a C++ crash: Python is no longer in the picture. **Point 3 below is therefore
  the first step**, because in the exe there's no console and this log file is the
  only trail we write ourselves.
- Point 4 is now moot: the page file is set to automatically managed (commit limit
  29.7 GB at 15.2 GB RAM, peak usage 477 MB), so memory pressure is no longer a
  plausible explanation here.

### Where the crash lives: Qt's screen management (25 Aug 2026, established)

The crash offset from the WER report can be translated back into a function by
reading the **export table of `Qt6Gui.dll`** and looking up the nearest export before
the offset (script in the scratchpad; PE header → DataDirectory → exports, ~60 lines
of stdlib). Both times it points the same way:

| crash | offset | nearest export |
|---|---|---|
| 20:46 | `0xe3880` | exactly the start of `QScreen::geometry()` |
| 21:25 | `0xe4a79` | `QScreen::virtualSiblings()` + `0x29` |

That's the well-known picture of a **dangling `QScreen`**: Windows rebuilds the
screen configuration, Qt throws away the old `QScreen` objects, and whoever still
points at one falls over with `0xc0000005` (QTBUG-42985 and relatives). Our own code
touches `QScreen` in exactly one place — `set_window_size` in skate_gui.py, with a
`None` check — and never calls `geometry()`/`virtualSiblings()` itself anywhere: Qt
uses those internally when placing windows. And the crashing flow opens several
windows in a row (cut window, batch dialog, target picker per clip).

**Suspected trigger: the Snipping Tool.** The system log shows activity from
`Microsoft.ScreenSketch` at **20:45:49**, 19 seconds before the 20:46:08 crash — a
screen overlay is exactly the kind of event that makes the screen list get rebuilt.
**Not yet tested**: after the crash-log build the flow ran three times without a
crash (fragments of 84, 252, and 62 frames), but none of those runs involved a
screenshot. The decisive test is therefore: start a batch and trigger Win+Shift+S a
few times partway through.

### Confirmed with the stack trace (25 Aug 2026, 22:19 session)

Reproduction: cut → batch analysis, and hit **Win+Shift+S** a few times during the
analysis. Three earlier runs without screenshots went through fine; this one
crashed. What the log caught:

```
Windows fatal exception: access violation

Thread 0x00005d74 (most recent call first):     <- the analysis worker, just running
  File "skate_yolo.py", line 692 in _detect_all
  File "skate_gui.py", line 1417 in run

Current thread 0x00005528 (most recent call first):   <- this is where it blew up
  File "skate_gui.py", line 6715 in main            (sys.exit(app.exec()))
```

**The main thread goes down inside `app.exec()` with a one-line Python stack.** That's
the proof the bug isn't in our code: if it were in one of our slots, our functions
would show up in that stack. So the access violation happens in Qt-internal C++ code
triggered by a Windows message (a screen change) — exactly matching the `QScreen`
offsets above. The worker was meanwhile sitting in `cap.read()` and just got dragged
down with it.

This is the well-known Qt bug family on Windows: QTBUG-81359 (access violation in
`QWindowsWindow::checkForScreenChanged` on screen changes) and QTBUG-42985. This is
running **PySide6 6.11.1**; 6.11.2 exists, but the release notes don't mention a fix
for this.

**What this means in practice:** the damage stays limited to the video that was
running — `BatchWorker` saves per clip, so videos already finished earlier in the
same batch are already in the library. And the crash costs no measurement data: it
sits in the drawing layer, not in the pipeline.

### Attempt at a minimal reproduction: failed (25 Aug 2026, 22:25-22:50)

The goal was a 30-second test instead of two minutes, to be able to compare PySide6
versions cleanly. Four setups, each poked with 10-15 snip overlays (`ms-screenclip:`
+ Esc, the same event as Win+Shift+S):

| setup | outcome |
|---|---|
| bare Qt window, 20 repaints/s + worker thread | survived 10× |
| same + opening/closing a dialog every 0.7 s (via `screen().availableGeometry()`) | survived 12× |
| same + **the real YOLO model on DirectML** in a thread (provider confirmed) | survived 15× |
| **the app itself**, idle on the library page | survived 15× |

So: repaints, dialog windows, GPU load via DirectML, and the app at rest are **each
individually not enough**. The crash needs the real analysis flow (cut → batch, with
the overlay in between). That makes it expensive to reproduce — budget two minutes
per attempt — but it does rule out the cheap explanations, and that's exactly what a
bug report to Qt should state.

The test script lives in the scratchpad (`qtcrash/minimal.py` + `poke.ps1`); not put
in the repo since it's tooling, not part of the app.

**Careful when debugging:** two running sessions (the exe and a standalone script)
write to the same log file, and then the ordering can no longer be read reliably.
That already produced one wrong conclusion on 25 Aug — a notice under an exe header
turned out to come from the standalone script. One at a time.

**The next attempt is best done in the venv** (`start_gui.bat`): there's a console
with a traceback there, and a code change doesn't cost a 5-minute rebuild.

### Addressed (25 Aug 2026, evening): fewer windows + PySide6 6.11.2

Two interventions, both at the only two places where we have any influence. The
access violation itself stays Qt-internal — nothing to catch there — but its
**precondition** is a stale `QScreen` reference in a window that Qt walks past on a
screen change, and the number of such windows is something we do control.

**1. Window leak plugged (was point 1 below, and it was bigger than expected).** A
`QDialog` with a parent stays alive after `exec()` as a hidden top-level window,
complete with a native Windows window and a `QScreen` pointer. All eleven modal
pickers now go through `show_dialog(dlg)` in skate_gui.py (`try: return dlg.exec()` /
`finally: dlg.deleteLater()`), plus two windows that weren't on that list: the cut
progress `QProgressDialog` (`close()` only hides it) and the **splash screen**, which
stayed alive for the whole session after `finish()`. In the crashing flow — cut →
batch of seven clips — that accounted for ~16 leftover windows, some with their own
`VideoPlayer` and `VideoCapture` inside. This probably also explains the **garbled
drawing** right before the crash (header over the explanatory text, progress bar in
the middle of the recordings table).

*Re-measured* (offscreen, with `MainWindow`; scripts in the scratchpad): three bare
`exec()` calls leave three pickers behind, six calls through the helper leave zero.
`deleteLater()` and not `WA_DeleteOnClose`, because every call site reads the result
only *after* `exec()` — separately confirmed that the dialog stays readable straight
through a `QProgressDialog` (which pumps `processEvents`) and through a nested
dialog, and only disappears once we're back in the main event loop, well before the
analysis starts. The same holds for the splash screen the other way round: its
`deleteLater()` runs before `app.exec()` (loop level 0) and gets cleaned up as soon
as the loop starts — also measured separately.

**2. PySide6 6.11.1 → 6.11.2** in `.venv-yolo`. A guess: the release notes don't
mention this bug. Smoke-tested (app starts cleanly offscreen), but the exe hasn't
been rebuilt **yet**.

**What this doesn't prove yet:** that the crash is gone. Reproduction costs two
minutes per attempt (cut → batch, with Win+Shift+S in between) and hasn't been run
since this change. Do that in the venv (`start_gui.bat`), and only once it survives
a few rounds is rebuilding the exe worth it. If it keeps crashing, option 3 is next:
report it to Qt (QTBUG-81359/QTBUG-42985 family) — the report is essentially ready
alongside this document.

### Crash number five (25 Aug 2026, 23:45) — and this one had no Win+Shift+S

**Most important caveat up front: this was the OLD exe.** `Report.wer` gives
`TargetAppVer=2026//08//25:19:33:50` and `Qt6Gui.dll 6.11.1.0`, while the fix above
went into the source after 19:33 and `build.bat` hadn't been run. So not a single
line of the cleanup ran during this crash; this crash says nothing about whether it
helps.

**Recomputed the offsets themselves** (a custom PE export-table reader on
`Qt6Gui.dll` from the bundle, 10,590 exports), because "nearest export" is a
heuristic I didn't want to inherit:

| time | offset | function | delta |
|---|---|---|---|
| 20:46:08 | `0xe3880` | `QScreen::geometry()` | **+0** |
| 20:46:34 | `0xe3880` | `QScreen::geometry()` | **+0** |
| 21:25:07 | `0xe4a79` | `QScreen::virtualSiblings()` | +0x29 |
| 22:21:50 | `0xe4a79` | `QScreen::virtualSiblings()` | +0x29 |
| 23:45:26 | `0xe4a7c` | `QScreen::virtualSiblings()` | +0x2c |

`virtualSiblings()` runs from `0xe4a50` to `0xe4e60`, so those three offsets fall
comfortably inside it. And a crash at **+0** of a member function means the `this`
pointer itself is broken — not an empty `d` pointer but a **discarded `QScreen`
object**. Use-after-free, hard evidence.

**Why no screenshot was needed: this machine has two displays.** The laptop panel
(`\\.\DISPLAY1`, 1280×800) and an **EIZO EV2480** (1920×1080, positioned above the
laptop screen: x=319, y=−1080). Both at 100% DPI, so no mixed scaling. The EIZO
hangs off **USB-C**: `Microsoft-Windows-DeviceSetupManager/Admin` shows a `USB
Billboard Device` at 20:57 (exactly what a DisplayPort-alt-mode connection
announces), a VIA USB hub (`VID_2109&PID_2817`) and two EIZO devices (`VID_056D` =
EIZO), followed at 20:58:14 by the `Generic Monitor (EV2480)` container.

That flips the whole "trigger" question around from what was assumed: **Win+Shift+S
was never the cause, just one way to make the screen list get rebuilt.** A USB-C
monitor produces more of those events on its own: the DP link can retrain, the
monitor can drop in and out of power saving, the hub can briefly drop out. None of
that logs anything to the event log — checked, between 23:30 and 23:47 `System` has
**nothing**.

**One more detail worth noting:** the AMD driver is `31.0.22048.7002` from
**25 Mar 2024**, on a 2026 Windows build. A driver over two years old, driving two
displays one of which is over USB-C, while DirectML is doing math on that same iGPU
at the same time. A driver update isn't a proven fix but is the cheapest remaining
external variable.

**What was done about this: making it visible.** `_screens_to_log()` in skate_gui.py
hooks `screenAdded`/`screenRemoved`/`primaryScreenChanged` and, per screen,
`geometryChanged`/`availableGeometryChanged`/`refreshRateChanged`/
`logicalDotsPerInchChanged`, and writes every change with a timestamp to the log.
It's wired up right after the `QApplication` is created, so even a change during the
heavy imports gets captured. So at the next crash there will be black-and-white
evidence of whether a screen appeared, disappeared, or changed size just before it —
exactly the line that's currently missing. Pure measurement; it fixes nothing.
Tested against the real setup:
`[screen 23:52:32] at startup: \\.\DISPLAY1 1280x800 at (0,0) @60Hz | EV2480 1920x1080 at (319,-1080) @60Hz`

**This does make the cleanup more plausible.** What's dangling is a `QScreen*` that
someone is still holding on to. Every hidden top-level window is such a holder, and a
cut→batch left sixteen of them behind. With two displays, one of them over USB-C,
the odds of the screen list getting rebuilt during a multi-minute analysis are much
higher than with a single fixed panel — which also explains why the minimal
reproductions from 22:25-22:50 did nothing: they never opened any pickers and so
never left anything behind to dangle.

### The test of 25 Aug 23:57 — no crash, but the log stayed empty (resolved)

The first cut→batch run with the cleanup in place **just went through** (batch
finished, analysis saved). Except that session's log had nothing in it: no
`[screen ...]` line, no ultralytics output, just the session header and the familiar
handled `0x8001010d`.

The cause, and it's a gap that would have cost every future diagnosis: that session
ran under **`pythonw.exe`**. There, `sys.stderr` is simply `None`, and `start_log()`
only redirected when the app was frozen or `SKATEANALYSIS_LOG` was set — neither was
true. So every `sys.stderr.write` raised an `AttributeError`, which the diagnostic
code quietly swallows. Confusingly, a session header *did* appear: `start_crashlog()`
writes that itself even when there's no redirection, so the log reads as "the app is
logging" while nothing is actually coming in.

`start_log()` now also redirects when `sys.stderr`/`sys.stdout` is `None`. Tested by
setting `sys.stderr` to `None` and running the route: redirection kicks in, and both
a `[screen ...]` line and a plain `print` end up in the file. NB: `start_gui.bat`
uses `python.exe` and never had this problem — it happens via a shortcut or launcher
that invokes `pythonw.exe`.

1. ~~**Clean up pickers.**~~ ✅ *25 Aug 2026* — see above; not just the four pickers
   named but all eleven `exec()` sites, plus the cut progress dialog and the splash
   screen.
2. **Finer cut progress.** In `_clip_to_temp` (`_report_progress`), report in
   per-mille instead of whole percent, or `setValue` on a frame basis with a time
   threshold.
3. ~~**Crash log.**~~ ✅ *25 Aug 2026* — lives in `skate_environment.start_crashlog()`
   (not in `main()` but next to `start_log()`, so before all the heavy imports: a
   crash while loading torch or Qt counts too). `faulthandler` on its own file
   descriptor into the log file, plus `sys.excepthook` and `threading.excepthook`.
   Every session ends with `=== cleanly closed … ===`, so its absence points to the
   crash — needed, because Qt consistently produces a handled `0x8001010d` on
   startup that faulthandler reports anyway. Tested in the self-test with a real
   access violation. **The next crash will therefore leave a stack trace behind;
   reproduce it in the venv.**
4. ~~**Page file.**~~ Moot: it's now set to automatically managed (commit limit
   29.7 GB at 15.2 GB RAM).
