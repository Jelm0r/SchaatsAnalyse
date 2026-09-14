# SkateAnalysis as an installable .exe

*Plan, drawn up 24 August 2026. Steps 1 and 2 executed 24 August 2026, steps 3 and 4 on
25 August 2026, steps 5 and 6 on 25 August, and the clean-machine test on 26 August 2026.
What still needs to be done by hand is listed per step under "Still to do".*

## Context

The app currently only runs on a machine where two Python environments have been built
by hand. That's a barrier for the other trainers: they'd need to install Python 3.11,
create a venv, put torch/ultralytics/PySide6/rtmlib into it and — differently per GPU
brand — pick the right onnxruntime variant. That GPU install also doesn't travel via
git (`.venv-yolo/` is in `.gitignore`), so everyone has to redo that work, usually
wrong.

Goal: **download one file and the app works**, including GPU acceleration, without
anything needing to be installed.

The most valuable side effect: by baking in `onnxruntime-directml`, everyone gets the
measured 2.2× speedup with no install work — on Intel, AMD, and NVIDIA alike.

## Starting points

| choice | decision |
|---|---|
| Distribution form | **Installer exe** (Inno Setup). Download one .exe, unpack once, then the app starts in ~2.5 s as it does now. NOT onefile — that unpacks ~1.5 GB to `%TEMP%` on every start and undoes the startup work (splash screen + lazy backend import). |
| GPU | **DirectML only.** One build for everyone, 2.2× faster than CPU, falls back to CPU where there's no GPU. No separate NVIDIA/CUDA build (+2.5 GB and two builds to maintain). |
| Models | **Bundled.** Works immediately and offline, and the model file is pinned — a later-downloaded different model could silently shift the measurements. |
| MediaPipe backend | **Left out.** `IS_YOLO` is always true in this package, so MediaPipe would be dead weight. Do close off the fallback path though (see step 1.6). |

Measured size (24 Aug 2026): `.venv-yolo` = 1.9 GB, site-packages = 1,838 MB
(PySide6 634, torch 496, polars runtime 176, cv2 112, onnxruntime 71, sympy 72).
Models together 557 MB. Expectation after trimming: **~1.3-1.6 GB installed, ~700-900 MB
download.**

---

## Step 1 — Code fixes: six spots, two helpers ✅ *done 24 Aug 2026*

### What's there now

All six spots done, plus the fallback rule. The helpers are called `app_dir()`,
`data_dir()`, and `is_frozen()` and live at the top of [skate_analysis.py](skate_analysis.py);
`data_dir()` creates the folder (`makedirs(exist_ok=True)`, errors swallowed — the
actual write still fails on its own if that's a problem).

**In the repo environment everything still resolves to exactly the same files as
before**, so the measurement doesn't change. Re-measured on this machine:

```
app_dir     C:\Apps\SchaatsAnalyse          data_dir  C:\Users\<u>\AppData\Local\SkateAnalysis
yolo model  C:\Apps\SchaatsAnalyse\yolo26x-pose.pt        (exists -> no re-download)
onnx path   C:\Apps\SchaatsAnalyse\yolo26x-pose-dml.onnx  (hypothetical .pt -> data_dir)
rtmpose     https://download.openmmlab.com/...            (no local file -> URL)
app_version 2026-08-24 . 85ba31fc+                        (git route, unchanged)
```

`_load_yolo()` on the new absolute path picks up the existing DirectML export and
exports or downloads nothing again. `skate_db.py`'s, `skate_yolo.py`'s, and
`skate_perspective.py`'s self-tests run, `py_compile` in both venvs, and the GUI
starts. The frozen path was simulated with `sys.frozen` + a hand-written `_version.py`
→ `2026-08-24 . 4df9ab5a` (with a `+` for dirty): the same format as the git route.

Two things later steps need to carry forward from this one:

- **`_version.py` must define `COMMIT`, `DATE`, and `DIRTY`** (str, str, bool) — that's
  what `_version_from_bundle()` in [skate_db.py](skate_db.py) reads. An empty or
  missing `COMMIT` means "no stamp" and falls back to `_git()`.
- **The RTMPose file must be named `rtmpose-x-halpe26-384x288.onnx`** and sit next to
  the exe; that name is `RTMPOSE_LOCAL` in [skate_yolo.py](skate_yolo.py). If it's not
  there, the app silently pulls the URL and downloads 178 MB on first use.

The fallback rule ended up a bit stricter than described above: when frozen, `IS_YOLO`/
`BACKEND_NAME` stay set and `_load_backend()` produces a `_backend_broken` that throws
one clear `RuntimeError`, and `_warn_backend_fallback()` there shows a **blocking**
message ("can't analyze right now; viewing the library and recordings still works")
instead of the MediaPipe warning.

### The original plan

Six places assume there's a script folder and a git repo next to the code. All six
get resolved with two small helpers in **`skate_analysis.py`** — that's the lowest
shared module: `skate_gui.py`, `skate_yolo.py`, and `skate_db.py` all three import
from it, so one place is enough.

```python
def app_dir():
    """Folder with the bundled files (models). Frozen: next to the exe."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def data_dir():
    """Writable folder for what the app creates itself (%LOCALAPPDATA%\\SkateAnalysis)."""
```

`data_dir()` follows the existing pattern of `config_path()` in
[skate_db.py](skate_db.py), which already uses `%APPDATA%` with `expanduser("~")` as a
fallback. The library config itself stays where it was, under `%APPDATA%` — an
existing install therefore just finds its Drive library again.

The six changes:

1. **[skate_gui.py](skate_gui.py)** — `_MODEL_DIR = dirname(__file__)` → `app_dir()`.
2. **[skate_analysis.py](skate_analysis.py)** — CLI model path → `app_dir()`.
3. **[skate_yolo.py](skate_yolo.py)** — `DEFAULT_YOLO_MODEL = "yolo26x-pose.pt"` is a
   bare filename and therefore depends on the working directory. Started from a
   shortcut, ultralytics wouldn't find the model and would **re-download 126 MB** into
   a random folder. Made absolute via `app_dir()` at the point of use.
4. **`_onnx_pad` [skate_yolo.py](skate_yolo.py)** — writes the DirectML export next to
   the `.pt` file. We ship that export, so normally nothing gets written; but if the
   file's ever missing, the export shouldn't fail on a read-only install folder. Rule:
   if it exists next to the `.pt` → use it; otherwise `data_dir()`.
5. **`RTMPOSE_MODEL` [skate_yolo.py](skate_yolo.py)** — currently a URL. rtmlib's
   `BaseTool.__init__` does `if not os.path.exists(onnx_model): download_checkpoint(...)`,
   so **a local path works with no patch needed**. Pass the bundled `.onnx` as the path
   at the two call sites, with the URL as a fallback.
6. **`app_version()` [skate_db.py](skate_db.py)** — runs `git` in the script folder.
   In an exe there's no repo, so every colleague's analysis would land in the library
   **without a version stamp** — exactly what the Info dialog and the title tooltip
   rely on. Solution: the build script generates a `_version.py` with the commit,
   date, and dirty flag; `app_version()` reads that first when `sys.frozen` is true and
   otherwise falls back to `_git()` as now. **The `label` format stays exactly
   `"2026-08-24 . 4df9ab5a"`**, so old and new analyses stay comparable.

Plus one robustness rule: `_load_backend()` ([skate_gui.py](skate_gui.py)) falls back
to MediaPipe when the YOLO import fails. That's not in the package, so that fallback
would only crash on the first analysis in an exe. When `sys.frozen` is true, that
should be a clean message instead of a fallback — `BACKEND_ERROR` already exists and
is already shown by `_warn_backend_fallback()`.

---

## Step 2 — Redirecting output (otherwise a `print` crashes) ✅ *done 24 Aug 2026*

### What's there now

A new, **stdlib-only** module [skate_environment.py](skate_environment.py):
`is_frozen()`, `app_dir()`, and `data_dir()` (moved out of
[skate_analysis.py](skate_analysis.py), which re-exports them so every existing import
keeps working unchanged) plus `start_log()`. That move is the core of this step: the
redirection needs `data_dir()` at exactly the moment cv2/numpy must not be loaded yet.

[skate_gui.py](skate_gui.py) calls `start_log()` **before the Qt import, and therefore
before `_start_splash_screen()`**, and only when `__name__ == "__main__"` — the same
rule as the splash screen itself, so `import skate_gui` in a measurement script hijacks
nothing. Cost: **~1 ms** (the 8 ms `-X importtime` shows is almost entirely
`threading`, which was already there).

The log lives at `%LOCALAPPDATA%\SkateAnalysis\skateanalysis.log`: line-buffered (so a
crash does leave the last complete lines behind), rotating at 1 MB to `.log.1`, with a
header block per start (time, program, `app_dir`, `data_dir`, Python version, frozen
yes/no). `sys.stdout` and `sys.stderr` point at the same stream, so the ordering stays
correct. Logging must never be able to bring the app down: every write sits in a
try/except, and if the file can't be opened (read-only folder, full disk) output goes
to a silent stream — output lost, but no crash, which was the whole point here.

**Shown to actually solve the problem**, with a simulated exe (`sys.frozen` + stdout
and stderr set to `None`):

| | without redirection | with |
|---|---|---|
| bare `print()` | **silent** — Python swallows it, no error | in the log |
| `sys.stdout.write(...)` | `AttributeError: 'NoneType' object has no attribute 'write'` | in the log |
| tqdm bar (ultralytics download/export, rtmlib) | **`AttributeError`** | in the log |
| logging handler on `sys.stdout` | silent | in the log |
| traceback of an uncaught error | silent | in the log |

The assumption at the top of this step turned out to be slightly off: a bare `print()`
doesn't crash (CPython just drops it if `sys.stdout` is None), but tqdm and every
direct `.write` do — and everything that doesn't crash disappears without a trace,
which is exactly what you're stuck with when a colleague calls to say it doesn't work.

**Tested**: self-test `python skate_environment.py` (redirection, a second session
after it, rotation to `.log.1`, and the fallback if the log file can't be opened) in
both venvs; `skate_db`'s/`skate_yolo`'s/`skate_perspective`'s self-tests unchanged;
`py_compile` in both venvs; and the GUI actually started with `SKATEANALYSIS_LOG=1` —
the header block showed up in the log. Nothing changes about the measurement: in the
repo environment, without that env var, nothing gets redirected.

Two things later steps need to carry forward from this one:

- **`SKATEANALYSIS_LOG=1` also turns on redirection in the plain venv.** That way this
  route can be tested without building an exe first; without the env var, all output
  stays on the console.
- **The blocking backend message now names the log path** (only when logging is
  actually happening), because a log nobody can find is worthless. `INSTALL.md`
  (step 6) should name that same path: "send me
  `%LOCALAPPDATA%\SkateAnalysis\skateanalysis.log`".

### The original plan

A PyInstaller build with `--windowed` has no console: `sys.stdout` is then `None` and
a bare `print()` throws `AttributeError`. This codebase prints in several places —
among others the warning helper in [skate_yolo.py](skate_yolo.py) when there's no
`warning_callback`, and the one-time export notice in `_load_yolo`. Those could break
in the exe.

In the frozen entry point, before all other imports: point `sys.stdout` and
`sys.stderr` at a log file in `data_dir()` (`skateanalysis.log`, rotating by size).
That solves the problem and immediately gives you something to ask for when a
colleague reports it doesn't work.

Mind the order: this has to come before `_start_splash_screen()` in
[skate_gui.py](skate_gui.py), since that's already a module-level side effect.

---

## Step 3 — PyInstaller spec ✅ *done 25 August 2026*

### What's there now

Three new files, and **not a single line changed in the existing modules** — the
standalone scripts and both venvs run unchanged, the exe sits alongside them.
Everything the bundle needs was already given to the code in steps 1 and 2.

| file | what |
|---|---|
| `skate_analysis.spec` | the PyInstaller recipe: onedir, windowed, noupx, models left out |
| `make_version.py` | generates `_version.py` with the git stamp (commit/date/dirty), same flags as `app_version()` |
| `build.bat` | version stamp → PyInstaller → put the models next to the exe |

Building: **`build.bat`** in the repo folder (PyInstaller 6.22.2, installed in
`.venv-yolo`). Takes ~5 minutes and produces `dist\SkateAnalysis\SkateAnalysis.exe`.

**Measured (25 August 2026):** app without models **787 MB** (`_internal` 740 MB +
exe 47 MB), with the three models added 1.3 GB. Biggest chunks: torch 365, cv2 112,
PySide6 104, onnxruntime 64, numpy.libs 21, matplotlib 15, PIL 13, torchvision 11 MB.
That stays under the estimated 1.3-1.6 GB, thanks to two excludes, both re-measured:

- **PySide6: 634 → 104 MB.** The app uses four Qt modules (QtCore/QtGui/QtWidgets/
  QtCharts); the rest is explicitly in `excludes` because matplotlib likes to drag in
  Qt and tk backends.
- **polars gone: −180 MB.** The polars runtime (177 MB) comes in via ultralytics, but
  *every* polars import there sits in training, benchmark, plot, and dataframe-export
  paths — commented in ultralytics itself as *"scope for faster 'import ultralytics'"*
  — and this app only does inference. Re-measured with polars hard-blocked via
  `sys.meta_path`: `import skate_yolo`, `_load_yolo()` (the DirectML route), and then
  `model.track()` and `model.predict()` on a frame all three run without polars ever
  being loaded.

**What was shown to work** (twice, before and after the polars exclude):

- The exe **starts frozen**: the header block in the log reports `frozen=True` and
  `app_dir` = the dist folder, so step 1 resolves there to the bundled models. No
  traceback in the log.
- **The lazy backend import succeeds in the bundle** — the riskiest spot in this step.
  30 seconds after startup, `Qt6Charts.dll`, `torch_cpu.dll`, and
  `onnxruntime_pybind11_state.pyd` are in the process (`Get-Process ... .Modules`).
  That's immediate proof that `_backend_available()`'s `find_spec` query works under
  PyInstaller's FrozenImporter (otherwise `IS_YOLO` would silently fall to False and
  torch would never have loaded at all) and that `_warm_backend_up()` gets its
  `import skate_yolo` through.
- The **analysis-critical data files** are in there: `ultralytics/cfg/default.yaml`,
  `ultralytics/cfg/trackers/bytetrack.yaml`, and `onnxruntime/capi/DirectML.dll`.
  rtmlib has no hook but is pure Python and sits fully inside the PYZ.
- **`_version` is present as a PYMODULE in the bundle**, so the exe can attach a
  version stamp to every analysis (the behavior of `_version_from_bundle()` itself
  was already tested in a simulated way in step 1).

**What this doesn't prove yet:** that a real analysis in the exe produces the same
measurement, and that DirectML is actually on rather than silently falling back to
CPU. That can't be shown with a startup check — that's step 5, and it's not for
nothing the most important step.

**Five things to remember:**

- **`--noconfirm` wipes the whole output folder**, so `build.bat` re-copies the 557 MB
  of models on every build (~30 s disk-to-disk). That's the price of keeping them out
  of the bundle: they don't have to go through the PyInstaller mill, and a model can
  be swapped without rebuilding.
- **`torch.distributed` is deliberately NOT excluded**, even though the plan mentioned
  it: ultralytics's trainer imports it and that chain hangs off
  `from ultralytics import YOLO`. The saving would be at most **6 MB** (it's Python
  source; the 315 MB in `torch/lib` are the DLLs and those have to stay), so that risk
  isn't worth it.
- **matplotlib stays** (31 → 15 MB): `ultralytics.utils.plotting` imports it on load,
  but PyInstaller figured out on its own that only the **Agg** backend is used.
- **No icon yet**: the exe carries the default PyInstaller icon. Adding a `.ico` and
  filling in `icon=` in the spec is enough — cosmetic, so saved for step 4.
- The warning **`Library nvcuda.dll required via ctypes not found`** is expected: this
  is the DirectML build, there's no CUDA in it. Same for
  `Hidden import "tzdata" not found` (a polars leftover).

### The original plan

Build **inside** `.venv-yolo` (Python 3.11), with `pip install pyinstaller`. A
`skate_analysis.spec` in the repo (not a long command line), so the build is
reproducible and lives in git.

- **Mode:** `onedir`, `--windowed`, `--noupx` (UPX mangles the Qt and torch DLLs).
- **Entry:** `skate_gui.py`.
- **Models: NOT in the bundle.** Inno puts those next to the exe instead (step 4).
  Saves hundreds of MB of copying on every rebuild and lets a model be swapped without
  rebuilding. `app_dir()` finds them there.
- **Include:** `--collect-data ultralytics` (the yaml configs, including
  `bytetrack.yaml` and `default.cfg`, aren't found automatically),
  `--collect-data rtmlib`.
- **Exclude** — this is where the savings are: `PySide6` ships every Qt module
  (634 MB) while the app uses four (`QtCore`, `QtGui`, `QtWidgets`, `QtCharts`,
  verified against the imports in [skate_gui.py](skate_gui.py)). Exclude explicitly:
  `QtWebEngine*`, `QtQuick*`, `QtQml`, `Qt3D*`, `QtMultimedia*`, `QtDesigner`,
  `QtTest`. Also `mediapipe`, `tkinter`, `pytest`, `IPython`, `torch.distributed`.
- **Likely hassle** (plan for it, don't wish it away): torch's DLL collection and the
  176 MB polars runtime that ultralytics brings along. Expect a few rounds of
  build → start → add missing module. `matplotlib` (31 MB) probably **can't** go:
  `ultralytics.utils.plotting` imports it on load.

A `build.bat` wrapped around it that in order generates `_version.py`, runs
PyInstaller, and calls Inno.

---

## Step 4 — Inno Setup script ✅ *done 25 August 2026*

### What's there now

`installer.iss` packages the folder from step 3 into **one 649 MB download**
(`dist\SkateAnalysis-setup.exe`, 1,313 MB unpacked). Compiling takes ~5 minutes and
hangs off `build.bat` as step 4/4; if Inno Setup is missing, that step is skipped with
a notice and the built folder is still runnable. Inno Setup 6.7, installed per-user
(`winget install JRSoftware.InnoSetup`).

Three small things around it:

- **`make_version.py --show`** prints the stamp in **ASCII** (`2026-08-24.a4be1f2a+`)
  without writing anything. `build.bat` passes that to the compiler as `/DVersion=`,
  so "Apps and features" shows which build is installed. The middle dot in `label`
  doesn't survive the console code page or a `for /f` loop in cmd.exe; the label in
  the library stays unchanged.
- **`skateanalysis.ico`** (the point saved from step 3): `icon=` in the spec,
  `SetupIconFile` in the installer, and with it the icon of every shortcut too. To
  replace it: overwrite the file and rebuild.
- **The installer refuses to compile if the build isn't complete.** Four
  `#if !FileExists` checks on the exe and the three models; without those checks, an
  installer could ship that looks fine and then silently downloads 178 MB on the
  first analysis (step 1.5).

**Smoke test** (silent install to a temp folder, start, remove again):

| | outcome |
|---|---|
| install (`/VERYSILENT`) | exit code 0, **43 s**, 3,313 files, 1,313 MB |
| models | all three present, with the names `app_dir()` expects |
| start menu shortcut | points at the installed exe (desktop checkbox skipped with `/TASKS=""`) |
| Apps and features | `SkateAnalysis` · version `2026-08-24.a4be1f2a+` |
| app starts | runs, window "SkateAnalysis"; log reports `frozen=True` and `app_dir` = the install folder, no traceback |
| uninstall | exit code 0, folder gone, both shortcuts gone |

**What this doesn't prove yet:** that an analysis from the install produces the same
measurement and that DirectML is actually on rather than silently falling back to
CPU — that's step 5. Nor whether it works on a clean machine (also step 5), or how
SmartScreen behaves (step 6).

**Four things to remember:**

- **Close a running `dist\SkateAnalysis\SkateAnalysis.exe` before rebuilding.**
  PyInstaller wipes the output folder with `--noconfirm`, trips on the locked exe with
  `PermissionError: [WinError 5]` — and has already half-cleaned the folder by then.
- **The install folder is `{localappdata}\Programs\SkateAnalysis`**
  (`PrivilegesRequired=lowest`): no admin rights needed, and writable, so the fallback
  from step 1.4 never comes into play.
- **Uninstalling doesn't touch `%LOCALAPPDATA%\SkateAnalysis`** (the log, and any
  self-made ONNX export), let alone the library in Drive. The log therefore survives
  a reinstall, and `config.json` in `%APPDATA%` keeps the path to the Drive folder.
- **Inno Setup 6.3 or newer is required**: `ArchitecturesAllowed=x64compatible`
  doesn't exist before that. `SetupLogging=yes` is on, so a failed install leaves a
  `Setup Log*.txt` in `%TEMP%` to ask for — same idea as the log file from step 2.

### The original plan

`installer.iss`, result `SkateAnalysis-setup.exe`.

- **`PrivilegesRequired=lowest`** → installs to `{localappdata}\Programs\SkateAnalysis`.
  Two reasons: no admin rights needed (matters if the trainers are on work laptops),
  and the folder is writable, so the fallback from step 1.4 works.
- `[Files]`: the PyInstaller output plus the three models — `yolo26x-pose.pt`,
  `yolo26x-pose-dml.onnx`, and the RTMPose model from
  `%USERPROFILE%\.cache\rtmlib\hub\checkpoints\`, **renamed to
  `rtmpose-x-halpe26-384x288.onnx`** (= `RTMPOSE_LOCAL`, see step 1). **Not**
  included: `pose_landmarker_*.task`, `yolo11*-pose.pt` — those belong to backends
  that aren't in this package.
- `Compression=lzma2/max`, `SolidCompression=yes`.
- Start menu shortcut + optional desktop; uninstaller.
- The library folder is **not** touched: it lives in Google Drive, and the path to it
  sits in `%APPDATA%\SkateAnalysis\config.json`. Removing the app must never touch
  training data.

---

## Step 5 — Verification: proving it's the same measurement ✅ *measurement part done 25 August 2026*

### What's there now

The core question of this step — **does the bundled app produce the same measurement
as the venv?** — has been answered on the standard test clip (`Schaats frontaal.MOV`,
103 frames, no target click, `corner=True`, smoothing 5 / threshold 0.015, i.e. exactly
the settings of the saved analysis). Three runs laid side by side: a fresh reference
in `.venv-yolo` on DirectML, a counter-check with `SKATEANALYSIS_CPU=1`, and the
analysis that landed in the library via `dist\SkateAnalysis\SkateAnalysis.exe`.

| | venv · DirectML | **exe · DirectML** | venv · CPU (counter-check) |
|---|---|---|---|
| Analysis time | 90.7 s | **~90 s** (stopwatched) | 202.5 s |
| Per frame | **0.88 s** | **~0.9 s** | **1.97 s** |
| Coverage | 103/103 | 103/103 | 103/103 |
| Pushes | 6 · RLRLRL | 6 · RLRLRL | 6 · RLRLRL |
| Event boundaries | 0-11, 12-30, 36-47, 48-66, 67-83, 84-102 | same | same |
| Angles | 42.2 / 42.5 / 42.9 / 40.0 / 45.6 / 50.0\* | same | same |

\* = `truncated`, doesn't count toward the statistic.

**This is 6 pushes, not the 8 from [GPU.md](GPU.md) chapters 3-4**: that reference run
used a target click *and* `corner=False`. What matters here isn't which of the two is
"better", but that all three runs above use the **same** settings — those of the saved
analysis, since otherwise you're comparing two different measurements (GPU.md
chapter 5, pitfall two).

**The exe analysis isn't "comparable" but identical**: all x/y landmarks of all 33
points across all 103 frames differ by **≤0.0001 px** from the fresh venv run (the
only difference worth mentioning is in the third column, `visibility`, at ≤6·10⁻⁵ —
float rounding, and that column feeds no measurement at all; `midline_dev` has the
same NaN pattern and differs by 0.0). Two exe runs measured: one that came out at
0.0000 px and the check run of 25 August 20:33 at 0.0001 px — so the difference
between two DirectML runs of each other is the same order of magnitude as between exe
and venv, i.e. a ten-thousandth of a pixel. `skate_eval.py compare` accordingly gives
0.0 px median and p95 on knees and ankles too. The CPU counter-check does deviate a
fraction as expected (median 0.0001 px, max 0.48 px on a heel, 0.28 px on the
measurement points) **without the measurement itself changing** — exactly the picture
from [GPU.md](GPU.md) chapter 4.

**Why that's not a coincidence, and why this step went so smoothly**: the bundle
contains literally the same code and the same computational core as the venv.
Re-measured:

- **All project modules bytecode-identical.** `_version`, `skate_analysis`,
  `skate_yolo`, `skate_db`, `skate_environment`, and `skate_perspective` from the PYZ,
  plus the entry script `skate_gui` from the CArchive, against a fresh `compile()` of
  the repo source: all seven identical at the instruction-byte level. `_version` is
  genuinely present too — without that module every analysis from the exe would show
  "unknown" in the Info dialog (step 3).
- **All binaries byte-identical.** 122 `.dll`/`.pyd` files in `_internal` that also
  exist in `.venv-yolo\Lib\site-packages`: **0 differences**, including
  `DirectML.dll`, `onnxruntime.dll`, `onnxruntime_pybind11_state.pyd`, the nine torch
  DLLs, `cv2.pyd`, and the fifteen numpy binaries.

**DirectML is on in the exe.** The log of an exe analysis shows
`Loading …\yolo26x-pose-dml.onnx for ONNX Runtime inference…`, and that path is only
taken in `_load_yolo` **when** `yolo_dml()` is true; if the route fails, an explicit
"GPU route (DirectML) could not be set up" message goes into that same log, and it's
not there. **Note the line right below it:**
`Using ONNX Runtime 1.24.4 with CPUExecutionProvider` is **not** proof of the
opposite — ultralytics logs its own request there, right before the
`_dml_sessions()` patch swaps the provider (ultralytics doesn't know DirectML, see
CLAUDE.md). Anyone still doubting this can just time it: 0.88 vs. 1.97 s/frame is not
a subtle difference — and that's exactly what the stopwatch on an exe analysis itself
confirmed (**~90 s** for 103 frames, not ~200 s).

**Startup time** (three starts in a row, warm machine, library on the Drive folder):
splash screen after **1.6 s**, main window after **2.7 s** — next to the 1.3 s / 2.5 s
of the standalone scripts, so the bundle costs ~0.2 s extra. The log checks out: one
header block per start, twelve starts, **zero tracebacks**.

**Self-test** `python skate_db.py` after the `app_version` change: *self-test OK*.

**Ahead of the clean-machine test (point 5):** the C runtime is in the bundle —
`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`,
`MSVCP140_ATOMIC_WAIT.dll`, `ucrtbase.dll`, and 40 `api-ms-win-*` stubs. So a PC
without the Visual C++ redistributable should just start fine.

### Functional walkthrough: how far it got (25 August 2026)

| action | outcome |
|---|---|
| open the library in Drive | ✅ recordings + analyses visible, status change ("in progress") saved |
| reopen an analysis | ✅ `IMG_9001.mov` loaded — so the npz, the video copy, and **QtCharts** all work frozen |
| view a recording (view window) | ✅ two points set and read back from `source_marking` |
| **cut a fragment** | ✅ `00005 13-16.mp4`: 84 frames, 1920×1080 @ 25 fps, 5.0 MB, readable — the `mp4v` `VideoWriter` from `opencv_videoio_ffmpeg500_64.dll` works in the bundle. This was the only path no other test touched |
| batch analysis on the fragment | ✅ run and saved three times (84, 252, and 62 frames) — ❌ but it **crashes** if a screen event occurs during the analysis, see below |
| Info dialog | ✅ |
| comparing two analyses | ✅ |
| the log afterward | ✅ one header block per start, no tracebacks |

**So the walkthrough passed**, with one exception that's not a bundling problem.

**The crash.** Faulting module `Qt6Gui.dll`, `0xc0000005` — and with the safety net
from TODO_CRASH point 3 (now built in) it was traceable: the **main thread goes down
in `app.exec()`** with a one-line Python stack, and the offsets point at
`QScreen::geometry()` and `QScreen::virtualSiblings()`. So it's Qt-internal code
reacting to a Windows message — a screen change — not our code; the analysis worker
was meanwhile just sitting in `cap.read()`. Reproducible by hitting Win+Shift+S during
the analysis; three runs with no screenshot went through fine. The same flow already
crashed the standalone scripts on 13 August. The full observation, the offset
analysis, and four failed attempts at a minimal reproduction are in
[TODO_CRASH.md](TODO_CRASH.md).

What bundling actually has to do with it: frozen there's no console, so a C++ crash
like that left **nothing** behind — Python never gets involved — and the answer had to
come from the Windows event log. That's why `faulthandler` now sits next to
`start_log()` in `skate_environment.py`, together with the Qt notices that would
otherwise go to the debugger on Windows. Every session ends with
`=== cleanly closed … ===`; if that line is missing, the app crashed there.

### The clean machine ✅ *26 August 2026*

Done on a second laptop with no Python, no VC++ runtime, and no GPU packages: pulled
the setup from Drive, installed it, started it, **analyzed one video**, and clicked
around — no problems. That covers the part of this step that can't in principle be
tested on your own machine: the bundle finds its models, the C runtime is in there,
and the app runs without a venv ever having been built.

**Exactly which build that was can no longer be determined.** The Info dialog for the
analysis there showed `43014c8b`, but that field names the version that made that
**analysis**, not the installed app — and the laptop is no longer available to check
*Settings → Apps*. So it was either `43014c8` or the later `6f96bf14`. That doesn't
change the conclusion: between those two commits not a single bundling file was
touched (`skate_analysis.spec`, `installer.iss`, `build.bat`, `make_version.py`, and
`skate_environment.py` are unchanged; the difference is in `skate_analysis`,
`skate_gui`, `skate_yolo`, and `skate_db`). What a clean machine needed to prove —
bundling, installing, paths, models, starting, measuring — is the same in either case.

**Lesson for next time:** note the version from *Settings → Apps → Installed apps*
while the machine still has it. That comes from the installer's `AppVersion` and says
which app is actually running; the version in an analysis's Info dialog is frozen at
the moment of analysis and doesn't answer that question.

### The original plan

This is the most important step. A different environment must not shift the
measurements — exactly the discipline from [GPU.md](GPU.md), reusable here one to one.

1. **Record a reference** — analyze `Schaats frontaal.MOV` (103 frames, with a target
   click) in the current `.venv-yolo`. Note coverage, number of pushes, legs, event
   boundaries, and angles. Save the npz.
2. **The same clip through the exe.** Expected: **the same coverage, the same 8
   pushes with the same legs and the same event boundaries**, and angles matching to
   within a tenth of a degree.
3. **Make it rigorous with the existing measurement tooling:**
   `python skate_eval.py compare old.npz new.npz` — expect 0.0 px median on knees and
   ankles, and **read the stance-leg number**, not the mixed average.
4. **Verify DirectML is really on in the exe** (not silently CPU): the analysis time
   should be close to 0.96 s/frame, not 2.14. Counter-check with
   `SKATEANALYSIS_CPU=1`.
5. **Clean machine** — a PC with no Python, no Visual C++ runtime, no manually
   installed GPU packages. Without this test you only know it works on your own
   machine.
6. **Functional walkthrough:** open the library in Drive, reopen an analysis, check
   the Info dialog (should show the version, not "unknown"), view a recording in the
   view window, cut a fragment, compare two analyses. Then check
   `%LOCALAPPDATA%\SkateAnalysis\skateanalysis.log`: there should be one header block
   per start, and otherwise no tracebacks (step 2).
7. **Measure startup time** — should be close to the current 2.5 s.
8. Run `python skate_db.py` (self-test) after the `app_version` change: it already
   checks the shape of `app_version()` and the automatic `app_version`/`app_commit` in
   the saved settings.

---

## Step 6 — SmartScreen + install instructions ✅ *done 25 August 2026*

### What's there now

[INSTALL.md](INSTALL.md) — the instructions for the trainers, written for
someone with no Python and no admin rights. The document's order is the order a
colleague actually runs into things: download → browser warning → SmartScreen →
install → first start → set Drive offline.

Four choices that matter for the content:

- **The SmartScreen step is already in the introduction**, not halfway through.
  Someone who sees that window unexpectedly stops — and calls to ask if they've
  caught a virus. So the explanation isn't "click Run anyway" but **why** the window
  shows up: a certificate costs €200-400 a year, unknown ≠ unsafe. Along with the one
  thing that can genuinely go wrong: *never turn off your antivirus*; for a
  quarantine, the route via Protection History → Allow on device is included, and
  otherwise call first.
- **Downloading from the Drive folder**, so it immediately says to make the file
  available offline first: starting a 650 MB setup from the streaming drive is slow
  and can fail partway through. Same pitfall as with the recordings, just at install
  time.
- **The library folder is marked as "the most important step"**, along with what goes
  wrong if you skip it (you then work in your Documents folder and nobody sees your
  analyses). That's the one setting where a wrong choice stays silent and only shows
  up weeks later.
- **The log file is written up as a first-aid tool, not a footnote**: how to open the
  folder (`Windows + R` → `%LOCALAPPDATA%\SkateAnalysis`), what to mention about it,
  and the criterion from step 2 — if `=== cleanly closed ... ===` is missing after
  your last session, the app crashed. That lets a trainer see for themselves whether
  there's anything to report.

It also covers things that aren't instructions but are the first questions anyway:
analysis time (~1 s/frame, GPU automatic via DirectML, CPU is ~2× slower), the corner
getting skipped, updating (install over the old one, library stays), removing
(doesn't touch Drive or the log), and the crash on Win+Shift+S during an analysis
from [TODO_CRASH.md](TODO_CRASH.md) — as a known quirk with the workaround, since
they'd otherwise find it themselves and it'd be a mystery.

**No certificate bought.** ~€200-400 a year for a handful of trainers doesn't outweigh
clicking through once, and an EV certificate (which SmartScreen does trust
immediately) is even more expensive. If the number of users ever grows, this is the
place to revisit.

### Still to do (can only be done by hand)

1. ~~Rerun `build.bat`~~ ✅ — `dist\SkateAnalysis-setup.exe` is now the build from
   26 Aug 19:28, `2026-08-26.6f96bf14`, with the crash-log safety net. Re-checked that
   the stamp matches on **both** sides: the `_version` module in the exe's PYZ archive
   says `COMMIT = '6f96bf14', DIRTY = False`, and the installer's `ProductVersion`
   says the same. Those two are fetched independently (steps 1 and 4 of `build.bat`,
   ~5 min apart), so a commit made *during* the build could make them diverge — read
   both rather than trusting just the file properties.
2. **Put the setup in the right place in Drive.** It currently sits in the root of
   `Mijn Drive`, while INSTALL.md points at
   `Mijn Drive\SkateAnalysis\app\SkateAnalysis-setup.exe`. That's not just a path
   difference: the folder shared with the trainers is `SkateAnalysis` — check whether
   they can even reach the file in the root of your Mijn Drive at all. The `app\`
   folder doesn't exist there yet; it doesn't get in the library's way
   (`sync_source_dir` only looks in `opnames\`, the conflict check only at
   `schaats*.db` in the root folder).
3. **Ship INSTALL.md alongside it** — in the same Drive folder as the setup,
   because a colleague seeing the SmartScreen window needs the explanation at that
   exact moment.
4. **Mind old installs now that the shared library is on schema v5** (since the
   interlacing commit `2246151`; the analysis from 26 Aug 17:26 in Drive was already
   made with it). An install from before that commit knows only v4 and refuses to
   open the library with `LibraryTooNew` — caught cleanly and without damage, but it
   is the signal that that machine needs the new setup. So stop handing out v4 builds.

### The original plan

The exe isn't signed, so Windows shows "Windows protected your PC" on first start
(click through via *More info → Run anyway*), and Defender flags PyInstaller builds
as suspicious fairly regularly. A certificate costs ~€200-400 a year and for a
handful of trainers is probably not worth it — but the install instructions **do**
need to mention this, or the first colleague will think there's a virus in it.

Action: a short `INSTALL.md` with the download, the SmartScreen step, the
instruction to pick the library folder on the shared Drive, and — if something does
go wrong — where the log file is:
`%LOCALAPPDATA%\SkateAnalysis\skateanalysis.log` (step 2).

---

## Files

| file | what |
|---|---|
| [skate_environment.py](skate_environment.py) | new — `is_frozen()`/`app_dir()`/`data_dir()` + the log file; stdlib-only so it can run before the splash screen |
| [skate_analysis.py](skate_analysis.py) | re-exports the three helpers from `skate_environment`; fixed CLI model path |
| [skate_gui.py](skate_gui.py) | `_MODEL_DIR` via `app_dir()`; `start_log()` before all imports; backend fallback |
| [skate_yolo.py](skate_yolo.py) | model path, `_onnx_pad`, RTMPose path |
| [skate_db.py](skate_db.py) | `app_version()` reads `_version.py` when frozen |
| `skate_analysis.spec` | new — PyInstaller recipe |
| `skateanalysis.ico` | new — icon of the exe, the installer, and the shortcuts; replaceable |
| `installer.iss` | new — Inno Setup: PyInstaller output + models → `dist\SkateAnalysis-setup.exe` |
| `make_version.py` | new — writes the git stamp into `_version.py`; `--show` prints it in ASCII for the installer |
| `build.bat` | new — generate version → PyInstaller → add models → Inno Setup |
| `_version.py` | new, **generated**, in `.gitignore` |
| [INSTALL.md](INSTALL.md) | new — instructions for the trainers: downloading from Drive, the SmartScreen step, picking the library folder, setting Drive offline, where the log file is |
| `.gitignore` | `build/`, `dist/`, `_version.py`, `*-setup.exe` |

## Time estimate

- **Steps 1 + 2 (code fixes)** — ~1 hour. These do no harm even without an exe; the
  model path from 1.3 is already a latent problem even now.
- **Step 3 (getting the spec working)** — half a day to a full day, almost entirely
  hidden-import puzzling with torch and ultralytics. This is the unpredictable part.
- **Steps 4 + 6 (installer + instructions)** — ~2 hours.
- **Step 5 (verification incl. clean machine)** — ~half a day.

Order of operations: finish steps 1 and 2 first and just test in the current venv.
Then build a bare-bones spec and see if it even starts — you'll know within an hour,
and that's the moment that tells you whether this is an afternoon or two days.
