# GPU.md — running the analysis on the graphics card

Status: 13 August 2026. This file describes **what's in the code** and **what needs to
be installed per laptop** — for an NVIDIA machine (CUDA) and for a machine with an
integrated GPU (DirectML).

Background: the analysis took ~2 s/frame and turned out to be running entirely on the
CPU. `.venv-yolo` had `torch 2.13.0+cpu` — the CPU-only build, which in principle can
never talk to CUDA, no matter how good the graphics card is. On the laptop with an
**RTX 3050** this was fixed with CUDA (~10× faster); on the laptop with an **AMD
Radeon integrated GPU** it runs via DirectML (~2.2× faster, chapter 4).

---

## 1. What's in the code (the same on every machine)

Everything lives in [`skate_yolo.py`](skate_yolo.py) — the MediaPipe backend is
untouched and runs unchanged on the CPU. **There's nothing to configure**: the code
decides for itself, per pass.

| Piece | What it does |
|---|---|
| `yolo_device()` | `'cuda'` if torch sees a usable GPU, otherwise `'cpu'`. For the detection pass (ultralytics). |
| `yolo_dml()` | True if the detection pass can run via **DirectML** — the route without an NVIDIA card. CUDA takes priority. |
| `rtmpose_device()` | `'cuda'` / `'dml'` / `'cpu'` for the refinement pass, based on what ONNXRuntime has as providers. |
| `_load_yolo(...)` | Loads the `.pt` model (CPU/CUDA) or, on the DirectML route, an ONNX export of the same weights. Exports that once per model (~15 s) to `<model>-dml.onnx`. |
| `_dml_sessions()` | Context manager that puts ONNXRuntime sessions created within the block on DirectML. Needed because ultralytics only knows three providers (CUDA, CoreML, CPU) and has no switch for DirectML. |
| `_DmlYolo` | Shell around the ONNX model: builds the session inside the patch (ultralytics only does that on the first inference) and falls back to the `.pt` model on the CPU if DirectML drops out partway. |
| `_infer(call, ...)` | Runs an ultralytics call on the chosen device and, on a **CUDA OOM**, permanently moves the rest of the run to the CPU instead of letting the analysis fail. |
| `_make_rtmpose(...)` | Falls back to the CPU if the GPU session fails to build. Needed because `get_available_providers()` says CUDA/DirectML is *compiled in* — not that the DLLs actually load. |
| `SKATEANALYSIS_CPU=1` | Forces both passes to the CPU. For A/B measurements without tearing down the environment. |

**Why the two passes are determined separately:** they run on different engines. The
detection pass goes through torch/CUDA (or through ONNX/DirectML), the RTMPose
refinement through ONNXRuntime — which gets its GPU support from a different package.
So a machine can perfectly well have one but not the other, and each pass needs to be
able to fall back independently of the other.

**`YOLO_AUTOINSTALL=false`** sits at the top of `skate_yolo.py`, before the ultralytics
import. That's because ultralytics's ONNX backend does `check_requirements("onnxruntime")`
and installs that package unasked — right over `onnxruntime-directml`, silently
erasing the GPU route. That actually happened once while building this route (the
measurement suddenly became 2× slower and the log line reported a different
ONNXRuntime version number).

---

## 2. Installation is **per laptop** — this doesn't travel via git

`.venv-yolo/` is in `.gitignore`. So what you pull from GitHub is only the code; the
CUDA packages need to be installed separately on every machine. On a machine with no
NVIDIA GPU, skip this whole chapter — the code falls back to the CPU on its own.

**On an NVIDIA machine** (done on the RTX 3050 laptop):

```bash
# 1. Which CUDA version can the driver handle? Shown top-right in the output.
nvidia-smi

# 2. Replace the CPU build of torch with the CUDA build. Keep the torch VERSION the
#    same (here 2.13.0), only the +cuXXX part changes -- that way ultralytics can't
#    trip over it. Pick a cuXXX index at or below what nvidia-smi reported.
.venv-yolo\Scripts\python.exe -m pip install "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" --index-url https://download.pytorch.org/whl/cu130

# 3. ONNXRuntime separately, for the RTMPose refinement. This really is a separate
#    step: a CUDA torch says nothing about what ONNXRuntime can do.
.venv-yolo\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv-yolo\Scripts\python.exe -m pip install onnxruntime-gpu
```

**On a machine with no NVIDIA GPU** (done on the laptop with AMD Radeon Graphics) —
here everything runs via DirectML, so torch is left untouched:

```bash
# 1. What's in there? CUDA is only useful with an NVIDIA card.
powershell -c "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion"

# 2. Replace ONNXRuntime with the DirectML build. Remove the old one first: both
#    packages provide the same `onnxruntime` module and can't coexist.
.venv-yolo\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv-yolo\Scripts\python.exe -m pip install onnxruntime-directml

# 3. Tooling for the one-time ONNX export of the YOLO model.
.venv-yolo\Scripts\python.exe -m pip install onnx onnxslim
```

On the first analysis after that, the code exports `yolo26x-pose-dml.onnx` itself
(~15 s, 220 MB, stays next to the `.pt` file; is in `.gitignore`).

**Checking what you're running on:**

```bash
.venv-yolo\Scripts\python.exe -c "import skate_yolo as s; print('yolo:', s.yolo_device(), '| dml:', s.yolo_dml(), '| rtmpose:', s.rtmpose_device())"
```

On an NVIDIA machine it should print `cuda` twice; if it says `cpu` while there is an
NVIDIA card, check `torch.__version__` first — if it ends in `+cpu`, step 2 didn't
succeed. On a machine with an integrated GPU it should say `yolo: cpu | dml: True |
rtmpose: dml`: torch stays on the CPU there (that's correct), the compute goes
through ONNX.

---

## 3. Reference measurement (RTX 3050 Laptop, 4 GB)

"Schaats frontaal.MOV", 103 frames, with a target click, `corner=False`, same machine
on CPU vs. GPU:

| | CPU | GPU |
|---|---|---|
| Analysis time | 194.0 s | **19.5 s** |
| Per frame | 1.88 s | 0.19 s |
| Coverage | 100/103 | 100/103 |
| Pushes | 8 | 8 |

**9.9× faster, and the measurement is identical**: the same 8 pushes, the same legs,
the same event boundaries, the same angles down to 0.00°. Landmarks differ by
sub-pixel amounts (median 0.048 px, p95 0.23 px); the only outliers up to ~3.6 px sit
on the **wrists** and shoulder — points that feed into no measurement at all. GPU
acceleration here is therefore purely a time saving, not a silent measurement change,
and old analyses stay comparable with new ones.

---

## 4. The laptop with an integrated GPU: DirectML (done, 13 August 2026)

**CUDA is NVIDIA-only.** On an Intel Iris Xe/UHD or AMD Radeon Graphics,
`torch.cuda.is_available()` simply returns `False`. Torch never talks to an iGPU at
all, so the route doesn't go through `device=` but through an **exported ONNX model**
run by ONNXRuntime with the **DirectML** provider (brand-agnostic: any DirectX-12
GPU, so AMD too). OpenVINO — the alternative — was ruled out: its GPU plugin is
Intel-only.

### Measurement (AMD Radeon Graphics, integrated; same clip and protocol as chapter 3)

| | CPU | DirectML |
|---|---|---|
| Analysis time | 220.6 s | **98.4 s** |
| Per frame | 2.14 s | 0.96 s |
| Coverage | 100/103 | 100/103 |
| Pushes | 8 | 8 |

**2.2× faster with an identical measurement**: the same eight pushes, the same legs
and event boundaries, the same angles, and 0.0 px median difference on knees and
ankles (p95 0.1 px). The RTMPose refinement on DirectML is even **bit-identical** to
the same pass on the CPU.

Measured beforehand on the bare model (1280×1280, ONNXRuntime without the rest of the
pipeline): detection pass 3.26 s → 0.60 s per frame, RTMPose 0.098 s → 0.035 s. That
the whole analysis is "only" 2.2× faster is because the CPU reference runs through
torch (2.14 s/frame) rather than ONNXRuntime-CPU, and because reading/decoding and the
color machinery stay on the CPU.

### The pitfall that almost broke this silently: **export with `dynamic=True`**

Ultralytics letterboxes a `.pt` model to a **rectangle** (only up to a multiple of the
stride), but an ONNX model with a **fixed** input shape gets the image pasted into a
**square**, with a wide gray border added. The network then sees a different picture.
Measured on this clip with a static export:

- coverage **89/103** instead of 100/103,
- event boundaries shifted, two push angles **18° different** (58.9° → 40.4° and
  53.9° → 64.7°),
- and that while the raw per-frame detections themselves were nearly identical — the
  difference traveled through the gap-filling and the color gate in the refinement
  pass.

With `dynamic=True`, ultralytics falls back to exactly the same rectangular letterbox
as the `.pt` model, and the measurement matches again. It costs **no speed**: every
frame of one video has the same shape, so DirectML builds its graph once (98.4 s
dynamic against 124.4 s static — dynamic was even faster).

That this was about the export and not DirectML's arithmetic was established with a
bisect: the same static ONNX model on the **CPU** provider also gave 89/103. That's
the tool chapter 5 describes, and it's exactly what it's for.

### What's still on the table

An iGPU shares its memory bandwidth with the CPU, so the RTX 3050's 10× isn't
available here. If you want this genuinely faster, a smaller model
(`yolo26m-pose.pt`, see `DEFAULT_YOLO_MODEL`) is probably a bigger win than further
GPU tuning — but that *is* a measurement change and must go through the protocol
below.

---

## 5. Measurement protocol — how to prove a speedup isn't a measurement change

The rule for this project: **speed may change, the result may not.** A speedup that
shifts the angles by half a degree isn't a speedup, it's a silent regression.

1. Run the same video twice: once with the new route, once with `SKATEANALYSIS_CPU=1`
   as the reference. If they differ, **bisect**: run the new *model* on the old
   *provider* (or vice versa). That's how the DirectML route's discrepancy turned out
   to be in the ONNX export and not the GPU (chapter 4).
2. Use **the same settings**, including the `doel_punt` of the original analysis
   (found in `analysis.settings_json` in the library DB).
3. Compare with the existing tooling: `python skate_eval.py compare cpu.npz new.npz`.
   **Always read the stance-leg number** — see the `skate_eval.py` section in
   CLAUDE.md for why the mixed number is misleading.
4. Passed = same coverage, same number of pushes with the same legs and event
   boundaries, and angles matching to within ~0.0°.

**Two pitfalls that genuinely tripped this measurement up while setting it up** — both
cost a four-minute run before you realize your own test script was wrong:

- `segment_pushes(resultaten, min_lengte=3)` — the second parameter is **`min_lengte`
  (minimum length), not fps**. Pass it `info.fps` (30) by accident and every single
  event gets filtered out, giving "0 pushes" while the analysis is perfectly fine.
- **Without `doel_punt`**, automatic target selection picks the biggest mover, and on
  this clip that isn't the same skater as in the saved analysis. You'd then be
  comparing two different measurements against each other.

---

## 6. Can CPU and GPU compute at the same time?

Technically yes, practically it buys almost nothing — and that's not a matter of
taste but of arithmetic. The GPU does a frame in ~0.15 s, the CPU in ~1.75 s. Split
the work optimally between the two and the slow side can handle at most **~9%** of
the frames; more than that and it becomes the bottleneck while the fast one sits
waiting. So the whole gain is that 9% (19.5 s → ~17.8 s). **The faster your GPU, the
less a CPU can still contribute.**

**Where the time actually goes** (measured on the RTX 3050, 103 frames,
`corner=False`, by weighing the time inside the model calls against the total
analysis time):

| | time | share |
|---|---|---|
| YOLO detection | 15.50 s | 78% |
| RTMPose refinement | 2.44 s | 12% |
| Everything else (decoding, color, tracking, smoothing, derivatives) | 1.77 s | **9%** |

That last number is the important one in the table: the CPU is **not** sitting idle
next to a waiting GPU. Had it been 40%, the feed would have been lagging and there
would have been something to gain — but by fixing the feed, not by cramming more
inference into it. (Caveat: the 150 ms per YOLO call also includes CPU preparation
inside ultralytics, so the pure GPU share is a bit smaller than 78%. That changes
nothing about the conclusion.)

**Two obstacles specific to this program:**

- **Tracking is inherently sequential.** The detection pass runs ByteTrack with
  `persist=True`: every frame builds on the previous one, which is how each skater
  keeps a continuous ID. Splitting frames across two workers breaks that chain.
  You'd have to cut the video into blocks and stitch the tracks back together
  afterward — hitting exactly the robustness this program relies on for crossing
  skaters, for a 9% gain.
- **The CPU is already busy** decoding, feeding, and processing. Load it up with its
  own inference too and the feed to the GPU slows down, and you can end up net
  slower.

**On an integrated GPU the idea is even less promising**, for a reason that isn't
immediately obvious: an iGPU has no memory of its own but shares system memory with
the processor. Having both compute at once means they fight over the same memory
bandwidth.

**What actually helps is less work, not more devices:** the corner detection already
in place (measured 45% time savings on "Kim tempo") and possibly a lighter model
(`yolo26m-pose`) — but the latter *is* a measurement change and must go through the
protocol in chapter 5.

---

## 7. What not to do

- **No `half=True` / fp16.** It's tempting (free speed on a GPU), but it changes the
  keypoints in the last decimal places and therefore the measured angles. That's a
  measurement change and shouldn't sneak in as a side effect of a speed measure.
- **Don't bump the torch version at the same time** as switching to a CUDA build.
  Change only the `+cuXXX` part, keeping ultralytics compatibility out of the blast
  radius.
- **Always install rtmlib with `--no-deps`** — it declares `opencv-contrib-python`
  and would otherwise overwrite the existing cv2 install (also noted in CLAUDE.md).
- **Don't build a device choice into the GUI.** The user (a trainer) can't know
  what's the right answer here; the code should determine that itself, as it does
  now.
- **Never run an ultralytics command without `YOLO_AUTOINSTALL=false`** on a
  DirectML machine — not even a quick one from the command line. It will install the
  CPU build of ONNXRuntime right over `onnxruntime-directml` and the GPU quietly
  disappears. To recover: `pip uninstall -y onnxruntime` +
  `pip install --force-reinstall onnxruntime-directml`.
- **No static ONNX export** (see chapter 4): that changes the letterbox and with it
  the measurement.
- **Don't shrink `_DmlYolo`'s warm-up frame** to save time. The model is dynamic, so
  any shape is allowed in principle — but the end2end head does a TopK over `max_det`
  (300) positions, and a small image doesn't have that many: at 64 px the DirectML
  session breaks immediately and the whole analysis falls back to the CPU anyway
  (tried: 99 s → 224 s, with the correct result).

---

## 8. Splitting work across the two laptops

The library lives in Google Drive, so analyzing and reviewing don't need to happen on
the same machine. The RTX 3050 laptop stays by far the fastest for **analyses** at
~0.19 s/frame; the laptop with the integrated GPU takes ~0.96 s/frame with DirectML
(was ~2.14 s) and is therefore fine for **watching and reviewing** — playback,
correcting skeletons, comparing, cutting — and from now on also usable for the
occasional standalone analysis.
