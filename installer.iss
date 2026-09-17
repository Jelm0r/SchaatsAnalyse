; ===========================================================================
;  installer.iss — Inno Setup recipe for SkateAnalysis (EXE.md step 4).
;
;  Packages the PyInstaller output from `dist\SkateAnalysis\` (step 3) plus the three
;  models into one download: `dist\SkateAnalysis-setup.exe`.
;
;  Building goes through `build.bat`, which in order creates the version stamp, runs
;  PyInstaller, puts the models next to the exe and compiles this script. Standalone:
;
;      "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /DVersion=2026-08-25.a4be1f2b installer.iss
;
;  Five choices that deserve an explanation:
;
;  1. **PrivilegesRequired=lowest** — installs to {localappdata}\Programs\SkateAnalysis.
;     Two reasons: no admin rights needed (the trainers are on work laptops), and the
;     install folder is writable, so the fallback in `_onnx_path` (step 1.4) isn't
;     immediately needed. If that folder ever turns out not to be writable after all,
;     the app writes to `data_dir()` instead.
;  2. **Everything from `dist\SkateAnalysis` in one line**, so including the models
;     `build.bat` already put there. No second copy route pulling from the rtmlib
;     cache: that could make the installer contain something different from the
;     folder you just tested. The `#if` checks below refuse to compile if a model is
;     missing — otherwise the app would silently start downloading 178 MB on the
;     first analysis (see step 1.5).
;  3. **Nothing in `%LOCALAPPDATA%\SkateAnalysis` or in the library is touched**,
;     not even on uninstall. That's where the log file, `config.json` (with the path
;     to the Drive folder) and possibly a self-made ONNX export live; the training
;     data lives in Drive. Removing the app must never touch training data, and a
;     reinstall should just find the library again.
;  4. **`SetupLogging=yes`** — same idea as the log file in step 2: if the install
;     goes wrong for a colleague, there's a `Setup Log*.txt` in `%TEMP%` to ask for.
;  5. **No version number in the file properties** (`VersionInfoVersion`): that field
;     requires `x.y.z.w` and our stamp is a date + commit hash. It does show up as
;     AppVersion in "Apps and features", which is where you'd look for it.
;
;  The exe isn't signed: SmartScreen shows "Windows protected your PC" on first launch
;  (More info → Run anyway). That belongs in INSTALL.md (step 6).
; ===========================================================================

#define Name "SkateAnalysis"
#define ExeName "SkateAnalysis.exe"
#define Out AddBackslash(SourcePath) + "dist\SkateAnalysis"

; The version stamp comes from `make_version.py --show` via build.bat (/DVersion=...).
; Compiling standalone works too; then "Apps and features" shows "unknown" — the
; analyses themselves keep their own stamp from `_version.py`, that's a separate path.
#ifndef Version
  #define Version "unknown"
#endif

; ── Refuse to compile if the build isn't complete ──────────────────────────
#if !FileExists(AddBackslash(Out) + ExeName)
  #error dist\SkateAnalysis\SkateAnalysis.exe is missing — run build.bat first (step 3).
#endif
#if !FileExists(AddBackslash(Out) + "yolo26x-pose.pt")
  #error yolo26x-pose.pt is missing from dist\SkateAnalysis — without this model the app downloads 126 MB on first use.
#endif
#if !FileExists(AddBackslash(Out) + "yolo26x-pose-dml.onnx")
  #error yolo26x-pose-dml.onnx is missing from dist\SkateAnalysis — without this export the detection pass falls back to the CPU (2.2x slower).
#endif
; This name is RTMPOSE_LOCAL in skate_yolo.py; if the file has a different name, the
; app silently falls back to the URL and downloads 178 MB on first use.
#if !FileExists(AddBackslash(Out) + "rtmpose-x-halpe26-384x288.onnx")
  #error rtmpose-x-halpe26-384x288.onnx is missing from dist\SkateAnalysis — without this model the app downloads 178 MB on first use.
#endif

[Setup]
; This GUID is the app's identity: it determines whether a second install is an
; upgrade or a second copy. Never change it.
AppId={{81C5E169-E35C-4FA6-93F1-58D66B23270E}
AppName={#Name}
AppVersion={#Version}
AppVerName={#Name} {#Version}
AppPublisher={#Name}
DefaultDirName={autopf}\{#Name}
DefaultGroupName={#Name}
; Nobody wants to pick a Start Menu folder.
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; The bundle is 64-bit (torch, onnxruntime, Qt); PySide6 requires Windows 10 or newer.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#SourcePath}dist
OutputBaseFilename={#Name}-setup
SetupIconFile={#SourcePath}skateanalysis.ico
UninstallDisplayIcon={app}\{#ExeName}
UninstallDisplayName={#Name}
; Half of the 1.3 GB is models (.pt is already a zip, .onnx are raw weights): expect
; little further gain there and a compile pass of tens of minutes.
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes

[Languages]
; The wizard stays in Dutch: the actual trainers using this installer are
; Dutch-speaking (see INSTALL.md), regardless of the source code's language.
Name: "nl"; MessagesFile: "compiler:Languages\Dutch.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; The whole PyInstaller output folder including `_internal` and the three models
; build.bat put next to it. `app_dir()` is, when frozen, the exe's own folder, so the
; app finds the models here with no configuration.
Source: "{#Out}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#Name}"; Filename: "{app}\{#ExeName}"
Name: "{autodesktop}\{#Name}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#StringChange(Name, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
