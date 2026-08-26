# -*- mode: python ; coding: utf-8 -*-
"""
schaatsanalyse.spec — PyInstaller-recept voor de gebundelde app (EXE.md stap 3).

Bouwen (in .venv-yolo, Python 3.11):

    .venv-yolo\Scripts\python.exe -m PyInstaller --noconfirm --clean schaatsanalyse.spec

of gewoon `bouw.bat`, dat er de versiestempel, het kopieren van de modellen en het
bouwen van de installer (stap 4) omheen zet.
Resultaat: `dist\SchaatsAnalyse\SchaatsAnalyse.exe`.

De losse scripts blijven onveranderd werken; dit is er puur naast gezet. Alles wat de
bundel nodig heeft, heeft de code al gekregen in stap 1 en 2 (`app_dir()`/`data_dir()` en
het logboek in schaats_omgeving.py) — hier wordt geen enkele module gepatcht.

Vier keuzes die verklaring verdienen:

1. **onedir, niet onefile.** Onefile pakt bij elke start ~1,5 GB uit naar %TEMP% en maakt
   daarmee juist het opstartwerk (opstartscherm + luie backend-import) ongedaan. Inno
   Setup (stap 4) verpakt deze map alsnog tot een enkele download.
2. **windowed, geen console.** Vandaar `start_logboek()` in schaats_gui.py: zonder console
   is `sys.stdout` None en loopt de tqdm-balk van ultralytics/rtmlib stuk (stap 2).
3. **noupx.** UPX sloopt Qt- en torch-DLL's.
4. **Modellen zitten er NIET in.** `yolo26x-pose.pt`, `yolo26x-pose-dml.onnx` en het
   RTMPose-model komen naast de exe te staan (bouw.bat kopieert ze voor het testen, Inno
   installeert ze in stap 4). `app_dir()` vindt ze daar. Scheelt 557 MB kopieerwerk bij
   elke herbouw en maakt een model vervangen mogelijk zonder opnieuw te bouwen.
"""

import os

from PyInstaller.utils.hooks import collect_data_files

HIER = os.path.abspath(SPECPATH)

# ── Meenemen ────────────────────────────────────────────────────────────────────
# ultralytics-data (bytetrack.yaml, default.cfg, ...) en torch/onnxruntime/cv2-DLL's
# regelen de meegeleverde hooks al. rtmlib heeft geen hook, dus die halen we zelf op.
datas = collect_data_files("rtmlib")

hiddenimports = [
    # De GUI importeert de backend pas bij de eerste analyse (`_laad_backend`), en
    # `_backend_beschikbaar()` vraagt via find_spec of ultralytics/torch/rtmlib er zijn —
    # dus die drie moeten in de bundel zitten ook al staat er nergens een top-level import.
    "schaats_yolo",
    "ultralytics",
    "torch",
    "rtmlib",
    # Gegenereerd door maak_versie.py; `_versie_uit_bundel()` in schaats_db.py doet
    # `import _versie` binnen een try, dus zonder deze regel zou de bundel zonder
    # versiestempel kunnen eindigen en elke analyse "onbekend" in de Info-dialoog geven.
    "_versie",
]

# ── Weglaten ────────────────────────────────────────────────────────────────────
# Hier zit de winst: PySide6 levert alle Qt-modules mee (634 MB) terwijl deze app er vier
# gebruikt (QtCore, QtGui, QtWidgets, QtCharts). Wat niet geimporteerd wordt komt er in
# principe niet in, maar matplotlib (via ultralytics.utils.plotting) sleept graag Qt- en
# tk-backends mee; deze lijst is het slot op de deur.
excludes = [
    # De MediaPipe-backend zit bewust niet in dit pakket: IS_YOLO is hier altijd waar.
    "mediapipe",
    # Qt-modules die de app niet aanraakt.
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtSpatialAudio", "PySide6.QtDesigner", "PySide6.QtUiTools", "PySide6.QtTest",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "PySide6.Qt3DExtras", "PySide6.Qt3DAnimation", "PySide6.QtDataVisualization",
    "PySide6.QtGraphs", "PySide6.QtGraphsWidgets", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtTextToSpeech",
    "PySide6.QtSerialBus", "PySide6.QtSerialPort", "PySide6.QtHttpServer",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors", "PySide6.QtHelp",
    "PySide6.QtLocation", "PySide6.QtPositioning", "PySide6.QtWebChannel",
    "PySide6.QtWebSockets", "PySide6.QtNetworkAuth",
    # GUI-toolkits en notebook-gereedschap dat matplotlib/ultralytics kan meetrekken.
    "tkinter", "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_webagg",
    "IPython", "jupyter", "notebook", "pytest", "PyInstaller",
    # De polars-runtime is met 177 MB het op twee na grootste brok van de bundel en komt
    # via ultralytics mee. Alle polars-imports daar staan in trainings-, benchmark-,
    # plot- en dataframe-exportpaden ("scope for faster 'import ultralytics'"), en deze
    # app doet alleen inferentie. Nagemeten met polars hard geblokkeerd via sys.meta_path:
    # `import schaats_yolo`, `_laad_yolo()` (DirectML-route) en daarna `model.track()` en
    # `model.predict()` op een frame draaien alle drie zonder polars ooit te laden.
    "polars", "_polars_runtime_32",
    # Wetenschappelijke pakketten die hier niet geinstalleerd zijn en waarvan een
    # optionele import in de graaf niets te zoeken heeft.
    "scipy", "pandas", "tensorflow", "jax",
]

a = Analysis(
    ["schaats_gui.py"],
    pathex=[HIER],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SchaatsAnalyse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX sloopt Qt- en torch-DLL's
    console=False,             # geen console: uitvoer gaat naar het logboek (stap 2)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Het icoon van de exe en daarmee van elke snelkoppeling die de installer
    # (stap 4) aanmaakt. Vervangen = schaatsanalyse.ico overschrijven en opnieuw bouwen.
    # Ontbreekt het bestand, dan stopt PyInstaller met "Unable to open icon file" —
    # geen terugval op zijn eigen icoon, dus het .ico hoort in git (en staat er ook in).
    icon=os.path.join(HIER, "schaatsanalyse.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SchaatsAnalyse",
)
