@echo off
rem ===========================================================================
rem  build.bat - builds SkateAnalysis into a runnable folder + an installer
rem  (EXE.md steps 3 and 4).
rem
rem  Does four things, in this order:
rem    1. make_version.py -> _version.py with the git stamp (otherwise every
rem       analysis out of the exe would land in the library without a version
rem       stamp).
rem    2. PyInstaller with skate_analysis.spec -> dist\SkateAnalysis\
rem    3. puts the models next to the exe. They're deliberately NOT in the bundle
rem       (557 MB of copying per rebuild); the installer does this in step 4, and
rem       here it happens so the built folder can be tested right away.
rem    4. compiles installer.iss with Inno Setup -> dist\SkateAnalysis-setup.exe.
rem       If Inno Setup is missing, this step is skipped with a notice: the folder
rem       from steps 2+3 is still runnable and testable.
rem
rem  The standalone scripts keep working as-is - this sits alongside them.
rem  Run from the repo folder:  build.bat
rem ===========================================================================
setlocal
cd /d "%~dp0"

set "PY=.venv-yolo\Scripts\python.exe"
set "OUT=dist\SkateAnalysis"
set "RTMCACHE=%USERPROFILE%\.cache\rtmlib\hub\checkpoints"
set "RTMSRC=%RTMCACHE%\rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.onnx"
rem This name is RTMPOSE_LOCAL in skate_yolo.py; if the file has a different name,
rem the app silently falls back to the URL and downloads 178 MB on first use.
set "RTMDEST=%OUT%\rtmpose-x-halpe26-384x288.onnx"

if not exist "%PY%" (
    echo [error] %PY% not found. Build must run inside .venv-yolo ^(Python 3.11^).
    exit /b 1
)

echo.
echo == 1/4  Version stamp ==
"%PY%" make_version.py || exit /b 1

echo.
echo == 2/4  PyInstaller ==
"%PY%" -m PyInstaller --noconfirm --clean skate_analysis.spec || exit /b 1

echo.
echo == 3/4  Models next to the exe ==
if not exist "%OUT%" (
    echo [error] %OUT% does not exist - the build failed.
    exit /b 1
)
rem PyInstaller wipes the whole output folder with --noconfirm, so this is ~557 MB of
rem copying from one place on disk to another on every build (~30 s). That's the price
rem for NOT putting the models in the bundle: a rebuild doesn't have to run them
rem through the PyInstaller mill, and a model can be swapped without rebuilding.
for %%M in (yolo26x-pose.pt yolo26x-pose-dml.onnx) do (
    if exist "%%M" (
        copy /Y "%%M" "%OUT%\" >nul || exit /b 1
        echo    %%M
    ) else (
        echo    [note] %%M is missing - the app will export or download it itself on
        echo           first use, into %%LOCALAPPDATA%%\SkateAnalysis.
    )
)
if exist "%RTMSRC%" (
    copy /Y "%RTMSRC%" "%RTMDEST%" >nul || exit /b 1
    echo    rtmpose-x-halpe26-384x288.onnx
) else (
    echo    [note] RTMPose model not found in the rtmlib cache:
    echo           %RTMSRC%
    echo           The app will download it itself ^(178 MB^) on first use.
)

echo.
echo == 4/4  Installer ^(Inno Setup^) ==
rem Inno Setup 6 installs per-user (winget --scope user) or machine-wide; check both
rem paths, then PATH. Not found isn't an error: this time you only get the folder,
rem and that's enough for testing on this machine.
set "ISCC="
for %%P in (
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
) do if not defined ISCC if exist %%P set "ISCC=%%~P"
if not defined ISCC for /f "delims=" %%P in ('where iscc 2^>nul') do if not defined ISCC set "ISCC=%%P"

if not defined ISCC (
    echo    [skipped] Inno Setup 6 not found.
    echo              Install with:  winget install JRSoftware.InnoSetup
    echo    The built folder still works: %OUT%\SkateAnalysis.exe
    goto :done
)

rem The version stamp goes along as /DVersion so "Apps and features" shows which
rem build is installed. ASCII form, since this goes through a for-loop in cmd.exe
rem (see make_version.py).
set "VERSION="
for /f "delims=" %%V in ('%PY% make_version.py --show') do set "VERSION=%%V"
if not defined VERSION set "VERSION=unknown"
echo    version %VERSION%
rem /Qp = quiet with a progress counter; without /Q a line scrolls by per file
rem (~2000 of them) and with /Q you see nothing for twenty minutes.
"%ISCC%" /Qp /DVersion=%VERSION% installer.iss || exit /b 1
echo    dist\SkateAnalysis-setup.exe

:done
echo.
echo Done: %OUT%\SkateAnalysis.exe
endlocal
