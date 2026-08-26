@echo off
rem ===========================================================================
rem  bouw.bat - bouwt SchaatsAnalyse tot een draaiende map + een installer
rem  (EXE.md stap 3 en 4).
rem
rem  Doet vier dingen, in deze volgorde:
rem    1. maak_versie.py  -> _versie.py met de git-stempel (anders zou elke analyse
rem       uit de exe zonder versiestempel in de bibliotheek belanden).
rem    2. PyInstaller met schaatsanalyse.spec -> dist\SchaatsAnalyse\
rem    3. de modellen naast de exe zetten. Die zitten bewust NIET in de bundel
rem       (557 MB kopieerwerk per herbouw); de installer doet dit in stap 4, en
rem       hier gebeurt het zodat de gebouwde map meteen te testen is.
rem    4. installer.iss compileren met Inno Setup -> dist\SchaatsAnalyse-setup.exe.
rem       Ontbreekt Inno Setup, dan wordt deze stap overgeslagen met een melding: de
rem       map uit stap 2+3 is dan nog steeds te starten en te testen.
rem
rem  De losse scripts blijven gewoon werken - dit staat er los naast.
rem  Draaien vanuit de repomap:  bouw.bat
rem ===========================================================================
setlocal
cd /d "%~dp0"

set "PY=.venv-yolo\Scripts\python.exe"
set "UIT=dist\SchaatsAnalyse"
set "RTMCACHE=%USERPROFILE%\.cache\rtmlib\hub\checkpoints"
set "RTMBRON=%RTMCACHE%\rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.onnx"
rem Deze naam staat als RTMPOSE_LOKAAL in schaats_yolo.py; heet het bestand anders,
rem dan pakt de app stilzwijgend de URL en downloadt 178 MB bij het eerste gebruik.
set "RTMDOEL=%UIT%\rtmpose-x-halpe26-384x288.onnx"

if not exist "%PY%" (
    echo [fout] %PY% niet gevonden. Bouwen moet in .venv-yolo ^(Python 3.11^).
    exit /b 1
)

echo.
echo == 1/4  Versiestempel ==
"%PY%" maak_versie.py || exit /b 1

echo.
echo == 2/4  PyInstaller ==
"%PY%" -m PyInstaller --noconfirm --clean schaatsanalyse.spec || exit /b 1

echo.
echo == 3/4  Modellen naast de exe ==
if not exist "%UIT%" (
    echo [fout] %UIT% bestaat niet - de build is niet gelukt.
    exit /b 1
)
rem PyInstaller wist met --noconfirm de hele uitvoermap, dus dit is per build ~557 MB
rem kopieerwerk van de ene plek op de schijf naar de andere (~30 s). Dat is de prijs voor
rem het NIET in de bundel stoppen van de modellen: een herbouw hoeft ze niet door de
rem PyInstaller-molen te halen, en een model vervangen kan zonder opnieuw te bouwen.
for %%M in (yolo26x-pose.pt yolo26x-pose-dml.onnx) do (
    if exist "%%M" (
        copy /Y "%%M" "%UIT%\" >nul || exit /b 1
        echo    %%M
    ) else (
        echo    [let op] %%M ontbreekt - de app exporteert of downloadt hem zelf bij
        echo             het eerste gebruik, naar %%LOCALAPPDATA%%\SchaatsAnalyse.
    )
)
if exist "%RTMBRON%" (
    copy /Y "%RTMBRON%" "%RTMDOEL%" >nul || exit /b 1
    echo    rtmpose-x-halpe26-384x288.onnx
) else (
    echo    [let op] RTMPose-model niet in de rtmlib-cache gevonden:
    echo             %RTMBRON%
    echo             De app downloadt hem dan zelf ^(178 MB^) bij het eerste gebruik.
)

echo.
echo == 4/4  Installer ^(Inno Setup^) ==
rem Inno Setup 6 installeert per gebruiker (winget --scope user) of machinebreed; beide
rem paden aflopen, en anders nog PATH. Niet gevonden = geen fout: dan lever je deze keer
rem alleen de map op, en dat is voor testen op deze machine genoeg.
set "ISCC="
for %%P in (
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
) do if not defined ISCC if exist %%P set "ISCC=%%~P"
if not defined ISCC for /f "delims=" %%P in ('where iscc 2^>nul') do if not defined ISCC set "ISCC=%%P"

if not defined ISCC (
    echo    [overgeslagen] Inno Setup 6 niet gevonden.
    echo                   Installeren:  winget install JRSoftware.InnoSetup
    echo    De gebouwde map is wel te gebruiken: %UIT%\SchaatsAnalyse.exe
    goto :klaar
)

rem De versiestempel gaat als /DVersie mee zodat "Apps en onderdelen" laat zien welke build
rem er staat. ASCII-vorm, want dit loopt door een for-lus in cmd.exe (zie maak_versie.py).
set "VERSIE="
for /f "delims=" %%V in ('%PY% maak_versie.py --toon') do set "VERSIE=%%V"
if not defined VERSIE set "VERSIE=onbekend"
echo    versie %VERSIE%
rem /Qp = stil met een voortgangsteller; zonder /Q rolt er een regel per bestand voorbij
rem (~2000 stuks) en met /Q zie je twintig minuten lang niets.
"%ISCC%" /Qp /DVersie=%VERSIE% installer.iss || exit /b 1
echo    dist\SchaatsAnalyse-setup.exe

:klaar
echo.
echo Klaar: %UIT%\SchaatsAnalyse.exe
endlocal
