; ===========================================================================
;  installer.iss — Inno Setup-recept voor SchaatsAnalyse (EXE.md stap 4).
;
;  Verpakt de PyInstaller-uitvoer uit `dist\SchaatsAnalyse\` (stap 3) plus de drie
;  modellen tot één download: `dist\SchaatsAnalyse-setup.exe`.
;
;  Bouwen gaat via `bouw.bat`, dat achtereenvolgens de versiestempel maakt,
;  PyInstaller draait, de modellen naast de exe zet en dit script compileert. Los:
;
;      "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /DVersie=2026-08-25.a4be1f2b installer.iss
;
;  Vijf keuzes die verklaring verdienen:
;
;  1. **PrivilegesRequired=lowest** — installeert naar {localappdata}\Programs\SchaatsAnalyse.
;     Twee redenen: geen beheerdersrechten nodig (de trainers zitten op werk-laptops), en
;     de installatiemap is schrijfbaar, zodat de terugval in `_onnx_pad` (stap 1.4) niet
;     meteen nodig is. Werkt die map ooit tóch niet, dan schrijft de app naar `data_dir()`.
;  2. **Alles uit `dist\SchaatsAnalyse` in één regel**, dus inclusief de modellen die
;     `bouw.bat` daar al neerzet. Geen tweede kopieerroute die uit de rtmlib-cache plukt:
;     dan zou de installer iets anders kunnen bevatten dan de map die je zojuist getest hebt.
;     De `#if`-controles hieronder weigeren te compileren als er een model ontbreekt — anders
;     zou de app bij de eerste analyse stilzwijgend 178 MB gaan downloaden (zie stap 1.5).
;  3. **Niets uit `%LOCALAPPDATA%\SchaatsAnalyse` of uit de bibliotheek wordt aangeraakt**,
;     ook niet bij verwijderen. Daar staan het logboek, `config.json` (met het pad naar de
;     Drive-map) en eventueel een zelf gemaakte ONNX-export; de trainingsdata staat in Drive.
;     Verwijderen van de app mag nooit trainingsdata raken, en een herinstallatie hoort de
;     bibliotheek gewoon terug te vinden.
;  4. **`SetupLogging=yes`** — zelfde gedachte als het logboek uit stap 2: gaat de installatie
;     bij een collega mis, dan staat er een `Setup Log*.txt` in `%TEMP%` om naar te vragen.
;  5. **Geen versie-nummer in de bestandseigenschappen** (`VersionInfoVersion`): dat veld eist
;     `x.y.z.w` en onze stempel is een datum + commit-hash. Die staat wél als AppVersion in
;     "Apps en onderdelen", en dat is de plek waar je hem zoekt.
;
;  De exe is niet gesigneerd: SmartScreen meldt bij de eerste start "Windows heeft uw pc
;  beschermd" (Meer informatie → Toch uitvoeren). Dat hoort in INSTALLEREN.md (stap 6).
; ===========================================================================

#define Naam "SchaatsAnalyse"
#define ExeNaam "SchaatsAnalyse.exe"
#define Uit AddBackslash(SourcePath) + "dist\SchaatsAnalyse"

; De versiestempel komt van `maak_versie.py --toon` via bouw.bat (/DVersie=...). Los
; compileren mag ook; dan staat er "onbekend" in Apps en onderdelen — de analyses zelf
; houden hun eigen stempel uit `_versie.py`, dat is een losse weg.
#ifndef Versie
  #define Versie "onbekend"
#endif

; ── Compileren weigeren als de build niet compleet is ──────────────────────────
#if !FileExists(AddBackslash(Uit) + ExeNaam)
  #error dist\SchaatsAnalyse\SchaatsAnalyse.exe ontbreekt — draai eerst bouw.bat (stap 3).
#endif
#if !FileExists(AddBackslash(Uit) + "yolo26x-pose.pt")
  #error yolo26x-pose.pt ontbreekt in dist\SchaatsAnalyse — zonder dit model downloadt de app 126 MB bij het eerste gebruik.
#endif
#if !FileExists(AddBackslash(Uit) + "yolo26x-pose-dml.onnx")
  #error yolo26x-pose-dml.onnx ontbreekt in dist\SchaatsAnalyse — zonder deze export valt de detectiepass terug op de CPU (2,2x trager).
#endif
; Deze naam staat als RTMPOSE_LOKAAL in schaats_yolo.py; heet het bestand anders, dan
; pakt de app stilzwijgend de URL en downloadt 178 MB bij het eerste gebruik.
#if !FileExists(AddBackslash(Uit) + "rtmpose-x-halpe26-384x288.onnx")
  #error rtmpose-x-halpe26-384x288.onnx ontbreekt in dist\SchaatsAnalyse — zonder dit model downloadt de app 178 MB bij het eerste gebruik.
#endif

[Setup]
; Deze GUID is de identiteit van de app: hij bepaalt of een tweede installatie een
; upgrade is of een tweede kopie. Nooit wijzigen.
AppId={{81C5E169-E35C-4FA6-93F1-58D66B23270E}
AppName={#Naam}
AppVersion={#Versie}
AppVerName={#Naam} {#Versie}
AppPublisher={#Naam}
DefaultDirName={autopf}\{#Naam}
DefaultGroupName={#Naam}
; Niemand wil een startmenumap kiezen.
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; De bundel is 64-bits (torch, onnxruntime, Qt); PySide6 vraagt Windows 10 of nieuwer.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#SourcePath}dist
OutputBaseFilename={#Naam}-setup
SetupIconFile={#SourcePath}schaatsanalyse.ico
UninstallDisplayIcon={app}\{#ExeNaam}
UninstallDisplayName={#Naam}
; De 1,3 GB bestaat voor de helft uit modellen (.pt is al een zip, .onnx zijn ruwe
; gewichten): reken op weinig winst daarop en op een compileerslag van tientallen minuten.
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes

[Languages]
Name: "nl"; MessagesFile: "compiler:Languages\Dutch.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; De hele uitvoermap van PyInstaller inclusief `_internal` en de drie modellen die
; bouw.bat ernaast heeft gezet. `app_dir()` is bevroren de map van de exe, dus de app
; vindt de modellen hier zonder configuratie.
Source: "{#Uit}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#Naam}"; Filename: "{app}\{#ExeNaam}"
Name: "{autodesktop}\{#Naam}"; Filename: "{app}\{#ExeNaam}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeNaam}"; Description: "{cm:LaunchProgram,{#StringChange(Naam, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
