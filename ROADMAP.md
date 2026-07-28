# Roadmap SchaatsAnalyse

Plan voor de volgende ontwikkelfase, in volgorde van bouwen. Gemaakte keuzes (juli 2026):

- **Delen**: één gedeelde cloudmap (OneDrive/Dropbox/netwerkschijf) met daarin een SQLite-bestand + mediabestanden. Geen server, geen accounts.
- **Video's**: worden mee-gekopieerd naar de gedeelde map, zodat elke trainer de analyse mét beeld kan terugkijken.
- **Schaal**: één team (±5–30 schaatsers, 1–5 trainers). Ontwerp mag daarop leunen; geen rechten-/rollensysteem nodig.
- **Skelet-editor**: correcties vloeien uit naar buurframes met een **instelbaar venster** (0 = alleen het bewerkte frame).
- **Opnameopstelling (aanname sinds juli 2026)**: er wordt **altijd recht van voren** gefilmd en de camera staat **altijd precies horizontaal**. Daardoor vervalt de noodzaak van camerakanteling-correctie (horizon) en perspectiefcorrectie in de dagelijkse workflow. Fases 5 en 7 blijven staan als **nice-to-have** voor eventuele latere opstellingen (schuine/schommelende camera), maar zijn **nu geen prioriteit**.

De fases bouwen op elkaar: 0 → 1 → 2 kunnen niet van volgorde wisselen; 3 (skelet-editor) en 4 (delen) zijn daarna onafhankelijk van elkaar te bouwen. Fase 6 (sneller analyseren) staat los van de rest en kan op elk moment, ook eerder. Fase 5 (horizon-tracking) en 7 (perspectiefcorrectie) zijn onder de vaste opnameopstelling **nice-to-have** (zie hierboven); fase 7 deelt bouwstenen met fase 5 (lijnen aanwijzen/tracken) en omvat de horizoncorrectie als speciaal geval.

---

## Fase 0 — Voorbereiding: resultaten serialiseerbaar maken ✅

> **Af (16 juli 2026)** — geverifieerd op een echte video via de GUI: opslaan, GUI herstarten, terugladen → identieke tabel/grafiek/overlay, in een seconde i.p.v. een volledige detectie.
>
> **Wat er staat:**
> - `schaats_analyse.py`, sectie "Serialisatie (fase 0)" vóór `segmenteer_afzetten`: `resultaten_naar_arrays()` / `arrays_naar_resultaten()` + `sla_landmarks_op()` / `laad_landmarks()` (`np.savez_compressed`).
> - CLI: `--save-npz PAD` (landmarks wegschrijven na de analyse) en `--from-npz PAD` (detectie overslaan, alleen afgeleiden herberekenen + overlay tekenen; model niet nodig).
> - GUI: knoppen "Landmarks opslaan (.npz)" (datapaneel) en "Landmarks laden (.npz)..." (startpagina). **Tijdelijk steigerwerk** — fase 1 vervangt het handmatig kiezen van bestanden door de bibliotheek; de functies eronder blijven gelijk.
> - `_analyse_klaar` is gesplitst: het weergave-deel is nu `_toon_resultaten(info, resultaten, events, bron=None)`, gedeeld door een verse en een geladen analyse. **Dit is de naad die fase 1 hergebruikt.**
>
> **Afwijkingen van het plan hieronder:**
> - Stap 2's "plain landmark-type" bleek al te bestaan (`Landmark`-namedtuple, `schaats_analyse.py` regel ~51, al gebruikt door de YOLO-backend) — geen nieuw type nodig.
> - `arrays_naar_resultaten(arrays)` neemt géén `info`-argument: de video-meta (w/h/fps/totaal) gaat mee ín het `.npz` en komt er als `VideoInfo` weer uit → `(VideoInfo, resultaten)`. Terugladen vereist de video dus niet.
> - Stap 4 klopte: `verwerk_afgeleiden()` had geen verborgen afhankelijkheid van de detectie-pass.
>
> **Niet meegeserialiseerd:** de perspectiefkalibratie (fase 7). Bij laden staat die dus uit — onschadelijk onder de frontale-camera-aanname.

Alles hierna staat of valt met het kunnen **opslaan en terugladen** van een analyse. Nu leeft de `FrameResultaat`-lijst alleen in het geheugen van de GUI, en het `lm`-veld bevat een MediaPipe-landmarkobject dat niet direct naar schijf kan.

**Te bouwen (in `schaats_analyse.py`):**

1. `resultaten_naar_arrays(resultaten)` → dict met numpy-arrays:
   - `landmarks`: `(n_frames, 33, 3)` float32 — genormaliseerde x, y, visibility (z gebruiken we nergens);
   - `pose_gevonden`: `(n_frames,)` bool; `horizon_deg`: `(n_frames,)` float32.
   - Afgeleiden (been/hoek/gewicht/events) **niet** opslaan als bron van waarheid: die zijn herberekenbaar uit de landmarks via `verwerk_afgeleiden()` + `segmenteer_afzetten()`. Wel cachen (zie fase 1) voor snelle weergave.
2. `arrays_naar_resultaten(arrays, info)` → verse `FrameResultaat`-lijst met een **plain landmark-type** (bv. een klein `Landmark`-dataclassje met `x/y/visibility`) i.p.v. het MediaPipe-object. Alle bestaande code (`get_landmarks`, `teken_alle_landmarks`, smoothing) leest alleen `.x/.y/.visibility`, dus dit werkt zonder verdere aanpassing — wel even verifiëren. De YOLO-backend (`_coco_naar_landmarks`) maakt al eigen landmark-objecten, dus dit trekt beide backends gelijk.
3. Opslag als **`.npz`** (`np.savez_compressed`): een video van 3000 frames is ± 1–2 MB. Los bestand naast de video, níet als blob in SQLite — houdt de database klein en cloudsync-vriendelijk.
4. `verwerk_afgeleiden()` moet los aanroepbaar zijn op een teruggeladen lijst (is hij in essentie al — controleren dat er geen verborgen afhankelijkheid van de detectie-pass is).

**Klaar wanneer:** een analyse wegschrijven naar `.npz`, GUI herstarten, terugladen en identieke tabel/grafiek/overlay zien — zonder de video opnieuw te analyseren.

---

## Fase 1 — Profielen + database ✅

> **Af (18 juli 2026)** — zelftest (`python schaats_db.py`) + headless GUI-rooktest groen in beide venvs; volledige cyclus (schaatser aanmaken → analyseren → heropenen) werkt.
>
> **Wat er staat:** `schaats_db.py` (config, schema `user_version=1`, CRUD, `sla_analyse_op` met video+npz eerst en DB-insert als laatste stap, zelftest); GUI-startpagina = bibliotheek (schaatsers links, analyses rechts uit de events-cache, dubbelklik = openen, hernoemen/verwijderen met bevestiging); `NieuweAnalyseDialog`; `AnalyseWorker` slaat na de analyse automatisch op (in de workerthread, met "Opslaan in bibliotheek..."-fase); heropenen via de fase 0-naad met de **opgeslagen** `smooth_n`/`threshold` uit `instellingen_json`. De fase 0-steigerknoppen (npz opslaan/laden) zijn weg.
>
> **Afwijkingen van het plan hieronder:**
> - De instellingen-groupbox verhuisde naar de `NieuweAnalyseDialog` (geen derde stackpagina — de flow was al een keten van modale dialogen).
> - Uit fase 4 naar voren gehaald (licht): bibliotheekpad-config in `%APPDATA%\SchaatsAnalyse\config.json` + knop "Bibliotheekmap...", én de cloud-veilige SQLite-discipline (journal DELETE, korte verbindingen per aanroep, busy_timeout, relatieve paden met forward slashes). De bibliotheek kan dus nu al in een gedeelde cloudmap (Google Drive/OneDrive/Dropbox) staan; fase 4 voegt alleen nog trainersnaam + conflictdetectie toe.
> - Faalt alléén het opslaan, dan blijft de (lange) analyse zichtbaar met een waarschuwing — hij is dan alleen niet bewaard.
> - De perspectiefkalibratie wordt (net als in fase 0) niet geserialiseerd; bij heropenen van zo'n analyse volgt een eenmalige melding dat de hoeken zonder correctie herberekend zijn.
> - Besloten open vragen: video wordt altijd gekopieerd (origineel blijft staan, bestandsnaam behouden in de uuid-map); geen import van oude losse analyses.

**Doel:** elke schaatser een profiel; elke analyse hoort bij een profiel.

### Opslagstructuur (de "bibliotheek")

Eén map, later te delen via de cloud (fase 4). Pad instelbaar; standaard lokaal, bv. `Documenten\SchaatsAnalyse`:

```
<bibliotheek>/
  schaats.db                 ← SQLite: profielen, analyses, events-cache
  media/
    <analyse-id>/
      video.mp4              ← gekopieerd origineel (naam behouden mag ook)
      landmarks.npz          ← gesmoothte landmarks (fase 0)
      landmarks_ruw.npz      ← idem, vóór handmatige edits (fase 3)
```

`analyse-id` = een UUID, zodat twee trainers nooit botsende mapnamen maken.

### Databaseschema (stdlib `sqlite3`, werkt in beide venvs, geen nieuwe dependency)

```sql
schaatser(id, naam, geboortejaar, notities, aangemaakt_op)
analyse(id TEXT PRIMARY KEY,          -- UUID, tevens mapnaam onder media/
        schaatser_id, titel, datum,
        video_bestand,                 -- relatief pad binnen de bibliotheek
        w, h, fps, totaal_frames,
        backend,                       -- 'yolo' | 'mediapipe'
        instellingen_json,             -- doel_punt, horizon, smooth, ...
        aangemaakt_door,               -- trainersnaam (vrije tekst, zie fase 4)
        bewerkt,                       -- 0/1: zijn er handmatige skelet-edits
        aangemaakt_op)
afzet_event_cache(analyse_id, idx, been, start_frame, eind_frame,
                  hoek, min_hoek, max_hoek, opmerking)
```

`afzet_event_cache` is puur voor snelle lijstweergave ("laatste analyse: gem. 42°") zonder eerst de `.npz` te laden; bij openen van een analyse wordt alles vers herberekend uit de landmarks.

Nieuwe module **`schaats_db.py`**: `open_db(pad)` (maakt schema aan indien nodig), CRUD voor schaatsers/analyses, `sla_analyse_op(...)` (kopieert video + schrijft npz + insert in één transactie), `laad_analyse(id)`. Houd alle SQL hier; GUI praat alleen met deze module.

### GUI-wijzigingen (`schaats_gui.py`)

- **Nieuwe startpagina = bibliotheek**: links de schaatserslijst (+ knop "nieuwe schaatser"), rechts de analyses van de geselecteerde schaatser (datum, titel, aantal afzetten, gem. hoek uit de cache). Dubbelklik → analyse openen.
- **"Nieuwe analyse"-flow**: schaatser kiezen (of aanmaken) → video kiezen → bestaande `DoelKiezer` → `AnalyseWorker` draait → bij `klaar` automatisch opslaan in de bibliotheek → analyse-weergave openen. De huidige weergavepagina blijft vrijwel ongewijzigd; alleen leest `cap_weergave` voortaan de gekopieerde video uit `media/<id>/`.
- **Aanhaakpunt uit fase 0**: een analyse openen = `laad_landmarks()` + `verwerk_afgeleiden()` + `segmenteer_afzetten()` → `_toon_resultaten(...)`. Die weg werkt al (de tijdelijke "Landmarks laden"-knop doet precies dit); fase 1 vervangt alleen de bestandsdialoog door de bibliotheekselectie en haalt daarna beide tijdelijke knoppen weg.
- Analyse hernoemen/verwijderen (verwijderen = DB-rij + mediamap, met bevestiging).
- Video-kopie kan bij grote bestanden even duren → in de worker-thread doen, niet op de UI-thread.

**Klaar wanneer:** volledige cyclus werkt — schaatser aanmaken, video analyseren, afsluiten, heropenen, analyse uit het profiel terugkijken.

---

## Fase 2 — Profielweergave verrijken

Klein maar waardevol vervolg op fase 1 (kan ook later):

- **Voortgang over tijd**: grafiekje per schaatser met de gemiddelde/beste afzethoek per analyse-datum (data zit al in `afzet_event_cache`).
- Notitieveld per analyse ("linkerbocht geoefend, wind tegen").
- CSV-export per schaatser (alle analyses) naast de bestaande per-analyse-export.

---

## Extra — Batch-analyse ✅

> **Af (20 juli 2026)** — buiten de fasering, bovenop de fase 1-flow. Meerdere video's in één keer analyseren.
>
> **Wat er staat:** `BatchAnalyseDialog` (video's + per rij schaatser/titel + gedeelde instellingen) en `BatchWorker` (`QThread`) in `schaats_gui.py`. De doel-/horizon-keuze gebeurt vooraf per video in `_nieuwe_batch_analyse`; daarna draait de hele rij onbewaakt en slaat elke analyse zelf op via `sla_analyse_op`. Eén mislukte clip stopt de batch niet (gemeld, rest loopt door); "Stop na deze video" = nette stop tussen clips. Geen roadmap-fase, wel logisch vervolg op de bibliotheek — vandaar hier genoteerd.

---

## Extra — Vergelijk schaatsers (twee analyses naast elkaar) ✅

> **Af (27 juli 2026)** — buiten de fasering. Twee opgeslagen analyses naast elkaar om schaatsers (of dezelfde schaatser op twee momenten) te vergelijken.
>
> **Voorwerk — videopaneel uitgefactoreerd.** Alle afspeel-state zat als losse `self.*`-attributen in `MainWindow`, dus twee spelers naast elkaar was onmogelijk. Nieuwe `VideoSpeler(QWidget)` met eigen capture/timer/zoom/pan en alle bediening; `MainWindow` benadert `video_info`/`resultaten`/`huidige_idx` nog via read-only properties, zodat de bestaande editor- en tabelcode ongewijzigd bleef. De skelet-editor haakt in via `overlay_tekenaar` + `op_muis_druk/_beweeg/_los` (plain callables); pannen blijft in de speler, vóór de callback, zodat de voorrangsregel per constructie klopt. Analysepagina werkt daarna identiek — inclusief een meegenomen fix: `slider.setRange` in `laad()` staat nu in `blockSignals`, zodat een korte analyse na een lange geen spook-seek meer geeft.
>
> **Wat er staat:** knop "Vergelijk schaatsers..." op de startpagina → derde pagina met twee `VergelijkKant`-widgets (kop, eigen `VideoSpeler`, "Kies analyse...", sync-punt, minimale afzettabel `#`/`Been`/`Hoek` met grijze markering voor onvolledige afzetten). `AnalyseKiezer` (schaatser → analyse) wordt tweemaal gebruikt bij het openen en per kant om te wisselen. Elke kant is los af te spelen; **"Start alles"** speelt beide vanaf hun sync-punt via **één masterklok** die het doelframe per kant uit de wandkloktijd × de eigen fps berekent — zelfcorrigerend en correct bij verschillende fps (twee losse timers zouden binnen seconden uit de pas lopen). Standaard ¼×. Gedeelde laadhelper `_laad_analyse_data` raakt geen `MainWindow`-state aan, zodat de vergelijking de instellingen van de geopende analyse niet overschrijft.
>
> **Bewust nog niet:** sync-punten worden niet opgeslagen (geen DB-kolom, zou een schemabump + migratie kosten); geen gecombineerde statistiek of grafiek over de twee kanten heen; de tabel is opzet minimaal — eerst kijken wat een trainer in de praktijk mist.

---

## Extra — Automatische zoom ✅

> **Af (28 juli 2026)** — het programma bepaalt de zoom zelf: de schaatser staat de hele clip helemaal in beeld met wat lucht eromheen. Buiten de fasering; hoorde bij de drie losse verbeterpunten van 27 juli.
>
> **Waarom:** gemeten over de bibliotheek wordt een schaatser tijdens een clip 1,4–5,8× groter in beeld. Eén vaste zoomfactor klopt dus hooguit een paar seconden en de trainer zat tijdens het afspelen aan de slider. De trainer wil de kadering ook niet zélf hoeven kiezen — dat is werk dat het programma kan doen.
>
> **Wat er staat:** `kader_reeks()` in `schaats_analyse.py` berekent bij het laden één keer offline per frame `(midden, straal)` van de schaatser — vooraf i.p.v. online, zodat scrubben exact dezelfde uitsnede geeft als ernaartoe afspelen. In `VideoSpeler` staan `_zoom` (ingesteld, 1–5×) en `_zoom_eff` (toegepast, tot 8×) los van elkaar; met de checkbox **"Automatische zoom"** (standaard uit) volgt `_zoom_eff` uit de kaderstraal plus 15% marge, en volgt de uitsnede het kader-middelpunt. Zolang de automaat aan staat zijn de zoom-slider en "Passend" uitgeschakeld; aan het muiswiel draaien neemt de zoom weer over. Zit in `VideoSpeler`, dus de vergelijkpagina heeft het per kant — twee schaatsers op verschillende afstand worden zo pas echt vergelijkbaar. Zie CLAUDE.md (bullet "Automatische zoom") voor de details.
>
> **Wat de meting uitwees (op echte analyses, alle frames nagelopen):**
> - Kaderen om `torso_centroid` verspilt een kwart van het beeld (dat punt ligt hoog in het lichaam) — vandaar het midden van álle zichtbare landmarks: beeldvulling 0,40 → 0,56.
> - Alleen smoothen vlakt de piek af en dan valt een uitgestrekt been buiten beeld; eerst een lopend maximum over één slag, dán smoothen. Resultaat: geen enkel frame met de schaatser buiten de uitsnede, bij < 2% zoomverandering per frame.
> - `KADER_POLY=1` i.p.v. de gedeelde `SMOOTH_POLY=2`: de kwadratische randfit extrapoleert de slag-golf en zit 15% mis op het eerste frame (lineair: 3%).
> - Het plafond hoort niet op de videoresolutie te zitten maar op de vergroting op het scherm: staande telefoonclips staan in een liggend paneel al gekrompen en mogen ver inzoomen, 4K-liggend nauwelijks.
> - Bij een lang detectiegat hield het kader de laatst bekende stand vast — dat leverde een 8×-uitvergroting van de plek waar de schaatser wás (zichtbaar op de eerste seconde van IMG_9002). Nu gaat het kader na `KADER_GAT_S` vloeiend open tot het volledige beeld.
>
> **Bewust niet:** geen persistentie (zoomstand en checkbox horen bij het kijken, niet bij de analyse); geen automatische keuze wanneer de automaat aan moet — dat blijft een vinkje.

---

## Extra — Soepeler vergelijken ✅

> **Af (28 juli 2026)** — de twee losse verbeterpunten van 27 juli op de vergelijkpagina.
>
> **Rechtstreeks vergelijken vanuit een geopende analyse:** knop **"⇄ Vergelijk met..."** in de transportbalk van de weergavepagina (naast "Bewerken", via `voeg_bedieningsknop`). De geopende analyse gaat altijd naar de **linkerkant** — vanuit een geopende analyse is er nooit een lege kant om slim over na te denken, dus voorspelbaar is beter — en voor rechts wordt meteen om een analyse gevraagd, voorgeselecteerd op dezelfde schaatser (dus "deze schaatser toen vs. nu" is twee klikken). Stond er rechts al een ándere analyse, dan blijft die staan inclusief sync-punt; annuleren van de kiezer opent de pagina gewoon met alleen links gevuld. De kant laadt de analyse **opnieuw uit de bibliotheek** i.p.v. de resultatenlijst van de weergavepagina te delen: de skelet-editor muteert die `FrameResultaat`-objecten in place. Daarvoor is `_kies_vergelijk_kant` gesplitst in de dialoog en een herbruikbaar `_zet_vergelijk_kant(kant, analyse_id, naam)`; de knop staat uit zolang er geen opgeslagen analyse open is (een niet-bewaarde analyse valt niet uit de bibliotheek te laden).
>
> **Eén kant wisselen** bleek er al te zijn: elke `VergelijkKant` had z'n eigen "Kies analyse..."-knop die alleen die kant vervangt (meegekomen met de `VideoSpeler`-refactor, maar nooit uit deze lijst gehaald). Wat er wél bij moest: de knop heet **"Wisselen..."** zodra er een analyse staat, er is een **✕**-knop om een kant leeg te maken (bedraad vanuit de pagina zodat de masterklok eerst losgelaten wordt), en dezelfde analyse opnieuw laden **houdt het sync-punt** — dat hoort bij de video, niet bij het laden. Een andere analyse begint nog steeds op frame 0.
>
> **Bewust niet:** nog steeds geen opgeslagen sync-punten (zie de sectie hierboven — schemabump); geen derde kant.

---

## Fase 3 — Skelet-editor (punten verslepen) ✅

> **Af (20 juli 2026)** — zelftest (`python schaats_db.py`) dekt de opslag-round-trip (bewerken → `landmarks_ruw.npz`-backup + `bewerkt=1` + verse events-cache; herstel origineel → npz terug + `bewerkt=0`); GUI handmatig getest.
>
> **Wat er staat:** "Bewerken"-knop op de weergavepagina (`_toggle_bewerken`) met sleepbare handles per landmark (`_teken_handles`/`_zoek_landmark`/`_handle_straal`), verslepen + uitvloeien naar ± N buurframes met cosinus-afbouw (`_editor_muis_druk`/`_beweeg`/`_los`, `_zet_landmark`, `_uitvloei_frames`), live herberekenen (`_na_edit` → `verwerk_afgeleiden` + `segmenteer_afzetten`, géén smoothing), undo/redo (`_undo`/`_redo`), "herstel origineel". Opslagkant in `schaats_db.py`: `bewaar_bewerkte_landmarks` (npz overschrijven + eenmalige `landmarks_ruw.npz`-backup + `bewerkt=1` + cache), `herstel_originele_landmarks`, `ververs_events_cache`, gedeelde `_schrijf_events_cache`.
>
> **Afwijkingen van het plan hieronder:**
> - Grijpradius schaalt met de torso-lengte op scherm (`GRIJP_MIN_PX`/`GRIJP_MAX_PX`) i.p.v. een vaste 12 px — leesbaar bij een verre schaatser en op elke zoomstand.
> - Pixel-correct bewerken op elke zoomstand vroeg eerst de **crop-and-magnify-zoom** (los gebouwd, zelfde commit; zie CLAUDE.md `_toon_pixmap`/`_crop_norm`).
> - Punten *plaatsen* op frames zónder pose: niet gedaan (conform de openstaande vraag: eerste versie alleen bestaande punten verslepen).

**Doel:** na een analyse kleine detectiefoutjes repareren door landmarkpunten te verslepen; werkt op elke uit de database geladen analyse.

### Interactie

- **Bewerk-knop** op de weergavepagina zet de editor aan: afspelen pauzeert, alle landmarks van het huidige frame krijgen sleepbare handles (cirkeltjes; grijpradius ± 12 px op schermresolutie).
- Muis-events op het videolabel: widget-coördinaten → frame-coördinaten terugrekenen (let op de schaling/letterboxing van het weergavelabel — dit omrekenpad bestaat al half in `DoelKiezer`/`HorizonKiezer`, herbruikbaar maken).
- Slepen werkt op de **gesmoothte** landmarks (wat je ziet is wat je bewerkt); de smoothing-stap wordt na een edit dus níet opnieuw gedraaid, anders wordt de correctie meteen weer weggepoetst.
- Versleepte punten krijgen `visibility = 1.0` (een handmatig gezet punt is per definitie betrouwbaar) en een markering "handmatig" zodat ze in de overlay een ander randje kunnen krijgen.

### Uitvloeien naar buurframes (instelbaar)

- Spinbox "uitvloeien: ± N frames" (default 8, 0 = alleen dit frame).
- De verplaatsing (delta-x, delta-y van dát punt) wordt over het venster gewogen toegepast met een cosinus-afbouw: frame op afstand `k` krijgt `delta * 0.5*(1+cos(pi*k/N))`. Geen sprong in de beweging, en op afstand N is het effect precies 0.
- Uitvloeien stopt bij een detectiegat (frames zonder pose) — niet over gaten heen smeren, zelfde principe als de smoothing.

### Herberekenen + opslaan

- Na elke drop (muisknop los): `verwerk_afgeleiden()` + `segmenteer_afzetten()` opnieuw over de resultatenlijst → tabel, grafiek en HUD verversen live. Dit is puur numpy-werk over reeds gedetecteerde landmarks, ruim snel genoeg.
- **Undo/redo**-stack (bewaar per edit: landmark-index, venster, deltas) — bij handwerk op 12 px-punten ga je gegarandeerd een keer missen.
- **Opslaan**: gewijzigde landmarks → `landmarks.npz` overschrijven; het origineel staat in `landmarks_ruw.npz` (aangemaakt bij de eerste edit) zodat er een knop **"herstel origineel"** kan zijn. `analyse.bewerkt = 1` in de DB en de events-cache verversen.

**Klaar wanneer:** een zichtbaar fout kniepunt in één sleepbeweging corrigeren, de hoektabel direct zien bijwerken, opslaan, heropenen — correctie staat er nog; "herstel origineel" zet alles terug.

---

## Fase 4 — Delen met meerdere trainers (gedeelde cloudmap) ✅

> **Af (20 juli 2026)** — de bibliotheek staat bij dit team in een **Google Drive**-map (Mirror). Zelftest (`python schaats_db.py`) uitgebreid met de v1→v2-migratie, conflictkopie-detectie en de aangemaakt_door/video_bytes-round-trip; headless GUI-rooktest groen in de YOLO-venv.
>
> **Wat er staat:**
> - **Trainersnaam**: knop "Jouw naam..." op de startpagina (naast "Vernieuwen"), bewaard in `config.json` (`trainer_naam`, per gebruiker — niet in de gedeelde map). Gaat als `analyse.aangemaakt_door` mee bij nieuwe én batch-analyses (via `AnalyseWorker`/`BatchWorker`), en verschijnt als tooltip "Aangemaakt door …" op de titel in de analysetabel.
> - **Conflictdetectie** (`schaats_db.detecteer_conflictkopieen`): bij het openen/wisselen van een bibliotheek en bij "Vernieuwen" wordt gewaarschuwd als er naast `schaats.db` andere `*.db`-bestanden staan (conflictkopie van de syncer, bv. `schaats-DESKTOP.db`). Detectie, geen preventie — de trainer ruimt handmatig op.
> - **Vernieuwen-knop**: leest de bibliotheek opnieuw van schijf (`_vernieuw_bibliotheek`) zodat analyses van collega's zichtbaar worden zonder herstart; herhaalt ook de conflictcheck.
> - **Video-sync-check**: schema-migratie naar `user_version=2` voegt `analyse.video_bytes` toe (grootte van de gekopieerde video bij het opslaan). Bij het openen bepaalt `video_sync_status()` of de video ontbreekt (nog niet gedownload) of onvolledig is (kleiner dan opgeslagen → cloud synct nog) en toont een nette melding i.p.v. een halve video te laden. Oude analyses (v1, `video_bytes` NULL) slaan de groottecheck over.
>
> **Afwijkingen van het plan hieronder:**
> - De cloud-veilige SQLite-discipline (journal DELETE, korte verbindingen, busy_timeout, relatieve paden) en de bibliotheekpad-config waren al in fase 1 naar voren gehaald — hier bleef alleen trainersnaam, conflictdetectie, vernieuwen en de sync-groottecheck over.
> - Het optionele lock-bestandje (`media/<id>/.lock`) tegen gelijktijdig bewerken van dezelfde analyse is **bewust niet gebouwd**: het is in de ROADMAP als "desgewenst" gemarkeerd en de kans is bij UUID-mappen + kleine ploeg verwaarloosbaar ("laatste schrijver wint" blijft de geaccepteerde beperking).

**Doel:** het hele team kijkt in dezelfde bibliotheek.

### Aanpak

- **Instellingenscherm**: bibliotheekpad kiezen. Elke trainer wijst dezelfde OneDrive-/Dropbox-/netwerkmap aan. Plus een veld "jouw naam" → gaat in `analyse.aangemaakt_door`. Beide onthouden in een lokaal configbestandje (`%APPDATA%\SchaatsAnalyse\config.json` — niet in de gedeelde map, want per gebruiker).
- Omdat fase 1 alles al relatief t.o.v. de bibliotheekmap opslaat, is delen daarna vooral *configuratie*, geen herbouw. Dit is de reden om die padden-discipline vanaf fase 1 strikt te houden.

### SQLite op een gesynchroniseerde map — de valkuilen en maatregelen

SQLite is niet ontworpen voor gelijktijdig schrijven via cloudsync. Op teamschaal is dit prima beheersbaar, mits:

1. **Geen WAL-mode** (`journal_mode=DELETE`): WAL maakt `-wal`/`-shm`-nevenbestanden die cloudsyncers half kunnen syncen → corruptiegevaar. DELETE-mode houdt het bij één bestand (plus een kortstondige journal).
2. **Kort verbinden**: verbinding openen → transactie → direct sluiten, nooit een connectie open laten staan tijdens het browsen. Dan is het DB-bestand vrijwel altijd "in rust" voor de syncer.
3. `busy_timeout` van een paar seconden voor het zeldzame geval dat twee trainers op hetzelfde moment schrijven via een echte netwerkschijf.
4. **Conflictdetectie i.p.v. -preventie**: als OneDrive tóch een conflictkopie maakt (`schaats-<pc-naam>.db`), detecteer dat bij het opstarten en waarschuw. Omdat analyses UUID-mappen zijn en trainers zelden binnen dezelfde minuut schrijven, is de praktische kans klein; mediabestanden (video/npz) worden alleen aangemaakt, nooit door twee mensen tegelijk beschreven.
5. **Vernieuwen-knop** in de bibliotheekweergave (DB opnieuw uitlezen) zodat je nieuwe analyses van een collega ziet zonder herstart. Automatisch pollen hoeft niet.

### Bewust geaccepteerde beperkingen (bij deze schaal oké)

- Geen accounts/rechten: iedereen met de map kan alles zien én verwijderen.
- "Laatste schrijver wint" bij het tegelijk bewerken van precies dezelfde analyse (bv. beiden in de skelet-editor) — zeldzaam; desgewenst mitigeren met een simpel lock-bestandje (`media/<id>/.lock` met trainersnaam) dat een waarschuwing toont.
- Grote video's syncen traag; wie net een analyse van een collega opent terwijl de video nog bin­nenkomt, krijgt een nette melding "video nog niet gesynchroniseerd" (bestaat het bestand + klopt de bestandsgrootte).

**Upgradepad**: mocht het later tóch clubbreed worden (accounts, rechten, tegelijk schrijven), dan is de stap naar een gehoste Postgres (bv. Supabase) beperkt tot het vervangen van `schaats_db.py` + uploaden van de media — de rest van de app merkt daar niets van. Dáárom alle SQL in één module houden.

---

## Fase 5 — Betere stabilisatie: horizon via twee getrackte punten

> **Nice-to-have (niet nu).** Sinds juli 2026 is de aanname dat de camera **altijd precies horizontaal** staat — dan is er geen camerakanteling om te corrigeren en is deze fase overbodig. Bewaard voor een eventuele latere opstelling met een schuine/schommelende camera; pas oppakken als die situatie zich echt voordoet.

**Doel:** de auto-horizon betrouwbaarder maken. De huidige `bepaal_horizon_reeks()` doet per frame een Hough-lijndetectie op het onderste beeld — die pakt soms de verkeerde lijn (boarding-reclame, schaduwrand). Nieuw idee: de gebruiker wijst in het eerste frame **twee punten aan die in werkelijkheid horizontaal van elkaar staan** (bv. twee markeringen op de boarding); die twee punten worden door de hele video getrackt en de hoek van hun verbindingslijn ís per frame de camerakanteling.

### Interactie

- Derde horizon-modus naast "vast" en "automatisch per frame": **"track twee punten"**. De bestaande `HorizonKiezer`-dialoog wordt uitgebreid: de gebruiker klikt twee punten (zoals nu al een lijn getekend wordt), maar kiest nu "volg deze punten door de video".
- Kies-tips in de dialoog: punten op **stilstaande, contrastrijke** details (boardingrand, lijnovergang, pilaar) die de hele video in beeld blijven — niet op ijs (spiegelend) of op personen.

### Techniek

1. **Tracking**: sparse Lucas–Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) per punt, met een **forward-backward-check** (punt terug-tracken; wijkt de terugreis > 1–2 px af → frame als onbetrouwbaar markeren). LK is subpixel-nauwkeurig en goedkoop (verwaarloosbaar naast de pose-detectie).
2. **Fallback per punt**: faalt LK (occlusie — bv. de schaatser schuift vóór het punt langs), dan template-matching (`cv2.matchTemplate`) in een zoekvenster rond de voorspelde positie; lukt ook dat niet → frame overslaan en later interpoleren.
3. **Hoekreeks**: per frame `atan2(dy, dx)` van de twee getrackte punten, minus de hoek in het referentieframe (de aangeklikte stand = per definitie 0°... nee: = de wáre horizontaal, dus de gemeten hoek is direct de kanteling). Daarna dezelfde opschoning die er al is: Hampel-uitschieters + Savitzky–Golay (`bepaal_horizon_reeks`-machinerie hergebruiken), resultaat per frame in `r.horizon_deg` — de rest van de pijplijn (aftrek in `bereken_hoek_tov_ijs`, meekantelende ijslijn in de overlay) werkt dan ongewijzigd.
4. **Punt raakt uit beeld** (pannende camera): detecteren wanneer een punt de framerand nadert en dan **overdragen op verse ankerpunten** — `cv2.goodFeaturesToTrack` in dezelfde beeldband zoekt nieuwe contrastrijke punten, die de op dat moment geldende hoekcalibratie erven. Zo blijft de meting doorlopen zonder dat de gebruiker opnieuw hoeft te klikken. Dit is de lastigste stap; eerste versie mag hem weglaten en gewoon waarschuwen + de laatste hoek vasthouden.
5. Als aparte, snelle video-pass geïntegreerd in `fase_voortgang()` (zoals de bestaande auto-horizon-pass), in beide backends.

**Klaar wanneer:** op een testvideo met zichtbaar schommelende camera geeft de getrackte-puntenmodus een vloeiende, geloofwaardige `horizon_deg`-reeks (witte ijslijn in de overlay blijft op de echte ijsrand liggen), ook wanneer de schaatser één van de punten kort passeert.

---

## Fase 6 — Sneller analyseren (meer uit de CPU/iGPU halen)

**Doel:** de YOLO-analyse (nu ~2 s/frame op CPU met yolo26x-pose op 1280) fors versnellen. Hardware hier: **Ryzen 7 7735U** (8 cores/16 threads) met **geïntegreerde Radeon 680M** — geen NVIDIA, dus geen CUDA; de realistische route is geoptimaliseerde CPU-inference en eventueel de iGPU via DirectML.

In oplopende moeite, cumulatief te stapelen — na elke stap meten met een vaste testvideo (zie meetprotocol hieronder):

1. **Batch-inference in de verfijningspass** (`_verfijn_landmarks`): de crops worden nu één voor één door `model.predict()` gehaald; ultralytics accepteert een lijst beelden. Crops verzamelen en in batches van bv. 8–16 voorspellen → minder overhead per frame, betere corebenutting. Weinig code, geen kwaliteitsverlies.
2. **Prefetch-thread voor het videolezen**: `cv2.VideoCapture.read()` + resize in een aparte thread met een kleine queue, zodat decoderen en inference elkaar overlappen i.p.v. afwisselen. Geldt voor alle passes (detectie, verfijning, auto-horizon).
3. **Lichter model voor de detectiepass, x voor de verfijning**: pass 1 hoeft alleen bboxes/track-IDs en globale keypoints te leveren; de nauwkeurige hoeken komen uit de crop-pass. `yolo26m-pose` (of zelfs `s`) op 1280 voor pass 1 + `yolo26x-pose` voor de crops kan een flink deel van de looptijd schelen. **Wel valideren** dat pass 1 de verre/bewegingsonscherpe schaatser nog vindt (dat was de reden voor 1280 × x) — op de testvideo controleren dat de dekking 100% blijft en de events identiek.
4. **Geëxporteerd model i.p.v. PyTorch**: `model.export(format=...)` van ultralytics en dan inferen met:
   - **OpenVINO** (`format="openvino"`): geoptimaliseerde CPU-runtime, werkt ook op AMD-CPU's; typisch 1.5–3× sneller dan torch-CPU, zelfde gewichten dus zelfde output (kleine numerieke afwijkingen).
   - **ONNX Runtime + DirectML** (`format="onnx"`, `onnxruntime-directml`): draait op de Radeon-iGPU. Potentieel de grootste sprong, maar iGPU-drivers/DirectML zijn de wisselvalligste van dit lijstje — als experiment plannen, met CPU-pad als terugval.
   Beide passen in `schaats_yolo.py` achter een klein abstractielaagje rond `model.track`/`model.predict`; ByteTrack-tracking blijft via ultralytics werken met een geëxporteerd model.
5. **GUI-keuze "snel / nauwkeurig"**: instelbaar profiel op de startpagina (snel = m-model + kleinere `DETECT_IMGSZ`; nauwkeurig = huidige instellingen). De gebruiker kiest per video of het om een snelle indruk of een precieze meting gaat.
6. **Quick wins checken** (kost bijna niets): `torch.set_num_threads(16)` expliciet zetten (torch pakt soms alleen de fysieke cores), OpenCV's threading niet laten concurreren tijdens inference (`cv2.setNumThreads(2)` tijdens de YOLO-pass), en laptop aan de lader + Windows-energiemodus "beste prestaties" (een U-chip throttlet fors op accu).

**Meetprotocol**: één vaste testvideo ("Schaats frontaal.MOV"), per stap noteren: totale analysetijd, pose-dekking (%), en of de afzet-events (aantal, been-volgorde, hoeken ±1°) gelijk blijven aan de referentie-run. Versnelling die de meting verandert is geen versnelling.

**Verwachting**: stappen 1+2+6 samen grofweg 1.5–2×; stap 3 nog eens ~2× op de detectiepass; stap 4 daar bovenop 1.5–3×. Ergens tussen "half uur per video" en "paar minuten per video" moet haalbaar zijn.

**Klaar wanneer:** de totale analysetijd van de testvideo minstens gehalveerd is zónder verlies van dekking of meetkwaliteit, en de snelste acceptabele configuratie als default staat.

---

## Fase 7 — Perspectiefcorrectie via baanlijnen

> **Nice-to-have (niet nu).** Sinds juli 2026 is de aanname dat er **altijd recht van voren** wordt gefilmd met een **horizontale** camera. Onder die opstelling kijkt de camera nagenoeg loodrecht op het bewegingsvlak en is de perspectiefvertekening klein, dus de dagelijkse workflow heeft deze correctie niet nodig. De bouwsteen (`schaats_perspectief.py` + zelftest) staat er al en blijft opt-in beschikbaar; volledige integratie is bewaard voor een eventuele latere opstelling met een schuin geplaatste camera. **Nu geen prioriteit.**

**Probleem:** de afzet- en kniehoek worden nu gemeten in het **beeldvlak** — de 2D-projectie van het been. Dat klopt alleen als de camera loodrecht op het bewegingsvlak van het been kijkt. Staat de camera niet midden in de baan (of komt de schaatser niet recht op de camera af), dan kijk je onder een schuine hoek en verkort het perspectief het been in één richting: de gemeten hoek wijkt structureel af van de echte, en — verraderlijker — de afwijking **verandert met de positie van de schaatser in beeld**. Dezelfde afzet lijkt dan aan het begin van de passage een andere hoek te hebben dan aan het eind. De bestaande horizoncorrectie repareert alleen camerarotatie om de kijkas (roll), niet deze vertekening.

**Kernidee:** de lijnen in het ijs zijn rechte, evenwijdige lijnen met bekende onderlinge afstand (standaard baanbreedte ± 4 m). Daaruit is de camerastand t.o.v. het ijsvlak te kalibreren, en met die kalibratie kunnen de hoeken per frame worden teruggerekend naar het echte, onvertekende vlak.

### Stappen

1. **Lijnen aanwijzen (interactie).** Uitbreiding van de bestaande `HorizonKiezer`-dialoog: de gebruiker trekt op één frame twee (of meer) baanlijnen na die in werkelijkheid evenwijdig lopen in de rijrichting, plus liefst één dwarslijn (start-/finishlijn, bochtmarkering). Optioneel geassisteerd met Hough-detectie (die machinerie bestaat al in `detecteer_ijslijn()`), maar handmatig natrekken is de betrouwbare basis — lijnen op het ijs zijn contrastarm en deels bekrast.
2. **Kalibratie uit de lijnen.** Evenwijdige lijnen snijden in beeld in een verdwijnpunt; de rijrichting-lijnen geven verdwijnpunt V1, de dwarslijn(en) V2. De lijn V1–V2 is de **verdwijnlijn van het ijsvlak = de ware horizon** (bijvangst: dit vervangt de Hough-horizonhack door iets principiëlers). Met de gebruikelijke aannames (principal point in het beeldmidden, vierkante pixels) volgt uit twee orthogonale verdwijnpunten een schatting van de brandpuntsafstand, en daarmee de volledige **homografie ijsvlak ↔ wereldvlak**. De bekende baanbreedte zet er schaal (meters) op.
3. **Hoekcorrectie.** De enkel staat (vrijwel) op het ijs → via de homografie is zijn wereldpositie bekend. De knie is een kijkstraal vanuit de camera; om die in 3D te prikken is één extra aanname nodig. Twee kandidaten, te kiezen na experiment:
   - **(a) Constante onderbeenlengte**: de afstand enkel–knie is per schaatser vast. Kalibreer die lengte op frames waar het been vrijwel loodrecht op de kijkrichting staat (daar is de projectie onvertekend), en snijd daarna per frame de knie-kijkstraal met de bol rond de enkel met die straal. Geometrisch het zuiverst.
   - **(b) Beenvlak-aanname**: neem aan dat het onderbeen in een verticaal vlak ligt met bekende oriëntatie (bv. de rijrichting uit het getrackte traject, of dwars daarop tijdens de zijwaartse afzet). Eenvoudiger, maar de aanname is bij een schaatsafzet (schuin zijwaarts-achterwaarts) discutabel — daarom eerst op testmateriaal verifiëren welke variant stabieler is.
   Uit de gereconstrueerde 3D-punten volgt de echte hoek t.o.v. het ijsvlak; die vervangt de beeldvlak-hoek in `bereken_hoek_tov_ijs()`-verband (zelfde plek in de pijplijn als de huidige horizonaftrek, dus stap 2/3 van de kern blijven ongewijzigd).
4. **Kwaliteitsindicator.** De correctie is groot en gevoelig wanneer de schaatser ver van de camera-as zit; toon per frame (HUD) en per afzet-event (tabel) hoe groot de toegepaste correctie was, en markeer metingen waar de geometrie onbetrouwbaar wordt (been bijna in de kijkrichting — dan is géén enkele correctie nog te redden).
5. **Bijvangst (gratis erbij):** met een metrische ijsvlak-homografie is de positie van de schaatser op de baan per frame bekend → echte **snelheid (m/s)** en **slaglengte per afzet** in de tabel.
6. **Camerabeweging.** Eerste versie: alleen **vaste camera (statief)** — één kalibratie voor de hele video. Voor een pannende/schommelende camera moet de homografie per frame meebewegen: de aangewezen lijnen tracken met dezelfde LK-optical-flow-machinerie als fase 5. Dat is een logisch vervolg, geen onderdeel van de eerste versie.

**Klaar wanneer:** dezelfde schaatser die op verschillende plekken in beeld passeert (dichtbij/veraf, links/rechts) krijgt ná correctie een stabiele afzethoek (± 2°), waar de ongecorrigeerde meting zichtbaar met de beeldpositie verloopt. Testopname: één schaatser, meerdere rondjes langs dezelfde vaste camera, hoeken per passage vergelijken.

---

## Volgorde & omvang (grove inschatting)

| Fase | Wat | Omvang |
|---|---|---|
| 0 | Serialisatie (`npz` + plain landmarks) | ✅ **af** (16 jul 2026) |
| 1 | `schaats_db.py` + bibliotheek-GUI + nieuwe-analyse-flow | ✅ **af** (18 jul 2026) |
| 2 | Voortgangsgrafiek, notities, export | klein, 1 sessie |
| — | Batch-analyse (extra, buiten de fasering) | ✅ **af** (20 jul 2026) |
| — | Vergelijk schaatsers + `VideoSpeler`-refactor (extra) | ✅ **af** (27 jul 2026) |
| 3 | Skelet-editor met uitvloeien + undo | ✅ **af** (20 jul 2026) |
| 4 | Instellingen, gedeelde map, conflictafhandeling | ✅ **af** (20 jul 2026) |
| 5 | Horizon via twee getrackte punten | *nice-to-have (niet nu — horizontale camera)*; middelgroot, 1–2 sessies (stap 4, punt-overdracht, is het meeste werk) |
| 6 | Sneller analyseren | gefaseerd: stappen 1+2+6 in 1 sessie; export/DirectML apart experiment |
| 7 | Perspectiefcorrectie via baanlijnen | *nice-to-have (niet nu — frontale, horizontale camera)*; groot, 2–3 sessies (stap 3, de 3D-reconstructie, is onderzoekswerk — eerst valideren op testmateriaal) |

## Openstaande vragen (beslissen wanneer de fase begint)

- ~~**Fase 1**: video altijd kopiëren?~~ **Besloten (jul 2026): altijd kopiëren, origineel laten staan.**
- ~~**Fase 1**: oude "losse" analyses importeerbaar?~~ **Besloten (jul 2026): niet nodig.**
- **Fase 3**: ook punten kunnen bewerken op frames zónder gedetecteerde pose (punt "plaatsen" i.p.v. verslepen)? Eerste versie: nee, alleen bestaande punten verslepen.
- ~~**Fase 4**: welke cloudprovider gebruikt het team feitelijk?~~ **Besloten (jul 2026): Google Drive** (Mirror-modus, dus alle bestanden lokaal op schijf). Conflictdetectie is provider-agnostisch (elk `*.db` naast `schaats.db`).
- **Fase 5** *(nice-to-have, niet nu)*: onder de huidige aanname (horizontale camera) is deze fase niet nodig. Wordt pas relevant als er tóch met een schuine/schommelende camera gefilmd gaat worden; dán ook: pant de camera mee (punt-overdracht nodig) of staat hij op statief?
- **Fase 6**: hoeveel meetafwijking is acceptabel voor het "snel"-profiel? (Voorstel: events moeten identiek blijven, hoeken mogen ±1° verschillen.)
- **Fase 7** *(nice-to-have, niet nu)*: onder de huidige aanname (frontaal, horizontaal) is de perspectiefvertekening klein en deze fase geen prioriteit. Wordt pas relevant bij een schuin geplaatste camera; dán ook: welke baanlijnen zijn scherp genoeg om na te trekken en is hun onderlinge afstand bekend (schaal in meters — zonder schaal werkt de hoekcorrectie ook, alleen snelheid/slaglengte niet)? En staat de camera dan op statief, of moet het lijn-tracken uit fase 5 mee?
