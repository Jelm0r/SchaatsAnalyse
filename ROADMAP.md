# Roadmap SchaatsAnalyse

Plan voor de volgende ontwikkelfase, in volgorde van bouwen. Gemaakte keuzes (juli 2026):

- **Delen**: één gedeelde cloudmap (OneDrive/Dropbox/netwerkschijf) met daarin een SQLite-bestand + mediabestanden. Geen server, geen accounts.
- **Video's**: worden mee-gekopieerd naar de gedeelde map, zodat elke trainer de analyse mét beeld kan terugkijken.
- **Schaal**: één team (±5–30 schaatsers, 1–5 trainers). Ontwerp mag daarop leunen; geen rechten-/rollensysteem nodig.
- **Skelet-editor**: correcties vloeien uit naar buurframes met een **instelbaar venster** (0 = alleen het bewerkte frame).
- **Opnameopstelling (aanname sinds juli 2026)**: er wordt **altijd recht van voren** gefilmd en de camera staat **altijd precies horizontaal**. Daardoor vervalt de noodzaak van camerakanteling-correctie (horizon) en perspectiefcorrectie in de dagelijkse workflow. Fases 5 en 7 blijven staan als **nice-to-have** voor eventuele latere opstellingen (schuine/schommelende camera), maar zijn **nu geen prioriteit**.

De fases bouwen op elkaar: 0 → 1 → 2 kunnen niet van volgorde wisselen; 3 (skelet-editor) en 4 (delen) zijn daarna onafhankelijk van elkaar te bouwen. Fase 6 (sneller analyseren) staat los van de rest en kan op elk moment, ook eerder. Fase 5 (horizon-tracking) en 7 (perspectiefcorrectie) zijn onder de vaste opnameopstelling **nice-to-have** (zie hierboven); fase 7 deelt bouwstenen met fase 5 (lijnen aanwijzen/tracken) en omvat de horizoncorrectie als speciaal geval. **Fase 8** (fragmenten knippen in de app + de opnames zelf in de bibliotheek, aangevraagd 10 augustus 2026) stond eveneens los en is **af sinds 11 augustus 2026**; hij leunt aan de analysekant volledig op de bestaande batch-flow uit fase 1 en voegde daar één schemabump aan toe (`bronvideo`, v3) voor de werklijst van nog niet geknipte opnames. Daarmee is de **eerstvolgende** fase weer fase 6 (sneller analyseren) of fase 2 (profielweergave, zodra de meting af is).

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
> - ~~De perspectiefkalibratie wordt (net als in fase 0) niet geserialiseerd~~ — **achterhaald sinds 11 augustus 2026**: de kalibratie-invoer (de nagetrokken lijnen + parameters) gaat mee in `instellingen_json` en de camerastand wordt bij het openen herberekend. Zie fase 7 hieronder. Analyses van vóór die datum krijgen nog wel de oude melding.
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

> **Wachten op de meting (besloten 8 augustus 2026).** Deze fase gaat over het *duiden* van de schaatstechniek: voortgang over tijd, notities, export. Dat heeft pas waarde als de onderliggende meting klopt — een voortgangsgrafiek van hoeken die nog verschuiven bij elke verbetering aan de tracking is misleidend, en oude analyses zouden er anders in staan dan nieuwe. Fase 2 wordt daarom **pas opgepakt als de analyse af is**; het is technisch een kleine klus, maar niet de volgende.

Klein maar waardevol vervolg op fase 1 (kan ook later):

- **Voortgang over tijd**: grafiekje per schaatser met de gemiddelde/beste afzethoek per analyse-datum (data zit al in `afzet_event_cache`).
- Notitieveld per analyse ("linkerbocht geoefend, wind tegen").
- CSV-export per schaatser (alle analyses) naast de bestaande per-analyse-export.

---

## Extra — Appversie per analyse + info-tabje ✅

> **Af (6 augustus 2026)** — zelftest (`python schaats_db.py`) groen in beide venvs, versie handmatig getoetst tegen `git log -1`, en een headless rooktest van de dialoog + de bibliotheek-tooltip (ook met een analyse van vóór deze functie).
>
> **Doel:** tijdens het ontwikkelen wijzigt de trackinglogica regelmatig; van een opgeslagen analyse moet je achteraf kunnen zien met welke versie van de app hij is gemaakt, zodat een vreemde meting te verklaren is ("dit is met de oude L/R-fixer gedaan").
>
> **Wat er staat:** `schaats_db.app_versie()` leest de git-repo naast het script (`git log -1 --abbrev=8 --format=%h%x09%cs` + `git status --porcelain -uno`) en levert `{commit, datum, vuil, label}`, één keer per proces gecacht; buiten een repo is alles leeg. `sla_analyse_op` zet `app_versie`, `app_commit` én de volledige `backend_naam` zelf in `instellingen_json` (`setdefault`, dus een caller die het invult wint) — één plek, dus enkele analyse, batch en zelftest leggen het alle drie vast zonder eraan te denken. Nieuw in de GUI: knop **"ℹ Info..."** op twee plekken — in de transportbalk naast "Bewerken"/"⇄ Vergelijk met..." (geopende analyse) en op de startpagina naast "Openen" (geselecteerde rij, dus zonder de analyse te hoeven openen) → `AnalyseInfoDialog` met titel, schaatser, analysedatum, maker, appversie, backend, videoformaat, of er handmatig bewerkt is, en de instellingen (smoothing, drempel, bocht overslaan, horizon, perspectief; de heavy-model-rij alleen bij een MediaPipe-analyse, want de YOLO-backend heeft één model en negeert die vlag). De waarden zijn selecteerbaar zodat de hash te kopiëren is. De bibliotheeklijst toont de versie als tweede regel in de bestaande titel-tooltip.
>
> **Keuzes:** de openstaande vraag hash-vs-label is **allebei geworden, maar allebei automatisch**: het label is `commitdatum · korte hash` (`2026-08-05 · 7e013fb7`), met een `+` als er ongecommitte wijzigingen waren. De commitdatum is het leesbare deel voor een trainer, de hash het precieze deel om `git show` op te doen. Een handmatig opgehoogde `APP_VERSIE`-constante is bewust geschrapt: die loopt juist tijdens snel ontwikkelen achter en liegt dan. Untracked bestanden tellen niet als "vuil" — video's en npz's naast de code zeggen niets over de gedraaide logica. Verder kreeg `schaats_db` een `analyse_meta()` (DB-rij + geparste instellingen, **zonder** het npz te lezen) en levert `lijst_analyses` de instellingen mee; de Info-knop haalt zijn meta daar vers op in plaats van een kopie op `MainWindow` te laten leven.
>
> **Bewust niet:** geen schemabump en geen eigen kolom — dit hangt in het bestaande JSON-veld. Analyses van vóór deze wijziging krijgen dus met terugwerkende kracht geen versie; die tonen "onbekend (van vóór deze functie)" en houden in de bibliotheek precies hun oude tooltip. De versie wordt bij een handmatige skelet-edit níet bijgewerkt: hij zegt waarmee de analyse *gedraaid* is (dat een analyse bewerkt is, staat apart in de dialoog).

---

## Extra — Bibliotheeklijst: videoduur i.p.v. afzetten/hoek ✅

> **Af (6 augustus 2026)** — headless rooktest van de bibliotheekpagina in de YOLO-venv (kolommen, rij-knoppen, rijhoogte) + `python schaats_db.py` groen.
>
> Uitgevoerd zoals hieronder beschreven, met twee toevoegingen die uit het gebruik kwamen:
> - **Duur én afzetten in één kolom**: `"5,4s (4 afzetten)"` (boven de minuut `"1:23 (12 afzetten)"` — "83,2s" leest niemand als anderhalve minuut). Het aantal afzetten hoefde dus niet weg; het staat alleen niet meer op de plek van de eerste blik. Formattering in `_duur_tekst()` (`schaats_gui.py`); `lijst_analyses()` levert er alleen `totaal_frames`+`fps` extra voor aan.
> - **De per-analyse knoppen verhuizen naar de rij zelf** (`_maak_rij_knoppen` → `setCellWidget` in de vierde kolom): "Openen", "ℹ Info...", "Hernoemen...", "Verwijderen" horen bij één analyse, terwijl "Nieuwe analyse..."/"Batch-analyse..."/"Vergelijk schaatsers..." bibliotheek-breed zijn — die stonden onderin op één rij door elkaar. Elke knop draagt zijn eigen analyse-id mee (default-argument in de lambda, anders krijgt elke rij de laatste lus-waarde), dus een klik werkt op zijn eigen rij en niet op de toevallige tabelselectie. `_hernoem_analyse`/`_verwijder_analyse` kregen daarvoor `(aid, titel)`-argumenten met de oude selectie-route als terugval. Bijvangst: de knoppenrij onderin ging van zeven naar drie knoppen, waarmee het **venster-minimum van 1738 → 1406 px** breed zakt (gemeten headless).

**Doel:** de analysetabel op de startpagina (`tabel_analyses`) toont nu "aantal afzetten" en "gem. hoek" per rij. Die twee kolommen zijn niet waar de trainer op eerste oogopslag naar kijkt; bruikbaarder is **hoe lang de video duurt** (seconden), en de gemiddelde hoek mag helemaal weg.

**Aanpak:**

- **Duur i.p.v. aantal afzetten**: de duur (`totaal_frames / fps`) staat al in de `analyse`-tabel (fase 1-schema, kolommen `totaal_frames`+`fps`), dus dit is een pure weergavewijziging in `lijst_analyses()` (`schaats_db.py`) + de kolomopbouw in `schaats_gui.py` — geen schemabump, geen nieuwe berekening. Formatteren als `m:ss` (of `s` bij korte clips).
- **Gem. hoek-kolom weghalen**: kolom uit `tabel_analyses` schrappen. De onderliggende berekening (`AVG(hoek)` met de `ONVOLLEDIG_MARKERS`-filtering in `lijst_analyses`) mag blijven bestaan voor fase 2's voortgangsgrafiek — alleen de kolom in déze tabel verdwijnt.
- **Aantal afzetten**: blijft eventueel bruikbaar elders (bv. als tooltip), maar is als kolom niet meer nodig zodra duur er staat — te beslissen of hij helemaal weg mag of blijft staan naast de duur.

**Klaar wanneer:** de bibliotheektabel toont per analyse titel/datum/duur (en evt. aantal afzetten), zonder de gemiddelde-hoek-kolom.

---

## Extra — Bochtdetectie: de bocht wordt niet meer geanalyseerd ✅

> **Af (5 augustus 2026)** — buiten de fasering. De bocht kostte analysetijd zonder ooit een bruikbare meting op te leveren; dat is nu beide opgelost.
>
> **Wat het signaal is:** `bocht_ratio` in `schaats_analyse.py` = **heupbreedte / romplengte** (schoudermidden→heupmidden), in pixels. Schaalvrij, net als `_strek_ratio` — en juist gevoelig voor de rotatie om de verticale as die de bocht maakt: frontaal staan de heupen naast elkaar, in de bocht achter elkaar terwijl de romp even lang blijft. **Gemeten over alle 22 analyses in de bibliotheek:** 18 frontale clips komen nooit onder 0,57 (mediaan 0,75–1,20); het bochtdeel van vier lange clips zit op 0,21–0,24. Marge ruim 3×. Vier alternatieve noemers (femur, heel been, schouderbreedte, combinaties) gaven allemaal minder scheiding (1,7–2,7×). De classificatie (`bepaal_bocht_reeks`, recept van `bepaal_horizon_reeks`: Hampel → Savitzky–Golay → hysterese 0,40 in / 0,50 uit → runs < 0,6 s weg) markeert op die 18 frontale clips **nul** frames als bocht.
>
> **Waar de tijdwinst zit:** `_BochtWacht` in `schaats_yolo.py`. De detectiepass is ~94% van de analysetijd, dus die moest de bocht overslaan — en dat kon niet met `model.track(source=pad, stream=True)`, want ultralytics leest en infereert daar zelf elk frame. De lus leest de frames nu zelf (decoderen is verwaarloosbaar, en zo blijft de framenummering exact) en infereert in de bocht nog maar elke ~0,3 s, precies zoals voorgesteld. De wacht gaat overslaan na 0,5 s bochtbewijs (iemand in beeld, maar gedraaid) of 3 s zonder enige meetbare persoon — die 3 s ligt bewust boven het langste detectiegat op een recht stuk in de bibliotheek (2,1 s, IMG_9001). Omstanders langs de boarding staan frontaal in beeld en zouden de analyse eeuwig op vol tempo houden; daarom telt alleen een persoon die beweegt **of groeit** (een schaatser die recht op de camera af komt verplaatst in beeld nauwelijks maar wordt ~18%/s groter).
>
> **Waarom het overslaan veilig is:** de verfijningspass vult detectiegaten tot `GAP_VUL_S` (1,0 s) met geïnterpoleerde bboxes en schat de pose daar alsnog top-down. De gaten die het overslaan achterlaat zijn 0,33 s, dus ruim daarbinnen. De bocht wordt daarom **bepaald op de ruwe pass-1-landmarks, vóór de verfijning**: heeft de wacht een stuk ónterecht overgeslagen, dan meldt het eerstvolgende controleframe binnen 0,33 s een frontale schaatser en draait de detectiepass meteen weer op vol tempo. Te veel overslaan kost dus (bijna) geen dekking, te weinig overslaan alleen tijd.
>
> **Een controleframe telt niet als meting.** Het frame dat in de overslaan-stand nog wél geïnfereerd wordt, krijgt een skelet — maar het staat midden in een stuk dat verder niet bekeken is, dus de buurframes die een afzet moeten aantonen ontbreken. `_detecteer_alles` levert die frames daarom mee in `buiten_meting`, en `_bocht_met_controleframes` houdt ze na elke classificatie op `bocht=True`. Hun óórdeel telt wél (ze mogen de bocht beëindigen — daarvoor zijn ze er), hun eigen hoek niet. Zonder die regel zou zo'n frame zichzelf op z'n eigen heupstand kunnen vrijpleiten: midden in een bocht draait een schaatser af en toe kort bijna frontaal.
>
> **Gemeten (5 aug 2026):**
>
> | | | |
> |---|---|---|
> | **"Kim tempo"** (888 frames, 52% bocht) | 2266 s → **1253 s** | **45% sneller** (1,8×) |
> | ... afzetten | 32 → 16 | de vijf hoeken van 74–87° aan het eind (de bocht) zijn weg |
> | ... rechte stuk (frame 0–427) | landmarks **identiek** (mediaan 0,00 px) | alleen de laatste 7 frames vóór de bocht wijken af |
> | **"Schaats frontaal"** (kruisende schaatsers) | bocht aan vs. uit: **byte-identiek** | 0 frames als bocht gemarkeerd |
> | **"7e ronde"** (staand telefoonbeeld) | 0 bocht, dekking 100%, zelfde 5 afzetten | geen rotatieprobleem door de eigen leeslus |
> | **De eigen leeslus zelf** | met bocht uit: **byte-identiek aan de opgeslagen analyse** | `model.track(source=...)` vervangen verandert niets |
>
> De hoekverschillen die op het rechte stuk van Kim tempo overblijven (tot 3,8°, één afzet minder) komen **niet** uit de detectie maar uit de meetlogica: de geschatte slagperiode (`STREK_MIN_SLAG_FRAC`) en de L/R-alternatiecontrole liepen voorheen mede over bochtruis. Zet je de bochtvlag op de ópgeslagen landmarks, dan komt er exact dezelfde eventlijst uit — dus dit is winst, geen afwijking.
>
> **Wat "bocht" betekent voor de meting:** één regel in `verwerk_afgeleiden` — een bochtframe krijgt geen `lm_data`. Been-toewijzing, afzet-voltooiing en event-segmentatie bouwen hun segmenten allemaal op "pose én lm_data", dus zij zien de bocht vanzelf als een detectiegat; aan de meetlogica is niets veranderd. Het skelet blijft wél getekend, met "BOCHT — niet gemeten" in beeld. Dat markeren i.p.v. hard afkappen is nodig omdat sommige clips juist ín de bocht beginnen (laatste slagen van de vorige ronde) en omdat een video met meerdere rondjes zo elk recht stuk blijft opleveren.
>
> **Bestaande analyses** veranderen niet vanzelf: hun npz kent de vlag niet en bij het openen wordt die niet alsnog berekend. Wil je zo'n analyse tóch schoon, dan doet de knop **"Bocht bepalen"** dat zonder opnieuw te analyseren — de landmarks van de hele clip staan er immers al in. Hij laat eerst zien wat het met de tabel doet en vraagt dan pas; op "Kim tempo" is dat 32 → 16 afzetten in een fractie van een seconde i.p.v. 21 minuten. Bewust een knop en geen automatisme: het verandert wat de trainer eerder gezien heeft.
>
> **Verder:** `FrameResultaat.bocht` gaat mee in het npz (oude npz's laden ongewijzigd — de runtime-skip valt niet uit landmarks te herleiden, dus die moet bewaard); checkbox **"Bocht overslaan (sneller)"** (standaard aan) in beide analyse-dialogen; de dekkingsteller telt bochtframes niet als openstaand werk en "⏭ Volgend gat" springt er niet in; `python schaats_eval.py bocht analyse.npz` print het signaal + de gevonden segmenten. CLI: `--no-bocht`.
>
> **Nog open:** de drempels zijn geijkt op vier TV-clips die in de bocht eindigen. Er is nog **geen clip uit de eigen opstelling die ín de bocht begint** — zodra die er is, met `schaats_eval.py bocht` narekenen en `BOCHT_IN`/`BOCHT_UIT` zo nodig bijstellen. `BOCHT_MIN_BEWEGING` (de omstander-filter) is het meest waarschijnlijke tweede afstelpunt.

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
> **Eén snelheid voor beide kanten:** de snelheidsregelaar per kant is weg (`VideoSpeler(toon_snelheid=False)`); de gedeelde regelaar onderaan de pagina stuurt nu ook het los afspelen van een kant. Twee video's naast elkaar op verschillend tempo laten lopen is precies wat je bij vergelijken níet wilt, en de per-kant combo nodigde daar wel toe uit.
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
> - Punten *plaatsen* op frames zónder pose: niet in de eerste versie — **alsnog gebouwd op 31 juli 2026**, zie hieronder.

> **Aanvulling: skelet plaatsen op een frame zonder pose (31 juli 2026)** — headless rooktest (klikreeks, undo/redo, annuleren, wegnavigeren) + `python schaats_db.py` groen.
>
> **Waarom:** een frame zonder pose breekt in `bepaal_afzet_uit_strek` de stand-run af en levert `ONV_AFGEKAPT` — twee ontbrekende frames kosten zo een hele afzetmeting. Kan de trainer het gat met de hand dichten, dan komt de meting terug. Dat is de eigenlijke opbrengst; het skelet zelf is maar het middel.
>
> **Wat er staat:** in de bewerk-modus is op een gat-frame **"➕ Maak skelet"** actief. Normaal gesproken wordt het skelet dan **overgenomen van de buurframes** (`maak_voorvulling` in `schaats_analyse.py`: interpoleren bij een kort gat, kopiëren bij een lang gat) en corrigeer je het met de gewone sleep-editor — één manier van werken voor álle frames. Alleen als er níets over te nemen valt (nergens in de analyse een pose) is er ook niets te verslepen; dan vraagt het programma de punten één voor één in vaste volgorde (schouders, heupen, knieën, enkels — `PLAATS_VOLGORDE`, 8 klikken) met het gevraagde punt als oranje ring-met-kruisdraad in beeld. Zoomen en pannen blijven beschikbaar — muiswiel zoomt en **rechts-slepen pant** (nieuw in `VideoSpeler`, ook op de vergelijkpagina). Statusbalk-teller **"Skelet: 123 van 126 frames"** plus een knop **"⏭ Volgend gat"**. Undo/redo maakt een geplaatst skelet in één stap ongedaan. Zie CLAUDE.md voor de details.
>
> **Keuzes:** slepen boven klikken — een klikreeks van acht namen lezen is bewerkelijker dan een skelet bijstellen dat er al ongeveer goed staat, en het houdt de bediening gelijk aan die van elk ander frame. 8 punten en niet 33 in de klikreeks — precies wat de metingen gebruiken plus de torso waar grijpradius en auto-zoom op rekenen; hiel/teen blijven op visibility 0, net als bij de YOLO-backend zonder RTMPose. Een klikreeks vastleggen kan pas als heup/knie/enkel van beide benen staan (anders komt er een 0°-hoek in de tabel), en wegnavigeren zonder ook maar één klik annuleert i.p.v. de voorvulling stilzwijgend als meting vast te leggen.
>
> **Bewust niet:** geen "vul het hele gat in één keer"-knop (interpoleren over een gat is precies wat de detector al niet kon); geen persistente markering welk frame handmatig is (de groene ringen zijn per sessie).

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

> **Voorwerk: gemeten op de doelmachine (19 juli 2026).** Er is een profileringssessie geweest op "Schaats frontaal.MOV" (1920×1080, torch 2.12.1+**cpu**, onnxruntime CPU-only). De uitkomsten staan hieronder omdat ze de volgorde van deze fase omkeren; ze zijn nooit in code omgezet, dus fase 6 is nog volledig open.
>
> **Waar de tijd zit:** detectiepass yolo11x-pose @1280 = **~2333 ms/frame = 94% van de rekentijd**; RTMPose-verfijning ~140 ms/bbox (alleen op doelframes). Alles wat niet de detectiepass is, valt in de ruis.
>
> **Doodlopend: meer hardware inzetten.** Beide netten zijn **geheugenbandbreedte-gebonden**, niet rekengebonden. Gemeten: yolo11x@1280 is **vlak van 1 → 16 threads** (~1950 ms/frame — meer threads doet niets); twee processen tegelijk kosten elk ~3080 ms (samen dus maar **1,3×**); vier processen elk ~7000 ms (langzamer dan serieel). RTMPose gedraagt zich hetzelfde (~1,3× bij twee processen). Multiprocessing, threading en het opschroeven van thread-instellingen leveren dus **hooguit ~1,3×** — dat is de reden dat stap 6 hieronder van "quick win" naar "waarschijnlijk zinloos" is verplaatst.
>
> **De lever: het detectiemodel verkleinen.** Gemeten per frame @1280: `yolo11m` = 810 ms (**2,9×** sneller dan x), `yolo11n` = 162 ms (**14×**, maar 5 van 6 detecties). Op de hele pijplijn is dat ruwweg 2,6× (m) tot 8× (n). Dit kan omdat de pass-1-keypoints **tóch worden overschreven** door de RTMPose-verfijning (`verfijnd.get(f)` wint; pass-1 `lm` is alleen terugval): de detectiepass hoeft alleen bbox + track-ID + torso-kleur + centroid te leveren. Het risico zit dus **niet** in hoekprecisie maar in **detectiedekking van de doelschaatser, tracking-robuustheid en het kleurhistogram** — precies wat `schaats_eval.py` meet. Verkleinen van `DETECT_IMGSZ` (1280 → 960) halveert de detectiekost ongeveer, maar raakt hetzelfde risico: 1280 was juist gekozen omdát 640 verre/bewegingsonscherpe schaatsers helemaal miste.
>
> **Sindsdien wél gebeurd, buiten deze lijst om:** de swap naar `yolo26x-pose` (21 jul 2026) gaf ~12% en kostte niets aan nauwkeurigheid (standbeen-hoekfout 1,34° vs 1,54° tegen de gouden referentie = gelijk binnen ruis), en de **bochtdetectie** (5 aug 2026) haalde 45% van de tijd weg op een clip die voor de helft bocht is. Dat laatste is feitelijk stap 2's "frames overslaan"-idee, toegepast op de plek waar het gratis was.
>
> **Niet gemeten, dus nog steeds schatting:** OpenVINO-export en DirectML op de iGPU (stap 4). Die getallen hieronder komen uit de literatuur, niet uit een test op deze machine.

In oplopende moeite, cumulatief te stapelen — na elke stap meten met een vaste testvideo (zie meetprotocol hieronder). **Volgorde na de meting van juli 2026: stap 3 eerst** — dat is de enige stap waar een groot getal onder ligt; 1 en 2 zijn randwerk op de 6% die níet de detectiepass is.

1. **Batch-inference in de verfijningspass** (`_verfijn_landmarks`): de crops worden nu één voor één door `model.predict()` gehaald; ultralytics accepteert een lijst beelden. Crops verzamelen en in batches van bv. 8–16 voorspellen → minder overhead per frame, betere corebenutting. Weinig code, geen kwaliteitsverlies. *Let op: de verfijning is maar ~6% van de looptijd, dus dit is hoogstens een paar procent op het geheel — en de "betere corebenutting" is bij een bandbreedte-gebonden net twijfelachtig.*
2. **Prefetch-thread voor het videolezen**: `cv2.VideoCapture.read()` + resize in een aparte thread met een kleine queue, zodat decoderen en inference elkaar overlappen i.p.v. afwisselen. Geldt voor alle passes (detectie, verfijning, auto-horizon). *Sinds de bochtdetectie (aug 2026) leest `_detecteer_alles` de frames zelf i.p.v. via `model.track(source=...)`, dus deze stap kan nu ook op de detectiepass.*

   **Aanpalende kans, gemeten bij de bochtdetectie:** de verfijningspass vult detectiegaten tot `GAP_VUL_S` (1,0 s) met geïnterpoleerde bboxes en herstelt daar de pose. Op het rechte stuk 1 op de N frames overslaan in pass 1 zou dus grotendeels door pass 2 opgevangen worden — de bocht-wacht doet precies dat al in de bocht. Alleen te doen mét het meetprotocol ernaast: hier gaat het wél om frames waarop gemeten wordt.
3. **Lichter model voor de detectiepass, x voor de verfijning** — ⭐ **begin hier**: pass 1 hoeft alleen bboxes/track-IDs en globale keypoints te leveren; de nauwkeurige hoeken komen uit de crop-pass. `yolo26m-pose` (of zelfs `s`) op 1280 voor pass 1 + RTMPose/`yolo26x-pose` voor de crops. **Gemeten voorspelling** (zie voorwerk): m ≈ 2,6×, n ≈ 8× op de hele pijplijn — verreweg het grootste getal in dit lijstje, en de enige stap die de 94% raakt. **Wel valideren** dat pass 1 de verre/bewegingsonscherpe schaatser nog vindt (dat was de reden voor 1280 × x): op de testvideo controleren dat de dekking 100% blijft, de events identiek zijn én de doelkeuze/stitching niet verslechtert (`yolo11n` miste in de meting 1 op 6 detecties — daar breekt eerst de tracking, niet de hoek). Het kleurhistogram hangt aan de bbox-kwaliteit, dus `_splits_op_kleur` en `_stik_keten` zijn de plekken waar een te klein model zich als eerste wreekt. Één A/B-run met de gouden referentie beslist dit.
4. **Geëxporteerd model i.p.v. PyTorch**: `model.export(format=...)` van ultralytics en dan inferen met:
   - **OpenVINO** (`format="openvino"`): geoptimaliseerde CPU-runtime, werkt ook op AMD-CPU's; typisch 1.5–3× sneller dan torch-CPU, zelfde gewichten dus zelfde output (kleine numerieke afwijkingen).
   - **ONNX Runtime + DirectML** (`format="onnx"`, `onnxruntime-directml`): draait op de Radeon-iGPU. Potentieel de grootste sprong, maar iGPU-drivers/DirectML zijn de wisselvalligste van dit lijstje — als experiment plannen, met CPU-pad als terugval.
   Beide passen in `schaats_yolo.py` achter een klein abstractielaagje rond `model.track`/`model.predict`; ByteTrack-tracking blijft via ultralytics werken met een geëxporteerd model.
5. **GUI-keuze "snel / nauwkeurig"**: instelbaar profiel op de startpagina (snel = m-model + kleinere `DETECT_IMGSZ`; nauwkeurig = huidige instellingen). De gebruiker kiest per video of het om een snelle indruk of een precieze meting gaat.
6. **Quick wins** — grotendeels achterhaald door de meting; wat er nog van over is, in aflopende zin:
   - **Energiemodus op "Beste prestaties"** (Instellingen → Systeem → Energie en batterij → Energiemodus, **niet** `powercfg`: op deze Windows-11-installatie bestaat er maar één schema, "Gebalanceerd", en de prestatiestand is een overlay uit de Instellingen-app). *Nagekeken 8 aug 2026: de machine stond op "Gebalanceerd" terwijl hij aan de lader hing.* De 7735U is een 15 W-chip met configureerbare TDP tot 28 W; een lager aanhoudend pakketvermogen drukt óók de fabric-/geheugencontrollerklok, en dát is precies de bottleneck. **De enige knop in deze stap waar realistisch 10–30% in kan zitten, en nooit gemeten.** Eén omzetting + één testrun.
   - **`cv2.setNumThreads(2)` tijdens de YOLO-pass.** De profilering van juli mat yolo *in isolatie*; in de echte pijplijn concurreren decoderen, resizen en het kleurhistogram om dezelfde cores. Twee regels, verwacht enkele procenten, geen risico voor de meting.
   - **Procesprioriteit boven normaal** voor de analyse-worker. Marginaal, gratis.
   - **Defender-uitsluiting op de bibliotheekmap.** Raakt de inference niet, wél het wegschrijven: `sla_analyse_op` kopieert de hele video en die wordt meegescand. Gevoelde wachttijd, geen analysetijd.
   - ~~`torch.set_num_threads(16)`~~ — **afgevoerd**: de threadschaling is vlak van 1 → 16, er valt geen corebenutting te winnen.

   **Twee doodlopende wegen, expliciet genoteerd zodat ze niet opnieuw onderzocht worden:**
   - **Geheugen upgraden kan niet en hoeft niet.** *Nagekeken 8 aug 2026:* 4× 4 GB **LPDDR5-6400 gesoldeerd op het moederbord** (16 GB, volle busbreedte). Geen single-channel-vergissing te repareren, geen SODIMM te vervangen — het geheugensubsysteem draait al op spec. Daarmee is de bandbreedte-bottleneck een gegeven, geen defect.
   - **Meerdere video's uit een batch tegelijk draaien.** Logische gedachte, maar precies het gemeten scenario: twee processen samen 1,3×, vier processen langzamer dan serieel. `BatchWorker` draait ze één voor één en dat moet zo blijven.

**Meetprotocol**: één vaste testvideo ("Schaats frontaal.MOV"), per stap noteren: totale analysetijd, pose-dekking (%), en of de afzet-events (aantal, been-volgorde, hoeken ±1°) gelijk blijven aan de referentie-run. Versnelling die de meting verandert is geen versnelling. Voor een modelswap komt daar `python schaats_eval.py vergelijk oud.npz nieuw.npz` + de gouden referentie bij — vergelijk dan wel **onbewerkte** analyses (`analyse.bewerkt = 0`), want een met de hand gezet skelet telt in de dekkingsmetric als detectie.

**Verwachting (bijgesteld op de meting van juli 2026)**: stap 3 is de hoofdprijs — **2,6× (m-model) tot 8× (n-model)**, mits de dekking overeind blijft. Stap 4 (OpenVINO) daar theoretisch 1,5–3× bovenop, maar ongetest op deze machine. Stappen 1 en 2 hooguit een paar procent, want ze raken de 6% die niet de detectiepass is. Stap 6 is één uitzondering waard: de **energiemodus** raakt wél de bottleneck (pakketvermogen → geheugenklok) en kan 10–30% zijn — begin daar zelfs mee, want het kost geen regel code. De eerder genoteerde "1,5–2× uit 1+2+6" was een schatting van vóór de profilering en is te optimistisch gebleken.

**Klaar wanneer:** de totale analysetijd van de testvideo minstens gehalveerd is zónder verlies van dekking of meetkwaliteit, en de snelste acceptabele configuratie als default staat.

---

## Fase 7 — Perspectiefcorrectie via baanlijnen

> **Nice-to-have voor de dagelijkse workflow, maar de validatie loopt (aug 2026).** Sinds juli 2026 is de aanname dat er **altijd recht van voren** wordt gefilmd met een **horizontale** camera. Onder die opstelling kijkt de camera nagenoeg loodrecht op het bewegingsvlak en is de perspectiefvertekening klein, dus de dagelijkse workflow heeft deze correctie niet nodig. Stappen 1 en 2 (wiskundekern + koppeling aan pijplijn en GUI) staan er en zijn opt-in; sinds 11 augustus 2026 wordt de kalibratie ook **bewaard en hergebruikt**. Wat rest is stap 3, de validatie op echt materiaal — en dát materiaal is er nu wél (zie "Stand van zaken" onderaan deze fase).

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

### Stand van zaken (11 augustus 2026)

**Stappen 1 en 2 zijn af** (7 juli 2026): de wiskundekern `schaats_perspectief.py` met zelftest, en de opt-in koppeling aan beide backends + de GUI. **Stap 3 (validatie op echt materiaal) staat open.**

**Testmateriaal gevonden.** De juli-video viel af (bewegende camera, geen dwarslijn, frontaal, 4,2 s). Het bruikbare materiaal zijn de **zeven fragmenten die uit `opnames/00005.MTS` geknipt zijn** (`bron_id` 1, titels "00005 8-41" t/m "00005 11-54", bronframes 13015–18106). Nagemeten:

| Eis | Vorige video | Deze zeven fragmenten |
|---|---|---|
| Vaste camera | ✗ ~38 px drift (≈2°) | ✅ **0–1 px over 3½ minuut**, alle zeven |
| Schuine kijkrichting (f zelfkalibreerbaar) | ✗ frontaal | ✅ duidelijk schuin langs de baan |
| ≥2 rijrichting-lijnen | ✗ 1 | ✅ **3** (blauwe baanlijn, ijs/sneeuw-rand, boardingvoet) |
| ≥1 dwarslijn | ✗ geen | ✅ met het oog aanwezig (handmatig natrekken; Hough vindt ze niet op bekrast ijs) |
| Meerdere passages | ✗ één | ✅ **zeven, ~50 tellende afzetten** |

Twee bevindingen uit dat nameten die het plan sturen:
- **De vertekening is al zichtbaar in de ongecorrigeerde analyses**: binnen elke passage loopt de afzethoek op naarmate de schaatser dichterbij komt (11-08: 44,4° op enkel-y 319 → 55,5° op y 640; 10-25: 38,0 → 41,9). Er valt dus echt iets te corrigeren.
- **Maar die drift verschilt sterk per passage over hetzelfde beeldgebied** (+2,7° bij 8-41 tegen +11° bij 11-08), dus een deel is techniekverandering of ruis. De validatie mag daarom niet leunen op "de hoek moet constant worden" en heeft een expliciet nulmodel nodig.
- **De gebouwkolommen staan zuiver loodrecht in beeld** (0–2 px over 60 px hoogte) → camera-roll ≈ 0. Daaruit volgt dat de verdwijnlijn van het ijsvlak exact horizontaal door V1 loopt; een gratis extra controle op de kalibratie, en een uitwijkroute mocht een dwarslijn ooit ontbreken.

**Validatieplan — vier tests, oplopend van goedkoop naar het klaar-criterium:**
- **A — kalibratie zonder ground truth (eerst).** De standbeen-enkel loopt over het ijs, dus het gereconstrueerde `r.wereld_xy` moet een **rechte lijn** zijn met een gladde snelheid. Rechtheidsresidu + plausibiliteit (tempo ≈ 10–12 m/s, slaglengte 5–8 m, gereconstrueerde heupbreedte vs. opgemeten) keuren de homografie af vóór er iets geannoteerd wordt. Faalt dit, dan is de rest zinloos.
- **B — robuustheid.** Laat-één-lijn-weg: herkalibreren op wisselende deelverzamelingen van de rijlijnen en meten hoeveel f, horizon en eindhoeken bewegen. Bewegen ze meer dan de geclaimde winst, dan is de kalibratie te wiebelig voor de praktijk.
- **C — diepte-drift (het klaar-criterium).** Regressie van de afzethoek op de beeldpositie per passage, vóór en na correctie; de helling moet naar nul. Nulmodel: de spreiding binnen één klein beeldgebied is de ruisvloer, en alleen een hellingreductie die daar duidelijk bovenuit komt telt.
- **D — consistentie tussen passages.** De zeven passages beslaan verschillende laterale banden (x = 31 tot x = 1219); na correctie moet de spreiding van het passage-gemiddelde krimpen. Minst gevoelig voor techniekdrift binnen één passage — en de reden dat één gedeelde kalibratie over alle zeven een harde voorwaarde is (zie hieronder).

**Afgekeurd als test:** links/rechts-antisymmetrie. Nagerekend: het L−R-verschil is nu al maar −0,9° tot +2,0°, dus geen onderscheidend vermogen.

**Geen enkele maat nodig voor de hoek (nagemeten 11 augustus 2026).** De trainer wil alleen correcte hoeken, geen snelheid of slaglengte — en dat kán, want de afzethoek is **schaalvrij**: hij volgt uit de richtingen van de lijnen, niet uit hun afstand. Gemeten met 2 baanlijnen + 2 dwarslijnen en `beenvlak`: **0,00° fout of je nu 0,5 m, 4 m of 50 m als lijnafstand invult**. De GUI heeft daarom "Alleen hoeken (geen snelheid/slaglengte)" als **standaard** (`schaal_bekend=False`). Twee dingen die daaruit volgen:
- **`onderbeen` kan niet zonder echte schaal**: die snijdt met een bol van een lengte in echte meters, dus een verzonnen lijnafstand gaf **49° fout, stilzwijgend**. De dialoog zet de methode daarom vast op `beenvlak` zolang alleen-hoeken aan staat. Voor de A/B van beide methodes (test A/C/D hierboven) moet je het vinkje uitzetten en de echte 4 m invullen.
- **Bij 3+ baanlijnen telt hun onderlinge afstand wél**, ook bij `beenvlak`: de verdwijnlijn komt dan uit de kruisverhouding. Ongelijk verdeelde lijnen die als gelijk verdeeld worden opgegeven leveren een **weigering** op (f² ≤ 0) — nooit stilzwijgend fout — en met de juiste `rij_offsets` weer 0,00°. `rij_offsets` zit in de module maar niet in de GUI; de foutmelding wijst daarom naar de uitweg: precies **2 baanlijnen + 2 dwarslijnen**, want dan komt V2 uit de dwarslijnen.

**Eerste praktijktest op `00005 11-23` (11/12 augustus 2026): de correctie maakte de hoeken slechter, en dat is uitgezocht.** Getekend werden 2 baanlijnen + 2 dwarslijnen, "alleen hoeken", methode `beenvlak`. Resultaat: correcties van −22,9° tot +71,2°, statusbalk "onbetrouwbaar". Drie oorzaken, alle drie nagemeten:
1. **`f` is uit deze camerastand principieel niet te schatten.** De twee dwarslijnen lopen in beeld vrijwel evenwijdig (helling −0,0219 vs −0,0204), dus V2 ligt op **133× de beeldmaat** — praktisch op oneindig, precies het geval waarvoor de moduledocstring `f_px` verplicht stelt. Gevolg: f = 7081 px = 3,3× de beeldbreedte ≈ **17° beeldhoek**, onmogelijk voor zo'n opname. En het is niet alleen fout maar *betekenisloos*: **één lijnuiteinde 5 px verschuiven laat f van 3827 px naar "onmogelijk" springen**. De camera kijkt hier bijna lángs de baan — mijn eerdere inschatting "duidelijk schuin" was verkeerd; hij is schuin genoeg voor een nette V1, maar de dwársrichting ligt vrijwel evenwijdig aan het beeldvlak.
2. **`beenvlak` is dégenereerd voor precies deze stand.** Die methode legt het onderbeen in een verticaal vlak in de rijrichting; kijkt de camera langs de baan, dan ligt de kijkstraal ín dat vlak (de bestaande `vlak_conditie_deg`-vlag). Gemeten over 164 frames: `beenvlak` markeert er **25–61 als onbetrouwbaar**, `onderbeen` maar **5–7**. De aanbeveling om `beenvlak` te gebruiken (omdat die geen maten nodig heeft) was dus verkeerd voor dit materiaal.
3. **Maar zelfs met de betere methode wint de correctie niet.** Met f gesweept van 1200 tot 6380 px: ongecorrigeerd is de spreiding van de tellende afzethoeken **sd 1,6°**; het beste gecorrigeerde geval is `onderbeen` met **sd 2,0°**, `beenvlak` blijft op 3,8–7,4°. Op deze clip valt er dus weinig te winnen — logisch, want een camera die langs de baan kijkt heeft juist wéinig perspectiefvertekening (de oorspronkelijke ROADMAP-premisse). Let wel: sd over vier afzetten is een zwak getal, en "consistent" is niet hetzelfde als "correct".

**Ingebouwde poort naar aanleiding hiervan:** `kalibreer_uit_lijnen` weigert nu de zelfkalibratie van f als het verste verdwijnpunt boven `VP_CONDITIE_MAX` (30× de beeldmaat) ligt, met uitleg dat `f_px` opgegeven moet worden; boven `VP_CONDITIE_WAARSCHUW` (5×) volgt een waarschuwing. Geijkt op de zelftest-camera's, die op 1,0 / 1,5 / 9,3 zitten en allemaal de juiste f leveren. Verder toont de `KalibratieKiezer` het **residu niet meer** bij precies 2+2 lijnen: het stelsel is dan exact bepaald, dus het residu is per constructie 0,00 px en las als "perfect gekalibreerd" — er staat nu dat er géén controle mogelijk is en dat een derde dwarslijn die wél geeft.

**Vervolg is dus: `f_px` los bepalen** (schaakbordkalibratie met dezelfde camcorder/zoomstand, of de cameraspecificatie), en pas daarna opnieuw meten — bij voorkeur op `11-08`, want die passage laat wél duidelijke drift zien (44,4° → 55,5°) terwijl `11-23` ongecorrigeerd al vlak is.

**Onderbeenlengte vs. heupbreedte.** Voorstel om heupbreedte als anker te gebruiken is nagerekend en afgeraden als liniaal: heupbreedte meet 28–44 px tegen 47–85 px voor het onderbeen (~60%, dus ~1,7× ruisgevoeliger), wordt zelf verkort door romprotatie (dat is precies het `bocht_ratio`-signaal) en legt het bekken vast in plaats van de knie — chainen naar de knie haalt de femurlengte erbij in plaats van eraf. Wél bruikbaar als onafhankelijke controle in test A. Let op het misverstand eronder: de **3D**-onderbeenlengte ís constant; alleen de projectie varieert, en die variatie is juist het signaal waar de bol-snijding op werkt. De praktische zorg klopt wel — de lengte uit de video afleiden is onbetrouwbaar (zie `kalibreer_onderbeenlengte`). Uitweg: `methode='beenvlak'` heeft **helemaal geen lengte nodig**; draai beide methodes naast elkaar en laat A/C/D beslissen.

**Voorwaarde ingebouwd (11 augustus 2026): de kalibratie wordt bewaard en is herbruikbaar.** Zonder dat zou test D zeven keer handmatig natrekken vergen — zeven nét andere kalibraties, en dan meet je die spreiding in plaats van het effect van de correctie. Wat er opgeslagen wordt is de invoer (`KalibratieInvoer`: lijnen + lijnafstand/`f_px`/offsets/notitie + beeldmaat) in `instellingen_json`; de camerastand wordt eruit herberekend. Heropenen herstelt de correctie, de batch-flow vraagt de kalibratie één keer voor de hele rij, en `_kies_perspectief` biedt eerdere kalibraties van dezelfde beeldmaat aan om over te nemen. Zie CLAUDE.md voor de details.

---

## Fase 8 — Lange video's: bruikbare fragmenten knippen in de app ✅

> **Af (11 augustus 2026)** — zelftest (`python schaats_db.py`) groen in beide venvs incl. de v1→v3- en v2→v3-migratie en de opnames-round-trip; headless rooktest van de opnameslijst, het knipvenster (markeren, sneltoetsen S/E/Delete, balk tekenen, fragmentlijst) en de batch-aansluiting (voorgevulde rijen dragen `bron_*` tot in `sla_analyse_op`).
>
> **Wat er staat**, precies volgens het plan hieronder — het knippen levert de invoer van de bestaande batch-flow, dus aan de analysekant is niets veranderd:
> - **`schaats_db`, schema v3**: `bronvideo`-tabel + `analyse.bron_id`/`bron_start_frame`/`bron_eind_frame`, met `synchroniseer_bronmap` / `lijst_bronvideos` / `bronvideo` / `wijzig_bronvideo` / `bron_fragmenten`. `open_db` maakt `opnames/` aan.
> - **`schaats_analyse.knip_fragmenten()`**: één sequentiële pass, exact op de gemarkeerde frames, `mp4v`.
> - **GUI**: tweede tabblad **"Opnames"** op de startpagina (status + notitie ter plekke te wijzigen, telling `3 fragmenten · 2 schaatsers`), **`FragmentKiezer`** + **`FragmentBalk`**, `VideoSpeler(snel_zoeken=, toon_overlay=)`, en `BatchAnalyseDialog(voorgevuld=...)`. De Info-dialoog van een analyse toont voortaan **"Uit opname: … (12:30–13:05)"**.
>
> **Werkbaar op een echte opname (nagemeten 11 augustus 2026, na de eerste praktijktest — de GUI liep vast op `00005.MTS`, 4,2 GB AVCHD 1920×1080 @ 25 fps, 34.728 frames ≈ 23 min).** Het decoderen bleek níet het probleem (~9 ms per frame, een `grab()` ~3 ms, een seek ~80 ms) — het aantal aanroepen wel. Drie oorzaken, alle drie verholpen:
> - **Eén frame terug = de video opnieuw doorspoelen vanaf frame 0.** `snel_zoeken` seekte alleen bij een sprong > 30 frames, dus juist de kleine stap achteruit viel in de sequentiële route: op frame 20.000 kostte dat ~84 s met een volledig bevroren venster. Nu seekt **elke** stap achteruit → 99 ms. Kleine sprongen vooruit blijven sequentieel (goedkoper dan een seek, en frame-voor-frame stappen rond een grens blijft exact).
> - **Slepen aan de tijdlijn stapelde honderden seeks op.** De slider zet nu alleen het laatst gevraagde frame klaar; een `QTimer` met interval 0 tekent het zodra de wachtrij leeg is, zodat alle tussenwaarden vervallen. Gemeten: 300 slider-signalen in 1 ms verwerkt, daarna één keer tekenen.
> - **Doorscannen kon helemaal niet.** De snelheidkeuze hield op bij 1×, dus één keer doorkijken duurde 23 minuten. Er staan nu **2×/4×/8×** in, uitgevoerd door frames **over te slaan** (4 frames per tik op het fps-tempo) i.p.v. sneller te decoderen — dat laatste haalt geen decoder.
> - **Het venster paste niet op het laptopscherm.** Op 1280×800 (werkgebied 752 px) eiste het knipvenster 723 px minimaal; met de titelbalk erbij zakte de knoppenbalk onder de rand en was "Klaar" onvindbaar. De minima zijn verlaagd (video-ondergrens 400×200, kleinere fragmenttabel, krappere marges) → **553 px**, en `zet_venstergrootte` draait nu als laatste, tegen een complete layout. De andere dialogen zijn nagemeten en passen — **ook `KalibratieKiezer`** (hernagemeten 11 augustus 2026: minimum 1008×460, opent op 1150×700; de eerdere claim van 1724 px was onjuist).
> - Bijvangst: `knip_fragmenten` gebruikt `grab()` zonder `retrieve()` voor frames buiten elk fragment. 10 s knippen op minuut 20 kost daarmee 96 s i.p.v. ~270 s — verwaarloosbaar naast de analyse (~2 s/frame) die erop volgt.
>
> **Gemeten (11 augustus 2026):**
> - **Seek-afwijking op echte iPhone-.MOV's** (de prijs van `snel_zoeken`): op `IMG_8997.mov` en `IMG_9001.mov` **0–1 frame** (0–33 ms). Op `Schaats frontaal.MOV`, waar `CAP_PROP_FRAME_COUNT` 108 frames meldt maar er 103 leesbaar zijn, loopt het op tot **4 frames (168 ms)** aan het eind van de clip — precies de VFR-drift waarvoor de weergavepagina nooit seekt. Voor een grens die je met het oog bepaalt is dat acceptabel; het fragment zelf blijft exact, want `knip_fragmenten` telt sequentieel vanaf frame 0.
> - **Hercodering** (de A/B die punt B hieronder vroeg): `Schaats frontaal.MOV` uit zichzelf geknipt (103 frames, 1920×1080) en beide met dezelfde code geanalyseerd. **Dekking 100% in beide, zes afzetten in beide, dezelfde L-R-volgorde (RLRLRL), nul alternatiefouten**, en de eventgrenzen op één frame na identiek (35 vs. 36). De hoeken: 42,2→41,0 · 42,5→42,5 · 42,9→42,5 · 40,0→40,5 · 45,6→45,9 (en de afgekapte 50,0→51,5, die toch niet meetelt) — dus **maximaal 1,2° op een tellende afzet, meestal ≤ 0,5°**. Gewrichtsposities verschillen 1–2 px mediaan (p95 6–11 px) op een beeld van 1920 px breed. Conclusie: **cv2 met `mp4v` blijft de default.** De afwijking zit in dezelfde orde als de ±1° die fase 6 als acceptabel voorstelt, en de uitweg (ffmpeg stream-copy) zou een GOP-marge van 1–2 s aan de voorkant terugbrengen — precies wat hier niet gewenst is. Wordt dit ooit tóch storend, dan is de eerlijke oplossing een betere codec-instelling of ffmpeg **mét** hercodering, niet `-c copy`.
>
> **Afwijkingen van het plan hieronder:**
> - De opnames zitten in een **tabblad** naast de schaatserslijst (niet als derde kolom): de werklijst hangt niet aan de schaatserselectie.
> - **De doelschaatser wordt niet in het knipvenster aangewezen** (de openstaande vraag onderaan). `DoelKiezer` staat dus op frame 0 van elke clip — en dat is precies het beeld waarop "start" gedrukt werd, want er wordt exact op de gemarkeerde frames geknipt. De backend-verbouwing (`_kies_seed(..., doel_frame)`) is daarmee nog niet gedaan.
> - `VideoSpeler` kreeg naast `snel_zoeken` ook **`toon_overlay`**: met lege `FrameResultaat`-objecten zou de overlay op élk frame "Geen pose gedetecteerd" zetten, en "Volg schaatser"/"Automatische zoom" zijn vinkjes die zonder analyse niets kunnen doen. Kleine sprongen (≤ `SEEK_DREMPEL_FRAMES`, 30) blijven sequentieel, zodat frame-voor-frame stappen rond een grens exact blijft.
> - Bijvangst: de **Info-dialoog** toont de herkomst van een fragment (`analyse_meta` haalt `bron_naam` met een LEFT JOIN mee).
>
> **Aanleiding uit de praktijk:** een training levert één opname van een half uur op. Die wordt nu buiten de app (Clipchamp) met de hand in bruikbare stukken geknipt — dat kost meer tijd dan de analyse zelf en het externe programma werkt slecht. Het knippen hoort in de app, náást de video die je toch al aan het bekijken bent.

**Doel:** een opname van een half uur openen, daarin de bruikbare stukken markeren (start/stop per stuk), en die stukken daarna in één keer laten analyseren — precies zoals de app nu al een losse clip analyseert.

> **Uitgangspunt: dit is een knipprogramma, en het knippen is volledig handmatig** (vastgelegd 10 augustus 2026). De app bepaalt **niets** zelf: niet wanneer de schaatser in beeld is, niet waar een stuk begint of eindigt, en er komt geen seconde marge bij of af. De trainer kijkt, drukt op start en stop, en dát zijn de grenzen. Alles wat het programma doet is die grenzen onthouden, tonen en er clips van wegschrijven. Elk voorstel om hier "slimheid" in te bouwen is bij voorbaat afgewezen — zie "Bewust overwogen en niet gekozen".

### De flow zoals gevraagd

1. Opname van een half uur kiezen uit de nieuwe lijst **"Opnames"** op de startpagina (`<bibliotheek>/opnames/`, gedeeld via Drive — zie hieronder).
2. Doorlopen/scrubben; bij een bruikbaar stuk: **"Start bruikbaar beeld"** → **"Stop bruikbaar beeld"**. Herhalen voor stuk 2, 3, … x.
3. Tijdens het markeren is **zichtbaar welke stukken al gemarkeerd zijn** (gekleurde blokken op een balk onder de tijdlijn) en welke stukken van deze bronvideo **in een eerdere sessie al geanalyseerd zijn**.
4. Op **"Klaar — analyseer x fragmenten"**: per fragment een schaatser (en titel) kiezen, dan per fragment de doelschaatser aanwijzen, daarna draait de analyse zoals nu.

### Architectuur: het knippen levert de invoer van de bestáánde batch-flow

De kern van dit ontwerp is dat er ná het knippen **niets nieuws** hoeft te gebeuren: `BatchAnalyseDialog.taken` is al een lijst `{input_pad, schaatser_id, titel}` en `_nieuwe_batch_analyse` vraagt daar al per video doelschaatser + horizon bij op, waarna `BatchWorker` de rij afdraait en elke analyse zelf opslaat. Zodra elk fragment een gewoon videobestandje is, valt de hele nieuwe functie uiteen in **twee stappen die vóór die dialoog komen te staan**:

- **A. `FragmentKiezer`** (nieuwe dialoog) — markeren op de bronvideo → lijst `(start_frame, eind_frame)`.
- **B. `knip_fragmenten()`** (nieuwe helper) — die frameranges als losse clips wegschrijven → lijst bestandspaden.

Daarna: `BatchAnalyseDialog` openen met die paden **voorgevuld** (rijen staan er al, de trainer vult alleen schaatser + titel in). Geen tweede analyse-pijplijn en geen tweede opslagroute — aan de analysekant verandert er niets. De schemabump hieronder gaat dan ook niet over het analyseren, maar over het bijhouden van de opnames zelf.

### A. `FragmentKiezer` — de knipdialoog

- **Hergebruik `VideoSpeler`** voor het afspelen: scrubslider, transportknoppen, snelheidcombo (op 4× door een half uur scannen) en zoom zitten er al in. De speler verwacht een `resultaten`-lijst (voor overlay, kader en sliderlengte); een lijst van `totaal_frames` lege `FrameResultaat`-objecten volstaat — `kader_reeks` geeft dan `None` en de automatische zoom valt terug op vaste zoom. **Eerst verifiëren** dat `laad()` daar niet over struikelt; zo niet, dan een kleine eigen speler zoals `DoelKiezer` er al een heeft.
- **Twee knoppen + sneltoetsen**: "Start bruikbaar beeld" (`S`) en "Stop bruikbaar beeld" (`E`). Na "Stop" wordt het fragment **meteen aan de lijst toegevoegd** en zichtbaar in de balk; de knop springt terug naar "Start" voor het volgende stuk. Zolang er een start openstaat is alleen "Stop" actief (en andersom) — dan kan er geen half fragment ontstaan.
- **Fragmentbalk** onder de tijdlijn: één widget zo breed als de slider, met per fragment een gekleurd blok op `start/totaal … eind/totaal`. Groen = zojuist gemarkeerd, grijs = in een eerdere sessie al geanalyseerd (zie hieronder), oranje = het lopende (nog niet gestopte) fragment. Klik op een blok → springt erheen en selecteert het; `Delete` gooit het weg. Dit is de enige echt nieuwe teken-code van de fase.
- **Lijstje ernaast** met per fragment `#`, begin–eind als `m:ss`, duur, en een verwijderknop. Overlappen twee fragmenten elkaar, dan wordt dat **zichtbaar gemaakt** (het overlappende stuk in een afwijkende kleur) maar er wordt níets automatisch samengevoegd of ingekort — de trainer past het zelf aan of laat het zoals het is.
- **Navigatiehulp**: knoppen ±1 s / ±10 s / ±1 min en een tijdinvoerveld. Op een half uur is de slider te grof om een afzet terug te vinden.
- **Optioneel: de doelschaatser meteen hier aanwijzen** (te beslissen bij het bouwen). Wie een fragment markeert, kijkt op dat moment naar de schaatser die hij bedoelt — dat is het natuurlijke moment om hem aan te klikken, en het scheelt straks x losse `DoelKiezer`-dialogen. Technisch past dat goed bij de YOLO-backend: die verzamelt álle detecties offline en stikt vanaf het seed-tracklet **voor- én achterwaarts** (`_stik_keten`), dus een seed midden in de clip is even goed als een seed op frame 0. Nodig is dan `_kies_seed(..., doel_punt, doel_frame)` dat vanaf `doel_frame` zoekt i.p.v. vanaf 0, plus het framenummer meesturen in `instellingen_json`. De MediaPipe-backend is streaming en kan dat niet zonder verbouwing — die houdt gewoon de bestaande frame-0-route (het is de terugvalbackend). Zolang dit er niet is, blijft `DoelKiezer` op frame 0 van de clip staan — dat is precies het beeld waarop "start" werd gedrukt, want er wordt exact op de gemarkeerde frames geknipt (zie hieronder).

### B. `knip_fragmenten()` — de clips wegschrijven

- **Eén sequentiële pass** over de bronvideo met `cv2.VideoCapture`, waarbij elk frame naar de `VideoWriter` van het fragment gaat waarin het valt. Zo wordt de video precies één keer gedecodeerd en hoeft er **nergens geseekt** te worden — de framenummering blijft exact die van de bron (zelfde motief als de eigen leeslus in `_detecteer_alles`). Voortgangsdialoog eromheen; een half uur decoderen kost enkele minuten, verwaarloosbaar naast de analyse die erop volgt.
- **Codec**: `mp4v` (zit in de opencv-python-wheel, `avc1` is op Windows vaak niet beschikbaar). Er wordt dus **her-gecodeerd** — kwaliteitsverlies is voor pose-detectie verwaarloosbaar, maar meet het één keer: knip een bekende clip uit zichzelf en vergelijk de analyse met `python schaats_eval.py vergelijk oud.npz nieuw.npz`. Wijkt dat merkbaar af, dan is de uitweg **ffmpeg met `-c copy`** (geen hercodering, vrijwel instant) als `shutil.which("ffmpeg")` iets vindt, met de cv2-route als terugval. **Maar let op — dat botst met "geen marges":** stream-copy kan alleen op keyframes beginnen, dus het fragment valt aan de voorkant tot een GOP (~1–2 s) langer uit dan wat je markeerde. Dat is precies de stilzwijgende marge die hier niet gewenst is, en het maakt frame 0 van de clip een ander beeld dan waarop je "start" drukte — met alle gevolgen voor de doelkeuze. Daarom: **cv2 met hercodering is de default**, en ffmpeg alleen als de A/B uitwijst dat de hercodering de meting echt raakt. In dat geval is de eerlijke oplossing niet stream-copy maar ffmpeg mét hercodering van alleen de eerste GOP (`-ss` ná `-i`), of een betere codec-instelling in cv2.
- **Schrijven naar een tijdelijke map**; `sla_analyse_op` kopieert de clip daarna zoals altijd naar `media/<uuid>/`. Dat is één extra kopie van een kort bestandje — niet de moeite om `sla_analyse_op` voor open te breken.
- **Geen automatische marges — er wordt exact op de gemarkeerde frames geknipt** (besloten 10 augustus 2026). De trainer kijkt tijdens het markeren naar het beeld en bepaalt de grenzen zelf; het programma hoort daar niet stilzwijgend seconden bij of af te halen. Twee dingen die daarbij goed zijn om te weten, maar géén reden voor automatiek:
  - Aan de **voorkant** valt sowieso niets te winnen: een stand-run die aan het begin is afgekapt telt gewoon mee (`bepaal_afzet_uit_strek`, `run['afgekapt']`) — daar mist alleen de load-fase, terwijl de push-voltooiing, waar de hoek uit komt, wél in beeld is. Bovendien zou lucht aan de voorkant de doelkeuze juist moeilijker maken: `DoelKiezer` krijgt **frame 0 van de clip** (`_lees_eerste_frame(pad)` in `_nieuwe_batch_analyse`) en `_kies_seed` zoekt het klikpunt in de eerste `KLIK_ZOEK_S` (6) seconden — sinds 26 augustus 2026, want met de oude 60 frames miste de klik een schaatser die pas na 3,4 s gedetecteerd werd — dus hoe verder de schaatser daar weg staat, hoe groter de kans op een misser, en bij een misser volgt de analyse de grootste beweger mét melding. Zo is frame 0 precies het beeld waarop "start" werd gedrukt.
  - Aan de **achterkant** eindigt de laatste stand-run op het clip-einde in plaats van op een beenwissel; die afzet krijgt `ONV_AFGEKAPT`, blijft grijs zichtbaar in de tabel maar valt buiten gem/min/max. Wil je die laatste afzet meetellen, dan druk je op "stop" ná de beenwissel — een keuze die de trainer bij het kijken maakt, niet het programma.

### De opnames zelf in de bibliotheek: `bronvideo` (schema v3)

> **Besloten 11 augustus 2026.** De database kent nu alleen geanalyseerde clips. Er hoort ook een plek te zijn voor de **nog niet geanalyseerde opnames** — die map bestaat al en staat, net als de bibliotheek, **in de gedeelde Google Drive**. Daarmee is "wat moet er nog geknipt worden" geen persoonlijk lijstje maar een **werklijst voor het team**, en dát is wat de schemabump rechtvaardigt: hij levert niet alleen de grijze blokken in het knipvenster op, maar ook een overzicht van openstaand werk.

**De ontwerpregel: de map is de waarheid over wélke bestanden er zijn, de database over wat wij ervan weten.** De bestandslijst wordt bij het openen van schijf gescand, niet uit de DB gelezen — anders loopt de DB scheef zodra iemand een bestand hernoemt of weggooit en zit je aan opruimwerk vast. In de DB staat alleen wat je nooit van schijf kunt aflezen: status, aantekening, en welke analyses uit welk stuk van welke opname komen.

**Waar de opnames staan:** `<bibliotheek>/opnames/`, dus **binnen** de bibliotheekmap. Dan blijven alle paden relatief met forward slashes (de fase 1-discipline) en is er géén extra pad-instelling per trainer nodig. Staat de map er niet, dan maakt de app hem aan.

**Schema `user_version=3`** — één nieuwe tabel plus drie kolommen, allebei cloud-veilig via het bestaande `_migreer`-patroon (`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN`, zoals bij v1→v2):

```sql
bronvideo(id INTEGER PRIMARY KEY,
          bestand,            -- relatief pad binnen de bibliotheek (opnames/…)
          naam, bytes,        -- identiteit: UNIQUE(naam, bytes)
          fps, totaal_frames, -- één keer uitgelezen, scheelt elke keer openen
          status,             -- 'nog doen' | 'bezig' | 'klaar' | 'onbruikbaar'
          notitie,            -- vrije tekst ("training 3 aug, tempo-serie")
          bijgewerkt_door, toegevoegd_op)

analyse … + bron_id, bron_start_frame, bron_eind_frame   -- NULL bij een losse clip
```

- **Identiteit = het relatieve pad** (`opnames/<bestandsnaam>`), `UNIQUE(bestand)`. Eén map kan geen twee bestanden met dezelfde naam bevatten, en omdat de map ín de bibliotheek zit is dat pad bij elke trainer hetzelfde — precies waarom de fase 1-discipline (alles relatief, forward slashes) hier zijn rente oplevert. Een hernoemd bestand geldt als nieuw; de oude rij blijft met zijn analyses bestaan en wordt getoond als "bestand niet gevonden".
- **`bytes` is géén identiteit, alleen de sync-check.** Dat onderscheid is wezenlijk: een opname die bij een collega nog binnenkomt, is op dat moment *kleiner* dan wat er in de DB staat. Zat de grootte in de sleutel, dan zag de scan een half gedownload bestand aan voor een nieuwe opname en kwam er een tweede rij bij — een rommelige lijst en verdwenen fragmentgeschiedenis, precies wanneer je die nodig hebt.
- **Oude analyses houden `bron_id` NULL** — dat klopt ook: die kwamen van een losse clip, niet uit een opname. Nergens een migratie die iets moet raden.
- **`synchroniseer_bronmap(bieb)`** scant `opnames/`, voegt nieuwe bestanden toe met `INSERT OR IGNORE` (twee trainers die tegelijk dezelfde nieuwe opname zien botsen dan niet) en laat rijen van verdwenen bestanden staan. Draait bij het openen van de bibliotheek en bij "Vernieuwen" (fase 4-knop, doet dit er gewoon bij). **Schrijft alleen als er echt iets nieuws is** — anders zou elke app-start van elke trainer de gedeelde DB aanraken, en die hoort volgens de fase 4-discipline zo veel mogelijk in rust te zijn voor de syncer.
- **De status zet je zelf.** Er wordt niets automatisch op "klaar" gezet als alle fragmenten geanalyseerd zijn: het programma kan niet weten of jij de opname af vindt. Het toont de telling (`3 fragmenten · 2 analyses`), jij zet de status. Zelfde lijn als de rest van deze fase.

**Sync-status, nu écht nodig.** Google Drive Mirror zet alle bestanden lokaal, maar een opname van een half uur in 4K is minutenlang onderweg. De `video_bytes`-truc uit fase 4 werkt hier één op één: de trainer die de opname als eerste toevoegt legt `bytes` vast, en bij een collega wiens lokale bestand kleiner is, is de download nog bezig. `video_sync_status()` kan daar ongewijzigd voor gebruikt worden — melden en niet openen, in plaats van het knipvenster op een half bestand laten stuklopen.

**Wat de GUI ermee doet:** de startpagina krijgt naast de schaatserslijst een **tweede weergave "Opnames"** (tabblad of knop): bestandsnaam, duur, status, notitie, en per opname `3 fragmenten · 2 analyses`. Dubbelklik → de `FragmentKiezer`. Status en notitie zijn ter plekke te wijzigen; `bijgewerkt_door` erbij, zodat zichtbaar is wie een opname op "klaar" zette — hetzelfde motief als `aangemaakt_door` bij een analyse.

**En het levert de grijze blokken:** met `bron_id` + `bron_start_frame`/`bron_eind_frame` op de analyse is "welke stukken van deze opname zijn al gedaan" één query in plaats van een scan door alle `instellingen_json`-velden. Het knipvenster tekent ze grijs, en je ziet bij het heropenen van dezelfde opname meteen waar je gebleven was. Bijvangst voor fase 2: analyses uit dezelfde training zijn voortaan als zodanig herkenbaar.

### Valkuilen

- **Achteruit scrubben op een half uur is nu onwerkbaar.** `VideoSpeler._lees_frame_exact` seekt bewust nooit (VFR-video's geven een frame-onnauwkeurige seek) en spoelt bij een sprong terug de video vanaf frame 0 opnieuw door. Op een clip van 10 s is dat niets, op 50.000 frames is het onbruikbaar. **Dit is de enige echte blokkade van deze fase** en de oplossing is dat het knipvenster een ándere afweging maakt dan de weergavepagina: hier is het beeld een **kijkje, geen meting**, dus een `CAP_PROP_POS_FRAMES`-seek mag. Bouw dat als een expliciete vlag op de speler (bv. `snel_zoeken=True`) zodat de weergavepagina onaangeraakt blijft, en herstel na de seek de interne cursor (`_weergave_pos`) zodat vooruit afspelen daarna weer klopt.

  **Wat dat kost:** op een VFR-bron kan het getoonde beeld enkele frames afwijken van het gerapporteerde nummer, dus de knip komt maximaal een paar frames naast het beeld waarop je drukte. Voor een grens die je met het oog bepaalt is dat onzichtbaar (~0,1 s) — en het alternatief, nooit seeken, maakt het doorbladeren van een half uur onmogelijk. Het fragment zelf blijft wél exact: `knip_fragmenten()` telt sequentieel vanaf frame 0, dus binnen de clip loopt niets uit de pas. Meet één keer op een iPhone-.MOV hoe groot de afwijking in de praktijk is.
- **VFR-bronnen**: de fragmenten worden met de gerapporteerde `info.fps` weggeschreven. De hele app rekent al met constante fps, dus dit is geen nieuwe afwijking — wel het noteren waard, want op een half uur loopt een VFR-drift verder op dan op 10 s.
- **`_lees_eerste_frame` + `DoelKiezer` per fragment** werken ongewijzigd zodra elk fragment een echt bestand is; ze lezen simpelweg frame 0 van die clip. Dat is meteen het argument om écht te knippen in plaats van frameranges door de pijplijn te sluizen.

### Bewust overwogen en niet gekozen

- **Niet knippen, maar analyses naar de bronvideo + framerange laten wijzen.** Verleidelijk nu de opname toch al in de bibliotheek staat: nul extra bytes, geen hercodering. Toch niet doen, en de reden is de **weergave**, niet de opslag. `VideoSpeler._lees_frame_exact` seekt bewust nooit en spoelt sequentieel; een analyse die begint op frame 40.000 van de bronvideo zou bij elke opening en bij elke sprong terug een half uur video moeten doorspoelen. Daar bovenop breekt het `media/<uuid>/` als eenheid: de cascade bij verwijderen (een analyse wissen mag de opname van vijf andere analyses niet meenemen), `video_bytes`, de sync-melding, de duurkolom en de vergelijkpagina rekenen er allemaal op dat één analyse één eigen videobestand heeft. De extra opslag valt bovendien mee: de fragmenten samen zijn een fractie van de opname waar ze uit komen.
- **Automatisch bruikbare stukken voorstellen** (een goedkope detectiepass die "frontale schaatser in beeld" zoekt — de `_BochtWacht`-machinerie doet feitelijk al zo'n classificatie): **afgewezen, en niet "voor later"**. De gevraagde functie is een knipprogramma; de trainer ziet zelf prima wat bruikbaar is en wil daar geen voorstel van de computer overheen. Bovendien zou zo'n voorpass precies de detectietijd kosten die deze fase juist bespaart. Als dit ooit terugkomt, dan als een apart idee met een eigen aanleiding — niet als onderdeel van fase 8.

**Klaar wanneer:** een opname van een half uur in `opnames/` zetten, hem in de nieuwe opnameslijst zien staan als "nog doen", daarin bijvoorbeeld zes bruikbare stukken markeren, op "Klaar" drukken en zonder verder handwerk zes analyses in de bibliotheek terugvinden — en bij het opnieuw openen van diezelfde opname zien welke stukken al gedaan zijn, ook op de pc van een collega.

**Omvang:** 2 sessies, in twee losse stukken te bouwen. (a) Schema v3 + `synchroniseer_bronmap` + de opnameslijst op de startpagina — dat is `schaats_db`-werk met een zelftest-uitbreiding en staat op zichzelf. (b) Het knipvenster + `knip_fragmenten()` + de aansluiting op `BatchAnalyseDialog`; de fragmentbalk en het snel-zoeken zijn daar het echte werk, de rest is bestaande onderdelen aan elkaar knopen.

---

## Losse eindjes in de doelkeuze (open, gevonden 26 augustus 2026)

Beide kwamen boven water bij het repareren van de doelklik (`KLIK_ZOEK_S`, zie de fix in
BUGS.md C2). Geen van beide is toen aangepakt: ze vielen buiten die vraag en verdienen een
eigen A/B.

### 1. De stitch-poort groeit tot bijna de hele beeldbreedte

`_stik_keten` laat zijn afstandspoort lineair meegroeien met het gat
(`STITCH_GATE_BASIS` + `STITCH_GATE_GROEI`·gat). Bij een gat dat net onder
`STITCH_MAX_GAP_S` (2,0 s) blijft is dat **0,06 + 0,015 × 49 = 0,795** — op een
genormaliseerd beeld van 1,0 breed is er dan feitelijk geen positie-eis meer over, en houdt
alleen de kleurpoort nog iets tegen.

**Gemeten op `00005 8-41`** (offline replay op een gedumpte detectiepass): de keten begint
met de frames 27 en 35 van ByteTrack-ID 16 op x ≈ 0,865, en plakt die over een gat van 49
frames (1,96 s) aan de doelschaatser die op frame 84 op x = 0,378 staat — een werkelijke
sprong van **0,490**, ruim binnen die poort van 0,795. Dat het om twee verschillende
personen gaat is hard te maken zonder naar het beeld te kijken: **datzelfde ID 16 is in de
frames 43–119 aantoonbaar ergens anders**, namelijk op x = 0,871 → 0,963 met een gestaag
groeiende bbox (area 0,0136 → 0,0294). De keten beweert dus dat één persoon tegelijk links
en rechts in beeld is. Dat de kleurpoort hem doorliet komt doordat `_splits_op_kleur` ID 16
zelf al in tweeën had geknipt (27–35 tegen 43–136) — de kop was dus qua pakkleur niet meer
dezelfde als de rest van dat ID.

**De schade was hier nul**: beide frames vallen in de bocht, dus ze leveren geen `lm_data`
en geen meting op. Dat is geluk van deze clip. In een fragment zonder bocht zet zo'n kop
meteen aan het begin een skelet op de verkeerde persoon, en dat is precies een plek waar de
eerste stand-run begint.

**Richtingen** (nog niets van gekozen): een **plafond** op de poort, zoals de MediaPipe-
tracker dat met `TRACK_GATE_MAX` al doet; of de groei koppelen aan de **voorspelde
verplaatsing** in plaats van lineair aan het aantal frames; of de kleureis strenger maken
naarmate het gat groeit. Let op dat dit de spiegeling is van de nearest-first-regel die in
`_stik_keten` al zit: die koos bewust het kleinste gat omdat een verre sprong te veel
vrijheid geeft — hier krijgt diezelfde verre sprong die vrijheid alsnog via de poort.

**Omvang**: klein qua code, maar de A/B is het werk — dit raakt elke bestaande analyse, dus
meten op meerdere clips (dekking, events, L-R-alternantie, botlengte-CV) vóór en ná, met
`schaats_eval.py vergelijk`.

### 2. De MediaPipe-backend kent het klik-zoekmechanisme niet

`DoelTracker._seed` (in `schaats_analyse.py`) pakt bij een muisklik de pose die het dichtst
bij het klikpunt ligt in het **eerste frame waarin überhaupt iemand gedetecteerd is** —
zonder afstandspoort, zonder zoekvenster en zonder melding. Een klik "lukt" daar dus altijd,
desnoods op een omstander tien meter verderop, en de gebruiker hoort er niets over. De
YOLO-backend heeft sinds 26 augustus 2026 wél een venster (`KLIK_ZOEK_S`), een poort
(`KLIK_POORT_BASIS`/`_GROEI`) en een waarschuwing.

**Lage prioriteit**, want deze backend wordt in de praktijk niet gebruikt: de GUI kiest YOLO
zodra torch/ultralytics er is (`IS_YOLO`) en in de gebundelde app zit MediaPipe niet eens.
Het blijft de terugval voor een omgeving zonder torch, dus wegwerken hoeft niet — weten dat
het verschil er is wel, want een analyse uit die backend is dan op een ander doel gebaseerd
dan de trainer aanwees.

**Omvang**: klein (dezelfde poort/venster-logica in `_seed`), maar zonder testmateriaal in
die venv is er weinig te valideren.

## Volgorde & omvang (grove inschatting)

| Fase | Wat | Omvang |
|---|---|---|
| 0 | Serialisatie (`npz` + plain landmarks) | ✅ **af** (16 jul 2026) |
| 1 | `schaats_db.py` + bibliotheek-GUI + nieuwe-analyse-flow | ✅ **af** (18 jul 2026) |
| 2 | Voortgangsgrafiek, notities, export | klein, 1 sessie — *pas ná de analyse* (zie fase 2) |
| — | Batch-analyse (extra, buiten de fasering) | ✅ **af** (20 jul 2026) |
| — | Vergelijk schaatsers + `VideoSpeler`-refactor (extra) | ✅ **af** (27 jul 2026) |
| — | Bochtdetectie: bocht niet meer analyseren (extra) | ✅ **af** (5 aug 2026) |
| — | Appversie per analyse + info-dialoog (extra) | ✅ **af** (6 aug 2026) |
| 3 | Skelet-editor met uitvloeien + undo | ✅ **af** (20 jul 2026; skelet plaatsen op gat-frames 31 jul 2026) |
| 4 | Instellingen, gedeelde map, conflictafhandeling | ✅ **af** (20 jul 2026) |
| 5 | Horizon via twee getrackte punten | *nice-to-have (niet nu — horizontale camera)*; middelgroot, 1–2 sessies (stap 4, punt-overdracht, is het meeste werk) |
| 6 | Sneller analyseren | **stap 3 (lichter pass-1-model) eerst** — 1 sessie incl. A/B tegen de gouden referentie; export/DirectML apart experiment. Stappen 1/2/6 zijn randwerk (zie de meting in fase 6) |
| 7 | Perspectiefcorrectie via baanlijnen | *nice-to-have (niet nu — frontale, horizontale camera)*; groot, 2–3 sessies (stap 3, de 3D-reconstructie, is onderzoekswerk — eerst valideren op testmateriaal) |
| 8 | Fragmenten knippen in de app + `bronvideo`/opnames in de bibliotheek | ✅ **af** (11 aug 2026) |
| — | Losse eindjes in de doelkeuze (stitch-poort, MediaPipe-klik) | **open** — code klein, de A/B is het werk; zie de eigen sectie hierboven |

## Openstaande vragen (beslissen wanneer de fase begint)

- ~~**Fase 1**: video altijd kopiëren?~~ **Besloten (jul 2026): altijd kopiëren, origineel laten staan.**
- ~~**Fase 1**: oude "losse" analyses importeerbaar?~~ **Besloten (jul 2026): niet nodig.**
- ~~**Fase 3**: ook punten kunnen bewerken op frames zónder gedetecteerde pose (punt "plaatsen" i.p.v. verslepen)?~~ **Gebouwd (31 jul 2026): ja** — begeleide klikreeks van 8 punten met voorvulling uit de buurframes, plus een dekkingsteller. Zie de aanvulling bij fase 3.
- ~~**Fase 4**: welke cloudprovider gebruikt het team feitelijk?~~ **Besloten (jul 2026): Google Drive** (Mirror-modus, dus alle bestanden lokaal op schijf). Conflictdetectie is provider-agnostisch (elk `*.db` naast `schaats.db`).
- **Fase 5** *(nice-to-have, niet nu)*: onder de huidige aanname (horizontale camera) is deze fase niet nodig. Wordt pas relevant als er tóch met een schuine/schommelende camera gefilmd gaat worden; dán ook: pant de camera mee (punt-overdracht nodig) of staat hij op statief?
- **Fase 6**: hoeveel meetafwijking is acceptabel voor het "snel"-profiel? (Voorstel: events moeten identiek blijven, hoeken mogen ±1° verschillen.) En: is een lichter pass-1-model überhaupt een *profiel*, of gewoon de nieuwe default? Als de A/B uitwijst dat `yolo26m` dezelfde dekking en events geeft, is er niets te kiezen — dan vervalt stap 5.
- **Bochtdetectie**: er is nog geen clip uit de eigen opstelling die **ín de bocht begint**; de drempels (`BOCHT_IN`/`BOCHT_UIT`) staan nu op de marge uit vier TV-clips die in de bocht éindigen. Zodra zo'n clip er is: `python schaats_eval.py bocht analyse.npz` en zo nodig bijstellen.
- **Fase 8**: ~~mag de knipdialoog een `CAP_PROP_POS_FRAMES`-seek gebruiken?~~ **Gebouwd (11 aug 2026): ja**, met een expliciete `snel_zoeken`-vlag op `VideoSpeler`; gemeten afwijking 0–1 frame op twee iPhone-.MOV's en tot 4 frames (168 ms) op een VFR-clip waar de framecount zelf al niet klopt. ~~Her-coderen met cv2 of stream-copy met ffmpeg?~~ **Besloten (11 aug 2026): cv2/`mp4v`** — zie de A/B bij fase 8. ~~Hoeveel marge rond een fragment?~~ **Besloten (10 aug 2026): geen enkele marge** — exact op de gemarkeerde frames knippen; de trainer beoordeelt de grenzen zelf tijdens het markeren. **Nog open:** mag de trainer de doelschaatser al in het knipvenster aanwijzen op een zelfgekozen frame, i.p.v. achteraf op frame 0 van de clip? Niet gebouwd; het vraagt `_kies_seed(..., doel_frame)` in de YOLO-backend plus het framenummer in `instellingen_json`, en frame 0 van een fragment is nu al precies het beeld waarop "start" gedrukt werd — dus de urgentie is klein geworden.
- **Fase 7** *(nice-to-have, niet nu)*: onder de huidige aanname (frontaal, horizontaal) is de perspectiefvertekening klein en deze fase geen prioriteit. Wordt pas relevant bij een schuin geplaatste camera; dán ook: welke baanlijnen zijn scherp genoeg om na te trekken en is hun onderlinge afstand bekend (schaal in meters — zonder schaal werkt de hoekcorrectie ook, alleen snelheid/slaglengte niet)? En staat de camera dan op statief, of moet het lijn-tracken uit fase 5 mee?
