# SchaatsAnalyse installeren

Deze handleiding is voor de trainers. Je hebt geen Python, geen programmeerkennis en geen
beheerdersrechten nodig — alleen een Windows-laptop en de gedeelde Google Drive-map.

**Je moet één keer door een waarschuwing van Windows heen klikken** ("Windows heeft uw pc
beschermd"). Dat is normaal en het staat hieronder uitgelegd bij stap 3. Er zit niets fout in
het programma; het is alleen niet ondertekend met een betaald certificaat.

---

## Wat je nodig hebt

| | |
|---|---|
| Windows | 10 of 11, 64-bits |
| Vrije schijfruimte | ~1,5 GB voor het programma (de video's staan in Drive) |
| Google Drive voor desktop | geïnstalleerd, en toegang tot de gedeelde map `SchaatsAnalyse` |
| Internet | alleen om te downloaden — daarna werkt het programma offline |

De installatie bevat alles: er wordt bij het eerste gebruik **niets** nagedownload.

---

## 1. Downloaden

De installatie staat in de gedeelde Drive-map:

```
Mijn Drive\SchaatsAnalyse\app\SchaatsAnalyse-setup.exe
```

Het bestand is groot (**ongeveer 650 MB**). Klik er in de Verkenner met de **rechtermuisknop** op,
kies **Offline beschikbaar maken** (of sleep het naar je bureaublad) en wacht tot het groene
vinkje verschijnt. Start de setup pas dán — rechtstreeks vanaf de Drive-schijf starten gaat
traag en kan halverwege afbreken.

## 2. De waarschuwing van je browser (als je via een link downloadt)

Edge of Chrome kan zeggen dat het bestand *"niet vaak wordt gedownload"* of *"schadelijk kan
zijn"*. Dat is een statistische waarschuwing: nieuwe programma's die maar door een handjevol
mensen worden gebruikt, krijgen die altijd. Kies **Behouden** (in Edge: `...` → **Behouden** →
**Toch behouden**).

## 3. "Windows heeft uw pc beschermd" — SmartScreen

Bij het starten van de setup verschijnt een blauw venster:

> **Windows heeft uw pc beschermd**
> Microsoft Defender SmartScreen heeft voorkomen dat een onbekende app is gestart.

Klik op **Meer informatie** en daarna op de knop **Toch uitvoeren**.

**Waarom dit gebeurt:** software wordt "bekend" bij Microsoft door een
code-ondertekeningscertificaat (€200–400 per jaar) of doordat duizenden mensen hem
downloaden. Voor een programma dat door een paar trainers gebruikt wordt is dat certificaat de
moeite niet — dus meldt Windows het als onbekend. Onbekend is niet hetzelfde als onveilig.

**Zet nooit je virusscanner uit.** Als Windows Defender het bestand alsnog in quarantaine
zet — het gebeurt zelden, maar dit soort verpakte Python-programma's wordt af en toe ten
onrechte aangemerkt — doe dan dit:

1. Start → **Beveiliging van Windows** → **Virus- en bedreigingsbeveiliging**
2. **Beveiligingsgeschiedenis** → het item over SchaatsAnalyse
3. **Acties** → **Toestaan op apparaat**

Twijfel je? Bel of app eerst even. Beter een dag wachten dan iets doen wat je niet vertrouwt.

## 4. Installeren

Dubbelklik op `SchaatsAnalyse-setup.exe` en volg de stappen. Er wordt **niet** om een
beheerderswachtwoord gevraagd.

- Het programma komt in je eigen profiel te staan
  (`C:\Users\<jouw naam>\AppData\Local\Programs\SchaatsAnalyse`)
- Grootte na installatie: ~1,3 GB
- Duur: ongeveer een minuut
- Vink **Snelkoppeling op het bureaublad** aan als je die wilt

## 5. De eerste start

Start **SchaatsAnalyse**. Er verschijnt eerst een klein opstartschermpje; het hoofdvenster
komt een paar seconden later. (De allereerste keer kan het langer duren, omdat de
virusscanner alle bestanden één keer nakijkt.)

Doe daarna twee dingen op de startpagina:

1. **Jouw naam...** — vul je naam in. Die komt bij elke analyse die jij maakt te staan, zodat
   in de gedeelde bibliotheek te zien is wie wat gedaan heeft.
2. **Bibliotheekmap...** — kies de gedeelde map in Drive:
   `G:\Mijn Drive\SchaatsAnalyse` (de schijfletter kan bij jou anders zijn; kijk in de
   Verkenner onder *Google Drive*).

   Dit is de belangrijkste stap. In die map staan de opnames, alle analyses en de database
   die jullie delen. Kies je hier je eigen Documenten-map, dan werk je in je eentje en ziet
   niemand je analyses.

Beide instellingen worden onthouden; je hoeft dit maar één keer te doen.

## 6. Zet de Drive-map offline beschikbaar

Dit is geen luxe maar noodzaak. Google Drive haalt bestanden standaard per stukje van
internet op. Gemeten op een opname van 4 GB: één sprong in het knipvenster kost dan **5 tot
45 seconden**, tegen een tiende seconde als het bestand op je laptop staat.

In de Verkenner: rechtermuisknop op de map **SchaatsAnalyse** → **Google Drive** → **Offline
beschikbaar maken**. Laat hem daarna rustig doorlopen (reken op ongeveer tien minuten per
opname van 4 GB). Heb je weinig schijfruimte, doe dan in elk geval de submap **`opnames`**, of
alleen de opname waar je die dag mee werkt.

De app waarschuwt je trouwens zelf: staat een opname niet op je pc, dan zegt de kolom **"Op
deze pc"** dat, en krijg je bij het openen een melding met deze instructie erbij.

---

## Video kijken: overal dezelfde toetsen

Of je nu een ruwe opname bekijkt, fragmenten knipt, een analyse terugkijkt of twee schaatsers
naast elkaar zet — de bediening is overal hetzelfde:

| Toets | Wat het doet |
| --- | --- |
| **spatie** | afspelen / pauze |
| **.** (punt) | doorspoelen op 6× zolang je hem ingedrukt houdt |
| **,** (komma) | terugspoelen op 6× |
| **← →** | één frame terug / verder |
| **Home / End** | naar het begin / het eind |
| **F11** | volledig scherm aan of uit |
| **muiswiel** | in- en uitzoomen op het beeld |

Eén tikje op de punt of komma schuift precies één frame op; vasthouden spoelt door. De toetsen
staan ook onder in beeld en in de tips bij de knoppen, dus je hoeft ze niet uit je hoofd te
leren.

Per venster komen daar de knoppen bij die alleen daar bestaan: **P** zet een punt in een
opname (**1**–**9** springt erheen, **Del** haalt hem weg), en in het knipvenster markeren
**S** en **E** het begin en het eind van een fragment.

---

## Met z'n tweeën in dezelfde bibliotheek

- Klik op **Vernieuwen** om te zien wat een collega intussen heeft toegevoegd — dat gaat niet
  vanzelf terwijl de app openstaat.
- Werk niet tegelijk aan dezelfde analyse. Er is geen slot: wie het laatst opslaat, wint.
- Ziet de app een tweede databasebestand naast `schaats.db` staan, dan meldt hij dat. Dat is
  een conflictkopie van Drive (twee mensen schreven bijna tegelijk). Laat het bestand staan en
  meld het even — `schaats.db` blijft gewoon de echte bibliotheek.

## Hoe lang duurt een analyse?

Ongeveer **één seconde per frame**, dus een fragment van vijf seconden (150 frames) kost een
paar minuten. Het programma gebruikt automatisch de grafische kaart van je laptop (elke
moderne Intel-, AMD- of NVIDIA-kaart); lukt dat niet, dan rekent het op de processor en duurt
het ongeveer twee keer zo lang. Je hoeft daar niets voor in te stellen.

De bocht wordt trouwens grotendeels overgeslagen — daar valt technisch niets te meten. Dat
scheelt op een lange opname bijna de helft van de tijd.

## Bekende hebbelijkheid

**Maak geen schermafbeelding (Win+Shift+S) terwijl er een analyse loopt.** Het programma kan
daardoor afsluiten. Je bent dan de analyse kwijt die op dat moment liep — bij een batch alleen
díe ene clip: elke video wordt apart opgeslagen, dus alles wat daarvóór klaar was staat gewoon
in de bibliotheek. Even wachten tot hij klaar is is de oplossing; hier wordt aan gewerkt.

---

## Als er iets misgaat

Het programma schrijft mee in een logboek. Dat is precies wat nodig is om "bij mij doet-ie het
niet" op te lossen.

**Het logboek openen:** druk op `Windows + R`, plak dit erin en druk op Enter:

```
%LOCALAPPDATA%\SchaatsAnalyse
```

Daar staat `schaatsanalyse.log`. Open hem met Kladblok, of stuur het bestand gewoon door. Elke
start van de app zet er een kopblok in, en elke nette afsluiting eindigt met een regel
`=== netjes afgesloten ... ===`. **Ontbreekt die regel achter je laatste sessie, dan is de app
gecrasht** — dat is meteen het bewijs dat er iets te onderzoeken valt.

Vermeld er bij het doorsturen bij:

- wat je aan het doen was (welke opname, welke knop)
- hoe laat het ongeveer was — daarmee is de juiste plek in het logboek terug te vinden
- of het opnieuw gebeurt als je het nog eens probeert

## Bijwerken naar een nieuwe versie

Download de nieuwe `SchaatsAnalyse-setup.exe` uit dezelfde Drive-map en voer hem uit. Hij
installeert over de oude heen; je naam, je bibliotheekmap en alle analyses blijven staan.

Welke versie je hebt zie je in **Instellingen → Apps → Geïnstalleerde apps** achter
SchaatsAnalyse (bijvoorbeeld `2026-08-25.a4be1f2b`), en per analyse onder de knop
**ℹ Info...**. Handig als er iets is opgelost: dan is te zien of jouw analyse nog met de oude
versie gemaakt is.

## Verwijderen

**Instellingen → Apps → Geïnstalleerde apps → SchaatsAnalyse → Verwijderen.** Je gedeelde
bibliotheek in Drive en het logboek blijven ongemoeid — er gaat geen enkele opname of analyse
verloren.
