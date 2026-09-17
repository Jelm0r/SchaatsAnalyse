# Installing SkateAnalysis

This guide is for the trainers. You don't need Python, programming knowledge, or admin
rights — just a Windows laptop and the shared Google Drive folder.

**You'll need to click through one Windows warning** ("Windows protected your PC" /
Dutch: "Windows heeft uw pc beschermd"). That's normal and explained below under step
3. There's nothing wrong with the program; it just isn't signed with a paid
certificate.

---

## What you need

| | |
|---|---|
| Windows | 10 or 11, 64-bit |
| Free disk space | ~1.5 GB for the program (the videos live in Drive) |
| Google Drive | access to the shared `SkateAnalysis` folder (via the Drive app on your PC, or via drive.google.com) |
| Internet | only to download — after that the program works offline |

The install contains everything: **nothing** gets downloaded afterward on first use.

---

## 1. Downloading

The installer sits in the shared Drive folder:

```
Mijn Drive\SkateAnalysis\app\SkateAnalysis-setup.exe
```

The file is large (**about 650 MB**). How you download it depends on how you reach
Drive:

- **Is Google Drive installed on your PC** (the folder shows up as a plain drive in
  your File Explorer, e.g. as `G:`)? Right-click the file, choose **Offline beschikbaar
  maken** (Make available offline) (or drag it to your desktop) and wait for the green
  checkmark. Only start the setup after that.
- **Are you in the Drive website (drive.google.com) or the Drive app**, without the
  folder showing up as a drive in File Explorer? Then "Make available offline" doesn't
  exist there. Just click **Downloaden** (Download) instead — the browser fetches the
  whole file in one go into your **Downloads** folder, which works exactly as well for
  this purpose. Wait for your browser's download to finish and start the setup from
  **Downloads**.

The point: the setup needs to be **fully on your own drive** before you start it.
Double-clicking the file directly while it's still being streamed from Drive (i.e.
without "Make available offline" and without a full download) is slow and can fail
partway through.

## 2. Your browser's warning (if you download via a link)

Edge or Chrome may say the file *"isn't downloaded often"* or *"could be harmful"*.
That's a statistical warning: new programs used by only a handful of people always get
it. Choose **Keep** (in Edge: `...` → **Behouden** (Keep) → **Toch behouden** (Keep
anyway)).

## 3. "Windows protected your PC" — SmartScreen

When you start the setup, a blue window appears:

> **Windows heeft uw pc beschermd**
> Microsoft Defender SmartScreen heeft voorkomen dat een onbekende app is gestart.
>
> (English: "Windows protected your PC" / "Microsoft Defender SmartScreen prevented
> an unrecognized app from starting.")

Click **Meer informatie** (More info) and then the **Toch uitvoeren** (Run anyway)
button.

**Why this happens:** software becomes "known" to Microsoft either through a
code-signing certificate (€200-400 a year) or because thousands of people download it.
For a program used by a handful of trainers, that certificate isn't worth it — so
Windows flags it as unknown. Unknown isn't the same as unsafe.

**Never turn off your antivirus.** If Windows Defender quarantines the file anyway —
it rarely happens, but packaged Python programs like this one occasionally get flagged
wrongly — do this instead:

1. Start → **Beveiliging van Windows** (Windows Security) → **Virus- en
   bedreigingsbeveiliging** (Virus & threat protection)
2. **Beveiligingsgeschiedenis** (Protection history) → the item about SkateAnalysis
3. **Acties** (Actions) → **Toestaan op apparaat** (Allow on device)

Not sure? Call or message first. Better to wait a day than to do something you don't
trust.

## 4. Installing

Double-click `SkateAnalysis-setup.exe` and follow the steps. You will **not** be asked
for an administrator password.

- The program is installed in your own profile
  (`C:\Users\<your name>\AppData\Local\Programs\SkateAnalysis`)
- Size after install: ~1.3 GB
- Takes about a minute
- Check **Snelkoppeling op het bureaublad** (Create a desktop shortcut) if you want one

## 5. First start

Start **SkateAnalysis**. A small splash screen appears first; the main window follows
a few seconds later. (The very first time can take longer, since the antivirus scans
every file once.)

Then do two things on the start page:

1. **Your name...** — enter your name. It's attached to every analysis you create, so
   the shared library shows who did what.
2. **Library folder...** — pick the shared folder in Drive:
   `G:\Mijn Drive\SkateAnalysis` (the drive letter may differ for you; check File
   Explorer under *Google Drive*).

   This is the most important step. That folder holds the recordings, all the
   analyses, and the database you share as a team. Pick your own Documents folder here
   instead and you'll be working alone, with nobody able to see your analyses.

Both settings are remembered; you only need to do this once.

## 6. Make the Drive folder available offline

This isn't a luxury, it's a necessity. By default, Google Drive fetches files from the
internet piece by piece. Measured on a 4 GB recording: one jump in the trim window
then costs **5 to 45 seconds**, against a tenth of a second when the file is on your
laptop.

In File Explorer: right-click the **SkateAnalysis** folder → **Google Drive** →
**Offline beschikbaar maken** (Make available offline). Then let it run in the
background (budget about ten minutes per 4 GB recording). Short on disk space? At
least do the **`opnames`** (recordings) subfolder, or just the recording you're
working with that day.

The app warns you about this itself, too: if a recording isn't on your PC, the **"On
this PC"** column says so, and opening it shows a notice with this same instruction.

---

## Copying recordings from the camera into the library

You no longer need File Explorer for that. On the **Recordings** tab: **📥 From camera
to library...** → pick the recordings from the memory card or camera (multiple at
once is fine; on a camcorder they usually sit in
`PRIVATE\AVCHD\BDMV\STREAM`, files like `00005.MTS`). The app copies them into the
library's `opnames` folder and shows progress, speed, and time remaining. Budget about
a minute per 4 GB from a fast memory card; to a Drive folder it can take longer.

After that they're in the list, and Google Drive uploads them to the shared drive on
its own — colleagues see them as soon as that upload finishes. A recording that's
already there never gets overwritten: if a new recording happens to have the same name
as an old one, rename it on the camera first. You can always **stop** partway through;
whatever finished copying stays.

---

## Watching video: the same keys everywhere

Whether you're viewing a raw recording, trimming fragments, reviewing an analysis, or
putting two skaters side by side — the controls are the same everywhere:

| Key | What it does |
| --- | --- |
| **space** | play / pause |
| **.** (period) | scrub forward at 6× for as long as you hold it |
| **,** (comma) | scrub backward at 6× |
| **← →** | one frame back / forward |
| **Home / End** | jump to the start / end |
| **F11** | toggle full screen |
| **mouse wheel** | zoom in and out on the picture |

A single tap of period or comma moves exactly one frame; holding it down scrubs
continuously. The keys are also shown at the bottom of the window and in the button
tooltips, so you don't need to memorize them.

Each window adds the buttons that only exist there: **P** drops a point in a recording
(**1**-**9** jumps to it, **Del** removes it), and in the trim window **S** and **E**
mark the start and end of a fragment.

---

## Two people in the same library

- Click **Refresh** to see what a colleague has added in the meantime — that doesn't
  happen automatically while the app is open.
- Don't work on the same analysis at the same time. There's no lock: whoever saves
  last wins.
- If the app sees a second database file next to `schaats.db`, it will tell you.
  That's a conflict copy from Drive (two people wrote at nearly the same time). Leave
  the file where it is and mention it — `schaats.db` stays the real library.

## How long does an analysis take?

Roughly **one second per frame**, so a five-second fragment (150 frames) takes a
couple of minutes. The program automatically uses your laptop's graphics card (any
modern Intel, AMD, or NVIDIA card); if that's not possible, it falls back to the
processor and takes about twice as long. There's nothing you need to configure for
this.

The corner also gets mostly skipped, by the way — there's technically nothing to
measure there. On a long recording that saves nearly half the time.

## Is the skater small in the frame?

A skater who starts far away is often too small for detection in the first few
seconds: the skeleton then only appears once they're closer, and on some footage there
are also gaps without a skeleton along the way. There's something you can do about
this when pointing out the skater: **drag a box around them** instead of clicking on
them. The program then follows them starting from that box, even where detection
itself doesn't see them yet.

First zoom in with the **mouse wheel** (around the cursor) and pan the view with the
**right mouse button**, so you can draw the box reasonably tightly around the skater —
head to skates. Then click **Follow this box**. A plain click still works too; the
box is only needed when the skater is small.

There is a lower limit, though. If the skater is smaller than about **70 pixels** on
the first frame (on a 1080p picture: smaller than a fingernail on the screen), there
simply aren't any legs left to measure — a box won't help there. The program will
warn you in that case. Start the fragment later instead, at the point where the skater
is bigger in the frame; a few-second fragment from that moment on is worth more than a
long fragment where they're still far away.

## Known quirk

**Don't take a screenshot (Win+Shift+S) while an analysis is running.** It can cause
the program to close. You'll then lose the analysis that was running at that moment —
in a batch, only that one clip: every video is saved separately, so anything finished
before that is still safely in the library. Waiting until it's done is the workaround;
this is being worked on.

---

## If something goes wrong

The program keeps a log file as it runs. That's exactly what's needed to solve an "it
doesn't work on my end" report.

**Opening the log:** press `Windows + R`, paste this in, and press Enter:

```
%LOCALAPPDATA%\SkateAnalysis
```

You'll find `skateanalysis.log` there. Open it with Notepad, or just forward the file
as-is. Every app start adds a header block, and every clean shutdown ends with a line
`=== cleanly closed ... ===`. **If that line is missing after your last session, the
app crashed** — that's immediate proof there's something to investigate.

When you forward it, please mention:

- what you were doing (which recording, which button)
- roughly what time it was — that makes the right spot in the log easy to find
- whether it happens again if you try once more

## Updating to a new version

Download the new `SkateAnalysis-setup.exe` from the same Drive folder and run it. It
installs over the old one; your name, your library folder, and every analysis stay
exactly where they are.

You can see which version you have under **Instellingen → Apps → Geïnstalleerde apps**
(Settings → Apps → Installed apps), next to SkateAnalysis (e.g. `2026-08-25.a4be1f2b`),
and per analysis under the **ℹ Info...** button. Handy when something's been fixed:
you can then check whether your analysis was still made with the old version.

## Uninstalling

**Instellingen → Apps → Geïnstalleerde apps → SkateAnalysis → Verwijderen** (Settings →
Apps → Installed apps → SkateAnalysis → Uninstall). Your shared library in Drive and
the log file are left untouched — no recording or analysis is ever lost.
