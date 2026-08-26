"""
schaats_omgeving.py — waar de bestanden staan, en waar de uitvoer heen gaat.

Drie vragen die beantwoord moeten zijn vóórdat er ook maar iets anders geladen is:

1. **Waar staan de bestanden?** `app_dir()` is de map met de meegeleverde bestanden (de
   modellen), `data_dir()` de schrijfbare map voor wat de app zelf aanmaakt.
2. **Waar gaat de uitvoer heen?** Een gebundelde .exe draait zonder console (PyInstaller
   `--windowed`): `sys.stdout` en `sys.stderr` zijn dan `None`. Een kále `print()`
   overleeft dat — Python slikt hem stilzwijgend — maar alles wat de stroom écht
   aanspreekt loopt stuk op `AttributeError: 'NoneType' object has no attribute 'write'`:
   `sys.stdout.write`, de tqdm-voortgangsbalk van ultralytics/rtmlib bij een download of
   een ONNX-export, en de logging-handler die ultralytics bij de import aan `sys.stdout`
   hangt. Bovendien is stilzwijgend verdwijnende uitvoer precies wat je mist als een
   collega meldt dat het niet werkt. `start_logboek()` zet er een roterend logbestand
   voor in de plaats.
3. **En als het écht misgaat?** Loggen met een `print` helpt alleen zolang Python nog
   draait. Crasht Qt of een rekenbibliotheek in C++ — zoals op 25-8-2026 gebeurde,
   `0xc0000005` in `Qt6Gui.dll`, zie TODO_CRASH.md — dan is er geen traceback en geen
   afsluitcode: het venster verdwijnt en het logboek stopt middenin. `start_crashlog()`
   hangt daarom `faulthandler` aan hetzelfde logbestand (die schrijft de C-stack
   rechtstreeks naar een filedescriptor, buiten de Python-machinerie om, en werkt dus nog
   tijdens een segfault) en vangt daarnaast onafgehandelde Python-fouten af — óók die uit
   gewone threads, waar ze nu spoorloos verdwijnen.

   **Hoe je in het logboek een échte crash herkent.** Elke sessie eindigt met
   `=== netjes afgesloten … ===`; ontbreekt die regel, dan is het proces hard gestopt.
   Dat onderscheid is nodig omdat faulthandler op Windows álle uitzonderingen met de
   ernst-bit meldt, ook de afgehandelde: sluit je het venster van buitenaf, dan staat er
   geregeld `Windows fatal exception: code 0x8001010d` (een COM-melding uit Qt) terwijl de
   app gewoon doorloopt. Een crash is dus: `access violation` of `Fatal Python error`,
   **en** geen slotregel erachter.

Dit is de énige module die vóór het opstartscherm van schaats_gui.py geladen wordt en
daarom bewust **stdlib-only**: geen cv2, numpy of Qt. Dat is meteen de reden dat
`is_bevroren()`/`app_dir()`/`data_dir()` hier staan en niet meer in schaats_analyse.py
(dat cv2+numpy binnentrekt): de omleiding heeft `data_dir()` nodig op het moment dat die
imports juist nog niet mogen gebeuren. schaats_analyse.py exporteert ze door, dus
`from schaats_analyse import app_dir, data_dir, is_bevroren` blijft overal werken.

Zelftest (tempmap, geen GUI): `python schaats_omgeving.py`
"""

import atexit
import io
import os
import sys
import threading
import time


# ── Waar staan de bestanden? ────────────────────────────────────────────────────
# Als los script liggen de modellen naast de code in de repomap, maar in een gebundelde
# .exe (PyInstaller) zit de code in een tijdelijke uitpakmap en staan de meegeleverde
# modellen naast het uitvoerbare bestand.

def is_bevroren():
    """Draait deze code uit een gebundelde .exe i.p.v. uit de losse scripts?"""
    return bool(getattr(sys, "frozen", False))


def app_dir():
    """Map met de meegeleverde bestanden (de modellen). Alleen om uit te lezen: een
    installatiemap mag read-only zijn — schrijf naar `data_dir()`."""
    if is_bevroren():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def data_dir():
    r"""Schrijfbare map voor wat de app zelf aanmaakt (%LOCALAPPDATA%\SchaatsAnalyse).

    Volgt het patroon van `config_pad()` in schaats_db.py: de env-var als die er is,
    anders de thuismap. Niet dezelfde map als de bibliotheek (die staat in de gedeelde
    Drive) en ook niet %APPDATA% (waar de bibliotheekconfig blijft staan, zodat een
    bestaande installatie zijn Drive-pad terugvindt). Aanmaken kan mislukken op een
    afgeschermde machine; dat mag hier niets slopen — de schrijfactie zelf faalt dan.
    """
    pad = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                       "SchaatsAnalyse")
    try:
        os.makedirs(pad, exist_ok=True)
    except OSError:
        pass
    return pad


# ── Waar gaat de uitvoer heen? ──────────────────────────────────────────────────

LOG_NAAM = "schaatsanalyse.log"
LOG_MAX_BYTES = 1_000_000        # daarboven: huidige log → .log.1, vers beginnen
_ORIGINEEL = None                # (stdout, stderr) van vóór de omleiding


def logboek_pad():
    """Pad van het logbestand — ook zinnig om te tonen als er iets misgaat."""
    return os.path.join(data_dir(), LOG_NAAM)


class _Stil(io.TextIOBase):
    """Uitvoer die nergens heen gaat. Beter dan `None`, want daar loopt `.write` op stuk;
    de terugval als het logbestand niet te openen is."""

    def write(self, tekst):
        return len(tekst)

    def writable(self):
        return True

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class _Logboek(io.TextIOBase):
    """stdout/stderr-vervanger die naar een logbestand schrijft en op grootte roteert.

    Regel: loggen mag de app nooit slopen. Elke schrijfactie zit daarom in een
    try/except; lukt het bestand niet (volle schijf, map ineens weg), dan gaat die
    uitvoer verloren in plaats van dat de analyse eraan onderdoor gaat — bij de volgende
    regel wordt gewoon opnieuw geprobeerd te openen.

    Regelgebufferd, zodat een crash de laatste complete regels wél op schijf achterlaat.
    """

    def __init__(self, pad, max_bytes=LOG_MAX_BYTES):
        self._pad = pad
        self._max = max(1024, int(max_bytes))
        self._slot = threading.Lock()   # de workerthreads schrijven ook (AnalyseWorker)
        self._fh = None
        self._grootte = 0
        self._open()

    def _open(self):
        try:
            self._grootte = os.path.getsize(self._pad)
        except OSError:
            self._grootte = 0
        self._fh = open(self._pad, "a", encoding="utf-8", errors="replace", buffering=1)

    def _roteer(self):
        """Huidige log → .log.1 (de vorige .1 verdwijnt), daarna vers beginnen."""
        fh, self._fh = self._fh, None
        try:
            fh.close()
        except (OSError, ValueError):
            pass
        try:
            os.replace(self._pad, self._pad + ".1")
        except OSError:
            try:
                os.remove(self._pad)
            except OSError:
                pass
        self._open()
        # De crashstroom heeft een eigen filedescriptor naar hetzelfde bestand en wijst na
        # de hernoeming nog naar .log.1. Opnieuw aanhaken, anders belandt een stack straks
        # in het vorige logboek terwijl je in het huidige zoekt.
        _herhaak_crashlog()

    def write(self, tekst):
        if not tekst:
            return 0
        with self._slot:
            try:
                if self._fh is None:
                    self._open()
                self._fh.write(tekst)
                # Tekens, geen bytes: exact op de byte roteren is de moeite niet, het
                # gaat erom dat het bestand niet ongemerkt volloopt.
                self._grootte += len(tekst)
                if self._grootte >= self._max:
                    self._roteer()
            except (OSError, ValueError):
                self._fh = None
        return len(tekst)

    def flush(self):
        with self._slot:
            try:
                if self._fh is not None:
                    self._fh.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        """Bewust géén echte close: sluit een bibliotheek per ongeluk `sys.stdout`, dan
        moet de rest van de sessie nog steeds gelogd kunnen worden."""
        self.flush()

    def writable(self):
        return True

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"

    @property
    def errors(self):
        return "replace"


def _schrijf_kop(stroom):
    """Eén blokje per start: zonder dat is achteraf niet te zien wélke sessie een fout
    gaf, en waar die zijn modellen zocht."""
    stroom.write("\n=== SchaatsAnalyse gestart %s ===\n"
                 % time.strftime("%Y-%m-%d %H:%M:%S"))
    stroom.write("    programma : %s\n" % sys.executable)
    stroom.write("    app_dir   : %s\n" % app_dir())
    stroom.write("    data_dir  : %s\n" % data_dir())
    stroom.write("    python    : %s   bevroren=%s\n"
                 % (sys.version.split()[0], is_bevroren()))


def start_logboek(forceer=False):
    """Stuurt `sys.stdout`/`sys.stderr` naar het logbestand in `data_dir()`.

    Gebeurt zodra er geen console is om naar te schrijven — bevroren, of gestart met
    `pythonw.exe` (`sys.stderr is None`) — of als de env-var SCHAATSANALYSE_LOG gezet is;
    dat laatste is de testhaak, zodat deze route in de gewone venv te draaien is zonder
    eerst een exe te bouwen. Retourneert het logpad, of
    None als er niets is omgeleid, of als het bestand niet te openen was: dan gaat de
    uitvoer naar `_Stil` — uitvoer kwijt, maar géén crash, en dat laatste was hier de
    hele bedoeling.
    """
    global _ORIGINEEL
    if _ORIGINEEL is not None:                    # al omgeleid; niet nog een keer
        return logboek_pad()
    # `sys.stderr is None` betekent: gestart met pythonw.exe — er ís geen console, net als
    # bevroren. Zonder omleiding verdwijnt dan **alle** uitvoer in het niets: elke print,
    # de Qt-meldingen, de schermwijzigingen, de ultralytics-regels. Dat is precies de
    # situatie waarin iemand zit te debuggen, en op 25-8-2026 kostte het een testsessie:
    # het logboek had wél een sessiekop (die schrijft `start_crashlog()` los) maar geen
    # enkele regel van de app, waardoor het leek alsof de diagnose-code niet draaide.
    zonder_console = sys.stderr is None or sys.stdout is None
    if not (forceer or is_bevroren() or zonder_console
            or os.environ.get("SCHAATSANALYSE_LOG")):
        return None
    pad = logboek_pad()
    try:
        stroom = _Logboek(pad)
    except OSError:
        stroom, pad = _Stil(), None
    _ORIGINEEL = (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = stroom              # één stroom: de volgorde blijft kloppen
    _schrijf_kop(stroom)
    return pad


def stop_logboek():
    """Zet stdout/stderr terug zoals ze waren. Voor de zelftest — de app zelf logt tot
    het einde van de sessie."""
    global _ORIGINEEL
    if _ORIGINEEL is None:
        return
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    sys.stdout, sys.stderr = _ORIGINEEL
    _ORIGINEEL = None


# ── En als het écht misgaat? ────────────────────────────────────────────────────
# Het logboek hierboven vangt alles op wat de app zélf schrijft, maar precies bij de
# ergste fouten schrijft ze niets meer: een access violation in Qt of onnxruntime laat
# Python niet eens aan een traceback toekomen, en in de exe is er geen console die de
# afsluitmelding toont. Twee vangnetten, allebei goedkoop en allebei naar hetzelfde
# logbestand — één pad om naar te vragen als een collega meldt dat het programma "zomaar
# weg was".

_CRASH_FH = None                 # eigen filedescriptor naar het logbestand
_OUDE_HOOKS = None               # (sys.excepthook, threading.excepthook) van vóór ons


def _crashstroom():
    """Opent het logbestand nog eens apart en hangt `faulthandler` eraan.

    Waarom een tweede bestandsobject en niet gewoon `sys.stderr`: faulthandler schrijft
    zijn stack **buiten de Python-machinerie om**, rechtstreeks naar een filedescriptor —
    dat is precies waarom het tijdens een segfault nog werkt. De `_Logboek`-wrapper is
    een Python-object zonder `fileno()` en is daar dus onbruikbaar voor. Twee schrijvers
    op één bestand in append-modus is hier geen bezwaar: de tweede schrijft alleen op het
    moment dat de eerste toch ophoudt te bestaan.
    """
    import faulthandler
    try:
        fh = open(logboek_pad(), "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return None
    try:
        faulthandler.enable(file=fh, all_threads=True)
    except (OSError, ValueError, RuntimeError):
        try:
            fh.close()
        except OSError:
            pass
        return None
    return fh


def _meld_fout(soort, waarde, sporen, herkomst):
    """Schrijft een onafgehandelde Python-fout naar het logbestand. Loggen mag nooit de
    oorzaak van een tweede fout zijn, dus alles binnen try/except."""
    import traceback
    try:
        tekst = "".join(traceback.format_exception(soort, waarde, sporen))
        _CRASH_FH.write("\n--- onafgehandelde fout in %s, %s ---\n%s"
                        % (herkomst, time.strftime("%Y-%m-%d %H:%M:%S"), tekst))
        _CRASH_FH.flush()
    except (OSError, ValueError, AttributeError, TypeError):
        pass


def _toon_ook_op_console():
    """Staat de oorspronkelijke uitvoer nog aan een console? Dan mag de standaardhook zijn
    werk doen; is stdout omgeleid, dan zou dat dezelfde traceback een tweede keer in het
    logboek zetten."""
    return _ORIGINEEL is None


def _excepthook(soort, waarde, sporen):
    _meld_fout(soort, waarde, sporen, "de hoofdthread")
    if _toon_ook_op_console() and _OUDE_HOOKS and _OUDE_HOOKS[0] is not None:
        _OUDE_HOOKS[0](soort, waarde, sporen)


def _thread_excepthook(args):
    """Fouten uit gewone threads. Zonder deze hook verdwijnen ze spoorloos — en er lopen
    er een paar: de backend-warmup van de GUI, de lokaal-proef op de opnames."""
    if args.exc_type is SystemExit:
        return
    naam = getattr(args.thread, "name", "?")
    _meld_fout(args.exc_type, args.exc_value, args.exc_traceback, "thread %r" % naam)
    if _toon_ook_op_console() and _OUDE_HOOKS and _OUDE_HOOKS[1] is not None:
        _OUDE_HOOKS[1](args)


def _afsluitregel():
    """Eén regel bij een nette afsluiting. Daarmee is het logboek zélf het antwoord op de
    vraag "is de app gecrasht of gewoon gesloten?": ontbreekt deze regel aan het eind van
    een sessie, dan is het proces hard gestopt. `atexit` draait immers niet meer als de
    boel in C++ omvalt — precies het geval dat we willen herkennen."""
    try:
        _CRASH_FH.write("=== netjes afgesloten %s ===\n"
                        % time.strftime("%Y-%m-%d %H:%M:%S"))
        _CRASH_FH.flush()
    except (OSError, ValueError, AttributeError):
        pass


def start_crashlog():
    """Legt een crash vast in het logbestand: de C-stack van een fatale fout
    (`faulthandler`) én onafgehandelde Python-fouten uit hoofd- en gewone threads.

    Retourneert het logpad, of None als het bestand niet te openen was. Anders dan
    `start_logboek()` gebeurt dit **altijd**, ook als los script: juist daar wordt
    gedebugd, en het kost één openstaand bestand. Aanroepen ná `start_logboek()`, zodat
    de sessiekop al boven de eventuele stack staat.
    """
    global _CRASH_FH, _OUDE_HOOKS
    if _CRASH_FH is not None:                     # al aan; niet nog een keer
        return logboek_pad()
    fh = _crashstroom()
    if fh is None:
        return None
    _CRASH_FH = fh
    # Zonder omleiding (los script) heeft start_logboek() niets geschreven, en dan zou een
    # stack straks zonder datum, pad of versie in het bestand staan — onbruikbaar als een
    # collega hem opstuurt. Dus hier alsnog de sessiekop.
    if _ORIGINEEL is None:
        _schrijf_kop(fh)
    # Wie dit bestand opstuurt moet het kunnen lezen zonder de broncode ernaast. Nodig,
    # want Qt levert bij het opstarten stelselmatig één afgehandelde COM-uitzondering op
    # (0x8001010d) die faulthandler tóch meldt — zonder deze regel leest elk logboek als
    # een crash.
    fh.write("    crashlog  : aan; elke sessie hoort te eindigen met "
             "'netjes afgesloten'.\n"
             "                Een 'Windows fatal exception' waar die slotregel op volgt "
             "is afgehandeld\n                en onschuldig; ontbreekt de slotregel, dan "
             "is de app daar gecrasht.\n")
    _OUDE_HOOKS = (sys.excepthook, getattr(threading, "excepthook", None))
    sys.excepthook = _excepthook
    if _OUDE_HOOKS[1] is not None:
        threading.excepthook = _thread_excepthook
    atexit.register(_afsluitregel)
    return logboek_pad()


def _herhaak_crashlog():
    """Na een rotatie wijst onze filedescriptor nog naar het hernoemde bestand; opnieuw
    openen. Lukt dat niet, dan houden we de oude — een stack in .log.1 is nog altijd beter
    dan geen stack."""
    global _CRASH_FH
    if _CRASH_FH is None:
        return
    oud = _CRASH_FH
    nieuw = _crashstroom()
    if nieuw is None:
        return
    _CRASH_FH = nieuw
    try:
        oud.close()
    except (OSError, ValueError):
        pass


def stop_crashlog():
    """Zet de hooks terug en laat het bestand los. Voor de zelftest — de app zelf houdt
    het vangnet tot het einde van de sessie gespannen."""
    global _CRASH_FH, _OUDE_HOOKS
    import faulthandler
    if _CRASH_FH is None:
        return
    atexit.unregister(_afsluitregel)
    faulthandler.disable()
    if _OUDE_HOOKS is not None:
        sys.excepthook = _OUDE_HOOKS[0]
        if _OUDE_HOOKS[1] is not None:
            threading.excepthook = _OUDE_HOOKS[1]
        _OUDE_HOOKS = None
    try:
        _CRASH_FH.close()
    except (OSError, ValueError):
        pass
    _CRASH_FH = None


# ── Zelftest ────────────────────────────────────────────────────────────────────

def _zelftest():
    import shutil
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="schaats_omgeving_")
    oud_local = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        assert data_dir() == os.path.join(tmp, "SchaatsAnalyse")
        assert os.path.isdir(data_dir()), "data_dir() maakt de map niet aan"
        assert app_dir() == os.path.dirname(os.path.abspath(__file__))
        assert not is_bevroren()

        # 1. Omleiden: alles wat op een console zou staan, staat in het bestand — ook
        #    het rechtstreekse .write dat op sys.stdout=None juist stukliep.
        pad = start_logboek(forceer=True)
        assert pad == logboek_pad() and os.path.isfile(pad), pad
        print("regel via print")
        print("regel via stderr", file=sys.stderr)
        sys.stdout.write("regel via write\n")
        assert not sys.stdout.isatty() and sys.stdout.encoding == "utf-8"
        stop_logboek()
        assert sys.stdout is sys.__stdout__, "stdout niet teruggezet"

        with open(pad, encoding="utf-8") as f:
            tekst = f.read()
        for verwacht in ("SchaatsAnalyse gestart", "data_dir  :",
                         "regel via print", "regel via stderr", "regel via write"):
            assert verwacht in tekst, f"{verwacht!r} ontbreekt in het logboek"

        # 2. Een tweede sessie hangt eraan vast in plaats van het bestand te wissen.
        start_logboek(forceer=True)
        print("tweede sessie")
        stop_logboek()
        with open(pad, encoding="utf-8") as f:
            tekst = f.read()
        assert tekst.count("SchaatsAnalyse gestart") == 2 and "regel via print" in tekst

        # 3. Roteren op grootte: de oude inhoud verhuist naar .log.1 en het logboek
        #    begint vers, zodat het nooit ongemerkt volloopt.
        rot = os.path.join(tmp, "rot.log")
        log = _Logboek(rot, max_bytes=2000)
        log.write("a" * 1500 + "\n")
        assert not os.path.exists(rot + ".1"), "te vroeg geroteerd"
        log.write("b" * 1000 + "\n")
        log.write("na de rotatie\n")
        log.flush()
        assert os.path.exists(rot + ".1"), "niet geroteerd"
        with open(rot, encoding="utf-8") as f:
            assert f.read() == "na de rotatie\n", "na rotatie niet vers begonnen"
        log.write("x" * 2500 + "\n")             # nog eens: .log.1 wordt overschreven
        log.flush()
        with open(rot + ".1", encoding="utf-8") as f:
            assert f.read().startswith("na de rotatie"), ".log.1 niet vervangen"

        # 4. Crashlog: een onafgehandelde Python-fout komt in het logboek terecht — ook
        #    die uit een gewone thread, want juist daar verdween hij spoorloos. Met het
        #    logboek aan, zodat de standaardhook hem niet óók op de console zet.
        start_logboek(forceer=True)
        assert start_crashlog() == pad
        import faulthandler
        assert faulthandler.is_enabled(), "faulthandler staat niet aan"
        try:
            raise ValueError("fout-in-hoofdthread")
        except ValueError:
            sys.excepthook(*sys.exc_info())

        def _kapotte_thread():
            raise KeyError("fout-in-thread")

        th = threading.Thread(target=_kapotte_thread, name="testthread")
        th.start()
        th.join()
        stop_crashlog()
        stop_logboek()
        assert not faulthandler.is_enabled(), "faulthandler niet uitgezet"
        assert sys.excepthook is sys.__excepthook__, "excepthook niet teruggezet"
        with open(pad, encoding="utf-8") as f:
            tekst = f.read()
        for verwacht in ("fout-in-hoofdthread", "ValueError",
                         "fout-in-thread", "testthread"):
            assert verwacht in tekst, f"{verwacht!r} ontbreekt in het logboek"

        # 5. En het geval waarvoor faulthandler er echt is: een access violation, precies
        #    de crash van 25-8-2026 (0xc0000005 in Qt6Gui.dll). Python komt dan niet meer
        #    aan een traceback toe, dus dit valt alleen in een apart proces te toetsen.
        code = ("import sys; sys.path.insert(0, %r)\n"
                "import schaats_omgeving as o; o.start_crashlog()\n"
                "import faulthandler; faulthandler._read_null()\n"
                % os.path.dirname(os.path.abspath(__file__)))
        # Eigen map: Windows Error Reporting kan het gecrashte proces nog even vasthouden,
        # en dan is het logbestand hierboven niet meer te verwijderen in het volgende punt.
        apart = os.path.join(tmp, "crash")
        sub = subprocess.run([sys.executable, "-c", code],
                             env=dict(os.environ, LOCALAPPDATA=apart),
                             capture_output=True, text=True)
        assert sub.returncode != 0, "het testproces crashte niet eens"
        with open(os.path.join(apart, "SchaatsAnalyse", LOG_NAAM), encoding="utf-8") as f:
            tekst = f.read()
        assert ("Windows fatal exception" in tekst or "Fatal Python error" in tekst),             "geen crash-stack in het logboek"
        assert 'File "<string>"' in tekst, "de stack mist de Python-frames"
        assert "access violation" in tekst, "niet als access violation herkend"
        assert "=== netjes afgesloten" not in tekst, "een crash mag niet als nette stop gelden"

        #    Andersom hoort een gewone afsluiting die regel juist wél te krijgen: dát is
        #    wat het logboek achteraf leesbaar maakt — geen slotregel = hard gestopt.
        netjes = os.path.join(tmp, "netjes")
        sub = subprocess.run(
            [sys.executable, "-c", code.replace("faulthandler; faulthandler._read_null()",
                                                "sys; sys.exit(0)")],
            env=dict(os.environ, LOCALAPPDATA=netjes), capture_output=True, text=True)
        assert sub.returncode == 0, sub.stderr
        with open(os.path.join(netjes, "SchaatsAnalyse", LOG_NAAM), encoding="utf-8") as f:
            assert "=== netjes afgesloten" in f.read(), "slotregel ontbreekt na een nette stop"

        # 6. Geen logbestand mogelijk (er staat een map met die naam in de weg) → géén
        #    crash, en print blijft veilig. Dat is de eis waar het in de exe om begonnen
        #    was. In een vérse map, want het logboek van punt 4 houdt zijn bestand nog
        #    open: `_Logboek.close()` flusht bewust alleen, zodat een bibliotheek die
        #    per ongeluk sys.stdout sluit de rest van de sessie niet blind maakt.
        os.environ["LOCALAPPDATA"] = os.path.join(tmp, "geenlog")
        os.makedirs(os.path.join(data_dir(), LOG_NAAM), exist_ok=True)
        assert start_logboek(forceer=True) is None
        assert start_crashlog() is None, "crashlog moet ook zonder bestand overleven"
        print("dit verdwijnt")
        sys.stdout.write("dit ook\n")
        stop_logboek()
        assert sys.stdout is sys.__stdout__

        print("Zelftest OK")
    finally:
        stop_logboek()
        if oud_local is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = oud_local
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _zelftest()
