"""
schaats_omgeving.py — waar de bestanden staan, en waar de uitvoer heen gaat.

Twee vragen die beantwoord moeten zijn vóórdat er ook maar iets anders geladen is:

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

Dit is de énige module die vóór het opstartscherm van schaats_gui.py geladen wordt en
daarom bewust **stdlib-only**: geen cv2, numpy of Qt. Dat is meteen de reden dat
`is_bevroren()`/`app_dir()`/`data_dir()` hier staan en niet meer in schaats_analyse.py
(dat cv2+numpy binnentrekt): de omleiding heeft `data_dir()` nodig op het moment dat die
imports juist nog niet mogen gebeuren. schaats_analyse.py exporteert ze door, dus
`from schaats_analyse import app_dir, data_dir, is_bevroren` blijft overal werken.

Zelftest (tempmap, geen GUI): `python schaats_omgeving.py`
"""

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

    Gebeurt alleen bevroren (dan ís er geen console) of als de env-var
    SCHAATSANALYSE_LOG gezet is — dat laatste is de testhaak, zodat deze route in de
    gewone venv te draaien is zonder eerst een exe te bouwen. Retourneert het logpad, of
    None als er niets is omgeleid, of als het bestand niet te openen was: dan gaat de
    uitvoer naar `_Stil` — uitvoer kwijt, maar géén crash, en dat laatste was hier de
    hele bedoeling.
    """
    global _ORIGINEEL
    if _ORIGINEEL is not None:                    # al omgeleid; niet nog een keer
        return logboek_pad()
    if not (forceer or is_bevroren() or os.environ.get("SCHAATSANALYSE_LOG")):
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


# ── Zelftest ────────────────────────────────────────────────────────────────────

def _zelftest():
    import shutil
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

        # 4. Geen logbestand mogelijk (hier: er staat een map in de weg) → géén crash,
        #    en print blijft veilig. Dat is de eis waar het in de exe om begonnen was.
        os.remove(pad)
        os.makedirs(os.path.join(data_dir(), LOG_NAAM), exist_ok=True)
        assert start_logboek(forceer=True) is None
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
