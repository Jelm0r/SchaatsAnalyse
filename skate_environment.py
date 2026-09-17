"""
skate_environment.py — where the files live, and where the output goes.

Three questions that must be answered before anything else is loaded:

1. **Where do the files live?** `app_dir()` is the folder with the bundled files (the
   models), `data_dir()` the writable folder for what the app creates itself.
2. **Where does the output go?** A bundled .exe runs without a console (PyInstaller
   `--windowed`): `sys.stdout` and `sys.stderr` are then `None`. A bare `print()`
   survives that — Python swallows it silently — but anything that actually touches
   the stream breaks with `AttributeError: 'NoneType' object has no attribute 'write'`:
   `sys.stdout.write`, the tqdm progress bar from ultralytics/rtmlib during a download or
   an ONNX export, and the logging handler ultralytics attaches to `sys.stdout` on
   import. And output silently disappearing is exactly what you miss when a colleague
   reports that it doesn't work. `start_log()` puts a rotating log file in its place.
3. **And if it really goes wrong?** Logging with a `print` only helps while Python is
   still running. If Qt or a numerical library crashes in C++ — as happened on
   2026-08-25, `0xc0000005` in `Qt6Gui.dll`, see TODO_CRASH.md — there's no traceback
   and no exit code: the window disappears and the log stops mid-session.
   `start_crashlog()` therefore attaches `faulthandler` to that same log file (it writes
   the C stack straight to a file descriptor, outside the Python machinery, so it still
   works during a segfault) and additionally catches unhandled Python errors — including
   ones from plain threads, where they currently vanish without a trace.

   **How to recognize a real crash in the log.** Every session ends with
   `=== cleanly closed … ===`; if that line is missing, the process was hard-stopped.
   That distinction is needed because faulthandler on Windows reports *all* exceptions
   with the severity bit set, including handled ones: close the window from outside and
   you'll regularly see `Windows fatal exception: code 0x8001010d` (a COM message from
   Qt) while the app keeps running just fine. So a crash is: `access violation` or
   `Fatal Python error`, **and** no closing line after it.

This is the only module loaded before skate_gui.py's splash screen, and is therefore
deliberately **stdlib-only**: no cv2, numpy, or Qt. That's exactly why
`is_frozen()`/`app_dir()`/`data_dir()` live here instead of in skate_analysis.py (which
pulls in cv2+numpy): the redirection needs `data_dir()` at the exact moment those
imports must not happen yet. skate_analysis.py re-exports them, so
`from skate_analysis import app_dir, data_dir, is_frozen` keeps working everywhere.

Self-test (temp folder, no GUI): `python skate_environment.py`
"""

import atexit
import io
import os
import sys
import threading
import time


# ── Where do the files live? ────────────────────────────────────────────────────
# As a standalone script the models sit next to the code in the repo folder, but in a
# bundled .exe (PyInstaller) the code lives in a temporary extraction folder and the
# bundled models sit next to the executable.

def is_frozen():
    """Is this code running from a bundled .exe rather than the loose scripts?"""
    return bool(getattr(sys, "frozen", False))


def app_dir():
    """Folder with the bundled files (the models). Read-only: an install folder may be
    read-only — write to `data_dir()` instead."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def data_dir():
    r"""Writable folder for what the app creates itself (%LOCALAPPDATA%\SkateAnalysis).

    Follows the pattern of `config_path()` in skate_db.py: the env var if it's set,
    otherwise the home folder. Not the same folder as the library (that lives in the
    shared Drive) and not %APPDATA% either (where the library config stays, so an
    existing install finds its Drive path again). Creating it can fail on a locked-down
    machine; that must not break anything here — only the write itself fails.

    One-time migration: earlier versions used the folder name "SchaatsAnalyse" (the
    product's old Dutch name). If that folder still exists and the new one doesn't yet,
    it's renamed in place so an existing install keeps its logs, crash log, and local
    library instead of appearing to start from scratch after an update.
    """
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "SkateAnalysis")
    if not os.path.isdir(path):
        old_path = os.path.join(base, "SchaatsAnalyse")
        if os.path.isdir(old_path):
            try:
                os.rename(old_path, path)
            except OSError:
                pass
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


# ── Where does the output go? ──────────────────────────────────────────────────

LOG_NAME = "skateanalysis.log"
LOG_MAX_BYTES = 1_000_000        # above this: current log → .log.1, start fresh
_ORIGINAL = None                 # (stdout, stderr) from before the redirection


def log_path():
    """Path of the log file — also worth showing when something goes wrong."""
    return os.path.join(data_dir(), LOG_NAME)


class _Silent(io.TextIOBase):
    """Output that goes nowhere. Better than `None`, since `.write` breaks on that;
    the fallback when the log file can't be opened."""

    def write(self, text):
        return len(text)

    def writable(self):
        return True

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class _LogFile(io.TextIOBase):
    """stdout/stderr replacement that writes to a log file and rotates by size.

    Rule: logging must never break the app. Every write therefore sits in a
    try/except; if the file doesn't cooperate (disk full, folder suddenly gone), that
    output is lost instead of taking the analysis down with it — the next line just
    tries to reopen it.

    Line-buffered, so a crash still leaves the last complete lines on disk.
    """

    def __init__(self, path, max_bytes=LOG_MAX_BYTES):
        self._path = path
        self._max = max(1024, int(max_bytes))
        self._lock = threading.Lock()   # worker threads write too (AnalysisWorker)
        self._fh = None
        self._size = 0
        self._open()

    def _open(self):
        try:
            self._size = os.path.getsize(self._path)
        except OSError:
            self._size = 0
        self._fh = open(self._path, "a", encoding="utf-8", errors="replace", buffering=1)

    def _rotate(self):
        """Current log → .log.1 (the previous .1 disappears), then start fresh."""
        fh, self._fh = self._fh, None
        try:
            fh.close()
        except (OSError, ValueError):
            pass
        try:
            os.replace(self._path, self._path + ".1")
        except OSError:
            try:
                os.remove(self._path)
            except OSError:
                pass
        self._open()
        # The crash stream has its own file descriptor to the same file and still
        # points at .log.1 after the rename. Reattach it, or a stack trace would end up
        # in the previous log while you're looking at the current one.
        _reopen_crashlog()

    def write(self, text):
        if not text:
            return 0
        with self._lock:
            try:
                if self._fh is None:
                    self._open()
                self._fh.write(text)
                # Characters, not bytes: rotating on the exact byte isn't worth the
                # trouble, the point is just that the file doesn't fill up unnoticed.
                self._size += len(text)
                if self._size >= self._max:
                    self._rotate()
            except (OSError, ValueError):
                self._fh = None
        return len(text)

    def flush(self):
        with self._lock:
            try:
                if self._fh is not None:
                    self._fh.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        """Deliberately NOT a real close: if a library closes `sys.stdout` by accident,
        the rest of the session still needs to be logged."""
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


def _write_header(stream):
    """One block per start: without it there's no way afterwards to tell which session
    produced an error, or where it looked for its models."""
    stream.write("\n=== SkateAnalysis started %s ===\n"
                 % time.strftime("%Y-%m-%d %H:%M:%S"))
    stream.write("    program   : %s\n" % sys.executable)
    stream.write("    app_dir   : %s\n" % app_dir())
    stream.write("    data_dir  : %s\n" % data_dir())
    stream.write("    python    : %s   frozen=%s\n"
                 % (sys.version.split()[0], is_frozen()))


def start_log(force=False):
    """Redirects `sys.stdout`/`sys.stderr` to the log file in `data_dir()`.

    Happens as soon as there's no console to write to — frozen, or started with
    `pythonw.exe` (`sys.stderr is None`) — or if the env var SKATEANALYSIS_LOG is set;
    the latter is the test hook, so this path can be run in the normal venv without
    building an exe first. Returns the log path, or None if nothing was redirected, or
    if the file couldn't be opened: then output goes to `_Silent` — output lost, but no
    crash, and that was the entire point.
    """
    global _ORIGINAL
    if _ORIGINAL is not None:                      # already redirected; don't do it twice
        return log_path()
    # `sys.stderr is None` means: started with pythonw.exe — there is no console, just
    # like frozen. Without redirection **all** output then vanishes into nothing: every
    # print, the Qt messages, the screen changes, the ultralytics lines. That's exactly
    # the situation someone is debugging in, and on 2026-08-25 it cost a whole test
    # session: the log did get a session header (written by `start_crashlog()`
    # independently) but not a single line from the app, making it look like the
    # diagnostic code wasn't even running.
    no_console = sys.stderr is None or sys.stdout is None
    if not (force or is_frozen() or no_console
            or os.environ.get("SKATEANALYSIS_LOG")):
        return None
    path = log_path()
    try:
        stream = _LogFile(path)
    except OSError:
        stream, path = _Silent(), None
    _ORIGINAL = (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = stream                # one stream: order stays correct
    _write_header(stream)
    return path


def stop_log():
    """Restores stdout/stderr to what they were. For the self-test — the app itself
    keeps logging until the end of the session."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    sys.stdout, sys.stderr = _ORIGINAL
    _ORIGINAL = None


# ── And if it really goes wrong? ────────────────────────────────────────────────
# The log above catches everything the app writes itself, but on exactly the worst
# errors it writes nothing at all: an access violation in Qt or onnxruntime doesn't
# even let Python get to a traceback, and in the exe there's no console to show the
# closing message. Two safety nets, both cheap and both going to the same log file —
# one path to ask for when a colleague reports the program "just disappeared".

_CRASH_FH = None                 # our own file descriptor to the log file
_OLD_HOOKS = None                # (sys.excepthook, threading.excepthook) from before us


def _crash_stream():
    """Opens the log file again, separately, and attaches `faulthandler` to it.

    Why a second file object and not just `sys.stderr`: faulthandler writes its stack
    **outside the Python machinery**, straight to a file descriptor — that's exactly
    why it still works during a segfault. The `_LogFile` wrapper is a Python object
    with no `fileno()` and is useless for that. Two writers to one file in append mode
    is not a problem here: the second only writes at the point where the first has
    already stopped existing anyway.
    """
    import faulthandler
    try:
        fh = open(log_path(), "a", encoding="utf-8", errors="replace", buffering=1)
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


def _report_error(kind, value, trace, origin):
    """Writes an unhandled Python error to the log file. Logging must never be the
    cause of a second error, so everything sits in a try/except."""
    import traceback
    try:
        text = "".join(traceback.format_exception(kind, value, trace))
        _CRASH_FH.write("\n--- unhandled error in %s, %s ---\n%s"
                        % (origin, time.strftime("%Y-%m-%d %H:%M:%S"), text))
        _CRASH_FH.flush()
    except (OSError, ValueError, AttributeError, TypeError):
        pass


def _also_show_on_console():
    """Is the original output still attached to a console? Then the default hook may
    do its job; if stdout is redirected, that would put the same traceback into the
    log a second time."""
    return _ORIGINAL is None


def _excepthook(kind, value, trace):
    _report_error(kind, value, trace, "the main thread")
    if _also_show_on_console() and _OLD_HOOKS and _OLD_HOOKS[0] is not None:
        _OLD_HOOKS[0](kind, value, trace)


def _thread_excepthook(args):
    """Errors from plain threads. Without this hook they vanish without a trace — and
    there are a few: the GUI's backend warmup, the local-file probe on recordings."""
    if args.exc_type is SystemExit:
        return
    name = getattr(args.thread, "name", "?")
    _report_error(args.exc_type, args.exc_value, args.exc_traceback, "thread %r" % name)
    if _also_show_on_console() and _OLD_HOOKS and _OLD_HOOKS[1] is not None:
        _OLD_HOOKS[1](args)


def _exit_line():
    """One line on a clean shutdown. That makes the log file itself the answer to "did
    the app crash or just close?": if this line is missing at the end of a session, the
    process was hard-stopped. `atexit` doesn't run any more once things collapse in
    C++ — exactly the case we want to recognize."""
    try:
        _CRASH_FH.write("=== cleanly closed %s ===\n"
                        % time.strftime("%Y-%m-%d %H:%M:%S"))
        _CRASH_FH.flush()
    except (OSError, ValueError, AttributeError):
        pass


def start_crashlog():
    """Records a crash in the log file: the C stack of a fatal error (`faulthandler`)
    and unhandled Python errors from both the main thread and plain threads.

    Returns the log path, or None if the file couldn't be opened. Unlike
    `start_log()`, this always runs, even as a loose script: that's exactly where
    debugging happens, and it costs one open file handle. Call it after `start_log()`,
    so the session header is already above any stack trace that follows.
    """
    global _CRASH_FH, _OLD_HOOKS
    if _CRASH_FH is not None:                      # already on; don't do it twice
        return log_path()
    fh = _crash_stream()
    if fh is None:
        return None
    _CRASH_FH = fh
    # Without redirection (loose script) start_log() wrote nothing, and then a stack
    # trace would end up in the file with no date, path or version — useless if a
    # colleague sends it over. So write the session header here too, in that case.
    if _ORIGINAL is None:
        _write_header(fh)
    # Whoever sends this file in must be able to read it without the source code next
    # to it. Needed because Qt reliably raises one handled COM exception on startup
    # (0x8001010d) that faulthandler reports anyway — without this line every log
    # would read like a crash.
    fh.write("    crashlog  : on; every session should end with "
             "'cleanly closed'.\n"
             "                A 'Windows fatal exception' followed by that closing "
             "line is handled\n                and harmless; if the closing line is "
             "missing, the app crashed there.\n")
    _OLD_HOOKS = (sys.excepthook, getattr(threading, "excepthook", None))
    sys.excepthook = _excepthook
    if _OLD_HOOKS[1] is not None:
        threading.excepthook = _thread_excepthook
    atexit.register(_exit_line)
    return log_path()


def _reopen_crashlog():
    """After a rotation our file descriptor still points at the renamed file; reopen
    it. If that fails, keep the old one — a stack trace in .log.1 still beats no stack
    trace at all."""
    global _CRASH_FH
    if _CRASH_FH is None:
        return
    old = _CRASH_FH
    new = _crash_stream()
    if new is None:
        return
    _CRASH_FH = new
    try:
        old.close()
    except (OSError, ValueError):
        pass


def stop_crashlog():
    """Restores the hooks and releases the file. For the self-test — the app itself
    keeps the safety net armed until the end of the session."""
    global _CRASH_FH, _OLD_HOOKS
    import faulthandler
    if _CRASH_FH is None:
        return
    atexit.unregister(_exit_line)
    faulthandler.disable()
    if _OLD_HOOKS is not None:
        sys.excepthook = _OLD_HOOKS[0]
        if _OLD_HOOKS[1] is not None:
            threading.excepthook = _OLD_HOOKS[1]
        _OLD_HOOKS = None
    try:
        _CRASH_FH.close()
    except (OSError, ValueError):
        pass
    _CRASH_FH = None


# ── Self-test ────────────────────────────────────────────────────────────────────

def _self_test():
    import shutil
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="skate_environment_")
    old_local = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        assert data_dir() == os.path.join(tmp, "SkateAnalysis")
        assert os.path.isdir(data_dir()), "data_dir() doesn't create the folder"
        assert app_dir() == os.path.dirname(os.path.abspath(__file__))
        assert not is_frozen()

        # 0. Migration: a folder using the old product name (from before the rename to
        #    English) is renamed in place instead of the app looking empty after an
        #    update.
        mig = os.path.join(tmp, "migration")
        os.environ["LOCALAPPDATA"] = mig
        old_folder = os.path.join(mig, "SchaatsAnalyse")
        os.makedirs(old_folder, exist_ok=True)
        with open(os.path.join(old_folder, "marker.txt"), "w") as f:
            f.write("old data")
        new_folder = data_dir()
        assert new_folder == os.path.join(mig, "SkateAnalysis")
        assert not os.path.isdir(old_folder), "old folder should have been renamed away"
        with open(os.path.join(new_folder, "marker.txt")) as f:
            assert f.read() == "old data", "migration lost the old folder's contents"
        os.environ["LOCALAPPDATA"] = tmp

        # 1. Redirection: everything that would go to a console ends up in the file —
        #    including the direct .write that used to break on sys.stdout=None.
        path = start_log(force=True)
        assert path == log_path() and os.path.isfile(path), path
        print("line via print")
        print("line via stderr", file=sys.stderr)
        sys.stdout.write("line via write\n")
        assert not sys.stdout.isatty() and sys.stdout.encoding == "utf-8"
        stop_log()
        assert sys.stdout is sys.__stdout__, "stdout not restored"

        with open(path, encoding="utf-8") as f:
            text = f.read()
        for expected in ("SkateAnalysis started", "data_dir  :",
                         "line via print", "line via stderr", "line via write"):
            assert expected in text, f"{expected!r} missing from the log"

        # 2. A second session appends instead of wiping the file.
        start_log(force=True)
        print("second session")
        stop_log()
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert text.count("SkateAnalysis started") == 2 and "line via print" in text

        # 3. Rotating by size: the old content moves to .log.1 and the log starts
        #    fresh, so it never fills up unnoticed.
        rot = os.path.join(tmp, "rot.log")
        log = _LogFile(rot, max_bytes=2000)
        log.write("a" * 1500 + "\n")
        assert not os.path.exists(rot + ".1"), "rotated too early"
        log.write("b" * 1000 + "\n")
        log.write("after the rotation\n")
        log.flush()
        assert os.path.exists(rot + ".1"), "didn't rotate"
        with open(rot, encoding="utf-8") as f:
            assert f.read() == "after the rotation\n", "didn't start fresh after rotation"
        log.write("x" * 2500 + "\n")               # once more: .log.1 gets overwritten
        log.flush()
        with open(rot + ".1", encoding="utf-8") as f:
            assert f.read().startswith("after the rotation"), ".log.1 not replaced"

        # 4. Crash log: an unhandled Python error ends up in the log file — including
        #    one from a plain thread, since that's exactly where it used to vanish
        #    without a trace. With the log on, so the default hook doesn't also put it
        #    on the console.
        start_log(force=True)
        assert start_crashlog() == path
        import faulthandler
        assert faulthandler.is_enabled(), "faulthandler is not enabled"
        try:
            raise ValueError("error-in-main-thread")
        except ValueError:
            sys.excepthook(*sys.exc_info())

        def _broken_thread():
            raise KeyError("error-in-thread")

        th = threading.Thread(target=_broken_thread, name="testthread")
        th.start()
        th.join()
        stop_crashlog()
        stop_log()
        assert not faulthandler.is_enabled(), "faulthandler not disabled"
        assert sys.excepthook is sys.__excepthook__, "excepthook not restored"
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for expected in ("error-in-main-thread", "ValueError",
                         "error-in-thread", "testthread"):
            assert expected in text, f"{expected!r} missing from the log"

        # 5. And the case faulthandler really exists for: an access violation, exactly
        #    the crash from 2026-08-25 (0xc0000005 in Qt6Gui.dll). Python never gets to
        #    a traceback then, so this can only be tested in a separate process.
        code = ("import sys; sys.path.insert(0, %r)\n"
                "import skate_environment as e; e.start_crashlog()\n"
                "import faulthandler; faulthandler._read_null()\n"
                % os.path.dirname(os.path.abspath(__file__)))
        # Own folder: Windows Error Reporting can hold on to the crashed process for a
        # bit, and then the log file above can't be removed yet at the next point.
        apart = os.path.join(tmp, "crash")
        sub = subprocess.run([sys.executable, "-c", code],
                             env=dict(os.environ, LOCALAPPDATA=apart),
                             capture_output=True, text=True)
        assert sub.returncode != 0, "the test process didn't even crash"
        with open(os.path.join(apart, "SkateAnalysis", LOG_NAME), encoding="utf-8") as f:
            text = f.read()
        assert ("Windows fatal exception" in text
                or "Fatal Python error" in text), "no crash stack in the log"
        assert 'File "<string>"' in text, "the stack is missing the Python frames"
        assert "access violation" in text, "not recognized as an access violation"
        assert "=== cleanly closed" not in text, "a crash must not count as a clean stop"

        #    And conversely, a normal shutdown should get that line: that's exactly
        #    what makes the log readable afterwards — no closing line = hard stopped.
        clean = os.path.join(tmp, "clean")
        sub = subprocess.run(
            [sys.executable, "-c", code.replace("faulthandler; faulthandler._read_null()",
                                                "sys; sys.exit(0)")],
            env=dict(os.environ, LOCALAPPDATA=clean), capture_output=True, text=True)
        assert sub.returncode == 0, sub.stderr
        with open(os.path.join(clean, "SkateAnalysis", LOG_NAME), encoding="utf-8") as f:
            assert "=== cleanly closed" in f.read(), "closing line missing after a clean stop"

        # 6. No log file possible (a folder with that name is in the way) → no crash,
        #    and print stays safe. That's the requirement this was all built for, in
        #    the exe. In a fresh folder, since the log from point 4 still holds its
        #    file open: `_LogFile.close()` deliberately only flushes, so a library that
        #    accidentally closes sys.stdout doesn't blind the rest of the session.
        os.environ["LOCALAPPDATA"] = os.path.join(tmp, "nolog")
        os.makedirs(os.path.join(data_dir(), LOG_NAME), exist_ok=True)
        assert start_log(force=True) is None
        assert start_crashlog() is None, "crashlog must survive without a file too"
        print("this disappears")
        sys.stdout.write("this too\n")
        stop_log()
        assert sys.stdout is sys.__stdout__

        print("Self-test OK")
    finally:
        stop_log()
        if old_local is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = old_local
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
