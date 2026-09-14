"""
schaats_db.py — transitional compatibility shim, not the real implementation.

The real module is now skate_db.py (see the translate-to-english plan). This file
exists only because schaats_schermtest.py isn't translated yet and does a bare
`import schaats_db`, then calls `schaats_db.maak_schaatser(...)` etc. throughout —
skate_gui.py itself was translated in an earlier phase and now does `import skate_db`
directly. skate_db.py itself already defines a Dutch-named alias for every one of
those calls (see the bottom of that file), so re-exporting everything here makes them
resolve exactly as before, just through this thin redirect.

Delete this file once schaats_schermtest.py is translated and does `import skate_db`
directly.
"""

from skate_db import *  # noqa: F401,F403
