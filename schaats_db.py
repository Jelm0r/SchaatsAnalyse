"""
schaats_db.py — transitional compatibility shim, not the real implementation.

The real module is now skate_db.py (see the translate-to-english plan). This file
exists only because schaats_gui.py and schaats_schermtest.py aren't translated yet and
each do a bare `import schaats_db`, then call `schaats_db.maak_schaatser(...)` etc.
throughout (75+ call sites in schaats_gui.py alone) — far too many to update from
skate_db.py's own phase. skate_db.py itself already defines a Dutch-named alias for
every one of those calls (see the bottom of that file), so re-exporting everything
here makes them resolve exactly as before, just through this thin redirect.

Delete this file once schaats_gui.py and schaats_schermtest.py are translated and do
`import skate_db` directly.
"""

from skate_db import *  # noqa: F401,F403
