"""
schaats_yolo.py — transitional compatibility shim, not the real implementation.

The real module is now skate_yolo.py (see the translate-to-english plan). This file
exists only because schaats_gui.py isn't translated yet and does a bare
`import schaats_yolo` inside `_laad_backend()`, then reads `schaats_yolo.BACKEND_NAAM`
and calls `schaats_yolo.analyseer(...)`. skate_yolo.py itself already defines a
Dutch-named alias for both of those (see the bottom of that file), so re-exporting
everything here makes them resolve exactly as before, just through this thin redirect.

Delete this file once schaats_gui.py is translated and does `import skate_yolo`
directly.
"""

from skate_yolo import *  # noqa: F401,F403
