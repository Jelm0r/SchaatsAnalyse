# Recording advice: free accuracy

The angle measurement is directly bounded by what's on the pixels. At a distance, a
lower leg is only ~30–60 px long; **2 px of keypoint error = 2–4° of angle error**.
Better recordings therefore pay off more than any algorithm improvement. Guidelines,
in order of effect:

1. **Film in 4K** (3840×2160) instead of 1080p. Doubles the linear pixel resolution:
   a distant skater's lower leg becomes 60–120 px instead of 30–60 px, and every
   pixel error counts half as much toward the angle. The analysis gets slower
   (bigger frames to decode), but the refinement pass only looks at the crop around
   the skater anyway.
2. **Short shutter speed** (sport/action mode, or manually ≥ 1/500 s). Motion blur
   smears the legs into vague streaks that trips up every pose model — this is the
   biggest source of "skeleton next to the leg" frames. There's usually enough light
   on an ice rink; a higher ISO (some noise) is far less harmful than blur.
3. **Record progressive, not interlaced** — only applies to camcorders; phones and
   action cams already do this correctly. Look in the menu for `PS`, `1080/50p`, or
   "recording mode" and avoid anything with an `i` in it (`1080/50i`). An interlaced
   camera shoots a *half* frame 50 times a second (first the even rows, then the odd
   ones) and weaves those two moments — 1/50 s apart — into one frame. On a moving
   leg the two halves therefore land in different places: the comb artifacts you see
   in the picture.
   **Measured** on `00005.MTS` (AVCHD 1080i50) against progressive clips: the two
   halves sit **6.1 px** apart on knees and ankles (p90 13.4; max 44), against 0.06
   px on progressive material — with "2 px = 2–4° angle error" that's the largest
   noise source in such recordings. The app filters it out (`deinterlace` in
   `skate_analysis.py`, auto-detected per video), but that's a repair: half the image
   rows on the moving part get interpolated instead of actually recorded. Recording
   it right the first time gives you real pixels *and* saves the half temporal
   resolution the filter throws away.

4. **50 or 60 fps** instead of 24/30. More frames per stride = better smoothing,
   tighter push detection (the weight transfer only takes a few frames), and smaller
   gaps when a detection is missed. Bonus: 60 fps already forces a shorter shutter
   speed on most phones.
5. **Fixed, level camera, straight-on from the front** (tripod or support). This is
   already the pipeline's assumption (no horizon correction needed); a wobbling
   camera adds an error source that can only be half-repaired afterward.
6. **Contrast helps tracking**: a suit that stands out against the ice and against
   the other riders makes the suit-color gatekeeper (target selection) more
   reliable. Two riders in identical suits crossing each other remains the hardest
   case.
7. **Sun/lights behind the camera**, not behind the skater: backlighting turns the
   skater into a silhouette and depresses the keypoint scores.

Validation: record one session with both the old and new settings and compare
`python skate_eval.py compare old.npz new.npz` (bone-length stability, jitter) —
see `skate_eval.py` for the measurement protocol.
