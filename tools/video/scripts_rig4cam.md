# rig_4cam demo -- script variations

Every number in a caption is measured on `examples/rig_4cam` (seeds fixed), not
written from memory. Each variation uses the same recorded take; only the cut
and the captions differ, in `cut_rig4cam.py`.

## A. "Two cameras that never meet" -- built (`cut_rig4cam.py`)

Hook on scale and on a problem people recognise: cameras with no overlap.

| # | Picture | Caption | Sub |
|---|---|---|---|
| 1 | four camera views flicking | 4 cameras. 3 boards. One solve. | |
| 2 | outer pair outlined, middle dimmed | The outer two never see the same board. | |
| 3 | setup tables | Two Charuco boards + a dot grid | one wide-angle lens in the mix |
| 4 | thumbnails lighting up, fast-forwarded | Detect all 120 images | |
| 5 | wide-lens frame, corners per board | Every board, its own colour | cam2: Double Sphere wide-angle lens |
| 6 | starting-point summary ("cam3: placed through cam2") | Every camera gets a starting point | cam3 via cam2 -- it shares no view with cam0 |
| 7 | Solve button, result cards, live convergence chart | 0.64 -> 0.13 px | Ceres, under 3 seconds |
| 8 | same panel, held-out card fills in | Unseen frames fit just as well | held-out error 0.99x training |
| 8b | Build report, then the verdict banner and top findings | Then it tells you what to check | |
| 9 | coverage card and per-camera coverage bars | It shows where your data is thin | cam2: data in 75% of the image |
| 10 | the "cannot be told apart" finding | ...and what it can't tell apart | cam2: focal length vs xi, correlation 0.999 |
| 11 | end card | Checked against ground truth | every camera within 0.5 mm |

Second version of the cut (after the GUI redesign):
- Every shot is staged: only its subject is shown, in a window shaped for the
  picture.
- Shot 8b is new.
- Shot 9 uses the coverage numbers instead of the coverage figure, whose two
  panels collapse in a phone-shaped picture.
- Shot 8 shows the held-out card instead of a log line, which was cut in half
  in the first version.

## B. "Your RMS is not the answer"

Hook on the misconception. Open on the solve curve ("0.13 px RMS. Done?"), cut
to "No." over the warnings, then rewind through setup -> detection -> solve at
speed, and land on the correlation finding and the ground-truth card. Riskier:
the opening shot is a plot, which reads poorly on a phone in the first second.

## C. "120 images to sub-millimetre"

Pure workflow speed: one continuous run, a running clock in the corner, captions
only at the stage changes, end card with the ground-truth numbers. Best for the
README GIF and the landscape cut; weakest hook for short-form feeds.

Chosen: **A** for both cuts, because its first two seconds are moving pictures
of the actual capture and its premise needs no calibration background to get.
