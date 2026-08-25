# Writing a detector

A detector is one `DetectorKind`, registered. Everything else — the config
format, the GUI table, the pipeline ordering, the overlay — is a consequence of
what that declaration says. Nothing in `mlti_cal` names any particular detector,
so a third-party one is not a second-class citizen: it reaches detection through
exactly the path the built-ins use.

A complete, working example is in
[`examples/detectors/aruco_grid.py`](../examples/detectors/aruco_grid.py).

## Installing one

Drop a `.py` file in either place and restart:

| Where | For |
|---|---|
| `~/.mlti_cal/detectors/` | your own detectors |
| `MLTI_CAL_DETECTOR_PATH` (`os.pathsep`-separated dirs) | running out of a working tree, and tests |
| a package with an `mlti_cal.detectors` entry point | shipping one to other people |

Files beginning with `_` are skipped, so `_helpers.py` next to your detector is
a support module and not a detector.

A plugin that raises while loading is recorded in `registry.PLUGIN_ERRORS` and
reported in the **+ Object** menu, not raised — one broken file must not stop
the application from starting. It is not swallowed either: a plugin that
silently fails to load is indistinguishable from one that was never installed,
which is the worst of both.

`MLTI_CAL_LOAD_PLUGINS=0` skips discovery entirely, for reproducing a result
without whatever happens to be installed.

## The declaration

```python
register_detector(
    DetectorKind(
        id="my_target",  # a valid Python identifier; re-using an id REPLACES that kind
        label="My target",  # shown in the menu
        description="...",  # shown as the menu tooltip; say what it is and when to use it
        coded=False,  # see below
        glyph="square",  # circle | square | diamond | triangle
        spec_cls=MySpec,
        group_cls=MultiMyDetector,
        settings_cls=MySettings,
        catalog=MY_CATALOG,  # one Option per settings field, exactly
        summary="...",
        board_fields=[...],  # one Option per spec field, minus `id`
    )
)
```

### `coded`

`True` when every feature carries its own identity — ArUco markers do, plain
chessboard corners and dots do not.

It is the only thing the pipeline needs in order to order detectors. Coded kinds
run first: clutter cannot fool them, and once located their boards are painted
out of the image. Uncoded kinds then run on the cleaned image, repeatedly, until
a pass finds nothing new. Declaring `coded=True` when your features are not
identifiable will produce confident wrong detections in mixed scenes.

### `board_fields`

One `Option` per field of `spec_cls`, excluding `id`. These are the GUI table
columns **and** the JSON keys of a saved board, so they are user-visible: use
your target's own vocabulary. A dot grid says `circles_x` and `spacing`, never
`squares_x` and `square_length`.

Fields of different kinds that mean the same thing share a table column by
declaring the same `Option.column`:

```python
Option(name="markers_x", kind="int", default=5, column="count_x", when="...")
```

`count_x`, `count_y` and `pitch` are the conventions the built-ins use. A field
with no `column` gets a column of its own, named after it.

A spec field carrying a default may be omitted from a config — it just will not
appear in the GUI unless you also declare it in `board_fields`.

### `catalog`

Every field of `settings_cls` must appear, and nothing else may. Registration is
refused otherwise, with the traceback pointing at your plugin. This is the same
rule the built-ins are held to: a knob the user can see but nothing reads is
worse than no knob at all.

Each `Option.when` is the guidance shown in the settings window. It is the
product — a dropdown of enum names helps nobody; the note saying which to pick
and why is the reason the setting is exposed.

## The two protocols

### `spec_cls` — one board

```python
@dataclass
class MySpec:
    id: str
    # ... exactly your board_fields ...

    @property
    def object_points(self) -> np.ndarray:   # (M,3) in the board frame, indexed by point id
    @property
    def outline_object_points(self) -> np.ndarray:  # (4,2) printed sheet outline
```

`outline_object_points` is what other detectors use to paint your board out of
the image. Get it wrong and you leave a strip of your board visible for a
chessboard detector to lock onto.

Validate in `__post_init__` and raise — a bad board should fail at setup, not
after an hour of detection.

### `group_cls` — every board of this kind

Built as `group_cls(specs, settings=settings)`, and must expose:

| | |
|---|---|
| `specs` | the list it was given |
| `detectors` | `board_id -> your per-board object` |
| `settings` | the settings object |
| `object_points(board_id)` | `(M,3)` |
| `mask_geometry` | `board_id -> MaskGeometry(object_points, outline, pitch)` |
| `detect_all(image)` | `list[Detection]` |

`pitch` is your board's repeat length — a square side, a dot spacing — used as
the unit for mask padding.

**Refuse what you cannot disambiguate.** If two of your boards are configured
identically and nothing distinguishes them in an image, raise at construction.
Assigning a detection to one of them by coin flip is worse than an error,
because a coin flip written into a calibration cannot be detected downstream.

### `Detection`

```python
Detection(board_id=..., point_ids=..., image_points=..., kind=KIND_ID)
```

`point_ids[i]` must index `spec.object_points`, and `image_points[i]` is where
that 3D point was seen. Pairing them wrongly is the one error nothing downstream
can catch: the solve converges, the reprojection error looks plausible, and the
result is wrong.

Return `None` (or omit the board from `detect_all`) when the target is not
found. Do not return a partial detection unless the ids are genuinely correct
for the points you did find — Charuco can, because its markers identify each
corner; a chessboard cannot.

## Taking part in masking

To be paintable out for other detectors, expose `mask_geometry`. To have others
painted out before *you* search, put these four fields in your settings (see
`masking.MaskSettings`):

```
mask_other_boards: bool
mask_padding_squares: float
mask_fill_value: int
mask_min_coverage: float
```

They belong to the detector **about to search**, not to the board being removed:
how much clutter has to be gone before a search is trustworthy is a property of
the search. Set `mask_other_boards=False` if your detector is not fooled by
clutter — marker decoding is not, a chessboard search very much is.
