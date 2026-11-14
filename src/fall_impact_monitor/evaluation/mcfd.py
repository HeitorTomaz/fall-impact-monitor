"""Ground truth of the Multiple Cameras Fall Dataset (#69).

Eight cameras mounted high in the corners of one room film 24 scenarios
(``dataset/chuteNN/camK.avi``). Source: E. Auvinet, C. Rougier, J. Meunier, A. St-Arnaud,
J. Rousseau, "Multiple cameras fall dataset", Technical Report 1350, DIRO, Université de
Montréal, 2010, with the erratum by M. Feng and J. Meunier (2025-11-28) applied.

The report tags every frame of one reference camera per scenario; code 2 is "falling"
and code 3 "lying on the ground". A fall here is a falling period plus the lying period
right after it. The cameras were stopped together, so camera ``c`` is aligned with the
reference camera by the difference of their delays (frames each camera has beyond the
shortest one). Frames are numbered from 1 (assumed; an off-by-one is one frame).

The files declare 120 frames per second, so a player shows them four times too fast.
The streams were captured at 30: the authors' paper (Auvinet et al., IEEE Trans. Inf.
Technol. Biomed. 15(2), 2011) states "Video streams (720 x 480 at 30 fps)", and the
content agrees (scenarios last 25 s to 3 min at 30 fps, the labelled falling phases
about one second). Labels and media time use ``MCFD_CAPTURE_FPS``.
"""

import re
from collections.abc import Callable
from pathlib import Path

from fall_impact_monitor.evaluation.truth import LabelledVideo, TruthEvent

# Report p. 16, Fig. 3: delay in frames of cameras 1-8 per scenario (erratum: scenario 5,
# camera 5 is 18).
DELAYS: dict[int, tuple[int, ...]] = {
    1: (3, 3, 8, 4, 23, 6, 6, 0),
    2: (25, 40, 0, 16, 18, 33, 33, 6),
    3: (12, 16, 8, 16, 35, 20, 20, 0),
    4: (72, 79, 78, 0, 68, 82, 83, 56),
    5: (17, 24, 5, 11, 18, 26, 28, 0),
    6: (0, 100, 106, 90, 89, 103, 104, 89),
    7: (28, 14, 16, 0, 1, 17, 18, 20),
    8: (92, 79, 0, 81, 64, 81, 82, 56),
    9: (18, 9, 1, 19, 13, 11, 12, 0),
    10: (14, 15, 19, 33, 12, 17, 19, 0),
    11: (23, 4, 20, 14, 0, 6, 7, 12),
    12: (21, 6, 13, 8, 0, 3, 7, 0),
    13: (16, 33, 0, 7, 27, 27, 36, 13),
    14: (49, 36, 38, 0, 29, 29, 7, 14),
    15: (15, 19, 19, 15, 34, 40, 23, 0),
    16: (23, 29, 0, 2, 12, 9, 3, 3),
    17: (21, 26, 15, 0, 10, 0, 29, 18),
    18: (99, 105, 86, 0, 84, 108, 109, 77),
    19: (19, 27, 16, 19, 5, 29, 0, 20),
    20: (25, 9, 3, 10, 10, 4, 5, 0),
    21: (20, 30, 22, 3, 8, 33, 32, 0),
    22: (0, 46, 51, 41, 53, 46, 47, 34),
    23: (31, 52, 52, 45, 54, 60, 50, 0),
    24: (3, 36, 7, 0, 37, 10, 33, 1),
}

# Report pp. 17-22: reference camera of each scenario (erratum: "11" is camera 8 and
# "10" is camera 7) and its falls as (first falling frame, last frame of the fall or of
# the lying period that follows it). Erratum periods applied to scenarios 2, 7, 8, 14, 18.
REFERENCE_CAMERA: dict[int, int] = {
    1: 8, 2: 4, 3: 8, 4: 6, 5: 8, 6: 1, 7: 6, 8: 4, 9: 8, 10: 8, 11: 7, 12: 8,
    13: 4, 14: 6, 15: 8, 16: 4, 17: 6, 18: 6, 19: 7, 20: 8, 21: 8, 22: 1, 23: 8, 24: 6,
}  # fmt: skip
FALLS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((1080, 1285),),
    2: ((395, 600),),
    3: ((591, 784),),
    4: ((288, 380), (601, 780)),
    5: ((311, 450),),
    6: ((583, 750),),
    7: ((491, 680),),
    8: ((351, 520),),
    9: ((628, 760),),
    10: ((512, 680),),
    11: ((464, 600),),
    12: ((605, 770),),
    13: ((823, 960),),
    14: ((1020, 1115),),
    15: ((755, 870),),
    16: ((891, 1000),),
    17: ((730, 860),),
    18: ((685, 740),),
    19: ((499, 770),),
    20: ((545, 800),),
    21: ((864, 1040),),
    22: ((767, 930),),
    23: ((3574, 3614),),  # got up right away
    24: (),
}

MCFD_CAPTURE_FPS = 30.0

_VIDEO = re.compile(r"chute(\d+)/cam(\d)\.avi$")


def _falls(scenario: int, camera: int, fps: float) -> tuple[TruthEvent, ...]:
    delays = DELAYS[scenario]
    shift = delays[camera - 1] - delays[REFERENCE_CAMERA[scenario] - 1]
    return tuple(
        TruthEvent((first + shift - 1) / fps, (last + shift) / fps)
        for first, last in FALLS[scenario]
    )


def mcfd_videos(root: Path, *, fps_of: Callable[[Path], float]) -> list[LabelledVideo]:
    """Every ``chuteNN/camK.avi`` under ``root`` with its falls, in scenario and camera
    order. Raises ``ValueError`` for a scenario the report does not describe."""
    found = []
    for path in root.rglob("*.avi"):
        name = path.relative_to(root).as_posix()
        match = _VIDEO.search(name)
        if match is None:
            continue
        scenario, camera = int(match[1]), int(match[2])
        if scenario not in FALLS or not 1 <= camera <= len(DELAYS[scenario]):
            raise ValueError(f"{name}: chute{scenario:02d} camera {camera} is not in the report")
        found.append((scenario, camera, name, path))
    return [
        LabelledVideo(name, path, _falls(scenario, camera, fps_of(path)))
        for scenario, camera, name, path in sorted(found)
    ]
