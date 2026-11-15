"""Real falls of older adults: the publicly released videos of "Falls experienced by older
adult residents in long-term care homes" (Robinovitch et al., Simon Fraser University;
Databrary volume 739), used with the lab's permission for private evaluation (#72).

The videos and the annotations stay on the installer's computer. Each video holds one
fall seen from up to three cameras one after another; ``<root>/clips/<clip>.mp4`` is one
camera view, and ``<root>/labels.csv`` (written locally, never published) gives, per
clip: ``clip, participant, fall_start_s, fall_end_s, quality, notes``. ``quality`` is
``clear``, ``uncertain`` (the person is far away, or the clip ends during the fall),
``assumed`` (a view of the same length as an annotated one, taken as synchronised) or
``not_visible`` (left out: no fall can be seen).

Video names are ``<participant>/<clip>.mp4`` so that splits and groups keep a person on
one side. The clips are falls only: they measure recall on real falls, not false alerts.
"""

import csv
from pathlib import Path

from fall_impact_monitor.evaluation.truth import LabelledVideo, TruthEvent

VISIBLE = ("clear", "uncertain", "assumed")


def sfu_public_videos(root: Path, *, clear_only: bool = False) -> list[LabelledVideo]:
    """The annotated clips under ``root`` whose fall can be seen (only ``clear`` ones
    with ``clear_only``), in clip order. Raises ``FileNotFoundError`` for a listed clip
    that is missing."""
    kept = ("clear",) if clear_only else VISIBLE
    videos = []
    with (root / "labels.csv").open(encoding="utf-8", newline="") as labels:
        for row in csv.DictReader(labels):
            if row["quality"] not in kept:
                continue
            path = root / "clips" / f"{row['clip']}.mp4"
            if not path.exists():
                raise FileNotFoundError(f"{path.name} is listed in labels.csv but missing")
            fall = TruthEvent(float(row["fall_start_s"]), float(row["fall_end_s"]))
            videos.append(LabelledVideo(f"{row['participant']}/{row['clip']}.mp4", path, (fall,)))
    return sorted(videos, key=lambda v: v.path.name)
