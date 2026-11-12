"""Does a camera's image look upside down? Upright people should have their shoulders
above their hips. A camera mounted upside down without its image flipped shows most
upright people the other way round; the service then hints at
``FIM_CAMERA_<ID>_ROTATE_DEG``. Lying people and unsure poses do not vote."""

from collections.abc import Iterable
from dataclasses import dataclass

from fall_impact_monitor.analysis.observation import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from fall_impact_monitor.perception.tracking import Track

_UPRIGHT_RATIO = 2.0  # vertical torso extent at least twice the horizontal one


def upright_vote(track: Track, keypoint_threshold: float = 0.5) -> bool | None:
    """True when an upright torso is upside down, False when it is the right way up,
    None when the person is not clearly upright or the joints are not visible."""
    if track.keypoints is None or track.keypoint_scores is None:
        return None
    joints = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if any(track.keypoint_scores[j] < keypoint_threshold for j in joints):
        return None
    points = track.keypoints
    shoulder_x = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2
    shoulder_y = (points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]) / 2
    hip_x = (points[LEFT_HIP][0] + points[RIGHT_HIP][0]) / 2
    hip_y = (points[LEFT_HIP][1] + points[RIGHT_HIP][1]) / 2
    dx, dy = hip_x - shoulder_x, hip_y - shoulder_y
    if abs(dy) < _UPRIGHT_RATIO * abs(dx) or dy == 0:
        return None
    return bool(dy < 0)


@dataclass
class OrientationCheck:
    """Counts the votes of upright people; flags the camera once enough agree."""

    min_votes: int = 50
    inverted_ratio: float = 0.8
    votes: int = 0
    upside_down: int = 0
    inverted: bool = False

    def update(self, tracks: Iterable[Track]) -> bool:
        """True only on the call that flags the camera."""
        if self.inverted:
            return False
        for track in tracks:
            vote = upright_vote(track)
            if vote is not None:
                self.votes += 1
                self.upside_down += vote
        if self.votes >= self.min_votes and self.upside_down >= self.inverted_ratio * self.votes:
            self.inverted = True
            return True
        return False
