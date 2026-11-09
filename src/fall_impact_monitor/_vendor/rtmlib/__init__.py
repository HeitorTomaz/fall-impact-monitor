"""RTMPose pre/post-processing from rtmlib (Apache-2.0, Open-MMLab)."""

from fall_impact_monitor._vendor.rtmlib.processing import (
    bbox_xyxy2cs,
    get_simcc_maximum,
    top_down_affine,
)

__all__ = ["bbox_xyxy2cs", "get_simcc_maximum", "top_down_affine"]
