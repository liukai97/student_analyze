from __future__ import annotations

import pytest

from student_analyze.page_geometry import (
    apply_matrix,
    coordinate_matrices,
    verify_coordinate_round_trip,
)
from student_analyze.page_models import ImageSize, PixelBox


@pytest.mark.parametrize(
    ("rotation", "expected_size"),
    [
        (0, ImageSize(width=6, height=4)),
        (90, ImageSize(width=4, height=6)),
        (180, ImageSize(width=6, height=4)),
        (270, ImageSize(width=4, height=6)),
    ],
)
def test_crop_rotation_matrices_round_trip(rotation: int, expected_size: ImageSize) -> None:
    crop = PixelBox(left=10, top=20, right=16, bottom=24)
    forward, inverse, page_size = coordinate_matrices(crop, rotation)

    assert page_size == expected_size
    verify_coordinate_round_trip(crop, forward, inverse, page_size)


def test_phase0_clockwise_matrix_matches_pixel_coordinate_convention() -> None:
    crop = PixelBox(left=0, top=0, right=5712, bottom=4284)
    forward, inverse, page_size = coordinate_matrices(crop, 90)

    assert forward == (
        (0.0, -1.0, 4283.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert page_size == ImageSize(width=4284, height=5712)
    assert apply_matrix(forward, (0.0, 0.0)) == (4283.0, 0.0)
    assert apply_matrix(inverse, (4283.0, 0.0)) == (0.0, 0.0)
