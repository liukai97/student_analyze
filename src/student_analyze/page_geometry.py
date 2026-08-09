"""Deterministic crop/rotation coordinate transforms for logical pages."""

from __future__ import annotations

from math import isclose

from student_analyze.errors import CaseValidationError
from student_analyze.page_models import ImageSize, Matrix3x3, PixelBox


def coordinate_matrices(
    crop_box: PixelBox, rotation_clockwise: int
) -> tuple[Matrix3x3, Matrix3x3, ImageSize]:
    left, top, right, bottom = (
        crop_box.left,
        crop_box.top,
        crop_box.right,
        crop_box.bottom,
    )
    width, height = crop_box.width, crop_box.height
    if rotation_clockwise == 0:
        forward: Matrix3x3 = (
            (1.0, 0.0, float(-left)),
            (0.0, 1.0, float(-top)),
            (0.0, 0.0, 1.0),
        )
        inverse: Matrix3x3 = (
            (1.0, 0.0, float(left)),
            (0.0, 1.0, float(top)),
            (0.0, 0.0, 1.0),
        )
        size = ImageSize(width=width, height=height)
    elif rotation_clockwise == 90:
        forward = (
            (0.0, -1.0, float(bottom - 1)),
            (1.0, 0.0, float(-left)),
            (0.0, 0.0, 1.0),
        )
        inverse = (
            (0.0, 1.0, float(left)),
            (-1.0, 0.0, float(bottom - 1)),
            (0.0, 0.0, 1.0),
        )
        size = ImageSize(width=height, height=width)
    elif rotation_clockwise == 180:
        forward = (
            (-1.0, 0.0, float(right - 1)),
            (0.0, -1.0, float(bottom - 1)),
            (0.0, 0.0, 1.0),
        )
        inverse = forward
        size = ImageSize(width=width, height=height)
    elif rotation_clockwise == 270:
        forward = (
            (0.0, 1.0, float(-top)),
            (-1.0, 0.0, float(right - 1)),
            (0.0, 0.0, 1.0),
        )
        inverse = (
            (0.0, -1.0, float(right - 1)),
            (1.0, 0.0, float(top)),
            (0.0, 0.0, 1.0),
        )
        size = ImageSize(width=height, height=width)
    else:
        raise CaseValidationError(
            f"rotation must be one of 0, 90, 180, 270; got {rotation_clockwise}"
        )
    return forward, inverse, size


def apply_matrix(matrix: Matrix3x3, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if isclose(denominator, 0.0, abs_tol=1e-12):
        raise CaseValidationError("coordinate matrix maps a point to infinity")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )


def verify_coordinate_round_trip(
    crop_box: PixelBox,
    raw_to_page: Matrix3x3,
    page_to_raw: Matrix3x3,
    page_size: ImageSize,
    *,
    tolerance: float = 1e-8,
) -> None:
    center_x = (crop_box.left + crop_box.right - 1) / 2
    center_y = (crop_box.top + crop_box.bottom - 1) / 2
    points = (
        (float(crop_box.left), float(crop_box.top)),
        (float(crop_box.right - 1), float(crop_box.top)),
        (float(crop_box.left), float(crop_box.bottom - 1)),
        (float(crop_box.right - 1), float(crop_box.bottom - 1)),
        (center_x, center_y),
    )
    for raw_point in points:
        page_point = apply_matrix(raw_to_page, raw_point)
        if not (
            -tolerance <= page_point[0] <= page_size.width - 1 + tolerance
            and -tolerance <= page_point[1] <= page_size.height - 1 + tolerance
        ):
            raise CaseValidationError(
                f"raw point {raw_point} maps outside logical page: {page_point}"
            )
        recovered = apply_matrix(page_to_raw, page_point)
        if not (
            isclose(raw_point[0], recovered[0], abs_tol=tolerance)
            and isclose(raw_point[1], recovered[1], abs_tol=tolerance)
        ):
            raise CaseValidationError(
                f"coordinate round trip failed: {raw_point} -> {page_point} -> {recovered}"
            )
