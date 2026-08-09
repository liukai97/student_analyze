"""Integrity and geometry verification for committed phase 2 outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from student_analyze.assets import artifact_digest
from student_analyze.errors import CaseValidationError
from student_analyze.models import CaseManifest
from student_analyze.page_geometry import coordinate_matrices, verify_coordinate_round_trip
from student_analyze.page_models import PageManifest
from student_analyze.validation import validate_json


def read_page_manifest(path: Path) -> PageManifest:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read page manifest {path}: {exc}") from exc
    return validate_json(PageManifest, raw)


def verify_page_outputs(
    case_dir: Path,
    case_manifest: CaseManifest,
    page_manifest: PageManifest,
) -> None:
    resolved_case = case_dir.resolve(strict=True)
    manifest_sha256, _ = artifact_digest(resolved_case / "case_manifest.json")
    if page_manifest.case_id != case_manifest.case_id:
        raise CaseValidationError("page manifest and case manifest use different case_id values")
    if page_manifest.source_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("page manifest references a different case manifest version")

    sources = {asset.asset_id: asset for asset in case_manifest.source_assets}
    page_source_ids = {page.source_asset_id for page in page_manifest.pages}
    if page_source_ids != set(sources):
        missing = sorted(set(sources) - page_source_ids)
        unexpected = sorted(page_source_ids - set(sources))
        raise CaseValidationError(
            f"page manifest source coverage mismatch; missing={missing}, unexpected={unexpected}"
        )

    source_metadata: dict[str, tuple[tuple[int, int], int | None]] = {}
    for source in case_manifest.source_assets:
        try:
            with Image.open(source.source_path) as image:
                image.load()
                source_metadata[source.asset_id] = (
                    image.size,
                    image.getexif().get(274),
                )
        except (OSError, UnidentifiedImageError) as exc:
            raise CaseValidationError(
                f"cannot decode source image {source.source_path}: {exc}"
            ) from exc

    for page in page_manifest.pages:
        source = sources[page.source_asset_id]
        if page.source_sha256 != source.sha256:
            raise CaseValidationError(f"source hash mismatch for logical page {page.page_id}")
        if page.source_relative_path != source.relative_path:
            raise CaseValidationError(f"source path mismatch for logical page {page.page_id}")
        expected_source_size, expected_exif_orientation = source_metadata[source.asset_id]
        if (
            page.source_image_size.width,
            page.source_image_size.height,
        ) != expected_source_size:
            raise CaseValidationError(f"source dimensions mismatch for {page.page_id}")
        if page.source_exif_orientation != expected_exif_orientation:
            raise CaseValidationError(f"source EXIF mismatch for {page.page_id}")

        expected_forward, expected_inverse, expected_size = coordinate_matrices(
            page.crop_box, page.rotation_clockwise
        )
        if page.raw_to_page_matrix != expected_forward:
            raise CaseValidationError(f"invalid raw_to_page_matrix for {page.page_id}")
        if page.page_to_raw_matrix != expected_inverse:
            raise CaseValidationError(f"invalid page_to_raw_matrix for {page.page_id}")
        if page.page_image_size != expected_size:
            raise CaseValidationError(f"invalid logical page size for {page.page_id}")
        verify_coordinate_round_trip(
            page.crop_box,
            page.raw_to_page_matrix,
            page.page_to_raw_matrix,
            page.page_image_size,
        )

        derived_path = (resolved_case / page.derived.relative_path).resolve(strict=False)
        if resolved_case not in derived_path.parents:
            raise CaseValidationError(f"derived page escapes case directory: {derived_path}")
        if not derived_path.is_file():
            raise CaseValidationError(f"derived page is missing: {derived_path}")
        digest, size = artifact_digest(derived_path)
        if digest != page.derived.sha256 or size != page.derived.size_bytes:
            raise CaseValidationError(f"derived page changed: {derived_path}")
        _verify_derived_image(derived_path, page.page_image_size.width, page.page_image_size.height)


def _verify_derived_image(path: Path, width: int, height: int) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise CaseValidationError(f"derived page is not JPEG: {path}")
            if image.size != (width, height):
                raise CaseValidationError(
                    f"derived page dimensions differ from manifest: {path}"
                )
            if image.getexif().get(274) is not None:
                raise CaseValidationError(
                    f"derived page must not carry EXIF orientation: {path}"
                )
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(f"cannot decode derived page {path}: {exc}") from exc
