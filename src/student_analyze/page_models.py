"""Phase 2 contracts for visual page decisions and generated logical pages."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Annotated, Literal

from pydantic import Field, model_validator

from student_analyze.models import Sha256, StableId, StrictModel


PAGE_SCHEMA_VERSION = "1.0.0"
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class PagePosition(str, Enum):
    SINGLE = "single"
    LEFT = "left"
    RIGHT = "right"


class PageLayout(str, Enum):
    SINGLE_PAGE = "single_page"
    DOUBLE_PAGE = "double_page"


class PixelBox(StrictModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> PixelBox:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("pixel box must have positive width and height")
        return self

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class ImageSize(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LogicalPageDecision(StrictModel):
    position: PagePosition
    crop_box: PixelBox
    rotation_clockwise: Literal[0, 90, 180, 270]
    orientation_confidence: Confidence
    boundary_confidence: Confidence
    requires_review: bool = False
    evidence: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class SourcePageDecision(StrictModel):
    source_asset_id: StableId
    source_sha256: Sha256
    layout: PageLayout
    layout_confidence: Confidence
    requires_review: bool = False
    pages: list[LogicalPageDecision] = Field(min_length=1, max_length=2)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layout(self) -> SourcePageDecision:
        positions = [page.position for page in self.pages]
        if len(positions) != len(set(positions)):
            raise ValueError("page positions must be unique within one source asset")
        expected = (
            [PagePosition.SINGLE]
            if self.layout == PageLayout.SINGLE_PAGE
            else [PagePosition.LEFT, PagePosition.RIGHT]
        )
        if positions != expected:
            raise ValueError(
                f"{self.layout.value} requires positions {[item.value for item in expected]}"
            )
        return self


class PageDecisionProvenance(StrictModel):
    method: Literal[
        "codex_standard_image_input",
        "human_review",
        "deterministic_test_fixture",
    ]
    model_identifier: str | None = None
    model_identifier_unavailable_reason: str | None = None
    prompt_version: str = Field(min_length=1)
    decided_at: datetime

    @model_validator(mode="after")
    def validate_model_identifier(self) -> PageDecisionProvenance:
        if self.model_identifier is None and not self.model_identifier_unavailable_reason:
            raise ValueError(
                "an unavailable reason is required when model_identifier is absent"
            )
        if self.model_identifier is not None and self.model_identifier_unavailable_reason:
            raise ValueError(
                "model_identifier_unavailable_reason must be absent when an identifier exists"
            )
        return self


class PageDecisionSet(StrictModel):
    schema_version: Literal[PAGE_SCHEMA_VERSION] = PAGE_SCHEMA_VERSION
    case_id: StableId
    source_manifest_sha256: Sha256
    provenance: PageDecisionProvenance
    assets: list[SourcePageDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_assets(self) -> PageDecisionSet:
        asset_ids = [asset.source_asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("decision set contains duplicate source_asset_id values")
        return self


class CropOperation(StrictModel):
    kind: Literal["crop"] = "crop"
    crop_box: PixelBox


class RotateOperation(StrictModel):
    kind: Literal["rotate_clockwise"] = "rotate_clockwise"
    degrees: Literal[90, 180, 270]


ExecutedTransform = Annotated[
    CropOperation | RotateOperation,
    Field(discriminator="kind"),
]


class DerivedPageAsset(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    media_type: Literal["image/jpeg"] = "image/jpeg"
    image_size: ImageSize
    exif_orientation: int | None = Field(default=None, ge=1, le=8)

    @model_validator(mode="after")
    def validate_relative_path(self) -> DerivedPageAsset:
        normalized = self.relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("derived page path must stay below the case directory")
        if normalized != self.relative_path:
            raise ValueError("derived page path must use forward slashes")
        return self


class LogicalPage(StrictModel):
    page_id: StableId
    source_asset_id: StableId
    source_sha256: Sha256
    source_relative_path: str = Field(min_length=1)
    source_position: PagePosition
    source_image_size: ImageSize
    source_exif_orientation: int | None = Field(default=None, ge=1, le=8)
    crop_box: PixelBox
    rotation_clockwise: Literal[0, 90, 180, 270]
    transforms: list[ExecutedTransform] = Field(min_length=1, max_length=2)
    page_image_size: ImageSize
    raw_to_page_matrix: Matrix3x3
    page_to_raw_matrix: Matrix3x3
    derived: DerivedPageAsset
    orientation_confidence: Confidence
    boundary_confidence: Confidence
    requires_review: bool = False
    decision_evidence: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_geometry(self) -> LogicalPage:
        if self.crop_box.right > self.source_image_size.width:
            raise ValueError("crop_box exceeds source image width")
        if self.crop_box.bottom > self.source_image_size.height:
            raise ValueError("crop_box exceeds source image height")
        expected_size = (
            ImageSize(width=self.crop_box.width, height=self.crop_box.height)
            if self.rotation_clockwise in {0, 180}
            else ImageSize(width=self.crop_box.height, height=self.crop_box.width)
        )
        if self.page_image_size != expected_size or self.derived.image_size != expected_size:
            raise ValueError("page and derived dimensions do not match crop plus rotation")

        expected_transforms: list[ExecutedTransform] = [
            CropOperation(crop_box=self.crop_box)
        ]
        if self.rotation_clockwise:
            expected_transforms.append(
                RotateOperation(degrees=self.rotation_clockwise)
            )
        if self.transforms != expected_transforms:
            raise ValueError("transforms must describe the crop and actual non-zero rotation")

        for matrix in (self.raw_to_page_matrix, self.page_to_raw_matrix):
            if not all(isfinite(value) for row in matrix for value in row):
                raise ValueError("coordinate matrices must contain finite values")
        return self


class PageManifest(StrictModel):
    schema_version: Literal[PAGE_SCHEMA_VERSION] = PAGE_SCHEMA_VERSION
    case_id: StableId
    source_manifest_sha256: Sha256
    stage_fingerprint: Sha256
    decision_sha256: Sha256
    provenance: PageDecisionProvenance
    created_at: datetime
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)
    pages: list[LogicalPage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pages(self) -> PageManifest:
        page_ids = [page.page_id for page in self.pages]
        derived_paths = [page.derived.relative_path for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("page manifest contains duplicate page_id values")
        if len(derived_paths) != len(set(derived_paths)):
            raise ValueError("page manifest contains duplicate derived paths")
        if self.requires_review != any(page.requires_review for page in self.pages):
            raise ValueError("manifest requires_review must reflect its logical pages")

        positions_by_asset: dict[str, list[PagePosition]] = {}
        for page in self.pages:
            positions_by_asset.setdefault(page.source_asset_id, []).append(
                page.source_position
            )
        for positions in positions_by_asset.values():
            if positions not in (
                [PagePosition.SINGLE],
                [PagePosition.LEFT, PagePosition.RIGHT],
            ):
                raise ValueError("each source must yield one single page or ordered left/right pages")
        return self
