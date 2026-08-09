"""Phase 2 logical-page generation driven by reviewed visual decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

from student_analyze import __version__
from student_analyze.assets import artifact_digest, verify_source_assets
from student_analyze.atomic import atomic_write_bytes
from student_analyze.config import AppConfig
from student_analyze.errors import (
    CaseValidationError,
    InvalidTransitionError,
    ReviewRequiredError,
)
from student_analyze.fingerprint import digest_value
from student_analyze.models import (
    SCHEMA_VERSION,
    CaseManifest,
    ImplementationVersions,
    PipelineStage,
    PipelineState,
)
from student_analyze.page_geometry import coordinate_matrices, verify_coordinate_round_trip
from student_analyze.page_models import (
    CropOperation,
    DerivedPageAsset,
    ImageSize,
    LogicalPage,
    PageDecisionSet,
    PageManifest,
    RotateOperation,
)
from student_analyze.page_verification import read_page_manifest, verify_page_outputs
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.validation import validate_json


InterruptHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PagePreparationResult:
    case_dir: Path
    manifest: PageManifest
    state: PipelineState
    reused: bool


def load_page_decisions(path: Path) -> PageDecisionSet:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read page decision file {path}: {exc}") from exc
    return validate_json(PageDecisionSet, raw)


def prepare_logical_pages(
    case_dir: Path,
    decisions: PageDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> PagePreparationResult:
    case_dir = case_dir.resolve(strict=True)
    case_manifest, state = verify_case(case_dir)
    _validate_decisions(case_dir, case_manifest, decisions)
    _reject_unresolved_decisions(decisions)

    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=decisions.provenance.model_identifier,
        prompt=decisions.provenance.prompt_version,
        skill=None,
        tools={"pillow": version("Pillow")},
    )
    decision_sha256 = digest_value(decisions.model_dump(mode="json"))
    stage_inputs = [
        {
            "source_manifest_sha256": decisions.source_manifest_sha256,
            "decision_sha256": decision_sha256,
        }
    ]
    stage_config = config.page_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.PAGES_READY,
        model_type=PageManifest,
        schema_id="page_manifest.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )

    existing_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.PAGES_READY
        ),
        None,
    )
    if existing_completion is not None:
        if existing_completion.stage_fingerprint == stage_fingerprint and not force:
            reference = next(
                reference
                for reference in existing_completion.artifacts
                if reference.schema_id == "page_manifest.schema.json"
            )
            manifest = read_page_manifest(case_dir / reference.relative_path)
            verify_page_outputs(case_dir, case_manifest, manifest)
            return PagePreparationResult(case_dir, manifest, state, reused=True)
        if not force:
            raise InvalidTransitionError(
                "pages_ready is already complete with a different fingerprint; "
                "use force to create a preserved version"
            )
    elif state.current_stage != PipelineStage.INGESTED:
        raise InvalidTransitionError(
            f"cannot generate pages from current stage {state.current_stage}"
        )

    verify_source_assets(case_manifest.source_assets)
    pages = _generate_pages(
        case_dir,
        case_manifest,
        decisions,
        stage_fingerprint,
        config,
        interrupt_hook=interrupt_hook,
    )
    manifest = PageManifest(
        case_id=case_manifest.case_id,
        source_manifest_sha256=decisions.source_manifest_sha256,
        stage_fingerprint=stage_fingerprint,
        decision_sha256=decision_sha256,
        provenance=decisions.provenance,
        created_at=datetime.now(UTC),
        requires_review=any(page.requires_review for page in pages),
        warnings=_unique_warnings(
            warning
            for asset in decisions.assets
            for warning in asset.warnings
        ),
        pages=pages,
    )

    orphan_path = (
        case_dir
        / "artifacts"
        / PipelineStage.PAGES_READY.value
        / stage_fingerprint
        / "page_manifest.json"
    )
    if orphan_path.exists() and not force:
        orphan = read_page_manifest(orphan_path)
        if (
            orphan.case_id != manifest.case_id
            or orphan.stage_fingerprint != manifest.stage_fingerprint
            or orphan.decision_sha256 != manifest.decision_sha256
        ):
            raise CaseValidationError(
                f"conflicting uncommitted page manifest at {orphan_path}"
            )
        manifest = orphan

    verify_page_outputs(case_dir, case_manifest, manifest)
    verify_source_assets(case_manifest.source_assets)
    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.PAGES_READY,
        artifact_name="page_manifest.json",
        payload=manifest,
        model_type=PageManifest,
        schema_id="page_manifest.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        interrupt_hook=interrupt_hook,
    )
    active_reference = committed.artifacts[0]
    active_manifest = read_page_manifest(case_dir / active_reference.relative_path)
    return PagePreparationResult(
        case_dir,
        active_manifest,
        committed.state,
        reused=committed.reused,
    )


def _validate_decisions(
    case_dir: Path,
    case_manifest: CaseManifest,
    decisions: PageDecisionSet,
) -> None:
    if decisions.case_id != case_manifest.case_id:
        raise CaseValidationError("page decisions belong to a different case")
    manifest_sha256, _ = artifact_digest(case_dir / "case_manifest.json")
    if decisions.source_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("page decisions reference a different case manifest")

    sources = {asset.asset_id: asset for asset in case_manifest.source_assets}
    decision_ids = {decision.source_asset_id for decision in decisions.assets}
    if decision_ids != set(sources):
        missing = sorted(set(sources) - decision_ids)
        unexpected = sorted(decision_ids - set(sources))
        raise CaseValidationError(
            f"page decisions must cover every source exactly once; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for decision in decisions.assets:
        if decision.source_sha256 != sources[decision.source_asset_id].sha256:
            raise CaseValidationError(
                f"page decision source hash mismatch: {decision.source_asset_id}"
            )


def _reject_unresolved_decisions(decisions: PageDecisionSet) -> None:
    unresolved: list[str] = []
    for asset in decisions.assets:
        if asset.requires_review:
            unresolved.append(asset.source_asset_id)
        unresolved.extend(
            f"{asset.source_asset_id}:{page.position.value}"
            for page in asset.pages
            if page.requires_review
        )
    if unresolved:
        raise ReviewRequiredError(
            "page decisions require review and cannot advance pages_ready: "
            + ", ".join(unresolved)
        )


def _generate_pages(
    case_dir: Path,
    case_manifest: CaseManifest,
    decisions: PageDecisionSet,
    stage_fingerprint: str,
    config: AppConfig,
    *,
    interrupt_hook: InterruptHook | None,
) -> list[LogicalPage]:
    decisions_by_asset = {
        decision.source_asset_id: decision for decision in decisions.assets
    }
    logical_pages: list[LogicalPage] = []
    for source in case_manifest.source_assets:
        source_decision = decisions_by_asset[source.asset_id]
        try:
            with Image.open(source.source_path) as image:
                image.load()
                source_size = ImageSize(width=image.width, height=image.height)
                source_exif_orientation = image.getexif().get(274)
                for page_decision in source_decision.pages:
                    box = page_decision.crop_box
                    if box.right > image.width or box.bottom > image.height:
                        raise CaseValidationError(
                            f"crop box exceeds source image {source.relative_path}: {box}"
                        )
                    page_id = _page_id(source.asset_id, page_decision.position.value)
                    rendered = image.crop((box.left, box.top, box.right, box.bottom))
                    rendered = _rotate_clockwise(
                        rendered, page_decision.rotation_clockwise
                    )
                    rendered = _as_rgb(rendered)
                    content = _encode_jpeg(rendered, config.jpeg_quality)
                    relative_path = Path(
                        "artifacts",
                        PipelineStage.PAGES_READY.value,
                        stage_fingerprint,
                        "pages",
                        f"{page_id}.jpg",
                    )
                    output_path = case_dir / relative_path
                    _commit_page_image(output_path, content, rendered.size)
                    _interrupt(interrupt_hook, "after_page_commit")

                    raw_to_page, page_to_raw, page_size = coordinate_matrices(
                        box, page_decision.rotation_clockwise
                    )
                    verify_coordinate_round_trip(
                        box, raw_to_page, page_to_raw, page_size
                    )
                    digest, size = artifact_digest(output_path)
                    transforms = [CropOperation(crop_box=box)]
                    if page_decision.rotation_clockwise:
                        transforms.append(
                            RotateOperation(
                                degrees=page_decision.rotation_clockwise
                            )
                        )
                    logical_pages.append(
                        LogicalPage(
                            page_id=page_id,
                            source_asset_id=source.asset_id,
                            source_sha256=source.sha256,
                            source_relative_path=source.relative_path,
                            source_position=page_decision.position,
                            source_image_size=source_size,
                            source_exif_orientation=source_exif_orientation,
                            crop_box=box,
                            rotation_clockwise=page_decision.rotation_clockwise,
                            transforms=transforms,
                            page_image_size=page_size,
                            raw_to_page_matrix=raw_to_page,
                            page_to_raw_matrix=page_to_raw,
                            derived=DerivedPageAsset(
                                relative_path=relative_path.as_posix(),
                                sha256=digest,
                                size_bytes=size,
                                image_size=page_size,
                                exif_orientation=None,
                            ),
                            orientation_confidence=page_decision.orientation_confidence,
                            boundary_confidence=page_decision.boundary_confidence,
                            requires_review=page_decision.requires_review,
                            decision_evidence=page_decision.evidence,
                            warnings=_unique_warnings(
                                [*source_decision.warnings, *page_decision.warnings]
                            ),
                        )
                    )
        except (OSError, UnidentifiedImageError) as exc:
            raise CaseValidationError(
                f"cannot decode source image {source.source_path}: {exc}"
            ) from exc
    return logical_pages


def _rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    operations = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    operation = operations[degrees]
    return image.copy() if operation is None else image.transpose(operation)


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if "A" in image.getbands():
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        exif=b"",
    )
    return buffer.getvalue()


def _commit_page_image(path: Path, content: bytes, expected_size: tuple[int, int]) -> None:
    expected_digest = sha256(content).hexdigest()
    if path.exists():
        digest, size = artifact_digest(path)
        if digest != expected_digest or size != len(content):
            raise CaseValidationError(
                f"existing page artifact differs for the same fingerprint: {path}"
            )
        _validate_jpeg(path, expected_size)
        return
    atomic_write_bytes(
        path,
        content,
        validator=lambda temporary: _validate_jpeg(temporary, expected_size),
    )


def _validate_jpeg(path: Path, expected_size: tuple[int, int]) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != expected_size:
                raise CaseValidationError(
                    f"encoded page does not match JPEG contract: {path}"
                )
            if image.getexif().get(274) is not None:
                raise CaseValidationError(
                    f"encoded page unexpectedly carries EXIF orientation: {path}"
                )
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(f"cannot verify encoded page {path}: {exc}") from exc


def _page_id(asset_id: str, position: str) -> str:
    suffix = asset_id.removeprefix("asset-")
    return f"page-{suffix}-{position}"


def _unique_warnings(warnings) -> list[str]:
    return list(dict.fromkeys(warnings))


def _interrupt(hook: InterruptHook | None, event: str) -> None:
    if hook is not None:
        hook(event)
