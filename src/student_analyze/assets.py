"""Read-only source discovery, hashing, and integrity verification."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Iterable

from student_analyze.errors import SourceIntegrityError
from student_analyze.fingerprint import digest_value
from student_analyze.models import SourceAsset


CHUNK_SIZE = 1024 * 1024


def discover_source_assets(root: Path, extensions: Iterable[str]) -> list[SourceAsset]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise SourceIntegrityError(f"source root is not a directory: {resolved_root}")

    allowed = {item.lower() for item in extensions}
    files: list[Path] = []
    for current, directories, names in os.walk(resolved_root, followlinks=False):
        current_path = Path(current)
        unsafe_directories = [name for name in directories if (current_path / name).is_symlink()]
        if unsafe_directories:
            raise SourceIntegrityError(
                f"symbolic-link directories are not accepted below source root: {unsafe_directories}"
            )
        for name in names:
            path = current_path / name
            if path.suffix.lower() in allowed:
                files.append(path)

    if not files:
        raise SourceIntegrityError(
            f"no supported source files found below {resolved_root}; extensions={sorted(allowed)}"
        )

    assets = [_build_source_asset(resolved_root, path) for path in files]
    return sorted(assets, key=lambda asset: asset.relative_path.casefold())


def verify_source_assets(assets: Iterable[SourceAsset]) -> None:
    for expected in assets:
        path = Path(expected.source_path)
        actual = _build_source_asset_from_relative(path, expected.relative_path)
        differences: list[str] = []
        if actual.sha256 != expected.sha256:
            differences.append("sha256")
        if actual.size_bytes != expected.size_bytes:
            differences.append("size_bytes")
        if actual.modified_time_ns != expected.modified_time_ns:
            differences.append("modified_time_ns")
        if differences:
            raise SourceIntegrityError(
                f"source asset changed ({', '.join(differences)}): {path}"
            )


def sha256_file(path: Path) -> tuple[str, int, int]:
    if path.is_symlink():
        raise SourceIntegrityError(f"symbolic-link source assets are not accepted: {path}")
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise SourceIntegrityError(f"cannot read source asset {path}: {exc}") from exc

    before_identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_identity = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_identity != after_identity:
        raise SourceIntegrityError(f"source asset changed while hashing: {path}")
    return digest.hexdigest(), after.st_size, after.st_mtime_ns


def artifact_digest(path: Path) -> tuple[str, int]:
    digest, size, _ = sha256_file(path)
    return digest, size


def _build_source_asset(root: Path, path: Path) -> SourceAsset:
    relative_path = path.relative_to(root).as_posix()
    return _build_source_asset_from_relative(path, relative_path)


def _build_source_asset_from_relative(path: Path, relative_path: str) -> SourceAsset:
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise SourceIntegrityError(f"linked source assets are not accepted: {path}")
    resolved = path.resolve(strict=True)
    digest, size, modified_time_ns = sha256_file(resolved)
    asset_key = digest_value({"relative_path": relative_path, "sha256": digest})[:20]
    media_type, _ = mimetypes.guess_type(resolved.name)
    return SourceAsset(
        asset_id=f"asset-{asset_key}",
        relative_path=relative_path,
        source_path=str(resolved),
        sha256=digest,
        size_bytes=size,
        modified_time_ns=modified_time_ns,
        media_type=media_type,
    )
