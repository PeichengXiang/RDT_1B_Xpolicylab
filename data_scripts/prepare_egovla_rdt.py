#!/usr/bin/env python3
"""Stage the validated EgoVLA conversion for RDT-1B without copying data.

The EgoVLA repository already contains the official XPolicyLab canonical
conversion. This command verifies the raw inventory and canonical schema,
then publishes a task/env/data tree below the RDT model root using hard links
(or relative symlinks when the filesystems differ). No Deprecated episode,
and not the extra close_drawer_smoke artifact, is staged.

The resulting layout is:
  <output>/Close-Drawer/ego_h1_inspire/data/episode_0000000.hdf5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


TASKS = OrderedDict(
    [
        ("close_drawer", ("Close-Drawer", 50)),
        ("flip_mug", ("Flip-Mug", 100)),
        ("insert_and_unload_cans", ("Insert-And-Unload-Cans", 900)),
        ("insert_cans", ("Insert-Cans", 100)),
        ("open_drawer", ("Open-Drawer", 100)),
        ("open_laptop", ("Open-Laptop", 100)),
        ("pour_balls", ("Pour-Balls", 102)),
        ("push_box", ("Push-Box", 100)),
        ("sort_cans", ("Sort-Cans", 101)),
        ("stack_can", ("Stack-Can", 100)),
        ("stack_can_into_drawer", ("Stack-Can-Into-Drawer", 50)),
        ("unload_cans", ("Unload-Cans", 100)),
    ]
)
JOINT_FIELDS = (
    ("left_arm_joint_states", 7),
    ("left_ee_joint_states", 12),
    ("right_arm_joint_states", 7),
    ("right_ee_joint_states", 12),
)
CAMERA_PATHS = (
    "vision/cam_head/colors",
    "vision/cam_left_wrist/colors",
    "vision/cam_right_wrist/colors",
)


def contains_deprecated(path: Path) -> bool:
    return any("deprecated" in part.lower() for part in path.parts)


def raw_inventory(raw_root: Path) -> tuple[int, int, list[str]]:
    active = 0
    deprecated = 0
    deprecated_paths: list[str] = []
    for path in sorted(raw_root.rglob("*.hdf5")):
        relative = path.relative_to(raw_root)
        if contains_deprecated(relative):
            deprecated += 1
            deprecated_paths.append(relative.as_posix())
        else:
            active += 1
    return active, deprecated, deprecated_paths


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def canonical_dir(canonical_root: Path, slug: str) -> Path:
    candidates = [
        canonical_root / f"{slug}_act_80k",
        canonical_root / f"{slug}_act_b64_80k",
    ]
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) != 1:
        names = ", ".join(str(candidate) for candidate in existing) or "<none>"
        raise RuntimeError(
            f"expected exactly one active canonical directory for {slug}; found {names}"
        )
    return existing[0]


def episode_files(directory: Path, expected: int) -> list[Path]:
    files = sorted(directory.glob("*.hdf5"), key=lambda path: natural_key(path.name))
    if len(files) != expected:
        raise RuntimeError(f"{directory}: expected {expected} episodes, found {len(files)}")
    if any(contains_deprecated(path.relative_to(directory)) for path in files):
        raise RuntimeError(f"Deprecated file found under {directory}")
    return files


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return as_text(value.item())
    return str(value)


def validate_episode(path: Path) -> int:
    with h5py.File(path, "r") as h5_file:
        if not {"state", "action", "vision"}.issubset(h5_file.keys()):
            raise ValueError(f"{path}: not an XPolicyLab canonical HDF5 file")
        lengths: set[int] = set()
        for group_name in ("state", "action"):
            group = h5_file[group_name]
            for field, width in JOINT_FIELDS:
                if field not in group:
                    raise ValueError(f"{path}: missing {group_name}/{field}")
                dataset = group[field]
                if dataset.ndim != 2 or dataset.shape[1] != width:
                    raise ValueError(
                        f"{path}: {group_name}/{field} has shape {dataset.shape}, expected (*,{width})"
                    )
                if not np.isfinite(dataset[:]).all():
                    raise ValueError(f"{path}: non-finite values in {group_name}/{field}")
                lengths.add(int(dataset.shape[0]))
        if len(lengths) != 1:
            raise ValueError(f"{path}: inconsistent state/action lengths {sorted(lengths)}")
        frames = next(iter(lengths))
        for camera_path in CAMERA_PATHS:
            if camera_path not in h5_file:
                raise ValueError(f"{path}: missing {camera_path}")
            image = h5_file[camera_path]
            if image.ndim != 4 or image.shape[0] != frames or image.shape[-1] != 3:
                raise ValueError(f"{path}: invalid {camera_path} shape {image.shape}")
        if "instruction" not in h5_file or not as_text(h5_file["instruction"][()]).strip():
            raise ValueError(f"{path}: missing instruction")
        provenance_path = "provenance/source_relative_path"
        if provenance_path in h5_file:
            source = as_text(h5_file[provenance_path][()])
            if "deprecated" in source.lower():
                raise ValueError(f"{path}: Deprecated source in provenance: {source}")
        return frames


def link_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return "symlink"


def prepare(
    raw_root: Path,
    canonical_root: Path,
    output_root: Path,
    validate_files: bool,
) -> dict[str, object]:
    raw_root = raw_root.expanduser().resolve(strict=True)
    canonical_root = canonical_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_root}. "
            "Choose a new path or remove this explicitly managed staging tree."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)

    active, deprecated, deprecated_paths = raw_inventory(raw_root)
    expected_active = sum(item[1] for item in TASKS.values())
    if active != expected_active or deprecated != 100:
        raise RuntimeError(
            f"raw inventory mismatch: active={active}, deprecated={deprecated}; "
            f"expected {expected_active}/100"
        )

    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    link_counts = {"hardlink": 0, "symlink": 0}
    task_reports: dict[str, object] = {}
    total_frames = 0
    try:
        for slug, (task_name, expected) in TASKS.items():
            source_dir = canonical_dir(canonical_root, slug)
            sources = episode_files(source_dir, expected)
            frame_count = 0
            for index, source in enumerate(sources):
                frames = validate_episode(source) if validate_files else None
                if frames is not None:
                    frame_count += frames
                destination = (
                    temporary / task_name / "ego_h1_inspire" / "data" /
                    f"episode_{index:07d}.hdf5"
                )
                kind = link_file(source, destination)
                link_counts[kind] += 1
            if validate_files:
                total_frames += frame_count
            task_reports[task_name] = {
                "slug": slug,
                "episode_count": expected,
                "canonical_source_dir": str(source_dir),
                "validated_frames": frame_count if validate_files else None,
            }
            print(f"TASK_OK {task_name} episodes={expected}", flush=True)

        manifest = {
            "format": "xpolicylab_rdt_egovla_stage_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw_root),
            "canonical_root": str(canonical_root),
            "output_root": str(output_root),
            "conversion_mode": "zero_copy_hardlink_or_relative_symlink_to_validated_canonical",
            "robot": "ego_h1_inspire",
            "action_dim": 38,
            "state_token_dim": 128,
            "frequency_hz": 30,
            "short_episode_policy": "keep_all_with_rdt_drop_short_episodes_0",
            "episode_count": expected_active,
            "validated_frame_count": total_frames if validate_files else None,
            "raw_inventory": {
                "active_episode_count": active,
                "deprecated_episode_count": deprecated,
                "deprecated_paths": deprecated_paths,
            },
            "link_counts": link_counts,
            "tasks": task_reports,
            "deprecated_policy": "exclude every path component containing Deprecated",
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "EgoVLA RDT-1B staging view.\n\n"
            "This tree contains 1,903 active canonical episodes across 12 tasks. "
            "The 100 Deprecated episodes and the close_drawer_smoke artifact are excluded. "
            "HDF5 files are hard links or relative symlinks; no episode bytes are copied.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except Exception:
        (temporary / "FAILED").write_text("staging failed\n", encoding="utf-8")
        raise

    print(
        f"ALL_OK active={active} deprecated_excluded={deprecated} "
        f"episodes={expected_active} links={link_counts} output={output_root}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw"),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/canonical"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--skip-file-validation",
        action="store_true",
        help="Validate inventory and canonical counts, but skip per-file HDF5 scans.",
    )
    args = parser.parse_args()
    prepare(
        args.raw_root,
        args.canonical_root,
        args.output_root,
        validate_files=not args.skip_file_validation,
    )


if __name__ == "__main__":
    main()
