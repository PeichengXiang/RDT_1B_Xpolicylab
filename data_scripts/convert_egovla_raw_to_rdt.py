#!/usr/bin/env python3
"""Convert EgoVLA raw H1-Inspire episodes into the RDT XPolicyLab HDF5 ABI.

The raw benchmark files contain a 50-D native H1 vector in
``observations/qpos`` and ``action`` plus one RGB camera in
``observations/images/main``.  RDT consumes the 38 controllable values in the
XPolicyLab canonical layout.  This converter writes only the small state/action
and metadata datasets; image frames remain zero-copy HDF5 external links to the
raw files.

The action contract is deliberately explicit: ``action[t]`` is copied to the
canonical action at the same timestamp.  No next-state substitution or temporal
shift is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


TASKS = OrderedDict(
    [
        ("Close-Drawer", (50, "close_drawer", "Close the opened drawer")),
        ("Flip-Mug", (100, "flip_mug", "Flip the mug")),
        (
            "Insert-And-Unload-Cans",
            (
                900,
                "insert_and_unload_cans",
                "Insert the left can into the slot and insert the right can into the slot, unload the left cans andd then unload the right cans",
            ),
        ),
        ("Insert-Cans", (100, "insert_cans", "Insert cans into the boxes")),
        ("Open-Drawer", (100, "open_drawer", "Open the closed drawer")),
        ("Open-Laptop", (100, "open_laptop", "open the laptop")),
        ("Pour-Balls", (102, "pour_balls", "pour balls in cup into bowl")),
        ("Push-Box", (100, "push_box", "push box to the marker")),
        (
            "Sort-Cans",
            (101, "sort_cans", "Put sprite cans to the left box, and orange cans to the right box"),
        ),
        ("Stack-Can", (100, "stack_can", "put can on the saucer")),
        (
            "Stack-Can-Into-Drawer",
            (50, "stack_can_into_drawer", "Open the drawer, and Put can on the saucer"),
        ),
        (
            "Unload-Cans",
            (100, "unload_cans", "unload the right cans and then unload the left cans"),
        ),
    ]
)

# Native H1 50-D layout -> logical [left arm 7, left hand 12, right arm 7,
# right hand 12].  These indices are also the inverse of the common bridge's
# ego_h1_inspire native scatter map.
RAW50_TO_LOGICAL38 = np.asarray(
    [4, 8, 12, 16, 20, 22, 24, 26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48,
     5, 9, 13, 17, 21, 23, 25, 31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49],
    dtype=np.int64,
)
JOINT_FIELDS = (
    ("left_arm_joint_states", 7),
    ("left_ee_joint_states", 12),
    ("right_arm_joint_states", 7),
    ("right_ee_joint_states", 12),
)


def natural_key(value: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _scalar_string(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=np.asarray(value, dtype=h5py.string_dtype("utf-8")))


def _map_native(values: np.ndarray, source: Path, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 50:
        raise ValueError(f"{source}: {name} must have shape (T, 50), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: {name} contains non-finite values")
    return values[:, RAW50_TO_LOGICAL38]


def _write_logical_group(parent: h5py.Group, name: str, values: np.ndarray) -> None:
    group = parent.create_group(name)
    offsets = (0, 7, 19, 26)
    for (field, width), start in zip(JOINT_FIELDS, offsets):
        group.create_dataset(field, data=values[:, start : start + width], dtype="f4")


def _validate_raw(source: Path):
    with h5py.File(source, "r") as raw:
        required = ("observations/qpos", "action", "observations/images/main")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"{source}: missing raw datasets {missing}")
        qpos = _map_native(raw["observations/qpos"][:], source, "observations/qpos")
        action = _map_native(raw["action"][:], source, "action")
        image = raw["observations/images/main"]
        if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"{source}: observations/images/main must be uint8 (T,H,W,3), got {image.shape}/{image.dtype}"
            )
        if image.shape[0] != qpos.shape[0] or action.shape[0] != qpos.shape[0]:
            raise ValueError(
                f"{source}: qpos/action/image lengths disagree: "
                f"{qpos.shape[0]}/{action.shape[0]}/{image.shape[0]}"
            )
        if tuple(image.shape[1:]) != (384, 384, 3):
            raise ValueError(f"{source}: expected 384x384 RGB frames, got {image.shape[1:]}")
        return qpos, action, tuple(int(v) for v in image.shape)


def _convert_episode(source: Path, destination: Path, task_name: str, instruction: str) -> dict[str, int]:
    qpos, action, image_shape = _validate_raw(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as out:
        out.attrs["format"] = "xpolicylab_canonical_egovla_rdt_v1"
        out.attrs["robot"] = "ego_h1_inspire"
        out.attrs["action_semantics"] = "raw action[t] direct; no next-state substitution or temporal shift"
        out.attrs["state_semantics"] = "raw observations/qpos[t]"
        out.attrs["image_color_order"] = "RGB"
        out.attrs["image_storage"] = "HDF5 external link to raw observations/images/main"
        out.attrs["camera_mode"] = "single head camera; wrist slots black at train and inference"
        out.attrs["image_size"] = "384x384"
        _scalar_string(out, "instruction", instruction)

        provenance = out.create_group("provenance")
        _scalar_string(provenance, "source_absolute_path", str(source))
        _scalar_string(provenance, "source_relative_path", source.name)
        _scalar_string(provenance, "action_source", "action[t]")
        _scalar_string(provenance, "state_source", "observations/qpos[t]")

        _write_logical_group(out, "state", qpos)
        _write_logical_group(out, "action", action)

        # Keep RGB payloads zero-copy. EgoVLA is a single-view benchmark:
        # the head camera is an external link to the raw RGB stream and both
        # wrist slots are explicit all-zero HWC datasets. The wrist datasets
        # are intentionally not links to the head view (or to occasional raw
        # hand cameras), so training and inference share the black-wrist ABI.
        vision = out.create_group("vision")
        head_group = vision.create_group("cam_head")
        head_group["colors"] = h5py.ExternalLink(str(source), "/observations/images/main")
        for camera in ("cam_left_wrist", "cam_right_wrist"):
            camera_group = vision.create_group(camera)
            camera_group.create_dataset(
                "colors",
                shape=image_shape,
                dtype="u1",
                chunks=(1, image_shape[1], image_shape[2], image_shape[3]),
                fillvalue=0,
            )

    return {"frames": int(qpos.shape[0]), "height": image_shape[1], "width": image_shape[2]}


def convert(raw_root: Path, output_root: Path, overwrite: bool = False) -> dict[str, object]:
    raw_root = raw_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(raw_root.rglob("*.hdf5"), key=lambda p: natural_key(str(p.relative_to(raw_root))))
    if any("deprecated" in part.lower() for p in all_files for part in p.relative_to(raw_root).parts):
        raise RuntimeError("raw_root contains a Deprecated path; refusing to train it")
    expected_total = sum(expected for expected, _, _ in TASKS.values())
    if len(all_files) != expected_total:
        raise RuntimeError(f"raw inventory has {len(all_files)} episodes, expected {expected_total}")

    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    task_reports: dict[str, object] = {}
    total_frames = 0
    try:
        for task_name, (expected, slug, instruction) in TASKS.items():
            source_dir = raw_root / task_name
            sources = sorted(source_dir.glob("*.hdf5"), key=lambda p: natural_key(p.name))
            if len(sources) != expected:
                raise RuntimeError(f"{source_dir}: found {len(sources)} episodes, expected {expected}")
            report_frames = 0
            frame_shapes = set()
            for index, source in enumerate(sources):
                destination = temporary / task_name / "ego_h1_inspire" / "data" / f"episode_{index:07d}.hdf5"
                report = _convert_episode(source, destination, task_name, instruction)
                report_frames += report["frames"]
                frame_shapes.add((report["height"], report["width"]))
            total_frames += report_frames
            task_reports[task_name] = {
                "slug": slug,
                "instruction": instruction,
                "episode_count": expected,
                "frame_count": report_frames,
                "image_shapes": sorted([list(shape) for shape in frame_shapes]),
                "action_semantics": "raw action[t] direct; no next-state substitution or temporal shift",
            }
            print(f"TASK_OK {task_name} episodes={expected} frames={report_frames}", flush=True)

        manifest = {
            "format": "xpolicylab_rdt_egovla_raw_conversion_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "conversion_mode": "38D state/action materialized; RGB frames external-linked zero-copy",
            "robot": "ego_h1_inspire",
            "action_dim": 38,
            "state_token_dim": 128,
            "frequency_hz": 30,
            "image_size": [384, 384],
            "image_color_order": "RGB",
            "camera_mode": "black_wrist",
            "action_semantics": "raw action[t] direct; no next-state substitution or temporal shift",
            "state_semantics": "raw observations/qpos[t]",
            "raw_native_action_dim": 50,
            "raw50_to_logical38": RAW50_TO_LOGICAL38.tolist(),
            "episode_count": expected_total,
            "validated_frame_count": total_frames,
            "raw_inventory": {
                "active_episode_count": expected_total,
                "deprecated_episode_count": 0,
            },
            "tasks": task_reports,
            "deprecated_policy": "no path containing Deprecated is accepted",
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "EgoVLA raw -> RDT XPolicyLab conversion.\n\n"
            "Actions are the raw HDF5 action[t] values at the same timestamp as qpos[t]. "
            "No next-state action is used. The single main RGB stream is RGB 384x384; "
            "wrist streams are black in both training and inference. Image frames are "
            "HDF5 external links to the raw files, so conversion is zero-copy for RGB.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"ALL_OK episodes={expected_total} frames={total_frames} output={output_root}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.raw_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
