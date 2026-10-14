#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate MoGe2-Aerial metric depth for ALL AnyVisLoc query images.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Keep these before numpy/torch where possible.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import h5py


DEFAULT_AERIALMETRIC_ROOT = "/ff/yhr/AerialMetric"
DEFAULT_ANYVISLOC_CODE_ROOT = "/ff/yhr/anyVisLoc/avl_demo"
DEFAULT_DATASET_ROOT = "/ff/yhr/anyVisLoc/AnyVisLoc_open_png_npy_json"
DEFAULT_SAVE_DIR = "/ff/yhr/MDE_Exp/AnyVisLoc_moge2_aerial"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run MoGe2-Aerial on AnyVisLoc and store ALL depth maps in one HDF5 file.",
    )
    p.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT, type=str)
    p.add_argument("--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    p.add_argument("--h5_name", default="anyvisloc_moge2_aerial_depth.h5", type=str)
    p.add_argument("--anyvisloc_code_root", default=DEFAULT_ANYVISLOC_CODE_ROOT, type=str)
    p.add_argument("--aerialmetric_root", default=DEFAULT_AERIALMETRIC_ROOT, type=str)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    p.add_argument("--resize", default=1024, type=int)
    p.add_argument("--resolution_level", default=9, type=int)
    p.add_argument("--lora_rank", default=96, type=int)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_projection", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--apply_mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--intrinsics_mode", default="none", type=str)
    p.add_argument("--sample_ratio", default=1.0, type=float)
    p.add_argument("--max_samples_per_scene", default=0, type=int)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute a sample even if its HDF5 depth dataset already exists.")
    p.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    p.add_argument("--gzip_level", type=int, default=4)
    p.add_argument("--flush_every", type=int, default=20,
                   help="Flush HDF5 metadata/data every N processed samples.")
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def scene_number(name: str) -> int:
    m = re.search(r"(\d+)$", str(name).strip())
    if m is None:
        raise ValueError(f"Cannot parse scene ID from {name!r}")
    return int(m.group(1))


def discover_scene_names(dataset_root: str) -> Dict[int, str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    mapping: Dict[int, str] = {}
    for ref_json in sorted(root.rglob("L??_reference.json")):
        try:
            mapping.setdefault(scene_number(ref_json.parent.name), ref_json.parent.name)
        except ValueError:
            pass

    if not mapping:
        for npz in sorted(root.rglob("L??_????.npz")):
            try:
                mapping.setdefault(scene_number(npz.parent.name), npz.parent.name)
            except ValueError:
                pass
    return mapping


def resolve_scenes(
    dataset_root: str,
    requested: Optional[Sequence[str]],
) -> Optional[List[str]]:
    if not requested:
        return None

    flat: List[str] = []
    for value in requested:
        flat.extend(x.strip() for x in str(value).split(",") if x.strip())

    available = discover_scene_names(dataset_root)
    if not available:
        return flat

    out: List[str] = []
    missing: List[str] = []
    for name in flat:
        sid = scene_number(name)
        if sid not in available:
            missing.append(name)
            continue
        exact = available[sid]
        if exact not in out:
            out.append(exact)

    if missing:
        raise ValueError(
            f"Requested scenes not found: {missing}; "
            f"available={list(available.values())}"
        )
    return out


def to_python_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if torch.is_tensor(v):
        if v.numel() == 1:
            return str(v.item())
        return str(v.detach().cpu().numpy())
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return str(v.item())
    return str(v)


def sample_image_to_bgr_uint8(image: Any) -> np.ndarray:
    """Convert AnyVisLoc sample['image'] (normally RGB) to OpenCV BGR uint8."""
    if torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    arr = np.asarray(arr)

    # CHW -> HWC
    if (
        arr.ndim == 3
        and arr.shape[0] in (1, 3, 4)
        and arr.shape[-1] not in (1, 3, 4)
    ):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Unexpected AnyVisLoc image shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Image contains no finite pixels.")
        lo = float(finite.min())
        hi = float(finite.max())

        arr = arr.astype(np.float32, copy=False)
        if lo >= -1.05 and hi <= 1.05 and lo < -0.01:
            # common [-1, 1] normalization
            arr = (arr + 1.0) * 127.5
        elif hi <= 1.5:
            # common [0, 1]
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # AnyVisLoc sample image is treated as RGB, consistent with the previous
    # depth-generation path used for query images.
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    vals = d[valid]
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    x = (d - vmin) / (vmax - vmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    x[~valid] = 0.0
    return cv2.applyColorMap(
        (x * 255.0).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )


def deterministic_uniform_per_scene_indices(
    dataset,
    sample_ratio: float,
    max_samples_per_scene: int,
    seed: int,
) -> List[int]:
    """Select all samples by default; supports cheap deterministic subset tests."""
    n = len(dataset)
    if sample_ratio >= 1.0 and max_samples_per_scene <= 0:
        return list(range(n))

    by_scene: Dict[str, List[int]] = {}
    for i in range(n):
        sample = dataset[i]
        scene = to_python_str(sample["scene_name"])
        by_scene.setdefault(scene, []).append(i)

    rng = np.random.default_rng(int(seed))
    selected: List[int] = []
    for scene in sorted(by_scene):
        ids = np.asarray(by_scene[scene], dtype=np.int64)
        if sample_ratio < 1.0:
            k = max(1, int(round(len(ids) * sample_ratio)))
            k = min(k, len(ids))
            # Uniform in index order rather than random image-content selection.
            positions = np.linspace(0, len(ids) - 1, k).round().astype(np.int64)
            ids = ids[positions]

        if max_samples_per_scene > 0:
            ids = ids[:max_samples_per_scene]

        selected.extend(ids.tolist())

    return selected


def _safe_h5_component(s: str) -> str:
    # AnyVisLoc scene/sample IDs normally contain no '/', but guard anyway.
    return str(s).replace("/", "__slash__")


def _write_scalar_or_json_attr(obj, key: str, value: Any) -> None:
    if value is None:
        obj.attrs[key] = "__NONE__"
    elif isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
        obj.attrs[key] = value
    else:
        obj.attrs[key] = json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    aerialmetric_root = Path(args.aerialmetric_root)
    anyvisloc_code_root = Path(args.anyvisloc_code_root)
    dataset_root = Path(args.dataset_root)
    save_root = Path(args.save_dir)
    h5_path = save_root / args.h5_name

    if not aerialmetric_root.exists():
        raise FileNotFoundError(f"AerialMetric root not found: {aerialmetric_root}")
    if not anyvisloc_code_root.exists():
        raise FileNotFoundError(f"AnyVisLoc code root not found: {anyvisloc_code_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    sys.path.insert(0, str(anyvisloc_code_root))
    sys.path.insert(0, str(aerialmetric_root))

    from avl_data import AnyVisLocNPZDataset
    from demo_infer import load_lora_model, infer_single, intrinsics_to_fov

    checkpoint_path = aerialmetric_root / "weights" / "Moge2-Aerial.pt"
    lora_config_path = aerialmetric_root / "MoGe" / "configs" / "train" / "v2.json"
    demo_path = aerialmetric_root / "demo_infer.py"
    for p in (checkpoint_path, lora_config_path, demo_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    scenes = resolve_scenes(str(dataset_root), args.scenes)
    dataset = AnyVisLocNPZDataset(
        root=str(dataset_root), scenes=scenes, load_reference=False,
        cache_reference=True, recursive=True,
    )
    selected = deterministic_uniform_per_scene_indices(
        dataset,
        sample_ratio=float(np.clip(args.sample_ratio, 1e-12, 1.0)),
        max_samples_per_scene=max(0, int(args.max_samples_per_scene)),
        seed=int(args.seed),
    )
    if int(args.limit) > 0:
        selected = selected[:int(args.limit)]

    save_root.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    compression_opts = int(args.gzip_level) if compression == "gzip" else None

    print("=" * 100)
    print("AnyVisLoc ALL-SCENE MoGe2-Aerial -> SINGLE HDF5")
    print(f"Dataset root       : {dataset_root}")
    print(f"Dataset samples    : {len(dataset)}")
    print(f"Selected samples   : {len(selected)}")
    print(f"Scenes             : {scenes or 'ALL'}")
    print(f"HDF5 output        : {h5_path}")
    print(f"Compression        : {args.compression}")
    print(f"Device             : {args.device}")
    print(f"Resize             : {args.resize}")
    print(f"Resolution level   : {args.resolution_level}")
    print(f"FP16               : {bool(args.fp16)}")
    print(f"Overwrite          : {bool(args.overwrite)}")
    print("=" * 100)

    # Open/create the single aggregate file before loading the model so a resume run
    # can discover whether everything is already present.
    with h5py.File(h5_path, "a", libver="latest") as h5:
        h5.attrs["format"] = "AnyVisLoc_MoGe2_Aerial_depth_v1"
        h5.attrs["dataset_root"] = str(dataset_root)
        h5.attrs["model"] = "MoGe2-Aerial"
        h5.attrs["checkpoint"] = str(checkpoint_path)
        h5.attrs["lora_config"] = str(lora_config_path)
        h5.attrs["resize_long_edge"] = int(args.resize)
        h5.attrs["resolution_level"] = int(args.resolution_level)
        h5.attrs["lora_rank"] = int(args.lora_rank)
        h5.attrs["depth_unit"] = "meter"
        h5.require_group("samples")

        # IMPORTANT:
        # Do NOT pre-scan selected samples via dataset[index] here.
        # AnyVisLocNPZDataset.__getitem__ loads the complete compressed NPZ,
        # so a pre-scan would read all 20k samples once before inference and
        # then read them again during the real loop. Resume/skip is therefore
        # checked lazily after each sample is loaded exactly once.
        print("Resume check        : lazy per-sample (no full NPZ pre-scan)")
        print("Loading MoGe2-Aerial once...")
        load_start = time.perf_counter()
        model = load_lora_model(
            config_path=str(lora_config_path),
            lora_path=str(checkpoint_path),
            lora_rank=int(args.lora_rank),
            device=str(args.device),
            fp16=bool(args.fp16 and str(args.device).startswith("cuda")),
        )
        print(f"Model loaded in {time.perf_counter() - load_start:.2f} s")

        success = skipped = failed = 0
        total_inference_time = 0.0
        processed_since_flush = 0

        for seq, dataset_index in enumerate(selected, start=1):
            sample = dataset[int(dataset_index)]
            scene_name = to_python_str(sample["scene_name"])
            sample_id = to_python_str(sample["sample_id"])
            npz_path = to_python_str(sample.get("npz_path", ""))
            scene_key = _safe_h5_component(scene_name)
            sid_key = _safe_h5_component(sample_id)
            group_path = f"samples/{scene_key}/{sid_key}"
            depth_path = f"{group_path}/depth"
            prefix = f"[{seq:05d}/{len(selected):05d}] {scene_name}/{sample_id}"

            if depth_path in h5 and not bool(args.overwrite):
                grp_existing = h5[group_path]
                if bool(grp_existing.attrs.get("complete", False)):
                    print(f"[skip]  {prefix}")
                    skipped += 1
                    continue
                # Interrupted/partial group: remove stale depth and recompute.
                print(f"[redo]  {prefix}  incomplete HDF5 group")
                if "depth" in grp_existing:
                    del grp_existing["depth"]

            try:
                image_bgr = sample_image_to_bgr_uint8(sample["image"])
                orig_h, orig_w = image_bgr.shape[:2]

                depth, intrinsics, elapsed = infer_single(
                    model=model,
                    img_bgr=image_bgr,
                    device=str(args.device),
                    fp16=bool(args.fp16 and str(args.device).startswith("cuda")),
                    resize=int(args.resize),
                    resolution_level=int(args.resolution_level),
                    force_projection=bool(args.force_projection),
                    apply_mask=bool(args.apply_mask),
                    intrinsics_mode=str(args.intrinsics_mode),
                    given_fov_x=None,
                )

                depth = np.asarray(depth, dtype=np.float32)
                if depth.ndim != 2:
                    depth = np.squeeze(depth)
                if depth.ndim != 2:
                    raise RuntimeError(f"Unexpected MoGe2 depth shape: {depth.shape}")
                if depth.shape != (orig_h, orig_w):
                    depth = cv2.resize(
                        depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                    ).astype(np.float32)

                valid = np.isfinite(depth) & (depth > 0)
                if not np.any(valid):
                    raise RuntimeError("Depth map contains no finite positive values.")
                vals = depth[valid]

                if intrinsics is not None:
                    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
                    try:
                        fov_info = intrinsics_to_fov(intrinsics_np, orig_w, orig_h)
                    except Exception:
                        fov_info = None
                else:
                    intrinsics_np = None
                    fov_info = None

                grp = h5.require_group(group_path)
                if "depth" in grp:
                    del grp["depth"]

                kwargs = {"compression": compression, "shuffle": True}
                if compression_opts is not None:
                    kwargs["compression_opts"] = compression_opts
                if compression is None:
                    kwargs.pop("compression")
                    kwargs.pop("shuffle")

                ds = grp.create_dataset(
                    "depth",
                    data=depth.astype(np.float32, copy=False),
                    dtype=np.float32,
                    chunks=True,
                    **kwargs,
                )

                meta = {
                    "dataset": "AnyVisLoc",
                    "scene": scene_name,
                    "sample_id": sample_id,
                    "dataset_index": int(dataset_index),
                    "npz_path": npz_path,
                    "orig_w": int(orig_w),
                    "orig_h": int(orig_h),
                    "infer_w": int(depth.shape[1]),
                    "infer_h": int(depth.shape[0]),
                    "depth_unit": "meter",
                    "depth_min": float(vals.min()),
                    "depth_max": float(vals.max()),
                    "depth_mean": float(vals.mean()),
                    "depth_median": float(np.median(vals)),
                    "n_valid": int(valid.sum()),
                    "valid_ratio": float(valid.mean()),
                    "model": "MoGe2-Aerial",
                    "checkpoint": str(checkpoint_path),
                    "lora_config": str(lora_config_path),
                    "lora_rank": int(args.lora_rank),
                    "resize_long_edge": int(args.resize),
                    "resolution_level": int(args.resolution_level),
                    "fp16": bool(args.fp16 and str(args.device).startswith("cuda")),
                    "device": str(args.device),
                    "force_projection": bool(args.force_projection),
                    "apply_mask": bool(args.apply_mask),
                    "intrinsics_mode": str(args.intrinsics_mode),
                    "predicted_fov": fov_info,
                    "inference_time_s": float(elapsed),
                }
                for k, v in meta.items():
                    _write_scalar_or_json_attr(grp, k, v)

                if "predicted_intrinsics" in grp:
                    del grp["predicted_intrinsics"]
                if intrinsics_np is not None:
                    grp.create_dataset(
                        "predicted_intrinsics",
                        data=intrinsics_np.astype(np.float32, copy=False),
                        dtype=np.float32,
                    )

                # Marker is written LAST. A partially interrupted sample without this
                # marker will be recomputed on resume.
                grp.attrs["complete"] = True

                total_inference_time += float(elapsed)
                success += 1
                processed_since_flush += 1
                print(
                    f"[ok]    {prefix}  {orig_w}x{orig_h}  "
                    f"depth=[{vals.min():.2f},{vals.max():.2f}]m  "
                    f"median={np.median(vals):.2f}m  {elapsed:.2f}s"
                )

                if processed_since_flush >= max(1, int(args.flush_every)):
                    h5.flush()
                    processed_since_flush = 0

            except Exception as exc:
                failed += 1
                print(f"[error] {prefix}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        h5.attrs["selected_samples"] = int(len(selected))
        h5.attrs["last_success"] = int(success)
        h5.attrs["last_skipped"] = int(skipped)
        h5.attrs["last_failed"] = int(failed)
        h5.flush()

    print()
    print("=" * 100)
    print("Finished")
    print(f"Success            : {success}")
    print(f"Skipped            : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Inference total(s) : {total_inference_time:.2f}")
    if success > 0:
        print(f"Inference avg(s)   : {total_inference_time / success:.3f}")
    print(f"Single HDF5        : {h5_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
