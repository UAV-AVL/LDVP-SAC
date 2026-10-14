#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared AnyVisLoc data, matching, geometry and cache plumbing.

Cleaned from the uploaded AnyVisLoc runners.  This module owns only shared
infrastructure; localization algorithms live in the other modules.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

from avl_data import AnyVisLocNPZDataset
import avl_utils as avl_utils_module
from avl_utils import (
    build_anyvisloc_reference_view,
    ensure_dir,
    matching_init,
    run_pixel_match_anyvisloc,
    tensor_rgb_to_bgr_uint8,
    to_numpy,
    to_python_str,
)

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key)) for key in keys})


def method_slug(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_anyvisloc_reference_array_cache() -> None:
    """Release heavy map/DSM arrays when moving to the next scene."""
    cache = getattr(avl_utils_module, "_REF_ARRAY_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    gc.collect()


def normalize_distortion_coeffs(distortion: np.ndarray) -> Optional[np.ndarray]:
    """Return a finite OpenCV distortion vector, or None when effectively zero/invalid."""
    if distortion is None:
        return None
    dist = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if dist.size == 0:
        return None
    finite = np.isfinite(dist)
    if not np.all(finite):
        dist = np.where(finite, dist, 0.0)
    if np.all(np.abs(dist) < 1e-15):
        return None
    return dist


def undistort_uav_image(
    image_bgr: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Undistort the full UAV image while preserving image size and intrinsic K.

    We deliberately pass newCameraMatrix=K to cv2.undistort. Therefore:
      1) the undistorted raster keeps the same HxW;
      2) matcher coordinates remain ordinary pixel coordinates;
      3) the same K can be used later to normalize matched UAV pixels;
      4) distortion must NOT be applied again in AH.
    """
    image = np.asarray(image_bgr, dtype=np.uint8)
    K64 = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_distortion_coeffs(distortion)
    if dist is None:
        return image.copy(), K64.astype(np.float32), False

    undistorted = cv2.undistort(
        image,
        K64,
        dist,
        None,
        K64,
    )
    return undistorted, K64.astype(np.float32), True


def make_true_pos(sample: Dict[str, Any]) -> Dict[str, Any]:
    image_size = to_numpy(sample.get("image_size", np.array([0, 0]))).astype(np.int32).reshape(-1)
    if image_size.size >= 2 and int(image_size[0]) > 0:
        height, width = int(image_size[0]), int(image_size[1])
    else:
        image = sample["image"]
        height, width = int(image.shape[-2]), int(image.shape[-1])

    xyz = to_numpy(sample["xyz"]).astype(np.float32).reshape(3)
    euler = to_numpy(sample["euler_deg"]).astype(np.float32).reshape(3)
    K = to_numpy(sample["K"]).astype(np.float32)
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "rel_alt": float(xyz[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "width": width,
        "height": height,
        "K": K,
    }


def compute_view_center_xy(true_pos: Dict[str, Any]) -> Tuple[float, float]:
    x = float(true_pos["x"])
    y = float(true_pos["y"])
    z = float(true_pos["z"])
    pitch = float(true_pos["pitch"])
    yaw = float(true_pos["yaw"])
    tan_pitch = float(np.tan(np.deg2rad(pitch)))
    if not np.isfinite(tan_pitch):
        return x, y
    yaw_rad = np.deg2rad(yaw)
    dx = z * tan_pitch * np.sin(yaw_rad)
    dy = -z * tan_pitch * np.cos(yaw_rad)
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return x, y
    return x + float(dx), y + float(dy)


def local_to_pixel(
    x: float,
    y: float,
    map_resolution: np.ndarray,
    map_origin_local: np.ndarray,
) -> Tuple[float, float]:
    resolution = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    origin = np.asarray(map_origin_local, dtype=np.float32).reshape(-1)
    col = (float(x) - float(origin[0])) / max(float(resolution[0]), 1e-12)
    row = (float(y) - float(origin[1])) / max(float(resolution[1]), 1e-12)
    return float(col), float(row)


def pixel_to_local(
    col: float,
    row: float,
    map_resolution: np.ndarray,
    map_origin_local: np.ndarray,
) -> Tuple[float, float]:
    resolution = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    origin = np.asarray(map_origin_local, dtype=np.float32).reshape(-1)
    x = float(col) * float(resolution[0]) + float(origin[0])
    y = float(row) * float(resolution[1]) + float(origin[1])
    return x, y


def rotate_image_keep_all(
    image: np.ndarray,
    angle_deg: float,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=interpolation,
        borderValue=(0, 0, 0),
    )
    return rotated, matrix.astype(np.float32)


def even_ceil(value: float, minimum: int = 2) -> int:
    output = max(int(minimum), int(math.ceil(float(value))))
    if output % 2:
        output += 1
    return output


def estimate_drone_resolution(true_pos: Dict[str, Any]) -> float:
    K = np.asarray(true_pos["K"], dtype=np.float32)
    fx = max(float(K[0, 0]), 1e-6)
    altitude = max(float(true_pos["z"]), 1e-6)
    pitch = float(true_pos["pitch"])
    denominator = max(abs(float(np.sin(np.deg2rad(90.0 + pitch)))), 1e-6)
    return float((altitude / denominator) / fx)


def estimate_patch_size_px(
    true_pos: Dict[str, Any],
    map_resolution: np.ndarray,
    opt: argparse.Namespace,
) -> Tuple[int, int, float, float]:
    drone_resolution = estimate_drone_resolution(true_pos)
    min_image_size = min(int(true_pos["width"]), int(true_pos["height"]))
    view_size_m = min_image_size * drone_resolution * float(opt.patch_scale)

    if float(opt.min_patch_size_m) > 0:
        view_size_m = max(view_size_m, float(opt.min_patch_size_m))
    if float(opt.max_patch_size_m) > 0:
        view_size_m = min(view_size_m, float(opt.max_patch_size_m))

    resolution = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    patch_width = even_ceil(view_size_m / max(float(resolution[0]), 1e-12))
    patch_height = even_ceil(view_size_m / max(float(resolution[1]), 1e-12))

    max_pixels = int(opt.max_patch_pixels)
    if max_pixels > 0:
        patch_width = min(patch_width, max_pixels)
        patch_height = min(patch_height, max_pixels)
        if patch_width % 2:
            patch_width = max(2, patch_width - 1)
        if patch_height % 2:
            patch_height = max(2, patch_height - 1)

    return patch_width, patch_height, float(view_size_m), float(drone_resolution)


def crop_gt_reference_patch(
    reference_map: np.ndarray,
    true_pos: Dict[str, Any],
    view: Dict[str, Any],
    opt: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    map_resolution = np.asarray(view["map_resolution"], dtype=np.float32).reshape(2)
    map_origin_local = np.asarray(view["map_origin_local"], dtype=np.float32).reshape(2)
    center_x, center_y = compute_view_center_xy(true_pos)
    center_col, center_row = local_to_pixel(center_x, center_y, map_resolution, map_origin_local)
    final_width, final_height, view_size_m, drone_resolution = estimate_patch_size_px(
        true_pos, map_resolution, opt
    )

    map_height, map_width = reference_map.shape[:2]
    center_col_int = int(round(center_col))
    center_row_int = int(round(center_row))

    if opt.pose_priori == "yp":
        pre_scale = max(1.0, float(opt.pre_crop_scale))
        crop_width = int(math.ceil(final_width * pre_scale))
        crop_height = int(math.ceil(final_height * pre_scale))
    else:
        crop_width, crop_height = final_width, final_height

    x1 = max(0, center_col_int - crop_width // 2)
    y1 = max(0, center_row_int - crop_height // 2)
    x2 = min(map_width, center_col_int + crop_width // 2)
    y2 = min(map_height, center_row_int + crop_height // 2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid reference crop center=({center_col:.1f},{center_row:.1f}), "
            f"map_size=({map_width},{map_height})"
        )

    pre_patch = reference_map[y1:y2, x1:x2].copy()
    if opt.pose_priori == "yp":
        rotated, rotation_matrix = rotate_image_keep_all(pre_patch, float(true_pos["yaw"]))
        rotated_height, rotated_width = rotated.shape[:2]
        rotated_center_x = rotated_width // 2
        rotated_center_y = rotated_height // 2
        rx1 = max(0, rotated_center_x - final_width // 2)
        ry1 = max(0, rotated_center_y - final_height // 2)
        rx2 = min(rotated_width, rotated_center_x + final_width // 2)
        ry2 = min(rotated_height, rotated_center_y + final_height // 2)
        if rx2 <= rx1 or ry2 <= ry1:
            raise ValueError("Invalid yaw-rotated reference crop.")
        reference_patch = rotated[ry1:ry2, rx1:rx2].copy()
        final_crop = [int(rx1), int(ry1), int(rx2), int(ry2)]
    else:
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        reference_patch = pre_patch
        final_crop = [0, 0, int(reference_patch.shape[1]), int(reference_patch.shape[0])]

    crop_info = {
        "rotation_matrix": rotation_matrix,
        "pre_crop": [int(x1), int(y1), int(x2), int(y2)],
        "final_crop": final_crop,
        "map_resolution": map_resolution,
        "map_origin_local": map_origin_local,
        "center_x_dataset_local": float(center_x),
        "center_y_dataset_local": float(center_y),
        "center_col": float(center_col),
        "center_row": float(center_row),
        "target_patch_size_m": float(view_size_m),
        "drone_resolution_m_per_px": float(drone_resolution),
        "patch_width_m": float(reference_patch.shape[1]) * float(map_resolution[0]),
        "patch_height_m": float(reference_patch.shape[0]) * float(map_resolution[1]),
        "pose_priori": str(opt.pose_priori),
    }
    return reference_patch, crop_info


def ref_patch_pts_to_local(
    reference_points_patch: np.ndarray,
    crop_info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(reference_points_patch, dtype=np.float32).reshape(-1, 2)
    rx1, ry1, _, _ = crop_info["final_crop"]
    x1, y1, _, _ = crop_info["pre_crop"]
    inverse_rotation = cv2.invertAffineTransform(
        np.asarray(crop_info["rotation_matrix"], dtype=np.float32)
    )

    xr = points[:, 0] + float(rx1)
    yr = points[:, 1] + float(ry1)
    homogeneous = np.column_stack([xr, yr, np.ones(len(points), dtype=np.float32)])
    pre_coordinates = homogeneous @ inverse_rotation.T
    map_cols = pre_coordinates[:, 0] + float(x1)
    map_rows = pre_coordinates[:, 1] + float(y1)

    resolution = np.asarray(crop_info["map_resolution"], dtype=np.float32).reshape(2)
    origin = np.asarray(crop_info["map_origin_local"], dtype=np.float32).reshape(2)
    local_x = map_cols * resolution[0] + origin[0]
    local_y = map_rows * resolution[1] + origin[1]
    return local_x.astype(np.float32), local_y.astype(np.float32)


def resize_for_matching(
    uav_bgr: np.ndarray,
    reference_patch: np.ndarray,
    true_pos: Dict[str, Any],
    map_resolution: np.ndarray,
    opt: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    requested_ratio = float(opt.resize_ratio)
    drone_resolution = estimate_drone_resolution(true_pos)
    reference_resolution = float(np.mean(np.asarray(map_resolution, dtype=np.float32).reshape(-1)))
    fine_scale = drone_resolution / max(reference_resolution, 1e-12)

    if requested_ratio < 1.0:
        uav_match = cv2.resize(
            uav_bgr,
            None,
            fx=requested_ratio,
            fy=requested_ratio,
            interpolation=cv2.INTER_AREA,
        )
        uav_scale = requested_ratio
    else:
        uav_match = uav_bgr
        uav_scale = 1.0

    reference_resize = uav_scale / max(fine_scale, 1e-12)
    if reference_resize <= 0:
        raise ValueError(f"Invalid reference resize factor: {reference_resize}")
    interpolation = cv2.INTER_AREA if reference_resize < 1.0 else cv2.INTER_LINEAR
    reference_match = cv2.resize(
        reference_patch,
        None,
        fx=reference_resize,
        fy=reference_resize,
        interpolation=interpolation,
    )
    return uav_match, reference_match, float(fine_scale), float(reference_resize), float(uav_scale)


def uniform_sample_indices_per_scene(
    dataset: AnyVisLocNPZDataset,
    sample_ratio: float,
    max_samples_per_scene: int = 0,
) -> Tuple[List[int], Dict[str, Dict[str, int]]]:
    scene_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, scene_name in enumerate(dataset.sample_scene):
        scene_to_indices[str(scene_name)].append(index)

    selected: List[int] = []
    report: Dict[str, Dict[str, int]] = {}
    for scene_name in sorted(scene_to_indices):
        indices = scene_to_indices[scene_name]
        total = len(indices)
        target = max(1, int(math.ceil(total * float(sample_ratio))))
        if int(max_samples_per_scene) > 0:
            target = min(target, int(max_samples_per_scene))
        target = min(target, total)
        positions = np.linspace(0, total - 1, num=target, dtype=np.int64)
        chosen = [indices[int(position)] for position in positions]
        selected.extend(chosen)
        report[scene_name] = {"total": total, "selected": len(chosen)}
    return selected, report


@dataclass
class CommonSampleData:
    scene_name: str
    sample_id: str
    npz_path: str
    sample_dir: Path
    true_pos: Dict[str, Any]
    K: np.ndarray
    distortion: np.ndarray
    uav_was_undistorted: bool
    uav_bgr: np.ndarray
    reference_map: np.ndarray
    reference_patch: np.ndarray
    crop_info: Dict[str, Any]
    view: Dict[str, Any]
    uav_match: np.ndarray
    reference_match: np.ndarray
    uav_points_match: np.ndarray
    reference_points_match: np.ndarray
    uav_points_original: np.ndarray
    reference_points_patch: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray
    fine_scale: float
    reference_resize: float
    uav_scale: float
    match_time_s: float
    reference_mode: str
    match_cache_path: Optional[str] = None
    match_cache_reused: bool = False


@dataclass
class GeometryMatchData:
    indices: np.ndarray
    image_points: np.ndarray
    map_xy_world: np.ndarray
    rays_camera: np.ndarray


class MatcherManager:
    """Lazy matcher initialization: cache-only runs never initialize a matcher."""
    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self._method_dict: Optional[Dict[str, Any]] = None

    def get_method_dict(self) -> Dict[str, Any]:
        if self._method_dict is None:
            yaml_path = Path(self.opt.yaml)
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"Matcher config not found: {yaml_path}. It is only required when a cache miss needs live matching."
                )
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            config["DEVICE"] = str(self.opt.device)
            method_dict: Dict[str, Any] = {"matching_method": str(self.opt.matching_method)}
            self._method_dict = matching_init(method_dict, self.opt, config)
        return self._method_dict


def _match_cache_name(uav_was_undistorted: bool) -> str:
    return "match_points_undistorted.npz" if uav_was_undistorted else "match_points_raw.npz"


def _cache_method_compatible(cached_method: Optional[Any], requested_method: str) -> bool:
    """Legacy caches without a matcher tag are treated as RoMa caches only."""
    requested = str(requested_method).strip().lower()
    if cached_method is None:
        return requested == "roma"
    if isinstance(cached_method, bytes):
        cached_method = cached_method.decode("utf-8", errors="ignore")
    return str(cached_method).strip().lower() == requested


def load_or_run_matches(
    uav_match: np.ndarray,
    reference_match: np.ndarray,
    output_sample_dir: Path,
    cache_sample_dir: Optional[Path],
    opt: argparse.Namespace,
    matcher_manager: MatcherManager,
    *,
    uav_was_undistorted: bool,
) -> Tuple[np.ndarray, np.ndarray, float, Path, bool]:
    """Load external/current match cache first; run matcher only on a cache miss."""
    cache_name = _match_cache_name(bool(uav_was_undistorted))
    candidates: List[Path] = []
    if cache_sample_dir is not None:
        candidates.append(Path(cache_sample_dir) / cache_name)
    candidates.append(Path(output_sample_dir) / cache_name)

    if not bool(opt.force_rematch):
        for path in candidates:
            if path.exists():
                with np.load(path) as data:
                    cached_method = None
                    if "matching_method" in data.files:
                        values = np.asarray(data["matching_method"]).reshape(-1)
                        if len(values):
                            cached_method = values[0]
                    if not _cache_method_compatible(cached_method, opt.matching_method):
                        continue
                    return (
                        data["uav_points_match"].astype(np.float32),
                        data["reference_points_match"].astype(np.float32),
                        0.0,
                        path,
                        True,
                    )

    if bool(opt.cache_only):
        tried = "\n  ".join(str(x) for x in candidates)
        raise FileNotFoundError(f"Match cache missing and --cache_only is set. Tried:\n  {tried}")

    method_dict = matcher_manager.get_method_dict()
    output_sample_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    uav_points, reference_points = run_pixel_match_anyvisloc(
        uav_match,
        reference_match,
        method_dict,
        str(output_sample_dir),
        "/match_raw.jpg",
        need_ransac=False,
        show_matches=bool(getattr(opt, "show_match_image", False)),
    )
    elapsed = time.time() - start
    uav_points = np.asarray(uav_points, dtype=np.float32).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    n = min(len(uav_points), len(reference_points))
    uav_points, reference_points = uav_points[:n], reference_points[:n]
    out_cache = output_sample_dir / cache_name
    np.savez_compressed(
        out_cache,
        uav_points_match=uav_points,
        reference_points_match=reference_points,
        matching_method=np.asarray([str(opt.matching_method)]),
    )
    return uav_points, reference_points, float(elapsed), out_cache, False


def prepare_common_sample_data(
    sample: Dict[str, Any],
    run_root: Path,
    opt: argparse.Namespace,
    matcher_manager: MatcherManager,
) -> CommonSampleData:
    scene_name = to_python_str(sample["scene_name"])
    sample_id = to_python_str(sample["sample_id"])
    sample_dir = Path(run_root) / scene_name / sample_id
    ensure_dir(sample_dir)

    true_pos = make_true_pos(sample)
    K_raw = to_numpy(sample["K"]).astype(np.float32)
    distortion = to_numpy(sample["dist"]).astype(np.float32)
    uav_bgr_raw = tensor_rgb_to_bgr_uint8(sample["image"])

    if bool(opt.disable_uav_undistortion):
        uav_bgr = uav_bgr_raw
        K = K_raw
        uav_was_undistorted = False
    else:
        uav_bgr, K, uav_was_undistorted = undistort_uav_image(uav_bgr_raw, K_raw, distortion)

    view = build_anyvisloc_reference_view(sample, str(opt.reference_mode))
    reference_map = view["map"]
    reference_patch, crop_info = crop_gt_reference_patch(reference_map, true_pos, view, opt)
    uav_match, reference_match, fine_scale, reference_resize, uav_scale = resize_for_matching(
        uav_bgr, reference_patch, true_pos, view["map_resolution"], opt
    )

    cache_sample_dir = None
    if getattr(opt, "match_cache_root", None):
        cache_sample_dir = Path(opt.match_cache_root) / scene_name / sample_id

    uav_points_match, reference_points_match, match_time, cache_path, cache_reused = load_or_run_matches(
        uav_match,
        reference_match,
        sample_dir,
        cache_sample_dir,
        opt,
        matcher_manager,
        uav_was_undistorted=bool(uav_was_undistorted),
    )
    uav_points_original = uav_points_match / max(float(uav_scale), 1e-12)
    reference_points_patch = reference_points_match / max(float(reference_resize), 1e-12)
    local_x, local_y = ref_patch_pts_to_local(reference_points_patch, crop_info)

    common = CommonSampleData(
        scene_name=scene_name,
        sample_id=sample_id,
        npz_path=to_python_str(sample.get("npz_path", "")),
        sample_dir=sample_dir,
        true_pos=true_pos,
        K=K,
        distortion=distortion,
        uav_was_undistorted=bool(uav_was_undistorted),
        uav_bgr=uav_bgr,
        reference_map=reference_map,
        reference_patch=reference_patch,
        crop_info=crop_info,
        view=view,
        uav_match=uav_match,
        reference_match=reference_match,
        uav_points_match=uav_points_match,
        reference_points_match=reference_points_match,
        uav_points_original=uav_points_original,
        reference_points_patch=reference_points_patch,
        local_x=local_x,
        local_y=local_y,
        fine_scale=float(fine_scale),
        reference_resize=float(reference_resize),
        uav_scale=float(uav_scale),
        match_time_s=float(match_time),
        reference_mode=str(opt.reference_mode),
        match_cache_path=str(cache_path),
        match_cache_reused=bool(cache_reused),
    )

    save_json(sample_dir / "common_metadata.json", {
        "scene_name": scene_name,
        "sample_id": sample_id,
        "npz_path": common.npz_path,
        "reference_mode": common.reference_mode,
        "matching_method": str(opt.matching_method),
        "pose_priori": opt.pose_priori,
        "num_matches": int(len(uav_points_match)),
        "fine_scale": float(fine_scale),
        "reference_resize": float(reference_resize),
        "uav_scale": float(uav_scale),
        "match_time_s": float(match_time),
        "match_cache_path": str(cache_path),
        "match_cache_reused": bool(cache_reused),
        "uav_was_undistorted": bool(uav_was_undistorted),
        "distortion_coefficients": np.asarray(distortion).reshape(-1).tolist(),
        "camera_matrix_used_for_matching_and_geometry": np.asarray(K).tolist(),
        "true_pos": true_pos,
        "crop_info": crop_info,
    })
    return common


def camera_rays_for_common(common: CommonSampleData) -> np.ndarray:
    """Ideal pinhole rays corresponding to common.uav_points_original."""
    image_points = np.asarray(common.uav_points_original, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(common.K, dtype=np.float64).reshape(3, 3)
    if bool(common.uav_was_undistorted):
        K_inv = np.linalg.inv(K)
        hom = np.column_stack([image_points, np.ones(len(image_points), dtype=np.float64)])
        rays = (K_inv @ hom.T).T
        rays /= np.clip(rays[:, 2:3], 1e-12, None)
        return rays
    dist = normalize_distortion_coeffs(common.distortion)
    xy = cv2.undistortPoints(image_points.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy), dtype=np.float64)])


def build_geometry_match_data(common: CommonSampleData) -> GeometryMatchData:
    image_points = np.asarray(common.uav_points_original, dtype=np.float64).reshape(-1, 2)
    map_xy_world = np.column_stack([
        np.asarray(common.local_x, dtype=np.float64).reshape(-1),
        -np.asarray(common.local_y, dtype=np.float64).reshape(-1),
    ])
    rays = np.asarray(camera_rays_for_common(common), dtype=np.float64).reshape(-1, 3)
    n = min(len(image_points), len(map_xy_world), len(rays))
    image_points, map_xy_world, rays = image_points[:n], map_xy_world[:n], rays[:n]
    valid = (
        np.isfinite(image_points).all(axis=1)
        & np.isfinite(map_xy_world).all(axis=1)
        & np.isfinite(rays).all(axis=1)
    )
    idx = np.where(valid)[0].astype(np.int64)
    return GeometryMatchData(
        indices=idx,
        image_points=image_points[idx],
        map_xy_world=map_xy_world[idx],
        rays_camera=rays[idx],
    )


def verify_reference_mode(match_cache_root: Optional[str], requested_mode: str) -> None:
    if not match_cache_root:
        return
    root = Path(match_cache_root)
    if not root.exists():
        raise FileNotFoundError(f"match_cache_root not found: {root}")
    args_file = root / "run_args.json"
    if not args_file.exists():
        return
    try:
        payload = json.loads(args_file.read_text(encoding="utf-8"))
    except Exception:
        return
    cached = payload.get("reference_mode")
    if cached is None and isinstance(payload.get("args"), dict):
        cached = payload["args"].get("reference_mode")
    if cached is not None and str(cached).strip().lower() != str(requested_mode).strip().lower():
        raise ValueError(f"reference_mode mismatch: cache={cached}, requested={requested_mode}")
