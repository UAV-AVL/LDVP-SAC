#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


SPLITS = ("train", "val", "test_inPlace", "test_outPlace")

QUERY_KEYS = (
    "query_points_original",
    "uav_points_original",
    "image_points_original",
    "q_pts",
    "query_points",
    "uav_points",
    "image_points_query",
    "query_pts",
    "query_keypoints",
    "mkpts0",
    "points0",
    "keypoints0",
    "kpts0",
    "pts0",
)

REF_ORIGINAL_KEYS = (
    "ref_points_original",
    "reference_points_original",
    "map_points_original",
    "dop_points_original",
    "ortho_points_original",
    "reference_pixels_original",
    "ref_pixels_original",
)

REF_ROTATED_KEYS = (
    "ref_points_rotated",
    "reference_points_rotated",
    "ref_points_match",
    "reference_points_match",
    "matched_ref_points",
    "points1",
    "mkpts1",
    "keypoints1",
    "kpts1",
    "pts1",
)

WORLD_XYZ_KEYS = (
    "world_xyz",
    "map_xyz",
    "matched_world_xyz",
    "matched_xyz",
    "reference_world_xyz",
    "ref_world_xyz",
    "xyz",
)

ROTATION_KEYS = (
    "rotation_M",
    "ref_rotation_M",
    "reference_rotation_M",
)


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
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def finite_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _nx2(a: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(a)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 2 or not np.issubdtype(a.dtype, np.number):
        return None
    return a[:, :2].astype(np.float64)


def _nx3(a: Any) -> Optional[np.ndarray]:
    try:
        a = np.asarray(a)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 3 or not np.issubdtype(a.dtype, np.number):
        return None
    return a[:, :3].astype(np.float64)


def split_names(set_name: str) -> List[str]:
    if set_name == "all":
        return list(SPLITS)
    if set_name not in SPLITS:
        raise ValueError(f"invalid set_name={set_name!r}")
    return [set_name]


def locate_dataset_root(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"OrthoLoC dataset root does not exist: {root}")
    full = root / "full"
    return full if full.is_dir() else root


def list_dataset_queries(
    dataset_dir: str | Path,
    set_name: str = "all",
    sample_glob: Optional[str] = None,
    sample_ratio: float = 1.0,
    seed: int = 0,
    limit: int = 0,
) -> List[Tuple[str, Path]]:
    root = locate_dataset_root(dataset_dir)
    items: List[Tuple[str, Path]] = []

    if sample_glob:
        paths = sorted(root.glob(sample_glob))
        for p in paths:
            if p.suffix.lower() != ".npz":
                continue
            split = next((x for x in SPLITS if x in p.parts), p.parent.name)
            if set_name == "all" or split == set_name:
                items.append((split, p))
    else:
        for split in split_names(set_name):
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            items.extend((split, p) for p in sorted(split_dir.glob("*.npz")))

    ratio = float(sample_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("--sample_ratio must be in (0, 1].")
    if ratio < 1.0 and items:
        n_keep = max(1, int(round(len(items) * ratio)))
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(items), size=n_keep, replace=False))
        items = [items[int(i)] for i in idx]

    if int(limit) > 0:
        items = items[: int(limit)]
    return items


def invert_world2camera_pose(extrinsics: np.ndarray) -> np.ndarray:
    """Invert a 3x4 or 4x4 world->camera rigid transform into 4x4 camera->world."""
    E = np.asarray(extrinsics, dtype=np.float64)
    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = E
        E = T
    if E.shape != (4, 4):
        raise ValueError(f"unsupported extrinsics shape: {E.shape}")
    R = E[:3, :3]
    t = E[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def load_ortholoc_sample(
    npz_path: str | Path,
    use_refined_extrinsics: bool = False,
) -> Dict[str, Any]:
    """
    Minimal direct OrthoLoC NPZ loader.

    The benchmark only needs image_query, image_dop, DSM world XYZ, query K and
    query pose.  It deliberately does not construct dense GT correspondences and
    does not need the archive key ``scale``.
    """
    p = Path(npz_path)
    with np.load(p, allow_pickle=True) as z:
        required = ("image_query", "image_dop", "dsm", "intrinsics")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"{p}: missing required NPZ fields: {missing}")

        image_query = np.asarray(z["image_query"])
        image_dop = np.asarray(z["image_dop"])
        dsm = np.asarray(z["dsm"], dtype=np.float64)
        K = np.asarray(z["intrinsics"], dtype=np.float64).reshape(3, 3)

        extr_key = (
            "extrinsics_refined"
            if use_refined_extrinsics and "extrinsics_refined" in z.files
            else "extrinsics"
        )
        if extr_key not in z.files:
            raise KeyError(f"{p}: neither usable extrinsics nor extrinsics_refined found")
        pose_c2w = invert_world2camera_pose(np.asarray(z[extr_key], dtype=np.float64))

        archive_sample_id = None
        if "sample_id" in z.files:
            try:
                raw = np.asarray(z["sample_id"])
                archive_sample_id = str(raw.item() if raw.size == 1 else raw)
            except Exception:
                archive_sample_id = None

    if dsm.ndim != 3 or dsm.shape[2] < 3:
        raise ValueError(f"{p}: expected DSM HxWx3 world XYZ, got {dsm.shape}")
    dsm = dsm[..., :3].copy()
    mask_dsm = np.isfinite(dsm).all(axis=2)
    dsm[~mask_dsm] = np.nan

    if image_dop.shape[:2] != dsm.shape[:2]:
        raise ValueError(
            f"{p}: image_dop shape {image_dop.shape[:2]} != DSM shape {dsm.shape[:2]}"
        )

    return {
        "sample_id": p.stem,
        "sample_id_original": archive_sample_id,
        "npz_path": str(p),
        "image_query": image_query,
        "image_dop": image_dop,
        "dsm": dsm,
        "mask_dsm": mask_dsm,
        "intrinsics_query": K,
        "pose_query2world": pose_c2w,
        "loader": "direct_npz_minimal_v1",
        "extrinsics_key": extr_key,
    }


def camera_center_from_c2w(pose_c2w: np.ndarray) -> np.ndarray:
    T = np.asarray(pose_c2w, dtype=np.float64)
    if T.shape == (3, 4):
        return T[:3, 3].copy()
    if T.shape == (4, 4):
        return T[:3, 3].copy()
    raise ValueError(f"unsupported camera-to-world pose shape: {T.shape}")


def extract_yaw_deg_from_pose_c2w(pose_c2w: np.ndarray) -> float:
    T = np.asarray(pose_c2w, dtype=np.float64)
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T
        T = T4
    if T.shape != (4, 4):
        raise ValueError(f"unsupported pose shape for yaw: {T.shape}")
    forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(math.degrees(math.atan2(forward_world[0], forward_world[1])))


def rotate_image_keep_all(
    image: np.ndarray,
    angle_deg: float,
    interp: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    c = abs(M[0, 0])
    s = abs(M[0, 1])
    new_w = int((h * s) + (w * c))
    new_h = int((h * c) + (w * s))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=interp)
    return rotated, M


def prepare_ref_image_and_info(
    image_dop: np.ndarray,
    pose_c2w: np.ndarray,
    rotate_ref_by_gt_yaw: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ref = np.asarray(image_dop)
    info: Dict[str, Any] = {
        "rotated": False,
        "rotation_M": np.eye(2, 3, dtype=np.float64),
        "yaw_deg": 0.0,
        "orig_h": int(ref.shape[0]),
        "orig_w": int(ref.shape[1]),
        "rot_h": int(ref.shape[0]),
        "rot_w": int(ref.shape[1]),
    }
    if not rotate_ref_by_gt_yaw:
        return ref.copy(), info

    yaw = extract_yaw_deg_from_pose_c2w(pose_c2w)
    candidates = (0, 90, 180, 270)
    snapped = min(
        candidates,
        key=lambda a: abs((yaw - a + 180.0) % 360.0 - 180.0),
    )
    ref_rot, M = rotate_image_keep_all(ref, snapped, interp=cv2.INTER_LINEAR)
    info.update(
        {
            "rotated": True,
            "rotation_M": np.asarray(M, dtype=np.float64),
            "yaw_deg": float(snapped),
            "rot_h": int(ref_rot.shape[0]),
            "rot_w": int(ref_rot.shape[1]),
        }
    )
    return ref_rot, info


def inverse_affine_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    M_inv = cv2.invertAffineTransform(np.asarray(M, dtype=np.float64).reshape(2, 3))
    ph = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return ph @ M_inv.T


def index_dsm_world_xyz(
    dsm: np.ndarray,
    ref_points_original: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(ref_points_original, dtype=np.float64).reshape(-1, 2)
    dsm = np.asarray(dsm, dtype=np.float64)
    h, w = dsm.shape[:2]
    col = np.rint(pts[:, 0]).astype(np.int64)
    row = np.rint(pts[:, 1]).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    xyz = np.full((len(pts), 3), np.nan, dtype=np.float64)
    xyz[inside] = dsm[row[inside], col[inside], :3]
    valid = inside & np.isfinite(xyz).all(axis=1)
    return xyz, valid


@dataclass
class MatchCacheIndex:
    root: Path
    by_sample_id: Dict[str, List[Path]]
    all_paths: List[Path]


def build_match_cache_index(match_root: Optional[str | Path]) -> Optional[MatchCacheIndex]:
    if not match_root:
        return None
    root = Path(match_root)
    if not root.exists():
        raise FileNotFoundError(f"match_root does not exist: {root}")

    all_paths = sorted(root.rglob("*.npz"))
    by_id: Dict[str, List[Path]] = {}
    generic_stems = {
        "match_geometry", "matches", "match", "match_points_raw",
        "match_points_undistorted", "correspondences",
    }
    for p in all_paths:
        candidates = {p.parent.name}
        if p.stem not in generic_stems:
            candidates.add(p.stem)
        # Many old runs have .../<split>/<sample_id>/match_geometry.npz.
        if len(p.parents) >= 2:
            candidates.add(p.parents[0].name)
        for sid in candidates:
            by_id.setdefault(sid, []).append(p)
    return MatchCacheIndex(root=root, by_sample_id=by_id, all_paths=all_paths)


def _cache_candidate_score(path: Path, split: str, sample_id: str) -> Tuple[int, int, int, str]:
    parts = set(path.parts)
    exact_name = int(path.name == "match_geometry.npz")
    sid_in_path = int(sample_id in parts or path.stem == sample_id)
    split_in_path = int(split in parts)
    # Higher is better.  Final string makes ordering deterministic.
    return (sid_in_path, split_in_path, exact_name, str(path))


def find_match_cache(
    index: Optional[MatchCacheIndex],
    split: str,
    sample_id: str,
) -> Optional[Path]:
    if index is None:
        return None

    direct = [
        index.root / split / sample_id / "match_geometry.npz",
        index.root / sample_id / "match_geometry.npz",
        index.root / split / sample_id / "matches.npz",
        index.root / sample_id / "matches.npz",
    ]
    for p in direct:
        if p.is_file():
            return p

    cands = list(index.by_sample_id.get(sample_id, []))
    if not cands:
        # Last-resort substring match, only against filenames/path components.
        cands = [p for p in index.all_paths if sample_id in p.parts or sample_id in p.stem]
    if not cands:
        return None
    return max(cands, key=lambda p: _cache_candidate_score(p, split, sample_id))


def _find_array(
    z: Mapping[str, Any],
    keys: Sequence[str],
    nx: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for k in keys:
        if k not in z:
            continue
        a = _nx2(z[k]) if nx == 2 else _nx3(z[k])
        if a is not None:
            return a, k
    return None, None


def load_match_geometry(
    cache_path: str | Path,
    sample: Dict[str, Any],
    allow_reconstruct_gt_yaw_rotation: bool = True,
) -> Dict[str, Any]:
    """
    Load historical/current OrthoLoC match caches.

    Geometry preference:
      1) query pixels + row-aligned cached world_xyz
      2) query + original reference pixels -> DSM lookup
      3) query + rotated reference pixels -> inverse rotation -> DSM lookup
    """
    p = Path(cache_path)
    with np.load(p, allow_pickle=True) as zf:
        z = {k: zf[k] for k in zf.files}
        keys = list(zf.files)

    cached_matching_method = None
    if "matching_method" in z:
        values = np.asarray(z["matching_method"]).reshape(-1)
        if len(values):
            cached_matching_method = values[0]
            if isinstance(cached_matching_method, bytes):
                cached_matching_method = cached_matching_method.decode("utf-8", errors="ignore")
            cached_matching_method = str(cached_matching_method)

    q, qkey = _find_array(z, QUERY_KEYS, 2)
    if q is None:
        for k in ("matches", "mkpts", "correspondences", "pairs"):
            if k in z:
                a = np.asarray(z[k])
                if a.ndim == 2 and a.shape[1] >= 4 and np.issubdtype(a.dtype, np.number):
                    q = a[:, :2].astype(np.float64)
                    qkey = f"{k}[:,0:2]"
                    break
    if q is None:
        raise KeyError(f"{p}: no usable query matched-point array; keys={keys}")

    xyz, xyzkey = _find_array(z, WORLD_XYZ_KEYS, 3)

    ref, refkey = _find_array(z, REF_ORIGINAL_KEYS, 2)
    ref_is_rotated = False
    if ref is None:
        ref, refkey = _find_array(z, REF_ROTATED_KEYS, 2)
        ref_is_rotated = ref is not None
    if ref is None:
        for k in ("matches", "mkpts", "correspondences", "pairs"):
            if k in z:
                a = np.asarray(z[k])
                if a.ndim == 2 and a.shape[1] >= 4 and np.issubdtype(a.dtype, np.number):
                    ref = a[:, 2:4].astype(np.float64)
                    refkey = f"{k}[:,2:4]"
                    ref_is_rotated = True
                    break

    rotation_M = None
    rotation_key = None
    for k in ROTATION_KEYS:
        if k in z:
            M = np.asarray(z[k], dtype=np.float64)
            if M.shape == (2, 3):
                rotation_M = M
                rotation_key = k
                break

    ref_original = None
    rotation_source = "not_needed"
    if ref is not None:
        if ref_is_rotated:
            if rotation_M is None and allow_reconstruct_gt_yaw_rotation:
                _, info = prepare_ref_image_and_info(
                    sample["image_dop"],
                    sample["pose_query2world"],
                    rotate_ref_by_gt_yaw=True,
                )
                rotation_M = np.asarray(info["rotation_M"], dtype=np.float64)
                rotation_source = "reconstructed_from_gt_yaw_pose"
            elif rotation_M is not None:
                rotation_source = f"cache:{rotation_key}"
            if rotation_M is None:
                raise RuntimeError(
                    f"{p}: rotated reference points found but no usable rotation_M."
                )
            ref_original = inverse_affine_points(ref, rotation_M)
        else:
            ref_original = np.asarray(ref, dtype=np.float64).copy()
            rotation_source = "reference_already_original"

    use_cached_xyz = xyz is not None and len(xyz) == len(q)
    if use_cached_xyz:
        n = min(len(q), len(xyz))
        q = q[:n]
        xyz = xyz[:n]
        if ref_original is not None:
            ref_original = ref_original[:n]
        geometry_source = f"cached_world_xyz:{xyzkey}"
    else:
        if ref_original is None:
            raise RuntimeError(
                f"{p}: cached world_xyz absent/misaligned and no usable reference pixels."
            )
        n = min(len(q), len(ref_original))
        q = q[:n]
        ref_original = ref_original[:n]
        xyz, _ = index_dsm_world_xyz(sample["dsm"], ref_original)
        geometry_source = "reference_pixels_to_ortholoc_dsm_xyz"

    qfinite = np.isfinite(q).all(axis=1)
    q = q[qfinite]
    xyz = xyz[qfinite]
    if ref_original is not None and len(ref_original) == len(qfinite):
        ref_original = ref_original[qfinite]

    certainty = None
    for k in ("certainty", "confidence", "scores", "match_confidence"):
        if k in z:
            c = np.asarray(z[k], dtype=np.float64).reshape(-1)
            if len(c) == len(qfinite):
                certainty = c[qfinite]
            elif len(c) == len(q):
                certainty = c
            break

    match_time = 0.0
    for k in ("match_time_s", "matching_time_s", "match_time"):
        if k in z:
            a = np.asarray(z[k]).reshape(-1)
            if len(a):
                v = finite_float(a[0])
                if v is not None:
                    match_time = float(v)
                    break

    dsm_valid = np.isfinite(xyz).all(axis=1)

    ref_gt_yaw_deg = None
    if "ref_gt_yaw_deg" in z:
        try:
            _a = np.asarray(z["ref_gt_yaw_deg"]).reshape(-1)
            if len(_a):
                ref_gt_yaw_deg = float(_a[0])
        except Exception:
            ref_gt_yaw_deg = None

    return {
        "query_points_original": q,
        "ref_points_original": ref_original,
        "world_xyz": xyz,
        "dsm_valid_mask": dsm_valid,
        "certainty": certainty,
        "num_matches": int(len(q)),
        "match_time_s": float(match_time),
        "cache_path": str(p),
        "cache_keys": keys,
        "query_key": qkey,
        "reference_key": refkey,
        "world_xyz_key": xyzkey,
        "geometry_source": geometry_source,
        "rotation_source": rotation_source,
        "rotation_M": rotation_M,
        "ref_gt_yaw_deg": ref_gt_yaw_deg,
        "matching_method": cached_matching_method,
        "source": "cache",
    }


def save_match_geometry(
    path: str | Path,
    geom: Dict[str, Any],
    ref_info: Optional[Dict[str, Any]] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    q = np.asarray(geom["query_points_original"], dtype=np.float64).reshape(-1, 2)
    ref = geom.get("ref_points_original")
    ref = (
        np.asarray(ref, dtype=np.float64).reshape(-1, 2)
        if ref is not None
        else np.full((len(q), 2), np.nan, dtype=np.float64)
    )
    xyz = np.asarray(geom["world_xyz"], dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(geom.get("dsm_valid_mask", np.isfinite(xyz).all(axis=1))).astype(np.uint8)
    certainty = geom.get("certainty")
    if certainty is None:
        certainty = np.full((len(q),), np.nan, dtype=np.float64)
    else:
        certainty = np.asarray(certainty, dtype=np.float64).reshape(-1)

    info = ref_info or {}
    M = np.asarray(info.get("rotation_M", np.eye(2, 3)), dtype=np.float64).reshape(2, 3)
    payload = {
        "query_points_original": q,
        "uav_points_original": q,
        "ref_points_original": ref,
        "reference_points_original": ref,
        "world_xyz": xyz,
        "dsm_valid_mask": valid,
        "num_matches": np.asarray([len(q)], dtype=np.int64),
        "match_time_s": np.asarray([float(geom.get("match_time_s", 0.0))], dtype=np.float64),
        "certainty": certainty,
        "ref_gt_yaw_deg": np.asarray([float(info.get("yaw_deg", 0.0))], dtype=np.float64),
        "ref_was_rotated": np.asarray([int(bool(info.get("rotated", False)))], dtype=np.uint8),
        "rotation_M": M,
        "ref_orig_h": np.asarray([int(info.get("orig_h", 0))], dtype=np.int64),
        "ref_orig_w": np.asarray([int(info.get("orig_w", 0))], dtype=np.int64),
        "ref_rot_h": np.asarray([int(info.get("rot_h", info.get("orig_h", 0)))], dtype=np.int64),
        "ref_rot_w": np.asarray([int(info.get("rot_w", info.get("orig_w", 0)))], dtype=np.int64),
    }
    if geom.get("matching_method") is not None:
        payload["matching_method"] = np.asarray([str(geom["matching_method"])])
    np.savez_compressed(p, **payload)


def _cache_method_compatible(cached_method: Optional[Any], requested_method: str) -> bool:
    """Legacy caches without a matcher tag are treated as RoMa caches only."""
    requested = str(requested_method).strip().lower()
    if cached_method is None:
        return requested == "roma"
    return str(cached_method).strip().lower() == requested


class LazyMatcher:
    """Lazily initialize the matcher selected by ``opt.matching_method``."""

    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self.method_dict: Optional[Dict[str, Any]] = None
        self.backend: Optional[str] = None

    def _ensure(self) -> None:
        if self.method_dict is not None:
            return

        import yaml
        from avl_utils import matching_init

        yaml_path = Path(self.opt.yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Matcher config not found: {yaml_path}. "
                "It is only required when a cache miss needs live matching."
            )
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        config["DEVICE"] = str(self.opt.device)
        method_dict: Dict[str, Any] = {
            "matching_method": str(self.opt.matching_method)
        }
        self.method_dict = matching_init(method_dict, self.opt, config)
        self.backend = f"avl_utils:{self.opt.matching_method}"

    def match(
        self,
        sample: Dict[str, Any],
        sample_dir: Path,
        rotate_ref_by_gt_yaw: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure()
        from avl_utils import run_pixel_match_anyvisloc

        query = np.asarray(sample["image_query"])
        ref_for_match, ref_info = prepare_ref_image_and_info(
            sample["image_dop"],
            sample["pose_query2world"],
            rotate_ref_by_gt_yaw=rotate_ref_by_gt_yaw,
        )

        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        q_pts, ref_pts_rot = run_pixel_match_anyvisloc(
            query,
            ref_for_match,
            self.method_dict,
            str(sample_dir),
            "/match_raw.jpg",
            need_ransac=False,
            show_matches=bool(getattr(self.opt, "show_match_image", False)),
        )
        elapsed = time.time() - t0

        q_pts = np.asarray(q_pts, dtype=np.float64).reshape(-1, 2)
        ref_pts_rot = np.asarray(ref_pts_rot, dtype=np.float64).reshape(-1, 2)
        n = min(len(q_pts), len(ref_pts_rot))
        q_pts = q_pts[:n]
        ref_pts_rot = ref_pts_rot[:n]
        valid = np.isfinite(q_pts).all(axis=1) & np.isfinite(ref_pts_rot).all(axis=1)
        q_pts = q_pts[valid]
        ref_pts_rot = ref_pts_rot[valid]

        ref_pts_orig = inverse_affine_points(ref_pts_rot, ref_info["rotation_M"])
        xyz, dsm_valid = index_dsm_world_xyz(sample["dsm"], ref_pts_orig)

        geom = {
            "query_points_original": q_pts,
            "ref_points_original": ref_pts_orig,
            "world_xyz": xyz,
            "dsm_valid_mask": dsm_valid,
            "certainty": None,
            "num_matches": int(len(q_pts)),
            "match_time_s": float(elapsed),
            "matching_method": str(self.opt.matching_method),
            "source": "live_match",
            "matcher_backend": self.backend,
            "geometry_source": "live_ref_pixels_to_ortholoc_dsm_xyz",
            "rotation_source": "live_gt_yaw" if rotate_ref_by_gt_yaw else "not_rotated",
        }
        return geom, ref_info

def resolve_match_geometry(
    sample: Dict[str, Any],
    split: str,
    sample_dir: Path,
    opt: argparse.Namespace,
    cache_index: Optional[MatchCacheIndex],
    lazy_matcher: Optional[LazyMatcher],
) -> Dict[str, Any]:
    sid = str(sample["sample_id"])

    if not bool(getattr(opt, "force_rematch", False)):
        cp = find_match_cache(cache_index, split, sid)
        if cp is not None:
            geom = load_match_geometry(
                cp,
                sample,
                allow_reconstruct_gt_yaw_rotation=bool(
                    getattr(opt, "allow_reconstruct_gt_yaw_rotation", True)
                ),
            )
            if _cache_method_compatible(geom.get("matching_method"), opt.matching_method):
                # Old untagged caches are accepted only for RoMa; tag copied caches.
                if geom.get("matching_method") is None:
                    geom["matching_method"] = "Roma"
                if bool(getattr(opt, "copy_match_cache_to_run", False)):
                    canonical = Path(sample_dir) / "match_geometry.npz"
                    save_match_geometry(canonical, geom)
                return geom

    if bool(getattr(opt, "cache_only", False)):
        raise FileNotFoundError(
            f"No compatible match cache found for {split}/{sid} "
            f"for matching_method={opt.matching_method!r} and --cache_only is enabled."
        )

    if lazy_matcher is None:
        lazy_matcher = LazyMatcher(opt)
    geom, ref_info = lazy_matcher.match(
        sample,
        sample_dir,
        rotate_ref_by_gt_yaw=bool(getattr(opt, "rotate_ref_by_gt_yaw", True)),
    )
    canonical = Path(sample_dir) / "match_geometry.npz"
    save_match_geometry(canonical, geom, ref_info=ref_info)
    geom["cache_path"] = str(canonical)
    return geom


def build_common(
    sample: Dict[str, Any],
    split: str,
    geometry: Dict[str, Any],
    sample_dir: str | Path,
) -> SimpleNamespace:
    q = np.asarray(geometry["query_points_original"], dtype=np.float64).reshape(-1, 2)
    xyz = np.asarray(geometry["world_xyz"], dtype=np.float64).reshape(-1, 3)
    n = min(len(q), len(xyz))
    q = q[:n]
    xyz = xyz[:n]

    ref = geometry.get("ref_points_original")
    if ref is not None:
        ref = np.asarray(ref, dtype=np.float64).reshape(-1, 2)[:n]

    K = np.asarray(sample["intrinsics_query"], dtype=np.float64).reshape(3, 3)
    pose = np.asarray(sample["pose_query2world"], dtype=np.float64)
    gt = camera_center_from_c2w(pose)

    c = SimpleNamespace()
    c.scene_name = split
    c.split_name = split
    c.sample_id = str(sample["sample_id"])
    c.npz_path = str(sample["npz_path"])
    c.sample_dir = Path(sample_dir)
    c.reference_mode = "aerial"

    c.uav_points_original = q
    c.query_points_original = q
    c.image_points_original = q
    c.uav_points_match = q
    c.uav_points_undistorted = q.copy()
    c.query_points_undistorted = q.copy()
    c.image_points = q.copy()

    if ref is not None:
        c.ref_points_original = ref
        c.reference_points_original = ref
        c.map_points_original = ref
        c.ref_points = ref
        c.reference_points = ref

    c.world_xyz = xyz
    c.map_xyz = xyz
    c.world_x = xyz[:, 0]
    c.world_y = xyz[:, 1]
    c.world_z = xyz[:, 2]
    c.dsm_z = xyz[:, 2]
    c.dsm_valid_mask = np.isfinite(xyz).all(axis=1)
    c.map_valid_mask = c.dsm_valid_mask

    # Physical OrthoLoC convention.  Dataset-specific AnyVisLoc reflection is
    # applied only inside ortholoc_adapters when a clean AnyVisLoc solver needs it.
    c.local_x = c.world_x
    c.local_y = c.world_y

    c.K = K
    c.intrinsics_query = K
    c.camera_matrix = K
    c.K_undistorted = K.copy()
    c.new_K = K.copy()
    c.dist = np.zeros(5, dtype=np.float64)
    c.distortion = np.zeros(5, dtype=np.float64)
    c.dist_coeffs = np.zeros(5, dtype=np.float64)
    c.use_dataset_distortion = False
    c.uav_was_undistorted = False

    c.gt_camera_center = gt
    c.gt_center = gt
    c.gt_xyz = gt
    c.true_pos = {"x": float(gt[0]), "y": float(gt[1]), "z": float(gt[2])}
    c.true_position = c.true_pos
    c.pose_query2world = pose
    c.gt_pose_c2w = pose

    c.dsm = np.asarray(sample["dsm"], dtype=np.float64)
    c.dsm_grid = c.dsm
    c.mask_dsm = np.asarray(sample["mask_dsm"]).astype(bool)

    c.image = np.asarray(sample["image_query"])
    c.image_query = c.image
    c.query_image = c.image
    c.image_uav = c.image
    c.uav_image = c.image
    c.image_shape = c.image.shape[:2]
    c.query_shape = c.image.shape[:2]

    c.reference_image = np.asarray(sample["image_dop"])
    c.image_reference = c.reference_image
    c.image_dop = c.reference_image
    c.ref_image = c.reference_image
    c.reference_shape = c.reference_image.shape[:2]

    c.ref_gt_yaw_deg = geometry.get("ref_gt_yaw_deg")
    c.ref_rotation_M = geometry.get("rotation_M")
    c.ref_rotation_info = {
        "yaw_deg": geometry.get("ref_gt_yaw_deg"),
        "rotation_M": geometry.get("rotation_M"),
        "source": geometry.get("rotation_source"),
    }

    dsm_z = np.asarray(c.dsm[..., 2], dtype=np.float32)
    dsm_xyz = np.asarray(c.dsm, dtype=np.float32)
    dsm_mask = np.asarray(c.mask_dsm).astype(bool) & np.isfinite(c.dsm).all(axis=2)
    c.view = {
        "dsm": dsm_z,
        "dsm_z": dsm_z,
        "height": dsm_z,
        "height_map": dsm_z,
        "dsm_xyz": dsm_xyz,
        "xyz": dsm_xyz,
        "world_xyz": dsm_xyz,
        "mask_dsm": dsm_mask,
        "dsm_mask": dsm_mask,
        "valid_mask": dsm_mask,
        "image": c.reference_image,
        "image_dop": c.reference_image,
        "reference_image": c.reference_image,
        "height_px": int(dsm_z.shape[0]),
        "width_px": int(dsm_z.shape[1]),
        "height_pixels": int(dsm_z.shape[0]),
        "width_pixels": int(dsm_z.shape[1]),
        "shape": tuple(dsm_z.shape),
    }
    if ref is not None:
        c.view.update(
            {
                "ref_points": ref,
                "ref_points_original": ref,
                "reference_points": ref,
                "reference_points_original": ref,
            }
        )
    c.reference_view = c.view
    c.aerial_view = c.view
    c.dsm_view = c.view
    c.dsm_height = dsm_z
    c.dsm_z_grid = dsm_z
    c.dsm_xyz_grid = dsm_xyz

    c.match_time_s = float(geometry.get("match_time_s", 0.0))
    c.match_confidence = geometry.get("certainty")
    c.num_matches = int(n)
    c.match_cache_path = geometry.get("cache_path")
    c.match_geometry_source = geometry.get("geometry_source")
    c.match_source = geometry.get("source")

    # Identity image geometry for OrthoLoC.
    c.resize_ratio = 1.0
    c.scale = 1.0
    c.query_scale = 1.0
    c.ref_scale = 1.0
    c.crop_offset = np.zeros(2, dtype=np.float64)
    c.query_crop_offset = np.zeros(2, dtype=np.float64)
    c.reference_crop_offset = np.zeros(2, dtype=np.float64)
    c.fine_scale = 1.0
    c.reference_resize = 1.0
    c.uav_scale = 1.0
    c.crop_info = {
        "pose_priori": "gt_yaw_rotated_dop"
        if geometry.get("rotation_M") is not None
        else "none"
    }
    return c
