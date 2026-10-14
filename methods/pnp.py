from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import cv2
import numpy as np

@dataclass
class PnPResult:
    status: str
    center_xyz_local: Optional[np.ndarray]
    inliers: np.ndarray
    reprojection_rmse_px: Optional[float]

def simple_pnp_ransac_optimized(
    pose_3d: np.ndarray,
    image_pts: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    ransac_iterations: int = 1000,
    reprojection_threshold: float = 2.0,
    min_inliers: int = 4,
    confidence: float = 0.99,
    random_seed: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, float]:
    """Retain the supplied four-point P3P sampling and full-set inlier scoring."""
    pose_3d = np.asarray(pose_3d, dtype=np.float32).reshape(-1, 3)
    image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
    valid = np.isfinite(pose_3d).all(axis=1) & np.isfinite(image_pts).all(axis=1)
    pose_3d = pose_3d[valid]
    image_pts = image_pts[valid]
    n_points = len(pose_3d)
    if n_points < 4:
        return None, None, np.zeros((0,), dtype=np.int32), float("inf")

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)

    K = np.asarray(K, dtype=np.float32)
    threshold_squared = float(reprojection_threshold) ** 2

    if int(ransac_iterations) <= 0:
        expected_inlier_ratio = 0.5
        sample_size = 4
        denominator = math.log(1.0 - expected_inlier_ratio**sample_size)
        iterations = int(math.log(1.0 - float(confidence)) / denominator)
        iterations = min(max(iterations, 100), 10000)
    else:
        iterations = int(ransac_iterations)

    best_rvec = None
    best_tvec = None
    best_inliers = np.zeros((0,), dtype=np.int32)
    best_inlier_count = 0
    best_average_error = float("inf")

    all_indices = np.arange(n_points)
    rng = np.random.default_rng(int(random_seed))

    for _ in range(iterations):
        indices = all_indices if n_points == 4 else rng.choice(all_indices, 4, replace=False)
        try:
            success, rvec, tvec = cv2.solvePnP(
                pose_3d[indices],
                image_pts[indices],
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_P3P,
                useExtrinsicGuess=False,
            )
            if not success:
                continue

            projected, _ = cv2.projectPoints(pose_3d, rvec, tvec, K, dist_coeffs)
            projected = projected.reshape(-1, 2)
            errors_squared = np.sum((projected - image_pts) ** 2, axis=1)
            inlier_mask = errors_squared < threshold_squared
            inlier_count = int(np.sum(inlier_mask))
            if inlier_count < int(min_inliers):
                continue

            average_error = float(np.sqrt(np.mean(errors_squared[inlier_mask])))
            if (
                inlier_count > best_inlier_count
                or (inlier_count == best_inlier_count and average_error < best_average_error)
            ):
                best_inlier_count = inlier_count
                best_average_error = average_error
                best_inliers = np.where(inlier_mask)[0].astype(np.int32)
                best_rvec = rvec
                best_tvec = tvec
                if inlier_count > n_points * 0.8:
                    break
        except cv2.error:
            continue

    return best_rvec, best_tvec, best_inliers, best_average_error

def _normalize_distortion(dist_coeffs):
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if len(d) == 0:
        return np.zeros((5, 1), dtype=np.float64)
    return np.where(np.isfinite(d), d, 0.0)

def _ippe_solutions(object_points, image_points, K, dist_coeffs):
    try:
        ret = cv2.solvePnPGeneric(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            np.asarray(K, dtype=np.float64),
            _normalize_distortion(dist_coeffs),
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return []
    if not ret or int(ret[0]) <= 0:
        return []
    return list(zip(ret[1], ret[2]))

def _score_pose(object_points, image_points, K, dist_coeffs, rvec, tvec,
                reprojection_threshold):
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec, tvec,
        np.asarray(K, dtype=np.float64),
        _normalize_distortion(dist_coeffs),
    )
    projected = projected.reshape(-1, 2)
    errors_squared = np.sum(
        (projected - np.asarray(image_points, dtype=np.float64)) ** 2, axis=1
    )
    inliers = np.where(
        errors_squared < float(reprojection_threshold) ** 2
    )[0].astype(np.int32)
    rmse = (
        float(np.sqrt(np.mean(errors_squared[inliers])))
        if len(inliers) else float("inf")
    )
    return inliers, rmse

def ippe_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    ransac_iterations: int = 1000,
    reprojection_threshold: float = 2.0,
    min_inliers: int = 4,
    confidence: float = 0.99,
    random_seed: int = 0,
):
    """Use the supplied IPPE hypothesis generator with the same RANSAC policy."""
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    object_points = object_points[valid]
    image_points = image_points[valid]
    n_points = len(object_points)
    if n_points < 4:
        return None, None, np.zeros((0,), dtype=np.int32), float("inf")

    if int(ransac_iterations) <= 0:
        expected_inlier_ratio = 0.5
        sample_size = 4
        denominator = math.log(1.0 - expected_inlier_ratio**sample_size)
        iterations = int(math.log(1.0 - float(confidence)) / denominator)
        iterations = min(max(iterations, 100), 10000)
    else:
        iterations = int(ransac_iterations)

    rng = np.random.default_rng(int(random_seed))
    all_indices = np.arange(n_points, dtype=np.int32)
    best_rvec = None
    best_tvec = None
    best_inliers = np.zeros((0,), dtype=np.int32)
    best_rmse = float("inf")
    stop_early = False

    for _ in range(iterations):
        sample = (
            all_indices
            if n_points == 4
            else rng.choice(all_indices, 4, replace=False)
        )
        solutions = _ippe_solutions(
            object_points[sample], image_points[sample], K, dist_coeffs
        )
        for rvec, tvec in solutions:
            try:
                inliers, rmse = _score_pose(
                    object_points, image_points, K, dist_coeffs,
                    rvec, tvec, reprojection_threshold,
                )
            except cv2.error:
                continue
            if len(inliers) < int(min_inliers):
                continue
            if (
                len(inliers) > len(best_inliers)
                or (len(inliers) == len(best_inliers) and rmse < best_rmse)
            ):
                best_rvec = rvec
                best_tvec = tvec
                best_inliers = inliers
                best_rmse = rmse
                if len(inliers) > n_points * 0.8:
                    stop_early = True
        if stop_early:
            break

    if best_rvec is None or len(best_inliers) < int(min_inliers):
        return None, None, best_inliers, best_rmse

    if len(best_inliers) >= 4:
        refit_solutions = _ippe_solutions(
            object_points[best_inliers], image_points[best_inliers], K, dist_coeffs
        )
        for rvec, tvec in refit_solutions:
            try:
                inliers, rmse = _score_pose(
                    object_points, image_points, K, dist_coeffs,
                    rvec, tvec, reprojection_threshold,
                )
            except cv2.error:
                continue
            if (
                len(inliers) > len(best_inliers)
                or (len(inliers) == len(best_inliers) and rmse < best_rmse)
            ):
                best_rvec = rvec
                best_tvec = tvec
                best_inliers = inliers
                best_rmse = rmse

    return best_rvec, best_tvec, best_inliers, best_rmse

def _result_from_pose(rvec, tvec, inliers, score):
    if rvec is None or tvec is None:
        return PnPResult("pnp_failed", None, inliers, None)
    R, _ = cv2.Rodrigues(rvec)
    center = (-R.T @ tvec).reshape(3).astype(np.float64)
    return PnPResult(
        "success", center, inliers,
        None if not np.isfinite(score) else float(score),
    )

def run_dsm_pnp(image_xy, map_xy, dsm_z, K, dist=None, **kw):
    xy = np.asarray(map_xy, np.float64).reshape(-1, 2)
    z = np.asarray(dsm_z, np.float64).reshape(-1)
    img = np.asarray(image_xy, np.float64).reshape(-1, 2)
    n = min(len(xy), len(z), len(img))
    rvec, tvec, inliers, score = simple_pnp_ransac_optimized(
        np.c_[xy[:n], z[:n]], img[:n], K, dist_coeffs=dist,
        ransac_iterations=kw.get("iterations", kw.get("ransac_iterations", 1000)),
        reprojection_threshold=kw.get("reprojection_threshold", 2.0),
        min_inliers=kw.get("min_inliers", 4),
        confidence=kw.get("confidence", 0.99),
        random_seed=kw.get("seed", kw.get("random_seed", 0)),
    )
    return _result_from_pose(rvec, tvec, inliers, score)

def run_planar_pnp(image_xy, map_xy, K, dist=None, **kw):
    xy = np.asarray(map_xy, np.float64).reshape(-1, 2)
    img = np.asarray(image_xy, np.float64).reshape(-1, 2)
    n = min(len(xy), len(img))
    rvec, tvec, inliers, score = ippe_ransac(
        np.c_[xy[:n], np.zeros(n)], img[:n], K, dist_coeffs=dist,
        ransac_iterations=kw.get("iterations", kw.get("ransac_iterations", 1000)),
        reprojection_threshold=kw.get("reprojection_threshold", 2.0),
        min_inliers=kw.get("min_inliers", 4),
        confidence=kw.get("confidence", 0.99),
        random_seed=kw.get("seed", kw.get("random_seed", 0)),
    )
    return _result_from_pose(rvec, tvec, inliers, score)
