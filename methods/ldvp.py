#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

LDVP_VERSION = 2
LDVP_TAG = "shared-globalrot-msac-fast-v1"
MSAC_VERSION = 1
MSAC_TAG = "simple-five-point-msac-v1"
_NUMERIC_REL_TOL = math.sqrt(np.finfo(np.float64).eps)

# Original final MSAC value recovered during reproducibility audit.
_SHARED_PLANE_VZ_EPS = 1e-10


def _wrap_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

def angles_to_world_to_camera(
    down_tilt_rad: float,
    azimuth_rad: float,
    roll_rad: float,
) -> np.ndarray:
    """
    Rotation parameterization.

    Camera +Z is the optical axis.
    down_tilt = 0 deg  : optical axis horizontal
    down_tilt = 90 deg : optical axis straight down

    Returns R_world_to_camera.
    """
    beta = float(np.clip(down_tilt_rad, 0.0, 0.5 * math.pi))
    psi = float(azimuth_rad)
    rho = float(roll_rad)

    cb, sb = math.cos(beta), math.sin(beta)
    cp, sp = math.cos(psi), math.sin(psi)
    cr, sr = math.cos(rho), math.sin(rho)

    # Camera optical axis expressed in world coordinates.
    z_w = np.array([cb * cp, cb * sp, -sb], dtype=np.float64)

    # Zero-roll transverse axis.
    x0_w = np.array([-sp, cp, 0.0], dtype=np.float64)
    y0_w = np.cross(z_w, x0_w)
    y0_w /= max(np.linalg.norm(y0_w), 1e-15)

    # Roll around optical axis.
    x_w = cr * x0_w + sr * y0_w
    y_w = -sr * x0_w + cr * y0_w

    R_camera_to_world = np.column_stack([x_w, y_w, z_w])
    return R_camera_to_world.T

def _soft_l1_weights(residual: np.ndarray, f_scale: float) -> np.ndarray:
    """
    IRLS weights for scipy-style soft_l1:
        rho(z) = 2 * (sqrt(1+z) - 1), z=(r/f)^2

    The stationarity-equivalent weighted LS weight is
        w = 1 / sqrt(1 + (r/f)^2).
    """
    r = np.asarray(residual, dtype=np.float64)
    f = max(float(f_scale), 1e-12)
    return 1.0 / np.sqrt(1.0 + (r / f) ** 2)

def _soft_l1_score_m(residual: np.ndarray, f_scale: float) -> float:
    """
    Robust score with meter-like units. Ranking is exactly the same as
    mean soft-L1 cost; sqrt is only for readable diagnostics.
    """
    r = np.asarray(residual, dtype=np.float64)
    f = max(float(f_scale), 1e-12)
    cost_m2 = 2.0 * f * f * (np.sqrt(1.0 + (r / f) ** 2) - 1.0)
    return float(np.sqrt(np.mean(cost_m2)))

def _solve_weighted_center(
    normals: np.ndarray,
    map_xy: np.ndarray,
    weights: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve min_c sum_i w_i [n_i^T(m_i-c)]^2.

    Coordinate centering is exact and only improves numerical conditioning.
    Returns (center, singular_values_of_weighted_design).
    """
    N = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    valid = (
        np.isfinite(N).all(axis=1)
        & np.isfinite(m).all(axis=1)
        & np.isfinite(w)
        & (w > 0.0)
    )
    if int(np.sum(valid)) < 4:
        return None, None

    N = N[valid]
    m = m[valid]
    w = w[valid]

    # Weighted centroid reduces large-coordinate cancellation.
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return None, None
    m0 = np.sum(w[:, None] * m, axis=0) / sw
    mc = m - m0[None, :]

    sqrtw = np.sqrt(w)
    A = sqrtw[:, None] * N
    b = sqrtw * np.sum(N * mc, axis=1)

    d, _, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    if int(rank) < 2:
        return None, np.asarray(singular_values, dtype=np.float64)

    s = np.asarray(singular_values, dtype=np.float64)
    if len(s) < 2 or s[-1] <= _NUMERIC_REL_TOL * max(s[0], 1e-15):
        return None, s

    return m0 + d, s

def _profile_angles(
    angles: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    f_scale: float = 1.0,
    irls_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Robust variable projection for one rotation.

    For fixed R:
        q_i = (R^T r_i)_xy
        n_i = [-q_iy, q_ix] / ||q_i||

    Instead of profiling C_xy with ordinary L2 while optimizing a soft-L1
    outer objective, profile C_xy with the SAME soft-L1 M-estimator:

        c*(R) = argmin_c sum_i rho_softL1(n_i^T(m_i-c)).

    The 2-D inner problem is solved by a few deterministic IRLS iterations.
    lambda_i remains analytic once c is known.
    """
    beta, psi, rho = np.asarray(angles, dtype=np.float64).reshape(3)
    R = angles_to_world_to_camera(beta, psi, rho)

    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    npts = min(len(rays), len(m))
    rays, m = rays[:npts], m[:npts]

    v = rays @ R
    q = v[:, :2]
    q2 = np.sum(q * q, axis=1)
    qnorm = np.sqrt(q2)

    valid = (
        np.isfinite(qnorm)
        & (qnorm > 1e-12)
        & np.isfinite(m).all(axis=1)
    )
    if int(np.sum(valid)) < 4:
        return {"valid": False, "reason": "insufficient_valid_horizontal_rays"}

    qv = q[valid]
    q2v = q2[valid]
    mv = m[valid]
    normals = np.column_stack(
        [-qv[:, 1] / np.sqrt(q2v), qv[:, 0] / np.sqrt(q2v)]
    )

    # Initial L2 center.
    weights = np.ones(len(mv), dtype=np.float64)
    c, singular_values = _solve_weighted_center(normals, mv, weights)
    if c is None:
        return {"valid": False, "reason": "center_rank_deficient"}

    # Robustly profile center under the SAME soft-L1 loss used outside.
    # This is a 2-D convex M-estimation subproblem; IRLS is only the numerical
    # inner solver and does not add pose hypotheses or nonlinear pose variables.
    irls_used = 0
    for _ in range(max(0, int(irls_iterations))):
        signed = np.sum(normals * (mv - c[None, :]), axis=1)
        weights = _soft_l1_weights(signed, f_scale)
        c_new, s_new = _solve_weighted_center(normals, mv, weights)
        if c_new is None:
            break
        delta_c = float(np.linalg.norm(c_new - c))
        c = c_new
        singular_values = s_new
        irls_used += 1
        if delta_c < 1e-7:
            break

    delta = mv - c[None, :]
    signed = np.sum(normals * delta, axis=1)
    lam = np.sum(qv * delta, axis=1) / np.maximum(q2v, 1e-15)

    positive_ratio = float(np.mean(lam > 0.0))
    rms = float(np.sqrt(np.mean(signed * signed)))
    robust_score = _soft_l1_score_m(signed, f_scale)

    full_signed = np.zeros(npts, dtype=np.float64)
    full_lambda = np.full(npts, np.nan, dtype=np.float64)
    full_valid = np.zeros(npts, dtype=bool)
    full_signed[valid] = signed
    full_lambda[valid] = lam
    full_valid[valid] = True

    s = np.asarray(singular_values, dtype=np.float64)
    cond = float(s[0] / max(s[-1], 1e-15)) if len(s) >= 2 else float("inf")

    return {
        "valid": True,
        "angles": np.array([beta, psi, rho], dtype=np.float64),
        "R": R,
        "center_xy": c,
        "lambda": full_lambda,
        "signed_residual": full_signed,
        "valid_mask": full_valid,
        "rms_line_residual_m": rms,
        "robust_score_m": robust_score,
        "positive_depth_ratio": positive_ratio,
        # Condition number of the weighted design A = sqrt(W) N, not of A^T A.
        "center_condition": cond,  # compatibility alias
        "center_design_condition": cond,
        "center_profile": "soft_l1_robust_profile",
        "center_profile_irls_iterations": int(irls_used),
        "center_profile_irls_max_iterations": int(irls_iterations),
    }

def _rotation_grid(
    tilt_step_deg: float,
    azimuth_step_deg: float,
    roll_step_deg: float,
) -> np.ndarray:
    """Structured coarse angle grid over the admissible UAV rotation domain."""
    dt = max(float(tilt_step_deg), 1e-6)
    da = max(float(azimuth_step_deg), 1e-6)
    dr = max(float(roll_step_deg), 1e-6)

    tilts = np.deg2rad(np.arange(0.0, 90.0 + 0.5 * dt, dt))
    tilts = tilts[tilts <= 0.5 * math.pi + 1e-12]
    if len(tilts) == 0 or abs(tilts[-1] - 0.5 * math.pi) > 1e-12:
        tilts = np.append(tilts, 0.5 * math.pi)

    azimuths = np.deg2rad(np.arange(-180.0, 180.0, da))
    rolls = np.deg2rad(np.arange(-180.0, 180.0, dr))

    return np.asarray(
        [(b, a, r) for b in tilts for a in azimuths for r in rolls],
        dtype=np.float64,
    )


@lru_cache(maxsize=32)
def _cached_coarse_grid_and_rotations(
    tilt_step_deg: float,
    azimuth_step_deg: float,
    roll_step_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cache the exact coarse angle grid and exact 3x3 rotations.

    Acceleration only. The same angles_to_world_to_camera() function is used,
    so the cached matrices are numerically identical to rebuilding them for
    every five-match MSAC hypothesis.
    """
    grid = _rotation_grid(
        float(tilt_step_deg),
        float(azimuth_step_deg),
        float(roll_step_deg),
    )
    rotations = np.stack(
        [angles_to_world_to_camera(*x) for x in grid],
        axis=0,
    ).astype(np.float64, copy=False)
    grid.setflags(write=False)
    rotations.setflags(write=False)
    return grid, rotations


def _coarse_grid_for_options(
    opt: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Positive globalrot_fast_precompute_max_mb enables exact rotation caching.
    Zero disables it and reproduces the original slow reconstruction path.
    """
    enabled = (
        float(getattr(opt, "globalrot_fast_precompute_max_mb", 192.0)) > 0.0
    )
    if enabled:
        return _cached_coarse_grid_and_rotations(
            float(opt.globalrot_tilt_step_deg),
            float(opt.globalrot_azimuth_step_deg),
            float(opt.globalrot_roll_step_deg),
        )
    return (
        _rotation_grid(
            float(opt.globalrot_tilt_step_deg),
            float(opt.globalrot_azimuth_step_deg),
            float(opt.globalrot_roll_step_deg),
        ),
        None,
    )

def _batch_grid_scores(
    angle_grid: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    batch_size: int,
    f_scale: float = 1.0,
    irls_iterations: int = 2,
    min_positive_ratio: float = 0.60,
    rotation_matrices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized robust profile for global rotation candidates.

    For every R, C_xy is profiled with a few soft-L1 IRLS steps and candidates
    are ranked by the SAME soft-L1 objective later used by R-only refinement.

    No beam, no top-K, no trimming, no pose multi-start.
    """
    angles = np.asarray(angle_grid, dtype=np.float64).reshape(-1, 3)
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)

    scores = np.full(len(angles), np.inf, dtype=np.float64)
    pos_ratio = np.zeros(len(angles), dtype=np.float64)
    physical = np.zeros(len(angles), dtype=bool)

    f = max(float(f_scale), 1e-12)
    bs = max(1, int(batch_size))

    # Fixed centering for the batched solve. Translation of coordinates does
    # not change the line residual objective.
    m0 = np.mean(m, axis=0)
    mc = m - m0[None, :]

    for start in range(0, len(angles), bs):
        aa = angles[start:start + bs]
        if rotation_matrices is None:
            Rs = np.stack(
                [angles_to_world_to_camera(*x) for x in aa],
                axis=0,
            )
        else:
            Rs = np.asarray(
                rotation_matrices[start:start + len(aa)],
                dtype=np.float64,
            )

        v = np.einsum("ni,bij->bnj", rays, Rs, optimize=True)
        q = v[:, :, :2]
        q2 = np.sum(q * q, axis=2)
        qnorm = np.sqrt(q2)
        valid_q = np.isfinite(qnorm) & (qnorm > 1e-12)

        nx = np.where(valid_q, -q[:, :, 1] / np.maximum(qnorm, 1e-15), 0.0)
        ny = np.where(valid_q,  q[:, :, 0] / np.maximum(qnorm, 1e-15), 0.0)
        b = nx * mc[None, :, 0] + ny * mc[None, :, 1]

        weights = valid_q.astype(np.float64)

        def solve_batch(w):
            a00 = np.sum(w * nx * nx, axis=1)
            a01 = np.sum(w * nx * ny, axis=1)
            a11 = np.sum(w * ny * ny, axis=1)
            r0 = np.sum(w * nx * b, axis=1)
            r1 = np.sum(w * ny * b, axis=1)

            det = a00 * a11 - a01 * a01
            tr = a00 + a11
            disc = np.sqrt(np.maximum((a00 - a11) ** 2 + 4.0 * a01 * a01, 0.0))
            eig_max = 0.5 * (tr + disc)
            eig_min = 0.5 * (tr - disc)

            ok = (
                np.isfinite(det)
                & (det > 0.0)
                & (eig_min > _NUMERIC_REL_TOL * np.maximum(eig_max, 1e-15))
                & (np.sum(valid_q, axis=1) >= 4)
            )

            dx = np.full(len(aa), np.nan, dtype=np.float64)
            dy = np.full(len(aa), np.nan, dtype=np.float64)
            dx[ok] = (a11[ok] * r0[ok] - a01[ok] * r1[ok]) / det[ok]
            dy[ok] = (-a01[ok] * r0[ok] + a00[ok] * r1[ok]) / det[ok]
            return dx, dy, ok

        dx, dy, ok = solve_batch(weights)

        for _ in range(max(0, int(irls_iterations))):
            deltax = mc[None, :, 0] - dx[:, None]
            deltay = mc[None, :, 1] - dy[:, None]
            signed = nx * deltax + ny * deltay
            robust_w = 1.0 / np.sqrt(1.0 + (signed / f) ** 2)
            weights = valid_q.astype(np.float64) * robust_w
            dx, dy, ok = solve_batch(weights)

        deltax = mc[None, :, 0] - dx[:, None]
        deltay = mc[None, :, 1] - dy[:, None]
        signed = nx * deltax + ny * deltay

        cost_m2 = 2.0 * f * f * (
            np.sqrt(1.0 + (signed / f) ** 2) - 1.0
        )
        count = np.maximum(np.sum(valid_q, axis=1), 1)
        mean_cost = (
            np.sum(np.where(valid_q, cost_m2, 0.0), axis=1) / count
        )
        robust_score = np.sqrt(np.maximum(mean_cost, 0.0))

        lam = np.where(
            valid_q,
            (q[:, :, 0] * deltax + q[:, :, 1] * deltay)
            / np.maximum(q2, 1e-15),
            np.nan,
        )
        pr = np.sum((lam > 0.0) & valid_q, axis=1) / count
        phys = pr >= float(min_positive_ratio)

        sl = slice(start, start + len(aa))
        scores[sl] = np.where(
            ok & phys & np.isfinite(robust_score),
            robust_score,
            np.inf,
        )
        pos_ratio[sl] = np.where(np.isfinite(pr), pr, 0.0)
        physical[sl] = ok & phys

    return scores, pos_ratio, physical

def _deduplicate_angles(values: np.ndarray) -> np.ndarray:
    """Clamp tilt, wrap periodic angles, and remove duplicates."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    out = []
    seen = set()
    for beta, psi, rho in vals:
        beta = float(np.clip(beta, 0.0, 0.5 * math.pi))
        psi = _wrap_pi(float(psi))
        rho = _wrap_pi(float(rho))
        key = (round(beta, 12), round(psi, 12), round(rho, 12))
        if key not in seen:
            seen.add(key)
            out.append([beta, psi, rho])
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)

def _local_rotation_grid(center: np.ndarray, step_rad: np.ndarray) -> np.ndarray:
    """
    3x3x3 neighborhood around ONE current best rotation.
    No beam and no multi-start.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(step_rad, dtype=np.float64).reshape(3)
    vals = []
    for db in (-s[0], 0.0, s[0]):
        for da in (-s[1], 0.0, s[1]):
            for dr in (-s[2], 0.0, s[2]):
                vals.append([c[0] + db, c[1] + da, c[2] + dr])
    return _deduplicate_angles(np.asarray(vals, dtype=np.float64))

def global_rotation_search(
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    opt: Any,
) -> Dict[str, Any]:
    """
    Minimal multi-resolution global rotation search.

    1) One coarse uniform grid over the full allowed rotation domain.
    2) Refine ONLY the single current best rotation with a 3x3x3 local grid.
       The angular step shrinks geometrically.

    No beam, no top-K, no multi-start, no robust re-ranking.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(m))
    rays, m = rays[:n], m[:n]

    if n < int(opt.depth_min_matches):
        return {"valid": False, "reason": "not_enough_matches"}

    max_points = max(0, int(opt.globalrot_max_points))
    if max_points > 0 and n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
        rays_search = rays[idx]
        m_search = m[idx]
    else:
        rays_search = rays
        m_search = m

    coarse_grid, coarse_Rs = _coarse_grid_for_options(opt)

    scores, pos_ratio, _ = _batch_grid_scores(
        coarse_grid,
        rays_search,
        m_search,
        int(opt.globalrot_batch_size),
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_grid_irls),
        min_positive_ratio=float(opt.globalrot_min_positive_ratio_init),
        rotation_matrices=coarse_Rs,
    )

    if not np.any(np.isfinite(scores)):
        return {"valid": False, "reason": "no_physical_grid_solution"}

    best_idx = int(np.nanargmin(scores))
    best_angles = coarse_grid[best_idx].copy()
    best_search_score = float(scores[best_idx])
    best_search_pos = float(pos_ratio[best_idx])

    total_evaluated = int(len(coarse_grid))

    base_step = np.deg2rad(
        [
            float(opt.globalrot_tilt_step_deg),
            float(opt.globalrot_azimuth_step_deg),
            float(opt.globalrot_roll_step_deg),
        ]
    )

    levels = max(0, int(opt.globalrot_refine_levels))
    factor = max(1.25, float(opt.globalrot_refine_factor))
    history = []

    for level in range(levels):
        step = base_step / (factor ** (level + 1))
        local_grid = _local_rotation_grid(best_angles, step)

        local_scores, local_pos, _ = _batch_grid_scores(
            local_grid,
            rays_search,
            m_search,
            int(opt.globalrot_batch_size),
            f_scale=float(opt.depth_robust_f_scale),
            irls_iterations=int(opt.globalrot_grid_irls),
            min_positive_ratio=float(opt.globalrot_min_positive_ratio_init),
        )
        total_evaluated += int(len(local_grid))

        if not np.any(np.isfinite(local_scores)):
            break

        j = int(np.nanargmin(local_scores))
        best_angles = local_grid[j].copy()
        best_search_score = float(local_scores[j])
        best_search_pos = float(local_pos[j])

        history.append(
            {
                "level": int(level + 1),
                "tilt_step_deg": float(np.rad2deg(step[0])),
                "azimuth_step_deg": float(np.rad2deg(step[1])),
                "roll_step_deg": float(np.rad2deg(step[2])),
                "score_m": float(best_search_score),
                "positive_depth_ratio": float(best_search_pos),
            }
        )

    best = _profile_angles(
        best_angles,
        rays,
        m,
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_profile_irls),
    )
    if not best.get("valid"):
        return {"valid": False, "reason": "best_grid_solution_failed_full_profile"}
    if float(best.get("positive_depth_ratio", 0.0)) < float(opt.globalrot_min_positive_ratio_init):
        return {"valid": False, "reason": "best_grid_solution_failed_cheirality"}

    return {
        "valid": True,
        "best": best,
        "num_coarse_grid_rotations": int(len(coarse_grid)),
        "num_grid_rotations_evaluated": int(total_evaluated),
        "num_search_points": int(len(rays_search)),
        "search_score_m": float(best_search_score),
        "search_positive_depth_ratio": float(best_search_pos),
        "refine_levels": int(levels),
        "refine_factor": float(factor),
        "level_history": history,
    }

def _r_only_residual(
    angles: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    f_scale: float,
    profile_irls: int,
) -> np.ndarray:
    """
    Variable-projection residual for local optimization.

    Nonlinear state: R only (3 parameters).
    C_xy is robustly profiled and lambda_i is analytically eliminated at every
    evaluation.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    out = _profile_angles(
        angles,
        rays,
        map_xy,
        f_scale=float(f_scale),
        irls_iterations=int(profile_irls),
    )

    if not out.get("valid"):
        return np.full(len(rays), 1e6, dtype=np.float64)

    residual = np.asarray(out["signed_residual"], dtype=np.float64).copy()
    valid = np.asarray(out["valid_mask"], dtype=bool)
    residual[~valid] = 0.0
    return residual

def refine_rotation_only(
    init_profile: Dict[str, Any],
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    opt: Any,
) -> Dict[str, Any]:
    """
    Local no-depth refinement with ONLY R as nonlinear state.

    The inner profile and the outer least-squares loss must use the same
    isotropic soft-L1 point-to-line objective.
    """
    loss_name = str(getattr(opt, "depth_robust_loss", "soft_l1")).strip().lower()
    if loss_name != "soft_l1":
        raise ValueError(
            "v3-clean requires --depth_robust_loss soft_l1 so the inner robust "
            "center profile and outer rotation objective remain identical."
        )

    x0 = np.asarray(init_profile["angles"], dtype=np.float64).reshape(3)

    lower = np.array([0.0, -np.inf, -np.inf], dtype=np.float64)
    upper = np.array([0.5 * math.pi, np.inf, np.inf], dtype=np.float64)

    ls = least_squares(
        _r_only_residual,
        x0,
        args=(
            rays_camera,
            map_xy,
            float(opt.depth_robust_f_scale),
            int(opt.globalrot_profile_irls),
        ),
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=float(opt.depth_robust_f_scale),
        max_nfev=max(20, int(opt.globalrot_local_max_nfev)),
        x_scale="jac",
    )

    prof = _profile_angles(
        ls.x,
        rays_camera,
        map_xy,
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_profile_irls),
    )
    if not prof.get("valid"):
        return {"valid": False, "reason": "r_only_refinement_profile_failed"}

    prof["meets_final_positive_depth_ratio"] = bool(
        float(prof.get("positive_depth_ratio", 0.0))
        >= float(opt.min_positive_depth_ratio_final)
    )
    prof["optimizer_cost"] = float(ls.cost)
    prof["optimizer_nfev"] = int(ls.nfev)
    prof["optimizer_message"] = str(ls.message)
    return prof

def _solve_weighted_shared_plane(
    projected_rays_xy: np.ndarray,
    map_xy: np.ndarray,
    weights: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Closed-form weighted nuisance solve for fixed R under the shared plane:

        min_{c,alpha} sum_i w_i ||m_i - c - alpha s_i||^2.

    Returns:
        center_xy,
        alpha,
        weighted spread information sum w_i ||s_i-sbar||^2.
    """
    s = np.asarray(projected_rays_xy, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    valid = (
        np.isfinite(s).all(axis=1)
        & np.isfinite(m).all(axis=1)
        & np.isfinite(w)
        & (w > 0.0)
    )
    if int(np.sum(valid)) < 4:
        return None, None, None

    s = s[valid]
    m = m[valid]
    w = w[valid]

    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return None, None, None

    mbar = np.sum(w[:, None] * m, axis=0) / sw
    sbar = np.sum(w[:, None] * s, axis=0) / sw
    dm = m - mbar[None, :]
    ds = s - sbar[None, :]

    denom = float(np.sum(w * np.sum(ds * ds, axis=1)))
    numeric_floor = _NUMERIC_REL_TOL * max(sw, 1.0)
    if not np.isfinite(denom) or denom <= numeric_floor:
        return None, None, denom

    numer = float(np.sum(w * np.sum(ds * dm, axis=1)))
    alpha = numer / denom
    c = mbar - alpha * sbar

    if not np.isfinite(c).all() or not np.isfinite(alpha):
        return None, None, denom
    return c, float(alpha), float(denom)

def _profile_shared_plane_angles(
    angles: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    f_scale: float = 1.0,
    irls_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Robust variable projection for the shared-horizontal-plane branch.

    Base model:
        m_i = c + lambda_i q_i(R)

    Additional structural constraint:
        lambda_i v_iz = alpha

    Therefore:
        m_i = c + alpha * (v_i,xy / v_iz).

    For fixed R, robustly profile (c_x, c_y, alpha) using isotropic soft-L1 on

        d_i = ||m_i - c - alpha s_i(R)||_2.

    The outer nonlinear state remains rotation only.
    """
    beta, psi, rho = np.asarray(angles, dtype=np.float64).reshape(3)
    R = angles_to_world_to_camera(beta, psi, rho)

    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    npts = min(len(rays), len(m))
    rays, m = rays[:npts], m[:npts]

    v = rays @ R
    q = v[:, :2]
    vz = v[:, 2]

    valid = (
        np.isfinite(v).all(axis=1)
        & np.isfinite(m).all(axis=1)
        & (np.abs(vz) > _SHARED_PLANE_VZ_EPS)
    )
    # Keep a fixed residual dimension and prevent a candidate from improving
    # its score by dropping near-horizon rays.
    if int(np.sum(valid)) != npts or npts < 4:
        return {"valid": False, "reason": "invalid_or_near_horizon_plane_rays"}

    s = q / vz[:, None]

    weights = np.ones(npts, dtype=np.float64)
    c, alpha, spread_info = _solve_weighted_shared_plane(s, m, weights)
    if c is None or alpha is None:
        return {"valid": False, "reason": "shared_plane_profile_rank_deficient"}

    irls_used = 0
    for _ in range(max(0, int(irls_iterations))):
        vec = m - c[None, :] - float(alpha) * s
        norms = np.linalg.norm(vec, axis=1)
        weights = _soft_l1_weights(norms, f_scale)
        c_new, alpha_new, spread_new = _solve_weighted_shared_plane(s, m, weights)
        if c_new is None or alpha_new is None:
            break
        dc = float(np.linalg.norm(c_new - c))
        da = abs(float(alpha_new) - float(alpha))
        c = c_new
        alpha = float(alpha_new)
        spread_info = spread_new
        irls_used += 1
        if max(dc, da) < 1e-7:
            break

    residual_2d = m - c[None, :] - float(alpha) * s
    residual_norm = np.linalg.norm(residual_2d, axis=1)

    lam = float(alpha) / vz
    positive_ratio = float(np.mean(lam > 0.0))
    rms = float(np.sqrt(np.mean(residual_norm * residual_norm)))
    robust_score = _soft_l1_score_m(residual_norm, f_scale)

    sw = max(float(np.sum(weights)), 1e-15)
    projected_spread = float(np.sqrt(max(float(spread_info or 0.0), 0.0) / sw))

    return {
        "valid": True,
        "angles": np.array([beta, psi, rho], dtype=np.float64),
        "R": R,
        "center_xy": np.asarray(c, dtype=np.float64),
        "lambda": np.asarray(lam, dtype=np.float64),
        "valid_mask": np.ones(npts, dtype=bool),
        "shared_plane_alpha_m": float(alpha),
        "shared_plane_projected_ray_spread": projected_spread,
        "plane_residual_2d": residual_2d,
        "plane_residual_norm": residual_norm,
        "rms_geometric_residual_m": rms,
        "rms_plane_residual_m": rms,
        "robust_score_m": robust_score,
        "positive_depth_ratio": positive_ratio,
        "nuisance_profile": "soft_l1_irls_closed_form_c_alpha",
        "nuisance_profile_irls_iterations": int(irls_used),
        "nuisance_profile_irls_max_iterations": int(irls_iterations),
        "structural_model": "shared_horizontal_plane",
        "structural_constraint": "lambda_i * (R^T r_i)_z = alpha",
    }

def _batch_shared_plane_scores(
    angle_grid: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    batch_size: int,
    f_scale: float = 1.0,
    irls_iterations: int = 2,
    min_positive_ratio: float = 0.60,
    rotation_matrices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized profiled scores for the shared-horizontal-plane model.

    Each IRLS step profiles c and alpha in closed form. Candidate ranking uses
    the same isotropic soft-L1 norm objective used by local R-only refinement.
    """
    angles = np.asarray(angle_grid, dtype=np.float64).reshape(-1, 3)
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)

    scores = np.full(len(angles), np.inf, dtype=np.float64)
    pos_ratio = np.zeros(len(angles), dtype=np.float64)
    physical = np.zeros(len(angles), dtype=bool)

    f = max(float(f_scale), 1e-12)
    bs = max(1, int(batch_size))
    m0 = np.mean(m, axis=0)
    mc = m - m0[None, :]

    for start in range(0, len(angles), bs):
        aa = angles[start:start + bs]
        B = len(aa)
        if rotation_matrices is None:
            Rs = np.stack(
                [angles_to_world_to_camera(*x) for x in aa],
                axis=0,
            )
        else:
            Rs = np.asarray(
                rotation_matrices[start:start + B],
                dtype=np.float64,
            )

        v = np.einsum("ni,bij->bnj", rays, Rs, optimize=True)
        q = v[:, :, :2]
        vz = v[:, :, 2]
        valid = (
            np.isfinite(v).all(axis=2)
            & (np.abs(vz) > _SHARED_PLANE_VZ_EPS)
        )
        all_valid = np.all(valid, axis=1)
        safe_vz = np.where(valid, vz, 1.0)
        s = q / safe_vz[:, :, None]
        s = np.where(valid[:, :, None], s, 0.0)

        weights = valid.astype(np.float64)

        def solve_batch(w):
            sw = np.sum(w, axis=1)
            safe_sw = np.maximum(sw, 1e-15)

            mbar = np.einsum("bn,ni->bi", w, mc, optimize=True) / safe_sw[:, None]
            sbar = np.einsum("bn,bni->bi", w, s, optimize=True) / safe_sw[:, None]

            dm = mc[None, :, :] - mbar[:, None, :]
            ds = s - sbar[:, None, :]

            numer = np.sum(w * np.sum(ds * dm, axis=2), axis=1)
            denom = np.sum(w * np.sum(ds * ds, axis=2), axis=1)

            ok = (
                all_valid
                & np.isfinite(sw)
                & (sw >= 4.0)
                & np.isfinite(denom)
                & (denom > _NUMERIC_REL_TOL * np.maximum(sw, 1.0))
            )

            alpha = np.full(B, np.nan, dtype=np.float64)
            alpha[ok] = numer[ok] / denom[ok]

            c_centered = np.full((B, 2), np.nan, dtype=np.float64)
            c_centered[ok] = mbar[ok] - alpha[ok, None] * sbar[ok]
            c = c_centered + m0[None, :]
            return c, alpha, ok

        c, alpha, ok = solve_batch(weights)

        for _ in range(max(0, int(irls_iterations))):
            vec = (
                m[None, :, :]
                - c[:, None, :]
                - alpha[:, None, None] * s
            )
            norms = np.linalg.norm(vec, axis=2)
            robust_w = 1.0 / np.sqrt(1.0 + (norms / f) ** 2)
            weights = valid.astype(np.float64) * robust_w
            c, alpha, ok = solve_batch(weights)

        vec = (
            m[None, :, :]
            - c[:, None, :]
            - alpha[:, None, None] * s
        )
        norms = np.linalg.norm(vec, axis=2)
        cost_m2 = 2.0 * f * f * (
            np.sqrt(1.0 + (norms / f) ** 2) - 1.0
        )
        mean_cost = np.mean(cost_m2, axis=1)
        robust_score = np.sqrt(np.maximum(mean_cost, 0.0))

        lam = alpha[:, None] / safe_vz
        pr = np.mean((lam > 0.0) & valid, axis=1)
        phys = pr >= float(min_positive_ratio)

        sl = slice(start, start + B)
        good = ok & phys & np.isfinite(robust_score)
        scores[sl] = np.where(good, robust_score, np.inf)
        pos_ratio[sl] = np.where(np.isfinite(pr), pr, 0.0)
        physical[sl] = good

    return scores, pos_ratio, physical

def global_rotation_search_shared_plane(
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    opt: Any,
) -> Dict[str, Any]:
    """
    Same global-to-local-grid search architecture as the free branch, but the
    candidate objective is the profiled shared-horizontal-plane objective.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(m))
    rays, m = rays[:n], m[:n]

    if n < int(opt.depth_min_matches):
        return {"valid": False, "reason": "not_enough_matches"}

    max_points = max(0, int(opt.globalrot_max_points))
    if max_points > 0 and n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
        rays_search = rays[idx]
        m_search = m[idx]
    else:
        rays_search = rays
        m_search = m

    coarse_grid, coarse_Rs = _coarse_grid_for_options(opt)
    scores, pos_ratio, _ = _batch_shared_plane_scores(
        coarse_grid,
        rays_search,
        m_search,
        int(opt.globalrot_batch_size),
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_grid_irls),
        min_positive_ratio=float(opt.globalrot_min_positive_ratio_init),
        rotation_matrices=coarse_Rs,
    )
    if not np.any(np.isfinite(scores)):
        return {"valid": False, "reason": "no_physical_shared_plane_grid_solution"}

    j = int(np.nanargmin(scores))
    best_angles = coarse_grid[j].copy()
    best_score = float(scores[j])
    best_pos = float(pos_ratio[j])
    total_evaluated = int(len(coarse_grid))

    base_step = np.deg2rad([
        float(opt.globalrot_tilt_step_deg),
        float(opt.globalrot_azimuth_step_deg),
        float(opt.globalrot_roll_step_deg),
    ])
    levels = max(0, int(opt.globalrot_refine_levels))
    factor = max(1.25, float(opt.globalrot_refine_factor))
    history = []

    for level in range(levels):
        step = base_step / (factor ** (level + 1))
        local_grid = _local_rotation_grid(best_angles, step)
        local_scores, local_pos, _ = _batch_shared_plane_scores(
            local_grid,
            rays_search,
            m_search,
            int(opt.globalrot_batch_size),
            f_scale=float(opt.depth_robust_f_scale),
            irls_iterations=int(opt.globalrot_grid_irls),
            min_positive_ratio=float(opt.globalrot_min_positive_ratio_init),
        )
        total_evaluated += int(len(local_grid))
        if not np.any(np.isfinite(local_scores)):
            break
        k = int(np.nanargmin(local_scores))
        best_angles = local_grid[k].copy()
        best_score = float(local_scores[k])
        best_pos = float(local_pos[k])
        history.append({
            "level": int(level + 1),
            "tilt_step_deg": float(np.rad2deg(step[0])),
            "azimuth_step_deg": float(np.rad2deg(step[1])),
            "roll_step_deg": float(np.rad2deg(step[2])),
            "score_m": best_score,
            "positive_depth_ratio": best_pos,
        })

    best = _profile_shared_plane_angles(
        best_angles,
        rays,
        m,
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_profile_irls),
    )
    if not best.get("valid"):
        return {"valid": False, "reason": "best_shared_plane_grid_solution_failed_full_profile"}
    if float(best.get("positive_depth_ratio", 0.0)) < float(opt.globalrot_min_positive_ratio_init):
        return {"valid": False, "reason": "best_shared_plane_grid_solution_failed_cheirality"}

    return {
        "valid": True,
        "best": best,
        "num_coarse_grid_rotations": int(len(coarse_grid)),
        "num_grid_rotations_evaluated": int(total_evaluated),
        "num_search_points": int(len(rays_search)),
        "search_score_m": float(best_score),
        "search_positive_depth_ratio": float(best_pos),
        "refine_levels": int(levels),
        "refine_factor": float(factor),
        "level_history": history,
    }

def _shared_plane_r_only_residual(
    angles: np.ndarray,
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    f_scale: float,
    profile_irls: int,
) -> np.ndarray:
    """
    One scalar isotropic geometric residual per correspondence:

        d_i(R) = ||m_i - c*(R) - alpha*(R) s_i(R)||_2.

    Therefore scipy soft-L1 optimizes the same robust objective used by the
    inner shared-plane profile.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    out = _profile_shared_plane_angles(
        angles,
        rays,
        map_xy,
        f_scale=float(f_scale),
        irls_iterations=int(profile_irls),
    )
    if not out.get("valid"):
        return np.full(len(rays), 1e6, dtype=np.float64)
    return np.asarray(out["plane_residual_norm"], dtype=np.float64).copy()

def refine_rotation_only_shared_plane(
    init_profile: Dict[str, Any],
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    opt: Any,
) -> Dict[str, Any]:
    """Continuous local refinement of rotation only for the shared-plane branch."""
    loss_name = str(getattr(opt, "depth_robust_loss", "soft_l1")).strip().lower()
    if loss_name != "soft_l1":
        raise ValueError("shared-plane branch requires --depth_robust_loss soft_l1")

    x0 = np.asarray(init_profile["angles"], dtype=np.float64).reshape(3)
    lower = np.array([0.0, -np.inf, -np.inf], dtype=np.float64)
    upper = np.array([0.5 * math.pi, np.inf, np.inf], dtype=np.float64)

    ls = least_squares(
        _shared_plane_r_only_residual,
        x0,
        args=(
            rays_camera,
            map_xy,
            float(opt.depth_robust_f_scale),
            int(opt.globalrot_profile_irls),
        ),
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=float(opt.depth_robust_f_scale),
        max_nfev=max(20, int(opt.globalrot_local_max_nfev)),
        x_scale="jac",
    )

    prof = _profile_shared_plane_angles(
        ls.x,
        rays_camera,
        map_xy,
        f_scale=float(opt.depth_robust_f_scale),
        irls_iterations=int(opt.globalrot_profile_irls),
    )
    if not prof.get("valid"):
        return {"valid": False, "reason": "shared_plane_r_only_refinement_profile_failed"}

    prof["meets_final_positive_depth_ratio"] = bool(
        float(prof.get("positive_depth_ratio", 0.0))
        >= float(opt.min_positive_depth_ratio_final)
    )
    prof["optimizer_cost"] = float(ls.cost)
    prof["optimizer_nfev"] = int(ls.nfev)
    prof["optimizer_message"] = str(ls.message)
    return prof

def _msac_adaptive_iteration_limit(
    current_limit: int,
    inlier_ratio: float,
    sample_size: int,
    confidence: float,
    min_iterations: int,
) -> int:
    """Standard RANSAC confidence bound; no heuristic score-dependent schedule."""
    limit = max(1, int(current_limit))
    min_it = max(1, int(min_iterations))
    w = float(np.clip(inlier_ratio, 0.0, 1.0))
    s = max(1, int(sample_size))
    p = float(np.clip(confidence, 1e-9, 1.0 - 1e-12))

    if w <= 0.0:
        return limit
    if w >= 1.0:
        return min(limit, min_it)

    p_good = float(np.clip(w ** s, 1e-15, 1.0 - 1e-15))
    denom = math.log(1.0 - p_good)
    if not np.isfinite(denom) or denom >= 0.0:
        return limit

    need = int(math.ceil(math.log(1.0 - p) / denom))
    return min(limit, max(min_it, need))

def _fixed_free_model_residuals(
    profile: Dict[str, Any],
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a FIXED Free hypothesis (R,c) on all correspondences.

    No nuisance variable is re-profiled here: doing so would allow the points
    being scored to alter the RANSAC hypothesis.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(m))
    rays, m = rays[:n], m[:n]

    R = np.asarray(profile["R"], dtype=np.float64).reshape(3, 3)
    c = np.asarray(profile["center_xy"], dtype=np.float64).reshape(2)

    v = rays @ R
    q = v[:, :2]
    q2 = np.sum(q * q, axis=1)
    qnorm = np.sqrt(q2)
    valid = (
        np.isfinite(qnorm)
        & (qnorm > 1e-12)
        & np.isfinite(m).all(axis=1)
    )

    residual = np.full(n, np.inf, dtype=np.float64)
    lam = np.full(n, np.nan, dtype=np.float64)
    if np.any(valid):
        qv = q[valid]
        mv = m[valid]
        qnv = qnorm[valid]
        normals = np.column_stack(
            [-qv[:, 1] / qnv, qv[:, 0] / qnv]
        )
        delta = mv - c[None, :]
        residual[valid] = np.abs(np.sum(normals * delta, axis=1))
        lam[valid] = (
            np.sum(qv * delta, axis=1)
            / np.maximum(q2[valid], 1e-15)
        )

    nvalid = int(np.sum(valid))
    positive_ratio = (
        float(np.sum((lam > 0.0) & valid) / nvalid)
        if nvalid > 0
        else 0.0
    )
    return residual, positive_ratio

def _fixed_shared_plane_model_residuals(
    profile: Dict[str, Any],
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a FIXED SharedPlane hypothesis (R,c,alpha) on all correspondences.

    Near-horizon rays are simply invalid/outliers for MSAC scoring.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(m))
    rays, m = rays[:n], m[:n]

    R = np.asarray(profile["R"], dtype=np.float64).reshape(3, 3)
    c = np.asarray(profile["center_xy"], dtype=np.float64).reshape(2)
    alpha = float(profile["shared_plane_alpha_m"])

    v = rays @ R
    q = v[:, :2]
    vz = v[:, 2]
    valid = (
        np.isfinite(v).all(axis=1)
        & np.isfinite(m).all(axis=1)
        & (np.abs(vz) > _SHARED_PLANE_VZ_EPS)
    )

    residual = np.full(n, np.inf, dtype=np.float64)
    lam = np.full(n, np.nan, dtype=np.float64)
    if np.any(valid):
        s = q[valid] / vz[valid, None]
        vec = m[valid] - c[None, :] - alpha * s
        residual[valid] = np.linalg.norm(vec, axis=1)
        lam[valid] = alpha / vz[valid]

    nvalid = int(np.sum(valid))
    positive_ratio = (
        float(np.sum((lam > 0.0) & valid) / nvalid)
        if nvalid > 0
        else 0.0
    )
    return residual, positive_ratio

def _msac_consensus(
    rays_camera: np.ndarray,
    map_xy: np.ndarray,
    opt: Any,
    random_seed: int,
    structural_model: str,
) -> Dict[str, Any]:
    """
    Standard single-best MSAC around the existing branch estimators.

    Hypotheses use only the structured global rotation search. The expensive
    continuous R-only refinement is intentionally reserved for the single
    final consensus refit.
    """
    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(m))
    rays, m = rays[:n], m[:n]

    sample_size = max(5, int(getattr(opt, "globalrot_msac_sample_size", 5)))
    max_iterations = max(1, int(getattr(opt, "globalrot_msac_iterations", 256)))
    min_iterations = max(
        1,
        min(
            max_iterations,
            int(getattr(opt, "globalrot_msac_min_iterations", 32)),
        ),
    )
    confidence = float(np.clip(
        getattr(opt, "globalrot_msac_confidence", 0.999),
        1e-9,
        1.0 - 1e-12,
    ))
    threshold = max(
        float(getattr(opt, "globalrot_msac_threshold_m", 5.0)),
        1e-9,
    )
    min_consensus = max(int(getattr(opt, "depth_min_matches", 8)), sample_size)

    if n < min_consensus:
        return {"valid": False, "reason": "not_enough_matches_for_msac"}

    model = str(structural_model).strip().lower()
    if model == "free":
        search_fun = global_rotation_search
        residual_fun = _fixed_free_model_residuals
    elif model in ("shared_plane", "sharedplane"):
        search_fun = global_rotation_search_shared_plane
        residual_fun = _fixed_shared_plane_model_residuals
    else:
        return {"valid": False, "reason": f"unknown_msac_structural_model:{model}"}

    # The five-match hypothesis needs the same branch search but must not be
    # rejected by the benchmark's larger full-fit minimum-match setting.
    hyp_opt = copy.copy(opt)
    hyp_opt.depth_min_matches = int(sample_size)
    hyp_opt.globalrot_max_points = 0
    # Do not run inner robust reweighting on a five-match hypothesis. MSAC on
    # the full correspondence set supplies the high-breakdown robustness; the
    # candidate ranking still uses the branch's existing geometric score, and
    # the single final consensus refit restores the original soft-L1 profiling.
    hyp_opt.globalrot_grid_irls = 0
    hyp_opt.globalrot_profile_irls = 0

    rng = np.random.default_rng(int(random_seed))
    adaptive_limit = int(max_iterations)
    best: Optional[Dict[str, Any]] = None
    num_valid_hypotheses = 0
    iterations_used = 0

    for it in range(max_iterations):
        if it >= adaptive_limit and it >= min_iterations:
            break
        iterations_used = it + 1

        sample_idx = rng.choice(n, size=sample_size, replace=False)
        gs = search_fun(
            rays[sample_idx],
            m[sample_idx],
            hyp_opt,
        )
        if not gs.get("valid"):
            continue

        profile = gs["best"]
        residual, positive_ratio = residual_fun(profile, rays, m)
        if (
            not np.isfinite(positive_ratio)
            or positive_ratio
            < float(getattr(opt, "globalrot_min_positive_ratio_init", 0.60))
        ):
            continue

        finite = np.isfinite(residual)
        inliers = finite & (residual <= threshold)
        nin = int(np.sum(inliers))
        if nin < min_consensus:
            continue

        num_valid_hypotheses += 1
        clipped = np.minimum(
            np.where(finite, residual * residual, threshold * threshold),
            threshold * threshold,
        )
        score = float(np.sum(clipped))
        key = (score, -nin)

        if best is None or key < best["key"]:
            best = {
                "key": key,
                "score_m2": score,
                "profile": profile,
                "inliers": inliers,
                "residual": residual,
                "positive_depth_ratio": float(positive_ratio),
                "sample_indices": np.asarray(sample_idx, dtype=np.int64),
            }
            adaptive_limit = _msac_adaptive_iteration_limit(
                adaptive_limit,
                nin / float(n),
                sample_size,
                confidence,
                min_iterations,
            )

    if best is None:
        return {
            "valid": False,
            "reason": "no_valid_msac_consensus",
            "msac_iterations_used": int(iterations_used),
            "msac_valid_hypotheses": int(num_valid_hypotheses),
        }

    inliers = np.asarray(best["inliers"], dtype=bool)

    # ONE final branch refit on the single best consensus. This is standard
    # RANSAC/MSAC re-estimation, not an LO-RANSAC iteration.
    final_gs = search_fun(rays[inliers], m[inliers], opt)
    if not final_gs.get("valid"):
        return {
            "valid": False,
            "reason": "msac_consensus_final_global_refit_failed",
            "msac_iterations_used": int(iterations_used),
            "msac_valid_hypotheses": int(num_valid_hypotheses),
            "msac_num_inliers": int(np.sum(inliers)),
        }

    if model == "free":
        final_profile = refine_rotation_only(
            final_gs["best"],
            rays[inliers],
            m[inliers],
            opt,
        )
    else:
        final_profile = refine_rotation_only_shared_plane(
            final_gs["best"],
            rays[inliers],
            m[inliers],
            opt,
        )

    if not final_profile.get("valid"):
        return {
            "valid": False,
            "reason": "msac_consensus_final_r_only_refit_failed",
            "msac_iterations_used": int(iterations_used),
            "msac_valid_hypotheses": int(num_valid_hypotheses),
            "msac_num_inliers": int(np.sum(inliers)),
        }

    # Diagnostics only: do NOT update the consensus and do NOT refit again.
    final_residual, final_positive_ratio = residual_fun(
        final_profile,
        rays,
        m,
    )
    final_full_inliers = (
        np.isfinite(final_residual)
        & (final_residual <= threshold)
    )

    return {
        "valid": True,
        "profile": final_profile,
        "consensus_mask": inliers,
        "hypothesis_profile": best["profile"],
        "msac_score_m2": float(best["score_m2"]),
        "msac_mean_score_m2": float(best["score_m2"] / n),
        "msac_num_inliers": int(np.sum(inliers)),
        "msac_inlier_ratio": float(np.mean(inliers)),
        "msac_threshold_m": float(threshold),
        "msac_sample_size": int(sample_size),
        "msac_iterations_used": int(iterations_used),
        "msac_adaptive_iteration_limit": int(adaptive_limit),
        "msac_max_iterations": int(max_iterations),
        "msac_min_iterations": int(min_iterations),
        "msac_confidence": float(confidence),
        "msac_valid_hypotheses": int(num_valid_hypotheses),
        "msac_hypothesis_positive_depth_ratio": float(
            best["positive_depth_ratio"]
        ),
        "msac_final_full_inlier_count_diagnostic": int(
            np.sum(final_full_inliers)
        ),
        "msac_final_full_inlier_ratio_diagnostic": float(
            np.mean(final_full_inliers)
        ),
        "msac_final_positive_depth_ratio_diagnostic": float(
            final_positive_ratio
        ),
        "final_global_search": final_gs,
    }


@dataclass(frozen=True)
class OrthoLDVPConfig:
    min_matches: int = 8
    robust_f_scale_m: float = 1.0
    min_positive_ratio_init: float = 0.60
    min_positive_ratio_final: float = 0.80
    tilt_step_deg: float = 15.0
    azimuth_step_deg: float = 30.0
    roll_step_deg: float = 30.0
    refine_levels: int = 3
    refine_factor: float = 3.0
    grid_irls: int = 2
    profile_irls: int = 3
    max_points: int = 0
    batch_size: int = 256
    fast_precompute_max_mb: float = 192.0
    local_max_nfev: int = 100
    msac_sample_size: int = 5
    msac_iterations: int = 256
    msac_min_iterations: int = 32
    msac_confidence: float = 0.999
    msac_threshold_m: float = 5.0


def _config_to_core_options(cfg: OrthoLDVPConfig) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(
        depth_min_matches=max(int(cfg.min_matches), int(cfg.msac_sample_size), 5),
        depth_robust_loss="soft_l1",
        depth_robust_f_scale=float(cfg.robust_f_scale_m),
        min_positive_depth_ratio_final=float(cfg.min_positive_ratio_final),
        globalrot_tilt_step_deg=float(cfg.tilt_step_deg),
        globalrot_azimuth_step_deg=float(cfg.azimuth_step_deg),
        globalrot_roll_step_deg=float(cfg.roll_step_deg),
        globalrot_refine_levels=int(cfg.refine_levels),
        globalrot_refine_factor=float(cfg.refine_factor),
        globalrot_grid_irls=int(cfg.grid_irls),
        globalrot_profile_irls=int(cfg.profile_irls),
        globalrot_min_positive_ratio_init=float(cfg.min_positive_ratio_init),
        globalrot_max_points=int(cfg.max_points),
        globalrot_batch_size=int(cfg.batch_size),
        globalrot_fast_precompute_max_mb=float(cfg.fast_precompute_max_mb),
        globalrot_local_max_nfev=int(cfg.local_max_nfev),
        globalrot_msac_sample_size=max(5, int(cfg.msac_sample_size)),
        globalrot_msac_iterations=max(1, int(cfg.msac_iterations)),
        globalrot_msac_min_iterations=max(1, int(cfg.msac_min_iterations)),
        globalrot_msac_confidence=float(cfg.msac_confidence),
        globalrot_msac_threshold_m=float(cfg.msac_threshold_m),
    )


def config_from_namespace(opt: Any) -> OrthoLDVPConfig:
    min_matches = int(
        getattr(
            opt,
            "globalrot_min_matches",
            getattr(opt, "depth_min_matches", 8),
        )
    )
    return OrthoLDVPConfig(
        min_matches=max(5, min_matches),
        robust_f_scale_m=float(getattr(opt, "depth_robust_f_scale", 1.0)),
        min_positive_ratio_init=float(
            getattr(opt, "globalrot_min_positive_ratio_init", 0.60)
        ),
        min_positive_ratio_final=float(
            getattr(opt, "min_positive_depth_ratio_final", 0.80)
        ),
        tilt_step_deg=float(getattr(opt, "globalrot_tilt_step_deg", 15.0)),
        azimuth_step_deg=float(getattr(opt, "globalrot_azimuth_step_deg", 30.0)),
        roll_step_deg=float(getattr(opt, "globalrot_roll_step_deg", 30.0)),
        refine_levels=int(getattr(opt, "globalrot_refine_levels", 3)),
        refine_factor=float(getattr(opt, "globalrot_refine_factor", 3.0)),
        grid_irls=int(getattr(opt, "globalrot_grid_irls", 2)),
        profile_irls=int(getattr(opt, "globalrot_profile_irls", 3)),
        max_points=int(getattr(opt, "globalrot_max_points", 0)),
        batch_size=int(getattr(opt, "globalrot_batch_size", 256)),
        fast_precompute_max_mb=float(
            getattr(opt, "globalrot_fast_precompute_max_mb", 192.0)
        ),
        local_max_nfev=int(getattr(opt, "globalrot_local_max_nfev", 100)),
        msac_sample_size=int(getattr(opt, "globalrot_msac_sample_size", 5)),
        msac_iterations=int(getattr(opt, "globalrot_msac_iterations", 256)),
        msac_min_iterations=int(getattr(opt, "globalrot_msac_min_iterations", 32)),
        msac_confidence=float(getattr(opt, "globalrot_msac_confidence", 0.999)),
        msac_threshold_m=float(getattr(opt, "globalrot_msac_threshold_m", 5.0)),
    )


def _compact_msac_metadata(msac: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "msac_score_m2",
        "msac_mean_score_m2",
        "msac_num_inliers",
        "msac_inlier_ratio",
        "msac_threshold_m",
        "msac_sample_size",
        "msac_iterations_used",
        "msac_adaptive_iteration_limit",
        "msac_max_iterations",
        "msac_min_iterations",
        "msac_confidence",
        "msac_valid_hypotheses",
        "msac_hypothesis_positive_depth_ratio",
        "msac_final_full_inlier_count_diagnostic",
        "msac_final_full_inlier_ratio_diagnostic",
        "msac_final_positive_depth_ratio_diagnostic",
    )
    return {k: msac[k] for k in keys if k in msac}


def _solve_branch(
    rays_camera: np.ndarray,
    map_xy_local: np.ndarray,
    cfg: OrthoLDVPConfig,
    random_seed: int,
    structural_model: str,
) -> Dict[str, Any]:
    t0 = time.time()

    rays = np.asarray(rays_camera, dtype=np.float64).reshape(-1, 3)
    map_xy = np.asarray(map_xy_local, dtype=np.float64).reshape(-1, 2)
    n = min(len(rays), len(map_xy))
    rays = rays[:n]
    map_xy = map_xy[:n]

    valid = np.isfinite(rays).all(axis=1) & np.isfinite(map_xy).all(axis=1)
    rays = rays[valid]
    map_xy = map_xy[valid]

    out: Dict[str, Any] = {
        "status": "failed",
        "solver": "LDVP-fast-shared-core",
        "solver_version": int(LDVP_VERSION),
        "solver_tag": LDVP_TAG,
        "structural_model": str(structural_model),
        "uses_dsm": False,
        "uses_depth_values": False,
        "uses_mde": False,
        "robust_loss": "soft_l1",
        "robust_f_scale_m": float(cfg.robust_f_scale_m),
        "shared_plane_vz_eps": float(_SHARED_PLANE_VZ_EPS),
        "num_input": int(len(rays)),
        "globalrot_min_matches": int(cfg.min_matches),
        "globalrot_msac_sample_size": int(cfg.msac_sample_size),
        "globalrot_msac_iterations": int(cfg.msac_iterations),
        "globalrot_msac_min_iterations": int(cfg.msac_min_iterations),
        "globalrot_msac_confidence": float(cfg.msac_confidence),
        "globalrot_msac_threshold_m": float(cfg.msac_threshold_m),
        "globalrot_fast_precompute_max_mb": float(cfg.fast_precompute_max_mb),
        "globalrot_fast_rotation_cache": bool(
            cfg.fast_precompute_max_mb > 0.0
        ),
    }

    min_needed = max(5, int(cfg.min_matches), int(cfg.msac_sample_size))
    if len(rays) < min_needed:
        out["failure_reason"] = (
            f"not_enough_matches: need >= {min_needed}, got {len(rays)}"
        )
        out["solver_time_s"] = float(time.time() - t0)
        return out

    core_opt = _config_to_core_options(cfg)
    msac = _msac_consensus(
        rays,
        map_xy,
        core_opt,
        random_seed=int(random_seed),
        structural_model=str(structural_model),
    )
    out.update(_compact_msac_metadata(msac))
    out["solver_time_s"] = float(time.time() - t0)

    if not msac.get("valid"):
        out["failure_reason"] = str(msac.get("reason", "msac_failed"))
        return out

    prof = msac["profile"]
    center = np.asarray(prof["center_xy"], dtype=np.float64).reshape(2)
    if not np.isfinite(center).all():
        out["failure_reason"] = "final_profile_nonfinite_center"
        return out

    angles = np.asarray(prof["angles"], dtype=np.float64).reshape(3)
    out.update(
        {
            "status": "success",
            "center_xy_local": center,
            "rotation_world_to_camera": np.asarray(prof["R"], dtype=np.float64),
            "down_tilt_deg": float(math.degrees(angles[0])),
            "azimuth_deg": float(math.degrees(_wrap_pi(angles[1]))),
            "roll_deg": float(math.degrees(_wrap_pi(angles[2]))),
            "positive_depth_ratio": float(prof["positive_depth_ratio"]),
            "robust_score_m": float(prof["robust_score_m"]),
            "optimizer_cost": float(prof["optimizer_cost"]),
            "optimizer_nfev": int(prof["optimizer_nfev"]),
            "optimizer_message": str(prof["optimizer_message"]),
        }
    )

    if structural_model == "free":
        out["rms_line_residual_m"] = float(prof["rms_line_residual_m"])
        out["center_condition"] = float(prof["center_condition"])
    else:
        out["rms_geometric_residual_m"] = float(prof["rms_geometric_residual_m"])
        out["rms_plane_residual_m"] = float(prof["rms_plane_residual_m"])
        out["shared_plane_alpha_m"] = float(prof["shared_plane_alpha_m"])

    final_gs = msac.get("final_global_search", {})
    for src, dst in (
        ("num_coarse_grid_rotations", "num_coarse_grid_rotations"),
        ("num_grid_rotations_evaluated", "num_grid_rotations_evaluated"),
        ("refine_levels", "global_refine_levels"),
        ("refine_factor", "global_refine_factor"),
        ("num_search_points", "num_global_search_points"),
        ("search_score_m", "global_search_score_m"),
    ):
        if src in final_gs:
            out[dst] = final_gs[src]

    return out


def solve_ldvp_free(
    rays_camera: np.ndarray,
    map_xy_local: np.ndarray,
    opt: Any,
    random_seed: int = 0,
) -> Dict[str, Any]:
    return _solve_branch(
        rays_camera,
        map_xy_local,
        config_from_namespace(opt),
        random_seed=int(random_seed),
        structural_model="free",
    )


