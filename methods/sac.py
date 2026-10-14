from __future__ import annotations
import hashlib
import numpy as np

def stable_seed(*parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(parts).encode()).digest()[:4],
        "little",
    )

def bilinear_sample(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    p = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    h, w = d.shape
    x, y = p[:, 0], p[:, 1]
    out = np.full(len(p), np.nan, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= w-1) & (y <= h-1)
    if not np.any(valid):
        return out
    ids = np.where(valid)[0]
    xv, yv = x[ids], y[ids]
    x0 = np.floor(xv).astype(int); y0 = np.floor(yv).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    wx = xv - x0; wy = yv - y0
    vals = np.stack([d[y0,x0], d[y0,x1], d[y1,x0], d[y1,x1]], axis=1)
    finite = np.isfinite(vals).all(axis=1)
    interp = ((1-wx)*(1-wy)*vals[:,0] + wx*(1-wy)*vals[:,1] +
              (1-wx)*wy*vals[:,2] + wx*wy*vals[:,3])
    out[ids[finite]] = interp[finite].astype(np.float32)
    return out

def backproject(uv, z, K):
    return (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T) * z[:, None]

def plane_ransac(P, rng, iterations, threshold):
    n = len(P)
    best = np.zeros(n, bool)
    if n < 3:
        return {"ok": False, "N1": 0, "mask": best}

    for _ in range(int(iterations)):
        a, b, c = P[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        nn = np.linalg.norm(normal)
        if nn < 1e-10:
            continue

        normal /= nn
        offset = -float(normal @ a)
        residual = np.abs(P @ normal + offset)
        mask = residual < float(threshold)

        if int(mask.sum()) > int(best.sum()):
            best = mask

    if int(best.sum()) < 3:
        return {"ok": False, "N1": 0, "mask": best}

    return {
        "ok": True,
        "N1": int(best.sum()),
        "mask": best,
    }

def compute_s1(depth, uv, K, *, matcher, scene_or_split, sample_id,
               iterations=2000, plane_threshold=0.003):
    z = bilinear_sample(depth, uv)
    ok = np.isfinite(z) & (z > 0)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)[ok]
    z = z[ok]
    if len(z) < 3:
        raise RuntimeError("too few valid depth matches")
    X = backproject(uv, z, np.asarray(K, dtype=np.float64))
    ok3 = np.isfinite(X).all(axis=1)
    X = X[ok3]
    z = z[ok3]
    if len(X) < 3:
        raise RuntimeError("too few finite 3D matches")
    median_depth = float(np.median(z))
    if not np.isfinite(median_depth) or median_depth <= 0:
        raise RuntimeError("invalid median matched depth")
    X = X / median_depth
    rng = np.random.default_rng(stable_seed(matcher, scene_or_split, sample_id))
    fit = plane_ransac(X, rng, int(iterations), float(plane_threshold))
    if not fit["ok"] or int(fit["N1"]) < 3:
        raise RuntimeError("plane RANSAC failed")
    n1 = int(fit["N1"])
    nmatch = int(len(X))
    return {
        "analysis_status": "success",
        "Nmatch": nmatch,
        "N1": n1,
        "S1": float(n1 / nmatch),
        "N1_over_Nmatch": float(n1 / nmatch),
        "plane1_threshold": float(plane_threshold),
        "plane1_threshold_units": "dimensionless_after_median_depth_normalization",
        "median_matched_depth": median_depth,
    }

def choose_branch(s1, threshold):
    if s1 is not None and np.isfinite(s1) and float(s1) >= float(threshold):
        return "Planar-PnP"
    return "LDVP"
