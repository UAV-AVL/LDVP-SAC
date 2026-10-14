# Runtime environment:
#   conda activate moge2_aerial
#
# Requirements:
#   1) AerialMetric / MoGe2-Aerial can already perform single-image inference successfully
#   2) The current environment has ortholoc installed
#
# Functionality:
#   1) Iterate over *.npz under OrthoLoC train / val / test_inPlace / test_outPlace
#   2) Read each sample using the official OrthoLoC dataset loader
#   3) Run MoGe2-Aerial metric depth estimation on sample["image_query"]
#   4) Save:
#        sample_id/
#          depth.npy
#          depth_colormap.png
#          meta.json
#   5) Keep the depth.npy interface compatible with the existing AnyVisLoc / DECO pipeline
#
# Important:
#   The official MoGe2-Aerial infer_single() performs:
#       original image -> resize to MOGE_RESIZE -> MoGe inference -> resize depth back to original image size
#
#   Therefore, the final output satisfies:
#       depth.shape == image_query.shape[:2]
#
#   depth unit:
#       meters

import os

# =============================================================================
# Recommended to set before importing torch
# =============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import cv2
import json
import h5py
import time
import torch
import numpy as np

from pathlib import Path
from collections import Counter
from tqdm import tqdm


# =============================================================================
# User configuration
# =============================================================================

# -----------------------------------------------------------------------------
# 1. OrthoLoC dataset
# -----------------------------------------------------------------------------

ORTHOLOC_ROOT = r"/ff/yhr/OrthoLoC/data/OrthoLoC/"

# Supports both of the following layouts:
#
# /data/OrthoLoC/
# ├── train/
# ├── val/
# ├── test_inPlace/
# └── test_outPlace/
#
# Or:
#
# /data/OrthoLoC/
# └── full/
#     ├── train/
#     ├── val/
#     ├── test_inPlace/
#     └── test_outPlace/

SPLITS = [
    "train",
    "val",
    "test_inPlace",
    "test_outPlace",
]


# -----------------------------------------------------------------------------
# 2. Output directory
# -----------------------------------------------------------------------------

SAVE_BASE_PATH = r"/ff/yhr/MDE_Exp/OrthoLoc_depth_MoGe2Aerial"
H5_NAME = "ortholoc_moge2_aerial_depth.h5"


# -----------------------------------------------------------------------------
# 3. AerialMetric / MoGe2-Aerial
# -----------------------------------------------------------------------------

AERIALMETRIC_ROOT = r"/ff/yhr/AerialMetric"

MOGE_CHECKPOINT = (
    r"/ff/yhr/AerialMetric/weights/Moge2-Aerial.pt"
)

MOGE_LORA_CONFIG = (
    r"/ff/yhr/AerialMetric/MoGe/configs/train/v2.json"
)


# -----------------------------------------------------------------------------
# 4. MoGe2-Aerial inference parameters
#
# Keep consistent with the command that was verified successfully:
#
# --resize 1024
# --resolution_level 9
# -----------------------------------------------------------------------------

MOGE_LORA_RANK = 96

MOGE_RESIZE = 1024

MOGE_RESOLUTION_LEVEL = 9

MOGE_FORCE_PROJECTION = True

MOGE_APPLY_MASK = True

MOGE_FP16 = True


# -----------------------------------------------------------------------------
# 5. Whether to use camera intrinsics provided by OrthoLoC to constrain FoV
#
# False:
#   Exactly reproduces the previously verified demo_infer.py setup.
#   MoGe estimates camera intrinsics / FoV by itself.
#
# True:
#   Compute fov_x from sample["intrinsics_query"],
#   then pass it to MoGe.
#
# False is recommended for the first batch generation to match the verified configuration exactly.
# -----------------------------------------------------------------------------

USE_ORTHOLOC_INTRINSICS_FOR_FOV = False


# -----------------------------------------------------------------------------
# 6. Whether to save split subdirectories
#
# False:
#
# SAVE_BASE_PATH/
# ├── sample001/
# │   └── depth.npy
# └── sample002/
#     └── depth.npy
#
# Compatible with the existing DECO layout:
#
#   depth_root / sample_id / depth.npy
#
# Fully compatible.
#
# If duplicate sample_id values exist across OrthoLoC splits,
# set this to True.
# -----------------------------------------------------------------------------

# Existing results are actually stored as:
#   SAVE_BASE_PATH/<split>/<sample_id>/depth.npy
# Therefore a split-aware path must be used.
SAVE_WITH_SPLIT = True


# -----------------------------------------------------------------------------
# 7. Whether to skip if depth.npy already exists
# -----------------------------------------------------------------------------

SKIP_EXISTING = True


# =============================================================================
# Path checks + imports
# =============================================================================

AERIALMETRIC_ROOT_PATH = Path(AERIALMETRIC_ROOT)

if not AERIALMETRIC_ROOT_PATH.exists():
    raise FileNotFoundError(
        f"AerialMetric root directory does not exist:\n{AERIALMETRIC_ROOT}"
    )

if not Path(MOGE_CHECKPOINT).exists():
    raise FileNotFoundError(
        f"MoGe2-Aerial checkpoint does not exist:\n{MOGE_CHECKPOINT}"
    )

if not Path(MOGE_LORA_CONFIG).exists():
    raise FileNotFoundError(
        f"MoGe2 config does not exist:\n{MOGE_LORA_CONFIG}"
    )


# Ensure /ff/yhr/AerialMetric/demo_infer.py can be imported
if AERIALMETRIC_ROOT not in sys.path:
    sys.path.insert(0, AERIALMETRIC_ROOT)


# =============================================================================
# Directly reuse the official demo_infer.py
#
# load_lora_model:
#     Load MoGe2 + PEFT LoRA
#
# infer_single:
#     BGR image -> metric depth
#
# intrinsics_to_fov:
#     intrinsics -> FoV
# =============================================================================

from demo_infer import (
    load_lora_model,
    infer_single,
    intrinsics_to_fov,
)


# Official OrthoLoC dataset loader
from ortholoc.dataset import OrthoLoC


# =============================================================================
# Device
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP16 = MOGE_FP16 and DEVICE.startswith("cuda")


# =============================================================================
# Data discovery
# =============================================================================

def find_npz_samples(root_dir: str, splits=None):
    """
    Find OrthoLoC *.npz files.

    Automatically supports:

    Layout A:
        root/train/*.npz

    Layout B:
        root/full/train/*.npz
    """

    root = Path(root_dir)

    if splits is None:
        splits = SPLITS

    if not root.exists():
        raise FileNotFoundError(
            f"OrthoLoC root directory does not exist: {root}"
        )

# Automatically detect whether root contains full/
    if (root / "full").is_dir():
        full_root = root / "full"
        print(f"[Data directory] Detected full/: {full_root}")
    else:
        full_root = root
        print(f"[Data directory] Using: {full_root}")

    items = []

    for split in splits:

        split_dir = full_root / split

        if not split_dir.exists():
            print(
                f"[Warning] Directory does not exist: {split_dir}; skipping"
            )
            continue

        npz_files = sorted(split_dir.glob("*.npz"))

        print(
            f"[{split}] Found {len(npz_files)} npz files"
        )

        for npz_path in npz_files:

            items.append(
                {
                    "split": split,
                    "sample_id": npz_path.stem,
                    "sample_path": npz_path,
                }
            )

    return items


# =============================================================================
# Image format conversion
# =============================================================================

def to_bgr_uint8(img):
    """
    Convert OrthoLoC image_query to BGR uint8 required by MoGe2-Aerial.

    Supports:
        H x W
        H x W x 3
        H x W x 4

    Also supports a small number of CHW cases:
        3 x H x W
        4 x H x W
    """

    img = np.asarray(img)

    # -------------------------------------------------------------------------
    # CHW -> HWC
    # -------------------------------------------------------------------------

    if (
        img.ndim == 3
        and img.shape[0] in (1, 3, 4)
        and img.shape[-1] not in (1, 3, 4)
    ):
        img = np.transpose(img, (1, 2, 0))

    # -------------------------------------------------------------------------
    # float -> uint8
    # -------------------------------------------------------------------------

    if img.dtype != np.uint8:

        if np.issubdtype(img.dtype, np.floating):

            finite = img[np.isfinite(img)]

            if finite.size == 0:
                raise ValueError(
                    "image_query contains only non-finite values"
                )

            max_value = float(finite.max())

            # Common OrthoLoC / torchvision:
            # float [0, 1]
            if max_value <= 1.0 + 1e-6:

                img = (
                    img * 255.0
                ).clip(
                    0, 255
                ).astype(
                    np.uint8
                )

            else:

                img = img.clip(
                    0, 255
                ).astype(
                    np.uint8
                )

        else:

            img = img.astype(np.uint8)

    # -------------------------------------------------------------------------
    # Gray -> BGR
    # -------------------------------------------------------------------------

    if img.ndim == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    # -------------------------------------------------------------------------
    # H W 1 -> BGR
    # -------------------------------------------------------------------------

    elif (
        img.ndim == 3
        and img.shape[2] == 1
    ):

        img = cv2.cvtColor(
            img[:, :, 0],
            cv2.COLOR_GRAY2BGR
        )

    # -------------------------------------------------------------------------
    # OrthoLoC is interpreted as RGB by default
    # RGB -> BGR
    # -------------------------------------------------------------------------

    elif (
        img.ndim == 3
        and img.shape[2] == 3
    ):

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    # -------------------------------------------------------------------------
    # RGBA -> BGR
    # -------------------------------------------------------------------------

    elif (
        img.ndim == 3
        and img.shape[2] == 4
    ):

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR
        )

    else:

        raise ValueError(
            f"Unsupported image_query shape: {img.shape}"
        )

    return np.ascontiguousarray(img)


# =============================================================================
# Depth visualization
# =============================================================================

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """
    depth.npy is the actual data used in downstream computation.

    This function is only used to generate depth_colormap.png.
    """

    d = np.asarray(
        depth,
        dtype=np.float32
    ).copy()

    valid = (
        np.isfinite(d)
        & (d > 0)
    )

    if not np.any(valid):

        return np.zeros(
            (*d.shape, 3),
            dtype=np.uint8
        )

    vmin = float(
        np.percentile(
            d[valid],
            2
        )
    )

    vmax = float(
        np.percentile(
            d[valid],
            98
        )
    )

    if vmax <= vmin:

        vmax = vmin + 1e-6

    norm = (
        d - vmin
    ) / (
        vmax - vmin
    )

    norm = np.clip(
        norm,
        0.0,
        1.0
    )

    norm[~valid] = 0

    color = cv2.applyColorMap(
        (norm * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    return color


# =============================================================================
# Compute the actual internal MoGe inference resolution
# =============================================================================

def compute_moge_process_size(
    h: int,
    w: int,
    resize: int
):
    """
    Corresponds to the logic in the official demo_infer.py:

        long edge -> resize
        then round dimensions down to multiples of 14

    Note:
        This is only the network's internal processing size.

        The final depth is resized back to:
            (h, w)
    """

    if resize is not None and resize > 0:

        scale = resize / max(h, w)

        new_w = int(
            w * scale
        )

        new_h = int(
            h * scale
        )

    else:

        new_w = w
        new_h = h

    new_w = max(
        14,
        (new_w // 14) * 14
    )

    new_h = max(
        14,
        (new_h // 14) * 14
    )

    return new_h, new_w


# =============================================================================
# MoGe2-Aerial inference
# =============================================================================

def infer_moge2_aerial_depth_from_bgr(
    model,
    image_bgr: np.ndarray,
    given_fov_x=None,
):
    """
    Use infer_single() from the official demo_infer.py.

    Input:
        image_bgr:
            H x W x 3
            uint8
            BGR

    Returns:
        depth:
            H x W
            float32
            metric depth
            meters

        pred_intrinsics:
            3 x 3
            Intrinsics predicted/used by MoGe

        elapsed:
            inference seconds
    """

    depth, pred_intrinsics, elapsed = infer_single(

        model=model,

        img_bgr=image_bgr,

        device=DEVICE,

        fp16=FP16,

        resize=MOGE_RESIZE,

        resolution_level=MOGE_RESOLUTION_LEVEL,

        force_projection=MOGE_FORCE_PROJECTION,

        apply_mask=MOGE_APPLY_MASK,

        intrinsics_mode="none",

        given_fov_x=given_fov_x,
    )

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    depth = np.squeeze(depth)

    if depth.ndim != 2:

        raise RuntimeError(
            f"Unexpected MoGe depth dimensions: {depth.shape}"
        )

    return (
        depth,
        pred_intrinsics,
        elapsed,
    )


# =============================================================================
# Save path
# =============================================================================

def get_save_dir(
    save_root,
    split,
    sample_id
):
    """
    Current split-aware layout:

        depth_root/<split>/<sample_id>/depth.npy

    Example:

        /ff/yhr/MDE_Exp/OrthoLoc_depth_MoGe2Aerial/
            train/L01_R0000/depth.npy
    """

    if SAVE_WITH_SPLIT:

        return (
            Path(save_root)
            / split
            / sample_id
        )

    return (
        Path(save_root)
        / sample_id
    )


# =============================================================================
# Resume support + scale-missing compatible loading
# =============================================================================

def collect_pending_samples(samples):
    """
    Keep only samples that actually require inference.

    Existing results are saved by split:
        SAVE_BASE_PATH/<split>/<sample_id>/depth.npy

    When SKIP_EXISTING=True, a sample is considered complete as long as the corresponding depth.npy exists,
    so it will not enter the OrthoLoC loader and MoGe will not be run again.
    """

    pending = []
    existing = 0

    per_split = {}

    for item in samples:
        split = item["split"]

        if split not in per_split:
            per_split[split] = {
                "total": 0,
                "existing": 0,
                "pending": 0,
            }

        per_split[split]["total"] += 1

        depth_path = (
            get_save_dir(
                SAVE_BASE_PATH,
                split,
                item["sample_id"],
            )
            / "depth.npy"
        )

        if SKIP_EXISTING and depth_path.is_file():
            existing += 1
            per_split[split]["existing"] += 1
        else:
            pending.append(item)
            per_split[split]["pending"] += 1

    return pending, existing, per_split


def load_ortholoc_sample_scale_safe(sample_path):
    """
    Prefer the official OrthoLoC loader.

    If the official loader fails and the original NPZ is confirmed to be missing scale, directly read only
    the fields actually required for the current MoGe2-Aerial depth generation and bypass scale.

    This does not fabricate scale or modify the original NPZ. For this script, metric depth
    inference input is image_query; with the current default USE_ORTHOLOC_INTRINSICS_FOR_FOV=False,
    scale is therefore not used by MoGe2-Aerial inference.
    """

    sample_path = Path(sample_path)

    try:
        dataset = OrthoLoC(
            sample_paths=[str(sample_path)],
            return_tensor=False,
        )
        sample = dataset[0]

        return sample, {
            "mode": "ortholoc_official",
            "scale_missing": False,
            "npz_keys": None,
            "official_loader_error": None,
        }

    except Exception as official_exc:
        # Enable fallback only when the original NPZ is confirmed to be missing scale.
        # If scale is actually present, the official loader failed for another reason,
        # so re-raise the original exception to avoid hiding the real data problem.
        with np.load(sample_path, allow_pickle=True) as data:
            keys = list(data.files)

            if "scale" in data.files:
                raise official_exc

            if "image_query" not in data.files:
                raise KeyError(
                    f"{sample_path.name} is missing both scale and image_query; "
                    f"keys={keys}"
                ) from official_exc

            sample = {
                "image_query": data["image_query"],
            }

            # These fields are not required by the current default MoGe inference, but they are small,
            # so retain them when present for FoV / metadata recording.
            for key in (
                "intrinsics_query",
                "pose_query2world",
            ):
                if key in data.files:
                    sample[key] = data[key]

            # image_dop / dsm are not required for depth inference in this script.
            # When scale is missing, do not attempt to construct world-coordinate DSM data manually, to avoid introducing incorrect geometry.

        return sample, {
            "mode": "raw_npz_missing_scale_fallback",
            "scale_missing": True,
            "npz_keys": keys,
            "official_loader_error": (
                f"{type(official_exc).__name__}: {official_exc}"
            ),
        }



# =============================================================================
# HDF5 helpers
# =============================================================================

def _safe_h5_component(s: str) -> str:
    return str(s).replace("/", "__slash__")


def _write_h5_attr(obj, key, value):
    if value is None:
        obj.attrs[key] = "__NONE__"
    elif isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
        obj.attrs[key] = value
    else:
        obj.attrs[key] = json.dumps(value, ensure_ascii=False)


def _h5_sample_complete(h5, split, sample_id):
    group_path = (
        f"samples/{_safe_h5_component(split)}/"
        f"{_safe_h5_component(sample_id)}"
    )
    if group_path not in h5:
        return False
    grp = h5[group_path]
    return ("depth" in grp) and bool(grp.attrs.get("complete", False))


def main():
    print("=" * 78)
    print("MoGe2-Aerial metric depth estimation for OrthoLoC -> SINGLE HDF5")
    print("=" * 78)
    print(f"Device                 : {DEVICE}")
    print(f"FP16                   : {FP16}")
    print(f"AerialMetric root      : {AERIALMETRIC_ROOT}")
    print(f"Checkpoint             : {MOGE_CHECKPOINT}")
    print(f"LoRA config            : {MOGE_LORA_CONFIG}")
    print(f"LoRA rank              : {MOGE_LORA_RANK}")
    print(f"Resize                 : {MOGE_RESIZE}")
    print(f"Resolution level       : {MOGE_RESOLUTION_LEVEL}")
    print(f"Use OrthoLoC intrinsics: {USE_ORTHOLOC_INTRINSICS_FOR_FOV}")
    print(f"Save root              : {SAVE_BASE_PATH}")
    print(f"HDF5 name              : {H5_NAME}")
    print("=" * 78)

    save_root = Path(SAVE_BASE_PATH)
    save_root.mkdir(parents=True, exist_ok=True)
    h5_path = save_root / H5_NAME

    samples = find_npz_samples(ORTHOLOC_ROOT, splits=SPLITS)
    print(f"\nFound {len(samples)} .npz samples in total\n")
    if not samples:
        print("No .npz files found. Please check ORTHOLOC_ROOT.")
        return

    # Open once for the whole run. The complete marker is written last per sample.
    with h5py.File(h5_path, "a", libver="latest") as h5:
        h5.attrs["format"] = "OrthoLoC_MoGe2_Aerial_depth_v1"
        h5.attrs["dataset_root"] = str(ORTHOLOC_ROOT)
        h5.attrs["model"] = "MoGe2-Aerial"
        h5.attrs["checkpoint"] = str(MOGE_CHECKPOINT)
        h5.attrs["lora_config"] = str(MOGE_LORA_CONFIG)
        h5.attrs["lora_rank"] = int(MOGE_LORA_RANK)
        h5.attrs["resize_long_edge"] = int(MOGE_RESIZE)
        h5.attrs["resolution_level"] = int(MOGE_RESOLUTION_LEVEL)
        h5.attrs["depth_unit"] = "meter"
        h5.require_group("samples")

        pending = []
        skipped_existing = 0
        split_stats = {
            split: {"total": 0, "existing": 0, "pending": 0}
            for split in SPLITS
        }
        for item in samples:
            split = str(item["split"])
            sid = str(item["sample_id"])
            split_stats.setdefault(
                split, {"total": 0, "existing": 0, "pending": 0}
            )
            split_stats[split]["total"] += 1
            if SKIP_EXISTING and _h5_sample_complete(h5, split, sid):
                skipped_existing += 1
                split_stats[split]["existing"] += 1
            else:
                pending.append(item)
                split_stats[split]["pending"] += 1

        print("=" * 78)
        print("HDF5 resume scan")
        print("=" * 78)
        print(f"HDF5                 : {h5_path}")
        print("Data layout            : samples/<split>/<sample_id>/depth")
        print(f"Source samples total   : {len(samples)}")
        print(f"Completed              : {skipped_existing}")
        print(f"Pending inference      : {len(pending)}")
        print("-" * 78)
        for split in SPLITS:
            st = split_stats.get(split, {"total": 0, "existing": 0, "pending": 0})
            print(
                f"{split:16s} total={st['total']:5d}  "
                f"existing={st['existing']:5d}  pending={st['pending']:5d}"
            )
        print("=" * 78)

        if not pending:
            print("All samples are already fully saved in HDF5; no inference is needed.")
            return

        print("\nLoading MoGe2-Aerial ...")
        t_model = time.time()
        model = load_lora_model(
            config_path=MOGE_LORA_CONFIG,
            lora_path=MOGE_CHECKPOINT,
            lora_rank=MOGE_LORA_RANK,
            device=DEVICE,
            fp16=FP16,
        )
        print(f"Model loaded: {time.time() - t_model:.1f}s\n")

        success = 0
        skipped = skipped_existing
        failed = 0
        total_infer_time = 0.0
        since_flush = 0
        flush_every = 20

        progress = tqdm(
            pending, total=len(pending), desc="MoGe2-Aerial",
            unit="sample", dynamic_ncols=True
        )

        for index, item in enumerate(progress, start=1):
            split = str(item["split"])
            sample_id = str(item["sample_id"])
            sample_path = Path(item["sample_path"])
            progress.set_description(f"{split}/{sample_id}")

            split_key = _safe_h5_component(split)
            sid_key = _safe_h5_component(sample_id)
            group_path = f"samples/{split_key}/{sid_key}"
            prefix_msg = f"[{index}/{len(pending)}] {split}/{sample_id}"

            # Re-check lazily in case the file changed after the initial scan.
            if SKIP_EXISTING and _h5_sample_complete(h5, split, sample_id):
                skipped += 1
                continue

            try:
                sample, sample_load_info = load_ortholoc_sample_scale_safe(sample_path)
                image_query = sample["image_query"]
                image_dop = sample.get("image_dop", None)
                dsm = sample.get("dsm", None)
                intrinsics_query = sample.get("intrinsics_query", None)
                pose_query2world = sample.get("pose_query2world", None)

                query_bgr = to_bgr_uint8(image_query)
                orig_h, orig_w = query_bgr.shape[:2]
                process_h, process_w = compute_moge_process_size(
                    orig_h, orig_w, MOGE_RESIZE
                )

                given_fov_x = None
                dataset_fov_info = None
                if USE_ORTHOLOC_INTRINSICS_FOR_FOV and intrinsics_query is not None:
                    dataset_fov_info = intrinsics_to_fov(
                        intrinsics_query, orig_w, orig_h
                    )
                    given_fov_x = dataset_fov_info["fov_x_deg"]

                depth, pred_intrinsics, elapsed = (
                    infer_moge2_aerial_depth_from_bgr(
                        model=model,
                        image_bgr=query_bgr,
                        given_fov_x=given_fov_x,
                    )
                )
                total_infer_time += float(elapsed)

                depth_h, depth_w = depth.shape
                if depth_h != orig_h or depth_w != orig_w:
                    raise RuntimeError(
                        "MoGe2-Aerial depth size does not match the original query image: "
                        f"RGB={orig_w}x{orig_h}, Depth={depth_w}x{depth_h}"
                    )

                valid_mask = np.isfinite(depth) & (depth > 0)
                d_valid = depth[valid_mask]
                if d_valid.size == 0:
                    raise RuntimeError("Depth map contains no valid values > 0")

                depth_min = float(d_valid.min())
                depth_max = float(d_valid.max())
                depth_median = float(np.median(d_valid))
                depth_mean = float(np.mean(d_valid))

                model_fov_info = None
                if pred_intrinsics is not None:
                    try:
                        model_fov_info = intrinsics_to_fov(
                            pred_intrinsics, orig_w, orig_h
                        )
                    except Exception:
                        model_fov_info = None

                grp = h5.require_group(group_path)

                # Mark incomplete before touching payload.
                grp.attrs["complete"] = False

                if "depth" in grp:
                    del grp["depth"]
                grp.create_dataset(
                    "depth",
                    data=depth.astype(np.float32, copy=False),
                    dtype=np.float32,
                    chunks=True,
                    compression="lzf",
                    shuffle=True,
                )

                if "predicted_intrinsics" in grp:
                    del grp["predicted_intrinsics"]
                if pred_intrinsics is not None:
                    grp.create_dataset(
                        "predicted_intrinsics",
                        data=np.asarray(pred_intrinsics, dtype=np.float32),
                        dtype=np.float32,
                    )

                meta = {
                    "dataset": "OrthoLoC",
                    "source_type": "full_npz",
                    "split": split,
                    "sample_id": sample_id,
                    "sample_path": str(sample_path),
                    "sample_load_mode": sample_load_info["mode"],
                    "source_scale_missing": bool(sample_load_info["scale_missing"]),
                    "official_loader_error":
                        sample_load_info["official_loader_error"],
                    "depth_model": "MoGe2-Aerial",
                    "checkpoint": MOGE_CHECKPOINT,
                    "lora_config": MOGE_LORA_CONFIG,
                    "lora_rank": int(MOGE_LORA_RANK),
                    "depth_unit": "meter",
                    "device": DEVICE,
                    "fp16": bool(FP16),
                    "resize": int(MOGE_RESIZE),
                    "resolution_level": int(MOGE_RESOLUTION_LEVEL),
                    "force_projection": bool(MOGE_FORCE_PROJECTION),
                    "apply_mask": bool(MOGE_APPLY_MASK),
                    "use_ortholoc_intrinsics_for_fov":
                        bool(USE_ORTHOLOC_INTRINSICS_FOR_FOV),
                    "orig_w": int(orig_w),
                    "orig_h": int(orig_h),
                    "process_w": int(process_w),
                    "process_h": int(process_h),
                    "depth_w": int(depth_w),
                    "depth_h": int(depth_h),
                    "infer_w": int(depth_w),
                    "infer_h": int(depth_h),
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                    "depth_mean": depth_mean,
                    "depth_median": depth_median,
                    "n_valid": int(valid_mask.sum()),
                    "valid_ratio": float(valid_mask.mean()),
                    "inference_seconds": float(elapsed),
                    "has_image_dop": image_dop is not None,
                    "has_dsm": dsm is not None,
                    "has_intrinsics_query": intrinsics_query is not None,
                    "has_pose_query2world": pose_query2world is not None,
                    "ortholoc_intrinsics_fov": dataset_fov_info,
                    "moge_pred_fov": model_fov_info,
                }

                if image_dop is not None:
                    meta["image_dop_shape"] = list(np.asarray(image_dop).shape)
                if dsm is not None:
                    meta["dsm_shape"] = list(np.asarray(dsm).shape)
                if intrinsics_query is not None:
                    meta["intrinsics_query"] = (
                        np.asarray(intrinsics_query).tolist()
                    )
                if pose_query2world is not None:
                    meta["pose_query2world"] = (
                        np.asarray(pose_query2world).tolist()
                    )

                for k, v in meta.items():
                    _write_h5_attr(grp, k, v)

                # Must be last: only a fully written sample is resumable/skippable.
                grp.attrs["complete"] = True

                success += 1
                since_flush += 1
                print(
                    f"[ok] {prefix_msg}  {orig_w}x{orig_h}  "
                    f"depth=[{depth_min:.2f},{depth_max:.2f}]m  "
                    f"median={depth_median:.2f}m  {elapsed:.2f}s"
                )

                if since_flush >= flush_every:
                    h5.flush()
                    since_flush = 0

            except KeyboardInterrupt:
                h5.flush()
                progress.close()
                print("\nInterrupted; HDF5 has been flushed. Re-run the same script to resume.")
                return
            except Exception as exc:
                failed += 1
                print(
                    f"[error] {prefix_msg}: "
                    f"{type(exc).__name__}: {exc}"
                )
                import traceback
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        progress.close()
        h5.attrs["source_samples"] = int(len(samples))
        h5.attrs["last_success"] = int(success)
        h5.attrs["last_skipped"] = int(skipped)
        h5.attrs["last_failed"] = int(failed)
        h5.flush()

    print("\n" + "=" * 78)
    print("Done")
    print("=" * 78)
    print(f"Success              : {success}")
    print(f"Skipped              : {skipped}")
    print(f"Failed               : {failed}")
    print(f"Inference total(s)   : {total_infer_time:.2f}")
    if success:
        print(f"Inference avg(s)     : {total_infer_time / success:.3f}")
    print(f"Single HDF5          : {h5_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
