from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py, cv2
from .ortholoc_common import (
    LazyMatcher, build_common, build_match_cache_index, list_dataset_queries,
    load_ortholoc_sample, resolve_match_geometry,
)

def query_list(opt):
    return list_dataset_queries(opt.dataset_root,set_name=opt.set_name,sample_glob=opt.sample_glob,
                                sample_ratio=opt.sample_ratio,seed=opt.seed,limit=opt.limit)

def prepare(split,npz_path,run_root,opt,cache_index,lazy_matcher):
    sample=load_ortholoc_sample(npz_path,use_refined_extrinsics=opt.use_refined_extrinsics)
    sd=Path(run_root)/split/str(sample["sample_id"]); sd.mkdir(parents=True,exist_ok=True)
    geom=resolve_match_geometry(sample,split,sd,opt,cache_index,lazy_matcher)
    return sample,build_common(sample,split,geom,sd)

def _origin(common):
    xy=np.c_[np.asarray(common.world_x,np.float64),np.asarray(common.world_y,np.float64)]
    finite=np.isfinite(xy).all(1)
    if not np.any(finite): raise ValueError("No finite OrthoLoC world XY")
    return np.median(xy[finite],axis=0)

def _rays(img,K):
    hom=np.c_[img,np.ones(len(img))]
    r=(np.linalg.inv(np.asarray(K,np.float64))@hom.T).T
    r/=np.clip(r[:,2:3],1e-12,None)
    return r

def normalized_inputs(common):
    img=np.asarray(common.uav_points_original,np.float64).reshape(-1,2)
    xy=np.c_[np.asarray(common.world_x,np.float64),np.asarray(common.world_y,np.float64)]
    n=min(len(img),len(xy)); img,xy=img[:n],xy[:n]
    origin=_origin(common); local=xy-origin
    rays=_rays(img,common.K)
    valid=np.isfinite(img).all(1)&np.isfinite(local).all(1)&np.isfinite(rays).all(1)
    return img[valid],local[valid],rays[valid],np.flatnonzero(valid),origin

def pnp_distortion(common): return None

def dsm_inputs(common):
    img=np.asarray(common.uav_points_original,np.float64).reshape(-1,2)
    xyz=np.asarray(common.world_xyz,np.float64).reshape(-1,3)
    n=min(len(img),len(xyz)); img,xyz=img[:n],xyz[:n]
    valid=np.isfinite(img).all(1)&np.isfinite(xyz).all(1); img,xyz=img[valid],xyz[valid]
    origin=_origin(common); local=xyz.copy(); local[:,0]-=origin[0]; local[:,1]-=origin[1]
    return img,local[:,:2],local[:,2],origin

def planar_inputs(common):
    img=np.asarray(common.uav_points_original,np.float64).reshape(-1,2)
    xy=np.c_[np.asarray(common.world_x,np.float64),np.asarray(common.world_y,np.float64)]
    n=min(len(img),len(xy)); img,xy=img[:n],xy[:n]; origin=_origin(common); local=xy-origin
    valid=np.isfinite(img).all(1)&np.isfinite(local).all(1)
    return img[valid],local[valid],origin

def restore_center(common,center_local,origin):
    c=np.asarray(center_local,np.float64).copy(); c[0]+=origin[0]; c[1]+=origin[1]; return c

def result_row(common,method,status,center=None,**extra):
    r={"scene_name":str(common.scene_name),"split_name":str(common.split_name),"sample_id":str(common.sample_id),
       "npz_path":str(common.npz_path),"method":method,"status":status,
       "true_x":float(common.true_pos["x"]),"true_y":float(common.true_pos["y"]),"true_z":float(common.true_pos["z"]),
       "num_matches":int(len(common.uav_points_original))}
    if center is not None and np.isfinite(center[:2]).all():
        r["estimated_x"],r["estimated_y"]=float(center[0]),float(center[1])
        r["estimated_z"]=float(center[2]) if len(center)>2 and np.isfinite(center[2]) else None
        dx=r["estimated_x"]-r["true_x"];dy=r["estimated_y"]-r["true_y"]
        r.update(error_x_meters=float(dx),error_y_meters=float(dy),error_meters=float(np.hypot(dx,dy)))
    else:r.update(estimated_x=None,estimated_y=None,estimated_z=None,error_x_meters=None,error_y_meters=None,error_meters=None)
    r.update(extra);return r

class DepthSource:
    """Read OrthoLoC depth only from samples/<split>/<sample_id>/depth in HDF5."""
    def __init__(self, root):
        self.path = Path(root).expanduser().resolve() if root else None
        self.h5 = None
        if self.path is not None:
            if not self.path.is_file():
                raise FileNotFoundError(f"OrthoLoC depth HDF5 not found: {self.path}")
            if self.path.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("--depth_root must be an OrthoLoC HDF5 file")
            self.h5 = h5py.File(self.path, "r")

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def load(self, common):
        if self.h5 is None:
            raise ValueError("--depth_root is required for LDVP+SAC")
        split = str(common.split_name)
        sid = str(common.sample_id)
        key = f"samples/{split}/{sid}/depth"
        if key not in self.h5:
            raise KeyError(f"HDF5 missing depth: {key}")
        depth = np.asarray(self.h5[key], dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"{key}: depth must be 2-D after squeeze, got {depth.shape}")

        uv = np.asarray(common.uav_points_original, dtype=np.float64).copy()
        K = np.asarray(common.K, dtype=np.float64).copy()
        H, W = common.image_shape
        hd, wd = depth.shape
        if (hd, wd) != (H, W):
            sx = wd / float(W)
            sy = hd / float(H)
            uv[:, 0] *= sx
            uv[:, 1] *= sy
            K[0, :] *= sx
            K[1, :] *= sy
        return depth, uv, K
