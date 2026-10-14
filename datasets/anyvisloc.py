from __future__ import annotations
from pathlib import Path
import cv2, h5py, numpy as np
from .anyvisloc_common import (
    AnyVisLocNPZDataset, MatcherManager, build_geometry_match_data,
    clear_anyvisloc_reference_array_cache, prepare_common_sample_data,
    set_random_seed, to_python_str,
    uniform_sample_indices_per_scene, verify_reference_mode,
)
from avl_utils import sample_dsm_local

def dataset_and_indices(opt):
    ds=AnyVisLocNPZDataset(root=opt.dataset_root,scenes=opt.scenes,load_reference=False,cache_reference=True,recursive=True)
    idx,report=uniform_sample_indices_per_scene(ds,opt.sample_ratio,opt.max_samples_per_scene)
    return ds,idx,report

def prepare(sample,run_root,opt,matcher):
    return prepare_common_sample_data(sample,run_root,opt,matcher)

def normalized_inputs(common):
    geom=build_geometry_match_data(common)
    # geom.map_xy_world is exactly the supplied AnyVisLoc LDVP convention [x,-y].
    return geom.image_points,geom.map_xy_world,geom.rays_camera,geom.indices

def pnp_distortion(common):
    return None if bool(common.uav_was_undistorted) else common.distortion

def dsm_inputs(common):
    img=np.asarray(common.uav_points_original,np.float64).reshape(-1,2)
    x=np.asarray(common.local_x,np.float64).reshape(-1)
    y=np.asarray(common.local_y,np.float64).reshape(-1)
    dsm=np.asarray(common.view["dsm"],np.float32)
    res=np.asarray(common.view["dsm_resolution"],np.float32).reshape(2)
    origin=np.asarray(common.view["dsm_origin_local"],np.float32).reshape(2)
    z,valid=sample_dsm_local(dsm,x,y,res,origin)
    n=min(len(img),len(x),len(y),len(z)); img,x,y,z,valid=img[:n],x[:n],y[:n],z[:n],valid[:n]
    valid=valid&np.isfinite(img).all(1)&np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    # Same PnP object convention as the supplied AnyVisLoc baseline.
    return img[valid],np.c_[x[valid],-y[valid]],z[valid]

def planar_inputs(common):
    img=np.asarray(common.uav_points_original,np.float64).reshape(-1,2)
    xy=np.c_[np.asarray(common.local_x,np.float64),-np.asarray(common.local_y,np.float64)]
    n=min(len(img),len(xy)); valid=np.isfinite(img[:n]).all(1)&np.isfinite(xy[:n]).all(1)
    return img[:n][valid],xy[:n][valid]

def restore_center(common,center_local):
    c=np.asarray(center_local,np.float64).copy()
    c[1]*=-1.0
    return c

def result_row(common,method,status,center=None,**extra):
    r={"scene_name":str(common.scene_name),"sample_id":str(common.sample_id),"npz_path":str(common.npz_path),
       "method":method,"status":status,"true_x":float(common.true_pos["x"]),"true_y":float(common.true_pos["y"]),
       "true_z":float(common.true_pos["z"]),"num_matches":int(len(common.uav_points_original))}
    if center is not None and np.isfinite(center[:2]).all():
        r["estimated_x"],r["estimated_y"]=float(center[0]),float(center[1])
        r["estimated_z"]=float(center[2]) if len(center)>2 and np.isfinite(center[2]) else None
        dx=r["estimated_x"]-r["true_x"]; dy=r["estimated_y"]-r["true_y"]
        r.update(error_x_meters=float(dx),error_y_meters=float(dy),error_meters=float(np.hypot(dx,dy)))
    else:
        r.update(estimated_x=None,estimated_y=None,estimated_z=None,error_x_meters=None,error_y_meters=None,error_meters=None)
    r.update(extra); return r

def _resize_depth_valid(d,w,h):
    d=np.asarray(d,np.float32).squeeze()
    if d.shape==(h,w): return d
    valid=np.isfinite(d)&(d>0)
    num=cv2.resize(np.where(valid,d,0).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    den=cv2.resize(valid.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    out=np.full((h,w),np.nan,np.float32); good=den>1e-6; out[good]=num[good]/den[good]
    return out

def _undistort_depth(d,K,dist):
    dist=np.asarray(dist,np.float64).reshape(-1)
    if not len(dist) or np.all(np.abs(dist)<1e-15): return d
    h,w=d.shape
    map1,map2=cv2.initUndistortRectifyMap(np.asarray(K,np.float64),dist,None,np.asarray(K,np.float64),(w,h),cv2.CV_32FC1)
    valid=np.isfinite(d)&(d>0)
    num=cv2.remap(np.where(valid,d,0).astype(np.float32),map1,map2,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    den=cv2.remap(valid.astype(np.float32),map1,map2,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    out=np.full_like(num,np.nan); good=den>1e-6; out[good]=num[good]/den[good]
    return out

class DepthSource:
    def __init__(self,root):
        self.root=Path(root) if root else None; self.h5=None
        if self.root and self.root.is_file() and self.root.suffix.lower() in (".h5",".hdf5"):
            self.h5=h5py.File(self.root,"r")
    def close(self):
        if self.h5 is not None:self.h5.close()
    def load(self,common):
        if self.root is None: raise ValueError("--depth_root is required for LDVP+SAC")
        s,q=str(common.scene_name),str(common.sample_id)
        if self.h5 is not None:
            candidates=(f"samples/{s}/{q}/depth",f"{s}/{q}/depth",f"{s}/{q}",f"{q}/depth",q)
            key=next((k for k in candidates if k in self.h5),None)
            if key is None:
                raise KeyError(f"depth H5 key not found for {s}/{q}; tried {candidates}")
            d=np.asarray(self.h5[key],np.float32)
        else:
            candidates=[self.root/s/q/"depth.npy",self.root/s/f"{q}.npy",self.root/q/"depth.npy",self.root/f"{q}.npy"]
            p=next((x for x in candidates if x.is_file()),None)
            if p is None:
                hits=list(self.root.rglob(f"{q}.npy"))
                if len(hits)!=1: raise FileNotFoundError(f"depth for {s}/{q}: {len(hits)} candidates")
                p=hits[0]
            d=np.asarray(np.load(p),np.float32).squeeze()
        h,w=common.uav_bgr.shape[:2]
        d=_resize_depth_valid(d,w,h)
        if bool(common.uav_was_undistorted):
            d=_undistort_depth(d,common.K,common.distortion)
        return d,np.asarray(common.uav_points_original,np.float64),np.asarray(common.K,np.float64)
