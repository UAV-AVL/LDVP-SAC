#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

from bootstrap import configure

METHODS=("DSM-PnP","Planar-PnP","LDVP","LDVP+SAC")
THRESHOLDS=(1.,2.,3.,5.)

def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def save_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(jsonable(x),indent=2,ensure_ascii=False),encoding="utf-8")
def save_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    if not rows:p.write_text("",encoding="utf-8");return
    keys=[];seen=set()
    for r in rows:
        for k in r:
            if k not in seen:seen.add(k);keys.append(k)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:jsonable(r.get(k)) for k in keys})
def ok(r):
    try:return str(r.get("status","")).lower()=="success" and np.isfinite(float(r.get("error_meters")))
    except:return False
def summarize(rows,methods,total):
    out=[]
    for m in methods:
        e=np.asarray([float(r["error_meters"]) for r in rows if r.get("method")==m and ok(r)],float)
        d={"method":m,"num_total":int(total),"num_success":int(len(e)),"success_rate":float(len(e)/total) if total else 0.,
           "mean_error_m":float(np.mean(e)) if len(e) else None,"median_error_m":float(np.median(e)) if len(e) else None,
           "p90_error_m":float(np.percentile(e,90)) if len(e) else None}
        for t in THRESHOLDS:d[f"T@{t:g}"]=float(np.sum(e<=t)/total) if total else 0.
        out.append(d)
    return out
def print_summary(rows,methods,total):
    print("\n"+"="*104);print(f"Summary over {total} attempted queries; failures count as misses")
    print(f"{'Method':16s} {'T@1':>8s} {'T@2':>8s} {'T@3':>8s} {'T@5':>8s} {'Succ':>8s} {'Median':>9s} {'P90':>9s}")
    for d in summarize(rows,methods,total):
        med="-" if d["median_error_m"] is None else f'{d["median_error_m"]:.3f}'
        p90="-" if d["p90_error_m"] is None else f'{d["p90_error_m"]:.3f}'
        print(f'{d["method"]:16s} {100*d["T@1"]:7.2f}% {100*d["T@2"]:7.2f}% {100*d["T@3"]:7.2f}% {100*d["T@5"]:7.2f}% {100*d["success_rate"]:7.2f}% {med:>9s} {p90:>9s}')
    print("="*104)

def parse_methods(v):
    if not v or v==["all"]:return list(METHODS)
    out=[]
    for x in v:
        for q in x.split(","):
            q=q.strip()
            aliases={"dsm":"DSM-PnP","planar":"Planar-PnP","ippe":"Planar-PnP","ldvp":"LDVP","sac":"LDVP+SAC","ldvp+sac":"LDVP+SAC"}
            q=aliases.get(q.lower(),q)
            if q not in METHODS:raise ValueError(f"Unknown method {q}")
            if q not in out:out.append(q)
    return out

def parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",choices=["anyvisloc","ortholoc"],required=True)
    p.add_argument("--dataset_root",required=True)
    p.add_argument("--methods",nargs="*",default=["all"])
    p.add_argument("--save_dir",default="./Result/LDVP_SAC_release")
    p.add_argument("--run_tag",default=None)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--print_every",type=int,default=100,help="Print interim metrics every N attempted queries; 0 disables.")
    p.add_argument("--save_every",type=int,default=100,help="Atomically flush the single run_state.json every N attempted queries; 0 disables.")
    p.add_argument("--resume",action="store_true",help="Resume from the single run_state.json in this run directory.")
    p.add_argument("--sample_ratio",type=float,default=1.0)
    p.add_argument("--matching_models_root",default=None)
    p.add_argument("--yaml",default=str(Path(__file__).resolve().parent/"configs/config_selectable_matchers.yaml"))
    p.add_argument("--matching_method",default="Roma");p.add_argument("--device",default="cuda")
    p.add_argument("--match_cache_root",default=None);p.add_argument("--cache_only",action="store_true");p.add_argument("--force_rematch",action="store_true")
    p.add_argument("--show_match_image",action="store_true");p.add_argument("--match_keypoints",type=int,default=3000)
    p.add_argument("--selectable_code_dir",default=None);p.add_argument("--selectable_module",default="selectable_sparse_matcher")
    p.add_argument("--gim_lg_ckpt",default=None);p.add_argument("--minima_lg_ckpt",default=None)
    # AnyVisLoc
    p.add_argument("--reference_mode",choices=["aerial","satellite"],default="aerial");p.add_argument("--scenes",nargs="*",default=None)
    p.add_argument("--max_samples_per_scene",type=int,default=0);p.add_argument("--pose_priori",choices=["yp","p"],default="yp")
    p.add_argument("--resize_ratio",type=float,default=.4);p.add_argument("--patch_scale",type=float,default=1.)
    p.add_argument("--pre_crop_scale",type=float,default=2.);p.add_argument("--min_patch_size_m",type=float,default=0.)
    p.add_argument("--max_patch_size_m",type=float,default=0.);p.add_argument("--max_patch_pixels",type=int,default=0)
    p.add_argument("--disable_uav_undistortion",action="store_true")
    # OrthoLoC
    p.add_argument("--set_name",default="all",choices=["all","train","val","test_inPlace","test_outPlace"])
    p.add_argument("--sample_glob",default=None);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--use_refined_extrinsics",action="store_true");p.add_argument("--rotate_ref_by_gt_yaw",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--allow_reconstruct_gt_yaw_rotation",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--copy_match_cache_to_run",action="store_true")
    # PnP
    p.add_argument("--pnp_iterations",type=int,default=1000);p.add_argument("--pnp_reproj_error",type=float,default=2.)
    p.add_argument("--pnp_min_inliers",type=int,default=4);p.add_argument("--pnp_confidence",type=float,default=.99)
    # LDVP defaults aligned with the final paper experiments for both datasets
    p.add_argument("--depth_min_matches",type=int,default=8);p.add_argument("--depth_robust_f_scale",type=float,default=.5)
    p.add_argument("--min_positive_depth_ratio_final",type=float,default=.80)
    p.add_argument("--globalrot_tilt_step_deg",type=float,default=15.);p.add_argument("--globalrot_azimuth_step_deg",type=float,default=30.)
    p.add_argument("--globalrot_roll_step_deg",type=float,default=30.);p.add_argument("--globalrot_refine_levels",type=int,default=3)
    p.add_argument("--globalrot_refine_factor",type=float,default=3.);p.add_argument("--globalrot_grid_irls",type=int,default=2)
    p.add_argument("--globalrot_profile_irls",type=int,default=3);p.add_argument("--globalrot_min_positive_ratio_init",type=float,default=.60)
    p.add_argument("--globalrot_max_points",type=int,default=0);p.add_argument("--globalrot_batch_size",type=int,default=256)
    p.add_argument("--globalrot_fast_precompute_max_mb",type=float,default=192.);p.add_argument("--globalrot_local_max_nfev",type=int,default=100)
    p.add_argument("--globalrot_msac_sample_size",type=int,default=5);p.add_argument("--globalrot_msac_iterations",type=int,default=256)
    p.add_argument("--globalrot_msac_min_iterations",type=int,default=128);p.add_argument("--globalrot_msac_confidence",type=float,default=.999)
    p.add_argument("--globalrot_msac_threshold_m",type=float,default=2.)
    # final SAC
    p.add_argument("--depth_root",default=None);p.add_argument("--s1_threshold",type=float,default=.5)
    p.add_argument("--plane_threshold",type=float,default=.003);p.add_argument("--plane_ransac_iterations",type=int,default=2000)
    p.add_argument("--mde_min_depth_matches",type=int,default=3, help="Compatibility option; final Relative-Depth S1 requires exactly a minimum of 3 valid matched 3D points.")
    return p

def run_methods(common,adapter,opt,depth_source,seed):
    from methods import pnp,ldvp,sac
    wanted=set(opt.methods); rows=[]; cache={}
    dist=adapter.pnp_distortion(common)
    pkw=dict(iterations=opt.pnp_iterations,reprojection_threshold=opt.pnp_reproj_error,min_inliers=opt.pnp_min_inliers)

    def planar():
        if "Planar-PnP" in cache:return cache["Planar-PnP"]
        vals=adapter.planar_inputs(common)
        if opt.dataset=="ortholoc":img,xy,origin=vals
        else:img,xy=vals;origin=None
        raw=pnp.run_planar_pnp(img,xy,common.K,dist,seed=seed+17,**pkw)
        center=None if raw.center_xyz_local is None else (adapter.restore_center(common,raw.center_xyz_local,origin) if opt.dataset=="ortholoc" else adapter.restore_center(common,raw.center_xyz_local))
        row=adapter.result_row(common,"Planar-PnP",raw.status,center,num_inliers=int(len(raw.inliers)),mean_inlier_reprojection_error_px=raw.reprojection_rmse_px,pnp_solver="OpenCV SOLVEPNP_IPPE",planar_scene_assumption=True)
        cache["Planar-PnP"]=row;return row

    def free_ldvp():
        if "LDVP" in cache:return cache["LDVP"]
        vals=adapter.normalized_inputs(common)
        if opt.dataset=="ortholoc":img,xy,rays,idx,origin=vals
        else:img,xy,rays,idx=vals;origin=None
        raw=ldvp.solve_ldvp_free(rays,xy,opt,random_seed=seed+41)
        center=None
        if str(raw.get("status","")).lower()=="success":
            c=np.asarray(raw["center_xy_local"],float)
            if opt.dataset=="ortholoc": center=np.array([c[0]+origin[0],c[1]+origin[1],np.nan])
            else: center=np.array([c[0],-c[1],np.nan])
        row=adapter.result_row(common,"LDVP",str(raw.get("status","failed")),center,**{k:v for k,v in raw.items() if k not in ("status","center_xy_local")})
        cache["LDVP"]=row;return row

    if "DSM-PnP" in wanted:
        vals=adapter.dsm_inputs(common)
        if opt.dataset=="ortholoc":img,xy,z,origin=vals
        else:img,xy,z=vals;origin=None
        raw=pnp.run_dsm_pnp(img,xy,z,common.K,dist,seed=seed+11,**pkw)
        center=None if raw.center_xyz_local is None else (adapter.restore_center(common,raw.center_xyz_local,origin) if opt.dataset=="ortholoc" else adapter.restore_center(common,raw.center_xyz_local))
        rows.append(adapter.result_row(common,"DSM-PnP",raw.status,center,num_inliers=int(len(raw.inliers)),mean_inlier_reprojection_error_px=raw.reprojection_rmse_px,pnp_solver="OpenCV SOLVEPNP_P3P"))

    if "Planar-PnP" in wanted: rows.append(planar())
    if "LDVP" in wanted: rows.append(free_ldvp())

    if "LDVP+SAC" in wanted:
        try:
            depth,uv_sac,K_sac=depth_source.load(common)
            scene_or_split=str(common.scene_name) if opt.dataset=="anyvisloc" else str(common.split_name)
            sr=sac.compute_s1(
                depth,uv_sac,K_sac,
                matcher=str(opt.matching_method),
                scene_or_split=scene_or_split,
                sample_id=str(common.sample_id),
                iterations=opt.plane_ransac_iterations,
                plane_threshold=opt.plane_threshold,
            )
            branch=sac.choose_branch(sr["S1"],opt.s1_threshold)
            source=planar() if branch=="Planar-PnP" else free_ldvp()
            row=dict(source);row["method"]="LDVP+SAC";row["selected_source"]=branch
            row.update(S1=sr["S1"],N1=sr["N1"],Nmatch=sr["Nmatch"],
                       matched_depth_median=sr["median_matched_depth"],
                       sac_status=sr["analysis_status"],
                       sac_plane_threshold=sr["plane1_threshold"],
                       sac_plane_threshold_units=sr["plane1_threshold_units"],
                       sac_depth_normalization="matched_3d / median(valid_matched_depth)",
                       s1_threshold=float(opt.s1_threshold))
        except Exception as e:
            row=adapter.result_row(
                common,
                "LDVP+SAC",
                "sac_failed",
                None,
                selected_source=None,
                S1=None,
                N1=0,
                Nmatch=0,
                sac_status="failed",
                sac_failure_reason=f"{type(e).__name__}: {e}",
                sac_plane_threshold=float(opt.plane_threshold),
                sac_plane_threshold_units="dimensionless_after_median_depth_normalization",
                sac_depth_normalization="matched_3d / median(valid_matched_depth)",
                s1_threshold=float(opt.s1_threshold),
            )
        rows.append(row)
    return rows

def _state_path(root):
    return Path(root)/"run_state.json"

def _atomic_save_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(jsonable(payload),indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

def _query_key(dataset, scene, sid):
    return f"{dataset}|{scene}|{sid}"

def _load_state(root):
    p=_state_path(root)
    if not p.is_file():
        return None
    data=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or "queries" not in data:
        raise ValueError(f"Invalid resume state: {p}")
    return data

def _flatten_rows(state):
    out=[]
    for q in state.get("queries",{}).values():
        out.extend(q.get("results",[]))
    return out

def _state_summary(state,methods):
    return summarize(_flatten_rows(state),methods,int(state.get("attempted",0)))

def _flush_state(root,state,methods):
    state["summary"]=_state_summary(state,methods)
    _atomic_save_json(_state_path(root),state)

def main():
    opt=parser().parse_args();opt.methods=parse_methods(opt.methods)
    configure(opt.matching_models_root)
    # Import dataset code only after bootstrap configured Matching_Models.
    if opt.dataset=="anyvisloc":
        from datasets import anyvisloc as adapter
        adapter.verify_reference_mode(opt.match_cache_root,opt.reference_mode)
        adapter.set_random_seed(opt.seed)
        ds,items,sampling=adapter.dataset_and_indices(opt); matcher=adapter.MatcherManager(opt)
    else:
        from datasets import ortholoc as adapter
        items=adapter.query_list(opt);sampling={"num_selected":len(items)}
        cache_index=adapter.build_match_cache_index(opt.match_cache_root);matcher=adapter.LazyMatcher(opt)

    if opt.resume and not opt.run_tag:
        raise ValueError("--resume requires --run_tag so the previous run directory is unambiguous")
    tag=opt.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    root=Path(opt.save_dir)/tag/f"{opt.dataset}-{opt.matching_method}";root.mkdir(parents=True,exist_ok=True)
    state=_load_state(root) if opt.resume else None
    if state is None:
        state={
            "format_version":1,
            "dataset":opt.dataset,
            "matching_method":opt.matching_method,
            "methods":list(opt.methods),
            "args":jsonable(vars(opt)),
            "sampling":jsonable(sampling),
            "sac_definition":{"S1":"N1/N_valid_matched_3d","plane_threshold":opt.plane_threshold,
                              "plane_ransac_iterations":opt.plane_ransac_iterations,
                              "selector":"Planar-PnP iff S1 >= s1_threshold; else LDVP"},
            "attempted":0,
            "queries":{},
            "failures":[],
            "summary":[],
        }
    else:
        if state.get("dataset")!=opt.dataset:
            raise ValueError(f"Resume dataset mismatch: state={state.get('dataset')} cli={opt.dataset}")
        if list(state.get("methods",[]))!=list(opt.methods):
            raise ValueError(f"Resume methods mismatch: state={state.get('methods')} cli={opt.methods}")
    depth_source=adapter.DepthSource(opt.depth_root) if "LDVP+SAC" in opt.methods else None
    rows=_flatten_rows(state);fail=state["failures"];previous_scene=None;attempted=int(state.get("attempted",0))
    bar=tqdm(items,desc=f"{opt.dataset} 4-method benchmark",unit="query",ncols=120)
    newly_processed=0
    for j,item in enumerate(bar):
        scene=sid=None
        try:
            if opt.dataset=="anyvisloc":
                sample=ds[int(item)]
                scene=adapter.to_python_str(sample.get("scene_name",""))
                sid=adapter.to_python_str(sample.get("sample_id",""))
                if previous_scene is not None and scene!=previous_scene:
                    adapter.clear_anyvisloc_reference_array_cache()
                previous_scene=scene
            else:
                split,npz_path=item
                scene=split;sid=Path(npz_path).stem

            key=_query_key(opt.dataset,scene,sid)
            oldq=state["queries"].get(key)
            if opt.resume and isinstance(oldq,dict) and oldq.get("complete") is True:
                bar.set_postfix_str(f"{scene}/{sid} [resume]")
                continue

            # This query is newly attempted in this run.
            attempted += 1; newly_processed += 1
            if opt.dataset=="anyvisloc":
                common=adapter.prepare(sample,root,opt,matcher)
            else:
                sample,common=adapter.prepare(split,npz_path,root,opt,cache_index,matcher)
                sid=str(common.sample_id)
                key=_query_key(opt.dataset,scene,sid)

            bar.set_postfix_str(f"{scene}/{sid}")
            rr=run_methods(common,adapter,opt,depth_source,opt.seed+j*1009)
            state["queries"][key]={
                "scene_name":str(scene),
                "sample_id":str(sid),
                "complete":True,
                "results":jsonable(rr),
            }
        except Exception as e:
            attempted += 0 if scene is not None and sid is not None and newly_processed>0 else 1
            rec={"item":str(item),"scene_name":scene,"sample_id":sid,
                 "error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}
            fail.append(rec)
            if scene is not None and sid is not None:
                key=_query_key(opt.dataset,scene,sid)
                state["queries"][key]={
                    "scene_name":str(scene),"sample_id":str(sid),
                    "complete":True,"results":[],"failure":rec,
                }

        state["attempted"]=attempted
        if opt.print_every>0 and newly_processed>0 and newly_processed%opt.print_every==0:
            rows=_flatten_rows(state)
            print_summary(rows,opt.methods,attempted)
        if opt.save_every>0 and newly_processed>0 and newly_processed%opt.save_every==0:
            _flush_state(root,state,opt.methods)

    if depth_source is not None:depth_source.close()
    state["attempted"]=attempted
    _flush_state(root,state,opt.methods)
    rows=_flatten_rows(state)
    print_summary(rows,opt.methods,attempted)
    print("Saved single resumable result:",_state_path(root))

if __name__=="__main__":main()
