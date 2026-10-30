from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
LEGACY_MODELS = WORKSPACE / "pilot_method_comparison_2026-05-21" / "models"
SRC = PROJECT / "src"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "web_app" / ".ultralytics"))
DEFAULT_TUNED = PROJECT / "results" / "external_video_model_comparison_20261028_v1" / "checkpoint" / "tug_all_frames_epoch5_best.pt"
APP_DIR = Path(__file__).resolve().parent


def model_path(env_name: str, filename: str, legacy: Path) -> Path:
    """Explicit environment override, portable app/models, then historical location."""
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser()
    portable = APP_DIR / "models" / filename
    return portable if portable.is_file() else legacy


def model_paths() -> dict[str, Path]:
    return {
        "yolo_standard": model_path("GAITLENS_YOLO_STANDARD", "yolo11n-pose.pt", LEGACY_MODELS / "yolo11n-pose.pt"),
        "mediapipe_standard": model_path("GAITLENS_MEDIAPIPE", "pose_landmarker.task", LEGACY_MODELS / "pose_landmarker.task"),
        "yolo_base": model_path("GAITLENS_YOLO_BASE", "yolo26n-pose.pt", LEGACY_MODELS / "yolo26n-pose.pt"),
        "yolo_proposed": model_path("GAITLENS_YOLO_PROPOSED", "tug_all_frames_epoch5_best.pt", DEFAULT_TUNED),
    }

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]
FOOT_NAMES = ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
MP_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
            "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]
FOOT_EDGES = COCO_EDGES + [(15, 17), (17, 18), (18, 19), (19, 15), (16, 20), (20, 21), (21, 22), (22, 16)]
MP_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (31, 27),
            (28, 30), (30, 32), (32, 28)]

# Match the 16-joint research comparison order. COCO17 has no heel or toe.
BODY12_IDS = [5, 6, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]
TUNED16_IDS = BODY12_IDS + [17, 20, 19, 22]
MP16_IDS = [11, 12, 13, 14, 15, 16, 25, 26, 23, 24, 27, 28, 31, 32, 29, 30]


def select_analysis_joints(model_key: str, names: list[str], edges: list[tuple[int, int]]):
    ids = BODY12_IDS if model_key == "yolo_standard" else MP16_IDS if model_key == "mediapipe_standard" else TUNED16_IDS
    lookup = {old: new for new, old in enumerate(ids)}
    return ids, [names[i] for i in ids], [(lookup[a], lookup[b]) for a, b in edges if a in lookup and b in lookup]


@dataclass
class AnalysisResult:
    video_path: Path
    csv_path: Path
    metrics: dict
    warnings: list[str]


def available_models() -> dict[str, dict]:
    paths = model_paths()
    return {
        "yolo_standard": {"label": "標準 YOLO11n-Pose", "ready": paths["yolo_standard"].is_file()},
        "mediapipe_standard": {"label": "標準 MediaPipe Pose", "ready": paths["mediapipe_standard"].is_file()},
        "yolo_proposed": {"label": "提案手法：追加学習 YOLO（23点）", "ready": paths["yolo_proposed"].is_file() and paths["yolo_base"].is_file()},
        "mediapipe_proposed": {"label": "提案手法：追加学習 MediaPipe（開発中）", "ready": False},
    }


def _mosaic_face(image: np.ndarray, points: np.ndarray, box: np.ndarray | None) -> None:
    h, w = image.shape[:2]
    face = points[:5] if len(points) >= 5 else np.empty((0, 3))
    valid = face[np.isfinite(face[:, :2]).all(1)] if len(face) else face
    if len(valid) >= 2:
        lo, hi = valid[:, :2].min(0), valid[:, :2].max(0)
        span = max(float((hi - lo).max()), 24.0)
        x1, y1 = lo - span * np.array([.7, .8])
        x2, y2 = hi + span * np.array([.7, 1.0])
    elif box is not None:
        x1, y1, x2, y2 = box
        y2 = y1 + (y2 - y1) * .24
    else:
        return
    x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    tiny = cv2.resize(roi, (max(1, (x2-x1)//16), max(1, (y2-y1)//16)), interpolation=cv2.INTER_LINEAR)
    image[y1:y2, x1:x2] = cv2.resize(tiny, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)


def _draw(image: np.ndarray, points: np.ndarray, edges: list[tuple[int, int]], threshold: float) -> None:
    for a, b in edges:
        if a < len(points) and b < len(points) and np.isfinite(points[[a, b], :2]).all() and np.all(points[[a, b], 2] >= threshold):
            cv2.line(image, tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int)), (44, 215, 255), 3, cv2.LINE_AA)
    for x, y, score in points:
        if np.isfinite(x) and np.isfinite(y) and score >= threshold:
            cv2.circle(image, (int(x), int(y)), 4, (10, 35, 40), -1, cv2.LINE_AA)
            cv2.circle(image, (int(x), int(y)), 3, (42, 235, 133), -1, cv2.LINE_AA)


def _select(candidates: list[tuple[np.ndarray, np.ndarray | None, float]], previous: np.ndarray | None, w: int, h: int):
    if not candidates:
        return None, None
    target = np.array([w/2, h/2]) if previous is None else previous
    def center(item):
        pts, box, _ = item
        if box is not None:
            return (box[:2] + box[2:]) / 2
        valid = pts[np.isfinite(pts[:, :2]).all(1)]
        return valid[:, :2].mean(0) if len(valid) else np.array([math.inf, math.inf])
    selected = min(candidates, key=lambda item: np.linalg.norm(center(item) - target))
    return selected, center(selected)


def _yolo_runner(kind: str, device: str):
    import torch
    if kind == "yolo_standard":
        from ultralytics import YOLO
        model = YOLO(str(model_paths()["yolo_standard"]))
        def infer(frame):
            result = model.predict(frame, device=device, imgsz=640, conf=.25, iou=.7, verbose=False)[0]
            found = []
            if result.keypoints is not None and result.keypoints.conf is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                for xy, score, box, conf in zip(result.keypoints.xy.cpu().numpy(), result.keypoints.conf.cpu().numpy(), boxes, confs):
                    found.append((np.c_[xy, score], box, float(conf)))
            return found
        return infer, COCO_NAMES, COCO_EDGES
    from foot23_models import extend_model
    from ultralytics.utils.nms import non_max_suppression
    paths = model_paths()
    model = extend_model(paths["yolo_base"], device)
    checkpoint = torch.load(paths["yolo_proposed"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    def infer(frame):
        h, w = frame.shape[:2]
        ratio = 640 / max(w, h); nw, nh = round(w*ratio), round(h*ratio)
        left, top = (640-nw)//2, (640-nh)//2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
        tensor = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).copy()).float().to(device)[None] / 255
        with torch.inference_mode():
            raw = model(tensor)[0]
            output = non_max_suppression(raw, .25, .45, nc=1)[0]
        found = []
        for row in output:
            box = row[:4].detach().cpu().numpy(); box[[0,2]]=(box[[0,2]]-left)/ratio; box[[1,3]]=(box[[1,3]]-top)/ratio
            pts = row[6:].reshape(23, 3).detach().cpu().numpy(); pts[:, :2]=(pts[:, :2]-[left,top])/ratio
            found.append((pts, box, float(row[4].item())))
        return found
    return infer, COCO_NAMES + FOOT_NAMES, FOOT_EDGES


def _mediapipe_runner():
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_paths()["mediapipe_standard"]), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=4,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5)
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    def infer(frame, timestamp_ms):
        h, w = frame.shape[:2]
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp_ms)
        found=[]
        for pose in result.pose_landmarks:
            pts=np.array([[p.x*w,p.y*h,min(p.visibility,p.presence)] for p in pose],np.float32)
            valid=pts[pts[:,2]>=.5,:2]
            box=np.r_[valid.min(0),valid.max(0)] if len(valid)>=4 else None
            found.append((pts,box,float(np.nanmean(pts[:,2]))))
        return found
    return infer, landmarker


def _quality(all_points: np.ndarray, threshold: float, fps: float, elapsed: float,
             edges: list[tuple[int, int]]) -> dict:
    scores=all_points[...,2]; valid=np.isfinite(all_points[...,:2]).all(2)&(scores>=threshold)
    available=float(valid.mean()) if valid.size else 0.0
    confidence=float(np.nanmean(np.where(valid,scores,np.nan))) if valid.any() else 0.0
    centers=[]; scales=[]
    for pts,ok in zip(all_points,valid):
        q=pts[ok,:2]
        centers.append(np.nanmean(q,0) if len(q) else [np.nan,np.nan])
        scales.append(np.linalg.norm(np.nanmax(q,0)-np.nanmin(q,0)) if len(q)>=2 else np.nan)
    centers=np.asarray(centers); scales=np.asarray(scales)
    accel=np.linalg.norm(np.diff(centers,n=2,axis=0),axis=1) if len(centers)>=3 else np.array([])
    denom=np.nanmedian(scales)
    jitter=float(np.nanmedian(accel)/denom) if len(accel) and denom>0 else np.nan
    temporal=float(100*np.exp(-8*jitter)) if np.isfinite(jitter) else 0.0
    lengths=[]
    # Use the model's own limb graph; exclude short face/foot loops from this check.
    for a,b in edges:
        if max(a,b)>=all_points.shape[1]: continue
        x=np.linalg.norm(all_points[:,a,:2]-all_points[:,b,:2],axis=1); ok=valid[:,a]&valid[:,b]
        if ok.sum()>=3 and np.mean(x[ok])>0: lengths.append(np.std(x[ok])/np.mean(x[ok]))
    bone=float(100*(1-min(float(np.mean(lengths)),1))) if lengths else 0.0
    quality=100*(.4*available+.3*confidence)+.15*temporal+.15*bone
    missing_by_joint={str(i):int((~valid[:,i]).sum()) for i in range(valid.shape[1])}
    return {"frames":int(len(all_points)),"duration_seconds":round(len(all_points)/fps,2),"processing_seconds":round(elapsed,2),
            "processing_fps":round(len(all_points)/elapsed,2) if elapsed else None,"available_percent":round(available*100,1),
            "missing_points":int((~valid).sum()),"total_points":int(valid.size),"mean_confidence":round(confidence,3),
            "temporal_stability":round(temporal,1),"bone_length_consistency":round(bone,1),
            "reference_free_quality":round(float(np.clip(quality,0,100)),1),
            "missing_by_joint":missing_by_joint}


def analyze_video(input_path: Path, output_dir: Path, model_key: str, threshold: float=.5,
                  device: str="cpu", progress: Callable[[float, str], None] | None=None) -> AnalysisResult:
    if model_key == "mediapipe_proposed": raise ValueError("追加学習MediaPipeは開発中で、まだ解析には使用できません。")
    output_dir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): raise ValueError("動画を開けませんでした。MP4またはMOVで再度お試しください。")
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out_path=output_dir/"skeleton_mosaic.mp4"
    writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    mp_landmarker=None
    if model_key == "mediapipe_standard":
        infer,mp_landmarker=_mediapipe_runner(); raw_names,raw_edges=MP_NAMES,MP_EDGES
    else: infer,raw_names,raw_edges=_yolo_runner(model_key,device)
    joint_ids,names,edges=select_analysis_joints(model_key,raw_names,raw_edges)
    all_points=[]; previous=None; warnings=[]; start=time.perf_counter(); frame_i=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            candidates=infer(frame,round(frame_i/fps*1000)) if model_key=="mediapipe_standard" else infer(frame)
            selected,previous=_select(candidates,previous,w,h)
            raw_pts=np.full((len(raw_names),3),np.nan,np.float32); box=None
            if selected is not None: raw_pts,box,_=selected
            _mosaic_face(frame,raw_pts,box)
            pts=raw_pts[joint_ids]
            all_points.append(pts.copy()); _draw(frame,pts,edges,threshold); writer.write(frame)
            frame_i+=1
            if progress and (frame_i%10==0 or frame_i==total): progress(min(frame_i/max(total,1),1),f"{frame_i:,} / {total:,} フレーム")
    finally:
        cap.release();writer.release()
        if mp_landmarker is not None: mp_landmarker.close()
    if not all_points: raise ValueError("解析できるフレームがありませんでした。")
    # OpenCV's mp4v is not consistently playable in Safari/Streamlit. Prefer an
    # H.264 copy when FFmpeg is available, while retaining mp4v as a fallback.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_path = output_dir / "skeleton_mosaic_h264.mp4"
        converted = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264_path)],
            capture_output=True, text=True)
        if converted.returncode == 0 and h264_path.is_file():
            out_path = h264_path
        else:
            warnings.append("H.264変換に失敗したため、互換性の低いMP4を返しました。")
    points=np.stack(all_points); metrics=_quality(points,threshold,fps,time.perf_counter()-start,edges)
    metrics["missing_by_joint"] = {name: metrics["missing_by_joint"][str(i)] for i, name in enumerate(names)}
    metrics["joints_with_missing"] = sum(count > 0 for count in metrics["missing_by_joint"].values())
    metrics["joint_count"] = len(names)
    csv_path=output_dir/"coordinates.csv"
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        writer_csv=csv.writer(f);writer_csv.writerow(["frame","time_seconds","joint","x_px","y_px","confidence","available"])
        for fi,frame in enumerate(points):
            for name,p in zip(names,frame): writer_csv.writerow([fi,round(fi/fps,4),name,*["" if not np.isfinite(v) else round(float(v),4) for v in p],bool(np.isfinite(p[:2]).all() and p[2]>=threshold)])
    if metrics["available_percent"]<80: warnings.append("出力率が80%未満です。全身が画面内に入り、遮蔽が少ない動画を推奨します。")
    warnings.append("信頼度や時間・骨格の指標は正解率ではありません。人手GTや計測機器との誤差を表しません。")
    return AnalysisResult(out_path,csv_path,metrics,warnings)


