import cv2
import numpy as np
from tqdm import tqdm
from tools.pose_mediapipe import PoseExtractor


def read_frames(video_path, target_fps=30, max_side=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps))) if src_fps > target_fps else 1
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = f.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1.0:
                f = cv2.resize(f, (int(w * s), int(h * s)))
            frames.append(f)
        idx += 1
    cap.release()
    return frames, target_fps


def extract_skeleton_sequence(frames, min_det=0.5):
    from tools.pose import AutoPoseExtractor
    ex = AutoPoseExtractor(min_conf=0.3)
    seq = []
    try:
        for f in tqdm(frames, desc="Pose"):
            skel = ex.process_bgr(f)  # (25,3) x,y,conf trong [0,1]
            if skel is None:
                skel = np.zeros((25, 3), np.float32)
            seq.append(skel)
    finally:
        ex.close()
    return np.stack(seq, axis=0)  # (T,25,3)


def extract_coco_sequence(frames, min_conf=0.3):
    """Tra ve (T,17,3) COCO truc tiep tu YOLO. Dung cho model mmaction."""
    from tools.pose import AutoPoseExtractor
    ex = AutoPoseExtractor(min_conf=min_conf)
    seq = []
    try:
        for f in tqdm(frames, desc="Pose-COCO"):
            c = ex.process_bgr_coco(f)
            if c is None:
                c = np.zeros((17, 3), np.float32)
            seq.append(c)
    finally:
        ex.close()
    return np.stack(seq, axis=0)


def normalize_coco(seq):
    """Chuan hoa EXACT theo mmaction PreNormalize2D (khong center theo hong!).

    x' = (x_px - w/2)/(w/2), y' = (y_px - h/2)/(h/2).
    Voi toa do [0,1] full-frame tu YOLO: x' = 2x-1, y' = 2y-1, giu nguyen conf.
    (Ban cu center theo hong + scale torso -> lech phan phoi train -> doan bua.)
    """
    out = seq.copy()
    out[:, :, 0] = seq[:, :, 0] * 2.0 - 1.0
    out[:, :, 1] = seq[:, :, 1] * 2.0 - 1.0
    return out


def to_mmaction_input(seq, T=100, num_person=2):
    """seq (t,17,3) -> (1,M,T,17,3). Uniform sample 100 frames."""
    t = seq.shape[0]
    idxs = np.linspace(0, max(t - 1, 0), T).astype(int) if t > 0 else np.zeros(T, int)
    sampled = seq[idxs] if t > 0 else np.zeros((T, 17, 3), np.float32)
    data = np.zeros((1, num_person, T, 17, 3), np.float32)
    data[0, 0] = sampled
    # person 2 = zeros (1 nguoi)
    return data


# pairs chuan mmaction JointToBone dataset='coco': bone[v1] = kp[v1] - kp[v2],
# score = trung binh 2 dau. Tinh SAU PreNormalize2D.
COCO_BONE_PAIRS = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0),
                   (7, 5), (8, 6), (9, 7), (10, 8), (11, 0), (12, 0),
                   (13, 11), (14, 12), (15, 13), (16, 14))


def to_bone(seq_norm):
    """seq (T,17,3) da prenormalize -> bone feats (T,17,3) chuan mmaction."""
    out = np.zeros_like(seq_norm)
    for v1, v2 in COCO_BONE_PAIRS:
        out[:, v1, :2] = seq_norm[:, v1, :2] - seq_norm[:, v2, :2]
        out[:, v1, 2] = (seq_norm[:, v1, 2] + seq_norm[:, v2, 2]) / 2
    return out


def normalize_center(seq):
    """Center theo hip-mid (joint 0), scale theo torso. seq (T,25,3) x,y,conf."""
    out = seq.copy()
    for t in range(len(out)):
        xy = out[t, :, :2]
        if (out[t, :, 2] == 0).all():
            continue
        center = (xy[12] + xy[16]) / 2 if (out[t, 12, 2] > 0 and out[t, 16, 2] > 0) else xy[0]
        xy = xy - center
        torso = np.linalg.norm(out[t, 2, :2] - center) + 1e-6
        # out[t,2] lúc này đã trừ center nên tính lại: dùng neck-center
        neck = xy[2]
        torso = float(np.linalg.norm(neck) + 1e-6)
        xy = xy / (torso * 2.0)  # scale về ~[-1,1]
        out[t, :, :2] = xy
    return out


def to_stgcn_input(seq, T=300):
    """seq (t,25,3) -> (1,3,T,25,1). Uniform sample / pad repeat."""
    t = seq.shape[0]
    idxs = np.linspace(0, max(t - 1, 0), T).astype(int) if t > 0 else np.zeros(T, int)
    sampled = seq[idxs] if t > 0 else np.zeros((T, 25, 3), np.float32)
    # interpolate frame mất (conf=0) bằng frame gần nhất
    for j in range(25):
        mask = sampled[:, j, 2] == 0
        if mask.all():
            continue
        for c in range(2):
            v = sampled[:, j, c]
            good = ~mask
            v[mask] = np.interp(np.where(mask)[0], np.where(good)[0], v[good])
            sampled[:, j, c] = v
    data = np.zeros((1, 3, T, 25, 1), np.float32)
    data[0, 0, :, :, 0] = sampled[:, :, 0]
    data[0, 1, :, :, 0] = sampled[:, :, 1]
    data[0, 2, :, :, 0] = sampled[:, :, 2]
    return data
