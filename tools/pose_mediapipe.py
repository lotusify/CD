"""MediaPipe -> NTU25 mapping."""
import numpy as np

# MediaPipe indices
MP = dict(nose=0, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14,
          l_wrist=15, r_wrist=16, l_pinky=17, r_pinky=18, l_index=19,
          r_index=20, l_thumb=21, r_thumb=22, l_hip=23, r_hip=24,
          l_knee=25, r_knee=26, l_ankle=27, r_ankle=28, l_heel=29,
          r_heel=30, l_foot=31, r_foot=32)


def _avg(pts, ids):
    pts = np.asarray(pts)
    return np.mean([pts[i] for i in ids], axis=0)


def mediapipe_to_ntu25(mp_lm):
    """mp_lm: (33,4) x,y,z,visibility hoặc (33,3). Return (25,3) x,y,conf."""
    mp_lm = np.asarray(mp_lm)
    xy = mp_lm[:, :2].astype(np.float32)
    vis = mp_lm[:, 3].astype(np.float32) if mp_lm.shape[1] >= 4 else np.ones(33, np.float32)
    out_xy = np.zeros((25, 2), np.float32)
    out_c = np.zeros(25, np.float32)

    def get(i):
        return xy[i], vis[i]

    def set_avg(dst, ids):
        nonlocal out_xy, out_c
        out_xy[dst] = np.mean([xy[i] for i in ids], axis=0)
        out_c[dst] = float(np.mean([vis[i] for i in ids]))

    def set_one(dst, src):
        out_xy[dst] = xy[src]
        out_c[dst] = float(vis[src])

    set_avg(0, [MP['l_hip'], MP['r_hip']])          # 0 spine base
    set_avg(1, [MP['l_shoulder'], MP['r_shoulder'], MP['l_hip'], MP['r_hip']])  # 1 spine mid
    set_avg(2, [MP['l_shoulder'], MP['r_shoulder']])  # 2 neck
    set_one(3, MP['nose'])                          # 3 head
    set_one(4, MP['l_shoulder'])
    set_one(5, MP['l_elbow'])
    set_one(6, MP['l_wrist'])
    set_one(7, MP['l_index'])                        # 7 hand left
    set_one(8, MP['r_shoulder'])
    set_one(9, MP['r_elbow'])
    set_one(10, MP['r_wrist'])
    set_one(11, MP['r_index'])
    set_one(12, MP['l_hip'])
    set_one(13, MP['l_knee'])
    set_one(14, MP['l_ankle'])
    set_one(15, MP['l_foot'])
    set_one(16, MP['r_hip'])
    set_one(17, MP['r_knee'])
    set_one(18, MP['r_ankle'])
    set_one(19, MP['r_foot'])
    set_avg(20, [MP['l_shoulder'], MP['r_shoulder']])  # 20 spine-shoulder
    set_one(21, MP['l_pinky'])                        # tip approx
    set_one(22, MP['l_thumb'])
    set_one(23, MP['r_pinky'])
    set_one(24, MP['r_thumb'])
    return np.concatenate([out_xy, out_c[:, None]], axis=1)  # (25,3)


class PoseExtractor:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5)

    def process_bgr(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        arr = np.zeros((33, 4), np.float32)
        for i, lm in enumerate(res.pose_landmarks.landmark):
            arr[i] = [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.9)]
        return mediapipe_to_ntu25(arr)

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass
