"""Unified pose backend: MediaPipe (py3.10/11) -> YOLOv8-pose (py3.13, CPU OK)."""
import numpy as np


def coco17_to_ntu25(coco_xy, coco_conf):
    """coco_xy (17,2) in [0,1], conf (17,) -> (25,3)."""
    out = np.zeros((25, 3), np.float32)
    # helper
    def avg(ids):
        return np.mean([coco_xy[i] for i in ids], axis=0), float(np.mean([coco_conf[i] for i in ids]))
    # 0 base spine
    out[0, :2], out[0, 2] = avg([11, 12])
    out[1, :2], out[1, 2] = avg([5, 6, 11, 12])
    out[2, :2], out[2, 2] = avg([5, 6])
    out[3, :2], out[3, 2] = coco_xy[0], coco_conf[0]
    pairs = [(4, 5), (5, 7), (6, 9), (7, 9), (8, 6), (9, 8), (10, 10), (11, 10),
             (12, 11), (13, 13), (14, 15), (15, 15),
             (16, 12), (17, 14), (18, 16), (19, 16)]
    for dst, src in pairs:
        out[dst, :2] = coco_xy[src]
        out[dst, 2] = coco_conf[src]
    out[20, :2], out[20, 2] = avg([5, 6])
    for dst, src in [(21, 9), (22, 9), (23, 10), (24, 10)]:
        out[dst, :2] = coco_xy[src]
        out[dst, 2] = coco_conf[src] * 0.9
    return out


class AutoPoseExtractor:
    """Tu dong chon backend kha dung. YOLO la mac dinh vi chay duoc py3.13 CPU."""

    def __init__(self, min_conf=0.3):
        self.backend = None
        self.min_conf = min_conf
        # 1. thu mediapipe legacy
        try:
            import mediapipe.solutions.pose  # noqa
            from tools.pose_mediapipe import PoseExtractor as MPExt
            self.mp = MPExt(min_detection_confidence=0.5)
            self.backend = 'mediapipe'
            print("[pose] backend=mediapipe")
            return
        except Exception:
            pass  # mediapipe solutions khong co tren py3.13 -> dung YOLO (khop COCO 17 voi model)
        # 2. YOLO
        from ultralytics import YOLO
        # yolov8n-pose nhe nhat, tu tai ve lan dau (~6MB)
        self.yolo = YOLO('yolov8n-pose.pt')
        self.backend = 'yolo'
        print("[pose] backend=yolov8n-pose")

    def process_bgr(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        if self.backend == 'mediapipe':
            return self.mp.process_bgr(bgr)
        # YOLO: lay nguoi dau tien conf cao nhat
        res = self.yolo.predict(bgr, verbose=False, conf=self.min_conf)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None
        kps = res.keypoints.xyn[0].cpu().numpy()  # (17,2) normalized
        conf = res.keypoints.conf[0].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)
        # YOLO xyn da [0,1]
        return coco17_to_ntu25(kps.astype(np.float32), conf.astype(np.float32))

    def process_bgr_coco(self, bgr):
        """Tra ve (17,3) x,y,conf trong [0,1]. Dung cho model mmaction COCO."""
        if self.backend == 'mediapipe':
            # mediapipe -> lay dai dien COCO tu NTU? tam dung NTU->COCO gan dung
            ntu = self.mp.process_bgr(bgr)
            if ntu is None:
                return None
            # map nguoc gan dung: nose, shoulders, elbows...
            coco = np.zeros((17, 3), np.float32)
            idx = [3, 3, 3, 3, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]
            for i, j in enumerate(idx):
                coco[i] = ntu[j]
            return coco
        res = self.yolo.predict(bgr, verbose=False, conf=self.min_conf)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None
        # chon nguoi co bbox lon nhat / conf cao nhat
        kps = res.keypoints.xyn.cpu().numpy()  # (M,17,2)
        confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kps.shape[:2])
        best = int(np.argmax(confs.mean(axis=1)))
        out = np.zeros((17, 3), np.float32)
        out[:, :2] = kps[best].astype(np.float32)
        out[:, 2] = confs[best].astype(np.float32)
        return out

    def close(self):
        try:
            if self.backend == 'mediapipe':
                self.mp.close()
        except Exception:
            pass
