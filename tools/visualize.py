import cv2
import numpy as np
from net.graph import NTU_EDGES


def draw_skeleton(frame, skel, color=(0, 255, 0)):
    """skel (25,3) x,y normalized [0,1] + conf. Vẽ đè lên frame."""
    h, w = frame.shape[:2]
    pts = []
    for x, y, c in skel:
        px, py = int(np.clip(x, 0, 1) * w), int(np.clip(y, 0, 1) * h)
        pts.append((px, py, c))
    for a, b in NTU_EDGES:
        if pts[a][2] < 0.2 or pts[b][2] < 0.2:
            continue
        cv2.line(frame, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)
    for px, py, c in pts:
        if c < 0.2:
            continue
        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
    return frame


def draw_skeleton_coco(frame, skel, color=(0, 255, 0)):
    """skel (17,3) COCO: 0 nose,5 L-sh,6 R-sh,7 L-elb,8 R-elb,9 L-wr,10 R-wr,11 L-hip,12 R-hip,13 L-knee,14 R-knee,15 L-ank,16 R-ank."""
    COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
                  (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15),
                  (12, 14), (14, 16), (0, 5), (0, 6), (11, 12)]
    h, w = frame.shape[:2]
    pts = []
    for x, y, c in skel:
        pts.append((int(np.clip(x, 0, 1) * w), int(np.clip(y, 0, 1) * h), c))
    for a, b in COCO_EDGES:
        if pts[a][2] < 0.2 or pts[b][2] < 0.2:
            continue
        cv2.line(frame, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)
    for px, py, c in pts:
        if c < 0.2:
            continue
        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
    return frame


def draw_label_bar(frame, topk, labels):
    # nền đen trên cùng
    h, w = frame.shape[:2]
    bar_h = 30 + 24 * len(topk)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 24
    for i, (cls, prob) in enumerate(topk):
        name = labels[cls] if cls < len(labels) else f"class {cls}"
        txt = f"{'->' if i == 0 else '  '} {name} ({prob*100:.1f}%)"
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if i == 0 else (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return frame
