"""Test pipeline không cần weights/video thật: sinh skeleton giả + vẽ."""
import numpy as np, cv2
from tools.preprocess import normalize_center, to_stgcn_input
from tools.visualize import draw_skeleton, draw_label_bar

T = 60
seq = np.zeros((T, 25, 3), np.float32)
for t in range(T):
    # giả lập tay vẫy: joint 6,7 di chuyển sin
    seq[t, :, 0] = 0.5
    seq[t, :, 1] = 0.4
    seq[t, :, 2] = 0.9
    seq[t, 6, 0] = 0.5 + 0.1 * np.sin(t * 0.3)
    seq[t, 7, 0] = 0.5 + 0.15 * np.sin(t * 0.3)

norm = normalize_center(seq)
data = to_stgcn_input(seq, T=300)
print("to_stgcn_input:", data.shape, data.dtype)  # expect (1,3,300,25,1)
assert data.shape == (1, 3, 300, 25, 1)

frame = np.ones((480, 640, 3), np.uint8) * 255
draw_skeleton(frame, seq[0])
draw_label_bar(frame, [(23, 0.62), (9, 0.2)], [f"action {i}" for i in range(60)])
cv2.imwrite("test_overlay.jpg", frame)
print("OK -> test_overlay.jpg")
