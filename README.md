# ST-GCN Demo — Nhận diện hành động từ video qua skeleton

Pipeline đúng đề cương 4 bước: đọc video → MediaPipe pose (33→25 NTU) → graph không gian-thời gian → ST-GCN → video overlay.

## 1. Cài (2 máy)

```bash
# Máy hiện tại (CPU only, không card rời)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Máy ở nhà RTX 5060 Ti 16GB
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Khuyên Python 3.10/3.11. Bạn đang dùng Python 3.13 vẫn thử được, nếu `mediapipe` lỗi thì tạo venv 3.11.

## 2. Lấy weights NTU60 X-Sub

```bash
python download_weights.py
# nếu script không tải được (link đổi), tải tay rồi copy vào weights/st_gcn.ntu-xsub.pt
# chưa có weights vẫn chạy được pose-only để test pipeline
```

## 3. Chạy demo

```bash
# Video tự quay (rõ người, camera cố định, toàn thân/bán thân)
python demo.py --video input.mp4 --out output.mp4

# Ép chạy CPU / GPU
python demo.py --video input.mp4 --out output.mp4 --device cpu
python demo.py --video input.mp4 --out output.mp4 --device cuda
```

Output: `output.mp4` vẽ skeleton NTU 25 khớp + Top-5 nhãn + confidence.

## 4. Test nhanh không cần video thật

```bash
python test_pose_only.py  # sinh clip giả, test vẽ + preprocess
```

## 5. Lưu ý báo cáo

- Input ST-GCN chuẩn: `(1,3,300,25,1)` — code đã uniform-sample 300 frames, center theo hông, scale torso.
- MediaPipe 2D vs Kinect 3D gốc → domain-gap, accuracy video tự quay sẽ thấp hơn số paper. Nêu trong báo cáo.
- 60 nhãn trong `labels_ntu60.txt` theo chuẩn NTU-RGB+D (falling=A43, hand waving=A23, clapping=A10...).
