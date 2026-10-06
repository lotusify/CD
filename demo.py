"""Demo ST-GCN 2 stage:
Stage 1: video -> skeleton .npz (pose COCO 17, doc duoc, luu lai de debug)
Stage 2: skeleton .npz -> ST-GCN nhan dien + video overlay

  # Chay full 2 stage + luu trung gian:
  python demo.py --video input.mp4 --out output.mp4 --save-skeleton cache/input.npz
  # Chi Stage 1 (kiem tra pose co dung khong):
  python demo.py --video input.mp4 --stage pose --save-skeleton cache/input.npz
  # Chi Stage 2 (bo qua pose, chay nhanh, debug output loi):
  python demo.py --video input.mp4 --stage infer --load-skeleton cache/input.npz --out output.mp4
"""
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tools.preprocess import (read_frames, extract_coco_sequence,
                              normalize_coco, to_mmaction_input, to_bone)
from tools.skeleton_io import save_skeleton, load_skeleton
from tools.visualize import draw_skeleton_coco, draw_label_bar
from net.stgcn_mmaction import RecognizerGCN, load_mmaction_weights


def load_labels(path):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def pick_device(arg):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def open_writer(path, fps, w, h):
    # mp4v chay on dinh tren Windows (khong can openh264); avc1 thi browser-friendly hon
    for codec in ['mp4v', 'avc1']:
        try:
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if vw.isOpened():
                print(f"[video] codec={codec} -> {path}")
                return vw
        except Exception:
            continue
    raise RuntimeError("Khong mo duoc VideoWriter (thieu codec avc1/mp4v)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='output.mp4')
    ap.add_argument('--stage', default='both', choices=['both', 'pose', 'infer'])
    ap.add_argument('--save-skeleton', default='')
    ap.add_argument('--load-skeleton', default='')
    ap.add_argument('--weights', default='weights/stgcn_mmaction_ntu60xsub2d.pth')
    ap.add_argument('--weights-bone', default='weights/stgcn_bone_ntu60xsub2d.pth')
    ap.add_argument('--no-fusion', action='store_true', help='chi dung joint stream')
    ap.add_argument('--labels', default='labels_ntu60.txt')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--target-fps', type=int, default=30)
    ap.add_argument('--T', type=int, default=100)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[device] {device} (cuda_available={torch.cuda.is_available()})")
    labels = load_labels(args.labels)

    # ---------- Stage 1: video -> skeleton ----------
    seq_coco, frames = None, None
    if args.load_skeleton and args.stage in ('infer', 'both'):
        # Stage 2 tu file co san (van can video goc de ve overlay)
        seq_coco, _ = load_skeleton(args.load_skeleton)
        frames, _ = read_frames(args.video, target_fps=args.target_fps)
        print(f"[video] {len(frames)} frames (de ve overlay), skeleton {seq_coco.shape} (tu file)")
    else:
        frames, _ = read_frames(args.video, target_fps=args.target_fps)
        print(f"[video] {len(frames)} frames @ {args.target_fps}fps from {args.video}")
        if len(frames) == 0:
            raise SystemExit("Video rong!")
        seq_coco = extract_coco_sequence(frames)  # (t,17,3) [0,1]
        if args.save_skeleton:
            save_skeleton(args.save_skeleton, seq_coco, fps=args.target_fps,
                          video_name=os.path.basename(args.video))
        elif args.stage == 'pose':
            # mac dinh stage pose van luu de de debug
            default = 'cache/' + os.path.splitext(os.path.basename(args.video))[0] + '.npz'
            save_skeleton(default, seq_coco, fps=args.target_fps,
                          video_name=os.path.basename(args.video))
            print(f"[stage1-only] xong. Muon nhan dien: --stage infer --load-skeleton {default}")

    if args.stage == 'pose':
        return

    # ---------- Stage 2: skeleton -> ST-GCN (two-stream joint+bone) ----------
    model_j = RecognizerGCN(num_classes=len(labels)).to(device).eval()
    loaded_j = False
    if os.path.isfile(args.weights):
        try:
            load_mmaction_weights(model_j, args.weights)
            loaded_j = True
        except Exception as e:
            print(f"[model-joint] load fail ({e}) -> dummy")
    model_b, loaded_b = None, False
    if not args.no_fusion and os.path.isfile(args.weights_bone):
        try:
            model_b = RecognizerGCN(num_classes=len(labels)).to(device).eval()
            load_mmaction_weights(model_b, args.weights_bone)
            loaded_b = True
        except Exception as e:
            print(f"[model-bone] load fail ({e}) -> joint only")
    print(f"[fusion] joint={loaded_j} bone={loaded_b} (two-stream 92.12% paper)")
    data_j = to_mmaction_input(normalize_coco(seq_coco), T=args.T)
    tensor_j = torch.from_numpy(data_j).to(device)
    with torch.no_grad():
        if loaded_j:
            p_j = F.softmax(model_j(tensor_j), dim=1)[0]
            if loaded_b:
                data_b = to_mmaction_input(to_bone(normalize_coco(seq_coco)), T=args.T)
                p_b = F.softmax(model_b(torch.from_numpy(data_b).to(device)), dim=1)[0]
                probs = ((p_j + p_b) / 2).cpu().numpy()
            else:
                probs = p_j.cpu().numpy()
        else:
            print("[warn] dummy mode")
            probs = np.ones(len(labels), np.float32) / len(labels)
    topk = [(int(i), float(probs[i])) for i in probs.argsort()[::-1][:args.topk]]
    print(f"[result] Top-1: {labels[topk[0][0]]} ({topk[0][1]*100:.1f}%)")
    for i, p in topk:
        print(f"  - {labels[i]}: {p*100:.2f}%")

    h, w = frames[0].shape[:2]
    vw = open_writer(args.out, args.target_fps, w, h)
    for fi, f in enumerate(frames):
        draw_skeleton_coco(f, seq_coco[min(fi, len(seq_coco) - 1)])
        draw_label_bar(f, topk, labels)
        vw.write(f)
    vw.release()
    print(f"[done] da ghi {args.out}")


if __name__ == '__main__':
    main()
