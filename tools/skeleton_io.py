"""Luu/load skeleton trung gian (.npz) cho pipeline 2 stage."""
import os
import numpy as np


def save_skeleton(path, seq_coco, fps=30, video_name="", extra=None):
    """seq_coco (T,17,3) x,y in [0,1] + conf."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = dict(skeleton=seq_coco.astype(np.float32), fps=np.array(fps),
             video_name=np.array(video_name))
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)
    print(f"[stage1] da luu skeleton {seq_coco.shape} -> {path}")
    return path


def load_skeleton(path):
    z = np.load(path, allow_pickle=True)
    seq = z['skeleton'].astype(np.float32)
    fps = int(z['fps']) if 'fps' in z else 30
    vname = str(z['video_name']) if 'video_name' in z else ""
    print(f"[stage2] da tai skeleton {seq.shape} tu {path} (video goc: {vname}, fps={fps})")
    # thong ke nhanh de debug output loi
    conf = seq[:, :, 2]
    print(f"  frames={len(seq)}, joints=17, mean_conf={conf.mean():.3f}, "
          f"missing_frames={(conf.max(axis=1) < 0.2).sum()}/{len(seq)}")
    return seq, fps
