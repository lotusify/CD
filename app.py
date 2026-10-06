"""Gradio UI 2-stage (English): Stage 1 video->skeleton.npz, Stage 2 skeleton->recognition.
Output labels are shown BELOW the video (markdown), the video itself only draws the skeleton.
"""
import asyncio
import os
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Silence harmless ConnectionResetError noise on Windows (gradio/uvicorn)
def _quiet_exceptions(loop, context):
    exc = context.get('exception')
    if isinstance(exc, (ConnectionResetError, asyncio.IncompleteReadError)):
        return
    loop.default_exception_handler(context)


try:
    asyncio.get_event_loop().set_exception_handler(_quiet_exceptions)
except Exception:
    pass

from tools.preprocess import (read_frames, extract_coco_sequence,
                              normalize_coco, to_mmaction_input, to_bone)
from tools.skeleton_io import save_skeleton, load_skeleton
from tools.visualize import draw_skeleton_coco
from net.stgcn_mmaction import RecognizerGCN, load_mmaction_weights

LABELS_PATH = 'labels_ntu60.txt'
WEIGHTS_J = 'weights/stgcn_mmaction_ntu60xsub2d.pth'
WEIGHTS_B = 'weights/stgcn_bone_ntu60xsub2d.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(LABELS_PATH, encoding='utf-8') as f:
    LABELS = [l.strip() for l in f if l.strip()]

MODEL_J = RecognizerGCN(num_classes=len(LABELS)).to(DEVICE).eval()
LOADED_J = os.path.isfile(WEIGHTS_J)
if LOADED_J:
    try:
        load_mmaction_weights(MODEL_J, WEIGHTS_J)
    except Exception as e:
        print(f"[UI] joint load fail: {e}")
        LOADED_J = False
MODEL_B, LOADED_B = None, False
if os.path.isfile(WEIGHTS_B):
    try:
        MODEL_B = RecognizerGCN(num_classes=len(LABELS)).to(DEVICE).eval()
        load_mmaction_weights(MODEL_B, WEIGHTS_B)
        LOADED_B = True
    except Exception as e:
        print(f"[UI] bone load fail: {e}")
LOADED = LOADED_J
print(f"[UI] two-stream fusion: joint={LOADED_J} bone={LOADED_B}")


def open_writer(path, fps, w, h):
    for codec in ['mp4v', 'avc1']:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if vw.isOpened():
            return vw
    raise RuntimeError("No video codec")


def run_inference(frames, seq):
    normed = normalize_coco(seq)
    tj = torch.from_numpy(to_mmaction_input(normed, T=100)).to(DEVICE)
    with torch.no_grad():
        if not LOADED_J:
            return np.ones(len(LABELS), np.float32) / len(LABELS)
        pj = F.softmax(MODEL_J(tj), 1)[0]
        if LOADED_B:
            tb = torch.from_numpy(to_mmaction_input(to_bone(normed), T=100)).to(DEVICE)
            pb = F.softmax(MODEL_B(tb), 1)[0]
            return ((pj + pb) / 2).cpu().numpy()
        return pj.cpu().numpy()


def format_results(probs, topk=5):
    top = [(int(i), float(probs[i])) for i in probs.argsort()[::-1][:int(topk)]]
    lines = [f"### Top-1: {LABELS[top[0][0]]} ({top[0][1]*100:.1f}%)", ""]
    lines += [f"{r}. {LABELS[i]} — {p*100:.2f}%" for r, (i, p) in enumerate(top, 1)]
    if not LOADED:
        lines += ["", "*Dummy mode: run download_weights.py to get real weights.*"]
    return top, "\n".join(lines)


def write_skeleton_video(frames, seq, fps=30):
    h, w = frames[0].shape[:2]
    fd, out_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    vw = open_writer(out_path, fps, w, h)
    for fi, f in enumerate(frames):
        draw_skeleton_coco(f, seq[min(fi, len(seq) - 1)])
        vw.write(f)
    vw.release()
    return out_path


def stage1_extract(video_path):
    """Stage 1: video -> skeleton preview + .npz file."""
    if not video_path or not os.path.isfile(video_path):
        return None, None, "Please upload an MP4/AVI video first.", None, None
    frames, _ = read_frames(video_path, target_fps=30)
    if not frames:
        return None, None, "Could not read the video.", None, None
    seq = extract_coco_sequence(frames)
    fd, npz_path = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    save_skeleton(npz_path, seq, fps=30, video_name=os.path.basename(video_path))
    preview = write_skeleton_video([f.copy() for f in frames[:90]],
                                   seq[:min(90, len(seq))])
    conf = seq[:, :, 2]
    info = (f"Done: {seq.shape[0]} frames x 17 joints (COCO), "
            f"mean_conf={conf.mean():.2f}, missing_frames={(conf.max(1) < 0.2).sum()}.")
    return preview, npz_path, info, video_path, npz_path


def _safe_topk(topk, default=5):
    try:
        topk = int(topk)
    except (TypeError, ValueError):
        return default
    return min(max(topk, 1), 10)


def recognize(inp_video, st_video, st_npz, npz_upload, topk=5):
    """Reuse Stage-1 state/.npz when possible, else extract fresh. No video re-render."""
    topk = _safe_topk(topk)
    video = inp_video if (inp_video and os.path.isfile(inp_video)) else st_video
    if not video or not os.path.isfile(video):
        return {}, "Please upload a video first."
    seq = None
    if npz_upload and os.path.isfile(npz_upload):
        try:
            seq, _ = load_skeleton(npz_upload)
        except Exception as e:
            return {}, f"Could not load .npz: {e}"
    if seq is None and st_npz and os.path.isfile(st_npz) and video == st_video:
        seq, _ = load_skeleton(st_npz)
    frames, _ = read_frames(video, target_fps=30)
    if not frames:
        return {}, "Could not read the video."
    if seq is None or len(seq) != len(frames):
        seq = extract_coco_sequence(frames)
    probs = run_inference(frames, seq)
    top, md = format_results(probs, topk)
    label_dict = {LABELS[i]: float(p) for i, p in top}
    return label_dict, md


def build_ui():
    import gradio as gr
    with gr.Blocks(title="ST-GCN Action Recognition (NTU60)") as demo:
        gr.Markdown(
            "## ST-GCN Action Recognition — NTU60 X-Sub 2D (two-stream joint+bone) "
            f"`{DEVICE}` | joint `{'ok' if LOADED_J else 'missing'}`, "
            f"bone `{'ok' if LOADED_B else 'missing'}`")

        gr.Markdown("### Step 1 — Extract skeleton")
        with gr.Row():
            inp1 = gr.Video(label="Input video")
            prev1 = gr.Video(label="Skeleton preview (first 90f)")
        btn1 = gr.Button("Extract skeleton", variant="primary")
        info1 = gr.Markdown()
        npz_dl = gr.File(label="Skeleton .npz (download)")

        gr.Markdown("### Step 2 — Recognize")
        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
        btn2 = gr.Button("Recognize", variant="primary")
        with gr.Row():
            out_label = gr.Label(label="Top-K confidences")
            outt = gr.Markdown(label="Predicted actions")

        with gr.Accordion("Advanced: reuse / debug skeleton (.npz)", open=False):
            gr.Markdown("Auto-reused when present. Upload an old `.npz` (or `.npz.txt`) "
                        "to skip pose estimation.")
            npz_up = gr.File(label="Reuse skeleton .npz (optional)",
                             file_types=[".npz", ".txt"])

        st_video = gr.State(value=None)
        st_npz = gr.State(value=None)
        btn1.click(stage1_extract, [inp1], [prev1, npz_dl, info1, st_video, st_npz])
        btn2.click(recognize, [inp1, st_video, st_npz, npz_up, topk], [out_label, outt])
    return demo


if __name__ == '__main__':
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
