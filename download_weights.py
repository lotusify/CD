# -*- coding: utf-8 -*-
"""Download STGCN mmaction2 NTU60 XSub 2D weights (link con song 2026)."""
import os
import urllib.request

os.makedirs('weights', exist_ok=True)
FILES = {
    # joint stream (Top1 88.95) + bone stream (Top1 91.69) -> fusion 92.12
    'weights/stgcn_mmaction_ntu60xsub2d.pth':
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
    'weights/stgcn_bone_ntu60xsub2d.pth':
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth",
}

for dst, url in FILES.items():
    if os.path.isfile(dst) and os.path.getsize(dst) > 1_000_000:
        print("Already have " + dst + ", skip.")
        continue
    print("Downloading " + url + " ...")
    urllib.request.urlretrieve(url, dst)
    print("Done -> " + dst)
