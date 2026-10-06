"""STGCN mmaction2-compatible (COCO 17 joints, 2D), load .pth ma khong can mmaction.
Checkpoint: stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d (Top1 88.95).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Graph COCO stgcn_spatial ----------
def _normalize_digraph(A):
    Dl = np.sum(A, 0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A.dot(Dn)


def _get_hop(num_node, edges, max_hop=1):
    A = np.eye(num_node)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop = np.zeros((num_node, num_node)) + np.inf
    mats = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive = (np.stack(mats) > 0)
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = d
    return hop


COCO_INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
               (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
               (1, 0), (3, 1), (2, 0), (4, 2)]
COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2),
              (1, 3), (2, 4), (0, 5), (0, 6), (11, 12)]


def coco_adjacency(mode='stgcn_spatial', max_hop=1):
    num_node = 17
    center = 0
    hop_dis = _get_hop(num_node, COCO_INWARD, max_hop)
    if mode == 'stgcn_spatial':
        adj = np.zeros((num_node, num_node))
        adj[hop_dis <= max_hop] = 1
        norm = _normalize_digraph(adj)
        A = []
        for hop in range(max_hop + 1):
            a_close = np.zeros((num_node, num_node))
            a_further = np.zeros((num_node, num_node))
            for i in range(num_node):
                for j in range(num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = norm[j, i]
                        else:
                            a_further[j, i] = norm[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A).astype(np.float32)
    # spatial fallback
    I = np.eye(num_node)
    return np.stack([I, I, I]).astype(np.float32)


# ---------- Blocks ----------
class UnitGCN(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.num_sub = A.shape[0]
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.PA = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        nn.init.constant_(self.PA, 1)
        self.conv = nn.Conv2d(in_c, out_c * self.num_sub, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, t, v = x.shape
        A = self.A * self.PA
        x = self.conv(x)
        x = x.view(n, self.num_sub, -1, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        return self.relu(self.bn(x))


class UnitTCN(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        pad = (9 - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, (9, 1), (stride, 1), (pad, 0))
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(0, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, (stride, 1))
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_c, out_c, A)
        self.tcn = UnitTCN(out_c, out_c, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = Residual(in_c, out_c, stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        A = coco_adjacency('stgcn_spatial')
        self.data_bn = nn.BatchNorm1d(3 * 17)
        cfg = [(3, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
               (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True),
               (128, 128, 1, True), (128, 256, 2, True), (256, 256, 1, True),
               (256, 256, 1, True)]
        self.gcn = nn.ModuleList([STGCNBlock(i, o, A, s, r) for i, o, s, r in cfg])

    def forward(self, x):
        # x: (N,M,T,V,C)
        N, M, T, V, C = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for b in self.gcn:
            x = b(x)
        _, C2, T2, V2 = x.shape
        return x.view(N, M, C2, T2, V2)


class RecognizerGCN(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        self.backbone = STGCNBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (N,M,T,V,C)
        f = self.backbone(x)  # (N,M,C,T,V)
        N, M, C, T, V = f.shape
        f = f.view(N * M, C, T, V)
        f = self.pool(f).view(N, M, C).mean(dim=1)
        return self.fc(f)


def load_mmaction_weights(model, path):
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # map: backbone.data_bn.* -> backbone.data_bn.*, backbone.gcn.i.* -> backbone.gcn.i.*, cls_head.fc.* -> fc.*
    mapped = {}
    for k, v in sd.items():
        if k.startswith('backbone.'):
            mapped[k.replace('backbone.', 'backbone.')] = v
        elif k.startswith('cls_head.fc.'):
            mapped[k.replace('cls_head.fc.', 'fc.')] = v
    # backbone keys already match (backbone.data_bn, backbone.gcn.i.gcn/tcn/residual)
    missing, unexpected = [], []
    try:
        res = model.load_state_dict(mapped, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    except Exception as e:
        print(f"[mmaction] load strict=False failed: {e}")
    print(f"[mmaction] loaded {path}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing)
    # PA/A buffers: checkpoint has PA + A as params/buffers; our A is buffer from init (same values) -> strict=False may leave A mismatched but ok since same init
    return model
