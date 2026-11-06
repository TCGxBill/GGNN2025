#!/usr/bin/env python3
"""
Make the edge geometry rotation-invariant at zero parameter cost.

src/data/graph_builder.py::_compute_edge_features stores the direction vector
    d_ij = (p_j - p_i) / ||p_j - p_i||
in the GLOBAL coordinate frame (paper Eq. 2). A rigid rotation of the structure therefore
changes the input: measured on 40 T_clean proteins, a 60 deg rotation altered the predicted
probability of every one of them (mean max |dp| = 0.082).

This script replaces dimensions 17:20 of edge_attr with the same direction expressed in a
local frame built from each residue's own Ca trace, so the features are invariant to rigid
motion. Dimensionality is unchanged (20), so the model has exactly the same parameter count.

Frame at residue i (ProteinMPNN / GVP convention, Ca-only variant):
    b = Ca_i - Ca_{i-1},  c = Ca_{i+1} - Ca_i,  n = b x c
    F_i = [ b_hat , n_hat , (n x b)_hat ]   (columns, orthonormal)
Chain termini and degenerate (collinear) triples reuse the nearest valid frame.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
SRC_TREE = OUT / "data_relabeled"
DST_TREE = OUT / "data_localframe"
EPS = 1e-8


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + EPS)


def local_frames(pos):
    """(N,3) Ca coordinates -> (N,3,3) orthonormal frames, columns = basis vectors."""
    n = len(pos)
    F = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    if n < 3:
        return F
    b = pos[1:-1] - pos[:-2]
    c = pos[2:] - pos[1:-1]
    nrm = np.cross(b, c)
    ok = np.linalg.norm(nrm, axis=1) > 1e-4          # skip collinear triples
    b_h, n_h = _unit(b), _unit(nrm)
    t_h = _unit(np.cross(n_h, b_h))
    mid = np.stack([b_h, n_h, t_h], axis=-1).astype(np.float32)   # (N-2,3,3)
    F[1:-1][ok] = mid[ok]
    # termini and degenerate positions inherit the nearest valid frame
    valid = np.zeros(n, dtype=bool)
    valid[1:-1] = ok
    if valid.any():
        last = None
        for i in range(n):
            if valid[i]:
                last = i
            elif last is not None:
                F[i] = F[last]
        nxt = None
        for i in range(n - 1, -1, -1):
            if valid[i]:
                nxt = i
            elif nxt is not None and not valid[i] and (i == 0 or not valid[:i].any()):
                F[i] = F[nxt]
    return F


def main():
    trees = ["combined/train", "combined/val", "combined/test", "scpdb", "pdbbind_refined",
             "sc6k", "coach420", "cryptobench", "dude_diverse", "moad_quality", "difficult_cases"]
    total = changed = 0
    for rel in trees:
        src, dst = SRC_TREE / rel, DST_TREE / rel
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.pt")):
            total += 1
            g = torch.load(p, weights_only=False)
            pos = g.pos.numpy().astype(np.float32)
            ei = g.edge_index.numpy()
            F = local_frames(pos)
            d = pos[ei[1]] - pos[ei[0]]
            d = _unit(d)
            # express in the frame of the SOURCE residue -> invariant to rigid motion
            d_local = np.einsum("ekj,ek->ej", F[ei[0]], d).astype(np.float32)
            ea = g.edge_attr.clone()
            ea[:, 17:20] = torch.from_numpy(d_local)
            g.edge_attr = ea
            torch.save(g, dst / p.name)
            changed += 1
        print(f"  {rel:22s} {changed}", flush=True)
    print(f"\ngraphs rewritten: {changed}/{total}  (edge_attr dim unchanged = 20)")


if __name__ == "__main__":
    main()
