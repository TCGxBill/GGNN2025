#!/usr/bin/env python3
"""
Extend the local-frame edge features with the RELATIVE ORIENTATION between the two residues'
frames: R_rel = F_i^T F_j, flattened (9 values).

localize_frames.py made the direction vector invariant by expressing it in the source residue's
frame. That tells the network where j sits relative to i, but not how j is *oriented* relative to
i — the information GVP and IPA carry as a relative rotation. R_rel is invariant to rigid motion
for the same reason the rotated direction is.

edge_attr: 20 -> 29  = [distance(1), distance bins(16), direction in frame_i(3), R_rel(9)]
Parameter cost: GATv2Conv's edge projection grows by 9 x sum(heads*out_dim) = 9 x 448 = 4,032,
so the model goes from 282,017 to roughly 286,049 — still well inside the 300K budget.
"""
import sys
from pathlib import Path
import numpy as np
import torch

OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
SRC_TREE = OUT / "data_localframe"
DST_TREE = OUT / "data_relframe"
sys.path.insert(0, str(OUT / "code"))
from localize_frames import local_frames


def main():
    trees = ["combined/train", "combined/val", "combined/test", "scpdb", "pdbbind_refined",
             "sc6k", "coach420", "cryptobench", "dude_diverse", "moad_quality", "difficult_cases"]
    n = 0
    for rel in trees:
        src, dst = SRC_TREE / rel, DST_TREE / rel
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.pt")):
            g = torch.load(p, weights_only=False)
            pos = g.pos.numpy().astype(np.float32)
            ei = g.edge_index.numpy()
            F = local_frames(pos)                      # (N,3,3), columns orthonormal
            # R_rel[e] = F_i^T F_j  -> how j's frame is rotated relative to i's
            R = np.einsum("ekm,ekn->emn", F[ei[0]], F[ei[1]]).astype(np.float32)
            g.edge_attr = torch.cat(
                [g.edge_attr, torch.from_numpy(R.reshape(len(R), 9))], dim=1)
            torch.save(g, dst / p.name)
            n += 1
        print(f"  {rel:22s} {n}", flush=True)
    print(f"\ngraphs rewritten: {n}  (edge_attr dim 20 -> 29)")


if __name__ == "__main__":
    main()
