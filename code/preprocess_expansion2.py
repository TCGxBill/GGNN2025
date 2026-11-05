#!/usr/bin/env python3
"""
Build graphs for the RCSB expansion set, in all three edge-feature variants in one pass:

  data_relabeled/  global-frame direction   (edge_attr 20)  - matches the original encoding
  data_localframe/ local-frame direction    (edge_attr 20)  - rotation invariant
  data_relframe/   + relative frame rotation (edge_attr 29) - adds F_i^T F_j

Features come from the repository's own ProteinPreprocessor / ProteinGraphBuilder at the same
6.0 A / k=15 settings as every existing graph; labels use ALL candidate ligands.

Only structures whose cluster contains no benchmark, val, T_clean or T_leaky protein are written
(filtering already done in splits/expansion2_paths.json).
"""
import collections, json, sys
from pathlib import Path

import numpy as np
import torch

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OUT / "code"))

from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder
from localize_frames import local_frames, _unit

EXCLUDE = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN",
           "FE", "SO4", "PO4"}
CUTOFF = 6.0
CFG = {"distance_threshold": 6.0, "k_neighbors": 15}


def all_ligands(path):
    g = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HETATM"):
            rn = line[17:20].strip().upper()
            if rn in EXCLUDE:
                continue
            g[(line[21], line[22:27], rn)].append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    keep = [v for v in g.values() if len(v) >= 3]
    return np.array([p for v in keep for p in v], dtype=np.float32) if keep else None


def main():
    paths = json.load(open(OUT / "splits/expansion2_paths.json"))
    pre, gb = ProteinPreprocessor(CFG), ProteinGraphBuilder(CFG)
    dirs = {k: OUT / k / "combined/train" for k in
            ("data_relabeled", "data_localframe", "data_relframe")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    written, failed, ids = 0, 0, []
    for name, path in paths.items():
        try:
            d = pre.process_pdb(path)
            if d is None or d.get("coordinates") is None:
                failed += 1
                continue
            P = np.asarray(d["coordinates"], dtype=np.float32)
            L = all_ligands(path)
            if L is None or len(P) < 5:
                failed += 1
                continue
            y = (np.linalg.norm(P[:, None, :] - L[None, :, :], axis=2).min(1) <= CUTOFF)
            if y.sum() == 0:
                failed += 1
                continue
            g = gb.build_graph(d["node_features"], d["coordinates"], y.astype(np.float32))
            torch.save(g, dirs["data_relabeled"] / f"{name}.pt")

            ei = g.edge_index.numpy()
            F = local_frames(P)
            dirn = _unit(P[ei[1]] - P[ei[0]])
            d_loc = np.einsum("ekj,ek->ej", F[ei[0]], dirn).astype(np.float32)
            g2 = g.clone()
            ea = g.edge_attr.clone()
            ea[:, 17:20] = torch.from_numpy(d_loc)
            g2.edge_attr = ea
            torch.save(g2, dirs["data_localframe"] / f"{name}.pt")

            R = np.einsum("ekm,ekn->emn", F[ei[0]], F[ei[1]]).astype(np.float32)
            g3 = g2.clone()
            g3.edge_attr = torch.cat([ea, torch.from_numpy(R.reshape(len(R), 9))], dim=1)
            torch.save(g3, dirs["data_relframe"] / f"{name}.pt")

            ids.append(name)
            written += 1
        except Exception:
            failed += 1
        if (written + failed) % 250 == 0:
            print(f"  {written+failed}/{len(paths)}  written={written}", flush=True)

    json.dump(ids, open(OUT / "splits/expansion2_ids.json", "w"))
    print(f"\nwritten {written} | failed {failed}")


if __name__ == "__main__":
    main()
