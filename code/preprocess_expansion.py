#!/usr/bin/env python3
"""
Build graphs for additional structures that are absent from the original pool, using the
repository's own ProteinPreprocessor / ProteinGraphBuilder so features are identical, but
labelling with ALL candidate ligands (as in relabel_all_ligands.py) rather than the largest one.

Candidate sources: external_data/p2rank-datasets/{holo4k,fptrain,chen11} and data/raw/biolip_sample.
Only structures whose MMseqs2 cluster contains no benchmark protein are written.

Writes into ggnn-homology-run/data_relabeled/combined/train/ . Read-only w.r.t. the source repo.
"""
import collections, json, os, sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
sys.path.insert(0, str(SRC))

import numpy as np
import torch
from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder

EXCLUDE = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN",
           "FE", "SO4", "PO4"}
CUTOFF = 6.0
# identical to scripts/preprocess_all.py, so the new graphs match the existing ones
CFG = {"distance_threshold": 6.0, "k_neighbors": 15}


def all_ligand_coords(path):
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
    if not keep:
        return None
    return np.array([p for v in keep for p in v], dtype=np.float32)


def main():
    wanted = json.load(open(OUT / "splits/expansion_paths.json"))
    pre, gb = ProteinPreprocessor(CFG), ProteinGraphBuilder(CFG)
    dst = OUT / "data_relabeled/combined/train"
    dst.mkdir(parents=True, exist_ok=True)

    written, failed, ids = 0, 0, []
    for name, path in wanted.items():
        try:
            d = pre.process_pdb(path)
            if d is None or d.get("coordinates") is None:
                failed += 1
                continue
            P = np.asarray(d["coordinates"], dtype=np.float32)
            L = all_ligand_coords(path)
            if L is None or len(P) < 5:
                failed += 1
                continue
            y = (np.linalg.norm(P[:, None, :] - L[None, :, :], axis=2).min(1) <= CUTOFF)
            if y.sum() == 0:
                failed += 1
                continue
            graph = gb.build_graph(d["node_features"], d["coordinates"],
                                   y.astype(np.float32))
            torch.save(graph, dst / f"{name}.pt")
            ids.append(name)
            written += 1
        except Exception:
            failed += 1
        if (written + failed) % 100 == 0:
            print(f"  {written+failed}/{len(wanted)}  written={written}", flush=True)

    json.dump(ids, open(OUT / "splits/expansion_ids.json", "w"))
    print(f"\nwritten {written} | failed {failed}")


if __name__ == "__main__":
    main()
