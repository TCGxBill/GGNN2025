#!/usr/bin/env python3
"""
Homology-controlled retraining driver for GGNN 2025.

READ-ONLY with respect to the original repository: imports model/loss code and reads
pre-built graphs from binding-site-prediction/, writes ONLY into ggnn-homology-run/.

Protocol fixes relative to the original scripts/train.py:
  * splits are cluster-disjoint (MMseqs2, 30% id / 50% cov)
  * model selection on validation AUPRC (not test AUC)
  * operating threshold derived on validation ONCE, saved, then applied to every test set
  * no warm-starting from any existing checkpoint
"""
import argparse, json, os, sys, time
from pathlib import Path

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC))

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, \
    f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from src.models.gcn_geometric import GeometricGNN
from src.training.trainer import CombinedLoss, CombinedFocalTverskyLoss, DiceLoss


class PerGraphCombinedFT(torch.nn.Module):
    """Weighted BCE + Focal-Tversky, with the Tversky index computed per protein.

    The shipped CombinedFocalTverskyLoss sums TP/FP/FN over the flattened batch, so with
    batch_size=32 a 1000-residue protein contributes ten times more to the overlap term than a
    100-residue one. Averaging the per-graph index instead gives every protein equal weight.
    """

    def __init__(self, bce_weight=0.3, ft_weight=0.7, pos_weight=12.0,
                 alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0):
        super().__init__()
        self.bce_weight, self.ft_weight = bce_weight, ft_weight
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, targets, batch):
        logits, targets = logits.view(-1), targets.view(-1)
        p = torch.sigmoid(logits)
        n = int(batch.max()) + 1
        z = lambda v: torch.zeros(n, device=v.device).index_add_(0, batch, v)
        TP, FP, FN = z(p * targets), z((1 - targets) * p), z(targets * (1 - p))
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        ft = ((1 - tversky) ** self.gamma).mean()
        return self.bce_weight * self.bce(logits, targets) + self.ft_weight * ft


def graph_index(root=None):
    """ID -> path for every pre-built graph (read-only)."""
    base = Path(root) if root else (SRC / "data/processed/combined")
    idx = {}
    for split in ("train", "val", "test"):
        for p in (base / split).glob("*.pt"):
            idx.setdefault(p.stem, p)
    return idx


def load_split(ids, idx):
    out = []
    for i in ids:
        p = idx.get(i)
        if p is None:
            continue
        try:
            g = torch.load(p, weights_only=False)
        except Exception:
            continue
        if g is None or not hasattr(g, "y") or g.y is None or g.x.shape[0] < 5:
            continue
        # drop bookkeeping attributes so graphs from different trees batch together
        for extra in ("n_ligands",):
            if extra in g:
                del g[extra]
        out.append(g)
    return out


def make_loss(name, cfg):
    if name == "combined":
        return CombinedLoss(bce_weight=cfg["bce_weight"], dice_weight=cfg["dice_weight"],
                            pos_weight=cfg["pos_weight"])
    if name == "combined_ft":
        return CombinedFocalTverskyLoss(bce_weight=cfg["bce_weight"], ft_weight=cfg["ft_weight"],
                                        pos_weight=cfg["pos_weight"], alpha=cfg["tversky_alpha"],
                                        beta=cfg["tversky_beta"], gamma=cfg["focal_gamma"])
    if name == "weighted_bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]]))
    if name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    raise ValueError(name)


@torch.no_grad()
def collect(model, loader):
    model.eval()
    P, Y = [], []
    for data in loader:
        out, _ = model(data)
        P.append(torch.sigmoid(out).cpu().numpy().ravel())
        Y.append(data.y.cpu().numpy().ravel())
    return np.concatenate(P), np.concatenate(Y)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n_residues=int(len(y)), prevalence=float(y.mean()),
                auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                mcc=float(matthews_corrcoef(y, pred)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def per_protein_auprc(model, graphs):
    """Protein is the unit of replication; pooled metrics hide between-protein variance."""
    vals = []
    model.eval()
    with torch.no_grad():
        for g in graphs:
            out, _ = model(g)
            p = torch.sigmoid(out).cpu().numpy().ravel()
            y = g.y.cpu().numpy().ravel()
            if y.sum() == 0 or y.sum() == len(y):
                continue
            vals.append(average_precision_score(y, p))
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="B_train",
                    help="key in splits_clean.json (B_train, C_train, B2_train, B3_train)")
    ap.add_argument("--loss", default="combined")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph-dir", default=None, help="alternative processed-graph tree")
    ap.add_argument("--resume", default=None,
                    help="checkpoint tag to warm-start from (legitimate only with validation-based stopping)")
    ap.add_argument("--ft-params", default=None,
                    help='JSON for the per-graph overlap term, e.g. {"alpha":0.5,"beta":0.5,"gamma":1.0} = Dice')
    ap.add_argument("--per-graph-loss", action="store_true",
                    help="compute Focal-Tversky per protein instead of over the flattened batch")
    ap.add_argument("--model-override", default=None,
                    help='JSON patch for config["model"], e.g. \'{"hidden_dims":[512,256,128]}\'')
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping patience on validation AUPRC; 0 disables")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get("GGNN_THREADS", 0))
                          or max(1, (os.cpu_count() or 4) - 2))

    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    mcfg, tcfg = cfg["model"], cfg["training"]
    if a.model_override:
        mcfg.update(json.loads(a.model_override))
        print(f"[{a.tag}] model override: {a.model_override}", flush=True)
    splits = json.load(open(OUT / "splits/splits_clean.json"))
    idx = graph_index(a.graph_dir)

    t0 = time.time()
    tr = load_split(splits[a.train_split], idx)
    va = load_split(splits["val"], idx)
    tc = load_split(splits["T_clean"], idx)
    tl = load_split(splits["T_leaky"], idx)
    print(f"[{a.tag}] graphs loaded in {time.time()-t0:.0f}s | "
          f"train={len(tr)} val={len(va)} T_clean={len(tc)} T_leaky={len(tl)}", flush=True)

    trl = DataLoader(tr, batch_size=tcfg["batch_size"], shuffle=True)
    val = DataLoader(va, batch_size=tcfg["batch_size"], shuffle=False)

    model = GeometricGNN(mcfg)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.tag}] trainable parameters = {n_par:,}", flush=True)
    if a.resume:
        ck = torch.load(OUT / f"checkpoints/{a.resume}.pth", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"[{a.tag}] warm-started from {a.resume}", flush=True)

    crit = PerGraphCombinedFT(bce_weight=tcfg["bce_weight"], ft_weight=tcfg["ft_weight"],
                              pos_weight=tcfg["pos_weight"],
                              **({"alpha": 0.3, "beta": 0.7, "gamma": 1.33}
                                 if not a.ft_params else json.loads(a.ft_params))) \
           if a.per_graph_loss else make_loss(a.loss, tcfg)
    # Test with adamW
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                           weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=tcfg["patience"],
                                                       factor=tcfg["factor"])

    best_auprc, best_state, hist = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for data in trl:
            opt.zero_grad()
            out, _ = model(data)
            if a.per_graph_loss:
                b = getattr(data, "batch", None)
                if b is None:
                    b = torch.zeros(out.shape[0], dtype=torch.long)
                loss = crit(out, data.y.float(), b)
            else:
                loss = crit(out.view(-1), data.y.view(-1).float())
            loss.backward(); opt.step()
            tot += float(loss)
        vp, vy = collect(model, val)
        vauprc = float(average_precision_score(vy, vp))
        vauroc = float(roc_auc_score(vy, vp))
        sched.step(1.0 - vauprc)
        hist.append(dict(epoch=ep, train_loss=tot / max(1, len(trl)),
                         val_auprc=vauprc, val_auroc=vauroc))
        if vauprc > best_auprc:
            best_auprc = vauprc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            # Mirror the best state to disk as it improves. The final save at the end of training
            # is the one that carries the locked threshold, but a long run that dies partway --
            # memory pressure, a killed shell, a machine restart -- previously lost every epoch of
            # work, because best_state lived only in memory. Written atomically so a crash during
            # the write cannot leave a truncated file in place of a good one.
            tmp = OUT / f"checkpoints/{a.tag}.partial.tmp"
            torch.save({"model_state_dict": best_state, "epoch": ep, "val_auprc": vauprc,
                        "tag": a.tag, "partial": True}, tmp)
            os.replace(tmp, OUT / f"checkpoints/{a.tag}.partial.pth")
        print(f"[{a.tag}] epoch {ep:3d}  train_loss={tot/max(1,len(trl)):.4f}  "
              f"val_AUPRC={vauprc:.4f}  val_AUROC={vauroc:.4f}", flush=True)
        if a.patience and (ep - best_ep) >= a.patience:
            print(f"[{a.tag}] early stop: {a.patience} epochs without validation improvement "
                  f"(best epoch {best_ep})", flush=True)
            break

    # ---- model selection: validation AUPRC. Threshold: validation F1, locked here.
    model.load_state_dict(best_state)
    vp, vy = collect(model, val)
    prec, rec, thrs = precision_recall_curve(vy, vp)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.nanargmax(f1s))
    thr = float(thrs[k]) if k < len(thrs) else 0.5
    print(f"[{a.tag}] best epoch {best_ep} (val AUPRC {best_auprc:.4f}); "
          f"validation-locked threshold = {thr:.4f}", flush=True)

    res = dict(tag=a.tag, graph_dir=a.graph_dir, model_override=a.model_override, per_graph_loss=a.per_graph_loss, train_split=a.train_split, loss=a.loss, seed=a.seed,
               n_parameters=int(n_par), best_epoch=int(best_ep),
               val_auprc_best=best_auprc, locked_threshold=thr,
               n_train=len(tr), n_val=len(va), history=hist, evaluations={})
    for name, graphs in (("validation", va), ("T_clean", tc), ("T_leaky", tl)):
        p, y = collect(model, DataLoader(graphs, batch_size=tcfg["batch_size"], shuffle=False))
        m = metrics_at(y, p, thr)
        mu, sd, nn_ = per_protein_auprc(model, graphs)
        m.update(n_proteins=len(graphs), per_protein_auprc_mean=mu,
                 per_protein_auprc_sd=sd, n_proteins_scored=nn_)
        res["evaluations"][name] = m
        print(f"[{a.tag}] {name:11s} n={len(graphs):4d}  AUPRC={m['auprc']:.4f}  "
              f"AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}", flush=True)

    torch.save({"model_state_dict": best_state, "threshold": thr, "tag": a.tag},
               OUT / f"checkpoints/{a.tag}.pth")
    json.dump(res, open(OUT / f"results/{a.tag}.json", "w"), indent=1)
    print(f"[{a.tag}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
