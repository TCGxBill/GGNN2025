from pathlib import Path
import sys,json,numpy as np,torch,yaml
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'verification/copied_code'));from gcn_geometric import GeometricGNN
cfg=yaml.safe_load((R/'verification/copied_code/config_optimized.yaml').read_text())['model'];sp=json.loads((R/'splits/splits_clean.json').read_text());torch.set_num_threads(4);out={}
for kind,tree in [('global','data_relabeled'),('localframe','data_localframe')]:
 idx={}
 for s in ['train','val','test']:
  for p in (R/tree/'combined'/s).glob('*.pt'):idx.setdefault(p.stem,p)
 graphs=[torch.load(idx[i],weights_only=False) for i in sp['T_clean'] if i in idx];y=np.concatenate([g.y.numpy().ravel() for g in graphs])
 for suffix in ['_s1','_s2','']:
  tag=f'B4_{kind}{suffix}';m=GeometricGNN(cfg);ck=torch.load(R/f'checkpoints/{tag}.pth',weights_only=False);m.load_state_dict(ck['model_state_dict'],strict=True);m.eval();ps=[]
  with torch.inference_mode():
   for g in graphs:ps.append(torch.sigmoid(m(g)[0]).numpy().ravel())
  p=np.concatenate(ps);out[tag]={'n_proteins':len(graphs),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p))};print(tag,out[tag],flush=True);(R/'verification/seed_checkpoint_verification.json').write_text(json.dumps(out,indent=2))
