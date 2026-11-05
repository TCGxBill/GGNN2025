from pathlib import Path
import sys,json,ast,collections
import numpy as np,torch,yaml
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parents[1];O=R/'verification';sys.path.insert(0,str(O/'copied_code'));from gcn_geometric import GeometricGNN
cfg=yaml.safe_load((O/'copied_code/config_optimized.yaml').read_text())['model'];m=GeometricGNN(cfg);m.load_state_dict(torch.load(R/'checkpoints/B5_localframe.pth',weights_only=False)['model_state_dict'],strict=True);m.eval();torch.set_num_threads(4)
sp=json.loads((R/'splits/splits_clean.json').read_text());idx={}
for s in ['train','val','test']:
 for p in (R/'data_localframe/combined'/s).glob('*.pt'):idx.setdefault(p.stem,p)
ex={'HOH','WAT','TIP3','SOL','NA','CL','K','MG','CA','ZN','MN','FE','SO4','PO4'};ys=[];ps=[];hits=[];dists=[];without=0
for code in sp['T_clean']:
 if code not in idx:continue
 g=torch.load(idx[code],weights_only=False);pos=g.pos.numpy()
 with torch.inference_mode():p=torch.sigmoid(m(g)[0]).numpy().ravel()
 ys.append(g.y.numpy());ps.append(p)
 src=next(R.parent/'binding-site-prediction/data/raw'/d/f'{code}.pdb' for d in ['holo4k','joined'] if (R.parent/'binding-site-prediction/data/raw'/d/f'{code}.pdb').exists())
 groups=collections.defaultdict(list)
 for line in src.read_text(errors='ignore').splitlines():
  if line.startswith('ENDMDL'):break
  if line.startswith('HETATM') and line[17:20].strip() not in ex:groups[(line[21],line[22:27],line[17:20])].append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
 sites=np.array([np.array(v,dtype=np.float32).mean(0) for v in groups.values() if len(v)>=3]);sel=np.where(p>=.7)[0];pockets=[]
 if len(sel):
  n,lab=connected_components(csr_matrix(cdist(pos[sel],pos[sel])<=8),directed=False)
  for k in range(n):
   ii=sel[lab==k]
   if len(ii)>=3:pockets.append((float(p[ii].sum()),(pos[ii]*p[ii,None]).sum(0)/p[ii].sum()))
 pockets.sort(key=lambda x:-x[0]);centres=np.array([v[1] for v in pockets])
 if len(centres):
  dd=cdist(centres,sites).min(1);hits.append([bool((dd[:k]<=4).any()) for k in [1,3,5]]);dists.append(float(dd[0]))
 else:hits.append([False]*3);without+=1
res={'checkpoint':'B5_localframe','n':len(ys),'auprc':float(average_precision_score(np.concatenate(ys),np.concatenate(ps))),'auroc':float(roc_auc_score(np.concatenate(ys),np.concatenate(ps))),'clustering':{'cut':.7,'link_angstrom':8,'minimum_residues':3},'top1_top3_top5':np.mean(hits,axis=0).tolist(),'median_first_centre_distance_conditional_on_prediction':float(np.median(dists)),'proteins_with_no_predicted_pocket':without}
(O/'original_figure_verification.json').write_text(json.dumps(res,indent=2));print(res)
