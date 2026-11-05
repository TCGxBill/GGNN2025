from pathlib import Path
import json,ast,collections,csv
import numpy as np,torch
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parents[1];SRC=R.parent/'binding-site-prediction';O=R/'verification';sp=json.loads((R/'splits/splits_clean.json').read_text())
def sets(file,names):
 out={}
 for a in ast.parse(file.read_text()).body:
  if isinstance(a,ast.Assign):
   for v in a.targets:
    if isinstance(v,ast.Name) and v.id in names:out[v.id]=ast.literal_eval(a.value)
 return out
ex=sets(R/'code/pocket_eval.py',['EXCLUDE']);ex.update(sets(R/'code/strict_labels.py',['ARTIFACT','MODRES']));excluded=set.union(*ex.values())
idx={};old={}
for d in ['train','val','test']:
 for p in (R/'data_strict/combined'/d).glob('*.pt'):idx.setdefault(p.stem,p)
 for p in (R/'data_localframe/combined'/d).glob('*.pt'):old.setdefault(p.stem,p)
keep=[];ys=[];yo=[];mismatch=[];edge_match={6:0,8:0};sszero=0;nres=0;cats=collections.Counter();enrich=np.zeros((2,20),int)
for code in sp['T_clean']:
 if code not in old:continue
 g0=torch.load(old[code],weights_only=False);yo.append(g0.y.numpy());keep.extend([code in idx]*len(g0.y))
 if code not in idx:continue
 g=torch.load(idx[code],weights_only=False);pos=g.pos.numpy();y=g.y.numpy();ys.append(y);nres+=len(y);sszero+=int((g.x[:,23:26].sum(1)==0).sum())
 for a,b in zip(g.x[:,:20].argmax(1).numpy(),y):enrich[0,a]+=1;enrich[1,a]+=int(b)
 p=next((SRC/'data/raw'/d/f'{code}.pdb' for d in ['holo4k','joined'] if (SRC/'data/raw'/d/f'{code}.pdb').exists()),None)
 groups=collections.defaultdict(list)
 for line in p.read_text(errors='ignore').splitlines():
  if line.startswith('ENDMDL'):break
  if line.startswith('HETATM'):
   rn=line[17:20].strip().upper();groups[(line[21],line[22:27],rn)].append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
 atoms=[]
 for (_,_,rn),coords in groups.items():
  if rn not in ex['EXCLUDE'] and len(coords)>=3:
   cats['permissive']+=1
   if rn in ex['ARTIFACT']:cats['artifact']+=1
   elif rn in ex['MODRES']:cats['modres']+=1
  if rn not in excluded and len(coords)>=3:atoms+=coords
 yy=(cKDTree(np.array(atoms,dtype=np.float32)).query(pos)[0]<=6).astype(int)
 if not np.array_equal(yy,y):mismatch.append([code,int((yy!=y).sum())])
 # Exact edge-set comparison from coordinate geometry (ties may affect kNN).
 tree=cKDTree(pos);kn=tree.query(pos,k=min(16,len(pos)))[1][:,1:];actual=set(map(tuple,g.edge_index.numpy().T.tolist()))
 for radius in [6,8]:
  pairs={(i,int(j)) for i,ns in enumerate(kn) for j in ns if i!=j}
  pairs.update((i,j) for i,ns in enumerate(tree.query_ball_point(pos,radius-1e-7)) for j in ns if i!=j)
  if pairs==actual:edge_match[radius]+=1
keep=np.array(keep);y=np.concatenate(ys);yo=np.concatenate(yo);res={'strict_label_mismatches':mismatch,'strict_test_graphs':len(ys),'edge_graph_exact_matches':edge_match,'secondary_structure_all_zero_residues':sszero,'residues':nres,'ligand_groups_244':dict(cats),'enrichment_counts':enrich.tolist(),'baselines':{}}
for name in ['fpocket','p2rank']:
 d=np.load(R/f'results/{name}_T_clean.npz');assert np.array_equal(d['y'],yo);s=d['s'][keep];assert len(s)==len(y);res['baselines'][name]={'auprc':float(average_precision_score(y,s)),'auroc':float(roc_auc_score(y,s))};np.savez_compressed(O/(name+'_strict_aligned.npz'),y=y,s=s)
clusters={r['protein']:r['cluster_rep'] for r in csv.DictReader((R/'splits/cluster_assignment.csv').open())}
for f in ['expansion_clusters','expansion2_clusters']:clusters.update(json.loads((R/f'splits/{f}.json').read_text()))
cs={k:{clusters[i] for i in sp[k] if i in idx and i in clusters} for k in ['B5_train','val','T_clean']}
res['cluster_manifest_overlap']={a+'__'+b:sorted(cs[a]&cs[b]) for a,b in [('B5_train','val'),('B5_train','T_clean'),('val','T_clean')]};res['unmapped_cluster_ids']=[i for k in cs for i in sp[k] if i in idx and i not in clusters]
(O/'data_verification.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k!='enrichment_counts'}),flush=True)
