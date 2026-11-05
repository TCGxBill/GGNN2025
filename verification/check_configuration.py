from pathlib import Path
import json,torch,numpy as np,importlib.util
R=Path(__file__).resolve().parents[1];sp=json.loads((R/'splits/splits_clean.json').read_text());idx={}
for d in ['train','val','test']:
 for p in (R/'data_strict/combined'/d).glob('*.pt'):idx.setdefault(p.stem,p)
res={}
for k in ['B5_train','val','T_clean','T_leaky']:
 n=0;dims=set();ss=acc=nr=0
 for code in sp[k]:
  if code not in idx:continue
  g=torch.load(idx[code],weights_only=False)
  if len(g.x)<5:continue
  n+=1;dims.add((g.x.shape[1],g.edge_attr.shape[1]));nr+=len(g.x);ss+=int((g.x[:,23:26].sum(1)==0).sum());acc+=int((g.x[:,26]==.5).sum())
 res[k]={'n_proteins':n,'dimensions':[list(d) for d in dims],'n_residues':nr,'zero_ss_residues':ss,'default_accessibility_residues':acc}
spec=importlib.util.spec_from_file_location('lf',R/'code/localize_frames.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
g=torch.load(idx[sp['T_clean'][0]],weights_only=False);F=mod.local_frames(g.pos.numpy());det=np.linalg.det(F);res['frame_determinant']={'minimum':float(det.min()),'maximum':float(det.max()),'note':'Orthogonal basis determinant -1 for nondegenerate frame; not SO(3)'}
(R/'verification/configuration_verification.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
