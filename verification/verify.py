from pathlib import Path
import sys,json,hashlib,time
import numpy as np,torch,yaml
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_recall_curve
from scipy.stats import ttest_rel,t
sys.path.insert(0,str(Path(__file__).parent/'copied_code'))
from gcn_geometric import GeometricGNN
R=Path(__file__).resolve().parents[1];O=R/'verification';torch.set_num_threads(4)
cfg=yaml.safe_load((O/'copied_code/config_optimized.yaml').read_text())['model'];m=GeometricGNN(cfg);ck=torch.load(R/'checkpoints/B5_strict_lf.pth',weights_only=False);m.load_state_dict(ck['model_state_dict'],strict=True);m.eval()
report={'configuration':cfg,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'state_tensor_elements':sum(p.numel() for p in ck['model_state_dict'].values()),'checkpoint_sha256':hashlib.sha256((R/'checkpoints/B5_strict_lf.pth').read_bytes()).hexdigest(),'splits':{}}
print(json.dumps(report),flush=True)
sp=json.loads((R/'splits/splits_clean.json').read_text());idx={}
for d in ['train','val','test']:
 for p in (R/'data_strict/combined'/d).glob('*.pt'):idx.setdefault(p.stem,p)
for split in ['val','T_clean','T_leaky']:
 ys=[];ps=[];ids=[];lengths=[]
 for i in sp[split]:
  if i not in idx:continue
  g=torch.load(idx[i],weights_only=False)
  if g.x.shape[0]<5:continue
  with torch.inference_mode():pr=torch.sigmoid(m(g)[0]).numpy().ravel()
  ys.append(g.y.numpy().ravel());ps.append(pr);ids.append(i);lengths.append(len(pr))
 y=np.concatenate(ys);p=np.concatenate(ps);tn,fp,fn,tp=confusion_matrix(y,p>=ck['threshold']).ravel()
 d={'n_proteins':len(ids),'n_residues':len(y),'positives':int(y.sum()),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
 if split=='val':
  prec,rec,thr=precision_recall_curve(y,p);k=np.argmax(2*prec*rec/(prec+rec+1e-12));d['rederived_threshold']=float(thr[k])
 report['splits'][split]=d;np.savez_compressed(O/(split+'_independent.npz'),y=y,p=p,ids=np.array(ids),lengths=np.array(lengths));(O/'verification.json').write_text(json.dumps(report,indent=2));print(split,json.dumps(d),flush=True)
# Fresh calculation from six stored seed runs, not the paired-test summary.
a=[];b=[]
for suffix in ['_s1','_s2','']:
 a.append(json.loads((R/f'results/B4_global{suffix}.json').read_text())['evaluations']['T_clean']['auprc'])
 b.append(json.loads((R/f'results/B4_localframe{suffix}.json').read_text())['evaluations']['T_clean']['auprc'])
d=np.array(b)-a;se=d.std(ddof=1)/np.sqrt(3)
report['paired_seed_test']={'global':a,'local':b,'mean_difference':float(d.mean()),'relative_gain':float(np.mean(b)/np.mean(a)-1),'t':float(ttest_rel(b,a).statistic),'p':float(ttest_rel(b,a).pvalue),'ci95':(d.mean()+np.array([-1,1])*t.ppf(.975,2)*se).tolist()}
(O/'verification.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)
