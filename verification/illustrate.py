from pathlib import Path
import numpy as np,torch,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
R=Path(__file__).resolve().parents[1];O=R/'verification';d=np.load(O/'T_clean_independent.npz');cuts=np.r_[0,np.cumsum(d['lengths'])]
idx={}
for folder in ['train','val','test']:
 for p in (R/'data_strict/combined'/folder).glob('*.pt'):idx.setdefault(p.stem,p)
code=next(i for i,n in sorted(zip(d['ids'],d['lengths'])) if 180<=n<=300);ii=list(d['ids']).index(code);g=torch.load(idx[code],weights_only=False);xyz=g.pos.numpy();X=xyz-xyz.mean(0);_,_,v=np.linalg.svd(X,full_matrices=False);xy=X@v[:2].T;prob=d['p'][cuts[ii]:cuts[ii+1]]
cmap=LinearSegmentedColormap.from_list('score',['#cbd8e5','#29788c','#e6a42c','#a83340'])
fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained',facecolor='white')
segments=[xy[i:i+2] for i in range(len(xy)-1) if np.linalg.norm(xyz[i+1]-xyz[i])<4.5]
for ax in axs:ax.set_aspect('equal');ax.axis('off');ax.set_xlim(xy[:,0].min()-3,xy[:,0].max()+3);ax.set_ylim(xy[:,1].min()-3,xy[:,1].max()+3)
axs[0].add_collection(LineCollection(segments,colors='#315573',linewidths=1.5));axs[0].scatter(*xy.T,s=6,c='#315573')
edges=g.edge_index.numpy().T;edges=edges[edges[:,0]<edges[:,1]]
axs[1].add_collection(LineCollection(xy[edges],colors='#9bafc6',linewidths=.25,alpha=.28));axs[1].scatter(*xy.T,s=7,c='#267c87')
axs[2].add_collection(LineCollection(segments,colors='#c8d2dc',linewidths=.6));sc=axs[2].scatter(*xy.T,s=15,c=prob,cmap=cmap,vmin=0,vmax=1);fig.colorbar(sc,ax=axs[2],shrink=.65,label='Predicted score',ticks=[0,.5,1])
for ax,title in zip(axs,['Cα coordinates','Residue graph','GGNN residue scores']):ax.set_title(title,fontsize=14,fontweight='bold',color='#25395a',pad=12)
fig.savefig(O/'structure_pipeline.png',dpi=260,bbox_inches='tight');fig.savefig(O/'structure_pipeline.pdf',bbox_inches='tight')
(O/'structure_example.json').write_text(json.dumps({'pdb':str(code),'n_residues':len(xy),'n_edges_directed':g.edge_index.shape[1],'selection':'First alphabetical strict-test ID with 180–300 residues; illustrative only','projection':'PCA of saved coordinates; identical projection in all panels'},indent=2))
print(code,len(xy))
