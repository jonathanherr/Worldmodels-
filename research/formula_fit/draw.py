import sys, pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shrink import norm, side, is_axis
P=pickle.load(open('best.pkl','rb'))
ns=[int(a) for a in sys.argv[1:]]
fig,axs=plt.subplots(1,len(ns),figsize=(7*len(ns),7))
for ax,n in zip(np.atleast_1d(axs),ns):
    p=norm(P[n])
    for x,y,a in p:
        c,s=np.cos(a),np.sin(a); R=np.array([[c,-s],[s,c]])
        pts=(R@np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T).T+[x,y]
        ax.fill(pts[:,0],pts[:,1],fc='#9ecae1' if abs(np.sin(2*a))<1e-12 else '#fd8d3c',ec='k',lw=.3)
    L=side(p); ax.set_xlim(0,L); ax.set_ylim(0,L); ax.set_aspect('equal'); ax.set_title(f'n={n} L={L:.6f}')
plt.savefig('draw.png',dpi=70,bbox_inches='tight')
