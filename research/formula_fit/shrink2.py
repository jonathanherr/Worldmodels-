"""Staircase band removal: delete a chain of axis-aligned squares crossing the box wall to wall
(steps allowed, as at a fault line) and drop everything above it by 1. Do the same for a column.
Side shrinks by 1, n shrinks by the chain lengths."""
import numpy as np, pickle, math, sys
from shrink import hext, is_axis, norm, side
E=1e-7
def row_paths(p, maxpaths=400):
    x,y,a=p[:,0],p[:,1],p[:,2]; h=hext(a); xl=x-h; xr=x+h; yl=y-h
    axm=is_axis(a); W=xr.max()
    idx=np.where(axm)[0]
    starts=idx[np.abs(xl[idx])<E]
    out=[]
    for s in starts:
        stack=[[s]]
        while stack and len(out)<maxpaths:
            path=stack.pop(); last=path[-1]
            if abs(xr[last]-W)<E: out.append(path); continue
            nxt=idx[(np.abs(xl[idx]-xr[last])<E)&(np.abs(yl[idx]-yl[last])<1-E)]
            for j in nxt: stack.append(path+[j])
    return out
def remove_row(p, path):
    x,y,a=p[:,0],p[:,1],p[:,2]; h=hext(a); xl=x-h; xr=x+h; yl=y-h; yh=y+h
    inp=np.zeros(len(p),bool); inp[path]=True
    pl=xl[path]; pr=xr[path]; pb=yl[path]; pt=pb+1
    up=np.zeros(len(p),bool)
    for i in np.where(~inp)[0]:
        ov=(pr>xl[i]+E)&(pl<xr[i]-E)
        if not ov.any(): return None
        if yl[i]>=pt[ov].max()-E: up[i]=True
        elif yh[i]<=pb[ov].min()+E: pass
        else: return None
    q=p.copy(); q[up,1]-=1.0
    return q[~inp]
def T(p): return np.c_[p[:,1],p[:,0],-p[:,2]]
def shrinks(p, max_out=3000):
    p=norm(p); res=[]
    for rp in row_paths(p):
        q=remove_row(p,rp)
        if q is None: continue
        qt=T(norm(q))
        for cp in row_paths(qt):
            r=remove_row(qt,cp)
            if r is not None: res.append(T(r))
            if len(res)>=max_out: return res
    return res
if __name__=='__main__':
    P=pickle.load(open('best.pkl','rb'))
    Ls={n:side(P[n]) for n in P}
    lo,hi=int(sys.argv[1]),int(sys.argv[2])
    found={}; stats=0
    for n in range(lo,hi):
        outs=shrinks(P[n]); stats+=len(outs)>0
        best={}
        for q in outs:
            m=len(q); s=side(q)
            if m not in best or s<best[m][0]: best[m]=(s,q)
        for m,(s,q) in best.items():
            for mm in range(m,max(m-80,0),-1):
                if Ls[mm]>s+1e-9:
                    g=(Ls[mm]-s)/math.sqrt(mm)
                    if mm not in found or s<found[mm][0]: found[mm]=(s,n,m,g,q)
    print(f'range {lo}-{hi}: packings with a removable row+col: {stats}; improved n: {len(found)} gain {sum(v[3] for v in found.values()):.6f}')
    for mm,v in sorted(found.items()): print(' ',mm,'ours',round(Ls[mm],6),'->',round(v[0],6),'from',v[1],'->',v[2],'gain',round(v[3],6))
    pickle.dump(found,open(f'sh2_{lo}_{hi}.pkl','wb'))
