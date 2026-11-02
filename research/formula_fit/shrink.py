"""Band removal: delete a clean height-1 row band and a clean width-1 column band from a packing.
Side drops by 1; n drops by the squares in the two bands."""
import numpy as np, pickle, math, json, sys, itertools
TOL=1e-9
def hext(a): return (np.abs(np.cos(a))+np.abs(np.sin(a)))/2
def is_axis(a): return np.abs(np.sin(2*a))<1e-12
def norm(p):
    p=p.copy(); h=hext(p[:,2]); p[:,0]-=(p[:,0]-h).min(); p[:,1]-=(p[:,1]-h).min(); return p
def side(p):
    h=hext(p[:,2]); return max(np.ptp(np.r_[p[:,0]-h,p[:,0]+h]),np.ptp(np.r_[p[:,1]-h,p[:,1]+h]))
def bands(p, ax):
    """Clean bands along axis ax (1 = horizontal row band in y). Returns list of (y0, mask_in_band)."""
    c=p[:,ax]; h=hext(p[:,2]); lo=c-h; hi=c+h; axm=is_axis(p[:,2])
    out=[]; seen=[]
    for y0 in np.unique(np.round(lo[axm],7)):
        if any(abs(y0-s)<1e-7 for s in seen): continue
        seen.append(y0)
        inter=(hi>y0+TOL)&(lo<y0+1-TOL)
        inb=inter&axm&(np.abs(lo-y0)<1e-7)
        if inter.sum()==inb.sum() and inb.sum()>0: out.append((y0,inb))
    return out
def remove(p, ax, y0, inb):
    q=p[~inb].copy(); h=hext(q[:,2])
    up=(q[:,ax]-h)>=y0+1-1e-7
    q[up,ax]-=1.0
    return q
def shrink_all(p, depth=1):
    """All packings from removing `depth` row bands and `depth` column bands."""
    res=[]
    def rec(q, rows_left, cols_left):
        if rows_left==0 and cols_left==0:
            res.append(q); return
        if rows_left:
            for y0,m in bands(q,1): rec(remove(q,1,y0,m), rows_left-1, cols_left)
        else:
            for x0,m in bands(q,0): rec(remove(q,0,x0,m), 0, cols_left-1)
    rec(norm(p), depth, depth)
    return res
if __name__=='__main__':
    P=pickle.load(open('best.pkl','rb'))
    Ls={n:side(P[n]) for n in P}
    depth=int(sys.argv[1]) if len(sys.argv)>1 else 1
    found={}
    for n in sorted(P):
        if n<10: continue
        L=Ls[n]
        outs=shrink_all(P[n],depth)
        best={}
        for q in outs:
            m=len(q); s=side(q)
            if m not in best or s<best[m][0]: best[m]=(s,q)
        for m,(s,q) in best.items():
            # improves any m' <= m with Ls[m'] > s
            for mm in range(m,0,-1):
                if Ls.get(mm,0)>s+1e-9:
                    g=(Ls[mm]-s)/math.sqrt(mm)
                    if mm not in found or s<found[mm][0]: found[mm]=(s,n,m,g)
                elif mm<m-60: break
    tot=sum(v[3] for v in found.values())
    print('depth',depth,'improved n:',len(found),'gain',tot)
    for mm,v in sorted(found.items()): print(mm, 'ours',round(Ls[mm],6),'->',round(v[0],6),'from n',v[1],'(shrunk to',v[2],') gain',round(v[3],6))
    pickle.dump(found,open(f'shrink_found_d{depth}.pkl','wb'))
