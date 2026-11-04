# Build the cut torus window explicitly and greedily add axis-aligned unit squares in the
# leftover border space (the "seam repair"). Returns total count.
import numpy as np
def tilted(L,th,o,align=0.0):
    c,s=np.cos(th),np.sin(th); u=np.array([c,s]); v=np.array([-s,c]); out=[]
    for k in range(-int(2*L)-2,int(2*L)+2):
        a=k+o; los=[];his=[]
        for sv in (a,a+1):
            los+= [sv*s/c, -sv*c/s]; his+=[(L+sv*s)/c-1,(L-sv*c)/s-1]
        tmin=max(los); tmax=min(his)
        if tmax<tmin: continue
        m=int(np.floor(tmax-tmin+1+1e-12)); slack=(tmax-tmin+1)-m
        t0=tmin+align*slack
        for j in range(m):
            out.append((t0+j+0.5)*u+(a+0.5)*v)
    return np.array(out).reshape(-1,2)
def corners(cx,cy,ang):
    c,s=np.cos(ang),np.sin(ang); d=np.array([[.5,.5],[-.5,.5],[-.5,-.5],[.5,-.5]])
    R=np.array([[c,-s],[s,c]]); return d@R.T+[cx,cy]
def overlap_axis(px,py,T,th):
    # axis square [px,px+1]x[py,py+1] vs tilted squares centred T, angle th (SAT, strict)
    if len(T)==0: return np.zeros(0,bool)
    c,s=np.cos(th),np.sin(th); h=0.5*(abs(c)+abs(s))
    cx,cy=px+.5,py+.5
    ok=(np.abs(T[:,0]-cx)<0.5+h-1e-9)&(np.abs(T[:,1]-cy)<0.5+h-1e-9)
    # project onto tilted axes
    dx=T[:,0]-cx; dy=T[:,1]-cy
    for ax in ((c,s),(-s,c)):
        p=np.abs(dx*ax[0]+dy*ax[1]); r=0.5+0.5*(abs(ax[0])+abs(ax[1]))
        ok&=p<r-1e-9
    return ok
def repair(L,th,o,align=0.0,step=0.02):
    T=tilted(L,th,o,align); A=[]
    grid=np.arange(0,L-1+1e-12,step)
    cand=[(x,y) for x in grid for y in grid if min(x,y,L-1-x,L-1-y)<1.6]
    cand.sort(key=lambda p:(min(p[0],p[1],L-1-p[0],L-1-p[1]),p[1],p[0]))
    for px,py in cand:
        if overlap_axis(px,py,T,th).any(): continue
        if any(abs(px-q[0])<1-1e-9 and abs(py-q[1])<1-1e-9 for q in A): continue
        A.append((px,py))
    return len(T),len(A)
