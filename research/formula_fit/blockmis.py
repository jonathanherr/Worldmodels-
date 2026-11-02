"""Block + 4 corner-anchored axis lattices, maximum non-overlapping subset via MILP."""
import numpy as np, math
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from blockfit import verts
M=1e-9
def sat_free(cells, P):
    """cells: (N,2) lower-left corners of axis unit squares; P: polygon verts (4,2) absolute. True if no overlap."""
    # separating axes: x, y, and polygon edge normals
    ok=np.zeros(len(cells),bool)
    cx0=cells[:,0]; cy0=cells[:,1]
    ok|=(cx0+1<=P[:,0].min()+M)|(cx0>=P[:,0].max()-M)|(cy0+1<=P[:,1].min()+M)|(cy0>=P[:,1].max()-M)
    for i in range(2):
        e=P[(i+1)%4]-P[i]; nrm=np.array([-e[1],e[0]])/np.hypot(*e)
        pp=P@nrm; lo,hi=pp.min(),pp.max()
        corners=np.stack([cells,cells+[1,0],cells+[0,1],cells+[1,1]],0)@nrm
        ok|=(corners.max(0)<=lo+M)|(corners.min(0)>=hi-M)
    return ok
def solve(L, th, a, b, cx, cy, time_limit=20):
    V=verts(a,b,th)+[cx,cy]
    if V.min()<M or V.max()>L-M: return -1,None
    K=int(math.floor(L)); f=L-K
    cand=[]
    for ox in (0.0,f):
        for oy in (0.0,f):
            I,J=np.meshgrid(np.arange(K),np.arange(K)); c=np.c_[I.ravel()+ox,J.ravel()+oy]
            cand.append(c)
    C=np.unique(np.round(np.vstack(cand),12),axis=0)
    C=C[sat_free(C,V)]
    # conflicts: overlapping cells
    dx=np.abs(C[:,None,0]-C[None,:,0]); dy=np.abs(C[:,None,1]-C[None,:,1])
    conf=np.triu((dx<1-M)&(dy<1-M),1)
    ii,jj=np.nonzero(conf)
    N=len(C)
    if len(ii):
        A=coo_matrix((np.ones(2*len(ii)),(np.r_[np.arange(len(ii)),np.arange(len(ii))],np.r_[ii,jj])),shape=(len(ii),N))
        cons=[LinearConstraint(A,-np.inf,1)]
    else: cons=[]
    r=milp(-np.ones(N),constraints=cons,integrality=np.ones(N),bounds=Bounds(0,1),options={'time_limit':time_limit})
    x=np.round(r.x).astype(bool)
    return int(x.sum())+a*b, (C[x],V)
if __name__=='__main__':
    L=19.656854249492+1e-7
    best=0
    for cx in np.arange(8.6,9.1,0.05):
        for cy in np.arange(8.6,9.1,0.05):
            c,_=solve(L,math.pi/4,15,8,cx,cy)
            best=max(best,c)
    print('366 4-lattice MIS best',best)
