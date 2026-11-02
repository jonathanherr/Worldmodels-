"""Any-angle single tilted block (a x b unit squares at angle th) + axis rows filled from both walls.
count(L, th, a, b, cx, cy) = a*b + best row split; rows below split anchored at bottom, above at top."""
import numpy as np, math
M=1e-9
def verts(a,b,th):
    c,s=math.cos(th),math.sin(th)
    loc=np.array([[-a/2,-b/2],[a/2,-b/2],[a/2,b/2],[-a/2,b/2]])
    return loc@np.array([[c,s],[-s,c]])
def slab(V, ylo, yhi):
    """x-extent of convex polygon V (rel. coords) inside slab ylo<y<yhi (arrays). nan if empty."""
    xs=[]
    for i in range(4):
        (x1,y1),(x2,y2)=V[i],V[(i+1)%4]
        # vertex inside slab
        inside=(y1>ylo)&(y1<yhi)
        xs.append(np.where(inside,x1,np.nan))
        if abs(y2-y1)>1e-15:
            for yl in (ylo,yhi):
                t=(yl-y1)/(y2-y1); ok=(t>=0)&(t<=1)
                xs.append(np.where(ok,x1+t*(x2-x1),np.nan))
    X=np.array(xs)
    return np.nanmin(X,0), np.nanmax(X,0)
def best_count(L, th, a, b, CX, CY):
    """CX, CY broadcastable grids of block centres. Returns count array."""
    V=verts(a,b,th); K=int(math.floor(L+1e-12))
    vx=V[:,0]; vy=V[:,1]
    inside=(CX+vx.min()>=M)&(CX+vx.max()<=L-M)&(CY+vy.min()>=M)&(CY+vy.max()<=L-M)
    full=math.floor(L)
    def rowcells(y0):
        lo,hi=slab(V,np.broadcast_to(y0-CY-M,np.broadcast(CX,CY).shape),np.broadcast_to(y0+1-CY+M,np.broadcast(CX,CY).shape))
        hit=~np.isnan(lo)
        left=np.floor(np.where(hit,CX+lo-M,0)); right=np.floor(np.where(hit,L-(CX+hi)-M,0))
        return np.where(hit,np.maximum(left,0)+np.maximum(right,0),full)
    bot=np.array([rowcells(float(i)) for i in range(K)])
    top=np.array([rowcells(L-1-i) for i in range(K)])
    cb=np.concatenate([np.zeros((1,)+bot.shape[1:]),np.cumsum(bot,0)])
    ct=np.concatenate([np.zeros((1,)+top.shape[1:]),np.cumsum(top,0)])
    tot=np.max(np.array([cb[h]+ct[K-h] for h in range(K+1)]),0)
    return np.where(inside,tot+a*b,-1)
if __name__=='__main__':
    L=19.656854249492+1e-7; g=np.arange(7,11,0.01)
    CX,CY=np.meshgrid(g,g)
    c=best_count(L,math.pi/4,15,8,CX,CY); print('366 model max count at 45deg 15x8:',c.max())
    c=best_count(12.656854249492+1e-7,math.pi/4,8,8,*np.meshgrid(np.arange(4,9,0.01),np.arange(4,9,0.01))); print('147 model:',c.max())
