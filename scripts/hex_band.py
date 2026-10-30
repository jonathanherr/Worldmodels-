# Usage: python3 scripts/hex_band.py sides.json KMIN KMAX  (sides.json = {n: best side}); writes best_KMIN_KMAX.json
"""Grid block A x B in the bottom-left corner + ONE tilted sheared ("brick") lattice
filling the rest of the L x L box, running seamlessly round the corner.
For each config, sorted needed-L gives L_min for every count; we track best L per n."""
import numpy as np, json, math, sys, itertools
EPS=1e-8; P=1+1e-8
sides={int(k):v for k,v in json.load(open(sys.argv[1])).items()}
kmin,kmax=int(sys.argv[2]),int(sys.argv[3])
rng=np.random.default_rng(0)

def lattice(th,s,t,R):
    c,sn=math.cos(th),math.sin(th); u=np.array([c,sn]); v=np.array([-sn,c])
    I,J=np.meshgrid(np.arange(-R,R+1),np.arange(-R,R+1),indexing='ij')
    I=I.ravel()+t[0]+s*J.ravel(); J=J.ravel()+t[1]
    pts=P*(I[:,None]*u+J[:,None]*v)
    return pts,u,v,(abs(c)+abs(sn))/2

def base(th,s,t,k):
    pts,u,v,h=lattice(th,s,t,int(1.5*k)+3)
    x,y=pts[:,0],pts[:,1]
    ok=(x>=h+EPS)&(y>=h+EPS)&(np.maximum(x,y)+h<=k)
    return pts[ok],u,v,h
def needL(bs,A,B):
    pts,u,v,h=bs; x,y=pts[:,0],pts[:,1]
    if A>0 and B>0:
        C=np.array([[0,0],[A,0],[0,B],[A,B]],float)
        pu,pv=pts@u,pts@v; cu,cv=C@u,C@v
        sep=(x-h>=A+EPS)|(y-h>=B+EPS)|(pu-.5>=cu.max()+EPS)|(pu+.5<=cu.min()-EPS)|(pv-.5>=cv.max()+EPS)|(pv+.5<=cv.min()-EPS)
        x,y=x[sep],y[sep]
    return np.sort(np.maximum(x,y)+h+EPS)

best={}
for k in range(kmin,kmax+1):
    ns=[n for n in range((k-1)**2+1,k*k+1) if n<=1000 and sides[n]<k-1e-9]
    if not ns: continue
    f=k-1
    combos=[(0,0)]+[(a,b) for a in range(max(1,f-4),f+1) for b in range(a,f+1)]
    for th in np.deg2rad(np.arange(0.5,45.01,1.0)):
        for s in np.linspace(0,1,8,endpoint=False):
            for t in itertools.product(np.linspace(0,1,5,endpoint=False),repeat=2):
                bs=base(th,s,t,k)
                for (A,B) in combos:
                    nl=needL(bs,A,B)
                    for n in ns:
                        m=n-A*B
                        if m<=0: L=max(A,B)
                        elif m<=len(nl): L=max(nl[m-1],A,B)
                        else: continue
                        if L<best.get(n,(9e9,))[0]: best[n]=(L,A,B,float(th),float(s),t)
    w=[n for n in ns if n in best and best[n][0]<sides[n]]
    print(k,len(ns),'wins',w,[(n,round(best[n][0],4),round(sides[n],4)) for n in ns[:3] if n in best],flush=True)
json.dump({n:[v[0],v[1],v[2],v[3],v[4],list(v[5])] for n,v in best.items()},open(f'best_{kmin}_{kmax}.json','w'))
