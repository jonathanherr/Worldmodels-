import json,numpy as np,sys
from multiprocessing import Pool
exec(open('torus_window.py').read().split('th=np.radians')[0])
from repair import repair
rows=json.load(open('deficit.json'))
def job(r):
    n,L,c,d,_,_=r
    th=np.radians(np.linspace(1,45,177)); o=np.linspace(0,1,40,endpoint=False)
    C=counts(L,th,o); idx=np.argsort(C.ravel())[::-1][:12]
    best=(0,)
    for k in idx:
        i,j=np.unravel_index(k,C.shape)
        for al in (0.0,0.5,1.0):
            t,a=repair(L,th[i],o[j],al,step=0.05)
            if t+a>best[0]: best=(t+a,t,a,float(np.degrees(th[i])),float(o[j]),al)
    return (n,L,best)
cand=[r for r in rows if r[0]<=400 and r[3]<=0.3*r[1]]
print(len(cand),flush=True)
with Pool(4) as p:
    out=p.map(job,cand)
json.dump(out,open('repair_out.json','w'))
w=[o for o in out if o[2][0]>=o[0]]
print('wins',w)
print('closest',sorted(out,key=lambda o:o[0]-o[2][0])[:8])
