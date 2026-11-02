import sys, math, numpy as np, json
from blockmis import solve
n=int(sys.argv[1]); L=float(sys.argv[2]); part=int(sys.argv[3]); nparts=int(sys.argv[4])
dims=json.loads(sys.argv[5]); c0=float(sys.argv[6])
ths=np.radians(np.arange(41,49.01,0.25))[part::nparts]
best=(0,)
for th in ths:
    for a,b in dims:
        for cx in np.arange(c0-0.6,c0+0.61,0.1):
            for cy in np.arange(c0-0.6,c0+0.61,0.1):
                c,_=solve(L,th,a,b,cx,cy,time_limit=5)
                if c>best[0]: best=(c,round(math.degrees(th),3),a,b,round(cx,2),round(cy,2))
    print('th',round(math.degrees(th),2),'best so far',best,flush=True)
print('FINAL',best)
