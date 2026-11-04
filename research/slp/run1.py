import sys,time,numpy as np
sys.path.insert(0,'/home/claude/worldmodels-/research/slp')
from slp import slp
D=np.load('/home/claude/work/cur.npz')
for n in sys.argv[1:]:
    a=D[n]; t0=time.time()
    x,y,t,L,v,L0=slp(a[:,0],a[:,1],a[:,2],iters=int(200),verbose=True,log_every=25)
    print(n,'L0 %.9f -> %.9f  v %.1e  gain %.2e  %.0fs'%(L0,L,v,L0-L,time.time()-t0),flush=True)
