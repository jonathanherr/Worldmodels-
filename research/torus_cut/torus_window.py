# Torus (wall-free) packings of unit squares are tilted lattices of rows that may slide.
# Cut one open: place a box window of side L, keep squares fully inside, each row slid
# to its best offset.  Returns the max count per L.
import numpy as np, json, sys
def counts(L, th, o):
    # th: (T,), o: (O,)  -> count (T,O)
    c=np.cos(th)[:,None,None]; s=np.sin(th)[:,None,None]
    kmax=int(np.ceil(L*1.5))+2
    k=np.arange(-kmax,kmax)[None,None,:]
    a=k+o[None,:,None]          # strip v-range [a,a+1]
    tot=np.zeros((len(th),len(o)))
    lo=-1e9; hi=1e9
    # constraints for both s=a and s=a+1
    los=[];his=[]
    for sv in (a,a+1):
        los.append(sv*s/c)                  # x>=0
        his.append((L+sv*s)/c-1)            # x<=L
        los.append(-sv*c/s)                 # y>=0
        his.append((L-sv*c)/s-1)            # y<=L
    tmin=np.maximum.reduce(los); tmax=np.minimum.reduce(his)
    n=np.floor(tmax-tmin+1+1e-12); n=np.clip(n,0,None)
    return n.sum(-1)
if __name__=="__main__":
  pass
th=np.radians(np.linspace(0.5,45,180)); o=np.linspace(0,1,40,endpoint=False)
Ls=np.arange(2.0,33.0,0.005)
best={}
res=[]
for L in Ls:
    C=counts(L,th,o); i=np.unravel_index(C.argmax(),C.shape)
    res.append((L,int(C[i]),float(np.degrees(th[i[0]])),float(o[i[1]])))
json.dump(res,open(sys.argv[1],'w'))
