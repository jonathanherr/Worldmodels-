import pandas as pd, numpy as np, sys, json
df=pd.read_csv(sys.argv[1])
out={}
for n,g in df.groupby('n'):
    x=g.x.str.lstrip('s').astype(float).values if g.x.dtype==object else g.x.values
    y=g.y.str.lstrip('s').astype(float).values if g.y.dtype==object else g.y.values
    a=g.angle.str.lstrip('s').astype(float).values if g.angle.dtype==object else g.angle.values
    c,s=np.cos(a),np.sin(a); h=0.5*(np.abs(c)+np.abs(s))
    L=max((x+h).max()-(x-h).min(),(y+h).max()-(y-h).min())
    out[int(n)]=L
json.dump(out,open(sys.argv[2],'w'))
print(sum(out[n]/np.sqrt(n)-1 for n in out))
