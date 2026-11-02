import json, math, mpmath as mp
mp.mp.dps=30
rows=json.load(open('sides.json'))
R={};O={}
for n,k,d,L,v,r in rows:
    K=math.isqrt(n); D=n-K*K
    if r: R[n]=(K,D,r-K)
    O[n]=(K,D,L-K)
# repeated leader fractions (robust), where we trail
vals={}
for n,(K,D,f) in R.items():
    if O[n][2]-f>1e-7: vals.setdefault(round(f,9),[]).append(n)
for f,ns in sorted(vals.items(), key=lambda x:-len(x[1])):
    x=mp.mpf(f)
    out=[]
    for deg in (2,3,4):
        p=mp.findpoly(x,deg,maxcoeff=60,tol=1e-8)
        if p: out.append(p); break
    print(f'{f:.9f}', ns, out)
