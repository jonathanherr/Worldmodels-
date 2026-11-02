import json,math
EXT=4
rows=json.load(open('sides.json'))
O={};R={}
for n,k,d,L,v,r in rows:
    K=math.isqrt(n); D=n-K*K; O[(K,D)]=L-K; R[(K,D)]=(r or L)-K
for M,name in ((O,'ours'),(R,'leaders')):
    byf={}
    for kd,f in M.items():
        if 1e-9<f<1-1e-9: byf.setdefault(round(f,8),[]).append(kd)
    cand={}
    for f,pts in byf.items():
        pts=sorted(pts)
        for i in range(len(pts)):
            for j in range(i+1,len(pts)):
                (K1,D1),(K2,D2)=pts[i],pts[j]
                if K2==K1: continue
                for K in range(K1+1,K2+EXT):
                    if K in (K1,K2): continue
                    num=(D2-D1)*(K-K1)
                    if num%(K2-K1): continue
                    D=D1+num//(K2-K1)
                    t=(K,D)
                    if t in O and O[t]>f+1e-9:
                        g=(O[t]-f)/math.sqrt(K*K+D)
                        if g>cand.get(t,(0,))[0]: cand[t]=(g,f,(K1,D1),(K2,D2), 'interp' if K1<K<K2 else 'extrap')
    tot=sum(v[0] for v in cand.values())
    print(name,'targets',len(cand),'max gain',round(tot,5))
    for t,v in sorted(cand.items(),key=lambda x:-x[1][0])[:25]: print('  n',t[0]**2+t[1],t,'ours',round(O[t],6),'->',round(v[1],6),v[2:],'leaders',round(R[t],6))
