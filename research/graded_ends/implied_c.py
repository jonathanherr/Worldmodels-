import csv, math, json, sys
from scipy.optimize import brentq
from fractions import Fraction
def phi_for(r, f):
    g=lambda p:(r+1)*math.cos(p)+math.sin(p)-(r+f)
    lo=math.atan(1/(r+1))
    if g(lo)<0: return None
    return brentq(g, lo, 1.5)
def implied(k,m,f):
    """max over r of end cost per row per end c that makes one-band model tight at side k+f"""
    best=None
    for r in range(3,k):
        p=phi_for(r,f)
        if p is None: continue
        W=2*k*f+f*f-m
        c=(W-f*(k-r)-(k+f)*math.sin(p))/(2*(r+f))
        if best is None or c>best[0]: best=(c,r,math.degrees(p))
    return best
R=list(csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv')))
out=[]
for x in R:
    n=int(x['n']); k=math.isqrt(n); m=n-k*k
    if m==0 or float(x['score_gap'])<=0: continue
    for who in ('best_L','our_L'):
        f=float(x[who])-k
        if f>=1: continue
        b=implied(k,m,f)
        if b: out.append(dict(n=n,k=k,m=m,who=who,f=f,c=b[0],r=b[1],phi=b[2]))
json.dump(out,open('/tmp/claude-0/ge/implied.json','w'))
import statistics as st
for who in ('best_L','our_L'):
    o=[d for d in out if d['who']==who and d['m']<=0.6*(2*d['k']+1)]
    print(who,len(o),'c median',round(st.median(d['c'] for d in o),4))
    # bucket by phi
    for lo in range(8,30,3):
        b=[d['c'] for d in o if lo<=d['phi']<lo+3]
        if b: print('  phi %d-%d: n=%d c med %.3f min %.3f max %.3f'%(lo,lo+3,len(b),st.median(b),min(b),max(b)))
    for lo in range(16,32,3):
        b=[d['c'] for d in o if lo<=d['k']<lo+3]
        if b: print('  k %d-%d: n=%d c med %.3f'%(lo,lo+3,len(b),st.median(b)))
