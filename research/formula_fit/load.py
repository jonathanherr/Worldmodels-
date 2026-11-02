import csv, json, math, numpy as np, pickle
def load_csv(path):
    P={}
    with open(path) as f:
        r=csv.reader(f); next(r)
        for row in r:
            n=int(row[1]); P.setdefault(n,[]).append((float(row[3][1:] if row[3].startswith('s') else row[3]),float(row[4][1:] if row[4].startswith('s') else row[4]),float(row[5][1:] if row[5].startswith('s') else row[5])))
    return {n:np.array(v) for n,v in P.items()}
def side(p):
    h=(np.abs(np.cos(p[:,2]))+np.abs(np.sin(p[:,2])))/2
    return max(np.ptp(np.r_[p[:,0]-h,p[:,0]+h]),np.ptp(np.r_[p[:,1]-h,p[:,1]+h]))
if __name__=='__main__':
    P=load_csv('best/submission_merged_fault_20261028.csv')
    pickle.dump(P,open('best.pkl','wb'))
    T={d['n']:d for d in json.load(open('/mnt/project-files/squares/public-certs/targets.json'))}
    rows=[]
    for n in range(1,1001):
        L=side(P[n]); k=math.ceil(math.sqrt(n)); t=T.get(n,{})
        rows.append((n,k,n-(k-1)**2 if k>1 else 0,L,t.get('verified'),t.get('reported')))
    json.dump(rows,open('sides.json','w'))
    gap=[(n,L-(r or L)) for n,k,d,L,v,r in rows]
    fin=[n for n,g in gap if g<1e-7]
    print('finished (= kaggle best):',len(fin),'trailing:',1000-len(fin))
    print('sum score gap',sum(g/math.sqrt(n) for n,g in gap))
    print(sorted(gap,key=lambda x:-x[1]/math.sqrt(x[0]))[:30])
