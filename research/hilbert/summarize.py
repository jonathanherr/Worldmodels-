import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # trials, repaired, back-to-L0, worse, wins
for r in rows:
    a = agg[(r["method"], r["m"])]
    a[0] += 1
    if r["L"] is not None:
        a[1] += 1
        d = r["L"] - r["L0"]
        if d < -1e-9: a[4] += 1
        elif d < 1e-6: a[2] += 1
        else: a[3] += 1
print("method   m  trials repaired same_L worse wins")
for (meth, m), a in sorted(agg.items()):
    print(f"{meth:8s}{m:3d}{a[0]:7d}{a[1]:9d}{a[2]:7d}{a[3]:6d}{a[4]:5d}")
