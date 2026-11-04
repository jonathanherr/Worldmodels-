"""CSL angle catalog for the full-width tilted-column band.

Mirror-wall argument: a band whose columns lean by phi meets a vertical wall like a symmetric tilt
boundary with misorientation 2*phi. 2*phi is a coincidence-site-lattice angle of the square lattice
exactly when tan(phi) = p/q is rational (cos 2phi = (q^2-p^2)/(q^2+p^2)); Sigma = p^2+q^2 (halved if
odd*odd). At such phi the band lattice (pitch 1/cos phi, column step (sin, cos)) has an exact
vertical period, so wall/end gadgets repeat. The band height fixes f exactly:
    f(r) = ((r+1) q + p) / sqrt(p^2+q^2) - r.
"""
import math
from math import gcd

def sigma(p, q):
    s = p * p + q * q
    return s // 2 if (p % 2 and q % 2) else s

def catalog(max_sigma=100, rmin=3, rmax=30, fmin=0.40, fmax=0.98):
    out = []
    for q in range(1, 40):
        for p in range(1, q):
            if gcd(p, q) != 1 or sigma(p, q) > max_sigma: continue
            h = math.hypot(p, q)
            for r in range(rmin, rmax + 1):
                f = ((r + 1) * q + p) / h - r
                if fmin <= f <= fmax:
                    out.append(dict(p=p, q=q, sigma=sigma(p, q), phi=math.degrees(math.atan2(p, q)), r=r, f=f))
    return out

if __name__ == '__main__':
    c = catalog()
    for x in c: print(x)
    print(len(c))
