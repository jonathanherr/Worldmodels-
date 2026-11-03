"""Waste per unit length of a straight seam between an axis grid (below) and a 45-degree grid (above)."""
import math, sys, itertools
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import affinity

def run(psi_deg, W=40.0, M=4.0, grid=10, cgrid=6):
    psi=math.radians(psi_deg); c,s=math.cos(psi),math.sin(psi)
    # work in seam frame: u along, v perp. axis cells rotated by -psi, 45 cells rotated by 45-psi
    def cell(cx,cy,ang):
        h=.5; ca,sa=math.cos(ang),math.sin(ang)
        return Polygon([(cx+ca*dx-sa*dy, cy+sa*dx+ca*dy) for dx,dy in((h,h),(-h,h),(-h,-h),(h,-h))])
    R=int(W+2*M)+4
    band=box(0,-M,W,M)
    best=(9,None)
    for c0 in np.arange(cgrid)/cgrid:
        # axis lattice points in world, transform to seam frame
        axis=[]
        for i in range(-R,R):
            for j in range(-R,R):
                x,y=i+.5,j+.5+c0
                u,v=x*c+y*s,-x*s+y*c
                if -2<u<W+2 and -M-1<v<1:
                    p=cell(u,v,-psi)
                    if p.bounds[3]<=0: axis.append(p)
        tree_a=STRtree(axis)
        for a1,a2 in itertools.product(np.arange(grid)/grid,repeat=2):
            e1=np.array([1,1])/math.sqrt(2); e2=np.array([-1,1])/math.sqrt(2)
            rot=[]
            for i in range(-2*R,2*R):
                for j in range(-2*R,2*R):
                    x,y=(i+a1)*e1+(j+a2)*e2
                    u,v=x*c+y*s,-x*s+y*c
                    if -2<u<W+2 and -1.5<v<M+1:
                        p=cell(u,v,math.pi/4-psi)
                        hits=tree_a.query(p)
                        if all(not p.intersects(axis[k]) or p.intersection(axis[k]).area<1e-9 for k in hits):
                            rot.append(p)
            cov=unary_union(axis+rot).intersection(band).area
            waste=(band.area-cov)/W
            if waste<best[0]: best=(waste,(c0,a1,a2))
    return best
if __name__=='__main__':
    for p in map(float,sys.argv[1:]):
        w,o=run(p); print(f'psi {p:6.2f}  waste/len {w:.4f}  offsets {o}',flush=True)
