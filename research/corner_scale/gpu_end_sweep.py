"""Run gpu_end over an (r, f) grid, one gadget at a time on the GPU, and print a summary table.

usage: python -m research.corner_scale.gpu_end_sweep minutes_each outdir [B] [r0 r1] [f,f,...]
"""
import sys, os, json
from research.corner_scale.gpu_end import run

if __name__ == '__main__':
    minutes, outdir = float(sys.argv[1]), sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    r0, r1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (7, 16)
    fs = [float(v) for v in sys.argv[6].split(',')] if len(sys.argv) > 6 else [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for r in range(r0, r1 + 1):
        for f in fs:
            w0, wb = run(r, f, minutes, outdir, B)
            H = r + f
            rows.append((r, f, w0 / H, (wb / H) if wb is not None else None))
            with open(os.path.join(outdir, f'summary_r{r0}-{r1}.json'), 'w') as fh:
                json.dump(rows, fh)
    print('r     f   dp/row  gpu/row')
    for r, f, a, b in rows:
        print(f'{r:2d} {f:.2f}  {a:.4f}  ' + (f'{b:.4f}' if b is not None else '  -'))
