"""Build the growth visualization page from grow_frames.py output (+ optional results HTML snippet)."""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--ref", required=True, help="JSON {x, y, t, L} of the record packing, box scaled to 1")
    p.add_argument("--results", default="", help="HTML snippet with the sweep tables")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tpl = (Path(__file__).parent.parent / "web" / "grow_viz_template.html").read_text()
    data = json.load(open(a.frames))
    html = (tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__REF__", Path(a.ref).read_text())
            .replace("__N__", str(data["runs"][0]["n"]))
            .replace("__RESULTS__", Path(a.results).read_text() if a.results else ""))
    Path(a.out).write_text(html)
