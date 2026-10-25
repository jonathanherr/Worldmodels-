"""The swarm's shared board: a folder of small JSON posts that every agent reads and writes.

Layout under the board directory (`$SWARM_BOARD`, default `swarm_board/` in the repo):

    posts/<time>-<agent>-<id>.json   one post per file, written via temp file + rename

A post is {"id", "ts", "agent", "persona", "kind", "text", ...}. Kinds:

* recipe      a tool pipeline (`recipe`, `recipe_id`), with `parents` (recipe ids it was built from)
              and `why` (the author's reasoning, from its theoretical grounding)
* hypothesis  a claim that a recipe helps in an n band (`recipe_id`, `band`); the board scores it
              from results as open / supported / refuted
* claim       "I am working on n" (`n`, `until`): others steer away unless they want to help
* result      one run of a recipe on one n (`n`, `recipe_id`, `before`, `side`, `accepted`, `gain`)
* note        free text, e.g. from a person or a Claude session acting as an agent

One file per post means writers never contend for a file, which works on a network share and
on a local disk alike; readers cache what they have parsed and only read new files. `digest()`
summarizes the board (agents, recipes, hypotheses, wins) for people and for agents choosing what
to try next.
"""

import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

REPO_BOARD = Path(__file__).resolve().parent.parent.parent / "swarm_board"


def board_dir() -> Path:
    return Path(os.environ.get("SWARM_BOARD", REPO_BOARD))


class Board:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else board_dir()
        self.posts_dir = self.root / "posts"
        self.posts_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ writing

    def post(self, agent: str, kind: str, text: str = "", persona: str = "", **fields) -> dict:
        p = {"id": uuid.uuid4().hex[:12], "ts": time.time(), "agent": agent, "persona": persona,
             "kind": kind, "text": text, **fields}
        name = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(p['ts']))}-{agent}-{p['id']}.json"
        tmp = self.posts_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(p, default=_jsonable))
        os.replace(tmp, self.posts_dir / name)
        self._cache[name] = p
        return p

    def claim(self, agent: str, n: int, minutes: float, persona: str = "", text: str = "") -> dict:
        return self.post(agent, "claim", text or f"working on n={n}", persona, n=n, until=time.time() + 60 * minutes)

    # ------------------------------------------------------------------ reading

    def posts(self, kind: str | None = None, since: float = 0.0) -> list[dict]:
        for f in self.posts_dir.iterdir():
            if f.name.startswith(".") or f.name in self._cache or not f.name.endswith(".json"):
                continue
            try:
                self._cache[f.name] = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):  # half-synced file on a lazy share; next read gets it
                continue
        out = [p for p in self._cache.values() if (kind is None or p["kind"] == kind) and p["ts"] >= since]
        return sorted(out, key=lambda p: p["ts"])

    def recipes(self) -> dict[str, dict]:
        """recipe_id -> the first post that published it."""
        out = {}
        for p in self.posts("recipe"):
            out.setdefault(p["recipe_id"], p)
        return out

    def active_claims(self, exclude_agent: str | None = None) -> dict[int, list[str]]:
        now = time.time()
        out = defaultdict(list)
        for p in self.posts("claim", since=now - 24 * 3600):
            if p["until"] > now and p["agent"] != exclude_agent:
                out[p["n"]].append(p["agent"])
        return dict(out)

    def recipe_stats(self) -> dict[str, dict]:
        """Per recipe: runs, wins (store improvements), total score gain, failures, seconds,
        and `rel` = mean relative side change vs the store side when it ran (negative is good)."""
        st = defaultdict(lambda: {"runs": 0, "wins": 0, "gain": 0.0, "errors": 0, "seconds": 0.0,
                                  "rel_sum": 0.0, "rel_n": 0, "ns": set(), "win_ns": []})
        for r in self.posts("result"):
            s = st[r["recipe_id"]]
            s["runs"] += 1
            s["seconds"] += r.get("seconds", 0.0)
            s["ns"].add(r["n"])
            if r.get("error"):
                s["errors"] += 1
            elif r.get("side"):
                s["rel_sum"] += r["side"] / r["before"] - 1
                s["rel_n"] += 1
            if r.get("accepted"):
                s["wins"] += 1
                s["gain"] += r.get("gain", 0.0)
                s["win_ns"].append(r["n"])
        for s in st.values():
            s["rel"] = s["rel_sum"] / s["rel_n"] if s["rel_n"] else math.inf
        return dict(st)

    def hypothesis_status(self, h: dict, stats: dict | None = None, min_runs: int = 12) -> str:
        lo, hi = h["band"]
        rs = [r for r in self.posts("result") if r["recipe_id"] == h["recipe_id"] and lo <= r["n"] <= hi]
        if any(r.get("accepted") for r in rs):
            return "supported"
        return "refuted" if len(rs) >= min_runs else "open"

    # ------------------------------------------------------------------ summary

    def digest(self, top: int = 15) -> str:
        stats = self.recipe_stats()
        recipes = self.recipes()
        results = self.posts("result")
        agents = defaultdict(lambda: {"runs": 0, "wins": 0, "gain": 0.0, "persona": ""})
        for r in results:
            a = agents[r["agent"]]
            a["runs"] += 1
            a["persona"] = r.get("persona", "")
            if r.get("accepted"):
                a["wins"] += 1
                a["gain"] += r.get("gain", 0.0)
        lines = [f"# Swarm board digest ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
                 f"{len(results)} runs, {sum(a['wins'] for a in agents.values())} store wins, "
                 f"total score gain {sum(a['gain'] for a in agents.values()):.5f}, {len(recipes)} recipes.", "",
                 "## Agents", "", "| agent | persona | runs | wins | gain |", "|---|---|---|---|---|"]
        for name, a in sorted(agents.items(), key=lambda kv: -kv[1]["gain"]):
            lines.append(f"| {name} | {a['persona']} | {a['runs']} | {a['wins']} | {a['gain']:.5f} |")
        lines += ["", "## Recipes (by gain, then mean side change)", "",
                  "| id | author | runs | wins | gain | mean dside | parents | recipe |", "|---|---|---|---|---|---|---|---|"]
        ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["gain"], kv[1]["rel"]))[:top]
        for rid, s in ranked:
            rp = recipes.get(rid, {})
            lines.append(f"| {rid} | {rp.get('agent', '?')} | {s['runs']} | {s['wins']} | {s['gain']:.5f} | "
                         f"{s['rel']:+.5f} | {','.join(rp.get('parents', [])) or '-'} | "
                         f"`{json.dumps(rp.get('recipe'), separators=(',', ':'))[:160]}` |")
        hyps = self.posts("hypothesis")
        if hyps:
            lines += ["", "## Hypotheses", ""]
            for h in hyps[-top:]:
                lines.append(f"- [{self.hypothesis_status(h, stats)}] {h['agent']}: {h['text']} "
                             f"(recipe {h['recipe_id']}, n {h['band'][0]}-{h['band'][1]})")
        wins = [r for r in results if r.get("accepted")]
        if wins:
            lines += ["", "## Store wins", "", "| n | before | after | agent | recipe |", "|---|---|---|---|---|"]
            for r in wins[-top:]:
                lines.append(f"| {r['n']} | {r['before']:.6f} | {r['side']:.6f} | {r['agent']} | {r['recipe_id']} |")
        notes = self.posts("note")
        if notes:
            lines += ["", "## Notes", ""] + [f"- {p['agent']}: {p['text']}" for p in notes[-top:]]
        return "\n".join(lines) + "\n"

    def write_digest(self) -> Path:
        p = self.root / "digest.md"
        tmp = self.root / ".digest.md.tmp"
        tmp.write_text(self.digest())
        os.replace(tmp, p)
        return p


def _jsonable(o):
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))
