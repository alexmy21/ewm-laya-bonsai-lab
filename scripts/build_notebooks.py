#!/usr/bin/env python3
# build_notebooks.py — generate the lab notebook collection from templates.
#
# The notebooks are the deliverable of this project. They are generated
# from the cell templates below so the code stays reviewable and
# reproducible: edit this file, then
#
#   python3 scripts/build_notebooks.py
#
# The notebooks are stdlib + lab only (numpy/matplotlib are not required).

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

PROLOGUE = '''import os, sys

# make the `lab` package importable whether jupyter starts here or in the repo root
for _p in (os.getcwd(), os.path.dirname(os.getcwd())):
    if os.path.exists(os.path.join(_p, "lab")):
        sys.path.insert(0, _p)
        PROJ = _p
        break
else:
    raise RuntimeError("run jupyter from the project root or the notebooks/ directory")

from lab import EwmScene, LayaDaemon, BonsaiClient
print("project root:", PROJ)
'''


def nb(cells, title):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


_CELL_SEQ = [0]


def _next_id() -> str:
    _CELL_SEQ[0] += 1
    return f"cell-{_CELL_SEQ[0]}"


def md(text):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": [l + "\n" for l in text.split("\n")]}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _next_id(),
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in text.split("\n")],
    }


# ═══════════════════════════════════════════════════════════════════════
# 01 — the ewm-scene surface
# ═══════════════════════════════════════════════════════════════════════
nb01 = nb([
    md("""# 01 — The ewm-sm sidecar surface (`ewm-scene`)

**Question this notebook answers:** what exactly crosses the boundary
between the Python controller and the HLLSet lattice?

Every ewm-sm interaction is a subprocess call: a JSONL file goes in, one
JSON document comes out. This notebook runs each flat-frame command and
shows the raw output. Companion reference: `docs/INTERACTION_MAP.md` §1."""),
    code(PROLOGUE),
    code('''ewm = EwmScene()
print("resolved binary:", ewm.bin)'''),
    md("""## Frames in

The controller's lingua franca is `tid{n}` strings — Bonsai token ids
lowered to strings. Here is a three-frame "conversation": frame 2 keeps
`tid2`/`tid3`, drops `tid1`/`tid4`, and adds `tid5`."""),
    code('''import json, os

WORK = os.path.join(PROJ, "work", "nb01")
os.makedirs(WORK, exist_ok=True)
union = os.path.join(WORK, "frames.jsonl")

frames = [
    {"id": 1, "tokens": ["tid1", "tid2", "tid3", "tid4"]},
    {"id": 2, "tokens": ["tid2", "tid3", "tid5"]},
    {"id": 3, "tokens": ["tid5", "tid6", "tid7"]},
]
with open(union, "w") as fh:
    for f in frames:
        fh.write(json.dumps(f) + "\\n")
print(open(union).read())'''),
    md("""## `ingest` — per-frame content key and popcount

`pop` is the HLLSet popcount — roughly `3 × distinct tokens` because each
token is ingested with three seeds. `key` is the content address of the
frame (`h:<sha1…>`)."""),
    code("ewm.ingest(union, verbose=True)"),
    md("""## `bss` — consecutive similarity

`tau[i]` is BSSτ = `|A ∩ B| / |B|` between frames i and i+1 ("how much of
the new frame was already in the old one"). `jaccard[i]` is
`|A ∩ B| / |A ∪ B|`."""),
    code("ewm.bss(union, verbose=True)"),
    md("""## `noether` — the D/R/N decomposition

Per transition: `dp` (departed), `rp` (retained), `np` (new) popcounts,
plus three indicators:

- `ind1 = dp / |R ∪ N|` — departure dominates
- `ind3 = np / |R ∪ D|` — arrival dominates
- `ind2 = BSS(R_prev, R)` — retention-chain stability (a **similarity**,
  so low = structurally surprising; it starts at transition 2)"""),
    code("ewm.noether(union, verbose=True)"),
    md("""## `materialize` — token restoration

The lattice restores each frame's tokens three ways: `ordered` (greedy
restoration of order), `set` (plain set), `beam2` (beam-2). The controller
uses `ordered` for full memory.

> Edge case: `materialize` panics on frames with 1–2 tids ("attempt to
> subtract with overflow"). `ingest`/`noether` are safe on tiny frames."""),
    code("ewm.materialize(union, verbose=True)"),
    md("""## `sidecar` — the Boolean-ring trajectory

Each frame is measured against the growing GF(2) basis *before* it is
inserted: `soft` (BSS weights over basis elements), `hard` (GF(2)
coordinates when in-span), `residual` (linear novelty), `step` (soft-key
L2 distance from the previous frame), and a jump detector over the step
series (`jumps`, threshold = mean + 3σ). `--freeze N` stops basis updates
after N frames so later soft keys share one coordinate system."""),
    code("ewm.sidecar(union, freeze=3, verbose=True)"),
    md("""## `ma` — moving-average BSS over trailing windows

`fast`/`slow` are BSS between consecutive trailing-union windows of length
`--short` / `--long`. They are the cheapest "is the conversation changing
pace?" signal."""),
    code("ewm.ma(union, short=1, long=2, verbose=True)"),
], "01 — ewm-scene surface")

# ═══════════════════════════════════════════════════════════════════════
# 02 — the Laya daemon surface
# ═══════════════════════════════════════════════════════════════════════
nb02 = nb([
    md("""# 02 — The Rust Laya daemon surface (`laya-jsonl`)

**Question this notebook answers:** what exactly crosses the boundary
between the controller and the System One decision model?

Laya never generates text. You send a **state** and **typed questions**;
it returns typed answers with calibrated probabilities in one forward
pass. The `laya-jsonl` daemon keeps the checkpoint loaded and speaks
JSON-lines on stdin/stdout. Companion reference:
`docs/INTERACTION_MAP.md` §2."""),
    code(PROLOGUE),
    code('''laya = LayaDaemon()
print("daemon binary:", laya.bin)
print("checkpoint:", laya.model_dir)'''),
    md("""## A raw `choice` request

`criteria` is an object `{name: description}`. The answer comes back as a
distribution over the **names**. Notice the response answers are sorted
alphabetically by question name (the daemon uses a `BTreeMap`)."""),
    code('''resp = laya.request({
    "state": "The user asked: \\"What is the capital of France?\\". The system's memory holds 0 token ids.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "How should the next context be built for the base model?",
            "criteria": {
                "full": "keep the full materialized memory",
                "compact": "keep only new, previously unseen tids",
            },
        },
    },
}, verbose=True)'''),
    md("""## A raw `noul` request

`noul` = "yes/no, does this statement hold". The answer is P(holds), a
single number between 0 and 1."""),
    code('''laya.request({
    "state": "The user asked: \\"Explain gravity in one sentence.\\". The memory holds 24 tids.",
    "questions": {"keep_short": {"type": "noul", "instructions": "The context should be kept short"}},
}, verbose=True)'''),
    md("""## A raw `score` request

`score` is ordinal: the answer is the expectation over level indices,
with a legend and per-level probabilities."""),
    code('''laya.request({
    "state": "The user asked: \\"Write a haiku about rain.\\". The memory holds 40 tids.",
    "questions": {
        "urgency": {
            "type": "score",
            "instructions": "How much context does this query need?",
            "criteria": ["none", "a little", "a lot"],
        },
    },
}, verbose=True)'''),
    md("""## Batching — one forward pass for all questions

All questions attached to one state go through a single batched forward
pass, so asking five questions costs about as much as asking one."""),
    code('''laya.request({
    "state": "The user asked: \\"Name three planets in our solar system.\\". The memory holds 12 tids.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which context mode fits this query?",
            "criteria": {"full": "whole memory", "compact": "new tids only", "surprise": "surprising frames only"},
        },
        "keep_short": {"type": "noul", "instructions": "The context should be kept short"},
        "needs_reasoning": {"type": "noul", "instructions": "This query needs a reasoning model"},
    },
}, verbose=True)'''),
    md("""## What the model actually sees

Each question is rendered as

```text
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

Budgets from `rl_agent_config.json`: whole sequence `max_len=1024`,
instructions+options `head_max_len=256`, each option truncated to 48
tokens, the state truncated to the remaining room. `usage.input_tokens`
reports the rendered sequence length; `output_tokens` is always 0 (no
generation). Calibration temperatures are per (type, option-count)
bucket."""),
    md("""## The prose-state convention

The fine-tuned `laya-typed-decisions` checkpoint rewards natural-language
states, not raw structs. The controller renders trajectory features as
prose before asking:"""),
    code('''print(LayaDaemon.prose_state(
    "What is the capital of France?",
    memory_tokens=24,
    bss={"llm_a": 0.55, "llm_b": 0.32, "llm_c": 0.21},
))'''),
    code("laya.close()\nprint('daemon closed')"),
], "02 — Laya daemon surface")

# ═══════════════════════════════════════════════════════════════════════
# 03 — the Bonsai HTTP surface
# ═══════════════════════════════════════════════════════════════════════
nb03 = nb([
    md("""# 03 — The Bonsai HTTP surface

**Question this notebook answers:** what exactly crosses the boundary
between the controller and the PrismML llama.cpp server?

Bonsai is a reasoning model served by a llama.cpp fork. Next to the
OpenAI-compatible chat API it exposes `/tokenize` and `/detokenize` — the
bidirectional token access that makes the context a *proposal* rather than
a transcript. Companion reference: `docs/INTERACTION_MAP.md` §3."""),
    code(PROLOGUE),
    code('''bonsai = BonsaiClient()
print("base url:", bonsai.base_url)
print("health:", bonsai.health(verbose=True))'''),
    md("""## Tokenize / detokenize — the two directions

`tid{n}` memory becomes text via `/detokenize`; Bonsai's answer text
becomes `tid{n}` frames via `/tokenize`. These are plain llama.cpp
endpoints."""),
    code('''toks = bonsai.tokenize("The capital of France is Paris.", verbose=True)
print()
print("tokens:", toks)
print()
print("roundtrip:", repr(bonsai.detokenize(toks, verbose=True)))'''),
    md("""## Chat — the OpenAI-compatible endpoint with reasoning

Bonsai emits `reasoning_content` (its thinking trace) before `content`
(the answer). `usage.prompt_tokens` is the honest per-turn context cost
the controller records."""),
    code('''full = bonsai.chat_full(
    "What is the capital of France? Answer in one short sentence.",
    max_tokens=64, temperature=0.0, verbose=True,
)
print()
print("answer:", repr(full["content"]))
print("reasoning:", repr(full["reasoning"][:200]))
print("usage:", full["usage"])'''),
    md("""## If the server is DOWN

If the cell above raised `URLError`, start Bonsai in another terminal:

```bash
cd ~/tools/Bonsai-demo
LD_LIBRARY_PATH="$PWD/bin/cuda" ./bin/cuda/llama-server \\
  -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \\
  -ngl 99 -fa on -c 2048 --host 127.0.0.1 --port 8081
```

Do **not** set `CUDA_VISIBLE_DEVICES` — the PrismML fork lists the RTX
3060 as CUDA0, opposite to `nvidia-smi`."""),
], "03 — Bonsai HTTP surface")

# ═══════════════════════════════════════════════════════════════════════
# 04 — ewm-sm → Laya
# ═══════════════════════════════════════════════════════════════════════
nb04 = nb([
    md("""# 04 — Interaction chain: ewm-sm → Laya

**Question this notebook answers:** how does lattice state become a typed
context-policy decision?

The ewm-sm sidecar is a feature extractor; Laya is a typed-decision head.
This notebook runs `ingest`/`noether`/`bss` on a synthetic conversation,
renders the trajectory features into Laya's prose convention, and asks
Laya to choose the next context mode. Companion reference:
`docs/INTERACTION_MAP.md` §4."""),
    code(PROLOGUE),
    code('''import json, os

WORK = os.path.join(PROJ, "work", "nb04")
os.makedirs(WORK, exist_ok=True)
union = os.path.join(WORK, "frames.jsonl")

turns = [
    ["tid1", "tid2", "tid3", "tid4", "tid5"],
    ["tid2", "tid3", "tid6", "tid7"],
    ["tid6", "tid7", "tid8", "tid9"],
    ["tid8", "tid9", "tid10"],
]
with open(union, "w") as fh:
    for i, tids in enumerate(turns, 1):
        fh.write(json.dumps({"id": i, "tokens": tids}) + "\\n")

ewm = EwmScene()
print("binary:", ewm.bin)
print("frames:", json.dumps(turns))'''),
    md("""## Step 1 — extract trajectory features from ewm-scene"""),
    code('''uni = ewm.ingest(union)
noe = ewm.noether(union)
bss = ewm.bss(union)

features = {
    "turn": len(uni["frames"]),
    "union_pop": uni["frames"][-1]["pop"],
    "key": uni["frames"][-1]["key"][:16],
    "full_memory_tids": len(turns[-1]),
    "novelty_tids": len({t for f in turns for t in f}),
    "last_dp": noe["dp"][-1],
    "last_ind1": round(noe["ind1"][-1], 3),
    "last_ind2": round(noe["ind2"][-1], 3),
    "last_ind3": round(noe["ind3"][-1], 3),
    "last_jaccard": round(bss["jaccard"][-1], 3),
}
print(json.dumps(features, indent=2))'''),
    md("""## Step 2 — render the features as a prose state

Laya's encoder prefers natural language over raw structs (notebook 17 in
the research repo measured the improvement)."""),
    code('''laya = LayaDaemon()

state = LayaDaemon.prose_state(
    "What does the conversation add right now?",
    memory_tokens=features["full_memory_tids"],
    bss={"memory_coverage": features["last_jaccard"]},
)
print(state)'''),
    md("""## Step 3 — ask Laya for the next context policy

One `choice` (the route) + one `noul` (`keep_short`) batched in a single
forward pass — exactly the request bonsai-ewm's `auto` advisor makes."""),
    code('''decision = laya.route(
    state,
    options={
        "full": "keep the full materialized memory as the context",
        "compact": "keep only new, previously unseen tids as the context",
        "surprise": "keep only structurally surprising content (D-part plus Noether-flagged frames)",
    },
    instructions="How should the next context be built for the base model?",
    extra_noul={"keep_short": "The context should be kept short"},
    verbose=True,
)
print()
print("decision:", decision["decision"])
print("confidence:", round(decision["confidence"], 4))
print("keep_short:", round(decision["aux"]["keep_short"], 4))
print("input_tokens:", decision["input_tokens"])'''),
    md("""## Step 4 — the same question, three different feature states

Watch how the typed decision follows the features: a crowded memory
should push `keep_short` up and favour `compact`; a memory full of
surprising transitions should favour `surprise`."""),
    code('''scenarios = [
    ("idle",       LayaDaemon.prose_state("Say hello.", memory_tokens=4, bss={"coverage": 0.9})),
    ("redundant",  LayaDaemon.prose_state("Repeat the last answer.", memory_tokens=60, bss={"coverage": 0.97})),
    ("surprising", LayaDaemon.prose_state("The last answer changed the whole structure.", memory_tokens=60, bss={"coverage": 0.2})),
]
for name, state in scenarios:
    d = laya.route(
        state,
        options={"full": "whole memory", "compact": "new tids only", "surprise": "surprising frames only"},
        instructions="How should the next context be built?",
        extra_noul={"keep_short": "The context should be kept short"},
        verbose=False,
    )
    print(f"{name:11s} -> {d['decision']:9s} conf={d['confidence']:.3f} keep_short={d['aux']['keep_short']:.3f}")'''),
    code("laya.close()\nprint('daemon closed')"),
], "04 — ewm-sm → Laya")

# ═══════════════════════════════════════════════════════════════════════
# 05 — the full chain
# ═══════════════════════════════════════════════════════════════════════
nb05 = nb([
    md("""# 05 — The full chain: ewm-sm → Laya → Bonsai

**Question this notebook answers:** how do the three systems cooperate
on one turn, and what does each hop contribute?

```text
query → ewm-scene features → Laya typed decision → context proposal
      → Bonsai answer → tokenize → new frame → ewm-scene S(t) → next turn
```

This notebook runs a minimal controller loop with the raw clients and
prints every hop. Companion reference: `docs/INTERACTION_MAP.md` §7."""),
    code(PROLOGUE),
    code('''import json, os

WORK = os.path.join(PROJ, "work", "nb05")
os.makedirs(WORK, exist_ok=True)

ewm = EwmScene()
bonsai = BonsaiClient()
print("ewm-scene:", ewm.bin)
print("bonsai health:", bonsai.health())

RUN_CHAIN = bonsai.health()
if not RUN_CHAIN:
    print()
    print("Bonsai is DOWN — the cells below will print the requests they WOULD send.")
    print("Start it with ~/tools/Bonsai-demo scripts, then re-run this notebook.")'''),
    code('''# Start Laya once; reuse the daemon across turns.
laya = LayaDaemon()

union = os.path.join(WORK, "chain_union.jsonl")
with open(union, "w") as fh:
    fh.write("")

# controller state
seen = set()
comp_mem = []          # D-part novelty stream
frame_tids = []        # per-turn answer tids
mem_tids = []          # full materialized memory (last frame, restored order)
policy_cap = 24

def displacement(tids):
    out = []
    for t in tids:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def proposal(mode, cap):
    if mode == "full":
        return mem_tids[-cap:]
    if mode == "compact":
        return comp_mem[-cap:]
    if mode == "surprise":
        extra = []
        try:
            noe = ewm.noether(union)
            thr = {"dp": 6, "ind1": 0.5, "ind2": 0.5, "ind3": 0.5}
            flags = [False] * len(noe["dp"])
            for i in range(len(noe["dp"])):
                flags[i] = (noe["dp"][i] > thr["dp"] or noe["ind1"][i] > thr["ind1"]
                            or noe["ind3"][i] > thr["ind3"]
                            or (i >= 1 and noe["ind2"][i-1] < thr["ind2"]))
            for i, f in enumerate(flags):
                if f:
                    for idx in (i, i + 1):
                        if idx < len(frame_tids):
                            extra.extend(frame_tids[idx])
        except Exception:
            pass
        raw = comp_mem[-cap:] + extra
        out = []
        for t in raw:
            if not out or out[-1] != t:
                out.append(t)
        return out[-cap:]
    return []

def features_for(query):
    uni = ewm.ingest(union)
    noe = ewm.noether(union)
    return {
        "turn": len(uni["frames"]),
        "union_pop": uni["frames"][-1]["pop"] if uni["frames"] else 0,
        "full_memory_tids": len(mem_tids),
        "novelty_tids": len(comp_mem),
        "last_dp": noe["dp"][-1] if noe.get("dp") else 0,
    }

def one_turn(query):
    global mem_tids
    # hop 1: ewm-scene features
    feats = features_for(query)
    state = LayaDaemon.prose_state(query, memory_tokens=feats["full_memory_tids"])

    # hop 2: Laya decides the context policy
    d = laya.route(
        state,
        options={
            "full": "keep the full materialized memory as the context",
            "compact": "keep only new, previously unseen tids as the context",
            "surprise": "keep only structurally surprising content",
        },
        instructions="How should the next context be built for the base model?",
        extra_noul={"keep_short": "The context should be kept short"},
        verbose=False,
    )
    mode = d["decision"] if d["decision"] in ("full", "compact", "surprise") else "full"
    cap = 8 if d["aux"]["keep_short"] > 0.5 else policy_cap

    # hop 3: context proposal -> detokenize -> prompt
    prefix_tids = proposal(mode, cap)
    if prefix_tids:
        prefix_text = bonsai.detokenize([int(t[3:]) for t in prefix_tids])
        prompt = f"Context memory ({mode}):\\n{prefix_text}\\n\\nQuery: {query}"
    else:
        prompt = query

    # hop 4: Bonsai answers (or prints the request if the server is down)
    if RUN_CHAIN:
        full = bonsai.chat_full(prompt, max_tokens=128, temperature=0.0)
        answer = full["content"]
        prompt_tokens = int(full["usage"].get("prompt_tokens", 0))
    else:
        print("── would POST /v1/chat/completions ──")
        print(json.dumps({"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 128, "temperature": 0.0, "stream": False}, indent=2))
        answer = "(bonsai down)"
        prompt_tokens = len(prompt.split())

    # hop 5: tokenize the answer back into tid frames
    if RUN_CHAIN:
        tids = [f"tid{i}" for i in bonsai.tokenize(answer)]
    else:
        tids = [f"tid{1000 + j}" for j in range(3)]

    # hop 6: ingest into the lattice
    comp_mem.extend(displacement(tids))
    frame_tids.append(tids)
    with open(union, "a") as fh:
        fh.write(json.dumps({"id": feats["turn"] + 1, "tokens": tids}) + "\\n")
    try:
        mat = ewm.materialize(union)
        mem_tids = [t for t in mat["frames"][-1]["ordered"]]
    except Exception:
        mem_tids = list(tids)

    uni = ewm.ingest(union)
    return {
        "turn": len(uni["frames"]),
        "query": query,
        "mode": mode,
        "cap": cap,
        "prefix_tids": len(prefix_tids),
        "prompt_tokens": prompt_tokens,
        "answer": answer,
        "pop": uni["frames"][-1]["pop"],
        "laya_conf": round(d["confidence"], 3),
        "keep_short": round(d["aux"]["keep_short"], 3),
    }'''),
    md("""## Run three turns and watch the chain"""),
    code('''rows = []
for q in [
    "What is the capital of France? Answer in one short sentence.",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
]:
    rows.append(one_turn(q))

print(f"{'t':>2} {'mode':9s} {'cap':>3} {'prefix':>6} {'prompt_tok':>10} {'pop':>4} {'conf':>6} {'short':>6}  answer")
for r in rows:
    print(f"{r['turn']:>2} {r['mode']:9s} {r['cap']:>3} {r['prefix_tids']:>6} "
          f"{r['prompt_tokens']:>10} {r['pop']:>4} {r['laya_conf']:>6} {r['keep_short']:>6}  {r['answer'][:48]!r}")'''),
    md("""## What to observe

- `prefix_tids` grows after turn 1 — Bonsai starts receiving a context
  prefix built from the lattice.
- `mode`/`cap` change per turn — Laya's typed decision, not a fixed rule.
- `prompt_tokens` is Bonsai's own accounting of the prefix + query.
- `pop` is the lattice's accounting of the last frame (≈ 3× distinct
  tids).
- When `keep_short > 0.5` the controller drops the cap to 8.
- **Bonsai is a reasoning model.** With a small `max_tokens`, the
  reasoning trace can consume the whole budget and `content` comes back
  empty (the lattice then ingests an empty frame, `pop=0`). This notebook
  uses `max_tokens=128`; the server log suggests `--reasoning-preserve`
  as the long-term fix."""),
    code("laya.close()\nprint('daemon closed')"),
], "05 — ewm-sm → Laya → Bonsai chain")

# ═══════════════════════════════════════════════════════════════════════
# 06 — decision models compared
# ═══════════════════════════════════════════════════════════════════════
nb06 = nb([
    md("""# 06 — Decision models compared on the same state

**Question this notebook answers:** what sits in the "decision model"
slot of the chain, and how do the options differ on identical input?

The chain is agnostic to the decision model. Three implementations exist:

| Model | Transport | Notes |
| --- | --- | --- |
| Rust Laya | local daemon, JSONL | open-source, CPU, ~0.6 s/decision |
| TypeSafe Jev | REST `POST /v1/systemone` | hosted, needs `TYPESAFE_API_KEY` |
| mock | in-process | deterministic fallback, no model |

This notebook sends the **same** state and questions to Laya and (if the
key is present) to Jev, and prints the decisions side by side."""),
    code(PROLOGUE),
    code('''import os

STATE = "The user asked: \\"Explain gravity in one sentence.\\" The system's memory holds 24 content-addressed token ids from earlier answers."
OPTIONS = {
    "full": "keep the full materialized memory as the context",
    "compact": "keep only new, previously unseen tids as the context",
    "surprise": "keep only structurally surprising content",
}
INSTRUCTIONS = "How should the next context be built for the base model?"'''),
    md("""## Laya (local daemon)"""),
    code('''laya = LayaDaemon()
d_laya = laya.route(STATE, OPTIONS, INSTRUCTIONS, extra_noul={"keep_short": "The context should be kept short"}, verbose=True)
laya.close()'''),
    md("""## Jev (TypeSafe REST) — only when the key is present

The request shape is identical in spirit; the wire format is the TypeSafe
System One REST API."""),
    code('''try:
    import sys as _sys
    sys.path.insert(0, os.path.join(PROJ, "..", "bonsai-ewm", "src"))
    from bonsai_ewm.adapters import JevAdapter
    jev = JevAdapter()
    if jev.mock:
        print("Jev: mock (TYPESAFE_API_KEY not set) — skipping real call")
    else:
        r = jev.route(
            {"turn": 2, "query": "Explain gravity in one sentence.",
             "union_pop": 18, "full_memory_tids": 24, "novelty_tids": 9,
             "last_answer_chars": 120},
            OPTIONS, INSTRUCTIONS, extra_noul={"keep_short": "The context should be kept short"},
        )
        print("Jev decision:", r["decision"])
        print("Jev probabilities:", r["probabilities"])
        print("Jev confidence:", round(r["confidence"], 4))
        print("Jev keep_short:", round(r["aux"]["keep_short"], 4))
        print("Jev model:", r["model"], "| input_tokens:", r["input_tokens"])
except Exception as exc:
    print("Jev call skipped:", exc)'''),
    md("""## Mock (deterministic, no model)

The mock exists so the chain runs without any decision model. It is
deterministic in the query, not intelligent — useful for tests, not for
real routing."""),
    code('''import zlib
seed = zlib.crc32(STATE.encode())
names = list(OPTIONS)
vals = [1.0 + ((seed >> i) & 3) for i in range(len(names))]
total = sum(vals)
probs = {n: v / total for n, v in zip(names, vals)}
print("mock decision:", max(probs, key=probs.get))
print("mock probabilities:", {k: round(v, 4) for k, v in probs.items()})
print("mock keep_short:", 0.5)'''),
    md("""## Summary

- All three fill the same slot: `(state, options, instructions, noul) →
  decision + confidence + probabilities`.
- Laya is local and ungated — the default for the chain in this project.
- Jev is the hosted sibling; the bonsai-ewm production controller uses it
  in `auto` mode when `TYPESAFE_API_KEY` is set.
- The mock keeps offline development moving and must never be mistaken
  for a real decision."""),
], "06 — decision models compared")

NOTEBOOKS = [
    ("01_ewm_scene_protocol.ipynb", nb01),
    ("02_laya_daemon_protocol.ipynb", nb02),
    ("03_bonsai_http_surface.ipynb", nb03),
    ("04_ewm_to_laya.ipynb", nb04),
    ("05_ewm_laya_bonsai_chain.ipynb", nb05),
    ("06_decision_models_compared.ipynb", nb06),
]


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    for filename, notebook in NOTEBOOKS:
        path = os.path.join(NB_DIR, filename)
        with open(path, "w") as fh:
            json.dump(notebook, fh, indent=1)
            fh.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
