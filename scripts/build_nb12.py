#!/usr/bin/env python3
"""Build notebooks/12_hllset_pagerank.ipynb — HLLSet PageRank exercise.

The lattice as a web: nodes = HLLSets, asymmetric BSS = directed coverage
links, HSF (HLLSet frequency) = popularity prior, PageRank = relatedness
navigation alongside the Merkle tree's exact-address lookup.
"""

import json

NB_PATH = "/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-laya-bonsai-lab/notebooks/12_hllset_pagerank.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": s,
}

cells = [
    md(
        "# 12 — HLLSet PageRank: the BSS lattice as a web of content, HSF as popularity\n"
        "\n"
        "The HLLSet lattice with its **asymmetric BSS edges** is structurally the\n"
        "same thing as the web: nodes (HLLSets) connected by directed,\n"
        "weighted links (BSS coverage), and a per-node **popularity** signal\n"
        "that on the web is page visits and here is **HSF** — HLLSet frequency\n"
        "(the `HllsetLut` touch-count `TH` that `ewm-sm-explore store` already\n"
        "ranks by).\n"
        "\n"
        "Right now we navigate the lattice with the **Merkle tree**:\n"
        "content-address, exact lookup. That is a dictionary, not a search\n"
        "engine. This notebook does the Google move: run **PageRank** over the\n"
        "BSS link graph, seeded with HSF as the teleportation prior, and use\n"
        "the stationary distribution to **search related HLLSets** — the\n"
        "neighbors a Merkle lookup can never surface because their content key\n"
        "differs.\n"
        "\n"
        "```text\n"
        "   web                           HLLSet lattice\n"
        "   page ────────────────►        HLLSet (a frame, a codebook gate, …)\n"
        "   hyperlink ───────────►        BSS edge (asymmetric coverage)\n"
        "   visits / popularity ─►        HSF = HllsetLut TH (touch count)\n"
        "   Google PageRank ─────►        this notebook\n"
        "   URL lookup ──────────►        Merkle tree (content address)\n"
        "```\n"
        "\n"
        "The demo uses the real open-loop frames of the structural-router run\n"
        "(notebook 21, cache dir) plus the three per-LLM cumulative codebook\n"
        "gates — a small but genuine lattice: 19 nodes, asymmetric BSS edges."
    ),
    code(
        "import json\n"
        "import os\n"
        "\n"
        "import mmh3\n"
        "import numpy as np\n"
        "\n"
        "BITS_PER_REG = 32\n"
        "P = 10          # 1024 registers — the project's soldered precision\n"
        "\n"
        "def frame_set(tokens, p=P):\n"
        "    \"\"\"G1 ∪ G2 ∪ G3 projection (CHANNEL_SEEDS = [0,1,2]) as a Python bitset.\"\"\"\n"
        "    s = 0\n"
        "    m = 1 << p\n"
        "    for t in tokens:\n"
        "        for seed in (0, 1, 2):\n"
        "            h = mmh3.hash64(t.encode(), seed=seed, signed=False)[0]\n"
        "            reg = h & (m - 1)\n"
        "            rem = h >> p\n"
        "            tz = 31 if rem == 0 else min((rem & -rem).bit_length() - 1, 31)\n"
        "            s |= 1 << (reg * BITS_PER_REG + tz)\n"
        "    return s\n"
        "\n"
        "CACHE = \"/home/alexmy/.cache/ewm-structural-router\"\n"
        "union_path = f\"{CACHE}/open_union.jsonl\"\n"
        "frame_path = f\"{CACHE}/frame_frontends.json\"\n"
        "assert os.path.exists(union_path), f\"run ewm-state-machine notebook 21 first: {union_path}\"\n"
        "assert os.path.exists(frame_path), f\"missing {frame_path}\"\n"
        "\n"
        "frames = [json.loads(line) for line in open(union_path)]\n"
        "frontends = json.load(open(frame_path))[\"dimensions\"]\n"
        "print(\"frames:\", len(frames), \"| codebook gates:\", len(frontends))\n"
        "print(\"hash parity check (tid0):\", mmh3.hash64(b\"tid0\", seed=0, signed=False)[0])"
    ),
    code(
        "# Nodes: every step frame of the open loop + the per-LLM cumulative\n"
        "# codebook gates. Each node is a bitset (Python int) over the 32768-bit\n"
        "# plane; popcount = active bits.\n"
        "nodes = [frame_set(f[\"tokens\"]) for f in frames]\n"
        "labels = [f\"t{i:02d}\" for i in range(1, len(frames) + 1)]\n"
        "for d in frontends:\n"
        "    nodes.append(frame_set(d[\"tokens\"]))\n"
        "    labels.append(d[\"name\"])\n"
        "\n"
        "n = len(nodes)\n"
        "pop = np.array([int(v.bit_count()) for v in nodes], dtype=np.float64)\n"
        "print(\"nodes:\", n, \"| popcounts:\", {l: int(p) for l, p in zip(labels, pop)})"
    ),
    md(
        "## §1 — The BSS link graph (asymmetric)\n"
        "\n"
        "`BSS(i, j) = |H_i ∩ H_j| / |H_j|` — the fraction of `H_j` covered by\n"
        "`H_i`. It is asymmetric: a step frame is fully covered by its\n"
        "cumulative gate (`BSS(step, gate) = 1` when the gate contains the\n"
        "step), but the gate is only partly covered by the step. We read each\n"
        "pair as a **directed coverage link**: `i` points at the nodes it\n"
        "covers, with weight `BSS(i, j)`."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "A = np.zeros((n, n))\n"
        "for i in range(n):\n"
        "    for j in range(n):\n"
        "        inter = int(nodes[i] & nodes[j]).bit_count()\n"
        "        A[i, j] = inter / pop[j] if pop[j] else 0.0   # BSS(i covers j)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "im = ax.imshow(A, cmap=\"magma\", vmin=0, vmax=1)\n"
        "ax.set_xticks(range(n), labels, rotation=45, ha=\"right\", fontsize=8)\n"
        "ax.set_yticks(range(n), labels, fontsize=8)\n"
        "ax.set_title(\"BSS coverage matrix A[i,j] = |H_i ∩ H_j| / |H_j| (asymmetric)\")\n"
        "plt.colorbar(im, label=\"coverage\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(f\"{CACHE}/bss_graph.png\", dpi=110)\n"
        "plt.show()\n"
        "\n"
        "asym = np.abs(A - A.T).sum() / 2\n"
        "print(\"total asymmetric mass:\", round(float(asym), 1), \"(0 would be a symmetric graph)\")"
    ),
    md(
        "## §2 — HSF: HLLSet frequency (popularity)\n"
        "\n"
        "The analog of page visits. Here HSF is the content frequency across\n"
        "the observations (how often this exact HLLSet appears); in the Rust\n"
        "store the production HSF is `HllsetLut::TH` — the touch count that\n"
        "`ewm-sm-explore store` ranks by. HSF is a popularity prior, not a\n"
        "relatedness measure: it sees a node's own frequency, but not how\n"
        "**central** it is in the coverage graph."
    ),
    code(
        "from collections import Counter\n"
        "\n"
        "content = Counter(nodes)\n"
        "hsf = np.array([content[v] for v in nodes], dtype=np.float64)\n"
        "hsf_rank = np.argsort(np.argsort(-hsf))  # 0 = most frequent\n"
        "\n"
        "order = np.argsort(-hsf)\n"
        "print(\"HSF ranking (popularity only):\")\n"
        "for k in range(6):\n"
        "    i = order[k]\n"
        "    print(f\"  {k+1}. {labels[i]:6s}  HSF={hsf[i]:.0f}  pop={pop[i]:.0f}\")"
    ),
    md(
        "## §3 — PageRank over the BSS graph, seeded by HSF\n"
        "\n"
        "The web model: each node distributes its vote among the nodes that\n"
        "**cover it**, proportionally to coverage — so the transition is\n"
        "`T[i, j] = BSS(i covers j) / Σ_k BSS(k covers j)`. Then\n"
        "`r = d·T·r + (1-d)·v` with teleportation vector `v = HSF` normalized.\n"
        "A node is important if it covers nodes that are themselves important;\n"
        "HSF biases the random-surfer jumps toward popular content."
    ),
    code(
        "d = 0.85\n"
        "\n"
        "col_sum = A.sum(axis=0)\n"
        "T = np.divide(A, col_sum, out=np.zeros_like(A), where=col_sum > 0)\n"
        "v = hsf / hsf.sum()\n"
        "\n"
        "r = np.full(n, 1.0 / n)\n"
        "for it in range(500):\n"
        "    r_new = d * (T @ r) + (1.0 - d) * v\n"
        "    if np.abs(r_new - r).max() < 1e-12:\n"
        "        r = r_new\n"
        "        break\n"
        "    r = r_new\n"
        "else:\n"
        "    print(\"did not converge in 500 iterations\")\n"
        "\n"
        "pr_rank = np.argsort(np.argsort(-r))\n"
        "\n"
        "def spearman(a, b):\n"
        "    ra = np.argsort(np.argsort(a)).astype(float)\n"
        "    rb = np.argsort(np.argsort(b)).astype(float)\n"
        "    return float(np.corrcoef(ra, rb)[0, 1])\n"
        "\n"
        "print(f\"converged in {it} iterations | PageRank sum {r.sum():.6f}\")\n"
        "print(\"PageRank ranking (importance over the coverage graph):\")\n"
        "for k in range(6):\n"
        "    i = np.argsort(-r)[k]\n"
        "    print(f\"  {k+1}. {labels[i]:6s}  PR={r[i]:.4f}  HSF={hsf[i]:.0f}  pop={pop[i]:.0f}\")\n"
        "print()\n"
        "print(\"Spearman(HSF rank, PageRank rank):\", round(spearman(hsf, r), 3))\n"
        "movers = [(labels[i], hsf_rank[i], pr_rank[i])\n"
        "          for i in range(n) if abs(hsf_rank[i] - pr_rank[i]) >= 3]\n"
        "print(\"nodes that move >= 3 rank positions (popularity alone mis-sorts them):\")\n"
        "for label, h, p in sorted(movers, key=lambda t: -abs(t[1]-t[2]))[:8]:\n"
        "    print(f\"  {label:6s} HSF rank {h:2d} -> PageRank rank {p:2d}\")"
    ),
    md(
        "## §4 — Related-HLLSet search: Merkle (exact) vs PageRank (related)\n"
        "\n"
        "The Merkle tree answers *\"which node has exactly this content?\"* — a\n"
        "content-address lookup. The BSS graph answers *\"which nodes are\n"
        "related to this one, and how important are they?\"*. For a query node\n"
        "we list two relatedness views:\n"
        "\n"
        "- **covers** — nodes the query covers (`BSS(q, ·)` high): its members,\n"
        "  the content it points at;\n"
        "- **covered by** — nodes covering the query (`BSS(·, q)` high): its\n"
        "  hubs, the gates/categories it belongs to.\n"
        "\n"
        "Both lists are re-ranked by PageRank, so *important* related nodes\n"
        "surface first — the Google move the Merkle tree cannot make."
    ),
    code(
        "q = 0   # the first step frame (\"What is the capital of France?\") as query\n"
        "print(f\"query: {labels[q]}  (pop={pop[q]:.0f})\")\n"
        "\n"
        "# Merkle mode: exact content-address lookup.\n"
        "exact = [labels[i] for i in range(n) if nodes[i] == nodes[q]]\n"
        "print(\"Merkle exact match:\", exact)\n"
        "\n"
        "# Relatedness mode: BSS neighbors, re-ranked by PageRank.\n"
        "def related(weights):\n"
        "    idx = np.argsort(-weights)\n"
        "    return [(labels[i], round(float(weights[i]), 3), round(float(r[i]), 4))\n"
        "            for i in idx if i != q and weights[i] > 0][:4]\n"
        "\n"
        "print()\n"
        "print(\"covers (query -> node, BSS(q,·)):\")\n"
        "for label, bss, pr in related(A[q]):\n"
        "    print(f\"  {label:6s}  BSS={bss:.3f}  PR={pr:.4f}\")\n"
        "print()\n"
        "print(\"covered by (node -> query, BSS(·,q)):\")\n"
        "for label, bss, pr in related(A[:, q]):\n"
        "    print(f\"  {label:6s}  BSS={bss:.3f}  PR={pr:.4f}\")"
    ),
    md(
        "## §5 — Where this plugs into ewm-sm\n"
        "\n"
        "- **HSF already exists**: `HllsetLut::TH` — `ewm-sm-explore store`\n"
        "  prints the ranked list. It is popularity only, and the Merkle tree\n"
        "  search does not consume it.\n"
        "- **PageRank is a second ranking dimension** over the same nodes:\n"
        "  importance in the BSS coverage graph. A natural production step is\n"
        "  to persist `PR` next to `TH` in the hllsetLUT table and expose both\n"
        "  in `ewm-sm-explore store` / the Arrow cache.\n"
        "- **Two navigation modes coexist**: the Merkle tree (exact\n"
        "  content-address) and the PageRank/BSS graph (related + important).\n"
        "  A router or a context manager would use the Merkle tree to fetch a\n"
        "  node and the PageRank graph to fetch *what relates to it* — the\n"
        "  lattice analog of \"I know the URL\" vs \"search the web\".\n"
        "\n"
        "The demo used 19 nodes built from the notebook-21 open loop; the same\n"
        "construction scales to every HLLSet the store registers (per-frame,\n"
        "per-gate, per-commit) — the `HllsetLut` is already the node registry."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "ewm-nanolm",
            "language": "python",
            "name": "ewm-nanolm",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")
print("wrote", NB_PATH, "with", len(cells), "cells")
