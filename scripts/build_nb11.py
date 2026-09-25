#!/usr/bin/env python3
"""Build notebooks/11_per_user_codebook_gates.ipynb in the lab's style."""

import json

NB_PATH = "/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-laya-bonsai-lab/notebooks/11_per_user_codebook_gates.ipynb"

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
        "# 11 — Per-user codebook gates: exact materialization + shared-lattice restore\n"
        "\n"
        "Notebook 10's codebook-filtered materialization is really a **gate**: the\n"
        "user's codebook, lifted into the bit plane, is an HLLSet `G_u` — and\n"
        "`G_u ∩ H` extracts exactly the user's component of any shared HLLSet\n"
        "`H`. This is the `Gn ∩ H` gate idea from `docs/SEPARATION.md`, made\n"
        "per-user; the same \"vocabulary is the gate\" pattern DeepSeek-OCR uses\n"
        "with its codebook anchors.\n"
        "\n"
        "The gate improves notebook 10 in two ways:\n"
        "\n"
        "1. **Materialization**: keep only candidates whose bits lie in\n"
        "   `S ∩ G_u` and whose tokens are in the codebook — recall and\n"
        "   precision both go to 1.0 (up to hash collisions).\n"
        "2. **Restore**: `U_u(t) ≈ c(t) ∩ G_u` — no per-user current set\n"
        "   needed at all. The only error is *cross-codebook hash collisions*\n"
        "   (another user's token hitting a bit inside `G_u`), which is far\n"
        "   smaller than nb10's `c(t) ∩ U_u(now)` overlap error.\n"
        "\n"
        "Finally we show the apparatus already supports gates: `ewm-scene\n"
        "project --frame` computes per-dimension gated intersections, so a\n"
        "per-user codebook is just a named dimension."
    ),
    code(
        "import json\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "import mmh3\n"
        "import numpy as np\n"
        "\n"
        "LAB = \"/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-laya-bonsai-lab\"\n"
        "sys.path.insert(0, LAB)\n"
        "\n"
        "from lab.ewm import EwmScene, write_frames\n"
        "\n"
        "BITS_PER_REG = 32\n"
        "\n"
        "def frame_set(tokens, P=10):\n"
        "    \"\"\"G1 u G2 u G3 projection (CHANNEL_SEEDS = [0,1,2]).\"\"\"\n"
        "    s = 0\n"
        "    M = 1 << P\n"
        "    for t in tokens:\n"
        "        for seed in (0, 1, 2):\n"
        "            h = mmh3.hash64(t.encode(), seed=seed, signed=False)[0]\n"
        "            reg = h & (M - 1)\n"
        "            rem = h >> P\n"
        "            tz = 31 if rem == 0 else min((rem & -rem).bit_length() - 1, 31)\n"
        "            s |= 1 << (reg * BITS_PER_REG + tz)\n"
        "    return s\n"
        "\n"
        "print(\"helpers defined\")"
    ),
    code(
        "USERS = {\n"
        "    \"acme-finance\":   [f\"fin_{i}\" for i in range(200)],\n"
        "    \"medco-clinical\": [f\"med_{i}\" for i in range(200)],\n"
        "    \"ops-logistics\":  [f\"ops_{i}\" for i in range(200)],\n"
        "}\n"
        "\n"
        "GATES = {u: frame_set(tokens) for u, tokens in USERS.items()}\n"
        "for u, g in GATES.items():\n"
        "    print(f\"gate {u:16s} popcount {g.bit_count():5d}  (codebook {len(USERS[u])} tokens x ~3 bits)\")"
    ),
    md("## §1 — Gates are near-disjoint for distinct codebooks\n\nTwo gates overlap only where the hash function collides bits across codebooks — with disjoint vocabularies this is pure hash collision, not vocabulary overlap."),
    code(
        "names = list(GATES)\n"
        "print(f\"{'pair':32s} {'intersection':>12s} {'jaccard':>10s}\")\n"
        "for i in range(len(names)):\n"
        "    for j in range(i + 1, len(names)):\n"
        "        a, b = GATES[names[i]], GATES[names[j]]\n"
        "        inter = (a & b).bit_count()\n"
        "        union = (a | b).bit_count()\n"
        "        print(f\"{names[i] + ' vs ' + names[j]:32s} {inter:12d} {inter / union:10.4f}\")\n"
        "\n"
        "# expected overlap for two random 600-bit sets on a 32768-bit plane:\n"
        "expected = 600 * 600 / 32768\n"
        "print(f\"\\nexpected random overlap for two ~600-bit gates: {expected:.1f} bits\")"
    ),
    md("## §2 — Gated materialization is exact\n\nMix all users' streams into the shared lattice, then materialize each user through `S ∩ G_u` + codebook filter."),
    code(
        "rng = np.random.default_rng(0)\n"
        "streams = {u: [] for u in USERS}\n"
        "S = 0\n"
        "for step in range(40):\n"
        "    for u, vocab in USERS.items():\n"
        "        toks = [vocab[int(rng.integers(0, len(vocab)))] for _ in range(30)]\n"
        "        streams[u].append(toks)\n"
        "        S |= frame_set(toks)\n"
        "\n"
        "# global LUT: bit -> tokens (any user) that ever hit it\n"
        "lut = {}\n"
        "for vocab in USERS.values():\n"
        "    for tok in vocab:\n"
        "        M = 1 << 10\n"
        "        for seed in (0, 1, 2):\n"
        "            h = mmh3.hash64(tok.encode(), seed=seed, signed=False)[0]\n"
        "            reg = h & (M - 1)\n"
        "            rem = h >> 10\n"
        "            tz = 31 if rem == 0 else min((rem & -rem).bit_length() - 1, 31)\n"
        "            lut.setdefault(reg * 32 + tz, set()).add(tok)\n"
        "\n"
        "def gated_materialize(user, S_bits):\n"
        "    vocab = set(USERS[user])\n"
        "    gated = S_bits & GATES[user]\n"
        "    candidates = set()\n"
        "    x = gated\n"
        "    while x:\n"
        "        low = x & -x\n"
        "        candidates |= lut.get(low.bit_length() - 1, set())\n"
        "        x ^= low\n"
        "    return {t for t in candidates if t in vocab}\n"
        "\n"
        "print(f\"{'user':16s} {'true':>6s} {'restored':>9s} {'recall':>8s} {'precision':>10s}\")\n"
        "for u in USERS:\n"
        "    truth = set().union(*[set(t) for t in streams[u]])\n"
        "    restored = gated_materialize(u, S)\n"
        "    rec = len(truth & restored) / len(truth)\n"
        "    prec = len(truth & restored) / len(restored) if restored else 0.0\n"
        "    print(f\"{u:16s} {len(truth):6d} {len(restored):9d} {rec:8.3f} {prec:10.3f}\")"
    ),
    md("## §3 — Shared-lattice restore through the gate (no per-user current set)\n\n`U_u(t) ≈ c(t) ∩ G_u`. Compare against notebook 10's `c(t) ∩ U_u(now)` — the gate removes the cross-user overlap that made nb10's estimate a loose superset."),
    code(
        "def simulate_restore(P):\n"
        "    rng = np.random.default_rng(1)\n"
        "    steps = 40\n"
        "    # per-user cumulative sets and shared lattice per step\n"
        "    U = {u: [0] * (steps + 1) for u in USERS}\n"
        "    S = [0] * (steps + 1)\n"
        "    gates = {u: frame_set(v, P) for u, v in USERS.items()}\n"
        "    for t in range(1, steps + 1):\n"
        "        S[t] = S[t - 1]\n"
        "        for u, vocab in USERS.items():\n"
        "            toks = [vocab[int(rng.integers(0, len(vocab)))] for _ in range(30)]\n"
        "            fs = frame_set(toks, P)\n"
        "            U[u][t] = U[u][t - 1] | fs\n"
        "            S[t] |= fs\n"
        "    gate_errs, now_errs = [], []\n"
        "    for u in USERS:\n"
        "        now = U[u][steps]\n"
        "        for t in range(1, steps):\n"
        "            denom = U[u][t].bit_count()\n"
        "            if not denom:\n"
        "                continue\n"
        "            gate_errs.append(((S[t] & gates[u]) & ~U[u][t]).bit_count() / denom)\n"
        "            now_errs.append(((S[t] & now) & ~U[u][t]).bit_count() / denom)\n"
        "    return np.mean(gate_errs), np.mean(now_errs)\n"
        "\n"
        "print(f\"{'P':>3s} {'gate restore err':>18s} {'nb10 (c(t) n U_now) err':>25s}\")\n"
        "for P in (10, 12, 14):\n"
        "    g, n = simulate_restore(P)\n"
        "    print(f\"{P:3d} {g:18.4f} {n:25.4f}\")"
    ),
    md("## §4 — The apparatus already speaks gates: `ewm-scene project --frame`\n\nA per-user codebook is just a named dimension; `project` computes `|S ∩ D_i| / |D_i|` per dimension — the gated intersection. Let's run the real binary on a mixed union with a per-user codebook frame."),
    code(
        "ewm = EwmScene()\n"
        "print(\"binary:\", ewm.bin)\n"
        "\n"
        "# a small mixed union: 3 frames, tokens from all three codebooks\n"
        "frames = [\n"
        "    {\"id\": 1, \"tokens\": [f\"fin_{i}\" for i in range(12)] + [f\"med_{i}\" for i in range(5)]},\n"
        "    {\"id\": 2, \"tokens\": [f\"med_{i}\" for i in range(10)] + [f\"ops_{i}\" for i in range(6)]},\n"
        "    {\"id\": 3, \"tokens\": [f\"ops_{i}\" for i in range(14)] + [f\"fin_{i}\" for i in range(4)]},\n"
        "]\n"
        "union_path = f\"{LAB}/work/nb11_union.jsonl\"\n"
        "frame_path = f\"{LAB}/work/nb11_frame.json\"\n"
        "write_frames(union_path, [f[\"tokens\"] for f in frames])\n"
        "with open(frame_path, \"w\") as fh:\n"
        "    json.dump({\"dimensions\": [\n"
        "        {\"name\": u, \"tokens\": vocab} for u, vocab in USERS.items()\n"
        "    ]}, fh)\n"
        "\n"
        "proj = ewm.project(union_path, frame_path)\n"
        "print(\"per-user gated coverage of the mixed union (bss = |S n G_u| / |G_u|):\")\n"
        "for f in proj[\"frames\"]:\n"
        "    print(f\"  frame {f['id']}: \" + \", \".join(f\"{n}={v:.3f}\" for n, v in zip(proj[\"names\"], f[\"bss\"])))"
    ),
    md(
        "## Summary\n"
        "\n"
        "- **A per-user codebook is a gate**: `G_u = μ(codebook)`; `G_u ∩ H`\n"
        "  extracts the user's component of any shared HLLSet — the\n"
        "  `Gn ∩ H` gate of `SEPARATION.md`, made per-user (and the same\n"
        "  \"vocabulary is the gate\" pattern as DeepSeek-OCR's codebook anchors).\n"
        "- **Materialization through the gate is exact**: recall and precision\n"
        "  both 1.0 on mixed streams (codebook filter removes every foreign\n"
        "  candidate).\n"
        "- **Restore needs no per-user current set**: `c(t) ∩ G_u` is a far\n"
        "  tighter estimate than nb10's `c(t) ∩ U_u(now)` — the error drops to\n"
        "  cross-codebook hash collisions, and shrinks with P.\n"
        "- **The apparatus already supports it**: `ewm-scene project --frame`\n"
        "  computes per-dimension gated intersections; a per-user codebook is a\n"
        "  named dimension today.\n"
        "\n"
        "Rust checklist: (1) a `gates/` store section or sidecar for per-user\n"
        "codebooks; (2) `ewm-scene materialize --gate <frame> --dim <user>` for\n"
        "gated materialization; (3) `ewm-sm-explore project-user --store <repo>\n"
        "--user <id> --commit <cid>` = `c(t) ∩ G_u` + gated materialization."
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
