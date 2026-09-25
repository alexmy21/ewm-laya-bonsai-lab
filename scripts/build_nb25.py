#!/usr/bin/env python3
"""Build notebooks/25_shared_lattice_projection.ipynb (pure Python, no GPU)."""

import json

NB_PATH = "/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-laya-bonsai-lab/notebooks/10_shared_lattice_projection.ipynb"

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
        "# 10 — Shared-lattice projection: restoring per-user state from commit HLLSets\n"
        "\n"
        "Per-user lattices can be **joined into the shared lattice** without\n"
        "breaking anything: HLLSets are token-agnostic, the hash function and\n"
        "P-precision are shared, so the union of per-user HLLSets is just the\n"
        "same Boolean lattice with more contributors.\n"
        "\n"
        "The shared commit DAG (`ewm-git`) already keeps the shared history: each\n"
        "commit stores the cumulative HLLSet `c(t)`. So a per-user state at\n"
        "time `t` can be **estimated** without keeping the user's own history:\n"
        "\n"
        "```text\n"
        "U_u(t)  ≈  c(t) ∩ U_u(now)\n"
        "```\n"
        "\n"
        "Because both lattices are monotone, `U_u(t) ⊆ c(t)` and\n"
        "`U_u(t) ⊆ U_u(now)`, so the estimate is a **superset** of the truth;\n"
        "the only error is *early attribution* — bits the user eventually sets\n"
        "that another user set earlier (cross-user overlap). This notebook\n"
        "measures that error, shows how it scales with P-precision, and checks\n"
        "that per-user codebook filtering makes materialization near-exact.\n"
        "\n"
        "Runs on the plain `python3` kernel (mmh3 + numpy only — the same hash\n"
        "the Rust crates use, `mmh3.hash64(seed=0, signed=False)`)."
    ),
    code(
        "import json\n"
        "\n"
        "import mmh3\n"
        "import numpy as np\n"
        "\n"
        "BITS_PER_REG = 32\n"
        "\n"
        "def token_bit(token: str, P: int) -> int:\n"
        "    \"\"\"The project's soldered rule: reg = hash & (M-1), tz = trailing\n"
        "    zeros of hash >> P, bit = reg*32 + tz (mmh3 x64-128 lower 64 bits).\"\"\"\n"
        "    M = 1 << P\n"
        "    h = mmh3.hash64(token.encode(), seed=0, signed=False)[0]\n"
        "    reg = h & (M - 1)\n"
        "    remaining = h >> P\n"
        "    tz = 31 if remaining == 0 else min((remaining & -remaining).bit_length() - 1, 31)\n"
        "    return reg * BITS_PER_REG + tz\n"
        "\n"
        "def frame_set(tokens, P: int) -> int:\n"
        "    \"\"\"G1 u G2 u G3 projection (CHANNEL_SEEDS = [0,1,2]).\"\"\"\n"
        "    s = 0\n"
        "    for t in tokens:\n"
        "        for seed in (0, 1, 2):\n"
        "            h = mmh3.hash64(t.encode(), seed=seed, signed=False)[0]\n"
        "            M = 1 << P\n"
        "            reg = h & (M - 1)\n"
        "            remaining = h >> P\n"
        "            tz = 31 if remaining == 0 else min((remaining & -remaining).bit_length() - 1, 31)\n"
        "            s |= 1 << (reg * BITS_PER_REG + tz)\n"
        "    return s\n"
        "\n"
        "print(\"hash parity check (tid0):\", mmh3.hash64(b\"tid0\", seed=0, signed=False)[0])"
    ),
    code(
        "def simulate(P: int, n_users: int = 3, vocab_size: int = 200,\n"
        "             tokens_per_step: int = 30, steps: int = 40, seed: int = 0):\n"
        "    \"\"\"Each user emits deterministic tokens from their own codebook; the\n"
        "    shared lattice is the monotone union of all users' frame HLLSets.\"\"\"\n"
        "    rng = np.random.default_rng(seed)\n"
        "    vocabs = [[f\"u{uid}_tok{i}\" for i in range(vocab_size)] for uid in range(n_users)]\n"
        "\n"
        "    U = [[0] * (steps + 1) for _ in range(n_users)]   # U[uid][t] cumulative to t\n"
        "    S = [0] * (steps + 1)                              # shared lattice c(t)\n"
        "    user_tokens = [[set() for _ in range(steps + 1)] for _ in range(n_users)]\n"
        "\n"
        "    for t in range(1, steps + 1):\n"
        "        S[t] = S[t - 1]\n"
        "        for uid in range(n_users):\n"
        "            toks = [vocabs[uid][int(rng.integers(0, vocab_size))]\n"
        "                    for _ in range(tokens_per_step)]\n"
        "            fs = frame_set(toks, P)\n"
        "            U[uid][t] = U[uid][t - 1] | fs\n"
        "            user_tokens[uid][t] = user_tokens[uid][t - 1] | set(toks)\n"
        "            S[t] |= fs\n"
        "    return U, S, user_tokens, vocabs\n"
        "\n"
        "print(\"simulate() defined\")\n"
        "# quick smoke\n"
        "U, S, _, _ = simulate(10, steps=5)\n"
        "print(\"P=10 smoke: S[5] popcount\", S[5].bit_count(), \"| U[0][5] popcount\", U[0][5].bit_count())"
    ),
    md("## §1 — Restore error vs P-precision\n\n`estimate = c(t) ∩ U_u(now)`; relative error = `|estimate \\ U_u(t)| / |U_u(t)|`."),
    code(
        "print(f\"{'P':>3s} {'M':>6s} {'plane bits':>10s} {'mean rel err':>14s} {'max rel err':>12s}\")\n"
        "for P in (10, 12, 14):\n"
        "    U, S, _, _ = simulate(P)\n"
        "    steps = len(S) - 1\n"
        "    rels = []\n"
        "    for uid in range(len(U)):\n"
        "        now = U[uid][steps]\n"
        "        for t in range(1, steps):\n"
        "            est = S[t] & now\n"
        "            err = (est & ~U[uid][t]).bit_count()\n"
        "            denom = U[uid][t].bit_count()\n"
        "            if denom:\n"
        "                rels.append(err / denom)\n"
        "    print(f\"{P:3d} {1 << P:6d} {(1 << P) * 32:10d} {np.mean(rels):14.4f} {np.max(rels):12.4f}\")"
    ),
    md("## §2 — The error is one-sided (superset property)\n\nFor every `(t, user)`: `U_u(t) \\ estimate` must be empty — the estimate never misses a bit the user truly had; it only adds early-attributed bits."),
    code(
        "U, S, _, _ = simulate(10)\n"
        "steps = len(S) - 1\n"
        "misses = 0\n"
        "early = 0\n"
        "for uid in range(len(U)):\n"
        "    now = U[uid][steps]\n"
        "    for t in range(1, steps):\n"
        "        est = S[t] & now\n"
        "        misses += (U[uid][t] & ~est).bit_count()          # must be 0\n"
        "        early  += (est & ~U[uid][t]).bit_count()          # early attribution\n"
        "print(\"bits the estimate MISSED (must be 0):\", misses)\n"
        "print(\"early-attributed bits (the whole error):\", early)\n"
        "print(\"=> estimate is always a superset of the true per-user state\")"
    ),
    md("## §3 — Per-user codebook-filtered materialization\n\nRestore the user's **tokens** at time `t`: take the fibers of the estimate bits (the global LUT), keep only candidates in the user's own codebook. True builders always survive because they set 3 bits that are all inside `U_u(t) ⊆ estimate`."),
    code(
        "P = 10\n"
        "U, S, user_tokens, vocabs = simulate(P)\n"
        "steps = len(S) - 1\n"
        "uid, t = 0, 20\n"
        "\n"
        "# global LUT: bit -> set of tokens that ever hit it (any user)\n"
        "lut = {}\n"
        "for v in vocabs:\n"
        "    for tok in v:\n"
        "        h = mmh3.hash64(tok.encode(), seed=0, signed=False)[0]\n"
        "        M = 1 << P\n"
        "        reg = h & (M - 1)\n"
        "        rem = h >> P\n"
        "        tz = 31 if rem == 0 else min((rem & -rem).bit_length() - 1, 31)\n"
        "        lut.setdefault(reg * 32 + tz, set()).add(tok)\n"
        "\n"
        "est = S[t] & U[uid][steps]\n"
        "candidates = set()\n"
        "bit = 0\n"
        "x = est\n"
        "while x:\n"
        "    low = x & -x\n"
        "    bit = low.bit_length() - 1\n"
        "    candidates |= lut.get(bit, set())\n"
        "    x ^= low\n"
        "\n"
        "user_vocab = set(vocabs[uid])\n"
        "restored = {tok for tok in candidates if tok in user_vocab}\n"
        "truth = user_tokens[uid][t]\n"
        "recall = len(truth & restored) / len(truth)\n"
        "precision = len(truth & restored) / len(restored) if restored else 0.0\n"
        "print(f\"user {uid} at t={t}: true builders {len(truth)}, restored {len(restored)}\")\n"
        "print(f\"recall {recall:.3f} | precision {precision:.3f}\")\n"
        "print(\"all true builders recovered:\", truth <= restored)"
    ),
    md(
        "## Summary\n"
        "\n"
        "- **Joining per-user lattices into the shared lattice is safe**: the\n"
        "  union lives on the same bit plane, and the shared commit DAG already\n"
        "  stores `c(t)` per commit.\n"
        "- **Restore is one intersection**: `c(t) ∩ U_u(now)` is a superset of\n"
        "  `U_u(t)`; the only error is early attribution from cross-user bit\n"
        "  overlap — and it shrinks as P-precision grows (more registers).\n"
        "- **Per-user codebook filtering makes materialization near-exact**: the\n"
        "  global LUT fiber may carry other users' tokens, but keeping only the\n"
        "  user's own codebook restores the true builders.\n"
        "\n"
        "Rust wiring (proposed): per-user HLLSets as branches/gates in\n"
        "`ewm-git` (or a per-user store), `c(t)` already exists, and a new\n"
        "`ewm-sm-explore project-user <user> <commit>` command that computes\n"
        "`c(t) ∩ U_user(now)` + codebook-filtered materialization."
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
