#!/usr/bin/env python3
"""Build notebooks/24_user_context_training.ipynb in the repo's style."""

import json

NB_PATH = "/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-laya-bonsai-lab/notebooks/09_user_context_training.ipynb"

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
        "# 09 — Per-user typed heads: built-in training for enterprise contexts\n"
        "\n"
        "Enterprise users solve their own small or big problems with a smaller,\n"
        "more consistent vocabulary. The ewm-sm apparatus stays **shared and\n"
        "vocabulary-agnostic** (`docs/SEPARATION.md`); what adapts per user is\n"
        "the **router** — here, a tiny typed head trained on the user's own\n"
        "analytical context.\n"
        "\n"
        "This notebook exercises `trainer/user_models.py`, the built-in support:\n"
        "\n"
        "```text\n"
        "UserContext -> VocabAdapter (user codebook)\n"
        "            -> collect_user_dataset (teacher loop -> routing_log.jsonl)\n"
        "            -> train_user_head (distill the teacher's soft labels)\n"
        "            -> register_user_model (registry)\n"
        "            -> UserHeadRouter.load(user_id) (serve)\n"
        "```\n"
        "\n"
        "Two synthetic enterprise users — `acme-finance` and `medco-clinical` —\n"
        "each with their own small codebook and query bank, each get their own\n"
        "head; then we cross-evaluate to show the per-user model matters.\n"
        "\n"
        "Registry layout:\n"
        "\n"
        "```text\n"
        "~/.cache/ewm-models/<user_id>/{config.json, trajectory.json,\n"
        "                              routing_log.jsonl, typed_head.pt, metrics.json}\n"
        "```"
    ),
    code(
        "import os, sys, json\n"
        "\n"
        "# Same GPU pin as notebooks 21–23: with PCI_BUS_ID on this machine the\n"
        "# Quadro M1200 is device 0 and the RTX 3060 is device 1.\n"
        "os.environ[\"CUDA_DEVICE_ORDER\"] = \"PCI_BUS_ID\"\n"
        "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"\n"
        "os.environ.setdefault(\"HF_HUB_DISABLE_PROGRESS_BARS\", \"1\")\n"
        "\n"
        "import numpy as np\n"
        "\n"
        "EWM_SM = \"/home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-state-machine\"\n"
        "sys.path.insert(0, EWM_SM)\n"
        "\n"
        "from trainer import (\n"
        "    DEFAULT_MODEL_ROOT,\n"
        "    EwmScene,\n"
        "    StructuralLlmRouter,\n"
        "    UserContext,\n"
        "    UserHeadRouter,\n"
        "    VocabAdapter,\n"
        "    collect_user_dataset,\n"
        "    evaluate_router_on_log,\n"
        "    register_user_model,\n"
        "    train_user_head,\n"
        ")\n"
        "\n"
        "print(\"model root:\", DEFAULT_MODEL_ROOT)\n"
        "ewm = EwmScene()\n"
        "teacher = StructuralLlmRouter(max_new_tokens=96)   # shared prompted teacher\n"
        "print(\"teacher mode:\", \"mock\" if teacher.mock else \"real\", \"|\", teacher.model_id)"
    ),
    code(
        "USERS = [\n"
        "    UserContext(\n"
        "        user_id=\"acme-finance\",\n"
        "        vocab=[f\"fin_{i}\" for i in range(48)],\n"
        "        queries=[\n"
        "            \"What is EBITDA?\",\n"
        "            \"Explain our revenue recognition policy.\",\n"
        "            \"What is the current guidance on the debt-to-equity ratio?\",\n"
        "            \"Define working capital.\",\n"
        "            \"Summarize the Q3 cash flow statement.\",\n"
        "            \"What is the difference between OpEx and CapEx?\",\n"
        "            \"How do we calculate net margin?\",\n"
        "            \"What drives churn in the finance segment?\",\n"
        "        ],\n"
        "    ),\n"
        "    UserContext(\n"
        "        user_id=\"medco-clinical\",\n"
        "        vocab=[f\"med_{i}\" for i in range(48)],\n"
        "        queries=[\n"
        "            \"What are the contraindications for beta blockers?\",\n"
        "            \"Summarize the phase II trial protocol.\",\n"
        "            \"What is the normal range for serum creatinine?\",\n"
        "            \"Define the inclusion criteria for cohort A.\",\n"
        "            \"How do we stage chronic kidney disease?\",\n"
        "            \"What are the side effects of the study drug?\",\n"
        "            \"Explain the informed consent process.\",\n"
        "            \"What is the readmission rate benchmark?\",\n"
        "        ],\n"
        "    ),\n"
        "]\n"
        "print(\"users:\", [u.user_id for u in USERS])\n"
        "print(\"vocab sizes:\", [len(u.vocab or []) for u in USERS])\n"
        "print(\"queries per user:\", [len(u.queries) for u in USERS])"
    ),
    md("## §1 — Collect, train, register (one head per user)"),
    code(
        "registry = {}\n"
        "for ctx in USERS:\n"
        "    print(f\"\\n=== {ctx.user_id} ===\", flush=True)\n"
        "    adapter = VocabAdapter(vocab=ctx.vocab)\n"
        "    open_result, result = collect_user_dataset(ctx, adapter, teacher, ewm, T=12)\n"
        "    bundle, metrics = train_user_head(ctx, result.routing_log)\n"
        "    work_dir = register_user_model(ctx, bundle, metrics)\n"
        "    registry[ctx.user_id] = {\"ctx\": ctx, \"result\": result, \"open\": open_result,\n"
        "                             \"bundle\": bundle, \"metrics\": metrics}\n"
        "    print(f\"  teacher label histogram (llm_a/llm_b/llm_c): \"\n"
        "          f\"{[round(v) for v in metrics['teacher_label_histogram']]}\")\n"
        "    print(f\"  holdout acc {metrics['holdout_acc']:.2f} | KL {metrics['holdout_kl']:.4f} \"\n"
        "          f\"| {metrics['n_labelled']} labelled states\")\n"
        "    print(f\"  registry: {work_dir}\")\n"
        "    print(f\"  files: {sorted(os.listdir(work_dir))}\")"
    ),
    md("## §2 — Serve a per-user head in the loop"),
    code(
        "from trainer import run_jev_loop\n"
        "\n"
        "ctx_a = registry[\"acme-finance\"][\"ctx\"]\n"
        "adapter_a = VocabAdapter(vocab=ctx_a.vocab)\n"
        "own_head = UserHeadRouter.load(\"acme-finance\")\n"
        "open_a = registry[\"acme-finance\"][\"open\"]\n"
        "serve_result = run_jev_loop(\n"
        "    adapter_a, own_head, ewm, open_a, ctx_a.queries, ctx_a.work_dir,\n"
        "    T=8, max_new_tokens=32, structural=True, ingest_decision=True,\n"
        ")\n"
        "print(\"acme-finance served by its own head:\")\n"
        "for d in serve_result.decision_log:\n"
        "    print(f\"  {d.decision:6s} conf={d.confidence:.3f} model={d.model}\")"
    ),
    md("## §3 — Cross-user evaluation: does the per-user model matter?"),
    code(
        "head_fin = UserHeadRouter.load(\"acme-finance\")\n"
        "head_med = UserHeadRouter.load(\"medco-clinical\")\n"
        "\n"
        "print(f\"{'evaluated on':18s} {'own head':>20s} {'foreign head':>20s}\")\n"
        "for ctx in USERS:\n"
        "    log = registry[ctx.user_id][\"result\"].routing_log\n"
        "    own = evaluate_router_on_log(head_fin if ctx.user_id == \"acme-finance\" else head_med, log)\n"
        "    foreign = evaluate_router_on_log(head_med if ctx.user_id == \"acme-finance\" else head_fin, log)\n"
        "    print(f\"{ctx.user_id:18s} \"\n"
        "          f\"acc={own['agreement']:.2f} KL={own['mean_kl']:.3f}   \"\n"
        "          f\"acc={foreign['agreement']:.2f} KL={foreign['mean_kl']:.3f}\")"
    ),
    code(
        "%matplotlib inline\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "labels = [\"acme-finance\", \"medco-clinical\"]\n"
        "own_accs, for_accs = [], []\n"
        "for ctx in USERS:\n"
        "    log = registry[ctx.user_id][\"result\"].routing_log\n"
        "    own = evaluate_router_on_log(head_fin if ctx.user_id == \"acme-finance\" else head_med, log)\n"
        "    foreign = evaluate_router_on_log(head_med if ctx.user_id == \"acme-finance\" else head_fin, log)\n"
        "    own_accs.append(own[\"agreement\"]); for_accs.append(foreign[\"agreement\"])\n"
        "\n"
        "x = np.arange(len(labels))\n"
        "axes[0].bar(x - 0.18, own_accs, 0.36, label=\"own head\", color=\"tab:green\")\n"
        "axes[0].bar(x + 0.18, for_accs, 0.36, label=\"foreign head\", color=\"tab:red\")\n"
        "axes[0].set_xticks(x, labels)\n"
        "axes[0].set_ylim(0, 1.05)\n"
        "axes[0].set_ylabel(\"agreement with teacher\")\n"
        "axes[0].set_title(\"per-user head vs cross-user head\")\n"
        "axes[0].legend(fontsize=8)\n"
        "axes[0].grid(alpha=0.25)\n"
        "\n"
        "for ax, ctx in zip(axes[1:], USERS):\n"
        "    hist = registry[ctx.user_id][\"metrics\"][\"teacher_label_histogram\"]\n"
        "    ax.bar([\"llm_a\", \"llm_b\", \"llm_c\"], hist, color=[\"tab:blue\", \"tab:orange\", \"tab:purple\"])\n"
        "    ax.set_title(f\"{ctx.user_id}: teacher decision histogram\")\n"
        "    ax.grid(alpha=0.25)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(f\"{DEFAULT_MODEL_ROOT}/user_heads.png\", dpi=110)\n"
        "plt.show()"
    ),
    md(
        "## Summary\n"
        "\n"
        "- **Built-in per-user training now exists** (`trainer/user_models.py`):\n"
        "  `UserContext` → `VocabAdapter` → `collect_user_dataset` →\n"
        "  `train_user_head` → `register_user_model` → `UserHeadRouter.load`.\n"
        "- The apparatus stayed shared and vocabulary-agnostic; only the tiny\n"
        "  typed head is per-user (a few KB on disk, millisecond CPU inference).\n"
        "- The registry is a directory per user under `~/.cache/ewm-models/` with\n"
        "  `config.json` (feature spec), `routing_log.jsonl` (distillation data),\n"
        "  `typed_head.pt` (model), `metrics.json` (holdout acc/KL).\n"
        "- The cross-user evaluation shows whether the per-user context actually\n"
        "  matters: the own head should track its user's teacher better than the\n"
        "  foreign head does.\n"
        "\n"
        "Production follow-ups: per-user **codebooks** on the Rust side (smaller\n"
        "vocabularies make materialization exact — SEPARATION.md), incremental\n"
        "re-training as a user's `routing_log` grows, reward fine-tuning on loop\n"
        "surprise, and a model-server that loads `UserHeadRouter`s by user id."
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
