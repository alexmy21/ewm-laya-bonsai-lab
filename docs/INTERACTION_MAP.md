# Interaction map — ewm-sm sidecar ↔ Rust Laya ↔ Bonsai

This document is the reference for the notebook collection in
`notebooks/`. It records, at protocol level, exactly what crosses each of
the three boundaries. Everything below was verified live on this machine
(September 2026) unless marked otherwise.

---

## 0. The three systems and their boundaries

```text
┌────────────────────────────────────────────────────────────────────┐
│                         controller (Python)                        │
│                                                                    │
│   ewm-scene CLI ◄──────► Laya daemon ◄──────► Bonsai llama.cpp     │
│   (HLLSet lattice)      (System One)         (base model)          │
└────────────────────────────────────────────────────────────────────┘

boundary 1                boundary 2            boundary 3
subprocess                subprocess            HTTP
JSONL file → JSON doc     JSONL stdin/stdout    JSON in / JSON out
```

| System | Artifact | Transport | Direction |
| --- | --- | --- | --- |
| ewm-sm sidecar | `ewm-scene` binary (`ewm-state-machine` repo) | subprocess, file in / JSON out | both ways (read state, write frames) |
| Rust Laya | `laya-jsonl` daemon (`~/tools/laya-rust`) | subprocess, JSONL stdin/stdout | request → typed answer |
| Bonsai 2 27B | PrismML llama.cpp `llama-server` (`~/tools/Bonsai-demo`) | HTTP on `127.0.0.1:8081` | request → text + usage |

The controller is deliberately **outside** all three systems. It is the
only place where their vocabularies meet:

- ewm-sm speaks `tid{n}` strings and HLLSet statistics.
- Laya speaks prose states and typed questions.
- Bonsai speaks token ids, detokenized text, and chat completions.

---

## 1. Boundary 1 — the ewm-sm sidecar (`ewm-scene`)

### 1.1 Binary and invocation

- Binary: `ewm-state-machine/target/{debug,release}/ewm-scene`
  (release build: `cargo build --release -p ewm-scene`).
- Resolution order in the lab: `EWM_SCENE_BIN` env → `ewm-scene` on PATH →
  sibling repo release → sibling repo debug.
- Every call is one subprocess invocation:

```bash
ewm-scene <command> <frames.jsonl> [options]
```

### 1.2 Input format

One JSON object per line, `{"id": 1, "tokens": ["tid12", ...]}`. The
`tid{n}` strings are the controller's lingua franca: Bonsai token ids
lowered to strings (`tid1234`).

```jsonl
{"id":1,"tokens":["tid1","tid2","tid3","tid4"]}
{"id":2,"tokens":["tid2","tid3","tid5"]}
{"id":3,"tokens":["tid5","tid6","tid7"]}
```

Each frame's tokens are ingested into an HLLSet (`G1 ∪ G2 ∪ G3`
projection, n-seed, no PAD). The **popcount** is the number of set bits,
so it is roughly `3 × distinct tokens` (three seeds per token).

### 1.3 Commands and output schemas

Flat-frame commands (`ingest | bss | ma | noether | materialize | sidecar |
boolring | project`):

| Command | Output | Semantics |
| --- | --- | --- |
| `ingest` | `{"frames":[{"id","key","pop"}...]}` | per-frame content key (`h:sha1…`) and popcount |
| `bss` | `{"tau":[...], "jaccard":[...], "pop":[...], "keys":[...]}` | consecutive BSSτ (`\|A∩B\|/\|B\|`) and Jaccard |
| `ma --short N --long N` | `{"t0":[...], "fast":[...], "slow":[...]}` | moving-average BSS over trailing union windows |
| `noether` | `{"dp":[u64...], "rp":[...], "np":[...], "ind1":[f64...], "ind2":[...], "ind3":[...]}` | per-transition D/R/N popcounts + 3 indicators |
| `materialize [--beam N]` | `{"frames":[{"id","ordered","set","beam2"}]}` | token restoration (greedy order, plain set, beam) |
| `sidecar [--cap N] [--freeze N]` | `{"frames":[{"soft","hard","basis_pop","residual","in_span","dim","rotation_count","rotation_mass","step","soft_first","spill_first"}], "jumps", "threshold", "basis_history"}` | Boolean-ring trajectory, soft/hard keys, jump detector (mean+3σ) |
| `project --frame <f.json>` | `{"names","pops","frames":[{"id","intersections","bss"}]}` | BSS coordinates of the stream over a frame of named dimensions |

Structured-frame commands (`grid | tensor | pyramid | subframes`) use their
own frame formats; the flat path above is the one the Bonsai chain uses.

### 1.4 The Noether decomposition (used by `surprise` and by Laya features)

For each consecutive frame pair `(A, B)`:

```text
D = A \ B      (departed)      dp = |D|
R = A ∩ B      (retained)      rp = |R|
N = B \ A      (new)           np = |N|

ind1 = dp / |R ∪ N|      departure dominates what remains+arrives
ind3 = np / |R ∪ D|      arrival dominates what remains+departed
ind2 = BSS(R_prev, R)    retention-chain stability (starts at 2nd transition)
```

`ind2` is a **similarity** — low values mean the retained structure
changed, which is surprising. `ind1`/`ind3` are high when surprised.

### 1.5 Known edge cases

- `materialize` **panics** ("attempt to subtract with overflow") on frames
  with 1–2 tids. `ingest` and `noether` are fine on tiny frames. The
  controller must fall back to raw tids for such frames (bonsai-ewm does).
- Hash collisions are possible (HLLSet), so D/R/N counts are approximate.
- `noether` series length: `N-1` transitions for `N` frames; `ind2` has
  `N-3` entries (it starts at transition 2).

---

## 2. Boundary 2 — the Laya daemon (`laya-jsonl`)

### 2.1 Daemon and lifecycle

```bash
~/tools/laya-rust/target/release/laya-jsonl \
  --model ~/.cache/laya/typed-decisions --device cpu
```

- Loads the checkpoint once; prints `laya-jsonl: ready` to **stderr**.
- Then answers many requests over **stdin/stdout**, one JSON object per
  line, until stdin closes.
- Checkpoint: ModernBERT-large encoder + 2-layer RL head,
  `laya-typed-decisions` (`fine_tuned: true`), f32 on CPU here (~0.6 s per
  decision). CUDA build is blocked on this machine (nvcc PTX ISA mismatch).
- `device` may be `cpu`, `metal`, or `cuda`.

### 2.2 Wire protocol

Request:

```json
{
  "id": 1,
  "state": "The user asked: \"What is the capital of France?\". ...",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "How should the next context be built?",
      "criteria": {"full": "keep the full materialized memory", "compact": "keep only new tids"}
    },
    "keep_short": {"type": "noul", "instructions": "The context should be kept short"}
  }
}
```

Response (note: **answers are sorted alphabetically by question name** —
the daemon uses a `BTreeMap`):

```json
{
  "id": 1,
  "response": {
    "model": "laya-typed-decisions",
    "answers": {
      "keep_short": {"type": "noul", "noul": 0.2724, "rl_agent": {"act_probability": 1.0}},
      "route": {
        "type": "choice",
        "choice": "compact",
        "probabilities": {"full": 0.4322, "compact": 0.5678},
        "confidence": 0.0133,
        "rl_agent": {"act_probability": 1.0}
      }
    },
    "usage": {"input_tokens": 104, "output_tokens": 0}
  }
}
```

### 2.3 Question types

| Type | Request shape | Answer fields |
| --- | --- | --- |
| `choice` | `{"type":"choice","instructions":str,"criteria":{name:desc}}` (or a list) | `choice`, `probabilities{name:p}`, `confidence` |
| `score` | `{"type":"score","instructions":str,"criteria":["lvl0",...]}` | `score` (expectation over level indices), `legend`, `probabilities`, `confidence` |
| `noul` | `{"type":"noul","instructions":str}` | `noul` = P(statement holds) |

`confidence` = 1 − normalized entropy of the answer distribution.
`rl_agent.act_probability` = the act head's probability of answering
rather than escalating. All questions for one state go through **one
batched forward pass** — five questions cost about as much as one.

### 2.4 What the model actually sees

Per question, the encoder sequence is:

```text
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

Budgets (from `rl_agent_config.json`):

- `max_len = 1024` — the whole sequence.
- `head_max_len = 256` — instructions + options; instructions are truncated
  to whatever the options leave (min 8 tokens).
- each option text truncated to **48 tokens**.
- the state is serialized (JSON compactly, prose verbatim) and truncated to
  the remaining room; the response does **not** flag truncation on the
  JSONL path (the web console does).

Calibration temperatures are per (type, option-count) bucket, e.g.
`choice:2 → 1.906`, `choice:3-5 → 1.760`, `noul:2 → 1.983`.

### 2.5 Prose state matters

The fine-tuned checkpoint rewards **natural-language states**, not raw
structs. The trainer's `LayaAdapter._prose_state` shapes it as:

```text
The user asked: "<query>". The system's memory holds <N> content-addressed
token ids from earlier answers. The models' current coverage of the
conversation is: llm_a 0.550, llm_b 0.320, llm_c 0.210.
```

---

## 3. Boundary 3 — the Bonsai server (PrismML llama.cpp)

### 3.1 Server

```bash
cd ~/tools/Bonsai-demo
LD_LIBRARY_PATH="$PWD/bin/cuda" ./bin/cuda/llama-server \
  -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa on -c 2048 --host 127.0.0.1 --port 8081
```

- Do **not** set `CUDA_VISIBLE_DEVICES`: the PrismML fork orders GPUs
  opposite to `nvidia-smi` (its CUDA0 = the RTX 3060).
- Model: ternary 27B, 7.2 GB GGUF. ~8 tok/s on the 3060. 262K context
  plane with prompt/KV cache (see the repo's `KV-CACHE.md`,
  `PROMPT-CACHE.md`).

### 3.2 Endpoints

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `{"status":"ok"}` |
| `POST /tokenize` | `{"content":"text"}` | `{"tokens":[123, ...]}` |
| `POST /detokenize` | `{"tokens":[123, ...]}` | `{"content":"text"}` |
| `POST /v1/chat/completions` | OpenAI-style (below) | OpenAI-style + `reasoning_content` |

Chat request used by the chain:

```json
{
  "messages": [{"role": "user", "content": "Context memory (compact):\n...\n\nQuery: ..."}],
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": false
}
```

Chat response fields of interest:

```json
{
  "choices": [{"message": {"content": "...", "reasoning_content": "..."}}],
  "usage": {"prompt_tokens": 92, "completion_tokens": 24, "total_tokens": 116}
}
```

Bonsai is a **reasoning** model: it emits a reasoning trace
(`reasoning_content`) before the answer (`content`). `prompt_tokens` is the
honest per-turn cost metric the chain records.

The `tokenize`/`detokenize` endpoints are what make the collaboration
possible: the controller turns `tid{n}` memory into text for the prompt,
and turns the answer text back into `tid{n}` frames for the lattice.

---

## 4. Interaction chain 1 — ewm-sm → Laya

This is the **context-policy decision**. The lattice is a feature
extractor; Laya is a typed-decision head.

### 4.1 Feature extraction from ewm-scene

For the current union file, the controller pulls:

```python
uni  = ewm.ingest(union_path)      # per-frame pop / key
noe  = ewm.noether(union_path)     # dp, rp, np, ind1, ind2, ind3
side = ewm.sidecar(union_path)     # soft/hard keys, residual, step, jumps
bss  = ewm.bss(union_path)         # consecutive BSSτ / Jaccard
```

Typical trajectory features handed to Laya:

```text
turn, union_pop, full_memory_tids, novelty_tids, last dp/ind1/ind2/ind3,
soft-key step, in_span, residual, jaccard to previous frame
```

### 4.2 The request

The features are rendered into a prose state and sent with one `choice`
question (the route) plus optional `noul` questions (e.g. `keep_short`):

```text
route:      "How should the next context be built for the base model?"
  options:  full / compact / surprise
keep_short: "The context should be kept short"            (noul)
```

### 4.3 The response and how it is used

```python
decision = resp["response"]["answers"]["route"]["choice"]
confidence = ...["confidence"]
keep_short = ...["answers"]["keep_short"]["noul"]
```

In bonsai-ewm's `auto` mode: `keep_short > 0.5 → cap = 8`, otherwise the
policy cap. In the trainer's multi-LLM loops: choices below a confidence
floor are **gated to a fallback** model, and the decision itself is lowered
to `jev_*` tokens and ingested into the union so the decision becomes a
first-class frame in the lattice.

---

## 5. Interaction chain 2 — Laya → Bonsai

The typed decision is turned into a **context proposal** for the base
model.

```text
Laya decision (mode, cap)
      │
      ▼
proposal_tids(mode, cap)          # full | compact | surprise from S(t)
      │
      ▼
detokenize([int(t[3:]) for t in tids])     # tids → text via Bonsai /detokenize
      │
      ▼
prompt = f"Context memory ({mode}):\n{prefix_text}\n\nQuery: {query}"
      │
      ▼
POST /v1/chat/completions                   # Bonsai reasons + answers
```

Mode semantics (from bonsai-ewm):

| Mode | tids that enter the prefix |
| --- | --- |
| `full` | `memory_tokens(materialize(S))`, last `cap` |
| `compact` | D-part (never-seen tids), last `cap` |
| `surprise` | D-part + full frames whose Noether indicators crossed thresholds |

The per-turn record includes `prefix_tokens` (controller-side count) and
Bonsai's own `prompt_tokens`, so the decision's cost is measurable.

---

## 6. Interaction chain 3 — Bonsai → ewm-sm

The answer returns to the lattice as a new frame.

```text
answer text + reasoning
      │
      ▼
tokenize(answer)                    # POST /tokenize
      │
      ▼
tids = [f"tid{i}" for i in tokens]
      │
      ▼
append {"id": t+1, "tokens": tids} to union JSONL
      │
      ▼
ewm.ingest / ewm.materialize / ewm.noether    # S(t) updates
```

The displacement (D-part) filter runs over the answer tids: only tids
never seen before enter the novelty stream. The full materialized memory is
the restored order of the last frame (with a raw-tid fallback when the
materializer hits the tiny-frame panic).

---

## 7. The full loop on one page

```text
                    ┌───────────────────────────────────────────────┐
                    │                 controller                    │
 user query ───────►│                                               │
                    │  1. ewm-scene ingest/noether/sidecar          │
                    │       → trajectory features                   │
                    │  2. prose state ──► laya-jsonl daemon         │
                    │       ← typed decision (mode, cap, keep_short)│
                    │  3. proposal_tids(mode, cap)                  │
                    │  4. Bonsai /detokenize → prefix text          │
                    │  5. Bonsai /v1/chat/completions               │
                    │       ← answer + reasoning + prompt_tokens    │
                    │  6. Bonsai /tokenize → tid stream             │
                    │  7. append frame → ewm-scene S(t)             │
                    └───────┬────────────────┬───────────────┬──────┘
                            │                │               │
                     subprocess         subprocess        HTTP
                     JSONL→JSON        JSONL lines       JSON docs
                            │                │               │
                     ┌──────▼────┐   ┌───────▼──────┐  ┌─────▼──────┐
                     │ ewm-scene │   │ laya-jsonl   │  │ Bonsai     │
                     │ (lattice) │   │ (System One) │  │ llama.cpp  │
                     └───────────┘   └──────────────┘  └────────────┘
```

---

## 8. Latency and cost profile (measured on this machine)

| Step | Cost |
| --- | --- |
| ewm-scene ingest/noether/sidecar | milliseconds (Rust, small files) |
| laya-jsonl decision | ~0.6 s/decision on CPU (one forward pass) |
| Bonsai chat | ~8 tok/s on the RTX 3060; a 24-token answer ≈ 3 s |
| Bonsai tokenize/detokenize | milliseconds |

---

## 9. Failure modes and fallbacks

| Failure | Behaviour |
| --- | --- |
| `ewm-scene` missing | `resolve_bin()` falls back to PATH/sibling; the client raises a clear `RuntimeError` on non-zero exit |
| `materialize` on 1–2-tid frames | Rust panic → controller falls back to raw tids (bonsai-ewm) |
| Laya binary or checkpoint missing | `LayaDaemon` raises `FileNotFoundError`; trainers fall back to a mock decision |
| Laya state too long | state is truncated to `max_len`; web console flags it, JSONL path does not |
| Bonsai `/health` DOWN | controller prints `health=DOWN` and keeps running; questions raise `URLError` |
| Bonsai chat returns empty content | the reasoning trace consumed the whole `max_tokens` budget — use a larger `max_tokens` (128+) or the server's `--reasoning-preserve`; the lattice then ingests an empty frame (`pop=0`) |
| Decision confidence too low | trainer gates the route to a fallback model; bonsai-ewm auto mode uses the choice as-is |
| No `TYPESAFE_API_KEY` | the Jev adapter (bonsai-ewm) uses a deterministic mock; Laya is the local alternative |

---

## 10. Source locations

| Artifact | Path |
| --- | --- |
| ewm-scene CLI main | `ewm-state-machine/crates/ewm-scene/src/main.rs` |
| ewm-scene frame math | `ewm-state-machine/crates/ewm-scene/src/lib.rs` |
| Laya daemon | `~/tools/laya-rust/src/bin/laya-jsonl.rs` |
| Laya question rendering | `~/tools/laya-rust/src/question.rs` |
| Laya response shapes | `~/tools/laya-rust/src/lib.rs` |
| Laya checkpoint config | `~/.cache/laya/typed-decisions/rl_agent_config.json` |
| Bonsai server docs | `~/tools/Bonsai-demo/{README,KV-CACHE,PROMPT-CACHE,SPECULATIVE}.md` |
| Trainer adapters (reference) | `ewm-state-machine/trainer/{ewm,adapters,bonsai_cli}.py` |
| Production controller | `bonsai-ewm/src/bonsai_ewm/{ewm,adapters,session,cli}.py` |
| Research notebooks on Laya | `ewm-state-machine/notebooks/16_*`, `17_*`, `18_*` |
