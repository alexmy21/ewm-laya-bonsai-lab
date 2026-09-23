# ewm-laya-bonsai-lab

A notebook collection that dissects the interaction chain between three
systems:

```text
ewm-sm sidecar (ewm-scene)  ⇄  Rust Laya (laya-jsonl)  ⇄  Bonsai 2 27B (llama.cpp)
```

The point of this project is **visibility**. The three systems already
cooperate inside the `bonsai-ewm` controller; here we open every boundary
and watch the raw JSON cross it, one hop at a time.

## Start here

1. Read [`docs/INTERACTION_MAP.md`](docs/INTERACTION_MAP.md) — the
   protocol-level reference for all three boundaries (verified live on
   this machine).
2. Run the notebooks in order:

   | # | Notebook | Boundary studied |
   | --- | --- | --- |
   | 01 | `ewm_scene_protocol` | controller ⇄ ewm-sm sidecar |
   | 02 | `laya_daemon_protocol` | controller ⇄ Rust Laya |
   | 03 | `bonsai_http_surface` | controller ⇄ Bonsai |
   | 04 | `ewm_to_laya` | **interaction chain 1:** lattice features → typed context-policy decision |
   | 05 | `ewm_laya_bonsai_chain` | **the full chain:** ewm → Laya → Bonsai → ewm, per turn |
   | 06 | `decision_models_compared` | Laya vs Jev vs mock in the decision slot |

## Prerequisites (this machine)

- `ewm-scene` — built from the sibling `ewm-state-machine` repo
  (the notebooks resolve it automatically; override with `EWM_SCENE_BIN`).
- `laya-jsonl` — `~/tools/laya-rust/target/release/laya-jsonl` with the
  checkpoint at `~/.cache/laya/typed-decisions`.
- Bonsai server — start it before notebook 03/05:

  ```bash
  cd ~/tools/Bonsai-demo
  LD_LIBRARY_PATH="$PWD/bin/cuda" ./bin/cuda/llama-server \
    -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on -c 2048 --host 127.0.0.1 --port 8081
  ```

  (Do **not** set `CUDA_VISIBLE_DEVICES` — the PrismML fork lists the RTX
  3060 as CUDA0.) Notebook 05 degrades gracefully if Bonsai is down: it
  prints the requests it would have sent.

## Running

```bash
cd ewm-laya-bonsai-lab
jupyter notebook notebooks/
```

Or execute a notebook headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_ewm_scene_protocol.ipynb
```

## Layout

```text
ewm-laya-bonsai-lab/
├── docs/
│   └── INTERACTION_MAP.md   # the protocol reference (read this first)
├── lab/                     # transparent clients for the three systems
│   ├── ewm.py               # ewm-scene subprocess client
│   ├── laya.py              # laya-jsonl daemon client
│   └── bonsai.py            # Bonsai llama.cpp HTTP client
├── notebooks/               # the collection (generated)
├── scripts/
│   └── build_notebooks.py   # regenerate the notebooks from templates
└── work/                    # scratch JSONL frames + union files (gitignored)
```

## Regenerating the notebooks

The notebooks are generated from `scripts/build_notebooks.py` — edit the
templates there, then:

```bash
python3 scripts/build_notebooks.py
```

## Relation to the other projects

- `ewm-state-machine/` — the research line; notebooks 16–18 there first
  wired ewm-sm to Rust Laya. This project zooms into those interactions.
- `bonsai-ewm/` — the production controller; its `auto` advisor and
  session machinery are the end-user product. This project is the
  companion that explains the chain it automates.
