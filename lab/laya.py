# laya.py — a raw client for the `laya-jsonl` daemon.
#
# Laya is the open-source System One decision model (non-autoregressive
# ModernBERT-large encoder + RL decision head, pure Rust on candle). The
# `laya-jsonl` binary loads the checkpoint once and answers many states
# over stdin/stdout, one JSON object per line.
#
# Protocol (verbatim from src/bin/laya-jsonl.rs):
#   in:  {"id": 1, "state": "prose or JSON", "questions": {...}}
#   out: {"id": 1, "response": {"model": "...", "answers": {...}, "usage": {...}}}
#
# Question types (src/question.rs):
#   choice -> {"type": "choice", "instructions": "...", "criteria": {name: desc}}
#   score  -> {"type": "score",  "instructions": "...", "criteria": ["lvl0", ...]}
#   noul   -> {"type": "noul",  "instructions": "a statement that holds or not"}
#
# The rendered sequence per question is:
#   [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
# with per-option text truncated to 48 tokens, the head to head_max_len
# (256) and the state to the remaining room in max_len (1024).

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

DEFAULT_BIN = os.path.expanduser("~/tools/laya-rust/target/release/laya-jsonl")
DEFAULT_MODEL = os.path.expanduser("~/.cache/laya/typed-decisions")


class LayaDaemon:
    """A persistent laya-jsonl process plus a transparent request helper."""

    def __init__(
        self,
        bin_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.bin = bin_path or os.environ.get("LAYA_BIN", DEFAULT_BIN)
        self.model_dir = model_dir or os.environ.get("LAYA_MODEL", DEFAULT_MODEL)
        self.device = device
        self.mock = not (
            os.path.exists(self.bin)
            and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))
        )
        self._proc = None
        self._req_id = 0
        if self.mock:
            raise FileNotFoundError(
                f"laya daemon or checkpoint missing ({self.bin}, {self.model_dir})"
            )

    def start(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self.bin, "--model", self.model_dir, "--device", self.device],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        return self

    def request(self, req: dict, verbose: bool = True) -> dict:
        """Send one raw JSON line; return the parsed `{"id", "response"}`."""
        self.start()
        self._req_id += 1
        req = {"id": self._req_id, **req}
        line = json.dumps(req)
        if verbose:
            print("── laya request ──")
            print(json.dumps(req, indent=2))
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline()
        resp = json.loads(out)
        if verbose:
            print("── laya response ──")
            print(json.dumps(resp, indent=2))
        return resp

    def route(
        self,
        state,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
        verbose: bool = True,
    ) -> dict:
        """The controller-facing helper used by bonsai-ewm's auto advisor.

        Builds a `choice` question for the route plus optional `noul`
        questions, sends them batched in one forward pass, and returns a
        plain decision dict (decision, confidence, probabilities, aux,
        model, usage, raw).
        """
        questions = {
            "route": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(options),
            }
        }
        for name, instr in (extra_noul or {}).items():
            questions[name] = {"type": "noul", "instructions": instr}

        resp = self.request({"state": state, "questions": questions}, verbose=verbose)
        r = resp["response"]
        ans = r["answers"]["route"]
        probabilities = {str(k): float(v) for k, v in ans.get("probabilities", {}).items()}
        aux = {
            name: float(r["answers"][name].get("noul", 0.0))
            for name in (extra_noul or {})
        }
        usage = r.get("usage", {})
        return {
            "decision": str(ans.get("choice", "")),
            "confidence": float(ans.get("confidence", 0.0)),
            "probabilities": probabilities,
            "aux": aux,
            "model": str(r.get("model", "laya")),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "mock": False,
            "raw": resp,
        }

    @staticmethod
    def prose_state(query: str, memory_tokens: int = 0, bss: Optional[dict] = None) -> str:
        """The prose shape Laya rewards (from the trainer's LayaAdapter).

        Laya's encoder prefers a natural-language situation description over
        raw structs: put the query first, name the situation, keep opaque
        tid memory short and framed, and report coverage numbers plainly.
        """
        parts = []
        query = str(query).strip()
        if query:
            parts.append(f'The user asked: "{query}".')
        if memory_tokens is not None:
            parts.append(
                f"The system's memory holds {memory_tokens} content-addressed token ids "
                f"from earlier answers."
            )
        if bss:
            parts.append(
                "The models' current coverage of the conversation is: "
                + ", ".join(f"{k} {float(v):.3f}" for k, v in bss.items())
                + "."
            )
        return " ".join(parts)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None
