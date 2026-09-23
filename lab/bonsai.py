# bonsai.py — a raw HTTP client for the PrismML Bonsai llama.cpp server.
#
# The PrismML llama.cpp fork exposes the OpenAI-compatible chat API *plus*
# llama.cpp's /tokenize and /detokenize endpoints. That bidirectional
# token access is the property that makes the context a proposal: the
# controller can turn lattice tids into text (detokenize) and turn Bonsai's
# answer back into tids (tokenize).
#
# Endpoints used by the chain (see docs/INTERACTION_MAP.md):
#   GET  /health                -> {"status": "ok"}
#   POST /tokenize              -> {"tokens": [123, ...]}
#   POST /detokenize            -> {"content": "text"}
#   POST /v1/chat/completions   -> OpenAI-style chat response (+ reasoning_content)

from __future__ import annotations

import json
import urllib.request
from typing import Optional


class BonsaiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, verbose: bool = False):
        if verbose:
            print(f"GET {self.base_url}{path}")
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if verbose:
            print(json.dumps(data, indent=2))
        return data

    def _post(self, path: str, payload: dict, verbose: bool = False) -> dict:
        if verbose:
            print(f"POST {self.base_url}{path}")
            print(json.dumps(payload, indent=2))
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if verbose:
            print(json.dumps(data, indent=2))
        return data

    def health(self, verbose: bool = False) -> bool:
        try:
            return str(self._get("/health", verbose=verbose).get("status")) == "ok"
        except Exception:
            if verbose:
                print("health: server unreachable")
            return False

    def tokenize(self, text: str, verbose: bool = False) -> list[int]:
        r = self._post("/tokenize", {"content": text}, verbose=verbose)
        return list(r.get("tokens", []))

    def detokenize(self, tokens: list[int], verbose: bool = False) -> str:
        r = self._post("/detokenize", {"tokens": tokens}, verbose=verbose)
        return str(r.get("content", ""))

    def chat_full(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> dict:
        r = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            verbose=verbose,
        )
        msg = r["choices"][0]["message"]
        return {
            "content": str(msg.get("content") or ""),
            "reasoning": str(msg.get("reasoning_content") or ""),
            "usage": dict(r.get("usage", {})),
            "raw": r,
        }

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kw):
        full = self.chat_full(prompt, max_tokens=max_tokens, temperature=temperature, **kw)
        return full["content"], full["reasoning"]
