"""Budgeted model client for eval-only packs (frozen; handed to the surface by the harness).

Every surface model call goes through this client: temperature 0, per-scope (question/task) token caps, capped
completion length, no thinking. The runner independently measures the service's token counters (budget.unit
`service_tokens`), so this client is the per-item guard and the runner is the absolute one.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request


class BudgetExceeded(Exception):
    pass


class BudgetedClient:
    def __init__(self, base_url: str, model: str, scope_cap: int, max_completion: int, timeout_s: float = 300):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.scope_cap, self.max_completion, self.timeout_s = scope_cap, max_completion, timeout_s
        self._used: dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls = 0

    def used(self, scope: str) -> int:
        return self._used.get(scope, 0)

    def total(self) -> int:
        return sum(self._used.values())

    def scoped(self, scope: str) -> "ScopedClient":
        return ScopedClient(self, scope)

    def chat(self, scope: str, messages: list[dict], max_tokens: int | None = None, stop: list[str] | None = None) -> str:
        with self._lock:
            left = self.scope_cap - self._used.get(scope, 0)
        if left <= 0:
            raise BudgetExceeded(f"{scope}: token cap {self.scope_cap} reached")
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "seed": 0,
                "max_tokens": min(max_tokens or self.max_completion, self.max_completion),
                "chat_template_kwargs": {"enable_thinking": False}}
        if stop:
            body["stop"] = stop
        out = self._post("/v1/chat/completions", body)
        u = out.get("usage") or {}
        with self._lock:
            self._used[scope] = self._used.get(scope, 0) + int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
            self.calls += 1
        return out["choices"][0]["message"]["content"] or ""

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 400:  # e.g. prompt longer than max_model_len: the caller's problem, not transient
                    raise ValueError(f"model rejected the request: {e.read()[:300]!r}") from None
                if attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


class ScopedClient:
    """What the surface sees: chat() bound to one question/task scope."""

    def __init__(self, client: BudgetedClient, scope: str):
        self._c, self.scope = client, scope

    def chat(self, messages: list[dict], max_tokens: int | None = None, stop: list[str] | None = None) -> str:
        return self._c.chat(self.scope, messages, max_tokens, stop)

    @property
    def tokens_left(self) -> int:
        return self._c.scope_cap - self._c.used(self.scope)
