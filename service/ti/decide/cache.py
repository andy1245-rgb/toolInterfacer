"""cache — bounded LRU of `choice` answers for the fast path in resolve.py.

On the CPU-only box a /v1/systemone call costs seconds, and the loop asks the same
question more than once: the same candidates, the same instructions, the same
intent, reached again through a retry, a second template step or the user
repeating themselves. A repeat is served from here in microseconds.

What a key is: a digest of (state, instructions, criteria) — the entire input of
ChoiceAnswer and nothing else. The same inputs have the same answer, so a hit
cannot be told from a fresh call: get() returns a copy, which is also why a
caller that mutates the probabilities it was handed cannot poison the entry.
The digest, not the raw state, is stored: a state may be ~2000 tokens, and the
table's footprint must not scale with the size of the questions in it.

What is *not* cached: `noul` (a per-artifact verdict the loop rewrites between
steps) and `extract` (already model-free in rule_extract). Only the choice path
of FastPathBackend reads this module.

A bench that repeats a state gets the stored answer rather than a fresh call:
clear() first when the measurement is about the model (bench/live_probe.py's rows
are distinct utterances, so it is unaffected).

Threading: the service is a ThreadingHTTPServer, so every method holds one lock.
The LRU is an OrderedDict; a hit moves its key to the end, a put over capacity
drops from the front, so the table stays bounded at decide.cache.max_entries.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from .primitives import ChoiceAnswer

DEFAULT_MAX_ENTRIES = 256


def choice_key(state: Any, instructions: str, criteria: dict[str, str]) -> str:
    """Stable identity of one choice question: digest(state, instructions, criteria).

    json.dumps does the normalisation: sort_keys makes dict ordering irrelevant
    (the state is assembled by several producers), sorting the criteria items
    does the same for the candidate map, and default=str keeps one unserialisable
    value from raising inside the decision path.
    """
    blob = json.dumps([state, str(instructions), sorted(criteria.items())],
                      sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _copy(answer: ChoiceAnswer) -> ChoiceAnswer:
    return ChoiceAnswer(answer.choice, dict(answer.probabilities), answer.confidence)


class DecisionCache:
    """Bounded LRU from choice_key() to ChoiceAnswer; see the module docstring."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[str, ChoiceAnswer] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> ChoiceAnswer | None:
        """A copy of the answer stored for `key` (marking it recent), or None."""
        with self._lock:
            answer = self._items.get(key)
            if answer is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return _copy(answer)

    def put(self, key: str, answer: ChoiceAnswer) -> None:
        """Store a copy of `answer`, evicting the least recently used entry."""
        with self._lock:
            self._items[key] = _copy(answer)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """Forget every entry and the counters (tests, and a config reload)."""
        with self._lock:
            self._items.clear()
            self._hits = self._misses = self._evictions = 0

    def stats(self) -> dict[str, Any]:
        """entries / max_entries / hits / misses / evictions / hit_rate."""
        with self._lock:
            seen = self._hits + self._misses
            return {"entries": len(self._items),
                    "max_entries": self.max_entries,
                    "hits": self._hits,
                    "misses": self._misses,
                    "evictions": self._evictions,
                    "hit_rate": (self._hits / seen) if seen else 0.0}


_SHARED: DecisionCache | None = None
_SHARED_LOCK = threading.Lock()


def cache_cfg() -> dict[str, Any]:
    """The decide.cache block, copied (load() hands out the shared cache)."""
    from ..conf import load
    return dict((load().get("decide") or {}).get("cache") or {})


def enabled() -> bool:
    """decide.cache.enabled — on unless the block says otherwise."""
    return bool(cache_cfg().get("enabled", True))


def shared() -> DecisionCache:
    """The process-wide cache, rebuilt when decide.cache.max_entries changes.

    resolve() builds a backend per hop, so the table cannot live on a backend
    instance; one locked singleton is what makes a repeat cheap across hops.
    """
    global _SHARED
    max_entries = max(1, int(cache_cfg().get("max_entries", DEFAULT_MAX_ENTRIES)))
    with _SHARED_LOCK:
        if _SHARED is None or _SHARED.max_entries != max_entries:
            _SHARED = DecisionCache(max_entries)
        return _SHARED


def clear() -> None:
    """Empty the process-wide cache."""
    shared().clear()


def stats() -> dict[str, Any]:
    """The process-wide cache's counters — what DecisionCache.stats() reports."""
    return shared().stats()
