"""SimLLM runtime for vLLM Ascend.

The bundle descriptor can be inspected without importing torch or vLLM.
Runtime classes are loaded only when requested by the worker.
"""

from importlib import import_module

_EXPORTS = {
    "SimLLMConfig": "config",
    "CachedTask": "kv_manager",
    "KVManager": "kv_manager",
    "KVReuseEngine": "kv_reuse",
    "SimHashHasher": "lsh",
    "extract_embedding": "embedding",
    "MatchResult": "similarity",
    "SimilarityIdentifier": "similarity",
    "SandwichConfig": "sandwich",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
