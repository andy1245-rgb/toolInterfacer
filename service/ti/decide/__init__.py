"""decide — the decision-model wall.

Pipelines and the runner address decisions only through primitives.py
(choice / noul / score / extract). Single backends live in backends/; composites
that route between them (cascade.py) register against the same module, so
cf/defaults.json picks whichever serves each usage by name.
"""
from __future__ import annotations

# The wall is addressed as `ti.decide.resolve(kind)` where a caller imports the
# package rather than the module, and without this re-export that import binds
# the *submodule*: the first call then dies on "'module' object is not callable"
# (service_entry's noul verifier did exactly that, so no post-gate could run).
# Bound last and eagerly — importing the submodule sets `ti.decide.resolve` to
# the module, so the function has to be bound after that import has happened.
from .resolve import resolve as resolve  # noqa: E402,F401  (public entry point)
