"""backends/ — one file per decision-model backend.

A backend registers itself when its module is imported, so this package imports
the ones ``primitives._ensure_backends_loaded`` does not name: make_backend()
then finds every backend whichever module a caller imported first.
"""
from . import openrouter  # noqa: F401  (registration)
