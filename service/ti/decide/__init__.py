"""decide — the decision-model wall.

Pipelines and the runner address decisions only through primitives.py
(choice / noul / score / extract). Backends live in backends/ and register
against that module; cf/defaults.json picks which backend serves each usage.
"""
