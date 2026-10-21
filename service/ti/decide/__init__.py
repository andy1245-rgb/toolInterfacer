"""decide — the decision-model wall.

Pipelines and the runner address decisions only through primitives.py
(choice / noul / score / extract). Single backends live in backends/; composites
that route between them (cascade.py) register against the same module, so
cf/defaults.json picks whichever serves each usage by name.
"""
