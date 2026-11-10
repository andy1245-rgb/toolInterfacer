"""harness — the shared measurement seam: any dataset, any predictor, one set of metrics.

Four seams, each independent of the others:

* ``rows``       datasets, and the ``Row`` contract every predictor sees
* ``protocol``   the ``Predictor`` protocol and the registry that builds one from a spec string
* ``metrics``    accuracy, the per-label ledger, exact McNemar, calibration, the reject curve,
                 latency, and grouped cross-validation
* ``controls``   the A/A floor, option-order permutation, the paraphrase control, and
                 pre-registered pass conditions

``run`` is the one CLI over all four. See ``docs/harness.md``.
"""
