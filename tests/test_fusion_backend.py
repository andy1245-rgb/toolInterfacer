"""The shipping form of the arm that passed a pre-registered test — off, and honest about it.

`docs/fusion-2026-11-10.md` option 3 passed all four conditions against the 3B. This is that arm
as a product backend, reachable through `decide.backends_in_use.choice = "fusion"` only when
`decide.fusion.enabled` is true. Two things are being tested here and they are different:

* the arm's arithmetic — an equal-weight mean, with `web_search` and `web_fetch` owned outright by
  the base;
* the switch — a disabled or unconfigured fusion **refuses** rather than serving the base's answer
  under the fusion's name, which is the rot class this repository audited for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.decide.backends.fusion import FusionBackend  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, make_backend  # noqa: E402

CRITERIA = {"filesys_read": "read", "filesys_glob": "glob", "filesys_grep": "grep",
            "decide_pick": "pick", "decide_extract": "extract", "web_search": "search",
            "web_fetch": "fetch", "reddit_digest": "digest"}


class Stub:
    """A backend with a hand-set answer, so the blend is checkable on paper."""

    def __init__(self, choice, probabilities=None, noul=0.5, blow_up=False):
        self.name = "stub"
        self._choice, self._probs, self._noul, self._boom = choice, probabilities, noul, blow_up
        self.called = 0

    def choice(self, state, instructions, criteria):
        self.called += 1
        if self._boom:
            raise AssertionError("the member was consulted on an op the base owns")
        return ChoiceAnswer(self._choice, dict(self._probs or {}),
                            (self._probs or {}).get(self._choice, 0.0))

    def noul(self, state, question):
        return self._noul


def _fusion(member, base, **cfg):
    config = {"enabled": True, "member": "stub", "base": "stub", "weights": 0.5,
              "protect": ["web_search", "web_fetch"]}
    config.update(cfg)
    return FusionBackend(config=config, member=member, base=base)


# ------------------------------------------------------------------ the switch

def test_the_default_configuration_is_off_and_refuses_loudly():
    with pytest.raises(NotImplementedError) as ctx:
        make_backend("fusion")
    message = str(ctx.value)
    assert "decide.fusion.enabled = true" in message
    assert 'decide.backends_in_use.choice = "fusion"' in message
    assert "refuses rather than falling back" in message


def test_an_explicitly_disabled_block_refuses_too():
    with pytest.raises(NotImplementedError):
        FusionBackend(config={"enabled": False, "member": "classical", "base": "d1_systemone"})


def test_the_backend_is_registered_so_the_config_name_resolves():
    from ti.decide import primitives

    primitives._ensure_backends_loaded()  # noqa: SLF001 - registration is the contract
    assert "fusion" in primitives._REGISTRY  # noqa: SLF001


# ------------------------------------------------------------------ the arithmetic

def test_the_mean_is_equal_weight_and_the_weight_is_on_the_base():
    member = Stub("filesys_read", {"filesys_read": 0.8, "filesys_glob": 0.2})
    base = Stub("filesys_glob", {"filesys_glob": 0.8, "filesys_read": 0.2})
    answer = _fusion(member, base).choice("anything", "", CRITERIA)
    assert answer.probabilities["filesys_read"] == pytest.approx(0.5)
    assert answer.probabilities["filesys_glob"] == pytest.approx(0.5)
    assert answer.choice in {"filesys_read", "filesys_glob"}


def test_the_weight_moves_the_mean_towards_the_base():
    member = Stub("filesys_read", {"filesys_read": 1.0, "filesys_glob": 0.0})
    base = Stub("filesys_glob", {"filesys_glob": 1.0, "filesys_read": 0.0})
    mostly_base = _fusion(member, base, weights=0.9).choice("anything", "", CRITERIA)
    assert mostly_base.choice == "filesys_glob"
    assert mostly_base.probabilities["filesys_glob"] == pytest.approx(0.9)


def test_the_base_owns_web_search_and_web_fetch():
    for owned in ("web_search", "web_fetch"):
        member = Stub(owned, {owned: 0.99, "filesys_read": 0.01}, blow_up=True)
        base = Stub(owned, {owned: 0.6, "filesys_read": 0.4})
        answer = _fusion(member, base).choice("anything", "", CRITERIA)
        assert answer.choice == owned
        assert member.called == 0, "the member must not even be asked on an owned op"


def test_the_member_still_votes_on_the_other_six():
    member = Stub("filesys_read", {"filesys_read": 0.9, "filesys_glob": 0.1})
    base = Stub("filesys_glob", {"filesys_glob": 0.55, "filesys_read": 0.45})
    answer = _fusion(member, base).choice("anything", "", CRITERIA)
    assert answer.choice == "filesys_read", "the member's vote must be able to win elsewhere"
    assert answer.confidence == pytest.approx(answer.probabilities["filesys_read"])


def test_a_member_without_probabilities_leaves_the_base_alone():
    member = Stub("filesys_read", None)
    base = Stub("filesys_glob", {"filesys_glob": 0.9, "filesys_read": 0.1})
    answer = _fusion(member, base).choice("anything", "", CRITERIA)
    assert answer.choice == "filesys_glob"


def test_the_distribution_covers_only_the_offered_options():
    member = Stub("filesys_read", {"filesys_read": 0.7, "filesys_glob": 0.2, "web_search": 0.1})
    base = Stub("filesys_read", {"filesys_read": 0.6, "filesys_glob": 0.3, "web_search": 0.1})
    answer = _fusion(member, base).choice("anything", "", {"filesys_read": "x", "filesys_glob": "y"})
    assert set(answer.probabilities) == {"filesys_read", "filesys_glob"}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_an_empty_offer_answers_nothing_rather_than_guessing():
    answer = _fusion(Stub("filesys_read", {"filesys_read": 1.0}),
                     Stub("filesys_read", {"filesys_read": 1.0})).choice("x", "", {})
    assert answer.choice is None


def test_noul_is_the_bases_judgement():
    assert _fusion(Stub("filesys_read", {"filesys_read": 1.0}),
                   Stub("filesys_read", {"filesys_read": 1.0}, noul=0.83)).noul("x", "q") == 0.83


def test_an_out_of_range_weight_is_refused():
    with pytest.raises(ValueError, match="must be in"):
        _fusion(Stub("filesys_read", {"filesys_read": 1.0}),
                Stub("filesys_read", {"filesys_read": 1.0}), weights=1.5)
