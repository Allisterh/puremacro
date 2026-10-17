"""Tests for puremacro.narrative.scoring.llm timing extraction.

Uses a stub backend (no network calls) to feed canned LLM responses
through score_llm and inspect the resulting NarrativeEvent.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from puremacro.narrative.scoring.llm import score_llm


class _StubBackend:
    """Returns canned response strings in FIFO order. One call per text."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.model = "stub"
        self.max_tokens = 0
        self.temperature = 0.0

    def call(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("stub backend out of canned responses")
        return self._responses.pop(0)


def _one_event_response(**fields) -> str:
    """Build a JSON-array response string with one event dict."""
    base = {
        "magnitude_usd_bn": 50.0,
        "target": "consumption",
        "subtarget": "tax",
        "sign": +1,
        "confidence": 0.9,
        "excerpt": "test excerpt",
    }
    base.update(fields)
    return json.dumps([base])


def _one_text_input(date: str = "2010-01-15") -> list[tuple]:
    return [(pd.Timestamp(date), "press text", "https://example.org/x")]


def test_score_llm_extracts_implementation_date():
    """LLM returns implementation_date='2010-04-01' → profile points there."""
    stub = _StubBackend([_one_event_response(implementation_date="2010-04-01")])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == [(pd.Timestamp("2010-04-01"), 1.0)]


def test_score_llm_extracts_horizon_quarters_when_date_null():
    """implementation_date null + horizon_quarters=2, announced 2010-01-15 (Q1).
    Profile should land on 2010-07-01 (Q3, two quarters out)."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=2,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == [(pd.Timestamp("2010-07-01"), 1.0)]


def test_score_llm_implementation_date_wins_when_both_present():
    """When LLM returns both, implementation_date wins; horizon_quarters ignored."""
    stub = _StubBackend([_one_event_response(
        implementation_date="2010-04-01", horizon_quarters=8,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == [(pd.Timestamp("2010-04-01"), 1.0)]


def test_score_llm_default_empty_profile_when_both_null():
    """Both fields null → profile is empty list; effective_profile falls back."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=None,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == []
    # Fallback semantics: effective_profile uses announcement date.
    assert events[0].effective_profile == [(pd.Timestamp("2010-01-15"), 1.0)]


def test_score_llm_skips_malformed_date_string():
    """Bad implementation_date string → profile empty; event still emitted."""
    stub = _StubBackend([_one_event_response(implementation_date="not a date")])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_skips_out_of_range_horizon():
    """horizon_quarters=20 (out of 0..16) → profile empty; event still emitted."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=20,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_rejects_implementation_predating_announcement():
    """implementation_date < announcement_date (Q comparison) → profile empty."""
    stub = _StubBackend([_one_event_response(implementation_date="2009-01-01")])
    events = score_llm(_one_text_input("2010-06-01"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_legacy_response_without_timing_fields_works():
    """Old-schema response with NO timing keys → profile empty; event ships."""
    # Build a response missing both new fields entirely (not just null).
    legacy = json.dumps([{
        "magnitude_usd_bn": 50.0,
        "target": "consumption",
        "subtarget": "tax",
        "sign": +1,
        "confidence": 0.9,
        "excerpt": "legacy",
    }])
    stub = _StubBackend([legacy])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


# A timing field the model got wrong costs the profile, not the event: the event is kept with
# the empty profile, which effective_profile reads as the announcement date.
_EARLIER_QUARTER_DATES = ["2009-12-31", "2009-01-01", "2005-06-15", "1999-12-31"]
_OUT_OF_RANGE_HORIZONS = [17, 20, 100, -1, -5, "many"]


@pytest.mark.parametrize("impl_date", _EARLIER_QUARTER_DATES)
def test_an_implementation_date_in_an_earlier_quarter_keeps_the_event_with_the_default_profile(impl_date):
    stub = _StubBackend([_one_event_response(implementation_date=impl_date)])
    events = score_llm(_one_text_input("2010-01-15"), backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []
    assert events[0].effective_profile == [(pd.Timestamp("2010-01-15"), 1.0)]


@pytest.mark.parametrize("horizon", _OUT_OF_RANGE_HORIZONS)
def test_an_out_of_range_horizon_keeps_the_event_with_the_default_profile(horizon):
    stub = _StubBackend([_one_event_response(implementation_date=None, horizon_quarters=horizon)])
    events = score_llm(_one_text_input("2010-01-15"), backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []
    assert events[0].effective_profile == [(pd.Timestamp("2010-01-15"), 1.0)]


def test_a_record_date_that_is_not_a_date_raises_before_the_model_is_asked():
    # The record's date is the caller's, not the model's: a bad one raised in 4.8.1 and raises
    # here, instead of counting the record's events as malformed; and no completion is spent on it.
    stub = _StubBackend([_one_event_response(implementation_date="2010-04-01")])
    with pytest.raises(ValueError):
        score_llm([("not-a-date", "press text", "https://example.org/x")], backend=stub, country="USA")
    assert len(stub._responses) == 1


# ---------------------------------------------------------------------------
# Answers of a real qwen3-4b, announced 2020-03-15. Through 4.8.1 a
# same-quarter implementation date gave the profile [(2020-01-01, 1.0)], which
# predates the announcement, so NarrativeEvent raised and score_llm aborted.
# ---------------------------------------------------------------------------
_QWEN3_ANSWER = (
    '[{{"magnitude_usd_bn": 500.0, "target": "investment", "subtarget": "infra", '
    '"sign": {sign}, "confidence": 1.0, "excerpt": "...", '
    '"implementation_date": {date}, "horizon_quarters": {horizon}}}]'
)


def _qwen3_answer(date="null", horizon="null", sign="+1") -> str:
    return _QWEN3_ANSWER.format(date=date, horizon=horizon, sign=sign)


_MARCH_15 = [("2020-03-15", "The government announced a $500 billion package.", "u")]


def test_real_answer_with_same_quarter_date_gives_an_empty_profile():
    stub = _StubBackend([_qwen3_answer(date='"2020-03-15"')])
    events = score_llm(_MARCH_15, backend=stub, kind="fiscal")
    assert len(events) == 1
    assert events[0].sign == 1 and events[0].magnitude == 500.0
    assert events[0].implementation_profile == []
    assert events[0].effective_profile == [(pd.Timestamp("2020-03-15"), 1.0)]


def test_real_answer_with_horizon_zero_gives_an_empty_profile():
    stub = _StubBackend([_qwen3_answer(horizon="0")])
    events = score_llm(_MARCH_15, backend=stub, kind="fiscal")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_same_quarter_date_wins_over_a_horizon():
    stub = _StubBackend([_qwen3_answer(date='"2020-03-15"', horizon="4")])
    events = score_llm(_MARCH_15, backend=stub, kind="fiscal")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_later_quarter_date_keeps_its_quarter_start():
    stub = _StubBackend([_qwen3_answer(date='"2020-04-15"')])
    events = score_llm(_MARCH_15, backend=stub, kind="fiscal")
    assert len(events) == 1
    assert events[0].implementation_profile == [(pd.Timestamp("2020-04-01"), 1.0)]


def test_same_quarter_date_no_longer_aborts_the_corpus():
    # Plain JSON: through 4.8.1 this raised ValueError from NarrativeEvent.
    stub = _StubBackend([_one_event_response(implementation_date="2010-02-01")])
    events = score_llm(_one_text_input("2010-01-15"), backend=stub, country="USA")
    assert len(events) == 1 and events[0].implementation_profile == []


def test_horizon_zero_no_longer_aborts_the_corpus():
    stub = _StubBackend([_one_event_response(implementation_date=None, horizon_quarters=0)])
    events = score_llm(_one_text_input("2010-01-15"), backend=stub, country="USA")
    assert len(events) == 1 and events[0].implementation_profile == []


def test_an_unbuildable_event_is_dropped_not_raised(monkeypatch, capsys):
    import puremacro.narrative.scoring.llm as llm_mod

    real = llm_mod._profile_from_timing_fields

    def far_before_for_the_first(ev_dict, *, announcement):
        if pd.Timestamp(announcement) == pd.Timestamp("2020-03-15"):
            return [(pd.Timestamp("2015-01-01"), 1.0)]      # NarrativeEvent refuses this
        return real(ev_dict, announcement=announcement)

    monkeypatch.setattr(llm_mod, "_profile_from_timing_fields", far_before_for_the_first)
    corpus = [_MARCH_15[0], ("2020-06-10", "A second package was announced.", "v")]
    # Plain "sign": 1, so that this test does not depend on the parser's "+1" handling.
    stub = _StubBackend([_qwen3_answer(sign="1"), _qwen3_answer(date='"2020-10-01"', sign="1")])
    events = score_llm(corpus, backend=stub, kind="fiscal")
    assert [e.date for e in events] == [pd.Timestamp("2020-06-10")]
    assert events[0].implementation_profile == [(pd.Timestamp("2020-10-01"), 1.0)]
    assert "dropped 1 malformed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------
from puremacro.narrative.scoring.llm import _parse_response  # noqa: E402


def test_parse_response_reads_a_plus_signed_number():
    assert _parse_response('[{"sign": +1, "x": +0.5}]') == [{"sign": 1, "x": 0.5}]


def test_parse_response_keeps_a_plus_inside_a_string():
    out = _parse_response('[{"sign": +1, "excerpt": "growth of +3% \\"or +4\\""}]')
    assert out == [{"sign": 1, "excerpt": 'growth of +3% "or +4"'}]


def test_parse_response_reads_a_plus_inside_a_fence_and_prose():
    fenced = '```json\n[{"sign": +1}]\n```'
    assert _parse_response(fenced) == [{"sign": 1}]
    prose = 'Here are the events:\n[{"sign": +1}]\nDone.'
    assert _parse_response(prose) == [{"sign": 1}]


def test_parse_response_accepts_an_events_object():
    assert _parse_response('{"events": [{"sign": +1}]}') == [{"sign": 1}]
    assert _parse_response('{"events": []}') == []


def test_parse_response_still_returns_empty_for_unparseable_answers():
    assert _parse_response("no json here") == []
    assert _parse_response('{"sign": 1}') == []                 # an object without events
    assert _parse_response('{"events": "none"}') == []          # events that are not a list
    assert _parse_response("[{broken") == []


def test_parse_response_falls_back_to_the_outer_brackets():
    assert _parse_response('Events: [{"sign": -1}] (one)') == [{"sign": -1}]


def test_score_llm_accepts_five_tuple_connector_records():
    """5-tuple records from validate_records / connectors unpack cleanly with metadata language."""
    records = [
        (pd.Timestamp("2020-03-15"), "El gobierno anunció un paquete fiscal.",
         "https://example.org/es", {"language": "es", "country": "MEX"}, 50.0)
    ]
    stub = _StubBackend([_one_event_response(implementation_date="2020-03-20")])
    events = score_llm(records, backend=stub, kind="fiscal", country="MEX")
    assert len(events) == 1
    assert events[0].language == "es"
    assert events[0].implementation_profile == []
    assert events[0].effective_profile == [(pd.Timestamp("2020-03-15"), 1.0)]

