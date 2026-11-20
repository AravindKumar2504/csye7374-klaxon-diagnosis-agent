"""The cause rule, copied from Klaxon (rubric version 4). scripts/verify_copied_rubric.py checks it against Klaxon's own."""

from __future__ import annotations

import pytest

from diagnosis_agent import grading

FIX = {"kind": "rollback", "target": "ledger-consumer", "summary": "roll it back"}


def test_the_rubric_is_pinned():
    # Changing any pattern changes this: then it is no longer the rubric verified against Klaxon's.
    assert grading.fingerprint() == "a9a5183786aeabe4f24dab64458ae37837aa80072448cb901b12d8df4e78b0bd"


@pytest.mark.parametrize(
    "rubric, hypothesis, verdict",
    [
        (
            "duplicate-delivery",
            "The ledger consumer posts redelivered webhooks twice: its dedupe is gone.",
            "names the cause",
        ),
        ("duplicate-delivery", "The ledger consumer's dedupe is gone, so it posts some events twice.", "partly"),
        ("duplicate-delivery", "The ledger consumer posts some events twice.", "misses the cause"),  # 1 of 3
        ("duplicate-delivery", "Payments are slow.", "misses the cause"),
        (
            "duplicate-delivery",
            "Redis lost its keys, so the ledger dedupe failed and resent webhooks posted twice.",
            "misses the cause",
        ),
        ("poison-message", "The ledger consumer retries one malformed message forever.", "names the cause"),
        (
            "poison-message",
            "The database was down, so the ledger consumer retried one invalid message.",
            "misses the cause",
        ),
        ("poison-message", "The ledger is behind.", "misses the cause"),
    ],
)
def test_verdicts(rubric, hypothesis, verdict):
    assert grading.compare(rubric, hypothesis, FIX)["verdict"] == verdict


def test_a_denied_wrong_cause_does_not_count():
    hypothesis = "Not Redis losing keys: the ledger consumer's dedupe is gone, so webhooks sent again are posted twice."
    assert grading.compare("duplicate-delivery", hypothesis, FIX)["verdict"] == "names the cause"


@pytest.mark.parametrize(
    "text, said",
    [
        ("The deploy list could not be read: docker did not answer.", True),
        ("recent_deploys failed, so the deploy that brought it is unconfirmed.", True),
        ("I could not check the deploy history.", True),
        ("The ledger consumer did not check for duplicates.", False),
        ("A deploy of the ledger consumer removed its dedupe.", False),
    ],
)
def test_could_not_check(text, said):
    assert grading.names(grading.COULD_NOT_CHECK, text) is said


def test_the_fix_fits_the_remedy_only_on_the_ledger():
    assert grading.compare("duplicate-delivery", "x", FIX)["fix_fits_the_remedy"]
    other = {"kind": "rollback", "target": "payments-api", "summary": "x"}
    assert not grading.compare("duplicate-delivery", "x", other)["fix_fits_the_remedy"]
