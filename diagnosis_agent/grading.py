# Copied from Klaxon (github.com/AravindKumar2504/klaxon) at commit fef450b (fef450b0aae7545fc3f86ba8eb70ccce2dff1676),
# file src/klaxon/grading.py: rubric version 4's rubrics for duplicate-delivery and poison-message, the elements its
# missing-deploys rubric adds, and the matcher (names, with its negation and hedge rules). Every pattern is copied
# unchanged. Changes for this mini-project:
#   - only these rubrics, and no older versions; compare() takes the rubric itself, and reads one text it is given
#   - the missing-evidence case is graded in two parts here (diagnosis_agent.check): the cause by fault 1's rubric
#     on the hypothesis, as Klaxon does, and "what it could not check" (Klaxon's `also` element) on the whole
#     diagnosis, not only on the hypothesis
"""How a diagnosis compares with the cause a fault records: a rule, written before Klaxon's first runs on the faults.

Each cause is cut into elements a correct diagnosis names: where it is (the ledger consumer, say), what is wrong there
(its dedupe is gone), and how that produces the symptom (the provider sends some webhooks again). Each element is a few
regular expressions, matched without regard to case. The rule reads the diagnosis's hypothesis, the field the
diagnosis contract keeps for "the cause, and how it produces the alert", and counts the required elements it names:

    names the cause    all of them
    partly             at least half of them
    misses the cause   fewer than half

Optional elements (the deploy that brought the bad setting) are reported, and do not change the verdict. A known wrong
cause (Redis losing its keys; the database down), given as the cause, makes it "misses the cause" whatever else it
names. The rule also says whether the proposed fix is of a kind, and on a target, that fit the fault's remedy.

A rule reads words, not meaning: "a deploy is unlikely" names a deploy, and the right cause put in other words can be
missed. So every check shows the rule's verdict next to the hypothesis and the recorded cause, for a person to check,
and always labels it as a rule's. In Klaxon, this rule agreed with Aravind's own blind verdicts on the cause on 27 of the
40 runs he labeled: 13 of 13 in its calibration, 11 of 17 in week 5, and 3 of 10 in week 6, where Klaxon's records call
the graders a weak guide on the false alarm and on missing deploys (Klaxon's docs/STATUS.md at fef450b).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

BY = (
    "a rule (Klaxon's rubric version 4, copied from Klaxon at fef450b, written there before its live runs): keyword"
    " patterns per element of the right answer. Not a person and not a model: it reads words, not meaning."
)


@dataclass(frozen=True)
class Element:
    name: str  # what it stands for, in a few words
    patterns: tuple[str, ...]  # regular expressions; one match is enough
    required: bool = True
    unless: tuple[str, ...] = ()  # one match, and the element is not named (it is put elsewhere, or denied)
    scoped: bool = False  # a match a negation denies or rules out does not count (_negated)


@dataclass(frozen=True)
class Rubric:
    elements: tuple[Element, ...]
    fix_kinds: tuple[str, ...]  # the kinds of proposed fix that fit the fault's remedy
    fix_target: str  # a regular expression the fix's target must match
    wrong: tuple[Element, ...] = ()  # known wrong causes; a hypothesis that names one misses the cause


LEDGER = Element("where: the ledger consumer", (r"\bledger\b",))
DEPLOY = (r"deploy", r"roll(ed)?[ -]?out", r"setting", r"config")


def _gap(words: int) -> str:
    """A gap of a few words within one clause (never across a full stop or a semicolon), none of them a negation:
    "PostgreSQL was down" counts, "PostgreSQL was not down", "PostgreSQL wasn't down" and "neither slow nor down" do
    not."""
    return rf"[^\w.;]+((?!(not|never|no|t|neither|nor)\b)\w+[^\w.;]+){{0,{words}}}"


def _more(element: Element, *patterns: str, unless: tuple[str, ...] = ()) -> Element:
    """An earlier version's element, with more patterns, or words that cancel it."""
    return replace(element, patterns=(*element.patterns, *patterns), unless=unless)


# Fault 1 (duplicate-delivery), version 1's elements; version 4 adds to the second.
DEDUPE_V1 = Element(
    "what: its dedupe is gone",
    (r"dedup", r"idempoten", r"already[ _-]processed", r"\bnaive\b", r"LEDGER_HANDLER"),
)
SENT_AGAIN = Element(
    "how: the provider sends some webhooks again",
    (
        r"re-?deliver",
        r"re-?sen[dt]",
        r"sen[dt]s? (it |them |the same \w+ )?again",
        r"retried (webhook|deliver|event)",
        r"(webhook|deliver\w*|event)s? (are |is |get |gets )?retried",
        r"acknowledg",
        r"\back\b",
        r"duplicate (webhook|deliver|event|message)",
        r"at[ -]least[ -]once",
    ),
)
DEPLOY_OF_HANDLER = Element("the deploy that brought it", (*DEPLOY, r"LEDGER_HANDLER", r"handler"), required=False)
# Known limit (version 1): "turned off the ledger consumer's duplicate check" is its dedupe gone.
DEDUPE_GONE = _more(DEDUPE_V1, r"duplicate[- ]?(check|detection|filter\w*|guard|suppression)")
REDIS_LOST = Element(  # known limit (version 1): Redis losing its keys is another cause than the handler's deploy
    "Redis lost the keys",
    (
        rf"redis{_gap(3)}(lost|loses|lose|losing|dropped|drops|evicted|evicts|flushed|flushes|expired|wiped|forgot)\b",
        r"\b(lost|dropped|evicted|flushed|expired|wiped) (\w+ ){0,3}(from|in) redis",
    ),
    scoped=True,
)

# Fault 2 (poison-message), version 1's elements; version 4 adds to the second.
UNREADABLE_V1 = Element(
    "what: one message it cannot read",
    (
        r"poison",
        r"malformed",
        r"unreadable",
        r"(cannot|can't|could not|couldn't|unable to|fails? to|failed to|failing to) (be )?"
        r"(parse|read|book|process|decode|handle)",
        r"invalid",
        r"bad (event|message)",
        r"decimal",
        r"12\.50",
        r"as a string",
        r"string amount",
    ),
)
RETRIES_FOREVER = Element(
    "how: it retries that message forever, so nothing behind it is booked",
    (r"retr(y|ies|ying|ied)", r"forever", r"endless", r"infinite", r"head[ -]of[ -]line", r"over and over"),
)
DEPLOY_OF_RETRY = Element("the deploy that brought it", (*DEPLOY, r"BAD_EVENTS", r"dead[ -]?letter"), required=False)
DATABASE = r"(postgres\w*|database|\bdb\b)"

RUBRICS = {
    "duplicate-delivery": Rubric(
        elements=(LEDGER, DEDUPE_GONE, SENT_AGAIN, DEPLOY_OF_HANDLER),
        fix_kinds=("rollback", "config_change"),
        fix_target=r"ledger",
        wrong=(REDIS_LOST,),
    ),
    "poison-message": Rubric(
        elements=(
            LEDGER,
            # Known limits (version 1): the services' own words for its amount, and "a string" put another way.
            _more(
                UNREADABLE_V1,
                r"not (a |an )?(positive )?(integer|whole number)",
                r"\b(is|was|are|were|as) (a |an )?(decimal |numeric |text |json )?string\b",
            ),
            RETRIES_FOREVER,
            DEPLOY_OF_RETRY,
        ),
        fix_kinds=("rollback", "config_change"),
        fix_target=r"ledger",
        # Known limit (version 1): a database outage that makes the ledger retry is another cause.
        wrong=(
            Element(
                "the database was down",
                (rf"{DATABASE}{_gap(2)}(down|unavailable|unreachable|offline|restarting|crashed)\b",),
                scoped=True,
            ),
        ),
    ),
}

# Klaxon's missing-deploys rubric: fault 1's elements, and this one (its `also`), required to name the cause.
COULD_NOT_CHECK = Element(
    "what it could not check: docker did not answer the deploy list or the health check",
    (
        # Klaxon could not check something docker shows: not "the handler did not check for duplicates".
        r"\b(could not|couldn't|cannot|can't|unable to|did not|didn't|failed to)\s+(\w+\s+){0,2}"
        r"(check|read|see|confirm|verify|list|reach|query|inspect|retrieve|access|get)\b[^.;]{0,60}?"
        r"\b(deploy\w*|docker|container\w*|health\w*|tools?|recent_deploys|service_health|setting\w*|history)\b",
        r"\b(deploy\w*|docker|container\w*|health\w*|recent_deploys|service_health)\b[^.;]{0,60}?"
        r"\b(could not|couldn't|cannot|can't) be (read|checked|seen|confirmed|verified|listed|reached)\b",
        r"\bdocker[^\w.;]+(\w+[^\w.;]+){0,4}(did not answer|didn't answer|not answer|no answer|timed out"
        r"|times out|time ?out|unavailable|unreachable|not responding|unresponsive|down\b|did not respond"
        r"|didn't respond|never responded|never answered|no response)",
        r"\b(recent_deploys|service_health|deploy (list|log|history)|deploy tool|health (check|tool))"
        r"[^\w.;]+(\w+[^\w.;]+){0,4}(failed|fails|timed out|times out|errored|error|unavailable|did not"
        r" answer|didn't answer|not available|returned nothing|did not respond|didn't respond|never answered)",
        r"\b(deploy\w*|docker|container\w*|health\w*|setting\w*|handler|recent_deploys|service_health)\b"
        r"[^.;]{0,80}?\b(unverified|unconfirmed|not (been |yet )?(confirmed|verified|checked))\b",
        r"\b(unverified|unconfirmed|not (been |yet )?(confirmed|verified|checked))\b[^.;]{0,40}?"
        r"\b(deploy\w*|docker|container\w*|health\w*|setting\w*|handler)\b",
    ),
    # Not scoped: its own words are negations ("did not answer"), and the outage's negations come with them.
)


def fingerprint() -> str:
    """The SHA-256 of the rubrics and the could-not-check element: every pattern and fitting fix. A test pins it, so
    no pattern can change without a test failing."""
    shape = {name: asdict(rubric) for name, rubric in sorted(RUBRICS.items())} | {"also": asdict(COULD_NOT_CHECK)}
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()


# What denies a match, or offers it only as a maybe, read within its own part of a sentence. A clause ends at a full
# stop, a semicolon, a colon or an opening bracket. Inside it, a comma ends a segment, except through a list of the
# same kind of thing ("not hung, stalled or crashed"): a negation reaches no further. In a segment, a turn of the
# sentence ("but", "and", "while" ...) after a negation ends its reach.
CUE = re.compile(
    r"\b(not|no|never|neither|nor|without|cannot|nothing|none|unlikely|excluded|rather than|instead of"
    r"|rule[sd]? out|ruling out|if)\b|n't\b",
    re.IGNORECASE,
)
TURN = re.compile(
    r"\b(but|however|although|though|whereas|while|when|after|until|as|yet|so|because|since|instead|and|then)\b",
    re.IGNORECASE,
)
# After a match, in its segment: ruled out ("is unlikely", "and found none"), or only supposed ("would show").
RULED_OUT = re.compile(
    r"\b(ruled out|excluded|unlikely|not the cause|not at fault|not involved|not to blame|found none|found nothing"
    r"|found no sign|would)\b",
    re.IGNORECASE,
)
OBJECT_DENIED = re.compile(r"\s+(nothing|none|no)\b", re.IGNORECASE)  # right after a match: "Redis lost nothing"
RULING_OUT_LEAD = re.compile(r"\s*(ruled out|excluded|unlikely|not likely|not the cause)\s*", re.IGNORECASE)
CLAUSE = ".;!?:("
SEGMENT_ENDS = ".;!?:(),"
# A wrong cause offered as a maybe, beside the right one: "the database could be down too", "Still open: whether...",
# ", or PostgreSQL was briefly unavailable". "Could not" is no hedge; "open" is one only before a colon ("Still open:").
HEDGE = re.compile(
    r"\b(could(?!\s*(not|n't))|might|may|possibly|perhaps|maybe|whether|either|unclear|uncertain)\b", re.IGNORECASE
)
HEDGE_LEAD = re.compile(r"\b(open|unknown|unclear|possible|possibly|maybe|perhaps|unverified)\b", re.IGNORECASE)
MAYBE_AFTER = re.compile(
    r"\b(is possible|was possible|possibly|a possibility|cannot be ruled out|can't be ruled out|not ruled out)\b",
    re.IGNORECASE,
)


def _clause(text: str, start: int) -> tuple[int, str]:
    """Where the clause of a match at start begins (after a full stop, a semicolon, a colon or an opening bracket), and
    the words before its colon when a colon opens it ("Ruled out: ...", "Still open: ...")."""
    opened = max(text.rfind(mark, 0, start) for mark in CLAUSE) + 1
    if opened and text[opened - 1] == ":":
        return opened, text[max(text.rfind(mark, 0, opened - 1) for mark in CLAUSE) + 1 : opened - 1]
    return opened, ""


def _segment(text: str, start: int, opened: int, patterns: tuple[str, ...]) -> int:
    """Where the comma segment of a match at start begins: after the last comma of its clause; or, when the segments
    before it each end in another match of the same patterns, where that list begins ("not hung, stalled or crashed")."""
    begins = max(text.rfind(",", opened, start) + 1, opened)
    while begins > opened:
        before = max(text.rfind(",", opened, begins - 1) + 1, opened)
        piece = text[before : begins - 1]
        if not any(re.search(rf"(?:{pattern})\W*$", piece, re.IGNORECASE) for pattern in patterns):
            break
        begins = before
    return begins


def _segment_end(text: str, end: int) -> int:
    found = [i for i in (text.find(mark, end) for mark in SEGMENT_ENDS) if i != -1]
    return min(found, default=len(text))


def _negated(text: str, start: int, end: int, patterns: tuple[str, ...] = ()) -> bool:
    """Whether the match at text[start:end] (of one of patterns) is denied, ruled out or only supposed: a negation
    earlier in its segment, with no turn of the sentence after it ("no sign of a poison message", "rather than Redis
    losing keys", "if it were stuck"), through a list of the same kind ("not hung, stalled or crashed"); a ruling-out
    before a colon ("Ruled out: Redis losing its keys"); or, later in its segment, words that rule it out or only
    suppose it ("is unlikely", "and found none", "would show"), or a denied object right after it ("Redis lost
    nothing")."""
    opened, lead = _clause(text, start)
    if RULING_OUT_LEAD.fullmatch(lead):
        return True
    begins = _segment(text, start, opened, patterns)
    cues = list(CUE.finditer(text, begins, start))
    if cues and not TURN.search(text, cues[-1].end(), start):
        return True
    if OBJECT_DENIED.match(text, end):
        return True
    return RULED_OUT.search(text, end, _segment_end(text, end)) is not None


def _hedged(text: str, start: int, end: int, patterns: tuple[str, ...] = ()) -> bool:
    """Whether the match at text[start:end] is offered as a maybe: a hedge in it ("could be down"); earlier in its
    segment with no turn of the sentence after it ("might", "whether"), or as an alternative (", or PostgreSQL was
    down"); one before its colon ("Still open: ..."); or later in its segment ("is possible", "cannot be ruled out")."""
    opened, lead = _clause(text, start)
    if HEDGE_LEAD.search(lead) or HEDGE.search(lead) or HEDGE.search(text, start, end):
        return True
    begins = _segment(text, start, opened, patterns)
    if text[begins:start].strip().lower() in ("or", "either", "or else"):  # an alternative: ", or X"
        return True
    hedges = list(HEDGE.finditer(text, begins, start))
    if hedges and not TURN.search(text, hedges[-1].end(), start):
        return True
    return MAYBE_AFTER.search(text, end, _segment_end(text, end)) is not None


def _stance(element: Element, text: str) -> str | None:
    """How text gives a known wrong cause: "asserted", "offered" (only as a maybe), or None (not at all, or only
    denied)."""
    offered = False
    for pattern in element.patterns:
        for found in re.finditer(pattern, text, re.IGNORECASE):
            if _negated(text, found.start(), found.end(), element.patterns):
                continue
            if not _hedged(text, found.start(), found.end(), element.patterns):
                return "asserted"
            offered = True
    return "offered" if offered else None


def _found(patterns: tuple[str, ...], text: str, scoped: bool = False) -> bool:
    if not scoped:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    return any(  # only a match no negation denies
        not _negated(text, found.start(), found.end(), patterns)
        for pattern in patterns
        for found in re.finditer(pattern, text, re.IGNORECASE)
    )


def names(element: Element, text: str) -> bool:
    """Whether text names the element: one of its patterns (for a scoped element, one no negation denies), and none of
    the words that cancel it (words it asserts, never ones it denies)."""
    return _found(element.patterns, text, element.scoped) and not _found(element.unless, text, scoped=True)


def compare(rubric_name: str, hypothesis: str, fix: Mapping) -> dict:
    """The rule's comparison of a hypothesis with a fault's recorded cause, and of the proposed fix with its remedy."""
    rubric = RUBRICS[rubric_name]
    found = [(element, names(element, hypothesis)) for element in rubric.elements]
    required = [named for element, named in found if element.required]
    hits = sum(required)
    verdict = (
        "names the cause" if hits == len(required) else "partly" if 2 * hits >= len(required) else "misses the cause"
    )
    stances = {element.name: _stance(element, hypothesis) for element in rubric.wrong}
    wrong = [name for name, said in stances.items() if said == "asserted"]
    offered = [name for name, said in stances.items() if said == "offered"]
    if wrong:  # a hypothesis that gives a known wrong cause as the cause misses it, whatever else it names
        verdict = "misses the cause"
    elif offered and verdict == "names the cause":  # the right cause beside a wrong one, unchosen
        verdict = "partly"
    fits = (
        fix.get("kind") in rubric.fix_kinds
        and re.search(rubric.fix_target, str(fix.get("target", "")), re.IGNORECASE) is not None
    )
    return {
        "by": BY,
        "rubric": rubric_name,
        "fingerprint": fingerprint(),
        "verdict": verdict,
        "elements": [
            {
                "element": element.name,
                "required": element.required,
                "named": named,
                "cancelled": _found(element.patterns, hypothesis) and not named,
            }
            for element, named in found
        ],
        "wrong_causes": wrong,
        "wrong_causes_offered": offered,
        "proposed_fix": {"kind": fix.get("kind"), "target": fix.get("target")},
        "fix_fits_the_remedy": fits,
        "fitting_fixes": {"kinds": list(rubric.fix_kinds), "target_matches": rubric.fix_target},
    }
