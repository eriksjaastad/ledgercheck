"""LLM-as-judge quality gate: score the golden suite against a strict rubric.

Each golden case (``ledgercheck/golden.py``) runs through the offline pipeline.
Its output is turned into a JSON document with the same layout as the case's
``expected`` block::

    {"extraction": <Invoice.from_dict shape>,
     "policy":   {"retrieved": [chunk ids], "hits": [hit, ...]},
     "approval": {"outcome": "<Outcome value>", "hits": [hit, ...]}}

A hit is ``{"rule_id", "severity", "field", "expected", "observed"}``. Hit
messages, reasons, run ids and the reranker are prose or metadata and are left
out. A judge scores that document against the golden expectation.

Rubric
------
Scores are integers from 0 to 5. A criterion passes when its score is at or
above its threshold.

``accuracy`` (threshold 5)
    The output matches the golden expectation. There are five checks, each
    worth one point: the extracted invoice (every field and line item), policy
    ``retrieved``, policy hits, approval outcome and approval hits. An
    unreadable section fails its checks.
``hallucination`` (threshold 5)
    Lose one point (down to 0) per claim the input and golden case do not
    support: an optional invoice or line item field filled where the golden
    value is null, an extra line item, a retrieved chunk id, or a policy or approval hit
    ``(rule_id, severity)`` beyond what the golden case expects. Only sections
    that parse are scored here; an unreadable section is scored by formatting.
``formatting`` (threshold 5)
    The output is well formed and schema valid. There are five checks, each
    worth one point: the top level is an object with exactly the three keys;
    ``extraction`` parses as an ``Invoice``; ``policy`` parses as a
    ``PolicyResult``; ``approval`` parses as an ``ApprovalDecision`` (so an
    ``approve`` with an ERROR hit fails); and the approval hits end with the
    policy hits. If the policy or approval section is missing or fails to
    parse, the last check also fails; an unreadable extraction does not
    affect it. An output that is not an object scores 0.

Per-case threshold: a case passes only when every criterion meets its
threshold. The defaults require a perfect score, because the offline pipeline
is deterministic and the golden case pins its whole output.

Suite rule: the suite passes when the passed cases are at least the minimum
pass rate 1.00 of all cases, i.e. every case must pass. An empty suite is an
error, never a vacuous pass.

Judges
------
``MockJudge`` (the default) is offline and deterministic. It scores by
comparing the document with the golden case, as described above. A correct
pipeline scores 5/5/5 on every case, and an injected wrong output loses points
on the criterion it breaks.

``LiveJudge`` (``--live``) will send the rubric, case and output to a model.
It goes through the live LLM spend gate in ``ledgercheck.agents.llm_client``:
it raises ``LiveLLMDisabled`` unless ``LEDGERCHECK_LLM=1`` and
``OPENROUTER_API_KEY`` is non-blank. Past the gate it asks the model router
(``ledgercheck.agents.routing``, task ``judge``) for its model, shown as
``LiveJudge.model``, then makes no network call and raises
``NotImplementedError``; the real request is not implemented yet.

Running
-------
``ledgercheck judge`` or ``python -m ledgercheck.eval.judge`` runs the whole
golden suite and prints one row per case (score per criterion, pass/FAIL,
then what the judge found) followed by the suite verdict. ``--json PATH`` also
writes the report as JSON. Exit codes:

    0  the suite passed
    1  the suite fell below the pass rule
    2  usage error, the live gate refused, the live judge is not implemented,
       or the golden dataset, a judge's scores or the JSON path were invalid

An unexpected crash prints a traceback and exits 1 (Python's default).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from ledgercheck.agents.approval import PipelineResult
from ledgercheck.agents.llm_client import API_KEY_ENV, ENV_FLAG, LiveLLMDisabled
from ledgercheck.agents.policy import PolicyResult
from ledgercheck.agents.routing import OpenRouterRouter, TaskKind
from ledgercheck.golden import (
    GoldenCase,
    GoldenError,
    _extraction,
    _hits,
    _keys,
    codes,
    load_golden_cases,
    run_case,
)
from ledgercheck.models import ApprovalDecision, to_jsonable

SCORE_MIN, SCORE_MAX = 0, 5


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    threshold: int
    question: str  # what the judge is asked; the live judge's prompt text


RUBRIC = (
    Criterion("accuracy", 5, "Does the output match the golden expectation exactly?"),
    Criterion("hallucination", 5, "Does the output claim anything the input and golden case "
                                  "do not support (fields, line items, chunks, hits)?"),
    Criterion("formatting", 5, "Is the output well formed and schema valid?"),
)
CRITERIA = tuple(c.name for c in RUBRIC)
THRESHOLDS = MappingProxyType({c.name: c.threshold for c in RUBRIC})
SUITE_MIN_PASS_RATE = Decimal("1.00")
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

_SECTIONS = ("extraction", "policy", "approval")
_HIT_FIELDS = ("rule_id", "severity", "field", "expected", "observed")
_PARSE_ERRORS = (GoldenError, KeyError, TypeError, ValueError, ArithmeticError)


class JudgeError(ValueError):
    """A judge returned scores that do not fit the rubric."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """A judge's scores for one case (one per rubric criterion) and its findings."""

    scores: Mapping[str, int]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scores = dict(self.scores)
        if set(scores) != set(CRITERIA):
            raise JudgeError(f"scores must cover exactly {list(CRITERIA)}, got {sorted(scores)}")
        for name, score in scores.items():
            if isinstance(score, bool) or not isinstance(score, int) or not (
                SCORE_MIN <= score <= SCORE_MAX
            ):
                raise JudgeError(f"{name}: score must be an int {SCORE_MIN}..{SCORE_MAX}, "
                                 f"got {score!r}")
        object.__setattr__(self, "scores", MappingProxyType({c: scores[c] for c in CRITERIA}))
        object.__setattr__(self, "notes", tuple(self.notes))


class Judge(Protocol):
    name: str

    def score(self, case: GoldenCase, output: Any) -> Verdict: ...


def output_document(result: PipelineResult) -> dict[str, Any]:
    """The pipeline output as the JSON document a judge scores (see module docstring)."""

    def hits(hs: Sequence[Any]) -> list[dict[str, Any]]:
        return [{k: to_jsonable(getattr(h, k)) for k in _HIT_FIELDS} for h in hs]

    return {
        "extraction": to_jsonable(result.extraction.invoice),
        "policy": {"retrieved": list(result.policy.retrieved), "hits": hits(result.policy.hits)},
        "approval": {"outcome": result.decision.outcome.value, "hits": hits(result.decision.hits)},
    }


def _parse(case: GoldenCase, name: str, raw: Any) -> Any:
    where = f"output.{name}"
    if name == "extraction":
        return _extraction(raw, "output")
    if name == "policy":
        _keys(raw, {"retrieved", "hits"}, where)
        if not isinstance(raw["retrieved"], list):
            raise GoldenError(f"{where}.retrieved: must be a list")
        return PolicyResult(None, _hits(raw["hits"], f"{where}.hits"), raw["retrieved"])
    _keys(raw, {"outcome", "hits"}, where)
    return ApprovalDecision(case.case_id, case.expected_decision.invoice_number,
                            raw["outcome"], _hits(raw["hits"], f"{where}.hits"))


def _sections(case: GoldenCase, output: Any) -> tuple[dict[str, Any], list[str]]:
    """Parsed sections (unreadable ones absent) and the formatting failures."""
    failures = []
    if not isinstance(output, Mapping):  # scored as an empty object, so every check fails
        failures.append(f"output is {type(output).__name__}, not an object")
        output = {}
    elif set(output) != set(_SECTIONS):
        failures.append(f"top-level keys {sorted(output)} != {sorted(_SECTIONS)}")
    parsed = {}
    for name in _SECTIONS:
        if name not in output:
            failures.append(f"{name}: missing")
            continue
        try:
            parsed[name] = _parse(case, name, output[name])
        except _PARSE_ERRORS as exc:
            failures.append(f"{name}: {exc}")
    policy, approval = parsed.get("policy"), parsed.get("approval")
    if policy is None or approval is None:
        failures.append("approval/policy hit consistency: a section is unreadable")
    elif approval.hits[len(approval.hits) - len(policy.hits):] != policy.hits:
        failures.append("approval hits do not end with the policy hits")
    return parsed, failures


def _filled(want: Any, got: Any, where: str) -> list[str]:
    """Optional fields ``got`` fills where the golden ``want`` has null."""
    return [f"{where}.{f.name} = {getattr(got, f.name)!r}, golden is null"
            for f in dataclasses.fields(want)
            if getattr(want, f.name) is None and getattr(got, f.name) is not None]


def _unsupported(case: GoldenCase, parsed: Mapping[str, Any]) -> list[str]:
    claims = []
    if (inv := parsed.get("extraction")) is not None:
        want = case.expected_extraction
        claims += _filled(want, inv, "extraction")
        for i, (w, g) in enumerate(zip(want.line_items, inv.line_items)):
            claims += _filled(w, g, f"extraction.line_items[{i}]")
        claims += [f"extra line item {li.description!r}"
                   for li in inv.line_items[len(want.line_items):]]
    if (pol := parsed.get("policy")) is not None:
        extra = Counter(pol.retrieved) - Counter(case.expected_policy.retrieved)
        claims += [f"retrieved {cid!r} not expected" for cid in extra.elements()]
    for name, expected in (("policy", case.expected_policy.hits),
                           ("approval", case.expected_decision.hits)):
        if (got := parsed.get(name)) is not None:
            extra = Counter(codes(got.hits)) - Counter(codes(expected))
            claims += [f"{name} hit {rule} ({sev.value}) not expected"
                       for rule, sev in extra.elements()]
    return claims


class MockJudge:
    """Deterministic offline judge: scores by comparison with the golden case."""

    name = "mock"

    def score(self, case: GoldenCase, output: Any) -> Verdict:
        parsed, format_failures = _sections(case, output)
        pol, dec = parsed.get("policy"), parsed.get("approval")
        checks = {
            "extraction": parsed.get("extraction") == case.expected_extraction,
            "policy.retrieved": pol is not None
                                and pol.retrieved == case.expected_policy.retrieved,
            "policy.hits": pol is not None and pol.hits == case.expected_policy.hits,
            "approval.outcome": dec is not None
                                and dec.outcome is case.expected_decision.outcome,
            "approval.hits": dec is not None and dec.hits == case.expected_decision.hits,
        }
        claims = _unsupported(case, parsed)
        notes = ([f"accuracy: {k} differs from golden" for k, ok in checks.items() if not ok]
                 + [f"hallucination: {c}" for c in claims]
                 + [f"formatting: {f}" for f in format_failures])
        return Verdict({
            "accuracy": sum(checks.values()),
            "hallucination": max(SCORE_MIN, SCORE_MAX - len(claims)),
            "formatting": SCORE_MAX - len(format_failures),
        }, tuple(notes))


class LiveJudge:
    """LLM judge behind the live spend gate; constructing one is the gate.

    Raises ``LiveLLMDisabled`` unless the flag and key are set (``env`` as in
    ``LLMClient``). ``model`` is the router's pick for the ``judge`` task.
    ``score`` makes no request yet: it raises ``NotImplementedError``.
    """

    name = "openrouter"

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._router = OpenRouterRouter(env)
        self.model = self._router.route(TaskKind.JUDGE).model

    def score(self, case: GoldenCase, output: Any) -> Verdict:
        raise NotImplementedError("live LLM judging is not implemented yet")


@dataclass(frozen=True, slots=True)
class CaseReport:
    case_id: str
    verdict: Verdict
    passed: bool


@dataclass(frozen=True, slots=True)
class SuiteReport:
    judge: str
    cases: tuple[CaseReport, ...]
    # default_factory: a mappingproxy is unhashable before Python 3.12, and
    # dataclasses rejects unhashable defaults. Same shared object either way.
    thresholds: Mapping[str, int] = field(default_factory=lambda: THRESHOLDS)
    min_pass_rate: Decimal = SUITE_MIN_PASS_RATE
    passed_count: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.cases:
            raise JudgeError("no cases were judged")
        if set(self.thresholds) != set(CRITERIA):
            raise JudgeError(f"thresholds must cover exactly {list(CRITERIA)}")
        if not Decimal(0) < self.min_pass_rate <= Decimal(1):
            raise JudgeError(f"min_pass_rate must be in (0, 1], got {self.min_pass_rate}")
        object.__setattr__(self, "passed_count", sum(c.passed for c in self.cases))

    @property
    def passed(self) -> bool:
        return Decimal(self.passed_count) >= self.min_pass_rate * len(self.cases)

    def to_json(self) -> dict[str, Any]:
        return {
            "judge": self.judge, "passed": self.passed, "passed_cases": self.passed_count,
            "total_cases": len(self.cases), "min_pass_rate": str(self.min_pass_rate),
            "thresholds": dict(self.thresholds),
            "cases": [{"case_id": c.case_id, "passed": c.passed, "scores": dict(c.verdict.scores),
                       "notes": list(c.verdict.notes)} for c in self.cases],
        }


def case_passes(verdict: Verdict, thresholds: Mapping[str, int] = THRESHOLDS) -> bool:
    """The per-case threshold: every criterion at or above its threshold."""
    return all(verdict.scores[c] >= thresholds[c] for c in CRITERIA)


def pipeline_output(case: GoldenCase) -> dict[str, Any]:
    """Run ``case`` offline and return the document a judge scores."""
    return output_document(run_case(case))


def run_suite(
    judge: Judge,
    cases: Sequence[GoldenCase] | None = None,
    *,
    produce: Callable[[GoldenCase], Any] | None = None,
    thresholds: Mapping[str, int] = THRESHOLDS,
    min_pass_rate: Decimal = SUITE_MIN_PASS_RATE,
) -> SuiteReport:
    """Judge every case (default: the whole golden suite) and apply the pass rules.

    ``produce`` turns a case into the output document (default
    ``pipeline_output``); tests pass one that corrupts it.
    """
    cases = load_golden_cases() if cases is None else cases
    produce = pipeline_output if produce is None else produce
    if set(thresholds) != set(CRITERIA):
        raise JudgeError(f"thresholds must cover exactly {list(CRITERIA)}")
    reports = []
    for case in cases:
        verdict = judge.score(case, produce(case))
        if not isinstance(verdict, Verdict):
            raise JudgeError(f"{case.case_id}: {judge.name} returned {type(verdict).__name__}")
        reports.append(CaseReport(case.case_id, verdict, case_passes(verdict, thresholds)))
    return SuiteReport(judge.name, tuple(reports), MappingProxyType(dict(thresholds)),
                       min_pass_rate)


def format_report(report: SuiteReport) -> str:
    width = max(len(c.case_id) for c in report.cases)
    lines = [f"{'case':<{width}}  " + " ".join(f"{c[:4]:>4}" for c in CRITERIA) + "  result"]
    for c in report.cases:
        scores = " ".join(f"{c.verdict.scores[k]:>4}" for k in CRITERIA)
        lines.append(f"{c.case_id:<{width}}  {scores}  {'pass' if c.passed else 'FAIL'}")
        lines += [f"    {note}" for note in c.verdict.notes]
    lines.append(
        f"suite ({report.judge} judge): {report.passed_count}/{len(report.cases)} cases passed, "
        f"minimum pass rate {report.min_pass_rate}: {'PASS' if report.passed else 'FAIL'}"
    )
    return "\n".join(lines)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the judge options to ``parser`` (a new one by default); --help shows this docstring."""
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ledgercheck.eval.judge")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--json", type=Path, metavar="PATH",
                        help="also write the report as JSON to PATH")
    parser.add_argument("--live", action="store_true",
                        help=f"use the live LLM judge; needs {ENV_FLAG}=1 and {API_KEY_ENV}, "
                             "and is not implemented yet (exit 2)")
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the suite for parsed ``args`` and return the exit code."""
    try:
        report = run_suite(LiveJudge() if args.live else MockJudge())
    except (LiveLLMDisabled, NotImplementedError, GoldenError, JudgeError) as exc:
        print(f"judge: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(format_report(report))
    if args.json is not None:
        try:
            args.json.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"judge: cannot write {args.json}: {exc}", file=sys.stderr)
            return EXIT_ERROR
    return EXIT_PASS if report.passed else EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
