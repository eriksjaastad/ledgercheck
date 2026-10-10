"""The judge gate: rubric drift guard, MockJudge scoring, pass rules, live judge and CLI exits."""

import copy
import json
import re
import socket
from decimal import Decimal

import pytest

from ledgercheck.agents.llm_client import LiveLLMDisabled
from ledgercheck.agents.routing import MockRouter, TaskKind
from ledgercheck.cli import main as cli_main
from ledgercheck.eval import judge
from ledgercheck.eval.judge import (
    CRITERIA,
    RUBRIC,
    SCORE_MAX,
    SCORE_MIN,
    SUITE_MIN_PASS_RATE,
    JudgeError,
    LiveJudge,
    MockJudge,
    Verdict,
    judge_prompt,
    parse_verdict,
    pipeline_output,
    run_suite,
)
from ledgercheck.golden import load_golden_cases

CASES = load_golden_cases()
BY_ID = {c.case_id: c for c in CASES}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_docstring_rubric_matches_constants() -> None:
    doc = judge.__doc__
    assert doc is not None
    assert re.findall(r"^``(\w+)`` \(threshold (\d+)\)$", doc, re.MULTILINE) == [
        (c.name, str(c.threshold)) for c in RUBRIC
    ]
    assert f"Scores are integers from {SCORE_MIN} to {SCORE_MAX}." in doc
    assert re.search(r"minimum\s+pass rate (\S+) ", doc).group(1) == str(SUITE_MIN_PASS_RATE)
    assert re.findall(r"^    (\d)  ", doc, re.MULTILINE) == [
        str(judge.EXIT_PASS), str(judge.EXIT_FAIL), str(judge.EXIT_ERROR)
    ]
    assert all(SCORE_MIN <= c.threshold <= SCORE_MAX for c in RUBRIC)


def test_full_golden_suite_scores_perfect() -> None:
    report = run_suite(MockJudge())
    assert report.passed and report.passed_count == len(CASES)
    for c in report.cases:
        assert dict(c.verdict.scores) == dict.fromkeys(CRITERIA, SCORE_MAX), c.case_id
        assert c.verdict.notes == (), c.case_id


def _hit(rule_id, severity):
    return {"rule_id": rule_id, "severity": severity, "field": None, "expected": None,
            "observed": None}


def _wrong_total(doc):
    doc["extraction"]["total"] = "999.99"


def _hallucinated_hit(doc):
    doc["policy"]["hits"].append(_hit("POL-PO-REQUIRED", "warning"))
    doc["approval"]["hits"].append(_hit("POL-PO-REQUIRED", "warning"))


def _invented_po(doc):
    doc["extraction"]["po_number"] = "PO-INVENTED"


def _invented_sku(doc):
    doc["extraction"]["line_items"][0]["sku"] = "SKU-INVENTED"


def _bad_outcome(doc):
    doc["approval"]["outcome"] = "maybe"


def _approve_with_error(doc):
    doc["approval"]["hits"].append(_hit("APR-SUBTOTAL", "error"))


@pytest.mark.parametrize("case_id, corrupt, scores", [
    ("clean__northwind_baseline", _wrong_total, (4, 5, 5)),
    ("clean__northwind_baseline", _hallucinated_hit, (3, 3, 5)),
    ("missing_po__freight_no_po", _invented_po, (4, 4, 5)),
    ("blank_field__empty_vendor_id_unknown_name", _invented_sku, (4, 4, 5)),
    ("clean__northwind_baseline", _bad_outcome, (3, 5, 3)),
    ("clean__northwind_baseline", _approve_with_error, (3, 5, 3)),
    ("clean__northwind_baseline", lambda doc: doc.pop("policy"), (3, 5, 2)),
])
def test_corrupted_output_drops_the_right_criterion(case_id, corrupt, scores) -> None:
    case = BY_ID[case_id]
    doc = pipeline_output(case)
    assert MockJudge().score(case, doc).scores["accuracy"] == SCORE_MAX
    corrupt(doc)
    verdict = MockJudge().score(case, doc)
    assert tuple(verdict.scores.values()) == scores, verdict.notes
    assert not run_suite(MockJudge(), [case], produce=lambda c: doc).passed


def _extra_retrieved(doc):
    doc["policy"]["retrieved"].append("vendor:V-INVENTED")


def _extra_line_item(doc):
    item = dict(doc["extraction"]["line_items"][0], description="Invented widget")
    doc["extraction"]["line_items"].append(item)


def _hits_out_of_order(doc):
    doc["approval"]["hits"].reverse()


@pytest.mark.parametrize("case_id, corrupt, scores, note", [
    ("clean__northwind_baseline", _extra_retrieved, (4, 4, 5),
     "hallucination: retrieved 'vendor:V-INVENTED' not expected"),
    ("clean__northwind_baseline", _extra_line_item, (4, 4, 5),
     "hallucination: extra line item 'Invented widget'"),
    ("missing_po__freight_no_po", _hits_out_of_order, (4, 5, 4),
     "formatting: approval hits do not end with the policy hits"),
])
def test_each_mock_rule_records_its_note(case_id, corrupt, scores, note) -> None:
    case = BY_ID[case_id]
    doc = pipeline_output(case)
    corrupt(doc)
    verdict = MockJudge().score(case, doc)
    assert tuple(verdict.scores.values()) == scores, verdict.notes
    assert note in verdict.notes
    assert not any("unreadable" in n for n in verdict.notes)


@pytest.mark.parametrize("section, key, value, formatting, hit_check_fails", [
    ("extraction", "total", "abc", 4, False),
    ("policy", "retrieved", "abc", 3, True),
    ("approval", "outcome", "abc", 3, True),
])
def test_only_unreadable_policy_or_approval_fails_hit_order(
        section, key, value, formatting, hit_check_fails) -> None:
    case = BY_ID["clean__northwind_baseline"]
    doc = pipeline_output(case)
    doc[section][key] = value
    verdict = MockJudge().score(case, doc)
    assert verdict.scores["formatting"] == formatting, verdict.notes
    note = "formatting: approval/policy hit consistency: a section is unreadable"
    assert (note in verdict.notes) is hit_check_fails, verdict.notes


@pytest.mark.parametrize("output", ["not json", None, [], {}])
def test_non_object_output_scores_zero_formatting(output) -> None:
    verdict = MockJudge().score(CASES[0], output)
    assert verdict.scores["formatting"] == 0 and verdict.scores["accuracy"] == 0


def test_judging_does_not_mutate_the_output() -> None:
    doc = pipeline_output(CASES[0])
    before = copy.deepcopy(doc)
    MockJudge().score(CASES[0], doc)
    assert doc == before


class FixedJudge:
    name = "fixed"

    def __init__(self, *scores):
        self.scores = iter(scores)

    def score(self, case, output):
        return Verdict(dict(zip(CRITERIA, next(self.scores))))


@pytest.mark.parametrize("scores, thresholds, passed", [
    ((5, 5, 5), None, True),
    ((4, 5, 5), None, False),
    ((5, 5, 4), None, False),
    ((3, 3, 3), {"accuracy": 3, "hallucination": 3, "formatting": 3}, True),
    ((3, 2, 3), {"accuracy": 3, "hallucination": 3, "formatting": 3}, False),
])
def test_case_thresholds_at_the_boundary(scores, thresholds, passed) -> None:
    kwargs = {} if thresholds is None else {"thresholds": thresholds}
    report = run_suite(FixedJudge(scores), CASES[:1], produce=lambda c: None, **kwargs)
    assert report.cases[0].passed is passed and report.passed is passed


@pytest.mark.parametrize("rate, passed", [("0.50", True), ("0.51", False), ("1.00", False)])
def test_suite_pass_rate_at_the_boundary(rate, passed) -> None:
    report = run_suite(FixedJudge((5, 5, 5), (0, 5, 5)), CASES[:2], produce=lambda c: None,
                       min_pass_rate=Decimal(rate))
    assert report.passed_count == 1 and report.passed is passed


@pytest.mark.parametrize("scores", [
    {"accuracy": 5, "hallucination": 5}, {"accuracy": 6, "hallucination": 5, "formatting": 5},
    {"accuracy": -1, "hallucination": 5, "formatting": 5},
    {"accuracy": True, "hallucination": 5, "formatting": 5},
    {"accuracy": 5.0, "hallucination": 5, "formatting": 5},
])
def test_verdict_rejects_scores_outside_the_rubric(scores) -> None:
    with pytest.raises(JudgeError):
        Verdict(scores)


@pytest.mark.parametrize("kwargs", [{"cases": []}, {"thresholds": {"accuracy": 5}},
                                    {"min_pass_rate": Decimal("0")},
                                    {"min_pass_rate": Decimal("1.01")}])
def test_run_suite_rejects_vacuous_rules(kwargs) -> None:
    kwargs.setdefault("cases", CASES[:1])
    with pytest.raises(JudgeError):
        run_suite(MockJudge(), **kwargs)


def test_suite_report_defaults_to_the_shared_read_only_thresholds() -> None:
    cases = run_suite(MockJudge(), cases=CASES[:1]).cases
    report = judge.SuiteReport(judge="mock", cases=cases)
    assert report.thresholds is judge.THRESHOLDS and report.passed


LIVE = {"LEDGERCHECK_LLM_PROVIDER": "openrouter", "LEDGERCHECK_MODEL_LARGE": "vendor/large-y",
        "LEDGERCHECK_SPEND_CAP_USD": "0.10"}
PERFECT = '{"accuracy": 5, "hallucination": 5, "formatting": 5, "notes": []}'


@pytest.mark.parametrize("env", [{}, {"LEDGERCHECK_LLM": "true", "OPENROUTER_API_KEY": "k"},
                                 {"LEDGERCHECK_LLM": "1"}, {"LEDGERCHECK_LLM": "1",
                                                            "OPENROUTER_API_KEY": "  "}])
def test_live_judge_refuses_without_the_gate(env) -> None:
    with pytest.raises(LiveLLMDisabled):
        LiveJudge({**LIVE, **env})


def test_live_judge_scores_the_model_reply() -> None:
    router = MockRouter({}, replies={TaskKind.JUDGE: f"Here you go:\n```json\n{PERFECT}\n```"})
    report = run_suite(LiveJudge(router=router), cases=CASES[:1])
    assert report.passed and report.judge == "mock"
    [(task, model, prompt)] = router.calls
    assert (task, model) == (TaskKind.JUDGE, "mock/large")
    assert prompt == judge_prompt(CASES[0], pipeline_output(CASES[0]))
    assert "GOLDEN: {" in prompt and "OUTPUT: {" in prompt
    assert all(c.question in prompt for c in RUBRIC)


@pytest.mark.parametrize("reply", [
    "", "5/5/5", "[1, 2]", '{"accuracy": 5}', '{"accuracy": 9, "hallucination": 5, "formatting": 5}',
    '{"accuracy": 5, "hallucination": 5, "formatting": 5, "notes": "fine"}', "{not json}",
])
def test_malformed_reply_fails_the_case_without_crashing(reply) -> None:
    verdict = parse_verdict(reply)
    assert dict(verdict.scores) == dict.fromkeys(CRITERIA, 0)
    assert verdict.notes[0].startswith("judge reply unreadable: ")
    report = run_suite(LiveJudge(router=MockRouter({}, replies={TaskKind.JUDGE: reply})),
                       cases=CASES[:2])
    assert report.passed_count == 0 and not report.passed


def test_parse_verdict_keeps_notes() -> None:
    reply = '{"accuracy": 4, "hallucination": 5, "formatting": 5, "notes": ["total off"]}'
    assert parse_verdict(reply) == Verdict({"accuracy": 4, "hallucination": 5, "formatting": 5},
                                           ("total off",))


def test_cli_passes_and_writes_json(tmp_path, capsys) -> None:
    out = tmp_path / "report.json"
    assert cli_main(["judge", "--json", str(out)]) == judge.EXIT_PASS
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] and report["passed_cases"] == report["total_cases"] == len(CASES)
    assert report["thresholds"] == {c.name: c.threshold for c in RUBRIC}
    printed = capsys.readouterr().out
    assert "clean__northwind_baseline" in printed and printed.rstrip().endswith("PASS")


def test_cli_fails_on_a_corrupted_case(monkeypatch, capsys) -> None:
    def produce(case):
        doc = pipeline_output(case)
        if case.case_id == "clean__northwind_baseline":
            _wrong_total(doc)
        return doc

    monkeypatch.setattr(judge, "pipeline_output", produce)
    assert judge.main([]) == judge.EXIT_FAIL
    printed = capsys.readouterr().out
    assert "accuracy: extraction differs from golden" in printed
    assert printed.rstrip().endswith("FAIL")


def test_cli_live_flag_exits_2_without_the_gate(monkeypatch, capsys) -> None:
    assert cli_main(["judge", "--live"]) == judge.EXIT_ERROR
    err = capsys.readouterr().err
    assert "provider openrouter" in err and "LEDGERCHECK_LLM" in err
    for name, value in LIVE.items():
        monkeypatch.setenv(name, value)
    assert judge.main(["--live"]) == judge.EXIT_ERROR
    assert "LEDGERCHECK_LLM" in capsys.readouterr().err
    monkeypatch.setenv("LEDGERCHECK_LLM", "1")
    assert judge.main(["--live"]) == judge.EXIT_ERROR
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY is not set" in err and ".env" in err


def test_cli_case_option_judges_only_those_cases(capsys) -> None:
    first, last = CASES[0].case_id, CASES[-1].case_id
    assert cli_main(["judge", "--case", last, "--case", first]) == judge.EXIT_PASS
    printed = capsys.readouterr().out
    assert "2/2 cases passed" in printed
    assert printed.index(first) < printed.index(last)  # suite order


def test_cli_unknown_case_exits_2(capsys) -> None:
    assert cli_main(["judge", "--case", "no_such_case"]) == judge.EXIT_ERROR
    assert "unknown golden case id(s): no_such_case" in capsys.readouterr().err


def test_cli_unwritable_json_exits_2(tmp_path, capsys) -> None:
    assert judge.main(["--json", str(tmp_path / "missing" / "r.json")]) == judge.EXIT_ERROR
    assert "cannot write" in capsys.readouterr().err


def test_help_documents_the_rubric(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["judge", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for c in RUBRIC:
        assert f"``{c.name}`` (threshold {c.threshold})" in text
    assert "--live" in text and "Exit codes" in text and "OPENROUTER_API_KEY" in text
    assert "--case" in text and "ledgercheck connections --help" in text
