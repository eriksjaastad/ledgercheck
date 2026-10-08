"""Offline checks for the Dockerfile, terraform/ and scripts/deploy.py.

No Docker, Terraform or Azure is needed: these read the files and call the
helper with a fake runner, so nothing is built, validated or applied here.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from ledgercheck.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
TF_DIR = ROOT / "terraform"
SECRET_VARS = ("langfuse_public_key", "langfuse_secret_key", "openrouter_api_key")


def _deploy():
    spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts" / "deploy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tf_files() -> list[Path]:
    return sorted(TF_DIR.glob("*.tf")) + [TF_DIR / "terraform.tfvars.example"]


def _variable_block(name: str) -> str:
    text = (TF_DIR / "variables.tf").read_text()
    match = re.search(r'variable "%s" \{\n(.*?)\n\}' % name, text, re.S)
    assert match, f"variable {name} missing"
    return match.group(1)


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


def _search(pattern: str, text: str, flags: int = 0) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    assert match, pattern
    return match


def test_key_files_exist() -> None:
    for rel in ("Dockerfile", ".dockerignore", "scripts/deploy.py", "terraform/versions.tf",
                "terraform/variables.tf", "terraform/main.tf", "terraform/outputs.tf",
                "terraform/terraform.tfvars.example"):
        assert (ROOT / rel).is_file(), rel


def test_dockerfile_is_multi_stage_non_root_and_runs_a_real_subcommand() -> None:
    text = _dockerfile()
    froms = re.findall(r"^FROM (\S+)", text, re.M)
    assert len(froms) == 2 and froms[0] == froms[1]
    assert re.search(r"^USER (?!root)\S+", text, re.M)
    assert 'ENTRYPOINT ["ledgercheck"]' in text
    cmd = _search(r'^CMD \["([^"]+)"\]', text, re.M).group(1)
    subcommands = next(a for a in build_parser()._actions if a.dest == "command").choices
    assert subcommands is not None and cmd in subcommands
    assert "docker build" in text  # how to build lives in the header


def test_dockerfile_python_meets_requires_python() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    floor = _search(r"^>=(\d+)\.(\d+)$", project["requires-python"]).groups()
    image = _search(r"^FROM python:(\d+)\.(\d+)", _dockerfile(), re.M).groups()
    assert tuple(map(int, image)) >= tuple(map(int, floor))


def test_dockerignore_keeps_secrets_and_venv_out_of_the_context() -> None:
    lines = set((ROOT / ".dockerignore").read_text().split())
    assert {".git", ".venv", ".env", ".env.*", "terraform"} <= lines


def test_terraform_defines_the_aca_stack() -> None:
    text = (TF_DIR / "main.tf").read_text()
    for resource in ("azurerm_container_app_environment", "azurerm_container_app_job",
                     "azurerm_user_assigned_identity", "azurerm_log_analytics_workspace"):
        assert f'resource "{resource}"' in text
    assert 'source  = "hashicorp/azurerm"' in (TF_DIR / "versions.tf").read_text()
    assert "backend" not in (TF_DIR / "versions.tf").read_text().split("terraform {", 1)[1]


@pytest.mark.parametrize("name", SECRET_VARS)
def test_secret_variables_are_sensitive_with_no_literal_default(name: str) -> None:
    block = _variable_block(name)
    assert re.search(r"^\s*sensitive\s*=\s*true$", block, re.M)
    default = re.search(r"^\s*default\s*=\s*(.+)$", block, re.M)
    assert default is None or default.group(1).strip() == "null"


@pytest.mark.parametrize("name", SECRET_VARS)
def test_secret_values_only_reach_the_job_through_the_variable(name: str) -> None:
    for path in _tf_files():
        for line in path.read_text().splitlines():
            if re.match(rf"\s*{name}\s*=", line):
                pytest.fail(f"{path.name} assigns {name}: {line.strip()}")


def test_no_secret_looking_literals_in_deploy_files() -> None:
    pattern = re.compile(r"sk-or-|sk-lf-|pk-lf-|AccountKey=|BEGIN [A-Z ]*PRIVATE KEY")
    for path in [*_tf_files(), ROOT / "Dockerfile", ROOT / "scripts" / "deploy.py"]:
        assert not pattern.search(path.read_text()), path.name


def test_tfvars_are_gitignored_but_the_example_is_not() -> None:
    def ignored(rel: str) -> bool:
        result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, timeout=60)
        return result.returncode == 0

    assert ignored("terraform/terraform.tfvars")
    assert ignored("terraform/prod.tfvars")
    # Terraform also auto-loads these, so they can carry secrets too.
    assert ignored("terraform/prod.auto.tfvars")
    assert ignored("terraform/terraform.tfvars.json")
    assert ignored("terraform/prod.auto.tfvars.json")
    assert not ignored("terraform/terraform.tfvars.example")


def test_every_terraform_file_says_it_is_a_write_only_reference() -> None:
    for path in _tf_files():
        header = path.read_text().split("\n\n", 1)[0]
        assert "WRITE-ONLY reference stack: nothing in this repo applies it." in header, path.name


@pytest.mark.parametrize("command", ["apply", "destroy"])
def test_deploy_refuses_apply_and_runs_nothing(command: str, capsys) -> None:
    calls: list = []
    code = _deploy().main([command], runner=lambda cmd, _cwd: calls.append(cmd) or 0,
                          which=lambda _: "/usr/bin/terraform")
    assert code == 2 and calls == []
    assert "is refused: this stack is a write-only reference" in capsys.readouterr().err


def test_deploy_validate_skips_cleanly_without_terraform(capsys) -> None:
    calls: list = []
    code = _deploy().main(["validate"], runner=lambda cmd, _cwd: calls.append(cmd) or 0,
                          which=lambda _: None)
    assert code == 0 and calls == []
    out = capsys.readouterr().out
    assert "terraform not found" in out and "terraform validate" in out


def test_deploy_validate_runs_only_init_and_validate() -> None:
    deploy = _deploy()
    calls: list = []
    code = deploy.main(["validate"], runner=lambda cmd, cwd: calls.append((tuple(cmd), cwd)) or 0,
                       which=lambda _: "/usr/bin/terraform")
    assert code == 0
    assert [c for c, _ in calls] == list(deploy.VALIDATE_STEPS)
    assert all(cwd == TF_DIR for _, cwd in calls)
    assert not any(word in ("apply", "destroy", "plan") for c, _ in calls for word in c)


def test_deploy_validate_stops_at_the_first_failure() -> None:
    calls: list = []
    code = _deploy().main(["validate"], runner=lambda cmd, _cwd: calls.append(cmd) or 1,
                          which=lambda _: "/usr/bin/terraform")
    assert code == 1 and len(calls) == 1


def test_deploy_run_passes_a_timeout(monkeypatch) -> None:
    deploy = _deploy()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    assert deploy._run(("terraform", "validate"), TF_DIR) == 0
    assert seen["cmd"] == ["terraform", "validate"] and seen["cwd"] == TF_DIR
    assert seen["timeout"] == deploy.STEP_TIMEOUT_SECONDS > 0


def test_deploy_validate_fails_cleanly_on_timeout(monkeypatch, capsys) -> None:
    deploy = _deploy()
    calls: list = []

    def stalled(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(deploy.subprocess, "run", stalled)
    code = deploy.main(["validate"], which=lambda _: "/usr/bin/terraform")
    assert code == deploy.TIMEOUT_EXIT != 0
    assert calls == [list(deploy.VALIDATE_STEPS[0])]  # validate never runs after init stalls
    assert "timed out after 600s" in capsys.readouterr().err


def test_deploy_help_documents_docker_build_and_the_rule(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _deploy().main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "docker build" in out and "write-only reference: nothing in this repo applies" in out


def test_readme_has_no_localhost_or_deploy_runbook() -> None:
    text = (ROOT / "README.md").read_text().lower()
    for banned in ("localhost", "127.0.0.1", "0.0.0.0", "terraform apply", "azurecontainerapps.io"):
        assert banned not in text
