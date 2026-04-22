# Ralph Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an outer restart loop ("ralph loop") to LBM so that when an agent exhausts repair attempts, it wipes the PR and starts the agent fresh — up to a configurable cap.

**Architecture:** The ralph loop lives entirely in `agent_ops.py` inside `cmd_dispatch_repair`. When repairs are exhausted, it checks a ralph counter (tracked via issue comments), and if under cap: summarizes the failure via LLM, closes+deletes the PR/branch, posts a restart marker on the issue, and re-dispatches the agent workflow. The config is modeled with stdlib dataclasses for type safety.

**Tech Stack:** Python 3.10+ stdlib (dataclasses, tomllib), gh CLI, GitHub Actions workflow_dispatch

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/models.py` | Create | Dataclass definitions: `AgentConfig`, `ChecksConfig`, `LLMConfig`, `LBMConfig` |
| `scripts/agent_ops.py` | Modify | Refactor to use dataclasses, extract helpers, add ralph loop logic |
| `scripts/config_parser.py` | Modify | No changes — workflows use `load_config()` for raw TOML dict; that stays |
| `templates/lbm.toml.j2` | Modify | Add `max_ralph_loops = 0` under `[checks]` |
| `test/test_models.py` | Create | Tests for dataclass `from_dict` methods |
| `test/test_agent_ops.py` | Modify | Add tests for new helpers and ralph loop flow |
| `test/conftest.py` | Modify | Update fixtures to use `AgentConfig` dataclass |

---

### Task 1: Dataclass config models

**Files:**
- Create: `scripts/models.py`
- Create: `test/test_models.py`

- [ ] **Step 1: Write the failing tests for `AgentConfig.from_dict`**

```python
# test/test_models.py
"""Tests for scripts/models.py — config dataclasses."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from models import AgentConfig, ChecksConfig, LBMConfig, LLMConfig


class TestAgentConfig:
    def test_from_dict(self):
        d = {
            "label": "agent:claude",
            "harness": "claude",
            "model_id": "claude-opus-4-6",
            "model_label": "opus-4-6",
            "branch_prefix": "claude-opus-4-6/",
            "name": "Agent A",
            "mention": "@claude",
        }
        agent = AgentConfig.from_dict(d)
        assert agent.label == "agent:claude"
        assert agent.harness == "claude"
        assert agent.name == "Agent A"
        assert agent.mention == "@claude"


class TestChecksConfig:
    def test_from_dict_with_defaults(self):
        d = {"required": ["CI"], "repair_from": ["CI"]}
        checks = ChecksConfig.from_dict(d)
        assert checks.required == ["CI"]
        assert checks.max_repair_attempts == 10
        assert checks.max_ralph_loops == 0

    def test_from_dict_with_overrides(self):
        d = {
            "required": ["CI"],
            "repair_from": ["CI"],
            "max_repair_attempts": 5,
            "max_ralph_loops": 3,
        }
        checks = ChecksConfig.from_dict(d)
        assert checks.max_repair_attempts == 5
        assert checks.max_ralph_loops == 3

    def test_from_dict_empty(self):
        checks = ChecksConfig.from_dict({})
        assert checks.required == []
        assert checks.repair_from == []
        assert checks.max_repair_attempts == 10
        assert checks.max_ralph_loops == 0


class TestLLMConfig:
    def test_from_dict_defaults(self):
        llm = LLMConfig.from_dict({})
        assert llm.provider == "anthropic"
        assert llm.summary_model == "claude-sonnet-4-6"

    def test_from_dict_portkey(self):
        llm = LLMConfig.from_dict({"provider": "portkey", "summary_model": "gpt-4o"})
        assert llm.provider == "portkey"
        assert llm.summary_model == "gpt-4o"


class TestLBMConfig:
    def test_from_parsed_toml(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        raw = tomllib.loads("""
[lbm]
version = 1
ready_label = "ready-for-dev"

[build]
runtime = "node"

[checks]
required = ["CI"]
repair_from = ["CI"]
max_repair_attempts = 10
max_ralph_loops = 2

[llm]
provider = "anthropic"
summary_model = "claude-sonnet-4-6"

[harnesses.claude]
mention = "@claude"

[harnesses.codex]
mention = "@codex"

[[agents]]
harness = "claude"
model_id = "claude-opus-4-6"
model_label = "opus-4-6"

[[agents]]
harness = "codex"
model_id = "gpt-5.3"
model_label = "gpt-5.3"
""")
        config = LBMConfig.from_parsed_toml(raw)
        assert len(config.agents) == 2
        assert config.agents[0].name == "Agent A"
        assert config.agents[1].name == "Agent B"
        assert config.checks.max_ralph_loops == 2
        assert config.llm.provider == "anthropic"

    def test_from_parsed_toml_minimal(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        raw = tomllib.loads("""
[harnesses.claude]
mention = "@claude"

[[agents]]
harness = "claude"
model_id = "x"
model_label = "y"
""")
        config = LBMConfig.from_parsed_toml(raw)
        assert len(config.agents) == 1
        assert config.checks.max_ralph_loops == 0
        assert config.llm.provider == "anthropic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement `scripts/models.py`**

```python
# scripts/models.py
"""Typed configuration models for LBM.

Stdlib dataclasses — no external dependencies. Workflow scripts run as bare
python3 on GH Actions runners, so pydantic is not an option.
"""

from __future__ import annotations

from dataclasses import dataclass, field

AGENT_NAME_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class AgentConfig:
    label: str
    harness: str
    model_id: str
    model_label: str
    branch_prefix: str
    name: str
    mention: str

    @classmethod
    def from_dict(cls, d: dict) -> AgentConfig:
        return cls(
            label=d["label"],
            harness=d["harness"],
            model_id=d["model_id"],
            model_label=d["model_label"],
            branch_prefix=d["branch_prefix"],
            name=d["name"],
            mention=d["mention"],
        )


@dataclass(frozen=True)
class ChecksConfig:
    required: list[str] = field(default_factory=list)
    repair_from: list[str] = field(default_factory=list)
    max_repair_attempts: int = 10
    max_ralph_loops: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> ChecksConfig:
        return cls(
            required=d.get("required", []),
            repair_from=d.get("repair_from", []),
            max_repair_attempts=d.get("max_repair_attempts", 10),
            max_ralph_loops=d.get("max_ralph_loops", 0),
        )


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "anthropic"
    summary_model: str = "claude-sonnet-4-6"

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(
            provider=d.get("provider", "anthropic"),
            summary_model=d.get("summary_model", "claude-sonnet-4-6"),
        )


@dataclass
class LBMConfig:
    agents: list[AgentConfig]
    checks: ChecksConfig
    llm: LLMConfig

    @classmethod
    def from_parsed_toml(cls, raw: dict) -> LBMConfig:
        """Build from a parsed TOML dict (output of tomllib.load)."""
        harnesses = raw.get("harnesses", {})
        agents_raw = raw.get("agents", [])

        agents: list[AgentConfig] = []
        seen_prefixes: set[str] = set()

        for i, entry in enumerate(agents_raw):
            harness = entry["harness"]
            if harness not in harnesses:
                raise ValueError(
                    f"Harness '{harness}' not defined in [harnesses]. "
                    f"Available: {list(harnesses.keys())}"
                )

            model_label = entry["model_label"]
            default_label = f"agent:{harness}-{model_label}"
            default_prefix = f"{harness}-{model_label}/"

            label = entry.get("override_label", default_label)
            branch_prefix = entry.get("override_branch_prefix", default_prefix)

            if branch_prefix in seen_prefixes:
                raise ValueError(
                    f"Duplicate branch_prefix '{branch_prefix}' -- "
                    f"each agent entry must have a unique prefix"
                )
            seen_prefixes.add(branch_prefix)

            name_letter = (
                AGENT_NAME_LETTERS[i]
                if i < len(AGENT_NAME_LETTERS)
                else str(i + 1)
            )

            agents.append(
                AgentConfig(
                    label=label,
                    harness=harness,
                    model_id=entry["model_id"],
                    model_label=model_label,
                    branch_prefix=branch_prefix,
                    name=f"Agent {name_letter}",
                    mention=harnesses[harness].get("mention", ""),
                )
            )

        checks = ChecksConfig.from_dict(raw.get("checks", {}))
        llm = LLMConfig.from_dict(raw.get("llm", {}))

        return cls(agents=agents, checks=checks, llm=llm)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /tmp/lbm-poc
git add scripts/models.py test/test_models.py
git commit -m "feat: add dataclass config models (AgentConfig, ChecksConfig, LLMConfig, LBMConfig)"
```

---

### Task 2: Migrate `agent_ops.py` to use dataclass config

**Files:**
- Modify: `scripts/agent_ops.py`
- Modify: `test/test_agent_ops.py`
- Modify: `test/conftest.py`

This task replaces the dict-based config in `agent_ops.py` with the new
dataclass models. The `load_lbm_config` function returns `LBMConfig` instead
of a dict. Agent lookup functions accept `list[AgentConfig]` instead of
`list[dict]`. All `cmd_*` callers are updated. Existing tests are migrated.

- [ ] **Step 1: Update `conftest.py` fixtures to use `AgentConfig`**

Replace `SAMPLE_AGENTS` dict list with `AgentConfig` instances:

```python
# test/conftest.py
"""Shared fixtures for dev-infra tests."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from models import AgentConfig

SAMPLE_AGENTS = [
    AgentConfig(
        label="agent:claude",
        branch_prefix="claude/",
        name="Agent A",
        mention="@claude",
        harness="claude",
        model_id="global.anthropic.claude-opus-4-6-v1",
        model_label="opus-4-6",
    ),
    AgentConfig(
        label="agent:codex",
        branch_prefix="codex/",
        name="Agent B",
        mention="@codex",
        harness="codex",
        model_id="gpt-5.3-codex",
        model_label="gpt-5.3",
    ),
    AgentConfig(
        label="agent:openhands",
        branch_prefix="openhands/",
        name="Agent C",
        mention="@openhands-agent",
        harness="openhands",
        model_id="gemini/gemini-3.1-pro-preview",
        model_label="gemini-3.1-pro",
    ),
]

SAMPLE_STATUS_TABLE = """## Agent Implementations

| Agent | Status | PR | Preview | Run |
|-------|--------|-----|---------|-----|
| Agent A | ⏳ Running... |  |  |  |
| Agent B | ⏳ Running... |  |  |  |
| Agent C | ⏳ Running... |  |  |  |

---
*Agents are working on this issue. This comment will be updated as each completes.*"""

SAMPLE_STATUS_TABLE_DONE = """## Agent Implementations

| Agent | Status | PR | Preview | Run |
|-------|--------|-----|---------|-----|
| Agent A | ✅ Done | #10 |  | [Logs](https://example.com/run1) |
| Agent B | ✅ Done | #11 |  | [Logs](https://example.com/run2) |
| Agent C | ❌ Failed |  |  | [Logs](https://example.com/run3) |

---
*Agents are working on this issue. This comment will be updated as each completes.*"""


@pytest.fixture
def agents():
    return list(SAMPLE_AGENTS)


@pytest.fixture
def status_table():
    return SAMPLE_STATUS_TABLE


@pytest.fixture
def status_table_done():
    return SAMPLE_STATUS_TABLE_DONE
```

- [ ] **Step 2: Update `agent_ops.py` — imports and config loading**

At the top of `agent_ops.py`, replace the inline `AGENT_NAME_LETTERS` and `load_lbm_config` with imports from `models.py`:

Remove:
- `AGENT_NAME_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"` (line 54)
- The entire `load_lbm_config` function (lines 57-111)

Add import after the `tomllib` block:

```python
from models import AgentConfig, ChecksConfig, LBMConfig, LLMConfig
```

Replace `load_lbm_config`:

```python
def load_lbm_config(path: str | None = None) -> LBMConfig:
    """Read lbm.toml and return typed config."""
    config_path = path or CONFIG_PATH
    with open(config_path, "rb") as f:
        parsed = tomllib.load(f)
    return LBMConfig.from_parsed_toml(parsed)
```

Update `load_config` to return `LBMConfig`:

```python
def load_config(path: str | None = None) -> LBMConfig:
    """Load typed config."""
    return load_lbm_config(path)
```

Update `load_agents` to return `list[AgentConfig]`:

```python
def load_agents(path: str | None = None) -> list[AgentConfig]:
    """Load the agents list from config."""
    return load_config(path).agents
```

- [ ] **Step 3: Update agent lookup functions for `AgentConfig` attribute access**

Change `branch_to_agent`:
```python
def branch_to_agent(agents: list[AgentConfig], branch: str) -> AgentConfig | None:
    """Find agent config by branch prefix."""
    for a in agents:
        if branch.startswith(a.branch_prefix) or branch.lower().startswith(a.branch_prefix.lower()):
            return a
    return None
```

Change `label_to_agent`:
```python
def label_to_agent(agents: list[AgentConfig], label: str) -> AgentConfig | None:
    """Find agent config by label."""
    for a in agents:
        if a.label == label:
            return a
    return None
```

Change `name_to_agent`:
```python
def name_to_agent(agents: list[AgentConfig], name: str) -> AgentConfig | None:
    """Find agent config by name (e.g. 'A', 'Agent A', 'agent a')."""
    name = name.strip().upper()
    if not name.startswith("AGENT"):
        name = f"AGENT {name}"
    for a in agents:
        if a.name.upper() == name:
            return a
    return None
```

- [ ] **Step 4: Update all `cmd_*` functions for attribute access**

In every `cmd_*` function, change dict access to attribute access. Key changes:

`cmd_lookup` — change `agent.get(field, "")` to `getattr(agent, field, "")` and `for k, v in agent.items()` to iterate dataclass fields:
```python
    if agent:
        if field:
            print(getattr(agent, field, ""))
        else:
            from dataclasses import fields
            for f in fields(agent):
                print(f"{f.name}={getattr(agent, f.name)}")
    else:
        sys.exit(1)
```

`cmd_close_previous_prs` — no changes needed (already uses positional args).

`cmd_post_agent_result` — change `agent["name"]` to `agent.name`, `agent.get("harness")` to `agent.harness`, `agent.get("model_label")` to `agent.model_label`:
```python
    agent_name = agent.name if agent else "Agent"
    ...
    harness_label = f"harness:{agent.harness}" if agent else ""
    model_label_tag = f"model:{agent.model_label}" if agent else ""
```

`cmd_dispatch_repair` — change `config.get("max_repair_attempts", 2)` to `config.checks.max_repair_attempts`, `config["agents"]` to `config.agents`, `agent["name"]` to `agent.name`, `agent.get("mention", "")` to `agent.mention`:
```python
    config = load_config()
    agents = config.agents

    ...
    agent_name = agent.name
    mention = agent.mention
```

`cmd_close_losing_prs` — change `agent["branch_prefix"]` to `agent.branch_prefix`.

`cmd_update_status` — change `agent["name"]` to `agent.name`.

`cmd_diagnostics` — change `agent["branch_prefix"]` to `agent.branch_prefix`.

`cmd_generate_config` — change to serialize the dataclass:
```python
def cmd_generate_config(args: list[str]) -> None:
    """Generate flat config output from lbm.toml (for debugging/validation)."""
    check_only = "--check" in args
    config_path = args[0] if args and not args[0].startswith("--") else CONFIG_PATH

    config = load_lbm_config(config_path)
    from dataclasses import asdict
    generated_json = json.dumps(asdict(config), indent=2) + "\n"

    if check_only:
        print("Config is valid.")
        print(generated_json)
    else:
        print(generated_json)
```

- [ ] **Step 5: Update test mocks for dataclass config**

In `test/test_agent_ops.py`, update mocks that return config dicts to return `LBMConfig`:

```python
# Add import at top
from models import AgentConfig, ChecksConfig, LBMConfig, LLMConfig
```

`TestDispatchRepair.test_max_repairs_reached` — change mock config:
```python
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_max_repairs_reached(self, mock_gh, mock_config):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:claude", harness="claude", model_id="x",
                    model_label="y", branch_prefix="claude/",
                    name="Agent A", mention="@claude",
                )
            ],
            checks=ChecksConfig(required=["CI"], repair_from=["CI"], max_repair_attempts=2),
            llm=LLMConfig(),
        )
        mock_gh.side_effect = [
            "claude/42-fix",  # branch
            "2",  # repair count
            "Implements #42",  # pr body
            "",  # issue comment
        ]
        agent_ops.cmd_dispatch_repair(["10", "CI failed"])
```

`TestDispatchRepair.test_not_agent_branch`:
```python
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_not_agent_branch(self, mock_gh, mock_config):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:claude", harness="claude", model_id="x",
                    model_label="y", branch_prefix="claude/",
                    name="Agent A", mention="@claude",
                )
            ],
            checks=ChecksConfig(required=["CI"], repair_from=["CI"], max_repair_attempts=2),
            llm=LLMConfig(),
        )
        mock_gh.return_value = "feature/something"
        agent_ops.cmd_dispatch_repair(["10", "CI failed"])
        assert mock_gh.call_count == 1
```

`TestPostAgentResult.test_applies_three_labels` — update the mock agent to be `AgentConfig`:
```python
        mock_agents.return_value = [
            AgentConfig(
                label="agent:claude", name="Agent A", branch_prefix="claude/",
                mention="@claude", harness="claude", model_id="claude-opus",
                model_label="opus-4-6",
            )
        ]
```

Apply the same pattern to `test_with_pr` and `test_no_pr` — replace dicts with `AgentConfig(...)`.

- [ ] **Step 6: Run full test suite**

Run: `cd /tmp/lbm-poc && uv run pytest test/ -v`
Expected: All 49 existing tests + 7 new model tests PASS

- [ ] **Step 7: Commit**

```bash
cd /tmp/lbm-poc
git add scripts/agent_ops.py test/conftest.py test/test_agent_ops.py
git commit -m "refactor: migrate agent_ops.py to dataclass config models"
```

---

### Task 3: Extract reusable helpers from `agent_ops.py`

**Files:**
- Modify: `scripts/agent_ops.py`
- Modify: `test/test_agent_ops.py`

Extract helpers that will be used by the ralph loop (and clean up existing
code). Each helper is independently testable.

- [ ] **Step 1: Write failing tests for `count_pr_comments` and `count_issue_comments`**

Add to `test/test_agent_ops.py`:

```python
class TestCountPrComments:
    @patch("agent_ops.gh")
    def test_counts_repair_markers(self, mock_gh):
        mock_gh.return_value = "3"
        result = agent_ops.count_pr_comments("10", "repair-attempt")
        assert result == 3
        mock_gh.assert_called_once()

    @patch("agent_ops.gh")
    def test_returns_zero_on_empty(self, mock_gh):
        mock_gh.return_value = ""
        result = agent_ops.count_pr_comments("10", "repair-attempt")
        assert result == 0


class TestCountIssueComments:
    @patch("agent_ops.gh")
    def test_counts_ralph_markers_scoped(self, mock_gh):
        comments = json.dumps([
            {"body": "[ralph-restart 1] Agent B — restarting..."},
            {"body": "[ralph-restart 2] Agent B — restarting..."},
            {"body": "[ralph-restart 1] Agent A — restarting..."},
            {"body": "Some other comment"},
        ])
        mock_gh.return_value = comments
        result = agent_ops.count_issue_comments("42", "ralph-restart", "Agent B")
        assert result == 2

    @patch("agent_ops.gh")
    def test_counts_without_scope(self, mock_gh):
        comments = json.dumps([
            {"body": "[ralph-restart 1] Agent B — restarting..."},
            {"body": "[ralph-restart 1] Agent A — restarting..."},
        ])
        mock_gh.return_value = comments
        result = agent_ops.count_issue_comments("42", "ralph-restart")
        assert result == 2

    @patch("agent_ops.gh")
    def test_returns_zero_on_no_comments(self, mock_gh):
        mock_gh.return_value = "[]"
        result = agent_ops.count_issue_comments("42", "ralph-restart", "Agent A")
        assert result == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestCountPrComments test/test_agent_ops.py::TestCountIssueComments -v`
Expected: FAIL with `AttributeError: module 'agent_ops' has no attribute 'count_pr_comments'`

- [ ] **Step 3: Implement `count_pr_comments` and `count_issue_comments`**

Add to `agent_ops.py` in a new `# Comment counting helpers` section after the agent lookup section:

```python
# ---------------------------------------------------------------------------
# Comment counting helpers
# ---------------------------------------------------------------------------


def count_pr_comments(pr_num: str, marker: str) -> int:
    """Count comments on a PR containing a [marker] tag."""
    result = gh(
        "pr", "view", pr_num,
        "--json", "comments",
        "--jq", f'[.comments[].body | select(contains("[{marker}"))] | length',
        check=False,
    )
    return int(result) if result.isdigit() else 0


def count_issue_comments(issue_num: str, marker: str, scope: str | None = None) -> int:
    """Count comments on an issue containing a [marker] tag.

    If scope is provided, only count comments that also contain that string
    (e.g. agent name for per-agent counting).
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        return 0
    raw = gh("api", f"repos/{repo}/issues/{issue_num}/comments", "--jq", ".", check=False)
    if not raw or raw == "null":
        return 0
    try:
        comments = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    count = 0
    for c in comments:
        body = c.get("body", "")
        if f"[{marker}" not in body:
            continue
        if scope and scope not in body:
            continue
        count += 1
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestCountPrComments test/test_agent_ops.py::TestCountIssueComments -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write failing tests for `close_and_cleanup_pr` and `extract_issue_from_pr`**

Add to `test/test_agent_ops.py`:

```python
class TestCloseAndCleanupPr:
    @patch("agent_ops.gh")
    def test_closes_and_deletes_branch(self, mock_gh):
        mock_gh.side_effect = [
            "",  # pr close
            "my-branch",  # get branch name
            "",  # delete ref
        ]
        agent_ops.close_and_cleanup_pr("10", "Closing for restart.")
        # Verify close was called
        assert mock_gh.call_args_list[0][0][:3] == ("pr", "close", "10")
        # Verify branch delete
        assert "git/refs/heads/my-branch" in str(mock_gh.call_args_list[2])


class TestExtractIssueFromPr:
    @patch("agent_ops.gh")
    def test_extracts_issue_number(self, mock_gh):
        mock_gh.return_value = "Implements #42\nSome other text"
        result = agent_ops.extract_issue_from_pr("10")
        assert result == "42"

    @patch("agent_ops.gh")
    def test_returns_none_when_missing(self, mock_gh):
        mock_gh.return_value = "No issue reference here"
        result = agent_ops.extract_issue_from_pr("10")
        assert result is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestCloseAndCleanupPr test/test_agent_ops.py::TestExtractIssueFromPr -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 7: Implement `close_and_cleanup_pr` and `extract_issue_from_pr`**

Add to `agent_ops.py`:

```python
# ---------------------------------------------------------------------------
# PR lifecycle helpers
# ---------------------------------------------------------------------------


def extract_issue_from_pr(pr_num: str) -> str | None:
    """Extract the linked issue number from a PR body ('Implements #N')."""
    body = gh("pr", "view", pr_num, "--json", "body", "--jq", ".body", check=False)
    m = re.search(r"Implements #(\d+)", body)
    return m.group(1) if m else None


def close_and_cleanup_pr(pr_num: str, comment: str) -> None:
    """Close a PR with a comment and delete its remote branch."""
    gh("pr", "close", pr_num, "--comment", comment, check=False)
    branch = gh("pr", "view", pr_num, "--json", "headRefName", "--jq", ".headRefName", check=False)
    if branch:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if repo:
            gh("api", "-X", "DELETE", f"repos/{repo}/git/refs/heads/{branch}", check=False)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestCloseAndCleanupPr test/test_agent_ops.py::TestExtractIssueFromPr -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Write failing test for `dispatch_agent`**

```python
class TestDispatchAgent:
    @patch("agent_ops.gh")
    def test_dispatches_workflow(self, mock_gh):
        mock_gh.return_value = ""
        agent_ops.dispatch_agent("42", "codex")
        mock_gh.assert_called_once_with(
            "workflow", "run", "lbm-agents.yml",
            "-f", "issue_number=42",
            "-f", "agent=codex",
            check=False,
        )
```

- [ ] **Step 10: Implement `dispatch_agent`**

```python
def dispatch_agent(issue_num: str, agent_harness: str) -> None:
    """Re-dispatch an agent workflow for an issue."""
    gh(
        "workflow", "run", "lbm-agents.yml",
        "-f", f"issue_number={issue_num}",
        "-f", f"agent={agent_harness}",
        check=False,
    )
```

- [ ] **Step 11: Run all tests**

Run: `cd /tmp/lbm-poc && uv run pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 12: Commit**

```bash
cd /tmp/lbm-poc
git add scripts/agent_ops.py test/test_agent_ops.py
git commit -m "refactor: extract reusable helpers (count_pr_comments, close_and_cleanup_pr, dispatch_agent, etc.)"
```

---

### Task 4: Extract `call_llm` helper and refactor `cmd_summarize_pr`

**Files:**
- Modify: `scripts/agent_ops.py`
- Modify: `test/test_agent_ops.py`

- [ ] **Step 1: Write failing test for `call_llm`**

```python
class TestCallLLM:
    @patch("agent_ops.load_lbm_config")
    def test_returns_none_without_api_key(self, mock_config):
        mock_config.return_value = LBMConfig(
            agents=[], checks=ChecksConfig(), llm=LLMConfig(),
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = agent_ops.call_llm("Hello", LLMConfig())
        assert result is None

    @patch("http.client.HTTPSConnection")
    def test_returns_text_on_success(self, mock_conn_cls):
        mock_resp = mock_conn_cls.return_value.getresponse.return_value
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            {"content": [{"text": "Summary here"}]}
        ).encode()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            result = agent_ops.call_llm("Summarize this", LLMConfig())
        assert result == "Summary here"

    @patch("http.client.HTTPSConnection")
    def test_returns_none_on_http_error(self, mock_conn_cls):
        mock_resp = mock_conn_cls.return_value.getresponse.return_value
        mock_resp.status = 401
        mock_resp.read.return_value = b'{"error": "unauthorized"}'

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            result = agent_ops.call_llm("Summarize this", LLMConfig())
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestCallLLM -v`
Expected: FAIL

- [ ] **Step 3: Implement `call_llm`**

Add to `agent_ops.py`:

```python
# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def call_llm(prompt: str, llm_config: LLMConfig) -> str | None:
    """Call an LLM API and return the response text, or None on failure."""
    import http.client

    if llm_config.provider == "portkey":
        api_key = os.environ.get("PORTKEY_API_KEY", "")
        host = "api.portkey.ai"
        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": api_key,
        }
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        host = "api.anthropic.com"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    if not api_key:
        return None

    body = json.dumps({
        "model": llm_config.summary_model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    })

    try:
        conn = http.client.HTTPSConnection(host, timeout=60)
        conn.request("POST", "/v1/messages", body=body.encode(), headers=headers)
        resp = conn.getresponse()
        resp_body = resp.read().decode()
        if resp.status != 200:
            print(f"LLM call failed: HTTP {resp.status} -- {resp_body}", file=sys.stderr)
            return None
        data = json.loads(resp_body)
        return data["content"][0]["text"]
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 4: Refactor `cmd_summarize_pr` to use `call_llm`**

Replace the ~60 lines of inline HTTP logic in `cmd_summarize_pr` (lines 656-723) with:

```python
def cmd_summarize_pr(args: list[str]) -> None:
    """Generate a concise summary of a PR's changes using an LLM."""
    if len(args) < 1:
        print("Usage: summarize-pr <pr_number> [issue_number]", file=sys.stderr)
        sys.exit(1)

    pr_number = args[0]
    issue_number = args[1] if len(args) > 1 else ""

    raw_diff = gh("pr", "diff", pr_number, check=False)
    if not raw_diff.strip():
        print("")
        return

    filtered_chunks: list[str] = []
    excluded_files: list[str] = []
    current_file = ""
    current_chunk: list[str] = []
    changed_lines = 0

    for line in raw_diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if current_file:
                if changed_lines <= MAX_FILE_DIFF_LINES:
                    filtered_chunks.append("".join(current_chunk))
                else:
                    excluded_files.append(f"{current_file} ({changed_lines} lines changed)")
            current_file = line.split(" b/")[-1].strip() if " b/" in line else ""
            current_chunk = [line]
            changed_lines = 0
        elif line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            current_chunk.append(line)
        elif line.startswith("+") or line.startswith("-"):
            current_chunk.append(line)
            changed_lines += 1

    if current_file:
        if changed_lines <= MAX_FILE_DIFF_LINES:
            filtered_chunks.append("".join(current_chunk))
        else:
            excluded_files.append(f"{current_file} ({changed_lines} lines changed)")

    diff = "".join(filtered_chunks)

    commits = gh(
        "pr", "view", pr_number,
        "--json", "commits", "--jq", '.commits[] | "- " + .messageHeadline',
        check=False,
    )
    if commits:
        diff = f"## Commits\n{commits}\n\n{diff}"

    issue_body = ""
    if issue_number:
        issue_body = gh("issue", "view", issue_number, "--json", "body", "--jq", ".body", check=False)

    config = load_config()
    prompt, was_truncated = build_summary_prompt(diff, issue_body)
    summary = call_llm(prompt, config.llm)

    if not summary:
        print("")
        return

    if was_truncated:
        summary += "\n\n> **Note:** The PR diff was truncated for review. Some changes may not be reflected above."
    if excluded_files:
        summary += "\n\n> **Large files excluded from review:** " + ", ".join(excluded_files)
    print(summary)
```

- [ ] **Step 5: Run full test suite**

Run: `cd /tmp/lbm-poc && uv run pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /tmp/lbm-poc
git add scripts/agent_ops.py test/test_agent_ops.py
git commit -m "refactor: extract call_llm helper, simplify cmd_summarize_pr"
```

---

### Task 5: Implement the ralph loop in `cmd_dispatch_repair`

**Files:**
- Modify: `scripts/agent_ops.py`
- Modify: `test/test_agent_ops.py`

- [ ] **Step 1: Write failing tests for ralph loop behavior**

```python
class TestRalphLoop:
    @patch("agent_ops.dispatch_agent")
    @patch("agent_ops.close_and_cleanup_pr")
    @patch("agent_ops.call_llm")
    @patch("agent_ops.count_issue_comments")
    @patch("agent_ops.count_pr_comments")
    @patch("agent_ops.extract_issue_from_pr")
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_ralph_restart_when_repairs_exhausted(
        self, mock_gh, mock_config, mock_extract, mock_count_pr,
        mock_count_issue, mock_llm, mock_close, mock_dispatch,
    ):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:codex", harness="codex", model_id="x",
                    model_label="y", branch_prefix="codex/",
                    name="Agent B", mention="@codex",
                )
            ],
            checks=ChecksConfig(
                required=["CI"], repair_from=["CI"],
                max_repair_attempts=3, max_ralph_loops=2,
            ),
            llm=LLMConfig(),
        )
        mock_gh.return_value = "codex/42-fix"  # branch lookup
        mock_extract.return_value = "42"
        mock_count_pr.return_value = 3  # at max repairs
        mock_count_issue.return_value = 0  # no ralph restarts yet
        mock_llm.return_value = "Added a Footer component that broke lint."

        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            agent_ops.cmd_dispatch_repair(["10", "lint errors"])

        mock_close.assert_called_once()
        assert "ralph restart" in mock_close.call_args[0][1].lower()
        mock_dispatch.assert_called_once_with("42", "codex")
        # Verify ralph restart comment posted on issue
        issue_comment_calls = [
            c for c in mock_gh.call_args_list
            if len(c[0]) >= 3 and c[0][:3] == ("issue", "comment", "42")
        ]
        assert len(issue_comment_calls) == 1
        assert "[ralph-restart 1]" in issue_comment_calls[0][0][4]  # --body arg

    @patch("agent_ops.dispatch_agent")
    @patch("agent_ops.count_issue_comments")
    @patch("agent_ops.count_pr_comments")
    @patch("agent_ops.extract_issue_from_pr")
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_manual_intervention_when_ralph_exhausted(
        self, mock_gh, mock_config, mock_extract, mock_count_pr,
        mock_count_issue, mock_dispatch,
    ):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:codex", harness="codex", model_id="x",
                    model_label="y", branch_prefix="codex/",
                    name="Agent B", mention="@codex",
                )
            ],
            checks=ChecksConfig(
                required=["CI"], repair_from=["CI"],
                max_repair_attempts=3, max_ralph_loops=2,
            ),
            llm=LLMConfig(),
        )
        mock_gh.return_value = "codex/42-fix"
        mock_extract.return_value = "42"
        mock_count_pr.return_value = 3
        mock_count_issue.return_value = 2  # at max ralph loops

        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            agent_ops.cmd_dispatch_repair(["10", "lint errors"])

        mock_dispatch.assert_not_called()
        intervention_calls = [
            c for c in mock_gh.call_args_list
            if len(c[0]) >= 3 and c[0][:3] == ("issue", "comment", "42")
        ]
        assert len(intervention_calls) == 1
        assert "manual intervention" in intervention_calls[0][0][4].lower()

    @patch("agent_ops.count_issue_comments")
    @patch("agent_ops.count_pr_comments")
    @patch("agent_ops.extract_issue_from_pr")
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_ralph_disabled_when_zero(
        self, mock_gh, mock_config, mock_extract, mock_count_pr,
        mock_count_issue,
    ):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:codex", harness="codex", model_id="x",
                    model_label="y", branch_prefix="codex/",
                    name="Agent B", mention="@codex",
                )
            ],
            checks=ChecksConfig(
                required=["CI"], repair_from=["CI"],
                max_repair_attempts=3, max_ralph_loops=0,  # disabled
            ),
            llm=LLMConfig(),
        )
        mock_gh.return_value = "codex/42-fix"
        mock_extract.return_value = "42"
        mock_count_pr.return_value = 3

        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo"}):
            agent_ops.cmd_dispatch_repair(["10", "lint errors"])

        # Should NOT count issue comments (ralph disabled)
        mock_count_issue.assert_not_called()
        # Should post manual intervention
        intervention_calls = [
            c for c in mock_gh.call_args_list
            if len(c[0]) >= 3 and c[0][:3] == ("issue", "comment", "42")
        ]
        assert len(intervention_calls) == 1
        assert "manual intervention" in intervention_calls[0][0][4].lower()

    @patch("agent_ops.count_pr_comments")
    @patch("agent_ops.load_config")
    @patch("agent_ops.gh")
    def test_normal_repair_when_under_max(self, mock_gh, mock_config, mock_count_pr):
        mock_config.return_value = LBMConfig(
            agents=[
                AgentConfig(
                    label="agent:codex", harness="codex", model_id="x",
                    model_label="y", branch_prefix="codex/",
                    name="Agent B", mention="@codex",
                )
            ],
            checks=ChecksConfig(
                required=["CI"], repair_from=["CI"],
                max_repair_attempts=10, max_ralph_loops=2,
            ),
            llm=LLMConfig(),
        )
        mock_gh.return_value = "codex/42-fix"
        mock_count_pr.return_value = 2  # under max

        with patch.dict(os.environ, {"PAT_TOKEN": "pat-123"}):
            agent_ops.cmd_dispatch_repair(["10", "lint errors"])

        # Should dispatch a repair comment (via subprocess, not our mock)
        # The normal repair path uses subprocess.run directly
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestRalphLoop -v`
Expected: FAIL

- [ ] **Step 3: Implement the refactored `cmd_dispatch_repair` with ralph loop**

Replace the entire `cmd_dispatch_repair` function:

```python
def cmd_dispatch_repair(args: list[str]) -> None:
    """Dispatch a repair or ralph restart for a failing PR."""
    if len(args) < 2:
        print("Usage: dispatch-repair <pr_number> <failure_context>", file=sys.stderr)
        sys.exit(1)

    pr_num = args[0]
    failure_context = args[1]

    config = load_config()

    # Identify the agent from the PR's branch
    branch = gh("pr", "view", pr_num, "--json", "headRefName", "--jq", ".headRefName", check=False)
    if not branch:
        print(f"PR #{pr_num} not found")
        return

    agent = branch_to_agent(config.agents, branch)
    if not agent:
        print(f"Not an agent branch: {branch}")
        return

    # Check repair count
    repair_count = count_pr_comments(pr_num, "repair-attempt")
    print(f"{agent.name} PR #{pr_num}: {repair_count} / {config.checks.max_repair_attempts} repairs")

    if repair_count < config.checks.max_repair_attempts:
        _dispatch_repair_comment(pr_num, agent, failure_context)
        return

    # Repairs exhausted — try ralph restart
    issue_num = extract_issue_from_pr(pr_num)
    if not issue_num:
        print("Cannot find linked issue for ralph restart")
        return

    if config.checks.max_ralph_loops > 0:
        ralph_count = count_issue_comments(issue_num, "ralph-restart", agent.name)
        if ralph_count < config.checks.max_ralph_loops:
            _ralph_restart(pr_num, issue_num, agent, config, ralph_count, failure_context)
            return

    # Truly exhausted
    _post_manual_intervention(issue_num, agent, pr_num, config.checks)


def _dispatch_repair_comment(pr_num: str, agent: AgentConfig, failure_context: str) -> None:
    """Post a repair-attempt comment on the PR to trigger the agent."""
    pat_token = os.environ.get("PAT_TOKEN", "")
    if not agent.mention or not pat_token:
        print("Cannot dispatch repair (no mention or no PAT_TOKEN)")
        return

    repair_body = (
        f"{agent.mention} [repair-attempt] {failure_context}\n\n"
        f"Fix ALL errors listed above -- there may be multiple issues across lint, typecheck, and build.\n"
        f"Before committing, run the full CI check locally.\n"
        f"Only commit and push when ALL steps pass."
    )

    env = {**os.environ, "GH_TOKEN": pat_token}
    subprocess.run(
        ["gh", "pr", "comment", pr_num, "--body", repair_body],
        env=env,
        check=False,
    )
    repair_count = count_pr_comments(pr_num, "repair-attempt")
    print(f"Dispatched repair attempt {repair_count} for {agent.name} PR #{pr_num}")


def _ralph_restart(
    pr_num: str,
    issue_num: str,
    agent: AgentConfig,
    config: LBMConfig,
    ralph_count: int,
    failure_context: str,
) -> None:
    """Wipe the PR and restart the agent from scratch."""
    max_loops = config.checks.max_ralph_loops
    attempt = ralph_count + 1

    # Generate summary of failed approach
    summary = _summarize_failed_attempt(pr_num, failure_context, config.llm)

    # Close PR and delete branch
    close_and_cleanup_pr(pr_num, f"Closing for ralph restart (attempt {attempt}/{max_loops}).")

    # Post restart marker on issue
    restart_body = f"[ralph-restart {attempt}] {agent.name} — restarting after {config.checks.max_repair_attempts} failed repairs on PR #{pr_num}."
    if summary:
        restart_body += f"\n\nPrevious approach: {summary}"
    gh("issue", "comment", issue_num, "--body", restart_body)

    # Re-dispatch the agent
    dispatch_agent(issue_num, agent.harness)
    print(f"Ralph restart {attempt}/{max_loops} for {agent.name} on issue #{issue_num}")


def _summarize_failed_attempt(pr_num: str, failure_context: str, llm_config: LLMConfig) -> str:
    """Generate a 2-3 sentence summary of a failed PR approach."""
    diff = gh("pr", "diff", pr_num, check=False)
    if not diff.strip():
        return ""

    # Truncate diff for the summary prompt
    truncated_diff = diff[:50000]
    truncated_context = failure_context[:2000]

    prompt = (
        "Summarize in 2-3 sentences what approach this PR took and why it kept failing. "
        "Focus on the overall strategy and the root cause of failure, not individual errors.\n\n"
        f"## Last CI errors\n{truncated_context}\n\n"
        f"## PR diff (truncated)\n```diff\n{truncated_diff}\n```"
    )

    return call_llm(prompt, llm_config) or ""


def _post_manual_intervention(issue_num: str, agent: AgentConfig, pr_num: str, checks: ChecksConfig) -> None:
    """Post the terminal failure message on the issue."""
    if checks.max_ralph_loops > 0:
        msg = (
            f"{agent.name} PR #{pr_num} has failed after {checks.max_ralph_loops} restart cycles "
            f"({checks.max_repair_attempts} repairs each). Manual intervention needed."
        )
    else:
        msg = (
            f"{agent.name} PR #{pr_num} has failed after {checks.max_repair_attempts} repair attempts. "
            f"Manual intervention needed."
        )
    gh("issue", "comment", issue_num, "--body", msg)
```

- [ ] **Step 4: Run ralph loop tests**

Run: `cd /tmp/lbm-poc && uv run pytest test/test_agent_ops.py::TestRalphLoop -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /tmp/lbm-poc && uv run pytest test/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /tmp/lbm-poc
git add scripts/agent_ops.py test/test_agent_ops.py
git commit -m "feat: implement ralph loop — wipe PR and restart agent when repairs exhausted"
```

---

### Task 6: Update template and linting

**Files:**
- Modify: `templates/lbm.toml.j2`

- [ ] **Step 1: Add `max_ralph_loops` to the TOML template**

In `templates/lbm.toml.j2`, after the existing `max_repair_attempts = 10` line, add:

```toml
max_ralph_loops = 0
```

The full `[checks]` section becomes:
```toml
[checks]
required = ["CI"]
repair_from = ["CI"]
max_repair_attempts = 10
max_ralph_loops = 0
```

- [ ] **Step 2: Run linter and tests**

Run: `cd /tmp/lbm-poc && uv run ruff check . && uv run ruff format --check . && uv run pytest test/ -v`
Expected: All clean, all tests PASS

- [ ] **Step 3: Fix any lint issues**

Run: `cd /tmp/lbm-poc && uv run ruff format .`

- [ ] **Step 4: Commit**

```bash
cd /tmp/lbm-poc
git add templates/lbm.toml.j2
git commit -m "feat: add max_ralph_loops config to lbm.toml template"
```

---

### Task 7: Enable ralph loop in 484reviewer for E2E testing

**Files:**
- Modify: `/home/ubuntu/484reviewer/lbm.toml`

This task configures 484reviewer for an E2E smoke test of the ralph loop by
setting `max_ralph_loops = 1` and `max_repair_attempts = 2` (low values so
the loop triggers quickly).

- [ ] **Step 1: Update 484reviewer's `lbm.toml`**

Add `max_ralph_loops = 1` and lower `max_repair_attempts` to `2`:

```toml
[checks]
required = ["CI"]
repair_from = ["CI"]
max_repair_attempts = 2
max_ralph_loops = 1
```

- [ ] **Step 2: Create a test issue designed to always fail CI**

Create a GitHub issue in 484reviewer with a task that will inevitably produce CI failures, forcing the repair loop to exhaust and trigger ralph:

```bash
gh issue create --repo henryre/484reviewer \
  --title "Test: ralph loop E2E (intentional CI failure)" \
  --body "Add a file called src/ralph-test.ts that exports a function with a deliberate type error (return type mismatch). This WILL fail typecheck, which is the point — we are testing the ralph loop restart mechanism.

DO NOT fix the type error. The goal is for CI to fail."
```

- [ ] **Step 3: Commit and push lbm.toml change, label the issue**

```bash
cd /home/ubuntu/484reviewer
git add lbm.toml
git commit -m "test: lower repair attempts and enable ralph loop for E2E test"
git push origin main
```

Then label the issue:
```bash
gh issue edit <ISSUE_NUM> --repo henryre/484reviewer --add-label "ready-for-dev"
```

- [ ] **Step 4: Monitor the workflow**

Watch for the ralph loop to trigger. Expected sequence:
1. Agent creates PR with type error
2. CI fails, repair 1/2 dispatched
3. CI fails again, repair 2/2 dispatched
4. CI fails, repairs exhausted → ralph restart 1/1 triggered
5. PR closed, branch deleted, new agent run dispatched
6. Agent creates new PR
7. CI fails, repair 1/2 dispatched
8. CI fails, repair 2/2 dispatched
9. CI fails, repairs exhausted, ralph exhausted → "Manual intervention needed"

- [ ] **Step 5: After verification, restore normal config**

```toml
[checks]
required = ["CI"]
repair_from = ["CI"]
max_repair_attempts = 10
max_ralph_loops = 0
```

```bash
cd /home/ubuntu/484reviewer
git add lbm.toml
git commit -m "revert: restore normal repair/ralph config after E2E test"
git push origin main
```
