#!/usr/bin/env python3
"""Consolidated dev-infra operations for the LBM agent orchestration.

Replaces: close-previous-prs.sh, post-agent-result.sh, close-losing-prs.sh,
dispatch-repair.sh, agent-lookup.py, update-status.py, summarize-pr.py.

Usage:
  python3 scripts/agent_ops.py <command> [args...]

Commands:
  lookup <subcommand> <value> [field]
  close-previous-prs <issue> <prefix> <label>
  post-agent-result <issue> <label> [pr] [run_url]
  close-losing-prs <issue> <winner_pr> [winner_name]
  record-no-winner <issue> [reason]
  dispatch-repair <pr> <context>
  update-status <issue> <label> <status> [pr] [preview] [run]
  summarize-pr <pr_number>
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import base64
import random

import config_parser
from models import AgentConfig, ChecksConfig, LBMConfig, LLMConfig

CONFIG_PATH = os.environ.get(
    "LBM_CONFIG_PATH",
    os.path.join(os.getcwd(), "lbm.toml"),
)


# ---------------------------------------------------------------------------
# gh CLI wrapper
# ---------------------------------------------------------------------------


def gh(*args: str, check: bool = True) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=check)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_lbm_config(path: str | None = None) -> LBMConfig:
    """Read lbm.toml and return a typed LBMConfig."""
    config_path = path or CONFIG_PATH
    with open(config_path, "rb") as f:
        parsed = tomllib.load(f)
    return LBMConfig.from_parsed_toml(parsed)


def load_config(path: str | None = None) -> LBMConfig:
    """Load the config as a typed LBMConfig."""
    return load_lbm_config(path)


def load_agents(path: str | None = None) -> list[AgentConfig]:
    """Load the agents list from config."""
    return load_config(path).agents


# ---------------------------------------------------------------------------
# Agent lookup (pure functions)
# ---------------------------------------------------------------------------


def branch_to_agent(agents: list[AgentConfig], branch: str) -> AgentConfig | None:
    """Find agent config by branch prefix."""
    for a in agents:
        if branch.startswith(a.branch_prefix) or branch.lower().startswith(a.branch_prefix.lower()):
            return a
    return None


def label_to_agent(agents: list[AgentConfig], label: str) -> AgentConfig | None:
    """Find agent config by label."""
    for a in agents:
        if a.label == label:
            return a
    return None


def name_to_agent(agents: list[AgentConfig], name: str) -> AgentConfig | None:
    """Find agent config by name (e.g. 'A', 'Agent A', 'agent a')."""
    name = name.strip().upper()
    if not name.startswith("AGENT"):
        name = f"AGENT {name}"
    for a in agents:
        if a.name.upper() == name:
            return a
    return None


# ---------------------------------------------------------------------------
# Agent aliases
# ---------------------------------------------------------------------------

ALIAS_ADJECTIVES = [
    "Swift", "Bold", "Keen", "Bright", "Calm", "Deft", "Wise", "Brave",
    "Quick", "Sharp", "Sly", "Warm", "Wild", "Pale", "Rare", "True",
    "Pure", "Firm", "Neat", "Still",
]

ALIAS_ANIMALS = [
    "Falcon", "Otter", "Panda", "Fox", "Dolphin", "Owl", "Wolf", "Lynx",
    "Raven", "Cobra", "Heron", "Badger", "Crane", "Viper", "Elk", "Finch",
    "Hawk", "Bear", "Wren", "Frog",
]

ALIAS_MARKER = "<!-- lbm:aliases:"


def generate_aliases(agents: list[AgentConfig]) -> dict[str, str]:
    """Generate random adjective+animal aliases for agents.

    Returns {alias: agent.label} mapping.
    """
    combos = [(adj, ani) for adj in ALIAS_ADJECTIVES for ani in ALIAS_ANIMALS]
    selected = random.sample(combos, min(len(agents), len(combos)))
    return {f"{adj} {ani}": agent.label for (adj, ani), agent in zip(selected, agents)}


def alias_to_agent(agents: list[AgentConfig], alias: str, mapping: dict[str, str]) -> AgentConfig | None:
    """Look up an alias in the mapping dict, then find the matching agent."""
    agent_label = mapping.get(alias)
    if not agent_label:
        return None
    return label_to_agent(agents, agent_label)


def encode_alias_mapping(mapping: dict[str, str]) -> str:
    """Encode alias mapping as b64 string for embedding in a comment."""
    return base64.b64encode(json.dumps(mapping).encode()).decode()


def decode_alias_mapping(encoded: str) -> dict[str, str]:
    """Decode a b64-encoded alias mapping."""
    return json.loads(base64.b64decode(encoded).decode())


def read_alias_mapping(issue_num: str) -> dict[str, str] | None:
    """Read alias mapping from the status comment on an issue.

    Returns the decoded mapping or None if not found.
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        return None

    raw = gh("api", f"repos/{repo}/issues/{issue_num}/comments", "--jq", ".", check=False)
    if not raw or raw == "null":
        return None

    try:
        comments = json.loads(raw)
    except json.JSONDecodeError:
        return None

    for c in comments:
        body = c.get("body", "")
        idx = body.find(ALIAS_MARKER)
        if idx == -1:
            continue
        start = idx + len(ALIAS_MARKER)
        end = body.find(" -->", start)
        if end == -1:
            continue
        encoded = body[start:end].strip()
        try:
            return decode_alias_mapping(encoded)
        except (json.JSONDecodeError, Exception):
            continue

    return None


def reverse_alias_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Return {agent_label: alias} from an {alias: agent_label} mapping."""
    return {v: k for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Comment counting helpers
# ---------------------------------------------------------------------------


def count_pr_comments(pr_num: str, marker: str) -> int:
    """Count comments on a PR containing a [marker] tag."""
    result = gh(
        "pr",
        "view",
        pr_num,
        "--json",
        "comments",
        "--jq",
        f'[.comments[].body | select(contains("[{marker}"))] | length',
        check=False,
    )
    return int(result) if result.isdigit() else 0


def count_issue_comments(issue_num: str, marker: str, scope: str | None = None) -> int:
    """Count comments on an issue containing a [marker] tag, optionally scoped to a string."""
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


# ---------------------------------------------------------------------------
# PR lifecycle helpers
# ---------------------------------------------------------------------------


def extract_issue_from_pr(pr_num: str) -> str | None:
    """Extract the linked issue number from a PR body (Implements #N)."""
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


def dispatch_agent(issue_num: str, agent_harness: str) -> None:
    """Re-dispatch an agent workflow via gh workflow run."""
    gh(
        "workflow",
        "run",
        "lbm-agents.yml",
        "-f",
        f"issue_number={issue_num}",
        "-f",
        f"agent={agent_harness}",
        check=False,
    )


# ---------------------------------------------------------------------------
# Plan step (two-phase iterations) — pure helpers
# ---------------------------------------------------------------------------

PLAN_STATS_MARKER = "<!-- lbm:stats"


def build_task_prompt(
    issue_num: str, phase: str, plan_dir: str = "lbm-plans", prototype: bool = False
) -> str:
    """Phase-specific agent instructions, appended to the issue body / system prompt.

    Plan phase: write a plan to the canonical per-issue path and open a PR, do
    not implement. Implement phase: implement against the approved plan already
    merged to the default branch. The path is identical on every branch so the
    winning plan merges cleanly with no rename. When ``prototype`` is set (plan
    phase only), the agent may first open a prototype PR to gather evidence
    before writing the plan.
    """
    path = f"{plan_dir}/issue-{issue_num}/plan.md"
    if phase == "plan":
        base = (
            f"PLAN PHASE — do NOT implement the feature yet. Research the codebase and "
            f"write a concrete implementation plan for issue #{issue_num} to `{path}` "
            f"(create the directory). Open a PR titled 'Plan: <summary>'. The plan should "
            f"cover approach, files to change, testing, and risks. Commit and push only "
            f"the plan file."
        )
        if prototype:
            base += (
                " Optionally, if hands-on evidence would materially improve the plan, you "
                "MAY first open a small PROTOTYPE PR (title 'Prototype: <summary>') with "
                "minimal code to test one uncertainty; its automated run will post a report "
                "and you will be re-invoked to write the plan using that report as context. "
                "Skip the prototype when it isn't warranted."
            )
        return base
    return (
        f"IMPLEMENT PHASE — implement issue #{issue_num} following the approved plan at "
        f"`{path}` (already merged to the default branch; read it first). Open a PR with "
        f"the implementation. Commit and push."
    )


def plan_file_url(repo: str, branch: str, issue_num: str, plan_dir: str = "lbm-plans") -> str:
    """GitHub blob URL that renders a branch's plan.md in Markdown-preview mode."""
    return f"https://github.com/{repo}/blob/{branch}/{plan_dir}/issue-{issue_num}/plan.md"


def _plan_rev_allowed(current_revs: int, cap: int) -> bool:
    """True if another plan-feedback revision is within the configured cap."""
    return current_revs < cap


# ---------------------------------------------------------------------------
# Status table helpers (pure functions)
# ---------------------------------------------------------------------------


def find_status_row(body: str, agent_name: str) -> re.Match | None:
    """Find a row in the status table for the given agent name."""
    pattern = re.compile(
        rf"^\| {re.escape(agent_name)} \|[^|]*\|[^|]*\|[^|]*\|[^|]*\|([^|]*\|)?$",
        re.MULTILINE,
    )
    return pattern.search(body)


def update_status_row(
    body: str, agent_name: str, status: str, pr: str, preview: str, run: str, preview_label: str = "Preview"
) -> str:
    """Update a single row in the status table. Returns the updated body.

    ``preview_label`` names the 4th-column link (default "Preview"; the plan
    phase reuses the same table shape with the label "plan")."""
    match = find_status_row(body, agent_name)
    if not match:
        return body

    old_row = match.group(0)
    cells = [c.strip() for c in old_row.split("|")[1:-1]]
    while len(cells) < 5:
        cells.append("")

    # Format cell values
    if status == "done" and pr:
        status_text = "✅ Done"
        pr_text = f"#{pr}"
    elif status == "failed":
        status_text = "❌ Failed"
        pr_text = ""
    elif status == "no-changes":
        status_text = "⚠️ No changes"
        pr_text = ""
    elif status == "blocked":
        # An environmental/infra failure the repair loop can't fix — needs a human.
        status_text = "⛔ Needs human"
        pr_text = None
    elif status == "preview":
        status_text = None
        pr_text = None
    else:
        status_text = "✅ Done"
        pr_text = ""

    if status_text is not None:
        cells[1] = status_text
    if pr_text is not None:
        cells[2] = pr_text
    if preview:
        cells[3] = f"[{preview_label}]({preview})"
    if run:
        cells[4] = f"[Logs]({run})"

    new_row = "| " + " | ".join(cells) + " |"
    return body[: match.start()] + new_row + body[match.end() :]


def check_all_done(body: str) -> str:
    """If no pending indicator remains in the body, replace the 'agents working' message."""
    if "Pending" not in body and "Running" not in body:
        return body.replace(
            "*Agents are working on this issue. This comment will be updated as each completes.*",
            "*All agents have completed. Review the PRs and use `/merge <alias>` to select the best one.*",
        )
    return body


# ---------------------------------------------------------------------------
# Summary prompt builder (pure function)
# ---------------------------------------------------------------------------

MAX_DIFF_LENGTH = 200000
MAX_FILE_DIFF_LINES = 500


def build_summary_prompt(diff: str, issue_body: str = "") -> tuple[str, bool]:
    """Build the LLM prompt for PR summary generation.

    Returns (prompt, was_truncated) tuple.
    """
    was_truncated = len(diff) > MAX_DIFF_LENGTH
    truncated = diff[:MAX_DIFF_LENGTH]

    if issue_body:
        return (
            f"""You are reviewing a pull request that implements features from a GitHub issue.

## Requested changes (from the issue)

{issue_body}

## PR diff

```diff
{truncated}
```

{"**Note: The diff was truncated. Some changes may not be visible.**" if was_truncated else ""}

Write a summary with these three sections:

### Coverage

Create a markdown table mapping each requested feature to what was done.
Use this exact format (one row per requested item):

| | Requested | Status |
|---|-----------|--------|
| <emoji> | <feature name from issue> | <what was implemented, or "Not implemented"> |

For the emoji column use exactly one of:
- if fully implemented (all requirements from the issue met)
- if partially implemented (state what's done and what's missing)
- if not implemented at all

### Other changes
- List changes that don't map to a requested feature (refactors, config, utilities). Omit section if none.

Focus ONLY on completeness — whether each requested feature was implemented or not.
Do NOT assess code quality, implementation approach, or whether it was done "the right way".
Do NOT flag concerns, suggest improvements, or critique decisions.
Be specific and terse. One sentence per bullet. No filler.""",
            was_truncated,
        )
    else:
        return (
            f"""You are reviewing a pull request diff. Write a concise summary.

### What changed
- Key behavioral/user-facing changes

### Implementation notes
- Notable implementation decisions

Be specific and terse. One sentence per bullet. No filler.

```diff
{truncated}
```

{"**Note: The diff was truncated. Some changes may not be visible.**" if was_truncated else ""}""",
            was_truncated,
        )


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

    body = json.dumps(
        {
            "model": llm_config.summary_model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
    )

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


# ---------------------------------------------------------------------------
# Commands (orchestrators with I/O)
# ---------------------------------------------------------------------------


def cmd_lookup(args: list[str]) -> None:
    """Lookup agent config. Usage: lookup <subcommand> <value> [field]"""
    if len(args) < 2:
        print("Usage: lookup <branch-to-name|label-to-name|name-to-label> <value> [field]", file=sys.stderr)
        sys.exit(1)

    subcmd, value = args[0], args[1]
    field = args[2] if len(args) > 2 else None
    agents = load_agents()

    if subcmd == "branch-to-name":
        agent = branch_to_agent(agents, value)
    elif subcmd == "label-to-name":
        agent = label_to_agent(agents, value)
    elif subcmd == "name-to-label":
        agent = name_to_agent(agents, value)
    else:
        print(f"Unknown lookup subcommand: {subcmd}", file=sys.stderr)
        sys.exit(1)

    if agent:
        if field:
            print(getattr(agent, field, ""))
        else:
            from dataclasses import fields as dc_fields

            for f in dc_fields(agent):
                print(f"{f.name}={getattr(agent, f.name)}")
    else:
        sys.exit(1)


def cmd_close_previous_prs(args: list[str]) -> None:
    """Close previous agent PRs for an issue."""
    if len(args) < 3:
        print("Usage: close-previous-prs <issue_number> <branch_prefix> <agent_label>", file=sys.stderr)
        sys.exit(1)

    issue_num, branch_prefix, agent_label = args[0], args[1], args[2]

    jq_filter = (
        f".[] | select("
        f'(.body | test("Implements #{issue_num}\\\\b")) and '
        f'((.headRefName | startswith("{branch_prefix}")) or '
        f'(.labels | map(.name) | index("{agent_label}"))))'
        f" | .number"
    )

    output = gh(
        "pr",
        "list",
        "--json",
        "number,body,headRefName,labels",
        "--jq",
        jq_filter,
        check=False,
    )

    for line in output.splitlines():
        pr = line.strip()
        if pr:
            print(f"Closing old PR #{pr}")
            gh("pr", "close", pr, "--comment", "Superseded by new agent run.", "--delete-branch", check=False)


def cmd_post_agent_result(args: list[str]) -> None:
    """Post agent result: label PR, update status table, comment on issue."""
    if len(args) < 2:
        print("Usage: post-agent-result <issue_number> <agent_label> [pr_number] [run_url]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    agent_label = args[1]
    pr_num = args[2] if len(args) > 2 else ""
    run_url = args[3] if len(args) > 3 else ""

    agents = load_agents()
    agent = label_to_agent(agents, agent_label)

    # Resolve display name via alias mapping
    mapping = read_alias_mapping(issue_num)
    if mapping:
        reverse = reverse_alias_mapping(mapping)
        agent_name = reverse.get(agent_label, agent.name if agent else "Agent")
    else:
        agent_name = agent.name if agent else "Agent"

    if not pr_num:
        cmd_update_status([issue_num, agent_label, "no-changes", "", "", run_url])
        return

    # Apply three labels: agent (stable ID), harness, model
    harness_label = f"harness:{agent.harness}" if agent and agent.harness else ""
    model_label_tag = f"model:{agent.model_label}" if agent and agent.model_label else ""
    all_labels = list(filter(None, [agent_label, harness_label, model_label_tag]))
    # Ensure labels exist on the repo (gh pr edit silently fails for missing labels)
    for lbl in all_labels:
        gh("label", "create", lbl, "--force", check=False)
    label_args_list: list[str] = []
    for lbl in all_labels:
        label_args_list += ["--add-label", lbl]
    gh("pr", "edit", pr_num, *label_args_list, check=False)
    cmd_update_status([issue_num, agent_label, "done", pr_num, "", run_url])

    # "Deploying..." only makes sense with an active deploy platform. For repos
    # whose preview is a report/comment (deploy=none), the preview arrives later
    # via a preview_comment_marker -> update-status, so start neutral.
    preview_placeholder = "Deploying..." if load_config().deploy.platform != "none" else "_pending_"
    body = f"""## {agent_name} Implementation
- **PR**: #{pr_num}
- **Preview**: {preview_placeholder}"""

    if run_url:
        body += f"\n- **Run**: [View logs]({run_url})"

    body += "\n\n---\n*Review and compare agent implementations, then merge the best one.*"

    gh("issue", "comment", issue_num, "--body", body)


def close_agent_prs(issue_num: str, comment: str, exclude_pr: str | None = None) -> list[str]:
    """Close every agent PR for an issue, optionally excluding one.

    Returns the list of closed PR numbers.
    """
    agents = load_agents()
    closed: list[str] = []
    for agent in agents:
        prefix = agent.branch_prefix
        jq_filter = (
            f".[] | select("
            f'(.body | test("Implements #{issue_num}\\\\b")) and '
            f'(.headRefName | startswith("{prefix}")))'
            f" | .number"
        )
        output = gh(
            "pr",
            "list",
            "--json",
            "number,body,headRefName",
            "--jq",
            jq_filter,
            check=False,
        )
        for line in output.splitlines():
            pr = line.strip()
            if pr and pr != exclude_pr:
                print(f"Closing PR #{pr}")
                gh("pr", "close", pr, "--comment", comment, "--delete-branch", check=False)
                closed.append(pr)
    return closed


def cmd_close_losing_prs(args: list[str]) -> None:
    """Close all agent PRs for an issue except the winner."""
    if len(args) < 2:
        print("Usage: close-losing-prs <issue_number> <winner_pr> [winner_name]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    winner_pr = args[1]
    winner_name = args[2] if len(args) > 2 else "Agent"

    close_agent_prs(
        issue_num,
        f"Closed: {winner_name} (PR #{winner_pr}) was selected.",
        exclude_pr=winner_pr,
    )


# ---------------------------------------------------------------------------
# No-winner verdict
# ---------------------------------------------------------------------------

# Durable marker the LBM Hub keys off to distinguish an explicit "no winner"
# verdict from an issue closed for any other reason.
NO_WINNER_MARKER = "<!-- lbm:no-winner -->"
NO_WINNER_LABEL = "outcome:no-winner"


def cmd_record_no_winner(args: list[str]) -> None:
    """Record an explicit 'no winner' verdict for an iteration.

    Closes every agent PR, posts a machine-readable marker comment, and labels
    the issue. The issue itself is closed by the calling workflow (mirroring how
    the /merge job closes the issue after merging).

    Usage: record-no-winner <issue_number> [reason]
    """
    if len(args) < 1:
        print("Usage: record-no-winner <issue_number> [reason]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    reason = args[1].strip() if len(args) > 1 else ""

    closed = close_agent_prs(issue_num, "Closed: no winner was selected for this iteration.")
    print(f"Closed {len(closed)} agent PR(s)")

    reason_text = f"\n\n> {reason}" if reason else ""
    body = (
        f"{NO_WINNER_MARKER}\n"
        "**No winner** — no agent produced a mergeable implementation for this iteration."
        f"{reason_text}"
    )
    gh("issue", "comment", issue_num, "--body", body, check=False)

    gh(
        "label",
        "create",
        NO_WINNER_LABEL,
        "--color",
        "6E738D",
        "--description",
        "Iteration closed with no winning agent",
        check=False,
    )
    gh("issue", "edit", issue_num, "--add-label", NO_WINNER_LABEL, check=False)


def cmd_dispatch_repair(args: list[str]) -> None:
    """Dispatch a repair or ralph restart for a failing PR."""
    if len(args) < 2:
        print("Usage: dispatch-repair <pr_number> <failure_context>", file=sys.stderr)
        sys.exit(1)

    pr_num = args[0]
    failure_context = args[1]

    config = load_config()

    branch = gh("pr", "view", pr_num, "--json", "headRefName", "--jq", ".headRefName", check=False)
    if not branch:
        print(f"PR #{pr_num} not found")
        return

    agent = branch_to_agent(config.agents, branch)
    if not agent:
        print(f"Not an agent branch: {branch}")
        return

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

    _post_manual_intervention(issue_num, agent, pr_num, config.checks)


def cmd_dispatch_plan_context(args: list[str]) -> None:
    """Prototype iteration: re-invoke an agent to write its plan using a report.

    Triggered when a prototype PR's auto-run report carries the plan-context
    marker. Resolves the agent + linked issue from the PR and posts an
    ``@mention [plan-context] <report>`` comment (via PAT so it triggers the
    harness) instructing the agent to write its plan. Capped at one re-invoke
    per PR so a mis-emitting reporter can't loop.

    Usage: dispatch-plan-context <pr_number> <context>
    """
    if len(args) < 2:
        print("Usage: dispatch-plan-context <pr_number> <context>", file=sys.stderr)
        sys.exit(1)

    pr_num, context = args[0], args[1]
    config = load_config()

    branch = gh("pr", "view", pr_num, "--json", "headRefName", "--jq", ".headRefName", check=False)
    if not branch:
        print(f"PR #{pr_num} not found")
        return
    agent = branch_to_agent(config.agents, branch)
    if not agent:
        print(f"Not an agent branch: {branch}")
        return

    if count_pr_comments(pr_num, "plan-context") >= 1:
        print(f"plan-context already dispatched for PR #{pr_num}; skipping")
        return

    pat_token = os.environ.get("PAT_TOKEN", "")
    if not agent.mention or not pat_token:
        print("Cannot dispatch plan-context (no mention or no PAT_TOKEN)")
        return

    plan_dir = config.plan.dir
    issue_num = extract_issue_from_pr(pr_num) or "?"
    body = (
        f"{agent.mention} [plan-context] {context}\n\n"
        f"Using the prototype results above as context, now write your implementation "
        f"plan to `{plan_dir}/issue-{issue_num}/plan.md` and open a 'Plan: <summary>' PR. "
        f"Do NOT implement the full feature yet. Commit and push."
    )
    subprocess.run(
        ["gh", "pr", "comment", pr_num, "--body", body],
        env={**os.environ, "GH_TOKEN": pat_token},
        check=False,
    )
    print(f"Dispatched plan-context re-invoke for {agent.name} PR #{pr_num}")


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

    summary = _summarize_failed_attempt(pr_num, failure_context, config.llm)

    close_and_cleanup_pr(pr_num, f"Closing for ralph restart (attempt {attempt}/{max_loops}).")

    restart_body = (
        f"[ralph-restart {attempt}] {agent.name} — restarting after "
        f"{config.checks.max_repair_attempts} failed repairs on PR #{pr_num}."
    )
    if summary:
        restart_body += f"\n\nPrevious approach: {summary}"
    gh("issue", "comment", issue_num, "--body", restart_body)

    dispatch_agent(issue_num, agent.harness)
    print(f"Ralph restart {attempt}/{max_loops} for {agent.name} on issue #{issue_num}")


def _summarize_failed_attempt(pr_num: str, failure_context: str, llm_config: LLMConfig) -> str:
    """Generate a 2-3 sentence summary of a failed PR approach."""
    diff = gh("pr", "diff", pr_num, check=False)
    if not diff.strip():
        return ""

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


def _set_impl_comment_preview(
    repo: str, issue_num: str, agent_name: str, pr_number: str, preview_url: str
) -> None:
    """Finalize the '## {agent} Implementation' comment's Preview line in place.

    The implementation comment is first posted with a placeholder (e.g.
    'Deploying...'); this rewrites its Preview line to the real URL so the
    comment closes the loop alongside the status table. No-op if not found.
    """
    if not (agent_name and pr_number and preview_url):
        return
    comments_json = gh(
        "api",
        f"repos/{repo}/issues/{issue_num}/comments",
        "--jq",
        f'[.[] | select(.body | startswith("## {agent_name} Implementation")) '
        f'| select(.body | contains("#{pr_number}"))] | last | {{id, body}}',
        check=False,
    )
    if not comments_json or comments_json == "null":
        return
    comment = json.loads(comments_json)
    new_body = re.sub(
        r"- \*\*Preview\*\*:.*", f"- **Preview**: {preview_url}", comment["body"]
    )
    if new_body != comment["body"]:
        gh(
            "api", "-X", "PATCH",
            f"repos/{repo}/issues/comments/{comment['id']}",
            "-f", f"body={new_body}",
            check=False,
        )


IMPL_STATUS_HEADER = "## Agent Implementations"
PLAN_STATUS_HEADER = "## Agent Plans"


def _apply_status_update(
    issue_num: str,
    agent_label: str,
    status: str,
    pr_number: str,
    preview_url: str,
    run_url: str,
    header: str = IMPL_STATUS_HEADER,
    preview_label: str = "Preview",
) -> str | None:
    """Patch one agent's row in the ``header`` status comment on an issue.

    Returns the resolved display name (or None if the comment/agent is missing).
    Shared by the implement-phase status table and the plan-phase table.
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("GITHUB_REPOSITORY not set", file=sys.stderr)
        sys.exit(1)

    agents = load_agents()
    agent = label_to_agent(agents, agent_label)
    if not agent:
        print(f"Unknown agent label: {agent_label}", file=sys.stderr)
        sys.exit(1)

    comments_json = gh(
        "api",
        f"repos/{repo}/issues/{issue_num}/comments",
        "--jq",
        f'[.[] | select(.body | startswith("{header}")) | {{id, body}}] | last',
        check=False,
    )

    if not comments_json or comments_json == "null":
        print(f"No status comment found for header '{header}'")
        return None

    comment = json.loads(comments_json)
    comment_id = comment["id"]
    body = comment["body"]

    mapping = read_alias_mapping(issue_num)
    if mapping:
        reverse = reverse_alias_mapping(mapping)
        agent_name = reverse.get(agent.label, agent.name)
    else:
        agent_name = agent.name

    new_body = update_status_row(body, agent_name, status, pr_number, preview_url, run_url, preview_label)
    new_body = check_all_done(new_body)

    gh(
        "api",
        "-X",
        "PATCH",
        f"repos/{repo}/issues/comments/{comment_id}",
        "-f",
        f"body={new_body}",
    )
    print(f"Updated {agent_name} [{header}]: status={status}, pr={pr_number}, link={preview_url}, run={run_url}")
    return agent_name


def cmd_update_status(args: list[str]) -> None:
    """Update an agent's row in the status comment on an issue."""
    if len(args) < 3:
        print("Usage: update-status <issue_number> <agent_label> <status> [pr] [preview] [run]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    agent_label = args[1]
    status = args[2]
    pr_number = args[3] if len(args) > 3 else ""
    preview_url = args[4] if len(args) > 4 else ""
    run_url = args[5] if len(args) > 5 else ""

    agent_name = _apply_status_update(issue_num, agent_label, status, pr_number, preview_url, run_url)

    # Close the loop on the per-agent implementation comment too (not just the
    # status table), so its "Preview" line reflects the final URL.
    if agent_name and status == "preview" and preview_url:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        _set_impl_comment_preview(repo, issue_num, agent_name, pr_number, preview_url)


def cmd_post_plan_result(args: list[str]) -> None:
    """Plan-phase analog of post-agent-result.

    Updates the ``## Agent Plans`` table row (status, plan PR, direct plan-file
    link) and posts a per-agent plan comment linking the plan in Markdown-preview
    mode on the agent's branch.

    Usage: post-plan-result <issue> <agent_label> [pr] [branch] [run_url]
    """
    if len(args) < 2:
        print("Usage: post-plan-result <issue> <agent_label> [pr] [branch] [run_url]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    agent_label = args[1]
    pr_num = args[2] if len(args) > 2 else ""
    branch = args[3] if len(args) > 3 else ""
    run_url = args[4] if len(args) > 4 else ""

    plan_dir = config_parser.get_plan_config(config_parser.load_config(CONFIG_PATH))["dir"]
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not pr_num:
        _apply_status_update(
            issue_num, agent_label, "no-changes", "", "", run_url, header=PLAN_STATUS_HEADER, preview_label="plan"
        )
        return

    url = plan_file_url(repo, branch, issue_num, plan_dir) if (repo and branch) else ""
    agent_name = _apply_status_update(
        issue_num, agent_label, "done", pr_num, url, run_url, header=PLAN_STATUS_HEADER, preview_label="plan"
    )

    # Per-agent plan comment with the direct MD-preview link (the user wants
    # issue comments to link straight to each plan file on its branch).
    if agent_name:
        display = agent_name
    else:
        a = label_to_agent(load_agents(), agent_label)
        display = a.name if a else "Agent"
    body = f"""## {display} Plan
- **Plan PR**: #{pr_num}
- **Plan**: {f'[{plan_dir}/issue-{issue_num}/plan.md]({url})' if url else '_pending_'}"""
    if run_url:
        body += f"\n- **Run**: [View logs]({run_url})"
    body += "\n\n---\n*Review the competing plans, then select one with `/merge-plan <alias> [feedback]`.*"
    gh("issue", "comment", issue_num, "--body", body)


def _find_agent_pr(issue_num: str, agent: AgentConfig) -> str:
    """Return the agent's latest PR number for an issue (branch-prefix or label), or ''."""
    jq_filter = (
        f"[.[] | select("
        f'(.body | test("Implements #{issue_num}\\\\b")) and '
        f'((.headRefName | startswith("{agent.branch_prefix}")) or '
        f'(.labels | map(.name) | index("{agent.label}"))))] | last | .number // empty'
    )
    return gh("pr", "list", "--json", "number,body,headRefName,labels", "--jq", jq_filter, check=False).strip()


def _write_stats_marker(issue_num: str, key: str, value: str) -> None:
    """Post a machine-readable stats comment the LBM Hub keys off (plan/code winner)."""
    payload = json.dumps({key: value, "issue": issue_num})
    gh("issue", "comment", issue_num, "--body", f"{PLAN_STATS_MARKER} {payload} -->")


def cmd_merge_plan(args: list[str]) -> None:
    """Select a winning plan (mirrors /merge).

    - No feedback -> FINALIZE: merge the plan PR to the default branch, close the
      other plan PRs, record the plan-winner stat, flip the issue to the implement
      phase, and re-apply the ready label to re-dispatch every agent to implement.
    - With feedback -> REVISE (bounded by [plan].feedback_revs): post the feedback
      to the agent's plan PR so it revises plan.md in place. The maintainer then
      runs /merge-plan <alias> (no feedback) to finalize.

    Usage: merge-plan <issue> <agent_label> [feedback...]
    """
    if len(args) < 2:
        print("Usage: merge-plan <issue> <agent_label> [feedback...]", file=sys.stderr)
        sys.exit(1)

    issue_num = args[0]
    agent_label = args[1]
    feedback = " ".join(args[2:]).strip()

    config = load_config()
    raw = config_parser.load_config(CONFIG_PATH)
    plan_cfg = config_parser.get_plan_config(raw)
    agent = label_to_agent(config.agents, agent_label)
    if not agent:
        gh("issue", "comment", issue_num, "--body", f"Unknown agent label '{agent_label}'.")
        return

    display = agent.name
    mapping = read_alias_mapping(issue_num)
    if mapping:
        display = reverse_alias_mapping(mapping).get(agent_label, agent.name)

    pr_num = _find_agent_pr(issue_num, agent)
    if not pr_num:
        gh("issue", "comment", issue_num, "--body", f"No {display} plan PR found for this issue.")
        return

    # --- Feedback revision path ------------------------------------------
    if feedback:
        revs = count_pr_comments(pr_num, "plan-rev")
        if not _plan_rev_allowed(revs, plan_cfg["feedback_revs"]):
            gh(
                "issue", "comment", issue_num,
                "--body", f"Plan feedback limit reached ({plan_cfg['feedback_revs']}). "
                f"Run `/merge-plan {display}` (no feedback) to finalize.",
            )
            return
        pat_token = os.environ.get("PAT_TOKEN", "")
        rules = (
            "IMPORTANT: You are fully autonomous. Revise ONLY the plan file per this "
            "feedback; do not implement. Commit and push."
        )
        rev_body = f"{agent.mention} [plan-rev] {feedback}\n\n---\n{rules}"
        if pat_token:
            subprocess.run(["gh", "pr", "comment", pr_num, "--body", rev_body],
                           env={**os.environ, "GH_TOKEN": pat_token}, check=False)
        else:
            gh("pr", "comment", pr_num, "--body", rev_body, check=False)
        # Mark the PR for auto-finalize: once the agent pushes its revision and CI
        # passes, the CI hook (plan-finalize job) merges it — no manual step. The
        # label must exist in the repo or --add-label silently no-ops.
        gh("label", "create", config_parser.PLAN_FINALIZE_LABEL,
           "--color", "0E8A16", "--description", "LBM: selected plan, auto-finalize after revision", "--force", check=False)
        gh("pr", "edit", pr_num, "--add-label", config_parser.PLAN_FINALIZE_LABEL, check=False)
        gh("issue", "comment", issue_num, "--body",
           f"Feedback sent to {display} (PR #{pr_num}). It will **auto-merge** once {display} pushes the revision and CI passes — no further command needed.")
        return

    # --- Finalize path ---------------------------------------------------
    gh("pr", "ready", pr_num, check=False)
    gh("pr", "merge", pr_num, "--squash", "--delete-branch", check=False)
    # Verify merge succeeded (surface conflicts rather than silently proceeding).
    state = gh("pr", "view", pr_num, "--json", "state", "--jq", ".state", check=False)
    if state and state != "MERGED":
        gh("issue", "comment", issue_num, "--body",
           f"Failed to merge {display} plan PR #{pr_num} (state={state}). Check for conflicts.")
        return

    close_agent_prs(issue_num, f"Plan not selected — {display} (PR #{pr_num}) was chosen.", exclude_pr=pr_num)
    _write_stats_marker(issue_num, "plan_winner", agent_label)

    # Flip to implement phase (durable label), then re-apply the ready label to
    # re-fire issues.labeled -> both claude (label_trigger) and the router
    # (which dispatches codex/openhands) re-run, now in the implement phase.
    ready_label = raw.get("lbm", {}).get("ready_label", "ready-for-dev")
    # gh issue edit --add-label silently no-ops for a label that doesn't exist
    # in the repo, so ensure it exists first (same guard used for agent labels).
    gh("label", "create", config_parser.PHASE_IMPLEMENT_LABEL,
       "--color", "5319E7", "--description", "LBM: implement phase active", "--force", check=False)
    gh("issue", "edit", issue_num, "--add-label", config_parser.PHASE_IMPLEMENT_LABEL, check=False)
    gh("issue", "edit", issue_num, "--remove-label", ready_label, check=False)
    gh("issue", "edit", issue_num, "--add-label", ready_label, check=False)

    gh("issue", "comment", issue_num, "--body",
       f"✅ Selected {display}'s plan (merged PR #{pr_num} to `{plan_cfg['dir']}/issue-{issue_num}/plan.md`). "
       f"Agents are now implementing against it — the `## Agent Implementations` table will track code PRs.")


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

    # Compact the diff: keep file headers, hunk markers, and changed lines only.
    # Strip unchanged context lines to reduce size by ~50%.
    # Also exclude files with more than MAX_FILE_DIFF_LINES changed lines.
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
        # Skip unchanged context lines

    if current_file:
        if changed_lines <= MAX_FILE_DIFF_LINES:
            filtered_chunks.append("".join(current_chunk))
        else:
            excluded_files.append(f"{current_file} ({changed_lines} lines changed)")

    diff = "".join(filtered_chunks)

    # Prepend commit messages as a compact overview
    commits = gh(
        "pr",
        "view",
        pr_number,
        "--json",
        "commits",
        "--jq",
        '.commits[] | "- " + .messageHeadline',
        check=False,
    )
    if commits:
        diff = f"## Commits\n{commits}\n\n{diff}"

    # Fetch issue body if issue number provided
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


def cmd_aliases(args: list[str]) -> None:
    """Alias operations. Usage: aliases <subcommand> [args...]

    Subcommands:
      generate               — Generate aliases for all configured agents (JSON output)
      resolve <issue> <text> — Resolve an alias from issue mapping to agent label
      read <issue>           — Read existing alias mapping from issue (JSON output)
    """
    if len(args) < 1:
        print("Usage: aliases <generate|resolve|read> [args...]", file=sys.stderr)
        sys.exit(1)

    subcmd = args[0]

    if subcmd == "generate":
        agents = load_agents()
        mapping = generate_aliases(agents)
        print(json.dumps(mapping))

    elif subcmd == "resolve":
        if len(args) < 3:
            print("Usage: aliases resolve <issue_number> <alias text>", file=sys.stderr)
            sys.exit(1)
        issue_num = args[1]
        alias_text = " ".join(args[2:])
        mapping = read_alias_mapping(issue_num)
        if not mapping:
            print(f"No alias mapping found on issue #{issue_num}", file=sys.stderr)
            sys.exit(1)
        agents = load_agents()
        agent = alias_to_agent(agents, alias_text, mapping)
        if agent:
            print(agent.label)
        else:
            print(f"Unknown alias: {alias_text}", file=sys.stderr)
            sys.exit(1)

    elif subcmd == "read":
        if len(args) < 2:
            print("Usage: aliases read <issue_number>", file=sys.stderr)
            sys.exit(1)
        issue_num = args[1]
        mapping = read_alias_mapping(issue_num)
        if mapping:
            print(json.dumps(mapping))
        else:
            print("{}")

    else:
        print(f"Unknown aliases subcommand: {subcmd}", file=sys.stderr)
        sys.exit(1)


def cmd_diagnostics(args: list[str]) -> None:
    """Print post-agent diagnostic info: git state, resolver output, branches."""
    agent_label = args[0] if args else ""

    print("--- Git status ---")
    print(subprocess.run(["git", "status"], capture_output=True, text=True).stdout)

    print("--- Recent commits ---")
    print(subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True).stdout)

    print("--- Diff stats ---")
    diff = subprocess.run(["git", "diff", "--stat", "HEAD"], capture_output=True, text=True)
    print(diff.stdout or "(no uncommitted changes)")

    # Agent-specific: show branches matching this agent's prefix
    if agent_label:
        agents = load_agents()
        agent = label_to_agent(agents, agent_label)
        if agent:
            prefix = agent.branch_prefix.rstrip("/")
            result = subprocess.run(["git", "branch", "-a"], capture_output=True, text=True)
            matching = [line.strip() for line in result.stdout.splitlines() if prefix in line]
            if matching:
                print(f"--- Branches matching '{prefix}' ---")
                for b in matching:
                    print(f"  {b}")

    # OpenHands-specific: resolver output summary
    oh_output = "/tmp/oh-output/output.jsonl"
    oh_log = "/tmp/oh-resolve.log"

    if os.path.exists(oh_output):
        print("--- Resolver output ---")
        try:
            with open(oh_output) as f:
                for line in f:
                    data = json.loads(line)
                    patch = data.get("git_patch", "")
                    if patch:
                        print(f"git_patch: {len(patch)} chars, {patch.count(chr(10))} lines")
        except Exception as e:
            print(f"Could not parse output.jsonl: {e}")

    if os.path.exists(oh_log):
        print("--- Resolve log (last 20 lines) ---")
        with open(oh_log) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line, end="")


def cmd_generate_config(args: list[str]) -> None:
    """Generate flat config output from lbm.toml (for debugging/validation)."""
    from dataclasses import asdict

    check_only = "--check" in args
    config_path = args[0] if args and not args[0].startswith("--") else CONFIG_PATH

    config = load_lbm_config(config_path)
    generated_json = json.dumps(asdict(config), indent=2) + "\n"

    if check_only:
        print("Config is valid.")
        print(generated_json)
    else:
        print(generated_json)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

COMMANDS = {
    "lookup": cmd_lookup,
    "close-previous-prs": cmd_close_previous_prs,
    "post-agent-result": cmd_post_agent_result,
    "post-plan-result": cmd_post_plan_result,
    "merge-plan": cmd_merge_plan,
    "close-losing-prs": cmd_close_losing_prs,
    "record-no-winner": cmd_record_no_winner,
    "dispatch-repair": cmd_dispatch_repair,
    "dispatch-plan-context": cmd_dispatch_plan_context,
    "update-status": cmd_update_status,
    "summarize-pr": cmd_summarize_pr,
    "aliases": cmd_aliases,
    "diagnostics": cmd_diagnostics,
    "generate-config": cmd_generate_config,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        print(f"Commands: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    COMMANDS[command](sys.argv[2:])


if __name__ == "__main__":
    main()
