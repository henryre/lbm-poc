"""Parse lbm.toml configuration files.

The lbm.toml format:
    [build]
    runtime = "node"          # node | python | custom
    install = "npm ci"
    lint = "npm run lint"
    typecheck = "npx tsc --noEmit"
    build = "npm run build"
    verify = "npm run verify"

    [deploy]
    platform = "vercel"       # vercel | fly | railway | none

    [harnesses.claude]
    mention = "@claude"

    [harnesses.codex]
    mention = "@codex"

    [[agents]]
    harness = "claude"
    model_id = "global.anthropic.claude-opus-4-6-v1"
    model_label = "opus-4-6"

    [checks]
    required = ["Vercel", "Lint"]

    [settings]
    max_repair_attempts = 10
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

AGENT_NAME_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------------------------------------------------------------------------
# Default build commands by runtime
# ---------------------------------------------------------------------------

DEFAULT_BUILD_COMMANDS: dict[str, dict[str, str]] = {
    "node": {
        "install": "npm ci",
        "lint": "npm run lint",
        "typecheck": "npx tsc --noEmit",
        "build": "npm run build",
        "verify": "npm run verify",
    },
    "python": {
        "install": "pip install -e '.[dev]'",
        "lint": "ruff check .",
        "typecheck": "mypy .",
        "build": "",
        "verify": "",
    },
    "custom": {
        "install": "",
        "lint": "",
        "typecheck": "",
        "build": "",
        "verify": "",
    },
}

# ---------------------------------------------------------------------------
# Allowed tools by runtime (for Claude harness)
# ---------------------------------------------------------------------------

_GIT_TOOLS = [
    "git status",
    "git diff",
    "git log",
    "git stash",
    "git revert",
    "git checkout",
    "git add",
    "git commit",
    "git rm",
]

RUNTIME_ALLOWED_TOOLS: dict[str, list[str]] = {
    "node": [
        "npm run lint",
        "npm run build",
        "npm run verify",
        "npx tsc",
        "npx prisma",
        "npm ci",
        *_GIT_TOOLS,
    ],
    "python": [
        "pip",
        "pytest",
        "ruff",
        "mypy",
        *_GIT_TOOLS,
    ],
    "custom": [
        *_GIT_TOOLS,
    ],
}

# Non-Bash tools that Claude Code needs for file operations.
# These are passed as bare names (not wrapped in Bash(...)).
CORE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "LS",
]

# ---------------------------------------------------------------------------
# Default patch filter patterns by runtime (for OpenHands patch application)
# ---------------------------------------------------------------------------

_COMMON_PATCH_FILTERS = [
    "__pycache__/",
    ".DS_Store",
]

RUNTIME_PATCH_FILTERS: dict[str, list[str]] = {
    "node": [
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "node_modules/",
        ".next/",
        *_COMMON_PATCH_FILTERS,
    ],
    "python": [
        "poetry.lock",
        "uv.lock",
        ".venv/",
        "*.egg-info/",
        *_COMMON_PATCH_FILTERS,
    ],
    "custom": [
        *_COMMON_PATCH_FILTERS,
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Read and parse a lbm.toml file, returning the raw dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_build_commands(config: dict) -> dict:
    """Extract build commands from config, filling defaults from runtime."""
    build = config.get("build", {})
    runtime = build.get("runtime", "custom")
    defaults = DEFAULT_BUILD_COMMANDS.get(runtime, DEFAULT_BUILD_COMMANDS["custom"])

    return {
        "runtime": runtime,
        "install": build.get("install", defaults["install"]),
        "lint": build.get("lint", defaults["lint"]),
        "typecheck": build.get("typecheck", defaults["typecheck"]),
        "build": build.get("build", defaults["build"]),
        "verify": build.get("verify", defaults["verify"]),
    }


def get_agents(config: dict) -> list[dict]:
    """Return agent list with all fields resolved (derived defaults + overrides).

    Each agent dict contains:
        label, harness, model_id, model_label, branch_prefix, name, mention
    """
    harnesses = config.get("harnesses", {})
    agents_raw = config.get("agents", [])

    agents: list[dict] = []
    seen_prefixes: set[str] = set()

    for i, entry in enumerate(agents_raw):
        harness = entry["harness"]
        model_id = entry["model_id"]
        model_label = entry["model_label"]

        if harness not in harnesses:
            raise ValueError(f"Harness '{harness}' not defined in [harnesses]. Available: {list(harnesses.keys())}")

        default_label = f"agent:{harness}-{model_label}"
        default_prefix = f"{harness}-{model_label}/"

        label = entry.get("override_label", default_label)
        branch_prefix = entry.get("override_branch_prefix", default_prefix)

        if branch_prefix in seen_prefixes:
            raise ValueError(f"Duplicate branch_prefix '{branch_prefix}' -- each agent entry must have a unique prefix")
        seen_prefixes.add(branch_prefix)

        name_letter = AGENT_NAME_LETTERS[i] if i < len(AGENT_NAME_LETTERS) else str(i + 1)
        name = f"Agent {name_letter}"

        agents.append(
            {
                "label": label,
                "harness": harness,
                "model_id": model_id,
                "model_label": model_label,
                "branch_prefix": branch_prefix,
                "name": name,
                "mention": harnesses[harness].get("mention", ""),
                # Optional freeform args appended verbatim to the harness CLI invocation
                # (e.g. claude_args -> claude-code-action `claude_args`). Lets a repo pass
                # model/API-specific flags from lbm.toml without an lbm-poc code change.
                "claude_args": entry.get("claude_args", ""),
            }
        )

    return agents


def get_check_names(config: dict) -> list[str]:
    """Return required CI check names."""
    checks = config.get("checks", {})
    return checks.get("required", [])


def get_deploy_platform(config: dict) -> str:
    """Return the deploy platform name (e.g. 'vercel', 'fly', 'none')."""
    deploy = config.get("deploy", {})
    return deploy.get("platform", "none")


def get_deploy_config(config: dict) -> dict:
    """Return full deploy configuration with defaults."""
    deploy = config.get("deploy", {})
    return {
        "platform": deploy.get("platform", "none"),
        "preview_env": deploy.get("preview_env", "Preview"),
        "app_prefix": deploy.get("app_prefix", ""),
        "region": deploy.get("region", "iad"),
        "registry": deploy.get("registry", "ghcr"),
    }


def is_active_deploy_platform(config: dict) -> bool:
    """Return True if the platform requires LBM to orchestrate deploys."""
    return get_deploy_platform(config) in ("fly", "railway")


def derive_allowed_tools(config: dict, agent: dict) -> list[str]:
    """Generate Claude allowed-tools list based on build.runtime.

    If the agent dict has an 'allowed_tools' key, use that instead.

    Returns a list of tool strings ready for --allowedTools:
      - Bash commands are wrapped as 'Bash(cmd:*)'
      - Core tools (Read, Write, etc.) are bare names
    """
    if "allowed_tools" in agent:
        return agent["allowed_tools"]

    build = config.get("build", {})
    runtime = build.get("runtime", "custom")
    bash_tools = RUNTIME_ALLOWED_TOOLS.get(runtime, RUNTIME_ALLOWED_TOOLS["custom"])

    # Format: Bash commands as 'Bash(cmd:*)', core tools as bare names
    formatted = [f"Bash({t}:*)" for t in bash_tools]
    formatted.extend(CORE_TOOLS)
    return formatted


def get_patch_filters(config: dict) -> list[str]:
    """Return list of file patterns to strip from OpenHands patches.

    Uses [build].patch_filter if set, otherwise defaults by runtime.
    """
    build = config.get("build", {})
    if "patch_filter" in build:
        return build["patch_filter"]
    runtime = build.get("runtime", "custom")
    return RUNTIME_PATCH_FILTERS.get(runtime, RUNTIME_PATCH_FILTERS["custom"])


def derive_repair_instructions(config: dict) -> str:
    """Generate repair prompt text from build commands.

    Returns a multi-line string telling an agent how to verify its fix.
    """
    cmds = get_build_commands(config)
    steps: list[str] = []
    step_num = 1

    if cmds["install"]:
        steps.append(f"{step_num}. `{cmds['install']}` (install deps)")
        step_num += 1
    if cmds["lint"]:
        steps.append(f"{step_num}. `{cmds['lint']}` (lint)")
        step_num += 1
    if cmds["typecheck"]:
        steps.append(f"{step_num}. `{cmds['typecheck']}` (typecheck)")
        step_num += 1
    if cmds["build"]:
        steps.append(f"{step_num}. `{cmds['build']}` (build)")
        step_num += 1

    if not steps:
        return "Verify your changes compile and pass any project-specific checks before committing."

    steps_text = "\n".join(steps)
    return (
        "Fix ALL errors listed above -- there may be multiple issues across lint, typecheck, and build.\n"
        "Before committing, run the full CI check locally:\n"
        f"{steps_text}\n\n"
        "Only commit and push when ALL steps pass."
    )


def get_repair_comment_triggers(config: dict) -> list[str]:
    """Comment substrings that trigger a repair (empty list = disabled)."""
    triggers = config.get("checks", {}).get("repair_comment_triggers", [])
    return triggers if isinstance(triggers, list) else []


def matches_repair_trigger(body: str, triggers: list[str]) -> bool:
    """True if ``body`` contains any configured trigger substring."""
    text = body or ""
    return any(t and t in text for t in triggers)


def strip_repair_triggers(body: str, triggers: list[str]) -> str:
    """Remove all trigger substrings from ``body`` (so dispatched context can't re-match)."""
    text = body or ""
    for t in triggers:
        if t:
            text = text.replace(t, "")
    return text.strip()


def get_preview_comment_marker(config: dict) -> str:
    """Marker whose presence in a PR comment sets that comment as the agent's
    preview link in the status table (empty = disabled)."""
    marker = config.get("checks", {}).get("preview_comment_marker", "")
    return marker if isinstance(marker, str) else ""


def get_blocked_comment_marker(config: dict) -> str:
    """Marker whose presence in a PR comment flags the agent as ⛔ needs-human in
    the status table — for environmental/infra failures the repair loop can't fix.
    Distinct from repair_comment_triggers, so a report that carries this marker
    (instead of a repair trigger) will NOT dispatch a pointless repair. Empty =
    disabled."""
    marker = config.get("checks", {}).get("blocked_comment_marker", "")
    return marker if isinstance(marker, str) else ""


def get_plan_config(config: dict) -> dict:
    """Plan-step config (absent/empty = disabled -> today's single-phase flow)."""
    p = config.get("plan", {})
    return {
        "enabled": bool(p.get("enabled", False)),
        "dir": p.get("dir", "lbm-plans"),
        "feedback_revs": int(p.get("feedback_revs", 1)),
        "prototype": bool(p.get("prototype", False)),
        "plan_context_comment_marker": p.get(
            "plan_context_comment_marker", "<!-- lbm-plan-context -->"
        ),
    }


# LBM-managed label that flips an iteration from the plan phase to the
# implement phase (applied by /merge-plan finalize). Phase is derived from
# durable issue labels so every harness resolves it identically and races
# with the triggering event are impossible.
PHASE_IMPLEMENT_LABEL = "lbm:phase-implement"
PHASE_PLAN_LABEL = "lbm:phase-plan"
# Marks a plan PR that a maintainer selected WITH feedback: once the agent
# pushes its one revision (CI passes), the CI hook auto-finalizes the merge.
PLAN_FINALIZE_LABEL = "lbm:plan-finalize"


def resolve_phase(config: dict, labels: list[str]) -> str:
    """Return the current iteration phase: 'plan' or 'implement'.

    Rule: if the plan step is enabled and the issue has not yet been flipped to
    implement (no PHASE_IMPLEMENT_LABEL), we are in the plan phase. Otherwise
    (plan disabled, or already flipped) it's the implement phase — today's flow.
    """
    if get_plan_config(config)["enabled"] and PHASE_IMPLEMENT_LABEL not in (labels or []):
        return "plan"
    return "implement"
