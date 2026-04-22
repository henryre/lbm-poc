# LBM

Multi-agent dev infra for competing AI implementations on GitHub Issues.

LBM lets you configure multiple AI coding agents (Claude, Codex, OpenHands, etc.)
to compete on the same GitHub issue. Each agent creates a PR, and you pick the best one.

## How it works

Target repos install **4 workflow files** plus a **`lbm.toml`** config file:

1. `lbm-dispatch.yml` — Triggers agent runs when issues are labeled
2. `lbm-comments.yml` — Handles `/merge`, `/retry`, and other comment commands
3. `lbm-agents.yml` — Dispatches individual agent harnesses
4. `lbm-ci-hooks.yml` — Monitors CI checks and deployments, triggers repairs on failure

All heavy logic lives in reusable workflows in this repo. Target repos get thin
wrappers that call back here.

## Quick start

```bash
uv sync
uv run lbm-dev init
```

The `init` command walks you through setup: runtime, deploy platform, agents, and
LLM provider. It generates `lbm.toml` and the 4 workflow files for your repo.

## Configuration

All per-repo configuration lives in `lbm.toml` at the repo root. See
`templates/lbm.toml.j2` for the full schema.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format .
```
