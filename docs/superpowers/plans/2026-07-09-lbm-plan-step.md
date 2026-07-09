# LBM Optional Plan Step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Opt-in 2-phase LBM iterations — agents propose plans (Phase 1) → `/merge-plan` picks a winner into `lbm-plans/` → agents implement against it (Phase 2) → normal `/merge`.

**Architecture:** Phase is explicit: `[plan].enabled` config + `lbm:phase-*` labels (visible) + a threaded `phase` input (authoritative). Reuses existing alias/status-table/dispatch/merge primitives; no string-sniffing.

**Tech Stack:** Python 3.12, pytest, GitHub Actions reusable workflows, gh CLI.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-lbm-plan-step-design.md`.
- **Opt-in**: `[plan].enabled` default **false** → today's single-phase flow unchanged.
- Plan file canonical path (same on every agent branch): `lbm-plans/issue-<N>/plan.md`.
- Command `/merge-plan <alias> [feedback]` mirrors `/merge`; `feedback_revs` default 1.
- `phase` input values: `plan` | `implement` (default `implement`).
- Reuse: aliases, status-table helpers, `dispatch_agent`, PR-close, `label_to_agent`.
- lbm-poc must work without Hub; `v1` retag gated by `enabled=false` default.
- Tests: `test/` via `uv run pytest`; template equivalence via `node --test test/equivalence.test.mjs`.

---

### Task 1: `PlanConfig` + `get_plan_config` (lbm-poc)

**Files:** Modify `scripts/models.py`, `scripts/config_parser.py`; Test `test/test_plan_step.py` (new)

**Produces:** `models.PlanConfig(enabled: bool, dir: str, feedback_revs: int, prototype: bool)` + `LBMConfig.plan`; `config_parser.get_plan_config(config) -> dict`.

- [ ] **Test** (`test/test_plan_step.py`):
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import config_parser  # noqa: E402
from models import PlanConfig  # noqa: E402

def test_get_plan_config_defaults():
    assert config_parser.get_plan_config({}) == {"enabled": False, "dir": "lbm-plans", "feedback_revs": 1, "prototype": False}
    assert config_parser.get_plan_config({"plan": {"enabled": True}})["enabled"] is True

def test_planconfig_from_dict():
    p = PlanConfig.from_dict({"enabled": True, "dir": "plans", "feedback_revs": 2})
    assert (p.enabled, p.dir, p.feedback_revs, p.prototype) == (True, "plans", 2, False)
    assert PlanConfig.from_dict({}).enabled is False
```
- [ ] Run → fail. `uv run pytest test/test_plan_step.py -q`
- [ ] **Implement** — `models.py` add dataclass + `LBMConfig.plan` (parse in `LBMConfig.from_dict`: `plan=PlanConfig.from_dict(raw.get("plan", {}))`):
```python
@dataclass(frozen=True)
class PlanConfig:
    enabled: bool = False
    dir: str = "lbm-plans"
    feedback_revs: int = 1
    prototype: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PlanConfig":
        return cls(
            enabled=d.get("enabled", False),
            dir=d.get("dir", "lbm-plans"),
            feedback_revs=d.get("feedback_revs", 1),
            prototype=d.get("prototype", False),
        )
```
`config_parser.py` append:
```python
def get_plan_config(config: dict) -> dict:
    """Plan-step config (empty/absent = disabled)."""
    p = config.get("plan", {})
    return {
        "enabled": bool(p.get("enabled", False)),
        "dir": p.get("dir", "lbm-plans"),
        "feedback_revs": int(p.get("feedback_revs", 1)),
        "prototype": bool(p.get("prototype", False)),
    }
```
- [ ] Run → pass; full `uv run pytest -q` (only the known pre-existing `test_updates_row` may fail).
- [ ] Commit: `feat(plan): PlanConfig + get_plan_config`

---

### Task 2: `build_task_prompt` (lbm-poc)

**Files:** Modify `scripts/agent_ops.py`; Test `test/test_plan_step.py`

**Produces:** `agent_ops.build_task_prompt(issue_num: str, phase: str, plan_dir: str = "lbm-plans") -> str` — phase-specific instructions appended to the issue body by the harness workflows.

- [ ] **Test** (append):
```python
from agent_ops import build_task_prompt  # noqa: E402

def test_build_task_prompt_plan():
    t = build_task_prompt("7", "plan", "lbm-plans")
    assert "lbm-plans/issue-7/plan.md" in t and "Do NOT implement" in t

def test_build_task_prompt_implement():
    t = build_task_prompt("7", "implement", "lbm-plans")
    assert "lbm-plans/issue-7/plan.md" in t and "Implement" in t
```
- [ ] Run → fail.
- [ ] **Implement** (agent_ops.py):
```python
def build_task_prompt(issue_num: str, phase: str, plan_dir: str = "lbm-plans") -> str:
    """Phase-specific agent instructions (appended to the issue body)."""
    path = f"{plan_dir}/issue-{issue_num}/plan.md"
    if phase == "plan":
        return (
            f"PLAN PHASE. Propose an implementation plan for issue #{issue_num}. "
            f"Write the plan to `{path}` on your branch and open a PR titled "
            f"'Plan: ...'. Do NOT implement the feature yet. Commit and push."
        )
    return (
        f"IMPLEMENT PHASE. Implement issue #{issue_num} following the approved plan "
        f"at `{path}` (already on the default branch). Open a PR. Commit and push."
    )
```
- [ ] Run → pass. Commit: `feat(plan): build_task_prompt (plan/implement modes)`

---

### Task 3: Plan status table + plan-link (lbm-poc)

**Files:** Modify `scripts/agent_ops.py`; Test `test/test_plan_step.py`

**Produces:** `agent_ops.plan_file_url(repo, branch, issue_num, plan_dir) -> str`; `cmd_post_plan_result(args)` (`post-plan-result <issue> <label> [pr] [branch] [run_url]`) — updates a `## Agent Plans` comment row with the plan PR + a branch-blob plan link.

- [ ] **Test** (append) — pure helper only (comment I/O covered at integration):
```python
from agent_ops import plan_file_url  # noqa: E402

def test_plan_file_url():
    u = plan_file_url("o/r", "claude-x/issue-7", "7", "lbm-plans")
    assert u == "https://github.com/o/r/blob/claude-x/issue-7/lbm-plans/issue-7/plan.md"
```
- [ ] Run → fail.
- [ ] **Implement**: `plan_file_url` (pure); `cmd_post_plan_result` mirroring `cmd_post_agent_result` but writing to a `## Agent Plans` status comment (columns `Alias | Status | Plan PR | Plan | Run`), setting the **Plan** cell to `[plan](plan_file_url(...))`. Reuse `read_alias_mapping`/`update_status_row` against a `## Agent Plans` header (parameterize the header, or add `update_status_row(..., header="## Agent Plans")`). Register `post-plan-result` in the CLI dispatch dict.
- [ ] Run → pass. Commit: `feat(plan): Agent Plans status table + plan-file links`

---

### Task 4: `/merge-plan` command logic (lbm-poc)

**Files:** Modify `scripts/agent_ops.py`; Test `test/test_plan_step.py`

**Produces:** `agent_ops.cmd_merge_plan(args)` (`merge-plan <issue> <agent_label> [feedback...]`): no feedback → finalize (merge plan PR → close others → stats → re-dispatch `phase=implement`); with feedback → post feedback to plan PR (bounded by `feedback_revs`). `write_stats_marker(issue, key, value)` for plan/code winners.

- [ ] **Test** (append) — pure pieces (rev-cap counter + stats marker parse):
```python
from agent_ops import _plan_rev_allowed  # noqa: E402

def test_plan_rev_cap():
    assert _plan_rev_allowed(0, 1) is True
    assert _plan_rev_allowed(1, 1) is False
```
- [ ] Run → fail.
- [ ] **Implement**: `_plan_rev_allowed(count, cap) -> bool`; `cmd_merge_plan` reusing `label_to_agent`, plan-PR lookup (by branch prefix + issue, like `/feedback`), `gh pr merge`, `close_agent_prs`(plans), `dispatch_agent(issue, harness, phase="implement")` (extend `dispatch_agent` with an optional `phase`), `write_stats_marker`. Register `merge-plan`. (`dispatch_agent` gains `phase` param → passed as a `phase` input to `lbm-agents.yml` workflow_dispatch.)
- [ ] Run → pass. Commit: `feat(plan): merge-plan (finalize + feedback rev) + winner stats`

---

### Task 5: Workflow wiring — `phase` threading + plan dispatch + `/merge-plan` job (lbm-poc)

**Files:** Modify `.github/workflows/_agents.yml`, `_agent-claude.yml`, `_dispatch.yml`, `_comments.yml`; templates `cli/templates/lbm-dispatch.yml.j2`, `lbm-agents.yml.j2`.

- [ ] **`_agents.yml`**: add `phase` workflow_call input (default `implement`); in "Build prompt", append `build_task_prompt(issue, phase, plan_dir)`.
- [ ] **`_agent-claude.yml`**: add `phase` input (default `implement`); inject `build_task_prompt(...)` into the claude prompt.
- [ ] **`_dispatch.yml`**: if `get_plan_config(cfg)["enabled"]` and no `lbm:phase-implement` label → Phase 1: apply `lbm:phase-plan`, post `## Agent Plans` (instead of `## Agent Implementations`), dispatch agents with `phase=plan`. Else single-phase (today).
- [ ] **`_comments.yml`**: add a `merge-plan` job (mirror the `merge` job's guards/token/checkout) that runs `agent_ops.py merge-plan <issue> <resolved-label> [feedback]` (resolve alias→label via existing `aliases resolve`).
- [ ] **Templates** (`lbm-dispatch.yml.j2`, `lbm-agents.yml.j2`): thread the `phase` input through the wrappers.
- [ ] **Validate**: `python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/_agents.yml','.github/workflows/_agent-claude.yml','.github/workflows/_dispatch.yml','.github/workflows/_comments.yml']]; print('OK')"`; `npm run build && node --test test/equivalence.test.mjs`.
- [ ] Commit: `feat(plan): workflow wiring — phase threading, plan dispatch, /merge-plan`

---

### Task 6: Template `[plan]` section + equivalence (lbm-poc)

**Files:** Modify `cli/templates/lbm.toml.j2`

- [ ] Add under `[checks]`/before `[deploy]`:
```jinja
[plan]
# Opt-in 2-phase iterations: agents propose plans, a winner is merged to
# lbm-plans/, then agents implement against it. See /merge-plan.
enabled = false
```
- [ ] `npm run build && node --test test/equivalence.test.mjs` → pass.
- [ ] Commit: `feat(plan): lbm.toml template [plan] section`

---

### Task 7: chess-rl opt-in + plan guidance (chess-rl)

**Files:** Modify `/home/ubuntu/chess-rl/lbm.toml`, `AGENTS.md`

- [ ] `lbm.toml`: add `[plan]\nenabled = true`.
- [ ] `AGENTS.md`: a "Plan phase" section (what a good plan is; write to `lbm-plans/issue-<N>/plan.md`; don't implement in plan phase).
- [ ] Commit + push (main).

---

## Verification (autonomous)

1. lbm-poc `uv run pytest -q` (new plan tests pass; only pre-existing `test_updates_row` may fail).
2. All touched workflow YAML valid; template equivalence green.
3. Push lbm-poc `main`; retag `v1` (gated by `enabled=false` default → other repos unaffected).
4. E2E on chess-rl: new issue → `ready-for-dev` → verify Phase-1 plan PRs + `## Agent Plans` with plan links → `/merge-plan <alias>` → plan merged to `lbm-plans/issue-<N>/plan.md` + Phase-2 implement dispatch → `/merge <alias>`. (Agent runs cost API; keep the task tiny.)

## Fast-follows (separate plans)

- **Prototype iteration** (`[plan].prototype`): prototype PR → auto-run → report → `<!-- lbm-plan-context -->` re-invoke → plan. Dual of the repair watcher.
- **Hub** (`../lbm-hub`): plan review/select UI (posts `/merge-plan`) + plan/code merge stats, reading `## Agent Plans`/`## Agent Implementations` + `lbm:phase-*` labels + `<!-- lbm:stats … -->` marker.

## Self-Review

- **Coverage:** config (T1), prompt (T2), plan table+links (T3), merge-plan+stats (T4), wiring (T5), template (T6), chess-rl opt-in (T7). Fast-follows scoped separately. ✓
- **Placeholders:** none — concrete code/tests for the unit-testable pieces; wiring tasks specify exact files + validation.
- **Type consistency:** `get_plan_config`/`PlanConfig`, `build_task_prompt(issue,phase,plan_dir)`, `plan_file_url`, `cmd_merge_plan`/`_plan_rev_allowed`, `phase` input (`plan`|`implement`) used consistently across tasks + workflows.
