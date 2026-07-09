# LBM Plan Step — Prototype Iteration — Design

**Date:** 2026-07-09
**Repo:** `lbm-poc` (generic mechanism); `chess-rl` (reporter emits the marker)
**Status:** Design for review — pre-implementation. Fast-follow A of
`docs/superpowers/specs/2026-07-09-lbm-plan-step-design.md`. Builds on the shipped
core plan step (see that spec + `2026-07-09-lbm-plan-step.md`).

## Summary

When enabled, an agent may run a **prototype** before writing its plan: open a small
prototype PR, let the existing auto-run produce a report, then write `plan.md` informed
by that evidence. Chosen mechanism (per user): **prototype PR → auto-run → report →
plan.** It is built entirely from existing primitives — the agent-label-gated auto-run
and a comment-marker watcher (the dual of the repair and preview watchers already
shipped). No GPU is forced: the agent decides per-issue whether a prototype is warranted.

## Configuration

```toml
[plan]
enabled = true
prototype = true                                  # allow a prototype before planning
# marker a report comment carries to re-invoke the agent to write its plan:
plan_context_comment_marker = "<!-- lbm-plan-context -->"   # default; empty disables re-invoke
```

- `PlanConfig.prototype` already exists (core). Add `plan_context_comment_marker`
  (default `<!-- lbm-plan-context -->`) to `PlanConfig` + `config_parser.get_plan_config`.

## Flow (plan phase, `prototype = true`)

1. **Prototype prompt.** In the plan phase, `build_task_prompt(issue, "plan", plan_dir)`
   gains a prototype-aware variant (selected when `prototype` is on): it tells the agent
   it *may* first open a **prototype PR** (`Prototype: <title>`, minimal experiment code)
   to gather evidence, wait for the auto-run report, and then write the plan; or, if a
   prototype isn't warranted, write `plan.md` directly. Agent-driven → no forced GPU.
2. **Auto-run (existing, unchanged).** The prototype PR carries the agent label, so the
   repo's existing auto-run (chess-rl `experiment.yml` `auto-run` job) fires on CI
   success and posts a report comment.
3. **Report → re-invoke.** The report comment (on the prototype PR) carries
   `plan_context_comment_marker`. A new LBM watcher `plan-context-comment` (in
   `_comments.yml`, dual of `repair-comment`/`preview-comment`) matches the marker,
   resolves the agent + linked issue, and re-invokes that agent — posting a
   `@mention [plan-context] <report>` comment (via the LBM/PAT token so it triggers the
   harness) instructing it to write `lbm-plans/issue-<N>/plan.md` on its plan branch
   using the prototype results as context.
4. **Plan PR.** The agent writes `plan.md` (the canonical plan path) → the normal
   `## Agent Plans` row + `/merge-plan` selection proceed exactly as in the core.

The prototype PR is left as evidence (linked from the plan) and is closed by the
iteration cleanup / not merged (it is not a plan PR and not an implementation PR).

## Design decisions (resolved defaults)

- **Prototype PR ≠ plan PR.** The prototype PR holds experiment code (so auto-run can run
  it); the plan PR holds `plan.md`. Keeping them separate reuses auto-run untouched and
  keeps the `## Agent Plans` table (which parses plan.md links) clean.
- **Opt-in is repo-wide (`prototype = true`), prototype-per-issue is agent-judged.** The
  prompt invites but does not force a prototype, so trivial plans don't burn GPU.
- **Re-invoke reuses the watcher+mention pattern**, not a new dispatch route — consistent
  with repair/preview and race-free (marker on a created comment via a workflow-triggering
  token).
- **Marker separation.** A prototype report uses `plan_context_comment_marker` (re-invoke
  to plan), distinct from `preview_comment_marker` (implement-phase preview) and
  `repair_comment_triggers` (failure repair). chess-rl's reporter selects the marker by
  the PR's phase/kind: prototype PR in plan phase → `lbm-plan-context`; implement PR →
  `lbm-preview`/`lbm-repair` as today.

## Changes by file

| Repo | File | Change |
|---|---|---|
| lbm-poc | `scripts/models.py` | `PlanConfig.plan_context_comment_marker` |
| lbm-poc | `scripts/config_parser.py` | surface it in `get_plan_config`; `matches`/marker helper reuse |
| lbm-poc | `scripts/agent_ops.py` | prototype-aware branch in `build_task_prompt`; `cmd_dispatch_plan_context` (resolve agent+issue from PR, post `@mention [plan-context] <ctx>` via PAT); register command |
| lbm-poc | `.github/workflows/_comments.yml` | `plan-context-comment` watcher (gate mirrors `preview-comment`; ignores `[plan-context]` echoes) |
| lbm-poc | `cli/templates/lbm.toml.j2` | `plan_context_comment_marker` under `[plan]` (commented) |
| chess-rl | `experiment.yml` + reporter | in plan phase on a prototype PR, append `<!-- lbm-plan-context -->` to the report (instead of `lbm-preview`); detect "prototype PR" (title `Prototype:` or plan-phase label) |
| chess-rl | `AGENTS.md` | repo-specific note on what a *prototype* experiment should measure (kept repo-specific per [[feedback_repo_agents_md_scope]]) |

## Guardrails

- **Loop bound.** The re-invoke fires once per report; the agent then writes the plan (it
  does not open another prototype). To be safe, cap plan-context re-invokes per issue
  (reuse `count_pr_comments`/`count_issue_comments` with a `[plan-context]` marker, cap 1)
  so a mis-emitting reporter can't loop.
- **Cost visibility.** `log`/comment when a prototype auto-run is triggered so the GPU
  spend is legible.
- **Disabled by default.** `prototype` defaults false; `plan_context_comment_marker` empty
  disables the re-invoke even if a report carries it.

## Testing / verification

- Unit: `get_plan_config` includes the new marker; `build_task_prompt` prototype variant
  mentions "prototype" + the plan path and still says don't implement the feature;
  `cmd_dispatch_plan_context` resolves agent/issue and composes the mention (mock `gh`).
- YAML validity for `_comments.yml`; template equivalence.
- E2E (GPU): chess-rl issue with `[plan] prototype=true` → agent opens a prototype PR →
  auto-run report with `lbm-plan-context` → agent re-invoked → writes plan.md → `/merge-plan`
  → implement. (Consumes GPU; run one tiny prototype.)

## Notes

- Reuses the shipped core untouched; `enabled=true, prototype=false` repos are unaffected.
- OpenHands remains unsupported in the plan phase (core limitation).
