# LBM Optional Plan Step — Design

**Date:** 2026-07-09
**Repo:** `lbm-poc` (feature); `lbm-hub` (Hub conveniences, fast-follow); consumed by `chess-rl`
**Status:** Approved design (decisions made autonomously per user steer) — pre-implementation

## Summary

An **opt-in two-phase iteration**: before agents implement, they each **propose a
plan**; a human (or Hub) picks the winning plan (optionally with one feedback
revision); the winner is **merged into a standardized `lbm-plans/` directory**; then
agents implement against that plan and the normal LBM flow (competing code PRs →
`/merge`) proceeds. Everything lives under the **same issue** (1:1). Works fully via
comment commands + labels (no Hub required); Hub adds review/select UI + stats.

## Goals

1. **Opt-in** via `lbm.toml [plan].enabled` (default off → today's behavior unchanged).
2. **Phase 1 (Plan):** each agent opens a plan PR adding `lbm-plans/issue-<N>/plan.md`
   on its branch. Issue comments link **directly to each plan file in MD-preview on the
   agent's branch**.
3. **Selection:** `/merge-plan <alias> [feedback]` (mirrors `/merge`). Optional **one**
   feedback revision before finalize. Finalize merges the winning plan to
   `lbm-plans/issue-<N>/plan.md` on the default branch.
4. **Phase 2 (Implement):** agents are re-dispatched to implement against the merged
   plan; normal competing-PR flow + `/merge <alias>` picks the winning code PR.
5. **Same issue** throughout (1:1 issue↔iteration).
6. **No fragile matching** — phase is explicit config + labels + a threaded `phase`
   input, not string-sniffing.

## Non-Goals (this pass — fast-follows)

- **Prototype iteration** (agent runs a prototype → report → writes plan with that
  context): mechanism decided below, built as a fast-follow.
- **LBM Hub UI** (`../lbm-hub`): plan review/select + plan/code merge stats — fast-follow;
  core must work without it.

---

## Configuration — `lbm.toml [plan]`

```toml
[plan]
enabled = true          # opt-in; default false (absent = today's single-phase flow)
dir = "lbm-plans"       # directory plans live under (default)
feedback_revs = 1       # max plan feedback revisions before finalize (default 1)
prototype = false       # fast-follow: allow a prototype run before planning
```

- `models.py`: add `PlanConfig(enabled=False, dir="lbm-plans", feedback_revs=1, prototype=False)`
  + `LBMConfig.plan`.
- `config_parser.py`: `get_plan_config(config) -> dict` (dict-based, for workflows).

---

## Phase model (labels + threaded input, 1:1 issue)

Two LBM-managed labels make the phase **visible** (for humans/Hub); a threaded
`phase` **input** is authoritative for each agent run (so nothing sniffs strings):

- `lbm:phase-plan` — Phase 1 active.
- `lbm:phase-implement` — Phase 2 active.

The dispatcher sets the label and passes `phase` to the agent workflows. When
`[plan].enabled` is false, no phase labels are used and dispatch is single-phase
(today's behavior).

---

## Phase 1 — Plan

**Router (`_dispatch.yml`, on `ready_label`, `[plan].enabled`):**
1. Generate aliases (reuse existing system) and post a **`## Agent Plans`** status
   comment: `| Alias | Status | Plan PR | Plan | Run |`.
2. Apply `lbm:phase-plan`.
3. Dispatch every agent with **`phase=plan`** (claude via the wrapper's claude job;
   codex/openhands via `lbm-agents.yml` workflow_dispatch — both gain a `phase` input).

**Each agent (`phase=plan`)** gets a plan-mode prompt (see "Prompt mode threading"):
> "Propose an implementation plan for issue #N. Write it to
> `lbm-plans/issue-<N>/plan.md` on your branch and open a PR titled `Plan: <title>`.
> Do NOT implement the feature yet. Follow the plan guidance in AGENTS.md."

Every agent writes the **same canonical path** `lbm-plans/issue-<N>/plan.md` on its
own branch — so finalize (merging the winner) lands the plan at that exact path with
no rename, and per-agent plan links differ only by branch.

**`post-agent-result` (plan phase):** update the `## Agent Plans` row —
- `Plan PR` → `#<pr>`
- **`Plan`** → a direct MD-preview link to the file on the agent's branch:
  `https://github.com/<owner>/<repo>/blob/<branch>/lbm-plans/issue-<N>/plan.md`
  (renders the markdown on GitHub — the user's "MD preview mode on respective branches").

---

## Selection — `/merge-plan <alias> [feedback]` (mirrors `/merge`)

New job in `_comments.yml`, gated like `/merge` (maintainer/bot):

- **`/merge-plan <alias>` (no feedback) → FINALIZE:**
  1. Resolve alias → agent → its plan PR.
  2. Merge that plan PR (plan.md → default branch at `lbm-plans/issue-<N>/plan.md`).
  3. Close the other plan PRs (comment: "Plan not selected").
  4. Record the **plan-winner** stat (state comment).
  5. Swap label `lbm:phase-plan` → `lbm:phase-implement`.
  6. **Re-dispatch** all agents with `phase=implement`.
- **`/merge-plan <alias> <feedback…>` → REVISE (bounded by `feedback_revs`, default 1):**
  1. Post the feedback to that agent's plan PR mentioning the agent (`@claude …`), so
     the agent revises `plan.md` in place (one rev).
  2. Do **not** finalize; the maintainer runs `/merge-plan <alias>` (no feedback) to
     finalize after the rev. (Revision count tracked via a `plan-rev` comment marker,
     capped at `feedback_revs`.)

Reuses the existing alias resolution, PR-close, and dispatch primitives from
`agent_ops.py` (no new matching logic).

---

## Phase 2 — Implement

Re-dispatch (from `/merge-plan` finalize) all agents with `phase=implement`:
> "Implement issue #N following the approved plan at `lbm-plans/issue-<N>/plan.md`
> (now on the default branch). Open a PR."

Agents open code PRs → existing CI / preview / repair / auto-run flow → normal
`/merge <alias>` picks the winning code PR, records the **code-winner** stat, closes
losers. The existing `## Agent Implementations` table is used for Phase 2.

---

## Prompt mode threading (no string-sniffing)

- `lbm-agents.yml` (workflow_dispatch) + `_agents.yml` + `_agent-claude.yml` gain a
  `phase` input (`plan` | `implement`; default `implement` = today's behavior).
- A shared helper `agent_ops.build_task_prompt(issue_num, phase, plan_dir)` returns the
  phase-specific instructions, appended to the issue body. Both harness workflows call
  it (claude via its prompt-build; codex via its "Build prompt" step).
- The dispatcher passes `phase`: router → `plan` (Phase 1), `/merge-plan` finalize →
  `implement` (Phase 2). When `[plan].enabled` is false, `phase` is always `implement`.

---

## Works without Hub

Entire flow is comment-command + label + status-comment driven: `ready-for-dev` →
plan PRs + `## Agent Plans` (with plan links) → `/merge-plan <alias> [feedback]` →
`## Agent Implementations` → `/merge <alias>`. Hub is purely additive.

---

## Fast-follow A — Prototype iteration

`[plan].prototype = true`. Chosen mechanism: **prototype PR → auto-run → report →
plan.** In Phase 1, before writing the plan, an agent opens a small prototype PR; the
existing experiment auto-run posts a report comment; the agent is then re-invoked
(a `<!-- lbm-plan-context -->` marker comment carrying the report, dual to the repair
watcher) to write/refine `plan.md` using the report as context. Reuses auto-run + the
comment re-invoke. Detailed design in the plan's fast-follow section.

## Fast-follow B — LBM Hub (`../lbm-hub`, Next.js)

- **Plan review/select:** render each agent's `plan.md` (from the branch blob links),
  a "Select" action (+ optional feedback) that posts `/merge-plan <alias> [feedback]`.
- **Stats:** show **plan-merge** (which agent's plan won) and **code-merge** (which
  agent's code won) per iteration.
- **Contract (what lbm-poc emits for Hub to read):** the `## Agent Plans` and
  `## Agent Implementations` status comments, the `lbm:phase-*` labels, per-agent plan
  blob links, and a machine-readable stats marker comment
  (`<!-- lbm:stats {...} -->`) written on `/merge-plan` finalize and `/merge`.

---

## Changes by file (core, this pass)

| Repo | File | Change |
|---|---|---|
| lbm-poc | `scripts/models.py` | `PlanConfig` + `LBMConfig.plan` |
| lbm-poc | `scripts/config_parser.py` | `get_plan_config(config)` |
| lbm-poc | `scripts/agent_ops.py` | `build_task_prompt`; `## Agent Plans` table + plan-link column; `cmd_merge_plan` (finalize + feedback rev + re-dispatch implement); plan/code winner stats marker |
| lbm-poc | `.github/workflows/_dispatch.yml` | Phase-1 plan dispatch: `lbm:phase-plan`, `## Agent Plans`, `phase=plan` |
| lbm-poc | `.github/workflows/_agents.yml` | `phase` input → mode prompt (codex/openhands) |
| lbm-poc | `.github/workflows/_agent-claude.yml` | `phase` input → mode prompt (claude) |
| lbm-poc | `.github/workflows/_comments.yml` | `/merge-plan` job |
| lbm-poc | `cli/templates/lbm.toml.j2` | `[plan]` section (default disabled; commented recommended values) |
| lbm-poc | wrappers (`lbm-dispatch.yml`, `lbm-agents.yml` templates) | thread `phase` input |
| chess-rl | `lbm.toml` | opt in `[plan] enabled = true` |
| chess-rl | `AGENTS.md` | plan-phase guidance (what a good plan looks like) |

## Testing

- **config_parser / models:** `get_plan_config` + `PlanConfig.from_dict` (present/absent
  defaults).
- **build_task_prompt:** plan vs implement text; plan path substitution.
- **merge-plan logic:** alias→plan-PR resolution; feedback vs finalize branch; rev cap.
- **YAML validity** for all touched workflows; **equivalence test** for the template.
- **E2E** on chess-rl: label an issue → verify Phase-1 plan PRs + `## Agent Plans` links
  → `/merge-plan <alias>` → plan merged + Phase-2 implement dispatch → `/merge`.

## Resolved mechanisms (as built)

The following were decided during implementation and supersede the open questions:

- **Phase detection is label-based, not input-threaded.** `config_parser.resolve_phase(cfg, labels)`
  returns `plan` iff `[plan].enabled` and the issue lacks `lbm:phase-implement`, else
  `implement`. Every harness workflow resolves phase identically from durable issue
  labels — no `phase` needs to be threaded through `workflow_dispatch`/`dispatch_agent`,
  and there is no race with the triggering event.
- **Claude gets plan instructions via `--append-system-prompt`** (appended to
  `claude_args`), computed from the resolved phase. This is race-free because it keys
  off labels rather than the webhook's issue-body snapshot (claude-code-action reads the
  issue from the event payload). Codex gets them appended to its built prompt.
- **Phase-2 re-dispatch = re-apply the ready label.** `/merge-plan` finalize adds
  `lbm:phase-implement`, then removes+re-adds the ready label. That re-fires
  `issues.labeled`, which re-runs both the claude wrapper (via `label_trigger`) and the
  router (which dispatches codex/openhands) — all now in the implement phase. This
  reuses existing triggers instead of inventing a claude `workflow_dispatch` route.
- **Status tables** reuse one 5-column helper with a configurable 4th-column label
  ("Plan" vs "Preview"); the router posts the phase-appropriate table idempotently.
- **Branch reuse:** Phase-2 opens a fresh implementation branch (plan branches are
  deleted on merge/close); `git push --force` covers any name reuse.
- **v1 retag** affects all `@v1` consumers; plan step is gated by `[plan].enabled`
  (default off) so other repos are unaffected.
- **Known limitation (fast-follow):** the OpenHands harness reads the issue via its
  resolver and does not yet receive plan-phase instructions, so `[plan].enabled` + an
  OpenHands agent is not fully supported. chess-rl uses claude + codex only; other
  consumers keep `enabled=false`.
