# LBM Repair-Comment Trigger Design

**Date:** 2026-06-24
**Repo:** `lbm-poc` (generic LBM feature); consumed by `henryre/chess-rl`
**Status:** Approved design — pre-implementation

## Summary

A generic LBM capability: **configurable comment-match strings that trigger the
repair loop.** When a PR comment contains any string listed in
`lbm.toml [checks].repair_comment_triggers`, LBM dispatches a repair to the
authoring agent with the comment body as failure context — reusing the existing
`dispatch-repair` machinery (attempt counting, `@agent` re-invoke, ralph-restart,
manual escalation). This generalizes today's failure→repair behavior (CI failures,
Vercel/Fly deploy failures) to "any tool that posts a comment with the configured
marker." The first consumer is `chess-rl`, whose experiment **report comment**
embeds the marker when a run fails, turning a failing report into a repair.

## Motivation

LBM already turns failures into agent repairs, but only from sources LBM itself
observes: CI `workflow_run` failures and `deployment_status` failures (Vercel) /
the active Fly deploy job. Repos with their own notion of "failure" — e.g. an RL
experiment whose report says the run was bad — have no hook into the loop. Rather
than teach LBM about experiments, give it a **domain-agnostic** hook: a comment
contains a configured string → repair. The repo decides what failure means and
emits the marker; LBM owns the loop.

## Goals

1. `lbm.toml` configures one or more **match strings**; a matching PR comment
   triggers a repair with that comment as context.
2. Reuse `dispatch-repair` end-to-end — no parallel counter, no new escalation.
   `max_repair_attempts` / `max_ralph_loops` govern it unchanged.
3. Safe against self-retrigger loops and unauthorized injection.
4. Disabled by default-safe behavior; opt-in per repo.
5. `chess-rl` consumes it: failing experiment report → repair.

## Non-Goals

- Changing how CI / Vercel / Fly failures dispatch repair (untouched).
- Regex matching in v1 (substring match; regex is a future extension).
- Any chess/experiment-specific logic in LBM core.

---

## Configuration — `lbm.toml [checks]`

```toml
[checks]
required = ["CI"]
repair_from = ["CI"]
max_repair_attempts = 10
max_ralph_loops = 0

# NEW: PR comments containing ANY of these substrings trigger a repair, with the
# comment body as the failure context. Empty list disables the feature.
repair_comment_triggers = ["<!-- lbm-repair -->"]
```

- **Type:** list of substrings. A comment matches if it contains any entry.
- **Default:** `["<!-- lbm-repair -->"]` — a hidden HTML-comment marker (invisible
  when rendered, near-impossible to type by accident). Shipped in
  `templates/lbm.toml.j2`. The hidden-marker default is the analog of LBM's
  built-in Vercel failure detection: works out of the box once a tool emits it.
- **`config_parser.py`:** add `repair_comment_triggers: list[str]` to the checks
  config, defaulting to `[]` when the key is absent (so existing repos are
  unaffected); the template provides the recommended non-empty value for new repos.

```python
# config_parser.py — in the checks config builder
"repair_comment_triggers": checks.get("repair_comment_triggers", []),
```

---

## Workflow — new job in `_comments.yml`

A job mirroring the existing command jobs (`/merge`, `/feedback`, `@codex`):

**Trigger / guard (`if:`)**
- `github.event_name == 'issue_comment'` and `github.event.issue.pull_request` (PR).
- Author authorized: `author_association in (OWNER, MEMBER, COLLABORATOR)` **or**
  the comment author is the LBM bot/app (same allowance as the existing
  `/repair`/mention path). This stops arbitrary contributors from injecting repairs.
- Body does **not** contain `[repair-attempt]` (LBM's own repair comments carry this
  marker — primary self-retrigger guard).

**Steps**
1. Generate LBM App token + checkout `.lbm/scripts` (as other jobs do).
2. **Check trigger** (Python step using `config_parser.load_config`): read
   `repair_comment_triggers`; if the comment body contains none → set
   `skip=true`. (The `if:` can't read `lbm.toml`, so the substring test lives in a
   step, exactly like the existing "Check trigger" step in the agent jobs.)
3. **Dispatch repair** (if not skipped): strip the trigger markers from the body,
   then `python3 .lbm/scripts/agent_ops.py dispatch-repair "$PR_NUM" "$CLEANED_BODY"`.

`dispatch-repair` is unchanged: resolves agent-by-branch (no-op on non-agent
branches), counts `repair-attempt` comments vs `max_repair_attempts`, posts the
`@agent [repair-attempt] <context>` comment, escalates to ralph / manual when
exhausted.

### Loop-prevention (defense in depth)

1. **Marker stripping:** the watcher removes all configured trigger substrings from
   the context before calling `dispatch-repair`, so the resulting
   `@agent [repair-attempt] …` comment cannot itself match a trigger.
2. **`[repair-attempt]` guard:** the watcher ignores any comment containing
   `[repair-attempt]` (belt-and-suspenders, covers the case where stripping is
   bypassed or context echoes the marker).
3. **Author guard:** only authorized/bot authors can trigger (above).
4. **Attempt cap:** `max_repair_attempts` bounds the cycle regardless.

### Token requirement

The triggering comment must be posted with the **LBM App token / PAT**, not the
default `GITHUB_TOKEN` — GitHub does not fire `issue_comment` workflows for comments
made with `GITHUB_TOKEN`. This is the same constraint the existing repair/mention
flow already relies on (`_dispatch_repair_comment` posts via `PAT_TOKEN`).

---

## Consumer — `chess-rl`

The chess-rl experiment pipeline already posts an LLM-written **report comment** on
the PR. To plug into the loop:

1. **Failure verdict** (`scripts/run_experiment.py`): overall status is `failed` if
   the entrypoint exits non-zero **or** `$CHESS_RL_OUTPUT_DIR/summary.json` has
   `status == "failed"`. Emitted to `$GITHUB_OUTPUT`. (`chessrl.results.write_summary`
   gains a `status` field, default `"ok"`.)
2. **Marker injection** (`scripts/make_report.py` / `experiment.yml`): on `failed`,
   read the first entry of `repair_comment_triggers` from `lbm.toml` (single source
   of truth) and append it to the report comment. The report body (diagnosis +
   run-log tail) is exactly the context the agent receives.
3. **Token:** post the report comment via the LBM token when LBM is installed, so it
   fires the watcher. Without LBM / without a configured trigger, post the report
   plain — today's manual behavior is unchanged.
4. **Run start / loop continuation:** agents post `/run-experiment <name>` when they
   finish (initial work and after each repair). Run → fail → marker report → repair →
   agent fixes → posts `/run-experiment` again. Bounded by `max_repair_attempts`.
5. **Success:** report posted **without** the marker; it stands as the preview.

`chess-rl` thus knows nothing about `dispatch-repair`; it only emits a configured
string into a comment. Any other repo can adopt the same pattern.

---

## End-to-end flow

```
agent posts /run-experiment <name>
  → experiment.yml runs experiments/<name>/run.py on Modal (synchronous)
  → run fails (non-zero exit OR summary.json status=failed)
  → make_report.py writes the report + appends <!-- lbm-repair --> marker
  → report comment posted via LBM token
  → _comments.yml repair-watcher: body contains a trigger, author=bot, no
    [repair-attempt] → dispatch-repair <pr> "<report minus marker>"
  → dispatch-repair: attempt N < max → posts "@claude [repair-attempt] <report>"
  → agent fixes, pushes, posts /run-experiment again
  → … repeat until success (report w/o marker = preview) or attempts exhausted
    → ralph-restart / manual intervention (existing behavior)
```

---

## Changes by file

| Repo | File | Change |
|---|---|---|
| lbm-poc | `scripts/config_parser.py` | add `repair_comment_triggers` to checks config (default `[]`) |
| lbm-poc | `.github/workflows/_comments.yml` | new repair-watcher job (check-trigger + dispatch-repair) |
| lbm-poc | `templates/lbm.toml.j2` | document + default `repair_comment_triggers = ["<!-- lbm-repair -->"]` |
| lbm-poc | `scripts/agent_ops.py` | (optional) helper to strip markers; or strip in the workflow step |
| chess-rl | `scripts/run_experiment.py` | emit `status` verdict to `$GITHUB_OUTPUT` |
| chess-rl | `scripts/make_report.py` / `experiment.yml` | append marker on failure; post report via LBM token |
| chess-rl | `chessrl/results.py` | `status` field on the summary |

## Testing

- **config_parser:** unit test that `repair_comment_triggers` parses (present →
  list; absent → `[]`).
- **marker matching / stripping:** unit-test the substring match + strip helper
  (matches any entry; strip removes all entries; `[repair-attempt]` bodies excluded).
- **chess-rl status verdict:** unit-test `run_experiment` status logic
  (exit≠0 → failed; summary `status:"failed"` → failed; else ok).
- **End-to-end:** on a chess-rl agent PR, a failing run posts a marker report and a
  `repair-attempt` comment appears (manual/integration check; agent dispatch needs
  the LBM App).

## Open questions / future

- **Regex matching:** v1 is substring; a `repair_comment_triggers_regex` could come
  later if needed.
- **Per-trigger context shaping:** all triggers pass the whole comment body; a future
  version could map specific markers to specific context extractors.
- **Success → status table:** optionally call `update-status … preview <report-url>`
  on success so the report shows in the issue status table (separable follow-up).
