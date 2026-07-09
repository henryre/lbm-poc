# LBM Repair-Comment Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A configurable `lbm.toml` list of comment-match strings that, when found in a PR comment, fire LBM's existing `dispatch-repair` with the comment body as context — with chess-rl's failing experiment report as the first consumer.

**Architecture:** Add config + a `_comments.yml` watcher job in `lbm-poc` that reuses `dispatch-repair` unchanged; chess-rl's `experiment.yml` posts a failing report as a NEW comment (via the LBM App token) carrying a `<!-- lbm-repair -->` marker, which the watcher matches → repair. Loop-safe via marker-strip, `[repair-attempt]` guard, author gate, and the existing attempt cap.

**Tech Stack:** Python 3.12, pytest, GitHub Actions reusable workflows, `gh` CLI, `actions/create-github-app-token`.

## Global Constraints

- **Spec:** `lbm-poc/docs/superpowers/specs/2026-06-24-lbm-repair-comment-trigger-design.md`.
- **`dispatch-repair` is unchanged** — reuse attempt counting / ralph / manual escalation.
- **Substring match only** (regex deferred).
- **Default marker:** `<!-- lbm-repair -->` (hidden HTML comment).
- **Config default `[]`** when the key is absent (existing repos unaffected / feature off).
- **Token:** marker comment must be posted with the **LBM App token / PAT**, and as a **`created`** comment (edits don't fire `issue_comment` workflows).
- **Loop guards:** strip markers from dispatched context; watcher ignores any body containing `[repair-attempt]`; author must be OWNER/MEMBER/COLLABORATOR or a Bot.
- lbm-poc tests: `test/`, run `uv run pytest` (or `pytest`); tests do `sys.path.insert(0, ".../scripts")` then `import config_parser` / `from models import ...`.
- chess-rl tests: `uv run pytest -q` from repo root.

---

### Task 1: Config field + match/strip helpers (lbm-poc)

**Files:**
- Modify: `scripts/models.py` (`ChecksConfig`, lines 44-58)
- Modify: `scripts/config_parser.py` (add 3 helpers)
- Test: `test/test_repair_triggers.py` (new)

**Produces:**
- `models.ChecksConfig.repair_comment_triggers: list[str]` (default `[]`)
- `config_parser.get_repair_comment_triggers(config: dict) -> list[str]`
- `config_parser.matches_repair_trigger(body: str, triggers: list[str]) -> bool`
- `config_parser.strip_repair_triggers(body: str, triggers: list[str]) -> str`

- [ ] **Step 1: Write the failing test** — `test/test_repair_triggers.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import config_parser  # noqa: E402
from models import ChecksConfig  # noqa: E402

MARK = "<!-- lbm-repair -->"


def test_get_triggers_present_and_absent():
    assert config_parser.get_repair_comment_triggers({"checks": {"repair_comment_triggers": [MARK]}}) == [MARK]
    assert config_parser.get_repair_comment_triggers({"checks": {}}) == []
    assert config_parser.get_repair_comment_triggers({}) == []


def test_matches_any_trigger():
    assert config_parser.matches_repair_trigger(f"report\n{MARK}\n", [MARK]) is True
    assert config_parser.matches_repair_trigger("plain report", [MARK]) is False
    assert config_parser.matches_repair_trigger("anything", []) is False


def test_strip_removes_all_markers_and_trims():
    assert config_parser.strip_repair_triggers(f"body {MARK} end {MARK}", [MARK]) == "body  end"
    assert config_parser.strip_repair_triggers("no marker", [MARK]) == "no marker"


def test_checksconfig_parses_triggers():
    assert ChecksConfig.from_dict({"repair_comment_triggers": [MARK]}).repair_comment_triggers == [MARK]
    assert ChecksConfig.from_dict({}).repair_comment_triggers == []
```

- [ ] **Step 2: Run → fail.** `cd /home/ubuntu/lbm-poc && uv run pytest test/test_repair_triggers.py -q` → FAIL (AttributeError / no `repair_comment_triggers`).

- [ ] **Step 3: Add the `ChecksConfig` field** — `scripts/models.py`, in the dataclass (after line 49) and `from_dict` (after line 57):

```python
    max_ralph_loops: int = 0
    repair_comment_triggers: list[str] = field(default_factory=list)
```
```python
            max_ralph_loops=d.get("max_ralph_loops", 0),
            repair_comment_triggers=d.get("repair_comment_triggers", []),
```

- [ ] **Step 4: Add the helpers** — append to `scripts/config_parser.py`:

```python
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
```

- [ ] **Step 5: Run → pass.** `uv run pytest test/test_repair_triggers.py -q` → PASS. Also `uv run pytest -q` (no regressions).

- [ ] **Step 6: Commit.**
```bash
cd /home/ubuntu/lbm-poc && git add scripts/models.py scripts/config_parser.py test/test_repair_triggers.py
git commit -m "feat(checks): repair_comment_triggers config + match/strip helpers"
```

---

### Task 2: `_comments.yml` repair-watcher job (lbm-poc)

**Files:**
- Modify: `.github/workflows/_comments.yml` (add a job, mirroring `feedback` at lines 234-347)

**Interfaces:**
- Consumes: `config_parser.load_config`, `get_repair_comment_triggers`, `matches_repair_trigger`, `strip_repair_triggers` (Task 1); `agent_ops.py dispatch-repair <pr> <context>` (existing).

- [ ] **Step 1: Add the job** at the end of `.github/workflows/_comments.yml` (same indentation as other jobs):

```yaml
  # -----------------------------------------------------------------------
  # Repair-comment trigger — a comment containing a configured
  # [checks].repair_comment_triggers marker dispatches a repair with the
  # comment body as context. Reuses dispatch-repair (attempt cap / ralph / manual).
  # -----------------------------------------------------------------------
  repair-comment:
    if: >-
      github.event_name == 'issue_comment' &&
      github.event.issue.pull_request &&
      !contains(github.event.comment.body, '[repair-attempt]') &&
      (
        contains(fromJson('["OWNER","MEMBER","COLLABORATOR"]'), github.event.comment.author_association) ||
        github.event.comment.user.type == 'Bot'
      )
    runs-on: ubuntu-latest
    concurrency:
      group: repair-comment-${{ github.event.issue.number }}
      cancel-in-progress: false
    steps:
      - name: Generate LBM App token
        id: gen-token
        uses: actions/create-github-app-token@v1
        continue-on-error: true
        with:
          app-id: ${{ secrets.LBM_APP_ID }}
          private-key: ${{ secrets.LBM_APP_PRIVATE_KEY }}

      - name: Set LBM token
        run: |
          if [ -n "${{ steps.gen-token.outputs.token }}" ]; then
            echo "LBM_TOKEN=${{ steps.gen-token.outputs.token }}" >> $GITHUB_ENV
          else
            echo "LBM_TOKEN=${{ secrets.PAT_TOKEN }}" >> $GITHUB_ENV
          fi

      - uses: actions/checkout@v4
        with:
          ref: main

      - uses: actions/checkout@v4
        with:
          repository: henryre/lbm-poc
          path: .lbm
          token: ${{ env.LBM_TOKEN }}

      - name: Match trigger and dispatch repair
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          PAT_TOKEN: ${{ env.LBM_TOKEN }}
          COMMENT_BODY: ${{ github.event.comment.body }}
          PR_NUM: ${{ github.event.issue.number }}
          LBM_CONFIG_PATH: ${{ inputs.config-path }}
        run: |
          CLEANED=$(python3 -c "
          import os, sys
          sys.path.insert(0, '.lbm/scripts')
          import config_parser
          cfg = config_parser.load_config(os.environ['LBM_CONFIG_PATH'])
          triggers = config_parser.get_repair_comment_triggers(cfg)
          body = os.environ['COMMENT_BODY']
          if not config_parser.matches_repair_trigger(body, triggers):
              print('__LBM_NO_TRIGGER__')
          else:
              print(config_parser.strip_repair_triggers(body, triggers))
          ")
          if [ "$CLEANED" = "__LBM_NO_TRIGGER__" ]; then
            echo "No repair trigger in comment; skipping."
            exit 0
          fi
          python3 .lbm/scripts/agent_ops.py dispatch-repair "$PR_NUM" "$CLEANED"
```

- [ ] **Step 2: Validate YAML.**
```bash
cd /home/ubuntu/lbm-poc && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/_comments.yml')); print('YAML OK')"
```
Expected: `YAML OK`.

- [ ] **Step 3: Commit.**
```bash
git add .github/workflows/_comments.yml
git commit -m "feat(comments): repair-comment watcher job (marker -> dispatch-repair)"
```

---

### Task 3: Template default (lbm-poc)

**Files:**
- Modify: `templates/lbm.toml.j2` (`[checks]` section)

- [ ] **Step 1: Inspect the template's checks block.**
```bash
cd /home/ubuntu/lbm-poc && grep -nE '\[checks\]|max_repair_attempts|max_ralph_loops|required|repair_from' templates/lbm.toml.j2
```

- [ ] **Step 2: Add the default** immediately after the `max_ralph_loops` line in `[checks]`:

```jinja
max_ralph_loops = {{ max_ralph_loops | default(0) }}

# PR comments containing any of these substrings trigger a repair, with the
# comment body as failure context. Empty list disables the feature.
repair_comment_triggers = ["<!-- lbm-repair -->"]
```
(If the template has no `[checks]` block, add one with `required`, `repair_from`, `max_repair_attempts`, `max_ralph_loops`, `repair_comment_triggers` mirroring the spec's config example. Match the file's existing Jinja style.)

- [ ] **Step 3: Commit.**
```bash
git add templates/lbm.toml.j2
git commit -m "feat(template): default repair_comment_triggers marker"
```

---

### Task 4: chess-rl failure verdict + summary status (chess-rl)

**Files:**
- Modify: `/home/ubuntu/chess-rl/chessrl/results.py` (`write_summary`)
- Modify: `/home/ubuntu/chess-rl/scripts/run_experiment.py` (`run`, add `summary_status`)
- Test: `/home/ubuntu/chess-rl/tests/test_scripts.py` (append)

**Produces:**
- `results.write_summary(out_dir, payload)` always writes a `status` key (default `"ok"`, overridable via `payload["status"]`).
- `run_experiment.summary_status(out_dir: str) -> str | None` (reads `summary.json`'s `status`).
- `run_experiment.run(name)` returns `1` when the entrypoint exits 0 **but** `summary.json` has `status == "failed"`.

- [ ] **Step 1: Write failing tests** — append to `/home/ubuntu/chess-rl/tests/test_scripts.py`:

```python
import json


def test_write_summary_defaults_status_ok(tmp_path):
    from chessrl import results

    path = results.write_summary(str(tmp_path), {"metrics": {"x": 1}})
    assert json.load(open(path))["status"] == "ok"


def test_write_summary_respects_explicit_status(tmp_path):
    from chessrl import results

    path = results.write_summary(str(tmp_path), {"status": "failed"})
    assert json.load(open(path))["status"] == "failed"


def test_summary_status_reads_file(tmp_path):
    from chessrl import results

    results.write_summary(str(tmp_path), {"status": "failed"})
    assert run_experiment.summary_status(str(tmp_path)) == "failed"


def test_summary_status_missing_file_is_none(tmp_path):
    assert run_experiment.summary_status(str(tmp_path)) is None
```

- [ ] **Step 2: Run → fail.** `cd /home/ubuntu/chess-rl && uv run pytest tests/test_scripts.py -q` → FAIL.

- [ ] **Step 3: Update `write_summary`** — `chessrl/results.py`, ensure a status key:

```python
def write_summary(out_dir: str, payload: dict[str, Any]) -> str:
    """Write ``payload`` as ``summary.json`` in ``out_dir``; return the path.

    Always includes a ``status`` key (default ``"ok"``) so the CI runner can
    detect a self-declared failure; pass ``status="failed"`` in ``payload`` to
    mark a bad run.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(out_dir) / "summary.json"
    body = {"status": "ok", **payload}
    dest.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return str(dest)
```

- [ ] **Step 4: Add `summary_status` + wire into `run`** — `scripts/run_experiment.py`:

```python
def summary_status(out_dir: str) -> str | None:
    """Return summary.json's ``status`` field, or None if absent/unreadable."""
    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (ValueError, OSError):
        return None
```
Add `import json` at the top if missing. In `run(name)`, replace `return proc.wait()` with:
```python
        code = proc.wait()
    # A clean exit with a self-declared failed summary is still a failure.
    if code == 0 and summary_status(str(out_dir)) == "failed":
        print("[run_experiment] summary.json reports status=failed", flush=True)
        return 1
    return code
```
(Note the `with` block ends before the status check; keep the tee-loop inside `with`.)

- [ ] **Step 5: Run → pass.** `uv run pytest -q` → PASS (all, incl. existing 28).

- [ ] **Step 6: Commit.**
```bash
cd /home/ubuntu/chess-rl && git checkout exp/0001-tweak-eval-rows
git add chessrl/results.py scripts/run_experiment.py tests/test_scripts.py
git commit -m "feat: summary status field + run_experiment failure verdict"
```

---

### Task 5: chess-rl report → repair marker (chess-rl)

**Files:**
- Modify: `/home/ubuntu/chess-rl/.github/workflows/experiment.yml` (on `main`)
- Modify: `/home/ubuntu/chess-rl/lbm.toml` (add trigger under `[checks]`)

**Interfaces:**
- Consumes: the `repair-comment` watcher (Task 2) + `repair_comment_triggers` (Task 1). Marker string `<!-- lbm-repair -->`.

- [ ] **Step 1: Ensure chess-rl opts in** — `/home/ubuntu/chess-rl/lbm.toml`, under `[checks]` (create the block if absent):
```toml
[checks]
required = ["CI"]
repair_comment_triggers = ["<!-- lbm-repair -->"]
```

- [ ] **Step 2: Add an LBM App token step** to `experiment.yml` (after `Install uv` / before the run), so the failing report can be posted with a workflow-triggering token:
```yaml
      - name: Generate LBM App token
        id: gen-token
        uses: actions/create-github-app-token@v1
        continue-on-error: true
        with:
          app-id: ${{ secrets.LBM_APP_ID }}
          private-key: ${{ secrets.LBM_APP_PRIVATE_KEY }}
```

- [ ] **Step 3: On failure, append the marker and post a NEW comment via the LBM token.** Replace the `Post / update report comment` step so success upserts (edit, any token) and failure posts a fresh comment (created, LBM token → fires the watcher):

```yaml
      - name: Post / update report comment
        if: always() && steps.resolve.outputs.name != ''
        env:
          GH_TOKEN: ${{ github.token }}
          LBM_TOKEN: ${{ steps.gen-token.outputs.token }}
        run: |
          NAME="${{ steps.resolve.outputs.name }}"
          MARKER="<!-- chess-rl-report:${NAME} -->"
          if [ "${{ steps.run.outcome }}" = "success" ]; then
            # Success: upsert the marker comment in place (no repair marker).
            CID="$(gh api "repos/${{ github.repository }}/issues/${{ github.event.issue.number }}/comments" --jq ".[] | select(.body | startswith(\"$MARKER\")) | .id" | head -n1)"
            if [ -n "$CID" ]; then
              gh api -X PATCH "repos/${{ github.repository }}/issues/comments/$CID" -F body=@report.md
            else
              gh pr comment ${{ github.event.issue.number }} --body-file report.md
            fi
          else
            # Failure: append the repair marker and post a NEW comment via the LBM
            # token so the issue_comment 'created' event fires the repair watcher.
            printf '\n\n<!-- lbm-repair -->\n' >> report.md
            if [ -n "$LBM_TOKEN" ]; then
              GH_TOKEN="$LBM_TOKEN" gh pr comment ${{ github.event.issue.number }} --body-file report.md
            else
              echo "::warning::LBM token unavailable; posting report without triggering repair"
              gh pr comment ${{ github.event.issue.number }} --body-file report.md
            fi
          fi
```

- [ ] **Step 4: Validate YAML.**
```bash
cd /home/ubuntu/chess-rl && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/experiment.yml')); print('YAML OK')"
```

- [ ] **Step 5: Commit `experiment.yml` on main; `lbm.toml` on both main and the PR branch.**
```bash
cd /home/ubuntu/chess-rl && git checkout main
git add .github/workflows/experiment.yml lbm.toml
git commit -m "ci(experiment): post failing report via LBM token with lbm-repair marker"
git push origin main
git checkout exp/0001-tweak-eval-rows && git checkout main -- lbm.toml && git add lbm.toml && git commit -m "chore: opt into repair_comment_triggers" && git push origin exp/0001-tweak-eval-rows
```

---

## Verification (autonomous)

1. **lbm-poc unit tests:** `cd /home/ubuntu/lbm-poc && uv run pytest -q` → all pass (incl. new `test_repair_triggers.py`).
2. **chess-rl unit tests:** `cd /home/ubuntu/chess-rl && uv run pytest -q` → all pass (incl. new verdict tests).
3. **YAML validity:** both `_comments.yml` and `experiment.yml` parse.
4. **Push lbm-poc** (`main`) so the reusable `_comments.yml` change is live for chess-rl's wrapper.
5. **Live plumbing check (no GPU):** post a comment containing `<!-- lbm-repair -->` on chess-rl PR #1 via the LBM token; confirm the `LBM: Comment Commands` workflow fires the `repair-comment` job. Since PR #1 is not an agent branch, `dispatch-repair` should log "Not an agent branch" and no-op — which confirms the watcher → dispatch-repair plumbing end-to-end without needing an agent. (Full agent dispatch requires an agent-branch PR from an LBM iteration.)

## Self-Review

- **Spec coverage:** config+helpers (T1), watcher job (T2), template default (T3), chess-rl verdict/status (T4), report marker + LBM-token post + opt-in (T5). ✓
- **Deviations (noted):** (a) failure verdict rides `run_experiment`'s exit code rather than a separate `$GITHUB_OUTPUT` — simpler, and `experiment.yml` already branches on `steps.run.outcome`; (b) failing report is a **new** comment (created) rather than an in-place edit — required because edits don't fire `issue_comment` workflows; (c) marker string hardcoded in chess-rl (matches its `lbm.toml`) rather than parsed from toml in bash — documented constant.
- **Placeholders:** none.
- **Type consistency:** `get_repair_comment_triggers`/`matches_repair_trigger`/`strip_repair_triggers`/`summary_status`/`write_summary(status)` used consistently across tasks and the workflow step.
