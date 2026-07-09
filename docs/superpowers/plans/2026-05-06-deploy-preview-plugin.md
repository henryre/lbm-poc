# Deploy Preview Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable preview deployment system to LBM that deploys live preview instances of agent PRs on Fly.io, with GHCR as the image registry, while keeping the existing Vercel/Netlify webhook flow intact.

**Architecture:** Two deployment patterns (passive/webhook for Vercel/Netlify, active/workflow for Fly/Railway) behind a unified LBM contract. A new reusable workflow `_deploy-fly.yml` handles image build, GHCR push, Fly Machine creation, and cleanup. The target repo (DeepTutor) gets a Caddyfile + fly.toml + Dockerfile update to expose both ports behind a single reverse proxy.

**Tech Stack:** Fly.io Machines API, GitHub Container Registry (ghcr.io), Caddy reverse proxy, GitHub Actions reusable workflows, Python (config parsing + CLI).

---

## File Structure

### LBM Core (`/home/ubuntu/lbm-poc`)

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/config_parser.py` | Modify | Add `get_deploy_config()` and `is_active_deploy_platform()` |
| `scripts/models.py` | Modify | Add `DeployConfig` dataclass |
| `templates/lbm.toml.j2` | Modify | Add `app_prefix`, `region`, `registry` fields |
| `cli/main.py` | Modify | Add Fly to deploy choices, prompt for app_prefix/region |
| `.github/workflows/_deploy-fly.yml` | Create | Reusable Fly deploy/destroy workflow |
| `.github/workflows/_ci-hooks.yml` | Modify | Add `deploy-active` + `cleanup` jobs |
| `test/test_config_parser.py` | Modify | Tests for new config functions |
| `test/test_models.py` | Modify | Tests for DeployConfig |

### DeepTutor (`/home/ubuntu/DeepTutor-explore`)

| File | Action | Responsibility |
|------|--------|---------------|
| `Caddyfile` | Create | Reverse proxy routing :8080 → :3782 / :8001 |
| `fly.toml` | Create | Fly service definition |
| `Dockerfile` | Modify | Install Caddy, add proxy to supervisord |

---

## Task 1: Add DeployConfig Model

**Files:**
- Modify: `scripts/models.py`
- Test: `test/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `test/test_models.py`:

```python
class TestDeployConfig:
    def test_from_dict_full(self):
        d = {
            "platform": "fly",
            "preview_env": "Preview",
            "app_prefix": "myapp-pr",
            "region": "iad",
            "registry": "ghcr",
        }
        cfg = DeployConfig.from_dict(d)
        assert cfg.platform == "fly"
        assert cfg.preview_env == "Preview"
        assert cfg.app_prefix == "myapp-pr"
        assert cfg.region == "iad"
        assert cfg.registry == "ghcr"

    def test_from_dict_defaults(self):
        d = {"platform": "vercel"}
        cfg = DeployConfig.from_dict(d)
        assert cfg.platform == "vercel"
        assert cfg.preview_env == "Preview"
        assert cfg.app_prefix == ""
        assert cfg.region == "iad"
        assert cfg.registry == "ghcr"

    def test_from_dict_empty(self):
        cfg = DeployConfig.from_dict({})
        assert cfg.platform == "none"

    def test_is_active(self):
        assert DeployConfig.from_dict({"platform": "fly"}).is_active is True
        assert DeployConfig.from_dict({"platform": "railway"}).is_active is True
        assert DeployConfig.from_dict({"platform": "vercel"}).is_active is False
        assert DeployConfig.from_dict({"platform": "none"}).is_active is False
```

Also add the import at the top of `test/test_models.py`:

```python
from models import AgentConfig, ChecksConfig, LLMConfig, LBMConfig, DeployConfig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/lbm-poc && python -m pytest test/test_models.py::TestDeployConfig -v`
Expected: FAIL with `ImportError: cannot import name 'DeployConfig'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/models.py` after the `LLMConfig` class:

```python
ACTIVE_DEPLOY_PLATFORMS = ("fly", "railway")


@dataclass(frozen=True)
class DeployConfig:
    platform: str = "none"
    preview_env: str = "Preview"
    app_prefix: str = ""
    region: str = "iad"
    registry: str = "ghcr"

    @property
    def is_active(self) -> bool:
        return self.platform in ACTIVE_DEPLOY_PLATFORMS

    @classmethod
    def from_dict(cls, d: dict) -> DeployConfig:
        return cls(
            platform=d.get("platform", "none"),
            preview_env=d.get("preview_env", "Preview"),
            app_prefix=d.get("app_prefix", ""),
            region=d.get("region", "iad"),
            registry=d.get("registry", "ghcr"),
        )
```

Also update `LBMConfig` to include deploy:

```python
@dataclass
class LBMConfig:
    agents: list[AgentConfig]
    checks: ChecksConfig
    llm: LLMConfig
    deploy: DeployConfig

    @classmethod
    def from_parsed_toml(cls, raw: dict) -> LBMConfig:
        # ... existing code ...
        deploy = DeployConfig.from_dict(raw.get("deploy", {}))
        return cls(agents=agents, checks=checks, llm=llm, deploy=deploy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/lbm-poc && python -m pytest test/test_models.py -v`
Expected: ALL PASS (including existing tests — `LBMConfig.from_parsed_toml` signature changed so verify `TestLBMConfig` still passes)

- [ ] **Step 5: Commit**

```bash
git add scripts/models.py test/test_models.py
git commit -m "feat: add DeployConfig model with is_active property"
```

---

## Task 2: Add Config Parser Functions

**Files:**
- Modify: `scripts/config_parser.py`
- Test: `test/test_config_parser.py`

- [ ] **Step 1: Write the failing tests**

Add to `test/test_config_parser.py`:

```python
class TestGetDeployConfig:
    def test_returns_full_config(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        toml = """
[deploy]
platform = "fly"
preview_env = "Preview"
app_prefix = "myapp-pr"
region = "iad"
registry = "ghcr"
"""
        config = tomllib.loads(toml)
        result = config_parser.get_deploy_config(config)
        assert result["platform"] == "fly"
        assert result["app_prefix"] == "myapp-pr"
        assert result["region"] == "iad"
        assert result["registry"] == "ghcr"

    def test_defaults_when_missing(self):
        result = config_parser.get_deploy_config({})
        assert result["platform"] == "none"
        assert result["preview_env"] == "Preview"
        assert result["app_prefix"] == ""
        assert result["region"] == "iad"
        assert result["registry"] == "ghcr"

    def test_existing_vercel_config(self, config):
        result = config_parser.get_deploy_config(config)
        assert result["platform"] == "vercel"
        assert result["preview_env"] == "Preview"


class TestIsActiveDeployPlatform:
    def test_fly_is_active(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        config = tomllib.loads('[deploy]\nplatform = "fly"')
        assert config_parser.is_active_deploy_platform(config) is True

    def test_vercel_is_not_active(self, config):
        assert config_parser.is_active_deploy_platform(config) is False

    def test_none_is_not_active(self):
        assert config_parser.is_active_deploy_platform({}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/lbm-poc && python -m pytest test/test_config_parser.py::TestGetDeployConfig -v`
Expected: FAIL with `AttributeError: module 'config_parser' has no attribute 'get_deploy_config'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/config_parser.py` after `get_deploy_platform()`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/lbm-poc && python -m pytest test/test_config_parser.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/config_parser.py test/test_config_parser.py
git commit -m "feat: add get_deploy_config and is_active_deploy_platform"
```

---

## Task 3: Update lbm.toml Template and CLI

**Files:**
- Modify: `templates/lbm.toml.j2`
- Modify: `cli/main.py`

- [ ] **Step 1: Update lbm.toml template**

Replace the `[deploy]` section in `templates/lbm.toml.j2`:

```jinja2
[deploy]
platform = "{{ deploy_platform }}"
{% if deploy_platform != "none" %}
preview_env = "Preview"
{% endif %}
{% if deploy_platform in ["fly", "railway"] %}
app_prefix = "{{ app_prefix }}"
region = "{{ deploy_region }}"
registry = "ghcr"
{% endif %}
```

- [ ] **Step 2: Update CLI init to add Fly option and prompts**

In `cli/main.py`, change the `deploy_platform` prompt:

```python
deploy_platform = click.prompt(
    "Deploy platform", type=click.Choice(["vercel", "netlify", "fly", "railway", "none"]), default="none"
)

# Active deploy platforms need additional config
app_prefix = ""
deploy_region = "iad"
if deploy_platform in ("fly", "railway"):
    app_prefix = click.prompt("App prefix (preview URLs will be {prefix}-{pr_number}.fly.dev)", default="app-pr")
    deploy_region = click.prompt("Deploy region", default="iad")
```

Add `app_prefix` and `deploy_region` to the `context` dict:

```python
context = {
    # ... existing fields ...
    "app_prefix": app_prefix,
    "deploy_region": deploy_region,
}
```

Also add to the setup instructions section:

```python
if deploy_platform == "fly":
    click.echo("  FLY_API_TOKEN (Fly.io API token)")
```

- [ ] **Step 3: Verify template renders correctly**

Run:
```bash
cd /home/ubuntu/lbm-poc && python -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'), keep_trailing_newline=True)
t = env.get_template('lbm.toml.j2')
print(t.render(
    runtime='node', install_cmd='npm ci', lint_cmd='npm run lint',
    typecheck_cmd='npx tsc --noEmit', build_cmd='npm run build',
    deploy_platform='fly', app_prefix='deeptutor-pr', deploy_region='iad',
    database_orm='none', llm_provider='anthropic',
    summary_model='claude-sonnet-4-6',
    harnesses={'claude': {'mention': '@claude'}},
    agents=[{'harness': 'claude', 'model_id': 'claude-opus-4-6', 'model_label': 'opus-4-6'}],
    guidance_file='AGENTS.md',
))
"
```

Expected: Output contains:
```
[deploy]
platform = "fly"
preview_env = "Preview"
app_prefix = "deeptutor-pr"
region = "iad"
registry = "ghcr"
```

- [ ] **Step 4: Commit**

```bash
git add templates/lbm.toml.j2 cli/main.py
git commit -m "feat: add Fly.io to deploy platform options in template and CLI"
```

---

## Task 4: Create `_deploy-fly.yml` Reusable Workflow

**Files:**
- Create: `.github/workflows/_deploy-fly.yml`

- [ ] **Step 1: Create the workflow file**

```yaml
name: "LBM: Deploy to Fly.io"

# Reusable workflow — deploys or destroys a Fly.io preview machine.
# Called by _ci-hooks.yml on CI success (deploy) or PR close (destroy).

on:
  workflow_call:
    inputs:
      action:
        description: "'deploy' or 'destroy'"
        required: true
        type: string
      pr_number:
        description: "PR number"
        required: true
        type: string
      branch:
        description: "Head branch name"
        required: false
        type: string
        default: ""
      issue_number:
        description: "Linked issue number"
        required: false
        type: string
        default: ""
      config-path:
        description: "Path to lbm.toml"
        required: false
        type: string
        default: "lbm.toml"

jobs:
  deploy:
    if: inputs.action == 'deploy'
    runs-on: ubuntu-latest
    environment: Preview
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ inputs.branch }}

      - uses: actions/checkout@v4
        with:
          repository: henryre/lbm-poc
          path: .lbm
          token: ${{ secrets.PAT_TOKEN }}

      - name: Read deploy config
        id: config
        run: |
          DEPLOY_CONFIG=$(python3 -c "
          import sys
          sys.path.insert(0, '.lbm/scripts')
          from config_parser import load_config, get_deploy_config
          cfg = load_config('${{ inputs.config-path }}')
          dc = get_deploy_config(cfg)
          print(f\"app_prefix={dc['app_prefix']}\")
          print(f\"region={dc['region']}\")
          ")
          echo "$DEPLOY_CONFIG" >> "$GITHUB_OUTPUT"

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:pr-${{ inputs.pr_number }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Install flyctl
        uses: superfly/flyctl-actions/setup-flyctl@master

      - name: Deploy to Fly.io
        id: fly-deploy
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
        run: |
          APP_NAME="${{ steps.config.outputs.app_prefix }}-${{ inputs.pr_number }}"
          REGION="${{ steps.config.outputs.region }}"
          IMAGE="ghcr.io/${{ github.repository }}:pr-${{ inputs.pr_number }}"

          # Create app if it doesn't exist
          if ! flyctl apps list --json | grep -q "\"$APP_NAME\""; then
            flyctl apps create "$APP_NAME" --org personal
          fi

          # Deploy the image
          flyctl deploy \
            --app "$APP_NAME" \
            --image "$IMAGE" \
            --region "$REGION" \
            --ha=false \
            --vm-size shared-cpu-2x \
            --vm-memory 1024 \
            --env BACKEND_PORT=8001 \
            --env FRONTEND_PORT=3782 \
            --env LLM_BINDING="${{ vars.LLM_BINDING || 'openai' }}" \
            --env LLM_MODEL="${{ vars.LLM_MODEL || 'gpt-4o-mini' }}" \
            --env LLM_HOST="${{ vars.LLM_HOST || 'https://api.openai.com/v1' }}" \
            --env EMBEDDING_BINDING="${{ vars.EMBEDDING_BINDING || 'openai' }}" \
            --env EMBEDDING_MODEL="${{ vars.EMBEDDING_MODEL || 'text-embedding-3-large' }}" \
            --env EMBEDDING_HOST="${{ vars.EMBEDDING_HOST || 'https://api.openai.com/v1/embeddings' }}" \
            2>&1 | tee /tmp/deploy.log

          if [ $? -ne 0 ]; then
            echo "deploy_failed=true" >> "$GITHUB_OUTPUT"
            echo "deploy_logs<<EOF" >> "$GITHUB_OUTPUT"
            tail -50 /tmp/deploy.log >> "$GITHUB_OUTPUT"
            echo "EOF" >> "$GITHUB_OUTPUT"
          else
            echo "deploy_failed=false" >> "$GITHUB_OUTPUT"
            echo "preview_url=https://${APP_NAME}.fly.dev" >> "$GITHUB_OUTPUT"
          fi

      - name: Set Fly secrets
        if: steps.fly-deploy.outputs.deploy_failed == 'false'
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
        run: |
          APP_NAME="${{ steps.config.outputs.app_prefix }}-${{ inputs.pr_number }}"
          flyctl secrets set \
            --app "$APP_NAME" \
            LLM_API_KEY="${{ secrets.LLM_API_KEY }}" \
            EMBEDDING_API_KEY="${{ secrets.EMBEDDING_API_KEY }}" \
            2>/dev/null || true

      - name: Report preview URL
        if: steps.fly-deploy.outputs.deploy_failed == 'false' && inputs.issue_number != ''
        env:
          GH_TOKEN: ${{ secrets.PAT_TOKEN || secrets.GITHUB_TOKEN }}
          LBM_CONFIG_PATH: ${{ inputs.config-path }}
        run: |
          PREVIEW_URL="${{ steps.fly-deploy.outputs.preview_url }}"
          BRANCH="${{ inputs.branch }}"

          AGENT_INFO=$(python3 .lbm/scripts/agent_ops.py lookup branch-to-name "$BRANCH" 2>/dev/null || echo "")
          if [ -z "$AGENT_INFO" ]; then
            echo "Not an agent branch: $BRANCH"
            exit 0
          fi

          AGENT_LABEL=$(echo "$AGENT_INFO" | grep "^label=" | cut -d= -f2)
          python3 .lbm/scripts/agent_ops.py update-status \
            "${{ inputs.issue_number }}" "$AGENT_LABEL" "preview" \
            "${{ inputs.pr_number }}" "$PREVIEW_URL" || true

      - name: Dispatch repair on failure
        if: steps.fly-deploy.outputs.deploy_failed == 'true'
        env:
          GH_TOKEN: ${{ secrets.PAT_TOKEN || secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          LBM_CONFIG_PATH: ${{ inputs.config-path }}
        run: |
          LOGS="${{ steps.fly-deploy.outputs.deploy_logs }}"
          python3 .lbm/scripts/agent_ops.py dispatch-repair \
            "${{ inputs.pr_number }}" \
            "Deployment to Fly.io failed (CI passed but deploy did not). Likely causes: missing env vars, runtime errors during startup, port binding issues.

          Deploy logs:
          \`\`\`
          ${LOGS}
          \`\`\`"

  destroy:
    if: inputs.action == 'destroy'
    runs-on: ubuntu-latest
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main

      - uses: actions/checkout@v4
        with:
          repository: henryre/lbm-poc
          path: .lbm
          token: ${{ secrets.PAT_TOKEN }}

      - name: Read deploy config
        id: config
        run: |
          DEPLOY_CONFIG=$(python3 -c "
          import sys
          sys.path.insert(0, '.lbm/scripts')
          from config_parser import load_config, get_deploy_config
          cfg = load_config('${{ inputs.config-path }}')
          dc = get_deploy_config(cfg)
          print(f\"app_prefix={dc['app_prefix']}\")
          ")
          echo "$DEPLOY_CONFIG" >> "$GITHUB_OUTPUT"

      - name: Install flyctl
        uses: superfly/flyctl-actions/setup-flyctl@master

      - name: Destroy Fly app
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
        run: |
          APP_NAME="${{ steps.config.outputs.app_prefix }}-${{ inputs.pr_number }}"
          flyctl apps destroy "$APP_NAME" --yes 2>/dev/null || echo "App $APP_NAME not found (already destroyed)"

      - name: Delete GHCR image
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          REPO="${{ github.repository }}"
          OWNER="${{ github.repository_owner }}"
          PACKAGE_NAME=$(echo "$REPO" | cut -d/ -f2)
          TAG="pr-${{ inputs.pr_number }}"

          # Find the version ID for this tag
          VERSION_ID=$(gh api \
            "/user/packages/container/${PACKAGE_NAME}/versions" \
            --jq ".[] | select(.metadata.container.tags[] == \"${TAG}\") | .id" \
            2>/dev/null || echo "")

          if [ -n "$VERSION_ID" ]; then
            gh api --method DELETE \
              "/user/packages/container/${PACKAGE_NAME}/versions/${VERSION_ID}" \
              2>/dev/null || echo "Failed to delete image tag ${TAG}"
            echo "Deleted GHCR image tag: ${TAG}"
          else
            echo "No GHCR image found for tag: ${TAG}"
          fi
```

- [ ] **Step 2: Validate YAML syntax**

Run: `cd /home/ubuntu/lbm-poc && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/_deploy-fly.yml'))"`
Expected: No error (valid YAML)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/_deploy-fly.yml
git commit -m "feat: add _deploy-fly.yml reusable workflow for Fly.io previews"
```

---

## Task 5: Update `_ci-hooks.yml` with Deploy Routing

**Files:**
- Modify: `.github/workflows/_ci-hooks.yml`

- [ ] **Step 1: Add deploy-active job**

Add after the `auto-merge` job in `_ci-hooks.yml`:

```yaml
  # -----------------------------------------------------------------------
  # Deploy (active platforms) — build + deploy on CI success
  # -----------------------------------------------------------------------
  deploy-active:
    if: >-
      github.event_name == 'workflow_run' &&
      github.event.workflow_run.conclusion == 'success'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main

      - uses: actions/checkout@v4
        with:
          repository: henryre/lbm-poc
          path: .lbm
          token: ${{ secrets.PAT_TOKEN }}

      - name: Check if active deploy platform
        id: check
        env:
          LBM_CONFIG_PATH: ${{ inputs.config-path }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          IS_ACTIVE=$(python3 -c "
          import sys
          sys.path.insert(0, '.lbm/scripts')
          from config_parser import load_config, is_active_deploy_platform
          cfg = load_config('${{ inputs.config-path }}')
          print('true' if is_active_deploy_platform(cfg) else 'false')
          ")
          echo "is_active=$IS_ACTIVE" >> "$GITHUB_OUTPUT"

          if [ "$IS_ACTIVE" = "false" ]; then
            echo "Not an active deploy platform, skipping"
            exit 0
          fi

          BRANCH="${{ github.event.workflow_run.head_branch }}"
          PR_NUM=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number // empty' 2>/dev/null || echo "")
          echo "pr_number=$PR_NUM" >> "$GITHUB_OUTPUT"
          echo "branch=$BRANCH" >> "$GITHUB_OUTPUT"

          if [ -z "$PR_NUM" ]; then
            echo "No PR for branch $BRANCH"
            exit 0
          fi

          # Check if this is an agent branch
          AGENT_INFO=$(python3 .lbm/scripts/agent_ops.py lookup branch-to-name "$BRANCH" 2>/dev/null || echo "")
          if [ -z "$AGENT_INFO" ]; then
            echo "Not an agent branch: $BRANCH"
            echo "is_agent=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          echo "is_agent=true" >> "$GITHUB_OUTPUT"

          # Get issue number from PR body
          PR_BODY=$(gh pr view "$PR_NUM" --json body --jq '.body' 2>/dev/null || echo "")
          ISSUE_NUM=$(echo "$PR_BODY" | grep -oP 'Implements #\K\d+' || echo "")
          echo "issue_number=$ISSUE_NUM" >> "$GITHUB_OUTPUT"

      - name: Dispatch Fly deploy
        if: steps.check.outputs.is_active == 'true' && steps.check.outputs.is_agent == 'true' && steps.check.outputs.pr_number != ''
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.PAT_TOKEN }}
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: '_deploy-fly.yml',
              ref: 'main',
              inputs: {
                action: 'deploy',
                pr_number: '${{ steps.check.outputs.pr_number }}',
                branch: '${{ steps.check.outputs.branch }}',
                issue_number: '${{ steps.check.outputs.issue_number }}',
                'config-path': '${{ inputs.config-path }}',
              }
            });
```

- [ ] **Step 2: Add cleanup job**

Add after the `deploy-active` job:

```yaml
  # -----------------------------------------------------------------------
  # Cleanup — destroy preview on PR close (active platforms only)
  # -----------------------------------------------------------------------
  cleanup:
    if: >-
      github.event_name == 'pull_request' &&
      github.event.action == 'closed'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main

      - uses: actions/checkout@v4
        with:
          repository: henryre/lbm-poc
          path: .lbm
          token: ${{ secrets.PAT_TOKEN }}

      - name: Check if active deploy + agent branch
        id: check
        run: |
          IS_ACTIVE=$(python3 -c "
          import sys
          sys.path.insert(0, '.lbm/scripts')
          from config_parser import load_config, is_active_deploy_platform
          cfg = load_config('${{ inputs.config-path }}')
          print('true' if is_active_deploy_platform(cfg) else 'false')
          ")
          echo "is_active=$IS_ACTIVE" >> "$GITHUB_OUTPUT"

          BRANCH="${{ github.event.pull_request.head.ref }}"
          AGENT_INFO=$(python3 .lbm/scripts/agent_ops.py lookup branch-to-name "$BRANCH" 2>/dev/null || echo "")
          if [ -z "$AGENT_INFO" ]; then
            echo "is_agent=false" >> "$GITHUB_OUTPUT"
          else
            echo "is_agent=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Dispatch Fly destroy
        if: steps.check.outputs.is_active == 'true' && steps.check.outputs.is_agent == 'true'
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.PAT_TOKEN }}
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: '_deploy-fly.yml',
              ref: 'main',
              inputs: {
                action: 'destroy',
                pr_number: '${{ github.event.pull_request.number }}',
                'config-path': '${{ inputs.config-path }}',
              }
            });
```

- [ ] **Step 3: Update the workflow trigger to accept `pull_request` events**

The `_ci-hooks.yml` `on.workflow_call` section stays the same (the calling workflow in the target repo is responsible for passing the right events). But document in the comment header that target repos using active deploy platforms must also trigger on `pull_request: types: [closed]`.

Update the top comment:

```yaml
# Reusable workflow — called by target repos on workflow_run + deployment_status + pull_request(closed).
# Handles CI failure repair, auto-merge, deploy failure repair, preview link posting, and cleanup.
```

- [ ] **Step 4: Validate YAML syntax**

Run: `cd /home/ubuntu/lbm-poc && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/_ci-hooks.yml'))"`
Expected: No error

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/_ci-hooks.yml
git commit -m "feat: add deploy-active and cleanup jobs to _ci-hooks.yml"
```

---

## Task 6: Add Caddyfile and fly.toml to DeepTutor

**Files:**
- Create: `/home/ubuntu/DeepTutor-explore/Caddyfile`
- Create: `/home/ubuntu/DeepTutor-explore/fly.toml`

- [ ] **Step 1: Create Caddyfile**

```
:8080 {
	handle /api/* {
		reverse_proxy localhost:8001
	}
	handle {
		reverse_proxy localhost:3782
	}
}
```

- [ ] **Step 2: Create fly.toml**

```toml
# Fly.io configuration for LBM preview deployments.
# The app name and image are overridden per-deploy via flyctl flags.

[build]

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  size = "shared-cpu-2x"
  memory = 1024
```

- [ ] **Step 3: Verify Caddyfile syntax**

Run: `cd /home/ubuntu/DeepTutor-explore && docker run --rm -v $(pwd)/Caddyfile:/etc/caddy/Caddyfile caddy:2-alpine caddy validate --config /etc/caddy/Caddyfile 2>&1 || echo "Caddy not available for validation, syntax is simple enough to trust"`

- [ ] **Step 4: Commit in DeepTutor repo**

```bash
cd /home/ubuntu/DeepTutor-explore
git add Caddyfile fly.toml
git commit -m "feat: add Caddyfile and fly.toml for LBM preview deployments"
```

---

## Task 7: Update DeepTutor Dockerfile to Include Caddy

**Files:**
- Modify: `/home/ubuntu/DeepTutor-explore/Dockerfile`

- [ ] **Step 1: Add Caddy installation to Stage 3 (production)**

In the production stage, after the existing `apt-get install` block (line ~128-138), add Caddy installation:

```dockerfile
# Install Caddy for reverse proxy (used in Fly.io preview deployments)
RUN curl -sSL "https://caddyserver.com/api/download?os=linux&arch=$(dpkg --print-architecture)" -o /usr/local/bin/caddy \
    && chmod +x /usr/local/bin/caddy
```

- [ ] **Step 2: Copy Caddyfile into the image**

After the `COPY pyproject.toml ./` line (around line 163), add:

```dockerfile
# Copy Caddy reverse proxy config (for single-port deployments like Fly.io)
COPY Caddyfile ./
```

- [ ] **Step 3: Add Caddy to supervisord config**

In the supervisord config section (the heredoc starting around line 187), add a new program block after `[program:frontend]`:

```ini
[program:caddy]
command=/usr/local/bin/caddy run --config /app/Caddyfile
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
```

- [ ] **Step 4: Add port 8080 to EXPOSE**

Change `EXPOSE 8001 3782` to:

```dockerfile
EXPOSE 8001 3782 8080
```

- [ ] **Step 5: Verify Docker build still works**

Run:
```bash
cd /home/ubuntu/DeepTutor-explore && docker build --target production -t deeptutor-test . 2>&1 | tail -20
```

Expected: Build completes successfully (will take several minutes due to npm install + pip install)

- [ ] **Step 6: Commit**

```bash
cd /home/ubuntu/DeepTutor-explore
git add Dockerfile
git commit -m "feat: add Caddy reverse proxy to Dockerfile for Fly.io previews"
```

---

## Task 8: End-to-End Verification with Fly.io

**Files:** None (testing only)

- [ ] **Step 1: Build and push image to GHCR manually**

```bash
cd /home/ubuntu/DeepTutor-explore
docker build --target production -t ghcr.io/henryre/deeptutor:pr-test .
echo $GITHUB_TOKEN | docker login ghcr.io -u henryre --password-stdin
docker push ghcr.io/henryre/deeptutor:pr-test
```

- [ ] **Step 2: Deploy to Fly.io manually**

```bash
export FLY_API_TOKEN=$(cat /home/ubuntu/.fly-token)
export PATH="/home/ubuntu/.fly/bin:$PATH"

flyctl apps create deeptutor-pr-test --org personal
flyctl deploy \
  --app deeptutor-pr-test \
  --image ghcr.io/henryre/deeptutor:pr-test \
  --region iad \
  --ha=false \
  --vm-size shared-cpu-2x \
  --vm-memory 1024 \
  --env BACKEND_PORT=8001 \
  --env FRONTEND_PORT=3782 \
  --env LLM_BINDING=openai \
  --env LLM_MODEL=gpt-4o-mini \
  --env LLM_HOST=https://api.openai.com/v1
```

- [ ] **Step 3: Verify the preview is accessible**

Run: `curl -sI https://deeptutor-pr-test.fly.dev | head -5`
Expected: HTTP 200 (or 302 redirect to login/welcome page)

Test API endpoint: `curl -s https://deeptutor-pr-test.fly.dev/api/system | python3 -m json.tool`
Expected: JSON response from the FastAPI backend

- [ ] **Step 4: Verify auto-stop behavior**

Wait 5+ minutes with no requests, then:
```bash
flyctl machines list --app deeptutor-pr-test
```
Expected: Machine status shows "stopped"

Then hit the URL again:
```bash
curl -sI https://deeptutor-pr-test.fly.dev | head -5
```
Expected: Response arrives (machine auto-started, may take 2-3s)

- [ ] **Step 5: Clean up test deployment**

```bash
flyctl apps destroy deeptutor-pr-test --yes
```

- [ ] **Step 6: Document results**

Note any issues found and adjustments needed. If Caddy routing or port configuration needed changes, go back and fix the relevant task.

---

## Task 9: Push Changes and Create PR

**Files:** None (git operations only)

- [ ] **Step 1: Push LBM changes**

```bash
cd /home/ubuntu/lbm-poc
git push origin main
```

- [ ] **Step 2: Push DeepTutor changes on a feature branch**

```bash
cd /home/ubuntu/DeepTutor-explore
git checkout -b feat/fly-preview
git push -u origin feat/fly-preview
```

- [ ] **Step 3: Create PR on DeepTutor**

```bash
cd /home/ubuntu/DeepTutor-explore
gh pr create --title "Add Fly.io preview deployment support" --body "$(cat <<'EOF'
## Summary
- Adds Caddyfile for single-port reverse proxy (routes /api/* to backend, everything else to frontend)
- Adds fly.toml for Fly.io machine configuration (auto-stop/start, shared-cpu-2x)
- Updates Dockerfile to install Caddy and run it alongside backend/frontend via supervisord

## Context
Part of the LBM deploy preview plugin system. When an LBM agent creates a PR, the CI pipeline builds a Docker image, pushes to GHCR, and deploys a Fly Machine so reviewers can interact with a live preview.

## Test plan
- [ ] Docker build completes without errors
- [ ] Caddy correctly proxies /api/* to :8001 and / to :3782
- [ ] Fly.io deployment accessible at https://{app-name}.fly.dev
- [ ] Auto-stop works after 5 min idle
- [ ] Auto-start works on next request
EOF
)"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `python -m pytest test/ -v` passes in LBM repo (all existing + new tests)
- [ ] `_deploy-fly.yml` is valid YAML
- [ ] `_ci-hooks.yml` is valid YAML with new jobs
- [ ] DeepTutor Docker image builds successfully with Caddy
- [ ] Manual Fly deploy/destroy cycle works end-to-end
- [ ] Preview URL is accessible and serves both frontend and API
- [ ] Existing Vercel/webhook flow is unaffected (no changes to `deploy-failure` or `post-preview` jobs)
