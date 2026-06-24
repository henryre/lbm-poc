// TS rendering route for `lbm init`. Mirrors the context-building logic in
// cli/main.py exactly; the equivalence test (test/equivalence.test.mjs) asserts
// byte-for-byte parity against the Python CLI across a config matrix.
import { ALL_TEMPLATES } from "./templates.js";
import { renderTemplate, type Scope } from "./jinja.js";

/** Input config — same shape the Hub form builds and `lbm init --from-json` reads. */
export interface LbmConfig {
  runtime: string;
  agents: string[];
  deploy_platform?: string;
  app_prefix?: string;
  deploy_region?: string;
  required_checks?: string[];
  /** Accepted for compatibility with the form payload; ignored (CLI hardcodes anthropic). */
  llm_provider?: string;
}

export interface RenderedFile {
  path: string;
  content: string;
}

// Mirror of DEFAULT_AGENTS in cli/main.py.
const DEFAULT_AGENTS: Record<string, { harness: string; model_id: string; model_label: string; mention: string }> = {
  claude: { harness: "claude", model_id: "claude-opus-4-8", model_label: "opus-4-8", mention: "@claude" },
  codex: { harness: "codex", model_id: "gpt-5.5", model_label: "gpt-5.5", mention: "@codex" },
  openhands: {
    harness: "openhands",
    model_id: "gemini/gemini-3.1-pro-preview",
    model_label: "gemini-3.1-pro",
    mention: "@openhands-agent",
  },
};

// Mirror of RUNTIME_DEFAULTS in cli/main.py.
const RUNTIME_DEFAULTS: Record<string, { install: string; lint: string; typecheck: string; build: string }> = {
  node: { install: "npm ci", lint: "npm run lint", typecheck: "npx tsc --noEmit", build: "npm run build" },
  python: { install: "pip install -e '.[dev]'", lint: "ruff check .", typecheck: "", build: "" },
  go: { install: "", lint: "golangci-lint run", typecheck: "", build: "go build ./..." },
  rust: { install: "", lint: "cargo clippy", typecheck: "", build: "cargo build" },
  custom: { install: "", lint: "", typecheck: "", build: "" },
};

/** Match Python's `json.dumps(list_of_str)` formatting: `["CI", "Lint"]` (comma+space). */
function pyJsonList(items: string[]): string {
  return "[" + items.map((s) => JSON.stringify(s)).join(", ") + "]";
}

function buildContext(config: LbmConfig): Scope {
  const runtime = config.runtime;
  const deployPlatform = config.deploy_platform ?? "none";
  const appPrefix = config.app_prefix ?? "";
  const deployRegion = config.deploy_region ?? "iad";
  const requiredChecks = config.required_checks ?? ["CI"];

  const available = Object.keys(DEFAULT_AGENTS);
  const selected = config.agents.map((a) => a.trim()).filter((a) => available.includes(a));

  const rt = RUNTIME_DEFAULTS[runtime] ?? RUNTIME_DEFAULTS.custom!;

  // Insertion order = selected order, matching the Python dict/list comprehensions.
  const harnesses: Record<string, { mention: string }> = {};
  for (const name of selected) harnesses[name] = { mention: DEFAULT_AGENTS[name]!.mention };
  const agents = selected.map((name) => DEFAULT_AGENTS[name]!);

  return {
    lbm_repo: "henryre/lbm-poc",
    lbm_ref: "v1",
    runtime,
    install_cmd: rt.install,
    lint_cmd: rt.lint,
    typecheck_cmd: rt.typecheck,
    build_cmd: rt.build,
    deploy_platform: deployPlatform,
    app_prefix: appPrefix,
    deploy_region: deployRegion,
    llm_provider: "anthropic",
    summary_model: "claude-sonnet-4-6",
    harnesses,
    agents,
    guidance_file: "AGENTS.md",
    required_checks: pyJsonList(requiredChecks),
  };
}

/**
 * Render all LBM files for a config. Returns every file `lbm init` would write
 * (lbm.toml, the four .github/workflows/lbm-*.yml, and AGENTS.md). The caller
 * decides whether to write AGENTS.md (the CLI skips it if one already exists).
 */
export function renderLbmFiles(config: LbmConfig): RenderedFile[] {
  const ctx = buildContext(config);
  return ALL_TEMPLATES.map((t) => ({ path: t.path, content: renderTemplate(t.source, ctx) }));
}
