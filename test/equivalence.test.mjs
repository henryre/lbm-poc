// Equivalence test: the TS rendering route (renderLbmFiles) must produce
// byte-for-byte identical files to the Python CLI route (`lbm init --from-json`)
// for every config. This is the gate that keeps the two generators in sync.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { renderLbmFiles } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..");
const cliMain = join(repoRoot, "cli", "main.py");

// Prefer the project venv so click/jinja2 are available; fall back to python3.
const VENV_PY = join(repoRoot, ".venv", "bin", "python");
const PYTHON = existsSync(VENV_PY) ? VENV_PY : "python3";

// Config matrix: exercises every conditional branch in the templates.
const CONFIGS = [
  { runtime: "node", deploy_platform: "none", agents: ["claude", "codex"], required_checks: ["CI"] },
  { runtime: "python", deploy_platform: "fly", agents: ["claude"], required_checks: ["CI", "Lint"] },
  { runtime: "go", deploy_platform: "none", agents: ["codex", "openhands"], required_checks: ["CI"] },
  { runtime: "rust", deploy_platform: "railway", agents: ["openhands"], required_checks: ["CI"] },
  { runtime: "custom", deploy_platform: "netlify", agents: ["claude", "codex", "openhands"], required_checks: ["CI", "Build", "E2E"] },
  { runtime: "node", deploy_platform: "vercel", agents: ["codex"], required_checks: ["CI"] }, // no claude block
];

function runPythonCli(config) {
  const dir = mkdtempSync(join(tmpdir(), "lbm-eq-"));
  try {
    const cfgPath = join(dir, "cfg.json");
    writeFileSync(cfgPath, JSON.stringify(config));
    execFileSync(PYTHON, [cliMain, "init", "--from-json", cfgPath], { cwd: dir, stdio: "pipe" });
    const files = {};
    for (const rel of [
      "lbm.toml",
      ".github/workflows/lbm-dispatch.yml",
      ".github/workflows/lbm-comments.yml",
      ".github/workflows/lbm-agents.yml",
      ".github/workflows/lbm-ci-hooks.yml",
      "AGENTS.md",
    ]) {
      const p = join(dir, rel);
      if (existsSync(p)) files[rel] = readFileSync(p, "utf8");
    }
    return files;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

for (const config of CONFIGS) {
  test(`equivalence: ${JSON.stringify(config)}`, () => {
    const pyFiles = runPythonCli(config);
    const tsFiles = Object.fromEntries(renderLbmFiles(config).map((f) => [f.path, f.content]));

    // Every file the CLI produced must exist in the TS output and match byte-for-byte.
    for (const [path, pyContent] of Object.entries(pyFiles)) {
      assert.ok(path in tsFiles, `TS route is missing file: ${path}`);
      assert.equal(tsFiles[path], pyContent, `Mismatch in ${path}`);
    }
    // And the TS route must not invent files the CLI didn't write.
    for (const path of Object.keys(tsFiles)) {
      assert.ok(path in pyFiles, `TS route produced an extra file: ${path}`);
    }
  });
}
