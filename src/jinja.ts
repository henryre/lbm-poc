// Minimal Jinja2-subset renderer — just enough to render the LBM templates in
// cli/templates/*.j2 with semantics identical to Python's Jinja2 as configured
// by the CLI (trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True).
//
// Because trim_blocks is False, text between tags (including the newlines that
// follow `%}`) is emitted verbatim whenever its branch is active, and dropped
// entirely when its branch is skipped. This file reproduces exactly that by
// keeping text nodes byte-for-byte and only emitting them inside active branches.
//
// Supported features (the only ones the LBM templates use):
//   {{ var }}, {{ a.b }}, {{ 'literal' }}
//   {% if EXPR %} / {% elif EXPR %} / {% else %} / {% endif %}
//   {% for x in LIST %} / {% for k, v in OBJ.items() %} / {% endfor %}
//   EXPR forms: truthiness, `a == "x"`, `a != "x"`, `a in ["x", "y"]`, `'x' in obj`

export type Scope = Record<string, unknown>;

type Node =
  | { kind: "text"; value: string }
  | { kind: "var"; expr: string }
  | { kind: "if"; branches: { cond: string; body: Node[] }[]; elseBody: Node[] | null }
  | { kind: "for"; targets: string[]; source: string; body: Node[] };

type Token =
  | { t: "text"; value: string }
  | { t: "var"; expr: string }
  | { t: "tag"; raw: string };

// Find the index of `close` ("}}" or "%}") at or after `from`, skipping any
// occurrence that sits inside a single- or double-quoted string. This matches
// Jinja, where e.g. `{{ '}}' }}` — the `}}` inside the quotes does not close
// the expression (used by the templates to emit literal GitHub Actions `${{ }}`).
function findClose(src: string, from: number, close: string): number {
  let i = from;
  let quote: string | null = null;
  while (i < src.length) {
    const c = src[i]!;
    if (quote) {
      if (c === quote) quote = null;
    } else if (c === "'" || c === '"') {
      quote = c;
    } else if (src.startsWith(close, i)) {
      return i;
    }
    i++;
  }
  return -1;
}

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  let last = 0;
  let i = 0;
  while (i < src.length) {
    const isVar = src.startsWith("{{", i);
    const isTag = src.startsWith("{%", i);
    if (!isVar && !isTag) {
      i++;
      continue;
    }
    const close = isVar ? "}}" : "%}";
    const end = findClose(src, i + 2, close);
    if (end === -1) {
      i++;
      continue; // unterminated — treat as text
    }
    if (i > last) tokens.push({ t: "text", value: src.slice(last, i) });
    const inner = src.slice(i + 2, end).trim();
    if (isVar) tokens.push({ t: "var", expr: inner });
    else tokens.push({ t: "tag", raw: inner });
    i = end + 2;
    last = i;
  }
  if (last < src.length) tokens.push({ t: "text", value: src.slice(last) });
  return tokens;
}

function parse(tokens: Token[], pos = 0, stop: string[] = []): { nodes: Node[]; pos: number; stopWord: string | null } {
  const nodes: Node[] = [];
  let i = pos;
  while (i < tokens.length) {
    const tok = tokens[i]!;
    if (tok.t === "text") {
      nodes.push({ kind: "text", value: tok.value });
      i++;
    } else if (tok.t === "var") {
      nodes.push({ kind: "var", expr: tok.expr });
      i++;
    } else {
      const word = tok.raw.split(/\s+/)[0]!;
      if (stop.includes(word)) {
        return { nodes, pos: i, stopWord: word };
      }
      if (word === "if") {
        const branches: { cond: string; body: Node[] }[] = [];
        let elseBody: Node[] | null = null;
        let cond = tok.raw.slice(2).trim();
        i++;
        while (true) {
          const r = parse(tokens, i, ["elif", "else", "endif"]);
          branches.push({ cond, body: r.nodes });
          i = r.pos;
          const stopTok = tokens[i] as Extract<Token, { t: "tag" }>;
          if (r.stopWord === "elif") {
            cond = stopTok.raw.slice(4).trim();
            i++;
            continue;
          }
          if (r.stopWord === "else") {
            i++;
            const er = parse(tokens, i, ["endif"]);
            elseBody = er.nodes;
            i = er.pos + 1; // consume endif
            break;
          }
          // endif
          i++;
          break;
        }
        nodes.push({ kind: "if", branches, elseBody });
      } else if (word === "for") {
        // for <targets> in <source>
        const body = tok.raw.slice(3).trim();
        const inIdx = body.indexOf(" in ");
        const targets = body.slice(0, inIdx).split(",").map((s) => s.trim());
        const source = body.slice(inIdx + 4).trim();
        i++;
        const r = parse(tokens, i, ["endfor"]);
        i = r.pos + 1; // consume endfor
        nodes.push({ kind: "for", targets, source, body: r.nodes });
      } else {
        throw new Error(`Unsupported tag: {% ${tok.raw} %}`);
      }
    }
  }
  return { nodes, pos: i, stopWord: null };
}

function parseLiteral(expr: string): { isLiteral: boolean; value?: unknown } {
  if ((expr.startsWith("'") && expr.endsWith("'")) || (expr.startsWith('"') && expr.endsWith('"'))) {
    return { isLiteral: true, value: expr.slice(1, -1) };
  }
  if (expr.startsWith("[") && expr.endsWith("]")) {
    const inner = expr.slice(1, -1).trim();
    if (inner === "") return { isLiteral: true, value: [] };
    const items = inner.split(",").map((s) => {
      const v = s.trim();
      return v.slice(1, -1); // strip quotes
    });
    return { isLiteral: true, value: items };
  }
  return { isLiteral: false };
}

function resolvePath(path: string, scope: Scope): unknown {
  const parts = path.split(".");
  let cur: unknown = scope;
  for (const p of parts) {
    if (p.endsWith("()")) {
      // only .items() is used; handled by caller, so ignore here
      const key = p.slice(0, -2);
      cur = (cur as Record<string, unknown>)?.[key];
      continue;
    }
    if (cur == null) return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

function evalValue(expr: string, scope: Scope): unknown {
  const lit = parseLiteral(expr);
  if (lit.isLiteral) return lit.value;
  return resolvePath(expr, scope);
}

function truthy(v: unknown): boolean {
  if (v == null || v === false) return false;
  if (typeof v === "string") return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  return Boolean(v);
}

function evalCondition(expr: string, scope: Scope): boolean {
  let m = expr.split(" != ");
  if (m.length === 2) return String(evalValue(m[0]!.trim(), scope)) !== String(evalValue(m[1]!.trim(), scope));
  m = expr.split(" == ");
  if (m.length === 2) return String(evalValue(m[0]!.trim(), scope)) === String(evalValue(m[1]!.trim(), scope));
  const inIdx = expr.indexOf(" in ");
  if (inIdx !== -1) {
    const left = evalValue(expr.slice(0, inIdx).trim(), scope);
    const right = evalValue(expr.slice(inIdx + 4).trim(), scope);
    if (Array.isArray(right)) return right.includes(left);
    if (right && typeof right === "object") return String(left) in (right as object);
    return false;
  }
  return truthy(evalValue(expr, scope));
}

function render(nodes: Node[], scope: Scope): string {
  let out = "";
  for (const n of nodes) {
    if (n.kind === "text") {
      out += n.value;
    } else if (n.kind === "var") {
      const v = evalValue(n.expr, scope);
      out += v == null ? "" : String(v);
    } else if (n.kind === "if") {
      let rendered = false;
      for (const b of n.branches) {
        if (evalCondition(b.cond, scope)) {
          out += render(b.body, scope);
          rendered = true;
          break;
        }
      }
      if (!rendered && n.elseBody) out += render(n.elseBody, scope);
    } else {
      // for
      let iterable: Array<unknown[]>;
      if (n.source.endsWith(".items()")) {
        const base = resolvePath(n.source.slice(0, -".items()".length), scope) as Record<string, unknown>;
        iterable = Object.entries(base ?? {});
      } else {
        const arr = (evalValue(n.source, scope) as unknown[]) ?? [];
        iterable = arr.map((el) => [el]);
      }
      for (const item of iterable) {
        const child: Scope = { ...scope };
        n.targets.forEach((tgt, idx) => {
          child[tgt] = item[idx];
        });
        out += render(n.body, child);
      }
    }
  }
  return out;
}

export function renderTemplate(source: string, scope: Scope): string {
  const tokens = tokenize(source);
  const { nodes } = parse(tokens);
  return render(nodes, scope);
}
