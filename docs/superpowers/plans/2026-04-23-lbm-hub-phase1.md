# LBM Hub Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP of LBM Hub — a Next.js SaaS dashboard for viewing GitHub repos that use LBM, with GitHub OAuth login, repo onboarding, agent leaderboard, config display, and iteration tracking.

**Architecture:** Next.js 15 App Router with server components for data fetching, NextAuth.js v5 for GitHub OAuth, Drizzle ORM over Neon Postgres for persistence, and Tailwind CSS v4 with Catppuccin Mocha palette. Vercel-style repo context-switching sidebar navigation — top-level nav when no repo selected, repo-scoped nav when inside a repo.

**Tech Stack:** Next.js 15, React 19, NextAuth.js v5, Drizzle ORM, Neon Postgres (replaces deprecated Vercel Postgres), Tailwind CSS v4, Inter + JetBrains Mono fonts.

**Design spec:** `docs/superpowers/specs/2026-04-23-lbm-hub-design.md`

---

## File Structure

```
lbm-hub/
├── auth.ts                          # NextAuth config (GitHub provider)
├── middleware.ts                     # Auth middleware (protect routes)
├── drizzle.config.ts                # Drizzle Kit config
├── next.config.ts                   # Next.js config
├── .env.local                       # Secrets (not committed)
├── drizzle/                         # Generated migration files
├── src/
│   ├── app/
│   │   ├── globals.css              # Tailwind import + Catppuccin theme
│   │   ├── layout.tsx               # Root layout (fonts, providers)
│   │   ├── page.tsx                 # Landing page (logged out) or redirect
│   │   ├── api/
│   │   │   └── auth/
│   │   │       └── [...nextauth]/
│   │   │           └── route.ts     # NextAuth route handler
│   │   ├── login/
│   │   │   └── page.tsx             # Login page
│   │   ├── repos/
│   │   │   ├── layout.tsx           # Top-level repos layout (sidebar)
│   │   │   ├── page.tsx             # Repos grid
│   │   │   └── [owner]/
│   │   │       └── [name]/
│   │   │           ├── layout.tsx   # Repo-scoped layout (sidebar switches)
│   │   │           ├── page.tsx     # Repo overview
│   │   │           ├── iterations/
│   │   │           │   ├── page.tsx # Iterations list
│   │   │           │   └── [id]/
│   │   │           │       └── page.tsx # Iteration detail
│   │   │           ├── agents/
│   │   │           │   └── page.tsx # Agents list
│   │   │           ├── settings/
│   │   │           │   └── page.tsx # Repo settings
│   │   │           └── onboard/
│   │   │               └── page.tsx # Onboard wizard
│   │   └── settings/
│   │       └── page.tsx             # User settings
│   ├── db/
│   │   ├── index.ts                 # Drizzle client
│   │   └── schema.ts               # All table definitions
│   ├── lib/
│   │   ├── github.ts               # GitHub API helpers
│   │   └── utils.ts                # General utilities (cn, etc.)
│   └── components/
│       ├── sidebar.tsx              # Sidebar nav component
│       ├── header.tsx               # Top header bar
│       ├── repo-selector.tsx        # Repo dropdown in header
│       ├── leaderboard-card.tsx     # Agent leaderboard card
│       ├── config-card.tsx          # lbm.toml display card
│       ├── iterations-table.tsx     # Iterations table
│       ├── status-light.tsx         # Status dot component
│       └── sign-in-button.tsx       # GitHub sign-in button
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `lbm-hub/` (entire project via create-next-app)
- Create: `.env.local`

- [ ] **Step 1: Create the Next.js project**

Run from the parent directory of where you want `lbm-hub/`:

```bash
npx create-next-app@latest lbm-hub --typescript --tailwind --eslint --app --src-dir --use-npm
```

When prompted:
- Would you like to use Turbopack? → Yes
- Would you like to customize the import alias? → No (keep `@/`)

- [ ] **Step 2: Install dependencies**

```bash
cd lbm-hub
npm install next-auth drizzle-orm @neondatabase/serverless
npm install -D drizzle-kit
```

- [ ] **Step 3: Create `.env.local`**

Create `lbm-hub/.env.local`:

```
# Auth — generate with: npx auth secret
AUTH_SECRET=

# GitHub OAuth App
GITHUB_ID=
GITHUB_SECRET=

# Neon Postgres
DATABASE_URL=

# App
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

- [ ] **Step 4: Generate auth secret**

```bash
npx auth secret
```

This appends `AUTH_SECRET` to `.env.local`.

- [ ] **Step 5: Verify the app runs**

```bash
npm run dev
```

Open `http://localhost:3000`. Confirm the default Next.js page loads.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold next.js project with dependencies"
```

---

## Task 2: Tailwind Theme + Fonts

**Files:**
- Modify: `src/app/globals.css`
- Modify: `src/app/layout.tsx`
- Create: `src/lib/utils.ts`

- [ ] **Step 1: Set up Catppuccin Mocha theme in globals.css**

Replace `src/app/globals.css` with:

```css
@import "tailwindcss";

@theme {
  /* Catppuccin Mocha */
  --color-base: #1e1e2e;
  --color-mantle: #181825;
  --color-crust: #11111b;
  --color-surface-0: #313244;
  --color-surface-1: #45475a;
  --color-surface-2: #585b70;
  --color-overlay-0: #6c7086;
  --color-overlay-1: #7f849c;
  --color-subtext-0: #a6adc8;
  --color-subtext-1: #bac2de;
  --color-text: #cdd6f4;

  /* Accent colors */
  --color-blue: #89b4fa;
  --color-lavender: #b4befe;
  --color-mauve: #cba6f7;
  --color-green: #a6e3a1;
  --color-peach: #fab387;
  --color-red: #f38ba8;
  --color-yellow: #f9e2af;
  --color-teal: #94e2d5;
  --color-sky: #89dceb;
  --color-pink: #f5c2e7;

  /* Semantic */
  --color-accent: #89b4fa;
  --color-border: #313244;

  /* Status lights */
  --color-status-waiting: #585b70;
  --color-status-working: #89b4fa;
  --color-status-done: #a6e3a1;
  --color-status-merged: #cba6f7;

  /* Font families */
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  /* Border radius */
  --radius-sm: 2px;
  --radius-md: 4px;
  --radius-lg: 6px;
}

body {
  background-color: var(--color-base);
  color: var(--color-text);
}
```

- [ ] **Step 2: Set up fonts in root layout**

Replace `src/app/layout.tsx` with:

```tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "LBM Hub",
  description: "Portal for repos using LBM",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Create utility helpers**

Create `src/lib/utils.ts`:

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

Install the dependencies:

```bash
npm install clsx tailwind-merge
```

- [ ] **Step 4: Create a test page to verify theme**

Replace `src/app/page.tsx` with:

```tsx
export default function Home() {
  return (
    <div className="min-h-screen bg-base p-8">
      <h1 className="text-2xl font-extrabold tracking-tight text-text">
        LBM Hub
      </h1>
      <p className="mt-2 font-mono text-sm text-subtext-0">
        Catppuccin Mocha theme active
      </p>
      <div className="mt-6 flex gap-3">
        <div className="h-8 w-8 rounded-md bg-blue" />
        <div className="h-8 w-8 rounded-md bg-green" />
        <div className="h-8 w-8 rounded-md bg-mauve" />
        <div className="h-8 w-8 rounded-md bg-peach" />
        <div className="h-8 w-8 rounded-md bg-red" />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify in browser**

```bash
npm run dev
```

Open `http://localhost:3000`. Confirm: dark raisin background, white text, Inter font for heading, JetBrains Mono for subtitle, colored squares.

- [ ] **Step 6: Commit**

```bash
git add src/app/globals.css src/app/layout.tsx src/app/page.tsx src/lib/utils.ts package.json package-lock.json
git commit -m "feat: catppuccin mocha theme + inter/jetbrains mono fonts"
```

---

## Task 3: Auth Setup (NextAuth + GitHub)

**Files:**
- Create: `auth.ts`
- Create: `middleware.ts`
- Create: `src/app/api/auth/[...nextauth]/route.ts`
- Create: `src/app/login/page.tsx`
- Create: `src/components/sign-in-button.tsx`
- Modify: `src/app/layout.tsx`

- [ ] **Step 1: Create NextAuth config**

Create `auth.ts` in the project root:

```ts
import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    GitHub({
      clientId: process.env.GITHUB_ID!,
      clientSecret: process.env.GITHUB_SECRET!,
      authorization: {
        params: {
          scope: "read:user repo",
        },
      },
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken as string;
      return session;
    },
  },
});
```

- [ ] **Step 2: Add type augmentation for session**

Create `src/types/next-auth.d.ts`:

```ts
import "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
  }
}
```

- [ ] **Step 3: Create route handler**

Create `src/app/api/auth/[...nextauth]/route.ts`:

```ts
import { handlers } from "@/auth";
export const { GET, POST } = handlers;
```

Note: The `@/` alias maps to the project root thanks to the `tsconfig.json` paths configured by create-next-app. `auth.ts` is at the root, so `@/auth` resolves correctly.

- [ ] **Step 4: Create middleware for route protection**

Create `middleware.ts` in the project root:

```ts
export { auth as middleware } from "@/auth";

export const config = {
  matcher: ["/repos/:path*", "/settings/:path*"],
};
```

This protects all `/repos/*` and `/settings/*` routes. Unauthenticated users get redirected to sign in.

- [ ] **Step 5: Create login page**

Create `src/app/login/page.tsx`:

```tsx
import { signIn } from "@/auth";

export default function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-base">
      <div className="w-full max-w-sm rounded-md border border-border bg-mantle p-8">
        <h1 className="text-xl font-bold tracking-tight text-text">
          LBM Hub
        </h1>
        <p className="mt-1 text-sm text-subtext-0">
          Sign in with GitHub to get started
        </p>
        <form
          className="mt-6"
          action={async () => {
            "use server";
            await signIn("github", { redirectTo: "/repos" });
          }}
        >
          <button
            type="submit"
            className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-base transition-opacity hover:opacity-90"
          >
            <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            Continue with GitHub
          </button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Update root layout with SessionProvider**

Modify `src/app/layout.tsx` — wrap children with SessionProvider:

```tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { SessionProvider } from "next-auth/react";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "LBM Hub",
  description: "Portal for repos using LBM",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans antialiased">
        <SessionProvider>{children}</SessionProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 7: Update landing page to redirect if logged in**

Replace `src/app/page.tsx`:

```tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Link from "next/link";

export default async function Home() {
  const session = await auth();
  if (session) redirect("/repos");

  return (
    <div className="flex min-h-screen items-center justify-center bg-base">
      <div className="text-center">
        <h1 className="text-4xl font-extrabold tracking-tight text-text">
          LBM Hub
        </h1>
        <p className="mt-2 text-lg text-subtext-0">
          Portal for your LBM repos
        </p>
        <Link
          href="/login"
          className="mt-6 inline-block rounded-md bg-accent px-6 py-2 text-sm font-semibold text-base transition-opacity hover:opacity-90"
        >
          Get Started
        </Link>
      </div>
    </div>
  );
}
```

- [ ] **Step 8: Fill in `.env.local` with GitHub OAuth credentials**

Add the Client ID and Client Secret from the GitHub OAuth App the user created:

```
GITHUB_ID=<paste-client-id>
GITHUB_SECRET=<paste-client-secret>
```

- [ ] **Step 9: Test auth flow**

```bash
npm run dev
```

1. Open `http://localhost:3000` → see landing page
2. Click "Get Started" → login page
3. Click "Continue with GitHub" → GitHub OAuth flow
4. After auth → redirected to `/repos` (will 404 for now, that's fine)

- [ ] **Step 10: Commit**

```bash
git add auth.ts middleware.ts src/types/next-auth.d.ts src/app/api/auth src/app/login src/app/layout.tsx src/app/page.tsx
git commit -m "feat: github oauth with next-auth v5"
```

---

## Task 4: Database Schema

**Files:**
- Create: `drizzle.config.ts`
- Create: `src/db/index.ts`
- Create: `src/db/schema.ts`

- [ ] **Step 1: Create Drizzle config**

Create `drizzle.config.ts`:

```ts
import { defineConfig } from "drizzle-kit";

export default defineConfig({
  schema: "./src/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL!,
  },
});
```

- [ ] **Step 2: Create database client**

Create `src/db/index.ts`:

```ts
import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";
import * as schema from "./schema";

const sql = neon(process.env.DATABASE_URL!);
export const db = drizzle(sql, { schema });
```

- [ ] **Step 3: Create schema**

Create `src/db/schema.ts`:

```ts
import {
  pgTable,
  uuid,
  text,
  timestamp,
  bigint,
  jsonb,
  integer,
  pgEnum,
  unique,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";

export const memberRoleEnum = pgEnum("member_role", ["read", "write", "admin"]);

export const iterationStatusEnum = pgEnum("iteration_status", [
  "waiting",
  "working",
  "done",
  "merged",
]);

export const agentOutcomeEnum = pgEnum("agent_outcome", [
  "win",
  "loss",
  "no_changes",
  "error",
]);

// ── Users ──────────────────────────────────────────────

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  githubId: bigint("github_id", { mode: "number" }).unique().notNull(),
  username: text("username").notNull(),
  displayName: text("display_name"),
  avatarUrl: text("avatar_url"),
  accessToken: text("access_token").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const usersRelations = relations(users, ({ many }) => ({
  repoMembers: many(repoMembers),
}));

// ── Repos ──────────────────────────────────────────────

export const repos = pgTable("repos", {
  id: uuid("id").primaryKey().defaultRandom(),
  githubRepoId: bigint("github_repo_id", { mode: "number" }).unique().notNull(),
  owner: text("owner").notNull(),
  name: text("name").notNull(),
  lbmConfig: jsonb("lbm_config"),
  lbmConfigFetchedAt: timestamp("lbm_config_fetched_at"),
  onboardedBy: uuid("onboarded_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const reposRelations = relations(repos, ({ many, one }) => ({
  members: many(repoMembers),
  iterations: many(iterations),
  agentResults: many(agentResults),
  onboarder: one(users, {
    fields: [repos.onboardedBy],
    references: [users.id],
  }),
}));

// ── Repo Members ───────────────────────────────────────

export const repoMembers = pgTable(
  "repo_members",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    repoId: uuid("repo_id")
      .references(() => repos.id, { onDelete: "cascade" })
      .notNull(),
    userId: uuid("user_id")
      .references(() => users.id, { onDelete: "cascade" })
      .notNull(),
    role: memberRoleEnum("role").notNull().default("read"),
    syncedAt: timestamp("synced_at").defaultNow().notNull(),
  },
  (t) => [unique().on(t.repoId, t.userId)]
);

export const repoMembersRelations = relations(repoMembers, ({ one }) => ({
  repo: one(repos, { fields: [repoMembers.repoId], references: [repos.id] }),
  user: one(users, { fields: [repoMembers.userId], references: [users.id] }),
}));

// ── Iterations ─────────────────────────────────────────

export const iterations = pgTable("iterations", {
  id: uuid("id").primaryKey().defaultRandom(),
  repoId: uuid("repo_id")
    .references(() => repos.id, { onDelete: "cascade" })
    .notNull(),
  githubIssueNumber: integer("github_issue_number").notNull(),
  githubIssueUrl: text("github_issue_url").notNull(),
  title: text("title").notNull(),
  status: iterationStatusEnum("status").notNull().default("waiting"),
  winningAgent: text("winning_agent"),
  agentCount: integer("agent_count").notNull().default(0),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const iterationsRelations = relations(iterations, ({ one, many }) => ({
  repo: one(repos, { fields: [iterations.repoId], references: [repos.id] }),
  agentResults: many(agentResults),
}));

// ── Agent Results ──────────────────────────────────────

export const agentResults = pgTable("agent_results", {
  id: uuid("id").primaryKey().defaultRandom(),
  repoId: uuid("repo_id")
    .references(() => repos.id, { onDelete: "cascade" })
    .notNull(),
  iterationId: uuid("iteration_id")
    .references(() => iterations.id, { onDelete: "cascade" })
    .notNull(),
  agentLabel: text("agent_label").notNull(),
  modelId: text("model_id"),
  harness: text("harness").notNull(),
  outcome: agentOutcomeEnum("outcome").notNull(),
  prNumber: integer("pr_number"),
  prUrl: text("pr_url"),
  repairAttempts: integer("repair_attempts").notNull().default(0),
  ralphLoops: integer("ralph_loops").notNull().default(0),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const agentResultsRelations = relations(agentResults, ({ one }) => ({
  repo: one(repos, {
    fields: [agentResults.repoId],
    references: [repos.id],
  }),
  iteration: one(iterations, {
    fields: [agentResults.iterationId],
    references: [iterations.id],
  }),
}));
```

- [ ] **Step 4: Set up database**

The user needs a Neon database. Set `DATABASE_URL` in `.env.local`, then:

```bash
npx drizzle-kit push
```

Expected: Tables created in the database. Output shows each table being created.

- [ ] **Step 5: Commit**

```bash
git add drizzle.config.ts src/db/index.ts src/db/schema.ts
git commit -m "feat: drizzle schema — users, repos, iterations, agent_results"
```

---

## Task 5: User Sync on Login

**Files:**
- Modify: `auth.ts`
- Create: `src/lib/github.ts`

- [ ] **Step 1: Add user upsert on sign-in**

Update `auth.ts` — add a `signIn` callback that upserts the user record:

```ts
import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import { db } from "@/src/db";
import { users } from "@/src/db/schema";
import { eq } from "drizzle-orm";

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    GitHub({
      clientId: process.env.GITHUB_ID!,
      clientSecret: process.env.GITHUB_SECRET!,
      authorization: {
        params: {
          scope: "read:user repo",
        },
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account, profile }) {
      if (!account || !profile) return false;
      const githubId = Number(profile.id);
      const existing = await db.query.users.findFirst({
        where: eq(users.githubId, githubId),
      });
      if (existing) {
        await db
          .update(users)
          .set({
            username: profile.login as string,
            displayName: user.name ?? profile.login as string,
            avatarUrl: user.image ?? null,
            accessToken: account.access_token!,
            updatedAt: new Date(),
          })
          .where(eq(users.githubId, githubId));
      } else {
        await db.insert(users).values({
          githubId,
          username: profile.login as string,
          displayName: user.name ?? profile.login as string,
          avatarUrl: user.image ?? null,
          accessToken: account.access_token!,
        });
      }
      return true;
    },
    async jwt({ token, account, profile }) {
      if (account) {
        token.accessToken = account.access_token;
      }
      if (profile) {
        token.githubId = Number(profile.id);
        token.username = profile.login;
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken as string;
      session.user.id = token.sub!;
      session.user.username = token.username as string;
      session.user.githubId = token.githubId as number;
      return session;
    },
  },
});
```

- [ ] **Step 2: Update type augmentation**

Update `src/types/next-auth.d.ts`:

```ts
import "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken: string;
    user: {
      id: string;
      name?: string | null;
      email?: string | null;
      image?: string | null;
      username: string;
      githubId: number;
    };
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    githubId?: number;
    username?: string;
  }
}
```

- [ ] **Step 3: Create GitHub API helper**

Create `src/lib/github.ts`:

```ts
const GITHUB_API = "https://api.github.com";

export async function githubFetch(
  path: string,
  accessToken: string,
  options?: RequestInit
) {
  const res = await fetch(`${GITHUB_API}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/vnd.github+json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`GitHub API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function getUserRepos(accessToken: string) {
  return githubFetch("/user/repos?per_page=100&sort=updated", accessToken);
}

export async function getRepoContents(
  accessToken: string,
  owner: string,
  name: string,
  path: string
) {
  const data = await githubFetch(
    `/repos/${owner}/${name}/contents/${path}`,
    accessToken
  );
  if (data.content) {
    return Buffer.from(data.content, "base64").toString("utf-8");
  }
  return null;
}

export async function getRepoPermission(
  accessToken: string,
  owner: string,
  name: string,
  username: string
): Promise<"admin" | "write" | "read" | "none"> {
  try {
    const data = await githubFetch(
      `/repos/${owner}/${name}/collaborators/${username}/permission`,
      accessToken
    );
    return data.permission;
  } catch {
    return "none";
  }
}
```

- [ ] **Step 4: Test login flow with DB**

```bash
npm run dev
```

1. Open `http://localhost:3000` → click "Get Started" → login with GitHub
2. Check the database — the `users` table should have a row for your GitHub user

- [ ] **Step 5: Commit**

```bash
git add auth.ts src/types/next-auth.d.ts src/lib/github.ts
git commit -m "feat: upsert user on github login, github api helpers"
```

---

## Task 6: Layout Shell (Header + Sidebar)

**Files:**
- Create: `src/components/header.tsx`
- Create: `src/components/sidebar.tsx`
- Create: `src/components/repo-selector.tsx`
- Create: `src/app/repos/layout.tsx`
- Create: `src/app/repos/page.tsx`

- [ ] **Step 1: Create header component**

Create `src/components/header.tsx`:

```tsx
import { auth } from "@/auth";
import Link from "next/link";

export async function Header() {
  const session = await auth();

  return (
    <header className="flex h-12 items-center justify-between border-b border-border px-6">
      <div className="flex items-center gap-3.5">
        <Link href="/repos" className="flex items-center gap-2">
          <div className="h-[18px] w-[18px] rounded-[3px] bg-accent" />
          <span className="text-sm font-bold tracking-tight text-text">
            LBM Hub
          </span>
        </Link>
      </div>
      <div className="flex items-center gap-3.5">
        {session?.user && (
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-overlay-0">
              {session.user.username}
            </span>
            {session.user.image && (
              <img
                src={session.user.image}
                alt=""
                className="h-[26px] w-[26px] rounded-md"
              />
            )}
          </div>
        )}
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Create sidebar component**

Create `src/components/sidebar.tsx`:

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/src/lib/utils";

interface NavItem {
  label: string;
  href: string;
}

interface SidebarProps {
  items: NavItem[];
  repoList?: { owner: string; name: string }[];
}

export function Sidebar({ items, repoList }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className="flex w-[190px] flex-shrink-0 flex-col border-r border-border">
      <nav className="flex flex-col gap-0.5 p-4">
        {items.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/repos" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs",
                isActive
                  ? "border-l-2 border-accent bg-accent/8 font-semibold text-text"
                  : "text-overlay-0 hover:text-subtext-0"
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
      {repoList && repoList.length > 0 && (
        <div className="mt-auto border-t border-border p-4">
          <div className="mb-2 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
            Repos
          </div>
          {repoList.map((repo) => (
            <Link
              key={`${repo.owner}/${repo.name}`}
              href={`/repos/${repo.owner}/${repo.name}`}
              className="block py-0.5 font-mono text-[11px] text-overlay-0 hover:text-subtext-0"
            >
              {repo.owner}/{repo.name}
            </Link>
          ))}
        </div>
      )}
    </aside>
  );
}
```

- [ ] **Step 3: Create repos layout**

Create `src/app/repos/layout.tsx`:

```tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import { db } from "@/src/db";
import { repos, repoMembers, users } from "@/src/db/schema";
import { eq } from "drizzle-orm";
import { Header } from "@/src/components/header";
import { Sidebar } from "@/src/components/sidebar";

const TOP_LEVEL_NAV = [
  { label: "Repos", href: "/repos" },
  { label: "Settings", href: "/settings" },
];

export default async function ReposLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session) redirect("/login");

  const dbUser = await db.query.users.findFirst({
    where: eq(users.githubId, session.user.githubId),
  });

  let repoList: { owner: string; name: string }[] = [];
  if (dbUser) {
    const memberships = await db.query.repoMembers.findMany({
      where: eq(repoMembers.userId, dbUser.id),
      with: { repo: true },
    });
    repoList = memberships.map((m) => ({
      owner: m.repo.owner,
      name: m.repo.name,
    }));
  }

  return (
    <div className="flex h-screen flex-col bg-base">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar items={TOP_LEVEL_NAV} repoList={repoList} />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create repos page (placeholder grid)**

Create `src/app/repos/page.tsx`:

```tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import { db } from "@/src/db";
import { repoMembers, users } from "@/src/db/schema";
import { eq } from "drizzle-orm";
import Link from "next/link";

export default async function ReposPage() {
  const session = await auth();
  if (!session) redirect("/login");

  const dbUser = await db.query.users.findFirst({
    where: eq(users.githubId, session.user.githubId),
  });

  if (!dbUser) redirect("/login");

  const memberships = await db.query.repoMembers.findMany({
    where: eq(repoMembers.userId, dbUser.id),
    with: { repo: true },
  });

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-tight text-text">Repos</h1>
        <Link
          href="/repos/onboard"
          className="rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-base"
        >
          Add a Repo
        </Link>
      </div>
      {memberships.length === 0 ? (
        <div className="rounded-md border border-border p-8 text-center">
          <p className="text-sm text-subtext-0">No repos onboarded yet.</p>
          <Link
            href="/repos/onboard"
            className="mt-2 inline-block text-sm text-accent hover:underline"
          >
            Onboard your first repo →
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {memberships.map((m) => (
            <Link
              key={m.repo.id}
              href={`/repos/${m.repo.owner}/${m.repo.name}`}
              className="rounded-md border border-border p-4 transition-colors hover:border-surface-1"
            >
              <div className="text-sm font-semibold text-text">
                {m.repo.owner}/{m.repo.name}
              </div>
              <div className="mt-1 font-mono text-xs text-overlay-0">
                {m.role}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Verify layout**

```bash
npm run dev
```

Log in and visit `/repos`. Confirm: header with logo + username, sidebar with Repos/Settings nav, empty repos grid with "Onboard your first repo" link.

- [ ] **Step 6: Commit**

```bash
git add src/components/header.tsx src/components/sidebar.tsx src/app/repos/layout.tsx src/app/repos/page.tsx
git commit -m "feat: layout shell — header, sidebar, repos page"
```

---

## Task 7: Repo Onboarding

**Files:**
- Create: `src/app/repos/onboard/page.tsx`

- [ ] **Step 1: Create onboard page**

Create `src/app/repos/onboard/page.tsx`:

```tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import { db } from "@/src/db";
import { repos, repoMembers, users } from "@/src/db/schema";
import { eq } from "drizzle-orm";
import { getUserRepos, getRepoContents, getRepoPermission } from "@/src/lib/github";
import { revalidatePath } from "next/cache";

export default async function OnboardPage() {
  const session = await auth();
  if (!session) redirect("/login");

  const githubRepos = await getUserRepos(session.accessToken);

  const existingRepos = await db.query.repos.findMany();
  const existingIds = new Set(existingRepos.map((r) => r.githubRepoId));

  const available = (githubRepos as any[]).filter(
    (r: any) =>
      !existingIds.has(r.id) &&
      (r.permissions?.admin || r.permissions?.push)
  );

  async function onboard(formData: FormData) {
    "use server";
    const repoFullName = formData.get("repo") as string;
    const [owner, name] = repoFullName.split("/");

    const session = await auth();
    if (!session) return;

    const ghRepo = (githubRepos as any[]).find(
      (r: any) => r.full_name === repoFullName
    );
    if (!ghRepo) return;

    let lbmConfig = null;
    try {
      const content = await getRepoContents(
        session.accessToken,
        owner,
        name,
        "lbm.toml"
      );
      if (content) {
        lbmConfig = { raw: content };
      }
    } catch {
      // lbm.toml not found — that's ok
    }

    const dbUser = await db.query.users.findFirst({
      where: eq(users.githubId, session.user.githubId),
    });
    if (!dbUser) return;

    const [newRepo] = await db
      .insert(repos)
      .values({
        githubRepoId: ghRepo.id,
        owner,
        name,
        lbmConfig,
        lbmConfigFetchedAt: new Date(),
        onboardedBy: dbUser.id,
      })
      .returning();

    const permission = await getRepoPermission(
      session.accessToken,
      owner,
      name,
      session.user.username
    );
    const role =
      permission === "admin" ? "admin" : permission === "write" ? "write" : "read";

    await db.insert(repoMembers).values({
      repoId: newRepo.id,
      userId: dbUser.id,
      role,
    });

    revalidatePath("/repos");
    redirect(`/repos/${owner}/${name}`);
  }

  return (
    <div className="mx-auto max-w-lg">
      <h1 className="text-xl font-bold tracking-tight text-text">
        Add a Repo
      </h1>
      <p className="mt-1 text-sm text-subtext-0">
        Select a repo with LBM installed. You need write access.
      </p>
      {available.length === 0 ? (
        <div className="mt-6 rounded-md border border-border p-6 text-center">
          <p className="text-sm text-overlay-0">
            No eligible repos found. Make sure you have write access and lbm.toml
            is set up.
          </p>
        </div>
      ) : (
        <form action={onboard} className="mt-6 flex flex-col gap-3">
          {available.map((repo: any) => (
            <label
              key={repo.id}
              className="flex cursor-pointer items-center gap-3 rounded-md border border-border p-3 transition-colors hover:border-surface-1 has-[:checked]:border-accent has-[:checked]:bg-accent/5"
            >
              <input
                type="radio"
                name="repo"
                value={repo.full_name}
                className="accent-accent"
              />
              <div>
                <div className="text-sm font-semibold text-text">
                  {repo.full_name}
                </div>
                <div className="font-mono text-xs text-overlay-0">
                  {repo.private ? "private" : "public"}
                </div>
              </div>
            </label>
          ))}
          <button
            type="submit"
            className="mt-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-base transition-opacity hover:opacity-90"
          >
            Onboard Repo
          </button>
        </form>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Test onboarding**

```bash
npm run dev
```

1. Go to `/repos` → click "Add a Repo"
2. See list of your GitHub repos with write access
3. Select one → click "Onboard Repo"
4. Should redirect to `/repos/[owner]/[name]` (will 404 — next task)
5. Check DB — `repos` and `repo_members` tables should have rows

- [ ] **Step 3: Commit**

```bash
git add src/app/repos/onboard/page.tsx
git commit -m "feat: repo onboarding — select github repo, fetch lbm.toml, create db records"
```

---

## Task 8: Repo-Scoped Layout (Context Switching Sidebar)

**Files:**
- Create: `src/app/repos/[owner]/[name]/layout.tsx`

- [ ] **Step 1: Create repo-scoped layout**

Create `src/app/repos/[owner]/[name]/layout.tsx`:

```tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import { db } from "@/src/db";
import { repos } from "@/src/db/schema";
import { eq, and } from "drizzle-orm";
import { Header } from "@/src/components/header";
import { Sidebar } from "@/src/components/sidebar";

interface Props {
  children: React.ReactNode;
  params: Promise<{ owner: string; name: string }>;
}

export default async function RepoLayout({ children, params }: Props) {
  const session = await auth();
  if (!session) redirect("/login");

  const { owner, name } = await params;

  const repo = await db.query.repos.findFirst({
    where: and(eq(repos.owner, owner), eq(repos.name, name)),
  });

  if (!repo) redirect("/repos");

  const basePath = `/repos/${owner}/${name}`;

  const repoNav = [
    { label: "Overview", href: basePath },
    { label: "Iterations", href: `${basePath}/iterations` },
    { label: "Agents", href: `${basePath}/agents` },
    { label: "Settings", href: `${basePath}/settings` },
  ];

  return (
    <div className="flex h-screen flex-col bg-base">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar items={repoNav} />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/repos/\\[owner\\]/\\[name\\]/layout.tsx
git commit -m "feat: repo-scoped layout with context-switching sidebar"
```

---

## Task 9: Repo Overview Page

**Files:**
- Create: `src/app/repos/[owner]/[name]/page.tsx`
- Create: `src/components/leaderboard-card.tsx`
- Create: `src/components/config-card.tsx`
- Create: `src/components/iterations-table.tsx`
- Create: `src/components/status-light.tsx`

- [ ] **Step 1: Create status light component**

Create `src/components/status-light.tsx`:

```tsx
import { cn } from "@/src/lib/utils";

const STATUS_COLORS = {
  waiting: "bg-status-waiting",
  working: "bg-status-working shadow-[0_0_6px_var(--color-status-working)]",
  done: "bg-status-done shadow-[0_0_6px_var(--color-status-done)]",
  merged: "bg-status-merged shadow-[0_0_6px_var(--color-status-merged)]",
} as const;

interface StatusLightProps {
  status: keyof typeof STATUS_COLORS;
  className?: string;
}

export function StatusLight({ status, className }: StatusLightProps) {
  return (
    <div
      className={cn(
        "h-2 w-2 flex-shrink-0 rounded-full",
        STATUS_COLORS[status],
        className
      )}
    />
  );
}
```

- [ ] **Step 2: Create leaderboard card**

Create `src/components/leaderboard-card.tsx`:

```tsx
interface Agent {
  label: string;
  harness: string;
  modelId: string | null;
  wins: number;
  losses: number;
  winRate: number;
}

interface LeaderboardCardProps {
  agents: Agent[];
}

export function LeaderboardCard({ agents }: LeaderboardCardProps) {
  return (
    <div className="rounded-md border border-border p-4">
      <div className="mb-3 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
        Leaderboard
      </div>
      <div className="flex flex-col gap-1">
        {agents.map((agent, i) => {
          const isLeader = i === 0 && agent.wins > 0;
          const total = agent.wins + agent.losses;
          const winPct = total > 0 ? (agent.wins / total) * 100 : 0;
          return (
            <div
              key={agent.label}
              className={
                isLeader
                  ? "flex items-center gap-2 rounded-[3px] border border-accent/10 bg-accent/5 px-2.5 py-1.5"
                  : "flex items-center gap-2 px-2.5 py-1.5"
              }
            >
              <span className="w-3.5 font-mono text-[9px] font-extrabold text-surface-2">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span
                className={`w-[68px] text-[11px] font-${isLeader ? "bold" : "medium"} ${isLeader ? "text-text" : "text-subtext-0"}`}
              >
                {agent.harness.charAt(0).toUpperCase() + agent.harness.slice(1)}
              </span>
              <span className="w-[52px] font-mono text-[9px] text-overlay-0">
                {agent.modelId?.split("-").slice(-2).join("-") ?? "unknown"}
              </span>
              <div className="flex flex-1 overflow-hidden rounded-sm">
                {total > 0 ? (
                  <>
                    <div
                      className={`h-3.5 ${isLeader ? "bg-accent" : "bg-surface-2"}`}
                      style={{ width: `${winPct}%` }}
                    />
                    <div
                      className="h-3.5 bg-surface-0"
                      style={{ width: `${100 - winPct}%` }}
                    />
                  </>
                ) : (
                  <div className="h-3.5 w-full bg-surface-0" />
                )}
              </div>
              <span
                className={`w-[34px] text-right text-[13px] font-extrabold tracking-tight ${isLeader ? "text-text" : "text-surface-2"}`}
              >
                {Math.round(agent.winRate)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create config card**

Create `src/components/config-card.tsx`:

```tsx
interface ConfigCardProps {
  config: Record<string, any> | null;
}

export function ConfigCard({ config }: ConfigCardProps) {
  if (!config?.raw && !config) {
    return (
      <div className="rounded-md border border-border p-4">
        <div className="mb-3 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
          Config
        </div>
        <p className="text-xs text-overlay-0">No lbm.toml found</p>
      </div>
    );
  }

  const raw = config.raw ?? JSON.stringify(config, null, 2);
  const lines: string[] = raw.split("\n");

  return (
    <div className="rounded-md border border-border p-4">
      <div className="mb-3 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
        Config
      </div>
      <pre className="font-mono text-[10px] leading-[1.8] text-subtext-0">
        {lines.map((line: string, i: number) => {
          if (line.startsWith("#") || line.startsWith("//")) {
            return (
              <div key={i} className="text-surface-2">
                {line}
              </div>
            );
          }
          const eqIdx = line.indexOf("=");
          if (eqIdx > -1) {
            const key = line.slice(0, eqIdx).trim();
            const val = line.slice(eqIdx + 1).trim();
            return (
              <div key={i}>
                <span className="text-blue">{key}</span>
                <span className="text-overlay-0"> = </span>
                <span className="text-green">{val}</span>
              </div>
            );
          }
          if (line.startsWith("[")) {
            return (
              <div key={i} className="mt-1 text-peach">
                {line}
              </div>
            );
          }
          return <div key={i}>{line}</div>;
        })}
      </pre>
    </div>
  );
}
```

- [ ] **Step 4: Create iterations table**

Create `src/components/iterations-table.tsx`:

```tsx
import Link from "next/link";
import { StatusLight } from "./status-light";

interface Iteration {
  id: string;
  title: string;
  githubIssueNumber: number;
  githubIssueUrl: string;
  status: "waiting" | "working" | "done" | "merged";
  agentCount: number;
  updatedAt: Date;
}

interface IterationsTableProps {
  iterations: Iteration[];
  repoPath: string;
}

function timeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

export function IterationsTable({
  iterations,
  repoPath,
}: IterationsTableProps) {
  return (
    <div className="rounded-md border border-border">
      <div className="flex items-center border-b border-border bg-mantle px-4 py-2">
        <span className="flex-1 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
          Iterations
        </span>
        <a
          href={`https://github.com/${repoPath}/issues`}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-[10px] text-accent hover:underline"
        >
          View on GitHub ↗
        </a>
      </div>
      {iterations.length === 0 ? (
        <div className="p-6 text-center text-xs text-overlay-0">
          No iterations yet
        </div>
      ) : (
        iterations.map((iter) => (
          <Link
            key={iter.id}
            href={`/repos/${repoPath}/iterations/${iter.id}`}
            className="flex items-center border-b border-border/50 px-4 py-2.5 transition-colors last:border-b-0 hover:bg-surface-0/30"
          >
            <StatusLight status={iter.status} />
            <span className="ml-3 flex-1 text-xs font-medium text-text/75">
              {iter.title}
              <a
                href={iter.githubIssueUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="ml-2 font-mono text-[10px] text-overlay-0 hover:text-accent"
              >
                #{iter.githubIssueNumber}
              </a>
            </span>
            <span className="w-[70px] font-mono text-[10px] text-subtext-0">
              {iter.status}
            </span>
            <span className="w-[50px] font-mono text-[10px] text-overlay-0">
              {iter.agentCount} agents
            </span>
            <span className="w-[36px] text-right font-mono text-[10px] text-surface-2">
              {timeAgo(iter.updatedAt)}
            </span>
          </Link>
        ))
      )}
    </div>
  );
}
```

- [ ] **Step 5: Create repo overview page**

Create `src/app/repos/[owner]/[name]/page.tsx`:

```tsx
import { db } from "@/src/db";
import { repos, iterations, agentResults } from "@/src/db/schema";
import { eq, and, desc } from "drizzle-orm";
import { redirect } from "next/navigation";
import { LeaderboardCard } from "@/src/components/leaderboard-card";
import { ConfigCard } from "@/src/components/config-card";
import { IterationsTable } from "@/src/components/iterations-table";

interface Props {
  params: Promise<{ owner: string; name: string }>;
}

export default async function RepoOverview({ params }: Props) {
  const { owner, name } = await params;
  const repoPath = `${owner}/${name}`;

  const repo = await db.query.repos.findFirst({
    where: and(eq(repos.owner, owner), eq(repos.name, name)),
  });

  if (!repo) redirect("/repos");

  const repoIterations = await db.query.iterations.findMany({
    where: eq(iterations.repoId, repo.id),
    orderBy: desc(iterations.createdAt),
    limit: 20,
  });

  const results = await db.query.agentResults.findMany({
    where: eq(agentResults.repoId, repo.id),
  });

  // Compute leaderboard
  const agentMap = new Map<
    string,
    { harness: string; modelId: string | null; wins: number; losses: number }
  >();
  for (const r of results) {
    const existing = agentMap.get(r.agentLabel) ?? {
      harness: r.harness,
      modelId: r.modelId,
      wins: 0,
      losses: 0,
    };
    if (r.outcome === "win") existing.wins++;
    else existing.losses++;
    agentMap.set(r.agentLabel, existing);
  }
  const agents = Array.from(agentMap.entries())
    .map(([label, data]) => ({
      label,
      ...data,
      winRate:
        data.wins + data.losses > 0
          ? (data.wins / (data.wins + data.losses)) * 100
          : 0,
    }))
    .sort((a, b) => b.winRate - a.winRate);

  return (
    <div>
      <div className="mb-1 text-[22px] font-extrabold tracking-tight text-text">
        {repoPath}
      </div>
      <div className="mb-6 font-mono text-[11px] text-overlay-0">
        {agents.length} agents · {repoIterations.length} iterations
      </div>

      <div className="mb-4 grid grid-cols-[260px_1fr] gap-3">
        <LeaderboardCard agents={agents} />
        <ConfigCard config={repo.lbmConfig as Record<string, any> | null} />
      </div>

      <IterationsTable iterations={repoIterations} repoPath={repoPath} />
    </div>
  );
}
```

- [ ] **Step 6: Verify in browser**

```bash
npm run dev
```

Navigate to an onboarded repo. Confirm: repo name + metadata, leaderboard card (empty if no data), config card (shows lbm.toml if found), iterations table (empty).

- [ ] **Step 7: Commit**

```bash
git add src/components/status-light.tsx src/components/leaderboard-card.tsx src/components/config-card.tsx src/components/iterations-table.tsx src/app/repos/\\[owner\\]/\\[name\\]/page.tsx
git commit -m "feat: repo overview — leaderboard, config card, iterations table"
```

---

## Task 10: Iteration Detail Page

**Files:**
- Create: `src/app/repos/[owner]/[name]/iterations/page.tsx`
- Create: `src/app/repos/[owner]/[name]/iterations/[id]/page.tsx`

- [ ] **Step 1: Create iterations list page**

Create `src/app/repos/[owner]/[name]/iterations/page.tsx`:

```tsx
import { db } from "@/src/db";
import { repos, iterations } from "@/src/db/schema";
import { eq, and, desc } from "drizzle-orm";
import { redirect } from "next/navigation";
import { IterationsTable } from "@/src/components/iterations-table";

interface Props {
  params: Promise<{ owner: string; name: string }>;
}

export default async function IterationsPage({ params }: Props) {
  const { owner, name } = await params;
  const repoPath = `${owner}/${name}`;

  const repo = await db.query.repos.findFirst({
    where: and(eq(repos.owner, owner), eq(repos.name, name)),
  });
  if (!repo) redirect("/repos");

  const allIterations = await db.query.iterations.findMany({
    where: eq(iterations.repoId, repo.id),
    orderBy: desc(iterations.createdAt),
  });

  return (
    <div>
      <h1 className="mb-6 text-xl font-bold tracking-tight text-text">
        Iterations
      </h1>
      <IterationsTable iterations={allIterations} repoPath={repoPath} />
    </div>
  );
}
```

- [ ] **Step 2: Create iteration detail page**

Create `src/app/repos/[owner]/[name]/iterations/[id]/page.tsx`:

```tsx
import { db } from "@/src/db";
import { iterations, agentResults } from "@/src/db/schema";
import { eq } from "drizzle-orm";
import { notFound } from "next/navigation";
import { StatusLight } from "@/src/components/status-light";

interface Props {
  params: Promise<{ owner: string; name: string; id: string }>;
}

export default async function IterationDetailPage({ params }: Props) {
  const { owner, name, id } = await params;

  const iteration = await db.query.iterations.findFirst({
    where: eq(iterations.id, id),
  });
  if (!iteration) notFound();

  const results = await db.query.agentResults.findMany({
    where: eq(agentResults.iterationId, id),
  });

  return (
    <div>
      <div className="mb-1 flex items-center gap-3">
        <StatusLight status={iteration.status} className="h-2.5 w-2.5" />
        <h1 className="text-xl font-bold tracking-tight text-text">
          {iteration.title}
        </h1>
      </div>
      <div className="mb-6 flex items-center gap-3 font-mono text-xs text-overlay-0">
        <a
          href={iteration.githubIssueUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          #{iteration.githubIssueNumber} on GitHub ↗
        </a>
        <span>·</span>
        <span>{iteration.status}</span>
        {iteration.winningAgent && (
          <>
            <span>·</span>
            <span>Winner: {iteration.winningAgent}</span>
          </>
        )}
      </div>

      <div className="mb-3 font-mono text-[9px] font-bold uppercase tracking-widest text-surface-2">
        Agent Results
      </div>
      {results.length === 0 ? (
        <div className="rounded-md border border-border p-6 text-center text-xs text-overlay-0">
          No agent results yet
        </div>
      ) : (
        <div className="rounded-md border border-border">
          {results.map((r) => (
            <div
              key={r.id}
              className="flex items-center border-b border-border/50 px-4 py-3 last:border-b-0"
            >
              <div className="flex-1">
                <div className="text-sm font-medium text-text">
                  {r.harness.charAt(0).toUpperCase() + r.harness.slice(1)}
                  <span className="ml-2 font-mono text-xs text-overlay-0">
                    {r.modelId}
                  </span>
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-surface-2">
                  {r.repairAttempts > 0 && `${r.repairAttempts} repairs`}
                  {r.ralphLoops > 0 && ` · ${r.ralphLoops} ralph loops`}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {r.prUrl && (
                  <a
                    href={r.prUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-mono text-xs text-accent hover:underline"
                  >
                    PR #{r.prNumber} ↗
                  </a>
                )}
                <span
                  className={`rounded-sm px-2 py-0.5 font-mono text-[10px] font-semibold ${
                    r.outcome === "win"
                      ? "bg-green/10 text-green"
                      : r.outcome === "loss"
                        ? "bg-red/10 text-red"
                        : r.outcome === "error"
                          ? "bg-red/10 text-red"
                          : "bg-surface-0 text-overlay-0"
                  }`}
                >
                  {r.outcome}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/app/repos/\\[owner\\]/\\[name\\]/iterations/
git commit -m "feat: iterations list + detail page with agent results"
```

---

## Task 11: Agents Page

**Files:**
- Create: `src/app/repos/[owner]/[name]/agents/page.tsx`

- [ ] **Step 1: Create agents page**

Create `src/app/repos/[owner]/[name]/agents/page.tsx`:

```tsx
import { db } from "@/src/db";
import { repos, agentResults } from "@/src/db/schema";
import { eq, and } from "drizzle-orm";
import { redirect } from "next/navigation";

interface Props {
  params: Promise<{ owner: string; name: string }>;
}

export default async function AgentsPage({ params }: Props) {
  const { owner, name } = await params;

  const repo = await db.query.repos.findFirst({
    where: and(eq(repos.owner, owner), eq(repos.name, name)),
  });
  if (!repo) redirect("/repos");

  const results = await db.query.agentResults.findMany({
    where: eq(agentResults.repoId, repo.id),
  });

  const agentMap = new Map<
    string,
    {
      harness: string;
      modelId: string | null;
      wins: number;
      losses: number;
      noChanges: number;
      errors: number;
      totalRepairs: number;
      totalRalphs: number;
    }
  >();

  for (const r of results) {
    const existing = agentMap.get(r.agentLabel) ?? {
      harness: r.harness,
      modelId: r.modelId,
      wins: 0,
      losses: 0,
      noChanges: 0,
      errors: 0,
      totalRepairs: 0,
      totalRalphs: 0,
    };
    if (r.outcome === "win") existing.wins++;
    else if (r.outcome === "loss") existing.losses++;
    else if (r.outcome === "no_changes") existing.noChanges++;
    else if (r.outcome === "error") existing.errors++;
    existing.totalRepairs += r.repairAttempts;
    existing.totalRalphs += r.ralphLoops;
    agentMap.set(r.agentLabel, existing);
  }

  const agents = Array.from(agentMap.entries())
    .map(([label, data]) => {
      const total = data.wins + data.losses;
      return {
        label,
        ...data,
        winRate: total > 0 ? (data.wins / total) * 100 : 0,
      };
    })
    .sort((a, b) => b.winRate - a.winRate);

  return (
    <div>
      <h1 className="mb-6 text-xl font-bold tracking-tight text-text">
        Agents
      </h1>
      {agents.length === 0 ? (
        <div className="rounded-md border border-border p-6 text-center text-xs text-overlay-0">
          No agent data yet
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {agents.map((agent, i) => (
            <div key={agent.label} className="rounded-md border border-border p-4">
              <div className="flex items-center gap-3">
                <span className="font-mono text-[9px] font-extrabold text-surface-2">
                  #{i + 1}
                </span>
                <div>
                  <div className="text-sm font-bold text-text">
                    {agent.harness.charAt(0).toUpperCase() + agent.harness.slice(1)}
                  </div>
                  <div className="font-mono text-[10px] text-overlay-0">
                    {agent.modelId ?? "unknown"}
                  </div>
                </div>
                <span className="ml-auto text-lg font-extrabold tracking-tight text-text">
                  {Math.round(agent.winRate)}%
                </span>
              </div>
              <div className="mt-3 grid grid-cols-4 gap-2 font-mono text-[10px]">
                <div>
                  <div className="text-surface-2">Wins</div>
                  <div className="text-green">{agent.wins}</div>
                </div>
                <div>
                  <div className="text-surface-2">Losses</div>
                  <div className="text-red">{agent.losses}</div>
                </div>
                <div>
                  <div className="text-surface-2">Repairs</div>
                  <div className="text-yellow">{agent.totalRepairs}</div>
                </div>
                <div>
                  <div className="text-surface-2">Ralphs</div>
                  <div className="text-peach">{agent.totalRalphs}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/repos/\\[owner\\]/\\[name\\]/agents/page.tsx
git commit -m "feat: agents page — per-agent stats cards"
```

---

## Task 12: Settings Pages

**Files:**
- Create: `src/app/repos/[owner]/[name]/settings/page.tsx`
- Create: `src/app/settings/page.tsx`
- Create: `src/app/settings/layout.tsx`

- [ ] **Step 1: Create repo settings page**

Create `src/app/repos/[owner]/[name]/settings/page.tsx`:

```tsx
import { auth } from "@/auth";
import { db } from "@/src/db";
import { repos } from "@/src/db/schema";
import { eq, and } from "drizzle-orm";
import { redirect } from "next/navigation";
import { getRepoContents } from "@/src/lib/github";
import { revalidatePath } from "next/cache";

interface Props {
  params: Promise<{ owner: string; name: string }>;
}

export default async function RepoSettingsPage({ params }: Props) {
  const { owner, name } = await params;

  const session = await auth();
  if (!session) redirect("/login");

  const repo = await db.query.repos.findFirst({
    where: and(eq(repos.owner, owner), eq(repos.name, name)),
  });
  if (!repo) redirect("/repos");

  async function refreshConfig() {
    "use server";
    const session = await auth();
    if (!session) return;

    let lbmConfig = null;
    try {
      const content = await getRepoContents(
        session.accessToken,
        owner,
        name,
        "lbm.toml"
      );
      if (content) lbmConfig = { raw: content };
    } catch {
      // not found
    }

    await db
      .update(repos)
      .set({ lbmConfig, lbmConfigFetchedAt: new Date(), updatedAt: new Date() })
      .where(eq(repos.id, repo!.id));

    revalidatePath(`/repos/${owner}/${name}`);
  }

  async function removeRepo() {
    "use server";
    await db.delete(repos).where(eq(repos.id, repo!.id));
    redirect("/repos");
  }

  return (
    <div className="max-w-lg">
      <h1 className="mb-6 text-xl font-bold tracking-tight text-text">
        Repo Settings
      </h1>

      <div className="mb-6 rounded-md border border-border p-4">
        <h2 className="mb-1 text-sm font-semibold text-text">
          Refresh Config
        </h2>
        <p className="mb-3 text-xs text-overlay-0">
          Re-fetch lbm.toml from the repo.
          {repo.lbmConfigFetchedAt && (
            <span>
              {" "}Last fetched:{" "}
              {new Date(repo.lbmConfigFetchedAt).toLocaleDateString()}
            </span>
          )}
        </p>
        <form action={refreshConfig}>
          <button
            type="submit"
            className="rounded-md bg-surface-0 px-3 py-1.5 text-xs font-medium text-text hover:bg-surface-1"
          >
            Refresh
          </button>
        </form>
      </div>

      <div className="rounded-md border border-red/20 p-4">
        <h2 className="mb-1 text-sm font-semibold text-red">
          Remove from LBM Hub
        </h2>
        <p className="mb-3 text-xs text-overlay-0">
          This removes the repo from LBM Hub. It does not affect the repo on
          GitHub.
        </p>
        <form action={removeRepo}>
          <button
            type="submit"
            className="rounded-md bg-red/10 px-3 py-1.5 text-xs font-medium text-red hover:bg-red/20"
          >
            Remove Repo
          </button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create user settings layout + page**

Create `src/app/settings/layout.tsx`:

```tsx
import { Header } from "@/src/components/header";
import { Sidebar } from "@/src/components/sidebar";

const TOP_LEVEL_NAV = [
  { label: "Repos", href: "/repos" },
  { label: "Settings", href: "/settings" },
];

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen flex-col bg-base">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar items={TOP_LEVEL_NAV} />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
```

Create `src/app/settings/page.tsx`:

```tsx
import { auth, signOut } from "@/auth";
import { redirect } from "next/navigation";

export default async function UserSettingsPage() {
  const session = await auth();
  if (!session) redirect("/login");

  return (
    <div className="max-w-lg">
      <h1 className="mb-6 text-xl font-bold tracking-tight text-text">
        Settings
      </h1>

      <div className="mb-6 rounded-md border border-border p-4">
        <h2 className="mb-3 text-sm font-semibold text-text">Account</h2>
        <div className="flex items-center gap-3">
          {session.user.image && (
            <img
              src={session.user.image}
              alt=""
              className="h-10 w-10 rounded-md"
            />
          )}
          <div>
            <div className="text-sm font-medium text-text">
              {session.user.name}
            </div>
            <div className="font-mono text-xs text-overlay-0">
              @{session.user.username}
            </div>
          </div>
        </div>
      </div>

      <form
        action={async () => {
          "use server";
          await signOut({ redirectTo: "/" });
        }}
      >
        <button
          type="submit"
          className="rounded-md bg-surface-0 px-3 py-1.5 text-xs font-medium text-text hover:bg-surface-1"
        >
          Sign Out
        </button>
      </form>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/app/repos/\\[owner\\]/\\[name\\]/settings/page.tsx src/app/settings/
git commit -m "feat: repo settings + user settings pages"
```

---

## Task 13: Final Polish + Verification

- [ ] **Step 1: Run the full app and test every route**

```bash
npm run dev
```

Test this flow:
1. `http://localhost:3000` — landing page, "Get Started" link
2. Login with GitHub — redirects to `/repos`
3. `/repos` — empty state with "Add a Repo" button
4. Click "Add a Repo" → `/repos/onboard` — see repo list
5. Select a repo → onboard → redirect to `/repos/[owner]/[name]`
6. Repo overview — leaderboard (empty), config card (shows lbm.toml), iterations (empty)
7. Sidebar switches to repo-scoped nav (Overview, Iterations, Agents, Settings)
8. Click through each nav item
9. `/settings` — user account info, sign out button
10. Sign out → back to landing

- [ ] **Step 2: Run type check**

```bash
npx tsc --noEmit
```

Fix any type errors.

- [ ] **Step 3: Run linter**

```bash
npm run lint
```

Fix any lint issues.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: type and lint fixes"
```

---

## Summary

This plan implements Phase 1 of LBM Hub across 13 tasks:

| Task | What it builds |
|------|---------------|
| 1 | Next.js scaffold + dependencies |
| 2 | Catppuccin Mocha theme + fonts |
| 3 | GitHub OAuth with NextAuth v5 |
| 4 | Drizzle schema (users, repos, iterations, agent_results) |
| 5 | User sync on login + GitHub API helpers |
| 6 | Layout shell (header + sidebar) |
| 7 | Repo onboarding flow |
| 8 | Repo-scoped layout (context-switching sidebar) |
| 9 | Repo overview page (leaderboard + config + iterations) |
| 10 | Iteration detail page |
| 11 | Agents page |
| 12 | Settings pages (repo + user) |
| 13 | Final polish + verification |
