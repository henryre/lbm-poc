"""Typed dataclass config models for lbm.toml configuration.

Provides AgentConfig, ChecksConfig, LLMConfig, and LBMConfig.
LBMConfig.from_parsed_toml resolves agents from [harnesses] and [[agents]]
sections, deriving labels, branch prefixes, and name letters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

AGENT_NAME_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ACTIVE_DEPLOY_PLATFORMS = ("fly", "railway")


@dataclass(frozen=True)
class AgentConfig:
    label: str
    harness: str
    model_id: str
    model_label: str
    branch_prefix: str
    name: str
    mention: str
    # Optional freeform args appended verbatim to the harness CLI invocation
    # (e.g. passed through to claude-code-action's `claude_args`). Lets a repo set
    # model/API-specific flags in lbm.toml without an lbm-poc code change.
    claude_args: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> AgentConfig:
        return cls(
            label=d["label"],
            harness=d["harness"],
            model_id=d["model_id"],
            model_label=d["model_label"],
            branch_prefix=d["branch_prefix"],
            name=d["name"],
            mention=d["mention"],
            claude_args=d.get("claude_args", ""),
        )


@dataclass(frozen=True)
class ChecksConfig:
    required: list[str] = field(default_factory=list)
    repair_from: list[str] = field(default_factory=list)
    max_repair_attempts: int = 10
    max_ralph_loops: int = 0
    repair_comment_triggers: list[str] = field(default_factory=list)
    preview_comment_marker: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> ChecksConfig:
        return cls(
            required=d.get("required", []),
            repair_from=d.get("repair_from", []),
            max_repair_attempts=d.get("max_repair_attempts", 10),
            max_ralph_loops=d.get("max_ralph_loops", 0),
            repair_comment_triggers=d.get("repair_comment_triggers", []),
            preview_comment_marker=d.get("preview_comment_marker", ""),
        )


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "anthropic"
    summary_model: str = "claude-sonnet-4-6"

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(
            provider=d.get("provider", "anthropic"),
            summary_model=d.get("summary_model", "claude-sonnet-4-6"),
        )


@dataclass(frozen=True)
class PlanConfig:
    """Opt-in two-phase iterations (plan phase -> implement phase).

    When ``enabled``, each agent first proposes a plan; a winner is merged into
    ``dir/issue-<N>/plan.md`` (via /merge-plan) and agents then implement against
    it. ``feedback_revs`` caps plan feedback rounds; ``prototype`` (fast-follow)
    allows a prototype run before planning.
    """

    enabled: bool = False
    dir: str = "lbm-plans"
    feedback_revs: int = 1
    prototype: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> PlanConfig:
        return cls(
            enabled=d.get("enabled", False),
            dir=d.get("dir", "lbm-plans"),
            feedback_revs=d.get("feedback_revs", 1),
            prototype=d.get("prototype", False),
        )


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


@dataclass
class LBMConfig:
    agents: list[AgentConfig]
    checks: ChecksConfig
    llm: LLMConfig
    deploy: DeployConfig
    plan: PlanConfig = field(default_factory=PlanConfig)

    @classmethod
    def from_parsed_toml(cls, raw: dict) -> LBMConfig:
        """Build LBMConfig from a raw parsed TOML dict.

        Resolves agents from [harnesses] and [[agents]] sections, deriving
        default label, branch_prefix, and name letter for each agent.
        Validates no duplicate branch prefixes and that all harnesses are defined.
        """
        harnesses = raw.get("harnesses", {})
        agents_raw = raw.get("agents", [])

        agents: list[AgentConfig] = []
        seen_prefixes: set[str] = set()

        for i, entry in enumerate(agents_raw):
            harness = entry["harness"]
            model_id = entry["model_id"]
            model_label = entry["model_label"]

            if harness not in harnesses:
                raise ValueError(f"Harness '{harness}' not defined in [harnesses]. Available: {list(harnesses.keys())}")

            default_label = f"agent:{harness}-{model_label}"
            default_prefix = f"{harness}-{model_label}/"

            label = entry.get("override_label", default_label)
            branch_prefix = entry.get("override_branch_prefix", default_prefix)

            if branch_prefix in seen_prefixes:
                raise ValueError(
                    f"Duplicate branch_prefix '{branch_prefix}' -- each agent entry must have a unique prefix"
                )
            seen_prefixes.add(branch_prefix)

            name_letter = AGENT_NAME_LETTERS[i] if i < len(AGENT_NAME_LETTERS) else str(i + 1)
            name = f"Agent {name_letter}"
            mention = harnesses[harness].get("mention", "")

            agents.append(
                AgentConfig(
                    label=label,
                    harness=harness,
                    model_id=model_id,
                    model_label=model_label,
                    branch_prefix=branch_prefix,
                    name=name,
                    mention=mention,
                    claude_args=entry.get("claude_args", ""),
                )
            )

        checks = ChecksConfig.from_dict(raw.get("checks", {}))
        llm = LLMConfig.from_dict(raw.get("llm", {}))
        deploy = DeployConfig.from_dict(raw.get("deploy", {}))
        plan = PlanConfig.from_dict(raw.get("plan", {}))

        return cls(agents=agents, checks=checks, llm=llm, deploy=deploy, plan=plan)
