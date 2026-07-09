"""Tests for the optional plan-step feature (config, prompts, plan links, rev cap)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import config_parser  # noqa: E402
from agent_ops import _plan_rev_allowed, build_task_prompt, plan_file_url  # noqa: E402
from models import PlanConfig  # noqa: E402


# --- Task 1: config -------------------------------------------------------


def test_get_plan_config_defaults():
    assert config_parser.get_plan_config({}) == {
        "enabled": False,
        "dir": "lbm-plans",
        "feedback_revs": 1,
        "prototype": False,
        "plan_context_comment_marker": "<!-- lbm-plan-context -->",
    }


def test_get_plan_config_enabled():
    cfg = config_parser.get_plan_config({"plan": {"enabled": True, "dir": "plans", "feedback_revs": 2}})
    assert cfg["enabled"] is True
    assert cfg["dir"] == "plans"
    assert cfg["feedback_revs"] == 2


def test_get_plan_config_prototype_marker_default():
    # Prototype iteration: report comments carrying this marker re-invoke the
    # agent to write its plan. Defaults to the recommended marker.
    assert config_parser.get_plan_config({})["plan_context_comment_marker"] == "<!-- lbm-plan-context -->"
    cfg = config_parser.get_plan_config({"plan": {"plan_context_comment_marker": "<!-- x -->"}})
    assert cfg["plan_context_comment_marker"] == "<!-- x -->"


def test_planconfig_from_dict():
    p = PlanConfig.from_dict({"enabled": True, "dir": "plans", "feedback_revs": 2})
    assert (p.enabled, p.dir, p.feedback_revs, p.prototype) == (True, "plans", 2, False)
    assert PlanConfig.from_dict({}).enabled is False


# --- Phase detection (label-based, race-free) ----------------------------


def test_resolve_phase_disabled_is_implement():
    # plan disabled -> always implement (today's behavior)
    assert config_parser.resolve_phase({}, []) == "implement"
    assert config_parser.resolve_phase({}, ["lbm:phase-implement"]) == "implement"


def test_resolve_phase_enabled_no_implement_label_is_plan():
    cfg = {"plan": {"enabled": True}}
    assert config_parser.resolve_phase(cfg, []) == "plan"
    assert config_parser.resolve_phase(cfg, ["ready-for-dev"]) == "plan"


def test_resolve_phase_enabled_with_implement_label_is_implement():
    cfg = {"plan": {"enabled": True}}
    assert config_parser.resolve_phase(cfg, ["lbm:phase-implement"]) == "implement"


# --- Task 2: build_task_prompt -------------------------------------------


def test_build_task_prompt_plan():
    t = build_task_prompt("7", "plan", "lbm-plans")
    assert "lbm-plans/issue-7/plan.md" in t
    assert "do NOT implement" in t
    assert "PLAN PHASE" in t


def test_build_task_prompt_implement():
    t = build_task_prompt("7", "implement", "lbm-plans")
    assert "lbm-plans/issue-7/plan.md" in t
    assert "IMPLEMENT PHASE" in t


def test_build_task_prompt_custom_dir():
    t = build_task_prompt("12", "plan", "plans")
    assert "plans/issue-12/plan.md" in t


def test_build_task_prompt_plan_prototype():
    t = build_task_prompt("7", "plan", "lbm-plans", prototype=True)
    assert "lbm-plans/issue-7/plan.md" in t
    assert "do NOT implement" in t
    assert "prototype" in t.lower()


def test_build_task_prompt_implement_ignores_prototype():
    # prototype only affects the plan phase
    assert build_task_prompt("7", "implement", "lbm-plans", prototype=True) == build_task_prompt(
        "7", "implement", "lbm-plans"
    )


# --- Task 3: plan_file_url -----------------------------------------------


def test_plan_file_url():
    u = plan_file_url("o/r", "claude-x/issue-7", "7", "lbm-plans")
    assert u == "https://github.com/o/r/blob/claude-x/issue-7/lbm-plans/issue-7/plan.md"


# --- Task 4: feedback rev cap --------------------------------------------


def test_plan_rev_cap():
    assert _plan_rev_allowed(0, 1) is True
    assert _plan_rev_allowed(1, 1) is False
    assert _plan_rev_allowed(1, 2) is True
    assert _plan_rev_allowed(0, 0) is False
