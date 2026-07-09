import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import config_parser  # noqa: E402
from models import ChecksConfig  # noqa: E402

MARK = "<!-- lbm-repair -->"


def test_get_triggers_present_and_absent():
    assert config_parser.get_repair_comment_triggers(
        {"checks": {"repair_comment_triggers": [MARK]}}
    ) == [MARK]
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
    assert ChecksConfig.from_dict({"repair_comment_triggers": [MARK]}).repair_comment_triggers == [
        MARK
    ]
    assert ChecksConfig.from_dict({}).repair_comment_triggers == []
