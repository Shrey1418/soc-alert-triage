import sys

import pytest

from soc_triage.utils.config import _deep_merge, load_config
from soc_triage.utils.exceptions import ConfigError, PipelineError
from soc_triage.utils.logger import get_logger


def test_deep_merge_keeps_untouched_nested_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}}
    assert _deep_merge(base, override) == {"a": {"x": 1, "y": 3}, "b": 1}


def test_load_config_reads_base():
    assert load_config()["paths"]["raw_dir"] == "data/raw"


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config("does_not_exist")


def test_pipeline_error_has_message_and_line():
    try:
        raise ValueError("boom")
    except ValueError as e:
        err = PipelineError(e, sys)
    assert "boom" in str(err)
    assert "line" in str(err)


def test_logger_does_not_duplicate_handlers():
    first = get_logger("soc_triage.test")
    count = len(first.handlers)
    assert len(get_logger("soc_triage.test").handlers) == count
