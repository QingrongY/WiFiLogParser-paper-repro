from apps_v2.logparser.services.log_extractor.fallback.regex_validator import RegexValidator
from apps_v2.logparser.services.log_extractor.common.config import LLMExtractorSettings
from apps_v2.logparser.services.log_extractor.runner import LogExtractionRunner
from apps_v2.logparser.services.log_extractor.timestamp_agent import (
    TimestampAgent,
    TimestampValidationReport,
    _build_rule_from_json,
)
from apps_v2.logparser.services.wifi_log_parser.template_postprocessor import TemplatePostProcessor


def test_validator_rejects_prefix_only_regex() -> None:
    validator = RegexValidator()
    pattern = r"prefix(?P<client_name>.*?) suffix"
    assert not validator.validate_regex(pattern, ["prefixclient suffix trailing"])


def test_validator_accepts_full_line_regex() -> None:
    validator = RegexValidator()
    pattern = r"prefix(?P<client_name>.*?) suffix"
    assert validator.validate_regex(pattern, ["prefixclient suffix"])


def test_field_extraction_requires_a_full_line_match() -> None:
    prefix_only = r"prefix(?P<client_name>.*?) suffix"
    full_line = r"prefix(?P<client_name>.*?) suffix"
    assert LogExtractionRunner._extract_fields(None, prefix_only, "prefixclient suffix trailing") == {}
    assert LogExtractionRunner._extract_fields(None, full_line, "prefixclient suffix") == {
        "client_name": "client"
    }


def test_timestamp_has_year_accepts_quoted_boolean_values() -> None:
    base = {
        "regex": r"(?P<date>\d{2}-\d{2})(?P<time>\d{2}:\d{2})",
        "date_format": "%m-%d",
        "time_format": "%H:%M",
    }
    assert _build_rule_from_json({**base, "has_year": "true"}).has_year is True
    assert _build_rule_from_json({**base, "has_year": "false"}).has_year is False


def test_timestamp_refinement_current_rule_excludes_matching_coverage() -> None:
    settings = LLMExtractorSettings(primary_model="test", fallback_models=())
    agent = TimestampAgent(settings)
    rule_data = {
        "regex": r"(?P<date>\d{2}-\d{2})(?P<time>\d{2}:\d{2})",
        "date_format": "%m-%d",
        "time_format": "%H:%M",
        "has_year": False,
    }
    report = TimestampValidationReport(
        total=10,
        success=8,
        success_rate=0.8,
        no_match=2,
        parse_failed=0,
        ts_too_far=0,
        empty_content=0,
        examples={"no_match": ["01-01 00:00 example"]},
    )
    captured: list[dict[str, str]] = []

    def fake_call(messages: list[dict[str, str]]) -> dict:
        captured.extend(messages)
        return rule_data

    agent._call_llm = fake_call  # type: ignore[method-assign]
    agent._repair_rule_with_llm(_build_rule_from_json(rule_data), report)
    prompt = captured[0]["content"]
    assert "matching_coverage" not in prompt
    assert "has_year: false" in prompt


def test_template_output_accepts_a_quoted_integer_label() -> None:
    parsed = TemplatePostProcessor().process_output(
        '{"template": "prefix {{client_name}}", "connect_flag": "1"}'
    )
    assert parsed.connect_flag == 1
