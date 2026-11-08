from jeff_router_under_test.settings import DEFAULTS, read_settings


def test_defaults_when_nothing_configured():
    assert read_settings(lambda key, default: default) == DEFAULTS


def test_valid_values_are_used():
    given = {"enabled": False, "url": "http://127.0.0.1:9999", "timeout_s": 1.5, "max_request_chars": 12000,
             "log": True, "log_max_mb": 5, "main_thinking": False, "thinking_router": True, "router_adapter": "r",
             "router_threshold": 0.7, "guards": False, "tool_shadow": True}
    assert read_settings(lambda key, default: given.get(key, default)) == given


def test_invalid_values_fall_back_to_defaults():
    given = {"enabled": "yes", "url": "", "timeout_s": -1, "max_request_chars": 0, "log_max_mb": None,
             "main_thinking": "off"}
    assert read_settings(lambda key, default: given.get(key, default)) == DEFAULTS


def test_bool_is_not_accepted_as_a_number():
    assert read_settings(lambda key, default: True if key == "timeout_s" else default)["timeout_s"] == 2.0


def test_non_finite_numbers_fall_back_to_defaults():
    for value in (float("inf"), float("-inf")):
        given = {"timeout_s": float("inf"), "log_max_mb": value}
        result = read_settings(lambda key, default, given=given: given.get(key, default))
        assert result["timeout_s"] == 2.0 and result["log_max_mb"] == 50
        assert result == DEFAULTS


def test_floats_fall_back_for_int_setting():
    for value in (1.5, 12000.0):
        result = read_settings(lambda key, default, value=value: value if key == "max_request_chars" else default)
        assert result["max_request_chars"] == 24000


def test_get_config_errors_fall_back_to_defaults():
    def broken(key, default):
        raise RuntimeError("config unreadable")
    assert read_settings(broken) == DEFAULTS


def test_router_threshold_above_one_falls_back():
    assert read_settings(lambda key, default: 1.5 if key == "router_threshold" else default)["router_threshold"] == 0.6


def test_v0_1_strata_thinking_name_still_works():
    assert read_settings(lambda key, default: {"strata_thinking": False}.get(key, default))["main_thinking"] is False
    both = {"strata_thinking": False, "main_thinking": True}
    assert read_settings(lambda key, default: both.get(key, default))["main_thinking"] is True


def test_router_threshold_zero_is_kept():
    assert read_settings(lambda key, default: 0 if key == "router_threshold" else default)["router_threshold"] == 0
    assert read_settings(lambda key, default: 0.0 if key == "router_threshold" else default)["router_threshold"] == 0.0
    assert read_settings(lambda key, default: False if key == "router_threshold" else default)["router_threshold"] == 0.6
    assert read_settings(lambda key, default: 0 if key == "timeout_s" else default)["timeout_s"] == 2.0
