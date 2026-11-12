"""MAR_PRIMARY_VENDOR=deepseek runs every call on the second vendor, with no standby, so a run never mixes vendors."""
import pytest

from medagent.core.config import Settings


def test_deepseek_becomes_the_only_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-flash")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://vendor.test/anthropic/")
    monkeypatch.setenv("MAR_PRIMARY_VENDOR", "deepseek")
    settings = Settings.load()
    assert (settings.model, settings.api_key, settings.api_url) == ("test-flash", "test-key", "https://vendor.test/anthropic")
    assert settings.fallback_api_key == "" and settings.structured_output is False and settings.judge_model == ""


def test_default_vendor_is_unchanged_and_unknown_is_rejected(monkeypatch):
    monkeypatch.delenv("MAR_PRIMARY_VENDOR", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-haiku")
    assert Settings.load().model == "test-haiku"
    monkeypatch.setenv("MAR_PRIMARY_VENDOR", "other")
    with pytest.raises(ValueError, match="unknown_primary_vendor"):
        Settings.load()
