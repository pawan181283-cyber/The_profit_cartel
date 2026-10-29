import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nse_deal_agent_standalone as bot

class TestTelegramIntegration:
    @patch('nse_deal_agent_standalone.requests.post')
    def test_send_telegram_alert_success(self, mock_post):
        mock_post.return_value.json.return_value = {"ok": True}
        mock_post.return_value.raise_for_status = MagicMock()
        os.environ["TELEGRAM_BOT_TOKEN"] = "test"
        os.environ["TELEGRAM_CHAT_ID"] = "test"
        assert bot.send_telegram_alert("Test") is True

    def test_send_telegram_alert_missing_creds(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        assert bot.send_telegram_alert("Test") is False

    @patch('nse_deal_agent_standalone.requests.post')
    def test_send_telegram_document_success(self, mock_post, tmp_path):
        mock_post.return_value.json.return_value = {"ok": True}
        mock_post.return_value.raise_for_status = MagicMock()
        os.environ["TELEGRAM_BOT_TOKEN"] = "test"
        os.environ["TELEGRAM_CHAT_ID"] = "test"
        f = tmp_path / "test.md"
        f.write_text("# Test")
        assert bot.send_telegram_document(str(f), "Cap") is True

class TestCLIDemoMode:
    def test_demo_mode_runs(self):
        os.environ.pop("GEMINI_API_KEY", None)
        exit_code = bot.main(["--demo", "--no-ai", "--top", "2", "--out", "test_out.md"])
        assert exit_code == 0
        assert os.path.exists("test_out.md")
        os.remove("test_out.md")