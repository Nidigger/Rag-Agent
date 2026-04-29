import os
from datetime import datetime
from unittest.mock import patch

import pytest

from app.config import settings
from app.utils.path_tool import get_abs_path


class TestPathResolution:
    def test_project_root_resolves_correctly(self):
        root = get_abs_path("")
        assert os.path.isdir(root)
        assert os.path.exists(os.path.join(root, "config"))
        assert os.path.exists(os.path.join(root, "prompts"))

    def test_config_path_resolves(self):
        path = get_abs_path("config/rag.yml")
        assert os.path.isfile(path)

    def test_chroma_db_path_resolves(self):
        path = get_abs_path(settings.chroma.persist_directory)
        assert os.path.isdir(path)

    def test_external_data_path_resolves(self):
        path = get_abs_path(settings.rag.external_data_path)
        assert os.path.isfile(path)


class TestCSVFix:
    def test_csv_dict_reader_parses_correctly(self):
        from app.agent.tools.agent_tools import _generate_external_data

        _generate_external_data()
        from app.agent.tools.agent_tools import external_data

        assert "1001" in external_data
        assert "2025-01" in external_data["1001"]
        record = external_data["1001"]["2025-01"]
        assert "特征" in record
        assert "清洁效率" in record
        assert "耗材" in record
        assert "对比" in record

    def test_fetch_external_data_returns_string(self):
        from app.agent.tools.agent_tools import fetch_external_data

        result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
        assert isinstance(result, str)
        assert "65" in result  # 65㎡ in the feature field

    def test_fetch_external_data_missing_returns_empty(self):
        from app.agent.tools.agent_tools import fetch_external_data

        result = fetch_external_data.invoke({"user_id": "9999", "month": "2099-01"})
        assert result == ""


class TestGetCurrentMonth:
    def test_returns_actual_month(self):
        from app.agent.tools.agent_tools import get_current_month

        result = get_current_month.invoke({})
        expected = datetime.now().strftime("%Y-%m")
        assert result == expected

    def test_format_is_yyyy_mm(self):
        from app.agent.tools.agent_tools import get_current_month

        result = get_current_month.invoke({})
        parts = result.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2


class TestPromptLoader:
    def test_load_system_prompts(self):
        from app.utils.prompt_loader import load_system_prompts

        result = load_system_prompts()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_rag_prompts(self):
        from app.utils.prompt_loader import load_rag_prompts

        result = load_rag_prompts()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_report_prompts(self):
        from app.utils.prompt_loader import load_report_prompts

        result = load_report_prompts()
        assert isinstance(result, str)
        assert len(result) > 0
