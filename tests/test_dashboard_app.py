"""Smoke test: the app renders with every source down (needs streamlit; skipped otherwise)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboards" / "app.py")


def test_app_renders_when_services_are_down(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API", "http://127.0.0.1:1")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("not reachable" in w.value for w in at.warning)
