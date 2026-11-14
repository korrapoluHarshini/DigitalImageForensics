"""End-to-end smoke tests for the Streamlit application.

These tests execute every sidebar page through Streamlit's AppTest harness.
They do not run model inference, so they stay fast and work without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


PAGES = [
    "Home Dashboard",
    "Image Manipulation Detection",
    "AI-Generated Image Detection",
    "Chat Screenshot Analysis",
    "Analysis History",
    "About the Project",
]


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(APP_PATH, default_timeout=180)
    instance.run()
    return instance


def test_home_page_renders(app):
    assert len(app.exception) == 0
    assert app.radio[0].value == "Home Dashboard"
    markdown = " ".join(block.value for block in app.markdown)
    assert "One workspace for image forensics" in markdown


@pytest.mark.parametrize("page", PAGES[1:])
def test_every_page_renders_without_exception(app, page):
    app.radio[0].set_value(page).run()
    assert len(app.exception) == 0, [str(exception.value) for exception in app.exception]


def test_ai_page_shows_placeholder(app):
    app.radio[0].set_value("AI-Generated Image Detection").run()
    markdown = " ".join(block.value for block in app.markdown)
    assert "placeholder" in markdown.lower() or "integration" in markdown.lower()
    assert len(app.file_uploader) == 1


def test_sidebar_reports_threshold_and_status(app):
    app.radio[0].set_value("Home Dashboard").run()
    assert len(app.slider) >= 2
    sliders = {slider.label: slider.value for slider in app.slider}
    assert sliders["Localization threshold"] == pytest.approx(0.4, abs=0.01)
    assert sliders["Image decision threshold"] == pytest.approx(0.42, abs=0.01)
    assert sliders["Uncertainty band (±)"] == pytest.approx(0.13, abs=0.01)


def test_uploader_enforces_the_configured_megabyte_limit(app):
    """Streamlit reads max_upload_size in MB; passing bytes displayed '25600GB'."""
    app.radio[0].set_value("Image Manipulation Detection").run()
    uploader = app.file_uploader[0]
    assert uploader.proto.max_upload_size_mb == 25
    assert set(uploader.allowed_type) == {".jpg", ".jpeg", ".png", ".webp"}
