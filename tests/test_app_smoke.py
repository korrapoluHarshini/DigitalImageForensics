"""End-to-end smoke tests for the Streamlit application.

These tests execute every sidebar page through Streamlit's AppTest harness.
They do not run model inference, so they stay fast and work without a GPU.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

from utils.config import settings

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


def test_ai_page_shows_integration_state(app):
    app.radio[0].set_value("AI-Generated Image Detection").run()
    markdown = " ".join(block.value for block in app.markdown).lower()
    assert "integration placeholder" in markdown or "detector integrated" in markdown
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


def test_about_metrics_table_is_arrow_compatible(tmp_path, monkeypatch, caplog):
    """The evaluation table mixed an int count with formatted strings.

    PyArrow inferred int64 from the first cell and then failed on '0.7466'
    (``Conversion failed for column Value``). Every Value cell must stay a
    string so the table serialises without Streamlit's automatic fix-ups.
    """
    artifact = {
        "artifact": "module1_test_metrics.json",
        "generated_utc": "2026-11-14T00:00:00+00:00",
        "threshold": 0.4,
        "provenance": "synthetic test artifact",
        "localization": {
            "images": 768,
            "global_iou": 0.7466,
            "mean_image_iou": 0.5891,
            "micro_precision": 0.8123,
            "micro_recall": 0.7012,
            "micro_f1": 0.7526,
        },
        "image_level": {"false_positive_rate": 0.12, "recall": 0.83, "f1": 0.79},
    }
    (tmp_path / "module1_test_metrics.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    monkeypatch.setattr(
        "utils.config.settings", replace(settings, evaluation_dir=tmp_path)
    )

    instance = AppTest.from_file(APP_PATH, default_timeout=180)
    instance.run()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="streamlit"):
        instance.radio[0].set_value("About the Project").run()

    assert len(instance.exception) == 0, [str(exc.value) for exc in instance.exception]
    assert "Arrow table was unsuccessful" not in caplog.text

    metric_frames = [
        frame.value
        for frame in instance.dataframe
        if "Metric" in frame.value.columns and "Images" in set(frame.value["Metric"])
    ]
    assert metric_frames, "the localization metrics table was not rendered"
    value_column = metric_frames[0]["Value"]
    assert all(isinstance(value, str) for value in value_column)
    assert metric_frames[0].loc[
        metric_frames[0]["Metric"] == "Images", "Value"
    ].iloc[0] == "768"
    # The exact conversion that failed in the live app must now succeed.
    pa.Table.from_pandas(metric_frames[0])
