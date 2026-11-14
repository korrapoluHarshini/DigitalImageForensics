"""Digital Image Authentication and Tampering Detection System.

A single Streamlit application integrating three independent forensic modules:

1. MVSS-Net manipulation detection and localization.
2. Pluggable AI-generated image detection (integration placeholder until weights
   are supplied).
3. Chat screenshot screening with OCR.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from modules.ai_generated_detection import detect_ai_generated_image, get_ai_status
from modules.chat_tampering import detect_chat_tampering, ocr_status
from modules.image_manipulation import (
    IMAGE_DECISION_AUTHENTIC,
    IMAGE_DECISION_TAMPERED,
    IMAGE_DECISION_UNCERTAIN,
    detect_image_manipulation,
    predictor_status,
)
from utils.config import settings
from utils.image_utils import (
    decode_image_bytes,
    describe_image,
    encode_mask_png_bytes,
    encode_png_bytes,
    image_fingerprint,
)
from utils.reporting import (
    append_history,
    build_ai_report,
    build_chat_report,
    build_manipulation_report,
    clear_history,
    load_history,
    record_from_ai,
    record_from_chat,
    record_from_manipulation,
)
from utils.validation import UploadValidationError, sanitize_filename


# ----------------------------------------------------------------------
# Page configuration and global styling
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Image Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    :root {
        --ink: #0f172a;
        --muted: #5b6b82;
        --line: #e2e8f0;
        --brand: #1d4ed8;
        --brand-2: #0ea5a4;
        --danger: #b42318;
        --warn: #b54708;
        --ok: #067647;
        --bg: #f6f8fc;
    }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
    #MainMenu, footer { visibility: hidden; }

    .df-hero {
        background: linear-gradient(120deg, #0b1f4b 0%, #123a86 45%, #0e7490 100%);
        color: #ffffff;
        padding: 2.1rem 2.4rem;
        border-radius: 20px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.4rem;
    }
    .df-hero h1 { color:#ffffff; font-size: 2.05rem; margin: 0 0 .45rem 0; letter-spacing:-.02em; }
    .df-hero p { color: #dbeafe; font-size: 1.02rem; max-width: 900px; margin: 0; line-height:1.55; }
    .df-hero .df-kicker {
        display:inline-block; font-size:.74rem; letter-spacing:.16em; text-transform:uppercase;
        background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.24);
        padding:.28rem .6rem; border-radius:999px; margin-bottom:.8rem; font-weight:700;
    }

    .df-card {
        background:#ffffff; border:1px solid var(--line); border-radius:16px;
        padding:1.1rem 1.15rem; height:100%;
        box-shadow:0 6px 18px rgba(15,23,42,.06);
    }
    .df-card h3 { margin:.2rem 0 .35rem 0; font-size:1.03rem; color:var(--ink); }
    .df-card p { color:var(--muted); font-size:.9rem; line-height:1.5; margin:0; }
    .df-card .df-icon { font-size:1.5rem; }
    .df-tag {
        display:inline-block; margin-top:.75rem; font-size:.7rem; font-weight:700;
        letter-spacing:.04em; text-transform:uppercase; padding:.22rem .5rem;
        border-radius:999px; background:#eef4ff; color:#1e40af; border:1px solid #dbe6ff;
    }
    .df-tag.alt { background:#ecfdf5; color:#047857; border-color:#c7f0e0; }
    .df-tag.warn { background:#fffaeb; color:#b54708; border-color:#fde7b0; }

    .df-badge {
        display:inline-block; padding:.24rem .65rem; border-radius:999px;
        font-size:.78rem; font-weight:700; letter-spacing:.01em;
    }
    .df-badge.ok { background:#e7f6ee; color:#067647; border:1px solid #b7e6cc; }
    .df-badge.warn { background:#fff6e5; color:#b54708; border:1px solid #f5d99a; }
    .df-badge.bad { background:#feeceb; color:#b42318; border:1px solid #f7c8c4; }
    .df-badge.neutral { background:#eef2f7; color:#334155; border:1px solid #dbe2ea; }

    .df-note {
        background:#f8fafc; border-left:4px solid #94a3b8; border-radius:8px;
        padding:.75rem .9rem; color:#475569; font-size:.88rem; line-height:1.5;
    }
    .df-section-title { font-size:1.18rem; font-weight:800; color:var(--ink); margin:.3rem 0 .6rem 0; }

    .df-decision {
        border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.5rem;
        margin:.2rem 0 .9rem 0; box-shadow:0 8px 24px rgba(15,23,42,.06);
    }
    .df-decision.ok { background:linear-gradient(120deg,#ecfdf3,#f6fffa); border-color:#b7e6cc; }
    .df-decision.bad { background:linear-gradient(120deg,#fef3f2,#fff8f7); border-color:#f7c8c4; }
    .df-decision.warn { background:linear-gradient(120deg,#fffaeb,#fffdf5); border-color:#fde7b0; }
    .df-decision .df-decision-kicker {
        font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
        font-weight:800; color:var(--muted);
    }
    .df-decision .df-decision-title {
        font-size:1.85rem; font-weight:900; letter-spacing:-.02em; margin:.18rem 0 .3rem 0;
    }
    .df-decision.ok .df-decision-title { color:#067647; }
    .df-decision.bad .df-decision-title { color:#b42318; }
    .df-decision.warn .df-decision-title { color:#b54708; }
    .df-decision .df-decision-sub { color:#475569; font-size:.9rem; line-height:1.5; }

    div[data-testid="stMetric"] {
        background:#ffffff; border:1px solid var(--line); border-radius:14px;
        padding:.7rem .8rem; box-shadow:0 4px 12px rgba(15,23,42,.04);
    }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
    .stTabs [data-baseweb="tab"] {
        background:#ffffff; border:1px solid var(--line); border-radius:10px;
        padding:.35rem .8rem;
    }
    .stTabs [aria-selected="true"] { background:#eef4ff; border-color:#c7d9ff; }
    section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
    .df-sidebar-brand { font-weight:900; font-size:1.05rem; color:var(--ink); letter-spacing:-.01em; }
    .df-sidebar-sub { color:var(--muted); font-size:.78rem; line-height:1.45; margin-top:.15rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached resources and process-wide state
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_predictor():
    """Lazily created MVSS-Net wrapper (weights load on first analysis)."""
    from modules.image_manipulation.model import get_default_predictor

    return get_default_predictor()


def _result_cache() -> dict:
    return st.session_state.setdefault("result_cache", {})


def _cache_get(key):
    return _result_cache().get(key)


def _cache_put(key, value) -> None:
    cache = _result_cache()
    cache[key] = value
    while len(cache) > 6:
        cache.pop(next(iter(cache)))


def _threshold() -> float:
    return float(st.session_state.get("threshold", settings.manipulation_threshold))


def _image_threshold() -> float:
    return float(st.session_state.get("image_threshold", settings.image_level_threshold))


def _image_uncertainty_margin() -> float:
    return float(
        st.session_state.get("image_uncertainty_margin", settings.image_uncertainty_margin)
    )


def _append_session_history(record: dict) -> None:
    st.session_state.setdefault("session_history", []).append(record)


# ----------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------

def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="df-badge {kind}">{text}</span>'


def finding_badge(finding: str) -> str:
    normalized = finding.lower()
    if "possible" in normalized or normalized in {"ai_generated", "uncertain"}:
        kind = "bad" if normalized == "ai_generated" else "warn"
        label = finding.replace("_", " ").title()
    elif "no_strong" in normalized or normalized == "real":
        kind = "ok"
        label = finding.replace("_", " ").title()
    else:
        kind, label = "neutral", finding.replace("_", " ").title()
    return badge(label, kind)


LOCALIZATION_BADGES = {
    "possible_manipulation": ("Pixel-level signal: Possible manipulation", "warn"),
    "no_strong_evidence": ("Pixel-level signal: No strong evidence", "ok"),
    "possible_visual_manipulation": ("Pixel-level signal: Possible manipulation", "warn"),
    "no_strong_visual_evidence": ("Pixel-level signal: No strong evidence", "ok"),
}


def localization_badge(finding: str) -> str:
    """Pixel-level badge, deliberately separate from the image-level label."""
    label, kind = LOCALIZATION_BADGES.get(
        finding.lower(), (finding.replace("_", " ").title(), "neutral")
    )
    return badge(label, kind)


def format_area(ratio: float, pixels: int | None = None) -> str:
    """Format a pixel-area proportion with enough precision for tiny signals."""
    percent = float(ratio) * 100.0
    if percent == 0.0:
        text = "0%"
    elif percent < 0.000001:
        text = "<0.000001%"
    elif percent < 0.01:
        text = f"{percent:.8f}%"
    elif percent < 1.0:
        text = f"{percent:.6f}%"
    else:
        text = f"{percent:.4f}%"
    if pixels is not None:
        text += f" ({pixels:,} px)"
    return text


def decision_card(result) -> str:
    """Prominent Authentic/Tampered/Uncertain banner for Module 1."""
    kind = {
        IMAGE_DECISION_TAMPERED: "bad",
        IMAGE_DECISION_AUTHENTIC: "ok",
    }.get(result.image_decision, "warn")
    sign = "+" if result.image_decision_margin >= 0 else ""
    detail = (
        f"Maximum pixel probability <b>{result.image_score:.3f}</b> &nbsp;·&nbsp; "
        f"image-level threshold <b>{result.image_threshold:.2f}</b> &nbsp;·&nbsp; "
        f"margin {sign}{result.image_decision_margin:.3f} &nbsp;·&nbsp; "
        f"uncertainty band ±{result.image_uncertainty_margin:.2f}"
    )
    if result.uncertain:
        detail += (
            "<br>The score lies inside the uncertainty band, so no definite class is "
            "asserted. Treat this image as inconclusive."
        )
    return (
        f'<div class="df-decision {kind}">'
        f'<div class="df-decision-kicker">Image-level prediction</div>'
        f'<div class="df-decision-title">{result.image_decision_label}</div>'
        f'<div class="df-decision-sub">{detail}</div>'
        f"</div>"
    )


def status_row(label: str, value: str, kind: str = "neutral") -> str:
    return f"{label} &nbsp; {badge(value, kind)}"


def section_title(text: str) -> None:
    st.markdown(f'<div class="df-section-title">{text}</div>', unsafe_allow_html=True)


def render_limitations(warnings: list[str], limitations: list[str]) -> None:
    if warnings:
        st.markdown("**Warnings**")
        for item in warnings:
            st.markdown(f"- {item}")
    if limitations:
        st.markdown("**Limitations**")
        for item in limitations:
            st.markdown(f"- {item}")


def download_row(prefix: str, files: dict[str, bytes]) -> None:
    columns = st.columns(len(files))
    extensions = {"PNG": ".png", "TXT": ".txt", "JSON": ".json", "CSV": ".csv"}
    for column, (label, payload) in zip(columns, files.items()):
        with column:
            suffix = next((value for key, value in extensions.items() if label.upper().endswith(key)), ".bin")
            file_name = f"{prefix}_{label.lower().replace(' ', '_')}{suffix}"
            st.download_button(
                label,
                data=payload,
                file_name=file_name,
                mime="application/octet-stream",
                key=f"{prefix}_download_{label}",
                use_container_width=True,
            )


def upload_image(key: str, label: str, help_text: str):
    uploaded = st.file_uploader(
        label,
        type=["jpg", "jpeg", "png", "webp"],
        key=key,
        max_upload_size=settings.max_upload_mb,
        help=help_text,
    )
    if uploaded is None:
        st.caption(
            f"Accepted: JPG, JPEG, PNG, WEBP · maximum {settings.max_upload_mb} MB. "
            "Files are processed in memory and are not saved by the app."
        )
        return None, None, None

    try:
        data = uploaded.getvalue()
        rgb = decode_image_bytes(data, getattr(uploaded, "name", None))
    except UploadValidationError as exc:
        st.error(f"Upload rejected: {exc}")
        return None, None, None
    except Exception:
        st.error("Upload rejected: the file could not be decoded safely.")
        return None, None, None

    info = describe_image(rgb)
    display_name = sanitize_filename(getattr(uploaded, "name", None))
    st.image(
        rgb,
        caption=f"{display_name} · {info['width']} × {info['height']} px · {info['megapixels']} MP",
        width="stretch",
    )
    return rgb, image_fingerprint(rgb), info


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

PAGES = [
    "Home Dashboard",
    "Image Manipulation Detection",
    "AI-Generated Image Detection",
    "Chat Screenshot Analysis",
    "Analysis History",
    "About the Project",
]

with st.sidebar:
    st.markdown(
        '<div style="font-size:2rem;line-height:1">🛡️</div>'
        '<div class="df-sidebar-brand">Digital Image Forensics</div>'
        '<div class="df-sidebar-sub">Authentication &amp; tampering analysis platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="df-sidebar-sub"><b>Module 1 thresholds</b></div>', unsafe_allow_html=True)
    st.session_state["image_threshold"] = st.slider(
        "Image decision threshold",
        min_value=0.05,
        max_value=0.99,
        value=float(
            st.session_state.get("image_threshold", settings.image_level_threshold)
        ),
        step=0.01,
        help=(
            "Maximum pixel probability at or above which the whole image is predicted "
            "Tampered. Selected on validation authentic + tampered images. The test "
            "split was not used for tuning."
        ),
    )
    st.session_state["image_uncertainty_margin"] = st.slider(
        "Uncertainty band (±)",
        min_value=0.0,
        max_value=0.20,
        value=float(
            st.session_state.get(
                "image_uncertainty_margin", settings.image_uncertainty_margin
            )
        ),
        step=0.01,
        help=(
            "Scores within this distance of the image decision threshold are reported "
            "as Uncertain instead of Authentic or Tampered."
        ),
    )
    st.caption(
        "Image decision: max pixel score ≥ threshold → Tampered; ≤ threshold − band "
        "→ Authentic; otherwise Uncertain."
    )
    st.session_state["threshold"] = st.slider(
        "Localization threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("threshold", settings.manipulation_threshold)),
        step=0.05,
        help=(
            "Pixel probability above which a pixel enters the binary mask and "
            "highlighted overlay. Default 0.40 was selected on the validation split. "
            "This threshold does not decide Authentic vs Tampered."
        ),
    )
    st.caption(
        "Localization threshold controls the mask only. Both settings are fixed per "
        "analysis and are not adapted to individual images."
    )

    st.divider()
    st.markdown('<div class="df-sidebar-sub"><b>System status</b></div>', unsafe_allow_html=True)
    m1 = predictor_status()
    st.markdown(
        status_row("MVSS-Net", "ready" if m1["available"] else "missing", "ok" if m1["available"] else "bad"),
        unsafe_allow_html=True,
    )
    st.caption(f"Device: {m1['device']} · checkpoint: {m1['checkpoint']}")

    ai_status = get_ai_status()
    st.markdown(
        status_row(
            "AI detector",
            "integrated" if ai_status["available"] else "placeholder",
            "ok" if ai_status["available"] else "warn",
        ),
        unsafe_allow_html=True,
    )
    ocr = ocr_status()
    st.markdown(
        status_row("OCR", ocr["backend"] or "unavailable", "ok" if ocr["available"] else "warn"),
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "Privacy: uploads stay in memory; history stores aggregate scores and a "
        "content fingerprint only — never screenshots, OCR text or chat content."
    )


# ----------------------------------------------------------------------
# Page: Home
# ----------------------------------------------------------------------

def render_home() -> None:
    st.markdown(
        """
        <div class="df-hero">
            <span class="df-kicker">Digital image authentication</span>
            <h1>One workspace for image forensics</h1>
            <p>
                Upload a photograph, an AI-generated image or a chat screenshot and
                receive a transparent, evidence-oriented analysis: predicted findings,
                confidence scores, localized suspicious regions and a clear statement
                of what the evidence can and cannot support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(3, gap="large")
    cards = [
        (
            "🔍",
            "Image Manipulation",
            "Pixel-level forgery localization with MVSS-Net (ResNet-50, CASIA2.0). "
            "Heatmap, binary mask and highlighted regions.",
            "Module 1 · pretrained model ready",
            "",
        ),
        (
            "🤖",
            "AI-Generated Image",
            "Separates real photographs from synthetic images and abstains when the "
            "evidence is too weak. Modular model integration.",
            "Module 2 · integration placeholder",
            "warn",
        ),
        (
            "💬",
            "Chat Screenshot",
            "Visual tampering screening plus OCR transcription for messaging "
            "screenshots. Explicitly not message-level verification.",
            "Module 3 · screening only",
            "alt",
        ),
    ]
    for column, (icon, title, text, tag, tag_kind) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="df-card">
                    <div class="df-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{text}</p>
                    <span class="df-tag {tag_kind}">{tag}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    section_title("How the platform works")
    steps = st.columns(3, gap="large")
    step_content = [
        ("01 · Upload", "A JPG, PNG or WEBP file is validated for size, real format and safe dimensions."),
        ("02 · Analyse", "The selected module runs independently; a failure in one module cannot break another."),
        ("03 · Review", "Compare the original, heatmap, mask and highlighted result, then read the limitations."),
    ]
    for column, (title, text) in zip(steps, step_content):
        with column:
            st.markdown(f"**{title}**  \n{text}")

    st.write("")
    st.markdown(
        '<div class="df-note"><b>Honest by design.</b> A model output is not proof of '
        "authenticity or manipulation. Scores are not calibrated probabilities, and the "
        "chat module cannot recover or verify an original conversation.</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Page: Module 1
# ----------------------------------------------------------------------

def render_manipulation_page() -> None:
    st.markdown("## Image Manipulation Detection")
    st.caption(
        "MVSS-Net predicts an Authentic/Tampered label for the whole image from the "
        "maximum manipulation probability, then localizes the suspicious pixels that "
        "produced it. There is no separate classification network: the label is a "
        "validated threshold on the existing probability map."
    )

    rgb, fingerprint, info = upload_image(
        "upload_manipulation",
        "Upload a photograph",
        "Best suited to CASIA-like photographic manipulation. Do not use the result as legal proof.",
    )

    with st.expander("Analysis options"):
        detailed = st.checkbox(
            "Detailed patch mode (slower)",
            value=False,
            help="Blends overlapping 512×512 patches instead of resizing the whole image. "
            "Better for large images, but several times slower on CPU.",
        )
        st.caption(
            "The image decision threshold and localization threshold are controlled "
            "separately in the sidebar. Both were selected on the validation split, "
            "not on the test set."
        )

    analyze = st.button(
        "Analyse image",
        type="primary",
        disabled=rgb is None,
        key="btn_manipulation",
        use_container_width=True,
    )

    if rgb is not None and analyze:
        key = (
            "manipulation",
            fingerprint,
            round(_threshold(), 3),
            round(_image_threshold(), 3),
            round(_image_uncertainty_margin(), 3),
            bool(detailed),
        )
        result = _cache_get(key)
        if result is None:
            with st.spinner("Running MVSS-Net localization… this can take ~10 s on CPU."):
                result = detect_image_manipulation(
                    rgb,
                    threshold=_threshold(),
                    image_threshold=_image_threshold(),
                    image_uncertainty_margin=_image_uncertainty_margin(),
                    predictor=cached_predictor(),
                    tiled=detailed,
                    max_tiles=12,
                )
                _cache_put(key, result)
                if result.status == "ok":
                    record = record_from_manipulation(result, fingerprint)
                    append_history(record)
                    _append_session_history({"timestamp": record.timestamp, "module": record.module,
                                             "finding": record.finding, "score": record.score,
                                             "image_decision": result.image_decision,
                                             "image_decision_label": result.image_decision_label,
                                             "localization_label": result.localization_label})
        st.session_state["last_manipulation"] = (fingerprint, result)

    payload = st.session_state.get("last_manipulation")
    if payload and payload[0] == fingerprint and rgb is not None:
        result = payload[1]
        if result.status != "ok":
            st.error("Analysis could not complete.")
            if result.error_message:
                st.code(result.error_message)
            render_limitations(result.warnings, result.limitations)
            return

        st.write("")
        st.markdown(decision_card(result), unsafe_allow_html=True)
        if result.uncertain:
            st.warning(
                f"Uncertain result: the maximum pixel probability "
                f"({result.image_score:.4f}) is within ±{result.image_uncertainty_margin:.2f} "
                f"of the image decision threshold ({result.image_threshold:.2f}). "
                "The app does not claim this image is definitely authentic or tampered."
            )

        first_metrics = st.columns(4)
        first_metrics[0].metric(
            "Image prediction",
            result.image_decision_label.replace(" (Predicted)", "").replace(
                " (Insufficient Evidence)", ""
            ),
        )
        first_metrics[1].metric("Max pixel probability", f"{result.image_score:.4f}")
        first_metrics[2].metric("Image threshold", f"{result.image_threshold:.2f}")
        first_metrics[3].metric(
            "Uncertainty margin", f"±{result.image_uncertainty_margin:.2f}"
        )
        second_metrics = st.columns(4)
        second_metrics[0].metric(
            "Localization threshold", f"{result.localization_threshold:.2f}"
        )
        second_metrics[1].metric(
            "Highlighted area (retained)",
            format_area(
                result.processed_suspicious_area_ratio,
                result.processed_suspicious_pixels,
            ),
        )
        second_metrics[2].metric(
            "Raw thresholded area",
            format_area(
                result.raw_suspicious_area_ratio, result.raw_suspicious_pixels
            ),
        )
        second_metrics[3].metric("Suspicious regions", str(len(result.regions)))
        st.caption(
            "Raw thresholded area counts every pixel above the localization "
            "threshold. Highlighted area counts only pixels inside connected "
            "components that passed the minimum-area filter; that is what the mask "
            "and overlay show. Connected components: "
            f"{result.raw_component_count} raw, {len(result.regions)} retained, "
            f"{result.dropped_component_count} not retained."
        )

        st.markdown(
            f"{localization_badge(result.finding)} &nbsp; "
            f"{badge('Detailed patch mode' if result.mode == 'tiled' else 'Official single-scale mode', 'neutral')}",
            unsafe_allow_html=True,
        )
        st.caption(
            "The badge above is only the pixel-level localization signal. It is "
            "independent of the image-level Authentic/Tampered/Uncertain prediction "
            "and does not prove that the whole image was manipulated."
        )
        st.markdown(
            f'<div class="df-note"><b>Why this prediction?</b> '
            f"{result.decision_explanation}</div>",
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(result.image_score, 0.0), 1.0),
            text=(
                f"Maximum pixel probability {result.image_score:.4f} vs image threshold "
                f"{result.image_threshold:.2f} (uncertainty band "
                f"{result.image_threshold - result.image_uncertainty_margin:.2f}–"
                f"{result.image_threshold + result.image_uncertainty_margin:.2f})"
            ),
        )

        tabs = st.tabs(["Visual explanation", "Suspicious regions", "Decision & thresholds", "Model & limitations"])
        with tabs[0]:
            first = st.columns(2, gap="medium")
            with first[0]:
                st.image(rgb, caption="Original image", width="stretch")
            with first[1]:
                st.image(result.heatmap, caption="Manipulation probability heatmap (blue → red)", width="stretch")
            second = st.columns(2, gap="medium")
            with second[0]:
                st.image(
                    result.mask_visual,
                    caption=(
                        f"Binary mask at localization threshold "
                        f"{result.localization_threshold:.2f} "
                        f"(retained {result.processed_suspicious_pixels:,} px; "
                        f"raw {result.raw_suspicious_pixels:,} px)"
                    ),
                    width="stretch",
                )
            with second[1]:
                st.image(
                    result.annotated,
                    caption="Highlighted overlay with numbered suspicious regions",
                    width="stretch",
                )
            st.caption(
                f"Analysis completed in {result.elapsed_seconds:.1f} s on {result.model_info.get('device', 'cpu')}. "
                f"Raw thresholded area: "
                f"{format_area(result.raw_suspicious_area_ratio, result.raw_suspicious_pixels)}. "
                f"Highlighted area after filtering: "
                f"{format_area(result.processed_suspicious_area_ratio, result.processed_suspicious_pixels)}."
            )

        with tabs[1]:
            if result.regions:
                table = pd.DataFrame([region.as_dict() for region in result.regions])
                st.dataframe(table, hide_index=True, width="stretch")
            elif result.raw_suspicious_pixels:
                st.info(
                    f"{result.raw_suspicious_pixels:,} pixel(s) crossed the "
                    f"localization threshold, but no connected component passed the "
                    "minimum-area filter. No region is drawn, because drawing one "
                    "would invent a shape the model did not support."
                )
            else:
                st.info("No pixel crossed the localization threshold.")
            st.markdown(
                '<div class="df-note">Region scores describe model attention, not a '
                "confirmed edit. Small or heavily compressed forgeries can be missed.</div>",
                unsafe_allow_html=True,
            )

        with tabs[2]:
            decision_rows = [
                {
                    "Setting": "Image decision threshold",
                    "Value": f"{result.image_threshold:.2f}",
                    "Applied to": "Maximum pixel probability (one score per image)",
                    "Purpose": "Authentic vs Tampered classification",
                },
                {
                    "Setting": "Uncertainty band",
                    "Value": f"±{result.image_uncertainty_margin:.2f}",
                    "Applied to": "Distance from the image decision threshold",
                    "Purpose": "Abstain when the score is too close to call",
                },
                {
                    "Setting": "Localization threshold",
                    "Value": f"{result.localization_threshold:.2f}",
                    "Applied to": "Every pixel probability",
                    "Purpose": "Binary mask and highlighted overlay only",
                },
            ]
            st.dataframe(pd.DataFrame(decision_rows), hide_index=True, width="stretch")
            area_rows = [
                {
                    "Measure": "Raw thresholded pixels",
                    "Pixels": f"{result.raw_suspicious_pixels:,}",
                    "Area": format_area(
                        result.raw_suspicious_area_ratio,
                        result.raw_suspicious_pixels,
                    ),
                    "Meaning": "Every pixel above the localization threshold",
                },
                {
                    "Measure": "Retained highlighted pixels",
                    "Pixels": f"{result.processed_suspicious_pixels:,}",
                    "Area": format_area(
                        result.processed_suspicious_area_ratio,
                        result.processed_suspicious_pixels,
                    ),
                    "Meaning": "Pixels inside components that passed the minimum-area filter",
                },
                {
                    "Measure": "Connected components",
                    "Pixels": f"{len(result.regions)} retained / {result.raw_component_count} raw",
                    "Area": "—",
                    "Meaning": "Post-processing can drop isolated pixels",
                },
            ]
            st.dataframe(pd.DataFrame(area_rows), hide_index=True, width="stretch")
            st.markdown(
                f"""
                **This image**

                - Image-level score (maximum pixel probability): `{result.image_score:.4f}`
                - Signed margin from the image threshold: `{result.image_decision_margin:+.4f}`
                - Image-level decision: **{result.image_decision_label}**
                - Pixel-level signal: **{result.localization_label}**
                - Raw thresholded area:
                  `{format_area(result.raw_suspicious_area_ratio, result.raw_suspicious_pixels)}`
                - Highlighted area after component filtering:
                  `{format_area(result.processed_suspicious_area_ratio, result.processed_suspicious_pixels)}`
                - Suspicious regions: `{len(result.regions)}`

                **Why this prediction?**

                {result.decision_explanation}

                The image-level label is a thresholded version of the same probability
                map used for localization. It is not a separately trained classifier and
                the score is not a calibrated confidence. The image threshold was chosen
                on the validation split using both authentic and tampered images; the test
                split was only used to report performance afterwards.
                """
            )
            if result.uncertain:
                st.warning(
                    "The score is inside the uncertainty band. The app abstains: the "
                    "image is not asserted to be authentic or tampered."
                )

        with tabs[3]:
            model_table = pd.DataFrame(
                {"Property": list(result.model_info.keys()), "Value": [str(value) for value in result.model_info.values()]}
            )
            st.dataframe(model_table, hide_index=True, width="stretch")
            render_limitations(result.warnings, result.limitations)

        st.write("")
        st.markdown("**Download results**")
        report = build_manipulation_report(result, fingerprint)
        download_row(
            "manipulation",
            {
                "Highlighted PNG": encode_png_bytes(result.annotated),
                "Mask PNG": encode_mask_png_bytes(result.binary_mask),
                "Heatmap PNG": encode_png_bytes(result.heatmap),
                "Report TXT": report.encode("utf-8"),
            },
        )


# ----------------------------------------------------------------------
# Page: Module 2
# ----------------------------------------------------------------------

def render_ai_page() -> None:
    st.markdown("## AI-Generated Image Detection")
    st.caption(
        "This module uses an independently trained detector supplied through a "
        "manifest. No weights are bundled, and no prediction is shown until a model "
        "is integrated."
    )

    status = get_ai_status()
    if status["available"]:
        st.markdown(
            badge("Detector integrated", "ok") + " &nbsp; " + badge(status["kind"], "neutral"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(badge("Integration placeholder", "warn"), unsafe_allow_html=True)

    rgb, fingerprint, info = upload_image(
        "upload_ai",
        "Upload an image",
        "JPG, PNG or WEBP. The image is analysed by the integrated detector, if one is present.",
    )

    analyze = st.button(
        "Analyse image",
        type="primary",
        disabled=rgb is None,
        key="btn_ai",
        use_container_width=True,
    )

    if rgb is not None and analyze:
        key = ("ai", fingerprint)
        result = _cache_get(key)
        if result is None:
            with st.spinner("Running AI-generated image detector…"):
                result = detect_ai_generated_image(rgb)
                _cache_put(key, result)
                if result.status == "ok":
                    record = record_from_ai(result, fingerprint)
                    append_history(record)
                    _append_session_history({"timestamp": record.timestamp, "module": record.module,
                                             "finding": record.finding, "score": record.score})
        st.session_state["last_ai"] = (fingerprint, result)

    payload = st.session_state.get("last_ai")
    if payload and payload[0] == fingerprint and rgb is not None:
        result = payload[1]
        if result.status == "unavailable":
            st.warning(result.reason or "No AI-generated image detector is integrated.")
            st.markdown(
                '<div class="df-note">This is a clearly labelled placeholder. The system '
                "does not use an untrained model and does not fabricate confidence scores.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("How to integrate a trained detector"):
                st.markdown(
                    """
                    1. Place the model file in `models/ai_detector/`.
                    2. Copy `models/ai_detector/manifest.example.json` to
                       `models/ai_detector/manifest.json`.
                    3. Declare the backend (`torchscript`, `onnx` or `python_adapter`),
                       class names, input size and normalisation.
                    4. Restart the app and run the evaluation script on a labelled dataset.

                    Full contract: `models/ai_detector/README.md`.
                    """
                )
            st.write(result.limitations)
            return

        if result.status == "error":
            st.error(result.error_message or "Module 2 failed.")
            render_limitations(result.warnings, result.limitations)
            return

        st.write("")
        label = {
            "real": "Real photograph",
            "ai_generated": "AI-generated image",
            "uncertain": "Uncertain — insufficient evidence",
        }.get(result.predicted_class or "", result.predicted_class or "Unknown")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Predicted class", label)
        metric_columns[1].metric("Real confidence", f"{(result.real_confidence or 0) * 100:.1f}%")
        metric_columns[2].metric("AI-generated confidence", f"{(result.ai_confidence or 0) * 100:.1f}%")
        st.markdown(finding_badge(result.predicted_class or "uncertain"), unsafe_allow_html=True)

        st.write("")
        st.progress(min(max(result.real_confidence or 0, 0.0), 1.0), text="Real-image confidence")
        st.progress(min(max(result.ai_confidence or 0, 0.0), 1.0), text="AI-generated confidence")
        st.caption(
            f"Backend: {result.backend or 'n/a'} · analysis time {result.elapsed_seconds:.1f} s. "
            "The uncertain state is an abstention, not a third learned class."
        )

        with st.expander("Backend details"):
            st.json(result.backend_info or {})
        render_limitations(result.warnings, result.limitations)
        report = build_ai_report(result, fingerprint)
        download_row("ai_generated", {"Report TXT": report.encode("utf-8")})


# ----------------------------------------------------------------------
# Page: Module 3
# ----------------------------------------------------------------------

def render_reference_comparison(comparison) -> None:
    """Render an aligned reference-vs-suspect comparison (Module 3)."""
    st.caption(
        "Differences are relative to the supplied reference. They do not establish which "
        "screenshot is the original, who edited it, or when."
    )
    kind = {
        "reference_differences_detected": "warn",
        "reference_no_differences_detected": "ok",
        "reference_mismatch": "bad",
    }.get(comparison.finding, "neutral")
    st.markdown(
        badge(comparison.finding.replace("_", " ").title(), kind), unsafe_allow_html=True
    )
    if comparison.status == "reference_invalid":
        st.error(comparison.error_message or "The reference screenshot could not be analysed.")
        return
    if comparison.status == "reference_mismatch":
        st.warning(
            "The screenshots could not be reliably aligned or differ almost everywhere. "
            "This usually means a different screen, theme, resolution or chat state — "
            "individual changes cannot be attributed."
        )

    alignment = comparison.alignment
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Alignment correlation",
        f"{alignment.correlation:.3f}" if alignment is not None else "—",
    )
    metric_columns[1].metric(
        "Changed pixels", f"{comparison.changed_pixel_ratio * 100:.4f}%"
    )
    metric_columns[2].metric("Changed regions", str(len(comparison.regions)))
    metric_columns[3].metric("Text differences", str(len(comparison.text_changes)))

    if comparison.side_by_side is not None:
        st.image(
            comparison.side_by_side,
            caption=(
                "Reference (left) and suspect (right). P = pixel-change region, "
                "D = OCR text difference."
            ),
            width="stretch",
        )
    visual_columns = st.columns(3, gap="medium")
    if comparison.heatmap is not None:
        with visual_columns[0]:
            st.image(comparison.heatmap, caption="Change heatmap", width="stretch")
    if comparison.overlay is not None:
        with visual_columns[1]:
            st.image(comparison.overlay, caption="Changes on suspect", width="stretch")
    if comparison.change_mask is not None:
        with visual_columns[2]:
            st.image(comparison.change_mask, caption="Binary change mask", width="stretch")

    if comparison.text_changes:
        rows = [
            {
                "#": change.change_id,
                "Category": change.kind.replace("_", " ").title(),
                "Where": change.location.title(),
                "Confidence": f"{change.confidence * 100:.0f}%",
                "Reference text": change.reference_text or "—",
                "Suspect text": change.suspect_text or "—",
            }
            for change in comparison.text_changes
        ]
        st.markdown("**Detected text differences (review cues, not proof)**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "These rows compare OCR transcripts. Recognition errors can appear as "
            "differences. Check the highlighted pixel regions before drawing a conclusion."
        )
    else:
        st.info(
            "No line-level OCR difference was detected. OCR can miss stylised, low-contrast "
            "or non-Latin text, so this is not proof that the screenshots are identical."
        )

    if comparison.reference_ocr is not None and comparison.reference_ocr.available:
        with st.expander("Reference transcript (not stored)"):
            st.text_area(
                "Reference OCR text",
                value=comparison.reference_ocr.full_text,
                height=180,
                key="reference_transcript",
            )

    st.markdown(
        '<div class="df-note">Re-rendering, theme changes, resizing and recompression can '
        "produce differences that are not tampering. A reference comparison is a strong "
        "review aid, not forensic proof of intent.</div>",
        unsafe_allow_html=True,
    )


def render_chat_page() -> None:
    st.markdown("## Chat Screenshot Analysis")
    st.caption(
        "Screens a messaging screenshot for possible visual manipulation and "
        "transcribes visible text. Message-level verification is only possible against "
        "a trusted original screenshot."
    )
    st.markdown(
        '<div class="df-note"><b>Scope:</b> visual screening only. A highlighted area '
        "means the forensic model found an anomalous region; it does not prove that a "
        "message was edited, and OCR text is never used as evidence of tampering. In the "
        "controlled synthetic evaluation, this out-of-domain screening flagged every "
        "authentic screenshot (false-positive rate 1.0), so treat it as a triage aid. Add "
        "an original reference screenshot below to compare visible differences.</div>",
        unsafe_allow_html=True,
    )

    rgb, fingerprint, info = upload_image(
        "upload_chat",
        "Upload a chat screenshot",
        "PNG or JPG screenshots from a messaging app. Avoid uploading sensitive conversations.",
    )

    reference_rgb = None
    reference_fingerprint = None
    with st.expander("Optional: compare against an original screenshot", expanded=False):
        st.caption(
            "Supply the genuine, unedited screenshot only. This is the only setting in "
            "which the app can report that specific messages or amounts differ from a "
            "reference. The reference is processed in memory and is never saved."
        )
        reference_rgb, reference_fingerprint, _reference_info = upload_image(
            "upload_chat_reference",
            "Upload the trusted original screenshot",
            "Same chat, same layout and, ideally, the same resolution as the screenshot above.",
        )

    with st.expander("Analysis options"):
        run_ocr = st.checkbox(
            "Extract visible text with OCR",
            value=True,
            help="First EasyOCR run may download model files (~100 MB). OCR never counts as tampering evidence.",
        )
        detailed = st.checkbox(
            "Detailed patch mode (slower)",
            value=False,
            help="Blends overlapping patches to preserve small UI details. Several times slower on CPU.",
        )

    analyze = st.button(
        "Analyse screenshot",
        type="primary",
        disabled=rgb is None,
        key="btn_chat",
        use_container_width=True,
    )

    if rgb is not None and analyze:
        key = (
            "chat",
            fingerprint,
            reference_fingerprint,
            round(_threshold(), 3),
            bool(detailed),
            bool(run_ocr),
        )
        result = _cache_get(key)
        if result is None:
            with st.spinner("Running visual screening and OCR… this can take 20–60 s on CPU."):
                result = detect_chat_tampering(
                    rgb,
                    threshold=_threshold(),
                    detailed=detailed,
                    max_tiles=12,
                    run_ocr=run_ocr,
                    predictor=cached_predictor(),
                    reference_rgb=reference_rgb,
                )
                _cache_put(key, result)
                if result.status in {"ok", "partial"}:
                    record = record_from_chat(result, fingerprint)
                    append_history(record)
                    _append_session_history({"timestamp": record.timestamp, "module": record.module,
                                             "finding": record.finding, "score": record.score})
        st.session_state["last_chat"] = (fingerprint, reference_fingerprint, result)

    payload = st.session_state.get("last_chat")
    if (
        payload
        and len(payload) == 3
        and payload[0] == fingerprint
        and payload[1] == reference_fingerprint
        and rgb is not None
    ):
        result = payload[2]
        if result.status == "error":
            st.error("Neither visual analysis nor OCR could complete.")
            if result.error_message:
                st.code(result.error_message)
            render_limitations(result.warnings, result.limitations)
            return
        if result.status == "partial":
            st.warning("Visual localization failed; only OCR results are available.")

        ocr_available = bool(result.ocr and result.ocr.available)
        metric_columns = st.columns(5)
        metric_columns[0].metric("Pixel-level signal", result.finding.replace("_", " ").title())
        metric_columns[1].metric("Max visual score", f"{result.visual_score:.4f}")
        metric_columns[2].metric("Suspicious regions", str(len(result.regions)))
        metric_columns[3].metric("OCR lines", str(len(result.ocr.lines) if ocr_available else 0))
        metric_columns[4].metric("OCR backend", result.ocr.backend if ocr_available else "unavailable")
        if result.status == "ok":
            st.markdown(localization_badge(result.finding), unsafe_allow_html=True)
            st.caption(
                "This badge is a pixel-level visual screening signal for the screenshot. "
                "It is not message-level verification and does not prove that the whole "
                "screenshot is authentic or manipulated. Measured on synthetic screenshots, "
                "the model flagged all authentic images, so do not rely on this badge alone."
            )
        st.progress(min(max(result.visual_score, 0.0), 1.0), text="Maximum visual anomaly score")

        comparison = result.reference_comparison
        tab_labels = ["Visual analysis", "OCR text", "Regions & text", "Limitations"]
        if comparison is not None:
            tab_labels.insert(3, "Reference comparison")
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            columns = st.columns(3, gap="medium")
            with columns[0]:
                st.image(rgb, caption="Original screenshot", width="stretch")
            if result.heatmap is not None:
                with columns[1]:
                    st.image(result.heatmap, caption="Visual anomaly heatmap", width="stretch")
                with columns[2]:
                    st.image(
                        result.combined_visual if result.combined_visual is not None else result.annotated,
                        caption="Highlighted regions + OCR boxes",
                        width="stretch",
                    )
                st.image(
                    result.mask_visual, caption="Binary mask at selected threshold", width="stretch"
                )
            else:
                st.info("No visual localization map is available for this screenshot.")

        with tabs[1]:
            if not ocr_available:
                st.warning(result.ocr.reason if result.ocr else "OCR is unavailable.")
                st.caption("Install EasyOCR or Tesseract to enable text extraction.")
            elif not result.ocr.lines:
                st.info("No readable text was detected.")
            else:
                st.caption(
                    f"{len(result.ocr.lines)} lines in {len(result.ocr.blocks)} visual blocks · "
                    f"backend: {result.ocr.backend} · OCR time {result.ocr.elapsed_seconds:.1f} s."
                )
                for block in result.ocr.blocks:
                    confidence = f"{block.confidence * 100:.0f}%"
                    st.markdown(
                        f'<div class="df-card" style="margin-bottom:.6rem">'
                        f'<div style="color:#5b6b82;font-size:.74rem;font-weight:700">'
                        f'BLOCK B{block.block_id} · confidence {confidence}</div>'
                        f'<div style="white-space:pre-wrap;color:#0f172a;margin-top:.3rem">'
                        f"{_html_escape(block.text)}</div></div>",
                        unsafe_allow_html=True,
                    )
                st.text_area(
                    "Plain transcript (copyable)",
                    value=result.ocr.full_text,
                    height=220,
                    key="ocr_transcript",
                )
                st.caption(
                    "OCR can misread stylised text. The transcript is not stored in history "
                    "and is not used as evidence of tampering."
                )

        with tabs[2]:
            if result.regions:
                region_rows = []
                matches = {match.region_id: match for match in result.region_matches}
                for region in result.regions:
                    row = region.as_dict()
                    match = matches.get(region.region_id)
                    row["overlapping OCR blocks"] = (
                        ", ".join(f"B{block}" for block in match.block_ids) if match else "—"
                    )
                    row["overlap"] = f"{match.overlap_ratio * 100:.0f}%" if match else "—"
                    region_rows.append(row)
                st.dataframe(pd.DataFrame(region_rows), hide_index=True, width="stretch")
                if result.region_matches and ocr_available:
                    st.markdown("**Text blocks overlapping highlighted regions**")
                    for match in result.region_matches:
                        for block_id in match.block_ids:
                            block = next((b for b in result.ocr.blocks if b.block_id == block_id), None)
                            if block:
                                st.markdown(f"**B{block_id}** · region R{match.region_id}")
                                st.text(block.text)
            else:
                st.info("No suspicious region crossed the threshold.")
            st.markdown(
                '<div class="df-note">Overlap between a highlighted region and a text box '
                "is a review cue only. The system cannot establish which message, if any, "
                "was edited.</div>",
                unsafe_allow_html=True,
            )

        with tabs[3]:
            if comparison is None:
                render_limitations(result.warnings, result.limitations)
            else:
                render_reference_comparison(comparison)

        if comparison is not None:
            with tabs[4]:
                render_limitations(result.warnings, result.limitations)

        st.write("")
        st.markdown("**Download results**")
        files: dict[str, bytes] = {"Report TXT": build_chat_report(result, fingerprint).encode("utf-8")}
        if result.combined_visual is not None:
            files["Highlighted PNG"] = encode_png_bytes(result.combined_visual)
        if result.heatmap is not None:
            files["Heatmap PNG"] = encode_png_bytes(result.heatmap)
        if result.mask_visual is not None:
            files["Mask PNG"] = encode_mask_png_bytes(result.manipulation.binary_mask)
        if comparison is not None:
            if comparison.side_by_side is not None:
                files["Reference comparison PNG"] = encode_png_bytes(comparison.side_by_side)
            if comparison.change_mask is not None:
                files["Reference change mask PNG"] = encode_mask_png_bytes(
                    comparison.change_mask
                )
        if ocr_available and result.ocr.full_text:
            files["Transcript TXT"] = result.ocr.full_text.encode("utf-8")
        download_row("chat", files)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ----------------------------------------------------------------------
# Page: History
# ----------------------------------------------------------------------

def render_history_page() -> None:
    st.markdown("## Analysis History")
    st.caption(
        "Metadata-only log of completed analyses. Screenshots, OCR text, chat content "
        "and file paths are never stored."
    )

    session_entries = st.session_state.get("session_history", [])
    persisted = load_history(limit=300)
    if not persisted and not session_entries:
        st.info("No analyses have been recorded yet. Run a module to populate history.")
        return

    if persisted:
        frame = pd.DataFrame(persisted)
        extras = (
            frame["extra"]
            if "extra" in frame.columns
            else pd.Series([{} for _ in range(len(frame))])
        )
        frame["image_result"] = [
            (
                entry.get("image_decision_label")
                or {
                    "authentic": "Authentic (Predicted)",
                    "tampered": "Tampered (Predicted)",
                    "uncertain": "Uncertain (Insufficient Evidence)",
                }.get(entry.get("image_decision"), entry.get("image_decision"))
                if isinstance(entry, dict)
                else None
            )
            for entry in extras
        ]
        frame["localization_signal"] = [
            (
                entry.get("localization_label")
                or LOCALIZATION_BADGES.get(
                    str(frame.at[index, "finding"]).lower(), (None, "neutral")
                )[0]
                if isinstance(entry, dict)
                else None
            )
            for index, entry in enumerate(extras)
        ]
        frame["raw_area"] = [
            (
                format_area(float(entry.get("raw_suspicious_area_ratio", 0.0)))
                if isinstance(entry, dict) and "raw_suspicious_area_ratio" in entry
                else None
            )
            for entry in extras
        ]
        frame["highlighted_area"] = [
            (
                format_area(
                    float(
                        entry.get(
                            "processed_suspicious_area_ratio",
                            entry.get("suspicious_area_ratio", 0.0),
                        )
                    )
                )
                if isinstance(entry, dict)
                and (
                    "processed_suspicious_area_ratio" in entry
                    or "suspicious_area_ratio" in entry
                )
                else None
            )
            for entry in extras
        ]
        display_columns = [
            column
            for column in [
                "timestamp",
                "module",
                "image_result",
                "localization_signal",
                "finding",
                "score",
                "threshold",
                "raw_area",
                "highlighted_area",
                "width",
                "height",
                "elapsed_seconds",
                "fingerprint",
            ]
            if column in frame.columns
        ]
        frame = frame[display_columns].rename(
            columns={
                "timestamp": "UTC time",
                "module": "Module",
                "image_result": "Image-level result",
                "localization_signal": "Pixel-level signal",
                "finding": "Legacy finding key",
                "score": "Score",
                "threshold": "Localization threshold",
                "raw_area": "Raw area",
                "highlighted_area": "Highlighted area",
                "width": "Width",
                "height": "Height",
                "elapsed_seconds": "Seconds",
                "fingerprint": "Content fingerprint",
            }
        )
        frame = frame.fillna("—")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(f"{len(frame)} persisted record(s). Newest first.")
        st.caption(
            "Image-level result is the Authentic/Tampered/Uncertain decision. The "
            "pixel-level signal is separate and only describes localized model "
            "attention. Legacy records may not contain the newer columns."
        )
    else:
        st.info("Session entries exist but nothing has been persisted yet.")

    with st.expander("Privacy details"):
        st.markdown(
            "- Only aggregate scores, dimensions, a content fingerprint and timing are stored.\n"
            "- No uploaded image, OCR transcript, screenshot or chat content is written to disk.\n"
            "- The content fingerprint is a truncated hash and cannot reconstruct the image.\n"
            "- Deleting `results/app/history.jsonl` clears persisted history."
        )

    confirm = st.checkbox("I understand that clearing history cannot be undone.", key="confirm_clear_history")
    if st.button("Clear persisted history", disabled=not confirm):
        clear_history()
        st.session_state["session_history"] = []
        st.success("Persisted history cleared. The page will refresh.")
        st.rerun()


# ----------------------------------------------------------------------
# Page: About
# ----------------------------------------------------------------------

def _load_evaluation_artifact(name: str) -> dict | None:
    path = settings.evaluation_dir / name
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def render_about_page() -> None:
    st.markdown("## About the Project")
    st.write(
        "An undergraduate-level digital forensics platform that integrates three "
        "independent detectors behind one interface. The emphasis is on visual "
        "explanation, honest confidence reporting and clear limits of evidence."
    )

    left, right = st.columns(2, gap="large")
    with left:
        section_title("System architecture")
        st.markdown(
            """
            - **Frontend:** Streamlit, multi-page sidebar dashboard.
            - **Backend:** Python, PyTorch, OpenCV, NumPy, Pillow, Pandas.
            - **Module 1:** MVSS-Net ResNet-50, CASIA2.0 checkpoint
              (`ckpt/mvssnet_casia.pt`), 512×512 input.
            - **Module 2:** manifest-driven TorchScript / ONNX / Python adapter.
            - **Module 3:** Module 1 localization + EasyOCR/Tesseract + region/text association.
            - **Isolation:** every module returns a structured result; failures are caught per module.
            """
        )
        section_title("Datasets")
        st.markdown(
            f"""
            - Training/evaluation dataset: CASIA2.0 revised (`Au/`, `Tp/`) with
              ground-truth masks, referenced through `DIGITAL_FORENSICS_DATA_ROOT`.
            - Dataset folder present: **{_yes_no(settings.dataset_dir.exists())}**.
            - Validation/test splits are read from `experiments/dataset_split.csv`;
              thresholds are selected on validation only.
            """
        )
    with right:
        section_title("Reproducible evaluation")
        metrics = _load_evaluation_artifact("module1_test_metrics.json")
        if not metrics:
            st.markdown(
                "No evaluation artifact is present yet. Generate one with:\n\n"
                "```bash\npython evaluation/evaluate_module1.py --split test\n```\n\n"
                "Module 2 and Module 3 evaluation scripts refuse to run without a "
                "labelled dataset and an integrated model. No results are fabricated."
            )
        else:
            st.caption(
                f"Source: `{metrics.get('artifact', 'module1_test_metrics.json')}` · "
                f"generated {metrics.get('generated_utc', 'unknown')} · "
                f"threshold {metrics.get('threshold', 'n/a')} (validation-selected)."
            )
            localization = metrics.get("localization", {})
            st.markdown("**Pixel-level localization (tampered test images)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": "Images", "Value": localization.get("images", "n/a")},
                        {"Metric": "Global IoU", "Value": _fmt_maybe(localization.get("global_iou"))},
                        {"Metric": "Mean image IoU", "Value": _fmt_maybe(localization.get("mean_image_iou"))},
                        {"Metric": "Micro precision", "Value": _fmt_maybe(localization.get("micro_precision"))},
                        {"Metric": "Micro recall", "Value": _fmt_maybe(localization.get("micro_recall"))},
                        {"Metric": "Micro F1", "Value": _fmt_maybe(localization.get("micro_f1"))},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            image_metrics = metrics.get("image_level", {})
            if image_metrics:
                st.markdown("**Image-level detection (prior estimate at localization threshold 0.40)**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Metric": "Authentic FPR", "Value": _fmt_maybe(image_metrics.get("false_positive_rate"))},
                            {"Metric": "Manipulated recall", "Value": _fmt_maybe(image_metrics.get("recall"))},
                            {"Metric": "Image F1", "Value": _fmt_maybe(image_metrics.get("f1"))},
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            selection = _load_evaluation_artifact("module1_image_level_threshold.json")
            if selection:
                selected = selection.get("selected_metrics", {})
                uncertainty = selection.get("uncertainty", {})
                samples = selection.get("samples", {})
                st.markdown("**Image-level threshold selected on validation only**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Metric": "Selected threshold", "Value": _fmt_maybe(selection.get("selected_threshold"))},
                            {"Metric": "Validation recall", "Value": _fmt_maybe(selected.get("recall"))},
                            {"Metric": "Validation FPR", "Value": _fmt_maybe(selected.get("false_positive_rate"))},
                            {"Metric": "Validation balanced accuracy", "Value": _fmt_maybe(selected.get("balanced_accuracy"))},
                            {"Metric": "Validation AUC", "Value": _fmt_maybe(selection.get("auc"))},
                            {"Metric": "Uncertainty band", "Value": f"±{float(uncertainty.get('margin', 0.0)):.2f}"},
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    f"Selected on {samples.get('authentic_scored', 'n/a')} authentic + "
                    f"{samples.get('tampered_scored', 'n/a')} tampered validation images. "
                    "The test split was not used for selection."
                )
            comparison = _load_evaluation_artifact(
                "module1_image_level_score_comparison.json"
            )
            if comparison:
                recommendation = comparison.get("recommendation", {})
                st.markdown("**Image-level score comparison (validation only)**")
                st.caption(
                    "Deployed score: maximum pixel probability. "
                    f"{recommendation.get('decision', '')}"
                )
            st.caption(metrics.get("provenance", "See docs/APPLICATION.md for the evaluation protocol."))

    section_title("Model limitations and responsible use")
    st.markdown(
        """
        - MVSS-Net outputs a pixel probability map, not a calibrated authenticity probability.
        - Performance drops on unseen cameras, social-media compression and synthetic images.
        - The AI-generated module is a labelled placeholder until trained weights are supplied;
          the application never invents scores for it.
        - Chat screenshots are out of domain for a CASIA-trained localization model. OCR
          transcribes pixels and cannot establish whether a message was edited.
        - No upload is written to disk; history contains metadata only.
        - The system is an academic decision-support demo, not a legal or journalistic verdict.
        """
    )

    section_title("Citation")
    st.code(
        "@InProceedings{MVSS_2021ICCV,\n"
        "  author = {Chen, Xinru and Dong, Chengbo and Ji, Jiaqi and Cao, Juan and Li, Xirong},\n"
        "  title = {Image Manipulation Detection by Multi-View Multi-Scale Supervision},\n"
        "  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},\n"
        "  year = {2021}\n"
        "}",
        language="bibtex",
    )
    st.caption("Digital Image Forensics · Streamlit application · Module isolation and privacy by design.")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _fmt_maybe(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

if page == "Home Dashboard":
    render_home()
elif page == "Image Manipulation Detection":
    render_manipulation_page()
elif page == "AI-Generated Image Detection":
    render_ai_page()
elif page == "Chat Screenshot Analysis":
    render_chat_page()
elif page == "Analysis History":
    render_history_page()
elif page == "About the Project":
    render_about_page()
