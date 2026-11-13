"""Generate a controlled, privacy-safe synthetic chat-screenshot dataset.

All names, messages, amounts and dates are fictional and rendered locally with
Pillow.  No real conversation is collected, downloaded or reproduced.  The
generator produces:

* authentic screenshots in three messenger themes;
* edited copies with changes to a header name, amount, message text, date or
  time;
* pixel-level ground-truth masks marking exactly the edited areas;
* expected transcripts for OCR quality measurement;
* a ``dataset_manifest``-compatible CSV.

Usage::

    python evaluation/build_synthetic_chat_dataset.py --conversations 8
    python evaluation/build_synthetic_chat_dataset.py --root D:\\ImageForensics_Datasets\\chat_synthetic_v1

The dataset is evaluation-only: every row is assigned to the ``test`` split and
is never used to train or tune any model.  Re-running the generator refuses to
overwrite an existing directory unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manifest.io import write_manifest
from dataset_manifest.schema import compute_image_hashes, validate_manifest
from utils.config import settings


CANVAS = (720, 1280)
LICENSE = "CC0-1.0 (locally generated synthetic data)"
DATASET_NAME = "chat_synthetic_v1"


# ---------------------------------------------------------------------------
# Themes and fonts
# ---------------------------------------------------------------------------

THEMES = {
    "ios_light": {
        "bg": (242, 242, 247),
        "header": (248, 248, 250),
        "header_line": (214, 214, 219),
        "header_text": (17, 17, 17),
        "header_sub": (110, 110, 115),
        "incoming": (229, 229, 234),
        "incoming_text": (10, 10, 10),
        "outgoing": (10, 132, 255),
        "outgoing_text": (255, 255, 255),
        "date_bg": (214, 214, 219),
        "date_text": (70, 70, 75),
        "status_text": (20, 20, 20),
        "accent": (10, 132, 255),
        "footer": (248, 248, 250),
        "footer_text": (140, 140, 145),
        "bubble_radius": 18,
    },
    "android_dark": {
        "bg": (18, 18, 18),
        "header": (31, 44, 52),
        "header_line": (44, 58, 66),
        "header_text": (240, 240, 240),
        "header_sub": (170, 180, 186),
        "incoming": (32, 44, 51),
        "incoming_text": (240, 240, 240),
        "outgoing": (0, 92, 75),
        "outgoing_text": (240, 240, 240),
        "date_bg": (45, 57, 64),
        "date_text": (200, 205, 210),
        "status_text": (235, 235, 235),
        "accent": (0, 168, 132),
        "footer": (31, 44, 52),
        "footer_text": (150, 160, 166),
        "bubble_radius": 12,
    },
    "whatsapp_light": {
        "bg": (236, 229, 221),
        "header": (7, 94, 84),
        "header_line": (6, 78, 70),
        "header_text": (255, 255, 255),
        "header_sub": (210, 235, 230),
        "incoming": (255, 255, 255),
        "incoming_text": (20, 20, 20),
        "outgoing": (220, 248, 198),
        "outgoing_text": (20, 20, 20),
        "date_bg": (255, 255, 255),
        "date_text": (110, 110, 110),
        "status_text": (255, 255, 255),
        "accent": (7, 94, 84),
        "footer": (240, 240, 240),
        "footer_text": (120, 120, 120),
        "bubble_radius": 10,
    },
}

_FONT_CANDIDATES = {
    False: (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    True: (
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
}

_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for candidate in _FONT_CANDIDATES[bold]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, size)
            break
    else:  # pragma: no cover - only on systems without any known font
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ---------------------------------------------------------------------------
# Fictional content
# ---------------------------------------------------------------------------

NAMES = [
    "Aarav Mehta",
    "Priya Sharma",
    "Daniel Cole",
    "Meera Nair",
    "Rahul Verma",
    "Sofia Rossi",
    "Kiran Rao",
    "James Lee",
]

ALT_NAMES = [
    "A. Mehta",
    "Priya S.",
    "Dan Cole",
    "M. Nair",
    "Rahul V.",
    "S. Rossi",
    "Kiran R.",
    "J. Lee",
]

DATES = ["12 May 2024", "03/06/2024", "28 Sep 2024", "17 Feb 2025"]

CONVERSATION_TEMPLATES = [
    {
        "messages": [
            ("left", "Hey, are we still meeting tomorrow at 6?"),
            ("right", "Yes! I will bring the notes."),
            ("left", "Perfect. I sent 500 for the tickets."),
            ("right", "Got it, thanks!"),
            ("left", "See you at the cafe."),
        ],
        "times": ["18:04", "18:06", "18:10", "18:12", "18:15"],
    },
    {
        "messages": [
            ("right", "Did the transfer go through?"),
            ("left", "Yes, 1200 arrived this morning."),
            ("right", "Great, I will book the room today."),
            ("left", "Please send the receipt tomorrow."),
            ("right", "Sure, sending it now."),
        ],
        "times": ["09:12", "09:15", "09:20", "09:24", "09:25"],
    },
    {
        "messages": [
            ("left", "The client approved the design."),
            ("right", "Excellent, that is a relief."),
            ("left", "Invoice is 2500 for this phase."),
            ("right", "I will forward it to accounts."),
        ],
        "times": ["14:02", "14:05", "14:09", "14:12"],
    },
    {
        "messages": [
            ("right", "Lunch on Friday?"),
            ("left", "Friday works. Same place as last time?"),
            ("right", "Yes, I will reserve a table."),
            ("left", "Thanks! I owe you 80."),
        ],
        "times": ["11:30", "11:33", "11:36", "11:38"],
    },
]


@dataclass
class Element:
    key: str
    text: str
    bbox: tuple[int, int, int, int]
    role: str


@dataclass
class Conversation:
    conversation_id: str
    profile: str
    header_name: str
    header_status: str
    date: str
    messages: list[tuple[str, str]]
    times: list[str]
    status_time: str


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _text_bbox(draw, xy, text, font) -> tuple[int, int, int, int]:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    return int(left), int(top), int(right), int(bottom)


def _draw_status_bar(draw, theme, time_text: str) -> Element:
    draw.rectangle([0, 0, CANVAS[0], 46], fill=theme["header"])
    font = _font(18, bold=True)
    bbox = _text_bbox(draw, (26, 12), time_text, font)
    draw.text((26, 12), time_text, font=font, fill=theme["status_text"])

    # Signal bars, Wi-Fi arc and battery: simple vector shapes, no external assets.
    base_x = CANVAS[0] - 150
    for index in range(4):
        height = 4 + index * 3
        draw.rectangle(
            [base_x + index * 7, 30 - height, base_x + index * 7 + 4, 30],
            fill=theme["status_text"],
        )
    draw.arc([base_x + 40, 14, base_x + 62, 34], start=210, end=330, fill=theme["status_text"], width=2)
    draw.ellipse([base_x + 49, 27, base_x + 53, 31], fill=theme["status_text"])
    draw.rounded_rectangle(
        [CANVAS[0] - 46, 15, CANVAS[0] - 18, 31], radius=3, outline=theme["status_text"], width=2
    )
    draw.rectangle([CANVAS[0] - 43, 18, CANVAS[0] - 25, 28], fill=theme["status_text"])
    draw.rectangle([CANVAS[0] - 16, 20, CANVAS[0] - 13, 26], fill=theme["status_text"])
    return Element("status_time", time_text, bbox, "status")


def _draw_header(draw, theme, conversation: Conversation) -> dict[str, Element]:
    draw.rectangle([0, 46, CANVAS[0], 136], fill=theme["header"])
    draw.line([0, 136, CANVAS[0], 136], fill=theme["header_line"], width=1)

    # Back chevron.
    draw.line([16, 82, 32, 92, 16, 102], fill=theme["accent"], width=3, joint="curve")

    # Avatar with initials.
    avatar_box = [50, 64, 110, 124]
    draw.ellipse(avatar_box, fill=theme["accent"])
    initials = "".join(part[0] for part in conversation.header_name.split()[:2]).upper()
    initials_font = _font(22, bold=True)
    initials_bbox = draw.textbbox((0, 0), initials, font=initials_font)
    initials_w = initials_bbox[2] - initials_bbox[0]
    initials_h = initials_bbox[3] - initials_bbox[1]
    draw.text(
        (80 - initials_w / 2, 94 - initials_h / 2 - 2),
        initials,
        font=initials_font,
        fill=(255, 255, 255),
    )

    name_font = _font(28, bold=True)
    name_bbox = _text_bbox(draw, (124, 66), conversation.header_name, name_font)
    draw.text((124, 66), conversation.header_name, font=name_font, fill=theme["header_text"])

    status_font = _font(16)
    status_bbox = _text_bbox(draw, (126, 102), conversation.header_status, status_font)
    draw.text(
        (126, 102), conversation.header_status, font=status_font, fill=theme["header_sub"]
    )

    # Call and menu glyphs.
    glyph_x = CANVAS[0] - 96
    draw.ellipse([glyph_x, 76, glyph_x + 28, 104], outline=theme["accent"], width=2)
    draw.arc([glyph_x + 8, 84, glyph_x + 20, 98], start=200, end=340, fill=theme["accent"], width=2)
    for dx in (-6, 0, 6):
        draw.ellipse([glyph_x + 62 + dx - 2, 88, glyph_x + 62 + dx + 2, 92], fill=theme["header_sub"])

    return {
        "header_name": Element("header_name", conversation.header_name, name_bbox, "name"),
    }


def _draw_date_pill(draw, theme, text: str) -> Element:
    font = _font(18, bold=True)
    width = int(draw.textlength(text, font=font))
    left = (CANVAS[0] - width) // 2 - 12
    box = [left, 154, left + width + 24, 184]
    draw.rounded_rectangle(box, radius=11, fill=theme["date_bg"])
    bbox = _text_bbox(draw, (left + 12, 160), text, font)
    draw.text((left + 12, 160), text, font=font, fill=theme["date_text"])
    return Element("date", text, bbox, "date")


def _draw_message(
    draw,
    theme,
    index: int,
    side: str,
    text: str,
    time_text: str,
    cursor_y: int,
) -> tuple[dict[str, Element], int]:
    body_font = _font(22)
    time_font = _font(17)
    padding_x = 16
    padding_y = 12
    max_bubble_width = int(CANVAS[0] * 0.62)
    lines = _wrap(draw, text, body_font, max_bubble_width - 2 * padding_x)
    line_height = 30
    text_width = max(int(draw.textlength(line, font=body_font)) for line in lines)
    bubble_width = min(max_bubble_width, text_width + 2 * padding_x)
    bubble_height = len(lines) * line_height + 2 * padding_y

    if side == "right":
        left = CANVAS[0] - 24 - bubble_width
    else:
        left = 24
    top = cursor_y
    box = [left, top, left + bubble_width, top + bubble_height]
    fill = theme["outgoing"] if side == "right" else theme["incoming"]
    draw.rounded_rectangle(box, radius=theme["bubble_radius"], fill=fill)

    text_color = theme["outgoing_text"] if side == "right" else theme["incoming_text"]
    text_bboxes = []
    for line_index, line in enumerate(lines):
        position = (left + padding_x, top + padding_y + line_index * line_height)
        text_bboxes.append(_text_bbox(draw, position, line, body_font))
        draw.text(position, line, font=body_font, fill=text_color)

    union = (
        min(item[0] for item in text_bboxes),
        min(item[1] for item in text_bboxes),
        max(item[2] for item in text_bboxes),
        max(item[3] for item in text_bboxes),
    )

    time_width = int(draw.textlength(time_text, font=time_font))
    time_x = (CANVAS[0] - 24 - time_width) if side == "right" else 28
    time_y = top + bubble_height + 4
    time_bbox = _text_bbox(draw, (time_x, time_y), time_text, time_font)
    draw.text((time_x, time_y), time_text, font=time_font, fill=theme["date_text"])

    elements = {
        f"msg_{index}": Element(f"msg_{index}", text, union, "message"),
        f"time_{index}": Element(f"time_{index}", time_text, time_bbox, "time"),
    }
    return elements, time_y + 26


def _draw_footer(draw, theme) -> None:
    top = CANVAS[1] - 92
    draw.rectangle([0, top, CANVAS[0], CANVAS[1]], fill=theme["footer"])
    draw.rounded_rectangle(
        [18, top + 16, CANVAS[0] - 76, top + 68],
        radius=24,
        fill=theme["bg"],
    )
    draw.text(
        (40, top + 30),
        "Message",
        font=_font(20),
        fill=theme["footer_text"],
    )
    center = (CANVAS[0] - 36, top + 42)
    draw.ellipse([center[0] - 20, center[1] - 20, center[0] + 20, center[1] + 20], fill=theme["accent"])
    draw.polygon(
        [
            (center[0] - 7, center[1]),
            (center[0] + 1, center[1] - 6),
            (center[0] + 1, center[1] + 6),
        ],
        fill=(255, 255, 255),
    )


def render_conversation(
    conversation: Conversation, overrides: dict[str, str] | None = None
) -> tuple[Image.Image, dict[str, Element]]:
    """Render one screenshot and return element bounding boxes for the mask."""

    overrides = overrides or {}
    theme = THEMES[conversation.profile]
    image = Image.new("RGB", CANVAS, theme["bg"])
    draw = ImageDraw.Draw(image)
    elements: dict[str, Element] = {}

    status_text = overrides.get("status_time", conversation.status_time)
    elements["status_time"] = _draw_status_bar(draw, theme, status_text)

    header_name = overrides.get("header_name", conversation.header_name)
    header_conversation = Conversation(
        conversation.conversation_id,
        conversation.profile,
        header_name,
        conversation.header_status,
        conversation.date,
        conversation.messages,
        conversation.times,
        conversation.status_time,
    )
    elements.update(_draw_header(draw, theme, header_conversation))

    elements["date"] = _draw_date_pill(draw, theme, overrides.get("date", conversation.date))

    cursor_y = 210
    for index, ((side, text), time_text) in enumerate(zip(conversation.messages, conversation.times)):
        message_text = overrides.get(f"msg_{index}", text)
        message_time = overrides.get(f"time_{index}", time_text)
        message_elements, cursor_y = _draw_message(
            draw, theme, index, side, message_text, message_time, cursor_y
        )
        elements.update(message_elements)

    _draw_footer(draw, theme)
    return image, elements


# ---------------------------------------------------------------------------
# Conversation and change construction
# ---------------------------------------------------------------------------


def build_conversations(count: int, rng: random.Random) -> list[Conversation]:
    conversations: list[Conversation] = []
    profiles = list(THEMES)
    for index in range(count):
        template = CONVERSATION_TEMPLATES[index % len(CONVERSATION_TEMPLATES)]
        name = NAMES[index % len(NAMES)]
        conversations.append(
            Conversation(
                conversation_id=f"conv_{index + 1:03d}",
                profile=profiles[index % len(profiles)],
                header_name=name,
                header_status="online",
                date=DATES[index % len(DATES)],
                messages=list(template["messages"]),
                times=list(template["times"]),
                status_time=f"{8 + index % 12:02d}:{rng.choice([5, 17, 32, 41, 56]):02d}",
            )
        )
    return conversations


def _amount_variant(text: str, rng: random.Random) -> str:
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return text
    value = int(digits)
    replacement = str(value * 10)
    return text.replace(digits, replacement, 1)


def plan_changes(
    conversation: Conversation, index: int, rng: random.Random
) -> tuple[dict[str, str], list[dict]]:
    """Return render overrides and the semantic expected changes."""

    overrides: dict[str, str] = {}
    expected: list[dict] = []
    kinds = ["header_name", "amount", "message_text", "time", "date"]
    primary = kinds[index % len(kinds)]

    if primary == "header_name":
        new_name = ALT_NAMES[index % len(ALT_NAMES)]
        if new_name == conversation.header_name:
            new_name = ALT_NAMES[(index + 3) % len(ALT_NAMES)]
        overrides["header_name"] = new_name
        expected.append(
            {
                "kind": "name_or_header",
                "original": conversation.header_name,
                "edited": new_name,
            }
        )
    elif primary == "amount":
        for message_index, (_side, text) in enumerate(conversation.messages):
            if any(char.isdigit() for char in text):
                edited = _amount_variant(text, rng)
                if edited != text:
                    overrides[f"msg_{message_index}"] = edited
                    expected.append(
                        {
                            "kind": "amount_or_number",
                            "original": text,
                            "edited": edited,
                        }
                    )
                    break
    elif primary == "message_text":
        swap_pairs = (
            ("tomorrow", "today"),
            ("notes", "files"),
            ("design", "draft"),
            ("receipt", "invoice"),
            ("Friday", "Thursday"),
            ("cafe", "office"),
        )
        for message_index, (_side, text) in enumerate(conversation.messages):
            for original_word, replacement_word in swap_pairs:
                if original_word in text:
                    edited = text.replace(original_word, replacement_word, 1)
                    overrides[f"msg_{message_index}"] = edited
                    expected.append(
                        {
                            "kind": "message_text",
                            "original": text,
                            "edited": edited,
                        }
                    )
                    break
            if overrides:
                break
        if not overrides:  # pragma: no cover - all templates contain a swap word
            message_index = max(
                range(len(conversation.messages)),
                key=lambda index: len(conversation.messages[index][1]),
            )
            text = conversation.messages[message_index][1]
            words = text.rstrip(".!?").split()
            if len(words) >= 2:
                edited = text.replace(words[-1], "updated", 1)
                overrides[f"msg_{message_index}"] = edited
                expected.append(
                    {"kind": "message_text", "original": text, "edited": edited}
                )
    elif primary == "time":
        message_index = len(conversation.messages) - 1
        original = conversation.times[message_index]
        hour, minute = original.split(":")
        edited = f"{(int(hour) + 1) % 24:02d}:{minute}"
        overrides[f"time_{message_index}"] = edited
        expected.append(
            {
                "kind": "time",
                "original": original,
                "edited": edited,
            }
        )
    elif primary == "date":
        original = conversation.date
        edited = "14 Jul 2025" if original != "14 Jul 2025" else "21 Aug 2025"
        overrides["date"] = edited
        expected.append(
            {"kind": "date_or_timestamp", "original": original, "edited": edited}
        )

    # Add a secondary amount change on every fourth conversation so that some
    # screenshots contain two independent edits.
    if index % 4 == 3 and primary != "amount":
        for message_index, (_side, text) in enumerate(conversation.messages):
            if any(char.isdigit() for char in text):
                edited = _amount_variant(text, rng)
                if edited != text:
                    overrides[f"msg_{message_index}"] = edited
                    expected.append(
                        {
                            "kind": "amount_or_number",
                            "original": text,
                            "edited": edited,
                        }
                    )
                    break
    return overrides, expected


def _union_box(
    first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    boxes = [box for box in (first, second) if box is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _transcript(elements: dict[str, Element]) -> str:
    ordered = sorted(elements.values(), key=lambda element: (element.bbox[1], element.bbox[0]))
    return "\n".join(element.text for element in ordered if element.text)


def _mask_for_boxes(
    size: tuple[int, int], boxes: list[tuple[int, int, int, int]], padding: int = 3
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(size[0] - 1, x2 + padding),
                min(size[1] - 1, y2 + padding),
            ],
            fill=255,
        )
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(root: Path, conversations_count: int, seed: int, force: bool) -> dict:
    if root.exists() and any(root.iterdir()):
        if not force:
            raise SystemExit(
                f"Refusing to overwrite non-empty dataset directory: {root}\n"
                "Pass --force to regenerate it."
            )
        shutil.rmtree(root)

    rng = random.Random(seed)
    for name in ("authentic", "manipulated", "masks", "ocr_refs"):
        (root / name).mkdir(parents=True, exist_ok=True)

    conversations = build_conversations(conversations_count, rng)
    records: list[dict] = []
    annotations: list[dict] = []

    for index, conversation in enumerate(conversations):
        authentic_image, authentic_elements = render_conversation(conversation)
        overrides, expected = plan_changes(conversation, index, rng)
        edited_image, edited_elements = render_conversation(conversation, overrides)

        authentic_path = root / "authentic" / f"{conversation.conversation_id}.png"
        edited_path = root / "manipulated" / f"{conversation.conversation_id}_edited.png"
        mask_path = root / "masks" / f"{conversation.conversation_id}_edited.png"
        authentic_ocr_path = root / "ocr_refs" / f"{conversation.conversation_id}.txt"
        edited_ocr_path = root / "ocr_refs" / f"{conversation.conversation_id}_edited.txt"

        authentic_image.save(authentic_path, optimize=True)
        edited_image.save(edited_path, optimize=True)
        authentic_ocr_path.write_text(_transcript(authentic_elements) + "\n", encoding="utf-8")

        boxes: list[tuple[int, int, int, int]] = []
        change_rows: list[dict] = []
        for item in expected:
            candidate_keys = [key for key, value in overrides.items() if value == item["edited"]]
            matched_key = next(
                (
                    key
                    for key in candidate_keys
                    if key in authentic_elements
                    and authentic_elements[key].text == item["original"]
                ),
                candidate_keys[0] if candidate_keys else None,
            )
            if matched_key is None:  # pragma: no cover - generator guard
                raise RuntimeError(
                    f"Could not locate edited/original element for {item!r} in {conversation.conversation_id}"
                )
            box = _union_box(
                authentic_elements[matched_key].bbox, edited_elements[matched_key].bbox
            )
            if box is not None:
                boxes.append(box)
            change_rows.append({**item, "element": matched_key, "bbox": list(box) if box else None})

        mask = _mask_for_boxes(edited_image.size, boxes)
        mask.save(mask_path, optimize=True)
        edited_ocr_path.write_text(_transcript(edited_elements) + "\n", encoding="utf-8")

        annotations.append(
            {
                "conversation_id": conversation.conversation_id,
                "profile": conversation.profile,
                "authentic_image": f"authentic/{conversation.conversation_id}.png",
                "manipulated_image": f"manipulated/{conversation.conversation_id}_edited.png",
                "mask": f"masks/{conversation.conversation_id}_edited.png",
                "authentic_transcript": f"ocr_refs/{conversation.conversation_id}.txt",
                "manipulated_transcript": f"ocr_refs/{conversation.conversation_id}_edited.txt",
                "width": edited_image.size[0],
                "height": edited_image.size[1],
                "changes": change_rows,
            }
        )

        for kind, path, label, mask_available in (
            ("Authentic", authentic_path, "Authentic", False),
            ("Tampered", edited_path, "Tampered", True),
        ):
            hashes = compute_image_hashes(path, include_decoded=True)
            records.append(
                {
                    "sample_id": f"synthetic-chat-{conversation.conversation_id}-{label.lower()}",
                    "dataset": DATASET_NAME,
                    "source_group": "unknown",
                    "split": "test",
                    "class_name": kind,
                    "image_path": path.relative_to(root).as_posix(),
                    "mask_path": (
                        mask_path.relative_to(root).as_posix() if mask_available else ""
                    ),
                    "pair_id": conversation.conversation_id,
                    "group_id": f"chat-synthetic:{conversation.conversation_id}",
                    "manipulation_type": (
                        ",".join(sorted({row["kind"] for row in change_rows}))
                        if mask_available
                        else ""
                    ),
                    "device_make": "",
                    "device_model": "",
                    "capture_type": "synthetic_render",
                    "width": hashes["width"],
                    "height": hashes["height"],
                    "format": "png",
                    "mask_available": mask_available,
                    "mask_type": "edited_text_regions" if mask_available else "",
                    "mask_quality": "synthetic_exact" if mask_available else "",
                    "license": LICENSE,
                    "provenance_url": "",
                    "sha256": hashes["sha256"],
                    "phash": hashes["phash"],
                    "dhash": hashes.get("dhash", ""),
                    "decoded_sha256": hashes.get("decoded_sha256", ""),
                    "file_size": hashes.get("file_size", ""),
                    "pixel_training_eligible": False,
                    "notes": "Synthetic, fictional conversation rendered locally with Pillow.",
                }
            )

    metadata = {
        "artifact": f"{DATASET_NAME}_annotations.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "license": LICENSE,
        "privacy": (
            "Every name, message, amount and date is fictional and rendered locally. "
            "No real conversation was collected or downloaded."
        ),
        "seed": seed,
        "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "profiles": sorted(THEMES),
        "conversations": annotations,
        "evaluation_protocol": (
            "Evaluation-only synthetic set. All rows are in the test split and were never "
            "used for training or threshold selection."
        ),
    }
    (root / "annotations.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = write_manifest(records, root / "manifest.csv")
    validation = validate_manifest(records, data_root=root, strict=False, check_paths=True)
    (root / "manifest_validation.json").write_text(
        json.dumps(validation.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if validation.errors:
        raise RuntimeError(
            f"Generated manifest failed validation with {len(validation.errors)} error(s); "
            f"see {root / 'manifest_validation.json'}"
        )
    return {
        "root": root,
        "manifest": manifest_path,
        "conversations": len(annotations),
        "records": len(records),
        "validation": validation.summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=settings.data_root / DATASET_NAME,
        help="Dataset output directory (default: <data root>/chat_synthetic_v1).",
    )
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate an existing directory.")
    args = parser.parse_args()

    if args.conversations < 1:
        parser.error("--conversations must be at least 1")

    result = build_dataset(args.root, args.conversations, args.seed, args.force)
    print("Synthetic chat dataset generated")
    print(f"  root: {result['root']}")
    print(f"  conversations: {result['conversations']}")
    print(f"  manifest rows: {result['records']}")
    print(f"  manifest: {result['manifest']}")
    print(f"  validation: {json.dumps(result['validation'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
