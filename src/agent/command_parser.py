from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TARGET_ALIASES = {
    # Chinese
    "沙发": "sofa",
    "沙发旁边": "sofa",
    "沙发边": "sofa",
    "床": "bed",
    "床边": "bed",
    "床旁边": "bed",
    "桌子": "table",
    "桌边": "table",
    "桌子旁边": "table",
    "餐桌": "table",
    "椅子": "chair",
    "椅子旁边": "chair",
    "厨房台面": "kitchen counter",
    "台面": "kitchen counter",

    # English
    "sofa": "sofa",
    "couch": "sofa",
    "bed": "bed",
    "table": "table",
    "desk": "table",
    "chair": "chair",
    "kitchen counter": "kitchen counter",
    "counter": "kitchen counter",
}


@dataclass
class ParsedCommand:
    raw_text: str
    target: Optional[str]
    ok: bool
    message: str


def parse_command(text: str) -> ParsedCommand:
    text = (text or "").strip()
    lowered = text.lower()

    # Match longer aliases first, e.g. "沙发旁边" before "沙发".
    aliases = sorted(TARGET_ALIASES.keys(), key=len, reverse=True)

    for alias in aliases:
        if alias.lower() in lowered:
            return ParsedCommand(
                raw_text=text,
                target=TARGET_ALIASES[alias],
                ok=True,
                message=f"parsed target = {TARGET_ALIASES[alias]}",
            )

    return ParsedCommand(
        raw_text=text,
        target=None,
        ok=False,
        message=(
            "暂时只支持：沙发 / 床 / 桌子 / 椅子 / 厨房台面。"
            "例如：请到沙发旁边。"
        ),
    )
