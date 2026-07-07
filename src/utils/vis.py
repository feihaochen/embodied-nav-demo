from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def draw_debug_overlay(
    rgb: np.ndarray,
    debug: Optional[Dict[str, Any]] = None,
    reply: str = "",
) -> np.ndarray:
    """
    Draw bbox + state text on an RGB image.

    Input and output are RGB arrays.
    """
    if rgb.dtype != np.uint8:
        img = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    else:
        img = rgb.copy()

    debug = debug or {}

    # Draw detection box if present.
    bbox = debug.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        score = debug.get("score")
        label = debug.get("target") or debug.get("label") or "target"
        text = f"{label}"
        if score is not None:
            text += f" {float(score):.2f}"

        cv2.putText(
            img,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"state: {debug.get('state', '-')}",
        f"action: {debug.get('action', '-')}",
        f"target: {debug.get('target', '-')}",
        f"visible: {debug.get('target_visible', '-')}",
    ]

    if debug.get("front_depth") is not None:
        lines.append(f"front_depth: {float(debug['front_depth']):.2f}m")

    if debug.get("target_distance") is not None:
        lines.append(f"target_dist: {float(debug['target_distance']):.2f}m")

    if reply:
        lines.append(f"reply: {reply[:45]}")

    y = 22
    for line in lines:
        cv2.putText(
            img,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22

    return img
