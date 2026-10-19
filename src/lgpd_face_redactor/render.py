"""Obscures regions of a frame. Pixelation is the default; `box` is the strongest option."""

from __future__ import annotations

import cv2
import numpy as np

from lgpd_face_redactor.types import Box, Style


def _clip(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1 = max(0, int(box.x)), max(0, int(box.y))
    x2, y2 = min(width, round(box.x + box.w)), min(height, round(box.y + box.h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def obscure(frame: np.ndarray, boxes: list[Box], style: Style = Style.PIXELATE, blocks: int = 8) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    for box in boxes:
        clipped = _clip(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        region = out[y1:y2, x1:x2]
        if style is Style.BOX:
            region[:] = 0
        elif style is Style.BLUR:
            k = max(31, ((max(region.shape[:2]) // 2) | 1))  # strong, odd kernel scaled to the face
            region[:] = cv2.GaussianBlur(region, (k, k), 0)
        else:
            small = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_AREA)
            region[:] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out
