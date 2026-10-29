"""Visible assignee cards and the upper-left 已安排 ribbon."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _arranged_ribbon():
    path = Path(__file__).resolve().parents[2] / 'assets/jiangnan/resident/arranged.png'
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def resident_is_arranged(image, x, y):
    """Check the corner of a card whose name row starts near (x, y)."""
    if image is None:
        return False
    # Rows can move slightly while scrolling. The ribbon sits above and to
    # the left of the portrait, away from talent and level badges.
    corner = image[max(0, y - 25):y + 60, max(0, x - 20):x + 65]
    template = _arranged_ribbon()
    if (template is None or corner.shape[0] < template.shape[0]
            or corner.shape[1] < template.shape[1]):
        return False
    gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
    score = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))[1]
    return score >= 0.82


def unarranged_assignee_cards(image):
    """Return visible, coloured portraits without 已安排, in reading order.

    Only use this on pages with a direct-start tile in the first slot.
    """
    if image is None or image.shape[0] < 650 or image.shape[1] < 1080:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark_fraction = (gray[170:640, 325:1080] < 105).mean(axis=1)
    dividers = []
    for offset in (dark_fraction > 0.50).nonzero()[0]:
        y = int(offset) + 170
        if not dividers or y - dividers[-1] > 4:
            dividers.append(y)
    tops = [55] + [y + 18 for y in dividers if y + 18 < 570]
    cards = []
    for row, y in enumerate(tops):
        for col, x in enumerate((334, 715)):
            if row == 0 and col == 0:
                continue
            portrait = image[y + 30:y + 147, x:x + 116]
            hsv = cv2.cvtColor(portrait, cv2.COLOR_RGB2HSV)
            if float((hsv[:, :, 1] > 55).mean()) <= 0.16:
                continue
            if not resident_is_arranged(image, x, y):
                cards.append((x, y))
    return cards
