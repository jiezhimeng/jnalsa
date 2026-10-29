"""Shared network dialog recognition, independent of the current game page."""
from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np


class NetworkPopupInterrupted(BaseException):
    """Unwind a task to the scheduler, including broad gameplay error handlers."""


MESSAGE_AREA = (386, 276, 875, 360)
RETRY_AREA = (546, 477, 751, 539)


@lru_cache(maxsize=1)
def _message_template():
    path = Path(__file__).resolve().parents[2] / 'assets/jiangnan/popup/network_timeout.png'
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = MESSAGE_AREA
    return image[y1:y2, x1:x2]


def network_popup_visible(image):
    if image is None or image.shape[:2] != (720, 1280):
        return False
    x1, y1, x2, y2 = MESSAGE_AREA
    region = cv2.cvtColor(image[y1-4:y2+4, x1-4:x2+4], cv2.COLOR_RGB2GRAY)
    score = cv2.minMaxLoc(cv2.matchTemplate(
        region, _message_template(), cv2.TM_CCOEFF_NORMED,
    ))[1]
    return score >= 0.88


def network_retry_ready(image):
    """Grey countdown buttons must never be clicked, even if text matches."""
    x1, y1, x2, y2 = RETRY_AREA
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, 90, 80), (14, 255, 255))
    red |= cv2.inRange(hsv, (170, 90, 80), (179, 255, 255))
    return cv2.countNonZero(red) / red.size >= 0.55


def parse_retry_countdown(text):
    match = re.fullmatch(r'\s*重[试試]\s*[(（]\s*(\d{1,3})\s*[)）]\s*', str(text))
    return int(match.group(1)) if match else None


def read_retry_countdown(image):
    from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE

    x1, y1, x2, y2 = RETRY_AREA
    original = image[y1:y2, x1:x2]
    enlarged = cv2.resize(original, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    lines = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus([original, enlarged])
    values = [parse_retry_countdown(line) for line in lines]
    return values[0] if len(values) == 2 and values[0] is not None and values[0] == values[1] else None
