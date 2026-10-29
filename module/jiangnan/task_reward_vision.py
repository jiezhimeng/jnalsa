"""Daily reward recognition from the user's 1280x720 reference screenshots."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2] / 'assets/jiangnan/task_reward'
PORTRAIT = (936, 474, 984, 535)
TASK_ENTRY = (932, 588, 983, 644)
PANEL = (335, 72, 369, 252)
DAILY_TITLE = (677, 69, 805, 103)
CLAIM = (1010, 570, 1077, 632)
TAKE = (527, 537, 753, 580)


@lru_cache(maxsize=12)
def _template(name, area):
    image = cv2.imdecode(np.fromfile(str(ROOT / (name + '.png')), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Missing daily task reward reference: {name}')
    x1, y1, x2, y2 = area
    return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def _valid(image):
    return image is not None and image.shape == (720, 1280, 3)


def _match(image, name, area, search, threshold=.88):
    if not _valid(image):
        return None
    x1, y1, x2, y2 = search
    template = _template(name, area)
    region = image[y1:y2, x1:x2]
    scores = cv2.matchTemplate(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY),
                               cv2.cvtColor(template, cv2.COLOR_RGB2GRAY),
                               cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < threshold:
        return None
    h, w = template.shape[:2]
    return x1+x, y1+y, x1+x+w, y1+y+h


def _near(area, margin=5):
    x1, y1, x2, y2 = area
    return x1-margin, y1-margin, x2+margin, y2+margin


def _red_ratio(region):
    if region.size == 0:
        return 0.
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, 90, 80), (14, 255, 255))
    red |= cv2.inRange(hsv, (170, 90, 80), (179, 255, 255))
    return cv2.countNonZero(red) / red.size


def task_entry(image):
    """Find the book icon in the main navigation; exclude dimmed modal pages."""
    found = _match(image, 'prompt', TASK_ENTRY, (780, 550, 1190, 710))
    if found:
        x1, y1, x2, y2 = found
        actual = image[y1:y2, x1:x2].astype(np.float32)
        # Grayscale correlation alone also recognizes disabled/dimmed pages.
        error = np.abs(actual - _template('prompt', TASK_ENTRY)).mean()
        if error < 25 and _red_ratio(image[y1:y2, x1:x2]) > .35:
            return found
    return None


def reward_prompt(image):
    """Yan's portrait and red marker above a visible task entry."""
    entry = task_entry(image)
    if entry is None:
        return None
    face = _match(image, 'prompt', PORTRAIT, (790, 390, 1190, 575), .90)
    if face is None:
        return None
    x1, y1, x2, y2 = face
    # Require the red marker beside the portrait, not a map character alone.
    if _red_ratio(image[y1+40:min(y1+88, 720), x1+20:min(x1+74, 1280)]) < .10:
        return None
    if abs((entry[0]+entry[2])/2 - (x1+x2)/2) > 95:
        return None
    return entry


def prompt_kind(image):
    """Notice the same portrait changing from a red 'new' label to a tick."""
    face = _match(image, 'prompt', PORTRAIT, (790, 390, 1190, 575), .90)
    if face is None:
        return None
    x, y = face[:2]
    area = (PORTRAIT[0]+20, PORTRAIT[1]+40, PORTRAIT[0]+74, PORTRAIT[1]+88)
    actual = image[y+40:y+88, x+20:x+74]
    expected = _template('prompt', area)
    def mask(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        return ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 169)) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 80)
    a, b = mask(actual), mask(expected)
    overlap = np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))
    return 'claim' if overlap >= .50 else 'new'


def task_panel(image):
    return _match(image, 'daily_available', PANEL, _near(PANEL), .90)


def daily_page(image):
    return (task_panel(image) is not None
            and _match(image, 'daily_available', DAILY_TITLE, _near(DAILY_TITLE), .90) is not None)


def claim_state(image):
    """Color distinguishes the same white label on red and grey buttons."""
    if not daily_page(image):
        return 'unknown'
    for name, state in (('daily_available', 'available'), ('daily_done', 'done')):
        found = _match(image, name, CLAIM, _near(CLAIM), .90)
        if found is None:
            continue
        x1, y1, x2, y2 = found
        region = image[y1:y2, x1:x2]
        error = np.abs(region.astype(np.float32)-_template(name, CLAIM)).mean()
        red = _red_ratio(region)
        if error < 20 and ((state == 'available' and red > .50)
                           or (state == 'done' and red < .08)):
            return state
    return 'unknown'


def reward_take(image):
    """Return a click box only when the actual red '收下' label is visible."""
    found = _match(image, 'reward', TAKE, _near(TAKE, 12), .90)
    if found:
        x1, y1, x2, y2 = found
        if _red_ratio(image[y1:y2, x1:x2]) > .55:
            return found
    return None
