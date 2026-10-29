"""One-shot account logout, independent of scheduler and login recovery."""
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np

from module.base.button import Button
from module.exception import RequestHumanTakeover
from module.jiangnan.app import JiangnanApp
from module.logger import logger


LOGOUT_ASSETS = Path(__file__).resolve().parents[2] / 'assets/jiangnan/logout'


@lru_cache(maxsize=6)
def _template(name):
    encoded = np.frombuffer((LOGOUT_ASSETS / (name + '.png')).read_bytes(), np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def _find(image, name, region, partial=False):
    x1, y1, x2, y2 = region
    search = image[y1:y2, x1:x2]
    template = _template(name)
    if partial:
        # At the top scroll position, the panel clips the lower glyph strokes.
        template = template[:15]
    if any(a < b for a, b in zip(search.shape, template.shape)):
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < (0.94 if partial else 0.90):
        return None
    return (x1 + x, y1 + y, x1 + x + template.shape[1], y1 + y + template.shape[0])


def detect_logout_screen(image):
    """Return the next control and its rectangle, scaled to the input frame."""
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None, None
    height, width = image.shape[:2]
    if height < 360 or width < 640:
        return None, None
    normalized = image if (width, height) == (1280, 720) else cv2.resize(image, (1280, 720))
    gray = cv2.cvtColor(normalized[:, :, :3], cv2.COLOR_RGB2GRAY)

    def result(state, rect):
        return state, tuple(round(value * (width / 1280 if i % 2 == 0 else height / 720))
                            for i, value in enumerate(rect)) if rect else None

    if _find(gray, 'settings_title', (190, 55, 315, 245)):
        region = (710, 245, 1080, 670)
        logout = _find(gray, 'logout', region) or _find(gray, 'logout', region, partial=True)
        if logout:
            return result('logout', logout)
        if _find(gray, 'return', (940, 65, 1085, 130)):
            return 'scroll_account', None
        account = _find(gray, 'account', (205, 245, 300, 355))
        if account:
            return result('account', account)
        return None, None
    settings = _find(gray, 'settings', (1165, 385, 1260, 550))
    if settings:
        return result('settings', settings)
    more = _find(gray, 'more', (1170, 595, 1275, 680))
    if more:
        return result('more', more)
    return None, None


class JiangnanLogout(JiangnanApp):
    def run(self, timeout=30):
        logger.hr('JIANGNAN LOGOUT', level=0)
        self.device.screenshot()
        state, _ = detect_logout_screen(self.device.image)
        if state in (None, 'more'):
            if not self.ensure_ready():
                raise RequestHumanTakeover('Cannot reach city main page for logout')
        deadline = time.monotonic() + timeout
        attempts = {}
        while time.monotonic() < deadline:
            self.device.screenshot()
            state, rect = detect_logout_screen(self.device.image)
            if state == 'logout':
                self.device.click(Button(area=rect, color=(), button=rect, name='LOGOUT_ACCOUNT'))
                # Do not run startup handlers here: they would log back in.
                self.device.sleep(10.0)
                self.device.app_stop()
                logger.info('Jiangnan account logged out; game closed')
                return True
            if state is not None:
                attempts[state] = attempts.get(state, 0) + 1
                if attempts[state] > 3:
                    break
                if state == 'scroll_account':
                    height, width = self.device.image.shape[:2]
                    self.device.swipe((int(width * .70), int(height * .80)),
                                      (int(width * .70), int(height * .35)), duration=.5)
                else:
                    self.device.click(Button(area=rect, color=(), button=rect,
                                             name='LOGOUT_' + state.upper()))
            self.device.sleep(1.0)
        self.save_abnormal_screenshot('logout_control_not_found')
        raise RequestHumanTakeover('Logout control not found; game left open for inspection')


def run_logout(config_name):
    return JiangnanLogout(config_name).run()
