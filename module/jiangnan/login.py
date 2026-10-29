"""Recognize login controls independently of the animated title scenery."""
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np

from module.base.button import Button
from module.exception import RequestHumanTakeover
from module.logger import logger


LOGIN_ASSETS = Path(__file__).resolve().parents[2] / 'assets/jiangnan/login'


@lru_cache(maxsize=4)
def _control_template(name):
    encoded = np.frombuffer((LOGIN_ASSETS / (name + '.png')).read_bytes(), np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _find_control(image, name, region, similarity=0.91):
    x1, y1, x2, y2 = region
    search = image[y1:y2, x1:x2, :3]
    template = _control_template(name)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    scores = cv2.matchTemplate(cv2.cvtColor(search, cv2.COLOR_RGB2GRAY),
                               cv2.cvtColor(template, cv2.COLOR_RGB2GRAY),
                               cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < similarity:
        return None
    patch = search[y:y + template.shape[0], x:x + template.shape[1]]
    # A dimmed login form behind an authorization overlay is not clickable.
    if np.abs(patch.astype(np.int16) - template.astype(np.int16)).mean() > 32:
        return None
    return (x + x1, y + y1, x + x1 + template.shape[1], y + y1 + template.shape[0])


def detect_login_screen(image):
    """Return (state, click rectangle) in screenshot coordinates."""
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None, None
    height, width = image.shape[:2]
    if height < 360 or width < 640:
        return None, None
    normalized = (image if (width, height) == (1280, 720)
                  else cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA))

    def scaled(rect):
        return tuple(round(value * (width / 1280 if index % 2 == 0 else height / 720))
                     for index, value in enumerate(rect))

    if _find_control(normalized, 'auth_loading', (550, 285, 735, 440)):
        return 'auth_loading', None
    title = _find_control(normalized, 'sms_title', (325, 90, 955, 300))
    if title:
        tap = _find_control(normalized, 'taptap', (600, 530, 790, 700))
        if tap:
            return 'taptap', scaled(tap)
        # The SMS form moves upward when social login is expanded. The
        # collapsed form's title is the anchor for its "other methods" link;
        # no scenery or transparent link background participates in matching.
        x, y = title[:2]
        if y > 195:
            return 'other_methods', scaled((x + 150, y + 392, x + 249, y + 420))
        return 'login_transition', None
    if _find_control(normalized, 'customer_service', (295, 605, 390, 710)):
        bar = normalized[656:675, 410:920, :3]
        dark = (bar.max(axis=2) < 100)
        red = (bar[:, :, 0] > 100) & (bar[:, :, 0] > bar[:, :, 1] * 1.4)
        if (dark | red).mean() > 0.55:
            return 'game_loading', None
    return None, None


class JiangnanLogin:
    LOGIN_TIMEOUT = 120
    LOGIN_CLICK_INTERVAL = 8
    LOGIN_MAX_CLICKS = 3

    def handle_login(self):
        state, rectangle = detect_login_screen(getattr(self.device, 'image', None))
        now = time.time()
        deadline = getattr(self, '_login_deadline', 0)
        if state is None:
            if not deadline:
                return False
            # A loaded game can open signin/reward/tax before its first main
            # frame. Hand those modals back to the startup dispatcher instead
            # of consuming every frame as an authorization transition.
            game_popup = False
            for name in ('is_signin_popup', 'is_reward_popup', 'is_tax_popup'):
                detector = getattr(self, name, None)
                if callable(detector) and detector():
                    game_popup = True
                    break
            if game_popup or self.is_unobstructed_main():
                self._login_deadline = 0
                self._login_clicks = {}
                self._login_last_click = {}
                self._login_state = None
                logger.info('Jiangnan login finished; game page ready')
                return False
            # Authorization may briefly replace the form with a blank frame.
            # Preserve the login wait until a fresh main frame or the deadline.
            state = 'login_transition'
        if not deadline:
            deadline = now + self.LOGIN_TIMEOUT
            self._login_deadline = deadline
            self._login_clicks = {}
            self._login_last_click = {}
        if now >= deadline:
            self._login_deadline = 0
            save = getattr(self, 'save_abnormal_screenshot', None)
            if callable(save):
                save('taptap_login_timeout')
            raise RequestHumanTakeover(
                f'TapTap login did not finish within {self.LOGIN_TIMEOUT} seconds'
            )

        if state != getattr(self, '_login_state', None):
            logger.info(f'Jiangnan login state: {state}')
            self._login_state = state
        if rectangle is not None:
            count = self._login_clicks.get(state, 0)
            last = self._login_last_click.get(state, float('-inf'))
            if count < self.LOGIN_MAX_CLICKS and now - last >= self.LOGIN_CLICK_INTERVAL:
                name = 'LOGIN_TAPTAP' if state == 'taptap' else 'LOGIN_OTHER_METHODS'
                self.device.click(Button(area=rectangle, color=(), button=rectangle, name=name))
                self._login_clicks[state] = count + 1
                self._login_last_click[state] = now
        clear = getattr(self.device, 'stuck_record_clear', None)
        if callable(clear):
            clear()
        self.device.sleep(1.0)
        return True
