import math
import json
import re
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

from module.base.button import Button
from module.base.utils import load_image
from module.jiangnan.assets import (
    GRAND_CANAL_ASSIGNEE_CHECK,
    GRAND_CANAL_CURRENT_LOCATION_OK,
    GRAND_CANAL_DOCK_PANEL_CHECK,
    GRAND_CANAL_ENTER_BUTTON,
    GRAND_CANAL_EVENT_CHECK,
    GRAND_CANAL_BIG_ORDER_CHECK,
    GRAND_CANAL_EXIT_BUTTON,
    GRAND_CANAL_LOCATE_BUTTON,
    GRAND_CANAL_MOVE_DIALOG_CHECK,
    GRAND_CANAL_PAGE_CHECK,
    GRAND_CANAL_REWARD_CHECK,
    GRAND_CANAL_SAILING_DIALOG_CHECK,
    GRAND_CANAL_SAILING_OK,
    GRAND_CANAL_SECOND_TASK_CARD_CHECK,
    GRAND_CANAL_TASK_CARD_CHECK,
    GRAND_CANAL_WORKING_CHECK,
    GRAND_CANAL_STEPS_COUNT,
    GRAND_CANAL_SINAN_COUNT,
    GRAND_CANAL_SINAN_BUTTON,
    GRAND_CANAL_COMPASS_CHECK,
    GRAND_CANAL_COMPASS_STEPS_COUNT,
    GRAND_CANAL_COMPASS_SPIN,
    GRAND_CANAL_TASK_BUTTON,
    GRAND_CANAL_SINAN_TASK_CHECK,
    GRAND_CANAL_SINAN_OVERFLOW_CHECK,
    GRAND_CANAL_SINAN_CLAIM,
    GRAND_CANAL_SINAN_CLAIM_DONE,
    GRAND_CANAL_SINAN_REWARD,
    GRAND_CANAL_RESOURCE_SHORTAGE_CHECK,
)
from module.jiangnan.farm_production import JiangnanFarmProduction
from module.jiangnan.canal_route import (
    choose_next, custom_route_position, load_topology, locate_ship, parse_custom_route,
    map_visible_destinations, occupied_route_nodes, custom_route_destination,
    route_approach_path, blue_point_translation,
)
from module.jiangnan.foundation_icons import find_foundation_icon
from module.jiangnan.inventory_state import ALL_RESOURCES, JiangnanInventoryState
from module.jiangnan.ocr import Digit, Ocr
from module.logger import logger


class JiangnanYangzhouGrandCanal(JiangnanFarmProduction):
    """扬州府漕运码头及大运河自动航行、运输和收获。"""

    TASK = 'JiangnanYangzhouGrandCanal'
    GROUP = 'yangzhou_grand_canal'
    CITY = '扬州府'
    BUILDING = '漕运码头'
    RETRY = timedelta(minutes=10)
    DEFAULT_TIME = datetime(2020, 1, 1)
    MAX_ROUTE_STEPS = 16
    SHIP_CENTER = (640, 360)

    REWARD_CARD_CENTERS = (393, 555, 718, 880)
    DOCK_FOUNDATION_REFERENCE = './assets/jiangnan/grand_canal/foundation_reference.png'
    DOCK_FOUNDATION_ICON_AREA = (616, 480, 664, 530)

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device, task=self.TASK)
        self.working_time_ocr = Ocr(
            Button(
                area=(1160, 225, 1278, 272), color=(),
                button=(1160, 225, 1278, 272),
                name='GRAND_CANAL_WORKING_TIME',
            ),
            lang='cnocr',
            name='GRAND_CANAL_WORKING_TIME_OCR',
        )
        self.second_working_time_ocr = Ocr(
            Button(
                area=(1160, 343, 1278, 390), color=(),
                button=(1160, 343, 1278, 390),
                name='GRAND_CANAL_SECOND_WORKING_TIME',
            ),
            lang='cnocr', name='GRAND_CANAL_SECOND_WORKING_TIME_OCR',
        )
        self.ship_working_time_ocr = Ocr(
            Button(area=(410, 575, 780, 635), color=(), button=(410, 575, 780, 635),
                   name='GRAND_CANAL_SHIP_WORKING_TIME'),
            lang='cnocr', name='GRAND_CANAL_SHIP_WORKING_TIME_OCR',
        )
        self.ship_repair_title_ocr = Ocr(
            Button(area=(450, 165, 825, 230), color=(), button=(450, 165, 825, 230),
                   name='GRAND_CANAL_SHIP_REPAIR_TITLE'),
            lang='cnocr', name='GRAND_CANAL_SHIP_REPAIR_TITLE_OCR',
        )
        self.ship_repair_time_ocr = Ocr(
            Button(area=(400, 360, 650, 410), color=(), button=(400, 360, 650, 410),
                   name='GRAND_CANAL_SHIP_REPAIR_TIME'),
            lang='cnocr', name='GRAND_CANAL_SHIP_REPAIR_TIME_OCR',
        )
        self._dock_foundation_icon = load_image(
            self.DOCK_FOUNDATION_REFERENCE, self.DOCK_FOUNDATION_ICON_AREA,
        )
        self.steps_ocr = Digit(GRAND_CANAL_STEPS_COUNT, lang='cnocr',
                               name='GRAND_CANAL_STEPS_OCR')
        self.sinan_ocr = Digit(GRAND_CANAL_SINAN_COUNT, lang='cnocr',
                               name='GRAND_CANAL_SINAN_OCR')
        self.compass_steps_ocr = Digit(
            GRAND_CANAL_COMPASS_STEPS_COUNT, lang='cnocr',
            name='GRAND_CANAL_COMPASS_STEPS_OCR',
        )

    def _key(self, field):
        return f'{self.TASK}.{self.GROUP}.{field}'

    def _get(self, field, default=None):
        return self.config.cross_get(self._key(field), default=default)

    def _set(self, field, value):
        self.config.modified[self._key(field)] = value

    def _set_schedule(self, next_run, status):
        next_run = next_run.replace(microsecond=0)
        self._set('NextRun', next_run)
        self._set('Status', status)
        self.config.update()
        claim_due = self._next_sinan_claim_due()
        if claim_due <= datetime.now() and getattr(self, '_sinan_claim_failed', False):
            claim_due = datetime.now().replace(microsecond=0) + self.RETRY
        self.config.task_delay(target=min(next_run, claim_due))
        logger.attr('GrandCanalNextRun', next_run)
        logger.attr('GrandCanalStatus', status)

    def _next_sinan_claim_due(self, now=None):
        now = now or datetime.now().replace(microsecond=0)
        raw = str(self._get('SinanClaimTime', '11:00'))
        try:
            hour, minute = map(int, raw.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(raw)
        except (TypeError, ValueError):
            logger.warning(f'Invalid Sinan claim time {raw!r}; use 11:00')
            hour, minute = 11, 0
        due = now.replace(hour=hour, minute=minute, second=0)
        claimed = self._get('LastSinanClaimDate', '')
        if isinstance(claimed, datetime):
            claimed = claimed.date().isoformat()
        if claimed == now.date().isoformat():
            return due + timedelta(days=1)
        return due if due > now else now

    def _read_canal_resources(self):
        """Read both changing numbers from the visible map, never infer from a status board."""
        if not self._is_canal_page():
            logger.warning('Grand Canal resource read attempted off the map')
            self.save_abnormal_screenshot('grand_canal_resources_off_map')
            return None
        readings = []
        for attempt in range(2):
            if attempt:
                self.device.sleep(0.25)
                self.device.screenshot()
            steps = self.steps_ocr.ocr(self.device.image)
            sinan = self.sinan_ocr.ocr(self.device.image)
            if not (isinstance(steps, int) and isinstance(sinan, int)
                    and 0 <= steps <= 9999 and 0 <= sinan <= 9999):
                logger.warning(f'Grand Canal resource OCR invalid: {steps}, {sinan}')
                return None
            readings.append((steps, sinan))
        if readings[0] != readings[1]:
            logger.warning(f'Grand Canal resource OCR disagreed: {readings}')
            return None
        logger.attr('GrandCanalResources', f'steps={steps}, sinan={sinan}')
        return readings[0]

    def _is_sinan_task_page(self):
        return self.appear(GRAND_CANAL_SINAN_TASK_CHECK,
                           offset=(5, 5), similarity=0.76)

    def _is_compass_panel(self):
        return self.appear(GRAND_CANAL_COMPASS_CHECK,
                           offset=(5, 5), similarity=0.76)

    def _wait_sinan_task_page(self, timeout=8):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self._is_sinan_task_page():
                return True
        return False

    def _return_from_sinan_tasks(self):
        """Close task/reward panels and confirm the unobstructed canal map."""
        for _ in range(2):
            self.device.screenshot()
            if self._wait_canal_page(timeout=2) is True:
                return True
            self.device.press_back()
        return self._wait_canal_page(timeout=8) is True

    def _claim_daily_sinan(self):
        """Claim once after the configured time; retry later on an uncertain UI."""
        now = datetime.now().replace(microsecond=0)
        self._sinan_claim_deferred = False
        if self._next_sinan_claim_due(now) > now:
            return True
        if not self._is_canal_page():
            return False
        self.device.click(GRAND_CANAL_TASK_BUTTON)
        if not self._wait_sinan_task_page():
            logger.warning('Grand Canal Sinan task page not confirmed')
            self.save_abnormal_screenshot('grand_canal_sinan_task_unconfirmed')
            self._return_from_sinan_tasks()
            return False
        done = self._sinan_claim_button_state() == 'done'
        if not done:
            if self._sinan_claim_button_state() != 'ready':
                logger.warning('Grand Canal claim button state unknown')
                self.save_abnormal_screenshot('grand_canal_sinan_claim_unknown')
                self._return_from_sinan_tasks()
                return False
            self.device.click(GRAND_CANAL_SINAN_CLAIM)
            reward_deadline = time.time() + 8
            while time.time() < reward_deadline:
                self.device.sleep(0.35)
                self.device.screenshot()
                if self.appear(GRAND_CANAL_SINAN_OVERFLOW_CHECK,
                               offset=(5, 5), similarity=0.80):
                    # Cancel the warning instead of accepting a lossy claim.
                    # Leave the claim date unset so it can be retried after
                    # navigation has consumed enough Sinan.
                    self._sinan_claim_deferred = True
                    logger.info('Grand Canal Sinan claim deferred: would overflow')
                    self.device.press_back()
                    self._return_from_sinan_tasks()
                    return False
                if self.appear(GRAND_CANAL_SINAN_REWARD,
                               offset=(5, 5), similarity=0.80):
                    break
            else:
                logger.warning('Grand Canal Sinan reward popup not confirmed')
                self.save_abnormal_screenshot('grand_canal_sinan_reward_unconfirmed')
                self._return_from_sinan_tasks()
                return False
            self.device.click(GRAND_CANAL_SINAN_REWARD)
            done_deadline = time.time() + 8
            while time.time() < done_deadline:
                self.device.sleep(0.35)
                self.device.screenshot()
                if (self._is_sinan_task_page()
                        and self._sinan_claim_button_state() == 'done'):
                    done = True
                    break
        if done:
            self._set('LastSinanClaimDate', now.date().isoformat())
            logger.info('Grand Canal daily Sinan claim checked')
        map_ready = self._return_from_sinan_tasks()
        return bool(done and map_ready)

    def _sinan_claim_button_state(self):
        """The label is identical in both states; its fill changes red to grey."""
        patch = self.device.image[593:661, 1002:1069]
        if patch.shape[:2] != (68, 67):
            return 'unknown'
        red = ((patch[:, :, 0] > 110)
               & (patch[:, :, 0] > patch[:, :, 1].astype(np.float32) * 1.3)
               & (patch[:, :, 1] < 130))
        count = int(np.count_nonzero(red))
        logger.attr('GrandCanalSinanClaimRedPixels', count)
        if count > 1000:
            return 'ready'
        if count < 300:
            return 'done'
        return 'unknown'

    def _compass_step_count(self):
        value = self.compass_steps_ocr.ocr(self.device.image)
        if isinstance(value, int) and 0 <= value <= 9999:
            return value
        logger.warning(f'Grand Canal compass step OCR invalid: {value!r}')
        return None

    def _wait_compass_step_gain(self, before, timeout=40):
        """The top counter advances only after the compass has finished."""
        deadline = time.time() + timeout
        last = None
        stable = 0
        while time.time() < deadline:
            self.device.sleep(0.5)
            self.device.screenshot()
            if not self.appear(GRAND_CANAL_COMPASS_CHECK,
                               offset=(5, 5), similarity=0.78):
                logger.warning('Grand Canal compass panel closed before step gain')
                return None
            current = self._compass_step_count()
            if current is None or not before < current <= before + 20:
                last, stable = None, 0
                continue
            stable = stable + 1 if current == last else 1
            last = current
            if stable >= 2:
                logger.attr('GrandCanalCompassSteps', f'{before} -> {current}')
                return current
        logger.warning(f'Grand Canal compass steps did not change within {timeout}s')
        return None

    def _close_compass_panel(self):
        if self.appear(GRAND_CANAL_COMPASS_CHECK,
                       offset=(5, 5), similarity=0.78):
            self.device.press_back()
        deadline = time.time() + 5
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if (not self.appear(GRAND_CANAL_COMPASS_CHECK,
                                offset=(5, 5), similarity=0.78)
                    and self._is_canal_page()):
                return True
        logger.warning('Grand Canal compass panel did not return to map')
        return False

    def _prepare_route_action(self):
        resources = self._read_canal_resources()
        if resources is None:
            return False, '[司南或通航步数识别失败][等待复查]'
        steps, sinan = resources
        threshold = max(0, int(self._get('MinNavigationSteps', 30)))
        reserve = max(0, int(self._get('SinanReserve', 30)))
        refill_target = 70
        if threshold and steps < threshold and steps < refill_target and sinan > reserve:
            # One spin at a time. Re-read the map after each spin so the
            # actual cost and added steps govern the next decision. The
            # threshold starts refilling; 70 steps stops it.
            for _ in range(24):
                if steps >= refill_target or sinan <= reserve:
                    break
                self.device.click(GRAND_CANAL_SINAN_BUTTON)
                self.device.sleep(0.4)
                self.device.screenshot()
                if not self.appear(GRAND_CANAL_COMPASS_CHECK,
                                   offset=(5, 5), similarity=0.78):
                    logger.warning('Grand Canal compass panel not confirmed')
                    return False, '[司南面板识别失败][等待复查]'
                panel_steps = self._compass_step_count()
                if panel_steps != steps:
                    logger.warning(
                        f'Grand Canal step counts disagree: map={steps}, '
                        f'compass={panel_steps}'
                    )
                    self._close_compass_panel()
                    return False, '[司南面板步数识别失败][等待复查]'
                self.device.click(GRAND_CANAL_COMPASS_SPIN)
                gained_steps = self._wait_compass_step_gain(steps)
                if not self._close_compass_panel():
                    return False, '[司南面板未返回航道][等待复查]'
                if gained_steps is None:
                    return False, '[司南转动未完成][等待复查]'
                refreshed = self._read_canal_resources()
                if (refreshed is None or refreshed[1] >= sinan
                        or refreshed[0] < gained_steps):
                    logger.warning(f'Grand Canal spin not confirmed: {resources} -> {refreshed}')
                    return False, '[司南转动结果未确认][等待复查]'
                steps, sinan = refreshed
                # Repeated refills legitimately alternate these two buttons.
                # Retire their click history only after both counters prove
                # progress; unsuccessful spins must retain stuck protection.
                if hasattr(self.device, 'click_record_remove'):
                    self.device.click_record_remove(GRAND_CANAL_SINAN_BUTTON)
                    self.device.click_record_remove(GRAND_CANAL_COMPASS_SPIN)
        if steps <= 0:
            return False, '[通航步数不足][等待复查]'
        self._route_steps_before_action = steps
        return True, ''

    def _is_event_dialog(self, similarity=0.76):
        """Ordinary events and large orders use the same dispatch lifecycle."""
        return (self.appear(GRAND_CANAL_EVENT_CHECK, offset=(6, 6), similarity=similarity)
                or self.appear(GRAND_CANAL_BIG_ORDER_CHECK, offset=(6, 6), similarity=similarity))

    def _is_canal_page(self):
        return self.appear(
            GRAND_CANAL_PAGE_CHECK, offset=(6, 6), similarity=0.78
        )

    def _is_current_location_dialog(self):
        # 地点名称和说明会变化（例如“漕运码头”与“驶向该处”），
        # 但无事件提示的“知道了”按钮固定；不要拿可变正文做模板。
        return self.appear(
            GRAND_CANAL_CURRENT_LOCATION_OK,
            offset=(6, 6), similarity=0.80,
        )

    def _is_sailing_dialog(self):
        return self.appear(
            GRAND_CANAL_SAILING_DIALOG_CHECK,
            offset=(6, 6), similarity=0.78,
        )

    def _is_resource_shortage(self):
        return self.appear(
            GRAND_CANAL_RESOURCE_SHORTAGE_CHECK,
            offset=(5, 5), similarity=0.78,
        )

    def _dismiss_resource_shortage(self):
        """Close the shortage notice without touching its paid refill button."""
        if not self._is_resource_shortage():
            return False
        logger.info('Grand Canal destination lacks resources; close without refill')
        self.device.press_back()
        deadline = time.time() + 4
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if not self._is_resource_shortage():
                break
        else:
            logger.warning('Grand Canal resource shortage notice remained open')
            return False
        return self._wait_canal_page(timeout=5) is True

    def _dismiss_sailing_dialog(self):
        if not self._is_sailing_dialog():
            return False
        logger.info('Grand Canal is still sailing; acknowledge and retry later')
        self.device.click(GRAND_CANAL_SAILING_OK)
        self.device.sleep(0.4)
        self.device.screenshot()
        return not self._is_sailing_dialog()

    def _dismiss_current_location_dialog(self, timeout=4):
        """Close the dock's no-event notice before selecting a route node."""
        if not self._is_current_location_dialog():
            return False
        # Returning from a transport job can move the ship back to its home
        # dock. The old route fix must not veto a fresh map fix there.
        if getattr(self, '_last_ship_notice_title', '') in ('漕运码头', '漕運碼頭'):
            self._canal_position_may_change = True
            logger.info('Grand Canal home dock confirmed; reacquire position from live map')
        logger.info('Current Grand Canal location has no event; close notice')
        self.device.click(GRAND_CANAL_CURRENT_LOCATION_OK)
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if not self._is_current_location_dialog() and self._is_canal_page():
                return True
        logger.warning('Grand Canal current-location notice did not close')
        return False

    def _find_search_arrow_target(self):
        """在搜索箭头下方确认临水的码头地基图标。"""
        arrow = self.find_building_search_arrow(
            allow_retained=False, report_missing=False,
        )
        if arrow is None:
            logger.warning('Grand Canal dock search target was not recognized')
            return None
        arrow_x, arrow_tip, _ = arrow
        image = self.device.image
        match = find_foundation_icon(image, arrow, self.BUILDING)
        if match is not None:
            target_x, target_y, score, family = match
            logger.info(
                f'Grand Canal dock foundation icon: family={family}, '
                f'score={score:.3f}, target=({target_x}, {target_y})'
            )
            return Button(
                area=(), color=(),
                button=(target_x - 9, target_y - 8,
                        target_x + 9, target_y + 8),
                name='GRAND_CANAL_DOCK_DYNAMIC',
            )
        # The older dock-specific reference remains useful for a future icon
        # variant that the shared family library has not seen yet.
        height, width = image.shape[:2]
        x1 = max(75, arrow_x - 130)
        y1 = max(20, arrow_tip + 120)
        x2 = min(width - 20, arrow_x + 130)
        y2 = min(height - 65, arrow_tip + 330)
        reference = self._dock_foundation_icon
        if (x2 - x1 < reference.shape[1]
                or y2 - y1 < reference.shape[0]):
            return None
        patch = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        template = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        logger.attr('GrandCanalDockFoundationIconScore', f'{score:.3f}')
        if score < 0.82:
            logger.warning('Grand Canal dock foundation icon not confirmed')
            return None
        target_x = x1 + location[0] + reference.shape[1] // 2
        target_y = y1 + location[1] + reference.shape[0] // 2
        target = Button(
            area=(), color=(),
            button=(target_x - 9, target_y - 8,
                    target_x + 9, target_y + 8),
            name='GRAND_CANAL_DOCK_DYNAMIC',
        )
        logger.info(
            f'Grand Canal dock foundation icon: score={score:.3f}, '
            f'arrow=({arrow_x}, {arrow_tip}), target=({target_x}, {target_y})'
        )
        return target

    def _open_dock_panel(self, timeout=10):
        """打开漕运码头；点击失配后重新搜索码头再试，绝不复用旧坐标。"""
        target = None
        for attempt in range(1, 3):
            self.device.screenshot()
            if attempt == 1:
                target = self._find_search_arrow_target()
            else:
                # Re-search once after a missed click and require the same
                # visual dock icon at the new camera position.
                if not self.is_in_main():
                    self.device.press_back()
                    self.device.sleep(0.5)
                    self.device.screenshot()
                if not self.search_building(
                    self.BUILDING,
                    result_validator=lambda value: self.BUILDING in str(value),
                    exact=False,
                    handle_interference=False,
                ):
                    logger.warning(
                        f'Unable to reposition Grand Canal dock: {attempt}/2'
                    )
                    target = None
                else:
                    self.device.screenshot()
                    target = self._find_search_arrow_target()
            if target is None:
                break
            logger.info(f'Open Grand Canal dock attempt: {attempt}/2')
            self._click_searched_building(target)
            attempt_deadline = time.time() + min(float(timeout), 3.0)
            while time.time() < attempt_deadline:
                self.device.sleep(0.3)
                self.device.screenshot()
                if self.handle_network_timeout_popup():
                    return False
                if self.appear(
                    GRAND_CANAL_DOCK_PANEL_CHECK,
                    offset=(5, 5), similarity=0.80,
                ):
                    return True
                # If this icon opens a different modal, its identity is no
                # longer trustworthy in this frame. Stop without trying
                # nearby map coordinates.
                if time.time() - (attempt_deadline - min(float(timeout), 3.0)) >= 0.7 and not self.is_in_main():
                    logger.warning(
                        'Grand Canal dock icon opened another modal; '
                        'stop rather than click neighboring buildings'
                    )
                    self.save_abnormal_screenshot(
                        'grand_canal_dock_wrong_modal'
                    )
                    return False
        logger.warning('Grand Canal dock panel did not appear')
        try:
            self.device.image_save('jiangnan_grand_canal_dock_open_failed.png')
        except Exception as exc:
            logger.warning(f'Unable to save Grand Canal debug image: {exc}')
        return False

    def _enter_canal(self):
        if not self.switch_city(self.CITY):
            return False
        if not self.search_building(
            self.BUILDING,
            result_validator=lambda value: self.BUILDING in str(value),
            exact=False,
        ):
            return False
        if not self._open_dock_panel():
            return False
        self.device.click(GRAND_CANAL_ENTER_BUTTON)
        deadline = time.time() + 45
        reward_clicks = 0
        last_reward_click = 0
        while time.time() < deadline:
            self.device.sleep(0.5)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            # Entering the canal can immediately show a Sinan reward over
            # the map. It must be collected before map checks can succeed.
            if self.appear(GRAND_CANAL_SINAN_REWARD,
                           offset=(5, 5), similarity=0.80):
                if reward_clicks < 2 and time.time() - last_reward_click >= 2:
                    self.device.click(GRAND_CANAL_SINAN_REWARD)
                    reward_clicks += 1
                    last_reward_click = time.time()
                continue
            if self._is_canal_page():
                logger.info('Grand Canal map arrived')
                return True
        logger.warning('Grand Canal loading timeout')
        return False

    def _leave_canal(self):
        self.device.screenshot()
        if self._is_sailing_dialog():
            if not self._dismiss_sailing_dialog():
                return False
            self.device.screenshot()
        if self._is_current_location_dialog():
            if not self._dismiss_current_location_dialog():
                return False
            self.device.screenshot()
        elif self._is_event_dialog(similarity=0.76) or self.appear(
            GRAND_CANAL_MOVE_DIALOG_CHECK, offset=(6, 6), similarity=0.75,
        ):
            logger.info('Close Grand Canal route dialog before leaving')
            self.device.press_back()
            self.device.sleep(0.5)
            self.device.screenshot()
        if self.appear(
            GRAND_CANAL_EXIT_BUTTON, offset=(6, 6), similarity=0.75
        ):
            self.device.click(GRAND_CANAL_EXIT_BUTTON)
        else:
            logger.warning('Grand Canal exit button was not recognized')
            return False
        deadline = time.time() + 45
        exit_clicks = 1
        last_exit_click = time.time()
        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self.handle_startup_popup():
                logger.info('Popup handled after leaving Grand Canal')
                continue
            if self.is_unobstructed_main():
                return True
            if self._is_current_location_dialog():
                if not self._dismiss_current_location_dialog():
                    return False
                continue
            if (exit_clicks < 4
                    and time.time() - last_exit_click >= 2
                    and self.appear(
                        GRAND_CANAL_EXIT_BUTTON,
                        offset=(6, 6), similarity=0.75,
                    )):
                exit_clicks += 1
                last_exit_click = time.time()
                logger.info(f'Retry Grand Canal exit: {exit_clicks}/4')
                self.device.click(GRAND_CANAL_EXIT_BUTTON)
        logger.warning('Unable to leave Grand Canal normally')
        return False

    def _center_ship_on_map(self):
        """Center the live ship without reopening its pending event."""
        self.device.screenshot()
        if not self.appear(
            GRAND_CANAL_LOCATE_BUTTON, offset=(6, 6), similarity=0.75
        ):
            logger.warning('Grand Canal locate button was not recognized')
            return False
        self.device.click(GRAND_CANAL_LOCATE_BUTTON)
        self.device.sleep(1.2)
        self.device.screenshot()
        # The game's locate action centers our ship. Its nameplate is not
        # needed to confirm the position and can be obscured by map terrain.
        return self._is_canal_page()

    def _locate_and_click_ship(self):
        self._last_ship_dialog = None
        self._last_ship_notice_title = ''
        if not self._center_ship_on_map():
            return False
        centered_map = self.device.image.copy()
        # “定位”保证漕船落在屏幕正中；这里只使用这个游戏提供的定位
        # 约束，不依赖船在大地图中的绝对坐标。
        cx, cy = self.SHIP_CENTER
        self.device.click(Button(
            area=(), color=(), button=(cx - 38, cy - 42, cx + 38, cy + 42),
            name='GRAND_CANAL_CENTERED_SHIP',
        ))
        deadline = time.time() + 5
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if self._is_sailing_dialog():
                self._last_ship_dialog = 'sailing'
                return True
            if hasattr(self, '_read_ship_repair_duration'):
                self._read_ship_repair_duration()
                if getattr(self, '_last_ship_dialog', None) == 'repair':
                    return True
            if self._is_current_location_dialog():
                self._last_ship_dialog = 'current'
                return True
            if self.appear(
                GRAND_CANAL_REWARD_CHECK, offset=(8, 8), similarity=0.76,
            ):
                self._last_ship_dialog = 'reward'
                return True
            if self._is_event_dialog(similarity=0.76):
                self._last_ship_dialog = 'event'
                return True
        # Some ordinary river nodes have no ship popup. A confirmed,
        # unchanged map with visible route nodes can use normal destination
        # selection; an unrecognized modal must still stop this operation.
        if (self._is_canal_page()
                and self._map_view_unchanged(centered_map, self.device.image)
                and self._detect_blue_destinations()):
            self._last_ship_dialog = 'map'
            logger.info('Grand Canal ship has no popup on the unchanged map; select a route destination')
            return True
        logger.warning(
            'Grand Canal ship click did not open a recognized location, '
            'reward, or event dialog; do not select a destination through it'
        )
        self.save_abnormal_screenshot('grand_canal_ship_dialog_missing')
        return False

    @staticmethod
    def _find_red_action_button(image):
        """动态定位普通出发、事件出发、运输货物或收下按钮。"""
        x1, y1, x2, y2 = (330, 420, 950, 705)
        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 75, 70), (13, 255, 255)),
            cv2.inRange(hsv, (168, 75, 70), (179, 255, 255)),
        )
        red = cv2.morphologyEx(
            red, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        )
        contours, _ = cv2.findContours(
            red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if 150 <= width <= 380 and 35 <= height <= 105:
                center_x = x1 + left + width // 2
                center_y = y1 + top + height // 2
                distance = abs(center_x - 640)
                candidates.append((distance, left, top, width, height))
        if not candidates:
            return None
        _, left, top, width, height = min(candidates)
        return Button(
            area=(), color=(),
            button=(
                x1 + left + 10, y1 + top + 6,
                x1 + left + width - 10, y1 + top + height - 6,
            ),
            name='GRAND_CANAL_ACTION_DYNAMIC',
        )

    def _read_action_text(self, button):
        reader = Ocr(
            Button(
                area=button.button, color=(), button=button.button,
                name='GRAND_CANAL_ACTION_TEXT',
            ),
            lang='cnocr',
            name='GRAND_CANAL_ACTION_TEXT_OCR',
        )
        return str(reader.ocr(self.device.image)).replace(' ', '')

    def _departure_click_button(self, action):
        """Click the visible departure label, clear of the left cost/icon."""
        area = getattr(action, 'button', None)
        if not isinstance(area, tuple) or len(area) != 4:
            return action
        x1, y1, x2, y2 = area
        # This helper is used only after OCR has confirmed "出发". The label
        # is on the right half of the dynamically detected red action button.
        # Use its actual ivory glyph bounds, not a fixed screen coordinate.
        left = (x1+x2)//2
        image = self.device.image[y1:y2, left:x2]
        if image.size == 0:
            return action
        mask = cv2.inRange(image, (175, 165, 140), (255, 255, 255))
        pixels = cv2.findNonZero(mask)
        if pixels is None or len(pixels) < 20:
            return action
        bx, by, width, height = cv2.boundingRect(pixels)
        if width < 10 or height < 8:
            return action
        cx, cy = left+bx+width//2, y1+by+height//2
        box = (max(x1, cx-3), max(y1, cy-3), min(x2, cx+3), min(y2, cy+3))
        logger.attr('GrandCanalDepartureTextTarget', box)
        return Button(area=area, color=(), button=box, name='GRAND_CANAL_DEPARTURE_TEXT')

    def _parse_working_duration(self, raw):
        # An orphan unit such as "飞分22秒" must not silently become 22 seconds.
        unmatched = re.sub(r'\d+\D{0,2}(?:小时|分钟|分|秒)', '', str(raw))
        if re.search(r'小时|分钟|分|秒', unmatched):
            return None
        return self.parse_duration(raw)

    def _read_ship_repair_duration(self):
        """The grounded-ship modal has its own title and countdown layout."""
        self._ship_repair_notice = False
        title = str(self.ship_repair_title_ocr.ocr(self.device.image)).replace(' ', '')
        self._last_ship_notice_title = title
        if '触礁' not in title:
            return None
        self._last_ship_dialog = 'repair'
        self._ship_repair_notice = True
        self._ship_working_notice = True
        raw = str(self.ship_repair_time_ocr.ocr(self.device.image))
        duration = self._parse_working_duration(raw)
        logger.attr('GrandCanalShipRepairTime', f'{raw!r} -> {duration}')
        return duration

    def _read_ship_working_duration(self):
        if hasattr(self, '_read_ship_repair_duration'):
            duration = self._read_ship_repair_duration()
            if getattr(self, '_ship_repair_notice', False):
                return duration
        if not self._is_event_dialog(similarity=0.76):
            return None
        raw = str(self.ship_working_time_ocr.ocr(self.device.image))
        if not re.search(r'还剩|剩余|正在', raw):
            return None
        self._ship_working_notice = True
        duration = self._parse_working_duration(raw)
        logger.attr('GrandCanalShipWorkingTime', f'{raw!r} -> {duration}')
        return duration

    def _read_working_duration(self, open_ship=False):
        self._ship_working_notice = False
        self.device.screenshot()
        duration = self._read_ship_working_duration()
        if duration is not None or self._ship_working_notice:
            return duration
        if not open_ship and not self.appear(
            GRAND_CANAL_WORKING_CHECK, offset=(5, 5), similarity=0.72
        ):
            return None
        # Open our own ship's large countdown rather than trusting the small
        # task card, whose leading minute/hour digits may be clipped by OCR.
        if self._locate_and_click_ship():
            if getattr(self, '_last_ship_dialog', None) == 'map':
                return None
            duration = self._read_ship_working_duration()
            if duration is not None or self._ship_working_notice:
                return duration
            self.device.press_back()
            self.device.sleep(0.3)
            self.device.screenshot()
        else:
            return None
        raw = str(self.working_time_ocr.ocr(self.device.image))
        duration = self._parse_working_duration(raw)
        logger.attr('GrandCanalWorkingTime', f'{raw!r} -> {duration}')
        return duration

    def _read_task_card_durations(self):
        """读取右侧任务卡；只用于比较事件前后新增的卡片。"""
        durations = []
        for slot, card, reader in (
            (1, GRAND_CANAL_TASK_CARD_CHECK, self.working_time_ocr),
            (2, GRAND_CANAL_SECOND_TASK_CARD_CHECK,
             self.second_working_time_ocr),
        ):
            if not self.appear(card, offset=(5, 5), similarity=0.80):
                continue
            raw = str(reader.ocr(self.device.image))
            duration = self._parse_working_duration(raw)
            logger.attr(
                'GrandCanalWorkingTime', f'{card.name}: {raw!r} -> {duration}'
            )
            if duration:
                durations.append((slot, duration))
        return durations

    def _detect_blue_destinations(self, full=False):
        """提取航道圆点；角度从正北开始顺时针计算。"""
        image = self.device.image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # A narrow blue route line can connect a circle to the screen edge.
        # The old distance-transform component then became too large and the
        # nearest node vanished from the candidate list. Opening removes the
        # thin line while keeping the filled destination circles.
        blue = cv2.inRange(hsv, (90, 45, 45), (130, 255, 150))
        circles = cv2.morphologyEx(
            blue, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(circles)
        cx0, cy0 = self.SHIP_CENTER
        points = []
        for index in range(1, count):
            left, top, width, height, area = map(int, stats[index])
            cx, cy = map(float, centroids[index])
            if not (
                80 <= area <= 500
                and 10 <= width <= 40
                and 10 <= height <= 30
            ):
                continue
            # 航点一定落在水面上。蓝瓦屋顶和建筑阴影也可能形成相同
            # 厚度的蓝色核心，仅看颜色/圆心会把它们当成航点；检查
            # 椭圆外环的水面占比，可保留与虚线航道相连的真实圆点。
            center_x, center_y = int(round(cx)), int(round(cy))
            image_height, image_width = image.shape[:2]
            ring_x1 = max(0, center_x - 32)
            ring_x2 = min(image_width, center_x + 33)
            ring_y1 = max(0, center_y - 24)
            ring_y2 = min(image_height, center_y + 25)
            ring_hsv = hsv[ring_y1:ring_y2, ring_x1:ring_x2]
            yy, xx = np.ogrid[:ring_hsv.shape[0], :ring_hsv.shape[1]]
            local_x = center_x - ring_x1
            local_y = center_y - ring_y1
            outer = (
                ((xx - local_x) ** 2) / (30 ** 2)
                + ((yy - local_y) ** 2) / (22 ** 2)
            ) <= 1
            inner = (
                ((xx - local_x) ** 2) / (16 ** 2)
                + ((yy - local_y) ** 2) / (10 ** 2)
            ) < 1
            ring = outer & ~inner
            ring_pixels = ring_hsv[ring]
            if not len(ring_pixels):
                continue
            water = (
                (ring_pixels[:, 0] >= 78)
                & (ring_pixels[:, 0] <= 108)
                & (ring_pixels[:, 1] >= 35)
                & (ring_pixels[:, 1] <= 165)
                & (ring_pixels[:, 2] >= 80)
                & (ring_pixels[:, 2] <= 220)
            )
            water_ratio = float(np.count_nonzero(water)) / len(ring_pixels)
            # A real node beside a dock loses part of its water ring to the
            # shore. Accept that case only near the centered ship; distant
            # blue roof/decoration candidates still need the full margin.
            distance_from_ship = math.hypot(cx - cx0, cy - cy0)
            minimum_water = 0.62 if distance_from_ship <= 320 else 0.72
            if water_ratio < minimum_water:
                logger.info(
                    f'Ignore blue non-water feature at '
                    f'({center_x}, {center_y}): water={water_ratio:.2f}'
                )
                continue
            angle = (
                math.degrees(math.atan2(cx - cx0, -(cy - cy0))) + 360
            ) % 360
            points.append((angle, distance_from_ship, cx, cy))

        # 同一个圆点偶尔会被抗锯齿分成两个核心，按距离合并。
        deduplicated = []
        for item in sorted(points, key=lambda value: (value[1], value[0])):
            if any(
                math.hypot(item[2] - old[2], item[3] - old[3]) < 24
                for old in deduplicated
            ):
                continue
            deduplicated.append(item)
        if full:
            return deduplicated
        # 虚线可能被建筑遮挡，不能要求像素连通。船居中后，同一
        # 屏幕方向的圆点视作同一航道，按距离只保留近点，再对各
        # 航道顺时针排序，避免直接越过相邻站点。
        nearby = []
        for item in deduplicated:
            if any(
                abs((item[0] - old[0] + 180) % 360 - 180) < 32
                and item[1] >= old[1] * 1.35
                for old in nearby
            ):
                continue
            nearby.append(item)
        logger.attr(
            'GrandCanalBlueDestinations',
            [
                f'({round(x)},{round(y)})@{angle:.1f}°'
                for angle, _, x, y in nearby
            ],
        )
        return nearby

    def _click_first_destination(self, excluded=()):
        # A previous target must never be credited to a later screen click.
        self._last_destination_node = None
        self._map_position_unconfirmed = False
        self.device.screenshot()
        # 船上的当前地点提示有时比点击动画晚一帧出现。选蓝点前
        # 再复核一次，避免隔着模态弹窗点到地图并空等目的地弹窗。
        if self._is_current_location_dialog():
            if not self._dismiss_current_location_dialog():
                return False
            self.device.screenshot()
        points = [
            point for point in self._detect_blue_destinations()
            if not any(math.hypot(point[2] - x, point[3] - y) < 30
                       for x, y in excluded)
        ]
        if not points:
            logger.warning('No clickable Grand Canal blue destination found')
            return False
        if getattr(self, '_map_routing_active', False):
            if self._click_map_destination(points, excluded):
                return True
            if str(self._get('CustomRoute', '') or '').strip():
                logger.warning('Grand Canal custom route unavailable; do not choose another route')
                return False
            logger.info('Grand Canal map is inconclusive; use visible blue destinations')
        # 航点相对居中的船分布在固定方向。记录上次所走的八方向，
        # 排除其反方向；不同距离的同向航点都属于回头路。只有该
        # 方向之外无点可走时才允许折返，其他点仍按屏幕顺时针排序。
        heading = getattr(self, '_route_heading', None)
        if heading is not None:
            reverse_sector = (int((heading + 22.5) // 45) + 4) % 8
            forward = [
                point for point in points
                if int((point[0] + 22.5) // 45) % 8 != reverse_sector
            ]
            if not forward:
                logger.info(
                    f'Grand Canal has no other destination; allow '
                    f'backtracking in direction {reverse_sector}'
                )
            elif len(forward) != len(points):
                logger.info(
                    f'Skip {len(points) - len(forward)} Grand Canal '
                    f'backtracking destination(s) in direction '
                    f'{reverse_sector}'
                )
                points = forward
        points.sort(key=lambda point: (point[0], point[1]))
        angle, distance, cx, cy = points[0]
        self._last_destination_center = (cx, cy)
        self._last_destination_angle = angle
        logger.info(
            f'Click clockwise Grand Canal destination: '
            f'({cx:.0f}, {cy:.0f}), distance={distance:.0f}, '
            f'angle={angle:.1f}°'
        )
        self.device.click(Button(
            area=(), color=(),
            button=(int(cx) - 10, int(cy) - 10, int(cx) + 10, int(cy) + 10),
            name='GRAND_CANAL_BLUE_DESTINATION',
        ))
        return True

    @staticmethod
    def _measure_route_pan(before, after):
        """Measure map translation from agreeing terrain patches, not touch travel."""
        if before is None or after is None or before.shape != after.shape:
            return None
        old = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        new = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        search = new[85:565, 100:1100]
        matches = []
        for y in (105, 190, 275, 360, 445):
            for x in (150, 330, 510, 750, 930):
                # Exclude our ship and its animated nameplate.
                if x == 510 and y in (190, 275, 360):
                    continue
                patch = old[y:y+60, x:x+80]
                if patch.std() < 10 or cv2.countNonZero(cv2.Canny(patch, 60, 130)) < 80:
                    continue
                scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
                _, score, _, (left, top) = cv2.minMaxLoc(scores)
                if score >= .88:
                    matches.append((100+left-x, 85+top-y, x, y, score))
        if len(matches) < 3:
            inliers = []
        else:
            inliers = max((
                [other for other in matches if math.hypot(
                    other[0]-candidate[0], other[1]-candidate[1]) <= 6]
                for candidate in matches
            ), key=len)
        # Repeated roof/water textures must not establish the translation alone.
        if (len(inliers) < 3 or len(inliers) <= len(matches)/2
                or len({item[2] for item in inliers}) < 2
                or len({item[3] for item in inliers}) < 2):
            # A large vertical pan can clip every full 60px terrain patch.
            # Local features still register the remaining coastline fragments.
            mask = np.zeros(old.shape, np.uint8)
            mask[85:565, 100:1100] = 255
            sift = cv2.SIFT_create(nfeatures=3000)
            old_keys, old_desc = sift.detectAndCompute(old, mask)
            new_keys, new_desc = sift.detectAndCompute(new, mask)
            if old_desc is None or new_desc is None or len(new_desc) < 2:
                return None
            good = [pair[0] for pair in cv2.BFMatcher().knnMatch(old_desc, new_desc, k=2)
                    if len(pair) == 2 and pair[0].distance < .68*pair[1].distance]
            if len(good) < 8:
                return None
            source = np.float32([old_keys[m.queryIdx].pt for m in good])
            destination = np.float32([new_keys[m.trainIdx].pt for m in good])
            matrix, selected = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=3)
            if matrix is None or selected is None:
                return None
            valid = selected.ravel().astype(bool)
            if (np.count_nonzero(valid) < 8 or valid.mean() < .30
                    or not .99 <= matrix[0, 0] <= 1.01 or abs(matrix[0, 1]) > .01
                    or np.ptp(source[valid, 0]) < 60 or np.ptp(source[valid, 1]) < 30):
                return None
            delta = np.median(destination[valid]-source[valid], axis=0)
            logger.attr('GrandCanalPanFeatures', f'shift={delta.tolist()}, inliers={valid.sum()}/{len(good)}')
            return tuple(float(value) for value in delta)
        shift = tuple(float(np.median([item[axis] for item in inliers])) for axis in (0, 1))
        logger.attr('GrandCanalPanMeasured', f'shift={shift}, terrain={len(inliers)}/{len(matches)}')
        return shift

    def _reveal_blocked_route_destination(self, current, route, index, occupied):
        """Reveal the next offscreen route point, passing confirmed occupied ships."""
        cycle = route[:-1]
        target = None
        skipped = []
        for offset in range(1, len(cycle)):
            node = cycle[(index + offset) % len(cycle)]
            if node in occupied:
                skipped.append(node)
                continue
            target = node
            break
        if target is None:
            return False
        map_points, _, work_points = load_topology()
        ox, oy = map_points[current]
        tx, ty = map_points[target]
        predicted = (640 + tx - ox, 360 - ty + oy)
        # This recovery is only for a point outside the current viewport.
        # A missing in-view point still needs its original retry path.
        if 100 <= predicted[0] <= 1100 and 100 <= predicted[1] <= 550:
            return False
        # Move only far enough to reveal the target, avoiding an unnecessary
        # diagonal pan that throws most existing terrain/blue anchors offscreen.
        dx = int(max(-220, min(220, max(250, min(1000, predicted[0])) - predicted[0])))
        dy = int(max(-220, min(220, max(180, min(430, predicted[1])) - predicted[1])))
        if abs(dx) + abs(dy) < 80:
            return False
        before = self.device.image.copy()
        before_points = [(p[2], p[3]) for p in self._detect_blue_destinations(full=True)]
        clicked = False
        try:
            logger.info(f'Reveal Grand Canal offscreen route target: {target}; occupied={skipped}')
            # Keep the whole gesture inside the map and start away from our ship.
            # MaaTouch swipe ignores duration and can fling the map; a linear drag
            # followed by a hold brakes the camera before releasing the finger.
            start = (int(640-dx/2), int(340-dy/2))
            end = (start[0]+dx, start[1]+dy)
            logger.attr('GrandCanalPanRequested',
                        f'target={target}, predicted={predicted}, shift=({dx}, {dy}), '
                        f'gesture={start}->{end}')
            self.save_abnormal_screenshot('grand_canal_pan_before')
            self.jiangnan_drag(start, end, swipe_duration=1.0, hold_duration=.4,
                               swipe_min_distance=0, name='GRAND_CANAL_REVEAL_ROUTE')
            self.device.sleep(.4)
            self.device.screenshot()
            self.save_abnormal_screenshot('grand_canal_pan_after')
            if not self._is_canal_page() or self.is_network_timeout_popup():
                logger.warning('Grand Canal pan interrupted by another page or network popup')
                return False
            blue_points = [(p[2], p[3]) for p in self._detect_blue_destinations(full=True)]
            logger.attr('GrandCanalPanBluePoints', blue_points)
            blue_shift = blue_point_translation(before_points, blue_points)
            shift = self._measure_route_pan(before, self.device.image)
            if blue_shift is not None:
                if shift is not None and math.hypot(shift[0]-blue_shift[0], shift[1]-blue_shift[1]) > 6:
                    logger.warning(f'Grand Canal pan blue/terrain disagreement: {blue_shift}, {shift}')
                    return False
                shift = blue_shift
            if shift is None:
                logger.warning('Grand Canal pan has no confirmed terrain or blue-point translation')
                return False
            # A correct displacement on one frame can still be camera inertia.
            # Click only once a second fresh frame confirms the same translation.
            self.device.sleep(.2)
            self.device.screenshot()
            if not self._is_canal_page() or self.is_network_timeout_popup():
                return False
            blue_points = [(p[2], p[3]) for p in self._detect_blue_destinations(full=True)]
            settled_shift = blue_point_translation(before_points, blue_points)
            if settled_shift is None:
                settled_shift = self._measure_route_pan(before, self.device.image)
            if settled_shift is None or math.hypot(
                    settled_shift[0]-shift[0], settled_shift[1]-shift[1]) > 3:
                logger.warning(f'Grand Canal pan is still moving: first={shift}, second={settled_shift}')
                self.save_abnormal_screenshot('grand_canal_pan_unsettled')
                return False
            shift = settled_shift
            actual_dx, actual_dy = shift
            # The touch direction/distance is only a request. Actual blue
            # circles and the measured map position decide whether to click.
            observed = [(x-actual_dx, y-actual_dy) for x, y in blue_points]
            mapped = map_visible_destinations(current, observed)
            occupied = occupied_route_nodes(current, self.device.image, screen_offset=shift)
            logger.attr('GrandCanalPanOccupied', sorted(occupied))
            node, target_index, _ = custom_route_destination(
                route, index, {node for _, node in mapped}, occupied,
                dock_nodes=work_points,
            )
            if node is None:
                logger.warning(f'Grand Canal pan target not revealed: expected={target}, '
                               f'mapped={mapped}, actual={shift}')
                return False
            virtual_x, virtual_y = next(point for point, mapped_node in mapped
                                        if mapped_node == node)
            cx, cy = virtual_x + actual_dx, virtual_y + actual_dy
            if not (110 <= cx <= 1090 and 110 <= cy <= 550):
                return False
            self._last_destination_center = (cx, cy)
            self._last_destination_angle = (
                math.degrees(math.atan2(virtual_x - 640, -(virtual_y - 360))) + 360
            ) % 360
            self._last_destination_node = node
            self._last_custom_route_move = (route, index, target_index)
            self.device.click(Button(
                area=(), color=(), button=(int(cx)-10, int(cy)-10, int(cx)+10, int(cy)+10),
                name='GRAND_CANAL_REVEALED_DESTINATION',
            ))
            clicked = True
            return True
        finally:
            if not clicked:
                # Later routing assumes a centered ship, even after failure.
                self._center_ship_on_map()

    def _click_map_destination(self, points, excluded):
        """Locate the ship from visible blue points, then follow the map graph."""
        self._last_custom_route_move = None
        self._last_approach_move = None
        self._map_position_unconfirmed = False
        observed = [
            (point[2], point[3])
            for point in self._detect_blue_destinations(full=True)
        ]
        previous_fix = str(self._get('LastPoint', '') or '').strip().upper()
        point_source = str(self._get('LastPointSource', 'unknown') or 'unknown')
        hint = (
            previous_fix if len(observed) >= 2
            or point_source in ('detected', 'computed') else None
        )
        match = locate_ship(
            observed, previous=hint,
            trusted_previous=(point_source in ('detected', 'computed')
                              and not getattr(self, '_canal_position_may_change', False)),
        )
        if match is None:
            self._map_position_unconfirmed = True
            logger.warning(
                f'Grand Canal map position is ambiguous; hint={hint}, '
                f'source={point_source}, blue_points={observed}; wait for a clearer view'
            )
            if hasattr(self, 'save_abnormal_screenshot'):
                self.save_abnormal_screenshot('grand_canal_position_unconfirmed')
            return False
        current, mapped = match
        self._canal_position_may_change = False
        logger.attr('GrandCanalFullScreenPositionMatch',
                    f'center={self.SHIP_CENTER}->{current}, blue_points={observed}, matches={mapped}')
        map_points, neighbors, work_points = load_topology()
        now = datetime.now().replace(microsecond=0)
        try:
            saved = json.loads(self._get('RecentDocks', '[]') or '[]')
            recent = [
                (node, datetime.fromisoformat(when))
                for node, when in saved
                if node in work_points
            ]
        except (TypeError, ValueError):
            recent = []
        recent = [(node, when) for node, when in recent
                  if now - timedelta(minutes=30) < when <= now]
        prior = previous_fix if previous_fix in map_points else None
        if current != prior or point_source != 'detected':
            if current == prior and point_source == 'computed':
                previous = str(self._get('PreviousPoint', '') or '')
            else:
                previous = (
                    prior if point_source == 'detected'
                    and prior in neighbors[current] else ''
                )
            self._set('PreviousPoint', previous)
            self._set('LastPoint', current)
            self._set('LastPointSource', 'detected')
            if current in work_points and (current != prior or point_source != 'detected'):
                recent = [(node, when) for node, when in recent if node != current]
                recent.append((current, now))
        self._set('RecentDocks', json.dumps(
            [(node, when.isoformat()) for node, when in recent],
            ensure_ascii=False,
        ))
        self.config.update()
        logger.attr('GrandCanalMapPosition', current)
        try:
            custom = parse_custom_route(self._get('CustomRoute', ''))
        except ValueError as exc:
            logger.warning(f'Invalid Grand Canal custom route: {exc}')
            return False
        route_index = None
        if custom:
            route_index, progress = custom_route_position(
                custom, current, str(self._get('PreviousPoint', '') or ''),
                self._get('RouteProgress', ''),
            )
            if route_index is not None:
                self._set('RouteProgress', progress)
                self.config.update()
                logger.attr('GrandCanalRouteIndex', route_index)
        available = [
            node for (x, y), node in mapped
            if any(math.hypot(x - point[2], y - point[3]) < 24
                   for point in points)
            if not any(math.hypot(x - ex, y - ey) < 30 for ex, ey in excluded)
        ]
        target_index = None
        if custom:
            # Near-point angular pruning must not hide a farther route node.
            mapped = map_visible_destinations(current, observed)
            available = [node for (x, y), node in mapped
                         if not any(math.hypot(x-ex, y-ey) < 30 for ex, ey in excluded)]
            occupied = occupied_route_nodes(current, getattr(self.device, 'image', None))
            # Topology includes virtual branch intersections (e.g. P185)
            # with no clickable circle. Confirm their geometry from at least
            # two visible adjacent endpoints, then sail through to the next
            # route point. A dock is never treated as a virtual intersection.
            visible_nodes = set(available)
            detected_nodes = {node for _, node in mapped}
            transit = {node for node in neighbors[current]
                       if node not in work_points and len(neighbors[node]) >= 3
                       and node not in detected_nodes and node not in occupied
                       and len(neighbors[node] & (visible_nodes | {current})) >= 2}
            if route_index is not None:
                next_node, target_index, skipped = custom_route_destination(
                    custom, route_index, visible_nodes, occupied, transit, work_points,
                )
            else:
                approach = route_approach_path(current, custom, set(available) - occupied)
                next_node, approach_index, skipped = custom_route_destination(
                    approach + (current,), 0, visible_nodes, occupied, transit, work_points,
                ) if len(approach) > 1 else (None, None, [])
                if next_node:
                    self._last_approach_move = approach[:approach_index+1]
                    logger.info(
                        f'Grand Canal approach configured route: {current} -> {next_node}; '
                        f'nearest route point={approach[-1]}, distance={len(approach)-1}'
                    )
            if skipped:
                logger.info(f'Grand Canal pass intermediate nodes: {skipped}; '
                            f'occupied={sorted(occupied)}, junctions={sorted(transit)}; target={next_node}')
            points = [(0, 0, x, y) for (x, y), _ in mapped]
        else:
            next_node = choose_next(
                current, available, recent_docks=[node for node, _ in recent],
                previous=str(self._get('PreviousPoint', '') or ''),
                custom_route=custom,
                route_index=route_index,
            )
        if next_node is None:
            if (custom and route_index is not None and not excluded
                    and hasattr(self, '_reveal_blocked_route_destination')
                    and self._reveal_blocked_route_destination(
                        current, custom, route_index, occupied)):
                return True
            logger.warning(f'No mapped Grand Canal destination from {current}')
            if custom:
                expected = (custom[(route_index + 1) % (len(custom) - 1)]
                            if route_index is not None else None)
                logger.warning(
                    f'Grand Canal route destination diagnostic: expected={expected}, '
                    f'index={route_index}, visible={sorted(set(available))}, '
                    f'occupied={sorted(occupied)}, excluded={excluded}'
                )
                if hasattr(self, 'save_abnormal_screenshot'):
                    self.save_abnormal_screenshot('grand_canal_route_destination_missing')
            return False
        selected = next(((x, y) for (x, y), node in mapped
                         if node == next_node and any(
                             math.hypot(x - point[2], y - point[3]) < 24
                             for point in points) and not any(
                             math.hypot(x - ex, y - ey) < 30
                             for ex, ey in excluded)), None)
        if selected is None:
            return False
        cx, cy = selected
        angle = (math.degrees(math.atan2(cx - self.SHIP_CENTER[0],
                                         -(cy - self.SHIP_CENTER[1]))) + 360) % 360
        self._last_destination_center = selected
        self._last_destination_angle = angle
        self._last_destination_node = next_node
        if target_index is not None:
            self._last_custom_route_move = (custom, route_index, target_index)
        logger.info(f'Grand Canal map route: {current} -> {next_node}')
        self.device.click(Button(
            area=(), color=(),
            button=(int(cx) - 10, int(cy) - 10, int(cx) + 10, int(cy) + 10),
            name='GRAND_CANAL_BLUE_DESTINATION',
        ))
        return True

    def _reward_readings(self):
        name_buttons = []
        amount_buttons = []
        for slot, center_x in enumerate(self.REWARD_CARD_CENTERS, start=1):
            name_area = (center_x - 58, 120, center_x + 58, 174)
            amount_area = (center_x - 55, 275, center_x + 55, 320)
            name_buttons.append(Button(
                area=name_area, color=(), button=name_area,
                name=f'GRAND_CANAL_REWARD_NAME_{slot}',
            ))
            amount_buttons.append(Button(
                area=amount_area, color=(), button=amount_area,
                name=f'GRAND_CANAL_REWARD_AMOUNT_{slot}',
            ))

        image = self.device.image
        values = Ocr(
            name_buttons + amount_buttons,
            lang='cnocr', name='GRAND_CANAL_REWARDS_BATCH',
        ).ocr(image)
        if not isinstance(values, list):
            values = [values]
        values.extend([''] * (len(name_buttons) + len(amount_buttons) - len(values)))
        names = values[:len(name_buttons)]
        amounts = values[len(name_buttons):]

        readings = {}
        for raw_name, amount_raw in zip(names, amounts):
            name = str(raw_name).strip()
            name = JiangnanInventoryState.normalize_resource(name)
            try:
                digits = ''.join(
                    character for character in str(amount_raw)
                    if character.isdigit()
                )
                amount = int(digits)
            except (TypeError, ValueError):
                amount = 0
            if name in ALL_RESOURCES and amount > 0:
                readings[name] = readings.get(name, 0) + amount
        logger.attr('GrandCanalWarehouseRewards', readings)
        return readings

    def _apply_rewards(self, readings, state):
        if not readings:
            return
        inventories = state.load_inventories((self.CITY,))
        inventory = inventories.setdefault(self.CITY, {})
        for resource, amount in readings.items():
            current = int(inventory.get(resource, -1))
            if current < 0:
                logger.warning(
                    f'Grand Canal reward cannot update unknown inventory: '
                    f'{self.CITY}, {resource} +{amount}'
                )
                inventory[resource] = -1
            else:
                inventory[resource] = current + amount
                logger.info(
                    f'Grand Canal inventory reward: {self.CITY}, '
                    f'{resource} +{amount}, remain={inventory[resource]}'
                )
        state.write_inventories(inventories)
        state.request_allocation((self.CITY,))

    def _collect_reward(self, state):
        self.device.screenshot()
        if not self.appear(
            GRAND_CANAL_REWARD_CHECK, offset=(8, 8), similarity=0.76
        ):
            return False
        readings = self._reward_readings()
        action = self._find_red_action_button(self.device.image)
        if action is None:
            logger.warning('Grand Canal reward take button was not recognized')
            return False
        self.device.click(action)
        deadline = time.time() + 12
        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if self._is_canal_page() and not self.appear(
                GRAND_CANAL_REWARD_CHECK, offset=(8, 8), similarity=0.76
            ):
                self._apply_rewards(readings, state)
                logger.info('Grand Canal reward collected')
                return True
        logger.warning('Grand Canal reward did not close after clicking take')
        return False

    def _wait_route_dialog(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return None
            if self._is_sailing_dialog():
                return 'sailing'
            if self._is_current_location_dialog():
                return 'current'
            if self._is_event_dialog(similarity=0.76):
                return 'event'
            if self.appear(
                GRAND_CANAL_MOVE_DIALOG_CHECK, offset=(6, 6), similarity=0.75
            ):
                return 'move'
            # The red button is visible while the dialog slides in from the
            # bottom. Its coordinates are still moving, so it cannot prove
            # that the departure action is ready to click.
        logger.warning(
            f'Grand Canal destination dialog timed out: '
            f'target={getattr(self, "_last_destination_node", None)}, '
            f'click={getattr(self, "_last_destination_center", None)}'
        )
        if hasattr(self, 'save_abnormal_screenshot'):
            self.save_abnormal_screenshot('grand_canal_destination_dialog_timeout')
        return None

    def _wait_canal_page(self, timeout=12):
        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if hasattr(self, '_is_resource_shortage') and self._is_resource_shortage():
                return 'shortage'
            if self._is_sailing_dialog():
                return 'sailing'
            # 弹窗覆盖在大运河地图上时，左侧“定位”仍可能透过蒙层
            # 命中模板。必须同时确认各弹窗已经消失。不要在地图上
            # 全局寻找红色矩形：岸边装饰偶尔会形成同样尺寸的色块，
            # 导致已返回地图却一直等待到超时。
            if (
                self._is_canal_page()
                and not self._is_current_location_dialog()
                and not (
                    hasattr(self, '_is_sinan_task_page')
                    and self._is_sinan_task_page()
                )
                and not (
                    hasattr(self, '_is_compass_panel')
                    and self._is_compass_panel()
                )
                and not self.appear(
                    GRAND_CANAL_MOVE_DIALOG_CHECK,
                    offset=(6, 6), similarity=0.72,
                )
                and not self._is_event_dialog(similarity=0.72)
                and not self.appear(
                    GRAND_CANAL_ASSIGNEE_CHECK, offset=(5, 5), similarity=0.74
                )
                and not self.appear(
                    GRAND_CANAL_REWARD_CHECK, offset=(8, 8), similarity=0.72
                )
            ):
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= 1.0:
                    return True
            else:
                stable_since = None
        return False

    def _wait_navigation_step_spent(self, timeout=7):
        """Confirm a voyage on the map; closing a dialog alone is not departure."""
        before = getattr(self, '_route_steps_before_action', None)
        if before is None:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            if self._is_canal_page():
                resources = self._read_canal_resources()
                if resources is not None and resources[0] < before:
                    logger.info(
                        f'Grand Canal departure confirmed: steps {before} -> '
                        f'{resources[0]}'
                    )
                    return True
            self.device.sleep(0.4)
        # Some voyages spend their navigation step only after the sailing
        # animation. Ask the centered ship for its explicit sailing notice
        # before treating an unchanged counter as a failed departure.
        if self._locate_and_click_ship():
            if getattr(self, '_last_ship_dialog', None) == 'sailing':
                self._dismiss_sailing_dialog()
                return 'sailing'
            if getattr(self, '_last_ship_dialog', None) == 'event':
                # Clicking the centered ship has opened its current event.
                # This is a work-state transition, not proof that the selected
                # route node was reached or that a navigation step was spent.
                logger.info('Grand Canal current ship opened an event while confirming departure')
                return 'event'
            if getattr(self, '_last_ship_dialog', None) == 'current':
                if (self._dismiss_current_location_dialog()
                        and self._wait_canal_page(timeout=4) is True):
                    resources = self._read_canal_resources()
                    if resources is not None and resources[0] < before:
                        return True
                    # A voyage may finish without a visible counter change.
                    # Only a live fix at the selected node can prove arrival;
                    # neither the saved point nor closing a notice suffices.
                    destination = getattr(self, '_last_destination_node', None)
                    if destination:
                        observed = [
                            (point[2], point[3]) for point in
                            self._detect_blue_destinations(full=True)
                        ]
                        match = locate_ship(observed, previous=destination)
                        if match is not None and match[0] == destination:
                            logger.info(
                                f'Grand Canal arrival confirmed by live map: {destination}'
                            )
                            return True
        logger.warning(f'Grand Canal departure unconfirmed: steps remain {before}')
        self.save_abnormal_screenshot('grand_canal_departure_unconfirmed')
        return False

    def _handle_event_transport(self):
        """Advance either the event work page or its subsequent departure prompt."""
        # Wait for the first action to settle before reading the next page.
        self.device.sleep(1.5)
        deadline = time.time() + 15
        next_action = None
        departure_retries = 0
        self._event_at_current_ship = False
        while time.time() < deadline:
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if hasattr(self, '_is_resource_shortage') and self._is_resource_shortage():
                return 'shortage'
            action = self._find_red_action_button(self.device.image)
            event_visible = action is not None and self._is_event_dialog(similarity=0.72)
            # Shore buildings can form a red rectangle of the same size as
            # a work button. Only a confirmed event modal makes it an action;
            # a map decoration must not suppress spent-step confirmation.
            if (not event_visible and hasattr(self, '_read_canal_resources')
                    and self._is_canal_page()
                    and getattr(self, '_route_steps_before_action', None) is not None):
                resources = self._read_canal_resources()
                if resources is not None and resources[0] < self._route_steps_before_action:
                    logger.info('Grand Canal event departure returned directly to map; step spent')
                    return 'departed'
                if resources is not None and hasattr(self, '_wait_navigation_step_spent'):
                    # A voyage may still be animating before its steps are
                    # charged. Reuse the map/ship sailing confirmation instead
                    # of waiting for a work modal over the moving map.
                    spent = self._wait_navigation_step_spent()
                    if spent == 'event':
                        if not self._event_at_current_ship:
                            deadline = time.time() + 15
                        self._event_at_current_ship = True
                        continue
                    return 'sailing' if spent == 'sailing' else ('departed' if spent else False)
            if event_visible:
                text = self._read_action_text(action)
                logger.attr('GrandCanalEventAction', text)
                # Some event destinations show their work preview first.
                # Clicking its action opens a separate "-1 出发" prompt.
                # Waiting only for another work button leaves that prompt
                # untouched indefinitely.
                if '出发' in text:
                    if departure_retries >= 2:
                        break
                    departure_retries += 1
                    logger.info(
                        f'Grand Canal confirm event departure '
                        f'({departure_retries}/2)'
                    )
                    self.device.click(self._departure_click_button(action))
                    page_result = self._wait_canal_page(timeout=8)
                    if page_result == 'shortage':
                        return 'shortage'
                    if page_result == 'sailing':
                        return 'sailing'
                    if page_result is True:
                        spent = self._wait_navigation_step_spent()
                        if spent == 'event':
                            if not self._event_at_current_ship:
                                deadline = time.time() + 15
                            self._event_at_current_ship = True
                            continue
                        return 'sailing' if spent == 'sailing' else ('departed' if spent else False)
                    self.device.sleep(0.5)
                    continue
                if ('运输' in text or '货物' in text or '开始工作' in text
                        or '完成大单子' in text):
                    next_action = action
                    self._last_event_action = text
                    break
            self.device.sleep(0.35)
        if next_action is None:
            if departure_retries:
                logger.warning('Grand Canal departure prompt remained after bounded confirmation attempts')
                self.save_abnormal_screenshot('grand_canal_departure_not_committed')
            else:
                logger.warning('Grand Canal event work button was not recognized')
                self.save_abnormal_screenshot('grand_canal_event_action_missing')
            return False
        existing_cards = {
            slot for slot, _ in self._read_task_card_durations()
        }
        self._last_event_duration = None
        self.device.click(next_action)

        deadline = time.time() + 18
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if hasattr(self, '_is_resource_shortage') and self._is_resource_shortage():
                return 'shortage'
            if self._is_sailing_dialog():
                self._dismiss_sailing_dialog()
                return 'sailing'
            if self.appear(
                GRAND_CANAL_ASSIGNEE_CHECK, offset=(5, 5), similarity=0.78
            ):
                break
            # “开始工作”可直接开始倒计时，不一定经过选人页面。
            if not self._is_event_dialog(similarity=0.72) and self._is_canal_page():
                new_cards = [
                    duration
                    for slot, duration in self._read_task_card_durations()
                    if slot not in existing_cards
                ]
                if new_cards:
                    self._last_event_duration = min(new_cards)
                    logger.info(
                        f'Grand Canal event work started directly: '
                        f'{self._last_event_duration}'
                    )
                    return True
        else:
            logger.warning('Grand Canal work neither started nor opened assignee panel')
            self.save_abnormal_screenshot('grand_canal_work_transition_timeout')
            return False

        # The full first row includes a Details control below the portrait.
        # Tap only the portrait; a row-wide random click can leave this page
        # open without assigning anyone.
        assignee_portrait = Button(
            area=(), color=(), button=(360, 105, 420, 175),
            name='GRAND_CANAL_FIRST_ASSIGNEE_PORTRAIT',
        )
        for attempt in range(3):
            self.device.click(assignee_portrait)
            page_result = self._wait_canal_page(timeout=8)
            if page_result == 'shortage':
                return 'shortage'
            if page_result == 'sailing':
                return 'sailing'
            if page_result is True:
                break
            self.device.screenshot()
            if not self.appear(
                GRAND_CANAL_ASSIGNEE_CHECK, offset=(5, 5), similarity=0.74
            ):
                logger.warning('Grand Canal assignee page changed unexpectedly')
                return False
            logger.warning(
                f'Grand Canal assignee portrait did not respond '
                f'({attempt + 1}/3)'
            )
        else:
            logger.warning('Grand Canal did not return after assigning person')
            self.save_abnormal_screenshot('grand_canal_assignee_stuck')
            return False
        new_cards = [
            duration
            for slot, duration in self._read_task_card_durations()
            if slot not in existing_cards
        ]
        if new_cards:
            self._last_event_duration = min(new_cards)
        return True

    def _record_route_departure(self):
        """Advance the displayed point from a confirmed map move, not OCR."""
        destination = getattr(self, '_last_destination_node', None)
        origin = str(self._get('LastPoint', '') or '').strip().upper()
        _, neighbors, _ = load_topology()
        pending = getattr(self, '_last_custom_route_move', None)
        approach = getattr(self, '_last_approach_move', None)
        approach_move = bool(
            approach and approach[0] == origin and approach[-1] == destination
            and all(b in neighbors.get(a, ()) for a, b in zip(approach, approach[1:]))
        )
        route_move = False
        if pending:
            route, start_index, target_index = pending
            try:
                route_move = (route == parse_custom_route(self._get('CustomRoute', ''))
                              and route[start_index] == origin
                              and route[target_index] == destination)
            except ValueError:
                pass
        if not destination or (destination not in neighbors.get(origin, ())
                               and not route_move and not approach_move):
            # The screen click may succeed even when the overview map cannot
            # identify its node. Do not present the old node as the arrival.
            self._set('LastPoint', '')
            self._set('LastPointSource', 'unknown')
            self._set('PreviousPoint', '')
            self.config.update()
            logger.warning('Grand Canal arrived at an unmapped blue destination')
            return
        self._set('PreviousPoint', origin)
        self._set('LastPoint', destination)
        self._set('LastPointSource', 'computed')
        if route_move:
            _, progress = custom_route_position(route, destination, progress=json.dumps({
                'route': json.loads(self._get('RouteProgress', '{}') or '{}').get('route'),
                'index': target_index,
            }))
            self._set('RouteProgress', progress)
        self.config.update()
        logger.attr('GrandCanalComputedPosition', destination)

    def _advance_route(self, state):
        harvested = False
        self._map_routing_active = True
        move_shortage_retries = {}
        bypass_current_event = False
        self._route_heading = None
        try:
            self._route_heading = float(self._get('LastRouteHeading')) % 360
            logger.info(
                f'Resume Grand Canal direction '
                f'{int((self._route_heading + 22.5) // 45) % 8}'
            )
        except (TypeError, ValueError):
            pass
        for step in range(1, self.MAX_ROUTE_STEPS + 1):
            logger.hr(f'GRAND CANAL ROUTE STEP {step}', level=2)
            duration = self._read_working_duration()
            if duration:
                finish = datetime.now().replace(microsecond=0) + duration
                return finish, f'[大运河任务][{finish}][进行中]'
            if getattr(self, '_ship_working_notice', False):
                return datetime.now().replace(microsecond=0) + timedelta(minutes=1), '[任务计时未识别][等待复查]'

            # Test fixtures load only selected methods; a real task always
            # has the resource preflight and checks both counters every step.
            if hasattr(self, '_prepare_route_action'):
                ready, reason = self._prepare_route_action()
                if not ready:
                    return None, reason

            if bypass_current_event:
                # A resource-starved event can be attached to the ship's
                # current position. Clicking the ship again would reopen it.
                # Arrival does not recenter the camera. Locate the ship again
                # without clicking it, then select a blue destination.
                bypass_current_event = False
                self.device.screenshot()
                if not self._is_canal_page():
                    return None, '[跳过缺材料事件后航道未确认][等待复查]'
                if not self._center_ship_on_map():
                    return None, '[跳过缺材料事件后定位失败][等待复查]'
                self._last_ship_dialog = 'current'
                logger.info('Skip resource-starved current event and choose next destination')
            else:
                if not self._locate_and_click_ship():
                    return None, '[定位漕船失败][等待复查]'
                if getattr(self, '_last_ship_dialog', None) == 'sailing':
                    self._dismiss_sailing_dialog()
                    return (datetime.now().replace(microsecond=0)
                            + timedelta(minutes=1)), '[航行中][等待结束]'

            # An active event may use a different task-card portrait. Check
            # the centered ship's popup before treating its red button as work.
            if (getattr(self, '_last_ship_dialog', None) in ('event', 'repair')
                    and hasattr(self, '_read_ship_working_duration')):
                remaining = self._read_ship_working_duration()
                if remaining is not None:
                    finish = datetime.now().replace(microsecond=0) + remaining + timedelta(seconds=3)
                    return finish, f'[大运河任务][{finish}][进行中]'
                if getattr(self, '_ship_working_notice', False):
                    return datetime.now().replace(microsecond=0) + timedelta(minutes=1), '[任务计时未识别][等待复查]'

            if self._collect_reward(state):
                harvested = True
                # 收下后立即继续同一轮航行，寻找下一次运输事件。
                continue

            if self._is_current_location_dialog():
                if not self._dismiss_current_location_dialog():
                    return None, '[当前地点提示未关闭][等待复查]'

            direct_event = getattr(self, '_last_ship_dialog', None) == 'event'
            if direct_event:
                # A pending event at the ship's current position opens its
                # final work dialog directly. There is no destination or
                # departure button to click before "开始工作".
                dialog = 'event'
                logger.info('Grand Canal ship opened pending event work')
            else:
                # The centered ship may open a no-event notice. Never
                # interpret "知道了" as the route's "出发" action.
                excluded = []
                for _ in range(8):
                    if not self._click_first_destination(excluded=excluded):
                        if getattr(self, '_map_position_unconfirmed', False):
                            return None, '[地图位置未确认][等待复查]'
                        return None, '[未识别到蓝色目的地][等待复查]'
                    dialog = self._wait_route_dialog()
                    if dialog == 'sailing':
                        self._dismiss_sailing_dialog()
                        return (datetime.now().replace(microsecond=0)
                                + timedelta(minutes=1)), '[航行中][等待结束]'
                    if dialog != 'current':
                        break
                    excluded.append(self._last_destination_center)
                    logger.warning(
                        f'Grand Canal node {self._last_destination_center} '
                        'opened the current location; try another destination'
                    )
                    if not self._dismiss_current_location_dialog():
                        return None, '[当前地点提示未关闭][等待复查]'
                else:
                    return None, '[蓝色目的地均指向当前地点][等待复查]'
            if dialog is None:
                return None, '[目的地弹窗识别失败][等待复查]'
            self.device.sleep(0.3)
            self.device.screenshot()
            dialog_stable = (self._is_event_dialog(similarity=0.75) if dialog == 'event'
                             else self.appear(GRAND_CANAL_MOVE_DIALOG_CHECK,
                                              offset=(6, 6), similarity=0.75))
            if not dialog_stable:
                logger.warning(
                    'Grand Canal destination dialog moved before departure; '
                    'retry on next run'
                )
                return None, '[目的地弹窗未稳定][等待复查]'
            if not direct_event:
                action = self._find_red_action_button(self.device.image)
                if action is None:
                    return None, '[出发按钮识别失败][等待复查]'
                logger.info(f'Grand Canal destination dialog: {dialog}')
                action_text = (getattr(self, '_read_action_text', lambda button: '出发')(action)
                               if dialog == 'event' else '出发')
                # A final work action belongs to the work handler. Clicking it
                # here would start work and leave that handler waiting on a map.
                if dialog != 'event' or '出发' in action_text:
                    target = self._departure_click_button(action) if dialog == 'event' else action
                    self.device.click(target)

            if dialog == 'event':
                event_result = self._handle_event_transport()
                # A confirmation fallback can discover work at the ship's
                # current node. Process it without advancing a guessed route.
                direct_event = direct_event or getattr(self, '_event_at_current_ship', False)
                if event_result == 'departed':
                    if not direct_event and hasattr(self, '_record_route_departure'):
                        self._record_route_departure()
                    self._route_heading = getattr(
                        self, '_last_destination_angle', None
                    )
                    if self._route_heading is not None:
                        self._set('LastRouteHeading', self._route_heading)
                    move_shortage_retries.clear()
                    # Arrival can reveal another event. Inspect that node
                    # on the next iteration instead of assuming work started.
                    continue
                if event_result == 'shortage':
                    if not self._dismiss_resource_shortage():
                        return None, '[资源不足弹窗未关闭][等待复查]'
                    if not direct_event:
                        before = getattr(self, '_route_steps_before_action', None)
                        if before is not None and hasattr(self, '_read_canal_resources'):
                            after = self._read_canal_resources()
                            if after is None:
                                return None, '[资源不足后通航步数未确认][等待复查]'
                            if after[0] >= before:
                                destination = getattr(
                                    self, '_last_destination_center', None
                                )
                                attempts = move_shortage_retries.get(destination, 0) + 1
                                move_shortage_retries[destination] = attempts
                                if attempts >= 2:
                                    return None, '[缺材料且未能出发][等待复查]'
                                logger.info(
                                    'Grand Canal event cargo is short before '
                                    'departure; retry sailing to the same node'
                                )
                                continue
                    # Cargo short after arrival only blocks work at this node.
                    if not direct_event:
                        if hasattr(self, '_record_route_departure'):
                            self._record_route_departure()
                        self._route_heading = getattr(
                            self, '_last_destination_angle', None
                        )
                        if self._route_heading is not None:
                            self._set('LastRouteHeading', self._route_heading)
                        move_shortage_retries.clear()
                    bypass_current_event = True
                    logger.info('Skip cargo-starved event at arrived node; continue voyage')
                    continue
                if event_result == 'sailing':
                    self._route_heading = getattr(self, '_last_destination_angle', None)
                    if self._route_heading is not None and not direct_event:
                        self._set('LastRouteHeading', self._route_heading)
                    return (datetime.now().replace(microsecond=0)
                            + timedelta(minutes=1)), '[航行中][等待结束]'
                if not event_result:
                    return None, '[码头运输派遣失败][等待复查]'
                if not direct_event:
                    if hasattr(self, '_record_route_departure'):
                        self._record_route_departure()
                    self._route_heading = getattr(
                        self, '_last_destination_angle', None
                    )
                    if self._route_heading is not None:
                        self._set('LastRouteHeading', self._route_heading)
                    move_shortage_retries.clear()
                work_type = (
                    '大单子运输' if '完成大单子' in getattr(self, '_last_event_action', '')
                    else '事件工作'
                    if '开始工作' in getattr(self, '_last_event_action', '')
                    else '运输货物'
                )
                duration = self._read_working_duration(open_ship=True)
                if duration is None and not getattr(self, '_ship_working_notice', False):
                    duration = getattr(self, '_last_event_duration', None)
                if duration is None:
                    return None, '[运输倒计时识别失败][等待复查]'
                # 右上角任务卡是人物加速后的准确时间，只留三秒页面误差。
                finish = (
                    datetime.now().replace(microsecond=0)
                    + duration + timedelta(seconds=3)
                )
                prefix = '[已收获]' if harvested else ''
                return finish, (
                    f'{prefix}[{work_type}][{finish}][进行中]'
                )

            page_result = self._wait_canal_page()
            if page_result == 'shortage':
                if not self._dismiss_resource_shortage():
                    return None, '[资源不足弹窗未关闭][等待复查]'
                destination = getattr(self, '_last_destination_center', None)
                before = getattr(self, '_route_steps_before_action', None)
                after = (self._read_canal_resources()
                         if hasattr(self, '_read_canal_resources') else None)
                if after is not None and before is not None and after[0] < before:
                    # A navigation step was spent, so departure succeeded;
                    # only the destination's task was short of cargo.
                    if hasattr(self, '_record_route_departure'):
                        self._record_route_departure()
                    self._route_heading = getattr(
                        self, '_last_destination_angle', None
                    )
                    if self._route_heading is not None:
                        self._set('LastRouteHeading', self._route_heading)
                    bypass_current_event = True
                    logger.info('Grand Canal sailed despite cargo shortage; skip node task')
                    continue
                attempts = move_shortage_retries.get(destination, 0) + 1
                move_shortage_retries[destination] = attempts
                if attempts >= 2:
                    logger.warning('Grand Canal departure still blocked after retry')
                    return None, '[出发未完成且缺材料][等待复查]'
                logger.info('Grand Canal departure not confirmed; retry same route once')
                continue
            if page_result == 'sailing':
                self._route_heading = getattr(self, '_last_destination_angle', None)
                if self._route_heading is not None:
                    self._set('LastRouteHeading', self._route_heading)
                self._dismiss_sailing_dialog()
                return (datetime.now().replace(microsecond=0)
                        + timedelta(minutes=1)), '[航行中][等待结束]'
            if not page_result:
                self.save_abnormal_screenshot('grand_canal_move_not_returned')
                return None, '[普通航行未返回地图][等待复查]'
            if hasattr(self, '_wait_navigation_step_spent'):
                spent = self._wait_navigation_step_spent()
                if spent == 'event':
                    # Close this verified modal before the next iteration
                    # centers the ship and handles its pending event directly.
                    self.device.press_back()
                    if self._wait_canal_page(timeout=4) is not True:
                        return None, '[当前事件弹窗未关闭][等待复查]'
                    continue
                if spent == 'sailing':
                    self._route_heading = getattr(self, '_last_destination_angle', None)
                    if self._route_heading is not None:
                        self._set('LastRouteHeading', self._route_heading)
                    return (datetime.now().replace(microsecond=0)
                            + timedelta(minutes=1)), '[航行中][等待结束]'
                if not spent:
                    return None, '[出发后通航步数未变化][等待复查]'
            if hasattr(self, '_record_route_departure'):
                self._record_route_departure()
            self._route_heading = getattr(self, '_last_destination_angle', None)
            if self._route_heading is not None:
                self._set('LastRouteHeading', self._route_heading)
            move_shortage_retries.clear()

        return None, '[连续航行达到安全上限][等待复查]'

    def run(self):
        logger.hr('JIANGNAN YANGZHOU GRAND CANAL', level=0)
        now = datetime.now().replace(microsecond=0)
        if not bool(self._get('Enable', True)):
            self.config.task_delay(success=True)
            return True
        configured = self._parse_time(self._get('NextRun')) or self.DEFAULT_TIME
        claim_due = self._next_sinan_claim_due(now)
        if configured > now and claim_due > now:
            self.config.task_delay(target=min(configured, claim_due))
            return True

        state = JiangnanInventoryState(self.config)
        if not self._enter_canal():
            self._sinan_claim_failed = True
            retry = now + self.RETRY
            self._set_schedule(retry, '[进入扬州大运河失败][等待复查]')
            return False

        claim_ok = self._claim_daily_sinan()
        self._sinan_claim_failed = not claim_ok
        if not claim_ok:
            if getattr(self, '_sinan_claim_deferred', False):
                logger.info('Grand Canal daily claim postponed to avoid overflow')
            else:
                logger.warning('Grand Canal daily claim check incomplete; will retry')
            if self._wait_canal_page(timeout=5) is not True:
                self._leave_canal()
                self._set_schedule(
                    datetime.now().replace(microsecond=0) + self.RETRY,
                    '[司南任务界面未返回航道][等待复查]',
                )
                return False
        if configured <= now:
            next_run, status = self._advance_route(state)
        else:
            next_run = configured
            status = ('[司南任务领取检查]' if claim_ok
                      else '[司南任务领取检查失败][等待复查]')
        self._leave_canal()
        if next_run is None:
            next_run = datetime.now().replace(microsecond=0) + self.RETRY
        self._set_schedule(next_run, status)
        return True
