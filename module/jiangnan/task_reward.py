"""Claim daily task rewards when Yan's main-screen reward portrait appears."""
import time

from module.base.button import Button
from module.jiangnan.app import JiangnanApp
from module.jiangnan.task_reward_trigger import park_task
from module.jiangnan.task_reward_vision import (
    PANEL, CLAIM, reward_prompt, task_panel, daily_page, claim_state, reward_take,
)
from module.logger import logger


class JiangnanTaskReward(JiangnanApp):
    def _reward_click(self, area, name):
        x1, y1, x2, y2 = area
        self.device.click(Button(area=(), color=(),
            button=(x1+8, y1+8, x2-8, y2-8), name=name))

    def open_daily_tasks(self, entry, timeout=12):
        self._reward_click(entry, 'DAILY_TASK_REWARD_ENTRY')
        deadline = time.monotonic() + timeout
        tab_clicked = False
        last_tab_click = float('-inf')
        while time.monotonic() < deadline:
            self.device.sleep(.3)
            self.device.screenshot()
            panel = task_panel(self.device.image)
            if panel is not None:
                # Always select the top book tab, even if it was selected.
                # Other task tabs must never be treated as daily rewards.
                if tab_clicked and daily_page(self.device.image):
                    return True
                if time.monotonic() - last_tab_click >= 2:
                    dx, dy = panel[0]-PANEL[0], panel[1]-PANEL[1]
                    self._reward_click((235+dx, 91+dy, 298+dx, 168+dy),
                                       'DAILY_TASK_BOOK_TAB')
                    tab_clicked = True
                    last_tab_click = time.monotonic()
        self.save_abnormal_screenshot('daily_task_page_timeout')
        return False

    def claim_daily_rewards(self, timeout=20):
        deadline = time.monotonic() + timeout
        last_claim = last_take = float('-inf')
        done_frames = 0
        while time.monotonic() < deadline:
            self.device.screenshot()
            image = self.device.image
            take = reward_take(image)
            if take is not None:
                done_frames = 0
                if time.monotonic() - last_take >= 1.5:
                    self._reward_click(take, 'DAILY_TASK_REWARD_TAKE')
                    last_take = time.monotonic()
            else:
                state = claim_state(image)
                if state == 'done':
                    done_frames += 1
                    if done_frames >= 2:
                        logger.info('Daily task rewards confirmed: one-click claim is grey')
                        return True
                else:
                    done_frames = 0
                    if state == 'available' and time.monotonic() - last_claim >= 2:
                        self._reward_click(CLAIM, 'DAILY_TASK_REWARD_CLAIM')
                        last_claim = time.monotonic()
            # No take popup is required: a grey claim button is sufficient.
            self.device.sleep(.4)
        self.save_abnormal_screenshot('daily_task_claim_timeout')
        return False

    def run(self):
        logger.hr('JIANGNAN DAILY TASK REWARD', level=1)
        success = False
        if self.ensure_ready():
            self.device.screenshot()
            entry = reward_prompt(self.device.image)
            if entry is None:
                logger.info('No daily task reward portrait; wait for next visual event')
                success = True
            else:
                opened = self.open_daily_tasks(entry)
                claimed = self.claim_daily_rewards() if opened else False
                returned = self.ui_goto_main()
                success = opened and claimed and returned
        park_task(self.device, self.config, success)
        return success
