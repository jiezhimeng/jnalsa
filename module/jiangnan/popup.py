import time
from datetime import datetime

from module.jiangnan.ui import JiangnanUI
from module.jiangnan.login import JiangnanLogin
from module.jiangnan.assets import (
    SIGNIN_CUMULATIVE_CHECK,
    SIGNIN_HINT_CHECK,
    SIGNIN_CLAIM_BUTTON,
    REWARD_TAKE_BUTTON,

    TAX_POPUP_CHECK,
    TAX_KNOW_BUTTON,
    NETWORK_TIMEOUT_CHECK,
    NETWORK_RETRY_BUTTON,
)
from module.logger import logger
from module.jiangnan.network_popup import network_retry_ready, read_retry_countdown


class JiangnanPopup(JiangnanLogin, JiangnanUI):

    def is_network_timeout_popup(self):
        message = self.appear(
            NETWORK_TIMEOUT_CHECK, offset=(4, 4), similarity=0.88
        )
        # 倒计时期间“重试(0)”会变灰，不能拿红色按钮模板否定弹窗。
        self.log_check('NetworkTimeoutCheck', message)
        return message

    def handle_network_timeout_popup(self, attempts=3):
        """Wait for red retry; force-restart immediately for countdowns >20s."""
        if not self.is_network_timeout_popup():
            return False

        previous = getattr(self.device, '_jiangnan_network_recovery_active', False)
        self.device._jiangnan_network_recovery_active = True
        try:
            return self._recover_network_timeout_popup(attempts)
        finally:
            self.device._jiangnan_network_recovery_active = previous

    def _restart_after_network_failure(self, reason):
        logger.warning(reason)
        if getattr(self, '_jiangnan_restart_active', False):
            self._network_failed_during_restart = True
            return True
        from module.exception import RequestHumanTakeover

        restart = getattr(self, 'restart_game_until_ready', None)
        if not callable(restart) or not restart(reason=reason, force_stop=True):
            raise RequestHumanTakeover('Network recovery could not restart the game')
        return True

    def _recover_network_timeout_popup(self, attempts):

        logger.hr('JIANGNAN NETWORK RECOVERY')
        self.device.stuck_record_clear()
        self.device.click_record_clear()
        for attempt in range(1, attempts + 1):
            # Allow the full 20-second countdown plus a small frame margin.
            enabled_deadline = time.monotonic() + 23
            next_countdown_read = 0
            while time.monotonic() < enabled_deadline:
                if not self.is_network_timeout_popup():
                    logger.info('Network popup disappeared before retry')
                    return True
                if network_retry_ready(self.device.image):
                    break
                if time.monotonic() >= next_countdown_read:
                    countdown = read_retry_countdown(self.device.image)
                    logger.attr('NetworkRetryCountdown', countdown)
                    if countdown is not None and countdown > 20:
                        return self._restart_after_network_failure(
                            f'network retry countdown {countdown}s exceeds 20s'
                        )
                    next_countdown_read = time.monotonic() + 1
                self.device.sleep(0.4)
                self.device.stuck_record_clear()
                self.device.screenshot()
            else:
                return self._restart_after_network_failure(
                    'network retry button stayed grey beyond 20s countdown'
                )
            logger.warning(f'Network retry: {attempt}/{attempts}')
            self.device.click(NETWORK_RETRY_BUTTON)
            deadline = time.monotonic() + 8
            absent_since = None
            while time.monotonic() < deadline:
                self.device.sleep(0.4)
                self.device.stuck_record_clear()
                self.device.screenshot()
                if self.is_network_timeout_popup():
                    if absent_since is not None or not network_retry_ready(self.device.image):
                        # A new failed request may start a longer countdown.
                        # Read it immediately in the next attempt.
                        break
                    absent_since = None
                    continue
                if absent_since is None:
                    absent_since = time.monotonic()
                # 留出三秒观察请求结果，避免弹窗短暂消失后又出现时
                # 被错误当成恢复成功。
                if time.monotonic() - absent_since >= 3:
                    logger.info(f'Network recovered after retry {attempt}')
                    return True

        return self._restart_after_network_failure('network retry exhausted')

    def is_signin_popup(self):
        """
        判断当前是否为月签到弹窗。

        调用前必须已经有 self.device.image。
        """

        cumulative = self.appear(
            SIGNIN_CUMULATIVE_CHECK,
            offset=(3, 3),
            similarity=0.90
        )

        hint = False
        if cumulative:
            hint = self.appear(
                SIGNIN_HINT_CHECK,
                offset=(3, 3),
                similarity=0.90
            )

        self.log_check(
            'SigninCumulativeCheck',
            cumulative
        )

        self.log_check(
            'SigninHintCheck',
            hint
        )

        return cumulative and hint

    def is_tax_popup(self):
        """
        判断是否出现税课司离线收益弹窗。
        """

        title = self.appear(
            TAX_POPUP_CHECK,
            offset=(3, 3),
            similarity=0.90
        )

        know = False
        if title:
            know = self.appear(
                TAX_KNOW_BUTTON,
                offset=(3, 3),
                similarity=0.90
            )

        self.log_check(
            'TaxPopupCheck',
            title
        )

        self.log_check(
            'TaxKnowCheck',
            know
        )

        return title and know

    def handle_tax_popup(self):
        """
        处理任意时机出现的税课司收益，不依赖启动或切城上下文。
        """

        from module.jiangnan.inventory_state import JiangnanInventoryState
        from module.jiangnan.tax_receipt import TAX_CURRENCIES, read_tax_resources

        if not self.is_tax_popup():
            state = getattr(self.config, '_jiangnan_inventory_state', None)
            if state is not None:
                self._recover_tax_receipts(state)
            return False

        logger.hr('JIANGNAN TAX POPUP')

        logger.info(
            'Tax popup detected'
        )

        state = getattr(self.config, '_jiangnan_inventory_state', None)
        if state is None:
            state = JiangnanInventoryState(self.config)
        resources = read_tax_resources(self.device.image)
        self.device.sleep(0.2)
        self.device.screenshot()
        if not self.is_tax_popup():
            return True
        confirmed = read_tax_resources(self.device.image)
        if resources != confirmed:
            resources = None
        city_hint = getattr(self, '_tax_popup_city_hint', None)
        receipt_id = (state.capture_tax_popup(resources, city_hint=city_hint)
                      if resources is not None else None)
        self.device.click(self.anchored_button(TAX_POPUP_CHECK, TAX_KNOW_BUTTON))

        self.device.sleep(0.8)

        self.device.screenshot()
        if self.is_tax_popup() or self.is_network_timeout_popup():
            logger.info('Tax popup close not confirmed; retain captured receipt')
            return True
        if resources and set(resources) <= TAX_CURRENCIES:
            # Currency is not warehouse stock. A verified currency-only reward
            # needs neither city OCR nor a warehouse scan if dismissal was slow.
            state.close_tax_popup_receipt(receipt_id, reason='currency_only')
            logger.info('Tax popup contains only currency; warehouse stock unchanged')
            return True
        # Background seals are deliberately blurred while the popup is open.
        # Identify the actual city on two fresh frames after dismissing it.
        city = self._confirm_tax_popup_city()
        # The verified visible city owns the receipt. Navigation hints are
        # only recovery clues, never a condition for accepting a valid popup.
        if receipt_id is not None and city is not None:
            state.apply_tax_popup_receipt(city, resources, receipt_id)
            state.close_tax_popup_receipt(receipt_id)
        elif receipt_id is not None and (self.is_tax_popup() or self.is_network_timeout_popup()):
            logger.info('Tax city confirmation interrupted; retain captured receipt')
            return True
        else:
            state.request_scan('税课司收益或归属城市识别不确定',
                               cities={value for value in (city, city_hint) if value} or None)
            if receipt_id is not None:
                state.close_tax_popup_receipt(receipt_id, reason='needs_scan')
        logger.info('Tax popup closed and inventory receipt handled')

        return True

    def _confirm_tax_popup_city(self):
        """Read the unblurred city on two frames after the receipt disappears."""
        from module.jiangnan.tax_receipt import read_tax_city

        for _ in range(3):
            first = read_tax_city(self.device.image)
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.is_tax_popup() or self.is_network_timeout_popup():
                return None
            second = read_tax_city(self.device.image)
            if first is not None and first == second:
                return first
        return None

    def _recover_tax_receipts(self, state):
        """Recover interrupted receipt writes before the task can change cities."""
        from module.jiangnan.tax_receipt import TAX_CURRENCIES
        for receipt in list(state.data.get('tax_popup_receipts', [])):
            if receipt.get('closed_at'):
                continue
            if self.is_network_timeout_popup():
                return
            if receipt.get('resources') and set(receipt['resources']) <= TAX_CURRENCIES:
                state.close_tax_popup_receipt(receipt['id'], reason='currency_only')
                logger.info('Recovered currency-only tax receipt without warehouse scan')
                continue
            city = receipt.get('city') or receipt.get('city_hint')
            if receipt.get('phase') == 'applied':
                state.close_tax_popup_receipt(receipt['id'])
            elif receipt.get('phase') == 'prepared' and city:
                state.apply_tax_popup_receipt(city, receipt['resources'], receipt['id'])
                state.close_tax_popup_receipt(receipt['id'])
            elif receipt.get('phase') == 'captured':
                # A slow dismissal leaves the receipt captured at the first
                # 0.8s check. The next popup pass still runs before navigation:
                # confirm the now-visible city and complete the same receipt,
                # just as for an immediately closed popup. A hint is not proof.
                confirmed_city = self._confirm_tax_popup_city()
                if confirmed_city is not None:
                    state.apply_tax_popup_receipt(
                        confirmed_city, receipt['resources'], receipt['id'])
                    state.close_tax_popup_receipt(receipt['id'])
                elif self.is_tax_popup() or self.is_network_timeout_popup():
                    return
                else:
                    state.request_scan('税课司收益归属城市识别不确定',
                                       cities={city} if city else None)
                    state.close_tax_popup_receipt(receipt['id'], reason='needs_scan')
            else:
                # A crash after dismissal can lose the blurred popup's city.
                # An actual warehouse scan includes the reward without a guess.
                state.request_scan('税课司收益领取被中断，需实扫确认',
                                   cities={city} if city else None)
                state.close_tax_popup_receipt(receipt['id'], reason='needs_scan')

    def handle_signin_popup(self):
        """
        自动领取月签到奖励。

        这里只负责：
            签到页 -> 点击签到

        后续奖励弹窗交给 handle_reward_popup()。
        """

        if not self.is_signin_popup():
            return False

        logger.hr('JIANGNAN SIGNIN')
        logger.info('Signin popup detected')

        # 领取奖励后月签日历仍会留在屏幕上；再次点击底部提示不会关闭，
        # 会让跨城等待循环不断点击同一位置。同一天已经尝试过签到时，
        # 第二次看到日历就按返回关闭它；跨日后必须允许领取新的奖励。
        today = datetime.now().date()
        if getattr(self, '_signin_claim_attempted_date', None) == today:
            logger.info('Signin reward already attempted; close calendar')
            self.device.press_back()
            self.device.sleep(0.8)
            return True

        logger.info('Claim daily signin reward')

        self.device.click(
            self.anchored_button(
                SIGNIN_CUMULATIVE_CHECK,
                SIGNIN_CLAIM_BUTTON,
                name='SIGNIN_CLAIM_BUTTON_DYNAMIC',
                inset=8,
            )
        )

        self._signin_claim_attempted_date = today

        self.device.sleep(1.2)

        return True

    def handle_startup_popup(self):
        """
        统一处理任意时机的已知弹窗；保留既有方法名供各流程调用。
        """

        # 网络弹窗覆盖所有页面，优先级最高。
        if self.handle_network_timeout_popup():
            return True

        login_handler = getattr(self, 'handle_login', None)
        if callable(login_handler) and login_handler():
            return True

        # 生产设施搜索可能在任务切换时遗留居民事件面板。该面板后方
        # 仍能看到主界面按钮，必须优先于主界面和税课司弹窗判断。
        residence_handler = getattr(
            self, 'handle_open_residence_panel', None
        )
        if callable(residence_handler) and residence_handler():
            return True

        # 月签到
        if self.handle_signin_popup():
            return True

        # 通用奖励弹窗
        if self.handle_reward_popup():
            return True

        # 税课司
        if self.handle_tax_popup():
            return True

        return False

    def is_reward_popup(self):
        """
        判断当前是否出现通用奖励领取弹窗。
        """

        reward = self.appear(
            REWARD_TAKE_BUTTON,
            offset=(5, 5),
            similarity=0.90
        )

        self.log_check(
            'RewardTakeCheck',
            reward
        )

        return reward

    def handle_reward_popup(self):
        """
        处理通用“收下”奖励弹窗。
        """

        if not self.is_reward_popup():
            return False

        logger.hr('JIANGNAN REWARD')
        logger.info('Reward popup detected')

        self.device.click(
            REWARD_TAKE_BUTTON
        )

        self.device.sleep(1.0)

        logger.info('Reward accepted')

        return True
