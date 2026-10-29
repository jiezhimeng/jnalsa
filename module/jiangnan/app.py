import time

from module.exception import RequestHumanTakeover
from module.jiangnan.popup import JiangnanPopup
from module.logger import logger


class JiangnanApp(JiangnanPopup):

    def restart_game_until_ready(
        self, reason='unknown page', attempts=3, timeout=90, force_stop=False
    ):
        """最多重启游戏三次，全部失败后才请求人工接管。"""
        if getattr(self, '_jiangnan_restart_active', False):
            return False

        self._jiangnan_restart_active = True
        previous_network_guard = getattr(self.device, '_jiangnan_network_recovery_active', False)
        self.device._jiangnan_network_recovery_active = True
        try:
            for attempt in range(1, attempts + 1):
                logger.hr(
                    f'JIANGNAN GAME RESTART {attempt}/{attempts}', level=1
                )
                logger.warning(f'Restart reason: {reason}')
                self._network_failed_during_restart = False
                try:
                    if force_stop:
                        self.device.app_stop_adb()
                    else:
                        self.device.app_stop()
                    self.device.sleep(2)
                    self.device.app_start()
                    # Give each fresh launch its own loading timeout.
                    self._login_deadline = 0
                    self._login_state = None
                    self._login_clicks = {}
                    self._login_last_click = {}
                    self.device.stuck_record_clear()
                    if self.wait_until_main(timeout=timeout):
                        logger.info(
                            f'Jiangnan recovered after restart '
                            f'{attempt}/{attempts}'
                        )
                        return True
                except RequestHumanTakeover as error:
                    logger.error(
                        f'Jiangnan restart {attempt}/{attempts} failed: '
                        f'{error}'
                    )
                except Exception as error:
                    logger.exception(error)
                    logger.error(
                        f'Jiangnan restart {attempt}/{attempts} failed'
                    )

            logger.critical(
                f'Jiangnan failed after {attempts} game restarts; '
                'request human takeover'
            )
            raise RequestHumanTakeover(
                f'Jiangnan recovery failed after {attempts} restarts: {reason}'
            )
        finally:
            self._jiangnan_restart_active = False
            self.device._jiangnan_network_recovery_active = previous_network_guard

    def wait_until_main(self, timeout=90, interval=1.0):
        """
        等待游戏进入城市主界面。
        同时处理已知的启动弹窗。
        """

        logger.hr('JIANGNAN WAIT MAIN')

        deadline = time.time() + timeout
        next_foreground_check = time.time() + 5.0

        while time.time() < deadline:

            # 一轮只截一张图
            self.device.screenshot()

            # 模态弹窗后方仍显示主界面导航按钮，必须先处理弹窗，
            # 否则会把遮罩后的按钮当作可操作主界面。
            if self.handle_startup_popup():
                logger.info(
                    'Startup popup handled, recheck'
                )
                # The TapTap authorization gets a bounded wait of its own.
                deadline = max(deadline, getattr(self, '_login_deadline', 0) + 1)
                if getattr(self, '_network_failed_during_restart', False):
                    logger.warning('Current restart still has no network')
                    return False
                continue

            # -------------------------
            # 已经进入主界面
            # -------------------------
            if self.is_unobstructed_main():
                logger.info(
                    'Jiangnan main page ready'
                )
                return True

            # Loaded game pages prove foreground without a slow package
            # query. Probe only when loading has persisted, at most every 10s.
            if time.time() >= next_foreground_check:
                running = self.app_is_running_recoverable()
                next_foreground_check = time.time() + 10.0
                if not running:
                    logger.info('Waiting for Jiangnan foreground')

            # -------------------------
            # 当前页面暂时不认识
            # -------------------------
            self.device.sleep(interval)

        logger.warning(
            f'Jiangnan main page timeout after {timeout}s'
        )

        try:
            self.device.image_save(
                'jiangnan_start_timeout.png'
            )
            logger.info(
                'Saved screenshot: '
                'jiangnan_start_timeout.png'
            )
        except Exception as e:
            logger.warning(
                f'Failed to save timeout screenshot: {e}'
            )

        if getattr(self, '_jiangnan_restart_active', False):
            return False
        return self.restart_game_until_ready(
            reason=f'main page timeout after {timeout}s'
        )



    def ensure_ready(self, start_timeout=90, popup_rounds=10):
        """
        确保江南百景图已经启动并处于城市主界面。

        优先级：
            1. 主界面
            2. 已知启动/奖励弹窗
            3. 普通二级页面 -> ui_goto_main()
        """

        logger.hr('JIANGNAN ENSURE READY')

        foreground_verified = False
        # 先连续处理可能叠加的已知弹窗
        for _ in range(popup_rounds):

            self.device.screenshot()

            # 模态弹窗优先于后方仍然可见的主界面锚点。
            if self.handle_startup_popup():

                logger.info(
                    'Known popup handled, recheck'
                )

                if getattr(self, '_login_deadline', 0):
                    return self.wait_until_main(timeout=start_timeout)

                self.device.sleep(0.2)

                continue

            # ------------------------------
            # 已经在主界面
            # ------------------------------
            identify = getattr(self, 'current_navigation_page', None)
            page = identify() if callable(identify) else (
                'main' if self.is_unobstructed_main() else 'unknown'
            )
            if page == 'main':

                logger.info(
                    'Jiangnan is already at main page'
                )

                return True

            # Always use a fresh frame first. The normal foreground query
            # takes ~5s on MuMu even when the game is already visible.
            if page in ('unknown', None):
                if not self.app_is_running_recoverable():
                    logger.info('Jiangnan is not running, start app')
                    self.device.app_start()
                    return self.wait_until_main(timeout=start_timeout)
                foreground_verified = True

            # ------------------------------
            # 既不是主界面，也不是已知弹窗
            # 说明大概率是普通二级页面
            # ------------------------------
            logger.info(
                'No known popup detected'
            )

            break

        # ==========================================
        # 3. 普通页面，使用 BACK 返回
        # ==========================================
        logger.info(
            'Try ui_goto_main'
        )

        if self.ui_goto_main(foreground_verified=foreground_verified):
            return True
        if getattr(self, '_jiangnan_restart_active', False):
            return False
        return self.restart_game_until_ready(
            reason='unable to return to main page'
        )
