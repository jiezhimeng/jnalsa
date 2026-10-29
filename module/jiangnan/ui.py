import time

import cv2

from module.base.button import Button
from module.jiangnan.base import JiangnanBase
from module.logger import logger
from module.jiangnan.assets import (
    INN_CRAFT_CHECK,
    INN_SPICE_CHECK,
    INN_EXIT_BUTTON,
    INN_EXIT_CONFIRM_CHECK,
    INN_EXIT_CONFIRM_BUTTON,
    GRAND_CANAL_EXIT_BUTTON,
    GRAND_CANAL_PAGE_CHECK,
    MAIN_BUILD_CHECK,
    MAIN_CHAT_CHECK,
    MAIN_TASK_CHECK,
)

class JiangnanUI(JiangnanBase):

    def post_station_page(self):
        from module.jiangnan.post_station_vision import page

        return page(self.device.image)

    def leave_post_station(self, timeout=90):
        """Recover an interrupted expedition without buying stamina or using BACK."""
        from module.jiangnan.post_station_vision import page

        def click(x, y, name):
            self.device.click(Button(area=(), color=(), button=(x-3, y-3, x+3, y+3), name=name))

        deadline = time.monotonic() + timeout
        loading = False
        last_click = 0
        main_frames = 0
        while time.monotonic() < deadline:
            self.device.screenshot()
            # Returning from an expedition can trigger offline tax income.
            # Use the shared handler so receipts are recorded before dismissal.
            handle_popup = getattr(self, 'handle_startup_popup', None)
            if callable(handle_popup) and handle_popup():
                main_frames = 0
                logger.info('Popup handled while leaving post station; recheck city')
                continue
            current = page(self.device.image)
            if current == 'unknown':
                identify = getattr(self, 'current_navigation_page', None)
                if (identify() == 'main' if callable(identify) else self.is_in_main()):
                    main_frames += 1
                    if main_frames >= 2:
                        return True
                else:
                    main_frames = 0
                self.device.sleep(.5)
                continue
            main_frames = 0
            if loading:
                self.device.sleep(.5)
                continue
            if time.monotonic() - last_click < 1:
                self.device.sleep(.25)
                continue
            if current == 'exit':
                click(816, 494, 'POST_STATION_RECOVER_RETURN_CITY')
                loading = True
            elif current == 'field':
                click(1220, 635, 'POST_STATION_RECOVER_LEAVE')
            elif current == 'exhausted':
                click(1020, 610, 'POST_STATION_RECOVER_DISMISS')
            elif current in ('list', 'characters', 'maps', 'overview', 'craft'):
                click(1145, 520, 'POST_STATION_RECOVER_CLOSE')
            last_click = time.monotonic()
            self.device.sleep(.4)
        self.save_abnormal_screenshot('post_station_recovery_timeout')
        return False

    def log_check(self, name, value, interval=2.0):
        """记录高频状态检测；变化立即写，同值按间隔限频。"""
        now = time.time()
        states = getattr(self, '_jiangnan_check_log_state', None)
        if states is None:
            states = {}
            self._jiangnan_check_log_state = states
        previous = states.get(name)
        if (
            previous is None
            or previous[0] != value
            or now - previous[1] >= interval
        ):
            logger.attr(name, value)
            states[name] = (value, now)

    def _main_navigation_panels(self):
        """返回右下主导航的红色面板，兼容不同城市的按钮偏移。"""
        image = self.device.image
        if image is None or image.size == 0:
            return []
        # 候选最终只允许出现在右下主导航区域，不必每次对整张
        # 1280x720 截图做 HSV 和连通域。保留 30px 余量避免裁边。
        origin_x, origin_y = 750, 500
        roi = image[origin_y:, origin_x:]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        red = cv2.inRange(hsv, (0, 90, 80), (14, 255, 255))
        red |= cv2.inRange(hsv, (170, 90, 80), (179, 255, 255))
        count, _, stats, _ = cv2.connectedComponentsWithStats(red)
        panels = []
        for left, top, width, height, area in stats[1:count]:
            left += origin_x
            top += origin_y
            if (
                left >= 780 and top >= 530
                and 55 <= width <= 100 and 90 <= height <= 175
                and area >= 2500
            ):
                panels.append((int(left), int(top), int(width), int(height)))
        panels.sort(key=lambda item: item[0])
        self.log_check('MainNavigationPanels', len(panels))
        return panels

    def _main_navigation_visible(self):
        """动态识别右下主导航，兼容不同城市导致的固定模板失配。"""
        return len(self._main_navigation_panels()) >= 3

    def find_main_build_navigation_button(self):
        """以右下导航最左侧红色面板定位“营造”入口。"""
        panels = self._main_navigation_panels()
        if len(panels) < 3:
            return None
        left, top, width, height = panels[0]
        return Button(
            area=(),
            color=(),
            button=(
                left + max(8, width // 5),
                top + max(18, height // 6),
                left + width - max(8, width // 5),
                top + height - max(14, height // 7),
            ),
            name='MAIN_BUILD_NAVIGATION_DYNAMIC',
        )

    def jiangnan_drag(
        self, p1, p2, name, hold_duration=0.2,
        swipe_duration=1.8, swipe_speed=None, swipe_min_distance=0,
        swipe_linear=True,
    ):
        """江南百景图精确拖动。

        默认使用确定性直线；短距离拖动保留取整后重复的中间触点，
        以保证同样达到 1.8 秒，不再使用随机贝塞尔曲线。若未显式指定 ``swipe_speed``，会根据
        路径长度和 ``swipe_duration`` 反算每 10ms 的位移量，
        使 MaaTouch/minitouch/nemu_ipc 的真实移动时间与日志一致。
        """
        if swipe_speed is None:
            distance = (
                (float(p2[0]) - float(p1[0])) ** 2
                + (float(p2[1]) - float(p1[1])) ** 2
            ) ** 0.5
            swipe_speed = max(
                0.1,
                distance / max(float(swipe_duration) * 100.0, 1.0),
            )
        logger.info(
            f'Jiangnan drag: {p1} -> {p2}, '
            f'move={swipe_duration}s/speed={swipe_speed:.3f}px-per-10ms, '
            f'min_step={swipe_min_distance}px, linear={swipe_linear}, '
            f'hold={hold_duration}s, name={name}'
        )
        self.device.drag(
            p1,
            p2,
            point_random=(0, 0, 0, 0),
            swipe_duration=swipe_duration,
            swipe_speed=swipe_speed,
            swipe_min_distance=swipe_min_distance,
            swipe_linear=swipe_linear,
            hold_duration=hold_duration,
            name=name,
        )

    @staticmethod
    def anchored_area(anchor, area):
        """把参考坐标区域移动到已识别页面锚点的实时位置。"""
        dx = int(anchor.button[0] - anchor._button[0])
        dy = int(anchor.button[1] - anchor._button[1])
        x1, y1, x2, y2 = area
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    @staticmethod
    def anchored_button(anchor, target, name=None, inset=0):
        """按已识别锚点的实时偏移生成点击区域。"""
        dx = int(anchor.button[0] - anchor._button[0])
        dy = int(anchor.button[1] - anchor._button[1])
        x1, y1, x2, y2 = JiangnanUI.anchored_area(
            anchor, target._button,
        )
        x1 += inset
        y1 += inset
        x2 -= inset
        y2 -= inset
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Invalid anchored button inset: {target}')
        button = Button(
            area=(),
            color=(),
            button=(x1, y1, x2, y2),
            name=name or f'{target.name}_ANCHORED',
        )
        logger.info(
            f'Anchored button: {target.name}, offset=({dx}, {dy}), '
            f'button={button.button}'
        )
        return button

    def is_inn_exit_confirm(self):
        """Require the inn-specific question and its affirmative action."""
        return (
            self.appear(INN_EXIT_CONFIRM_CHECK, offset=(12, 12), similarity=0.85)
            and self.appear(INN_EXIT_CONFIRM_BUTTON, offset=(12, 12), similarity=0.85)
        )

    def is_in_inn(self):
        """Two unique labels distinguish the inn from similar red menus."""
        return (
            self.appear(INN_CRAFT_CHECK, offset=(8, 8), similarity=0.85)
            and self.appear(INN_SPICE_CHECK, offset=(8, 8), similarity=0.85)
        )

    def leave_inn(self, timeout=45):
        """Click return once, await the question, then confirm city arrival.

        Never issue BACK during the delayed question or city loading screen.
        A timeout leaves the current screen untouched for inspection.
        """
        deadline = time.time() + timeout
        requested = False
        confirmed = False
        main_frames = 0
        while time.time() < deadline:
            self.device.screenshot()
            if getattr(self, 'handle_network_timeout_popup', lambda: False)():
                return False
            if self.is_inn_exit_confirm():
                if not confirmed:
                    target = self.anchored_button(
                        INN_EXIT_CONFIRM_BUTTON, INN_EXIT_CONFIRM_BUTTON,
                        name='INN_LEAVE_CONFIRMED',
                    )
                    self.device.click(target)
                    confirmed = True
                main_frames = 0
            elif self.is_in_inn():
                if not requested and not confirmed:
                    if not self.appear(INN_EXIT_BUTTON, offset=(8, 8), similarity=0.85):
                        return False
                    self.device.click(INN_EXIT_BUTTON)
                    requested = True
                main_frames = 0
            else:
                if getattr(self, 'handle_startup_popup', lambda: False)():
                    main_frames = 0
                    self.device.sleep(0.3)
                    continue
                identify = getattr(self, 'current_navigation_page', None)
                page = identify() if callable(identify) else None
                if page == 'main' or (page is None and self.is_in_main()):
                    main_frames += 1
                    if main_frames >= 2:
                        logger.info('Returned from inn to city main')
                        return True
                else:
                    main_frames = 0
            self.device.sleep(0.4)
        logger.warning('Inn exit timed out while waiting for confirmation or city')
        self.save_abnormal_screenshot('inn_exit_timeout')
        return False

    def is_in_main(self):

        # Exclude special maps before either static or dynamic city checks.
        if (getattr(self, 'is_inn_exit_confirm', lambda: False)()
                or getattr(self, 'is_in_inn', lambda: False)()
                or getattr(self, 'post_station_page', lambda: 'unknown')() != 'unknown'
                or self.appear(GRAND_CANAL_PAGE_CHECK, offset=(6, 6), similarity=0.78)):
            self.log_check('MainCheck', False)
            return False

        chat = self.appear(
            MAIN_CHAT_CHECK,
            offset=(4, 4),
            similarity=0.85
        )
        build = False
        task = False
        if chat:
            build = self.appear(
                MAIN_BUILD_CHECK,
                offset=(4, 4),
                similarity=0.85
            )
            if not build:
                task = self.appear(
                    MAIN_TASK_CHECK,
                    offset=(4, 4),
                    similarity=0.85
                )

        self.log_check('MainBuildCheck', build)
        self.log_check('MainChatCheck', chat)
        self.log_check('MainTaskCheck', task)

        if getattr(self, '_is_canal_overlay', lambda: False)():
            self.log_check('MainCheck', False)
            return False
        result = chat and (build or task)
        if not result:
            result = self._main_navigation_visible()
            if result:
                logger.info('Jiangnan main page recognized by navigation panels')

        self.log_check('MainCheck', result)

        return result

    def is_unobstructed_main(self):
        """要求主界面可操作，而非仅看到遮罩后方的导航按钮。"""
        identify = getattr(self, 'current_navigation_page', None)
        if callable(identify):
            return identify() == 'main'
        return self.is_in_main()

    def ui_goto_main(self, max_back=10, foreground_verified=False):
        """
        尝试从当前普通游戏页面返回城市主界面。

        Returns:
            bool:
                True  - 成功回到主界面
                False - 无法返回
        """

        logger.hr('JIANGNAN GOTO MAIN')

        back_count = 0
        page_checks = 0
        while page_checks < max_back * 5 + 30:
            page_checks += 1

            # 每一轮只截一次图
            self.device.screenshot()

            station = getattr(self, 'post_station_page', lambda: 'unknown')()
            if station != 'unknown':
                return self.leave_post_station()

            # 先清理覆盖在主界面按钮之上的模态弹窗，尤其是居民事件
            # 面板；否则后方导航按钮会让 is_in_main() 误判为 True。
            if self.handle_startup_popup():
                logger.info('Popup handled while returning to main')
                login_deadline = getattr(self, '_login_deadline', 0)
                wait_main = getattr(self, 'wait_until_main', None)
                if login_deadline and callable(wait_main):
                    return wait_main(timeout=max(1, login_deadline - time.time()))
                self.device.sleep(0.2)
                continue

            identify = getattr(self, 'current_navigation_page', None)
            page = identify() if callable(identify) else None

            # 已经在主界面，不做任何点击
            if page == 'main' or (
                page is None and self.is_in_main()
            ):
                logger.info(
                    f'Arrived at main, back_count={back_count}'
                )
                return True

            # Recognized game pages already prove foreground. For an unknown
            # frame, reuse ensure_ready's verification until the next BACK.
            if page in ('unknown', None) and not foreground_verified:
                if not self.app_is_running_recoverable():
                    logger.warning('Jiangnan is not running')
                    return False
                foreground_verified = True

            # 已经达到最大返回次数
            if back_count >= max_back:
                break

            if page in ('inn', 'inn_exit_confirm'):
                return self.leave_inn()

            if page == 'grand_canal' and self.appear(
                GRAND_CANAL_EXIT_BUTTON, offset=(6, 6), similarity=0.75,
            ):
                logger.info('Leave Grand Canal before returning to city main')
                self.device.click(GRAND_CANAL_EXIT_BUTTON)
                # 离开大运河会重新加载城市。加载页不响应页面识别，
                # 此时连续 BACK 会直接把游戏退出到模拟器桌面。
                for _ in range(45):
                    self.device.sleep(1.0)
                    self.device.screenshot()
                    if self.handle_startup_popup():
                        logger.info(
                            'Popup handled after leaving Grand Canal'
                        )
                        continue
                    loaded_page = identify() if callable(identify) else None
                    if loaded_page == 'main' or (
                        loaded_page is None and self.is_in_main()
                    ):
                        logger.info(
                            f'Arrived at main after Grand Canal exit, '
                            f'back_count={back_count}'
                        )
                        return True
                    if loaded_page not in ('unknown', 'grand_canal', None):
                        break
                else:
                    logger.warning(
                        'Grand Canal exit stayed on loading or unknown page; '
                        'stop before pressing BACK'
                    )
                    return False
                continue

            logger.info(
                f'Not in main, press BACK '
                f'({back_count + 1}/{max_back})'
            )

            self.device.press_back()
            back_count += 1
            foreground_verified = False
            # Observe the transition instead of sleeping 2s and then making
            # another ~5s foreground query. A known main frame finishes early.
            changed_page = None
            for _ in range(10):
                self.device.sleep(0.2)
                self.device.screenshot()
                if self.handle_startup_popup():
                    changed_page = None
                    continue
                following = identify() if callable(identify) else None
                if following == 'main' or (
                    following is None and self.is_in_main()
                ):
                    logger.info(f'Arrived at main, back_count={back_count}')
                    return True
                if following not in ('unknown', None, page):
                    if following == changed_page:
                        break
                    changed_page = following
                else:
                    changed_page = None

        logger.warning(
            f'Failed to goto main after {back_count} BACK presses'
        )

        return False
