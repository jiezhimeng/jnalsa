import time

from module.jiangnan.app import JiangnanApp
from module.jiangnan.assets import (
    MAIN_CHAT_CHECK,
    MAIN_LEVEL_BUTTON,
    PROFILE_CHECK,
    PROFILE_SIGNIN_BUTTON,

    DAILY_SIGNIN_TITLE_CHECK,
    DAILY_SIGNIN_DAILY_CHECK,
    DAILY_SIGNIN_BUTTON,
    DAILY_SIGNIN_DONE_CHECK,

    SIGNIN_DAILY_TAB_BUTTON,
)
from module.logger import logger


class JiangnanSignin(JiangnanApp):

    def is_profile_popup(self):
        """
        判断是否位于左上角等级打开的玩家信息弹窗。
        """

        profile = self.appear(
            PROFILE_CHECK,
            offset=(3, 3),
            similarity=0.90
        )

        signin = self.appear(
            PROFILE_SIGNIN_BUTTON,
            offset=(3, 3),
            similarity=0.90
        )

        logger.attr('ProfileCheck', profile)
        logger.attr('ProfileSigninCheck', signin)

        return profile and signin

    def is_daily_signin_page(self):
        """
        判断是否位于“日常签到”页面。

        左侧“日常”标签会因为活动数量变化而上下移动，
        因此允许较大的纵向偏移。
        """

        title = self.appear(
            DAILY_SIGNIN_TITLE_CHECK,
            offset=(5, 5),
            similarity=0.88
        )

        daily = self.appear(
            DAILY_SIGNIN_DAILY_CHECK,
            offset=(10, 110),
            similarity=0.85
        )

        logger.attr('DailySigninTitleCheck', title)
        logger.attr('DailySigninDailyCheck', daily)

        return title and daily

    def find_daily_tab(self):
        """
        动态寻找左侧“日常”标签。

        有活动和无活动时，“日常”标签的纵向位置不同，
        所以不能使用固定坐标直接点击。
        """

        result = self.appear(
            SIGNIN_DAILY_TAB_BUTTON,
            offset=(10, 110),
            similarity=0.85
        )

        logger.attr(
            'SigninDailyTabCheck',
            result
        )

        return result

    def goto_profile(self, timeout=8):
        """
        主界面 -> 玩家信息弹窗
        """

        logger.hr('JIANGNAN OPEN PROFILE')

        if not self.ensure_ready():
            logger.warning('Unable to prepare Jiangnan main page')
            return False

        logger.info('Click player level')

        self.device.click(
            self.anchored_button(
                MAIN_CHAT_CHECK,
                MAIN_LEVEL_BUTTON,
                name='MAIN_LEVEL_BUTTON_DYNAMIC',
                inset=5,
            )
        )

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            if self.is_profile_popup():
                logger.info('Player profile popup arrived')
                return True

            self.device.sleep(0.5)

        logger.warning('Player profile popup timeout')
        return False

    def goto_daily_signin(self, timeout=10):
        """
        玩家信息弹窗 -> 日常签到页面。

        点击“签到奖励”以后有两种情况：

        1. 直接进入日常签到页
        2. 进入限时签到页
        -> 点击左侧“日常”
        -> 再进入日常签到页
        """

        logger.hr('JIANGNAN GOTO DAILY SIGNIN')

        if not self.goto_profile():
            return False

        logger.info('Click signin reward entry')

        self.device.click(
            PROFILE_SIGNIN_BUTTON
        )

        deadline = time.time() + timeout

        daily_tab_clicked = False

        while time.time() < deadline:

            self.device.screenshot()

            # ========================================
            # 情况 1：
            # 已经直接进入日常签到页
            # ========================================
            if self.is_daily_signin_page():

                logger.info(
                    'Daily signin page arrived'
                )

                return True

            # ========================================
            # 情况 2：
            # 进入了限时签到页
            #
            # 如果看得到左侧“日常”，点击一次
            # ========================================
            if not daily_tab_clicked:

                if self.find_daily_tab():

                    logger.info(
                        'Signin event page detected, '
                        'switch to daily tab'
                    )

                    self.device.click(
                        SIGNIN_DAILY_TAB_BUTTON
                    )

                    daily_tab_clicked = True

                    self.device.sleep(0.8)

                    continue

            self.device.sleep(0.5)

        logger.warning(
            'Daily signin page timeout'
        )

        try:
            self.device.image_save(
                'jiangnan_daily_signin_timeout.png'
            )
        except Exception:
            pass

        return False

    def is_daily_signin_done(self):
        """
        判断当天日常签到是否已经完成。
        """

        done = self.appear(
            DAILY_SIGNIN_DONE_CHECK,
            offset=(5, 5),
            similarity=0.90
        )

        logger.attr(
            'DailySigninDone',
            done
        )

        return done

    def claim_daily_signin(self, timeout=10):
        """
        执行每日7日签到。

        明确区分三种状态：

            红色“签到”
                -> 今天还没签到

            灰色“已签到”
                -> 今天已经签到

            两个都没有
                -> 未知状态，不做猜测
        """

        logger.hr('JIANGNAN DAILY SIGNIN')

        self.device.screenshot()

        # ========================================
        # 必须首先确认还在日常签到页
        # ========================================
        if not self.is_daily_signin_page():

            logger.warning(
                'Not in daily signin page'
            )

            return False

        # ========================================
        # 情况 1：已经签到
        # ========================================
        if self.is_daily_signin_done():

            logger.info(
                'Daily signin already completed today'
            )

            result = self.ui_goto_main()

            logger.attr(
                'DailySigninReturnMain',
                result
            )

            return result

        # ========================================
        # 情况 2：可以签到
        # ========================================
        can_sign = self.appear(
            DAILY_SIGNIN_BUTTON,
            offset=(5, 5),
            similarity=0.90
        )

        logger.attr(
            'DailySigninAvailable',
            can_sign
        )

        if not can_sign:

            # 既没有“签到”
            # 也没有“已签到”
            #
            # 不允许猜测
            logger.warning(
                'Unable to determine daily signin state'
            )

            try:
                self.device.image_save(
                    'jiangnan_daily_signin_unknown.png'
                )
            except Exception:
                pass

            return False

        # ========================================
        # 真正点击签到
        # ========================================
        logger.info(
            'Daily signin available, claim it'
        )

        self.device.click(
            DAILY_SIGNIN_BUTTON
        )

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.sleep(0.5)

            self.device.screenshot()

            # ------------------------------------
            # 有些签到可能弹出通用“收下”
            # ------------------------------------
            if self.handle_reward_popup():

                logger.info(
                    'Daily signin reward popup handled'
                )

                continue

            # ------------------------------------
            # 签到按钮已经变成“已签到”
            # 这是最明确的成功证据
            # ------------------------------------
            if self.is_daily_signin_page():

                if self.is_daily_signin_done():

                    logger.info(
                        'Daily signin confirmed'
                    )

                    result = self.ui_goto_main()

                    logger.attr(
                        'DailySigninReturnMain',
                        result
                    )

                    return result

            # ------------------------------------
            # 某些情况下领取后直接回主页
            # ------------------------------------
            if self.is_in_main():

                logger.info(
                    'Daily signin claimed, '
                    'returned directly to main'
                )

                return True

        logger.warning(
            'Daily signin confirmation timeout'
        )

        try:
            self.device.image_save(
                'jiangnan_daily_signin_claim_timeout.png'
            )
        except Exception:
            pass

        return False

    def run(self):
        """
        Scheduler 任务入口。
        """

        logger.hr(
            'JIANGNAN SIGNIN TASK',
            level=0
        )

        if not self.goto_daily_signin():

            logger.warning(
                'Failed to open daily signin'
            )

            # 失败后较短时间重试
            self.config.task_delay(
                success=False
            )

            return False

        if not self.claim_daily_signin():

            logger.warning(
                'Daily signin failed'
            )

            self.config.task_delay(
                success=False
            )

            return False

        logger.info(
            'Jiangnan signin task completed'
        )

        # 成功以后推迟到下一次每日刷新
        self.config.task_delay(
            server_update=True
        )

        return True
