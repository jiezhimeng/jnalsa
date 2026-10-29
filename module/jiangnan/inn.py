"""Yingtian inn navigation, cooking, diners and daily keeper clicks."""
import time

import cv2

from module.base.button import Button
from module.jiangnan.building import JiangnanBuilding
from module.jiangnan.foundation_icons import find_foundation_icon
from module.jiangnan.inn_map import locate_inn_view, save_map_observation
from module.jiangnan.inn_workflow import InnFeatureResult, InnWorkflow
from module.jiangnan.inn_goods import InnGoodsCollector
from module.jiangnan.inn_diners import InnDiners
from module.logger import logger


class JiangnanYingtianInn(JiangnanBuilding):
    TASK = 'JiangnanYingtianInn'
    GROUP = 'yingtian_inn'
    CITY = '应天府'
    BUILDING = '客栈'

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device, task=self.TASK)

    def _find_inn_entrance(self):
        """Locate the inn's foundation building icon below the search arrow."""
        arrow = self.find_building_search_arrow(report_missing=False)
        match = find_foundation_icon(self.device.image, arrow, self.BUILDING)
        if match is None:
            return None
        x, y, score, _ = match
        logger.attr('InnBuildingIcon', (x, y, round(score, 3)))
        return Button(area=(), color=(), button=(x-7, y-7, x+7, y+7),
                      name='INN_ENTRANCE_DYNAMIC')

    def enter_inn(self, timeout=45):
        self.device.screenshot()
        if self.is_inn_exit_confirm():
            if not self.leave_inn():
                return False
        elif self.is_in_inn():
            return True
        if not self.switch_city(self.CITY):
            return False
        if not self.search_building(self.BUILDING, exact=True, handle_interference=False):
            return False
        self.device.screenshot()
        target = self._find_inn_entrance()
        if target is None:
            self.save_abnormal_screenshot('inn_entrance_missing')
            return False
        self._click_searched_building(target)
        deadline = time.time() + timeout
        stable = 0
        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            stable = stable + 1 if (not self.is_inn_exit_confirm() and self.is_in_inn()) else 0
            if stable >= 2:
                logger.info('Yingtian inn entered')
                return True
        self.save_abnormal_screenshot('inn_enter_timeout')
        return False

    def run(self):
        logger.hr('JIANGNAN YINGTIAN INN', level=0)
        action = self.config.cross_get(f'{self.TASK}.{self.GROUP}.Action', default='auto')
        if action == 'auto':
            return InnWorkflow(self).run()
        if action not in ('enter_exit', 'enter', 'exit'):
            raise ValueError('Invalid inn action: %r' % action)
        if action == 'exit':
            success = self.leave_inn()
        else:
            success = self.enter_inn()
            if success:
                self.update_map_location()
            if success and action == 'enter_exit':
                success = self.leave_inn()
        status = ('[已进入客栈]' if action == 'enter' else '[已返回城市主界面]') if success else '[进出客栈未完成，请查看截图]'
        self.config.modified[f'{self.TASK}.{self.GROUP}.Status'] = status
        self.config.update()
        self.config.task_delay(success=success)
        return success

    def run_cooking_sales(self, now):
        """烹饪预留入口。保留旧方法名，兼容现有调度和调试调用。

        这里只负责后续的做菜、上菜；食客物资交易已拆到 run_diners。
        烹饪界面尚未提供，不能把进入客栈当成已完成烹饪，也不能自动点菜。
        """
        logger.info('Inn cooking: awaiting cooking gameplay UI')
        return InnFeatureResult('pending', '[烹饪框架已就绪][待接入做菜与上菜识别]')

    def run_diners(self, now):
        return InnDiners(self).run(now)

    def run_goods_collection(self, now):
        return InnGoodsCollector(self).run(now)

    def click_inn_keeper(self, area):
        """仅点击两帧均确认的老板娘头上气泡，area 不是人物坐标。"""
        self.device.click(Button(area=(), color=(), button=area, name='INN_KEEPER_BUBBLE_CLICK'))

    def click_inn_reward(self, area):
        """仅供补天石奖励窗的“收下”按钮经过连续两帧确认后调用。"""
        self.device.click(Button(area=(), color=(), button=area, name='INN_KEEPER_REWARD_ACCEPT'))

    def update_map_location(self):
        """Record the last confirmed inn frame without touching the camera."""
        config_name = self.config.config_name
        location = None
        reason = ''
        try:
            self.device.screenshot()
            if self.is_inn_exit_confirm() or not self.is_in_inn():
                reason = '当前画面不是无遮挡的客栈主界面'
            else:
                location = locate_inn_view(self.device.image)
                if location is None:
                    reason = '画面特征不足，未确认地图位置'
        except (OSError, ValueError, cv2.error) as exc:
            reason = '客栈地图读取或匹配失败'
            logger.warning('Inn panorama registration failed: %s' % exc)
        try:
            save_map_observation(config_name, location, reason)
        except OSError as exc:
            # Map diagnostics are optional; never prevent the confirmed exit.
            logger.warning('Unable to save inn map observation: %s' % exc)
        if location is not None:
            logger.attr('InnMapCenter', tuple(round(v, 1) for v in location['center']))
        return location
