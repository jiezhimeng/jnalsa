from datetime import datetime, timedelta

from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.building import JiangnanBuilding
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.logger import logger


class JiangnanTaxHarvest(JiangnanBuilding):
    """逐城收取税课司并固定开启一级四小时工作。"""

    TASK = 'JiangnanTaxHarvest'
    LEVEL_ONE_DURATION = timedelta(hours=4)

    @staticmethod
    def _parse_time(value):
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return datetime(2020, 1, 1)

    def _next_tax_run(self, now=None):
        """根据本次税课司结果计算下一次检查时间，不写配置。"""
        if now is None:
            now = datetime.now().replace(microsecond=0)

        if self.last_tax_action == 'started' and self.last_tax_level == 1:
            started_at = getattr(self, 'last_tax_started_at', None)
            if not isinstance(started_at, datetime):
                started_at = now
            finish_at = started_at + self.LEVEL_ONE_DURATION
            logger.info(f'Tax fixed four-hour finish: {finish_at}')
            return finish_at if finish_at > now else now + timedelta(seconds=30)

        remaining = self.last_tax_remaining
        observed_at = self.last_tax_remaining_at

        if (
            type(remaining) is int
            and remaining >= 0
            and isinstance(observed_at, datetime)
        ):
            # 倒计时对应的是读取时刻，
            # 不是处理完所有城市的时刻。
            finish_at = observed_at + timedelta(seconds=remaining)

            # 到期后留 30 秒缓冲，避免游戏倒计时与本机时间
            # 存在几秒误差，导致过早进入。
            next_run = max(
                finish_at + timedelta(seconds=30),
                now + timedelta(seconds=30),
            )

            logger.info(
                'Tax next run from OCR: %s -> %s'
                % (finish_at, next_run)
            )
            return next_run

        logger.warning(
            'Tax remaining time unknown; recheck in 10 minutes'
        )
        return now + timedelta(minutes=10)

    def _disable_tax_city(self, city_key, city):
        """已确认没有税课司时，复用通用建筑配置关闭逻辑。"""
        return self.disable_city_building_feature(
            task=self.TASK,
            group_prefix='tax',
            city_key=city_key,
            city=city,
            building_name='税课司',
            next_run_field='NextRun',
            next_run_value=datetime(2020, 1, 1),
        )

    def _search_tax_office(self):
        """税课司直接复用所有建筑统一的“扫空当前屏幕后再重搜”清场。"""
        return self.search_building('税课司', exact=True)

    def run(self):
        logger.hr('JIANGNAN TAX HARVEST', level=0)
        now = datetime.now().replace(microsecond=0)
        next_times = []
        due = []
        for city_key, city in CITY_KEY_TO_NAME.items():
            enabled = bool(getattr(self.config, f'tax_{city_key}_Enable'))
            if not enabled:
                continue
            next_run = self._parse_time(
                getattr(self.config, f'tax_{city_key}_NextRun')
            )
            if next_run <= now:
                due.append((city_key, city))
            else:
                next_times.append(next_run)

        logger.attr('TaxHarvestDueCities', [city for _, city in due])
        inventory_state = JiangnanInventoryState(self.config)
        # 到期只代表“理论上可收”，不再直接改确认库存。税课司可能在
        # 地图上收走生产结果；真实入账由后续设施状态确认或仓库实扫完成。
        due_nodes = inventory_state.apply_due_arrivals(now)
        if due_nodes:
            logger.info(
                f'Observed {due_nodes} due inventory nodes before tax harvest'
            )
        for city_key, city in due:
            if inventory_state.due_dock_shipments():
                logger.info('Yield tax harvest to arrived dock shipment')
                inventory_state.sync_dock_schedule()
                break
            if not self.switch_city(city):
                next_run = (
                    datetime.now().replace(microsecond=0)
                    + timedelta(minutes=10)
                )
            elif not self._search_tax_office():
                if self.building_search_confirmed_missing():
                    self._disable_tax_city(city_key, city)
                    continue

                # 输入失败、OCR 不匹配、页面超时等情况，
                # 不代表这座城市没有税课司。
                logger.warning(
                    f'Tax search failed in {city}; '
                    f'status={self.last_search_status}, '
                    'retry in 10 minutes'
                )
                next_run = (
                    datetime.now().replace(microsecond=0)
                    + timedelta(minutes=10)
                )
            elif not self.collect_tax():
                next_run = (
                    datetime.now().replace(microsecond=0)
                    + timedelta(minutes=10)
                )
            else:
                next_run = self._next_tax_run()
            self.config.modified[
                f'{self.TASK}.tax_{city_key}.NextRun'
            ] = next_run
            self.config.update()
            next_times.append(next_run)
            logger.attr(f'TaxNextRun_{city}', next_run)

        if next_times:
            self.config.task_delay(target=min(next_times))
        else:
            self.config.task_delay(success=True)
        return True
