from datetime import datetime, timedelta

from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.inventory_state import (
    ALL_RESOURCES,
    resources_for_city,
    DEFAULT_TIME,
    JiangnanInventoryState,
)
from module.jiangnan.warehouse import JiangnanWarehouse
from module.logger import logger


class JiangnanWarehouseInventory(JiangnanWarehouse):
    """定时实扫仓库，并把结果同步到“仓库库存”配置页。"""

    def enabled_cities(self):
        cities = []
        for city_key, city_name in CITY_KEY_TO_NAME.items():
            enabled = self.config.cross_get(
                keys=(
                    f'JiangnanWarehouseInventory.'
                    f'warehouse_{city_key}.Scan'
                ),
                default=True,
            )
            if bool(enabled):
                cities.append(city_name)
        return cities

    @staticmethod
    def _city_key(city_name):
        for city_key, name in CITY_KEY_TO_NAME.items():
            if name == city_name:
                return city_key
        raise KeyError(city_name)

    def city_next_run(self, city_name):
        city_key = self._city_key(city_name)
        value = self.config.cross_get(
            keys=(
                f'JiangnanWarehouseInventory.'
                f'warehouse_{city_key}.NextRun'
            ),
            default=DEFAULT_TIME,
        )
        return JiangnanInventoryState._parse_time(value) or DEFAULT_TIME

    def due_cities(self, cities, now=None):
        now = (now or datetime.now()).replace(microsecond=0)
        return [city for city in cities if self.city_next_run(city) <= now]

    def reset_cities(self, cities):
        """返回被用户明确重置到初始时间的城市。

        ``2020-01-01`` 是逐城页面的“立即重新识别”哨兵值。
        只扫描用户明确重置的城市，不因库存不确定而扩大全城扫描。
        """
        return [
            city for city in cities
            if self.city_next_run(city) <= DEFAULT_TIME
        ]

    def sync_schedule(self, scanned_cities=(), scanned_next_run=None):
        """更新逐城时间，并用最近的逐城时间驱动页面总倒计时。"""
        enabled = self.enabled_cities()
        scanned = set(scanned_cities)
        times = []
        for city in enabled:
            city_key = self._city_key(city)
            if city in scanned and scanned_next_run is not None:
                next_run = scanned_next_run
                self.config.modified[
                    f'JiangnanWarehouseInventory.'
                    f'warehouse_{city_key}.NextRun'
                ] = next_run
            else:
                next_run = self.city_next_run(city)
            times.append(next_run)
        if times:
            self.config.task_delay(target=min(times))
        return min(times) if times else None

    def run_scan(self, cities=None):
        cities = list(cities or self.enabled_cities())
        if not cities:
            logger.warning('No Jiangnan cities are enabled for warehouse scan')
            return {}, None

        state = JiangnanInventoryState(self.config)
        inventories = {}
        city_next_runs = {}

        def persist_city(city, inventory, capacity, special_capacity):
            # 每城实扫后立即写入 GUI。
            # 即使后续城市扫描被中断，已完成的数据也不会继续显示旧值。
            if inventory is None:
                inventory = {resource: -1 for resource in resources_for_city(city)}
            else:
                inventory = dict(inventory)
                for resource in resources_for_city(city):
                    inventory.setdefault(resource, -1)
            inventories[city] = inventory
            capacities = (
                {city: int(capacity)}
                if capacity is not None and int(capacity) > 0
                else {}
            )
            special_capacities = (
                {city: dict(special_capacity)}
                if special_capacity else {}
            )
            next_run = state.record_scan(
                {city: inventory},
                capacities,
                special_capacities,
            )
            # A running scheduler may import this task after inventory_state
            # was already loaded. Older record_scan implementations don't
            # persist city clocks. Keep that combination safe without adding
            # another write when the new implementation already saved it.
            city_key = self._city_key(city)
            next_key = f'JiangnanWarehouseInventory.warehouse_{city_key}.NextRun'
            current = (self.config.modified[next_key] if next_key in self.config.modified
                       else self.config.cross_get(keys=next_key, default=None))
            if JiangnanInventoryState._parse_time(current) != next_run:
                self.config.modified[next_key] = next_run
                self.config.update()
            city_next_runs[city] = next_run
            logger.attr(f'WarehouseInventorySaved_{city}', len(inventory))

        scanned = self.scan_inventory_by_city(
            cities=cities,
            on_city_scanned=persist_city,
            # Each unreadable cell already has local OCR fallbacks. Revisiting
            # the entire warehouse immediately repeats the same failed crops.
            # Keep -1 explicit and retry on the normal city schedule instead.
            max_attempts=1,
        )
        # 扫描流程只会回调成功进入仓库的城市。切城失败时保留 GUI 中
        # 上一次成功库存，并等下一次正常定时扫描，不能用整页
        # -1 覆盖已有数据。
        for city in cities:
            if city not in inventories:
                retry_at = datetime.now().replace(microsecond=0) + timedelta(
                    hours=state.scan_interval_hours()
                )
                city_key = self._city_key(city)
                self.config.modified[
                    f'JiangnanWarehouseInventory.'
                    f'warehouse_{city_key}.NextRun'
                ] = retry_at
                self.config.update()
                city_next_runs[city] = retry_at
                logger.warning(
                    f'Warehouse inventory kept unchanged after failed scan: '
                    f'{city}, next periodic scan={retry_at}'
                )

        # 实扫结果是库存基线；record_scan 已把被本轮实扫覆盖的
        # 到期运输标为入账，下面只补记未扫到城市的到期运输。
        state.apply_due_arrivals()
        next_run = min(city_next_runs.values()) if city_next_runs else None
        return inventories, next_run

    def run(self):
        logger.hr('JIANGNAN WAREHOUSE INVENTORY', level=0)
        cities = self.enabled_cities()
        if not cities:
            state = JiangnanInventoryState(self.config)
            self.config.task_delay(minute=state.scan_interval_hours() * 60)
            logger.info('Warehouse inventory skipped: all cities disabled')
            return True
        now = datetime.now().replace(microsecond=0)
        state = JiangnanInventoryState(self.config)
        reset_cities = self.reset_cities(cities)
        if reset_cities:
            # UI 把某一城改回 2020 年表示用户点名要求重扫该城。
            due_cities = reset_cities
            logger.attr('WarehouseInventoryResetCities', reset_cities)
        else:
            # Event uncertainty never adds an extra warehouse visit.
            due_cities = self.due_cities(cities, now=now)
        logger.attr('WarehouseInventoryDueCities', due_cities)
        if not due_cities:
            next_run = self.sync_schedule()
            logger.attr('WarehouseInventoryNextRun', next_run)
            return True

        inventories, scanned_next_run = self.run_scan(cities=due_cities)
        if scanned_next_run is None:
            retry_at = now + timedelta(hours=state.scan_interval_hours())
            next_run = self.sync_schedule(due_cities, retry_at)
            logger.attr('WarehouseInventoryNextRun', next_run)
            return False
        # run_scan 已逐城写入各自的时间；这里只汇总页面总时间。
        next_run = self.sync_schedule()
        logger.attr('WarehouseInventoryCities', len(inventories))
        logger.attr('WarehouseInventoryNextRun', next_run)
        return True
