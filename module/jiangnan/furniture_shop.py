"""绍兴家具铺：遍历所有等级和实例，固定使用第一档生产。"""
from datetime import datetime, timedelta
from pathlib import Path
import re
import time

from module.base.button import Button
from module.jiangnan.building import BuildingResultOcr
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.jiangnan.farm_production import FarmTextOcr
from module.jiangnan.production import JiangnanProduction
from module.jiangnan.production_state import (
    production_next_run, production_record_pending,
    production_should_yield_to_dock,
)
from module.logger import logger


class JiangnanShaoxingFurnitureShop(JiangnanProduction):
    TASK = 'JiangnanShaoxingFurnitureShop'
    CITY = '绍兴府'
    STATE_FILE = Path('jiangnan_furniture_shop_state.json')
    # Local mappings keep furniture out of the ordinary warehouse production
    # tasks: its output is coins and has no warehouse threshold or ledger node.
    FACILITY_RESOURCES = {'家具铺': '铜钱'}
    TASK_FACILITIES = {TASK: '家具铺'}
    FIRST_RESULT_AREA = (329, 283, 470, 317)
    # The selector moves vertically as the camera centers different shops.
    # The ordinary production crop ends at y=470 and clips the lower button.
    FARM_NAVIGATION_AREA = (930, 100, 1200, 700)
    NAVIGATION_VERSION = 1

    def __init__(self, config, device=None, task=None, path=None):
        super().__init__(
            config=config, device=device, task=task, path=path,
            facility='家具铺', task_name=self.TASK,
        )
        self._active_level = None
        self._discover_groups = False
        self._discovered_groups = None
        self._workers_exhausted = False

    def _use_special_task(self, city):
        return city == self.CITY and bool(self.config.cross_get(
            f'{self.TASK}.shaoxing_furniture_shop.UseSpecialTask', default=True,
        ))

    def _result_validator(self, building_name):
        def validate(value):
            level, total = self.parse_search_result(value, self.facility)
            return bool(level and total and (
                self._active_level is None or level == self._active_level
            ))
        return validate

    @classmethod
    def _parse_groups(cls, labels):
        groups = {}
        for label in labels:
            level, total = cls.parse_search_result(label, '家具铺')
            if level is not None and total is not None and total > 0:
                if level in groups and groups[level] != total:
                    return None
                groups[level] = total
        return groups

    def _on_building_search_result(self, selected):
        if not self._discover_groups:
            return
        reader = BuildingResultOcr()
        reader.buttons = [self.FIRST_RESULT_AREA] + list(self.BUILDING_OTHER_RESULT_AREAS)
        readings = []
        for attempt in range(2):
            if attempt:
                self.device.sleep(0.3)
                self.device.screenshot()
            if not self.is_building_search_page():
                return False
            raw = reader.ocr(self.device.image)
            readings.append(self._parse_groups(raw if isinstance(raw, list) else [raw]))
        level, total = self.parse_search_result(selected, self.facility)
        if (readings[0] and readings[0] == readings[1]
                and readings[0].get(level) == total):
            # Start with the card already selected by search_building.
            self._discovered_groups = {level: total}
            self._discovered_groups.update(readings[0])
            logger.attr('FurnitureShopGroups', self._discovered_groups)
            return True
        return False

    def _search_group(self, level):
        self._active_level = level
        return self.search_building(
            self.facility, result_validator=self._result_validator(self.facility),
            exact=True,
        )

    @staticmethod
    def _parse_navigation_count(text):
        matched = re.fullmatch(r'\s*(\d+)\s*[/／]\s*(\d+)\s*', str(text))
        if matched:
            index, total = map(int, matched.groups())
            if 1 <= index <= total:
                return index, total
        return None

    def _navigation_state(self):
        buttons = self._find_navigation_buttons()
        matches = []
        # River edges can satisfy the same teal colour/size filter. Validate
        # button pairs and their counter instead of requiring exactly two
        # candidates or treating the lowest coloured patch as "next".
        for position, upper_button in enumerate(buttons):
            upper = upper_button.button
            for lower_button in buttons[position + 1:]:
                lower = lower_button.button
                if (abs(upper[0] - lower[0]) > 15
                        or abs(upper[2] - lower[2]) > 15
                        or abs((upper[3] - upper[1]) - (lower[3] - lower[1])) > 10
                        or not 40 <= lower[1] - upper[1] <= 100):
                    continue
                count_area = (upper[0], max(0, upper[1] - 72), upper[2], upper[1] - 7)
                text = FarmTextOcr(count_area, 'FURNITURE_INSTANCE_COUNT').ocr(self.device.image)
                count = self._parse_navigation_count(text)
                if count:
                    matches.append((count[0], count[1], lower_button))
        return matches[0] if len(matches) == 1 else None

    def _wait_navigation(self, index=None, total=None, timeout=5):
        """Confirm the actual counter twice; never infer it from click count."""
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() < deadline:
            self.device.screenshot()
            found = self._navigation_state()
            if found and (index is None or found[0] == index) and (
                total is None or found[1] == total
            ):
                if previous == found[:2]:
                    return found
                previous = found[:2]
            else:
                previous = None
            self.device.sleep(0.3)
        logger.attr('FurnitureNavigationCandidates', [button.button for button in self._find_navigation_buttons()])
        self.save_abnormal_screenshot('furniture_selector_unconfirmed')
        return None

    def _select_production_instance(self, facility, index):
        expected_total = self.data.get('groups', {}).get(
            str(self._active_level), {},
        ).get('total')
        # A lone shop is centered directly by search and has no previous/next
        # panel. Its validated result already proves the level and count.
        if expected_total == 1 and index == 1:
            selected = self.parse_search_result(
                getattr(self, 'last_search_result_name', ''), facility,
            )
            if selected == (self._active_level, 1):
                logger.attr('FurnitureShopInstance', f'{self._active_level}级 1/1 (single search result)')
                return True
            return False
        for attempt in range(2):
            if attempt and not self._search_group(self._active_level):
                return False
            found = self._wait_navigation(total=expected_total)
            if not found or found[0] > index:
                continue
            while found[0] < index:
                current, total, button = found
                self.device.click(Button(
                    area=(), color=(), button=button.button,
                    name='FURNITURE_NEXT_CONFIRMED',
                ))
                found = self._wait_navigation(index=current + 1, total=total)
                if found is None:
                    logger.warning(
                        f'Furniture selector did not confirm {current + 1}/{total}'
                    )
                    break
            if found and found[0] == index:
                logger.attr('FurnitureShopInstance', f'{self._active_level}级 {index}/{found[1]}')
                return True
        return False

    def _invalidate_unverified_clocks(self):
        if self.data.get('navigation_version') == self.NAVIGATION_VERSION:
            return
        # Old traversal trusted clicks. Re-read non-first instances once;
        # an existing timer could belong to the previous selected shop.
        for group in self.data.get('groups', {}).values():
            for index, record in group.get('buildings', {}).items():
                if int(index) > 1:
                    record['state'] = 'uninspected'
        self.data['navigation_version'] = self.NAVIGATION_VERSION
        self.save()

    def _start_first(self, index):
        """Read slot one, select its title, then reuse production assignees."""
        amount, duration = self._read_option(1)
        if amount is None or duration is None:
            self._close_panel()
            return 'unknown', None, 0
        self.device.click(Button(
            area=(), color=(), button=self.LEVEL_BUTTONS[1],
            name='FURNITURE_PRODUCTION_LEVEL_1',
        ))
        if not self._direct_start(city=self.CITY, expected_resources={}):
            state = 'no_workers' if self.last_no_workers else 'unknown'
            self._workers_exhausted = self._workers_exhausted or self.last_no_workers
            if not self.is_in_main():
                self._close_panel()
            return state, None, 0
        duration = self._actual_started_duration(duration, building_index=index)
        return 'working', duration, amount

    def _inspect_group(self, level, total, inventory_state, prelocated=False):
        key = str(level)
        groups = self.data.setdefault('groups', {})
        old = groups.get(key, {}).get('buildings', {})
        buildings = {
            str(index): dict(old.get(str(index), {
                'state': 'uninspected', 'index': index, 'level': level,
            }), total=total)
            for index in range(1, total + 1)
        }
        groups[key] = {'total': total, 'buildings': buildings}
        self._active_level = level
        for index in range(1, total + 1):
            record = buildings[str(index)]
            if production_record_pending(record):
                continue
            if production_should_yield_to_dock(inventory_state):
                inventory_state.sync_dock_schedule()
                self.save()
                return False
            # Pending clocks can skip instance one. The initial search still
            # positions this level's selector, ready to advance to the first
            # due instance without opening the search page again.
            if not prelocated:
                if not self._search_group(level):
                    self.save()
                    return False
            if not self._select_production_instance(self.facility, index):
                record.update(state='selection_failed', finish_at=None)
                self.save()
                return False
            prelocated = False
            state, duration = self._open_current_production()
            if state is None and getattr(self, '_foundation_icon_occluded', False):
                state, duration = 'occluded', timedelta(hours=12)
            state, duration = self._confirm_live_production_state(
                record, state, duration, f'{self.CITY} {level}级家具铺 {index}/{total}',
            )
            amount = int(record.get('amount') or 0)
            if state == 'idle':
                if self._workers_exhausted:
                    state, duration, amount = 'no_workers', None, 0
                    self._close_panel()
                else:
                    state, duration, amount = self._start_first(index)
            elif state in ('working', 'upgrading'):
                self._close_panel()
            elif state != 'occluded' and not self.is_in_main():
                self._close_panel()
            now = datetime.now().replace(microsecond=0)
            finish = (now + duration if duration and state in (
                'working', 'upgrading', 'occluded',
            ) else None)
            buildings[str(index)] = {
                'level': level, 'index': index, 'total': total,
                'state': state or 'unknown', 'amount': amount,
                'finish_at': self._time_text(finish) if finish else None,
                'updated_at': self._time_text(now),
            }
            # Persist each started building before touching the next one.
            self.save()
        return True

    def _finish(self, fallback, status=None, retry=False):
        now = datetime.now().replace(microsecond=0)
        times, lines = [], []
        labels = {
            'working': '生产中', 'upgrading': '升级中',
            'occluded': '地基图标遮挡', 'no_workers': '人物不足，等待复查',
            'uninspected': '尚未检查',
            'selection_failed': '建筑切换失败，等待重试',
        }
        for level, group in self.data.get('groups', {}).items():
            buildings = group.get('buildings', {})
            times.append(production_next_run(buildings, now, self.RETRY, fallback))
            for index, record in buildings.items():
                state = labels.get(record.get('state'), '状态识别失败，等待复查')
                finish = (record.get('finish_at') if record.get('state') in (
                    'working', 'upgrading', 'occluded',
                ) else None) or '等待检查'
                lines.append(f'[{level}级家具铺{index}/{group["total"]}][{state}][{finish}]')
        if retry:
            times.append(now + self.RETRY)
        self.config.modified[f'{self.TASK}.shaoxing_furniture_shop.Status'] = (
            '\n'.join(([status] if status else []) + lines) or '未找到家具铺'
        )
        self.config.task_delay(target=min(times) if times else fallback)
        return True

    def run(self):
        logger.hr('JIANGNAN SHAOXING FURNITURE SHOP', level=0)
        now = datetime.now().replace(microsecond=0)
        fallback = now + self.NO_FARM_RETRY
        inventory_state = JiangnanInventoryState(self.config)
        if production_should_yield_to_dock(inventory_state):
            inventory_state.sync_dock_schedule()
            return self._finish(fallback, '等待码头收货后复查', retry=True)
        self._active_level = None
        self._discover_groups = True
        self._discovered_groups = None
        self._workers_exhausted = False
        self._invalidate_unverified_clocks()
        if not self.switch_city(self.CITY):
            return self._finish(fallback, '无法进入绍兴府，等待复查', retry=True)
        self.handle_building_map_interference()
        try:
            located = self._search_group(None)
        finally:
            self._discover_groups = False
        if not located:
            if self.building_search_confirmed_missing():
                self.data['groups'] = {}
                self.save()
                return self._finish(fallback, '绍兴府无家具铺，明日复查')
            return self._finish(fallback, '家具铺搜索失败，等待复查', retry=True)
        if not self._discovered_groups:
            return self._finish(fallback, '家具铺等级/数量识别不一致，等待复查', retry=True)
        # Remove vanished levels only after two identical complete-page reads.
        self.data['groups'] = {
            str(level): {
                'total': total,
                'buildings': {
                    str(index): dict(self.data.get('groups', {}).get(str(level), {})
                                     .get('buildings', {}).get(str(index), {
                                         'state': 'uninspected', 'index': index,
                                         'level': level,
                                     }), total=total)
                    for index in range(1, total + 1)
                },
            }
            for level, total in self._discovered_groups.items()
        }
        complete = True
        for position, (level, total) in enumerate(self._discovered_groups.items()):
            if not self._inspect_group(
                level, total, inventory_state, prelocated=position == 0,
            ):
                complete = False
                if production_should_yield_to_dock(inventory_state):
                    break
        self.save()
        return self._finish(fallback, retry=not complete)
