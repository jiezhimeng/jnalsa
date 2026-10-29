"""Limited-time corn production, with a separate clock for each supported city."""
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import re
import time

import cv2
from filelock import FileLock

from module.base.button import Button
from module.base.utils import load_image
from module.jiangnan.building import BuildingResultOcr
from module.jiangnan.farm_production import FarmTextOcr
from module.jiangnan.production import JiangnanProduction
from module.jiangnan.production_state import cached_working_group
from module.logger import logger


@lru_cache(maxsize=1)
def _corn_working_banner():
    path = Path(__file__).resolve().parents[2] / 'assets/jiangnan/corn_drying_mat/working_banner.png'
    return cv2.cvtColor(load_image(str(path)), cv2.COLOR_RGB2GRAY)


class CornProductionInventory:
    """Corn is event currency; only its recipe inputs belong to city stock."""

    def __init__(self, inventory):
        self.inventory = inventory

    def __getattr__(self, name):
        return getattr(self.inventory, name)

    def production_below_threshold(self, city, resource):
        if resource == '玉米':
            return True
        return self.inventory.production_below_threshold(city, resource)

    def production_threshold_states(self, city, resources):
        ordinary = tuple(resource for resource in resources if resource != '玉米')
        states = self.inventory.production_threshold_states(city, ordinary) if ordinary else {}
        if '玉米' in resources:
            states['玉米'] = True
        return states

    def schedule_inventory_change(self, city, resource, *args, **kwargs):
        if resource != '玉米':
            return self.inventory.schedule_inventory_change(city, resource, *args, **kwargs)

    def reconcile_production_harvest(self, **kwargs):
        if kwargs.get('resource') != '玉米':
            return self.inventory.reconcile_production_harvest(**kwargs)


class JiangnanCornDryingMat(JiangnanProduction):
    TASK = 'JiangnanCornDryingMat'
    STATE_FILE = Path('jiangnan_corn_drying_mat_state.json')
    PRODUCTION_CITIES = {'yingtianfu': '应天府', 'suzhoufu': '苏州府'}
    TASK_FACILITIES = {TASK: '玉米晒簟'}
    FACILITY_RESOURCES = {'玉米晒簟': '玉米'}
    FACILITY_INPUT_RESOURCES = {'玉米晒簟': ('菜肴',)}
    COIN_COST_FACILITIES = {'玉米晒簟'}
    GROUP_VERSION = 3
    FARM_NAVIGATION_AREA = (930, 100, 1200, 700)
    DEFAULT_PRIORITY = ('many', 'rich', 'some', 'little')
    OPTION_NAME = {
        'many': '第三档玉米', 'rich': '第四档玉米',
        'some': '第二档玉米', 'little': '第一档玉米',
    }

    def __init__(self, config, device=None, **kwargs):
        super().__init__(config, device=device, task_name=self.TASK, **kwargs)
        self._active_level = None
        self._discovered_groups = None

    def _requires_group_discovery(self, city):
        return self._city_group(city).get('group_version') != self.GROUP_VERSION

    def _city_group(self, city):
        return (self.data.get('cities', {}).get(city, {})
                .get('facilities', {}).get(self.facility, {}))

    def _result_validator(self, building_name):
        def validate(text):
            level, total = self.parse_search_result(text, building_name)
            active = getattr(self, '_active_level', None)
            return bool(level and total and (active is None or level == active))
        return validate

    def _find_foundation_production_icon(self, arrow):
        # Residents can cover the scissors while the corn production banner
        # remains visible. Match its corn emblem and text near the selected
        # arrow, including the area above a retained arrow hidden by the flag.
        image = self.device.image
        x, tip, _ = arrow
        left, top = max(0, x - 75), max(0, tip - 180)
        right, bottom = min(image.shape[1], x + 75), min(image.shape[0], tip + 140)
        template = _corn_working_banner()
        crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        if crop.shape[0] >= template.shape[0] and crop.shape[1] >= template.shape[1]:
            scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(scores)
            if score >= 0.90:
                center_x = left + point[0] + template.shape[1] // 2
                center_y = top + point[1] + template.shape[0] // 2
                logger.attr('CornWorkingBanner', f'{score:.3f} at ({center_x}, {center_y})')
                return Button(area=(), color=(), button=(center_x - 5, center_y - 5,
                    center_x + 5, center_y + 5), name='CORN_WORKING_BANNER')
        return super()._find_foundation_production_icon(arrow)

    def _find_centered_production(self):
        # The animated search arrow can cover the production flag while
        # residents cover the scissors. Wait for the arrow to fade before
        # declaring this selected building inaccessible.
        self._foundation_icon_occluded = False
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            arrow = self.find_building_search_arrow()
            if arrow is None:
                break
            target = self._find_foundation_production_icon(arrow)
            if target is not None:
                return target
            self.device.sleep(0.5)
            self.device.screenshot()
        return super()._find_centered_production()

    def _on_building_search_result(self, selected):
        """Read every level card twice, including searches repeated after harvest."""
        reader = BuildingResultOcr()
        reader.buttons = [(329, 283, 486, 317)] + [
            (x1, y1, min(x2 + 15, 1104), y2)
            for x1, y1, x2, y2 in self.BUILDING_OTHER_RESULT_AREAS
        ]
        readings = []
        for attempt in range(2):
            if attempt:
                self.device.sleep(0.3)
                self.device.screenshot()
            if not self.is_building_search_page():
                logger.warning('Corn search page changed during group confirmation; retry later')
                return False
            raw = reader.ocr(self.device.image)
            groups = {}
            for label in raw if isinstance(raw, list) else [raw]:
                level, total = self.parse_search_result(label, self.facility)
                if level and total:
                    if level in groups and groups[level] != total:
                        return False
                    groups[level] = total
            readings.append(groups)
        level, _ = self.parse_search_result(selected, self.facility)
        if not readings[0] or readings[0] != readings[1] or level not in readings[0]:
            logger.warning('Corn level/count readings disagree; keep all existing records')
            return False
        if getattr(self, '_active_level', None) is None:
            # Selected card is processed first. A harvest may complete an
            # upgrade, so the final recentered search replaces earlier groups.
            self._discovered_groups = {level: readings[0][level]}
            self._discovered_groups.update(readings[0])
        elif readings[0] != self._discovered_groups:
            self._groups_changed = True
            logger.warning('Corn groups changed while traversing; rediscover before reuse')
            return False
        self._selected_group = (level, readings[0][level])
        # Search's narrow first-card OCR can omit the trailing count. Keep
        # the full-card count for the map click and subsequent group parsing.
        self.last_search_result_name = f'{level}级{self.facility}({readings[0][level]})'
        logger.attr('CornProductionGroups', readings[0])
        return True

    def _navigation_state(self):
        buttons = self._find_navigation_buttons()
        matches = []
        for position, upper_button in enumerate(buttons):
            upper = upper_button.button
            for lower_button in buttons[position + 1:]:
                lower = lower_button.button
                if (abs(upper[0] - lower[0]) > 15 or abs(upper[2] - lower[2]) > 15
                        or not 40 <= lower[1] - upper[1] <= 100):
                    continue
                area = (upper[0], max(0, upper[1] - 72), upper[2], upper[1] - 7)
                text = FarmTextOcr(area, 'CORN_INSTANCE_COUNT').ocr(self.device.image)
                count = re.fullmatch(r'\s*(\d+)\s*[/／]\s*(\d+)\s*', str(text))
                if count:
                    index, total = map(int, count.groups())
                    if 1 <= index <= total:
                        matches.append((index, total, lower_button))
        return matches[0] if len(matches) == 1 else None

    def _wait_navigation(self, index=None, timeout=5):
        deadline = time.monotonic() + timeout
        previous = None
        expected = self._discovered_groups[self._active_level]
        while time.monotonic() < deadline:
            self.device.screenshot()
            found = self._navigation_state()
            if found and found[1] == expected and (index is None or found[0] == index):
                if previous == found[:2]:
                    return found
                previous = found[:2]
            else:
                previous = None
            self.device.sleep(0.3)
        logger.warning(f'Corn instance counter not confirmed: level={self._active_level}, index={index}')
        return None

    def _select_production_instance(self, facility, index):
        expected = self._discovered_groups[self._active_level]
        if expected == 1 and index == 1:
            return self._selected_group == (self._active_level, 1)
        for attempt in range(2):
            if attempt and not self.search_building(
                facility, result_validator=self._result_validator(facility), exact=True,
            ):
                return False
            found = self._wait_navigation()
            if not found or found[0] > index:
                continue
            while found[0] < index:
                current, _, button = found
                self.device.click(Button(area=(), color=(), button=button.button, name='CORN_NEXT_CONFIRMED'))
                found = self._wait_navigation(index=current + 1)
                if found is None:
                    break
            if found and found[0] == index:
                # The next-instance button moved the map. Refresh the retained
                # frame now so a faded arrow still anchors this new building.
                self._remember_centered_search_arrow()
                logger.attr('CornProductionInstance', f'{self._active_level}级 {index}/{expected}')
                return True
        return False

    @staticmethod
    def _flatten_groups(groups):
        """Give the scheduler one complete city set, including uninspected slots."""
        result = {}
        total = sum(group['total'] for group in groups.values())
        for level, group in groups.items():
            for index in range(1, group['total'] + 1):
                record = dict(group['buildings'].get(str(index), {'state': 'uninspected'}))
                position = len(result) + 1
                record.update(level=int(level), index=position, total=total,
                              group_index=index, group_total=group['total'])
                result[str(position)] = record
        return result

    def _write_groups(self, city, groups=None, buildings=None):
        with FileLock(str(self.path) + '.lock', timeout=10):
            latest = self._load_state()
            data = (latest.setdefault('cities', {}).setdefault(city, {})
                    .setdefault('facilities', {}).setdefault(self.facility, {}))
            if groups is not None:
                data['groups'] = groups
                data['group_version'] = self.GROUP_VERSION
            if buildings is not None:
                data['groups'][str(self._active_level)]['buildings'] = buildings
            if getattr(self, '_groups_changed', False):
                data['group_version'] = 0
            # An obscured event building has no confirmed production clock.
            # Retry normally instead of inheriting the generic 12-hour delay.
            for group in data['groups'].values():
                for record in group['buildings'].values():
                    if record.get('state') == 'occluded':
                        record.update(state='unknown', finish_at=None)
            data['unavailable'] = False
            data['buildings'] = self._flatten_groups(data['groups'])
            self.data = latest
            self.save()

    def _save_production_progress(self, city, buildings, facility=None):
        self._write_groups(city, buildings=buildings)

    def _confirm_live_production_state(self, old, state, duration, label):
        if state == 'occluded':
            logger.warning(f'{label} is obscured; retry without a production clock')
            state = 'unknown'
        return super()._confirm_live_production_state(old, state, duration, label)

    def _known_amount(self, city, facility, index, level):
        group = self._city_group(city).get('groups', {}).get(str(level), {})
        return max(0, int(group.get('buildings', {}).get(str(index), {}).get('amount') or 0))

    def _refresh_live_inventory_from_production_panel(self, *args, **kwargs):
        # Event currency has no city warehouse row or warehouse capacity.
        # Recipe quantities are read live by _read_costs and deducted normally.
        return True

    def inspect_city(self, city_key, city, inventory_state, prelocated_result=None):
        if self.PRODUCTION_CITIES.get(city_key) != city:
            raise ValueError(f'Unsupported corn production city: {city!r}')
        inventory = CornProductionInventory(inventory_state)
        self._active_level = None
        self._discovered_groups = None
        self._groups_changed = False
        if not self.search_building(self.facility, result_validator=self._result_validator(self.facility), exact=True):
            return False
        if not self._discovered_groups:
            return False
        old = self._city_group(city)
        rediscover = old.get('group_version') != self.GROUP_VERSION
        previous = old.get('groups', {})
        if not previous:
            # Preserve the first confirmed building from the earlier schema;
            # its clock remains valid while the missing slots are inspected.
            for record in old.get('buildings', {}).values():
                level = str(record['level'])
                group = previous.setdefault(level, {'buildings': {}})
                group['buildings'][str(record['index'])] = dict(record)
        groups = {
            str(level): {'total': total, 'buildings': {
                str(index): dict(previous.get(str(level), {}).get('buildings', {}).get(str(index),
                    {'state': 'uninspected'}), level=level, index=index, total=total)
                for index in range(1, total + 1)
            }} for level, total in self._discovered_groups.items()
        }
        self._write_groups(city, groups=groups)
        complete = True
        try:
            for position, (level, total) in enumerate(self._discovered_groups.items()):
                self._active_level = level
                records = self._city_group(city)['groups'][str(level)]['buildings']
                if not rediscover and cached_working_group(records) is not None:
                    continue
                if position and not self.search_building(
                    self.facility, result_validator=self._result_validator(self.facility), exact=True,
                ):
                    complete = False
                    break
                if not self._select_production_instance(self.facility, 1):
                    complete = False
                    continue
                # Ordinary production inspects one level group. Its local
                # view is temporary; checkpoints always save every group.
                self._city_group(city)['buildings'] = records
                result = super().inspect_facility(
                    city_key, city, self.facility, self.resource, inventory,
                    prelocated_result=f'{level}级{self.facility}({total})',
                    # Starting production can reorder a same-level group.
                    # On an active group, inspect each actual counter instead
                    # of skipping a new position using another building's clock.
                    verify_all_instances=True,
                )
                self._write_groups(city, buildings=result['buildings'] if result else records)
                complete = complete and result is not None
        finally:
            self._active_level = None
            # Restore the aggregate view even after a partially completed group.
            self.data = self._load_state()
        return complete and not self._groups_changed

    def sync_ui(self, fallback=None, update=True):
        now = datetime.now().replace(microsecond=0)
        times = []
        for city_key, city in self.PRODUCTION_CITIES.items():
            if not self.config.cross_get(self._production_key(city_key, 'Enable'), default=True):
                continue
            key = self._production_key(city_key, 'NextRun')
            saved = self._parse_time(self.config.modified.get(
                key, self.config.cross_get(key, default=self.DEFAULT_TIME),
            ))
            group = (self.data.get('cities', {}).get(city, {})
                     .get('facilities', {}).get(self.facility, {}))
            if not group and (saved is None or saved <= now):
                # Yielding before inspection is not a production observation
                # or a failed city switch. Leave the unvisited city due.
                self.config.modified[self._production_key(city_key, 'Status')] = '等待首次检查'
                self.config.modified[key] = saved or self.DEFAULT_TIME
                times.append(now)
                continue
            # An unvisited city or a failed city switch must keep its own wake.
            city_fallback = saved if saved and saved > now else (fallback or now + self.RETRY)
            times.append(self._sync_city_ui(city_key, city, city_fallback))
        if update:
            self.config.update()
        return times

    def _sync_city_ui(self, city_key, city, fallback):
        next_run = super()._sync_city_ui(city_key, city, fallback)
        if self._city_group(city) and self._requires_group_discovery(city):
            # A failed complete-page check must not reuse the old partial
            # group's future clock as if discovery had succeeded.
            next_run = min(next_run, datetime.now().replace(microsecond=0) + self.RETRY)
            self.config.modified[self._production_key(city_key, 'NextRun')] = next_run
        return next_run

    def run(self):
        for city_key, city in self.PRODUCTION_CITIES.items():
            if (self._city_group(city) and self._requires_group_discovery(city)
                    and self.config.cross_get(self._production_key(city_key, 'Enable'), default=True)):
                self.config.modified[self._production_key(city_key, 'NextRun')] = self.DEFAULT_TIME
        if self.config.modified:
            self.config.update()
        if getattr(self.config, '_jiangnan_debug_task', None) == self.TASK:
            # A manual first inspection should also recover the retry clocks
            # written by older versions that exited before visiting a city.
            now = datetime.now()
            for city_key, city in self.PRODUCTION_CITIES.items():
                group = (self.data.get('cities', {}).get(city, {})
                         .get('facilities', {}).get(self.facility, {}))
                key = self._production_key(city_key, 'NextRun')
                saved = self._parse_time(self.config.cross_get(key))
                if not group and saved and saved > now:
                    self.config.modified[key] = self.DEFAULT_TIME
            if self.config.modified:
                self.config.update()
        return super().run()
