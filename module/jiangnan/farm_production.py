import json
import re
import time
from datetime import datetime, timedelta
from module.jiangnan.production_state import (
    cached_production_group, production_record_pending, production_next_run,
    production_should_yield_to_dock,
)
from pathlib import Path

import cv2
from filelock import FileLock

from module.base.button import Button
from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.assets import (
    ALLOCATION_ASSIGNEE_CHECK,
    ALLOCATION_DIRECT_START_BUTTON,
    FARM_IDLE_CHECK,
    FARM_WORKING_CHECK,
    PRODUCTION_WORKING_CHECK,
    PRODUCTION_RESIDENT_DETAIL_CHECK,
)
from module.jiangnan.harvest import JiangnanHarvest
from module.jiangnan.foundation_icons import find_foundation_icon
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.logger import logger
from module.jiangnan.ocr import Ocr
from module.jiangnan.resident_reservation import is_reserved_resident, reserved_special_names
from module.jiangnan.resident_selection import unarranged_assignee_cards


class FarmTextOcr(Ocr):
    """识别农田面板标题或倒计时文本。"""

    def __init__(self, area, name):
        super().__init__(
            buttons=Button(area=area, color=(), button=area, name=name),
            lang='cnocr',
            name=name,
        )


class JiangnanFarmProduction(JiangnanHarvest):
    """逐城扫描农田，并按用户配置的挡位优先级开启生产。"""

    TASK = 'JiangnanFarmProduction'
    STATE_FILE = Path('jiangnan_farm_state.json')
    RESOURCE = '蔬菜'
    RETRY = timedelta(minutes=10)
    NO_FARM_RETRY = timedelta(hours=24)
    DEFAULT_TIME = datetime(2020, 1, 1)

    # 常规负责人页第一格是“直接开始”，第一名人物在第二格；
    # 人物紧缺时“直接开始”会消失，第一格会变成人物卡。
    # Keep every randomized click inside the portrait. The previous bottom
    # y=248 overlapped the red 详情 strip and opened the resident profile.
    SPECIAL_TASK_ASSIGNEE = (735, 105, 810, 180)

    # Only click the option title. The former full-quadrant hit boxes reached
    # the ingredient row (y≈270/480), where a click opens a material tooltip
    # instead of the assignee page. This applies to every production panel.
    LEVEL_BUTTONS = {
        1: (455, 120, 585, 165),
        2: (835, 120, 965, 165),
        3: (455, 335, 585, 380),
        4: (835, 335, 965, 380),
    }
    LEVEL_AMOUNT = {1: 14, 2: 150, 3: 225, 4: 300}
    LEVEL_DURATION = {
        1: timedelta(minutes=3),
        2: timedelta(minutes=30),
        3: timedelta(hours=1),
        4: timedelta(hours=1, minutes=30),
    }
    # 配置值使用稳定的英文键，中文只交给 i18n 显示。
    # 挡位编号与游戏面板位置对应：少许=1、一些=2、许多=3、丰厚=4。
    OPTION_LEVEL = {
        'little': 1,
        'some': 2,
        'many': 3,
        'rich': 4,
    }
    OPTION_NAME = {
        'little': '少许蔬菜',
        'some': '一些蔬菜',
        'many': '许多蔬菜',
        'rich': '丰厚蔬菜',
    }
    DEFAULT_PRIORITY = ('some', 'little', 'many', 'rich')
    LEVEL_CN = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五'}

    FARM_WORKING_TIME_AREA = (445, 205, 710, 255)
    FARM_NAVIGATION_AREA = (930, 150, 1200, 470)

    def __init__(self, config, device=None, task=None, path=None):
        super().__init__(config=config, device=device, task=task)
        self.path = Path(path) if path else self.STATE_FILE
        self.data = self._load_state()
        self.working_time_ocr = FarmTextOcr(
            self.FARM_WORKING_TIME_AREA, 'FARM_WORKING_TIME_OCR'
        )
        self.speed_title_ocr = FarmTextOcr(
            (405, 125, 900, 195), 'PRODUCTION_SPEED_TITLE_OCR'
        )
        self.last_insufficient_inventory = None
        self.last_priority_empty = False
        self.last_no_workers = False
        self._last_farm_target = None
        self._last_farm_map_frame = None

    def _load_state(self):
        try:
            with self.path.open('r', encoding='utf-8') as stream:
                data = json.load(stream)
            if isinstance(data, dict):
                data.setdefault('cities', {})
                return data
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f'Unable to read Jiangnan farm state: {exc}')
        return {'version': 1, 'cities': {}}

    def save(self):
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)
        temporary.replace(self.path)

    def _save_production_progress(self, city, buildings, facility=None):
        """Checkpoint only the inspected group, preserving other runners' updates."""
        with FileLock(str(self.path) + '.lock', timeout=10):
            latest = self._load_state()
            city_data = latest.setdefault('cities', {}).setdefault(city, {})
            if facility is None:
                city_data['buildings'] = buildings
            else:
                group = city_data.setdefault('facilities', {}).setdefault(facility, {})
                group.update(unavailable=False, buildings=buildings)
            self.data = latest
            self.save()

    @staticmethod
    def _parse_time(value):
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _time_text(value):
        return value.replace(microsecond=0).isoformat(sep=' ')

    @staticmethod
    def _ocr_text(reader, image):
        raw = reader.ocr(image)
        if isinstance(raw, list):
            raw = raw[0] if raw else ''
        return re.sub(r'\s+', '', str(raw))

    @classmethod
    def parse_duration(cls, text):
        text = re.sub(r'\s+', '', str(text)).translate(str.maketrans({
            'i': '1', 'I': '1', 'l': '1', '|': '1', '!': '1',
            'O': '0', 'o': '0',
        }))
        # 描边文字会在数字和单位之间夹入 t/` 等噪声，
        # 例如“26t秒”。允许最多两个非数字噪声，仍然保留到秒。
        hour = re.search(r'(\d+)\D{0,2}(?:小时|时)', text)
        minute = re.search(r'(\d+)\D{0,2}(?:分钟|分)', text)
        second = re.search(r'(\d+)\D{0,2}秒', text)
        if not any((hour, minute, second)):
            return None
        return timedelta(
            hours=int(hour.group(1)) if hour else 0,
            minutes=int(minute.group(1)) if minute else 0,
            seconds=int(second.group(1)) if second else 0,
        )

    @staticmethod
    def parse_search_result(text):
        text = re.sub(r'\s+', '', str(text))
        matched = re.search(
            r'([12345一二三四五])级农田(?:\D*(\d+)\D*)?$',
            text,
        )
        if not matched:
            return None, None
        levels = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
        level = levels.get(matched.group(1), matched.group(1))
        return int(level), int(matched.group(2) or 1)

    def _farm_key(self, city_key, field):
        return f'{self.TASK}.farm_{city_key}.{field}'

    def _farm_label(self, level, index, total):
        level_text = self.LEVEL_CN.get(level, str(level))
        return f'{level_text}级农田{index}/{total}'

    def _reference(self, city, level, index):
        return f'farm:{city}:{level}:{index}'

    @classmethod
    def _is_standard_farm_result(cls, result_name):
        """只接受“X级农田(数量)”，明确排除“改良农田”。"""
        if '改良农田' in str(result_name):
            return False
        level, total = cls.parse_search_result(result_name)
        return level is not None and total is not None and total > 0

    def _disable_city_without_farm(self, city, search_result=''):
        city_key = next(
            (key for key, name in CITY_KEY_TO_NAME.items() if name == city),
            None,
        )
        logger.warning(
            f'No standard farm in {city}; disable farm production, '
            f'search_result={search_result or "<empty>"}'
        )
        self.data.setdefault('cities', {})[city] = {
            'buildings': {},
            'unavailable': True,
            'search_result': str(search_result),
            'updated_at': self._time_text(datetime.now()),
        }
        self.save()
        if city_key is not None:
            self.disable_city_building_feature(
                task=self.TASK,
                group_prefix='farm',
                city_key=city_key,
                city=city,
                building_name='普通农田',
                status_field='Status',
                status_text='无普通农田，已自动关闭',
                next_run_field='NextRun',
                next_run_value=self.DEFAULT_TIME,
            )

    def _find_navigation_buttons(self):
        """用青色按钮轮廓动态定位农田导航，不使用固定点击点。"""
        x1, y1, x2, y2 = self.FARM_NAVIGATION_AREA
        image = self.device.image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (65, 45, 70), (110, 255, 230))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        buttons = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if 100 <= width <= 190 and 30 <= height <= 75:
                buttons.append(Button(
                    area=(), color=(),
                    button=(
                        x1 + left + 8,
                        y1 + top + 5,
                        x1 + left + width - 8,
                        y1 + top + height - 5,
                    ),
                    name='FARM_NAVIGATION_DYNAMIC',
                ))
        return sorted(buttons, key=lambda button: button.button[1])

    def _next_farm(self):
        self.device.screenshot()
        buttons = self._find_navigation_buttons()
        if len(buttons) < 2:
            logger.warning('Farm next button was not recognized')
            return False
        found = buttons[-1]
        target = Button(
            area=(), color=(), button=found.button,
            name='FARM_NEXT_DYNAMIC',
        )
        self.device.click(target)
        self.device.sleep(0.8)
        return True

    def _find_centered_farm(self):
        """Use the gather icon below the search arrow to avoid nearby tiles."""
        for attempt in range(3):
            arrow = self.find_building_search_arrow()
            match = find_foundation_icon(
                getattr(self.device, 'image', None), arrow, '农田',
                max_horizontal=75,
            )
            if match is not None:
                x, y, score, family = match
                logger.info(
                    f'Farm foundation icon: family={family}, '
                    f'center=({x}, {y}), score={score:.3f}'
                )
                return Button(
                    area=(), color=(), button=(x - 6, y - 6, x + 6, y + 6),
                    name='FARM_FOUNDATION_ICON_DYNAMIC',
                )
            target = self.find_centered_search_target()
            if target is not None:
                target.name = 'FARM_ARROW_TARGET_DYNAMIC'
                return target
            if attempt < 2:
                self.device.sleep(0.25)
                self.device.screenshot()
        logger.warning('Farm search arrow missing; skip uncertain map click')
        return None

    def _close_panel(self):
        self.device.press_back()
        deadline = time.time() + 8
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.is_in_main():
                return True
        return False

    def _refresh_live_inventory_from_production_panel(
        self, city, inventory_state, resources, required_resources=None,
        timeout=8,
    ):
        """Refresh authoritative stock via the warehouse button on a production panel."""
        from module.jiangnan.resource import WAREHOUSE_SPECIAL_RESOURCES
        from module.jiangnan.resource import SHAOXING_MERCHANT_RESOURCES
        from module.jiangnan.warehouse import JiangnanWarehouse

        resources = tuple(dict.fromkeys(
            inventory_state.normalize_resource(resource)
            for resource in resources if resource
        ))
        self._last_production_warehouse_readings = {}
        if not resources:
            return True
        required = tuple(dict.fromkeys(
            inventory_state.normalize_resource(resource)
            for resource in (
                required_resources if required_resources is not None
                else resources
            )
            if resource
        ))

        warehouse = JiangnanWarehouse(self.config, device=self.device)
        if hasattr(inventory_state, 'capacities'):
            warehouse.warehouse_capacity = inventory_state.capacities().get(city)
        self.device.screenshot()
        button = warehouse.find_build_warehouse_button(similarity=0.62)
        if button is None:
            logger.warning(
                f'Production warehouse button not recognized: {city}, '
                f'targets={resources}'
            )
            return False

        logger.info(
            f'Refresh production inventory from live warehouse: '
            f'{city}, targets={resources}'
        )
        self.device.click(button)
        opened = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if warehouse.is_warehouse_page():
                opened = True
                break
        if not opened:
            logger.warning('Production warehouse did not open')
            return False

        readings = {}
        restored = False
        scan_success = False
        try:
            merchant_targets = (
                set(resources) & set(SHAOXING_MERCHANT_RESOURCES)
                if city == '绍兴府' else set()
            )
            if city == '绍兴府':
                if not warehouse.select_shaoxing_warehouse('城市'):
                    return False
            # This entry opens the warehouse around goods related to the current
            # production panel. Read the actual visible rows; do not reset top.
            readings.update(warehouse.read_current_page(
                verify_names=True,
                target_resources=None,
            ))

            # The panel shortcut opens the warehouse near its current item.
            # A required resource may be on a different page (e.g. 原木 while
            # the warehouse opens at 油纸). Search only when the first page
            # misses it, using the warehouse's visual row tracker.
            missing_required = (
                set(required) - set(WAREHOUSE_SPECIAL_RESOURCES)
                - set(SHAOXING_MERCHANT_RESOURCES)
            ) - {
                resource for resource, value in readings.items()
                if int(value) >= 0
            }
            if missing_required and warehouse.reset_warehouse_top():
                for page in range(20):
                    self.device.screenshot()
                    current = warehouse.read_current_page(
                        verify_names=True,
                        target_resources=missing_required,
                    )
                    readings.update(current)
                    missing_required -= {
                        resource for resource, value in current.items()
                        if int(value) >= 0
                    }
                    if not missing_required:
                        break
                    if warehouse.scroll_warehouse_down() is not True:
                        break
                logger.attr(
                    'ProductionWarehouseTargetScan',
                    f'pages={page + 1}, missing={sorted(missing_required)}',
                )

            special_targets = tuple(WAREHOUSE_SPECIAL_RESOURCES)
            if special_targets:
                area = (300, 55, 1100, 590)
                text = self._ocr_text(Ocr(
                    Button(
                        area=area, color=(), button=area,
                        name='PRODUCTION_WAREHOUSE_SPECIAL_NAMES',
                    ),
                    lang='cnocr',
                    name='PRODUCTION_WAREHOUSE_SPECIAL_NAMES',
                ), self.device.image)
                visible_special = [
                    resource for resource in special_targets
                    if resource in text
                ]
                if visible_special:
                    readings.update(warehouse.read_special_items(
                        target_resources=visible_special,
                    ))

            capacity = warehouse.read_visible_warehouse_capacity()
            if merchant_targets:
                merchant = warehouse.scan_shaoxing_merchant(
                    target_resources=merchant_targets,
                )
                if merchant is not None:
                    readings.update(merchant)
                if warehouse.merchant_capacity:
                    groups = inventory_state.data.setdefault(
                        'special_capacities', {}
                    ).setdefault(city, {})
                    groups['绍兴商行'] = int(warehouse.merchant_capacity)
                    inventory_state.save()

            valid = {
                inventory_state.normalize_resource(resource): int(value)
                for resource, value in readings.items()
                if int(value) >= 0
            }
            self._last_production_warehouse_readings = dict(valid)
            if valid:
                inventory_state.reconcile_live_inventory(
                    city,
                    valid,
                    reason='production panel live warehouse scan (all visible resources)',
                    update=False,
                )

            if capacity is not None and capacity > 0:
                self.config.modified[
                    inventory_state._capacity_key(city)
                ] = int(capacity)

            if valid or capacity:
                self.config.update()

            missing = [resource for resource in resources if resource not in valid]
            missing_required = [
                resource for resource in required if resource not in valid
            ]
            logger.attr(
                'ProductionWarehouseRefresh',
                f'city={city}, read={valid}, missing={missing}, '
                f'required_missing={missing_required}, capacity={capacity}',
            )
            scan_success = not missing_required
        finally:
            self.device.press_back()
            deadline = time.time() + timeout
            while time.time() < deadline:
                self.device.sleep(0.25)
                self.device.screenshot()
                if warehouse.is_warehouse_page():
                    continue
                state, _ = self._panel_state()
                if state is not None:
                    restored = True
                    break
            if not restored:
                logger.warning(
                    'Production panel was not restored after warehouse refresh'
                )
        return scan_success and restored

    @staticmethod
    def _upgrade_portrait_frames(image):
        """升级弹窗下方有三张人物卡，普通生产加速只有中间一张。"""
        if image is None or image.shape[0] < 660 or image.shape[1] < 805:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        side_borders = (
            gray[570:655, 475:487],
            gray[570:655, 793:805],
        )
        return all(float((border < 90).mean()) >= 0.20
                   for border in side_borders)

    def _speed_popup_role(self):
        """Return working/upgrading only when the popup provides evidence."""
        if not self.appear(
            PRODUCTION_WORKING_CHECK, offset=(5, 5), similarity=0.80
        ):
            return None
        if not self._upgrade_portrait_frames(self.device.image):
            return 'working'
        # Real production speedups can show three or four worker portraits.
        # Portrait count alone misclassified a 瓷窑加速 popup as upgrading.
        title = re.sub(r'\s+', '', self._ocr_text(
            self.speed_title_ocr, self.device.image
        ))
        logger.attr('ProductionSpeedTitle', title)
        if '升级' in title:
            return 'upgrading'
        if '加速' in title:
            return 'working'
        return 'unknown'

    def _is_upgrade_panel(self):
        return self._speed_popup_role() == 'upgrading'

    def _panel_state(self):
        speed_role = self._speed_popup_role()
        if speed_role == 'unknown':
            return None, None
        if speed_role == 'upgrading':
            time_text = self._ocr_text(self.working_time_ocr, self.device.image)
            duration = self.parse_duration(time_text)
            logger.attr('FarmUpgradeTimeText', time_text)
            if duration is not None and duration > timedelta(0):
                return 'upgrading', duration
            return None, None
        if speed_role == 'working' or self.appear(
            FARM_WORKING_CHECK, offset=(4, 4), similarity=0.82
        ):
            time_text = self._ocr_text(self.working_time_ocr, self.device.image)
            duration = self.parse_duration(time_text)
            logger.attr('FarmWorkingTimeText', time_text)
            return 'working', duration
        if self.appear(FARM_IDLE_CHECK, offset=(4, 4), similarity=0.82):
            return 'idle', None
        return None, None

    def _confirm_live_production_state(self, old, state, duration, label):
        """台账状态或完成时间与面板不符时，再读一次当前面板。"""
        self._last_production_record_drift = False
        if state not in ('idle', 'working', 'upgrading', 'overflow'):
            return state, duration
        old_state = str((old or {}).get('state') or '')
        expected = (
            old_state if old_state in ('working', 'upgrading', 'overflow')
            else 'idle' if old_state in (
                'idle', 'threshold', 'no_workers', 'no_priority'
            ) else None
        )
        now = datetime.now().replace(microsecond=0)
        old_finish = self._parse_time((old or {}).get('finish_at'))
        observed_finish = (
            now + duration if state in ('working', 'upgrading')
            and duration is not None else None
        )
        changed = expected is not None and expected != state
        drifted = (
            expected == state and old_finish is not None
            and observed_finish is not None
            and abs((observed_finish - old_finish).total_seconds()) > 120
        )
        if not changed and not drifted:
            return state, duration
        completed = (
            expected in ('working', 'upgrading') and state == 'idle'
            and old_finish is not None and old_finish <= now
        )
        log_state = logger.info if completed else logger.warning
        log_state(
            f'Production record differs from live panel: {label}, '
            f'recorded={old_state}, finish={old_finish}, '
            f'live={state}, remaining={duration}; verify again'
        )
        if not completed:
            self.save_abnormal_screenshot('production_state_mismatch')
        self.device.sleep(0.3)
        self.device.screenshot()
        if state == 'overflow':
            confirmed = (
                self.find_warehouse_full() is not None
                or getattr(self, '_warehouse_full_banner_visible', False)
            )
            checked_state, checked_duration = (
                ('overflow', None) if confirmed else (None, None)
            )
        else:
            checked_state, checked_duration = self._panel_state()
        durations_agree = (
            state not in ('working', 'upgrading') or (
                duration is not None and checked_duration is not None
                and abs((duration - checked_duration).total_seconds()) <= 90
            )
        )
        if checked_state != state or not durations_agree:
            logger.warning(
                f'Production panel reads disagree: {label}; retry soon'
            )
            self._close_panel()
            return None, None
        logger.info(
            f'Production state correction confirmed: {label}, '
            f'{old_state} -> {checked_state}, remaining={checked_duration}'
        )
        self._last_production_record_drift = drifted
        return checked_state, checked_duration

    def _open_current_farm(self, timeout=8, farm_button=None):
        self.device.screenshot()
        navigation_buttons = self._find_navigation_buttons() if farm_button is None else ()
        if farm_button is None and len(navigation_buttons) < 2:
            # 只有一座农田或“完成”旗遮住导航时，当前搜索结果仍可能
            # 已经居中；继续以农田本体确认，避免放弃读取倒计时。
            logger.warning(
                'Farm navigation panel is missing; '
                'continue with centered-farm detection'
            )
        if farm_button is None:
            farm_button = self._find_centered_farm()
        if farm_button is None:
            return None, None
        selected_frame = self.device.image.copy()
        self._last_farm_target = farm_button
        self._last_farm_map_frame = selected_frame
        self._click_searched_building(farm_button, navigation_buttons)
        deadline = time.time() + timeout
        clicked_at = time.time()
        retried = False
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            full = self.find_warehouse_full()
            if full is not None:
                return 'overflow', None
            state, duration = self._panel_state()
            if state is not None:
                return state, duration
            # 如果第一次点到的是“完成”旗帜，库存未满时
            # 会直接收获并留在主界面。确认仍在农田导航状态后
            # 再点一次居中建筑，打开空闲生产面板。
            if (
                not retried
                and time.time() - clicked_at >= 1.5
                and self.is_in_main()
            ):
                retried = True
                retry_button = self._find_centered_farm()
                if retry_button is None:
                    retry_button = self.find_still_visible_map_target(
                        selected_frame, farm_button
                    )
                if (
                    retry_button is not None
                    and farm_button.name == 'FARM_ARROW_TARGET_DYNAMIC'
                    and retry_button.name == 'FARM_ARROW_TARGET_DYNAMIC'
                ):
                    # On foundation tiles the generic contour can cover the
                    # tile's upper edge while the actual farm icon sits lower.
                    # Keep the second click inside that same selected tile.
                    x1, y1, x2, y2 = retry_button.button
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2 + 22
                    retry_button = Button(
                        area=(), color=(),
                        button=(center_x - 6, center_y - 6,
                                center_x + 6, center_y + 6),
                        name='FARM_ARROW_TARGET_LOWER_DYNAMIC',
                    )
                    logger.info(
                        f'Retry farm at lower part of selected tile: '
                        f'({center_x}, {center_y})'
                    )
                if retry_button is not None:
                    self.device.click(retry_button)
        logger.warning('Farm panel state was not recognized')
        self.save_abnormal_screenshot('farm_panel_unknown')
        return None, None

    def _use_special_task(self, city):
        city_key = next(
            (key for key, name in CITY_KEY_TO_NAME.items() if name == city),
            None,
        )
        if city_key is None:
            return False
        if self.TASK == 'JiangnanFarmProduction':
            prefix = 'farm'
        elif self.TASK == 'JiangnanImprovedFarmProduction':
            prefix = 'improved_farm'
        elif (
            self.TASK == 'JiangnanRestaurantProduction'
            and city_key == 'huizhoufu'
        ):
            prefix = 'restaurant'
        else:
            prefix = 'production'
        return bool(self.config.cross_get(
            keys=f'{self.TASK}.{prefix}_{city_key}.UseSpecialTask',
            default=True,
        ))

    def _read_resource_shortage(self, expected_resources=None):
        """Return live inventory shown by a resource-shortage popup."""
        return None

    def _dismiss_resource_shortage(self):
        """Close the shortage popup without ever clicking paid replenish."""
        self.device.press_back()
        deadline = time.time() + 5
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            state, _ = self._panel_state()
            if state == 'idle' or self.is_in_main():
                return True
        logger.warning('Resource-shortage popup did not close cleanly')
        return False

    def _is_no_workers_popup(self):
        """Identify the centered 提示 dialog before treating it as a stall."""
        text = str(FarmTextOcr(
            (380, 308, 860, 380), 'PRODUCTION_NO_WORKERS_OCR',
        ).ocr(self.device.image))
        normalized = re.sub(r'\s+', '', text)
        return '工人' in normalized and (
            '没有更多' in normalized or '没有多余' in normalized
        )

    def _handle_no_workers_popup(self, city):
        if not self._is_no_workers_popup():
            return False
        self.last_no_workers = True
        logger.warning(f'No free workers in {city}; skip production start')
        self._dismiss_resource_shortage()
        return True

    def _is_no_special_worker_toast(self):
        """Recognize unavailable or task-restricted special residents."""
        text = str(FarmTextOcr(
            (375, 215, 945, 257), 'PRODUCTION_NO_SPECIAL_WORKER_OCR',
        ).ocr(self.device.image))
        normalized = re.sub(r'\s+', '', text)
        return (
            '无空闲特殊居民' in normalized and '工作' in normalized
        ) or any(text in normalized for text in (
            # The narrow toast crop often clips the trailing “任务”.
            '禁止参与此', '禁止参加此',
        ))

    def _first_worker_card_visible(self):
        """Detect a portrait when the direct-start tile has disappeared."""
        image = getattr(self.device, 'image', None)
        if image is None or image.shape[0] < 205 or image.shape[1] < 452:
            return False
        portrait = image[85:202, 334:450]
        hsv = cv2.cvtColor(portrait, cv2.COLOR_BGR2HSV)
        colored = float((hsv[:, :, 1] > 55).mean())
        logger.attr('ProductionFirstWorkerCardColor', round(colored, 3))
        return colored > 0.16

    def _assignee_row_tops(self):
        """Locate the visible card rows from their long horizontal dividers."""
        image = getattr(self.device, 'image', None)
        if image is None or image.shape[0] < 650 or image.shape[1] < 1080:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_fraction = (gray[170:640, 325:1080] < 105).mean(axis=1)
        dividers = []
        for offset in (dark_fraction > 0.58).nonzero()[0]:
            y = int(offset) + 170
            if not dividers or y - dividers[-1] > 4:
                dividers.append(y)
        tops = [55]
        tops.extend(y + 18 for y in dividers if y + 18 < 570)
        logger.attr('ProductionAssigneeRowTops', tops)
        return tops

    def _visible_special_worker_cards(self, direct_start_visible):
        """Return possible card slots at the current scrolled row positions."""
        slots = []
        for row, y in enumerate(self._assignee_row_tops()):
            for col, x in enumerate((334, 715)):
                if row == 0 and col == 0 and direct_start_visible:
                    continue
                slots.append((x, y))
        return slots

    def _unreserved_production_assignee(self, direct_start_visible, reserved):
        """Choose a clearly read resident outside this city's reserved list."""
        slots = (unarranged_assignee_cards(getattr(self.device, 'image', None))
                 if direct_start_visible
                 else self._visible_special_worker_cards(False))
        if not slots:
            return None
        if not reserved:
            x, y = slots[0]
            return (x + 20, y + 45, x + 96, y + 110)
        buttons = []
        for x, y in slots:
            name_x = x + 119
            areas = (
                (name_x, y - 16, name_x + 205, y + 44),
                (name_x + 4, y - 7, name_x + 172, y + 33),
            )
            for area in areas:
                buttons.append(Button(
                    area=area, color=(), button=area,
                    name='PRODUCTION_ASSIGNEE_NAME',
                ))
        raw_names = Ocr(
            buttons=buttons, lang='cnocr',
            name='PRODUCTION_ASSIGNEE_NAMES_OCR',
        ).ocr(self.device.image)
        if not isinstance(raw_names, (list, tuple)):
            raw_names = [raw_names]
        for index, (x, y) in enumerate(slots):
            readings = [
                re.sub(r'[^\u4e00-\u9fff]', '', str(raw))
                for raw in raw_names[index * 2:index * 2 + 2]
            ]
            logger.attr('ProductionAssigneeName', readings)
            if (len(readings) != 2 or len(readings[0]) < 2
                    or readings[0] != readings[1]):
                continue
            name = readings[0]
            if is_reserved_resident(name, reserved):
                logger.info(f'Keep reserved resident {name} available for exclusive tasks')
                continue
            return (x + 20, y + 45, x + 96, y + 110)
        return None

    def _recover_production_resident_detail(self):
        """Return to assignees after a profile tap, without assigning a resident."""
        if not self.appear(
            PRODUCTION_RESIDENT_DETAIL_CHECK, offset=(12, 12), similarity=0.80,
        ):
            return None
        logger.warning('Resident details opened during production selection; return to portraits')
        self.device.press_back()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if self.appear(ALLOCATION_ASSIGNEE_CHECK, offset=(8, 8), similarity=0.82):
                return 'restored'
        logger.warning('Unable to return from resident details to production assignees')
        return 'failed'

    def _direct_start(
        self, city=None, timeout=12, expected_resources=None
    ):
        self.last_insufficient_inventory = None
        self.last_no_workers = False

        def handle_shortage():
            shortage = self._read_resource_shortage(expected_resources)
            if not shortage:
                return False
            self.last_insufficient_inventory = shortage
            logger.warning(
                f'Production resource shortage popup: city={city}, '
                f'live_inventory={shortage}'
            )
            self._dismiss_resource_shortage()
            return True

        deadline = time.time() + timeout
        main_confirmations = 0
        while time.time() < deadline:
            self.device.screenshot()
            recovered = self._recover_production_resident_detail()
            if recovered == 'failed':
                return False
            if recovered:
                continue
            if self.appear(
                ALLOCATION_ASSIGNEE_CHECK,
                offset=(8, 8),
                similarity=0.82,
            ):
                break
            # The city navigation remains visible behind the no-workers
            # dialog.  Checking only once per second can therefore accept
            # two false "main page" confirmations before reading the dialog.
            if self._handle_no_workers_popup(city):
                return False
            if handle_shortage():
                return False
            if self.is_in_main():
                main_confirmations += 1
                if main_confirmations >= 2:
                    logger.info(
                        'Production option returned directly to the city; '
                        'verify the started timer on the retained building'
                    )
                    return True
            else:
                main_confirmations = 0
            self.device.sleep(0.35)
        else:
            logger.warning('Farm assignee page did not appear')
            self.save_abnormal_screenshot('production_assignee_timeout')
            return False

        direct_start_visible = self.appear(
            ALLOCATION_DIRECT_START_BUTTON, offset=(8, 8), similarity=0.82,
        )
        first_worker_visible = (
            not direct_start_visible and self._first_worker_card_visible()
        )
        if not direct_start_visible and not first_worker_visible:
            logger.warning(f'{city} assignee page has no start tile or worker card')
            self.last_no_workers = True
            self._close_panel()
            return False

        use_special = city is not None and self._use_special_task(city)
        reserved = reserved_special_names(getattr(self, 'config', None), city)
        safe_assignee = None
        if use_special and (reserved or direct_start_visible):
            safe_assignee = self._unreserved_production_assignee(
                direct_start_visible, reserved
            )
            if safe_assignee is None:
                logger.info(
                    f'No usable unarranged/unreserved assignee in {city}; '
                    'use direct start'
                )
                if not direct_start_visible:
                    self.last_no_workers = True
                    self._close_panel()
                    return False

        if use_special and (safe_assignee is not None
                            or (not direct_start_visible and not reserved)):
            logger.info(
                f'{city} production: choose available character'
            )
            assignee = Button(
                area=(), color=(),
                button=(safe_assignee if safe_assignee is not None
                        else (355, 105, 430, 180)),
                name='PRODUCTION_SPECIAL_TASK_ASSIGNEE',
            )
            # 人物卡存在时点击后会直接返回城市；无人可用时该位置为空，
            # 页面保持不变，短暂确认后自然回退“直接开始”。补点一次可
            # 区分偶发吞点与真正无人可用。
            special_unavailable = False
            for attempt in range(1, 3):
                self.device.click(assignee)
                confirm_deadline = time.time() + 2.5
                left_assignee_page = False
                recovered_detail = False
                main_confirmations = 0
                while time.time() < confirm_deadline:
                    self.device.sleep(0.3)
                    self.device.screenshot()
                    recovered = self._recover_production_resident_detail()
                    if recovered == 'failed':
                        return False
                    if recovered:
                        recovered_detail = True
                        break
                    if self._handle_no_workers_popup(city):
                        return False
                    if handle_shortage():
                        return False
                    if self._is_no_special_worker_toast():
                        logger.info(
                            f'{city} special resident unavailable or forbidden for this task; '
                            'use direct start'
                        )
                        special_unavailable = True
                        break
                    if self.is_in_main():
                        main_confirmations += 1
                        if main_confirmations >= 2:
                            logger.info(
                                f'{city} production special character selected'
                            )
                            return True
                    else:
                        main_confirmations = 0
                    if not self.appear(
                        ALLOCATION_ASSIGNEE_CHECK,
                        offset=(8, 8),
                        similarity=0.82,
                    ):
                        left_assignee_page = True
                if recovered_detail:
                    logger.info('Retry production selection inside portrait after leaving details')
                    continue
                if special_unavailable:
                    self.device.sleep(0.6)
                    self.device.screenshot()
                    break
                if left_assignee_page:
                    # 已离开选人页说明人物点击生效，只是城市页面仍在
                    # 过渡中；此时不能在未知页面继续补点或点“直接开始”。
                    transition_deadline = time.time() + timeout
                    while time.time() < transition_deadline:
                        self.device.sleep(0.35)
                        self.device.screenshot()
                        recovered = self._recover_production_resident_detail()
                        if recovered == 'failed':
                            return False
                        if recovered:
                            recovered_detail = True
                            break
                        if self._handle_no_workers_popup(city):
                            return False
                        if handle_shortage():
                            return False
                        if self.is_in_main():
                            logger.info(
                                f'{city} production special character '
                                'selected after transition'
                            )
                            return True
                    if recovered_detail:
                        continue
                    logger.warning(
                        'Production left assignee page but did not return '
                        'to the city'
                    )
                    self.save_abnormal_screenshot('production_assignee_transition_stuck')
                    return False
                logger.info(
                    'Special-character selection was not committed: '
                    f'{attempt}/2'
                )
            logger.warning(
                f'{city} has no usable special character; '
                'fall back to direct start'
            )

        if not direct_start_visible:
            # When workers are scarce the first tile is a portrait, so the
            # usual direct-start coordinate would click that portrait again.
            logger.warning(f'{city} direct-start tile is unavailable')
            self.last_no_workers = True
            self._close_panel()
            return False

        logger.info('Farm production: choose direct start')
        self.device.click(self.anchored_button(
            ALLOCATION_ASSIGNEE_CHECK,
            ALLOCATION_DIRECT_START_BUTTON,
            name='FARM_DIRECT_START_DYNAMIC',
            inset=8,
        ))
        deadline = time.time() + timeout
        main_confirmations = 0
        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self._handle_no_workers_popup(city):
                return False
            if handle_shortage():
                return False
            if self.is_in_main():
                main_confirmations += 1
                if main_confirmations >= 2:
                    return True
            else:
                main_confirmations = 0
        logger.warning('Farm direct start was not confirmed')
        return False

    def _actual_started_duration(self, option_duration, farm_index=1):
        """开工后从加速面板读取真实倒计时，失败时保守沿用选项时间。"""
        # 开工会取消搜索箭头，但地图通常保持原位。先用刚才打开的
        # 农田位置读取加速倒计时，只有地图移动或原位重开失败才重搜。
        self.device.screenshot()
        target = getattr(self, '_last_farm_target', None)
        previous_map = getattr(self, '_last_farm_map_frame', None)
        if (target is not None and previous_map is not None
                and self.is_in_main()
                and self._map_view_unchanged(
                    previous_map, getattr(self.device, 'image', None),
                )):
            logger.info('Farm map stayed in place after start; reopen same farm')
            state, duration = self._open_current_farm(
                timeout=9, farm_button=target,
            )
            if state == 'working' and duration is not None and duration > timedelta(0):
                observed_finish = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
                return max(
                    timedelta(seconds=1),
                    observed_finish - datetime.now().replace(microsecond=0),
                )
            if state is not None:
                self._close_panel()
            logger.warning('Farm time unreadable at retained location; re-search')
        if not self.search_building(
            '农田',
            result_validator=self._is_standard_farm_result,
            rejected_is_not_found=lambda value: '改良农田' in value,
        ):
            logger.warning(
                'Unable to relocate farm after start; use option duration'
            )
            return option_duration
        for _ in range(max(0, int(farm_index) - 1)):
            if not self._next_farm():
                logger.warning(
                    'Unable to select started farm; use option duration'
                )
                return option_duration
        state, duration = self._open_current_farm(timeout=9)
        if state == 'working' and duration is not None and duration > timedelta(0):
            observed_finish = datetime.now().replace(microsecond=0) + duration
            logger.info(
                f'Farm actual duration after start: {duration} '
                f'(option={option_duration})'
            )
            self._close_panel()
            # 调用方会用“返回时刻 + duration”登记完成节点，扣除关闭
            # 面板所耗时间，保持与 OCR 那一秒观察到的倒计时一致。
            return max(
                timedelta(seconds=1),
                observed_finish - datetime.now().replace(microsecond=0),
            )
        if state is not None:
            self._close_panel()
        logger.warning(
            'Unable to read actual farm duration after start; '
            f'use option duration {option_duration}'
        )
        return option_duration

    def _production_priority(self, city=None):
        """读取当前城市优先级；“无”和重复项都会被跳过。"""
        city_key = next(
            (key for key, name in CITY_KEY_TO_NAME.items() if name == city),
            None,
        )
        priority = []
        for index in range(1, 5):
            if city_key is None:
                key = f'{self.TASK}.FarmProduction.Priority{index}'
            else:
                key = f'{self.TASK}.farm_{city_key}.Priority{index}'
            value = self.config.cross_get(
                keys=key, default=self.DEFAULT_PRIORITY[index - 1]
            )
            value = str(value)
            if value in self.OPTION_LEVEL and value not in priority:
                priority.append(value)
        return priority

    def _select_production_level(self, farm_level, city=None):
        available_max = min(max(int(farm_level), 1), 4)
        for option in self._production_priority(city=city):
            selected = self.OPTION_LEVEL[option]
            if selected <= available_max:
                return option, selected
        return None, None

    def _start_by_priority(self, farm_level, city=None, farm_index=1):
        self.last_priority_empty = False
        if getattr(self, 'defer_new_production', False):
            self.last_no_workers = True
            self._close_panel()
            return None, None
        option, selected = self._select_production_level(
            farm_level, city=city
        )
        self.last_priority_empty = selected is None
        if selected is None:
            logger.info(
                f'Do not start {city or "current city"} farm: '
                'all configured priorities are 无 or unavailable'
            )
            self._close_panel()
            return None, None
        target = Button(
            area=(), color=(), button=self.LEVEL_BUTTONS[selected],
            name=f'FARM_LEVEL_{selected}_DYNAMIC',
        )
        logger.info(
            f'Start farm production option by priority: '
            f'{self.OPTION_NAME[option]} (level={selected})'
        )
        self.device.click(target)
        if not self._direct_start(city=city):
            if self.last_no_workers:
                return None, None
            # 档位点击也会偶发被吞；只有仍明确停留在空闲生产面板时
            # 才安全地重点同一档一次。
            state, _ = self._panel_state()
            if state != 'idle':
                return None, None
            logger.warning('Farm option click was not committed; retry once')
            self.device.click(target)
            if not self._direct_start(city=city):
                return None, None
        duration = self._actual_started_duration(
            self.LEVEL_DURATION[selected], farm_index=farm_index
        )
        return self.LEVEL_AMOUNT[selected], duration

    def _known_amount(self, city, level, index):
        buildings = self.data.get('cities', {}).get(city, {}).get('buildings', {})
        item = buildings.get(str(index), {})
        try:
            amount = int(item.get('amount', 0))
        except (TypeError, ValueError):
            amount = 0
        # 手动开工的农田仅能从面板读到剩余时间，无法反推出选了哪一档。
        # 建筑等级不等于生产档位，不能把最高档产量当作已确认数量。
        return amount if amount > 0 else 0

    def inspect_city(self, city, inventory_state, prelocated_result=None):
        cached = cached_production_group(
            getattr(self, 'data', {}).get('cities', {}).get(city, {}).get('buildings', {}),
            inventory_state, city, (self.RESOURCE,),
        )
        if cached is not None:
            logger.info(f'Skip working farm group before finish: {city}')
            return cached
        if prelocated_result:
            self.last_search_result_name = str(prelocated_result)
            self.last_search_status = self.SEARCH_FOUND
            self.last_search_executed = True
            located = True
        else:
            located = self.search_building(
                '农田',
                result_validator=self._is_standard_farm_result,
                rejected_is_not_found=lambda value: '改良农田' in value,
            )
        if not located:
            # 必须已经实际点击搜索并读取过结果，才认定没有普通农田。
            # 搜索入口、输入法或界面加载失败不会误关城市。
            if self.building_search_confirmed_missing():
                self._disable_city_without_farm(
                    city, self.last_search_result_name
                )
                return {}
            return None
        level, total = self.parse_search_result(self.last_search_result_name)
        if level is None or total is None or total <= 0:
            # 已进入结果页但 OCR 结构异常不等于没有农田。只有搜索层
            # 明确认定 not_found 时才允许自动关闭，其他情况短时复查。
            if self.building_search_confirmed_missing():
                self._disable_city_without_farm(
                    city, self.last_search_result_name
                )
                return {}
            logger.warning(
                f'Farm search result cannot be parsed in {city}: '
                f'{self.last_search_result_name or "<empty>"}'
            )
            return None

        logger.attr('FarmSearchGroup', f'level={level}, count={total}')
        now = datetime.now().replace(microsecond=0)
        previous = (
            self.data.get('cities', {}).get(city, {}).get('buildings', {})
        )
        # A warehouse scan can pause the whole farm group after the first
        # inspected field. Keep the untouched records until the next live
        # inspection instead of searching the remaining fields needlessly.
        buildings = {
            str(index): dict(previous[str(index)])
            for index in range(1, total + 1)
            if str(index) in previous
        }
        inventory_refreshed = False
        workers_exhausted = False
        inspected = False
        for index in range(1, total + 1):
            old = previous.get(str(index), {})
            if production_record_pending(old):
                logger.info(f'Skip working farm {index}/{total} before {old.get("finish_at")}')
                continue
            if (old.get('state') == 'threshold'
                    and inventory_state.production_below_threshold(city, self.RESOURCE) is False):
                # A later completed field still needs harvesting. Do not
                # reopen already-paused fields every time that clock is due.
                continue
            if index > 1 and production_should_yield_to_dock(inventory_state):
                logger.info(
                    f'Yield farm {index}/{total} to arrived dock shipment'
                )
                inventory_state.sync_dock_schedule()
                break
            # 打开一块农田后，游戏会取消右侧实例选择器。
            # 因此每块都重新搜索回到 1/N，再识别点击“下一个”
            # 到目标序号，不假设操作后选择器仍然存在。
            if index > 1:
                if inspected and not self.search_building(
                    '农田',
                    result_validator=self._is_standard_farm_result,
                    rejected_is_not_found=lambda value: '改良农田' in value,
                ):
                    logger.warning(f'Unable to relocate farm {index}/{total}')
                    break
                located = True
                for _ in range(index - 1):
                    if not self._next_farm():
                        located = False
                        break
                if not located:
                    logger.warning(f'Unable to select farm {index}/{total}')
                    break
            old = previous.get(str(index), {})
            inspected = True
            state, duration = self._open_current_farm()
            state, duration = self._confirm_live_production_state(
                old, state, duration, f'{city} 农田 {index}/{total}'
            )
            amount = self._known_amount(city, level, index)
            if old.get('state') == 'working' and state == 'upgrading':
                inventory_state.cancel_inventory_change(
                    self._reference(city, level, index)
                )
            if state == 'working' and (
                old.get('state') != 'working'
                or self._last_production_record_drift
            ):
                amount = 0
                inventory_state.cancel_inventory_change(
                    self._reference(city, level, index)
                )
            finish_at = None
            reconciliation = None

            if state == 'overflow':
                inventory_state.cancel_inventory_change(
                    self._reference(city, level, index),
                    rollback_credited=True,
                )
                if self.resolve_warehouse_full(force=True):
                    logger.info(
                        'Force-harvest completed farm production; '
                        'reopen before threshold decision'
                    )
                    state, duration = self._open_current_farm()
                else:
                    finish_at = now

            if state == 'idle':
                use_cached = inventory_refreshed and old.get('state') not in (
                    'working', 'overflow'
                )
                if use_cached:
                    logger.info(f'Reuse farm inventory scan: {city} {self.RESOURCE}')
                refreshed = use_cached or self._refresh_live_inventory_from_production_panel(
                    city, inventory_state, (self.RESOURCE,)
                )
                inventory_refreshed = inventory_refreshed or refreshed
                if refreshed and old.get('state') in ('working', 'overflow'):
                    reconciliation = inventory_state.reconcile_production_harvest(
                        city=city,
                        resource=old.get('resource') or self.RESOURCE,
                        amount=old.get('amount') or amount,
                        reference=self._reference(
                            city, int(old.get('level') or level), index
                        ),
                        expected_at=old.get('finish_at'),
                        observed_at=datetime.now().replace(microsecond=0),
                        included_in_scan=True,
                    )
                if not refreshed:
                    logger.warning(
                        f'Unable to refresh live inventory before farm '
                        f'production: {city} {self.RESOURCE}'
                    )
                    below_threshold = None
                else:
                    below_threshold = inventory_state.production_below_threshold(
                        city, self.RESOURCE
                    )
                if below_threshold is False:
                    inventory_state.pause_city_resource_production(
                        city,
                        self.RESOURCE,
                        self.TASK,
                        group_prefix='farm',
                        reason='库存达到97%',
                    )
                    state = 'threshold'
                    self._close_panel()
                elif below_threshold is None:
                    state = 'unknown'
                    logger.warning(
                        f'Unable to evaluate 97% production threshold: '
                        f'{city} {self.RESOURCE}'
                    )
                    self._close_panel()
                elif workers_exhausted:
                    state = 'no_workers'
                    self._close_panel()
                else:
                    amount, duration = self._start_by_priority(
                        level, city=city, farm_index=index
                    )
                    if amount is not None and duration is not None:
                        state = 'working'
                        finish_at = datetime.now().replace(microsecond=0) + duration
                    elif self.last_priority_empty:
                        state = 'no_priority'
                    elif self.last_no_workers:
                        state = 'no_workers'
                        workers_exhausted = True
                    else:
                        state = 'unknown'
                        self._close_panel()
            elif state == 'working':
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
            elif state == 'upgrading':
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()

            if state == 'working' and finish_at is not None and amount > 0:
                inventory_state.schedule_inventory_change(
                    city,
                    self.RESOURCE,
                    amount,
                    finish_at,
                    reason='生产',
                    reference=self._reference(city, level, index),
                )

            buildings[str(index)] = {
                'level': level,
                'index': index,
                'total': total,
                'state': state or 'unknown',
                'resource': self.RESOURCE,
                'amount': int(amount or 0),
                'finish_at': self._time_text(finish_at) if finish_at else None,
                'harvest_reconciliation': reconciliation,
                'harvest_reconciled_at': (
                    self._time_text(datetime.now()) if reconciliation else None
                ),
                'updated_at': self._time_text(datetime.now()),
            }
            # Keep confirmed clocks even if a later field fails or the task
            # is interrupted while yielding to another scheduled job.
            self._save_production_progress(city, buildings)
            if state == 'threshold':
                logger.info(
                    f'Stop searching remaining farms in {city}: '
                    f'{self.RESOURCE} reached the production threshold'
                )
                break
        self._save_production_progress(city, buildings)
        return buildings

    def _sync_city_ui(self, city_key, city, fallback):
        city_data = self.data.get('cities', {}).get(city, {})
        buildings = city_data.get('buildings', {})
        lines = []
        has_overflow = any(item.get('state') == 'overflow' for item in buildings.values())
        for item in sorted(
            buildings.values(), key=lambda value: int(value.get('index', 0))
        ):
            state = item.get('state')
            if state == 'unknown':
                label = self._farm_label(
                    int(item.get('level', 0)),
                    int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                lines.append(f'[{label}][状态识别失败][等待复查]')
                continue
            if state == 'no_workers':
                label = self._farm_label(
                    int(item.get('level', 0)),
                    int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                lines.append(f'[{label}][人物不足][等待复查]')
                continue
            if state == 'upgrading':
                finish = self._parse_time(item.get('finish_at'))
                if finish is None:
                    lines.append('[设施升级中][倒计时识别失败][等待复查]')
                    continue
                label = self._farm_label(
                    int(item.get('level', 0)),
                    int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                lines.append(
                    f'[{label}][升级中][{self._time_text(finish)}]'
                )
                continue
            if state == 'no_priority':
                label = self._farm_label(
                    int(item.get('level', 0)),
                    int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                lines.append(f'[{label}][未选择生产挡位]')
                continue
            if state == 'threshold':
                label = self._farm_label(
                    int(item.get('level', 0)),
                    int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                lines.append(f'[{label}][库存达到97%][已暂停]')
                continue
            if state not in ('working', 'overflow'):
                continue
            finish = self._parse_time(item.get('finish_at'))
            if finish is None:
                label = self._farm_label(
                    int(item.get('level', 0)), int(item.get('index', 0)),
                    int(item.get('total', 0)),
                )
                detail = '生产中][倒计时识别失败' if state == 'working' else '溢出][等待收获'
                lines.append(f'[{label}][{detail}][等待复查]')
                continue
            reason = '生产' if state == 'working' else '溢出'
            label = self._farm_label(
                int(item.get('level', 0)),
                int(item.get('index', 0)),
                int(item.get('total', 0)),
            )
            amount = int(item.get('amount', 0))
            amount_text = f'+{amount}' if amount > 0 else '产量未知'
            lines.append(
                f'[{label}][{amount_text}]'
                f'[{self._time_text(finish)}][{reason}]'
            )
        now = datetime.now().replace(microsecond=0)
        paused = not bool(self.config.cross_get(
            keys=self._farm_key(city_key, 'Enable'), default=True
        )) if has_overflow else False
        idle_fallback = fallback
        if not buildings:
            configured = self._parse_time(self.config.cross_get(
                keys=self._farm_key(city_key, 'NextRun'), default=self.DEFAULT_TIME,
            ))
            if configured is not None and configured > now:
                idle_fallback = configured
        next_run = production_next_run(
            buildings, now, self.RETRY, idle_fallback, paused=paused,
        )
        if city_data.get('unavailable'):
            status_text = '无普通农田，已自动关闭'
        else:
            status_text = '\n'.join(lines) if lines else '无'
        self.config.modified[self._farm_key(city_key, 'Status')] = status_text
        self.config.modified[self._farm_key(city_key, 'NextRun')] = next_run
        return next_run

    def sync_ui(self, fallback=None, update=True):
        fallback = fallback or (
            datetime.now().replace(microsecond=0) + self.NO_FARM_RETRY
        )
        times = [
            self._sync_city_ui(city_key, city, fallback)
            for city_key, city in CITY_KEY_TO_NAME.items()
            if bool(self.config.cross_get(
                keys=self._farm_key(city_key, 'Enable'), default=True
            ))
        ]
        if update:
            self.config.update()
        return times

    def run(self):
        logger.hr('JIANGNAN FARM PRODUCTION', level=0)
        inventory_state = JiangnanInventoryState(self.config)
        now = datetime.now().replace(microsecond=0)
        inventory_state.apply_due_arrivals(now)
        future_times = []

        for city_key, city in CITY_KEY_TO_NAME.items():
            now = datetime.now().replace(microsecond=0)
            if production_should_yield_to_dock(inventory_state):
                logger.info(
                    'Yield farm production to arrived dock shipment'
                )
                inventory_state.sync_dock_schedule()
                break
            if not bool(getattr(self.config, f'farm_{city_key}_Enable')):
                continue
            next_run = self._parse_time(
                getattr(self.config, f'farm_{city_key}_NextRun')
            ) or self.DEFAULT_TIME
            if next_run > now:
                future_times.append(next_run)
                continue

            buildings = self.data.get('cities', {}).get(city, {}).get('buildings', {})
            if cached_production_group(buildings, inventory_state, city, (self.RESOURCE,), now) is not None:
                next_run = self._sync_city_ui(city_key, city, now + self.NO_FARM_RETRY)
            elif not self.switch_city(city):
                next_run = datetime.now().replace(microsecond=0) + self.RETRY
            else:
                self.handle_building_map_interference()
                buildings = self.inspect_city(city, inventory_state)
                if buildings is None:
                    next_run = datetime.now().replace(microsecond=0) + self.RETRY
                else:
                    next_run = self._sync_city_ui(
                        city_key,
                        city,
                        datetime.now().replace(microsecond=0) + self.NO_FARM_RETRY,
                    )
            self.config.modified[self._farm_key(city_key, 'NextRun')] = next_run
            self.config.update()
            if bool(self.config.cross_get(
                keys=self._farm_key(city_key, 'Enable'), default=True
            )):
                future_times.append(next_run)
            else:
                logger.info(
                    f'Farm production paused in {city}; '
                    'exclude it from task countdown'
                )
            logger.attr(f'FarmNextRun_{city}', next_run)

        # 未到期的城市也需要保持状态文本，且 UI 中始终列出全部城市。
        synced = self.sync_ui(update=True)
        future_times.extend(synced)
        if future_times:
            self.config.task_delay(target=min(future_times))
        else:
            self.config.task_delay(success=True)
        return True
