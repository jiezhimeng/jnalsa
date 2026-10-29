import time
from collections import Counter
from datetime import datetime, timedelta

import cv2
import numpy as np

from module.base.button import Button
from module.config.config import TaskEnd
from module.exception import (
    GameBugError, GameNotRunningError, GamePageUnknownError,
    GameStuckError, GameTooManyClickError, RequestHumanTakeover, ScriptError,
)
from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.assets import (
    BUILDING_SEARCH_CLEAR_BUTTON,
    BUILDING_SEARCH_PAGE_CHECK,
)
from module.jiangnan.building import BuildingResultOcr
from module.jiangnan.city import CITY_KEY_TO_PRODUCTION_TASK
from module.jiangnan.farm_production import JiangnanFarmProduction
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.jiangnan.multi_product_production import (
    JiangnanMultiProductProduction,
)
from module.jiangnan.production import JiangnanProduction
from module.jiangnan.production_state import cached_production_group
from module.jiangnan.resource_layout import page_config, production_layout_page
from module.logger import logger


class JiangnanCityProduction(JiangnanFarmProduction):
    """进入一座城市一次，并连续处理该城全部已启用生产设施。"""

    RETRY = timedelta(minutes=10)
    IDLE_RETRY = timedelta(hours=24)
    LAYOUT_REFRESH_INTERVAL = timedelta(days=3)
    TASK_TO_CITY_KEY = {
        task: city_key
        for city_key, task in CITY_KEY_TO_PRODUCTION_TASK.items()
    }
    PRODUCTION_FILTER_BUTTON = (884, 620, 934, 661)
    PRODUCTION_FILTER_CHECKBOX = (499, 180, 536, 216)
    PRODUCTION_FILTER_RESET = (439, 588, 617, 641)
    PRODUCTION_FILTER_CONFIRM = (660, 588, 837, 641)
    PRODUCTION_GRID_X = (400, 555, 712, 868, 1025)

    def __init__(self, config, device=None, task_name=None):
        self.city_task = task_name or getattr(config, 'command', None)
        self.city_key = self.TASK_TO_CITY_KEY.get(self.city_task)
        if self.city_key is None:
            raise ValueError(
                f'Unknown Jiangnan city production task: {self.city_task!r}'
            )
        self.city = CITY_KEY_TO_NAME[self.city_key]
        self._production_layout_scanned_this_run = False
        self._production_city_empty = False
        self._production_visual_unreliable = False
        super().__init__(config=config, device=device)
        # JiangnanCity already owns the shared auxiliary-layout manager under
        # city_layout.  Keep the resource-oriented alias used by this module.
        self.resource_layout = self.city_layout

    @staticmethod
    def _facility_definitions():
        yield '农田', JiangnanFarmProduction.TASK, JiangnanFarmProduction
        for task, facility in JiangnanProduction.TASK_FACILITIES.items():
            yield facility, task, JiangnanProduction
        for task, facility in JiangnanMultiProductProduction.TASK_FACILITIES.items():
            yield facility, task, JiangnanMultiProductProduction

    def _facility_enabled(self, task):
        return bool(self.config.cross_get(
            keys=f'{task}.ProductionFacilitySettings.Enable',
            default=False,
        ))

    def _current_config(self, key, default=None):
        """读取本轮尚未提交的配置，避免设施扫描与到期判断使用不同状态。"""
        if key in self.config.modified:
            return self.config.modified[key]
        return self.config.cross_get(keys=key, default=default)

    def _layout_scan_deadline(self):
        next_scan = self._parse_time(
            self._current_config(self._layout_next_scan_key())
        )
        if next_scan is not None:
            return next_scan
        key = f'{self.city_task}.CityProductionStatus.LastLayoutScan'
        last_scan = self._parse_time(self._current_config(key))
        return ((last_scan + self.LAYOUT_REFRESH_INTERVAL)
                if last_scan is not None else self.DEFAULT_TIME)

    def _layout_next_scan_key(self):
        return ('JiangnanAuxiliaryProductionLayout.'
                f'production_layout_{self.city_key}.NextScan')

    def _layout_has_confirmed_no_facilities(self):
        key, _ = page_config(production_layout_page(self.city))
        return self._current_config(key) == '无生产设施'

    @staticmethod
    def _facility_group(task):
        return f'city_facility_{task}'

    def _city_paths(self, runner, task):
        if isinstance(runner, JiangnanMultiProductProduction):
            key = runner._production_key
        elif isinstance(runner, JiangnanProduction):
            key = runner._production_key
        else:
            key = runner._farm_key
        legacy_status = key(self.city_key, 'Status')
        legacy_next_run = key(self.city_key, 'NextRun')
        base = f'{self.city_task}.{self._facility_group(task)}'
        return (
            f'{base}.Enable',
            f'{base}.Status',
            f'{base}.NextRun',
            legacy_status,
            legacy_next_run,
        )

    def _sync_facility_switches(self, layout):
        """根据本城市生产辅助表同步页面中的全部设施开关。"""
        facilities = {
            facility
            for label in dict(layout or {})
            for facility in [self._facility_from_label(label)]
            if facility is not None
        }
        enabled = []
        disabled = []
        for facility, task, _ in self._facility_definitions():
            value = facility in facilities
            key = (
                f'{self.city_task}.'
                f'{self._facility_group(task)}.Enable'
            )
            self.config.modified[key] = value
            (enabled if value else disabled).append(facility)
        logger.attr('ProductionFacilityAutoEnabled', enabled)
        logger.attr('ProductionFacilityAutoDisabled', disabled)
        return facilities

    def _new_runner(self, task, facility, runner_class):
        if runner_class is JiangnanFarmProduction:
            return runner_class(config=self.config, device=self.device)
        return runner_class(
            config=self.config,
            device=self.device,
            facility=facility,
            task_name=task,
        )

    def _sync_facility_inventory(self, inventory_state, runner, facility, task, enable_key):
        """Skip the entire facility while maintained output stock is at 97%."""
        marker = self._auto_pause_marker(inventory_state, task)
        if marker is not None:
            self.config.modified[enable_key] = False
            status_key = enable_key.rsplit('.', 1)[0] + '.Status'
            self.config.modified[status_key] = (
                f'[{"/".join(marker.get("resources") or ())}][库存达到97%][已自动暂停]'
            )
            return
        if not bool(self._current_config(enable_key, default=True)):
            return
        products = getattr(runner, 'products', None) or (
            getattr(runner, 'resource', None) or getattr(runner, 'RESOURCE', None),
        )
        products = tuple(resource for resource in products if resource)
        if not products:
            return
        states = inventory_state.production_threshold_states(self.city, products)
        # Unknown quantities do not justify closing a facility. Multi-product
        # facilities remain runnable while any product still needs production.
        if len(states) != len(products) or not all(state is False for state in states.values()):
            return
        if inventory_state._pause_production_target(
                self.city, products, task, 'farm' if facility == '农田' else 'production',
                self.city_key, reason='库存达到97%'):
            self.config.modified[enable_key] = False
            self.config.modified[enable_key.rsplit('.', 1)[0] + '.Status'] = (
                f'[{"/".join(products)}][库存达到97%][已自动暂停]'
            )
            inventory_state.save()

    @staticmethod
    def _button(rect, name):
        return Button(area=(), color=(), button=rect, name=name)

    def _production_filter_active(self):
        """复用公共建筑搜索层的筛选状态识别。"""
        return self.building_search_filter_active(
            log_name='ProductionFilterState'
        )

    def _production_filter_dialog_open(self, require_reset=True):
        """Recognize dialog actions; reset can flash white after a click."""
        image = self.device.image
        if image is None or image.size == 0:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        actions = [(690, 600, 820, 635)]
        if require_reset:
            actions.append((470, 600, 600, 635))
        for x1, y1, x2, y2 in actions:
            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                return False
            red = cv2.bitwise_or(
                cv2.inRange(patch, (0, 85, 70), (15, 255, 255)),
                cv2.inRange(patch, (165, 85, 70), (179, 255, 255)),
            )
            if cv2.countNonZero(red) / red.size < 0.45:
                return False
        return True

    def _clear_production_filter(self):
        """复用公共建筑搜索层，保证生产筛选也从默认状态开始。"""
        return self.ensure_building_search_unfiltered(
            log_name='ProductionFilterState'
        )

    def _clear_production_search_text(self):
        for attempt in range(3):
            # 清除 X 时输入法可能仍盖住底部漏斗。先关输入法，避免
            # 后续筛选点击落在输入法上而脚本误以为弹窗没打开。
            try:
                _, keyboard_shown = self.device.u2.current_ime()
            except Exception:
                keyboard_shown = False
            if keyboard_shown:
                logger.info('Close keyboard before production search cleanup')
                self.device.press_back()
                self.device.sleep(0.3)
            self.device.screenshot()
            if not self.is_building_search_page():
                logger.warning('Production search page closed during text cleanup')
                return False
            if not self.appear(
                BUILDING_SEARCH_CLEAR_BUTTON, offset=(4, 4), similarity=0.82
            ):
                return True
            logger.info(
                f'Clear retained production building search text: '
                f'{attempt + 1}/3'
            )
            self.device.click(self.anchored_button(
                BUILDING_SEARCH_PAGE_CHECK,
                BUILDING_SEARCH_CLEAR_BUTTON,
                name='PRODUCTION_SEARCH_CLEAR',
            ))
            self.device.sleep(0.35)
        self.device.screenshot()
        cleared = not self.appear(
            BUILDING_SEARCH_CLEAR_BUTTON, offset=(4, 4), similarity=0.82
        )
        if not cleared:
            logger.warning('Production search text remains after clear retries')
        return cleared

    def _restore_unfiltered_search(self):
        """城市生产结束后停留在无筛选、无搜索文字的默认搜索页。"""
        self.device.screenshot()
        # “没有满足当前筛选条件的建筑”出现时筛选弹窗不会自动关闭，
        # 此时先返回一次，才能回到没有 X 的普通搜索页。
        if self._production_city_empty:
            logger.info('Close empty production filter dialog')
            self.device.press_back()
            if not self._wait_search_page(timeout=5):
                logger.warning('Unable to close empty production filter dialog')
                return False
            self._production_city_empty = False
        if not self.is_building_search_page() and not self.goto_building_search():
            logger.warning('Unable to reopen search page for filter cleanup')
            return False
        if not self._clear_production_filter():
            return False
        if not self._clear_production_search_text():
            return False
        self.device.screenshot()
        clean = (
            self.is_building_search_page()
            and not self._production_filter_active()
        )
        logger.info(f'Production search page restored: clean={clean}')
        return clean

    def _open_production_results(self):
        """打开空搜索页，通过漏斗只显示当前城市的生产设施。"""
        if not self.goto_building_search():
            return False
        # 搜索界面可能沿用上次生产任务的筛选。先恢复成图2的默认状态，
        # 再开始本轮筛选，不能在已有筛选上继续叠加。
        if not self._clear_production_filter():
            return False
        # 搜索页可能保留上一设施的文字。这里只清空，不打开输入法。
        if not self._clear_production_search_text():
            return False
        # The bottom edge of the funnel's old click box sometimes misses;
        # then the reset click closes search and the rest runs on the map.
        for attempt in range(3):
            self.device.click(self._button(
                (896, 626, 919, 649), 'PRODUCTION_FILTER_BUTTON'
            ))
            self.device.sleep(0.45)
            self.device.screenshot()
            if self._production_filter_dialog_open():
                break
            logger.warning(f'Production filter dialog did not open: {attempt + 1}/3')
            if not self.is_building_search_page():
                return False
        else:
            return False
        # 每次先重置，保证不会把上一次保留的“生产”反向取消，也不会
        # 混入功能/商业等其他类型。
        self.device.click(self._button(
            self.PRODUCTION_FILTER_RESET, 'PRODUCTION_FILTER_RESET'
        ))
        self.device.sleep(0.2)
        self.device.screenshot()
        # The reset button flashes white while pressed. The confirm button
        # remains red and identifies the dialog during that animation.
        if not self._production_filter_dialog_open(require_reset=False):
            logger.warning('Production filter dialog disappeared after reset')
            return False
        self.device.click(self._button(
            self.PRODUCTION_FILTER_CHECKBOX,
            'PRODUCTION_FILTER_CHECKBOX',
        ))
        self.device.sleep(0.25)
        self.device.click(self._button(
            self.PRODUCTION_FILTER_CONFIRM, 'PRODUCTION_FILTER_CONFIRM'
        ))
        confirmed_at = time.monotonic()
        deadline = confirmed_at + 8
        confirm_retried = False
        while time.monotonic() < deadline:
            self.device.sleep(0.2)
            self.device.screenshot()
            # 没有任何生产设施时游戏停留在筛选弹窗，并显示一条会移动
            # 后消失的浮动提示。优先用图片匹配，不再依赖短时文字 OCR。
            if self.is_building_search_no_result(use_ocr=False):
                self._production_city_empty = True
                logger.info(
                    f'No production facilities in city: {self.city}'
                )
                return None
            if self.is_building_search_page():
                logger.info('Production-only building list opened')
                return True
            # 无结果时筛选弹窗会保持打开；确认点击偶尔也会落空，
            # 因此复点一次，随后仍停留在弹窗才认定为空列表。
            if self._production_filter_dialog_open(require_reset=False):
                elapsed = time.monotonic() - confirmed_at
                if not confirm_retried and elapsed >= 1.0:
                    logger.info('Production filter remains open; retry confirm')
                    self.device.click(self._button(
                        self.PRODUCTION_FILTER_CONFIRM,
                        'PRODUCTION_FILTER_CONFIRM_RETRY',
                    ))
                    confirmed_at = time.monotonic()
                    confirm_retried = True
                elif confirm_retried and elapsed >= 1.25:
                    self._production_city_empty = True
                    logger.info(
                        f'No production facilities in city: {self.city}; '
                        'filter dialog remained after confirm retry'
                    )
                    return None
        logger.warning('Production-only building list did not open')
        return False

    @classmethod
    def _label_areas(cls, ys):
        return [
            (x - 70, y, x + 70, y + 35)
            for y in ys for x in cls.PRODUCTION_GRID_X
        ]

    @classmethod
    def _visible_production_cells(cls, image, ys):
        """按名称条与上方纸张的色差排除当前画面的空白格。"""
        cells = []
        if image is None:
            return cells
        for row, y in enumerate(ys):
            if y < 20 or y + 24 >= image.shape[0]:
                continue
            for column, x in enumerate(cls.PRODUCTION_GRID_X):
                label = image[y + 5:y + 24, x - 65:x + 65]
                background = image[y - 18:y - 5, x - 65:x + 65]
                if label.size == 0 or background.size == 0:
                    continue
                contrast = np.median(label, axis=(0, 1)).astype(float) - \
                    np.median(background, axis=(0, 1)).astype(float)
                # Green/blue category bars may be darker than the paper above.
                # A positive-only contrast test silently drops real cards
                # such as 桐油种植园 while keeping their neighbors.
                # 长名称（如左侧贴边的“螺钿漆坊(2)”）会让整条的
                # 中位色被深色文字拉回纸张色；两端只要有一侧仍能看出
                # 名称条，就应送入 OCR，而不是直接当作空格。
                edge_contrast = []
                for left, right in ((x - 65, x - 45),
                                    (x + 40, x + 65)):
                    edge_label = image[y + 5:y + 24, left:right]
                    edge_background = image[y - 18:y - 5, left:right]
                    edge_contrast.append(float(np.max(np.abs(
                        np.median(edge_label, axis=(0, 1)).astype(float)
                        - np.median(edge_background, axis=(0, 1)).astype(float)
                    ))))
                if (float(np.max(np.abs(contrast))) >= 7
                        or max(edge_contrast) >= 12):
                    cells.append((row, column))
        return cells

    def _read_production_rows(self, ys):
        """一次 OCR 当前页有名称条的格子，不逐个重新截图。"""
        cells = self._visible_production_cells(self.device.image, ys)
        rows = [[''] * len(self.PRODUCTION_GRID_X) for _ in ys]
        if not cells:
            return [[] for _ in ys]
        areas = [
            (self.PRODUCTION_GRID_X[column] - 70, ys[row],
             self.PRODUCTION_GRID_X[column] + 70, ys[row] + 35)
            for row, column in cells
        ]
        ocr = BuildingResultOcr()
        ocr.buttons = [
            Button(
                area=area, color=(), button=area,
                name=f'PRODUCTION_GRID_LABEL_{index + 1}',
            )
            for index, area in enumerate(areas)
        ]
        raw = ocr.ocr(self.device.image)
        if not isinstance(raw, list):
            raw = [raw]
        for (row, column), value in zip(cells, raw):
            label = self.normalize_building_name(value)
            # 绍兴的“珍馐”在浅色名称条上偶尔被 OCR 读成“珍体”。
            rows[row][column] = label.replace('珍体食肆', '珍馐食肆')
        for row in rows:
            while row and not row[-1]:
                row.pop()
        logger.attr('ProductionGridRows', rows)
        return rows

    @classmethod
    def _visible_production_label_ys(cls, image):
        """从白色或彩色名称条找完整可见行，不推算滚动距离。"""
        if image is None or image.shape[0] < 590:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hits = []
        for column, x in enumerate(cls.PRODUCTION_GRID_X):
            # 常规建筑的浅色名称条没有纯白像素；在两端取样避开文字。
            edges = image[125:590, x - 68:x - 48]
            edges = cv2.hconcat((edges, image[125:590, x + 48:x + 68]))
            pale = (
                (edges[:, :, 0] >= 188)
                & (edges[:, :, 1] >= 178)
                & (edges[:, :, 2] >= 165)
            ).mean(axis=1) >= 0.6
            starts = [index for index, value in enumerate(pale)
                      if value and (index == 0 or not pale[index - 1])]
            ends = [index for index, value in enumerate(pale)
                    if value and (index + 1 == len(pale) or not pale[index + 1])]
            for start, end in zip(starts, ends):
                if 10 <= end - start + 1 <= 37:
                    hits.append((125 + start - 3, False, column, True))
                elif 4 <= end - start + 1 < 10:
                    # 较长的建筑名会压到两端取样区，把完整白条切成
                    # 短段。短段只能为相邻的完整白条作旁证。
                    hits.append((125 + start - 3, False, column, False))

            # 生产分类可能将名称条染成绿色，仍保留彩色条检测。
            strip = hsv[115:585, x - 50:x + 50]
            green = cv2.inRange(strip, (35, 25, 110), (85, 95, 240))
            coverage = (green > 0).mean(axis=1)
            active = coverage >= 0.55
            starts = [index for index, value in enumerate(active)
                      if value and (index == 0 or not active[index - 1])]
            ends = [index for index, value in enumerate(active)
                    if value and (index + 1 == len(active) or not active[index + 1])]
            for start, end in zip(starts, ends):
                if 17 <= end - start + 1 <= 45:
                    top = 115 + start
                    # 名称条也有文字、阴影和纸纹；阈值过严会把整行绿色
                    # 建筑名排除。连续 17-45 行的横向绿色覆盖率已是主判据，
                    # 这里仅排除色彩起伏极大的树冠等明显误报。
                    colored_edges = cv2.hconcat((
                        image[top:top + 20, x - 50:x - 25],
                        image[top:top + 20, x + 25:x + 50],
                    ))
                    label_pixels = image[
                        top + 5:top + 24, x - 65:x + 65
                    ]
                    above_pixels = image[
                        top - 18:top - 5, x - 65:x + 65
                    ]
                    color_step = np.max(np.abs(
                        np.median(label_pixels, axis=(0, 1)).astype(float)
                        - np.median(above_pixels, axis=(0, 1)).astype(float)
                    ))
                    if (colored_edges.std(axis=(0, 1)).mean() < 45
                            and color_step <= 45):
                        hits.append((top - 3, True, column, True))
        groups = []
        for hit in sorted(hits):
            if groups and hit[0] - groups[-1][-1][0] <= 12:
                groups[-1].append(hit)
            else:
                groups.append([hit])
        ys = []
        for group in groups:
            row_y = int(round(float(np.median([hit[0] for hit in group]))))
            if (any(hit[1] for hit in group)
                    or (any(hit[3] for hit in group)
                        and len({hit[2] for hit in group}) >= 2)):
                ys.append(row_y)
                continue
            # 最后一行可能只有左侧一个建筑。单格浅色条需经格子
            # 前景/背景对比再确认，不能因缺少第二格就丢掉整行。
            if (any(hit[3] and hit[2] == 0 for hit in group)
                    and not any(abs(row_y - known_y) < 35 for known_y in ys)
                    and (0, 0) in cls._visible_production_cells(image, [row_y])):
                ys.append(row_y)
        # 纸纹或文字会把同一条浅色名称条切成两段；相邻建筑行相隔
        # 约 195px，35px 内的候选实际属于同一行。
        ys = [y for index, y in enumerate(ys)
              if index == 0 or y - ys[index - 1] >= 35]
        logger.attr('ProductionVisibleLabelYs', ys)
        return ys

    def _read_visible_production_rows(self):
        ys = self._visible_production_label_ys(self.device.image)
        if not ys:
            logger.warning('Production label rows not visible on current page')
            return [], []
        return ys, self._read_production_rows(ys=ys)

    def _production_row_offset(self, known_rows, visible):
        """用相同列的设施名匹配新旧画面的重叠行。"""
        scores = {}
        for current_index, current in enumerate(visible):
            for known_index, known in enumerate(known_rows):
                score = 0
                for column, actual in enumerate(current):
                    expected = known[column] if column < len(known) else ''
                    if not actual or not expected:
                        continue
                    if actual == expected:
                        score += 3
                    elif (self._facility_from_label(actual) is not None
                          and self._facility_from_label(actual)
                          == self._facility_from_label(expected)):
                        score += 1
                if score:
                    offset = known_index - current_index
                    scores[offset] = scores.get(offset, 0) + score
        if not scores:
            return None
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
            logger.warning(f'Ambiguous production row overlap: {ranking}')
            return None
        return ranking[0][0]

    def _production_drag_x(self, visible=None):
        """Pick an actually occupied card column for the scroll gesture."""
        for values in reversed(list(visible or [])):
            for column, label in enumerate(values):
                if self._facility_from_label(label) is not None:
                    return self.PRODUCTION_GRID_X[column]
        return self.PRODUCTION_GRID_X[0]

    def _scroll_production_results(self, visible=None, upward=False):
        """短距离翻页；手势位移不参与行号或点击坐标计算。"""
        self._city_list_search_seq += 1
        drag_x = self._production_drag_x(visible)
        logger.attr(
            'ProductionVisualDrag',
            f'x={drag_x}, upward={upward}, re-detect rows after gesture',
        )
        # 设施行间距约 195px。175px 的手势可减少重复翻页，同时
        # 保留至少一行名称重叠；行号仍由画面内容而非位移计算。
        start, end = ((375, 550) if upward else (550, 375))
        self.jiangnan_drag(
            (drag_x, start), (drag_x, end),
            name=(
                f'PRODUCTION_GRID_SCROLL_'
                f'{self._city_list_search_seq:04d}'
            ),
        )
        self.device.sleep(0.65)

    def _scan_production_layout(self, enable_city_task=False):
        """按画面可见名称条逐页建立本城市设施辅助表。"""
        # 一轮城市任务最多完整遍历一次；失败时不能把不完整扫描
        # 写入辅助表，也不能为每个缺失设施反复遍历整张列表。
        self._production_layout_scanned_this_run = True
        opened = self._open_production_results()
        if opened is None and self._production_city_empty:
            page = production_layout_page(self.city)
            # 即使暂时没有设施，城市任务也要保留三天一次的复查。
            self._sync_facility_switches({})
            scanned_at = datetime.now().replace(microsecond=0)
            next_scan = scanned_at + self.LAYOUT_REFRESH_INTERVAL
            if enable_city_task:
                self.config.modified[f'{self.city_task}.Scheduler.Enable'] = True
                self.config.modified[f'{self.city_task}.Scheduler.NextRun'] = next_scan
            self.config.modified[
                f'{self.city_task}.CityProductionStatus.LastLayoutScan'
            ] = scanned_at
            self.config.modified[
                self._layout_next_scan_key()
            ] = next_scan
            self.resource_layout.replace_status(page, '无生产设施')
            logger.info(
                f'Production layout is empty; rescan in 3 days: '
                f'city={self.city}'
            )
            return {}
        if not opened:
            return None
        self.device.screenshot()
        visible_ys, visible = self._read_visible_production_rows()
        if not visible or not any(visible):
            self.save_abnormal_screenshot('production_rows_missing')
            self._production_visual_unreliable = True
            return None
        rows = list(visible)
        stalled_pages = 0
        # A short drag is only a page-turn request. All row positions and
        # global indices come from the next screenshot and name overlap.
        for page_index in range(24):
            last_row = rows[-1]
            visible_cells = self._visible_production_cells(
                self.device.image, visible_ys
            )
            occupied_columns = [
                column for visual_row, column in visible_cells
                if visual_row == len(visible) - 1
            ]
            if not occupied_columns:
                logger.warning('Production last row has no visible name bars')
                self.save_abnormal_screenshot('production_last_row_missing')
                self._production_visual_unreliable = True
                return None
            if any(column >= len(last_row) or not last_row[column]
                   for column in occupied_columns):
                logger.warning('Production last row has unreadable building names')
                self.save_abnormal_screenshot('production_names_unreadable')
                self._production_visual_unreliable = True
                return None
            # 底部刚露出的短行可能只有左侧名称条进入检测区域。
            # 先将这一行翻到画面中部，再据完整可见的名称条判断结尾。
            if max(occupied_columns) < 4 and visible_ys[-1] <= 460:
                break
            previous_ys = visible_ys
            self._scroll_production_results(visible=visible)
            self.device.screenshot()
            visible_ys, visible = self._read_visible_production_rows()
            if not visible:
                logger.warning('Lost production labels after page turn')
                self.save_abnormal_screenshot('production_labels_lost')
                self._production_visual_unreliable = True
                return None
            offset = self._production_row_offset(rows, visible)
            if offset is None or offset < 0 or offset > len(rows):
                logger.warning('Unable to anchor production page; discard partial scan')
                self.save_abnormal_screenshot('production_page_anchor_failed')
                self._production_visual_unreliable = True
                return None
            changed = False
            for index, values in enumerate(visible):
                target_row = offset + index
                if target_row >= len(rows):
                    if target_row != len(rows):
                        logger.warning('Production page has an unobserved row')
                        return None
                    rows.append(values)
                    changed = True
                else:
                    merged = list(rows[target_row])
                    if len(merged) < len(values):
                        merged.extend([''] * (len(values) - len(merged)))
                    for column, value in enumerate(values):
                        if value and not merged[column]:
                            merged[column] = value
                            changed = True
                    while merged and not merged[-1]:
                        merged.pop()
                    rows[target_row] = merged
            logger.attr('ProductionGridAnchor',
                        f'offset={offset}, changed={changed}, page={page_index + 1}')
            stalled_pages = 0 if changed else stalled_pages + 1
            if stalled_pages >= 2 and visible_ys == previous_ys:
                # 列表滑到底时，最后一行可能固定在 y≈533，无法再被
                # 拖到 460 以上。两次翻页后名称和位置都不再变化，就以
                # 已验证可读的最后一行结束扫描并保存。
                break
        else:
            logger.warning('Production grid did not reach the end within 24 turns')
            return None

        # 筛选结果按从左到右连续排布。中间行或末行的非尾部出现
        # 空格，说明有名称条/OCR 漏读；不能据此关闭本城设施开关。
        for row_index, names in enumerate(rows):
            if (row_index < len(rows) - 1 and len(names) != 5) or \
                    any(not name for name in names):
                logger.warning(
                    f'Production row {row_index + 1} has unreadable cells; '
                    'keep existing layout'
                )
                self.save_abnormal_screenshot('production_row_incomplete')
                self._production_visual_unreliable = True
                return None

        ignored = sorted({
            name
            for names in rows
            for name in names
            if name and self._facility_from_label(name) is None
        })
        if ignored:
            logger.attr('ProductionFacilitiesIgnored', ignored)
        recognized = [
            (name, (row, column))
            for row, names in enumerate(rows)
            for column, name in enumerate(names)
            # 龙宫食品铺、珊瑚工坊等尚未列入生产配置的设施不应
            # 进入辅助信息；对应现有设施开关保持关闭，等同未开放。
            if name and self._facility_from_label(name) is not None
        ]
        counts = Counter(name for name, _ in recognized)
        seen = Counter()
        layout = {}
        for name, position in recognized:
            seen[name] += 1
            # 未标等级的同名建筑会占多格，例如两个“印刷厂”。
            # 不能用原名作为字典键，否则后一个会覆盖前一个。
            label = (f'{name}第{seen[name]}座'
                     if counts[name] > 1 else name)
            layout[label] = position
        if not layout:
            logger.warning(f'No production buildings recognized in {self.city}')
            return None
        self._sync_facility_switches(layout)
        if enable_city_task:
            self.config.modified[
                f'{self.city_task}.Scheduler.Enable'
            ] = True
        scanned_at = datetime.now().replace(microsecond=0)
        self.config.modified[
            f'{self.city_task}.CityProductionStatus.LastLayoutScan'
        ] = scanned_at
        self.config.modified[
            self._layout_next_scan_key()
        ] = scanned_at + self.LAYOUT_REFRESH_INTERVAL
        self.resource_layout.replace_layout(
            production_layout_page(self.city), layout
        )
        logger.info(
            f'Production layout scanned: city={self.city}, '
            f'rows={len(rows)}, buildings={len(layout)}'
        )
        self._production_visual_unreliable = False
        return layout

    @staticmethod
    def _label_matches_facility(label, facility):
        label = str(label or '')
        if not label or facility not in label:
            return False
        # OCR 垃圾页经常把单个汉字、数字或残缺字符误拼成设施名片段。
        # 只有包含完整设施关键词，并且周围仍有中文/等级信息时才认为
        # 是有效建筑，避免滚动到底部后把空白页当成生产列表继续恢复。
        if not any('一' <= char <= '鿿' for char in label):
            return False
        if facility == '农田':
            return '农田' in label and '改良农田' not in label
        # 特产铺、雕刻行等搜索结果没有等级前缀或数量后缀；
        # 完整设施名本身就是有效标签。
        if label == facility:
            return True
        return len(label) >= len(facility) + 1

    def _production_layout_target(self, facility):
        layout = self.resource_layout.load(production_layout_page(self.city))
        candidates = [
            (position, label)
            for label, position in layout.items()
            if self._label_matches_facility(label, facility)
        ]
        return min(candidates, default=None)

    def _facility_from_label(self, label):
        facilities = sorted(
            (item[0] for item in self._facility_definitions()),
            key=len,
            reverse=True,
        )
        return next(
            (name for name in facilities
             if self._label_matches_facility(label, name)),
            None,
        )

    def _locate_production_facility(
        self, facility, handle_interference=True,
    ):
        """Find each facility by name; filtering is only for the initial scan."""
        self._production_cleanup_blocked = False
        if facility == '农田':
            validator = self._is_standard_farm_result
            rejected = lambda value: '改良农田' in value
        else:
            validator = lambda value: self._label_matches_facility(value, facility)
            rejected = None
        if not self.search_building(
            facility,
            result_validator=validator,
            rejected_is_not_found=rejected,
            exact=True,
            handle_interference=handle_interference,
        ):
            self._production_cleanup_blocked = bool(
                getattr(self, 'last_harvest_blocked_reason', None)
            )
            return None
        return self.last_search_result_name

    def _transfer_centered_building_anchor(self, runner):
        """Pass the current map anchor to a facility runner sharing this device."""
        runner._retained_search_arrow = self._retained_search_arrow
        runner._retained_search_arrow_size = self._retained_search_arrow_size
        frame = self._retained_search_frame
        runner._retained_search_frame = (
            frame.copy() if frame is not None else None
        )

    def _inspect_facility(
        self, runner, inventory_state, now,
        status_key, next_run_key, legacy_status_key,
        prelocated_result=None,
    ):
        if prelocated_result:
            # 搜索与设施检查使用不同的实例，但共享同一张地图和设备。
            # 居民事件处理完后箭头往往已过期，必须把已验证的锚点交给
            # 设施实例，否则它会误判定位丢失并重新搜索。
            self._transfer_centered_building_anchor(runner)
        if isinstance(runner, JiangnanProduction):
            complete = runner.inspect_city(
                self.city_key, self.city, inventory_state,
                prelocated_result=prelocated_result,
            )
            retry_after = (
                runner.NO_FARM_RETRY if complete else runner.RETRY
            )
        else:
            complete = runner.inspect_city(
                self.city, inventory_state,
                prelocated_result=prelocated_result,
            )
            retry_after = (
                runner.NO_FARM_RETRY
                if complete is not None else runner.RETRY
            )
        fallback = datetime.now().replace(microsecond=0) + retry_after
        next_run = runner._sync_city_ui(
            self.city_key, self.city, fallback
        )
        status = self.config.modified.get(
            legacy_status_key,
            self.config.cross_get(keys=legacy_status_key, default='无'),
        )
        self.config.modified[status_key] = status
        self.config.modified[next_run_key] = next_run
        return next_run

    def _set_city_status(self, lines, next_run, now):
        base = f'{self.city_task}.CityProductionStatus'
        enabled = sum('设施总开关关闭' not in line for line in lines)
        self.config.modified[f'{base}.Status'] = (
            f'已更新 {len(lines)} 个设施模块，其中 {enabled} 个设施已启用；'
            '详细状态见下方各设施模块'
        )
        self.config.modified[f'{base}.LastRun'] = now
        self.config.modified[f'{self.city_task}.Scheduler.NextRun'] = next_run

    @staticmethod
    def _format_status(status):
        text = str(status or '无').strip()
        return '；'.join(
            part.strip() for part in text.splitlines() if part.strip()
        ) or '无'

    def _facility_line(self, facility, status, next_run=None):
        status = self._format_status(status)
        if next_run is None:
            time_text = '--'
        else:
            time_text = self._time_text(next_run)
        return f'【{facility}】【{status}】【下次：{time_text}】'

    def _runner_buildings(self, runner, facility):
        city_data = getattr(runner, 'data', {}).get('cities', {}).get(
            self.city, {}
        )
        if facility == '农田':
            return city_data.get('buildings', {})
        return (
            city_data.get('facilities', {}).get(facility, {})
            .get('buildings', {})
        )

    def _runner_has_active_work(self, runner, facility):
        return any(
            item.get('state') in ('working', 'upgrading', 'overflow')
            for item in self._runner_buildings(runner, facility).values()
        )

    def _reuse_pending_facility_clock(
        self, runner, facility, status_key, next_run_key, legacy_status_key, inventory_state,
    ):
        """Repair stale scheduler clocks before any city/search navigation."""
        resources = getattr(runner, 'products', None) or (
            getattr(runner, 'resource', None) or getattr(runner, 'RESOURCE', None),
        )
        if cached_production_group(
            self._runner_buildings(runner, facility), inventory_state, self.city, resources,
        ) is None:
            return False
        next_run = runner._sync_city_ui(
            self.city_key, self.city,
            datetime.now().replace(microsecond=0) + self.RETRY,
        )
        self.config.modified[status_key] = self._current_config(
            legacy_status_key, default='无',
        )
        self.config.modified[next_run_key] = next_run
        logger.info(f'Reuse working facility clock without search: {self.city} {facility}')
        return True

    def _auto_pause_marker(self, inventory_state, task):
        return next((
            marker for marker in inventory_state.data.get(
                'overflow_paused_production', []
            )
            if marker.get('city') == self.city
            and marker.get('task') == task
        ), None)

    def _defer_auto_paused_facility(
        self, marker, status_key, next_run_key, now,
    ):
        resources = '/'.join(marker.get('resources') or ())
        status = f'[{resources}][库存达到97%][已自动暂停]'
        next_run = now + self.IDLE_RETRY
        self.config.modified[status_key] = status
        self.config.modified[next_run_key] = next_run
        return status, next_run

    def _next_worker_release(self, inventory_state, records, now):
        """Earliest known completion that may release a city worker."""
        candidates = []
        data = getattr(inventory_state, 'data', {})
        for item in data.get('scheduled_changes', []):
            if (item.get('city') != self.city
                    or item.get('credited_at')
                    or item.get('reason') != '生产'):
                continue
            finish = self._parse_time(item.get('execute_at'))
            if finish is not None and finish > now:
                candidates.append(finish)
        for item in data.get('shipments', []):
            if (item.get('source') != self.city
                    or item.get('status') != 'transporting'):
                continue
            finish = self._parse_time(item.get('finish_at'))
            if finish is not None and finish > now:
                candidates.append(finish)
        for facility, runner, *_ in records:
            if runner is None:
                continue
            for item in self._runner_buildings(runner, facility).values():
                if item.get('state') not in ('working', 'upgrading'):
                    continue
                finish = self._parse_time(item.get('finish_at'))
                if finish is not None and finish > now:
                    candidates.append(finish)
        # Merchant and canal jobs also occupy characters, but are scheduled
        # outside the city-production runner. Only use clocks whose status
        # explicitly says the job is in progress.
        external = {
            '徽州府': (
                'JiangnanHuizhouMerchantSale',
                tuple(f'huizhou_merchant_hall_{i}' for i in range(1, 5)),
                '销售中',
            ),
            '扬州府': (
                'JiangnanYangzhouGrandCanal',
                ('yangzhou_grand_canal',),
                '进行中',
            ),
            '宁波府': (
                'JiangnanNingboMerchantShip',
                tuple(f'ningbo_merchant_ship_{i}' for i in range(1, 5)),
                '探索中',
            ),
        }.get(self.city)
        if external and getattr(self, 'config', None) is not None:
            task, groups, marker = external
            if self._current_config(f'{task}.Scheduler.Enable', False):
                for group in groups:
                    prefix = f'{task}.{group}'
                    if not self._current_config(f'{prefix}.Enable', True):
                        continue
                    status = str(self._current_config(
                        f'{prefix}.Status', ''
                    ))
                    if marker not in status:
                        continue
                    finish = self._parse_time(self._current_config(
                        f'{prefix}.NextRun', None
                    ))
                    if finish is not None and finish > now:
                        candidates.append(finish)
        return min(candidates) if candidates else now + self.RETRY

    def run(self):
        logger.hr(f'JIANGNAN CITY PRODUCTION: {self.city}', level=0)
        now = datetime.now().replace(microsecond=0)
        inventory_state = JiangnanInventoryState(self.config)
        inventory_state.apply_due_arrivals(now)
        if hasattr(self, '_sync_facility_inventory'):
            inventory_state.resume_overflow_production(
                inventory_state.load_inventories((self.city,)),
                inventory_state.load_capacities((self.city,)),
                inventory_state.special_capacities(), now=now, update=False,
            )
        worker_shortages = inventory_state.data.setdefault(
            'worker_shortages', {}
        )
        workers_blocked_until = self._parse_time(
            worker_shortages.get(self.city)
        )
        if workers_blocked_until is not None and workers_blocked_until <= now:
            worker_shortages.pop(self.city, None)
            inventory_state.save()
            workers_blocked_until = None

        status_lines = []
        future_times = []
        records = []
        due_records = []
        dock_yield_until = None
        due_dock = inventory_state.due_dock_shipments()
        if due_dock and inventory_state.request_dock_harvest(shipments=due_dock):
            dock_yield_until = datetime.now().replace(microsecond=0) + timedelta(minutes=1)
            logger.info('Dock shipment already arrived; defer city navigation')

        configured_layout = self.resource_layout.load(
            production_layout_page(self.city)
        )
        switch_ok = True
        city_ready = False
        production_page_used = False
        if configured_layout:
            self._sync_facility_switches(configured_layout)
            if (self._layout_scan_deadline() <= now
                    and workers_blocked_until is None
                    and dock_yield_until is None):
                switch_ok = self.switch_city(self.city)
                city_ready = switch_ok
                if switch_ok:
                    self.handle_building_map_interference()
                    production_page_used = True
                    logger.info(f'Refresh production layout after 3 days: {self.city}')
                    self._scan_production_layout()
        else:
            # 空辅助表不能让全部自动开关保持关闭后直接结束，否则首次
            # 运行永远没有机会建立该城市的设施清单。
            if (self._layout_has_confirmed_no_facilities()
                    and self._layout_scan_deadline() > now):
                self._production_city_empty = True
            elif workers_blocked_until is None and dock_yield_until is None:
                switch_ok = self.switch_city(self.city)
                city_ready = switch_ok
                if switch_ok:
                    self.handle_building_map_interference()
                    production_page_used = True
                    self._scan_production_layout()

        # 每座城市始终列出全部生产设施。设施自己的隐藏 NextRun 是独立
        # 时钟；未到期的设施只展示状态，不会因为别的设施到期而被重扫。
        for facility, task, runner_class in self._facility_definitions():
            if not self._facility_enabled(task):
                base = f'{self.city_task}.{self._facility_group(task)}'
                records.append((
                    facility, None, None,
                    f'{base}.Status', f'{base}.NextRun',
                    task, None, None,
                ))
                continue
            runner = self._new_runner(task, facility, runner_class)
            (
                enable_key, status_key, next_run_key,
                legacy_status_key, legacy_next_run_key,
            ) = self._city_paths(runner, task)
            record = (
                facility, runner, enable_key, status_key, next_run_key, task,
                legacy_status_key, legacy_next_run_key,
            )
            records.append(record)
            if hasattr(self, '_sync_facility_inventory'):
                self._sync_facility_inventory(inventory_state, runner, facility, task, enable_key)
            if not bool(self._current_config(enable_key, default=True)):
                continue
            self._reuse_pending_facility_clock(
                runner, facility, status_key, next_run_key, legacy_status_key, inventory_state,
            )
            facility_next = self._parse_time(self._current_config(
                next_run_key,
                default=self.DEFAULT_TIME,
            )) or self.DEFAULT_TIME
            if (facility_next <= now
                    and self._auto_pause_marker(inventory_state, task) is None):
                due_records.append(record)

        if (due_records and workers_blocked_until is None
                and dock_yield_until is None):
            if not city_ready:
                switch_ok = self.switch_city(self.city)
                city_ready = switch_ok
                if switch_ok:
                    self.handle_building_map_interference()
            if switch_ok and not self.resource_layout.load(
                production_layout_page(self.city)
            ) and not getattr(self, '_production_layout_scanned_this_run', False):
                production_page_used = True
                self._scan_production_layout()

        for (
            facility, runner, enable_key, status_key, next_run_key, task,
            legacy_status_key, legacy_next_run_key,
        ) in records:
            # Earlier facilities can take several minutes. Retry clocks and
            # due checks must use this facility's actual start, not city entry.
            now = datetime.now().replace(microsecond=0)
            if runner is None:
                self.config.modified[status_key] = '设施总开关关闭'
                self.config.modified[next_run_key] = self.DEFAULT_TIME
                status_lines.append(self._facility_line(
                    facility, '设施总开关关闭', None
                ))
                continue

            # Earlier facilities can update this city's maintained inventory.
            # Check again before opening the next facility, not just at entry.
            if hasattr(self, '_sync_facility_inventory'):
                self._sync_facility_inventory(inventory_state, runner, facility, task, enable_key)
            city_enabled = bool(self.config.modified.get(
                enable_key,
                self.config.cross_get(keys=enable_key, default=True),
            ))
            status = self.config.modified.get(
                status_key,
                self.config.cross_get(keys=status_key, default='未运行'),
            )
            if not city_enabled:
                status_lines.append(self._facility_line(
                    facility, status or '该城市已暂停', None
                ))
                continue

            pause_marker = self._auto_pause_marker(inventory_state, task)
            if pause_marker is not None:
                status, facility_next = self._defer_auto_paused_facility(
                    pause_marker, status_key, next_run_key, now,
                )
                status_lines.append(self._facility_line(
                    facility, status, facility_next,
                ))
                future_times.append(facility_next)
                continue

            facility_next = self._parse_time(self.config.modified.get(
                next_run_key,
                self.config.cross_get(
                    keys=next_run_key,
                    default=self.DEFAULT_TIME,
                ),
            )) or self.DEFAULT_TIME
            if workers_blocked_until is not None:
                active_work = self._runner_has_active_work(runner, facility)
                facility_next = max(facility_next, workers_blocked_until)
                if not active_work:
                    status = '人物不足，等待该城任务结束'
                    self.config.modified[status_key] = status
                self.config.modified[next_run_key] = facility_next
                if legacy_next_run_key is not None:
                    self.config.modified[legacy_next_run_key] = facility_next
                status_lines.append(self._facility_line(
                    facility, status, facility_next
                ))
                future_times.append(facility_next)
                continue
            if facility_next > now:
                status_lines.append(self._facility_line(
                    facility, status, facility_next
                ))
                future_times.append(facility_next)
                continue

            if dock_yield_until is None:
                due_dock = inventory_state.due_dock_shipments()
                if due_dock and inventory_state.request_dock_harvest(shipments=due_dock):
                    dock_yield_until = datetime.now().replace(microsecond=0) + timedelta(minutes=1)
                    logger.info('Yield remaining city production to arrived dock shipment')
            if dock_yield_until is not None:
                # The facility runner can already yield between buildings.
                # Do not then search every remaining facility before letting
                # the scheduler run the dock task. Advance all due clocks so
                # config migration cannot immediately requeue this city.
                facility_next = dock_yield_until
                self.config.modified[next_run_key] = facility_next
                if legacy_next_run_key is not None:
                    self.config.modified[legacy_next_run_key] = facility_next
                status_lines.append(self._facility_line(facility, status, facility_next))
                future_times.append(facility_next)
                continue

            # A clock can become due while building the city task's records.
            # It was absent from due_records at entry, so switch before search.
            if not city_ready and switch_ok:
                switch_ok = self.switch_city(self.city)
                city_ready = switch_ok
                if switch_ok:
                    self.handle_building_map_interference()
            if not switch_ok:
                facility_next = now + self.RETRY
                status = '城市切换失败，等待重试'
                self.config.modified[next_run_key] = facility_next
                self.config.modified[status_key] = status
                status_lines.append(self._facility_line(
                    facility, status, facility_next
                ))
                future_times.append(facility_next)
                continue

            if self._production_city_empty:
                facility_next = now + self.IDLE_RETRY
                status = '该城市没有生产设施，等待三天后复查'
                self.config.modified[next_run_key] = facility_next
                self.config.modified[status_key] = status
                status_lines.append(self._facility_line(
                    facility, status, facility_next
                ))
                future_times.append(facility_next)
                continue

            logger.hr(f'{self.city} - {facility}', level=1)
            try:
                production_page_used = True
                self._production_cleanup_blocked = False
                prelocated_result = self._locate_production_facility(facility)
                if self._production_cleanup_blocked:
                    facility_next = datetime.now().replace(microsecond=0) + self.RETRY
                    status = '仓满收获未完成，稍后复查'
                    self.config.modified[next_run_key] = facility_next
                    self.config.modified[status_key] = status
                    status_lines.append(self._facility_line(
                        facility, status, facility_next,
                    ))
                    future_times.append(facility_next)
                    continue
                # Initial facility discovery can disable missing facilities.
                # Subsequent facility navigation only searches by name.
                if not bool(self.config.modified.get(
                    enable_key,
                    self.config.cross_get(keys=enable_key, default=False),
                )):
                    facility_next = now + self.IDLE_RETRY
                    status = '辅助信息更新后未发现该设施'
                    self.config.modified[next_run_key] = facility_next
                    self.config.modified[status_key] = status
                    status_lines.append(self._facility_line(
                        facility, status, facility_next
                    ))
                    continue
                facility_next = self._inspect_facility(
                    runner, inventory_state, now,
                    status_key, next_run_key, legacy_status_key,
                    prelocated_result=prelocated_result,
                )
                if getattr(runner, 'last_no_workers', False):
                    release_at = self._next_worker_release(
                        inventory_state, records,
                        datetime.now().replace(microsecond=0),
                    )
                    workers_blocked_until = release_at
                    worker_shortages[self.city] = self._time_text(release_at)
                    inventory_state.save()
                    facility_next = release_at
                    self.config.modified[next_run_key] = release_at
                    if legacy_next_run_key is not None:
                        self.config.modified[legacy_next_run_key] = release_at
                    logger.warning(
                        f'{self.city} workers unavailable; defer remaining '
                        f'production until {release_at}'
                    )
            except (
                TaskEnd, GameBugError, GameNotRunningError,
                GamePageUnknownError, GameStuckError, GameTooManyClickError,
                RequestHumanTakeover, ScriptError,
            ):
                # 交给顶层调度器处理重启、停止和人工接管，不能继续点击。
                raise
            except Exception as exc:
                logger.exception(
                    f'Failed to process {facility} in {self.city}: {exc}'
                )
                facility_next = datetime.now().replace(microsecond=0) + self.RETRY
                self.config.modified[next_run_key] = facility_next
                self.config.modified[status_key] = '处理异常，等待重试'

            pause_marker = self._auto_pause_marker(inventory_state, task)
            if pause_marker is not None:
                self._defer_auto_paused_facility(
                    pause_marker, status_key, next_run_key, now,
                )
                facility_next = now + self.IDLE_RETRY

            status = self.config.modified.get(
                status_key,
                self.config.cross_get(keys=status_key, default='无'),
            )
            status_lines.append(self._facility_line(
                facility, status, facility_next
            ))
            # 仓满或确认无建筑会关闭该城市设施，不应继续参与倒计时。
            city_enabled = self.config.modified.get(
                enable_key,
                self.config.cross_get(keys=enable_key, default=True),
            )
            if bool(city_enabled):
                future_times.append(facility_next)

        if production_page_used and switch_ok:
            # 所有到期设施处理完以后显式取消生产筛选。下一项功能进入
            # 搜索页时看到的必须是没有 X 的默认建筑列表。
            self._restore_unfiltered_search()

        now = datetime.now().replace(microsecond=0)
        layout_deadline = self._layout_scan_deadline()
        if layout_deadline <= now:
            # Failed or deferred scans must retry, but must not wake the
            # scheduler in a tight loop with an already overdue timestamp.
            layout_deadline = now + self.RETRY
        if workers_blocked_until is not None:
            layout_deadline = max(layout_deadline, workers_blocked_until)
        self.config.modified[
            self._layout_next_scan_key()
        ] = layout_deadline
        next_run = (min(future_times) if future_times else
                    layout_deadline if self._production_city_empty else
                    now + self.IDLE_RETRY)
        next_run = min(next_run, layout_deadline)
        if workers_blocked_until is not None:
            # Earlier facilities may have appended a shorter retry before the
            # shortage was confirmed. Do not wake this city just to skip them.
            next_run = max(next_run, workers_blocked_until)
        self._set_city_status(status_lines, next_run, now)
        self.config.task_delay(target=next_run)
        logger.attr(f'CityProductionNextRun_{self.city}', next_run)
        return bool(switch_ok)
