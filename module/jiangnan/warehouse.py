import re
import time
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from module.base.button import Button
from module.base.utils import crop, load_image
from module.jiangnan.city import JiangnanCity, KNOWN_CITIES
from module.jiangnan.app import JiangnanApp
from module.jiangnan.assets import (
    MAIN_BUILD_CHECK,
    BUILD_TITLE_CHECK,
    BUILD_WAREHOUSE_BUTTON,
    WAREHOUSE_TITLE_CHECK,
    WAREHOUSE_SPECIAL_COUNT_SLOTS,
    WAREHOUSE_SPECIAL_CAPACITY_SLOTS,
)
from module.jiangnan.resource import (
    WAREHOUSE_RESOURCE_ROWS,
    WAREHOUSE_RESOURCES,
    WAREHOUSE_DYNAMIC_RESOURCES,
    SHAOXING_MERCHANT_RESOURCES,
    WAREHOUSE_SPECIAL_RESOURCES,
    WAREHOUSE_OCR_ALIASES,
)
from module.jiangnan.resource_layout import (
    PAGE_WAREHOUSE,
    ResourceLayoutManager,
)
from module.jiangnan.ocr import Ocr, Digit, read_confirmed_digits
from module.jiangnan.digit_recognition import read_special_digits
from module.jiangnan.ppocr_v6 import PPOCR_V6_MEDIUM
from module.jiangnan.vision import stable_feature_match
from module.logger import logger


# ============================================================
# 已知普通仓库资源
#
# 作用：
# OCR 可能识别成：
#
#   原木应天府苏州府
#
# 我们只需要：
#
#   原木
#
# 对未知的新资源不会强行猜测，而是保留 OCR 原始中文。
# ============================================================


# ============================================================
# OCR
# ============================================================

class WarehouseNameOcr(Ocr):

    def __init__(self, buttons):
        super().__init__(
            buttons=buttons,
            lang='cnocr',
            name='WAREHOUSE_NAME_OCR'
        )

    def pre_process(self, image):
        """
        仓库资源名称。

        仓库纸张背景本身比较深，
        不能使用 180 这种过高阈值，
        否则背景也会被当成文字。
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        # 只保留真正较深的文字
        binary = cv2.inRange(
            gray,
            0,
            120
        )

        binary = cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        return binary


class WarehouseCountDigitOcr(Digit):

    def __init__(
        self,
        buttons,
        lang='azur_lane',
        threshold=128,
    ):
        super().__init__(
            buttons=buttons,
            lang=lang,
            letter=(54, 47, 43),
            threshold=threshold,
            name=f'WAREHOUSE_COUNT_{lang.upper()}'
        )

class WarehouseCountOcr(Ocr):

    def __init__(self, buttons):
        super().__init__(
            buttons=buttons,
            lang='cnocr',

            # 只允许识别数字
            alphabet='0123456789',

            name='WAREHOUSE_COUNT_OCR'
        )

    def pre_process(self, image):
        """
        仓库当前库存数量 OCR。

        这里只识别上面的当前数量：
            8948
        而不是下面的：
            /3600
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        binary = cv2.inRange(
            gray,
            0,
            120
        )

        binary = cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        return binary

class WarehouseSpecialDigitOcr(Digit):

    def __init__(
        self,
        buttons,
        lang='azur_lane',
        threshold=128,
    ):
        super().__init__(
            buttons=buttons,
            lang=lang,
            letter=(54, 47, 43),
            threshold=threshold,
            name=f'WAREHOUSE_SPECIAL_DIGIT_{lang.upper()}'
        )

# ============================================================
# Warehouse
# ============================================================

class JiangnanWarehouse(JiangnanApp):
    RESOURCE_ICON_DIR = Path('assets/jiangnan/allocation/resource_icons')
    SPECIAL_ICON_CENTERS = {
        '茶油': (750, 408),
        '竹编': (824, 408),
        '甲酒': (898, 408),
        '盆景': (972, 408),
        '木雕': (370, 529),
        '竹雕': (444, 529),
        '砖雕': (518, 529),
        '石雕': (592, 529),
    }

    def __init__(self, config, device=None, task=None):
        super().__init__(
            config=config,
            device=device,
            task=task
        )

        # ========================================
        # 仓库固定顺序定位状态
        # ========================================

        # 是否已经确认仓库顶部
        self._warehouse_tracker_ready = False

        # 当前相对于仓库顶部累计向下滚动了多少像素
        self._warehouse_scroll_offset = 0.0

        # 仓库第 1 行在顶部状态下的 Y
        self._warehouse_row_origin = None

        # 正常每一资源行高度，实际约 120 px
        self._warehouse_row_height = 120.0

        self._warehouse_drag_seq = 0

        # 四行滑动闭环控制：58 是实机标定的基准终点，之后根据
        # 图像匹配得到的真实位移修正下一次终点。
        self._warehouse_drag_end_y = 58.0
        self._warehouse_scroll_step_count = 0

        # 是否已经确认到达仓库底部
        self._warehouse_bottom_confirmed = False

        # 当前城市普通物资的单项库存上限，以及最近一次逐城扫描结果。
        self.warehouse_capacity = None
        self.warehouse_capacity_by_city = {}
        self.warehouse_special_capacity_by_city = {}
        self.warehouse_special_capacity = {}
        self.merchant_capacity = None
        # “查看仓库”跟随当前仓库建筑卡片上下移动。不同城市的第一张
        # 仓库卡片高度不同，不能只在 assets.py 的固定坐标处识别。
        self._build_warehouse_template = load_image(
            BUILD_WAREHOUSE_BUTTON.file,
            BUILD_WAREHOUSE_BUTTON.area,
        )
        self._main_build_template = load_image(
            MAIN_BUILD_CHECK.file,
            MAIN_BUILD_CHECK.area,
        )
        self.resource_layout = ResourceLayoutManager(self.config)
        self.city_layout = self.resource_layout

    # 仓库滚动区域
    LIST_X1 = 330
    LIST_Y1 = 55
    LIST_X2 = 1090
    LIST_Y2 = 580

    BUILD_WAREHOUSE_SEARCH_AREA = (1060, 105, 1280, 250)
    SHAOXING_WAREHOUSE_TABS = {
        '城市': (91, 82, 151, 163),
        '商行': (91, 222, 151, 302),
    }

    def _shaoxing_tab_selected(self, warehouse_type):
        image = self.device.image
        if image is None or image.shape[0] < 270 or image.shape[1] < 120:
            return False
        # 标签空白处的亮度随选中状态切换；避开文字与红色边框。
        city = float(np.median(image[115:140, 96:116, 0]))
        merchant = float(np.median(image[235:260, 96:116, 0]))
        return (city - merchant > 25 if warehouse_type == '城市'
                else merchant - city > 25)

    def select_shaoxing_warehouse(self, warehouse_type):
        """绍兴两种仓库共用窗口，按左侧标签切换并核实页面仍在。"""
        area = self.SHAOXING_WAREHOUSE_TABS[warehouse_type]
        self.device.click(Button(
            area=(), color=(), button=area,
            name=f'SHAOXING_{warehouse_type}_WAREHOUSE_TAB',
        ))
        for _ in range(12):
            self.device.sleep(0.25)
            self.device.screenshot()
            if (self.is_warehouse_page()
                    and self._shaoxing_tab_selected(warehouse_type)):
                return True
        logger.warning(f'Shaoxing {warehouse_type} warehouse tab did not open')
        return False

    def _shaoxing_merchant_top_visible(self):
        """Use the first merchant row as a visual top anchor."""
        buttons = []
        for x1, x2, side in ((330, 710, 'left'), (710, 1090, 'right')):
            rows = self._detect_column_rows(x1, x2, side)
            if not rows:
                continue
            name_buttons, _ = self._create_row_buttons(rows[:1], side)
            buttons.extend(name_buttons)
        if not buttons:
            return False
        raw_names = self._as_list(WarehouseNameOcr(buttons).ocr(self.device.image))
        names = {self.normalize_resource_name(name) for name in raw_names}
        visible = bool({'白醪酒', '元红酒'} & names)
        logger.attr('ShaoxingMerchantTopNames', sorted(names))
        return visible

    def scan_shaoxing_merchant(self, target_resources=None, max_pages=16):
        """按画面货名逐屏读取商行仓库，滑动只用于翻页。"""
        if not self.select_shaoxing_warehouse('商行'):
            return None
        self._warehouse_tracker_ready = False
        self._warehouse_bottom_confirmed = False
        targets = set(target_resources or SHAOXING_MERCHANT_RESOURCES)
        targets &= set(SHAOXING_MERCHANT_RESOURCES)
        if not targets:
            return {}

        # 动态背景和边界回弹会令整幅图持续变化；只用首行货名确认顶部。
        self.device.screenshot()
        if self.is_network_timeout_popup():
            logger.warning('Network popup interrupted Shaoxing merchant scan')
            return None
        top_visible = self._shaoxing_merchant_top_visible()
        for _ in range(12):
            if top_visible:
                break
            self._warehouse_fast_swipe(
                (650, 100), (650, 550), name='SHAOXING_MERCHANT_TO_TOP'
            )
            self.device.sleep(0.4)
            self.device.screenshot()
            if self.is_network_timeout_popup():
                logger.warning('Network popup interrupted Shaoxing merchant scan')
                return None
            top_visible = self._shaoxing_merchant_top_visible()
        if not top_visible:
            logger.warning('Shaoxing merchant warehouse top not confirmed')
            self.save_abnormal_screenshot('shaoxing_merchant_top_unconfirmed')
            return None
        self.merchant_capacity = self.read_warehouse_capacity()
        readings = {}
        seen_visible = set()
        repeated_visible = 0
        for page in range(max_pages):
            # Top confirmation captured the first frame; the previous drag
            # captured every later one. Read those frames before acquiring
            # another screenshot that repeats the same page.
            if self.is_network_timeout_popup():
                logger.warning('Network popup interrupted Shaoxing merchant scan')
                return None
            # Read every merchant label even in a targeted supplemental scan.
            # Otherwise an unreadable target yields {} on every page and we
            # cannot tell that the list has stopped moving.
            current = self.read_current_page(
                target_resources=set(SHAOXING_MERCHANT_RESOURCES)
            )
            visible = frozenset(current)
            repeated_visible = (
                repeated_visible + 1 if visible and visible <= seen_visible
                else 0
            )
            # Bottom bounce can alternate three/four visible labels. Requiring
            # identical consecutive sets kept dragging an already exhausted
            # list while looking for an unreadable item near the top.
            seen_visible.update(visible)
            readings.update({
                name: value for name, value in current.items()
                if name in targets and int(value) >= 0
            })
            logger.attr(
                'ShaoxingMerchantPage',
                f'page={page + 1}, read={sorted(readings)}, '
                f'missing={sorted(targets - readings.keys())}',
            )
            if targets <= readings.keys():
                break
            if repeated_visible >= 2:
                logger.info(
                    'Shaoxing merchant warehouse stopped on repeated '
                    'resource rows without newly visible items'
                )
                break
            before = self._warehouse_list_image()
            self._warehouse_drag(
                (650, 550), (650, 130), name='SHAOXING_MERCHANT_DOWN'
            )
            self.device.screenshot()
            if self.is_network_timeout_popup():
                logger.warning('Network popup interrupted Shaoxing merchant scan')
                return None
            if self._image_change(before, self._warehouse_list_image()) < 2.0:
                logger.info('Shaoxing merchant warehouse bottom confirmed')
                break
        else:
            logger.warning('Shaoxing merchant warehouse scan reached page limit')
        if targets - readings.keys():
            logger.warning(
                'Shaoxing merchant scan ended with unreadable or missing items: '
                f'{sorted(targets - readings.keys())}'
            )
        return readings

    @staticmethod
    def _extract_resource_icon(image, center, radius=30):
        """按已识别中心提取徽州特产单图标，避开名称和库存数字。"""
        cx, cy = map(int, center)
        y1, y2 = max(0, cy - 22), min(image.shape[0], cy + 25)
        x1, x2 = max(0, cx - radius), min(image.shape[1], cx + radius)
        roi = image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None
        return roi

    def _save_resource_icon_template(self, resource, center=None, icon=None):
        if icon is None:
            icon = self._extract_resource_icon(self.device.image, center)
        if icon is None:
            logger.warning(f'Unable to extract resource icon: {resource}')
            return False
        path = self.RESOURCE_ICON_DIR / f'{resource}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = cv2.imencode(
            '.png', cv2.cvtColor(icon, cv2.COLOR_RGB2BGR)
        )[1]
        encoded.tofile(str(path))
        logger.info(f'Resource icon mapped: {resource} -> {path}')
        return True

    def _capture_normal_resource_icon(self, resource, row, side):
        top, bottom = row
        # 仓库库存非零时，同一图标会横向重复绘制。保存完整图标带比
        # 猜测其中某一枚的中心更稳定；运输页用局部特征在图标带中匹配。
        x1, x2 = ((334, 610) if side == 'left' else (718, 994))
        # The first row is bounded by the window header, rather than a normal
        # separator. Its name/icons sit lower (as do its quantity OCR boxes).
        # A fixed top+33..top+82 crop learned the name and upper half of rice
        # bowls, and also cut off the bases of ordinary paper/charcoal icons.
        # A scrolled ordinary row can also start near the top of the window;
        # only the taller header-bounded interval needs the extra offset.
        header_row = top < 80 and bottom - top > 125
        y1 = max(0, top + (58 if header_row else 33))
        y2 = min(self.device.image.shape[0], bottom - 18)
        icon = self.device.image[y1:y2, x1:x2].copy()
        if icon.size == 0 or icon.shape[0] < 20:
            logger.warning(f'Unable to crop warehouse icon band: {resource}')
            return False
        return self._save_resource_icon_template(resource, icon=icon)

    def find_main_build_button(self, similarity=0.63):
        """在右下营造入口附近匹配图标，适配不同城市按钮上下偏移。"""
        self.device.screenshot()
        template = self._main_build_template
        search = self.device.image
        score, loc, gray_score, edge_score = stable_feature_match(
            search, template, gray_weight=0.35
        )
        logger.attr(
            'MainBuildDynamicScore',
            f'stable={score:.3f}, gray={gray_score:.3f}, '
            f'edge={edge_score:.3f}',
        )
        if score < similarity:
            return None
        left, top = loc
        return Button(
            area=(), color=(),
            # 匹配到的图标本身就在可点击区域内。只点识别框内部，避免
            # 扩展后的随机落点越过窄版城市按钮（江宁县会点到右侧空白）。
            button=(left, top, left + template.shape[1],
                    top + template.shape[0]),
            name='MAIN_BUILD_BUTTON_DYNAMIC',
        )

    # 仓库回到顶部后，第一行左右两侧的“/库存上限”数字区域。
    CAPACITY_BUTTONS = [
        Button(
            area=(620, 130, 710, 170),
            color=(),
            button=(620, 130, 710, 170),
            name='WAREHOUSE_CAPACITY_LEFT',
        ),
        Button(
            area=(1000, 130, 1090, 170),
            color=(),
            button=(1000, 130, 1090, 170),
            name='WAREHOUSE_CAPACITY_RIGHT',
        ),
    ]

    def read_warehouse_capacity(self):
        """读取第一页两列重复显示的“/库存上限”，用重复结果防 OCR 误判。"""
        candidates = []
        raw = WarehouseCountDigitOcr(
            self.CAPACITY_BUTTONS,
            lang='cnocr',
        ).ocr(self.device.image)
        for value in self._as_list(raw):
            value = self.normalize_count(value)
            if value is not None and 0 < value <= 100000:
                candidates.append(value)

        counts = Counter(candidates)
        if not counts:
            logger.warning('Unable to read warehouse capacity')
            return None

        capacity, repeats = counts.most_common(1)[0]
        logger.attr('WarehouseCapacityCandidates', candidates)
        if repeats < 2:
            logger.warning(
                f'Warehouse capacity OCR has no repeated result: {candidates}'
            )
            return None

        logger.attr('WarehouseCapacity', capacity)
        return capacity

    def read_visible_warehouse_capacity(self):
        """Only read the fixed capacity labels when the first row is visible."""
        if not self._warehouse_top_visible():
            logger.info('Warehouse capacity skipped: first row is not visible')
            return None
        return self.read_warehouse_capacity()

    def _consensus_digit(self, button, name, special=False):
        """用 PP-OCRv6 三级模型链识别原始数字裁剪。"""
        ocr_type = WarehouseSpecialDigitOcr if special else WarehouseCountDigitOcr
        value = ocr_type(button, lang='cnocr').ocr(self.device.image)
        try:
            value = int(value)
        except (TypeError, ValueError):
            logger.warning(f'Warehouse OCR unreadable: {name}')
            return None
        if value < 0:
            logger.warning(f'Warehouse OCR unreadable: {name}')
            return None
        capacity = getattr(self, 'warehouse_capacity', None)
        if capacity and value > capacity:
            self.device.screenshot()
            confirmed = read_confirmed_digits(self.device.image, button.area)
            logger.info(f'Warehouse over-capacity verification: {name}, first={value}, confirmed={confirmed}')
            return confirmed
        logger.info(f'Warehouse PP-OCRv6: {name}, result={value}')
        return value

    def _visual_zero_confirmed(self, button):
        """Confirm that a disputed zero occupies one digit at the count edge."""
        x1, y1, x2, y2 = button.area
        # The denominator can start at the bottom of a clipped first row.
        # Inspect only the upper part of the current-count crop.
        number = crop(self.device.image, (x1, y1, x2, max(y1 + 1, y2 - 10)))
        if number.size == 0:
            return False
        gray = cv2.cvtColor(number, cv2.COLOR_RGB2GRAY)
        ink = cv2.inRange(gray, 0, 105)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        glyphs = [
            (x, y, width, height)
            for x, y, width, height, area in stats[1:]
            if area >= 20 and height >= 12
        ]
        if len(glyphs) != 1:
            return False
        x, y, width, height = glyphs[0]
        image_width = gray.shape[1]
        if not (
            9 <= width <= 25
            and 15 <= height <= 35
            and x >= image_width * 0.55
            and x + width <= image_width
        ):
            return False
        digit = ink[y:y + height, x:x + width]
        contours, hierarchy = cv2.findContours(
            digit, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            return False
        # A true 0 has an enclosed counter; a stray stroke or 1 does not.
        return any(
            item[3] >= 0 and cv2.contourArea(contours[index]) >= 5
            for index, item in enumerate(hierarchy[0])
        )

    def _consensus_digit_batch(self, buttons, names, verify_over_capacity=True):
        """一次批量识别当前页的所有普通物资数字。"""
        reader = WarehouseCountDigitOcr(buttons, lang='cnocr')
        raw_values = self._as_list(reader.ocr(self.device.image))
        result = []
        for i, name in enumerate(names):
            raw = raw_values[i] if i < len(raw_values) else None
            try:
                value = int(raw)
            except (TypeError, ValueError):
                result.append(None)
                continue
            if value < 0:
                result.append(None)
                continue
            logger.info(f'Warehouse PP-OCRv6 batch: {name}, result={value}')
            result.append(value)

        # PP-OCRv6 偶尔会漏掉清晰数字并返回 -1（实测茶叶、竹子）。
        # 只对失败格使用另一套纯数字 OCR 重读，不重复整页，也不把
        # 第一次失败直接写入库存。
        missing = [index for index, value in enumerate(result) if value is None]
        if missing:
            retry_buttons = [buttons[index] for index in missing]
            # ``_DirectPPOcrMixin`` 默认把原始 RGB 裁剪直接交给模型，
            # 因而仅换一个 OCR 类并不会执行 WarehouseCountOcr 的二值化
            # 预处理，旧“fallback”实际上仍在重读同一张缓存图。这里显式
            # 生成紧凑的黑底白字放大图，专门处理页面首行偶发漏掉的单个
            # 0（江宁县茶叶/竹子会同时遇到）。
            fallback_reader = WarehouseCountOcr(retry_buttons)
            retry_images = [
                fallback_reader.pre_process(
                    crop(self.device.image, button.area)
                )
                for button in retry_buttons
            ]
            retry_raw = self._as_list(
                fallback_reader.ocr(retry_images, direct_ocr=True)
            )
            for retry_index, result_index in enumerate(missing):
                raw = (
                    retry_raw[retry_index]
                    if retry_index < len(retry_raw) else None
                )
                value = self.normalize_count(raw)
                if value is None or value < 0:
                    continue
                result[result_index] = value
                logger.info(
                    f'Warehouse binary digit fallback: '
                    f'{names[result_index]}, result={value}'
                )

        # 第一种重读仍失败时，再把数量框向上扩展。四行滑动后的首行
        # 分隔线可能被顶部裁掉几像素，名称仍可读，但数字基线会落在
        # 常规框边缘。扩展框只覆盖名称与“当前数量”，不触及下方容量。
        missing = [index for index, value in enumerate(result) if value is None]
        if missing:
            expanded_buttons = []
            for result_index in missing:
                x1, y1, x2, y2 = buttons[result_index].area
                area = (
                    max(0, x1 - 12),
                    max(self.LIST_Y1, y1 - 20),
                    x2,
                    max(y1 + 20, y2 - 8),
                )
                expanded_buttons.append(Button(
                    area=area,
                    color=(),
                    button=area,
                    name=f'WAREHOUSE_COUNT_EXPANDED_{result_index}',
                ))
            expanded_reader = WarehouseCountOcr(expanded_buttons)
            expanded_images = [
                expanded_reader.pre_process(
                    crop(self.device.image, button.area)
                )
                for button in expanded_buttons
            ]
            expanded_raw = self._as_list(
                expanded_reader.ocr(expanded_images, direct_ocr=True)
            )
            for retry_index, result_index in enumerate(missing):
                raw = (
                    expanded_raw[retry_index]
                    if retry_index < len(expanded_raw) else None
                )
                value = self.normalize_count(raw)
                if value is None or value < 0:
                    continue
                result[result_index] = value
                logger.info(
                    f'Warehouse expanded digit fallback: '
                    f'{names[result_index]}, result={value}'
                )
        # 零值也是有效 OCR 结果，但首行裁剪不完整时，3600 曾被误读成
        # 0 并直接覆盖真实库存。用不同预处理复核零值；有分歧就留作未读，
        # 不把一次可疑识别传给库存和调配。
        zero_indices = [index for index, value in enumerate(result) if value == 0]
        if zero_indices:
            zero_disagreed = False
            zero_buttons = [buttons[index] for index in zero_indices]
            zero_reader = WarehouseCountOcr(zero_buttons)
            zero_images = [
                zero_reader.pre_process(crop(self.device.image, button.area))
                for button in zero_buttons
            ]
            zero_raw = self._as_list(
                zero_reader.ocr(zero_images, direct_ocr=True)
            )
            for retry_index, result_index in enumerate(zero_indices):
                raw = zero_raw[retry_index] if retry_index < len(zero_raw) else None
                confirmed = self.normalize_count(raw)
                if confirmed != 0:
                    if self._visual_zero_confirmed(buttons[result_index]):
                        logger.info(
                            f'Warehouse zero confirmed by single-digit shape: '
                            f'{names[result_index]}, second_read={confirmed}'
                        )
                        continue
                    zero_disagreed = True
                    result[result_index] = None
                    logger.warning(
                        f'Warehouse zero OCR not confirmed: '
                        f'{names[result_index]}, second_read={confirmed}'
                    )
            if zero_disagreed:
                self.save_abnormal_screenshot('warehouse_zero_unconfirmed')
        capacity = getattr(self, 'warehouse_capacity', None)
        over_capacity = [i for i, value in enumerate(result)
                         if verify_over_capacity and value is not None
                         and capacity and value > capacity]
        if over_capacity:
            self.device.screenshot()
            for index in over_capacity:
                confirmed = read_confirmed_digits(self.device.image, buttons[index].area)
                logger.info(f'Warehouse over-capacity verification: {names[index]}, '
                            f'first={result[index]}, confirmed={confirmed}')
                # Over-capacity is a verification trigger, never an upper bound.
                result[index] = confirmed
        for index, value in enumerate(result):
            if value is None:
                logger.warning(f'Warehouse OCR unreadable: {names[index]}')
        return result


    def _read_special_number(
        self,
        button,
        name='SPECIAL'
    ):
        """
        徽州特产数字使用 PP-OCRv6 tiny/small/medium 级联识别。
        """

        logger.info(
            f'Warehouse special Digit area: '
            f'{name}={button.area}'
        )

        return self._consensus_digit(button, name=name, special=True)

    def _warehouse_drag(
        self,
        p1,
        p2,
        name
    ):
        """
        仓库专用拖动。

        每次生成唯一 control name，
        避免 ALAS 把连续滚动误判成同一按钮被反复点击。
        """

        self._warehouse_drag_seq += 1

        control_name = (
            f'{name}_{self._warehouse_drag_seq:04d}'
        )

        logger.info(
            f'Warehouse drag: '
            f'{p1} -> {p2}, '
            f'move=1.8s, hold=0.2s, '
            f'name={control_name}'
        )

        self.jiangnan_drag(
            p1,
            p2,
            name=control_name,
            # 确定性直线四行手势；速度按实际距离反算，移动约 1.8s。
            # MaaTouch、minitouch 和 nemu_ipc 使用反算后的
            # swipe_speed；uiautomator2/ADB 使用 swipe_duration。
            swipe_duration=1.8,
            swipe_speed=None,
            swipe_min_distance=1,
            swipe_linear=True,
            hold_duration=0.2,
        )

    def _warehouse_fast_swipe(self, p1, p2, name):
        """仓库回边界专用快速滑动；抬手后不作终点停留。"""
        self._warehouse_drag_seq += 1
        control_name = f'{name}_{self._warehouse_drag_seq:04d}'
        logger.info(
            f'Warehouse fast swipe: {p1} -> {p2}, '
            f'hold=0s, name={control_name}'
        )
        self.device.swipe(
            p1,
            p2,
            duration=0.1,
            name=control_name,
        )

    def _warehouse_top_visible(self):
        """Confirm the first warehouse row by its two fixed resource labels."""
        left_rows = self._detect_column_rows(330, 710, 'TOP_CHECK_LEFT')
        right_rows = self._detect_column_rows(710, 1090, 'TOP_CHECK_RIGHT')
        if left_rows and right_rows:
            templates = getattr(self, '_warehouse_top_templates', None)
            if templates is None:
                templates = {}
                for resource in ('蔬菜', '肉类'):
                    path = self.RESOURCE_ICON_DIR / f'{resource}.png'
                    if path.exists():
                        template = load_image(str(path), area=(0, 0, 60, 25))
                        templates[resource] = cv2.cvtColor(
                            template, cv2.COLOR_RGB2GRAY
                        )
                self._warehouse_top_templates = templates
            scores = {}
            for resource, rows, x1, x2 in (
                ('蔬菜', left_rows, 330, 610),
                ('肉类', right_rows, 710, 990),
            ):
                template = templates.get(resource)
                if template is None:
                    continue
                top = rows[0][0]
                patch = self.device.image[top + 5:top + 77, x1:x2]
                if (patch.shape[0] < template.shape[0]
                        or patch.shape[1] < template.shape[1]):
                    continue
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                scores[resource] = float(cv2.matchTemplate(
                    gray, template, cv2.TM_CCOEFF_NORMED
                ).max())
            logger.attr('WarehouseTopVisualScores', scores)
            if all(scores.get(resource, 0) >= 0.80
                   for resource in ('蔬菜', '肉类')):
                logger.attr('WarehouseTopVisible', True)
                return True
        buttons = []
        for rows, side in ((left_rows, 'left'), (right_rows, 'right')):
            if not rows:
                continue
            name_buttons, _ = self._create_row_buttons(rows[:1], side)
            buttons.extend(name_buttons)
        if not buttons:
            return False
        raw_names = self._as_list(WarehouseNameOcr(buttons).ocr(self.device.image))
        names = {
            self.normalize_resource_name(value)
            for value in raw_names
        }
        names.discard('')
        is_top = bool({'蔬菜', '肉类'} & names)
        logger.attr('WarehouseTopNames', sorted(names))
        logger.attr('WarehouseTopVisible', is_top)
        return is_top

    # --------------------------------------------------------
    # 页面识别
    # --------------------------------------------------------

    def _is_warehouse_inventory_layout(self):
        """用仓库资源表的双列行结构识别新版仓库页。

        新版界面不再显示旧版 ``WAREHOUSE_TITLE_CHECK`` 对应的标题，
        但仓库右列始终会显示至少三行约 120px 高的资源行，左右两列的
        分隔线也会对齐。营造建筑列表虽然也有横线，却不具备这个结构。
        """
        left_rows = self._detect_column_rows(330, 710, 'PAGE_LEFT')
        right_rows = self._detect_column_rows(710, 1090, 'PAGE_RIGHT')

        aligned = 0
        for left_top, left_bottom in left_rows:
            if any(
                abs(left_top - right_top) <= 4
                and abs(left_bottom - right_bottom) <= 4
                for right_top, right_bottom in right_rows
            ):
                aligned += 1

        result = len(right_rows) >= 3 and aligned >= 2
        logger.attr(
            'WarehouseInventoryLayout',
            f'{result} (left={len(left_rows)}, right={len(right_rows)}, '
            f'aligned={aligned})',
        )
        return result

    def is_build_page(self):

        # “查看仓库”在新版仓库页本身也存在。先排除已经进入仓库的
        # 情况，否则会把仓库误判成营造页并不断重复点击该按钮。
        if self._is_warehouse_inventory_layout():
            logger.attr('BuildTitleCheck', False)
            logger.attr('BuildWarehouseEntryCheck', False)
            return False

        title = self.appear(
            BUILD_TITLE_CHECK,
            offset=(5, 5),
            similarity=0.88
        )

        logger.attr(
            'BuildTitleCheck',
            title
        )

        # 新版营造页左上会被玩家经验区域覆盖，旧版“营造”标题模板可能
        # 消失；右上“查看仓库”是该页稳定且可动态定位的专属控件。
        warehouse_entry = self.find_build_warehouse_button()
        logger.attr(
            'BuildWarehouseEntryCheck',
            warehouse_entry is not None,
        )
        return title or warehouse_entry is not None

    def find_build_warehouse_button(self, similarity=0.66):
        """在营造页右上区域动态定位“查看仓库”按钮。"""
        search = self.device.image
        template = self._build_warehouse_template
        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None

        score, location, gray_score, edge_score = stable_feature_match(
            search, template, gray_weight=0.40
        )
        logger.attr(
            'BuildWarehouseDynamicScore',
            f'stable={score:.3f}, gray={gray_score:.3f}, '
            f'edge={edge_score:.3f}',
        )
        if score < similarity:
            return None

        height, width = template.shape[:2]
        left, top = location
        logger.info(
            f'Warehouse button found: score={score:.3f}, '
            f'center=({left + width // 2}, {top + height // 2})'
        )
        return Button(
            area=(),
            color=(),
            # 整个模板包含图标、文字和边框，随机点击大框时可能落到不
            # 响应的装饰区。点击框仍由模板实时定位，只取模板中心。
            button=(
                left + width // 2 - 10,
                top + height // 2 - 10,
                left + width // 2 + 10,
                top + height // 2 + 10,
            ),
            name='BUILD_WAREHOUSE_BUTTON_DYNAMIC',
        )

    def is_warehouse_page(self):

        result = self.appear(
            WAREHOUSE_TITLE_CHECK,
            offset=(5, 5),
            similarity=0.88
        )

        if not result:
            result = self._is_warehouse_inventory_layout()

        logger.attr('WarehousePageCheck', result)
        return result

    # --------------------------------------------------------
    # 主界面 -> 营造
    # --------------------------------------------------------

    def goto_build(self, timeout=8):

        logger.hr(
            'JIANGNAN GOTO BUILD'
        )

        if not self.ensure_ready():

            logger.warning(
                'Unable to prepare main page'
            )

            return False

        logger.info(
            'Click build'
        )

        build_button = self.find_main_build_button()
        if build_button is None:
            build_button = self.find_main_build_navigation_button()
        if build_button is None:
            # A delayed city popup can arrive after ensure_ready() has checked
            # the main page. Give it a short window to appear and be dismissed.
            recovery_deadline = time.time() + 2
            while build_button is None and time.time() < recovery_deadline:
                self.device.screenshot()
                if self.handle_startup_popup():
                    logger.info('Popup appeared while opening build; retry')
                    self.device.sleep(0.5)
                    self.device.screenshot()
                build_button = self.find_main_build_button()
                if build_button is None:
                    build_button = self.find_main_build_navigation_button()
                if build_button is None:
                    self.device.sleep(0.3)
            if build_button is None:
                logger.warning('Main build button was not recognized')
                return False
        self.device.click(build_button)
        click_count = 1
        last_click = time.time()

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            # 城市刚切换完成时，税课司等弹窗可能在点击“营造”之后才到达。
            # 先清掉遮挡，再按当前画面重新找按钮，避免整段等待都停在弹窗上。
            if self.handle_startup_popup():
                self.device.sleep(0.3)
                self.device.screenshot()
                build_button = self.find_main_build_button()
                if build_button is None:
                    build_button = self.find_main_build_navigation_button()
                if build_button is not None:
                    self.device.click(build_button)
                    click_count += 1
                    last_click = time.time()
                continue

            if self.is_build_page():

                logger.info(
                    'Build page arrived'
                )

                return True

            if click_count < 3 and time.time() - last_click >= 2:
                build_button = self.find_main_build_button()
                if build_button is None:
                    build_button = self.find_main_build_navigation_button()
                if build_button is not None:
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry recognized build button: {click_count}/3'
                    )
                    self.device.click(build_button)

            self.device.sleep(0.4)

        logger.warning(
            'Build page timeout'
        )

        return False

    # --------------------------------------------------------
    # 营造 -> 仓库
    # --------------------------------------------------------

    def goto_warehouse(self, timeout=8):

        logger.hr(
            'JIANGNAN GOTO WAREHOUSE'
        )

        self.device.screenshot()

        if self.is_warehouse_page():

            logger.info(
                'Already in warehouse'
            )

            return True

        if not self.is_build_page():
            if not self.goto_build():
                return False
        else:
            logger.info('Already in build page')

        self.device.screenshot()

        # 不同城市的仓库建筑卡片高度不同，“查看仓库”会随卡片下移。
        # 在右上区域动态匹配，点击本次截图中的实际位置。
        warehouse_button = self.find_build_warehouse_button()
        if warehouse_button is None:

            logger.warning(
                'Warehouse button not found'
            )

            return False

        logger.info(
            'Click view warehouse'
        )

        self.device.click(
            warehouse_button
        )
        click_count = 1
        last_click = time.time()

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            if self.is_warehouse_page():

                logger.info(
                    'Warehouse page arrived'
                )

                return True

            if click_count < 3 and time.time() - last_click >= 2:
                warehouse_button = self.find_build_warehouse_button()
                if warehouse_button is not None:
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry recognized warehouse button: {click_count}/3'
                    )
                    self.device.click(warehouse_button)

            self.device.sleep(0.4)

        logger.warning(
            'Warehouse page timeout'
        )

        return False

    # ========================================================
    # OCR normalization
    # ========================================================

    @staticmethod
    def normalize_resource_name(name):
        """
        OCR 名称只作为辅助校验。

        正式资源身份由：
            仓库顶部位置
            + 固定资源顺序
        决定。
        """

        if name is None:
            return ''

        name = ''.join(
            re.findall(
                r'[\u3400-\u9fff]+',
                str(name)
            )
        )

        if not name:
            return ''

        if name in WAREHOUSE_RESOURCES or name in WAREHOUSE_DYNAMIC_RESOURCES:
            return name

        if name in WAREHOUSE_OCR_ALIASES:

            corrected = WAREHOUSE_OCR_ALIASES[
                name
            ]

            logger.info(
                f'Warehouse OCR alias: '
                f'{name} -> {corrected}'
            )

            return corrected

        # 正确资源名称位于 OCR 开头
        for resource in sorted(
            WAREHOUSE_RESOURCES + WAREHOUSE_DYNAMIC_RESOURCES,
            key=len,
            reverse=True
        ):

            if name.startswith(resource):
                return resource

        # 已知错误别名位于 OCR 开头
        for alias, resource in sorted(
            WAREHOUSE_OCR_ALIASES.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):

            if name.startswith(alias):
                return resource

        return ''

    @staticmethod
    def normalize_count(value):

        if value is None:
            return None

        text = str(value)

        # OCR 常见数字误识别
        text = (
            text
            .replace('|', '1')
            .replace('I', '1')
            .replace('l', '1')
            .replace('O', '0')
            .replace('o', '0')
        )

        digits = ''.join(
            re.findall(
                r'\d+',
                text
            )
        )

        if not digits:
            return None

        try:
            return int(digits)

        except ValueError:
            return None

    @staticmethod
    def _as_list(result):

        if isinstance(result, list):
            return result

        return [result]

    # ========================================================
    # 自动检测资源行
    # ========================================================

    def _detect_column_rows(
        self,
        x1,
        x2,
        name
    ):
        """
        自动检测仓库某一列的完整资源行。

        不依赖固定 Y 坐标。

        仓库每个资源格之间都有一条很长的横线，
        利用 horizontal morphology 把这些横线找出来。

        不完整的顶部/底部半行自动忽略。
        """

        image = self.device.image

        # Scan the complete frame vertically.  Scroll landing errors can move
        # resource separators away from the historical LIST_Y1/LIST_Y2 crop.
        y1 = 0
        y2 = image.shape[0]
        roi = image[:, x1:x2]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2GRAY
        )

        # 保留较深色的横线
        dark = cv2.inRange(
            gray,
            0,
            120
        )

        # 仓库分隔线非常长。
        # 资源进度条虽然也横向，
        # 但长度明显短于整列。
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (180, 1)
        )

        horizontal = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            kernel
        )

        score = np.count_nonzero(
            horizontal,
            axis=1
        )

        # 单列宽约 380 px，
        # 真正分隔线基本贯穿整列。
        positions = np.where(
            score > 320
        )[0]

        groups = []

        for value in positions:

            value = int(value)

            if (
                not groups
                or
                value > groups[-1][-1] + 1
            ):
                groups.append([value])

            else:
                groups[-1].append(value)

        separators = []

        for group in groups:

            y = int(
                np.median(group)
            ) + y1

            # 忽略窗口上下边缘
            if (
                y > y1 + 15
                and
                y < y2 - 15
            ):
                separators.append(y)

        # Only intervals bounded by two detected separators are rows.
        # Do not synthesize image-edge bounds: 580..720 happens to be close to
        # one row high but belongs to the upgrade/footer area.
        bounds = separators

        rows = []

        for top, bottom in zip(
            bounds[:-1],
            bounds[1:]
        ):

            height = (
                bottom - top
            )

            # 正常资源行约 120 px。
            #
            # 顶部或底部只显示一半的行
            # 会被排除。
            if 90 <= height <= 145:
                # A real warehouse resource row is split by the long vertical
                # divider shared by the left/right columns.  The footer happens
                # to be row-height too, but has no such divider.
                divider_x = x2 if x1 < image.shape[1] // 2 else x1
                strip = image[
                    top + 8:bottom - 8,
                    max(0, divider_x - 3):min(image.shape[1], divider_x + 4),
                ]
                if strip.size == 0:
                    continue
                divider_gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
                divider_strength = float(np.max(np.mean(
                    divider_gray < 120, axis=0
                )))
                if divider_strength < 0.45:
                    continue
                rows.append((top, bottom))

        logger.attr(
            f'WarehouseRows_{name}',
            rows
        )

        return rows

    def _initialize_position_tracker(self):
        """
        当前画面必须已经位于仓库最顶部。

        根据顶部第一页自动得到：

            第 1 行起始 Y
            正常资源行高度

        并将累计滚动距离清零。
        """

        logger.hr(
            'WAREHOUSE INIT POSITION TRACKER'
        )

        self.device.screenshot()

        left_rows = self._detect_column_rows(
            330,
            710,
            'LEFT'
        )

        right_rows = self._detect_column_rows(
            710,
            1090,
            'RIGHT'
        )

        row_sets = [rows for rows in (left_rows, right_rows) if rows]
        if not row_sets:
            logger.warning(
                'Unable to initialize warehouse position tracker: no rows'
            )
            self._warehouse_tracker_ready = False
            return False

        # ========================================
        # 第一行顶部
        # ========================================

        origins = [rows[0][0] for rows in row_sets]

        self._warehouse_row_origin = float(
            np.median(origins)
        )

        # ========================================
        # 自动计算正常行高
        # ========================================

        heights = []

        for top, bottom in [row for rows in row_sets for row in rows]:

            height = bottom - top

            if 100 <= height <= 135:
                heights.append(height)

        if heights:

            self._warehouse_row_height = float(
                np.median(heights)
            )

        else:

            self._warehouse_row_height = 120.0

        # The decorative top border can be detected a few pixels before the
        # first true row separator.  Use all top-page rows to solve the common
        # row origin instead of trusting only the first boundary.
        aligned_origins = []
        for rows in row_sets:
            for index, (top, _) in enumerate(rows[:4]):
                aligned_origins.append(
                    float(top) - index * self._warehouse_row_height
                )
        if aligned_origins:
            self._warehouse_row_origin = float(np.median(aligned_origins))

        # 当前已经明确在顶部
        self._warehouse_scroll_offset = 0.0

        # 每次重新确认顶部都重新开始闭环，不能沿用上一座城市或
        # 上一次仓库扫描留下的修正值。
        self._warehouse_drag_end_y = 58.0
        self._warehouse_scroll_step_count = 0

        self._warehouse_tracker_ready = True

        logger.attr(
            'WarehouseRowOrigin',
            self._warehouse_row_origin
        )

        logger.attr(
            'WarehouseRowHeight',
            self._warehouse_row_height
        )

        logger.attr(
            'WarehouseScrollOffset',
            self._warehouse_scroll_offset
        )

        return True

    def _update_warehouse_position_from_names(
        self, entries, raw_names, update=True
    ):
        """Use recognized ordinary resources as absolute scroll anchors."""
        if not self._warehouse_tracker_ready or self._warehouse_row_origin is None:
            return None
        layout = self.resource_layout.load(PAGE_WAREHOUSE)
        anchors = []
        for entry, raw_name in zip(entries, raw_names):
            resource = self.normalize_resource_name(raw_name)
            position = layout.get(resource)
            if position is None:
                continue
            row_index, column = position
            # Special resources use a different bottom layout and cannot share
            # the ordinary 120px-row coordinate system.
            if row_index >= len(WAREHOUSE_RESOURCE_ROWS):
                continue
            expected_column = 0 if entry['side'] == 'left' else 1
            if column != expected_column:
                continue
            top = float(entry['row'][0])
            offset = (
                self._warehouse_row_origin
                + row_index * self._warehouse_row_height
                - top
            )
            anchors.append((float(offset), resource, row_index, top))
        if not anchors:
            logger.info('Warehouse position anchors not recognized on current frame')
            return None
        values = np.array([item[0] for item in anchors], dtype=np.float32)
        center = float(np.median(values))
        tolerance = max(18.0, self._warehouse_row_height * 0.22)
        inliers = [
            item for item in anchors
            if abs(item[0] - center) <= tolerance
        ]
        # 底部普通货物只剩最后一行时，画面上可能只有“铜板”一个
        # 可用锚点。两组徽州特产分隔线同时可见时已经能独立确认底部，
        # 此时接受这个单锚点，避免退回到重复纹理的图像位移估计。
        final_row_anchor = (
            len(inliers) == 1
            and inliers[0][1] == WAREHOUSE_RESOURCE_ROWS[-1][0]
            and self._special_group_dividers() is not None
        )
        if len(inliers) < 2 and not final_row_anchor:
            logger.warning(
                f'Warehouse position anchors disagree: anchors={anchors}'
            )
            return None
        observed = max(0.0, float(np.median([
            item[0] for item in inliers
        ])))
        residual = max(
            abs(item[0] - observed) for item in inliers
        ) if inliers else 0.0
        logger.attr(
            'WarehouseVisualPosition',
            f'offset={observed:.1f}, anchors={len(inliers)}/{len(anchors)}, '
            f'residual={residual:.1f}, names='
            + ','.join(item[1] for item in inliers),
        )
        if update:
            self._warehouse_scroll_offset = observed
        return observed

    def _visible_warehouse_row_offset(self, entries, raw_names):
        """Infer list position from several recognized names without scroll pixels."""
        canonical = {
            resource: (row, column)
            for row, pair in enumerate(WAREHOUSE_RESOURCE_ROWS)
            for column, resource in enumerate(pair)
            if resource is not None
        }
        screen_rows = {'left': 0, 'right': 0}
        offsets = []
        for entry, raw_name in zip(entries, raw_names):
            side = entry['side']
            screen_row = screen_rows[side]
            screen_rows[side] += 1
            resource = self.normalize_resource_name(raw_name)
            position = canonical.get(resource)
            column = 0 if side == 'left' else 1
            if position is not None and position[1] == column:
                offsets.append(position[0] - screen_row)
        if len(offsets) < 3:
            return None
        offset = max(set(offsets), key=offsets.count)
        if offset < 0 or offsets.count(offset) < max(3, len(offsets) - 1):
            return None
        logger.attr(
            'WarehouseVisibleRowOffset',
            f'row={offset}, anchors={offsets.count(offset)}/{len(offsets)}',
        )
        return offset

    def _warehouse_page_context(self):
        """Share geometry and raw names on an exactly unchanged screenshot.

        Never cache quantities or inferred row offsets: they must still be
        read/reconciled by their callers. The copied frame also detects image
        buffers changed in place by a screenshot backend.
        """
        image = self.device.image
        cached = getattr(self, '_warehouse_context_cache', None)
        if (isinstance(image, np.ndarray) and cached is not None
                and np.array_equal(image, cached[0])):
            return [dict(entry) for entry in cached[1]], list(cached[2])
        entries = []
        for side, x1, x2 in (('left', 330, 710), ('right', 710, 1090)):
            rows = self._detect_column_rows(x1, x2, side.upper())
            name_buttons, count_buttons = self._create_row_buttons(rows, side)
            for row, name_button, count_button in zip(rows, name_buttons, count_buttons):
                entries.append({
                    'side': side, 'row': row, 'name_button': name_button,
                    'count_button': count_button,
                })
        if not entries:
            return [], []
        raw_names = self._as_list(WarehouseNameOcr(
            [entry['name_button'] for entry in entries]
        ).ocr(image))[:len(entries)]
        raw_names.extend([None] * (len(entries) - len(raw_names)))
        if isinstance(image, np.ndarray):
            self._warehouse_context_cache = (
                image.copy(), [dict(entry) for entry in entries], list(raw_names),
            )
        return entries, raw_names

    def _observe_warehouse_position(self, update=True):
        """Re-anchor from the current frame's shared visible resource names."""
        entries, raw_names = self._warehouse_page_context()
        if not entries:
            return None
        return self._update_warehouse_position_from_names(
            entries, raw_names, update=update
        )

    def _get_resource_row_index(
        self,
        screen_top
    ):
        """
        根据当前完整资源行的顶部 Y，
        计算它在整个仓库固定顺序中的行号。

        返回 0-based index。

        例如：
            0 -> 蔬菜 / 肉类
            1 -> 菜肴 / 原木
            2 -> 木板 / 纸
            ...
        """

        if not self._warehouse_tracker_ready:
            return None

        global_y = (
            self._warehouse_scroll_offset
            + screen_top
        )

        relative = (
            global_y
            - self._warehouse_row_origin
        )

        index_float = (
            relative
            / self._warehouse_row_height
        )

        index = int(
            round(index_float)
        )

        # ========================================
        # 检查误差
        # ========================================

        expected_y = (
            self._warehouse_row_origin
            + index
            * self._warehouse_row_height
        )

        error = abs(
            global_y
            - expected_y
        )

        logger.info(
            f'Warehouse row locate: '
            f'screen_y={screen_top}, '
            f'global_y={global_y:.1f}, '
            f'row={index + 1}, '
            f'error={error:.1f}'
        )

        # 正常不应该偏离一个资源行中心太多
        # 滚动后顶部经常只露出半行。它的数量区会同时包含
        # 当前值和“/库存上限”，必须排除，等下一页的完整行再读。
        # 顶部第一行的分隔线会被窗口裁掉少量，实机日志中该行会有
        # 13~14px 偏差，而其余三行仍只有 1~2px。18px 仍远小于
        # 半行距离，不会把相邻货物行混淆。
        if error > 18:

            logger.warning(
                f'Warehouse row position uncertain: '
                f'error={error:.1f}'
            )
            self.save_abnormal_screenshot('warehouse_row_uncertain')

            return None

        if (
            index < 0
            or
            index >= len(
                WAREHOUSE_RESOURCE_ROWS
            )
        ):

            return None

        return index

    def _get_expected_resource(
        self,
        row_index,
        side
    ):
        """
        根据固定资源顺序得到资源名称。
        """

        if (
            row_index is None
            or
            row_index < 0
            or
            row_index >= len(
                WAREHOUSE_RESOURCE_ROWS
            )
        ):
            return None

        column = 0 if side == 'left' else 1 if side == 'right' else None
        if column is None:
            return None
        return self.resource_layout.resource_at(
            PAGE_WAREHOUSE,
            row_index,
            column,
        )

    @staticmethod
    def _estimate_scroll_shift(
        before,
        after,
        minimum_shift=0.0,
    ):
        """
        估计仓库内容实际向上移动了多少像素。

        before / after 都是仓库列表区域。

        返回：
            (shift, score)

        shift:
            正数表示内容向上移动多少 px。
        """

        before_gray = cv2.cvtColor(
            before,
            cv2.COLOR_RGB2GRAY
        )

        after_gray = cv2.cvtColor(
            after,
            cv2.COLOR_RGB2GRAY
        )

        h, w = before_gray.shape

        # 每屏四个完整货物行，底部还会露出下一行的
        # 名称。直接用这段“未显示全的文字”作锨点：四行
        # 上滑后，它正好会出现在新一屏的开头。
        #
        # 旧实现取 h-145:h-65，四行（约 480px）滑动后
        # 这块区域已经移到屏幕外，只能靠重复短滑动猜位置。
        # 手势距离大于内容的四行位移，锨点只取最底部
        # 的文字尾部，即使实际移动接近 510px 仍有重叠可匹配。
        template_y1 = max(0, h - 15)
        template_y2 = h - 3

        # 只保留左右两列的货物名区，并拼成一条。进度条、
        # 长分隔线每行都几乎相同，若一起参与匹配会容易错到
        # 前后 1~2 行；文字区则能唯一对齐下一屏的开头货物。
        left_x1, left_x2 = 8, min(w, 210)
        right_x1, right_x2 = min(w, 388), min(w, 640)
        template = np.hstack((
            before_gray[template_y1:template_y2, left_x1:left_x2],
            before_gray[template_y1:template_y2, right_x1:right_x2],
        ))
        search = np.hstack((
            after_gray[:, left_x1:left_x2],
            after_gray[:, right_x1:right_x2],
        ))

        if (
            template.size == 0
            or
            search.size == 0
        ):
            return None, 0.0

        result = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        # 仓库名称区仍有大量相似笔画，仅取全局最高分会偶尔错到
        # 相差整整三行的位置（日志中同一手势曾被误判为 113px，
        # 随后所有货物行均错位）。正常四行手势只在预期位移附近
        # 搜索；只有已经露出仓库末尾、可见正常行很少时，调用方才
        # 会允许短位移。
        result_height = result.shape[0]
        search_y1 = max(
            0,
            int(np.floor(template_y1 - 540.0)),
        )
        search_y2 = min(
            result_height - 1,
            int(np.ceil(template_y1 - float(minimum_shift))),
        )
        if search_y2 < search_y1:
            return None, 0.0

        restricted = result[search_y1:search_y2 + 1, :]
        _, max_score, _, restricted_loc = cv2.minMaxLoc(restricted)
        max_loc = (
            restricted_loc[0],
            restricted_loc[1] + search_y1,
        )

        new_y = max_loc[1]

        shift = (
            template_y1
            - new_y
        )

        # 这里只允许“向下浏览仓库”
        if shift < 0:
            return None, max_score

        # 四行滑动约 480px，留出少量手势误差。
        if shift > 540:
            return None, max_score

        # 相似度过低说明没有可靠找到重叠区域
        if max_score < 0.60:
            return None, max_score

        return float(shift), float(max_score)

    # ========================================================
    # 动态建立 OCR 区域
    # ========================================================

    def _create_row_buttons(
        self,
        rows,
        side
    ):

        name_buttons = []
        count_buttons = []

        if side == 'left':

            name_x1 = 338
            name_x2 = 540

            # 只包含右侧数字，避开资源图标和进度条。
            count_x1 = 610
            count_x2 = 710

        else:

            name_x1 = 718
            name_x2 = 970

            count_x1 = 990
            count_x2 = 1090

        for index, (top, bottom) in enumerate(rows):

            # -----------------------------
            # 资源名称位于一行上方
            # -----------------------------

            name_area = (
                name_x1,
                top + 10,
                name_x2,
                min(
                    top + 40,
                    bottom - 5
                )
            )

            name_buttons.append(
                Button(
                    area=name_area,
                    color=(),
                    button=name_area,
                    name=(
                        f'WAREHOUSE_{side.upper()}_'
                        f'NAME_{index}'
                    ),
                )
            )

            # -----------------------------
            # 当前库存数字
            #
            # 特意不包含下面的 /3600
            # -----------------------------

            # 顶部被窗口裁掉的短行，当前数量位置更靠上；
            # 完整行则位于 top+35 左右。分开处理可避免读到下方分母。
            row_height = bottom - top
            if top < 80 and row_height >= 110:
                # 仓库首行上沿被标题遮住，实际数字比普通完整行低约
                # 25px。旧框 (top+25, top+75) 只截到 3600 的末尾。
                count_y1 = top + 40
                count_y2 = top + 92
            elif row_height < 110:
                count_y1 = top + 10
                count_y2 = top + 60
            else:
                count_y1 = top + 25
                count_y2 = top + 75

            count_area = (
                count_x1,
                count_y1,
                count_x2,
                min(
                    count_y2,
                    bottom - 5
                )
            )

            count_buttons.append(
                Button(
                    area=count_area,
                    color=(),
                    button=count_area,
                    name=(
                        f'WAREHOUSE_{side.upper()}_'
                        f'COUNT_{index}'
                    ),
                )
            )

        return (
            name_buttons,
            count_buttons
        )

    # ========================================================
    # OCR 一列
    # ========================================================

    def _read_column(
        self,
        rows,
        side
    ):

        if not rows:
            return {}

        (
            name_buttons,
            count_buttons
        ) = self._create_row_buttons(
            rows,
            side
        )

        names_raw = WarehouseNameOcr(
            name_buttons
        ).ocr(
            self.device.image
        )

        counts_raw = WarehouseCountOcr(
            count_buttons
        ).ocr(
            self.device.image
        )

        names_raw = self._as_list(
            names_raw
        )

        counts_raw = self._as_list(
            counts_raw
        )

        result = {}

        for raw_name, raw_count in zip(
            names_raw,
            counts_raw
        ):

            name = self.normalize_resource_name(
                raw_name
            )

            count = self.normalize_count(
                raw_count
            )

            # 没有名称或没有数量，
            # 说明可能是：
            #
            # 1. 不完整资源行
            # 2. 徽州特产复合格
            # 3. OCR 失败
            #
            # 普通资源解析不强行猜测。
            if not name:
                continue

            if count is None:

                logger.info(
                    f'Skip warehouse row '
                    f'without normal count: '
                    f'{name}'
                )

                continue

            result[name] = count

        return result

    def _read_column_fixed(
        self,
        rows,
        side,
        verify_names=False,
        target_resources=None,
        prefetched_names=None,
        prefetched_counts=None,
    ):
        """
        使用“固定资源顺序”读取当前一列。

        正式资源名称：
            根据位置确定。

        OCR：
            只负责数量。

        verify_names=True 时，
            才额外 OCR 名称进行调试校验。
        """

        if not rows:
            return {}

        (
            name_buttons,
            count_buttons
        ) = self._create_row_buttons(
            rows,
            side
        )

        # ========================================
        # 数量是正式数据
        #
        # 普通资源与徽州特产一样：
        # azur_lane + cnocr 双模型
        # ========================================

        expected_resources = []
        valid_indices = []
        for index, (top, _) in enumerate(rows):
            row_index = self._get_resource_row_index(top)
            resource = (
                self._get_expected_resource(row_index, side)
                if row_index is not None else None
            )
            expected_resources.append(resource)
            if resource is not None:
                valid_indices.append(index)

        # 顶部半截行和徽州特产组合区可能具有普通行的边缘轮廓，
        # 但它们无法映射到仓库位置表。不要把这些伪行送入数字 OCR，
        # 特产数量由下方专用区域单独读取。
        if len(valid_indices) != len(rows):
            rows = [rows[index] for index in valid_indices]
            name_buttons = [name_buttons[index] for index in valid_indices]
            count_buttons = [count_buttons[index] for index in valid_indices]
            expected_resources = [
                expected_resources[index] for index in valid_indices
            ]
            if not rows:
                return {}

        # 平时完全信任可编辑位置表，只 OCR 数字。仅在补扫未识别货物时
        # 批量 OCR 当前页名称，用真实行列修正候选范围并累积学习次数。
        names_raw = None
        recognized_resources = [None] * len(rows)
        if verify_names:
            names_raw = (
                list(prefetched_names)
                if prefetched_names is not None
                else self._as_list(
                    WarehouseNameOcr(name_buttons).ocr(self.device.image)
                )
            )
            names_raw.extend([None] * (len(rows) - len(names_raw)))
            column = 0 if side == 'left' else 1
            for index, raw_name in enumerate(names_raw[:len(rows)]):
                recognized = self.normalize_resource_name(raw_name)
                recognized_resources[index] = recognized or None
                row_index = self._get_resource_row_index(rows[index][0])
                if recognized and row_index is not None:
                    self.resource_layout.confirm_position(
                        PAGE_WAREHOUSE,
                        recognized,
                        (row_index, column),
                    )

        if target_resources is not None:
            target_resources = set(target_resources)
            selected = [
                index for index, resource in enumerate(expected_resources)
                if (
                    resource in target_resources
                    or recognized_resources[index] in target_resources
                )
            ]
            rows = [rows[index] for index in selected]
            name_buttons = [name_buttons[index] for index in selected]
            count_buttons = [count_buttons[index] for index in selected]
            expected_resources = [expected_resources[index] for index in selected]
            recognized_resources = [
                recognized_resources[index] for index in selected
            ]
            if names_raw is not None:
                names_raw = [
                    names_raw[index] for index in selected
                ]
            if not rows:
                return {}
        counts = (
            list(prefetched_counts)
            if prefetched_counts is not None
            else self._consensus_digit_batch(
                count_buttons,
                expected_resources,
            )
        )

        # ========================================
        # 名称只用于可选校验
        # ========================================

        result = {}

        for i, row in enumerate(rows):

            top, bottom = row

            row_index = (
                self._get_resource_row_index(
                    top
                )
            )

            if row_index is None:

                logger.warning(
                    f'Unable to locate warehouse row: '
                    f'side={side}, top={top}'
                )

                continue

            resource = (
                self._get_expected_resource(
                    row_index,
                    side
                )
            )

            # 补扫时若名称批量识别明确找到了目标货物，以识别名称为准；
            # 位置表要等同一新位置累计确认三次后才会持久化更新。
            if (
                target_resources is not None
                and i < len(recognized_resources)
                and recognized_resources[i] in target_resources
            ):
                resource = recognized_resources[i]

            # 最后一行右侧是 None
            if resource is None:
                continue

            # 仓库顺序固定且位置追踪误差已经校验在 10px 内。在这里把
            # 资源名称和实际图标绑定。运输页先用辅助信息预定位，再用
            # 这里保存的图标确认身份，位置与识别必须同时通过。
            count = counts[i] if i < len(counts) else None

            # 每次确认仓库行后都刷新仓库原生图标映射。非零库存的重复
            # 图标会作为图标带保存，运输页通过特征点匹配其中任意一枚。
            self._capture_normal_resource_icon(resource, row, side)

            # ========================================
            # 可选名称校验
            # ========================================

            if (
                verify_names
                and
                names_raw is not None
                and
                i < len(names_raw)
            ):

                raw_name = names_raw[i]

                recognized = (
                    self.normalize_resource_name(
                        raw_name
                    )
                )

                logger.info(
                    f'Warehouse verify: '
                    f'row={row_index + 1}, '
                    f'expected={resource}, '
                    f'ocr={raw_name}, '
                    f'normalized={recognized}'
                )

                if (
                    recognized
                    and
                    recognized != resource
                ):

                    logger.warning(
                        f'Warehouse resource verify mismatch: '
                        f'expected={resource}, '
                        f'OCR={recognized}'
                    )

            logger.info(
                f'Warehouse resource: '
                f'row={row_index + 1}, '
                f'side={side}, '
                f'{resource}={count}'
            )

            result[resource] = -1 if count is None else count

        return result

    # ========================================================
    # 识别当前屏幕所有完整普通资源
    # ========================================================

    def read_current_page(
        self,
        verify_names=False,
        target_resources=None,
        observed_layout=None,
    ):
        """按当前画面 OCR 名称和库存，不依赖页码或累计滑动位置。"""
        logger.hr('READ WAREHOUSE PAGE')

        self._warehouse_read_position = None
        entries, raw_names = self._warehouse_page_context()
        if not entries:
            logger.warning('No complete warehouse rows are visible')
            return {}

        # Reconstruct absolute list offset from all recognized resource names
        # before using position as an OCR correction source.
        observed_position = self._update_warehouse_position_from_names(
            entries, raw_names, update=True
        )
        if observed_position is not None:
            self._warehouse_read_position = (
                self.device.image, observed_position,
                ([dict(entry) for entry in entries], list(raw_names)),
            )
        visible_offset = self._visible_warehouse_row_offset(entries, raw_names)

        targets = set(target_resources) if target_resources is not None else None
        ordinary_names = {
            name for pair in WAREHOUSE_RESOURCE_ROWS for name in pair
            if name is not None
        }
        selected = []
        screen_rows = {'left': 0, 'right': 0}
        for entry, raw_name in zip(entries, raw_names):
            screen_row = screen_rows[entry['side']]
            screen_rows[entry['side']] += 1
            recognized = self.normalize_resource_name(raw_name)
            # The special four-item groups have ordinary-row separators too.
            # Never infer an ordinary resource from one of these group rows.
            if '特产' in str(raw_name):
                continue
            # Several recognized names give a stronger page anchor than a
            # single separator at the clipped top edge of the list.
            row_index = None
            expected = None
            if visible_offset is not None:
                inferred_row = visible_offset + screen_row
                if 0 <= inferred_row < len(WAREHOUSE_RESOURCE_ROWS):
                    column = 0 if entry['side'] == 'left' else 1
                    expected = WAREHOUSE_RESOURCE_ROWS[inferred_row][column]
                    row_index = inferred_row
            if expected is None:
                row_index = self._get_resource_row_index(entry['row'][0])
                expected = self._get_expected_resource(
                    row_index, entry['side']
                )
            resource = recognized
            if expected and recognized != expected and recognized not in ordinary_names:
                # 仓库普通货物的顺序在所有城市完全一致。滑动跟踪
                # 已把行定位误差限制在 10px 内时，固定位置比单次
                # 名称 OCR 更可靠，可修正“黏土 -> 黏十”、
                # “油墨 -> 油黑”等偶发错字或空结果。
                resource = expected
                logger.info(
                    f'Warehouse OCR corrected by position: '
                    f'row={row_index + 1}, side={entry["side"]}, '
                    f'raw={raw_name}, recognized={recognized or "<empty>"}, '
                    f'expected={expected}'
                )
            if not resource:
                logger.warning(
                    f'Warehouse row name unreadable: '
                    f'side={entry["side"]}, row={entry["row"]}, raw={raw_name}'
                )
                continue
            if targets is not None and resource not in targets:
                continue
            entry['raw_name'] = raw_name
            entry['resource'] = resource
            selected.append(entry)

        if not selected:
            if targets is not None and any(
                self.normalize_resource_name(name) for name in raw_names
            ):
                logger.info('No targeted warehouse resource on current view')
            else:
                logger.warning(
                    'No warehouse resource names recognized on current view'
                )
            return {}

        counts = self._consensus_digit_batch(
            [entry['count_button'] for entry in selected],
            [entry['resource'] for entry in selected],
        )

        canonical_positions = {
            resource: (row_index, column)
            for row_index, pair in enumerate(WAREHOUSE_RESOURCE_ROWS)
            for column, resource in enumerate(pair)
            if resource is not None
        }
        result = {}
        for index, entry in enumerate(selected):
            resource = entry['resource']
            count = counts[index] if index < len(counts) else None
            # Only a readable name may teach an icon. A positional correction
            # can recover a count, but must not overwrite another cargo's icon.
            if self.normalize_resource_name(entry['raw_name']) == resource:
                self._capture_normal_resource_icon(
                    resource, entry['row'], entry['side']
                )
            if observed_layout is not None and resource in canonical_positions:
                observed_layout[resource] = canonical_positions[resource]
            logger.info(
                f'Warehouse resource OCR: side={entry["side"]}, '
                f'raw={entry["raw_name"]}, {resource}={count}'
            )
            result[resource] = -1 if count is None else count

        logger.attr('WarehousePageInventory', result)
        return result

    def _read_current_page_fixed_legacy(
        self,
        verify_names=False,
        target_resources=None,
        observed_layout=None,
    ):
        """
        根据固定资源顺序读取当前仓库完整行。

        必须先执行：
            reset_warehouse_top()

        或者由：
            scan_inventory()

        自动初始化位置追踪。
        """

        logger.hr(
            'READ WAREHOUSE PAGE'
        )

        if not self._warehouse_tracker_ready:

            logger.warning(
                'Warehouse position tracker is not initialized. '
                'Call reset_warehouse_top() first.'
            )

            return {}

        left_rows = self._detect_column_rows(
            330,
            710,
            'LEFT'
        )

        right_rows = self._detect_column_rows(
            710,
            1090,
            'RIGHT'
        )

        # 左右栏属于同一帧。先为两栏同时建立固定位置计划：普通扫描把
        # 全部数量合成一次 OCR；校验扫描把名称和数量也合成同一批次。
        plans = []
        for rows, side in ((left_rows, 'left'), (right_rows, 'right')):
            name_buttons, count_buttons = self._create_row_buttons(rows, side)
            valid = []
            for index, (top, _) in enumerate(rows):
                row_index = self._get_resource_row_index(top)
                resource = (
                    self._get_expected_resource(row_index, side)
                    if row_index is not None else None
                )
                if resource is not None:
                    valid.append((index, resource, row_index))
            column = 0 if side == 'left' else 1
            plans.append({
                'rows': rows,
                'side': side,
                'name_buttons': [name_buttons[i] for i, _, _ in valid],
                'count_buttons': [count_buttons[i] for i, _, _ in valid],
                'resources': [resource for _, resource, _ in valid],
                'positions': [(row, column) for _, _, row in valid],
            })

        if verify_names:
            all_name_buttons = [
                button for plan in plans for button in plan['name_buttons']
            ]
            all_valid_count_buttons = [
                button for plan in plans for button in plan['count_buttons']
            ]
            # 校验模式也只提交一次：左右栏的名称和数量一起读取。数量
            # 使用本地数字归一化，不必为不同字段再拆一个模型请求。
            raw_page = (
                self._as_list(Ocr(
                    all_name_buttons + all_valid_count_buttons,
                    lang='cnocr', name='WAREHOUSE_PAGE_VERIFY_BATCH',
                ).ocr(self.device.image))
                if all_name_buttons or all_valid_count_buttons else []
            )
            raw_names = raw_page[:len(all_name_buttons)]
            raw_all_counts = raw_page[len(all_name_buttons):]
            offset = 0
            count_offset = 0
            for plan in plans:
                size = len(plan['name_buttons'])
                plan['names'] = raw_names[offset:offset + size]
                count_size = len(plan['count_buttons'])
                plan['all_counts'] = [
                    self.normalize_count(value)
                    for value in raw_all_counts[
                        count_offset:count_offset + count_size
                    ]
                ]
                offset += size
                count_offset += count_size
        else:
            for plan in plans:
                plan['names'] = None
                plan['all_counts'] = None

        all_count_buttons = []
        all_count_names = []
        for plan in plans:
            recognized = [None] * len(plan['resources'])
            if plan['names'] is not None:
                recognized = [
                    self.normalize_resource_name(value) or None
                    for value in plan['names']
                ]
                if observed_layout is not None:
                    for resource, position in zip(
                        recognized, plan['positions']
                    ):
                        if resource:
                            observed_layout[resource] = position
            if target_resources is None:
                selected = list(range(len(plan['resources'])))
            else:
                targets = set(target_resources)
                selected = [
                    index for index, resource in enumerate(plan['resources'])
                    if resource in targets or recognized[index] in targets
                ]
            plan['selected'] = selected
            all_count_buttons.extend(
                plan['count_buttons'][index] for index in selected
            )
            all_count_names.extend(
                plan['resources'][index] for index in selected
            )

        if verify_names:
            for plan in plans:
                plan['counts'] = [
                    plan['all_counts'][index]
                    for index in plan['selected']
                ]
        else:
            raw_counts = (
                self._consensus_digit_batch(all_count_buttons, all_count_names)
                if all_count_buttons else []
            )
            offset = 0
            for plan in plans:
                size = len(plan['selected'])
                plan['counts'] = raw_counts[offset:offset + size]
                offset += size

        left = self._read_column_fixed(
            left_rows,
            'left',
            verify_names=verify_names,
            target_resources=target_resources,
            prefetched_names=plans[0]['names'],
            prefetched_counts=plans[0]['counts'],
        )

        right = self._read_column_fixed(
            right_rows,
            'right',
            verify_names=verify_names,
            target_resources=target_resources,
            prefetched_names=plans[1]['names'],
            prefetched_counts=plans[1]['counts'],
        )

        result = {}

        result.update(left)
        result.update(right)

        logger.attr(
            'WarehousePageInventory',
            result
        )

        return result

    # ========================================================
    # 仓库列表画面比较
    # ========================================================

    def _warehouse_list_image(self):

        return self.device.image[
            self.LIST_Y1:self.LIST_Y2,
            self.LIST_X1:self.LIST_X2
        ].copy()

    @staticmethod
    def _image_change(
        before,
        after
    ):

        diff = cv2.absdiff(
            before,
            after
        )

        return float(
            diff.mean()
        )

    # ========================================================
    # 回顶部
    # ========================================================

    def reset_warehouse_top(
        self,
        max_swipes=15
    ):
        """
        回到仓库绝对顶部，并初始化固定资源位置追踪。
        """
        self._warehouse_bottom_confirmed = False
        self._warehouse_context_cache = None
        self._warehouse_page_frame = None
        self._warehouse_read_position = None
        logger.hr(
            'WAREHOUSE RESET TOP'
        )

        self._warehouse_tracker_ready = False

        self.device.screenshot()
        if self._warehouse_top_visible():
            logger.info('Warehouse already at top')
            return self._initialize_position_tracker()

        for _ in range(max_swipes):

            self.device.screenshot()

            before = self._warehouse_list_image()

            # 手指向下拖
            # = 内容向下
            # = 回仓库顶部
            self._warehouse_fast_swipe(
                (650, 100),
                (650, 550),
                name='WAREHOUSE_TO_TOP'
            )

            self.device.sleep(0.4)

            self.device.screenshot()

            after = self._warehouse_list_image()

            change = self._image_change(
                before,
                after
            )

            # 快速滑到顶部会出现橡皮筋回弹，而手势从边界起滑失败时
            # 也可能完全没有变化。二者都不能再由像素差直接判顶；
            # 只有首行 OCR 看到“蔬菜/肉类”才算真正到顶。
            if change < 16.0:
                logger.info(
                    f'Warehouse possible top boundary: change={change:.2f}, '
                    f'verify first-row names'
                )
            top_confirmed = self._warehouse_top_visible()

            logger.attr(
                'WarehouseListChange',
                f'{change:.2f}'
            )

            # 已经无法继续向上
            if top_confirmed:

                # 等顶部回弹完全结束后再建立首行坐标。
                self.device.sleep(0.8)
                self.device.screenshot()

                logger.info(
                    'Warehouse top reached'
                )

                # ====================================
                # 这里正式建立：
                #
                # 顶部 = 固定资源第 1 行
                # ====================================

                return (
                    self._initialize_position_tracker()
                )

            if change < 0.8:
                logger.warning(
                    'Warehouse swipe did not move, but first row is not top; '
                    'retry fast swipe from inside the list'
                )

        logger.warning(
            'Warehouse reset top reached max swipes'
        )

        return False

    # ========================================================
    # 向后浏览一段
    # ========================================================

    def _warehouse_bottom_visible(self, page_context=None):
        """Confirm the final ordinary row and both special rows on this frame."""
        if self._special_group_dividers() is None:
            return False
        entries, raw_names = (page_context if page_context is not None
                              else self._warehouse_page_context())
        bottom = any(
            entry['side'] == 'left' and self.normalize_resource_name(name) == '铜板'
            for entry, name in zip(entries, raw_names)
        )
        logger.attr('WarehouseBottomVisible', bottom)
        return bottom

    def scroll_warehouse_down(
        self,
        target_rows=4,
        max_swipes=2
    ):
        """Scroll down, using recognized resources as the position truth."""
        # A page read has just established this frame's absolute offset.
        # Consume it once, before the fresh pre-drag screenshot invalidates
        # the frame cache through background animation or tiny pixel changes.
        read_position = getattr(self, '_warehouse_read_position', None)
        self._warehouse_read_position = None
        reuse_position = (read_position is not None
                          and read_position[0] is self.device.image)
        if not self._warehouse_tracker_ready:
            logger.warning('Warehouse tracker is not ready')
            return None
        if self._warehouse_bottom_confirmed:
            logger.info('Warehouse bottom already confirmed')
            return False
        self.device.screenshot()
        if self.is_network_timeout_popup():
            self._warehouse_tracker_ready = False
            return None
        bottom_visible = (self._warehouse_bottom_visible(page_context=read_position[2])
                          if reuse_position else self._warehouse_bottom_visible())
        if bottom_visible:
            self._warehouse_bottom_confirmed = True
            self._warehouse_page_frame = self.device.image
            logger.info('Warehouse bottom confirmed by final row and special groups')
            return False

        target_shift = self._warehouse_row_height * target_rows
        stall_shift_threshold = 5.0
        logger.info(
            f'Warehouse scroll target: {target_shift:.1f}px '
            f'({target_rows} rows)'
        )

        # No gesture occurred since the page read. Reuse its confirmed offset,
        # retaining the fresh popup check above and the post-drag visual anchors
        # below. Independent/retried scrolling still obtains its own anchors.
        if reuse_position:
            observed_before = read_position[1]
            logger.attr('WarehousePreDragPositionReused', observed_before)
        else:
            observed_before = self._observe_warehouse_position(update=True)
        before_offset = (
            observed_before
            if observed_before is not None
            else self._warehouse_scroll_offset
        )
        near_bottom_offset = (
            max(0, len(WAREHOUSE_RESOURCE_ROWS) - 5)
            * self._warehouse_row_height
            - self._warehouse_row_height * 0.5
        )
        near_bottom = before_offset >= near_bottom_offset
        attempts = max(1, int(max_swipes))

        for attempt in range(attempts):
            if attempt:
                self.device.screenshot()
                if self.is_network_timeout_popup():
                    self._warehouse_tracker_ready = False
                    return None
            before = self._warehouse_list_image()
            drag_end_y = int(round(max(
                25.0,
                min(130.0, self._warehouse_drag_end_y - attempt * 8.0),
            )))
            logger.attr(
                'WarehouseAdaptiveDrag',
                f'end_y={drag_end_y}, attempt={attempt + 1}/{attempts}',
            )
            self._warehouse_drag(
                (650, 550),
                (650, drag_end_y),
                name='WAREHOUSE_DOWN'
            )
            self.device.screenshot()
            after = self._warehouse_list_image()
            if self.is_network_timeout_popup():
                logger.warning('Network popup appeared during warehouse swipe')
                self._warehouse_tracker_ready = False
                return None
            if self._warehouse_bottom_visible():
                self._warehouse_bottom_confirmed = True
                self._warehouse_page_frame = self.device.image
                logger.info('Warehouse bottom confirmed after swipe')
                return False
            change = self._image_change(before, after)
            logger.attr('WarehouseListChange', f'{change:.2f}')

            if change < 3.0:
                if near_bottom:
                    self._warehouse_bottom_confirmed = True
                    self._warehouse_page_frame = self.device.image
                    logger.info('Warehouse bottom reached')
                    return False
                logger.warning(
                    f'Warehouse drag did not move away from bottom: '
                    f'change={change:.2f}, retry={attempt + 1}/{attempts}'
                )
                continue

            # First try absolute visual anchors from all currently visible rows.
            # Only if that fails do we fall back to image-translation matching.
            observed_after = self._observe_warehouse_position(update=False)
            source = 'visual_anchors'
            score = 1.0
            if observed_after is not None:
                shift = observed_after - before_offset
            else:
                source = 'image_match_fallback'
                minimum_shift = (
                    0.0 if near_bottom else max(
                        0.0,
                        target_shift - self._warehouse_row_height * 1.5,
                    )
                )
                shift, score = self._estimate_scroll_shift(
                    before,
                    after,
                    minimum_shift=minimum_shift,
                )

            logger.attr(
                'WarehouseScrollPositionSource',
                f'{source}, score={score:.3f}'
            )
            if shift is None:
                logger.warning('Unable to determine warehouse scroll distance')
                self._warehouse_tracker_ready = False
                return None
            if shift <= stall_shift_threshold:
                if near_bottom:
                    logger.info(f'Warehouse bottom stall: shift={shift:.1f}px')
                    self._warehouse_bottom_confirmed = True
                    self._warehouse_page_frame = self.device.image
                    return False
                logger.warning(
                    f'Warehouse zero-shift before bottom: '
                    f'shift={shift:.1f}px, retry={attempt + 1}/{attempts}'
                )
                continue

            if observed_after is not None:
                self._warehouse_scroll_offset = max(0.0, observed_after)
            else:
                self._warehouse_scroll_offset = max(
                    0.0, before_offset + float(shift)
                )

            # Correct the next gesture from what the screen actually moved,
            # instead of from an accumulated nominal page counter.
            error = target_shift - float(shift)
            correction = max(-40.0, min(40.0, error * 0.70))
            self._warehouse_drag_end_y = max(
                25.0,
                min(130.0, self._warehouse_drag_end_y - correction),
            )
            logger.attr(
                'WarehouseScrollCorrection',
                f'observed_shift={shift:.1f}px, error={error:.1f}px, '
                f'next_end_y={self._warehouse_drag_end_y:.1f}',
            )
            logger.attr('WarehouseScrollStep', f'{shift:.1f}')
            logger.attr(
                'WarehouseScrollOffset',
                f'{self._warehouse_scroll_offset:.1f}'
            )

            logger.info(
                f'Warehouse scroll complete: {shift:.1f}px '
                f'(target={target_shift:.1f}px, source={source})'
            )
            self._warehouse_page_frame = self.device.image
            return True

        logger.warning('Warehouse did not move before expected bottom')
        self._warehouse_tracker_ready = False
        return None

    def _special_zero_confirmed(self, button):
        """Recognize an isolated centred 0 only after both OCR reads failed."""
        number = crop(self.device.image, button.area)
        if number.size == 0:
            return False
        ink = cv2.inRange(cv2.cvtColor(number, cv2.COLOR_RGB2GRAY), 0, 105)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        glyphs = [
            (x, y, w, h) for x, y, w, h, area in stats[1:]
            if area >= 15 and h >= 10
        ]
        if len(glyphs) != 1:
            return False
        x, y, w, h = glyphs[0]
        if not (7 <= w <= 18 and 12 <= h <= 24):
            return False
        glyph = ink[y:y + h, x:x + w]
        # The captured serif zero is symmetric in both axes. Require this
        # independent shape check so a single counter in a 6 cannot pass.
        if (np.mean(glyph != np.fliplr(glyph)) > 0.12
                or np.mean(glyph != np.flipud(glyph)) > 0.12):
            return False
        contours, hierarchy = cv2.findContours(
            glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        if hierarchy is None:
            return False
        holes = [contour for contour, item in zip(contours, hierarchy[0])
                 if item[3] >= 0 and cv2.contourArea(contour) >= 5]
        if len(holes) != 1:
            return False
        moments = cv2.moments(holes[0])
        cx = moments['m10'] / moments['m00'] / w
        cy = moments['m01'] / moments['m00'] / h
        # The counters in 6 and 9 are off-centre; 8 has two counters.
        return 0.30 <= cx <= 0.65 and 0.32 <= cy <= 0.62

    def read_special_items(self, target_resources=None):

        logger.hr(
            'READ WAREHOUSE SPECIAL ITEMS'
        )

        result = {}
        dividers = self._special_group_dividers()
        if dividers is None:
            logger.warning('Warehouse special rows not located; skip uncertain counts')
            self.save_abnormal_screenshot('warehouse_special_rows_missing')
            return result
        right_divider, left_divider = dividers
        aligned = abs(right_divider - left_divider) <= 6
        count_buttons = {}
        slot_dividers = {}
        for name, reference in WAREHOUSE_SPECIAL_COUNT_SLOTS.items():
            x1, _, x2, _ = reference.area
            divider = right_divider if x1 >= 710 else left_divider
            if aligned:
                # Copper mirrors complete the final ordinary row. Tea/wine
                # now occupy the left group, carvings the right, on one row.
                shift = -380 if x1 >= 710 else 380
                x1, x2 = x1 + shift, x2 + shift
            slot_dividers[name] = divider
            area = (x1, divider - 29, x2, divider - 5)
            count_buttons[name] = Button(
                area=area, color=(), button=area,
                name=f'WAREHOUSE_SPECIAL_COUNT_{name}',
            )
        selected = [
            (name, button)
            for name, button in count_buttons.items()
            if target_resources is None or name in target_resources
        ]

        for name, button in selected:

            logger.info(
                f'Warehouse special count area: '
                f'{name}={button.area}'
            )

            center = self.SPECIAL_ICON_CENTERS.get(name)
            if center is not None:
                if aligned:
                    center = (button.area[0] + 35, slot_dividers[name] - 48)
                    offset = 0
                else:
                    offset = (right_divider - 456 if center[0] >= 710
                              else left_divider - 576)
                self._save_resource_icon_template(
                    name, (center[0], center[1] + offset)
                )

        slots = [
            (name, button.area[0], slot_dividers[name])
            for name, button in selected
        ]
        try:
            result, diagnostics = read_special_digits(
                self.device.image, slots,
                PPOCR_V6_MEDIUM.atomic_ocr_for_single_lines_scored,
            )
        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
            logger.warning(f'Warehouse special digit recognition failed: {exc}')
            result, diagnostics = {}, {'error': str(exc)}
        missing = [name for name, _ in selected if name not in result]
        if missing:
            logger.warning(f'Unable to confirm special items: {missing}; reads={diagnostics}')
            # Keep a full screenshot and exact crop/read metadata at failure,
            # so future -1 values can be checked against the actual frame.
            path = self.save_abnormal_screenshot('warehouse_special_unreadable')
            if isinstance(path, (str, Path)):
                try:
                    Path(path).with_suffix('.json').write_text(json.dumps({
                        'missing': missing, 'slots': slots, 'readings': diagnostics,
                    }, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError as exc:
                    logger.warning(f'Unable to save warehouse digit diagnostics: {exc}')

        logger.attr(
            'WarehouseSpecialInventory',
            result
        )

        return result

    def _special_group_dividers(self):
        """Locate tea/wine then carving separators in either warehouse layout."""
        image = self.device.image
        if image is None or image.shape[0] < 580:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # The updated warehouse has two aligned special groups. Locate both
        # real title labels before accepting their lower separators, so an
        # ordinary resource row cannot masquerade as a special group.
        reference = getattr(self, '_special_header_template', None)
        if reference is None:
            from pathlib import Path
            path = Path(__file__).resolve().parents[2] / 'assets/jiangnan/warehouse/special_header.png'
            reference = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
            self._special_header_template = reference
        headers = []
        for x in (330, 710):
            scores = cv2.matchTemplate(gray[270:580, x:x+200], reference, cv2.TM_CCOEFF_NORMED)
            _, score, _, (_, top) = cv2.minMaxLoc(scores)
            if score < .88:
                break
            headers.append(top + 270)
        if len(headers) == 2 and abs(headers[0] - headers[1]) <= 6:
            aligned = []
            for x, header in zip((340, 720), headers):
                y1, y2 = header + 85, min(header + 125, 595)
                darkness = np.mean(gray[y1:y2, x:x+360] < 105, axis=1)
                rows = np.flatnonzero(darkness >= .45)
                if not len(rows):
                    return None
                # The thick outer frame sits below the thin row separator.
                # Its darker maximum would crop off the top of digits like 6.
                line = int(rows[0])
                while line + 1 < len(darkness) and darkness[line + 1] >= .45:
                    line += 1
                aligned.append(y1 + line + 1)
            if abs(aligned[0] - aligned[1]) <= 6:
                logger.attr('WarehouseSpecialAlignedDividers', aligned)
                # Return tea/wine then carving, consistent with legacy callers.
                return tuple(aligned)
            return None
        candidates = []
        for x1, x2, y1, y2 in (
            (720, 1080, 340, 475),
            (340, 700, 460, 580),
        ):
            darkness = np.mean(gray[y1:y2, x1:x2] < 105, axis=1)
            row = int(np.argmax(darkness))
            if darkness[row] < 0.45:
                return None
            candidates.append(y1 + row)
        right, left = candidates
        if not 115 <= left - right <= 125:
            logger.warning(
                f'Warehouse special row spacing uncertain: {candidates}'
            )
            return None
        logger.attr('WarehouseSpecialDividers', f'right={right}, left={left}')
        return right, left

    def read_special_capacities(self):
        """读取徽州两组四物共享的仓库上限。"""
        result = {}
        dividers = self._special_group_dividers()
        if dividers is None:
            return result
        right, left = dividers
        aligned = abs(right - left) <= 6
        selected = []
        for group, reference in WAREHOUSE_SPECIAL_CAPACITY_SLOTS.items():
            x1, _, x2, _ = reference.area
            divider = right if x1 >= 710 else left
            if aligned:
                shift = -380 if x1 >= 710 else 380
                x1, x2 = x1 + shift, x2 + shift
            area = (x1, divider - 42, x2, divider + 2)
            selected.append((group, Button(
                area=area, color=(), button=area,
                name=f'WAREHOUSE_SPECIAL_CAPACITY_{group}',
            )))
        values = self._consensus_digit_batch(
            [button for _, button in selected],
            [group for group, _ in selected],
            verify_over_capacity=False,
        )
        for (group, _), value in zip(selected, values):
            if value is not None and value > 0:
                result[group] = value
        logger.attr('WarehouseSpecialCapacities', result)
        return result

    # ========================================================
    # 合并库存
    # ========================================================

    @staticmethod
    def _merge_inventory(
        inventory,
        new_data,
        capacity=None,
        conflicts=None,
    ):

        for name, count in new_data.items():

            if conflicts is not None and name in conflicts:
                continue

            # 第一次识别
            if name not in inventory:

                inventory[name] = count
                continue

            old_count = inventory[name]

            # 两次结果一样
            if old_count == count:
                continue

            # Neither capacity nor read order establishes which OCR value is
            # correct. Keep conflicting valid readings unknown for this scan;
            # otherwise a later overlapping page can silently revive a bad 0.
            if int(old_count) >= 0 and int(count) >= 0:
                final_count = -1
                if conflicts is not None:
                    conflicts.add(name)
            elif int(count) >= 0:
                final_count = int(count)
            else:
                final_count = int(old_count)

            logger.warning(
                f'Warehouse inventory OCR differs: '
                f'{name}, '
                f'old={old_count}, '
                f'new={count}, '
                f'resolved={final_count}'
            )

            inventory[name] = final_count

    # ========================================================
    # 完整扫描
    # ========================================================

    def scan_inventory(
        self,
        max_pages=30,
        return_main=True,
        target_resources=None,
        learn_layout=False,
        city_name=None,
    ):
        """
        完整读取仓库库存。

        流程：

            主界面
            -> 营造
            -> 查看仓库
            -> 回到列表顶部
            -> 逐屏读取普通资源
            -> 滚动
            -> 判断到底
            -> 读取徽州特产
            -> 返回主界面

        Returns:
            dict:
                {
                    '蔬菜': 8948,
                    '肉类': 2465,
                    ...
                }

            None:
                扫描未完整完成
        """

        logger.hr(
            'JIANGNAN WAREHOUSE SCAN',
            level=0
        )


        # 防止切换城市后沿用上一座城市的容量。
        self.warehouse_capacity = None
        self.warehouse_special_capacity = {}
        self.merchant_capacity = None

        if not self.goto_warehouse():

            logger.warning(
                'Unable to open warehouse'
            )

            return None

        target_resources = (
            set(target_resources) if target_resources is not None else None
        )
        inventory = {}
        conflicts = set()
        observed_layout = {} if learn_layout else None
        complete = False

        try:

            if city_name == '绍兴府' and not self.select_shaoxing_warehouse('城市'):
                return None

            if not self.reset_warehouse_top():

                logger.warning(
                    'Unable to reset warehouse top'
                )

                return None

            # 第一页每一项都显示“当前数量/库存上限”。左右两列读取四次
            # （两列 x 两个 OCR 模型），得到当前城市真实的单项库存上限。
            self.device.screenshot()
            self.warehouse_capacity = self.read_warehouse_capacity()

            for page in range(max_pages):

                logger.hr(
                    f'WAREHOUSE PAGE {page + 1}'
                )

                # Scrolling has just captured and located this exact frame.
                # Reuse it before another operation can replace device.image.
                if (page == 0 or getattr(self, '_warehouse_page_frame', None)
                        is not self.device.image):
                    self.device.screenshot()
                if self.is_network_timeout_popup():
                    logger.warning('Network popup interrupted warehouse scan')
                    return None

                current = self.read_current_page(
                    verify_names=(
                        target_resources is not None or learn_layout
                    ),
                    target_resources=target_resources,
                    observed_layout=observed_layout,
                )

                self._merge_inventory(
                    inventory,
                    current,
                    capacity=self.warehouse_capacity,
                    conflicts=conflicts,
                )

                logger.attr(
                    'WarehouseItemCount',
                    len(inventory)
                )

                # 滚动一次。
                #
                # False = 已经到底。
                scroll_result = (
                    self.scroll_warehouse_down()
                )

                # ========================================
                # 已到底
                # ========================================

                if scroll_result is False:

                    if getattr(self, '_warehouse_page_frame', None) is not self.device.image:
                        self.device.screenshot()
                    if self.is_network_timeout_popup():
                        logger.warning('Network popup interrupted warehouse scan')
                        return None

                    # A drag may reveal the final ordinary row and report
                    # bottom in the same step.  The pre-drag page did not
                    # contain that row (notably 铜板), so read this frame
                    # before collecting the special groups below it.
                    final_normal = self.read_current_page(
                        verify_names=(
                            target_resources is not None or learn_layout
                        ),
                        target_resources=target_resources,
                        observed_layout=observed_layout,
                    )
                    self._merge_inventory(
                        inventory, final_normal,
                        capacity=self.warehouse_capacity,
                        conflicts=conflicts,
                    )

                    special = self.read_special_items(
                        target_resources=target_resources,
                    )

                    self._merge_inventory(
                        inventory,
                        special,
                        conflicts=conflicts,
                    )
                    self.warehouse_special_capacity = (
                        self.read_special_capacities()
                    )

                    expected_normal = [
                        name for name in WAREHOUSE_RESOURCES
                        if target_resources is None or name in target_resources
                    ]
                    expected_special = [
                        name for name in WAREHOUSE_SPECIAL_RESOURCES
                        if target_resources is None or name in target_resources
                    ]
                    missing = [
                        name for name in expected_normal + expected_special
                        if name not in inventory
                    ]
                    missing_capacity_groups = [
                        name for name in WAREHOUSE_SPECIAL_CAPACITY_SLOTS
                        if name not in self.warehouse_special_capacity
                    ]
                    if missing:
                        logger.warning(
                            f'Warehouse resources unreadable, fill -1: '
                            f'{missing}'
                        )
                        inventory.update({name: -1 for name in missing})
                    if self.warehouse_capacity is None:
                        logger.warning(
                            'Warehouse scan incomplete, capacity is missing'
                        )
                    if missing_capacity_groups:
                        logger.warning(
                            f'Warehouse scan incomplete, missing shared '
                            f'capacities: {missing_capacity_groups}'
                        )
                    complete = not (
                        self.warehouse_capacity is None
                        or missing_capacity_groups
                    )

                    if complete and learn_layout:
                        default_layout = self.resource_layout.load(
                            PAGE_WAREHOUSE
                        )
                        for resource in WAREHOUSE_SPECIAL_RESOURCES:
                            position = default_layout.get(resource)
                            if position is not None:
                                observed_layout[resource] = position
                        self.resource_layout.replace_layout(
                            PAGE_WAREHOUSE, observed_layout
                        )

                    break


                # ========================================
                # 滚动定位失败
                #
                # 宁可停止，也不能把资源数量
                # 记到错误的资源名下面。
                # ========================================

                if scroll_result is None:

                    logger.warning(
                        'Warehouse position tracking lost, '
                        'stop inventory scan'
                    )

                    complete = False

                    break

            if not complete:

                logger.warning(
                    'Warehouse scan reached max pages '
                    'before bottom'
                )

            if complete and city_name == '绍兴府':
                merchant_targets = (
                    set(SHAOXING_MERCHANT_RESOURCES)
                    if target_resources is None else
                    set(target_resources) & set(SHAOXING_MERCHANT_RESOURCES)
                )
                if merchant_targets:
                    merchant = self.scan_shaoxing_merchant(
                        target_resources=merchant_targets,
                    )
                    if merchant is None:
                        complete = False
                    else:
                        inventory.update(merchant)
                    if self.merchant_capacity:
                        self.warehouse_special_capacity['绍兴商行'] = (
                            self.merchant_capacity
                        )

        finally:

            if return_main:

                logger.hr(
                    'WAREHOUSE RETURN MAIN'
                )

                result = self.ui_goto_main()

                logger.attr(
                    'WarehouseReturnMain',
                    result
                )

        if not complete:

            return None

        self.warehouse_inventory = inventory

        logger.hr(
            'WAREHOUSE INVENTORY RESULT'
        )

        # ============================================================
        # 按仓库固定顺序输出普通资源
        # ============================================================

        logger.info(
            '========== NORMAL RESOURCES =========='
        )

        for row_index, (left, right) in enumerate(
            WAREHOUSE_RESOURCE_ROWS,
            start=1
        ):

            left_value = (
                inventory.get(left, 'MISSING')
                if left is not None
                else '-'
            )

            right_value = (
                inventory.get(right, 'MISSING')
                if right is not None
                else '-'
            )

            logger.info(
                f'{row_index:02d}  '
                f'{left or "-":<8} {left_value!s:<8} | '
                f'{right or "-":<8} {right_value}'
            )


        # ============================================================
        # 输出特殊资源
        # ============================================================

        logger.info(
            '========== SPECIAL RESOURCES =========='
        )

        for name in WAREHOUSE_SPECIAL_RESOURCES:

            value = inventory.get(
                name,
                'MISSING'
            )

            logger.info(
                f'{name}: {value}'
            )


        logger.attr(
            'WarehouseTotalItems',
            len(inventory)
        )

        return inventory

    def scan_inventory_by_city(
        self,
        cities=None,
        output_file='jiangnan_warehouse_inventory_by_city.json',
        max_pages=30,
        max_attempts=2,
        target_resources_by_city=None,
        on_city_scanned=None,
    ):
        """
        逐个城市切换并导出仓库库存。

        Returns:
            dict[str, dict[str, int] | None]
        """

        logger.hr(
            'JIANGNAN WAREHOUSE SCAN BY CITY',
            level=0
        )

        if cities is None:
            cities = list(KNOWN_CITIES)

        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        def save_partial_results():
            """每扫完一城就落盘，长时间全量扫描中途也不会丢数据。"""
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open('w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(
                    f'Warehouse inventory export saved: {output_path}'
                )
            except Exception as e:
                logger.warning(
                    f'Unable to save warehouse inventory export: {e}'
                )

        results = {}
        capacities = {}
        special_capacities = {}
        expected_by_city = {}

        city_helper = JiangnanCity(
            config=self.config,
            device=self.device,
            task=getattr(self, 'task', None)
        )

        target_resources_by_city = target_resources_by_city or {}

        # 第一阶段：每座城市只进行一次正常扫描。不要在城市之间穿插
        # OCR 重试，否则一次全量任务看起来会把同一座城市连续扫多遍，
        # 也无法在补扫前得到完整的 -1 汇总。
        for city_name in cities:
            logger.hr(
                f'WAREHOUSE CITY: {city_name}'
            )

            requested = target_resources_by_city.get(city_name)
            requested = set(requested) if requested is not None else None
            logger.info(
                f'Warehouse city full scan: {city_name}, '
                f'targets={"all" if requested is None else sorted(requested)}'
            )
            scanned = None
            if city_helper.switch_city(city_name):
                scanned = self.scan_inventory(
                    max_pages=max_pages,
                    return_main=True,
                    target_resources=requested,
                    city_name=city_name,
                )
            else:
                logger.warning(f'Unable to switch city: {city_name}')

            # 记录本城应识别的货物集合，供唯一一次补扫复用。整城导航
            # 失败用 None 表示；只有已经进入仓库但个别货物读不到时，
            # 才允许对应货物保留 -1。
            expected = (
                set(requested)
                if requested is not None
                else set(WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES)
                | (set(SHAOXING_MERCHANT_RESOURCES)
                   if city_name == '绍兴府' else set())
            )
            expected_by_city[city_name] = expected
            if scanned is None:
                # 切城或进仓库失败不是一整仓货物都“无法识别”。保留明确
                # 的失败状态供补扫使用，但绝不能用 65 个 -1 覆盖 GUI
                # 中该城市上一次成功扫描的库存。
                results[city_name] = None
                capacities[city_name] = None
                special_capacities[city_name] = {}
                logger.warning(
                    f'Warehouse scan failed in city: {city_name}'
                )
                save_partial_results()
                continue

            inventory = dict(scanned or {})
            for name in expected:
                inventory.setdefault(name, -1)
            results[city_name] = inventory
            capacities[city_name] = (
                self.warehouse_capacity if scanned is not None else None
            )
            special_capacities[city_name] = (
                dict(self.warehouse_special_capacity)
                if scanned is not None else {}
            )

            unknown = sorted(
                name for name, value in inventory.items()
                if int(value) < 0
            )
            if unknown:
                if max_attempts > 1:
                    logger.warning(
                        f'Warehouse unreadable resources queued for one '
                        f'supplemental scan in {city_name}: {unknown}'
                    )
                else:
                    logger.info(
                        f'Warehouse unreadable resources kept unknown until '
                        f'next scheduled scan in {city_name}: {unknown}'
                    )

            save_partial_results()
            if on_city_scanned is not None:
                try:
                    on_city_scanned(
                        city_name,
                        inventory,
                        capacities[city_name],
                        special_capacities[city_name],
                    )
                except Exception as e:
                    logger.exception(
                        f'Unable to persist warehouse city result: '
                        f'{city_name}, {e}'
                    )

        # 第二阶段：完整城市轮次结束后，只访问仍含 -1 的城市，并且只
        # 请求这些 -1 项。无论结果如何都不再进行第三轮。
        if max_attempts > 1:
            retry_cities = [
                city_name for city_name in cities
                if results.get(city_name) is None
                or any(
                    int(value) < 0
                    for value in results.get(city_name, {}).values()
                )
            ]
            logger.hr('WAREHOUSE SUPPLEMENTAL UNKNOWN SCAN')
            logger.attr('WarehouseSupplementalCities', retry_cities)
            for city_name in retry_cities:
                if results.get(city_name) is None:
                    unknown = set(expected_by_city.get(city_name, ()))
                else:
                    unknown = {
                        name for name, value in results[city_name].items()
                        if int(value) < 0
                    }
                if not unknown:
                    continue
                logger.info(
                    f'Warehouse supplemental scan: {city_name}, '
                    f'targets={sorted(unknown)}'
                )
                if not city_helper.switch_city(city_name):
                    logger.warning(
                        f'Unable to switch city for supplemental scan: '
                        f'{city_name}'
                    )
                    continue
                scanned = self.scan_inventory(
                    max_pages=max_pages,
                    return_main=True,
                    target_resources=unknown,
                    city_name=city_name,
                )
                if scanned is not None:
                    if results.get(city_name) is None:
                        results[city_name] = {
                            name: -1 for name in expected_by_city.get(
                                city_name,
                                (),
                            )
                        }
                    for name, value in scanned.items():
                        if name in unknown and int(value) >= 0:
                            results[city_name][name] = int(value)
                    if self.warehouse_capacity is not None:
                        capacities[city_name] = self.warehouse_capacity
                    if self.warehouse_special_capacity:
                        special_capacities[city_name] = dict(
                            self.warehouse_special_capacity
                        )
                remaining = (
                    sorted(expected_by_city.get(city_name, ()))
                    if results.get(city_name) is None
                    else sorted(
                        name for name, value in results[city_name].items()
                        if int(value) < 0
                    )
                )
                logger.attr(
                    f'WarehouseUnknownAfterSupplement_{city_name}',
                    remaining,
                )
                save_partial_results()
                if on_city_scanned is not None and results.get(city_name) is not None:
                    try:
                        on_city_scanned(
                            city_name,
                            results[city_name],
                            capacities[city_name],
                            special_capacities[city_name],
                        )
                    except Exception as e:
                        logger.exception(
                            f'Unable to persist supplemental warehouse '
                            f'result: {city_name}, {e}'
                        )

        self.warehouse_capacity_by_city = capacities
        self.warehouse_special_capacity_by_city = special_capacities
        logger.attr('WarehouseCapacityByCity', capacities)
        logger.attr('WarehouseSpecialCapacityByCity', special_capacities)

        return results
