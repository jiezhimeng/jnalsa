import re
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from module.base.button import Button
from module.base.utils import crop, load_image
from module.jiangnan.assets import (
    ALLOCATION_AMOUNT_BUTTON,
    ALLOCATION_ASSIGNEE_CHECK,
    ALLOCATION_DEPART_BUTTON,
    ALLOCATION_DESTINATION_CHECK,
    ALLOCATION_DIRECT_START_BUTTON,
    ALLOCATION_OVERVIEW_CHECK,
    ALLOCATION_RESOURCE_CHECK,
)
from module.jiangnan.city import JiangnanCity, KNOWN_CITIES
from module.jiangnan.resource import (
    AUTOMATIC_EXPORT_RESOURCES,
    WAREHOUSE_OCR_ALIASES,
    WAREHOUSE_RESOURCES,
    WAREHOUSE_SPECIAL_RESOURCES,
)
from module.jiangnan.resource_layout import (
    ResourceLayoutManager,
    allocation_layout_page,
)
from module.logger import logger
from module.jiangnan.ocr import Digit, Ocr, read_confirmed_digits
from module.jiangnan.vision import stable_feature_match
from module.jiangnan.resident_reservation import is_reserved_resident, reserved_special_names
from module.jiangnan.resident_selection import unarranged_assignee_cards


CITY_KEY_TO_NAME = {
    'yingtianfu': '应天府',
    'suzhoufu': '苏州府',
    'hangzhoufu': '杭州府',
    'songjiangfu': '松江府',
    'huizhoufu': '徽州府',
    'yangzhoufu': '扬州府',
    'shaoxingfu': '绍兴府',
    'ningbofu': '宁波府',
    'jimingshan': '鸡鸣山',
    'jiangningxian': '江宁县',
    'baixuezhen': '白雪镇',
    'zongyingdao': '棕影岛',
    'longgong': '龙宫',
}


class AllocationAmountOcr(Ocr):
    """识别运输页面左侧当前填写的数量。"""

    DEFAULT_AREA = (464, 224, 544, 260)

    def __init__(self, area=None, lang='cnocr', threshold=125):
        area = area or self.DEFAULT_AREA
        self.binary_threshold = threshold
        super().__init__(
            buttons=Button(
                area=area,
                color=(),
                button=area,
                name='ALLOCATION_AMOUNT_OCR',
            ),
            lang=lang,
            alphabet='0123456789',
            name='ALLOCATION_AMOUNT_OCR',
        )

    def pre_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.inRange(gray, 0, self.binary_threshold)
        return cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )


class AllocationCapacityOcr(Ocr):
    """识别目的地仓库的当前剩余容量。"""

    # 只截取冒号右侧数字。把“目的地仓库剩余容量”整句送给纯数字 OCR
    # 时，中文笔画会产生 4、433 等伪候选。
    DEFAULT_AREA = (480, 116, 620, 153)

    def __init__(self, area=None, lang='cnocr', threshold=135):
        area = area or self.DEFAULT_AREA
        self.binary_threshold = threshold
        super().__init__(
            buttons=Button(
                area=area,
                color=(),
                button=area,
                name='ALLOCATION_CAPACITY_OCR',
            ),
            lang=lang,
            alphabet='0123456789',
            name='ALLOCATION_CAPACITY_OCR',
        )

    def pre_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.inRange(gray, 0, self.binary_threshold)
        return cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )


class AllocationInTransitOcr(AllocationCapacityOcr):
    """读取目的地已在途的同类物资，不把它当成已入库库存。"""

    DEFAULT_AREA = (530, 150, 620, 185)


class AllocationDurationOcr(Ocr):
    """识别点击“出发”前页面底部显示的运输时间。"""

    DEFAULT_AREA = (635, 580, 875, 645)
    FIRST_NUMBER_AREA = (680, 602, 702, 642)
    SECOND_NUMBER_AREA = (752, 602, 790, 642)

    def __init__(self, area=None, threshold=150):
        area = area or self.DEFAULT_AREA
        self.binary_threshold = threshold
        super().__init__(
            buttons=Button(
                area=area,
                color=(),
                button=area,
                name='ALLOCATION_DURATION_OCR',
            ),
            lang='cnocr',
            name='ALLOCATION_DURATION_OCR',
        )

    def pre_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.inRange(gray, 0, self.binary_threshold)
        return cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )


class AllocationSlotStatusOcr(Ocr):
    """读取城镇总览船位顶部的“空闲”文字。"""

    def __init__(self, area):
        super().__init__(
            buttons=Button(
                area=area,
                color=(),
                button=area,
                name='ALLOCATION_SLOT_STATUS_OCR',
            ),
            lang='cnocr',
            name='ALLOCATION_SLOT_STATUS_OCR',
        )


class JiangnanAllocation(JiangnanCity):
    """执行一笔明确的跨城物资运输订单。"""

    OVERVIEW_TEMPLATE = './assets/jiangnan/allocation/overview.png'
    # CITY_LIST_SLOTS 返回的是名称 OCR 格位的中心。第一列实际中心为
    # x=253；旧值 251 会把只有 58px 宽的“空闲”模板整体右移 2px，
    # 在底部列表的纸张纹理上足以令匹配分数从 0.86 跌成负数。
    OVERVIEW_REFERENCE_CITY_CENTER = (253, 264)
    IDLE_SLOT_REFERENCE = (
        # “空闲”文字区域、整张船位点击区域
        ((316, 182, 374, 218), (306, 177, 380, 286)),
        ((401, 182, 459, 218), (391, 177, 465, 286)),
    )

    RESOURCE_GRID_AREA = (624, 92, 1075, 480)
    RESOURCE_GRID_X = (683, 793, 903, 1014)
    RESOURCE_FIRST_ROW_Y = 140
    RESOURCE_ROW_HEIGHT = 110
    RESOURCE_VIEW_ROWS = 3
    RESOURCE_VIEW_STRIDE = 2
    RESOURCE_SAFE_BOTTOM = 438
    RESOURCE_ICON_DIR = Path('assets/jiangnan/allocation/resource_icons')
    # 仓库裁下来的货物图标与运输页存在缩放差异，但实际日志中可靠匹配
    # 基本都在 0.46 以上。旧阈值 0.18 会让 0.32 左右的相似轮廓也被
    # 当成目标货物；这种误选直到出发后才暴露，代价过高，宁可跳过。
    # 仓库截图是同一图标连续平铺的横条，而运输页只显示一枚图标。
    # 全页搜索以“领先同屏第二名”共同判定，不能沿用固定格的高门槛。
    RESOURCE_ICON_THRESHOLD = 0.22
    # 辅助表已经把候选约束到唯一行列，仓库横条模板拆成单图标后只需
    # 通过较低的基础特征门槛；最终仍由目的地剩余容量二次确认。
    RESOURCE_ICON_FIXED_THRESHOLD = 0.16
    # 图标达到此分数且点击格与辅助信息完全一致时，身份已经足够可靠。
    # 目的地库存可能在两次仓库扫描之间被生产、运输或税课司改变，不能
    # 再用旧库存推导出的剩余容量否决一次强图标识别。
    RESOURCE_ICON_STRONG_THRESHOLD = 0.40
    RESOURCE_ICON_CELL_MARGIN = 0.055
    # 页面定位不是靠单枚图标拍脑袋，而是随机抽样多个格子并用辅助信息
    # 中的固定顺序共同投票。单枚图标分数可以略低，但至少四枚必须支持
    # 同一个全局行，且要明显领先第二种页面位置。
    RESOURCE_CALIBRATION_SAMPLE_SIZE = 10
    RESOURCE_CALIBRATION_ICON_THRESHOLD = 0.22
    RESOURCE_CALIBRATION_MARGIN = 0.05

    def __init__(self, config, device=None, task=None):
        super().__init__(config=config, device=device, task=task)
        self.last_error = None
        self.last_amount = 0
        self.last_duration = None
        self.last_slot_index = None
        self._idle_templates = [
            load_image(self.OVERVIEW_TEMPLATE, area)
            for area, _ in self.IDLE_SLOT_REFERENCE
        ]
        # The short 空闲 label stays visually stable when the surrounding
        # paper/ship texture changes.  The full-slot template scored 0.351
        # on a genuinely idle 徽州府 slot while this text patch scored 0.946.
        self._idle_label_template = cv2.cvtColor(
            load_image(self.OVERVIEW_TEMPLATE, (407, 183, 453, 207)),
            cv2.COLOR_RGB2GRAY,
        )
        self._resource_drag_seq = 0
        # 进入“选择要运送的物资”页时列表固定在顶部。
        # 后续记录 0/2/4... 这些三行视图的首行；若最后一个合法
        # 首行是奇数（例如 6 行列表的末屏为 3），最后一步只滑一行。
        # 常规步进仍滑两行，保留一行重叠作为定位冗余。
        # 定位时不再为了判断当前页反复识别整屏图标。
        self._resource_view_start = None
        # Absolute visual scroll offset from the physical top.  This is updated
        # from what the frame actually moved, never from a nominal page count.
        self._resource_scroll_offset = None
        self.last_resource_screen = None
        self.last_resource_grid = None
        self.last_resource_icon_score = 0.0
        self._resource_identity_tentative = False
        self.last_reconciled_inventory = None
        self.last_reconciled_source_inventory = None
        self._resource_icon_templates = {}
        self._resource_icon_features = {}
        self._resource_cell_feature_frame = None
        self._resource_cell_feature_cache = {}
        self._resource_cell_content_cache = {}
        self._resource_icon_band_cache = {}
        self._resource_sift = cv2.SIFT_create(
            nfeatures=100,
            contrastThreshold=0.02,
        )
        self.resource_layout = ResourceLayoutManager(self.config)
        # 城镇和货物位置共用同一学习状态缓存，避免同一轮内分别更新
        # LayoutLearning.Status 时互相覆盖。
        self.city_layout = self.resource_layout

    # ========================================================
    # Page recognition
    # ========================================================

    def is_allocation_overview(self):
        return self.appear(
            ALLOCATION_OVERVIEW_CHECK,
            offset=(3, 3),
            similarity=0.90,
        )

    def is_destination_page(self):
        return self.appear(
            ALLOCATION_DESTINATION_CHECK,
            offset=(3, 3),
            similarity=0.90,
        )

    def is_resource_page(self):
        return self.appear(
            ALLOCATION_RESOURCE_CHECK,
            offset=(3, 3),
            similarity=0.90,
        )

    def is_assignee_page(self):
        return self.appear(
            ALLOCATION_ASSIGNEE_CHECK,
            offset=(3, 3),
            similarity=0.90,
        )

    def _wait_page(self, checker, name, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            if checker():
                logger.info(f'Allocation page arrived: {name}')
                return True
            self.device.sleep(0.5)
        logger.warning(f'Allocation page timeout: {name}')
        return False

    # ========================================================
    # Source city and idle ship
    # ========================================================

    @staticmethod
    def _moved_area(area, dx, dy):
        x1, y1, x2, y2 = area
        return (
            int(x1 + dx),
            int(y1 + dy),
            int(x2 + dx),
            int(y2 + dy),
        )

    @staticmethod
    def _template_score(image, template):
        if image.shape != template.shape or image.size == 0:
            return 0.0
        score, _, _, _ = stable_feature_match(
            image, template, gray_weight=0.45
        )
        return score

    def click_idle_slot(self, city_match):
        """点击指定城市右侧两个运输位中的第一个空闲位。"""
        city_x, city_y, _ = city_match
        ref_x, ref_y = self.OVERVIEW_REFERENCE_CITY_CENTER
        dx = city_x - ref_x
        # 名称模板命中的是文字中心，不同城市图标/字号会让该中心偏离
        # 卡片所在行几十像素。先从当前截图判定四行中的哪一行，再锚定
        # 船位；不能把模板 y 的偏差直接传给船位点击坐标。
        screen_row, _ = self._city_match_screen_position(city_match)
        row_name_centers = (263, 395, 525, 654)
        snapped_city_y = row_name_centers[screen_row]
        dy = snapped_city_y - ref_y
        logger.attr(
            'AllocationSourceRow',
            f'city_y={city_y}, screen_row={screen_row + 1}, '
            f'anchor_y={snapped_city_y}',
        )

        candidates = []
        weak_buttons = []
        for index, ((check_area, click_area), template) in enumerate(
            zip(self.IDLE_SLOT_REFERENCE, self._idle_templates),
            start=1,
        ):
            moved_check = self._moved_area(check_area, dx, dy)
            moved_click = self._moved_area(click_area, dx, dy)
            # 列表抵达底端后，整组卡片相对固定 OCR 格位最多会上移约
            # 20px。城市名仍能被宽松 OCR 格读到，但按格位硬平移的
            # “空闲”小框会落到船图标上。只在竖直方向搜索一个很小的
            # 邻域，横向仍严格绑定当前城市，避免串到相邻城市。
            x1, y1, x2, y2 = moved_check
            search_area = (x1, max(0, y1 - 24), x2, min(720, y2 + 24))
            candidate = crop(self.device.image, search_area, copy=False)
            if (
                candidate.shape[0] >= template.shape[0]
                and candidate.shape[1] >= template.shape[1]
            ):
                score, _, _, _ = stable_feature_match(
                    candidate, template, gray_weight=0.45
                )
            else:
                score = 0.0
            logger.attr(f'AllocationIdleSlot{index}', f'{score:.3f}')
            label_score = 0.0
            label_area = moved_check
            label_template = getattr(self, '_idle_label_template', None)
            if (label_template is not None
                    and candidate.shape[0] >= label_template.shape[0]
                    and candidate.shape[1] >= label_template.shape[1]):
                _, label_score, _, label_at = cv2.minMaxLoc(cv2.matchTemplate(
                    cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY),
                    label_template, cv2.TM_CCOEFF_NORMED,
                ))
                label_x = search_area[0] + label_at[0]
                label_y = search_area[1] + label_at[1]
                label_area = (
                    max(0, label_x - 5), max(0, label_y - 2),
                    min(1280, label_x + label_template.shape[1] + 5),
                    min(720, label_y + label_template.shape[0] + 4),
                )
            logger.attr(f'AllocationIdleLabel{index}', f'{label_score:.3f}')
            candidates.append((index, search_area, moved_click, score, label_score))
            if score < 0.55 and label_score < 0.80:
                # OCR only the label.  The expanded search region also
                # contains the ship drawing and time/count text; on a truly
                # idle slot it was repeatedly read as "20" or "0".
                weak_buttons.append(Button(
                    area=label_area, color=(), button=label_area,
                    name=f'ALLOCATION_SLOT_STATUS_{index}',
                ))

        weak_texts = []
        if weak_buttons:
            weak_texts = Ocr(
                weak_buttons,
                lang='cnocr',
                name='ALLOCATION_SLOT_STATUS_BATCH',
            ).ocr(self.device.image)
            if not isinstance(weak_texts, list):
                weak_texts = [weak_texts]
        weak_iter = iter(weak_texts)
        observed_statuses = []

        for index, _, moved_click, score, label_score in candidates:
            # 同一页面不同船位的纸张纹理会降低彩色模板分数；模板弱的
            # 所有船位合并成一次文字 OCR，不再逐船位提交。
            if score < 0.55 and label_score < 0.80:
                raw = next(weak_iter, '')
                status_text = re.sub(r'\s+', '', str(raw))
                logger.attr(
                    f'AllocationIdleSlot{index}Text', status_text
                )
                observed_statuses.append(status_text)
                # 当前游戏的窄船位标签常只识别出首字“空”。船位已用
                # 当前城市行对齐，允许将它作为待验证候选；只有随后确实
                # 进入目的地页才会继续执行运输。
                if (status_text != '空' and '空闲' not in status_text
                        and '空閒' not in status_text):
                    continue
                if status_text == '空':
                    logger.info(f'Idle slot {index} OCR truncated; verify by page transition')

            logger.info(f'Click idle transport slot {index}')
            self.last_slot_index = index - 1
            self.device.click(Button(
                area=(),
                color=(),
                button=moved_click,
                name=f'ALLOCATION_IDLE_SLOT_{index}',
            ))
            return True

        # “点”只是空闲船位图标被 OCR 后的噪声，绝不能据此把整座城市
        # 标成码头未修复。两个船位都没找到时，再识别当前城市右侧的
        # 整行提示；只有明确出现“未修复/尚未修复”才写入该状态。
        status_area = (
            max(0, city_x + 55),
            max(0, city_y - 80),
            min(1280, city_x + 430),
            min(720, city_y - 20),
        )
        dock_status_text = Ocr(
            Button(
                area=status_area,
                color=(),
                button=status_area,
                name='ALLOCATION_DOCK_STATUS',
            ),
            lang='cnocr',
            name='ALLOCATION_DOCK_STATUS',
        ).ocr(self.device.image)
        dock_status_text = re.sub(r'\s+', '', str(dock_status_text))
        logger.attr('AllocationDockStatusText', dock_status_text)
        dock_unrepaired = (
            '未修复' in dock_status_text
            or '尚未修复' in dock_status_text
            or '未修復' in dock_status_text
            or '尚未修復' in dock_status_text
        )
        if dock_unrepaired:
            logger.warning('Source city dock is not repaired')
            self.last_error = 'dock_unrepaired'
        else:
            logger.warning('No idle transport slot in source city')
            self.last_error = 'no_idle_slot'
            self.save_abnormal_screenshot('allocation_idle_slot_missing')
        return False

    def open_transport(self, source):
        logger.hr(f'ALLOCATION SOURCE: {source}')

        if not self.goto_city_list():
            return False
        if not self.is_allocation_overview():
            logger.warning('Not in city allocation overview')
            return False

        match = self.locate_city(source)
        if match is None:
            logger.warning(f'Allocation source city not found: {source}')
            return False

        # find_city() 的最后一轮已经截过图，直接用这一帧判断空闲船位。
        if not self.click_idle_slot(match):
            return False

        arrived = self._wait_page(
            self.is_destination_page,
            'destination',
        )
        if not arrived:
            self.last_error = 'destination_page_timeout'
            self.save_abnormal_screenshot('allocation_destination_timeout')
        return arrived

    # ========================================================
    # Destination
    # ========================================================

    def choose_destination(self, destination):
        logger.hr(f'ALLOCATION DESTINATION: {destination}')

        if not self.is_destination_page():
            logger.warning('Not in destination selection page')
            return False

        match = self.locate_city(destination)
        if match is None:
            logger.warning(
                f'Allocation destination city not found: {destination}'
            )
            return False

        self.click_city_match(match)
        arrived = self._wait_page(self.is_resource_page, 'resource')
        if arrived:
            # 游戏会保留上一次打开物资列表时的滚动位置，刚进入页面并
            # 不等于位于顶部。由后续回顶流程确认后再赋值为 0。
            self._resource_view_start = None
            self._resource_scroll_offset = None
        return arrived

    # ========================================================
    # Resource selection
    # ========================================================

    def _resource_grid_image(self):
        x1, y1, x2, y2 = self.RESOURCE_GRID_AREA
        return self.device.image[y1:y2, x1:x2].copy()

    @staticmethod
    def _image_change(before, after):
        return float(cv2.absdiff(before, after).mean())

    @staticmethod
    def _estimate_scroll_shift(before, after, expected_range=None):
        """估算物资网格实际向上移动的像素，避免假定滑动距离。"""
        before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        height, width = before_gray.shape

        template_y1 = max(80, height - 145)
        template_y2 = height - 55
        template = before_gray[template_y1:template_y2, 15:width - 15]
        search = after_gray[:, 15:width - 15]
        if template.size == 0 or search.size == 0:
            return None, 0.0

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        shift = template_y1 - location[1]
        if shift < 0 or shift > 360 or score < 0.55:
            return None, float(score)
        # 物资格子每 110px 重复一次，相关匹配可能返回真实位移减去一
        # 个或多个行高。根据本次受控拖动的预期范围选择同余候选，既
        # 支持远距离大步移动，也不会把短拖的 67px 错改成 177px。
        if expected_range is not None:
            low, preferred, high = expected_range
            candidates = [
                shift + row_delta * JiangnanAllocation.RESOURCE_ROW_HEIGHT
                for row_delta in range(-3, 4)
                if 3 < shift + row_delta * JiangnanAllocation.RESOURCE_ROW_HEIGHT <= 360
            ]
            in_range = [
                value for value in candidates if low <= value <= high
            ]
            selected = min(
                in_range or candidates,
                key=lambda value: abs(value - preferred),
            )
            if selected != shift:
                logger.info(
                    f'Correct periodic resource scroll shift: '
                    f'{shift} -> {selected} (expected={low}-{high})'
                )
            shift = selected
        elif 3 < shift < 60:
            logger.info(
                f'Correct periodic resource scroll shift: '
                f'{shift} -> {shift + JiangnanAllocation.RESOURCE_ROW_HEIGHT}'
            )
            shift += JiangnanAllocation.RESOURCE_ROW_HEIGHT
        return float(shift), float(score)

    @classmethod
    def _resource_view_starts(cls, max_row):
        """返回三行可见、两行步进的所有首行。"""
        max_row = max(0, int(max_row))
        last_start = max(0, max_row - cls.RESOURCE_VIEW_ROWS + 1)
        starts = list(range(0, last_start + 1, cls.RESOURCE_VIEW_STRIDE))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _drag_resource_view(self, downward, rows=None):
        """按指定行数滑动物资列表，不做途中识别。"""
        rows = self.RESOURCE_VIEW_STRIDE if rows is None else int(rows)
        rows = max(1, min(self.RESOURCE_VIEW_STRIDE, rows))
        distance = int(self.RESOURCE_ROW_HEIGHT * rows)
        if downward:
            start, end = (850, 435), (850, 435 - distance)
        else:
            start, end = (850, 215), (850, 215 + distance)
        self._resource_drag_seq += 1
        self.jiangnan_drag(
            start,
            end,
            hold_duration=0.2,
            name=(
                f'ALLOCATION_RESOURCE_PAGE_'
                f'{"DOWN" if downward else "UP"}_{rows}ROW_'
                f'{self._resource_drag_seq:04d}'
            ),
        )
        self.device.sleep(0.6)
        # This helper intentionally performs no visual measurement.  Never let
        # callers reuse an absolute offset across such a blind gesture.
        self._resource_scroll_offset = None

    def _reset_resource_top_boundary(self, max_swipes=8):
        """快速回到物资列表顶部，以连续稳定画面确认边界。"""
        logger.hr('ALLOCATION RESOURCE RESET TOP')
        self.device.screenshot()
        stable_frames = 0
        for index in range(max_swipes):
            before = self._resource_grid_image()
            self._resource_drag_seq += 1
            self.device.swipe(
                (850, 160),
                (850, 560),
                duration=0.1,
                name=(
                    f'ALLOCATION_RESOURCE_TO_TOP_'
                    f'{self._resource_drag_seq:04d}'
                ),
            )
            # 等快速滑动和顶部回弹结束后再比较稳定画面。
            self.device.sleep(0.4)
            self.device.screenshot()
            after = self._resource_grid_image()
            change = self._image_change(before, after)
            logger.attr(
                'AllocationResourceTopChange',
                f'attempt={index + 1}, change={change:.2f}',
            )
            # 各城市的本地可生产物资不同，第一格不一定是蔬菜，
            # 也不能用任何固定货物当顶部锚点。快速上滑后画面
            # 已经不再变化，就是到达当前城市列表的物理顶边界。
            # 背景动画使白雪镇顶部的静止变化保持在 3~5 之间；
            # 连续两次小变化比单次严格像素阈值更能区分实际滚动。
            stable_frames = stable_frames + 1 if change <= 5.5 else 0
            if stable_frames >= 2:
                self._resource_view_start = 0
                self._resource_scroll_offset = 0.0
                logger.info(
                    f'Allocation resource top reached by stable boundary: '
                    f'{stable_frames} consecutive frames'
                )
                return True
        logger.warning('Unable to confirm allocation resource top')
        return False

    def _move_resource_to_view(self, target_start, max_row):
        """Move to a logical view, measuring every real scroll from pixels."""
        starts = self._resource_view_starts(max_row)
        target_start = min(
            starts, key=lambda value: abs(value - int(target_start))
        )
        if self._resource_scroll_offset is None:
            if not self._reset_resource_top_boundary():
                return False

        target_offset = float(target_start * self.RESOURCE_ROW_HEIGHT)
        for attempt in range(10):
            current = float(self._resource_scroll_offset or 0.0)
            error = target_offset - current
            if abs(error) <= 16.0:
                self._resource_view_start = target_start
                logger.attr(
                    'AllocationResourceView',
                    f'first_row={target_start}, offset={current:.1f}, '
                    f'error={error:.1f}, visual=True',
                )
                return True

            # Upward image translation is much harder to disambiguate in this
            # periodic grid.  For a target above the current position, return
            # to the visually confirmed physical top and approach it downward.
            if error < 0:
                if not self._reset_resource_top_boundary():
                    return False
                continue
            downward = True
            requested = min(220.0, max(28.0, abs(error)))
            if downward:
                start = (850, 445)
                end = (850, int(round(445 - requested)))
            else:
                start = (850, 205)
                end = (850, int(round(205 + requested)))

            self.device.screenshot()
            before = self._resource_grid_image()
            self._resource_drag_seq += 1
            self.jiangnan_drag(
                start, end, hold_duration=0.2,
                name=(f'ALLOCATION_RESOURCE_VISUAL_MOVE_'
                      f'{self._resource_drag_seq:04d}'),
            )
            self.device.sleep(0.65)
            self.device.screenshot()
            after = self._resource_grid_image()
            change = self._image_change(before, after)
            if change <= 1.0:
                logger.warning(
                    f'Allocation resource visual move stalled: error={error:.1f}'
                )
                return False

            expected = (
                max(4.0, requested * 0.35),
                requested,
                min(360.0, requested * 1.75 + 35.0),
            )
            shift, score = self._estimate_scroll_shift(
                before, after, expected_range=expected
            )
            if shift is None or shift <= 3.0:
                logger.warning(
                    f'Allocation resource visual move untracked: score={score:.3f}'
                )
                return False
            signed_shift = float(shift if downward else -shift)
            self._resource_scroll_offset = max(
                0.0, current + signed_shift
            )
            logger.attr(
                'AllocationResourceVisualOffset',
                f'attempt={attempt + 1}, requested={requested:.1f}, '
                f'observed={signed_shift:.1f}, '
                f'offset={self._resource_scroll_offset:.1f}, score={score:.3f}',
            )

        logger.warning(
            f'Allocation resource visual move did not converge: '
            f'target_row={target_start}'
        )
        return False

    def reset_resource_top(self, max_swipes=None):
        """快速回顶，并以边界画面不再变化确认顶部。"""
        return self._reset_resource_top_boundary(
            max_swipes=8 if max_swipes is None else max_swipes
        )

    def _scroll_resource_until_visible(self, logical_y, max_swipes=18):
        offset = 0.0
        for _ in range(max_swipes + 1):
            screen_y = logical_y - offset
            if 115 <= screen_y <= self.RESOURCE_SAFE_BOTTOM:
                return screen_y

            self.device.screenshot()
            before = self._resource_grid_image()
            # 距离目标较远时使用两行左右的受控拖动，接近目标后自动
            # 缩成中/短拖。长按抑制惯性，且最大手势仍保留一屏重叠，
            # 方便用图像相关结果追踪实际位移。
            remaining = screen_y - self.RESOURCE_SAFE_BOTTOM
            if remaining > 520:
                drag_start, drag_end = (850, 445), (850, 215)
                expected_range = (150, 230, 315)
                distance_name = 'LONG'
            elif remaining > 230:
                drag_start, drag_end = (850, 435), (850, 270)
                expected_range = (95, 165, 240)
                distance_name = 'MEDIUM'
            else:
                drag_start, drag_end = (850, 420), (850, 330)
                expected_range = (40, 90, 155)
                distance_name = 'SHORT'
            self._resource_drag_seq += 1
            self.jiangnan_drag(
                drag_start,
                drag_end,
                hold_duration=0.2,
                name=(
                    f'ALLOCATION_RESOURCE_DOWN_{distance_name}_'
                    f'{self._resource_drag_seq:04d}'
                ),
            )
            self.device.sleep(0.8)
            self.device.screenshot()
            if not self.is_resource_page():
                logger.warning('Left resource page while scrolling list')
                return None
            after = self._resource_grid_image()
            if self._image_change(before, after) < 0.8:
                logger.warning('Resource list bottom reached before target')
                return None

            shift, score = self._estimate_scroll_shift(
                before, after, expected_range=expected_range
            )
            logger.attr('AllocationScrollMatch', f'{score:.3f}')
            if shift is None or shift <= 3:
                logger.warning('Unable to track allocation resource scroll')
                return None
            offset += shift
            logger.attr('AllocationResourceOffset', f'{offset:.1f}')

            # 目标可能恰好在最后一次允许的滑动后进入安全区域。旧逻辑
            # 要等下一轮循环才返回，因此 max_swipes 的最后一滑即使已经
            # 到达“兵器”等靠后物资，也会被误判为查找失败。
            screen_y = logical_y - offset
            if 115 <= screen_y <= self.RESOURCE_SAFE_BOTTOM:
                return screen_y

        return None

    @staticmethod
    def normalize_resource_name(resource):
        resource = str(resource).strip()
        return WAREHOUSE_OCR_ALIASES.get(resource, resource)

    @staticmethod
    def _read_rgb(path):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        except (OSError, ValueError):
            return None
        if image is None:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 仓库模板可能是包含多枚重复图标的横向图标带，必须保留原始
        # 尺寸。这里只处理透明通道，不再强行缩成运输格大小。
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        paper = np.array([181, 178, 166], dtype=np.float32)
        return (rgb * alpha + paper * (1.0 - alpha)).astype(np.uint8)

    @staticmethod
    def _write_rgb(path, image):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = cv2.imencode(
            '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )[1]
        encoded.tofile(str(path))

    @staticmethod
    def _resource_icon_crop(image, center, search=False):
        cx, cy = (int(center[0]), int(center[1]))
        # 图标位于方格中心略偏上，底部数量文字不属于图标特征。
        left, right = ((38, 38) if search else (28, 28))
        top, bottom = ((38, 28) if search else (32, 20))
        y1, y2 = max(0, cy - top), min(image.shape[0], cy + bottom)
        x1, x2 = max(0, cx - left), min(image.shape[1], cx + right)
        return image[y1:y2, x1:x2].copy()

    def _load_resource_icon_templates(self):
        resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        loaded = getattr(self, '_resource_icon_templates', {})
        versions = getattr(self, '_resource_icon_file_versions', {})
        features = getattr(self, '_resource_icon_features', {})
        bands = getattr(self, '_resource_icon_band_cache', {})
        current_versions = {}
        for resource in resources:
            path = self.RESOURCE_ICON_DIR / f'{resource}.png'
            try:
                stat = path.stat()
                version = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino)
            except OSError:
                version = None
            current_versions[resource] = version
            if resource in versions and versions[resource] == version:
                continue
            image = self._read_rgb(path)
            if image is not None and image.size:
                loaded[resource] = image
            else:
                loaded.pop(resource, None)
            features.pop(resource, None)
            bands.pop(resource, None)
        for resource in set(loaded) - set(resources):
            loaded.pop(resource, None)
            features.pop(resource, None)
            bands.pop(resource, None)
        self._resource_icon_file_versions = current_versions
        self._resource_icon_templates = loaded
        self._resource_icon_features = features
        self._resource_icon_band_cache = bands
        return loaded

    def _save_resource_icon(self, resource, image):
        if image is None or image.size == 0:
            return False
        path = self.RESOURCE_ICON_DIR / f'{resource}.png'
        self._write_rgb(path, image)
        self._resource_icon_templates[resource] = image
        self._resource_icon_features.pop(resource, None)
        getattr(self, '_resource_icon_band_cache', {}).pop(resource, None)
        getattr(self, '_resource_icon_file_versions', {}).pop(resource, None)
        return True

    def find_resource_cells(self):
        """识别当前画面所有完整物资格子，不假定目标所在行列。"""
        # Detect square cells on the complete frame.  Only their stable
        # four-column X geometry is used as a structural filter; Y is never
        # clipped to an assumed scroll landing range.
        image = self.device.image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 100)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cells = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (75 <= width <= 115 and 65 <= height <= 115):
                continue
            center = (left + width // 2, top + height // 2)
            column = min(
                range(4),
                key=lambda value: abs(center[0] - self.RESOURCE_GRID_X[value]),
            )
            if abs(center[0] - self.RESOURCE_GRID_X[column]) > 38:
                continue
            if any(
                abs(center[0] - old[0]) <= 7
                and abs(center[1] - old[1]) <= 7
                for old in cells
            ):
                continue
            cells.append(center)
        cells.sort(key=lambda point: (point[1], point[0]))
        logger.attr('AllocationVisibleResourceCells', cells)
        return cells

    def _yellow_resource_cells(self, cells):
        """一次颜色分割找出当前屏的黄边本地物资格。"""
        hsv = cv2.cvtColor(self.device.image, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(hsv, (15, 70, 120), (40, 255, 255))
        result = []
        for cell in cells:
            x, y = map(int, cell)
            x1, x2 = max(0, x - 47), min(yellow.shape[1], x + 47)
            y1, y2 = max(0, y - 47), min(yellow.shape[0], y + 47)
            roi = yellow[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            ring = np.zeros(roi.shape, dtype=np.uint8)
            width = min(7, max(1, min(roi.shape) // 5))
            ring[:width, :] = 1
            ring[-width:, :] = 1
            ring[:, :width] = 1
            ring[:, -width:] = 1
            pixels = int(np.count_nonzero((roi > 0) & (ring > 0)))
            logger.attr(
                f'AllocationYellowBorder_{x}_{y}', pixels
            )
            # 1280x720 实机黄框的环带约 660 像素，白框
            # 为 0。阈值保留足够的拖动抗锯齿余量。
            if pixels >= 220:
                result.append(cell)
        return result

    def _resource_icon_score(self, cell, template):
        search = self._resource_icon_crop(
            self.device.image, cell, search=True
        )
        search = self._normalize_visible_resource_icon(search)
        if search is None or template is None or template.size == 0:
            return 0.0
        return self._compare_resource_icon_features(
            self._resource_icon_features_for(search),
            self._resource_icon_features_for(template),
        )

    def _resource_icon_features_for(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 35, 0), (179, 255, 255)),
            cv2.inRange(gray, 0, 130),
        )
        histogram = cv2.calcHist(
            [hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]
        )
        cv2.normalize(histogram, histogram)
        _, descriptors = self._resource_sift.detectAndCompute(gray, None)
        return descriptors, histogram

    @staticmethod
    def _compare_resource_icon_features(left, right):
        left_descriptors, left_histogram = left
        right_descriptors, right_histogram = right
        sift_score = 0.0
        if left_descriptors is not None and right_descriptors is not None:
            matcher = cv2.BFMatcher()

            def directional(query, train):
                if len(query) < 1 or len(train) < 2:
                    return 0.0
                matches = matcher.knnMatch(query, train, k=2)
                good = sum(
                    len(pair) == 2
                    and pair[0].distance < 0.75 * pair[1].distance
                    for pair in matches
                )
                return good / max(1, len(query))

            # 非零库存时仓库横向重复绘制同一图标。运输格只有一枚，
            # 因此双向匹配并取较高值，避免重复模板被分母稀释。
            sift_score = max(
                directional(left_descriptors, right_descriptors),
                directional(right_descriptors, left_descriptors),
            )
        color_score = max(0.0, float(cv2.compareHist(
            left_histogram, right_histogram, cv2.HISTCMP_CORREL
        )))
        # SIFT 区分纸/布等近色轮廓，HSV 则解决木炭等纹理较少的图标。
        return float(sift_score + 0.25 * color_score)

    def _template_icon_features(self, name, template):
        features = self._resource_icon_features.get(name)
        if features is None:
            # 仓库库存较多时，同一货物会横向绘制多枚图标，保存下来的
            # 模板因此是宽横条；运输选择页则只有一枚。直接把整条模板
            # 和单格比较会让棉花、竹子等分数异常偏低。沿横条生成若干
            # 单图标宽度的窗口并归一化，最终取与当前格最相似的一枚。
            variants = []
            height, width = template.shape[:2]
            if width <= 90:
                normalized = self._normalize_visible_resource_icon(template)
                if normalized is not None:
                    variants.append(normalized)
            else:
                window_width = min(68, max(52, height + 12))
                last_start = max(0, width - window_width)
                starts = list(range(0, last_start + 1, 10))
                if not starts or starts[-1] != last_start:
                    starts.append(last_start)
                for start in starts:
                    window = template[:, start:start + window_width]
                    normalized = self._normalize_visible_resource_icon(window)
                    if normalized is not None:
                        variants.append(normalized)
            if not variants:
                variants = [template]
            features = [
                self._resource_icon_features_for(variant)
                for variant in variants
            ]
            self._resource_icon_features[name] = features
        return features

    def _compare_cell_to_template(self, cell_features, name, template):
        variants = self._template_icon_features(name, template)
        return max(
            self._compare_resource_icon_features(cell_features, candidate)
            for candidate in variants
        )

    def _cell_icon_features(self, cell):
        frame = id(self.device.image)
        if frame != self._resource_cell_feature_frame:
            self._resource_cell_feature_frame = frame
            self._resource_cell_frame_image = self.device.image
            self._resource_cell_feature_cache = {}
            self._resource_cell_content_cache = {}
            self._resource_cell_match_cache = {}
        key = (int(cell[0]), int(cell[1]))
        if key in self._resource_cell_feature_cache:
            return self._resource_cell_feature_cache[key]
        search = self._resource_icon_crop(
            self.device.image, cell, search=True
        )
        search = self._normalize_visible_resource_icon(search)
        if search is None:
            self._resource_cell_feature_cache[key] = None
            return None
        features = self._resource_icon_features_for(search)
        self._resource_cell_feature_cache[key] = features
        return features

    @staticmethod
    def _normalize_visible_resource_icon(image):
        """从方格中分离图标，排除边框、缺货角标和下方库存数字。"""
        if image is None or image.size == 0 or min(image.shape[:2]) < 20:
            return None
        # 方格边框和左上角叹号都在外圈；库存数字位于最下方。
        # 数量文字会压到图标下沿附近；学习身份时宁可少取几像素下沿，
        # 也不能让不同库存数字成为图标特征。
        roi = image[3:max(4, image.shape[0] - 18),
                    7:max(8, image.shape[1] - 7)].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 34, 0), (179, 255, 255)),
            cv2.inRange(gray, 0, 120),
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        kept = np.zeros_like(mask)
        height, width = mask.shape
        for label in range(1, count):
            left, top, item_width, item_height, area = stats[label]
            cx, cy = centroids[label]
            if area < 7:
                continue
            # 排除贴边的黄色方框、红色缺货角标和残留数量笔画。
            if cx < width * 0.20 and cy < height * 0.28:
                continue
            if cy > height * 0.88:
                continue
            if item_width >= item_height * 2.6 and cy > height * 0.62:
                continue
            kept[labels == label] = 255
        points = cv2.findNonZero(kept)
        if points is None or len(points) < 25:
            return None
        left, top, item_width, item_height = cv2.boundingRect(points)
        left, top = max(0, left - 2), max(0, top - 2)
        right = min(width, left + item_width + 4)
        bottom = min(height, top + item_height + 4)
        obj = roi[top:bottom, left:right]
        obj_mask = kept[top:bottom, left:right]
        scale = min(48 / obj.shape[1], 44 / obj.shape[0])
        size = (
            max(1, int(round(obj.shape[1] * scale))),
            max(1, int(round(obj.shape[0] * scale))),
        )
        obj = cv2.resize(obj, size, interpolation=cv2.INTER_AREA)
        obj_mask = cv2.resize(obj_mask, size, interpolation=cv2.INTER_NEAREST)
        paper = np.array([181, 178, 166], dtype=np.uint8)
        canvas = np.full((52, 56, 3), paper, dtype=np.uint8)
        x, y = (56 - size[0]) // 2, (52 - size[1]) // 2
        target = canvas[y:y + size[1], x:x + size[0]]
        target[obj_mask > 0] = obj[obj_mask > 0]
        return canvas

    def _find_resource_icon(
        self, resource, cells=None, validate=True, threshold=None
    ):
        """用目标货物自己的仓库模板确认候选格。"""
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        template = self._resource_icon_templates.get(resource)
        if template is None:
            logger.warning(f'Resource icon template is missing: {resource}')
            return None
        if cells is None:
            cells = self.find_resource_cells()
        cell_features = {
            cell: self._cell_icon_features(cell) for cell in cells
        }
        scored = sorted((
            (
                self._compare_cell_to_template(
                    features, resource, template
                )
                if features is not None else 0.0,
                cell,
            )
            for cell, features in cell_features.items()
        ), reverse=True)
        if not scored:
            return None
        best_score, best_cell = scored[0]
        second_cell_score = scored[1][0] if len(scored) > 1 else 0.0

        logger.attr(
            f'AllocationIcon_{resource}',
            f'target={best_score:.3f}, second_cell={second_cell_score:.3f}',
        )
        if validate:
            if threshold is None:
                threshold = self.RESOURCE_ICON_THRESHOLD
            if best_score < threshold:
                return None
            if best_score - second_cell_score < self.RESOURCE_ICON_CELL_MARGIN:
                logger.warning(f'Ambiguous resource icon cells: {resource}')
                return None
        return best_cell, best_score

    def _resource_cell_layout(self, cells):
        """将识别到的方框整理成画面行号和列号。"""
        rows = []
        for cell in sorted(cells, key=lambda point: (point[1], point[0])):
            row = next(
                (items for items in rows if abs(items[0][1] - cell[1]) <= 18),
                None,
            )
            if row is None:
                row = []
                rows.append(row)
            row.append(cell)
        layout = []
        for row_index, row in enumerate(rows):
            for cell in sorted(row):
                column = min(
                    range(4),
                    key=lambda value: abs(cell[0] - self.RESOURCE_GRID_X[value]),
                )
                if abs(cell[0] - self.RESOURCE_GRID_X[column]) <= 35:
                    layout.append((row_index, column, cell))
        return layout

    def _identify_resource_cells(self, cells):
        """一次提取所有格位特征，再同时识别当前屏的货物图标。"""
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        result = {}
        for cell in cells:
            features = self._cell_icon_features(cell)
            if features is None:
                continue
            scored = sorted((
                self._compare_cell_to_template(
                    features, name, template
                ),
                name,
            ) for name, template in self._resource_icon_templates.items())
            best_score, best_name = scored[-1]
            second_score = scored[-2][0] if len(scored) > 1 else 0.0
            logger.attr(
                f'AllocationCellIcon_{cell[0]}_{cell[1]}',
                f'{best_name}:{best_score:.3f}, second={second_score:.3f}',
            )
            if (
                best_score >= self.RESOURCE_ICON_FIXED_THRESHOLD
                and best_score - second_score >= 0.02
            ):
                result[cell] = best_name
        return result

    def _verify_resource_cell_identity(
        self, cell, resource, threshold=0.25
    ):
        """用黄框格中的实际图标确认身份，返回目标内容分数。

        ``_compare_cell_to_template`` 会把边框、数字和纸张背景一并纳入
        比较，实测不同货物会同时饱和到约 1.2，不能用于最终点击确认。
        这里改用辅助信息学习阶段已经验证过的内容分类器，并在全部已
        知货物中排名；不能依赖可能已经过期的本城辅助顺序来缩小范围。
        """
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        candidates = list(self._resource_icon_templates)
        scored = []
        for name in candidates:
            template = self._resource_icon_templates.get(name)
            if template is None:
                continue
            scored.append((
                self._resource_icon_content_score(cell, name, template),
                name,
            ))
        scored.sort(reverse=True)
        if not scored:
            return None
        best_score, best_name = scored[0]
        target_score = next((
            score for score, name in scored if name == resource
        ), 0.0)
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        logger.attr(
            f'AllocationCellContent_{cell[0]}_{cell[1]}',
            f'expected={resource}:{target_score:.3f}, '
            f'best={best_name}:{best_score:.3f}, second={second_score:.3f}',
        )
        if (
            best_name == resource
            and target_score >= threshold
            and target_score - second_score >= 0.025
        ):
            # Check a clear content winner before signature fallbacks. In
            # particular, a well-matched paper scroll must not be downgraded
            # to tentative merely because it also has the paper signature.
            self._resource_identity_tentative = False
            return float(target_score)
        if (resource == '歙砚'
                and target_score >= 0.35
                and self._dark_inkstone_signature(cell)):
            # 仓库图标带同时含数枚砚台，通用分类器甚至会把真实砚台
            # 排在茴香豆后面。运输格里的砚台近黑且低饱和，整页实图
            # 只有这格满足特征。仍标为待目的地容量复核。
            self._resource_identity_tentative = True
            logger.info('Allocation dark inkstone confirmed by live icon')
            return float(target_score)
        if (resource == '纸'
                and target_score >= 0.40
                and self._paper_scroll_signature(cell)):
            # 白色纸卷与棉花的通用分数常只差 0.003；纸卷的白色
            # 长条覆盖更大且颜色更淡。仍需目的地容量确认。
            self._resource_identity_tentative = True
            logger.info('Allocation paper scroll confirmed by live icon')
            return float(target_score)
        if (
            best_name == resource
            and (
                target_score >= 0.65
                or (
                    target_score >= 0.45
                    and target_score - second_score >= 0.015
                )
            )
        ):
            # 仓库图标带与运输格的背景不同，菜肴实图只得 0.485，
            # 次名 0.464。把这种弱但排名第一的结果留待目的地容量复核。
            self._resource_identity_tentative = True
            logger.info(
                f'Allocation icon close tie with expected resource: '
                f'{resource}={target_score:.3f}, second={second_score:.3f}; '
                'require capacity confirmation'
            )
            return float(target_score)
        # Warehouse icon strips sometimes contain only part of one icon.
        # In the captured 羊毛 cell the content matcher scored 年历 0.421 and
        # 羊毛 0.382, while the independent normalized-icon matcher clearly
        # preferred 羊毛 (0.418 vs 0.270). Accept only a close content tie
        # supported by that second matcher; capacity must confirm it later.
        if (
            best_name != resource
            and target_score >= max(threshold, 0.35)
            and best_score - target_score <= 0.05
        ):
            features = self._cell_icon_features(cell)
            if features is not None:
                target_features = self._compare_cell_to_template(
                    features, resource, self._resource_icon_templates[resource]
                )
                best_features = self._compare_cell_to_template(
                    features, best_name, self._resource_icon_templates[best_name]
                )
                if (target_features >= 0.35
                        and target_features - best_features >= 0.10):
                    self._resource_identity_tentative = True
                    logger.info(
                        f'Allocation icon close tie resolved by independent '
                        f'features: {resource}={target_features:.3f}, '
                        f'{best_name}={best_features:.3f}; require capacity'
                    )
                    return float(target_score)
        logger.warning(
            f'Allocation cell identity mismatch: expected={resource}, '
            f'identified={best_name}, target={target_score:.3f}, '
            f'best={best_score:.3f}, cell={cell}'
        )
        return None

    def _dark_inkstone_signature(self, cell):
        """歙砚运输图标的局部深色特征，排除歙石和木炭。"""
        cx, cy = int(cell[0]), int(cell[1])
        image = self.device.image
        roi = image[
            max(0, cy - 25):min(image.shape[0], cy + 15),
            max(0, cx - 26):min(image.shape[1], cx + 26),
        ]
        if roi.shape[0] < 35 or roi.shape[1] < 45:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        dark_fraction = float((gray < 110).mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        logger.attr('AllocationDarkInkstone',
                    f'dark={dark_fraction:.3f}, sat={mean_saturation:.1f}')
        return dark_fraction >= 0.82 and mean_saturation <= 55

    def _paper_scroll_signature(self, cell):
        """区分白色纸卷与相邻的棉花图标。"""
        cx, cy = int(cell[0]), int(cell[1])
        image = self.device.image
        roi = image[
            max(0, cy - 29):min(image.shape[0], cy + 16),
            max(0, cx - 28):min(image.shape[1], cx + 28),
        ]
        if roi.shape[0] < 40 or roi.shape[1] < 50:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        white_fraction = float((gray > 165).mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        logger.attr('AllocationPaperScroll',
                    f'white={white_fraction:.3f}, sat={mean_saturation:.1f}')
        return white_fraction >= 0.49 and mean_saturation <= 40

    def _resource_cell_content(self, cell):
        """提取运输格中的单枚货物图标及其前景，排除边框、角标和数字。"""
        frame = id(self.device.image)
        if frame != self._resource_cell_feature_frame:
            self._resource_cell_feature_frame = frame
            self._resource_cell_frame_image = self.device.image
            self._resource_cell_feature_cache = {}
            self._resource_cell_content_cache = {}
            self._resource_cell_match_cache = {}
        key = (int(cell[0]), int(cell[1]))
        if key in self._resource_cell_content_cache:
            return self._resource_cell_content_cache[key]

        image = self._resource_icon_crop(
            self.device.image, cell, search=True
        )
        if image is None or image.size == 0:
            self._resource_cell_content_cache[key] = None
            return None
        roi = image[3:max(4, image.shape[0] - 8),
                    7:max(8, image.shape[1] - 7)].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 34, 0), (179, 255, 255)),
            cv2.inRange(gray, 0, 120),
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        kept = np.zeros_like(mask)
        height, width = mask.shape
        for label in range(1, count):
            left, top, item_width, item_height, area = stats[label]
            cx, cy = centroids[label]
            if area < 7:
                continue
            if cx < width * 0.20 and cy < height * 0.28:
                continue
            if cy > height * 0.88:
                continue
            if item_width >= item_height * 2.6 and cy > height * 0.62:
                continue
            kept[labels == label] = 255
        points = cv2.findNonZero(kept)
        if points is None:
            self._resource_cell_content_cache[key] = None
            return None
        left, top, item_width, item_height = cv2.boundingRect(points)
        content_image = roi[
            top:top + item_height, left:left + item_width
        ]
        _, descriptors = self._resource_sift.detectAndCompute(
            cv2.cvtColor(content_image, cv2.COLOR_RGB2GRAY), None
        )
        content = (
            content_image,
            kept[top:top + item_height, left:left + item_width],
            descriptors,
        )
        self._resource_cell_content_cache[key] = content
        return content

    def _resource_icon_band_features(self, name, template):
        """预处理已裁掉名称的仓库图标带，保留图形完整高度。"""
        cached = self._resource_icon_band_cache.get(name)
        if cached is not None:
            return cached
        # Warehouse captures already exclude the row name. Cutting another
        # 8px removes the top of short icons (notably the fishing net) and
        # prevents a full-sized transport icon from matching the strip.
        band = template.copy()
        gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
        saturated = cv2.inRange(hsv, (0, 60, 0), (179, 255, 255))
        histogram = cv2.calcHist(
            [hsv], [0, 1], saturated, [24, 6], [0, 180, 0, 256]
        )
        cv2.normalize(histogram, histogram)
        _, descriptors = self._resource_sift.detectAndCompute(gray, None)
        cached = (
            gray,
            cv2.Canny(gray, 40, 120),
            histogram,
            int(np.count_nonzero(saturated)),
            descriptors,
        )
        self._resource_icon_band_cache[name] = cached
        return cached

    def _resource_icon_content_score(self, cell, name, template):
        """直接用当前格的实际图标在仓库图标带中搜索，不依赖排列顺序。"""
        content = self._resource_cell_content(cell)
        if content is None:
            return 0.0
        image, foreground, descriptors = content
        # The target cell is unchanged while compared against every cargo.
        # Generate its histogram and scaled edges once, independent of which
        # warehouse template is being tested. Retain the content reference so
        # cache entries cannot survive replacement/reuse of an object id.
        cache = getattr(self, '_resource_cell_match_cache', None)
        if cache is None:
            cache = self._resource_cell_match_cache = {}
        key = (int(cell[0]), int(cell[1]))
        prepared = cache.get(key)
        if prepared is None or prepared[0] is not content:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturated = cv2.inRange(hsv, (0, 60, 0), (179, 255, 255))
            histogram = cv2.calcHist([hsv], [0, 1], saturated, [24, 6], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram)
            color_pixels = int(np.count_nonzero(saturated))
            scales = []
            for scale in np.arange(0.72, 1.31, 0.04):
                width = int(max(8, round(image.shape[1] * scale)))
                height = int(max(8, round(image.shape[0] * scale)))
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                scales.append((width, height, gray, cv2.Canny(gray, 40, 120)))
            prepared = (content, hsv, histogram, color_pixels, scales)
            cache[key] = prepared
        _, hsv, histogram, color_pixels, scales = prepared
        if name in ('布鞋', '木炭', '油墨', '徽墨'):
            # Green cloth shoes and dark ink/charcoal share strong grayscale
            # keypoints. Reject contradictory pigment before shape matching;
            # colour alone never supplies a positive identity score.
            green = float(((hsv[:, :, 0] > 50) & (hsv[:, :, 0] < 90)
                           & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 35)).mean())
            if (name == '布鞋' and green < 0.06
                    or name != '布鞋' and green >= 0.12):
                return 0.0
        (
            band_gray, band_edges, band_histogram, band_color_pixels,
            band_descriptors,
        ) = (
            self._resource_icon_band_features(name, template)
        )
        color_score = 0.0
        if color_pixels > 20 and band_color_pixels > 20:
            color_score = max(0.0, float(cv2.compareHist(
                histogram, band_histogram, cv2.HISTCMP_CORREL
            )))

        sift_score = 0.0
        if (
            descriptors is not None
            and band_descriptors is not None
            and len(descriptors) >= 1
            and len(band_descriptors) >= 2
        ):
            pairs = cv2.BFMatcher().knnMatch(
                descriptors, band_descriptors, k=2
            )
            good = sum(
                len(pair) == 2
                and pair[0].distance < 0.75 * pair[1].distance
                for pair in pairs
            )
            sift_score = good / max(1, len(descriptors))

        best_edge = 0.0
        best_gray = 0.0
        for width, height, resized_gray, edges in scales:
            if width > band_edges.shape[1] or height > band_edges.shape[0]:
                continue
            scores = cv2.matchTemplate(
                band_edges, edges, cv2.TM_CCOEFF_NORMED
            )
            score = float(np.max(np.nan_to_num(scores)))
            best_edge = max(best_edge, score)
            gray_scores = cv2.matchTemplate(
                band_gray, resized_gray, cv2.TM_CCOEFF_NORMED
            )
            best_gray = max(
                best_gray, float(np.max(np.nan_to_num(gray_scores)))
            )
        score = (
            0.65 * best_edge + 0.35 * best_gray
            + 0.25 * color_score + 0.80 * sift_score
        )
        if name in ('渔网', '鱼干'):
            # Fishing nets are pale yellow; dried fish are orange/red. Their
            # repeated outlines are similar, so a shape-only near tie is unsafe.
            pigment = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 90)
            hues = hsv[:, :, 0][pigment]
            if len(hues) >= 30:
                yellow = float(((hues >= 20) & (hues <= 40)).mean())
                red = float(((hues <= 15) | (hues >= 170)).mean())
                expected, opposite = (yellow, red) if name == '渔网' else (red, yellow)
                if opposite >= 0.50 and expected <= 0.15:
                    return 0.0
        return score

    def _resource_content_scores(self, cells):
        """返回每个格子对全部货物模板的实际图像匹配分数。"""
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        return {
            cell: {
                name: self._resource_icon_content_score(cell, name, template)
                for name, template in self._resource_icon_templates.items()
            }
            for cell in cells
        }

    def _find_resource_icon_content(
        self, resource, cells, validate=True, threshold=0.25
    ):
        """在黄框候选中按实际图标内容查找目标货物。"""
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        template = self._resource_icon_templates.get(resource)
        if template is None:
            logger.warning(f'Resource icon template is missing: {resource}')
            return None
        if resource == '歙砚':
            distinctive = [
                cell for cell in cells
                if self._dark_inkstone_signature(cell)
            ]
            if not distinctive:
                return None
            cells = distinctive
        if resource == '纸':
            distinctive = [
                cell for cell in cells
                if self._paper_scroll_signature(cell)
            ]
            if not distinctive:
                return None
            cells = distinctive
        scored = sorted((
            (
                self._resource_icon_content_score(cell, resource, template),
                cell,
            )
            for cell in cells
        ), reverse=True)
        if not scored:
            return None
        best_score, best_cell = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        logger.attr(
            f'AllocationContentIcon_{resource}',
            f'target={best_score:.3f}, second_cell={second_score:.3f}',
        )
        if validate and (
            best_score < threshold
            or (
                len(scored) > 1
                and best_score - second_score
                < self.RESOURCE_ICON_CELL_MARGIN
            )
        ):
            logger.warning(f'Ambiguous resource content cells: {resource}')
            return None
        return best_cell, float(best_score)

    def scan_local_resource_layout(self, source, max_pages=10):
        """两行步进遍历黄框，并逐格识别其中实际显示的货物图标。"""
        if not self.is_resource_page():
            logger.warning('Allocation resource page is not open')
            return False
        if not self._reset_resource_top_boundary():
            return False
        page = allocation_layout_page(source)
        view_start = 0
        yellow_total = 0
        visible_cell_total = 0
        yellow_positions = set()
        observations = {}

        for page_index in range(max_pages):
            self.device.screenshot()
            if not self.is_resource_page():
                return False
            cells = self.find_resource_cells()
            visible = self._resource_cell_layout(cells)
            visible_cell_total += len(visible)
            yellow_cells = set(self._yellow_resource_cells(cells))
            # 同一页的黄框共享一张快照，避免为每个格子复制整帧。
            page_image = self.device.image.copy()
            yellow_total += len(yellow_cells)
            for screen_row, column, cell in visible:
                if cell not in yellow_cells:
                    continue
                position = (
                    int(view_start + screen_row), int(column)
                )
                yellow_positions.add(position)
                observations.setdefault(position, []).append(
                    (page_image, cell)
                )
            yellow_rows = {
                row for row, _, cell in visible if cell in yellow_cells
            }
            white_rows = {
                row for row, _, cell in visible if cell not in yellow_cells
            }
            logger.attr(
                'AllocationLayoutRecognition',
                f'city={source}, first_row={view_start}, '
                f'yellow={len(yellow_cells)}, '
                f'yellow_positions={len(yellow_positions)}',
            )
            if not yellow_rows or (
                white_rows and max(white_rows) >= max(yellow_rows)
            ):
                break
            self._drag_resource_view(downward=True)
            view_start += self.RESOURCE_VIEW_STRIDE
            self._resource_view_start = view_start

        ordered_positions = sorted(yellow_positions)
        if not ordered_positions and visible_cell_total:
            self.resource_layout.replace_status(page, '无生产')
            logger.info(
                f'Allocation layout updated: city={source}, status=无生产'
            )
            return True
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()
        resources = list(self._resource_icon_templates)
        if not ordered_positions or not resources:
            logger.warning(
                f'No local allocation resources recognized: {source}, '
                f'yellow_cells={yellow_total}'
            )
            return False

        # 第一阶段已经完整确定黄框范围。第二阶段才把保存下来的黄框格
        # 送入图标识别；白框从未进入内容分类器。
        logger.info(
            f'Allocation yellow range confirmed: city={source}, '
            f'positions={len(ordered_positions)}; start icon recognition'
        )
        scored_observations = {}
        live_image = self.device.image
        try:
            for position in ordered_positions:
                scored_observations[position] = []
                for image, cell in observations.get(position, []):
                    self.device.image = image
                    scores = self._resource_content_scores([cell])
                    scored_observations[position].append(
                        scores.get(cell, {})
                    )
        finally:
            self.device.image = live_image

        matrix = np.zeros((len(ordered_positions), len(resources)), dtype=float)
        for row, position in enumerate(ordered_positions):
            samples = scored_observations.get(position, [])
            for column, resource in enumerate(resources):
                values = [
                    float(sample.get(resource, 0.0)) for sample in samples
                ]
                matrix[row, column] = (
                    float(np.median(values)) if values else 0.0
                )

        # 每种货物在列表中只出现一次。全局唯一匹配仅用于消除形状相近
        # 图标的局部歧义，不引入仓库顺序、城市产地或预期货物列表。
        rows, columns = linear_sum_assignment(-matrix)
        recognized = {}
        weak = []
        for row, column in zip(rows, columns):
            position = ordered_positions[int(row)]
            resource = resources[int(column)]
            score = float(matrix[int(row), int(column)])
            ranked = np.sort(matrix[int(row)])[::-1]
            second = float(ranked[1]) if len(ranked) > 1 else 0.0
            logger.attr(
                f'AllocationRecognized_{position[0]}_{position[1]}',
                f'{resource}:{score:.3f}, local_second={second:.3f}',
            )
            if score < 0.25:
                weak.append((position, resource, round(score, 3)))
            recognized[resource] = position
        if weak or len(recognized) != len(ordered_positions):
            logger.warning(
                f'Allocation icon recognition incomplete: city={source}, '
                f'weak={weak}, recognized={len(recognized)}/'
                f'{len(ordered_positions)}'
            )
            return False
        self.resource_layout.replace_layout(page, recognized)
        logger.info(
            f'Allocation layout updated: city={source}, '
            f'resources={len(recognized)}'
        )
        return True

    def _infer_visible_global_row(
        self, layout, configured_layout, allowed_first_rows=None
    ):
        """一次扫描全部可见图标，按辅助信息推断画面首个完整行。

        对每个候选首行，只比较该位置在辅助信息中应出现的图标。这样每
        个可见格只提取一次 SIFT/颜色特征，也不用把每个格子和全部货物
        做笛卡尔积比较。返回 None 表示证据不足，调用方必须走安全兜底。
        """
        if not layout or not configured_layout:
            return None
        if not self._resource_icon_templates:
            self._load_resource_icon_templates()

        reverse_layout = {
            tuple(position): name
            for name, position in configured_layout.items()
        }
        max_row = max(
            (position[0] for position in configured_layout.values()),
            default=-1,
        )
        if max_row < 0:
            return None

        sampled = list(layout)
        sample_count = len(sampled)
        cell_features = {
            cell: self._cell_icon_features(cell)
            for _, _, cell in sampled
        }
        logger.attr(
            'AllocationCalibrationSamples',
            [(row, column, cell) for row, column, cell in sampled],
        )

        if allowed_first_rows is not None:
            allowed_first_rows = set(allowed_first_rows)

        candidates = []
        for first_row in range(max_row + 1):
            # 一次 120px 的短拖最多只会跨过约三行。物资图标中存在
            # 外形相近的纸张/布匹，若不结合上一屏的位置约束，少量弱
            # 特征可能把第 2 行误判成第 13 行，继而触发整表回顶重扫。
            if (
                allowed_first_rows is not None
                and first_row not in allowed_first_rows
            ):
                continue
            scores = []
            matches = []
            for screen_row, column, cell in sampled:
                name = reverse_layout.get((first_row + screen_row, column))
                features = cell_features.get(cell)
                template = self._resource_icon_templates.get(name)
                if name is None or features is None or template is None:
                    continue
                score = self._compare_cell_to_template(
                    features, name, template,
                )
                scores.append(score)
                if score >= self.RESOURCE_CALIBRATION_ICON_THRESHOLD:
                    matches.append((name, score))
            if len(matches) < 4:
                continue
            # 最强的三枚图标足以定位；弱纹理/缺货角标不会拖垮整屏结果。
            strongest = sorted(scores, reverse=True)[:3]
            quality = float(sum(strongest) / len(strongest))
            candidates.append((quality, len(matches), first_row, matches))

        if not candidates:
            logger.info('Allocation icon calibration has insufficient votes')
            return None
        candidates.sort(reverse=True)
        quality, votes, selected, matches = candidates[0]
        second_quality = candidates[1][0] if len(candidates) > 1 else 0.0
        if quality - second_quality < self.RESOURCE_CALIBRATION_MARGIN:
            logger.warning(
                f'Ambiguous allocation page position: row={selected}, '
                f'score={quality:.3f}, second={second_quality:.3f}'
            )
            return None
        logger.attr(
            'AllocationIconCalibrationRow',
            f'row={selected}, votes={votes}/{sample_count}, '
            f'score={quality:.3f}, anchors='
            + ','.join(f'{name}:{score:.2f}' for name, score in matches),
        )
        return selected

    def _click_calibrated_resource(
        self, resource, expected_position, visible_layout, first_row
    ):
        """目标位于当前画面时，按配置格定位并用图标做最终确认。"""
        expected_row, expected_column = expected_position
        screen_row = expected_row - first_row
        cell = next((
            item_cell
            for item_row, item_column, item_cell in visible_layout
            if item_row == screen_row and item_column == expected_column
        ), None)
        if cell is None:
            return None
        if cell not in self._yellow_resource_cells([cell]):
            logger.warning(
                f'Refuse non-local calibrated allocation cell: {resource}'
            )
            self.last_error = 'resource_not_local'
            return False
        match = self._find_resource_icon(
            resource,
            cells=[cell],
            validate=True,
            # 整屏投票只能证明当前页面行，不能证明目标格中的图标身份。
            # 实机油纸误选日志中页面校准 0.960，但目标自身仅 0.248。
            # 目标分数不足强识别门槛时不要点击，交给页面提示/视觉定位兜底。
            threshold=self.RESOURCE_ICON_STRONG_THRESHOLD,
        )
        if match is None:
            logger.warning(
                f'Configured resource cell failed icon verification: '
                f'{resource}, global_row={expected_row}, '
                f'column={expected_column}'
            )
            return None
        (x, y), score = match
        logger.info(
            f'Select allocation resource by calibrated layout: {resource}, '
            f'row={expected_row}, column={expected_column}, '
            f'center=({x}, {y}), score={score:.3f}'
        )
        self.last_resource_screen = (int(x), int(y))
        self.last_resource_grid = (int(expected_row), int(expected_column))
        self.last_resource_icon_score = float(score)
        self.device.click(Button(
            area=(), color=(),
            button=(x - 42, int(y) - 42, x + 42, int(y) + 42),
            name=f'ALLOCATION_RESOURCE_CALIBRATED_{resource}',
        ))
        self.device.sleep(0.8)
        self.device.screenshot()
        return self.is_resource_page()

    def _select_resource_from_current_position(
        self, resource, expected_position, configured_layout, max_moves=20
    ):
        """从列表当前停留位置定位目标；无法可靠定位时返回 None。"""
        top_confirmed = False
        if self._resource_scroll_offset is None:
            # 进入物资页后游戏可能保留旧滚动位置。先通过边界画面
            # 确认顶部，再以首行 0 为确定锚点；图标投票在实机上曾把
            # 明明位于顶部的画面误判成首行 1。
            if not self._reset_resource_top_boundary():
                return None
            top_confirmed = True
        previous_row = None
        previous_direction = None
        for _ in range(max_moves + 1):
            self.device.screenshot()
            cells = self.find_resource_cells()
            visible_layout = self._resource_cell_layout(cells)
            allowed_first_rows = None
            if previous_row is not None:
                if previous_direction == 'down':
                    allowed_first_rows = range(
                        previous_row, previous_row + 4
                    )
                else:
                    allowed_first_rows = range(
                        max(0, previous_row - 3), previous_row + 1
                    )
            # 同一帧把全部可见格的特征各提取一次，用整体投票定位。
            # 旧逻辑随机抽样三轮，会对同一批图标重复做三遍 SIFT。
            if top_confirmed:
                first_row = 0
                top_confirmed = False
                logger.info('Allocation resource first row anchored at confirmed top')
            else:
                first_row = self._infer_visible_global_row(
                    visible_layout,
                    configured_layout,
                    allowed_first_rows=allowed_first_rows,
                )
            if first_row is None and allowed_first_rows is not None:
                # The swipe may travel farther/shorter than expected.  Retry
                # against the complete configured layout instead of declaring
                # the current screen unlocatable because it left a fixed band.
                logger.warning(
                    'Allocation local row window missed; retry full-layout calibration'
                )
                first_row = self._infer_visible_global_row(
                    visible_layout,
                    configured_layout,
                    allowed_first_rows=None,
                )
            if first_row is None:
                return None

            expected_row, _ = expected_position
            visible_rows = [row for row, _, _ in visible_layout]
            if not visible_rows:
                return None
            last_row = first_row + max(visible_rows)
            if first_row <= expected_row <= last_row:
                return self._click_calibrated_resource(
                    resource,
                    expected_position,
                    visible_layout,
                    first_row,
                )

            # 根据识别出的绝对行决定方向和下一次拖动距离。
            if previous_row == first_row:
                logger.warning('Allocation page row calibration did not advance')
                return None
            downward = expected_row > last_row
            row_gap = (
                expected_row - last_row
                if downward else first_row - expected_row
            )
            drag_distance = 115 if row_gap <= 1 else 220
            start_y = 420 if downward else 300
            end_y = (
                max(170, start_y - drag_distance)
                if downward else min(550, start_y + drag_distance)
            )
            before = self._resource_grid_image()
            self._resource_drag_seq += 1
            self.jiangnan_drag(
                (850, start_y),
                (850, end_y),
                hold_duration=0.2,
                name=(
                    f'ALLOCATION_RESOURCE_LOCATE_'
                    f'{self._resource_drag_seq:04d}'
                ),
            )
            self.device.sleep(0.8)
            self.device.screenshot()
            if not self.is_resource_page():
                return False
            after = self._resource_grid_image()
            if self._image_change(before, after) < 0.8:
                logger.warning('Allocation resource list stopped before target')
                return None
            logger.attr(
                'AllocationAdaptiveDrag',
                f'first_row={first_row}, target_row={expected_row}, '
                f'gap={row_gap}, distance={drag_distance}, '
                f'direction={"down" if downward else "up"}',
            )
            previous_row = first_row
            previous_direction = 'down' if downward else 'up'
        return None

    def calibrate_resource_icon_library(self, max_swipes=24):
        """图标从仓库截图学习；调配页位置由辅助信息单独维护。"""
        logger.warning(
            'Resource icons must be learned by the warehouse scanner; '
            'use auxiliary layout for allocation-page positioning'
        )
        return False

    def _select_resource_at_position(
        self, resource, position, view_start, threshold=None,
        allow_automatic_export=False,
    ):
        """在已知分页中只验证辅助信息指定的一个格子。"""
        expected_row, expected_column = position
        screen_row = int(expected_row) - int(view_start)
        if not 0 <= screen_row < self.RESOURCE_VIEW_ROWS:
            return False

        self.device.screenshot()
        if not self.is_resource_page():
            return False
        cells = self.find_resource_cells()
        visible_layout = self._resource_cell_layout(cells)
        cell = next((
            item_cell
            for item_row, item_column, item_cell in visible_layout
            if item_row == screen_row and item_column == expected_column
        ), None)
        if cell is None:
            # 方框轮廓受缺货角标干扰时，仍只允许在指定
            # 行列的小范围内找格子，不扩大到整屏。
            logical_y = (
                self.RESOURCE_FIRST_ROW_Y
                + screen_row * self.RESOURCE_ROW_HEIGHT
            )
            cell = self.find_resource_cell(expected_column, logical_y)
        if cell is None:
            logger.warning(
                f'Allocation configured cell is missing: {resource}, '
                f'position={position}, view_start={view_start}'
            )
            return False
        if (not allow_automatic_export
                and cell not in self._yellow_resource_cells([cell])):
            logger.warning(
                f'Refuse non-local allocation resource with white border: '
                f'{resource}, position={position}'
            )
            self.last_error = 'resource_not_local'
            return False

        score = self._verify_resource_cell_identity(
            cell,
            resource,
            threshold=(0.25 if threshold is None else threshold),
        )
        if not score:
            logger.warning(
                f'Allocation configured cell icon mismatch: {resource}, '
                f'position={position}, view_start={view_start}'
            )
            self.last_error = 'resource_identity_mismatch'
            return False

        if getattr(self, 'last_error', None) == 'resource_identity_mismatch':
            self.last_error = None
        x, y = cell
        logger.info(
            f'Select allocation resource by page hint: {resource}, '
            f'row={expected_row}, column={expected_column}, '
            f'view_start={view_start}, center=({x}, {y}), '
            f'score={score:.3f}'
        )
        self.last_resource_screen = (int(x), int(y))
        self.last_resource_grid = (
            int(expected_row), int(expected_column)
        )
        self.last_resource_icon_score = float(score)
        self.device.click(Button(
            area=(), color=(),
            button=(x - 42, int(y) - 42, x + 42, int(y) + 42),
            name=f'ALLOCATION_RESOURCE_HINT_{resource}',
        ))
        self.device.sleep(0.8)
        self.device.screenshot()
        return self.is_resource_page()

    def _scan_resource_views(
        self, resource, configured_layout, layout_page=None
    ):
        """辅助格失配时，两行步进快速遍历黄边区。"""
        max_row = max(
            (position[0] for position in configured_layout.values()),
            default=0,
        )
        # The hint may omit new local goods (Ningbo's net moved from row 5
        # to row 6). Its old length cannot bound a recovery scan. Use the
        # known cargo library as a finite upper bound; the live yellow/white
        # boundary below still stops the scan as soon as local goods end.
        max_row = max(max_row, (len(getattr(self, '_resource_icon_templates', {})) - 1) // 4)
        starts = self._resource_view_starts(max_row)
        self._move_resource_to_view(starts[0], max_row)
        best = None
        yellow_positions = set()

        for view_start in starts:
            self._move_resource_to_view(view_start, max_row)
            self.device.screenshot()
            if not self.is_resource_page():
                return False

            # 每屏只找一次所有格子和黄边框。白底格既不提取
            # SIFT 特征，也不继续向下扫描。
            cells = self.find_resource_cells()
            visible_layout = self._resource_cell_layout(cells)
            yellow_cells = set(self._yellow_resource_cells(cells))
            yellow_positions.update(
                (int(view_start + screen_row), int(column))
                for screen_row, column, cell in visible_layout
                if cell in yellow_cells
            )
            visible_cells = [
                cell for _, _, cell in visible_layout
                if cell in yellow_cells
            ]
            candidate = self._find_resource_icon_content(
                resource,
                cells=visible_cells,
                validate=True,
                threshold=0.25,
            )
            if candidate is not None:
                cell, score = candidate
                item = next((
                    (screen_row, column)
                    for screen_row, column, item_cell in visible_layout
                    if item_cell == cell
                ), None)
                if item is not None and (
                    best is None or score > best[0]
                ):
                    screen_row, column = item
                    best = (
                        float(score), int(view_start),
                        (int(view_start + screen_row), int(column)),
                    )
            logger.attr(
                'AllocationResourceFastScan',
                f'{resource}: first_row={view_start}, '
                f'yellow={len(visible_cells)}, '
                f'best={0.0 if best is None else best[0]:.3f}',
            )

            # 黄边货物从列表顶部连续排列。同一屏已出现
            # 黄边之后的白底格，说明最后一行黄边已完整展示。
            # 保留的一行重叠使这个终止判断不依赖滑动精度。
            yellow_rows = {
                screen_row
                for screen_row, _, cell in visible_layout
                if cell in yellow_cells
            }
            white_rows = {
                screen_row
                for screen_row, _, cell in visible_layout
                if cell not in yellow_cells
            }
            if not yellow_rows or (
                white_rows and max(white_rows) >= max(yellow_rows)
            ):
                logger.info(
                    f'Allocation local-resource boundary reached at '
                    f'first_row={view_start}'
                )
                break

        # 旧版辅助表包含全部白底货物。首次快速遍历后
        # 只保留实际黄边格；目标图标若在新位置被确认，
        # 同步修正它的行列。
        confirmed_layout = None
        if layout_page is not None and yellow_positions:
            trimmed = {
                name: tuple(position)
                for name, position in configured_layout.items()
                if tuple(position) in yellow_positions
            }
            if best is not None:
                observed = tuple(best[2])
                old_position = trimmed.get(resource)
                displaced = next((
                    name for name, position in trimmed.items()
                    if name != resource and tuple(position) == observed
                ), None)
                if displaced is not None:
                    if old_position in yellow_positions:
                        trimmed[displaced] = tuple(old_position)
                    else:
                        del trimmed[displaced]
                trimmed[resource] = observed
            if trimmed != configured_layout:
                confirmed_layout = trimmed

        if best is None or best[0] < 0.25:
            logger.warning(f'Resource icon not recognized: {resource}')
            self.last_error = 'resource_icon_not_found'
            return False

        score, view_start, observed = best
        logger.info(
            f'Allocation fast scan located resource: {resource}, '
            f'position={observed}, view_start={view_start}, score={score:.3f}'
        )
        # 遍历结束后直接回到最佳屏，只复核已识别到的
        # 那一格，不再从顶部逐屏重放识别。
        self._move_resource_to_view(view_start, max_row)
        selected = self._select_resource_at_position(
            resource,
            observed,
            view_start,
            threshold=0.25,
        )
        # The fast scan only proposes a location. The per-cell identity
        # check can still reject a visually similar cargo icon; never save
        # that unconfirmed proposal as the next run's layout hint.
        if selected and confirmed_layout is not None:
            self.resource_layout.replace_layout(layout_page,
                                                confirmed_layout)
        return selected

    def _select_resource_legacy(self, resource, source=None):
        resource = self.normalize_resource_name(resource)
        # 每笔运输必须产生自己的已验证位置，不能沿用上一笔选择留下的
        # 格位参与本次辅助信息学习。
        self.last_resource_screen = None
        self.last_resource_grid = None
        self.last_resource_icon_score = 0.0
        self._resource_identity_tentative = False
        resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        if resource not in resources:
            logger.warning(f'Unknown allocation resource: {resource}')
            return False
        if not self.is_resource_page():
            logger.warning('Not in resource selection page')
            return False
        self._load_resource_icon_templates()
        if resource not in self._resource_icon_templates:
            logger.warning(
                f'Resource icon mapping is missing: {resource}; '
                'request warehouse scan instead of selecting by position'
            )
            self.last_error = 'resource_icon_missing'
            return False

        # 辅助信息是调配页面位置的唯一依据。先随机抽样当前画面图标，
        # 反推出列表停留行；能够定位时直接向目标行移动，避免每次都回
        # 顶部和盲扫整个列表。
        index = resources.index(resource)
        layout_page = allocation_layout_page(source or '应天府')
        configured_layout = self.resource_layout.load(layout_page)
        expected_position = configured_layout.get(resource)
        if expected_position is None:
            expected_position = divmod(index, 4)
        expected_row, expected_column = expected_position
        max_layout_row = max(
            (position[0] for position in configured_layout.values()),
            default=expected_row,
        )
        max_views = max_layout_row + 1
        logger.attr(
            'AllocationResourcePositionHint',
            f'{resource}: row={expected_row}, column={expected_column}, '
            f'views={max_views}',
        )

        # 每一笔运输都从已知顶部建立唯一坐标基准。旧逻辑会先随机
        # 校准当前位置并试探滚动，失败后又回顶，造成“先下、再上、再
        # 下”的怪异轨迹。辅助信息只负责预定位，图标仍负责最终确认。
        if not self.reset_resource_top():
            return False

        # 运输页货物顺序由辅助信息给出。先按固定行列直接滚到
        # 目标附近，只对目标格做一次图标确认；资源身份不需要 PP-OCR，
        # 也不必在每一屏对所有格子计算相似度。若页面因版本差异或滚动
        # 惯性导致位置不一致，下面仍保留全局图标搜索作为兜底。
        logical_y = (
            self.RESOURCE_FIRST_ROW_Y
            + expected_row * self.RESOURCE_ROW_HEIGHT
        )
        screen_y = self._scroll_resource_until_visible(
            logical_y,
            max_swipes=min(18, expected_row + 3),
        )
        if screen_y is not None:
            self.device.screenshot()
            fixed_cell = self.find_resource_cell(
                expected_column,
                screen_y,
            )
            if fixed_cell is not None:
                fixed_match = self._find_resource_icon(
                    resource,
                    cells=[fixed_cell],
                    validate=True,
                    threshold=self.RESOURCE_ICON_FIXED_THRESHOLD,
                )
                if fixed_match is not None:
                    (x, fixed_y), score = fixed_match
                    logger.info(
                        f'Select allocation resource by fixed order: '
                        f'{resource}, row={expected_row}, '
                        f'column={expected_column}, center=({x}, {fixed_y}), '
                        f'score={score:.3f}'
                    )
                    self.last_resource_screen = (int(x), int(fixed_y))
                    self.last_resource_grid = (
                        int(expected_row), int(expected_column)
                    )
                    self.last_resource_icon_score = float(score)
                    self.device.click(Button(
                        area=(),
                        color=(),
                        button=(
                            x - 42, int(fixed_y) - 42,
                            x + 42, int(fixed_y) + 42,
                        ),
                        name=f'ALLOCATION_RESOURCE_FIXED_{resource}',
                    ))
                    self.device.sleep(0.8)
                    self.device.screenshot()
                    if self.is_resource_page():
                        return True
                    return False

            # 累计滚动偏移只用于把目标送进当前视野。兜底识别也必须
            # 限制在配置列和目标行附近，绝不能让一个弱相似图标把棉花
            # 从 (1,2) 跳到 (0,3)，或让竹子从第 6 行漂到第 9 行。
            visible_cells = self.find_resource_cells()
            nearby_cells = [
                cell for cell in visible_cells
                if abs(cell[0] - self.RESOURCE_GRID_X[expected_column]) <= 35
                and abs(cell[1] - screen_y) <= 70
            ]
            nearby_match = self._find_resource_icon(
                resource,
                cells=nearby_cells,
                validate=True,
                threshold=self.RESOURCE_ICON_FIXED_THRESHOLD,
            )
            if nearby_match is not None:
                (x, nearby_y), score = nearby_match
                logger.info(
                    f'Select allocation resource in constrained cell: '
                    f'{resource}, expected=({expected_row}, '
                    f'{expected_column}), center=({x}, {nearby_y}), '
                    f'score={score:.3f}'
                )
                self.last_resource_screen = (int(x), int(nearby_y))
                self.last_resource_grid = (
                    int(expected_row), int(expected_column)
                )
                self.last_resource_icon_score = float(score)
                self.device.click(Button(
                    area=(), color=(),
                    button=(
                        x - 42, int(nearby_y) - 42,
                        x + 42, int(nearby_y) + 42,
                    ),
                    name=f'ALLOCATION_RESOURCE_NEARBY_{resource}',
                ))
                self.device.sleep(0.8)
                self.device.screenshot()
                if self.is_resource_page():
                    return True
                return False

        # 位置只负责缩小首轮范围，图标仍是最终依据。邻域找不到时按
        # 用户配置的策略执行全表图标识别；不能在这里提前返回，否则
        # 下方的批量识别与位置自动修正永远不可达。
        logger.warning(
            f'Resource icon not found near configured position: {resource}, '
            f'expected=({expected_row}, {expected_column})'
        )
        logger.warning(
            f'Fixed-order resource lookup failed: {resource}; '
            'fallback to global icon search'
        )
        if not self.reset_resource_top():
            return False

        search_offset = 0.0

        def next_view(label):
            nonlocal search_offset
            before = self._resource_grid_image()
            self._resource_drag_seq += 1
            self.jiangnan_drag(
                (850, 420), (850, 300), hold_duration=0.2,
                name=f'{label}_{self._resource_drag_seq:04d}',
            )
            self.device.sleep(0.8)
            self.device.screenshot()
            if not self.is_resource_page():
                return None
            after = self._resource_grid_image()
            moved = self._image_change(before, after) >= 0.8
            if moved:
                shift, score = self._estimate_scroll_shift(before, after)
                if shift is None:
                    # 固定顺序学习只在三次观察一致后才回写；追踪失败时
                    # 用一行作为保守估计，不会因单次误差污染配置。
                    shift = self.RESOURCE_ROW_HEIGHT
                search_offset += shift
                logger.attr(
                    'AllocationLayoutSearchOffset',
                    f'{search_offset:.1f} (match={score:.3f})',
                )
            return moved

        # 第一遍只观察，不点击：在允许的位置范围内找全局最高相似度。
        # 这样油纸不会因为前面先看到外形相似的纸而被提前误选。
        best_score = -1.0
        best_view = None
        for view in range(max_views):
            self.device.screenshot()
            candidate = self._find_resource_icon(resource, validate=False)
            if candidate is not None and candidate[1] > best_score:
                best_score = candidate[1]
                best_view = view
            moved = next_view('ALLOCATION_RESOURCE_PROBE')
            if moved is None:
                return False
            if not moved:
                break

        if best_view is None or best_score < self.RESOURCE_ICON_THRESHOLD:
            logger.warning(f'Resource icon not recognized: {resource}')
            self.last_error = 'resource_icon_not_found'
            return False

        # 回到顶部重放到最佳页。每次仍重新识别，允许滑动像素略有差异；
        # 分数必须接近第一遍的全局最佳值才会点击。
        if not self.reset_resource_top():
            return False
        search_offset = 0.0
        match = None
        replay_threshold = max(
            self.RESOURCE_ICON_THRESHOLD, best_score - 0.06
        )
        for view in range(min(max_views, best_view + 3)):
            self.device.screenshot()
            candidate = self._find_resource_icon(resource, validate=True)
            if candidate is not None and candidate[1] >= replay_threshold:
                match = candidate
                break
            if view >= best_view + 2:
                break
            moved = next_view('ALLOCATION_RESOURCE_REPLAY')
            if moved is None:
                return False
            if not moved:
                break
        if match is None:
            logger.warning(
                f'Resource icon replay failed: {resource}, '
                f'best={best_score:.3f}'
            )
            self.last_error = 'resource_icon_not_found'
            return False

        (x, screen_y), score = match
        logger.info(
            f'Select allocation resource by icon: {resource}, '
            f'center=({x}, {screen_y}), score={score:.3f}'
        )
        self.last_resource_screen = (int(x), int(screen_y))
        observed_row = max(0, int(round(
            (
                float(screen_y)
                + search_offset
                - self.RESOURCE_FIRST_ROW_Y
            ) / self.RESOURCE_ROW_HEIGHT
        )))
        observed_column = min(
            range(4),
            key=lambda column: abs(x - self.RESOURCE_GRID_X[column]),
        )
        self.last_resource_grid = (observed_row, observed_column)
        self.device.click(Button(
            area=(),
            color=(),
            button=(x - 42, int(screen_y) - 42, x + 42, int(screen_y) + 42),
            name=f'ALLOCATION_RESOURCE_ICON_{resource}',
        ))
        self.device.sleep(0.8)
        self.device.screenshot()
        if not self.is_resource_page():
            return False
        return True

    def select_resource(self, resource, source=None):
        """选择黄边本地货物或经明确配置的城市自动产出货物。"""
        resource = self.normalize_resource_name(resource)
        self.last_resource_screen = None
        self.last_resource_grid = None
        self.last_resource_icon_score = 0.0
        self._resource_identity_tentative = False
        resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        if resource not in resources:
            logger.warning(f'Unknown allocation resource: {resource}')
            return False
        if not self.is_resource_page():
            logger.warning('Not in resource selection page')
            return False

        self._load_resource_icon_templates()
        if resource not in self._resource_icon_templates:
            logger.warning(f'Resource icon mapping is missing: {resource}')
            self.last_error = 'resource_icon_missing'
            return False

        layout_page = allocation_layout_page(source or '应天府')
        configured_layout = self.resource_layout.load(layout_page)
        expected_position = configured_layout.get(resource)
        if resource in AUTOMATIC_EXPORT_RESOURCES.get(source, ()):
            # 自动产出的菜肴没有餐馆，因此鸡鸣山的黄边辅助表为空。
            # 货物顺序仍固定；只尝试这一格，并要求图标确认。
            expected_position = divmod(resources.index(resource), 4)
            view_start = (
                expected_position[0] // self.RESOURCE_VIEW_STRIDE
            ) * self.RESOURCE_VIEW_STRIDE
            if not self._move_resource_to_view(
                view_start, expected_position[0],
            ):
                return False
            return self._select_resource_at_position(
                resource, expected_position, view_start,
                allow_automatic_export=True,
            )
        if expected_position is None:
            logger.warning(
                f'Refuse exporting non-local resource: source={source}, '
                f'resource={resource}; not present in yellow-border layout'
            )
            self.last_error = 'resource_not_local'
            return False


        # 旧版配置默认写入 65 种全部货物。每个出发城
        # 首次使用时用两行步进扫到黄边结束处，立即将
        # 辅助表裁成本地可生产集合。之后不再执行该遍历。
        # Copper mirrors were added after the old 65-item default table.
        # That saved table still needs migration even with the new item absent.
        if (set(resources) - {'铜镜'}).issubset(configured_layout):
            logger.info(
                f'Migrate full allocation layout to yellow-only layout: '
                f'{source}'
            )
            return self._scan_resource_views(
                resource,
                configured_layout,
                layout_page=layout_page,
            )

        expected_row, expected_column = expected_position
        max_row = max(
            (position[0] for position in configured_layout.values()),
            default=expected_row,
        )
        starts = self._resource_view_starts(max_row)
        target_start = min(
            (expected_row // self.RESOURCE_VIEW_STRIDE)
            * self.RESOURCE_VIEW_STRIDE,
            starts[-1],
        )
        logger.attr(
            'AllocationResourcePositionHint',
            f'{source}/{resource}: row={expected_row}, '
            f'column={expected_column}, view_start={target_start}',
        )

        # 从已确认的物理顶部出发，每次移动都由画面测量实际位移。
        # 旧的自适应短拖没有更新滚动状态，随后再按辅助表移动时会多拖
        # 一次，把相邻行（如茶叶）误当成目标刺绣。
        if self._move_resource_to_view(target_start, max_row):
            if self._select_resource_at_position(
                resource,
                expected_position,
                target_start,
            ):
                return True
            if self.last_error == 'resource_not_local':
                return False

        # 辅助格失效才进入快速学习：每次滑两行、保留
        # 一行重叠，只识别黄边格，看到完整白底行即停止。
        logger.warning(
            f'Allocation yellow-layout hint mismatch; fast rescan: '
            f'{source}/{resource}'
        )
        return self._scan_resource_views(
            resource,
            configured_layout,
            layout_page=layout_page,
        )

    @staticmethod
    def _parse_inventory_text(value):
        result = {}
        for line in str(value or '').splitlines():
            matched = re.match(r'^\s*([^:：]+)\s*[:：]\s*(-?\d+)\s*$', line)
            if matched:
                result[matched.group(1).strip()] = int(matched.group(2))
        return result

    def _configured_resource_remaining(self, destination, resource):
        """返回配置库存推导的单物资剩余容量；未知时返回 None。"""
        resource = self.normalize_resource_name(resource)
        if resource not in WAREHOUSE_RESOURCES:
            # 徽州共享仓库不能用单物资 capacity-inventory 校验。
            return None
        city_key = next(
            (key for key, value in CITY_KEY_TO_NAME.items()
             if value == destination),
            None,
        )
        if city_key is None:
            return None
        capacity = self.config.cross_get(
            keys=(
                f'JiangnanWarehouseInventory.'
                f'warehouse_{city_key}.Capacity'
            ),
            default=None,
        )
        inventory_text = self.config.cross_get(
            keys=(
                f'JiangnanWarehouseInventory.'
                f'warehouse_{city_key}.Inventory'
            ),
            default='',
        )
        try:
            capacity = int(capacity)
        except (TypeError, ValueError):
            return None
        inventory = self._parse_inventory_text(inventory_text).get(resource)
        if inventory is None or inventory < 0 or capacity <= 0:
            return None
        return max(0, capacity - inventory)

    def _reconcile_inventory_from_remaining(
        self, destination, resource, actual_remaining
    ):
        """用游戏实时剩余容量反推并回写目的城市的单项库存。"""
        resource = self.normalize_resource_name(resource)
        city_key = next(
            (key for key, value in CITY_KEY_TO_NAME.items()
             if value == destination),
            None,
        )
        if city_key is None or resource not in WAREHOUSE_RESOURCES:
            return False
        inventory_key = (
            f'JiangnanWarehouseInventory.'
            f'warehouse_{city_key}.Inventory'
        )
        capacity_key = (
            f'JiangnanWarehouseInventory.'
            f'warehouse_{city_key}.Capacity'
        )
        try:
            capacity = int(self.config.cross_get(
                keys=capacity_key, default=None
            ))
            remaining = int(actual_remaining)
        except (TypeError, ValueError):
            return False
        if capacity <= 0 or remaining < 0 or remaining > capacity:
            logger.warning(
                f'Cannot reconcile inventory from invalid capacity: '
                f'city={destination}, resource={resource}, '
                f'capacity={capacity}, remaining={remaining}'
            )
            return False

        inventory_text = self.config.cross_get(
            keys=inventory_key, default=''
        )
        inventory = self._parse_inventory_text(inventory_text)
        previous = int(inventory.get(resource, -1))
        inferred = capacity - remaining
        if remaining == 0 and previous > capacity:
            # “剩余 0”只能证明库存不低于上限，无法得知具体溢出量。
            # 保留已经扫描到的更大值，避免把真实溢出库存压回上限。
            inferred = previous
        if previous == inferred:
            return False

        inventory[resource] = inferred
        self.last_reconciled_inventory = (
            destination, resource, int(inferred), int(inferred - previous)
        )
        resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        self.config.modified[inventory_key] = '\n'.join(
            f'{name}: {int(inventory.get(name, -1))}'
            for name in resources
        )
        self.config.update()
        logger.warning(
            f'Reconcile warehouse inventory from live allocation page: '
            f'{destination} {resource} {previous} -> {inferred}, '
            f'capacity={capacity}, remaining={remaining}'
        )
        return True

    def _reconcile_source_inventory(self, source, resource, actual_stock):
        """Replace the source baseline with the stock shown on the selected tile."""
        city_key = next(
            (key for key, value in CITY_KEY_TO_NAME.items() if value == source),
            None,
        )
        resource = self.normalize_resource_name(resource)
        if city_key is None or resource not in (
            WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        ):
            return False
        try:
            actual_stock = int(actual_stock)
        except (TypeError, ValueError):
            return False
        if actual_stock < 0:
            return False
        inventory_key = (
            f'JiangnanWarehouseInventory.warehouse_{city_key}.Inventory'
        )
        inventory = self._parse_inventory_text(self.config.cross_get(
            keys=inventory_key, default=''
        ))
        previous = int(inventory.get(resource, -1))
        if previous == actual_stock:
            return False
        inventory[resource] = actual_stock
        self.last_reconciled_source_inventory = (
            source, resource, actual_stock, actual_stock - previous
        )
        resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
        self.config.modified[inventory_key] = '\n'.join(
            f'{name}: {int(inventory.get(name, -1))}' for name in resources
        )
        self.config.update()
        logger.warning(
            f'Reconcile source stock from live allocation tile: '
            f'{source} {resource} {previous} -> {actual_stock}'
        )
        return True

    def verify_selected_resource(
        self, destination, resource, actual_remaining, source=None
    ):
        """按配置库存校验图标识别结果。

        返回目的地剩余容量；无法可靠确认时返回 None，宁可取消
        本笔运输，也不把另一种货物错误发出。
        """
        expected = self._configured_resource_remaining(destination, resource)
        if getattr(self, '_resource_identity_tentative', False) and (
            expected is None or actual_remaining is None
        ):
            logger.warning(
                f'Allocation icon close tie lacks capacity confirmation: '
                f'{resource}; refuse departure'
            )
            return None
        if actual_remaining is None:
            return actual_remaining
        if expected is None:
            # A strong, unambiguous icon can establish identity even when the
            # prior stock is missing. Keep the live destination baseline.
            reconciled = self._reconcile_inventory_from_remaining(
                destination, resource, actual_remaining
            )
            if not reconciled and resource in WAREHOUSE_SPECIAL_RESOURCES:
                logger.info(
                    f'Destination shared capacity does not identify '
                    f'individual special-goods stock: '
                    f'{destination} {resource}'
                )
            return actual_remaining
        if abs(int(actual_remaining) - expected) <= 5:
            logger.info(
                f'Allocation resource verified by capacity: '
                f'{resource}, remaining={actual_remaining}'
            )
            self._reconcile_inventory_from_remaining(
                destination, resource, actual_remaining
            )
            return actual_remaining

        layout_page = allocation_layout_page(source or '应天府')
        configured_position = self.resource_layout.position(
            layout_page, resource
        )
        if (configured_position is None
                and source is not None
                and resource in AUTOMATIC_EXPORT_RESOURCES.get(source, ())):
            # 鸡鸣山菜肴等自动产物不在本地生产黄边表里，但选择时已
            # 按固定货物顺序定位并核对图标。目的地旧库存失准时，
            # 仍可用同一固定格位做强匹配复核。
            resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
            configured_position = divmod(resources.index(resource), 4)
        strong_fixed_match = (
            not getattr(self, '_resource_identity_tentative', False)
            and configured_position is not None
            and tuple(self.last_resource_grid or ()) == tuple(configured_position)
            and float(self.last_resource_icon_score or 0.0)
            >= self.RESOURCE_ICON_STRONG_THRESHOLD
        )
        if strong_fixed_match:
            logger.warning(
                f'Configured inventory changed after its last scan: '
                f'resource={resource}, expected_remaining={expected}, '
                f'actual={actual_remaining}, '
                f'icon_score={self.last_resource_icon_score:.3f}; '
                f'accept strong fixed-cell recognition'
            )
            self._reconcile_inventory_from_remaining(
                destination, resource, actual_remaining
            )
            return actual_remaining

        logger.warning(
            f'Allocation icon verification failed: resource={resource}, '
            f'expected_remaining={expected}, actual={actual_remaining}; '
            'do not correct by row position'
        )
        return None

    def find_resource_cell(self, column, expected_y):
        """从当前截图识别物资方格轮廓，返回目标列最接近期望行的中心。"""
        x1, y1, x2, y2 = self.RESOURCE_GRID_AREA
        image = self.device.image[y1:y2, x1:x2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 100)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if 75 <= width <= 115 and 65 <= height <= 115:
                center = (
                    x1 + left + width // 2,
                    y1 + top + height // 2,
                )
                # 同一方框的内外轮廓只保留一个中心。
                if not any(
                    abs(center[0] - old[0]) <= 6
                    and abs(center[1] - old[1]) <= 6
                    for old in candidates
                ):
                    candidates.append(center)

        reference_x = self.RESOURCE_GRID_X[column]
        same_column = [
            center for center in candidates
            if abs(center[0] - reference_x) <= 35
        ]
        if not same_column:
            logger.warning(
                f'Unable to recognize resource grid column: {column}'
            )
            return None
        center = min(
            same_column,
            key=lambda point: abs(point[1] - expected_y),
        )
        if abs(center[1] - expected_y) > 65:
            logger.warning(
                f'Resource cell recognition too far from expected row: '
                f'expected={expected_y:.1f}, detected={center}'
            )
            return None
        logger.info(
            f'Resource cell recognized: column={column}, '
            f'expected_y={expected_y:.1f}, center={center}'
        )
        return center

    # ========================================================
    # Amount, departure and assignee
    # ========================================================

    @staticmethod
    def normalize_amount(amount):
        if isinstance(amount, bool):
            return None
        if isinstance(amount, float) and not amount.is_integer():
            return None
        if isinstance(amount, str) and not re.fullmatch(r'\d+', amount.strip()):
            return None
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            return None
        return amount if amount > 0 else None

    def read_destination_capacity(self, max_value=100000):
        self.device.screenshot()
        area = self.anchored_area(
            ALLOCATION_RESOURCE_CHECK,
            AllocationCapacityOcr.DEFAULT_AREA,
        )
        capacity = self.read_digit_consensus(
            AllocationCapacityOcr,
            area,
            name='AllocationDestinationCapacity',
            max_value=max_value,
            allow_zero=True,
        )
        logger.attr('AllocationDestinationCapacity', capacity)
        return capacity

    def read_destination_in_transit(self, max_value=100000):
        """选中物资后，读取“正在运输的同类物资”右侧数字。"""
        self.device.screenshot()
        label_area = self.anchored_area(
            ALLOCATION_RESOURCE_CHECK, (340, 150, 530, 185),
        )
        x1, y1, x2, y2 = label_area
        image = getattr(self.device, 'image', None)
        if image is None or image.size == 0:
            return None
        label = image[y1:y2, x1:x2]
        if label.size == 0:
            return None
        dark_pixels = int(cv2.countNonZero(cv2.inRange(
            cv2.cvtColor(label, cv2.COLOR_RGB2GRAY), 0, 129,
        )))
        logger.attr('AllocationInTransitLabelPixels', dark_pixels)
        if dark_pixels < 100:
            # With no incoming shipment the game hides the whole second
            # line. An empty crop is then a known zero, not failed OCR.
            logger.attr('AllocationDestinationInTransit', 0)
            return 0
        area = self.anchored_area(
            ALLOCATION_RESOURCE_CHECK,
            AllocationInTransitOcr.DEFAULT_AREA,
        )
        amount = self.read_digit_consensus(
            AllocationInTransitOcr, area,
            name='AllocationDestinationInTransit',
            max_value=max_value, allow_zero=True,
        )
        logger.attr('AllocationDestinationInTransit', amount)
        return amount

    def read_digit_consensus(
        self,
        ocr_type,
        area,
        name,
        max_value,
        allow_zero=False,
    ):
        """PP-OCRv6 tiny/small/medium 级联识别原始数字裁剪。"""
        raw = ocr_type(area=area, lang='cnocr').ocr(self.device.image)
        if isinstance(raw, list):
            raw = raw[0] if raw else ''
        digits = ''.join(re.findall(r'\d+', str(raw)))
        if not digits:
            logger.warning(f'{name} OCR returned no digits: {raw!r}')
            return None
        value = int(digits)
        if value > max_value or (not allow_zero and value <= 0):
            logger.warning(f'{name} OCR value out of range: {value}')
            return None
        logger.info(f'{name} PP-OCRv6 result: {value}')
        return value

    def _confirm_selected_stock(self, area, candidates):
        confirmed = read_confirmed_digits(self.device.image, area)
        if confirmed is not None and confirmed in candidates:
            logger.info(f'Allocation source stock confirmed by independent OCR models: {confirmed}')
            return confirmed
        return None

    def _read_selected_source_stock(self):
        """Read the selected grid cell's live stock, not cached planner stock."""
        self._source_stock_is_lower_bound = False
        center = getattr(self, 'last_resource_screen', None)
        if center is None:
            return None
        x, y = center
        height, width = self.device.image.shape[:2]
        # The number sits at the lower right of the selected tile. The old
        # crop included the resource drawing and read a visible 0 as 20.
        area = (
            max(0, int(x) + 5), max(0, int(y) + 18),
            min(width, int(x) + 48), min(height, int(y) + 40),
        )
        if area[2] <= area[0] or area[3] <= area[1]:
            return None
        stock = self.read_digit_consensus(
            AllocationAmountOcr, area,
            name='AllocationSelectedSourceStock', max_value=100000,
            allow_zero=True,
        )
        # Even a four-digit value may look like one or two digits in the
        # narrow crop (4029 -> 29).  Always inspect the entire number line.
        full_area = (max(0, int(x) - 25), area[1], area[2], area[3])
        full_stock = self.read_digit_consensus(
            AllocationAmountOcr, full_area,
            name='AllocationSelectedSourceStockFull',
            max_value=100000, allow_zero=True,
        )
        if stock is not None and stock == full_stock and stock >= 100:
            return stock
        if full_stock is not None and full_stock >= 100:
            # The narrow crop can also lose a digit in the middle (2409 ->
            # 109).  A second full-width read with a slightly different
            # boundary must agree before that value can control departure.
            alternate_area = (
                max(0, int(x) - 31), max(0, int(y) + 20),
                area[2], min(height, int(y) + 42),
            )
            alternate = self.read_digit_consensus(
                AllocationAmountOcr, alternate_area,
                name='AllocationSelectedSourceStockAlternate',
                max_value=100000, allow_zero=True,
            )
            if full_stock != alternate and hasattr(self, '_confirm_selected_stock'):
                confirmed = self._confirm_selected_stock(alternate_area, (stock, full_stock, alternate))
                if confirmed is not None:
                    return confirmed
            expected = getattr(self, '_source_stock_hint', None)
            if (full_stock > 20000 and alternate is not None
                    and 1000 <= alternate <= 20000 and expected == alternate
                    and stock is not None and stock > 0
                    and str(alternate).endswith(str(stock))
                    and len(str(full_stock)) == len(str(alternate)) + 1
                    and any(str(full_stock)[:i] + str(full_stock)[i+1:] == str(alternate)
                            for i in (0, 1))):
                # Copper plates' dark icon stroke becomes an extra leading
                # digit (2640 -> 21640). Accept the smaller live reading only
                # when its suffix and a confirmed earlier stock agree too.
                self._source_stock_is_lower_bound = True
                logger.info(f'Allocation stock icon-stroke lower bound: narrow={stock}, '
                            f'full={full_stock}, alternate={alternate}, previous={expected}')
                return alternate
            if (alternate == full_stock and stock is not None
                    and expected is not None and expected >= 100
                    and len(str(full_stock)) == len(str(expected)) >= 4
                    and str(full_stock)[1:] == str(expected)[1:]
                    and str(full_stock).endswith(str(stock))
                    and full_stock != expected):
                # Both wide crops can include the same first glyph of the
                # resource icon: 1410 was read as 9410 in the live paper tile.
                # A previous scan agreeing on every other digit is enough to
                # bound the shipment, but not to overwrite exact inventory.
                bound = min(full_stock, expected)
                self._source_stock_is_lower_bound = True
                logger.warning(
                    'Allocation source stock first digit conflicts with '
                    f'previous scan: narrow={stock}, full={full_stock}, '
                    f'previous={expected}; use bound={bound}'
                )
                return bound
            # Both wide crops can include the same dark stroke of the icon.
            # The tea tile in the 23:14 log visibly held 3055, while both
            # crops read 23055 and the narrow crop read 55.  Per-item stock
            # cannot reach 20,000 here, so discard that shared false prefix.
            if (alternate == full_stock and full_stock > 20000
                    and stock is not None and stock > 0):
                suffix = int(str(full_stock)[1:])
                if (100 <= suffix <= 20000
                        and str(suffix).endswith(str(stock))):
                    logger.info(
                        f'Allocation stock ignored shared icon prefix: '
                        f'narrow={stock}, full={full_stock}, '
                        f'alternate={alternate}, stock={suffix}'
                    )
                    return suffix
            # A resource icon can add a false leading digit to just one wide
            # crop (4030 -> 24030 or 1887 -> 91887).  No known per-item
            # warehouse capacity approaches 20,000; keep the independently
            # supported suffix when only the prefixed reading exceeds it.
            if (stock is not None and stock > 0 and alternate is not None
                    and full_stock != alternate):
                if (full_stock <= 20000 < alternate
                        and str(full_stock).endswith(str(stock)[-2:])
                        and str(alternate).endswith(str(full_stock))):
                    logger.info(
                        f'Allocation stock ignored prefixed alternate crop: '
                        f'narrow={stock}, full={full_stock}, '
                        f'alternate={alternate}'
                    )
                    return full_stock
                # When the resource drawing overlaps the first two digits,
                # one crop can lose a repeated digit (2778 -> 278) while
                # the widest crop adds an icon prefix (2778 -> 22778).
                # Require the narrow suffix and this exact one-digit
                # deletion pattern before using the recovered count as a
                # conservative availability bound.
                if alternate > 20000 and 100 <= full_stock < 1000:
                    recovered = int(str(alternate)[1:])
                    digits = str(recovered)
                    if (1000 <= recovered <= 20000
                            and digits.endswith(str(stock))
                            and any(
                                digits[i] == digits[i + 1]
                                and digits[:i] + digits[i + 1:]
                                == str(full_stock)
                                for i in range(len(digits) - 1)
                            )):
                        self._source_stock_is_lower_bound = True
                        logger.info(
                            f'Allocation stock recovered repeated digit '
                            f'under icon: narrow={stock}, '
                            f'full={full_stock}, alternate={alternate}, '
                            f'bound={recovered}'
                        )
                        return recovered
                if (alternate <= 20000 < full_stock
                        and str(alternate).endswith(str(stock))
                        and str(full_stock).endswith(str(alternate))):
                    logger.info(
                        f'Allocation stock ignored prefixed full crop: '
                        f'narrow={stock}, full={full_stock}, '
                        f'alternate={alternate}'
                    )
                    return alternate
            plausible_extension = (
                stock is not None and stock > 0
                and str(full_stock).endswith(str(stock))
            )
            if alternate == full_stock and (
                plausible_extension or full_stock >= 1000
            ):
                logger.info(
                    f'Allocation stock recovered from full crops: '
                    f'narrow={stock}, full={full_stock}'
                )
                return full_stock
            if (
                alternate is not None
                and stock is not None
                and full_stock <= 20000
                and alternate <= 20000
                and len(str(full_stock)) == len(str(alternate)) >= 3
                and str(full_stock)[1:] == str(alternate)[1:]
                and str(full_stock).endswith(str(stock))
            ):
                # A wider left crop can turn the first 7 into 9, while
                # both views and the narrow crop agree on every remaining
                # digit. The smaller value is safe as an availability bound;
                # it must not be written back as an exact inventory reading.
                lower_bound = min(full_stock, alternate)
                self._source_stock_is_lower_bound = True
                logger.info(
                    f'Allocation source stock uses conservative bound: '
                    f'narrow={stock}, full={full_stock}, '
                    f'alternate={alternate}, bound={lower_bound}'
                )
                return lower_bound
            if (alternate is not None and stock is not None and stock > 0
                    and 100 <= full_stock <= 20000 and 100 <= alternate <= 20000
                    and str(full_stock).endswith(str(stock))
                    and str(alternate).endswith(str(stock))):
                smaller, larger = sorted((str(full_stock), str(alternate)), key=len)
                if len(larger) == len(smaller) + 1 and any(
                        larger[i] == larger[i + 1]
                        and larger[:i] + larger[i + 1:] == smaller
                        for i in range(len(larger) - 1)):
                    # 5111 can lose a repeated 1 underneath the resource icon
                    # and become 511. Use only the lower availability bound;
                    # neither reading is an authoritative inventory correction.
                    bound = min(full_stock, alternate)
                    self._source_stock_is_lower_bound = True
                    logger.info(f'Allocation stock repeated-digit lower bound: '
                                f'narrow={stock}, full={full_stock}, alternate={alternate}, bound={bound}')
                    return bound
            logger.warning(
                f'Allocation source stock full crops disagree: '
                f'full={full_stock}, alternate={alternate}'
            )
            return None
        if stock is not None and 0 < stock < 100:
            # For a one- or two-digit count, the number is right-aligned.
            # The wider crop can include a dark resource icon and turn a real
            # "0" into "50". Independently read only the number's right edge.
            right_area = (
                max(0, int(x) + 24), area[1], area[2], area[3]
            )
            right_stock = self.read_digit_consensus(
                AllocationAmountOcr, right_area,
                name='AllocationSelectedSourceStockRight',
                max_value=100, allow_zero=True,
            )
            if right_stock != stock:
                # The right-edge crop can clip the tens digit (24 -> 4).
                # Confirm the complete two-digit number with a shifted crop
                # before accepting it; a visible zero still takes priority.
                if (10 <= stock < 100 and full_stock == stock
                        and right_stock is not None and 0 < right_stock < 10
                        and stock % 10 == right_stock):
                    confirm_area = (
                        area[0], max(0, int(y) + 20),
                        area[2], min(height, int(y) + 42),
                    )
                    confirmed = self.read_digit_consensus(
                        AllocationAmountOcr, confirm_area,
                        name='AllocationSelectedSourceStockSmallConfirmed',
                        max_value=100, allow_zero=True,
                    )
                    if confirmed == stock:
                        logger.info(
                            f'Allocation stock confirmed clipped tens digit: '
                            f'stock={stock}, right={right_stock}'
                        )
                        return stock
                logger.warning(
                    f'Allocation source stock crops disagree: '
                    f'wide={stock}, right={right_stock}; '
                    'use right-aligned count only when it is zero'
                )
                return 0 if right_stock == 0 else None
        if stock is None and full_stock is not None:
            logger.warning(
                f'Allocation source stock narrow crop unreadable: '
                f'full={full_stock}'
            )
            return None
        if stock is not None and stock >= 100 and full_stock != stock:
            expected = getattr(self, '_source_stock_hint', None)
            if (full_stock is not None and full_stock > 0
                    and expected is not None and expected >= stock
                    and str(expected).endswith(str(stock))
                    and str(stock).endswith(str(full_stock))):
                # 菜肴图标压住四位数的首位时，窄/宽裁剪可能依次读成
                # 141/41，旧仓库记录则为 1141。只用窄裁剪作可运输下界，
                # 且后续输入框必须确认实际数量；不要据此改写精确库存。
                self._source_stock_is_lower_bound = True
                logger.warning(
                    f'Allocation source stock uses cropped lower bound: '
                    f'narrow={stock}, full={full_stock}, '
                    f'previous={expected}, bound={stock}'
                )
                return stock
            return None
        return stock

    def input_amount(self, amount, timeout=8):
        amount = self.normalize_amount(amount)
        if amount is None:
            logger.warning('Allocation amount must be a positive integer')
            return False

        logger.info(f'Input allocation amount: {amount}')
        # A selected resource with zero stock leaves this editor disabled.
        # The destination capacity can still be positive, so check the live
        # source cell before waiting for a keyboard that cannot open.
        if getattr(self, 'last_resource_screen', None) is not None:
            self.device.screenshot()
            source_stock = self._read_selected_source_stock()
            if source_stock == 0:
                logger.warning(
                    'Selected allocation resource has no source stock; '
                    'skip amount input'
                )
                self.last_error = 'source_empty'
                return False
        amount_button = self.anchored_button(
            ALLOCATION_RESOURCE_CHECK,
            ALLOCATION_AMOUNT_BUTTON,
            name='ALLOCATION_AMOUNT_BUTTON_DYNAMIC',
            inset=5,
        )
        x1, y1, x2, y2 = amount_button.button
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        amount_button = Button(
            area=(), color=(),
            button=(center_x - 5, center_y - 5,
                    center_x + 5, center_y + 5),
            name='ALLOCATION_AMOUNT_CENTER_DYNAMIC',
        )

        deadline = time.time() + timeout
        click_count = 0
        last_click = 0
        while time.time() < deadline:
            if click_count == 0 or (
                click_count < 2 and time.time() - last_click >= 2
            ):
                if click_count:
                    self.device.screenshot()
                    if not self.appear(
                        ALLOCATION_RESOURCE_CHECK, offset=(4, 4),
                        similarity=0.85,
                    ):
                        logger.warning(
                            'Allocation resource page left before amount retry'
                        )
                        return False
                    logger.info('Retry centered allocation amount field')
                self.device.click(amount_button)
                click_count += 1
                last_click = time.time()
            _, shown = self.device.u2.current_ime()
            if shown:
                break
            self.device.sleep(0.3)
        else:
            logger.warning('Allocation amount keyboard did not open')
            self.last_error = 'amount_keyboard_unavailable'
            self.save_abnormal_screenshot('allocation_amount_keyboard_unavailable')
            return False

        try:
            # MuMuManager input_text returns after only the first character
            # reaches this Android number editor (856 becomes 8).  Android's
            # own input command finishes synchronously for ASCII digits.
            self.device.adb_shell(['input', 'text', str(amount)])
            area = self.anchored_area(
                ALLOCATION_RESOURCE_CHECK,
                AllocationAmountOcr.DEFAULT_AREA,
            )
            source_stock = None
            editing_amount = None
            partial_amount = None
            partial_frames = 0
            suffix_retried = False
            for _ in range(max(3, min(20, int(timeout / 0.25)))):
                self.device.sleep(0.25)
                self.device.screenshot()
                editing_amount = self.read_digit_consensus(
                    AllocationAmountOcr, area,
                    name='AllocationAmountWhileEditing', max_value=100000,
                )
                if editing_amount == amount:
                    break
                if (editing_amount is not None and 0 < editing_amount < amount
                        and source_stock is None):
                    source_stock = self._read_selected_source_stock()
                if (source_stock is not None and source_stock < amount
                        and editing_amount == source_stock):
                    break
                # MuMu's number editor can consume only the first digit of
                # one `input text` command (220 -> 2). After the value stays
                # unchanged, append the missing digits as separate Android
                # key events; never press Enter on an unverified value.
                if (not suffix_retried and editing_amount is not None
                        and str(amount).startswith(str(editing_amount))
                        and editing_amount != amount):
                    if editing_amount == partial_amount:
                        partial_frames += 1
                    else:
                        partial_amount = editing_amount
                        partial_frames = 1
                    if partial_frames >= 3:
                        suffix = str(amount)[len(str(editing_amount)):]
                        logger.info(
                            f'Complete stalled allocation amount with '
                            f'key events: visible={editing_amount}, suffix={suffix}'
                        )
                        for digit in suffix:
                            self.device.adb_shell([
                                'input', 'keyevent', str(7 + int(digit))
                            ])
                            self.device.sleep(0.1)
                        suffix_retried = True
                else:
                    partial_amount = None
                    partial_frames = 0
            else:
                logger.warning(
                    f'Allocation editor did not reach requested amount: '
                    f'requested={amount}, visible={editing_amount}, '
                    f'source_stock={source_stock}'
                )
                self.last_error = 'amount_mismatch'
                self.save_abnormal_screenshot('allocation_amount_mismatch')
                self.device.adb_shell(['input', 'keyevent', '4'])
                return None
            self.device.adb_shell(['input', 'keyevent', '66'])
        except Exception as exc:
            logger.warning(f'Unable to input allocation amount: {exc}')
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            _, shown = self.device.u2.current_ime()
            if not shown:
                break
            self.device.sleep(0.3)
        else:
            logger.warning('Allocation amount keyboard did not close')
            return False

        self.device.sleep(0.8)
        self.device.screenshot()
        if not self.is_resource_page():
            logger.warning('Resource page missing after amount input')
            return False

        area = self.anchored_area(
            ALLOCATION_RESOURCE_CHECK,
            AllocationAmountOcr.DEFAULT_AREA,
        )
        actual_amount = self.read_digit_consensus(
            AllocationAmountOcr,
            area,
            name='AllocationAmountConfirmed',
            max_value=100000,
        )
        logger.attr('AllocationAmountConfirmed', actual_amount)
        if actual_amount is None or actual_amount <= 0:
            logger.warning('Unable to confirm a positive allocation amount')
            return None
        if actual_amount > amount:
            logger.warning(
                f'Allocation amount mismatch: expected={amount}, actual={actual_amount}'
            )
            return None
        if actual_amount < amount:
            source_stock = self._read_selected_source_stock()
            if source_stock == actual_amount:
                logger.warning(
                    f'Allocation limited by live source stock: '
                    f'requested={amount}, available={source_stock}; '
                    f'depart with confirmed amount'
                )
                return actual_amount
            logger.warning(
                f'Allocation amount input mismatch: '
                f'requested={amount}, actual={actual_amount}, '
                f'source_stock={source_stock}; refuse departure'
            )
            self.last_error = 'amount_mismatch'
            self.save_abnormal_screenshot('allocation_amount_mismatch')
            return None
        return actual_amount

    def _special_worker_slot(self, direct_start_visible, source=None):
        """Locate the first visible portrait on the assignee page."""
        reserved = reserved_special_names(getattr(self, 'config', None), source)
        image = getattr(self.device, 'image', None)
        if image is None or image.shape[0] < 205 or image.shape[1] < 830:
            return None
        left_first = ((335, 85, 450, 202), (355, 105, 430, 180))
        right_first = ((715, 85, 830, 202), (735, 105, 810, 180))
        slots = ([right_first] if direct_start_visible
                 else [left_first, right_first])
        if reserved:
            # Try other visible cards when a reserved resident is first.
            slots.extend([
                ((335, 300, 450, 417), (355, 320, 430, 395)),
                ((715, 300, 830, 417), (735, 320, 810, 395)),
                ((335, 515, 450, 632), (355, 535, 430, 610)),
                ((715, 515, 830, 632), (735, 535, 810, 610)),
            ])
        if direct_start_visible:
            slots = [
                ((x, y + 30, x + 116, y + 147),
                 (x + 20, y + 45, x + 96, y + 110))
                for x, y in unarranged_assignee_cards(image)
            ]
        for crop_area, click_area in slots:
            x1, y1, x2, y2 = crop_area
            if y2 > image.shape[0] or x2 > image.shape[1]:
                continue
            hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
            colored = float((hsv[:, :, 1] > 55).mean())
            logger.attr('AllocationSpecialWorkerCardColor', round(colored, 3))
            if colored > 0.16:
                if reserved:
                    # Direct-start pages use divider-relative card positions,
                    # including rows moved by scrolling, just like production.
                    if direct_start_visible:
                        name_y = y1 - 30
                        areas = (
                            (x1 + 119, name_y - 16, x1 + 324, name_y + 44),
                            (x1 + 123, name_y - 7, x1 + 291, name_y + 33),
                        )
                    else:
                        areas = (
                            (x1 + 119, y1 - 20, x1 + 324, y1 + 42),
                            (x1 + 123, y1 - 11, x1 + 291, y1 + 22),
                        )
                    readings = Ocr(
                        [Button(area=area, color=(), button=area,
                                name='ALLOCATION_WORKER_NAME')
                         for area in areas],
                        lang='cnocr', name='ALLOCATION_WORKER_NAMES_OCR',
                    ).ocr(image)
                    if not isinstance(readings, (list, tuple)):
                        readings = [readings]
                    names = [re.sub(r'[^\u4e00-\u9fff]', '', str(value))
                             for value in readings]
                    logger.attr('AllocationYangzhouWorkerName', names)
                    if (len(names) != 2 or len(names[0]) < 2
                            or names[0] != names[1]):
                        continue
                    if is_reserved_resident(names[0], reserved):
                        continue
                return click_area
        return None

    def _special_worker_duration(self, base_duration, slot):
        """Apply the transport-time reduction printed on the chosen card."""
        text_x = 845 if slot[0] > 600 else 465
        area = (text_x, slot[1] + 15, text_x + 230, slot[1] + 65)
        raw = str(Ocr(
            Button(area=area, color=(), button=area,
                   name='ALLOCATION_SPECIAL_TIME'),
            lang='cnocr', name='ALLOCATION_SPECIAL_TIME_OCR',
        ).ocr(self.device.image))
        match = re.search(r'(\d{1,3})\s*[%％]', raw)
        if match is None:
            logger.warning(f'Allocation special transport bonus unreadable: {raw!r}')
            return None
        reduction = int(match.group(1))
        if not 0 <= reduction < 100:
            logger.warning(f'Allocation special transport bonus invalid: {raw!r}')
            return None
        duration = base_duration * (100 - reduction) / 100
        logger.attr('AllocationSpecialTransportDuration',
                    f'{base_duration} - {reduction}% -> {duration}')
        return duration

    def depart(self, use_special=True, source=None):
        self._selected_transport_duration = None
        self.device.screenshot()
        if not self.appear(
            ALLOCATION_DEPART_BUTTON,
            offset=(4, 4),
            similarity=0.85,
        ):
            logger.warning(
                'Departure button is unavailable; check stock and destination capacity'
            )
            return False

        logger.info('Click allocation departure')
        self.device.click(ALLOCATION_DEPART_BUTTON)
        if not self._wait_page(self.is_assignee_page, 'assignee'):
            return False

        # A friend-city discovery toast can briefly cover the upper-left
        # "无人" card even though the assignee page is already recognized.
        # Wait for that overlay to clear before rejecting the dispatch.
        direct_start_visible = False
        visible_deadline = time.time() + 4
        while time.time() < visible_deadline:
            self.device.screenshot()
            if self.appear(
                ALLOCATION_DIRECT_START_BUTTON,
                offset=(18, 18),
                similarity=0.82,
            ):
                direct_start_visible = True
                break
            self.device.sleep(0.25)
        if use_special:
            special_slot = self._special_worker_slot(direct_start_visible, source)
            if special_slot is not None:
                base_duration = getattr(self, '_base_transport_duration', None)
                adjusted_duration = (
                    self._special_worker_duration(base_duration, special_slot)
                    if base_duration is not None else None
                )
                for attempt in range(2):
                    logger.info(
                        f'Choose allocation special resident '
                        f'({attempt + 1}/2)'
                    )
                    self.device.click(Button(
                        area=(), color=(), button=special_slot,
                        name='ALLOCATION_SPECIAL_ASSIGNEE',
                    ))
                    deadline = time.time() + 4
                    while time.time() < deadline:
                        self.device.sleep(0.5)
                        self.device.screenshot()
                        if (self.is_allocation_overview()
                                or self.is_world_map() or self.is_in_main()):
                            logger.info('Allocation special resident dispatched')
                            self._selected_transport_duration = adjusted_duration
                            return True
                        if self.handle_startup_popup():
                            continue
                        if not self.is_assignee_page():
                            # The selection took effect and the page is
                            # transitioning; never click another person.
                            deadline = max(deadline, time.time() + 8)
                            break
                    if not self.is_assignee_page():
                        while time.time() < deadline:
                            self.device.sleep(0.5)
                            self.device.screenshot()
                            if (self.is_allocation_overview()
                                    or self.is_world_map() or self.is_in_main()):
                                logger.info('Allocation special resident dispatched')
                                self._selected_transport_duration = adjusted_duration
                                return True
                        logger.warning('Allocation special selection changed page without confirmation')
                        self.last_error = 'special_dispatch_unconfirmed'
                        return False
                logger.warning('Allocation special resident unavailable; try direct start')
            else:
                logger.info('No visible allocation special resident; try direct start')

        if not direct_start_visible:
            logger.warning('Direct-start button was not found')
            self.save_abnormal_screenshot('allocation_direct_start_missing')
            self.last_error = 'direct_start_not_found'
            return False

        def click_direct_start(retry=0):
            # 模板匹配使用左侧“无人”图标，Button.button 已映射到右侧
            # “直接开始”文字区。日志证明只点图标中心 (395, 162) 会
            # 一直留在负责人页；重试时从文字区逐步向左覆盖整块卡片。
            x1, y1, x2, y2 = ALLOCATION_DIRECT_START_BUTTON.button
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            x_offsets = (0, -55, -105)
            click_x = center_x + x_offsets[min(retry, 2)]
            name = 'ALLOCATION_DIRECT_START_CENTER'
            if retry:
                name += f'_RETRY_{retry}'
            logger.info(
                f'Click recognized direct-start center: '
                f'anchor=({center_x}, {center_y}), '
                f'click=({click_x}, {center_y}), retry={retry}'
            )
            self.device.click(Button(
                area=(),
                color=(),
                button=(
                    click_x - 10, center_y - 10,
                    click_x + 10, center_y + 10,
                ),
                name=name,
            ))

        click_direct_start()

        deadline = time.time() + 15
        retry_count = 0
        next_retry = time.time() + 2.0
        while time.time() < deadline:
            self.device.sleep(0.6)
            self.device.screenshot()
            if (
                self.is_allocation_overview()
                or self.is_world_map()
                or self.is_in_main()
            ):
                logger.info('Allocation dispatch completed')
                return True

            if self.handle_startup_popup():
                continue

            # 页面明确仍停留在负责人选择页，说明上次点击没有生效。
            # 最多重试两次；页面已经开始切换时不会盲目重复点击。
            if (
                retry_count < 2
                and time.time() >= next_retry
                and self.is_assignee_page()
            ):
                if self.appear(
                    ALLOCATION_DIRECT_START_BUTTON,
                    offset=(18, 18),
                    similarity=0.82,
                ):
                    retry_count += 1
                    click_direct_start(retry=retry_count)
                    next_retry = time.time() + 2.0

        logger.warning('Allocation direct start was not confirmed')
        self.last_error = 'direct_start_failed'
        return False

    @staticmethod
    def parse_transport_duration(text):
        text = str(text).replace(' ', '')
        # 窄字体“小时”经常只识别成“时”或“小”，日志中真实出现过
        # “●1小8”。这里接受独立“小”作为小时单位，末尾无单位的噪声
        # 不参与分钟计算。
        hour = re.search(r'(\d+)\s*(?:小时|时|小)', text)
        minute = re.search(r'(\d+)\s*(?:分钟|分)', text)
        second = re.search(r'(\d+)\s*秒', text)
        # 倒计时常显示“23分30秒”，中文 OCR 偶尔只漏掉最后一个“秒”。
        # 已有“分”单位时，允许把末尾独立数字作为秒数恢复。
        if second is None and minute is not None:
            second = re.search(r'(\d+)$', text[minute.end():])
        if not hour and not minute and not second:
            return None
        hours = int(hour.group(1)) if hour else 0
        minutes = int(minute.group(1)) if minute else 0
        seconds = int(second.group(1)) if second else 0
        if (
            hours > 240 or minutes > 60 or seconds > 59
            or (hours == 0 and minutes == 0 and seconds == 0)
        ):
            return None
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def read_transport_duration(self):
        """在离开物资页之前直接读取一次运输时长。"""
        self.device.screenshot()
        area = self.anchored_area(
            ALLOCATION_RESOURCE_CHECK,
            AllocationDurationOcr.DEFAULT_AREA,
        )
        digit_areas = [
            self.anchored_area(ALLOCATION_RESOURCE_CHECK, source_area)
            for source_area in (
                AllocationDurationOcr.FIRST_NUMBER_AREA,
                AllocationDurationOcr.SECOND_NUMBER_AREA,
            )
        ]
        buttons = [
            Button(area=area, color=(), button=area,
                   name='ALLOCATION_DURATION_TEXT'),
            *[
                Button(area=item, color=(), button=item,
                       name=f'ALLOCATION_DURATION_DIGIT_{index + 1}')
                for index, item in enumerate(digit_areas)
            ],
        ]
        values = Ocr(
            buttons, lang='cnocr', name='ALLOCATION_DURATION_BATCH'
        ).ocr(self.device.image)
        if not isinstance(values, list):
            values = [values]
        values.extend([''] * (3 - len(values)))
        raw = values[0]
        parsed = self.parse_transport_duration(raw)
        logger.attr('AllocationDurationText', raw)
        digit_values = []
        for value in values[1:3]:
            digits = ''.join(re.findall(r'\d+', str(value)))
            number = int(digits) if digits else None
            digit_values.append(number if number and number > 0 else None)
        logger.attr('AllocationDurationDigitValues', digit_values)

        # 中文 OCR 可能只读到“54分钟”而漏掉前面的“9小时”。当第二段
        # 数字与已识别分钟一致时，用第一段补回小时。分秒布局的窄区域
        # 可能读到截断的分钟和完整秒数（“30分30秒” -> [3, 30]），
        # 不能据此补小时；末尾漏掉“秒”的数字也属于分秒布局。
        if parsed is not None:
            duration = parsed
            first, second = digit_values
            if (
                duration < timedelta(hours=1)
                and not re.search(r'秒|分(?:钟)?\s*\d', str(raw))
                and first is not None
                and second is not None
                and second == int(duration.total_seconds() // 60)
                and 0 < first <= 240
            ):
                duration += timedelta(hours=first)
            logger.attr('AllocationDurationConfirmed', duration)
            return duration

        first, second = digit_values
        if (
            first is not None and second is not None
            and 0 < first <= 240 and 0 <= second <= 60
        ):
            duration = timedelta(hours=first, minutes=second)
            logger.attr('AllocationDurationConfirmed', duration)
            return duration

        # A whole-hour trip renders no minute digits at all (for example,
        # 扬州府 -> 绍兴府 shows only "1小时").  OCR separates the narrow "1"
        # from the unit, so require the visible hour unit before accepting it.
        if (
            first is not None and second is None
            and 0 < first <= 240
            and re.search(r'小时|时', str(raw))
            and not re.search(r'分钟|分|秒', str(raw))
        ):
            duration = timedelta(hours=first)
            logger.attr('AllocationDurationConfirmed', duration)
            return duration

        logger.warning('Unable to recognize transport duration before departure')
        return None

    # ========================================================
    # Public entry
    # ========================================================

    @staticmethod
    def city_name(value):
        value = str(value).strip()
        return CITY_KEY_TO_NAME.get(value, value)

    def transport(self, source, destination, resource, amount):
        self.last_error = None
        self.last_amount = 0
        self.last_duration = None
        self.last_slot_index = None
        self.last_reconciled_inventory = None
        self.last_reconciled_source_inventory = None
        self.last_source_stock_lower_bound = False
        self.last_destination_in_transit = None
        self._source_stock_hint = None
        source = self.city_name(source)
        destination = self.city_name(destination)
        resource = self.normalize_resource_name(resource)
        requested_amount = self.normalize_amount(amount)

        logger.hr(
            f'JIANGNAN ALLOCATION: {source} -> {destination}, '
            f'{resource} x{amount}',
            level=0,
        )

        if source not in KNOWN_CITIES or destination not in KNOWN_CITIES:
            logger.warning('Unknown source or destination city')
            return False
        if source == destination:
            logger.warning('Source and destination cities must be different')
            return False
        if requested_amount is None:
            logger.warning('Allocation amount must be a positive integer')
            return False

        if not self.open_transport(source):
            return False
        if not self.choose_destination(destination):
            return False
        # 容量文字显示的是“当前选中物资”的可接收量。进入页面时默认是
        # 蔬菜，必须先切换到目标物资，否则运输丝绸时读到的会是蔬菜
        # 容量，造成错误限量或误报仓库已满。
        layout_page = allocation_layout_page(source)
        if not self.select_resource(resource, source=source):
            if self.last_error == 'resource_identity_mismatch':
                self.save_abnormal_screenshot('allocation_resource_identity_mismatch')
            return False
        capacity = self.read_destination_capacity()
        if capacity is None:
            logger.warning('Unable to read destination warehouse capacity')
            self.last_error = 'capacity_unknown'
            return False
        capacity = self.verify_selected_resource(
            destination,
            resource,
            capacity,
            source=source,
        )
        if capacity is None:
            logger.warning('Unable to verify selected allocation resource')
            self.last_error = 'resource_mismatch'
            return False
        in_transit = self.read_destination_in_transit()
        if in_transit is None:
            # 居民关系等短暂通知会遮住左上角的在途数字。等它消失后
            # 在同一调配页重读，避免整笔订单因一次遮挡而失败。
            for attempt in range(3):
                self.device.sleep(1.5)
                self.device.screenshot()
                if not self.is_resource_page():
                    break
                logger.info(
                    f'Retry allocation in-transit reading after overlay: '
                    f'{attempt + 1}/3'
                )
                in_transit = self.read_destination_in_transit()
                if in_transit is not None:
                    break
        if in_transit is None:
            logger.warning('Unable to read in-transit goods for destination')
            self.last_error = 'in_transit_unknown'
            self.save_abnormal_screenshot('allocation_in_transit_unknown')
            return False
        self.last_destination_in_transit = in_transit
        # The first line is current free space, while the second line is
        # goods already on the way. Existing shipments are not current stock,
        # but they must reserve room before another shipment departs.
        available_on_arrival = max(0, capacity - in_transit)
        logger.attr('AllocationDestinationFreeAfterTransit', available_on_arrival)
        # 只有库存容量变化确认“点中的确实是目标货物”后，识别到的
        # 行列才有资格参与辅助信息学习。旧逻辑在这里之前记数，导致
        # 年历等相似图标误匹配时把错误位置写入学习状态。
        if (self.last_resource_grid is not None
                and resource not in AUTOMATIC_EXPORT_RESOURCES.get(
                    source, (),
                )):
            self.resource_layout.confirm_position(
                layout_page,
                resource,
                self.last_resource_grid,
            )
        if available_on_arrival <= 0:
            logger.warning('Destination warehouse has no remaining capacity')
            self.last_error = 'destination_full'
            return False

        logger.info('Choose direct start without assignee')

        city_key = next(
            (key for key, value in CITY_KEY_TO_NAME.items() if value == source),
            None,
        )
        if city_key is not None:
            inventory_text = self.config.cross_get(
                keys=(f'JiangnanWarehouseInventory.'
                      f'warehouse_{city_key}.Inventory'),
                default='',
            )
            previous_stock = self._parse_inventory_text(
                inventory_text
            ).get(resource, -1)
            if previous_stock is not None and int(previous_stock) >= 0:
                self._source_stock_hint = int(previous_stock)

        # Both values must be read from the selected resource page before
        # departure. The source number is a live stock baseline, not merely
        # an editor hint; record it before the shipment is deducted.
        self.device.screenshot()
        source_stock = self._read_selected_source_stock()
        if source_stock is not None and city_key is not None:
            try:
                source_capacity = int(self.config.cross_get(
                    keys=f'JiangnanWarehouseInventory.warehouse_{city_key}.Capacity', default=0,
                ) or 0)
            except (TypeError, ValueError):
                source_capacity = 0
            if source_capacity > 0 and source_stock > source_capacity:
                x, y = self.last_resource_screen
                image_height, image_width = self.device.image.shape[:2]
                area = (max(0, int(x) - 31), max(0, int(y) + 20),
                        min(image_width, int(x) + 48), min(image_height, int(y) + 42))
                first_stock = source_stock
                source_stock = read_confirmed_digits(self.device.image, area)
                self._source_stock_is_lower_bound = False
                logger.info(f'Allocation over-capacity stock verification: '
                            f'first={first_stock}, confirmed={source_stock}')
        self.last_source_stock_lower_bound = bool(getattr(
            self, '_source_stock_is_lower_bound', False,
        ))
        if source_stock is None:
            logger.warning('Unable to read selected source resource stock')
            self.last_error = 'source_stock_unknown'
            self.save_abnormal_screenshot('allocation_source_stock_unknown')
            return False
        if source_stock <= 0:
            self.save_abnormal_screenshot('allocation_source_empty')
            self._reconcile_source_inventory(source, resource, 0)
            logger.warning('Selected source resource has no stock')
            self.last_error = 'source_empty'
            return False

        reserve = max(0, int(getattr(self, 'source_reserve', 0)))
        recalculate = getattr(self, 'recalculate_live_amount', None)
        live_amount = None
        if callable(recalculate):
            live_amount, reserve = recalculate(source_stock, capacity, in_transit)
            reserve = max(0, int(reserve))
        exportable = max(0, source_stock - reserve)
        if exportable <= 0:
            # No amount is entered on this path, so confirm the tile with
            # independent OCR before replacing the stale source baseline.
            # A conservative availability bound is not an exact stock count.
            if not self.last_source_stock_lower_bound:
                x, y = self.last_resource_screen
                height, width = self.device.image.shape[:2]
                area = (max(0, int(x) - 31), max(0, int(y) + 20),
                        min(width, int(x) + 48), min(height, int(y) + 42))
                if self._confirm_selected_stock(area, (source_stock,)) is not None:
                    self._reconcile_source_inventory(source, resource, source_stock)
            logger.warning(
                f'Refuse export below source reserve: {source} {resource}, '
                f'live_stock={source_stock}, reserve={reserve}'
            )
            self.last_error = 'source_reserve'
            return False
        actual_amount = min(requested_amount if live_amount is None else max(0, int(live_amount)),
                            available_on_arrival, exportable)
        if actual_amount <= 0:
            self.last_error = 'destination_satisfied'
            return False
        if actual_amount != requested_amount:
            logger.warning(
                f'Allocation amount recalculated from live stock/capacity: '
                f'requested={requested_amount}, source={source_stock}, '
                f'destination_free_after_transit={available_on_arrival}, '
                f'actual={actual_amount}'
            )
        confirmed_amount = self.input_amount(actual_amount)
        if not confirmed_amount:
            if self.last_error == 'source_empty':
                self._reconcile_source_inventory(source, resource, 0)
            return False
        actual_amount = confirmed_amount
        duration = self.read_transport_duration()
        if duration is None:
            self.last_error = 'duration_unknown'
            return False
        self._base_transport_duration = duration
        # An OCR candidate alone must not overwrite stock: a disabled amount
        # editor is evidence that a spurious icon glyph was read as a digit.
        # Reconcile only after the entered amount is visible and confirmed.
        if self.last_source_stock_lower_bound:
            logger.info(
                f'Source stock {source_stock} is a lower bound; '
                'defer exact inventory reconciliation to warehouse scan'
            )
        else:
            self._reconcile_source_inventory(source, resource, source_stock)
        config_reader = getattr(getattr(self, 'config', None), 'cross_get', None)
        use_special = (config_reader(
            keys='JiangnanCityAllocation.AllocationSettings.UseSpecialTask',
            default=True,
        ) if callable(config_reader) else True)
        result = self.depart(use_special=bool(use_special), source=source)
        if result:
            self.last_amount = actual_amount
            self.last_duration = (
                getattr(self, '_selected_transport_duration', None) or duration
            )
        return result
