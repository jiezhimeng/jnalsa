import re
import subprocess
import time

import cv2
import numpy as np

from module.base.button import Button
from module.base.utils import crop, load_image
from module.jiangnan.assets import (
    BUILDING_FIRST_RESULT,
    BUILDING_SEARCH_BUTTON,
    BUILDING_SEARCH_CLEAR_BUTTON,
    BUILDING_SEARCH_INPUT,
    BUILDING_SEARCH_PAGE_CHECK,
    BUILDING_SEARCH_TOOL_BUTTON,
    CITY_PANEL_CHECK,
    TAX_MAP_ICON,
    TAX_IDLE_CHECK,
    TAX_WORKING_CHECK,
    TAX_LEVEL_LOCK_CHECK,
    TAX_LEVEL_LOCK_AREAS,
    TAX_LEVEL_BUTTONS,
    TAX_REMAINING_TIME,
    BUILDING_SEARCH_NO_RESULT,
    BUILDING_INTERFERENCE_KNOT,
    BUILDING_INTERFERENCE_SCISSORS,
    RESIDENCE_PANEL_CHECK,
)
from module.jiangnan.city import JiangnanCity
from module.logger import logger
from module.jiangnan.ocr import Ocr
from module.jiangnan.vision import (
    crop_map_icon_foreground,
    stable_feature_map,
    stable_feature_match,
)
from datetime import datetime


class BuildingResultOcr(Ocr):
    """识别搜索结果第一张建筑卡片下方的建筑名。"""

    def __init__(self):
        super().__init__(
            buttons=Button(
                area=(329, 283, 470, 317),
                color=(),
                button=(329, 283, 470, 317),
                name='BUILDING_RESULT_OCR',
            ),
            lang='cnocr',
            name='BUILDING_RESULT_OCR',
        )

    def pre_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.inRange(gray, 0, 145)
        return cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


class BuildingSearchInputOcr(Ocr):
    """确认搜索框中实际提交的建筑名，防止继续读取旧结果列表。"""

    def __init__(self):
        super().__init__(
            buttons=Button(
                # 排除右侧清空叉号；PP-OCRv6 直接读取原图即可。
                area=(660, 72, 932, 107),
                color=(),
                button=(660, 72, 932, 107),
                name='BUILDING_SEARCH_INPUT_OCR',
            ),
            lang='cnocr',
            name='BUILDING_SEARCH_INPUT_OCR',
        )

class TaxRemainingTimeOcr(Ocr):
    """识别税课司工作中的剩余时间，例如：3小时39分44秒。"""

    def __init__(self):
        super().__init__(
            buttons=TAX_REMAINING_TIME,
            lang='cnocr',
            name='TAX_REMAINING_TIME_OCR',
        )

    def pre_process(self, image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # 保留较深的文字，去掉浅色纸张背景。
        binary = cv2.inRange(
            gray,
            0,
            150,
        )

        binary = cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )

        return binary

class JiangnanBuilding(JiangnanCity):
    """在城市主界面按建筑名称搜索并将建筑移动到屏幕中心。"""

    SEARCH_FOUND = 'found'
    SEARCH_NOT_FOUND = 'not_found'
    SEARCH_OPEN_FAILED = 'open_failed'
    SEARCH_INPUT_FAILED = 'input_failed'
    SEARCH_PAGE_LOST = 'page_lost'
    SEARCH_BUTTON_MISSING = 'button_missing'
    SEARCH_RESULT_MISMATCH = 'result_mismatch'
    SEARCH_CENTER_TIMEOUT = 'center_timeout'

    SEARCH_TOOL_AREA = (0, 180, 110, 560)
    SEARCH_ARROW_AREA = (300, 90, 950, 390)
    SEARCH_ARROW_FIXED_ANCHOR = (640, 265, 'down')
    SEARCH_TARGET_MAX_Y = 590
    # 建筑搜索页只要存在任意筛选，漏斗左侧都会出现红色圆形 X。
    # 所有直接按名称搜索建筑的入口都必须先回到“未筛选”状态，
    # 否则例如搜索“码头”会只在上一次保留的生产建筑筛选中查询。
    BUILDING_FILTER_CANCEL = (845, 620, 886, 661)


    def __init__(self, config, device=None, task=None):
        super().__init__(config=config, device=device, task=task)

        self.result_ocr = BuildingResultOcr()
        self.search_input_ocr = BuildingSearchInputOcr()
        self.tax_remaining_ocr = TaxRemainingTimeOcr()

        self._search_tool_template = load_image(
            BUILDING_SEARCH_TOOL_BUTTON.file,
            BUILDING_SEARCH_TOOL_BUTTON.area,
        )
        self._foundation_selected_template = load_image(
            './assets/jiangnan/building/foundation.png',
            (299, 269, 340, 375),
        )
        self._foundation_unselected_template = load_image(
            './assets/jiangnan/building/foundation_unselected.png',
        )
        self.last_foundation_option = None
        knot_template = load_image(
            BUILDING_INTERFERENCE_KNOT.file,
            BUILDING_INTERFERENCE_KNOT.area,
        )
        scissors_template = load_image(
            BUILDING_INTERFERENCE_SCISSORS.file,
            BUILDING_INTERFERENCE_SCISSORS.area,
        )
        self._map_interference_templates = (
            (
                '中国结', knot_template,
                self._interference_color_mask('中国结', knot_template),
                300, 0.60,
            ),
            (
                '剪刀', scissors_template,
                self._interference_color_mask('剪刀', scissors_template),
                160, 0.62,
            ),
        )

        self.last_tax_action = None
        self.last_tax_level = None
        self.last_tax_remaining = None
        self.last_tax_remaining_at = None
        self.last_tax_started_at = None
        self.last_search_result_name = ''
        self.last_search_status = 'unknown'
        self.last_search_executed = False
        self.last_interference_attempted = False
        self.last_search_target_arrow_fallback = False
        self.last_search_arrow_size = None
        self._retained_search_arrow = None
        self._retained_search_arrow_size = None
        self._retained_search_frame = None

    def find_building_search_tool(self, similarity=0.55):
        """在整条左侧工具栏中识别搜索图标并返回实时点击区域。"""
        template = self._search_tool_template
        # Matches outside the left toolbar were rejected anyway. Restrict the
        # search to valid positions, including the full template at x=150.
        search = self.device.image[:, :150 + template.shape[1]]
        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None

        score, location, gray_score, edge_score = stable_feature_match(
            search, template, gray_weight=0.30
        )
        logger.attr(
            'BuildingSearchToolScore',
            f'stable={score:.3f}, gray={gray_score:.3f}, '
            f'edge={edge_score:.3f}',
        )
        if score < similarity:
            return None

        height, width = template.shape[:2]
        left, top = location
        # Vertical position remains dynamic across city toolbar layouts.
        if left > 150:
            logger.warning(
                f'Building search tool match outside toolbar: ({left}, {top})'
            )
            return None
        logger.info(
            f'Building search tool found: score={score:.3f}, '
            f'center=({left + width // 2}, {top + height // 2})'
        )
        return Button(
            area=(),
            color=(),
            button=(left, top, left + width, top + height),
            name='BUILDING_SEARCH_TOOL_DYNAMIC',
        )

    def is_building_search_page(self):
        return self.appear(
            BUILDING_SEARCH_PAGE_CHECK,
            offset=(3, 3),
            similarity=0.88,
        )

    def current_navigation_page(self):
        # 搜索遮罩后方仍有主界面按钮，必须先识别搜索页。
        if self.is_building_search_page():
            self.log_check('JiangnanNavigationPage', 'building_search')
            return 'building_search'
        return super().current_navigation_page()

    def building_search_filter_active(self, log_name='BuildingFilterState'):
        """判断建筑搜索页是否保留了任意筛选条件。"""
        x1, y1, x2, y2 = self.BUILDING_FILTER_CANCEL
        image = self.device.image
        if image is None or image.size == 0:
            return False
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 85, 70), (15, 255, 255)),
            cv2.inRange(hsv, (165, 85, 70), (179, 255, 255)),
        )
        pixels = int(cv2.countNonZero(red))
        active = pixels >= 120
        logger.attr(log_name, f'active={active}, red_pixels={pixels}')
        return active

    def ensure_building_search_unfiltered(
        self, timeout=5, log_name='BuildingFilterState'
    ):
        """搜索建筑前强制恢复成无筛选状态。"""
        self.device.screenshot()
        if not self.is_building_search_page():
            logger.warning('Cannot clear building filter outside search page')
            return False
        if not self.building_search_filter_active(log_name=log_name):
            return True

        logger.info('Cancel retained building search filter before search')
        x1, y1, x2, y2 = self.BUILDING_FILTER_CANCEL
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        self.device.click(Button(
            area=(), color=(),
            button=(center_x - 5, center_y - 5,
                    center_x + 5, center_y + 5),
            name='BUILDING_FILTER_CANCEL_DYNAMIC',
        ))
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if not self.is_building_search_page():
                logger.warning('Building search page lost while clearing filter')
                return False
            if not self.building_search_filter_active(log_name=log_name):
                logger.info('Building search filter cleared')
                return True
        logger.warning('Building search filter was not cancelled')
        return False

    @staticmethod
    def _map_view_unchanged(previous, current):
        """Check the map itself, allowing small animated or cleared overlays."""
        if previous is None or current is None or previous.shape != current.shape:
            return False
        old = cv2.cvtColor(previous[115:560, 180:1080], cv2.COLOR_RGB2GRAY)
        new = cv2.cvtColor(current[115:560, 180:1080], cv2.COLOR_RGB2GRAY)
        if old.size == 0:
            return False
        unchanged = float((cv2.absdiff(old, new) < 14).mean())
        old_edges = cv2.Canny(old, 60, 130)
        new_edges = cv2.Canny(new, 60, 130)
        union = cv2.bitwise_or(old_edges, new_edges)
        overlap = cv2.bitwise_and(old_edges, new_edges)
        edge_count = cv2.countNonZero(union)
        edge_agreement = (
            cv2.countNonZero(overlap) / edge_count if edge_count >= 300 else 0
        )
        logger.attr(
            'BuildingMapStationary',
            f'pixels={unchanged:.3f}, edges={edge_agreement:.3f}',
        )
        # 人物走动、旗帜消失或生产动画会改变局部边缘。大量像素仍逐点
        # 相同且边缘仍有明显重合时，镜头没有移动；日志里的同一画面
        # 曾达到 pixels=0.815 / edges=0.467，却被旧门槛误判为移动。
        return (
            unchanged >= 0.70 and edge_agreement >= 0.50
        ) or (
            unchanged >= 0.80 and edge_agreement >= 0.40
        )

    def find_building_search_arrow(
        self, allow_retained=True, report_missing=True
    ):
        """识别短时箭头；消失后仅在地图未移动时复用已确认锚点。"""
        image = self.device.image
        x1 = y1 = 0
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 河水与箭头色相接近。宽松阈值会把箭头和整片河水连成一个
        # 巨大轮廓，导致画面上明明有箭头却识别失败。
        mask = cv2.inRange(hsv, (75, 85, 145), (95, 255, 255))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        arrows = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (28 <= width <= 95 and 55 <= height <= 170):
                continue
            cx = x1 + left + width // 2
            cy = y1 + top + height // 2
            # 左侧等级/功能栏和最右任务栏也有相似青色轮廓；它们不可能
            # 是建筑搜索箭头。仍扫描整屏，只剔除明确属于 UI 边栏的候选。
            if cx < 90 or cx > image.shape[1] - 90:
                continue
            # 游戏搜索箭头方向恒定向下，包围框底边就是箭头尖端。
            tip_y = top + height
            if tip_y > self.SEARCH_TARGET_MAX_Y + 40:
                continue
            arrows.append((
                (cx - image.shape[1] // 2) ** 2
                + (cy - image.shape[0] * 0.35) ** 2,
                cx, tip_y, width, height,
            ))
        if not arrows:
            if (allow_retained
                    and getattr(self, '_retained_search_arrow', None) is not None
                    and self._map_view_unchanged(
                        getattr(self, '_retained_search_frame', None), image
                    )):
                self.last_search_arrow = self._retained_search_arrow
                self.last_search_arrow_size = self._retained_search_arrow_size
                logger.info(
                    'Reuse building search arrow position on unchanged map: '
                    f'{self.last_search_arrow}'
                )
                return self.last_search_arrow
            self._retained_search_arrow = None
            self._retained_search_arrow_size = None
            self._retained_search_frame = None
            self.last_search_arrow = None
            self.last_search_arrow_size = None
            if report_missing:
                logger.warning('Building selection arrow was not recognized')
                self.save_abnormal_screenshot('building_arrow_missing')
            return None
        _, arrow_x, arrow_tip, width, height = min(arrows)
        self.last_search_arrow = (
            int(arrow_x), int(arrow_tip), 'down'
        )
        self.last_search_arrow_size = (int(width), int(height))
        logger.info(
            f'Building search arrow found: axis={arrow_x}, tip={arrow_tip}, '
            f'direction=down, size={width}x{height}'
        )
        return self.last_search_arrow

    def _remember_centered_search_arrow(self):
        """Capture the short-lived arrow immediately after search centering."""
        arrow = self.find_building_search_arrow(
            allow_retained=False, report_missing=False
        )
        if arrow is None:
            arrow = self.SEARCH_ARROW_FIXED_ANCHOR
            size = (53, 120)
            logger.info(
                f'Search arrow already expired; use confirmed screen anchor '
                f'{arrow}'
            )
        else:
            size = self.last_search_arrow_size
        self._retained_search_arrow = arrow
        self._retained_search_arrow_size = size
        self._retained_search_frame = self.device.image.copy()

    def find_centered_search_target(self):
        """沿搜索后的青色箭头定位建筑主体，并裁掉周围地面。

        建筑模板会受城市地面、季节滤镜和等级外观影响；搜索箭头本身
        却始终稳定。先找箭头，再只在尖端下方的小区域寻找与箭头同轴
        的建筑前景轮廓，可供税课司、议事堂等所有特殊建筑复用。
        """
        self.last_search_target_arrow_fallback = False
        self.last_search_target_score = None
        arrow = self.find_building_search_arrow()
        if arrow is None:
            return None
        arrow_x, arrow_bottom, _ = arrow
        arrow_width, arrow_height = self.last_search_arrow_size or (0, 0)
        # 特殊建筑可复用箭头本身的稳定锚点，而不必接受通用前景轮廓
        # 的最终候选。议事堂的大块浅色占地尤其需要这个信息。

        image_height, image_width = self.device.image.shape[:2]
        left = max(0, arrow_x - 115)
        right = min(image_width, arrow_x + 115)
        top = min(self.SEARCH_TARGET_MAX_Y - 20, arrow_bottom + 6)
        # 搜索箭头只指向紧邻尖端的建筑占地。原来的 220px 范围会在
        # 议事堂这类浅色大地块上抓不到主体后，误把两百像素外的道路
        # 或其他地面小图标当成目标（日志中 arrow=278, target=482）。
        # 限定到箭头下方 130px；无可靠前景时使用箭头轴上的安全点。
        bottom = min(self.SEARCH_TARGET_MAX_Y, arrow_bottom + 130)
        if right <= left or bottom <= top:
            logger.warning(
                f'Invalid building target ROI from arrow: '
                f'arrow=({arrow_x}, {arrow_bottom}), roi=({left}, {top}, {right}, {bottom})'
            )
            return None
        search = self.device.image[top:bottom, left:right]
        if search.size == 0:
            logger.warning('Building target ROI is empty')
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 45, 130)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            item_left, item_top, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cx = left + item_left + width // 2
            cy = top + item_top + height // 2
            if not (
                10 <= width <= 170 and 7 <= height <= 140
                and area >= 35 and abs(cx - arrow_x) <= 64
                and 6 <= cy - arrow_bottom <= 118
            ):
                continue
            # 地面边线通常又宽又扁；真正建筑在箭头轴附近且靠近尖端。
            if width >= height * 4 and height <= 12:
                continue
            score = (
                area * 3
                - abs(cx - arrow_x) * 35
                - (cy - arrow_bottom) * 12
            )
            candidates.append((score, area, cx, cy, width, height))
        if not candidates:
            # 箭头尖端本身已经给出可靠横坐标。建筑轮廓被完成旗覆盖时，
            # 点击尖端下方 42px 的狭小区域，仍比全屏模板猜测安全。
            target_x, target_y = arrow_x, min(bottom - 8, arrow_bottom + 42)
            self.last_search_target_arrow_fallback = True
            logger.warning(
                'Building foreground below selection arrow was not recognized; '
                f'use arrow fallback=({target_x}, {target_y})'
            )
            self.save_abnormal_screenshot('building_foreground_missing')
        else:
            score, area, target_x, target_y, width, height = max(candidates)
            self.last_search_target_score = score
            logger.info(
                f'Centered building foreground: arrow=({arrow_x}, '
                f'{arrow_bottom}, {arrow_width}x{arrow_height}), '
                f'target=({target_x}, {target_y}), size={width}x{height}, '
                f'contour={area:.0f}, score={score:.0f}'
            )
        return Button(
            area=(), color=(),
            button=(target_x - 7, target_y - 7,
                    target_x + 7, target_y + 7),
            name='CENTERED_SEARCH_BUILDING_DYNAMIC',
        )

    def find_still_visible_map_target(self, previous_image, target):
        """Relocate a selected building icon after its search arrow disappears."""
        if previous_image is None or self.device.image is None:
            return None
        x1, y1, x2, y2 = target.button
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius, search_radius = 25, 35
        height, width = self.device.image.shape[:2]
        if (cx < radius + search_radius or cy < radius + search_radius
                or cx + radius + search_radius >= width
                or cy + radius + search_radius >= height):
            return None
        template = previous_image[
            cy - radius:cy + radius, cx - radius:cx + radius
        ]
        gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        # A blank foundation tile matches itself perfectly but is not a
        # building.  Require an actual detailed icon in the reference patch.
        if gray.std() < 20 or cv2.Canny(gray, 45, 130).mean() < 20:
            return None
        search = self.device.image[
            cy - radius - search_radius:cy + radius + search_radius,
            cx - radius - search_radius:cx + radius + search_radius,
        ]
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(
            search, template, cv2.TM_CCOEFF_NORMED
        ))
        if score < 0.90:
            return None
        matched_x = cx - search_radius + location[0]
        matched_y = cy - search_radius + location[1]
        logger.info(
            f'Search arrow vanished but building icon remains: '
            f'center=({matched_x}, {matched_y}), score={score:.3f}'
        )
        return Button(
            area=(), color=(),
            button=(matched_x - 7, matched_y - 7,
                    matched_x + 7, matched_y + 7),
            name='STILL_VISIBLE_BUILDING_ICON_DYNAMIC',
        )

    # 搜索结果居中后，居民事件的中国结/剪刀竖旗可能遮挡建筑图标。
    # 位置随城市布局变化，不能使用固定坐标。
    MAP_INTERFERENCE_AREA = (80, 20, 1180, 620)
    RESIDENT_RED_DOT_AREA = (620, 140, 1080, 490)

    @staticmethod
    def _interference_color_mask(kind, image):
        """生成居民事件模板/候选共用的颜色掩码。"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if kind == '中国结':
            return cv2.bitwise_or(
                cv2.inRange(hsv, (0, 90, 50), (12, 255, 240)),
                cv2.inRange(hsv, (170, 90, 50), (179, 255, 240)),
            )
        return cv2.inRange(hsv, (25, 60, 35), (90, 255, 220))

    def _find_map_interference_marker(self, excluded=()):
        """动态查找一个尚未处理的中国结或剪刀居民事件标记。"""
        x1, y1, _, _ = self.MAP_INTERFERENCE_AREA
        search = crop(
            self.device.image, self.MAP_INTERFERENCE_AREA, copy=False
        )
        matches = []
        for (
            kind, template, template_color, minimum_pixels, similarity
        ) in self._map_interference_templates:
            if (
                search.shape[0] < template.shape[0]
                or search.shape[1] < template.shape[1]
            ):
                continue
            score_map, gray_map, edge_map = stable_feature_map(
                search, template, gray_weight=0.30
            )
            height, width = template.shape[:2]
            # 不能只验证全屏最高分候选：完成牌、工具按钮等假目标可能
            # 分数更高但颜色不符，从而遮住第二高分的真实剪刀。依次
            # 验证多组局部峰值，每次都屏蔽已检查邻域。
            for _ in range(24):
                _, score, _, location = cv2.minMaxLoc(score_map)
                if score < similarity:
                    break
                left, top = location
                center = (
                    x1 + left + width // 2,
                    y1 + top + height // 2,
                )
                sx1 = max(0, left - width // 2)
                sy1 = max(0, top - height // 2)
                sx2 = min(score_map.shape[1], left + width + width // 2)
                sy2 = min(score_map.shape[0], top + height + height // 2)
                score_map[sy1:sy2, sx1:sx2] = -1
                if any(
                    abs(center[0] - old[0]) <= 36
                    and abs(center[1] - old[1]) <= 44
                    for old in excluded
                ):
                    continue
                patch = search[top:top + height, left:left + width]
                patch_color = self._interference_color_mask(kind, patch)
                color_pixels = cv2.countNonZero(patch_color)
                color_score = float(cv2.matchTemplate(
                    patch_color, template_color, cv2.TM_CCOEFF_NORMED
                )[0, 0])
                # 灰度竖旗、完成牌和左侧工具按钮的轮廓可能相似，但不会
                # 同时具有中国结的红色纹路或剪刀的绿色纹路。
                if color_pixels < minimum_pixels or color_score < 0.68:
                    logger.info(
                        f'Reject interference-like feature: {kind}, '
                        f'score={score:.3f}, color_pixels={color_pixels}, '
                        f'color_score={color_score:.3f}, center={center}'
                    )
                    continue
                gray_score = float(gray_map[top, left])
                edge_score = float(edge_map[top, left])
                matches.append((
                    float(score), kind, center, gray_score, edge_score,
                    color_score,
                ))

        if not matches:
            return None
        score, kind, (cx, cy), gray_score, edge_score, color_score = max(matches)
        logger.info(
            f'Building interference found: {kind}, score={score:.3f}, '
            f'gray={gray_score:.3f}, edge={edge_score:.3f}, '
            f'color={color_score:.3f}, center=({cx}, {cy})'
        )
        return kind, Button(
            area=(), color=(),
            button=(cx - 8, cy - 8, cx + 8, cy + 8),
            name=f'BUILDING_INTERFERENCE_{kind}_DYNAMIC',
        ), (cx, cy)

    def _is_residence_panel(self):
        return self.appear(
            RESIDENCE_PANEL_CHECK, offset=(4, 4), similarity=0.82
        )

    def _wait_residence_panel(self, present, timeout=8):
        deadline = time.time() + timeout
        stable = 0
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            matched = self._is_residence_panel()
            stable = stable + 1 if matched == present else 0
            if stable >= 2:
                return True
        return False

    def _find_resident_red_dots(self):
        """识别居民头像右上角的小圆红点，并返回对应头像点击区域。"""
        x1, y1, _, _ = self.RESIDENT_RED_DOT_AREA
        image = crop(
            self.device.image, self.RESIDENT_RED_DOT_AREA, copy=False
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 100, 65), (10, 255, 225)),
            cv2.inRange(hsv, (172, 100, 65), (179, 255, 225)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        dots = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * 3.141592653589793 * area / (perimeter * perimeter)
                if perimeter > 0 else 0
            )
            if not (
                9 <= width <= 18
                and 8 <= height <= 17
                and 55 <= area <= 180
                and circularity >= 0.45
            ):
                continue
            cx = x1 + left + width // 2
            cy = y1 + top + height // 2
            # 红点本身位于卡片右上角，点击其左下方的头像内部，避免
            # 点到红点边缘或相邻姓名栏。
            dots.append((cy, cx, Button(
                area=(), color=(),
                button=(cx - 44, cy + 34, cx - 25, cy + 58),
                name='RESIDENT_RED_DOT_AVATAR_DYNAMIC',
            )))
        dots.sort()
        logger.attr('ResidentRedDots', len(dots))
        return [item[2] for item in dots]

    def _handle_residence_interference(self, marker_kind, timeout=4):
        if not self._wait_residence_panel(True, timeout=timeout):
            logger.warning(
                f'{marker_kind} marker did not open residence panel'
            )
            return False

        handled = 0
        for _ in range(12):
            dots = self._find_resident_red_dots()
            if not dots:
                # 连续两帧确认红点清零，避免详情返回动画期间漏检。
                self.device.sleep(0.35)
                self.device.screenshot()
                if self._find_resident_red_dots():
                    continue
                logger.info(
                    f'{marker_kind} residence red dots cleared: {handled}'
                )
                self.device.press_back()
                self._wait_residence_panel(False, timeout=8)
                return True

            self.device.click(dots[0])
            if not self._wait_residence_panel(False, timeout=6):
                logger.warning('Resident detail page did not open')
                return False
            handled += 1
            self.device.press_back()
            if not self._wait_residence_panel(True, timeout=8):
                logger.warning('Residence panel did not return from detail')
                return False

        logger.warning(f'Too many resident red dots under {marker_kind}')
        return False

    def handle_open_residence_panel(self):
        """从任务切换、超时或重启后恢复一个已经打开的民宅面板。"""
        if not self._is_residence_panel():
            return False
        logger.info('Recover opened residence interference panel')
        handled = self._handle_residence_interference('民宅', timeout=2)
        if not handled:
            logger.warning('Residence panel recovery will retry next round')
        # 页面已被明确识别，即使本轮处理失败也必须声明“已处理”，
        # 阻止后续税课司/主界面逻辑在这个模态面板上继续盲点。
        return True

    def handle_building_map_interference(self, max_markers=12):
        """清理当前地图上会遮挡目标建筑的居民事件标记。"""
        self.last_interference_attempted = False
        self._cleanup_selection_lost = False
        excluded = list(getattr(self, '_cleanup_failed_marker_centers', ()))
        handled = 0
        for _ in range(max_markers):
            self.device.screenshot()
            found = self._find_map_interference_marker(excluded)
            if found is None:
                break
            kind, button, center = found
            # 模板只截取中国结/剪刀本身，但游戏的响应区域经常落在
            # 竖旗下半部或旗杆对应的建筑上。旧逻辑始终点击图案中心，
            # 日志中会稳定识别却连续取消搜索箭头，民宅面板一次也没
            # 打开。以识别中心为锚点依次尝试旗面下部、旗杆和图案，
            # 每次都以民宅面板真正出现为成功条件。
            cx, cy = center
            attempted_offsets = 0
            click_offsets = (
                (0, 0),
                (0, 40),
                (0, 62),
                (-18, 52),
                (18, 52),
            )
            opened = False
            for attempt, (dx, dy) in enumerate(click_offsets, start=1):
                attempted_offsets = attempt
                target_x = max(12, min(1268, cx + dx))
                target_y = max(12, min(708, cy + dy))
                logger.info(
                    f'Try opening {kind} residence marker: '
                    f'attempt={attempt}/{len(click_offsets)}, '
                    f'offset=({dx}, {dy}), '
                    f'target=({target_x}, {target_y})'
                )
                self.device.click(Button(
                    area=(), color=(),
                    button=(
                        target_x - 7, target_y - 7,
                        target_x + 7, target_y + 7,
                    ),
                    name=(
                        f'BUILDING_INTERFERENCE_{kind}_'
                        f'ATTEMPT_{attempt}_DYNAMIC'
                    ),
                ))
                if self._wait_residence_panel(True, timeout=2.2):
                    opened = True
                    logger.info(
                        f'{kind} residence marker opened: '
                        f'offset=({dx}, {dy})'
                    )
                    break
                # 未知模态页面不能继续盲点；交给上层恢复/重搜。
                if not self.is_in_main():
                    logger.warning(
                        f'{kind} marker click entered an unknown page: '
                        f'offset=({dx}, {dy})'
                    )
                    break
                # A successful tap may clear the marker while leaving the
                # search arrow visible. Never use the next offset against a
                # disappeared flag: it can land on the bottom Tasks button.
                remaining = self._find_map_interference_marker(excluded)
                same_marker = (
                    remaining is not None
                    and remaining[0] == kind
                    and max(abs(remaining[2][0] - cx),
                            abs(remaining[2][1] - cy)) <= 24
                )
                if not same_marker:
                    if self.find_centered_search_target() is None:
                        self._cleanup_selection_lost = True
                    logger.info(
                        f'{kind} marker moved or disappeared after click; '
                        'stop using old click offsets'
                    )
                    break
                logger.info(
                    f'{kind} marker still at {center}; try lower hit area'
                )

            if opened and self._handle_residence_interference(
                kind, timeout=0.8
            ):
                self.last_interference_attempted = True
                handled += 1
                # 下轮重新查找同一位置。只有事件旗真的消失才算清理完；
                # 若预先排除这个坐标，误以为成功时会永远漏掉该剪刀。
            else:
                excluded.append(center)
                failed_centers = getattr(
                    self, '_cleanup_failed_marker_centers', None
                )
                if failed_centers is not None:
                    failed_centers.add(center)
                # 稳定出现但点不开民宅面板的是完成旗/地图装饰，不算已
                # 处理事件，也不能因此让搜索流程再次重搜并放大误操作。
                if not opened:
                    logger.warning(
                        f'{kind} marker did not open residence panel after '
                        f'{attempted_offsets} hit-area attempts'
                    )
                logger.info(
                    f'Ignore non-residence interference candidate: {center}'
                )
                # 仅当这次点击确实取消了搜索箭头或进入未知页面时才
                # 要求上层重搜；无效点击保持原选择，直接继续即可。
                if not self.is_in_main():
                    self.last_interference_attempted = True
                    break
                if self.find_centered_search_target() is None:
                    self._cleanup_selection_lost = True
                    self.last_interference_attempted = True
                    break
        if handled:
            logger.info(f'Building interference handled: {handled}')
        return handled

    def force_harvest_visible_completions(self):
        """强制收获搜索建筑后当前画面中所有可识别的“完成”节点。"""
        # harvest.py 继承 JiangnanBuilding，放在运行时导入以避免模块循环。
        from module.jiangnan.harvest import JiangnanHarvest

        helper = self if isinstance(self, JiangnanHarvest) else getattr(
            self, '_building_search_harvest_helper', None
        )
        if helper is None:
            helper = JiangnanHarvest(
                config=self.config,
                device=self.device,
                task=getattr(self, 'task', None),
            )
            self._building_search_harvest_helper = helper

        success = helper.harvest_all_visible(
            force=True,
            limit=32,
            require_one=False,
            initial_wait=0.4,
            # 搜索居中状态会隐藏普通主界面锚点，但仍然允许收获完成旗。
            require_main_for_empty=False,
        )
        count = int(getattr(helper, 'last_harvest_count', 0) or 0)
        self.last_harvest_blocked_reason = getattr(
            helper, 'last_harvest_blocked_reason', None,
        )
        if count:
            label = ('Harvested' if success else 'Attempted to harvest')
            logger.info(
                f'{label} visible completions after building search: {count}'
            )
        if not success:
            message = 'Unable to clear all visible completed nodes after building search'
            if getattr(helper, 'last_harvest_blocked_reason', None) == 'warehouse_full_no_discard':
                logger.info(
                    'Skip remaining completed nodes after building search: '
                    'warehouse full, ads disabled'
                )
            else:
                logger.warning(message)
        return success, count

    def ensure_building_search_area_clear(
        self, building_name, reposition, max_rounds=64,
        harvest_completions=True,
    ):
        """尽量清理当前屏幕，再保留或重新定位搜索目标。

        同一屏幕中的可处理目标必须连续处理：居民事件由
        ``handle_building_map_interference`` 连续扫描，生产“完成”由
        ``force_harvest_visible_completions`` 逐个收取并在每次点击后重新
        识别当前屏幕。清理后核对地图背景；保持原位则保存目标箭头
        锚点，只有地图改变或锚点丢失时才调用 ``reposition``。
        """
        max_rounds = max(1, int(max_rounds))
        # A stable decorative/completed banner must not be clicked again on
        # every reposition of the same building.  Reset for the next target.
        self._cleanup_failed_marker_centers = set()
        total_harvested = 0
        reference = self.device.image.copy()
        arrow = self.find_building_search_arrow()
        arrow_size = self.last_search_arrow_size
        for scan_round in range(1, max_rounds + 1):
            logger.info(
                f'Building cleanup current-screen sweep: '
                f'{building_name}, round={scan_round}/{max_rounds}'
            )
            self.handle_building_map_interference()
            interference = bool(self.last_interference_attempted)
            harvested = 0
            if harvest_completions:
                harvest_success, harvested = self.force_harvest_visible_completions()
                total_harvested += harvested
                if not harvest_success:
                    if getattr(self, 'last_harvest_blocked_reason', None):
                        logger.warning(
                            f'Building cleanup blocked by harvest: '
                            f'{self.last_harvest_blocked_reason}; '
                            'defer this target without another search'
                        )
                        return False
                    logger.warning(
                        f'Building cleanup incomplete for {building_name}; '
                        'continue to the searched building if it is still locatable'
                    )
                    break

            logger.info(
                f'Building cleanup current-screen sweep finished: '
                f'{building_name}, interference={interference}, '
                f'harvested={harvested}, total_harvested={total_harvested}'
            )
            if not interference and harvested == 0:
                if (total_harvested > 0
                        and self.find_building_search_arrow(
                            allow_retained=False, report_missing=False,
                        ) is None):
                    # Harvesting can remove the selected building's marker
                    # while leaving the map stationary. A retained arrow
                    # then points at bare ground, so center the result again.
                    logger.info(
                        f'Recenter {building_name} after clearing '
                        f'{total_harvested} completed nodes'
                    )
                    if not reposition():
                        logger.warning(
                            f'Failed to recenter {building_name} after harvest'
                        )
                        return False
                logger.info(
                    f'Building target area is clear: {building_name}'
                )
                return True

            if scan_round >= max_rounds:
                break
            if (arrow is not None
                    and not getattr(self, '_cleanup_selection_lost', False)
                    and self._map_view_unchanged(
                        reference, self.device.image
                    )):
                self._retained_search_arrow = arrow
                self._retained_search_arrow_size = arrow_size
                self._retained_search_frame = self.device.image.copy()
                logger.info(
                    f'Building map stayed in place after cleanup; '
                    f'reuse {building_name} at {arrow}'
                )
                continue
            logger.info(
                f'Building map changed; reposition after cleanup: '
                f'{building_name}'
            )
            self._retained_search_arrow = None
            self._retained_search_arrow_size = None
            self._retained_search_frame = None
            if not reposition():
                logger.warning(
                    f'Failed to reposition building after cleanup: '
                    f'{building_name}'
                )
                return False
            reference = self.device.image.copy()
            arrow = self.find_building_search_arrow(allow_retained=False)
            arrow_size = self.last_search_arrow_size

        logger.warning(
            f'Building cleanup did not converge: {building_name}, '
            f'harvested={total_harvested}; check searched target'
        )
        if (total_harvested > 0
                and self.find_building_search_arrow(
                    allow_retained=False, report_missing=False,
                ) is None):
            return bool(reposition())
        if (arrow is not None
                and not getattr(self, '_cleanup_selection_lost', False)
                and self._map_view_unchanged(reference, self.device.image)):
            self._retained_search_arrow = arrow
            self._retained_search_arrow_size = arrow_size
            self._retained_search_frame = self.device.image.copy()
            logger.info(f'Reuse searched {building_name} after cleanup')
            return True
        if reposition():
            logger.info(f'Searched {building_name} repositioned after cleanup')
            return True
        logger.warning(f'Searched {building_name} could not be relocated')
        return False

    BUILDING_NO_RESULT_SEARCH_AREA = (
        330, 170, 960, 420
    )

    def is_building_search_no_result(
        self, similarity=0.65, use_ocr=False
    ):
        """
        检测建筑搜索的“没有满足当前筛选条件的建筑”浮动提示。

        该提示会在屏幕中间上下移动，因此不能使用固定 appear()。
        """
        image = self.device.image

        if image is None:
            return False

        BUILDING_SEARCH_NO_RESULT.ensure_template()
        template = BUILDING_SEARCH_NO_RESULT.image

        x1, y1, x2, y2 = self.BUILDING_NO_RESULT_SEARCH_AREA
        search = crop(
            image,
            (x1, y1, x2, y2),
            copy=False,
        )

        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return False

        score, location, gray_score, edge_score = stable_feature_match(
            search, template, gray_weight=0.45
        )

        logger.attr(
            'BuildingSearchNoResultScore',
            f'stable={score:.3f}, gray={gray_score:.3f}, '
            f'edge={edge_score:.3f}',
        )

        if score < similarity:
            if not use_ocr:
                return False

            # 浮动提示会上下移动、透明度也在变化，鸡鸣山实机
            # 的模板分数最高只有 0.350。在筛选提交后的指定帧中，
            # 将可能经过的高度分带一次 OCR，直接识别提示文字。
            areas = [
                (350, y, 1050, y + 60)
                for y in (180, 230, 280, 330, 380)
            ]
            texts = Ocr(
                [
                    Button(
                        area=area,
                        color=(),
                        button=area,
                        name=f'BUILDING_NO_RESULT_OCR_{index + 1}',
                    )
                    for index, area in enumerate(areas)
                ],
                lang='cnocr',
                name='BUILDING_NO_RESULT_OCR',
            ).ocr(image)
            if not isinstance(texts, list):
                texts = [texts]
            normalized = [
                re.sub(r'\s+', '', str(text)) for text in texts
            ]
            logger.attr('BuildingSearchNoResultText', normalized)
            detected = any(
                '没有满足当前筛选条件的建筑' in text
                or (
                    '满足当前筛选条件' in text
                    and '建筑' in text
                )
                for text in normalized
            )
            if not detected:
                return False
            logger.info('Building search no-result detected by OCR')
            return True

        left = x1 + location[0]
        top = y1 + location[1]

        logger.info(
            'Building search no-result detected: '
            f'score={score:.3f}, '
            f'position=({left}, {top})'
        )

        return True

    def _wait_search_page(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            if self.is_building_search_page():
                return True
            self.device.sleep(0.4)
        logger.warning('Building search page timeout')
        return False

    def is_foundation_mode_selected(self):
        """按“地基”字样定位按钮；兼容五列和六列府印面板。"""
        area = self.anchored_area(CITY_PANEL_CHECK, (275, 255, 400, 390))
        image = crop(self.device.image, area, copy=False)
        self.last_foundation_option = None
        self.last_foundation_red_ratio = None
        if image is None or image.size == 0:
            logger.warning('Foundation option search area is unavailable')
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scores = []
        for selected, template in (
            (True, self._foundation_selected_template),
            (False, self._foundation_unselected_template),
        ):
            reference = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            if (gray.shape[0] < reference.shape[0]
                    or gray.shape[1] < reference.shape[1]):
                continue
            result = cv2.matchTemplate(
                gray, reference, cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)
            scores.append((float(score), selected, location, reference.shape))
        if not scores:
            return False
        score, selected, location, shape = max(scores)
        logger.attr('FoundationOptionMatch', f'score={score:.3f}, selected={selected}')
        if score < 0.72:
            logger.warning('Foundation option was not recognized in city panel')
            return False
        center_x = area[0] + location[0] + shape[1] // 2
        center_y = area[1] + location[1] + shape[0] // 2
        self.last_foundation_option = Button(
            area=(), color=(),
            button=(center_x - 12, center_y - 18,
                    center_x + 12, center_y + 18),
            name='FOUNDATION_BUTTON_DYNAMIC',
        )
        option_area = (center_x - 20, center_y - 52,
                       center_x + 20, center_y + 52)
        hsv = cv2.cvtColor(
            crop(self.device.image, option_area, copy=False),
            cv2.COLOR_RGB2HSV,
        )
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 65, 65), (13, 255, 255)),
            cv2.inRange(hsv, (168, 65, 65), (179, 255, 255)),
        )
        red_ratio = cv2.countNonZero(red) / float(red.size)

        # Keep color as a diagnostic and fallback after a deliberate click;
        # city filters change its absolute value too much for primary use.
        self.last_foundation_red_ratio = red_ratio
        logger.attr(
            'FoundationModeState',
            f'selected={selected}, red_ratio={red_ratio:.3f}, '
            f'position=({center_x}, {center_y}), score={score:.3f}',
        )
        return selected

    def _open_building_search_tool(self, search_tool):
        """Open a visually confirmed toolbar icon and verify the search page."""
        logger.info('Open building search')
        self.device.click(search_tool)
        click_count = 1
        last_click = time.time()
        deadline = time.time() + 10
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self.is_network_timeout_popup():
                logger.warning('Network popup interrupted building search opening')
                self.handle_network_timeout_popup()
                return False
            if self.is_building_search_page():
                return self.ensure_building_search_unfiltered()
            # A swallowed click may leave the toolbar visible. Re-recognize
            # the icon before retrying; never click stale coordinates.
            if click_count < 3 and time.time() - last_click >= 2:
                search_tool = self.find_building_search_tool()
                if search_tool is not None:
                    click_count += 1
                    last_click = time.time()
                    logger.info(f'Retry recognized building search tool: {click_count}/3')
                    self.device.click(search_tool)
        logger.warning('Building search page timeout')
        return False

    def goto_building_search(self):
        """先从府印面板确认地基模式，再进入建筑搜索。"""
        logger.hr('JIANGNAN BUILDING SEARCH')
        # 普通模式也有搜索按钮，且能打开同一个搜索页；两者都不能证明
        # 地基模式已启用。尤其宁波商船后续只能通过地基图标定位，必须
        # 先读取府印面板中的实际选中状态，不能仅凭搜索入口跳过此步。
        if not self.goto_city_panel():
            return False

        def close_city_panel(attempt):
            seal = self.find_city_seal_button()
            if seal is None:
                seal = Button(
                    area=(), color=(), button=(62, 570, 92, 625),
                    name=f'CITY_SEAL_CLOSE_PANEL_FALLBACK_{attempt}',
                )
            self.device.click(seal)

        self.device.screenshot()
        selected = self.is_foundation_mode_selected()
        initial_foundation_ratio = getattr(
            self, 'last_foundation_red_ratio', None
        )
        logger.attr('FoundationSelected', selected)
        if selected:
            logger.info('Foundation mode is already selected; close city panel')
            close_city_panel(1)
        else:
            logger.info('Foundation mode is not selected; select it')
            if self.last_foundation_option is None:
                return False
            self.device.click(self.last_foundation_option)

        search_tool = None
        panel_absent_frames = 0
        select_click_count = 1 if not selected else 0
        close_click_count = 1 if selected else 0
        last_click = time.time()
        deadline = time.time() + 8
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if self.is_network_timeout_popup():
                logger.warning('Network popup interrupted foundation selection')
                self.handle_network_timeout_popup()
                return False
            if self.is_city_panel():
                panel_absent_frames = 0
                # 选择“地基”只会改变选项底色，城市面板仍会留在屏幕上。
                # 必须在每次点击后重新读取当前状态；旧代码一直沿用进入
                # 循环前的 selected，因而会重复点击并把地基模式来回切换。
                current_selected = self.is_foundation_mode_selected()
                current_ratio = getattr(
                    self, 'last_foundation_red_ratio', None
                )
                # 部分城市/季节下选中态模板会失效，但点击“地基”后按钮
                # 会从红底切成浅底，红色占比显著下降。日志实测出现
                # 0.802 -> 0.546；这个相对变化只在本次明确点击后使用，
                # 不拿绝对颜色阈值去误判鸡鸣山等特殊背景。
                if (
                    not current_selected
                    and select_click_count > 0
                    and initial_foundation_ratio is not None
                    and current_ratio is not None
                    and initial_foundation_ratio - current_ratio >= 0.15
                ):
                    current_selected = True
                    logger.info(
                        'Foundation mode confirmed by toggle color change: '
                        f'{initial_foundation_ratio:.3f} -> {current_ratio:.3f}'
                    )
                if current_selected:
                    if close_click_count < 3 and (
                        close_click_count == 0
                        or time.time() - last_click >= 2
                    ):
                        close_click_count += 1
                        last_click = time.time()
                        logger.info(
                            'Foundation mode selected; close city panel '
                            f'({close_click_count}/3)'
                        )
                        close_city_panel(close_click_count)
                elif (
                    select_click_count < 3
                    and time.time() - last_click >= 2
                ):
                    select_click_count += 1
                    last_click = time.time()
                    logger.info(
                        'Retry foundation mode selection: '
                        f'{select_click_count}/3'
                    )
                    if self.last_foundation_option is None:
                        logger.warning('Foundation option lost during retry')
                        return False
                    self.last_foundation_option.name = (
                        f'FOUNDATION_BUTTON_DYNAMIC_RETRY_'
                        f'{select_click_count}'
                    )
                    self.device.click(self.last_foundation_option)
                continue

            panel_absent_frames += 1
            # 旧逻辑点击后 0.19 秒便用一个搜索图标假确认。必须先连续
            # 两帧确认城市面板已经关闭，再接受地基工具栏的搜索图标。
            if panel_absent_frames < 2:
                continue
            search_tool = self.find_building_search_tool()
            if search_tool is not None:
                logger.info('Foundation mode confirmed after panel closed')
                break

        if search_tool is None:
            logger.warning('Foundation mode was not confirmed')
            return False

        return self._open_building_search_tool(search_tool)

    @staticmethod
    def normalize_building_name(name):
        return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', str(name)))

    def _building_selector_visible(self):
        """Confirm a live n/total counter above the paired selector buttons."""
        image = getattr(self.device, 'image', None)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
            return False
        x1, y1, x2, y2 = (930, 150, 1200, 470)
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return False
        mask = cv2.inRange(cv2.cvtColor(region, cv2.COLOR_RGB2HSV),
                           (65, 45, 70), (110, 255, 230))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda box: box[1])
        boxes = [box for box in boxes if 100 <= box[2] <= 190 and 30 <= box[3] <= 75]
        for index, (left, top, width, height) in enumerate(boxes):
            if not any(abs(left - other[0]) <= 15 and abs(width - other[2]) <= 15
                       and abs(height - other[3]) <= 10 and 40 <= other[1] - top <= 100
                       for other in boxes[index + 1:]):
                continue
            area = (x1 + left, max(0, y1 + top - 68), x1 + left + width, y1 + top - 2)
            raw = Ocr(Button(area=area, color=(), button=area,
                             name='BUILDING_SELECTOR_COUNT'), lang='cnocr',
                      name='BUILDING_SELECTOR_COUNT_OCR').ocr(image)
            count = re.fullmatch(r'\s*(\d+)\s*[/／]\s*(\d+)\s*', str(raw))
            if count and 1 <= int(count.group(1)) <= int(count.group(2)) and int(count.group(2)) > 1:
                return True
        return False

    def _click_searched_building(self, target, navigation_buttons=()):
        """Use two taps only while the current frame has a multi-building selector."""
        # Search totals and previously detected buttons survive production
        # start; the selector itself does not. Always inspect the current frame.
        if self._building_selector_visible():
            self.device.multi_click(target, n=2, interval=0.12)
        else:
            self.device.click(target)

    @classmethod
    def building_result_is_no_result(cls, text):
        """识别搜索提交后覆盖在结果首项上的“抱歉”无结果浮字。"""
        normalized = cls.normalize_building_name(text)
        return bool(normalized) and (
            '没有满足当前筛选条件的建筑' in normalized
            or '抱歉' in normalized
            # 浮字移动、淡出时 PP-OCR 偶发读成“拘歉/地歉”。搜索已
            # 成功提交且结果区域只剩这个短词时，同样属于明确无结果。
            or ('歉' in normalized and len(normalized) <= 4)
        )

    def building_search_confirmed_missing(self):
        """只有游戏明确无结果，或业务校验确认全是排除项时才返回 True。"""
        return self.last_search_status == self.SEARCH_NOT_FOUND

    def disable_city_building_feature(
        self,
        task,
        group_prefix,
        city_key,
        city,
        building_name,
        status_field=None,
        status_text=None,
        next_run_field=None,
        next_run_value=None,
        update=True,
    ):
        """公共的“城市没有某建筑”配置关闭逻辑，供后续模块复用。"""
        # 这是所有建筑模块最终共用的保险门：只有本轮搜索已经得到
        # 游戏明确的“无结果”（或业务明确排除的结果）才允许改配置。
        # OCR 失败、界面超时、状态识别失败和仓库溢出都不能关闭功能。
        if not self.building_search_confirmed_missing():
            logger.warning(
                f'Refuse to disable {building_name} in {city}: '
                f'search_status={self.last_search_status}'
            )
            return None
        base = f'{task}.{group_prefix}_{city_key}'
        enable_path = f'{base}.Enable'
        self.config.modified[enable_path] = False
        if status_field:
            self.config.modified[f'{base}.{status_field}'] = (
                status_text or f'无{building_name}，已自动关闭'
            )
        if next_run_field and next_run_value is not None:
            self.config.modified[f'{base}.{next_run_field}'] = next_run_value
        if update:
            self.config.update()
        logger.warning(
            f'{building_name} not found in {city}; disable {enable_path}'
        )
        return enable_path

    def input_building_name(self, name, timeout=8):
        self.device.screenshot()
        clear_button = self.anchored_button(
            BUILDING_SEARCH_PAGE_CHECK,
            BUILDING_SEARCH_CLEAR_BUTTON,
            name='BUILDING_SEARCH_CLEAR_DYNAMIC',
            inset=7,
        )
        x1, y1, x2, y2 = clear_button.button
        clear_crop = self.device.image[y1:y2, x1:x2]
        clear_hsv = cv2.cvtColor(clear_crop, cv2.COLOR_RGB2HSV)
        clear_red = cv2.bitwise_or(
            cv2.inRange(clear_hsv, (0, 65, 80), (15, 255, 255)),
            cv2.inRange(clear_hsv, (165, 65, 80), (179, 255, 255)),
        )
        has_dynamic_clear = cv2.countNonZero(clear_red) >= 45
        if has_dynamic_clear or self.appear(
            BUILDING_SEARCH_CLEAR_BUTTON, offset=(3, 3), similarity=0.82
        ):
            logger.info('Clear previous building search text')
            self.device.click(clear_button)
            self.device.sleep(0.4)

        self.device.click(self.anchored_button(
            BUILDING_SEARCH_PAGE_CHECK,
            BUILDING_SEARCH_INPUT,
            name='BUILDING_SEARCH_INPUT_DYNAMIC',
            inset=6,
        ))
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, shown = self.device.u2.current_ime()
            if shown:
                break
            self.device.sleep(0.3)
        else:
            logger.warning('Building search keyboard did not open')
            return False

        try:
            instance = self.device.emulator_instance
            instance_id = instance.MuMuPlayer12_id if instance else None
            if instance_id is not None:
                # MuMu 15 禁止 ADB 启用 FastInputIME，uiautomator2.send_keys()
                # 会退回到 focused selector 并长时间卡住。MuMuManager 的
                # input_text 是模拟器原生中文输入接口，可以直接输入建筑名。
                manager = instance.emulator.single_to_console(instance.path)
                completed = subprocess.run(
                    [
                        manager,
                        'adb',
                        '-v',
                        str(instance_id),
                        '-c',
                        'input_text',
                        name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if completed.returncode != 0:
                    logger.warning(
                        f'MuMu input_text failed: {completed.stderr.strip()}'
                    )
                    return False
            else:
                self.device.u2.send_keys(text=name, clear=True)

            # MuMuManager 返回时中文可能仍在下一帧才真正写入输入框；
            # 立即发送 ENTER 会偶发提交空串，只留下占位文字。
            self.device.sleep(0.45)
            # 游戏的原生输入框把 ENTER 当作右下角“确定”，不依赖
            # FastInputIME 的 send_action。
            self.device.adb_shell(['input', 'keyevent', '66'])
        except Exception as exc:
            logger.warning(f'Unable to input building name: {exc}')
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            _, shown = self.device.u2.current_ime()
            if not shown:
                return True
            self.device.sleep(0.3)
        logger.warning('Building search keyboard did not close')
        return False

    def _search_input_matches(self, name):
        """读取当前输入框；结果不符时不能继续消费旧搜索列表。"""
        self.device.screenshot()
        raw = self.search_input_ocr.ocr(self.device.image)
        if isinstance(raw, list):
            raw = raw[0] if raw else ''
        actual = self.normalize_building_name(raw)
        expected = self.normalize_building_name(name)
        # 允许 OCR 偶尔漏掉一个字；输入框多出文字则可能是上次内容
        # 没有清空，尤其“议事堂议事堂”会被包含判断误当成正确输入。
        matched = bool(actual) and (
            actual == expected
            or (
                len(actual) >= max(2, len(expected) - 1)
                and actual in expected
            )
        )
        logger.attr(
            'BuildingSearchInput',
            f'expected={expected}, actual={actual or "<empty>"}, '
            f'matched={matched}',
        )
        return matched

    # 搜索结果网格最多显示两整行。第一项继续使用实机校准区域；
    # 第一项不符合调用方要求时，再扫描其余卡片。
    BUILDING_OTHER_RESULT_AREAS = (
        (486, 283, 627, 317), (643, 283, 784, 317),
        (800, 283, 941, 317), (957, 283, 1098, 317),
        (329, 478, 470, 512), (486, 478, 627, 512),
        (643, 478, 784, 512), (800, 478, 941, 512),
        (957, 478, 1098, 512),
    )

    @staticmethod
    def _building_result_click(area):
        x1, y1, x2, _ = area
        center_x = (x1 + x2) // 2
        center_y = y1 - 70
        return Button(
            area=(), color=(),
            button=(center_x - 10, center_y - 10,
                    center_x + 10, center_y + 10),
            name='BUILDING_MATCHED_RESULT_DYNAMIC',
        )

    def search_building(
        self,
        name,
        result_validator=None,
        rejected_is_not_found=None,
        exact=False,
        handle_interference=True,
        retry_stale_result=True,
    ):
        """搜索建筑并点击第一条同名结果，使该建筑居中。"""
        self._retained_search_arrow = None
        self._retained_search_arrow_size = None
        self._retained_search_frame = None
        self.last_search_result_name = ''
        self.last_search_status = 'unknown'
        self.last_search_executed = False
        name = self.normalize_building_name(name)
        if not name:
            self.last_search_status = self.SEARCH_RESULT_MISMATCH
            logger.warning('Building name is empty')
            return False
        if not self.goto_building_search():
            self.last_search_status = self.SEARCH_OPEN_FAILED
            return False
        input_ok = False
        for input_attempt in range(1, 4):
            if (
                self.input_building_name(name)
                and self._search_input_matches(name)
            ):
                input_ok = True
                break
            logger.warning(
                f'Building search input was not committed: {name}, '
                f'attempt={input_attempt}/3'
            )
        if not input_ok:
            self.last_search_status = self.SEARCH_INPUT_FAILED
            logger.warning(f'Building search input verification failed: {name}')
            return False

        if result_validator is None:
            if exact:
                result_validator = lambda value: value == name
            else:
                result_validator = lambda value: bool(value) and (
                    name in value or value in name
                )

        self.device.screenshot()
        if not self.is_building_search_page():
            self.last_search_status = self.SEARCH_PAGE_LOST
            logger.warning('Building search page missing after input')
            return False
        # 搜索按钮模板会随窗口边框出现 1~数像素偏移，先识别再点击，
        # 不直接依赖 assets.py 中的参考坐标。
        if not self.appear(
            BUILDING_SEARCH_BUTTON,
            offset=(12, 12),
            similarity=0.82,
        ):
            self.last_search_status = self.SEARCH_BUTTON_MISSING
            logger.warning('Building search button was not found')
            return False
        self.device.click(BUILDING_SEARCH_BUTTON)
        self.last_search_executed = True

        # 搜索完成后，先观察短时间内是否出现“无结果”浮动提示。
        #
        # 提示会从中部向上移动然后消失，因此连续观察，
        # 而不是只在一个固定时间点截图。
        # 点击后的第一帧仍可能是旧结果页。该静态页面与提示模板的
        # stable 分数长期固定在 0.522；真正的无结果浮字出现后实机
        # 分数为 0.70~0.96。先等界面提交搜索，再以 0.65 检测，避免
        # 把仍然存在的码头/生产建筑误判成不存在。
        self.device.sleep(0.45)
        no_result_deadline = time.time() + 3.5
        quick_result = ''
        quick_result_frames = 0
        stabilized_result = ''

        while time.time() < no_result_deadline:
            self.device.sleep(0.2)
            self.device.screenshot()

            # 网络弹窗可能恰好覆盖搜索结果。旧列表仍隐约可见，若继续
            # OCR 会被误判为“结果不匹配”，甚至把该城市的设施关闭。
            # 恢复网络后中止本轮搜索，交给设施任务按复查策略重试。
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                self.last_search_status = self.SEARCH_PAGE_LOST
                logger.warning(
                    f'Building search interrupted by network recovery: {name}'
                )
                return False

            if self.is_building_search_no_result():
                self.last_search_status = self.SEARCH_NOT_FOUND
                self.last_search_result_name = ''

                logger.info(
                    f'Building not found: {name}'
                )

                return False

            # 有结果的搜索不需要陪“无结果”提示等满 3.5 秒。第一项名称
            # 连续两帧命中才提前结束，既能避开搜索提交动画，也不会因
            # PP-OCR 的单帧误识别点击错误建筑；找不到时仍完整观察浮字。
            raw_quick = self.result_ocr.ocr(self.device.image)
            if isinstance(raw_quick, list):
                raw_quick = raw_quick[0] if raw_quick else ''
            current_quick = self.normalize_building_name(raw_quick)
            if self.building_result_is_no_result(current_quick):
                self.last_search_status = self.SEARCH_NOT_FOUND
                self.last_search_result_name = ''
                logger.info(
                    f'Building not found from result toast: {name}, '
                    f'text={current_quick}'
                )
                return False
            # Other level groups of this same building are valid search
            # results too. Once stable, inspect the other cards immediately
            # instead of waiting out the no-result toast window for a level
            # that is not on the first card.
            if current_quick and (
                result_validator(current_quick) or name in current_quick
            ):
                if current_quick == quick_result:
                    quick_result_frames += 1
                else:
                    quick_result = current_quick
                    quick_result_frames = 1
                if quick_result_frames >= 2:
                    stabilized_result = quick_result
                    logger.info(
                        f'Building search result stabilized early: '
                        f'{quick_result}'
                    )
                    break
            else:
                quick_result = ''
                quick_result_frames = 0

        # 连续两帧得到同一建筑名称时直接复用第二帧，避免第三次 OCR。
        # 该卡片若不满足等级校验，下面仍会检查其余卡片后再决定点击。
        # 尚未稳定时才执行最终识别。
        if stabilized_result:
            result_name = stabilized_result
        else:
            self.device.screenshot()
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                self.last_search_status = self.SEARCH_PAGE_LOST
                logger.warning(
                    f'Building search interrupted by network recovery: {name}'
                )
                return False

            raw = self.result_ocr.ocr(self.device.image)
            if isinstance(raw, list):
                raw = raw[0] if raw else ''
            result_name = self.normalize_building_name(raw)
            if self.building_result_is_no_result(result_name):
                self.last_search_status = self.SEARCH_NOT_FOUND
                self.last_search_result_name = ''
                logger.info(
                    f'Building not found from final result toast: {name}, '
                    f'text={result_name}'
                )
                return False
        self.last_search_result_name = result_name
        logger.attr('BuildingSearchResult', result_name)
        matched_button = self.anchored_button(
            BUILDING_SEARCH_PAGE_CHECK,
            BUILDING_FIRST_RESULT,
            name='BUILDING_FIRST_RESULT_DYNAMIC',
        )
        candidates = [result_name] if result_name else []
        if not result_validator(result_name):
            other_ocr = BuildingResultOcr()
            other_ocr.buttons = list(self.BUILDING_OTHER_RESULT_AREAS)
            raw_results = other_ocr.ocr(self.device.image)
            for area, candidate in zip(
                self.BUILDING_OTHER_RESULT_AREAS, raw_results
            ):
                candidate = self.normalize_building_name(candidate)
                if candidate:
                    candidates.append(candidate)
                if result_validator(candidate):
                    result_name = candidate
                    self.last_search_result_name = candidate
                    matched_button = self._building_result_click(area)
                    logger.info(f'Validated building result found: {candidate}')
                    break
        if not result_validator(result_name):
            if (
                rejected_is_not_found is not None
                and any(rejected_is_not_found(value) for value in candidates)
            ):
                self.last_search_status = self.SEARCH_NOT_FOUND
            else:
                self.last_search_status = self.SEARCH_RESULT_MISMATCH
            logger.warning(
                f'Building search result mismatch: expected={name}, '
                f'candidates={candidates}, status={self.last_search_status}'
            )
            # 输入框正确但结果仍是官府、码头等默认列表，说明搜索按钮
            # 被游戏吞掉。返回地图完整重提一次；明确排除项（例如只搜
            # 到改良农田）已经是 not_found，不会进入这个分支。
            if (
                retry_stale_result
                and self.last_search_status == self.SEARCH_RESULT_MISMATCH
            ):
                logger.warning(f'Retry stale building search result once: {name}')
                self.device.press_back()
                self.device.sleep(0.6)
                return self.search_building(
                    name,
                    result_validator=result_validator,
                    rejected_is_not_found=rejected_is_not_found,
                    exact=exact,
                    handle_interference=handle_interference,
                    retry_stale_result=False,
                )
            return False

        # Tasks with several level groups may inspect the complete result page
        # before selecting the validated card.
        observer = getattr(self, '_on_building_search_result', None)
        if callable(observer) and observer(result_name) is False:
            self.last_search_status = self.SEARCH_RESULT_MISMATCH
            return False
        self.device.click(matched_button)
        deadline = time.time() + 12
        search_absent_count = 0
        while time.time() < deadline:
            self.device.sleep(0.5)
            self.device.screenshot()
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                self.last_search_status = self.SEARCH_PAGE_LOST
                logger.warning(
                    f'Building centering interrupted by network recovery: {name}'
                )
                return False
            if self.is_building_search_page():
                search_absent_count = 0
                continue
            search_absent_count += 1
            # 建筑居中后，地基模式会同时隐藏通用主界面锚点和
            # 左侧搜索图标，只留下右侧“上一个/下一个”面板。
            # 因此连续两帧确认搜索页已消失，即可判定点击成功。
            if search_absent_count >= 2:
                self.last_search_status = self.SEARCH_FOUND
                logger.info(f'Building centered: {name}')
                self._remember_centered_search_arrow()
                # 居民事件竖旗会覆盖居中的建筑图标。先清理中国结和
                # 剪刀事件及其中的全部红点详情，再交给设施模块点击。
                if handle_interference:
                # 箭头可能在清理期间超时；保留其空间锚点，只在地图
                # 移动或无法校验时才重新搜索。
                    if not self.ensure_building_search_area_clear(
                        name,
                        reposition=lambda: self.search_building(
                            name,
                            result_validator=result_validator,
                            rejected_is_not_found=rejected_is_not_found,
                            exact=exact,
                            handle_interference=False,
                            retry_stale_result=retry_stale_result,
                        ),
                    ):
                        self.last_search_status = self.SEARCH_CENTER_TIMEOUT
                        return False
                return True
        self.last_search_status = self.SEARCH_CENTER_TIMEOUT
        logger.warning(f'Building did not return to main page: {name}')
        return False

        # ========================================================
    # Tax office
    # ========================================================

    TAX_ICON_SEARCH_AREA = (140, 80, 1140, 570)
    TAX_ICON_SIMILARITY = 0.66
    TAX_LOCK_SIMILARITY = 0.85
    TAX_UNLOCKED_MAX_SCORE = 0.55

    def is_tax_idle(self):
        """判断当前截图是否为税课司空闲等级面板。"""
        return self.appear(
            TAX_IDLE_CHECK,
            offset=(5, 5),
            similarity=0.85,
        )

    def is_tax_working(self):
        """判断当前截图是否为税课司工作中界面。"""
        return self.appear(
            TAX_WORKING_CHECK,
            offset=(5, 5),
            similarity=0.85,
        )

    def tax_panel_state(self):
        """只识别当前截图，不截图、不点击。"""
        working = self.is_tax_working()
        idle = self.is_tax_idle()

        if working and not idle:
            return 'working'

        if idle and not working:
            return 'idle'

        return None

    @staticmethod
    def parse_tax_remaining_time(text):
        """
        解析税课司中文倒计时。

        支持：
            3小时39分44秒
            39分44秒
            44秒

        Returns:
            int | None:
                剩余秒数；无法可靠解析时返回 None。
        """
        if text is None:
            return None

        text = re.sub(r'\s+', '', str(text)).translate(str.maketrans({
            'i': '1', 'I': '1', 'l': '1', '|': '1', '!': '1',
            'O': '0', 'o': '0',
            '肘': '时',
        }))

        # 描边文字可能在数字与单位间混入一个拉丁字符，但仍要求
        # 整段只有时分秒结构，不能从任意 OCR 文本中截取数字。
        match = re.fullmatch(
            r'\D{0,2}'
            r'(?:(\d{1,2})\D{0,2}小时)?'
            r'(?:(\d{1,2})\D{0,2}分)?'
            r'(\d{1,2})\D{0,2}秒'
            r'\D{0,2}',
            text,
        )

        if match is None:
            logger.warning(
                f'Unable to parse tax remaining time: {text}'
            )
            return None

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3))

        if minutes >= 60 or seconds >= 60:
            logger.warning(
                f'Invalid tax remaining time: {text}'
            )
            return None

        total = hours * 3600 + minutes * 60 + seconds

        return total

    def read_tax_remaining_time(self):
        """
        读取当前税课司工作中的剩余时间。

        调用前应已经截图。
        不点击、不切换页面。

        Returns:
            int | None:
                剩余秒数。
        """
        if not self.is_tax_working():
            logger.warning(
                'Tax office is not on working panel'
            )
            return None

        # 使用工作中面板的实时锚点修正 OCR 区域。
        # TAX_REMAINING_TIME.area 是完整截图中的参考坐标。
        area = self.anchored_area(
            TAX_WORKING_CHECK,
            TAX_REMAINING_TIME.area,
        )

        self.tax_remaining_ocr.buttons = area

        raw = self.tax_remaining_ocr.ocr(
            self.device.image
        )

        if isinstance(raw, list):
            raw = raw[0] if raw else ''

        raw = str(raw).strip()

        logger.attr(
            'TaxRemainingRaw',
            raw,
        )

        remaining = self.parse_tax_remaining_time(raw)

        logger.attr(
            'TaxRemainingSeconds',
            remaining,
        )

        if remaining is not None:
            logger.info(
                'Tax remaining time: '
                f'{remaining // 3600}h '
                f'{remaining % 3600 // 60}m '
                f'{remaining % 60}s'
            )

        return remaining

    def find_tax_office_icon(self, similarity=None):
        """
        在当前城市地图中查找离屏幕中心最近的屋顶图标。

        返回动态 Button 或 None。
        不截图、不点击、不移动地图。
        """
        if similarity is None:
            similarity = self.TAX_ICON_SIMILARITY

        image = self.device.image

        if image is None or image.shape[:2] != (720, 1280):
            logger.warning(
                'Tax icon search requires a 1280x720 screenshot'
            )
            return None

        if not self.is_in_main():
            logger.warning(
                'Tax icon search requires the city main page'
            )
            return None

        TAX_MAP_ICON.ensure_template()
        template = crop_map_icon_foreground(TAX_MAP_ICON.image)

        x1, y1, x2, y2 = self.TAX_ICON_SEARCH_AREA
        search = crop(
            image,
            (x1, y1, x2, y2),
            copy=False,
        )

        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None

        scores, gray_scores, edge_scores = stable_feature_map(
            search, template, gray_weight=0.30
        )
        height, width = template.shape[:2]
        candidates = []

        # 每次取最高峰，并抑制同一屋顶附近的重复峰值。
        for _ in range(32):
            _, score, _, point = cv2.minMaxLoc(scores)

            if score < similarity:
                break

            px, py = point
            cx = x1 + px + width // 2
            cy = y1 + py + height // 2

            distance = (
                (cx - image.shape[1] / 2) ** 2
                + (cy - image.shape[0] / 2) ** 2
            )

            candidates.append((
                distance,
                -float(score),
                cx,
                cy,
                float(gray_scores[py, px]),
                float(edge_scores[py, px]),
            ))

            cv2.rectangle(
                scores,
                (
                    max(0, px - width),
                    max(0, py - height),
                ),
                (
                    min(scores.shape[1] - 1, px + width),
                    min(scores.shape[0] - 1, py + height),
                ),
                -1.0,
                -1,
            )

        if not candidates:
            logger.warning(
                'No tax office roof icon found'
            )
            return None

        # 税课司不是唯一屋顶图标，不能依赖图标相似度区分。
        # 正确规则：搜索箭头指向的区域内，离屏幕中心最近的屋顶就是目标。
        # stable score 只用于过滤明显错误候选，不参与最终排序。
        arrow = getattr(self, 'last_search_arrow', None)
        if arrow is not None and candidates:
            arrow_x, arrow_tip, _ = arrow
            directed = [
                item for item in candidates
                if abs(item[2] - arrow_x) < 80
                and item[3] >= arrow_tip
            ]
            if directed:
                candidates = directed
                logger.info(
                    f'Tax icon candidates filtered by search arrow: {len(candidates)}'
                )

        candidates.sort(key=lambda item: item[0])
        distance, negative_score, cx, cy, gray_score, edge_score = candidates[0]

        logger.info(
            'Tax roof candidates=%s, selected=(%s, %s), '
            'stable=%.3f, gray=%.3f, edge=%.3f'
            % (
                len(candidates), cx, cy, -negative_score,
                gray_score, edge_score,
            )
        )

        return Button(
            area=(),
            color=(),
            button=(
                cx - 6,
                cy - 6,
                cx + 6,
                cy + 6,
            ),
            name='TAX_MAP_ICON_DYNAMIC',
        )

    def open_tax_office(self, timeout=10):
        """
        打开税课司并验证界面。

        返回 'working'、'idle' 或 None。
        不选择工作。
        """
        self.device.screenshot()

        state = self.tax_panel_state()
        if state is not None:
            return state

        if not self.is_in_main():
            logger.warning(
                'Cannot open tax office from an unknown page'
            )
            return None

        # 税课司搜索结果中经常有居民站在箭头下方。通用前景轮廓会把
        # 居民的头/身体误当建筑（松江府实测从应点的约 640,387 偏到
        # 583,327）。仍先运行通用检测，以保留鸡鸣山等无可用轮廓时
        # 已验证有效的箭头 +42px 兜底；若确实找到了轮廓，则忽略它的
        # 横向偏移，改点箭头轴线下方约 100px 的税课司主体。
        target = self.find_centered_search_target()
        arrow = getattr(self, 'last_search_arrow', None)
        if (
            target is not None
            and arrow is not None
            and self.last_search_target_arrow_fallback
        ):
            # A generic +42px arrow fallback landed on bare ground in
            # 徽州府. The tax roof is still visible farther below the arrow.
            roof = self.find_tax_office_icon()
            if roof is not None:
                x1, y1, x2, y2 = roof.button
                roof_x, roof_y = (x1 + x2) // 2, (y1 + y2) // 2
                arrow_x, arrow_tip, _ = arrow
                if (abs(roof_x - arrow_x) <= 80
                        and arrow_tip <= roof_y <= arrow_tip + 170):
                    target = roof
                    logger.info(
                        f'Tax office roof replaces arrow fallback: '
                        f'({roof_x}, {roof_y})'
                    )
        if (
            target is not None
            and arrow is not None
            and not self.last_search_target_arrow_fallback
        ):
            arrow_x, arrow_tip, _ = arrow
            target_y = min(self.SEARCH_TARGET_MAX_Y - 10, arrow_tip + 100)
            target = Button(
                area=(), color=(),
                button=(
                    arrow_x - 7, target_y - 7,
                    arrow_x + 7, target_y + 7,
                ),
                name='TAX_OFFICE_BY_SEARCH_ARROW',
            )
            logger.info(
                f'Tax office click corrected to search-arrow axis: '
                f'arrow=({arrow_x}, {arrow_tip}), '
                f'target=({arrow_x}, {target_y})'
            )
        if target is None:
            target = self.find_tax_office_icon()
        if target is None:
            return None

        # Keep the confirmed map position for the immediate return from the
        # tax work panel. The search arrow may vanish, but the map stays put.
        self._tax_map_target = target
        self._tax_map_frame = self.device.image.copy()
        self._click_searched_building(target)

        deadline = time.time() + timeout
        clicked_at = time.time()
        retried = False

        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()

            state = self.tax_panel_state()
            if state is not None:
                logger.info('Tax office opened: %s' % state)
                return state

            if (
                not retried
                and time.time() - clicked_at >= 1.5
                and self.is_in_main()
                and self._map_view_unchanged(
                    self._tax_map_frame, self.device.image,
                )
            ):
                # The first tap can only dismiss the search arrow. 龙宫
                # leaves the same roof at the same screen position afterward.
                retried = True
                logger.info(
                    'Tax search arrow disappeared while map stayed in place; '
                    'tap the same office again'
                )
                retry_target = Button(
                    area=(), color=(), button=target.button,
                    name='TAX_OFFICE_SAME_PLACE_RETRY',
                )
                self.device.click(retry_target)
                clicked_at = time.time()

        logger.warning('Tax office panel did not appear')
        self._tax_map_target = None
        self._tax_map_frame = None
        self._save_tax_debug('jiangnan_tax_open_timeout.png')
        return None

    def _tax_lock_score(self, level):
        """在指定等级区域匹配锁图标。"""
        if level not in TAX_LEVEL_LOCK_AREAS:
            raise ValueError(
                'No lock area for tax level %s' % level
            )

        TAX_LEVEL_LOCK_CHECK.ensure_template()
        template = TAX_LEVEL_LOCK_CHECK.image

        area = self.anchored_area(
            TAX_IDLE_CHECK,
            TAX_LEVEL_LOCK_AREAS[level],
        )

        search = crop(
            self.device.image,
            area,
            copy=False,
        )

        if (
            search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return None

        score, _, _, _ = stable_feature_match(
            search, template, gray_weight=0.35
        )
        return float(score)

    def highest_tax_level(self):
        """
        从四级向一级判断最高可选等级。

        返回 1、2、3、4 或 None。
        调用前必须已经处于空闲面板。
        """
        if not self.is_tax_idle():
            return None

        for level in (4, 3, 2):
            score = self._tax_lock_score(level)

            logger.attr(
                'TaxLevel%sLockScore' % level,
                score,
            )

            if score is None:
                return None

            # 明确识别到锁，跳过该等级。
            if score >= self.TAX_LOCK_SIMILARITY:
                continue

            # 分数处于不确定区间，不冒险点击。
            if score > self.TAX_UNLOCKED_MAX_SCORE:
                logger.warning(
                    'Tax level %s lock state is uncertain: %.3f'
                    % (level, score)
                )
                return None

            logger.info(
                'Highest tax level without a lock: %s' % level
            )
            return level

        logger.info(
            'Highest tax level without a lock: 1'
        )
        return 1

    def select_tax_work(self, level=1, timeout=10):
        """
        在税课司空闲面板中固定开启一级四小时工作。

        成功确认开工：返回 1，不再重开建筑读取时间。
        未能确认工作开始：返回 None。
        """
        logger.hr('JIANGNAN SELECT TAX WORK')

        # 1. 确认当前确实是空闲面板。
        self.device.screenshot()

        if not self.is_tax_idle():
            logger.warning('Tax office is not idle')
            return None

        # 自动收获固定开启一级税课司，一级倒计时明确为四小时。
        # 不再自动选择最高级，避免不同等级时长无法可靠排程。
        if level != 1:
            logger.warning('Only tax office level 1 is scheduled automatically')
            return None

        logger.info(f'Selected tax level: {level}')

        # 3. 再截图确认一次，避免界面状态已经变化。
        self.device.sleep(0.3)
        self.device.screenshot()

        if not self.is_tax_idle():
            logger.warning('Tax office is no longer idle')
            return None

        # 4. 根据空闲面板锚点生成动态点击区域。
        # 不直接使用固定屏幕坐标，也不点击“升级”。
        target = self.anchored_button(
            TAX_IDLE_CHECK,
            TAX_LEVEL_BUTTONS[level],
            name=f'TAX_LEVEL_{level}_DYNAMIC',
            inset=4,
        )

        logger.info(f'Click tax work level {level}')

        # 5. 只点击一次。
        started_at = datetime.now().replace(microsecond=0)
        self.device.click(target)

        # 6. 等待工作开始。
        #
        # 游戏可能有两种表现：
        # A. 直接进入“正在工作”面板；
        # B. 返回城市地图，出现“收税”悬挂牌。
        #
        # B 连续确认返回地图即可，不再二次打开税课司确认时间。
        deadline = time.time() + timeout
        main_confirmed = 0

        while time.time() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()

            state = self.tax_panel_state()

            # 情况 A：直接进入工作中界面。
            if state == 'working':
                self.last_tax_started_at = started_at
                logger.info(
                    f'Tax work started: level {level}'
                )
                return level

            # 仍在空闲面板，可能是点击后的短暂动画。
            if state == 'idle':
                main_confirmed = 0
                continue

            # 情况 B：游戏返回城市地图。
            # 连续两次确认主界面，避免在切换动画中立即点击。
            if self.is_in_main():
                main_confirmed += 1

                if main_confirmed < 2:
                    continue

                logger.info(
                    'Tax work returned to main; '
                    'schedule fixed four-hour work without reopening'
                )
                self.last_tax_started_at = started_at
                return level

            main_confirmed = 0

        logger.warning(
            f'Tax work start was not confirmed: level {level}'
        )

        self._save_tax_debug(
            'jiangnan_tax_after_selection.png'
        )

        return None

    def _save_tax_debug(self, filename):
        """保存税课司异常截图，截图失败不覆盖原始错误。"""
        try:
            self.device.image_save(filename)
        except Exception as exc:
            logger.warning(
                f'Unable to save tax screenshot: {exc}'
            )

    def _record_tax_remaining(self):
        """读取工作倒计时，并记录本次读数对应的时间。"""
        self.last_tax_remaining = None
        self.last_tax_remaining_at = None

        # 动画刚结束时文字可能仍有残影。最多读取三帧；一旦得到
        # 合法的完整时分秒就停止，不会用局部数字猜测倒计时。
        for attempt in range(1, 4):
            observed_at = datetime.now().replace(microsecond=0)
            try:
                remaining = self.read_tax_remaining_time()
            except Exception as exc:
                logger.warning(
                    f'Unable to read tax remaining time '
                    f'(attempt {attempt}/3): {exc}'
                )
                remaining = None

            if type(remaining) is int and remaining >= 0:
                self.last_tax_remaining = remaining
                self.last_tax_remaining_at = observed_at
                return remaining

            if attempt < 3:
                self.device.sleep(0.35)
                self.device.screenshot()

        logger.warning('Tax remaining time is unavailable after 3 attempts')
        return None

    def collect_tax(self):
        """处理税课司收益，并在空闲时固定开启一级工作。"""
        logger.hr('JIANGNAN TAX OFFICE')

        # 每次处理一座城市前清空上一次的结果，
        # 防止逐城任务误用上一座城市的倒计时。
        self.last_tax_action = None
        self.last_tax_level = None
        self.last_tax_remaining = None
        self.last_tax_remaining_at = None
        self._tax_map_target = None
        self._tax_map_frame = None
        self.last_tax_started_at = None

        self.device.screenshot()

        # 复用现有税课司收益弹窗处理。
        if self.is_tax_popup():
            if not self.handle_tax_popup():
                return False

            deadline = time.time() + 8

            while time.time() < deadline:
                self.device.screenshot()

                if not self.is_tax_popup():
                    break

                self.device.sleep(0.4)
            else:
                logger.warning('Tax reward popup did not close')
                return False

        # 打开建筑，或识别已经打开的税课司面板。
        state = self.open_tax_office()

        if state == 'working':
            logger.info('Tax office is already working')

            self.last_tax_action = 'working'

            # 读取真实剩余时间。
            # OCR 失败不影响“已经工作”的成功状态。
            self._record_tax_remaining()
            return True

        if state == 'idle':
            # 保留当前固定一级的自动任务策略。
            level = self.select_tax_work(level=1)

            if level is None:
                return False

            self.last_tax_action = 'started'
            self.last_tax_level = level

            # 一级固定四小时；按点击开工时刻计时，不二次打开或 OCR。
            return True

        logger.warning(
            'Tax office state is unknown; no work was selected'
        )

        self._save_tax_debug(
            'jiangnan_tax_state_unknown.png'
        )

        return False
