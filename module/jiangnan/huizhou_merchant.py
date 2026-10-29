import re
import time
from datetime import datetime, timedelta

import cv2

from module.base.button import Button
from module.jiangnan.assets import (
    COUNCIL_HALL_BUSINESS_BUTTON,
    COUNCIL_HALL_INFO_BUTTON,
    COUNCIL_HALL_MAP_ICON,
    COUNCIL_HALL_PANEL_CHECK,
    MERCHANT_ASSIGNEE_CHECK,
    MERCHANT_DEPART_BUTTON,
    MERCHANT_SALE_PANEL_CHECK,
    MERCHANT_WORKING_CHECK,
    REWARD_TAKE_BUTTON,
)
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.jiangnan.ocr import Digit, Ocr
from module.jiangnan.farm_production import JiangnanFarmProduction
from module.jiangnan.vision import crop_map_icon_foreground, stable_feature_map
from module.logger import logger


class JiangnanHuizhouMerchantSale(JiangnanFarmProduction):
    """徽州府四座议事堂的徽商销售调度。"""

    TASK = 'JiangnanHuizhouMerchantSale'
    CITY = '徽州府'
    BUILDING = '议事堂'
    RESOURCES = ('竹子', '茶叶', '桐油', '生漆', '汪笔', '徽墨', '澄纸', '歙石')
    RETRY = timedelta(minutes=10)
    DEFAULT_TIME = datetime(2020, 1, 1)

    # 左右边界各能完整显示七格，中间六格重叠。固定顺序只用于缩小
    # 识别区域，库存数字与黄色垄断令牌仍以当前画面为准。
    LEFT_SLOT_CENTERS = (405, 515, 625, 735, 845, 955, 1065)
    RIGHT_SLOT_CENTERS = (365, 475, 585, 695, 805, 915, 1025)
    TIER_BUTTONS = {
        1: (469, 447, 603, 495),
        2: (611, 447, 744, 495),
        3: (754, 447, 887, 495),
        4: (897, 447, 1032, 495),
    }
    # 建筑搜索选中后的青色箭头位置与方向固定。议事堂定位只把这个
    # 固定尖端当作空间锚点，不再识别箭头像素；首次点击后箭头可能
    # 消失，但仍需继续在同一锚点附近找到厅堂图标并补点。
    COUNCIL_HALL_ARROW_ANCHOR = (640, 280)

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device, task=self.TASK)
        self._merchant_resource_boundary = None
        self._hall_map_anchor = None
        self.working_time_ocr = Ocr(
            Button(
                # 包含绿色时钟后的整行倒计时。给文字上下各留少量余量，
                # 避免面板动画或字体基线有轻微偏移时裁成空白。
                area=(438, 494, 700, 542), color=(),
                button=(438, 494, 700, 542),
                name='MERCHANT_WORKING_TIME',
            ),
            lang='cnocr',
            name='MERCHANT_WORKING_TIME_OCR',
        )
        self.hall_index_ocr = Ocr(
            Button(
                # 搜索议事堂后右侧选择器固定显示“当前/总数”，例如 1/4。
                area=(1015, 225, 1150, 288), color=(),
                button=(1015, 225, 1150, 288),
                name='COUNCIL_HALL_INDEX',
            ),
            lang='cnocr',
            name='COUNCIL_HALL_INDEX_OCR',
        )

    @staticmethod
    def _hall_group(index):
        return f'huizhou_merchant_hall_{index}'

    def _key(self, index, field):
        return f'{self.TASK}.{self._hall_group(index)}.{field}'

    def _get(self, index, field, default=None):
        return self.config.cross_get(self._key(index, field), default=default)

    def _set(self, index, field, value):
        self.config.modified[self._key(index, field)] = value

    def _trade_threshold(self):
        try:
            return max(1, int(self.config.cross_get(
                f'{self.TASK}.huizhou_merchant_settings.TradeThreshold', default=3000,
            )))
        except (TypeError, ValueError):
            return 3000

    def _hall_busy(self, index):
        status = str(self._get(index, 'Status', '') or '')
        return any(marker in status for marker in ('销售中', '外出经商中', '实际时间读取失败', '经商时间识别失败', '收获失败', '收获待确认'))


    def _is_general_panel(self):
        return self.appear(
            COUNCIL_HALL_PANEL_CHECK, offset=(5, 5), similarity=0.82
        )

    def _is_sale_panel(self):
        return self.appear(
            MERCHANT_SALE_PANEL_CHECK, offset=(5, 5), similarity=0.80
        )

    def _is_assignee_panel(self):
        return self.appear(
            MERCHANT_ASSIGNEE_CHECK, offset=(5, 5), similarity=0.80
        )

    def _is_working_panel(self):
        return self.appear(
            MERCHANT_WORKING_CHECK, offset=(5, 5), similarity=0.78
        )

    @staticmethod
    def _select_aligned_hall_candidate(candidates, max_lateral=90):
        """只在搜索箭头正下方的窄列选择议事堂图标。"""
        aligned = [item for item in candidates if item[3] <= max_lateral]
        return min(aligned) if aligned else None

    def _find_centered_hall(self, similarity=0.55, click_offset=(0, 0)):
        """识别当前搜索选中的议事堂，避免把上一座议事堂当成目标。"""
        # 箭头固定为向下，直接使用固定尖端。这里只识别真正的厅堂图标。
        arrow_x, arrow_tip = self.COUNCIL_HALL_ARROW_ANCHOR

        COUNCIL_HALL_MAP_ICON.ensure_template()
        # 四座议事堂位于不同地块，统一裁掉模板地面，只保留灰色厅堂
        # 线条。搜索结果把目标放在箭头正下方；邻近议事堂可能与箭头
        # 直线距离相同，因此先限定横向对齐，再按距离选。
        template = crop_map_icon_foreground(COUNCIL_HALL_MAP_ICON.image)
        x1 = 100
        x2 = 1160
        y1 = min(610, arrow_tip + 12)
        y2 = min(650, arrow_tip + 360)
        if (
            x2 - x1 < template.shape[1]
            or y2 - y1 < template.shape[0]
        ):
            logger.warning(
                f'Council hall directional search area is too small: '
                f'fixed_arrow=({arrow_x}, {arrow_tip}, down), '
                f'area=({x1}, {y1}, {x2}, {y2})'
            )
            return None
        search = self.device.image[y1:y2, x1:x2]
        scores, gray_scores, edge_scores = stable_feature_map(
            search, template, gray_weight=0.32
        )
        height, width = template.shape[:2]
        candidates = []
        for _ in range(24):
            _, score, _, point = cv2.minMaxLoc(scores)
            if score < similarity:
                break
            left, top = point
            cx = x1 + left + width // 2
            cy = y1 + top + height // 2
            forward = cy - arrow_tip
            lateral = abs(cx - arrow_x)
            if not (10 <= forward <= 360):
                cv2.rectangle(
                    scores,
                    (max(0, left - width), max(0, top - height)),
                    (min(scores.shape[1] - 1, left + width),
                     min(scores.shape[0] - 1, top + height)),
                    -1.0,
                    -1,
                )
                continue
            distance = float(
                ((cx - arrow_x) ** 2 + (cy - arrow_tip) ** 2) ** 0.5
            )
            candidates.append((
                distance, -float(score), forward, lateral, cx, cy,
                float(gray_scores[top, left]),
                float(edge_scores[top, left]),
            ))
            cv2.rectangle(
                scores,
                (max(0, left - width), max(0, top - height)),
                (min(scores.shape[1] - 1, left + width),
                 min(scores.shape[0] - 1, top + height)),
                -1.0,
                -1,
            )
        if not candidates:
            logger.warning(
                f'Council hall icon was not recognized in arrow direction: '
                f'fixed_arrow=({arrow_x}, {arrow_tip}, down)'
            )
            return None
        selected = self._select_aligned_hall_candidate(candidates)
        if selected is None:
            logger.warning(
                f'Council hall icons found, but none aligned with search arrow: '
                f'arrow_x={arrow_x}, candidates='
                f'{[(item[4], item[5], item[3]) for item in candidates]}'
            )
            return None
        distance, negative, forward, lateral, cx, cy, gray, edge = selected
        logger.info(
            f'Council hall icon found: center=({cx}, {cy}), '
            f'fixed_arrow=({arrow_x}, {arrow_tip}, down), '
            f'distance={distance:.1f}, forward={forward}, lateral={lateral}, '
            f'stable={-negative:.3f}, gray={gray:.3f}, edge={edge:.3f}'
        )
        click_x = cx + int(click_offset[0])
        click_y = cy + int(click_offset[1])
        return Button(
            area=(), color=(),
            # 先点模板主体中心；补点时由调用方在同一已识别图标内上下
            # 改变落点。小点击框避免随机落到图标外的地面。
            button=(click_x - 4, click_y - 4,
                    click_x + 4, click_y + 4),
            name='COUNCIL_HALL_MAP_ICON_DYNAMIC',
        )

    @staticmethod
    def _hall_button_center(button):
        x1, y1, x2, y2 = button.button
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _visible_selected_hall(self):
        """Only reuse the hall icon at the previously confirmed map location."""
        anchor = self._hall_map_anchor
        if anchor is None:
            return None
        target = self._find_centered_hall()
        if target is None:
            return None
        center = self._hall_button_center(target)
        if max(abs(center[0] - anchor[0]), abs(center[1] - anchor[1])) > 35:
            logger.warning(
                f'Council hall icon moved: expected={anchor}, observed={center}'
            )
            return None
        return target

    def _open_current_hall(self, timeout=10, require_same_location=False):
        self.device.screenshot()
        click_offsets = ((0, 0), (0, 14), (0, -14))
        target = (
            self._visible_selected_hall() if require_same_location
            else self._find_centered_hall(click_offset=click_offsets[0])
        )
        if target is None:
            # 地图模板在某些议事堂地块上频繁失配。保留当前箭头与
            # 图标同屏画面，便于区分被遮挡和模板失配。
            self.save_abnormal_screenshot('council_hall_icon_missing')
            return False
        self._click_searched_building(target, self._find_navigation_buttons())
        click_count = 1
        last_click = time.time()
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            # 到期议事堂可能先弹特殊事件，也可能直接进入“收下”奖励页。
            # 这两种覆盖层出现时，后方的议事堂面板检查不一定可靠。
            if (
                self._find_special_event_ack_button() is not None
                or self.appear(
                    REWARD_TAKE_BUTTON, offset=(5, 5), similarity=0.88
                )
            ):
                return True
            if self._is_general_panel() or self._is_sale_panel():
                return True
            # 箭头会在第一次点击后消失，但其锚点固定。直接重新识别离
            # 固定锚点最近的议事堂，并在同一图标内换落点补点。
            if click_count < 3 and time.time() - last_click >= 2:
                target = (
                    self._visible_selected_hall() if require_same_location
                    else self._find_centered_hall(
                        click_offset=click_offsets[click_count]
                    )
                )
                if target is not None:
                    if require_same_location:
                        x, y = self._hall_button_center(target)
                        dx, dy = click_offsets[click_count]
                        target = Button(
                            area=(), color=(),
                            button=(x + dx - 4, y + dy - 4,
                                    x + dx + 4, y + dy + 4),
                            name='COUNCIL_HALL_MAP_ICON_RETRY_DYNAMIC',
                        )
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry current council hall icon: '
                        f'{click_count}/3, '
                        f'offset={click_offsets[click_count - 1]}'
                    )
                    self.device.click(target)
        logger.warning('Council hall panel did not appear')
        return False

    def _find_special_event_ack_button(self):
        """动态识别徽商特殊事件底部的红色“知道了”按钮。"""
        x1, y1, x2, y2 = (450, 590, 830, 700)
        crop = self.device.image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 75), (13, 255, 255)),
            cv2.inRange(hsv, (168, 80, 75), (179, 255, 255)),
        )
        red = cv2.morphologyEx(
            red,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        )
        contours, _ = cv2.findContours(
            red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if 150 <= width <= 330 and 35 <= height <= 90:
                center_x = x1 + left + width // 2
                center_y = y1 + top + height // 2
                # “知道了”位于页面水平中央；按距离排序可排除边缘 UI。
                distance = abs(center_x - 640) + abs(center_y - 645)
                candidates.append((distance, left, top, width, height))
        if not candidates:
            return None
        _, left, top, width, height = min(candidates)
        return Button(
            area=(), color=(),
            button=(
                x1 + left + 10,
                y1 + top + 6,
                x1 + left + width - 10,
                y1 + top + height - 6,
            ),
            name='MERCHANT_SPECIAL_EVENT_ACK_DYNAMIC',
        )

    def _harvest_completed_sale(self, timeout=18):
        """处理“特殊事件(可选) -> 收下 -> 空闲议事堂”的完整链路。

        Returns:
            tuple[bool, bool]: (流程是否正常, 本轮是否执行了收获)
        """
        deadline = time.time() + timeout
        handled = False
        while time.time() < deadline:
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False, handled

            acknowledge = self._find_special_event_ack_button()
            if acknowledge is not None:
                logger.info('Merchant special event detected; click acknowledge')
                self.device.click(acknowledge)
                self.device.sleep(0.8)
                handled = True
                continue

            if self.appear(
                REWARD_TAKE_BUTTON, offset=(5, 5), similarity=0.88
            ):
                logger.info('Merchant sale reward detected; click take')
                self.device.click(REWARD_TAKE_BUTTON)
                self.device.sleep(1.0)
                handled = True
                continue

            if handled:
                if self._is_general_panel():
                    logger.info('Merchant sale harvest completed')
                    return True, True
                self.device.sleep(0.3)
                continue

            # 普通空闲/经营中议事堂没有需要收获的覆盖层。
            if self._is_general_panel() or self._is_sale_panel():
                return True, False
            self.device.sleep(0.3)

        logger.warning(
            'Merchant harvest did not return to council hall panel: '
            f'handled={handled}'
        )
        return False, handled

    def _close_hall_panel(self):
        self.device.press_back()
        deadline = time.time() + 8
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            if self.is_in_main():
                return True
        return False

    def _navigate_hall(self, forward=True):
        self.device.screenshot()
        buttons = self._find_navigation_buttons()
        if len(buttons) < 2:
            logger.warning('Council hall navigation buttons were not recognized')
            return False
        found = buttons[-1] if forward else buttons[0]
        self.device.click(Button(
            area=(), color=(), button=found.button,
            name=(
                'COUNCIL_HALL_NEXT_DYNAMIC'
                if forward else 'COUNCIL_HALL_PREVIOUS_DYNAMIC'
            ),
        ))
        self.device.sleep(0.8)
        return True

    def _next_hall(self):
        return self._navigate_hall(forward=True)

    def _read_hall_index(self):
        """读取搜索选择器的 n/4，防止配置编号与实际议事堂错位。"""
        raw = str(self.hall_index_ocr.ocr(self.device.image))
        compact = raw.replace(' ', '').replace('／', '/')
        match = re.fullmatch(r'([1-4])/4', compact)
        if match:
            index = int(match.group(1))
            logger.attr('CouncilHallSelector', f'{raw!r} -> {index}/4')
            return index
        logger.warning(f'Council hall selector OCR failed: {raw!r}')
        return None

    def _confirm_hall_index(self, attempts=4, expected=None):
        """两帧读到预期编号后才认为搜索选择器已经稳定。"""
        previous = None
        for attempt in range(attempts):
            self.device.screenshot()
            observed = self._read_hall_index()
            if observed is not None and observed == previous \
                    and (expected is None or observed == expected):
                return observed
            previous = observed
            if attempt < attempts - 1:
                self.device.sleep(0.2)
        return None

    def _select_hall(self, target):
        """选择严格对应 target/4 的议事堂。

        不根据搜索初始编号或点击次数猜测当前编号；读不到稳定的实际
        编号时停止，避免把另一座议事堂的销售记录写到目标编号下。
        """
        target = max(1, min(4, int(target)))
        current = self._confirm_hall_index()
        if current is None:
            logger.warning(
                f'Unable to confirm current council hall before selecting '
                f'{target}/4'
            )
            return False
        for _ in range(4):
            if current == target:
                logger.info(f'Council hall selected: {target}/4')
                return True
            forward = current < target
            if not self._navigate_hall(forward=forward):
                return False
            expected = current + (1 if forward else -1)
            observed = self._confirm_hall_index(expected=expected)
            if observed is None:
                logger.warning(
                    f'Council hall index unconfirmed after navigation: '
                    f'previous={current}/4, expected={expected}/4, '
                    f'target={target}/4'
                )
                return False
            current = observed
        logger.warning(
            f'Council hall selection exceeded navigation limit: target={target}/4'
        )
        return False

    def _search_hall_selector(self, target):
        """定位目标 n/4，清理该区域干扰后保留可点击的目标状态。

        正确顺序是搜索议事堂、切换并确认目标 n/4、扫描当前屏幕的
        中国结/剪刀，最后才沿箭头寻找议事堂。处理事件后先确认原
        图标仍在同一位置；地图确实改变时才重新搜索和选择编号。
        """
        target = max(1, min(4, int(target)))
        search_kwargs = dict(
            result_validator=lambda value: self.BUILDING in str(value),
            exact=False,
            # 必须先切到目标 n/4，再在目标所在区域处理干扰。
            handle_interference=False,
        )
        def position_target():
            self.device.screenshot()
            if self._visible_selected_hall() is not None:
                logger.info(
                    f'Council hall {target}/4 is still at the selected location; '
                    'reuse current map'
                )
                return True
            if not self.search_building(self.BUILDING, **search_kwargs):
                return False

            # 读取搜索选择器的实际编号，再切换到要进入的议事堂。
            if not self._select_hall(target):
                return False
            logger.info(
                f'Council hall target positioned: {target}/4'
            )
            self.device.screenshot()
            icon = self._find_centered_hall()
            self._hall_map_anchor = (
                self._hall_button_center(icon) if icon is not None else None
            )
            return True

        self._hall_map_anchor = None
        if not position_target():
            return False
        return self.ensure_building_search_area_clear(
            f'{self.BUILDING} {target}/4',
            reposition=position_target,
        )

    def _goto_sale_panel(self):
        if self._is_sale_panel():
            # 新打开的议事堂可能沿用游戏上次的横向位置，不能继承脚本
            # 上一座议事堂记录的边界状态。
            self._merchant_resource_boundary = None
            return True
        if not self._is_general_panel():
            return False
        self.device.click(COUNCIL_HALL_BUSINESS_BUTTON)
        deadline = time.time() + 8
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self._is_sale_panel():
                self._merchant_resource_boundary = None
                return True
        return False

    def _goto_information_panel(self):
        if self._is_general_panel():
            return True
        if not self._is_sale_panel():
            return False
        self.device.click(COUNCIL_HALL_INFO_BUTTON)
        deadline = time.time() + 8
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self._is_general_panel():
                return True
        return False

    def _drag_to_left_boundary(self):
        if self._merchant_resource_boundary == 'left':
            return
        # 八格货物只有一格宽度超出视口，一次 500px 拖动一定能抵达
        # 边界。旧逻辑固定拖两遍，并在已位于边界时继续空拖。
        self.jiangnan_drag(
            (455, 282), (955, 282),
            name='MERCHANT_GO_LEFT', hold_duration=0.2,
        )
        self.device.sleep(0.35)
        self._merchant_resource_boundary = 'left'

    def _drag_to_right_boundary(self):
        if self._merchant_resource_boundary == 'right':
            return
        self.jiangnan_drag(
            (955, 282), (455, 282),
            name='MERCHANT_GO_RIGHT', hold_duration=0.2,
        )
        self.device.sleep(0.35)
        self._merchant_resource_boundary = 'right'

    @staticmethod
    def _has_monopoly_badge(image, center_x):
        crop = image[215:285, max(0, center_x - 62):center_x - 12]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(hsv, (16, 105, 105), (39, 255, 255))
        count, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
        return any(
            180 <= int(area) <= 2200 and 10 <= int(width) <= 48
            and 14 <= int(height) <= 65
            for _, _, width, height, area in stats[1:count]
        )

    def _read_slot_inventory(self, center_x, resource):
        area = (center_x - 48, 323, center_x + 48, 362)
        value = Digit(
            Button(area=area, color=(), button=area, name=f'MERCHANT_{resource}'),
            lang='azur_lane', letter=(54, 47, 43), threshold=180,
            name=f'MERCHANT_{resource}_INVENTORY',
        ).ocr(self.device.image)
        value = int(value)
        return value if 0 <= value <= 9999999 else -1

    def _scan_view(self, start_index, centers):
        slots = []
        buttons = []
        for offset, center_x in enumerate(centers):
            resource_index = start_index + offset
            if resource_index >= len(self.RESOURCES):
                break
            resource = self.RESOURCES[resource_index]
            area = (center_x - 48, 323, center_x + 48, 362)
            slots.append((resource_index, center_x, resource))
            buttons.append(Button(
                area=area, color=(), button=area,
                name=f'MERCHANT_{resource}',
            ))
        values = Digit(
            buttons,
            lang='azur_lane', letter=(54, 47, 43), threshold=180,
            name='MERCHANT_VIEW_INVENTORY_BATCH',
        ).ocr(self.device.image) if buttons else []
        if not isinstance(values, list):
            values = [values]

        readings = {}
        monopolies = set()
        for position, (resource_index, center_x, _) in enumerate(slots):
            try:
                value = int(values[position])
            except (IndexError, TypeError, ValueError):
                value = -1
            readings[resource_index] = (
                value if 0 <= value <= 9999999 else -1
            )
            if self._has_monopoly_badge(self.device.image, center_x):
                monopolies.add(resource_index)
        return readings, monopolies

    def _scan_resources(self):
        self._drag_to_left_boundary()
        self.device.screenshot()
        left_values, left_monopolies = self._scan_view(0, self.LEFT_SLOT_CENTERS)

        self._drag_to_right_boundary()
        self.device.screenshot()
        right_values, right_monopolies = self._scan_view(1, self.RIGHT_SLOT_CENTERS)

        # 左页最后一格会被右边框截断，右页第一格会被左边框截断。
        # 左页采信 0..5，右页采信 6..7；中间完整重叠格只校验。
        values = {
            index: value for index, value in left_values.items()
            if index <= 5 and value >= 0
        }
        values.update({
            index: value for index, value in right_values.items()
            if index >= 6 and value >= 0
        })
        for index in range(2, 6):
            old = left_values.get(index, -1)
            value = right_values.get(index, -1)
            if value >= 0 and old >= 0 and value != old:
                logger.warning(
                    f'Merchant inventory mismatch for {self.RESOURCES[index]}: '
                    f'left={old}, right={value}; keep boundary-safe reading'
                )
        monopolies = left_monopolies | right_monopolies
        named = {
            self.RESOURCES[index]: value
            for index, value in values.items() if value >= 0
        }
        named_monopolies = [self.RESOURCES[index] for index in sorted(monopolies)]
        logger.attr('MerchantInventories', named)
        logger.attr('MerchantMonopolies', named_monopolies)
        return named, named_monopolies

    def _write_huizhou_inventory(self, readings, state):
        if not readings:
            return
        state.reconcile_live_inventory(self.CITY, readings, reason='徽商销售现场库存')

    def _configured_monopolies(self, hall_index):
        result = []
        for index in range(1, 5):
            value = str(self._get(hall_index, f'Monopoly{index}', 'none'))
            if value in self.RESOURCES and value not in result:
                result.append(value)
        return result

    def _update_monopolies(self, hall_index, resources):
        values = list(resources[:4]) + ['none'] * (4 - len(resources[:4]))
        for index, value in enumerate(values, start=1):
            self._set(hall_index, f'Monopoly{index}', value)
        self.config.update()

    def _show_resource(self, resource_index):
        # 扫描结束时停在右边界；最右优先的正常候选都在这里。索引 0
        # 需要回左边界，其余索引使用右边界的完整格位。
        if resource_index == 0:
            self._drag_to_left_boundary()
            self.device.screenshot()
            center_x = self.LEFT_SLOT_CENTERS[0]
        else:
            self._drag_to_right_boundary()
            self.device.screenshot()
            center_x = self.RIGHT_SLOT_CENTERS[resource_index - 1]
        self.device.click(Button(
            area=(), color=(),
            button=(center_x - 38, 240, center_x + 38, 323),
            name=f'MERCHANT_RESOURCE_{self.RESOURCES[resource_index]}',
        ))
        self.device.sleep(0.35)

    def _read_tier_amount(self, tier):
        area = self.TIER_BUTTONS[tier]
        value = Digit(
            Button(area=area, color=(), button=area, name=f'MERCHANT_TIER_{tier}'),
            lang='azur_lane', letter=(245, 235, 218), threshold=160,
            name=f'MERCHANT_TIER_{tier}_AMOUNT',
        ).ocr(self.device.image)
        value = int(value)
        return value if 0 < value <= 9999999 else None

    def _read_tier_amounts(self):
        tiers = sorted(self.TIER_BUTTONS)
        buttons = [
            Button(
                area=self.TIER_BUTTONS[tier], color=(),
                button=self.TIER_BUTTONS[tier],
                name=f'MERCHANT_TIER_{tier}',
            )
            for tier in tiers
        ]
        values = Digit(
            buttons,
            lang='azur_lane', letter=(245, 235, 218), threshold=160,
            name='MERCHANT_TIER_AMOUNTS_BATCH',
        ).ocr(self.device.image)
        if not isinstance(values, list):
            values = [values]
        amounts = {}
        for index, tier in enumerate(tiers):
            try:
                value = int(values[index])
            except (IndexError, TypeError, ValueError):
                value = -1
            amounts[tier] = value if 0 < value <= 9999999 else None
        logger.attr('MerchantSaleTierAmounts', amounts)
        return amounts

    def _update_tier_requirement(self, hall_index, tier, amount):
        """保存 UI 当前挡位数量，供下次运行在进城前检查库存。"""
        try:
            tier = int(tier)
            amount = int(amount)
        except (TypeError, ValueError):
            return False
        if tier not in self.TIER_BUTTONS or amount <= 0:
            return False
        old_tier = int(self._get(hall_index, 'RecognizedSaleTier', 0) or 0)
        old_amount = int(self._get(hall_index, 'TierRequiredAmount', 0) or 0)
        if old_tier == tier and old_amount == amount:
            return True
        self._set(hall_index, 'RecognizedSaleTier', tier)
        self._set(hall_index, 'TierRequiredAmount', amount)
        self.config.update()
        logger.info(
            f'Merchant tier requirement updated: hall={hall_index}, '
            f'tier={tier}, amount={amount}'
        )
        return True

    def _known_tier_requirement(self, hall_index):
        """只返回与当前 UI 挡位一致的已识别数量。"""
        try:
            tier = int(self._get(hall_index, 'SaleTier', 4) or 4)
            recognized = int(
                self._get(hall_index, 'RecognizedSaleTier', 0) or 0
            )
            amount = int(
                self._get(hall_index, 'TierRequiredAmount', 0) or 0
            )
        except (TypeError, ValueError):
            return None
        return amount if recognized == tier and amount > 0 else None

    def _can_skip_hall_for_inventory(self, hall_index, state):
        """库存确定全部不足时，无需进城和打开议事堂。"""
        # 开启降档后，高挡位库存不足仍可能销售较低挡位。
        if self._hall_busy(hall_index):
            return False
        amount = self._trade_threshold()
        monopolies = self._configured_monopolies(hall_index) or list(self.RESOURCES)
        inventory = state.load_inventories((self.CITY,)).get(self.CITY, {})
        readings = {}
        for resource in monopolies:
            try:
                available = int(inventory.get(resource, -1))
            except (TypeError, ValueError):
                return False
            if available < 0:
                return False
            readings[resource] = available
        insufficient = all(value < amount for value in readings.values())
        logger.info(
            f'Merchant inventory precheck: hall={hall_index}, '
            f'amount={amount}, inventories={readings}, skip={insufficient}'
        )
        return insufficient

    def _select_sale(self, hall_index, inventories, monopolies):
        preferred_tier = int(self._get(hall_index, 'SaleTier', 4) or 4)
        preferred_tier = max(1, min(4, preferred_tier))
        allow_downgrade = bool(
            self._get(hall_index, 'AllowTierDowngrade', False)
        )
        allowed = set(monopolies)
        if not allowed:
            logger.warning(f'Council hall {hall_index} has no configured monopoly goods')
            return None, None, 'no_monopoly'
        confirmed_insufficient = 0
        candidates = 0
        best = None
        displayed_resource_index = None
        threshold = self._trade_threshold()
        for resource_index in sorted(range(len(self.RESOURCES)), key=lambda i: (int(inventories.get(self.RESOURCES[i], -1)), i), reverse=True):
            resource = self.RESOURCES[resource_index]
            if resource not in allowed:
                continue
            candidates += 1
            if 0 <= int(inventories.get(resource, -1)) < threshold:
                confirmed_insufficient += 1
                continue
            self._show_resource(resource_index)
            displayed_resource_index = resource_index
            self.device.screenshot()
            tier_amounts = self._read_tier_amounts()
            preferred_amount = tier_amounts.get(preferred_tier)
            if preferred_amount is not None:
                self._update_tier_requirement(
                    hall_index, preferred_tier, preferred_amount)
            available = int(inventories.get(resource, -1))
            affordable = next((
                (sale_tier, tier_amounts[sale_tier])
                for sale_tier in range(
                    preferred_tier,
                    0 if allow_downgrade else preferred_tier - 1,
                    -1,
                )
                if available >= threshold and tier_amounts.get(sale_tier) is not None
                and available >= tier_amounts[sale_tier]
            ), None)
            logger.info(
                f'Merchant candidate: hall={hall_index}, resource={resource}, '
                f'preferred_tier={preferred_tier}, '
                f'preferred_amount={preferred_amount}, inventory={available}, '
                f'affordable={affordable}'
            )
            minimum_tier = 1 if allow_downgrade else preferred_tier
            if available >= 0 and tier_amounts.get(minimum_tier) is not None \
                    and available < tier_amounts[minimum_tier]:
                confirmed_insufficient += 1
            if affordable is not None:
                sale_tier, amount = affordable
                if best is None or sale_tier > best[0]:
                    best = (sale_tier, resource_index, amount)
                if sale_tier == preferred_tier:
                    break
        if best is not None:
            sale_tier, resource_index, amount = best
            resource = self.RESOURCES[resource_index]
            # 最优货物已显示时直接点；切回之前检查的货物则复核数量。
            if displayed_resource_index != resource_index:
                self._show_resource(resource_index)
                self.device.screenshot()
                confirmed_amount = self._read_tier_amounts().get(sale_tier)
                if confirmed_amount != amount:
                    logger.warning(
                        f'Merchant tier amount changed before selection: '
                        f'{resource}, tier={sale_tier}, '
                        f'{amount} -> {confirmed_amount}'
                    )
                    return None, None, 'unreadable'
            logger.info(
                f'Merchant sale selected: hall={hall_index}, '
                f'resource={resource}, tier={sale_tier}, amount={amount}'
            )
            self.device.click(Button(
                area=(), color=(), button=self.TIER_BUTTONS[sale_tier],
                name=f'MERCHANT_SALE_TIER_{sale_tier}',
            ))
            self.device.sleep(0.3)
            return resource, amount, 'selected'
        if candidates and confirmed_insufficient == candidates:
            return None, None, 'insufficient'
        return None, None, 'unreadable'

    def _depart_and_assign(self):
        self.device.click(MERCHANT_DEPART_BUTTON)
        deadline = time.time() + 12
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self._is_assignee_panel():
                break
        else:
            logger.warning('Merchant assignee page did not appear')
            return False
        # 第一位人物可能正被占用；实机失败截图中点了蓝边第一位后
        # 选人页始终未关闭，旁边的金边人物可供尝试。每次只点头像内部，
        # 并确认仍在选人页后才换人，避免过渡时误点其他界面。
        for x, y in self._merchant_assignee_candidates():
            self.device.click(Button(
                area=(), color=(),
                button=(x - 12, y - 12, x + 12, y + 12),
                name=f'MERCHANT_ASSIGNEE_{x}_{y}',
            ))
            deadline = time.time() + 3
            while time.time() < deadline:
                self.device.sleep(0.35)
                self.device.screenshot()
                if self.handle_network_timeout_popup():
                    return False
                if self.is_in_main():
                    return True
            if not self._is_assignee_panel():
                break
        deadline = time.time() + 5
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False
            if self.is_in_main():
                return True
        logger.warning('Merchant did not return to map after assigning person')
        return False

    def _merchant_assignee_candidates(self):
        """按头像上沿的金色边框优先排列当前可见人物。"""
        image = self.device.image
        candidates = []
        for x in (390, 770):
            ribbon = image[74:83, x - 55:x + 45]
            if ribbon.size == 0:
                continue
            hsv = cv2.cvtColor(ribbon, cv2.COLOR_RGB2HSV)
            gold = cv2.inRange(hsv, (12, 80, 55), (40, 255, 255))
            ratio = cv2.countNonZero(gold) / gold.size
            candidates.append((ratio, x, 140))
        if not candidates:
            return [(390, 140)]
        candidates.sort(key=lambda item: item[0], reverse=True)
        logger.attr(
            'MerchantAssigneeBorders',
            [(x, round(ratio, 3)) for ratio, x, _ in candidates],
        )
        return [(x, y) for _, x, y in candidates]

    def _read_working_duration(self, attempts=4):
        if not self._goto_information_panel():
            logger.warning('Merchant working information was not recognized')
            return None

        # 议事堂地图面板的纸张和固定文字会先出现，经营倒计时稍后才
        # 完成绘制。过去在补点打开 2/4 后约 0.2 秒便读取，容易得到
        # 空字符串。等待新帧并有限重试，只重读倒计时区域，不重搜建筑。
        for attempt in range(1, max(1, int(attempts)) + 1):
            self.device.sleep(0.45 if attempt == 1 else 0.30)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return None
            if not self._is_general_panel():
                logger.warning(
                    'Merchant working panel changed before duration OCR: '
                    f'attempt={attempt}/{attempts}'
                )
                return None
            if not self._is_working_panel():
                logger.info(
                    'Merchant working marker is not stable yet: '
                    f'attempt={attempt}/{attempts}'
                )
                continue
            raw = self.working_time_ocr.ocr(self.device.image)
            duration = self.parse_duration(str(raw))
            logger.attr(
                'MerchantActualWorkingTime',
                f'attempt={attempt}/{attempts}, {raw!r} -> {duration}',
            )
            if duration is not None:
                return duration
        logger.warning(
            f'Merchant working duration OCR failed after {attempts} attempts'
        )
        return None

    def _relocate_hall(self, hall_index, try_current=True):
        if try_current and self._open_current_hall(require_same_location=True):
            return True
        if not self._search_hall_selector(hall_index):
            return False
        return self._open_current_hall()

    def _inspect_hall(self, hall_index, state):
        now = datetime.now().replace(microsecond=0)
        if not self._open_current_hall():
            logger.warning(
                f'Council hall {hall_index}/4 did not open; '
                're-search this hall once'
            )
            if not self._relocate_hall(hall_index, try_current=False):
                return now + self.RETRY, '[打开议事堂失败][等待复查]'

        harvest_ok, harvested = self._harvest_completed_sale()
        if not harvest_ok:
            self._close_hall_panel()
            return now + self.RETRY, '[徽商收获失败][等待复查]'
        if harvested:
            logger.info(
                f'Council hall {hall_index}/4 harvested; continue next sale'
            )

        # 已经有徽商在外时只读取实际剩余时间，不进入销售流程。
        if self._is_general_panel() and self._is_working_panel():
            duration = self._read_working_duration()
            if duration:
                self._close_hall_panel()
                finish = datetime.now().replace(microsecond=0) + duration + timedelta(minutes=1)
                return finish, f'[销售中][外出经商中][{self._time_text(finish)}]'
            # 面板动画期间的固定背景偶尔会让“经营中”模板瞬时命中。
            # 重读后若已稳定为空闲页，继续经商流程；真正仍在经营中但
            # 倒计时无法识别时才进入重试，避免 2/4 被永久当作读取失败。
            if self._is_general_panel() and not self._is_working_panel():
                logger.warning(
                    f'Council hall {hall_index}/4 working state was transient; '
                    'continue as idle hall'
                )
            else:
                self._close_hall_panel()
                return now + self.RETRY, '[经商时间识别失败][等待复查]'

        if not self._goto_sale_panel():
            self._close_hall_panel()
            return now + self.RETRY, '[销售界面识别失败][等待复查]'

        inventories, detected = self._scan_resources()
        self._write_huizhou_inventory(inventories, state)
        if bool(self._get(hall_index, 'AutoDetectMonopoly', True)):
            self._update_monopolies(hall_index, detected)
            monopolies = detected
        else:
            monopolies = self._configured_monopolies(hall_index)

        resource, amount, selection_status = self._select_sale(
            hall_index, inventories, monopolies
        )
        if resource is None:
            self._close_hall_panel()
            if selection_status == 'insufficient':
                return self.DEFAULT_TIME, '[库存未达交易阈值或挡位需求][等待物资]'
            return now + self.RETRY, '[库存不足或无垄断货物][等待复查]'
        if not self._depart_and_assign():
            return now + self.RETRY, f'[{resource}][出发失败][等待复查]'

        # 选择人物并返回地图即代表销售已经成立，库存必须立刻扣除；
        # 后续重新打开信息页失败也不能漏记本次支出。
        if not state.consume_inventory(self.CITY, {resource: int(amount)}, event_type='merchant_sale',
                                       reason=f'议事堂{hall_index}徽商销售'):
            state.request_scan('merchant departure stock deduction failed', cities={self.CITY})

        # 人物会缩短经营时间，只有出发后重新进入信息页读到的时间准确。
        # 回到地图后先核对先前确认的图标位置；仍在原处就直接打开。
        # 只有图标确实移动或无法识别时才重新搜索并选择同一编号。
        if not self._relocate_hall(hall_index, try_current=True):
            return now + self.RETRY, f'[{resource}][实际时间读取失败][等待复查]'
        duration = self._read_working_duration()
        self._close_hall_panel()
        if duration is None:
            return now + self.RETRY, f'[{resource}][实际时间读取失败][等待复查]'

        finish = datetime.now().replace(microsecond=0) + duration + timedelta(minutes=1)
        return finish, (
            f'[{resource}][-{amount}][{self._time_text(finish)}][销售中]'
        )

    def run(self):
        logger.hr('JIANGNAN HUIZHOU MERCHANT SALE', level=0)
        now = datetime.now().replace(microsecond=0)
        state = JiangnanInventoryState(self.config)
        schedules = []
        for index in range(1, 5):
            if not bool(self._get(index, 'Enable', True)):
                continue
            next_run = self._parse_time(self._get(index, 'NextRun')) or self.DEFAULT_TIME
            status = str(self._get(index, 'Status', '') or '')
            # An idle hall has no sale-start clock. Only an in-progress sale
            # or a failed operation awaiting retry keeps a future check time.
            if not self._hall_busy(index) and '等待复查' not in status:
                next_run = self.DEFAULT_TIME
                self._set(index, 'NextRun', next_run)
            schedules.append((next_run, index))

        # Hall clocks describe availability only. Stock controls whether an
        # idle hall becomes runnable; busy due halls are always checked/collected.
        waiting = set()
        for next_run, index in schedules:
            if self._can_skip_hall_for_inventory(index, state):
                waiting.add(index)
                self._set(index, 'NextRun', self.DEFAULT_TIME)
                self._set(index, 'Status', '[议事堂空闲][库存未达交易阈值或挡位需求][等待物资]')
        due = sorted(
            ((next_run, index) for next_run, index in schedules
             if index not in waiting and next_run <= now),
            key=lambda item: (not self._hall_busy(item[1]), item[0], item[1]),
        )
        handled_index = None
        handled_next_run = None
        if due:
            _, handled_index = due[0]
            was_busy = self._hall_busy(handled_index)
            logger.info(f'Run council hall {handled_index}/4 independently')
            if not self.switch_city(self.CITY):
                handled_next_run = now + self.RETRY
                self._set(handled_index, 'NextRun', handled_next_run)
                self._set(
                    handled_index, 'Status', '[进入徽州府失败][等待复查]'
                )
            else:
                retry = datetime.now().replace(microsecond=0) + self.RETRY
                if not self._search_hall_selector(handled_index):
                    handled_next_run = retry
                    self._set(handled_index, 'NextRun', handled_next_run)
                    self._set(
                        handled_index, 'Status', '[未找到议事堂][等待复查]'
                    )
                else:
                    handled_next_run, status = self._inspect_hall(
                        handled_index, state
                    )
                    self._set(handled_index, 'NextRun', handled_next_run)
                    self._set(handled_index, 'Status', status)
                self.ui_goto_main()

            # Failure to locate/open a working hall must not erase its pending
            # harvest and turn it into an idle hall gated by stock next time.
            if was_busy:
                status = str(self.config.modified.get(self._key(handled_index, 'Status'), ''))
                if '等待复查' in status:
                    self._set(handled_index, 'Status', status + '[收获待确认]')

        self.config.update()
        next_runs = []
        for next_run, index in schedules:
            if index in waiting:
                continue
            if index == handled_index and handled_next_run is not None:
                if handled_next_run == self.DEFAULT_TIME:
                    continue  # Available, but no qualifying goods after live scan.
                next_runs.append(handled_next_run)
            else:
                next_runs.append(next_run)
        # Periodic stock fallback belongs to the task clock, never hall availability.
        self.config.task_delay(target=min(next_runs) if next_runs else now+timedelta(hours=1))
        return True
