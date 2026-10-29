import re
import time
from datetime import datetime, timedelta

import cv2

from module.base.button import Button
from module.jiangnan.assets import (
    MERCHANT_ASSIGNEE_CHECK,
    NINGBO_SHIP_ASSIGNEE_CONFIRM,
)
from module.jiangnan.farm_production import JiangnanFarmProduction
from module.jiangnan.foundation_icons import find_foundation_icon
from module.jiangnan.inventory_state import ALL_RESOURCES, JiangnanInventoryState
from module.jiangnan.ocr import Digit, Ocr
from module.logger import logger


class JiangnanNingboMerchantShip(JiangnanFarmProduction):
    """宁波府四艘商船的独立订单、收获和计时。"""

    TASK = 'JiangnanNingboMerchantShip'
    CITY = '宁波府'
    BUILDING = '商船'
    DEFAULT_TIME = datetime(2020, 1, 1)
    RETRY = timedelta(minutes=10)

    SELECTOR_INDEX_AREA = (1010, 220, 1155, 285)
    SHIP_NAME_AREA = (405, 75, 625, 125)
    IDLE_HEADING_AREA = (790, 135, 1015, 178)
    WORKING_TIME_AREA = (640, 590, 905, 650)
    # 商船搜索箭头只用于定位船位；状态必须打开面板后读取。

    ORDER_AREAS = {
        'top': {
            'card': (710, 80, 1085, 325),
            'title': (720, 80, 1080, 125),
            'dried_fish': (775, 175, 875, 215),
            'cargo': (900, 175, 1000, 215),
            'time': (790, 215, 905, 260),
            'people': (940, 215, 1020, 260),
            'coin': (850, 275, 945, 320),
            'tool': (965, 275, 1060, 320),
        },
        'bottom': {
            'card': (710, 340, 1085, 585),
            'title': (720, 340, 1080, 385),
            'dried_fish': (775, 435, 875, 475),
            'cargo': (900, 435, 1000, 475),
            'time': (790, 475, 905, 520),
            'people': (940, 475, 1020, 520),
            'coin': (850, 535, 945, 580),
            'tool': (965, 535, 1060, 580),
        },
    }

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device, task=self.TASK)
        self.selector_ocr = Ocr(
            Button(
                area=self.SELECTOR_INDEX_AREA, color=(),
                button=self.SELECTOR_INDEX_AREA,
                name='NINGBO_SHIP_SELECTOR_INDEX',
            ),
            lang='cnocr',
            name='NINGBO_SHIP_SELECTOR_INDEX_OCR',
        )
        self.ship_name_ocr = Ocr(
            Button(
                area=self.SHIP_NAME_AREA, color=(),
                button=self.SHIP_NAME_AREA,
                name='NINGBO_SHIP_NAME',
            ),
            lang='cnocr',
            name='NINGBO_SHIP_NAME_OCR',
        )
        self.idle_heading_ocr = Ocr(
            Button(
                area=self.IDLE_HEADING_AREA, color=(),
                button=self.IDLE_HEADING_AREA,
                name='NINGBO_SHIP_IDLE_HEADING',
            ),
            lang='cnocr',
            name='NINGBO_SHIP_IDLE_HEADING_OCR',
        )
        self.working_time_ocr = Ocr(
            Button(
                area=self.WORKING_TIME_AREA, color=(),
                button=self.WORKING_TIME_AREA,
                name='NINGBO_SHIP_WORKING_TIME',
            ),
            lang='cnocr',
            name='NINGBO_SHIP_WORKING_TIME_OCR',
        )

    @staticmethod
    def _group(index):
        return f'ningbo_merchant_ship_{int(index)}'

    def _key(self, index, field):
        return f'{self.TASK}.{self._group(index)}.{field}'

    def _get(self, index, field, default=None):
        return self.config.cross_get(
            self._key(index, field), default=default
        )

    def _set(self, index, field, value):
        self.config.modified[self._key(index, field)] = value

    @staticmethod
    def _clean_text(value):
        return re.sub(r'\s+', '', str(value or ''))

    def _ocr_text(self, area, name):
        reader = Ocr(
            Button(area=area, color=(), button=area, name=name),
            lang='cnocr',
            name=f'{name}_OCR',
        )
        return self._clean_text(reader.ocr(self.device.image))

    def _ocr_number(self, area, name):
        value = Digit(
            Button(area=area, color=(), button=area, name=name),
            lang='azur_lane',
            name=f'{name}_OCR',
        ).ocr(self.device.image)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _read_selector_index(self):
        raw = self._clean_text(self.selector_ocr.ocr(self.device.image))
        raw = raw.replace('／', '/')
        matched = re.search(r'([1-4])/4', raw)
        if matched:
            index = int(matched.group(1))
            logger.attr('NingboShipSelector', f'{raw!r} -> {index}/4')
            return index
        logger.warning(f'Ningbo ship selector OCR failed: {raw!r}')
        return None

    def _navigate_ship(self, forward=True):
        self.device.screenshot()
        buttons = self._find_navigation_buttons()
        if len(buttons) < 2:
            logger.warning('Ningbo ship navigation buttons were not recognized')
            return False
        target = buttons[-1] if forward else buttons[0]
        self.device.click(Button(
            area=(), color=(), button=target.button,
            name=(
                'NINGBO_SHIP_NEXT_DYNAMIC'
                if forward else 'NINGBO_SHIP_PREVIOUS_DYNAMIC'
            ),
        ))
        self.device.sleep(0.8)
        return True

    def _select_ship(self, target, assumed_current=1):
        target = max(1, min(4, int(target)))
        self.device.screenshot()
        current = self._read_selector_index() or assumed_current
        for _ in range(4):
            if current == target:
                logger.info(f'Ningbo merchant ship selected: {target}/4')
                return True
            forward = current < target
            if not self._navigate_ship(forward=forward):
                return False
            expected = current + (1 if forward else -1)
            self.device.screenshot()
            current = self._read_selector_index() or expected
        logger.warning(f'Unable to select Ningbo merchant ship {target}/4')
        return False

    def _search_ship_selector(self, target):
        target = max(1, min(4, int(target)))
        search_kwargs = dict(
            result_validator=lambda value: self.BUILDING in str(value),
            exact=False,
            handle_interference=False,
        )

        def position_target():
            if not self.search_building(self.BUILDING, **search_kwargs):
                return False
            if not self._select_ship(target, assumed_current=1):
                return False
            logger.info(f'Ningbo ship target positioned: {target}/4')
            return True

        if not position_target():
            return False
        return self.ensure_building_search_area_clear(
            f'{self.BUILDING} {target}/4',
            reposition=position_target,
            # A "完成" flag on a Ningbo ship opens its own voyage scroll.
            # Generic city production harvesting cannot finish that scroll
            # and otherwise keeps searching the ship before we can inspect it.
            harvest_completions=False,
        )

    def _find_action_areas(self):
        """Find lower-page red action rectangles without running OCR."""
        x1, y1, x2, y2 = (430, 500, 1090, 700)
        image = self.device.image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 70), (15, 255, 255)),
            cv2.inRange(hsv, (165, 70, 70), (179, 255, 255)),
        )
        red = cv2.morphologyEx(
            red,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        )
        contours, _ = cv2.findContours(
            red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        areas = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (90 <= width <= 360 and 32 <= height <= 100):
                continue
            area = (
                x1 + left, y1 + top,
                x1 + left + width, y1 + top + height,
            )
            areas.append(area)
        return areas

    def _parse_action_buttons(self, areas, texts):
        result = []
        for area, raw in zip(areas, texts):
            text = self._clean_text(raw)
            if '完成' not in text and '收下' not in text:
                continue
            result.append((
                text,
                Button(
                    area=(), color=(),
                    button=(area[0] + 8, area[1] + 5,
                            area[2] - 8, area[3] - 5),
                    name=f'NINGBO_SHIP_ACTION_{text}',
                ),
            ))
        return result

    def _find_action_buttons(self):
        """Find lower-page red buttons used by 完成/收下."""
        areas = self._find_action_areas()
        if not areas:
            return []
        buttons = [
            Button(area=area, color=(), button=area,
                   name=f'NINGBO_SHIP_ACTION_{index + 1}')
            for index, area in enumerate(areas)
        ]
        texts = Ocr(
            buttons, lang='cnocr', name='NINGBO_SHIP_ACTION_BATCH'
        ).ocr(self.device.image)
        if not isinstance(texts, list):
            texts = [texts]
        return self._parse_action_buttons(areas, texts)

    def _read_working_duration(self):
        raw = self._clean_text(self.working_time_ocr.ocr(self.device.image))
        duration = self.parse_duration(raw)
        logger.attr('NingboShipWorkingTime', f'{raw!r} -> {duration}')
        return duration

    def _no_assignee_toast(self):
        area = (375, 215, 945, 257)
        text = self._clean_text(Ocr(
            Button(area=area, color=(), button=area,
                   name='NINGBO_SHIP_NO_ASSIGNEE_TOAST'),
            lang='cnocr', name='NINGBO_SHIP_NO_ASSIGNEE_TOAST',
        ).ocr(self.device.image))
        return '无空闲特殊居民' in text and '工作' in text

    def _next_assignee_release(self):
        now = datetime.now().replace(microsecond=0)
        candidates = []
        # Only an underway merchant ship is known to occupy a special
        # resident. A normal production completion may free ordinary workers
        # and must not be treated as proof that a ship captain is available.
        for index in range(1, 5):
            if '探索中' not in str(self._get(index, 'Status', '')):
                continue
            finish = self._parse_time(self._get(index, 'NextRun'))
            if finish is not None and finish > now:
                candidates.append(finish)
        return min(candidates) if candidates else now + self.RETRY

    def _ship_panel_visible(self):
        """Verify the ship detail scroll before reading any ship state."""
        image = getattr(self.device, 'image', None)
        if image is None or image.shape[0] < 300 or image.shape[1] < 330:
            return False
        # The left 商船 title panel is fixed while sea, foundations and the
        # search selector are not. This paper colour comes from live detail
        # screenshots and stays outside the moving ship illustration.
        panel = image[75:290, 208:320]
        paper = (
            (panel[:, :, 0] >= 135) & (panel[:, :, 0] <= 170)
            & (panel[:, :, 1] >= 119) & (panel[:, :, 1] <= 155)
            & (panel[:, :, 2] >= 126) & (panel[:, :, 2] <= 161)
        )
        return float(paper.mean()) >= 0.40

    def _panel_state(self):
        if not self._ship_panel_visible():
            return None, None
        action_areas = self._find_action_areas()
        action_buttons = [
            Button(area=area, color=(), button=area,
                   name=f'NINGBO_SHIP_ACTION_{index + 1}')
            for index, area in enumerate(action_areas)
        ]
        state_buttons = [
            Button(area=self.WORKING_TIME_AREA, color=(),
                   button=self.WORKING_TIME_AREA,
                   name='NINGBO_SHIP_WORKING_TIME'),
            Button(area=self.IDLE_HEADING_AREA, color=(),
                   button=self.IDLE_HEADING_AREA,
                   name='NINGBO_SHIP_IDLE_HEADING'),
        ]
        # 动作按钮、工作倒计时和空闲标题来自同一帧，一次 OCR 后分流。
        raw = Ocr(
            action_buttons + state_buttons,
            lang='cnocr', name='NINGBO_SHIP_PANEL_BATCH'
        ).ocr(self.device.image)
        if not isinstance(raw, list):
            raw = [raw]
        raw.extend([''] * (len(action_buttons) + 2 - len(raw)))
        actions = self._parse_action_buttons(
            action_areas, raw[:len(action_buttons)]
        )
        if actions:
            return 'completed', None
        state_raw = raw[len(action_buttons):]
        duration_text = self._clean_text(state_raw[0])
        duration = self.parse_duration(duration_text)
        logger.attr(
            'NingboShipWorkingTime', f'{duration_text!r} -> {duration}'
        )
        if duration is not None:
            return 'working', duration
        heading = self._clean_text(state_raw[1])
        if '订单' in heading and ('需求' in heading or '需' in heading):
            return 'idle', None
        return None, None

    def _find_ship_foundation_target(self):
        """用搜索箭头和商船地基图标定位当前船位。"""
        arrow = self.find_building_search_arrow(report_missing=False)
        if arrow is None:
            return None
        # 箭头位于当前船位（及完成/探索旗帜）的上方，图标在同一列。
        # 旧版固定向左移 160px 会选中相邻船；长船位和箭头浮动会让
        # 图标距尖端超过 100px，实机截图为 103~150px。
        axis, tip, direction = arrow
        match = find_foundation_icon(
            self.device.image, arrow, self.BUILDING, threshold=0.60,
        )
        if match is None:
            return None
        x, y, score, family = match
        if abs(x - axis) > 75 or not 12 <= y - tip <= 180:
            return None
        logger.info(
            f'Ningbo ship foundation icon: family={family}, '
            f'center=({x}, {y}), score={score:.3f}'
        )
        return Button(
            area=(), color=(), button=(x - 7, y - 7, x + 7, y + 7),
            name='NINGBO_SHIP_FOUNDATION_ICON_DYNAMIC',
        )

    def _open_current_ship(self, timeout=12):
        self.device.screenshot()
        target = self._find_ship_foundation_target()
        if target is None:
            logger.warning('Ningbo ship position could not be confirmed')
            self.save_abnormal_screenshot('ningbo_ship_position_missing')
            return None, None
        selected_image = self.device.image.copy()
        self._click_searched_building(target)
        clicked_at = time.time()
        retried = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.sleep(0.35)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return None, None
            state, duration = self._panel_state()
            if state is not None:
                return state, duration
            # 首次点击可能只关闭搜索选择框；船位图标仍在时再点一次。
            if not retried and time.time() - clicked_at >= 1.8:
                retry_target = self._find_ship_foundation_target()
                if retry_target is None:
                    retry_target = self.find_still_visible_map_target(
                        selected_image, target,
                    )
                if retry_target is not None:
                    retried = True
                    logger.info('Retry opening Ningbo ship by current ship icon')
                    self.device.click(retry_target)
        logger.warning('Ningbo merchant ship panel did not appear')
        try:
            self.device.image_save('jiangnan_ningbo_ship_open_failed.png')
        except Exception as exc:
            logger.warning(f'Unable to save Ningbo ship debug image: {exc}')
        return None, None

    def _harvest_completed(self, initial_state, timeout=25):
        """Drive 完成 -> 完成 -> 收下 until the order page is idle."""
        if initial_state != 'completed':
            return True, False
        harvested = False
        clicks = 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return False, harvested
            actions = self._find_action_buttons()
            if actions:
                text, button = actions[0]
                clicks += 1
                if clicks > 4:
                    logger.warning('Ningbo ship harvest exceeded click limit')
                    return False, harvested
                logger.info(
                    f'Ningbo ship harvest action {clicks}: {text}'
                )
                self.device.click(button)
                self.device.sleep(0.9)
                harvested = True
                continue
            state, _ = self._panel_state()
            if state == 'idle':
                logger.info(
                    f'Ningbo ship harvest completed: actions={clicks}'
                )
                return True, harvested
            self.device.sleep(0.3)
        logger.warning('Ningbo ship harvest did not reach idle order page')
        return False, harvested

    @staticmethod
    def _title_resource(title):
        compact = re.sub(r'\s+', '', str(title or ''))
        # Longest first prevents a shorter alias from shadowing a full name.
        candidates = sorted(ALL_RESOURCES, key=len, reverse=True)
        return next(
            (resource for resource in candidates if resource in compact),
            None,
        )

    def _read_order(self, side):
        areas = self.ORDER_AREAS[side]
        title = self._ocr_text(
            areas['title'], f'NINGBO_SHIP_{side.upper()}_TITLE'
        )
        resource = self._title_resource(title)
        dried_fish = self._ocr_number(
            areas['dried_fish'], f'NINGBO_SHIP_{side.upper()}_DRIED_FISH'
        )
        cargo = self._ocr_number(
            areas['cargo'], f'NINGBO_SHIP_{side.upper()}_CARGO'
        )
        duration_text = self._ocr_text(
            areas['time'], f'NINGBO_SHIP_{side.upper()}_TIME'
        )
        duration = self.parse_duration(duration_text)
        people = self._ocr_number(
            areas['people'], f'NINGBO_SHIP_{side.upper()}_PEOPLE'
        )
        coin = self._ocr_number(
            areas['coin'], f'NINGBO_SHIP_{side.upper()}_COIN'
        )
        tools = self._ocr_number(
            areas['tool'], f'NINGBO_SHIP_{side.upper()}_TOOLS'
        )
        valid = (
            resource in ALL_RESOURCES
            and all(
                value is not None and value > 0
                for value in (dried_fish, cargo, people, coin)
            )
            and isinstance(duration, timedelta)
            and duration > timedelta(0)
            and tools is not None and tools >= 0
        )
        order = {
            'side': side,
            'title': title,
            'resource': resource,
            'costs': (
                {'鱼干': dried_fish + cargo}
                if resource == '鱼干'
                else {'鱼干': dried_fish, resource: cargo}
            ) if valid else {},
            'duration': duration,
            'people': people,
            'coin': coin,
            'tools': tools,
            'value': (
                float(coin) / (people * duration.total_seconds())
                if valid else -1.0
            ),
            'valid': valid,
        }
        logger.attr('NingboShipOrder', order)
        return order

    @staticmethod
    def _number_from_ocr(value):
        text = str(value or '').translate(str.maketrans({
            'O': '0', 'o': '0', 'I': '1', 'l': '1',
            'S': '5', 'B': '8',
        }))
        matched = re.search(r'\d+', text)
        return int(matched.group()) if matched else None

    def _read_orders_batch(self):
        """从同一张订单截图一次读取左右两张订单的全部字段。"""
        fields = ('title', 'dried_fish', 'cargo', 'time',
                  'people', 'coin', 'tool')
        buttons = []
        locations = []
        for side in ('top', 'bottom'):
            for field in fields:
                area = self.ORDER_AREAS[side][field]
                buttons.append(Button(
                    area=area, color=(), button=area,
                    name=f'NINGBO_SHIP_{side.upper()}_{field.upper()}',
                ))
                locations.append((side, field))
        raw = Ocr(
            buttons, lang='cnocr', name='NINGBO_SHIP_ORDERS_BATCH'
        ).ocr(self.device.image)
        if not isinstance(raw, list):
            raw = [raw]
        values = {'top': {}, 'bottom': {}}
        for (side, field), value in zip(locations, raw):
            values[side][field] = value

        orders = []
        for side in ('top', 'bottom'):
            item = values[side]
            title = self._clean_text(item.get('title'))
            resource = self._title_resource(title)
            duration_text = self._clean_text(item.get('time'))
            duration = self.parse_duration(duration_text)
            dried_fish = self._number_from_ocr(item.get('dried_fish'))
            cargo = self._number_from_ocr(item.get('cargo'))
            people = self._number_from_ocr(item.get('people'))
            coin = self._number_from_ocr(item.get('coin'))
            tools = self._number_from_ocr(item.get('tool'))
            valid = (
                resource in ALL_RESOURCES
                and all(value is not None and value > 0 for value in (
                    dried_fish, cargo, people, coin,
                ))
                and isinstance(duration, timedelta)
                and duration > timedelta(0)
                and tools is not None and tools >= 0
            )
            order = {
                'side': side,
                'title': title,
                'resource': resource,
                'costs': (
                    {'鱼干': dried_fish + cargo}
                    if resource == '鱼干'
                    else {'鱼干': dried_fish, resource: cargo}
                ) if valid else {},
                'duration': duration,
                'people': people,
                'coin': coin,
                'tools': tools,
                'value': (
                    float(coin) / (people * duration.total_seconds())
                    if valid else -1.0
                ),
                'valid': valid,
            }
            logger.attr('NingboShipOrder', order)
            orders.append(order)
        return orders

    def _choose_order(self, state):
        inventory = state.load_inventories((self.CITY,)).get(self.CITY, {})
        orders = self._read_orders_batch()
        candidates = []
        self._last_order_failure = '订单识别失败'
        self._last_order_shortages = []
        for order in orders:
            if not order['valid']:
                continue
            enough = all(
                int(inventory.get(resource, -1)) >= amount
                for resource, amount in order['costs'].items()
            )
            logger.info(
                f'Ningbo ship order candidate: side={order["side"]}, '
                f'costs={order["costs"]}, inventory='
                f'{dict((r, inventory.get(r, -1)) for r in order["costs"])}, '
                f'coin={order["coin"]}, tools={order["tools"]}, '
                f'people={order["people"]}, duration={order["duration"]}, '
                f'value={order["value"]:.9f}, enough={enough}'
            )
            if enough:
                candidates.append(order)
            else:
                missing = [f'{resource}{inventory.get(resource, -1)}/{amount}'
                           for resource, amount in order['costs'].items()
                           if int(inventory.get(resource, -1)) < amount]
                self._last_order_shortages.append(f'{order["side"]}:' + ','.join(missing))
                unknown = any(int(inventory.get(resource, -1)) < 0 for resource in order['costs'])
                if unknown:
                    self._last_order_failure = '订单库存未确认'
                    state.request_scan('Ningbo ship order inventory unknown', cities={self.CITY})
                elif self._last_order_failure != '订单库存未确认':
                    self._last_order_failure = '订单库存不足'
        if not candidates:
            return None
        # 用户定义的性价比是通宝/(人物数*时间)。完全相同时用更多
        # 营造工具、再用更短时间做稳定的次级排序。
        return max(
            candidates,
            key=lambda order: (
                order['value'], order['tools'],
                -order['duration'].total_seconds(),
            ),
        )

    def _depart(self, order, state):
        self.last_no_assignee = False
        areas = self.ORDER_AREAS[order['side']]
        self.device.click(Button(
            area=(), color=(), button=areas['card'],
            name=f'NINGBO_SHIP_ORDER_{order["side"].upper()}',
        ))
        deadline = time.time() + 12
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return None
            if self._no_assignee_toast():
                self.last_no_assignee = True
                logger.warning('Ningbo ship has no idle special resident')
                return None
            if self.appear(
                MERCHANT_ASSIGNEE_CHECK, offset=(5, 5), similarity=0.76
            ):
                break
        else:
            logger.warning('Ningbo ship assignee page did not appear')
            return None

        self.device.click(Button(
            area=(), color=(), button=(378, 128, 405, 157),
            name='NINGBO_SHIP_FIRST_ASSIGNEE',
        ))
        # The portrait itself can start the voyage immediately; tapping the
        # 详情 strip below it instead opens a resident card with a 分配 button.
        confirm_deadline = time.time() + 6
        working_duration = None
        while time.time() < confirm_deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.handle_network_timeout_popup():
                return None
            if self._ship_panel_visible():
                working_duration = self._read_working_duration()
                if working_duration is not None:
                    logger.info('Ningbo ship departed directly from portrait')
                    break
            if self.appear(
                NINGBO_SHIP_ASSIGNEE_CONFIRM,
                offset=(55, 12), similarity=0.76,
            ):
                self.device.click(NINGBO_SHIP_ASSIGNEE_CONFIRM)
                logger.info('Confirm Ningbo ship assignee')
                break
        else:
            logger.warning('Ningbo ship assignee confirm button did not appear')
            return None
        if working_duration is None:
            deadline = time.time() + 15
            while time.time() < deadline:
                self.device.sleep(0.4)
                self.device.screenshot()
                if self.handle_network_timeout_popup():
                    return None
                if self._ship_panel_visible():
                    working_duration = self._read_working_duration()
                    if working_duration is not None:
                        break
        if working_duration is None:
            logger.warning('Ningbo ship actual duration was not recognized')
            return None

        if not state.consume_inventory(
            self.CITY, order['costs'], event_type='merchant_ship_consumption',
            reason='宁波商船出航',
        ):
            # The order is already committed.  Do not request an early scan;
            # retain the departure and let the normal warehouse schedule fix
            # any external inventory change later.
            logger.warning(
                f'Ningbo ship departed but inventory deduction failed: '
                f'{order["costs"]}'
            )
        return working_duration

    def _close_panel(self):
        self.device.press_back()
        deadline = time.time() + 10
        while time.time() < deadline:
            self.device.sleep(0.3)
            self.device.screenshot()
            if self.is_in_main():
                return True
            if self.handle_network_timeout_popup():
                return False
        return False

    def _inspect_ship(self, index, state):
        panel_state, duration = self._open_current_ship()
        if panel_state is None:
            return datetime.now().replace(microsecond=0) + self.RETRY, '[打开商船失败][等待复查]'

        recorded_status = str(self._get(index, 'Status', '') or '')
        recorded_state = (
            'working' if '探索中' in recorded_status else
            'idle' if '订单库存不足' in recorded_status else None
        )
        recorded_next = self._parse_time(self._get(index, 'NextRun'))
        observed_at = datetime.now().replace(microsecond=0)
        observed_finish = (
            observed_at + duration if panel_state == 'working'
            and duration is not None else None
        )
        changed = recorded_state is not None and recorded_state != panel_state
        drifted = (
            recorded_state == panel_state == 'working'
            and recorded_next is not None and observed_finish is not None
            and abs((observed_finish - recorded_next).total_seconds()) > 120
        )
        if changed or drifted:
            logger.warning(
                f'Ningbo ship {index}/4 recorded state differs from live panel: '
                f'recorded={recorded_status!r}, next={recorded_next}, '
                f'live={panel_state}, remaining={duration}; verify again'
            )
            self.save_abnormal_screenshot('ningbo_ship_state_mismatch')
            self.device.sleep(0.3)
            self.device.screenshot()
            checked_state, checked_duration = self._panel_state()
            durations_agree = (
                panel_state != 'working' or (
                    duration is not None and checked_duration is not None
                    and abs((duration - checked_duration).total_seconds()) <= 90
                )
            )
            if checked_state != panel_state or not durations_agree:
                self._close_panel()
                logger.warning(
                    f'Ningbo ship {index}/4 panel reads disagree; '
                    'keep record and retry soon'
                )
                return (
                    datetime.now().replace(microsecond=0) + self.RETRY,
                    f'{recorded_status}[面板状态冲突][等待复查]',
                )
            panel_state, duration = checked_state, checked_duration
            logger.info(
                f'Ningbo ship {index}/4 state correction confirmed: '
                f'{recorded_state} -> {panel_state}, remaining={duration}'
            )

        if panel_state == 'working':
            self._close_panel()
            finish = datetime.now().replace(microsecond=0) + duration + timedelta(seconds=3)
            return finish, f'[探索中][{finish}]'

        ok, harvested = self._harvest_completed(panel_state)
        if not ok:
            self._close_panel()
            return datetime.now().replace(microsecond=0) + self.RETRY, '[商船收获失败][等待复查]'

        order = self._choose_order(state)
        if order is None:
            self._close_panel()
            # Missing stock can arrive today; an unreadable order can recover
            # on a new frame. Neither is a reason to postpone an idle ship 24h.
            reason = getattr(self, '_last_order_failure', '订单识别失败')
            retry = timedelta(hours=1) if reason == '订单库存不足' else self.RETRY
            next_run = datetime.now().replace(microsecond=0) + retry
            details = ';'.join(getattr(self, '_last_order_shortages', []))
            return next_run, f'[{reason}][{details or "等待重新识别"}][等待复查]'

        duration = self._depart(order, state)
        if duration is None:
            self._close_panel()
            if getattr(self, 'last_no_assignee', False):
                release = self._next_assignee_release()
                return (
                    release,
                    f'[无空闲特殊居民][等待至{release}]',
                )
            return datetime.now().replace(microsecond=0) + self.RETRY, '[商船出发失败][等待复查]'
        self._close_panel()
        finish = datetime.now().replace(microsecond=0) + duration + timedelta(seconds=3)
        prefix = '[已收获]' if harvested else ''
        return finish, (
            f'{prefix}[{order["resource"]}]'
            f'[通宝{order["coin"]}][营造工具{order["tools"]}]'
            f'[{finish}][探索中]'
        )

    def run(self):
        logger.hr('JIANGNAN NINGBO MERCHANT SHIP', level=0)
        now = datetime.now().replace(microsecond=0)
        state = JiangnanInventoryState(self.config)
        schedules = []
        for index in range(1, 5):
            if not bool(self._get(index, 'Enable', True)):
                continue
            next_run = self._parse_time(
                self._get(index, 'NextRun')
            ) or self.DEFAULT_TIME
            if '明日复查' in str(self._get(index, 'Status', '')) and next_run > now:
                next_run = now
                self._set(index, 'NextRun', next_run)
            schedules.append((next_run, index))

        due = sorted(
            (
                (next_run, index)
                for next_run, index in schedules
                if next_run <= now
            ),
            # 旧 NextRun 只决定哪些船到期；按船号处理可避免在 3→1→4→2
            # 之间来回切换，并减少同屏选择器操作。
            key=lambda item: item[1],
        )
        next_runs = {index: next_run for next_run, index in schedules}
        if due:
            logger.info(
                f'Run due Ningbo merchant ships together: '
                f'{[index for _, index in due]}'
            )
            entered_city = self.switch_city(self.CITY)
            for _, index in due:
                retry = datetime.now().replace(microsecond=0) + self.RETRY
                if not entered_city:
                    ship_next_run = retry
                    status = '[进入宁波府失败][等待复查]'
                else:
                    if not self._search_ship_selector(index):
                        ship_next_run = retry
                        status = '[未找到商船][等待复查]'
                    else:
                        ship_next_run, status = self._inspect_ship(index, state)
                    # 面板或搜索失败都可能留下非主界面，下一艘船前先恢复。
                    if not self.ui_goto_main():
                        entered_city = False
                self._set(index, 'NextRun', ship_next_run)
                self._set(index, 'Status', status)
                next_runs[index] = ship_next_run

        self.config.update()
        if next_runs:
            self.config.task_delay(target=min(next_runs.values()))
        else:
            self.config.task_delay(success=True)
        return True
