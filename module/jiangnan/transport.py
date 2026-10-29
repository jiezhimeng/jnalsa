import re
import time
from datetime import datetime, timedelta

import cv2

from module.base.button import Button
from module.base.utils import crop, load_image
from module.jiangnan.allocation import JiangnanAllocation
from module.jiangnan.assets import (
    TRANSPORT_ARRIVAL_PROMPT_CHECK,
    TRANSPORT_GO_BUTTON,
)
from module.jiangnan.harvest import JiangnanHarvest
from module.jiangnan.resource_layout import PAGE_CITY
from module.logger import logger
from module.jiangnan.ocr import Digit, Ocr


class TransportTimeOcr(Ocr):
    """识别运输位顶部的中文剩余时间，例如“46分钟”。"""

    def __init__(self, area, threshold=90):
        areas = area if isinstance(area, list) else [area]
        buttons = [
            Button(
                area=item, color=(), button=item,
                name=f'TRANSPORT_TIME_OCR_{index + 1}',
            )
            for index, item in enumerate(areas)
        ]
        super().__init__(
            buttons=buttons if len(buttons) > 1 else buttons[0],
            lang='cnocr',
            letter=(210, 200, 165),
            threshold=threshold,
            name='TRANSPORT_TIME_OCR',
        )


class TransportTimeDigitOcr(Digit):
    """只读取倒计时开头的数字，作为中文单位 OCR 的兜底。"""

    def __init__(self, area, threshold=90):
        super().__init__(
            buttons=Button(
                area=area, color=(), button=area,
                name='TRANSPORT_TIME_DIGIT_OCR',
            ),
            lang='azur_lane',
            letter=(210, 200, 165),
            threshold=threshold,
            name='TRANSPORT_TIME_DIGIT_OCR',
        )


class JiangnanTransportStatus(JiangnanAllocation):
    """读取运输位状态，收取已抵达物资，并给出最近完成时间。"""

    TIME_REFERENCE_AREA = (
        (302, 176, 382, 212),
        (387, 176, 467, 212),
    )
    # 真正不存在的灰纸船位在现有城市实测为 25～35；鸡鸣山的有效
    # 船位被“已抵达”货物图标或倒计时覆盖后会降到 44～46。旧值 52
    # 会在模板/OCR 尚未运行前把两种有效状态一起短路成 unavailable。
    UNAVAILABLE_SATURATION_MAX = 38
    STATUS_TEMPLATE = './assets/jiangnan/allocation/status.png'
    # 固定状态只裁文字，避开会随货物与城市变化的卡片下半部。
    STATIC_STATUS_AREAS = {
        'idle': (402, 182, 458, 206),
        'arrived': (317, 312, 373, 336),
    }
    STATIC_STATUS_THRESHOLD = 0.88

    def __init__(self, config, device=None, task=None):
        super().__init__(config=config, device=device, task=task)
        self.last_snapshot = {}
        self.last_collected_slots = []
        self.last_collection_discarded = False
        self.last_collection_destination = None
        self._static_status_templates = {
            state: cv2.cvtColor(
                load_image(self.STATUS_TEMPLATE, area), cv2.COLOR_RGB2GRAY,
            )
            for state, area in self.STATIC_STATUS_AREAS.items()
        }

    def _static_slot_status(self, areas):
        """从当前截图匹配固定状态；不确定时交给文字识别。"""
        x1, y1, x2, y2 = areas['time']
        image = self.device.image
        height, width = image.shape[:2]
        search = crop(image, (
            max(0, x1 - 25), max(0, y1 - 25),
            min(width, x2 + 25), min(height, y2 + 25),
        ), copy=False)
        if search.size == 0:
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)
        scores = {}
        for state, template in self._static_status_templates.items():
            if (gray.shape[0] < template.shape[0]
                    or gray.shape[1] < template.shape[1]):
                continue
            scores[state] = float(cv2.minMaxLoc(cv2.matchTemplate(
                gray, template, cv2.TM_CCOEFF_NORMED,
            ))[1])
        if not scores:
            return None
        best = max(scores, key=scores.get)
        second = max((score for state, score in scores.items()
                      if state != best), default=0.0)
        logger.attr('TransportStaticStatus',
                    f'{best}:{scores[best]:.3f}, second={second:.3f}')
        if (scores[best] >= self.STATIC_STATUS_THRESHOLD
                and scores[best] - second >= 0.20):
            return best
        return None

    def _read_slot_states(self, areas_list):
        """先处理固定状态，仅将剩余船位批量送入 OCR。"""
        states = [self._static_slot_status(areas) for areas in areas_list]
        pending = [index for index, state in enumerate(states) if state is None]
        raw_values = [''] * len(areas_list)
        if pending:
            areas = [self._expanded_time_area(areas_list[index]['time'])
                     for index in pending]
            values = TransportTimeOcr(areas).ocr(self.device.image)
            if not isinstance(values, list):
                values = [values]
            for index, value in zip(pending, values):
                raw_values[index] = value
            # 列表刚滚到边界时，城市卡片可能只露出倒计时文字的
            # 上半截。窄裁图偶尔只得到“一”，把已抵达误判为
            # occupied_unknown。仅对无法解释的结果扩大纵向裁图复读。
            unclear = [
                index for index in pending
                if not any(token in str(raw_values[index]) for token in (
                    '空闲', '空閒', '抵', '达', '達',
                )) and self.parse_remaining_time(raw_values[index]) is None
            ]
            if unclear:
                wide_areas = [
                    self._wide_time_area(areas_list[index]['time'])
                    for index in unclear
                ]
                retry_values = TransportTimeOcr(wide_areas).ocr(
                    self.device.image
                )
                if not isinstance(retry_values, list):
                    retry_values = [retry_values]
                for index, value in zip(unclear, retry_values):
                    if (any(token in str(value) for token in (
                        '空闲', '空閒', '抵', '达', '達',
                    )) or self.parse_remaining_time(value) is not None):
                        logger.attr('TransportTimeWideRetry',
                                    f'{raw_values[index]} -> {value}')
                        raw_values[index] = value
        digit_values = self._time_digit_fallbacks(areas_list, raw_values)
        return states, raw_values, digit_values

    @staticmethod
    def _wide_time_area(area):
        return (
            max(0, area[0] - 8), max(0, area[1] - 12),
            min(1280, area[2] + 8), min(720, area[3] + 24),
        )

    @staticmethod
    def parse_remaining_time(text):
        """将“1天2小时30分钟”之类文本转成 timedelta。"""
        text = re.sub(r'\s+', '', str(text))
        # 浅色窄体数字 1 经常被中文 OCR 读成 i/l/I/竖线，例如
        # “51分钟”得到“5i分钟”。这里只在时间文本中做数字归一化。
        text = text.translate(str.maketrans({
            'i': '1', 'I': '1', 'l': '1', '|': '1', '!': '1',
            'O': '0', 'o': '0',
        }))
        day = re.search(r'(\d+)天', text)
        hour = re.search(r'(\d+)小时', text)
        minute = re.search(r'(\d+)分钟?', text)
        second = re.search(r'(\d+)秒', text)
        if not any((day, hour, minute, second)):
            return None
        return timedelta(
            days=int(day.group(1)) if day else 0,
            hours=int(hour.group(1)) if hour else 0,
            minutes=int(minute.group(1)) if minute else 0,
            seconds=int(second.group(1)) if second else 0,
        )

    def _slot_areas(self, city_match, slot_index):
        city_x, city_y, _ = city_match
        ref_x, ref_y = self.OVERVIEW_REFERENCE_CITY_CENTER
        dx = city_x - ref_x
        dy = city_y - ref_y
        idle_check, click_area = self.IDLE_SLOT_REFERENCE[slot_index]
        return {
            'idle_check': self._moved_area(idle_check, dx, dy),
            'click': self._moved_area(click_area, dx, dy),
            'time': self._moved_area(self.TIME_REFERENCE_AREA[slot_index], dx, dy),
        }

    def _interpret_slot(
        self, areas, raw, slot_index, fallback_digit=None
    ):
        idle_image = crop(self.device.image, areas['idle_check'], copy=False)
        idle_hsv = cv2.cvtColor(idle_image, cv2.COLOR_RGB2HSV)
        mean_saturation = float(idle_hsv[:, :, 1].mean())
        logger.attr(
            f'TransportSlot{slot_index + 1}Saturation',
            f'{mean_saturation:.1f}',
        )
        texts = [str(raw)]
        time_area = areas['time']
        normalized_raw = re.sub(r'\s+', '', str(raw))
        duration = self.parse_remaining_time(raw)
        logger.attr('TransportRemainingText', raw)
        # 状态文字是运输位的唯一主判据。模板和卡片颜色会随城市背景、
        # 天气及货物图标明显变化，鸡鸣山曾因此把“已抵达”和倒计时都
        # 判成不可用。完整文字只可能是空闲、已抵达或一个剩余时间。
        if '空闲' in normalized_raw or '空閒' in normalized_raw:
            return 'idle', None, areas['click']
        normalized_arrived = normalized_raw.lower()
        arrived_tokens = (
            '抵达', '抵達', '已达', '已達', '到达', '到達',
            'arrived',
        )
        if any(token in normalized_arrived for token in arrived_tokens):
            return 'arrived', None, areas['click']
        if any(ch in normalized_arrived for ch in ('达', '達', '抵')) and duration is None:
            return 'arrived', None, areas['click']
        if duration is not None and duration > timedelta(0):
            logger.attr('TransportRemainingDuration', duration)
            return 'transporting', duration, areas['click']

        # 只有文字识别失败后才判断该格是否根本不存在。真正缺失的船位
        # 是低饱和度灰纸背景；这项检查不再有机会盖过有效状态文字。
        if mean_saturation < self.UNAVAILABLE_SATURATION_MAX:
            return 'unavailable', None, areas['click']

        if duration is None or duration <= timedelta(0):
            # “5小时”中的数字与汉字粘连时，整句 OCR 偶尔只剩“小时”。
            # 单独裁取卡片左侧数字，再与已经识别出的单位组合。复杂时间
            # 只采用第一个单位，最多导致提前复查，不会错过抵达。
            normalized_texts = [
                text.translate(str.maketrans({
                    'i': '1', 'I': '1', 'l': '1', '|': '1', '!': '1',
                    'O': '0', 'o': '0',
                }))
                for text in texts
            ]
            unit = None
            if any('天' in text for text in normalized_texts):
                unit = 'day'
            elif any('小时' in text for text in normalized_texts):
                unit = 'hour'
            elif any('分' in text for text in normalized_texts):
                unit = 'minute'
            if unit is not None:
                digit_area = (
                    time_area[0], time_area[1],
                    min(time_area[0] + 36, time_area[2]), time_area[3],
                )
                value = fallback_digit
                try:
                    value = int(''.join(re.findall(r'\d+', str(value))))
                except (TypeError, ValueError):
                    value = -1
                if value > 0:
                    duration = {
                        'day': timedelta(days=value),
                        'hour': timedelta(hours=value),
                        'minute': timedelta(minutes=value),
                    }[unit]
                    logger.attr(
                        'TransportTimeDigitFallback',
                        f'unit={unit}, value={value}, result={duration}',
                    )
                    return 'transporting', duration, areas['click']
            logger.warning(
                f'Unable to read occupied transport time: {texts}'
            )
            return 'occupied_unknown', None, areas['click']
        logger.attr('TransportRemainingDuration', duration)
        return 'transporting', duration, areas['click']

    @staticmethod
    def _expanded_time_area(area):
        return (
            max(0, area[0] - 8), max(0, area[1] - 4),
            min(1280, area[2] + 8), min(720, area[3] + 4),
        )

    def _time_digit_fallbacks(self, areas_list, raw_values):
        """只为整句 OCR 漏掉数字的时间文本批量读取数字。"""
        fallback_values = [None] * len(raw_values)
        positions = []
        buttons = []
        for position, (areas, raw) in enumerate(zip(areas_list, raw_values)):
            text = str(raw).replace(' ', '')
            if self.parse_remaining_time(text) is not None:
                continue
            if not any(unit in text for unit in ('天', '小时', '分')):
                continue
            time_area = areas['time']
            digit_area = (
                time_area[0], time_area[1],
                min(time_area[0] + 36, time_area[2]), time_area[3],
            )
            positions.append(position)
            buttons.append(Button(
                area=digit_area, color=(), button=digit_area,
                name=f'TRANSPORT_TIME_DIGIT_{position + 1}',
            ))
        if not buttons:
            return fallback_values
        values = Digit(
            buttons,
            lang='azur_lane',
            name='TRANSPORT_TIME_DIGIT_FALLBACK_BATCH',
        ).ocr(self.device.image)
        if not isinstance(values, list):
            values = [values]
        for position, value in zip(positions, values):
            fallback_values[position] = value
        return fallback_values

    def inspect_slots(self, city_match, slot_indices=(0, 1)):
        """用同一张截图一次识别一座城市的全部运输船位。"""
        slot_indices = tuple(slot_indices)
        areas_list = [
            self._slot_areas(city_match, index) for index in slot_indices
        ]
        states, raw_values, digit_values = self._read_slot_states(areas_list)
        return [
            ((states[position], None, areas['click'])
             if states[position] is not None else self._interpret_slot(
                 areas, raw_values[position], slot_index,
                 fallback_digit=digit_values[position],
             ))
            for position, (areas, slot_index) in enumerate(
                zip(areas_list, slot_indices)
            )
        ]

    def inspect_slot(self, city_match, slot_index):
        return self.inspect_slots(city_match, (slot_index,))[0]

    def inspect_city_page(self, city_matches):
        """一次 OCR 读取当前页面多座城市的全部船位。"""
        entries = []
        for city, match in city_matches:
            for slot_index in (0, 1):
                entries.append((
                    city,
                    slot_index,
                    self._slot_areas(match, slot_index),
                ))
        if not entries:
            return {}

        areas_list = [areas for _, _, areas in entries]
        states, raw_values, digit_values = self._read_slot_states(areas_list)

        result = {}
        for position, (city, slot_index, areas) in enumerate(entries):
            status = (
                (states[position], None, areas['click'])
                if states[position] is not None else self._interpret_slot(
                    areas, raw_values[position], slot_index,
                    fallback_digit=digit_values[position],
                )
            )
            result.setdefault(city, []).append(status)
        return result

    def _visible_status_cities(self, cities, similarity=0.82):
        """从当前一帧同时定位所有目标城市，不读取固定格位文字。"""
        templates = self._load_city_templates()
        x1, y1, x2, y2 = (190, 155, 1080, 680)
        search = self.device.image[y1:y2, x1:x2]
        found = []
        scores = []
        for city in cities:
            template = templates.get(city)
            if template is None or (
                template.shape[0] > search.shape[0]
                or template.shape[1] > search.shape[1]
            ):
                continue
            score_map = cv2.matchTemplate(
                search, template, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, location = cv2.minMaxLoc(score_map)
            height, width = template.shape[:2]
            center_x = x1 + location[0] + width // 2
            center_y = y1 + location[1] + height // 2
            scores.append(f'{city}:{score:.3f}@({center_x},{center_y})')
            # 用户实机的顶部、底部两个视图都完整显示四行船位。底行
            # 名称中心约为 635，状态文字仍位于窗口边框之上。
            if score >= similarity and 225 <= center_y <= 665:
                found.append((city, (center_x, center_y, float(score))))
        found.sort(key=lambda item: (item[1][1], item[1][0]))
        logger.attr('TransportPageCityMatches', scores)
        logger.attr('TransportPageCities', [city for city, _ in found])
        return found

    def _wait_arrival_prompt(self, timeout=8):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            if self.appear(
                TRANSPORT_ARRIVAL_PROMPT_CHECK,
                offset=(3, 3),
                similarity=0.88,
            ):
                return True
            self.device.sleep(0.4)
        logger.warning('Transport arrival prompt timeout')
        return False

    def collect_arrived_slot(self, click_area):
        self.last_collection_destination = None
        logger.info('Open arrived transport slot')
        self.device.click(Button(
            area=(), color=(), button=click_area, name='TRANSPORT_ARRIVED_SLOT'
        ))
        if not self._wait_arrival_prompt():
            return False

        logger.info('Go to transport destination city')
        self.device.click(self.anchored_button(
            TRANSPORT_ARRIVAL_PROMPT_CHECK,
            TRANSPORT_GO_BUTTON,
            name='TRANSPORT_GO_BUTTON_DYNAMIC',
            inset=8,
        ))
        go_clicked_at = time.time()
        if not self.wait_city_arrival(timeout=40, go_clicked_at=go_clicked_at):
            return False

        self.device.screenshot()
        self.last_collection_destination = self.read_current_city_from_seal()
        logger.attr(
            'TransportArrivalDestination',
            self.last_collection_destination or 'unknown',
        )

        harvest = JiangnanHarvest(
            config=self.config,
            device=self.device,
            task=getattr(self, 'task', None),
        )
        # 搜索阶段只负责定位码头；是否还有待收货，最终以运输城市列表
        # 中的“已抵达”状态为准，不能用当前地图画面的“完成”牌代替。
        if not harvest.search_building(
            '码头', handle_interference=False
        ):
            logger.warning('Unable to locate dock before collecting arrivals')
            return False
        result = harvest.harvest_all_visible(
            force=True,
            require_one=True,
        )
        self.last_collection_discarded = harvest.last_discarded
        return result

    def collect_shipment(self, shipment):
        """按运输台账中的来源城和船位收取一个到期节点。"""
        city = shipment['source']
        slot_index = int(shipment['slot'])
        match = self._locate_status_city(city)
        if match is None:
            return 'failed', None
        state, duration, click_area = self.inspect_slot(match, slot_index)
        logger.info(
            f'Dock harvest status: {city} slot {slot_index + 1}: {state}'
        )
        if state == 'arrived':
            if self.collect_arrived_slot(click_area):
                return 'collected', None
            return 'failed', None
        if state == 'transporting' and duration is not None:
            return 'transporting', datetime.now() + duration
        return state, None

    def _locate_status_city(self, city):
        """在运输总览中定位城市；同一城市的两个船位共用一次定位。"""
        if not self.goto_city_list():
            return None
        configured = self.city_layout.load(PAGE_CITY)
        expected = configured.get(city)
        match = self.locate_city(city)
        if match is None:
            logger.warning(f'Transport city not found: {city}')
            return None

        # 顶部固定页头会遮住第一行的船位状态区；底部半行也可能没有
        # 完整露出。当前处于顶部还是底部、目标处于屏幕第几行，都能
        # 直接由城市辅助信息算出；不应在滑动后再用八个固定 OCR 格猜
        # 位置。只有辅助位置的第四行需要向上移动一行。
        screen_row = None
        if expected is not None:
            expected_row, expected_column = expected
            max_row = max(row for row, _ in configured.values())
            first_row = 0 if expected_row <= 3 else max(0, max_row - 3)
            screen_row = expected_row - first_row
            logger.attr(
                'TransportCityAuxPosition',
                f'city={city}, global={expected}, first_row={first_row}, '
                f'screen_row={screen_row}',
            )
        city_y = match[1]
        if screen_row == 3:
            delta = -130
            self.jiangnan_drag(
                (570, 365),
                (570, 365 + delta),
                name='TRANSPORT_CITY_ROW_TO_CENTER',
            )
            self.device.sleep(0.8)
            self.device.screenshot()
            # 辅助信息给出新位置；城市名模板只负责确认实际落点，不再
            # 决定当前是哪一页或哪一行。
            match = self.match_city_template(city)
            if match is None:
                logger.warning(
                    f'Transport city auxiliary alignment verification failed: '
                    f'{city}, expected_y={city_y + delta}'
                )
                return None
        return match

    def scan(
        self,
        cities,
        collect_arrived=True,
        collect_slots=None,
        skip_missing=False,
        _final_city_list_pass=False,
    ):
        """
        Args:
            collect_slots: 可选的 ``(来源城市, 船位序号)`` 集合。传入后只
                收取台账中允许自动收获的船，避免绕过目的城市的收获开关。
            skip_missing: 城市尚未开放或总览中找不到时继续扫描其余城市。

        Returns:
            (success, finish_times, has_unknown_occupied, collected_count)
        """
        finish_times = []
        has_unknown = False
        collected = 0
        self.last_snapshot = {}
        self.last_collected_slots = []
        self.last_collection_discarded = False

        cities = list(dict.fromkeys(cities))
        if not cities:
            return True, finish_times, has_unknown, collected
        pending = set(cities)
        arrivals = []

        # 城镇列表只有顶部、底部两个有效视图。每个视图一次定位所有
        # 城市、一次读取全部船位；从顶部直接拉到底，中途不做识别。
        if not self.goto_city_list() or not self.city_list_reset_top():
            return False, finish_times, has_unknown, collected

        for position in ('top', 'bottom'):
            self.device.screenshot()
            matches = self._visible_status_cities(
                [city for city in cities if city in pending]
            )
            page_results = self.inspect_city_page(matches)
            for city, _ in matches:
                slot_results = page_results.get(city, [])
                if len(slot_results) < 2:
                    continue
                pending.discard(city)
                self.last_snapshot[city] = []
                for slot_index, result in enumerate(slot_results):
                    state, duration, click_area = result
                    logger.info(
                        f'Transport status: {city} '
                        f'slot {slot_index + 1}: {state}'
                    )
                    finish_time = (
                        datetime.now() + duration
                        if state == 'transporting' and duration is not None
                        else None
                    )
                    slot_status = {
                        'slot': slot_index,
                        'state': state,
                        'finish_time': finish_time,
                    }
                    self.last_snapshot[city].append(slot_status)
                    if state == 'transporting':
                        finish_times.append(finish_time)
                    elif state == 'occupied_unknown':
                        has_unknown = True
                    elif state == 'arrived':
                        should_collect = (
                            collect_arrived
                            and (
                                collect_slots is None
                                or (city, slot_index) in collect_slots
                            )
                        )
                        if should_collect:
                            arrivals.append((city, slot_index, slot_status))

            if not pending:
                break
            if position == 'top' and not self.city_list_reset_bottom():
                break

        if pending:
            for city in cities:
                if city not in pending:
                    continue
                logger.warning(
                    f'Skip unavailable transport city during audit: {city}'
                )
                self.last_snapshot[city] = []
            if not skip_missing:
                return False, finish_times, has_unknown, collected

        # 收货会离开运输总览，因此放在整页审计完成后逐个复核。一般
        # 没有已抵达货物时，整个扫描过程不会再做任何单城市定位。
        for city, slot_index, slot_status in arrivals:
            match = self._locate_status_city(city)
            if match is None:
                return False, finish_times, has_unknown, collected
            state, duration, click_area = self.inspect_slot(match, slot_index)
            logger.info(
                f'Dock harvest status: {city} '
                f'slot {slot_index + 1}: {state}'
            )
            if state == 'arrived':
                if not self.collect_arrived_slot(click_area):
                    return False, finish_times, has_unknown, collected
                collected += 1
                self.last_collected_slots.append(
                    (city, slot_index, self.last_collection_discarded,
                     False, self.last_collection_destination)
                )
                slot_status['state'] = 'collected'
            elif state == 'transporting' and duration is not None:
                finish_time = datetime.now() + duration
                finish_times.append(finish_time)
                slot_status['state'] = state
                slot_status['finish_time'] = finish_time
            elif state == 'occupied_unknown':
                has_unknown = True
                slot_status['state'] = state
            elif state == 'idle':
                # 审计时已明确看到“已抵达”，但前一个码头的强制批量
                # 收获可能顺带清掉了这个船位。复核变为空闲等价于已经
                # 收取，必须同步消费对应运输台账。
                collected += 1
                self.last_collected_slots.append((city, slot_index, False))
                slot_status['state'] = 'collected'
            else:
                slot_status['state'] = state

        # 收货会离开运输城市列表，因此只要本轮实际收过货，结束前必须
        # 重新回到同一个城市列表完整扫描。若复查又发现“已抵达”，
        # 继续收取并再次复查；只有整轮列表 collected=0 才允许结束。
        if collect_arrived and collected > 0 and not _final_city_list_pass:
            aggregate_slots = list(self.last_collected_slots)
            total_collected = collected
            for verify_round in range(1, 5):
                logger.info(
                    f'Final transport city-list verification: '
                    f'{verify_round}/4'
                )
                ok, verify_finish, verify_unknown, verify_collected = self.scan(
                    cities=cities,
                    collect_arrived=collect_arrived,
                    collect_slots=collect_slots,
                    skip_missing=skip_missing,
                    _final_city_list_pass=True,
                )
                verify_slots = list(self.last_collected_slots)
                aggregate_slots.extend(verify_slots)
                self.last_collected_slots = list(aggregate_slots)
                self.last_collection_discarded = any(
                    bool(item[2]) for item in aggregate_slots
                )
                if not ok:
                    logger.warning(
                        'Final transport city-list verification failed'
                    )
                    return False, verify_finish, verify_unknown, total_collected
                total_collected += verify_collected
                finish_times = verify_finish
                has_unknown = verify_unknown
                if verify_collected == 0:
                    logger.info(
                        'Final transport city-list verification passed: '
                        'no arrived cargo remains'
                    )
                    collected = total_collected
                    break
                logger.warning(
                    f'Final city-list verification found more arrived cargo: '
                    f'collected={verify_collected}; verify again'
                )
            else:
                logger.warning(
                    'Transport city list still has newly collected arrivals '
                    'after 4 verification rounds'
                )
                return False, finish_times, has_unknown, total_collected

        return True, finish_times, has_unknown, collected
