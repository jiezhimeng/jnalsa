"""Shared city-station workflow: rations, full-stamina captains and pinned items."""
from datetime import datetime
from contextlib import contextmanager
from decimal import Decimal
import time

import cv2

from module.base.button import Button
from module.base.utils import crop
from module.jiangnan.building import JiangnanBuilding
from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
from module.jiangnan.post_station_cities import STATIONS, MAP_REGION, map_fields
from module.jiangnan.post_station_rules import (
    Captain, MAP_NAMES, MAP_FIELDS, can_depart, choose_captain, enabled_maps,
    interval_label, interval_minutes, names, next_map, normalize, parse_stock, parse_stamina,
)
from module.jiangnan.post_station_vision import (
    bubble, bubble_click_point, match, page, red_arrow, pinned_labels, resource_foreground, resource_fallback,
    resource_target_obstructed, template, unreachable,
)
from module.logger import logger


class ResourceUnreachable(Exception):
    """The game explicitly rejected the current gathering destination."""


class ResourceNotFound(Exception):
    """A completed resource-list search did not contain the requested item."""


class JiangnanYingtianPostStation(JiangnanBuilding):
    TASK = 'JiangnanYingtianPostStation'
    GROUP = 'post_station'
    CITY = '应天府'
    MAP_NAMES = MAP_NAMES
    MAP_FIELDS = MAP_FIELDS

    def setting(self, field, default=None):
        return self.config.cross_get(f'{self.TASK}.{self.GROUP}.{field}', default=default)

    def save(self, **values):
        for key, value in values.items():
            self.config.modified[f'{self.TASK}.{self.GROUP}.{key}'] = value
        self.config.update()

    def click_at(self, x, y, name):
        self.device.click(Button(area=(), color=(), button=(x-3, y-3, x+3, y+3), name=name))

    def wait_page(self, expected, timeout=20, stop_on_city=False):
        # Device screenshots trigger the idle watchdog after 60 seconds.
        # A failed transition must return to task cleanup before that limit.
        deadline = time.monotonic() + min(timeout, 45)
        stable = 0
        city_frames = 0
        while time.monotonic() < deadline:
            self.device.sleep(.35)
            self.device.screenshot()
            current = page(self.device.image)
            # Gathering can report insufficient stamina after the next search
            # has already started. Hand it back to collect_once immediately.
            if current == 'exhausted' and current not in expected:
                logger.info('Post station transition interrupted by insufficient stamina')
                return None
            stable = stable + 1 if current in expected else 0
            if stable >= 2:
                return current
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                return None
            if stop_on_city:
                city_frames = city_frames + 1 if (
                    current == 'unknown' and self.current_navigation_page() == 'main'
                ) else 0
                if city_frames >= 2:
                    logger.warning('Post station transition returned to city')
                    self.save_abnormal_screenshot('post_station_transition_returned_city')
                    return None
        self.save_abnormal_screenshot('post_station_transition_timeout')
        return None

    def current_navigation_page(self):
        current = page(self.device.image)
        return 'post_station_' + current if current != 'unknown' else super().current_navigation_page()

    def _read_ratio(self, area, expected, switching=False):
        values = []
        for _ in range(2):
            self.device.screenshot()
            if page(self.device.image) != expected:
                return None
            original = crop(self.device.image, area)
            lines = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(
                [original, cv2.resize(original, None, fx=3, fy=3)],
            )
            parsed = [parse_stock(line, switching) for line in lines]
            if len(parsed) != 2 or parsed[0] is None or parsed[0] != parsed[1]:
                logger.warning(f'Post station ratio uncertain: {lines}')
                return None
            values.append(parsed[0])
            self.device.sleep(.15)
        return values[0] if values[0] == values[1] else None

    def read_rations(self, switching=False):
        ratio = self._read_ratio((632, 421, 701, 456) if switching else (592, 461, 655, 497),
                                 'exit' if switching else 'overview', switching)
        self.save(Rations=str(ratio[0]) if ratio else '未识别')
        return ratio

    def open_station(self):
        if page(self.device.image) == 'overview':
            return True
        if not self.switch_city(self.CITY):
            return False
        if not self.search_building('驿站', exact=True, handle_interference=False):
            return False
        self.device.screenshot()
        arrow = self.find_building_search_arrow()
        if arrow is None:
            return False
        box = self._find_station_entrance(arrow)
        if box is None:
            self.save_abnormal_screenshot('post_station_entrance_missing')
            return False
        x, y = (box[0]+box[2])//2, (box[1]+box[3])//2
        self._click_searched_building(Button(
            area=(), color=(), button=(x-3, y-3, x+3, y+3), name='POST_STATION_ENTER',
        ))
        return bool(self.wait_page(('overview',)))

    def _find_station_entrance(self, arrow, timeout=2.0):
        """Wait for the entrance after search overlays stop covering it.

        The search result can leave an NPC speech bubble over the station
        entrance for one or more frames. A single strict template check then
        reports a missing entrance even though the entrance is visible as soon
        as the bubble shifts. Match the full icon first; if its right side is
        covered, require a strict match of the left half and click inside that
        visible half. Never lower the full-icon confidence threshold.
        """
        x, tip, _ = arrow
        region = (
            max(90, x-190), max(30, tip-20),
            min(1170, x+190), min(560, tip+260),
        )
        initial = self.device.image
        visible_half = template('entrance')[:, :24]
        deadline = time.monotonic() + timeout
        while True:
            image = self.device.image
            if image is not initial and not self._map_view_unchanged(initial, image):
                return None
            box = match(image, 'entrance', region, threshold=.78, tone=45)
            partial = False
            if box is None:
                box = match(image, visible_half, region, threshold=.90, tone=26)
                partial = box is not None
            if box is not None:
                center_x = (box[0] + box[2]) // 2
                if abs(center_x - x) <= 70 and tip + 45 <= box[1] <= tip + 240:
                    if partial:
                        logger.info(
                            f'Post station entrance visible half confirmed: '
                            f'box={box}, arrow=({x}, {tip})'
                        )
                    return box
            if time.monotonic() >= deadline:
                return None
            self.device.sleep(.20)
            self.device.screenshot()

    def depart_station(self, timeout=15):
        """Retry a missed departure only while the station overview is visible."""
        deadline = time.monotonic() + timeout
        clicks = 0
        last_click = float('-inf')
        stable = 0
        while time.monotonic() < deadline:
            self.device.screenshot()
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                return False
            current = page(self.device.image)
            if current == 'characters':
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
                if current == 'overview' and clicks < 3 and time.monotonic()-last_click >= 2:
                    clicks += 1
                    logger.info(f'Post station departure click: {clicks}/3')
                    self.click_at(950, 532, 'POST_STATION_DEPART')
                    last_click = time.monotonic()
                elif current not in ('overview', 'unknown'):
                    logger.warning(f'Post station departure changed to {current}')
                    return False
            self.device.sleep(.35)
        logger.warning('Post station departure did not open captain selection')
        self.save_abnormal_screenshot('post_station_departure_timeout')
        return False

    def select_captain(self):
        """Read only six visible cards, never scroll this page."""
        if not self.wait_page(('characters',)):
            return False
        observations = []
        for _ in range(2):
            self.device.screenshot()
            if page(self.device.image) != 'characters':
                return False
            areas = []
            for row in range(3):
                for col in range(2):
                    dx, dy = col * 382, row * 216
                    areas.extend(((458+dx, 72+dy, 583+dx, 110+dy),
                                  (546+dx, 136+dy, 688+dx, 165+dy)))
            lines = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(
                [crop(self.device.image, area) for area in areas])
            candidates = []
            for slot in range(6):
                name = normalize(lines[slot*2])
                ratio = parse_stamina(lines[slot*2+1])
                if ratio and all(n == int(n) for n in ratio):
                    candidates.append(Captain(slot, name, int(ratio[0]), int(ratio[1])))
            observations.append(candidates)
            self.device.sleep(.2)
        confirmed = [c for c in observations[0] if c in observations[1]]
        captain = choose_captain(confirmed, names(self.setting('PreferredCaptains', '')))
        if captain is None:
            self.save_abnormal_screenshot('post_station_no_usable_captain')
            self.save(Status='第一页没有可确认的正体力队长；稍后重试')
            return False
        logger.info(f'Post station captain: {captain.name} {captain.current}/{captain.maximum}')
        self.save(Captain=captain.name or f'第一页第{captain.slot+1}位（姓名未识别）')
        self.click_at(390 + captain.slot % 2 * 382, 170 + captain.slot // 2 * 216, 'POST_STATION_CAPTAIN')
        return bool(self.wait_page(('maps', 'field'), timeout=45, stop_on_city=True))

    def select_map(self, key):
        self._pinned_resource_templates = {}
        self._pinned_search_initialized = False
        if page(self.device.image) == 'field':
            return True  # Switching captains reloads the same destination.
        # Reset to the upper end, then search downward. The map names are vertical.
        for _ in range(2):
            if page(self.device.image) != 'maps':
                return False
            self.device.drag((960, 240), (960, 590), name='POST_STATION_MAP_TOP')
            self.device.sleep(.4)
            self.device.screenshot()
        for attempt in range(4):
            if page(self.device.image) != 'maps':
                return False
            box = match(self.device.image, 'map_' + key, MAP_REGION)
            if box:
                # Save destination before loading, allowing restart recovery.
                self.save(CurrentMap=key)
                self.click_at((box[0]+box[2])//2, (box[1]+box[3])//2, 'POST_STATION_MAP_SELECT')
                return bool(self.wait_page(('field',), timeout=90))
            if attempt < 3:
                self.device.drag((960, 590), (960, 220), name='POST_STATION_MAP_DOWN')
                self.device.sleep(.4)
                self.device.screenshot()
        return False

    def open_pinned_search(self):
        self.device.screenshot()
        current = self._wait_resource_frame()
        if current == 'field':
            self.click_at(1220, 236, 'POST_STATION_ITEM_SEARCH')
            if not self.wait_page(('list',)):
                return False
        elif current != 'list':
            return False
        if not getattr(self, '_pinned_search_initialized', False):
            # Preserve the game's pinned-first order in the All tab. No gather
            # filter or scrolling on subsequent resource searches.
            self.click_at(524,84,'POST_STATION_ITEMS_ALL')
            self.device.sleep(.2)
            self.device.drag((1040,205),(1040,600),name='POST_STATION_PINNED_TOP_ONCE')
            self.device.sleep(.3)
            self.device.screenshot()
            if page(self.device.image) != 'list':
                return False
            self._pinned_search_initialized = True
        return True

    def discover_pinned_resources(self):
        if not self.open_pinned_search():
            return None
        previous = None
        stable = 0
        for _ in range(12):
            boxes = pinned_labels(self.device.image)
            signature = tuple(boxes)
            if (previous is not None and len(previous) == len(signature)
                    and all(max(abs(a-b) for a,b in zip(old,new)) <= 3
                            for old,new in zip(previous,signature))):
                stable += 1
            else:
                stable = 1
            if stable >= 3:
                break
            previous = signature
            self.device.sleep(.15)
            self.device.screenshot()
            if page(self.device.image) != 'list':
                return None
        else:
            self.save_abnormal_screenshot('post_station_pins_unstable')
            return None
        labels = [crop(self.device.image,box).copy() for box in boxes]
        lines = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines(labels) if labels else []
        resources = {}
        for index, label in enumerate(labels):
            title = normalize(lines[index]) if index < len(lines) else ''
            title = title or f'置顶资源{index+1}'
            if title in resources:
                title = f'{title}（{index+1}）'
            resources[title] = label
        self._pinned_resource_templates = resources
        self.save(PinnedResources='、'.join(resources) if resources else '当前地图首页没有置顶资源')
        logger.attr('PostStationPinnedResources', tuple(resources))
        return tuple(resources)

    def open_resource(self, name):
        reference = getattr(self, '_pinned_resource_templates', {}).get(name)
        if reference is None:
            raise ResourceNotFound(name)
        if not self.open_pinned_search():
            return False
        box = self.stable_resource_label(reference)
        if not box:
            if self._resource_list_unstable or page(self.device.image) != 'list':
                return False
            raise ResourceNotFound(name)
        # Still require a pin on this exact card: do not collect an unpinned
        # item if the player edits the game's pins while the task is running.
        if not any(max(abs(a-b) for a,b in zip(pin,box)) <= 6
                   for pin in pinned_labels(self.device.image)):
            raise ResourceNotFound(name)
        self._resource_arrow_samples = []
        self._resource_arrow_frame = None
        self._resource_arrow_at = 0
        logger.info(f'Post station locate pinned resource: {name}, label={box}')
        self.click_at((box[0]+box[2])//2,(box[1]+box[3])//2,'POST_STATION_RESOURCE')
        return self.wait_resource_target()

    def stable_resource_label(self, filename, timeout=3):
        """Require three matching label positions after scrolling/rebounding."""
        deadline = time.monotonic() + timeout
        previous = None
        stable = 0
        absent = 0
        self._resource_list_unstable = False
        while time.monotonic() < deadline:
            if page(self.device.image) != 'list':
                return None
            box = match(self.device.image, filename, (345,112,1080,650), threshold=.87)
            if box:
                self._resource_list_unstable = True
                absent = 0
                stable = stable+1 if previous and max(abs(a-b) for a,b in zip(previous,box)) <= 3 else 1
                previous = box
                if stable >= 3:
                    self._resource_list_unstable = False
                    return box
            else:
                absent += 1
                stable = 0
                previous = None
                if absent >= 3:
                    self._resource_list_unstable = False
                    return None
            self.device.sleep(.2)
            self.device.screenshot()
        return None

    def remember_resource_arrow(self):
        arrow = red_arrow(self.device.image)
        if arrow:
            self._resource_arrow_samples.append(arrow)
            self._resource_arrow_frame = self.device.image.copy()
            self._resource_arrow_at = time.monotonic()
        return arrow

    def _wait_resource_frame(self, timeout=2):
        """Wait through a short blurred transition without sending map input."""
        deadline = time.monotonic() + timeout
        self.check_resource_reachable()
        current = page(self.device.image)
        while current == 'unknown' and time.monotonic() < deadline:
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                return 'interrupted'
            self.device.sleep(.10)
            self.device.screenshot()
            self.check_resource_reachable()
            current = page(self.device.image)
        return current

    @contextmanager
    def _fast_resource_frames(self):
        """Use 100ms frames locally and restore the caller's interval on exit."""
        timer = getattr(self.device, '_screenshot_interval', None)
        previous = getattr(timer, 'limit', None)
        setter = getattr(self.device, 'screenshot_interval_set', None)
        changed = isinstance(previous, (int, float)) and previous > .1 and callable(setter)
        if changed:
            setter(.1)
        try:
            yield
        finally:
            if changed:
                setter(previous)

    def wait_resource_target(self, timeout=8):
        with self._fast_resource_frames():
            return self._wait_resource_target(timeout)

    def _wait_resource_target(self, timeout):
        """Capture the short-lived arrow on the first field frame, before sleeping."""
        deadline = time.monotonic() + timeout
        field_frames = 0
        while time.monotonic() < deadline:
            self.device.screenshot()
            self.check_resource_reachable()
            current = page(self.device.image)
            if current == 'field':
                field_frames += 1
                arrow = self.remember_resource_arrow()
                if arrow:
                    logger.attr('PostStationResourceArrow', arrow)
                    return True
                if bubble(self.device.image):
                    return True
                if field_frames >= 8:
                    logger.warning('Post station resource list closed without a visible target arrow')
                    return True  # collect_once retries the search, without guessing a point.
            elif current not in ('list', 'unknown'):
                logger.warning(f'Post station resource search changed to {current}')
                return False
            self.device.sleep(.05)
        logger.warning('Post station resource search did not return to the field')
        self.save_abnormal_screenshot('post_station_resource_selection_timeout')
        return False

    def locate_bubble(self):
        with self._fast_resource_frames():
            return self._locate_resource_bubble()

    def _locate_resource_bubble(self):
        """Confirm foreground in two fresh frames, then make one precise click."""
        previous_target = None
        previous_frame = None
        started_at = time.monotonic()
        deadline = started_at + 1.8
        while time.monotonic() < deadline:
            self.check_resource_reachable()
            if self._wait_resource_frame() != 'field':
                return None
            found = bubble(self.device.image)
            if found:
                return found
            arrow = red_arrow(self.device.image)
            fresh_arrow = arrow is not None
            if arrow:
                self.remember_resource_arrow()
            else:
                samples = getattr(self, '_resource_arrow_samples', ())
                reference = getattr(self, '_resource_arrow_frame', None)
                age = time.monotonic()-getattr(self, '_resource_arrow_at', 0)
                # Re-segment the current image; never reuse an old click point.
                # A disappearing arrow is usable briefly only for a stationary
                # target and camera, as in the first frame after list dismissal.
                if (samples and age <= 1.2
                        and max(s[0] for s in samples)-min(s[0] for s in samples) <= 12
                        and self._map_view_unchanged(reference, self.device.image)):
                    arrow = samples[-1]
            target = resource_foreground(self.device.image, arrow) if arrow else None
            if target and getattr(target, 'ambiguous', False):
                self._resource_target_ambiguous = True
                logger.warning('Post station foreground joins separate objects; skip without a target or fallback click')
                return None
            if target is None and fresh_arrow:
                target = resource_fallback(arrow)
            if target and resource_target_obstructed(self.device.image, target.point):
                self._resource_target_obstructed = True
                logger.info('Post station resource target is covered by a building roof; skip this turn')
                return None
            if (target and previous_target
                    and target.method == previous_target.method
                    and abs(target.point[0]-previous_target.point[0]) <= 12
                    and (target.method == 'arrow_140'
                         or abs(target.point[1]-previous_target.point[1]) <= 14)
                    and self._map_view_unchanged(previous_frame, self.device.image)):
                logger.attr('PostStationResourceForeground', (target.point, target.box))
                logger.attr('PostStationTargetMethod', target.method)
                logger.attr('PostStationTargetReadyMs', round((time.monotonic()-started_at)*1000))
                self._resource_target_clicked = True
                self.click_at(*target.point, 'POST_STATION_RESOURCE_TARGET')
                # A missing bubble is a failed attempt, not permission to probe
                # other heights. collect_once preserves the normal loop order.
                for _ in range(8):
                    self.device.sleep(.10)
                    self.device.screenshot()
                    self.check_resource_reachable()
                    if self._wait_resource_frame() != 'field':
                        return None
                    found = bubble(self.device.image)
                    if found:
                        logger.attr('PostStationGatherBubble', found)
                        return found
                logger.warning('Post station recognized target clicked without a gathering bubble')
                return None
            previous_target = target
            previous_frame = self.device.image.copy() if target else None
            self.device.sleep(.08)
            self.device.screenshot()
        logger.warning('Post station foreground not confirmed; skip without guessing a click')
        return None

    def _field_stamina_exhausted(self):
        """Confirm zero on the field HUD after a failed target interaction."""
        for attempt in range(2):
            current = page(self.device.image)
            if current == 'exhausted':
                return True
            if current != 'field':
                return False
            # Read only the current value. The separate "最大..." row must
            # never be merged into it or confused with the remaining stamina.
            original = crop(self.device.image, (10, 150, 80, 180))
            readings = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus([
                original, cv2.resize(original, None, fx=3, fy=3),
            ], cand_alphabet='0123456789')
            if len(readings) != 2 or any(str(value).strip() != '0' for value in readings):
                return False
            if attempt == 0:
                self.device.sleep(.15)
                self.device.screenshot()
        logger.info('Post station field stamina confirmed zero; stop gathering')
        return True

    def collect_once(self, resource):
        self._resource_target_clicked = False
        self._resource_target_obstructed = False
        self._resource_target_ambiguous = False
        try:
            result = self._collect_resource(resource)
            if result == 'failed':
                current = self._wait_resource_frame()
                if current == 'craft' or (
                        self._resource_target_clicked
                        and current not in ('field', 'exhausted', 'interrupted')):
                    if current == 'craft':
                        returned = self.close_resource_craft()
                    else:
                        # A target interaction can open another building or
                        # panel. Send one BACK, then require the field before
                        # resuming the resource loop; never confirm that panel.
                        if self.is_network_timeout_popup():
                            self.handle_network_timeout_popup()
                            return 'failed'
                        logger.info(f'Post station target opened {current}; return to field')
                        self.device.press_back()
                        returned = self.wait_page(('field', 'exhausted'), timeout=4)
                    if not returned:
                        return 'failed'
                    if page(self.device.image) == 'exhausted' or self._field_stamina_exhausted():
                        self.save(Status='体力已耗尽，结束采集并进入换队长或回城流程')
                        return 'exhausted'
                    logger.info(f'Post station unexpected panel closed while locating {resource}; '
                                'continue the resource loop')
                    self.save(Status=f'{resource} 点到其他界面，已返回，继续循环采集')
                    return 'skipped'
                if current == 'exhausted' or (current == 'field' and self._field_stamina_exhausted()):
                    self.save(Status='体力已耗尽，结束采集并进入换队长或回城流程')
                    return 'exhausted'
                if current in ('field','list'):
                    logger.warning(f'Post station resource attempt failed; skip this turn only: {resource}')
                    self.save(Status=f'{resource} 本次采集未完成，跳过一次，下一轮继续尝试')
                    return 'skipped'
            return result
        except ResourceNotFound:
            return 'missing'
        except ResourceUnreachable:
            logger.info(f'Post station resource unreachable; skip this expedition: {resource}')
            self.save(Status=f'{resource} 无法到达，本次出征不再尝试，继续其他资源')
            # A floating toast can overlap the start of the next resource.
            # Let it disappear before attributing a new rejection to that item.
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                self.device.sleep(.10)
                self.device.screenshot()
                if not unreachable(self.device.image):
                    return 'unreachable'
            logger.warning('Post station unreachable toast did not disappear')
            return 'failed'

    def check_resource_reachable(self):
        if unreachable(self.device.image):
            raise ResourceUnreachable()

    def _collect_resource(self, resource):
        if not self.open_resource(resource):
            if page(self.device.image) == 'craft':
                return 'failed'  # collect_once dismisses this known non-gathering panel.
            if page(self.device.image) == 'exhausted':
                return 'exhausted'
            logger.warning(f'Post station resource selection failed: {resource}; '
                           'do not mark this item as absent')
            self.save_abnormal_screenshot('post_station_resource_selection_failed')
            return 'failed'
        box = self.locate_bubble()
        if box is None:
            if getattr(self, '_resource_target_ambiguous', False):
                self.save_abnormal_screenshot('post_station_target_ambiguous')
                self.save(Status=f'{resource} 目标与相邻物体混连，本轮跳过，稍后重试')
                return 'skipped'
            if getattr(self, '_resource_target_obstructed', False):
                self.save(Status=f'{resource} 被建筑遮挡，本轮跳过，下一轮继续尝试')
                return 'skipped'
            current = page(self.device.image)
            if (current in ('craft', 'exhausted')
                    or (getattr(self, '_resource_target_clicked', False) and current != 'field')):
                return 'exhausted' if current == 'exhausted' else 'failed'
            self.save_abnormal_screenshot('post_station_arrow_or_bubble_missing')
            return 'failed'
        point = bubble_click_point(self.device.image, box)
        if point is None:
            logger.warning('Post station gathering bubble has no point clear of the arrow')
            return 'failed'
        # Confirm inside the bubble, clear of the moving arrow.
        self.click_at(*point, 'POST_STATION_GATHER_CONFIRM')
        # Rejection toasts can arrive after an initially unchanged field.
        # Observe for 1.2s without opening another search, so a late rejection
        # stays attributed to this resource. Stop immediately on exhaustion.
        with self._fast_resource_frames():
            for attempt in range(7):
                self.device.sleep(.3 if attempt == 0 else .15)
                self.device.screenshot()
                current = self._wait_resource_frame()
                if current == 'exhausted':
                    self.save(Status='体力不足，结束采集并进入换队长或回城流程')
                    return 'exhausted'
                if current != 'field':
                    logger.warning(f'Post station gather confirmation changed to {current}')
                    return 'failed'
        self.save(Status=f'{resource} 已点击采集，继续下一项')
        return 'submitted'

    def gather(self, key, resources=None):
        if resources is None:
            resources = self.discover_pinned_resources()
            if resources is None:
                return False
            if not resources:
                self.save(Status=f'{self.MAP_NAMES[key]} 没有置顶资源，跳过此地图')
                return 'empty'
        missing = set()
        deferred = set()
        skipped = {}
        no_progress = 0
        while True:
            progressed = False
            for resource in resources:
                if resource in missing or resource in deferred:
                    continue
                # Different resource names still use the same two device
                # button names. Leave room for cleanup before the global
                # alternating-click watchdog (six of each) would fire.
                history = getattr(self.device, 'click_record', ())
                if (history.count('POST_STATION_ITEM_SEARCH') >= 5
                        and history.count('POST_STATION_RESOURCE') >= 5):
                    logger.warning('Post station search click budget reached; return safely before retry')
                    self.save(Status=f'{self.MAP_NAMES[key]} 连续搜索未完成，安全返回后重试')
                    return False
                result = self.collect_once(resource)
                if result == 'exhausted':
                    return True
                if result == 'failed':
                    return False
                if result == 'missing':
                    missing.add(resource)
                else:
                    if result in ('skipped', 'unreachable'):
                        no_progress += 1
                        if result == 'unreachable':
                            deferred.add(resource)
                            logger.warning(f'Post station defer unreachable resource for this expedition: {resource}')
                        # Bound failures across resources as well as per item.
                        # Do not erase device history or disable its watchdog.
                        if no_progress >= 4:
                            logger.warning('Post station stop after 4 consecutive unsuccessful resource attempts')
                            self.save(Status=f'{self.MAP_NAMES[key]} 连续4次采集未完成，安全返回后重试')
                            return False
                        if result == 'unreachable':
                            continue
                        skipped[resource] = skipped.get(resource, 0) + 1
                        if skipped[resource] >= 3:
                            deferred.add(resource)
                            logger.warning(
                                f'Post station defer resource after 3 consecutive '
                                f'unsuccessful attempts: {resource}'
                            )
                            continue
                    else:
                        no_progress = 0
                        skipped[resource] = 0
                    # Retry transient recognition failures in list order.
                    # Confirmed unreachable resources stay deferred until a
                    # new expedition; no resource or map setting is disabled.
                    progressed = True
            if not progressed:
                if deferred:
                    self.save(Status=f'{self.MAP_NAMES[key]} 部分资源连续失败，稍后重试')
                    return False
                self.save(Status=f'{self.MAP_NAMES[key]} 已无可采集的置顶资源，跳过此地图')
                return 'empty'

    def close_resource_craft(self):
        if page(self.device.image) != 'craft':
            return False
        # Tap outside the paper panel; the make button spends materials/stamina.
        self.click_at(1145, 520, 'POST_STATION_CLOSE_CRAFT')
        return bool(self.wait_page(('field',), timeout=5))

    def exit_dialog(self):
        self.device.screenshot()
        current = page(self.device.image)
        if current == 'craft':
            if not self.close_resource_craft():
                return False
            current = page(self.device.image)
        if current == 'exhausted':
            self.click_at(1020, 610, 'POST_STATION_DISMISS_EXHAUSTED')
            # Exhaustion can arrive after the next item search has already
            # opened its list. Dismissing it restores that list, not the field.
            if not self.wait_page(('field', 'list', 'exit')):
                return False
            current = page(self.device.image)
        if current == 'list':
            self.click_at(1145, 520, 'POST_STATION_CLOSE_LIST')
            if not self.wait_page(('field', 'exit')):
                return False
        if page(self.device.image) == 'exit':
            return True
        if page(self.device.image) != 'field':
            return False
        # A failed collection can leave a bubble; dismiss it before the exit.
        if bubble(self.device.image):
            self.click_at(1020, 540, 'POST_STATION_CLOSE_BUBBLE')
            self.device.sleep(.25)
        self.click_at(1220, 635, 'POST_STATION_LEAVE_MAP')
        return bool(self.wait_page(('exit',)))

    def return_city(self):
        self.click_at(816, 494, 'POST_STATION_RETURN_CITY')
        # Loading can last tens of seconds. Never send BACK during this period.
        deadline = time.monotonic() + 90
        main_frames = 0
        while time.monotonic() < deadline:
            self.device.sleep(.5)
            self.device.screenshot()
            if self.handle_startup_popup():
                main_frames = 0
                logger.info('Popup handled after post station return; recheck city')
                continue
            if page(self.device.image) == 'unknown' and super().current_navigation_page() == 'main':
                main_frames += 1
                if main_frames >= 2:
                    self.save(CurrentMap='')
                    return True
            else:
                main_frames = 0
        self.save_abnormal_screenshot('post_station_return_city_timeout')
        return False

    def _cleanup(self):
        current = page(self.device.image)
        if current in ('field', 'list', 'exhausted', 'exit', 'craft'):
            return self.exit_dialog() and self.return_city()
        if current in ('characters', 'maps', 'overview'):
            # These ordinary paper panels close by tapping outside the right edge.
            for _ in range(3):
                self.click_at(1145, 520, 'POST_STATION_CLOSE_PANEL')
                self.device.sleep(.4)
                self.device.screenshot()
                current = page(self.device.image)
                if current == 'exit':
                    return self.return_city()
                if current == 'unknown':
                    return super().ui_goto_main()
            return False
        return super().ui_goto_main()

    def run(self):
        logger.hr(f'JIANGNAN POST STATION {self.CITY}', level=0)
        minutes = interval_minutes(self.setting('IntervalRations', 1))
        self.save(Interval=interval_label(self.setting('IntervalRations', 1)))
        available = enabled_maps({key:self.setting(field,True) for key,field in self.MAP_FIELDS.items()}, self.MAP_NAMES)
        if not available:
            self.save(Status='未勾选地图，本次跳过驿站')
            self.config.task_delay(minute=minutes)
            return True
        threshold = Decimal(str(self.setting('RationThreshold', 3)))
        if not threshold.is_finite() or threshold < 0:
            raise ValueError('粮草阈值必须是非负数')
        self.device.screenshot()
        # Recover a previous interrupted expedition before starting a new one.
        if page(self.device.image) != 'overview' and not self._cleanup():
            self.save(Status='未能从上次界面安全返回；等待重试')
            self.config.task_delay(success=False)
            return False
        target = next_map(available, self.setting('LastMap', ''), self.MAP_NAMES)
        success = True
        try:
            while target:
                if not self.open_station():
                    success = False
                    break
                rations = self.read_rations()
                if rations is None:
                    self.save(Status='粮草识别不确定，未点击出发')
                    success = False
                    break
                if not can_depart(rations[0], threshold):
                    self.save(Status=f'粮草 {rations[0]} 未大于阈值 {threshold}；等待恢复')
                    break
                if not self.depart_station():
                    success = False
                    break
                if not self.select_captain() or not self.select_map(target):
                    success = False
                    break
                while True:
                    exhausted = self.gather(target)
                    if not exhausted:
                        success = False
                        break
                    if exhausted == 'empty':
                        self.save(LastMap=target, **{
                            self.MAP_FIELDS[target]: False,
                            'Status': f'{self.MAP_NAMES[target]} 没有可采集的置顶资源，已自动取消勾选',
                        })
                        logger.info(f'Post station map auto-disabled without pinned resources: {target}')
                        available = tuple(key for key in available if key != target)
                    else:
                        self.save(LastMap=target)
                    upcoming = next_map(available, target, self.MAP_NAMES)
                    if not self.exit_dialog():
                        success = False
                        break
                    # Read stock/cost here too. Switching may cost multiple rations.
                    remaining = self.read_rations(switching=True)
                    if (exhausted is True and upcoming == target and remaining
                            and can_depart(remaining[0], threshold, cost=remaining[1])):
                        self.click_at(644, 494, 'POST_STATION_SWITCH_CAPTAIN')
                        if self.select_captain() and self.select_map(target):
                            continue
                        success = False
                        break
                    if not self.return_city():
                        success = False
                        break
                    target = upcoming
                    break
                if not success:
                    break
        except Exception:
            # Cleanup can succeed after a workflow exception. It makes the UI
            # safe, but must not turn an interrupted expedition into success.
            success = False
            raise
        finally:
            self.device.screenshot()
            cleaned = self._cleanup()
            success = success and cleaned
            if not success:
                self.save_abnormal_screenshot('post_station_incomplete')
                self.save(Status=(
                    '驿站流程未完成；已安全返回城市，10分钟后重试'
                    if cleaned else '驿站流程未完成；未能安全返回城市，需要检查'
                ))
            self.save(LastRun=datetime.now().replace(microsecond=0))
            self.config.task_delay(minute=minutes if success else 10)
        # The scheduler's boolean means whether it is safe to continue other
        # tasks. Keep an incomplete expedition visible in Status and retry it
        # in ten minutes, without turning recovered UI misses into a global
        # stop after three runs. Failed cleanup must still propagate failure.
        return cleaned


def station_runner(task, config, device=None):
    """All city tasks share the same tested workflow with isolated config keys."""
    spec = STATIONS[task]
    cls = type(task, (JiangnanYingtianPostStation,), {
        'TASK': task, 'GROUP': spec['group'], 'CITY': spec['city'],
        'MAP_NAMES': spec['maps'], 'MAP_FIELDS': map_fields(spec['maps']),
    })
    return cls(config=config, device=device, task=task)
