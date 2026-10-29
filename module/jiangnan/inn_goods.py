"""Position the counter, then click a verified keeper reward or ledger bubble.

After the bubble click, confirm the gem reward and accept it once. Completion
requires two fresh, unobstructed inn frames before normal exit can continue.
"""
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from module.jiangnan.inn_map import (
    INN_MAP_DIR, _read_rgb, locate_inn_view, map_to_screen,
)
from module.jiangnan.inn_schedule import (
    GOODS_GROUP, current_goods_slot, goods_next_run, inn_now,
    next_goods_slot, slot_key,
)
from module.jiangnan.inn_workflow import InnFeatureResult


COUNTER_REFERENCE = INN_MAP_DIR.parent / 'goods_counter.png'
COUNTER_CROP = (620, 378, 752, 418)  # Wooden front of desk only; no person/bubble.
COUNTER_MAP_POINT = (1097, 1431)
KEEPER_MAP_POINT = (1094, 1368)
# 老板娘本人的位置只用于理解布局，不能作为点击目标。实际目标是她头上
# 补天石图标气泡；地图投影仅限定搜索范围，最终点击坐标来自当前截图。
KEEPER_BUBBLE_MAP_POINT = (1105, 1293)
KEEPER_BUBBLE_REFERENCE = INN_MAP_DIR.parent / 'keeper_bubble.png'
KEEPER_LEDGER_REFERENCE = INN_MAP_DIR.parent / 'keeper_ledger.png'
KEEPER_WINE_REFERENCE = INN_MAP_DIR.parent / 'keeper_wine.png'

# Match only the day/time header, never the changing reward icon/name below it.
KEEPER_NOTICE_HEADERS = (('tomorrow', 10, 'keeper_notice_tomorrow10.png'),
                         ('today', 16, 'keeper_notice_today16.png'))


def find_keeper_next_notice(image, location, now):
    """A future领取 banner means this slot is already claimed, not clickable."""
    if image is None or location is None or image.shape[:2] != (720, 1280):
        return None
    px, py = keeper_screen_point(location)
    if not (210 <= px <= 1040 and 90 <= py <= 480):
        return None
    scale = 1/location['scale']
    left, top = max(100, int(px-110*scale)), max(25, int(py-65*scale))
    right, bottom = min(1150, int(px+110*scale)), min(545, int(py+30*scale))
    roi = image[top:bottom, left:right]
    for day, hour, name in KEEPER_NOTICE_HEADERS:
        due = (now+timedelta(days=1) if day == 'tomorrow' else now).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        if due <= now:
            continue  # An expired banner never proves today's slot was claimed.
        path = INN_MAP_DIR.parent / name
        template = cv2.resize(_bubble_template(str(path), path.stat().st_mtime_ns), None,
                              fx=scale, fy=scale)
        h, w = template.shape[:2]
        if roi.shape[0] < h or roi.shape[1] < w:
            continue
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(
            cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), cv2.cvtColor(template, cv2.COLOR_RGB2GRAY),
            cv2.TM_CCOEFF_NORMED))
        # Small text loses subpixel alignment when the game zooms out. Retain
        # the full day/time header, brightness gate and two-frame confirmation.
        threshold = 0.87 if scale < 0.95 else 0.91
        if score < threshold or np.abs(roi[y:y+h, x:x+w].astype(np.int16)-template.astype(np.int16)).mean() > 25:
            continue
        return due
    return None


@lru_cache(maxsize=2)
def _counter_template(path, modified_ns):
    x1, y1, x2, y2 = COUNTER_CROP
    return _read_rgb(path)[y1:y2, x1:x2]


def keeper_screen_point(location):
    return map_to_screen(location, KEEPER_BUBBLE_MAP_POINT)


@lru_cache(maxsize=2)
def _bubble_template(path, modified_ns):
    return _read_rgb(path)


def find_keeper_click(image, location):
    match = find_keeper_bubble(image, location)
    return match[1] if match else None


def find_keeper_bubble(image, location):
    """先确认柜台，再识别老板娘头上的领取气泡，绝不回退点击人物。

    识别补天石、蓝色账本或酒坛及其不透明纸底，返回类型和实际点击范围。
    背景和人物不参与；识别到气泡不等同于已完成领取。
    未到点的“今日16:00领取补天石”矩形提示不视作可领取气泡。
    """
    if image is None or location is None or image.shape[:2] != (720, 1280):
        return None
    px, py = keeper_screen_point(location)
    cx, cy = map_to_screen(location, COUNTER_MAP_POINT)
    if not (250 <= px <= 1020 and 100 <= py <= 490):
        return None
    reference = Path(COUNTER_REFERENCE)
    template = _counter_template(str(reference), reference.stat().st_mtime_ns)
    scale = 1 / location['scale']
    template = cv2.resize(template, None, fx=scale, fy=scale)
    th, tw = template.shape[:2]
    left, top = max(90, int(cx-tw/2-45)), max(30, int(cy-th/2-45))
    right, bottom = min(1150, int(cx+tw/2+45)), min(545, int(cy+th/2+45))
    if right-left < tw or bottom-top < th:
        return None
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
    if score < 0.82:
        return None
    dx, dy = left+point[0]+tw/2-cx, top+point[1]+th/2-cy
    if abs(dx) > 32 or abs(dy) > 32:
        return None
    # 柜台验证只是身份锚点，不能直接按地图上的老板娘位置点击。
    # 在头上气泡的预期位置附近另找实际图标，兼容轻微浮动和镜头缩放。
    for kind, reference in (('gem', KEEPER_BUBBLE_REFERENCE), ('ledger', KEEPER_LEDGER_REFERENCE),
                            ('wine', KEEPER_WINE_REFERENCE)):
        reference = Path(reference)
        bubble = _bubble_template(str(reference), reference.stat().st_mtime_ns)
        bubble = cv2.resize(bubble, None, fx=scale, fy=scale)
        bh, bw = bubble.shape[:2]
        bx, by = px+dx, py+dy
        left, top = max(100, int(bx-bw/2-35)), max(30, int(by-bh/2-35))
        right, bottom = min(1150, int(bx+bw/2+35)), min(545, int(by+bh/2+35))
        if right-left < bw or bottom-top < bh:
            continue
        roi = image[top:bottom, left:right]
        _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(
            cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), cv2.cvtColor(bubble, cv2.COLOR_RGB2GRAY),
            cv2.TM_CCOEFF_NORMED))
        if score < 0.88:
            continue
        patch = roi[point[1]:point[1]+bh, point[0]:point[0]+bw]
        if np.abs(patch.astype(np.int16)-bubble.astype(np.int16)).mean() > 25:
            continue  # 遮罩后的气泡不可点击。
        x, y = int(round(left+point[0]+bw/2)), int(round(top+point[1]+bh/2))
        return kind, (x-8, y-8, x+8, y+8)
    return None


def keeper_drag(location):
    """Move the projected keeper bubble toward a safe center with a short drag."""
    x, y = keeper_screen_point(location)
    dx = int(round(np.clip(640-x, -390, 390)))
    dy = int(round(np.clip(320-y, -220, 220)))
    # End above the fixed bottom-right menu even for diagonal drags.
    if 640+dx > 870:
        dy = min(dy, 140)
    if abs(dx) < 35 and abs(dy) < 35:
        return None
    return (640, 380), (640+dx, 380+dy)


def find_reward_accept(image, reward_kind='gem'):
    panel = _find_reward_panel(image, reward_kind)
    return panel[0] if panel else None


def _find_reward_panel(image, reward_kind):
    # Ledger bubbles can yield either gems or the weekly ticket observed on
    # Oct 28. Require a known title together with the accept control.
    titles = {
        'gem': (('reward_title', (-8, 380), '补天石'),),
        'ledger': (('reward_title', (-8, 380), '补天石'),
                   ('reward_weekly_ticket_title', (0, 380), '周加速票')),
        'wine': (('reward_wine_title', (16, 380), '精酿三白酒'),),
    }
    for title_name, button_offset, reward_name in titles.get(reward_kind, ()):
        target = _find_reward_accept(image, title_name, button_offset)
        if target is not None:
            return target, reward_name
    return None


def _find_reward_accept(image, title_name, button_offset):
    """同时识别已知奖励标题和“收下”，不匹配模糊背景或奖励数量。

    模板只包含不透明控件。确认标题与按钮相对位置，避免在其他弹窗上
    仅凭一个通用的红按钮就点击，也兼容弹窗整体的小幅位移。
    """
    if image is None or image.shape[:2] != (720, 1280):
        return None
    found = []
    for name, region in (
        (title_name, (520, 110, 765, 220)),
        ('reward_accept', (450, 475, 830, 640)),
    ):
        path = INN_MAP_DIR.parent / (name+'.png')
        template = _bubble_template(str(path), path.stat().st_mtime_ns)
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        h, w = template.shape[:2]
        _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(
            cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), cv2.cvtColor(template, cv2.COLOR_RGB2GRAY),
            cv2.TM_CCOEFF_NORMED))
        if score < 0.91:
            return None
        x, y = point
        if np.abs(roi[y:y+h, x:x+w].astype(np.int16)-template.astype(np.int16)).mean() > 25:
            return None
        found.append((x1+x, y1+y, x1+x+w, y1+y+h))
    title, button = found
    if max(abs((button[0]-title[0])-button_offset[0]),
           abs((button[1]-title[1])-button_offset[1])) > 6:
        return None
    return button


class InnGoodsCollector:
    MAX_DRAGS = 6
    SEARCH_DRAGS = ((-300, 0), (0, -200), (380, 0), (0, 200), (-380, 0), (0, -200))

    def __init__(self, inn, clock=inn_now):
        self.inn, self.config, self.clock = inn, inn.config, clock

    def _get(self, field, default=''):
        return self.config.cross_get('%s.%s.%s' % (self.inn.TASK, GOODS_GROUP, field), default=default)

    def _set(self, field, value):
        self.config.modified['%s.%s.%s' % (self.inn.TASK, GOODS_GROUP, field)] = value

    def _retry(self, now, message):
        self._set('RetrySlot', slot_key(current_goods_slot(now)))
        return InnFeatureResult('retry', message, min(now+timedelta(minutes=10), next_goods_slot(now)))

    def _locate(self):
        try:
            return locate_inn_view(self.inn.device.image)
        except (OSError, ValueError, cv2.error):
            return None

    def _position_counter(self):
        self.already_claimed = False
        for attempt in range(self.MAX_DRAGS+1):
            self.inn.device.screenshot()
            if self.inn.handle_network_timeout_popup():
                return None
            if self.inn.is_inn_exit_confirm() or not self.inn.is_in_inn():
                return None
            location = self._locate()
            if location is not None:
                observed_slot = slot_key(current_goods_slot(self.clock()))
                notice = find_keeper_next_notice(self.inn.device.image, location, self.clock())
                if notice is not None:
                    self.inn.device.sleep(0.25)
                    self.inn.device.screenshot()
                    if (self.inn.handle_network_timeout_popup() or self.inn.is_inn_exit_confirm()
                            or not self.inn.is_in_inn()):
                        return None
                    current = self._locate()
                    fresh = find_keeper_next_notice(self.inn.device.image, current, self.clock())
                    if fresh == notice and slot_key(current_goods_slot(self.clock())) == observed_slot:
                        self.already_claimed = True
                        return None
                    continue
                match = find_keeper_bubble(self.inn.device.image, location)
                if match is not None:
                    kind, target = match
                    # Confirm against a second frame; never click stale coordinates
                    # after inertia, a drag, a popup or a network recovery.
                    self.inn.device.sleep(0.25)
                    self.inn.device.screenshot()
                    if self.inn.handle_network_timeout_popup():
                        return None
                    if self.inn.is_inn_exit_confirm() or not self.inn.is_in_inn():
                        return None
                    current = self._locate()
                    fresh = find_keeper_bubble(self.inn.device.image, current) if current else None
                    if fresh is not None and fresh[0] == kind and max(
                            abs(a-b) for a, b in zip(fresh[1], target)) <= 6:
                        self.bubble_kind = kind
                        return fresh[1]
                    continue
                drag = keeper_drag(location)
                if drag is None:
                    return None  # 柜台或头上气泡未确认，不能回退点击老板娘本人。
            else:
                dx, dy = self.SEARCH_DRAGS[min(attempt, len(self.SEARCH_DRAGS)-1)]
                drag = (640, 380), (640+dx, 380+dy)
            if attempt >= self.MAX_DRAGS:
                break
            self.inn.jiangnan_drag(drag[0], drag[1], name='INN_KEEPER_POSITION')
            self.inn.device.sleep(0.7)
        return None

    def _collect_reward(self):
        """等弹窗稳定后只点击一次“收下”，再确认实际返回客栈。

        点击气泡后可能有延迟或动画，不能立即按固定坐标连续点击。
        网络恢复会改变画面，发现恢复后交回上层保留现场，不沿用旧目标。
        """
        kind = getattr(self, 'bubble_kind', 'gem')
        previous = None
        for _ in range(40):  # 最多等待约 12 秒。
            self.inn.device.sleep(0.3)
            self.inn.device.screenshot()
            if self.inn.handle_network_timeout_popup():
                return False
            panel = _find_reward_panel(self.inn.device.image, kind)
            if (panel is not None and previous is not None
                    and panel[1] == previous[1] and max(
                        abs(a-b) for a, b in zip(panel[0], previous[0])) <= 6):
                break
            previous = panel
        else:
            return False
        # 只发一次输入；若传输异常或窗口未消失，不盲目补发，以免点击穿透。
        self.inn.click_inn_reward(panel[0])
        stable = 0
        for _ in range(40):
            self.inn.device.sleep(0.3)
            self.inn.device.screenshot()
            if self.inn.handle_network_timeout_popup():
                return False
            returned = (find_reward_accept(self.inn.device.image, kind) is None
                        and not self.inn.is_inn_exit_confirm() and self.inn.is_in_inn()
                        and self._locate() is not None)
            # 奖励窗里的底部菜单仍然可见，因此还要确认建筑定位成功，
            # 不能仅凭菜单或“收下”暂时消失就把模糊/过渡画面当作领取成功。
            stable = stable+1 if returned else 0
            if stable >= 2:
                self._collected_reward_name = panel[1]
                return True
        return False

    def run(self, now=None):
        now = now or self.clock()
        due = goods_next_run(now, self._get('LastSlot'), self._get('AttemptedSlot'),
                             self._get('RetrySlot'), self._get('NextRun'))
        if due > now:
            return InnFeatureResult('done', '[等待下一收获时段]', due)
        try:
            target = self._position_counter()
        except (OSError, ValueError, cv2.error):
            target = None
        if getattr(self, 'already_claimed', False):
            checked_at = self.clock()
            key = slot_key(current_goods_slot(checked_at))
            if key:
                self._set('LastSlot', key)
            self._set('RetrySlot', '')
            status = '[已显示下一时段领取提示][本次已领过，跳过]'
            self._set('Status', status)
            self.config.update()
            return InnFeatureResult('done', status, next_goods_slot(checked_at))
        if target is None:
            return self._retry(now, '[老板娘柜台或头上气泡未确认][未点击，等待复查]')

        click_time = self.clock()
        key = slot_key(current_goods_slot(click_time))
        if not key or key in (self._get('LastSlot'), self._get('AttemptedSlot')):
            return InnFeatureResult('done', '[本时段已处理或尚未到点]', next_goods_slot(click_time))
        # Flush intent BEFORE device input. An ambiguous click or process crash
        # must not produce a second keeper click when the scheduler restarts.
        self._set('AttemptedSlot', key)
        self._set('RetrySlot', '')
        self._set('Status', '[准备点击老板娘头上气泡][等待领取奖励]')
        self.config.update()
        try:
            self.inn.click_inn_keeper(target)
        except Exception:
            self._set('Status', '[点击结果不明][同一时段不自动重发，请查看画面]')
            self.config.update()
            return InnFeatureResult('retry', self._get('Status'), next_goods_slot(click_time), keep_open=True)
        self._set('LastClick', click_time)
        kind = getattr(self, 'bubble_kind', 'gem')
        waiting = {'ledger': '[已点击老板娘蓝色账本气泡][等待识别后续画面]',
                   'wine': '[已点击老板娘酒坛领取气泡][等待识别奖励画面]',
                   'gem': '[已点击老板娘头上气泡][等待收下补天石]'}
        self._set('Status', waiting[kind])
        self.config.update()
        try:
            collected = self._collect_reward()
        except Exception:
            # 气泡点击已持久化；收下操作结果不明时，同一时段不重发气泡点击。
            collected = False
        if not collected:
            message = {'ledger': '蓝色账本后续画面未确认', 'wine': '酒坛奖励后续画面未确认',
                       'gem': '补天石收下未确认'}[kind]
            status = '['+message+'][保留现场，同一时段不自动重发]'
            self._set('Status', status)
            self.config.update()
            try:
                self.inn.save_abnormal_screenshot('inn_keeper_reward_unconfirmed')
            except (OSError, ValueError):
                pass
            return InnFeatureResult('retry', status, next_goods_slot(click_time), keep_open=True)
        # LastSlot records confirmed collection or an explicit next-time banner.
        # 旧配置中的 LastSlot 仍参与防重复，不能因升级就重领同一时段。
        self._set('LastSlot', key)
        reward_name = getattr(self, '_collected_reward_name',
                              '精酿三白酒' if kind == 'wine' else '补天石')
        status = '[已收下'+reward_name+'][已返回客栈]'
        self._set('Status', status)
        self.config.update()
        return InnFeatureResult('done', status, next_goods_slot(click_time))
