"""客栈食客的预备实现；默认关闭，开启前可单独调试这里的纯识别函数。

流程：定位桌子 -> 找物资气泡（不看对号）-> 打开需求 -> 白名单和库存判断
-> 给予或赶走 -> 确认返回客栈。烹饪、上菜和老板娘收获不属于本模块。
"""
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

from module.jiangnan.inn_map import _read_rgb, locate_inn_view, map_to_screen
from module.jiangnan.inn_workflow import InnFeatureResult


DINERS_GROUP = 'yingtian_inn_diners'
DINERS_INTERVAL = 12
ASSETS = Path(__file__).resolve().parents[2] / 'assets/jiangnan/inn/diners'
# 当前客栈布局中的六张桌子，坐标属于全景地图，不能直接用于屏幕点击。
# 每次拖动后都重新用建筑定位，再投影到当前屏幕；人物和气泡不参与地图定位。
# 如果以后玩家移动了桌子，应重新标注这里，而不是放宽气泡匹配去猜位置。
TABLES = ((1320, 1570), (1667, 1510), (1629, 1725),
          (1938, 1640), (1901, 1909), (1743, 1207))


@dataclass(frozen=True)
class DinerDemand:
    resource: str
    quantity: int
    stock: int


def sale_whitelist(value):
    """空白名单也有明确含义：所有已确认的食客需求都赶走，不售卖。

    只按完整物资名称匹配，不做模糊匹配，也不把图标识别结果当作物资名称。
    支持 UI 中逐行填写，或使用逗号、顿号、分号分隔。
    """
    if not isinstance(value, str):
        return frozenset()
    return frozenset(item for item in re.split(r'[\s,，、;；]+', value.strip()) if item)


def parse_demand(resource_line, stock_line):
    """输入必须分别来自“羊毛180个”和“(1790)”的裁剪，不接收整窗文字。

    不用强制数字字符集，以免把本来不可靠的中文/符号 OCR 变成可用数字。
    不能识别、数量为零、缺少名称、负数等均返回 None，绝不默认成零库存。
    """
    line = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(resource_line or '')))
    stock = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(stock_line or '')))
    demand = re.fullmatch(r'([\u4e00-\u9fff]{1,12})([1-9][0-9]{0,7})(?:个|份|件|根|匹|张|块|卷|箱)?', line)
    inventory = re.fullmatch(r'\(?([0-9]{1,8}|[1-9][0-9]{0,2}(?:,[0-9]{3})+)\)?', stock)
    if not demand or not inventory or stock.startswith('(') != stock.endswith(')'):
        return None
    return DinerDemand(demand[1], int(demand[2]), int(inventory[1].replace(',', '')))


def should_give(demand, whitelist):
    # 用户要求严格小于：180 < 1790 可给，180 == 180 也必须赶走。
    return (isinstance(demand, DinerDemand) and demand.resource in whitelist
            and 0 < demand.quantity < demand.stock)


@lru_cache(maxsize=5)
def _template(name):
    return cv2.cvtColor(_read_rgb(ASSETS / (name + '.png')), cv2.COLOR_RGB2GRAY)


def _match(image, name, region, threshold=0.91):
    x1, y1, x2, y2 = region
    roi = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    template = _template(name)
    h, w = template.shape
    if roi.shape[0] < h or roi.shape[1] < w:
        return None
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
    if score < threshold:
        return None
    patch = roi[y:y+h, x:x+w]
    # 遮罩/转场中的按钮即使轮廓相同，也不允许点击。
    if np.abs(patch.astype(np.int16)-template.astype(np.int16)).mean() > 25:
        return None
    return x+x1, y+y1, x+x1+w, y+y1+h


def demand_panel(image):
    """同时确认竖排标题、赶走、给予，背景和奖励列表完全不参与匹配。"""
    if image is None or image.shape[:2] != (720, 1280):
        return None
    title = _match(image, 'request_title', (175, 40, 340, 420))
    expel = _match(image, 'expel', (455, 562, 725, 676))
    give = _match(image, 'give', (695, 562, 965, 676))
    if not title or not expel or not give:
        return None
    dx, dy = title[0]-215, title[1]-76
    if max(abs(expel[0]-495-dx), abs(expel[1]-592-dy),
           abs(give[0]-735-dx), abs(give[1]-592-dy)) > 6:
        return None
    # 库存来自该弹窗（可能是其他城市的仓库），不用当前城市缓存或资源账本。
    return {'demand': (483+dx, 176+dy, 750+dx, 218+dy),
            'stock': (374+dx, 238+dy, 482+dx, 273+dy),
            'expel': expel, 'give': give}


def find_table_bubble(image, location, table):
    """只找指定桌子上方的完整气泡。对号、物资图标均不作为判断依据。

    mask 只保留奶白色气泡的外框，遮掉中央图案，故无需为每种物资制作模板。
    搜索区域来自实时地图投影；老板娘的矩形提示及底部菜单不在目标区域。
    """
    if image is None or image.shape[:2] != (720, 1280) or location is None:
        return None
    x, y = map_to_screen(location, table)
    scale = 1 / location['scale']
    left, top = max(100, int(x-85*scale)), max(25, int(y-195*scale))
    right, bottom = min(1175, int(x+85*scale)), min(695, int(y-50*scale))
    if right <= left or bottom <= top:
        return None
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    template, mask = _template('bubble'), _template('bubble_mask')
    best = None
    for adjustment in (1.0, 0.97, 1.03):
        size = scale*adjustment
        ref = cv2.resize(template, None, fx=size, fy=size)
        weights = cv2.resize(mask, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = ref.shape
        if gray.shape[0] < h or gray.shape[1] < w:
            continue
        scores = cv2.matchTemplate(gray, ref, cv2.TM_SQDIFF_NORMED, mask=weights)
        scores[~np.isfinite(scores)] = 1
        # 仅屏幕右下角被菜单遮挡，不能把整条 y>545 的区域都丢弃。
        scores[max(0, 560-top-h):, max(0, 890-left-w):] = 1
        score, _, (bx, by), _ = cv2.minMaxLoc(scores)
        if score <= 0.075 and (best is None or score < best[0]):
            # 点击气泡中心小区域，避免触碰桌子/客人/气泡外的其他控件。
            cx, cy = left+bx+w//2, top+by+h//2
            best = score, (cx-7, cy-7, cx+7, cy+7)
    return best[1] if best else None


def read_panel_demand(image, panel):
    # 延迟导入 OCR，便于对纯识别、白名单、调度进行无设备的离线测试。
    from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
    crops = []
    for key in ('demand', 'stock'):
        x1, y1, x2, y2 = panel[key]
        crop = image[y1:y2, x1:x2]
        crops.append(cv2.copyMakeBorder(crop, 6, 6, 8, 8, cv2.BORDER_REPLICATE))
    # 每项都要求两个模型给出相同文字；后续操作还会跨两张截图再确认一次。
    texts = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(crops)
    return parse_demand(*texts) if len(texts) == 2 else None


class InnDiners:
    MAX_DRAGS = 6

    def __init__(self, inn):
        self.inn, self.config = inn, inn.config

    def _get(self, field, default=None):
        return self.config.cross_get('%s.%s.%s' % (self.inn.TASK, DINERS_GROUP, field), default=default)

    def _capture(self):
        self.inn.device.screenshot()
        return not self.inn.handle_network_timeout_popup()

    def _scene(self):
        if self.inn.is_inn_exit_confirm() or not self.inn.is_in_inn():
            return None
        try:
            return locate_inn_view(self.inn.device.image)
        except (OSError, ValueError, cv2.error):
            return None

    def _click(self, area, name):
        from module.base.button import Button
        self.inn.device.click(Button(area=(), color=(), button=area, name=name))

    def _position(self, table):
        """把桌子移到屏幕安全区域。每段拖动后等惯性停下，再重新定位。"""
        for attempt in range(self.MAX_DRAGS+1):
            if not self._capture():
                return False
            location = self._scene()
            if location is None:
                return False  # 定位丢失时不猜方向、不按旧坐标点击。
            x, y = map_to_screen(location, table)
            if 240 <= x <= 1010 and 250 <= y <= 510:
                return True
            if attempt == self.MAX_DRAGS:
                break
            dx, dy = int(np.clip(640-x, -320, 320)), int(np.clip(400-y, -180, 180))
            self.inn.jiangnan_drag((640, 370), (640+dx, 370+dy), name='INN_DINER_TABLE_POSITION')
            self.inn.device.sleep(0.7)
        return False

    def _inspect(self, table):
        """两帧都没有气泡才视为空桌；两帧目标一致才允许打开需求。"""
        first = None
        for frame in range(2):
            if not self._capture():
                return False, None
            location = self._scene()
            if location is None:
                return False, None
            bubble = find_table_bubble(self.inn.device.image, location, table)
            if frame:
                stable = ((first is None and bubble is None) or
                          (first is not None and bubble is not None and
                           max(abs(a-b) for a, b in zip(first, bubble)) <= 6))
                return stable, bubble
            first = bubble
            self.inn.device.sleep(0.25)

    def _serve(self, whitelist):
        # 等待需求窗口出现。有界轮询，点击气泡后绝不连续盲点同一位置。
        panel = None
        for _ in range(8):
            self.inn.device.sleep(0.3)
            if not self._capture():
                return None
            panel = demand_panel(self.inn.device.image)
            if panel is not None:
                break
        if panel is None:
            return None
        first = read_panel_demand(self.inn.device.image, panel)
        self.inn.device.sleep(0.3)
        if not self._capture():
            return None
        fresh = demand_panel(self.inn.device.image)
        if fresh is None:
            return None
        second = read_panel_demand(self.inn.device.image, fresh)
        # 名称、需求、库存均一致且满足白名单，才能给予。OCR 失败或不一致
        # 同样不满足售卖条件，只在已确认的需求窗口上点“赶走”。
        action = 'give' if first == second and should_give(second, whitelist) else 'expel'
        self._click(fresh[action], 'INN_DINER_'+action.upper())
        # 操作是否完成以弹窗消失且客栈主界面重新出现为准。若未确认，不再
        # 自动补发给予/赶走，保留现场供调试，防止把下一位食客当作上一位。
        stable = 0
        for _ in range(10):
            self.inn.device.sleep(0.3)
            if not self._capture():
                return None
            image = self.inn.device.image
            returned = demand_panel(image) is None and self.inn.is_in_inn() and not self.inn.is_inn_exit_confirm()
            stable = stable+1 if returned else 0
            if stable >= 2:
                return action
        return None

    def run(self, now):
        # 双重关闭保护：调度器会过滤未启用功能，直接调用本入口也不会操作设备。
        # 本次仅预备逻辑和配置，默认 false；不要通过烹饪开关迁移启用食客。
        if self._get('Enable', False) is not True:
            return InnFeatureResult('pending', '[食客未启用][预备逻辑]')
        next_run = now+timedelta(minutes=DINERS_INTERVAL)
        whitelist = sale_whitelist(self._get('SaleWhitelist', ''))
        given = expelled = 0
        for table in TABLES:
            if not self._position(table):
                return InnFeatureResult('retry', '[食客桌子定位未确认][等待复查]', next_run)
            stable, bubble = self._inspect(table)
            if not stable:
                return InnFeatureResult('retry', '[食客气泡画面不稳定][等待复查]', next_run)
            if bubble is None:
                continue
            self._click(bubble, 'INN_DINER_BUBBLE')
            result = self._serve(whitelist)
            if result is None:
                self.inn.save_abnormal_screenshot('inn_diner_unconfirmed')
                return InnFeatureResult('retry', '[食客操作结果未确认][保留现场]', next_run, keep_open=True)
            given += result == 'give'
            expelled += result == 'expel'
        # 每轮每桌至多处理一次，避免刚刷新来的新食客让本轮变成无限循环。
        return InnFeatureResult('done', '[食客检查完成][给予%d，赶走%d]' % (given, expelled), next_run)
