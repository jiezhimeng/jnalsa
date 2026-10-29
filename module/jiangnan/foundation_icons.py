"""Match foundation-mode icon families independently of ground colour."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ICON_FACILITIES = {
    'civic': '码头 税课司 议事堂 漕运码头 朝暮鼓 客栈',
    'gather': ('农田 林场 鱼塘 棉花田 黏土矿 草药田 桐油种植园 漆树林 '
               '菱角池塘 养蚕场 养鸭场 花簪制坊 竹林 茶园 蟹塘 煮盐场 '
               '油菜花田 养猪场 酒曲田 羊圈 苎麻田 改良农田'),
    'food': '餐馆 鱼干坊 特产铺',
    'craft': ('锯木厂 造纸厂 织布坊 裁缝铺 瓷窑 烧炭厂 炼丹炉 油纸厂 '
              '制伞坊 云锦织坊 漆器作坊 螺钿漆坊 丝织坊 绣坊 铁矿场 '
              '冶铁工坊 兵器工坊 铜矿 折扇作坊 团扇作坊 标布作坊 制鞋坊 '
              '徽墨作坊 澄纸作坊 歙砚作坊 玉器工坊 越瓷窑 乌毡帽厂 '
              '织网工坊 琥珀酒坊 珍馐食肆 琳琅商铺 汪笔作坊 采石场 玉米晒簟'),
    'furniture': '家具铺',
    'powder': '香粉厂 印刷厂',
    'fishing': '渔船 雕刻行',
    'merchant': '商船',
}
FACILITY_ICON = {
    facility: family
    for family, facilities in ICON_FACILITIES.items()
    for facility in facilities.split()
}
ICON_DIR = Path(__file__).resolve().parents[2] / 'assets/jiangnan/foundation_icons'
ICON_VARIANTS = {'civic': ('civic.png', 'civic_canal.png')}


def _strokes(rgb):
    """Keep locally dark line work and discard slowly changing ground colour."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local_background = cv2.GaussianBlur(gray, (0, 0), 7)
    return cv2.threshold(
        cv2.subtract(local_background, gray), 22, 255, cv2.THRESH_BINARY,
    )[1]


@lru_cache(maxsize=8)
def _templates(family):
    templates = []
    for filename in ICON_VARIANTS.get(family, (f'{family}.png',)):
        image = cv2.imread(str(ICON_DIR / filename), cv2.IMREAD_COLOR)
        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            templates.append(_strokes(rgb))
            # 实机商船地基图标比旧截图模板大约大 5%；保留小范围
            # 尺寸变体，避免降低阈值后把相邻人物或水纹当成目标。
            if family == 'merchant':
                for scale in (1.05, 1.10):
                    templates.append(_strokes(cv2.resize(
                        rgb, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )))
    return tuple(templates)


def find_foundation_icon(image, arrow, facility, threshold=0.72, max_horizontal=75):
    """Find the topmost matching icon directly below the search arrow."""
    family = FACILITY_ICON.get(facility)
    if family is None or image is None or arrow is None:
        return None
    references = _templates(family)
    if not references:
        return None
    axis, tip, _ = arrow
    height, width = image.shape[:2]
    horizontal_limit = 75 if max_horizontal is None else max_horizontal
    # Crop before extracting strokes or matching templates. Include half an
    # icon and a blur margin so icons at the column boundary stay complete.
    half_width = max(reference.shape[1] for reference in references) // 2
    half_height = max(reference.shape[0] for reference in references) // 2
    radius = int(horizontal_limit + half_width + 16)
    left, right = max(0, axis - radius), min(width, axis + radius)
    top, bottom = max(0, tip + 12), min(height - 20, tip + 280 + half_height + 16)
    if right <= left or bottom <= top:
        return None
    strokes = _strokes(image[top:bottom, left:right])
    candidates = []
    for reference in references:
        if (strokes.shape[1] < reference.shape[1]
                or strokes.shape[0] < reference.shape[0]):
            continue
        scores = cv2.matchTemplate(strokes, reference, cv2.TM_CCOEFF_NORMED)
        peaks = scores == cv2.dilate(scores, np.ones((15, 15), np.uint8))
        rows, cols = np.where(peaks & (scores >= threshold))
        for row, col in zip(rows, cols):
            x = left + int(col) + reference.shape[1] // 2
            y = top + int(row) + reference.shape[0] // 2
            horizontal = abs(x - axis)
            # Production tiles share icon families. If the selected icon is
            # hidden, a good match on a neighbouring tile is still wrong.
            if horizontal > horizontal_limit:
                continue
            if 12 <= y - tip <= 280:
                score = float(scores[row, col])
                # The first icon below the arrow is the selected building.
                # Lower icons must not win by distance or template score.
                candidates.append((y, horizontal, -score, x, score))
    if not candidates:
        return None
    y, _, _, x, score = min(candidates)
    return x, y, score, family
