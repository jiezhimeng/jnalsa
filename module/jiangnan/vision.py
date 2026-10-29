from collections import OrderedDict

import cv2
import numpy as np


_TEMPLATE_FEATURE_CACHE = OrderedDict()
_TEMPLATE_FEATURE_CACHE_LIMIT = 128


def crop_map_icon_foreground(
    template,
    background_threshold=22,
    padding=2,
    min_component_area=3,
):
    """裁掉地图建筑模板四周的地面，只保留建筑/地基图标主体。

    江南地图的地面颜色会随城市、地块和滤镜变化。地图建筑模板若把
    大片地面一起参与匹配，很容易继续命中旧地块，或在另一座城市失配。
    这里用模板边框的中位色估计地面，再按前景连通域求紧致包围框。

    页面按钮、文字和货物图标不应调用此函数；它专用于地图建筑图标。
    """
    if template is None or template.size == 0:
        return template
    if template.ndim != 3 or min(template.shape[:2]) < 8:
        return template

    border = np.concatenate((
        template[0], template[-1], template[:, 0], template[:, -1],
    ))
    background = np.median(border, axis=0).astype(np.int16)
    difference = np.max(
        np.abs(template.astype(np.int16) - background), axis=2
    )
    mask = np.uint8(difference >= int(background_threshold)) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    foreground = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_component_area):
            foreground[labels == index] = 255

    points = cv2.findNonZero(foreground)
    if points is None:
        return template
    left, top, width, height = cv2.boundingRect(points)
    image_height, image_width = template.shape[:2]
    left = max(0, left - int(padding))
    top = max(0, top - int(padding))
    right = min(image_width, left + width + int(padding) * 2)
    bottom = min(image_height, top + height + int(padding) * 2)
    if right - left < 8 or bottom - top < 8:
        return template
    return template[top:bottom, left:right].copy()


def _gray(image):
    """Return a contiguous uint8 gray image for Jiangnan template matching."""
    if image is None:
        return None
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(gray, dtype=np.uint8)


def _template_features(template):
    """Cache gray/Canny features for immutable template arrays."""
    if template is None:
        return None, None
    key = (id(template), tuple(template.shape), str(template.dtype))
    cached = _TEMPLATE_FEATURE_CACHE.get(key)
    if cached is not None and cached[0] is template:
        _TEMPLATE_FEATURE_CACHE.move_to_end(key)
        return cached[1], cached[2]

    gray = _gray(template)
    edge = cv2.Canny(gray, 40, 120) if gray is not None else None
    _TEMPLATE_FEATURE_CACHE[key] = (template, gray, edge)
    _TEMPLATE_FEATURE_CACHE.move_to_end(key)
    while len(_TEMPLATE_FEATURE_CACHE) > _TEMPLATE_FEATURE_CACHE_LIMIT:
        _TEMPLATE_FEATURE_CACHE.popitem(last=False)
    return gray, edge


def stable_feature_map(search, template, gray_weight=0.35):
    """Build a tint-resistant score map from local tone and line structure.

    Jiangnan overlays often tint the same icon with different paper/background
    colours.  Gray matching keeps the internal dark/light layout, while Canny
    matching emphasizes the icon's characteristic strokes and border.  Their
    weighted sum is substantially less sensitive to those global colour shifts
    than raw RGB matching.
    """
    search_gray = _gray(search)
    template_gray, template_edge = _template_features(template)
    if search_gray is None or template_gray is None:
        raise ValueError('search and template images are required')
    if (
        search_gray.shape[0] < template_gray.shape[0]
        or search_gray.shape[1] < template_gray.shape[1]
    ):
        raise ValueError('template is larger than search image')

    gray_map = cv2.matchTemplate(
        search_gray, template_gray, cv2.TM_CCOEFF_NORMED
    )
    search_edge = cv2.Canny(search_gray, 40, 120)
    edge_map = cv2.matchTemplate(
        search_edge, template_edge, cv2.TM_CCOEFF_NORMED
    )
    weight = min(1.0, max(0.0, float(gray_weight)))
    stable_map = gray_map * weight + edge_map * (1.0 - weight)
    return stable_map, gray_map, edge_map


def stable_feature_match(search, template, gray_weight=0.35):
    """Return best stable, gray and edge scores plus the best location."""
    stable_map, gray_map, edge_map = stable_feature_map(
        search, template, gray_weight=gray_weight
    )
    _, score, _, location = cv2.minMaxLoc(stable_map)
    x, y = location
    return (
        float(score),
        location,
        float(gray_map[y, x]),
        float(edge_map[y, x]),
    )
