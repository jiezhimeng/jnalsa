"""Read city seals and resource cards on the offline tax receipt popup."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
from module.jiangnan.resource import (
    WAREHOUSE_OCR_ALIASES, WAREHOUSE_RESOURCES, WAREHOUSE_SPECIAL_RESOURCES,
    SHAOXING_MERCHANT_RESOURCES,
)
from module.logger import logger


TAX_CURRENCIES = {'铜钱', '玉米'}

TAX_RESOURCES = set(
    WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES + list(SHAOXING_MERCHANT_RESOURCES)
) | TAX_CURRENCIES


@lru_cache(maxsize=1)
def _card_borders():
    path = Path(__file__).resolve().parents[2] / 'assets/jiangnan/city/tax_popup.png'
    source = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    return gray[125:141, 334:459], gray[324:340, 334:459]


def tax_card_areas(image):
    """Locate complete scroll cards, requiring both top and bottom borders."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    top, bottom = _card_borders()
    left, upper, right, lower = 326, 112, 1100, 550
    region = gray[upper:lower, left:right]
    scores = cv2.matchTemplate(region, top, cv2.TM_CCOEFF_NORMED)
    peaks = scores == cv2.dilate(scores, np.ones((9, 9), np.uint8))
    rows, columns = np.where(peaks & (scores >= 0.85))
    cards = []
    for row, column in sorted(zip(rows, columns)):
        x, y = left + int(column), upper + int(row)
        if y + 215 > lower or x + 125 > right:
            continue
        if any(abs(x - old_x) < 25 and abs(y - old_y) < 25 for old_x, old_y in cards):
            continue
        tail = gray[y + 196:y + 218, x - 3:x + 128]
        if (tail.shape[0] < bottom.shape[0]
                or tail.shape[1] < bottom.shape[1]
                or cv2.matchTemplate(tail, bottom, cv2.TM_CCOEFF_NORMED).max() < 0.85):
            continue
        cards.append((x, y))
    return [
        ((x + 4, y + 17, x + 121, y + 55),
         (x + 4, y + 162, x + 121, y + 199))
        for x, y in cards
    ]


def _text(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value)))


def read_tax_resources(image):
    areas = tax_card_areas(image)
    if not areas:
        return None
    crops = []
    for name_area, amount_area in areas:
        for x1, y1, x2, y2 in (name_area, amount_area):
            original = image[y1:y2, x1:x2]
            crops.extend((original, cv2.resize(original, None, fx=3, fy=3)))
    values = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(crops)
    if len(values) != 4 * len(areas):
        return None
    resources = {}
    for index in range(0, len(values), 4):
        names = [WAREHOUSE_OCR_ALIASES.get(_text(v), _text(v))
                 for v in values[index:index + 2]]
        names = ['铜钱' if v == '銅錢' else v for v in names]
        amounts = [_text(v).replace(',', '').replace('，', '')
                   for v in values[index + 2:index + 4]]
        if (names[0] not in TAX_RESOURCES or names[0] != names[1]
                or not all(value.isdigit() for value in amounts)
                or int(amounts[0]) != int(amounts[1])):
            logger.warning(f'Tax resource card uncertain: {values[index:index + 4]}')
            return None
        resources[names[0]] = resources.get(names[0], 0) + int(amounts[0])
    logger.attr('TaxPopupResources', resources)
    return resources


def read_tax_city(image):
    # Lazy import: popup.py is itself a base of JiangnanCity.
    from module.jiangnan.city import CITY_SEAL_CHARACTER_AREAS, CITY_SEAL_TO_CITY
    originals = [image[y1:y2, x1:x2]
                 for x1, y1, x2, y2 in CITY_SEAL_CHARACTER_AREAS]
    values = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(originals)
    city = CITY_SEAL_TO_CITY.get(''.join(_text(value) for value in values))
    if city:
        logger.attr('TaxPopupCity', city)
        return city
    crops = []
    for x1, y1, x2, y2 in CITY_SEAL_CHARACTER_AREAS:
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        ink = cv2.inRange(gray, 155, 255)
        crops.append(cv2.resize(255 - ink, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST))
    values = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(crops)
    city = CITY_SEAL_TO_CITY.get(''.join(_text(value) for value in values))
    logger.attr('TaxPopupCity', city)
    return city
