"""Identify only the currency beside the drum button, never account balances."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


CURRENCY_AREA = (970, 576, 1008, 610)
COIN_AMOUNT_AREA = (999, 578, 1060, 609)


@lru_cache(maxsize=1)
def _references():
    root = Path(__file__).resolve().parents[2] / 'assets/jiangnan/chaomu_drum'
    return {
        name: cv2.cvtColor(cv2.imdecode(
            np.fromfile(str(root / (name + '_cost.png')), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        ), cv2.COLOR_BGR2RGB)
        for name in ('gem', 'coin')
    }


def drum_cost_currency(image, area=CURRENCY_AREA):
    if image is None:
        return None
    x1, y1, x2, y2 = area
    patch = image[y1:y2, x1:x2]
    for name, template in _references().items():
        if (patch.shape[0] >= template.shape[0]
                and patch.shape[1] >= template.shape[1]
                and cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED).max() >= 0.86):
            return name
    return None
