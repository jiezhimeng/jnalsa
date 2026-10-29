"""Conservative, scored recognition of the small warehouse special counts."""
import math

import cv2
import numpy as np


def padded(image, scale=3):
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image, 15, 15, 24, 24, cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))


def cleaned(image):
    """Keep baseline-aligned numeral strokes, excluding short icon fragments."""
    ink = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 120)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    digits = np.zeros_like(ink)
    height, width = ink.shape
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (area >= 12 and h >= 9 and 3 <= w <= 40
                and y + h >= height - 4 and x > 2 and x + w < width - 2):
            digits[labels == index] = 255
    return padded(cv2.cvtColor(cv2.bitwise_not(digits), cv2.COLOR_GRAY2RGB))


def confirmed_number(readings):
    """Two strong pure-number reads must agree; conflicts remain unknown."""
    valid = []
    for text, score in readings:
        text = str(text).strip()
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if (math.isfinite(score) and score >= 0.80
                and text.isascii() and text.isdigit()):
            valid.append(int(text))
    return valid[0] if len(valid) >= 2 and len(set(valid)) == 1 else None


def read_special_digits(image, slots, predict):
    """Read (name, left, divider) slots; return counts and diagnostic readings.

    Only the unresolved cells receive cleaned-crop inference. A blank or
    conflicting cell never becomes zero through an unconfirmed shape guess.
    """
    slots = list(slots)
    counts, diagnostics, views = {}, {}, {}
    if image is None:
        return counts, diagnostics
    for name, left, divider in slots:
        if (left < 0 or divider < 24 or left + 70 > image.shape[1]
                or divider > image.shape[0]):
            diagnostics[name] = {'error': 'crop_out_of_bounds'}
            continue
        views[name] = [padded(image[divider-gap:divider-5, left:left+70])
                       for gap in (21, 24)]
    names = [name for name, _, _ in slots if name in views]
    if not names:
        return counts, diagnostics
    primary = list(predict([im for name in names for im in views[name]]))
    if len(primary) != len(names) * 2:
        raise ValueError('Incomplete warehouse digit OCR response')
    for index, name in enumerate(names):
        readings = primary[index*2:index*2+2]
        diagnostics[name] = {'primary': readings}
        count = confirmed_number(readings)
        if count is not None:
            counts[name] = count
    missing = [(name, left, divider) for name, left, divider in slots
               if name in views and name not in counts]
    if missing:
        fallback = list(predict([cleaned(image[divider-gap:divider-5, left:left+70])
                                 for _, left, divider in missing for gap in (22, 23)]))
        if len(fallback) != len(missing) * 2:
            raise ValueError('Incomplete warehouse fallback OCR response')
        for index, (name, _, _) in enumerate(missing):
            readings = fallback[index*2:index*2+2]
            diagnostics[name]['fallback'] = readings
            count = confirmed_number(diagnostics[name]['primary'] + readings)
            if count is not None:
                counts[name] = count
    return counts, diagnostics
