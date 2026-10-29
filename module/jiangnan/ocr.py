"""OCR classes used exclusively by Jiangnan automation.

PP-OCRv6 is given the original RGB crop.  The threshold/letter arguments are
kept for source compatibility with the old ALAS OCR classes, but Jiangnan no
longer turns text into a binary mask before recognition.
"""

import time

import cv2

from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
from module.base.utils import crop, float2str
from module.logger import logger
from module.ocr.ocr import Digit as LegacyDigit
from module.ocr.ocr import Ocr as LegacyOcr


def read_confirmed_digits(image, area):
    """Require the same full number from original and padded, enlarged crops."""
    original = crop(image, area)
    if original is None or original.size == 0:
        return None
    # Padding keeps edge digits away from the recognition crop boundary.
    enlarged = cv2.resize(original, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.copyMakeBorder(enlarged, 10, 10, 18, 18, cv2.BORDER_REPLICATE)
    values = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus(
        [original, enlarged], cand_alphabet='0123456789',
    )
    valid = [int(value) for value in values if str(value).isdigit()]
    if len(valid) == 2 and valid[0] == valid[1]:
        return valid[0]
    logger.warning(f'Digit confirmation disagreed: {values}')
    return None


class _DirectPPOcrMixin:
    """Run recognition once on untouched crops, bypassing legacy preprocessing."""

    def ocr(self, image, direct_ocr=False):
        start_time = time.time()
        if direct_ocr:
            image_list = list(image)
        else:
            image_list = [crop(image, area) for area in self.buttons]

        result_list = self.cnocr.atomic_ocr_for_single_lines(
            image_list,
            self.alphabet,
        )
        result_list = [self.after_process(str(result)) for result in result_list]

        if len(self.buttons) == 1:
            result_list = result_list[0]
        if self.SHOW_LOG:
            logger.attr(
                name='%s %ss' % (
                    self.name,
                    float2str(time.time() - start_time),
                ),
                text=str(result_list),
            )
        return result_list


class Ocr(_DirectPPOcrMixin, LegacyOcr):
    """Direct PP-OCRv6 medium text recognition."""

    @property
    def cnocr(self):
        return PPOCR_V6_CASCADE


class Digit(_DirectPPOcrMixin, LegacyDigit):
    """Direct PP-OCRv6 medium digit recognition."""

    @property
    def cnocr(self):
        return PPOCR_V6_CASCADE

    def after_process(self, result):
        """Keep an unreadable crop distinct from a real on-screen zero."""
        result = str(result).replace('I', '1').replace('D', '0')
        result = result.replace('S', '5').replace('B', '8')
        return int(result) if result else -1
