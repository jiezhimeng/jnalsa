import time

import cv2

from module.base.button import Button
from module.base.utils import crop, load_image
from module.jiangnan.assets import (
    DISCARD_PROMPT_CHECK,
    HARVEST_CANCEL_BUTTON,
    HARVEST_DISCARD_BUTTON,
)
from module.jiangnan.building import JiangnanBuilding
from module.jiangnan.vision import stable_feature_match
from module.logger import logger


class JiangnanHarvest(JiangnanBuilding):
    """搜索指定建筑并收取其“完成”产物。"""

    COMPLETE_REFERENCE = './assets/jiangnan/harvest/complete.png'
    # 只截取旗帜下半段“完成”二字。上半段是货物/建筑图标，不同码头
    # 货物会把整旗模板分数拉低到 0.59~0.70，导致明明完成却不点击。
    COMPLETE_TEMPLATE_AREA = (742, 273, 784, 352)
    # “前往目的地”后的镜头并不保证把码头和完成牌同时放在中央；
    # 部分城市的完成牌会贴近屏幕上沿。覆盖整块城市可操作区域，避开
    # 左侧等级栏和最右任务栏即可，不能再从 y=100 才开始搜索。
    COMPLETE_SEARCH_AREA = (75, 20, 1180, 680)
    FULL_REFERENCE = './assets/jiangnan/harvest/full.png'
    FULL_TEMPLATE_AREA = (724, 283, 781, 427)
    # 强制收获提示跟随被点击的完成牌出现，也可能靠近画面边缘。
    FULL_SEARCH_AREA = COMPLETE_SEARCH_AREA

    def __init__(self, config, device=None, task=None):
        super().__init__(config=config, device=device, task=task)
        self.last_discarded = False
        self.last_harvest_blocked_reason = None
        self.last_harvest_count = 0
        self._last_full_score_log = 0.0
        self._harvest_click_serial = 0
        self._complete_template = load_image(
            self.COMPLETE_REFERENCE,
            self.COMPLETE_TEMPLATE_AREA,
        )
        # A selected full-storage banner changes 完成 from white to red.
        # It remains actionable while its normal/video buttons are expanded.
        self._selected_complete_template = load_image(
            './assets/jiangnan/harvest/complete_selected.png'
        )
        self._full_template = load_image(
            self.FULL_REFERENCE,
            self.FULL_TEMPLATE_AREA,
        )
        # The red tool button in the warehouse-full reference is the normal
        # force-harvest action. Some buildings show it alone, without a video
        # button, so a required left/right pair misses the real action.
        self._force_action_template = load_image(
            self.FULL_REFERENCE, (736, 449, 805, 518)
        )
        # Some completed goods use a white scroll glyph instead of the tool.
        # Only its visible glyph is matched: neighboring flags can cover the
        # left half, and residents can join the two red button contours.
        self._force_action_variants = (
            load_image('./assets/jiangnan/harvest/normal_scroll_action.png'),
        )
        # The right-hand warehouse-full action can also be "看广告".  Only
        # the captured "丢弃" label authorizes discarding without leaving
        # the game; button color and position alone cannot distinguish them.
        self._discard_label = load_image(
            './assets/jiangnan/harvest/discard.png', (735, 490, 810, 526)
        )

    def find_complete(self, similarity=0.61):
        """在整个城市可操作区域匹配竖排“完成”旗帜。"""
        x1, y1, _, _ = self.COMPLETE_SEARCH_AREA
        search = crop(self.device.image, self.COMPLETE_SEARCH_AREA, copy=False)
        score, location, gray_score, edge_score = stable_feature_match(
            search, self._complete_template, gray_weight=0.30
        )
        logger.attr(
            'HarvestCompleteScore',
            f'stable={score:.3f}, gray={gray_score:.3f}, '
            f'edge={edge_score:.3f}',
        )
        template = self._complete_template
        if score < similarity:
            selected = getattr(self, '_selected_complete_template', None)
            if selected is None:
                return None
            selected_score, selected_location, _, _ = stable_feature_match(
                search, selected, gray_weight=0.30
            )
            if selected_score < 0.78:
                return None
            logger.attr('HarvestSelectedCompleteScore', f'{selected_score:.3f}')
            template, location = selected, selected_location
        height, width = template.shape[:2]
        left = x1 + location[0]
        top = y1 + location[1]
        # 模板已经只有“完成”文字，直接点击匹配文字内部。
        return Button(
            area=(),
            color=(),
            button=(
                left + max(3, width // 10),
                top + 3,
                left + width - max(3, width // 10),
                top + height - 4,
            ),
            name=(f'HARVEST_COMPLETE_DYNAMIC_'
                  f'{left + width // 2}_{top + height // 2}'),
        )

    def find_complete_action(self, complete):
        """Find the non-video action revealed after tapping a 完成 banner."""
        x1, y1, x2, y2 = complete.button
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        height, width = self.device.image.shape[:2]
        left = max(75, center_x - 160)
        top = max(20, center_y + 55)
        right = min(width - 20, center_x + 160)
        # Some tall buildings place the actions ~200px below the text.
        # Include their entire 70px circles, not just the upper halves;
        # clipped contours fail the circle-shape check and repeat the banner.
        bottom = min(height - 75, center_y + 260)
        if right <= left or bottom <= top:
            return None
        hsv = cv2.cvtColor(
            self.device.image[top:bottom, left:right], cv2.COLOR_RGB2HSV
        )
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 60), (15, 255, 255)),
            cv2.inRange(hsv, (165, 90, 60), (179, 255, 255)),
        )
        contours, _ = cv2.findContours(
            red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circles = []
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            # The video glyph cuts a large hole in its red circle.  On real
            # screenshots its contour area is about 1680 while the normal
            # action beside it is about 3500; both still form the same pair.
            if (48 <= bw <= 85 and 48 <= bh <= 85
                    and 0.8 <= bw / bh <= 1.2 and area >= 1400):
                circles.append((left + bx + bw // 2, top + by + bh // 2))
        for first in circles:
            for second in circles:
                if (55 <= second[0] - first[0] <= 125
                        and abs(second[1] - first[1]) <= 12
                        and abs((first[0] + second[0]) // 2 - center_x) <= 40):
                    logger.info(
                        f'Completed banner revealed normal/video action pair: '
                        f'normal={first}, video={second}'
                    )
                    return Button(
                        area=(), color=(),
                        button=(first[0] - 10, first[1] - 10,
                                first[0] + 10, first[1] + 10),
                        name='HARVEST_NORMAL_ACTION_DYNAMIC',
                    )
        # After the short warehouse-full banner fades, the normal red tool
        # icon remains below 完成. Match its glyph, not just a red circle: the
        # adjacent ad action can have exactly the same red background.
        return self._find_force_action(
            (center_x - 130, center_y + 45,
             center_x + 105, center_y + 250),
            'HARVEST_NORMAL_ACTION_DYNAMIC',
        )

    def _find_force_action(self, area, name):
        image = self.device.image
        height, width = image.shape[:2]
        x1, y1, x2, y2 = area
        x1, y1 = max(75, x1), max(20, y1)
        x2, y2 = min(width - 20, x2), min(height - 20, y2)
        patch = image[y1:y2, x1:x2]
        best = None
        templates = (self._force_action_template,) + tuple(
            getattr(self, '_force_action_variants', ())
        )
        for index, template in enumerate(templates):
            if (patch.shape[0] < template.shape[0]
                    or patch.shape[1] < template.shape[1]):
                continue
            score, location, _, _ = stable_feature_match(
                patch, template, gray_weight=0.35
            )
            # The partially visible scroll needs a stronger match than the
            # larger original tool template; a red circle alone is invalid.
            if score < (0.66 if index == 0 else 0.82):
                continue
            if best is None or score > best[0]:
                best = score, location, template
        if best is None:
            return None
        score, location, template = best
        logger.attr('ForceHarvestIconScore', f'{score:.3f}')
        cx = x1 + location[0] + template.shape[1] // 2
        cy = y1 + location[1] + template.shape[0] // 2
        return Button(
            area=(), color=(),
            button=(cx - 8, cy - 8, cx + 8, cy + 8),
            name=name,
        )

    def find_warehouse_full(self, similarity=0.60):
        """动态识别仓满状态，并定位实际强制收获按钮。"""
        self._warehouse_full_banner_visible = False
        x1, y1, _, _ = self.FULL_SEARCH_AREA
        search = crop(self.device.image, self.FULL_SEARCH_AREA, copy=False)
        score, location, gray_score, edge_score = stable_feature_match(
            search, self._full_template, gray_weight=0.35
        )
        now = time.time()
        if score >= similarity or now - self._last_full_score_log >= 2.0:
            logger.attr(
                'WarehouseFullScore',
                f'stable={score:.3f}, gray={gray_score:.3f}, '
                f'edge={edge_score:.3f}',
            )
            self._last_full_score_log = now
        if score >= similarity:
            left = x1 + location[0]
            top = y1 + location[1]
        else:
            # A banner at the upper edge loses its top 30-50 pixels.  Match
            # the distinctive lower text directly; the full template cannot
            # fit there.  The high cutoff avoids confusing a 完成 banner with
            # the red 仓库已满 banner.
            clipped = self._full_template[50:]
            image = self.device.image
            top_strip = image[:170, 75:min(image.shape[1] - 100, 1180)]
            if (clipped.size == 0 or top_strip.shape[0] < clipped.shape[0]
                    or top_strip.shape[1] < clipped.shape[1]):
                return None
            source_gray = cv2.cvtColor(top_strip, cv2.COLOR_RGB2GRAY)
            clipped_gray = cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY)
            result = cv2.matchTemplate(
                source_gray, clipped_gray, cv2.TM_CCOEFF_NORMED
            )
            _, clipped_score, _, clipped_location = cv2.minMaxLoc(result)
            if clipped_score < 0.78:
                return None
            left = 75 + clipped_location[0]
            top = clipped_location[1] - 50
            logger.attr('WarehouseFullClippedScore', f'{clipped_score:.3f}')
            score = clipped_score
        # Stable edge matching alone can accept a gold 完成 banner. The actual
        # 仓库已满 banner is mostly red; require that color before acting.
        banner = self.device.image[
            max(0, top + 10):max(0, top + 130),
            max(0, left + 5):max(0, left + 52),
        ]
        if banner.size == 0:
            return None
        banner_hsv = cv2.cvtColor(banner, cv2.COLOR_RGB2HSV)
        banner_red = cv2.bitwise_or(
            cv2.inRange(banner_hsv, (0, 90, 60), (15, 255, 255)),
            cv2.inRange(banner_hsv, (165, 90, 60), (179, 255, 255)),
        )
        red_ratio = cv2.countNonZero(banner_red) / banner_red.size
        if red_ratio < 0.35:
            return None
        self._warehouse_full_banner_visible = True
        # The captured banner has only this one normal tool action. It is
        # anchored to the banner but has its own distinctive white glyph.
        action = self._find_force_action(
            (left - 20, top + 115, left + 205, top + 325),
            'FORCE_HARVEST_DYNAMIC',
        )
        if action is not None:
            logger.info(
                f'Warehouse full found: score={score:.3f}, '
                f'red={red_ratio:.3f}, force={action.button}'
            )
            return action
        image_h, image_w = self.device.image.shape[:2]
        # 强制收获按钮不是固定在仓满旗下面，而是跟随触发仓满的建筑完成牌。
        # 因此不要使用仓满旗坐标推算按钮位置，改为在当前可操作区域寻找
        # 红色棒状按钮，再结合仓满提示位置过滤候选。
        # 强制收获是独立红色棒状图标，不与仓满旗保持固定坐标关系。
        # 全屏寻找红色操作图标，再根据形状筛选，避免因为镜头移动导致误点。
        circles = []
        # 左侧圆钮可以落在 y≈630；右侧红色导航栏从 y≈575 开始，
        # 所以左右分别扫描，避免右侧圆钮与底栏连成一个大轮廓。
        regions = (
            (75, 20, min(825, image_w - 20), image_h - 15),
            (min(825, image_w - 20), 20, image_w - 20, image_h - 145),
        )
        for wx1, wy1, wx2, wy2 in regions:
            patch = self.device.image[wy1:wy2, wx1:wx2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, 90, 60), (15, 255, 255)),
                cv2.inRange(hsv, (165, 90, 60), (179, 255, 255)),
            )
            contours, _ = cv2.findContours(
                red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                bx, by, width, height = cv2.boundingRect(contour)
                area = float(cv2.contourArea(contour))
                ratio = width / max(height, 1)
                if not (55 <= width <= 130 and 25 <= height <= 90
                        and 0.8 <= ratio <= 4.0 and area >= 600):
                    continue
                cx = wx1 + bx + width / 2
                cy = wy1 + by + height / 2
                distance = abs(cx - (left + 40)) * 0.5 + abs(cy - (top + 160))
                if distance <= 240:
                    circles.append((distance, cx, cy, width, height))
        if not circles:
            logger.info('Warehouse-full action pair not found in current frame')
            return None
        # 实机截图里左侧普通收获钮与仓满竖旗/桥面连成一个大轮廓，
        # 上面的红色连通域只能找到右侧视频钮。沿仓满旗附近用圆边缘
        # 同时找两枚钮；必须确认左右成对，绝不把孤立的视频钮当作
        # 强制收获钮点击。
        pairs = []
        for distance, video_x, video_y, _, _ in circles:
            # The banner may be clipped at the top, and the action buttons
            # follow the building rather than the banner. Search around each
            # red candidate instead of assuming a fixed banner offset.
            hx1 = max(75, int(video_x - 145))
            hy1 = max(20, int(video_y - 60))
            hx2 = min(image_w - 20, int(video_x + 55))
            hy2 = min(image_h - 5, int(video_y + 60))
            gray = cv2.cvtColor(
                self.device.image[hy1:hy2, hx1:hx2], cv2.COLOR_RGB2GRAY
            )
            gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
            found = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55,
                param1=100, param2=24, minRadius=25, maxRadius=42,
            )
            outlines = [] if found is None else [
                (hx1 + float(x), hy1 + float(y)) for x, y, _ in found[0]
            ]
            if not any(
                abs(x - video_x) <= 16 and abs(y - video_y) <= 16
                for x, y in outlines
            ):
                continue
            for normal_x, normal_y in outlines:
                spacing = video_x - normal_x
                # The normal action starts just to the right of the full
                # banner. A circular decoration to its left can otherwise
                # form a false pair with the normal action itself.
                normal_offset = normal_x - left
                video_offset = video_x - left
                if (55 <= spacing <= 125
                        and abs(normal_y - video_y) <= 16
                        # In the 21:25 capture the ordinary action is almost
                        # exactly below the red banner's left edge.  Keep
                        # both circular outlines and their spacing as the
                        # safety check; do not require a rightward offset.
                        and -20 <= normal_offset <= 100
                        and 75 <= video_offset <= 190):
                    pairs.append((
                        distance + abs(spacing - 90), normal_x, normal_y,
                        video_x, video_y,
                    ))
        if not pairs:
            logger.warning(
                'Warehouse-full normal/video action pair is not confirmed; '
                'skip unsafe force harvest click'
            )
            return None
        _, normal_x, normal_y, video_x, video_y = min(pairs)
        click_x = int(round(normal_x))
        # 圆钮贴住底边时，其几何中心可能已经落到截图外；点击仍可见的
        # 上半部，而不是产生 y=720/721 这类无效 ADB 坐标。
        click_y = int(round(min(normal_y, image_h - 7)))
        click_x = max(7, min(image_w - 7, click_x))
        click_y = max(7, min(image_h - 7, click_y))
        button = Button(
            area=(), color=(),
            button=(click_x - 6, click_y - 6, click_x + 6, click_y + 6),
            name='FORCE_HARVEST_DYNAMIC',
        )
        logger.info(
            f'Warehouse full found: score={score:.3f}, '
            f'center=({left + 28}, {top + 72}), '
            f'force=({click_x}, {click_y}), '
            f'video=({video_x:.0f}, {video_y:.0f})'
        )
        return button

    def is_warehouse_full(self):
        return self.find_warehouse_full() is not None

    def find_discard_prompt_buttons(self):
        """定位仓满确认框的左右按钮；右侧可能是“丢弃”或“看广告”。"""
        x1, y1, x2, y2 = (300, 420, 980, 590)
        patch = self.device.image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 70), (15, 255, 255)),
            cv2.inRange(hsv, (165, 90, 70), (179, 255, 255)),
        )
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if 150 <= width <= 260 and 35 <= height <= 90:
                candidates.append((x1 + left, y1 + top, width, height))
        pairs = []
        for index, first in enumerate(candidates):
            for second in candidates[index + 1:]:
                a, b = sorted((first, second), key=lambda item: item[0])
                ac = (a[0] + a[2] / 2, a[1] + a[3] / 2)
                bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
                if 170 <= bc[0] - ac[0] <= 360 and abs(ac[1] - bc[1]) <= 28:
                    score = abs((ac[0] + bc[0]) / 2 - 640) + abs(ac[1] - bc[1]) * 3
                    pairs.append((score, a, b))
        if not pairs:
            return None
        _, left_button, right_button = min(pairs)
        def make_button(item, name):
            left, top, width, height = item
            return Button(area=(), color=(), button=(left + 8, top + 5, left + width - 8, top + height - 5), name=name)
        return make_button(left_button, 'HARVEST_CANCEL_DYNAMIC'), make_button(right_button, 'HARVEST_DISCARD_DYNAMIC')

    def is_discard_prompt(self):
        return self.appear(DISCARD_PROMPT_CHECK, offset=(28, 28), similarity=0.76) or self.find_discard_prompt_buttons() is not None

    def _is_discard_action(self, button):
        """Require the captured 丢弃 label before clicking the right action."""
        x1, y1, x2, y2 = button.button
        image = self.device.image
        height, width = image.shape[:2]
        x1, y1 = max(0, x1 - 8), max(0, y1 - 8)
        x2, y2 = min(width, x2 + 8), min(height, y2 + 8)
        patch = image[y1:y2, x1:x2]
        reference = self._discard_label
        if (patch.size == 0 or patch.shape[0] < reference.shape[0]
                or patch.shape[1] < reference.shape[1]):
            return False
        # Match bright glyphs rather than the red button background; a
        # second red button labelled 看广告 is visually identical otherwise.
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        patch_text = cv2.threshold(patch_gray, 170, 255, cv2.THRESH_BINARY)[1]
        reference_text = cv2.threshold(reference_gray, 170, 255, cv2.THRESH_BINARY)[1]
        score = float(cv2.matchTemplate(
            patch_text, reference_text, cv2.TM_CCOEFF_NORMED
        ).max())
        logger.attr('HarvestDiscardLabelScore', f'{score:.3f}')
        return score >= 0.65

    def _remember_harvest_complete(self, button):
        x1, y1, x2, y2 = button.button
        self._harvest_complete_reference = (
            (x1, y1, x2, y2), self.device.image[y1:y2, x1:x2].copy(),
        )

    def _match_harvest_complete(self):
        area, reference = self._harvest_complete_reference
        x1, y1, x2, y2 = area
        left, top = max(0, x1 - 25), max(0, y1 - 25)
        search = self.device.image[top:y2 + 25, left:x2 + 25]
        if reference.size == 0 or search.shape[0] < reference.shape[0] or search.shape[1] < reference.shape[1]:
            return None, None
        _, score, _, position = cv2.minMaxLoc(cv2.matchTemplate(
            cv2.cvtColor(search, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), cv2.TM_CCOEFF_NORMED,
        ))
        x, y = left + position[0], top + position[1]
        return score, Button(area=(), color=(),
                             button=(x, y, x + reference.shape[1], y + reference.shape[0]),
                             name='HARVEST_REOPEN_CONFIRMED_COMPLETE')

    def resolve_warehouse_full(self, force=True, timeout=6, force_button=None):
        """处理仓满；不点击广告，右侧不是“丢弃”时取消本次收获。"""
        if force_button is None:
            # Direct production-panel calls may follow another building's
            # cleanup. Only an immediate, supplied action can reuse its flag.
            self._harvest_complete_reference = None
        self.last_harvest_blocked_reason = None
        self.last_discarded = False
        prompt_visible = (
            False if force_button is not None else self.is_discard_prompt()
        )
        if prompt_visible:
            force_button = None
        elif force_button is None:
            force_button = self.find_warehouse_full()
        if force_button is not None:
            for attempt in range(2):
                x1, y1, x2, y2 = force_button.button
                if (x1 + x2) // 2 < 825 or (y1 + y2) // 2 < 520:
                    break
                # The task/build toolbar intercepts clicks well above its
                # visible top edge. Move the map and detect the live icon again.
                logger.warning('Force harvest action covered by bottom navigation')
                self._harvest_complete_reference = None
                self.device.swipe(
                    (650, 470), (650, 310), duration=0.25,
                    name='FORCE_HARVEST_UNCOVER_ACTION',
                )
                self.device.sleep(0.4)
                self.device.screenshot()
                force_button = self.find_warehouse_full()
                if force_button is None:
                    break
            if force_button is None or (
                ((force_button.button[0] + force_button.button[2]) // 2 >= 825)
                and ((force_button.button[1] + force_button.button[3]) // 2 >= 520)
            ):
                logger.warning('Force harvest action remains covered after repositioning')
                self.save_abnormal_screenshot('force_harvest_action_covered')
                return False
        if force_button is None and not prompt_visible:
            # 强制收获按钮可能被基础 UI 或固定界面遮挡。
            # 无法安全点击时跳过本次收获，不能误报为已处理仓满。
            logger.info('Force harvest button unavailable; skip this warehouse-full harvest')
            self.last_harvest_blocked_reason = 'warehouse_full_action_unconfirmed'
            return False
        logger.warning('Warehouse is full')
        if force_button is not None:
            # The banner/action pair has already been confirmed in this frame.
            # Re-sampling can consume the entire lifetime of this temporary
            # button; tap it before taking another screenshot.
            self.device.click(force_button)
        click_count = int(force_button is not None)
        last_click = time.time()
        reopened_complete = False
        target_absent_since = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.device.screenshot()
            prompt_buttons = self.find_discard_prompt_buttons()
            prompt_template = self.appear(
                DISCARD_PROMPT_CHECK, offset=(28, 28), similarity=0.76
            )
            if prompt_buttons is not None or prompt_template:
                if prompt_buttons is not None:
                    # The confirm dialog slides upwards after it first becomes
                    # detectable. A coordinate from that frame can end up well
                    # below the button by the time the device receives a click.
                    self.device.sleep(0.3)
                    self.device.screenshot()
                    settled_buttons = self.find_discard_prompt_buttons()
                    if settled_buttons is None:
                        logger.info('Warehouse-full modal buttons disappeared while settling')
                        continue
                    previous_right = getattr(prompt_buttons[1], 'button', None)
                    current_right = getattr(settled_buttons[1], 'button', None)
                    if previous_right is not None and current_right is not None:
                        previous_center = (
                            (previous_right[0] + previous_right[2]) // 2,
                            (previous_right[1] + previous_right[3]) // 2,
                        )
                        current_center = (
                            (current_right[0] + current_right[2]) // 2,
                            (current_right[1] + current_right[3]) // 2,
                        )
                        if max(abs(a - b) for a, b in zip(previous_center, current_center)) > 8:
                            logger.info(
                                f'Warehouse-full modal still moving: '
                                f'{previous_center} -> {current_center}'
                            )
                            continue
                    prompt_buttons = settled_buttons
                if prompt_buttons is not None:
                    cancel_button, right_button = prompt_buttons
                    logger.info('Use live warehouse-full modal buttons')
                else:
                    cancel_button = self.anchored_button(
                        DISCARD_PROMPT_CHECK, HARVEST_CANCEL_BUTTON,
                        name='HARVEST_CANCEL_DYNAMIC', inset=8,
                    )
                    right_button = self.anchored_button(
                        DISCARD_PROMPT_CHECK, HARVEST_DISCARD_BUTTON,
                        name='HARVEST_RIGHT_ACTION_DYNAMIC', inset=8,
                    )
                discard_available = force and self._is_discard_action(right_button)
                if force and not discard_available:
                    logger.info(
                        'Warehouse full: right action is not confirmed as 丢弃; '
                        'skip harvest because ads are disabled'
                    )
                    self.last_harvest_blocked_reason = 'warehouse_full_no_discard'
                target = right_button if discard_available else cancel_button
                logger.info('Discard overflow materials' if discard_available else 'Cancel harvest because warehouse is full')
                self.device.click(target)
                if not discard_available:
                    self.device.sleep(0.8)
                    return False
                # A successful click is not proof that the final discard was
                # accepted. Confirm the modal has closed on the main map.
                absent_frames = 0
                confirm_deadline = time.time() + timeout
                while time.time() < confirm_deadline:
                    self.device.sleep(0.25)
                    self.device.screenshot()
                    still_open = self.is_discard_prompt()
                    absent_frames = absent_frames + 1 if not still_open else 0
                    if absent_frames >= 2 and self.is_in_main():
                        self.last_discarded = True
                        logger.info('Discard confirmed: modal closed on main map')
                        return True
                logger.warning('Discard action did not return to main map')
                self.last_harvest_blocked_reason = 'discard_unconfirmed'
                self.save_abnormal_screenshot('discard_unconfirmed')
                return False

            current_full = self.find_warehouse_full()
            if (current_full is None and click_count > 0
                    and time.time() - last_click >= 1.2
                    and getattr(self, '_harvest_complete_reference', None) is not None
                    and self.is_in_main()):
                score, complete = self._match_harvest_complete()
                if score is not None and score < 0.45:
                    if target_absent_since is None:
                        target_absent_since = time.time()
                    elif time.time() - target_absent_since >= 1.5:
                        self.last_discarded = bool(force)
                        logger.info('Warehouse-full target collected without a confirmation modal')
                        return True
                else:
                    target_absent_since = None
                if score is not None and score >= 0.75 and not reopened_complete:
                    self.device.click(complete)
                    reopened_complete = True
                    click_count = 0
                    last_click = time.time()
                    deadline = max(deadline, last_click + 3)
                    logger.info('Reopen unchanged completed target after lost warehouse-full action')
                    self.device.sleep(0.25)
                    continue

            # 实机偶尔吞掉强制收获点击。只有仓满旗仍明确存在时才重试，
            # 避免在确认弹窗动画期间盲点；每次都重新定位按钮当前位置。
            # 重新展开“完成”后 click_count 为零，新出现的短暂操作钮
            # 必须立即点击，不能沿用展开旗帜那次点击的 1.2 秒等待。
            if click_count < 3 and (click_count == 0 or time.time() - last_click >= 1.2):
                retry_button = self.find_warehouse_full()
                if retry_button is not None:
                    x1, y1, x2, y2 = retry_button.button
                    if (x1 + x2) // 2 >= 825 and (y1 + y2) // 2 >= 520:
                        logger.warning(
                            'Retry force harvest action covered by bottom navigation'
                        )
                        self.save_abnormal_screenshot('force_harvest_action_covered')
                        return False
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry force harvest button: {click_count}/3'
                    )
                    self.device.click(retry_button)
            self.device.sleep(0.25)
        logger.warning('Discard confirmation prompt timeout')
        self.save_abnormal_screenshot('discard_prompt_timeout')
        return False

    def harvest_centered_building(self, force=True):
        """收获当前居中的建筑；仓满时按 force 决定丢弃或取消。"""
        self.last_discarded = False
        self.device.screenshot()
        complete = self.find_complete()
        if complete is None:
            logger.info('Centered building has no completed production')
            return True

        logger.info('Click completed production')
        if hasattr(self, '_remember_harvest_complete'):
            self._remember_harvest_complete(complete)
        self.device.click(complete)

        # 完成点击后立即寻找棒状强制收获按钮。
        # 不等待普通清场循环，避免完成旗状态覆盖强制收获流程。
        for _ in range(10):
            self.device.sleep(0.1)
            self.device.screenshot()
            force_button = self.find_warehouse_full()
            if force_button is not None:
                logger.info('Immediate force harvest button found after complete click')
                # A second 完成 tap can close this temporary action. The
                # current frame already confirms its left-hand icon.
                if not self.resolve_warehouse_full(
                    force=force, force_button=force_button,
                ):
                    return False
                break

        click_count = 1
        last_click = time.time()
        absent_since = None
        deadline = time.time() + 8
        while time.time() < deadline:
            # 仓满旗帜大约只显示一秒，取消浮字 OCR 后直接快速
            # 采样该旗帜，不等待随机位置的增加数量动画。
            self.device.sleep(0.15)
            self.device.screenshot()
            force_button = self.find_warehouse_full()
            if force_button is not None:
                if not self.resolve_warehouse_full(
                    force=force, force_button=force_button,
                ):
                    return False
                # 强制收获只是处理仓满阻塞，不代表当前收获流程结束。
                # 仓满确认后重新扫描完成标记，继续处理剩余收获。
                absent_since = None
                last_click = time.time()
                logger.info('Force harvest handled, continue scanning completed nodes')
                continue

            current = self.find_complete()
            if current is None:
                # 点击后的缩放/闪烁动画会让模板短暂低于阈值。只有“完成”连续
                # 1.5 秒都没有恢复，才可以认定资源确实已经被收取。
                if absent_since is None:
                    absent_since = time.time()
                if self.is_in_main() and time.time() - absent_since >= 1.5:
                    logger.info('Building harvest completed')
                    return True
            else:
                absent_since = None
                # 标记仍在说明上次点击没有生效（常见于动画或随机点到边缘）。
                # 限制重试次数和间隔，避免快速连点。
                if click_count < 3 and time.time() - last_click >= 2:
                    click_count += 1
                    last_click = time.time()
                    logger.info(f'Retry completed production: {click_count}/3')
                    self.device.click(current)

        logger.warning('Building harvest result timeout')
        self.save_abnormal_screenshot('building_harvest_timeout')
        return False

    def harvest_all_visible(
        self,
        force=True,
        limit=32,
        require_one=False,
        initial_wait=4.0,
        require_main_for_empty=True,
    ):
        """强制收完当前城市画面中所有可见的“完成”节点。

        每次点击后重新截图和定位，不复用旧坐标。因此多个完成牌距离
        很远、动画导致短暂消失或其中一个触发仓满提示时，仍会继续
        处理剩余节点。
        """
        discarded_any = False
        click_count = 0
        self.last_harvest_count = 0
        self.last_harvest_blocked_reason = None
        empty_since = None
        last_complete_center = None
        same_complete_count = 0
        last_action_center = None
        navigation_overlap_pans = 0
        left_overlay_pans = 0
        for index in range(max(1, int(limit))):
            self.device.screenshot()
            if getattr(self, 'is_network_timeout_popup', lambda: False)():
                logger.warning('Network popup interrupted visible harvest')
                self.handle_network_timeout_popup()
                empty_since = None
                continue
            complete = self.find_complete()
            if complete is None:
                # 收获动画会让旗帜短暂消失。连续确认一小段时间，避免第
                # 二条完成节点刚好在动画结束后出现而被误判为全部清空。
                if empty_since is None:
                    empty_since = time.time()
                # A search already waited for centering before this scan.
                # Honour its shorter empty check; keep the full animation
                # grace period after any actual harvest.
                wait_time = 1.2 if click_count else max(float(initial_wait), 0.4)
                page = self.current_navigation_page()
                if page in ('build', 'building_search', 'warehouse'):
                    logger.warning(
                        f'Harvest left city map for {page}; stop clearing'
                    )
                    self.save_abnormal_screenshot('harvest_left_map')
                    self.last_harvest_count = click_count
                    self.last_discarded = discarded_any
                    return False
                main_ready = not require_main_for_empty or page == 'main'
                if main_ready and time.time() - empty_since >= wait_time:
                    self.last_harvest_count = click_count
                    if require_one and click_count == 0:
                        logger.warning(
                            'Arrived transport has no visible completed node; '
                            'do not mark it collected'
                        )
                        self.last_discarded = discarded_any
                        return False
                    logger.info(
                        f'All completed production harvested: clicks={click_count}'
                    )
                    self.last_discarded = discarded_any
                    return True
                self.device.sleep(0.2)
                continue

            empty_since = None
            x1, y1, x2, y2 = complete.button
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            action = self.find_complete_action(complete)
            if (action is None and 75 <= center[0] <= 365
                    and 200 <= center[1] <= 500
                    and left_overlay_pans < 2):
                # 建筑搜索留下的左侧展开菜单可能盖住“完成”旗帜下方
                # 的普通收获按钮。仅将地图移开遮挡区，再从新画面识别，
                # 不按旧位置盲点或把被挡住的按钮当成不存在。
                left_overlay_pans += 1
                logger.info('Move left-edge harvest action away from overlay')
                self.device.swipe(
                    (640, 400), (820, 400), duration=0.25,
                    name='HARVEST_UNCOVER_LEFT_ACTION',
                )
                self.device.sleep(0.4)
                last_complete_center = None
                last_action_center = None
                continue
            if action is not None:
                ax1, ay1, ax2, ay2 = action.button
                action_x = (ax1 + ax2) // 2
                action_y = (ay1 + ay2) // 2
                if action_x >= 825 and action_y >= 570:
                    # Bottom navigation intercepts this action: an actual
                    # 造纸厂 run clicked 营造 128 times. Pan the map upward,
                    # then locate the banner and action again from new pixels.
                    if navigation_overlap_pans >= 2:
                        logger.warning(
                            'Harvest action remains covered by bottom navigation'
                        )
                        self.save_abnormal_screenshot('harvest_action_covered')
                        self.last_harvest_count = click_count
                        self.last_discarded = discarded_any
                        return False
                    navigation_overlap_pans += 1
                    self.device.swipe(
                        (650, 470), (650, 310), duration=0.25,
                        name='HARVEST_UNCOVER_ACTION',
                    )
                    self.device.sleep(0.4)
                    last_complete_center = None
                    last_action_center = None
                    continue
                if (last_action_center is not None
                        and abs(last_action_center[0] - center[0]) <= 20
                        and abs(last_action_center[1] - center[1]) <= 20):
                    logger.warning(
                        f'Completed normal action unchanged after click: '
                        f'center={center}'
                    )
                    self.save_abnormal_screenshot('harvest_action_unchanged')
                    self.last_harvest_count = click_count
                    self.last_discarded = discarded_any
                    return False
                self.device.click(action)
                click_count += 1
                last_action_center = center
                same_complete_count = 0
                last_complete_center = None
                self.device.sleep(0.5)
                self.device.screenshot()
                if self.is_discard_prompt():
                    if not self.resolve_warehouse_full(force=force):
                        self.last_harvest_count = click_count
                        self.last_discarded = discarded_any
                        return False
                    discarded_any = discarded_any or self.last_discarded
                continue
            if (
                last_complete_center is not None
                and abs(center[0] - last_complete_center[0]) <= 20
                and abs(center[1] - last_complete_center[1]) <= 20
            ):
                same_complete_count += 1
            else:
                same_complete_count = 1
                last_complete_center = center
            if same_complete_count >= 3:
                logger.warning(
                    f'Completed marker unchanged after repeated clicks: center={center}'
                )
                # 同一位置连续三次仍是“完成”时，盲目重复点击只会把整轮
                # 清场拖到 32 次上限。保留现场，交给下一轮重试。
                self.save_abnormal_screenshot('harvest_marker_unchanged')
                self.last_harvest_count = click_count
                self.last_discarded = discarded_any
                return False

            logger.info(f'Harvest completed node {index + 1}/{limit}')
            target = complete
            self._harvest_click_serial += 1
            # Device 的防连点按按钮名统计。批量清场会在短时间内合法点击
            # 很多不同的完成牌，不能都共用 HARVEST_COMPLETE_DYNAMIC。
            target.name = f'HARVEST_COMPLETE_{self._harvest_click_serial}'
            if hasattr(self, '_remember_harvest_complete'):
                self._remember_harvest_complete(complete)
            self.device.click(target)
            click_count += 1

            # 仓满提示出现时间很短，点击后快速检查。若没有仓满提示，
            # 外层循环重新识别当前位置；既能点击下一条，也能重试一次
            # 未生效的点击，始终不复用旧坐标。
            check_deadline = time.time() + 1.2
            while time.time() < check_deadline:
                self.device.sleep(0.15)
                self.device.screenshot()
                force_button = self.find_warehouse_full()
                if force_button is not None:
                    if not self.resolve_warehouse_full(
                        force=force, force_button=force_button,
                    ):
                        if self.last_harvest_blocked_reason == 'warehouse_full_no_discard':
                            logger.info('Skip remaining visible harvest: warehouse full, ads disabled')
                        else:
                            logger.warning('Warehouse full unresolved; stop visible harvest')
                        self.last_harvest_count = click_count
                        self.last_discarded = discarded_any
                        return False
                    discarded_any = discarded_any or self.last_discarded
                    # 仓满弹窗会让完成旗产生视觉残留，清除旧位置缓存，等待下一轮真实截图识别。
                    same_complete_count = 0
                    last_complete_center = None
                    self.device.sleep(0.8)
                    break

        self.device.screenshot()
        if self.find_complete() is not None:
            logger.warning(
                f'Completed production remains after {limit} harvests'
            )
            self.save_abnormal_screenshot('harvest_limit_reached')
            self.last_harvest_count = click_count
            self.last_discarded = discarded_any
            return False
        self.last_harvest_count = click_count
        self.last_discarded = discarded_any
        return not require_one or click_count > 0

    def harvest_all_centered_building(self, force=True, limit=32):
        """兼容旧调用；识别范围现在是当前画面的全部可操作区域。"""
        return self.harvest_all_visible(force=force, limit=limit)

    def harvest_building(self, name, force=True):
        logger.hr(f'JIANGNAN HARVEST: {name}, force={force}', level=0)
        if not self.search_building(name):
            return False
        return self.harvest_centered_building(force=force)

    def harvest_all_building(
        self,
        name,
        force=True,
        limit=32,
        exact=False,
    ):
        logger.hr(
            f'JIANGNAN HARVEST ALL: {name}, force={force}',
            level=0,
        )
        if not self.search_building(name, exact=exact):
            return False
        return self.harvest_all_visible(
            force=force,
            limit=limit,
        )
