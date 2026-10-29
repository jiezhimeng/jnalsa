import time
import re
import os
import cv2
import numpy as np

from module.jiangnan.app import JiangnanApp
from module.jiangnan.city_metadata import CITY_KEY_TO_PRODUCTION_TASK
from module.jiangnan.assets import (
    CITY_MAP_BUTTON,
    CITY_OVERVIEW_BUTTON,
    CITY_LIST_CHECK,
    CITY_TYPE_ALL_BUTTON,
    CITY_LIST_SLOTS,
    CITY_PANEL_CHECK,
    FARM_IDLE_CHECK,
    FARM_WORKING_CHECK,
    PRODUCTION_IDLE_CHECK,
    PRODUCTION_WORKING_CHECK,
    WAREHOUSE_TITLE_CHECK,
    BUILD_TITLE_CHECK,
    GRAND_CANAL_PAGE_CHECK,
    GRAND_CANAL_SINAN_TASK_CHECK,
    GRAND_CANAL_COMPASS_CHECK,
    GRAND_CANAL_SINAN_REWARD,
    GRAND_CANAL_SINAN_OVERFLOW_CHECK,
    TRANSPORT_ARRIVAL_PROMPT_CHECK,
    TRANSPORT_GO_LABEL_CHECK,
    TRANSPORT_GO_BUTTON,
)
from module.jiangnan.ocr import Ocr
from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
from module.jiangnan.resource_layout import (
    PAGE_CITY,
    ResourceLayoutManager,
)
from module.logger import logger

from difflib import SequenceMatcher
from module.base.button import Button
from module.base.utils import crop


# ============================================================
# Jiangnan known city / town names
# ============================================================

KNOWN_CITIES = [
    '应天府',
    '苏州府',
    '杭州府',
    '松江府',
    '徽州府',
    '扬州府',
    '绍兴府',
    '宁波府',

    '鸡鸣山',

    '江宁县',
    '白雪镇',
    '棕影岛',
    '龙宫',
]

# 府印与城市的明确对应关系由用户提供；只接受四字完整匹配。
CITY_SEAL_TO_CITY = {
    '应天府印': '应天府',
    '苏州府印': '苏州府',
    '杭州府印': '杭州府',
    '松江府印': '松江府',
    '鸡鸣山印': '鸡鸣山',
    '徽州府印': '徽州府',
    '扬州府印': '扬州府',
    '绍兴府印': '绍兴府',
    '宁波府印': '宁波府',
    '江宁县印': '江宁县',
    '白雪镇印': '白雪镇',
    '棕影岛印': '棕影岛',
    '龙宫之印': '龙宫',
}

CITY_SEAL_CHARACTER_AREAS = (
    (15, 496, 69, 551), (73, 496, 128, 551),
    (15, 553, 69, 609), (73, 553, 128, 609),
)

# 城市列表固定字体在当前 PP-OCR 模型下会稳定地产生这些结果。
# 这里只收录日志中反复出现且指向唯一城市的误识别，避免放宽通用
# 相似度后把其他城市猜错。
CITY_OCR_ALIASES = {
    '山县': '江宁县',
    '山芸': '江宁县',
    '山门县': '江宁县',
    '山玄': '鸡鸣山',
    '日当琪': '白雪镇',
    '给六府': '绍兴府',
    '标影奇': '棕影岛',
}

CITY_TEMPLATE_DIR = './assets/jiangnan/city/names'

class JiangnanCityNameOcr(Ocr):
    """
    江南百景图城市名称 OCR。

    城市名称是浅色卡片上的深色汉字，
    所以这里将其转换成黑底白字后交给 cnocr。
    """

    def __init__(self):
        super().__init__(
            buttons=CITY_LIST_SLOTS,
            lang='cnocr',
            name='CITY_NAME_OCR'
        )

    def pre_process(self, image):
        """
        城市名称 OCR 预处理。

        1. 灰度化
        2. 二值化
        3. 放大 3 倍，提高中文识别准确率
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, binary = cv2.threshold(
            gray,
            180,
            255,
            cv2.THRESH_BINARY_INV
        )

        binary = cv2.resize(
            binary,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        return binary


class JiangnanCity(JiangnanApp):

    def __init__(self, config, device=None, task=None):

        super().__init__(
            config=config,
            device=device,
            task=task
        )

        self.city_ocr = JiangnanCityNameOcr()
        self.city_layout = ResourceLayoutManager(config)
        self._city_list_reset_seq = 0
        self._city_list_search_seq = 0
        self._city_template_cache = None
        self._city_template_scale_cache = {}
        self.city_seal_ocr = Ocr(
            [Button(area=area, color=(), button=area,
                    name=f'CITY_SEAL_CHARACTER_{index}')
             for index, area in enumerate(CITY_SEAL_CHARACTER_AREAS)],
            lang='cnocr', name='CITY_SEAL_CHARACTERS',
        )

    # ========================================================
    # Page recognition
    # ========================================================

    def find_city_map_button(self):
        """
        在城市菜单中搜索“大地图”按钮。

        按钮数量变化时，大地图可能左右移动，
        所以使用较大的横向 offset。
        """

        result = self.appear(
            CITY_MAP_BUTTON,
            offset=(80, 12),
            similarity=0.85
        )

        self.log_check(
            'CityMapButtonCheck',
            result
        )

        return result

    def is_city_panel(self):
        result = self.appear(
            CITY_PANEL_CHECK,
            offset=(5, 5),
            similarity=0.85
        )

        self.log_check(
            'CityPanelCheck',
            result
        )

        return result

    def is_world_map(self):
        """
        是否在大地图。
        """

        result = self.appear(
            CITY_OVERVIEW_BUTTON,
            offset=(5, 5),
            similarity=0.90
        )

        self.log_check(
            'WorldMapCheck',
            result
        )

        return result

    def is_city_list(self):
        """
        是否在“城镇与运输”列表。
        """

        result = self.appear(
            CITY_LIST_CHECK,
            offset=(3, 3),
            similarity=0.90
        )

        self.log_check(
            'CityListCheck',
            result
        )

        return result

    def _is_canal_overlay(self):
        return any(self.appear(button, offset=(5, 5), similarity=0.80)
                   for button in (GRAND_CANAL_SINAN_TASK_CHECK,
                                  GRAND_CANAL_COMPASS_CHECK,
                                  GRAND_CANAL_SINAN_REWARD,
                                  GRAND_CANAL_SINAN_OVERFLOW_CHECK))

    def current_navigation_page(self):
        """识别当前导航页面；调用者负责先更新截图和处理弹窗。"""
        if getattr(self, 'is_network_timeout_popup', lambda: False)():
            page = 'network_popup'
        elif getattr(self, 'post_station_page', lambda: 'unknown')() != 'unknown':
            page = 'post_station_' + self.post_station_page()
        elif getattr(self, 'is_inn_exit_confirm', lambda: False)():
            page = 'inn_exit_confirm'
        elif getattr(self, 'is_in_inn', lambda: False)():
            page = 'inn'
        elif getattr(self, '_is_canal_overlay', lambda: False)():
            page = 'grand_canal_overlay'
        elif self.is_city_list():
            page = 'city_list'
        elif self.is_world_map():
            page = 'world_map'
        elif self.is_city_panel():
            page = 'city_panel'
        elif self.appear(
            GRAND_CANAL_PAGE_CHECK, offset=(6, 6), similarity=0.78,
        ):
            page = 'grand_canal'
        elif self.is_in_main():
            # 生产弹窗等页面后方仍能看到主导航，不能因此认为主界面
            # 已可操作。只在主导航命中时检查这些遮挡它的面板。
            panel_checks = (
                ('production_working', PRODUCTION_WORKING_CHECK),
                ('farm_working', FARM_WORKING_CHECK),
                ('production_idle', PRODUCTION_IDLE_CHECK),
                ('farm_idle', FARM_IDLE_CHECK),
                ('warehouse', WAREHOUSE_TITLE_CHECK),
                ('build', BUILD_TITLE_CHECK),
            )
            page = next((name for name, button in panel_checks
                         if self.appear(button, offset=(4, 4), similarity=0.88)),
                        'main')
        else:
            page = 'unknown'
        self.log_check('JiangnanNavigationPage', page)
        return page

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def normalize_city_name(name):
        """
        去掉 OCR 中可能出现的空格、符号等。
        只保留中文字符。
        """

        if not name:
            return ''

        return ''.join(
            re.findall(
                r'[\u4e00-\u9fff]+',
                str(name)
            )
        )
    @staticmethod
    def _is_subsequence(short, long):
        """
        判断 short 是否是 long 的字符子序列。

        例如：
            鸡山 -> 鸡鸣山
            True
        """

        if not short or not long:
            return False

        it = iter(long)

        return all(
            char in it
            for char in short
        )


    def correct_city_name(self, name):
        """
        将 OCR 结果修正为合法地点名称。

        修正规则：

        1. 完全匹配
        2. OCR 漏字，但剩余字符仍按顺序存在
        3. 最后才使用受限制的相似度匹配

        无法可靠判断时返回原 OCR 结果，
        不强行猜测。
        """

        name = self.normalize_city_name(name)

        if not name:
            return ''

        # ========================================
        # 1. 完全正确
        # ========================================

        if name in KNOWN_CITIES:
            return name

        if name in CITY_OCR_ALIASES:
            corrected = CITY_OCR_ALIASES[name]
            logger.info(
                f'City OCR corrected by known alias: '
                f'{name} -> {corrected}'
            )
            return corrected

        # ========================================
        # 2. 处理 OCR 漏字
        #
        # 鸡山
        # ↓
        # 鸡鸣山
        # ========================================

        subsequence_matches = [
            city
            for city in KNOWN_CITIES
            if self._is_subsequence(
                name,
                city
            )
        ]

        # 只有唯一候选才接受
        if len(subsequence_matches) == 1:

            corrected = subsequence_matches[0]

            logger.info(
                f'City OCR corrected by subsequence: '
                f'{name} -> {corrected}'
            )

            return corrected

        # ========================================
        # 3. 同长度单字误识别
        #
        # 城市卡片字体下，宁波府常被识别成“丁波府”，绍兴府会变成
        # “绍六府”。它们与合法名称长度相同且只错一个字；只有唯一
        # 候选时可以安全修正，不必依赖偏高的通用相似度阈值。
        # ========================================

        one_char_matches = [
            city
            for city in KNOWN_CITIES
            if len(city) == len(name)
            and sum(left != right for left, right in zip(name, city)) == 1
        ]
        if len(one_char_matches) == 1:
            corrected = one_char_matches[0]
            logger.info(
                f'City OCR corrected by one-character mismatch: '
                f'{name} -> {corrected}'
            )
            return corrected

        # ========================================
        # 4. 相似度纠错
        # ========================================

        scores = []

        for city in KNOWN_CITIES:

            score = SequenceMatcher(
                None,
                name,
                city
            ).ratio()

            scores.append(
                (score, city)
            )

        scores.sort(
            reverse=True
        )

        best_score, best_city = scores[0]

        second_score = (
            scores[1][0]
            if len(scores) > 1
            else 0
        )

        # 必须：
        #
        # 最佳相似度 >= 0.72
        # 且明显优于第二候选 >= 0.15
        #
        # 否则宁可不纠正
        if (
            best_score >= 0.72
            and
            best_score - second_score >= 0.15
        ):

            logger.info(
                f'City OCR corrected: '
                f'{name} -> {best_city} '
                f'(score={best_score:.3f})'
            )

            return best_city

        logger.warning(
            f'Unknown city OCR result: {name}'
        )

        return name

    def read_city_names(self):
        """
        OCR 当前列表中的地点名称，
        并使用合法地点词典进行纠错。
        """

        # 一张截图批量识别八个固定名称格。旧版优先使用城市名截图模板，
        # 游戏缩放、雨天与卡片状态变化后同一文字的模板分数会明显下降，
        # 导致肉眼可见的城市仍被判为不存在。现在直接使用中文 OCR；一次
        # 调用仍会同时处理所有格位，不为每个城市重复启动识别。
        city_images = [
            crop(self.device.image, slot.area)
            for slot in CITY_LIST_SLOTS
        ]

        def is_known_city_result(value, _index):
            value = self.normalize_city_name(value)
            return value in KNOWN_CITIES or value in CITY_OCR_ALIASES

        result = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_validated(
            city_images,
            validator=is_known_city_result,
        )

        if not isinstance(result, list):
            result = [result]

        # ----------------------------------------
        # OCR 原始结果
        # ----------------------------------------

        raw_names = [
            self.normalize_city_name(name)
            for name in result
        ]

        logger.attr(
            'CityNamesRaw',
            raw_names
        )

        # ----------------------------------------
        # 合法名称纠错
        # ----------------------------------------

        names = [
            self.correct_city_name(name)
            for name in raw_names
        ]

        logger.attr(
            'CityNames',
            names
        )

        return names

    def match_city_ocr(self, target):
        """在当前城镇列表截图中一次 OCR 定位目标城市。"""
        target = self.normalize_city_name(target)
        names = self.read_city_names()
        for index, name in enumerate(names[:len(CITY_LIST_SLOTS)]):
            if name != target:
                continue
            slot = CITY_LIST_SLOTS[index]
            x1, y1, x2, y2 = slot.area
            match = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
                1.0,
            )
            logger.info(
                f'City OCR found: {target}, slot={index + 1}, '
                f'position=({match[0]}, {match[1]})'
            )
            return match
        logger.warning(f'City OCR did not find: {target}')
        return None

    def _load_city_templates(self):
        """一次加载全部固定城市名模板，后续页面复用内存图像。"""
        if self._city_template_cache is not None:
            return self._city_template_cache
        templates = {}
        for city in KNOWN_CITIES:
            filename = os.path.join(CITY_TEMPLATE_DIR, f'{city}.png')
            try:
                data = np.fromfile(filename, dtype=np.uint8)
                template = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[city] = cv2.cvtColor(
                        template, cv2.COLOR_BGR2RGB
                    )
            except Exception as exc:
                logger.warning(
                    f'Unable to cache city template: {filename}, {exc}'
                )
        self._city_template_cache = templates
        return templates

    def _read_city_names_by_template(self, similarity=0.78, margin=0.04):
        """在一张截图的固定格位中批量识别所有可见城市名。"""
        templates = self._load_city_templates()
        names = []
        scores_log = []
        for slot in CITY_LIST_SLOTS:
            x1, y1, x2, y2 = slot.area
            search = self.device.image[y1:y2, x1:x2]
            scores = []
            for city, template in templates.items():
                if (
                    search.shape[0] < template.shape[0]
                    or search.shape[1] < template.shape[1]
                ):
                    continue
                result = cv2.matchTemplate(
                    search, template, cv2.TM_CCOEFF_NORMED
                )
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append((float(score), city))
            scores.sort(reverse=True)
            best_score, best_city = scores[0] if scores else (0.0, '')
            second_score = scores[1][0] if len(scores) > 1 else 0.0
            accepted = (
                best_score >= similarity
                and best_score - second_score >= margin
            )
            names.append(best_city if accepted else '')
            scores_log.append(
                f'{best_city or "?"}:{best_score:.3f}/{second_score:.3f}'
            )
        logger.attr('CityTemplateBatch', scores_log)
        return names

    # ========================================================
    # Navigation
    # ========================================================

    def goto_city_panel(self, timeout=8):
        """
        主界面 -> 城市菜单。
        """

        logger.hr(
            'JIANGNAN CITY PANEL'
        )

        self.device.screenshot()
        page = None
        if not self.handle_startup_popup():
            page = self.current_navigation_page()
        if page == 'city_panel':
            logger.info('City panel is already open')
            return True

        # A fresh, recognized city map already proves the game is ready.
        # Avoid the slow foreground/package probe on every building search.
        if page != 'main' and not self.ensure_ready():

            logger.warning(
                'Unable to prepare main page'
            )

            return False

        # 城市面板不会遮住主界面识别锚点，因此 ensure_ready() 在面板已
        # 打开时也会成功。这里必须先判断当前状态；旧逻辑会再次点击城市
        # 印章把面板关掉，随后世界地图、地基等入口便会连锁识别失败。
        self.device.screenshot()
        if self.is_city_panel():
            logger.info('City panel is already open')
            return True

        logger.info(
            'Click current city seal'
        )

        seal = self.find_city_seal_button()
        if seal is None:
            # ensure_ready 已确认处于主界面；部分城市印章被红点、文字
            # 或边缘裁切拆成多个连通域时，使用主界面的校准入口兜底。
            logger.warning(
                'Current city seal was not recognized; use calibrated fallback'
            )
            seal = Button(
                area=(), color=(), button=(62, 570, 92, 625),
                name='CITY_SEAL_BUTTON_FALLBACK',
            )
        self.device.click(seal)
        click_count = 1
        last_click = time.time()

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            # 网络故障可以在点击府印后才弹出，遮住城市面板的识别区。
            # 此时继续重点击府印只会耗尽本轮重试。
            if self.handle_startup_popup():
                logger.info('Popup handled while opening city panel')
                deadline = time.time() + timeout
                click_count = 0
                last_click = 0
                continue

            if self.is_city_panel():

                logger.info(
                    'City panel arrived'
                )

                return True

            if click_count < 3 and time.time() - last_click >= 2:
                seal = self.find_city_seal_button()
                if seal is not None:
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry recognized city seal: {click_count}/3'
                    )
                    self.device.click(seal)

            self.device.sleep(0.4)

        logger.warning(
            'City panel timeout'
        )

        return False

    def find_city_seal_button(self):
        """按左下红色印章区域定位入口，不依赖城市名称和固定坐标。"""
        image = self.device.image[450:720, :180]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area > 900 and h > 80 and (best is None or area > best[-1]):
                best = (x, y, w, h, area)
        if best is None:
            return None
        x, y, w, h, _ = best
        center_x = x + w // 2
        center_y = y + 450 + h // 2
        return Button(
            area=(), color=(),
            # 连通域可能只覆盖印章文字或装饰边缘。点击识别结果中心的
            # 小范围，避免随机落在印章顶部红点、透明缺口或外沿。
            button=(center_x - 10, center_y - 10,
                    center_x + 10, center_y + 10),
            name='CITY_SEAL_BUTTON_DYNAMIC',
        )

    def read_current_city_from_seal(self):
        """Read the complete four-character seal and map it to a city."""
        image = self.device.image
        if image is None or image.shape[0] < 610 or image.shape[1] < 130:
            return None
        crops = []
        for x1, y1, x2, y2 in CITY_SEAL_CHARACTER_AREAS:
            patch = image[y1:y2, x1:x2]
            # Seal lettering is pale on red; present each character as dark
            # ink on white paper to the single-line OCR model.
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            ink = cv2.inRange(gray, 155, 255)
            crops.append(cv2.resize(
                255 - ink, None, fx=2, fy=2,
                interpolation=cv2.INTER_NEAREST,
            ))
        raw = self.city_seal_ocr.ocr(crops, direct_ocr=True)
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        chars = [self.normalize_city_name(value) for value in raw]
        if any(len(value) != 1 for value in chars):
            logger.attr('CitySealCharacters', f'{raw!r} -> uncertain')
            return None
        seal_text = ''.join(chars)
        city = CITY_SEAL_TO_CITY.get(seal_text)
        logger.attr('CitySealCharacters', f'{raw!r} -> {city or "uncertain"}')
        return city

    def goto_world_map(self, timeout=10, retry_route=True):
        """
        城市菜单 -> 大地图。
        """

        logger.hr(
            'JIANGNAN WORLD MAP'
        )

        self.device.screenshot()
        if not self.handle_startup_popup():
            page = self.current_navigation_page()
            if page == 'world_map':
                logger.info('Already on world map')
                return True
            if page == 'city_list':
                # Back from the overview normally reveals the map directly.
                # Verify the resulting frame before accepting this shortcut.
                self.device.press_back()
                self.device.sleep(0.4)
                self.device.screenshot()
                if self.current_navigation_page() == 'world_map':
                    logger.info('Returned directly from city list to world map')
                    return True

        if not self.goto_city_panel():
            return False

        logger.info(
            'Find world map button'
        )

        self.device.screenshot()

        if not self.find_city_map_button():
            logger.warning(
                'World map button not found'
            )
            return False

        logger.info(
            'Click world map'
        )

        # appear() 已把模板的实时偏移写入 CITY_MAP_BUTTON。把点击区域
        # 收窄到识别结果中心，避免在竖向长按钮的透明边缘随机落点。
        def click_recognized_map_button():
            x1, y1, x2, y2 = CITY_MAP_BUTTON.button
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.device.click(Button(
                area=(), color=(),
                button=(center_x - 10, center_y - 10,
                        center_x + 10, center_y + 10),
                name='CITY_MAP_BUTTON_DYNAMIC',
            ))

        click_recognized_map_button()
        click_count = 1
        last_click = time.time()

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            # 城市菜单点击期间若误入或遗留居民事件面板，先完整处理，
            # 再从主界面重新进入城市菜单和大地图。
            if self.handle_startup_popup():
                logger.info('Popup handled while opening world map; retry route')
                return self.goto_world_map(
                    timeout=timeout,
                    retry_route=retry_route,
                )

            if self.is_world_map():

                logger.info(
                    'World map arrived'
                )

                return True

            # 游戏偶尔吞掉第一次点击。只有城市面板仍在时才重新识别并
            # 点击，防止加载过程中误触大地图上的其他区域。
            if click_count < 3 and time.time() - last_click >= 2:
                if self.is_city_panel() and self.find_city_map_button():
                    click_count += 1
                    last_click = time.time()
                    logger.info(
                        f'Retry recognized world map button: {click_count}/3'
                    )
                    click_recognized_map_button()

            self.device.sleep(0.5)

        logger.warning(
            'World map timeout'
        )
        # 点击大地图后城市面板偶尔直接关闭，却没有进入地图，原来的
        # “仅当城市面板仍在”重试无法恢复。完整回主界面重走一次。
        if retry_route:
            logger.warning('Retry complete world-map route once')
            if self.ui_goto_main():
                return self.goto_world_map(
                    timeout=timeout,
                    retry_route=False,
                )
        return False

    def goto_city_list(self, timeout=10):
        """
        大地图 -> 城镇总览。
        """

        logger.hr(
            'JIANGNAN CITY LIST'
        )

        # 运输状态会连续读取多个城市/船位。已经在城镇列表时直接复用，
        # 不能先退回主界面再重新走一遍大地图。
        self.device.screenshot()
        page = 'unknown'
        if not self.handle_startup_popup():
            page = self.current_navigation_page()
        if page == 'city_list':
            logger.info('Already in city list')
            return True

        if page != 'world_map' and not self.goto_world_map():
            return False

        logger.info(
            'Click city overview'
        )

        self.device.screenshot()
        if not self.appear(
            CITY_OVERVIEW_BUTTON,
            offset=(24, 24),
            similarity=0.82,
        ):
            logger.warning('City overview button not found')
            return False

        self.device.click(
            CITY_OVERVIEW_BUTTON
        )

        deadline = time.time() + timeout

        while time.time() < deadline:

            self.device.screenshot()

            if self.is_city_list():

                logger.info(
                    'City list arrived'
                )

                # 不判断当前是“小镇”还是“城市”
                # 每次都主动切换到“全部”
                logger.info(
                    'Set city type to ALL'
                )

                self.device.click(
                    self.anchored_button(
                        CITY_LIST_CHECK,
                        CITY_TYPE_ALL_BUTTON,
                        name='CITY_TYPE_ALL_BUTTON_DYNAMIC',
                        inset=5,
                    )
                )

                self.device.sleep(0.8)

                return True

            self.device.sleep(0.5)

        logger.warning(
            'City list timeout'
        )

        return False
    def city_list_reset_top(self):
        """快速向下划四行；从列表任意位置一次必定到达顶部。"""
        logger.hr('CITY LIST RESET TOP')
        self._city_list_reset_seq += 1
        control_name = f'CITY_LIST_TO_TOP_{self._city_list_reset_seq:04d}'
        self.device.swipe(
            (570, 190),
            (570, 630),
            duration=0.1,
            name=control_name,
        )
        self.device.sleep(0.4)
        return True

    def city_list_reset_bottom(self):
        """向上拖四行；从列表任意位置一次必定到达底部。"""
        logger.hr('CITY LIST RESET BOTTOM')
        self._city_list_reset_seq += 1
        control_name = f'CITY_LIST_TO_BOTTOM_{self._city_list_reset_seq:04d}'
        self.jiangnan_drag(
            (570, 630),
            (570, 190),
            name=control_name,
        )
        self.device.sleep(0.8)
        return True

    def match_city_template(self, target, similarity=0.82):
        """
        在当前“城镇与运输”列表中搜索指定城市名称。

        不使用 OCR。
        直接使用目标城市名称模板进行匹配。

        Returns:
            (x, y, score) | None
        """

        target = self.normalize_city_name(target)

        template_file = os.path.join(CITY_TEMPLATE_DIR, f'{target}.png')
        template = self._load_city_templates().get(target)
        if template is None:

            logger.warning(
                f'City template not found: {template_file}'
            )

            return None

        # ----------------------------------------
        # 只搜索城镇列表内容区域
        #
        # 不搜索标题、类型按钮等区域
        # ----------------------------------------

        x1, y1, x2, y2 = (
            190,
            155,
            1080,
            680
        )

        search = self.device.image[
            y1:y2,
            x1:x2
        ]

        # 城镇总览中的名称是模板原始大小；“选择目的地”页会把同一
        # 名称放大约 5%～10%。同时尝试三个比例，避免把目的地页上
        # 肉眼可见的城市误判成不存在。
        best = None
        for scale in (1.0, 1.05, 1.10):
            cache_key = (target, scale)
            scaled = self._city_template_scale_cache.get(cache_key)
            if scaled is None:
                scaled = (
                    template if scale == 1.0 else cv2.resize(
                        template, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
                self._city_template_scale_cache[cache_key] = scaled

            if (
                scaled.shape[0] > search.shape[0]
                or scaled.shape[1] > search.shape[1]
            ):
                continue

            result = cv2.matchTemplate(
                search,
                scaled,
                cv2.TM_CCOEFF_NORMED
            )
            _, score, _, location = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (float(score), location, scaled, scale)
            # 原比例已经高度吻合时无需再做两次全页缩放匹配。
            if scale == 1.0 and score >= 0.90:
                break

        if best is None:
            return None

        max_score, max_loc, matched_template, matched_scale = best

        logger.attr(
            f'CityTemplate_{target}',
            f'{max_score:.3f}'
        )

        if max_score < similarity:

            return None

        h, w = matched_template.shape[:2]

        # 模板中心的屏幕绝对坐标
        center_x = (
            x1
            + max_loc[0]
            + w // 2
        )

        center_y = (
            y1
            + max_loc[1]
            + h // 2
        )

        logger.info(
            f'City template found: {target}, '
            f'score={max_score:.3f}, '
            f'scale={matched_scale:.2f}, '
            f'position=({center_x}, {center_y})'
        )

        return (
            center_x,
            center_y,
            max_score
        )
    def click_city_match(self, match):
        """
        根据城市名称模板匹配位置，
        点击对应的城市卡片。
        """

        x, y, score = match

        # 城市名称上方约 45 px 是城市卡片主体
        click_x = int(x)
        click_y = int(y - 45)

        logger.info(
            f'Click matched city ({click_x}, {click_y})'
        )

        # ALAS Device.click() 需要 Button 对象，
        # 不能直接传 (x, y) tuple
        click_button = Button(
            area=(),
            color=(),
            button=(
                click_x - 10,
                click_y - 10,
                click_x + 10,
                click_y + 10,
            ),
            name='CITY_MATCH_CLICK',
        )

        self.device.click(
            click_button
        )

    def find_city(self, target, max_swipes=10):
        """
        优先用城市名图片查找目标，失配时再识别文字。

        列表顶部显示前八座城市，底部显示后七座城市；扬州、绍兴是
        两个视图的重叠锚点。因此只检查顶部和底部，不逐页搜索。
        """
        target = self.normalize_city_name(target)
        logger.hr(f'FIND CITY: {target}')
        if not self.city_list_reset_top():
            return None

        for position in ('top', 'bottom'):
            self.device.screenshot()
            match = self.match_city_template(target)
            if match is None:
                match = self.match_city_ocr(target)
            if match is not None:
                logger.info(f'City found at list {position}: {target}')
                return match
            if position == 'top' and not self.city_list_reset_bottom():
                break

        logger.warning(f'City not found: {target}')
        return None

    @staticmethod
    def _city_match_screen_position(match):
        """把模板命中坐标换算成当前页面的行列。"""
        x, y, _ = match
        column = 0 if abs(x - 253) <= abs(x - 703) else 1
        row_centers = (263, 395, 525, 654)
        screen_row = min(
            range(len(row_centers)),
            key=lambda index: abs(y - row_centers[index]),
        )
        return screen_row, column

    def find_city_by_layout(self, target, max_moves=8):
        """按辅助信息跳到目标边界，优先图片定位，失配时 OCR 兜底。"""
        target = self.normalize_city_name(target)
        configured = self.city_layout.load(PAGE_CITY)
        expected = configured.get(target)
        if expected is None:
            return None

        expected_row, _ = expected
        max_row = max(row for row, _ in configured.values())
        bottom_first_row = max(0, max_row - 3)
        preferred = 'top' if expected_row <= 3 else 'bottom'
        if preferred == 'top':
            self.city_list_reset_top()
            first_row = 0
        else:
            self.city_list_reset_bottom()
            first_row = bottom_first_row

        # 用户给出的四行滑动量从任意位置一次就会抵达对应边界，不再因
        # 一次识别失败又反向滑去另一端。这样每次切城最多只滑动一次。
        self.device.screenshot()
        match = self.match_city_template(target)
        if match is None:
            # 边界滑动后的卡片仍可能在回弹。日志中绍兴府首帧分数 0.647，
            # 同页稳定后达到 0.901；先重截图片，避免立刻做昂贵的 OCR。
            for _ in range(2):
                self.device.sleep(0.5)
                self.device.screenshot()
                match = self.match_city_template(target)
                if match is not None:
                    break
        if match is None:
            # 缩放、雨天和卡片状态会使模板失配；只在这时识别文字。
            logger.warning(
                f'City template missed {target} at list {preferred}; '
                'try same-screen OCR'
            )
            match = self.match_city_ocr(target)
        if match is None:
            return None
        screen_row, column = self._city_match_screen_position(match)
        observed = (first_row + screen_row, column)
        if match[2] < 1.0 and observed != expected:
            # Low-confidence image matches can land on a neighboring city.
            # Confirm the name before changing the saved layout or tapping it.
            logger.warning(
                f'City template position disagrees with layout: '
                f'{target}, expected={expected}, observed={observed}; '
                'verify by OCR'
            )
            verified = self.match_city_ocr(target)
            if verified is None:
                return None
            match = verified
            screen_row, column = self._city_match_screen_position(match)
            observed = (first_row + screen_row, column)
        self.city_layout.confirm_position(
            PAGE_CITY,
            target,
            observed,
        )
        logger.info(f'City found at list {preferred}: {target}')
        return match

    def locate_city(self, target):
        """按辅助位置选择一个边界页，失败时只检查另一个边界页。"""
        target = self.normalize_city_name(target)

        # “选择目的地”等页面使用五列卡片，并不是两列的城镇与运输
        # 总览。旧逻辑仍先读取两列 8 格城市名，既得不到有效位置，又
        # 白白触发一次昂贵 OCR。非总览页面直接用目标名称模板定位；
        # 目的地页本身已经一次展示全部城市，无需滚动位置辅助。
        if not self.is_city_list():
            match = self.match_city_template(target)
            if match is not None:
                logger.info(
                    f'City found directly on non-overview page: {target}'
                )
                return match
            logger.warning(
                f'City not found on non-overview page: {target}'
            )
            return None

        match = self.find_city_by_layout(target)
        if match is not None:
            return match

        logger.warning(f'City not found at configured list boundary: {target}')
        return None

    def _retry_city_entry_from_list(self, target):
        """Retry only when the same city card remains visible in the overview."""
        if not self.is_city_list():
            return False
        match = self.match_city_template(target)
        if match is None:
            return False
        logger.info(f'Retry unresponsive city card: {target}')
        self.click_city_match(match)
        return True

    def wait_city_arrival(self, timeout=30, go_clicked_at=None,
                          city_target=None):
        """
        点击目标城市以后等待新城市加载。

        注意：
        加载阶段不允许使用 BACK。

        可能出现：
            税课司
            其他已知奖励弹窗

        统一交给 handle_startup_popup()。
        """

        logger.hr(
            'WAIT CITY ARRIVAL'
        )

        deadline = time.time() + timeout
        go_retries = 0
        last_go_retry = go_clicked_at or 0
        go_prompt_frames = 0
        city_retries = 0
        last_city_retry = time.time()
        next_foreground_check = time.time() + 5.0
        main_frames = 0

        while time.time() < deadline:

            self.device.screenshot()

            # 运输提示框的首次“前往”点击偶尔被游戏吞掉。只在标题和
            # 按钮文字同时可见时重试，避免把其他“提示”框当作运输框。
            if (self.appear(
                TRANSPORT_ARRIVAL_PROMPT_CHECK,
                offset=(3, 3), similarity=0.88,
            ) and self.appear(
                TRANSPORT_GO_LABEL_CHECK,
                offset=(4, 4), similarity=0.88,
            )):
                go_prompt_frames += 1
                main_frames = 0
                if go_retries >= 3:
                    logger.warning('Transport go prompt remained after three retries')
                    return False
                # A screenshot taken immediately after the first click still
                # shows the old frame. Require two observations and a short
                # transition window before treating the click as swallowed.
                if go_prompt_frames >= 2 and time.time() - last_go_retry >= 1.5:
                    button = self.anchored_button(
                        TRANSPORT_ARRIVAL_PROMPT_CHECK,
                        TRANSPORT_GO_BUTTON,
                        name='TRANSPORT_GO_BUTTON_RETRY', inset=12,
                    )
                    logger.info(f'Retry transport go button: {go_retries + 1}/3')
                    self.device.click(button)
                    go_retries += 1
                    last_go_retry = time.time()
                self.device.sleep(0.4)
                continue
            go_prompt_frames = 0

            # 弹窗后方可能仍满足主界面检测条件，所以先处理弹窗。
            self._tax_popup_city_hint = city_target
            popup_started = time.time()
            try:
                popup_handled = self.handle_startup_popup()
            finally:
                self._tax_popup_city_hint = None
            if popup_handled:
                main_frames = 0

                logger.info(
                    'Popup handled during '
                    'city switching'
                )

                self.device.sleep(0.5)

                # Receipt OCR and ledger persistence can take longer than the
                # loading timeout. Charge only loading time to this deadline.
                popup_elapsed = max(0.0, time.time() - popup_started)
                deadline += popup_elapsed
                last_city_retry += popup_elapsed
                next_foreground_check += popup_elapsed

                continue

            # ========================================
            # 新城市已经进入
            # ========================================

            if self.is_unobstructed_main():
                # Without the former slow package query, the first frame can
                # still be transitioning. Confirm the new city on two frames.
                main_frames += 1
                if main_frames < 2:
                    self.device.sleep(0.2)
                    continue

                logger.info(
                    'New city main page arrived'
                )

                return True
            main_frames = 0

            if (city_target is not None and city_retries < 2
                    and time.time() - last_city_retry >= 4.0
                    and self._retry_city_entry_from_list(city_target)):
                city_retries += 1
                last_city_retry = time.time()
                self.device.sleep(0.4)
                continue

            # ========================================
            # 其余状态视为加载中
            # 不按 BACK，不乱点
            # ========================================
            if time.time() >= next_foreground_check:
                if not self.app_is_running_recoverable():
                    logger.warning('Jiangnan left foreground during city switching')
                    return False
                next_foreground_check = time.time() + 10.0
            self.device.sleep(0.3)

        logger.warning(
            'City switching timeout'
        )

        self.save_abnormal_screenshot('city_switch_timeout')

        return False

    def switch_city(self, target):
        """
        切换到指定城市。
        """

        target = self.normalize_city_name(
            target
        )

        logger.hr(
            f'JIANGNAN SWITCH CITY: {target}',
            level=0
        )

        self.device.screenshot()
        if self.is_in_main():
            current = self.read_current_city_from_seal()
            if current == target:
                logger.info(f'Already in {target}; skip city navigation')
                return True

        # ========================================
        # 1. 主界面 -> 城镇列表
        # ========================================

        if not self.goto_city_list():

            logger.warning(
                'Unable to open city list'
            )

            return False

        # 先用辅助信息从列表当前停留位置确定方向和距离。只有页面位置
        # 无法校准或配置位置失效时，才回顶部执行完整兜底搜索。
        match = self.locate_city(target)

        if match is None:

            try:
                self.device.image_save(
                    'jiangnan_city_not_found.png'
                )
            except Exception:
                pass

            return False

        logger.info(
            f'Switch to city: {target}'
        )

        self.click_city_match(
            match
        )

        # ========================================
        # 5. 等待城市加载
        # ========================================

        if not self.wait_city_arrival(city_target=target):

            logger.warning(
                f'Failed to enter city: {target}'
            )

            return False

        logger.info(
            f'City switching completed: {target}'
        )

        # The selected list entry and completed arrival establish the city.
        # Seal OCR is only an aid before navigation, never an arrival gate.
        return True
