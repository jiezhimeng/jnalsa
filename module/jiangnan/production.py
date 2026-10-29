import json
import re
import time
from datetime import datetime, timedelta
from module.jiangnan.production_state import (
    cached_production_group, production_record_pending, production_next_run,
    production_should_yield_to_dock,
)
from pathlib import Path

import cv2
from filelock import FileLock

from module.base.button import Button
from module.base.utils import load_image
from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.assets import PRODUCTION_IDLE_CHECK, PRODUCTION_WORKING_CHECK
from module.jiangnan.farm_production import FarmTextOcr, JiangnanFarmProduction
from module.jiangnan.foundation_icons import FACILITY_ICON, find_foundation_icon
from module.jiangnan.inventory_state import ALL_RESOURCES, JiangnanInventoryState
from module.logger import logger
from module.jiangnan.ocr import Digit, Ocr


class ProductionOptionOcr(Ocr):
    """读取生产选项中的基础产量或生产时间。"""

    def __init__(self, area, name):
        super().__init__(
            buttons=Button(area=area, color=(), button=area, name=name),
            lang='cnocr',
            name=name,
        )

    def pre_process(self, image):
        # 生产面板的黑字带灰色抗锯齿，硬阈值会把“3/5/分”等细笔画
        # 切断。保留原始 RGB，仅放大给 cnocr。
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


class JiangnanProduction(JiangnanFarmProduction):
    """逐城管理农田以外的生产设施。"""

    TASK = 'JiangnanProduction'
    STATE_FILE = Path('jiangnan_production_state.json')
    DEFAULT_PRIORITY = ('rich', 'many', 'some', 'little')
    PRODUCTION_CITIES = CITY_KEY_TO_NAME

    # 顺序同时用于 UI 默认值、扫描顺序和状态输出。
    FACILITIES = (
        ('餐馆', '菜肴'),
        ('林场', '原木'),
        ('鱼塘', '肉类'),
        ('锯木厂', '木板'),
        ('造纸厂', '纸'),
        ('棉花田', '棉花'),
        ('织布坊', '布'),
        ('裁缝铺', '成衣'),
        ('黏土矿', '黏土'),
        ('瓷窑', '瓷器'),
        ('烧炭厂', '木炭'),
        ('草药田', '草药'),
        ('炼丹炉', '丹丸'),
        ('桐油种植园', '桐油'),
        ('油纸厂', '油纸'),
        ('制伞坊', '油纸伞'),
        ('云锦织坊', '云锦'),
        ('漆树林', '生漆'),
        ('漆器作坊', '漆器'),
        ('螺钿漆坊', '螺钿'),
        ('菱角池塘', '蔬菜'),
        ('养蚕场', '蚕丝'),
        ('丝织坊', '丝绸'),
        ('绣坊', '刺绣'),
        ('养鸭场', '肉类'),
        ('铁矿场', '铁矿'),
        ('冶铁工坊', '铁板'),
        ('兵器工坊', '兵器'),
        ('花簪制坊', '花簪'),
        ('铜矿', '铜板'),
        ('竹林', '竹子'),
        ('茶园', '茶叶'),
        ('折扇作坊', '折扇'),
        ('团扇作坊', '团扇'),
        ('蟹塘', '肉类'),
        ('煮盐场', '盐'),
        ('标布作坊', '标布'),
        ('制鞋坊', '布鞋'),
        ('油菜花田', '蔬菜'),
        ('养猪场', '肉类'),
        ('徽墨作坊', '徽墨'),
        ('澄纸作坊', '澄纸'),
        ('歙砚作坊', '歙砚'),
        ('汪笔作坊', '汪笔'),
        ('采石场', '歙石'),
        ('铜镜厂', '铜镜'),
        ('玉器工坊', '玉器'),
        ('香粉厂', '香粉'),
        ('酒曲田', '酒曲'),
        ('越瓷窑', '越瓷'),
        ('羊圈', '羊毛'),
        ('乌毡帽厂', '乌毡帽'),
        ('渔船', '渔获'),
        ('鱼干坊', '鱼干'),
        ('苎麻田', '苎麻'),
        ('织网工坊', '渔网'),
        ('印刷厂', '年历'),
        ('琥珀酒坊', '白醪酒'),
        ('珍馐食肆', '梅干菜'),
        ('琳琅商铺', '油灯'),
        ('锦绣饰坊', '香囊'),
        ('绫罗鞋行', '木板鞋'),
        ('玉冠帽坊', '斗笠'),
        ('子衿衣阁', '深衣'),
        ('特产铺', '茶油'),
        ('雕刻行', '木雕'),
        ('改良农田', '蔬菜'),
    )
    FACILITY_RESOURCES = dict(FACILITIES)
    # 生产面板中仓库物资图标从左到右的稳定顺序。数值仍读取当前
    # 游戏界面，不能把攻略中的数值写死，因为不同城市的同名建筑
    # 可能使用不同配方（例如苏州餐馆）。未列物资表示只消耗铜钱。
    FACILITY_INPUT_RESOURCES = {
        '餐馆': ('蔬菜', '肉类'),
        '林场': (),
        '鱼塘': (),
        '锯木厂': ('菜肴', '原木'),
        '造纸厂': ('菜肴', '原木', '木板'),
        '棉花田': (),
        '织布坊': ('菜肴', '棉花'),
        '裁缝铺': ('菜肴', '布'),
        '黏土矿': ('菜肴',),
        '瓷窑': ('菜肴', '黏土'),
        '烧炭厂': ('菜肴', '原木'),
        '草药田': (),
        '炼丹炉': ('菜肴', '草药', '木炭'),
        '桐油种植园': ('菜肴',),
        '油纸厂': ('菜肴', '桐油'),
        '制伞坊': ('菜肴', '木板', '油纸'),
        '云锦织坊': ('菜肴', '布', '丝绸'),
        '漆树林': ('菜肴',),
        '漆器作坊': ('木板', '生漆'),
        '螺钿漆坊': ('菜肴', '漆器'),
        '菱角池塘': (),
        '养蚕场': (),
        '丝织坊': ('菜肴', '蚕丝'),
        '绣坊': ('菜肴', '丝绸'),
        '养鸭场': (),
        '铁矿场': ('菜肴',),
        '冶铁工坊': ('菜肴', '铁矿', '原木'),
        '兵器工坊': ('菜肴', '铁板', '原木'),
        '花簪制坊': ('菜肴', '铁板', '玉石'),
        '铜矿': ('菜肴',),
        '竹林': (),
        '茶园': ('菜肴',),
        '折扇作坊': ('竹子', '纸'),
        '团扇作坊': ('竹子', '丝绸'),
        '蟹塘': ('菜肴',),
        '煮盐场': ('菜肴',),
        '标布作坊': ('菜肴', '黑核棉'),
        '制鞋坊': ('菜肴', '标布'),
        '油菜花田': (),
        '养猪场': (),
        '徽墨作坊': ('木炭',),
        '澄纸作坊': ('纸',),
        '歙砚作坊': ('歙石',),
        '汪笔作坊': ('竹子',),
        '采石场': (),
        '铜镜厂': ('铜板',),
        '玉器工坊': ('菜肴', '玉石'),
        # 花瓣不占城市仓库容量，只扣除仓库内的粟米。
        '香粉厂': ('粟米',),
        '酒曲田': ('菜肴',),
        '越瓷窑': ('菜肴', '黏土'),
        '羊圈': (),
        '乌毡帽厂': ('羊毛',),
        '渔船': (),
        '鱼干坊': ('渔获',),
        '苎麻田': (),
        '织网工坊': ('苎麻',),
        '印刷厂': (),
        '琥珀酒坊': (),
        '珍馐食肆': (),
        '琳琅商铺': (),
        '锦绣饰坊': (),
        '绫罗鞋行': (),
        '玉冠帽坊': (),
        '子衿衣阁': (),
        # 多产物建筑会在 JiangnanMultiProductProduction 中按选中
        # 的产物覆盖这三个占位映射。
        '特产铺': (),
        '雕刻行': (),
        '改良农田': (),
    }
    # 采集类建筑和餐馆的行内有铜钱；制造类配方通常没有铜钱。
    # 该集合只决定跳过多少个非仓库数字，实际消耗量始终 OCR。
    COIN_COST_FACILITIES = {
        '餐馆', '林场', '鱼塘', '棉花田', '黏土矿', '草药田',
        '桐油种植园', '漆树林', '菱角池塘', '养蚕场', '养鸭场',
        '铁矿场', '铜矿', '采石场',
        '竹林', '茶园', '蟹塘',
        '油菜花田', '养猪场', '改良农田',
        '酒曲田', '越瓷窑', '羊圈', '乌毡帽厂',
        '渔船', '鱼干坊', '苎麻田', '织网工坊',
    }
    # 徽州部分作坊成本行末尾还有金箔。金箔不是仓库物资，读取库存
    # 成本时必须从尾部跳过，不能误记成最后一种原料。
    FACILITY_COST_SUFFIXES = {
        '汪笔作坊': 1,
        '徽墨作坊': 1,
        '澄纸作坊': 1,
        '歙砚作坊': 1,
        '香粉厂': 1,
    }
    TASK_FACILITIES = {
        'JiangnanRestaurantProduction': '餐馆',
        'JiangnanLoggingCampProduction': '林场',
        'JiangnanFishPondProduction': '鱼塘',
        'JiangnanSawmillProduction': '锯木厂',
        'JiangnanPaperMillProduction': '造纸厂',
        'JiangnanCottonFieldProduction': '棉花田',
        'JiangnanWeavingWorkshopProduction': '织布坊',
        'JiangnanTailorShopProduction': '裁缝铺',
        'JiangnanClayMineProduction': '黏土矿',
        'JiangnanPorcelainKilnProduction': '瓷窑',
        'JiangnanCharcoalKilnProduction': '烧炭厂',
        'JiangnanHerbFieldProduction': '草药田',
        'JiangnanAlchemyFurnaceProduction': '炼丹炉',
        'JiangnanTungOilPlantationProduction': '桐油种植园',
        'JiangnanOilPaperWorkshopProduction': '油纸厂',
        'JiangnanUmbrellaWorkshopProduction': '制伞坊',
        'JiangnanBrocadeWorkshopProduction': '云锦织坊',
        'JiangnanLacquerForestProduction': '漆树林',
        'JiangnanLacquerWorkshopProduction': '漆器作坊',
        'JiangnanMotherOfPearlWorkshopProduction': '螺钿漆坊',
        'JiangnanWaterCaltropPondProduction': '菱角池塘',
        'JiangnanSilkwormFarmProduction': '养蚕场',
        'JiangnanSilkWorkshopProduction': '丝织坊',
        'JiangnanEmbroideryWorkshopProduction': '绣坊',
        'JiangnanDuckFarmProduction': '养鸭场',
        'JiangnanIronMineProduction': '铁矿场',
        'JiangnanIronWorkshopProduction': '冶铁工坊',
        'JiangnanWeaponWorkshopProduction': '兵器工坊',
        'JiangnanHairpinWorkshopProduction': '花簪制坊',
        'JiangnanCopperMineProduction': '铜矿',
        'JiangnanBambooForestProduction': '竹林',
        'JiangnanTeaGardenProduction': '茶园',
        'JiangnanFoldingFanWorkshopProduction': '折扇作坊',
        'JiangnanRoundFanWorkshopProduction': '团扇作坊',
        'JiangnanCrabPondProduction': '蟹塘',
        'JiangnanSaltFieldProduction': '煮盐场',
        'JiangnanStandardClothWorkshopProduction': '标布作坊',
        'JiangnanShoeWorkshopProduction': '制鞋坊',
        'JiangnanRapeseedFieldProduction': '油菜花田',
        'JiangnanPigFarmProduction': '养猪场',
        'JiangnanInkWorkshopProduction': '徽墨作坊',
        'JiangnanChengPaperWorkshopProduction': '澄纸作坊',
        'JiangnanSheInkstoneWorkshopProduction': '歙砚作坊',
        'JiangnanWangBrushWorkshopProduction': '汪笔作坊',
        'JiangnanQuarryProduction': '采石场',
        'JiangnanCopperMirrorFactoryProduction': '铜镜厂',
        'JiangnanJadeWorkshopProduction': '玉器工坊',
        'JiangnanFragrancePowderFactoryProduction': '香粉厂',
        'JiangnanWineYeastFieldProduction': '酒曲田',
        'JiangnanYuePorcelainKilnProduction': '越瓷窑',
        'JiangnanSheepPenProduction': '羊圈',
        'JiangnanBlackFeltHatFactoryProduction': '乌毡帽厂',
        'JiangnanFishingBoatProduction': '渔船',
        'JiangnanDriedFishWorkshopProduction': '鱼干坊',
        'JiangnanRamieFieldProduction': '苎麻田',
        'JiangnanNetWeavingWorkshopProduction': '织网工坊',
    }

    OPTION_NAME = {
        'little': '少许',
        'some': '一些',
        'many': '许多',
        'rich': '丰厚',
    }
    # OCR 失败时使用游戏常见基础值；正常启动前会以当前面板 OCR 为准。
    LEVEL_AMOUNT = {1: 75, 2: 150, 3: 225, 4: 300}
    LEVEL_DURATION = {
        1: timedelta(minutes=3),
        2: timedelta(minutes=30),
        3: timedelta(hours=1),
        4: timedelta(hours=1, minutes=30),
    }
    OPTION_AMOUNT_AREAS = {
        # 只截取第一个黑色“+基础产量”，避开后方绿色人物加成。
        1: (445, 157, 510, 198),
        2: (820, 157, 900, 198),
        3: (445, 365, 515, 405),
        4: (820, 365, 900, 405),
    }
    OPTION_TIME_AREAS = {
        1: (475, 200, 665, 245),
        2: (850, 200, 1045, 245),
        3: (475, 408, 665, 455),
        4: (845, 408, 1055, 455),
    }
    OPTION_COST_AREAS = {
        1: (330, 245, 710, 300),
        2: (715, 245, 1090, 300),
        3: (330, 455, 710, 510),
        4: (715, 455, 1090, 510),
    }

    def __init__(
        self, config, device=None, task=None, path=None,
        facility=None, task_name=None,
    ):
        resolved_task = task_name or getattr(config, 'command', None) or task
        if resolved_task in self.TASK_FACILITIES:
            self.TASK = resolved_task
            facility = self.TASK_FACILITIES[resolved_task]
        if facility not in self.FACILITY_RESOURCES:
            raise ValueError(
                f'Unknown Jiangnan production facility: {facility!r}; '
                f'task={resolved_task!r}'
            )
        self.facility = facility
        self.resource = self.FACILITY_RESOURCES[facility]
        super().__init__(
            config=config,
            device=device,
            task=task,
            path=path or self.STATE_FILE,
        )
        self.option_amount_ocr = {
            level: ProductionOptionOcr(area, f'PRODUCTION_AMOUNT_{level}')
            for level, area in self.OPTION_AMOUNT_AREAS.items()
        }
        self.option_time_ocr = {
            level: ProductionOptionOcr(area, f'PRODUCTION_TIME_{level}')
            for level, area in self.OPTION_TIME_AREAS.items()
        }
        self.working_time_ocr = FarmTextOcr(
            self.FARM_WORKING_TIME_AREA, 'PRODUCTION_WORKING_TIME_OCR'
        )
        self._story_panel_corner = load_image(
            './assets/jiangnan/building/story_panel_corner.png',
            (0, 0, 27, 89),
        )
        self._last_production_target = None
        self._last_production_map_frame = None

    def _is_unexpected_story_panel(self):
        """识别误点地图装饰后打开的建筑介绍页。"""
        image = getattr(self.device, 'image', None)
        if image is None or image.shape[0] < 124 or image.shape[1] < 192:
            return False
        region = cv2.cvtColor(image[27:124, 157:192], cv2.COLOR_RGB2GRAY)
        reference = cv2.cvtColor(
            self._story_panel_corner, cv2.COLOR_RGB2GRAY,
        )
        score = cv2.minMaxLoc(cv2.matchTemplate(
            region, reference, cv2.TM_CCOEFF_NORMED,
        ))[1]
        return score >= 0.92

    def _read_resource_shortage(self, expected_resources=None):
        """Infer stock only when a known recipe explains the displayed deficit.

        The red number is the missing amount, not current inventory. A popup
        without a matching recipe still blocks production, but cannot safely
        update warehouse stock.
        """
        expected_amounts = {
            JiangnanInventoryState.normalize_resource(resource): int(amount)
            for resource, amount in dict(expected_resources or {}).items()
            if isinstance(amount, (int, float)) and amount > 0
        }
        expected = tuple(expected_amounts)
        image = self.device.image
        x1, y1, x2, y2 = (330, 240, 1080, 320)
        region = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 100, 60), (15, 255, 255)),
            cv2.inRange(hsv, (165, 100, 60), (179, 255, 255)),
        )
        contours, _ = cv2.findContours(
            red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if (10 <= height <= 35 and 2 <= width <= 100
                    and cv2.contourArea(contour) >= 8):
                boxes.append((left, top, left + width, top + height))
        boxes.sort()

        groups = []
        for box in boxes:
            if groups and box[0] - groups[-1][2] <= 20:
                old = groups[-1]
                groups[-1] = (
                    old[0], min(old[1], box[1]),
                    max(old[2], box[2]), max(old[3], box[3]),
                )
            else:
                groups.append(box)

        count_buttons = []
        name_buttons = []
        for index, (left, top, right, bottom) in enumerate(groups):
            center_x = x1 + (left + right) // 2
            count_area = (
                max(0, x1 + left - 10), max(0, y1 + top - 8),
                min(image.shape[1], x1 + right + 10),
                min(image.shape[0], y1 + bottom + 8),
            )
            count_buttons.append(Button(
                area=count_area, color=(), button=count_area,
                name=f'PRODUCTION_SHORTAGE_COUNT_{index + 1}',
            ))

            name_area = (
                max(330, center_x - 85), 125,
                min(1080, center_x + 85), 185,
            )
            name_buttons.append(Button(
                area=name_area, color=(), button=name_area,
                name=f'PRODUCTION_SHORTAGE_NAME_{index + 1}',
            ))

        # 标题、全部物资名称和全部数量都来自同一张弹窗。使用一个通用
        # 中文 OCR 批次读取，数字在本地提取，避免按字段启动三轮推理。
        header_button = Button(
            area=(330, 65, 700, 125), color=(),
            button=(330, 65, 700, 125),
            name='PRODUCTION_SHORTAGE_HEADER',
        )
        raw_values = Ocr(
            [header_button] + name_buttons + count_buttons,
            lang='cnocr',
            name='PRODUCTION_SHORTAGE_BATCH',
        ).ocr(image)
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        raw_values.extend(
            [''] * (1 + len(name_buttons) + len(count_buttons) - len(raw_values))
        )
        header = re.sub(r'\s+', '', str(raw_values[0]))
        if '缺少' not in header or '资源' not in header:
            return None
        names = raw_values[1:1 + len(name_buttons)]
        counts = raw_values[1 + len(name_buttons):]

        readings = {}
        for index, (count, raw_name) in enumerate(zip(counts, names)):
            try:
                digits = ''.join(re.findall(r'\d+', str(count)))
                count = int(digits)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            name = JiangnanInventoryState.normalize_resource(raw_name)
            if name not in expected:
                matched = next(
                    (resource for resource in expected if resource in name),
                    None,
                )
                if matched is not None:
                    name = matched
                elif len(expected) == 1 and len(groups) == 1:
                    name = expected[0]
            if name in expected_amounts and count <= expected_amounts[name]:
                readings[name] = expected_amounts[name] - count
                logger.info(
                    f'Production shortage implies live stock: {name} '
                    f'{expected_amounts[name]} required - {count} missing '
                    f'= {readings[name]}'
                )
            elif name in expected or (not expected and name in ALL_RESOURCES):
                logger.info(
                    f'Production shortage without known requirement: '
                    f'{name} missing={count}; keep warehouse stock unchanged'
                )
            else:
                logger.warning(
                    f'Ignore unidentified shortage item: name={name!r}, '
                    f'count={count}, expected={expected}'
                )

        logger.attr('ProductionShortageInventory', readings)
        # Header recognition alone is enough to prove this is the blocking
        # popup.  Keep a truthy unknown marker so the caller still closes it;
        # unreadable numbers leave the existing inventory unchanged.
        return readings or {
            resource: -1 for resource in expected
        } or {'未知物资': -1}

    def _load_state(self):
        try:
            with self.path.open('r', encoding='utf-8') as stream:
                data = json.load(stream)
            if isinstance(data, dict):
                data.setdefault('cities', {})
                return data
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f'Unable to read Jiangnan production state: {exc}')
        return {'version': 1, 'cities': {}}

    @staticmethod
    def parse_search_result(text, building_name=None):
        text = re.sub(r'\s+', '', str(text))
        name = re.escape(str(building_name or ''))
        matched = re.search(
            # cnocr 偶尔会漏掉建筑等级后的“级”（实机菱角池塘为
            # “4菱角池塘(3)”）。建筑名仍要求完整一致，因此允许省略
            # 这个单字不会把相似建筑当成目标。
            rf'([12345一二三四五])级?{name}(?:\D*(\d+)\D*)?$', text
        )
        if not matched:
            # 改良农田的建筑搜索卡片只写“改良农田(数量)”，没有
            # “X级”前缀；它的面板仍提供四个生产档位。只对此建筑接受
            # 无等级标签，避免把其他设施的残缺 OCR 当成有效结果。
            if building_name == '改良农田':
                ungraded = re.fullmatch(
                    r'改良农田[（(]?(\d+)[）)]?', text
                )
                if ungraded:
                    return 4, int(ungraded.group(1))
            return None, None
        levels = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
        return int(levels.get(matched.group(1), matched.group(1))), int(
            matched.group(2) or 1
        )

    def _production_key(self, city_key, field):
        if self.TASK == 'JiangnanImprovedFarmProduction':
            group = f'improved_farm_{city_key}'
        elif self.TASK in {
            'JiangnanAmberWineryProduction',
            'JiangnanGourmetRestaurantProduction',
            'JiangnanExquisiteGoodsShopProduction',
            'JiangnanEmbroideryAccessoriesWorkshopProduction',
            'JiangnanLuoShoesProduction',
            'JiangnanJadeCrownHatWorkshopProduction',
            'JiangnanZijinClothingPavilionProduction',
        }:
            group = f'multi_product_{city_key}'
        elif (
            self.TASK == 'JiangnanRestaurantProduction'
            and city_key == 'huizhoufu'
        ):
            group = 'restaurant_huizhoufu'
        else:
            group = f'production_{city_key}'
        return f'{self.TASK}.{group}.{field}'

    def _production_priority(self, city=None, field_prefix='Priority'):
        city_key = next(
            (key for key, name in CITY_KEY_TO_NAME.items() if name == city),
            None,
        )
        priority = []
        for index in range(1, 5):
            key = (
                self._production_key(
                    city_key, f'{field_prefix}{index}'
                )
                if city_key is not None
                else f'{self.TASK}.ProductionPriority.Priority{index}'
            )
            value = str(self.config.cross_get(
                keys=key,
                default=self.DEFAULT_PRIORITY[index - 1],
            ))
            if value in self.OPTION_LEVEL and value not in priority:
                priority.append(value)
        return priority

    def _result_validator(self, building_name):
        return lambda value: self.parse_search_result(
            value, building_name
        )[0] is not None

    def _find_centered_production(self):
        """按设施类别找地基图标；图标缺失时避免误点邻近建筑。"""
        self._foundation_icon_occluded = False
        arrow = self.find_building_search_arrow()
        if arrow is not None:
            icon = self._find_foundation_production_icon(arrow)
            if icon is not None:
                return icon
            if self.facility in FACILITY_ICON:
                # 切换同类建筑后，搜索箭头常先出现，镜头稍后才把
                # 地基图标移到箭头下方。短暂等待实际画面稳定后再判缺失。
                for _ in range(2):
                    self.device.sleep(0.35)
                    self.device.screenshot()
                    arrow = self.find_building_search_arrow()
                    if arrow is None:
                        continue
                    icon = self._find_foundation_production_icon(arrow)
                    if icon is not None:
                        return icon
                # The old paper-color fallback often matched an empty tile
                # below the arrow (for example 杭州府 养蚕场), causing repeated
                # map taps instead of the configured occlusion delay.
                logger.warning(
                    f'Expected {FACILITY_ICON[self.facility]} foundation icon '
                    f'for {self.facility} not found; skip uncertain map tap'
                )
                self._foundation_icon_occluded = True
                self.save_abnormal_screenshot('production_foundation_icon_missing')
                return None
        # 普通建筑定位只检查尖端下方 130px；生产地基图标在实机中
        # 可以位于下方约 180px，因此先使用上面的图标定位。
        target = self.find_centered_search_target()
        if target is None:
            logger.warning(
                'Production building below selection arrow was not recognized'
            )
            return None
        score = getattr(self, 'last_search_target_score', None)
        if score is not None and score < 0:
            # 负分轮廓通常是远离箭头的地面线条。实机日志中林场被选到
            # score=-1447 的道路斑块，连点两次也不会打开生产面板。
            banner = self._production_banner_at_arrow()
            if banner is not None:
                return banner
            logger.warning(
                f'Production foreground candidate is unreliable: score={score}; '
                'skip map tap'
            )
            return None
        if self.last_search_target_arrow_fallback:
            banner = self._production_banner_at_arrow()
            if banner is not None:
                # 地基模式的“生产”竖旗会盖住建筑图标。通用轮廓找不到
                # 建筑时，尖端下方 42px 已是旗杆或空地；点可见的旗身。
                return banner
        return Button(
            area=(),
            color=(),
            button=target.button,
            name='PRODUCTION_BUILDING_DYNAMIC',
        )

    def _find_foundation_production_icon(self, arrow):
        """Use the user-supplied icon family, independent of tile background."""
        match = find_foundation_icon(
            getattr(self.device, 'image', None), arrow, self.facility,
            max_horizontal=75,
        )
        if match is None:
            return None
        center_x, center_y, score, family = match
        logger.info(
            'Production foundation icon found: '
            f'facility={self.facility}, family={family}, '
            f'center=({center_x}, {center_y}), '
            f'score={score:.3f}'
        )
        return Button(
            area=(), color=(),
            button=(center_x - 6, center_y - 6,
                    center_x + 6, center_y + 6),
            name='PRODUCTION_FOUNDATION_ICON_DYNAMIC',
        )

    def _production_banner_at_arrow(self):
        arrow = self.last_search_arrow
        image = self.device.image
        if arrow is None or image is None:
            return None
        x, tip, _ = arrow
        height, width = image.shape[:2]
        if not (x >= 20 and x + 20 < width and tip + 16 < height):
            return None
        patch = image[tip + 2:tip + 16, x - 20:x + 20]
        # 生产竖旗的纸面比地面暗，箭头消失后仍在原处；只在纸面占据
        # 绝大多数窄条时才改点旗身，避免对空白地基误点。
        paper = (
            (patch[:, :, 0] >= 150)
            & (patch[:, :, 0] <= 210)
        )
        coverage = float(paper.mean())
        logger.attr('ProductionBannerBelowArrow', f'{coverage:.3f}')
        if coverage < 0.65:
            return None
        return Button(
            area=(), color=(),
            button=(x - 5, tip + 5, x + 5, tip + 11),
            name='PRODUCTION_BANNER_DYNAMIC',
        )

    def _panel_state(self):
        speed_role = self._speed_popup_role()
        if speed_role == 'unknown':
            return None, None
        if speed_role is not None:
            text = self._ocr_text(self.working_time_ocr, self.device.image)
            duration = self.parse_duration(text)
            if speed_role == 'upgrading':
                logger.attr('ProductionUpgradeTimeText', text)
                if duration is not None and duration > timedelta(0):
                    return 'upgrading', duration
                return None, None
            logger.attr('ProductionWorkingTimeText', text)
            return 'working', duration
        if self.appear(PRODUCTION_IDLE_CHECK, offset=(5, 5), similarity=0.80):
            return 'idle', None
        return None, None

    def _open_current_production(self, timeout=9, target=None):
        self.device.screenshot()
        navigation_buttons = self._find_navigation_buttons() if target is None else ()
        if target is None and len(navigation_buttons) < 2:
            # “上一个/下一个”按钮只在切换同类建筑时必需。首个建筑
            # 已由搜索箭头明确定位，按钮颜色受城市滤镜影响时仍应继续
            # 点击箭头下的建筑，不能把导航识别失败等同于建筑不匹配。
            logger.warning(
                'Production navigation panel is missing; '
                'continue with selection-arrow target'
            )
        if target is None:
            target = self._find_centered_production()
        if target is None:
            return None, None
        selection_frame = getattr(self.device, 'image', None)
        if selection_frame is not None:
            selection_frame = selection_frame.copy()
        self._last_production_target = target
        self._last_production_map_frame = selection_frame
        self._click_searched_building(target, navigation_buttons)
        deadline = time.time() + timeout
        clicked_at = time.time()
        retried = False
        while time.time() < deadline:
            self.device.sleep(0.25)
            self.device.screenshot()
            force_button = self.find_warehouse_full()
            if (force_button is not None
                    or getattr(self, '_warehouse_full_banner_visible', False)):
                # The full banner can be certain while its safe left-hand
                # action is covered.  Report overflow and let the caller
                # skip harvest; do not wait until the banner disappears and
                # mislabel the city map as an unknown production panel.
                return 'overflow', None
            state, duration = self._panel_state()
            if state is not None:
                return state, duration
            # 建筑介绍和生产面板共用外框；先识别可操作的生产面板。
            if self._is_unexpected_story_panel():
                # The production panel shares the story frame corner. During
                # its opening animation the idle anchor can miss for one
                # frame; do not press BACK on that transient screenshot.
                if time.time() - clicked_at < 0.9:
                    continue
                self.device.sleep(0.3)
                self.device.screenshot()
                state, duration = self._panel_state()
                if state is not None:
                    return state, duration
                if self._is_unexpected_story_panel():
                    logger.warning(
                        'Building story opened instead of production panel'
                    )
                    self.device.press_back()
                    return None, None
            if not retried and time.time() - clicked_at >= 1.5 and self.is_in_main():
                retried = True
                if self.find_building_search_arrow(
                    allow_retained=False, report_missing=False,
                ) is None:
                    if self._map_view_unchanged(
                        selection_frame, getattr(self.device, 'image', None),
                    ):
                        logger.info(
                            'Production selection arrow disappeared while '
                            'map stayed in place; tap the same building again'
                        )
                        retry_target = Button(
                            area=(), color=(), button=target.button,
                            name='PRODUCTION_BUILDING_SAME_PLACE_RETRY',
                        )
                        self.device.click(retry_target)
                        clicked_at = time.time()
                        continue
                    logger.warning(
                        'Production selection disappeared and map changed; '
                        're-search the building'
                    )
                    return None, None
                retry_target = self._find_centered_production()
                if retry_target is not None:
                    self.device.click(retry_target)
                elif self._foundation_icon_occluded:
                    # The old click did not open a panel and the icon has
                    # since become hidden. Waiting longer cannot resolve it.
                    return None, None
        logger.warning('Production panel state was not recognized')
        self.save_abnormal_screenshot('production_panel_unknown')
        return None, None

    def _actual_started_duration(self, option_duration, building_index=1):
        """开工后重新打开设施，以实际加速倒计时校正完成节点。"""
        self.device.screenshot()
        target = getattr(self, '_last_production_target', None)
        previous_map = getattr(self, '_last_production_map_frame', None)
        if (target is not None and previous_map is not None
                and self.is_in_main()
                and self._map_view_unchanged(
                    previous_map, getattr(self.device, 'image', None),
                )):
            logger.info(
                f'Production map stayed in place after start; '
                f'reopen {self.facility} at the same location'
            )
            state, duration = self._open_current_production(
                timeout=9, target=target,
            )
            if state == 'working' and duration is not None and duration > timedelta(0):
                observed_finish = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
                return max(
                    timedelta(seconds=1),
                    observed_finish - datetime.now().replace(microsecond=0),
                )
            if state is not None:
                self._close_panel()
            logger.warning(
                f'Unable to read {self.facility} time at retained location; '
                'fall back to building search'
            )
        else:
            logger.info(
                f'Production map changed or retained target unavailable; '
                f'relocate {self.facility}'
            )
        if not self.search_building(
            self.facility,
            result_validator=self._result_validator(self.facility),
            exact=True,
        ):
            logger.warning(
                f'Unable to relocate {self.facility} after start; '
                f'use option duration {option_duration}'
            )
            return option_duration
        if not self._select_production_instance(
            self.facility, building_index,
        ):
            logger.warning(
                f'Unable to select started {self.facility}; '
                f'use option duration {option_duration}'
            )
            return option_duration
        state, duration = self._open_current_production(timeout=9)
        if state == 'working' and duration is not None and duration > timedelta(0):
            observed_finish = datetime.now().replace(microsecond=0) + duration
            logger.info(
                f'Production actual duration after start: {duration} '
                f'(option={option_duration})'
            )
            self._close_panel()
            return max(
                timedelta(seconds=1),
                observed_finish - datetime.now().replace(microsecond=0),
            )
        if state is not None:
            self._close_panel()
        logger.warning(
            'Unable to read actual production duration after start; '
            f'use option duration {option_duration}'
        )
        return option_duration

    @staticmethod
    def _parse_amount(text):
        text = str(text).translate(str.maketrans({
            'O': '0', 'o': '0', 'I': '1', 'l': '1',
            # cnocr 常把末尾细长的 0 识别成左括号。
            '(': '0', '（': '0',
        }))
        matched = re.search(r'\+\D{0,2}(\d+)', text)
        if matched is None:
            matched = re.search(r'(\d+)', text)
        return int(matched.group(1)) if matched else None

    def _read_option(self, selected):
        # 连续读三帧。不同建筑的产量和耗时差异很大，不能拿农田或
        # 餐馆的常量猜测，否则会污染库存台账。
        for attempt in range(1, 4):
            # 产量和耗时位于同一张生产面板，同帧一次提交给 OCR。
            # 旧逻辑依次调用两次模型，既多一次 IPC，也可能在动画中
            # 读到两个不同时间点的画面。
            buttons = [
                Button(
                    area=self.OPTION_AMOUNT_AREAS[selected], color=(),
                    button=self.OPTION_AMOUNT_AREAS[selected],
                    name=f'PRODUCTION_AMOUNT_{selected}',
                ),
                Button(
                    area=self.OPTION_TIME_AREAS[selected], color=(),
                    button=self.OPTION_TIME_AREAS[selected],
                    name=f'PRODUCTION_TIME_{selected}',
                ),
            ]
            raw = Ocr(
                buttons=buttons, lang='cnocr',
                name=f'PRODUCTION_OPTION_{selected}_BATCH',
            ).ocr(self.device.image)
            if not isinstance(raw, list):
                raw = [raw]
            raw.extend([''] * (2 - len(raw)))
            amount_text, time_text = map(str, raw[:2])
            amount = self._parse_amount(amount_text)
            duration = self.parse_duration(time_text)
            logger.attr(
                f'ProductionOption{selected}Try{attempt}',
                f'amount={amount_text!r}->{amount}, '
                f'time={time_text!r}->{duration}',
            )
            if (
                amount is not None and 0 < amount <= 100000
                and duration is not None
                and timedelta(0) < duration <= timedelta(days=7)
            ):
                return amount, duration
            if attempt < 3:
                self.device.sleep(0.25)
                self.device.screenshot()
        logger.warning(
            f'Production option {selected} amount/time OCR failed; '
            'do not start with guessed data'
        )
        return None, None

    def _cost_number_areas(self, image, selected):
        """按连通域拆出成本行数字，避免图标使整行 OCR 数字粘连。"""
        x1, y1, x2, y2 = self.OPTION_COST_AREAS[selected]
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        components = sorted([
            (x, y, width, height, area)
            for x, y, width, height, area in stats[1:]
            if (
                2 <= width <= 18 and 16 <= height <= 27
                and 25 <= area <= 260 and 12 <= y <= 28
            )
        ])
        groups = []
        for component in components:
            x, _, _, _, _ = component
            if (
                groups
                and x - (groups[-1][-1][0] + groups[-1][-1][2]) <= 8
            ):
                groups[-1].append(component)
            else:
                groups.append([component])
        return [
            (
                x1 + group[0][0] - 3,
                y1 + 10,
                x1 + group[-1][0] + group[-1][2] + 3,
                y2 - 8,
            )
            for group in groups
        ]

    def _read_cost_values(self, selected):
        areas = self._cost_number_areas(self.device.image, selected)
        buttons = []
        for index, area in enumerate(areas, start=1):
            buttons.append(Button(
                area=area, color=(), button=area,
                name=f'PRODUCTION_COST_{selected}_{index}',
            ))
        if not buttons:
            return []
        # 同一成本行中的全部数字来自同一帧，一次批量识别，避免每个
        # 图标后的数字都单独启动一次 PP-OCR 推理。
        raw = Digit(
            buttons, lang='azur_lane', letter=(54, 47, 43),
            threshold=180,
            name=f'PRODUCTION_COST_{selected}_BATCH',
        ).ocr(self.device.image)
        if not isinstance(raw, list):
            raw = [raw]
        values = []
        for value in raw:
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                values.append(-1)
        return values

    def _read_costs(self, selected):
        resources = self.FACILITY_INPUT_RESOURCES[self.facility]
        if not resources:
            return {}
        prefix = 2 if self.facility in self.COIN_COST_FACILITIES else 1
        suffix = self.FACILITY_COST_SUFFIXES.get(self.facility, 0)
        readings = []
        for attempt in range(1, 4):
            values = self._read_cost_values(selected)
            expected_max = prefix + len(resources) + suffix
            if len(values) > expected_max:
                # On the observed 花簪制坊/玉器工坊 rows, an icon stroke
                # becomes an extra OCR group read as 0. Remove it only when
                # doing so restores the exact expected column count.
                middle_end = len(values) - suffix if suffix else len(values)
                corrected = (
                    values[:prefix]
                    + [value for value in values[prefix:middle_end]
                       if value != 0]
                    + (values[middle_end:] if suffix else [])
                )
                if len(corrected) == expected_max:
                    logger.info(
                        f'Ignore extra zero-valued cost glyph: '
                        f'{values} -> {corrected}'
                    )
                    values = corrected
            material_values = values[prefix:-suffix] if suffix else values[prefix:]
            valid = (
                1 <= len(material_values) <= len(resources)
                and all(0 < value <= 100000 for value in material_values)
            )
            logger.attr(
                f'ProductionCost{selected}Try{attempt}',
                f'all={values}, materials={material_values}, valid={valid}',
            )
            if valid:
                costs = dict(zip(resources, material_values))
                readings.append(costs)
                if len(readings) >= 2 and readings[-1] == readings[-2]:
                    return costs
            if attempt < 3:
                self.device.sleep(0.25)
                self.device.screenshot()
        logger.warning(
            f'Production option {selected} cost OCR failed or unstable; '
            'do not start production'
        )
        return None

    def _start_by_priority(
        self, building_level, city, inventory_state, building_index=1
    ):
        self.last_priority_empty = False
        if getattr(self, 'defer_new_production', False):
            self.last_no_workers = True
            self._close_panel()
            return None, None, None
        option, selected = self._select_production_level(
            building_level, city=city
        )
        self.last_priority_empty = selected is None
        if selected is None:
            logger.info(
                f'Do not start {city} {self.facility}: '
                'all configured priorities are 无 or unavailable'
            )
            self._close_panel()
            return None, None, None
        amount, duration = self._read_option(selected)
        if amount is None or duration is None:
            return None, None, None
        costs = self._read_costs(selected)
        if costs is None:
            return None, None, None
        enough, missing = inventory_state.can_consume_inventory(city, costs)
        if not enough:
            logger.warning(
                f'Do not start {city} {self.facility}: '
                f'insufficient warehouse inventory {missing}'
            )
            return None, None, None
        logger.info(
            f'Start production by priority: {self.OPTION_NAME[option]} '
            f'(slot={selected}, amount={amount}, duration={duration}, '
            f'costs={costs})'
        )
        self.device.click(Button(
            area=(), color=(), button=self.LEVEL_BUTTONS[selected],
            name=f'PRODUCTION_LEVEL_{selected}_DYNAMIC',
        ))
        if not self._direct_start(
            city=city, expected_resources=costs
        ):
            if self.last_no_workers:
                return None, None, None
            if self.last_insufficient_inventory:
                inventory_state.reconcile_live_inventory(
                    city,
                    self.last_insufficient_inventory,
                    reason=f'{self.facility}开工资源不足提示',
                )
                return None, None, None
            state, _ = self._panel_state()
            if state != 'idle':
                return None, None, None
            logger.warning(
                f'{self.facility} option click was not committed; retry once'
            )
            self.device.click(Button(
                area=(), color=(), button=self.LEVEL_BUTTONS[selected],
                name=f'PRODUCTION_LEVEL_{selected}_RETRY',
            ))
            if not self._direct_start(
                city=city, expected_resources=costs
            ):
                if self.last_insufficient_inventory:
                    inventory_state.reconcile_live_inventory(
                        city,
                        self.last_insufficient_inventory,
                        reason=f'{self.facility}开工资源不足提示',
                    )
                return None, None, None
        duration = self._actual_started_duration(
            duration, building_index=building_index
        )
        if not inventory_state.consume_inventory(city, costs):
            # 游戏已经开始生产，但台账无法安全扣除时必须尽快实扫校正。
            inventory_state.request_scan(
                f'{city}{self.facility}生产已开始但成本扣除失败',
                cities={city},
            )
        return amount, duration, costs

    def _next_production(self):
        return self._next_farm()

    def _select_production_instance(self, facility, index):
        """收获可能关闭实例选择器；仅切换失败时重新搜索一次。"""
        for attempt in range(2):
            if attempt and not self.search_building(
                facility,
                result_validator=self._result_validator(facility),
                exact=True,
            ):
                return False
            if all(self._next_production() for _ in range(max(0, int(index) - 1))):
                return True
            logger.info(
                f'{facility} instance selector disappeared; re-search '
                f'{index} after cleanup'
            )
        return False

    def _reference(self, city, facility, level, index):
        return f'production:{city}:{facility}:{level}:{index}'

    def _known_amount(self, city, facility, index, level):
        item = (
            self.data.get('cities', {}).get(city, {})
            .get('facilities', {}).get(facility, {})
            .get('buildings', {}).get(str(index), {})
        )
        try:
            amount = int(item.get('amount', 0))
        except (TypeError, ValueError):
            amount = 0
        # 首次接管时已经在生产的建筑无法从加速弹窗反推出原生产档位。
        # 只复用本模块此前准确 OCR 后保存的产量，不猜测。
        return amount if amount > 0 else 0

    def _disable_facility(self, city_key, city, facility):
        if self.TASK == 'JiangnanImprovedFarmProduction':
            group_prefix = 'improved_farm'
        elif self.TASK in {
            'JiangnanAmberWineryProduction',
            'JiangnanGourmetRestaurantProduction',
            'JiangnanExquisiteGoodsShopProduction',
            'JiangnanEmbroideryAccessoriesWorkshopProduction',
            'JiangnanLuoShoesProduction',
            'JiangnanJadeCrownHatWorkshopProduction',
            'JiangnanZijinClothingPavilionProduction',
        }:
            group_prefix = 'multi_product'
        elif (
            self.TASK == 'JiangnanRestaurantProduction'
            and city_key == 'huizhoufu'
        ):
            group_prefix = 'restaurant'
        else:
            group_prefix = 'production'
        self.disable_city_building_feature(
            task=self.TASK,
            group_prefix=group_prefix,
            city_key=city_key,
            city=city,
            building_name=facility,
            status_field='Status',
            status_text=f'无{facility}，已自动关闭',
            next_run_field='NextRun',
            next_run_value=self.DEFAULT_TIME,
        )

    def inspect_facility(
        self, city_key, city, facility, resource, inventory_state,
        prelocated_result=None, verify_all_instances=False,
    ):
        cached = cached_production_group(
            getattr(self, 'data', {}).get('cities', {}).get(city, {})
            .get('facilities', {}).get(facility, {}).get('buildings', {}),
            inventory_state, city, (resource,),
        )
        if cached is not None and not verify_all_instances:
            logger.info(f'Skip working production group before finish: {city} {facility}')
            return {'unavailable': False, 'buildings': cached}
        if prelocated_result:
            self.last_search_result_name = str(prelocated_result)
            self.last_search_status = self.SEARCH_FOUND
            self.last_search_executed = True
            located = True
        else:
            located = self.search_building(
                facility,
                result_validator=self._result_validator(facility),
                exact=True,
            )
        if not located:
            if self.building_search_confirmed_missing():
                self._disable_facility(city_key, city, facility)
                return {'unavailable': True, 'buildings': {}}
            return None
        level, total = self.parse_search_result(
            self.last_search_result_name, facility
        )
        if level is None or total is None or total <= 0:
            return None

        logger.attr('ProductionSearchGroup', f'{facility}: level={level}, count={total}')
        previous = (
            self.data.get('cities', {}).get(city, {})
            .get('facilities', {}).get(facility, {}).get('buildings', {})
        )
        # Yielding to a dock job or failing to select an instance must not
        # erase the uninspected buildings and their production clocks.
        buildings = {
            str(index): dict(previous[str(index)])
            for index in range(1, total + 1)
            if str(index) in previous
        }
        inventory_refreshed = False
        workers_exhausted = False
        inspected = False
        now = datetime.now().replace(microsecond=0)
        for index in range(1, total + 1):
            old = previous.get(str(index), {})
            if not verify_all_instances and production_record_pending(old):
                logger.info(f'Skip working {facility} {index}/{total} before {old.get("finish_at")}')
                continue
            if (old.get('state') == 'threshold'
                    and inventory_state.production_below_threshold(city, resource) is False):
                continue
            if index > 1 and production_should_yield_to_dock(inventory_state):
                logger.info(
                    f'Yield {facility} {index}/{total} '
                    f'to arrived dock shipment'
                )
                inventory_state.sync_dock_schedule()
                break
            if index > 1:
                if inspected and not self.search_building(
                    facility,
                    result_validator=self._result_validator(facility),
                    exact=True,
                ):
                    logger.warning(f'Unable to relocate {facility} {index}/{total}')
                    break
                if not self._select_production_instance(facility, index):
                    logger.warning(f'Unable to select {facility} {index}/{total}')
                    break

            inspected = True
            state, duration = self._open_current_production()
            if state is None and getattr(self, '_foundation_icon_occluded', False):
                state = 'occluded'
            state, duration = self._confirm_live_production_state(
                old, state, duration, f'{city} {facility} {index}/{total}'
            )
            amount = self._known_amount(city, facility, index, level)
            if old.get('state') == 'working' and state == 'upgrading':
                inventory_state.cancel_inventory_change(
                    self._reference(city, facility, level, index)
                )
            if state == 'working' and (
                old.get('state') != 'working'
                or self._last_production_record_drift
            ):
                amount = 0
                inventory_state.cancel_inventory_change(
                    self._reference(city, facility, level, index)
                )
            finish_at = None
            costs = {}
            reconciliation = None
            if state == 'overflow':
                inventory_state.cancel_inventory_change(
                    self._reference(city, facility, level, index),
                    rollback_credited=True,
                )
                if self.resolve_warehouse_full(force=True):
                    logger.info(
                        f'Force-harvest completed {facility}; '
                        'reopen before threshold decision'
                    )
                    state, duration = self._open_current_production()
                else:
                    finish_at = now
            if state == 'idle':
                use_cached = inventory_refreshed and old.get('state') not in (
                    'working', 'overflow'
                )
                if use_cached:
                    logger.info(f'Reuse facility inventory scan: {city} {facility}')
                refresh_targets = (resource,) + tuple(
                    self.FACILITY_INPUT_RESOURCES.get(facility, ())
                )
                refreshed = use_cached or self._refresh_live_inventory_from_production_panel(
                    city,
                    inventory_state,
                    refresh_targets,
                    required_resources=(resource,),
                )
                inventory_refreshed = inventory_refreshed or refreshed
                if refreshed and old.get('state') in ('working', 'overflow'):
                    reconciliation = inventory_state.reconcile_production_harvest(
                        city=city,
                        resource=old.get('resource') or resource,
                        amount=old.get('amount') or amount,
                        reference=self._reference(
                            city, facility,
                            int(old.get('level') or level), index,
                        ),
                        expected_at=old.get('finish_at'),
                        observed_at=datetime.now().replace(microsecond=0),
                        included_in_scan=True,
                    )
                if not refreshed:
                    logger.warning(
                        f'Unable to refresh live inventory before production: '
                        f'{city} {facility}'
                    )
                    below_threshold = None
                else:
                    below_threshold = inventory_state.production_below_threshold(
                        city, resource
                    )
                if below_threshold is False:
                    inventory_state.pause_city_resource_production(
                        city,
                        resource,
                        self.TASK,
                        reason='库存达到97%',
                    )
                    state = 'threshold'
                    self._close_panel()
                elif below_threshold is None:
                    state = 'unknown'
                    logger.warning(
                        f'Unable to evaluate 97% production threshold: '
                        f'{city} {resource}'
                    )
                    self._close_panel()
                elif workers_exhausted:
                    state = 'no_workers'
                    self._close_panel()
                else:
                    amount, duration, costs = self._start_by_priority(
                        level, city, inventory_state, building_index=index
                    )
                    if amount is not None and duration is not None:
                        state = 'working'
                        finish_at = datetime.now().replace(microsecond=0) + duration
                    elif self.last_priority_empty:
                        state = 'no_priority'
                    elif self.last_no_workers:
                        state = 'no_workers'
                        workers_exhausted = True
                    else:
                        state = 'unknown'
                        self._close_panel()
            elif state == 'working':
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
            elif state == 'upgrading':
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
            elif state == 'occluded':
                finish_at = datetime.now().replace(microsecond=0) + timedelta(hours=12)
                logger.warning(
                    f'{city} {facility} {index}/{total} foundation icon '
                    f'occluded; defer until {self._time_text(finish_at)}'
                )

            if state == 'working' and finish_at is not None and amount > 0:
                inventory_state.schedule_inventory_change(
                    city, resource, amount, finish_at,
                    reason='生产',
                    reference=self._reference(city, facility, level, index),
                )
            buildings[str(index)] = {
                'level': level, 'index': index, 'total': total,
                'state': state or 'unknown', 'resource': resource,
                'amount': int(amount or 0),
                'costs': dict(costs or {}),
                'finish_at': self._time_text(finish_at) if finish_at else None,
                'harvest_reconciliation': reconciliation,
                'harvest_reconciled_at': (
                    self._time_text(datetime.now()) if reconciliation else None
                ),
                'updated_at': self._time_text(datetime.now()),
            }
            self._save_production_progress(city, buildings, facility=facility)
            if state == 'threshold':
                logger.info(
                    f'Stop searching remaining {facility} in {city}: '
                    f'{resource} reached the production threshold'
                )
                break
        return {'unavailable': False, 'buildings': buildings}

    def inspect_city(
        self, city_key, city, inventory_state, prelocated_result=None,
    ):
        result = self.inspect_facility(
            city_key, city, self.facility, self.resource, inventory_state,
            prelocated_result=prelocated_result,
        )
        # City production creates its runners before inspecting any facility.
        # Each runner therefore owns an older copy of the same state file.
        # Merge only this inspection into the latest state, after game actions
        # finish, so later runners cannot revert earlier facilities or cities.
        with FileLock(str(self.path) + '.lock', timeout=10):
            latest = self._load_state()
            city_data = latest.setdefault('cities', {}).setdefault(city, {})
            facilities = city_data.setdefault('facilities', {})
            if result is not None:
                facilities[self.facility] = result
            self.data = latest
            self.save()
        return result is not None

    def _sync_city_ui(self, city_key, city, fallback):
        city_data = self.data.get('cities', {}).get(city, {})
        lines = []
        has_overflow = False
        for facility, _ in ((self.facility, self.resource),):
            data = city_data.get('facilities', {}).get(facility, {})
            has_overflow = any(item.get('state') == 'overflow' for item in data.get('buildings', {}).values())
            if data.get('unavailable'):
                lines.append(f'[{facility}][无此建筑][已关闭]')
                continue
            for item in sorted(
                data.get('buildings', {}).values(),
                key=lambda value: int(value.get('index', 0)),
            ):
                state = item.get('state')
                finish = self._parse_time(item.get('finish_at'))
                if state == 'occluded':
                    label = f'{facility}{item.get("index", 0)}/{item.get("total", 0)}'
                    lines.append(
                        f'[{label}][地基图标遮挡]'
                        f'[{self._time_text(finish) if finish is not None else "等待复查"}]'
                    )
                    continue
                if state == 'upgrading':
                    level = self.LEVEL_CN.get(
                        int(item.get('level', 0)), str(item.get('level', 0))
                    )
                    label = (
                        f'{level}级{facility}{item.get("index", 0)}/'
                        f'{item.get("total", 0)}'
                    )
                    if finish is None:
                        lines.append(f'[{label}][升级中][倒计时识别失败]')
                    else:
                        lines.append(
                            f'[{label}][升级中][{self._time_text(finish)}]'
                        )
                    continue
                if state == 'no_priority':
                    level = self.LEVEL_CN.get(
                        int(item.get('level', 0)), str(item.get('level', 0))
                    )
                    label = (
                        f'{level}级{facility}{item.get("index", 0)}/'
                        f'{item.get("total", 0)}'
                    )
                    lines.append(f'[{label}][未选择生产挡位]')
                    continue
                if state == 'threshold':
                    level = self.LEVEL_CN.get(
                        int(item.get('level', 0)), str(item.get('level', 0))
                    )
                    label = (
                        f'{level}级{facility}{item.get("index", 0)}/'
                        f'{item.get("total", 0)}'
                    )
                    lines.append(f'[{label}][库存达到97%][已暂停]')
                    continue
                if state == 'unknown':
                    level = self.LEVEL_CN.get(
                        int(item.get('level', 0)), str(item.get('level', 0))
                    )
                    label = (
                        f'{level}级{facility}{item.get("index", 0)}/'
                        f'{item.get("total", 0)}'
                    )
                    lines.append(f'[{label}][状态识别失败][等待复查]')
                    continue
                if state == 'no_workers':
                    level = self.LEVEL_CN.get(
                        int(item.get('level', 0)), str(item.get('level', 0))
                    )
                    label = (
                        f'{level}级{facility}{item.get("index", 0)}/'
                        f'{item.get("total", 0)}'
                    )
                    lines.append(f'[{label}][人物不足][等待复查]')
                    continue
                if state not in ('working', 'overflow'):
                    continue
                level = self.LEVEL_CN.get(int(item.get('level', 0)), str(item.get('level', 0)))
                label = f'{level}级{facility}{item.get("index", 0)}/{item.get("total", 0)}'
                if finish is None:
                    detail = '生产中][倒计时识别失败' if state == 'working' else '溢出][等待收获'
                    lines.append(f'[{label}][{detail}][等待复查]')
                    continue
                reason = '生产' if state == 'working' else '溢出'
                amount = int(item.get('amount', 0))
                amount_text = f'+{amount}' if amount > 0 else '产量未知'
                lines.append(
                    f'[{label}][{amount_text}]'
                    f'[{self._time_text(finish)}][{reason}]'
                )
        now = datetime.now().replace(microsecond=0)
        paused = not bool(self.config.cross_get(
            keys=self._production_key(city_key, 'Enable'), default=True
        )) if has_overflow else False
        next_run = production_next_run(
            data.get('buildings', {}), now, self.RETRY, fallback, paused=paused,
        )
        self.config.modified[self._production_key(city_key, 'Status')] = (
            '\n'.join(lines) if lines else '无'
        )
        self.config.modified[self._production_key(city_key, 'NextRun')] = next_run
        return next_run

    def sync_ui(self, fallback=None, update=True):
        fallback = fallback or (
            datetime.now().replace(microsecond=0) + self.NO_FARM_RETRY
        )
        times = [
            self._sync_city_ui(city_key, city, fallback)
            for city_key, city in getattr(self, 'PRODUCTION_CITIES', CITY_KEY_TO_NAME).items()
            if bool(self.config.cross_get(
                keys=self._production_key(city_key, 'Enable'), default=True
            ))
        ]
        if update:
            self.config.update()
        return times

    def run(self):
        logger.hr(f'JIANGNAN PRODUCTION: {self.facility}', level=0)
        inventory_state = JiangnanInventoryState(self.config)
        now = datetime.now().replace(microsecond=0)
        inventory_state.apply_due_arrivals(now)
        future_times = []
        for city_key, city in getattr(self, 'PRODUCTION_CITIES', CITY_KEY_TO_NAME).items():
            now = datetime.now().replace(microsecond=0)
            if production_should_yield_to_dock(inventory_state):
                logger.info(
                    f'Yield {self.facility} to arrived dock shipment'
                )
                inventory_state.sync_dock_schedule()
                break
            if not bool(self.config.cross_get(
                keys=self._production_key(city_key, 'Enable'), default=True
            )):
                continue
            next_run = self._parse_time(
                self.config.cross_get(
                    keys=self._production_key(city_key, 'NextRun'),
                    default=self.DEFAULT_TIME,
                )
            ) or self.DEFAULT_TIME
            if next_run > now:
                future_times.append(next_run)
                continue
            buildings = (self.data.get('cities', {}).get(city, {})
                         .get('facilities', {}).get(self.facility, {}).get('buildings', {}))
            resources = getattr(self, 'products', None) or (self.resource,)
            discover = getattr(self, '_requires_group_discovery', lambda city: False)(city)
            if not discover and cached_production_group(buildings, inventory_state, city, resources, now) is not None:
                next_run = self._sync_city_ui(city_key, city, now + self.NO_FARM_RETRY)
            elif not self.switch_city(city):
                next_run = datetime.now().replace(microsecond=0) + self.RETRY
            else:
                self.handle_building_map_interference()
                complete = self.inspect_city(city_key, city, inventory_state)
                next_run = self._sync_city_ui(
                    city_key, city,
                    datetime.now().replace(microsecond=0)
                    + (self.NO_FARM_RETRY if complete else self.RETRY),
                )
            self.config.modified[self._production_key(city_key, 'NextRun')] = next_run
            self.config.update()
            if bool(self.config.cross_get(
                keys=self._production_key(city_key, 'Enable'), default=True
            )):
                future_times.append(next_run)
            else:
                logger.info(
                    f'{self.facility} paused in {city}; '
                    'exclude it from task countdown'
                )
            logger.attr(f'ProductionNextRun_{city}', next_run)
        future_times.extend(self.sync_ui(update=True))
        if future_times:
            self.config.task_delay(target=min(future_times))
        else:
            self.config.task_delay(success=True)
        return True
