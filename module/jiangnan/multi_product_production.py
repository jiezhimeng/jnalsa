from datetime import datetime, timedelta
from module.jiangnan.production_state import (
    cached_production_group, production_record_pending, production_should_yield_to_dock,
)

import cv2

from module.base.button import Button
from module.jiangnan.allocation_planner import SPECIAL_GROUPS
from module.jiangnan.inventory_state import JiangnanInventoryState
from module.jiangnan.production import JiangnanProduction
from module.jiangnan.resource import SHAOXING_MERCHANT_RESOURCES
from module.jiangnan.ocr import Ocr
from module.logger import logger


class JiangnanMultiProductProduction(JiangnanProduction):
    """管理一个生产面板内可以选择多种产物的建筑。"""

    TASK_FACILITIES = {
        'JiangnanSpecialtyShopProduction': '特产铺',
        'JiangnanCarvingWorkshopProduction': '雕刻行',
        'JiangnanImprovedFarmProduction': '改良农田',
        'JiangnanPrintingFactoryProduction': '印刷厂',
        'JiangnanAmberWineryProduction': '琥珀酒坊',
        'JiangnanGourmetRestaurantProduction': '珍馐食肆',
        'JiangnanExquisiteGoodsShopProduction': '琳琅商铺',
        'JiangnanEmbroideryAccessoriesWorkshopProduction': '锦绣饰坊',
        'JiangnanLuoShoesProduction': '绫罗鞋行',
        'JiangnanJadeCrownHatWorkshopProduction': '玉冠帽坊',
        'JiangnanZijinClothingPavilionProduction': '子衿衣阁',
    }
    FACILITY_PRODUCTS = {
        '特产铺': ('茶油', '竹编', '甲酒', '盆景'),
        '雕刻行': ('木雕', '竹雕', '砖雕', '石雕'),
        '改良农田': ('蔬菜', '黑核棉'),
        '印刷厂': ('年历', '经书'),
        '琥珀酒坊': ('白醪酒', '元红酒', '加饭酒', '善酿酒'),
        '珍馐食肆': ('梅干菜', '茴香豆', '腐乳', '酱油'),
        '琳琅商铺': ('油灯', '磨喝乐', '胭脂', '铁锅'),
        '锦绣饰坊': ('香囊', '玉佩', '发簪', '璎珞'),
        '绫罗鞋行': ('木板鞋', '翘头履', '马靴', '羊毛靴'),
        '玉冠帽坊': ('斗笠', '幞头', '云巾', '䯼髻'),
        '子衿衣阁': ('深衣', '胡服', '戏服', '礼服'),
    }
    # 库存相对缺口相同时，按百科“率”降序选择；率相同则生产时间长的优先。
    # 未列出的设施继续使用界面从左到右的稳定顺序。
    PRODUCT_RATE_PRIORITY = {
        '特产铺': ('茶油', '甲酒', '竹编', '盆景'),
        '雕刻行': ('木雕', '竹雕', '砖雕', '石雕'),
    }
    PRODUCT_INPUT_RESOURCES = {
        '特产铺': {
            '茶油': ('茶叶', '瓷器'),
            '竹编': ('竹子', '生漆'),
            '甲酒': ('菜肴', '酒曲'),
            '盆景': ('原木', '越瓷'),
        },
        '雕刻行': {
            '木雕': ('木板', '草药'),
            '竹雕': ('竹子', '菜肴'),
            '砖雕': ('黏土', '盐'),
            '石雕': ('歙石', '丹丸'),
        },
        '改良农田': {
            '蔬菜': (),
            '黑核棉': ('菜肴',),
        },
        '印刷厂': {
            '年历': ('纸', '油墨'),
            '经书': ('纸', '油墨'),
        },
        '琥珀酒坊': {
            '白醪酒': ('菜肴', '酒曲'),
            '元红酒': ('菜肴', '酒曲'),
            '加饭酒': ('菜肴', '酒曲'),
            '善酿酒': ('酒曲', '元红酒'),
        },
        '珍馐食肆': {
            '梅干菜': ('蔬菜', '盐'),
            '茴香豆': ('菜肴', '盐'),
            '腐乳': ('菜肴', '盐'),
            '酱油': ('菜肴', '盐'),
        },
        '琳琅商铺': {
            '油灯': ('棉花', '越瓷'),
            '磨喝乐': ('刺绣', '黏土'),
            '胭脂': ('香粉', '螺钿'),
            '铁锅': ('木炭', '铁板'),
        },
        '锦绣饰坊': {
            '香囊': ('香粉', '丝绸'),
            '玉佩': ('玉器', '棉花'),
            '发簪': ('玉石', '名贵珍珠'),
            '璎珞': ('玉器', '名贵珍珠'),
        },
        '绫罗鞋行': {
            '木板鞋': ('木板', '竹子'),
            '翘头履': ('丝绸', '黑核棉'),
            '马靴': ('黑核棉', '标布'),
            '羊毛靴': ('标布', '羊毛'),
        },
        '玉冠帽坊': {
            '斗笠': ('竹子', '油纸'),
            '幞头': ('标布', '羊毛'),
            '云巾': ('云锦', '刺绣'),
            '䯼髻': ('羊毛', '铁矿'),
        },
        '子衿衣阁': {
            '深衣': ('布', '黑核棉'),
            '胡服': ('丝绸', '刺绣'),
            '戏服': ('刺绣', '标布'),
            '礼服': ('刺绣', '云锦'),
        },
    }
    PRODUCT_SLOTS = {
        '特产铺': {'茶油': 1, '竹编': 2, '甲酒': 3, '盆景': 4},
        '雕刻行': {'木雕': 1, '竹雕': 2, '砖雕': 3, '石雕': 4},
        '琥珀酒坊': {'白醪酒': 1, '元红酒': 2, '加饭酒': 3, '善酿酒': 4},
        '珍馐食肆': {'梅干菜': 1, '茴香豆': 2, '腐乳': 3, '酱油': 4},
        '琳琅商铺': {'油灯': 1, '磨喝乐': 2, '胭脂': 3, '铁锅': 4},
        '锦绣饰坊': {'香囊': 1, '玉佩': 2, '发簪': 3, '璎珞': 4},
        '绫罗鞋行': {'木板鞋': 1, '翘头履': 2, '马靴': 3, '羊毛靴': 4},
        '玉冠帽坊': {'斗笠': 1, '幞头': 2, '云巾': 3, '䯼髻': 4},
        '子衿衣阁': {'深衣': 1, '胡服': 2, '戏服': 3, '礼服': 4},
    }
    PRODUCT_SWITCH_GROUPS = {
        '琥珀酒坊': 'amber_winery_products',
        '珍馐食肆': 'gourmet_restaurant_products',
        '琳琅商铺': 'exquisite_goods_products',
        '锦绣饰坊': 'embroidery_accessories_products',
        '绫罗鞋行': 'luo_shoes_products',
        '玉冠帽坊': 'jade_crown_products',
        '子衿衣阁': 'zijin_clothing_products',
    }
    GOLD_SUFFIX_FACILITIES = {'特产铺', '雕刻行'}
    PRODUCT_COST_PREFIXES = {
        '改良农田': 2,
        '琥珀酒坊': 2, '珍馐食肆': 2, '琳琅商铺': 2,
        '锦绣饰坊': 2, '玉冠帽坊': 2,
        '绫罗鞋行': 2,
        '子衿衣阁': 2,
    }
    PRODUCT_COST_SUFFIXES = {
        ('印刷厂', '年历'): 1,
        ('印刷厂', '经书'): 1,
    }
    # 名贵珍珠显示在生产面板的材料行，但不属于城市或商行仓库。
    # 仍识别它以确认材料列没有错位，仓库台账只扣除其余材料。
    EXTERNAL_COST_RESOURCES = {'名贵珍珠'}
    PAGED_PRODUCTS = {
        '改良农田': {
            '蔬菜': (('水稻', '蔬菜'), 'ImprovedFarmVegetablePriority'),
            # 游戏内名称为“黑棉桃”。保留“黑核棉”作为库存台账的兼容
            # 键，同时识别新旧文字，避免既有配置和历史库存失效。
            '黑核棉': (('黑棉桃', '黑核棉', '黑胡桃'), 'ImprovedFarmCottonPriority'),
        },
        '印刷厂': {
            '年历': (('年历',), 'PrintingCalendarPriority'),
            '经书': (('经书',), 'PrintingScripturePriority'),
        },
    }

    def __init__(self, config, device=None, task=None, path=None,
                 facility=None, task_name=None):
        super().__init__(
            config=config,
            device=device,
            task=task,
            path=path,
            facility=facility,
            task_name=task_name,
        )
        self.products = self.FACILITY_PRODUCTS[self.facility]
        self._active_product = None
        # The recognition model reads one text line per crop. Feeding the
        # entire two-column panel to it returned an empty string on visible
        # 水稻/黑核棉 options, triggering needless boundary swipes.
        title_areas = [
            (430, y, 655, min(y + 54, 572))
            for y in (55, 130, 205, 280, 355, 430, 505)
        ]
        self.product_panel_ocr = Ocr(
            [Button(area=area, color=(), button=area,
                    name=f'MULTI_PRODUCT_TITLE_{index}')
             for index, area in enumerate(title_areas)],
            lang='cnocr', name='MULTI_PRODUCT_PANEL_OCR',
        )
        # 改良农田滚到底后，黑棉桃的首行停在面板中部，不能再按
        # 普通生产面板的顶端坐标识别和点击。
        cotton_areas = ((430, 165, 655, 225), (810, 165, 1055, 225))
        self.improved_farm_cotton_ocr = Ocr(
            [Button(area=area, color=(), button=area,
                    name=f'IMPROVED_FARM_COTTON_{index}')
             for index, area in enumerate(cotton_areas)],
            lang='cnocr', name='IMPROVED_FARM_COTTON_OCR',
        )
        self._paged_base_areas = {
            name: dict(getattr(self, name))
            for name in ('OPTION_AMOUNT_AREAS', 'OPTION_TIME_AREAS',
                         'OPTION_COST_AREAS', 'LEVEL_BUTTONS')
        }

    def _set_paged_option_shift(self, shift):
        """Keep option reading and clicks aligned with the recognized row."""
        for name, original in self._paged_base_areas.items():
            setattr(self, name, {
                level: (x1, y1 + shift, x2, y2 + shift)
                for level, (x1, y1, x2, y2) in original.items()
            })

    @staticmethod
    def _special_group(resource):
        return next(
            (name for name, members in SPECIAL_GROUPS.items()
             if resource in members),
            None,
        )

    def _product_upper_limits(self, city, inventory_state):
        """所有多产物设施统一以仓库容量的 97% 作为生产上限。"""
        capacities = inventory_state.load_capacities((city,))
        special_capacities = inventory_state.special_capacities()
        capacity = int(capacities.get(city, 0))
        limits = {}
        for resource in self.products:
            group = self._special_group(resource)
            if group:
                group_capacity = int(
                    special_capacities.get(city, {}).get(group, 0)
                )
                upper = int(group_capacity * 0.97) // len(SPECIAL_GROUPS[group])
            elif city == '绍兴府' and resource in SHAOXING_MERCHANT_RESOURCES:
                merchant_capacity = int(
                    special_capacities.get(city, {}).get('绍兴商行', 0)
                )
                upper = int(merchant_capacity * 0.97)
            else:
                upper = int(capacity * 0.97)
            limits[resource] = max(0, int(upper))
        return limits

    def _product_enabled(self, resource):
        group = self.PRODUCT_SWITCH_GROUPS.get(self.facility)
        if group is None:
            return True
        slot = self.PRODUCT_SLOTS[self.facility][resource]
        config = getattr(self, 'config', None)
        if config is None:
            return True
        return bool(config.cross_get(
            keys=f'{self.TASK}.{group}.Product{slot}', default=True,
        ))

    def _product_candidates(self, city, inventory_state, building_level):
        inventories = inventory_state.load_inventories((city,))
        projected = inventory_state.project_for_planning(inventories)[city]
        limits = self._product_upper_limits(city, inventory_state)
        if self.facility in self.PAGED_PRODUCTS:
            available = list(self.products)
        else:
            available = list(self.products[:min(max(int(building_level), 1), 4)])
        available = [resource for resource in available
                     if self._product_enabled(resource)]
        self._all_products_disabled = not available
        rate_priority = self.PRODUCT_RATE_PRIORITY.get(
            self.facility, self.products
        )
        threshold_states = inventory_state.production_threshold_states(
            city, available
        )
        eligible = [
            resource for resource in available
            if threshold_states.get(resource) is True
        ]
        ranked = sorted(
            eligible,
            key=lambda resource: (
                int(projected.get(resource, 0)) / max(1, limits.get(resource, 0)),
                rate_priority.index(resource),
            ),
        )
        all_saturated = bool(available) and all(
            threshold_states.get(resource) is False
            for resource in available
        )
        logger.attr(
            'MultiProductStock',
            ', '.join(
                f'{resource}={int(projected.get(resource, 0))}/'
                f'{limits.get(resource, 0)}:'
                f'{threshold_states.get(resource)}'
                for resource in available
            ),
        )
        return ranked, all_saturated

    def _show_paged_product(self, resource):
        """Search a paged product by scrolling and re-OCRing each landing frame."""
        self._set_paged_option_shift(0)
        tokens, _ = self.PAGED_PRODUCTS[self.facility][resource]
        target_index = self.products.index(resource)
        for attempt in range(6):
            self.device.screenshot()
            raw = self.product_panel_ocr.ocr(self.device.image)
            values = raw if isinstance(raw, list) else [raw]
            text = ''.join(str(value).replace(' ', '') for value in values)
            logger.attr(
                'PagedProductPanel',
                f'{self.facility}:{resource}:attempt={attempt + 1}:{text}',
            )
            # The option amount/time/button crops below assume the selected
            # product starts in the top row. Seeing only its first title at
            # the bottom edge (as with 黑核棉) does not make slot 4 visible.
            top_titles = ''.join(
                str(value).replace(' ', '') for value in values[:2]
            )
            if any(token in top_titles for token in tokens):
                if self.facility == '改良农田':
                    self._set_paged_option_shift(
                        -40 if resource == '蔬菜' else 70
                    )
                logger.info(
                    f'Paged product aligned at top: {self.facility} {resource}'
                )
                return True
            if self.facility == '改良农田' and resource == '黑核棉':
                cotton_raw = self.improved_farm_cotton_ocr.ocr(
                    self.device.image
                )
                cotton_titles = (
                    cotton_raw if isinstance(cotton_raw, list)
                    else [cotton_raw]
                )
                if len(cotton_titles) >= 2 and all(
                    any(token in str(title).replace(' ', '')
                        for token in tokens)
                    for title in cotton_titles[:2]
                ):
                    self._set_paged_option_shift(70)
                    logger.info(
                        'Paged product aligned at middle: 改良农田 黑核棉'
                    )
                    return True
            if attempt >= 5:
                break

            # Improved Farm opens on vegetables. Direction only pushes content;
            # it never proves which page was reached. Re-OCR after every swipe
            # and stop only after the target is seen or the panel hits a boundary.
            if target_index > 0:
                p1, p2 = (900, 500), (900, 190)
                direction = 'down'
            else:
                p1, p2 = (900, 190), (900, 500)
                direction = 'up'
            x1, y1, x2, y2 = (320, 85, 1090, 525)
            before = self.device.image[y1:y2, x1:x2].copy()
            logger.info(
                f'Scroll {self.facility} panel to find {resource}: '
                f'direction={direction}, attempt={attempt + 1}'
            )
            self.jiangnan_drag(
                p1, p2,
                name=f'PAGED_PRODUCT_SCROLL_{attempt + 1}',
                hold_duration=0.2,
            )
            self.device.sleep(0.7)
            self.device.screenshot()
            after = self.device.image[y1:y2, x1:x2]
            change = float(cv2.absdiff(before, after).mean())
            logger.attr('PagedProductScrollChange', f'{change:.2f}')
            if change < 0.8:
                logger.warning(
                    f'Paged product reached scroll boundary before finding '
                    f'{resource}'
                )
                break
        logger.warning(f'{self.facility} {resource} options were not recognized')
        return False

    def _refresh_multi_product_inventory(
        self, city, inventory_state, building_level=None, extra_products=(),
    ):
        """Refresh live output stock before choosing among multiple products."""
        products = [resource for resource in self.products
                    if self._product_enabled(resource)]
        if building_level is not None and self.facility not in self.PAGED_PRODUCTS:
            highest_slot = min(max(int(building_level), 1), 4)
            products = [resource for resource in products
                        if self.PRODUCT_SLOTS[self.facility][resource]
                        <= highest_slot]
        products.extend(resource for resource in extra_products
                        if resource in self.products and resource not in products)
        if not products:
            return True
        if self.facility in self.PAGED_PRODUCTS:
            verified = set()
            for resource in products:
                if not self._show_paged_product(resource):
                    continue
                targets = (resource,) + tuple(
                    self.PRODUCT_INPUT_RESOURCES[self.facility].get(resource, ())
                )
                self._refresh_live_inventory_from_production_panel(
                    city,
                    inventory_state,
                    targets,
                    required_resources=(resource,),
                )
                verified.update(
                    getattr(self, '_last_production_warehouse_readings', {})
                )
            # A later warehouse view can show both products even if the
            # earlier view failed to scroll back to one of them.
            return all(resource in verified for resource in products)

        targets = []
        for resource in products:
            targets.append(resource)
            targets.extend(
                self.PRODUCT_INPUT_RESOURCES[self.facility].get(resource, ())
            )
        return self._refresh_live_inventory_from_production_panel(
            city,
            inventory_state,
            tuple(dict.fromkeys(targets)),
            required_resources=products,
        )

    def _product_slot(self, resource, building_level, city=None):
        if self.facility in self.PAGED_PRODUCTS:
            field_prefix = 'Priority'
            if self.facility == '改良农田':
                field_prefix = (
                    'VegetablePriority'
                    if resource == '蔬菜'
                    else 'CottonPriority'
                )
            priority = self._production_priority(
                city=city, field_prefix=field_prefix
            )
            available_max = min(max(int(building_level), 1), 4)
            for option in priority:
                selected = self.OPTION_LEVEL[option]
                if selected <= available_max:
                    logger.info(
                        f'{self.facility} {resource} priority selected: '
                        f'{self.OPTION_NAME[option]} (slot={selected})'
                    )
                    return selected
            return None
        return self.PRODUCT_SLOTS[self.facility][resource]

    def _read_product_costs(self, resource, selected):
        resources = self.PRODUCT_INPUT_RESOURCES[self.facility][resource]
        if not resources:
            return {}
        readings = []
        for attempt in range(1, 4):
            values = self._read_cost_values(selected)
            prefix = self.PRODUCT_COST_PREFIXES.get(self.facility, 1)
            suffix = self.PRODUCT_COST_SUFFIXES.get(
                (self.facility, resource),
                1 if self.facility in self.GOLD_SUFFIX_FACILITIES else 0,
            )
            if self.facility == '改良农田' and resource == '黑核棉':
                # The green coin icon can be split into an extra numeric
                # component. The dish count is always the rightmost number
                # on this option row (after worker count and coin cost).
                material_values = values[-1:] if len(values) >= 3 else []
            else:
                material_values = values[
                    prefix:len(values) - suffix if suffix else None
                ]
            valid = (
                len(material_values) == len(resources)
                and all(0 < value <= 100000 for value in material_values)
            )
            logger.attr(
                f'MultiProductCost{selected}Try{attempt}',
                f'all={values}, materials={material_values}, valid={valid}',
            )
            if valid:
                costs = dict(zip(resources, material_values))
                readings.append(costs)
                if len(readings) >= 2 and readings[-1] == readings[-2]:
                    external = {
                        name: value for name, value in costs.items()
                        if name in self.EXTERNAL_COST_RESOURCES
                    }
                    if external:
                        logger.attr('MultiProductExternalCosts', external)
                    return {
                        name: value for name, value in costs.items()
                        if name not in self.EXTERNAL_COST_RESOURCES
                    }
            if attempt < 3:
                self.device.sleep(0.25)
                self.device.screenshot()
        return None

    def _start_product(
        self, candidates, building_level, city, inventory_state,
        building_index=1,
    ):
        self.last_priority_empty = False
        if getattr(self, 'defer_new_production', False):
            self.last_no_workers = True
            self._close_panel()
            return None
        had_selectable_priority = False
        for resource in candidates:
            selected = self._product_slot(
                resource, building_level, city=city
            )
            if selected is None:
                logger.info(
                    f'Skip {self.facility} {resource}: no production priority'
                )
                continue
            had_selectable_priority = True
            if self.facility in self.PAGED_PRODUCTS:
                if not self._show_paged_product(resource):
                    continue
            amount, duration = self._read_option(selected)
            if amount is None or duration is None:
                continue
            costs = self._read_product_costs(resource, selected)
            if costs is None:
                continue
            enough, missing = inventory_state.can_consume_inventory(city, costs)
            if not enough:
                logger.warning(
                    f'Skip {self.facility} {resource}: insufficient {missing}'
                )
                continue
            self._active_product = resource
            logger.info(
                f'Start {self.facility}: {resource}, slot={selected}, '
                f'amount={amount}, duration={duration}, costs={costs}'
            )
            self.device.click(Button(
                area=(), color=(), button=self.LEVEL_BUTTONS[selected],
                name=f'MULTI_PRODUCT_{selected}_DYNAMIC',
            ))
            if not self._direct_start(
                city=city, expected_resources=costs
            ):
                if self.last_insufficient_inventory:
                    reconciled = inventory_state.reconcile_live_inventory(
                        city,
                        self.last_insufficient_inventory,
                        reason=(
                            f'{self.facility}{resource}开工资源不足提示'
                        ),
                    )
                    if not reconciled:
                        # The popup was handled, but there is no trustworthy
                        # stock number.  Stop this facility instead of trying
                        # another recipe against the same stale inventory.
                        return None
                    # 库存已经以游戏实值更正；重新评估下一候选产物，
                    # 不要在同一档位重复点击，也绝不能点击立即补齐。
                    continue
                return None
            duration = self._actual_started_duration(
                duration, building_index=building_index,
            )
            if not inventory_state.consume_inventory(city, costs):
                inventory_state.request_scan(
                    f'{city}{self.facility}{resource}已开工但成本扣除失败',
                    cities={city},
                )
            return resource, amount, duration, costs
        self.last_priority_empty = (
            bool(candidates) or getattr(self, '_all_products_disabled', False)
        ) and not had_selectable_priority
        return None

    def inspect_facility(
        self, city_key, city, facility, resource, inventory_state,
        prelocated_result=None,
    ):
        cached = cached_production_group(
            getattr(self, 'data', {}).get('cities', {}).get(city, {})
            .get('facilities', {}).get(facility, {}).get('buildings', {}),
            inventory_state, city, self.products,
        )
        if cached is not None:
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

        previous = (
            self.data.get('cities', {}).get(city, {})
            .get('facilities', {}).get(facility, {}).get('buildings', {})
        )
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
            if production_record_pending(old):
                logger.info(f'Skip working {facility} {index}/{total} before {old.get("finish_at")}')
                continue
            if old.get('state') == 'threshold':
                states = inventory_state.production_threshold_states(city, self.products)
                if states and all(states.get(product) is False for product in self.products):
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
                    break
                if not self._select_production_instance(facility, index):
                    break

            old = previous.get(str(index), {})
            known_resource = old.get('resource')
            resource = known_resource or '未知产物'
            amount = int(old.get('amount', 0) or 0)
            inspected = True
            state, duration = self._open_current_production()
            if state is None and getattr(self, '_foundation_icon_occluded', False):
                state = 'occluded'
            state, duration = self._confirm_live_production_state(
                old, state, duration, f'{city} {facility} {index}/{total}'
            )
            if old.get('state') == 'working' and state == 'upgrading':
                inventory_state.cancel_inventory_change(
                    self._reference(city, facility, level, index)
                )
            if state == 'working' and (
                old.get('state') != 'working'
                or self._last_production_record_drift
            ):
                # 外部手动开工时当前产品可能不同于上轮，面板倒计时
                # 无法反推货物种类与数量，不能沿用旧产物入账。
                known_resource = None
                resource = '未知产物'
                amount = 0
                inventory_state.cancel_inventory_change(
                    self._reference(city, facility, level, index)
                )
            finish_at = None
            costs = {}
            reconciliation = None
            candidates = []
            all_saturated = False

            if state == 'overflow':
                reference = self._reference(city, facility, level, index)
                inventory_state.cancel_inventory_change(
                    reference, rollback_credited=True
                )
                if self.resolve_warehouse_full(force=True):
                    logger.info(
                        f'Force-harvest {facility} completed overflow; '
                        f'recheck 97% threshold before producing'
                    )
                    state, duration = self._open_current_production()
                else:
                    finish_at = now
                    resource = known_resource or '未知产物'

            if state == 'idle':
                use_cached = inventory_refreshed and old.get('state') not in (
                    'working', 'overflow'
                )
                if use_cached:
                    logger.info(f'Reuse facility inventory scan: {city} {facility}')
                refreshed = use_cached or self._refresh_multi_product_inventory(
                    city, inventory_state, building_level=level,
                    extra_products=(known_resource,) if old.get('state') in (
                        'working', 'overflow',
                    ) else (),
                )
                inventory_refreshed = inventory_refreshed or refreshed
                if refreshed and old.get('state') in ('working', 'overflow'):
                    reconciliation = inventory_state.reconcile_production_harvest(
                        city=city,
                        resource=old.get('resource') or resource,
                        amount=old.get('amount') or amount,
                        reference=self._reference(
                            city, facility,
                            int(old.get('level') or level), index
                        ),
                        expected_at=old.get('finish_at'),
                        observed_at=datetime.now().replace(microsecond=0),
                        included_in_scan=True,
                    )
                if not refreshed:
                    logger.warning(
                        f'Unable to refresh all live product inventory before '
                        f'{self.facility} production: {city}'
                    )
                    state = 'unknown'
                    resource = known_resource or '未知产物'
                    self._close_panel()
                    all_saturated = False
                else:
                    candidates, all_saturated = self._product_candidates(
                        city, inventory_state, level
                    )
                    if all_saturated:
                        inventory_state.pause_multi_product_production(
                            city,
                            self.products,
                            self.TASK,
                            reason='库存达到97%',
                        )
                        state = 'threshold'
                        resource = known_resource or '已达97%阈值'
                        self._close_panel()
                    elif workers_exhausted:
                        state = 'no_workers'
                        resource = known_resource or '未知产物'
                        self._close_panel()
                    else:
                        started = self._start_product(
                            candidates, level, city, inventory_state,
                            building_index=index,
                        )
                        if started:
                            resource, amount, duration, costs = started
                            state = 'working'
                            finish_at = datetime.now().replace(microsecond=0) + duration
                        elif self.last_priority_empty:
                            state = 'no_priority'
                            resource = known_resource or '未选择产物'
                            self._close_panel()
                        elif self.last_no_workers:
                            state = 'no_workers'
                            resource = known_resource or '未知产物'
                            workers_exhausted = True
                        else:
                            state = 'unknown'
                            resource = known_resource or '未知产物'
                            self._close_panel()
            elif state == 'working':
                resource = known_resource or '未知产物'
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
            elif state == 'upgrading':
                resource = known_resource or '升级中'
                if duration is not None and duration > timedelta(0):
                    finish_at = datetime.now().replace(microsecond=0) + duration
                self._close_panel()
            elif state == 'occluded':
                finish_at = datetime.now().replace(microsecond=0) + timedelta(hours=12)

            if (
                state == 'working' and finish_at is not None
                and resource in self.products and amount > 0
            ):
                inventory_state.schedule_inventory_change(
                    city, resource, amount, finish_at,
                    reason='生产',
                    reference=self._reference(city, facility, level, index),
                )
            buildings[str(index)] = {
                'level': level, 'index': index, 'total': total,
                'state': state or 'unknown', 'resource': resource,
                'amount': int(amount or 0), 'costs': dict(costs or {}),
                'finish_at': self._time_text(finish_at) if finish_at else None,
                'all_products_saturated': bool(all_saturated),
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
                    'all products reached the production threshold'
                )
                break
        return {'unavailable': False, 'buildings': buildings}

    def _sync_city_ui(self, city_key, city, fallback):
        result = super()._sync_city_ui(city_key, city, fallback)
        key = self._production_key(city_key, 'Status')
        text = self.config.modified.get(key, '无')
        data = (
            self.data.get('cities', {}).get(city, {})
            .get('facilities', {}).get(self.facility, {})
        )
        products = {
            int(item.get('index', 0)): item.get('resource', '未知产物')
            for item in data.get('buildings', {}).values()
        }
        for index, product in products.items():
            marker = f'{self.facility}{index}/'
            text = text.replace(marker, f'{self.facility}{index}·{product}/')
        self.config.modified[key] = text
        return result
