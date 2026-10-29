import re
from collections import namedtuple
from datetime import datetime, timedelta

from module.jiangnan.allocation import CITY_KEY_TO_NAME, JiangnanAllocation
from module.jiangnan.resource import (
    AUTOMATIC_EXPORT_RESOURCES,
    WAREHOUSE_OCR_ALIASES,
    WAREHOUSE_RESOURCES,
    WAREHOUSE_SPECIAL_RESOURCES,
)
from module.jiangnan.resource_layout import allocation_layout_page
from module.jiangnan.warehouse import JiangnanWarehouse
from module.jiangnan.transport import JiangnanTransportStatus
from module.jiangnan.inventory_state import (
    JiangnanInventoryState, resources_for_city,
)
from module.logger import logger


AllocationOrder = namedtuple(
    'AllocationOrder',
    ['source', 'destination', 'resource', 'amount'],
)

KNOWN_RESOURCES = set(WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES)
SPECIAL_GROUPS = {
    '徽州特产1': ('茶油', '竹编', '甲酒', '盆景'),
    '徽州特产2': ('木雕', '竹雕', '砖雕', '石雕'),
}
SPECIAL_GROUP_KEYS = {
    name: f'__special_group__:{name}' for name in SPECIAL_GROUPS
}
ALLOCATION_STRATEGY_NONE = 'none'
ALLOCATION_STRATEGY_RECEIVE_FIRST = 'receive_first'
ALLOCATION_STRATEGY_SEND_FIRST = 'send_first'
ALLOCATION_STRATEGY_ALIASES = {
    ALLOCATION_STRATEGY_NONE: ALLOCATION_STRATEGY_NONE,
    ALLOCATION_STRATEGY_RECEIVE_FIRST: ALLOCATION_STRATEGY_RECEIVE_FIRST,
    ALLOCATION_STRATEGY_SEND_FIRST: ALLOCATION_STRATEGY_SEND_FIRST,
    '无策略': ALLOCATION_STRATEGY_NONE,
    '优先接收': ALLOCATION_STRATEGY_RECEIVE_FIRST,
    '优先送出': ALLOCATION_STRATEGY_SEND_FIRST,
}


class AllocationRuleMap(dict):
    """保留哪些物资由用户逐项指定，供渐进默认规则避让。"""

    def __init__(self, *args, explicit_resources=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_resources = set(explicit_resources or ())


class JiangnanAllocationPlanner(JiangnanWarehouse):
    """扫描各城市仓库，并根据配置的库存区间自动生成运输订单。"""

    DEFAULT_BASE_LOWER_RATIO = 0.6
    DEFAULT_BASE_UPPER_RATIO = 0.7
    DEFAULT_HIGH_LOWER_RATIO = 0.9
    DEFAULT_HIGH_UPPER_RATIO = 0.95
    DEFAULT_FILTER = '60% < 全部物资 < 70%'

    # 不由常规“生产设施”任务管理，但实际上会稳定产出对应物资的
    # 城市。鸡鸣山的菜肴来自其专属供给，不能依赖不存在的餐馆开关；
    # 否则餐馆搜索失败自动关闭后，它会被误判为菜肴非生产城市。
    SPECIAL_PRODUCER_CITIES = {
        resource: tuple(
            city for city, resources in AUTOMATIC_EXPORT_RESOURCES.items()
            if resource in resources
        )
        for resource in {
            item for resources in AUTOMATIC_EXPORT_RESOURCES.values()
            for item in resources
        }
    }

    @staticmethod
    def missing_warehouse_baseline_cities(inventory_state, cities):
        """Find only cities whose stored stock or capacity is incomplete."""
        known_capacity = inventory_state.capacities()
        known_special = inventory_state.special_capacities()
        known_inventory = inventory_state.load_inventories(cities)
        return {
            city for city in cities
            if (city not in known_capacity
                or city not in known_special
                or any(resource not in known_inventory.get(city, {})
                       for resource in resources_for_city(city)))
        }

    def delay_retry(self, cities, minutes=None):
        """同时推迟逐城时钟和页面总时钟，避免加载配置后立刻重跑。"""
        if minutes is None:
            try:
                minutes = int(self.config.Scheduler_FailureInterval)
            except (TypeError, ValueError):
                minutes = 10
        target = (
            datetime.now().replace(microsecond=0)
            + timedelta(minutes=max(int(minutes), 1))
        )
        name_to_key = {
            name: key for key, name in CITY_KEY_TO_NAME.items()
        }
        for city in set(cities or ()):
            city_key = name_to_key.get(city)
            if city_key is not None:
                self.config.modified[
                    f'JiangnanCityAllocation.{city_key}.NextRun'
                ] = target
        self.config.task_delay(target=target)
        return target

    RANGE_PATTERN = re.compile(
        r'^\s*(\d+)\s*(?:<=|<|≤)\s*([^\d<≤]+?)\s*'
        r'(?:<=|<|≤)\s*(\d+)\s*$'
    )
    SIMPLE_RANGE_PATTERN = re.compile(
        r'^\s*([^:：]+?)\s*[:：]\s*(\d+)\s*(?:-|~|～|至)\s*(\d+)\s*$'
    )
    PERCENT_RANGE_PATTERN = re.compile(
        r'^\s*(\d+(?:\.\d+)?)\s*%\s*(?:<=|<|≤)\s*([^\d<≤]+?)\s*'
        r'(?:<=|<|≤)\s*(\d+(?:\.\d+)?)\s*%\s*$'
    )
    SIMPLE_PERCENT_RANGE_PATTERN = re.compile(
        r'^\s*([^:：]+?)\s*[:：]\s*(\d+(?:\.\d+)?)\s*%\s*'
        r'(?:-|~|～|至)\s*(\d+(?:\.\d+)?)\s*%\s*$'
    )

    @staticmethod
    def normalize_resource(resource):
        resource = re.sub(r'\s+', '', str(resource))
        return WAREHOUSE_OCR_ALIASES.get(resource, resource)

    @staticmethod
    def special_group_name(resource):
        """返回特产单品或内部共享仓库键所属的共享仓库。"""
        return next(
            (
                name for name, members in SPECIAL_GROUPS.items()
                if resource == SPECIAL_GROUP_KEYS[name] or resource in members
            ),
            None,
        )

    @classmethod
    def parse_filter(cls, text, city):
        """
        将一个城市的多行目标量转换为：
            {'蔬菜': (下限, 上限)}

        支持：
            100 < 蔬菜 < 500
            蔬菜: 100-500
        """
        # 分开保存统一规则和单物资规则，最后再合并。这样单物资显式
        # 配置始终覆盖“全部物资”，不受两行在 UI 中书写顺序影响。
        global_rules = {}
        explicit_rules = {}
        errors = []
        text = '' if text is None else str(text)
        text = text.replace('；', '\n').replace(';', '\n')

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue

            percent = False
            match = cls.PERCENT_RANGE_PATTERN.match(line)
            if match:
                lower, resource, upper = match.groups()
                percent = True
            else:
                match = cls.SIMPLE_PERCENT_RANGE_PATTERN.match(line)
                if match:
                    resource, lower, upper = match.groups()
                    percent = True
                else:
                    match = cls.RANGE_PATTERN.match(line)
                    if match:
                        lower, resource, upper = match.groups()
                    else:
                        match = cls.SIMPLE_RANGE_PATTERN.match(line)
                        if not match:
                            errors.append(
                                f'{city} 第 {line_number} 行格式错误：{raw_line}'
                            )
                            continue
                        resource, lower, upper = match.groups()

            resource = cls.normalize_resource(resource)
            if percent:
                lower = float(lower) / 100
                upper = float(upper) / 100
            else:
                lower = int(lower)
                upper = int(upper)

            all_resources = resource in ('全部', '全部物资')
            if not all_resources and resource not in KNOWN_RESOURCES:
                errors.append(
                    f'{city} 第 {line_number} 行包含未知物资：{resource}'
                )
                continue
            limit = 1 if percent else float('inf')
            if lower < 0 or upper < 0 or lower > upper or upper > limit:
                errors.append(
                    f'{city} 第 {line_number} 行库存范围无效：{lower}-{upper}'
                )
                continue

            # “全部物资”对徽州特产按四件共享仓容处理，不能展开成四个
            # 独立的 60%~70% 目标，否则目标总量会超过共享上限。
            targets = (
                WAREHOUSE_RESOURCES if all_resources else (resource,)
            )
            if all_resources and percent:
                targets = tuple(targets) + tuple(SPECIAL_GROUP_KEYS.values())
            for target in targets:
                target_rules = global_rules if all_resources else explicit_rules
                if target in target_rules and not all_resources:
                    logger.warning(
                        f'Duplicate allocation range, use last one: '
                        f'{city}, {target}'
                    )
                target_rules[target] = (lower, upper)

        rules = AllocationRuleMap(
            global_rules,
            explicit_resources=explicit_rules,
        )
        rules.update(explicit_rules)
        return rules, errors

    def load_city_rules(self):
        city_rules = {}
        errors = []
        allocation_cities = []
        export_cities = []
        import_cities = []

        for city_key, city_name in CITY_KEY_TO_NAME.items():
            allow_export = bool(self.config.cross_get(
                keys=f'JiangnanCityAllocation.{city_key}.Export',
                default=True,
            ))
            allow_import = bool(self.config.cross_get(
                keys=f'JiangnanCityAllocation.{city_key}.Import',
                default=True,
            ))
            if not allow_export and not allow_import:
                logger.info(f'Allocation city disabled: {city_name}')
                continue
            allocation_cities.append(city_name)
            if allow_export:
                export_cities.append(city_name)
            if allow_import:
                import_cities.append(city_name)

            filter_text = self.config.cross_get(
                keys=f'JiangnanCityAllocation.{city_key}.Filter',
                default=self.DEFAULT_FILTER,
            )
            if filter_text is None or not str(filter_text).strip():
                filter_text = self.DEFAULT_FILTER
                logger.info(
                    f'Use default allocation range for {city_name}: '
                    f'{self.DEFAULT_FILTER}'
                )
            rules, city_errors = self.parse_filter(
                filter_text,
                city=city_name,
            )
            errors.extend(city_errors)
            for resource in AUTOMATIC_EXPORT_RESOURCES.get(city_name, ()):
                if resource not in rules.explicit_resources:
                    # 自动产出的物资默认是本城输出，不应套用“全部物资”的
                    # 保有量目标，更不应因为低于该目标而向产地补货。
                    # 逐项配置仍可指定需要保留的库存。
                    rules[resource] = (0, 0)
                    rules.explicit_resources.add(resource)
                    logger.info(
                        f'Automatic export range: {city_name}, '
                        f'{resource}, 0-0'
                    )
            if rules:
                city_rules[city_name] = rules

        return (
            city_rules,
            errors,
            allocation_cities,
            export_cities,
            import_cities,
        )

    def load_city_strategies(self, cities=None):
        """读取逐城调配策略；未知旧值安全回退到无策略。"""
        enabled = set(cities or CITY_KEY_TO_NAME.values())
        strategies = {}
        for city_key, city_name in CITY_KEY_TO_NAME.items():
            if city_name not in enabled:
                continue
            raw = self.config.cross_get(
                keys=f'JiangnanCityAllocation.{city_key}.Strategy',
                default=ALLOCATION_STRATEGY_NONE,
            )
            strategy = ALLOCATION_STRATEGY_ALIASES.get(str(raw).strip())
            if strategy is None:
                logger.warning(
                    f'Unknown allocation strategy for {city_name}: '
                    f'{raw!r}; use no strategy'
                )
                strategy = ALLOCATION_STRATEGY_NONE
            strategies[city_name] = strategy
        logger.attr('AllocationCityStrategies', strategies)
        return strategies

    def disable_unavailable_exports(self, snapshot):
        """两个运输位均不可用时，认定码头未修复并关闭该城运出。"""
        disabled = set()
        for city, slots in snapshot.items():
            if not slots or not all(
                slot.get('state') == 'unavailable' for slot in slots
            ):
                continue
            city_key = next(
                key for key, name in CITY_KEY_TO_NAME.items() if name == city
            )
            config_key = f'JiangnanCityAllocation.{city_key}.Export'
            if self.config.cross_get(keys=config_key, default=True):
                self.config.modified[config_key] = False
                logger.warning(
                    f'Dock unavailable, disable material export: {city}'
                )
            disabled.add(city)
        if self.config.modified:
            self.config.update()
        return disabled

    @staticmethod
    def resolve_percentage_rules(city_rules, capacities, special_capacities=None):
        """将百分比库存范围换算成各城市仓库的实际数量。"""
        resolved = {}
        errors = []
        special_capacities = special_capacities or {}
        for city, rules in city_rules.items():
            city_rules_resolved = AllocationRuleMap(
                explicit_resources=getattr(rules, 'explicit_resources', ()),
            )
            for resource, (lower, upper) in rules.items():
                if isinstance(lower, float) or isinstance(upper, float):
                    group_name = JiangnanAllocationPlanner.special_group_name(
                        resource
                    )
                    capacity = (
                        special_capacities.get(city, {}).get(group_name)
                        if group_name else capacities.get(city)
                    )
                    if capacity is None:
                        errors.append(f'{city} 仓库库存上限识别失败')
                        break
                    actual_lower = int(capacity * lower)
                    actual_upper = int(capacity * upper)
                    city_rules_resolved[resource] = (
                        actual_lower,
                        actual_upper,
                    )
                else:
                    city_rules_resolved[resource] = (lower, upper)
            else:
                resolved[city] = city_rules_resolved
                if city_rules_resolved:
                    sample = next(iter(city_rules_resolved.values()))
                    logger.info(
                        f'Allocation range resolved: {city}, '
                        f'capacity={capacities.get(city)}, '
                        f'default={sample[0]}-{sample[1]}'
                    )
        return resolved, errors

    @classmethod
    def build_progressive_defaults(
        cls, city_rules, capacities, special_capacities=None,
    ):
        """标记默认 60%~70% 规则，并计算 90%~95% 溢出区间。"""
        defaults = {}
        special_capacities = special_capacities or {}
        for city, rules in city_rules.items():
            explicit_resources = getattr(rules, 'explicit_resources', set())
            for resource, bounds in rules.items():
                if resource in explicit_resources:
                    # 即使显式值刚好也是 60%~70%，它仍然是固定目标，
                    # 不能被识别成统一默认值后推进到溢出区间。
                    continue
                group_name = next(
                    (name for name, key in SPECIAL_GROUP_KEYS.items()
                     if resource == key),
                    None,
                )
                capacity = (
                    special_capacities.get(city, {}).get(group_name)
                    if group_name else capacities.get(city)
                )
                if capacity is None:
                    continue
                base_lower = int(capacity * cls.DEFAULT_BASE_LOWER_RATIO)
                base_upper = int(capacity * cls.DEFAULT_BASE_UPPER_RATIO)
                if tuple(bounds) != (base_lower, base_upper):
                    # 绝对数量和用户单独覆盖的范围保持原样。
                    continue
                high_lower = int(capacity * cls.DEFAULT_HIGH_LOWER_RATIO)
                high_upper = int(capacity * cls.DEFAULT_HIGH_UPPER_RATIO)
                if group_name:
                    for item in SPECIAL_GROUPS[group_name]:
                        # 单个特产的显式范围优先于共享仓库的统一范围。
                        # 不再为它生成普通→溢出的渐进默认目标。
                        if item in rules:
                            continue
                        defaults.setdefault(city, {})[item] = (
                            base_lower // len(SPECIAL_GROUPS[group_name]),
                            base_upper // len(SPECIAL_GROUPS[group_name]),
                            high_lower // len(SPECIAL_GROUPS[group_name]),
                            high_upper // len(SPECIAL_GROUPS[group_name]),
                        )
                else:
                    defaults.setdefault(city, {})[resource] = (
                        base_lower, base_upper, high_lower, high_upper,
                    )
        return defaults

    @staticmethod
    def progressive_targets(
        inventories, progressive_defaults, producer_cities, log_status=False,
    ):
        """按物资在普通区间、非生产城溢出区间、全城溢出区间推进。"""
        effective = {}
        resources = sorted({
            resource
            for defaults in progressive_defaults.values()
            for resource in defaults
        })
        for resource in resources:
            cities = [
                city for city, defaults in progressive_defaults.items()
                if resource in defaults and city in inventories
            ]
            if not cities:
                continue
            base_reached = all(
                int(inventories[city].get(resource, -1)) >= 0
                and int(inventories[city].get(resource, -1))
                >= progressive_defaults[city][resource][0]
                for city in cities
            )
            producers = set(producer_cities.get(resource, ()))
            non_producers = [city for city in cities if city not in producers]
            if not base_reached:
                high_cities = set()
                phase = '全部城市维持60%~70%'
            elif non_producers and not all(
                int(inventories[city].get(resource, -1)) >= 0
                and int(inventories[city].get(resource, -1))
                >= progressive_defaults[city][resource][2]
                for city in non_producers
            ):
                high_cities = set(non_producers)
                phase = '非生产城市维持90%~95%'
            else:
                high_cities = set(cities)
                phase = '全部城市维持90%~95%'
            for city in cities:
                base_lower, base_upper, high_lower, high_upper = (
                    progressive_defaults[city][resource]
                )
                target = (
                    (high_lower, high_upper)
                    if city in high_cities
                    else (base_lower, base_upper)
                )
                effective.setdefault(city, {})[resource] = target
            if log_status:
                logger.info(
                    f'Progressive allocation target: {resource}, phase={phase}, '
                    f'producers={sorted(producers.intersection(cities))}'
                )
        return effective

    def load_producer_cities(self, local_exports=None):
        """返回每种物资的生产城市。

        物资调配传入黄框辅助信息时，它就是本城实际可生产货物的唯一
        依据。生产设施开关会因 97% 阈值、满仓或用户设置临时关闭，
        不能再用来反推城市是否具备生产能力。
        """
        if local_exports is not None:
            producer_cities = {}
            for city, resources in local_exports.items():
                for resource in resources:
                    producer_cities.setdefault(resource, set()).add(city)
            for resource, cities in self.SPECIAL_PRODUCER_CITIES.items():
                producer_cities.setdefault(resource, set()).update(cities)
            logger.attr(
                'AllocationProducerCitiesSource',
                'yellow-border auxiliary layouts',
            )
            return producer_cities

        # 兼容没有黄框辅助信息的独立调用。
        # 局部导入可避免多产物模块与调配计划器之间形成导入环。
        from module.jiangnan.farm_production import JiangnanFarmProduction
        from module.jiangnan.multi_product_production import (
            JiangnanMultiProductProduction,
        )
        from module.jiangnan.production import JiangnanProduction

        task_products = {
            JiangnanFarmProduction.TASK: ('蔬菜',),
        }
        for task, facility in JiangnanProduction.TASK_FACILITIES.items():
            task_products[task] = (
                JiangnanProduction.FACILITY_RESOURCES[facility],
            )
        for task, facility in (
            JiangnanMultiProductProduction.TASK_FACILITIES.items()
        ):
            task_products[task] = (
                JiangnanMultiProductProduction.FACILITY_PRODUCTS[facility]
            )

        producer_cities = {
            resource: set(cities)
            for resource, cities in self.SPECIAL_PRODUCER_CITIES.items()
        }
        for task, products in task_products.items():
            if not self.config.cross_get(
                keys=f'{task}.Scheduler.Enable', default=False,
            ):
                continue
            for city_key, city in CITY_KEY_TO_NAME.items():
                if task == JiangnanFarmProduction.TASK:
                    city_group = f'farm_{city_key}'
                elif task == 'JiangnanImprovedFarmProduction':
                    city_group = f'improved_farm_{city_key}'
                elif task in {
                    'JiangnanAmberWineryProduction',
                    'JiangnanGourmetRestaurantProduction',
                    'JiangnanExquisiteGoodsShopProduction',
                    'JiangnanEmbroideryAccessoriesWorkshopProduction',
                    'JiangnanLuoShoesProduction',
                    'JiangnanJadeCrownHatWorkshopProduction',
                    'JiangnanZijinClothingPavilionProduction',
                }:
                    city_group = f'multi_product_{city_key}'
                elif (
                    task == 'JiangnanRestaurantProduction'
                    and city_key == 'huizhoufu'
                ):
                    city_group = 'restaurant_huizhoufu'
                else:
                    city_group = f'production_{city_key}'
                if not self.config.cross_get(
                    keys=f'{task}.{city_group}.Enable',
                    default=False,
                ):
                    continue
                for resource in products:
                    producer_cities.setdefault(resource, set()).add(city)
        return producer_cities

    def load_local_export_resources(self, cities):
        """读取黄边物资及城市自动产出物资，用于限制出口。"""
        result = {}
        for city in cities:
            try:
                layout = self.resource_layout.load(
                    allocation_layout_page(city)
                )
            except KeyError:
                if city not in AUTOMATIC_EXPORT_RESOURCES:
                    continue
                layout = {}
            result[city] = set(layout) | set(
                AUTOMATIC_EXPORT_RESOURCES.get(city, ())
            )
            logger.attr(
                f'AllocationLocalExports_{city}',
                sorted(result[city]),
            )
        return result

    @staticmethod
    def validate_inventory(inventories, city_rules):
        errors = []
        for city, rules in city_rules.items():
            inventory = inventories.get(city)
            if inventory is None:
                errors.append(f'{city} 仓库扫描失败')
                continue
            for resource in rules:
                if resource in SPECIAL_GROUP_KEYS.values():
                    continue
                if resource not in inventory:
                    errors.append(f'{city} 仓库缺少物资识别结果：{resource}')
                elif int(inventory.get(resource, -1)) < 0:
                    logger.warning(
                        f'{city} {resource} 库存未知，本轮跳过该城市物资'
                    )
        return errors

    @staticmethod
    def build_orders(
        inventories,
        city_rules,
        blocked_sources=None,
        blocked_destinations=None,
        log_status=True,
        progressive_defaults=None,
        producer_cities=None,
        city_strategies=None,
        local_exports=None,
        source_inventories=None,
    ):
        """按物资贪心匹配最大输出量与最大缺口。"""
        blocked_sources = set(blocked_sources or ())
        blocked_destinations = set(blocked_destinations or ())
        source_inventories = inventories if source_inventories is None else source_inventories
        progressive_targets = JiangnanAllocationPlanner.progressive_targets(
            source_inventories,
            progressive_defaults or {},
            producer_cities or {},
            log_status=log_status,
        )
        city_strategies = city_strategies or {}
        local_exports = local_exports or {}

        def can_export(city, resource):
            allowed = local_exports.get(city)
            return allowed is None or resource in allowed

        def strategy(city):
            return city_strategies.get(city, ALLOCATION_STRATEGY_NONE)

        def source_rank(city):
            return {
                ALLOCATION_STRATEGY_SEND_FIRST: 0,
                ALLOCATION_STRATEGY_NONE: 1,
                ALLOCATION_STRATEGY_RECEIVE_FIRST: 1,
            }.get(strategy(city), 1)

        def destination_rank(city):
            return {
                ALLOCATION_STRATEGY_RECEIVE_FIRST: 0,
                ALLOCATION_STRATEGY_NONE: 1,
                ALLOCATION_STRATEGY_SEND_FIRST: 1,
            }.get(strategy(city), 1)

        orders = []
        unresolved = []
        resources = sorted({
            resource
            for rules in city_rules.values()
            for resource in rules
        })

        def priority(resource):
            """返回所有城市中该物资的最大相对缺口，越大越优先。"""
            group = next(
                (members for key, members in SPECIAL_GROUPS.items()
                 if resource == SPECIAL_GROUP_KEYS[key]),
                None,
            )
            best = 0.0
            for city, rules in city_rules.items():
                if group is None:
                    if resource not in rules:
                        continue
                    lower, _ = rules[resource]
                    current = int(inventories[city].get(resource, -1))
                    if current < 0:
                        continue
                    target = float(
                        progressive_targets.get(city, {})
                        .get(resource, (lower, lower))[0]
                    )
                    if target > 0 and current < target:
                        best = max(best, (target - current) / target)
                else:
                    for item in group:
                        if item in rules:
                            item_lower = float(rules[item][0])
                        elif resource in rules:
                            item_lower = float(rules[resource][0]) / len(group)
                        else:
                            continue
                        target = float(
                            progressive_targets.get(city, {})
                            .get(item, (item_lower, item_lower))[0]
                        )
                        if target <= 0:
                            continue
                        current = int(inventories[city].get(item, -1))
                        if current < 0:
                            continue
                        if current < target:
                            best = max(best, (target - current) / target)
            return best

        # 先处理相对缺口最大的物资，而不是按中文名称排序。
        resources.sort(key=lambda resource: (-priority(resource), resource))

        for resource in resources:
            group_name = JiangnanAllocationPlanner.special_group_name(resource)
            group = SPECIAL_GROUPS.get(group_name)
            group_key = SPECIAL_GROUP_KEYS.get(group_name)
            if (
                group is not None
                and resource != group_key
                and group_key in resources
            ):
                # 共享组分支会同时合并此单品的显式规则。
                continue
            if resource != group_key:
                group = None
            if group is not None:
                # 每组四种物品共享一个仓库上限：默认目标按整组计算，
                # 因此每一种各分到四分之一。运输必须严格同物品匹配，
                # 不能用茶油的余量去“补”竹编的缺口。
                for group_item in group:
                    sources, destinations = [], []
                    for city, rules in city_rules.items():
                        if group_item in rules:
                            item_lower, item_upper = rules[group_item]
                        elif resource in rules:
                            lower, upper = rules[resource]
                            item_lower = lower // len(group)
                            item_upper = upper // len(group)
                        else:
                            continue
                        item_lower, item_upper = (
                            progressive_targets.get(city, {}).get(
                                group_item, (item_lower, item_upper)
                            )
                        )
                        current = int(
                            inventories[city].get(group_item, -1)
                        )
                        source_current = int(source_inventories.get(city, {}).get(group_item, -1))
                        if current < 0:
                            continue
                        if (
                            source_current > item_upper
                            and city not in blocked_sources
                            and can_export(city, group_item)
                        ):
                            sources.append([city, source_current - item_upper])
                        if (
                            current < item_lower
                            and city not in blocked_destinations
                        ):
                            destinations.append([city, item_lower - current])

                    sources.sort(
                        key=lambda item: (
                            source_rank(item[0]), -item[1], item[0]
                        )
                    )
                    destinations.sort(
                        key=lambda item: (
                            destination_rank(item[0]), -item[1], item[0]
                        )
                    )
                    for destination_city, need in destinations:
                        for source_index, source in enumerate(sources):
                            source_city, available = source
                            if (
                                source_city == destination_city
                                or available <= 0
                                or need <= 0
                            ):
                                continue
                            amount = min(available, need)
                            orders.append(AllocationOrder(
                                source=source_city,
                                destination=destination_city,
                                resource=group_item,
                                amount=amount,
                            ))
                            sources[source_index][1] -= amount
                            need -= amount
                        if need > 0:
                            unresolved.append(
                                (destination_city, group_item, need)
                            )
                continue
            sources = []
            destinations = []

            for city, rules in city_rules.items():
                if resource not in rules:
                    continue
                lower, upper = rules[resource]
                lower, upper = progressive_targets.get(city, {}).get(
                    resource, (lower, upper)
                )
                current = int(inventories[city][resource])
                source_current = int(source_inventories.get(city, {}).get(resource, -1))
                if current < 0:
                    continue

                if (
                    source_current > upper
                    and city not in blocked_sources
                    and can_export(city, resource)
                ):
                    available = source_current - upper
                    sources.append([city, available])
                    if log_status:
                        logger.info(
                            f'Available output: {city}, {resource}, '
                            f'confirmed={source_current}, projected={current}, upper={upper}, amount={available}'
                        )
                if current < lower and city not in blocked_destinations:
                    need = lower - current
                    destinations.append([city, need])
                    if log_status:
                        logger.info(
                            f'Requires allocation: {city}, {resource}, '
                            f'current={current}, lower={lower}, amount={need}'
                        )

            sources.sort(
                key=lambda item: (
                    source_rank(item[0]), -item[1], item[0]
                )
            )
            destinations.sort(
                key=lambda item: (
                    destination_rank(item[0]), -item[1], item[0]
                )
            )

            for destination in destinations:
                destination_city, need = destination

                for source in sources:
                    source_city, available = source
                    if source_city == destination_city or available <= 0 or need <= 0:
                        continue

                    amount = min(available, need)
                    orders.append(AllocationOrder(
                        source=source_city,
                        destination=destination_city,
                        resource=resource,
                        amount=amount,
                    ))
                    source[1] -= amount
                    need -= amount

                if need > 0:
                    unresolved.append((destination_city, resource, need))

        # 策略优先级需要跨物资生效；否则前一种物资可能先占满船位，
        # 导致后面“优先接收”的城市仍然排不到。本轮执行每发一船都会
        # 重新规划，所以只要保证订单首项严格遵守逐城策略即可。
        resource_priorities = {
            resource: priority(resource) for resource in resources
        }
        orders.sort(key=lambda order: (
            destination_rank(order.destination),
            source_rank(order.source),
            -resource_priorities.get(order.resource, 0.0),
            -order.amount,
            order.destination,
            order.source,
            order.resource,
        ))
        return orders, unresolved

    @staticmethod
    def log_plan(orders, unresolved, inventory_state=None):
        logger.hr('JIANGNAN ALLOCATION PLAN')
        if not orders:
            logger.info('No material transfer is currently required')
        for index, order in enumerate(orders, start=1):
            logger.info(
                f'Order {index}: {order.source} -> {order.destination}, '
                f'{order.resource} x{order.amount}'
            )
        current = {(city, resource): int(amount) for city, resource, amount in unresolved}
        previous_rows = (inventory_state.data.get('allocation_shortage_log', [])
                         if inventory_state is not None else [])
        previous = {(city, resource): int(amount) for city, resource, amount in previous_rows}
        changed = [(city, resource, amount) for (city, resource), amount in sorted(current.items())
                   if previous.get((city, resource)) != amount]
        resolved = sorted(set(previous) - set(current))
        logger.info(f'Allocation shortages: total={len(current)}, changed={len(changed)}, '
                    f'resolved={len(resolved)}, unchanged={len(current) - len(changed)}')
        # Keep the complete prior result per profile, across task processes.
        # Only changed shortages deserve warnings; unchanged demand is normal.
        max_details = 10
        for city, resource, amount in changed[:max_details]:
            logger.warning(
                f'Unresolved shortage: {city}, {resource} x{amount}'
            )
        if len(changed) > max_details:
            logger.info(
                f'Changed shortages: {len(changed)} total, '
                f'{len(changed) - max_details} more omitted'
            )
        for city, resource in resolved[:max_details]:
            logger.info(f'Shortage resolved: {city}, {resource}')
        if inventory_state is not None and current != previous:
            inventory_state.data['allocation_shortage_log'] = [
                [city, resource, amount] for (city, resource), amount in sorted(current.items())
            ]
            inventory_state.save()

    def recalculate_live_amount(
        self, order, source_stock, destination_free, in_transit,
        inventories, city_rules, progressive_defaults=None, producer_cities=None,
    ):
        """Re-evaluate stock phases and demand from the selected game's tile."""
        live = {city: dict(stock) for city, stock in inventories.items()}
        live.setdefault(order.source, {})[order.resource] = int(source_stock)
        destination_stock = int(live.get(order.destination, {}).get(order.resource, -1))
        # Special goods share capacity, so free space cannot identify the
        # destination's stock of an individual special resource.
        total_capacity = getattr(self, 'warehouse_capacity_by_city', {}).get(order.destination)
        if order.resource in WAREHOUSE_RESOURCES and total_capacity is not None:
            destination_stock = max(0, int(total_capacity) - int(destination_free))
            live.setdefault(order.destination, {})[order.resource] = destination_stock
        targets = self.progressive_targets(
            live, progressive_defaults or {}, producer_cities or {},
        )

        def bounds(city):
            configured = city_rules[city].get(order.resource)
            if configured is None:
                group = self.special_group_name(order.resource)
                configured = city_rules[city].get(SPECIAL_GROUP_KEYS.get(group), (0, 0))
                size = len(SPECIAL_GROUPS.get(group, ())) or 1
                configured = (configured[0] // size, configured[1] // size)
            return targets.get(city, {}).get(order.resource, configured)

        reserve = int(bounds(order.source)[1])
        need = (max(0, int(bounds(order.destination)[0]) - destination_stock - int(in_transit))
                if destination_stock >= 0 else 0)
        amount = min(max(0, int(source_stock) - reserve), need,
                     max(0, int(destination_free) - int(in_transit)))
        logger.info(
            f'Recalculate allocation from live stock: {order.source} -> {order.destination}, '
            f'{order.resource}, planned={order.amount}, source={source_stock}, '
            f'reserve={reserve}, destination={destination_stock}, in_transit={in_transit}, '
            f'need={need}, actual={amount}'
        )
        return amount, reserve

    def execute_orders(
        self,
        inventories,
        city_rules,
        inventory_state,
        initial_blocked_sources=None,
        initial_blocked_destinations=None,
        progressive_defaults=None,
        producer_cities=None,
        city_strategies=None,
        local_exports=None,
        source_inventories=None,
    ):
        executor = JiangnanAllocation(
            config=self.config,
            device=self.device,
            task=getattr(self, 'task', None),
        )
        blocked_sources = set(initial_blocked_sources or ())
        blocked_destinations = set(initial_blocked_destinations or ())
        blocked_resources = set()
        satisfied_demands = set()
        local_exports = dict(local_exports or {})
        completed = []
        completed_slots = []
        completed_finish_times = []
        virtual_inventory = {
            city: dict(inventory)
            for city, inventory in inventories.items()
        }
        confirmed_inventory = {
            city: dict(inventory) for city, inventory in
            (inventories if source_inventories is None else source_inventories).items()
        }

        # 每完成或阻塞一笔后重新规划。这样最大余量城市没有空闲船位时，
        # 会自动改用下一个有余量的城市，而不是反复卡在同一来源。
        for _ in range(100):
            # 调度器是单线程的。若本轮调配期间已有运输到期，立即结束
            # 发船循环，把执行权交回优先级更高的码头收获，避免为了继续
            # 发几十条船而让已抵达货物等待数分钟。
            now = datetime.now().replace(microsecond=0)
            urgent_dock = [
                shipment
                for shipment in inventory_state.active_shipments(due_before=now)
                if bool(self.config.cross_get(
                    keys=inventory_state._dock_key(
                        shipment.get('destination'), 'Enable'
                    ),
                    default=True,
                ))
            ]
            if urgent_dock:
                logger.info(
                    f'Yield allocation to dock harvest: '
                    f'{len(urgent_dock)} shipment(s) due'
                )
                break

            orders, _ = self.build_orders(
                virtual_inventory,
                city_rules,
                blocked_sources=blocked_sources,
                blocked_destinations=blocked_destinations,
                log_status=False,
                progressive_defaults=progressive_defaults,
                producer_cities=producer_cities,
                city_strategies=city_strategies,
                local_exports={
                    city: set(resources) - {
                        resource for source, resource in blocked_resources
                        if source == city
                    }
                    for city, resources in local_exports.items()
                },
                source_inventories=confirmed_inventory,
            )
            orders = [order for order in orders
                      if (order.source, order.resource) not in blocked_resources
                      and (order.destination, order.resource) not in satisfied_demands]
            if not orders:
                break
            order = orders[0]

            # Live stock may be much lower than the projected planner stock.
            # The executor must preserve the same source threshold even then.
            source_targets = self.progressive_targets(
                confirmed_inventory, progressive_defaults or {}, producer_cities or {},
            )
            source_bounds = city_rules[order.source].get(order.resource)
            if source_bounds is None:
                group = self.special_group_name(order.resource)
                bounds = city_rules[order.source].get(SPECIAL_GROUP_KEYS.get(group), (0, 0))
                size = len(SPECIAL_GROUPS.get(group, ())) or 1
                source_bounds = (bounds[0] // size, bounds[1] // size)
            executor.source_reserve = source_targets.get(order.source, {}).get(
                order.resource, source_bounds,
            )[1]
            executor.recalculate_live_amount = lambda stock, free, transit: self.recalculate_live_amount(
                order, stock, free, transit, confirmed_inventory, city_rules,
                progressive_defaults=progressive_defaults, producer_cities=producer_cities,
            )

            result = executor.transport(
                source=order.source,
                destination=order.destination,
                resource=order.resource,
                amount=order.amount,
            )
            if (not result and getattr(executor, 'last_error', '') == 'source_reserve'
                    and getattr(executor, 'last_reconciled_source_inventory', None) is None):
                inventory_state.request_scan(
                    'live source stock is below export reserve', cities={order.source},
                )
            # 首次进入某城市时，执行器可能已把旧的“全货物”
            # 辅助表当场裁成黄边集合。立即刷新本轮计划器视图，
            # 避免同一轮又为该城市生成白底货物订单。
            try:
                refreshed = executor.resource_layout.load(
                    allocation_layout_page(order.source)
                )
                local_exports[order.source] = set(refreshed) | set(
                    AUTOMATIC_EXPORT_RESOURCES.get(order.source, ())
                )
            except KeyError:
                pass
            for reconciled in (
                executor.last_reconciled_inventory,
                getattr(executor, 'last_reconciled_source_inventory', None),
            ):
                if reconciled is None:
                    continue
                live_city, live_resource, live_amount, live_delta = reconciled
                confirmed_inventory.setdefault(live_city, {})[live_resource] = int(live_amount)
                if abs(int(live_delta)) >= max(100, int(live_amount) // 5):
                    inventory_state.request_scan(
                        f'large live inventory correction: {live_resource}, delta={live_delta}',
                        cities={live_city},
                    )
                if live_city in virtual_inventory:
                    virtual_inventory[live_city][live_resource] = max(
                        0,
                        int(virtual_inventory[live_city].get(
                            live_resource, 0
                        )) + int(live_delta),
                    )
                    logger.info(
                        f'Update allocation working inventory from live '
                        f'allocation page: {live_city} {live_resource}, '
                        f'actual={live_amount}, delta={live_delta:+d}, '
                        f'projected='
                        f'{virtual_inventory[live_city][live_resource]}'
                    )
            live_transit = getattr(executor, 'last_destination_in_transit', None)
            destination_stock = confirmed_inventory.get(order.destination, {}).get(order.resource, -1)
            if live_transit is not None and int(destination_stock) >= 0:
                virtual_inventory[order.destination][order.resource] = (
                    int(destination_stock) + int(live_transit)
                )
            if result:
                actual_order = order._replace(amount=executor.last_amount)
                finish_at = (
                    datetime.now().replace(microsecond=0)
                    + executor.last_duration
                )
                completed.append(actual_order)
                completed_slots.append(executor.last_slot_index)
                completed_finish_times.append(finish_at)
                # 每条船点击出发成功后立刻扣除来源库存并登记台账。以前
                # 等整批订单全部执行完才统一落盘；一轮跨城操作耗时较长
                # 时，前几条船甚至已经抵达，造成倒计时早于登记时间。
                inventory_state.record_departures(
                    [actual_order],
                    [executor.last_slot_index],
                    finish_times=[finish_at],
                )
                if getattr(executor, 'last_source_stock_lower_bound', False):
                    inventory_state.request_scan(
                        'allocation source stock first digit was ambiguous',
                        cities={order.source},
                    )
                virtual_inventory[order.source][order.resource] -= (
                    executor.last_amount
                )
                confirmed_inventory[order.source][order.resource] -= executor.last_amount
                # This view is used only to reserve goods already in transit
                # while planning the next order. The destination's confirmed
                # inventory stays unchanged until dock collection succeeds.
                virtual_inventory[order.destination][order.resource] += (
                    executor.last_amount
                )
                continue

            if executor.last_error in (
                'source_empty', 'source_reserve', 'destination_full', 'destination_satisfied',
            ):
                if executor.last_error in ('source_empty', 'source_reserve'):
                    blocked_resources.add((order.source, order.resource))
                else:
                    satisfied_demands.add((order.destination, order.resource))
                logger.warning(
                    f'Postpone resource {order.source}/{order.resource}: '
                    f'live allocation amount is zero ({executor.last_error}); continue other resources'
                )
                continue

            if executor.last_error == 'no_idle_slot':
                blocked_sources.add(order.source)
                inventory_state.record_no_idle_source(order.source)
                logger.warning(
                    f'Postpone remaining orders from {order.source}: no idle slot'
                )
                continue

            if executor.last_error == 'capacity_unknown':
                blocked_destinations.add(order.destination)
                logger.warning(
                    f'Postpone remaining orders to {order.destination}: '
                    f'unable to confirm destination capacity'
                )
                continue

            if executor.last_error == 'duration_unknown':
                # 尚未点击出发，库存和船位都没有变化。只暂停当前来源城，
                # 让本轮继续处理其他城市，避免一次 OCR 抖动把整个任务
                # 连续失败三次并触发人工接管。
                blocked_sources.add(order.source)
                logger.warning(
                    f'Postpone remaining orders from {order.source}: '
                    f'unable to confirm transport duration'
                )
                continue

            if executor.last_error in (
                'resource_mismatch',
                'resource_icon_missing',
                'resource_icon_not_found',
                'resource_identity_mismatch',
            ):
                # 图标表缺失、识别不唯一或容量与目标物资不一致时，禁止
                # 按行列补救。仓库实扫会重新建立货物名到图标的映射。
                if executor.last_error == 'resource_identity_mismatch':
                    blocked_resources.add((order.source, order.resource))
                else:
                    blocked_sources.add(order.source)
                if executor.last_error == 'resource_mismatch':
                    # A live capacity/resource disagreement can involve both
                    # cities. Keep this reconciliation immediate.
                    inventory_state.request_scan(
                        f'allocation resource icon failure: {executor.last_error}',
                        cities={order.source, order.destination},
                    )
                else:
                    # The selection failed in the source city's resource list;
                    # scanning the destination cannot repair its icon. Persist
                    # a per-city/resource cooldown so an unresolved icon does
                    # not requeue an expensive scan on every allocation pass.
                    requests = inventory_state.data.setdefault('allocation_icon_scan_requests', {})
                    scan_key = f'{order.source}/{order.resource}'
                    try:
                        last_request = datetime.fromisoformat(str(requests.get(scan_key)))
                        age = (now - last_request).total_seconds()
                    except (TypeError, ValueError):
                        age = -1
                    # An immediate warehouse scan already refreshes the icon.
                    # Persistent transport-layout failures need no new full
                    # warehouse visit every half hour; allow one per six hours.
                    cooldown_seconds = 6 * 60 * 60
                    if not 0 <= age < cooldown_seconds:
                        requests[scan_key] = now.isoformat(sep=' ')
                        inventory_state.request_scan(
                            f'allocation resource icon failure: {order.resource}, {executor.last_error}',
                            cities={order.source},
                        )
                    else:
                        logger.info(
                            f'Warehouse icon repair scan cooling down: '
                            f'{order.source}/{order.resource}, retry in {int(cooldown_seconds - age)}s'
                        )
                logger.warning(
                    f'Postpone unrecognized resource {order.source}/{order.resource}'
                    if executor.last_error == 'resource_identity_mismatch' else
                    f'Postpone remaining orders from {order.source}: '
                    f'unable to recognize selected resource {order.resource}'
                )
                continue

            if executor.last_error == 'resource_not_local':
                blocked_sources.add(order.source)
                logger.warning(
                    f'Postpone remaining orders from {order.source}: '
                    f'{order.resource} is not a yellow-border local resource'
                )
                continue

            # 单笔交互失败不能让后面所有城市的空闲船位一起闲置。
            # 当前来源城先退出本轮，下一次调度再复查；executor 的入口
            # 会自行返回城市列表，因此可以安全继续尝试其他来源城市。
            blocked_sources.add(order.source)
            logger.warning(
                f'Postpone remaining orders from {order.source}: '
                f'allocation interaction failed ({executor.last_error or "unknown"})'
            )
            continue

        else:
            logger.warning('Allocation execution exceeded 100 orders')
            return (
                False, completed, completed_slots,
                completed_finish_times,
                blocked_sources, blocked_destinations,
            )

        return (
            True, completed, completed_slots,
            completed_finish_times,
            blocked_sources, blocked_destinations,
        )

    def run(self):
        logger.hr('JIANGNAN AUTOMATIC MATERIAL ALLOCATION', level=0)

        (
            city_rules,
            errors,
            allocation_cities,
            export_cities,
            import_cities,
        ) = self.load_city_rules()
        city_strategies = self.load_city_strategies(allocation_cities)
        for error in errors:
            logger.warning(error)
        if errors:
            self.delay_retry(allocation_cities)
            return False

        inventory_state = JiangnanInventoryState(self.config)
        # 到时只说明船理论上已经抵达，不能在码头真正领取前把货物写进
        # 确认库存。规划层仍会通过 project_for_planning() 计入这批在途量。
        arrived_count = inventory_state.apply_due_arrivals()
        logger.attr('AllocationDueInventoryNodes', arrived_count)

        # 船位仍被占用时必须把控制权交给优先级更高的独立码头收获任务，
        # 真正收完以后才把目的库存入账，并立即重新唤醒调配。
        # 码头收获会立即重新唤醒物资调配。本轮若继续
        # 自己扫描和规划，会造成“已到达但没有立刻调用码头收获”。
        due_dock = inventory_state.due_dock_shipments()
        if due_dock and inventory_state.request_dock_harvest(
            shipments=due_dock
        ):
            logger.info(
                'Yield material allocation to JiangnanDockHarvest: '
                f'{len(due_dock)} shipment(s) arrived'
            )
            # 防止当前调配任务仍处于到期状态与码头任务反复竞争；码头
            # 真正收获后会把调配时间重新提前到当时。
            self.config.task_delay(minute=1)
            return True

        # 先逐城读取两个船位。正在运输的船占用来源城市运输资格；每个
        # 船位的结束时间会保留在 last_snapshot，供逐城 NextRun 和运输
        # 台账使用。
        status = JiangnanTransportStatus(
            config=self.config,
            device=self.device,
            task=getattr(self, 'task', None),
        )
        status_cities = list(dict.fromkeys(
            allocation_cities + sorted(inventory_state.active_sources())
        ))
        due_cities = inventory_state.due_cities(status_cities)
        logger.attr('AllocationDueCities', due_cities)
        if not due_cities:
            # 页面总倒计时只负责唤醒模块；真正执行时仍以逐城时间为准。
            # 手动提前运行时不扫描、更不发船，只重新对齐最近的逐城时间。
            next_runs = inventory_state.all_city_next_runs(
                status_cities,
                fallback=datetime.now().replace(microsecond=0)
                + timedelta(minutes=10),
            )
            if next_runs:
                self.config.task_delay(target=min(next_runs.values()))
            else:
                self.config.task_delay(success=True)
            return True
        # 调配状态扫描若已经亲眼看到船位“已抵达”，应当当场完成收取，
        # 不能继续把它当作占用船位留给稍后的独立码头任务。仅收取台账中
        # 且目的城市启用码头收获的节点，保留逐城开关语义。
        active_by_slot = {}
        for shipment in sorted(
            inventory_state.active_shipments(),
            key=lambda item: item.get('departed_at') or '',
        ):
            active_by_slot.setdefault(
                (shipment.get('source'), int(shipment.get('slot', -1))),
                shipment,
            )
        collect_slots = {
            key for key, shipment in active_by_slot.items()
            if bool(self.config.cross_get(
                keys=inventory_state._dock_key(
                    shipment.get('destination'), 'Enable'
                ),
                default=True,
            ))
        }
        status_success, finish_times_before, status_unknown, collected = status.scan(
            cities=due_cities,
            collect_arrived=True,
            collect_slots=collect_slots,
        )
        logger.attr('TransportArrivalsCollected', collected)
        if not status_success:
            self.delay_retry(due_cities)
            return False

        matched_arrivals, unknown_arrivals = inventory_state.apply_collected(
            status.last_collected_slots
        )
        logger.attr('TransportArrivalsMatched', matched_arrivals)
        logger.attr('TransportArrivalsUnknown', unknown_arrivals)
        if matched_arrivals:
            inventory_state.sync_dock_schedule()
        initial_snapshot = dict(status.last_snapshot)
        inventory_state.record_slot_snapshot(initial_snapshot)
        unavailable_sources = self.disable_unavailable_exports(initial_snapshot)
        export_cities = [
            city for city in export_cities if city not in unavailable_sources
        ]

        # 只有两个船位都被正在运输/未知状态占用的城市，才在本轮直接
        # 禁止作为来源；刚收取完成的船位已经重新空闲。
        occupied_sources = inventory_state.occupied_sources(status_cities)
        # 未到逐城 NextRun 通常无需复查，但最近快照明确还有空闲船位的
        # 城市仍可继续发船。旧逻辑把“另一艘船尚未结束”误当成整座城市
        # 冷却，导致大量第二船位长期空闲。
        idle_sources = inventory_state.idle_sources(export_cities)
        occupied_sources.update(
            (set(export_cities) - set(due_cities)) - idle_sources
        )
        occupied_sources.update(
            set(allocation_cities) - set(export_cities)
        )
        logger.attr('AllocationIdleSources', sorted(idle_sources))
        blocked_imports = set(allocation_cities) - set(import_cities)
        logger.attr('AllocationOccupiedSources', sorted(occupied_sources))
        logger.attr('AllocationBlockedImports', sorted(blocked_imports))

        if not city_rules:
            logger.warning('No city inventory ranges are configured')
            if finish_times_before:
                self.config.task_delay(
                    target=min(finish_times_before) + timedelta(minutes=1)
                )
            elif status_unknown:
                self.config.task_delay(minute=10)
            else:
                self.config.task_delay(success=True)
            return True

        cities = list(city_rules.keys())

        # 物资调配永远只读取“仓库库存”配置和运输台账。实扫由独立的
        # JiangnanWarehouseInventory 定时任务负责，调配任务不再内联扫描。
        next_scan = inventory_state.next_scan_time()
        if unknown_arrivals > 0:
            inventory_state.request_scan('untracked transport arrival')
        # Missing baselines and stock differences do not advance the periodic
        # warehouse clock. Orders use the configured/calculated inventory.
        inventories = inventory_state.load_inventories(cities)
        logger.info(
            f'Use configured/calculated warehouse inventory; '
            f'next real scan={next_scan}'
        )

        self.warehouse_capacity_by_city = inventory_state.load_capacities(cities)
        self.warehouse_special_capacity_by_city = {
            city: {
                group: capacity for group in SPECIAL_GROUPS
            }
            for city, capacity in self.warehouse_capacity_by_city.items()
        }
        city_rules, capacity_errors = self.resolve_percentage_rules(
            city_rules,
            self.warehouse_capacity_by_city,
            self.warehouse_special_capacity_by_city,
        )
        errors = capacity_errors + self.validate_inventory(
            inventories,
            city_rules,
        )
        for error in errors:
            logger.warning(error)
        if errors:
            self.delay_retry(cities)
            return False

        planning_inventories = inventory_state.project_for_planning(inventories)
        progressive_defaults = self.build_progressive_defaults(
            city_rules,
            self.warehouse_capacity_by_city,
            self.warehouse_special_capacity_by_city,
        )
        local_exports = self.load_local_export_resources(city_rules)
        producer_cities = self.load_producer_cities(local_exports)
        orders, unresolved = self.build_orders(
            planning_inventories,
            city_rules,
            blocked_sources=occupied_sources,
            blocked_destinations=blocked_imports,
            progressive_defaults=progressive_defaults,
            producer_cities=producer_cities,
            city_strategies=city_strategies,
            local_exports=local_exports,
            source_inventories=inventories,
        )
        self.log_plan(orders, unresolved, inventory_state=inventory_state)

        (
            success,
            completed,
            completed_slots,
            completed_finish_times,
            blocked_sources,
            blocked_destinations,
        ) = (
            self.execute_orders(
                planning_inventories,
                city_rules,
                inventory_state,
                initial_blocked_sources=occupied_sources,
                initial_blocked_destinations=blocked_imports,
                progressive_defaults=progressive_defaults,
                producer_cities=producer_cities,
                city_strategies=city_strategies,
                local_exports=local_exports,
                source_inventories=inventories,
            )
        )
        logger.attr('AllocationOrdersPlanned', len(orders))
        logger.attr('AllocationOrdersCompleted', len(completed))
        logger.attr('AllocationBlockedDestinations', list(blocked_destinations))
        if not success:
            self.delay_retry(due_cities)
            return False

        # 来源库存与运输台账已在每次点击“出发”成功后立即保存；这里不
        # 再批量登记，避免重复扣减。

        # 新订单发出后只复查本轮真正使用过的来源城市，读取新船结束
        # 时间；其他城市沿用本轮开头或上一次保存的船位状态。
        dispatched_sources = sorted({order.source for order in completed})
        if dispatched_sources:
            status_success, finish_times, status_unknown_after, _ = status.scan(
                cities=dispatched_sources,
                collect_arrived=False,
            )
            if not status_success:
                self.delay_retry(dispatched_sources)
                return False
            final_snapshot = dict(status.last_snapshot)
        else:
            finish_times = []
            status_unknown_after = False
            final_snapshot = {}

        combined_snapshot = dict(initial_snapshot)
        combined_snapshot.update(final_snapshot)
        # Only a fresh post-departure observation can clear a failed selection.
        # The merged initial snapshot may still contain the stale idle label.
        if final_snapshot:
            inventory_state.record_slot_snapshot(final_snapshot)
        inventory_state.record_slot_snapshot(combined_snapshot, clear_no_idle=False)
        inventory_state.assign_finish_times(combined_snapshot)
        now = datetime.now().replace(microsecond=0)
        allocation_fallback = (
            next_scan if next_scan > now else now + timedelta(minutes=10)
        )
        city_next_runs = inventory_state.set_city_next_runs(
            combined_snapshot,
            fallback=allocation_fallback,
        )
        # 先给 UI 中的全部城市补齐有效时间，避免未参与本轮扫描的城市
        # 继续显示 2020 占位值；外层调度仍只考虑本轮参与调配的城市。
        displayed_city_next_runs = inventory_state.all_city_next_runs(
            list(CITY_KEY_TO_NAME.values()),
            fallback=allocation_fallback,
            normalize_past=True,
        )
        all_city_next_runs = {
            city: displayed_city_next_runs[city]
            for city in status_cities
            if city in displayed_city_next_runs
        }
        logger.attr('AllocationCityNextRuns', all_city_next_runs)

        if finish_times or all_city_next_runs:
            # 总任务使用所有城市 NextRun 和下次仓库实扫中最近的时间；
            # UI 中仍保留每座城市自己的时间。
            targets = list(all_city_next_runs.values())
            next_finish = min(targets)
            logger.attr('NextTransportFinish', next_finish)
            self.config.task_delay(target=targets)
        elif status_unknown or status_unknown_after or blocked_sources:
            # OCR 暂时失败或已有船位被占用时，用短间隔重新确认，避免漏收。
            self.config.task_delay(minute=10)
        else:
            self.config.task_delay(success=True)
        return True
