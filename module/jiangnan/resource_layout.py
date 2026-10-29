import json
import re

from module.jiangnan.resource import (
    WAREHOUSE_RESOURCE_POSITION,
    WAREHOUSE_RESOURCE_ROWS,
    WAREHOUSE_RESOURCES,
    WAREHOUSE_SPECIAL_RESOURCES,
)
from module.logger import logger


PAGE_WAREHOUSE = 'warehouse'
PAGE_ALLOCATION = 'allocation'
PAGE_CITY = 'city'
PAGE_PRODUCTION = 'production'

CITY_LAYOUT_NAMES = (
    '应天府', '苏州府',
    '杭州府', '松江府',
    '鸡鸣山', '徽州府',
    '扬州府', '绍兴府',
    '宁波府', '江宁县',
    '白雪镇', '棕影岛',
    '龙宫',
)

CITY_LAYOUT_KEYS = {
    '应天府': 'yingtianfu',
    '苏州府': 'suzhoufu',
    '杭州府': 'hangzhoufu',
    '松江府': 'songjiangfu',
    '鸡鸣山': 'jimingshan',
    '徽州府': 'huizhoufu',
    '扬州府': 'yangzhoufu',
    '绍兴府': 'shaoxingfu',
    '宁波府': 'ningbofu',
    '江宁县': 'jiangningxian',
    '白雪镇': 'baixuezhen',
    '棕影岛': 'zongyingdao',
    '龙宫': 'longgong',
}

PAGE_CONFIG = {
    PAGE_WAREHOUSE: (
        'JiangnanAuxiliaryWarehouseLayout.WarehouseLayout.Distribution',
        2,
    ),
    PAGE_CITY: (
        'JiangnanAuxiliaryCityLayout.CityLayout.Distribution',
        2,
    ),
}
LEARNING_STATUS_KEY = (
    'JiangnanAuxiliaryAllocationLayout.LayoutLearning.Status'
)


def allocation_layout_page(city):
    """Return the independent allocation-layout page for a source city."""
    city = str(city or '').strip()
    city_key = CITY_LAYOUT_KEYS.get(city, city)
    if city_key not in CITY_LAYOUT_KEYS.values():
        raise KeyError(f'Unknown allocation source city: {city}')
    return f'{PAGE_ALLOCATION}_{city_key}'


def production_layout_page(city):
    """Return the independent production-building layout for one city."""
    city = str(city or '').strip()
    city_key = CITY_LAYOUT_KEYS.get(city, city)
    if city_key not in CITY_LAYOUT_KEYS.values():
        raise KeyError(f'Unknown production city: {city}')
    return f'{PAGE_PRODUCTION}_{city_key}'


def page_config(page):
    if page in PAGE_CONFIG:
        return PAGE_CONFIG[page]
    prefix = f'{PAGE_ALLOCATION}_'
    if str(page).startswith(prefix):
        city_key = str(page)[len(prefix):]
        if city_key in CITY_LAYOUT_KEYS.values():
            return (
                'JiangnanAuxiliaryAllocationLayout.'
                f'allocation_{city_key}.Distribution',
                4,
            )
    prefix = f'{PAGE_PRODUCTION}_'
    if str(page).startswith(prefix):
        city_key = str(page)[len(prefix):]
        if city_key in CITY_LAYOUT_KEYS.values():
            return (
                'JiangnanAuxiliaryProductionLayout.'
                f'production_layout_{city_key}.Distribution',
                5,
            )
    # Compatibility for callers that have not supplied a source city yet.
    if page == PAGE_ALLOCATION:
        return (
            'JiangnanAuxiliaryAllocationLayout.'
            'allocation_yingtianfu.Distribution',
            4,
        )
    raise KeyError(f'Unknown auxiliary layout page: {page}')


def default_warehouse_layout():
    """仓库普通区为两列；底部两组徽州特产按同一行对齐。"""
    result = {}
    for resource, position in WAREHOUSE_RESOURCE_POSITION.items():
        result[resource] = (
            int(position['row']),
            0 if position['side'] == 'left' else 1,
        )

    first_special_row = len(WAREHOUSE_RESOURCE_ROWS)
    right = WAREHOUSE_SPECIAL_RESOURCES[:4]
    left = WAREHOUSE_SPECIAL_RESOURCES[4:]
    for offset, resource in enumerate(left):
        result[resource] = (first_special_row + offset, 0)
    for offset, resource in enumerate(right):
        result[resource] = (first_special_row + offset, 1)
    return result


def default_allocation_layout():
    resources = WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
    return {
        resource: divmod(index, 4)
        for index, resource in enumerate(resources)
    }


def default_city_layout():
    return {
        city: divmod(index, 2)
        for index, city in enumerate(CITY_LAYOUT_NAMES)
    }


DEFAULT_LAYOUTS = {
    PAGE_WAREHOUSE: default_warehouse_layout(),
    PAGE_ALLOCATION: default_allocation_layout(),
    PAGE_CITY: default_city_layout(),
}


def default_layout(page):
    if page == PAGE_WAREHOUSE:
        return DEFAULT_LAYOUTS[PAGE_WAREHOUSE]
    if page == PAGE_CITY:
        return DEFAULT_LAYOUTS[PAGE_CITY]
    if page == PAGE_ALLOCATION or str(page).startswith(f'{PAGE_ALLOCATION}_'):
        return DEFAULT_LAYOUTS[PAGE_ALLOCATION]
    if str(page).startswith(f'{PAGE_PRODUCTION}_'):
        # Each city owns a different filtered building list.  It is learned
        # from the live search page instead of borrowing another city's list.
        return {}
    raise KeyError(f'Unknown auxiliary layout page: {page}')


def serialize_layout(layout, page):
    """UI 按“行号: 从左到右的货物”显示，便于整体调整顺序。"""
    _, columns = page_config(page)
    max_row = max((position[0] for position in layout.values()), default=-1)
    by_position = {
        tuple(position): name for name, position in layout.items()
    }
    lines = []
    for row in range(max_row + 1):
        values = []
        for column in range(columns):
            values.append(by_position.get((row, column), '-'))
        # 部分城市调配页的最后一行不足整行。末尾空列无需用“-”展示；解析器仍兼容
        # 用户已有配置中的占位符。
        while values and values[-1] == '-':
            values.pop()
        lines.append(f'{row + 1}: {", ".join(values)}')
    return '\n'.join(lines)


DEFAULT_WAREHOUSE_LAYOUT_TEXT = serialize_layout(
    DEFAULT_LAYOUTS[PAGE_WAREHOUSE], PAGE_WAREHOUSE,
)
DEFAULT_ALLOCATION_LAYOUT_TEXT = serialize_layout(
    DEFAULT_LAYOUTS[PAGE_ALLOCATION], PAGE_ALLOCATION,
)


class ResourceLayoutManager:
    """读取可编辑位置表，并在连续三次确认错误后自学习更新。"""

    def __init__(self, config):
        self.config = config
        self._cache = {}
        self._status_cache = None

    @staticmethod
    def _normalize_resource(value):
        return ''.join(re.findall(r'[\u4e00-\u9fff]+', str(value or '')))

    @classmethod
    def _normalize_layout_item(cls, page, value):
        if str(page).startswith(f'{PAGE_PRODUCTION}_'):
            # 等级和同等级建筑数量是生产结果标签的一部分，例如
            # “4级餐馆(2)”。保留数字，括号本身不影响解析。
            return ''.join(re.findall(
                r'[\u4e00-\u9fffA-Za-z0-9]+', str(value or '')
            ))
        return cls._normalize_resource(value)

    def load(self, page):
        if page in self._cache:
            return dict(self._cache[page])
        key, columns = page_config(page)
        default = default_layout(page)
        empty_by_default = str(page).startswith((
            f'{PAGE_ALLOCATION}_', f'{PAGE_PRODUCTION}_',
        ))
        text = self.config.cross_get(
            keys=key,
            default='' if empty_by_default else serialize_layout(default, page),
        )
        parsed = {}
        for raw_line in str(text or '').splitlines():
            grid = re.match(r'^\s*(\d+)\s*[:：]\s*(.+?)\s*$', raw_line)
            if grid:
                row = int(grid.group(1)) - 1
                values = re.split(r'[,，]', grid.group(2))
                for column, raw_resource in enumerate(values[:columns]):
                    resource = self._normalize_layout_item(page, raw_resource)
                    if (
                        resource
                        and row >= 0
                        and (
                            resource in default
                            or str(page).startswith(f'{PAGE_PRODUCTION}_')
                        )
                    ):
                        parsed[resource] = (row, column)
                continue
            matched = re.match(
                r'^\s*([^:：]+)\s*[:：]\s*(\d+)\s*[,，]\s*(\d+)\s*$',
                raw_line,
            )
            if not matched:
                continue
            resource = self._normalize_layout_item(page, matched.group(1))
            row = int(matched.group(2)) - 1
            column = int(matched.group(3)) - 1
            if (
                not resource
                or row < 0
                or not 0 <= column < columns
                or (
                    resource not in default
                    and not str(page).startswith(f'{PAGE_PRODUCTION}_')
                )
            ):
                logger.warning(
                    f'Ignore invalid {page} resource layout: {raw_line}'
                )
                continue
            parsed[resource] = (row, column)

        # 物资调配页的配置仅列出黄边“本地可生产”物资。
        # 因此未写的货物是有意排除，不能再用完整默认表
        # 补回。仓库和城镇位置表仍保留原有容错。
        if str(page).startswith((
            f'{PAGE_ALLOCATION}_', f'{PAGE_PRODUCTION}_',
        )):
            result = dict(parsed)
        else:
            result = dict(default)
            result.update(parsed)
        self._cache[page] = result
        return dict(result)

    def replace_layout(self, page, layout):
        """用一次完整页面扫描结果替换对应城市的辅助位置表。"""
        default = default_layout(page)
        accepts_dynamic = str(page).startswith(f'{PAGE_PRODUCTION}_')
        cleaned = {
            str(resource): (int(position[0]), int(position[1]))
            for resource, position in dict(layout).items()
            if (accepts_dynamic and self._normalize_layout_item(page, resource))
            or resource in default
        }
        self._cache[page] = cleaned
        key, _ = page_config(page)
        self.config.modified[key] = serialize_layout(cleaned, page)

        status = self._load_status()
        if page in status:
            del status[page]
            self._save_status(status)
        self.config.update()
        logger.info(
            f'Replaced auxiliary layout: page={page}, '
            f'resources={len(cleaned)}'
        )
        return dict(cleaned)

    def replace_status(self, page, status):
        """写入没有格位表的有效状态，例如无生产或码头未修复。"""
        status = str(status or '').strip()
        self._cache[page] = {}
        key, _ = page_config(page)
        self.config.modified[key] = status

        learning = self._load_status()
        if page in learning:
            del learning[page]
            self._save_status(learning)
        self.config.update()
        logger.info(
            f'Replaced auxiliary layout status: page={page}, status={status}'
        )
        return status

    def position(self, page, resource):
        return self.load(page).get(str(resource))

    def resource_at(self, page, row, column):
        target = (int(row), int(column))
        return next(
            (
                resource for resource, position in self.load(page).items()
                if position == target
            ),
            None,
        )

    def _load_status(self):
        if self._status_cache is not None:
            return self._status_cache
        raw = self.config.cross_get(keys=LEARNING_STATUS_KEY, default='{}')
        try:
            value = json.loads(str(raw or '{}'))
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
        self._status_cache = value if isinstance(value, dict) else {}
        return self._status_cache

    def _save_status(self, status):
        self._status_cache = status
        self.config.modified[LEARNING_STATUS_KEY] = json.dumps(
            status,
            ensure_ascii=False,
            sort_keys=True,
        )

    def confirm_position(self, page, resource, observed):
        """记录一次完整识别得到的位置；连续三次一致才更新位置表。"""
        observed = (int(observed[0]), int(observed[1]))
        expected = self.position(page, resource)
        status = self._load_status()
        page_status = status.setdefault(page, {})

        if expected == observed:
            if resource in page_status:
                del page_status[resource]
                self._save_status(status)
            return False

        previous = page_status.get(resource, {})
        candidate = list(observed)
        count = (
            int(previous.get('count', 0)) + 1
            if previous.get('position') == candidate else 1
        )
        page_status[resource] = {
            'position': candidate,
            'count': count,
        }
        logger.warning(
            f'Resource layout mismatch: page={page}, resource={resource}, '
            f'configured={expected}, observed={observed}, count={count}/3'
        )

        if count < 3:
            self._save_status(status)
            return False

        layout = self.load(page)
        old_position = layout.get(resource)
        displaced = next((
            name for name, position in layout.items()
            if name != resource and position == observed
        ), None)
        if displaced is not None and old_position is not None:
            layout[displaced] = old_position
        layout[resource] = observed
        self._cache[page] = layout
        key, _ = page_config(page)
        self.config.modified[key] = serialize_layout(layout, page)
        del page_status[resource]
        self._save_status(status)
        logger.warning(
            f'Auxiliary layout updated after three confirmations: '
            f'page={page}, resource={resource}, position={observed}'
        )
        # 位置表属于用户需要在 UI 中立即看到的辅助信息，不能只留在
        # 当前任务的 modified 缓存里等待任务结束。第三次确认后立刻
        # 持久化，页面重新进入/刷新时即可读取新值。
        self.config.update()
        return True
