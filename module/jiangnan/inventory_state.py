import json
from uuid import uuid4
from datetime import datetime, timedelta
from pathlib import Path

from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.city import CITY_KEY_TO_PRODUCTION_TASK
from module.jiangnan.resource import (
    WAREHOUSE_OCR_ALIASES,
    WAREHOUSE_RESOURCES,
    WAREHOUSE_SPECIAL_RESOURCES,
    SHAOXING_MERCHANT_RESOURCES,
)
from module.logger import logger
from module.jiangnan.inventory_ledger import InventoryLedger


ALL_RESOURCES = tuple(dict.fromkeys(
    WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES
    + list(SHAOXING_MERCHANT_RESOURCES)
))


def resources_for_city(city):
    base = tuple(WAREHOUSE_RESOURCES + WAREHOUSE_SPECIAL_RESOURCES)
    return (base + SHAOXING_MERCHANT_RESOURCES
            if city == '绍兴府' else base)
CITY_NAME_TO_KEY = {name: key for key, name in CITY_KEY_TO_NAME.items()}
DEFAULT_TIME = datetime(2020, 1, 1)
PRODUCTION_THRESHOLD_RATIO = 0.97


class JiangnanInventoryState:
    """仓库 UI 数据、扫描元数据和自动运输台账。"""

    STATE_FILE = Path('jiangnan_warehouse_state.json')
    TASK = 'JiangnanWarehouseInventory'

    def __init__(self, config, path=None):
        self.config = config
        self.path = Path(path) if path else self.STATE_FILE
        self.data = self._load_state()
        self.inventory_ledger = InventoryLedger(self.data)
        self._inventory_text_cache = {}
        self._production_cost_reservations = {}
        # Popup handlers must use the task's current ledger. Loading a second
        # ledger while switching cities would let the outer task overwrite it.
        self.config._jiangnan_inventory_state = self
        if any(self._production_inventory_change(item)
               for item in self.data.get('scheduled_changes', [])):
            self.sync_inventory_change_schedule(update=False)

    @staticmethod
    def _production_inventory_change(change):
        return (
            bool(change.get('exclude_from_inventory'))
            or str(change.get('reference') or '').startswith(('production:', 'farm:'))
            or str(change.get('reason') or '生产').startswith('生产')
        )

    @staticmethod
    def _parse_time(value):
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _time_text(value):
        return value.replace(microsecond=0).isoformat(sep=' ')

    def _load_state(self):
        try:
            with self.path.open('r', encoding='utf-8') as stream:
                data = json.load(stream)
            if isinstance(data, dict):
                data.setdefault('version', 1)
                data.setdefault('capacities', {})
                data.setdefault('special_capacities', {})
                data.setdefault('shipments', [])
                data.setdefault('scheduled_changes', [])
                data.setdefault('slot_status', {})
                data.setdefault('overflow_paused_production', [])
                data.setdefault('inventory_snapshot', {})
                data.setdefault('inventory_events', [])
                data.setdefault('inventory_event_version', 1)
                return data
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f'Unable to read Jiangnan inventory state: {exc}')
        return {
            'version': 1,
            'capacities': {},
            'special_capacities': {},
            'shipments': [],
            'scheduled_changes': [],
            'slot_status': {},
            'overflow_paused_production': [],
        }

    def save(self):
        now = datetime.now()
        retention = timedelta(days=7)
        self.inventory_ledger.compact(now=now, retention=retention)

        shipments = []
        for shipment in self.data.get('shipments', []):
            collected_at = self._parse_time(shipment.get('collected_at'))
            if (
                shipment.get('status') != 'collected'
                or collected_at is None
                or now - collected_at <= retention
            ):
                shipments.append(shipment)
        self.data['shipments'] = shipments

        scheduled_changes = []
        for change in self.data.get('scheduled_changes', []):
            credited_at = self._parse_time(change.get('credited_at'))
            if credited_at is None or now - credited_at <= retention:
                scheduled_changes.append(change)
        self.data['scheduled_changes'] = scheduled_changes
        self.data['tax_popup_receipts'] = [
            item for item in self.data.get('tax_popup_receipts', [])
            if (self._parse_time(item.get('closed_at')) is None
                or now - self._parse_time(item['closed_at']) <= retention)
        ]
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)
        temporary.replace(self.path)

    @staticmethod
    def normalize_resource(value):
        value = str(value).strip()
        return WAREHOUSE_OCR_ALIASES.get(value, value)

    @classmethod
    def parse_inventory(cls, text):
        result = {}
        text = '' if text is None else str(text)
        for raw_line in text.replace('；', '\n').replace(';', '\n').splitlines():
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            separator = '：' if '：' in line else ':'
            if separator not in line:
                continue
            resource, value = line.split(separator, 1)
            resource = cls.normalize_resource(resource)
            if resource not in ALL_RESOURCES:
                continue
            try:
                value = int(str(value).strip())
            except (TypeError, ValueError):
                continue
            result[resource] = value if value >= 0 else -1
        return result

    @staticmethod
    def format_inventory(inventory, city=None):
        def display_value(resource):
            try:
                value = int(inventory.get(resource, -1))
            except (TypeError, ValueError):
                value = -1
            # -1 is the explicit OCR-unknown marker.  Do not turn it into 0;
            # zero means the game was successfully read and the stock is empty.
            return value if value >= 0 else -1

        return '\n'.join(
            f'{resource}: {display_value(resource)}'
            for resource in (
                tuple(resource for resource in resources_for_city(city)
                      if resource not in SHAOXING_MERCHANT_RESOURCES)
                if city == '绍兴府' else
                resources_for_city(city) if city else ALL_RESOURCES
            )
        )

    @staticmethod
    def format_merchant_inventory(inventory):
        lines = []
        for resource in SHAOXING_MERCHANT_RESOURCES:
            try:
                value = int(inventory.get(resource, -1))
            except (TypeError, ValueError):
                value = -1
            lines.append(f'{resource}: {value if value >= 0 else -1}')
        return '\n'.join(lines)

    @staticmethod
    def _inventory_key(city):
        return (
            f'JiangnanWarehouseInventory.'
            f'warehouse_{CITY_NAME_TO_KEY[city]}.Inventory'
        )

    @staticmethod
    def _merchant_inventory_key():
        return f'{JiangnanInventoryState.TASK}.warehouse_shaoxingfu.MerchantInventory'

    @staticmethod
    def _merchant_capacity_key():
        return f'{JiangnanInventoryState.TASK}.warehouse_shaoxingfu.MerchantCapacity'

    @staticmethod
    def _capacity_key(city):
        return (
            f'JiangnanWarehouseInventory.'
            f'warehouse_{CITY_NAME_TO_KEY[city]}.Capacity'
        )

    @staticmethod
    def _scheduled_changes_key(city):
        return (
            f'JiangnanWarehouseInventory.'
            f'warehouse_{CITY_NAME_TO_KEY[city]}.ScheduledChanges'
        )

    @staticmethod
    def _dock_key(city, field):
        return (
            f'JiangnanDockHarvest.'
            f'dock_{CITY_NAME_TO_KEY[city]}.{field}'
        )

    def load_inventories(self, cities):
        inventories = {}
        for city in cities:
            key = self._inventory_key(city)
            if key in self.config.modified:
                raw = self.config.modified[key]
            else:
                raw = self.config.cross_get(keys=key, default='')
            text = str(raw or '')
            merchant_text = ''
            if city == '绍兴府':
                merchant_key = self._merchant_inventory_key()
                merchant_text = str(self.config.modified.get(
                    merchant_key,
                    self.config.cross_get(keys=merchant_key, default=''),
                ) or '')
            cache_text = (text, merchant_text)
            cached = self._inventory_text_cache.get(city)
            if cached is not None and cached[0] == cache_text:
                inventory = cached[1]
            else:
                inventory = self.parse_inventory(text)
                if merchant_text:
                    inventory.update(self.parse_inventory(merchant_text))
                self._inventory_text_cache[city] = (cache_text, dict(inventory))
            inventories[city] = dict(inventory)
        return inventories

    def write_inventories(self, inventories, update=True):
        for city, inventory in inventories.items():
            text = self.format_inventory(inventory, city=city)
            self.config.modified[self._inventory_key(city)] = text
            merchant_text = ''
            if city == '绍兴府':
                merchant_text = self.format_merchant_inventory(inventory)
                self.config.modified[self._merchant_inventory_key()] = (
                    merchant_text
                )
            self._inventory_text_cache[city] = (
                (text, merchant_text), dict(inventory)
            )
        if update:
            self.config.update()

    def record_inventory_event(
        self, city, resource, amount, event_type, reason='',
        confirmed=False, event_id=None, now=None,
        metadata=None, save=True,
    ):
        """记录库存变化事件，不直接污染仓库实数。

        事件账本用于解决：生产完成、运输到达、人工领取、仓库扫描之间
        时间顺序无法确定的问题。真实库存只由确认事件或扫描基线决定。
        """
        resource = self.normalize_resource(resource)
        if city not in CITY_NAME_TO_KEY or resource not in ALL_RESOURCES:
            return False
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            return False
        if amount == 0:
            return False
        now = (now or datetime.now()).replace(microsecond=0)
        event = {
            'id': event_id or str(uuid4()),
            'city': city,
            'resource': resource,
            'amount': amount,
            'type': str(event_type),
            'reason': str(reason),
            'created_at': self._time_text(now),
            'confirmed_at': self._time_text(now) if confirmed else None,
            'confirmed': bool(confirmed),
            'included_in_scan': False,
            'metadata': dict(metadata or {}),
        }
        if event_id and (
            event_id in self.data.get('inventory_archived_event_ids', [])
            or any(e.get('id') == event_id for e in self.data.get('inventory_events', []))
        ):
            return False
        self.data.setdefault('inventory_events', []).append(event)
        if save:
            self.save()
        return True

    def _record_applied_change(self, city, resource, before, after, kind, reason='', now=None, delta=None, effective_at=None):
        """Audit actual applied deltas and seed replay from the known starting stock."""
        if before == after and delta is None:
            return
        snapshot = self.data.setdefault('inventory_snapshot', {}).setdefault(city, {'inventory': {}})
        tracked = snapshot.setdefault('delta_tracking_resources', [])
        if resource not in tracked:
            snapshot['inventory'][resource] = int(before)
            # Legacy writes predate delta tracking. Start from today's actual
            # baseline for this resource rather than replaying an incomplete history.
            for event in self.data.get('inventory_events', []):
                if event.get('city') == city and event.get('resource') == resource:
                    event['included_in_scan'] = True
            tracked.append(resource)
        self.record_inventory_event(
            city, resource, int(after)-int(before) if delta is None else delta, kind, reason=reason,
            confirmed=True, now=now, save=False,
            metadata={'before': int(before), 'after': int(after), 'applied': after >= 0,
                      'effective_at': self._time_text(
                          self._parse_time(effective_at) or now or datetime.now())},
        )
        if kind == 'live_correction':
            snapshot['inventory'][resource] = int(after)
            for event in self.data.get('inventory_events', []):
                if event.get('city') == city and event.get('resource') == resource:
                    event['included_in_scan'] = True

    def mark_snapshot_included_events(self, scan_time, cities=None):
        """标记已经被仓库实扫覆盖的历史事件，避免重复入账。"""
        scanned_cities = set(cities) if cities is not None else None
        for event in self.data.get('inventory_events', []):
            if scanned_cities is not None and event.get('city') not in scanned_cities:
                continue
            created = self._parse_time(event.get('created_at'))
            if created is not None and created <= scan_time:
                event['included_in_scan'] = True

    def _ensure_snapshot(self, city, inventory, now=None):
        now = (now or datetime.now()).replace(microsecond=0)
        snapshot = self.data.setdefault('inventory_snapshot', {})
        snapshot[city] = {
            'scan_time': self._time_text(now),
            'inventory': dict(inventory),
        }

    def _absorb_live_resource_shipments(self, city, resources, now):
        """Absorb arrived receipts covered by a resource-level live reading."""
        changed = False
        for shipment in self.data.get('shipments', []):
            if (shipment.get('status') != 'transporting'
                    or shipment.get('destination') != city
                    or self.normalize_resource(shipment.get('resource')) not in resources
                    or shipment.get('destination_credited_at')):
                continue
            finish = self._parse_time(shipment.get('finish_at'))
            if finish is None or finish > now:
                continue
            shipment['destination_credited_at'] = self._time_text(now)
            shipment['inventory_reconciliation'] = 'live_resource_scan_after_arrival'
            logger.info(
                f'Arrival already included in live resource stock: {city}, '
                f'{shipment.get("resource")} +{shipment.get("amount")}'
            )
            changed = True
        return changed

    def _wake_merchant_sales(self, inventory, now=None):
        """Wake only idle halls selling confirmed goods above the shared threshold."""
        task = 'JiangnanHuizhouMerchantSale'
        if not self.config.cross_get(f'{task}.Scheduler.Enable', default=False):
            return False
        now = (now or datetime.now()).replace(microsecond=0)
        try:
            threshold = max(1, int(self.config.cross_get(
                f'{task}.huizhou_merchant_settings.TradeThreshold', default=3000)))
        except (TypeError, ValueError):
            threshold = 3000
        resources = ('竹子', '茶叶', '桐油', '生漆', '汪笔', '徽墨', '澄纸', '歙石')
        ready = {r for r in resources if int(inventory.get(r, -1)) >= threshold}
        if not ready:
            return False
        changed = False
        for index in range(1, 5):
            prefix = f'{task}.huizhou_merchant_hall_{index}'
            if not self.config.cross_get(f'{prefix}.Enable', default=True):
                continue
            status = str(self.config.cross_get(f'{prefix}.Status', default=''))
            if any(marker in status for marker in ('销售中', '外出经商中', '实际时间读取失败', '经商时间识别失败', '收获失败', '收获待确认')):
                continue
            monopolies = {self.config.cross_get(f'{prefix}.Monopoly{i}', default='none') for i in range(1, 5)} & set(resources)
            if monopolies and not (monopolies & ready):
                continue
            # Unknown mapping needs discovery before a sale can be chosen.
            next_run = self._parse_time(self.config.cross_get(f'{prefix}.NextRun', default=''))
            if next_run and next_run > now and '等待复查' not in status:
                self.config.modified[f'{prefix}.NextRun'] = DEFAULT_TIME
                next_run = DEFAULT_TIME
            if next_run is None or next_run <= now:
                changed = True
        if changed:
            scheduled = self._parse_time(self.config.cross_get(f'{task}.Scheduler.NextRun', default=''))
            self.config.modified[f'{task}.Scheduler.NextRun'] = min(now, scheduled) if scheduled else now
        return changed

    def reconcile_live_inventory(self, city, readings, reason='', update=True):
        """Overwrite selected goods with authoritative numbers shown in game."""
        if city not in CITY_NAME_TO_KEY:
            logger.warning(f'Cannot reconcile inventory for unknown city: {city}')
            return False
        normalized = {}
        for resource, amount in dict(readings or {}).items():
            resource = self.normalize_resource(resource)
            try:
                amount = int(amount)
            except (TypeError, ValueError):
                continue
            if resource in ALL_RESOURCES and amount >= 0:
                normalized[resource] = amount
        if not normalized:
            logger.warning(
                f'No valid live inventory reading to reconcile: {readings}'
            )
            return False

        inventories = self.load_inventories((city,))
        inventory = inventories.setdefault(city, {})
        large_differences = []
        changed = any(int(inventory.get(resource, -1)) != amount for resource, amount in normalized.items())
        for resource, amount in normalized.items():
            previous = int(inventory.get(resource, -1))
            inventory[resource] = amount
            if previous != amount:
                self._record_applied_change(city, resource, previous, amount, 'live_correction', reason=reason)
                logger.warning(
                    f'Reconcile live inventory: {city} {resource} '
                    f'{previous} -> {amount}; reason={reason or "live UI"}'
                )
                if previous >= 0 and abs(previous-amount) >= max(100, amount // 5):
                    large_differences.append(f'{resource} {previous}->{amount}')
        now = datetime.now().replace(microsecond=0)
        reservations = getattr(self, '_production_cost_reservations', {}).get(city, {})
        for resource in normalized:
            reservations.pop(resource, None)
        absorbed = self._absorb_live_resource_shipments(city, set(normalized), now)
        if absorbed:
            self.sync_inventory_change_schedule(update=False)
        if city == '徽州府':
            self._wake_merchant_sales(inventory)
        self.write_inventories(inventories, update=update)
        self.resume_overflow_production(
            inventories, self.load_capacities((city,)), self.special_capacities(),
            now=now, update=update,
        )
        if changed or absorbed:
            self.save()
        if large_differences:
            logger.info(
                f'Live inventory correction applied without another full scan: '
                f'{city}, ' + ', '.join(large_differences)
            )
        return True

    def can_consume_inventory(self, city, costs):
        """按实测库存和本轮临时材料占用检查是否能开工，不修改仓库账面。"""
        if city not in CITY_NAME_TO_KEY:
            return False, {'城市': city}
        normalized = {}
        for resource, amount in dict(costs or {}).items():
            resource = self.normalize_resource(resource)
            try:
                amount = int(amount)
            except (TypeError, ValueError):
                amount = 0
            if resource not in ALL_RESOURCES or amount <= 0:
                return False, {str(resource): amount}
            normalized[resource] = normalized.get(resource, 0) + amount
        inventory = self.load_inventories((city,)).get(city, {})
        reserved = getattr(self, '_production_cost_reservations', {}).get(city, {})
        missing = {
            resource: amount - (int(inventory.get(resource, 0)) - reserved.get(resource, 0))
            for resource, amount in normalized.items()
            if int(inventory.get(resource, 0)) - reserved.get(resource, 0) < amount
        }
        return not missing, missing

    def consume_inventory(self, city, costs, event_type='production_consumption', reason=''):
        """校验材料；生产不扣账，商行等其他明确消耗照常扣账。"""
        normalized = {}
        for resource, amount in dict(costs or {}).items():
            resource = self.normalize_resource(resource)
            try:
                amount = int(amount)
            except (TypeError, ValueError):
                amount = 0
            if resource not in ALL_RESOURCES or amount <= 0:
                raise ValueError(
                    f'Invalid inventory consumption: {resource}, {amount}'
                )
            normalized[resource] = normalized.get(resource, 0) + amount
        if not normalized:
            return True
        if city not in CITY_NAME_TO_KEY:
            logger.warning(
                f'Insufficient production inventory in {city}: '
                f'{{"城市": {city!r}}}'
            )
            return False

        inventories = self.load_inventories((city,))
        inventory = inventories.setdefault(city, {})
        missing = {
            resource: amount - int(inventory.get(resource, 0))
            for resource, amount in normalized.items()
            if int(inventory.get(resource, 0)) < amount
        }
        if missing:
            logger.warning(
                f'Insufficient production inventory in {city}: {missing}'
            )
            return False

        if event_type == 'production_consumption':
            enough, missing = self.can_consume_inventory(city, normalized)
            if not enough:
                logger.warning(f'Insufficient unreserved production materials: {city}, {missing}')
                return False
            if not hasattr(self, '_production_cost_reservations'):
                self._production_cost_reservations = {}
            reserved = self._production_cost_reservations.setdefault(city, {})
            for resource, amount in normalized.items():
                reserved[resource] = reserved.get(resource, 0) + amount
            logger.info(f'Production inventory debit excluded: {city}, {normalized}')
            return True

        for resource, amount in normalized.items():
            before = int(inventory.get(resource, 0))
            inventory[resource] = before - amount
            self._record_applied_change(city, resource, before, inventory[resource], event_type, reason=reason)
            logger.info(
                f'Inventory consumed: {city}, '
                f'{resource} -{amount}, remain={inventory[resource]}'
            )
        self.write_inventories(inventories, update=False)
        self.resume_overflow_production(
            inventories,
            self.load_capacities((city,)),
            self.special_capacities(),
            update=False,
        )
        self.config.update()
        self.save()
        return True

    def load_capacities(self, cities):
        stored = self.capacities()
        result = {}
        for city in cities:
            value = self.config.cross_get(
                keys=self._capacity_key(city),
                default=stored.get(city, 0),
            )
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                result[city] = value
        return result

    def scan_interval_hours(self):
        value = self.config.cross_get(
            keys=f'{self.TASK}.WarehouseScan.Interval',
            default=12,
        )
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 12
        return value if value in (3, 6, 12, 24) else 12

    def next_scan_time(self):
        value = self.config.cross_get(
            keys=f'{self.TASK}.Scheduler.NextRun',
            default=DEFAULT_TIME,
        )
        return self._parse_time(value) or DEFAULT_TIME

    def scan_due(self, cities, now=None):
        """Only the configured periodic clock decides when to scan."""
        now = now or datetime.now()
        return self.next_scan_time() <= now

    def request_scan(self, reason, cities=None):
        """Legacy callers record uncertainty without advancing warehouse clocks.

        Changes are calculated from the inventory ledger. Baseline replacement
        waits for the configured periodic scan or an explicit user reset.
        """
        next_run = self.next_scan_time()
        self.data['inventory_review_pending'] = True
        self.data['scan_reason'] = str(reason)
        self.save()
        logger.warning(f'Inventory uncertainty recorded; keep periodic scan at {next_run}: {reason}')
        return next_run

    def request_allocation(self, cities, when=None, update=True):
        """库存发生变化时，只唤醒受影响城市的调配时钟。"""
        if not self.config.cross_get(
            keys='JiangnanCityAllocation.Scheduler.Enable',
            default=False,
        ):
            return False
        when = (when or datetime.now()).replace(microsecond=0)
        changed = False
        for city in set(cities):
            city_key = CITY_NAME_TO_KEY.get(city)
            if city_key is None:
                continue
            self.config.modified[
                f'JiangnanCityAllocation.{city_key}.NextRun'
            ] = when
            changed = True
        if changed:
            self.config.modified[
                'JiangnanCityAllocation.Scheduler.NextRun'
            ] = when
            if update:
                self.config.update()
        return changed

    def record_scan(self, inventories, capacities, special_capacities, now=None):
        now = (now or datetime.now()).replace(microsecond=0)
        next_run = now + timedelta(hours=self.scan_interval_hours())
        self.write_inventories(inventories, update=False)
        # 仓库实扫是最高可信度基线。扫描前已经发生的事件不再允许
        # 后续再次通过事件回放叠加到库存。
        for city, inventory in inventories.items():
            getattr(self, '_production_cost_reservations', {}).pop(city, None)
            self._ensure_snapshot(city, inventory, now=now)
            # Persist the per-city scan clock with the inventory baseline.
            # The caller used to issue another complete config write for it.
            city_prefix = self._inventory_key(city).rsplit('.', 1)[0]
            self.config.modified[f'{city_prefix}.NextRun'] = next_run
        # 到达时间已过的运输若被仓库实扫覆盖，直接采用游戏中读到的
        # 数值。随后到期任务和码头收获都不能再按订单量重复加一次。
        for shipment in self.data.get('shipments', []):
            if (shipment.get('status') != 'transporting'
                    or shipment.get('destination') not in inventories):
                continue
            finish = self._parse_time(shipment.get('finish_at'))
            if shipment.get('destination_credited_at') or (
                finish is not None and finish <= now
            ):
                if not shipment.get('destination_credited_at'):
                    shipment['destination_credited_at'] = self._time_text(now)
                shipment['inventory_reconciliation'] = 'warehouse_scan_after_arrival'
        self.mark_snapshot_included_events(now, cities=inventories)
        for city, capacity in capacities.items():
            self.config.modified[self._capacity_key(city)] = int(capacity)
        if '绍兴府' in special_capacities:
            merchant_capacity = special_capacities['绍兴府'].get('绍兴商行')
            if merchant_capacity is not None and int(merchant_capacity) > 0:
                self.config.modified[self._merchant_capacity_key()] = int(
                    merchant_capacity
                )
        self.config.modified[f'{self.TASK}.Scheduler.NextRun'] = next_run
        # 满仓时由生产模块自动关闭的城市开关，只能由新一轮仓库实扫
        # 重新打开。先写入新库存，再判断“满仓 -> 不满仓”，让恢复后的
        # 生产任务读取到的就是本轮基线。
        self.resume_overflow_production(
            inventories,
            capacities,
            special_capacities,
            now=now,
            update=False,
        )
        # 新实扫数据写入后，只唤醒已开启的调配任务；扫描任务本身不依赖
        # 调配是否开启，二者通过配置库存这一层交换数据。
        self.request_allocation(inventories.keys(), when=now, update=False)
        if '徽州府' in inventories:
            self._wake_merchant_sales(inventories['徽州府'], now=now)
        self.config.update()
        self.data['last_scan_at'] = self._time_text(now)
        self.data['next_scan_at'] = self._time_text(next_run)
        self.data['inventory_review_pending'] = False
        stored_capacities = self.data.setdefault('capacities', {})
        stored_capacities.update({
            city: int(capacity) for city, capacity in capacities.items()
        })
        # 绍兴商行具有独立的单项容量，保留实扫读到的专用上限。
        stored_special = self.data.setdefault('special_capacities', {})
        self.data.pop('merchant_special_inventory', None)
        for city, capacity in capacities.items():
            groups = {
                '徽州特产1': int(capacity),
                '徽州特产2': int(capacity),
            }
            groups.update({
                name: int(value)
                for name, value in special_capacities.get(city, {}).items()
                if int(value) > 0
            })
            stored_special[city] = groups
        self.save()
        logger.attr('WarehouseInventoryNextRun', next_run)
        return next_run

    @staticmethod
    def _production_config_paths(task, group_prefix, city_key):
        if task == 'JiangnanImprovedFarmProduction':
            group_name = f'improved_farm_{city_key}'
        elif task == 'JiangnanRestaurantProduction' and city_key == 'huizhoufu':
            group_name = 'restaurant_huizhoufu'
        else:
            group_name = f'{group_prefix}_{city_key}'
        base = f'{task}.{group_name}'
        return (
            f'{base}.Enable',
            f'{base}.NextRun',
            f'{CITY_KEY_TO_PRODUCTION_TASK[city_key]}.Scheduler.NextRun',
        )

    def _pause_production_target(
        self, city, resources, task, group_prefix, city_key,
        reason='仓库已满',
    ):
        """关闭一个生产任务的城市开关，并登记可逆的满仓原因。"""
        resources = tuple(dict.fromkeys(
            self.normalize_resource(resource) for resource in resources
            if resource
        ))
        if not resources:
            return False
        enable_key, next_run_key, _ = self._production_config_paths(
            task, group_prefix, city_key,
        )
        # 已由用户或“无此建筑”逻辑关闭的任务绝不能登记成满仓暂停，
        # 否则库存下降后会错误地把它重新打开。
        enabled = self.config.cross_get(keys=enable_key, default=True)
        if not bool(enabled):
            return False

        paused = self.data.setdefault('overflow_paused_production', [])
        existing = next((
            item for item in paused
            if item.get('task') == task and item.get('city') == city
        ), None)
        marker = {
            'city': city,
            'city_key': city_key,
            'task': task,
            'group_prefix': group_prefix,
            'resources': list(resources),
            'paused_at': self._time_text(datetime.now()),
        }
        if existing is None:
            paused.append(marker)
        else:
            existing.update(marker)
        self.config.modified[enable_key] = False
        self.config.modified[next_run_key] = DEFAULT_TIME
        status_key = next_run_key.rsplit('.', 1)[0] + '.Status'
        self.config.modified[status_key] = (
            f'[{"/".join(resources)}][{reason}][已自动暂停]'
        )
        logger.warning(
            f'Production paused: {task} in {city}, reason={reason}, '
            f'resources={list(resources)}'
        )
        return True

    def pause_city_resource_production(
        self, city, resource, task, group_prefix='production', update=True,
        reason='仓库已满',
    ):
        """只关闭本次已经确认处于溢出状态的生产设施任务。"""
        city_key = CITY_NAME_TO_KEY.get(city)
        resource = self.normalize_resource(resource)
        if city_key is None or not resource or not task:
            return False
        if resource not in ALL_RESOURCES:
            logger.warning(
                f'Cannot auto-pause production for unscanned resource: '
                f'{city} {resource}'
            )
            return False
        changed = self._pause_production_target(
            city, (resource,), task, group_prefix, city_key, reason=reason,
        )
        if changed:
            self.save()
            if update:
                self.config.update()
        return changed

    def pause_multi_product_production(
        self, city, resources, task, update=True, reason='仓库已满'
    ):
        """全部产物满仓时暂停一个多产物设施。"""
        city_key = CITY_NAME_TO_KEY.get(city)
        if city_key is None:
            return False
        resources = tuple(
            self.normalize_resource(resource) for resource in resources
        )
        if not resources or any(
            resource not in ALL_RESOURCES for resource in resources
        ):
            logger.warning(
                f'Cannot auto-pause {task} in {city}: '
                f'warehouse does not scan every product {list(resources)}'
            )
            return False
        changed = self._pause_production_target(
            city, resources, task, 'production', city_key, reason=reason,
        )
        if changed:
            self.save()
            if update:
                self.config.update()
        return changed

    @staticmethod
    def _special_capacity_group(resource):
        groups = {
            '徽州特产1': ('茶油', '竹编', '甲酒', '盆景'),
            '徽州特产2': ('木雕', '竹雕', '砖雕', '石雕'),
        }
        return next((
            (name, members) for name, members in groups.items()
            if resource in members
        ), (None, ()))

    def _resource_is_not_full(
        self, city, resource, inventory, capacity, special_capacity
    ):
        """返回 True/False；数据不足时返回 None，禁止猜测恢复。"""
        value = inventory.get(resource)
        if value is None or int(value) < 0:
            return None
        group_name, members = self._special_capacity_group(resource)
        if city == '绍兴府' and resource in SHAOXING_MERCHANT_RESOURCES:
            try:
                limit = int((special_capacity or {}).get('绍兴商行', 0))
            except (TypeError, ValueError):
                return None
            return (int(value) < limit * PRODUCTION_THRESHOLD_RATIO
                    if limit > 0 else None)
        if group_name:
            try:
                limit = int((special_capacity or {}).get(group_name, 0))
                values = [int(inventory.get(item, -1)) for item in members]
            except (TypeError, ValueError):
                return None
            if limit <= 0 or any(value < 0 for value in values):
                return None
            return sum(values) < limit * PRODUCTION_THRESHOLD_RATIO
        try:
            limit = int(capacity)
        except (TypeError, ValueError):
            return None
        if limit <= 0:
            return None
        return int(value) < limit * PRODUCTION_THRESHOLD_RATIO

    def production_threshold_states(self, city, resources):
        """批量判断产物是否低于统一的 97% 生产阈值。"""
        inventories = self.load_inventories((city,))
        # Production and automatic resume must use the same confirmed stock.
        # Pending shipments or unfinished output belong to the allocation
        # forecast; they must not pause an actually underfilled warehouse.
        confirmed = inventories.get(city, {})
        capacities = self.load_capacities((city,))
        special_capacities = self.special_capacities()
        results = {}
        for resource in resources:
            normalized = self.normalize_resource(resource)
            results[resource] = self._resource_is_not_full(
                city,
                normalized,
                confirmed,
                capacities.get(city),
                special_capacities.get(city, {}),
            )
        logger.attr(
            'ProductionThreshold',
            f'{city}: below_97_percent={results}',
        )
        return results

    def production_below_threshold(self, city, resource):
        return self.production_threshold_states(city, (resource,)).get(resource)

    def resume_overflow_production(
        self, inventories, capacities, special_capacities,
        now=None, update=True,
    ):
        """实扫确认库存出现空间后，恢复对应城市的自动暂停设施。"""
        now = (now or datetime.now()).replace(microsecond=0)
        paused = list(self.data.get('overflow_paused_production', []))
        remaining = []
        resumed = []
        scanned_cities = set(inventories)
        for marker in paused:
            city = marker.get('city')
            if city not in scanned_cities:
                remaining.append(marker)
                continue
            inventory = inventories.get(city) or {}
            resources = marker.get('resources') or ()
            states = [
                self._resource_is_not_full(
                    city,
                    resource,
                    inventory,
                    capacities.get(city),
                    special_capacities.get(city, {}),
                )
                for resource in resources
            ]
            # 单产物设施在 A 不满时恢复；多产物设施只要任一产物组出现
            # 空间就恢复，由模块自己的缺口策略决定下一种生产什么。
            if not any(state is True for state in states):
                remaining.append(marker)
                continue
            task = marker.get('task')
            city_key = marker.get('city_key') or CITY_NAME_TO_KEY.get(city)
            group_prefix = marker.get('group_prefix', 'production')
            if not task or city_key is None:
                remaining.append(marker)
                continue
            enable_key, next_run_key, scheduler_key = (
                self._production_config_paths(task, group_prefix, city_key)
            )
            resume_status = (
                f'[{"/".join(resources)}][库存低于97%][已自动开启]'
            )
            self.config.modified[enable_key] = True
            self.config.modified[next_run_key] = now
            self.config.modified[scheduler_key] = now
            city_task = CITY_KEY_TO_PRODUCTION_TASK.get(city_key)
            if city_task:
                city_facility = (
                    f'{city_task}.city_facility_{task}'
                )
                # The city runner uses its own per-facility clock. Waking
                # only the legacy runner leaves this facility asleep there.
                self.config.modified[f'{city_facility}.NextRun'] = now
                self.config.modified[f'{city_facility}.Status'] = resume_status
                self.config.modified[f'{city_facility}.Enable'] = True
                self.config.modified[f'{city_task}.Scheduler.NextRun'] = now
            status_key = next_run_key.rsplit('.', 1)[0] + '.Status'
            self.config.modified[status_key] = resume_status
            resumed.append((city, task, list(resources)))
            logger.info(
                f'Warehouse has space again: resume {task} in {city}, '
                f'resources={list(resources)}'
            )
        if resumed:
            self.data['overflow_paused_production'] = remaining
            self.save()
            if update:
                self.config.update()
        return resumed

    def capacities(self):
        return dict(self.data.get('capacities', {}))

    def special_capacities(self):
        return {
            city: dict(groups)
            for city, groups in self.data.get('special_capacities', {}).items()
        }

    def record_departures(
        self,
        orders,
        slot_indices,
        finish_times=None,
        finish_snapshot=None,
    ):
        """扣除来源库存，并登记来源城市的具体船位。"""
        if not orders:
            return
        cities = set()
        for order in orders:
            cities.add(order.source)
            cities.add(order.destination)
        inventories = self.load_inventories(cities)
        now = datetime.now().replace(microsecond=0)
        finish_times = list(finish_times or ())
        for order_index, (order, slot_index) in enumerate(
            zip(orders, slot_indices)
        ):
            source = inventories.setdefault(order.source, {})
            before = int(source.get(order.resource, -1))
            source[order.resource] = before-int(order.amount) if before >= int(order.amount) else -1
            self._record_applied_change(
                order.source, order.resource, before, source[order.resource], 'transport_departure',
                reason=f'{order.source}->{order.destination}, slot={slot_index}',
                now=now, delta=-int(order.amount),
            )
            if source[order.resource] < 0:
                self.request_scan('departed cargo exceeds confirmed source stock', cities={order.source})
            shipment = {
                'source': order.source,
                'slot': int(slot_index),
                'destination': order.destination,
                'resource': order.resource,
                'amount': int(order.amount),
                'departed_at': self._time_text(now),
                'finish_at': (
                    self._time_text(finish_times[order_index])
                    if order_index < len(finish_times)
                    and finish_times[order_index] is not None
                    else None
                ),
                'dock_next_at': (
                    self._time_text(finish_times[order_index])
                    if order_index < len(finish_times)
                    and finish_times[order_index] is not None
                    else None
                ),
                # 来源库存发船即扣除；目的库存到达时间一到就入账一次。
                'destination_credited_at': None,
                'status': 'transporting',
            }
            self.data.setdefault('shipments', []).append(shipment)
        self.write_inventories(inventories, update=False)
        self.resume_overflow_production(
            inventories,
            self.load_capacities(cities),
            self.special_capacities(),
            now=now,
            update=False,
        )
        self.config.update()
        if finish_snapshot:
            self.assign_finish_times(finish_snapshot)
        self.save()
        # 一次 update 同时写入仓库的定时增加明细与码头节点。
        self.sync_inventory_change_schedule(update=False)
        self.sync_dock_schedule()

    def assign_finish_times(self, snapshot):
        by_slot = {
            (city, int(slot['slot'])): slot.get('finish_time')
            for city, slots in snapshot.items()
            for slot in slots
            if slot.get('state') == 'transporting'
        }
        for shipment in self.data.get('shipments', []):
            if shipment.get('status') != 'transporting':
                continue
            finish = by_slot.get((shipment.get('source'), int(shipment.get('slot', -1))))
            # 新订单的 finish_at 来自点击“出发”前的运输时间识别，它是
            # 库存入账、码头收获和调配冷却的统一基准，不再被总览页的
            # 分钟级 OCR 覆盖。这里只为旧台账补缺失时间。
            if finish is not None and not shipment.get('finish_at'):
                shipment['finish_at'] = self._time_text(finish)
                shipment['dock_next_at'] = self._time_text(finish)
            else:
                slots = snapshot.get(shipment.get('source'), [])
                slot = next(
                    (
                        item for item in slots
                        if int(item.get('slot', -1))
                        == int(shipment.get('slot', -1))
                    ),
                    None,
                )
                # 台账认为船仍在运输，但游戏里船位已经空闲，通常表示
                # 用户在脚本外手动收取过。目标物资去向无法可靠推算，
                # 下一轮必须实扫，而不是继续展示错误估算。
                if slot is not None and slot.get('state') == 'idle':
                    self.data['inventory_review_pending'] = True
                    logger.warning(
                        f'Active shipment disappeared from slot: '
                        f'{shipment.get("source")} slot '
                        f'{int(shipment.get("slot", -1)) + 1}; '
                        f'keep periodic warehouse schedule'
                    )
        self.save()

    def project_for_planning(self, inventories):
        """把在途物资计入目标城市，避免重复向同一缺口发货。"""
        projected = {
            city: dict(inventory) for city, inventory in inventories.items()
        }
        for shipment in self.data.get('shipments', []):
            if shipment.get('status') != 'transporting':
                continue
            if shipment.get('destination_credited_at'):
                # 兼容旧台账：已经进入确认库存的运输不能再次投影。
                continue
            city = shipment.get('destination')
            resource = shipment.get('resource')
            if city not in projected or resource not in ALL_RESOURCES:
                continue
            if int(projected[city].get(resource, -1)) < 0:
                continue
            projected[city][resource] = (
                int(projected[city].get(resource, 0))
                + int(shipment.get('amount', 0))
            )
        # 生产计时保留供任务调度使用，但不再投影为预计库存。
        for change in self.data.get('scheduled_changes', []):
            if self._production_inventory_change(change):
                continue
            if change.get('credited_at'):
                continue
            city = change.get('city')
            resource = self.normalize_resource(change.get('resource'))
            amount = int(change.get('amount', 0))
            if city not in projected or resource not in ALL_RESOURCES or amount <= 0:
                continue
            if int(projected[city].get(resource, -1)) < 0:
                continue
            projected[city][resource] = (
                int(projected[city].get(resource, 0)) + amount
            )
        return projected

    def _credit_inventory(self, city, resource, amount, reason, now=None, effective_at=None):
        """Apply one confirmed positive inventory delta to the actual ledger."""
        now = (now or datetime.now()).replace(microsecond=0)
        resource = self.normalize_resource(resource)
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            amount = 0
        if city not in CITY_NAME_TO_KEY or resource not in ALL_RESOURCES or amount <= 0:
            logger.warning(
                f'Invalid confirmed inventory credit: {city}, {resource}, {amount}'
            )
            self.request_scan(f'invalid confirmed inventory credit: {reason}')
            return False

        inventories = self.load_inventories((city,))
        inventory = inventories.setdefault(city, {})
        current = int(inventory.get(resource, -1))
        if current < 0:
            inventory[resource] = -1
            self.data['inventory_review_pending'] = True
            self.data['scan_reason'] = f'{city}{resource}{reason}时原库存未知'
            logger.warning(
                f'Confirmed inventory credit cannot update unknown stock: '
                f'{city}, {resource}; reason={reason}'
            )
        else:
            inventory[resource] = current + amount
            logger.info(
                f'Confirmed inventory credit: {city}, {resource} '
                f'{current} + {amount} = {inventory[resource]}; reason={reason}'
            )
        self._record_applied_change(
            city, resource, current, inventory[resource], 'confirmed_receipt',
            reason=reason, now=now, delta=amount, effective_at=effective_at,
        )
        if city == '徽州府':
            self._wake_merchant_sales(inventory, now=now)
        self.write_inventories(inventories, update=False)
        return True

    def _invalidate_inventory(self, city, resource, reason, now=None):
        """Mark one stock value unknown when event ordering cannot be proven."""
        now = (now or datetime.now()).replace(microsecond=0)
        resource = self.normalize_resource(resource)
        if city not in CITY_NAME_TO_KEY or resource not in ALL_RESOURCES:
            self.request_scan(f'invalid inventory uncertainty: {reason}')
            return False
        inventories = self.load_inventories((city,))
        inventories.setdefault(city, {})[resource] = -1
        self.write_inventories(inventories, update=False)
        self.data['inventory_review_pending'] = True
        self.data['scan_reason'] = f'{city}{resource}: {reason}'
        logger.warning(
            f'Inventory invalidated until live scan: {city}, {resource}; '
            f'reason={reason}'
        )
        return True

    def capture_tax_popup(self, resources, city_hint=None):
        """Persist a popup token before dismissing it, without guessing its city."""
        resources = {str(key): int(value) for key, value in dict(resources).items()}
        records = self.data.setdefault('tax_popup_receipts', [])
        for receipt in reversed(records):
            if (not receipt.get('closed_at') and receipt.get('resources') == resources
                    and receipt.get('city_hint') == city_hint):
                return receipt['id']
        receipt = {
            'id': str(uuid4()), 'resources': resources, 'city_hint': city_hint,
            'created_at': self._time_text(datetime.now()), 'phase': 'captured',
        }
        records.append(receipt)
        self.save()
        return receipt['id']

    def apply_tax_popup_receipt(self, city, resources, receipt_id):
        """Credit a confirmed popup exactly once, including interrupted writes."""
        if city not in CITY_NAME_TO_KEY:
            return False
        records = self.data.setdefault('tax_popup_receipts', [])
        receipt = next((item for item in records if item['id'] == receipt_id), None)
        if receipt is None:
            raise ValueError('Tax receipt must be captured before it is applied')
        if {str(key): int(value) for key, value in dict(resources).items()} != receipt['resources']:
            raise ValueError('Tax receipt resources changed during recovery')
        if receipt.get('phase') == 'applied':
            return False
        normalized = {
            self.normalize_resource(key): int(value)
            for key, value in dict(resources).items()
            if self.normalize_resource(key) in ALL_RESOURCES and int(value) > 0
        }
        if receipt.get('phase') == 'prepared':
            if receipt.get('city') != city:
                raise ValueError('Tax receipt city changed during recovery')
            inventories = self.load_inventories((city,))
            inventory = inventories[city]
            for resource, before in receipt['before'].items():
                after = receipt['after'][resource]
                current = int(inventory.get(resource, -1))
                if current in (before, after):
                    inventory[resource] = after
                else:
                    # Another observation superseded the prepared transaction.
                    # Never add twice or overwrite it with an old stock value.
                    inventory[resource] = -1
                if inventory[resource] < 0:
                    self.data['inventory_review_pending'] = True
                    self.data['scan_reason'] = '税课司收益入账中断，需实扫确认'
            self.write_inventories(inventories, update=False)
        else:
            before = self.load_inventories((city,))[city]
            receipt.update(city=city, phase='prepared', before={
                resource: int(before.get(resource, -1)) for resource in normalized
            })
            for resource, amount in normalized.items():
                self._credit_inventory(city, resource, amount, reason='税课司离线收益')
            after = self.load_inventories((city,))[city]
            receipt['after'] = {resource: int(after.get(resource, -1)) for resource in normalized}
        # Save receipt + audit events before publishing the new UI inventory.
        self.save()
        if normalized:
            self.request_allocation((city,), update=False)
        self.config.update()
        receipt['phase'] = 'applied'
        receipt['applied_at'] = self._time_text(datetime.now())
        self.save()
        logger.info(f'Tax popup inventory credited: {city}, {normalized}, receipt={receipt_id}')
        return True

    def close_tax_popup_receipt(self, receipt_id, reason='closed'):
        for receipt in self.data.get('tax_popup_receipts', []):
            if receipt.get('id') == receipt_id and not receipt.get('closed_at'):
                receipt['closed_at'] = self._time_text(datetime.now())
                receipt['close_reason'] = reason
                self.save()
                return True
        return False

    def _confirm_scheduled_change(
        self, change, observed_at, reason, included_in_scan=False,
    ):
        """Credit non-production receipts only; production clocks are excluded."""
        if self._production_inventory_change(change):
            return False
        observed_at = (
            self._parse_time(observed_at)
            or datetime.now().replace(microsecond=0)
        )
        if change.get('credited_at'):
            return True
        city = change.get('city')
        resource = self.normalize_resource(change.get('resource'))
        amount = int(change.get('amount', 0))
        if included_in_scan:
            # A live warehouse read is already the final stock value. Adding
            # the scheduled production amount again double-counts a harvest.
            credited = True
            logger.info(
                f'Production harvest covered by live warehouse scan: '
                f'{city}, {resource} +{amount}'
            )
        else:
            credited = self._credit_inventory(
                city, resource, amount, reason=reason, now=observed_at
            )
        change['credited_at'] = self._time_text(observed_at)
        change['credited_reason'] = str(reason)
        self.request_allocation((city,), when=observed_at, update=False)
        self.save()
        self.sync_inventory_change_schedule(update=False)
        self.config.update()
        return credited

    def apply_due_arrivals(self, now=None, wake_dock=True):
        """运输到达给目的城入账一次；生产计时不参与库存计算。"""
        now = (now or datetime.now()).replace(microsecond=0)
        due_shipments = []
        due_production = []
        credited_destinations = set()

        for shipment in self.data.get('shipments', []):
            if shipment.get('status') != 'transporting':
                continue
            finish = self._parse_time(shipment.get('finish_at'))
            if finish is None or finish > now:
                continue
            resource = self.normalize_resource(shipment.get('resource'))
            amount = int(shipment.get('amount', 0))
            if resource not in ALL_RESOURCES or amount <= 0:
                self.data['inventory_review_pending'] = True
                self.data['scan_reason'] = 'invalid due shipment'
                logger.warning('Invalid due shipment; keep periodic warehouse schedule')
                continue
            due_shipments.append(shipment)
            if not shipment.get('destination_credited_at'):
                destination = shipment.get('destination')
                if self._credit_inventory(
                    destination, resource, amount,
                    reason='运输到达时间已到', now=now,
                    effective_at=finish,
                ):
                    shipment['destination_credited_at'] = self._time_text(now)
                    shipment['inventory_reconciliation'] = 'arrival_time'
                    if destination in CITY_NAME_TO_KEY:
                        credited_destinations.add(destination)

        for change in self.data.get('scheduled_changes', []):
            if self._production_inventory_change(change):
                continue
            if change.get('credited_at'):
                continue
            execute_at = self._parse_time(change.get('execute_at'))
            if execute_at is None or execute_at > now:
                continue
            city = change.get('city')
            resource = self.normalize_resource(change.get('resource'))
            amount = int(change.get('amount', 0))
            if (
                city not in CITY_NAME_TO_KEY
                or resource not in ALL_RESOURCES
                or amount <= 0
            ):
                self.data['inventory_review_pending'] = True
                self.data['scan_reason'] = 'invalid scheduled inventory change'
                logger.warning('Invalid scheduled inventory change; keep periodic warehouse schedule')
                continue
            if not change.get('ready_at'):
                change['ready_at'] = self._time_text(now)
            due_production.append(change)

        if due_shipments and wake_dock:
            self.request_dock_harvest(
                now=now, shipments=due_shipments, update=False
            )
        if credited_destinations:
            self.request_allocation(
                credited_destinations, when=now, update=False,
            )
        if due_shipments or due_production:
            self.save()
            self.sync_inventory_change_schedule()
        return len(due_shipments) + len(due_production)

    def reconcile_production_harvest(
        self,
        city,
        resource,
        amount,
        reference,
        expected_at=None,
        observed_at=None,
        included_in_scan=False,
    ):
        """保留生产状态回调；收获、补账和未知产量均不改仓库账面。

        真实仓库扫描仍按实测值更新；材料不足和97%阈值由生产模块
        原有检查负责，不依赖本方法推算收获量。
        """
        return 'excluded_from_inventory'

    def schedule_inventory_change(
        self,
        city,
        resource,
        amount,
        execute_at,
        reason='生产',
        reference=None,
    ):
        """登记一笔未来库存增加，供生产等模块复用。"""
        resource = self.normalize_resource(resource)
        execute_at = self._parse_time(execute_at)
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            amount = 0
        if (
            city not in CITY_NAME_TO_KEY
            or resource not in ALL_RESOURCES
            or amount <= 0
            or execute_at is None
        ):
            raise ValueError(
                f'Invalid inventory change: {city}, {resource}, '
                f'{amount}, {execute_at}'
            )
        item = {
            'city': city,
            'resource': resource,
            'amount': amount,
            'execute_at': self._time_text(execute_at),
            'reason': str(reason or '生产'),
            'reference': str(reference) if reference else None,
            'credited_at': None,
        }
        existing = None
        if reference:
            existing = next((
                change for change in self.data.get('scheduled_changes', [])
                if not change.get('credited_at')
                and change.get('reference') == str(reference)
            ), None)
        if existing is None:
            self.data.setdefault('scheduled_changes', []).append(item)
        else:
            if all(existing.get(key) == value for key, value in item.items()):
                # An identical countdown observation adds no state change.
                # Still flush pending UI/config changes from the caller.
                self.sync_inventory_change_schedule()
                return
            # 生产中倒计时可能被加速；再次扫描同一建筑时
            # 更正原节点，而不是创建重复的库存增加。
            existing.update(item)
        self.save()
        self.sync_inventory_change_schedule()

    def cancel_inventory_change(self, reference, rollback_credited=False):
        """取消建筑对应的定时增加；仓满时可撤销已到时的账面入库。"""
        reference = str(reference or '')
        if not reference:
            return False
        changed = False
        inventories = {}
        kept = []
        for item in self.data.get('scheduled_changes', []):
            if item.get('reference') != reference:
                kept.append(item)
                continue
            changed = True
            if (rollback_credited and item.get('credited_at')
                    and not self._production_inventory_change(item)):
                city = item.get('city')
                resource = self.normalize_resource(item.get('resource'))
                amount = max(0, int(item.get('amount', 0)))
                if city in CITY_NAME_TO_KEY and resource in ALL_RESOURCES:
                    if city not in inventories:
                        inventories.update(self.load_inventories((city,)))
                    inventory = inventories.setdefault(city, {})
                    inventory[resource] = max(
                        0, int(inventory.get(resource, 0)) - amount
                    )
                    logger.info(
                        f'Rollback uncollected production inventory: '
                        f'{city}, {resource} -{amount}'
                    )
        if not changed:
            return False
        self.data['scheduled_changes'] = kept
        if inventories:
            self.write_inventories(inventories, update=False)
        self.save()
        self.sync_inventory_change_schedule()
        return True

    def pending_inventory_changes(self, city=None):
        """返回尚未入账的增加节点，运输与通用节点统一展示。"""
        result = []
        for shipment in self.data.get('shipments', []):
            destination = shipment.get('destination')
            if (
                shipment.get('status') != 'transporting'
                or shipment.get('destination_credited_at')
                or (city is not None and destination != city)
            ):
                continue
            execute_at = self._parse_time(shipment.get('finish_at'))
            resource = self.normalize_resource(shipment.get('resource'))
            amount = int(shipment.get('amount', 0))
            if execute_at is None or resource not in ALL_RESOURCES or amount <= 0:
                continue
            result.append({
                'city': destination,
                'resource': resource,
                'amount': amount,
                'execute_at': execute_at,
                'reason': '调配',
            })
        for change in self.data.get('scheduled_changes', []):
            if self._production_inventory_change(change):
                continue
            destination = change.get('city')
            if (
                change.get('credited_at')
                or (city is not None and destination != city)
            ):
                continue
            execute_at = self._parse_time(change.get('execute_at'))
            resource = self.normalize_resource(change.get('resource'))
            amount = int(change.get('amount', 0))
            if execute_at is None or resource not in ALL_RESOURCES or amount <= 0:
                continue
            result.append({
                'city': destination,
                'resource': resource,
                'amount': amount,
                'execute_at': execute_at,
                'reason': str(change.get('reason') or '生产'),
            })
        return sorted(result, key=lambda item: item['execute_at'])

    def sync_inventory_change_schedule(self, update=True):
        """把每座城市的定时增加明细同步到仓库 UI。"""
        # Scan and sort the ledger once, then distribute its entries by city.
        # Repeating the full scan for each city grows costly on long runs.
        grouped = {city: [] for city in CITY_NAME_TO_KEY}
        for item in self.pending_inventory_changes():
            if item['city'] in grouped:
                grouped[item['city']].append(item)
        for city in CITY_NAME_TO_KEY:
            lines = [
                f'[{item["resource"]}][+{item["amount"]}]'
                f'[{self._time_text(item["execute_at"])}]'
                f'[{item["reason"]}]'
                for item in grouped[city]
            ]
            key = self._scheduled_changes_key(city)
            text = '\n'.join(lines) if lines else '无'
            current = (self.config.modified[key] if key in self.config.modified
                       else self.config.cross_get(keys=key, default=''))
            if current != text:
                self.config.modified[key] = text
        # Some callers also queue inventory or scheduler changes before this
        # sync. Flush those even if the display itself has not changed.
        if update and self.config.modified:
            self.config.update()

    def apply_collected(self, collected_slots):
        """Consume dock nodes; known deliveries are credited at arrival or here.

        Tuple item 3 is the existing ``discarded`` flag.  Optional item 4 marks
        a slot merely observed empty. It still proves a recorded shipment has
        finished, even when the exact collection moment is unknown.
        """
        if not collected_slots:
            return 0, 0
        active = self.data.get('shipments', [])
        matched = []
        matched_ids = set()
        unknown = 0
        discarded_shipments = set()
        observed_only_shipments = set()
        for collected_item in collected_slots:
            city, slot_index = collected_item[:2]
            discarded = bool(collected_item[2]) if len(collected_item) > 2 else False
            observed_only = (
                bool(collected_item[3]) if len(collected_item) > 3 else False
            )
            candidates = [
                shipment for shipment in active
                if shipment.get('status') == 'transporting'
                and shipment.get('source') == city
                and int(shipment.get('slot', -1)) == int(slot_index)
            ]
            if not candidates:
                unknown += 1
                continue
            candidates.sort(key=lambda item: item.get('departed_at', ''))
            shipment = candidates[0]
            if id(shipment) not in matched_ids:
                matched.append(shipment)
                matched_ids.add(id(shipment))
            if discarded:
                discarded_shipments.add(id(shipment))
                self.data['inventory_review_pending'] = True
            if observed_only:
                observed_only_shipments.add(id(shipment))

        now = datetime.now().replace(microsecond=0)
        affected_cities = set()
        for shipment in matched:
            finish = self._parse_time(shipment.get('finish_at'))
            if finish is None or finish > now:
                shipment['finish_at'] = self._time_text(now)
                shipment['dock_next_at'] = self._time_text(now)

            source = shipment.get('source')
            destination = shipment.get('destination')
            affected_cities.update((source, destination))
            discarded = id(shipment) in discarded_shipments
            observed_only = id(shipment) in observed_only_shipments
            if discarded:
                # 仓满丢弃时实际接收数量未知，绝不能按订单量硬加。
                self._invalidate_inventory(
                    destination,
                    shipment.get('resource'),
                    reason='码头仓满，实际接收数量未知',
                    now=now,
                )
                shipment['inventory_reconciliation'] = 'scan_required_discard'
            elif not shipment.get('destination_credited_at'):
                self._credit_inventory(
                    destination,
                    shipment.get('resource'),
                    shipment.get('amount', 0),
                    reason=(
                        '运输船位已空' if observed_only else '码头收获确认'
                    ),
                    now=now,
                )
                shipment['destination_credited_at'] = self._time_text(now)
                shipment['inventory_reconciliation'] = (
                    'observed_empty_slot' if observed_only
                    else 'confirmed_collection'
                )

            shipment['status'] = 'collected'
            shipment['collected_at'] = self._time_text(now)

        self.request_allocation(
            (city for city in affected_cities if city in CITY_NAME_TO_KEY),
            when=now,
            update=False,
        )
        self.save()
        self.sync_inventory_change_schedule(update=False)
        self.config.update()
        return len(matched), unknown

    def active_shipments(self, city=None, due_before=None):
        result = []
        for shipment in self.data.get('shipments', []):
            if shipment.get('status') != 'transporting':
                continue
            if city is not None and shipment.get('destination') != city:
                continue
            dock_next = self._parse_time(
                shipment.get('dock_next_at') or shipment.get('finish_at')
            )
            if due_before is not None and (
                dock_next is None or dock_next > due_before
            ):
                continue
            result.append(shipment)
        return sorted(
            result,
            key=lambda item: (
                item.get('dock_next_at')
                or item.get('finish_at')
                or '9999-12-31 23:59:59'
            ),
        )

    def due_dock_shipments(self, now=None):
        """返回所有目的城市中已到期且启用码头收获的节点。"""
        now = (now or datetime.now()).replace(microsecond=0)
        return [
            shipment for shipment in self.active_shipments(due_before=now)
            if bool(self.config.cross_get(
                keys=self._dock_key(shipment.get('destination'), 'Enable'),
                default=True,
            ))
        ]

    def request_dock_harvest(
        self, now=None, shipments=None, update=True
    ):
        """立即唤醒码头收获，并同步到期目的城市的页面时间。

        这是运输到达后的统一交接入口。只处理已启用码头收获的目的
        城市，并尊重码头收获总开关；不会因为运输到达而擅自开启用户
        已关闭的任务。
        """
        now = (now or datetime.now()).replace(microsecond=0)
        if not bool(self.config.cross_get(
            keys='JiangnanDockHarvest.Scheduler.Enable', default=True
        )):
            logger.info('Dock harvest is disabled; skip immediate wake-up')
            return False

        candidates = list(
            shipments if shipments is not None
            else self.due_dock_shipments(now)
        )
        due = []
        for shipment in candidates:
            if shipment.get('status') != 'transporting':
                continue
            dock_next = self._parse_time(
                shipment.get('dock_next_at') or shipment.get('finish_at')
            )
            if dock_next is None or dock_next > now:
                continue
            destination = shipment.get('destination')
            if destination not in CITY_NAME_TO_KEY:
                continue
            if not bool(self.config.cross_get(
                keys=self._dock_key(destination, 'Enable'), default=True
            )):
                continue
            due.append(shipment)
        if not due:
            return False

        destinations = sorted({
            shipment.get('destination') for shipment in due
        })
        self.config.modified['JiangnanDockHarvest.Scheduler.NextRun'] = now
        for city in destinations:
            self.config.modified[self._dock_key(city, 'NextRun')] = now
        logger.warning(
            'Transport arrived; wake JiangnanDockHarvest immediately: '
            f'cities={destinations}, shipments={len(due)}, now={now}'
        )
        if update:
            self.config.update()
        return True

    def update_shipment_finish(self, shipment, finish_time):
        """用运输总览的实时剩余时间更正收获和库存入账节点。"""
        finish_time = finish_time.replace(microsecond=0)
        now = datetime.now().replace(microsecond=0)
        old_finish = self._parse_time(shipment.get('finish_at'))

        # 发船页 OCR 偶尔会把 2 小时读成 2 分钟。若旧节点已经因此
        # 提前加到目的库存，而总览仍明确显示运输中，必须先撤销早到的
        # 账面数量；否则 UI 库存会在真正抵达前长期虚高。
        credited = bool(shipment.get('destination_credited_at'))
        if credited and finish_time > now:
            city = shipment.get('destination')
            resource = self.normalize_resource(shipment.get('resource'))
            amount = max(0, int(shipment.get('amount', 0)))
            if (shipment.get('inventory_reconciliation') == 'arrival_time'
                    and city in CITY_NAME_TO_KEY
                    and resource in ALL_RESOURCES and amount):
                inventories = self.load_inventories((city,))
                inventory = inventories.setdefault(city, {})
                current = int(inventory.get(resource, -1))
                inventory[resource] = (
                    max(0, current - amount) if current >= 0 else -1
                )
                self.write_inventories(inventories, update=False)
                logger.warning(
                    f'Rollback premature transport arrival: {city}, '
                    f'{resource} -{amount}, old_finish={old_finish}, '
                    f'corrected_finish={finish_time}'
                )
            shipment['destination_credited_at'] = None
            self.request_allocation((city,), when=now, update=False)

        shipment['finish_at'] = self._time_text(finish_time)
        shipment['dock_next_at'] = self._time_text(finish_time)
        self.save()
        self.sync_inventory_change_schedule(update=False)
        self.sync_dock_schedule()

    def apply_shipment_harvest(self, shipment, discarded=False):
        """Consume one dock node and credit only a confirmed received amount."""
        now = datetime.now().replace(microsecond=0)
        destination = shipment.get('destination')
        if discarded:
            self._invalidate_inventory(
                destination,
                shipment.get('resource'),
                reason='码头仓满，实际接收数量未知',
                now=now,
            )
            shipment['inventory_reconciliation'] = 'scan_required_discard'
            logger.warning(
                'Dock overflow was discarded; request warehouse scan '
                'because the accepted amount is unknown'
            )
        elif not shipment.get('destination_credited_at'):
            self._credit_inventory(
                destination,
                shipment.get('resource'),
                shipment.get('amount', 0),
                reason='码头收获确认',
                now=now,
            )
            shipment['destination_credited_at'] = self._time_text(now)
            shipment['inventory_reconciliation'] = 'confirmed_collection'

        shipment['status'] = 'collected'
        shipment['collected_at'] = self._time_text(now)
        # 收获后来源城市的船位已经释放，目的城市也拿到了实货，两边都
        # 应立即重新规划。
        self.request_allocation(
            (shipment.get('source'), destination),
            when=now,
            update=False,
        )
        self.save()
        self.sync_inventory_change_schedule(update=False)
        self.sync_dock_schedule(update=False)
        self.config.update()

    def sync_dock_schedule(self, update=True, fallback=None):
        """把每座城市的全部待收获节点同步到码头收获 UI。"""
        now = datetime.now().replace(microsecond=0)
        automatic_fallback = fallback is None
        fallback = fallback or now + timedelta(hours=24)

        def current_value(key):
            return (self.config.modified[key] if key in self.config.modified
                    else self.config.cross_get(keys=key, default=None))

        def set_changed(key, value):
            current = current_value(key)
            if isinstance(value, datetime):
                current = self._parse_time(current)
            if current != value:
                self.config.modified[key] = value

        grouped = {city: [] for city in CITY_NAME_TO_KEY}
        for shipment in self.active_shipments():
            destination = shipment.get('destination')
            if destination in grouped:
                grouped[destination].append(shipment)
        all_times = []
        city_minimums = {}
        for city in CITY_NAME_TO_KEY:
            shipments = grouped[city]
            enabled = bool(self.config.cross_get(
                keys=self._dock_key(city, 'Enable'),
                default=True,
            ))
            lines = []
            city_times = []
            for shipment in shipments:
                dock_next = self._parse_time(
                    shipment.get('dock_next_at') or shipment.get('finish_at')
                )
                if dock_next is not None:
                    city_times.append(dock_next)
                    if enabled:
                        all_times.append(dock_next)
                    finish_text = self._time_text(dock_next)
                else:
                    finish_text = '等待识别'
                lines.append(
                    f'{finish_text} | {shipment.get("source")} '
                    f'船位{int(shipment.get("slot", 0)) + 1} | '
                    f'{shipment.get("resource")} x{shipment.get("amount")}'
                )
            next_run = min(city_times) if city_times else fallback
            pending_key = self._dock_key(city, 'Pending')
            next_key = self._dock_key(city, 'NextRun')
            if automatic_fallback and not shipments and current_value(pending_key) == '无':
                previous = self._parse_time(current_value(next_key))
                # Keep an already idle city's future recheck stable. Moving
                # it by a few seconds on every other city's update causes
                # unnecessary file writes and endless countdown drift.
                if previous is not None and previous > now:
                    next_run = previous
            if city_times:
                city_minimums[city] = next_run
            if enabled and not city_times:
                all_times.append(next_run)
            set_changed(pending_key, '\n'.join(lines) if lines else '无')
            set_changed(next_key, next_run)
        set_changed('JiangnanDockHarvest.Scheduler.NextRun',
            min(all_times) if all_times else fallback
        )
        logger.attr(
            'DockHarvestGlobalNextRun',
            min(all_times) if all_times else fallback,
        )
        logger.attr(
            'DockHarvestCityNextRuns',
            ', '.join(
                f'{city}={self._time_text(value)}'
                for city, value in city_minimums.items()
            ),
        )
        if update and self.config.modified:
            self.config.update()
        return all_times

    def set_city_next_runs(self, snapshot, fallback):
        """把每座城市的最近船只结束时间写入物资调配 UI。"""
        now = datetime.now().replace(microsecond=0)
        result = {}
        for city, slots in snapshot.items():
            ledger_finish_times = [
                self._parse_time(
                    shipment.get('dock_next_at')
                    or shipment.get('finish_at')
                )
                for shipment in self.data.get('shipments', [])
                if shipment.get('status') == 'transporting'
                and shipment.get('source') == city
                and self._parse_time(
                    shipment.get('dock_next_at')
                    or shipment.get('finish_at')
                ) is not None
            ]
            # 有脚本发船台账时优先使用出发页识别结果；没有台账的手动
            # 运输才使用总览页剩余时间。
            finish_times = ledger_finish_times or [
                slot.get('finish_time') for slot in slots
                if slot.get('state') == 'transporting'
                and slot.get('finish_time') is not None
            ]
            has_unknown = any(
                slot.get('state') == 'occupied_unknown' for slot in slots
            )
            # finish_at 是库存入账时间；船已抵达但尚未收获时它可能已经
            # 过去。dock_next_at 才是当前码头复查时间。即使旧台账仍有
            # 过期值，也绝不能把逐城 NextRun 再写回过去形成调度热循环。
            future_finish_times = [value for value in finish_times if value > now]
            has_active_ledger = any(
                shipment.get('status') == 'transporting'
                and shipment.get('source') == city
                for shipment in self.data.get('shipments', [])
            )
            has_arrived = any(
                slot.get('state') == 'arrived' for slot in slots
            )
            has_idle = any(
                slot.get('state') in ('idle', 'collected') for slot in slots
            )
            # 两个船位属于两份独立的运输资格。只要仍有一个空闲位，
            # 整座城市就不能被另一艘在途船的结束时间锁住；按正常回退
            # 时间复查即可。当前执行循环会优先把能发出的空闲位填满。
            if has_idle:
                next_run = fallback
            elif future_finish_times:
                next_run = min(future_finish_times) + timedelta(minutes=1)
            elif has_unknown or has_arrived or has_active_ledger:
                next_run = now + timedelta(minutes=10)
            else:
                next_run = fallback
            retry_at = self._parse_time(
                self.data.get('allocation_no_idle_until', {}).get(city)
            )
            if retry_at is not None and retry_at > now:
                next_run = retry_at
            result[city] = next_run.replace(microsecond=0)
            city_key = CITY_NAME_TO_KEY[city]
            self.config.modified[
                f'JiangnanCityAllocation.{city_key}.NextRun'
            ] = result[city]
        self.config.update()
        return result

    def city_next_time(self, city):
        city_key = CITY_NAME_TO_KEY[city]
        value = self.config.cross_get(
            keys=f'JiangnanCityAllocation.{city_key}.NextRun',
            default=DEFAULT_TIME,
        )
        return self._parse_time(value) or DEFAULT_TIME

    def due_cities(self, cities, now=None):
        now = now or datetime.now()
        # The cooldown is written to NextRun when slot selection fails.
        # A manual reset or dock collection may deliberately bring it forward;
        # let that fresh observation override the earlier failed selection.
        return [city for city in cities if self.city_next_time(city) <= now]

    def all_city_next_runs(self, cities, fallback, normalize_past=False):
        """读取逐城时间，可在一轮任务结束时清理已经消费的过期值。

        用户手动把某城时间改到过去时，任务入口仍需看到该值并只扫描
        该城；所以默认不改过去时间。任务已经完成本轮处理后传入
        ``normalize_past=True``，未被新快照覆盖的旧到期值才会回退到
        下一次正常检查时间，防止页面总时间反复写回过去形成热循环。
        """
        result = {}
        changed = False
        now = datetime.now().replace(microsecond=0)
        for city in cities:
            value = self.city_next_time(city)
            if value <= DEFAULT_TIME or (normalize_past and value <= now):
                value = fallback
                city_key = CITY_NAME_TO_KEY[city]
                self.config.modified[
                    f'JiangnanCityAllocation.{city_key}.NextRun'
                ] = value
                changed = True
            result[city] = value
        if changed:
            self.config.update()
        return result

    def record_no_idle_source(self, city, now=None):
        """Persist a failed slot selection until a bounded, city-local recheck."""
        now = (now or datetime.now()).replace(microsecond=0)
        finishes = [
            self._parse_time(slot.get('finish_time'))
            for slot in self.data.get('slot_status', {}).get(city, [])
            if slot.get('state') == 'transporting'
        ]
        finishes.extend(
            self._parse_time(item.get('dock_next_at') or item.get('finish_at'))
            for item in self.data.get('shipments', [])
            if item.get('source') == city and item.get('status') == 'transporting'
        )
        candidates = [value + timedelta(minutes=1) for value in finishes
                      if value is not None and value > now]
        retry_at = min([now + timedelta(minutes=10)] + candidates)
        self.data.setdefault('allocation_no_idle_until', {})[city] = self._time_text(retry_at)
        self.save()
        city_key = CITY_NAME_TO_KEY[city]
        self.config.modified[f'JiangnanCityAllocation.{city_key}.NextRun'] = retry_at
        self.config.update()
        logger.info(f'Allocation source slot recheck: {city}, next={retry_at}')
        return retry_at

    def record_slot_snapshot(self, snapshot, clear_no_idle=True):
        stored = self.data.setdefault('slot_status', {})
        for city, slots in snapshot.items():
            if clear_no_idle and any(slot.get('state') in ('idle', 'collected') for slot in slots):
                self.data.get('allocation_no_idle_until', {}).pop(city, None)
            stored[city] = [
                {
                    'slot': int(slot['slot']),
                    'state': slot['state'],
                    'finish_time': (
                        self._time_text(slot['finish_time'])
                        if slot.get('finish_time') is not None else None
                    ),
                }
                for slot in slots
            ]
        self.save()

    def occupied_sources(self, cities):
        """依据最近船位快照判断两个运输资格是否都被占用。"""
        result = set()
        stored = self.data.get('slot_status', {})
        active_slots = {
            (shipment.get('source'), int(shipment.get('slot', -1)))
            for shipment in self.data.get('shipments', [])
            if shipment.get('status') == 'transporting'
        }
        for city in cities:
            retry_at = self._parse_time(
                self.data.get('allocation_no_idle_until', {}).get(city)
            )
            if retry_at is not None and retry_at > datetime.now():
                result.add(city)
                continue
            slots = stored.get(city, [])
            available = {
                int(slot.get('slot', -1)) for slot in slots
                if slot.get('state') != 'unavailable'
            }
            occupied = {
                int(slot.get('slot', -1)) for slot in slots
                if slot.get('state') not in (
                    'idle', 'collected', 'unavailable'
                )
            }
            occupied.update(
                slot for source, slot in active_slots if source == city
            )
            # 有效船位全部占用，或该城市当前根本没有可用码头，都不能
            # 作为本轮运输来源。
            if (available and available <= occupied) or not available:
                result.add(city)
        return result

    def idle_sources(self, cities):
        """返回最近快照中至少有一个真实空闲运输位的城市。"""
        stored = self.data.get('slot_status', {})
        active_slots = {
            (shipment.get('source'), int(shipment.get('slot', -1)))
            for shipment in self.data.get('shipments', [])
            if shipment.get('status') == 'transporting'
        }
        result = set()
        for city in cities:
            retry_at = self._parse_time(
                self.data.get('allocation_no_idle_until', {}).get(city)
            )
            if retry_at is not None and retry_at > datetime.now():
                continue
            if any(
                slot.get('state') in ('idle', 'collected')
                and (city, int(slot.get('slot', -1))) not in active_slots
                for slot in stored.get(city, [])
            ):
                result.add(city)
        return result

    def active_sources(self):
        """返回仍需跟踪抵达/收取的在途来源城市。"""
        return {
            shipment.get('source')
            for shipment in self.data.get('shipments', [])
            if shipment.get('status') == 'transporting'
            and shipment.get('source') in CITY_NAME_TO_KEY
        }
