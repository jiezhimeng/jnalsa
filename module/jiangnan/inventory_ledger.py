from datetime import datetime, timedelta
from uuid import uuid4


class InventoryLedger:
    """江南库存事件账本。

    规则：
    1. snapshot 是唯一真实基线。
    2. event 保存变化原因，不允许模块私改库存数字。
    3. replay 根据 snapshot + confirmed events 重建确认库存。
    """

    VERSION = 1

    def __init__(self, state):
        self.state = state
        self.state.setdefault('inventory_snapshot', {})
        self.state.setdefault('inventory_events', [])

    @staticmethod
    def now():
        return datetime.now().replace(microsecond=0)

    def add_event(self, city, resource, amount, event_type,
                  confirmed=False, metadata=None):
        event_time = self.now()
        event = {
            'id': str(uuid4()),
            'city': city,
            'resource': resource,
            'amount': int(amount),
            'type': event_type,
            'created_at': event_time.isoformat(),
            'confirmed': bool(confirmed),
            'metadata': metadata or {},
        }
        self.state['inventory_events'].append(event)
        return event

    def create_snapshot(self, city, inventory):
        snapshot_time = self.now().isoformat()
        self.state['inventory_snapshot'][city] = {
            'time': snapshot_time,
            'inventory': dict(inventory),
        }
        for event in self.state['inventory_events']:
            if event.get('city') == city:
                event['covered_by_snapshot'] = True

    def compact(self, now=None, retention=timedelta(days=7)):
        """Expire confirmed history already absorbed into a snapshot.

        Keep event IDs as small tombstones so explicit retries remain
        idempotent even after their full audit records have expired.
        Unconfirmed, uncovered and undated events must remain available.
        """
        now = now or self.now()
        kept = []
        archived_ids = set(self.state.get('inventory_archived_event_ids', []))
        removed = 0
        for event in self.state['inventory_events']:
            covered = event.get('covered_by_snapshot') or event.get('included_in_scan')
            confirmed = event.get('confirmed') or event.get('confirmed_at')
            try:
                created = datetime.fromisoformat(str(event.get('created_at')))
                expired = now - created > retention
            except (TypeError, ValueError):
                expired = False
            if covered and confirmed and expired:
                if event.get('id'):
                    archived_ids.add(event['id'])
                removed += 1
            else:
                kept.append(event)
        if removed:
            self.state['inventory_events'] = kept
            self.state['inventory_archived_event_ids'] = sorted(archived_ids)
        return removed

    def replay(self, city):
        snapshot = self.state['inventory_snapshot'].get(city, {})
        result = dict(snapshot.get('inventory', {}))
        for event in self.state['inventory_events']:
            if event.get('city') != city:
                continue
            if event.get('covered_by_snapshot') or event.get('included_in_scan'):
                continue
            if not (event.get('confirmed') or event.get('confirmed_at')):
                continue
            resource = event['resource']
            current = int(result.get(resource, -1))
            result[resource] = current + int(event['amount']) if current >= 0 else -1
        return result
