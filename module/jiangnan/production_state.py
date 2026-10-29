"""Reuse confirmed production clocks until an actual inspection is due."""
from datetime import datetime


def production_record_pending(record, now=None):
    if record.get('state') not in ('working', 'upgrading', 'occluded'):
        return False
    finish = record.get('finish_at')
    try:
        if not isinstance(finish, datetime):
            finish = datetime.fromisoformat(str(finish))
        return finish > (now or datetime.now())
    except (TypeError, ValueError):
        return False


def cached_working_group(buildings, now=None, saturated=False):
    """Reuse a complete group while every instance has a future confirmed clock."""
    if not buildings:
        return None
    try:
        totals = {int(item.get('total', 0)) for item in buildings.values()}
    except (TypeError, ValueError):
        return None
    if totals != {len(buildings)} or set(buildings) != {str(i) for i in range(1, len(buildings) + 1)}:
        return None
    now = now or datetime.now()
    if not all(
        production_record_pending(item, now)
        or (saturated and item.get('state') == 'threshold')
        for item in buildings.values()
    ):
        return None
    return {key: dict(item) for key, item in buildings.items()}


def cached_production_group(buildings, inventory_state, city, resources, now=None):
    cached = cached_working_group(buildings, now)
    if cached is not None:
        return cached
    # Only consult stock when paused instances are the reason the group
    # cannot reuse its clocks; unknown/due instances always need inspection.
    if cached_working_group(buildings, now, saturated=True) is None:
        return None
    states = inventory_state.production_threshold_states(city, resources)
    if resources and all(states.get(resource) is False for resource in resources):
        return {key: dict(item) for key, item in buildings.items()}
    return None


def production_next_run(buildings, now, retry, fallback, paused=False):
    """Combine all instance clocks; an error must never delay another harvest."""
    candidates = []
    needs_retry = False
    for item in buildings.values():
        state = item.get('state')
        if state in ('working', 'upgrading', 'occluded'):
            try:
                finish = item.get('finish_at')
                if not isinstance(finish, datetime):
                    finish = datetime.fromisoformat(str(finish))
                if finish > now:
                    candidates.append(finish)
                else:
                    needs_retry = True
            except (TypeError, ValueError):
                needs_retry = True
        elif state not in ('threshold', 'no_priority'):
            if state != 'overflow' or not paused:
                needs_retry = True
    if buildings:
        try:
            totals = {int(item.get('total', 0)) for item in buildings.values()}
            # Older records without a total still retain their own clocks.
            total = max(totals)
            if total > 0 and set(buildings) != {str(i) for i in range(1, total + 1)}:
                needs_retry = True
            if len(totals) > 1:
                needs_retry = True
        except (TypeError, ValueError):
            needs_retry = True
    if needs_retry:
        candidates.append(now + retry)
    if candidates:
        return min(candidates)
    return fallback if fallback > now else now + retry


def production_should_yield_to_dock(inventory_state):
    """Yield only when an enabled dock task actually accepts the handoff."""
    config = getattr(inventory_state, 'config', None)
    debug_task = getattr(config, '_jiangnan_debug_task', None)
    if isinstance(debug_task, str) and debug_task.startswith('Jiangnan'):
        return False
    due = inventory_state.due_dock_shipments()
    return bool(due and inventory_state.request_dock_harvest(shipments=due))
