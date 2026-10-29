"""Daily inn keeper click slots, in China Standard Time (UTC+8)."""
from datetime import datetime, timedelta, timezone


GOODS_HOURS = (10, 16, 21)
GOODS_GROUP = 'yingtian_inn_goods'
CHINA_TIME = timezone(timedelta(hours=8))


def inn_now():
    return datetime.now(CHINA_TIME).replace(tzinfo=None, microsecond=0)


def china_local(value):
    return value.astimezone(CHINA_TIME).replace(tzinfo=None) if value.tzinfo is not None else value


def current_goods_slot(now):
    now = china_local(now)
    slots = [now.replace(hour=hour, minute=0, second=0, microsecond=0)
             for hour in GOODS_HOURS]
    return next((slot for slot in reversed(slots) if slot <= now), None)


def next_goods_slot(now):
    now = china_local(now)
    for hour in GOODS_HOURS:
        slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if slot > now:
            return slot
    return (now+timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)


def slot_key(slot):
    return slot.strftime('%Y-%m-%d %H:%M') if slot is not None else ''


def goods_next_run(now, last_slot='', attempted_slot='', retry_slot='', saved_next=None):
    """Catch up only today's latest slot; never replay an attempted click."""
    now = china_local(now)
    current = current_goods_slot(now)
    key = slot_key(current)
    upcoming = next_goods_slot(now)
    if not key or key in (last_slot, attempted_slot):
        return upcoming
    if retry_slot == key:
        try:
            saved = saved_next if isinstance(saved_next, datetime) else datetime.fromisoformat(str(saved_next))
            saved = china_local(saved)
            if now < saved < upcoming:
                return saved
        except (ValueError, TypeError):
            pass
    return now
