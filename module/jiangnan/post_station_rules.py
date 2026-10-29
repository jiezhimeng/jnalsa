"""Pure configuration and selection rules for 应天府驿站."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata

MAPS = ('qinhuai', 'yingtian', 'liniang')
MAP_NAMES = dict(zip(MAPS, ('秦淮河畔', '应天城郊', '丽娘府宅')))
MAP_FIELDS = {key: 'Map' + key.capitalize() for key in MAPS}
MINUTES_PER_RATION = 40


def names(value):
    return tuple(dict.fromkeys(filter(None, re.split(r'[\s,，、;；]+', str(value or '').strip()))))


def normalize(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value or '')))


def parse_stock(value, switching=False):
    text = normalize(value)
    if switching:
        # The exit dialog displays a wheat icon, a decorative minus, stock/cost.
        text = text.lstrip('-−一')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', text)
    if not match:
        return None
    current, total = map(Decimal, match.groups())
    if total <= 0 or (not switching and current > total):
        return None
    return current, total


def parse_stamina(value):
    text = normalize(value)
    # Read the whole label, because a two-digit numerator starts farther right
    # than a three-digit numerator. Never strip arbitrary OCR garbage.
    text = re.sub(r'^(?:探险队体力|探險隊體力|探险队|探險隊|体力|體力)', '', text)
    return parse_stock(text)


def can_depart(stock, threshold, cost=1):
    return stock is not None and stock > Decimal(str(threshold)) and stock >= cost


def interval_minutes(value):
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValueError('间隔粮草量必须为正数') from None
    if not amount.is_finite() or amount <= 0 or amount > 10000:
        raise ValueError('间隔粮草量必须大于0且不超过10000')
    return float(amount * MINUTES_PER_RATION)


def interval_label(value):
    minutes = interval_minutes(value)
    hours, remaining = divmod(minutes, 60)
    text = (f'{hours:g}小时' if hours else '') + (f'{remaining:g}分钟' if remaining else '')
    return f'{value}粮草 × 40分钟 = {text}（{minutes:g}分钟）'


def enabled_maps(selected, order=MAPS):
    return tuple(key for key in order if selected.get(key, False))


def next_map(available, previous='', order=MAPS):
    if not available:
        return None
    order = tuple(order)
    if previous not in order:
        return available[0]
    start = order.index(previous)
    return next(order[(start + step) % len(order)] for step in range(1, len(order)+1)
                if order[(start + step) % len(order)] in available)


@dataclass(frozen=True)
class Captain:
    slot: int
    name: str
    current: int
    maximum: int


def choose_captain(candidates, preferred=()):
    usable = [c for c in candidates if 0 < c.current <= c.maximum and 0 <= c.slot < 6]
    full = [c for c in usable if c.current == c.maximum]
    for name in preferred:
        match = next((c for c in full if c.name == name), None)
        if match:
            return match
    return (min(full, key=lambda c: (-c.maximum, c.slot)) if full else
            min(usable, key=lambda c: (-c.current, c.slot)) if usable else None)
