"""Per-city special residents kept available for exclusive activities."""
import re
from difflib import SequenceMatcher


RESERVATION_TASK = 'JiangnanSpecialResidentReservation'
CITY_NAMES = {
    'yingtianfu': '应天府', 'suzhoufu': '苏州府', 'hangzhoufu': '杭州府',
    'songjiangfu': '松江府', 'huizhoufu': '徽州府', 'yangzhoufu': '扬州府',
    'shaoxingfu': '绍兴府', 'ningbofu': '宁波府', 'jimingshan': '鸡鸣山',
    'jiangningxian': '江宁县', 'baixuezhen': '白雪镇',
    'zongyingdao': '棕影岛', 'longgong': '龙宫',
}


def normalize_resident_name(name):
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', str(name)).translate(
        str.maketrans('麥滿', '麦满')
    )


def reserved_special_names(config, city):
    """Read only the configured names; missing city lists reserve nobody."""
    city_key = next((key for key, name in CITY_NAMES.items()
                     if city in (key, name)), None)
    if city_key is None:
        return ()
    data = config if isinstance(config, dict) else getattr(config, 'data', {})
    fields = data.get(RESERVATION_TASK, {}).get(f'reserved_{city_key}', {})
    text = fields.get('Names', '')
    names = [normalize_resident_name(name)
             for name in re.split(r'[,，、;；\s]+', str(text or ''))]
    return tuple(dict.fromkeys(name for name in names if name))


def is_reserved_resident(name, reserved):
    """Also reject close OCR readings or clipped names of reserved residents."""
    name = normalize_resident_name(name)
    for resident in reserved:
        resident = normalize_resident_name(resident)
        if name == resident:
            return True
        if (min(len(name), len(resident)) >= 2
                and (name in resident or resident in name
                     or SequenceMatcher(None, name, resident).ratio() >= 2 / 3)):
            return True
    return False
