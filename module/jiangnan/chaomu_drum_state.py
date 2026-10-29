"""Daily drum limits and clocks, independent of OCR and device imports."""
import re
import unicodedata
from datetime import datetime, timedelta


DRUM_INTERVAL = timedelta(hours=2)
DRUM_DAILY_LIMIT = 10
DEFAULT_TIME = datetime(2020, 1, 1)
DRUM_COIN_COST_TO_USED = {1000: 5, 2000: 6, 3000: 7, 4000: 8, 5000: 9}


def drum_time(value):
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return DEFAULT_TIME


def parse_drum_count(text):
    """Accept the daily-use sentence only, never the button's assignment badge."""
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(text)))
    # The panel's horizontal border can be recognized as a leading check box.
    text = text.lstrip('☑☐✓✔')
    match = re.fullmatch(r'今日已[击擊]鼓([0-9]{1,3})次', text)
    return int(match.group(1)) if match else None


def drum_used_count(record, now):
    count_date = record.get('CountDate')
    if isinstance(count_date, datetime):
        count_date = count_date.date().isoformat()
    if count_date != now.date().isoformat():
        return 0
    try:
        return max(0, int(record.get('UsedCount', 0)))
    except (TypeError, ValueError):
        # A damaged daily ledger cannot authorize additional clicks.
        return DRUM_DAILY_LIMIT


def drum_next_run(record, now):
    """A manual NextRun edit cannot bypass cooldown or a full daily quota."""
    target = drum_time(record.get('NextRun'))
    last_attempt = drum_time(record.get('LastAttempt'))
    if last_attempt != DEFAULT_TIME:
        target = max(target, last_attempt + DRUM_INTERVAL)
    if drum_used_count(record, now) >= DRUM_DAILY_LIMIT:
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        target = max(target, tomorrow)
    return target
