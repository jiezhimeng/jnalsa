"""Bounded retention for Jiangnan's diagnostic screenshots."""

from datetime import datetime, timedelta
from pathlib import Path
import re


MAX_ABNORMAL_SCREENSHOTS = 300
MAX_ABNORMAL_BYTES = 384 * 1024 * 1024
MAX_ABNORMAL_AGE = timedelta(days=7)
_DATE_DIR = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_ABNORMAL_NAME = re.compile(r'^\d{6}_\d{6}_[A-Za-z0-9_-]+\.png$')


def prune_abnormal_screenshots(root, now=None):
    """Remove oldest generated evidence first, leaving other files alone.

    Returns the number of removed images and bytes. A failing deletion is
    skipped so diagnostics can never interrupt the automation task.
    """
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return 0, 0
    now = now or datetime.now()
    candidates = []
    try:
        date_dirs = list(root.iterdir())
    except OSError:
        return 0, 0
    for date_dir in date_dirs:
        if (not date_dir.is_dir() or date_dir.is_symlink()
                or not _DATE_DIR.fullmatch(date_dir.name)):
            continue
        try:
            entries = list(date_dir.iterdir())
        except OSError:
            continue
        for path in entries:
            if (not path.is_file() or path.is_symlink()
                    or not _ABNORMAL_NAME.fullmatch(path.name)):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime, path, stat.st_size))
    candidates.sort(reverse=True)
    kept_count = kept_bytes = removed_count = removed_bytes = 0
    cutoff = (now - MAX_ABNORMAL_AGE).timestamp()
    for modified, path, size in candidates:
        keep = (modified >= cutoff
                and kept_count < MAX_ABNORMAL_SCREENSHOTS
                and kept_bytes + size <= MAX_ABNORMAL_BYTES)
        if keep:
            kept_count += 1
            kept_bytes += size
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed_count += 1
        removed_bytes += size
    return removed_count, removed_bytes
