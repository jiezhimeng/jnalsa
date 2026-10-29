import os
from datetime import datetime

from module.config.utils import filepath_config, DEFAULT_TIME
from module.logger import logger


class ConfigWatcher:
    config_name = 'alas'
    start_mtime = DEFAULT_TIME

    def start_watching(self) -> None:
        stat = os.stat(filepath_config(self.config_name))
        self.start_mtime = datetime.fromtimestamp(stat.st_mtime)
        self._start_signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def get_mtime(self) -> datetime:
        """
        Last modify time of the file
        """
        timestamp = os.stat(filepath_config(self.config_name)).st_mtime
        mtime = datetime.fromtimestamp(timestamp)
        return mtime

    def should_reload(self) -> bool:
        """
        Returns:
            bool: Whether the file has been modified and configs should reload
        """
        stat = os.stat(filepath_config(self.config_name))
        mtime = datetime.fromtimestamp(stat.st_mtime)
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        baseline = getattr(self, '_start_signature', None)
        # Atomic saves can replace a file within the same second, or restore
        # an older timestamp. Keep native precision and detect replacements.
        changed = signature != baseline if baseline is not None else mtime != self.start_mtime
        if changed:
            logger.info(f'Config "{self.config_name}" changed at {mtime}')
            return True
        else:
            return False
