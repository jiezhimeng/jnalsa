import re

from module.device.device import Device
from module.device.method.adb import retry
from module.logger import logger
from module.jiangnan.network_popup import NetworkPopupInterrupted, network_popup_visible


class JiangnanDevice(Device):
    """
    江南百景图专用 Device。
    """

    def screenshot(self):
        image = super().screenshot()
        if (getattr(self, '_jiangnan_network_watch_enabled', False)
                and not getattr(self, '_jiangnan_network_recovery_active', False)
                and network_popup_visible(image)):
            # Abort the old page's actions before any caller can click through
            # the modal. The scheduler owns recovery and retries the task.
            raise NetworkPopupInterrupted()
        observer = getattr(self, '_jiangnan_task_reward_observer', None)
        if callable(observer):
            # Observation only. Queue writes and all UI actions belong to the
            # scheduler after the current task has finished.
            observer(image)
        return image

    @staticmethod
    def _foreground_package(text):
        """Read both Android's window focus and newer top-resumed activity."""
        for pattern in (
            r'mCurrentFocus=Window\{[^\n]*?\s([\w.]+)/[^\s}]+',
            r'(?:topResumedActivity|mResumedActivity)[=:]\s*ActivityRecord\{[^\n]*?\s([\w.]+)/[^\s}]+',
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    @retry
    def app_current(self):
        """Avoid the legacy `window windows`/full activity dump on MuMu."""
        if self.is_wsa:
            return super().app_current()
        # MuMu's Android returns no focus for `dumpsys window windows`.
        # The old u2 fallback also misses `topResumedActivity`, and then its
        # full `activity top` dump takes ~5s. These current queries take <0.1s.
        for command in (['dumpsys', 'window'], ['dumpsys', 'activity', 'activities']):
            package = self._foreground_package(self.adb_shell(command, timeout=3))
            if package:
                return package
        # Retain the existing platform fallback when both formats are unknown.
        return super().app_current()


    def press_back(self):
        logger.info('Press Android BACK')
        self.adb_shell(
            ['input', 'keyevent', '4']
        )
