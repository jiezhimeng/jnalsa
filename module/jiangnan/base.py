import re
import time
from datetime import datetime
from pathlib import Path

import cv2

from module.base.base import ModuleBase
from module.config.config import AzurLaneConfig
from module.exception import RequestHumanTakeover
from module.jiangnan.device import JiangnanDevice
from module.jiangnan.retention import prune_abnormal_screenshots
from module.logger import logger


_ABNORMAL_SCREENSHOT_LAST = {}
_ABNORMAL_RETENTION_INITIALIZED = False


class JiangnanBase(ModuleBase):

    def save_abnormal_screenshot(self, reason, interval=30):
        """保存当前异常画面，按原因限流；截图和运行日志放在同一目录下。"""
        image = getattr(self.device, 'image', None)
        if image is None:
            return None
        now = time.monotonic()
        if now - _ABNORMAL_SCREENSHOT_LAST.get(reason, float('-inf')) < interval:
            return None
        _ABNORMAL_SCREENSHOT_LAST[reason] = now
        try:
            directory = (Path(__file__).resolve().parents[2] / 'log'
                         / 'screenshots' / datetime.now().strftime('%Y-%m-%d'))
            directory.mkdir(parents=True, exist_ok=True)
            safe_reason = re.sub(r'[^a-zA-Z0-9_-]', '_', str(reason))[:64]
            name = f'{datetime.now():%H%M%S_%f}_{safe_reason}.png'
            # device.image 是 RGB；OpenCV 写文件前转换为 BGR。
            encoded_ok, encoded = cv2.imencode(
                '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not encoded_ok:
                return None
            path = directory / name
            path.write_bytes(encoded.tobytes())
            logger.warning(f'Abnormal screenshot saved: {path}')
            removed, removed_bytes = prune_abnormal_screenshots(directory.parent)
            if removed:
                logger.info(
                    f'Rotated {removed} old abnormal screenshots '
                    f'({removed_bytes // 1024 // 1024} MiB)'
                )
            return path
        except (OSError, ValueError, cv2.error) as exc:
            logger.warning(f'Unable to save abnormal screenshot ({reason}): {exc}')
            return None

    def app_is_running_recoverable(self):
        """Recover the connection; MuMu may only reconnect, never restart."""
        try:
            return self.device.app_is_running()
        except RequestHumanTakeover as error:
            if getattr(self, '_jiangnan_emulator_recovery_active', False):
                raise
            if getattr(self.device, 'is_mumu_family', False):
                logger.warning('MuMu restart is disabled; attempt ADB reconnect only')
                self._jiangnan_emulator_recovery_active = True
                try:
                    self.device.adb_reconnect()
                    return self.device.app_is_running()
                finally:
                    self._jiangnan_emulator_recovery_active = False
            self._jiangnan_emulator_recovery_active = True
            try:
                logger.error(
                    f'Jiangnan device connection lost: {error}; '
                    'restart emulator (up to 3 internal attempts)'
                )
                for attempt in range(1, 4):
                    logger.info(f'Jiangnan emulator recovery attempt {attempt}/3')
                    try:
                        if not self.device.emulator_start():
                            continue
                        running = self.device.app_is_running()
                        if not running:
                            self.device.app_start()
                            self.device.sleep(3)
                            running = self.device.app_is_running()
                        if not running:
                            continue
                        logger.info('Jiangnan emulator connection recovered, app_running=True')
                        return True
                    except RequestHumanTakeover as retry_error:
                        logger.warning(f'Jiangnan emulator recovery attempt failed: {retry_error}')
                raise RequestHumanTakeover('Jiangnan emulator recovery failed after 3 attempts')
            finally:
                self._jiangnan_emulator_recovery_active = False

    def __init__(self, config, device=None, task=None):
        """
        Jiangnan 专用初始化。

        与 ModuleBase 的主要区别：
        默认创建 JiangnanDevice，而不是原版 Device。
        """

        # 配置名称，例如 'alas'
        if isinstance(config, str):
            config = AzurLaneConfig(config, task=task)

        # 默认使用江南专用 Device
        if device is None:
            device = JiangnanDevice(config=config)

        # 如果开发时传入 serial
        elif isinstance(device, str):
            config.override(Emulator_Serial=device)
            device = JiangnanDevice(config=config)

        # config 和 device 已经初始化，
        # 所以这里 task=None，避免重复 init_task
        super().__init__(
            config=config,
            device=device,
            task=None
        )
        global _ABNORMAL_RETENTION_INITIALIZED
        if not _ABNORMAL_RETENTION_INITIALIZED:
            root = Path(__file__).resolve().parents[2] / 'log' / 'screenshots'
            removed, removed_bytes = prune_abnormal_screenshots(root)
            if removed:
                logger.info(
                    f'Rotated {removed} old abnormal screenshots '
                    f'({removed_bytes // 1024 // 1024} MiB)'
                )
            _ABNORMAL_RETENTION_INITIALIZED = True

    def app_check(self):
        """
        检查江南百景图是否正在前台。
        """
        logger.hr('JIANGNAN APP CHECK')

        running = self.app_is_running_recoverable()
        logger.attr('JiangnanRunning', running)

        return running

    def screenshot_test(self, filename='jiangnan_test.png'):
        """
        截取一张当前游戏画面并保存。
        """
        logger.hr('JIANGNAN SCREENSHOT')

        image = self.device.screenshot()

        logger.attr('ImageShape', image.shape)

        self.device.image_save(filename)

        logger.info(f'Screenshot saved: {filename}')

        return image
