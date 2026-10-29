import queue
import time

import cv2


_screenshot_queue = None
_screenshot_enabled = None
_last_publish = 0.0
PUBLISH_INTERVAL = 1.0
JPEG_QUALITY = 82


def set_screenshot_queue(screenshot_queue, screenshot_enabled=None):
    """绑定由 GUI 主进程创建的跨进程最新截图队列。"""
    global _screenshot_queue, _screenshot_enabled, _last_publish
    _screenshot_queue = screenshot_queue
    _screenshot_enabled = screenshot_enabled
    _last_publish = 0.0


def publish_screenshot(image):
    """非阻塞发布最新截图；监控异常不得影响自动化主流程。"""
    global _last_publish
    if _screenshot_queue is None or image is None:
        return
    if _screenshot_enabled is not None and not _screenshot_enabled.is_set():
        return

    now = time.monotonic()
    if now - _last_publish < PUBLISH_INTERVAL:
        return
    _last_publish = now

    try:
        # Automation screenshots are kept in RGB order by the screenshot
        # backends, while OpenCV's encoder expects BGR.  Convert only the
        # monitoring copy so OCR/click processing keeps its original image.
        monitor_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            '.jpg', monitor_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not success:
            return
        payload = (time.time(), encoded.tobytes())
        try:
            _screenshot_queue.put_nowait(payload)
        except queue.Full:
            # 队列只保留一张图。消费端来不及时丢弃旧图而不是阻塞脚本。
            try:
                _screenshot_queue.get_nowait()
            except queue.Empty:
                pass
            _screenshot_queue.put_nowait(payload)
    except Exception:
        # 编码器、代理队列或 GUI 进程退出都不能中断设备截图。
        pass
