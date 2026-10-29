import argparse
import os
import queue
import secrets
import threading
from multiprocessing import Process
from typing import Dict, List, Union

import inflection
from rich.console import Console, ConsoleRenderable

# Since this file does not run under the same process or subprocess of app.py
# the following code needs to be repeated
# Import fake module before import pywebio to avoid importing unnecessary module PIL
from module.webui.fake_pil_module import *

import_fake_pil_module()

from module.logger import logger, set_file_logger, set_func_logger
from module.submodule.submodule import load_mod
from module.submodule.utils import get_available_func, get_available_mod, get_available_mod_func, get_config_mod, \
    get_func_mod, list_mod_instance
from module.webui.setting import State


class ProcessManager:
    _processes: Dict[str, "ProcessManager"] = {}

    def __init__(self, config_name: str = "alas") -> None:
        self.config_name = config_name
        self._renderable_queue: queue.Queue[ConsoleRenderable] = State.manager.Queue()
        self._screenshot_queue = State.manager.Queue(1)
        self._screenshot_enabled = State.manager.Event()
        self._screenshot_enabled.set()
        self.renderables: List[ConsoleRenderable] = []
        self.renderables_max_length = 400
        self.renderables_reduce_length = 80
        self._process: Process = None
        self._start_lock = threading.RLock()
        self._active_func = None
        self._task_status = None
        self._process_locks: Dict[str, threading.Lock] = {}
        self.thd_log_queue_handler: threading.Thread = None
        self.thd_screenshot_queue_handler: threading.Thread = None
        self.latest_screenshot = None
        self.latest_screenshot_time = 0.0
        # 监控截图端点不复用可猜测的实例名直接开放画面。令牌只存在于
        # GUI 主进程内，并由当前 PyWebIO 页面写入截图 URL。
        self.monitor_token = secrets.token_urlsafe(24)

    def start(self, func, ev: threading.Event = None) -> bool:
        with self._start_lock:
            if self.alive:
                return False
            if func is None:
                func = get_config_mod(self.config_name)
            # Each launch owns a fresh status channel, so an old worker cannot
            # overwrite the next worker's current task.
            self._task_status = State.manager.dict()
            self._process = Process(
                target=ProcessManager.run_process,
                args=(
                    self.config_name,
                    func,
                    self._renderable_queue,
                    self._screenshot_queue,
                    self._screenshot_enabled,
                    ev,
                    self._task_status,
                ),
            )
            self._process.start()
            self._active_func = func
            self.start_log_queue_handler()
            self.start_screenshot_queue_handler()
            return True

    def start_log_queue_handler(self):
        if (
            self.thd_log_queue_handler is not None
            and self.thd_log_queue_handler.is_alive()
        ):
            return
        self.thd_log_queue_handler = threading.Thread(
            target=self._thread_log_queue_handler
        )
        self.thd_log_queue_handler.start()

    def start_screenshot_queue_handler(self):
        if (
            self.thd_screenshot_queue_handler is not None
            and self.thd_screenshot_queue_handler.is_alive()
        ):
            return
        self.thd_screenshot_queue_handler = threading.Thread(
            target=self._thread_screenshot_queue_handler,
            daemon=True,
        )
        self.thd_screenshot_queue_handler.start()

    def stop(self) -> None:
        try:
            lock = self._process_locks[self.config_name]
        except KeyError:
            lock = threading.Lock()
            self._process_locks[self.config_name] = lock

        with self._start_lock, lock:
            if self.alive:
                self._process.kill()
                # Wait for the multiprocessing state to observe process exit
                # before joining queue threads; otherwise they may still see
                # alive=True for one polling cycle and report a false timeout.
                self._process.join(timeout=1)
                self.renderables.append(
                    f"[{self.config_name}] exited. Reason: Manual stop\n"
                )
            if self.thd_log_queue_handler is not None:
                self.thd_log_queue_handler.join(timeout=1)
                if self.thd_log_queue_handler.is_alive():
                    logger.warning(
                        "Log queue handler thread does not stop within 1 seconds"
                    )
            if self.thd_screenshot_queue_handler is not None:
                self.thd_screenshot_queue_handler.join(timeout=1)
                if self.thd_screenshot_queue_handler.is_alive():
                    logger.warning(
                        "Screenshot queue handler thread does not stop within 1 second"
                    )
        logger.info(f"[{self.config_name}] exited")

    def start_logout(self) -> bool:
        """Immediately kill the current task, then exclusively run logout."""
        with self._start_lock:
            if self.alive and self._active_func == 'JiangnanLogout':
                return False
            self.stop()
            # Never let two processes issue device controls at the same time.
            if self.alive:
                return False
            return self.start('JiangnanLogout')

    def _thread_log_queue_handler(self) -> None:
        while self.alive:
            try:
                log = self._renderable_queue.get(timeout=1)
            except (queue.Empty, EOFError, OSError):
                continue
            self.renderables.append(log)
            if len(self.renderables) > self.renderables_max_length:
                self.renderables = self.renderables[self.renderables_reduce_length :]
        logger.info("End of log queue handler loop")

    def _thread_screenshot_queue_handler(self) -> None:
        while self.alive:
            try:
                captured_at, image = self._screenshot_queue.get(timeout=1)
            except (queue.Empty, EOFError, OSError):
                continue
            self.latest_screenshot_time = float(captured_at)
            self.latest_screenshot = image

        # 进程刚退出时队列中可能还有最后一帧，尽量保留下来。
        try:
            while True:
                captured_at, image = self._screenshot_queue.get_nowait()
                self.latest_screenshot_time = float(captured_at)
                self.latest_screenshot = image
        except (queue.Empty, EOFError, OSError):
            pass

    @property
    def alive(self) -> bool:
        if self._process is not None:
            return self._process.is_alive()
        else:
            return False

    @property
    def current_task(self):
        if self.alive and self._task_status is not None:
            return self._task_status.get('current')
        return None

    @property
    def monitor_enabled(self) -> bool:
        return self._screenshot_enabled.is_set()

    def set_monitor_enabled(self, enabled: bool) -> None:
        if enabled:
            self._screenshot_enabled.set()
        else:
            self._screenshot_enabled.clear()

    @property
    def state(self) -> int:
        if self.alive:
            return 1
        elif len(self.renderables) == 0:
            return 2
        else:
            console = Console(no_color=True)
            with console.capture() as capture:
                console.print(self.renderables[-1])
            s = capture.get().strip()
            if s.endswith("Reason: Manual stop"):
                return 2
            elif s.endswith("Reason: Finish"):
                return 2
            elif s.endswith("Reason: Update"):
                return 4
            else:
                return 3

    @classmethod
    def get_manager(cls, config_name: str) -> "ProcessManager":
        """
        Create a new alas if not exists.
        """
        if config_name not in cls._processes:
            cls._processes[config_name] = ProcessManager(config_name)
        return cls._processes[config_name]

    @staticmethod
    def run_process(
        config_name, func: str, q: queue.Queue, screenshot_queue,
        screenshot_enabled,
        e: threading.Event = None,
        task_status=None,
    ) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--electron", action="store_true", help="Runs by electron client."
        )
        args, _ = parser.parse_known_args()
        State.electron = args.electron

        # Setup logger
        set_file_logger(name=config_name)
        if State.electron:
            # https://github.com/LmeSzinc/AzurLaneAutoScript/issues/2051
            logger.info("Electron detected, remove log output to stdout")
            from module.logger import console_hdlr
            logger.removeHandler(console_hdlr)
        set_func_logger(func=q.put)

        from module.device.monitor import set_screenshot_queue
        set_screenshot_queue(screenshot_queue, screenshot_enabled)

        from module.config.config import AzurLaneConfig

        # Remove fake PIL module, because subprocess will use it
        remove_fake_pil_module()

        AzurLaneConfig.stop_event = e
        from module.base.task_status import report_task, set_task_status_sink

        set_task_status_sink(task_status)
        try:
            if str(func).startswith('JiangnanDebug:'):
                report_task(str(func).split(':', 1)[1])
            elif func != 'alas' and func not in get_available_mod():
                report_task(str(func).split(':', 1)[0])
            # Run alas
            if func == "alas":
                from alas import AzurLaneAutoScript

                if e is not None:
                    AzurLaneAutoScript.stop_event = e
                AzurLaneAutoScript(config_name=config_name).loop()
            elif func == 'JiangnanLogout':
                from module.jiangnan.logout import run_logout

                run_logout(config_name)
            elif str(func).startswith("JiangnanDebug:"):
                from module.jiangnan.debug import run_debug_task

                task = str(func).split(':', 1)[1]
                if not run_debug_task(config_name, task):
                    logger.error(f"[{config_name}] exited. Reason: Debug task failed\n")
                    return
            elif str(func).startswith("JiangnanAuxiliaryRecognize:"):
                from module.jiangnan.auxiliary_recognition import (
                    run_auxiliary_recognition,
                )

                _, kind, target = str(func).split(":", 2)
                run_auxiliary_recognition(
                    config_name=config_name,
                    kind=kind,
                    target=target,
                )
            elif func in get_available_func():
                from alas import AzurLaneAutoScript

                AzurLaneAutoScript(config_name=config_name).run(inflection.underscore(func), skip_first_screenshot=True)
            elif func in get_available_mod():
                mod = load_mod(func)

                if e is not None:
                    mod.set_stop_event(e)
                mod.loop(config_name)
            elif func in get_available_mod_func():
                getattr(load_mod(get_func_mod(func)), inflection.underscore(func))(config_name)
            else:
                logger.critical(f"No function matched: {func}")
            logger.info(f"[{config_name}] exited. Reason: Finish\n")
        except Exception as e:
            logger.exception(e)
        finally:
            report_task()
            set_task_status_sink(None)

    @classmethod
    def running_instances(cls) -> List["ProcessManager"]:
        l = []
        for process in cls._processes.values():
            if process.alive:
                l.append(process)
        return l

    @staticmethod
    def restart_processes(
        instances: List[Union["ProcessManager", str]] = None, ev: threading.Event = None
    ):
        """
        After update and reload, or failed to perform an update,
        restart all alas that running before update
        """
        logger.hr("Restart alas")

        # Load MOD_CONFIG_DICT
        list_mod_instance()

        if instances is None:
            instances = []

        _instances = set()

        for instance in instances:
            if isinstance(instance, str):
                _instances.add(ProcessManager.get_manager(instance))
            elif isinstance(instance, ProcessManager):
                _instances.add(instance)

        try:
            with open("./config/reloadalas", mode="r") as f:
                for line in f.readlines():
                    line = line.strip()
                    _instances.add(ProcessManager.get_manager(line))
        except FileNotFoundError:
            pass

        for process in _instances:
            logger.info(f"Starting [{process.config_name}]")
            process.start(func=get_config_mod(process.config_name), ev=ev)

        try:
            os.remove("./config/reloadalas")
        except:
            pass
        logger.info("Start alas complete")
