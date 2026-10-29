import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional
from urllib.parse import quote

# Import fake module before import pywebio to avoid importing unnecessary module PIL
from module.webui.fake_pil_module import import_fake_pil_module

import_fake_pil_module()

from pywebio import config as webconfig
from pywebio.input import file_upload, input, input_group, select
from pywebio.output import (
    Output,
    clear,
    close_popup,
    popup,
    put_button,
    put_buttons,
    put_collapse,
    put_column,
    put_error,
    put_html,
    put_link,
    put_loading,
    put_markdown,
    put_row,
    put_scope,
    put_table,
    put_text,
    put_warning,
    toast,
    use_scope,
)
from pywebio.pin import pin, pin_on_change
from pywebio.session import download, go_app, info, local, register_thread, run_js, set_env

import module.webui.lang as lang
from module.config.config import AzurLaneConfig, Function
from module.config.deep import deep_get, deep_iter, deep_set
from module.config.env import IS_ON_PHONE_CLOUD
from module.config.server import to_server
from module.config.utils import (
    alas_instance,
    alas_template,
    dict_to_kv,
    filepath_args,
    filepath_config,
    read_file,
)
from module.logger import logger
from module.jiangnan.canal_route import load_topology, parse_custom_route
from module.jiangnan.debug import debug_command, debug_task_names, visible_menu
from module.jiangnan.task_priority import priority_updates, task_priority_rows
from module.jiangnan.chaomu_drum_state import drum_next_run
from module.submodule.submodule import load_config
from module.submodule.utils import get_config_mod
from module.webui.base import Frame
from module.webui.fastapi import CANAL_MAP_DIR, INN_MAP_DIR, asgi_app
from module.jiangnan.inn_map import map_instance_id
from module.webui.lang import _t, t
from module.webui.patch import fix_py37_subprocess_communicate, patch_executor, patch_mimetype
from module.webui.pin import put_input, put_select
from module.webui.process_manager import ProcessManager
from module.webui.remote_access import RemoteAccess
from module.webui.setting import State
from module.webui.task_priority import put_task_priority_list
from module.webui.updater import updater
from module.webui.utils import (
    Icon,
    Switch,
    TaskHandler,
    add_css,
    filepath_css,
    get_alas_config_listen_path,
    get_localstorage,
    get_window_visibility_state,
    login,
    parse_pin_value,
    raise_exception,
    re_fullmatch,
    to_pin_value,
)
from module.webui.widgets import (
    BinarySwitchButton,
    RichLog,
    T_Output_Kwargs,
    put_icon_buttons,
    put_loading_text,
    put_none,
    put_output,
)

patch_executor()
patch_mimetype()
fix_py37_subprocess_communicate()
task_handler = TaskHandler()


class AlasGUI(Frame):
    ALAS_MENU: Dict[str, Dict[str, List[str]]]
    ALAS_ARGS: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]
    theme = "default"

    def initial(self) -> None:
        self.ALAS_MENU = read_file(filepath_args("menu", self.alas_mod))
        self.ALAS_ARGS = read_file(filepath_args("args", self.alas_mod))
        self._init_alas_config_watcher()

    def __init__(self) -> None:
        super().__init__()
        # modified keys, return values of pin_wait_change()
        self.modified_config_queue = queue.Queue()
        # alas config name
        self.alas_name = ""
        self.alas_mod = "alas"
        self.alas_config = AzurLaneConfig("template")
        self.initial()
        # rendered state cache
        self.rendered_cache = []
        self.inst_cache = []
        self.load_home = False
        self.af_flag = False
        # 侧栏会在 run() 完成后台任务注册前先显示。用户此时点击实例会
        # 立即进入 ui_alas()，因此状态开关必须在构造阶段就存在，不能
        # 等 show() 之后再创建。
        self.state_switch = Switch(
            status=self.set_status,
            get_state=lambda: getattr(getattr(self, "alas", -1), "state", 0),
            name="state",
        )

    @use_scope("aside", clear=True)
    def set_aside(self) -> None:
        # TODO: update put_icon_buttons()

        current_date = datetime.now().date()
        if current_date.month == 4 and current_date.day == 1:
            self.af_flag = True

        put_icon_buttons(
            Icon.DEVELOP,
            buttons=[{"label": t("Gui.Aside.Home"), "value": "Home", "color": "aside"}],
            onclick=[self.ui_develop],
        )
        put_scope("aside_instance", [
            put_scope(f"alas-instance-{i}", [])
            for i, _ in enumerate(alas_instance())
        ])
        self.set_aside_status()
        put_icon_buttons(
            Icon.SETTING,
            buttons=[
                {
                    "label": t("Gui.AddAlas.Manage"),
                    "value": "AddAlas",
                    "color": "aside",
                }
            ],
            onclick=[lambda: go_app("manage", new_window=False)],
        )


    @use_scope("aside_instance")
    def set_aside_status(self) -> None:
        flag = True

        def update(name, seq):
            with use_scope(f"alas-instance-{seq}", clear=True):
                icon_html = Icon.RUN
                rendered_state = ProcessManager.get_manager(inst).state
                if rendered_state == 1 and self.af_flag:
                    icon_html = icon_html[:31] + ' anim-rotate' + icon_html[31:]
                put_icon_buttons(
                    icon_html,
                    buttons=[{"label": name, "value": name, "color": "aside"}],
                    onclick=self.ui_alas,
                )
            return rendered_state

        if not len(self.rendered_cache) or self.load_home:
            # Reload when add/delete new instance | first start app.py | go to HomePage (HomePage load call force reload)
            flag = False
            self.inst_cache.clear()
            self.inst_cache = alas_instance()
        if flag:
            for index, inst in enumerate(self.inst_cache):
                # Check for state change
                state = ProcessManager.get_manager(inst).state
                if state != self.rendered_cache[index]:
                    self.rendered_cache[index] = update(inst, index)
                    flag = False
        else:
            self.rendered_cache.clear()
            clear("aside_instance")
            for index, inst in enumerate(self.inst_cache):
                self.rendered_cache.append(update(inst, index))
            self.load_home = False
        if not flag:
            # Redraw lost focus, now focus on aside button
            aside_name = get_localstorage("aside")
            self.active_button("aside", aside_name)

        return

    @use_scope("header_status")
    def set_status(self, state: int) -> None:
        """
        Args:
            state (int):
                1 (running)
                2 (not running)
                3 (warning, stop unexpectedly)
                4 (stop for update)
                0 (hide)
                -1 (*state not changed)
        """
        if state == -1:
            return
        clear()

        if state == 1:
            put_loading_text(t("Gui.Status.Running"), color="success")
        elif state == 2:
            put_loading_text(t("Gui.Status.Inactive"), color="secondary", fill=True)
        elif state == 3:
            put_loading_text(t("Gui.Status.Warning"), shape="grow", color="warning")
        elif state == 4:
            put_loading_text(t("Gui.Status.Updating"), shape="grow", color="success")

    @classmethod
    def set_theme(cls, theme="default") -> None:
        cls.theme = theme
        State.deploy_config.Theme = theme
        State.theme = theme
        webconfig(theme=theme)

    @use_scope("menu", clear=True)
    def alas_set_menu(self) -> None:
        """
        Set menu
        """
        put_buttons(
            [
                {
                    "label": t("Gui.MenuAlas.Overview"),
                    "value": "Overview",
                    "color": "menu",
                }
            ],
            onclick=[self.alas_overview],
        ).style(f"--menu-Overview--")

        menus = visible_menu(self.ALAS_MENU) if self.alas_mod == 'alas' else self.ALAS_MENU
        for menu, task_data in menus.items():
            if task_data.get("page") == "tool":
                _onclick = self.alas_daemon_overview
            else:
                _onclick = self.alas_set_group

            if task_data.get("menu") == "collapse":
                task_btn_list = [
                    put_buttons(
                        [
                            {
                                "label": (t('Gui.JiangnanDebug.RuntimeSettings')
                                          if self.alas_mod == 'alas' and task == 'Alas'
                                          else t(f"Task.{task}.name")),
                                "value": task,
                                "color": "menu",
                            }
                        ],
                        onclick=_onclick,
                    ).style(f"--menu-{task}--")
                    for task in task_data.get("tasks", [])
                ]
                title = (t('Gui.JiangnanDebug.RuntimeSettings')
                         if self.alas_mod == 'alas' and menu == 'Alas'
                         else t(f'Menu.{menu}.name'))
                put_collapse(title=title, content=task_btn_list)
            else:
                title = t(f"Menu.{menu}.name")
                put_html(
                    '<div class="hr-task-group-box">'
                    '<span class="hr-task-group-line"></span>'
                    f'<span class="hr-task-group-text">{title}</span>'
                    '<span class="hr-task-group-line"></span>'
                    '</div>'
                )
                for task in task_data.get("tasks", []):
                    put_buttons(
                        [
                            {
                                "label": t(f"Task.{task}.name"),
                                "value": task,
                                "color": "menu",
                            }
                        ],
                        onclick=_onclick,
                    ).style(f"--menu-{task}--").style(f"padding-left: 0.75rem")

        self.alas_overview()

    @use_scope("content", clear=True)
    def alas_set_group(self, task: str) -> None:
        """
        Set arg groups from dict
        """
        self.init_menu(name=task)
        self.set_title(t('Gui.JiangnanDebug.RuntimeSettings')
                       if self.alas_mod == 'alas' and task == 'Alas'
                       else t(f"Task.{task}.name"))

        put_scope("_groups", [put_none(), put_scope("groups"), put_scope("navigator")])

        task_help: str = t(f"Task.{task}.help")
        if task_help:
            put_scope(
                "group__info",
                scope="groups",
                content=[put_text(task_help).style("font-size: 1rem")],
            )

        config = self.alas_config.read_file(self.alas_name)
        if task == 'JiangnanCornDryingMat':
            put_scope('corn-production-countdowns', scope='groups')
            self.refresh_corn_countdowns(config)
        if task == 'JiangnanTaskPriority':
            put_scope('task-priority-current', scope='groups')
            self._task_priority_signature = None
            self.refresh_task_priority(config)
            return
        is_city_production = (
            task.startswith('Jiangnan')
            and (
                task.endswith('fuProduction')
                or task in {
                    'JiangnanJimingshanProduction',
                    'JiangnanJiangningxianProduction',
                    'JiangnanBaixuezhenProduction',
                    'JiangnanZongyingdaoProduction',
                    'JiangnanLonggongProduction',
                }
            )
        )
        production_labels = []
        if is_city_production:
            city_key = task[len('Jiangnan'):-len('Production')].lower()
            layout_text = deep_get(
                config,
                [
                    'JiangnanAuxiliaryProductionLayout',
                    f'production_layout_{city_key}',
                    'Distribution',
                ],
                '',
            )
            for line in str(layout_text or '').splitlines():
                values = line.split(':', 1)[-1].replace('，', ',')
                production_labels.extend(
                    value.strip() for value in values.split(',') if value.strip()
                )
        for group, arg_dict in deep_iter(self.ALAS_ARGS[task], depth=1):
            group_name = group[0]
            if (self.alas_mod == 'alas' and task == 'Alas'
                    and group_name not in {'Emulator', 'EmulatorInfo', 'Error', 'Optimization'}):
                continue
            # 城市生产共有 65 种设施，但每座城市实际只拥有其中一小
            # 部分。页面每次打开都重新读取该城生产辅助信息，只渲染
            # 当前城市实际存在的模块，显著缩短首屏加载时间。
            if is_city_production and group_name.startswith('city_facility_'):
                facility = _t(f'{group_name}._info.name', 'zh-CN')
                if facility == '农田':
                    visible = any(
                        '农田' in label and '改良农田' not in label
                        for label in production_labels
                    )
                else:
                    visible = any(
                        facility in label for label in production_labels
                    )
                enable_path = f'{task}.{group_name}.Enable'
                current = bool(deep_get(config, enable_path, False))
                if current != visible:
                    # 辅助信息是城市设施开关的唯一事实来源。每次打开
                    # 页面都同步一次；识别按钮和城市生产运行时也会用
                    # 同一规则更新，因此不会保留过期的手工开关状态。
                    deep_set(config, enable_path, visible)
                    self.modified_config_queue.put({
                        'name': enable_path,
                        'value': visible,
                    })
                if not visible:
                    continue
            if self.set_group(group, arg_dict, config, task):
                self.set_navigator(group)

        if task == 'JiangnanYingtianInn':
            with use_scope('groups'):
                if all((INN_MAP_DIR / name).is_file() for name in (
                        'overview.html', 'viewer.js', 'viewer.css', 'panorama.png')):
                    instance = map_instance_id(self.alas_name)
                    put_html(
                        '<div class="canal-map-panel inn-map-panel">'
                        '<p>应天府客栈全景地图 · 点击取坐标，滚轮缩放，拖动平移</p>'
                        f'<iframe src="/inn-map/overview.html?instance={instance}" '
                        'title="应天府客栈全景定位" loading="lazy"></iframe>'
                        '</div>'
                    )
                else:
                    put_warning('客栈全景地图文件不完整，请检查 assets/jiangnan/inn/map。')

        if task == 'JiangnanYangzhouGrandCanal':
            with use_scope('groups'):
                map_ready = (
                    (CANAL_MAP_DIR / 'overview.html').is_file()
                    and (CANAL_MAP_DIR / 'overview_preview.jpg').is_file()
                    and (CANAL_MAP_DIR / 'map_tiles.js').is_file()
                    and (CANAL_MAP_DIR / 'custom_route.js').is_file()
                )
                if map_ready:
                    preview_route = quote(str(deep_get(
                        config, [task, 'yangzhou_grand_canal', 'CustomRoute'], ''
                    ) or ''), safe='')
                    current_point = quote(str(deep_get(
                        config, [task, 'yangzhou_grand_canal', 'LastPoint'], ''
                    ) or ''), safe='')
                    put_html(
                        '<div class="canal-map-panel">'
                        '<p>大运河地图 · 紫色：自定义路线 · 橙色：当前位置</p>'
                        f'<iframe src="/canal-map/overview.html?route={preview_route}&amp;point={current_point}" '
                        'title="扬州大运河地图" loading="lazy"></iframe>'
                        '</div>'
                    )
                else:
                    put_warning(
                        '未找到地图。请将 overview.html、overview_preview.jpg、map_tiles.js、custom_route.js 和 tiles 文件夹 '
                        '放在 assets/jiangnan/grand_canal/map 文件夹中。'
                    )

    def refresh_corn_countdowns(self, config=None) -> None:
        """Show each event city's own persisted wake without modifying either clock."""
        if not self.visible or self.page != 'JiangnanCornDryingMat' or not self.alas_name:
            return
        if config is None:
            config = read_file(filepath_config(self.alas_name, self.alas_mod))
        now = datetime.now().replace(microsecond=0)
        rows = []
        for key in ('yingtianfu', 'suzhoufu'):
            group = f'production_{key}'
            data = deep_get(config, ['JiangnanCornDryingMat', group], {})
            try:
                finish = datetime.fromisoformat(str(data.get('NextRun')))
            except (TypeError, ValueError):
                finish = now
            if not data.get('Enable', True):
                remaining = t('Gui.CornDryingMat.Disabled')
            elif finish <= now:
                remaining = t('Gui.CornDryingMat.Due')
            else:
                seconds = int((finish - now).total_seconds())
                remaining = f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
            rows.append([t(f'{group}._info.name'), remaining])
        with use_scope('corn-production-countdowns', clear=True):
            put_table(rows, header=[t('Gui.CornDryingMat.City'), t('Gui.CornDryingMat.Countdown')])

    def refresh_task_priority(self, config=None) -> None:
        """Show the saved order as a draggable list."""
        if not self.visible or self.page != 'JiangnanTaskPriority' or not self.alas_name:
            return
        if config is None:
            config = read_file(filepath_config(self.alas_name, self.alas_mod))
        rows = task_priority_rows(config, self.alas_config.SCHEDULER_PRIORITY)
        signature = (self.alas_name, tuple(rows), lang.LANG)
        if signature == getattr(self, '_task_priority_signature', None):
            return
        self._task_priority_signature = signature
        config_name = self.alas_name
        config_updater = self.alas_config

        def save_order(order):
            return self.save_task_priority(order, config_name, config_updater)

        with use_scope('task-priority-current', clear=True):
            put_task_priority_list(rows, save_order)

    def save_task_priority(self, order, config_name, config_updater):
        """Save a complete order together, bound to the instance that rendered it."""
        config = config_updater.read_file(config_name)
        try:
            modified = priority_updates(order, config, config_updater.SCHEDULER_PRIORITY)
        except ValueError as exc:
            logger.warning(f'Invalid task order: {exc}')
            return False, [row[0] for row in task_priority_rows(config, config_updater.SCHEDULER_PRIORITY)]
        self._save_config(modified.copy(), config_name, config_updater)
        saved = config_updater.read_file(config_name)
        success = all(deep_get(saved, key) == value for key, value in modified.items())
        return success, [row[0] for row in task_priority_rows(saved, config_updater.SCHEDULER_PRIORITY)]

    def refresh_canal_point(self) -> None:
        """Keep the editable point display current without interrupting typing."""
        if (not self.visible or self.page != 'JiangnanYangzhouGrandCanal'
                or not self.alas_name):
            return
        if time.monotonic() - getattr(self, '_last_canal_point_edit_at', 0) < 4:
            return
        # This poll only needs two saved fields. Full config migration is
        # already handled when opening the settings page or saving changes.
        config = read_file(filepath_config(self.alas_name, self.alas_mod))
        point = str(deep_get(
            config,
            'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.LastPoint',
            '',
        ) or '')
        route = str(deep_get(
            config,
            'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.CustomRoute',
            '',
        ) or '')
        run_js(
            """
            (() => {
                const field = document.querySelector(
                    'input[name="JiangnanYangzhouGrandCanal_yangzhou_grand_canal_LastPoint"]'
                );
                if (field && document.activeElement !== field && field.value !== point) {
                    field.value = point;
                }
                const frame = document.querySelector('.canal-map-panel iframe');
                const routeField = document.querySelector(
                    '[name="JiangnanYangzhouGrandCanal_yangzhou_grand_canal_CustomRoute"]'
                );
                if (frame && frame.contentWindow) {
                    frame.contentWindow.postMessage({
                        type: 'canal-map-state',
                        route: routeField ? routeField.value : route,
                        point: field ? field.value : point,
                    }, location.origin);
                }
            })();
            """,
            point=point,
            route=route,
        )

    @use_scope("groups")
    def set_group(self, group, arg_dict, config, task):
        group_name = group[0]
        server = to_server(deep_get(config, "Alas.Emulator.PackageName", "cn"))

        output_list: List[Output] = []
        for arg, arg_dict in deep_iter(arg_dict, depth=1):
            if task == 'Alas' and group_name == 'Optimization' and arg[0] == 'CombatScreenshotInterval':
                continue
            output_kwargs: T_Output_Kwargs = arg_dict.copy()

            # Skip hide
            display: Optional[str] = output_kwargs.pop("display", None)
            if display == "hide":
                continue
            # Disable
            elif display == "disabled":
                output_kwargs["disabled"] = True
            # Output type
            output_kwargs["widget_type"] = output_kwargs.pop("type")

            arg_name = arg[0]  # [arg_name,]
            # Internal pin widget name
            output_kwargs["name"] = f"{task}_{group_name}_{arg_name}"
            # Display title
            output_kwargs["title"] = t(f"{group_name}.{arg_name}.name")

            # Get value from config
            value = deep_get(
                config, [task, group_name, arg_name], output_kwargs["value"]
            )
            # idk
            value = str(value) if isinstance(value, datetime) else value
            # Default value
            output_kwargs["value"] = value
            # Options
            options = output_kwargs.pop("option", [])
            server_options = output_kwargs.get(f"option_{server}")
            if output_kwargs["widget_type"] == "select" and isinstance(server_options, list) and server_options:
                options = server_options
            output_kwargs["options"] = options
            if (
                task == "GemsFarming"
                and group_name == "Campaign"
                and arg_name == "Event"
                and output_kwargs["widget_type"] == "select"
                and len(options) == 1
            ):
                continue
            if output_kwargs["widget_type"] == "select" and len(options) == 1:
                only_option = options[0]
                if only_option in output_kwargs.get("option_bold", []):
                    output_kwargs["widget_type"] = "state"
            # Options label
            options_label = []
            for opt in options:
                options_label.append(t(f"{group_name}.{arg_name}.{opt}"))
            output_kwargs["options_label"] = options_label
            # Help
            arg_help = t(f"{group_name}.{arg_name}.help")
            if arg_help == "" or not arg_help:
                arg_help = None
            output_kwargs["help"] = arg_help
            # Invalid feedback
            output_kwargs["invalid_feedback"] = t("Gui.Text.InvalidFeedBack", value)

            o = put_output(output_kwargs)
            if o is not None:
                # output will inherit current scope when created, override here
                o.spec["scope"] = f"#pywebio-scope-group_{group_name}"
                output_list.append(o)

        if not output_list:
            return 0

        with use_scope(f"group_{group_name}"):
            put_text(t(f"{group_name}._info.name"))
            group_help = t(f"{group_name}._info.help")
            if group_help != "":
                put_text(group_help)
            put_html('<hr class="hr-group">')
            for output in output_list:
                output.show()

        return len(output_list)

    @use_scope("navigator")
    def set_navigator(self, group):
        js = f"""
            $("#pywebio-scope-groups").scrollTop(
                $("#pywebio-scope-group_{group[0]}").position().top
                + $("#pywebio-scope-groups").scrollTop() - 59
            )
        """
        put_button(
            label=t(f"{group[0]}._info.name"),
            onclick=lambda: run_js(js),
            color="navigator",
        )

    @use_scope("content", clear=True)
    def alas_overview(self) -> None:
        self.init_menu(name="Overview")
        self.set_title(t(f"Gui.MenuAlas.Overview"))
        self._overview_config_signature = None
        self._overview_task_snapshot = None

        put_scope("overview", [put_scope("schedulers"), put_scope("logs")])

        with use_scope("schedulers"):
            put_scope(
                "scheduler-bar",
                [
                    put_text(t("Gui.Overview.Scheduler")).style(
                        "font-size: 1.25rem; margin: auto .5rem auto;"
                    ),
                    self.put_scheduler_actions(),
                ],
            )
            put_scope(
                "running",
                [
                    put_text(t("Gui.Overview.Running")),
                    put_html('<hr class="hr-group">'),
                    put_scope("running_tasks"),
                ],
            )
            put_scope(
                "pending",
                [
                    put_text(t("Gui.Overview.Pending")),
                    put_html('<hr class="hr-group">'),
                    put_scope("pending_tasks"),
                ],
            )
            put_scope(
                "waiting",
                [
                    put_text(t("Gui.Overview.Waiting")),
                    put_html('<hr class="hr-group">'),
                    put_scope("waiting_tasks"),
                ],
            )

        switch_scheduler = BinarySwitchButton(
            label_on=t("Gui.Button.Stop"),
            label_off=t("Gui.Button.Start"),
            onclick_on=lambda: self.alas.stop(),
            onclick_off=lambda: self.alas.start(None, updater.event),
            get_state=lambda: self.alas.alive,
            color_on="off",
            color_off="on",
            scope="scheduler_btn",
        )

        log = RichLog("log")

        with use_scope("logs"):
            put_scope(
                "monitor",
                [
                    put_scope(
                        "monitor-bar",
                        [
                            put_text(t("Gui.Overview.Monitor")).style(
                                "font-size: 1.25rem; margin: auto .5rem auto;"
                            ),
                            put_scope("monitor_btn"),
                        ],
                    ),
                    put_html(
                        '<div class="monitor-frame">'
                        '<img id="alas-monitor-screenshot" alt="monitor screenshot">'
                        '<span id="alas-monitor-empty">'
                        f'{t("Gui.Overview.NoScreenshot")}'
                        '</span></div>'
                    ),
                ],
            )
            put_scope(
                "log-bar",
                [
                    put_text(t("Gui.Overview.Log")).style(
                        "font-size: 1.25rem; margin: auto .5rem auto;"
                    ),
                    put_scope(
                        "log-bar-btns",
                        [
                            put_scope("log_scroll_btn"),
                        ],
                    ),
                ],
            )
            put_scope("log", [put_html("")])

        monitor_storage_key = f"alas.monitor.{self.alas_name}"
        stored_monitor_state = get_localstorage(monitor_storage_key)
        if stored_monitor_state is not None:
            self.alas.set_monitor_enabled(stored_monitor_state != "false")
        self.set_screenshot_monitor_enabled(self.alas.monitor_enabled, persist=False)

        log.console.width = log.get_width()

        switch_log_scroll = BinarySwitchButton(
            label_on=t("Gui.Button.ScrollON"),
            label_off=t("Gui.Button.ScrollOFF"),
            onclick_on=lambda: log.set_scroll(False),
            onclick_off=lambda: log.set_scroll(True),
            get_state=lambda: log.keep_bottom,
            color_on="on",
            color_off="off",
            scope="log_scroll_btn",
        )

        switch_monitor = BinarySwitchButton(
            label_on=t("Gui.Button.MonitorON"),
            label_off=t("Gui.Button.MonitorOFF"),
            onclick_on=lambda: self.set_screenshot_monitor_enabled(False),
            onclick_off=lambda: self.set_screenshot_monitor_enabled(True),
            get_state=lambda: self.alas.monitor_enabled,
            color_on="on",
            color_off="off",
            scope="monitor_btn",
        )

        self.task_handler.add(switch_scheduler.g(), 1, True)
        self.task_handler.add(switch_log_scroll.g(), 1, True)
        self.task_handler.add(switch_monitor.g(), 1, True)
        self.task_handler.add(self.alas_update_overview_task, 1, True)
        self.task_handler.add(log.put_log(self.alas), 0.25, True)

    def set_screenshot_monitor_enabled(
        self, enabled: bool, persist: bool = True
    ) -> None:
        """切换监控视窗，同时停止/恢复子进程中的截图编码。"""
        enabled = bool(enabled)
        self.alas.set_monitor_enabled(enabled)
        if persist:
            run_js(
                "localStorage.setItem(key, value)",
                key=f"alas.monitor.{self.alas_name}",
                value="true" if enabled else "false",
            )
        run_js(
            """
            const frame = document.querySelector(
                '#pywebio-scope-monitor .monitor-frame'
            );
            const logs = document.getElementById('pywebio-scope-logs');
            if (logs) logs.classList.toggle('monitor-disabled', !enabled);
            if (frame) frame.style.display = enabled ? 'flex' : 'none';
            const screenshot = document.getElementById(
                'alas-monitor-screenshot'
            );
            const empty = document.getElementById('alas-monitor-empty');
            const timerKey = '__alasMonitorSnapshotTimer';
            const requestKey = '__alasMonitorSnapshotInFlight';
            if (window[timerKey]) {
                clearInterval(window[timerKey]);
                window[timerKey] = null;
            }
            window[requestKey] = false;
            const refresh = () => {
                if (!enabled || !screenshot || window[requestKey]) return;
                window[requestKey] = true;
                fetch(snapshotUrl + '&_=' + Date.now(), {
                    cache: 'no-store'
                }).then((response) => {
                    if (!response.ok || response.status === 204) return null;
                    return response.blob();
                }).then((blob) => {
                    if (!blob || !enabled || !screenshot) return;
                    const oldUrl = screenshot.dataset.objectUrl;
                    const nextUrl = URL.createObjectURL(blob);
                    screenshot.onload = () => {
                        if (oldUrl) URL.revokeObjectURL(oldUrl);
                    };
                    screenshot.dataset.objectUrl = nextUrl;
                    screenshot.src = nextUrl;
                    screenshot.style.display = 'block';
                    if (empty) empty.style.display = 'none';
                }).catch(() => {
                    // 网络短暂波动时保留上一次成功显示的截图。
                }).finally(() => {
                    window[requestKey] = false;
                });
            };
            if (screenshot) {
                if (enabled) {
                    refresh();
                    window[timerKey] = setInterval(refresh, 1000);
                } else {
                    screenshot.style.display = 'none';
                    const oldUrl = screenshot.dataset.objectUrl;
                    if (oldUrl) URL.revokeObjectURL(oldUrl);
                    delete screenshot.dataset.objectUrl;
                    screenshot.removeAttribute('src');
                    if (empty) empty.style.display = 'inline';
                }
            }
            """,
            enabled=enabled,
            snapshotUrl=(
                f'/monitor/{quote(self.alas_name, safe="")}.jpg'
                f'?token={quote(self.alas.monitor_token, safe="")}'
            ),
        )

    def _init_alas_config_watcher(self) -> None:
        def put_queue(path, value):
            if (path.split('.')[0].endswith('PostStation')
                    and path.endswith('.IntervalRations')):
                from module.jiangnan.post_station_rules import interval_label

                try:
                    label = interval_label(value)
                except (ValueError, TypeError):
                    label = '请输入大于0的粮草量'
                pin[path.rsplit('.',1)[0].replace('.','_')+'_Interval'] = label
            if path == 'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.LastPoint':
                self._last_canal_point_edit_at = time.monotonic()
            self.modified_config_queue.put({"name": path, "value": value})

        for path in get_alas_config_listen_path(self.ALAS_ARGS):
            pin_on_change(
                name="_".join(path), onchange=partial(put_queue, ".".join(path))
            )
        logger.info("Init config watcher done.")

    def _alas_thread_update_config(self) -> None:
        modified = {}
        while self.alive:
            try:
                d = self.modified_config_queue.get(timeout=10)
                config_name = self.alas_name
                config_updater = self.alas_config
            except queue.Empty:
                continue
            modified[d["name"]] = d["value"]
            while True:
                try:
                    d = self.modified_config_queue.get(timeout=1)
                    modified[d["name"]] = d["value"]
                except queue.Empty:
                    self._save_config(modified, config_name, config_updater)
                    modified.clear()
                    break

    def _save_config(
            self,
            modified: Dict[str, str],
            config_name: str,
            config_updater: AzurLaneConfig = State.config_updater,
    ) -> None:
        try:
            valid = []
            invalid = []
            reschedule_tasks = set()
            config = config_updater.read_file(config_name)
            n = datetime.now()
            for p, v in deep_iter(config, depth=3):
                if p[-1].endswith('un') and not isinstance(v, bool):
                    if (v - n).days >= 31:
                        deep_set(config, p, '')
            for k, v in modified.copy().items():
                valuetype = deep_get(self.ALAS_ARGS, k + ".valuetype")
                v = parse_pin_value(v, valuetype)
                validate = deep_get(self.ALAS_ARGS, k + ".validate")
                if k == 'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.CustomRoute':
                    try:
                        parse_custom_route(v)
                    except ValueError as exc:
                        modified.pop(k)
                        invalid.append(k)
                        logger.warning(f'Invalid Grand Canal route: {exc}')
                        continue
                if k == 'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.LastPoint':
                    v = str(v).strip().upper()
                    if v and v not in load_topology()[0]:
                        modified.pop(k)
                        invalid.append(k)
                        logger.warning(f'Unknown Grand Canal map point: {v}')
                        continue
                    source_key = 'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.LastPointSource'
                    previous_key = 'JiangnanYangzhouGrandCanal.yangzhou_grand_canal.PreviousPoint'
                    deep_set(config, source_key, 'manual')
                    deep_set(config, previous_key, '')
                    modified[source_key] = 'manual'
                    modified[previous_key] = ''
                    pin['_'.join(k.split('.'))] = v
                reservation_names = (
                    k.startswith('JiangnanSpecialResidentReservation.reserved_')
                    and k.endswith('.Names')
                )
                if not len(str(v)) and not reservation_names:
                    default = deep_get(self.ALAS_ARGS, k + ".value")
                    modified[k] = default
                    deep_set(config, k, default)
                    valid.append(k)
                    pin["_".join(k.split("."))] = default

                elif not validate or re_fullmatch(validate, v):
                    deep_set(config, k, v)
                    modified[k] = v
                    valid.append(k)
                    key_parts = k.split('.')
                    if (
                        len(key_parts) == 3
                        and key_parts[0].startswith('Jiangnan')
                        and key_parts[1] != 'Scheduler'
                        and key_parts[2] in {
                            'NextRun', 'Enable', 'Scan', 'Export', 'Import', 'TradeThreshold'
                        }
                    ):
                        reschedule_tasks.add(key_parts[0])
                    for set_key, set_value in config_updater.save_callback(
                        k, v, config=config
                    ):
                        modified[set_key] = set_value
                        deep_set(config, set_key, set_value)
                        valid.append(set_key)
                        pin["_".join(set_key.split("."))] = to_pin_value(set_value)
                else:
                    modified.pop(k)
                    invalid.append(k)
                    logger.warning(f"Invalid value {v} for key {k}, skip saving.")

            # A city clock is user-editable.  Keep the read-only page clock in
            # sync immediately so the scheduler wakes at the earliest enabled
            # city instead of waiting for the task's previous total time.
            for task in reschedule_tasks:
                if task == 'JiangnanHuizhouMerchantSale':
                    # Reevaluate edited settings once. The merchant workflow
                    # decides its next wake from stock and busy hall clocks.
                    total_key = f'{task}.Scheduler.NextRun'
                    total = n.replace(microsecond=0)
                    deep_set(config, total_key, total)
                    modified[total_key] = total
                    valid.append(total_key)
                    pin['_'.join(total_key.split('.'))] = to_pin_value(total)
                    continue
                city_times = []
                for group, group_data in config.get(task, {}).items():
                    if group == 'Scheduler' or not isinstance(group_data, dict):
                        continue
                    next_run = group_data.get('NextRun')
                    if not next_run:
                        continue
                    enabled = group_data.get('Enable')
                    if enabled is None:
                        enabled = group_data.get('Scan')
                    if enabled is None and (
                        'Export' in group_data or 'Import' in group_data
                    ):
                        enabled = bool(group_data.get('Export')) or bool(
                            group_data.get('Import')
                        )
                    if enabled is False:
                        continue
                    try:
                        if not isinstance(next_run, datetime):
                            next_run = datetime.fromisoformat(str(next_run))
                    except (TypeError, ValueError):
                        logger.warning(
                            f'Invalid Jiangnan city NextRun: '
                            f'{task}.{group}.NextRun={next_run}'
                        )
                        continue
                    if task == 'JiangnanChaomuDrum' and group.startswith('chaomu_drum_'):
                        next_run = drum_next_run(group_data, n)
                        city_key = f'{task}.{group}.NextRun'
                        deep_set(config, city_key, next_run)
                        modified[city_key] = next_run
                        valid.append(city_key)
                        pin['_'.join(city_key.split('.'))] = to_pin_value(next_run)
                    city_times.append(next_run.replace(microsecond=0))
                if city_times:
                    total_key = f'{task}.Scheduler.NextRun'
                    total = min(city_times)
                    deep_set(config, total_key, total)
                    modified[total_key] = total
                    valid.append(total_key)
                    pin['_'.join(total_key.split('.'))] = to_pin_value(total)
            self.pin_remove_invalid_mark(valid)
            self.pin_set_invalid_mark(invalid)
            if modified:
                toast(
                    t("Gui.Toast.ConfigSaved"),
                    duration=1,
                    position="right",
                    color="success",
                )
                logger.info(
                    f"Save config {filepath_config(config_name)}, {dict_to_kv(modified)}"
                )
                config_updater.write_file(config_name, config)
        except Exception as e:
            logger.exception(e)

    def alas_update_overview_task(self) -> None:
        if not self.visible:
            return
        stat = os.stat(filepath_config(self.alas_name, self.alas_mod))
        signature = (id(self.alas_config), stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature != getattr(self, '_overview_config_signature', None):
            self.alas_config.load()
            self._overview_config_signature = signature
        self.alas_config.get_next_task()

        current = self.alas.current_task
        running = []
        if current is not None:
            command, started_at = current
            task = Function({})
            task.command = command
            task.next_run = started_at
            running.append(task)
        pending = self.alas_config.pending_task
        waiting = self.alas_config.waiting_task
        # A task may already have saved its next run while still executing.
        # Keep it in the running column until the worker reports completion.
        if current is not None:
            pending = [task for task in pending if task.command != command]
            waiting = [task for task in waiting if task.command != command]
        if self.alas_mod == 'alas':
            pending = [task for task in pending if task.command.startswith('Jiangnan')]
            waiting = [task for task in waiting if task.command.startswith('Jiangnan')]
        snapshot = (lang.LANG, tuple(tuple((task.command, task.next_run) for task in tasks)
                                     for tasks in (running, pending, waiting)))
        if snapshot == getattr(self, '_overview_task_snapshot', None):
            return
        def put_task(func: Function):
            with use_scope(f"overview-task_{func.command}"):
                title_key = f"Task.{func.command}.name"
                title = t(title_key)
                if title == title_key:
                    title = func.command
                put_column(
                    [
                        put_text(title).style("--arg-title--"),
                        put_text(
                            t('Gui.Overview.TaskRewardWaiting')
                            if func.command == 'JiangnanTaskReward' and func.next_run.year == 2099
                            else str(func.next_run)
                        ).style("--arg-help--"),
                    ],
                    size="auto auto",
                )
                if func.command in self.alas_config.data:
                    put_button(
                        label=t("Gui.Button.Setting"),
                        onclick=lambda: self.alas_set_group(func.command),
                        color="off",
                    )

        clear("running_tasks")
        clear("pending_tasks")
        clear("waiting_tasks")
        with use_scope("running_tasks"):
            if running:
                for task in running:
                    put_task(task)
            else:
                put_text(t("Gui.Overview.NoTask")).style("--overview-notask-text--")
        with use_scope("pending_tasks"):
            if pending:
                for task in pending:
                    put_task(task)
            else:
                put_text(t("Gui.Overview.NoTask")).style("--overview-notask-text--")
        with use_scope("waiting_tasks"):
            if waiting:
                for task in waiting:
                    put_task(task)
            else:
                put_text(t("Gui.Overview.NoTask")).style("--overview-notask-text--")
        self._overview_task_snapshot = snapshot

    @use_scope("content", clear=True)
    def alas_daemon_overview(self, task: str) -> None:
        if task == 'JiangnanDebug':
            self.jiangnan_debug()
            return
        self.init_menu(name=task)
        self.set_title(t(f"Task.{task}.name"))

        log = RichLog("log")

        if self.is_mobile:
            put_scope(
                "daemon-overview",
                [
                    put_scope("scheduler-bar"),
                    put_scope("groups"),
                    put_scope("log-bar"),
                    put_scope("log", [put_html("")]),
                ],
            )
        else:
            put_scope(
                "daemon-overview",
                [
                    put_none(),
                    put_scope(
                        "_daemon",
                        [
                            put_scope(
                                "_daemon_upper",
                                [put_scope("scheduler-bar"), put_scope("log-bar")],
                            ),
                            put_scope("groups"),
                            put_scope("log", [put_html("")]),
                        ],
                    ),
                    put_none(),
                ],
            )

        log.console.width = log.get_width()

        with use_scope("scheduler-bar"):
            put_text(t("Gui.Overview.Scheduler")).style(
                "font-size: 1.25rem; margin: auto .5rem auto;"
            )
            self.put_scheduler_actions()

        switch_scheduler = BinarySwitchButton(
            label_on=t("Gui.Button.Stop"),
            label_off=t("Gui.Button.Start"),
            onclick_on=lambda: self.alas.stop(),
            onclick_off=lambda: self.alas.start(task),
            get_state=lambda: self.alas.alive,
            color_on="off",
            color_off="on",
            scope="scheduler_btn",
        )

        with use_scope("log-bar"):
            put_text(t("Gui.Overview.Log")).style(
                "font-size: 1.25rem; margin: auto .5rem auto;"
            )
            put_scope(
                "log-bar-btns",
                [
                    put_scope("log_scroll_btn"),
                ],
            )

        switch_log_scroll = BinarySwitchButton(
            label_on=t("Gui.Button.ScrollON"),
            label_off=t("Gui.Button.ScrollOFF"),
            onclick_on=lambda: log.set_scroll(False),
            onclick_off=lambda: log.set_scroll(True),
            get_state=lambda: log.keep_bottom,
            color_on="on",
            color_off="off",
            scope="log_scroll_btn",
        )

        config = self.alas_config.read_file(self.alas_name)
        for group, arg_dict in deep_iter(self.ALAS_ARGS[task], depth=1):
            if group[0] == "Storage":
                continue
            self.set_group(group, arg_dict, config, task)

        run_js(
            """
            $("#pywebio-scope-log").css(
                "grid-row-start",
                -2 - $("#pywebio-scope-_daemon").children().filter(
                    function(){
                        return $(this).css("display") === "none";
                    }
                ).length
            );
            $("#pywebio-scope-log").css(
                "grid-row-end",
                -1
            );
        """
        )

        self.task_handler.add(switch_scheduler.g(), 1, True)
        self.task_handler.add(switch_log_scroll.g(), 1, True)
        self.task_handler.add(log.put_log(self.alas), 0.25, True)

    def put_scheduler_actions(self):
        manager = self.alas
        return put_scope('scheduler-actions', [
            put_button(t('Gui.Button.Logout'),
                       onclick=partial(self.start_jiangnan_logout, manager), color='off'),
            put_scope('scheduler_btn'),
        ]).style('display: flex; align-items: center; gap: .5rem;')

    def start_jiangnan_logout(self, manager):
        if manager.start_logout():
            toast(t('Gui.JiangnanLogout.Started'), color='success')
        else:
            toast(t('Gui.JiangnanLogout.Busy'), color='warn')

    def start_jiangnan_debug(self, task, manager, config_updater):
        if manager.alive:
            toast(t('Gui.JiangnanDebug.Busy'), color='warn')
            return False
        try:
            command = debug_command(task, config_updater.read_file(manager.config_name))
        except ValueError:
            toast(t('Gui.JiangnanDebug.InvalidTask'), color='error')
            return False
        if not manager.start(command):
            toast(t('Gui.JiangnanDebug.Busy'), color='warn')
            return False
        toast(t('Gui.JiangnanDebug.Started', t(f'Task.{task}.name')), color='success')
        return True

    @use_scope('content', clear=True)
    def jiangnan_debug(self):
        self.init_menu(name='JiangnanDebug')
        self.set_title(t('Task.JiangnanDebug.name'))
        manager = self.alas
        config_updater = self.alas_config
        config = config_updater.read_file(manager.config_name)
        tasks = debug_task_names(config)
        ordered = [row[0] for row in task_priority_rows(config, config_updater.SCHEDULER_PRIORITY)]
        tasks = [task for task in ordered if task in tasks]
        selected = getattr(self, '_jiangnan_debug_task', None)
        if selected not in tasks:
            selected = tasks[0] if tasks else None
        put_scope('jiangnan-debug', [
            put_scope('jiangnan-debug-controls'), put_scope('log', [put_html('')]),
        ])
        with use_scope('jiangnan-debug-controls'):
            put_text(t('Task.JiangnanDebug.help')).style('--arg-help--')
            if not tasks:
                put_warning(t('Gui.JiangnanDebug.InvalidTask'))
                return
            put_select('jiangnan_debug_task', label=t('Gui.JiangnanDebug.SelectTask'),
                       options=[{'label': t(f'Task.{task}.name'), 'value': task} for task in tasks],
                       value=selected)
            put_scope('jiangnan-debug-actions')

        def remember_task(value):
            self._jiangnan_debug_task = value

        pin_on_change('jiangnan_debug_task', onchange=remember_task, clear=True)

        def run_selected():
            task = pin['jiangnan_debug_task']
            if self.start_jiangnan_debug(task, manager, config_updater):
                self._jiangnan_debug_task = task
            refresh_actions()

        last_state = [None]

        def refresh_actions():
            state = manager.state
            if state == last_state[0]:
                return
            last_state[0] = state
            with use_scope('jiangnan-debug-actions', clear=True):
                put_buttons([
                    {'label': t('Gui.JiangnanDebug.RunOnce'), 'value': 'run', 'disabled': manager.alive},
                    {'label': t('Gui.JiangnanDebug.Stop'), 'value': 'stop', 'disabled': not manager.alive},
                ], onclick=[run_selected, manager.stop], serial_mode=True)
                key = 'Running' if state == 1 else ('Failed' if state == 3 else 'Idle')
                put_text(t(f'Gui.JiangnanDebug.{key}')).style('--arg-help--')

        refresh_actions()
        log = RichLog('log')
        log.console.width = log.get_width()
        self.task_handler.add(refresh_actions, 1, True)
        self.task_handler.add(log.put_log(manager), 0.25, True)

    @use_scope("menu", clear=True)
    def dev_set_menu(self) -> None:
        self.init_menu(collapse_menu=False, name="Develop")

        put_button(
            label=t("Gui.MenuDevelop.HomePage"),
            onclick=self.show,
            color="menu",
        ).style(f"--menu-HomePage--")

        # put_button(
        #     label=t("Gui.MenuDevelop.Translate"),
        #     onclick=self.dev_translate,
        #     color="menu",
        # ).style(f"--menu-Translate--")

        put_button(
            label=t("Gui.MenuDevelop.Update"),
            onclick=self.dev_update,
            color="menu",
        ).style(f"--menu-Update--")

        put_button(
            label=t("Gui.MenuDevelop.Remote"),
            onclick=self.dev_remote,
            color="menu",
        ).style(f"--menu-Remote--")

        put_button(
            label=t("Gui.MenuDevelop.Utils"),
            onclick=self.dev_utils,
            color="menu",
        ).style(f"--menu-Utils--")

    def dev_translate(self) -> None:
        go_app("translate", new_window=True)
        lang.TRANSLATE_MODE = True
        self.show()

    @use_scope("content", clear=True)
    def dev_update(self) -> None:
        self.init_menu(name="Update")
        self.set_title(t("Gui.MenuDevelop.Update"))

        if State.restart_event is None:
            put_warning(t("Gui.Update.DisabledWarn"))

        put_row(
            content=[put_scope("updater_loading"), None, put_scope("updater_state")],
            size="auto .25rem 1fr",
        )

        put_scope("updater_btn")
        put_scope("updater_info")

        def update_table():
            with use_scope("updater_info", clear=True):
                local_commit = updater.get_commit(short_sha1=True)
                upstream_commit = updater.get_commit(
                    f"origin/{updater.Branch}", short_sha1=True
                )
                put_table(
                    [
                        [t("Gui.Update.Local"), *local_commit],
                        [t("Gui.Update.Upstream"), *upstream_commit],
                    ],
                    header=[
                        "",
                        "SHA1",
                        t("Gui.Update.Author"),
                        t("Gui.Update.Time"),
                        t("Gui.Update.Message"),
                    ],
                )
            with use_scope("updater_detail", clear=True):
                put_text(t("Gui.Update.DetailedHistory"))
                history = updater.get_commit(
                    f"origin/{updater.Branch}", n=20, short_sha1=True
                )
                put_table(
                    [commit for commit in history],
                    header=[
                        "SHA1",
                        t("Gui.Update.Author"),
                        t("Gui.Update.Time"),
                        t("Gui.Update.Message"),
                    ],
                )

        def u(state):
            if state == -1:
                return
            clear("updater_loading")
            clear("updater_state")
            clear("updater_btn")
            if state == 0:
                put_loading("border", "secondary", "updater_loading").style(
                    "--loading-border-fill--"
                )
                put_text(t("Gui.Update.UpToDate"), scope="updater_state")
                put_button(
                    t("Gui.Button.CheckUpdate"),
                    onclick=updater.check_update,
                    color="info",
                    scope="updater_btn",
                )
                update_table()
            elif state == 1:
                put_loading("grow", "success", "updater_loading").style(
                    "--loading-grow--"
                )
                put_text(t("Gui.Update.HaveUpdate"), scope="updater_state")
                put_button(
                    t("Gui.Button.ClickToUpdate"),
                    onclick=updater.run_update,
                    color="success",
                    scope="updater_btn",
                )
                update_table()
            elif state == "checking":
                put_loading("border", "primary", "updater_loading").style(
                    "--loading-border--"
                )
                put_text(t("Gui.Update.UpdateChecking"), scope="updater_state")
            elif state == "failed":
                put_loading("grow", "danger", "updater_loading").style(
                    "--loading-grow--"
                )
                put_text(t("Gui.Update.UpdateFailed"), scope="updater_state")
                put_button(
                    t("Gui.Button.RetryUpdate"),
                    onclick=updater.run_update,
                    color="primary",
                    scope="updater_btn",
                )
            elif state == "start":
                put_loading("border", "primary", "updater_loading").style(
                    "--loading-border--"
                )
                put_text(t("Gui.Update.UpdateStart"), scope="updater_state")
                put_button(
                    t("Gui.Button.CancelUpdate"),
                    onclick=updater.cancel,
                    color="danger",
                    scope="updater_btn",
                )
            elif state == "wait":
                put_loading("border", "primary", "updater_loading").style(
                    "--loading-border--"
                )
                put_text(t("Gui.Update.UpdateWait"), scope="updater_state")
                put_button(
                    t("Gui.Button.CancelUpdate"),
                    onclick=updater.cancel,
                    color="danger",
                    scope="updater_btn",
                )
            elif state == "run update":
                put_loading("border", "primary", "updater_loading").style(
                    "--loading-border--"
                )
                put_text(t("Gui.Update.UpdateRun"), scope="updater_state")
                put_button(
                    t("Gui.Button.CancelUpdate"),
                    onclick=updater.cancel,
                    color="danger",
                    scope="updater_btn",
                    disabled=True,
                )
            elif state == "reload":
                put_loading("grow", "success", "updater_loading").style(
                    "--loading-grow--"
                )
                put_text(t("Gui.Update.UpdateSuccess"), scope="updater_state")
                update_table()
            elif state == "finish":
                put_loading("grow", "success", "updater_loading").style(
                    "--loading-grow--"
                )
                put_text(t("Gui.Update.UpdateFinish"), scope="updater_state")
                update_table()
            elif state == "cancel":
                put_loading("border", "danger", "updater_loading").style(
                    "--loading-border--"
                )
                put_text(t("Gui.Update.UpdateCancel"), scope="updater_state")
                put_button(
                    t("Gui.Button.CancelUpdate"),
                    onclick=updater.cancel,
                    color="danger",
                    scope="updater_btn",
                    disabled=True,
                )
            else:
                put_text(
                    "Something went wrong, please contact develops",
                    scope="updater_state",
                )
                put_text(f"state: {state}", scope="updater_state")

        updater_switch = Switch(
            status=u, get_state=lambda: updater.state, name="updater"
        )

        update_table()
        self.task_handler.add(updater_switch.g(), delay=0.5, pending_delete=True)

        updater.check_update()

    @use_scope("content", clear=True)
    def dev_utils(self) -> None:
        self.init_menu(name="Utils")
        self.set_title(t("Gui.MenuDevelop.Utils"))
        put_button(label="Raise exception", onclick=raise_exception)

        def _force_restart():
            if State.restart_event is not None:
                toast("Alas will restart in 3 seconds", duration=0, color="error")
                clearup()
                State.restart_event.set()
            else:
                toast("Reload not enabled", color="error")

        put_button(label="Force restart", onclick=_force_restart)

    @use_scope("content", clear=True)
    def dev_remote(self) -> None:
        self.init_menu(name="Remote")
        self.set_title(t("Gui.MenuDevelop.Remote"))
        put_row(
            content=[put_scope("remote_loading"), None, put_scope("remote_state")],
            size="auto .25rem 1fr",
        )
        put_scope("remote_info")

        def u(state):
            if state == -1:
                return
            clear("remote_loading")
            clear("remote_state")
            clear("remote_info")
            if state in (1, 2):
                put_loading("grow", "success", "remote_loading").style(
                    "--loading-grow--"
                )
                put_text(t("Gui.Remote.Running"), scope="remote_state")
                put_text(t("Gui.Remote.EntryPoint"), scope="remote_info")
                entrypoint = RemoteAccess.get_entry_point()
                if entrypoint:
                    if State.electron:  # Prevent click into url in electron client
                        put_text(entrypoint, scope="remote_info").style(
                            "text-decoration-line: underline"
                        )
                    else:
                        put_link(name=entrypoint, url=entrypoint, scope="remote_info")
                else:
                    put_text("Loading...", scope="remote_info")
            elif state in (0, 3):
                put_loading("border", "secondary", "remote_loading").style(
                    "--loading-border-fill--"
                )
                if (
                        State.deploy_config.EnableRemoteAccess
                        and State.deploy_config.Password
                ):
                    put_text(t("Gui.Remote.NotRunning"), scope="remote_state")
                else:
                    put_text(t("Gui.Remote.NotEnable"), scope="remote_state")
                put_text(t("Gui.Remote.ConfigureHint"), scope="remote_info")
                url = "http://app.azurlane.cloud" + (
                    "" if State.deploy_config.Language.startswith("zh") else "/en.html"
                )
                put_html(
                    f'<a href="{url}" target="_blank">{url}</a>', scope="remote_info"
                )
                if state == 3:
                    put_warning(
                        t("Gui.Remote.SSHNotInstall"),
                        closable=False,
                        scope="remote_info",
                    )

        remote_switch = Switch(
            status=u, get_state=RemoteAccess.get_state, name="remote"
        )

        self.task_handler.add(remote_switch.g(), delay=1, pending_delete=True)

    def ui_develop(self) -> None:
        if not self.is_mobile:
            self.show()
            return
        self.init_aside(name="Home")
        self.set_title(t("Gui.Aside.Home"))
        self.dev_set_menu()
        self.alas_name = ""
        if hasattr(self, "alas"):
            del self.alas
        self.state_switch.switch()

    def ui_alas(self, config_name: str) -> None:
        if config_name == self.alas_name:
            self.expand_menu()
            return
        self.init_aside(name=config_name)
        clear("content")
        self.alas_name = config_name
        self.alas_mod = get_config_mod(config_name)
        self.alas = ProcessManager.get_manager(config_name)
        self.alas_config = load_config(config_name)
        self.state_switch.switch()
        self.initial()
        self.alas_set_menu()

    def ui_add_alas(self) -> None:
        with popup(t("Gui.AddAlas.PopupTitle")) as s:

            def get_unused_name():
                all_name = alas_instance()
                for i in range(2, 100):
                    if f"alas{i}" not in all_name:
                        return f"alas{i}"
                else:
                    return ""

            def add():
                name = pin["AddAlas_name"]
                origin = pin["AddAlas_copyfrom"]

                if name in alas_instance():
                    err = "Gui.AddAlas.FileExist"
                elif set(name) & set(".\\/:*?\"'<>|"):
                    err = "Gui.AddAlas.InvalidChar"
                elif name.lower().startswith("template"):
                    err = "Gui.AddAlas.InvalidPrefixTemplate"
                else:
                    err = ""
                if err:
                    clear(s)
                    put(name, origin)
                    put_error(t(err), scope=s)
                    return

                r = load_config(origin).read_file(origin)
                State.config_updater.write_file(name, r, get_config_mod(origin))
                self.set_aside()
                self.active_button("aside", self.alas_name)
                close_popup()

            def put(name=None, origin=None):
                put_input(
                    name="AddAlas_name",
                    label=t("Gui.AddAlas.NewName"),
                    value=name or get_unused_name(),
                    scope=s,
                )
                put_select(
                    name="AddAlas_copyfrom",
                    label=t("Gui.AddAlas.CopyFrom"),
                    options=alas_template() + alas_instance(),
                    value=origin or "template-alas",
                    scope=s,
                )
                put_buttons(
                    buttons=[
                        {"label": t("Gui.AddAlas.Confirm"), "value": "confirm"},
                        {"label": t("Gui.AddAlas.Manage"), "value": "manage"},
                    ],
                    onclick=[
                        add,
                        lambda: go_app("manage", new_window=False),
                    ],
                    scope=s,
                )

            put()

    def show(self) -> None:
        self._show()
        self.load_home = True
        self.set_aside()
        self.init_aside(name="Home")
        self.dev_set_menu()
        self.init_menu(name="HomePage")
        self.alas_name = ""
        if hasattr(self, "alas"):
            del self.alas
        self.set_status(0)

        def set_language(l):
            lang.set_language(l)
            self.show()

        def set_theme(t):
            self.set_theme(t)
            run_js("location.reload()")

        with use_scope("content"):
            put_text("Select your language / 选择语言").style("text-align: center")
            put_buttons(
                [
                    {"label": "简体中文", "value": "zh-CN"},
                    {"label": "繁體中文", "value": "zh-TW"},
                    {"label": "English", "value": "en-US"},
                    {"label": "日本語", "value": "ja-JP"},
                ],
                onclick=lambda l: set_language(l),
            ).style("text-align: center")
            put_text("Change theme / 更改主题").style("text-align: center")
            put_buttons(
                [
                    {"label": "Light", "value": "default", "color": "light"},
                    {"label": "Dark", "value": "dark", "color": "dark"},
                ],
                onclick=lambda t: set_theme(t),
            ).style("text-align: center")

            # show something
            put_markdown(
                """
            Alas is a free open source software, if you paid for Alas from any channel, please refund.
            Alas 是一款免费开源软件，如果你在任何渠道付费购买了Alas，请退款。
            Project repository 项目地址：`https://github.com/LmeSzinc/AzurLaneAutoScript`
            """
            ).style("text-align: center")

        if lang.TRANSLATE_MODE:
            lang.reload()

            def _disable():
                lang.TRANSLATE_MODE = False
                self.show()

            toast(
                _t("Gui.Toast.DisableTranslateMode"),
                duration=0,
                position="right",
                onclick=_disable,
            )

    def run(self) -> None:
        # setup gui
        set_env(title="Alas", output_animation=False)
        add_css(filepath_css("alas"))
        if self.is_mobile:
            add_css(filepath_css("alas-mobile"))
        else:
            add_css(filepath_css("alas-pc"))

        if self.theme == "dark":
            add_css(filepath_css("dark-alas"))
        else:
            add_css(filepath_css("light-alas"))

        # Auto refresh when lost connection
        # [For develop] Disable by run `reload=0` in console
        run_js(
            """
        reload = 1;
        WebIO._state.CurrentSession.on_session_close(
            ()=>{
                setTimeout(
                    ()=>{
                        if (reload == 1){
                            location.reload();
                        }
                    }, 4000
                )
            }
        );
        """
        )

        aside = get_localstorage("aside")
        self.show()

        # init config watcher
        self._init_alas_config_watcher()

        # save config
        _thread_save_config = threading.Thread(target=self._alas_thread_update_config)
        register_thread(_thread_save_config)
        _thread_save_config.start()

        visibility_state_switch = Switch(
            status={
                True: [
                    lambda: self.__setattr__("visible", True),
                    lambda: self.alas_update_overview_task()
                    if self.page == "Overview"
                    else 0,
                    lambda: self.task_handler._task.__setattr__("delay", 15),
                ],
                False: [
                    lambda: self.__setattr__("visible", False),
                    lambda: self.task_handler._task.__setattr__("delay", 1),
                ],
            },
            get_state=get_window_visibility_state,
            name="visibility_state",
        )

        def goto_update():
            self.ui_develop()
            self.dev_update()

        update_switch = Switch(
            status={
                1: lambda: toast(
                    t("Gui.Toast.ClickToUpdate"),
                    duration=0,
                    position="right",
                    color="success",
                    onclick=goto_update,
                )
            },
            get_state=lambda: updater.state,
            name="update_state",
        )

        self.task_handler.add(self.state_switch.g(), 2)
        self.task_handler.add(self.set_aside_status, 2)
        self.task_handler.add(self.refresh_canal_point, 3)
        self.task_handler.add(self.refresh_corn_countdowns, 1)
        self.task_handler.add(visibility_state_switch.g(), 15)
        if State.deploy_config.AutoUpdate:
            self.task_handler.add(update_switch.g(), 1)
        self.task_handler.start()

        # Return to previous page
        if aside not in ["Home", None]:
            self.ui_alas(aside)


def app_manage():
    def _import():
        resp = file_upload(
            label=t("Gui.AppManage.Import"),
            placeholder=t("Gui.Text.ChooseFile"),
            help_text=t("Gui.AppManage.OverrideWarning"),
            accept=".json",
            required=False,
            max_size="1M",
        )

        if resp is None:
            return

        file: bytes = resp["content"]
        file_name: str = resp["filename"]

        if IS_ON_PHONE_CLOUD:
            config_name = mod_name = "alas"
        elif len(file_name.split(".")) == 2:
            config_name, _ = file_name.split(".")
            mod_name = "alas"
        else:
            config_name, mod_name, _ = file_name.rsplit(".", maxsplit=2)

        config = json.loads(file.decode(encoding="utf-8"))
        State.config_updater.write_file(config_name, config, mod_name)
        toast(t("Gui.AppManage.ImportSuccess"), color="success")

        _show_table()

    def _export(config_name: str):
        mod_name = get_config_mod(config_name)
        if mod_name == "alas":
            filename = f"{config_name}.json"
        else:
            filename = f"{config_name}.{mod_name}.json"
        with open(filepath_config(config_name, mod_name), "rb") as f:
            download(filename, f.read())

    def _new():
        def get_unused_name():
            all_name = alas_instance()
            for i in range(2, 100):
                if f"alas{i}" not in all_name:
                    return f"alas{i}"
            else:
                return ""

        def validate(s: str):
            if s in alas_instance():
                return t("Gui.AppManage.NameExist")
            if set(s) & set(".\\/:*?\"'<>|"):
                return t("Gui.AppManage.InvalidChar")
            if s.lower().startswith("template"):
                return t("Gui.AppManage.InvalidPrefixTemplate")
            return None

        resp = input_group(
            label=t("Gui.AppManage.TitleNew"),
            inputs=[
                input(
                    label=t("Gui.AppManage.NewName"),
                    name="config_name",
                    value=get_unused_name(),
                    validate=validate,
                ),
                select(
                    label=t("Gui.AppManage.CopyFrom"),
                    name="copy_from",
                    options=alas_template() + alas_instance(),
                    value="template-alas",
                ),
            ],
            cancelable=True,
        )

        if resp is None:
            return

        config_name = resp["config_name"]
        origin = resp["copy_from"]

        r = load_config(origin).read_file(origin)
        State.config_updater.write_file(config_name, r, get_config_mod(origin))
        toast(t("Gui.AppManage.NewSuccess"), color="success")
        _show_table()

    def _show_table():
        clear("config_table")
        put_table(
            tdata=[
                (
                    name,
                    get_config_mod(name),
                    put_buttons(
                        buttons=[
                            {"label": t("Gui.AppManage.Export"), "value": name},
                            # {
                            #     "label": t("Gui.AppManage.Delete"),
                            #     "value": name,
                            #     "disabled": True,
                            #     "color": "danger",
                            # },
                        ],
                        onclick=[
                            partial(_export, name),
                            # partial(_delete, name),
                        ],
                        group=True,
                        small=True,
                    ),
                )
                for name in alas_instance()
            ],
            header=[
                t("Gui.AppManage.Name"),
                t("Gui.AppManage.Mod"),
                t("Gui.AppManage.Actions"),
            ],
            scope="config_table",
        )

    set_env(title="Alas", output_animation=False)
    run_js("$('head').append('<style>.footer{display:none}</style>')")

    put_html(f"<h2>{t('Gui.AppManage.PageTitle')}</h2>")
    put_scope("config_table")
    put_buttons(
        buttons=[
            {
                "label": t("Gui.AppManage.New"),
                "value": "new",
                "disabled": IS_ON_PHONE_CLOUD,
            },
            {"label": t("Gui.AppManage.Import"), "value": "import"},
            {"label": t("Gui.AppManage.Back"), "value": "back"},
        ],
        onclick=[
            (lambda: None) if IS_ON_PHONE_CLOUD else _new,
            _import,
            partial(go_app, "index", new_window=False),
        ],
    )
    _show_table()


def debug():
    """For interactive python.
    $ python3
    >>> from module.webui.app import *
    >>> debug()
    >>>
    """
    startup()
    AlasGUI().run()


def startup():
    State.init()
    lang.reload()
    updater.event = State.manager.Event()
    if State.deploy_config.AutoUpdate and updater.delay > 0:
        task_handler.add(updater.check_update, updater.delay)
    if State.deploy_config.AutoUpdate:
        task_handler.add(updater.schedule_update(), 86400)
    task_handler.start()
    if (
            State.deploy_config.EnableRemoteAccess
            and State.deploy_config.Password is not None
    ):
        task_handler.add(RemoteAccess.keep_ssh_alive(), 60)


def clearup():
    """
    Notice: Ensure run it before uvicorn reload app,
    all process will NOT EXIT after close electron app.
    """
    logger.info("Start clearup")
    RemoteAccess.kill_ssh_process()
    for alas in ProcessManager._processes.values():
        alas.stop()
    State.clearup()
    task_handler.stop()
    logger.info("Alas closed.")


def app():
    parser = argparse.ArgumentParser(description="Alas web service")
    parser.add_argument(
        "-k", "--key", type=str, help="Password of alas. No password by default"
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Use jsdelivr cdn for pywebio static files (css, js). Self host cdn by default.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        type=str,
        help="Run alas by config names on startup",
    )
    args, _ = parser.parse_known_args()

    # Apply config
    AlasGUI.set_theme(theme=State.deploy_config.Theme)
    lang.LANG = State.deploy_config.Language
    key = args.key or State.deploy_config.Password
    cdn = args.cdn if args.cdn else State.deploy_config.CDN
    runs = None
    if args.run:
        runs = args.run
    elif State.deploy_config.Run:
        # TODO: refactor poor_yaml_read() to support list
        tmp = State.deploy_config.Run.split(",")
        runs = [l.strip(" ['\"]") for l in tmp if len(l)]
    instances: List[str] = runs

    logger.hr("Webui configs")
    logger.attr("Theme", State.deploy_config.Theme)
    logger.attr("Language", lang.LANG)
    logger.attr("Password", True if key else False)
    logger.attr("CDN", cdn)
    logger.attr("IS_ON_PHONE_CLOUD", IS_ON_PHONE_CLOUD)

    from deploy.atomic import atomic_failure_cleanup
    atomic_failure_cleanup('./config')

    def index():
        if key is not None and not login(key):
            logger.warning(f"{info.user_ip} login failed.")
            time.sleep(1.5)
            run_js("location.reload();")
            return
        gui = AlasGUI()
        local.gui = gui
        gui.run()

    def manage():
        if key is not None and not login(key):
            logger.warning(f"{info.user_ip} login failed.")
            time.sleep(1.5)
            run_js("location.reload();")
            return
        app_manage()

    app = asgi_app(
        applications=[index, manage],
        cdn=cdn,
        static_dir=None,
        debug=True,
        on_startup=[
            startup,
            lambda: ProcessManager.restart_processes(
                instances=instances, ev=updater.event
            ),
        ],
        on_shutdown=[clearup],
    )

    return app
