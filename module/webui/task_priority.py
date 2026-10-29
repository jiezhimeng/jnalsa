"""Draggable task order widget using the existing PyWebIO callback channel."""
from html import escape
from pathlib import Path
from uuid import uuid4

from pywebio.io_ctrl import output_register_callback
from pywebio.output import put_html
from pywebio.session import run_js

from module.logger import logger
from module.webui.lang import t


DRAG_SCRIPT = (Path(__file__).resolve().parents[2]
               / 'assets/gui/js/task-priority.js').read_text(encoding='utf-8')


def put_task_priority_list(rows, onchange):
    container_id = 'task-priority-' + uuid4().hex
    saved_order = [row[0] for row in rows]

    def save(order):
        nonlocal saved_order
        try:
            success, saved_order = onchange(order)
        except Exception as exc:
            logger.exception(exc)
            success = False
        run_js('''
            const root = document.getElementById(container_id);
            if (root && root.finishSave) root.finishSave(success, saved_order);
        ''', container_id=container_id, success=success, saved_order=saved_order)

    callback_id = output_register_callback(save, serial_mode=True)
    items = []
    for index, (command, _, enabled) in enumerate(rows, 1):
        name = escape(t(f'Task.{command}.name'))
        status = escape(t('Gui.TaskPriority.Enabled' if enabled else 'Gui.TaskPriority.Disabled'))
        handle = escape(t('Gui.TaskPriority.DragHandle'))
        items.append(
            f'<li class="task-priority-item" data-task="{escape(command, quote=True)}">'
            f'<span class="task-priority-rank" aria-hidden="true">{index}</span>'
            f'<button type="button" class="task-priority-handle" draggable="true" '
            f'aria-label="{handle}: {name}" title="{handle}">⠿</button>'
            f'<span class="task-priority-name">{name}</span>'
            f'<span class="task-priority-status">{status}</span></li>'
        )
    put_html(
        f'<div id="{container_id}" class="task-priority-editor">'
        f'<p class="task-priority-hint">{escape(t("Gui.TaskPriority.DragHint"))}</p>'
        '<p class="task-priority-feedback" role="status" aria-live="polite"></p>'
        f'<ol class="task-priority-list">{"".join(items)}</ol></div>'
    )
    run_js(DRAG_SCRIPT, container_id=container_id, callback_id=callback_id, labels={
        'saving': t('Gui.TaskPriority.Saving'),
        'saved': t('Gui.TaskPriority.Saved'),
        'failed': t('Gui.TaskPriority.SaveFailed'),
    })
