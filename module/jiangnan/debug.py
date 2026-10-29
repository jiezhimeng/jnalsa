"""Jiangnan UI visibility and validated one-shot task dispatch."""


def visible_menu(menu):
    """Show only Jiangnan settings and the shared emulator configuration."""
    result = {}
    for name, category in menu.items():
        if name == 'Alas':
            tasks = [task for task in category.get('tasks', []) if task == 'Alas']
        elif name == 'Tool':
            tasks = ['JiangnanDebug']
        elif name.startswith('Jiangnan'):
            tasks = [task for task in category.get('tasks', []) if task.startswith('Jiangnan')]
        else:
            continue
        if tasks:
            result[name] = dict(category, tasks=tasks)
    return result


def debug_task_names(definitions):
    return [name for name, settings in definitions.items()
            if name.startswith('Jiangnan') and isinstance(settings, dict)
            and isinstance(settings.get('Scheduler'), dict)]


def debug_command(task, definitions):
    if task not in debug_task_names(definitions):
        raise ValueError(f'Not a scheduled Jiangnan task: {task}')
    return f'JiangnanDebug:{task}'


def run_debug_task(config_name, task):
    """Run exactly one selected task, without entering the scheduler loop."""
    import inflection

    from alas import AzurLaneAutoScript
    from module.config.config import name_to_function
    from module.logger import logger

    script = AzurLaneAutoScript(config_name=config_name)
    config = script.config
    debug_command(task, config.data)
    method = inflection.underscore(task)
    if not callable(getattr(script, method, None)):
        raise ValueError(f'No Jiangnan task handler: {task}')
    config.bind(task)
    config.task = name_to_function(task)
    # This process runs only the selected task, so yielding cannot dispatch
    # another scheduler task. Keep the marker in memory, outside saved config.
    config._jiangnan_debug_task = task
    logger.hr(f'JIANGNAN DEBUG: {task}', level=0)
    return script.run(method, skip_first_screenshot=True)
