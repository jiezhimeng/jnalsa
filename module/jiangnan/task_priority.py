"""Shared task ordering for the Jiangnan scheduler and settings page."""


def priority_defaults(tasks, default_order):
    """Keep the existing order when upgrading a config without priorities."""
    tasks = dict.fromkeys(tasks)
    ordered = [name.strip() for name in default_order.split('>')
               if name.strip() in tasks]
    ordered = list(dict.fromkeys(ordered))
    ordered.extend(name for name in tasks if name not in ordered)
    return {name: index for index, name in enumerate(ordered, 1)}


def task_priority_rows(data, default_order):
    """Return (command, priority, enabled) in effective scheduling order."""
    tasks = {
        name: settings['Scheduler']
        for name, settings in data.items()
        if name.startswith('Jiangnan') and isinstance(settings, dict)
        and isinstance(settings.get('Scheduler'), dict)
    }
    defaults = priority_defaults(tasks, default_order)
    configured = data.get('JiangnanTaskPriority', {}).get('TaskPriority', {})
    rows = []
    for name, scheduler in tasks.items():
        priority = configured.get(name, defaults[name])
        if type(priority) is not int or priority < 1:
            priority = defaults[name]
        rows.append((name, priority, bool(scheduler.get('Enable', False))))
    return sorted(rows, key=lambda row: (row[1], defaults[row[0]]))


def scheduler_priority(data, default_order):
    """Replace the Jiangnan block, preserving the other scheduler priorities."""
    rows = task_priority_rows(data, default_order)
    if not rows:
        return default_order
    commands = [row[0] for row in rows]
    result = []
    inserted = False
    for name in (part.strip() for part in default_order.split('>')):
        if name.startswith('Jiangnan'):
            if not inserted:
                result.extend(commands)
                inserted = True
        elif name:
            result.append(name)
    if not inserted:
        result.extend(commands)
    return ' > '.join(result)


def priority_updates(order, data, default_order):
    """Validate a complete drag result before assigning consecutive priorities."""
    expected = {row[0] for row in task_priority_rows(data, default_order)}
    if (not isinstance(order, list) or len(order) != len(expected)
            or any(not isinstance(name, str) for name in order)
            or set(order) != expected):
        raise ValueError('Task order must contain every scheduled Jiangnan task exactly once')
    return {
        f'JiangnanTaskPriority.TaskPriority.{name}': index
        for index, name in enumerate(order, 1)
    }
