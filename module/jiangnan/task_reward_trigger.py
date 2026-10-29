"""Queue reward collection at task boundaries, driven only by screenshot events."""
import time
from datetime import datetime

from module.jiangnan.task_reward_vision import reward_prompt, task_entry, prompt_kind
from module.logger import logger


TASK = 'JiangnanTaskReward'
# An inactive event has no due time. This hidden sentinel parks it in the
# existing scheduler until a screenshot explicitly calls the task.
WAITING_FOR_PROMPT = datetime(2099, 1, 1)


def enabled(config):
    return bool(config.data.get(TASK, {}).get('Scheduler', {}).get('Enable', False))


class TaskRewardWatch:
    def __init__(self):
        self.pending = False
        self.latched = False
        self.next_probe = 0.
        self.retry_after = 0.
        self.prompt_kind = None

    def observe(self, image):
        now = time.monotonic()
        if now < self.next_probe or now < self.retry_after:
            return
        self.next_probe = now + 1.
        prompt = reward_prompt(image)
        if prompt is not None:
            kind = prompt_kind(image)
            if not self.latched or kind != self.prompt_kind:
                self.pending = True
                self.latched = True
                self.prompt_kind = kind
                logger.info('Daily task reward portrait detected; queue after current task')
        elif task_entry(image) is not None:
            # A modal/other page is not evidence that the prompt disappeared.
            self.latched = False
            self.prompt_kind = None

    def finish(self, success):
        self.pending = False
        self.latched = bool(success)
        if not success:
            self.retry_after = time.monotonic() + 60.


def watcher(device, config):
    key = getattr(config, 'config_name', None)
    if getattr(device, '_task_reward_profile', None) != key or not isinstance(
            getattr(device, '_task_reward_watch', None), TaskRewardWatch):
        device._task_reward_profile = key
        device._task_reward_watch = TaskRewardWatch()
    return device._task_reward_watch


def flush_request(device, config):
    watch = watcher(device, config)
    if not enabled(config):
        watch.pending = watch.latched = False
        watch.prompt_kind = None
        return False
    if not watch.pending:
        return False
    scheduled = config.data[TASK]['Scheduler'].get('NextRun')
    if isinstance(scheduled, datetime) and scheduled <= datetime.now():
        watch.pending = False
        return False  # Already queued, including an interrupted collection.
    called = config.task_call(TASK, force_call=False)
    if called:
        watch.pending = False
    return called


def poll_task_rewards(device, config):
    """Observe while idle with the game left open; never opens the game or clicks."""
    if not enabled(config):
        return False
    scheduled = config.data[TASK]['Scheduler'].get('NextRun')
    if isinstance(scheduled, datetime) and scheduled <= datetime.now():
        return True  # A goto_main run may have queued it before idle watching.
    watch = watcher(device, config)
    watch.observe(device.screenshot())
    return flush_request(device, config)


def park_task(device, config, success):
    config.modified[f'{TASK}.Scheduler.NextRun'] = WAITING_FOR_PROMPT
    config.update()
    watcher(device, config).finish(success)
