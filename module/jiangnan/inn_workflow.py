"""Independent inn feature clocks sharing one confirmed visit to the inn."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from module.jiangnan.inn_schedule import (
    GOODS_GROUP, current_goods_slot, goods_next_run, inn_now, next_goods_slot, slot_key,
)


@dataclass(frozen=True)
class InnFeatureResult:
    # pending means scaffold only, done means a verified gameplay outcome.
    state: str
    status: str
    next_run: Optional[datetime] = None
    keep_open: bool = False

    def __post_init__(self):
        if self.state not in ('pending', 'done', 'retry'):
            raise ValueError('Invalid inn feature result: %s' % self.state)
        if self.next_run is not None and not isinstance(self.next_run, datetime):
            raise TypeError('Inn feature next_run must be a datetime')


@dataclass(frozen=True)
class InnFeature:
    group: str
    label: str
    handler: str
    default_interval: Optional[int]


FEATURES = (
    InnFeature('yingtian_inn_cooking', '烹饪', 'run_cooking_sales', 5),
    InnFeature(GOODS_GROUP, '定时收获', 'run_goods_collection', None),
    # 独立开关默认关闭；不能沿用旧“做饭卖菜”的 Enable 启用物资交易。
    InnFeature('yingtian_inn_diners', '食客', 'run_diners', 12),
)


def parse_next_run(value):
    if not isinstance(value, datetime):
        try:
            value = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return datetime(2020, 1, 1)
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value


class InnWorkflow:
    RETRY = timedelta(minutes=10)

    def __init__(self, inn):
        self.inn = inn
        self.config = inn.config

    def _get(self, feature, field, default=None):
        return self.config.cross_get(
            '%s.%s.%s' % (self.inn.TASK, feature.group, field), default=default)

    def _set(self, feature, field, value):
        self.config.modified['%s.%s.%s' % (self.inn.TASK, feature.group, field)] = value

    def _interval(self, feature):
        if feature.group == 'yingtian_inn_diners':
            return timedelta(minutes=12)  # 食客固定每 12 分钟一轮。
        value = self._get(feature, 'CheckInterval', feature.default_interval)
        try:
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError('Expected whole minutes')
            value = int(value)
            if not 1 <= value <= 1440:
                raise ValueError('Expected 1..1440 minutes')
        except (ValueError, TypeError, OverflowError):
            value = feature.default_interval
        return timedelta(minutes=value)

    def _record(self, feature, result, now):
        if not isinstance(result, InnFeatureResult):
            raise TypeError('Inn handlers must return an explicit InnFeatureResult')
        if result.next_run is not None:
            next_run = parse_next_run(result.next_run)
        elif result.state == 'retry':
            interval = self._interval(feature) if feature.group == 'yingtian_inn_diners' else self.RETRY
            next_run = now + interval
        elif feature.group == GOODS_GROUP:
            next_run = next_goods_slot(now)
        else:
            next_run = now + self._interval(feature)
        next_run = max(next_run, now + timedelta(seconds=30))
        self._set(feature, 'LastCheck', now)
        self._set(feature, 'NextRun', next_run)
        self._set(feature, 'Status', result.status)
        if feature.group == GOODS_GROUP and result.state == 'retry':
            self._set(feature, 'RetrySlot', slot_key(current_goods_slot(now)))
        self.config.update()
        return next_run

    def _finish(self, status, success, next_run=None):
        self.config.modified['%s.yingtian_inn.Status' % self.inn.TASK] = status
        self.config.update()
        if next_run is not None:
            self.config.task_delay(target=next_run)
        else:
            self.config.task_delay(success=success)
        return success

    def run(self, now=None):
        now = now or inn_now()
        enabled = [feature for feature in FEATURES if self._get(feature, 'Enable', False)]
        if not enabled:
            return self._finish('[烹饪、食客、定时收获均未启用]', True)
        next_runs = {}
        for feature in enabled:
            saved = self._get(feature, 'NextRun')
            if feature.group == GOODS_GROUP:
                next_runs[feature.group] = goods_next_run(
                    now, self._get(feature, 'LastSlot', ''),
                    self._get(feature, 'AttemptedSlot', ''),
                    self._get(feature, 'RetrySlot', ''), saved)
                self._set(feature, 'NextRun', next_runs[feature.group])
            else:
                next_runs[feature.group] = parse_next_run(saved)
        due = [feature for feature in enabled if next_runs[feature.group] <= now]
        if not due:
            return self._finish('[等待下次客栈检查]', True, min(next_runs.values()))
        if not self.inn.enter_inn():
            for feature in due:
                retry = self._interval(feature) if feature.group == 'yingtian_inn_diners' else self.RETRY
                next_runs[feature.group] = now+retry
                if feature.group == GOODS_GROUP:
                    next_runs[feature.group] = min(next_runs[feature.group], next_goods_slot(now))
                    self._set(feature, 'RetrySlot', slot_key(current_goods_slot(now)))
                self._set(feature, 'NextRun', next_runs[feature.group])
                self._set(feature, 'Status', '[进入客栈失败][等待复查]')
            return self._finish('[进入客栈失败]', False, min(next_runs.values()))

        results = []
        keep_open = False
        try:
            self.inn.update_map_location()
            for feature in due:
                try:
                    result = getattr(self.inn, feature.handler)(now)
                    if isinstance(result, InnFeatureResult):
                        keep_open = result.keep_open
                    next_runs[feature.group] = self._record(feature, result, now)
                except Exception:
                    self._record(feature, InnFeatureResult('retry', '[执行异常][等待复查]'), now)
                    raise
                results.append(result)
                if keep_open:
                    break
        finally:
            # 确认收下并返回客栈后正常离开；食客/收获结果不明时保留现场。
            left = True if keep_open else self.inn.leave_inn()

        if not left:
            return self._finish('[退出客栈未确认][等待复查]', False,
                                min(min(next_runs.values()), now + self.RETRY))
        success = all(result.state == 'done' for result in results)
        status = (results[-1].status if keep_open else
                  '[客栈检查完成][已返回城市]' if success
                  else '[已返回城市][功能待接入或等待复查]')
        return self._finish(status, success, min(next_runs.values()))
