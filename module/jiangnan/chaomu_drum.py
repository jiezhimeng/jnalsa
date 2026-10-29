"""Search each due city and strike its drum once, with a persisted quota."""
import time
from datetime import datetime, timedelta

import cv2

from module.base.button import Button
from module.base.utils import crop
from module.jiangnan.allocation import CITY_KEY_TO_NAME
from module.jiangnan.building import JiangnanBuilding
from module.jiangnan.chaomu_drum_state import (
    DRUM_DAILY_LIMIT, DRUM_INTERVAL, drum_next_run, drum_used_count,
    parse_drum_count, DRUM_COIN_COST_TO_USED,
)
from module.jiangnan.chaomu_drum_cost import (
    CURRENCY_AREA, COIN_AMOUNT_AREA, drum_cost_currency,
)
from module.jiangnan.foundation_icons import find_foundation_icon
from module.jiangnan.ocr import read_confirmed_digits
from module.jiangnan.ppocr_v6 import PPOCR_V6_CASCADE
from module.logger import logger


PANEL = Button(
    area=(218, 82, 294, 294), color=(), button=(218, 82, 294, 294),
    file='./assets/jiangnan/chaomu_drum/panel.png', name='CHAOMU_DRUM_PANEL',
)
STRIKE = Button(
    area=(982, 620, 1040, 657), color=(), button=(985, 622, 1034, 651),
    file='./assets/jiangnan/chaomu_drum/panel.png', name='CHAOMU_DRUM_STRIKE',
)
SKIP = Button(
    area=(271, 492, 369, 525), color=(), button=(275, 494, 365, 523),
    file='./assets/jiangnan/chaomu_drum/animation.png', name='CHAOMU_DRUM_SKIP',
)


class JiangnanChaomuDrum(JiangnanBuilding):
    TASK = 'JiangnanChaomuDrum'
    RETRY = timedelta(minutes=10)
    # The sentence is right-aligned and grows leftward for counts 10 and above.
    COUNT_AREA = (900, 577, 1093, 609)
    WORK_COUNT_AREA = (1048, 615, 1072, 640)

    def _record(self, city_key):
        group = f'{self.TASK}.chaomu_drum_{city_key}'
        data = dict(self.config.cross_get(group, default={}))
        for key, value in self.config.modified.items():
            if key.startswith(group + '.'):
                data[key[len(group) + 1:]] = value
        return data

    def _save_city(self, city_key, **values):
        for field, value in values.items():
            self.config.modified[
                f'{self.TASK}.chaomu_drum_{city_key}.{field}'
            ] = value
        # The quota reservation MUST reach disk before the strike click.
        self.config.update()

    def _panel_ready(self):
        return (
            self.appear(PANEL, offset=(12, 12), similarity=0.88)
            and self.appear(STRIKE, offset=(12, 12), similarity=0.85)
        )

    def current_navigation_page(self):
        # Both overlays leave city navigation visible behind them.
        if self.appear(SKIP, offset=(16, 16), similarity=0.85):
            return 'chaomu_drum_result'
        if self.appear(PANEL, offset=(12, 12), similarity=0.88):
            return 'chaomu_drum'
        return super().current_navigation_page()

    def ui_goto_main(self, *args, **kwargs):
        self.device.screenshot()
        if self.appear(SKIP, offset=(16, 16), similarity=0.85):
            return self._finish_strike()
        return super().ui_goto_main(*args, **kwargs)

    def _open_panel(self):
        self.device.screenshot()
        if self._panel_ready():
            return True
        arrow = self.find_building_search_arrow()
        match = find_foundation_icon(self.device.image, arrow, '朝暮鼓')
        if match is None:
            logger.warning('No drum foundation icon near the confirmed search arrow')
            return False
        x, y, score, _ = match
        logger.attr('ChaomuDrumMapIcon', (x, y, score))
        self._click_searched_building(Button(
            area=(), color=(), button=(x - 7, y - 7, x + 7, y + 7),
            name='CHAOMU_DRUM_MAP_ICON',
        ))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self._panel_ready():
                return True
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                return False
        return False

    def _read_count(self):
        """Two frames, each confirmed across independent OCR models/crops."""
        counts = []
        for _ in range(2):
            self.device.screenshot()
            if not self._panel_ready():
                return None
            count = self._count_from_frame()
            if count is None:
                return None
            counts.append(count)
            self.device.sleep(0.2)
        return counts[0] if counts[0] == counts[1] else None

    def _cost_currency(self):
        return drum_cost_currency(
            self.device.image, self.anchored_area(PANEL, CURRENCY_AREA),
        )

    def _count_from_frame(self):
        currency = self._cost_currency()
        if currency == 'gem':
            # Any gem fee is forbidden, even if its amount cannot be read.
            logger.warning('Drum gem cost detected: daily limit reached; clicking is forbidden')
            return DRUM_DAILY_LIMIT
        if currency == 'coin':
            amount = read_confirmed_digits(
                self.device.image, self.anchored_area(PANEL, COIN_AMOUNT_AREA),
            )
            used = DRUM_COIN_COST_TO_USED.get(amount)
            logger.attr('ChaomuDrumCostCount', (currency, amount, used))
            return used
        original = crop(
            self.device.image, self.anchored_area(PANEL, self.COUNT_AREA),
        )
        enlarged = cv2.resize(original, None, fx=3, fy=3)
        values = PPOCR_V6_CASCADE.atomic_ocr_for_single_lines_consensus([original, enlarged])
        parsed = [parse_drum_count(value) for value in values]
        if len(parsed) == 2 and parsed[0] is not None and parsed[0] == parsed[1]:
            return parsed[0]
        logger.warning(f'Drum daily count uncertain: {values}')
        return None

    def _stop_at_limit(self, city_key, used=DRUM_DAILY_LIMIT):
        now = datetime.now().replace(microsecond=0)
        record = self._record(city_key)
        used = max(used, drum_used_count(record, now))
        record.update(UsedCount=used, CountDate=now.date().isoformat(), NextRun=now)
        self._save_city(
            city_key, UsedCount=used, CountDate=record['CountDate'], Pending=False,
            NextRun=drum_next_run(record, now),
            Status=f'今日已用{used}/10次，严禁继续击鼓，明日再检查',
        )
        self.ui_goto_main()

    def _finish_strike(self):
        """Only dismiss the result; never retry the strike button."""
        deadline = time.monotonic() + 20
        settled_at = time.monotonic() + 2
        skipped = False
        main_frames = 0
        panel_checked = False
        logger.info('Drum clicked; wait for animation or a confirmed daily-count change')
        while time.monotonic() < deadline:
            self.device.sleep(0.4)
            self.device.screenshot()
            if self.is_network_timeout_popup():
                self.handle_network_timeout_popup()
                return False
            if self.appear(SKIP, offset=(16, 16), similarity=0.85):
                self.device.click(SKIP)
                skipped = True
                main_frames = 0
                continue
            if (not skipped and not panel_checked
                    and time.monotonic() >= settled_at and self._panel_ready()):
                panel_checked = True
                observed = self._read_count()
                expected = getattr(self, '_drum_expected_count', None)
                if observed is not None and expected is not None:
                    if observed >= expected:
                        self._drum_confirmed_count = observed
                        logger.info(f'Drum completed without a visible animation: count={observed}')
                        self.ui_goto_main()
                        return True
                    if observed == expected - 1 and self._read_work_count() == 0:
                        self._drum_no_work_count = observed
                        logger.info('Drum click had no work and did not consume a daily use')
                        return False
            if (skipped and not self._panel_ready()
                    and not self.is_building_search_page()
                    and self.is_unobstructed_main()):
                main_frames += 1
                if main_frames >= 2:
                    return True
            else:
                main_frames = 0
        return False

    def _read_work_count(self):
        """Read available assignments, separately from the daily-use sentence."""
        counts = []
        for _ in range(2):
            self.device.screenshot()
            if not self._panel_ready():
                return None
            count = read_confirmed_digits(
                self.device.image, self.anchored_area(PANEL, self.WORK_COUNT_AREA),
            )
            if count is None:
                return None
            counts.append(count)
            self.device.sleep(0.2)
        result = counts[0] if counts[0] == counts[1] else None
        logger.attr('ChaomuDrumAvailableWork', result)
        return result

    def _save_no_work(self, city_key, observed):
        now = datetime.now().replace(microsecond=0)
        record = self._record(city_key)
        local = drum_used_count(record, now)
        used = max(local, observed)
        pending = bool(record.get('Pending'))
        # Only release the single tentative use when two-frame OCR confirms
        # an unchanged daily count AND zero assignments. Confirmed uses stay.
        if pending and local == observed + 1:
            used = observed
            pending = False
        elif observed >= local:
            pending = False
        record.update(
            UsedCount=used, CountDate=now.date().isoformat(),
            Pending=pending, NextRun=now + DRUM_INTERVAL,
        )
        self._save_city(
            city_key, UsedCount=used, CountDate=record['CountDate'], Pending=pending,
            NextRun=drum_next_run(record, now),
            Status=f'暂无可执行工作，未消耗次数；今日记录{used}/10次' + (
                '（含待核实预记）' if pending else ''
            ),
        )
        logger.info(f'No drum work in {CITY_KEY_TO_NAME[city_key]}; check again in 2 hours')

    def _retry_city(self, city_key, status):
        now = datetime.now().replace(microsecond=0)
        record = self._record(city_key)
        record['NextRun'] = now + self.RETRY
        self._save_city(
            city_key, Status=status, NextRun=drum_next_run(record, now),
        )

    def _strike_city(self, city_key):
        now = datetime.now().replace(microsecond=0)
        record = self._record(city_key)
        target = drum_next_run(record, now)
        if target > now:
            self._save_city(city_key, NextRun=target)
            return
        if not self.search_building('朝暮鼓', exact=True):
            if self.building_search_confirmed_missing():
                self.disable_city_building_feature(
                    task=self.TASK, group_prefix='chaomu_drum',
                    city_key=city_key, city=CITY_KEY_TO_NAME[city_key],
                    building_name='朝暮鼓', status_field='Status',
                    status_text='未找到朝暮鼓，已自动关闭本城',
                )
                self.ui_goto_main()
            else:
                self._retry_city(city_key, '搜索未完成，稍后重试')
            return
        if not self._open_panel():
            self._retry_city(city_key, '未能确认朝暮鼓界面，未敲鼓')
            return
        observed = self._read_count()
        now = datetime.now().replace(microsecond=0)
        if observed is None:
            self._retry_city(city_key, '今日次数识别不确定，未敲鼓')
            self.ui_goto_main()
            return
        # Reading the in-game sentence is mandatory even after a date rollover.
        # Never lower a same-day reservation after an ambiguous click/restart.
        record = self._record(city_key)
        used = max(drum_used_count(record, now), observed)
        record.update(UsedCount=used, CountDate=now.date().isoformat())
        if used >= DRUM_DAILY_LIMIT:
            self._stop_at_limit(city_key, used)
            return
        available = self._read_work_count()
        if available == 0:
            self._save_no_work(city_key, observed)
            self.ui_goto_main()
            return
        if available is None:
            self._save_city(city_key, UsedCount=used, CountDate=record['CountDate'])
            self._retry_city(city_key, '可执行工作数量识别不确定，未敲鼓')
            self.ui_goto_main()
            return
        # Recheck eligibility after OCR and preserve a single atomic reservation.
        target = drum_next_run(record, now)
        if target > now:
            self._save_city(
                city_key, UsedCount=used, CountDate=record['CountDate'],
                NextRun=target,
            )
            self.ui_goto_main()
            return
        record.update(
            UsedCount=used + 1, LastAttempt=now, Pending=True,
            NextRun=now + DRUM_INTERVAL,
        )
        self._save_city(
            city_key, UsedCount=record['UsedCount'], CountDate=record['CountDate'],
            LastAttempt=now, Pending=True, NextRun=drum_next_run(record, now),
            Status=f'已预记第{used + 1}/10次，等待敲鼓结果',
        )
        # Disk reservation precedes this ONE click, including device exceptions.
        self.device.screenshot()
        if not self._panel_ready():
            self._save_city(
                city_key, Status=f'第{used + 1}/10次已预记，界面变化，暂停点击待复查',
            )
            self.ui_goto_main()
            return
        fresh = self._read_count()
        if fresh is not None and fresh >= DRUM_DAILY_LIMIT:
            self._stop_at_limit(city_key, fresh)
            return
        if fresh is None or fresh != observed:
            self._save_city(city_key, Status='点击前次数或费用变化，保留预记并停止击鼓待复查')
            self.ui_goto_main()
            return
        self.device.screenshot()
        if not self._panel_ready():
            self._save_city(city_key, Status='点击前界面变化，停止击鼓待复查')
            self.ui_goto_main()
            return
        if self._cost_currency() == 'gem':
            self._stop_at_limit(city_key)
            return
        self.device.click(self.anchored_button(PANEL, STRIKE))
        self._drum_expected_count = used + 1
        self._drum_no_work_count = None
        self._drum_confirmed_count = None
        if self._finish_strike():
            confirmed = max(used + 1, self._drum_confirmed_count or 0)
            record['UsedCount'] = confirmed
            self._save_city(
                city_key, UsedCount=confirmed, Pending=False,
                NextRun=drum_next_run(record, datetime.now().replace(microsecond=0)),
                Status=f'敲鼓完成，今日{confirmed}/10次' + (
                    '；今日已达上限' if confirmed >= DRUM_DAILY_LIMIT else ''
                ),
            )
        elif self._drum_no_work_count is not None:
            self._save_no_work(city_key, self._drum_no_work_count)
            self.ui_goto_main()
        else:
            # The server may have accepted it. Keep the reserved quota and clock.
            self._save_city(
                city_key, Status=f'第{used + 1}/10次结果待核实，已保留次数和2小时间隔',
            )
            self.ui_goto_main()

    def run(self):
        logger.hr('JIANGNAN CHAOMU DRUM', level=0)
        for city_key, city in CITY_KEY_TO_NAME.items():
            record = self._record(city_key)
            if not record.get('Enable', False):
                continue
            now = datetime.now().replace(microsecond=0)
            target = drum_next_run(record, now)
            if target > now:
                self._save_city(city_key, NextRun=target)
                continue
            if not self.switch_city(city):
                self._retry_city(city_key, '切换城市失败，未敲鼓')
                continue
            self._strike_city(city_key)

        now = datetime.now().replace(microsecond=0)
        times = [
            drum_next_run(self._record(key), now)
            for key in CITY_KEY_TO_NAME if self._record(key).get('Enable', False)
        ]
        self.config.modified[f'{self.TASK}.ChaomuDrumSettings.Status'] = (
            '逐城检查完成；每城间隔2小时，每日最多10次'
            if times else '未启用任何城市'
        )
        self.config.task_delay(target=min(times) if times else now + DRUM_INTERVAL)
        return True
