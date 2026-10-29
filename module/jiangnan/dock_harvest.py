from datetime import datetime, timedelta

from module.jiangnan.inventory_state import (
    CITY_NAME_TO_KEY,
    JiangnanInventoryState,
)
from module.jiangnan.transport import JiangnanTransportStatus
from module.logger import logger


class JiangnanDockHarvest(JiangnanTransportStatus):
    """扫描运输总览并收取全部已抵达船只。"""

    # 台账无法覆盖玩家手动发出的运输、旧版本遗留运输以及曾经 OCR
    # 失败的节点，因此即使没有台账倒计时，也要定期实扫运输总览。
    AUDIT_INTERVAL_MINUTES = 10

    @staticmethod
    def _request_untracked_arrival_scan(
        state, collected_slots, tracked_keys, unknown_count, reason,
    ):
        """实扫手动运输的来源和目的城；目的地未识别时才全城补扫。"""
        if not unknown_count:
            return
        untracked = [
            item for item in collected_slots
            if (item[0], int(item[1])) not in tracked_keys
        ]
        cities = set()
        for item in untracked:
            destination = item[4] if len(item) > 4 else None
            if destination not in CITY_NAME_TO_KEY:
                state.request_scan(reason)
                return
            cities.update((item[0], destination))
        if len(untracked) != unknown_count or not cities:
            state.request_scan(reason)
            return
        state.request_scan(reason, cities=cities)

    def _full_audit_due(self, state, now):
        last = state._parse_time(state.data.get('last_full_dock_audit_at'))
        return (
            last is None
            or now - last >= timedelta(minutes=self.AUDIT_INTERVAL_MINUTES)
        )

    @staticmethod
    def _stored_arrived_cities(state):
        return {
            city
            for city, slots in state.data.get('slot_status', {}).items()
            if any(slot.get('state') == 'arrived' for slot in slots)
        }

    def run(self):
        logger.hr('JIANGNAN DOCK HARVEST', level=0)
        state = JiangnanInventoryState(self.config)
        now = datetime.now().replace(microsecond=0)
        # 到达时间先计入目的城库存；本任务只确认船位、处理仓满和手动运输。
        arrived_count = state.apply_due_arrivals(now, wake_dock=False)
        logger.attr('DockDueInventoryNodes', arrived_count)
        # 台账倒计时只用于准确唤醒，不能作为“是否扫描/是否收获”的
        # 唯一依据。每次运行都实扫所有来源城市的船位，这样手动发出、
        # 旧版本遗留或漏记的运输也会被发现。
        due = state.due_dock_shipments(now)
        logger.attr('DockHarvestDueNodes', len(due))
        due_sources = {
            shipment.get('source') for shipment in due
            if shipment.get('source') in CITY_NAME_TO_KEY
        }
        targeted_cities = due_sources | self._stored_arrived_cities(state)
        full_audit = self._full_audit_due(state, now)
        if full_audit:
            # 无台账的手动运输仍靠周期全量审计补漏，但最多十分钟一次。
            # 这里只决定扫描范围；必须等整次扫描成功后才能登记审计
            # 时间。若在前往目的城市、切换地基模式或查找码头时失败，
            # 提前登记会让下一轮误以为刚审计过，从而漏掉页面上已抵达
            # 但不在本地台账中的运输。
            audit_cities = list(CITY_NAME_TO_KEY)
        else:
            audit_cities = [
                city for city in CITY_NAME_TO_KEY if city in targeted_cities
            ]
        logger.attr('DockHarvestFullAudit', full_audit)
        logger.attr('DockHarvestAuditCities', audit_cities)
        if not audit_cities:
            next_audit = (
                state._parse_time(state.data.get('last_full_dock_audit_at'))
                + timedelta(minutes=self.AUDIT_INTERVAL_MINUTES)
            )
            state.sync_dock_schedule(fallback=next_audit)
            self.config.task_delay(target=next_audit)
            return True
        success, _, has_unknown, collected = self.scan(
            cities=audit_cities,
            collect_arrived=True,
            # 不限制为台账中的船位：这正是实扫用于补漏的部分。
            collect_slots=None,
            skip_missing=True,
        )
        if not success:
            # 导航、输入法或动画造成的一次短暂失败不应让同一条过期
            # 台账节点立即再次唤醒任务，并在几十秒内累计三次失败退出。
            # 已经成功收取的船位先入账；仍到期的节点只延后“码头复查
            # 时间”，不改运输完成时间和已经结算的目的地库存。
            tracked_keys = {
                (item.get('source'), int(item.get('slot', -1)))
                for item in state.active_shipments()
            }
            matched, unknown_arrivals = state.apply_collected(
                self.last_collected_slots
            )
            self._request_untracked_arrival_scan(
                state, self.last_collected_slots, tracked_keys,
                unknown_arrivals,
                'untracked dock arrival harvested before retry',
            )
            retry_at = now + timedelta(
                minutes=self.AUDIT_INTERVAL_MINUTES
            )
            deferred = 0
            for shipment in state.due_dock_shipments(now):
                shipment['dock_next_at'] = state._time_text(retry_at)
                deferred += 1
            state.save()
            state.sync_dock_schedule(fallback=retry_at)
            logger.warning(
                'Dock harvest audit will retry after transient failure: '
                f'collected={matched}, deferred={deferred}, retry={retry_at}'
            )
            self.config.task_delay(target=retry_at)
            # 本轮已安全安排重试，不把可恢复的 UI 失败计入调度器的
            # 连续失败次数。
            return True

        if full_audit:
            state.data['last_full_dock_audit_at'] = state._time_text(now)
            state.save()

        # 用这次实扫纠正已经到期、但总览仍显示运输中的台账时间，避免
        # 旧的过期节点让调度器立即重复运行。若码头的一次批量收获顺带
        # 清掉了其他来源的完成货物，它们在后续总览中会显示 idle，也要
        # 一并消费对应台账。
        snapshot_by_slot = {
            (city, int(slot.get('slot', -1))): slot
            for city, slots in self.last_snapshot.items()
            for slot in slots
        }
        for shipment in list(state.active_shipments()):
            slot_key = (
                shipment.get('source'),
                int(shipment.get('slot', -1)),
            )
            slot = snapshot_by_slot.get(slot_key)
            if slot is None:
                continue
            dock_next = state._parse_time(
                shipment.get('dock_next_at') or shipment.get('finish_at')
            )
            if (
                slot.get('state') == 'transporting'
                and slot.get('finish_time') is not None
                and (dock_next is None or dock_next <= now)
            ):
                state.update_shipment_finish(
                    shipment,
                    slot['finish_time'].replace(microsecond=0),
                )
            elif slot.get('state') == 'occupied_unknown' and (
                dock_next is None or dock_next <= now
            ):
                state.update_shipment_finish(
                    shipment,
                    now + timedelta(minutes=self.AUDIT_INTERVAL_MINUTES),
                )

        tracked_keys = {
            (item.get('source'), int(item.get('slot', -1)))
            for item in state.active_shipments()
        }
        matched, unknown_arrivals = state.apply_collected(
            self.last_collected_slots
        )
        collected_keys = {
            (item[0], int(item[1])) for item in self.last_collected_slots
        }
        cleared_tracked_slots = []
        for shipment in state.active_shipments():
            slot_key = (
                shipment.get('source'),
                int(shipment.get('slot', -1)),
            )
            slot = snapshot_by_slot.get(slot_key)
            if (
                slot_key not in collected_keys
                and slot is not None
                and slot.get('state') in ('idle', 'collected')
            ):
                # 这里只知道“台账还在运输，但现在船位已空”，并没有亲眼
                # 看到本轮点击收取。把它标成 observed_only，库存交给实扫
                # 纠正，避免玩家手动领取后又按订单量重复加账。
                cleared_tracked_slots.append(
                    (slot_key[0], slot_key[1], False, True)
                )
        batch_matched, batch_unknown = state.apply_collected(
            cleared_tracked_slots
        )
        matched += batch_matched
        unknown_arrivals += batch_unknown
        logger.attr('DockHarvestMatchedArrivals', matched)
        logger.attr('DockHarvestUnknownArrivals', unknown_arrivals)
        state.record_slot_snapshot(self.last_snapshot)
        if unknown_arrivals:
            # 手动运输没有可靠的货物种类和数量，以仓库实扫更新库存。
            self._request_untracked_arrival_scan(
                state, self.last_collected_slots, tracked_keys,
                unknown_arrivals, 'untracked dock arrival harvested',
            )
        if has_unknown:
            logger.warning('Occupied transport slot has unknown remaining time')

        fallback = now + timedelta(minutes=self.AUDIT_INTERVAL_MINUTES)
        pending_times = state.sync_dock_schedule(fallback=fallback)
        logger.attr('DockHarvestCollected', collected)
        if pending_times:
            self.config.task_delay(target=min(pending_times))
        else:
            self.config.task_delay(success=True)
        return True
