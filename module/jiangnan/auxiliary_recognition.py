"""One-shot recognition actions launched by auxiliary-information buttons."""

from module.jiangnan.allocation import CITY_KEY_TO_NAME, JiangnanAllocation
from module.jiangnan.city import (
    CITY_KEY_TO_PRODUCTION_TASK,
    JiangnanCity,
    KNOWN_CITIES,
)
from module.jiangnan.city_production import JiangnanCityProduction
from module.jiangnan.resource_layout import PAGE_CITY, allocation_layout_page
from module.jiangnan.warehouse import JiangnanWarehouse
from module.logger import logger


def _recognize_city_layout(config, device):
    runner = JiangnanCity(config=config, device=device)
    if not runner.goto_city_list():
        return False

    observed = {}
    runner.city_list_reset_top()
    device.screenshot()
    top_names = runner.read_city_names()
    for index, name in enumerate(top_names):
        if name in KNOWN_CITIES:
            observed[name] = divmod(index, 2)

    runner.city_list_reset_bottom()
    device.screenshot()
    bottom_names = runner.read_city_names()
    anchors = []
    for index, name in enumerate(bottom_names):
        if name in observed:
            anchors.append(observed[name][0] - index // 2)
    bottom_first_row = max(anchors) if anchors else 3
    for index, name in enumerate(bottom_names):
        if name in KNOWN_CITIES:
            observed[name] = (
                bottom_first_row + index // 2,
                index % 2,
            )

    missing = [city for city in KNOWN_CITIES if city not in observed]
    if missing:
        logger.warning(
            f'City auxiliary recognition incomplete; keep existing layout: '
            f'recognized={len(observed)}/{len(KNOWN_CITIES)}, '
            f'missing={missing}'
        )
        runner.ui_goto_main()
        return False
    runner.city_layout.replace_layout(PAGE_CITY, observed)
    runner.ui_goto_main()
    logger.info(f'City auxiliary information updated: {len(observed)} cities')
    return True


def _recognize_warehouse_layout(config, device):
    runner = JiangnanWarehouse(config=config, device=device)
    result = runner.scan_inventory(
        return_main=True,
        learn_layout=True,
    )
    return result is not None


def _recognize_allocation_layout(config, device, city_key):
    source = CITY_KEY_TO_NAME[city_key]
    runner = JiangnanAllocation(config=config, device=device)
    try:
        if not runner.open_transport(source):
            if runner.last_error == 'dock_unrepaired':
                runner.resource_layout.replace_status(
                    allocation_layout_page(source), '码头未修复'
                )
                logger.info(
                    f'Allocation auxiliary information updated: '
                    f'city={source}, status=码头未修复'
                )
                return True
            return False
        destinations = [
            city for city in CITY_KEY_TO_NAME.values() if city != source
        ]
        opened = False
        for destination in destinations:
            if runner.choose_destination(destination):
                opened = True
                break
            device.screenshot()
            if not runner.is_destination_page():
                break
        if not opened:
            logger.warning(
                f'Unable to open allocation resources for {source}'
            )
            return False
        return runner.scan_local_resource_layout(source)
    finally:
        JiangnanCity(config=config, device=device).ui_goto_main()


def _recognize_production_layout(config, device, city_key):
    city = CITY_KEY_TO_NAME[city_key]
    task = CITY_KEY_TO_PRODUCTION_TASK[city_key]
    runner = JiangnanCityProduction(
        config=config,
        device=device,
        task_name=task,
    )
    try:
        if not runner.switch_city(city):
            return False
        result = runner._scan_production_layout(enable_city_task=True)
        return result is not None
    finally:
        runner._restore_unfiltered_search()
        runner.ui_goto_main()


def run_auxiliary_recognition(config_name, kind, target):
    """Entry point used by the WebUI action-button subprocess."""
    from alas import AzurLaneAutoScript

    logger.hr(
        f'JIANGNAN AUXILIARY RECOGNITION: {kind}/{target}', level=0
    )
    app = AzurLaneAutoScript(config_name=config_name)
    if kind == 'city':
        return _recognize_city_layout(app.config, app.device)
    if kind == 'warehouse':
        return _recognize_warehouse_layout(app.config, app.device)
    if target not in CITY_KEY_TO_NAME:
        raise KeyError(f'Unknown Jiangnan auxiliary city: {target}')
    if kind == 'allocation':
        return _recognize_allocation_layout(
            app.config, app.device, target
        )
    if kind == 'production':
        return _recognize_production_layout(
            app.config, app.device, target
        )
    raise KeyError(f'Unknown Jiangnan auxiliary recognition kind: {kind}')
