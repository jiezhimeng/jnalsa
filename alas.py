import os
import re
import threading
import time
from datetime import datetime, timedelta

import inflection
from cached_property import cached_property

from module.base.decorator import del_cached_property
from module.config.config import AzurLaneConfig, TaskEnd
from module.config.deep import deep_get, deep_set
from module.exception import *
from module.logger import logger
from module.notify import handle_notify
from module.jiangnan.network_popup import NetworkPopupInterrupted
from module.base.task_status import running_task


class AzurLaneAutoScript:
    stop_event: threading.Event = None

    def __init__(self, config_name='alas'):
        logger.hr('Start', level=0)
        self.config_name = config_name
        # Skip first restart
        self.is_first_task = True
        # Failure count of tasks
        # Key: str, task name, value: int, failure count
        self.failure_record = {}

    @cached_property
    def config(self):
        try:
            config = AzurLaneConfig(config_name=self.config_name)
            return config
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            logger.exception(e)
            exit(1)

    @cached_property
    def device(self):
        try:
            from module.jiangnan.device import JiangnanDevice

            device = JiangnanDevice(
                config=self.config
            )

            return device
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            logger.exception(e)
            exit(1)


    def run(self, command, skip_first_screenshot=False):
        previous_network_watch = getattr(self.device, '_jiangnan_network_watch_enabled', False)
        previous_reward_observer = getattr(self.device, '_jiangnan_task_reward_observer', None)
        watch_rewards = command != 'jiangnan_task_reward' and (
            command.startswith('jiangnan_') or command in ('start', 'goto_main', 'restart'))
        if watch_rewards:
            from module.jiangnan.task_reward_trigger import enabled, watcher
            self.device._jiangnan_task_reward_observer = (
                watcher(self.device, self.config).observe if enabled(self.config) else None)
        self.device._jiangnan_network_watch_enabled = True
        try:
            try:
                if not skip_first_screenshot:
                    self.device.screenshot()
                result = self.__getattribute__(command)()
            except NetworkPopupInterrupted:
                from module.jiangnan.app import JiangnanApp

                logger.warning('Network popup interrupted task; recover before resuming scheduling')
                app = JiangnanApp(config=self.config, device=self.device)
                app.handle_network_timeout_popup()
                # The interrupted task remains due. Re-enter it from its
                # normal page checks instead of continuing stale coordinates.
                return True
            return result is not False
        except TaskEnd:
            return True
        except GameNotRunningError as e:
            logger.warning(e)
            self.config.task_call('Restart')
            return False
        except (GameStuckError, GameTooManyClickError) as e:
            logger.error(e)
            self.save_error_log()
            logger.warning(f'Game stuck, {self.device.package} will be restarted in 10 seconds')
            logger.warning('If you are playing by hand, please stop Alas')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GameBugError as e:
            logger.warning(e)
            self.save_error_log()
            logger.warning('An error has occurred in the Jiangnan game client')
            logger.warning(f'Restarting {self.device.package} to fix it')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GamePageUnknownError:
            logger.critical('Jiangnan page unknown; inspect the saved screenshot')
            self.save_error_log()
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Jiangnan <{self.config_name}> crashed",
                content=f"<{self.config_name}> GamePageUnknownError",
            )
            raise RequestHumanTakeover('Jiangnan page unknown')
        except ScriptError as e:
            logger.exception(e)
            logger.critical('This is likely to be a mistake of developers, but sometimes just random issues')
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> crashed",
                content=f"<{self.config_name}> ScriptError",
            )
            exit(1)
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> crashed",
                content=f"<{self.config_name}> RequestHumanTakeover",
            )
            exit(1)
        except Exception as e:
            logger.exception(e)
            self.save_error_log()
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> crashed",
                content=f"<{self.config_name}> Exception occured",
            )
            exit(1)
        finally:
            self.device._jiangnan_network_watch_enabled = previous_network_watch
            self.device._jiangnan_task_reward_observer = previous_reward_observer
            if watch_rewards:
                from module.jiangnan.task_reward_trigger import flush_request
                flush_request(self.device, self.config)

    def save_error_log(self):
        """
        Save last 60 screenshots in ./log/error/<timestamp>
        Save logs to ./log/error/<timestamp>/log.txt
        """
        from module.base.utils import save_image
        from module.base.sensitive_info import (handle_sensitive_image,
                                                   handle_sensitive_logs)
        if self.config.Error_SaveError:
            if not os.path.exists('./log/error'):
                os.mkdir('./log/error')
            folder = f'./log/error/{int(time.time() * 1000)}'
            logger.warning(f'Saving error: {folder}')
            os.mkdir(folder)
            for data in self.device.screenshot_deque:
                image_time = datetime.strftime(data['time'], '%Y-%m-%d_%H-%M-%S-%f')
                image = handle_sensitive_image(data['image'])
                save_image(image, f'{folder}/{image_time}.png')
            with open(logger.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start = 0
                for index, line in enumerate(lines):
                    line = line.strip(' \r\t\n')
                    if re.match('^═{15,}$', line):
                        start = index
                lines = lines[start - 2:]
                lines = handle_sensitive_logs(lines)
            with open(f'{folder}/log.txt', 'w', encoding='utf-8') as f:
                f.writelines(lines)

    def restart(self):
        from module.jiangnan.app import JiangnanApp

        logger.hr(
            'JIANGNAN RESTART'
        )

        app = JiangnanApp(
            self.config,
            device=self.device
        )

        app.restart_game_until_ready(
            reason='scheduler restart task', attempts=3
        )

        # Restart 自己也必须安排下一次运行
        self.config.task_delay(
            server_update=True
        )

    def start(self):
        from module.jiangnan.app import JiangnanApp

        app = JiangnanApp(
            self.config,
            device=self.device
        )

        return app.ensure_ready()

    def goto_main(self):
        from module.jiangnan.app import JiangnanApp

        app = JiangnanApp(
            self.config,
            device=self.device
        )

        return app.ensure_ready()




































































    def wait_until(self, future):
        """
        Wait until a specific time.

        Args:
            future (datetime):

        Returns:
            bool: True if wait finished, False if config changed.
        """
        future = future + timedelta(seconds=1)
        self.config.start_watching()
        while 1:
            if datetime.now() > future:
                return True
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"[{self.config_name}] exited. Reason: Update")
                    exit(0)

            time.sleep(5)

            if self.config.should_reload():
                return False
            if self.config.Optimization_WhenTaskQueueEmpty != 'close_game':
                from module.jiangnan.task_reward_trigger import poll_task_rewards
                if poll_task_rewards(self.device, self.config):
                    return False

    def get_next_task(self):
        """
        Returns:
            str: Name of the next task.
        """
        while 1:
            task = self.config.get_next()
            self.config.task = task
            self.config.bind(task)

            from module.base.resource import release_resources
            if self.config.task.command != 'Alas':
                release_resources(next_task=task.command)

            if task.next_run > datetime.now():
                logger.info(f'Wait until {task.next_run} for task `{task.command}`')
                self.is_first_task = False
                method = self.config.Optimization_WhenTaskQueueEmpty
                if method == 'close_game':
                    logger.info('Close game during wait')
                    self.device.app_stop()
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                    if task.command != 'Restart':
                        self.config.task_call('Restart')
                        del_cached_property(self, 'config')
                        continue
                elif method == 'goto_main':
                    logger.info('Goto main page during wait')
                    self.run('goto_main')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                elif method == 'stay_there':
                    logger.info('Stay there during wait')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                else:
                    logger.warning(f'Invalid Optimization_WhenTaskQueueEmpty: {method}, fallback to stay_there')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
            break

        AzurLaneConfig.is_hoarding_task = False
        return task.command

    def loop(self):
        logger.set_file_logger(self.config_name)
        logger.info(f'Start scheduler loop: {self.config_name}')

        while 1:
            # Check update event from GUI
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"Alas [{self.config_name}] exited.")
                    break
            # Get task
            task = self.get_next_task()
            # Init device and change server
            _ = self.device
            self.device.config = self.config
            # Skip first restart
            if self.is_first_task and task == 'Restart':
                logger.info('Skip task `Restart` at scheduler start')
                self.config.task_delay(server_update=True)
                del_cached_property(self, 'config')
                continue

            # Run
            with running_task(task):
                logger.info(f'Scheduler: Start task `{task}`')
                self.device.stuck_record_clear()
                self.device.click_record_clear()
                logger.hr(task, level=0)
                success = self.run(inflection.underscore(task))
                logger.info(f'Scheduler: End task `{task}`')
            self.is_first_task = False

            # Check failures
            failed = deep_get(self.failure_record, keys=task, default=0)
            failed = 0 if success else failed + 1
            deep_set(self.failure_record, keys=task, value=failed)
            if failed >= 3:
                logger.critical(f"Task `{task}` failed 3 or more times.")
                logger.critical("Possible reason #1: You haven't used it correctly. "
                                "Please read the help text of the options.")
                logger.critical("Possible reason #2: There is a problem with this task. "
                                "Please contact developers or try to fix it yourself.")
                logger.critical('Request human takeover')
                handle_notify(
                    self.config.Error_OnePushConfig,
                    title=f"Alas <{self.config_name}> crashed",
                    content=f"<{self.config_name}> RequestHumanTakeover\nTask `{task}` failed 3 or more times.",
                )
                exit(1)

            if success:
                del_cached_property(self, 'config')
                continue
            elif self.config.Error_HandleError:
                # self.config.task_delay(success=False)
                del_cached_property(self, 'config')
                continue
            else:
                break

    def jiangnan_signin(self):
        from module.jiangnan.signin import JiangnanSignin

        return JiangnanSignin(
            config=self.config,
            device=self.device
        ).run()

    def jiangnan_task_reward(self):
        from module.jiangnan.task_reward import JiangnanTaskReward

        return JiangnanTaskReward(config=self.config, device=self.device).run()

    def jiangnan_warehouse_export(self):
        from module.jiangnan.warehouse import JiangnanWarehouse

        return JiangnanWarehouse(
            config=self.config,
            device=self.device
        ).scan_inventory_by_city()

    def jiangnan_warehouse_inventory(self):
        from module.jiangnan.warehouse_inventory import JiangnanWarehouseInventory

        return JiangnanWarehouseInventory(
            config=self.config,
            device=self.device
        ).run()

    def jiangnan_city_allocation(self):
        from module.jiangnan.allocation_planner import JiangnanAllocationPlanner

        return JiangnanAllocationPlanner(
            config=self.config,
            device=self.device
        ).run()

    def jiangnan_dock_harvest(self):
        from module.jiangnan.dock_harvest import JiangnanDockHarvest

        return JiangnanDockHarvest(
            config=self.config,
            device=self.device
        ).run()

    def jiangnan_tax_harvest(self):
        from module.jiangnan.tax_harvest import JiangnanTaxHarvest

        return JiangnanTaxHarvest(
            config=self.config,
            device=self.device
        ).run()

    def jiangnan_chaomu_drum(self):
        from module.jiangnan.chaomu_drum import JiangnanChaomuDrum

        return JiangnanChaomuDrum(
            config=self.config, device=self.device
        ).run()

    def jiangnan_farm_production(self):
        from module.jiangnan.farm_production import JiangnanFarmProduction

        return JiangnanFarmProduction(
            config=self.config,
            device=self.device
        ).run()

    def _jiangnan_city_production(self, task_name):
        from module.jiangnan.city_production import JiangnanCityProduction

        return JiangnanCityProduction(
            config=self.config,
            device=self.device,
            task_name=task_name,
        ).run()

    def jiangnan_yingtianfu_production(self):
        return self._jiangnan_city_production('JiangnanYingtianfuProduction')

    def jiangnan_suzhoufu_production(self):
        return self._jiangnan_city_production('JiangnanSuzhoufuProduction')

    def jiangnan_hangzhoufu_production(self):
        return self._jiangnan_city_production('JiangnanHangzhoufuProduction')

    def jiangnan_songjiangfu_production(self):
        return self._jiangnan_city_production('JiangnanSongjiangfuProduction')

    def jiangnan_jimingshan_production(self):
        return self._jiangnan_city_production('JiangnanJimingshanProduction')

    def jiangnan_huizhoufu_production(self):
        return self._jiangnan_city_production('JiangnanHuizhoufuProduction')

    def jiangnan_yangzhoufu_production(self):
        return self._jiangnan_city_production('JiangnanYangzhoufuProduction')

    def jiangnan_shaoxingfu_production(self):
        return self._jiangnan_city_production('JiangnanShaoxingfuProduction')

    def jiangnan_ningbofu_production(self):
        return self._jiangnan_city_production('JiangnanNingbofuProduction')

    def jiangnan_jiangningxian_production(self):
        return self._jiangnan_city_production('JiangnanJiangningxianProduction')

    def jiangnan_baixuezhen_production(self):
        return self._jiangnan_city_production('JiangnanBaixuezhenProduction')

    def jiangnan_zongyingdao_production(self):
        return self._jiangnan_city_production('JiangnanZongyingdaoProduction')

    def jiangnan_longgong_production(self):
        return self._jiangnan_city_production('JiangnanLonggongProduction')

    def _jiangnan_production_facility(self, facility, task_name):
        from module.jiangnan.production import JiangnanProduction

        return JiangnanProduction(
            config=self.config,
            device=self.device,
            facility=facility,
            task_name=task_name,
        ).run()

    def jiangnan_corn_drying_mat(self):
        from module.jiangnan.corn_drying_mat import JiangnanCornDryingMat

        return JiangnanCornDryingMat(config=self.config, device=self.device).run()

    def _jiangnan_multi_product_facility(self, facility, task_name):
        from module.jiangnan.multi_product_production import (
            JiangnanMultiProductProduction,
        )

        return JiangnanMultiProductProduction(
            config=self.config,
            device=self.device,
            facility=facility,
            task_name=task_name,
        ).run()

    def jiangnan_restaurant_production(self):
        return self._jiangnan_production_facility('餐馆', 'JiangnanRestaurantProduction')

    def jiangnan_logging_camp_production(self):
        return self._jiangnan_production_facility('林场', 'JiangnanLoggingCampProduction')

    def jiangnan_fish_pond_production(self):
        return self._jiangnan_production_facility('鱼塘', 'JiangnanFishPondProduction')

    def jiangnan_sawmill_production(self):
        return self._jiangnan_production_facility('锯木厂', 'JiangnanSawmillProduction')

    def jiangnan_paper_mill_production(self):
        return self._jiangnan_production_facility('造纸厂', 'JiangnanPaperMillProduction')

    def jiangnan_cotton_field_production(self):
        return self._jiangnan_production_facility('棉花田', 'JiangnanCottonFieldProduction')

    def jiangnan_weaving_workshop_production(self):
        return self._jiangnan_production_facility('织布坊', 'JiangnanWeavingWorkshopProduction')

    def jiangnan_tailor_shop_production(self):
        return self._jiangnan_production_facility('裁缝铺', 'JiangnanTailorShopProduction')

    def jiangnan_clay_mine_production(self):
        return self._jiangnan_production_facility('黏土矿', 'JiangnanClayMineProduction')

    def jiangnan_porcelain_kiln_production(self):
        return self._jiangnan_production_facility('瓷窑', 'JiangnanPorcelainKilnProduction')

    def jiangnan_charcoal_kiln_production(self):
        return self._jiangnan_production_facility('烧炭厂', 'JiangnanCharcoalKilnProduction')

    def jiangnan_herb_field_production(self):
        return self._jiangnan_production_facility('草药田', 'JiangnanHerbFieldProduction')

    def jiangnan_alchemy_furnace_production(self):
        return self._jiangnan_production_facility('炼丹炉', 'JiangnanAlchemyFurnaceProduction')

    def jiangnan_tung_oil_plantation_production(self):
        return self._jiangnan_production_facility('桐油种植园', 'JiangnanTungOilPlantationProduction')

    def jiangnan_oil_paper_workshop_production(self):
        return self._jiangnan_production_facility('油纸厂', 'JiangnanOilPaperWorkshopProduction')

    def jiangnan_umbrella_workshop_production(self):
        return self._jiangnan_production_facility('制伞坊', 'JiangnanUmbrellaWorkshopProduction')

    def jiangnan_brocade_workshop_production(self):
        return self._jiangnan_production_facility('云锦织坊', 'JiangnanBrocadeWorkshopProduction')

    def jiangnan_lacquer_forest_production(self):
        return self._jiangnan_production_facility('漆树林', 'JiangnanLacquerForestProduction')

    def jiangnan_lacquer_workshop_production(self):
        return self._jiangnan_production_facility('漆器作坊', 'JiangnanLacquerWorkshopProduction')

    def jiangnan_mother_of_pearl_workshop_production(self):
        return self._jiangnan_production_facility('螺钿漆坊', 'JiangnanMotherOfPearlWorkshopProduction')

    def jiangnan_water_caltrop_pond_production(self):
        return self._jiangnan_production_facility('菱角池塘', 'JiangnanWaterCaltropPondProduction')

    def jiangnan_silkworm_farm_production(self):
        return self._jiangnan_production_facility('养蚕场', 'JiangnanSilkwormFarmProduction')

    def jiangnan_silk_workshop_production(self):
        return self._jiangnan_production_facility('丝织坊', 'JiangnanSilkWorkshopProduction')

    def jiangnan_embroidery_workshop_production(self):
        return self._jiangnan_production_facility('绣坊', 'JiangnanEmbroideryWorkshopProduction')

    def jiangnan_duck_farm_production(self):
        return self._jiangnan_production_facility('养鸭场', 'JiangnanDuckFarmProduction')

    def jiangnan_iron_mine_production(self):
        return self._jiangnan_production_facility('铁矿场', 'JiangnanIronMineProduction')

    def jiangnan_iron_workshop_production(self):
        return self._jiangnan_production_facility('冶铁工坊', 'JiangnanIronWorkshopProduction')

    def jiangnan_weapon_workshop_production(self):
        return self._jiangnan_production_facility('兵器工坊', 'JiangnanWeaponWorkshopProduction')

    def jiangnan_hairpin_workshop_production(self):
        return self._jiangnan_production_facility('花簪制坊', 'JiangnanHairpinWorkshopProduction')

    def jiangnan_copper_mine_production(self):
        return self._jiangnan_production_facility('铜矿', 'JiangnanCopperMineProduction')

    def jiangnan_bamboo_forest_production(self):
        return self._jiangnan_production_facility('竹林', 'JiangnanBambooForestProduction')

    def jiangnan_tea_garden_production(self):
        return self._jiangnan_production_facility('茶园', 'JiangnanTeaGardenProduction')

    def jiangnan_folding_fan_workshop_production(self):
        return self._jiangnan_production_facility('折扇作坊', 'JiangnanFoldingFanWorkshopProduction')

    def jiangnan_round_fan_workshop_production(self):
        return self._jiangnan_production_facility('团扇作坊', 'JiangnanRoundFanWorkshopProduction')

    def jiangnan_crab_pond_production(self):
        return self._jiangnan_production_facility('蟹塘', 'JiangnanCrabPondProduction')

    def jiangnan_salt_field_production(self):
        return self._jiangnan_production_facility('煮盐场', 'JiangnanSaltFieldProduction')

    def jiangnan_standard_cloth_workshop_production(self):
        return self._jiangnan_production_facility('标布作坊', 'JiangnanStandardClothWorkshopProduction')

    def jiangnan_shoe_workshop_production(self):
        return self._jiangnan_production_facility('制鞋坊', 'JiangnanShoeWorkshopProduction')

    def jiangnan_rapeseed_field_production(self):
        return self._jiangnan_production_facility('油菜花田', 'JiangnanRapeseedFieldProduction')

    def jiangnan_pig_farm_production(self):
        return self._jiangnan_production_facility('养猪场', 'JiangnanPigFarmProduction')

    def jiangnan_ink_workshop_production(self):
        return self._jiangnan_production_facility('徽墨作坊', 'JiangnanInkWorkshopProduction')

    def jiangnan_cheng_paper_workshop_production(self):
        return self._jiangnan_production_facility('澄纸作坊', 'JiangnanChengPaperWorkshopProduction')

    def jiangnan_she_inkstone_workshop_production(self):
        return self._jiangnan_production_facility('歙砚作坊', 'JiangnanSheInkstoneWorkshopProduction')

    def jiangnan_wang_brush_workshop_production(self):
        return self._jiangnan_production_facility('汪笔作坊', 'JiangnanWangBrushWorkshopProduction')

    def jiangnan_quarry_production(self):
        return self._jiangnan_production_facility('采石场', 'JiangnanQuarryProduction')

    def jiangnan_copper_mirror_factory_production(self):
        return self._jiangnan_production_facility('铜镜厂', 'JiangnanCopperMirrorFactoryProduction')

    def jiangnan_jade_workshop_production(self):
        return self._jiangnan_production_facility('玉器工坊', 'JiangnanJadeWorkshopProduction')

    def jiangnan_fragrance_powder_factory_production(self):
        return self._jiangnan_production_facility('香粉厂', 'JiangnanFragrancePowderFactoryProduction')

    def jiangnan_wine_yeast_field_production(self):
        return self._jiangnan_production_facility('酒曲田', 'JiangnanWineYeastFieldProduction')

    def jiangnan_yue_porcelain_kiln_production(self):
        return self._jiangnan_production_facility('越瓷窑', 'JiangnanYuePorcelainKilnProduction')

    def jiangnan_sheep_pen_production(self):
        return self._jiangnan_production_facility('羊圈', 'JiangnanSheepPenProduction')

    def jiangnan_black_felt_hat_factory_production(self):
        return self._jiangnan_production_facility('乌毡帽厂', 'JiangnanBlackFeltHatFactoryProduction')

    def jiangnan_fishing_boat_production(self):
        return self._jiangnan_production_facility('渔船', 'JiangnanFishingBoatProduction')

    def jiangnan_dried_fish_workshop_production(self):
        return self._jiangnan_production_facility('鱼干坊', 'JiangnanDriedFishWorkshopProduction')

    def jiangnan_ramie_field_production(self):
        return self._jiangnan_production_facility('苎麻田', 'JiangnanRamieFieldProduction')

    def jiangnan_net_weaving_workshop_production(self):
        return self._jiangnan_production_facility('织网工坊', 'JiangnanNetWeavingWorkshopProduction')

    def jiangnan_specialty_shop_production(self):
        return self._jiangnan_multi_product_facility('特产铺', 'JiangnanSpecialtyShopProduction')

    def jiangnan_carving_workshop_production(self):
        return self._jiangnan_multi_product_facility('雕刻行', 'JiangnanCarvingWorkshopProduction')

    def jiangnan_improved_farm_production(self):
        return self._jiangnan_multi_product_facility('改良农田', 'JiangnanImprovedFarmProduction')

    def jiangnan_printing_factory_production(self):
        return self._jiangnan_multi_product_facility('印刷厂', 'JiangnanPrintingFactoryProduction')

    def jiangnan_amber_winery_production(self):
        return self._jiangnan_multi_product_facility('琥珀酒坊', 'JiangnanAmberWineryProduction')

    def jiangnan_gourmet_restaurant_production(self):
        return self._jiangnan_multi_product_facility('珍馐食肆', 'JiangnanGourmetRestaurantProduction')

    def jiangnan_exquisite_goods_shop_production(self):
        return self._jiangnan_multi_product_facility('琳琅商铺', 'JiangnanExquisiteGoodsShopProduction')

    def jiangnan_embroidery_accessories_workshop_production(self):
        return self._jiangnan_multi_product_facility('锦绣饰坊', 'JiangnanEmbroideryAccessoriesWorkshopProduction')

    def jiangnan_luo_shoes_production(self):
        return self._jiangnan_multi_product_facility('绫罗鞋行', 'JiangnanLuoShoesProduction')

    def jiangnan_jade_crown_hat_workshop_production(self):
        return self._jiangnan_multi_product_facility('玉冠帽坊', 'JiangnanJadeCrownHatWorkshopProduction')

    def jiangnan_zijin_clothing_pavilion_production(self):
        return self._jiangnan_multi_product_facility('子衿衣阁', 'JiangnanZijinClothingPavilionProduction')

    def jiangnan_huizhou_merchant_sale(self):
        from module.jiangnan.huizhou_merchant import JiangnanHuizhouMerchantSale

        return JiangnanHuizhouMerchantSale(
            config=self.config,
            device=self.device,
        ).run()

    def jiangnan_yingtian_post_station(self):
        from module.jiangnan.post_station import JiangnanYingtianPostStation

        return JiangnanYingtianPostStation(config=self.config, device=self.device).run()

    def jiangnan_suzhou_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanSuzhouPostStation', self.config, self.device).run()

    def jiangnan_hangzhou_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanHangzhouPostStation', self.config, self.device).run()

    def jiangnan_songjiang_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanSongjiangPostStation', self.config, self.device).run()

    def jiangnan_huizhou_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanHuizhouPostStation', self.config, self.device).run()

    def jiangnan_yangzhou_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanYangzhouPostStation', self.config, self.device).run()

    def jiangnan_ningbo_post_station(self):
        from module.jiangnan.post_station import station_runner
        return station_runner('JiangnanNingboPostStation', self.config, self.device).run()

    def jiangnan_yingtian_inn(self):
        from module.jiangnan.inn import JiangnanYingtianInn

        return JiangnanYingtianInn(config=self.config, device=self.device).run()

    def jiangnan_shaoxing_furniture_shop(self):
        from module.jiangnan.furniture_shop import JiangnanShaoxingFurnitureShop

        return JiangnanShaoxingFurnitureShop(
            config=self.config, device=self.device,
        ).run()

    def jiangnan_yangzhou_grand_canal(self):
        from module.jiangnan.grand_canal import JiangnanYangzhouGrandCanal

        return JiangnanYangzhouGrandCanal(
            config=self.config,
            device=self.device,
        ).run()

    def jiangnan_ningbo_merchant_ship(self):
        from module.jiangnan.ningbo_merchant_ship import (
            JiangnanNingboMerchantShip,
        )

        return JiangnanNingboMerchantShip(
            config=self.config,
            device=self.device,
        ).run()


if __name__ == '__main__':
    alas = AzurLaneAutoScript()
    alas.loop()
