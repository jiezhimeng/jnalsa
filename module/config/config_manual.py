from pywebio.io_ctrl import Output
import module.config.server as server


class ManualConfig:
    @property
    def SERVER(self):
        return server.server

    SCHEDULER_PRIORITY = """
    Restart
    > JiangnanSignin
    > JiangnanTaskReward
    > JiangnanWarehouseInventory
    > JiangnanDockHarvest
    > JiangnanCityAllocation
    > JiangnanTaxHarvest
    > JiangnanChaomuDrum
    > JiangnanHuizhouMerchantSale
    > JiangnanNingboMerchantShip
    > JiangnanYangzhouGrandCanal
    > JiangnanYingtianInn
    > JiangnanShaoxingFurnitureShop
    > JiangnanYingtianPostStation
    > JiangnanSuzhouPostStation
    > JiangnanHangzhouPostStation
    > JiangnanSongjiangPostStation
    > JiangnanHuizhouPostStation
    > JiangnanYangzhouPostStation
    > JiangnanNingboPostStation
    > JiangnanCornDryingMat
    > JiangnanYingtianfuProduction
    > JiangnanSuzhoufuProduction
    > JiangnanHangzhoufuProduction
    > JiangnanSongjiangfuProduction
    > JiangnanJimingshanProduction
    > JiangnanHuizhoufuProduction
    > JiangnanYangzhoufuProduction
    > JiangnanShaoxingfuProduction
    > JiangnanNingbofuProduction
    > JiangnanJiangningxianProduction
    > JiangnanBaixuezhenProduction
    > JiangnanZongyingdaoProduction
    > JiangnanLonggongProduction
    """

    ASSETS_FOLDER = './assets'
    ASSETS_MODULE = './module'
    ASSETS_RESOLUTION = (1280, 720)
    BUTTON_OFFSET = 30
    WAIT_BEFORE_SAVING_SCREEN_SHOT = 1
    DEVICE_OVER_HTTP = False
    FORWARD_PORT_RANGE = (20000, 21000)
    REVERSE_SERVER_PORT = 7903
    ASCREENCAP_FILEPATH_LOCAL = './bin/ascreencap'
    ASCREENCAP_FILEPATH_REMOTE = '/data/local/tmp/ascreencap'
    DROIDCAST_VERSION = 'DroidCast'
    DROIDCAST_FILEPATH_LOCAL = './bin/DroidCast/DroidCast_raw-release-1.0.apk'
    DROIDCAST_FILEPATH_REMOTE = '/data/local/tmp/DroidCast_raw.apk'
    MINITOUCH_FILEPATH_REMOTE = '/data/local/tmp/minitouch'
    HERMIT_FILEPATH_LOCAL = './bin/hermit/hermit.apk'
    SCRCPY_FILEPATH_LOCAL = './bin/scrcpy/scrcpy-server-v1.20.jar'
    SCRCPY_FILEPATH_REMOTE = '/data/local/tmp/scrcpy-server-v1.20.jar'
    MAATOUCH_FILEPATH_LOCAL = './bin/MaaTouch/maatouchsync'
    MAATOUCH_FILEPATH_REMOTE = '/data/local/tmp/maatouchsync'
    COMMON_CV_THRESHOLD = 0.9


class OutputConfig(Output, ManualConfig):
    pass
