"""City task identifiers shared by configuration and game automation.

Keep this module free of device, image and OCR imports so configuration
callbacks can use the mapping without loading game modules.
"""


CITY_KEY_TO_PRODUCTION_TASK = {
    'yingtianfu': 'JiangnanYingtianfuProduction',
    'suzhoufu': 'JiangnanSuzhoufuProduction',
    'hangzhoufu': 'JiangnanHangzhoufuProduction',
    'songjiangfu': 'JiangnanSongjiangfuProduction',
    'jimingshan': 'JiangnanJimingshanProduction',
    'huizhoufu': 'JiangnanHuizhoufuProduction',
    'yangzhoufu': 'JiangnanYangzhoufuProduction',
    'shaoxingfu': 'JiangnanShaoxingfuProduction',
    'ningbofu': 'JiangnanNingbofuProduction',
    'jiangningxian': 'JiangnanJiangningxianProduction',
    'baixuezhen': 'JiangnanBaixuezhenProduction',
    'zongyingdao': 'JiangnanZongyingdaoProduction',
    'longgong': 'JiangnanLonggongProduction',
}
