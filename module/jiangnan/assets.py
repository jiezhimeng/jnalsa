from module.base.button import Button


MAIN_BUILD_CHECK = Button(
    area=(850, 584, 899, 626),
    color=(),
    button=(850, 584, 899, 626),
    file='./assets/jiangnan/ui/main.png',
    name='MAIN_BUILD_CHECK',
)


MAIN_CHAT_CHECK = Button(
    area=(18, 258, 70, 299),
    color=(),
    button=(18, 258, 70, 299),
    file='./assets/jiangnan/ui/main.png',
    name='MAIN_CHAT_CHECK',
)

SIGNIN_CUMULATIVE_CHECK = Button(
    area=(368, 500, 420, 614),
    color=(),
    button=(368, 500, 420, 614),
    file='./assets/jiangnan/popup/signin.png',
    name='SIGNIN_CUMULATIVE_CHECK',
)


SIGNIN_HINT_CHECK = Button(
    area=(525, 630, 825, 670),
    color=(),
    button=(525, 630, 825, 670),
    file='./assets/jiangnan/popup/signin.png',
    name='SIGNIN_HINT_CHECK',
)

SIGNIN_CLAIM_BUTTON = Button(
    area=(525, 630, 825, 670),
    color=(),
    button=(525, 630, 825, 670),
    file='./assets/jiangnan/popup/signin.png',
    name='SIGNIN_CLAIM_BUTTON',
)

REWARD_TAKE_BUTTON = Button(
    area=(510, 525, 770, 590),
    color=(),
    button=(510, 525, 770, 590),
    file='./assets/jiangnan/popup/reward.png',
    name='REWARD_TAKE_BUTTON',
)

# “游戏数据同步失败”网络弹窗。英文错误详情可能是 Request timeout、
# Unable to complete SSL connection 等，只匹配固定的中文提示和重试按钮。
NETWORK_TIMEOUT_CHECK = Button(
    area=(386, 276, 875, 360),
    color=(),
    button=(386, 276, 875, 416),
    file='./assets/jiangnan/popup/network_timeout.png',
    name='NETWORK_TIMEOUT_CHECK',
)

NETWORK_RETRY_BUTTON = Button(
    area=(546, 477, 751, 539),
    color=(),
    button=(546, 477, 751, 539),
    file='./assets/jiangnan/popup/network_timeout.png',
    name='NETWORK_RETRY_BUTTON',
)

# ============================================================
# Huizhou merchant / council hall
# ============================================================

COUNCIL_HALL_MAP_ICON = Button(
    area=(773, 382, 840, 454), color=(),
    button=(786, 398, 826, 444),
    file='./assets/jiangnan/huizhou_merchant/hall_map.png',
    name='COUNCIL_HALL_MAP_ICON',
)
COUNCIL_HALL_PANEL_CHECK = Button(
    area=(205, 75, 305, 322), color=(), button=(205, 75, 305, 322),
    file='./assets/jiangnan/huizhou_merchant/hall_idle.png',
    name='COUNCIL_HALL_PANEL_CHECK',
)
COUNCIL_HALL_BUSINESS_BUTTON = Button(
    area=(202, 589, 310, 658), color=(), button=(202, 589, 310, 658),
    file='./assets/jiangnan/huizhou_merchant/hall_idle.png',
    name='COUNCIL_HALL_BUSINESS_BUTTON',
)
COUNCIL_HALL_INFO_BUTTON = Button(
    area=(202, 529, 310, 588), color=(), button=(202, 529, 310, 588),
    file='./assets/jiangnan/huizhou_merchant/working.png',
    name='COUNCIL_HALL_INFO_BUTTON',
)
MERCHANT_SALE_PANEL_CHECK = Button(
    area=(348, 48, 570, 104), color=(), button=(348, 48, 570, 104),
    file='./assets/jiangnan/huizhou_merchant/sale_left.png',
    name='MERCHANT_SALE_PANEL_CHECK',
)
MERCHANT_ASSIGNEE_CHECK = Button(
    area=(205, 75, 305, 322), color=(), button=(205, 75, 305, 322),
    file='./assets/jiangnan/huizhou_merchant/assignee.png',
    name='MERCHANT_ASSIGNEE_CHECK',
)
NINGBO_SHIP_ASSIGNEE_CONFIRM = Button(
    area=(972, 601, 1083, 665), color=(), button=(989, 612, 1068, 651),
    file='./assets/jiangnan/ningbo_ship/assignee_detail.jpg',
    name='NINGBO_SHIP_ASSIGNEE_CONFIRM',
)
MERCHANT_WORKING_CHECK = Button(
    # “外出经商中”是经营中页面独有的稳定文字。旧区域 y=286..332
    # 落在商人头像和姓名上，空闲页面也有相同结构，会把空闲议事堂
    # 误判为经营中，继而在本来为空的倒计时区域反复 OCR。
    area=(338, 348, 480, 394), color=(), button=(338, 348, 480, 394),
    file='./assets/jiangnan/huizhou_merchant/working.png',
    name='MERCHANT_WORKING_CHECK',
)
MERCHANT_DEPART_BUTTON = Button(
    area=(943, 613, 1082, 676), color=(), button=(943, 613, 1082, 676),
    file='./assets/jiangnan/huizhou_merchant/sale_left.png',
    name='MERCHANT_DEPART_BUTTON',
)

# ============================================================
# Yangzhou Grand Canal
# ============================================================

GRAND_CANAL_DOCK_PANEL_CHECK = Button(
    area=(205, 70, 330, 340), color=(), button=(205, 70, 330, 340),
    file='./assets/jiangnan/grand_canal/dock_panel.png',
    name='GRAND_CANAL_DOCK_PANEL_CHECK',
)
GRAND_CANAL_ENTER_BUTTON = Button(
    area=(568, 500, 860, 570), color=(), button=(578, 508, 850, 560),
    file='./assets/jiangnan/grand_canal/dock_panel.png',
    name='GRAND_CANAL_ENTER_BUTTON',
)
GRAND_CANAL_PAGE_CHECK = Button(
    area=(17, 360, 85, 455), color=(), button=(24, 370, 78, 446),
    file='./assets/jiangnan/grand_canal/canal_map.png',
    name='GRAND_CANAL_PAGE_CHECK',
)
GRAND_CANAL_LOCATE_BUTTON = Button(
    area=(17, 360, 85, 455), color=(), button=(24, 370, 78, 446),
    file='./assets/jiangnan/grand_canal/canal_map.png',
    name='GRAND_CANAL_LOCATE_BUTTON',
)
GRAND_CANAL_EXIT_BUTTON = Button(
    area=(1184, 574, 1265, 705), color=(), button=(1192, 585, 1256, 696),
    file='./assets/jiangnan/grand_canal/canal_map.png',
    name='GRAND_CANAL_EXIT_BUTTON',
)
GRAND_CANAL_CENTER_SHIP = Button(
    area=(575, 285, 705, 405), color=(), button=(600, 300, 685, 395),
    file='./assets/jiangnan/grand_canal/canal_map.png',
    name='GRAND_CANAL_CENTER_SHIP',
)
GRAND_CANAL_MOVE_DIALOG_CHECK = Button(
    # 目的地标题会是“驶向该处”或“漕运码头”等，固定的船票图标和
    # “-1 出发”按钮才能确认这是可出发弹窗。
    area=(545, 475, 705, 510), color=(), button=(545, 475, 705, 510),
    file='./assets/jiangnan/grand_canal/move_dialog.png',
    name='GRAND_CANAL_MOVE_DIALOG_CHECK',
)
GRAND_CANAL_SAILING_DIALOG_CHECK = Button(
    area=(555, 164, 725, 214), color=(), button=(555, 164, 725, 214),
    file='./assets/jiangnan/grand_canal/sailing_dialog.png',
    name='GRAND_CANAL_SAILING_DIALOG_CHECK',
)
GRAND_CANAL_SAILING_OK = Button(
    area=(545, 479, 750, 538), color=(), button=(555, 485, 740, 533),
    file='./assets/jiangnan/grand_canal/sailing_dialog.png',
    name='GRAND_CANAL_SAILING_OK',
)
GRAND_CANAL_CURRENT_LOCATION_CHECK = Button(
    area=(390, 290, 885, 375), color=(), button=(390, 290, 885, 375),
    file='./assets/jiangnan/grand_canal/current_location.jpg',
    name='GRAND_CANAL_CURRENT_LOCATION_CHECK',
)
GRAND_CANAL_CURRENT_LOCATION_OK = Button(
    area=(525, 468, 755, 515), color=(), button=(530, 474, 750, 510),
    file='./assets/jiangnan/grand_canal/current_location.jpg',
    name='GRAND_CANAL_CURRENT_LOCATION_OK',
)
GRAND_CANAL_EVENT_CHECK = Button(
    area=(290, 20, 455, 115), color=(), button=(290, 20, 455, 115),
    file='./assets/jiangnan/grand_canal/event_dialog.png',
    name='GRAND_CANAL_EVENT_CHECK',
)
GRAND_CANAL_BIG_ORDER_CHECK = Button(
    area=(290, 20, 455, 115), color=(), button=(290, 20, 455, 115),
    file='./assets/jiangnan/grand_canal/big_order_dialog.png',
    name='GRAND_CANAL_BIG_ORDER_CHECK',
)
GRAND_CANAL_ASSIGNEE_CHECK = Button(
    area=(205, 75, 305, 322), color=(), button=(205, 75, 305, 322),
    file='./assets/jiangnan/grand_canal/assignee.png',
    name='GRAND_CANAL_ASSIGNEE_CHECK',
)
GRAND_CANAL_WORKING_CHECK = Button(
    area=(1180, 158, 1275, 258), color=(), button=(1180, 158, 1275, 258),
    file='./assets/jiangnan/grand_canal/working.png',
    name='GRAND_CANAL_WORKING_CHECK',
)
GRAND_CANAL_TASK_CARD_CHECK = Button(
    area=(1185, 160, 1200, 180), color=(), button=(1185, 160, 1200, 180),
    file='./assets/jiangnan/grand_canal/working.png',
    name='GRAND_CANAL_TASK_CARD_CHECK',
)
GRAND_CANAL_SECOND_TASK_CARD_CHECK = Button(
    area=(1185, 278, 1200, 298), color=(), button=(1185, 278, 1200, 298),
    file='./assets/jiangnan/grand_canal/working.png',
    name='GRAND_CANAL_SECOND_TASK_CARD_CHECK',
)
GRAND_CANAL_REWARD_CHECK = Button(
    area=(465, 600, 815, 680), color=(), button=(480, 610, 800, 670),
    file='./assets/jiangnan/grand_canal/reward.png',
    name='GRAND_CANAL_REWARD_CHECK',
)
GRAND_CANAL_STEPS_COUNT = Button(
    area=(65, 574, 106, 611), color=(), button=(65, 574, 106, 611),
    name='GRAND_CANAL_STEPS_COUNT',
)
GRAND_CANAL_SINAN_COUNT = Button(
    area=(75, 650, 142, 694), color=(), button=(75, 650, 142, 694),
    name='GRAND_CANAL_SINAN_COUNT',
)
GRAND_CANAL_SINAN_BUTTON = Button(
    area=(), color=(), button=(20, 630, 140, 703),
    name='GRAND_CANAL_SINAN_BUTTON',
)
GRAND_CANAL_COMPASS_CHECK = Button(
    area=(463, 183, 550, 269), color=(), button=(463, 183, 550, 269),
    file='./assets/jiangnan/grand_canal/compass_panel.png',
    name='GRAND_CANAL_COMPASS_CHECK',
)
GRAND_CANAL_COMPASS_STEPS_COUNT = Button(
    area=(627, 72, 680, 112), color=(), button=(627, 72, 680, 112),
    name='GRAND_CANAL_COMPASS_STEPS_COUNT',
)
GRAND_CANAL_COMPASS_SPIN = Button(
    area=(), color=(), button=(608, 320, 673, 410),
    name='GRAND_CANAL_COMPASS_SPIN',
)
GRAND_CANAL_TASK_BUTTON = Button(
    area=(), color=(), button=(143, 626, 191, 702),
    name='GRAND_CANAL_TASK_BUTTON',
)
GRAND_CANAL_SINAN_TASK_CHECK = Button(
    area=(205, 75, 311, 395), color=(), button=(205, 75, 311, 395),
    file='./assets/jiangnan/grand_canal/sinan_task_ready.png',
    name='GRAND_CANAL_SINAN_TASK_CHECK',
)
GRAND_CANAL_SINAN_CLAIM = Button(
    area=(1002, 593, 1069, 661), color=(), button=(1008, 599, 1065, 655),
    file='./assets/jiangnan/grand_canal/sinan_task_ready.png',
    name='GRAND_CANAL_SINAN_CLAIM',
)
GRAND_CANAL_SINAN_CLAIM_DONE = Button(
    area=(1002, 593, 1069, 661), color=(), button=(1008, 599, 1065, 655),
    file='./assets/jiangnan/grand_canal/sinan_task_done.png',
    name='GRAND_CANAL_SINAN_CLAIM_DONE',
)
GRAND_CANAL_SINAN_REWARD = Button(
    area=(496, 526, 780, 591), color=(), button=(505, 532, 770, 585),
    file='./assets/jiangnan/grand_canal/sinan_reward.png',
    name='GRAND_CANAL_SINAN_REWARD',
)
GRAND_CANAL_SINAN_OVERFLOW_CHECK = Button(
    area=(389, 299, 873, 385), color=(), button=(389, 299, 873, 385),
    file='./assets/jiangnan/grand_canal/sinan_overflow.png',
    name='GRAND_CANAL_SINAN_OVERFLOW_CHECK',
)
GRAND_CANAL_RESOURCE_SHORTAGE_CHECK = Button(
    area=(204, 72, 313, 365), color=(), button=(204, 72, 313, 365),
    file='./assets/jiangnan/grand_canal/resource_shortage.png',
    name='GRAND_CANAL_RESOURCE_SHORTAGE_CHECK',
)

# ============================================================
# Jiangnan daily signin
# ============================================================

# 主界面左上角等级区域
# 只用于点击，不用于判断等级，因为等级数字会变化
MAIN_LEVEL_BUTTON = Button(
    area=(10, 16, 74, 248),
    color=(),
    button=(10, 16, 74, 248),
    file='./assets/jiangnan/ui/main.png',
    name='MAIN_LEVEL_BUTTON',
)


# 玩家信息弹窗：“玩家经验”
PROFILE_CHECK = Button(
    area=(93, 17, 205, 52),
    color=(),
    button=(93, 17, 205, 52),
    file='./assets/jiangnan/signin/profile.png',
    name='PROFILE_CHECK',
)


# 玩家信息弹窗：“签到奖励”
# 特意不包含右边红点，因为红点会消失
PROFILE_SIGNIN_BUTTON = Button(
    area=(102, 181, 300, 223),
    color=(),
    button=(102, 181, 300, 223),
    file='./assets/jiangnan/signin/profile.png',
    name='PROFILE_SIGNIN_BUTTON',
)


# 签到页面左侧竖排“签到奖励”
DAILY_SIGNIN_TITLE_CHECK = Button(
    area=(210, 74, 310, 382),
    color=(),
    button=(210, 74, 310, 382),
    file='./assets/jiangnan/signin/daily.png',
    name='DAILY_SIGNIN_TITLE_CHECK',
)


# 签到页面左下角“日常”
DAILY_SIGNIN_DAILY_CHECK = Button(
    area=(198, 519, 308, 576),
    color=(),
    button=(198, 519, 308, 576),
    file='./assets/jiangnan/signin/daily.png',
    name='DAILY_SIGNIN_DAILY_CHECK',
)


# 右下“签到”
DAILY_SIGNIN_BUTTON = Button(
    area=(806, 548, 1062, 607),
    color=(),
    button=(806, 548, 1062, 607),
    file='./assets/jiangnan/signin/daily.png',
    name='DAILY_SIGNIN_BUTTON',
)

SIGNIN_DAILY_TAB_BUTTON = Button(
    area=(198, 521, 306, 576),
    color=(),
    button=(198, 521, 306, 576),
    file='./assets/jiangnan/signin/limited.png',
    name='SIGNIN_DAILY_TAB_BUTTON',
)

DAILY_SIGNIN_DONE_CHECK = Button(
    area=(806, 548, 1062, 607),
    color=(),
    button=(806, 548, 1062, 607),
    file='./assets/jiangnan/signin/daily_done.png',
    name='DAILY_SIGNIN_DONE_CHECK',
)

MAIN_TASK_CHECK = Button(
    area=(940, 584, 989, 626),
    color=(),
    button=(940, 584, 989, 626),
    file='./assets/jiangnan/ui/main.png',
    name='MAIN_TASK_CHECK',
)


# ============================================================
# Jiangnan city switch
# ============================================================

# ------------------------------------------------------------
# 主界面左下当前城市印
#
# 每个城市文字不同，例如：
# 龙宫之印
# 应天府印
# ...
#
# 所以这里只作为点击区域，不进行模板识别。
# ------------------------------------------------------------

CITY_SEAL_BUTTON = Button(
    area=(),
    color=(),
    button=(10, 492, 138, 708),
    name='CITY_SEAL_BUTTON',
)


# ------------------------------------------------------------
# 点击城市印后的菜单：“大地图”
# ------------------------------------------------------------

CITY_MAP_BUTTON = Button(
    area=(126, 258, 190, 387),
    color=(),
    button=(126, 258, 190, 387),
    file='./assets/jiangnan/city/city_panel.png',
    name='CITY_MAP_BUTTON',
)


# ------------------------------------------------------------
# 大地图左下：“城镇总览”
# ------------------------------------------------------------

CITY_OVERVIEW_BUTTON = Button(
    area=(21, 548, 168, 710),
    color=(),
    button=(21, 548, 168, 710),
    file='./assets/jiangnan/city/world_map.png',
    name='CITY_OVERVIEW_BUTTON',
)


# ------------------------------------------------------------
# 城镇列表标题：“城镇与运输”
# ------------------------------------------------------------

CITY_LIST_CHECK = Button(
    area=(207, 64, 494, 118),
    color=(),
    button=(207, 64, 494, 118),
    file='./assets/jiangnan/city/city_list.png',
    name='CITY_LIST_CHECK',
)


# ------------------------------------------------------------
# 类型：“全部”
#
# 不判断当前是什么类型。
# 只要进入列表，就主动点击一次“全部”。
# ------------------------------------------------------------

CITY_TYPE_ALL_BUTTON = Button(
    area=(),
    color=(),
    button=(802, 72, 890, 114),
    name='CITY_TYPE_ALL_BUTTON',
)


# ------------------------------------------------------------
# 城镇列表中最多同时可见 8 个位置。
#
# area   = OCR 城市名称的位置
# button = 点击整个城市卡片的位置
# ------------------------------------------------------------

CITY_LIST_SLOTS = [

    # ========================================================
    # 第1行
    # ========================================================

    Button(
        area=(207, 247, 300, 280),
        color=(),
        button=(207, 170, 300, 286),
        name='CITY_SLOT_1',
    ),

    Button(
        area=(657, 247, 750, 280),
        color=(),
        button=(657, 170, 750, 286),
        name='CITY_SLOT_2',
    ),

    # ========================================================
    # 第2行
    # ========================================================

    Button(
        area=(207, 378, 300, 412),
        color=(),
        button=(207, 302, 300, 416),
        name='CITY_SLOT_3',
    ),

    Button(
        area=(657, 378, 750, 412),
        color=(),
        button=(657, 302, 750, 416),
        name='CITY_SLOT_4',
    ),

    # ========================================================
    # 第3行
    # ========================================================

    Button(
        area=(207, 508, 300, 543),
        color=(),
        button=(207, 432, 300, 546),
        name='CITY_SLOT_5',
    ),

    Button(
        area=(657, 508, 750, 543),
        color=(),
        button=(657, 432, 750, 546),
        name='CITY_SLOT_6',
    ),

    # ========================================================
    # 第4行
    # ========================================================

    Button(
        area=(207, 638, 300, 670),
        color=(),
        button=(207, 562, 300, 677),
        name='CITY_SLOT_7',
    ),

    Button(
        area=(657, 638, 750, 670),
        color=(),
        button=(657, 562, 750, 677),
        name='CITY_SLOT_8',
    ),
]


# ============================================================
# 税课司
# ============================================================

TAX_POPUP_CHECK = Button(
    area=(208, 68, 324, 295),
    color=(),
    button=(208, 68, 324, 295),
    file='./assets/jiangnan/city/tax_popup.png',
    name='TAX_POPUP_CHECK',
)


TAX_KNOW_BUTTON = Button(
    area=(493, 570, 784, 650),
    color=(),
    button=(493, 570, 784, 650),
    file='./assets/jiangnan/city/tax_popup.png',
    name='TAX_KNOW_BUTTON',
)

CITY_PANEL_CHECK = Button(
    area=(100, 165, 205, 230),
    color=(),
    button=(100, 165, 205, 230),
    file='./assets/jiangnan/city/city_panel.png',
    name='CITY_PANEL_CHECK',
)

# ============================================================
# Jiangnan Warehouse
# ============================================================

# 主界面右下“营造”
# 只负责点击，不用于模板识别
MAIN_BUILD_BUTTON = Button(
    area=(),
    color=(),
    button=(833, 574, 922, 707),
    name='MAIN_BUILD_BUTTON',
)


# 营造界面左侧大字“营造”
BUILD_TITLE_CHECK = Button(
    area=(204, 64, 312, 224),
    color=(),
    button=(204, 64, 312, 224),
    file='./assets/jiangnan/warehouse/build.png',
    name='BUILD_TITLE_CHECK',
)


# 营造界面右上“查看仓库”
BUILD_WAREHOUSE_BUTTON = Button(
    area=(1088, 124, 1267, 170),
    color=(),
    button=(1088, 124, 1267, 170),
    file='./assets/jiangnan/warehouse/build.png',
    name='BUILD_WAREHOUSE_BUTTON',
)


# 仓库界面左侧大字“仓库”
WAREHOUSE_TITLE_CHECK = Button(
    area=(204, 65, 312, 225),
    color=(),
    button=(204, 65, 312, 225),
    file='./assets/jiangnan/warehouse/warehouse.png',
    name='WAREHOUSE_TITLE_CHECK',
)


# ============================================================
# 仓库最底部两个“徽州特产”复合格
#
# 这里名字是固定的，只 OCR 数量。
# 只有确认已经滚动到底以后才会读取这些区域。
# ============================================================

WAREHOUSE_SPECIAL_COUNT_SLOTS = {

    # ========================================
    # 右侧：徽州特产
    # 统一宽度 70 px
    # ========================================

    '茶油': Button(
        area=(716, 434, 786, 453),
        color=(),
        button=(716, 434, 786, 453),
        name='WAREHOUSE_SPECIAL_COUNT_TEA_OIL'
    ),

    '竹编': Button(
        area=(790, 434, 860, 453),
        color=(),
        button=(790, 434, 860, 453),
        name='WAREHOUSE_SPECIAL_COUNT_BAMBOO'
    ),

    '甲酒': Button(
        area=(864, 434, 934, 453),
        color=(),
        button=(864, 434, 934, 453),
        name='WAREHOUSE_SPECIAL_COUNT_WINE'
    ),

    '盆景': Button(
        area=(938, 434, 1008, 453),
        color=(),
        button=(938, 434, 1008, 453),
        name='WAREHOUSE_SPECIAL_COUNT_BONSAI'
    ),


    # ========================================
    # 左侧：徽州特产
    # 统一宽度 70 px
    # ========================================

    '木雕': Button(
        area=(334, 555, 404, 575),
        color=(),
        button=(334, 555, 404, 575),
        name='WAREHOUSE_SPECIAL_COUNT_WOOD'
    ),

    '竹雕': Button(
        area=(408, 555, 478, 575),
        color=(),
        button=(408, 555, 478, 575),
        name='WAREHOUSE_SPECIAL_COUNT_BAMBOO_CARVING'
    ),

    '砖雕': Button(
        area=(482, 555, 552, 575),
        color=(),
        button=(482, 555, 552, 575),
        name='WAREHOUSE_SPECIAL_COUNT_BRICK'
    ),

    '石雕': Button(
        area=(556, 555, 626, 575),
        color=(),
        button=(556, 555, 626, 575),
        name='WAREHOUSE_SPECIAL_COUNT_STONE'
    ),
}

# 徽州特产不是八个独立仓位，而是每四件共用一个上限。
# 数字位位于每组最后一个物品右侧的“/上限”区域。
WAREHOUSE_SPECIAL_CAPACITY_SLOTS = {
    '徽州特产1': Button(
        area=(1008, 414, 1090, 454),
        color=(),
        button=(1008, 414, 1090, 454),
        name='WAREHOUSE_SPECIAL_CAPACITY_GROUP_1',
    ),
    '徽州特产2': Button(
        area=(632, 536, 710, 576),
        color=(),
        button=(632, 536, 710, 576),
        name='WAREHOUSE_SPECIAL_CAPACITY_GROUP_2',
    ),
}


# ============================================================
# Jiangnan material allocation
# ============================================================

# “城镇与运输”页面标题。
ALLOCATION_OVERVIEW_CHECK = Button(
    area=(207, 64, 494, 118),
    color=(),
    button=(207, 64, 494, 118),
    file='./assets/jiangnan/allocation/overview.png',
    name='ALLOCATION_OVERVIEW_CHECK',
)

# 选择目的地页面左侧竖排标题。
ALLOCATION_DESTINATION_CHECK = Button(
    area=(208, 72, 315, 450),
    color=(),
    button=(208, 72, 315, 450),
    file='./assets/jiangnan/allocation/destination.png',
    name='ALLOCATION_DESTINATION_CHECK',
)

# 运输物资页面左侧竖排标题。
ALLOCATION_RESOURCE_CHECK = Button(
    area=(207, 75, 318, 333),
    color=(),
    button=(207, 75, 318, 333),
    file='./assets/jiangnan/allocation/resource.png',
    name='ALLOCATION_RESOURCE_CHECK',
)

# 左侧已选择物资的数量。点击后会打开 Android 数字输入框。
ALLOCATION_AMOUNT_BUTTON = Button(
    area=(),
    color=(),
    button=(459, 218, 544, 261),
    name='ALLOCATION_AMOUNT_BUTTON',
)

# 数量有效时出现的红色“出发”按钮。
ALLOCATION_DEPART_BUTTON = Button(
    area=(921, 592, 1068, 647),
    color=(),
    button=(921, 592, 1068, 647),
    file='./assets/jiangnan/allocation/resource.png',
    name='ALLOCATION_DEPART_BUTTON',
)

# 负责人页面左侧竖排标题。
ALLOCATION_ASSIGNEE_CHECK = Button(
    area=(207, 115, 320, 486),
    color=(),
    button=(207, 115, 320, 486),
    file='./assets/jiangnan/allocation/assignee.png',
    name='ALLOCATION_ASSIGNEE_CHECK',
)

# 不派遣负责人，直接开始运输。
ALLOCATION_DIRECT_START_BUTTON = Button(
    # 只用左侧带斜杠的图标做模板，人物姓名和说明文字会随语言/字体
    # 变化；真正的点击框则放在右侧“直接开始”文字卡片。
    area=(338, 105, 452, 219),
    color=(),
    button=(485, 110, 680, 215),
    file='./assets/jiangnan/allocation/assignee.png',
    name='ALLOCATION_DIRECT_START_BUTTON',
)


# ============================================================
# Jiangnan building search
# ============================================================

# 城市印章面板中的“地基”。选择后按钮底色会由红色变成白色。
FOUNDATION_BUTTON = Button(
    area=(291, 253, 368, 389),
    color=(),
    # Click inside the visible 地基 tile, away from its right edge. The old
    # 338..371 box landed at x=357..360 outside the tile in 江宁县.
    button=(302, 277, 338, 365),
    name='FOUNDATION_BUTTON',
)

FOUNDATION_SELECTED_CHECK = Button(
    area=(291, 253, 368, 389),
    color=(),
    button=(291, 253, 368, 389),
    file='./assets/jiangnan/building/foundation.png',
    name='FOUNDATION_SELECTED_CHECK',
)

# 建筑搜索图标模板来源。不同城市可能额外显示“调板”等按钮，导致搜索
# 图标上下移动；运行时只使用这里的图像模板，不使用这里的固定点击坐标。
BUILDING_SEARCH_TOOL_BUTTON = Button(
    area=(13, 310, 74, 370),
    color=(),
    button=(13, 310, 74, 370),
    file='./assets/jiangnan/building/foundation.png',
    name='BUILDING_SEARCH_TOOL_BUTTON',
)

BUILDING_SEARCH_PAGE_CHECK = Button(
    area=(207, 65, 318, 255),
    color=(),
    button=(207, 65, 318, 255),
    file='./assets/jiangnan/building/search.png',
    name='BUILDING_SEARCH_PAGE_CHECK',
)

BUILDING_SEARCH_INPUT = Button(
    area=(),
    color=(),
    button=(654, 69, 973, 111),
    name='BUILDING_SEARCH_INPUT',
)

# 搜索框已有内容时出现的红色叉号，用于避免上一次搜索内容被继续追加。
BUILDING_SEARCH_CLEAR_BUTTON = Button(
    area=(937, 72, 970, 107),
    color=(),
    button=(937, 72, 970, 107),
    file='./assets/jiangnan/building/result.png',
    name='BUILDING_SEARCH_CLEAR_BUTTON',
)

BUILDING_SEARCH_BUTTON = Button(
    area=(990, 68, 1088, 114),
    color=(),
    button=(990, 68, 1088, 114),
    file='./assets/jiangnan/building/search.png',
    name='BUILDING_SEARCH_BUTTON',
)

BUILDING_FIRST_RESULT = Button(
    area=(),
    color=(),
    # 搜索结果的名称条仅用于显示，不响应点击。点击建筑图片中心的小范围，
    # 避免 ALAS 的随机落点落入卡片空白处。
    button=(390, 210, 410, 230),
    name='BUILDING_FIRST_RESULT',
)

# 搜索建筑后偶尔会有居民事件标记覆盖在建筑附近。标记位置随地图
# 变化，因此这里只保存特化图像模板，运行时在可视地图内动态查找。
BUILDING_INTERFERENCE_KNOT = Button(
    area=(600, 315, 644, 365),
    color=(),
    button=(600, 315, 644, 365),
    file='./assets/jiangnan/building/interference_knot.png',
    name='BUILDING_INTERFERENCE_KNOT',
)

BUILDING_INTERFERENCE_SCISSORS = Button(
    area=(643, 310, 691, 361),
    color=(),
    button=(643, 310, 691, 361),
    file='./assets/jiangnan/building/interference_scissors.png',
    name='BUILDING_INTERFERENCE_SCISSORS',
)

# 点击上述标记后打开的民宅居民面板。只取左侧固定“民宅”标题，
# 不包含居民头像、姓名、倒计时或会消失的红点。
RESIDENCE_PANEL_CHECK = Button(
    area=(202, 70, 330, 220),
    color=(),
    button=(202, 70, 330, 220),
    file='./assets/jiangnan/building/residence_panel.png',
    name='RESIDENCE_PANEL_CHECK',
)

FARM_IDLE_CHECK = Button(
    area=(204, 65, 330, 220),
    color=(),
    button=(204, 65, 330, 220),
    file='./assets/jiangnan/farm/idle.png',
    name='FARM_IDLE_CHECK',
)

FARM_WORKING_CHECK = Button(
    area=(500, 132, 775, 198),
    color=(),
    button=(500, 132, 775, 198),
    file='./assets/jiangnan/farm/working.png',
    name='FARM_WORKING_CHECK',
)

# 通用生产建筑详情页的“介绍”按钮。建筑名称不同，但这个按钮及其
# 相对位置一致，可用于确认空闲生产面板。
PRODUCTION_RESIDENT_DETAIL_CHECK = Button(
    area=(791, 55, 1079, 187), color=(), button=(791, 55, 1079, 187),
    file='./assets/jiangnan/farm/resident_detail.png',
    name='PRODUCTION_RESIDENT_DETAIL_CHECK',
)

PRODUCTION_IDLE_CHECK = Button(
    area=(202, 515, 308, 565),
    color=(),
    button=(202, 515, 308, 565),
    file='./assets/jiangnan/farm/idle.png',
    name='PRODUCTION_IDLE_CHECK',
)

# 通用生产加速弹窗的“立即完成”按钮，标题会随建筑变化，因此不取标题。
PRODUCTION_WORKING_CHECK = Button(
    area=(385, 385, 535, 450),
    color=(),
    button=(385, 385, 535, 450),
    file='./assets/jiangnan/farm/working.png',
    name='PRODUCTION_WORKING_CHECK',
)


# ============================================================
# Jiangnan harvest
# ============================================================

WAREHOUSE_FULL_CHECK = Button(
    area=(724, 283, 781, 427),
    color=(),
    button=(724, 283, 781, 427),
    file='./assets/jiangnan/harvest/full.png',
    name='WAREHOUSE_FULL_CHECK',
)

FORCE_HARVEST_BUTTON = Button(
    area=(733, 444, 805, 519),
    color=(),
    button=(733, 444, 805, 519),
    file='./assets/jiangnan/harvest/full.png',
    name='FORCE_HARVEST_BUTTON',
)

DISCARD_PROMPT_CHECK = Button(
    area=(584, 166, 699, 225),
    color=(),
    button=(340, 91, 961, 628),
    file='./assets/jiangnan/harvest/discard.png',
    name='DISCARD_PROMPT_CHECK',
)

HARVEST_CANCEL_BUTTON = Button(
    area=(),
    color=(),
    button=(408, 478, 612, 539),
    name='HARVEST_CANCEL_BUTTON',
)

HARVEST_DISCARD_BUTTON = Button(
    area=(),
    color=(),
    button=(670, 478, 875, 539),
    name='HARVEST_DISCARD_BUTTON',
)


# ============================================================
# Jiangnan transport completion
# ============================================================

TRANSPORT_ARRIVAL_PROMPT_CHECK = Button(
    area=(584, 166, 699, 225),
    color=(),
    button=(312, 86, 971, 633),
    file='./assets/jiangnan/allocation/arrival_prompt.png',
    name='TRANSPORT_ARRIVAL_PROMPT_CHECK',
)

TRANSPORT_GO_BUTTON = Button(
    area=(),
    color=(),
    button=(669, 478, 875, 539),
    name='TRANSPORT_GO_BUTTON',
)

TRANSPORT_GO_LABEL_CHECK = Button(
    area=(734, 490, 809, 523),
    color=(),
    button=(734, 490, 809, 523),
    file='./assets/jiangnan/allocation/arrival_prompt.png',
    name='TRANSPORT_GO_LABEL_CHECK',
)

BUILDING_SEARCH_NO_RESULT = Button(
    # 这是提示文字在参考截图中的区域
    area=(375, 300, 910, 350),
    color=(),
    button=(375, 300, 910, 350),
    file='./assets/jiangnan/building/building_search_no_result.png',
    name='BUILDING_SEARCH_NO_RESULT',
)


# ============================================================
# Jiangnan tax office
# ============================================================
# 所有坐标基于 1280x720 完整截图。
# area 用于识别，button 用于点击。
# 不需要额外保存裁切后的模板图片。

# 图1：地图中的屋顶图标。
TAX_MAP_ICON = Button(
    area=(616, 367, 666, 415),
    color=(),
    button=(635, 386, 645, 396),
    file='./assets/jiangnan/building/tax_office_map.png',
    name='TAX_MAP_ICON',
)

# 图3：空闲税课司左侧的竖排标题。
TAX_IDLE_CHECK = Button(
    area=(215, 67, 305, 290),
    color=(),
    button=(215, 67, 305, 290),
    file='./assets/jiangnan/building/tax_office_idle.png',
    name='TAX_IDLE_CHECK',
)

# 图2：工作中界面的固定说明文字。
# 不使用会变化的剩余时间。
TAX_WORKING_CHECK = Button(
    area=(495, 420, 790, 452),
    color=(),
    button=(495, 420, 790, 452),
    file='./assets/jiangnan/building/tax_office_working.png',
    name='TAX_WORKING_CHECK',
)

TAX_REMAINING_TIME = Button(
    # 给首尾数字留出额外边距；原区域左侧只有约 9px，面板发生
    # 轻微锚点偏移时会把小时数字裁掉。
    area=(660, 245, 900, 302),
    color=(),
    button=(660, 245, 900, 302),
    file='./assets/jiangnan/building/tax_office_working.png',
    name='TAX_REMAINING_TIME',
)

# 图3：只取一把锁作为模板。
# 后续在二、三、四级各自的区域中匹配同一把锁。
TAX_LEVEL_LOCK_CHECK = Button(
    area=(799, 185, 856, 228),
    color=(),
    button=(799, 185, 856, 228),
    file='./assets/jiangnan/building/tax_office_idle.png',
    name='TAX_LEVEL_LOCK_CHECK',
)

# 各等级的锁定图标搜索区域。
TAX_LEVEL_LOCK_AREAS = {
    2: (785, 170, 875, 245),
    3: (408, 385, 495, 455),
    4: (785, 385, 875, 455),
}

# 各等级工作卡片内部的安全点击区域。
# 不包含右下角“升级”按钮。
TAX_LEVEL_BUTTONS = {
    1: Button(
        area=(),
        color=(),
        button=(485, 175, 555, 225),
        name='TAX_LEVEL_1',
    ),
    2: Button(
        area=(),
        color=(),
        button=(875, 175, 945, 225),
        name='TAX_LEVEL_2',
    ),
    3: Button(
        area=(),
        color=(),
        button=(495, 390, 565, 440),
        name='TAX_LEVEL_3',
    ),
    4: Button(
        area=(),
        color=(),
        button=(875, 390, 945, 440),
        name='TAX_LEVEL_4',
    ),
}


# Yingtian inn: separate controls from both city and canal navigation.
INN_CRAFT_CHECK = Button(
    area=(922, 659, 980, 695), color=(), button=(922, 659, 980, 695),
    file='./assets/jiangnan/inn/main.png', name='INN_CRAFT_CHECK',
)
INN_SPICE_CHECK = Button(
    area=(1010, 659, 1069, 695), color=(), button=(1010, 659, 1069, 695),
    file='./assets/jiangnan/inn/main.png', name='INN_SPICE_CHECK',
)
INN_EXIT_BUTTON = Button(
    area=(1192, 582, 1249, 641), color=(), button=(1194, 589, 1247, 690),
    file='./assets/jiangnan/inn/main.png', name='INN_EXIT_BUTTON',
)
INN_EXIT_CONFIRM_CHECK = Button(
    area=(385, 326, 831, 362), color=(), button=(385, 326, 831, 362),
    file='./assets/jiangnan/inn/exit_confirm.png', name='INN_EXIT_CONFIRM_CHECK',
)
INN_EXIT_CONFIRM_BUTTON = Button(
    area=(735, 492, 809, 525), color=(), button=(694, 490, 850, 526),
    file='./assets/jiangnan/inn/exit_confirm.png', name='INN_EXIT_CONFIRM_BUTTON',
)
INN_MAP_ICON = Button(
    # The food sign beside the cyan search arrow is the inn entrance.
    area=(572, 181, 622, 241), color=(), button=(581, 193, 612, 228),
    file='./assets/jiangnan/inn/map.png', name='INN_MAP_ICON',
)
