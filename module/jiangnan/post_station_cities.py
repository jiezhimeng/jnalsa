"""City-specific station destinations, in their in-game scroll order."""

STATIONS = {
    'JiangnanYingtianPostStation': {
        'city': '应天府', 'group': 'post_station',
        'maps': {'qinhuai': '秦淮河畔', 'yingtian': '应天城郊', 'liniang': '丽娘府宅'},
    },
    'JiangnanSuzhouPostStation': {
        'city': '苏州府', 'group': 'post_station_suzhou',
        'maps': {'changmen': '阊门', 'qilishantang': '七狸山塘', 'huqiu': '虎丘', 'taohuawu': '桃花坞'},
    },
    'JiangnanHangzhouPostStation': {
        'city': '杭州府', 'group': 'post_station_hangzhou',
        'maps': {'qiantangmen': '钱塘门', 'jiushi_baijuyi': '旧事·白居易',
                 'jiushi_xiaoman': '旧事·小蛮', 'changhenge': '长恨歌', 'qiantangchunxing': '钱塘春行'},
    },
    'JiangnanSongjiangPostStation': {
        'city': '松江府', 'group': 'post_station_songjiang',
        'maps': {'huatingzhongcai': '华亭重彩', 'jinxihexi': '今夕何夕', 'changxiangsi': '长相思'},
    },
    'JiangnanHuizhouPostStation': {
        'city': '徽州府', 'group': 'post_station_huizhou',
        'maps': {'xiuninglongwu': '休宁龙舞', 'huangshanyunhai': '黄山云海',
                 'shexianshuangli': '歙县双鲤', 'huihanggudao': '徽杭古道'},
    },
    'JiangnanYangzhouPostStation': {
        'city': '扬州府', 'group': 'post_station_yangzhou',
        'maps': {'wanhuahui': '万花会', 'jiushi_bailichenxiang': '旧事·百里沉香', 'chuanxingqianli': '船行千里'},
    },
    'JiangnanNingboPostStation': {
        'city': '宁波府', 'group': 'post_station_ningbo',
        'maps': {'lvmeimao': '绿眉毛'},
    },
}

ALL_MAP_NAMES = {key: name for spec in STATIONS.values() for key, name in spec['maps'].items()}
MAP_REGION = (190, 50, 1090, 667)


def map_fields(maps):
    return {key: 'Map' + key.capitalize() for key in maps}
