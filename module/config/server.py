"""Jiangnan package metadata; resource language is simplified Chinese."""
server = 'cn'
VALID_SERVER = ['cn']
VALID_PACKAGE = {'com.cis.jiangnan.taptap': 'cn'}
VALID_CHANNEL_PACKAGE = {}
VALID_SERVER_LIST = {}
DICT_PACKAGE_TO_ACTIVITY = {
    'com.cis.jiangnan.taptap': 'com.cis.privacy.CISPrivacyActivity',
}


def set_server(package_or_server):
    global server
    server = to_server(package_or_server)
    from module.base.resource import release_resources
    release_resources()


def to_server(package_or_server):
    return 'cn'


def to_package(package_or_server):
    if package_or_server in VALID_PACKAGE:
        return package_or_server
    if package_or_server == 'cn':
        return next(iter(VALID_PACKAGE))
    raise ValueError('Unsupported Jiangnan package: %s' % package_or_server)
