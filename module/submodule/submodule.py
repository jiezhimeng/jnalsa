"""Load the sole supported Jiangnan configuration."""
def load_config(config_name):
    from module.config.config import AzurLaneConfig
    return AzurLaneConfig(config_name, '')


def load_mod(name):
    raise ValueError('External game modules are not supported: %s' % name)
