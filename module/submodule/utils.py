import os

MOD_DICT = {}
MOD_FUNC_DICT = {}
MOD_CONFIG_DICT = {}


def get_available_func():
    return ()


def get_available_mod():
    return set(MOD_DICT)


def get_available_mod_func():
    return set(MOD_FUNC_DICT)


def get_func_mod(func):
    return MOD_FUNC_DICT.get(func)


def list_mod_dir():
    return list(MOD_DICT.items())


def get_mod_dir(name):
    return MOD_DICT.get(name)


def get_mod_filepath(name):
    return os.path.join('./submodule', get_mod_dir(name))


def list_mod_template():
    return []


def list_mod_instance():
    global MOD_CONFIG_DICT
    MOD_CONFIG_DICT.clear()
    return []


def get_config_mod(config_name):
    """
    Args:
        config_name (str):
    """
    return 'alas'
