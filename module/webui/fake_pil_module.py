import sys
from types import ModuleType


def import_fake_pil_module():
    fake_pil_module = ModuleType('PIL')
    fake_pil_module.Image = ModuleType('PIL.Image')
    fake_pil_module.ImageDraw = ModuleType('PIL.ImageDraw')
    fake_pil_module.Image.Image = type('MockPILImage', (), dict(__init__=None))
    fake_pil_module.ImageDraw.ImageDraw = type('MockImageDraw', (), dict(__init__=None))
    sys.modules['PIL'] = fake_pil_module
    sys.modules['PIL.Image'] = fake_pil_module.Image
    sys.modules['PIL.ImageDraw'] = fake_pil_module.ImageDraw


def remove_fake_pil_module():
    sys.modules.pop('PIL', None)
    sys.modules.pop('PIL.Image', None)
    sys.modules.pop('PIL.ImageDraw', None)
