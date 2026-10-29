"""Regenerate the Jiangnan schema and clean template from the repository root."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
from module.config.config_updater import ConfigGenerator, ConfigUpdater

if __name__ == '__main__':
    ConfigGenerator().generate()
    ConfigUpdater().update_file('template', is_template=True)
