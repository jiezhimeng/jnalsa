"""Locate ADB in both the bundled toolkit and a standard Python environment."""
import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_adb(configured=None):
    candidates = [configured, './bin/adb/adb.exe',
                  './toolkit/Lib/site-packages/adbutils/binaries/adb.exe']
    spec = importlib.util.find_spec('adbutils')
    if spec is not None and spec.origin:
        binary = 'adb.exe' if os.name == 'nt' else 'adb'
        candidates.append(str(Path(spec.origin).parent / 'binaries' / binary))
    candidates.append('/usr/bin/adb')
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path.resolve().as_posix()
    return 'adb'
