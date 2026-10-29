"""Validate a clean source release without starting the GUI or any device."""
import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct
import sys
import warnings
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAP = 'assets/jiangnan/grand_canal/map/overview_stitched.png'


class ResourceLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href', 'xlink:href') and value:
                self.links.append(value)


def validate_map_resources(root, errors):
    directory = root / 'assets/jiangnan/grand_canal/map'
    # Must agree with build_canal_tiles.py and map_tiles.js, including edge tiles.
    tile_count = 0
    for factor in (1, 4):
        span = 2048 * factor
        for row in range((18677 + span - 1) // span):
            for column in range((27020 + span - 1) // span):
                tile = directory / 'tiles' / ('l%s' % factor) / ('%s_%s.jpg' % (column, row))
                if not tile.is_file():
                    errors.append('Missing map tile: ' + tile.relative_to(root).as_posix())
                tile_count += 1
    for page in (root/'assets/jiangnan').rglob('*.html'):
        parser = ResourceLinks()
        parser.feed(page.read_text(encoding='utf-8'))
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
                continue
            if not (page.parent / unquote(url.path)).is_file():
                errors.append('Missing map page resource in %s: %s' % (
                    page.relative_to(root).as_posix(), link))
    print('Map resources: %s expected tiles and local HTML links checked.' % tile_count)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root):
    errors = []
    manifest = json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    paths = [row['path'] for row in entries]
    if len(paths) != len(set(paths)):
        errors.append('Duplicate manifest paths')
    for row in entries:
        path = root/row['path']
        if not path.is_file():
            errors.append('Missing file: ' + row['path'])
        elif path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            errors.append('File differs from the release: ' + row['path'])

    image = root/MAP
    if image.is_file():
        with image.open('rb') as stream:
            header = stream.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n':
            errors.append('Original map is not a PNG; run git lfs pull if it is a pointer')
        elif image.stat().st_size <= 100 * 1024 * 1024:
            errors.append('Original map has an unexpected size')
        elif struct.unpack('>II', header[16:24]) != (27020, 18677):
            errors.append('Original map has unexpected dimensions')
        else:
            print('Built-in map: %s x %s; %.2f MiB' % (
                *struct.unpack('>II', header[16:24]), image.stat().st_size/1024**2))
    else:
        errors.append('Built-in original map is missing')
    for map_file in ['assets/jiangnan/grand_canal/map/overview.html',
                     'assets/jiangnan/grand_canal/map/map_tiles.js',
                     'assets/jiangnan/inn/map/panorama.png',
                     'assets/jiangnan/inn/map/static_anchors.json']:
        if not (root/map_file).is_file(): errors.append('Missing map resource: ' + map_file)
    validate_map_resources(root, errors)

    schema = json.loads((root/'module/config/argument/args.json').read_text(encoding='utf-8'))
    if {key for key in schema if not key.startswith('Jiangnan')} != {'Alas', 'Restart'}:
        errors.append('Unexpected legacy task in configuration schema')
    if 'DropRecord' in schema.get('Alas', {}):
        errors.append('Legacy statistics configuration remains')
    sources = list((root/'module').rglob('*.py')) + list((root/'deploy').rglob('*.py'))
    sources += [root/'alas.py', root/'gui.py'] + list((root/'tools').glob('*.py'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', SyntaxWarning)
        for path in sources:
            relative = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding='utf-8-sig')
                tree = ast.parse(text, filename=relative)
                compile(text, relative, 'exec')
            except SyntaxError as exc:
                errors.append('Python syntax error in %s: %s' % (relative, exc))
                continue
            for node in ast.walk(tree):
                value = getattr(node, 's', None)
                if not isinstance(value, str) or '\n' in value:
                    continue
                value = value[2:] if value.startswith('./') else value
                if (path.parent != root/'tools' and value.startswith('assets/')
                        and '{' not in value and not (root/value).exists()):
                    errors.append('Missing runtime asset in %s: %s' % (relative, value))
            for node in ast.walk(tree):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for name in modules:
                    if not name.startswith(('module.', 'deploy.')): continue
                    module_path = root.joinpath(*name.split('.'))
                    if not module_path.is_dir() and not module_path.with_suffix('.py').is_file():
                        errors.append('Missing local import in %s: %s' % (relative, name))

    forbidden = ['dev_tools', 'webapp', 'backup', '.test_tmp', '.venv', '.runtime', 'toolkit', 'log',
                 'module/jiangnan.zip', 'assets/gooey', 'config/alas.json', 'config/deploy.yaml']
    for name in forbidden:
        if any(path == name or path.startswith(name + '/') for path in paths):
            errors.append('Development/private artifact was included in manifest: ' + name)
    if errors:
        for error in sorted(set(errors)):
            print('ERROR: ' + error)
        return 1
    print('OK: %s files match SHA-256; %s Python sources and %s task definitions checked.' % (
        len(entries), len(sources), len(schema)))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    sys.exit(validate(args.root.resolve()))
