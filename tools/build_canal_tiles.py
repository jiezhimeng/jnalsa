"""Build browser-friendly tiles from the exact bundled Grand Canal map."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / 'assets' / 'jiangnan' / 'grand_canal' / 'map'
SOURCE = MAP_DIR / 'overview_stitched.png'
TILE_SIZE = 2048


def save_jpeg(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, 'JPEG', quality=88, optimize=True)


def main():
    # This is a trusted, locally generated map; its 505 million pixels are
    # intentionally above Pillow's generic decompression-bomb threshold.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(SOURCE) as source:
        source.load()
        width, height = source.size
        if source.mode != 'RGB':
            source = source.convert('RGB')

        preview = source.reduce(16)
        save_jpeg(preview, MAP_DIR / 'overview_preview.jpg')
        print(f'preview: {preview.size}', flush=True)

        for factor in (4, 1):
            directory = MAP_DIR / 'tiles' / f'l{factor}'
            span = TILE_SIZE * factor
            columns = (width + span - 1) // span
            rows = (height + span - 1) // span
            for row in range(rows):
                for column in range(columns):
                    box = (
                        column * span, row * span,
                        min((column + 1) * span, width),
                        min((row + 1) * span, height),
                    )
                    tile = source.crop(box)
                    if factor > 1:
                        tile = tile.reduce(factor)
                    save_jpeg(tile, directory / f'{column}_{row}.jpg')
                print(f'l{factor}: row {row + 1}/{rows}', flush=True)
    print('Grand Canal map tiles ready', flush=True)


if __name__ == '__main__':
    main()
