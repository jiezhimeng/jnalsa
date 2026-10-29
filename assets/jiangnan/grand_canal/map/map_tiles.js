(() => {
    const svg = document.getElementById('map');
    const layer = document.getElementById('detail-tiles');
    if (!svg || !layer) return;

    const mapWidth = 27020;
    const mapHeight = 18677;
    const tilePixels = 2048;
    const namespace = 'http://www.w3.org/2000/svg';
    const loaded = new Map();
    let activeFactor = null;
    let pendingFrame = 0;

    function updateTiles() {
        pendingFrame = 0;
        const view = svg.viewBox.baseVal;
        const renderedWidth = svg.getBoundingClientRect().width;
        if (!renderedWidth) return;

        // At normal size the 197 KB preview already has more pixels than
        // the embedded viewport. Switch to details only as the user zooms.
        const mapPixelsPerScreenPixel = view.width / renderedWidth;
        const factor = mapPixelsPerScreenPixel > 12 ? 16
            : mapPixelsPerScreenPixel > 3 ? 4 : 1;
        if (factor !== activeFactor) {
            for (const image of loaded.values()) image.remove();
            loaded.clear();
            activeFactor = factor;
        }
        if (factor === 16) return;

        const span = tilePixels * factor;
        const maxColumn = Math.ceil(mapWidth / span) - 1;
        const maxRow = Math.ceil(mapHeight / span) - 1;
        const firstColumn = Math.max(0, Math.floor(view.x / span) - 1);
        const lastColumn = Math.min(maxColumn, Math.floor((view.x + view.width) / span) + 1);
        const firstRow = Math.max(0, Math.floor(view.y / span) - 1);
        const lastRow = Math.min(maxRow, Math.floor((view.y + view.height) / span) + 1);
        const wanted = new Set();

        for (let row = firstRow; row <= lastRow; row++) {
            for (let column = firstColumn; column <= lastColumn; column++) {
                const key = `${factor}:${column}:${row}`;
                wanted.add(key);
                if (loaded.has(key)) continue;
                const x = column * span;
                const y = row * span;
                const image = document.createElementNS(namespace, 'image');
                image.setAttribute('x', x);
                image.setAttribute('y', y);
                image.setAttribute('width', Math.min(span, mapWidth - x));
                image.setAttribute('height', Math.min(span, mapHeight - y));
                image.setAttribute('preserveAspectRatio', 'none');
                image.setAttribute('href', `tiles/l${factor}/${column}_${row}.jpg`);
                layer.appendChild(image);
                loaded.set(key, image);
            }
        }
        for (const [key, image] of loaded) {
            if (!wanted.has(key)) {
                image.remove();
                loaded.delete(key);
            }
        }
    }

    function scheduleUpdate() {
        if (!pendingFrame) pendingFrame = requestAnimationFrame(updateTiles);
    }

    new MutationObserver(scheduleUpdate).observe(svg, {
        attributes: true,
        attributeFilter: ['viewBox'],
    });
    new ResizeObserver(scheduleUpdate).observe(svg);
    scheduleUpdate();
})();
