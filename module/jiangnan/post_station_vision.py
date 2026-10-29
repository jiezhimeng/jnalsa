"""Station screenshot recognition, calibrated against the supplied MuMu frames."""
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
from module.jiangnan.post_station_cities import ALL_MAP_NAMES, MAP_REGION

ASSETS = Path(__file__).resolve().parents[2] / 'assets/jiangnan/post_station'
RESOURCE_SEARCH_MIN = 50
RESOURCE_SEARCH_MAX = 300
RESOURCE_FALLBACK_OFFSET = 140
RESOURCE_CLICK_JITTER = 3


@lru_cache(maxsize=64)
def template(name):
    data = np.frombuffer((ASSETS / (name if name.endswith('.png') else name + '.png')).read_bytes(), np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=1)
def resource_templates():
    return json.loads((ASSETS / 'resources.json').read_text(encoding='utf-8'))


def match(image, name, region, threshold=0.86, tone=26):
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    reference = template(name) if isinstance(name, str) else name
    h, w = reference.shape[:2]
    if roi.shape[0] < h or roi.shape[1] < w:
        return None
    scores = cv2.matchTemplate(roi, reference, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < threshold:
        return None
    # Correlation alone also matches controls behind a modal dimming layer.
    difference = np.mean(np.abs(roi[y:y+h, x:x+w].astype(float) - reference))
    if difference > tone:
        return None
    return x1 + x, y1 + y, x1 + x + w, y1 + y + h


def pinned_labels(image):
    """Return name-label boxes of visible cards with a top-right pin badge.

    Pins are detected independently of item names/icons, so new resources need
    no bundled resource template. The badge itself is never a click target.
    """
    reference = template('pin')
    roi = image[108:650, 345:1085]
    scores = cv2.matchTemplate(roi, reference, cv2.TM_CCOEFF_NORMED)
    peaks = (scores >= .86) & (scores == cv2.dilate(scores, np.ones((15,15),np.uint8)))
    ys, xs = np.where(peaks)
    result = []
    for y, x in zip(ys, xs):
        left, top = int(x)+345, int(y)+108
        # Only the six resource-card columns can contain pin badges.
        if min(abs(left-(444+122*col)) for col in range(6)) > 10:
            continue
        box = (left-83, top+114, left+11, top+158)
        if box[1] >= 112 and box[3] <= 650:
            result.append(box)
    return sorted(result, key=lambda box:(box[1]//80,box[0]))


@lru_cache(maxsize=32)
def _map_page_template(key):
    reference = template('map_'+key)
    return cv2.resize(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),None,fx=.5,fy=.5)


def _is_map_page(image):
    """Cheap half-size candidates, then full-resolution local confirmation.

    All cities share navigation, so scanning 24 large full-screen templates on
    every city frame would slow unrelated production and navigation tasks.
    """
    x1,y1,x2,y2 = MAP_REGION
    search = cv2.resize(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY),None,fx=.5,fy=.5)
    for key in (*ALL_MAP_NAMES,'unreleased'):
        coarse = _map_page_template(key)
        _,score,_,(x,y) = cv2.minMaxLoc(cv2.matchTemplate(search,coarse,cv2.TM_CCOEFF_NORMED))
        if score < .65:
            continue
        h,w = template('map_'+key).shape[:2]
        region = (max(x1,x1+x*2-5),max(y1,y1+y*2-5),
                  min(x2,x1+x*2+w+5),min(y2,y1+y*2+h+5))
        if match(image,'map_'+key,region):
            return True
    return False


def page(image):
    if image is None:
        return 'unknown'
    # All-tab pins can point to crafting buildings as well as gathering nodes.
    # Confirm both the blue paper frame and the make button; never infer this
    # state from a blurred field control or the red button colour alone.
    if (match(image, 'resource_craft', (475, 535, 800, 610))
            and match(image, 'resource_craft_frame', (295, 55, 370, 255))):
        return 'craft'
    checks = (
        ('exhausted', (530, 155, 755, 235)),
        ('exit', (365, 305, 715, 380)),
        ('list', (198, 54, 312, 410)),
        ('characters', (200, 140, 310, 445)),
        ('overview', (410, 442, 605, 510)),
        ('field', (1173, 175, 1275, 235)),
    )
    for name, region in checks:
        if match(image, name, region):
            return name
    if _is_map_page(image):
        return 'maps'
    return 'unknown'


def unreachable(image):
    """Match the floating '你无法到达这里' glyphs over changing map scenery.

    The toast rises from the screen centre. Match its light text strokes over
    a vertical band rather than matching the translucent background or one y.
    """
    if image is None:
        return False

    def strokes(rgb):
        red, green, blue = cv2.split(rgb)
        return cv2.inRange(rgb, (185, 170, 140), (255, 250, 230)) & cv2.inRange(
            cv2.subtract(red, blue), 12, 70)

    reference = strokes(template('unreachable'))
    search = strokes(image[100:440, 435:845])
    score = cv2.minMaxLoc(cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED))[1]
    return score >= .78


def red_arrow(image):
    """Return the arrow tip, excluding sidebars and red flags/buttons."""
    # Actual arrow fill is RGB (166, 52, 40), and forms a solid, tall polygon.
    mask = cv2.inRange(image, (148, 32, 20), (182, 72, 65))
    mask[:, :90] = 0
    mask[:, 1170:] = 0
    # A map clamped against its boundary can leave the target near the bottom.
    # Exclude the actual bag/return controls, not the entire lower map strip.
    mask[565:, 1070:] = 0
    mask[image.shape[0]-15:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (45 <= w <= 80 and 100 <= h <= 145):
            continue
        area = cv2.contourArea(contour)
        if not .58 < area / (w*h) < .85:
            continue
        # Shaft occupies the middle 2/3; the bottom narrows to a point.
        rows = mask[y:y+h, x:x+w] > 0
        if rows[-5:].sum(axis=1).mean() > w * .28:
            continue
        found.append((x + w // 2, y + h - 1, h))
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class ResourceForeground:
    point: tuple
    box: tuple
    method: str = 'foreground'
    ambiguous: bool = False


def resource_fallback(arrow):
    """User-selected fallback on the current arrow axis, 140px below its tip."""
    if arrow is None:
        return None
    x, tip, _ = arrow
    y = tip+RESOURCE_FALLBACK_OFFSET
    margin = RESOURCE_CLICK_JITTER
    if not 95+margin < x < 1160-margin or not 70+margin < y < 700-margin:
        return None
    if x+margin >= 1070 and y+margin >= 565:
        return None
    return ResourceForeground((int(x), int(y)), None, 'arrow_140')


def resource_target_obstructed(image, point):
    """Reject known tiled roofs covering azalea and Huizhou pine targets.

    Match actual roof texture, rather than rejecting all large/gray objects
    (which would also reject ore and tree canopies). Check fallback clicks too.
    """
    x, y = point
    for name in ('resource_obstruction_roof', 'resource_obstruction_huizhou_roof'):
        box = match(image, name,
                    (max(95, x-240), max(65, y-180),
                     min(1170, x+240), min(700, y+180)), threshold=.92, tone=15)
        if (box and box[0]-20 <= x <= box[2]+20
                and box[1]-20 <= y <= box[3]+20):
            return True
    return False


def _quantized_colors(pixels, shift):
    """Exact lexicographic RGB counts without sorting thousands of rows."""
    bits = 8-shift
    values = pixels.astype(np.uint16) >> shift
    packed = (values[:, 0] << (bits*2)) | (values[:, 1] << bits) | values[:, 2]
    histogram = np.bincount(packed, minlength=1 << (bits*3))
    present = np.flatnonzero(histogram)
    mask = (1 << bits)-1
    colors = np.column_stack((present >> (bits*2), (present >> bits) & mask, present & mask))
    return colors, histogram[present]


def _background_palette(image):
    """Learn the large, flat grass/water colours from this map frame."""
    # Subsampling also avoids rebuilding a full-frame colour histogram for
    # every short-lived arrow frame. Object outlines are excluded as non-flat.
    sample = image[80:560:3, 100:1170:3]
    smooth = cv2.GaussianBlur(image, (7, 7), 0)[80:560:3, 100:1170:3]
    flat = np.max(np.abs(sample.astype(np.int16)-smooth), axis=2) < 3
    pixels = sample[flat]
    if len(pixels) < 100:
        return ()
    values, counts = _quantized_colors(pixels, 4)
    colors = []
    for index in np.argsort(counts)[::-1]:
        if counts[index] < len(pixels)*.025:
            break
        color = values[index].astype(np.float32)*16+8
        if all(np.linalg.norm(color-other) > 40 for other in colors):
            colors.append(color)
        if len(colors) == 3:
            break
    return colors


def resource_foreground(image, arrow=None):
    """Find solid foreground on the arrow's centre line, without a y offset.

    The search arrow supplies x only. Flat map colours are removed; connected
    silhouettes and separate foreground colours provide possible objects. The
    latter separate a brown boar from an adjoining blue mountain. The click is
    in the lower middle of the visible silhouette on the exact arrow axis,
    within 50..300 pixels below the current arrow tip. The 3-pixel click
    randomization also stays inside that search band.
    Return None for unsupported targets rather than scanning
    lower scenery with a sequence of speculative clicks.
    """
    if image is None or image.shape[:2] != (720, 1280):
        return None
    arrow = red_arrow(image) if arrow is None else arrow
    if arrow is None:
        return None
    x, tip, _ = arrow
    if not 95 < x < 1160 or tip+RESOURCE_SEARCH_MIN >= 700:
        return None
    colors = _background_palette(image)
    if not len(colors):
        return None
    x1, x2 = max(95, x-220), min(1170, x+220)
    y1 = max(65, tip+RESOURCE_SEARCH_MIN)
    y2 = min(700, tip+RESOURCE_SEARCH_MAX+1)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    search = roi.astype(np.float32)
    closest = np.full(roi.shape[:2], np.inf, dtype=np.float32)
    for color in colors:
        delta = search-color
        squared = cv2.transform(delta*delta, np.ones((1, 3), np.float32))
        np.minimum(closest, squared, out=closest)
    foreground = np.uint8(closest > 39*39)*255
    # The bag/return controls must never become foreground candidates.
    foreground[max(0, 565-y1):, max(0, 1070-x1):] = 0
    mask = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    masks = [mask]
    axis = x-x1
    pixels = roi[:, max(0, axis-7):axis+8].reshape(-1, 3)
    values, counts = _quantized_colors(pixels, 5)
    # The wider band contains more terrain colours; retain enough seeds for
    # small tree canopies as well as the water/rocks farther down the axis.
    for index in np.argsort(counts)[::-1][:24]:
        color = values[index].astype(np.float64)*32+16
        if min(np.linalg.norm(color-other) for other in colors) < 42:
            continue
        part = cv2.inRange(roi, np.maximum(0, color-32), np.minimum(255, color+32))
        # A colour range may include water even when its seed is foreground.
        part[foreground == 0] = 0
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        masks.append(cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    candidates = []
    for part in masks:
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 400 or w < 15 or h < 30:
                continue
            if bx <= 1 or bx+w >= roi.shape[1]-1:
                continue
            # A silhouette may cross the search band's lower edge. Only the
            # actual screen edge is forbidden, not an artificial band edge.
            if y2 == 700 and by+h >= roi.shape[0]-1:
                continue
            if not bx <= axis < bx+w or abs(bx+w/2-axis) > 24:
                continue
            filled = np.zeros_like(part)
            cv2.drawContours(filled, [contour], -1, 255, -1)
            if np.count_nonzero((gray < 90) & (filled > 0)) < max(8, area*.015):
                continue
            score = by+abs(bx+w/2-axis)*3-min(60, area**.5*.5)
            candidates.append((score, filled, part, (bx+x1, by+y1, bx+x1+w, by+y1+h)))
    selected_box = None
    for _, filled, part, box in sorted(candidates, key=lambda entry: entry[0]):
        # Padding makes a silhouette cut by the arrow/crop have a real edge.
        padded = cv2.copyMakeBorder(filled, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        clearance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        support = cv2.blur((part > 0).astype(np.float32), (9, 9))
        column = clearance[:, axis].copy()
        column[support[:, axis] < .65] = 0
        if column.max() < 3:
            continue
        # Keep alternate colour silhouettes on the same visible object.
        if selected_box is None:
            selected_box = box
        else:
            overlap = max(0, min(box[3], selected_box[3])-max(box[1], selected_box[1]))
            if overlap < min(box[3]-box[1], selected_box[3]-selected_box[1])*.5:
                continue
        middle = (box[1]+box[3])/2
        preferred = box[1]+(box[3]-box[1])*.65
        minimum = max(middle, tip+RESOURCE_SEARCH_MIN+RESOURCE_CLICK_JITTER)
        maximum = min(699-RESOURCE_CLICK_JITTER,
                      tip+RESOURCE_SEARCH_MAX-RESOURCE_CLICK_JITTER)
        rows = np.arange(len(column))+y1
        eligible = np.flatnonzero((column >= RESOURCE_CLICK_JITTER+1)
                                 & (rows >= minimum) & (rows <= maximum))
        if not len(eligible):
            continue
        cy = int(eligible[np.argmin(np.abs(rows[eligible]-preferred))])
        point = (int(x), cy+y1)
        if point[0] >= 1070 and point[1] >= 565:
            return None
        # A large outline can join a tree to a cart farther down the axis.
        # Separate substantial colour components above and at the proposed
        # click, with a visible vertical gap, contradict one solid target.
        # Preserve the evidence for the caller; never silently fall back to a
        # guessed height when segmentation itself is ambiguous.
        ambiguous = False
        if box[3]-box[1] >= 170:
            parts = [other for _, _, _, other in candidates
                     if other != box
                     and other[0] >= box[0]-6 and other[2] <= box[2]+6
                     and other[1] >= box[1]-6 and other[3] <= box[3]+6
                     and other[0]+RESOURCE_CLICK_JITTER < x < other[2]-RESOURCE_CLICK_JITTER
                     and other[2]-other[0] >= (box[2]-box[0])*.25]
            upper = [other for other in parts
                     if other[3]-other[1] >= 60 and other[1] <= box[1]+20
                     and other[3] <= point[1]-24]
            lower = [other for other in parts
                     if other[3]-other[1] >= 40 and other[1] <= point[1] <= other[3]
                     and other[3] >= box[3]-20]
            ambiguous = any(below[1]-above[3] >= 12 for above in upper for below in lower)
        return ResourceForeground(point, box, ambiguous=ambiguous)
    return None


def bubble_click_point(image, box):
    """Choose an interior point outside the arrow's full bobbing column."""
    x1, y1, x2, y2 = box
    center = ((x1+x2)//2, (y1+y2)//2)
    arrow = red_arrow(image)
    if arrow is None:
        return center
    ax, tip, height = arrow
    if tip+20 < y1 or tip-height-20 > y2 or ax+48 < x1 or ax-48 > x2:
        return center
    # The arrow is at most 80px wide. Keep 8px beyond each side for
    # click jitter and animation; never aim below its current tip.
    inset = min(20, (x2-x1)//4)
    left, right = x1+inset, x2-inset
    regions = [(left, min(right, ax-48)), (max(left, ax+48), right)]
    regions = [(start, end) for start, end in regions if end > start]
    if not regions:
        return None
    start, end = max(regions, key=lambda region: region[1]-region[0])
    return ((start+end)//2, center[1])


def bubble(image):
    """Find a gathering ellipse, allowing the target arrow to cover its top."""
    red, green, blue = cv2.split(image)
    spread = cv2.subtract(cv2.max(cv2.max(red, green), blue),
                          cv2.min(cv2.min(red, green), blue))
    mask = cv2.inRange(image, (176, 166, 146), (255, 255, 255))
    mask &= cv2.compare(spread, 65, cv2.CMP_LT)
    ivory = mask > 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    arrow = None
    arrow_checked = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (170 <= w <= 225 and 110 <= h <= 190 and x > 90 and x+w < 1170
                and y > 35 and y+h < image.shape[0]-20):
            continue
        if x+w >= 1070 and y+h >= 565:
            continue
        if not 1.0 < w/h < 1.6:
            continue
        component = np.zeros((h, w), np.uint8)
        cv2.drawContours(component, [contour - np.array([[[x, y]]])], -1, 1, -1)
        # The moving red arrow can cut into the ellipse from its upper edge.
        # Only discount pixels of a verified, centred arrow inside the visible
        # contour's hull; ordinary red scenery must not relax bubble checks.
        if not arrow_checked:
            arrow = red_arrow(image)
            arrow_checked = True
        # A low bobbing arrow can hide the entire narrow tail. Its ivory
        # contour then contains just the ellipse body (126px in the log).
        # Relax the tail/height checks only when that verified arrow covers
        # the expected tail, centred below a compact ellipse body.
        hidden_tail = bool(
            arrow and 110 <= h <= 145 and 1.3 < w/h < 1.9
            and abs(arrow[0]-(x+w/2)) <= 12
            and y+h+12 <= arrow[1] <= y+h+65
            and arrow[1]-arrow[2]+1 < y+h*.45
        )
        if h < 140 and not hidden_tail:
            continue
        occlusion = np.zeros((h, w), bool)
        if arrow:
            ax, tip, height = arrow
            if hidden_tail or (abs(ax-(x+w/2)) <= w*.15 and y+20 < tip < y+h-50
                               and tip-height+1 < y+20):
                hull = np.zeros((h, w), np.uint8)
                cv2.drawContours(hull, [cv2.convexHull(contour)-np.array([[[x, y]]])], -1, 1, -1)
                arrow_pixels = cv2.inRange(image[y:y+h, x:x+w], (148,32,20), (182,72,65))
                arrow_pixels[max(0, tip-y+1):] = 0
                arrow_pixels[:max(0, tip-height+1-y)] = 0
                arrow_pixels[:, :max(0, ax-40-x)] = 0
                arrow_pixels[:, max(0, ax+41-x):] = 0
                occlusion = (arrow_pixels > 0) & (hull > 0)
        area = cv2.contourArea(contour) + np.count_nonzero(occlusion & (component == 0))
        if area < w*h*.50:
            continue
        # The ellipse has a very narrow, central tail in its last ~35px.
        tail = component[-25:-5].sum(axis=1)
        if not hidden_tail and (tail.mean() > w*.18 or tail.mean() < 2):
            continue
        lower = h-15 if hidden_tail else h-50
        inside = ivory[y+22:y+lower, x+25:x+w-25]
        visible = ~occlusion[22:lower, 25:w-25]
        if inside[visible].mean() > .70:
            if hidden_tail:
                # Confirm the visible left interior, clear of the arrow.
                candidates.append((x+30, y+30, arrow[0]-40, y+h-25))
            else:
                candidates.append((x, y, x+w, y+h-35))
    # Inner and outer borders can both yield the same ellipse.
    if not candidates:
        return None
    first = candidates[0]
    if any(abs(c[0]-first[0]) + abs(c[1]-first[1]) > 30 for c in candidates):
        return None
    return max(candidates, key=lambda c: (c[2]-c[0])*(c[3]-c[1]))
