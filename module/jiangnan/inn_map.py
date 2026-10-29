"""Inn panorama coordinates and screenshot registration (no device input)."""
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INN_MAP_DIR = ROOT / 'assets' / 'jiangnan' / 'inn' / 'map'
PANORAMA = INN_MAP_DIR / 'panorama.png'
STATE_DIR = ROOT / 'log' / 'inn_map'
STATIC_ANCHORS = INN_MAP_DIR / 'static_anchors.json'


def map_instance_id(config_name):
    return hashlib.sha256(str(config_name).encode('utf-8')).hexdigest()[:16]


def map_state_path(config_name):
    return STATE_DIR / ('state-' + map_instance_id(config_name) + '.json')


def _read_rgb(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unable to read inn panorama: %s' % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detector():
    return cv2.ORB_create(nfeatures=12000, fastThreshold=12, edgeThreshold=20)


def _static_anchor_mask(image, annotations=STATIC_ANCHORS):
    """Allow only annotated architecture; missing annotations fail closed."""
    data = json.loads(Path(annotations).read_text(encoding='utf-8'))
    height, width = image.shape[:2]
    if data['map_size'] != [width, height]:
        raise ValueError('Inn static anchors do not match the panorama dimensions')
    annotation_width, annotation_height = data['annotation_size']
    scale = np.float32([width / annotation_width, height / annotation_height])
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in data['polygons']:
        points = np.rint(np.float32(region['points']) * scale).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    for rectangle in data['exclude_rectangles']:
        left, top = np.floor(np.float32(rectangle[:2]) * scale).astype(int)
        right, bottom = np.ceil(np.float32(rectangle[2:]) * scale).astype(int)
        mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 0
    valid = np.uint8(np.max(image, axis=2) > 12) * 255
    valid = cv2.erode(valid, np.ones((21, 21), np.uint8))
    return cv2.bitwise_and(mask, valid)


def _architecture_features(gray, mask):
    """Keep the entire rotated ORB patch on architecture, at every octave.

    Masking only a keypoint's center still lets its descriptor see a nearby
    person or bubble. Require the full surrounding patch to be whitelisted.
    """
    keys, descriptors = _detector().detectAndCompute(gray, mask)
    if descriptors is None:
        return [], None
    integral = cv2.integral(np.uint8(mask > 0))
    height, width = mask.shape
    keep = []
    for index, key in enumerate(keys):
        radius = int(math.ceil(key.size / math.sqrt(2))) + 3
        x, y = int(round(key.pt[0])), int(round(key.pt[1]))
        left, top, right, bottom = x-radius, y-radius, x+radius+1, y+radius+1
        if left < 0 or top < 0 or right > width or bottom > height:
            continue
        area = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
        if area == (right-left) * (bottom-top):
            keep.append(index)
    if not keep:
        return [], None
    return [keys[i] for i in keep], descriptors[keep]


@lru_cache(maxsize=2)
def _atlas_features(path, modified_ns, anchor_path, anchor_modified_ns):
    image = _read_rgb(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = _static_anchor_mask(image, anchor_path)
    keypoints, descriptors = _architecture_features(gray, mask)
    return image.shape[:2], keypoints, descriptors, mask


def _match_architecture(keys, desc, atlas_keys, atlas_desc):
    if desc is None or atlas_desc is None or len(desc) < 20:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, atlas_desc, k=2)
    candidates = [a for pair in pairs if len(pair) == 2
                  for a, b in [pair] if a.distance < 0.72 * b.distance]
    unique = {}
    for match in sorted(candidates, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 18:
        return None
    source = np.float32([keys[m.queryIdx].pt for m in good])
    target = np.float32([atlas_keys[m.trainIdx].pt for m in good])
    matrix, valid = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995, refineIters=10)
    if matrix is None or valid is None:
        return None
    return matrix, source, target, valid.ravel().astype(bool)


def locate_inn_view(image, panorama=PANORAMA, anchors=STATIC_ANCHORS):
    """Return a verified screenshot -> panorama affine transform, or None.

    The game's camera pans and zooms without rotating. Reject weak/repeated
    floor matches, excessive rotation and transforms outside the known atlas.
    Every reference descriptor uses only whitelisted permanent architecture;
    people, bubbles and movable furnishings never contribute reference features.
    Input uses RGB pixels, like JiangnanDevice.image.
    """
    if image is None or image.ndim != 3 or image.shape[:2] != (720, 1280):
        return None
    path = Path(panorama)
    anchor_path = Path(anchors)
    (map_h, map_w), atlas_keys, atlas_desc, atlas_mask = _atlas_features(
        str(path), path.stat().st_mtime_ns, str(anchor_path), anchor_path.stat().st_mtime_ns)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in ((0, 0, 90, 520), (0, 480, 150, 720),
                           (1060, 0, 1280, 280), (890, 550, 1280, 720),
                           (0, 0, 1280, 20)):
        mask[y1:y2, x1:x2] = 0
    keys, desc = _detector().detectAndCompute(gray, mask)
    # A coarse candidate uses only architecture in the reference. Then project
    # that whitelist into this screenshot and extract only architectural
    # patches here as well. Only the second fit can become a reported position.
    coarse = _match_architecture(keys, desc, atlas_keys, atlas_desc)
    if coarse is None or int(coarse[3].sum()) < 18:
        return None
    coarse_matrix = coarse[0]
    coarse_scale = math.hypot(coarse_matrix[0, 0], coarse_matrix[1, 0])
    if not 0.5 <= coarse_scale <= 2 or abs(math.degrees(math.atan2(
            coarse_matrix[1, 0], coarse_matrix[0, 0]))) > 3:
        return None
    projected_mask = cv2.warpAffine(
        atlas_mask, cv2.invertAffineTransform(coarse_matrix), (1280, 720),
        flags=cv2.INTER_NEAREST, borderValue=0)
    keys, desc = _architecture_features(gray, cv2.bitwise_and(mask, projected_mask))
    refined = _match_architecture(keys, desc, atlas_keys, atlas_desc)
    if refined is None:
        return None
    matrix, source, target, inliers = refined
    count = int(inliers.sum())
    ratio = count / len(source)
    if count < 18 or ratio < 0.45:
        return None
    spread = np.ptp(source[inliers], axis=0)
    if spread[0] < 220 or spread[1] < 130:
        return None
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if not 0.5 <= scale <= 2.0 or abs(rotation) > 3.0:
        return None
    projected = cv2.transform(source[inliers].reshape(1, -1, 2), matrix)[0]
    residual = float(np.median(np.linalg.norm(projected - target[inliers], axis=1)))
    if residual > 2.5:
        return None
    corners = np.float32([[[0, 0], [1280, 0], [1280, 720], [0, 720]]])
    viewport = cv2.transform(corners, matrix)[0]
    center = cv2.transform(np.float32([[[640, 360]]]), matrix)[0, 0]
    if not (0 <= center[0] < map_w and 0 <= center[1] < map_h):
        return None
    # Most of the viewport must intersect the known panorama bounds.
    left, top = viewport.min(axis=0)
    right, bottom = viewport.max(axis=0)
    overlap = max(0, min(map_w, right)-max(0, left)) * max(0, min(map_h, bottom)-max(0, top))
    if overlap / max(1, (right-left)*(bottom-top)) < 0.65:
        return None
    return {
        'matrix': matrix.tolist(), 'center': center.tolist(),
        'viewport': viewport.tolist(), 'map_size': [map_w, map_h],
        'screen_size': [1280, 720], 'inliers': count,
        'inlier_ratio': round(ratio, 3), 'error_px': round(residual, 3),
        'scale': scale,
    }


def screen_to_map(location, point):
    matrix = np.asarray(location['matrix'], dtype=np.float64)
    return tuple(np.dot(matrix[:, :2], point) + matrix[:, 2])


def map_to_screen(location, point):
    matrix = np.asarray(location['matrix'], dtype=np.float64)
    return tuple(np.dot(cv2.invertAffineTransform(matrix), [point[0], point[1], 1]))


def save_map_observation(config_name, location, reason=''):
    """Atomically replace state; a failed read must never preserve old coordinates."""
    path = map_state_path(config_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'observed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
               'matched': location is not None, 'location': location, 'reason': reason}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
