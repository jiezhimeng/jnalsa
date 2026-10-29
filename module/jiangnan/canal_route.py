"""Map-based positioning and route selection for the Yangzhou Grand Canal."""

import json
import hashlib
import math
import re
from collections import deque
from functools import lru_cache
from pathlib import Path


TOPOLOGY_PATH = (Path(__file__).resolve().parents[2] / 'assets' /
                 'jiangnan' / 'grand_canal' / 'map' / 'topology.json')
SHIP_CENTER = (640, 360)
MATCH_TOLERANCE = 54


@lru_cache(maxsize=1)
def load_topology():
    data = json.loads(TOPOLOGY_PATH.read_text(encoding='utf-8'))
    points = {key: tuple(value) for key, value in data['points'].items()}
    neighbors = {key: set() for key in points}
    for start, end in data['segments']:
        neighbors[start].add(end)
        neighbors[end].add(start)
    return points, neighbors, set(data['workPoints'])


def parse_custom_route(raw, neighbors=None):
    """Accept a closed walk of directly connected P nodes, including repeats."""
    if not str(raw or '').strip():
        return ()
    if neighbors is None:
        _, neighbors, _ = load_topology()
    route = tuple(part.upper() for part in re.split(r'\s*[-－—]\s*', str(raw).strip()))
    if len(route) < 4 or any(not re.fullmatch(r'P\d+', p) for p in route):
        raise ValueError('路线格式应为 P1-P2-P3-P1，至少包含三个不同点位')
    if route[0] != route[-1]:
        raise ValueError('路线首尾必须是同一个点位')
    if len(set(route[:-1])) < 3:
        raise ValueError('路线至少包含三个不同点位')
    for start, end in zip(route, route[1:]):
        if start not in neighbors or end not in neighbors[start]:
            raise ValueError(f'{start}-{end} 不是大地图中的直接连接')
    return route


def custom_route_position(route, current, previous=None, progress=''):
    """Track an occurrence in a closed walk using visually confirmed nodes."""
    if not route:
        return None, ''
    cycle = route[:-1]
    signature = hashlib.sha256('-'.join(route).encode('ascii')).hexdigest()
    try:
        saved = json.loads(progress or '{}')
        index = saved.get('index')
        if (saved.get('route') != signature or type(index) is not int
                or not 0 <= index < len(cycle)):
            index = None
    except (TypeError, ValueError, AttributeError):
        index = None
    if index is not None and cycle[index] == current:
        selected = index
    elif index is not None and cycle[(index + 1) % len(cycle)] == current:
        selected = (index + 1) % len(cycle)
    else:
        candidates = [i for i, node in enumerate(cycle) if node == current]
        matching = [i for i in candidates if cycle[(i - 1) % len(cycle)] == previous]
        candidates = matching or candidates
        if not candidates:
            return None, progress
        selected = min(candidates, key=lambda i: (i - index) % len(cycle)) if index is not None else candidates[0]
    return selected, json.dumps({'route': signature, 'index': selected})


def _matched_nodes(origin, observed, points, neighbors):
    ox, oy = points[origin]
    # Locate centers our own ship on its occupied map node. That node is an
    # explicit anchor even when the hull hides its blue circle.
    matches = [(0.0, origin)]
    candidates = []
    for index, (sx, sy) in enumerate(observed):
        for node, (x, y) in points.items():
            if node == origin:
                continue
            error = math.hypot((x - ox) - (sx - SHIP_CENTER[0]),
                               (oy - y) - (sy - SHIP_CENTER[1]))
            if error <= MATCH_TOLERANCE:
                candidates.append((error, index, node))
    used_points, used_nodes = set(), {origin}
    for error, index, node in sorted(candidates):
        if index not in used_points and node not in used_nodes:
            used_points.add(index)
            used_nodes.add(node)
            matches.append((error, node))
    adjacent = sum(node in neighbors[origin] for _, node in matches)
    return matches, adjacent


def locate_ship(observed, previous=None, trusted_previous=False):
    """Match screen blue-point offsets to the map, anchored at the centered ship.

    Returns (current map node, screen-point-to-map-node pairs) only when the
    distribution is sufficiently distinctive. The saved position is checked
    first but never substitutes for visible blue points.
    """
    points, neighbors, _ = load_topology()
    # The center belongs to the ship, never to a second external landmark or
    # a clickable destination. Do not double count a visible circle there.
    observed = [(sx, sy) for sx, sy in observed
                if math.hypot(sx - SHIP_CENTER[0], sy - SHIP_CENTER[1]) > 24]
    if previous not in points:
        previous = None
    if not observed or (len(observed) < 2 and not previous):
        return None
    ranked = []
    # Test the UI hint first, then score every possible position. A hint
    # cannot win against a materially better blue-point distribution.
    if len(observed) >= 2:
        origins = ([previous] if previous else []) + [
            node for node in points if node != previous
        ]
    else:
        origins = [previous] + sorted(neighbors[previous])
    for origin in origins:
        whole_matches, adjacent = _matched_nodes(origin, observed, points, neighbors)
        matches = [(error, node) for error, node in whole_matches if node != origin]
        # Every detected full-screen point must fit this same translation.
        # Matching two nearby points while ignoring a contradictory far point
        # is precisely how similar river junctions became false positions.
        if len(matches) != len(observed) or adjacent < 1:
            continue
        mean_error = sum(error for error, _ in matches) / len(matches)
        score = (len(matches) * 100 + adjacent * 35
                 - mean_error + (5 if previous == origin else 0))
        ranked.append((score, origin, matches, adjacent))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    best = ranked[0]
    # A distant two-point lookalike must not hide a precise one-hop arrival.
    # Only use continuity when the old node no longer fits any translation;
    # if it still fits, retain the ambiguity checks below (including distant
    # identical geometry). Never use an unverified/manual hint this way.
    if (trusted_previous and previous is not None and len(best[2]) == 2
            and best[1] != previous and best[1] not in neighbors[previous]
            and not any(item[1] == previous for item in ranked)):
        nearby = [item for item in ranked if item[1] in neighbors[previous]]
        if (nearby and nearby[0][3] >= 2
                and max(error for error, _ in nearby[0][2]) <= 10):
            # Keep all local rivals for the normal score-gap check, even if
            # only one of them passed the tight arrival geometry above.
            ranked = nearby
            best = ranked[0]
    if (trusted_previous and previous is not None
            and best[1] != previous and best[1] not in neighbors[previous]):
        # Two similar blue-point offsets can fit a distant river junction.
        # A confirmed arrival already records the destination of long/skipped
        # moves, so a fresh two-point match must not teleport that record and
        # reset the route cursor. Distant reacquisition needs three precise
        # live landmarks. They may form a continuous river chain with only
        # one immediate neighbor visible, as at P2 -> P15 -> P16 -> P18.
        if (len(best[2]) < 3 or max(error for error, _ in best[2]) > 10):
            return None
        if best[3] < 2:
            landmarks = {best[1]} | {node for _, node in best[2]}
            connected = {best[1]}
            pending = [best[1]]
            while pending:
                for node in (neighbors[pending.pop()] & landmarks) - connected:
                    connected.add(node)
                    pending.append(node)
            if connected != landmarks:
                return None
    needed_gap = 35 if len(best[2]) == 1 else 25 if len(best[2]) == 2 else 15
    if len(ranked) > 1 and best[0] - ranked[1][0] < needed_gap:
        # Two visible neighbors at P391 match within 3px, but a distant
        # lookalike can still miss the score-gap threshold. A previously
        # confirmed move may break that tie only with tight live geometry.
        best_error = sum(error for error, _ in best[2]) / len(best[2])
        rival_error = sum(error for error, _ in ranked[1][2]) / len(ranked[1][2])
        confirmed_match = (
            trusted_previous and best[1] == previous
            and len(best[2]) >= 2 and best[3] >= 2
            and max(error for error, _ in best[2]) <= 10
            and rival_error - best_error >= 10
        )
        if not confirmed_match:
            return None
    if len(best[2]) == 1 and best[2][0][0] > 20:
        return None
    return best[1], map_visible_destinations(best[1], observed)


def _distance_to_route(start, targets, neighbors):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, distance = queue.popleft()
        if node in targets:
            return distance
        for neighbor in neighbors[node] - seen:
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return float('inf')


def map_visible_destinations(current, observed):
    """Map full-screen circles, including nodes beyond an occupied neighbor."""
    points, _, _ = load_topology()
    ox, oy = points[current]
    candidates = []
    for index, (sx, sy) in enumerate(observed):
        for node, (x, y) in points.items():
            if node == current:
                continue
            error = math.hypot(sx - (640 + x - ox), sy - (360 - y + oy))
            if error <= MATCH_TOLERANCE:
                candidates.append((error, index, node))
    used_points, used_nodes, mapped = set(), set(), []
    for _, index, node in sorted(candidates):
        if index not in used_points and node not in used_nodes:
            mapped.append((observed[index], node))
            used_points.add(index); used_nodes.add(node)
    return mapped


def blue_point_translation(before, after):
    """Infer camera displacement from at least two uniquely agreeing blue points."""
    hypotheses = []
    for ax, ay in before:
        for bx, by in after:
            dx, dy = bx-ax, by-ay
            pairs = sorted((math.hypot(x+dx-u, y+dy-v), i, j)
                           for i, (x, y) in enumerate(before)
                           for j, (u, v) in enumerate(after))
            used_a, used_b, shifts = set(), set(), []
            for error, i, j in pairs:
                if error <= 4 and i not in used_a and j not in used_b:
                    used_a.add(i); used_b.add(j)
                    shifts.append((after[j][0]-before[i][0], after[j][1]-before[i][1]))
            if len(shifts) >= 2:
                hypotheses.append((len(shifts), sum(s[0] for s in shifts)/len(shifts),
                                   sum(s[1] for s in shifts)/len(shifts)))
    if not hypotheses:
        return None
    best = max(hypotheses)
    if any(count == best[0] and math.hypot(dx-best[1], dy-best[2]) > 6
           for count, dx, dy in hypotheses):
        return None
    return best[1:]


def occupied_route_nodes(current, image, screen_offset=(0, 0)):
    """Identify a wooden ship covering a predicted node, without clicking it."""
    import cv2
    import numpy as np
    if image is None:
        return set()
    points, _, _ = load_topology()
    ox, oy = points[current]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width = image.shape[:2]
    occupied = set()
    for node, (x, y) in points.items():
        if node == current:
            continue
        cx = int(round(640+x-ox+screen_offset[0]))
        cy = int(round(360-y+oy+screen_offset[1]))
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        # A pier/roof above the node may look like a vertical wooden hull.
        # A blocking ship must cover the node itself. A visible dark-blue
        # circle at that position is positive evidence that it is free.
        center = hsv[max(0, cy-8):min(height, cy+9), max(0, cx-12):min(width, cx+13)]
        center_wood = cv2.inRange(center, (8, 55, 40), (35, 230, 205))
        center_blue = cv2.inRange(center, (90, 45, 45), (130, 255, 150))
        if np.count_nonzero(center_wood) < 20 or np.count_nonzero(center_blue) >= 80:
            continue
        # 大型漕船会占满旧版 65px 宽的窗口，随后因宽度超过 60px
        # 被过滤。扩展窗口以容纳完整船身，同时保留中心遮挡与纵向
        # 船形限制，避免把节点上方的码头、横向屋顶认成占位船。
        for half_width, maximum_width in ((32, 60), (48, 97)):
            patch = hsv[max(0, cy-95):min(height, cy+25),
                        max(0, cx-half_width):min(width, cx+half_width+1)]
            wood = cv2.inRange(patch, (8, 55, 40), (35, 230, 205))
            if np.count_nonzero(wood) < 120:
                continue
            hull = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((9, 3), dtype=np.uint8))
            count, _, stats, _ = cv2.connectedComponentsWithStats(hull)
            for i in range(1, count):
                left, top, w, h, area = stats[i]
                if 12 <= w <= maximum_width and h >= 35 and h >= w and area >= 160:
                    occupied.add(node)
                    break
            if node in occupied:
                break
    return occupied


def custom_route_destination(route, index, visible_nodes, occupied_nodes, transit_nodes=(), dock_nodes=None):
    """Prefer the next visible dock, retaining dock order while skipping transit points."""
    cycle = route[:-1]
    skipped = []
    fallback = None
    for offset in range(1, len(cycle)):
        target_index = (index + offset) % len(cycle)
        node = cycle[target_index]
        if node in occupied_nodes or (node in transit_nodes and node not in visible_nodes):
            skipped.append(node)
            continue
        if dock_nodes is not None and node not in dock_nodes:
            if node in visible_nodes:
                fallback = (node, target_index, skipped.copy())
            skipped.append(node)
            continue
        if node in visible_nodes:
            return node, target_index, skipped
        break
    if fallback is not None:
        return fallback
    return None, None, skipped


def route_approach_path(current, route, preferred_nodes=()):
    """Shortest navigable path to any configured route node, one stop at a time."""
    _, neighbors, docks = load_topology()
    targets = set(route)
    distance = {node: 0 for node in targets}
    queue = deque(sorted(targets))
    while queue:
        node = queue.popleft()
        for adjacent in sorted(neighbors.get(node, ())):
            if adjacent not in distance:
                distance[adjacent] = distance[node] + 1
                queue.append(adjacent)
    if current not in distance:
        return ()
    path = [current]
    while distance[path[-1]]:
        choices = [node for node in neighbors[path[-1]]
                   if distance.get(node) == distance[path[-1]] - 1]
        # Prefer a clickable first step, then docks, among equally short paths.
        following = min(choices, key=lambda node: (
            len(path) == 1 and node not in preferred_nodes,
            node not in docks, int(node[1:]),
        ))
        path.append(following)
    return tuple(path)


def choose_next(current, visible_nodes, recent_docks=(), previous=None,
                custom_route=(), route_index=None):
    """Choose a visible adjacent node, favoring unvisited docks ahead."""
    _, neighbors, work_points = load_topology()
    visible = set(visible_nodes) & neighbors[current]
    if not visible:
        return None
    if custom_route:
        cycle = custom_route[:-1]
        if current in cycle:
            if route_index is None:
                route_index, _ = custom_route_position(custom_route, current, previous)
            if not 0 <= route_index < len(cycle) or cycle[route_index] != current:
                return None
            desired = cycle[(route_index + 1) % len(cycle)]
            return desired if desired in visible else None
        approach = route_approach_path(current, custom_route, visible)
        # Never take a longer detour solely because the nearest step is missing.
        return approach[1] if len(approach) > 1 and approach[1] in visible else None

    recent = set(recent_docks)
    # Look ahead without revisiting a node in the same prospective path. The
    # number of distinct, non-recent docks is the main objective.
    def future_value(node, seen, depth):
        gain = 1 if node in work_points and node not in recent else 0
        if depth == 0:
            return gain
        following = [n for n in neighbors[node] if n not in seen]
        return gain + (max((future_value(n, seen | {n}, depth - 1)
                            for n in following), default=0))

    def reachable_docks(start):
        blocked = {current} | recent
        queue = deque([start])
        seen = blocked | {start}
        count = 0
        while queue:
            node = queue.popleft()
            count += node in work_points and node not in recent
            for neighbor in neighbors[node] - seen:
                seen.add(neighbor)
                queue.append(neighbor)
        return count

    alternatives = [node for node in visible if node not in recent]
    if alternatives:
        visible = set(alternatives)
    forward = [node for node in visible if node != previous]
    if forward:
        visible = set(forward)
    return max(visible, key=lambda node: (
        reachable_docks(node),
        future_value(node, {current, node}, 8),
        node in work_points and node not in recent,
        -_distance_to_route(node, work_points - recent, neighbors),
        -int(node[1:])))
