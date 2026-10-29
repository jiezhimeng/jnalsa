"""Scale-independent matching for small transparent raster icons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np


ImageSource = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class IconFeatures:
    rgb: np.ndarray
    alpha: np.ndarray
    edge: np.ndarray


@dataclass(frozen=True)
class IconTemplate:
    name: str
    path: str
    rgba: np.ndarray
    features: IconFeatures


@dataclass(frozen=True)
class IconScore:
    name: str
    path: str
    similarity: float
    shape: float
    color: float
    edge: float


@dataclass(frozen=True)
class IconMatch:
    """One classification result and its ranked candidates."""

    name: str
    accepted: bool
    confidence: float
    results: tuple[IconScore, ...]

    @property
    def best(self) -> Optional[IconScore]:
        return self.results[0] if self.results else None


@dataclass(frozen=True)
class IconDetection:
    """One icon found inside a larger screen image."""

    name: str
    path: str
    similarity: float
    confidence: float
    center: tuple[int, int]
    box: tuple[int, int, int, int]
    scale: float


def read_icon(path: Union[str, Path]) -> np.ndarray:
    """Read a Unicode path and return an RGBA uint8 image."""
    path = Path(path)
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f'Unable to read icon: {path}: {exc}') from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Invalid icon image: {path}')
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        alpha = np.full(image.shape, 255, dtype=np.uint8)
    elif image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = image[:, :, 3]
    else:
        raise ValueError(f'Unsupported icon channels: {path}: {image.shape}')
    return np.dstack((rgb, alpha))


class TransparentIconMatcher:
    """Match transparent icons independent of canvas size and alpha padding.

    Templates are normalized once when loaded, so repeated matching only
    computes the query features.  Confidence is a heuristic based on both the
    absolute similarity and the margin over the runner-up; it is not a learned
    probability.
    """

    def __init__(
        self,
        icon_directory: Optional[Union[str, Path]] = None,
        *,
        recursive: bool = False,
        threshold: float = 0.60,
        canvas_size: int = 96,
        content_size: int = 80,
        alpha_cutoff: int = 3,
    ):
        if canvas_size < 16:
            raise ValueError('canvas_size must be at least 16')
        if not 1 <= content_size <= canvas_size:
            raise ValueError('content_size must be within canvas_size')
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.canvas_size = int(canvas_size)
        self.content_size = int(content_size)
        self.alpha_cutoff = max(0, min(254, int(alpha_cutoff)))
        self.templates = []
        self.skipped = []
        self._scaled_template_cache = {}
        if icon_directory is not None:
            self.load_directory(icon_directory, recursive=recursive)

    @staticmethod
    def _as_rgba(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            alpha = np.full(image.shape, 255, dtype=np.uint8)
            return np.dstack((rgb, alpha)).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError('Icon array must have shape HxW, HxWx3 or HxWx4')
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            return image
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return np.dstack((image, alpha))

    def _read_source(self, source: ImageSource) -> np.ndarray:
        if isinstance(source, (str, Path)):
            return read_icon(source)
        # Arrays use the project's RGB/RGBA convention, unlike cv2.imread.
        return self._as_rgba(source)

    def normalize(self, source: ImageSource) -> IconFeatures:
        rgba = self._read_source(source)
        alpha_u8 = rgba[:, :, 3]
        points = cv2.findNonZero(
            np.uint8(alpha_u8 > self.alpha_cutoff) * 255
        )
        if points is None:
            raise ValueError('Icon is fully transparent')
        left, top, width, height = cv2.boundingRect(points)
        rgb = rgba[top:top + height, left:left + width, :3].astype(np.float32)
        alpha = (
            alpha_u8[top:top + height, left:left + width].astype(np.float32)
            / 255.0
        )

        scale = min(self.content_size / width, self.content_size / height)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        premultiplied = rgb * alpha[:, :, None]
        resized_alpha = cv2.resize(
            alpha, (target_width, target_height), interpolation=interpolation
        )
        resized_color = cv2.resize(
            premultiplied,
            (target_width, target_height),
            interpolation=interpolation,
        )
        if resized_alpha.ndim == 0:
            resized_alpha = resized_alpha.reshape((1, 1))
            resized_color = resized_color.reshape((1, 1, 3))
        resized_rgb = np.clip(
            resized_color / np.maximum(resized_alpha[:, :, None], 1e-6),
            0,
            255,
        )

        canvas_rgb = np.zeros(
            (self.canvas_size, self.canvas_size, 3), dtype=np.float32
        )
        canvas_alpha = np.zeros(
            (self.canvas_size, self.canvas_size), dtype=np.float32
        )
        x = (self.canvas_size - target_width) // 2
        y = (self.canvas_size - target_height) // 2
        canvas_rgb[y:y + target_height, x:x + target_width] = resized_rgb
        canvas_alpha[y:y + target_height, x:x + target_width] = resized_alpha

        visible = np.uint8(canvas_alpha >= 0.10) * 255
        silhouette_edge = cv2.Canny(visible, 30, 100)
        gray = cv2.cvtColor(np.uint8(canvas_rgb), cv2.COLOR_RGB2GRAY)
        detail_edge = cv2.Canny(gray, 45, 130)
        detail_edge[canvas_alpha < 0.10] = 0
        edge = cv2.bitwise_or(silhouette_edge, detail_edge)
        return IconFeatures(canvas_rgb, canvas_alpha, edge)

    @staticmethod
    def _soft_iou(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.maximum(left, right).sum())
        if denominator <= 1e-6:
            return 0.0
        return float(np.minimum(left, right).sum() / denominator)

    @staticmethod
    def _edge_f1(left: np.ndarray, right: np.ndarray) -> float:
        kernel = np.ones((3, 3), dtype=np.uint8)
        left_mask = left > 0
        right_mask = right > 0
        if not left_mask.any() or not right_mask.any():
            return 0.0
        left_near = cv2.dilate(np.uint8(left_mask), kernel) > 0
        right_near = cv2.dilate(np.uint8(right_mask), kernel) > 0
        precision = (
            float(np.count_nonzero(left_mask & right_near))
            / np.count_nonzero(left_mask)
        )
        recall = (
            float(np.count_nonzero(right_mask & left_near))
            / np.count_nonzero(right_mask)
        )
        return 2.0 * precision * recall / max(1e-6, precision + recall)

    @classmethod
    def compare(
        cls, query: IconFeatures, candidate: IconFeatures
    ) -> tuple[float, float, float, float]:
        shape_score = cls._soft_iou(query.alpha, candidate.alpha)
        overlap = np.minimum(query.alpha, candidate.alpha)
        weight = float(overlap.sum())
        if weight <= 1e-6:
            color_score = 0.0
        else:
            difference = query.rgb - candidate.rgb
            mse = float(
                (difference * difference * overlap[:, :, None]).sum()
                / (weight * 3.0)
            )
            color_score = math.exp(-((math.sqrt(max(0.0, mse)) / 55.0) ** 2))
        edge_score = cls._edge_f1(query.edge, candidate.edge)
        similarity = (
            0.30 * shape_score + 0.45 * color_score + 0.25 * edge_score
        )
        return tuple(map(float, (
            similarity, shape_score, color_score, edge_score,
        )))

    @staticmethod
    def icon_paths(
        directory: Union[str, Path], recursive: bool = False
    ) -> Iterable[Path]:
        directory = Path(directory)
        pattern = '**/*.png' if recursive else '*.png'
        yield from sorted(
            directory.glob(pattern), key=lambda item: str(item).casefold()
        )

    def add(self, name: str, source: ImageSource, path: str = '') -> None:
        rgba = self._read_source(source)
        self.templates.append(IconTemplate(
            name=str(name),
            path=str(path),
            rgba=rgba,
            features=self.normalize(rgba),
        ))
        self._scaled_template_cache.clear()

    def load_directory(
        self, directory: Union[str, Path], *, recursive: bool = False
    ) -> int:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f'Icon directory does not exist: {directory}')
        self.templates = []
        self.skipped = []
        self._scaled_template_cache.clear()
        for path in self.icon_paths(directory, recursive):
            try:
                self.add(path.stem, path, str(path.resolve()))
            except ValueError as exc:
                self.skipped.append((str(path), str(exc)))
        return len(self.templates)

    def match(self, query: ImageSource, *, top: int = 5) -> IconMatch:
        if not self.templates:
            raise ValueError('No icon templates are loaded')
        query_features = self.normalize(query)
        results = []
        for template in self.templates:
            similarity, shape, color, edge = self.compare(
                query_features, template.features
            )
            results.append(IconScore(
                name=template.name,
                path=template.path,
                similarity=similarity,
                shape=shape,
                color=color,
                edge=edge,
            ))
        results.sort(key=lambda item: item.similarity, reverse=True)
        results = results[:max(1, int(top))]
        best = results[0].similarity
        second = results[1].similarity if len(results) > 1 else 0.0
        separation = 1.0 / (
            1.0 + math.exp(-(best - second - 0.025) / 0.035)
        )
        confidence = max(0.0, min(1.0, best * separation))
        accepted = bool(best >= self.threshold)
        return IconMatch(
            name=results[0].name if accepted else 'unknown',
            accepted=accepted,
            confidence=float(confidence),
            results=tuple(results),
        )

    def _foreground(self, rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha = rgba[:, :, 3]
        points = cv2.findNonZero(
            np.uint8(alpha > self.alpha_cutoff) * 255
        )
        if points is None:
            raise ValueError('Icon is fully transparent')
        left, top, width, height = cv2.boundingRect(points)
        return (
            rgba[top:top + height, left:left + width, :3],
            alpha[top:top + height, left:left + width],
        )

    def _scaled_template(
        self, template: IconTemplate, scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cached RGB, mask, gray and edge images at one scale."""
        key = (id(template.rgba), round(float(scale), 6))
        cached = self._scaled_template_cache.get(key)
        if cached is not None:
            return cached
        icon_rgb, icon_alpha = self._foreground(template.rgba)
        source_height, source_width = icon_alpha.shape
        width = max(4, int(round(source_width * scale)))
        height = max(4, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized_rgb = cv2.resize(
            icon_rgb, (width, height), interpolation=interpolation
        )
        resized_alpha = cv2.resize(
            icon_alpha, (width, height), interpolation=cv2.INTER_LINEAR
        )
        mask = np.uint8(resized_alpha >= 160) * 255
        if np.count_nonzero(mask) < 12:
            mask = np.uint8(resized_alpha > self.alpha_cutoff) * 255
        resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        resized_edge = cv2.Canny(resized_gray, 45, 130)
        resized_edge[mask == 0] = 0
        cached = (resized_rgb, mask, resized_gray, resized_edge)
        self._scaled_template_cache[key] = cached
        return cached

    @staticmethod
    def _scales(
        minimum: float, maximum: float, step: float
    ) -> tuple[float, ...]:
        minimum = max(0.05, float(minimum))
        maximum = max(minimum, float(maximum))
        step = max(1.01, float(step))
        values = []
        value = minimum
        while value <= maximum * 1.0001:
            values.append(value)
            value *= step
        if not values or values[-1] < maximum * 0.98:
            values.append(maximum)
        return tuple(values)

    @staticmethod
    def _iou(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if not intersection:
            return 0.0
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        return intersection / max(1, left_area + right_area - intersection)

    @staticmethod
    def _candidate_locations(
        score_map: np.ndarray, count: int, width: int, height: int
    ) -> list[tuple[float, tuple[int, int]]]:
        """Return separated maxima without sorting the complete score map."""
        work = np.nan_to_num(
            score_map.copy(), nan=-1.0, posinf=-1.0, neginf=-1.0
        )
        result = []
        radius_x = max(2, width // 3)
        radius_y = max(2, height // 3)
        for _ in range(max(1, int(count))):
            _, score, _, location = cv2.minMaxLoc(work)
            if score < 0:
                break
            result.append((float(score), location))
            x, y = location
            work[
                max(0, y - radius_y):min(work.shape[0], y + radius_y + 1),
                max(0, x - radius_x):min(work.shape[1], x + radius_x + 1),
            ] = -1.0
        return result

    def locate_all(
        self,
        screen: ImageSource,
        *,
        threshold: float = 0.78,
        min_scale: float = 0.80,
        max_scale: float = 1.20,
        scale_step: float = 1.20,
        max_per_icon: int = 1,
        candidates_per_scale: int = 3,
        nms_iou: float = 0.35,
    ) -> list[IconDetection]:
        """Find loaded transparent icons inside a larger RGB/RGBA image.

        Alpha-masked, multi-scale grayscale matching proposes locations.  RGB
        error at each proposal then rejects similarly shaped icons with the
        wrong colour.  Cross-template non-maximum suppression guarantees that
        one screen object is not returned under several file names.
        """
        if not self.templates:
            raise ValueError('No icon templates are loaded')
        screen_rgb = self._read_source(screen)[:, :, :3]
        screen_gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
        screen_height, screen_width = screen_gray.shape
        threshold = max(0.0, min(1.0, float(threshold)))
        scales = self._scales(min_scale, max_scale, scale_step)
        detections = []

        for template in self.templates:
            icon_candidates = []
            for scale in scales:
                (
                    resized_rgb, mask, resized_gray, resized_edge,
                ) = self._scaled_template(template, scale)
                height, width = mask.shape
                if width > screen_width or height > screen_height:
                    continue
                difference = cv2.matchTemplate(
                    screen_gray,
                    resized_gray,
                    cv2.TM_SQDIFF_NORMED,
                    mask=mask,
                )
                gray_score_map = 1.0 - np.nan_to_num(
                    difference, nan=2.0, posinf=2.0, neginf=2.0
                )
                for gray_score, (x, y) in self._candidate_locations(
                    gray_score_map, candidates_per_scale, width, height
                ):
                    patch = screen_rgb[y:y + height, x:x + width].astype(
                        np.float32
                    )
                    weights = mask.astype(np.float32) / 255.0
                    weight = float(weights.sum())
                    if weight <= 1e-6:
                        continue
                    delta = patch - resized_rgb.astype(np.float32)
                    mse = float(
                        (delta * delta * weights[:, :, None]).sum()
                        / (weight * 3.0)
                    )
                    color_score = math.exp(
                        -((math.sqrt(max(0.0, mse)) / 70.0) ** 2)
                    )
                    patch_gray = cv2.cvtColor(
                        np.uint8(patch), cv2.COLOR_RGB2GRAY
                    )
                    patch_edge = cv2.Canny(patch_gray, 45, 130)
                    patch_edge[mask == 0] = 0
                    edge_score = self._edge_f1(
                        resized_edge, patch_edge
                    )
                    similarity = (
                        0.3375 * gray_score
                        + 0.4125 * color_score
                        + 0.25 * edge_score
                    )
                    if similarity < threshold:
                        continue
                    box = (x, y, x + width, y + height)
                    icon_candidates.append(IconDetection(
                        name=template.name,
                        path=template.path,
                        similarity=float(similarity),
                        confidence=float(similarity),
                        center=(x + width // 2, y + height // 2),
                        box=box,
                        scale=float(scale),
                    ))

            icon_candidates.sort(
                key=lambda item: item.similarity, reverse=True
            )
            kept = []
            for candidate in icon_candidates:
                if any(
                    self._iou(candidate.box, old.box) > nms_iou
                    for old in kept
                ):
                    continue
                kept.append(candidate)
                if len(kept) >= max(1, int(max_per_icon)):
                    break
            detections.extend(kept)

        # The same object can score against multiple visually similar files.
        detections.sort(key=lambda item: item.similarity, reverse=True)
        kept = []
        for candidate in detections:
            if any(
                self._iou(candidate.box, old.box) > nms_iou
                for old in kept
            ):
                continue
            kept.append(candidate)
        return kept

    def track(
        self,
        screen: ImageSource,
        previous: Iterable[IconDetection],
        *,
        threshold: float = 0.72,
        search_radius: int = 12,
    ) -> list[IconDetection]:
        """Quickly re-locate previous detections in a new screen frame.

        This is the high-frequency counterpart to :meth:`locate_all`.  It
        checks only the previously identified template and only around its old
        position.  Call ``locate_all`` again when this method loses detections
        or when the page layout changes.
        """
        if not self.templates:
            raise ValueError('No icon templates are loaded')
        screen_rgb = self._read_source(screen)[:, :, :3]
        screen_gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
        screen_height, screen_width = screen_gray.shape
        threshold = max(0.0, min(1.0, float(threshold)))
        radius = max(0, int(search_radius))
        by_name = {template.name: template for template in self.templates}
        tracked = []

        for old in previous:
            template = by_name.get(old.name)
            if template is None:
                continue
            resized_rgb, mask, resized_gray, resized_edge = (
                self._scaled_template(template, old.scale)
            )
            height, width = mask.shape
            left = max(0, int(old.box[0]) - radius)
            top = max(0, int(old.box[1]) - radius)
            right = min(screen_width, int(old.box[2]) + radius)
            bottom = min(screen_height, int(old.box[3]) + radius)
            search = screen_gray[top:bottom, left:right]
            if search.shape[0] < height or search.shape[1] < width:
                continue
            difference = cv2.matchTemplate(
                search,
                resized_gray,
                cv2.TM_SQDIFF_NORMED,
                mask=mask,
            )
            difference = np.nan_to_num(
                difference, nan=2.0, posinf=2.0, neginf=2.0
            )
            minimum, _, location, _ = cv2.minMaxLoc(difference)
            gray_score = 1.0 - float(minimum)
            x = left + int(location[0])
            y = top + int(location[1])
            patch = screen_rgb[y:y + height, x:x + width].astype(
                np.float32
            )
            weights = mask.astype(np.float32) / 255.0
            weight = float(weights.sum())
            if weight <= 1e-6:
                continue
            delta = patch - resized_rgb.astype(np.float32)
            mse = float(
                (delta * delta * weights[:, :, None]).sum()
                / (weight * 3.0)
            )
            color_score = math.exp(
                -((math.sqrt(max(0.0, mse)) / 70.0) ** 2)
            )
            patch_gray = cv2.cvtColor(
                np.uint8(patch), cv2.COLOR_RGB2GRAY
            )
            patch_edge = cv2.Canny(patch_gray, 45, 130)
            patch_edge[mask == 0] = 0
            edge_score = self._edge_f1(resized_edge, patch_edge)
            similarity = (
                0.3375 * gray_score
                + 0.4125 * color_score
                + 0.25 * edge_score
            )
            if similarity < threshold:
                continue
            tracked.append(IconDetection(
                name=template.name,
                path=template.path,
                similarity=float(similarity),
                confidence=float(similarity),
                center=(x + width // 2, y + height // 2),
                box=(x, y, x + width, y + height),
                scale=float(old.scale),
            ))
        tracked.sort(key=lambda item: item.similarity, reverse=True)
        return tracked
