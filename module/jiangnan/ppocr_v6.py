"""PP-OCRv6 medium-only bridge for Jiangnan modules.

The main project is pinned to Python 3.7 while PaddleOCR 3.7 runs in a modern
Python environment.  This module keeps one long-lived child process and exposes
the tiny subset of the old CnOCR interface used by :mod:`module.ocr.ocr`.
"""

import atexit
import base64
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2

from module.logger import logger


class PPOcrV6Model:
    READY_TIMEOUT = 300
    REQUEST_TIMEOUT = 60
    PROTOCOL_PREFIX = 'PPOCR_JSON '
    MIN_CONFIDENCE = 0.25
    CACHE_LIMIT = 1024

    def __init__(self, model_name):
        self.MODEL_NAME = str(model_name)
        self._process = None
        self._lock = threading.RLock()
        self._request_id = 0
        self._alphabet = None
        self._result_cache = OrderedDict()
        self._message_queue = None
        self._reader_thread = None

    @staticmethod
    def _root():
        return Path(__file__).resolve().parents[2]

    def _command(self):
        root = self._root()
        python = root / '.venv' / 'Scripts' / 'python.exe'
        server = root / 'module' / 'jiangnan' / 'ppocr_v6_server.py'
        if not python.exists():
            raise RuntimeError(
                'PP-OCRv6 runtime is missing: run '
                'python -m venv .venv and install paddlepaddle==3.3.1, '
                'paddleocr==3.7.0'
            )
        return [str(python), '-u', str(server)]

    def _read_message(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is None:
                raise RuntimeError('PP-OCRv6 process is not running')
            if self._message_queue is None:
                raise RuntimeError('PP-OCRv6 output reader is not running')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._message_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f'PP-OCRv6 process exited with '
                        f'code {self._process.returncode}'
                    )
                continue
            if line is None:
                raise RuntimeError(
                    f'PP-OCRv6 output closed; process code '
                    f'{self._process.poll()}'
                )
            line = line.strip()
            if not line.startswith(self.PROTOCOL_PREFIX):
                # Paddle/oneDNN prints several harmless capability messages for
                # every inference.  Keep the application log useful and only
                # surface worker output that can explain an actual failure.
                if line and any(marker in line for marker in (
                        'ERROR', 'Error', 'Traceback', 'Exception')):
                    logger.warning(f'PP-OCRv6 worker: {line}')
                continue
            return json.loads(line[len(self.PROTOCOL_PREFIX):])
        raise TimeoutError('PP-OCRv6 response timeout')

    @staticmethod
    def _collect_output(process, messages):
        """Read blocking pipe calls off-thread so request deadlines are real."""
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        root = self._root()
        environment = os.environ.copy()
        runtime_home = root / 'bin' / 'ppocr_runtime'
        runtime_home.mkdir(parents=True, exist_ok=True)
        # Paddle 仍会在 import 阶段创建 ``~/.cache/paddle``，即使模型
        # 通过 model_dir 显式指定。只为子进程重定向 USERPROFILE，避免
        # 写入用户目录，也不改动主进程的 HOME/CODEX_HOME。
        environment['USERPROFILE'] = str(runtime_home)
        environment.setdefault('GLOG_minloglevel', '2')
        environment.setdefault('FLAGS_minloglevel', '2')
        environment.setdefault('PADDLE_LOG_LEVEL', 'ERROR')
        environment.setdefault('PADDLE_PDX_MODEL_SOURCE', 'BOS')
        environment.setdefault(
            'PADDLE_PDX_CACHE_HOME',
            str(root / 'bin' / 'ppocr_models'),
        )
        environment['PPOCR_MODEL_NAME'] = self.MODEL_NAME
        environment['PYTHONIOENCODING'] = 'utf-8'
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        logger.info(
            f'Loading OCR model in worker: {self.MODEL_NAME}'
        )
        self._process = subprocess.Popen(
            self._command(),
            cwd=str(root),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=flags,
        )
        self._message_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._collect_output,
            args=(self._process, self._message_queue),
            name=f'PP-OCRv6 output {self.MODEL_NAME}',
            daemon=True,
        )
        self._reader_thread.start()
        message = self._read_message(self.READY_TIMEOUT)
        if message.get('event') != 'ready':
            raise RuntimeError(
                f'PP-OCRv6 worker failed to start: {message}'
            )
        logger.info(
            f'PP-OCRv6 worker ready: {message.get("model")}'
        )

    @staticmethod
    def _encode(image):
        success, data = cv2.imencode('.png', image)
        if not success:
            raise ValueError('Unable to encode OCR image')
        return base64.b64encode(data.tobytes()).decode('ascii')

    def _predict(self, images, alphabet=None, with_scores=False):
        with self._lock:
            images = list(images)
            alphabet_key = '__raw_scored__' if with_scores else ''.join(alphabet) if alphabet else ''
            keys = [(
                alphabet_key,
                tuple(image.shape),
                str(image.dtype),
                hashlib.sha1(image.tobytes()).digest(),
            ) for image in images]
            texts = [None] * len(images)
            # 同一批 OCR 里可能包含完全相同的空白格/重复裁剪。旧逻辑在
            # LRU 尚未写入前会把它们全部发给 Paddle；按缓存键先去重，
            # 每个唯一图块只推理一次，再把结果回填到全部原始位置。
            missing = OrderedDict()
            for index, (key, image) in enumerate(zip(keys, images)):
                cached = self._result_cache.get(key)
                if cached is None:
                    item = missing.get(key)
                    if item is None:
                        missing[key] = [image, [index]]
                    else:
                        item[1].append(index)
                else:
                    self._result_cache.move_to_end(key)
                    texts[index] = cached

            if not missing:
                return texts

            missing_keys = list(missing)
            encoded_images = [
                self._encode(cv2.cvtColor(missing[key][0], cv2.COLOR_RGB2BGR)
                             if with_scores and missing[key][0].ndim == 3
                             else missing[key][0]) for key in missing_keys
            ]
            response = None
            # Paddle's Windows native backend can terminate with an access
            # violation instead of raising a Python exception.  Restart the
            # isolated worker and retry only the interrupted OCR request, so a
            # long warehouse scan is not discarded.
            for attempt in range(2):
                try:
                    self._start()
                    self._request_id += 1
                    request_id = self._request_id
                    request = {
                        'id': request_id,
                        'images': encoded_images,
                    }
                    self._process.stdin.write(
                        json.dumps(request, ensure_ascii=True) + '\n'
                    )
                    self._process.stdin.flush()
                    response = self._read_message(self.REQUEST_TIMEOUT)
                    if response.get('id') != request_id:
                        raise RuntimeError(
                            f'PP-OCRv6 response id mismatch: {response}'
                        )
                    if response.get('error'):
                        raise RuntimeError(response['error'])
                    break
                except (BrokenPipeError, OSError, RuntimeError, TimeoutError) as exc:
                    self.close()
                    if attempt:
                        raise
                    logger.warning(
                        f'PP-OCRv6 worker interrupted ({exc}); '
                        f'restarting and retrying current request once'
                    )

            predicted = [
                str(value) for value in response.get('texts', [])
            ]
            scores = response.get('scores', [])
            if with_scores:
                if len(predicted) != len(missing_keys) or len(scores) != len(missing_keys):
                    raise ValueError('Incomplete scored PP-OCR response')
                scored = []
                for value, score in zip(predicted, scores):
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = -1.0
                    scored.append((value, score))
                for key, value in zip(missing_keys, scored):
                    for target in missing[key][1]:
                        texts[target] = value
                    self._result_cache[key] = value
                    self._result_cache.move_to_end(key)
                while len(self._result_cache) > self.CACHE_LIMIT:
                    self._result_cache.popitem(last=False)
                return texts
            for index, score in enumerate(scores):
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    continue
                if index < len(predicted) and score < self.MIN_CONFIDENCE:
                    # 低置信度按空串交给调用方复查；级联模式下也可触发
                    # 下一模型。非空误识别保留为 INFO，最终业务失败
                    # 再由调用方告警。
                    if predicted[index]:
                        logger.info(
                            f'{self.MODEL_NAME} low confidence rejected: '
                            f'text={predicted[index]!r}, score={score:.3f}'
                        )
                    predicted[index] = ''
            if alphabet:
                # PP-OCR 对单个窄数字最常见的结果是把 0 读成 o/O、
                # 把 1 读成 I/l。旧代码先按数字字母表过滤，导致这些
                # 字符在 Digit.after_process 有机会修正前就变成空串。
                # 只在调用者明确要求纯数字时提前做形似字符转换，中文
                # 和普通文本 OCR 完全不受影响。
                allowed = set(alphabet)
                if allowed and allowed <= set('0123456789'):
                    digit_glyphs = str.maketrans({
                        'o': '0', 'O': '0', 'D': '0',
                        'I': '1', 'l': '1', '|': '1',
                        'S': '5', 's': '5', 'B': '8',
                    })
                    predicted = [
                        value.translate(digit_glyphs)
                        for value in predicted
                    ]
                predicted = [
                    ''.join(char for char in value if char in allowed)
                    for value in predicted
                ]
            predicted.extend([''] * (len(missing_keys) - len(predicted)))
            for key, value in zip(missing_keys, predicted):
                for target in missing[key][1]:
                    texts[target] = value
                self._result_cache[key] = value
                self._result_cache.move_to_end(key)
            while len(self._result_cache) > self.CACHE_LIMIT:
                self._result_cache.popitem(last=False)
            return texts

    def set_cand_alphabet(self, cand_alphabet):
        self._alphabet = cand_alphabet

    def atomic_ocr_for_single_lines_scored(self, images):
        """Preserve raw text and confidence for strict digit confirmation."""
        return self._predict(images, with_scores=True)

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        alphabet = cand_alphabet if cand_alphabet is not None else self._alphabet
        return self._predict(list(img_list), alphabet=alphabet)

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        return self.atomic_ocr_for_single_lines(
            [img_fp], cand_alphabet=cand_alphabet
        )[0]

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        return self.atomic_ocr_for_single_line(
            img_fp, cand_alphabet=cand_alphabet
        )

    def ocr_for_single_lines(self, img_list):
        return self.atomic_ocr_for_single_lines(img_list)

    def ocr_for_single_line(self, img_fp):
        return self.atomic_ocr_for_single_line(img_fp)

    def ocr(self, img_fp):
        return self.atomic_ocr_for_single_line(img_fp)

    def close(self):
        process = self._process
        self._process = None
        reader = self._reader_thread
        self._reader_thread = None
        self._message_queue = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=3)
            except Exception:
                pass
        finally:
            for pipe in (process.stdin, process.stdout):
                try:
                    pipe.close()
                except Exception:
                    pass
            if reader is not None:
                reader.join(timeout=0.2)

class PPOcrV6CascadeModel:
    """Recognition adapter supporting a single model or a model cascade."""

    def __init__(self, models):
        self.models = tuple(models)
        self._alphabet = None

    def set_cand_alphabet(self, cand_alphabet):
        self._alphabet = cand_alphabet

    def atomic_ocr_for_single_lines_consensus(self, img_list, cand_alphabet=None):
        """Use the configured model; require model agreement in cascade mode.

        In medium-only mode, callers retain their original/enlarged crop or
        repeated-frame comparisons, without invoking the smaller models.
        """
        from collections import Counter
        images = list(img_list)
        if len(self.models) == 1:
            return self.atomic_ocr_for_single_lines(
                images, cand_alphabet=cand_alphabet,
            )
        votes = [Counter() for _ in images]
        for model in self.models:
            try:
                values = model._predict(images, alphabet=cand_alphabet)
            except Exception as exc:
                logger.warning(f'OCR confirmation tier unavailable: {model.MODEL_NAME}, {exc}')
                continue
            for index, value in enumerate(values[:len(images)]):
                if str(value or ''):
                    votes[index][str(value)] += 1
        return [counts.most_common(1)[0][0]
                if counts and counts.most_common(1)[0][1] >= 2 else ''
                for counts in votes]

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        images = list(img_list)
        alphabet = cand_alphabet if cand_alphabet is not None else self._alphabet
        results = [''] * len(images)
        unresolved = list(range(len(images)))
        for tier, model in enumerate(self.models):
            if not unresolved:
                break
            batch = [images[index] for index in unresolved]
            try:
                values = model._predict(batch, alphabet=alphabet)
            except Exception as exc:
                logger.warning(
                    f'OCR tier unavailable: {model.MODEL_NAME}, {exc}'
                )
                continue
            next_unresolved = []
            for index, value in zip(unresolved, values):
                value = str(value or '')
                if value:
                    results[index] = value
                else:
                    next_unresolved.append(index)
            if next_unresolved and tier + 1 < len(self.models):
                logger.info(
                    f'OCR escalate: {model.MODEL_NAME} -> '
                    f'{self.models[tier + 1].MODEL_NAME}, '
                    f'crops={len(next_unresolved)}'
                )
            unresolved = next_unresolved
        return results

    def atomic_ocr_for_single_lines_validated(
        self,
        img_list,
        validator,
        cand_alphabet=None,
    ):
        """Escalate non-empty but semantically invalid recognition results.

        The normal cascade can only tell that OCR returned some text.  Callers
        with a closed vocabulary (for example the city list) can use this
        method so a confident-looking wrong result from tiny is retried by
        small and then medium as well.
        """
        images = list(img_list)
        alphabet = cand_alphabet if cand_alphabet is not None else self._alphabet
        results = [''] * len(images)
        unresolved = list(range(len(images)))
        for tier, model in enumerate(self.models):
            if not unresolved:
                break
            batch = [images[index] for index in unresolved]
            try:
                values = model._predict(batch, alphabet=alphabet)
            except Exception as exc:
                logger.warning(
                    f'OCR tier unavailable: {model.MODEL_NAME}, {exc}'
                )
                continue
            next_unresolved = []
            for index, value in zip(unresolved, values):
                value = str(value or '')
                results[index] = value
                try:
                    valid = bool(value) and bool(validator(value, index))
                except Exception as exc:
                    logger.warning(f'OCR result validator failed: {exc}')
                    valid = False
                if not valid:
                    next_unresolved.append(index)
            if next_unresolved and tier + 1 < len(self.models):
                logger.info(
                    f'OCR semantic escalate: {model.MODEL_NAME} -> '
                    f'{self.models[tier + 1].MODEL_NAME}, '
                    f'crops={len(next_unresolved)}'
                )
            unresolved = next_unresolved
        return results

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        return self.atomic_ocr_for_single_lines(
            [img_fp], cand_alphabet=cand_alphabet
        )[0]

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        return self.atomic_ocr_for_single_line(
            img_fp, cand_alphabet=cand_alphabet
        )

    def ocr_for_single_lines(self, img_list):
        return self.atomic_ocr_for_single_lines(img_list)

    def ocr_for_single_line(self, img_fp):
        return self.atomic_ocr_for_single_line(img_fp)

    def ocr(self, img_fp):
        return self.atomic_ocr_for_single_line(img_fp)

    def close(self):
        for model in self.models:
            model.close()


PPOCR_V6_TINY = PPOcrV6Model('PP-OCRv6_tiny_rec')
PPOCR_V6_SMALL = PPOcrV6Model('PP-OCRv6_small_rec')
PPOCR_V6_MEDIUM = PPOcrV6Model('PP-OCRv6_medium_rec')
PPOCR_V6_CASCADE = PPOcrV6CascadeModel((
    PPOCR_V6_MEDIUM,
))
atexit.register(PPOCR_V6_CASCADE.close)
