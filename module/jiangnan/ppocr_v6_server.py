"""Python 3.13 worker hosting one requested PP-OCRv6 recognition model."""

import base64
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'BOS')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('FLAGS_minloglevel', '2')
os.environ.setdefault('PADDLE_LOG_LEVEL', 'ERROR')

import cv2
import numpy as np
from paddleocr import TextRecognition


PREFIX = 'PPOCR_JSON '
MODEL_NAME = os.environ.get(
    'PPOCR_MODEL_NAME', 'PP-OCRv6_medium_rec'
)
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / 'bin' / 'ppocr_models' / f'{MODEL_NAME}_infer'
)


def emit(value):
    sys.stdout.write(PREFIX + json.dumps(value, ensure_ascii=True) + '\n')
    sys.stdout.flush()


def decode_image(value):
    data = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unable to decode OCR image')
    return image


def result_value(result):
    payload = result.json
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = payload.get('res', payload)
    return (
        str(payload.get('rec_text', '')),
        float(payload.get('rec_score', 0.0)),
    )


def predict_ordered(model, images, batch_size=4):
    """Avoid padding every crop to the widest line; preserve caller order."""
    order = sorted(range(len(images)),
                   key=lambda index: images[index].shape[1] / images[index].shape[0])
    values = [None] * len(images)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        results = list(model.predict(input=[images[index] for index in indices],
                                     batch_size=len(indices)))
        if len(results) != len(indices):
            raise ValueError('PP-OCR result count does not match input crops')
        for index, result in zip(indices, results):
            values[index] = result_value(result)
    return values


def main():
    options = dict(
        model_name=MODEL_NAME,
        device='cpu',
        engine='paddle_static',
        # oneDNN is a little faster, but this Windows build has shown native
        # 0xC0000005 crashes during long batches.  Stability matters more than
        # the small gain because this worker is used by hours-long scans.
        enable_mkldnn=False,
        cpu_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    # 已打包模型使用固定目录；tiny/small 首次使用时若尚未安装，
    # 交给 PaddleX 的官方模型缓存下载，之后会直接复用本地缓存。
    if MODEL_DIR.exists():
        options['model_dir'] = str(MODEL_DIR)
    model = TextRecognition(**options)
    emit({'event': 'ready', 'model': MODEL_NAME})
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            images = [decode_image(value) for value in request['images']]
            values = predict_ordered(model, images)
            emit({
                'id': request.get('id'),
                'texts': [value[0] for value in values],
                'scores': [value[1] for value in values],
            })
        except Exception as exc:
            emit({
                'id': request.get('id') if request else None,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })


if __name__ == '__main__':
    main()
