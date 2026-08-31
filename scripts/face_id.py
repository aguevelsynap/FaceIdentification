#!/usr/bin/env python3
import argparse
import base64
import ctypes
import json
import os
import re
import socket
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import numpy as np

TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
STRIDES = (8, 16, 32)


class SharedState:
    def __init__(self, db_path):
        self.lock = threading.RLock()
        self.db_path = str(db_path)
        self.frame_jpeg = None
        self.frame = None
        self.rows = []
        self.fps = 0.0
        self.capture_fps = 0.0
        self.frame_count = 0
        self.last_inference_ms = 0.0
        self.last_error = ""
        self.current_app = "none"
        self.pending_app = None
        self.debounce_remaining = 0.0
        self.tv_connected = False
        self.tv_host = ""
        self.tv_mac = ""
        self.last_wol_time = 0.0
        self.target_user = "Aguevel"
        self.target_recognized = False
        self.matched_target = None
        self.kids_user = "kids,Mike"
        self.kids_recognized = False
        self.matched_kid = None
        self.running = True

    def set_frame(self, frame, jpeg):
        with self.lock:
            self.frame = frame.copy()
            self.frame_jpeg = jpeg

    def snapshot(self):
        with self.lock:
            return self.frame_jpeg, [dict(r) for r in self.rows], self.fps, self.capture_fps, self.frame_count, self.last_inference_ms, self.last_error


def require_symbols(lib, names):
    missing = [name for name in names if not hasattr(lib, name)]
    if missing:
        raise RuntimeError('Missing TFLite C API symbols: ' + ', '.join(missing))


class TFLiteModel:
    def __init__(self, path, threads=4):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.lib = ctypes.CDLL('/usr/lib/libtensorflow-lite.so')
        L = self.lib
        require_symbols(L, [
            'TfLiteModelCreateFromFile', 'TfLiteModelDelete',
            'TfLiteInterpreterOptionsCreate', 'TfLiteInterpreterOptionsDelete',
            'TfLiteInterpreterOptionsSetNumThreads',
            'TfLiteInterpreterCreate', 'TfLiteInterpreterDelete',
            'TfLiteInterpreterGetInputTensor',
            'TfLiteInterpreterGetOutputTensorCount',
            'TfLiteInterpreterGetOutputTensor',
            'TfLiteInterpreterAllocateTensors',
            'TfLiteInterpreterInvoke',
            'TfLiteTensorNumDims', 'TfLiteTensorDim',
            'TfLiteTensorData', 'TfLiteTensorByteSize',
        ])
        L.TfLiteModelCreateFromFile.argtypes = [ctypes.c_char_p]
        L.TfLiteModelCreateFromFile.restype = ctypes.c_void_p
        L.TfLiteModelDelete.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterOptionsCreate.restype = ctypes.c_void_p
        L.TfLiteInterpreterOptionsDelete.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterOptionsSetNumThreads.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.TfLiteInterpreterCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.TfLiteInterpreterCreate.restype = ctypes.c_void_p
        L.TfLiteInterpreterDelete.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterGetInputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.TfLiteInterpreterGetInputTensor.restype = ctypes.c_void_p
        L.TfLiteInterpreterGetOutputTensorCount.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterGetOutputTensorCount.restype = ctypes.c_int
        L.TfLiteInterpreterGetOutputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.TfLiteInterpreterGetOutputTensor.restype = ctypes.c_void_p
        L.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterAllocateTensors.restype = ctypes.c_int
        L.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]
        L.TfLiteInterpreterInvoke.restype = ctypes.c_int
        L.TfLiteTensorNumDims.argtypes = [ctypes.c_void_p]
        L.TfLiteTensorNumDims.restype = ctypes.c_int
        L.TfLiteTensorDim.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.TfLiteTensorDim.restype = ctypes.c_int
        L.TfLiteTensorData.argtypes = [ctypes.c_void_p]
        L.TfLiteTensorData.restype = ctypes.c_void_p
        L.TfLiteTensorByteSize.argtypes = [ctypes.c_void_p]
        L.TfLiteTensorByteSize.restype = ctypes.c_size_t

        self.model = L.TfLiteModelCreateFromFile(os.fsencode(self.path))
        if not self.model:
            raise RuntimeError('TfLiteModelCreateFromFile failed: ' + self.path)
        self.options = L.TfLiteInterpreterOptionsCreate()
        if not self.options:
            raise RuntimeError('TfLiteInterpreterOptionsCreate failed')
        L.TfLiteInterpreterOptionsSetNumThreads(self.options, int(threads))
        self.interpreter = L.TfLiteInterpreterCreate(self.model, self.options)
        if not self.interpreter:
            raise RuntimeError('TfLiteInterpreterCreate failed: ' + self.path)
        if L.TfLiteInterpreterAllocateTensors(self.interpreter) != 0:
            raise RuntimeError('TfLiteInterpreterAllocateTensors failed: ' + self.path)

    def __del__(self):
        try:
            if getattr(self, 'interpreter', None):
                self.lib.TfLiteInterpreterDelete(self.interpreter)
            if getattr(self, 'options', None):
                self.lib.TfLiteInterpreterOptionsDelete(self.options)
            if getattr(self, 'model', None):
                self.lib.TfLiteModelDelete(self.model)
        except Exception:
            pass

    def _shape(self, tensor):
        n = self.lib.TfLiteTensorNumDims(tensor)
        return tuple(self.lib.TfLiteTensorDim(tensor, i) for i in range(n))

    def set_input_f32(self, x):
        tensor = self.lib.TfLiteInterpreterGetInputTensor(self.interpreter, 0)
        expected = self._shape(tensor)
        if tuple(x.shape) != expected:
            raise RuntimeError(f'{self.path}: input shape {expected}, got {x.shape}')
        nbytes = self.lib.TfLiteTensorByteSize(tensor)
        if nbytes != x.nbytes:
            raise RuntimeError(f'{self.path}: input bytes {nbytes}, got {x.nbytes}')
        ctypes.memmove(self.lib.TfLiteTensorData(tensor), x.ctypes.data, x.nbytes)

    def invoke(self):
        if self.lib.TfLiteInterpreterInvoke(self.interpreter) != 0:
            raise RuntimeError('TfLiteInterpreterInvoke failed: ' + self.path)

    def outputs_f32(self):
        out = []
        count = self.lib.TfLiteInterpreterGetOutputTensorCount(self.interpreter)
        for i in range(count):
            tensor = self.lib.TfLiteInterpreterGetOutputTensor(self.interpreter, i)
            shape = self._shape(tensor)
            nbytes = self.lib.TfLiteTensorByteSize(tensor)
            address = self.lib.TfLiteTensorData(tensor)
            buf = (ctypes.c_ubyte * nbytes).from_address(address)
            out.append(np.frombuffer(buf, dtype=np.float32).copy().reshape(shape))
        return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a['box']
    bx1, by1, bx2, by2 = b['box']
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def detect_yunet(model, frame, score_threshold=0.6, nms_threshold=0.3):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    # Validated board/WSL YuNet TFLite contract: BGR, NHWC, float32, 0..255.
    x = resized.astype(np.float32)[None, ...]
    model.set_input_f32(x)
    model.invoke()
    outputs = model.outputs_f32()
    if len(outputs) != 12:
        raise RuntimeError(f'YuNet expected 12 outputs, got {len(outputs)}')
    faces = []
    for level, stride in enumerate(STRIDES):
        cls = outputs[level][0, :, 0]
        obj = outputs[3 + level][0, :, 0]
        boxes = outputs[6 + level][0]
        kps = outputs[9 + level][0]
        grid = 640 // stride
        scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
        if len(scores) != grid * grid:
            raise RuntimeError(f'YuNet stride {stride}: unexpected output size {len(scores)}')
        for idx, score in enumerate(scores):
            score = float(score)
            if score < score_threshold:
                continue
            gy, gx = divmod(idx, grid)
            cx = (gx + boxes[idx, 0]) * stride
            cy = (gy + boxes[idx, 1]) * stride
            bw = np.exp(boxes[idx, 2]) * stride
            bh = np.exp(boxes[idx, 3]) * stride
            landmarks = [
                ((gx + kps[idx, 2*k]) * stride, (gy + kps[idx, 2*k+1]) * stride)
                for k in range(5)
            ]
            faces.append({'score': score, 'box': [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], 'landmarks': landmarks})
    faces.sort(key=lambda f: f['score'], reverse=True)
    kept = []
    for face in faces:
        if all(iou(face, other) < nms_threshold for other in kept):
            kept.append(face)
    sx, sy = w / 640.0, h / 640.0
    for face in kept:
        x1, y1, x2, y2 = face['box']
        face['box'] = [x1*sx, y1*sy, x2*sx, y2*sy]
        face['landmarks'] = [(x*sx, y*sy) for x, y in face['landmarks']]
    return kept


def align_face(frame, landmarks):
    src = np.asarray(landmarks, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError('Could not compute face alignment transform')
    return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def sface_embedding(model, aligned_face):
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    model.set_input_f32(rgb[None, ...])
    model.invoke()
    embedding = model.outputs_f32()[0].reshape(-1)
    if embedding.size != 128:
        raise RuntimeError(f'SFace expected 128-D output, got {embedding.shape}')
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    return embedding.astype(np.float32)


def load_db(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError('face DB must be a JSON object')
    return data


def save_db(path, db):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(db, indent=2) + '\n')
    tmp.replace(p)


def best_match(embedding, db):
    best_name, best_score = None, -1.0
    for name, values in db.items():
        ref = np.asarray(values, dtype=np.float32).reshape(-1)
        if ref.size != embedding.size:
            continue
        ref /= max(float(np.linalg.norm(ref)), 1e-12)
        score = float(np.dot(embedding, ref))
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def annotate(frame, detections, threshold, db):
    out = frame.copy()
    rows = []
    for idx, d in enumerate(detections):
        aligned = align_face(frame, d['landmarks'])
        embedding = sface_embedding(_SFACE_MODEL, aligned)
        name, similarity = best_match(embedding, db) if db else (None, -1.0)
        identity = name if name is not None and similarity >= threshold else 'Unknown'
        x1, y1, x2, y2 = [int(v) for v in d['box']]
        color = (0, 200, 0) if identity != 'Unknown' else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{identity} {similarity:.3f}' if name is not None else f'Unknown {d["score"]:.3f}'
        cv2.putText(out, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        for x, y in d['landmarks']:
            cv2.circle(out, (int(x), int(y)), 3, (255, 0, 255), -1)
        rows.append({
            'face_index': idx,
            'detector_score': d['score'],
            'identity': identity,
            'similarity': similarity,
            'embedding': embedding,
            'box': d['box'],
            'landmarks': d['landmarks'],
        })
    return out, rows


def make_rtp_writer(host, port, width, height, fps, bitrate=2000):
    pipeline = [
        'gst-launch-1.0', '-q',
        'fdsrc', 'fd=0', '!',
        'rawvideoparse', 'format=bgr',
        f'width={width}', f'height={height}', f'framerate={fps}/1', '!',
        'queue', 'max-size-buffers=2', 'leaky=downstream', '!',
        'videoconvert', '!',
        'videoscale', '!',
        'video/x-raw,width=640,height=360,format=I420', '!',
        'x264enc', 'tune=zerolatency', 'speed-preset=ultrafast',
        f'bitrate={bitrate}', f'key-int-max={fps}', '!',
        'video/x-h264,stream-format=byte-stream,alignment=au', '!',
        'rtph264pay', 'pt=96', 'config-interval=1', '!',
        'udpsink', f'host={host}', f'port={port}', 'sync=false', 'async=false',
    ]
    print('GStreamer RTP:', ' '.join(pipeline), flush=True)
    return subprocess.Popen(pipeline, stdin=subprocess.PIPE), pipeline


class WebHandler(BaseHTTPRequestHandler):
    state = None
    db_path = None
    threshold = 0.363
    tv = None

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            candidates = [
                Path(__file__).resolve().parent.parent / 'web' / 'index.html',
                Path(__file__).resolve().parent / 'web' / 'index.html',
                Path(__file__).resolve().parent / 'index.html',
                Path.cwd() / 'web' / 'index.html',
                Path.cwd() / 'index.html',
            ]
            html_path = next((p for p in candidates if p.is_file()), None)
            if html_path:
                body = html_path.read_bytes()
            else:
                body = b'<!doctype html><html><body><h1>Error: index.html not found</h1><p>Looked in candidate paths.</p></body></html>'
            self._send(HTTPStatus.OK, 'text/html; charset=utf-8', body)
            return
        if self.path == '/status':
            _, rows, fps, capfps, count, infer_ms, error = self.state.snapshot()
            with self.state.lock:
                payload = {
                    'running': self.state.running,
                    'fps': fps,
                    'capture_fps': capfps,
                    'frames': count,
                    'inference_ms': infer_ms,
                    'current_app': getattr(self.state, 'current_app', 'none'),
                    'tv': {
                        'connected': getattr(self.state, 'tv_connected', False),
                        'host': getattr(self.state, 'tv_host', ''),
                        'mac': getattr(self.state, 'tv_mac', ''),
                        'last_wol_ago': round(time.monotonic() - self.state.last_wol_time, 1) if getattr(self.state, 'last_wol_time', 0.0) > 0 else None,
                        'current_app': getattr(self.state, 'current_app', 'none'),
                        'pending_app': getattr(self.state, 'pending_app', None),
                        'debounce_remaining': getattr(self.state, 'debounce_remaining', 0.0),
                        'target_user': getattr(self.state, 'target_user', 'Aguevel'),
                        'target_recognized': getattr(self.state, 'target_recognized', False),
                        'matched_target': getattr(self.state, 'matched_target', None),
                        'kids_user': getattr(self.state, 'kids_user', 'kids,Mike'),
                        'kids_recognized': getattr(self.state, 'kids_recognized', False),
                        'matched_kid': getattr(self.state, 'matched_kid', None),
                    },
                    'enrolled_users': sorted(list(load_db(self.db_path).keys())),
                    'faces': [
                        {'index': r['face_index'], 'identity': r['identity'], 'similarity': r['similarity'], 'detector': r['detector_score']}
                        for r in rows
                    ],
                    'error': error,
                }
            self._send(HTTPStatus.OK, 'application/json', json.dumps(payload).encode())
            return
        if self.path == '/snapshot.jpg':
            jpeg, *_ = self.state.snapshot()
            if jpeg is None:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, 'text/plain; charset=utf-8', b'No frame yet')
            else:
                self._send(HTTPStatus.OK, 'image/jpeg', jpeg)
            return
        if self.path == '/stream.mjpg':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            last = None
            try:
                while self.state.running:
                    jpeg, *_ = self.state.snapshot()
                    if jpeg is not None and jpeg != last:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        last = jpeg
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(HTTPStatus.NOT_FOUND, 'text/plain; charset=utf-8', b'Not found')

    def do_POST(self):
        if self.path in ('/tv/wake', '/wake_tv'):
            if self.tv:
                success = self.tv.wake()
                self._send(HTTPStatus.OK, 'application/json', json.dumps({
                    'ok': True,
                    'message': 'Wake-on-LAN packet sent and wake sequence executed',
                    'mac': self.tv.mac,
                    'connected': self.tv.connected
                }).encode())
            else:
                self._send(HTTPStatus.OK, 'application/json', json.dumps({
                    'ok': False,
                    'message': 'Android TV integration is not enabled'
                }).encode())
            return
        if self.path in ('/flush', '/flush_db', '/clear_users'):
            save_db(self.db_path, {})
            print(f"[Web] Flushed all enrolled users from database: {self.db_path}", flush=True)
            self._send(HTTPStatus.OK, 'application/json', json.dumps({'ok': True, 'message': 'All enrolled users flushed'}).encode())
            return
        if self.path != '/enroll':
            self._send(HTTPStatus.NOT_FOUND, 'text/plain; charset=utf-8', b'Not found')
            return
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length)
        fields = parse_qs(body.decode())
        name = fields.get('name', [''])[0].strip()
        try:
            face_index = int(fields.get('face_index', ['0'])[0])
        except ValueError:
            face_index = 0
        if not name:
            self._send(HTTPStatus.BAD_REQUEST, 'text/plain; charset=utf-8', b'Name is required')
            return
        if any(ch in name for ch in '\r\n') or len(name) > 64:
            self._send(HTTPStatus.BAD_REQUEST, 'text/plain; charset=utf-8', b'Invalid name')
            return
        with self.state.lock:
            rows = list(self.state.rows)
            # If the browser submits the default/invalid index but there is exactly
            # one unknown face, enroll that face automatically. This also makes the
            # UI robust when the face list changes between status refreshes.
            if face_index < 0 or face_index >= len(rows):
                unknown = [r for r in rows if r.get('identity') == 'Unknown']
                if len(unknown) == 1:
                    row = unknown[0]
                else:
                    self._send(HTTPStatus.BAD_REQUEST, 'text/plain; charset=utf-8',
                               f'No usable face index. Current faces: {len(rows)}; unknown faces: {len(unknown)}'.encode())
                    return
            else:
                row = rows[face_index]
            if row.get('identity') != 'Unknown':
                self._send(HTTPStatus.BAD_REQUEST, 'text/plain; charset=utf-8', b'The selected face is not Unknown')
                return
            embedding = np.asarray(row['embedding'], dtype=np.float32)
        db = load_db(self.db_path)
        db[name] = embedding.tolist()
        save_db(self.db_path, db)
        self._send(HTTPStatus.OK, 'text/html; charset=utf-8', f'<html><body><h2>Enrolled {name}</h2><a href="/">Back</a></body></html>'.encode())

    def log_message(self, fmt, *args):
        return


def start_web_server(state, db_path, host, port, threshold, tv=None):
    WebHandler.state = state
    WebHandler.db_path = db_path
    WebHandler.threshold = threshold
    WebHandler.tv = tv
    server = ThreadingHTTPServer((host, port), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_camera(args, yunet, sface, tv=None):
    global _SFACE_MODEL
    _SFACE_MODEL = sface
    db = load_db(args.db)
    state = SharedState(args.db)
    server = None
    if args.web_port > 0:
        server = start_web_server(state, args.db, args.web_host, args.web_port, args.threshold, tv=tv)
        print(f'Web UI: http://{args.web_host}:{args.web_port}/', flush=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit('Cannot open camera: ' + args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.camera_width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.camera_height
    actual_fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or args.camera_fps

    writer, pipeline = make_rtp_writer(args.rtp_host, args.rtp_port, actual_w, actual_h, max(1, actual_fps), args.rtp_bitrate)
    print(f'camera: {args.camera} {actual_w}x{actual_h}@{actual_fps}', flush=True)
    print(f'RTP: {args.rtp_host}:{args.rtp_port}', flush=True)
    print(f'inference every {args.infer_every} frame(s)', flush=True)

    last_rows = []
    frame_count = 0
    report_start = time.monotonic()
    report_frames = 0
    last_infer_time = time.monotonic()

    # State tracking for Android TV app switching and debounce
    current_app = None
    pending_app = None
    pending_app_since = None
    debounce_seconds = max(0.1, float(args.debounce))
    target_user = args.target_user
    kids_user = getattr(args, 'kids_user', 'kids')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera frame read failed', flush=True)
                break
            frame_count += 1
            report_frames += 1
            now = time.monotonic()
            if frame_count == 1 or frame_count % max(1, args.infer_every) == 0:
                infer_start = time.monotonic()
                db = load_db(args.db)
                detections = detect_yunet(yunet, frame, args.score_threshold, args.nms_threshold)
                annotated, last_rows = annotate(frame, detections, args.threshold, db)
                state.last_inference_ms = (time.monotonic() - infer_start) * 1000.0
                last_infer_time = now

                # Detailed logging for detected faces and similarity scores
                if last_rows:
                    face_logs = [
                        f"Face {r['face_index']}: identity='{r['identity']}', similarity={r['similarity']:.3f}, detector_score={r['detector_score']:.3f}"
                        for r in last_rows
                    ]
                    print(f"[Face ID] Frame {frame_count}: Detected {len(last_rows)} face(s) -> " + "; ".join(face_logs), flush=True)
                else:
                    print(f"[Face ID] Frame {frame_count}: No faces detected", flush=True)
            else:
                annotated = frame.copy()
                for row in last_rows:
                    x1, y1, x2, y2 = [int(v) for v in row['box']]
                    color = (0, 200, 0) if row['identity'] != 'Unknown' else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = row['identity'] if row['identity'] != 'Unknown' else 'Unknown'
                    cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                    for x, y in row['landmarks']:
                        cv2.circle(annotated, (int(x), int(y)), 3, (255, 0, 255), -1)

            # Android TV app decision and debouncing logic
            raw_kids = getattr(args, 'kids_user', 'kids,Mike')
            kids_list = [k.strip().lower() for k in raw_kids.split(',') if k.strip()]
            if 'kids' not in kids_list:
                kids_list.append('kids')
            if 'kid' not in kids_list:
                kids_list.append('kid')

            target_list = [t.strip().lower() for t in target_user.split(',') if t.strip()]

            # Find matching identities from detected faces
            matched_kid = next((r['identity'] for r in last_rows if str(r.get('identity', '')).strip().lower() in kids_list), None)
            is_kids_recognized = matched_kid is not None

            matched_target = next((r['identity'] for r in last_rows if str(r.get('identity', '')).strip().lower() in target_list), None)
            is_target_recognized = matched_target is not None

            if is_kids_recognized:
                desired_app = "youtube_kids"
                reason = f"kids user '{matched_kid}' recognized"
            elif is_target_recognized:
                desired_app = "netflix"
                reason = f"target user '{matched_target}' recognized"
            else:
                desired_app = "youtube"
                reason = f"default (neither target nor kids recognized)"

            if tv is not None:
                if desired_app != current_app:
                    if pending_app != desired_app:
                        pending_app = desired_app
                        pending_app_since = now
                        print(
                            f"[App Switch] Candidate switch to '{desired_app}' ({reason}). "
                            f"Starting {debounce_seconds:.1f}s debounce timer...",
                            flush=True
                        )
                    else:
                        elapsed = now - pending_app_since
                        if elapsed >= debounce_seconds:
                            print(
                                f"[App Switch] Debounce threshold ({debounce_seconds:.1f}s) met. "
                                f"Switching app: '{current_app}' -> '{pending_app}'",
                                flush=True
                            )
                            if pending_app == "netflix":
                                tv.launch_netflix()
                            elif pending_app == "youtube_kids":
                                tv.launch_youtube_kids()
                            elif pending_app == "youtube":
                                tv.launch_youtube()
                            current_app = pending_app
                            state.current_app = current_app
                            pending_app = None
                            pending_app_since = None
                else:
                    if pending_app is not None:
                        print(
                            f"[App Switch] Reverted to current app '{current_app}'. Cancelling pending switch to '{pending_app}'.",
                            flush=True
                        )
                        pending_app = None
                        pending_app_since = None

            app_display = f'app={current_app or "none"}'
            debounce_rem = max(0.0, debounce_seconds - (now - pending_app_since)) if (pending_app is not None and pending_app_since is not None) else 0.0
            if pending_app is not None and pending_app_since is not None:
                app_display += f' -> {pending_app} ({debounce_rem:.1f}s)'

            with state.lock:
                state.current_app = current_app or "none"
                state.pending_app = pending_app
                state.debounce_remaining = round(debounce_rem, 1)
                state.tv_connected = getattr(tv, 'connected', False) if tv else False
                state.tv_host = tv.host if tv else "Disabled"
                state.tv_mac = getattr(tv, 'mac', '') if tv else ""
                state.last_wol_time = getattr(tv, 'last_wol_time', 0.0) if tv else 0.0
                state.target_user = target_user
                state.target_recognized = is_target_recognized
                state.matched_target = matched_target
                state.kids_user = raw_kids
                state.kids_recognized = is_kids_recognized
                state.matched_kid = matched_kid

            cv2.putText(annotated, f'face-id cpu  faces={len(last_rows)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'infer={state.last_inference_ms:.0f}ms', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, app_display, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            ok_jpg, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok_jpg:
                state.set_frame(annotated, jpg.tobytes())
            state.rows = [dict(r) for r in last_rows]
            state.frame_count = frame_count

            writer.stdin.write(np.ascontiguousarray(annotated).tobytes())
            writer.stdin.flush()

            elapsed = time.monotonic() - report_start
            if elapsed >= 5.0:
                fps = report_frames / max(elapsed, 1e-6)
                state.fps = fps
                state.capture_fps = actual_fps
                print(f'frames={frame_count} processed_fps={fps:.1f} faces={len(last_rows)} infer_ms={state.last_inference_ms:.1f} app={current_app}', flush=True)
                report_start = time.monotonic()
                report_frames = 0
    except (BrokenPipeError, OSError) as exc:
        state.last_error = str(exc)
        print('RTP pipeline closed:', exc, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        cap.release()
        try:
            if writer.stdin:
                writer.stdin.close()
            writer.wait(timeout=2)
        except Exception:
            try:
                writer.kill()
            except Exception:
                pass
        if tv:
            try:
                tv.disconnect()
            except Exception:
                pass
        if server:
            server.shutdown()
    return 0


def send_wake_on_lan(mac_address: str, target_ip: str = "255.255.255.255", port: int = 9) -> bool:
    """Send a Wake-on-LAN magic packet (102 bytes) to the specified MAC address."""
    clean_mac = re.sub(r'[^0-9A-Fa-f]', '', mac_address or '')
    if len(clean_mac) != 12:
        print(f"[WoL] Invalid MAC address format: '{mac_address}'", flush=True)
        return False
    mac_bytes = bytes.fromhex(clean_mac)
    packet = b'\xff' * 6 + mac_bytes * 16

    destinations = [("255.255.255.255", port)]
    if port != 7:
        destinations.append(("255.255.255.255", 7))

    clean_ip = (target_ip or "").split(':')[0].strip()
    if clean_ip and clean_ip != "255.255.255.255":
        parts = clean_ip.split('.')
        if len(parts) == 4:
            subnet_bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
            destinations.append((subnet_bcast, port))
        destinations.append((clean_ip, port))

    success = False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for dest_ip, dest_port in destinations:
            try:
                sock.sendto(packet, (dest_ip, dest_port))
                success = True
            except Exception as exc:
                print(f"[WoL] Failed sending to {dest_ip}:{dest_port} ({exc})", flush=True)

    formatted_mac = ":".join(clean_mac[i:i+2] for i in range(0, 12, 2))
    if success:
        print(f"[WoL] Sent magic packet for {formatted_mac} to {destinations}", flush=True)
    return success


def resolve_mac_address(ip: str) -> str:
    """Attempt to resolve MAC address for a given IP from ARP table or ip neigh."""
    clean_ip = (ip or "").split(':')[0].strip()
    # 1. Read /proc/net/arp (standard on Linux/SL2619)
    try:
        if os.path.exists('/proc/net/arp'):
            with open('/proc/net/arp', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] == clean_ip:
                        mac = parts[3].lower()
                        if mac != "00:00:00:00:00:00" and len(mac) == 17:
                            return mac
    except Exception:
        pass
    # 2. Try ip neigh show <ip>
    try:
        res = subprocess.run(['ip', 'neigh', 'show', clean_ip], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout:
            match = re.search(r'lladdr\s+([0-9a-fA-F:]{17})', res.stdout)
            if match:
                return match.group(1).lower()
    except Exception:
        pass
    # 3. Known default for user's network if 192.168.1.173
    if clean_ip == '192.168.1.173':
        return '70:54:b4:fe:8e:ca'
    return ""


class AndroidTV:
    def __init__(self, host: str, mac: str = None, port: int = 5038):
        self.host = host
        self.mac = mac.strip().lower() if mac else ""
        self.port = port
        self.connected = False
        self.last_wol_time = 0.0
        if not self.mac:
            self.mac = resolve_mac_address(self.host)
            if self.mac:
                print(f"[AndroidTV] Auto-resolved MAC for {self.host}: {self.mac}", flush=True)

    def _run(self, *args):
        """Run an adb command using isolated ADB server port and return its output."""
        cmd = ["adb", "-P", str(self.port), *args]
        print(f"[ADB] Executing: {' '.join(cmd)}", flush=True)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout:
                print(f"[ADB] stdout: {stdout}", flush=True)
            if stderr:
                print(f"[ADB] stderr: {stderr}", flush=True)
            return result
        except subprocess.TimeoutExpired:
            print(f"[ADB] Command timed out (10s): {' '.join(cmd)}", flush=True)
            return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="TimeoutExpired")
        except Exception as exc:
            print(f"[ADB] Command exception ({exc}): {' '.join(cmd)}", flush=True)
            return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr=str(exc))

    def wake_on_lan(self) -> bool:
        """Send Wake-on-LAN magic packet to TV."""
        if not self.mac:
            self.mac = resolve_mac_address(self.host)
        if not self.mac:
            print(f"[AndroidTV] Cannot send WoL: MAC address is unknown for {self.host}", flush=True)
            return False
        ip_only = self.host.split(":")[0]
        ok = send_wake_on_lan(self.mac, target_ip=ip_only)
        if ok:
            self.last_wol_time = time.monotonic()
        return ok

    def wake_screen(self):
        """Wake the TV screen if sleeping using ADB keyevent 224 (KEYCODE_WAKEUP)."""
        print("[AndroidTV] Sending KEYCODE_WAKEUP (224) to ensure display is ON...", flush=True)
        res = self._run("-s", self.host, "shell", "input", "keyevent", "224")
        return res.returncode == 0

    def wake(self) -> bool:
        """Full wake sequence: send WoL, try connecting ADB, and wake screen."""
        print(f"[AndroidTV] Initiating wake sequence for {self.host}...", flush=True)
        wol_sent = self.wake_on_lan()
        if not self.connected:
            print("[AndroidTV] Waiting 2s for TV network stack to awaken after WoL...", flush=True)
            time.sleep(2.0)
            self.connect(retry_with_wol=False)
        if self.connected:
            self.wake_screen()
        return wol_sent

    def connect(self, retry_with_wol: bool = True) -> bool:
        """Connect to the Android TV, attempting WoL if initial attempt fails."""
        print(f"[AndroidTV] Ensuring ADB server is started on port {self.port}...", flush=True)
        self._run("start-server")
        print(f"[AndroidTV] Connecting to {self.host}...", flush=True)
        result = self._run("connect", self.host)

        if result.returncode != 0 and retry_with_wol:
            print(f"[AndroidTV] ADB connection failed: {result.stderr.strip()}. Attempting Wake-on-LAN...", flush=True)
            self.wake_on_lan()
            time.sleep(2.5)
            result = self._run("connect", self.host)

        # Verify the device is listed
        devices = self._run("devices").stdout
        if self.host.split(":")[0] in devices:
            print(f"[AndroidTV] Connected successfully to {self.host}.", flush=True)
            self.connected = True
            self.wake_screen()
            if not self.mac:
                self._query_device_mac()
            return True

        if retry_with_wol and not self.connected:
            print(f"[AndroidTV] Device {self.host} not responding to ADB. Sending WoL and retrying...", flush=True)
            self.wake_on_lan()
            time.sleep(3.0)
            self._run("connect", self.host)
            devices = self._run("devices").stdout
            if self.host.split(":")[0] in devices:
                print(f"[AndroidTV] Connected successfully to {self.host} after WoL.", flush=True)
                self.connected = True
                self.wake_screen()
                if not self.mac:
                    self._query_device_mac()
                return True

        print(f"[AndroidTV] Device {self.host} not found in adb devices list.", flush=True)
        self.connected = False
        return False

    def _query_device_mac(self):
        """Query connected Android TV for its MAC address if unknown."""
        for iface in ('wlan0', 'eth0'):
            res = self._run("-s", self.host, "shell", "cat", f"/sys/class/net/{iface}/address")
            mac_val = res.stdout.strip().lower()
            if len(mac_val) == 17 and mac_val != "00:00:00:00:00:00":
                self.mac = mac_val
                print(f"[AndroidTV] Auto-detected TV MAC from {iface}: {self.mac}", flush=True)
                return
        res = self._run("-s", self.host, "shell", "getprop", "ro.boot.wifimacaddr")
        mac_val = res.stdout.strip().lower()
        if len(mac_val) == 17:
            self.mac = mac_val
            print(f"[AndroidTV] Auto-detected TV MAC from getprop: {self.mac}", flush=True)

    def ensure_connected(self) -> bool:
        """Ensure device is connected and awake before sending app commands."""
        if not self.connected:
            return self.connect(retry_with_wol=True)
        self.wake_screen()
        return True

    def launch_netflix(self):
        """Launch Netflix."""
        self.ensure_connected()
        print("[AndroidTV] Launching Netflix (com.netflix.ninja)...", flush=True)
        result = self._run(
            "-s", self.host,
            "shell",
            "monkey",
            "-p", "com.netflix.ninja",
            "-c", "android.intent.category.LAUNCHER",
            "1"
        )

        if result.returncode == 0:
            print("[AndroidTV] Netflix launched successfully.", flush=True)
        else:
            print(f"[AndroidTV] Failed to launch Netflix: {result.stderr.strip()}", flush=True)

    def launch_youtube(self):
        """Launch YouTube."""
        self.ensure_connected()
        print("[AndroidTV] Launching YouTube (com.google.android.youtube.tv)...", flush=True)
        result = self._run(
            "-s", self.host,
            "shell",
            "monkey",
            "-p", "com.google.android.youtube.tv",
            "-c", "android.intent.category.LAUNCHER",
            "1"
        )

        if result.returncode == 0:
            print("[AndroidTV] YouTube launched successfully.", flush=True)
        else:
            print(f"[AndroidTV] Failed to launch YouTube: {result.stderr.strip()}", flush=True)

    def launch_youtube_kids(self):
        """Launch YouTube Kids."""
        self.ensure_connected()
        print("[AndroidTV] Launching YouTube Kids (com.google.android.youtube.tvkids)...", flush=True)
        result = self._run(
            "-s", self.host,
            "shell",
            "monkey",
            "-p", "com.google.android.youtube.tvkids",
            "-c", "android.intent.category.LAUNCHER",
            "1"
        )
        if result.returncode != 0:
            result = self._run(
                "-s", self.host,
                "shell",
                "monkey",
                "-p", "com.google.android.apps.youtube.kids",
                "-c", "android.intent.category.LAUNCHER",
                "1"
            )

        if result.returncode == 0:
            print("[AndroidTV] YouTube Kids launched successfully.", flush=True)
        else:
            print(f"[AndroidTV] Failed to launch YouTube Kids: {result.stderr.strip()}", flush=True)

    def disconnect(self):
        """Disconnect from Android TV."""
        print(f"[AndroidTV] Disconnecting from {self.host}...", flush=True)
        self.connected = False
        self._run("disconnect", self.host)


def run_image(args, yunet, sface):
    global _SFACE_MODEL
    _SFACE_MODEL = sface
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit('Cannot read image: ' + args.image)
    db = load_db(args.db)
    detections = detect_yunet(yunet, frame, args.score_threshold, args.nms_threshold)
    annotated, rows = annotate(frame, detections, args.threshold, db)
    if args.enroll:
        if len(rows) != 1:
            raise SystemExit('--enroll requires exactly one detected face')
        db[args.enroll] = rows[0]['embedding'].tolist()
        save_db(args.db, db)
        print('enrolled:', args.enroll)
    if args.output:
        if not cv2.imwrite(args.output, annotated):
            raise SystemExit('Failed to write: ' + args.output)
        print('saved:', args.output)
    print('faces:', len(rows))
    for i, row in enumerate(rows):
        print(f'face {i}: detector={row["detector_score"]:.4f} identity={row["identity"]} similarity={row["similarity"]:.4f}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='SL2619 CPU face ID using TFLite C API')
    ap.add_argument('--yunet', required=True)
    ap.add_argument('--sface', required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image')
    src.add_argument('--camera', help='V4L2 camera device, e.g. /dev/video0')
    ap.add_argument('--output')
    ap.add_argument('--db', default='face_db.json')
    ap.add_argument('--enroll')
    ap.add_argument('--threshold', type=float, default=0.363)
    ap.add_argument('--score-threshold', type=float, default=0.6)
    ap.add_argument('--nms-threshold', type=float, default=0.3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--camera-width', type=int, default=1280)
    ap.add_argument('--camera-height', type=int, default=720)
    ap.add_argument('--camera-fps', type=int, default=30)
    ap.add_argument('--infer-every', type=int, default=5)
    ap.add_argument('--rtp-host', default='192.168.1.39')
    ap.add_argument('--rtp-port', type=int, default=5001)
    ap.add_argument('--rtp-bitrate', type=int, default=2000)
    ap.add_argument('--web-host', default='0.0.0.0')
    ap.add_argument('--web-port', type=int, default=8080)
    ap.add_argument('--tv-host', default='192.168.100.2:5555', help='Android TV ADB host:port (default: 192.168.100.2:5555)')
    ap.add_argument('--tv-mac', default='70:54:b4:fe:8e:ca', help='Android TV MAC address for Wake-on-LAN (default: 70:54:b4:fe:8e:ca)')
    ap.add_argument('--tv-adb-port', type=int, default=5038, help='Local ADB server port on board to avoid conflicts (default: 5038)')
    ap.add_argument('--target-user', default='Aguevel', help='Target user identity to trigger Netflix (default: Aguevel)')
    ap.add_argument('--kids-user', default='kids,Mike', help='Comma-separated user identities to trigger YouTube Kids (default: kids,Mike)')
    ap.add_argument('--debounce', type=float, default=2.5, help='App switch debounce time in seconds (default: 2.5)')
    ap.add_argument('--no-tv', action='store_true', help='Disable Android TV control')
    args = ap.parse_args()
    if args.enroll and not args.image:
        ap.error('--enroll requires --image')
    if args.infer_every < 1:
        ap.error('--infer-every must be >= 1')

    tv = None
    if not args.no_tv and args.tv_host:
        tv = AndroidTV(args.tv_host, mac=args.tv_mac, port=args.tv_adb_port)
        tv.connect()

    yunet = TFLiteModel(args.yunet, args.threads)
    sface = TFLiteModel(args.sface, args.threads)
    if args.image:
        return run_image(args, yunet, sface)
    return run_camera(args, yunet, sface, tv)


_SFACE_MODEL = None

if __name__ == '__main__':
    raise SystemExit(main())
