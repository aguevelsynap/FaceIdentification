#!/usr/bin/env python3
import argparse
import base64
import ctypes
import json
import os
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

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            body = Path(Path(__file__).resolve().parent.parent / 'web' / 'index.html').read_bytes()
            self._send(HTTPStatus.OK, 'text/html; charset=utf-8', body)
            return
        if self.path == '/status':
            _, rows, fps, capfps, count, infer_ms, error = self.state.snapshot()
            payload = {
                'running': self.state.running,
                'fps': fps,
                'capture_fps': capfps,
                'frames': count,
                'inference_ms': infer_ms,
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


def start_web_server(state, db_path, host, port, threshold):
    WebHandler.state = state
    WebHandler.db_path = db_path
    WebHandler.threshold = threshold
    server = ThreadingHTTPServer((host, port), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_camera(args, yunet, sface):
    global _SFACE_MODEL
    _SFACE_MODEL = sface
    db = load_db(args.db)
    state = SharedState(args.db)
    server = None
    if args.web_port > 0:
        server = start_web_server(state, args.db, args.web_host, args.web_port, args.threshold)
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
            cv2.putText(annotated, f'face-id cpu  faces={len(last_rows)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'infer={state.last_inference_ms:.0f}ms', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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
                print(f'frames={frame_count} processed_fps={fps:.1f} faces={len(last_rows)} infer_ms={state.last_inference_ms:.1f}', flush=True)
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
        if server:
            server.shutdown()
    return 0


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
    args = ap.parse_args()
    if args.enroll and not args.image:
        ap.error('--enroll requires --image')
    if args.infer_every < 1:
        ap.error('--infer-every must be >= 1')
    yunet = TFLiteModel(args.yunet, args.threads)
    sface = TFLiteModel(args.sface, args.threads)
    if args.image:
        return run_image(args, yunet, sface)
    return run_camera(args, yunet, sface)


_SFACE_MODEL = None

if __name__ == '__main__':
    raise SystemExit(main())
