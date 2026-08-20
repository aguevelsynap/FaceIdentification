#!/bin/sh
set -eu
printf 'arch: '; uname -m
printf 'python: '; python3 --version
printf 'opencv: '; python3 -c 'import cv2; print(cv2.__version__)'
printf 'numpy: '; python3 -c 'import numpy; print(numpy.__version__)'
printf 'cv2.dnn: '; python3 -c 'import cv2; print(hasattr(cv2, "dnn"))'
printf 'tflite: '; ls -lh /usr/lib/libtensorflow-lite.so | awk '{print $5}'
printf 'gstreamer: '; gst-launch-1.0 --version | head -1
printf 'SynAP: '; synap_cli --version 2>/dev/null || true
python3 - <<'PY'
import ctypes
L=ctypes.CDLL('/usr/lib/libtensorflow-lite.so')
for n in ('TfLiteModelCreateFromFile','TfLiteInterpreterCreate','TfLiteInterpreterInvoke'):
    assert hasattr(L,n), n
print('tflite C API: OK')
PY
