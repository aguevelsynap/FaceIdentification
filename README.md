# SL2619 CPU Face ID

Board-native CPU face identification prototype using the TFLite C API already present on the SL2619 image.

## Runtime

- YuNet TFLite detector
- SFace TFLite embedding model
- `/usr/lib/libtensorflow-lite.so`
- OpenCV 4.9 for camera capture, alignment and drawing
- GStreamer for H.264/RTP output
- Python standard-library HTTP server for the web UI

No ONNX Runtime is required on the board.

## Models

Copy the validated models into `models/`:

- `face_detection_yunet_2023mar_float32.tflite`
- `face_recognition_sface_2021dec_float32.tflite`

## Start on SL2619

```sh
cd /home/root/face-id-cpu
python3 scripts/face_id.py \
  --yunet models/face_detection_yunet_2023mar_float32.tflite \
  --sface models/face_recognition_sface_2021dec_float32.tflite \
  --camera /dev/video0 \
  --db /home/root/face_db.json \
  --rtp-host 192.168.1.39 \
  --rtp-port 5001 \
  --infer-every 5 \
  --web-host 0.0.0.0 \
  --web-port 8080
```

## PC video receiver

```sh
gst-launch-1.0 \
  udpsrc port=5001 \
  caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" \
  ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
```

## Browser UI

Open:

`http://<BOARD_IP>:8080/`

The page provides:

- live annotated MJPEG video
- current FPS and inference latency
- current detected identities
- an enrollment form for the latest Unknown face

When an unknown face is visible, enter a name and a face index (normally `0`) and submit **Enroll visible unknown**. The application writes the normalized 128-D embedding to the JSON database atomically.

## Important

Only one process should own `/dev/video0` at a time. Stop the standalone `gst-launch-1.0 v4l2src ...` camera sender before starting `scripts/face_id.py`.

YuNet uses the validated TFLite interface: BGR, NHWC, float32, range 0..255. YuNet scoring is `sqrt(cls * obj)` with exponential width/height decoding. SFace embeddings are L2-normalized before cosine matching.
