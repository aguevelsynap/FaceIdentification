# Project status

Validated on the SL2619 target before the web/live UI addition:

- YuNet TFLite loads through `/usr/lib/libtensorflow-lite.so`.
- YuNet detects one face from the test image with detector score about 0.946.
- SFace TFLite produces a 128-D embedding.
- Enrolled identity matching returned similarity 1.0000 on the same image.
- USB webcam `/dev/video0` works as MJPEG 1280x720.
- H.264/RTP transport to the development PC works on UDP 5001.

This final package adds:

- HTTP server on port 8080.
- Browser MJPEG live feedback at `/stream.mjpg`.
- `/status` JSON endpoint.
- `/snapshot.jpg` endpoint.
- `/enroll` form to enroll the current Unknown face into `face_db.json`.
- Existing RTP output remains available on UDP 5001.

The browser MJPEG and RTP outputs are both encoded from the same annotated frame.
