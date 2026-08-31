#!/bin/sh
set -e
cd "$(dirname "$0")"

TV_HOST="${TV_HOST:-192.168.1.173:5555}"
TV_MAC="${TV_MAC:-70:54:b4:fe:8e:ca}"

# Pre-wake Android TV via Wake-on-LAN broadcast
if [ -n "$TV_MAC" ]; then
    echo "[run.sh] Sending Wake-on-LAN broadcast to $TV_MAC..."
    python3 -c "
import socket, re
clean = re.sub(r'[^0-9A-Fa-f]', '', '$TV_MAC')
if len(clean) == 12:
    pkt = b'\xff' * 6 + bytes.fromhex(clean) * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for dest in ('255.255.255.255', '192.168.1.255'):
            for port in (9, 7):
                try: s.sendto(pkt, (dest, port))
                except Exception: pass
" 2>/dev/null || true
fi

# 1. Allow explicit --camera argument to take precedence
case "$*" in
  *--camera*)
    exec python3 scripts/face_id.py \
      --yunet models/face_detection_yunet_2023mar_float32.tflite \
      --sface models/face_recognition_sface_2021dec_float32.tflite \
      --db /home/root/face_db.json \
      --rtp-host 192.168.1.123 \
      --rtp-port 5001 \
      --infer-every 5 \
      --web-host 0.0.0.0 \
      --web-port 8080 \
      --tv-host "$TV_HOST" \
      --tv-mac "$TV_MAC" \
      --kids-user "kids,Mike" \
      "$@"
    ;;
esac

# 2. Dynamic camera detection: find valid video capture device
CAMERA_DEV=""

for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    if command -v v4l2-ctl >/dev/null 2>&1; then
        caps=$(v4l2-ctl --device="$dev" -D 2>/dev/null || true)
        if echo "$caps" | grep -q "Video Capture"; then
            if echo "$caps" | grep -q "uvcvideo"; then
                CAMERA_DEV="$dev"
                break
            elif [ -z "$CAMERA_DEV" ]; then
                CAMERA_DEV="$dev"
            fi
        fi
    else
        CAMERA_DEV="$dev"
        break
    fi
done

if [ -z "$CAMERA_DEV" ]; then
    if [ -e "/dev/video2" ]; then
        CAMERA_DEV="/dev/video2"
    elif [ -e "/dev/video0" ]; then
        CAMERA_DEV="/dev/video0"
    else
        CAMERA_DEV="/dev/video0"
    fi
fi

echo "[run.sh] Auto-detected camera: $CAMERA_DEV"

exec python3 scripts/face_id.py \
  --yunet models/face_detection_yunet_2023mar_float32.tflite \
  --sface models/face_recognition_sface_2021dec_float32.tflite \
  --camera "$CAMERA_DEV" \
  --db /home/root/face_db.json \
  --rtp-host 192.168.1.123 \
  --rtp-port 5001 \
  --infer-every 5 \
  --web-host 0.0.0.0 \
  --web-port 8080 \
  --tv-host "$TV_HOST" \
  --tv-mac "$TV_MAC" \
  --kids-user "kids,Mike" \
  "$@"