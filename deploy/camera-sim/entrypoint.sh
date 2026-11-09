#!/bin/sh
# Loop FIM_SIM_FILE (relative to /media) forever into FIM_SIM_TARGET without transcoding.
set -eu
: "${FIM_SIM_FILE:?FIM_SIM_FILE is required}"
: "${FIM_SIM_TARGET:?FIM_SIM_TARGET is required}"
while true; do
  ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "/media/${FIM_SIM_FILE}" \
    -c copy -f rtsp -rtsp_transport tcp "${FIM_SIM_TARGET}" || true
  echo "camera-sim: publisher exited, restarting in 2 s" >&2
  sleep 2
done
