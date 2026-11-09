#!/bin/sh
# Loop SIM_FILE (relative to /media) forever into SIM_TARGET without transcoding.
set -eu
: "${SIM_FILE:?SIM_FILE is required}"
: "${SIM_TARGET:?SIM_TARGET is required}"
while true; do
  ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "/media/${SIM_FILE}" \
    -c copy -f rtsp -rtsp_transport tcp "${SIM_TARGET}" || true
  echo "camera-sim: publisher exited, restarting in 2 s" >&2
  sleep 2
done
