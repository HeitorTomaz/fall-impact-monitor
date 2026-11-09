# Development-only helper: publishes a looped local video file to an RTSP server.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sim
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
USER sim
ENTRYPOINT ["sh", "/usr/local/bin/entrypoint.sh"]
