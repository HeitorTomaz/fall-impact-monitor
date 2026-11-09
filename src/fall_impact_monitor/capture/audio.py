"""Audio from a camera's stream (through the MediaMTX relay), as 8 kHz mono float32.

Runs in its own thread. Each chunk is passed with the monotonic time of its first
sample, estimated from its arrival (arrival minus its duration). Network streams are
reopened with capped backoff; a source without an audio track is reported once and the
reader stops (audio is optional: head-impact rules work without it).
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import av
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.capture.reader import is_network_source
from fall_impact_monitor.config import mask_url

log = logging.getLogger(__name__)
RATE = 8000


class AudioReader:
    def __init__(
        self,
        url: str,
        camera_id: str,
        on_samples: Callable[[float, NDArray[np.float32]], None],
        *,
        open_timeout_s: float = 10.0,
        read_timeout_s: float = 5.0,
        backoff_max_s: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._camera_id = camera_id
        self._on_samples = on_samples
        self._timeouts = (open_timeout_s, read_timeout_s)
        self._backoff_max_s = backoff_max_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        network = is_network_source(self._url)
        backoff_s = 1.0
        while not self._stop.is_set():
            try:
                if not self._read_once(network):
                    return  # no audio track, or a file that ended
                backoff_s = 1.0
            except Exception as error:
                log.warning(
                    "audio stream failed",
                    extra={"camera": self._camera_id, "error": type(error).__name__},
                )
                if not network:
                    return
            self._sleep(backoff_s)
            backoff_s = min(backoff_s * 2, self._backoff_max_s)

    def _read_once(self, network: bool) -> bool:
        """Read until the stream ends; False when there is nothing more to read."""
        options = {"rtsp_transport": "tcp"} if network else {}
        container: Any = av.open(self._url, options=options, timeout=self._timeouts)
        try:
            if not container.streams.audio:
                log.info(
                    "no audio track, thump evidence disabled",
                    extra={"camera": self._camera_id, "source": mask_url(self._url)},
                )
                return False
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
            for packet in container.demux(stream):
                if self._stop.is_set():
                    return False
                for frame in packet.decode():
                    for out in resampler.resample(frame):
                        samples = np.asarray(out.to_ndarray(), dtype=np.float32).reshape(-1)
                        self._on_samples(self._monotonic() - samples.size / RATE, samples)
        finally:
            container.close()
        return network  # a network stream that ended is reopened; a file is done
