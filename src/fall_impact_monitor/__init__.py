"""Fall Impact Monitor: local detection of possible falls and possible head impacts."""

import os

# Official ONNX Runtime builds send usage telemetry to Microsoft by default. Processing
# stays local: turn it off before ONNX Runtime starts, unless the user set it explicitly.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
