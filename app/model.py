"""Loads the ONNX-exported classifier and runs category prediction.

Serves the ONNX FP32 artifact, not the raw sklearn pipeline — chosen in
the latency-optimization stage for a 4.4x P50 latency win over the
sklearn baseline at zero accuracy cost (see docs/technical-decisions.md).

Loading is deferred to an explicit load_model() call (triggered by
app/main.py's FastAPI lifespan at real startup) rather than run at import
time — models/ is gitignored, so the artifact doesn't exist in a fresh
checkout (e.g. CI running pytest); eager module-level loading would break
importing this module there. predict_category() still loads lazily on
first use as a fallback for direct/script usage outside the API.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pipeline_fp32.onnx"

_session: ort.InferenceSession | None = None
_input_name: str | None = None


def load_model() -> None:
    global _session, _input_name
    _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name


def predict_category(text: str) -> str:
    if _session is None:
        load_model()
    x = np.array([[text]], dtype=object)
    return _session.run(None, {_input_name: x})[0][0]
