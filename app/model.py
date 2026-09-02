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

import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "pipeline_fp32.onnx"
VOCABULARY_PATH = MODELS_DIR / "vocabulary.json"

_TOKEN_PATTERN = re.compile(r"\b\w\w+\b")

_session: ort.InferenceSession | None = None
_input_name: str | None = None
_vocabulary: set[str] | None = None


def load_model() -> None:
    global _session, _input_name, _vocabulary
    _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _vocabulary = set(json.loads(VOCABULARY_PATH.read_text()))


def predict_category(text: str) -> str:
    if _session is None:
        load_model()
    x = np.array([[text]], dtype=object)
    return _session.run(None, {_input_name: x})[0][0]


def has_known_vocabulary(text: str) -> bool:
    """Whether the input shares any vocabulary with the training data.

    A proxy for "does the TF-IDF feature vector have any signal at all" --
    found via real-world testing that short/colloquial inputs (e.g.
    "stomachache", not a substring match of "stomach") can produce an
    all-zero feature vector, at which point the classifier's prediction is
    driven entirely by its structural class bias, not real evidence. See
    docs/technical-decisions.md.
    """
    if _vocabulary is None:
        load_model()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return any(token in _vocabulary for token in tokens)
