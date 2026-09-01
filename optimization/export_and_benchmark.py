"""MLflow experiment 4: latency optimization.

architecture.md's plan branches by winning model type (linear -> ONNX +
INT8 quantization; Random Forest -> ONNX + pruning). The actual winner,
ComplementNB, is neither, but behaves like a linear model for export
purposes: its decision rule is a dot product against a dense per-class
weight matrix (feature_log_prob_), the same shape as a linear model's
coef_ -- so it's treated as the linear branch. Quantization compresses
that dense matrix; it has no analogue for tree ensembles, which is why
that technique is Random-Forest-specific in the original plan.

Reports P50/P95/P99 over the whole predict path (TF-IDF vectorization +
classifier inference) for a single document at a time, not just
model.predict() on a pre-vectorized batch -- preprocessing can dominate
total latency (see docs/technical-decisions.md).

Quantization needs two workarounds specific to skl2onnx's text-pipeline
graph (TfIdfVectorizer + Naive Bayes ops aren't the standard vision/NLP
graphs onnxruntime's quantization tooling is built and tested against —
see docs/technical-decisions.md for the full story of both failures):
1. `quant_pre_process(..., skip_symbolic_shape=True)` — full symbolic
   shape inference throws ("Incomplete symbolic shape inference") on this
   graph; skipping just that sub-step still lets basic shape inference +
   model optimization run.
2. `extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT}` on
   `quantize_dynamic` — without it, quantization fails outright
   ("Unable to find data type for weight_name='sum_result'") on an
   intermediate tensor from ComplementNB's decision-rule subgraph that
   the quantizer's type inference can't resolve on its own.
"""

import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.metrics import f1_score

from training import mlflow_config
from training.data import load_pool_and_test

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
ONNX_FP32_PATH = MODELS_DIR / "pipeline_fp32.onnx"
ONNX_PREPROCESSED_PATH = MODELS_DIR / "pipeline_fp32_preprocessed.onnx"
ONNX_INT8_PATH = MODELS_DIR / "pipeline_int8.onnx"

N_WARMUP = 20
N_RUNS = 500
RANDOM_STATE = 42


def benchmark(predict_one, warmup_docs: list[str], bench_docs: list[str]) -> dict:
    for doc in warmup_docs:
        predict_one(doc)

    latencies_ms = []
    for doc in bench_docs:
        start = time.perf_counter()
        predict_one(doc)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def run() -> dict:
    _, test = load_pool_and_test()
    y_true = test["condition_name"]
    X_full = test["medical_abstract"].tolist()

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(test), size=N_RUNS + N_WARMUP, replace=True)
    sample_docs = test["medical_abstract"].iloc[sample_idx].tolist()
    warmup_docs, bench_docs = sample_docs[:N_WARMUP], sample_docs[N_WARMUP:]

    MODELS_DIR.mkdir(exist_ok=True)
    pipeline = joblib.load(PIPELINE_PATH)
    results = {}

    # --- Baseline: sklearn pipeline ---
    def sklearn_predict_one(doc):
        return pipeline.predict([doc])[0]

    latency = benchmark(sklearn_predict_one, warmup_docs, bench_docs)
    f1 = f1_score(y_true, pipeline.predict(X_full), average="macro")
    results["baseline_sklearn"] = {
        **latency,
        "f1_macro": f1,
        "size_bytes": PIPELINE_PATH.stat().st_size,
    }

    # --- ONNX FP32 export ---
    onx = to_onnx(pipeline, initial_types=[("input", StringTensorType([None, 1]))])
    ONNX_FP32_PATH.write_bytes(onx.SerializeToString())

    sess_fp32 = ort.InferenceSession(
        str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"]
    )
    input_name = sess_fp32.get_inputs()[0].name

    def onnx_fp32_predict_one(doc):
        x = np.array([[doc]], dtype=object)
        return sess_fp32.run(None, {input_name: x})[0][0]

    latency = benchmark(onnx_fp32_predict_one, warmup_docs, bench_docs)
    preds = [onnx_fp32_predict_one(d) for d in X_full]
    f1 = f1_score(y_true, preds, average="macro")
    results["onnx_fp32"] = {
        **latency,
        "f1_macro": f1,
        "size_bytes": ONNX_FP32_PATH.stat().st_size,
    }

    # --- ONNX INT8 dynamic quantization ---
    quant_pre_process(
        str(ONNX_FP32_PATH), str(ONNX_PREPROCESSED_PATH), skip_symbolic_shape=True
    )
    quantize_dynamic(
        str(ONNX_PREPROCESSED_PATH),
        str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    sess_int8 = ort.InferenceSession(
        str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"]
    )
    input_name_int8 = sess_int8.get_inputs()[0].name

    def onnx_int8_predict_one(doc):
        x = np.array([[doc]], dtype=object)
        return sess_int8.run(None, {input_name_int8: x})[0][0]

    latency = benchmark(onnx_int8_predict_one, warmup_docs, bench_docs)
    preds = [onnx_int8_predict_one(d) for d in X_full]
    f1 = f1_score(y_true, preds, average="macro")
    results["onnx_int8"] = {
        **latency,
        "f1_macro": f1,
        "size_bytes": ONNX_INT8_PATH.stat().st_size,
    }

    mlflow_config.configure()
    mlflow.set_experiment("latency-optimization")
    for name, metrics in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"variant": name, "n_runs": N_RUNS})
            mlflow.log_metrics(metrics)

    for name, metrics in results.items():
        print(
            f"{name:18s} p50={metrics['p50_ms']:7.3f}ms  p95={metrics['p95_ms']:7.3f}ms  "
            f"p99={metrics['p99_ms']:7.3f}ms  f1_macro={metrics['f1_macro']:.4f}  "
            f"size={metrics['size_bytes'] / 1024:.1f}KB"
        )

    return results


if __name__ == "__main__":
    run()
