# Multi-stage build: builder installs deps via uv, runtime stays slim.
#
# Only [project.dependencies] are installed here (--no-dev) -- the
# training/optimization toolchain (mlflow, dvc, scikit-learn, pandas,
# matplotlib, skl2onnx) lives in the "training" dependency group and has
# no business in the served image; the app serves the exported ONNX
# model, not the sklearn pipeline. See pyproject.toml.
#
# models/pipeline_fp32.onnx is DVC-tracked (S3), not committed to git --
# it must already be present in the build context (`dvc pull` before
# `docker build`, see docs/architecture.md § CI/CD) since it isn't fetched
# inside this Dockerfile.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app/ ./app/
RUN uv sync --locked --no-dev

FROM python:3.11-slim AS runtime

# onnxruntime's StringNormalizer op (used by the TF-IDF vectorizer's ONNX
# graph) requires the en_US.UTF-8 locale at runtime -- python:3.11-slim
# doesn't have it by default, and its absence fails model loading
# entirely (caught via an actual `docker run`, not assumed to work).
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app
COPY models/pipeline_fp32.onnx ./models/pipeline_fp32.onnx

ENV PATH="/app/.venv/bin:${PATH}"

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
