FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
# Install CPU-only torch first so bert-score reuses it instead of pulling the
# multi-GB CUDA build (no GPU is available to the container).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -e ".[dashboard,api]"

FROM python:3.12-slim-bookworm AS runtime

ENV HOME=/home/appuser \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib

# libgomp1: OpenMP for torch/numpy. libglib2.0-0: opencv-python-headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --from=builder /app /app

RUN mkdir -p /app/evaluation/data /app/evaluation/results /app/output /home/appuser/.cache \
 && chown -R appuser:appuser /app /home/appuser

USER appuser

ENTRYPOINT ["python", "-m", "screen_recording_description"]
