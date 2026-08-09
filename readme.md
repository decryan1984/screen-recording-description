# Screen Recording Description Service

Builds a per-frame timeline of desktop screen recordings and infers the user's overall intent, using local Vision Language Models (default is **qwen3.5:4b**, served via **Ollama**) and, optionally, **Gemini** in the cloud. Outputs are scored by an LLM-as-judge (**phi4:14b**) plus ROUGE and can be explored in a Streamlit dashboard.

## Project Structure

```
├── src/screen_recording_description/
│   ├── __main__.py           # CLI entry point (evaluation/batch mode)
│   ├── config.py             # All configurable parameters and prompts
│   ├── pipeline.py           # Video processing orchestration
│   ├── vlm_inference.py      # Local Ollama VLM: frame inference, differencing, intent inference
│   ├── online_inference.py   # Gemini (cloud VLM) inference worker
│   ├── llm_inference.py      # LLM-as-judge evaluation + ROUGE scoring
│   ├── service.py            # Single service: submission API + in-process worker
│   ├── dashboard.py          # Combined Streamlit app: switches between the two dashboards
│   ├── results_dashboard.py  # Streamlit dashboard for exploring evaluation results
│   └── service_dashboard.py  # Streamlit dashboard for live service metrics
├── evaluation/               # Evaluation mode: dataset + batch results
│   ├── data/                 # GUI-World dataset
│   └── results/              # Batch run outputs (read by the dashboard)
└── output/                   # Service mode: runs/ (per-run pipeline output)
```

| Module | Purpose |
|---|---|
| `config.py` | Model settings, frame selection thresholds, evaluation prompts, paths |
| `pipeline.py` | Decode video, select key frames, describe each, infer intent, evaluate |
| `vlm_inference.py` | Run per-frame Ollama inference, compute frame diffs, infer intent from timelines |
| `online_inference.py` | Parallel Gemini inference worker |
| `llm_inference.py` | Load annotations, LLM-as-judge scoring, ROUGE |
| `__main__.py` | CLI argument parsing and dispatch (evaluation mode) |
| `service.py` | Single service (service mode): submit runs, in-process worker runs the pipeline, status + live metrics |
| `dashboard.py` | Combined Streamlit app which can switch between the two dashboards |
| `results_dashboard.py` | Streamlit dashboard comparing models/variants |
| `service_dashboard.py` | Streamlit dashboard showing live service metrics |

## Prerequisites

1. **Install Ollama** — https://ollama.com/download
2. **Pull the models:**
```bash
ollama pull qwen3.5:4b       # local VLM (default)
ollama pull qwen3.5:latest   # larger Qwen variant (tested for comparison)
ollama pull gemma3:4b        # second local VLM (for comparison)
ollama pull phi4:14b         # LLM judge used for evaluation
```
3. **Install the package:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                 # core pipeline
pip install -e ".[dashboard]"    # add Streamlit dashboard deps
```
4. **(Optional) Gemini** — the cloud run is off by default. To include it, set the `GEMINI_API_KEY` environment variable (e.g. `export GEMINI_API_KEY=...`, or add it to the service's Docker env) and pass `--run-online-inference`. Without an API key set, the online run is automatically skipped.

### Hardware

The local models run on the GPU where available, otherwise they fall back to CPU, which is much slower. The Docker containers (API, dashboard, evaluation harness) are CPU-only.
- **Disk** — the local models are pulled via Ollama and stored on the host: the 4B VLMs are ~3 GB each and the `phi4:14b` 'judge' is ~9 GB, so budget ~15 GB (plus your videos and results).
- **RAM / unified memory** — 16 GB is the practical minimum, 24 GB+ is reasonably comfortable. Only one VLM model is resident at a time but using the judge in evaluation mode uses a lot of memory.

The project runs in two modes, described below:

- **[Evaluation mode](#evaluation-mode)** — batch-process the dataset, score the outputs, and compare models in a dashboard.
- **[Service mode](#service-mode)** — run a local API that accepts videos and processes them on demand.

---

## Evaluation mode

Batch-process the GUI-World dataset (or your own videos), score the outputs with the LLM-as-judge plus ROUGE, and compare models in a dashboard. Results are written to `evaluation/results/`.

### Dataset

Download the videos from the `multi` subdirectory on Hugging Face:

https://huggingface.co/datasets/ONE-Lab/GUI-World/tree/main/multi

Place the video files in:

```
evaluation/data/GUI-World/multi/
```

### Run in Docker (sandboxed)

Runs the pipeline in a locked-down container. Ollama stays on the host (for GPU) and the container connects to it.

Start Ollama on the host, bound to all interfaces so the container can reach it:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
ollama pull qwen3.5:4b && ollama pull qwen3.5:latest && ollama pull gemma3:4b && ollama pull phi4:14b
```

Build the image, then run a batch over the dataset via the `pipeline` service:

```bash
docker compose build

docker compose run --rm pipeline --videos 457
docker compose run --rm pipeline --videos 457 --full-param-run
docker compose run --rm pipeline --model qwen3.5:4b gemma3:4b --videos all-eligible-videos --full-param-run
```

The dataset is mounted read-only from `./evaluation/data`. Results are written to `./evaluation/results`.

To use Gemini, export `GEMINI_API_KEY` on the host (forwarded into the container by `docker-compose.yml`), then pass `--run-online-inference`:

```bash
export GEMINI_API_KEY=...
docker compose run --rm pipeline --videos 457 --run-online-inference
```

### Run from the command line

Run the same pipeline directly on the host (activate the venv from [Prerequisites](#prerequisites) first):

```bash
# Single video by ID
python -m screen_recording_description --videos 457

# Full parameter run: all diff thresholds + resolution/token variants
python -m screen_recording_description --videos 457 --full-param-run

# Multiple videos, also running the Gemini cloud model (off by default)
python -m screen_recording_description --videos 457 458 --run-online-inference

# Compare specific models across every eligible GUI-World video, full param
python -m screen_recording_description --model qwen3.5:4b gemma3:4b --videos all-eligible-videos --full-param-run
```

Or via the installed script:

```bash
screen-recording-description --videos 457
```

### Dashboards

Two separate dashboards run from a single app — use the switcher at the top of the sidebar to move between **Evaluation Results** and **Service Metrics**. Activate the venv from [Prerequisites](#prerequisites), then:

```bash
streamlit run src/screen_recording_description/dashboard.py
# http://localhost:8501
```

The **Evaluation Results** page explores and compares batch results (reads from `evaluation/results/`). Alongside the quality scores (LLM-judge + ROUGE), each variant also reports several cost/efficiency signals:

- **Inference Time** — measured time to process the video (frame VLM + intent). For Gemini this is remote API latency, not local compute.
- **Compute (TERAFLOPS)** — analytical model work for comparing models independent of hardware. Local models only, parameters come from Ollama (`/api/show`).
- **Memory (MB)** — resident model size while loaded, from Ollama (`/api/ps`), local models only.
- **Gemini Cost (USD)** — measured for the Gemini run and projected onto local variants (frames x Gemini's per-frame token rates, priced from `GEMINI_*_USD_PER_1M` in `config.py`). Full-resolution variants are omitted, as their image tokens differ from the baseline. Gemini needs a 4x higher output-token cap than the local VLMs ("thinking" tokens count toward the limit), but its actual per-frame token usage is comparable or lower.

---

## Service mode

Run a local FastAPI service that accepts a directory of videos, processes each through the pipeline in a background worker, and exposes run status + live metrics over HTTP. Per-run output is written to `./output/runs`. The queue, status, and metrics live in memory only (nothing persists between restarts). Cloud (Gemini) inference is off in service mode — submissions are always processed locally.

Each request can optionally pass in a custom **`analysis_prompt`** that determines the final analysis done on the frame timeline — for example a compliance or best-practice check. This defaults to the "what was the user trying to do?" intent inference.

### Run in Docker

Start Ollama on the host (as in [Evaluation mode](#run-in-docker-sandboxed)), then bring up the service at `http://127.0.0.1:8000`:

```bash
docker compose build
docker compose up -d
# Or read videos from a specific directory instead:
INPUT_DIR=/path/to/videos docker compose up -d
```

The service reads videos from `INPUT_DIR`, mounted read-only at the same path inside the container so absolute paths line up. **`INPUT_DIR` defaults to this project's root** (the repo you run `docker compose` from), so place the videos you want to process anywhere under the project — only that directory is shared with the container, never your whole home directory.

Stop the service:

```bash
docker compose down
```

### Use the API

Submit videos by referencing a directory (no upload). Omit `filenames` to take every eligible video in the directory. Pass `analysis_prompt` to override the default intent inference with a domain-specific question. Plain HTTP on loopback — no TLS or auth:

```bash
# Health check
curl -s http://127.0.0.1:8000/health
# -> {"status":"ok"}

# Submit specific files within a directory
curl -s -X POST http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"directory": "'"$HOME"'/videos", "filenames": ["a.mp4", "b.mp4"]}'
# -> {"accepted": [{"filename": "a.mp4", "run_id": "abc123def456"}], "rejected": []}

# Submit every eligible video in a directory (omit "filenames")
curl -s -X POST http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"directory": "'"$HOME"'/videos"}'

# Submit with a custom analysis prompt (redirects the final output away from intent)
curl -s -X POST http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"directory": "'"$HOME"'/videos", "filenames": ["a.mp4"],
       "analysis_prompt": "Did the user expose any confidential information on screen? Answer yes/no and state the evidence."}'

# Poll a single run's status: queued | running | succeeded | failed
curl -s http://127.0.0.1:8000/runs/abc123def456
```

### Service metrics dashboard

The **Service Metrics** page of the combined dashboard gives a live view of the service (it polls `GET /metrics` every few seconds, so the service must be running to show live data). Launch the combined app as in [Dashboards](#dashboards) and pick **Service Metrics** in the sidebar:

```bash
streamlit run src/screen_recording_description/dashboard.py
# http://localhost:8501
```

Point it at a non-default service with `API_URL=http://host:8000 streamlit run …/dashboard.py`.

### Interactive API docs

FastAPI generates browser-based documentation automatically:

- **Swagger UI** — <http://127.0.0.1:8000/docs>: interactive. Expand an endpoint, click **Try it out**, fill in the request body, and **Execute** to send a real request and see the live response.
- **ReDoc** — <http://127.0.0.1:8000/redoc>: a clean, read-only reference of the same endpoints, request/response models, and schemas.
- **OpenAPI schema** — <http://127.0.0.1:8000/openapi.json>: the machine-readable spec behind both UIs. Import it into an API client (Postman, Insomnia, Bruno) or use it to generate a client.

### Monitoring

- **Logs** — structured JSON events: `docker compose logs -f api`
- **Metrics** — `GET /metrics` (in-memory): uptime, model, Ollama reachability, run counts by status (received/queued/running/succeeded/failed), and the last run's duration.
- **Health** — `docker compose ps` reflects the container healthcheck (`GET /health`).

---

## Configuration

All parameters are in `src/screen_recording_description/config.py` — frame selection thresholds, VLM model, prompts, etc.

### Custom Ollama endpoint

```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434 python -m screen_recording_description --videos 457
```

## Example output

Each run writes one `<video_name>.json` into a timestamped run directory —
`evaluation/results/<timestamp>/` in evaluation mode, `output/runs/<timestamp>_<run_id>/`
in service mode. Video-level metadata sits at the top level, per-model results (frame
timeline, inferred intent, performance) are nested under a model block:

```json
{
  "video": "/path/to/recording.mp4",
  "total_frames": 900,
  "fps": 30.0,
  "video_length_sec": 30.0,
  "resolution": "2560x1440",
  "reduced_resolution": "1280x720",
  "frames_decoded": 900,
  "frames_processed": 47,
  "vlm": {
    "model": "qwen3.5:4b",
    "diff_threshold": 0.01,
    "analysis_prompt": "In one or two sentences, state what the user was trying to accomplish...",
    "intent": "The user is composing an email...",
    "timeline": [
      {
        "frame_number": 1,
        "timestamp_sec": 0.03,
        "latency_sec": 8.21,
        "frame_description": "The user has a web browser open showing...",
        "prompt_tokens": 512,
        "completion_tokens": 84
      }
    ],
    "performance": { "avg_frame_latency_sec": 7.9, "total_completion_tokens": 3900 }
  }
}
```

Alongside the `<video>.json`, each run directory also contains a small companion
artifact. **Evaluation mode** writes `config.json` (the run's shared model params and
prompts). **Service mode** writes `status.json` per run (status, timings, video length —
what the metrics dashboard's run history reads). In evaluation mode the model block
additionally gains `evaluation` (LLM-judge) and `rouge` sub-blocks. A
full parameter runs produce several model blocks (e.g. `vlm_diff0.01`, `vlm_fullres`)
and, with `--run-online-inference`, a `gemini` block.

## Notes

On Linux, if you hit permission errors writing outputs, run `sudo chown 10001:10001 evaluation/results output`.
