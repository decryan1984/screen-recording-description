# Screen Recording Description Service

Generates text descriptions and summaries of desktop screen recordings using local Vision Language Models (defaults are **qwen3.5:4b** and **gemma3:4b**, served via **Ollama**) and, optionally, **Gemini** in the cloud. Outputs are scored by an LLM-as-judge (**phi4:14b**) plus ROUGE and can be explored in a Streamlit dashboard.

## Project Structure

```
├── src/screen_recording_description/
│   ├── __main__.py           # CLI entry point (evaluation/batch mode)
│   ├── config.py             # All configurable parameters and prompts
│   ├── pipeline.py           # Video processing orchestration
│   ├── vlm_inference.py      # Local Ollama VLM: frame inference, differencing, summarisation
│   ├── online_inference.py   # Gemini (cloud VLM) inference worker
│   ├── llm_inference.py      # LLM-as-judge evaluation + ROUGE scoring
│   ├── service.py            # Single service: submission API + in-process worker
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
| `pipeline.py` | Decode video, select key frames, describe each, summarise, evaluate |
| `vlm_inference.py` | Run per-frame Ollama inference, compute frame diffs, summarise timelines |
| `online_inference.py` | Parallel Gemini inference worker |
| `llm_inference.py` | Load annotations, LLM-as-judge scoring, ROUGE |
| `__main__.py` | CLI argument parsing and dispatch (evaluation mode) |
| `service.py` | Single service (service mode): submit runs, in-process worker runs the pipeline, status + live metrics |
| `results_dashboard.py` | Streamlit dashboard comparing models/variants |
| `service_dashboard.py` | Streamlit dashboard showing live service metrics |

## Prerequisites

1. **Install Ollama** — https://ollama.com/download
2. **Pull the models:**
```bash
ollama pull qwen3.5:4b   # local VLM
ollama pull gemma3:4b    # second local VLM (for comparison)
ollama pull phi4:14b     # LLM judge used for evaluation
```
3. **Install the package:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                 # core pipeline
pip install -e ".[dashboard]"    # add Streamlit dashboard deps
```
4. **(Optional) Gemini** — to include the parallel cloud inference, set `GEMINI_API_KEY` in `src/screen_recording_description/config.py` (keep it empty in version control and fill it in locally only).
Gemini is off by default. To include it, set `GEMINI_API_KEY` and pass `--run-online-inference`.
Without a key set, the online run is automatically skipped.

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
ollama pull qwen3.5:4b && ollama pull gemma3:4b && ollama pull phi4:14b
```

Build the image, then run a batch over the dataset via the `pipeline` service:

```bash
docker compose build

docker compose run --rm pipeline --videos 457
docker compose run --rm pipeline --videos 457 --full-param-run
docker compose run --rm pipeline --model all --videos all-eligible-videos --full-param-run
```

The dataset is mounted read-only from `./evaluation/data`; results are written to `./evaluation/results`.

To use Gemini, set `GEMINI_API_KEY` in `src/screen_recording_description/config.py`, then:

```bash
docker compose run --rm pipeline --videos 457 --run-online-inference
```

### Run from the command line

Run the same pipeline directly on the host (activate the venv from [Prerequisites](#prerequisites) first):

```bash
# Single video by ID
python -m screen_recording_description --videos 457

# Full parameter sweep: all diff thresholds + resolution/token variants
python -m screen_recording_description --videos 457 --full-param-run

# Multiple videos, also running the Gemini cloud model (off by default)
python -m screen_recording_description --videos 457 458 --run-online-inference

# Every model in VLM_MODELS across every eligible GUI-World video, full sweep
python -m screen_recording_description --model all --videos all-eligible-videos --full-param-run
```

Or via the installed script:

```bash
screen-recording-description --videos 457
```

### Results dashboard

Explore and compare batch results (reads from `evaluation/results/`). Activate the venv from [Prerequisites](#prerequisites), then:

```bash
streamlit run src/screen_recording_description/results_dashboard.py
# http://localhost:8501
```

---

## Service mode

Run a local FastAPI service that accepts a directory of videos, processes each through the pipeline in a background worker, and exposes run status + live metrics over HTTP. Per-run output is written to `./output/runs`; the queue, status, and metrics live in memory only (nothing persists between restarts). Cloud (Gemini) inference is off in service mode — submissions are always processed locally.

### Run in Docker

Start Ollama on the host (as in [Evaluation mode](#run-in-docker-sandboxed)), then bring up the service at `http://127.0.0.1:8000`:

```bash
docker compose build
docker compose up -d                       # shares the project root by default
# Or read videos from a specific directory instead:
INPUT_DIR=/path/to/videos docker compose up -d
```

The service reads videos from `INPUT_DIR`, mounted read-only at the same path inside the container so absolute paths line up. **`INPUT_DIR` defaults to this project's root** (the repo you run `docker compose` from), so place the videos you want to process anywhere under the project — only that directory is shared with the container, never your whole home directory.

Stop the service:

```bash
docker compose down
```

### Use the API

Submit videos by referencing a directory (no upload). Omit `filenames` to take every eligible video in the directory. Plain HTTP on loopback — no TLS or auth:

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

# Poll a single run's status: queued | running | succeeded | failed
curl -s http://127.0.0.1:8000/runs/abc123def456
```

### Service metrics dashboard

Live view of the service's metrics (polls `GET /metrics` every few seconds; needs the service running). Activate the venv from [Prerequisites](#prerequisites), then:

```bash
streamlit run src/screen_recording_description/service_dashboard.py --server.port 8502
# http://localhost:8502
```

Point it at a non-default service with `API_URL=http://host:8000 streamlit run …/service_dashboard.py`.

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
in service mode. Video-level metadata sits at the top level; per-model results (frame
timeline, summary, inferred intent, performance) are nested under a model block:

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
    "summary": "The user opened a browser and...",
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
artifact: **evaluation mode** writes `config.json` (the run's shared model params and
prompts); **service mode** writes `status.json` per run (status, timings, video length —
what the metrics dashboard's run history reads). In evaluation mode the model block
additionally gains `evaluation` (LLM-judge) and `rouge` sub-blocks; a
full parameter sweep produces several model blocks (e.g. `vlm_diff0.01`, `vlm_fullres`)
and, with `--run-online-inference`, a `gemini` block.

## Notes

On Linux, if you hit permission errors writing outputs, run `sudo chown 10001:10001 evaluation/results output`.
