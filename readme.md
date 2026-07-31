# Screen Recording Description Service

Generates text descriptions and summaries of desktop screen recordings using local Vision Language Models (defaults are **qwen3.5:4b** and **gemma3:4b**, served via **Ollama**) and, optionally, **Gemini** in the cloud. Outputs are scored by an LLM-as-judge (**phi4:14b**) plus BERTScore/ROUGE and can be explored in a Streamlit dashboard.

## Project Structure

```
src/screen_recording_description/
├── __init__.py           # Package marker
├── __main__.py           # CLI entry point
├── config.py             # All configurable parameters and prompts
├── vlm_inference.py      # Local Ollama VLM: frame inference, frame differencing, summarisation
├── online_inference.py   # Gemini (cloud VLM) inference worker
├── llm_inference.py      # LLM-as-judge evaluation + BERTScore/ROUGE scoring
├── pipeline.py           # Video processing orchestration
└── results.py            # Streamlit dashboard for exploring results
```

| Module | Purpose |
|---|---|
| `config.py` | Model settings, frame selection thresholds, evaluation prompts |
| `vlm_inference.py` | Run per-frame Ollama inference, compute frame diffs, summarise timelines |
| `online_inference.py` | Parallel Gemini inference worker |
| `llm_inference.py` | Load annotations, LLM-as-judge scoring, BERTScore/ROUGE |
| `pipeline.py` | Decode video, select key frames, describe each, summarise, evaluate |
| `results.py` | Streamlit dashboard comparing models/variants |
| `__main__.py` | CLI argument parsing and dispatch |

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

## Running in Docker (sandboxed)

Runs the pipeline in a locked-down container. Ollama stays on the host (for GPU) and the container connects to it.

Start Ollama on the host, bound to all interfaces so the container can reach it:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
ollama pull qwen3.5:4b && ollama pull gemma3:4b && ollama pull phi4:14b
```

Build the image:

```bash
docker compose build
```

Process a video:

```bash
docker compose run --rm pipeline --videos 457
docker compose run --rm pipeline --videos 457 --full-param-run
docker compose run --rm pipeline --model all --videos all-eligible-videos --full-param-run
```

The dataset is mounted read-only from `./data`; results are written to `./results`.

The evaluation step (BERTScore) uses `roberta-large`. The container reads it from your
host HuggingFace cache (`~/.cache/huggingface`, mounted read-only) and runs offline, so it
never downloads models itself. Make sure the model is cached on the host first — e.g. from a
host (non-Docker) run, or `huggingface-cli download roberta-large`.

To use Gemini, set `GEMINI_API_KEY` in `src/screen_recording_description/config.py`, then:

```bash
docker compose run --rm pipeline --videos 457 --run-online-inference
```

Explore results in the dashboard at http://127.0.0.1:8501:

```bash
docker compose --profile dashboard up dashboard
```

On Linux, if you hit permission errors writing to `./results`, run `sudo chown 10001:10001 results`.

## Video Dataset

Download the videos from the `multi` subdirectory on Hugging Face:

https://huggingface.co/datasets/ONE-Lab/GUI-World/tree/main/multi

Place the video files in:

```
data/GUI-World/multi/
```

## Usage

### Process a video

```bash
# Single video by ID
python -m screen_recording_description --videos 457

# Full parameter sweep: all diff thresholds + resolution/token variants (default set from config)
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

### Explore results in the dashboard

```bash
streamlit run src/screen_recording_description/results.py
```

### Configuration

All parameters are in `src/screen_recording_description/config.py` — frame selection thresholds, VLM model, prompts, etc.

### Custom Ollama endpoint

```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434 python -m screen_recording_description --videos 457
```

### Example output

Results are written to `results/<video_name>_<timestamp>_vlm_description.json`:

```json
{
  "video": "recording.mp4",
  "total_frames": 900,
  "fps": 30.0,
  "resolution": "2560x1440",
  "diff_threshold": 0.0002,
  "max_fps": 5,
  "frames_decoded": 900,
  "frames_processed": 47,
  "summary": "The user opened a browser and...",
  "timeline": [
    {
      "frame_number": 1,
      "timestamp_sec": 0.03,
      "latency_sec": 8.21,
      "frame_description": "The user has a web browser open showing..."
    }
  ]
}
```