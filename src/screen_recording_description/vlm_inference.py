"""VLM interaction: model loading, frame selection, per-frame inference, and frame differencing."""

import base64
import math
import sys
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import requests

from .config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_FRAME_WIDTH,
    FRAME_DESCRIPTION_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_CHUNK_SIZE,
    SUMMARY_MAX_DESC_CHARS,
    ENABLE_GRAYSCALE_CONVERSION,
)


def get_vlm(model_name=MODEL_NAME):
    """Verify the Ollama model is available and return an inference function.
    Returns an inference function and the model info dict.
    """
    print(f"Checking Ollama model: {model_name} ...")
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/show", json={"name": model_name})
        resp.raise_for_status()
        model_info = resp.json()
    except requests.ConnectionError:
        sys.exit(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
    except requests.HTTPError:
        sys.exit(f"Model '{model_name}' not found in Ollama.")

    print(f"Model ready: {model_name}")

    def inference_func(frame_bgr, prompt=FRAME_DESCRIPTION_PROMPT,
                       max_frame_width=MAX_FRAME_WIDTH, max_tokens=MAX_TOKENS):
        return get_vlm_description(model_name, frame_bgr, prompt,
                                 max_frame_width=max_frame_width, max_tokens=max_tokens)

    return inference_func, model_info


def _get_downscaled_frame(frame, max_width):
    """Downscale a frame using the smallest integer ratio that keeps width <= max_width.

    Integer ratios (2×, 3×, …) produce cleaner results than fractional scaling,
    especially for on-screen text.  If the frame is already at or below max_width,
    it is returned unchanged.
    """
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame

    # Find smallest integer ratio where w // ratio <= max_width
    ratio = math.ceil(w / max_width)
    new_w = w // ratio
    new_h = h // ratio
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _get_metrics(resp_json):
    """Extract token counts and server-side timing from an Ollama response.

    ``inference_sec`` is the model's own compute time with the model-load
    duration removed, so a cold start (or a mid-run model reload) does not
    inflate it. Ollama reports both ``total_duration`` and ``load_duration``
    in nanoseconds.
    """
    total_ns = resp_json.get("total_duration", 0)
    load_ns = resp_json.get("load_duration", 0)
    return {
        "prompt_tokens": resp_json.get("prompt_eval_count", 0),
        "completion_tokens": resp_json.get("eval_count", 0),
        "inference_sec": max(total_ns - load_ns, 0) / 1e9,
        "load_sec": load_ns / 1e9,
    }


def get_vlm_description(model_name, frame_bgr, prompt=FRAME_DESCRIPTION_PROMPT,
                      max_frame_width=MAX_FRAME_WIDTH, max_tokens=MAX_TOKENS):
    """Run VLM inference on a single BGR frame via Ollama.

    Returns (response_text, metrics_dict).
    """
    if max_frame_width > 0:
        frame_bgr = _get_downscaled_frame(frame_bgr, max_frame_width)

    # Encode frame as base64 PNG for the Ollama API
    _, buf = cv2.imencode(".png", frame_bgr)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": TEMPERATURE,
        },
    }

    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"], _get_metrics(data)


def get_frame_difference(frame_a, frame_b, norm_width=640,
                             enable_grayscale=ENABLE_GRAYSCALE_CONVERSION):
    """Compute a normalised pixel difference score between two frames.

    Used to skip near-identical frames and only send meaningful changes to the VLM.
    Both frames are downscaled to norm_width before comparison so that the
    threshold behaves consistently regardless of source resolution.
    Returns a float in [0, 1] where 0 = identical.
    """
    if frame_a is None or frame_b is None:
        return 1.0

    frame_a = _get_downscaled_frame(frame_a, norm_width)
    frame_b = _get_downscaled_frame(frame_b, norm_width)

    if enable_grayscale:
        # Single-channel comparison (faster, may miss colour-only changes)
        a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        # Full colour comparison
        a, b = frame_a, frame_b
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff)) / 255.0


def print_model_info(model_info):
    """Print model details retrieved from Ollama."""
    details = model_info.get("details", {})
    model_info_fields = model_info.get("model_info", {})

    print("\nModel Info (from Ollama)")
    for key in ("family", "parameter_size", "quantization_level"):
        if key in details:
            print(f"  {key}: {details[key]}")

    # Print any extra info Ollama exposes (parameter count, context length, etc.)
    for key, value in model_info_fields.items():
        if "param" in key.lower() or "context" in key.lower():
            print(f"  {key}: {value}")


def _get_prompt_response(text_prompt, model_name=MODEL_NAME, max_tokens=1024):
    """Send a text-only prompt to Ollama and return (response_text, metrics_dict)."""
    payload = {
        "model": model_name,
        "prompt": text_prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": TEMPERATURE,
        },
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"], _get_metrics(data)


def get_timeline_summary(
    timeline,
    model_name=MODEL_NAME,
    max_entries_per_chunk=SUMMARY_CHUNK_SIZE,
    max_desc_chars=SUMMARY_MAX_DESC_CHARS,
):
    """Generate a composite description of the full video from per-frame descriptions.

    For large timelines, splits entries into chunks, summarises each chunk,
    then produces a final summary-of-summaries.
    """
    if not timeline:
        return "", []

    entries = [
        f"[Frame {e['frame_number']}] [{e['timestamp_sec']}s] {e['frame_description'][:max_desc_chars]}"
        for e in timeline
    ]

    all_metrics = []

    # If small enough, summarise in one pass
    if len(entries) <= max_entries_per_chunk:
        prompt = SUMMARY_PROMPT.format(entries="\n".join(entries))
        text, metrics = _get_prompt_response(prompt, model_name)
        all_metrics.append(metrics)
        return text, all_metrics

    chunk_summaries = []
    for i in range(0, len(entries), max_entries_per_chunk):
        chunk = entries[i : i + max_entries_per_chunk]
        t_start = timeline[i]["timestamp_sec"]
        t_end = timeline[min(i + max_entries_per_chunk, len(timeline)) - 1][
            "timestamp_sec"
        ]
        print(
            f"  Summarising chunk {len(chunk_summaries)+1} "
            f"({t_start}s – {t_end}s, {len(chunk)} entries)..."
        )
        prompt = SUMMARY_PROMPT.format(entries="\n".join(chunk))
        text, metrics = _get_prompt_response(prompt, model_name)
        chunk_summaries.append(text)
        all_metrics.append(metrics)

    print(f"  Merging {len(chunk_summaries)} chunk summaries...")
    combined = "\n".join(
        f"[Part {i+1}] {s}" for i, s in enumerate(chunk_summaries)
    )
    final_prompt = SUMMARY_PROMPT.format(entries=combined)
    text, metrics = _get_prompt_response(final_prompt, model_name)
    all_metrics.append(metrics)
    return text, all_metrics
