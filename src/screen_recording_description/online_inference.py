"""Online VLM inference via Google Gemini REST API.

Runs in parallel with the local Ollama pipeline and adds Gemini inference, evaluation, and performance results to the shared results file.
"""

import base64
import json
import os
import queue
import time
from threading import Thread

import av
import cv2
import requests

from .config import (
    GEMINI_API_KEY,
    GEMINI_API_URL,
    GEMINI_MODEL,
    GEMINI_MAX_TOKENS,
    TEMPERATURE,
    MAX_FRAME_WIDTH,
    FRAME_DESCRIPTION_PROMPT,
    USER_INTENT_PROMPT,
    INTENT_CHUNK_SIZE,
    INTENT_MAX_DESC_CHARS,
)
from .vlm_inference import _get_downscaled_frame
from .llm_inference import get_annotations, get_intent_evaluation, get_rouge_scores

def _get_gemini_url(model, method="generateContent"):
    return f"{GEMINI_API_URL}/{model}:{method}?key={GEMINI_API_KEY}"


def _get_metrics(resp_json):
    """Extract token counts from the Gemini response."""
    usage = resp_json.get("usageMetadata", {})
    return {
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0),
    }


def _get_text(resp_json):
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        return ""


def _get_gemini_response(payload, max_retries=4):
    """POST a payload to the Gemini API and return the parsed JSON response.
    Retries with exponential backoff on 429 (rate limit) and 5xx (server) errors.
    """
    url = _get_gemini_url(GEMINI_MODEL)
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=120)
        retryable = resp.status_code == 429 or resp.status_code >= 500
        if retryable and attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"  [Gemini] {resp.status_code} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

def get_gemini_description(frame_bgr, prompt=FRAME_DESCRIPTION_PROMPT):
    """Send a single frame to Gemini and return (text, metrics)."""
    if MAX_FRAME_WIDTH > 0:
        frame_bgr = _get_downscaled_frame(frame_bgr, MAX_FRAME_WIDTH)

    _, buf = cv2.imencode(".png", frame_bgr)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }

    data = _get_gemini_response(payload)
    return _get_text(data), _get_metrics(data)


def _get_prompt_response(text_prompt, max_tokens=GEMINI_MAX_TOKENS):
    """Send a text-only prompt to Gemini and return (text, metrics)."""
    payload = {
        "contents": [{"parts": [{"text": text_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": TEMPERATURE,
        },
    }
    data = _get_gemini_response(payload)
    return _get_text(data), _get_metrics(data)


def generate_gemini_user_intent(
    timeline,
    max_entries_per_chunk=INTENT_CHUNK_SIZE,
    max_desc_chars=INTENT_MAX_DESC_CHARS,
):
    """Infer the user's overall intent from per-frame descriptions via Gemini.

    Returns ``(intent_text, metrics_list)``.
    """
    if not timeline:
        return "", []

    entries = [
        f"[{e['timestamp_sec']}s] {e['frame_description'][:max_desc_chars]}"
        for e in timeline
    ]

    all_metrics = []

    if len(entries) <= max_entries_per_chunk:
        prompt = USER_INTENT_PROMPT.format(entries="\n".join(entries))
        text, metrics = _get_prompt_response(prompt)
        all_metrics.append(metrics)
        return text.strip(), all_metrics

    chunk_summaries = []
    for i in range(0, len(entries), max_entries_per_chunk):
        chunk = entries[i : i + max_entries_per_chunk]
        t_start = timeline[i]["timestamp_sec"]
        t_end = timeline[min(i + max_entries_per_chunk, len(timeline)) - 1]["timestamp_sec"]
        print(
            f"  [Gemini] Summarising chunk {len(chunk_summaries)+1} "
            f"({t_start}s – {t_end}s, {len(chunk)} entries)..."
        )
        prompt = USER_INTENT_PROMPT.format(entries="\n".join(chunk))
        text, metrics = _get_prompt_response(prompt)
        chunk_summaries.append(text)
        all_metrics.append(metrics)

    print(f"  [Gemini] Merging {len(chunk_summaries)} chunk summaries into a final intent synopsis.")
    combined = "\n".join(f"[Part {i+1}] {s}" for i, s in enumerate(chunk_summaries))
    final_prompt = USER_INTENT_PROMPT.format(entries=combined)
    text, metrics = _get_prompt_response(final_prompt)
    all_metrics.append(metrics)
    return text.strip(), all_metrics


def _process_frame_queue(frame_queue, video_path, output_json_path, total_frames):
    """Pull frames from queue, describe each and infer the user's intent."""
    timeline = []

    while True:
        item = frame_queue.get()
        if item is None: # there are no more frames
            break

        frame_bgr, frame_idx, timestamp_sec = item

        start = time.perf_counter()
        try:
            description, metrics = get_gemini_description(frame_bgr, FRAME_DESCRIPTION_PROMPT)
        except Exception as exc:
            print(f"  [Gemini] Frame {frame_idx} failed: {exc}")
            continue
        latency = time.perf_counter() - start

        entry = {
            "frame_number": frame_idx,
            "timestamp_sec": round(timestamp_sec, 2),
            "latency_sec": round(latency, 2),
            "frame_description": description,
            "prompt_tokens": metrics["prompt_tokens"],
            "completion_tokens": metrics["completion_tokens"],
        }
        timeline.append(entry)

        print(
            f"  [Gemini] [{frame_idx}/{total_frames}] "
            f"time={entry['timestamp_sec']}s (latency={entry['latency_sec']}s)"
        )
        desc_preview = entry["frame_description"][:200]
        if len(entry["frame_description"]) > 200:
            desc_preview += "..."
        print(f"    {desc_preview}\n")

    # Infer the user's overall intent
    intent = ""
    intent_latency = 0.0
    intent_metrics = []
    if timeline:
        print("  [Gemini] Inferring user intent...")
        start = time.perf_counter()
        try:
            intent, intent_metrics = generate_gemini_user_intent(timeline)
        except Exception as exc:
            print(f"  [Gemini] Intent inference failed: {exc}")
        intent_latency = time.perf_counter() - start
        print(f"\n  [Gemini] --- Inferred Intent ({intent_latency:.1f}s) ---")
        print(f"  {intent}\n")

    # Build the gemini block in memory
    gemini_block = {
        "model": GEMINI_MODEL,
        "max_tokens": GEMINI_MAX_TOKENS,
        "intent": intent,
        "timeline": timeline,
    }

    # Run evaluation
    eval_latency = 0.0
    if intent:
        eval_start = time.perf_counter()
        annotations = get_annotations(video_path)
        if annotations is None:
            print("  [Gemini] Skipping evaluation: no reference annotations found.")
        else:
            print("  [Gemini] Running evaluation...")
            evaluation = get_intent_evaluation(intent, timeline, annotations)
            print(
                f"  [Gemini] Composite score: {evaluation['composite_score']}/{evaluation['max_score']} "
                f"({evaluation['latency_sec']:.1f}s)"
            )
            gemini_block["evaluation"] = evaluation

            print("  [Gemini] Running ROUGE...")
            rouge_result = get_rouge_scores(timeline, annotations)
            if rouge_result:
                print(
                    f"  [Gemini] ROUGE-1 F1={rouge_result['rouge1']['f1']:.4f}  "
                    f"ROUGE-2 F1={rouge_result['rouge2']['f1']:.4f}  "
                    f"ROUGE-L F1={rouge_result['rougeL']['f1']:.4f}"
                )
                gemini_block["rouge"] = rouge_result
        eval_latency = time.perf_counter() - eval_start

    # Performance metrics — same structure and maths as the VLM pipeline so the
    # dashboard computes Gemini efficiency identically. Per-frame latency_sec here
    # is the API round-trip time (network and inference).
    if timeline:
        from .pipeline import _compute_performance, _standalone_pipeline_sec
        gemini_block["performance"] = _compute_performance(
            timeline, intent_metrics, GEMINI_MODEL,
            intent_latency, _standalone_pipeline_sec(timeline, intent_latency, eval_latency),
        )

    # Single atomic merge into the shared result file
    if os.path.exists(output_json_path):
        with open(output_json_path) as f:
            result = json.load(f)
    else:
        result = {}
    result["gemini"] = gemini_block
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [Gemini] Merged results into {output_json_path}")


def start_online_worker(video_path, output_json_path):
    if not GEMINI_API_KEY:
        print("  [Gemini] Skipping online inference — GEMINI_API_KEY not set.")
        return None, None

    # Read total_frames from the video file for progress display
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    container.close()

    q = queue.Queue()
    thread = Thread(
        target=_process_frame_queue,
        args=(q, video_path, output_json_path, total_frames),
        daemon=True,
    )
    thread.start()
    print(f"  [Gemini] Online worker started → {output_json_path}")
    return q, thread
