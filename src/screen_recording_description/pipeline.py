"""Video processing pipeline: decode frames, select key frames, describe, and summarise."""

import json
import math
import os
import statistics
import time
from datetime import datetime

import av

from .config import (
    DEFAULT_VLM,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_FRAME_WIDTH,
    ENABLE_GRAYSCALE_CONVERSION,
    DEFAULT_FRAME_DIFF_THRESHOLD,
    MULTI_VARIABLE_CONFIGS,
    MAX_FPS,
    INTENT_CHUNK_SIZE,
    INTENT_MAX_DESC_CHARS,
    FRAME_DESCRIPTION_PROMPT,
    USER_INTENT_PROMPT,
    RESULTS_DIR,
    EVAL_MODEL_NAME,
    EVAL_PROMPT_INTENT,
    EVAL_PROMPT_ACCURACY,
    EVAL_PROMPT_COVERAGE,
    EVAL_PROMPT_NON_REPETITION,
)

from .vlm_inference import (
    get_vlm,
    get_frame_difference,
    print_model_info,
    generate_user_intent,
    get_model_memory,
    get_model_params,
)
from .llm_inference import (
    get_annotations,
    get_intent_evaluation,
    get_rouge_scores,
)
from .online_inference import start_online_worker


def _video_length_sec(video_stream, container):
    """Return the length of the video in seconds."""
    if video_stream.duration is not None:
        return round(float(video_stream.duration * video_stream.time_base), 2)
    if container.duration is not None:
        return round(container.duration / 1_000_000, 2)  # AV_TIME_BASE = 1e6
    return None


def describe_frame(frame_bgr, frame_idx, timestamp_sec, inference_func, prompt,
                   max_frame_width=MAX_FRAME_WIDTH, max_tokens=MAX_TOKENS):
    """Run VLM inference on a single frame and return a timeline entry."""
    start = time.perf_counter()
    description, metrics = inference_func(
        frame_bgr, prompt, max_frame_width=max_frame_width, max_tokens=max_tokens
    )
    # Prefer Ollama's server-side compute time (model-load excluded) so a cold
    # start or a mid-run model reload doesn't inflate the frame latency - fall
    # back to wall-clock if the response didn't report timings.
    latency = metrics.get("inference_sec") or (time.perf_counter() - start)

    entry = {
        "frame_number": frame_idx,
        "timestamp_sec": round(timestamp_sec, 2),
        "latency_sec": round(latency, 2),
        "frame_description": description,
        "prompt_tokens": metrics["prompt_tokens"],
        "completion_tokens": metrics["completion_tokens"],
    }
    return entry


def _intent_seconds(intent_metrics_list, wall_sec):
    """Intent-inference compute time with model-load excluded (as for frame latency).

    Sums Ollama's server-side ``inference_sec`` across intent chunks; falls
    back to wall-clock if those timings are unavailable."""
    total = sum(m.get("inference_sec", 0) for m in intent_metrics_list)
    return total or wall_sec


def _standalone_pipeline_sec(timeline, intent_latency, eval_latency):
    """Processing time attributable to a single variant: VLM inference for its
    own frames plus its own intent inference and evaluation.

    Independent of shared decode/diff/model-load overhead, so variants remain
    directly comparable whether they were produced by the single-video,
    multi-threshold (shared descriptions), or multi-variable path.
    """
    return sum(e["latency_sec"] for e in timeline) + intent_latency + eval_latency


def _compute_performance(timeline, intent_metrics_list, model_name,
                         intent_latency, standalone_pipeline_sec):
    """Compute aggregate performance metrics from the pipeline run."""
    latencies = [e["latency_sec"] for e in timeline]
    desc_lengths = [len(e["frame_description"]) for e in timeline]
    empty_count = sum(1 for e in timeline if not e["frame_description"])

    total_prompt_tokens = sum(e["prompt_tokens"] for e in timeline)
    total_completion_tokens = sum(e["completion_tokens"] for e in timeline)
    for m in intent_metrics_list:
        total_prompt_tokens += m["prompt_tokens"]
        total_completion_tokens += m["completion_tokens"]

    total_vlm_sec = sum(latencies)
    completion_tokens_in_frames = sum(e["completion_tokens"] for e in timeline)
    avg_tokens_per_sec = completion_tokens_in_frames / total_vlm_sec if total_vlm_sec > 0 else 0

    performance = {
        "vlm_model": model_name,
        "standalone_pipeline_sec": round(standalone_pipeline_sec, 2),
        "total_vlm_inference_sec": round(total_vlm_sec, 2),
        "intent_latency_sec": round(intent_latency, 2),
        "frames_processed": len(timeline),
        "frames_with_empty_description": empty_count,
        "avg_frame_latency_sec": round(statistics.mean(latencies), 2) if latencies else 0,
        "median_frame_latency_sec": round(statistics.median(latencies), 2) if latencies else 0,
        "min_frame_latency_sec": round(min(latencies), 2) if latencies else 0,
        "max_frame_latency_sec": round(max(latencies), 2) if latencies else 0,
        "avg_description_length_chars": round(statistics.mean(desc_lengths), 1) if desc_lengths else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens_per_sec": round(avg_tokens_per_sec, 1),
    }
    # Memory and parameter count. Local models only — Gemini exposes neither.
    if "gemini" not in str(model_name).lower():
        performance.update(get_model_memory(model_name))
        performance.update(get_model_params(model_name))
    return performance


def _save(result, output_json_path):
    """Merge *result* into the on-disk JSON so concurrent writers (e.g. the
    Gemini thread) are not clobbered."""
    if os.path.exists(output_json_path):
        with open(output_json_path) as f:
            on_disk = json.load(f)
        on_disk.update(result)
        result.update(on_disk)
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)


def write_run_config(run_dir, model_name=None, thresholds=None):
    """Write run-level configuration shared by every video to ``config.json``.

    This holds everything that is not specific to an individual video (model,
    parameters, prompts) so it is stored once per run rather than duplicated in
    every ``<video>.json``.  Merges with any existing file so repeated calls
    within a run accumulate rather than clobber.
    """
    model_name = model_name or DEFAULT_VLM
    config = {
        "model": model_name,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_frame_width": MAX_FRAME_WIDTH,
        "enable_grayscale_conversion": ENABLE_GRAYSCALE_CONVERSION,
        "max_fps": MAX_FPS,
        "default_diff_threshold": DEFAULT_FRAME_DIFF_THRESHOLD,
        "intent_chunk_size": INTENT_CHUNK_SIZE,
        "intent_max_desc_chars": INTENT_MAX_DESC_CHARS,
        "multi_variable_configs": MULTI_VARIABLE_CONFIGS,
        "frame_description_prompt": FRAME_DESCRIPTION_PROMPT,
        "user_intent_prompt": USER_INTENT_PROMPT,
        "eval_model": EVAL_MODEL_NAME,
        "eval_prompts": {
            "intent": EVAL_PROMPT_INTENT,
            "accuracy": EVAL_PROMPT_ACCURACY,
            "coverage": EVAL_PROMPT_COVERAGE,
            "non_repetition": EVAL_PROMPT_NON_REPETITION,
        },
    }
    if thresholds is not None:
        config["diff_thresholds"] = thresholds

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            on_disk = json.load(f)
        on_disk.update(config)
        config = on_disk
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def _run_evaluation(video_path, intent, timeline, model_block, result, output_json_path, label=""):
    """Look up reference annotations and run LLM-as-judge evaluation."""
    prefix = f"[{label}] " if label else ""
    annotations = get_annotations(video_path)
    if annotations is None:
        print(f"\n{prefix}Skipping evaluation: no reference annotations found for this video.")
        return

    print(f"\n{prefix}--- Evaluation (LLM-as-judge) ---")
    evaluation = get_intent_evaluation(intent, timeline, annotations)
    print(
        f"\n{prefix}Composite score: {evaluation['composite_score']}/{evaluation['max_score']} "
        f"({evaluation['latency_sec']:.1f}s)"
    )
    model_block["evaluation"] = evaluation
    _save(result, output_json_path)
    print(f"{prefix}Updated {output_json_path} with evaluation.")

    print(f"\n{prefix}--- Evaluation (ROUGE) ---")
    rouge_result = get_rouge_scores(timeline, annotations)
    if rouge_result:
        print(
            f"  {prefix}ROUGE-1 F1={rouge_result['rouge1']['f1']:.4f}  "
            f"ROUGE-2 F1={rouge_result['rouge2']['f1']:.4f}  "
            f"ROUGE-L F1={rouge_result['rougeL']['f1']:.4f}"
        )
        model_block["rouge"] = rouge_result
        _save(result, output_json_path)
        print(f"{prefix}Updated {output_json_path} with ROUGE.")


def run_pipeline(video_path, model_name=None, skip_online=False, run_dir=None, evaluate=True, analysis_prompt=None):
    """Process a video file end-to-end: describe selected frames and generate a summary.

    evaluate=True (evaluation mode) scores the summary against reference
    annotations; evaluate=False (service mode) skips scoring entirely.

    analysis_prompt overrides the default user intent analysis run over the timeline.
    Passing a custom prompt (e.g. did the user comply with company policy?) will
    produce a response specific to the users requirements instead.
    None keeps the default intent prompt.
    """
    model_name = model_name or DEFAULT_VLM
    analysis_prompt = analysis_prompt or USER_INTENT_PROMPT
    diff_threshold = DEFAULT_FRAME_DIFF_THRESHOLD
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    output_json_path = os.path.join(run_dir, f"{video_name}.json")

    input_container = av.open(video_path)
    video_stream = input_container.streams.video[0]
    total_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    width, height = video_stream.width, video_stream.height
    video_length_sec = _video_length_sec(video_stream, input_container)

    min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

    print(f"\nVideo: {video_path} ({total_frames} frames, {fps:.0f} FPS, {width}x{height})")
    print(
        f"Config: model={model_name}, max_tokens={MAX_TOKENS}, "
        f"max_frame_width={MAX_FRAME_WIDTH}, enable_grayscale_conversion={ENABLE_GRAYSCALE_CONVERSION}, "
        f"diff_threshold={diff_threshold}, max_fps={MAX_FPS}, "
        f"intent_chunk_size={INTENT_CHUNK_SIZE}, intent_max_desc_chars={INTENT_MAX_DESC_CHARS}"
    )

    print("Loading VLM...")
    inference_func, model_info = get_vlm(model_name)
    print_model_info(model_info)

    print(f"Output: {output_json_path}")
    print(f"Prompt: {FRAME_DESCRIPTION_PROMPT}\n")

    # Start online (Gemini) worker if enabled
    online_queue, online_thread = None, None
    if not skip_online:
        online_queue, online_thread = start_online_worker(
            video_path, output_json_path,
        )

    timeline = []
    frame_count = 0
    processed_count = 0
    prev_frame = None
    last_processed_ts = 0.0

    for av_frame in input_container.decode(video=0):
        frame = av_frame.to_ndarray(format="bgr24")
        frame_count += 1
        timestamp_sec = frame_count / fps
        is_first = frame_count == 1
        is_last = frame_count == total_frames

        if not is_first and not is_last:
            # Max FPS — skip frames too close to the last processed one
            if min_interval > 0 and (timestamp_sec - last_processed_ts) < min_interval:
                continue

            # Frame differencing — skip near-identical frames
            diff_score = get_frame_difference(prev_frame, frame)
            if diff_score < diff_threshold:
                continue
        else:
            diff_score = get_frame_difference(prev_frame, frame)

        prev_frame = frame.copy()
        last_processed_ts = timestamp_sec
        processed_count += 1

        # Push frame to Gemini worker (before local inference so they overlap)
        if online_queue is not None:
            online_queue.put((frame.copy(), frame_count, timestamp_sec))

        entry = describe_frame(
            frame, frame_count, timestamp_sec, inference_func, FRAME_DESCRIPTION_PROMPT
        )
        timeline.append(entry)

        print(
            f"[{frame_count}/{total_frames}] time={entry['timestamp_sec']}s "
            f"(diff={diff_score:.5f}, latency={entry['latency_sec']}s)"
        )
        print(
            f"  {entry['frame_description'][:200]}"
            f"{'...' if len(entry['frame_description']) > 200 else ''}\n"
        )

    input_container.close()

    # Signal online worker that all frames have been sent
    if online_queue is not None:
        online_queue.put(None)

    # Build top-level result with nested model blocks
    if MAX_FRAME_WIDTH > 0 and width > MAX_FRAME_WIDTH:
        ratio = math.ceil(width / MAX_FRAME_WIDTH)
        reduced_res = f"{width // ratio}x{height // ratio}"
    else:
        reduced_res = None

    result = {
        "video": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "video_length_sec": video_length_sec,
        "resolution": f"{width}x{height}",
        "reduced_resolution": reduced_res,
        "frames_decoded": frame_count,
        "frames_processed": processed_count,
        "vlm": {
            "model": model_name,
            "diff_threshold": diff_threshold,
            "analysis_prompt": analysis_prompt,
            "intent": "",
            "timeline": timeline,
        },
    }
    _save(result, output_json_path)
    print(f"\nProcessed {processed_count}/{frame_count} frames. Output saved to {output_json_path}")

    # Infer the user's overall intent from all per-frame descriptions
    intent_latency = 0.0
    eval_latency = 0.0
    intent_metrics_list = []
    if timeline:
        print("\nInferring user intent...")
        start = time.perf_counter()
        intent, intent_metrics_list = generate_user_intent(
            timeline, model_name=model_name, analysis_prompt=analysis_prompt
        )
        intent_latency = _intent_seconds(intent_metrics_list, time.perf_counter() - start)
        result["vlm"]["intent"] = intent
        _save(result, output_json_path)
        print(f"\n--- Inferred Intent ({intent_latency:.1f}s) ---")
        print(intent)

        eval_start = time.perf_counter()
        if evaluate:
            _run_evaluation(video_path, intent, timeline, result["vlm"], result, output_json_path)
        eval_latency = time.perf_counter() - eval_start

    # Compute and save performance metrics
    performance = _compute_performance(
        timeline, intent_metrics_list, model_name,
        intent_latency, _standalone_pipeline_sec(timeline, intent_latency, eval_latency),
    )
    result["vlm"]["performance"] = performance
    _save(result, output_json_path)

    print(f"\n--- Performance ---")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    print(f"Updated {output_json_path} with performance metrics.")

    # Wait for the online worker to finish
    if online_thread is not None:
        print("\nWaiting for Gemini online worker to finish...")
        online_thread.join()
        print("Gemini online worker complete.")


def _model_prefix(model_name):
    """Result-block key prefix for a model. Strips '_' so the dashboard's
    first-underscore split cleanly separates the model prefix from the variant
    suffix (e.g. 'qwen3.5:4b_diff0.01' -> model='qwen3.5:4b', suffix='diff0.01')."""
    return (model_name or DEFAULT_VLM).replace("_", "-")


def perform_multi_threshold_inferences(video_path, thresholds, model_name=None, run_dir=None,
                                       block_prefix=None):
    """Run the pipeline once with multiple diff thresholds.

    Each unique frame is described by the VLM exactly once.  Per-threshold
    timelines are assembled from the shared frame descriptions, then
    summarised and evaluated independently.  All threshold results are stored
    in a single ``results/<timestamp>/<video>.json`` with keys like
    ``vlm_diff0.005``, ``vlm_diff0.01``, etc.
    """
    pipeline_start = time.perf_counter()
    model_name = model_name or DEFAULT_VLM
    block_prefix = block_prefix or _model_prefix(model_name)
    thresholds = sorted(thresholds)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if run_dir is None:
        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RESULTS_DIR, ts_label)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{video_name}.json")

    # Open video
    input_container = av.open(video_path)
    video_stream = input_container.streams.video[0]
    total_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    width, height = video_stream.width, video_stream.height
    min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

    print(f"\nVideo: {video_path} ({total_frames} frames, {fps:.0f} FPS, {width}x{height})")
    print(f"Diff thresholds: {thresholds}")
    print(f"Output: {out_path}\n")

    # Load VLM once
    print("Loading VLM...")
    inference_func, model_info = get_vlm(model_name)
    print_model_info(model_info)

    # Per-threshold state
    t_state = {}
    for t in thresholds:
        t_state[t] = {
            "prev_frame": None,
            "last_ts": 0.0,
            "selected": [],          # ordered list of frame_numbers
        }

    # Cache of VLM results keyed by frame_number
    described: dict[int, dict] = {}
    frame_count = 0

    for av_frame in input_container.decode(video=0):
        frame = av_frame.to_ndarray(format="bgr24")
        frame_count += 1
        timestamp_sec = frame_count / fps
        is_first = frame_count == 1
        is_last = frame_count == total_frames

        selected_by: list[float] = []

        for t in thresholds:
            st = t_state[t]
            if is_first or is_last:
                selected_by.append(t)
                st["prev_frame"] = frame.copy()
                st["last_ts"] = timestamp_sec
                st["selected"].append(frame_count)
                continue

            # Max FPS gate
            if min_interval > 0 and (timestamp_sec - st["last_ts"]) < min_interval:
                continue

            diff = get_frame_difference(st["prev_frame"], frame)
            if diff >= t:
                selected_by.append(t)
                st["prev_frame"] = frame.copy()
                st["last_ts"] = timestamp_sec
                st["selected"].append(frame_count)

        # Describe with VLM once if any threshold selected this frame
        if selected_by and frame_count not in described:
            entry = describe_frame(
                frame, frame_count, timestamp_sec,
                inference_func, FRAME_DESCRIPTION_PROMPT,
            )
            described[frame_count] = entry
            thresholds_str = ", ".join(f"{t}" for t in selected_by)
            print(
                f"[{frame_count}/{total_frames}] time={entry['timestamp_sec']}s "
                f"(latency={entry['latency_sec']}s, thresholds=[{thresholds_str}])"
            )
            print(
                f"  {entry['frame_description'][:200]}"
                f"{'...' if len(entry['frame_description']) > 200 else ''}\n"
            )

    input_container.close()

    # Report frame counts per threshold
    print(f"\n--- Frame Selection Summary ---")
    print(f"Unique frames described: {len(described)}")
    for t in thresholds:
        n = len(t_state[t]["selected"])
        print(f"  threshold={t}: {n} frames selected")

    # Build shared result with top-level metadata
    if MAX_FRAME_WIDTH > 0 and width > MAX_FRAME_WIDTH:
        ratio = math.ceil(width / MAX_FRAME_WIDTH)
        reduced_res = f"{width // ratio}x{height // ratio}"
    else:
        reduced_res = None

    result = {
        "video": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "reduced_resolution": reduced_res,
        "frames_decoded": frame_count,
    }

    # Build per-threshold model blocks
    for t in thresholds:
        selected = t_state[t]["selected"]
        timeline = [described[fn] for fn in selected if fn in described]
        block_key = f"{block_prefix}_diff{t}"

        block = {
            "model": model_name,
            "diff_threshold": t,
            "frames_processed": len(timeline),
            "intent": "",
            "timeline": timeline,
        }

        print(f"\n{'='*60}")
        print(f"Threshold {t} — {len(timeline)} frames")
        print(f"{'='*60}")

        intent_latency = 0.0
        eval_latency = 0.0
        intent_metrics_list = []
        if timeline:
            print("Inferring user intent...")
            start = time.perf_counter()
            intent, intent_metrics_list = generate_user_intent(timeline, model_name=model_name)
            intent_latency = _intent_seconds(intent_metrics_list, time.perf_counter() - start)
            print(f"Intent ({intent_latency:.1f}s): {intent}\n")
            block["intent"] = intent

            # Temporarily attach block to result so _run_evaluation can save
            result[block_key] = block
            _save(result, out_path)
            eval_start = time.perf_counter()
            _run_evaluation(video_path, intent, timeline, block, result, out_path,
                            label=f"diff={t}")
            eval_latency = time.perf_counter() - eval_start

        performance = _compute_performance(
            timeline, intent_metrics_list, model_name,
            intent_latency, _standalone_pipeline_sec(timeline, intent_latency, eval_latency),
        )
        block["performance"] = performance
        result[block_key] = block
        _save(result, out_path)
        print(f"Saved {block_key} to {out_path}")

    total_time = time.perf_counter() - pipeline_start
    print(f"\nMulti-threshold inference complete in {total_time:.1f}s — {len(described)} unique frames described across {len(thresholds)} thresholds.")


def perform_multi_variable_inferences(video_path, model_name=None, run_dir=None,
                                      diff_threshold=DEFAULT_FRAME_DIFF_THRESHOLD,
                                      block_prefix=None):
    """Run one independent inference pass per entry in ``MULTI_VARIABLE_CONFIGS``.

    Each variant overrides selected parameters (``max_frame_width``,
    ``max_tokens``, ``enable_grayscale``) at a fixed diff threshold.  Because
    these parameters change how frames are selected and described, every variant
    is a full, independent pass — frame descriptions are not shared between
    variants.  Results are added as ``vlm_<name>`` blocks to the shared
    ``results/<timestamp>/<video>.json``.
    """
    model_name = model_name or DEFAULT_VLM
    block_prefix = block_prefix or _model_prefix(model_name)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if run_dir is None:
        run_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{video_name}.json")

    print("Loading VLM...")
    inference_func, model_info = get_vlm(model_name)
    print_model_info(model_info)

    for cfg in MULTI_VARIABLE_CONFIGS:
        name = cfg["name"]
        max_frame_width = cfg.get("max_frame_width", MAX_FRAME_WIDTH)
        max_tokens = cfg.get("max_tokens", MAX_TOKENS)
        enable_grayscale = cfg.get("enable_grayscale", ENABLE_GRAYSCALE_CONVERSION)

        print(f"\n{'='*60}")
        print(
            f"Variant '{name}' — diff_threshold={diff_threshold}, "
            f"max_frame_width={max_frame_width}, max_tokens={max_tokens}, "
            f"enable_grayscale={enable_grayscale}"
        )
        print(f"{'='*60}")

        _run_variable_pass(
            video_path, out_path, f"{block_prefix}_{name}", name, model_name, inference_func,
            diff_threshold, max_frame_width, max_tokens, enable_grayscale,
        )

    print(f"\nMulti-variable inference complete — {len(MULTI_VARIABLE_CONFIGS)} variants.")


def _run_variable_pass(video_path, out_path, block_key, label, model_name, inference_func,
                       diff_threshold, max_frame_width, max_tokens, enable_grayscale):
    """Single decode/select/describe/summarise/evaluate pass for one variant."""
    input_container = av.open(video_path)
    video_stream = input_container.streams.video[0]
    total_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    width, height = video_stream.width, video_stream.height
    min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

    timeline = []
    frame_count = 0
    prev_frame = None
    last_processed_ts = 0.0

    for av_frame in input_container.decode(video=0):
        frame = av_frame.to_ndarray(format="bgr24")
        frame_count += 1
        timestamp_sec = frame_count / fps
        is_first = frame_count == 1
        is_last = frame_count == total_frames

        if not is_first and not is_last:
            if min_interval > 0 and (timestamp_sec - last_processed_ts) < min_interval:
                continue
            diff_score = get_frame_difference(prev_frame, frame, enable_grayscale=enable_grayscale)
            if diff_score < diff_threshold:
                continue
        else:
            diff_score = get_frame_difference(prev_frame, frame, enable_grayscale=enable_grayscale)

        prev_frame = frame.copy()
        last_processed_ts = timestamp_sec

        entry = describe_frame(
            frame, frame_count, timestamp_sec, inference_func, FRAME_DESCRIPTION_PROMPT,
            max_frame_width=max_frame_width, max_tokens=max_tokens,
        )
        timeline.append(entry)
        print(
            f"[{frame_count}/{total_frames}] time={entry['timestamp_sec']}s "
            f"(diff={diff_score:.5f}, latency={entry['latency_sec']}s)"
        )
        print(
            f"  {entry['frame_description'][:200]}"
            f"{'...' if len(entry['frame_description']) > 200 else ''}\n"
        )

    input_container.close()

    if max_frame_width > 0 and width > max_frame_width:
        ratio = math.ceil(width / max_frame_width)
        reduced_res = f"{width // ratio}x{height // ratio}"
    else:
        reduced_res = None

    block = {
        "model": model_name,
        "diff_threshold": diff_threshold,
        "max_frame_width": max_frame_width,
        "max_tokens": max_tokens,
        "enable_grayscale": enable_grayscale,
        "reduced_resolution": reduced_res,
        "frames_processed": len(timeline),
        "intent": "",
        "timeline": timeline,
    }

    # Minimal video-specific metadata (merged, not clobbered, by _save)
    result = {
        "video": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
    }

    intent_latency = 0.0
    eval_latency = 0.0
    intent_metrics_list = []
    if timeline:
        print("Inferring user intent...")
        start = time.perf_counter()
        intent, intent_metrics_list = generate_user_intent(timeline, model_name=model_name)
        intent_latency = _intent_seconds(intent_metrics_list, time.perf_counter() - start)
        print(f"Intent ({intent_latency:.1f}s): {intent}\n")
        block["intent"] = intent

        result[block_key] = block
        _save(result, out_path)
        eval_start = time.perf_counter()
        _run_evaluation(video_path, intent, timeline, block, result, out_path,
                        label=label)
        eval_latency = time.perf_counter() - eval_start

    performance = _compute_performance(
        timeline, intent_metrics_list, model_name,
        intent_latency, _standalone_pipeline_sec(timeline, intent_latency, eval_latency),
    )
    block["performance"] = performance
    result[block_key] = block
    _save(result, out_path)
    print(f"Saved {block_key} to {out_path}")


def perform_gemini_baseline(video_path, run_dir=None):
    """Run a single baseline Gemini pass and merge a ``"gemini"`` block into the
    shared ``results/<timestamp>/<video>.json``.

    Frame selection mirrors ``run_pipeline``'s online feed (baseline
    ``DEFAULT_FRAME_DIFF_THRESHOLD`` + ``MAX_FPS`` gate). Runs synchronously so
    it can be called after the local-VLM runs have written their blocks.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if run_dir is None:
        run_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{video_name}.json")

    print(f"\n{'='*60}")
    print(f"Gemini baseline — diff_threshold={DEFAULT_FRAME_DIFF_THRESHOLD}")
    print(f"{'='*60}")

    online_queue, online_thread = start_online_worker(video_path, out_path)
    if online_thread is None:
        return

    input_container = av.open(video_path)
    video_stream = input_container.streams.video[0]
    total_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

    frame_count = 0
    prev_frame = None
    last_processed_ts = 0.0

    for av_frame in input_container.decode(video=0):
        frame = av_frame.to_ndarray(format="bgr24")
        frame_count += 1
        timestamp_sec = frame_count / fps
        is_first = frame_count == 1
        is_last = frame_count == total_frames

        if not is_first and not is_last:
            if min_interval > 0 and (timestamp_sec - last_processed_ts) < min_interval:
                continue
            if get_frame_difference(prev_frame, frame) < DEFAULT_FRAME_DIFF_THRESHOLD:
                continue

        prev_frame = frame.copy()
        last_processed_ts = timestamp_sec
        online_queue.put((frame.copy(), frame_count, timestamp_sec))

    input_container.close()
    online_queue.put(None)
    print("\nWaiting for Gemini baseline worker to finish...")
    online_thread.join()
    print("Gemini baseline complete.")
