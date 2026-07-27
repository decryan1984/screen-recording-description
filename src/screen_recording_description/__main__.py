"""CLI entry point: python -m screen_recording_description --videos <id> [<id> ...]"""
import argparse
import glob
import os
import re
from datetime import datetime

from .config import (
    DEFAULT_DIFF_THRESHOLD_VALUES,
    GUI_WORLD_BLACKLIST,
    GUI_WORLD_VIDEO_DIR,
    GUI_WORLD_MIN_VIDEO_LENGTH_SEC,
    MODEL_NAME,
    RESULTS_DIR,
    VLM_MODELS,
)
from .pipeline import (
    run_pipeline,
    perform_multi_threshold_inferences,
    perform_multi_variable_inferences,
    perform_gemini_baseline,
    write_run_config,
)

def main():
    parser = argparse.ArgumentParser(
        prog="screen-recording-description",
        description="Analyse desktop screen recordings with a Vision Language Model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to run, or 'all' for every model in VLM_MODELS "
             "(default: config MODEL_NAME)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Video ID(s) to process (e.g. 457 458), or 'all-eligible-videos' "
             "for every eligible GUI-World video (30s+, not blacklisted)")
    parser.add_argument(
        "--full-param-run",
        action="store_true",
        help="Run all parameter variants on every specified VLM: all diff thresholds plus the full-resolution and 128/512 token variants. ",
    )
    parser.add_argument(
        "--run-online-inference",
        action="store_true",
        help="Also run Gemini online inference (off by default; "
             "requires the GEMINI_API_KEY environment variable)",
    )
    args = parser.parse_args()

    # Resolve video list
    if args.videos is None:
        parser.error("No video specified; --videos <id> [...] or --videos all-eligible-videos is required")
    if args.videos == ["all-eligible-videos"]:
        video_paths = get_guiworld_videos()
        if not video_paths:
            parser.error(f"No videos meeting the criteria were found in {GUI_WORLD_VIDEO_DIR}")
        print(f"All eligible GUI-World videos: {len(video_paths)}")
    else:
        video_paths = [get_video_path(v) for v in args.videos]

    # Resolve the model list: 'all' runs every configured VLM, otherwise a
    # single model (defaulting to config MODEL_NAME).
    if args.model == "all":
        models = VLM_MODELS
    elif args.model:
        models = [args.model]
    else:
        models = [MODEL_NAME]

    # Resolve thresholds once (shared by every video in the run)
    thresholds = DEFAULT_DIFF_THRESHOLD_VALUES if args.full_param_run else None

    # Create a single shared run directory for all videos
    run_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    # Write run-level config once (prompts, params) shared across all videos
    write_run_config(run_dir, model_name=models[0], thresholds=thresholds)
    print(f"Run directory: {run_dir}")
    print(f"Videos: {len(video_paths)}\n")

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'-'*60}")
        print(f"Video {i}/{len(video_paths)}: {video_path}")
        print(f"{'-'*60}")

        if thresholds is not None:
            # Sweep every selected VLM through the full parameter set.
            for m in models:
                perform_multi_threshold_inferences(video_path, thresholds, model_name=m,
                                                    run_dir=run_dir)
                perform_multi_variable_inferences(video_path, model_name=m,
                                                  run_dir=run_dir)
            # Gemini runs once at baseline params (no local sweep applies).
            if args.run_online_inference:
                perform_gemini_baseline(video_path, run_dir=run_dir)
        else:
            for j, m in enumerate(models):
                # Attach the Gemini online worker to only the first pass so it
                # doesn't run once per model.
                run_online = args.run_online_inference and j == 0
                run_pipeline(video_path, model_name=m,
                             skip_online=not run_online, run_dir=run_dir)

    print(f"\n{'-'*60}")
    print(f"Run complete. {len(video_paths)} videos processed.")
    print(f"Results: {run_dir}")
    print(f"{'-'*60}")

def get_video_path(video_id):
    """Resolve a video ID (e.g. '457') to its full path in GUI_WORLD_VIDEO_DIR."""
    matches = glob.glob(os.path.join(GUI_WORLD_VIDEO_DIR, f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(f"No video file found for ID '{video_id}' in {GUI_WORLD_VIDEO_DIR}")
    return matches[0]

def get_guiworld_videos():
    """Return sorted list of GUI-World video paths, excluding blacklisted videos
    and ones shorter than minimum size limit."""
    import av

    paths = sorted(
        glob.glob(os.path.join(GUI_WORLD_VIDEO_DIR, "*")),
        key=lambda p: (
            int(re.sub(r"\D", "", os.path.splitext(os.path.basename(p))[0]) or "999999"),
            p,
        ),
    )
    videos = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            video_id = int(name)
        except ValueError:
            continue
        if video_id in GUI_WORLD_BLACKLIST:
            continue
        # Check duration
        try:
            container = av.open(p)
            stream = container.streams.video[0]
            duration_sec = float(stream.frames) / float(stream.average_rate)
            container.close()
        except Exception:
            continue
        if duration_sec < GUI_WORLD_MIN_VIDEO_LENGTH_SEC:
            continue
        videos.append(p)
    return videos

if __name__ == "__main__":
    main()
