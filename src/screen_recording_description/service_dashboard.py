"""Live service monitoring dashboard.
Launch with Streamlit: streamlit run src/screen_recording_description/dashboard.py
"""

import glob
import json
import os
import re
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

from config import SERVICE_RUNS_DIR, GEMINI_COST_PER_FRAME_USD

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REFRESH_SEC = 5
OVERVIEW_LABEL = "Main Dashboard"

# Matches the results dashboard's palette.
COLOR_SCHEME = "tableau10"
TABLEAU_BLUE = "#4C78A8"

# Relative time windows for the main dashboard.
PERIODS = {
    "Last hour": 1 / 24,
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "All time": None,
}


def fetch_json(path):
    """GET {API_URL}{path} and return parsed JSON, or None on any failure."""
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def format_timestamp(value):
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def time_per_frame(processing_sec, frames_processed):
    if processing_sec and frames_processed:
        return round(processing_sec / frames_processed, 2)
    return None


def within_period(stamp, cutoff):
    """True if an ISO timestamp falls within the window (cutoff=None means all time)."""
    if cutoff is None:
        return True
    if not stamp:
        return False
    try:
        return datetime.fromisoformat(stamp) >= cutoff
    except ValueError:
        return False


def load_disk_runs():
    """Maps run_id -> {'status': {...}, 'videos': {name: data}} from output/runs/,
    newest first. A run is included once it has a status.json or a video result."""
    runs = {}
    if not os.path.isdir(SERVICE_RUNS_DIR):
        return runs
    for name in sorted(os.listdir(SERVICE_RUNS_DIR), reverse=True):
        run_dir = os.path.join(SERVICE_RUNS_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        status = {}
        status_path = os.path.join(run_dir, "status.json")
        if os.path.isfile(status_path):
            try:
                with open(status_path) as f:
                    status = json.load(f)
            except (OSError, ValueError):
                status = {}
        videos = {}
        for json_file in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
            if os.path.basename(json_file) == "status.json":
                continue
            try:
                with open(json_file) as f:
                    videos[os.path.splitext(os.path.basename(json_file))[0]] = json.load(f)
            except (OSError, ValueError):
                continue
        if status or videos:
            runs[name] = {"status": status, "videos": videos}
    return runs


def get_model_block(data):
    """The per-model result block (usually 'vlm')."""
    if isinstance(data.get("vlm"), dict):
        return data["vlm"]
    return next((v for v in data.values()
                 if isinstance(v, dict) and "timeline" in v), None)


def parse_run_start(run_id, status):
    """Best-effort UTC start time for a run: the status timestamp, or else the
    timestamp embedded in the run id (YYYYMMDD_HHMMSS...)."""
    stamp = status.get("started_at") or status.get("submitted_at")
    if stamp:
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            pass
    match = re.match(r"(\d{8}_\d{6})", run_id)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def build_perf_df(disk_runs, cutoff):
    """Per-video performance rows (from the run result JSON) within the window,
    for the trend charts. One row per video, oldest first."""
    rows = []
    for run_id, run in disk_runs.items():
        start = parse_run_start(run_id, run.get("status", {}))
        if start is None or (cutoff is not None and start < cutoff):
            continue
        for video_name, data in run["videos"].items():
            block = get_model_block(data) or {}
            perf = block.get("performance") or {}
            frames = data.get("frames_processed") or len(block.get("timeline") or [])
            params = perf.get("model_params")
            tokens = (perf.get("total_prompt_tokens") or 0) + (perf.get("total_completion_tokens") or 0)
            compute = 2 * params * tokens / 1e12 if params and tokens else None
            cost = frames * GEMINI_COST_PER_FRAME_USD if frames else None
            rows.append({
                "start": start.replace(tzinfo=None),
                "Video": os.path.basename(data.get("video", "")) or video_name,
                "Total inference (sec)": perf.get("total_vlm_inference_sec"),
                "Avg": perf.get("avg_frame_latency_sec"),
                "Median": perf.get("median_frame_latency_sec"),
                "Min": perf.get("min_frame_latency_sec"),
                "Max": perf.get("max_frame_latency_sec"),
                "Compute (TERAFLOPS)": compute,
                "Gemini Cost (USD)": cost,
            })
    rows.sort(key=lambda r: r["start"])
    return pd.DataFrame(rows)


def render_trend_charts(perf_df, cutoff=None):
    if perf_df.empty:
        st.caption("No completed runs with performance data in this period.")
        return

    # Pin the x-axis to the selected window so the picker redraws the axis.
    if cutoff is not None:
        now = datetime.now(timezone.utc)
        x_scale = alt.Scale(domain=[cutoff.replace(tzinfo=None).isoformat(),
                                    now.replace(tzinfo=None).isoformat()])
    else:
        x_scale = alt.Scale()
    x_axis = alt.X("start:T", title="Start time", scale=x_scale)

    left, right = st.columns(2)

    with left:
        st.markdown("**Total Inference Time**")
        inference = perf_df.dropna(subset=["Total inference (sec)"])
        if inference.empty:
            st.caption("No data.")
        else:
            chart = alt.Chart(inference).mark_line(point=True, color=TABLEAU_BLUE).encode(
                x=x_axis,
                y=alt.Y("Total inference (sec):Q", title="Total inference (sec)"),
                tooltip=["Video:N", alt.Tooltip("start:T", title="Start"),
                         alt.Tooltip("Total inference (sec):Q", format=".2f")],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown("**Frame Latency**")
        latency = perf_df.melt(
            id_vars=["start", "Video"],
            value_vars=["Avg", "Median", "Min", "Max"],
            var_name="Metric", value_name="Latency (sec)",
        ).dropna(subset=["Latency (sec)"])
        if latency.empty:
            st.caption("No data.")
        else:
            chart = alt.Chart(latency).mark_line(point=True).encode(
                x=x_axis,
                y=alt.Y("Latency (sec):Q", title="Latency (sec)"),
                color=alt.Color("Metric:N", title=None,
                                scale=alt.Scale(domain=["Avg", "Median", "Min", "Max"],
                                                scheme=COLOR_SCHEME),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Video:N", "Metric:N", alt.Tooltip("start:T", title="Start"),
                         alt.Tooltip("Latency (sec):Q", format=".2f")],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)


def render_overview():
    st.header("Main Dashboard")
    st.caption(f"Polling {API_URL}/metrics every {REFRESH_SEC} sec")

    # Global date picker drives the range for the status counters, graphs, and history.
    picker_col, _ = st.columns([1, 3])
    period_label = picker_col.selectbox("Period", list(PERIODS.keys()), index=2, key="period")
    days = PERIODS[period_label]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days) if days else None

    metrics = fetch_json("/metrics")
    if metrics is None:
        st.error(f"Cannot reach the service at {API_URL}. Is it running?")
        return

    ollama = "reachable" if metrics.get("ollama_reachable") else "unreachable"
    st.subheader("Current Status")
    status_line = (
        f"**Model:** {metrics.get('model', '?')}  ·  "
        f"**Ollama:** {ollama}  ·  "
        f"**Uptime:** {metrics.get('uptime_sec', 0):.0f} sec"
    )
    if metrics.get("last_memory_mb") is not None:
        status_line += f"  ·  **Memory:** {metrics['last_memory_mb']:.0f} MB"
    st.markdown(status_line)

    # All runs within the selected window, counters are derived from this set.
    history = [r for r in (fetch_json("/runs") or {}).get("runs", [])
               if within_period(r.get("started_at") or r.get("submitted_at"), cutoff)]
    counts = {s: 0 for s in ("queued", "running", "succeeded", "failed", "interrupted")}
    for r in history:
        if r.get("status") in counts:
            counts[r["status"]] += 1

    cols = st.columns(6)
    cols[0].metric("Received", len(history))
    cols[1].metric("Queued", counts["queued"])
    cols[2].metric("Running", counts["running"])
    cols[3].metric("Succeeded", counts["succeeded"])
    cols[4].metric("Failed", counts["failed"])
    cols[5].metric("Interrupted", counts["interrupted"])

    perf_df = build_perf_df(load_disk_runs(), cutoff)
    if not perf_df.empty:
        costs = perf_df["Gemini Cost (USD)"].dropna()
        total_cost, n_cost = costs.sum(), costs.size
        total_compute = perf_df["Compute (TERAFLOPS)"].dropna().sum()
        hc = st.columns(3)
        hc[0].metric("Equivalent Gemini Cost (USD)", f"${total_cost:.2f}")
        hc[1].metric("Avg Cost / Video", f"${total_cost / n_cost:.4f}" if n_cost else "—")
        hc[2].metric("Total Compute (TERAFLOPS)", f"{total_compute:.1f}" if total_compute else "—")

    st.subheader("Trends")
    render_trend_charts(perf_df, cutoff)

    st.subheader("Run History")
    if not history:
        st.caption("No runs in this period.")
    else:
        # history sorted so newest is first.
        df = pd.DataFrame([{
            "Start Time": format_timestamp(r.get("started_at") or r.get("submitted_at")),
            "Run ID": r.get("run_id"),
            "Video": os.path.basename(r.get("video_path", "")),
            "Status": r.get("status"),
            "Processing Time (sec)": r.get("processing_sec"),
            "Frames Processed": r.get("frames_processed"),
            "Time per Frame (sec)": time_per_frame(r.get("processing_sec"), r.get("frames_processed")),
        } for r in history])
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Run ID": st.column_config.TextColumn(width="medium"),
                "Processing Time (sec)": st.column_config.NumberColumn(format="%.2f"),
                "Time per Frame (sec)": st.column_config.NumberColumn(
                    format="%.2f",
                    help="Processing time ÷ frames processed — the average cost per frame.",
                ),
            },
        )


def render_video(name, data):
    filename = os.path.basename(data.get("video", "")) or name
    st.subheader(f"Video: {filename}")
    st.markdown(f"**Path:** {data.get('video', '—')}")

    length = data.get("video_length_sec")
    fps = data.get("fps")
    video_info = [
        f"**Duration:** {length} sec" if length is not None else None,
        f"**Frames:** {data.get('total_frames', '—')}",
        f"**FPS:** {fps:.1f}" if fps else None,
        f"**Resolution:** {data.get('resolution', '—')}",
    ]
    st.markdown(" · ".join(p for p in video_info if p))

    block = get_model_block(data)
    if not block:
        st.info("No model output stored for this video.")
        return

    st.markdown(
        f"**Model:** {block.get('model', '—')}  ·  "
        f"**Diff threshold:** {block.get('diff_threshold', '—')}  ·  "
        f"**Frames processed:** {data.get('frames_processed', '—')}"
    )

    st.subheader("Output")

    intent = block.get("intent")
    if intent:
        st.markdown("#### Analysis")
        st.write(intent)

    timeline = block.get("timeline", [])
    if timeline:
        st.markdown("#### Timeline")
        timeline_df = pd.DataFrame([{
            "Frame": e.get("frame_number"),
            "Time (sec)": round(e["timestamp_sec"], 2) if e.get("timestamp_sec") is not None else None,
            "Latency (sec)": round(e["latency_sec"], 2) if e.get("latency_sec") is not None else None,
            "Prompt tokens": e.get("prompt_tokens"),
            "Completion tokens": e.get("completion_tokens"),
            "Description": e.get("frame_description", ""),
        } for e in timeline])
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)

    performance = block.get("performance")
    if performance:
        st.subheader("Performance")
        # vlm_model and frames_processed are already shown above, so omit them here.
        perf_items = [
            ("Total VLM inference (sec)", performance.get("total_vlm_inference_sec")),
            ("Analysis latency (sec)", performance.get("intent_latency_sec")
             or performance.get("summary_latency_sec")),
            ("Standalone pipeline (sec)", performance.get("standalone_pipeline_sec")),
            ("Avg frame latency (sec)", performance.get("avg_frame_latency_sec")),
            ("Median frame latency (sec)", performance.get("median_frame_latency_sec")),
            ("Min frame latency (sec)", performance.get("min_frame_latency_sec")),
            ("Max frame latency (sec)", performance.get("max_frame_latency_sec")),
            ("Prompt tokens", performance.get("total_prompt_tokens")),
            ("Completion tokens", performance.get("total_completion_tokens")),
            ("Completion tokens/sec", performance.get("avg_completion_tokens_per_sec")),
            ("Empty descriptions", performance.get("frames_with_empty_description")),
            ("Avg description length (chars)", performance.get("avg_description_length_chars")),
        ]
        cols = st.columns(4)
        for i, (label, value) in enumerate(perf_items):
            cols[i % 4].metric(label, value if value is not None else "—")


def render_detail(run_id, run, selected_video):
    info = run["status"]
    st.header(f"Run {run_id}")
    meta = [
        f"**Status:** {info.get('status', '—')}",
        f"**Submitted:** {format_timestamp(info.get('submitted_at'))}",
        f"**Finished:** {format_timestamp(info.get('finished_at'))}" if info.get("finished_at") else None,
        f"**Processing:** {info.get('processing_sec')} sec" if info.get("processing_sec") is not None else None,
    ]
    st.markdown(" · ".join(p for p in meta if p))

    if info.get("status") == "failed" and info.get("error"):
        st.error(f"{info.get('reason', 'error')}: {info['error']}")

    if not run["videos"]:
        st.info("No output stored for this run yet.")
        return
    render_video(selected_video, run["videos"][selected_video])


@st.fragment(run_every=REFRESH_SEC)
def live_overview():
    # Fix for 'ghost' widgets greyed out on screen.
    render_overview()


try:
    st.set_page_config(
        page_title="Screen Recording Description — Service Metrics",
        page_icon="📈",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set

st.title("Screen Recording Description — Service Metrics")

disk_runs = load_disk_runs()
options = [OVERVIEW_LABEL] + list(disk_runs.keys())
selected = st.sidebar.selectbox("Select Run", options)

run = None
selected_video = None
if selected != OVERVIEW_LABEL:
    run = disk_runs[selected]
    video_names = sorted(run["videos"].keys())
    selected_video = st.sidebar.selectbox(
        "Video", video_names,
        format_func=lambda s: os.path.basename(run["videos"][s].get("video", "")) or s,
    ) if video_names else None

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(disk_runs)} runs on disk")

if selected == OVERVIEW_LABEL:
    live_overview()
else:
    render_detail(selected, run, selected_video)
