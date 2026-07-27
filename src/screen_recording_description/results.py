"""Streamlit dashboard for viewing description and evaluation results.
    Launch with: streamlit run src/screen_recording_description/results.py
"""

import glob
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from config import DEFAULT_FRAME_DIFF_THRESHOLD, GEMINI_MAX_TOKENS, MAX_FRAME_WIDTH, MAX_TOKENS, RESULTS_DIR

# Score-table display formatting
PINNED_COLUMNS = {"Variant": st.column_config.Column(pinned=True)}
# Numeric columns default to 2 decimal places
# Text similarity scores are set to 4 decimal places
TEXT_SIMILARITY_COLUMNS = ["BERTScore F1", "ROUGE-1 F1",
                           "Avg BERTScore F1", "Avg ROUGE-1 F1"]
TEXT_COLUMNS = {"Variant", "Model", "Resolution", "Frames", "Avg Frames"}
COLUMN_FORMAT_OVERRIDES = {
    **{c: "{:.4f}" for c in TEXT_SIMILARITY_COLUMNS},
    "Diff Threshold": "{:g}",
    "Max Tokens": "{:.0f}",
}
AVG_COLUMN_ALIASES = {
    "Frames": "Avg Frames",
    "Overall Time (sec)": "Avg Overall Time (sec)",
    "Time per Frame (sec)": "Avg Time per Frame (sec)",
    "Accuracy": "Avg Accuracy", "Coverage": "Avg Coverage",
    "Non-Repetition": "Avg Non-Repetition", "User Intent": "Avg User Intent",
    "BERTScore F1": "Avg BERTScore F1", "ROUGE-1 F1": "Avg ROUGE-1 F1",
}
PARAM_CHART_MIN_HEIGHT = 400
COLOR_SCHEME = "tableau10"

# Parameters whose individual effect is viewable in the Parameter Effect graphs.
PARAM_EFFECT_PARAMS = ["Diff Threshold", "Max Tokens", "Resolution"]
# Each parameter's effect is measured with the others held at these defaults.
PARAM_EFFECT_BASELINE = {
    "Diff Threshold": DEFAULT_FRAME_DIFF_THRESHOLD,
    "Max Tokens": MAX_TOKENS,
    "Resolution": "Full" if MAX_FRAME_WIDTH == 0 else "Reduced",
}

RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")
BASELINE_VARIANT_KEYS = {"vlm", "gemini"}
SCORE_COLUMN_ALIASES = {
    "intent": "User Intent",
    "accuracy": "Accuracy",
    "coverage": "Coverage",
    "non_repetition": "Non-Repetition",
}

def is_variant_result(val):
    return isinstance(val, dict) and ("timeline" in val or "model" in val)


def get_variant_label(key, block=None, default_diff=None):
    """UI friendly 'Variant' label for a variant result, e.g. 'qwen3.5:4b diff=0.05'."""
    name = block.get("model", key) if block else key
    _, sep, suffix = key.partition("_")
    if not sep:
        # Baseline variants run at the default diff threshold.
        if key in BASELINE_VARIANT_KEYS and default_diff is not None:
            return f"{name} diff={float(default_diff):g}"
        return name
    if suffix.startswith("diff") and suffix[4:5].isdigit():
        return f"{name} diff={suffix[4:]}"
    return f"{name} {suffix}"


def get_nested_value(data, *keys):
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def get_frame_inference_time(block):
    recorded = get_nested_value(block, "performance", "total_vlm_inference_sec")
    if recorded is not None:
        return recorded
    latencies = [f["latency_sec"] for f in block.get("timeline", [])
                 if isinstance(f, dict) and f.get("latency_sec") is not None]
    return sum(latencies) if latencies else None


def get_formatted_number(value, decimals=4):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def get_bar_height(n_groups, n_series=1, px_per_bar=18):
    return int(max(120, n_groups * (n_series * px_per_bar + 10) + 40))


def get_horizontal_bar_chart(df, value_cols, order, height, x_title="", color_field=None, color_domain=None):
    """Horizontal bar chart indexed by Variant."""
    id_cols = ["Variant"] + ([color_field] if color_field and color_field != "Variant" else [])
    long_df = (df[id_cols + value_cols]
               .melt(id_vars=id_cols, var_name="Metric", value_name="Value")
               .dropna(subset=["Value"]))
    long_df["Metric"] = long_df["Metric"].str.replace("Avg ", "", regex=False)
    tooltip = ["Variant:N"]
    if color_field and color_field != "Variant":
        tooltip.append(f"{color_field}:N")
    tooltip += ["Metric:N", alt.Tooltip("Value:Q", format=".2f")]
    encoding = {
        "y": alt.Y("Variant:N", sort=list(order), title=None, axis=alt.Axis(labelLimit=0)),
        "x": alt.X("Value:Q", title=x_title or None),
        "tooltip": tooltip,
    }
    if len(value_cols) > 1:
        encoding["yOffset"] = alt.YOffset("Metric:N")
        encoding["color"] = alt.Color("Metric:N", title=None,
                                      scale=alt.Scale(scheme=COLOR_SCHEME),
                                      legend=alt.Legend(orient="bottom"))
    elif color_field:
        encoding["color"] = alt.Color(f"{color_field}:N", title=color_field,
                                      scale=alt.Scale(domain=color_domain, scheme=COLOR_SCHEME)
                                      if color_domain else alt.Scale(scheme=COLOR_SCHEME),
                                      legend=alt.Legend(orient="bottom"))
    return alt.Chart(long_df).mark_bar().encode(**encoding).properties(height=height)


def get_all_runs():
    runs = {}
    if not os.path.isdir(RESULTS_DIR):
        return runs

    for name in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(run_dir) or not RUN_DIR_PATTERN.match(name):
            continue
        try:
            label = datetime.strptime(name, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            label = name

        videos = {}
        config = {}
        for json_file in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
            filename = os.path.basename(json_file)
            if filename == "config.json":
                with open(json_file) as f:
                    config = json.load(f)
                continue
            with open(json_file) as f:
                data = json.load(f)
            if not any(is_variant_result(v) for v in data.values()):
                continue
            videos[os.path.splitext(filename)[0]] = data

        if videos:
            runs[name] = {"label": label, "videos": videos, "config": config}

    return dict(sorted(runs.items()))


def get_all_variants(runs):
    """All variant keys across every run, in the order they first appear."""
    keys = []
    for run_info in runs.values():
        for data in run_info["videos"].values():
            for key, value in data.items():
                if is_variant_result(value) and key not in keys:
                    keys.append(key)
    return keys


def get_sorted_variants(keys, default_diff=None):
    """Order variant keys by config (diff threshold etc.), pairing variants with the same config."""
    def sort_key(key):
        if key in BASELINE_VARIANT_KEYS:
            if default_diff is not None:
                return (1, -float(default_diff), "", key)
            return (0, 0.0, "", key)
        model, sep, suffix = key.partition("_")
        if suffix.startswith("diff") and suffix[4:5].isdigit():
            try:
                return (1, -float(suffix[4:]), "", model)
            except ValueError:
                pass
        return (2, 0.0, suffix, model)
    return sorted(keys, key=sort_key)


def get_variant_fields(block, config=None, key=None):
    config = config or {}
    max_width = block.get("max_frame_width", config.get("max_frame_width"))
    # Gemini generates with GEMINI_MAX_TOKENS; older blocks don't store it, so
    # avoid inheriting the VLM's run-level max_tokens as a fallback.
    is_gemini = "gemini" in str(block.get("model", "")).lower()
    max_tokens_fallback = GEMINI_MAX_TOKENS if is_gemini else config.get("max_tokens")
    return {
        "Variant": get_variant_label(key, block, config.get("default_diff_threshold")) if key else block.get("model", ""),
        "Model": block.get("model", ""),
        "Diff Threshold": block.get("diff_threshold", config.get("default_diff_threshold")),
        "Max Tokens": block.get("max_tokens", max_tokens_fallback),
        "Resolution": ("Full" if max_width == 0 else "Reduced") if max_width is not None else None,
    }


def get_param_effect_table(config_df, baseline, param, metric):
    others = [col for col in PARAM_EFFECT_PARAMS if col != param]
    mask = pd.Series(True, index=config_df.index)
    for col in others:
        if baseline.get(col) is not None:
            mask &= (config_df[col] == baseline[col])
    subset = config_df[mask][[param, "Model", metric]].dropna()
    if subset[param].nunique() < 2:
        return None
    return subset.groupby([param, "Model"], as_index=False)[metric].mean()


def get_scores_table(data, variant_keys, config=None):
    rows = []
    keys = get_sorted_variants(
        [k for k in variant_keys if k in data and is_variant_result(data[k])],
        (config or {}).get("default_diff_threshold"))
    for key in keys:
        block = data[key]
        cols = get_variant_fields(block, config, key)
        frames = len(block.get("timeline", []))
        total_frames = data.get("total_frames")
        # Frames shown as "processed (percent of total)", e.g. "12 (5.00%)".
        frames_pct = round(frames / total_frames * 100, 2) if frames and total_frames else None
        frames_display = f"{frames} ({frames_pct:.2f}%)" if frames_pct is not None else str(frames)
        vlm_time = get_frame_inference_time(block)
        summary_time = get_nested_value(block, "performance", "summary_latency_sec")
        overall_time = (vlm_time + (summary_time or 0)) if vlm_time is not None else None
        per_frame_time = (vlm_time / frames) if vlm_time is not None and frames else None
        overall_score = get_nested_value(block, "evaluation", "composite_score")
        efficiency = round(overall_score / overall_time * 60, 2) if overall_score is not None and overall_time else None
        rows.append({
            "Variant": cols["Variant"],
            "Model": cols["Model"],
            "Diff Threshold": cols["Diff Threshold"],
            "Max Tokens": cols["Max Tokens"],
            "Resolution": cols["Resolution"],
            "Frames": frames_display,
            "Overall Time (sec)": round(overall_time, 2) if overall_time is not None else None,
            "Time per Frame (sec)": round(per_frame_time, 2) if per_frame_time is not None else None,
            "Accuracy": get_nested_value(block, "evaluation", "scores", "accuracy", "score"),
            "Coverage": get_nested_value(block, "evaluation", "scores", "coverage", "score"),
            "Non-Repetition": get_nested_value(block, "evaluation", "scores", "non_repetition", "score"),
            "User Intent": get_nested_value(block, "evaluation", "scores", "intent", "score"),
            "Overall Score": overall_score,
            "Efficiency (pts/min)": efficiency,
            "BERTScore F1": get_nested_value(block, "bertscore", "f1"),
            "ROUGE-1 F1": get_nested_value(block, "rouge", "rouge1", "f1"),
        })
    return pd.DataFrame(rows)


def get_shaded_variant_groups(display_df, full_df):
    """Styler that shades alternating variant groups so variants with the same params read more easily as one block."""
    def suffix(variant, model):
        variant, model = str(variant), str(model)
        return variant[len(model):].strip() if variant.startswith(model) else variant

    shades, group, prev = [], -1, None
    for variant, model in zip(full_df["Variant"], full_df["Model"]):
        config_suffix = suffix(variant, model)
        if not config_suffix:  # plain baseline model — leave clear
            shades.append(False)
            continue
        if config_suffix != prev:
            group += 1
            prev = config_suffix
        shades.append(group % 2 == 0)

    tint = "background-color: rgba(128, 128, 128, 0.14)"
    shaded_by_index = {idx: shaded for idx, shaded in zip(display_df.index, shades)}

    def style_row(row):
        return [tint if shaded_by_index.get(row.name) else "" for _ in row]

    column_formats = {col: COLUMN_FORMAT_OVERRIDES.get(col, "{:.2f}")
                      for col in display_df.columns if col not in TEXT_COLUMNS}
    return display_df.style.apply(style_row, axis=1).format(column_formats, na_rep="—")


def render_run_config(config, variant_df=None):
    """Show run-level parameters and prompts from config.json.

    ``config`` only records the baseline defaults, so when a variant table is
    supplied the actual set of Models and Max Tokens used in the run is shown.
    """
    if not config:
        return
    st.subheader("Run Configuration")

    # Distinct values actually used across the run's variants (baseline config
    # only stores a single default for each).
    used = {}
    if variant_df is not None and not variant_df.empty:
        for col in ("Model", "Max Tokens"):
            if col in variant_df.columns:
                used[col] = sorted(variant_df[col].dropna().unique().tolist())

    fields = [
        ("model", "Model"),
        ("temperature", "Temperature"),
        ("max_tokens", "Max Tokens"),
        ("max_frame_width", "Max Frame Width"),
        ("enable_grayscale_conversion", "Grayscale Differencing"),
        ("max_fps", "Max FPS"),
        ("default_diff_threshold", "Default Diff Threshold"),
        ("diff_thresholds", "Diff Thresholds"),
        ("eval_model", "Eval Model"),
    ]
    overrides = {"model": ("Models", used.get("Model")),
                 "max_tokens": ("Max Tokens", used.get("Max Tokens"))}
    def fmt(v):
        return f"{int(v)}" if isinstance(v, float) and v.is_integer() else str(v)

    parts = []
    for key, label in fields:
        override_label, values = overrides.get(key, (None, None))
        if values:
            parts.append(f"**{override_label}:** {', '.join(fmt(v) for v in values)}")
        elif config.get(key) is not None:
            parts.append(f"**{label}:** {config[key]}")
    if parts:
        st.markdown(" · ".join(parts))

    if config.get("frame_description_prompt"):
        with st.expander("Frame Description Prompt"):
            st.code(config["frame_description_prompt"], language="text")
    if config.get("summary_prompt"):
        with st.expander("Summary Prompt"):
            st.code(config["summary_prompt"], language="text")
    for name, text in (config.get("eval_prompts") or {}).items():
        with st.expander(f"Eval Prompt — {name}"):
            st.code(text, language="text")


def get_timeline_table(block):
    timeline = block.get("timeline", [])
    if not timeline:
        return pd.DataFrame()
    rows = []
    for entry in timeline:
        timestamp = entry.get("timestamp_sec")
        latency = entry.get("latency_sec")
        rows.append({
            "Frame": entry.get("frame_number"),
            "Time (sec)": round(timestamp, 2) if timestamp is not None else None,
            "Latency (sec)": round(latency, 2) if latency is not None else None,
            "Description": entry.get("frame_description", ""),
        })
    return pd.DataFrame(rows)


def get_paired_variant_order(df):
    """Order variants so equivalent configs across models are adjacent for comparison."""
    def suffix(variant, model):
        if isinstance(variant, str) and isinstance(model, str) and variant.startswith(model):
            return variant[len(model):].strip()
        return str(variant)
    suffixes = [suffix(v, m) for v, m in zip(df["Variant"], df["Model"])]
    suffix_rank = {s: i for i, s in enumerate(dict.fromkeys(suffixes))}
    ordered = df.assign(suffix=suffixes)
    ordered = ordered.assign(rank_order=ordered["suffix"].map(suffix_rank))
    return ordered.sort_values(["rank_order", "Model"], kind="stable")["Variant"].tolist()


def render_charts_and_effects(df):
    if df.empty or "Variant" not in df.columns:
        return
    order = get_paired_variant_order(df)
    model_domain = sorted(df["Model"].dropna().unique().tolist())
    variant_count = len(df)
    # Top pair represents the 4-series Individual Scores chart, so use thinner bars.
    top_height = get_bar_height(variant_count, 4, px_per_bar=12)
    bottom_height = get_bar_height(variant_count, 2)

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    with top_left:
        if "Overall Score" in df.columns and df["Overall Score"].notna().any():
            st.subheader("Overall Scores")
            st.altair_chart(get_horizontal_bar_chart(df, ["Overall Score"], order, top_height, "Score",
                                  color_field="Model", color_domain=model_domain),
                            width='stretch')

    with top_right:
        score_cols = [c for c in ["Accuracy", "Coverage", "Non-Repetition", "User Intent"]
                      if c in df.columns]
        if score_cols and df[score_cols].notna().any().any():
            st.subheader("Individual Scores")
            st.altair_chart(get_horizontal_bar_chart(df, score_cols, order, top_height, "Score"),
                            width='stretch')

    with bottom_left:
        if "Efficiency (pts/min)" in df.columns and df["Efficiency (pts/min)"].notna().any():
            st.subheader("Efficiency Scores")
            st.altair_chart(get_horizontal_bar_chart(df, ["Efficiency (pts/min)"], order, bottom_height, "pts/min",
                                  color_field="Model", color_domain=model_domain),
                            width='stretch')

    with bottom_right:
        similarity_cols = [c for c in ["BERTScore F1", "ROUGE-1 F1"] if c in df.columns]
        if similarity_cols and df[similarity_cols].notna().any().any():
            st.subheader("Text Similarity Scores")
            st.altair_chart(get_horizontal_bar_chart(df, similarity_cols, order, bottom_height, "Similarity"),
                            width='stretch')

    # Parameter-effect isolation
    metric_options = {
        "Overall Score": "Overall Score",
        "Efficiency (pts/min)": "Efficiency (pts/min)",
        "Time per Frame (sec)": "Time per Frame (sec)",
        "Accuracy": "Accuracy",
        "Coverage": "Coverage",
        "Non-Repetition": "Non-Repetition",
        "User Intent": "User Intent",
        "BERTScore F1": "BERTScore F1",
        "ROUGE-1 F1": "ROUGE-1 F1",
    }
    metric_options = {k: v for k, v in metric_options.items()
                      if v in df.columns and df[v].notna().any()}
    baseline = PARAM_EFFECT_BASELINE
    probe_metric = next(iter(metric_options.values()), None)
    varied_params = ([param for param in PARAM_EFFECT_PARAMS
                      if probe_metric and get_param_effect_table(df, baseline, param, probe_metric) is not None]
                     if probe_metric else [])
    if varied_params and metric_options:
        st.subheader("Parameter Effect")
        baseline_desc = " · ".join(f"{param}={baseline[param]}" for param in PARAM_EFFECT_PARAMS if baseline.get(param) is not None)
        st.caption(f"The effect of each parameter on the selected metric, split by model, with the "
                   f"other parameters held at baseline ({baseline_desc}).")
        metric_label = st.selectbox("Metric", list(metric_options.keys()), key="param_effect_metric")
        metric_col = metric_options[metric_label]
        cols = st.columns(min(len(varied_params), 2))
        for i, param in enumerate(varied_params):
            effect = get_param_effect_table(df, baseline, param, metric_col)
            if effect is None:
                continue
            with cols[i % len(cols)]:
                st.markdown(f"**{param}**")
                effect = effect.copy()
                effect[param] = effect[param].astype(str)
                param_order = list(dict.fromkeys(effect[param].tolist()))[::-1]
                chart_height = max(get_bar_height(effect[param].nunique(), effect["Model"].nunique()),
                                   PARAM_CHART_MIN_HEIGHT)
                chart = alt.Chart(effect).mark_bar().encode(
                    y=alt.Y(field=param, type="nominal", sort=param_order,
                            title=None, axis=alt.Axis(labelLimit=0)),
                    yOffset=alt.YOffset("Model:N"),
                    x=alt.X(field=metric_col, type="quantitative", title=metric_label),
                    color=alt.Color("Model:N", title="Model",
                                    scale=alt.Scale(domain=model_domain, scheme=COLOR_SCHEME),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip(field=param, type="nominal"),
                             alt.Tooltip("Model:N"),
                             alt.Tooltip(field=metric_col, type="quantitative", format=".2f")],
                ).properties(height=chart_height)
                st.altair_chart(chart, width='stretch')


st.set_page_config(
    page_title="Screen Recording Description — Results",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Screen Recording Description — Results")

runs = get_all_runs()
if not runs:
    st.warning(f"No result files found in `{RESULTS_DIR}`")
    st.stop()

variant_keys = get_all_variants(runs)

# Sidebar: run & video selection
run_ids = list(runs.keys())
run_labels = [f"{runs[rid]['label']}  ({len(runs[rid]['videos'])} videos)" for rid in run_ids]

selected_idx = st.sidebar.selectbox(
    "Select Run",
    range(len(run_ids)),
    index=len(run_ids) - 1,
    format_func=lambda i: run_labels[i],
)
selected_run_id = run_ids[selected_idx]
run = runs[selected_run_id]

video_names = sorted(run["videos"].keys(), key=lambda x: (int(x) if x.isdigit() else float("inf"), x))

# View options: aggregate first (if multiple videos), then all individual videos.
AGGREGATE_LABEL = "All Videos (Aggregated)"
view_options = []
if len(video_names) > 1:
    view_options.append(AGGREGATE_LABEL)
view_options.extend(video_names)

selected_view = st.sidebar.selectbox("View", view_options)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(runs)} runs · {sum(len(r['videos']) for r in runs.values())} total video results")

if selected_view == AGGREGATE_LABEL:
    # Aggregate view: compare configurations across all videos.
    st.header(f"Run: {run['label']} — {len(video_names)} videos")

    # Run-level video metadata (shared across variants), averaged over the videos.
    videos_meta = list(run["videos"].values())
    total_frame_vals = [v.get("total_frames") for v in videos_meta if v.get("total_frames")]
    fps_vals = [v.get("fps") for v in videos_meta if v.get("fps")]
    duration_vals = [v["total_frames"] / v["fps"] for v in videos_meta
                     if v.get("total_frames") and v.get("fps")]
    reduced_resolutions = {v.get("reduced_resolution") for v in videos_meta if v.get("reduced_resolution")}

    avg_frame_count = round(sum(total_frame_vals) / len(total_frame_vals)) if total_frame_vals else None
    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else None
    avg_duration = sum(duration_vals) / len(duration_vals) if duration_vals else None
    reduced_resolution = (next(iter(reduced_resolutions)) if len(reduced_resolutions) == 1 else "varies") \
        if reduced_resolutions else None

    meta_parts = [
        f"**Reduced Resolution:** {reduced_resolution}" if reduced_resolution else None,
        f"**Avg Frame Count:** {avg_frame_count}" if avg_frame_count is not None else None,
        f"**Avg FPS:** {avg_fps:.1f}" if avg_fps else None,
        f"**Avg Duration:** {int(avg_duration // 60)}m {int(avg_duration % 60):02d}s" if avg_duration else None,
    ]
    st.markdown(" · ".join(p for p in meta_parts if p))

    run_variant_keys = set()
    for video_data in run["videos"].values():
        for key, value in video_data.items():
            if is_variant_result(value):
                run_variant_keys.add(key)
    run_variant_keys = get_sorted_variants(
        run_variant_keys, (run.get("config") or {}).get("default_diff_threshold"))

    config_rows = []
    for key in run_variant_keys:
        intent_scores, accuracy_scores, coverage_scores, non_repetition_scores, overall_scores = [], [], [], [], []
        frame_counts, overall_times, per_frame_times = [], [], []
        total_frame_counts = []
        bert_scores, rouge1_scores = [], []

        for video_data in run["videos"].values():
            block = video_data.get(key)
            if not block or not is_variant_result(block):
                continue

            intent = get_nested_value(block, "evaluation", "scores", "intent", "score")
            accuracy = get_nested_value(block, "evaluation", "scores", "accuracy", "score")
            coverage = get_nested_value(block, "evaluation", "scores", "coverage", "score")
            non_repetition = get_nested_value(block, "evaluation", "scores", "non_repetition", "score")
            overall_score = get_nested_value(block, "evaluation", "composite_score")
            if intent is not None: intent_scores.append(intent)
            if accuracy is not None: accuracy_scores.append(accuracy)
            if coverage is not None: coverage_scores.append(coverage)
            if non_repetition is not None: non_repetition_scores.append(non_repetition)
            if overall_score is not None: overall_scores.append(overall_score)

            timeline = block.get("timeline", [])
            frame_counts.append(len(timeline))
            video_total_frames = video_data.get("total_frames")
            if video_total_frames:
                total_frame_counts.append(video_total_frames)

            # Overall time = frame VLM + summary, excluding the judge-eval step.
            vlm_time = get_frame_inference_time(block)
            summary_time = get_nested_value(block, "performance", "summary_latency_sec")
            if vlm_time is not None:
                overall_times.append(vlm_time + (summary_time or 0))
                if timeline: per_frame_times.append(vlm_time / len(timeline))

            bert = get_nested_value(block, "bertscore", "f1")
            if bert is not None: bert_scores.append(bert)
            rouge1 = get_nested_value(block, "rouge", "rouge1", "f1")
            if rouge1 is not None: rouge1_scores.append(rouge1)

        def average(values, decimals=2):
            return round(sum(values) / len(values), decimals) if values else None

        avg_time = average(overall_times)
        avg_score = average(overall_scores)
        # Efficiency: score/time in pts/min.
        efficiency = (round(avg_score / avg_time * 60, 2)
                      if avg_score is not None and avg_time else None)

        avg_frames = average(frame_counts)
        avg_total_frames = average(total_frame_counts)
        # Avg Frames shown as "processed (percent of total)", e.g. "12 (5.00%)".
        frames_pct = round(avg_frames / avg_total_frames * 100, 2) if avg_frames and avg_total_frames else None
        frames_display = f"{avg_frames:.0f} ({frames_pct:.2f}%)" if avg_frames is not None and frames_pct is not None else "—"

        sample_block = next((video_data.get(key) for video_data in run["videos"].values()
                             if video_data.get(key) and is_variant_result(video_data.get(key))), None)
        cols = get_variant_fields(sample_block or {}, run.get("config"), key)
        config_rows.append({
            "Variant": cols["Variant"],
            "Model": cols["Model"],
            "Diff Threshold": cols["Diff Threshold"],
            "Max Tokens": cols["Max Tokens"],
            "Resolution": cols["Resolution"],
            "Frames": frames_display,
            "Overall Time (sec)": avg_time,
            "Time per Frame (sec)": average(per_frame_times),
            "Accuracy": average(accuracy_scores),
            "Coverage": average(coverage_scores),
            "Non-Repetition": average(non_repetition_scores),
            "User Intent": average(intent_scores),
            "Overall Score": avg_score,
            "Efficiency (pts/min)": efficiency,
            "BERTScore F1": average(bert_scores, 4),
            "ROUGE-1 F1": average(rouge1_scores, 4),
        })

    if config_rows:
        config_df = pd.DataFrame(config_rows)

        st.subheader("Scores")
        st.caption("**Diff Threshold** — the pixel-difference level used to detect significant changes "
                   "between consecutive screenshots, lower values means more frames are processed.  \n"
                   "**Accuracy**, **Coverage**, **Non-Repetition** and **User Intent** are each "
                   "scored from 1 to 5, **Overall Score** is out of 20.  \n"
                   "**BERTScore** and **ROUGE** are text similarity scores from 0 to 1.  \n"
                   "**Note** — Gemini requires approximately 4 times as many tokens "
                   "in order to generate similar output to the VLMs.")
        agg_display = config_df.rename(columns=AVG_COLUMN_ALIASES)
        st.dataframe(get_shaded_variant_groups(agg_display, config_df),
                     width='content', hide_index=True,
                     column_config=PINNED_COLUMNS)

        render_charts_and_effects(config_df)

    render_run_config(run.get("config"), config_df if config_rows else None)

else:
    # Individual video detail view.
    selected_video = selected_view
    data = run["videos"][selected_video]

    st.header(f"Video {selected_video}")

    total_frames = data.get("total_frames")
    fps = data.get("fps")
    duration = round(total_frames / fps, 1) if total_frames and fps else None
    meta_parts = [
        f"**Original Resolution:** {data.get('resolution', '—')}",
        f"**Reduced Resolution:** {data.get('reduced_resolution')}" if data.get("reduced_resolution") else None,
        f"**Total frames:** {total_frames or '—'}",
        f"**FPS:** {fps:.1f}" if fps else None,
        f"**Duration:** {int(duration // 60)}m {int(duration % 60):02d}s" if duration else None,
        f"**Frames processed:** {data.get('frames_processed')}" if data.get("frames_processed") is not None else None,
    ]
    st.markdown(" · ".join(p for p in meta_parts if p))

    scores_df = get_scores_table(data, variant_keys, run.get("config"))
    if not scores_df.empty:
        st.subheader("Scores")
        st.caption("**Diff Threshold** — the pixel-difference level used to detect significant changes "
                   "between consecutive screenshots; lower values process more frames (more sensitive to change).  \n"
                   "**Accuracy**, **Coverage**, **Non-Repetition** and **User Intent** are each "
                   "scored from 1 to 5, **Overall Score** is out of 20.  \n"
                   "**BERTScore** and **ROUGE** are text similarity scores from 0 to 1.  \n"
                   "**Note** — Gemini requires approximately 4 times as many tokens "
                   "in order to generate similar output to the VLMs.")
        st.dataframe(get_shaded_variant_groups(scores_df, scores_df),
                     width='content', hide_index=True,
                     column_config=PINNED_COLUMNS)

        render_charts_and_effects(scores_df)

    default_diff = (run.get("config") or {}).get("default_diff_threshold")
    keys = get_sorted_variants(
        [k for k in variant_keys if k in data and is_variant_result(data[k])],
        default_diff)

    st.subheader("Descriptions")
    for key in keys:
        block = data[key]
        timeline_df = get_timeline_table(block)
        label = get_variant_label(key, block, default_diff)
        with st.expander(f"{label} — {len(timeline_df)} frames" if not timeline_df.empty else label):
            intent = block.get("intent", "")
            if intent:
                st.markdown(f"**Inferred Intent:** {intent}")
            if not timeline_df.empty:
                st.dataframe(timeline_df, hide_index=True)

    has_reasoning = any(get_nested_value(data[key], "evaluation", "scores") for key in keys)
    if has_reasoning:
        st.subheader("Score Reasoning")
        for key in keys:
            block = data[key]
            scores = get_nested_value(block, "evaluation", "scores")
            if not scores:
                continue
            with st.expander(get_variant_label(key, block, default_diff)):
                for criterion, info in scores.items():
                    label = SCORE_COLUMN_ALIASES.get(criterion, criterion)
                    st.markdown(f"**{label} ({get_formatted_number(info.get('score'), 2)}):** {info.get('reasoning', '')}")

    render_run_config(run.get("config"), scores_df)
