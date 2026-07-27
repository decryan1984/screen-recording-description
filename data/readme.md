# Data

This project uses the publicly available **GUI-World** dataset (ONE-Lab), specifically the **`multi`** subdirectory of multi-application screen recordings.

The dataset files are **not committed** to this repository — download them separately and place them under `data/GUI-World/`.

## GUI-World `multi` videos

Download the videos from the `multi` subdirectory on Hugging Face:

https://huggingface.co/datasets/ONE-Lab/GUI-World/tree/main/multi

Place the video files in:

```
data/GUI-World/multi/
```

The pipeline resolves videos by ID from this directory (see `GUI_WORLD_VIDEO_DIR` in `src/screen_recording_description/config.py`).

## Annotations

Evaluation reads GUI-World annotations from both:

```
data/GUI-World/Annotation/train/multi.jsonl
data/GUI-World/Annotation/benchmark/multi.jsonl
```

These come from the `Annotation` directory of the same dataset:

https://huggingface.co/datasets/ONE-Lab/GUI-World/tree/main

## Expected layout

```
data/
└── GUI-World/
    ├── multi/                     # video files (from the multi subdirectory)
    └── Annotation/
        ├── train/multi.jsonl
        └── benchmark/multi.jsonl
```
