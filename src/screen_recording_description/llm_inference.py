"""LLM-as-judge, BERTScore and ROUGE evaluation: scoring summaries against reference descriptions."""

import json
import os
import re
import time

import requests
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from .config import (
    OLLAMA_BASE_URL,
    EVAL_MODEL_NAME,
    GUI_WORLD_ANNOTATIONS,
    EVAL_PROMPT_INTENT,
    EVAL_PROMPT_ACCURACY,
    EVAL_PROMPT_COVERAGE,
    EVAL_PROMPT_NON_REPETITION,
)

def _get_prompt_response(text_prompt, model_name, max_tokens=1024):
    """Send a text-only prompt to Ollama and return the response."""
    payload = {
        "model": model_name,
        "prompt": text_prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["response"]


def _get_text(value):
    """Coerce an annotation field to a string.

    GUI-World stores some description fields as a list of strings; join those into a
    single space-separated string so downstream text scorers receive a plain ``str``.
    """
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return value or ""


def get_annotations(video_path):
    """Look up GUI-World annotations for a video by its path.

    Searches both train and benchmark annotation files.
    Returns a dict with full annotation metadata, or None if not found.
    """
    # Normalise to the relative path format used in the JSONL (e.g. "multi/457.mov")
    video_basename = os.path.basename(video_path)
    candidates = [
        f"multi/{video_basename}",
        video_basename,
        video_path,
    ]

    for annot_path in GUI_WORLD_ANNOTATIONS:
        if not os.path.exists(annot_path):
            continue
        with open(annot_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("video_path") in candidates:
                    keyframes = entry.get("keyframes", [])

                    # Extract QA fields
                    static_qa = entry.get("static QA", {})
                    mcqa = entry.get("MCQA", {})
                    sequential_qa = entry.get("Sequential-QA", {})
                    prediction = entry.get("Prediction", {})
                    reasoning = entry.get("Reasoning", {})

                    # App list
                    app_raw = entry.get("app", [])
                    apps = []
                    for a in app_raw:
                        for p in a.replace("\uff0c", ",").split(","):
                            p = p.strip()
                            if p:
                                apps.append(p)

                    return {
                        "caption": entry.get("Caption", ""),
                        "goal": entry.get("goal", ""),
                        "keyframes": keyframes,
                        "apps": apps,
                        # Some GUI-World entries store descriptions as a list of
                        # strings rather than a single string; normalise to text
                        # so downstream scorers (BERTScore, ROUGE) get str inputs.
                        "description1": _get_text(entry.get("Description1", "")),
                        "description2": _get_text(entry.get("Description2", "")),
                        # QA fields
                        "static_question": static_qa.get("Question", ""),
                        "static_answer": static_qa.get("Answer", ""),
                        "mcqa_question": mcqa.get("Question", ""),
                        "mcqa_options": mcqa.get("Options", []),
                        "mcqa_answer": mcqa.get("Correct Answer", ""),
                        "sequential_question": sequential_qa.get("Question", ""),
                        "sequential_answer": sequential_qa.get("Answer", ""),
                        "prediction_question": prediction.get("Question", ""),
                        "prediction_answer": prediction.get("Answer", ""),
                        "reasoning_question": reasoning.get("Question", ""),
                        "reasoning_answer": reasoning.get("Correct Answer", ""),
                    }

    return None


def _get_parsed_eval_response(response_text):
    """Parse JSON from an eval model response, handling markdown fences and truncation."""
    text = response_text.strip()
    # Strip markdown code if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex fallback for truncated or malformed JSON
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)

    if score_match:
        score = int(score_match.group(1))
        reasoning = reasoning_match.group(1) if reasoning_match else "(truncated)"
        return {"score": score, "reasoning": reasoning}

    return {"score": 0, "reasoning": f"Failed to parse eval response: {text[:200]}"}


def _get_numbered_keyframes(keyframes):
    """Format keyframes as a 1..N numbered checklist for the coverage prompt."""
    lines = []
    for i, kf in enumerate(keyframes, 1):
        label = kf.get("sub_goal", "") or f"frame {kf.get('frame', '?')}"
        parts = [label]
        if kf.get("mouse") and kf["mouse"] != "none":
            parts.append(f"mouse={kf['mouse']}")
        if kf.get("keyboard") and kf["keyboard"] != "none":
            parts.append(f"keyboard={kf['keyboard']}")
        if kf.get("keyboardOperation"):
            parts.append(f"typed=\"{kf['keyboardOperation']}\"")
        lines.append(f"{i}. " + " | ".join(parts))
    return "\n".join(lines)


def _get_parsed_coverage_response(response_text):
    """Parse the coverage checklist JSON: reasoning + covered action numbers + covered apps."""
    text = response_text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Regex fallback for truncated/malformed JSON
    result = {}
    reason = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if reason:
        result["reasoning"] = reason.group(1)
    actions = re.search(r'"covered_actions"\s*:\s*\[([^\]]*)\]', text)
    if actions:
        result["covered_actions"] = [int(n) for n in re.findall(r"\d+", actions.group(1))]
    apps = re.search(r'"covered_apps"\s*:\s*\[([^\]]*)\]', text)
    if apps:
        result["covered_apps"] = re.findall(r'"([^"]*)"', apps.group(1))
    return result


def _get_coverage_evaluation(timeline_text, annotations, eval_model):
    """Coverage: fraction of reference keyframe actions + applications the timeline captures.

    Uses a single per-item checklist LLM call (the judge does the semantic matching), then
    computes coverage = covered / total. Maps the fraction to a 1-5 score (1 + 4*coverage)
    so it stays comparable with the other criteria in the composite.
    """
    keyframes = annotations.get("keyframes") or []
    apps = annotations.get("apps") or []
    total = len(keyframes) + len(apps)
    if total == 0:
        return {"score": None, "coverage": None, "covered": 0, "total": 0,
                "reasoning": "No reference keyframes or applications to score against."}

    prompt = EVAL_PROMPT_COVERAGE.format(
        keyframes=_get_numbered_keyframes(keyframes),
        app_list=", ".join(apps) if apps else "(none)",
        timeline=timeline_text,
    )
    parsed = _get_parsed_coverage_response(_get_prompt_response(prompt, model_name=eval_model))

    # Validate covered action indices against the 1..N range
    covered_actions = set()
    for x in parsed.get("covered_actions", []) or []:
        try:
            n = int(x)
        except (ValueError, TypeError):
            continue
        if 1 <= n <= len(keyframes):
            covered_actions.add(n)

    # Match covered app names (case-insensitive, allowing partial containment)
    apps_by_lower = {a.lower(): a for a in apps}
    covered_apps = set()
    for a in parsed.get("covered_apps", []) or []:
        if not isinstance(a, str):
            continue
        key = a.strip().lower()
        if not key:
            continue
        if key in apps_by_lower:
            covered_apps.add(apps_by_lower[key])
            continue
        for al, orig in apps_by_lower.items():
            if al and (al in key or key in al):
                covered_apps.add(orig)
                break

    covered = len(covered_actions) + len(covered_apps)
    coverage = covered / total
    reasoning = parsed.get("reasoning", "")
    return {
        "score": round(1 + 4 * coverage, 2),
        "coverage": round(coverage, 3),
        "covered": covered,
        "total": total,
        "covered_actions": sorted(covered_actions),
        "covered_apps": sorted(covered_apps),
        "reasoning": reasoning if isinstance(reasoning, str) else "",
    }

def _get_formatted_timeline(timeline):
    """Format VLM timeline entries for eval prompts."""
    return "\n".join(
        f"[Frame {e['frame_number']}] [{e['timestamp_sec']}s] {e['frame_description']}"
        for e in timeline
    )


def get_summary_evaluation(summary, intent, timeline, annotations, eval_model=EVAL_MODEL_NAME):
    """Run LLM-as-judge evaluation against reference annotations.

    Scores intent, accuracy and non-repetition on a 1–5 scale via the LLM judge, and
    coverage as an objective fraction mapped to 1–5. Returns a dict with
    per-criterion scores, reasoning, and the composite score.
    """
    timeline_text = _get_formatted_timeline(timeline)
    mcqa_options_text = " | ".join(annotations["mcqa_options"]) if annotations["mcqa_options"] else "(none)"

    # Build prompt kwargs per criterion
    criteria = [
        ("intent", EVAL_PROMPT_INTENT, {
            "goal": annotations["goal"],
            "caption": annotations["caption"],
            "reasoning_question": annotations["reasoning_question"],
            "reasoning_answer": annotations["reasoning_answer"],
            "prediction_question": annotations["prediction_question"],
            "prediction_answer": annotations["prediction_answer"],
            "intent": intent,
            "timeline": timeline_text,
        }),
        ("accuracy", EVAL_PROMPT_ACCURACY, {
            "caption": annotations["caption"],
            "keyframes": _get_numbered_keyframes(annotations.get("keyframes") or []) or "(none)",
            "static_question": annotations["static_question"],
            "static_answer": annotations["static_answer"],
            "sequential_question": annotations["sequential_question"],
            "sequential_answer": annotations["sequential_answer"],
            "mcqa_question": annotations["mcqa_question"],
            "mcqa_options": mcqa_options_text,
            "mcqa_answer": annotations["mcqa_answer"],
            "timeline": timeline_text,
        }),
        ("non_repetition", EVAL_PROMPT_NON_REPETITION, {
            "timeline": timeline_text,
        }),
    ]

    scores = {}
    total_latency = 0.0

    for criterion_name, prompt_template, kwargs in criteria:
        prompt = prompt_template.format(**kwargs)
        start = time.perf_counter()
        raw_response = _get_prompt_response(prompt, model_name=eval_model)
        latency = time.perf_counter() - start
        total_latency += latency

        parsed = _get_parsed_eval_response(raw_response)
        scores[criterion_name] = {
            "score": parsed.get("score", 0),
            "reasoning": parsed.get("reasoning", ""),
        }
        print(f"  {criterion_name}: {parsed.get('score', '?')}/5 — {parsed.get('reasoning', '')}")

    # Coverage: per-item checklist over reference keyframes + apps
    cov_start = time.perf_counter()
    scores["coverage"] = _get_coverage_evaluation(timeline_text, annotations, eval_model)
    total_latency += time.perf_counter() - cov_start
    cov = scores["coverage"]
    print(f"  coverage: {cov['covered']}/{cov['total']} "
          f"= {cov['coverage'] if cov['coverage'] is not None else 'n/a'} -> score {cov['score']}")

    # Keep a stable display order regardless of computation order
    scores = {k: scores[k] for k in ("intent", "accuracy", "coverage", "non_repetition") if k in scores}

    scored = [s["score"] for s in scores.values() if isinstance(s.get("score"), (int, float))]
    composite = round(sum(scored), 2)

    return {
        "eval_model": eval_model,
        "reference_goal": annotations["goal"],
        "reference_caption": annotations["caption"],
        "reference_apps": annotations["apps"],
        "reference_keyframes": annotations["keyframes"],
        "scores": scores,
        "composite_score": composite,
        "max_score": len(scored) * 5,
        "latency_sec": round(total_latency, 2),
    }


def get_bertscore(summary, annotations):
    """Compute BERTScore of a summary against reference descriptions and keyframe sub-goals.

    Returns a dict with precision, recall, and F1 scores (each the max across
    the reference texts).
    """
    references = []
    if annotations["description1"]:
        references.append(annotations["description1"])
    if annotations["description2"]:
        references.append(annotations["description2"])
    sub_goals = [kf.get("sub_goal", "") for kf in annotations.get("keyframes", []) if kf.get("sub_goal")]
    if sub_goals:
        references.append(" ".join(sub_goals))

    if not references:
        return None

    start = time.perf_counter()
    candidates = [summary] * len(references)
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    latency = time.perf_counter() - start

    # Take the max score across all references
    best_idx = F1.argmax().item()

    return {
        "precision": round(P[best_idx].item(), 4),
        "recall": round(R[best_idx].item(), 4),
        "f1": round(F1[best_idx].item(), 4),
        "num_references": len(references),
        "latency_sec": round(latency, 2),
    }


def get_rouge_scores(summary, annotations):
    """Compute ROUGE scores of a summary against reference descriptions and keyframe sub-goals.

    Returns a dict with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores (each the max
    across the reference texts).
    """
    references = []
    if annotations["description1"]:
        references.append(annotations["description1"])
    if annotations["description2"]:
        references.append(annotations["description2"])
    sub_goals = [kf.get("sub_goal", "") for kf in annotations.get("keyframes", []) if kf.get("sub_goal")]
    if sub_goals:
        references.append(" ".join(sub_goals))

    if not references:
        return None

    start = time.perf_counter()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    best_scores = {}
    for ref in references:
        scores = scorer.score(ref, summary)
        for metric, score_val in scores.items():
            if metric not in best_scores or score_val.fmeasure > best_scores[metric]["f1"]:
                best_scores[metric] = {
                    "precision": round(score_val.precision, 4),
                    "recall": round(score_val.recall, 4),
                    "f1": round(score_val.fmeasure, 4),
                }
    latency = time.perf_counter() - start

    return {
        "rouge1": best_scores["rouge1"],
        "rouge2": best_scores["rouge2"],
        "rougeL": best_scores["rougeL"],
        "num_references": len(references),
        "latency_sec": round(latency, 4),
    }