import os

# ---------------------
# VLM
# ---------------------

# ollama pull qwen3.5:4b
MODEL_NAME = "qwen3.5:4b"
# Local VLMs run through the full parameter sweep (multi-threshold + multi-variable)
# when no --model override is given. Each model's result blocks are namespaced
# separately so they can be compared side by side within a single run.
# ollama pull gemma3:4b
VLM_MODELS = ["qwen3.5:4b", "gemma3:4b"]
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Maximum tokens to generate per inference call
MAX_TOKENS = 256
# Sampling temperature for VLM frame description and summary calls.
# 0 is deterministic (greedy), higher values increase output variability.
TEMPERATURE = 0
# Maximum image width sent to the VLM: frames wider than this are downscaled
# using the nearest integer to avoid fractional scaling
# to allow cleaner text rendering. A value of 0 results in no downscaling.
MAX_FRAME_WIDTH = 2048
# Use grayscale for frame differencing (faster but may miss colour-only UI changes)
ENABLE_GRAYSCALE_CONVERSION = False
# Frame pixel differencing threshold: frames with a diff score below this
# value relative to the last processed frame are skipped.
# A threshold of 0 processes every frame, higher values result in fewer frames.
DEFAULT_FRAME_DIFF_THRESHOLD = 0.01
DEFAULT_DIFF_THRESHOLD_VALUES = [0.001, 0.005, 0.01, 0.05]
# Parameter variants for perform_multi_variable_inferences(). Each entry runs a
# full, independent inference pass at DEFAULT_FRAME_DIFF_THRESHOLD, overriding
# only the listed parameters (defaults come from the constants above).
# Comment out any line to skip that variant.
MULTI_VARIABLE_CONFIGS = [
    {"name": "fullres", "max_frame_width": 0},
    {"name": "tokens_128", "max_tokens": 128},
    # tokens_256 omitted: identical to the baseline threshold run (MAX_TOKENS = 256)
    {"name": "tokens_512", "max_tokens": 512},
]
# Maximum frames per second to process.
# Limits how many frames are sent to the VLM per second of video.
# A value of 0 means no limit.
MAX_FPS = 3
FRAME_DESCRIPTION_PROMPT = (
    "Describe the current action that the user is performing on the desktop. "
    "State the action as a sentence (e.g. 'The user is clicking on the 'send' button.'). "
    "Quote ALL visible user-entered text in the application — this includes text in form fields "
    "(To, Subject, search bars, etc.) and any text being actively typed "
    "(excluding any greyed out placeholder text in search bars or form fields). "
    "Do NOT speculate about what the user might do next or infer their intentions."
)

# ---------------------
# ONLINE VLM
# ---------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MAX_TOKENS = 1024 # If the context is lower than this the results degrade considerably

# ---------------------
# SUMMARY
# ---------------------

# Maximum frame descriptions per chunk when summarising long timelines
SUMMARY_CHUNK_SIZE = 20
# Maximum characters per description sent to the summary prompt
SUMMARY_MAX_DESC_CHARS = 500
SUMMARY_PROMPT = (
    "Below is a chronological sequence of frame descriptions from a screen recording. "
    "Each entry has a frame number, timestamp, and description of what the user is doing.\n\n"
    "{entries}\n\n"
    "Perform the following tasks:\n"
    "1. COLLATE: List every frame in order, preserving the original frame number "
    "and timestamp for each. Quote ALL visible user-entered text exactly as typed.\n"
    "2. INTENT: Based on the sequence of actions, state what the user was trying "
    "to accomplish in this recording.\n\n"
    "Respond in this exact format:\n"
    "INTENT: <one or two sentences describing the user's goal>\n\n"
    "TIMELINE:\n"
    "[Frame <N>] [<timestamp>s] <description>\n"
    "[Frame <N>] [<timestamp>s] <description>\n"
    "...\n\n"
    "IMPORTANT: In the TIMELINE, do not infer, assume, or invent any actions not present "
    "in the original descriptions above. Do not autocorrect any text the user has typed. "
    "The INTENT section should infer the overall goal from the observed actions."
)

# ---------------------
# EVALUATION
# ---------------------

# ollama pull phi4:14b
EVAL_MODEL_NAME = "phi4:14b"
EVAL_PROMPT_INTENT = (
    "You are evaluating an AI-generated analysis of a screen recording.\n\n"
    "The goal was to accurately describe the user's actions in the frames and infer their intentions.\n\n"
    "GOAL: {goal}\n"
    "CAPTION: {caption}\n\n"
    "REASONING QUESTION: {reasoning_question}\n"
    "REASONING ANSWER: {reasoning_answer}\n\n"
    "PREDICTION QUESTION: {prediction_question}\n"
    "PREDICTION ANSWER: {prediction_answer}\n\n"
    "AI INFERRED INTENT: {intent}\n\n"
    "AI TIMELINE:\n{timeline}\n\n"
    "TASK: Score how well the AI's inferred intent matches the user's underlying objective, "
    "described by the goal and caption together. The goal is sometimes a terse or low-level "
    "action label, so rely on the caption to understand the true objective. Judge the core "
    "objective, not surface wording: credit semantic matches and paraphrases, and do not "
    "penalise a more detailed or differently worded intent that still captures it.\n"
    "Then ground the score in the AI TIMELINE the intent was inferred from. For the reasoning "
    "and prediction questions, look at the specific entities the question or its answer names "
    "(recipients, email addresses, subjects, titles, names, values), find the matching detail "
    "in the timeline, and state explicitly whether it MATCHES or DIFFERS. A timeline detail "
    "that differs from the answer (e.g. a different email address or subject) is a "
    "CONTRADICTION — a plausible-sounding intent built on contradicted details is not accurate.\n"
    "Scoring rule: if the timeline contradicts a named detail in either answer, the score is at "
    "most 2, however well the high-level goal is phrased. If it does not contradict anything but "
    "the timeline is too sparse to provide any evidence for the answers, lower the score by one. "
    "Award 5 only when the timeline is consistent with the answers' specifics.\n"
    "First reason in one or two sentences (name the specific detail that matched or contradicted), "
    "then give the score on a 1-5 scale.\n"
    "1 = Unrelated: the inferred intent is about a different task or objective\n"
    "2 = Mostly off: only a peripheral aspect overlaps; the core objective is missed\n"
    "3 = Partial: the core objective is partly captured but with notable gaps or inaccuracies\n"
    "4 = Largely accurate: the core objective is captured with only minor omissions or imprecision\n"
    "5 = Accurate: the core objective is fully captured, even if worded differently or with extra detail\n\n"
    "Respond with ONLY valid JSON: {{\"reasoning\": \"<one or two sentences>\", \"score\": <int>}}"
)
EVAL_PROMPT_ACCURACY = (
    "You are evaluating an AI-generated analysis of a screen recording.\n\n"
    "Below is the ground truth for the video: a caption, a NUMBERED list of keyframes "
    "(what the user actually did, including any text they typed), and reference "
    "questions/answers. Treat all of these as ground truth.\n\n"
    "CAPTION: {caption}\n\n"
    "KEYFRAMES:\n{keyframes}\n\n"
    "STATIC QA: {static_question}\n"
    "STATIC QA ANSWER: {static_answer}\n\n"
    "SEQUENTIAL-QA: {sequential_question}\n"
    "SEQUENTIAL-QA ANSWER: {sequential_answer}\n\n"
    "MCQA: {mcqa_question}\n"
    "MCQA OPTIONS: {mcqa_options}\n"
    "MCQA CORRECT ANSWER: {mcqa_answer}\n\n"
    "AI FRAME DESCRIPTIONS (in order):\n{timeline}\n\n"
    "TASK: Score ACCURACY — is the information stated in the AI frame descriptions CORRECT? "
    "This is about correctness, NOT coverage: do NOT penalise omitted actions or details "
    "(coverage is scored separately). Penalise only descriptions that CONTRADICT the keyframes "
    "or answers, or are fabricated — invented interactions, wrong applications, or made-up "
    "specifics. Check specifics against the KEYFRAMES (typed text, recipients, titles, and the "
    "active application) and flag any that disagree.\n"
    "Then, for each QA above: would the timeline let you answer it consistently with the given "
    "answer? Missing information is not an accuracy failure, but a timeline that would produce a "
    "WRONG answer is.\n"
    "STARTING-STATE EXCEPTION: applications or windows merely described as visible or open "
    "in the first few frames may be background context the user has not yet interacted with — "
    "do NOT treat these as hallucinations. Once the descriptions report active interaction, "
    "hold later claims to a stricter standard: invented or contradicted actions must be penalised.\n"
    "First reason in one or two sentences, then give the score on a 1-5 scale.\n"
    "1 = Multiple clear factual errors, or fabricated/contradicted actions or content\n"
    "2 = Several inaccuracies or some fabricated content\n"
    "3 = Roughly half correct; noticeable contradicted or fabricated details\n"
    "4 = Mostly correct, with only minor errors\n"
    "5 = Factually correct, consistent with the references, and free of hallucinated actions\n\n"
    "Respond with ONLY valid JSON: {{\"reasoning\": \"<one or two sentences>\", \"score\": <int>}}"
)
# COVERAGE: which reference keyframe actions and applications the timeline captures.
# Scored as a per-item checklist (fraction covered) rather than a holistic 1-5 rating.
EVAL_PROMPT_COVERAGE = (
    "You are evaluating an AI-generated analysis of a screen recording.\n\n"
    "Below is a NUMBERED list of ground-truth keyframes (the reference actions) and the list of "
    "applications used in the video. Decide which of them the AI timeline actually describes.\n\n"
    "KEYFRAMES:\n{keyframes}\n\n"
    "APPLICATIONS: {app_list}\n\n"
    "AI TIMELINE:\n{timeline}\n\n"
    "TASK: For each numbered keyframe, decide whether the AI timeline clearly describes that "
    "same action (allow paraphrasing and different wording, but a related topic alone does "
    "NOT count — the action itself must be conveyed). Separately decide which of the listed "
    "applications are mentioned in the timeline.\n"
    "First reason in one or two sentences, then list exactly what is covered.\n"
    "Respond with ONLY valid JSON: {{"
    "\"reasoning\": \"<one or two sentences>\", "
    "\"covered_actions\": [<numbers of covered keyframes>], "
    "\"covered_apps\": [<names of covered applications>]}}"
)
EVAL_PROMPT_NON_REPETITION = (
    "You are evaluating an AI-generated analysis of a screen recording.\n\n"
    "Below is the AI's frame-by-frame timeline. Assess NON-REPETITION: the proportion "
    "of entries that add new information versus those that merely repeat the previous "
    "action or on-screen state without a meaningful change (consecutive near-duplicates "
    "are the clearest redundancy).\n\n"
    "AI TIMELINE:\n{timeline}\n\n"
    "TASK: Judge the SHARE of redundant entries relative to the timeline's length.\n"
    "First reason in one or two sentences, then give the score on a 1-5 scale.\n"
    "1 = A large share of entries are redundant\n"
    "2 = Many entries are redundant\n"
    "3 = A moderate share of entries are redundant\n"
    "4 = A few redundant entries\n"
    "5 = Few or no redundant entries, relative to the timeline's length\n\n"
    "Respond with ONLY valid JSON: {{\"reasoning\": \"<one or two sentences>\", \"score\": <int>}}"
)
# Project root (the directory containing src/, evaluation/, output/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Evaluation-mode (batch/CLI) results — read by the dashboard.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")

# ---------------------
# DATASET
# ---------------------

# Evaluation dataset lives under evaluation/data.
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "evaluation", "data")
# Path to the GUI-World multi-video directory
GUI_WORLD_VIDEO_DIR = os.path.join(EVAL_DATA_DIR, "GUI-World", "multi")
# Videos to exclude from batch runs.
# 189–242: VR footage
# 243–425: GitLab full screen presentations
GUI_WORLD_BLACKLIST = set(range(189, 243)) | set(range(243, 426))
GUI_WORLD_MIN_VIDEO_LENGTH_SEC = 30  # seconds
# Paths to the GUI-World annotations JSONL files (train + benchmark)
GUI_WORLD_DATA_DIR = os.path.join(EVAL_DATA_DIR, "GUI-World", "Annotation")
GUI_WORLD_ANNOTATIONS = [
    os.path.join(GUI_WORLD_DATA_DIR, "train", "multi.jsonl"),
    os.path.join(GUI_WORLD_DATA_DIR, "benchmark", "multi.jsonl"),
]

# ---------------------
# SERVICE
# ---------------------

# Accepted video extensions.
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
# Reject files larger than this before decoding.
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(2 * 1024 ** 3)))

SERVICE_ROOT = os.path.join(PROJECT_ROOT, "output")
SERVICE_RUNS_DIR = os.path.join(SERVICE_ROOT, "runs")