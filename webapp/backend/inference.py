"""
inference.py — Model loading and caption generation.

Loads the saved .keras model directly by:
  1) Enabling unsafe deserialization (model uses a Lambda layer)
  2) Injecting `tf` and `SEQ_LEN` into builtins so the Lambda's pickled code runs
  3) Patching Lambda.__init__ to include output_shape so Keras 3 can trace the graph

Falls back to DEMO MODE if TensorFlow is not installed.
"""

import csv
import io
import json
import math
import hashlib
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

from config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    IMAGE_SIZE,
    MAX_LEN,
    BEAM_SIZE,
    LENGTH_PENALTY,
    REROUTE_THRESHOLD,
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------------
# Try to import TensorFlow
# ---------------------------------------------------------------------------
tf = None
keras = None
try:
    import tensorflow as _tf
    tf = _tf
    import keras as _keras
    keras = _keras
    print(f"[inference] TensorFlow {tf.__version__} available")
except ImportError:
    print("[inference] TensorFlow not installed — DEMO MODE will be used")

# Architecture constants
SEQ_LEN = 29          # max_len(30) - 1 for teacher-forcing

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CaptionResult:
    caption: str
    confidence: float
    route: str
    rationale: str
    tokens: list[int]
    log_prob: float
    demo_mode: bool = False
    rag_caption: Optional[str] = None
    rag_used: bool = False
    retrieved_captions: list[str] = field(default_factory=list)
    retrieval_sim: Optional[float] = None


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------
_DEMO_CAPTIONS = [
    ("a giraffe standing in a field with trees in the background", 0.78),
    ("a group of people walking down a street with umbrellas", 0.65),
    ("a cat sitting on top of a laptop computer", 0.82),
    ("a large clock tower with a clock on its side", 0.41),
    ("a man riding a skateboard down a sidewalk", 0.73),
    ("a kitchen with a stove and a refrigerator", 0.85),
    ("a dog sitting on a couch next to a stuffed animal", 0.69),
    ("a woman holding a tennis racket on a court", 0.57),
    ("a bathroom with a toilet and a sink", 0.88),
    ("a plate of food with broccoli and rice", 0.62),
    ("a red double decker bus driving down a street", 0.76),
    ("a person on a snowboard in the air", 0.44),
    ("a bird perched on a branch of a tree", 0.71),
    ("a couple of boats sitting on top of a body of water", 0.38),
    ("a living room with a couch and a television", 0.81),
    ("a pizza sitting on top of a wooden cutting board", 0.74),
    ("a zebra standing in a field of tall grass", 0.79),
    ("a train traveling down tracks near a station", 0.52),
    ("a person riding a bike down a dirt road", 0.67),
    ("a sandwich sitting on top of a white plate", 0.83),
]


def _demo_caption(image_bytes: bytes) -> CaptionResult:
    img_hash = hashlib.md5(image_bytes[:4096]).hexdigest()
    idx = int(img_hash, 16) % len(_DEMO_CAPTIONS)
    caption, confidence = _DEMO_CAPTIONS[idx]
    size_factor = (len(image_bytes) % 100) / 1000.0 - 0.05
    confidence = round(min(max(confidence + size_factor, 0.05), 0.99), 4)
    route = "AUTO" if confidence >= REROUTE_THRESHOLD else "HUMAN_REVIEW"
    rationale = (
        f"Confidence {confidence:.1%} is above threshold ({REROUTE_THRESHOLD:.1%}). Caption approved for auto-publish."
        if route == "AUTO" else
        f"Confidence {confidence:.1%} is below threshold ({REROUTE_THRESHOLD:.1%}). Routed to human review for quality assurance."
    )
    return CaptionResult(caption=caption, confidence=confidence, route=route,
                         rationale=rationale, tokens=[], log_prob=-1.5, demo_mode=True)


# ---------------------------------------------------------------------------
# RAG retrieval corpus
# ---------------------------------------------------------------------------
# Loaded once at startup: list of dicts with keys:
#   base_caption, retrieved_topk_json, sim_topk_json
_rag_corpus: list[dict] = []
_rag_ready: bool = False

# Gate thresholds from Stage 7 best config
_RAG_SIM_THRESH  = 0.60   # min Jaccard similarity to use retrieval
_RAG_CONF_THRESH = 0.40   # only apply RAG when confidence < this


def _load_rag_corpus() -> None:
    """Load pre-computed retrieval candidates from Stage 7 CSV."""
    global _rag_corpus, _rag_ready
    corpus_path = Path(PROJECT_ROOT) / "artifacts" / "stage7" / "retrieval_candidates_topk.csv"
    if not corpus_path.exists():
        print(f"[rag] Corpus not found at {corpus_path} — RAG disabled")
        return
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            _rag_corpus = list(csv.DictReader(f))
        _rag_ready = True
        print(f"[rag] Loaded {len(_rag_corpus)} retrieval candidates")
    except Exception as e:
        print(f"[rag] Failed to load corpus: {e}")


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _retrieve(caption: str, top_k: int = 5) -> tuple[list[str], float]:
    """
    Find the most similar pre-computed candidate entry by Jaccard similarity
    between the generated caption and corpus base_captions.
    Returns (top_k retrieved captions, best similarity score).
    """
    if not _rag_corpus:
        return [], 0.0
    best_sim = -1.0
    best_row = None
    for row in _rag_corpus:
        sim = _jaccard(caption, row.get("base_caption", ""))
        if sim > best_sim:
            best_sim = sim
            best_row = row
    if best_row is None:
        return [], 0.0
    try:
        retrieved = json.loads(best_row["retrieved_topk_json"])[:top_k]
    except Exception:
        retrieved = [best_row.get("retrieved_caption_top1", "")]
    return retrieved, best_sim


def apply_rag(caption: str, confidence: float) -> tuple[str, list[str], float, bool]:
    """
    Apply RAG refinement using Stage 7 gate policy.
    Returns (final_caption, retrieved_list, sim_score, rag_was_used).
    """
    if not _rag_ready:
        return caption, [], 0.0, False

    retrieved, sim = _retrieve(caption)

    # Gate: only replace when similarity is high enough AND confidence is low
    if sim >= _RAG_SIM_THRESH and confidence < _RAG_CONF_THRESH and retrieved:
        return retrieved[0], retrieved, sim, True

    return caption, retrieved, sim, False


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_caption_model = None
_word_index: dict[str, int] = {}
_index_word: dict[int, str] = {}
_start_id: int = 0
_end_id: int = 0
_models_ready: bool = False
_demo_mode: bool = False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def load_models() -> dict:
    global _caption_model, _word_index, _index_word, _start_id, _end_id
    global _models_ready, _demo_mode

    if tf is None:
        _demo_mode = True
        _models_ready = True
        print("[inference] ⚡ DEMO MODE active — returning sample COCO captions")
        _load_rag_corpus()
        return {"ok": True, "demo_mode": True, "errors": ["TensorFlow not installed"]}

    errors: list[str] = []

    # --- Load the full model directly ------------------------------------
    try:
        # The saved .keras model contains a Lambda layer whose pickled code
        # references `tf` and `SEQ_LEN` as globals. We inject them via builtins
        # so the Lambda function can execute.
        builtins.tf = tf
        builtins.SEQ_LEN = SEQ_LEN

        # Enable unsafe deserialization (Lambda layers contain pickled code)
        keras.config.enable_unsafe_deserialization()

        # Patch Lambda.__init__ to provide output_shape so Keras 3 can trace
        _orig_init = keras.layers.Lambda.__init__
        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault(
                'output_shape',
                lambda input_shape: (input_shape[0], SEQ_LEN, input_shape[-1])
            )
            _orig_init(self, *args, **kwargs)
        keras.layers.Lambda.__init__ = _patched_init

        # Load the model with all weights (MobileNetV2 + caption head)
        _caption_model = keras.models.load_model(
            str(MODEL_PATH), compile=False, safe_mode=False
        )

        # Restore original Lambda init
        keras.layers.Lambda.__init__ = _orig_init

        print(f"[inference] Model loaded from {MODEL_PATH}")
        for i, inp in enumerate(_caption_model.inputs):
            print(f"  input[{i}]: name={inp.name}  shape={inp.shape}")
        for i, out in enumerate(_caption_model.outputs):
            print(f"  output[{i}]: name={out.name}  shape={out.shape}")

    except Exception as e:
        errors.append(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

    # --- Tokenizer --------------------------------------------------------
    try:
        with open(TOKENIZER_PATH, "r") as f:
            tok_json = f.read()

        try:
            from tf_keras.preprocessing.text import tokenizer_from_json
            _tokenizer = tokenizer_from_json(tok_json)
            _word_index = _tokenizer.word_index
            _index_word = {int(k): v for k, v in _tokenizer.index_word.items()}
        except (ImportError, AttributeError):
            tok_data = json.loads(tok_json)
            config = tok_data.get("config", tok_data)
            _word_index = json.loads(config.get("word_index", "{}"))
            _index_word = {int(v): k for k, v in _word_index.items()}

        _start_id = _word_index.get("<start>", _word_index.get("start", 2))
        _end_id = _word_index.get("<end>", _word_index.get("end", 3))
        print(f"[inference] Tokenizer loaded — vocab {len(_word_index)} words, "
              f"start_id={_start_id}, end_id={_end_id}")
    except Exception as e:
        errors.append(f"Failed to load tokenizer: {e}")

    if errors:
        _demo_mode = True
        _models_ready = True
        print(f"[inference] ⚡ DEMO MODE active (fallback) — {len(errors)} load error(s)")
        _load_rag_corpus()
        return {"ok": True, "demo_mode": True, "errors": errors}

    _models_ready = True
    _demo_mode = False
    _load_rag_corpus()
    return {"ok": True, "demo_mode": False, "errors": []}


def is_ready() -> bool:
    return _models_ready

def is_demo() -> bool:
    return _demo_mode

def is_rag_ready() -> bool:
    return _rag_ready


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Read raw bytes → PIL → (1, 224, 224, 3) MobileNetV2-preprocessed."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _tokens_to_caption(token_ids: list[int]) -> str:
    words = []
    for tid in token_ids:
        if tid == _end_id:
            break
        word = _index_word.get(tid, "")
        if word and word not in ("<start>", "<end>", "<pad>", "<unk>"):
            words.append(word)
    return " ".join(words)


def _greedy_decode(img_array: np.ndarray) -> tuple[list[int], float]:
    seq = np.zeros((1, SEQ_LEN), dtype=np.int32)
    seq[0, 0] = _start_id
    log_prob_sum = 0.0

    for i in range(SEQ_LEN - 1):
        preds = _caption_model.predict([img_array, seq], verbose=0)
        next_probs = preds[0, i, :]
        next_id = int(np.argmax(next_probs))
        prob = float(next_probs[next_id])
        log_prob_sum += math.log(max(prob, 1e-12))
        if next_id == _end_id or next_id == 0:
            break
        seq[0, i + 1] = next_id

    token_ids = [int(t) for t in seq[0] if t != 0]
    return token_ids, log_prob_sum


def _beam_search_decode(img_array: np.ndarray, beam_size: int = BEAM_SIZE) -> tuple[list[int], float]:
    initial_seq = np.zeros((SEQ_LEN,), dtype=np.int32)
    initial_seq[0] = _start_id
    beams = [(initial_seq.copy(), 0.0)]
    completed = []

    for step in range(1, SEQ_LEN):
        all_candidates = []
        for seq, score in beams:
            if step > 1 and seq[step - 1] == _end_id:
                completed.append((seq, score))
                continue
            seq_batch = np.expand_dims(seq, axis=0)
            preds = _caption_model.predict([img_array, seq_batch], verbose=0)
            next_probs = preds[0, step - 1, :]
            next_probs = np.clip(next_probs, 1e-12, 1.0)
            log_probs = np.log(next_probs)
            top_k_ids = np.argsort(log_probs)[-beam_size:]
            for tid in top_k_ids:
                new_seq = seq.copy()
                new_seq[step] = int(tid)
                new_score = score + float(log_probs[tid])
                all_candidates.append((new_seq, new_score))
        if not all_candidates:
            break
        all_candidates.sort(
            key=lambda x: x[1] / max(1, _count_tokens(x[0])) ** LENGTH_PENALTY,
            reverse=True,
        )
        beams = all_candidates[:beam_size]

    completed.extend(beams)
    if not completed:
        return _greedy_decode(img_array)

    best_seq, best_score = max(
        completed,
        key=lambda x: x[1] / max(1, _count_tokens(x[0])) ** LENGTH_PENALTY,
    )
    n_tokens = _count_tokens(best_seq)
    normalised_score = best_score / max(1, n_tokens) ** LENGTH_PENALTY
    return [int(t) for t in best_seq if t != 0], normalised_score


def _count_tokens(seq) -> int:
    count = 0
    for t in seq:
        if t == 0:
            break
        count += 1
    return max(count - 1, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_caption(image_bytes: bytes, use_beam: bool = True, use_rag: bool = False) -> CaptionResult:
    if not _models_ready:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    if _demo_mode:
        result = _demo_caption(image_bytes)
        if use_rag:
            rag_cap, retrieved, sim, rag_used = apply_rag(result.caption, result.confidence)
            result.rag_caption = rag_cap
            result.rag_used = rag_used
            result.retrieved_captions = retrieved
            result.retrieval_sim = round(sim, 4)
        return result

    # 1. Preprocess image → (1, 224, 224, 3)
    img_array = _preprocess_image(image_bytes)

    # 2. Decode (model takes raw image, MobileNetV2 is inside)
    if use_beam and BEAM_SIZE > 1:
        tokens, log_prob = _beam_search_decode(img_array, BEAM_SIZE)
    else:
        tokens, log_prob = _greedy_decode(img_array)

    # 3. Caption text
    caption = _tokens_to_caption(tokens)

    # 4. Confidence
    n_tokens = max(len([t for t in tokens if t not in (0, _start_id, _end_id)]), 1)
    avg_log_prob = log_prob / (n_tokens ** LENGTH_PENALTY)
    confidence = 1.0 / (1.0 + math.exp(-avg_log_prob - 1.0))
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    # 5. Route
    route = "AUTO" if confidence >= REROUTE_THRESHOLD else "HUMAN_REVIEW"
    rationale = (
        f"Confidence {confidence:.1%} is above threshold ({REROUTE_THRESHOLD:.1%}). Caption approved for auto-publish."
        if route == "AUTO" else
        f"Confidence {confidence:.1%} is below threshold ({REROUTE_THRESHOLD:.1%}). Routed to human review for quality assurance."
    )

    # 6. Optional RAG refinement
    rag_caption = None
    retrieved_captions: list[str] = []
    retrieval_sim: Optional[float] = None
    rag_used = False

    if use_rag:
        rag_caption, retrieved_captions, sim, rag_used = apply_rag(caption, confidence)
        retrieval_sim = round(sim, 4)

    return CaptionResult(
        caption=caption or "(no caption generated)",
        confidence=confidence,
        route=route,
        rationale=rationale,
        tokens=tokens,
        log_prob=round(log_prob, 4),
        demo_mode=False,
        rag_caption=rag_caption,
        rag_used=rag_used,
        retrieved_captions=retrieved_captions,
        retrieval_sim=retrieval_sim,
    )
