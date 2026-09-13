"""Experimental repeated phonemes and validated first-copy boundaries for Persian speech."""

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional

from .const import BOS, EOS, PAD

MAX_TEXT_CHARACTERS = 64
MAX_PHONEMES = 40
MIN_CONTEXT_PHONEMES = 32


def is_short_persian_text(text: str) -> bool:
    """Accept one Arabic-script word, preserving marks and optional edge punctuation."""
    text = text.strip()
    if not text or len(text) > MAX_TEXT_CHARACTERS or "[[" in text:
        return False
    text = text.strip(".,،؛:!?؟…\"'«»()")
    has_letter = False
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("L") and unicodedata.name(char, "").startswith(
            "ARABIC "
        ):
            has_letter = True
        elif not category.startswith("M") and char not in "\u200c\u200d":
            return False
    return has_letter


def validate_duration_output(session: Any, hop_length: int) -> None:
    """Require the explicitly prepared model's existing duration output, in frames."""
    outputs = session.get_outputs()
    metadata = session.get_modelmeta().custom_metadata_map
    source_hash = metadata.get("piper.source_model_sha256", "")
    if (
        len(outputs) != 2
        or metadata.get("piper.duration_output") != outputs[1].name
        or metadata.get("piper.duration_unit") != "frames"
        or metadata.get("piper.duration_hop_length") != str(hop_length)
        or not isinstance(hop_length, int)
        or hop_length <= 0
        or outputs[1].type != "tensor(float)"
        or len(outputs[1].shape) != 3
        or outputs[1].shape[1] != 1
        or isinstance(outputs[1].shape[2], int)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ValueError(
            "Short speech repetition requires a prepared model with valid frame durations"
        )


@dataclass(frozen=True)
class RepeatPlan:
    """Actual inference input and the first copy's ID interval, including its pads."""

    phonemes: tuple[str, ...]
    phoneme_ids: tuple[int, ...]
    first_start_id: int
    first_end_id: int
    copy_id_spans: tuple[tuple[int, int], ...]


def plan_repeated_speech(
    phonemes: Sequence[str],
    phoneme_id_map: Mapping[str, Sequence[int]],
) -> Optional[RepeatPlan]:
    """Repeat a bounded chunk without changing symbols or assuming one ID per symbol."""
    if not 0 < len(phonemes) <= MAX_PHONEMES:
        return None
    original = tuple(phonemes)
    repeated = original + (" ",) + original
    while len(repeated) < MIN_CONTEXT_PHONEMES:
        repeated += (" ",) + original
    for symbol in (BOS, PAD, EOS, *repeated):
        ids = phoneme_id_map.get(symbol)
        if not ids or any(
            not isinstance(value, Integral) or value < 0 for value in ids
        ):
            raise ValueError("Short speech requires a complete, valid phoneme ID map")
    pad = tuple(phoneme_id_map[PAD])
    ids = [*phoneme_id_map[BOS], *pad]
    first_start = len(ids)
    first_end = first_start
    copy_start = first_start
    copy_spans = []
    for index, symbol in enumerate(repeated):
        if index % (len(original) + 1) == 0:
            copy_start = len(ids)
        ids.extend(phoneme_id_map[symbol])
        ids.extend(pad)
        if index % (len(original) + 1) == len(original) - 1:
            copy_spans.append((copy_start, len(ids)))
        if index + 1 == len(original):
            first_end = len(ids)
    ids.extend(phoneme_id_map[EOS])
    return RepeatPlan(repeated, tuple(ids), first_start, first_end, tuple(copy_spans))


def frame_counts_to_samples(frame_counts: Sequence[Real], hop_length: int) -> list[int]:
    """Reject malformed frame counts before an integer cast could hide bad alignments."""
    if any(
        not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
        for value in frame_counts
    ):
        raise ValueError(
            "Short speech durations must contain finite, nonnegative whole frames"
        )
    return [int(value) * hop_length for value in frame_counts]


def first_copy_sample_bounds(
    plan: RepeatPlan,
    phoneme_id_samples: Sequence[int],
    audio_samples: int,
) -> tuple[int, int]:
    """Reject invalid alignments before any repeated audio can reach playback."""
    if len(phoneme_id_samples) != len(plan.phoneme_ids):
        raise ValueError("Short speech alignment length does not match the phoneme IDs")
    if any(
        not isinstance(value, Integral) or value < 0 for value in phoneme_id_samples
    ):
        raise ValueError(
            "Short speech alignments require nonnegative integer sample counts"
        )
    counts = [int(value) for value in phoneme_id_samples]
    if sum(counts) != audio_samples:
        raise ValueError("Short speech alignment samples do not match the audio length")
    start = sum(counts[: plan.first_start_id])
    end = start + sum(counts[plan.first_start_id : plan.first_end_id])
    if not 0 <= start < end < audio_samples:
        raise ValueError(
            "Short speech alignment gives an empty or invalid first-copy interval"
        )
    if any(
        sum(counts[start_id:end_id]) <= 0 for start_id, end_id in plan.copy_id_spans
    ):
        raise ValueError(
            "Short speech alignments must give every repeated copy a nonempty interval"
        )
    return start, end
