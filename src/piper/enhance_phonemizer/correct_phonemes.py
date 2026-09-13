import re
import argparse
import torch
import os
import logging
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
import unicodedata
import string
from piper.phonemize_espeak import EspeakPhonemizer, ESPEAK_DATA_DIR
from collections import defaultdict
from .local_hazm import stopwords_list, Lemmatizer
from collections import Counter
import threading
from functools import lru_cache
from pathlib import Path


_LOGGER = logging.getLogger(__name__)

_ESPEAK_PHONEMIZER = None
_HOMOGRAPH_DATA = None
_HOMOGRAPH_DATA_PATH = None
_HOMOGRAPH_DATA_LOCK = threading.Lock()
_LEMMATIZER = None
_PERSIAN_STOPWORDS = None


class _PhonemizationCancelled(Exception):
    pass


def _check_cancelled(cancelled_callback):
    if cancelled_callback is not None and cancelled_callback():
        raise _PhonemizationCancelled()


def _get_espeak_phonemizer():
    global _ESPEAK_PHONEMIZER
    if _ESPEAK_PHONEMIZER is None:
        _ESPEAK_PHONEMIZER = EspeakPhonemizer(ESPEAK_DATA_DIR)
    return _ESPEAK_PHONEMIZER


def get_espeak_phonemizer():
    from piper.voice import _ESPEAK_PHONEMIZER_LOCK

    with _ESPEAK_PHONEMIZER_LOCK:
        return _get_espeak_phonemizer()


def _map_language_to_espeak_voice(language: str) -> str:
    lang = (language or "fa").lower()
    if lang.startswith("en"):
        return "en-us"
    return lang


@lru_cache(maxsize=10000)
def persian_phonemization_cached(text, language="fa"):
    from piper.voice import _ESPEAK_PHONEMIZER_LOCK

    voice = _map_language_to_espeak_voice(language)
    with _ESPEAK_PHONEMIZER_LOCK:
        phonemizer = _get_espeak_phonemizer()
        phoneme_sentences = phonemizer.phonemize(voice, str(text))
    sublist_strings = [''.join(sublist) for sublist in phoneme_sentences]
    result = ' '.join(sublist_strings)
    return result


def persian_phonemization(text, language="fa"):
    return persian_phonemization_cached(
        str(text), _map_language_to_espeak_voice(language)
    )


def _predict_ezafe_stream(text, model, tokenizer, cancelled_callback=None):
    _check_cancelled(cancelled_callback)
    words = text.split()
    if not words:
        return

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

    _check_cancelled(cancelled_callback)
    results = {}
    seen_word_ids = set()
    next_word_idx = 0
    for batch_idx in range(len(inputs["input_ids"])):
        _check_cancelled(cancelled_callback)
        model_inputs = {
            key: value[batch_idx : batch_idx + 1]
            for key, value in inputs.items()
            if key != "overflow_to_sample_mapping"
        }
        with torch.no_grad():
            outputs = model(**model_inputs)
            _check_cancelled(cancelled_callback)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)

        for token_idx, word_idx in enumerate(inputs.word_ids(batch_index=batch_idx)):
            _check_cancelled(cancelled_callback)
            if word_idx is None or word_idx in seen_word_ids:
                continue
            if 0 <= word_idx < len(words):
                label = predicted_labels[0][token_idx].item()
                results[word_idx] = {
                    'word': words[word_idx],
                    'needs_ezafe': bool(label),
                    'confidence': float(predictions[0][token_idx][label].item()),
                }
                seen_word_ids.add(word_idx)

        while next_word_idx in results:
            _check_cancelled(cancelled_callback)
            result = results.pop(next_word_idx)
            next_word_idx += 1
            yield result

    while next_word_idx < len(words):
        _check_cancelled(cancelled_callback)
        result = results.pop(
            next_word_idx,
            {
                "word": words[next_word_idx],
                "needs_ezafe": False,
                "confidence": 0.0,
            },
        )
        next_word_idx += 1
        yield result


def predict_ezafe_simple(text, model, tokenizer):
    return list(_predict_ezafe_stream(text, model, tokenizer))


SYMBOLS = set(string.punctuation) | set([
    '،', '؛', '«', '»', '؟', 'ـ', '“', '”', '…', '–', '—', '.', '…', '!', '?'
])

def remove_special_characters(text):
    # Define the characters to be removed
    chars_to_remove = ["ˈ", "ˌ", "ː", 'ʔ']

    # Remove each specified character from the text
    for char in chars_to_remove:
        text = text.replace(char, "")

    return text


def is_english(word):
    for char in word:
        # Check if character is in basic Latin range (A-Z, a-z)
        if not (
            '\u0041' <= char <= '\u005A' or  # A-Z
            '\u0061' <= char <= '\u007A' or  # a-z
            char == "'"  # Allow apostrophes (e.g., "don't")
        ):
            return False
    return bool(word)  # Empty string is neither


def remove_symbols(text):
    # Remove each specified character from the text
    for char in SYMBOLS:
        text = text.replace(char, "")

    return text


def split_punctuation(word):
    if not word:
        return "", "", ""

    prefix = ""
    while word and word[0] in SYMBOLS:
        prefix += word[0]
        word = word[1:]

    suffix = ""
    while word and word[-1] in SYMBOLS:
        suffix = word[-1] + suffix
        word = word[:-1]

    return prefix, word, suffix


def load_homograph_dataset(path=None):
    import pandas as pd

    configured_path = os.environ.get("HOMOGRAPH_DICT_PATH")
    if path is None:
        path = configured_path or "data/piper/homograph_dictionary.parquet"
    path = Path(path).expanduser()
    if path.exists():
        return pd.read_parquet(path)

    offline = any(
        os.environ.get(name, "").upper() in {"1", "ON", "YES", "TRUE"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    if configured_path or offline:
        raise FileNotFoundError(f"Homograph dictionary not found: {path}")

    _LOGGER.info("Downloading homograph dataset")
    dataset = pd.read_parquet(
        "https://huggingface.co/datasets/MahtaFetrat/"
        "HomoRich-G2P-Persian/resolve/main/data/train-01.parquet"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(path)
    except OSError as err:
        _LOGGER.warning("Unable to cache homograph dataset at %s: %s", path, err)
    else:
        _LOGGER.info("Homograph dataset saved to %s", path)
    return dataset


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = Lemmatizer()
    return _LEMMATIZER


def _get_persian_stopwords():
    global _PERSIAN_STOPWORDS
    if _PERSIAN_STOPWORDS is None:
        _PERSIAN_STOPWORDS = set(stopwords_list())
    return _PERSIAN_STOPWORDS


def extract_and_lemmatize_persian_words(text):
    """Extract Persian words and return their lemmas"""
    lemmatizer = _get_lemmatizer()
    persian_stopwords = _get_persian_stopwords()
    # Persian alphabet pattern (updated to match your exact specification)
    persian_pattern = r'[اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+'
    words = re.findall(persian_pattern, str(text))

    # Lemmatize and clean words
    lemmatized_words = []
    for word in words:
        word = word.lower()
        if len(word) > 1 and word not in persian_stopwords:
            # Get lemma and ensure it's not empty
            lemma = lemmatizer.lemmatize(word)
            if lemma and len(lemma) > 1:
                lemmatized_words.append(lemma)

    return lemmatized_words


def _load_homograph_data():
    global _HOMOGRAPH_DATA, _HOMOGRAPH_DATA_PATH

    path = (
        Path(
            os.environ.get("HOMOGRAPH_DICT_PATH")
            or "data/piper/homograph_dictionary.parquet"
        )
        .expanduser()
        .resolve()
    )
    with _HOMOGRAPH_DATA_LOCK:
        if _HOMOGRAPH_DATA is not None and _HOMOGRAPH_DATA_PATH == path:
            return _HOMOGRAPH_DATA

        try:
            dataset = load_homograph_dataset(path).copy()
            for column in (
                "Homograph Grapheme",
                "Homograph Phoneme",
                "IPA Homograph Phoneme",
            ):
                dataset[column] = dataset[column].map(
                    lambda value: value.strip() if isinstance(value, str) else ""
                )
                dataset = dataset[~dataset[column].isin(("", "None"))].copy()

            hom_phonemes = {}
            homograph_dict = defaultdict(lambda: defaultdict(list))
            for _, row in dataset.iterrows():
                grapheme = row["Homograph Grapheme"]
                phoneme = row["Homograph Phoneme"]
                ipa = row["IPA Homograph Phoneme"]
                hom_phonemes.setdefault(grapheme, {})[phoneme] = {
                    "Mapped": row["Mapped Homograph Phoneme"],
                    "IPA": ipa,
                }
                homograph_dict[grapheme][ipa].extend(
                    extract_and_lemmatize_persian_words(row["Grapheme"])
                )

            homograph_dict = {key: dict(value) for key, value in homograph_dict.items()}
            data = (set(hom_phonemes), hom_phonemes, homograph_dict)
        except Exception as err:
            _LOGGER.error("Failed to load homograph data from %s: %s", path, err)
            return set(), {}, {}

        if data[0]:
            _HOMOGRAPH_DATA = data
            _HOMOGRAPH_DATA_PATH = path
        return data


def homograph_text_to_phoneme(word, sentence_words, cancelled_callback=None):
    _, _, homograph_dict = _load_homograph_data()
    _check_cancelled(cancelled_callback)
    if word not in homograph_dict:
        return persian_phonemization(word)

    lemmatizer = _get_lemmatizer()
    persian_stopwords = _get_persian_stopwords()

    # Process context words (remove stopwords and count frequencies)
    context_words = Counter(
        lemmatizer.lemmatize(w) for w in sentence_words
        # w for w in sentence_words
        if w not in persian_stopwords and len(w) > 1 and w != word
    )

    # Get all phoneme options for this word
    phoneme_options = homograph_dict[word]
    best_phoneme = None
    max_normalized_score = -1
    best_overlap = -1

    for phoneme, phoneme_word_list in phoneme_options.items():
        _check_cancelled(cancelled_callback)
        # Count word frequencies in phoneme's associated words
        phoneme_word_counts = Counter(phoneme_word_list)
        total_phoneme_words = len(phoneme_word_list)

        # Calculate normalized weighted overlap score
        weighted_overlap = sum(
            count * phoneme_word_counts[word]
            for word, count in context_words.items()
            if word in phoneme_word_counts
        )

        # Normalize by phoneme word list length (avoid division by zero)
        normalized_score = weighted_overlap / total_phoneme_words if total_phoneme_words > 0 else 0

        # Select best phoneme
        if (normalized_score, weighted_overlap) > (max_normalized_score, best_overlap):
            max_normalized_score = normalized_score
            best_overlap = weighted_overlap
            best_phoneme = phoneme

    _check_cancelled(cancelled_callback)
    return best_phoneme if best_phoneme is not None else persian_phonemization(word)


def augment_subsentences_with_homograph_phonemes(subsentences, cancelled_callback=None):
    for i, subsentence in enumerate(subsentences):
        _check_cancelled(cancelled_callback)
        context_words = [remove_symbols(w) for w, E, H in subsentence]
        updated_subsentence = []

        for clean_word, (word, E, H) in zip(context_words, subsentence):
            _check_cancelled(cancelled_callback)
            if H:
                correct_phoneme = homograph_text_to_phoneme(clean_word, context_words, cancelled_callback)

                updated_subsentence.append((word, E, H, correct_phoneme))
            else:
                updated_subsentence.append((word, E, H, None))

        subsentences[i] = updated_subsentence

    return subsentences


end_of_sentence_punctuation = {'.', '?', '!', '؟'}
MAX_LENGTH = 20
confidence_threshold = 0.7

def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    text_marked = re.sub(r'([.،?!؟\n]+)', r'\1<SEP>', text)
    sentences = [s.strip() for s in text_marked.split('<SEP>') if s.strip()]
    return sentences


def _iter_subsentences(text, model, tokenizer, cancelled_callback=None):
    _check_cancelled(cancelled_callback)
    homograph_words, _, _ = _load_homograph_data()
    _check_cancelled(cancelled_callback)
    words = text.split()
    word_queue = []

    for i, word_tag in enumerate(
        _predict_ezafe_stream(text, model, tokenizer, cancelled_callback)
    ):
        _check_cancelled(cancelled_callback)
        word = word_tag['word']

        clean_word = remove_symbols(word)
        next_word = words[i + 1] if i + 1 < len(words) else ""
        clean_next = remove_symbols(next_word)

        E = (
            word_tag['needs_ezafe']
            and word_tag['confidence'] > confidence_threshold
            and bool(clean_word)
            and bool(clean_next)
            and not (word[-1] in SYMBOLS)
            and clean_next != "و"
        )

        H = (clean_word in homograph_words)
        eos_symbol = any(punct in word for punct in end_of_sentence_punctuation)

        word_queue.append((word, E, H))

        if (
            eos_symbol
            or (len(word_queue) >= MAX_LENGTH and not E)
            or (i == len(words) - 1)
        ):
            yield word_queue
            word_queue = []


def _phonemize_subsentence(subsentence, cancelled_callback=None):
    _check_cancelled(cancelled_callback)
    subsentences = augment_subsentences_with_homograph_phonemes(
        [subsentence], cancelled_callback
    )

    phoneme_words = []

    for subsentence in subsentences:
        for word, E, H, H_phoneme in subsentence:
            _check_cancelled(cancelled_callback)

            prefix, core_word, suffix = split_punctuation(word)

            if not core_word:
                phoneme_words.append(',')
                continue

            if H_phoneme:
                phoneme = H_phoneme
            else:
                lang = 'en' if is_english(core_word) else 'fa'
                _check_cancelled(cancelled_callback)
                phoneme = persian_phonemization(core_word, lang)
                _check_cancelled(cancelled_callback)

            if E:
                if phoneme.endswith('i') or phoneme.endswith('iː'):
                    phoneme = phoneme + 'je'
                if not (phoneme.endswith('e') or phoneme.endswith('eː')):
                    phoneme = phoneme + 'e'

            if prefix:
                phoneme = ',' + phoneme
            if suffix:
                phoneme = phoneme + ','

            phoneme_words.append(phoneme)

    return ' '.join(phoneme_words)


def _process_sentence(text, model, tokenizer):
    return " ".join(
        _phonemize_subsentence(subsentence)
        for subsentence in _iter_subsentences(text, model, tokenizer)
    )


def correct_output_stream(text, model, tokenizer, simplify=True, cancelled_callback=None):
    if not text or not text.strip():
        return

    try:
        _check_cancelled(cancelled_callback)
        for sentence in split_sentences(text):
            _check_cancelled(cancelled_callback)
            words = sentence.split()
            emitted_words = 0
            subsentences = _iter_subsentences(
                sentence, model, tokenizer, cancelled_callback
            )
            while emitted_words < len(words):
                _check_cancelled(cancelled_callback)
                try:
                    subsentence = next(subsentences)
                except StopIteration:
                    break
                except _PhonemizationCancelled:
                    raise
                except Exception as err:
                    _check_cancelled(cancelled_callback)
                    _LOGGER.error("Error predicting Persian phonemes: %s", err)
                    
                    while emitted_words < len(words):
                        _check_cancelled(cancelled_callback)
                        remaining = words[emitted_words : emitted_words + MAX_LENGTH]
                        processed = persian_phonemization(" ".join(remaining))
                        _check_cancelled(cancelled_callback)
                        if simplify:
                            processed = remove_special_characters(processed)
                        emitted_words += len(remaining)
                        yield list(unicodedata.normalize("NFC", processed))
                    break

                try:
                    processed = _phonemize_subsentence(subsentence, cancelled_callback)
                except _PhonemizationCancelled:
                    raise
                except Exception as err:
                    _check_cancelled(cancelled_callback)
                    _LOGGER.error("Error phonemizing Persian context: %s", err)
                    processed = persian_phonemization(
                        " ".join(word for word, E, H in subsentence)
                    )

                _check_cancelled(cancelled_callback)
                if simplify:
                    processed = remove_special_characters(processed)
                emitted_words += len(subsentence)
                yield list(unicodedata.normalize("NFC", processed))
    except _PhonemizationCancelled:
        return


def correct_output(text, model, tokenizer, simplify=True):
    return list(
        " ".join(
            "".join(chunk)
            for chunk in correct_output_stream(text, model, tokenizer, simplify)
        )
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='G2P correction script')
    parser.add_argument('--simplify', action='store_true', default=False, help='Whether to simplify the output by removing special characters.')

    # Use parse_known_args to avoid crashing if Piper passes other CLI arguments
    args, _ = parser.parse_known_args()

    # Load quantized ezafe model
    quantized_model_path = "ezafe_model_quantized"
    model = ORTModelForTokenClassification.from_pretrained(quantized_model_path)
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

    _load_homograph_data()

    # Import the communication helper
    from piper.communication import CrossPlatformServer

    # Initialize the server communicator
    server = CrossPlatformServer()
    print("✅ Ezafe G2P correction server ready")

    try:
        while True:
            # Wait for data from the input channel (blocking)
            data = server.wait_for_data()

            text = data.get("text", "").strip()

            if not text:
                continue

            corrected_phonemes = correct_output(text, model, tokenizer, simplify=args.simplify)

            # Send response back
            server.send_response(corrected_phonemes)

    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        # Clean up communication files
        server.cleanup()
