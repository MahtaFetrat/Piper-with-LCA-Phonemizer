import pandas as pd
import re
import sys
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
from hazm import stopwords_list, Lemmatizer
from collections import Counter


_LOGGER = logging.getLogger(__name__)

_ESPEAK_PHONEMIZER = None

def get_espeak_phonemizer():
    global _ESPEAK_PHONEMIZER
    if _ESPEAK_PHONEMIZER is None:
        _ESPEAK_PHONEMIZER = EspeakPhonemizer(ESPEAK_DATA_DIR)
    return _ESPEAK_PHONEMIZER

def _map_language_to_espeak_voice(language: str) -> str:
    lang = (language or "fa").lower()
    if lang.startswith("en"):
        return "en-us"
    return lang


def persian_phonemization(text, language="fa"):
    phonemizer = get_espeak_phonemizer()
    voice = _map_language_to_espeak_voice(language)
    phoneme_sentences = phonemizer.phonemize(voice, str(text))
    sublist_strings = [''.join(sublist) for sublist in phoneme_sentences]
    result = ' '.join(sublist_strings)
    return result


def predict_ezafe_simple(text, model, tokenizer):
    words = text.split()
    if not words:
        return []

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_labels = torch.argmax(predictions, dim=-1)

    word_ids = inputs.word_ids()
    results = []

    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            if word_idx < len(words):
                results.append({
                    'word': words[word_idx],
                    'needs_ezafe': bool(predicted_labels[0][i].item()),
                    'confidence': float(predictions[0][i][predicted_labels[0][i]].item())
                })
            previous_word_idx = word_idx

    return results


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


def load_homograph_dataset():
    homograph_dict_path = os.getenv("HOMOGRAPH_DICT_PATH", "./data/piper/homograph_dictionary.parquet")
    if os.path.exists(homograph_dict_path):
        return pd.read_parquet(homograph_dict_path)
    else:
        _LOGGER.info(f"HOMOGRAPH_DICT_PATH not set or file does not exist. Downloading dataset...")
        homograph_dict_url = "https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian/resolve/main/data/train-01.parquet"
        dataset = pd.read_parquet(homograph_dict_url)
        dataset.to_parquet(homograph_dict_path)
        _LOGGER.info(f"Dataset downloaded and saved to: {homograph_dict_path}")
        return dataset


dataset = load_homograph_dataset()

homograph_words = set(dataset['Homograph Grapheme'])

dataset = dataset[
    dataset['Homograph Phoneme'].notna() &
    (dataset['Homograph Phoneme'] != '') &
    (dataset['Homograph Phoneme'] != 'None')
]


# First, make sure the relevant columns are stripped of whitespace
dataset['Homograph Grapheme'] = dataset['Homograph Grapheme'].str.strip()
dataset['Homograph Phoneme'] = dataset['Homograph Phoneme'].str.strip()

# Group by Homograph Grapheme and Homograph Phoneme
hom_phonemes = {}

for _, row in dataset.iterrows():
    grapheme = row['Homograph Grapheme']
    phoneme = row['Homograph Phoneme']
    mapped = row['Mapped Homograph Phoneme']
    ipa = row['IPA Homograph Phoneme']

    if grapheme not in hom_phonemes:
        hom_phonemes[grapheme] = {}

    hom_phonemes[grapheme][phoneme] = {
        'Mapped': mapped,
        'IPA': ipa
    }


# Initialize Hazm components
persian_stopwords = set(stopwords_list())
lemmatizer = Lemmatizer()


def extract_and_lemmatize_persian_words(text):
    """Extract Persian words and return their lemmas"""
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


# Create the nested dictionary structure
homograph_dict = defaultdict(lambda: defaultdict(list))

for _, row in dataset.iterrows():
    if pd.notna(row['Homograph Grapheme']) and pd.notna(row['Homograph Phoneme']):
        grapheme = row['Homograph Grapheme']
        homograph_phoneme = row['IPA Homograph Phoneme']

        # Extract and lemmatize Persian words
        phoneme_lemmas = extract_and_lemmatize_persian_words(row['Grapheme'])

        # Add lemmas to the dictionary
        homograph_dict[grapheme][homograph_phoneme].extend(phoneme_lemmas)

# Convert defaultdict to regular dict
homograph_dict = {k: dict(v) for k, v in homograph_dict.items()}


# Get Persian stopwords
persian_stopwords = set(stopwords_list())

def homograph_text_to_phoneme(word, sentence_words):
    if not homograph_dict or lemmatizer is None:
        return persian_phonemization(word)

    # Process context words (remove stopwords and count frequencies)
    context_words = Counter(
        lemmatizer.lemmatize(w) for w in sentence_words
        # w for w in sentence_words
        if w not in persian_stopwords and len(w) > 1 and w != word
    )

    # Return default if word not in homograph dict
    if word not in homograph_dict:
        return persian_phonemization(word)

    # Get all phoneme options for this word
    phoneme_options = homograph_dict[word]
    best_phoneme = None
    max_normalized_score = -1

    for phoneme, phoneme_word_list in phoneme_options.items():
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
        if normalized_score > max_normalized_score:
            max_normalized_score = normalized_score
            best_phoneme = phoneme
        elif normalized_score == max_normalized_score:
            # Tiebreaker: prefer the phoneme with higher raw overlap
            if best_phoneme is None or weighted_overlap > sum(
                context_words[word] * Counter(phoneme_options[best_phoneme]).get(word, 0)
                for word in context_words
            ):
                best_phoneme = phoneme

    return best_phoneme if best_phoneme is not None else persian_phonemization(word)


def augment_subsentences_with_homograph_phonemes(subsentences):
    for i, subsentence in enumerate(subsentences):
        context_words = [remove_symbols(w) for w, E, H in subsentence]
        updated_subsentence = []

        for clean_word, (word, E, H) in zip(context_words, subsentence):
            if H:
                correct_phoneme = homograph_text_to_phoneme(clean_word, context_words)

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


def _process_sentence(text, model, tokenizer):
    word_tags = predict_ezafe_simple(text, model, tokenizer)

    word_queue = []
    subsentences = []
    for i, word_tag in enumerate(word_tags):
        word = word_tag['word']

        clean_word = remove_symbols(word)
        clean_next = "" if i == len(word_tags) - 1 else remove_symbols(word_tags[i+1]['word'])

        E = (
            word_tag['needs_ezafe']
            and word_tag['confidence'] > confidence_threshold
            and clean_word and len(clean_next) > 0
            and not (word[-1] in SYMBOLS)
            and not (word_tags[i+1]['word'] == "و")
        )

        H = (clean_word in homograph_words)
        eos_symbol = any(punct in word for punct in end_of_sentence_punctuation)

        if (eos_symbol) or (len(word_queue) >= MAX_LENGTH and not E) or (i == len(word_tags) - 1):
            word_queue.append((word, E, H))
            subsentences.append(word_queue)
            word_queue = []

        else:
            word_queue.append((word, E, H))

    subsentences = augment_subsentences_with_homograph_phonemes(subsentences)

    phoneme_words = []

    for subsentence in subsentences:
        for word, E, H, H_phoneme in subsentence:

            prefix, core_word, suffix = split_punctuation(word)

            if not core_word:
                phoneme_words.append(',')
                continue

            phoneme = persian_phonemization(core_word, 'en' if is_english(core_word) else 'fa') if not H_phoneme else H_phoneme

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


def correct_output(text, model, tokenizer, simplify=True):
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)

    all_phonemes_parts = []

    for sent in sentences:
        try:
            processed_part = _process_sentence(sent, model, tokenizer)
            all_phonemes_parts.append(processed_part)
        except Exception as e:
            _LOGGER.error(f"Error processing sentence chunk '{sent}': {e}")
            all_phonemes_parts.append(persian_phonemization(sent))

    corrected_phonemes = ' '.join(all_phonemes_parts)

    if simplify:
        corrected_phonemes = remove_special_characters(corrected_phonemes)

    return list(unicodedata.normalize("NFC", corrected_phonemes))


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

    # Import the communication helper
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
