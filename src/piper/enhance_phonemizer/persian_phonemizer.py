import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

from .correct_phonemes import correct_output

_LOGGER = logging.getLogger(__name__)


class PersianPhonemizer:
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "ezafe_model_quantized"

        self.model_path = str(model_path)
        _LOGGER.info("Loading Persian Ezafe model from: %s", self.model_path)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = ORTModelForTokenClassification.from_pretrained(self.model_path)
        except Exception as e:
            _LOGGER.error("Failed to load Persian Ezafe model: %s", e)
            raise RuntimeError(f"Could not load Persian G2P model from {self.model_path}") from e

    def phonemize(self, text: str) -> List[List[str]]:
        if not text or not text.strip():
            return []

        try:
            return [correct_output(text.replace('-', ' '), self.model, self.tokenizer, False)]

        except Exception as e:
            _LOGGER.error(f"Error in Persian phonemization: {e}")
            raise e
