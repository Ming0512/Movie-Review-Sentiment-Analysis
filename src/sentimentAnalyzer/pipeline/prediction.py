import re
import json

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

from src.sentimentAnalyzer.config.configuration import ConfigurationManager
from src.sentimentAnalyzer.logging import logger

_word_index = None


def _get_word_index() -> dict:
    """Return the IMDB word->index mapping, shifted to match Keras's
    reserved special tokens (0=PAD, 1=START, 2=UNK, 3=UNUSED). Cached
    after first call since it's a fairly large dictionary."""
    global _word_index
    if _word_index is None:
        raw_index = imdb.get_word_index()
        _word_index = {word: (index + 3) for word, index in raw_index.items()}
        _word_index["<PAD>"] = 0
        _word_index["<START>"] = 1
        _word_index["<UNK>"] = 2
        _word_index["<UNUSED>"] = 3
    return _word_index


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _encode_review(text: str, vocab_size: int) -> list:
    word_index = _get_word_index()
    words = _clean_text(text).split()
    encoded = [1]  # <START> token
    for word in words:
        idx = word_index.get(word, 2)  # 2 = <UNK> if not found in vocab
        encoded.append(idx if idx < vocab_size else 2)
    return encoded


class PredictionPipeline:
    """Loads the trained model + its preprocessing config once, and serves
    predict() calls for raw review strings. Padding/truncating direction
    here is read from the saved model_config.json, so it always matches
    what the model was actually trained on."""

    def __init__(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        self.trained_model_path = model_trainer_config.trained_model_path
        self.model_config_path = model_trainer_config.model_config_path

        logger.info(f"Loading trained model from: {self.trained_model_path}")
        self.model = load_model(self.trained_model_path)

        with open(self.model_config_path, "r") as f:
            self.model_config = json.load(f)

    def predict(self, text: str, threshold: float = 0.5):
        max_length = self.model_config["max_length"]
        vocab_size = self.model_config["vocab_size"]
        padding = self.model_config.get("padding", "pre")
        truncating = self.model_config.get("truncating", "pre")

        encoded = _encode_review(text, vocab_size)
        padded = pad_sequences(
            [encoded], maxlen=max_length, padding=padding, truncating=truncating
        )
        prediction = self.model.predict(padded, verbose=0)[0][0]
        label = "Positive" if prediction >= threshold else "Negative"
        return label, float(prediction)
