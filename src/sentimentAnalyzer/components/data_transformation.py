import numpy as np
from tensorflow.keras.utils import pad_sequences

from src.sentimentAnalyzer.logging import logger
from src.sentimentAnalyzer.entity import DataTransformationConfig
from src.sentimentAnalyzer.config.configuration import ConfigurationManager


class DataTransformation:
    """Handles padding/truncation and the train-shuffle fix.

    NOTE: Keras's IMDB dataset is sorted by label (negatives first, then
    positives). model.fit's validation_split takes the LAST N% of the
    array WITHOUT shuffling first, so without shuffling here, the
    validation split ends up almost entirely one class and val_accuracy
    gets stuck near 0.50 regardless of how well the model is learning.

    Padding direction matters too: for a SimpleRNN, the final hidden state
    is most influenced by whatever it processed last, so "pre" padding
    (padding at the front, real content ending at max_length) works far
    better than "post" padding for short sequences.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
        # Data ingestion's local_data_file path comes from config.yaml,
        # so re-fetch it here rather than duplicating the path.
        ingestion_config = ConfigurationManager().get_data_ingestion_config()
        raw = np.load(ingestion_config.local_data_file, allow_pickle=True)
        X_train, y_train = raw["X_train"], raw["y_train"]
        X_test, y_test = raw["X_test"], raw["y_test"]

        logger.info("Padding/truncating sequences...")
        X_train = pad_sequences(
            X_train,
            maxlen=self.config.max_length,
            padding=self.config.padding,
            truncating=self.config.truncating,
        )
        X_test = pad_sequences(
            X_test,
            maxlen=self.config.max_length,
            padding=self.config.padding,
            truncating=self.config.truncating,
        )

        logger.info("Shuffling training data before any validation split...")
        rng = np.random.RandomState(self.config.random_seed)
        shuffle_idx = rng.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        np.savez(self.config.transformed_train_file, X=X_train, y=y_train)
        np.savez(self.config.transformed_test_file, X=X_test, y=y_test)
        logger.info(
            f"Saved transformed train/test artifacts to "
            f"{self.config.transformed_train_file} and {self.config.transformed_test_file}"
        )
