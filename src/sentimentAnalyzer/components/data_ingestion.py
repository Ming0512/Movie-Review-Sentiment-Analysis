import numpy as np

from tensorflow.keras.datasets import imdb

from src.sentimentAnalyzer.logging import logger
from src.sentimentAnalyzer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_save_data(self):
        """Load the IMDB dataset via Keras and persist the raw arrays
        as a single .npz artifact for downstream stages."""
        logger.info("Loading IMDB dataset via keras.datasets.imdb ...")
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
            num_words=self.config.num_words
        )

        np.savez(
            self.config.local_data_file,
            X_train=np.array(X_train, dtype=object),
            y_train=y_train,
            X_test=np.array(X_test, dtype=object),
            y_test=y_test,
        )
        logger.info(
            f"Saved raw IMDB data artifact to: {self.config.local_data_file}"
        )
