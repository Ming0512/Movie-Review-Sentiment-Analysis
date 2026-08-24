import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.sentimentAnalyzer.logging import logger
from src.sentimentAnalyzer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _build_model(self) -> Sequential:
        model = Sequential([
            Input(shape=(self.config.max_length,)),
            Embedding(
                input_dim=self.config.num_words,
                output_dim=self.config.embedding_dim,
            ),
            SimpleRNN(
                self.config.rnn_units,
                return_sequences=False,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.recurrent_dropout,
            ),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, train_npz_path, test_npz_path):
        train_data = np.load(train_npz_path)
        X_train, y_train = train_data["X"], train_data["y"]

        gpus = tf.config.list_physical_devices("GPU")
        device = "/GPU:0" if gpus else "/CPU:0"
        logger.info(f"GPUs available: {gpus if gpus else 'none'} | using {device}")

        model = self._build_model()
        model.summary(print_fn=logger.info)

        mode = "max" if "accuracy" in self.config.monitor_metric else "min"
        checkpoint_callback = ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            monitor=self.config.monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1,
        )
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1,
        )

        with tf.device(device):
            model.fit(
                X_train,
                y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=[checkpoint_callback, early_stop_callback],
            )

        with open(self.config.model_config_path, "w") as f:
            json.dump(
                {
                    "max_length": self.config.max_length,
                    "vocab_size": self.config.num_words,
                    "padding": self.config.padding,
                    "truncating": self.config.truncating,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved model config to: {self.config.model_config_path}")
        logger.info(f"Best model checkpoint saved to: {self.config.trained_model_path}")
