import numpy as np
from tensorflow.keras.models import load_model

from src.sentimentAnalyzer.logging import logger
from src.sentimentAnalyzer.entity import ModelEvaluationConfig
from src.sentimentAnalyzer.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self, test_npz_path):
        test_data = np.load(test_npz_path)
        X_test, y_test = test_data["X"], test_data["y"]

        logger.info(f"Loading trained model from: {self.config.trained_model_path}")
        model = load_model(self.config.trained_model_path)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")

        save_json(
            path=self.config.metric_file_name,
            data={"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)},
        )
