from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    num_words: int


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_train_file: Path
    transformed_test_file: Path
    max_length: int
    padding: str
    truncating: str
    random_seed: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    model_config_path: Path
    num_words: int
    max_length: int
    padding: str
    truncating: str
    embedding_dim: int
    rnn_units: int
    dropout: float
    recurrent_dropout: float
    epochs: int
    batch_size: int
    validation_split: float
    early_stopping_patience: int
    monitor_metric: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    model_config_path: Path
    metric_file_name: Path
