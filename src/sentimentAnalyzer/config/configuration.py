from src.sentimentAnalyzer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.sentimentAnalyzer.utils.common import read_yaml, create_directories
from src.sentimentAnalyzer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.DataParams
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            num_words=params.num_words,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.DataParams
        train_params = self.params.TrainingParams
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            transformed_train_file=config.transformed_train_file,
            transformed_test_file=config.transformed_test_file,
            max_length=params.max_length,
            padding=params.padding,
            truncating=params.truncating,
            random_seed=train_params.random_seed,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_params = self.params.DataParams
        model_params = self.params.ModelParams
        train_params = self.params.TrainingParams
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            model_config_path=config.model_config_path,
            num_words=data_params.num_words,
            max_length=data_params.max_length,
            padding=data_params.padding,
            truncating=data_params.truncating,
            embedding_dim=model_params.embedding_dim,
            rnn_units=model_params.rnn_units,
            dropout=model_params.dropout,
            recurrent_dropout=model_params.recurrent_dropout,
            epochs=train_params.epochs,
            batch_size=train_params.batch_size,
            validation_split=train_params.validation_split,
            early_stopping_patience=train_params.early_stopping_patience,
            monitor_metric=train_params.monitor_metric,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            trained_model_path=self.config.model_trainer.trained_model_path,
            model_config_path=self.config.model_trainer.model_config_path,
            metric_file_name=config.metric_file_name,
        )
