from src.sentimentAnalyzer.config.configuration import ConfigurationManager
from src.sentimentAnalyzer.components.model_trainer import ModelTrainer
from src.sentimentAnalyzer.logging import logger

STAGE_NAME = "Model Trainer stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train(
            train_npz_path=data_transformation_config.transformed_train_file,
            test_npz_path=data_transformation_config.transformed_test_file,
        )


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
