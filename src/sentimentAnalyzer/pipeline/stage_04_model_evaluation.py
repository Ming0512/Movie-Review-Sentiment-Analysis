from src.sentimentAnalyzer.config.configuration import ConfigurationManager
from src.sentimentAnalyzer.components.model_evaluation import ModelEvaluation
from src.sentimentAnalyzer.logging import logger

STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate(
            test_npz_path=data_transformation_config.transformed_test_file
        )


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
