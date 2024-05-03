from objectDetection.config.configuration import ConfigurationManager
from objectDetection.components.training import Training
from objectDetection import logger
import mlflow
from ultralytics import settings

# settings.update({'mlflow': False,
#                  'datasets_dir':'../artifacts/data_ingestion/dataset'})  

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        print(training_config)
        mlflow.set_experiment(training_config.params_project)
        training = Training(config=training_config)
        with mlflow.start_run(run_name=training_config.params_name):
            mlflow.log_params(vars(training_config))
            training.get_base_model()
            training.train()
                  
     
      

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e