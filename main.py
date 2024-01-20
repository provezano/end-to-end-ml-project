from ml_project import logger
from ml_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ml_project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from ml_project.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from ml_project.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from ml_project.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'
        
try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.exception(e)


STAGE_NAME = 'Data Validation Stage'

try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.exception(e)

STAGE_NAME = 'Data Transformation Stage'

try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.exception(e)


STAGE_NAME = 'Model Trainer Stage'

try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.exception(e)




STAGE_NAME = 'Model Evaluation Stage'
import os

os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/provezano/end-to-end-ml-project.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME']='provezano'
os.environ['MLFLOW_TRACKING_PASSWORD']='c54ccb19ca6b05fb3141d21e684fc1c894086d7a'

try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.exception(e)