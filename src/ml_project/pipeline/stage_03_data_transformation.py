from ml_project import logger
from ml_project.components.data_transformation import DataTransformation
from ml_project.config.configuration import ConfigurationManager

STAGE_NAME = 'Data Transformation Stage'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.train_test_split()
        except Exception as e:
            raise e
        

if __name__ == '__main__':
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<')
    except Exception as e:
        logger.exception(e)