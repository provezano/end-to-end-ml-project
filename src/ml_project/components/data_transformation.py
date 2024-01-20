import os
import pandas as pd
from ml_project import logger
from ml_project.utils.common import get_size
from sklearn.model_selection import train_test_split
from ml_project.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data, random_state=42)
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info('Splitted data into training and test sets')
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)