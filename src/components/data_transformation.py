import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.data_transforamton_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical Column: {numerical_features}")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(drop='first')),
                    
                ]
            )

            logging.info(f"Categorical Column: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            logging.info("Preprocessor pipeline done")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transforamtion(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read")

            logging.info("Obtaing preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_feature = "math_score"
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            input_feature_train_df = train_df.drop(columns=[target_feature])
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature])
            target_feature_test_df = test_df[target_feature]

            logging.info("Applying preprocessing on training and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing objects")

            save_object(
                file_path = self.data_transforamton_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transforamton_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)




