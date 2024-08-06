import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.utils import evaluate_model

@dataclass
class ModelTrainingCongfig:
    model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):

        self.model_training_config = ModelTrainingCongfig()

    def initiate_model_training(self, train_array, test_array):

        try:

            logging.info("Splitting Test and Train data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            #initializing diffrent models

            models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor(),

                }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            #Best model from the model_report dictionary
            best_model_score = max(list(model_report.values()))

            #Model name with best score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            # setting the threshold for model score as 0.6
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("best model found after training")

            #saving the model
            save_object(file_path = self.model_training_config.model_file_path, 
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException(e,sys)


