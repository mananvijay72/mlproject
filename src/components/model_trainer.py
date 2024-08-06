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
            
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegressor":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models = models, params = params)

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


