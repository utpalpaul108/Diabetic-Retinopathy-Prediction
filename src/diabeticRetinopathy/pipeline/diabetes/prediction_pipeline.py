from diabeticRetinopathy.config import ConfigurationManager
from diabeticRetinopathy.constants import *
from diabeticRetinopathy.utils import read_yaml, load_object
import pandas as pd
import numpy as np
import os

class PredictionPipeline:

    def __init__(self):
        self.config = ConfigurationManager()
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.training_config = self.config.get_training_config()

    def _get_prediction_classes(self):
        classes = sorted(os.listdir(self.data_ingestion_config.dataset_dir))
        class_id_to_label = {i: class_name for i, class_name in enumerate(classes)}
        return class_id_to_label
    
    def predict(self, features):

        # Load Preprocessor and Model
        # preprocessor = load_object(self.training_config.trained_ml_model_path)
        # model = load_object(self.training_config.ml_preprocessor_path)

        preprocessor = load_object(os.path.join('artifacts','training','preprocessor.pkl'))
        model = load_object(os.path.join('artifacts','training','diabetes_prediction_model.pkl'))

        # # Scale the Features and Predict
        features_scale = preprocessor.transform(features)
        predict = model.predict(features_scale)
        return predict
    

# Generate Dataframe From Input Data
class CustomData:
    def __init__(self, Pregnancies:int, Glucose:int, BloodPressure:int, SkinThickness:int, Insulin:int, BMI:float, DiabetesPedigreeFunction:float, Age:int):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_dataframe(self):
        try:
            # Generate the Dataframe
            custom_input_data = {
                'Pregnancies': [self.Pregnancies],
                'Glucose': [self.Glucose],
                'BloodPressure': [self.BloodPressure],
                'SkinThickness': [self.SkinThickness],
                'Insulin': [self.Insulin],
                'BMI': [self.BMI],
                'DiabetesPedigreeFunction': [self.DiabetesPedigreeFunction],
                'Age': [self.Age]
            }

            df = pd.DataFrame(custom_input_data)
            # logging.info('Dataframe Generated from Input Data')
            return df

        except Exception as e:
            # logging.info('Custom Data Generation Exception')
            # raise CustomException(e, sys)
            pass








