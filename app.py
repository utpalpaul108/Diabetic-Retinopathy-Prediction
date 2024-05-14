from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from diabeticRetinopathy.pipeline.diabetes.prediction_pipeline import PredictionPipeline as MLPrediction
from diabeticRetinopathy.pipeline.diabetic_retinopathy.prediction_pipeline import PredictionPipeline as DRPrediction
from diabeticRetinopathy.pipeline.diabetes.training_pipeline import TrainingPipeline as MLTraining
from diabeticRetinopathy.pipeline.diabetic_retinopathy.training_pipeline import TrainingPipeline as DLTraining
import os
import pandas as pd
import numpy as np
from diabeticRetinopathy.utils import read_yaml, load_object, create_directories
from diabeticRetinopathy.constants import *
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train-diabetes-prediction-model')
def trainDiabetesPredictionRoute():
    training_pipeline = MLTraining()
    training_pipeline.train()
    return 'Diabetes Prediction Model trained successfully'

@app.route('/train-diabetic-retinopathy-prediction-model')
def trainDiabeticRetinopathyPredictionRoute():
    training_pipeline = DLTraining()
    training_pipeline.train()
    return 'Diabetic Retinopathy Model trained successfully'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        dr_class_label = ''
        config = read_yaml(CONFIG_FILE_PATH)
        params = read_yaml(PARAMS_FILE_PATH)

        input_features = {
            'Pregnancies': [int(request.form.get('Pregnancies'))],
            'Glucose': [int(request.form.get('Glucose'))],
            'BloodPressure': [int(request.form.get('BloodPressure'))],
            'SkinThickness': [int(request.form.get('SkinThickness'))],
            'Insulin': [int(request.form.get('Insulin'))],
            'BMI': [float(request.form.get('BMI'))],
            'DiabetesPedigreeFunction': [float(request.form.get('DiabetesPedigreeFunction'))],
            'Age': [int(request.form.get('Age'))]
        }

        # ML Model Prediction
        input_features = pd.DataFrame(input_features)
        prediction_pipeline = MLPrediction()
        ml_prediction, ml_probabilities = prediction_pipeline.predict(input_features)

        # DL Model Prediction
        prediction_config = config.prediction
        create_directories([prediction_config.root_dir])
        img_path = os.path.join(prediction_config.root_dir, 'inputImage.jpg')
        eye_image = request.files['eye_image']
        eye_image.save(img_path)
        classifier = DRPrediction(img_path)
        dl_prediction, dl_probabilities = classifier.predict() 

        if (ml_prediction != 0) or (dl_prediction != 0):
            if dl_prediction != 0:
                probability = dl_probabilities[1]
            else:
                probability = ml_probabilities[1]
            status = 'Diabetic'
        else:
            probability = dl_probabilities[0]
            status = 'Non-Diabetic'
            

        return render_template('index.html', status=status, probability=probability*100, show_prediction=True)

    
    except Exception as e:
        raise e


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)