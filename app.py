from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from diabeticRetinopathy.pipeline.diabetes.prediction_pipeline import PredictionPipeline, CustomData
import os
import pandas as pd
import numpy as np
from diabeticRetinopathy.utils import read_yaml, load_object, create_directories, decodeImage
import joblib
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
    pass
    return 'Model trained successfully'

@app.route('/train-diabetic-retinopathy-prediction-model')
def trainDiabeticRetinopathyPredictionRoute():
    pass
    return 'Model trained successfully'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        dr_class_label = ''

        input_data = CustomData(
            Pregnancies=int(request.form.get('Pregnancies')),
            Glucose=int(request.form.get('Glucose')),
            BloodPressure=int(request.form.get('BloodPressure')),
            SkinThickness=int(request.form.get('SkinThickness')),
            Insulin=int(request.form.get('Insulin')),
            BMI=float(request.form.get('BMI')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age=int(request.form.get('Age'))
        )

        input_data = input_data.get_data_as_dataframe()

        model = joblib.load('artifacts/training/diabetes_prediction_model.pkl')
        preprocessor = joblib.load('artifacts/training/preprocessor.pkl')

        features_scale = preprocessor.transform(input_data)
        prediction = model.predict(features_scale)[0]
        status = 'Non-Diabetic'
        
        if prediction != 0:
            status = 'Diabetic'

            config = read_yaml(CONFIG_FILE_PATH)
            params = read_yaml(PARAMS_FILE_PATH)

            # prediction_config = config.prediction
            create_directories(['artifacts/prediction'])

            img_path = os.path.join('artifacts/prediction', 'inputImage.jpg')

            eye_image = request.files['eye_image']
            eye_image.save(img_path)

            # Load Model
            dl_model = load_model('artifacts/training/dr_detection_model.h5')

            # Load and preprocess the test image
            img = image.load_img(img_path, target_size=(512, 512))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = dl_model.predict(img_array)
            predicted_class = np.argmax(predictions)
            classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            dr_class_label = classes[predicted_class]
        

        return render_template('index.html', prediction=status, dr_class= dr_class_label, show_prediction=True)

    
    except Exception as e:
        raise e


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)