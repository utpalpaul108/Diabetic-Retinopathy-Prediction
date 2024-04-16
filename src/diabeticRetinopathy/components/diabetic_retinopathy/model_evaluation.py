from diabeticRetinopathy.entity import EvaluationConfig
from diabeticRetinopathy.utils import save_json
from pathlib import Path
import pandas as pd
import tensorflow as tf


class Evaluate:

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        
        train_df = pd.read_csv(self.config.training_path)
        train_df['id_code'] = train_df['id_code'].apply(lambda x: x+'.png')
        train_df['diagnosis'] = train_df['diagnosis'].astype('str')

        train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.2,
            horizontal_flip=True)
        

        self.valid_generator=train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=self.config.training_images_path,
            x_col="id_code",
            y_col="diagnosis",
            batch_size=self.config.params_batch_size,
            class_mode="categorical", 
            target_size=self.config.params_image_size[:-1],
            subset='validation')

    def load_model(self, path:Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        score = model.evaluate(self.valid_generator)
        save_json(path = self.config.evaluation_score_path, data={'loss': score[0], 'accuracy': score[1]})
