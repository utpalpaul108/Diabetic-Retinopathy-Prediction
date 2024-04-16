from pathlib import Path
import tensorflow as tf
import pandas as pd
from diabeticRetinopathy.entity import TrainingConfig


class Traing:
    def __init__(self, config:TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):

        train_df = pd.read_csv(self.config.training_path)
        train_df['id_code'] = train_df['id_code'].apply(lambda x: x+'.png')
        train_df['diagnosis'] = train_df['diagnosis'].astype('str')

        train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.2,
            horizontal_flip=True)
        

        self.train_generator=train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=self.config.training_images_path,
            x_col="id_code",
            y_col="diagnosis",
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            target_size=self.config.params_image_size[:-1],
            subset='training')
        

        self.valid_generator=train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=self.config.training_images_path,
            x_col="id_code",
            y_col="diagnosis",
            batch_size=self.config.params_batch_size,
            class_mode="categorical", 
            target_size=self.config.params_image_size[:-1],
            subset='validation')

    def save_model(self, path: Path, model: tf.keras.Model):
        model.save(path)
        
    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.valid_generator,
            callbacks = callback_list
        )

        self.save_model(
            path = self.config.trained_model_path,
            model = self.model
        )