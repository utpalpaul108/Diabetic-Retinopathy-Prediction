from pathlib import Path
import tensorflow as tf
from diabeticRetinopathy.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def _get_base_model(self):
        self.model = tf.keras.applications.ResNet50(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )
        self._save_model(path=self.config.base_model_path, model=self.model)

    def _save_model(self, path: Path, model: tf.keras.Model):
        model.save(path)

    def _prepare_full_model(self, model: tf.keras.Model, classes: list, freeze_all: bool, freeze_till, learning_rate: float, dropout_rate: float=0.5):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[freeze_till]:
                layer.trainable = False
        
        avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(avg_pooling)
        dense_layer = tf.keras.layers.Dense(2048, activation='relu')(dropout_layer)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(dense_layer)
        final_output = tf.keras.layers.Dense(classes, activation='softmax', name='final_output')(dropout_layer)
        full_model = tf.keras.models.Model(inputs=model.input, outputs=final_output)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        full_model.summary()

        return full_model
           
    def _update_base_model(self):
        self.full_model = self._prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )
        
        self._save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    def initiate_prepare_base_model(self):
        self._get_base_model()
        self._update_base_model()
    