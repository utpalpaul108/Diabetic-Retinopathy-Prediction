from diabeticRetinopathy.pipeline.diabetic_retinopathy.training_pipeline import TrainingPipeline as DLTrainingPipeline
from diabeticRetinopathy.pipeline.diabetes.training_pipeline import TrainingPipeline as MLTrainingPipeline

if __name__ == '__main__':
    # ML Training Pipeline
    ml_training_pipeline = MLTrainingPipeline()
    ml_training_pipeline.train()

    # DL Training Pipeline
    dl_training_pipeline = DLTrainingPipeline()
    dl_training_pipeline.train()
