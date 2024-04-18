import opendatasets as od
import shutil
import os
from PIL import Image
from diabeticRetinopathy.utils import create_directories
from diabeticRetinopathy.entity import MLDataIngestionConfig

# Data Ingestion Component
class DataIngestion:
    def __init__(self, config: MLDataIngestionConfig):
        self.config = config

    
    def _download_dataset(self):
        if not os.path.exists(self.config.ml_dataset_dir) or (not os.listdir(self.config.ml_dataset_dir)):
            try:
                create_directories([self.config.raw_dataset_dir])
                # Download the dataset
                od.download(self.config.ml_data_source_url,data_dir=self.config.raw_dataset_dir)
            except Exception as e:
                raise e
        

    def _preprocess_dataset(self):
        if os.path.exists(self.config.raw_dataset_dir):
            try:
                create_directories([self.config.ml_dataset_dir])
                for root, dirs, files in os.walk(self.config.raw_dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Check if the file is an image and valid
                        if file.lower().endswith(('.csv')):
                            # Copy the file to the destination folder
                            shutil.copy(file_path, self.config.ml_data_path)

                shutil.rmtree(self.config.raw_dataset_dir)
            
            except Exception as e:
                raise e
            
    
    def initiate_data_ingestion(self):
        self._download_dataset()
        self._preprocess_dataset()

