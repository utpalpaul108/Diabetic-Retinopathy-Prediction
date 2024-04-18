import pandas as pd
import numpy as np
from diabeticRetinopathy.entity import MLDataPreprocessingConfig
from diabeticRetinopathy.utils import save_object

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # For Handling Missing Values
from sklearn.preprocessing import StandardScaler # For Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # For Ordinal Encoding

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# DataPreprocessing Component
class DataPreprocessing:
    def __init__(self, config: MLDataPreprocessingConfig):
        self.config = config

    def _data_preprocessor(self, numerical_features, categorical_features):
        '''
        Preprocess the raw dataset
        '''

        # Numerical Pipeline
        num_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ]
        )

        # Categorical Pipeline
        cat_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_features),
            ('cat_pipeline', cat_pipeline, categorical_features)
        ])

        return preprocessor

    
    def initiate_data_preprocessing(self) -> None:
        
        try:
            df = pd.read_csv(self.config.data_path)
            
            # Split into independent and dependent features
            X = df.iloc[:,:-1]
            y = df.iloc[:,-1]

            # Data over sampling
            oversample = SMOTE()
            X,y = oversample.fit_resample(X, y)

            # Segregating Numerical and Categorical features
            numerical_features = [feature for feature in X.columns if X[feature].dtypes !=object]
            categorical_features = [feature for feature in X.columns if X[feature].dtypes ==object]

            preprocessor = self._data_preprocessor(numerical_features=numerical_features, categorical_features=categorical_features)
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42,shuffle=True)
            
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.fit_transform(X_test)

            save_object(self.config.preprocessor_path, preprocessor)

            return X_train, X_test, y_train, y_test 

        except Exception as e:
            raise e
    
    