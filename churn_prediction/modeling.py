import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (PowerTransformer,
                                   OneHotEncoder,
                                   LabelEncoder,
                                   MinMaxScaler,
                                   StandardScaler)

def build_preprocessor(categorical_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHot",OneHotEncoder(),categorical_columns),
            ("Power",PowerTransformer(),["TotalCharges"]),
            ("Standard",StandardScaler(),["MonthlyCharges"]),
            ("MinMax",StandardScaler(),["tenure"]),
        ])
    return preprocessor

def build_model(config):
    model_dispatcher = {
        "rf" : RandomForestClassifier(**(config["models_parameters"]["rf_params"])),
        "xgb" : XGBClassifier(**config["models_parameters"]["xgb_params"]),
        "lr" : LogisticRegression(**config["models_parameters"]["lr_params"]),
        "svc" : SVC(**config["models_parameters"]["svc_params"]),
    }
    return model_dispatcher[config["use_model"]]


def build_pipeline(config,categorical_columns):
    preprocessor = build_preprocessor(categorical_columns)
    model = build_model(config["models"])
    pipeline = Pipeline(
        steps=[
            ("preprocessing",preprocessor),
            ("PCA",PCA(n_components=17)),
            ("model",model)
    ])
    return pipeline