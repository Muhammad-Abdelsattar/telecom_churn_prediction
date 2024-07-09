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
        "rf" : RandomForestClassifier,
        "xgb" : XGBClassifier,
        "lr" : LogisticRegression,
        "svc" : SVC,
    }
    return model_dispatcher[config["use_model"]](**config["params"])


def build_pipeline(config,categorical_columns):
    preprocessor = build_preprocessor(categorical_columns)
    model = build_model(config["model"])
    pipeline = Pipeline(
        steps=[
            ("preprocessing",preprocessor),
            ("PCA",PCA(n_components=17)),
            ("model",model)
    ])
    return pipeline