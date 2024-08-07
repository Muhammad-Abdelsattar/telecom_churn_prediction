import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_total_charges(x):
    if(x==' '):
        return np.nan
    return x

def clean_senior_citizen(x):
    if(x==0):
        return "No"
    elif(x==1):
        return "Yes"
    return x

def drop_ininformative_columns(data):
    return data.drop(axis=1,labels=["customerID"])

def handle_null_values(data):
    return data.dropna()

def remove_duplicates(data):
    return data.drop_duplicates()

def clean_data(data):
    data["TotalCharges"] = data["TotalCharges"].map(clean_total_charges).astype(float)
    data["SeniorCitizen"] = data["SeniorCitizen"].map(clean_senior_citizen).astype(object)
    data = drop_ininformative_columns(data)
    data = handle_null_values(data)
    data = remove_duplicates(data)
    return data

def get_features_target(data):
    le = LabelEncoder()
    target = data["Churn"]
    target = le.fit_transform(target)
    data = data.drop(axis=1,labels=["Churn"])
    data_map = {"features":data, "target": target}
    return data_map
    
def prepare_data(data_path):
    data = pd.read_csv(data_path)
    data = clean_data(data)
    return data
