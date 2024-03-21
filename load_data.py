import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocessing import type_summary, detect_type, detect_missed, missing_rate

def load(filename: str):
    data = pd.read_csv(filename)
    all_type = type_summary(data)
    feature_type = detect_type(all_type, data)

    for i in feature_type[str]:
        dummy = pd.get_dummies(data[i], prefix=i)
        data = pd.concat([data, dummy], axis=1)
        data = data.drop(i, axis = 1)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = train.drop('is_canceled', axis=1)
    y_train = train['is_canceled']
    X_test = test.drop('is_canceled', axis=1)
    y_test = test['is_canceled']
    #Standarization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test

def load_dataset(filename: str):
    data = pd.read_csv(filename)
    all_type = type_summary(data)
    feature_type = detect_type(all_type, data)

    for i in feature_type[str]:
        dummy = pd.get_dummies(data[i], prefix=i)
        data = pd.concat([data, dummy], axis=1)
        data = data.drop(i, axis = 1)
    return data