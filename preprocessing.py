import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler

#return the missing rate of each column
def missing_rate(data: pd.DataFrame) -> dict:
    rate = {}
    for i in data.columns:
        rate[i] = (data[i].isnull().sum()/data.shape[0]).round(4)
    return rate

#find the features with missing values
def detect_missed(rate: dict) -> list:
    features = []
    for feature in rate.keys():
        if rate[feature] > 0:
            features.append(feature)
    return features

#figure out the number of different types in this dataset
def type_summary(data: pd.DataFrame) -> dict:
    summary = {}
    ele = list(data.loc[0])
    for i in ele:
        t = type(i)
        if t not in summary.keys():
            summary[t] = 1
        else:
            summary[t] += 1
    return summary

#find out features in each type
def detect_type(all_type: dict, data: pd.DataFrame) -> list:
    Type = {}
    for target_type in all_type.keys():
        ele = list(data.loc[3])
        features = []
        for index, e in enumerate(ele):
            if type(e) == target_type:
                features.append(data.columns[index])
            Type[target_type] = features
    return Type
