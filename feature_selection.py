import pandas as pd
import numpy as np
import time
import tensorflow as tf
from datetime import datetime

from load_data import load, load_dataset
from models import lasso, neural_network
from country import selection_specific_countries, selection_information, country_clustering
from preprocessing import detect_missed, missing_rate

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV, Perceptron
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
'''In this function, we first drop the features with high missing rate.
   Then we drop all NAN rows.
   Next we drop the features that will lead to information leak
   We also change the month from str to int.
   As how to detect these features, please go to the 'feature_analysis.py' file.'''

'''hotel_canceling: the original data
   hotel_canceling_null: data without information leak and missing value
   hotel_canceling_country: data after dealing with 'country' 
   hotel_canceling_cleaned: data after dealing with 'country' and 'reseervation_status_date' '''
def basic_selection(filename: str, new_filename: str) -> None:
    data = pd.read_csv(filename)
    #drop NAN
    data = data.drop('company', axis = 1)
    data = data.dropna().reset_index(drop=True)
    #information leak
    data = data.drop('assigned_room_type', axis = 1)
    data = data.drop('reservation_status', axis = 1)
    #month
    data['arrival_month'] = list(map(lambda x: datetime.strptime(x, '%B').month, data['arrival_date_month']))
    data = data.drop('arrival_date_month', axis = 1)
    data.to_csv(new_filename, index=False)
    print(f"The dataset without features that will cause information leak has been saved to {new_filename}")

def output_result(result: dict) -> None:
    for metrics, value in result.items():
        print(metrics, value)

def evaluate(predicted, y_test) -> dict:
    accuracy = accuracy_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    pre = precision_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    result = {
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": pre,
        "F1 score": f1
    }
    return result

def selection_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, variance: float) -> pd.DataFrame:
    #Standarization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #PCA
    pca = PCA()
    pca.fit(X_train)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.where(cumsum >= variance)[0][0] + 1
    X_train_pca = X_train[:, :d]
    X_test_pca = X_test[:, :d]

    return X_train_pca, X_test_pca

def selection_lasso(filename: str, new_filename: str, threshold: float) -> list:
    data = load_dataset(filename).iloc[:,1:]
    _, lasso_coef = lasso(filename, threshold)
    drop_feature = data.columns[lasso_coef == 0]
    for i in drop_feature:
        data = data.drop(i, axis = 1)
    data.to_csv(new_filename, index = False) 
    return drop_feature

#method: specific, information, embedding
def country_selection(filename: str, new_filename: str, method: str) -> dict:
    data = pd.read_csv(filename)
    if method == 'specific':
        for num in range(4):
            selection_specific_countries(filename, new_filename, num)
            print(f'If we keep the most important {num+1} countries:')
            result = neural_network(new_filename, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())
    elif method == 'information':
        for mode in range(3):
            selection_information(filename, new_filename, mode)
            print(f'If we use mode {mode}:')
            result = neural_network(new_filename, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())
    elif method == 'embedding':
        country_clustering(filename, new_filename, 2, 5, 'rbf')
        print(f'If we use word embedding and clustering:')
        result = neural_network(new_filename, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())
    return result
