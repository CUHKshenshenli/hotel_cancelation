import pandas as pd
import numpy as np
import warnings
import time
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from load_data import load

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

def output_result(result: dict) -> None:
    for metrics, value in result.items():
        print(metrics, value)

def baseline(filename: str) -> float:
    X_train, y_train, X_test, y_test = load(filename)
    print('Below are the accuracy for baseline:')
    result = max(1-np.sum(y_test)/y_test.shape[0], np.sum(y_test)/y_test.shape[0])
    print(result)
    return result

def decision_tree(filename: str, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1) -> dict:
    X_train, y_train, X_test, y_test = load(filename)

    start_time = time.time()
    decisionTree = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 42)
    decisionTree.fit(X_train, y_train)
    predicted = decisionTree.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    result = evaluate(predicted, y_test)
    result['Elapsed time'] = elapsed_time
    print('Below are the results for Decision Tree:')
    output_result(result)
    return result

#As you know, the time complexity of SVM is very large, so we may only use a small part of the original data
#data_size is the percentage of data you want to use
#Here, we only support three kinds of kernels: linear, rbf and poly
def kernel_svm(filename: str, data_size: float, kernel: str, C = 100, gamma = 'scale', degree = 3) -> dict:
    X_train, y_train, X_test, y_test = load(filename)
    num_traing = int(X_train.shape[0]*data_size)
    num_test = int(X_test.shape[0]*data_size)

    X_train_small = pd.DataFrame(X_train).head(num_traing)
    y_train_small = y_train.head(num_traing)
    X_test_small = pd.DataFrame(X_test).head(num_test)
    y_test_small = y_test.head(num_test)

    if kernel == 'rbf':
        start_time = time.time()
        SVM = svm.SVC(kernel = kernel, C=C, gamma = gamma)
        SVM.fit(X_train_small, y_train_small)
        predicted = SVM.predict(X_test_small)
        end_time = time.time()
        elapsed_time = end_time - start_time

    elif kernel == 'poly':
        start_time = time.time()
        SVM = svm.SVC(kernel = kernel, degree = degree)
        SVM.fit(X_train_small, y_train_small)
        predicted = SVM.predict(X_test_small)
        end_time = time.time()
        elapsed_time = end_time - start_time

    result = evaluate(predicted, y_test_small)
    result['Elapsed time'] = elapsed_time
    print('Below are the results for SVM:')
    output_result(result)
    return result

def logistic(filename: str, threshold: float) -> dict:
    X_train, y_train, X_test, y_test = load(filename)

    start_time = time.time()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    is_cancel_proba = [y_pred_proba[i][1] for i in range(y_pred_proba.shape[0])]
    predicted = [i > threshold for i in is_cancel_proba]
    end_time = time.time()
    elapsed_time = end_time - start_time
    result = evaluate(predicted, y_test)
    result['Elapsed time'] = elapsed_time
    print('Below are the results for logistic:')
    output_result(result)
    return result

def knn(filename: str, k: int) -> dict:
    X_train, y_train, X_test, y_test = load(filename)

    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    result = evaluate(predicted, y_test)
    result['Elapsed time'] = elapsed_time
    print('Below are the results for KNN:')
    output_result(result)
    return result

def lasso(filename: str, threshold: float):
    X_train, y_train, X_test, y_test = load(filename)

    np.random.seed(42)
    start_time = time.time()
    lasso_cv = LassoCV(cv = 10)
    lasso_cv.fit(X_train, y_train)
    lasso_coef = lasso_cv.coef_
    y_pred = lasso_cv.predict(X_test)
    end_time = time.time()
    predicted = y_pred > threshold
    elapsed_time = end_time - start_time
    result = evaluate(predicted, y_test)
    result['Elapsed time'] = elapsed_time
    print('Below are the results for LASSO:')
    output_result(result)
    return result, lasso_coef

def neural_network(filename: str, batch_size: int, epochs: int, loss = tf.keras.losses.BinaryCrossentropy(), optimizer = tf.keras.optimizers.Adam()) -> dict:
    X_train, y_train, X_test, y_test = load(filename)

    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.float32)
    X_te = X_test.astype(np.float32)
    y_te = y_test.astype(np.float32)

    X_tr = tf.convert_to_tensor(X_tr)
    y_tr = tf.convert_to_tensor(y_tr)
    X_te = tf.convert_to_tensor(X_te)
    y_te = tf.convert_to_tensor(y_te)

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    loss_object = loss
    optimizer = optimizer

    callbacks = [
        keras.callbacks.ModelCheckpoint("callback/save_at_{epoch}.keras"),
    ]

    model.compile(
        optimizer = optimizer,
        loss = loss_object,
        metrics = ["accuracy"],
    )
    model.fit(
        x = X_tr, y = y_tr,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_split = 0.2,
        verbose = 1
    )
    predicted = (model.predict(X_te))> 0.5
    result = evaluate(predicted, y_te)
    print('Below are the results for Neural Network:')
    output_result(result)
    return result

def neural_network_by_array(X_train, y_train, X_test, y_test, batch_size: int, epochs: int, loss = tf.keras.losses.BinaryCrossentropy(), optimizer = tf.keras.optimizers.Adam()) -> dict:
    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.float32)
    X_te = X_test.astype(np.float32)
    y_te = y_test.astype(np.float32)

    X_tr = tf.convert_to_tensor(X_tr)
    y_tr = tf.convert_to_tensor(y_tr)
    X_te = tf.convert_to_tensor(X_te)
    y_te = tf.convert_to_tensor(y_te)

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    loss_object = loss
    optimizer = optimizer

    callbacks = [
        keras.callbacks.ModelCheckpoint("callback/save_at_{epoch}.keras"),
    ]

    model.compile(
        optimizer = optimizer,
        loss = loss_object,
        metrics = ["accuracy"],
    )
    model.fit(
        x = X_tr, y = y_tr,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_split = 0.2,
        verbose = 1
    )
    predicted = (model.predict(X_te))> 0.5
    result = evaluate(predicted, y_te)
    print('Below are the results for Neural Network:')
    output_result(result)
    return result
