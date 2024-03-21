from models import baseline, decision_tree, kernel_svm, logistic, knn, lasso, neural_network, neural_network_by_array
from feature_selection import selection_lasso, selection_pca, country_selection, basic_selection
from load_data import load, load_dataset
from country import selection_specific_countries
from reservation_status_date import reservation_information
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def model_compare_null(filename: str) -> dict:
    model_name = ['baseline', 'decision_tree', 'kernel_svm', 'logistic', 'knn', 'neural_network']
    model = [baseline(filename), decision_tree(filename), kernel_svm(filename, 0.01, 'rbf', 2000, 0.00001), 
             logistic(filename, 0.4), knn(filename, 3), neural_network(filename, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())]
    Result = {}
    for i, m in enumerate(model_name):
        if m == 'lasso':
            result, _ = model[i]
        else:
            result = model[i]
        Result[i] = result
    return Result

#dimension resuction by PCA and Lasso
def basic_dimension_reduction(filename: str, new_filename: str, method: str) -> None:
    #PCA
    X_train, y_train, X_test, y_test = load(filename)
    if method == 'pca':
        X_train_pca, X_test_pca = selection_pca(X_train, X_test, 0.95)
        df_train = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(y_train)], axis=1)
        df_test = pd.concat([pd.DataFrame(X_test_pca), pd.DataFrame(y_test)], axis=1)
        df = pd.concat([df_train, df_test], axis=0)
        df.to_csv(new_filename, index = False)
        neural_network_by_array(X_train_pca, y_train, X_test_pca, y_test, 64, 1)
    #Lasso
    elif method == 'lasso':
        data = load_dataset(filename)
        drop_feature = selection_lasso(filename, new_filename, 0.35)
        for i in drop_feature:
            data = data.drop(i, axis = 1)
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        X_train = train.drop('is_canceled', axis=1)
        y_train = train['is_canceled']
        X_test = test.drop('is_canceled', axis=1)
        y_test = test['is_canceled']
        neural_network_by_array(X_train, y_train, X_test, y_test, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())
        data.to_csv(new_filename)

def compare_reservation_status_date(filename: str, new_filename: str) -> None:
    for m in range(3):
        reservation_information(filename, new_filename, mode = m)
        neural_network(new_filename, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())

def feature_engineering(filename: str) -> None:
    new_filename = 'dataset/hotel_canceling'
    #Experiment 1
    print('Experiment 1: We test the performance of the two basic methods.')
    print('For PCA:')
    newfile = new_filename+'_pca.csv'
    basic_dimension_reduction(filename, newfile, 'pca')
    print('For Lasso:')
    newfile = new_filename+'_lasso.csv'
    basic_dimension_reduction(filename, newfile, 'lasso')
   
    #Experiment 2
    ###|---------------------------------------------------Important Notice-----------------------------------------------------|###
    ###|If you want to use word embedding, please change the OpenAI API key in country.py to yours or you may encounter an error|###
    ###|------------------------------------------------------------------------------------------------------------------------|###
    print('Experiment 2: We want to use some other information to take the place of country.')
    print('Due to the high running time, we only show the experiments that lead to the best result, which makes user easier to replicate our code.')
    print('If you want to test the result of all the experiments, you can go to the file feature_selection.py, country.py, reservation_status_date.py. You will find everything you need in these three files. ')
    print('After testing, we find that use clustering and word embedding has the highest recall.')
    newfile = new_filename+'_country_embedding.csv'
    #country_selection(filename, newfile, 'embedding')
    
    print('Besides, keep the three most important countries has the highest accuracy and precision.')
    newfile = new_filename+'_country_3.csv'
    selection_specific_countries(filename, newfile, 3)
    neural_network(newfile, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())

    #Experiment 3
    #Here we use the dataset that replaces country by the result of clustering to do further feature engineering

    ###|---------------------------------------------------Important Notice-----------------------------------------------------|###
    ###|If you want to use word embedding, please change the OpenAI API key in country.py to yours or you may encounter an error|###
    ###|------------------------------------------------------------------------------------------------------------------------|###
    #If you do not have an OpenAI API Key, please change it into file_after_country = new_filename+'_country_3.csv'
    #file_after_country = new_filename+'_country_embedding.csv'
    file_after_country = new_filename+'_country_3.csv'

    print('Experiment 3: We want to deal with the reservation_status_date.')
    print('After testing, we find that split reservation_status_date into year, month, day has the best performance.')
    newfile = new_filename+'_cleaned.csv'
    reservation_information(file_after_country, newfile, 2)
    neural_network(newfile, 64, 1, tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam())

def main():
    print('First, we drop the features that will lead to information leak.')
    basic_selection('dataset/hotel_canceling.csv', 'dataset/hotel_canceling_null.csv')
    print('Then we want to test the performance of different models on this dataset.')#
    model_compare_null('dataset/hotel_canceling_null.csv')
    #Feature engineering
    print('Although the performance of neural network in this dataset is very good, we want to modify it to make it better.')
    print('We want to reduce the dimensionality.')
    feature_engineering('dataset/hotel_canceling_null.csv')

###|---------------------------------------------------Important Notice-----------------------------------------------------|###
###|If you want to use word embedding, please change the OpenAI API key in country.py to yours or you may encounter an error|###
###|------------------------------------------------------------------------------------------------------------------------|###
main()
