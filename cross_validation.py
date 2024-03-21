from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import warnings
def cv_decision_tree(X_train, y_train):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        decisionTree_grid = DecisionTreeClassifier()
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10, 20],
        }

        grid_search = GridSearchCV(decisionTree_grid, param_grid, cv = 5)
        grid_search.fit(X_train, y_train)
        print("The best hyper-parameter combinations:", grid_search.best_params_)
        print("The corresponding accuracy:", grid_search.best_score_)

def cv_svm(X_train, y_train):
    param_grid = {'C': [0.01 , 0.1 , 1 , 10 , 100 , 1000, 2000, 3000] , 'gamma': [0, 0.00001, 0.0001 , 0.01 , 0.1], 'kernel': ['linear','rbf','poly']}
    #param_grid1 = {'kernel': ['linear','rbf','poly']}
    svc_cv = GridSearchCV(svm.SVC() , param_grid = param_grid , cv = 5)
    svc_cv.fit(X_train, y_train)

    print('Optimal C is:', svc_cv.best_params_['C'])
    print('Optimal gamma is:', svc_cv.best_params_['gamma'])
    #print('Optimal kernel is:', svc_cv.best_params_['kernel'])

def roc_logistic(y_test, is_cancel_proba):
    fpr, tpr, thresholds = roc_curve(y_test, is_cancel_proba)
    roc_auc_sklearn = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_sklearn))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic')
    plt.legend(loc="lower right")
    plt.show()

def cv_knn(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 20))}
    knn_cv = GridSearchCV(KNeighborsClassifier() , param_grid = param_grid , cv = 5)
    knn_cv.fit(X_train, y_train)
    print('Optimal k is:', knn_cv.best_params_['n_neighbors'])