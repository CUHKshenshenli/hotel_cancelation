from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def evaluate(predicted, y_test) -> None:
    accuracy = accuracy_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    pre = precision_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", pre)
    print("F1 score:", f1)
