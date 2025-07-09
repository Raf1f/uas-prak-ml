
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def train_evaluate_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', C=1, gamma='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print("SVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'name': 'SVM', 'y_test': y_test, 'y_pred': y_pred}
