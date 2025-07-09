
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_evaluate_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred))
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'name': 'KNN', 'y_test': y_test, 'y_pred': y_pred}
