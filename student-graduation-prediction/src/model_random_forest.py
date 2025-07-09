
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_evaluate_rf(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'name': 'Random Forest', 'y_test': y_test, 'y_pred': y_pred}
