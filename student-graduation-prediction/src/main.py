
from preprocessing import load_and_preprocess
from model_knn import train_evaluate_knn
from model_random_forest import train_evaluate_rf
from model_svm import train_evaluate_svm
from evaluate import compare_models

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess()

# Train and evaluate models
knn_results = train_evaluate_knn(X_train, X_test, y_train, y_test)
rf_results = train_evaluate_rf(X_train, X_test, y_train, y_test)
svm_results = train_evaluate_svm(X_train, X_test, y_train, y_test)

# Compare models
compare_models(knn_results, rf_results, svm_results)
