
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def compare_models(*models):
    for model in models:
        print(f"=== {model['name']} ===")
        ConfusionMatrixDisplay.from_predictions(model['y_test'], model['y_pred'])
        plt.title(f"Confusion Matrix - {model['name']}")
        plt.show()
