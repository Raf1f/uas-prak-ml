
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load dataset
    data = pd.read_csv('../data/student-por.csv')
    
    # Define target variable
    data['pass'] = data['G3'] >= 10
    y = data['pass'].astype(int)
    X = data.drop(columns=['G3', 'pass'])
    
    # One-hot encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
