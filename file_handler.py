import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(path = 'diabetes_data.csv'):
    data = pd.read_csv(path)

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    target   = 'Outcome'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


