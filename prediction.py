
import numpy as np 
import joblib 


# load trained models 
model_lr   = joblib.load('trained_model_lr.joblib')
model_dtc  = joblib.load('trained_model_dtc.joblib')
model_lsvc = joblib.load('trained_model_lsvc.joblib')
model_svc  = joblib.load('trained_model_svc.joblib')
model_rfc  = joblib.load('trained_model_rfc.joblib')
model_gbc  = joblib.load('trained_model_gbc.joblib')


# Create A Dummy Data 
def new_data(x1, x2, x3, x4, x5, x6, x7, x8):
    return np.array([[x1, x2, x3, x4, x5, x6, x7, x8]])

data = new_data(1, 85, 66, 29, 0, 26.6, 0.351, 31)
data = new_data(6,148,72,35,0,33.6,0.627,50)


# Make prediction 
predict_lr   = model_lr.predict(data)
predict_dtc  = model_dtc.predict(data)
predict_lsvc = model_lsvc.predict(data)
predict_svc  = model_svc.predict(data)
predict_rfc  = model_rfc.predict(data)
predict_gbc  = model_gbc.predict(data)

def result(pred):
    if pred == 0:
        return 'Negative'
    if pred == 1:
        return 'Positive'


# Print the predicted results 
print()
print(f'LogisticRegression          :   {predict_lr[0]} - {result(predict_lr[0])}')
print(f'DecisionTreeClassifier      :   {predict_dtc[0]} - {result(predict_dtc[0])}')
print(f'LinearSVC                   :   {predict_lsvc[0]} - {result(predict_lsvc[0])}')
print(f'SVC                         :   {predict_svc[0]} - {result(predict_svc[0])}')
print(f'RandomForestClassifier      :   {predict_rfc[0]} - {result(predict_rfc[0])}')
print(f'GradientBoostingClassifier  :   {predict_gbc[0]} - {result(predict_gbc[0])}')
