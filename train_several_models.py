
from file_handler import process_data

X_train, X_test, y_train, y_test = process_data('diabetes_data.csv')

# Import the Models 
from sklearn.linear_model   import LogisticRegression 
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import LinearSVC
from sklearn.svm            import SVC 
from sklearn.ensemble       import RandomForestClassifier 
from sklearn.ensemble       import GradientBoostingClassifier


# Instantiate the Models
model_lr   = LogisticRegression()
model_dtc  = DecisionTreeClassifier()
model_lsvc = LinearSVC()
model_svc  = SVC()
model_rfc  = RandomForestClassifier()
model_gbc  = GradientBoostingClassifier()


# Train the Models
model_lr.fit(X_train, y_train)
model_dtc.fit(X_train, y_train)
model_lsvc.fit(X_train, y_train)
model_svc.fit(X_train, y_train)
model_rfc.fit(X_train, y_train)
model_gbc.fit(X_train, y_train)


# Save the Models
import joblib 
filename_lr   = 'trained_model_lr.joblib'
filename_dtc  = 'trained_model_dtc.joblib'
filename_lsvc = 'trained_model_lsvc.joblib'
filename_svc  = 'trained_model_svc.joblib'
filename_rfc  = 'trained_model_rfc.joblib'
filename_gbc  = 'trained_model_gbc.joblib'



joblib.dump(model_lr, filename = 'trained_model_lr.joblib') 
joblib.dump(model_dtc, filename = 'trained_model_dtc.joblib') 
joblib.dump(model_lsvc, filename = 'trained_model_lsvc.joblib') 
joblib.dump(model_svc, filename = 'trained_model_svc.joblib') 
joblib.dump(model_rfc, filename = 'trained_model_rfc.joblib') 
joblib.dump(model_gbc, filename = 'trained_model_gbc.joblib') 



# Check the accuracy of the Models 
accuracy_lr   = model_lr.score(X_test, y_test)
accuracy_dtc  = model_dtc.score(X_test, y_test)
accuracy_lsvc = model_lsvc.score(X_test, y_test)
accuracy_svc  = model_svc.score(X_test, y_test)
accuracy_rfc  = model_rfc.score(X_test, y_test)
accuracy_gbc  = model_gbc.score(X_test, y_test)



# Print the Accuracy Results 
print()
print(f'LogisticRegression          :   {round(accuracy_lr   * 100, 2)} %')
print(f'DecisionTreeClassifier      :   {round(accuracy_dtc  * 100, 2)} %')
print(f'LinearSVC                   :   {round(accuracy_lsvc * 100, 2)} %')
print(f'SVC                         :   {round(accuracy_svc  * 100, 2)} %')
print(f'RandomForestClassifier      :   {round(accuracy_rfc  * 100, 2)} %')
print(f'GradientBoostingClassifier  :   {round(accuracy_gbc  * 100, 2)} %')

