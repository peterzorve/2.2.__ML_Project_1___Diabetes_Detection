from file_handler import process_data
from sklearn.linear_model import LogisticRegression 


X_train, X_test, y_train, y_test = process_data('diabetes_data.csv')

""" Instantiate the Model """
model = LogisticRegression()


""" Train the Model """
model.fit(X_train, y_train)

""" Check the Accuracy """
accuracy = model.score(X_test, y_test)

""" Make Prediction """
prediction = model.predict(X_test)



# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt 
# plot_confusion_matrix(model, X_test, y_test, display_labels=['area', 'perimeter', 'compactness'])
# plot_confusion_matrix(model, X_test, y_test, display_labels=['length', 'width',   'asymmetry_coefficient'])
# plt.show()

print(accuracy)