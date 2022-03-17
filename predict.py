import sys
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report

from feature_engineering import fill_missing_values, drop_column, transform_altitude

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 


#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]


# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)







y_test_predicted = loaded_model.predict(X_test)

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_test_predicted)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_predicted)))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_predicted)))


'''print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

#feature eng on test data
print("Feature engineering")
X_test = transform_altitude(X_test)
X_test = drop_column(X_test, col_name='Unnamed: 0')
X_test = drop_column(X_test, col_name='Quakers')
X_test = fill_missing_values(X_test)

y_test_pred = loaded_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print (f"MSE on test is: {mse_test}")
'''