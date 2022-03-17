

# in terminal
# python predict.py models/SVC.sav data/X_test.csv data/y_test.csv



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from feature_engineering import fill_missing_values, drop_column, transform_altitude
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report

# Set random seed 
RSEED = 42

# load train data
Train_data = pd.read_csv('data/Train.csv')

#cleaning data and preparing

labels = [0,Train_data['total_cost'].median(), Train_data['total_cost'].max()]
Train_data['total_cost_binned'] = pd.cut(Train_data['total_cost'], bins=labels)


le = LabelEncoder()

Train_data['total_cost_binned'] = LabelEncoder().fit_transform(Train_data['total_cost_binned'])


X = Train_data.drop(['total_cost', 'total_cost_binned', 'ID'], axis=1)
y = Train_data['total_cost_binned']


# splittin into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Creating list for categorical/numerical predictors/features

cat_features = list(Train_data.columns[Train_data.dtypes==object])
cat_features.remove('ID')
cat_features.remove('package_accomodation')

num_features = list(Train_data.columns[Train_data.dtypes!=object])
num_features.remove('total_cost')
num_features.remove('total_cost_binned')


#creating pipelines

# Pipline for numerical features
num_pipeline = Pipeline([
    ('imputer_num', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Pipeline for categorical features 
cat_pipeline = Pipeline([
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
    ('1hot', OneHotEncoder(handle_unknown='ignore'))
])

# Complete pipeline for numerical and categorical features
# 'ColumnTranformer' applies transformers (num_pipeline/ cat_pipeline)
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

from sklearn.svm import SVC

# Building a full pipeline with our preprocessor and a LogisticRegression Classifier
pipe_svc = Pipeline([
    ('preprocessor', preprocessor),
    ('svc', SVC ())
])


# RandomizedSearchCV
rand_list = {"svc__C": np.arange(2, 6, 2),
             "svc__gamma": np.arange(0.1, 0.5, 0.2),
             'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
              
rand_search = RandomizedSearchCV(pipe_svc, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = 3, random_state = 2017, scoring='recall') 
rand_search.fit(X_train, y_train)





# Show best parameters
print('Best score:\n{:.2f}'.format(rand_search.best_score_))
print("Best parameters:\n{}".format(rand_search.best_params_))

best_model = rand_search.best_estimator_
best_model

#saving the model
print("Saving model in the model folder")
filename = 'models/SVC.sav'
pickle.dump(best_model, open(filename, 'wb'))

# Calculating the accuracy, recall and precision for the test set with the optimized model
y_test_predicted = best_model.predict(X_test)

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_test_predicted)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_predicted)))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_predicted)))