import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib


X = pd.DataFrame(load_iris()['data'], columns=load_iris()['feature_names'])
y = load_iris()['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=11)

def model(X, y):
    X = X.copy()
    y = y.copy()
    pipeline = Pipeline(steps=[['scaler', MinMaxScaler()],
                               ['feature_selection', SelectKBest(score_func=mutual_info_classif)],
                               ['classifier', LogisticRegression(random_state=11, max_iter=1000)]])
    
    param_grid = {'feature_selection__k': range(1, X.shape[1]),
                  'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=-1,
                               cv=3)
    
    grid_search.fit(X, y)
    
    return grid_search


pipeline = Pipeline(steps=[['scaler', MinMaxScaler()],
                           ['feature_selection', SelectKBest(score_func=mutual_info_classif,
                                                             k=3)],
                           ['classifier', LogisticRegression(random_state=11,
                                                             max_iter=1000,
                                                             C=1000)]])

#Refitting the pipeline to the data to find features selected
pipeline.fit(X_train, y_train)
feature_selection = (pipeline['feature_selection']).scores_
feature_scores = {key: value for key,value in zip(X_train.columns, (np.round(pipeline['feature_selection'].scores_,2)))}

#Fitting the final model from GridSearchCV
iris_model = model(X_train, y_train)
joblib.dump(iris_model, 'iris_model.pkl')

print(f'Best params: {iris_model.best_params_}\nBest score: {iris_model.best_score_}\nFeatures scores: {feature_scores}')
view raw1.py