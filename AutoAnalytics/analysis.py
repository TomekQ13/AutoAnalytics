import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import isnull
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class AutoAnalytics:
    def __init__(self, path, dependent_variable, train_test_column, sep=';'):
        '''
        This method reads the csv file and splits the datasets for testing and training sets
        based on missing values in the dependent variable column
        '''
        df = pd.read_csv('AutoAnalytics/' + path, sep=';')
        test_set = df[df[train_test_column] == 'test']
        train_set = df[df[train_test_column] == 'train']
        self.path = path
        self.train_test_column = train_test_column

        self.dependent_variable = dependent_variable
        self.train_set = train_set
        self.test_set = test_set

        scaler = StandardScaler()
        data = train_set.drop(columns=[self.dependent_variable, self.train_test_column])
        scaler.fit(data)
        self.scaler = scaler

        self.X_train = self.scaler.transform(data)
        self.y_train = self.train_set[self.dependent_variable].copy()

        self.X_test = self.test_set.drop(columns=[self.dependent_variable, self.train_test_column])
        
    def fit_models(self):
        linear_regression = LinearRegression()
        linear_regression.fit(self.X_train, self.y_train)

        random_forest = RandomForestRegressor()
        random_forest.fit(self.X_train, self.y_train)

        #fit some more models
        
        self.linear_regression = linear_regression
        self.random_forest = random_forest

    def make_prediction(self):
        self.pred_test = self.test_set.copy()
        self.pred_train = self.train_set.copy()

        train_pred = self.linear_regression.predict(self.X_train)
        test_pred = self.linear_regression.predict(self.X_test)

        self.pred_train['LinearRegression'] = np.round(train_pred, 2)
        self.pred_test['LinearRegression'] = np.round(test_pred, 2)

        train_pred = self.random_forest.predict(self.X_train)
        test_pred = self.random_forest.predict(self.X_test)

        self.pred_train['RandomForest'] = np.round(train_pred, 2)
        self.pred_test['RandomForest'] = np.round(test_pred, 2)

        return self.pred_train, self.pred_test

    def save_prediction(self):
        data = pd.concat([self.pred_train, self.pred_test])
        data.to_csv('AutoAnalytics/' + self.path, index=False)
        return data

