import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RidgeLogisticRegression():
    def __init__(self, lam, iteration):
        self.X = 0
        self.y = 0
        self.beta = 0
        self.cost_record = None
        self.lam = lam
        self.iterations = iteration

    def sigmoid(self):
        return 1/(1+np.exp(-(self.X @ self.beta)))

    def cost(self, regularization = True):
        m = len(self.y)
        h = self.sigmoid()
        epsilon = 1e-5
        
        if regularization == True:
            return np.sum(self.y*self.X*self.beta - np.log(1 + np.exp(self.X @ self.beta))) - ((self.lam/(2)) * self.beta.T @ self.beta)
        else:
            return np.sum(self.y*self.X*self.beta - np.log(1 + np.exp(self.X @ self.beta)))

    def gradient(self, regularization = True):
        m = len(self.y)
        if regularization == True:
            return (self.X.T @ (self.y - self.sigmoid())) - self.lam*self.beta
        else:
            return (self.X.T @ (self.sigmoid() - self.y))

    def hessian(self, regularization = True):
        m = len(self.y)
        S = np.diag(np.ravel(np.exp(self.X@self.beta)/np.power(1+np.exp(self.X@self.beta),2)))
        if regularization == True:
            return (-1 *self.X.T @ S @ self.X) - (self.lam * np.identity(X.shape[1]))
        else:
            return (-1 *self.X.T @ S @ self.X)

    def newton_method(self, regularization):
        m = len(self.y)
        self.cost_record = np.zeros((self.iterations,1))
        counter = 0
        
        for i in range(self.iterations):
            self.beta -= np.linalg.inv(self.hessian(regularization))*self.gradient(regularization)
            self.cost_record[counter] = self.cost()
            counter+=1

    def gradient_descent(self, learing_rate, iterations):
        m = len(self.y)
        cost_record = np.zeros((self.iterations,1))

        for i in range(self.iterations):
            self.beta = self.beta - (learning_rate * self.gradient()/len(self.y))
            self.cost_record[i] = self.cost()
    
    
    def fit(self, X, y, learning_rate = 0.03, method = 'Newton_Method', regularization = True):
        self.X = X
        self.y = y
        n = np.size(X,1)
        self.beta = np.matrix(np.zeros(X_train.shape[1])).T

        if method == 'Gradident_Descent':
            self.gradient_descent(X, y, beta, learning_rate, self.iterations)
        elif method == 'Newton_Method':
            self.newton_method(regularization)
        else:
            print('Unsupported Method')

    def predict(self, X):
        return np.round(1/(1+np.exp(-(X @ self.beta))))
    
    def get_cost_record(self):
        return self.cost_record
    
    def get_beta(self):
        return self.beta

# Testing Case
columns = ['class'] + ["V" + str(i) for i in np.arange(1,7130)]
train_duke = pd.read_csv('Duke_train.txt', names = columns)
X_train = train_duke[["V" + str(i) for i in np.arange(1,7130)]]
y_train = train_duke['class']
X = np.array(X_train)
y = np.array(y_train).reshape(-1,1)
model = RidgeLogisticRegression(lam = 0.1, iteration = 10)
model.fit(X, y)
test_duke = pd.read_csv('Duke_test.txt', names = columns)
X_test = test_duke[["V" + str(i) for i in np.arange(1,7130)]]
y_test = test_duke['class']
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)
model.predict(X_test)