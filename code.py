import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

from bokeh.io import show, output_notebook
from bokeh.models import LogColorMapper, ColorBar, FixedTicker
from bokeh.palettes import Viridis6, Greys256, Viridis256
from bokeh.plotting import figure

pd.options.display.float_format = '{:.2f}'.format

from sklearn.datasets import fetch_openml


mnist = fetch_openml("mnist_784")


X = mnist.data[0:7000]
y = mnist.target[0:7000].reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y, stratify = y)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(),param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)