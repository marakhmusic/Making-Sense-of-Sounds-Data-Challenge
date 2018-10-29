import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# read in the advertising dataset
data = pd.read_csv('Advertising.csv', index_col=0)
# create a Python list of three feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the DataFrame (X)
X = data[feature_cols]

# select the Sales column as the response (y)
y = data.Sales

# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores)

# fix the sign of MSE scores
mse_scores = -scores
print(mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)


# calculate the average RMSE
print(rmse_scores.mean())


# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())
