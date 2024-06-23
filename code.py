# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""**Display the first few rows of the dataframe**"""

df = pd.read_csv("House Pricing.csv")

print("First few rows of the dataset:")
print(df.head())

"""**Discretize the "age" variable into three bins: 'Young', 'Middle-aged', and 'Old'.**"""

max = df['age'].max()
min = df['age'].min()

print(f"Maximum age value: {max}")
print(f"Minimum age value: {min}")

Bins = [min, 30, 70, max]
Labels = ['Young', 'Middle-aged', 'Old']

df['DiscretizeAge'] = pd.cut(df['age'], bins = Bins, labels = Labels, right=False)

print("\n Discretized age is:")
print(df[['age', 'DiscretizeAge']].head())

"""**Create a binary variable "is_charles_river" based on the "chas" column.**"""

df['is_charles_river'] = df['chas'].apply(lambda x: 1 if x == 1 else 0)

print("Dataset with the new binary variable 'is_charles_river'")
print(df[['chas', 'is_charles_river']].head())

"""**Detect and remove outliers for each numerical column in the dataset using the Interquartile Range (IQR) method.**"""

def outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    Lower_bound = Q1 - 1.5 * IQR
    Upper_bound = Q3 + 1.5 * IQR

    CleanedData = column[(column >= Lower_bound) & (column <= Upper_bound)]
    return CleanedData

numerical_columns = df.select_dtypes(include='number').columns
WithOutliers = []
WithoutOutliers = []

for i in numerical_columns:
  WithOutliers.append(df[i].copy())
  df[i] = outliers(df[i])
  WithoutOutliers.append(df[i])

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.boxplot(WithOutliers, labels=numerical_columns)
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.boxplot(WithoutOutliers, labels=numerical_columns)
plt.title('Data after Outlier Removal')

plt.show()

"""**Identify and remove noisy data points from the dataset using z-score**"""

def NoisyData(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    noisy_data = z_scores > threshold
    removenoise = df[~noisy_data.any(axis=1)]
    return removenoise

numericalcol = df.select_dtypes(include='number')

removenoise = NoisyData(numericalcol)

print(f"Number of rows before removing noisy data: {len(df)}")
print(f"Number of rows after removing noisy data: {len(removenoise)}")

print(removenoise.head())

"""**Apply smoothing to the "rm" column and create a new smoothed column.**"""

df.dropna(subset=['rm'], inplace=True)

NoOfBins = 10

Minrm = df['rm'].min()
Maxrm = df['rm'].max()
BinWidth = (Maxrm - Minrm) / NoOfBins
BinEdges = [Minrm + i * BinWidth for i in range(NoOfBins + 1)]

def BinMean(value):
    bin_index = int((value - Minrm) // BinWidth)
    BinMean = (BinEdges[bin_index] + BinEdges[bin_index + 1]) / 2
    return BinMean

df['rm_smoothed'] = df['rm'].apply(BinMean)

print(df[['rm', 'rm_smoothed']].head())

"""**Normalize the "tax" and "lstat" columns using Min-Max normalization.**"""

NormalizeColumns = ['tax', 'lstat']

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

for col in NormalizeColumns:
    df[col + '_normalized'] = min_max_normalize(df[col])

print("Normalized columns:")
print(df[['tax', 'tax_normalized', 'lstat', 'lstat_normalized']].head())

"""**Simple linear regression to predict the median value of "medv" based on the "rm" variable.**"""

df.dropna(subset=['medv'], inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

X = df[['rm']]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

slope = model.coef_[0]
intercept = model.intercept_
print("Regression Equation:")
print("medv =", slope, "* rm +", intercept)

print("\nMean Squared Error:", mse)
print("R-squared:", r2,"\n")

plt.scatter(X_test, y_test, color='lightblue')
plt.plot(X_test, y_pred, color='gray', linewidth=2)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Linear Regression: RM vs MEDV')
plt.show()

"""**Relationship between "medv" and "rm," interpretations based on the regression results.**

The linear regression and the slope indicates the relationship between 'rm' and 'medv'.
By analyzing, it suggests a positive relationship between the 'rm' and 'medv'. As the 'rm' increases, the 'medv' tends to increase as well.

Since, the MSE measures the average squared difference between actual and predicted values.A lower MSE indicates a better fit of the model to the data.

R2 measures the proportion of variance in the target variable (medv) that can be explained by the predictor variable (rm).
The line does not fit best, since R-squared is 0.36
"""