import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
startups_df = pd.read_csv('50_Startups.csv')

# Separate independent and dependent variables
X = startups_df.iloc[:, :-1]    # Independent variables
y = startups_df.iloc[:, -1]     # Dependent variable

# Encode categorical data using OneHotEncoder
enc = OneHotEncoder(drop='first') # Drop the first dummy variable (K-1)
enc_df = pd.DataFrame(enc.fit_transform(X[['State']]).toarray())
enc_df.columns = ['Florida', 'New York']
X = X.join(enc_df)
X = X.drop('State', axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))

# Scatter plot
plt.scatter(y_test, y_pred, color='blue')
plt.title('Actual vs Predicted Profits (Multiple Linear Regression)')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')

# Add line representing perfect predictions
min_value = min(min(y_test), min(y_pred))
max_value = max(max(y_test), max(y_pred))
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')

plt.show()

# Evaluate model
score = r2_score(y_test, y_pred)
print('Model Score (R-squared):', score)
