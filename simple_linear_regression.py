import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib's pyplot module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOADING DATASET
stud_scores = pd.read_csv('students_scores.csv')

# CREATING FEATURE MATRIX AND RESPONSE VECTOR
X = stud_scores[['Hours']].values  # feature matrix
y = stud_scores['Scores'].values   # response vector

# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# FITTING LINEAR REGRESSION MODEL / TRAINING
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTION OF TEST RESULT
y_pred = regressor.predict(X_test)

# EVALUATING MODEL METRICS
print('MAE:', mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Model Score (R-squared):', r2_score(y_test, y_pred))

# Plotting
sns.set(style="whitegrid")  # Set the style of the seaborn plot

# Plotting the regression line
sns.regplot(x='Hours', y='Scores', data=stud_scores, ci=None, 
            scatter_kws={'s': 100, 'facecolor': 'red'}, line_kws={'color': 'blue'})

# Adding labels and title
plt.title('Linear Regression - Hours Studied vs. Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')

# Display the plot
plt.show()
