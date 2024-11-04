import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import io

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')
    return df.dropna()

# App Title
st.title("Car MPG Prediction App")

# Load data and display overview
df = load_data()
st.header("Data Overview")
st.write("Data head")
st.write(df.head())
st.write("#### Data Info")


buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


st.write("Data discription")
st.write(df.describe())



# Data Preprocessing
y = df['mpg']  # Target variable
X = df[['displacement', 'horsepower', 'weight', 'acceleration']]  # Features

# Scaling Data
ss = StandardScaler()
X_scaled = ss.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, random_state=2529)

# Linear Regression
st.write("### Linear Regression Model")
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model Coefficients
st.write("Intercept:", lr.intercept_)
st.write("Coefficients:", lr.coef_)

# Predict Test Data and evaluate
y_pred_lr = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_lr)
mape = mean_absolute_percentage_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

st.write("Mean Absolute Error:", mae)
st.write("Mean Absolute Percentage Error:", mape)
st.write("R² Score:", r2)

# Polynomial Regression
st.write("### Polynomial Regression Model")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lr.fit(X_train_poly, y_train)
y_pred_poly = lr.predict(X_test_poly)

# Evaluate Polynomial Regression
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mape_poly = mean_absolute_percentage_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

st.write("Polynomial Regression Mean Absolute Error:", mae_poly)
st.write("Polynomial Regression Mean Absolute Percentage Error:", mape_poly)
st.write("Polynomial Regression R² Score:", r2_poly)

# Random Forest Regressor with GridSearchCV
st.write("### Random Forest Regression with Grid Search")
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
st.write("Best parameters found:", grid_search.best_params_)
st.write("Random Forest R² Score:", r2_score(y_test, y_pred_rf))

# Visualization
st.write("### Visualization: Actual vs Predicted MPG")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred_lr, marker='o', label='Linear Regression Predictions', ax=ax,color='green')
sns.regplot(x=y_test, y=y_pred_poly, marker='x', color='red', label='Polynomial Regression Predictions', ax=ax)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.legend()
st.pyplot(fig)

# Random Forest Visualization
st.write("### Random Forest Actual vs Predicted Values")
fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs. Predicted Values (Random Forest)')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())
st.pyplot(fig_rf)
