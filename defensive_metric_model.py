# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Data Collection - Dummy Data Generation
def generate_dummy_data():
    np.random.seed(0) # For reproducibility
    data = {
        'Blocks': np.random.randint(0, 3, 100),
        'Steals': np.random.randint(0, 5, 100),
        'Defensive_Rebounds': np.random.randint(3, 10, 100),
        'Defensive_Win_Shares': np.random.random(100) * 5,
        'DBPM': np.random.random(100) * 10 - 5,
        'DRtg': np.random.randint(95, 115, 100),
        'Deflections': np.random.randint(0, 5, 100),
        'Loose_Balls_Recovered': np.random.randint(0, 5, 100),
        'Contest_Rate': np.random.random(100) * 100, # Assuming percentage
        'OFG_at_Rim': np.random.random(100) * 100, # Opponent FG% at Rim, assuming percentage
        'DVI': np.random.random(100), # Assuming a normalized score between 0 and 1
        'Fouls': np.random.randint(0, 5, 100),
        'Defensive_Fouls_Drawn': np.random.randint(0, 3, 100),
        'On_Off_Court_Impact': np.random.random(100) * 10 - 5,
        'All_Defensive_Votes': np.random.randint(0, 100, 100) # Simulated vote counts
    }
    return pd.DataFrame(data)

df = generate_dummy_data()

# 3. Feature Engineering - Normalize and Scale Features
scaler = StandardScaler()
X = df.drop('All_Defensive_Votes', axis=1)
X_scaled = scaler.fit_transform(X)
y = df['All_Defensive_Votes']

# 4. Model Development
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear regression model for simplicity
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Displaying the coefficients as a rudimentary form of feature importance
features = X.columns
coefficients = model.coef_
for feature, coeff in zip(features, coefficients):
    print(f"{feature}: {coeff}")

# 6. Predictions Example
# This step is for applying the model to new data
# Here's how you might predict All-Defensive Team votes using a new set of features
example_features = np.array([X_test.iloc[0]]) # Example features from the test set
example_prediction = model.predict(example_features)
print(f"Predicted All-Defensive Team votes: {example_prediction[0]}")
