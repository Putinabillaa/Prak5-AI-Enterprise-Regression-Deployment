import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('issues.csv')

# Get unique programming languages
unique_proglangs = df['name'].unique()

# Dictionary to store models for each programming language
models = {}

for proglang in unique_proglangs:
    # Filter dataframe for the current programming language
    df_filtered = df[df['name'] == proglang]

    # Check if there are sufficient data points for splitting
    if len(df_filtered) <= 1:
        print(f"Not enough data for {proglang}. Skipping...")
        continue

    # Extract features and target variable
    X = df_filtered[['year']]
    y = df_filtered['count']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    # Store the trained model in the dictionary
    models[proglang] = {'model': model_rf, 'scaler': scaler}

# Serialize models
for proglang, model_info in models.items():
    model_folder = os.path.join('models', proglang)
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{proglang}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)