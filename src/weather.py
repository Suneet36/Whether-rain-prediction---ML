# STEP 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# STEP 2: Load the dataset
file_path = r"C:\Users\Suneet Deshpande\Desktop\Suneet's website\seattle-weather.csv"
data = pd.read_csv(file_path)

# STEP 3: Display basic information
print(data.head())
print(data.info())

# STEP 4: Convert target column (weather) to binary
# Rain = 1, No Rain = 0
data['RainTomorrow'] = data['weather'].apply(
    lambda x: 1 if x == 'rain' else 0
)

# STEP 5: Select features and target
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['RainTomorrow']

# STEP 6: Handle missing values (if any)
X = X.fillna(X.mean())

# STEP 7: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 8: Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# STEP 9: Make predictions
y_pred = model.predict(X_test)

# STEP 10: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# Feature importance
importances = model.feature_importances_
features = X.columns
