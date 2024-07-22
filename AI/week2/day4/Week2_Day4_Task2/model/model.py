import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
# Load dataset

data = pd.read_csv(r'data\insurance.csv')
data = data.dropna()
data = data.drop_duplicates()
# Convert categorical columns to numerical
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Split the data into features and target
X = data.drop('charges', axis=1)
y = data['charges']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, r'model\insurance_charges_model.pkl')
