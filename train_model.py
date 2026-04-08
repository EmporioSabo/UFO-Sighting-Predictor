"""Train UFO sighting country prediction model."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
ufos = pd.read_csv('../../session-8/data/ufos.csv')
print(f"Raw data shape: {ufos.shape}")
print(ufos.head())

# Prepare features
ufos = pd.DataFrame({
    'Seconds': ufos['duration (seconds)'],
    'Country': ufos['country'],
    'Latitude': ufos['latitude'],
    'Longitude': ufos['longitude']
})

print(f"\nCountries: {ufos.Country.unique()}")

# Clean data
ufos.dropna(inplace=True)
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
print(f"\nCleaned data shape: {ufos.shape}")

# Encode countries
le = LabelEncoder()
ufos['Country'] = le.fit_transform(ufos['Country'])
print(f"Country mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split
X = ufos[['Seconds', 'Latitude', 'Longitude']]
y = ufos['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"\n{classification_report(y_test, predictions)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

# Save model
pickle.dump(model, open('models/ufo-model.pkl', 'wb'))
print("\nModel saved to models/ufo-model.pkl")

# Test prediction
test_pred = model.predict([[50, 44, -12]])
countries = ["Australia", "Canada", "Germany", "UK", "US"]
print(f"Test prediction for [50s, lat=44, lon=-12]: {countries[test_pred[0]]}")
