import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Target column
target = "is_fraud"

# Drop unnecessary columns
drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num',
    'merchant', 'category', 'first', 'last',
    'gender', 'street', 'city', 'state',
    'zip', 'lat', 'long', 'city_pop',
    'job', 'dob', 'trans_num', 'unix_time',
    'merch_lat', 'merch_long'
]

train_data = train_data.drop(columns=drop_cols)
test_data = test_data.drop(columns=drop_cols)

# Features & target
X_train = train_data.drop(target, axis=1)
y_train = train_data[target]

X_test = test_data.drop(target, axis=1)
y_test = test_data[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Supervised model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Unsupervised model
iso = IsolationForest(
    contamination=0.02,
    n_estimators=50,
    random_state=42
)

iso.fit(X_train_scaled)

# Save models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(iso, "iso_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Test model
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

print("Models trained and saved successfully!")
