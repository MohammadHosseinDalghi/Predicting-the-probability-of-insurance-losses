# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import dataset
df = pd.read_csv('Insurance-claims-data.csv')

# balance the dataset using oversampling
majority = df[df.claim_status == 0]
minority = df[df.claim_status == 1]

minority_resampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

oversampled_data = pd.concat([majority, minority_resampled])

# Modeling

# it’s notable that policy_id has a very high importance, which might not be intuitively relevant for prediction, so we drop it.
oversampled_data = oversampled_data.drop('policy_id', axis=1)

X_oversampled = oversampled_data.drop('claim_status', axis=1)
y_oversampled = oversampled_data['claim_status']

X_oversampled_encoded = X_oversampled.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

X_train, X_test, y_train, y_test = train_test_split(
    X_oversampled_encoded,
    y_oversampled,
    random_state=42,
    test_size=0.3
)

rf_model_resampled = RandomForestClassifier(random_state=42)
rf_model_resampled.fit(X_train, y_train)

y_pred = rf_model_resampled.predict(X_test)

print(classification_report(y_test, y_pred))