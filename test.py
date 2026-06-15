import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data cleaning

df = pd.read_csv("Autism_Data.csv")
df.head()
df.info()

# Check how many ? values exist
print((df["age"] == "?").sum())

# Replace ? with NaN
df["age"] = df["age"].replace("?", np.nan)

# Convert to numeric
df["age"] = pd.to_numeric(df["age"])

# Fill missing values with median
df["age"] = df["age"].fillna(df["age"].median())

# Verify
print(df["age"].dtype)
print(df["age"].isnull().sum())
print(df["age"].head())

print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())
print("Shape:", df.shape)

#target value 
print(df["Class/ASD"].value_counts())

#columns conversion
for col in df.select_dtypes(include="object"):
    print("\n", col)
    print(df[col].unique())

#ethnicity column cleaning
print((df["ethnicity"] == "?").sum())
df["ethnicity"] = df["ethnicity"].replace("?", "Unknown")
print((df["ethnicity"] == "?").sum())

#realtionship column cleaning
print((df["relation"] == "?").sum())
df["relation"] = df["relation"].replace("?", "Unknown")
print((df["relation"] == "?").sum())
 #AGE DISC column cleaning
print(df["age_desc"].nunique())
#WE HV TO DROP CUZ IT HAS ONLY 1 UNIQUE VALUE
df.drop("age_desc", axis=1, inplace=True)

#CHECKNG FOR ANY OTHER ? VALUES
print((df == "?").sum())

print(df.columns)

#CONVERTING CHAR TO NUMBERS
print(df.select_dtypes(include="object").columns)

# label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
import joblib

encoders = {}

categorical_cols = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
    "Class/ASD"
]

for col in categorical_cols:
    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")

print(encoders.keys())

#FEATURE EXTRACTION
X = df.drop(["Class/ASD", "result"], axis=1)
y = df["Class/ASD"]

print(X.shape)
print(y.shape)

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

#MODEL TRAINING 
# RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Model trained successfully!")

#to make predictions
y_pred = rf.predict(X_test)
print(y_pred[:10])

#evaluate model 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
#accuracy is 0.97 for random forest

#svm classifier
#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#TRAINING SVM
from sklearn.svm import SVC
SVM = SVC(kernel="rbf", random_state=42)
SVM.fit(X_train_scaled, y_train)
print("SVM model trained successfully!")

#ACCURACY
y_pred_svm = SVM.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
#accuracy is 0.98 for svm

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree model trained successfully!")
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test,y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
#accuracy id 0.90

#xgboost classifier
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train,y_train)
print("XGBoost model trained successfully!")
y_pred_xgb= xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)
#accuracy is 0.98 for xgboost

#hybrid model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('rf', rf),
    ('svm', svm),
    ('xgb', xgb)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stack_model.fit(X_train, y_train)
print("Stacking model trained successfully!")

#training of stacking model
y_pred_stack = stack_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print("Stacking Model Accuracy:", accuracy_stack)
#accuracy is 0.99 for stacking model

#save the model 
import joblib

joblib.dump(stack_model, "hybrid_autism_model.pkl")
joblib.dump(scaler, "scaler.pkl")# its important to save the scaler as well for future use when we want to make predictions on new data

#confusion matrix for stacking model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_stack)
print("Confusion Matrix:\n", cm)
print(X.columns.tolist())

import joblib

encoders = joblib.load("encoders.pkl")

print(encoders.keys())

print(encoders["gender"].classes_)