"""
Loan Approval ML Model — trains on CSV dataset and exposes prediction API.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

MODEL_PATH = "models/loan_model.joblib"
ENCODER_PATH = "models/label_encoders.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_names.joblib"

CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
NUMERICAL_COLS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
ALL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

FEATURE_IMPORTANCE_NAMES = {
    'Credit_History': 'Credit History',
    'ApplicantIncome': 'Applicant Income',
    'CoapplicantIncome': 'Co-Applicant Income',
    'LoanAmount': 'Loan Amount',
    'Loan_Amount_Term': 'Loan Term',
    'Education': 'Education Level',
    'Married': 'Marital Status',
    'Self_Employed': 'Employment Type',
    'Gender': 'Gender',
    'Dependents': 'Number of Dependents',
    'Property_Area': 'Property Location'
}


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Loan_ID'], errors='ignore')

    # Fill missing values
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def train_model(csv_path: str = "data/loan_approval.csv"):
    os.makedirs("models", exist_ok=True)
    df = load_and_preprocess(csv_path)

    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    if 'Loan_Status' not in df.columns:
        raise ValueError("Dataset must contain 'Loan_Status' column")

    le_target = LabelEncoder()
    df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    label_encoders['Loan_Status'] = le_target

    X = df[ALL_FEATURES]
    y = df['Loan_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ALL_FEATURES, FEATURES_PATH)

    print(f"Model saved to {MODEL_PATH}")
    return model, label_encoders, scaler, accuracy


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, label_encoders, scaler


def predict_loan(applicant_data: dict):
    model, label_encoders, scaler = load_model()
    if model is None:
        return {"error": "Model not trained. Run train_model() first."}

    try:
        df = pd.DataFrame([applicant_data])

        # Fill defaults
        defaults = {
            'Gender': 'Male', 'Married': 'No', 'Dependents': '0',
            'Education': 'Graduate', 'Self_Employed': 'No',
            'ApplicantIncome': 3000, 'CoapplicantIncome': 0,
            'LoanAmount': 100, 'Loan_Amount_Term': 360, 'Credit_History': 1.0,
            'Property_Area': 'Urban'
        }
        for k, v in defaults.items():
            if k not in df.columns or pd.isna(df[k].iloc[0]):
                df[k] = v

        for col in CATEGORICAL_COLS:
            le = label_encoders.get(col)
            if le:
                val = str(df[col].iloc[0])
                if val not in le.classes_:
                    val = le.classes_[0]
                    df[col] = val
                df[col] = le.transform([val])[0]

        X = df[ALL_FEATURES].astype(float)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        le_target = label_encoders['Loan_Status']
        result_label = le_target.inverse_transform([prediction])[0]
        approved = result_label == 'Y'

        # Feature importances
        importances = model.feature_importances_
        feature_impact = []
        for feat, imp in sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True):
            feature_impact.append({
                "feature": FEATURE_IMPORTANCE_NAMES.get(feat, feat),
                "importance": round(float(imp) * 100, 2),
                "value": str(applicant_data.get(feat, "N/A"))
            })

        # Compute DTI ratio
        total_income = float(applicant_data.get('ApplicantIncome', 0)) + float(applicant_data.get('CoapplicantIncome', 0))
        emi = (float(applicant_data.get('LoanAmount', 0)) * 1000) / max(float(applicant_data.get('Loan_Amount_Term', 360)), 1)
        dti = (emi / max(total_income, 1)) * 100

        return {
            "approved": approved,
            "confidence": round(float(max(probability)) * 100, 2),
            "approval_probability": round(float(probability[1]) * 100, 2) if len(probability) > 1 else 0,
            "rejection_probability": round(float(probability[0]) * 100, 2) if len(probability) > 1 else 0,
            "feature_impact": feature_impact[:6],
            "dti_ratio": round(dti, 2),
            "total_income": total_income,
            "monthly_emi": round(emi, 2)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import subprocess, sys
    if not os.path.exists("data/loan_approval.csv"):
        print("Generating dataset...")
        subprocess.run([sys.executable, "generate_dataset.py"])
    train_model()
