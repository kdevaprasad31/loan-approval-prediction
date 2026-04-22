import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Convert target
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Loan Approval Prediction App")

st.write("Enter customer details to predict loan approval")

# User Inputs
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Term")
Credit_History = st.selectbox("Credit History", [0,1])

# Prediction button
if st.button("Predict Loan Status"):

    input_data = pd.DataFrame({
        'ApplicantIncome':[ApplicantIncome],
        'CoapplicantIncome':[CoapplicantIncome],
        'LoanAmount':[LoanAmount],
        'Loan_Amount_Term':[Loan_Amount_Term],
        'Credit_History':[Credit_History]
    })

    # Add missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")


