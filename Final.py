import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Sidebar selection
st.sidebar.title("Select Disease Prediction")
choice = st.sidebar.selectbox("Choose a disease:", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# Common function to load a model
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
# ----------------- Kidney Disease -----------------
if choice == "Kidney Disease":
    st.title('Kidney Disease Prediction')

    # Load and clean data
    kd = pd.read_csv(r"C:\Users\usre\Downloads\kidney_disease - kidney_disease.csv")
    kd.replace('?', np.nan, inplace=True)
    for col in ['sc', 'bu', 'hemo', 'pcv']:
        kd[col] = pd.to_numeric(kd[col], errors='coerce')
        kd[col] = kd[col].fillna(kd[col].median())
    kd['sg'] = pd.to_numeric(kd['sg'], errors='coerce')
    kd['sg'] = kd['sg'].fillna(kd['sg'].mode()[0])

    # Load model and scaler
    model = load_model(r'C:\Users\usre\random_forest_model.pkl')
    scaler = load_model(r'C:\Users\usre\scaler.pkl')

    st.subheader("Model Info")
    st.write("**Model:** Random Forest | **Accuracy:** 98.75%")

    st.subheader("Input Features")
    sc = st.number_input('Serum Creatinine (sc)', min_value=0.0, step=0.01)
    bu = st.number_input('Blood Urea (bu)', min_value=0.0, step=0.1)
    hemo = st.number_input('Hemoglobin (hemo)', min_value=3.0, step=0.1)
    pcv = st.number_input('Packed Cell Volume (pcv)', min_value=10, step=1)
    sg = st.selectbox('Specific Gravity (sg)', options=[1.005, 1.010, 1.015, 1.020, 1.025])

    if st.button('Predict Kidney Disease'):
        input_data = np.array([[sc, bu, hemo, pcv, sg]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error(f"⚠️ Prediction: **Kidney Disease Detected** ({prediction})")
        else:
            st.success(f"✅ Prediction: **No Kidney Disease Detected** ({prediction})")

# ----------------- Liver Disease -----------------
elif choice == "Liver Disease":
    st.title('Liver Disease Prediction')

    l = pd.read_csv(r"C:\Users\usre\Downloads\indian_liver_patient - indian_liver_patient.csv")
    l['Gender'] = l['Gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0}).astype(int)

    model = load_model(r'C:\Users\usre\random_forest_liver_model.pkl')

    st.subheader("Model Info")
    st.write("**Model:** Random Forest | **Accuracy:** 76.00%")

    st.subheader("Input Features")
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, format="%.2f")
    alk_phos = st.number_input("Alkaline Phosphotase", min_value=0, step=1)
    alt = st.number_input("Alamine Aminotransferase (ALT)", min_value=0, step=1)
    total_protein = st.number_input("Total Proteins", min_value=0.0, format="%.2f")

    if st.button("Predict Liver Disease"):
        input_data = np.array([[total_bilirubin, alk_phos, alt, total_protein]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error(f"⚠️ Prediction: **Liver Disease Detected ( {prediction} )**")
            st.write("🔴 Your test results indicate the presence of **liver disease**. Please consult a doctor for further medical evaluation.")
        else:
            st.success(f"✅ Prediction: **No Liver Disease ( {prediction} )**")
            st.write("🟢 Your test results indicate **no liver disease**. However, maintain a healthy lifestyle and regular checkups.")

# ----------------- Parkinson's Disease -----------------
else:
    st.title("Parkinson's Disease Prediction")

    p = pd.read_csv(r"C:\Users\usre\Downloads\parkinsons - parkinsons.csv")
    features = ['PPE', 'RPDE', 'DFA', 'HNR', 'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'D2']
    model = load_model(r'C:\Users\usre\random_forest_par_model.pkl')

    st.subheader("Model Info")
    st.write("**Model:** Random Forest | **Accuracy:** 90.00%")

    st.subheader("Input Features")
    PPE = st.number_input("PPE", min_value=0.0, format="%.2f")
    RPDE = st.number_input("RPDE", min_value=0.0, format="%.2f")
    DFA = st.number_input("DFA", min_value=0.0, format="%.2f")
    HNR = st.number_input("HNR", min_value=0.0, format="%.2f")
    MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.2f")
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.2f")
    D2 = st.number_input("D2", min_value=0.0, format="%.2f")

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([[PPE, RPDE, DFA, HNR, MDVP_Fo_Hz, MDVP_Jitter, D2]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error(f"⚠️ Prediction: **Parkinson's Detected ( {prediction} )**")
            st.write("🔴 Your test results indicate the presence of **Parkinson's disease**. Please consult a doctor for further medical evaluation.")
        else:
            st.success(f"✅ Prediction: **No Parkinson's Detected ( {prediction} )**")
            st.write("🟢 Your test results indicate **No Parkinson's Detected**. However, maintain a healthy lifestyle and regular checkups.")
