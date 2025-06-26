import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Kolom fitur sesuai dataset (36 fitur, tanpa kolom target "Biopsy")
feature_names = [
    'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
    'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
    'IUD', 'IUD (years)',
    'STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
    'STDs:HPV', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    'STDs: Number of diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx',
    'Hinselmann', 'Schiller', 'Citology'
]

st.set_page_config(page_title="Prediksi Kanker Serviks", layout="wide")
st.title("🧬 Prediksi Risiko Kanker Serviks")

# Input form
user_inputs = []
with st.form("form_prediksi"):
    st.write("Silakan masukkan data pasien:")
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(feature, min_value=0.0, step=1.0, format="%.2f", key=feature)
            user_inputs.append(val)
    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    if len(user_inputs) != len(feature_names):
        st.error(f"Jumlah input hanya {len(user_inputs)}. Seharusnya {len(feature_names)} kolom.")
    else:
        input_scaled = scaler.transform([user_inputs])
        prediction = model.predict(input_scaled)[0]
        result = "✅ Pasien Berisiko Kanker Serviks" if prediction == 1 else "✅ Pasien Tidak Berisiko"
        st.success(result)
