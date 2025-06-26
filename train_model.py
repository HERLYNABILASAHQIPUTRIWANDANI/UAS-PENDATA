import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# Hapus baris dengan target NaN
df = df.dropna(subset=["Biopsy"])

# Fitur yang akan digunakan
raw_features = [
    "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
    "Smokes", "Smokes (years)", "Smokes (packs/year)",
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
    "IUD", "IUD (years)",
    "STDs", "STDs (number)",
    "STDs:condylomatosis", "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes",
    "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",
    "STDs:Hepatitis B", "STDs:HPV",
    "STDs: Time since first diagnosis", "STDs: Time since last diagnosis",
    "STDs: Number of diagnosis",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx",
    "Hinselmann", "Schiller", "Citology"
]

# Target
y = df["Biopsy"]
X = df[raw_features]

# Rename kolom agar sesuai dengan form
X.columns = [
    "age", "partners", "first_sex", "pregnancies",
    "smokes", "smoke_years", "smoke_packs",
    "hc", "hc_years", "iud", "iud_years",
    "stds", "stds_number", "stds_condylo", "stds_cervical",
    "stds_vaginal", "stds_vulvo", "stds_syphilis", "stds_pid",
    "stds_herpes", "stds_molluscum", "stds_aids", "stds_hiv",
    "stds_hepb", "stds_hpv", "stds_first_diag", "stds_last_diag",
    "stds_num_diag", "dx_cancer", "dx_cin", "dx_hpv", "dx",
    "hinselmann", "schiller", "cytology"
]

# Imputasi & Scaling
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model dan scaler berhasil disimpan.")
