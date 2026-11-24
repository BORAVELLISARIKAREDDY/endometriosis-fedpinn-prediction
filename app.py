import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).parent
DATA_PATH = BASE / "advanced_endometriosis_dataset.csv"
MODEL_PATH = BASE / "endometriosis_predictor.keras"
ENCODERS_PATH = BASE / "category_encoders.json"
FEATURES = [
    "Age", "BMI", "Pelvic_Pain_Level", "Infertility_History",
    "Hormonal_Imbalance", "Family_History_Endometriosis",
    "Ultrasound_Findings", "MRI_Findings", "CA125_Level",
]
CATEGORICAL = ["Infertility_History", "Hormonal_Imbalance", "Family_History_Endometriosis"]
TARGET = "Diagnosis"

st.set_page_config(page_title="Endometriosis Research Predictor", page_icon="🧬", layout="wide")

@st.cache_resource(show_spinner="Preparing the prediction model (first launch may take a minute)...")
def get_model_and_encoders():
    if not DATA_PATH.exists():
        raise FileNotFoundError("advanced_endometriosis_dataset.csv is missing from the repository.")

    rebuild = True
    if MODEL_PATH.exists() and ENCODERS_PATH.exists():
        try:
            candidate = tf.keras.models.load_model(MODEL_PATH, compile=False)
            rebuild = candidate.input_shape[-1] != len(FEATURES)
            if not rebuild:
                with open(ENCODERS_PATH, encoding="utf-8") as f:
                    encoders = json.load(f)
                return candidate, encoders
        except Exception:
            rebuild = True

    data = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = data[FEATURES + [TARGET]].dropna().copy()
    encoders = {}
    for col in CATEGORICAL + [TARGET]:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        encoders[col] = {str(label): int(index) for index, label in enumerate(encoder.classes_)}

    X = data[FEATURES].astype("float32").to_numpy()
    y = data[TARGET].astype("float32").to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train, y_train, validation_split=0.2, epochs=12, batch_size=32, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )],
    )
    model.evaluate(X_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    with open(ENCODERS_PATH, "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2)
    return model, encoders

st.title("🧬 Endometriosis Risk Prediction")
st.caption("Research prototype | Neural-network prediction using the supplied tabular dataset")
st.warning(
    "Educational/research demonstration only. This tool is not clinically validated, "
    "does not diagnose endometriosis, and must not be used to make medical decisions. "
    "Predictions should be interpreted only by qualified healthcare professionals."
)

try:
    model, encoders = get_model_and_encoders()
except Exception as exc:
    st.error(f"The model could not be initialized: {exc}")
    st.stop()

st.success("Prediction model is ready.")
st.markdown("Enter the sample clinical and imaging-related fields below.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=12, max_value=80, value=25, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.2, step=0.1)
        pelvic_pain = st.slider("Pelvic pain level", min_value=0, max_value=10, value=6)
    with c2:
        infertility = st.selectbox("Infertility history", list(encoders["Infertility_History"].keys()))
        hormonal = st.selectbox("Hormonal imbalance", list(encoders["Hormonal_Imbalance"].keys()))
        family = st.selectbox("Family history of endometriosis", list(encoders["Family_History_Endometriosis"].keys()))
    with c3:
        st.markdown("**Imaging finding codes**")
        st.caption("The dataset stores these as numeric codes 0, 1 and 2; no code-to-description legend was supplied.")
        ultrasound = st.selectbox("Ultrasound findings (dataset code)", [0, 1, 2], index=1)
        mri = st.selectbox("MRI findings (dataset code)", [0, 1, 2], index=1)
        ca125 = st.number_input("CA-125 level (dataset units)", min_value=0.0, max_value=1000.0, value=35.0, step=0.5)
    submitted = st.form_submit_button("Generate prediction", type="primary", use_container_width=True)

if submitted:
    row = {
        "Age": float(age), "BMI": float(bmi), "Pelvic_Pain_Level": float(pelvic_pain),
        "Infertility_History": float(encoders["Infertility_History"][infertility]),
        "Hormonal_Imbalance": float(encoders["Hormonal_Imbalance"][hormonal]),
        "Family_History_Endometriosis": float(encoders["Family_History_Endometriosis"][family]),
        "Ultrasound_Findings": float(ultrasound), "MRI_Findings": float(mri),
        "CA125_Level": float(ca125),
    }
    values = np.array([[row[name] for name in FEATURES]], dtype=np.float32)
    probability = float(model.predict(values, verbose=0).reshape(-1)[0])
    positive_probability = probability if encoders[TARGET].get("Yes") == 1 else 1.0 - probability
    prediction = "Positive model output" if positive_probability >= 0.5 else "Negative model output"
    st.divider()
    st.subheader("Prediction output")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Model-estimated positive-class score", f"{positive_probability:.1%}")
        st.write(f"**Output category:** {prediction}")
    with right:
        st.progress(max(0.0, min(1.0, positive_probability)))
        st.caption("This is a model score from the supplied dataset, not a confirmed disease probability.")
    with st.expander("View submitted model features"):
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)

with st.expander("About this prototype"):
    st.write(
        "The supplied legacy H5 model has a 5-feature input, while the dataset and intended "
        "feature list contain 9 predictors. This app rebuilds a 9-feature neural-network baseline "
        "from the included CSV. The original custom physics-loss implementation was inconsistent "
        "and does not demonstrate a verified physics constraint. Do not describe this deployed "
        "baseline as a validated FedPINN or clinically validated digital twin."
    )
