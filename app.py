import streamlit as st
import pandas as pd
import pickle

# Set Page Config
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Diagnosis Prediction App")
st.write("Input the feature values below or load a test sample to evaluate the model.")

# Load Trained Pipeline and LabelEncoder using Pickle
@st.cache_resource
def load_models():
    try:
        with open("diagonsis_detection.pkl", "rb") as model_file:
            pipeline = pickle.load(model_file)
            
        with open("label_encoder.pkl", "rb") as le_file:
            label_encoder = pickle.load(le_file)
            
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'pipeline.pkl' and 'label_encoder.pkl' are in the same folder as this script.")
        return None, None

pipe, lb = load_models()

# ==========================================
# EXACT VARIABLES EXTRACTED FROM YOUR FILE
# ==========================================
file_variables = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# ==========================================
# SAMPLE DATA PROFILES FOR TESTING
# ==========================================
sample_data_benign = {
    'radius_mean': 13.54, 'texture_mean': 14.36, 'perimeter_mean': 87.46, 'area_mean': 566.3, 'smoothness_mean': 0.09779,
    'compactness_mean': 0.08129, 'concavity_mean': 0.06664, 'concave points_mean': 0.04781, 'symmetry_mean': 0.1885, 'fractal_dimension_mean': 0.05766,
    'radius_se': 0.2699, 'texture_se': 0.7886, 'perimeter_se': 2.058, 'area_se': 23.56, 'smoothness_se': 0.008462,
    'compactness_se': 0.0146, 'concavity_se': 0.02387, 'concave points_se': 0.01315, 'symmetry_se': 0.0198, 'fractal_dimension_se': 0.0023,
    'radius_worst': 15.11, 'texture_worst': 19.26, 'perimeter_worst': 99.7, 'area_worst': 711.2, 'smoothness_worst': 0.144,
    'compactness_worst': 0.1773, 'concavity_worst': 0.239, 'concave points_worst': 0.1288, 'symmetry_worst': 0.2977, 'fractal_dimension_worst': 0.07259
}

sample_data_malignant = {
    'radius_mean': 19.69, 'texture_mean': 21.25, 'perimeter_mean': 130.0, 'area_mean': 1203.0, 'smoothness_mean': 0.1096,
    'compactness_mean': 0.1599, 'concavity_mean': 0.1974, 'concave points_mean': 0.1279, 'symmetry_mean': 0.2069, 'fractal_dimension_mean': 0.05999,
    'radius_se': 0.7456, 'texture_se': 1.055, 'perimeter_se': 5.129, 'area_se': 82.03, 'smoothness_se': 0.00615,
    'compactness_se': 0.04006, 'concavity_se': 0.03832, 'concave points_se': 0.02058, 'symmetry_se': 0.0225, 'fractal_dimension_se': 0.004571,
    'radius_worst': 23.57, 'texture_worst': 25.53, 'perimeter_worst': 152.5, 'area_worst': 1709.0, 'smoothness_worst': 0.1444,
    'compactness_worst': 0.4245, 'concavity_worst': 0.4504, 'concave points_worst': 0.243, 'symmetry_worst': 0.3613, 'fractal_dimension_worst': 0.08758
}

# Initialize session state for inputs
for feature in file_variables:
    if feature not in st.session_state:
        st.session_state[feature] = 0.0

# Sample Selector Widget
st.subheader("🧪 Quick Test Samples")
selected_sample = st.selectbox(
    "Load a pre-configured sample profile:",
    ["Choose a sample...", "Benign Sample (B)", "Malignant Sample (M)"]
)

if selected_sample == "Benign Sample (B)":
    for k, v in sample_data_benign.items():
        st.session_state[k] = v
elif selected_sample == "Malignant Sample (M)":
    for k, v in sample_data_malignant.items():
        st.session_state[k] = v

st.markdown("---")
st.subheader("Input Feature Values")

# Dynamically generate input fields bound to session state
input_data = {}
columns = st.columns(3)

for index, feature in enumerate(file_variables):
    col = columns[index % 3]
    with col:
        input_data[feature] = st.number_input(
            f"{feature}", 
            key=feature, 
            format="%.5f"
        )

st.markdown("---")

# Prediction logic
if st.button("Predict Diagnosis", type="primary"):
    if pipe is not None and lb is not None:
        input_df = pd.DataFrame([input_data])
        
        prediction_num = pipe.predict(input_df)[0]
        prediction_label = lb.inverse_transform([prediction_num])[0]
        
        st.subheader("Result:")
        if prediction_label == 'M':
            st.error("**Predicted Diagnosis:** Malignant (M)")
        else:
            st.success("**Predicted Diagnosis:** Benign (B)")
            
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(input_df)[0]
            st.info(f"**Confidence:** {probs[prediction_num] * 100:.2f}%")