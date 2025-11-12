import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load model and features
# -------------------------------
model = joblib.load("final_house_price_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏡 Advanced House Price Prediction App")
st.write("Enter the property details below to predict the sale price.")

# Sidebar
st.sidebar.header("ℹ️ Model Information")
st.sidebar.write("**Algorithm:** Random Forest Regressor")
st.sidebar.write("**RMSE:** ≈ 29,000")
st.sidebar.write("**R² Score:** ≈ 0.88")
st.sidebar.write("**Trained on:** Ames Housing Dataset")

# -------------------------------
# Feature Importance Plot
# -------------------------------
st.sidebar.subheader("📊 Feature Importances")

# Get feature importances from the model
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True).tail(10)

fig, ax = plt.subplots()
feat_imp.plot(kind="barh", ax=ax)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.sidebar.pyplot(fig)

# Input UI
st.subheader("Enter Property Features")
col1, col2, col3 = st.columns(3)

with col1:
    OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
    GarageCars = st.slider("Garage Capacity (Cars)", 0, 4, 2)

with col2:
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    YearBuilt = st.number_input("Year Built", 1870, 2025, 2000)
    FullBath = st.slider("Full Bathrooms", 0, 3, 2)

with col3:
    Neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert"])
    ExterQual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"])

# Prepare input data
input_df = pd.DataFrame({
    "OverallQual": [OverallQual],
    "GrLivArea": [GrLivArea],
    "GarageCars": [GarageCars],
    "TotalBsmtSF": [TotalBsmtSF],
    "YearBuilt": [YearBuilt],
    "FullBath": [FullBath],
    "Neighborhood": [Neighborhood],
    "ExterQual": [ExterQual],
})

# Encode input (no drop_first)
input_encoded = pd.get_dummies(input_df, columns=["Neighborhood", "ExterQual"], drop_first=False)

# Align with model features
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# Predict
if st.button("💰 Predict House Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")
