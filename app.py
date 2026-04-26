import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Title
st.title("HR Attrition Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select target column
    st.subheader("Select Target Column")
    target = st.selectbox("Choose target column", df.columns)

    if target:
        # Convert target if needed
        if df[target].dtype == "object":
            df[target] = df[target].map({"Yes": 1, "No": 0})

        # Split features & target
        X = df.drop(target, axis=1)
        y = df[target]

        # Handle categorical
        X = pd.get_dummies(X)

        st.subheader("Feature Data")
        st.write(X.head())

        # Train model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        model.fit(X, y)

        st.success("Model trained successfully!")

        # Predictions preview
        preds = model.predict(X)
        df["Prediction"] = preds

        st.subheader("Prediction Preview")
        st.write(df.head())

        # 📊 Accuracy (optional)
        acc = model.score(X, y)
        st.write(f"Training Accuracy: {acc:.2f}")

        # 💾 Save model + columns
        model_bytes = io.BytesIO()
        joblib.dump({
            "model": model,
            "columns": X.columns
        }, model_bytes)
        model_bytes.seek(0)

        # 📥 Download button
        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="attrition_model.pkl",
            mime="application/octet-stream"
        )

    else:
        st.warning("Please select a target column")