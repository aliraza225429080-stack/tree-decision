import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="centered"
)

# App title and description
st.title("☕ Starbucks Beverage Calories Predictor")
st.write("Enter the nutritional details of your Starbucks beverage to predict its calorie count and calorie category tier using trained Machine Learning models.")

# Load the saved models with error handling
@st.cache_resource
def load_models():
    try:
        regressor = joblib.load('starbucks_dt_regressor.joblib')
        classifier = joblib.load('starbucks_dt_classifier.joblib')
        return regressor, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

reg_model, clf_model = load_models()

if reg_model is not None and clf_model is not None:
    # Create Input fields
    st.header("Input Nutritional Features")

    col1, col2 = st.columns(2)

    with col1:
        fat = st.number_input("Total Fat (g)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        carb = st.number_input("Total Carbohydrates (g)", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
        fiber = st.number_input("Dietary Fiber (g)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)

    with col2:
        protein = st.number_input("Protein (g)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        sugar = st.number_input("Sugars (g)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

    # Features ordered according to model training
    input_features = np.array([[fat, carb, fiber, protein, sugar]])

    # Prediction button
    if st.button("Predict Calories", type="primary"):
        # Make predictions
        predicted_calories = reg_model.predict(input_features)[0]
        predicted_tier = clf_model.predict(input_features)[0]

        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Estimated Calories", value=f"{predicted_calories:.1f} kcal")
        with metric_col2:
            color_map = {"Low": "green", "Medium": "orange", "High": "red"}
            color = color_map.get(predicted_tier, "blue")
            st.markdown(f"### Calorie Tier: :{color}[{predicted_tier}]")
else:
    st.info("Please ensure 'starbucks_dt_regressor.joblib' and 'starbucks_dt_classifier.joblib' are present in the directory.")
