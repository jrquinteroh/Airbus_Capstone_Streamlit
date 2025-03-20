import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Config
st.set_page_config(page_title="Fuel Leak Detection Software", layout="wide")

# Load model
model = joblib.load("xgboost_fuel_leak_model_v2.pkl")

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"  # Default page

# BUTTON CONFIGURATION
# ====================
st.markdown("""
    <style>
    .stButton > button {
        height: auto;
        padding-top: 20px;
        padding-bottom: 20px;
        font-weight: bold !important;
        font-size: 69px !important; /* Larger font size for buttons */
        color: white; /* white text in normal state */
        border: 5px solid black; /* black border */
        border-radius: 40px;
        width: 100%;
        cursor: pointer;
        background-color: #00205B; /* dark blue background in normal state */
    }
    .stButton > button:hover {
        background-color: white; /* white background on hover */
        color: black; /* black text on hover */
        border: 5px solid black; /* Ensures border stays black */
    }
    .button-container {
        display: flex;
        flex-direction: column; /* Stack buttons vertically */
        gap: 30px; /* Space between buttons */
        align-items: flex-start; /* Align buttons to the left */
        margin-top: 150px; /* Move buttons lower on the y-axis */
    }
    .header-text {
        color: #003087; /* Airbus blue */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stApp {
        color: #333;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #003087;
    }
    </style>
    """, unsafe_allow_html=True)

# Layout with image and navigation
st.markdown("<h1 style='text-align: center;'>Fuel Leak Detection Software</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)
    st.image('airbus_streamlit.png', width=600)
with col2:
    st.markdown("<h2 style='text-align: center; font-size: 36px;'>FuelGuard: Protecting the Skies, One Flight at a Time</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>A Software Designed to Protect Aircraft From Fuel Leaks Using Machine Learning</h2>", unsafe_allow_html=True)
    # Navigation buttons in a container
    st.markdown('<div class="button-container" style="margin-top: 50px;">', unsafe_allow_html=True)  # Reduced margin-top to bring buttons closer
    if st.button("Home"):
        st.session_state["page"] = "Home"
    if st.button("Model Input"):
        st.session_state["page"] = "Model Input"
    if st.button("Prediction & Recommendations"):
        st.session_state["page"] = "Prediction & Recommendations"
    st.markdown('</div>', unsafe_allow_html=True)

# Use session state to determine the current page
page = st.session_state["page"]

# Page 1: Landing Page
if page == "Home":
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .section-title {
            color: #003087;  /* Airbus blue color */
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .content-box {
            background-color: #f5f7fa;  /* Light gray background for contrast */
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);  /* Subtle shadow for depth */
        }
        .content-text {
            font-size: 16px;
            line-height: 1.6;  /* Better line spacing for readability */
            color: #333;  /* Darker text for better contrast */
        }
        .bullet-list {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            margin-left: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main title
    st.markdown("<h1 style='text-align: center;'>Fuel Leak Detection in Aircraft (A400M) Information</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='section-title'>Welcome to Our Leak Detection App!</div>", unsafe_allow_html=True)
        st.markdown('''
        <div class='content-text'>
This app is designed to help you identify potential fuel leaks in your aircrafts during flight operations, particularly during the cruise phase when the aircraft is most stable. Early detection is critical to prevent operational inefficiencies, reduce maintenance costs, and enhance overall safety.
        </div>
    ''', unsafe_allow_html=True)


    with col2:


        st.markdown("<div class='section-title'>How Are We Solving The Problem?</div>", unsafe_allow_html=True)
        st.markdown('''
        <div class='content-text'>
        We addressed these challenges by developing a powerful machine learning model specifically trained to detect leaks based on various calculated features derived from the raw data. 
        The process included thorough data cleaning, feature engineering, and the development of three different approaches for detecting leaks:
        <ul>
        <li>Leak detection by comparing Fuel On Board (FOB) vs. Estimated FOB.</li>
        <li>Leak detection by analyzing differences in engine consumption (FUEL_FLAG).</li>
        <li>Leak detection by comparing Tank FOB vs. Estimated FOB.</li>
        </ul>
        </div>
    ''', unsafe_allow_html=True)

    with col3:
# Centered Conclusion Section
         st.markdown('''
        <div style="display: flex; justify-content: center; margin-top: 30px;">
            <div style="border: 2px solid #ccc; padding: 20px; width: 300px; height: 300px; display: flex; align-items: center; justify-content: center; text-align: center;">
                <div>
                    <h2>WHY?</h2>
                    <p>With our Leak Detection App, you can detect leaks early, reduce operational costs, and enhance the safety and reliability of your aircrafts.<br><br>
                    Thank you for choosing our app powered by SKYWISE to enhance your operations!</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)




# Page 2: Model Input
elif page == "Model Input":
    st.markdown("<h1 style='text-align: center;'>Upload Data for Prediction</h1>", unsafe_allow_html=True)
    
    # Adding your text for explanation
    st.markdown("### About Our Model")
    st.write("""
    #### The Model We Use - XGBoost
    XGBoost was chosen for its robustness in handling non-linear relationships and delivering high accuracy. 
    It works by building an ensemble of decision trees, where each model improves upon the errors of the previous one. 
    To achieve the best performance, we used Optuna for hyperparameter tuning, allowing us to efficiently explore different configurations and optimize the model.
    """)

    st.markdown("### Model Performance")
    st.write("""
    The performance of our model is evaluated using the following metrics:
    - **Accuracy**: 94%
    - **Precision**: 91.2%
    - **Recall**: 64%
    - **F1-Score**: 75%
    - **ROC-AUC**: 84.4%

    These metrics show that our model effectively detects fuel leaks with high precision, ensuring reliable results.
    """)

    st.markdown("<h1 style='text-align: center;'>Upload Data for Prediction</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_parquet(uploaded_file)
        st.session_state["data"] = data
        st.success("File uploaded successfully!", icon="✅")


    # Page 3: Prediction & Recommendations
elif page == "Prediction & Recommendations":
    st.markdown("<h1 style='text-align: center; color: #003087;'>Prediction Results and Next Steps</h1>", unsafe_allow_html=True)
    if "data" in st.session_state:
        data = st.session_state["data"]
        
        # Define columns to drop for model prediction (same as during training)
        columns_to_drop = [
            "FLIGHT_PHASE_COUNT", "UTC_TIME", "sequence_id", "is_long_sequence",
            "Leak_FOB", "LEAK", "Leak_Tank", "MSN", "Flight", "fuel_flag"
        ]
        
        # Prepare data for prediction by dropping columns not used during training
        data_for_prediction = data.drop(columns=columns_to_drop, errors="ignore")
        
        # Debug: Show only the data shape and sample after preprocessing
        st.write("### Data Information")
        st.write("Data shape:", data_for_prediction.shape)
        st.write("Data sample:")
        st.dataframe(data_for_prediction, use_container_width=True)
        
        # Ensure the number of features matches the model's expectations
        expected_num_features = len(model.feature_names_in_)
        if data_for_prediction.shape[1] != expected_num_features:
            st.error(
                f"Feature shape mismatch! Model expects {expected_num_features} features, "
                f"but the data has {data_for_prediction.shape[1]} features. "
                f"Expected columns: {model.feature_names_in_}"
            )
        else:
            # Convert to numpy array for prediction
            data_array = data_for_prediction.to_numpy()
            
            # Make prediction
            prediction = model.predict(data_array)
            
            # Identify FUEL_USED columns, excluding FUEL_USED_SINCE_START
            fuel_used_cols = [col for col in data.columns if col.startswith("FUEL_USED_") and col != "FUEL_USED_SINCE_START"]
            if not fuel_used_cols:
                st.error("No FUEL_USED columns found in the data (excluding FUEL_USED_SINCE_START). Please check the input file.")
            else:
                # Calculate fuel_combined as the mean of FUEL_USED columns for each row
                fuel_used_data = data[fuel_used_cols]
                fuel_combined = fuel_used_data.mean(axis=1)
                
                # Calculate the 3% threshold for each row
                threshold = 0.03  # 3% threshold
                lower_bound = fuel_combined * (1 - threshold)
                upper_bound = fuel_combined * (1 + threshold)
                
                # Identify engines outside the 3% threshold and their direction (above/below)
                threshold_violations = []
                violation_directions = []
                for i in range(len(data)):
                    violations = []
                    directions = []
                    for col in fuel_used_cols:
                        fuel_value = data.iloc[i][col]
                        if fuel_value < lower_bound.iloc[i]:
                            engine_num = col.split("_")[-1]
                            violations.append(f"Engine {engine_num}")
                            directions.append("below")
                        elif fuel_value > upper_bound.iloc[i]:
                            engine_num = col.split("_")[-1]
                            violations.append(f"Engine {engine_num}")
                            directions.append("above")
                    if violations:
                        threshold_violations.append(", ".join(violations))
                        violation_directions.append(", ".join([f"{v} ({d})" for v, d in zip(violations, directions)]))
                    else:
                        threshold_violations.append("None")
                        violation_directions.append("None")
                
                # Create a DataFrame for the prediction results
                prediction_df = pd.DataFrame({
                    "UTC_TIME": data["UTC_TIME"],
                    **{col: data[col] for col in fuel_used_cols},
                    "3% Threshold Violation": threshold_violations
                })

                # Display the filtered table
                st.markdown("### Prediction Output", unsafe_allow_html=True)
                st.dataframe(prediction_df, use_container_width=True, height=400)

                # Display LEAK DETECTED sign if leaks are found
                if any(violation != "None" for violation in threshold_violations):
                    st.markdown(
                        """
                        <div style='background-color: #FFCDD2; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                            <h1 style='color: #D32F2F; text-align: center;'>LEAK DETECTED!</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )


                st.image("airbus_streamlit.jpeg")

                # Display precision
                st.markdown("### Precision")
                precision = 0.9117
                st.metric("Precision", f"{precision:.2%}")
                
                # Recommendations if a leak is detected
                st.markdown("### Recommendations")
                if 1 in prediction:
                    violating_engines = set()
                    for violation in threshold_violations:
                        if violation != "None":
                            engines = violation.split(", ")
                            violating_engines.update(engines)
                    
                    if violating_engines:
                        st.write(f"- **Approach 2 (Engine Fuel Consumption)**: Anomaly in engine usage. Check fuel systems in: {', '.join(violating_engines)}.")
                    else:
                        st.write("- **Approach 2 (Engine Fuel Consumption)**: Anomaly in engine usage. Check fuel systems in all engines.")
                else:
                    st.write("No immediate actions required. Continue monitoring fuel systems as per standard protocols.")
                
                # Further steps
                st.markdown("### Further Steps")
                st.write("- Schedule a detailed fuel system inspection.")
                st.write("- Cross-check with maintenance logs.")
                st.write("- Consult Airbus protocols for escalation.")
    else:
        st.warning("Please upload a file first!")
