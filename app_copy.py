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

        # Section 1: What is the Problem?
        st.markdown('<div class="section-title">What is the Problem?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            Fuel leaks are typically detected through visual inspections either before takeoff or after landing. However, by the time a leak is visible, it already poses significant safety and economic risks. Additionally, undetected mid-flight leaks can compromise aircraft stability, leading to emergency landings and reputational damage for the airline and Airbus.<br><br>
            Current leak detection methods rely on:
            <ul class="bullet-list">
                <li>Fuel-on-board (FOB) comparisons</li>
                <li>Flight Management System (FMSB) predictions</li>
                <li>Sensor data</li>
            </ul>
            Nonetheless, these methods have limitations, such as delayed detection, sensor blind spots, and false positives due to environmental conditions.<br><br>
            Our proposal leverages machine learning models to enhance predictive capabilities. By focusing on the aircraft's cruising phase—where factors such as pitch and roll are minimized—we can improve accuracy and detect leaks at an early stage. This proactive approach will:
            <ul class="bullet-list">
                <li>Reduce inspection time and costs</li>
                <li>Optimize flight planning</li>
                <li>Ensure safety while minimizing the financial impact of fuel loss</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Section 2: How Are We Targeting It?
        st.markdown('<div class="section-title">How Are We Targeting It?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            We faced three major challenges:
            <ul class="bullet-list">
                <li>Significant feature differences across datasets, particularly between a flight-test aircraft dataset and in-service aircraft datasets</li>
                <li>A high proportion of null values requiring imputation or removal</li>
                <li>The absence of a target variable in most datasets, necessitating feature engineering to create an equivalent target variable for modeling</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Section 3: How Are We Solving It?
        st.markdown('<div class="section-title">How Are We Solving It?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="content-text">
            After defining our target variable and preprocessing the data, we tested six different model types for fuel leak detection:
            <ul class="bullet-list">
                <li>Random Forest</li>
                <li>Logistic Regression</li>
                <li>Support Vector Machines (SVMs)</li>
                <li>XGBoost</li>
                <li>Autoencoders</li>
                <li>Isolation Forest</li>
            </ul>
            Each model had its strengths, but after extensive testing and hyperparameter tuning, XGBoost consistently outperformed the others, capturing a high percentage of actual leaks while minimizing false positives.<br><br>
            For model training, we:
            <ul class="bullet-list">
                <li>Removed redundant features to avoid tampering with our results, specifically the binary features used to build our target</li>
                <li>Maintained the sequential integrity of the data during train-test splits with a ‘fixed’ split, due to the time series nature of our data</li>
                <li>Applied Min-Max scaling to improve optimization</li>
            </ul>
            Hyperparameter tuning was conducted using Optuna, allowing efficient exploration of configurations. Once optimized, XGBoost delivered the best performance, making it our final choice to construct predictions.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# Page 2: Model Input
elif page == "Model Input":
    st.markdown("<h1 style='text-align: center;'>Upload Data for Prediction</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_parquet(uploaded_file)
        st.session_state["data"] = data
        st.success("File uploaded successfully!", icon="✅")
    st.markdown("### About Our Model")
    st.write("XGBoost chosen for...")

# Page 3: Prediction & Recommendations
elif page == "Prediction & Recommendations":
    st.markdown("<h1 style='text-align: center;'>Prediction Results and Next Steps</h1>", unsafe_allow_html=True)
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
                    "Prediction (0 = No Leak, 1 = Leak)": prediction,
                    "Result": ["Leak" if pred == 1 else "No Leak" for pred in prediction],
                    "3% Threshold Violation": threshold_violations
                })
                
                # Display the prediction results in a styled table
                st.write("### Prediction Output")
                st.markdown(
                    """
                    <style>
                    .prediction-table {
                        font-size: 18px !important;
                        width: 100%;
                    }
                    .prediction-table th {
                        background-color: #1976D2 !important;
                        color: white !important;
                        font-size: 20px !important;
                        padding: 10px !important;
                    }
                    .prediction-table td {
                        padding: 10px !important;
                    }
                    .leak {
                        color: #1976D2 !important;
                        font-weight: bold !important;
                    }
                    .no-leak {
                        color: #2E7D32 !important;
                        font-weight: bold !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                def style_result(val):
                    color = '#1976D2' if val == "Leak" else '#2E7D32'
                    return f'color: {color}; font-weight: bold;'
                
                styled_df = prediction_df.style.applymap(
                    style_result, subset=["Result"]
                ).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#1976D2'), ('color', 'white'), ('font-size', '20px'), ('padding', '10px')]},
                    {'selector': 'td', 'props': [('padding', '10px'), ('font-size', '18px')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Summarize the prediction for the sequence based on fuel_flag
                if data["fuel_flag"].sum() > 0:
                    st.write("**Prediction Summary: Leak Detected!**", unsafe_allow_html=True)
                    
                    # Summarize unique engines with threshold violations and their directions
                    engine_violations = {}
                    for violation, direction in zip(threshold_violations, violation_directions):
                        if violation != "None":
                            engines = violation.split(", ")
                            directions = direction.split(", ")
                            for eng, dir in zip(engines, directions):
                                direction_clean = dir.split('(')[1].replace(')', '')
                                if eng in engine_violations:
                                    if engine_violations[eng] != "above":
                                        engine_violations[eng] = direction_clean
                                else:
                                    engine_violations[eng] = direction_clean
                    
                    # Create the summary message with unique engines
                    if engine_violations:
                        violation_summary = [f"{eng} ({direction})" for eng, direction in engine_violations.items()]
                        engine_message = f"Engines with potential leaks: {', '.join(violation_summary)}."
                    else:
                        engine_message = "No engines significantly deviate from the fuel_combined ±3% threshold."
                    
                    # Display the summary with the engine message in a styled box
                    st.markdown(
                        f"""
                        <div style='background-color: #FFCDD2; padding: 10px; border-radius: 5px;'>
                            <span style='color: #D32F2F; font-weight: bold; font-size: 24px;'>A fuel leak was detected in this sequence.</span><br>
                            <span style='color: #D32F2F; font-weight: bold; font-size: 20px;'>{engine_message}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.write("**Prediction Summary: No Leak Detected**", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div style='background-color: #C8E6C9; padding: 10px; border-radius: 5px;'>
                            <span style='color: #2E7D32; font-weight: bold; font-size: 24px;'>No fuel leak was detected in this sequence.</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                st.image("airbus_streamlit.jpeg")

                # Display precision
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