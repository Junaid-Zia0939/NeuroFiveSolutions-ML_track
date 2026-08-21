import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AV Braking & Trajectory Safety System",
    page_icon="🚗",
    layout="wide"
)

# 1. Load both saved pipelines
@st.cache_resource
def load_models():
    reg_model = joblib.load("av_time_pipeline.joblib")
    clf_model = joblib.load("av_action_classifier.joblib")
    return reg_model, clf_model

time_pipeline, action_pipeline = load_models()

st.title("🚗 Autonomous Vehicle Collision Avoidance System")
st.markdown("Real-time perception analysis, stopping horizon calculation, and evasive action recommendation.")
st.divider()

# 2. Sidebar for User Telemetry Input
with st.sidebar:
    st.header("⚙️ Environment & Road Conditions")
    weather = st.selectbox("Weather Condition", ["clear", "rainy", "snowy", "foggy"])
    surface = st.selectbox("Road Surface Condition", ["dry", "wet", "snow", "ice"])
    behavior = st.selectbox("Current Driving Behavior", ["cautious", "normal", "aggressive"])
    visibility = st.slider("Visibility Range (m)", 10.0, 500.0, 150.0)

    st.header("🚘 Vehicle & Radar Telemetry")
    speed_mps = st.slider("Ego Speed (m/s)", 2.0, 45.0, 20.0, help="20 m/s is ~72 km/h")
    ego_accel = st.number_input("Ego Acceleration (m/s²)", -10.0, 5.0, 0.0)
    speed_limit = st.slider("Speed Limit (km/h)", 30.0, 130.0, 80.0)
    
    obstacle_dist = st.slider("Obstacle Distance (m)", 2.0, 150.0, 45.0)
    rel_speed = st.slider("Relative Closing Speed (m/s)", 0.5, 40.0, 15.0)
    num_obstacles = st.slider("Obstacles Ahead", 0, 10, 1)
    lane_offset = st.slider("Lane Offset (m)", -2.0, 2.0, 0.0)
    
    # Secondary features
    traffic_density = st.slider("Traffic Density (veh/km)", 5.0, 100.0, 25.0)
    road_width = st.number_input("Road Width (m)", 3.0, 15.0, 7.5)
    road_curvature = st.number_input("Road Curvature (1/m)", 0.0, 0.05, 0.001, format="%.4f")
    steering_angle = st.slider("Steering Angle (deg)", -45.0, 45.0, 0.0)
    yaw_rate = st.number_input("Yaw Rate (rad/s)", -1.0, 1.0, 0.0)
    throttle_pos = st.slider("Throttle Position (%)", 0.0, 100.0, 20.0)
    brake_press = st.slider("Brake Pressure (bar)", 0.0, 150.0, 0.0)
    risk_prob = st.slider("Risk Probability", 0.0, 1.0, 0.15)

# 3. Assemble single-row DataFrame matching training features
input_df = pd.DataFrame([{
    'obstacle_distance_m': obstacle_dist,
    'relative_speed_mps': rel_speed,
    'num_obstacles': num_obstacles,
    'lane_offset_m': lane_offset,
    'traffic_density_veh_per_km': traffic_density,
    'risk_probability': risk_prob,
    'road_curvature_1pm': road_curvature,
    'road_width_m': road_width,
    'speed_limit_kmh': speed_limit,
    'ego_speed_mps': speed_mps,
    'ego_acceleration_mps2': ego_accel,
    'steering_angle_deg': steering_angle,
    'yaw_rate_rads': yaw_rate,
    'throttle_position': throttle_pos,
    'brake_pressure': brake_press,
    'weather_condition': weather,
    'visibility_range_m': visibility,
    'road_surface_condition': surface,
    'behavior_label': behavior
}])

# 4. Model Predictions
pred_times = time_pipeline.predict(input_df)[0]
pred_ttc = pred_times[0]
pred_stop = pred_times[1]
safety_margin = pred_ttc - pred_stop

predicted_action = action_pipeline.predict(input_df)[0]
action_probabilities = action_pipeline.predict_proba(input_df)[0]

# 5. Render Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Available TTC", f"{pred_ttc:.2f} s")
col2.metric("Required Stopping Time", f"{pred_stop:.2f} s")
col3.metric("Safety Margin (Δt)", f"{safety_margin:.2f} s")

st.markdown("---")

# 6. Display Recommended Control Action
st.subheader("🎯 Primary Control Action")

if predicted_action == "Maintain Speed":
    st.success(f"🟢 **Recommendation:** {predicted_action}")
elif predicted_action in ["Gentle Decelerate", "Change Lane"]:
    st.warning(f"🟡 **Recommendation:** {predicted_action}")
else:
    st.error(f"🔴 **CRITICAL ACTION:** {predicted_action}")

# 7. Model Confidence Breakdown
st.write("#### Action Prediction Confidence")
confidence_df = pd.DataFrame({
    'Action': action_pipeline.classes_,
    'Confidence (%)': (action_probabilities * 100).round(1)
}).sort_values(by='Confidence (%)', ascending=False)

st.dataframe(confidence_df, use_container_width=True, hide_index=True)