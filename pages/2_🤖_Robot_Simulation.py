import streamlit as st
import numpy as np
from pathlib import Path
import sys

import importlib.util

if importlib.util.find_spec("pybullet") is None:
    st.error("PyBullet is NOT installed in this Streamlit environment.")
    st.stop()

st.success("PyBullet is installed!")

from simulator.simulation import run_simulation
# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SIM_DATA = PROJECT_ROOT / "demo_data" / "simulation"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Robot Simulation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Robot Arm Simulation")

st.write(
    "Select one of the recorded robot demonstrations and "
    "run the trained model through the simulation."
)

st.info(
    "The simulation compares the model's predicted robot actions "
    "against the recorded demonstration. The blue robot motion is "
    "an approximate visualization of the predicted actions."
)


# ---------------------------------------------------------
# Find available demos
# ---------------------------------------------------------

demo_ids_file = SIM_DATA / "demo_ids.npy"

if not demo_ids_file.exists():
    st.error(
        "Simulation data was not found. "
        "Make sure demo_data/simulation is included in the project."
    )
    st.stop()

demo_ids = np.load(demo_ids_file)

unique_demo_ids = np.unique(demo_ids)


# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    selected_demo = st.selectbox(
        "Select Demonstration",
        unique_demo_ids,
        format_func=lambda x: f"Demo {x}"
    )

with col2:
    mode = st.selectbox(
        "Simulation Mode",
        ["one_step", "cumulative"],
        format_func=lambda x: (
            "One Step" if x == "one_step"
            else "Cumulative"
        )
    )


st.write("")

run_button = st.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# Run simulation
# ---------------------------------------------------------
if not PYBULLET_AVAILABLE:
    st.warning(
        "Robot simulation is unavailable on this deployment because "
        "PyBullet is not installed in the cloud environment."
    )
    st.info(
        "The robot prediction model is still fully available on the Single Prediction page."
    )
    st.stop()
if run_button:

    with st.spinner(
        f"Running Demo {selected_demo} simulation..."
    ):

        try:

            # Import here so the Streamlit page itself
            # doesn't load the heavy model until needed.
            from simulator.simulation import run_simulation

            result = run_simulation(
                demo_id=int(selected_demo),
                mode=mode
            )

        except Exception as e:

            st.error("Simulation failed.")

            st.exception(e)

            st.stop()

    st.success("Simulation complete!")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    st.subheader("Simulation Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Mean BC Error",
            f'{result["mean_error_cm"]:.2f} cm'
        )

    with metric_col2:
        st.metric(
            "Demonstration",
            f'Demo {selected_demo}'
        )

    # -----------------------------------------------------
    # Video
    # -----------------------------------------------------

    st.subheader("Robot Simulation")

    video_path = Path(result["video"])

    if video_path.exists():

        st.video(
            str(video_path)
        )

    else:

        st.warning(
            "Simulation video was not found."
        )

    # -----------------------------------------------------
    # Error plot
    # -----------------------------------------------------

    if result.get("error_plot"):

        error_plot = Path(result["error_plot"])

        if error_plot.exists():

            st.subheader("Prediction Error")

            st.image(
                str(error_plot),
                use_container_width=True
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    if result.get("report"):

        report_path = Path(result["report"])

        if report_path.exists():

            with st.expander("Simulation Report"):

                with open(report_path, "r") as f:
                    report = f.read()

                st.code(
                    report,
                    language="json"
                )