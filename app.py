import streamlit as st
import sqlite3
import json
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd

from model.predict import predict_robot_action


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Robot Arm Predictor",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.4rem !important;
}

h3 {
    font-size: 1.15rem !important;
}

p, label, .stMarkdown {
    font-size: 0.9rem !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}

img {
    max-height: 300px;
    object-fit: contain;
}

div[data-testid="stMetric"] {
    padding: 0.4rem !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
}

button {
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DB_PATH = "predictions.db"
DATA_DIR = "demo_data"


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                prediction TEXT,
                ground_truth TEXT,
                error TEXT,
                mae REAL,
                dataset_split TEXT,
                dataset_index INTEGER,
                timestamp TEXT
            )
        """)

        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)")
        }

        required_columns = {
            "image_name": "TEXT",
            "prediction": "TEXT",
            "ground_truth": "TEXT",
            "error": "TEXT",
            "mae": "REAL",
            "dataset_split": "TEXT",
            "dataset_index": "INTEGER",
            "timestamp": "TEXT",
        }

        for column, column_type in required_columns.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {column_type}"
                )


def save_prediction(
    image_name,
    prediction,
    ground_truth,
    error,
    mae,
    dataset_split,
    dataset_index
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                image_name,
                prediction,
                ground_truth,
                error,
                mae,
                dataset_split,
                dataset_index,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_name,
                str(prediction.tolist()),
                str(ground_truth.tolist()),
                str(error.tolist()),
                float(mae),
                str(dataset_split),
                int(dataset_index),
                datetime.now().isoformat()
            )
        )

def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()


initialize_database()


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

@st.cache_resource
def load_dataset(split):

    split_dir = f"{DATA_DIR}/{split}"

    top_images = np.load(
        f"{split_dir}/image.npy",
        mmap_mode="r"
    )

    wrist_images = np.load(
        f"{split_dir}/image_wrist.npy",
        mmap_mode="r"
    )

    states = np.load(
        f"{split_dir}/states.npy",
        mmap_mode="r"
    )

    actions = np.load(
        f"{split_dir}/actions.npy",
        mmap_mode="r"
    )

    return (
        top_images,
        wrist_images,
        states,
        actions
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🤖 Vision-Based Robot Arm AI")

st.write(
    "AI-powered robotic manipulation using computer vision."
)

st.divider()


# ---------------------------------------------------------
# Input method
# ---------------------------------------------------------

st.header("1. Select Input Method")

input_mode = st.radio(
    "Choose how to provide the robot inputs:",
    [
        "Dataset Sample",
        "Upload Images"
    ],
    horizontal=True
)


top_image = None
wrist_image = None
state = None
ground_truth = None

image_name = "uploaded_image"

dataset_split = None
dataset_index = None


# =========================================================
# DATASET MODE
# =========================================================

if input_mode == "Dataset Sample":

    st.subheader("Dataset Sample")

    dataset_split = st.selectbox(
        "Dataset split",
        [
            "train",
            "val",
            "test"
        ],
        format_func=lambda x: x.capitalize()
    )

    try:

        (
            top_images,
            wrist_images,
            states,
            actions
        ) = load_dataset(dataset_split)

        sample_number = st.number_input(
            "Dataset sample",
            min_value=0,
            max_value=len(states) - 1,
            value=0,
            step=1
        )

        dataset_index = int(sample_number)

        top_image = top_images[dataset_index]
        wrist_image = wrist_images[dataset_index]
        state = states[dataset_index]

        # Ground truth from actions.npy
        ground_truth = actions[dataset_index]

        image_name = (
            f"{dataset_split}_sample_{dataset_index}"
        )

        st.success(
            f"Loaded {dataset_split} sample "
            f"{dataset_index}"
        )

    except Exception as e:

        st.error(
            f"Could not load {dataset_split} dataset."
        )

        st.exception(e)


# =========================================================
# UPLOAD MODE
# =========================================================

else:

    st.subheader("Upload Camera Images")

    col1, col2 = st.columns(2)

    with col1:

        top_file = st.file_uploader(
            "Top-camera image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            key="top_upload"
        )

        if top_file:

            top_image = np.array(
                Image.open(top_file)
                .convert("RGB")
            )

            st.image(
                top_image,
                caption="Top Camera",
                use_container_width=True
            )

    with col2:

        wrist_file = st.file_uploader(
            "Wrist-camera image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            key="wrist_upload"
        )

        if wrist_file:

            wrist_image = np.array(
                Image.open(wrist_file)
                .convert("RGB")
            )

            st.image(
                wrist_image,
                caption="Wrist Camera",
                use_container_width=True
            )

    image_name = (
        top_file.name
        if top_file
        else "uploaded_image"
    )


# =========================================================
# SHOW DATASET IMAGES
# =========================================================

if (
    input_mode == "Dataset Sample"
    and top_image is not None
):

    st.subheader("Camera Images")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            top_image,
            caption="Top Camera",
            use_container_width=True
        )

    with col2:

        st.image(
            wrist_image,
            caption="Wrist Camera",
            use_container_width=True
        )


st.divider()


# =========================================================
# ROBOT STATE
# =========================================================

st.header("2. Robot State")


state_names = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "ee_pose_1",
    "ee_pose_2",
    "ee_pose_3",
    "ee_pose_4",
    "ee_pose_5",
    "ee_pose_6",
    "ee_pose_7",
    "gripper_position"
]


if (
    input_mode == "Dataset Sample"
    and state is not None
):

    st.write(
        "State automatically loaded from the dataset:"
    )

    state_cols = st.columns(5)

    for i, name in enumerate(state_names):

        with state_cols[i % 5]:

            st.metric(
                name,
                f"{state[i]:.6f}"
            )


elif input_mode == "Upload Images":

    st.write(
        "Enter the 15 robot state values:"
    )

    state_values = []

    for i in range(0, 15, 5):

        cols = st.columns(5)

        for j, col in enumerate(cols):

            index = i + j

            with col:

                value = st.number_input(
                    state_names[index],
                    value=0.0,
                    format="%.6f",
                    key=f"manual_state_{index}"
                )

                state_values.append(value)

    state = np.array(
        state_values,
        dtype=np.float32
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================
st.header("3. Robot Action Prediction")


if st.button(
    "Generate Prediction",
    type="primary",
    use_container_width=True
):

    if top_image is None:

        st.warning(
            "Please provide a top-camera image."
        )

    elif wrist_image is None:

        st.warning(
            "Please provide a wrist-camera image."
        )

    elif state is None:

        st.warning(
            "Please provide the robot state."
        )

    elif len(state) != 15:

        st.error(
            "Robot state must contain exactly 15 values."
        )

    else:

        with st.spinner(
            "Running robot model..."
        ):

            prediction = predict_robot_action(
                top_image,
                wrist_image,
                state
            )

            ground_truth = actions[dataset_index]

            error = np.abs(
                prediction - ground_truth
            )

            mae = float(np.mean(error))

            save_prediction(
                image_name=image_name,
                prediction=prediction,
                ground_truth=ground_truth,
                error=error,
                mae=mae,
                dataset_split=dataset_split,
                dataset_index=dataset_index
            )

        st.session_state[
            "latest_prediction"
        ] = prediction

        st.session_state[
            "latest_ground_truth"
        ] = ground_truth

        st.session_state[
            "latest_split"
        ] = dataset_split

        st.session_state[
            "latest_index"
        ] = dataset_index

        st.success(
            "Prediction generated successfully!"
        )


# =========================================================
# PREDICTION RESULTS
# =========================================================

if "latest_prediction" in st.session_state:

    prediction = (
        st.session_state["latest_prediction"]
    )

    ground_truth = (
        st.session_state["latest_ground_truth"]
    )

    st.subheader(
        "Predicted Actuator Commands"
    )

    action_cols = st.columns(7)

    for i, col in enumerate(action_cols):

        with col:

            st.metric(
                f"Action {i + 1}",
                f"{prediction[i]:.5f}"
            )


    # -----------------------------------------------------
    # Ground truth comparison
    # -----------------------------------------------------

    if ground_truth is not None:

        st.subheader(
            "Predicted vs. Ground Truth"
        )

        error = np.abs(
            prediction - ground_truth
        )

        mae = np.mean(error)

        comparison_cols = st.columns(4)

        with comparison_cols[0]:

            st.metric(
                "Mean Absolute Error",
                f"{mae:.6f}"
            )

        with comparison_cols[1]:

            st.metric(
                "Max Error",
                f"{np.max(error):.6f}"
            )

        with comparison_cols[2]:

            st.metric(
                "Dataset",
                st.session_state["latest_split"]
            )

        with comparison_cols[3]:

            st.metric(
                "Sample",
                st.session_state["latest_index"]
            )


        st.subheader("Prediction vs Ground Truth")
        
        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown("### Predicted")
            for i, value in enumerate(prediction):
                st.write(f"**Action {i + 1}:** {float(value):.4f}")

        with col2:
            st.markdown("### Ground Truth")
            for i, value in enumerate(ground_truth):
                st.write(f"{float(value):.4f}")

        with col3:
            st.markdown("### Absolute Error")
            for i, value in enumerate(error):
                st.write(f"{float(value):.4f}")

        st.metric("Mean Absolute Error", f"{mae:.4f}")


st.divider()


# =========================================================
# ROBOT SIMULATION
# =========================================================

st.header("4. Robot Simulation")

st.info(
    "MuJoCo robot simulation will be connected here."
)


st.divider()


# =========================================================
# HISTORY
# =========================================================

st.header("5. Prediction History")

history = get_history()

if history:

    for record in history:

        (
            prediction_id,
            timestamp,
            image_name,
            actions_json,
            ground_truth_json,
            split,
            sample_index
        ) = record

        with st.expander(
            f"Prediction #{prediction_id} - "
            f"{timestamp}"
        ):

            st.write(
                "Input:",
                image_name
            )

            if split:

                st.write(
                    f"Dataset: {split}, "
                    f"Sample: {sample_index}"
                )

            st.write(
                "Prediction:"
            )

            st.json(
                json.loads(actions_json)
            )

            if ground_truth_json:

                st.write(
                    "User Action:"
                )

                st.json(
                    json.loads(ground_truth_json)
                )

else:

    st.write(
        "No predictions have been generated yet."
    )


st.caption(
    "Deep Learning Project | "
    "VGG16 Encoder + Custom Decoder"
)