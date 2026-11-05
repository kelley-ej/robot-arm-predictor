import streamlit as st
import sqlite3
import json
from datetime import datetime
import numpy as np
from PIL import Image

from model.predict import predict_robot_action


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Robot Arm AI",
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


# =========================================================
# SETTINGS
# =========================================================

DB_PATH = "predictions.db"
DATA_DIR = "demo_data"


# =========================================================
# DATABASE
# =========================================================

def initialize_database():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_name TEXT,
                actions TEXT
            )
        """)


def save_prediction(image_name, actions):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            INSERT INTO predictions
            (timestamp, image_name, actions)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                image_name,
                json.dumps(actions.tolist())
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


# =========================================================
# LOAD DEMO DATASET
# =========================================================

@st.cache_resource
def load_demo_dataset():

    top_images = np.load(
        f"{DATA_DIR}/image.npy",
        mmap_mode="r"
    )

    wrist_images = np.load(
        f"{DATA_DIR}/image_wrist.npy",
        mmap_mode="r"
    )

    states = np.load(
        f"{DATA_DIR}/states.npy",
        mmap_mode="r"
    )

    return top_images, wrist_images, states


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Vision-Based Robot Arm AI")

st.write(
    "AI-powered robotic manipulation using computer vision."
)

st.divider()


# =========================================================
# INPUT MODE
# =========================================================

st.header("1. Select Input Method")

input_mode = st.radio(
    "Choose how to provide the robot inputs:",
    [
        "Dataset Sample",
        "Upload Images"
    ],
    horizontal=True
)


# =========================================================
# VARIABLES
# =========================================================

top_image = None
wrist_image = None
state = None
image_name = "uploaded_image"


# =========================================================
# DATASET MODE
# =========================================================

if input_mode == "Dataset Sample":

    st.subheader("Dataset Sample")

    try:

        top_images, wrist_images, states = (
            load_demo_dataset()
        )

        sample_number = st.number_input(
            "Dataset sample",
            min_value=0,
            max_value=len(states) - 1,
            value=0,
            step=1
        )

        top_image = top_images[sample_number]
        wrist_image = wrist_images[sample_number]
        state = states[sample_number]

        image_name = (
            f"dataset_sample_{sample_number}"
        )

        st.success(
            f"Loaded dataset sample {sample_number}"
        )

    except Exception as e:

        st.error(
            "Could not load demo dataset."
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
                Image.open(
                    top_file
                ).convert("RGB")
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
                Image.open(
                    wrist_file
                ).convert("RGB")
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
# DISPLAY DATASET IMAGES
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


# =========================================================
# ROBOT STATE
# =========================================================

st.divider()

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


# ---------------------------------------------------------
# Dataset state
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Manual state
# ---------------------------------------------------------

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


# =========================================================
# PREDICTION
# =========================================================

st.divider()

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

            prediction = (
                predict_robot_action(
                    top_image,
                    wrist_image,
                    state
                )
            )

        save_prediction(
            image_name,
            prediction
        )

        st.session_state[
            "latest_prediction"
        ] = prediction

        st.success(
            "Prediction generated successfully!"
        )


# =========================================================
# DISPLAY PREDICTION
# =========================================================

if "latest_prediction" in st.session_state:

    st.subheader(
        "Predicted Actuator Commands"
    )

    prediction = st.session_state[
        "latest_prediction"
    ]

    action_cols = st.columns(7)

    for i, col in enumerate(action_cols):

        with col:

            st.metric(
                f"Action {i + 1}",
                f"{prediction[i]:.5f}"
            )

    st.subheader("Raw Prediction")

    st.json(
        prediction.tolist()
    )


# =========================================================
# SIMULATION
# =========================================================

st.divider()

st.header("4. Robot Simulation")

st.info(
    "MuJoCo robot simulation will be connected here."
)


# =========================================================
# HISTORY
# =========================================================

st.divider()

st.header("5. Prediction History")

history = get_history()


if history:

    for record in history:

        prediction_id, timestamp, image_name, actions = record

        with st.expander(
            f"Prediction #{prediction_id} - {timestamp}"
        ):

            st.write(
                "Input:",
                image_name
            )

            st.json(
                json.loads(actions)
            )

else:

    st.write(
        "No predictions have been generated yet."
    )


st.caption(
    "Deep Learning Project | "
    "VGG16 Encoder + Custom Decoder"
)