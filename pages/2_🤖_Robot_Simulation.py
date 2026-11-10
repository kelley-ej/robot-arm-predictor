import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "demo_data" / "simulation_videos"

st.set_page_config(
    page_title="Robot Simulation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Robot Arm Simulation")

st.write(
    "View the trained robot policy running through one of the "
    "recorded robot demonstrations."
)

st.info(
    "These simulations were generated using the recorded camera images "
    "and robot states from the selected demonstration. The blue robot "
    "shows an approximate visualization of the model's predicted actions."
)

# Find available videos
videos = sorted(VIDEO_DIR.glob("lift_panda_demo_*_one_step.mp4"))

if not videos:
    st.error("No simulation videos were found.")
    st.stop()

# Convert filenames into demo numbers
demo_numbers = []

for video in videos:
    try:
        demo_number = int(video.stem.split("_")[3])
        demo_numbers.append(demo_number)
    except (IndexError, ValueError):
        continue

if not demo_numbers:
    st.error("No valid simulation videos were found.")
    st.stop()

selected_demo = st.selectbox(
    "Select Demonstration",
    demo_numbers,
    format_func=lambda x: f"Demo {x + 1}"
)

selected_video = VIDEO_DIR / f"lift_panda_demo_{selected_demo}_one_step.mp4"

st.subheader(f"Demo {selected_demo + 1}")

if selected_video.exists():
    st.video(str(selected_video))
else:
    st.error("Simulation video could not be found.")

st.caption(
    "Simulation mode: One Step — the predicted action is compared "
    "against the recorded next robot state at each timestep."
)