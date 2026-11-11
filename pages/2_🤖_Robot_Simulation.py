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
    "View the trained robot policy running on recorded robot demonstrations."
)

st.info(
    "The green robot shows the recorded demonstration, while the blue robot "
    "shows the predicted behavior from the trained policy."
)

videos = sorted(
    VIDEO_DIR.glob("lift_panda_demo_*_overlay_4x.mp4")
)

if not videos:
    st.error("No simulation videos were found.")
    st.stop()

demo_numbers = []

for video in videos:
    try:
        demo_number = int(video.stem.split("_")[3])
        demo_numbers.append(demo_number)
    except (IndexError, ValueError):
        continue

demo_numbers = sorted(demo_numbers)

selected_demo = st.selectbox(
    "Select Demonstration",
    demo_numbers,
    format_func=lambda x: f"Demo {x + 1}"
)

selected_video = (
    VIDEO_DIR /
    f"lift_panda_demo_{selected_demo}_overlay_4x.mp4"
)

st.subheader(f"Demo {selected_demo + 1}")

if selected_video.exists():
    st.video(str(selected_video))
else:
    st.error("Simulation video could not be found.")

st.caption(
    "Simulation mode: Overlay — the predicted robot action is visualized "
    "against the recorded robot demonstration. Videos are shown at 4× speed."
)