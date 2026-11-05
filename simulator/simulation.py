import numpy as np
import robosuite as suite


def run_simulation():

    # Create a simulated robot environment
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=480,
        camera_widths=640,
        control_freq=20,
        horizon=100
    )

    try:
        # Reset simulation
        obs = env.reset()

        # Get the expected action dimensions
        low, high = env.action_spec

        # Temporary neutral action
        action = np.zeros_like(low)

        # Run simulation for 20 steps
        for _ in range(20):
            obs, reward, done, info = env.step(action)

            if done:
                break

        # Get the camera image
        frame = obs["agentview_image"]

        # Ensure image is suitable for display
        frame = np.asarray(frame, dtype=np.uint8)

        return frame, action.tolist()

    finally:
        env.close()