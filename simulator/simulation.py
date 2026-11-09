import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data
import torch
import torchvision
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

from model.lift_bc_rollout import RobotModel, image_tensor


def joint_ids(robot):
    infos = {}
    for j in range(pb.getNumJoints(robot)):
        info = pb.getJointInfo(robot, j)
        infos[info[1].decode()] = j

    arm = [infos[f"panda_joint{i}"] for i in range(1, 8)]
    fingers = [infos[k] for k in ("panda_finger_joint1", "panda_finger_joint2")]
    ee = infos.get("panda_hand", infos.get("panda_joint7"))

    return arm, fingers, ee


def set_pose(robot, arm_ids, finger_ids, q, gripper):
    for j, value in zip(arm_ids, q):
        pb.resetJointState(robot, j, float(value))

    opening = float(np.clip(gripper, 0, 1) * 0.04)

    for j in finger_ids:
        pb.resetJointState(robot, j, opening)


def ee_pose(robot, ee_link):
    ls = pb.getLinkState(
        robot,
        ee_link,
        computeForwardKinematics=True
    )

    return np.asarray(ls[4]), np.asarray(ls[5])


def move_blue(
    robot,
    arm_ids,
    finger_ids,
    ee_link,
    q,
    grip,
    action,
    translation_scale,
    rotation_scale
):
    """Approximate delta-EE adapter."""

    set_pose(robot, arm_ids, finger_ids, q, grip)

    pos, quat = ee_pose(robot, ee_link)

    target_pos = pos + action[:3] * translation_scale

    old_rotation = Rotation.from_quat(quat)

    target_rotation = (
        Rotation.from_rotvec(action[3:6] * rotation_scale)
        * old_rotation
    )

    target_quat = target_rotation.as_quat()

    lower = [pb.getJointInfo(robot, j)[8] for j in arm_ids]
    upper = [pb.getJointInfo(robot, j)[9] for j in arm_ids]

    ranges = [
        hi - lo
        for lo, hi in zip(lower, upper)
    ]

    ik = pb.calculateInverseKinematics(
        robot,
        ee_link,
        target_pos.tolist(),
        target_quat.tolist(),
        lowerLimits=lower,
        upperLimits=upper,
        jointRanges=ranges,
        restPoses=list(map(float, q)),
        maxNumIterations=120,
        residualThreshold=1e-5,
    )

    next_q = np.asarray(ik[:7], dtype=np.float64)

    next_grip = float(
        np.clip(
            grip + action[6] * 0.05,
            0,
            1
        )
    )

    set_pose(
        robot,
        arm_ids,
        finger_ids,
        next_q,
        next_grip
    )

    return next_q, next_grip


def load_policy(checkpoint, state_dim, device):

    ckpt = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    if (
        ckpt["state_dim"] != state_dim
        or ckpt["action_dim"] != 7
    ):
        raise ValueError(
            f"Checkpoint dims "
            f"{ckpt['state_dim']}/{ckpt['action_dim']} "
            f"vs dataset {state_dim}/7"
        )

    backbone = torchvision.models.vgg16(
       weights=None
    )

    policy = RobotModel(
        backbone,
        backbone,
        state_dim,
        7
    )

    policy.load_state_dict(
        ckpt["model_state_dict"]
    )
    policy = policy.to(device)
    policy.eval()

    return policy


def add_scene_props():

    slab = pb.createVisualShape(
        pb.GEOM_BOX,
        halfExtents=[0.36, 0.30, 0.025],
        rgbaColor=[0.65, 0.53, 0.40, 1]
    )

    cube = pb.createVisualShape(
        pb.GEOM_BOX,
        halfExtents=[0.025] * 3,
        rgbaColor=[0.9, 0.45, 0.18, 1]
    )

    for x in (-0.60, 0.60):

        pb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=slab,
            basePosition=[x, 0.48, -0.035]
        )

        pb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cube,
            basePosition=[x, 0.48, 0.015]
        )


def plot_errors(frames, path, mode):

    x = [
        f["step"]
        for f in frames
    ]

    ref = [
        f["reference_replay_ee_error_m"] * 100
        for f in frames
    ]

    bc = [
        f["bc_ee_error_m"] * 100
        for f in frames
    ]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        x,
        ref,
        label="Recorded actions through approximate adapter",
        lw=1.8
    )

    ax.plot(
        x,
        bc,
        label=f"BC model ({mode})",
        lw=1.8
    )

    ax.set(
        xlabel="Demonstration timestep",
        ylabel="End-effector position difference (cm)",
        title="Controller replay validation vs BC visualization"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)


def run_simulation(
    demo_id,
    mode="one_step",
    data_dir="demo_data/simulation",
    checkpoint="model/robot_finetuned.pth",
    output_dir="simulation_outputs",
    fps=20,
    device=None,
    translation_scale=0.15,
    rotation_scale=0.75,
    corrected_preprocessing=False,
    no_illustrative_props=False,
):

    import pybullet as pb
    import pybullet_data
    data = Path(data_dir)

    checkpoint = Path(checkpoint)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ids = np.load(
        data / "demo_ids.npy",
        mmap_mode="r"
    )

    rows = np.flatnonzero(
        ids == demo_id
    )

    if len(rows) < 2:
        raise ValueError(
            f"Demo {demo_id} has fewer than 2 timesteps."
        )

    q = np.load(
        data / "state_q.npy",
        mmap_mode="r"
    )

    grip = np.load(
        data / "state_gripper_position.npy",
        mmap_mode="r"
    )

    states = np.load(
        data / "states.npy",
        mmap_mode="r"
    )

    top = np.load(
        data / "image.npy",
        mmap_mode="r"
    )

    wrist = np.load(
        data / "image_wrist.npy",
        mmap_mode="r"
    )

    actual_actions = np.load(
        data / "actions.npy",
        mmap_mode="r"
    )

    if (
        q.shape[1] != 7
        or grip.shape[1] != 1
        or actual_actions.shape[1] != 7
    ):
        raise ValueError(
            "Dataset q/gripper/actions dimensions are unexpected."
        )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    policy = load_policy(
        checkpoint,
        states.shape[1],
        device
    )

    client = pb.connect(pb.DIRECT)

    if client < 0:
        raise RuntimeError(
            "Could not connect to PyBullet DIRECT renderer."
        )

    try:

        pb.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        pb.setGravity(
            0,
            0,
            -9.81
        )

        pb.loadURDF(
            "plane.urdf",
            [0, 0, -0.02]
        )

        if not no_illustrative_props:
            add_scene_props()

        human = pb.loadURDF(
            "franka_panda/panda.urdf",
            [-0.60, 0, 0],
            useFixedBase=True
        )

        predicted = pb.loadURDF(
            "franka_panda/panda.urdf",
            [0.60, 0, 0],
            useFixedBase=True
        )

        h_arm, h_fingers, h_ee = joint_ids(human)

        b_arm, b_fingers, b_ee = joint_ids(predicted)

        for j in range(pb.getNumJoints(human)):
            pb.changeVisualShape(
                human,
                j,
                rgbaColor=[0.15, 0.75, 0.30, 1]
            )

        for j in range(pb.getNumJoints(predicted)):
            pb.changeVisualShape(
                predicted,
                j,
                rgbaColor=[0.15, 0.35, 0.95, 1]
            )

        set_pose(
            human,
            h_arm,
            h_fingers,
            q[rows[0]],
            grip[rows[0], 0]
        )

        set_pose(
            predicted,
            b_arm,
            b_fingers,
            q[rows[0]],
            grip[rows[0], 0]
        )

        blue_q = np.array(
            q[rows[0]]
        )

        blue_grip = float(
            grip[rows[0], 0]
        )

        view = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.35],
            distance=3.6,
            yaw=45,
            pitch=-25,
            roll=0,
            upAxisIndex=2,
        )

        projection = pb.computeProjectionMatrixFOV(
            55,
            960 / 720,
            0.08,
            10
        )

        output = (
            output_dir
            / f"lift_panda_demo_{demo_id}_{mode}.mp4"
        )

        measurements = []

        with imageio.get_writer(
            output,
            fps=fps,
            macro_block_size=16
        ) as writer:

            for t, i in enumerate(rows[:-1]):

                set_pose(
                    human,
                    h_arm,
                    h_fingers,
                    q[rows[t + 1]],
                    grip[rows[t + 1], 0]
                )

                with torch.inference_mode():

                    a_top = image_tensor(
                        top[i],
                        device,
                        not corrected_preprocessing
                    )

                    a_wrist = image_tensor(
                        wrist[i],
                        device,
                        not corrected_preprocessing
                    )

                    a_state = torch.as_tensor(
                        np.asarray(states[i]),
                        device=device,
                        dtype=torch.float32
                    )[None]

                    act = (
                        policy(
                            a_top,
                            a_wrist,
                            a_state
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                act = np.clip(
                    act,
                    -1,
                    1
                )

                recorded_q = np.asarray(
                    q[i]
                )

                recorded_grip = float(
                    grip[i, 0]
                )

                move_blue(
                    predicted,
                    b_arm,
                    b_fingers,
                    b_ee,
                    recorded_q,
                    recorded_grip,
                    np.clip(
                        np.asarray(
                            actual_actions[i]
                        ),
                        -1,
                        1
                    ),
                    translation_scale,
                    rotation_scale
                )

                replay_pos, _ = ee_pose(
                    predicted,
                    b_ee
                )

                if mode == "one_step":

                    source_q = recorded_q
                    source_grip = recorded_grip

                else:

                    source_q = blue_q
                    source_grip = blue_grip

                blue_q, blue_grip = move_blue(
                    predicted,
                    b_arm,
                    b_fingers,
                    b_ee,
                    source_q,
                    source_grip,
                    act,
                    translation_scale,
                    rotation_scale
                )

                human_pos, _ = ee_pose(
                    human,
                    h_ee
                )

                blue_pos, _ = ee_pose(
                    predicted,
                    b_ee
                )

                human_local = (
                    human_pos
                    - np.array([-0.6, 0, 0])
                )

                blue_local = (
                    blue_pos
                    - np.array([0.6, 0, 0])
                )

                ee_error = float(
                    np.linalg.norm(
                        human_local - blue_local
                    )
                )

                replay_local = (
                    replay_pos
                    - np.array([0.6, 0, 0])
                )

                ref_error = float(
                    np.linalg.norm(
                        human_local - replay_local
                    )
                )

                measurements.append({
                    "step": t,
                    "ee_position_error_m": ee_error,
                    "bc_ee_error_m": ee_error,
                    "reference_replay_ee_error_m": ref_error,
                    "prediction": act.tolist(),
                    "reference_action": np.asarray(
                        actual_actions[i]
                    ).tolist()
                })

                _, _, rgba, _, _ = pb.getCameraImage(
                    960,
                    720,
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=pb.ER_TINY_RENDERER
                )

                scene = (
                    Image.fromarray(
                        np.asarray(
                            rgba,
                            dtype=np.uint8
                        ).reshape(
                            720,
                            960,
                            4
                        ),
                        "RGBA"
                    )
                    .convert("RGB")
                )

                frame = Image.new(
                    "RGB",
                    (1280, 720),
                    (247, 247, 247)
                )

                frame.paste(
                    scene,
                    (0, 0)
                )

                frame.paste(
                    Image.fromarray(
                        np.asarray(
                            top[i]
                        )
                    ).resize(
                        (288, 288)
                    ),
                    (976, 56)
                )

                frame.paste(
                    Image.fromarray(
                        np.asarray(
                            wrist[i]
                        )
                    ).resize(
                        (288, 288)
                    ),
                    (976, 386)
                )

                draw = ImageDraw.Draw(frame)

                draw.rectangle(
                    (0, 0, 1280, 50),
                    fill=(245, 245, 245)
                )

                draw.text(
                    (18, 10),
                    f"GREEN: recorded q    BLUE: BC {mode}    Demo {demo_id} / step {t}",
                    fill=(0, 0, 0)
                )

                draw.text(
                    (18, 28),
                    f"BC diff {ee_error * 100:.1f} cm   "
                    f"Recorded-action replay diff {ref_error * 100:.1f} cm",
                    fill=(0, 0, 0)
                )

                draw.text(
                    (980, 60),
                    "Recorded overhead",
                    fill=(0, 0, 0),
                    stroke_width=1,
                    stroke_fill="white"
                )

                draw.text(
                    (980, 390),
                    "Recorded wrist",
                    fill=(0, 0, 0),
                    stroke_width=1,
                    stroke_fill="white"
                )

                draw.text(
                    (980, 685),
                    "Cube/table are illustrative",
                    fill=(0, 0, 0)
                )

                writer.append_data(
                    np.asarray(frame)
                )

        error_plot = (
            output_dir
            / f"lift_panda_demo_{demo_id}_{mode}_errors.png"
        )

        plot_errors(
            measurements,
            error_plot,
            mode
        )

        mean_bc_error = float(
            np.mean([
                x["bc_ee_error_m"]
                for x in measurements
            ])
        )

        mean_replay_error = float(
            np.mean([
                x["reference_replay_ee_error_m"]
                for x in measurements
            ])
        )

        report = {
            "demo": int(demo_id),
            "mode": mode,
            "approximate_controller": True,
            "props_are_illustrative": not no_illustrative_props,
            "mean_ee_position_error_m": mean_bc_error,
            "mean_recorded_action_replay_error_m": mean_replay_error,
            "frames": measurements,
        }

        report_path = (
            output_dir
            / f"lift_panda_demo_{demo_id}_{mode}.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2
            )
        )

        return {
            "video": output,
            "error_plot": error_plot,
            "report": report_path,
            "mean_error_cm": mean_bc_error*100,
            "mean_bc_error": mean_bc_error
        }

    finally:
        pb.disconnect()


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="demo_data/simulation"
    )

    p.add_argument(
        "--checkpoint",
        default="model/robot_finetuned.pth"
    )

    p.add_argument(
        "--demo",
        type=int,
        required=True
    )

    p.add_argument(
        "--mode",
        choices=("one_step", "cumulative"),
        default="one_step"
    )

    p.add_argument(
        "--out",
        default=None
    )

    p.add_argument(
        "--fps",
        type=int,
        default=20
    )

    args = p.parse_args()

    result = run_simulation(
        demo_id=args.demo,
        mode=args.mode,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        fps=args.fps,
    )

    print("Saved:", result["video"])
    print(
        "Mean BC error (cm):",
        round(result["mean_bc_error"], 2)
    )


if __name__ == "__main__":
    main()