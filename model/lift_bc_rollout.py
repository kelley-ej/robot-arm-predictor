"""Run the custom two-camera VGG behavior-cloning policy in robomimic Lift.

Usage: python lift_bc_rollout.py --dataset image.hdf5 --checkpoint robot_model.pth \
  --state-keys robot0_joint_pos robot0_joint_vel robot0_gripper_qpos \
  --out-dir lift_rollouts --episodes 5

IMPORTANT: --state-keys MUST match the exact fields/order used for states.npy.
The script checks the concatenated dimension but cannot infer ordering.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Must precede simulator imports.

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


class RobotModel(torch.nn.Module):
    """Architecture copied from CNNs(1).ipynb, including its large flatten layer."""
    def __init__(self, vgg_model1, vgg_model2, state_dim, action_dim):
        super().__init__()
        import copy
        self.features1 = copy.deepcopy(vgg_model1.features)
        self.avgpool1 = copy.deepcopy(vgg_model1.avgpool)
        self.features2 = copy.deepcopy(vgg_model2.features)
        self.avgpool2 = copy.deepcopy(vgg_model2.avgpool)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(2 * 25088 + state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

    def forward(self, img1, img2, state):
        a = self.avgpool1(self.features1(img1)).flatten(1)
        b = self.avgpool2(self.features2(img2)).flatten(1)
        return self.regressor(torch.cat((a, b, state), dim=1))


def image_tensor(image, device, legacy_double_divide):
    """Match notebook: 0..255 -> 0..1, optionally divide by 255 AGAIN."""
    from torchvision.transforms import functional as TF
    x = torch.as_tensor(np.ascontiguousarray(image), device=device)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image HWC, got {tuple(x.shape)}")
    x = x.permute(2, 0, 1).float()
    # Image observations from robomimic may be float32 [0,1], unlike stored uint8.
    if image.dtype == np.uint8 or float(x.max()) > 1.5:
        x = x / 255.0
    if legacy_double_divide:
        x = x / 255.0
    x = TF.resize(x, [224, 224], antialias=True)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x.unsqueeze(0)


def state_tensor(obs, keys, device, expected_dim):
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(f"State keys not in simulator observation: {missing}; available={list(obs)}")
    state = np.concatenate([np.asarray(obs[key]).reshape(-1) for key in keys])
    if len(state) != expected_dim:
        raise ValueError(
            f"State length {len(state)} differs from checkpoint state_dim {expected_dim}. "
            f"Check --state-keys AND their order; keys={keys}"
        )
    return torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def save_plot(actions, path):
    names = ["x", "y", "z", "rot x", "rot y", "rot z", "gripper"]
    fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(np.arange(len(actions)), actions[:, j], linewidth=1.1)
        ax.set_ylabel(names[j])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Simulation control step")
    fig.suptitle("Predicted 7D Lift actions (closed-loop rollout)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Processed image HDF5 (with env metadata)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-keys", nargs="+", required=True, help="Exact obs keys/order used for states.npy")
    parser.add_argument("--top-key", default="agentview_image")
    parser.add_argument("--wrist-key", default="robot0_eye_in_hand_image")
    parser.add_argument("--out-dir", default="lift_rollouts")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corrected-preprocessing", action="store_true", help="ONLY for models retrained without accidental second /255")
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("episodes and horizon must be positive")

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt["action_dim"] != 7:
        raise ValueError(f"Expected seven actions; checkpoint says {ckpt['action_dim']}")
    # No pretrained weights download: state_dict supplies ALL trained weights.
    vgg = torchvision.models.vgg16(weights=None)
    model = RobotModel(vgg, vgg, ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    del vgg, ckpt
    model = model.to(device).eval()

    meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    if meta["env_name"].lower() != "lift":
        raise ValueError(f"Dataset describes {meta['env_name']!r}, not Lift")
    print("Lift environment metadata:", json.dumps(meta, indent=2, default=str))
    env = EnvUtils.create_env_from_metadata(
        env_meta=meta, render=False, render_offscreen=True, use_image_obs=True
    )
    if env.action_dimension != 7:
        raise ValueError(f"Simulator action dimension {env.action_dimension}, expected 7")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            print("Observation fields:", {k: np.shape(v) for k, v in obs.items()})
            for key in (args.top_key, args.wrist_key):
                if key not in obs:
                    raise KeyError(f"Missing camera {key!r}; present={list(obs)}")
            steps = []
            total_reward = 0.0
            success = False
            video_file = output / f"episode_{ep:03d}.mp4"
            # Use camera rendering, not direct replay of HDF5 frames.
            with imageio.get_writer(video_file, fps=args.fps) as writer:
                for t in range(args.horizon):
                    with torch.inference_mode():
                        top = image_tensor(obs[args.top_key], device, not args.corrected_preprocessing)
                        wrist = image_tensor(obs[args.wrist_key], device, not args.corrected_preprocessing)
                        state = state_tensor(obs, args.state_keys, device, model.regressor[0].in_features - 50176)
                        action = model(top, wrist, state).squeeze(0).cpu().numpy()
                    # OSC_POSE inputs are conventionally normalized to [-1, 1].
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    obs, reward, done, info = env.step(action)
                    total_reward += float(reward)
                    steps.append(action)
                    frame = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
                    writer.append_data(np.asarray(frame, dtype=np.uint8))
                    success = bool(env.is_success().get("task", False))
                    if success or done:
                        break
            actions = np.stack(steps)
            np.save(output / f"episode_{ep:03d}_actions.npy", actions)
            save_plot(actions, output / f"episode_{ep:03d}_actions.png")
            result = {"episode": ep, "success": success, "steps": len(actions), "return": total_reward}
            records.append(result)
            print(result)
    finally:
        env.close()

    summary = {"episodes": len(records), "success_rate": sum(r["success"] for r in records) / len(records), "results": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
