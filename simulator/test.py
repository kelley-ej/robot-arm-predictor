import robosuite as suite
import mujoco
from robosuite.utils.binding_utils import MjModel

original_function = MjModel.get_joint_qpos_addr

def debug_joint(self, name):
    joint_id = self.joint_name2id(name)

    # Check the exact value used by robosuite
    jnt_type = self.jnt_type[joint_id]

    print("\n--- JOINT DEBUG ---")
    print("Joint:", name)
    print("Raw type:", self.jnt_type[joint_id])
    print("Wrapper type:", repr(jnt_type))
    print("Wrapper type class:", type(jnt_type))
    print("Hinge value:", mujoco.mjtJoint.mjJNT_HINGE)
    print("Slide value:", mujoco.mjtJoint.mjJNT_SLIDE)
    print("Hinge comparison:", jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    print("Slide comparison:", jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)

    return original_function(self, name)

MjModel.get_joint_qpos_addr = debug_joint

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False
)

print("Simulation initialized successfully!")
env.close()