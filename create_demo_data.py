import os
import numpy as np

SOURCE_DIR = "/work/cssema416/202710/13/src/bc_data"
OUTPUT_DIR = "demo_data"

NUM_SAMPLES = 100
SEED = 42


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Opening dataset with memory mapping...")

top_images = np.load(
    f"{SOURCE_DIR}/image.npy",
    mmap_mode="r"
)

wrist_images = np.load(
    f"{SOURCE_DIR}/image_wrist.npy",
    mmap_mode="r"
)

states = np.load(
    f"{SOURCE_DIR}/states.npy",
    mmap_mode="r"
)

actions = np.load(
    f"{SOURCE_DIR}/actions.npy",
    mmap_mode="r"
)


print(f"Full dataset contains {len(states)} samples.")


# Pick reproducible random samples
rng = np.random.default_rng(SEED)

indices = rng.choice(
    len(states),
    size=NUM_SAMPLES,
    replace=False
)

indices.sort()


print("\nSelected sample indices:")
print(indices)


# Copy only the selected samples
print("\nCopying top-camera images...")

np.save(
    f"{OUTPUT_DIR}/image.npy",
    np.array(top_images[indices])
)


print("Copying wrist-camera images...")

np.save(
    f"{OUTPUT_DIR}/image_wrist.npy",
    np.array(wrist_images[indices])
)


print("Copying states...")

np.save(
    f"{OUTPUT_DIR}/states.npy",
    np.array(states[indices])
)


print("Copying actions...")

np.save(
    f"{OUTPUT_DIR}/actions.npy",
    np.array(actions[indices])
)


# Save the original dataset indices too
np.save(
    f"{OUTPUT_DIR}/original_indices.npy",
    indices
)


print("\nDemo dataset created successfully!")

print(f"Samples: {NUM_SAMPLES}")
print(f"Output directory: {OUTPUT_DIR}")